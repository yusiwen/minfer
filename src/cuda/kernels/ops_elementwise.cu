// src/cuda/kernels/ops_elementwise.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


// ─── Quantize f32 → Q8_0 (1 thread per 32-element block) ─────
// Matches CPU scalar path: half delta + 32 signed int8 values

__global__ void quantize_q8_0(
    const float* __restrict__ x,
    uint8_t* __restrict__ y,
    int dim, int nt
) {
    int nb = dim / 32;
    int total = nt * nb;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int t = tid / nb;
    int b = tid % nb;

    const float* src = x + t * dim + b * 32;
    uint8_t* dst = y + (t * nb + b) * Q8B;

    float am = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) am = fmaxf(am, fabsf(src[j]));
    float d = am / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    *reinterpret_cast<__half*>(dst) = __float2half(d);

    for (int j = 0; j < 32; j++) {
        int q = int(rintf(src[j] * id));
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        dst[2 + j] = uint8_t(int8_t(q));
    }
}

// ─── RMSNorm (32 threads per row, no shared memory) ──────────
// y[t][i] = x[t][i] * rsqrt(mean(x[t]²) + eps) * w[i]

__global__ void rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int d, float eps, int n
) {
    int row = blockIdx.x;
    if (row >= n) return;

    int tid = threadIdx.x;
    int d4 = d / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x + row * d);

    float ss = 0.0f;
    for (int i = tid; i < d4; i += WARP) {
        float4 v = x4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_reduce_sum(ss);

    float scale = rsqrtf(ss / (float)d + eps);

    float4* y4 = reinterpret_cast<float4*>(y + row * d);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (int i = tid; i < d4; i += WARP) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        y4[i].x = xv.x * scale * wv.x;
        y4[i].y = xv.y * scale * wv.y;
        y4[i].z = xv.z * scale * wv.z;
        y4[i].w = xv.w * scale * wv.w;
    }
}

// ─── Add bias: y[t][i] += b[i] ───────────────────────────────


// D3-5 1a: decode fused rms_norm + pad40 q8 epilogue. The rms body is
// rms_norm_f32 verbatim (bit-identical f32 y); the epilogue re-reads the row
// this block just wrote (L1-hot after __syncthreads) and runs the standalone
// per-block quantize body verbatim. Same launch geometry as rms_norm_f32
// (grid n, WARP threads).
__global__ void __launch_bounds__(128) rms_norm_quant_pad40(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    uint8_t* __restrict__ q8,
    int d, float eps, int n
) {
    int row = blockIdx.x;
    if (row >= n) return;

    int tid = threadIdx.x;
    int d4 = d / 4;

    // D3-7 2c: wide-block geometry (launch picks 32 or 128 threads). The
    // reduction is bitwise-preserved: lanes 0..31 keep the exact 32-thread
    // form's element->lane mapping, serial per-lane accumulation order and
    // warp_reduce_sum tree; the unroll only deepens load pipelining. scale
    // reaches the whole block through shared memory. The write and quantize
    // loops are per-element / per-32-block independent, so their wider
    // thread mapping cannot change any output bit.
    __shared__ float s_scale;
    float scale;
    if (tid < WARP) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row * d);

        float ss = 0.0f;
        #pragma unroll 8
        for (int i = tid; i < d4; i += WARP) {
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = warp_reduce_sum(ss);

        scale = rsqrtf(ss / (float)d + eps);
        if (tid == 0) s_scale = scale;
    }
    __syncthreads();
    scale = s_scale;

    float4* y4 = reinterpret_cast<float4*>(y + row * d);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4w = reinterpret_cast<const float4*>(x + row * d);
    for (int i = tid; i < d4; i += blockDim.x) {
        float4 wv = w4[i];
        float4 xv = x4w[i];
        y4[i].x = xv.x * scale * wv.x;
        y4[i].y = xv.y * scale * wv.y;
        y4[i].z = xv.z * scale * wv.z;
        y4[i].w = xv.w * scale * wv.w;
    }

    // epilogue: whole block arrives (all threads share `row`), then each
    // thread quantizes blocks tid, tid+blockDim.x, ... of its own row.
    __syncthreads();
    int nb = d / 32;
    const float* src = y + (size_t)row * d;
    uint8_t* dst = q8 + (size_t)row * nb * Q8PB;
    for (int b = tid; b < nb; b += blockDim.x)
        quantize_pad40_block(src + (size_t)b * 32, dst + (size_t)b * Q8PB);
}

__global__ void add_bias_f32(
    float* __restrict__ y,
    const float* __restrict__ b,
    int d
) {
    int t = blockIdx.x, i = threadIdx.x + blockIdx.y * blockDim.x;
    if (i >= d) return;
    y[t * d + i] += b[i];
}

// ─── Element-wise add: z = x + y ─────────────────────────────

__global__ void add_f32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    z[tid] = x[tid] + y[tid];
}

// ─── Element-wise multiply: z = x * y ────────────────────────

__global__ void mul_f32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    z[tid] = x[tid] * y[tid];
}

// ─── SiLU in-place: y = y / (1 + exp(-y)) ────────────────────

__global__ void silu_f32(float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float v = y[tid];
    y[tid] = v / (1.0f + expf(-v));
}

// ─── SwiGLU fused: dst = silu(gate) * up ─────────────────────

// 7e⑤: in-place split swiglu over one buffer — buf[i] = silu(buf[i]) *
// buf[off + i] (the fused FFN concat matmul output: gate rows 0..nf, up
// rows nf..2*nf; results written back into the gate rows).
__global__ void swiglu_f32_off(float* __restrict__ buf, int n, int off) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float g = buf[tid];
    buf[tid] = (g / (1.0f + expf(-g))) * buf[off + tid];
}


// D3-5 1a: decode fused swiglu + pad40 q8 epilogue. Body = swiglu_f32_off
// verbatim (guarded, no early return — every thread reaches the barrier).
// Block bx wrote output elements [bx*256, bx*256+256) = quant blocks
// bx*8 .. bx*8+7, so 8 threads per block re-read them (L1-hot) and quantize;
// across the grid this is the same thread-count as the standalone kernel
// (one thread per 32-block). REQUIRES the 256-thread launch geometry.
__global__ void swiglu_quant_pad40(
    float* __restrict__ buf,
    uint8_t* __restrict__ q8,
    int n, int off
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float g = buf[tid];
        buf[tid] = (g / (1.0f + expf(-g))) * buf[off + tid];
    }
    __syncthreads();
    int b = blockIdx.x * 8 + (int)threadIdx.x;
    if (threadIdx.x < 8 && b < (n >> 5))
        quantize_pad40_block(buf + (size_t)b * 32, q8 + (size_t)b * Q8PB);
}

__global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ dst,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float g = gate[tid];
    dst[tid] = (g / (1.0f + expf(-g))) * up[tid];
}

// ─── I32 input decode: positions/token ids arrive as f32::from_bits(v)
// bit patterns (graph convention, alloc.rs fill_input_i32) while the rope /
// store / attention kernels read raw int32. One elementwise pass
// reinterprets the bits into a scratch buffer — fully device-side, so the
// per-layer path needs no host sync (and stays CUDA-Graph-replayable).

__global__ void f32_bits_to_i32(
    const float* __restrict__ src,
    int* __restrict__ dst,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = __float_as_int(src[tid]);
}

// ─── RoPE (NEOX-style, in-place) ─────────────────────────────
// x layout: [nt][n_head][n_dims] — pairs (x[i], x[i+half])
// NEOX-style: pairs (x[i], x[i+hd/2]) for each head

__global__ void rope_f32(
    float* x,
    int n_head, int n_dims, int nt,
    float freq_base, float freq_scale,
    const int* positions
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= nt || h >= n_head) return;

    int half = n_dims / 2;
    int base = (t * n_head + h) * n_dims;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = freq_scale / powf(freq_base, (2.0f * i) / n_dims);
        float theta = positions[t] * freq;
        float cs = cosf(theta), sn = sinf(theta);
        int j = base + i;
        int j2 = j + half;
        float x0 = x[j], x1 = x[j2];
        x[j]  = x0 * cs - x1 * sn;
        x[j2] = x0 * sn + x1 * cs;
    }
}

extern "C" {

void launch_swiglu_f32_off(
    float* buf, int n, int off, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    minfer_launch_prelude("launch:swiglu_f32_off", "swiglu_f32_off");
    swiglu_f32_off<<<grid, minfer_launch_block("launch:swiglu_f32_off", block), 0, stream>>>(buf, n, off);
    minfer_launch_ok("launch:swiglu_f32_off", "swiglu_f32_off");
}


// D3-5 1a: decode fused swiglu + pad40 q8 epilogue. The 256-thread block size
// is part of the epilogue's block->quant-block mapping (8 blocks per 256
// elements) — do not change it without changing the kernel.
void launch_swiglu_quant_pad40(
    float* buf, uint8_t* q8, int n, int off, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    minfer_launch_prelude("launch:swiglu_quant_pad40", "swiglu_quant_pad40");
    swiglu_quant_pad40<<<grid, minfer_launch_block("launch:swiglu_quant_pad40", block), 0, stream>>>(buf, q8, n, off);
    minfer_launch_ok("launch:swiglu_quant_pad40", "swiglu_quant_pad40");
}

void launch_quantize_q8_0(
    const float* x, uint8_t* y, int dim, int nt, cudaStream_t stream
) {
    int nb = dim / 32;
    int total = nt * nb;
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((total + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:quantize_q8_0", "quantize_q8_0");
    quantize_q8_0<<<grid, minfer_launch_block("launch:quantize_q8_0", block), 0, stream>>>(x, y, dim, nt);
    minfer_launch_ok("launch:quantize_q8_0", "quantize_q8_0");
}

void launch_rms_norm_f32(
    const float* x, const float* w, float* y,
    int d, float eps, int n, cudaStream_t stream
) {
    dim3 block(WARP, 1, 1);
    dim3 grid(n, 1, 1);
    minfer_launch_prelude("launch:rms_norm_f32", "rms_norm_f32");
    rms_norm_f32<<<grid, minfer_launch_block("launch:rms_norm_f32", block), 0, stream>>>(x, w, y, d, eps, n);
    minfer_launch_ok("launch:rms_norm_f32", "rms_norm_f32");
}


// D3-5 1a: decode fused rms_norm + pad40 q8 epilogue (n==1 decode producers).
void launch_rms_norm_quant_pad40(
    const float* x, const float* w, float* y, uint8_t* q8,
    int d, float eps, int n, cudaStream_t stream
) {
    // D3-7 2c: 128-thread wide block (was WARP). The body is
    // blockDim.x-relative and bitwise-identical at either geometry; 128
    // threads cut the per-row write/quantize latency chains 4x (census:
    // 9.4 -> target ~4 us at hidden 5120, 94.6 launches/decode-step).
    minfer_launch_prelude("launch:rms_norm_quant_pad40", "rms_norm_quant_pad40");
    rms_norm_quant_pad40<<<n, minfer_launch_block("launch:rms_norm_quant_pad40", 128), 0, stream>>>(x, w, y, q8, d, eps, n);
    minfer_launch_ok("launch:rms_norm_quant_pad40", "rms_norm_quant_pad40");
}

void launch_add_bias_f32(
    float* y, const float* b, int d, int n, cudaStream_t stream
) {
    dim3 block(64, 1, 1); // 64 threads in x, grid y handles dim remainder
    dim3 grid(n, (d + 63) / 64, 1);
    minfer_launch_prelude("launch:add_bias_f32", "add_bias_f32");
    add_bias_f32<<<grid, minfer_launch_block("launch:add_bias_f32", block), 0, stream>>>(y, b, d);
    minfer_launch_ok("launch:add_bias_f32", "add_bias_f32");
}

void launch_add_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:add_f32", "add_f32");
    add_f32<<<grid, minfer_launch_block("launch:add_f32", block), 0, stream>>>(x, y, z, n);
    minfer_launch_ok("launch:add_f32", "add_f32");
}

void launch_mul_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:mul_f32", "mul_f32");
    mul_f32<<<grid, minfer_launch_block("launch:mul_f32", block), 0, stream>>>(x, y, z, n);
    minfer_launch_ok("launch:mul_f32", "mul_f32");
}

void launch_silu_f32(float* y, int n, cudaStream_t stream) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:silu_f32", "silu_f32");
    silu_f32<<<grid, minfer_launch_block("launch:silu_f32", block), 0, stream>>>(y, n);
    minfer_launch_ok("launch:silu_f32", "silu_f32");
}

void launch_swiglu_f32(
    const float* gate, const float* up, float* dst, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:swiglu_f32", "swiglu_f32");
    swiglu_f32<<<grid, minfer_launch_block("launch:swiglu_f32", block), 0, stream>>>(gate, up, dst, n);
    minfer_launch_ok("launch:swiglu_f32", "swiglu_f32");
}

void launch_f32_bits_to_i32(
    const float* src, int* dst, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:f32_bits_to_i32", "f32_bits_to_i32");
    f32_bits_to_i32<<<grid, minfer_launch_block("launch:f32_bits_to_i32", block), 0, stream>>>(src, dst, n);
    minfer_launch_ok("launch:f32_bits_to_i32", "f32_bits_to_i32");
}

void launch_rope_f32(
    float* x, int n_head, int n_dims, int nt,
    float freq_base, float freq_scale,
    const int* positions, cudaStream_t stream
) {
    int block_sz = 64; // threads per head dimension
    dim3 block(block_sz, 1, 1);
    dim3 grid(nt, n_head, 1);
    minfer_launch_prelude("launch:rope_f32", "rope_f32");
    rope_f32<<<grid, minfer_launch_block("launch:rope_f32", block), 0, stream>>>(x, n_head, n_dims, nt, freq_base, freq_scale, positions);
    minfer_launch_ok("launch:rope_f32", "rope_f32");
}
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// These kernels are plain `__global__` functions, but their module is loaded
// by the pre-warm, so the family keeps its own registration entry.
extern "C" void minfer_prewarm_ops_elementwise_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, swiglu_f32_off);
    MINFER_PREWARM_ONE(a, rms_norm_f32);
    MINFER_PREWARM_ONE(a, rope_f32);
}
