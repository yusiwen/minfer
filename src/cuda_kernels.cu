// CUDA kernels for minfer — Q4_0 matmul + element-wise ops.
// Translated from: src/metal.metal (Metal shaders) and
//   llama.cpp/ggml/src/ggml-cuda (CUDA kernel patterns)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ─── Block size constants (must match src/block.rs) ───────────
#define Q4B  18   // sizeof(BlockQ4_0): half d + uchar qs[16]
#define Q41B 20   // sizeof(BlockQ4_1): half d + half m + uchar qs[16]
#define Q8B  34   // sizeof(BlockQ8_0): half d + char qs[32]
#define Q4KB 144  // sizeof(BlockQ4_K)
#define Q6KB 210  // sizeof(BlockQ6_K)
#define WARP 32

// ─── Helper: warp-level sum reduction ─────────────────────────
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ─── Helper: fp16 → f32 (using CUDA intrinsics) ──────────────
__device__ float h2f(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ─── Q4_0 × Q8_0 matrix multiplication (bit-exact with CPU) ──
// Thread block: 64 threads (2 warps × 32 lanes)
// Each warp computes NR0=4 consecutive output rows
// Grid: x = ceil(od / (NR0*NSG)), y = nt

__global__ void q4_0_q8_0_matmul(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nb = id / 32;
    int q4s = nb * Q4B;
    int q8s = nb * Q8B;

    const uint8_t* xr = acts + t * q8s;

    float sumf[NR0];
    #pragma unroll
    for (int row = 0; row < NR0; row++) sumf[row] = 0.0f;

    // Each lane handles every WARP-th block
    for (int b = lane_id; b < nb; b += WARP) {
        // Q8_0 block
        float d8 = h2f(*reinterpret_cast<const uint16_t*>(xr + b * Q8B));
        const int8_t* xq = reinterpret_cast<const int8_t*>(xr + b * Q8B + 2);

        for (int row = 0; row < NR0; row++) {
            int o = r0 + row;
            if (o >= od) break;

            const uint8_t* wr = weights + o * q4s;
            float d4 = h2f(*reinterpret_cast<const uint16_t*>(wr + b * Q4B));
            const uint8_t* wq = wr + b * Q4B + 2;

            int bs = 0;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                uint8_t byte = wq[j];
                bs += (int(byte & 0x0F) - 8) * int(xq[j])
                    + (int(byte >> 4) - 8) * int(xq[j + 16]);
            }
            sumf[row] += float(bs) * d4 * d8;
        }
    }

    // Warp-level reduction and write
    for (int row = 0; row < NR0; row++) {
        int o = r0 + row;
        if (o < od) {
            float total = warp_reduce_sum(sumf[row]);
            if (lane_id == 0) {
                output[t * od + o] = total;
            }
        }
    }
}

// ─── Q4_0 × f32 matrix multiplication ─────────────────────────
// Thread block: 64 threads (2 warps), each warp computes 4 rows
// Grid: x = ceil(od / 8), y = nt

__device__ float block_q4_0_dot_y(const uint8_t* block, float sumy, const float* yl, int il) {
    float d = h2f(*reinterpret_cast<const uint16_t*>(block));
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(block + 2) + il / 2;
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        uint16_t v = qs[i / 2];
        acc0 += yl[i + 0] * float(v & 0x000F);
        acc1 += yl[i + 1] * float(v & 0x0F00);
        acc2 += yl[i + 8] * float(v & 0x00F0);
        acc3 += yl[i + 9] * float(v & 0xF000);
    }
    return d * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
}

__global__ void q4_0_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;
    const int QK = 32;
    const int NW = 32;
    const int NQ = 16;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt) return;

    int nb = id / QK;
    int q4s = nb * Q4B;

    const uint8_t* ax0 = weights + (r0 + 0) * q4s;
    const uint8_t* ax1 = weights + (r0 + 1) * q4s;
    const uint8_t* ax2 = weights + (r0 + 2) * q4s;
    const uint8_t* ax3 = weights + (r0 + 3) * q4s;
    const float* y = acts + t * id;

    int ix = lane_id / (NW / NQ);
    int il = (lane_id % (NW / NQ)) * 8;

    float sumf0 = 0, sumf1 = 0, sumf2 = 0, sumf3 = 0;
    float yl[16];
    const float* yb = y + ix * QK + il;

    for (int ib = ix; ib < nb; ib += NQ) {
        float sumy0 = 0, sumy1 = 0;
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            sumy0 += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
            yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
        }
        float sy = sumy0 + sumy1;
        if (r0 + 0 < od) sumf0 += block_q4_0_dot_y(ax0 + ib * Q4B, sy, yl, il);
        if (r0 + 1 < od) sumf1 += block_q4_0_dot_y(ax1 + ib * Q4B, sy, yl, il);
        if (r0 + 2 < od) sumf2 += block_q4_0_dot_y(ax2 + ib * Q4B, sy, yl, il);
        if (r0 + 3 < od) sumf3 += block_q4_0_dot_y(ax3 + ib * Q4B, sy, yl, il);
        yb += QK * NQ;
    }

    sumf0 = warp_reduce_sum(sumf0);
    sumf1 = warp_reduce_sum(sumf1);
    sumf2 = warp_reduce_sum(sumf2);
    sumf3 = warp_reduce_sum(sumf3);
    if (lane_id == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
        if (r0 + 2 < od) output[t * od + r0 + 2] = sumf2;
        if (r0 + 3 < od) output[t * od + r0 + 3] = sumf3;
    }
}

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

// ─── KV cache store: scatter nt rows into persistent cache ───

__global__ void store_kv_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int nkt, int nt,
    const int* positions
) {
    int t = blockIdx.x;
    int j = blockIdx.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = src[t * nkt + j];
}

// ─── GQA Attention (online softmax, 32 threads/head/token) ───
// q/k/v/o layout: [nt][nh][hd]; k/v stored as [nkv][nk][hd]

__global__ void gqa_attn_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    const int* positions,
    int nh, int nk, int hd,
    float scale, int nt
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= nt || h >= nh) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int hk = h / gqa;
    int ne_q = nh * hd;
    int stride_kv = nk * hd;

    const float* qhead = q + t * ne_q + h * hd;
    float* ohead = o + t * ne_q + h * hd;

    int tid = threadIdx.x;
    int hd4 = hd / 4;
    const float4* q4 = reinterpret_cast<const float4*>(qhead);

    // Online softmax with persistent accumulators
    const int NE = 2;
    const int C = WARP * NE;

    float mx = -INFINITY;
    float S = 0.0f;
    float4 oc[32];
    #pragma unroll
    for (int i = 0; i < hd4; i++) oc[i] = make_float4(0, 0, 0, 0);

    for (int batch = 0; batch < nkv; batch += C) {
        float s0 = -INFINITY, s1 = -INFINITY;
        int kv0 = batch + tid * NE;
        int kv1 = kv0 + 1;

        if (kv0 < nkv) {
            const float4* k4 = reinterpret_cast<const float4*>(k + kv0 * stride_kv + hk * hd);
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = k4[i];
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s0 = d * scale;
        }
        if (kv1 < nkv) {
            const float4* k4 = reinterpret_cast<const float4*>(k + kv1 * stride_kv + hk * hd);
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = k4[i];
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s1 = d * scale;
        }

        float batch_mx = fmaxf(s0, s1);
        // Warp-level max reduction
        for (int off = 16; off > 0; off >>= 1)
            batch_mx = fmaxf(batch_mx, __shfl_xor_sync(0xFFFFFFFF, batch_mx, off));
        float new_mx = fmaxf(mx, batch_mx);
        float corr = expf(mx - new_mx);

        float e0 = expf(s0 - new_mx);
        float e1 = expf(s1 - new_mx);

        #pragma unroll
        for (int i = 0; i < hd4; i++) oc[i].x *= corr, oc[i].y *= corr, oc[i].z *= corr, oc[i].w *= corr;
        S *= corr;

        if (kv0 < nkv) {
            const float4* v4 = reinterpret_cast<const float4*>(v + kv0 * stride_kv + hk * hd);
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = v4[i];
                oc[i].x += e0 * vv.x; oc[i].y += e0 * vv.y;
                oc[i].z += e0 * vv.z; oc[i].w += e0 * vv.w;
            }
        }
        if (kv1 < nkv) {
            const float4* v4 = reinterpret_cast<const float4*>(v + kv1 * stride_kv + hk * hd);
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = v4[i];
                oc[i].x += e1 * vv.x; oc[i].y += e1 * vv.y;
                oc[i].z += e1 * vv.z; oc[i].w += e1 * vv.w;
            }
        }
        S += e0 + e1;
        mx = new_mx;
    }

    // Warp-level reduction of S and oc
    S = warp_reduce_sum(S);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        oc[i].x = warp_reduce_sum(oc[i].x);
        oc[i].y = warp_reduce_sum(oc[i].y);
        oc[i].z = warp_reduce_sum(oc[i].z);
        oc[i].w = warp_reduce_sum(oc[i].w);
    }

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float4* o4 = reinterpret_cast<float4*>(ohead);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        o4[i].x = oc[i].x * inv;
        o4[i].y = oc[i].y * inv;
        o4[i].z = oc[i].z * inv;
        o4[i].w = oc[i].w * inv;
    }
}

// ====================================================================
// extern "C" launch wrappers (called from Rust via FFI)
// ====================================================================

extern "C" {

void launch_q4_0_q8_0_matmul(
    const uint8_t* weights, const uint8_t* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q4_0_q8_0_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_q4_0_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q4_0_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_quantize_q8_0(
    const float* x, uint8_t* y, int dim, int nt, cudaStream_t stream
) {
    int nb = dim / 32;
    int total = nt * nb;
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((total + block_sz - 1) / block_sz, 1, 1);
    quantize_q8_0<<<grid, block, 0, stream>>>(x, y, dim, nt);
}

void launch_rms_norm_f32(
    const float* x, const float* w, float* y,
    int d, float eps, int n, cudaStream_t stream
) {
    dim3 block(WARP, 1, 1);
    dim3 grid(n, 1, 1);
    rms_norm_f32<<<grid, block, 0, stream>>>(x, w, y, d, eps, n);
}

void launch_add_bias_f32(
    float* y, const float* b, int d, int n, cudaStream_t stream
) {
    dim3 block(64, 1, 1); // 64 threads in x, grid y handles dim remainder
    dim3 grid(n, (d + 63) / 64, 1);
    add_bias_f32<<<grid, block, 0, stream>>>(y, b, d);
}

void launch_add_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    add_f32<<<grid, block, 0, stream>>>(x, y, z, n);
}

void launch_mul_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    mul_f32<<<grid, block, 0, stream>>>(x, y, z, n);
}

void launch_silu_f32(float* y, int n, cudaStream_t stream) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    silu_f32<<<grid, block, 0, stream>>>(y, n);
}

void launch_swiglu_f32(
    const float* gate, const float* up, float* dst, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    swiglu_f32<<<grid, block, 0, stream>>>(gate, up, dst, n);
}

void launch_rope_f32(
    float* x, int n_head, int n_dims, int nt,
    float freq_base, float freq_scale,
    const int* positions, cudaStream_t stream
) {
    int block_sz = 64; // threads per head dimension
    dim3 block(block_sz, 1, 1);
    dim3 grid(nt, n_head, 1);
    rope_f32<<<grid, block, 0, stream>>>(x, n_head, n_dims, nt, freq_base, freq_scale, positions);
}

void launch_store_kv_f32(
    const float* src, float* dst, int nkt, int nt,
    const int* positions, cudaStream_t stream
) {
    dim3 grid(nt, nkt, 1);
    store_kv_f32<<<grid, dim3(1, 1, 1), 0, stream>>>(src, dst, nkt, nt, positions);
}

void launch_gqa_attn_f32(
    const float* q, const float* k, const float* v, float* o,
    const int* positions, int nh, int nk, int hd,
    float scale, int nt, cudaStream_t stream
) {
    dim3 block(WARP, 1, 1); // 32 threads per block (1 warp)
    dim3 grid(nt, nh, 1);
    gqa_attn_f32<<<grid, block, 0, stream>>>(q, k, v, o, positions, nh, nk, hd, scale, nt);
}

} // extern "C"
