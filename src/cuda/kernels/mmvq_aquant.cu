// src/cuda/kernels/mmvq_aquant.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


// ─── 8e-reversal: llama.cpp MMVQ structure for DECODE (nt == 1) ───────────
// The original 8e verdict ("116 GB/s = the platform's streaming limit") was
// wrong: a plain read-only kernel does 252.7 GB/s on GB10 (93% of the 273
// GB/s theoretical). The f32-activation kernel achieved only ~46% because it
// runs 2 warps x 4 rows with each lane serially processing whole 144B blocks
// (~28K threads in flight at 7B ffn_down) — not enough parallelism to hide
// LPDDR latency. llama.cpp's mul_mat_vec_q uses ONE output row per block
// with the row's (block, 32-element sub-block) units spread round-robin over
// 256 threads (8 warps), int dp4a dots over q8-quantized activations, and a
// block-wide reduction — ~917K threads in flight at the same shape.
// Measured (bench8e2, L2-defeated, 7B shapes): 194–207 GB/s vs 112–117,



// 40B layout: 2B f16 d, 2B pad, 32B int8 payload (offset 4), 4B i32 sum of the
// quantized values (offset 36 — the pad40 slack). The sum feeds the MMQ prefill
// GEMM's min-term correction (llama.cpp's q8_1 "s"); the MMVQ decode kernels
// only read d and the payload, so the extra word is invisible to them.
__global__ void quantize_q8_0_pad40(
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
    const float* src = x + (size_t)t * dim + b * 32;
    uint8_t* dst = y + ((size_t)t * nb + b) * Q8PB;
    // P6: tree-reduced amax (the serial fmaxf chain was latency-bound)
    // and 16B loads / 4B register-packed stores. Math is bit-identical:
    // max is exact for any association, the rintf pass is unchanged.
    float4 sv[8];
    #pragma unroll
    for (int v = 0; v < 8; v++)
        sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
    float am = 0.0f;
    #pragma unroll
    for (int v = 0; v < 8; v++)
        am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                             fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
    float d = am / 127.0f;
    float di = (d != 0.0f) ? 1.0f / d : 0.0f;
    *reinterpret_cast<__half*>(dst) = __float2half(d);
    int s = 0;
    uint32_t packed[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        const float* e = &sv[v].x;
        uint32_t p = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int q = int(rintf(e[j] * di));
            q = max(-128, min(127, q));
            p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);
            s += q;
        }
        packed[v] = p;
    }
    #pragma unroll
    for (int v = 0; v < 8; v++)
        *reinterpret_cast<uint32_t*>(dst + 4 + 4 * v) = packed[v];
    *reinterpret_cast<uint32_t*>(dst + 36) = uint32_t(s);
}

// --- P6 r34: transposed-A q8_0 quantize prepass ----------------------------
// Quantizes activations to q8_0 and writes the result PRE-TRANSPOSED into the
// exact layout mmq_raw_nb_bt_kernel stages via bulk LDG->STS (llama.cpp's
// quantize_mmq_q8_1 design; the transpose that the NB kernel used to do per
// (block, k-tile) is hoisted into this one-per-GEMM prepass). The qs plane is
// emitted swizzled per-64-token-block ([ntb][nchunk][2048]) and the d|ssum
// packed scale into [ntb][nchunk][256], both byte-identical (after the stored
// swizzle) to what the old NB smem staging produced — the quantized values
// (qs bytes, d, ssum) are bit-identical to quantize_q8_0_pad40, only reordered.
__global__ void quantize_q8_0_pad40_t(
    const float* __restrict__ x,
    uint8_t* __restrict__ yqs,    // [ntb][nchunk][2048]
    uint8_t* __restrict__ ysda,   // [ntb][nchunk][256]
    int dim, int nt, int nchunk, int ntb
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pad = ntb * MMQ_A_BLK * nchunk;
    if (tid >= total_pad) return;
    int t = tid / nchunk;
    int b = tid % nchunk;
    const int r = t & (MMQ_A_BLK - 1);  // local token 0..63
    const int tb = t >> 6;              // 64-token block index

    // Quantize (t, b). For the padded tail tokens (t >= nt) emit zeros so the
    // transposed buffer is deterministic regardless of buffer reuse; those rows
    // are never written to C (the kernel's write-back guards i < nt).
    float d = 0.0f; int ssum = 0; uint32_t packed[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) packed[v] = 0;
    if (t < nt) {
        const float* src = x + (size_t)t * dim + b * 32;
        float4 sv[8];
        #pragma unroll
        for (int v = 0; v < 8; v++)
            sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
        float am = 0.0f;
        #pragma unroll
        for (int v = 0; v < 8; v++)
            am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                                 fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
        d = am / 127.0f;
        float di = (d != 0.0f) ? 1.0f / d : 0.0f;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const float* e = &sv[v].x;
            uint32_t p = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int q = int(rintf(e[j] * di));
                q = max(-128, min(127, q));
                p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);
                ssum += q;
            }
            packed[v] = p;
        }
    }

    // Swizzled qs write — byte-identical to the NB kernel's old smem staging
    // (qa8 + (R&~3)*32 + ((((R&3)<<1 + (u>>2)) ^ ((R>>2)&7)) << 4) + (u&3)*4).
    const int t4 = r & 3, grp = r & ~3;
    const int xswz = (r >> 2) & 7;
    size_t qbase = ((size_t)tb * nchunk + b) * MMQ_A_QASZ + grp * 32;
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        const int off = (((t4 * 2 + (u >> 2)) ^ xswz) << 4) + (u & 3) * 4;
        *reinterpret_cast<uint32_t*>(yqs + qbase + off) = packed[u];
    }
    // Packed d|ssum (r31 Q-major region split of the old sda_q).
    const int g = r >> 4, t15 = r & 15, q = t15 & 7, half = t15 >> 3;
    const int rg = g >> 1, gsel = g & 1;
    size_t sbase = ((size_t)tb * nchunk + b) * MMQ_A_SDASZ
                   + (rg * 32 + q * 4 + gsel * 2 + half) * 4;
    __half dh = __float2half(d);
    uint16_t dbits = *reinterpret_cast<uint16_t*>(&dh);
    *reinterpret_cast<uint32_t*>(ysda + sbase) =
        (uint32_t)dbits | ((uint32_t)(uint16_t)ssum << 16);
}

// --- P6 r51: producer-fused A-quantize (rms_norm / swiglu -> pad40_t) ------
// Fuses the MMQ A-quantize prepass INTO its producers: in the qwen2 prefill
// graph every rms_norm/swiglu output is EXCLUSIVELY a GEMM input, so the
// standalone prepass re-reads data the producer JUST wrote (r47 table: the
// prepass was 7.4% of the wall after r49's shared-A dedup). Each fused
// kernel emits the producer's f32 output (bit-identical to the standalone
// kernel: same per-element expressions, same reduction mapping/order) AND
// the pad40_t transposed quantize plane (bit-identical to
// quantize_q8_0_pad40_t: the quantize body and swizzled stores are that
// kernel's code verbatim, reading the just-written rows back through L1/L2).
// The plane is consumed through the r49 MmqCache — the host wrapper
// registers it keyed on the f32 output's device pointer, so prefill_mmq
// needs no change. Gated by MINFER_MMQ_A_FUSE=1 ANDed with the full MMQ
// gate set; see CudaState::rms_norm_quant / swiglu_quant (src/cuda.rs).

__global__ void rms_norm_quant_f32_t(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    uint8_t* __restrict__ yqs,   // [ntb][nchunk][2048] swizzled qs plane
    uint8_t* __restrict__ ysda,  // [ntb][nchunk][256] packed d|ssum
    int d, float eps, int n, int nchunk, int ntb
) {
    // Phase 1: rms_norm — one warp per row, lane mapping and accumulation
    // order identical to rms_norm_f32 (bit-identical output).
    const int row = blockIdx.x * RMSQ_RPB + (threadIdx.x >> 5);
    const int lane = threadIdx.x & (WARP - 1);
    if (row < n) {
        int d4 = d / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x + row * d);
        float ss = 0.0f;
        for (int i = lane; i < d4; i += WARP) {
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = warp_reduce_sum(ss);
        float scale = rsqrtf(ss / (float)d + eps);
        float4* y4 = reinterpret_cast<float4*>(y + row * d);
        const float4* w4 = reinterpret_cast<const float4*>(w);
        for (int i = lane; i < d4; i += WARP) {
            float4 wv = w4[i];
            float4 xv = x4[i];
            y4[i].x = xv.x * scale * wv.x;
            y4[i].y = xv.y * scale * wv.y;
            y4[i].z = xv.z * scale * wv.z;
            y4[i].w = xv.w * scale * wv.w;
        }
    }
    __syncthreads();
    // Phase 2: quantize this block's rows into the pad40_t plane — the
    // quantize_q8_0_pad40_t body verbatim (one thread per (token, chunk),
    // strided over the block's 8 rows). The grid covers the 64-padded token
    // count, so the padded tail rows are zero-filled exactly like the
    // standalone prepass (deterministic plane regardless of scratch reuse).
    const int row0 = blockIdx.x * RMSQ_RPB;
    const int nrows = min(RMSQ_RPB, ntb * 64 - row0);
    const int tasks = nrows * nchunk;
    for (int k = threadIdx.x; k < tasks; k += blockDim.x) {
        const int t = row0 + k / nchunk;
        const int b = k % nchunk;
        const int r = t & (MMQ_A_BLK - 1);
        const int tb = t >> 6;
        float dsc = 0.0f; int ssum = 0; uint32_t packed[8];
        #pragma unroll
        for (int v = 0; v < 8; v++) packed[v] = 0;
        if (t < n) {
            const float* src = y + (size_t)t * d + b * 32;
            float4 sv[8];
            #pragma unroll
            for (int v = 0; v < 8; v++)
                sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
            float am = 0.0f;
            #pragma unroll
            for (int v = 0; v < 8; v++)
                am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                                     fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
            dsc = am / 127.0f;
            float di = (dsc != 0.0f) ? 1.0f / dsc : 0.0f;
            #pragma unroll
            for (int v = 0; v < 8; v++) {
                const float* e = &sv[v].x;
                uint32_t p = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int q = int(rintf(e[j] * di));
                    q = max(-128, min(127, q));
                    p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);
                    ssum += q;
                }
                packed[v] = p;
            }
        }
        const int t4 = r & 3, grp = r & ~3;
        const int xswz = (r >> 2) & 7;
        size_t qbase = ((size_t)tb * nchunk + b) * MMQ_A_QASZ + grp * 32;
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            const int off = (((t4 * 2 + (u >> 2)) ^ xswz) << 4) + (u & 3) * 4;
            *reinterpret_cast<uint32_t*>(yqs + qbase + off) = packed[u];
        }
        const int g = r >> 4, t15 = r & 15, qq = t15 & 7, h15 = t15 >> 3;
        const int rg = g >> 1, gsel = g & 1;
        size_t sbase = ((size_t)tb * nchunk + b) * MMQ_A_SDASZ
                       + (rg * 32 + qq * 4 + gsel * 2 + h15) * 4;
        __half dh = __float2half(dsc);
        uint16_t dbits = *reinterpret_cast<uint16_t*>(&dh);
        *reinterpret_cast<uint32_t*>(ysda + sbase) =
            (uint32_t)dbits | ((uint32_t)(uint16_t)ssum << 16);
    }
}

// Fused swiglu + pad40_t quantize: one block per token row. Phase 1 computes
// dst = silu(gate) * up coalesced (per-element expression identical to
// swiglu_f32 — bit-identical output); phase 2 re-quantizes the row's chunks
// with the quantize_q8_0_pad40_t body verbatim, reading dst back through
// L1/L2 (rows this block just wrote — a cross-thread register hand-off would
// need the amax groups and lane mapping to align, and an uncoalesced f32
// store pattern would cost more than the L2 re-read).
__global__ void swiglu_quant_f32_t(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ dst,
    uint8_t* __restrict__ yqs,   // [ntb][nchunk][2048]
    uint8_t* __restrict__ ysda,  // [ntb][nchunk][256]
    int dim, int nt, int nchunk, int ntb
) {
    const int t = blockIdx.x;  // one token row per block (grid = ntb*64)
    if (t < nt) {
        const float4* g4 = reinterpret_cast<const float4*>(gate + (size_t)t * dim);
        const float4* u4 = reinterpret_cast<const float4*>(up + (size_t)t * dim);
        float4* o4 = reinterpret_cast<float4*>(dst + (size_t)t * dim);
        const int n4 = dim / 4;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float4 gv = g4[i];
            float4 uv = u4[i];
            float4 ov;
            ov.x = (gv.x / (1.0f + expf(-gv.x))) * uv.x;
            ov.y = (gv.y / (1.0f + expf(-gv.y))) * uv.y;
            ov.z = (gv.z / (1.0f + expf(-gv.z))) * uv.z;
            ov.w = (gv.w / (1.0f + expf(-gv.w))) * uv.w;
            o4[i] = ov;
        }
    }
    __syncthreads();
    const int r = t & (MMQ_A_BLK - 1);
    const int tb = t >> 6;
    for (int b = threadIdx.x; b < nchunk; b += blockDim.x) {
        float dsc = 0.0f; int ssum = 0; uint32_t packed[8];
        #pragma unroll
        for (int v = 0; v < 8; v++) packed[v] = 0;
        if (t < nt) {
            const float* src = dst + (size_t)t * dim + b * 32;
            float4 sv[8];
            #pragma unroll
            for (int v = 0; v < 8; v++)
                sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
            float am = 0.0f;
            #pragma unroll
            for (int v = 0; v < 8; v++)
                am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                                     fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
            dsc = am / 127.0f;
            float di = (dsc != 0.0f) ? 1.0f / dsc : 0.0f;
            #pragma unroll
            for (int v = 0; v < 8; v++) {
                const float* e = &sv[v].x;
                uint32_t p = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int q = int(rintf(e[j] * di));
                    q = max(-128, min(127, q));
                    p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);
                    ssum += q;
                }
                packed[v] = p;
            }
        }
        const int t4 = r & 3, grp = r & ~3;
        const int xswz = (r >> 2) & 7;
        size_t qbase = ((size_t)tb * nchunk + b) * MMQ_A_QASZ + grp * 32;
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            const int off = (((t4 * 2 + (u >> 2)) ^ xswz) << 4) + (u & 3) * 4;
            *reinterpret_cast<uint32_t*>(yqs + qbase + off) = packed[u];
        }
        const int g = r >> 4, t15 = r & 15, qq = t15 & 7, h15 = t15 >> 3;
        const int rg = g >> 1, gsel = g & 1;
        size_t sbase = ((size_t)tb * nchunk + b) * MMQ_A_SDASZ
                       + (rg * 32 + qq * 4 + gsel * 2 + h15) * 4;
        __half dh = __float2half(dsc);
        uint16_t dbits = *reinterpret_cast<uint16_t*>(&dh);
        *reinterpret_cast<uint32_t*>(ysda + sbase) =
            (uint32_t)dbits | ((uint32_t)(uint16_t)ssum << 16);
    }
}
extern "C" {

void launch_quantize_q8_0_pad40(
    const float* x, uint8_t* y, int dim, int nt, cudaStream_t stream
) {
    int nb = dim / 32;
    long long total = (long long)nt * nb;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    minfer_launch_prelude("launch:quantize_q8_0_pad40", "quantize_q8_0_pad40");
    quantize_q8_0_pad40<<<(int)grid, minfer_launch_block("launch:quantize_q8_0_pad40", block), 0, stream>>>(x, y, dim, nt);
    minfer_launch_ok("launch:quantize_q8_0_pad40", "quantize_q8_0_pad40");
}

void launch_quantize_q8_0_pad40_t(
    const float* x, uint8_t* yqs, uint8_t* ysda,
    int dim, int nt, int nchunk, int ntb, cudaStream_t stream
) {
    long long total = (long long)ntb * MMQ_A_BLK * nchunk;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    minfer_launch_prelude("launch:quantize_q8_0_pad40_t", "quantize_q8_0_pad40_t");
    quantize_q8_0_pad40_t<<<(int)grid, minfer_launch_block("launch:quantize_q8_0_pad40_t", block), 0, stream>>>(x, yqs, ysda, dim, nt, nchunk, ntb);
    minfer_launch_ok("launch:quantize_q8_0_pad40_t", "quantize_q8_0_pad40_t");
}

// r51: producer-fused rms_norm + pad40_t quantize (see the kernel comment).
// The grid covers the 64-PADDED token count so the padded tail rows are
// zero-filled exactly like the standalone quantize_q8_0_pad40_t prepass.
void launch_rms_norm_quant_f32_t(
    const float* x, const float* w, float* y,
    uint8_t* yqs, uint8_t* ysda,
    int d, float eps, int n, int nchunk, int ntb, cudaStream_t stream
) {
    int grid = ntb * (MMQ_A_BLK / RMSQ_RPB);
    minfer_launch_prelude("launch:rms_norm_quant_f32_t", "rms_norm_quant_f32_t");
    rms_norm_quant_f32_t<<<grid, minfer_launch_block("launch:rms_norm_quant_f32_t", RMSQ_RPB * WARP), 0, stream>>>(
        x, w, y, yqs, ysda, d, eps, n, nchunk, ntb);
    minfer_launch_ok("launch:rms_norm_quant_f32_t", "rms_norm_quant_f32_t");
}

// r51: producer-fused swiglu + pad40_t quantize (see the kernel comment).
void launch_swiglu_quant_f32_t(
    const float* gate, const float* up, float* dst,
    uint8_t* yqs, uint8_t* ysda,
    int dim, int nt, int nchunk, int ntb, cudaStream_t stream
) {
    minfer_launch_prelude("launch:swiglu_quant_f32_t", "swiglu_quant_f32_t");
    swiglu_quant_f32_t<<<ntb * MMQ_A_BLK, minfer_launch_block("launch:swiglu_quant_f32_t", 256), 0, stream>>>(
        gate, up, dst, yqs, ysda, dim, nt, nchunk, ntb);
    minfer_launch_ok("launch:swiglu_quant_f32_t", "swiglu_quant_f32_t");
}
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// These kernels are plain `__global__` functions, but their module is loaded
// by the pre-warm, so the family keeps its own registration entry.
extern "C" void minfer_prewarm_mmvq_aquant_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, quantize_q8_0_pad40);
    MINFER_PREWARM_ONE(a, quantize_q8_0_pad40_t);
}
