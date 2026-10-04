// src/cuda/kernels/mmvq_skipwrite.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


// --- P6 r52: mode-2 skip-write variants (MINFER_MMQ_A_FUSE=2) ---------------
// Same pad40_t planes as the r51 fused kernels, but the producer's f32 output
// is NOT written — the global write + L1/L2 re-read round-trip disappears:
//   rms:    one warp per row as before; each warp re-derives its row's y
//           values per chunk in registers (phase-1 expression verbatim) and
//           the 8 lanes owning a 32-float chunk exchange amax/ssum via
//           shfl_xor (max/int-sum are order-insensitive, so the plane is
//           byte-identical to the mode-1 kernel's).
//   swiglu: one thread per (token, chunk) holds the 32 silu*up values in
//           registers (the r51 phase-2 L2 re-read disappears with the write).
// LEGAL ONLY when the f32 output is provably dead: its only consumers are the
// immediately following consecutive MatMul nodes, which consume the PLANE via
// the r49 MmqCache keyed on the (unwritten) f32 pointer (window-safety proof:
// docs/CUDA_OPTIMIZATION.md P6 r52). The cache refuses to re-quantize a
// dead-write buffer (MmqCache::dead_write guard in src/cuda.rs), so any
// window violation fails loudly instead of silently reading garbage.
__global__ void rms_norm_quant_nw_f32_t(
    const float* __restrict__ x,
    const float* __restrict__ w,
    uint8_t* __restrict__ yqs,   // [ntb][nchunk][2048] swizzled qs plane
    uint8_t* __restrict__ ysda,  // [ntb][nchunk][256] packed d|ssum
    int d, float eps, int n, int nchunk, int ntb
) {
    const int row = blockIdx.x * RMSQ_RPB + (threadIdx.x >> 5);
    const int lane = threadIdx.x & (WARP - 1);
    // Phase 1: identical to rms_norm_quant_f32_t — same lane mapping and
    // accumulation order over x, so `scale` is bit-identical. No y store.
    float scale = 0.0f;
    if (row < n) {
        const int d4 = d / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x + row * d);
        float ss = 0.0f;
        for (int i = lane; i < d4; i += WARP) {
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = warp_reduce_sum(ss);
        scale = rsqrtf(ss / (float)d + eps);
    }
    // Common swizzle constants for this row (same expressions as mode 1).
    const int r = row & (MMQ_A_BLK - 1);
    const int tb = row >> 6;
    const int t4 = r & 3, grp = r & ~3, xswz = (r >> 2) & 7;
    const int g = r >> 4, t15 = r & 15, qq = t15 & 7, h15 = t15 >> 3;
    const int rg = g >> 1, gsel = g & 1;
    if (row >= n) {
        // Padded-tail row: zero-fill this row's plane slots exactly like the
        // standalone prepass (deterministic plane regardless of scratch
        // reuse). grid = ntb*(64/RPB) covers every padded row.
        for (int b = lane; b < nchunk; b += WARP) {
            size_t qbase = ((size_t)tb * nchunk + b) * MMQ_A_QASZ + grp * 32;
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                const int off = (((t4 * 2 + (u >> 2)) ^ xswz) << 4) + (u & 3) * 4;
                *reinterpret_cast<uint32_t*>(yqs + qbase + off) = 0;
            }
            size_t sbase = ((size_t)tb * nchunk + b) * MMQ_A_SDASZ
                           + (rg * 32 + qq * 4 + gsel * 2 + h15) * 4;
            *reinterpret_cast<uint32_t*>(ysda + sbase) = 0;
        }
        return;
    }
    // Phase 2: quantize THIS warp's row (warp-uniform row => the shfl_xor
    // reductions below never see divergence). Chunk c = 4k + lane/8 covers
    // float4s 32k+lane; d % 256 == 0 makes d4 % 32 == 0 (exact loop).
    const int d4 = d / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x + row * d);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (int k = 0; k < d4 / 32; k++) {
        float4 xv = x4[k * 32 + lane];
        float4 wv = w4[k * 32 + lane];
        // y expression verbatim from rms_norm_quant_f32_t phase 1 (bit-identical
        // f32 values — the mode-1 kernel quantizes these after a memory
        // round-trip, which is exact for f32).
        float v0 = xv.x * scale * wv.x;
        float v1 = xv.y * scale * wv.y;
        float v2 = xv.z * scale * wv.z;
        float v3 = xv.w * scale * wv.w;
        float am = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        // 8-lane group reduce (lanes [g8*8, g8*8+8) own one 32-float chunk).
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        float dsc = am / 127.0f;
        float di = (dsc != 0.0f) ? 1.0f / dsc : 0.0f;
        int q0 = max(-128, min(127, int(rintf(v0 * di))));
        int q1 = max(-128, min(127, int(rintf(v1 * di))));
        int q2 = max(-128, min(127, int(rintf(v2 * di))));
        int q3 = max(-128, min(127, int(rintf(v3 * di))));
        int ssum = (q0 + q1) + (q2 + q3);
        ssum += __shfl_xor_sync(0xffffffffu, ssum, 4);
        ssum += __shfl_xor_sync(0xffffffffu, ssum, 2);
        ssum += __shfl_xor_sync(0xffffffffu, ssum, 1);
        const uint32_t p = (uint32_t)(uint8_t)(int8_t)q0
                         | ((uint32_t)(uint8_t)(int8_t)q1 << 8)
                         | ((uint32_t)(uint8_t)(int8_t)q2 << 16)
                         | ((uint32_t)(uint8_t)(int8_t)q3 << 24);
        const int b = k * 4 + (lane >> 3);   // chunk index: float4 (32k+lane)/8
        const int u = lane & 7;              // packed word = float4 position
        size_t qbase = ((size_t)tb * nchunk + b) * MMQ_A_QASZ + grp * 32;
        const int off = (((t4 * 2 + (u >> 2)) ^ xswz) << 4) + (u & 3) * 4;
        *reinterpret_cast<uint32_t*>(yqs + qbase + off) = p;
        if (u == 0) {
            size_t sbase = ((size_t)tb * nchunk + b) * MMQ_A_SDASZ
                           + (rg * 32 + qq * 4 + gsel * 2 + h15) * 4;
            __half dh = __float2half(dsc);
            uint16_t dbits = *reinterpret_cast<uint16_t*>(&dh);
            *reinterpret_cast<uint32_t*>(ysda + sbase) =
                (uint32_t)dbits | ((uint32_t)(uint16_t)ssum << 16);
        }
    }
}

// Mode-2 swiglu: no dst write, single phase. Per token row, the block sweeps
// the row's float4s in COALESCED rounds (lane l loads float4 rd*256+l of
// gate/up — 512 B per warp access, the r51 phase-1 pattern), computes
// silu*mul in registers, and quantizes via the rms-nw lane mapping: the 8
// lanes holding one 32-float chunk (float4s 8c..8c+7 = lanes 8g..8g+7 of the
// round) exchange amax/ssum with 3 __shfl_xor steps. (A naive per-thread
// chunk mapping — one thread quantizing a whole 128-B chunk — makes the gate/
// up loads lane-strided and measured SLOWER than the mode-1 write+re-read.)
// silu*mul expression verbatim from swiglu_f32; tail rows (t >= nt) zero-fill
// their plane slots exactly like the mode-1 kernel / standalone prepass.
__global__ void swiglu_quant_nw_f32_t(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    uint8_t* __restrict__ yqs,   // [ntb][nchunk][2048]
    uint8_t* __restrict__ ysda,  // [ntb][nchunk][256]
    int dim, int nt, int nchunk, int ntb
) {
    const int t = blockIdx.x;  // one token row per block (grid = ntb*64)
    const int r = t & (MMQ_A_BLK - 1);
    const int tb = t >> 6;
    const int t4 = r & 3, grp = r & ~3, xswz = (r >> 2) & 7;
    const int g = r >> 4, t15 = r & 15, qq = t15 & 7, h15 = t15 >> 3;
    const int rg = g >> 1, gsel = g & 1;
    const int n4 = dim / 4;
    // dim % 256 == 0 => n4 % 64 == 0 => every 8-lane shuffle group maps to
    // WHOLE chunks (a group's 8 float4s are all < n4 or all >= n4).
    for (int f = threadIdx.x; f < ((n4 + blockDim.x - 1) / blockDim.x) * blockDim.x;
         f += blockDim.x) {
        const bool active = t < nt && f < n4;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if (active) {
            const float4* g4 = reinterpret_cast<const float4*>(gate + (size_t)t * dim);
            const float4* u4 = reinterpret_cast<const float4*>(up + (size_t)t * dim);
            float4 gv = g4[f];
            float4 uv = u4[f];
            v0 = (gv.x / (1.0f + expf(-gv.x))) * uv.x;
            v1 = (gv.y / (1.0f + expf(-gv.y))) * uv.y;
            v2 = (gv.z / (1.0f + expf(-gv.z))) * uv.z;
            v3 = (gv.w / (1.0f + expf(-gv.w))) * uv.w;
        }
        float am = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        // 8-lane group reduce (lanes 8g..8g+7 hold chunk (f/8)'s 32 values).
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        float dsc = am / 127.0f;
        float di = (dsc != 0.0f) ? 1.0f / dsc : 0.0f;
        int q0 = max(-128, min(127, int(rintf(v0 * di))));
        int q1 = max(-128, min(127, int(rintf(v1 * di))));
        int q2 = max(-128, min(127, int(rintf(v2 * di))));
        int q3 = max(-128, min(127, int(rintf(v3 * di))));
        int ssum = (q0 + q1) + (q2 + q3);
        ssum += __shfl_xor_sync(0xffffffffu, ssum, 4);
        ssum += __shfl_xor_sync(0xffffffffu, ssum, 2);
        ssum += __shfl_xor_sync(0xffffffffu, ssum, 1);
        if (f < n4) {
            // t >= nt rows land here with all-zero v/am/ssum — exactly the
            // standalone prepass's deterministic padded-tail zero-fill.
            const uint32_t p = (uint32_t)(uint8_t)(int8_t)q0
                             | ((uint32_t)(uint8_t)(int8_t)q1 << 8)
                             | ((uint32_t)(uint8_t)(int8_t)q2 << 16)
                             | ((uint32_t)(uint8_t)(int8_t)q3 << 24);
            const int c = f >> 3;                    // chunk index of this float4
            const int u = f & 7;                     // packed word = float4 position
            size_t qbase = ((size_t)tb * nchunk + c) * MMQ_A_QASZ + grp * 32;
            const int off = (((t4 * 2 + (u >> 2)) ^ xswz) << 4) + (u & 3) * 4;
            *reinterpret_cast<uint32_t*>(yqs + qbase + off) = p;
            if (u == 0) {
                size_t sbase = ((size_t)tb * nchunk + c) * MMQ_A_SDASZ
                               + (rg * 32 + qq * 4 + gsel * 2 + h15) * 4;
                __half dh = __float2half(dsc);
                uint16_t dbits = *reinterpret_cast<uint16_t*>(&dh);
                *reinterpret_cast<uint32_t*>(ysda + sbase) =
                    (uint32_t)dbits | ((uint32_t)(uint16_t)ssum << 16);
            }
        }
    }
}

__global__ void __launch_bounds__(256) q4_k_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = (id + 255) / 256;
    const int row_stride = nbe * Q4KB;
    const int nsub = (id + 31) / 32; // ceil — partial tail super-blocks excluded
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 3, sub = u & 7;
        const uint8_t* blk = weights + (size_t)row * row_stride + blk_i * Q4KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        uint8_t s8, m8;
        get_scale_min_k4(sub, blk + 4, &s8, &m8);
        // sub-block nibbles: chunk (sub>>1) of 32B, lo nibbles for even sub,
        // hi for odd; element l of the sub-block ↔ byte l.
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 16 + (sub >> 1) * 32);
        const bool lo = (sub & 1) == 0;
        const uint8_t* x8 = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
        int dot = 0, sx = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t w = qw[v];
            const int n = lo ? (int)(w & 0x0F0F0F0F) : (int)((w >> 4) & 0x0F0F0F0F);
            const int xa = (int)xw[v]; // q8 block covers exactly this sub-block
            dot = __dp4a(n, xa, dot);
            sx  = __dp4a(0x01010101, xa, sx);
        }
        // value = d*s*nib − dm*m (dm is the block's own dmin, not d)
        acc += d8 * ((float)s8 * (float)d * (float)dot - (float)m8 * (float)dm * (float)sx);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) v += warp_sums[k];
        output[(size_t)t * od + row] = v;
    }
}

// 8e follow-up: decode (nt == 1) Q6_K MMVQ — same llama.cpp GB10 structure
// as q4_k_q8_mmvq (one row per 256-thread block, sub-block units round-robin
// across lanes, dp4a over q8 activations). Q6_K sub-blocks are 16 elements
// (16 signed 6-bit scales per 256-element super-block, no min term), so the
// unit is half of a 32-element q8 activation block and the per-unit dot is
// 4 dp4a. Element l of sub-block s (l in [0,16), s in [0,16)):
//   chunk = s/8, group g = (s/2)%4, half is = s%2
//   ql byte  = chunk*64 + (g%2)*32 + is*16 + l   (lo nibble for g<2, hi else)
//   qh byte  = 128 + chunk*32 + is*16 + l        (2-bit pair g per element)
//   scale    = (int8) blk[192 + s]; value = d * scale * (q6 - 32)
// q6_K block strides are 210B raw / 224B padded (7e② repack) — both even but
// not 4-aligned, so the weight side reads 2-byte halves (llama.cpp
// get_int_b2 style); the 256B-aligned base keeps every access 2B-aligned.
__global__ void __launch_bounds__(256) q6_k_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = (id + 255) >> 8;
    const int row_stride = nbe * blk_stride;
    const int nsub = (id + 15) >> 4; // ceil — partial tail super-blocks excluded
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 4, s = u & 15;
        const int chunk = s >> 3, g = (s >> 1) & 3, is = s & 1;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)blk_i * blk_stride;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
        const float sc = (float)(int8_t)blk[192 + s];
        const uint8_t* ql = blk + chunk * 64 + (g & 1) * 32 + is * 16;
        const uint8_t* qh = blk + 128 + chunk * 32 + is * 16;
        const uint8_t* x8 = x8row + (size_t)(u >> 1) * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4) + (u & 1) * 4;
        int dot = 0;
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            const uint32_t wl = (uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * v) |
                                ((uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * v + 2) << 16);
            const uint32_t wh = (uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * v) |
                                ((uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * v + 2) << 16);
            const uint32_t nib = (g < 2) ? (wl & 0x0F0F0F0F) : ((wl >> 4) & 0x0F0F0F0F);
            const uint32_t hi = ((wh >> (2 * g)) & 0x03030303) << 4;
            // q6 nibble+high pair is 0..63; subtract 32 per byte (in-range,
            // never saturates) to get the signed value for dp4a
            const int vi = __vsubss4((int)(nib | hi), 0x20202020);
            dot = __dp4a(vi, (int)xw[v], dot);
        }
        acc += d8 * sc * d * (float)dot;
    }

    mmvq_block_reduce(acc, output, od, t);
}

// 8e follow-up: decode (nt == 1) Q5_K MMVQ — the q4_k_q8_mmvq structure with
// the q5 high-bit plane folded in. Sub-blocks are 32 elements (scales/mins
// packed like q4_K via get_scale_min_k4). Element l of sub-block s:
//   nibble byte = 48 + (s/2)*32 + l   (lo nibble for even s, hi for odd)
//   high bit    = (qh[l] >> s) & 1    (qh plane at byte 16, 1 bit per element)
//   value = d*s5*(nib | bit<<4) − dm*m5  → acc += d8*(s8*d*dot − m8*dm*sx)
// 176B block stride is 16-byte aligned, so the weight side uses uint32 loads.
__global__ void __launch_bounds__(256) q5_k_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = (id + 255) >> 8;
    const int row_stride = nbe * Q5KB;
    const int nsub = (id + 31) >> 5; // ceil — partial tail super-blocks excluded
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 3, sub = u & 7;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)blk_i * Q5KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        uint8_t s8, m8;
        get_scale_min_k4(sub, blk + 4, &s8, &m8);
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 48 + (sub >> 1) * 32);
        const bool lo = (sub & 1) == 0;
        const uint8_t* x8 = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
        int dot = 0, sx = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t w = qw[v];
            // qh word v holds the high bits of elements 4v..4v+3 (byte l of
            // the qh plane, bit `sub`) — one word per nibble word
            const uint32_t qh32 = *reinterpret_cast<const uint32_t*>(blk + 16 + 4 * v);
            const uint32_t nib = lo ? (w & 0x0F0F0F0F) : ((w >> 4) & 0x0F0F0F0F);
            const uint32_t hi = ((qh32 >> sub) & 0x01010101) << 4;
            const int xa = (int)xw[v];
            dot = __dp4a((int)(nib | hi), xa, dot);
            sx  = __dp4a(0x01010101, xa, sx);
        }
        // value = d*s*(nib|bit) − dm*m (dm is the block's own dmin, not d)
        acc += d8 * ((float)s8 * (float)d * (float)dot - (float)m8 * (float)dm * (float)sx);
    }

    mmvq_block_reduce(acc, output, od, t);
}

// R2: weight-streaming rework of the K-quant MMVQ kernels. The 8e kernels
// read each 32B nibble chunk per SUB-BLOCK (the sibling sub re-reads the
// same bytes for the other nibble half — 2× the load instructions, L1
// absorbed) and q6_K used eight 2-byte loads per 16-byte ql/qh piece. The
// v2 kernels map one thread to a 32-element CHUNK (q4_K/q5_K: a sub-pair
// sharing its nibble bytes; q6_K: an is-pair sharing ql/qh bytes), so each
// weight byte is loaded exactly once per row and every access in the
// padded (224B-stride) q6_K layout is 4-byte aligned. Dispatch prefers v2
// when id % 256 == 0 (full super-blocks); MINFER_MMVQ_V1=1 forces the old
// kernels for A/B.
__global__ void __launch_bounds__(256) q4_k_q8_mmvq_v2(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * Q4KB;
    const int npair = id >> 6;         // 64-element chunks (sub-pairs)
    const int nsub = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < npair; u += 256) {
        const int kbx = u >> 2, c = u & 3;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * Q4KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        const int s0 = 2 * c, s1 = 2 * c + 1;
        uint8_t s8a, m8a, s8b, m8b;
        get_scale_min_k4(s0, blk + 4, &s8a, &m8a);
        get_scale_min_k4(s1, blk + 4, &s8b, &m8b);
        // one 32B nibble chunk: lo nibbles = sub s0's 32 elements, hi = sub s1's
        // (16B-aligned: 144·kbx + 16 + 32·c ≡ 0 mod 16)
        const uint4 w0 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32);
        const uint4 w1 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32 + 16);
        const uint32_t ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const uint8_t* x8a = x8row + (size_t)(kbx * 8 + s0) * Q8PB;
        const uint8_t* x8b = x8row + (size_t)(kbx * 8 + s1) * Q8PB;
        const float d8a = h2f(*reinterpret_cast<const uint16_t*>(x8a));
        const float d8b = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xa = reinterpret_cast<const uint32_t*>(x8a + 4);
        const uint32_t* xb = reinterpret_cast<const uint32_t*>(x8b + 4);
        int dota = 0, sxa = 0, dotb = 0, sxb = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t wv = ws[v];
            const int xa_v = (int)xa[v], xb_v = (int)xb[v];
            dota = __dp4a((int)(wv & 0x0F0F0F0F), xa_v, dota);
            sxa  = __dp4a(0x01010101, xa_v, sxa);
            dotb = __dp4a((int)((wv >> 4) & 0x0F0F0F0F), xb_v, dotb);
            sxb  = __dp4a(0x01010101, xb_v, sxb);
        }
        acc += d8a * ((float)s8a * d * (float)dota - (float)m8a * dm * (float)sxa)
             + d8b * ((float)s8b * d * (float)dotb - (float)m8b * dm * (float)sxb);
    }

    mmvq_block_reduce(acc, output, od, t);
}

__global__ void __launch_bounds__(256) q5_k_q8_mmvq_v2(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * Q5KB;
    const int npair = id >> 6;
    const int nsub = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < npair; u += 256) {
        const int kbx = u >> 2, c = u & 3;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * Q5KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        const int s0 = 2 * c, s1 = 2 * c + 1;
        uint8_t s8a, m8a, s8b, m8b;
        get_scale_min_k4(s0, blk + 4, &s8a, &m8a);
        get_scale_min_k4(s1, blk + 4, &s8b, &m8b);
        const uint4 w0 = *reinterpret_cast<const uint4*>(blk + 48 + c * 32);
        const uint4 w1 = *reinterpret_cast<const uint4*>(blk + 48 + c * 32 + 16);
        // the qh plane is 32 bytes SHARED by all 8 sub-blocks (byte l holds
        // one high bit per sub for element l) — every chunk reads the same
        // bytes, only the bit index (s0/s1) differs
        const uint4 h0 = *reinterpret_cast<const uint4*>(blk + 16);
        const uint4 h1 = *reinterpret_cast<const uint4*>(blk + 16 + 16);
        const uint32_t ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const uint32_t hs[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const uint8_t* x8a = x8row + (size_t)(kbx * 8 + s0) * Q8PB;
        const uint8_t* x8b = x8row + (size_t)(kbx * 8 + s1) * Q8PB;
        const float d8a = h2f(*reinterpret_cast<const uint16_t*>(x8a));
        const float d8b = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xa = reinterpret_cast<const uint32_t*>(x8a + 4);
        const uint32_t* xb = reinterpret_cast<const uint32_t*>(x8b + 4);
        int dota = 0, sxa = 0, dotb = 0, sxb = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t wv = ws[v];
            // qh byte l holds one high bit per sub for element l: bit s of
            // the bytes covering this chunk's elements
            const uint32_t qhv = hs[v];
            const uint32_t hia = (((qhv >> s0) & 0x01010101u) << 4);
            const uint32_t hib = (((qhv >> s1) & 0x01010101u) << 4);
            const int xa_v = (int)xa[v], xb_v = (int)xb[v];
            dota = __dp4a((int)((wv & 0x0F0F0F0F) | hia), xa_v, dota);
            sxa  = __dp4a(0x01010101, xa_v, sxa);
            dotb = __dp4a((int)(((wv >> 4) & 0x0F0F0F0F) | hib), xb_v, dotb);
            sxb  = __dp4a(0x01010101, xb_v, sxb);
        }
        acc += d8a * ((float)s8a * d * (float)dota - (float)m8a * dm * (float)sxa)
             + d8b * ((float)s8b * d * (float)dotb - (float)m8b * dm * (float)sxb);
    }

    mmvq_block_reduce(acc, output, od, t);
}

// q6_K v2: one thread per 32-element is-pair (two 16-element sub-blocks
// sharing their ql/qh bytes and one q8 block). Requires the padded 224B
// block stride so every ql/qh access is 4-byte aligned (u32 loads).
__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * blk_stride;
    const int npair = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < npair; u += 256) {
        const int kbx = u >> 3, pair = u & 7;
        const int s0 = 2 * pair, s1 = 2 * pair + 1;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * blk_stride;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
        const float sc0 = (float)(int8_t)blk[192 + s0];
        const float sc1 = (float)(int8_t)blk[192 + s1];
        // v1 mapping with s = 2*pair + half: chunk = s>>3 = pair>>2,
        // g = (s>>1)&3 = pair&3, is = s&1 = half (the pair's two subs share
        // chunk/g; only the 16-byte is-half differs)
        const int chunk = pair >> 2, g = pair & 3;
        // padded 224B stride ⇒ every ql/qh piece is 16B aligned
        const uint4 qla = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32);
        const uint4 qlb = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32 + 16);
        const uint4 qha = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32);
        const uint4 qhb = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32 + 16);
        const uint32_t qls[8] = {qla.x, qla.y, qla.z, qla.w, qlb.x, qlb.y, qlb.z, qlb.w};
        const uint32_t qhs[8] = {qha.x, qha.y, qha.z, qha.w, qhb.x, qhb.y, qhb.z, qhb.w};
        const uint32_t shift = 2 * g;
        const uint8_t* x8 = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
        int dot0 = 0, dot1 = 0;
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            const uint32_t wl0 = qls[v], wl1 = qls[v + 4];
            const uint32_t wh0 = qhs[v], wh1 = qhs[v + 4];
            const uint32_t nib0 = (g < 2) ? (wl0 & 0x0F0F0F0F) : ((wl0 >> 4) & 0x0F0F0F0F);
            const uint32_t nib1 = (g < 2) ? (wl1 & 0x0F0F0F0F) : ((wl1 >> 4) & 0x0F0F0F0F);
            const uint32_t hi0 = ((wh0 >> shift) & 0x03030303) << 4;
            const uint32_t hi1 = ((wh1 >> shift) & 0x03030303) << 4;
            const int vi0 = __vsubss4((int)(nib0 | hi0), 0x20202020);
            const int vi1 = __vsubss4((int)(nib1 | hi1), 0x20202020);
            dot0 = __dp4a(vi0, (int)xw[v], dot0);
            dot1 = __dp4a(vi1, (int)xw[v + 4], dot1);
        }
        acc += d8 * sc0 * d * (float)dot0 + d8 * sc1 * d * (float)dot1;
    }

    mmvq_block_reduce(acc, output, od, t);
}

extern "C" {

void launch_q4_k_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q4_k_q8_mmvq", "q4_k_q8_mmvq");
    q4_k_q8_mmvq<<<grid, minfer_launch_block("launch:q4_k_q8_mmvq", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q4_k_q8_mmvq", "q4_k_q8_mmvq");
}

void launch_q6_k_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int blk_stride, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q6_k_q8_mmvq", "q6_k_q8_mmvq");
    q6_k_q8_mmvq<<<grid, minfer_launch_block("launch:q6_k_q8_mmvq", 256), 0, stream>>>(weights, acts8, output, od, id, nt, blk_stride);
    minfer_launch_ok("launch:q6_k_q8_mmvq", "q6_k_q8_mmvq");
}

void launch_q5_k_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q5_k_q8_mmvq", "q5_k_q8_mmvq");
    q5_k_q8_mmvq<<<grid, minfer_launch_block("launch:q5_k_q8_mmvq", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q5_k_q8_mmvq", "q5_k_q8_mmvq");
}

// R2: weight-streaming rework (see the v2 kernel comments). Same signatures
// as the v1 launchers so dispatch can A/B via MINFER_MMVQ_V1=1.
void launch_q4_k_q8_mmvq_v2(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q4_k_q8_mmvq_v2", "q4_k_q8_mmvq_v2");
    q4_k_q8_mmvq_v2<<<grid, minfer_launch_block("launch:q4_k_q8_mmvq_v2", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q4_k_q8_mmvq_v2", "q4_k_q8_mmvq_v2");
}

void launch_q6_k_q8_mmvq_v2(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int blk_stride, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q6_k_q8_mmvq_v2", "q6_k_q8_mmvq_v2");
    q6_k_q8_mmvq_v2<<<grid, minfer_launch_block("launch:q6_k_q8_mmvq_v2", 256), 0, stream>>>(weights, acts8, output, od, id, nt, blk_stride);
    minfer_launch_ok("launch:q6_k_q8_mmvq_v2", "q6_k_q8_mmvq_v2");
}

void launch_q5_k_q8_mmvq_v2(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q5_k_q8_mmvq_v2", "q5_k_q8_mmvq_v2");
    q5_k_q8_mmvq_v2<<<grid, minfer_launch_block("launch:q5_k_q8_mmvq_v2", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q5_k_q8_mmvq_v2", "q5_k_q8_mmvq_v2");
}

// r52: mode-2 launchers (no f32 output write; see the kernel comments). Grid
// geometry identical to the mode-1 launchers.
void launch_rms_norm_quant_nw_f32_t(
    const float* x, const float* w,
    uint8_t* yqs, uint8_t* ysda,
    int d, float eps, int n, int nchunk, int ntb, cudaStream_t stream
) {
    int grid = ntb * (MMQ_A_BLK / RMSQ_RPB);
    minfer_launch_prelude("launch:rms_norm_quant_nw_f32_t", "rms_norm_quant_nw_f32_t");
    rms_norm_quant_nw_f32_t<<<grid, minfer_launch_block("launch:rms_norm_quant_nw_f32_t", RMSQ_RPB * WARP), 0, stream>>>(
        x, w, yqs, ysda, d, eps, n, nchunk, ntb);
    minfer_launch_ok("launch:rms_norm_quant_nw_f32_t", "rms_norm_quant_nw_f32_t");
}

void launch_swiglu_quant_nw_f32_t(
    const float* gate, const float* up,
    uint8_t* yqs, uint8_t* ysda,
    int dim, int nt, int nchunk, int ntb, cudaStream_t stream
) {
    minfer_launch_prelude("launch:swiglu_quant_nw_f32_t", "swiglu_quant_nw_f32_t");
    swiglu_quant_nw_f32_t<<<ntb * MMQ_A_BLK, minfer_launch_block("launch:swiglu_quant_nw_f32_t", 256), 0, stream>>>(
        gate, up, yqs, ysda, dim, nt, nchunk, ntb);
    minfer_launch_ok("launch:swiglu_quant_nw_f32_t", "swiglu_quant_nw_f32_t");
}
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// These kernels are plain `__global__` functions, but their module is loaded
// by the pre-warm, so the family keeps its own registration entry.
extern "C" void minfer_prewarm_mmvq_skipwrite_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, rms_norm_quant_nw_f32_t);
    MINFER_PREWARM_ONE(a, swiglu_quant_nw_f32_t);
    MINFER_PREWARM_ONE(a, q4_k_q8_mmvq);
    MINFER_PREWARM_ONE(a, q6_k_q8_mmvq);
}
