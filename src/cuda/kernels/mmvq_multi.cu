// src/cuda/kernels/mmvq_multi.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


// ─── Step 82: multi-token MMVQ (nt in [2, 8]) ──────────────────────────
// The decode kernels above fix the token in the launch grid (grid.y = nt):
// every (row, token) block re-streams its whole weight row, so nt in
// [2, 15] costs nt full weight passes — the dispatch hole measured in doc
// 81 (D5-1a). The llama.cpp mul_mat_vec_q structure instead keeps the
// token loop inside the block (LLAMA-CPP-MMQ-ANALYSIS.md §12): weight
// bytes are loaded once per row and dotted against ≤ 8 activation rows.
// The kernels below are the multi-token variants of the v1/v2 decode
// kernels; the single-token kernels stay the nt == 1 hot path (unchanged
// code, capture graphs included). Accumulators are a fixed 8-lane array
// with a uniform `t < nt` guard so every index stays compile-time (no
// local-memory spill); nt is uniform across the block (no divergence).
// Per (row, token) the op order matches the sibling single-token kernel,
// so a multi launch is bitwise-equal to nt separate single launches.


__global__ void __launch_bounds__(256) q4_k_q8_mmvq_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int nbe = (id + 255) / 256;
    const int row_stride = nbe * Q4KB;
    const int nsub = (id + 31) / 32; // ceil — partial tail super-blocks excluded
    // D5-R stage 4b (doc 87): token groups of 8 — nt <= 8 runs exactly one
    // group with the original accumulation order (bitwise). Group g>0
    // re-reads the row's weights from DRAM (L2 cannot hold the streamed
    // rows) and measured at parity with the padded GEMM, so the dispatch
    // stays at nt <= 8; the group structure makes the acc[8] cap explicit.
    const int ngrp = (nt + 7) >> 3;
    for (int g = 0; g < ngrp; ++g) {
    const int t0 = g << 3;
    const int tmax = min(8, nt - t0);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 3, sub = u & 7;
        const uint8_t* blk = weights + (size_t)row * row_stride + blk_i * Q4KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        uint8_t s8, m8;
        get_scale_min_k4(sub, blk + 4, &s8, &m8);
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 16 + (sub >> 1) * 32);
        const bool lo = (sub & 1) == 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < tmax) {
                const uint8_t* x8 = acts8 + ((size_t)(t0 + t) * nsub + (size_t)u) * Q8PB;
                const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
                int dot = 0, sx = 0;
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const uint32_t w = qw[v];
                    const int n = lo ? (int)(w & 0x0F0F0F0F) : (int)((w >> 4) & 0x0F0F0F0F);
                    const int xa = (int)xw[v];
                    dot = __dp4a(n, xa, dot);
                    sx  = __dp4a(0x01010101, xa, sx);
                }
                acc[t] += d8 * ((float)s8 * (float)d * (float)dot - (float)m8 * (float)dm * (float)sx);
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, tmax, t0);
    }
}

__global__ void __launch_bounds__(256) q4_k_q8_mmvq_v2_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int nbe = id >> 8;
    const int row_stride = nbe * Q4KB;
    const int npair = id >> 6;         // 64-element chunks (sub-pairs)
    const int nsub = id >> 5;
    // D5-R stage 4b (doc 87): token groups of 8 — nt <= 8 runs exactly one
    // group with the original accumulation order (bitwise). Group g>0
    // re-reads the row's weights from DRAM (L2 cannot hold the streamed
    // rows) and measured at parity with the padded GEMM, so the dispatch
    // stays at nt <= 8; the group structure makes the acc[8] cap explicit.
    const int ngrp = (nt + 7) >> 3;
    for (int g = 0; g < ngrp; ++g) {
    const int t0 = g << 3;
    const int tmax = min(8, nt - t0);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = threadIdx.x; u < npair; u += 256) {
        const int kbx = u >> 2, c = u & 3;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * Q4KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        const int s0 = 2 * c, s1 = 2 * c + 1;
        uint8_t s8a, m8a, s8b, m8b;
        get_scale_min_k4(s0, blk + 4, &s8a, &m8a);
        get_scale_min_k4(s1, blk + 4, &s8b, &m8b);
        const uint4 w0 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32);
        const uint4 w1 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32 + 16);
        const uint32_t ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < tmax) {
                const uint8_t* x8a = acts8 + ((size_t)(t0 + t) * nsub + (size_t)(kbx * 8 + s0)) * Q8PB;
                const uint8_t* x8b = acts8 + ((size_t)(t0 + t) * nsub + (size_t)(kbx * 8 + s1)) * Q8PB;
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
                acc[t] += d8a * ((float)s8a * d * (float)dota - (float)m8a * dm * (float)sxa)
                        + d8b * ((float)s8b * d * (float)dotb - (float)m8b * dm * (float)sxb);
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, tmax, t0);
    }
}

__global__ void __launch_bounds__(256) q5_k_q8_mmvq_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int nbe = (id + 255) >> 8;
    const int row_stride = nbe * Q5KB;
    const int nsub = (id + 31) >> 5; // ceil — partial tail super-blocks excluded

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 3, sub = u & 7;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)blk_i * Q5KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        uint8_t s8, m8;
        get_scale_min_k4(sub, blk + 4, &s8, &m8);
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 48 + (sub >> 1) * 32);
        const bool lo = (sub & 1) == 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < nt) {
                const uint8_t* x8 = acts8 + ((size_t)t * nsub + (size_t)u) * Q8PB;
                const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
                int dot = 0, sx = 0;
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const uint32_t w = qw[v];
                    const uint32_t qh32 = *reinterpret_cast<const uint32_t*>(blk + 16 + 4 * v);
                    const uint32_t nib = lo ? (w & 0x0F0F0F0F) : ((w >> 4) & 0x0F0F0F0F);
                    const uint32_t hi = ((qh32 >> sub) & 0x01010101) << 4;
                    const int xa = (int)xw[v];
                    dot = __dp4a((int)(nib | hi), xa, dot);
                    sx  = __dp4a(0x01010101, xa, sx);
                }
                acc[t] += d8 * ((float)s8 * (float)d * (float)dot - (float)m8 * (float)dm * (float)sx);
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, nt, 0);
}

__global__ void __launch_bounds__(256) q5_k_q8_mmvq_v2_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int nbe = id >> 8;
    const int row_stride = nbe * Q5KB;
    const int npair = id >> 6;
    const int nsub = id >> 5;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < nt) {
                const uint8_t* x8a = acts8 + ((size_t)t * nsub + (size_t)(kbx * 8 + s0)) * Q8PB;
                const uint8_t* x8b = acts8 + ((size_t)t * nsub + (size_t)(kbx * 8 + s1)) * Q8PB;
                const float d8a = h2f(*reinterpret_cast<const uint16_t*>(x8a));
                const float d8b = h2f(*reinterpret_cast<const uint16_t*>(x8b));
                const uint32_t* xa = reinterpret_cast<const uint32_t*>(x8a + 4);
                const uint32_t* xb = reinterpret_cast<const uint32_t*>(x8b + 4);
                int dota = 0, sxa = 0, dotb = 0, sxb = 0;
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const uint32_t wv = ws[v];
                    const uint32_t qhv = hs[v];
                    const uint32_t hia = (((qhv >> s0) & 0x01010101u) << 4);
                    const uint32_t hib = (((qhv >> s1) & 0x01010101u) << 4);
                    const int xa_v = (int)xa[v], xb_v = (int)xb[v];
                    dota = __dp4a((int)((wv & 0x0F0F0F0F) | hia), xa_v, dota);
                    sxa  = __dp4a(0x01010101, xa_v, sxa);
                    dotb = __dp4a((int)(((wv >> 4) & 0x0F0F0F0F) | hib), xb_v, dotb);
                    sxb  = __dp4a(0x01010101, xb_v, sxb);
                }
                acc[t] += d8a * ((float)s8a * d * (float)dota - (float)m8a * dm * (float)sxa)
                        + d8b * ((float)s8b * d * (float)dotb - (float)m8b * dm * (float)sxb);
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, nt, 0);
}

__global__ void __launch_bounds__(256) q6_k_q8_mmvq_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int nbe = (id + 255) >> 8;
    const int row_stride = nbe * blk_stride;
    const int nsub = (id + 15) >> 4; // ceil — partial tail super-blocks excluded

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 4, s = u & 15;
        const int chunk = s >> 3, g = (s >> 1) & 3, is = s & 1;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)blk_i * blk_stride;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
        const float sc = (float)(int8_t)blk[192 + s];
        const uint8_t* ql = blk + chunk * 64 + (g & 1) * 32 + is * 16;
        const uint8_t* qh = blk + 128 + chunk * 32 + is * 16;
        // the four 2-byte weight pairs are token-independent: dequant once
        // per u, then dot against each token's q8 block
        int vi[4];
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
            vi[v] = __vsubss4((int)(nib | hi), 0x20202020);
        }
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < nt) {
                const uint8_t* x8 = acts8 + ((size_t)t * (size_t)(id >> 5) + (size_t)(u >> 1)) * Q8PB;
                const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4) + (u & 1) * 4;
                int dot = 0;
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    dot = __dp4a(vi[v], (int)xw[v], dot);
                }
                acc[t] += d8 * sc * d * (float)dot;
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, nt, 0);
}

__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int nbe = id >> 8;
    const int row_stride = nbe * blk_stride;
    const int npair = id >> 5;
    const int nsub = id >> 5;
    // D5-R stage 4b (doc 87): token groups of 8 — nt <= 8 runs exactly one
    // group with the original accumulation order (bitwise). Group g>0
    // re-reads the row's weights from DRAM (L2 cannot hold the streamed
    // rows) and measured at parity with the padded GEMM, so the dispatch
    // stays at nt <= 8; the group structure makes the acc[8] cap explicit.
    const int ngrp = (nt + 7) >> 3;
    for (int g = 0; g < ngrp; ++g) {
    const int t0 = g << 3;
    const int tmax = min(8, nt - t0);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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
        const int chunk = pair >> 2, gq = pair & 3;
        // padded 224B stride ⇒ every ql/qh piece is 16B aligned
        const uint4 qla = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (gq & 1) * 32);
        const uint4 qlb = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (gq & 1) * 32 + 16);
        const uint4 qha = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32);
        const uint4 qhb = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32 + 16);
        const uint32_t qls[8] = {qla.x, qla.y, qla.z, qla.w, qlb.x, qlb.y, qlb.z, qlb.w};
        const uint32_t qhs[8] = {qha.x, qha.y, qha.z, qha.w, qhb.x, qhb.y, qhb.z, qhb.w};
        const uint32_t shift = 2 * gq;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < tmax) {
                const uint8_t* x8 = acts8 + ((size_t)(t0 + t) * nsub + (size_t)u) * Q8PB;
                const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
                int dot0 = 0, dot1 = 0;
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    const uint32_t wl0 = qls[v], wl1 = qls[v + 4];
                    const uint32_t wh0 = qhs[v], wh1 = qhs[v + 4];
                    const uint32_t nib0 = (gq < 2) ? (wl0 & 0x0F0F0F0F) : ((wl0 >> 4) & 0x0F0F0F0F);
                    const uint32_t nib1 = (gq < 2) ? (wl1 & 0x0F0F0F0F) : ((wl1 >> 4) & 0x0F0F0F0F);
                    const uint32_t hi0 = ((wh0 >> shift) & 0x03030303) << 4;
                    const uint32_t hi1 = ((wh1 >> shift) & 0x03030303) << 4;
                    const int vi0 = __vsubss4((int)(nib0 | hi0), 0x20202020);
                    const int vi1 = __vsubss4((int)(nib1 | hi1), 0x20202020);
                    dot0 = __dp4a(vi0, (int)xw[v], dot0);
                    dot1 = __dp4a(vi1, (int)xw[v + 4], dot1);
                }
                // textually identical to the v2 accumulation statement (same contraction)
                acc[t] += d8 * sc0 * d * (float)dot0 + d8 * sc1 * d * (float)dot1;
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, tmax, t0);
    }
}

// Launchers: one block per weight row (grid.x = od, grid.y = 1) — the
// token loop is in-block, so the launch grid no longer multiplies weight
// traffic by nt.
extern "C" {
void launch_q4_k_q8_mmvq_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q4_k_q8_mmvq_multi", "q4_k_q8_mmvq_multi");
    q4_k_q8_mmvq_multi<<<grid, minfer_launch_block("launch:q4_k_q8_mmvq_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q4_k_q8_mmvq_multi", "q4_k_q8_mmvq_multi");
}

void launch_q4_k_q8_mmvq_v2_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q4_k_q8_mmvq_v2_multi", "q4_k_q8_mmvq_v2_multi");
    q4_k_q8_mmvq_v2_multi<<<grid, minfer_launch_block("launch:q4_k_q8_mmvq_v2_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q4_k_q8_mmvq_v2_multi", "q4_k_q8_mmvq_v2_multi");
}

void launch_q5_k_q8_mmvq_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q5_k_q8_mmvq_multi", "q5_k_q8_mmvq_multi");
    q5_k_q8_mmvq_multi<<<grid, minfer_launch_block("launch:q5_k_q8_mmvq_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q5_k_q8_mmvq_multi", "q5_k_q8_mmvq_multi");
}

void launch_q5_k_q8_mmvq_v2_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q5_k_q8_mmvq_v2_multi", "q5_k_q8_mmvq_v2_multi");
    q5_k_q8_mmvq_v2_multi<<<grid, minfer_launch_block("launch:q5_k_q8_mmvq_v2_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q5_k_q8_mmvq_v2_multi", "q5_k_q8_mmvq_v2_multi");
}

void launch_q6_k_q8_mmvq_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int blk_stride, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q6_k_q8_mmvq_multi", "q6_k_q8_mmvq_multi");
    q6_k_q8_mmvq_multi<<<grid, minfer_launch_block("launch:q6_k_q8_mmvq_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt, blk_stride);
    minfer_launch_ok("launch:q6_k_q8_mmvq_multi", "q6_k_q8_mmvq_multi");
}

void launch_q6_k_q8_mmvq_v2_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int blk_stride, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q6_k_q8_mmvq_v2_multi", "q6_k_q8_mmvq_v2_multi");
    q6_k_q8_mmvq_v2_multi<<<grid, minfer_launch_block("launch:q6_k_q8_mmvq_v2_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt, blk_stride);
    minfer_launch_ok("launch:q6_k_q8_mmvq_v2_multi", "q6_k_q8_mmvq_v2_multi");
}
}

// === doc 103: q4_0 / q8_0 decode MMVQ =====================================
// The 8e MMVQ structure (one row per 256-thread block, 32-element units
// round-robin across lanes, dp4a over the shared pad40 q8 activation plane)
// applied to the two legacy f32-activation types. NEW CODE ONLY — every
// landed kernel/dispatch (q4_K/q5_K/q6_K MMVQ, raw-BT) is untouched; the
// dispatch arms below the new branches keep the f32 kernels verbatim.
// Weight strides are 18 B (q4_0) / 34 B (q8_0): even but not 4-aligned, so
// the payload reads use the q6_K 2-byte-half pattern (get_int_b2 style).
// nt-invariance: the multi variant is the nt=1 kernel with the per-u
// accumulation order and the block reduction kept bitwise (Step 82 rule),
// which is what the doc 93/94 identity contract needs.

__global__ void __launch_bounds__(256) q4_0_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nb = id >> 5; // dispatch gate: id % 32 == 0
    const int row_stride = nb * Q4B;
    const uint8_t* x8row = acts8 + (size_t)t * nb * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nb; u += 256) {
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)u * Q4B;
        const float d4 = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const uint8_t* x8b = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8b + 4);
        int dot = 0, sx = 0;
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            // 18-B stride: payload is 2B-aligned only — two u16 halves/word
            const uint32_t w =
                (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v) |
                ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v + 2) << 16);
            const uint32_t lo = w & 0x0F0F0F0F;           // elements 4v..4v+3
            const uint32_t hi = (w >> 4) & 0x0F0F0F0F;    // elements 16+4v..
            dot = __dp4a((int)lo, (int)xw[v], dot);
            dot = __dp4a((int)hi, (int)xw[v + 4], dot);
            sx  = __dp4a(0x01010101, (int)xw[v], sx);
            sx  = __dp4a(0x01010101, (int)xw[v + 4], sx);
        }
        // q4_0 value = (nibble - 8) * d  →  Σ = d * (dot - 8 * sx)
        acc += d8 * d4 * (float)(dot - 8 * sx);
    }
    mmvq_block_reduce(acc, output, od, t);
}

__global__ void __launch_bounds__(256) q8_0_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nb = id >> 5;
    const int row_stride = nb * Q8B;
    const uint8_t* x8row = acts8 + (size_t)t * nb * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nb; u += 256) {
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)u * Q8B;
        const float d4 = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const uint8_t* x8b = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8b + 4);
        int dot = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            // 34-B stride: 2B-aligned payload, two u16 halves per word
            const uint32_t w =
                (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v) |
                ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v + 2) << 16);
            dot = __dp4a((int)w, (int)xw[v], dot);
        }
        acc += d8 * d4 * (float)dot;
    }
    mmvq_block_reduce(acc, output, od, t);
}

__global__ void __launch_bounds__(256) q4_0_q8_mmvq_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int nb = id >> 5;
    const int row_stride = nb * Q4B;
    const int ngrp = (nt + 7) >> 3;
    for (int g = 0; g < ngrp; ++g) {
    const int t0 = g << 3;
    const int tmax = min(8, nt - t0);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = threadIdx.x; u < nb; u += 256) {
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)u * Q4B;
        const float d4 = h2f(*reinterpret_cast<const uint16_t*>(blk));
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < tmax) {
                const uint8_t* x8b = acts8 + ((size_t)(t0 + t) * nb + (size_t)u) * Q8PB;
                const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8b));
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8b + 4);
                int dot = 0, sx = 0;
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    const uint32_t w =
                        (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v) |
                        ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v + 2) << 16);
                    const uint32_t lo = w & 0x0F0F0F0F;
                    const uint32_t hi = (w >> 4) & 0x0F0F0F0F;
                    dot = __dp4a((int)lo, (int)xw[v], dot);
                    dot = __dp4a((int)hi, (int)xw[v + 4], dot);
                    sx  = __dp4a(0x01010101, (int)xw[v], sx);
                    sx  = __dp4a(0x01010101, (int)xw[v + 4], sx);
                }
                acc[t] += d8 * d4 * (float)(dot - 8 * sx);
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, tmax, t0);
    }
}

__global__ void __launch_bounds__(256) q8_0_q8_mmvq_multi(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int nb = id >> 5;
    const int row_stride = nb * Q8B;
    const int ngrp = (nt + 7) >> 3;
    for (int g = 0; g < ngrp; ++g) {
    const int t0 = g << 3;
    const int tmax = min(8, nt - t0);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = threadIdx.x; u < nb; u += 256) {
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)u * Q8B;
        const float d4 = h2f(*reinterpret_cast<const uint16_t*>(blk));
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < tmax) {
                const uint8_t* x8b = acts8 + ((size_t)(t0 + t) * nb + (size_t)u) * Q8PB;
                const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8b));
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8b + 4);
                int dot = 0;
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const uint32_t w =
                        (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v) |
                        ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v + 2) << 16);
                    dot = __dp4a((int)w, (int)xw[v], dot);
                }
                acc[t] += d8 * d4 * (float)dot;
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, tmax, t0);
    }
}

extern "C" void launch_q4_0_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q4_0_q8_mmvq", "q4_0_q8_mmvq");
    q4_0_q8_mmvq<<<grid, minfer_launch_block("launch:q4_0_q8_mmvq", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q4_0_q8_mmvq", "q4_0_q8_mmvq");
}

extern "C" void launch_q4_0_q8_mmvq_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q4_0_q8_mmvq_multi", "q4_0_q8_mmvq_multi");
    q4_0_q8_mmvq_multi<<<grid, minfer_launch_block("launch:q4_0_q8_mmvq_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q4_0_q8_mmvq_multi", "q4_0_q8_mmvq_multi");
}

extern "C" void launch_q8_0_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q8_0_q8_mmvq", "q8_0_q8_mmvq");
    q8_0_q8_mmvq<<<grid, minfer_launch_block("launch:q8_0_q8_mmvq", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q8_0_q8_mmvq", "q8_0_q8_mmvq");
}

extern "C" void launch_q8_0_q8_mmvq_multi(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q8_0_q8_mmvq_multi", "q8_0_q8_mmvq_multi");
    q8_0_q8_mmvq_multi<<<grid, minfer_launch_block("launch:q8_0_q8_mmvq_multi", 256), 0, stream>>>(weights, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q8_0_q8_mmvq_multi", "q8_0_q8_mmvq_multi");
}

// === doc 104: q8_0 p32 split-plane decode MMVQ ============================
// Doc 103's raw-34B kernel costs ~2x the L1TEX wavefront work per weight
// byte of the q4_0 kernel (16 two-byte loads per block at a 34-byte lane
// stride, each instruction scattering across ~34 32B sectors). Measured
// (cold-L2 rotating microbench, 7B shapes): the raw kernel runs 235-250
// GB/s while this split-plane variant runs 260-266 — closing to the doc-99
// probe ceiling — and its output is BYTE-EQUAL to the raw kernel (same
// int8 values, same dp4a order, same reduction): pure load-pattern change.
// Layout: payload plane 32 B/block (16B-aligned, uint4 x2 loads) + dense
// d plane 2 B/block (one u16). Total = the raw 34 B/block; the raw
// registration stays untouched for the f32 fallback (memory +~94% for the
// planes, MINFER_NO_Q80_P32=1 reverts both the planes and the dispatch).
__global__ void __launch_bounds__(256) q8_0_p32_q8_mmvq(
    const uint8_t* __restrict__ planeP,
    const uint8_t* __restrict__ planeD,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nb = id >> 5;
    const uint4* prow = reinterpret_cast<const uint4*>(planeP + (size_t)row * nb * 32);
    const uint16_t* drow = reinterpret_cast<const uint16_t*>(planeD + (size_t)row * nb * 2);
    const uint8_t* x8row = acts8 + (size_t)t * nb * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nb; u += 256) {
        const uint4 p0 = __ldg(&prow[2 * u]);
        const uint4 p1 = __ldg(&prow[2 * u + 1]);
        const float d4 = h2f(drow[u]);
        const uint8_t* x8b = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8b + 4);
        const uint32_t w0[4] = {p0.x, p0.y, p0.z, p0.w};
        const uint32_t w1[4] = {p1.x, p1.y, p1.z, p1.w};
        int dot = 0;
        #pragma unroll
        for (int v = 0; v < 4; v++) dot = __dp4a((int)w0[v], (int)xw[v], dot);
        #pragma unroll
        for (int v = 0; v < 4; v++) dot = __dp4a((int)w1[v], (int)xw[v + 4], dot);
        acc += d8 * d4 * (float)dot;
    }
    mmvq_block_reduce(acc, output, od, t);
}

// Multi-token variant: the weight words are loaded ONCE per unit (hoisted
// out of the token loop — doc 103's multi re-read them per token from L1)
// and the per-token accumulation order matches the nt == 1 kernel exactly
// (Step 82 bitwise rule, verified by the doc 93/94 identity battery).
__global__ void __launch_bounds__(256) q8_0_p32_q8_mmvq_multi(
    const uint8_t* __restrict__ planeP,
    const uint8_t* __restrict__ planeD,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int nb = id >> 5;
    const uint4* prow = reinterpret_cast<const uint4*>(planeP + (size_t)row * nb * 32);
    const uint16_t* drow = reinterpret_cast<const uint16_t*>(planeD + (size_t)row * nb * 2);
    const int ngrp = (nt + 7) >> 3;
    for (int g = 0; g < ngrp; ++g) {
    const int t0 = g << 3;
    const int tmax = min(8, nt - t0);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = threadIdx.x; u < nb; u += 256) {
        const uint4 p0 = __ldg(&prow[2 * u]);
        const uint4 p1 = __ldg(&prow[2 * u + 1]);
        const float d4 = h2f(drow[u]);
        const uint32_t w0[4] = {p0.x, p0.y, p0.z, p0.w};
        const uint32_t w1[4] = {p1.x, p1.y, p1.z, p1.w};
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < tmax) {
                const uint8_t* x8b = acts8 + ((size_t)(t0 + t) * nb + (size_t)u) * Q8PB;
                const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8b));
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8b + 4);
                int dot = 0;
                #pragma unroll
                for (int v = 0; v < 4; v++) dot = __dp4a((int)w0[v], (int)xw[v], dot);
                #pragma unroll
                for (int v = 0; v < 4; v++) dot = __dp4a((int)w1[v], (int)xw[v + 4], dot);
                acc[t] += d8 * d4 * (float)dot;
            }
        }
    }
    mmvq_block_reduce_multi(acc, output, od, tmax, t0);
    }
}

extern "C" void launch_q8_0_p32_q8_mmvq(
    const uint8_t* planeP, const uint8_t* planeD, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q8_0_p32_q8_mmvq", "q8_0_p32_q8_mmvq");
    q8_0_p32_q8_mmvq<<<grid, minfer_launch_block("launch:q8_0_p32_q8_mmvq", 256), 0, stream>>>(planeP, planeD, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q8_0_p32_q8_mmvq", "q8_0_p32_q8_mmvq");
}

extern "C" void launch_q8_0_p32_q8_mmvq_multi(
    const uint8_t* planeP, const uint8_t* planeD, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, 1);
    minfer_launch_prelude("launch:q8_0_p32_q8_mmvq_multi", "q8_0_p32_q8_mmvq_multi");
    q8_0_p32_q8_mmvq_multi<<<grid, minfer_launch_block("launch:q8_0_p32_q8_mmvq_multi", 256), 0, stream>>>(planeP, planeD, acts8, output, od, id, nt);
    minfer_launch_ok("launch:q8_0_p32_q8_mmvq_multi", "q8_0_p32_q8_mmvq_multi");
}
