// src/cuda/kernels/mmq_int8.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"



// ─── R1: int8 MMQ prefill GEMM ────────────────────────────────────────────
// llama.cpp's MMQ math structure on minfer's 8p tile skeleton. Activations
// are quantized to q8_0 once per launch (pad40 blocks; the kernel writes the
// per-block int sum into the 4 slack bytes at offset 36), weights stay RAW —
// no f16 dequant pass, no w16 cache. A tiled mma.m16n8k32 (s8) GEMM
// accumulates one 32-k int chunk per step; the int C fragment is rescaled
// per (token, row, k-block) with the block-scale products and, for
// min-carrying types, a rank-1 offset term (weight min × activation block
// sum) — exactly llama.cpp's per-block correction, so results sit within
// f32 rounding of the CPU q8_0-activation dot path.
//
// K-quants keep their nibbles UNSIGNED in the int GEMM and carry the min
// term separately (llama.cpp's unpack_scales trick — the nibble grid is
// 0..15/0..31, so the "integer part" is non-negative and the scale/offset
// pair (d·s, −dmin·m) is applied per 32-k sub-block). q6_K's scales live on
// 16-element sub-blocks: each 32-k chunk spans two of them, so the chunk
// runs as TWO m16n8k16 mmas (low/high k halves) with separate int
// accumulators, rescaled by (d·sc0, d·sc1) — k32 staging/sync cadence, full
// mma throughput (a per-16 chunk loop measured 4× slower end-to-end: q6_K
// carries the 7B model's ffn_down + lm_head).
//
// Fragment layouts follow the PTX ISA mma.m16n8k32/m16n8k16 row.col
// documentation (A: row = groupID, k = tig·4 quads; B: col = groupID;
// C: row = groupID + 8·(l>>1), col = tig·2 + (l&1) — the C mapping is the
// production-proven llama.cpp tile_C get_i/get_j).
//
// Block tile 64 tokens × 64 rows, warp tile 32 tokens × 16 rows (the 8m/8p
// wm/wn warp mapping — consecutive blocks share one od-tile's weight panel
// in L2). Double-buffered shared tiles, synchronous staging (cp.async can't
// unpack quantized bytes): per k-chunk the A side moves 40B/token of q8 and
// the B side 16–34B/row of raw weight bytes — 2–4× less DRAM traffic than
// the f16 cache path, from the tensor-core int8 pipe. B staging uses two
// threads per row (one 16B half each) so consecutive threads read
// consecutive bytes instead of striding across rows.

                // ~94KB smem/block, llama.cpp ITER_K-style; measured faster than
                // KD=4 (2 blocks/SM) under load — revisit on a quiet GPU

// A-side q8 staging: thread t handles token row t>>2 (words t&3 and 4+(t&3)).
__device__ __forceinline__ void mmq_stage_a(
    const uint8_t* __restrict__ q8x, int i0, int nt, int nb32, int kb,
    int* qa, float* da, int* sa
) {
    int r = threadIdx.x >> 2, q = threadIdx.x & 3;
    int tok = i0 + r;
    int w0 = 0, w1 = 0; float d = 0.0f; int s = 0;
    if (tok < nt) {
        const uint8_t* blk = q8x + ((size_t)tok * nb32 + kb) * 40;
        w0 = *(const uint32_t*)(blk + 4 + 4 * q);
        w1 = *(const uint32_t*)(blk + 20 + 4 * q);
        if (q == 0) {
            d = h2f(*(const uint16_t*)blk);
            s = (int)*(const uint32_t*)(blk + 36);
        }
    }
    qa[r * MMQ_WS + q] = w0;
    qa[r * MMQ_WS + 4 + q] = w1;
    if (q == 0) { da[r] = d; sa[r] = s; }
}

// B-side raw-weight staging: 128 threads → 2 per weight row (thread half
// t&1 covers words 4·half..4·half+3 of the 32-k chunk); q8_0 uses 4 threads
// per row. Headers (scale/offset) come from the half-0 threads. Type math
// mirrors the bqa_* dequantizers / MMVQ kernels exactly.
template <int TYPE>
__device__ __forceinline__ void mmq_stage_b(
    const uint8_t* __restrict__ W, int j0, int od, int nb32, int c, int bstride,
    int* qb, float* ds, float* dm, float* ds1
) {
    int t = threadIdx.x;
    int r = t >> 1, half = t & 1;          // 2 threads per row (q8_0: 4)
    if constexpr (TYPE == 0) {
        r = t >> 2, half = t & 3;
    } else {
        if (t >= 128) return;              // 2 threads × 64 rows
    }
    int j = j0 + r;
    if (j >= od) {
        if constexpr (TYPE == 0) {
            #pragma unroll
            for (int w = half; w < 8; w += 4) qb[r * MMQ_WS + w] = 0;
            if (half == 0) { ds[r] = 0.0f; }
        } else {
            #pragma unroll
            for (int w = 4 * half; w < 4 * half + 4; w++) qb[r * MMQ_WS + w] = 0;
            if (half == 0) { ds[r] = 0.0f; dm[r] = 0.0f; if (TYPE == 7) ds1[r] = 0.0f; }
        }
        return;
    }
    if constexpr (TYPE == 0) {          // q8_0: 34B blocks, signed payload
        const uint8_t* blk = W + (size_t)j * (nb32 * 34) + (size_t)c * 34;
        if (half == 0) ds[r] = h2f(*(const uint16_t*)blk);
        #pragma unroll
        for (int w = half; w < 8; w += 4)
            qb[r * MMQ_WS + w] =
                (int)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * w)
                | ((int)*reinterpret_cast<const uint16_t*>(blk + 4 + 4 * w) << 16);
    } else if constexpr (TYPE == 1) {   // q4_0: value = d·(nib − 8)
        const uint8_t* blk = W + (size_t)j * (nb32 * 18) + (size_t)c * 18;
        if (half == 0) ds[r] = h2f(*(const uint16_t*)blk);
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            qb[r * MMQ_WS + w] = __vsubss4((int)nib, 0x08080808);
        }
    } else if constexpr (TYPE == 2) {   // q4_1: value = d·nib + m
        const uint8_t* blk = W + (size_t)j * (nb32 * 20) + (size_t)c * 20;
        if (half == 0) {
            ds[r] = h2f(*(const uint16_t*)blk);
            dm[r] = h2f(*(const uint16_t*)(blk + 2));
        }
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            qb[r * MMQ_WS + w] = (int)nib;
        }
    } else if constexpr (TYPE == 3) {   // q5_0: value = d·(nib + 16·bit − 16)
        const uint8_t* blk = W + (size_t)j * (nb32 * 22) + (size_t)c * 22;
        uint32_t qh;
        if (half == 0) {
            ds[r] = h2f(*(const uint16_t*)blk);
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2);
        } else {
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4);
        }
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 8 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            uint32_t hi = ((((qh >> (4 * wb + 0)) & 1u) << 0) | (((qh >> (4 * wb + 1)) & 1u) << 8)
                        | (((qh >> (4 * wb + 2)) & 1u) << 16) | (((qh >> (4 * wb + 3)) & 1u) << 24)) << 4;
            qb[r * MMQ_WS + w] = __vsubss4((int)(nib | hi), 0x10101010);
        }
    } else if constexpr (TYPE == 4) {   // q5_1: value = d·(nib + 16·bit) + m
        const uint8_t* blk = W + (size_t)j * (nb32 * 24) + (size_t)c * 24;
        uint32_t qh;
        if (half == 0) {
            ds[r] = h2f(*(const uint16_t*)blk);
            dm[r] = h2f(*(const uint16_t*)(blk + 2));
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4);
        } else {
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6);
        }
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 8 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 10 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            uint32_t hi = ((((qh >> (4 * wb + 0)) & 1u) << 0) | (((qh >> (4 * wb + 1)) & 1u) << 8)
                        | (((qh >> (4 * wb + 2)) & 1u) << 16) | (((qh >> (4 * wb + 3)) & 1u) << 24)) << 4;
            qb[r * MMQ_WS + w] = (int)(nib | hi);
        }
    } else if constexpr (TYPE == 5) {   // q4_K: value = d·s·nib − dmin·m (nib unsigned)
        int sb = c >> 3, s = c & 7;
        const uint8_t* blk = W + (size_t)j * ((nb32 >> 3) * 144) + (size_t)sb * 144;
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            uint32_t N = *(const uint32_t*)(blk + 16 + (s >> 1) * 32 + 4 * w);
            uint32_t nib = (s & 1) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            qb[r * MMQ_WS + w] = (int)nib;
        }
        if (half == 0) {
            uint8_t sc, m;
            get_scale_min_k4(s, blk + 4, &sc, &m);
            ds[r] = h2f(*(const uint16_t*)blk) * (float)sc;
            dm[r] = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);
        }
    } else if constexpr (TYPE == 6) {   // q5_K: q5 high-bit plane over the q4_K layout
        int sb = c >> 3, s = c & 7;
        const uint8_t* blk = W + (size_t)j * ((nb32 >> 3) * 176) + (size_t)sb * 176;
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            uint32_t N = *(const uint32_t*)(blk + 48 + (s >> 1) * 32 + 4 * w);
            uint32_t nib = (s & 1) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            uint32_t QH = *(const uint32_t*)(blk + 16 + 4 * w);
            uint32_t hi = ((QH >> s) & 0x01010101u) << 4;
            qb[r * MMQ_WS + w] = (int)(nib | hi);
        }
        if (half == 0) {
            uint8_t sc, m;
            get_scale_min_k4(s, blk + 4, &sc, &m);
            ds[r] = h2f(*(const uint16_t*)blk) * (float)sc;
            dm[r] = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);
        }
    } else {                            // q6_K: value = d·sc·(q6 − 32), 16-elem sub-scales
        // one 32-k chunk = two subs: s0 = 2c (words 0..3), s1 = 2c+1 (words 4..7);
        // s wraps at 16 into the next super-block (sb = c>>3)
        int sb = c >> 3;
        const uint8_t* blk = W + (size_t)j * ((nb32 >> 3) * bstride) + (size_t)sb * bstride;
        if (half == 0) {
            float d = h2f(*(const uint16_t*)(blk + 208));
            ds[r]  = d * (float)(int8_t)blk[192 + (2 * c) % 16];
            ds1[r] = d * (float)(int8_t)blk[192 + (2 * c + 1) % 16];
        }
        int s = (2 * c + half) % 16;
        int chunk = s >> 3, g = (s >> 1) & 3, is = s & 1;
        const uint8_t* ql = blk + chunk * 64 + (g & 1) * 32 + is * 16;
        const uint8_t* qh = blk + 128 + chunk * 32 + is * 16;
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t QL = (uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * wb)
                       | ((uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * wb + 2) << 16);
            uint32_t QH = (uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * wb)
                       | ((uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * wb + 2) << 16);
            uint32_t nib = (g < 2) ? (QL & 0x0F0F0F0Fu) : ((QL >> 4) & 0x0F0F0F0Fu);
            uint32_t hi = ((QH >> (2 * g)) & 0x03030303u) << 4;
            qb[r * MMQ_WS + w] = __vsubss4((int)(nib | hi), 0x20202020);
        }
    }
}

// TYPE: 0 q8_0, 1 q4_0, 2 q4_1, 3 q5_0, 4 q5_1, 5 q4_K, 6 q5_K, 7 q6_K.
// KSPLIT: 1 = one m16n8k32 per chunk (types 0-6); 2 = two m16n8k16 with
// separate int accumulators (q6_K's per-16 sub-scales). HAS_OFF = the
// min-carrying types (rank-1 offset term in the rescale).
template <int TYPE, int KSPLIT, bool HAS_OFF>
__global__ void __launch_bounds__(256) mmq_nt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id, int bstride
) {
#if __CUDA_ARCH__ >= 800
    // mma.m16n8k32 (s8) needs sm_80+; sm_75 keeps the f16 w16-cache path.
    // Dynamic shared (llama.cpp-style 256-k staging depth): staging ONE
    // 32-k chunk per sync exposed the full global-load latency every
    // chunk — 2.5 TMAC/s on GB10. 8 chunks per buffer amortize it 8×.
    extern __shared__ int mmq_sh[];
    int* qa = mmq_sh;                                   // [2][KD][BI*WS]
    int* qb = qa + 2 * MMQ_KD * (MMQ_BI * MMQ_WS);      // [2][KD][BJ*WS]
    int* ssa = qb + 2 * MMQ_KD * (MMQ_BJ * MMQ_WS);     // [2][KD][BI]
    float* sda = reinterpret_cast<float*>(ssa + 2 * MMQ_KD * MMQ_BI);
    float* sds = sda + 2 * MMQ_KD * MMQ_BI;             // [2][KD][BJ]
    float* sds1 = sds + 2 * MMQ_KD * MMQ_BI;            // q6_K: second 16-sub
    float* sdm = sds1 + 2 * MMQ_KD * MMQ_BI;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1;   // same warp mapping as 8m/8p
    const int i0 = blockIdx.x * MMQ_BI;
    const int j0 = blockIdx.y * MMQ_BJ;
    const int i0w = wn * 32;
    const int j0w = wm * 16;
    const int nb32 = id >> 5;
    const int nchunk = id >> 5;
    const int nktile = (nchunk + MMQ_KD - 1) / MMQ_KD;

    float sum[16] = {0.0f};   // [nh][h][l]: 2 B-frags × 2 A-frags × 4 C regs

#define MMQ_STAGE_TILE(kt, b)                                                   \
    for (int kd = 0; kd < MMQ_KD; kd++) {                                       \
        const int c = (kt) * MMQ_KD + kd;                                       \
        if (c < nchunk) {                                                       \
            mmq_stage_a(q8x, i0, nt, nb32, c,                                   \
                        qa + ((b) * MMQ_KD + kd) * (MMQ_BI * MMQ_WS),           \
                        sda + ((b) * MMQ_KD + kd) * MMQ_BI,                     \
                        ssa + ((b) * MMQ_KD + kd) * MMQ_BI);                    \
            mmq_stage_b<TYPE>(W, j0, od, nb32, c, bstride,                      \
                              qb + ((b) * MMQ_KD + kd) * (MMQ_BJ * MMQ_WS),     \
                              sds + ((b) * MMQ_KD + kd) * MMQ_BJ,               \
                              sdm + ((b) * MMQ_KD + kd) * MMQ_BJ,               \
                              sds1 + ((b) * MMQ_KD + kd) * MMQ_BJ);             \
        }                                                                       \
    }

    MMQ_STAGE_TILE(0, 0)
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
        if (kt + 1 < nktile) MMQ_STAGE_TILE(kt + 1, buf ^ 1)

        for (int kd = 0; kd < MMQ_KD; kd++) {
            const int c = kt * MMQ_KD + kd;
            if (c >= nchunk) break;
            const int* qat = qa + (buf * MMQ_KD + kd) * (MMQ_BI * MMQ_WS);
            const int* qbt = qb + (buf * MMQ_KD + kd) * (MMQ_BJ * MMQ_WS);
            const float* sdat = sda + (buf * MMQ_KD + kd) * MMQ_BI;
            const int* ssat = ssa + (buf * MMQ_KD + kd) * MMQ_BI;
            const float* sdst = sds + (buf * MMQ_KD + kd) * MMQ_BJ;
            const float* sds1t = sds1 + (buf * MMQ_KD + kd) * MMQ_BJ;
            const float* sdmt = sdm + (buf * MMQ_KD + kd) * MMQ_BJ;

            // A fragments: 2× 16-token halves; B fragments: 2× 8-row halves.
            // Fragment word layout is the k32 one for every type; KSPLIT=2
            // only splits the mma into low/high k halves (words 0..3 / 4..7).
            int a[2][4], b[2][2];
            int clow[2][2][4], chigh[2][2][4];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r0 = (i0w + h * 16 + (lane >> 2)) * MMQ_WS + (lane & 3);
                const int r1 = (i0w + h * 16 + 8 + (lane >> 2)) * MMQ_WS + (lane & 3);
                a[h][0] = qat[r0];
                a[h][1] = qat[r1];
                a[h][2] = qat[r0 + 4];
                a[h][3] = qat[r1 + 4];
            }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const int rb = (j0w + nh * 8 + (lane >> 2)) * MMQ_WS + (lane & 3);
                b[nh][0] = qbt[rb];
                b[nh][1] = qbt[rb + 4];
            }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) { clow[nh][h][l] = 0; chigh[nh][h][l] = 0; }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    if constexpr (KSPLIT == 1) {
                        mmq_mma_k32(clow[nh][h], a[h], b[nh]);
                    } else {
                        mmq_mma_k16(clow[nh][h], a[h], b[nh][0]);
                        mmq_mma_k16(chigh[nh][h], a[h] + 2, b[nh][1]);
                    }
                }

            // per-(token, row, k-block) rescale: value = ds·int (+ dm·sa), all × da
            const float da_q[4] = {
                sdat[i0w +  lane / 4], sdat[i0w + 8 + lane / 4],
                sdat[i0w + 16 + lane / 4], sdat[i0w + 24 + lane / 4],
            };
            const float sa_q[4] = {
                (float)ssat[i0w +  lane / 4], (float)ssat[i0w + 8 + lane / 4],
                (float)ssat[i0w + 16 + lane / 4], (float)ssat[i0w + 24 + lane / 4],
            };
            float dsv[2][8], dsv1[2][8], dmv[2][8];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) {
                    dsv[nh][jj] = sdst[j0w + nh * 8 + jj];
                    dsv1[nh][jj] = sds1t[j0w + nh * 8 + jj];
                    dmv[nh][jj] = sdmt[j0w + nh * 8 + jj];
                }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        // C element l: row = (l>>1)*8 + lane/4, col = (lane&3)*2 + (l&1)
                        const float da = da_q[h * 2 + (l >> 1)];
                        const float sa = sa_q[h * 2 + (l >> 1)];
                        const int jj = (lane & 3) * 2 + (l & 1);
                        const int idx = nh * 8 + h * 4 + l;
                        if constexpr (KSPLIT == 1) {
                            sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
                            if (HAS_OFF) sum[idx] += da * dmv[nh][jj] * sa;
                        } else {
                            sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
                            sum[idx] += da * dsv1[nh][jj] * (float)chigh[nh][h][l];
                        }
                    }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = i0 + i0w + h * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od)
                    C[(size_t)i * od + j] = sum[nh * 8 + h * 4 + l];
            }
#endif // __CUDA_ARCH__ >= 800
}

extern "C" int cuda_shared_per_sm() {
    // T2 (plan R2 fix): query the CURRENT device, not a hardcoded 0 —
    // cudaSetDevice(best_device) ran at init, so this is the selected GPU.
    int dev = 0;
    cudaGetDevice(&dev);
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    return v;
}

extern "C" int cuda_shared_per_block_optin() {
    int dev = 0;
    cudaGetDevice(&dev);
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return v;
}

// T2 (plan §6.2): the BT tile config's dynamic-smem demand — single source of
// truth shared by launch_mmq_nt and the Rust-side feasibility check (the
// formula is owned by the kernel it serves; MMQ_BI/BJ/KD/WS are the tile
// constants at the top of this file).
static int mmq_dynamic_smem_bytes() {
    return (2 * MMQ_KD * (MMQ_BI * MMQ_WS) + 2 * MMQ_KD * (MMQ_BJ * MMQ_WS)
            + 2 * MMQ_KD * MMQ_BI + 4 * 2 * MMQ_KD * MMQ_BI)
           * 4;
}

extern "C" int cuda_mmq_smem_bytes() { return mmq_dynamic_smem_bytes(); }

// #147: returns 1 = launched and accepted, 0 = refused (the dynamic-smem
// opt-in failed or the launch errored; both are named at the site). The Rust
// caller turns a 0 into an `Err`.
extern "C" int launch_mmq_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, int q6_stride, cudaStream_t stream
) {
    dim3 grid((nt + 63) / 64, (od + 63) / 64);
    // dynamic shared: qa+qb tiles, sda/sds/sds1/sdm (float) + ssa (int)
    const int smem = mmq_dynamic_smem_bytes();
    int ok = 0;
#define MMQ_LAUNCH(NAME, KERN)                                                       \
    do {                                                                             \
        if (minfer_smem_optin("attr:mmq_nt", NAME,                                   \
                              reinterpret_cast<const void*>(&(KERN)), smem)) {       \
            minfer_launch_prelude("launch:mmq_nt", NAME);                            \
            (KERN)<<<grid, 256, minfer_launch_smem("launch:mmq_nt", smem), stream>>>( \
                w, q8, c, nt, od, id, q6_stride);                                    \
            ok = minfer_launch_ok("launch:mmq_nt", NAME) ? 1 : 0;                    \
        }                                                                            \
    } while (0)
    switch (type_id) {
        case 0: MMQ_LAUNCH("mmq_nt_kernel<0,1,false>", (mmq_nt_kernel<0, 1, false>)); break;
        case 1: MMQ_LAUNCH("mmq_nt_kernel<1,1,false>", (mmq_nt_kernel<1, 1, false>)); break;
        case 2: MMQ_LAUNCH("mmq_nt_kernel<2,1,true>", (mmq_nt_kernel<2, 1, true>)); break;
        case 3: MMQ_LAUNCH("mmq_nt_kernel<3,1,false>", (mmq_nt_kernel<3, 1, false>)); break;
        case 4: MMQ_LAUNCH("mmq_nt_kernel<4,1,true>", (mmq_nt_kernel<4, 1, true>)); break;
        case 5: MMQ_LAUNCH("mmq_nt_kernel<5,1,true>", (mmq_nt_kernel<5, 1, true>)); break;
        case 6: MMQ_LAUNCH("mmq_nt_kernel<6,1,true>", (mmq_nt_kernel<6, 1, true>)); break;
        default: MMQ_LAUNCH("mmq_nt_kernel<7,2,false>", (mmq_nt_kernel<7, 2, false>)); break;
    }
#undef MMQ_LAUNCH
    return ok;
}
