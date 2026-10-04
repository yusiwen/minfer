// src/cuda/kernels/mmq_raw.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"



// ─── P6: raw-byte MMQ (llama.cpp structure) — q4_K first ─────────────
// smem holds RAW quant bytes; staging is pure cp.async; dequant happens
// in registers inside the mma loop. Only the per-(row, chunk) scales are
// pre-computed at staging (shared by the whole warp; the C-fragment
// rescale needs all 8 fragment rows, and computing them per lane would
// be 4x redundant work in the hot loop).
//   A: pad40 chunk = d(2B) qs(32B @4) ssum(4B @36) — consumed as-is
//      (word w of the fragment = the raw int8 lane group k 4w..4w+3).
//   B: Q4KB=144B super-block per row per 256-k; nibbles unpacked at mma.
// Requires whole super-blocks: nb32 % 8 == 0 (launcher guard).
__device__ __forceinline__ void mmq_cp8(uint8_t* smem_dst, const uint8_t* gsrc,
                                        bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 8 : 0; // src-size 0 => zero-fill the 8B chunk
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}

template <int KDR>
__global__ void __launch_bounds__(256) mmq_raw_nt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id
) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_raw_sh[];
    uint8_t* qa8 = mmq_raw_sh;                                    // [2][KDR][BI][40]
    uint8_t* qb8 = qa8 + 2 * KDR * MMQ_BI * 40;                   // [2][BI][144]
    float* sds = reinterpret_cast<float*>(qb8 + 2 * MMQ_BI * 144);// [2][KDR][BI]
    float* sdm = sds + 2 * KDR * MMQ_BI;                          // [2][KDR][BI]

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1;   // same warp mapping as R1
    const int i0 = blockIdx.x * MMQ_BI;
    const int j0 = blockIdx.y * MMQ_BJ;
    const int i0w = wn * 32;
    const int j0w = wm * 16;
    const int nb32 = id >> 5;
    const int nchunk = nb32;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    float sum[16] = {0.0f};   // [nh][h][l]: 2 B-frags x 2 A-frags x 4 C regs

#define RAW_STAGE(kt, b)                                                       \
    do {                                                                       \
        /* A: (token, chunk) pad40 chunks as 5x 8B cp.async */                 \
        for (int x = threadIdx.x; x < MMQ_BI * KDR * 5; x += blockDim.x) {     \
            int kd = x / (MMQ_BI * 5);                                         \
            int rem = x % (MMQ_BI * 5);                                        \
            int r = rem / 5, u = rem % 5;                                      \
            int c = (kt) * KDR + kd;                                           \
            int tok = i0 + r;                                                  \
            const uint8_t* src =                                               \
                q8x + ((size_t)tok * nb32 + c) * 40 + u * 8;                   \
            uint8_t* dst = qa8 + ((size_t)(b) * KDR + kd) * MMQ_BI * 40        \
                         + (size_t)r * 40 + u * 8;                             \
            mmq_cp8(dst, src, tok < nt && c < nchunk);                         \
        }                                                                      \
        /* B: whole 144B super-block per row as 9x 16B cp.async */             \
        int sb = ((kt) * KDR) >> 3;                                            \
        for (int x = threadIdx.x; x < MMQ_BI * 9; x += blockDim.x) {           \
            int r = x / 9, u = x % 9;                                          \
            int j = j0 + r;                                                    \
            const uint8_t* src =                                               \
                W + (size_t)j * ((size_t)nsb * 144) + (size_t)sb * 144         \
                  + u * 16;                                                    \
            uint8_t* dst = qb8 + ((size_t)(b) * MMQ_BI + r) * 144 + u * 16;    \
            gemm_cp16((__half*)(void*)dst, (const __half*)(const void*)src,    \
                      j < od && sb < nsb);                                     \
        }                                                                      \
        /* per-(row, chunk) scales from global (a few B per unit, L2-hot) */   \
        for (int x = threadIdx.x; x < MMQ_BI * KDR; x += blockDim.x) {         \
            int kd = x / MMQ_BI, r = x % MMQ_BI;                               \
            int c = (kt) * KDR + kd;                                           \
            int j = j0 + r;                                                    \
            float dv = 0.0f, mv = 0.0f;                                        \
            if (j < od && c < nchunk) {                                        \
                int sg = c & 7;                                                \
                const uint8_t* blk =                                           \
                    W + (size_t)j * ((size_t)nsb * 144)                        \
                      + (size_t)(c >> 3) * 144;                                \
                uint8_t sc, m;                                                 \
                get_scale_min_k4(sg, blk + 4, &sc, &m);                        \
                dv = h2f(*(const uint16_t*)blk) * (float)sc;                   \
                mv = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);           \
            }                                                                  \
            sds[((b) * KDR + kd) * MMQ_BI + r] = dv;                           \
            sdm[((b) * KDR + kd) * MMQ_BI + r] = mv;                           \
        }                                                                      \
    } while (0)

    RAW_STAGE(0, 0);
    gemm_cp_commit();

    int buf = 0;
    for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
        if (kt + 1 < nktile) RAW_STAGE(kt + 1, buf ^ 1);
        gemm_cp_commit();
        gemm_cp_wait1();          // at most the prefetch group pending
        __syncthreads();          // scales (plain stores) + landed bytes visible

        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)(buf * KDR + kd) * MMQ_BI * 40;
            const float* sdst = sds + (buf * KDR + kd) * MMQ_BI;
            const float* sdmt = sdm + (buf * KDR + kd) * MMQ_BI;

            // A fragments: int8 lane words straight out of the raw chunk.
            int a[2][4], b[2][2];
            int clow[2][2][4], chigh[2][2][4];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r0 = i0w + h * 16 + (lane >> 2);
                const int r1 = r0 + 8;
                const uint8_t* p0 = qat + (size_t)r0 * 40 + 4;
                const uint8_t* p1 = qat + (size_t)r1 * 40 + 4;
                a[h][0] = *(const int*)(p0 + 4 * (lane & 3));
                a[h][1] = *(const int*)(p1 + 4 * (lane & 3));
                a[h][2] = *(const int*)(p0 + 4 * ((lane & 3) + 4));
                a[h][3] = *(const int*)(p1 + 4 * ((lane & 3) + 4));
            }
            // B fragments: unpack the raw nibbles in registers.
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const int jr = j0w + nh * 8 + (lane >> 2);
                const uint8_t* rb8 = qb8 + (size_t)(buf * MMQ_BI + jr) * 144;
                uint32_t n0 = *(const uint32_t*)(rb8 + 16 + (sg >> 1) * 32
                                                + 4 * (lane & 3));
                uint32_t n1 = *(const uint32_t*)(rb8 + 16 + (sg >> 1) * 32
                                                + 4 * ((lane & 3) + 4));
                b[nh][0] = (int)((sg & 1) ? ((n0 >> 4) & 0x0F0F0F0Fu)
                                          : (n0 & 0x0F0F0F0Fu));
                b[nh][1] = (int)((sg & 1) ? ((n1 >> 4) & 0x0F0F0F0Fu)
                                          : (n1 & 0x0F0F0F0Fu));
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
                for (int h = 0; h < 2; h++)
                    mmq_mma_k32(clow[nh][h], a[h], b[nh]);

            // rescale: identical math/layout to the R1 kernel; A-side
            // d/ssum come straight from the raw chunk.
            float da_q[4];
            int sa_q[4];
            #pragma unroll
            for (int t4 = 0; t4 < 4; t4++) {
                const uint8_t* at = qat + (size_t)(i0w + (lane >> 2) + t4 * 8) * 40;
                da_q[t4] = h2f(*(const uint16_t*)at);
                sa_q[t4] = (int)*(const uint32_t*)(at + 36);
            }
            // r15: the dmv correction term is rank-1 in (token, od-col) —
            // the row-side product da*sa is shared by the od-col pair of
            // each C fragment, so fold it once per row (4 FMUL/chunk)
            // instead of once per C value (8 FMUL/chunk). The dsv term and
            // the per-chunk scale application are unchanged.
            const float dma[4] = { da_q[0] * (float)sa_q[0],
                                   da_q[1] * (float)sa_q[1],
                                   da_q[2] * (float)sa_q[2],
                                   da_q[3] * (float)sa_q[3] };
            float dsv[2][8], dmv[2][8];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) {
                    dsv[nh][jj] = sdst[j0w + nh * 8 + jj];
                    dmv[nh][jj] = sdmt[j0w + nh * 8 + jj];
                }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        const float da = da_q[h * 2 + (l >> 1)];
                        const int jj = (lane & 3) * 2 + (l & 1);
                        const int idx = nh * 8 + h * 4 + l;
                        sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
                        sum[idx] += dma[h * 2 + (l >> 1)] * dmv[nh][jj];
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

template <int KDR>
// (wide clone: 128-token x 128-od block tile; each warp owns 16 od-rows and
// issues 16 independent mma chains per 32-k chunk = 2 B-frags (8 od-rows
// each) x 8 A-frags (16 tokens each) — llama.cpp MMQ accumulator depth.)
#undef RAW_STAGE
__global__ void __launch_bounds__(256) mmq_raw_wide_nt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id
) {
#define MMQ_WBI 128
#define MMQ_WBJ 128
#define MMQ_WBQ 48  // padded per-(sg,row) qb8 slot: 16B-aligned, 12r mod 32
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_raw_sh[];
    // Single-buffer sync-staged layout — KD=8 totals 98,304B (1 block/SM;
    // KD=4 is 73,728B; resident warps hide the staging latency):
    //   qa8   [KDR][128] x 32B  chunk qs only (d/ssum in sda_q). r22: the
    //                         16B granules are XOR-swizzled inside 128B
    //                         super-rows (4 rows each): granule
    //                         ((row&3)*2 + h) ^ ((row>>2)&7) — the 32B row
    //                         stride puts ldmatrix rows 4 apart on the same
    //                         bank phase (2-way conflict); the swizzle gives
    //                         every ldmatrix phase 8 distinct phases, zero
    //                         smem growth.
    //   sda_q [KDR][128] x 8B   (d f16 | ssum i16) packed, uint2-tiling
    //                           [KDR][16 g][8 q]: one LDS.64 serves the
    //                           token pair (t, t+8) a C fragment needs
    //   qb8   [8][128][48]      B sub-blocks pre-expanded to per-k int8,
    //                           SLOT-MAJOR (sg-major); 48B row stride puts
    //                           every ldmatrix row on a distinct bank phase
    //                           (raw nibbles 0..15; 128 od-rows per tile)
    //   sds   [KDR][128] float2 (d | dmin*m): one float4 load serves the
    //                           (j, j+1) od-col pair per minitile
    uint8_t* qa8 = mmq_raw_sh;
    uint32_t* sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_WBI * 32);
    uint8_t* qb8 = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_WBI * 2);
    float2* sds = reinterpret_cast<float2*>(qb8 + 8 * MMQ_WBJ * MMQ_WBQ);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Each warp owns a private 16-od-row slice and reads the FULL 128-token
    // tile: B fragments become warp-exclusive, A fragments are warp-shared.
    const int i0 = blockIdx.x * MMQ_WBI;
    const int j0 = blockIdx.y * MMQ_WBJ;
    const int j0w = warp * 16;
    const int nb32 = id >> 5;
    const int nchunk = nb32;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    float sum[64] = {0.0f};   // [g][nh][l]: 8 A-frags x 2 B-frags x 4 C regs

#define RAW_STAGE(kt)                                                          \
    do {                                                                       \
        /* llama.cpp-style synchronous staging, single buffer: plain global    \
         * -> smem loads, one syncthreads orders them. At 128x64 tiles the     \
         * smem is small enough for 2 blocks/SM - latency hiding comes from    \
         * occupancy, not prefetch depth. */                                   \
        /* r20: split-phase A staging. The old interleaved LDG->STS chains    \
         * stalled the warp at the first store of every 4-deep unroll batch   \
         * (PC-sampled: the leading STS held 16.7% of all warp stalls = one   \
         * full memory latency per batch, ~4 batches per kt). Issue ALL       \
         * global loads into registers first - one deep independent LDG batch \
         * per warp per kt - then store to smem. Identical addresses, traffic \
         * and instruction count; only the dependency schedule changes. */    \
        /* r22: r20's split-phase A staging is kept verbatim; only the qa8    \
         * store address gained the XOR swizzle. The d/ssum stream fold      \
         * (single 9-word-per-chunk pass) was tried and REVERTED: the old    \
         * scattered d/ssum loads are L1 hits (the qs pass of the same       \
         * chunks has the lines resident), so folding saves little sector    \
         * traffic while the flat enumeration costs ALU + branchy batches    \
         * (-19% wall, see docs r22). */                                     \
        {                                                                      \
            unsigned av[KDR * 4];          /* qs words: 128*KDR*8 / 256 thr */ \
            unsigned short dv[KDR / 2];    /* d f16 words: 128*KDR / 256 */    \
            unsigned sv[KDR / 2];          /* ssum words */                    \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 4; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),             \
                          kd = x >> 10;    /* x/(8*MMQ_WBI), 8*128 = 1024 */   \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned v = 0;                                                \
                if (tok < nt && c < nchunk)                                    \
                    v = *(const unsigned*)(q8x                                 \
                        + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4);          \
                av[i] = v;                                                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR / 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_WBI - 1), kd = x >> 7;  /* x/MMQ_WBI */ \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned short d16 = 0;                                        \
                unsigned ss = 0;                                               \
                if (tok < nt && c < nchunk) {                                  \
                    const uint8_t* src = q8x                                   \
                        + ((size_t)tok * nb32 + c) * 40;                       \
                    d16 = *(const unsigned short*)src;                         \
                    ss = (unsigned)(short)*(const int*)(src + 36);             \
                }                                                              \
                dv[i] = d16;                                                   \
                sv[i] = ss;                                                    \
            }                                                                              _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 4; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),             \
                          kd = x >> 10;                                        \
                const int R = kd * MMQ_WBI + r;                                \
                *(unsigned*)(qa8 + (size_t)(R & ~3) * 32                       \
                    + (size_t)(((((R & 3) << 1) + (u >> 2))                    \
                                ^ ((R >> 2) & 7)) << 4)                        \
                    + (size_t)(u & 3) * 4) = av[i];                            \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR / 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_WBI - 1), kd = x >> 7;                  \
                *(unsigned*)(sda_q + ((size_t)kd * MMQ_WBI + (r >> 4) * 8      \
                                      + (r & 7)) * 2 + ((r >> 3) & 1)) =       \
                    dv[i] | (sv[i] << 16);                                     \
            }                                                                  \
        }                                                                      \
        /* B: expand ONE 256-k super-block to per-k int8 AT STAGING - raw     \
         * nibble values 0..15 (folding (nib - m) here would skew the dmin    \
         * term, which the epilogue scales by dmin*m, not d*sc; the two-term  \
         * dsv/dmv rescale stays). Pair p covers sub-blocks 2p (low           \
         * nibbles) and 2p+1 (high nibbles) over qs bytes p*32..p*32+31;      \
         * slot bytes are element-ordered so compute reads plain words;       \
         * slots live at [sg][row][48B] (sg-major) so one ldmatrix.x4 loads   \
         * a whole 16-od-row x 32-k B fragment per warp-chunk.                \
         * At KDR=4 two consecutive k-tiles share the super-block and the     \
         * expanded qb8 persists across the kt barrier - restage only when    \
         * this k-tile starts a new super-block. */                           \
        if (((kt) * KDR & 7) == 0) {                                           \
        for (int x = threadIdx.x; x < MMQ_WBJ * 4; x += blockDim.x) {          \
            const int r = x >> 2, p = x & 3;                                   \
            const int j = j0 + r, sb = ((kt) * KDR) >> 3;                      \
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = make_uint4(0, 0, 0, 0);    \
            if (j < od && sb < nsb) {                                          \
                const uint8_t* src =                                           \
                    W + (size_t)j * ((size_t)nsb * 144)                        \
                      + (size_t)sb * 144 + 16 + p * 32;                        \
                v0 = *(const uint4*)(src);                                     \
                v1 = *(const uint4*)(src + 16);                                \
            }                                                                  \
            const unsigned M = 0x0F0F0F0Fu;                                    \
            uint8_t* dst = qb8 + (size_t)(p * 2) * (MMQ_WBJ * MMQ_WBQ)         \
                         + (size_t)r * MMQ_WBQ;                                \
            *(uint4*)(dst)      = make_uint4(v0.x & M, v0.y & M,               \
                                             v0.z & M, v0.w & M);              \
            *(uint4*)(dst + 16) = make_uint4(v1.x & M, v1.y & M,               \
                                             v1.z & M, v1.w & M);              \
            uint8_t* dst1 = dst + MMQ_WBJ * MMQ_WBQ;                           \
            *(uint4*)(dst1)     = make_uint4((v0.x >> 4) & M,                  \
                                             (v0.y >> 4) & M,                  \
                                             (v0.z >> 4) & M,                  \
                                             (v0.w >> 4) & M);                 \
            *(uint4*)(dst1 + 16) = make_uint4((v1.x >> 4) & M,                 \
                                              (v1.y >> 4) & M,                 \
                                              (v1.z >> 4) & M,                 \
                                              (v1.w >> 4) & M);                \
        }                                                                      \
        }                                                                      \
        for (int x = threadIdx.x; x < MMQ_WBJ * KDR; x += blockDim.x) {        \
            int r = x % MMQ_WBJ, kd = x / MMQ_WBJ;                             \
            int j = j0 + r, c = (kt) * KDR + kd;                               \
            float dv = 0.0f, mv = 0.0f;                                        \
            if (j < od && c < nchunk) {                                        \
                const uint8_t* blk =                                           \
                    W + (size_t)j * ((size_t)nsb * 144)                        \
                      + (size_t)(c >> 3) * 144;                                \
                float d = h2f(*(const uint16_t*)blk);                          \
                float dmin = h2f(*(const uint16_t*)(blk + 2));                 \
                uint8_t sc, m;                                                 \
                get_scale_min_k4(c & 7, blk + 4, &sc, &m);                     \
                dv = d * (float)sc;                                            \
                mv = -(dmin * (float)m);                                       \
            }                                                                  \
            sds[(size_t)kd * MMQ_WBJ + r] = make_float2(dv, mv);               \
        }                                                                      \
    } while (0)

    RAW_STAGE(0);

    // r22: precomputed swizzled A-frag byte offsets. The 8 ldmatrix
    // addresses per chunk are lane-invariant except for the kd base:
    // addr = qat + G[g], G[g] = g*512 + (lane&12)*32
    //      + (((lane&3)*2 + (lane>>4&1)) ^ (lane>>2&3) ^ ((g&1)*4)) << 4.
    // One IADD per ldmatrix (below baseline's 3), and the granule XOR
    // gives every ldmatrix phase 8 distinct bank phases (the 32B row
    // stride is 2-way conflicted).
    const unsigned l12m = (unsigned)(lane & 12) * 32;
    const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                             ^ ((lane >> 2) & 3)) << 4;
    unsigned G[8];
    #pragma unroll
    for (int g = 0; g < 8; g++)
        G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);

    int buf = 0; (void)buf;
    for (int kt = 0; kt < nktile; ++kt) {
        if (kt > 0) RAW_STAGE(kt);
        __syncthreads();          // single-buffer stage visible to all warps

        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)kd * MMQ_WBI * 32;

            // A fragments: 8 independent 16-token groups tile the full
            // 128-token row, one ldmatrix.x4 per group (16 rows x 32B:
            // lanes 0-7 -> rows 0-7 byte 0, 8-15 -> rows 8-15 byte 0,
            // 16-23 -> rows 0-7 byte 16, 24-31 -> rows 8-15 byte 16 —
            // the standard m16n8k32 A-fragment distribution).
            // r22: addresses are the precomputed swizzled offsets G[g]
            // (see above) — XOR-swizzled granule index, same map the
            // staging stores use.
            int a[8][4], b[2][2];
            int clow[8][2][4];
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint8_t* p = qat + G[g];
                unsigned r0_, r1_, r2_, r3_;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                    : "r"((unsigned)__cvta_generic_to_shared(p)));
                a[g][0] = (int)r0_; a[g][1] = (int)r1_;
                a[g][2] = (int)r2_; a[g][3] = (int)r3_;
            }
            // B fragments: ONE ldmatrix.x4 serves both 8-od-row
            // minitiles (matrices 0/1 = od-rows 0-7 at k-halves 0/1,
            // matrices 2/3 = od-rows 8-15). reg_i of lane L = matrix_i row
            // L/4, bytes (L%4)*4 — the exact mma.m16n8k32 B-operand
            // distribution the plain LDS pattern produced. Per-lane address
            // parts are loop-invariant; only the sg term moves per chunk.
            {
                const uint8_t* rb8 = qb8
                    + (size_t)sg * (MMQ_WBJ * MMQ_WBQ)
                    + (size_t)(j0w + (lane >> 4) * 8 + (lane & 7)) * MMQ_WBQ
                    + (size_t)((lane >> 3) & 1) * 16;
                unsigned b0_, b1_, b2_, b3_;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(b0_), "=r"(b1_), "=r"(b2_), "=r"(b3_)
                    : "r"((unsigned)__cvta_generic_to_shared(rb8)));
                b[0][0] = (int)b0_; b[0][1] = (int)b1_;
                b[1][0] = (int)b2_; b[1][1] = (int)b3_;
            }
            #pragma unroll
            for (int g = 0; g < 8; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) clow[g][nh][l] = 0;
            // 16 independent mma chains per thread per chunk, all C
            // fragments live simultaneously (llama.cpp accumulator depth).
            #pragma unroll
            for (int g = 0; g < 8; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    mmq_mma_k32(clow[g][nh], a[g], b[nh]);

            // rescale: identical math/layout to the R1 kernel; A-side
            // d/ssum come straight from the raw chunk. da/sa load per
            // token-group to keep registers for the accumulators.
            // od-col scales: one float4 per minitile serves the (j, j+1)
            // column pair the C fragment consumes (float2-packed at staging).
            float dsv[2][2], dmv[2][2];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const float4 sc4 = *(const float4*)(sds
                    + (size_t)kd * MMQ_WBJ + j0w + nh * 8 + (lane & 3) * 2);
                dsv[nh][0] = sc4.x; dsv[nh][1] = sc4.z;
                dmv[nh][0] = sc4.y; dmv[nh][1] = sc4.w;
            }
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                float da_q[2];
                int sa_q[2];
                // token pair (t, t+8) in one LDS.64 (uint2 tiling)
                const uint2 pk2 = *(const uint2*)(sda_q
                    + (size_t)kd * MMQ_WBI * 2 + g * 16 + (lane >> 2) * 2);
                da_q[0] = h2f((unsigned short)(pk2.x & 0xFFFF));
                sa_q[0] = (int)(short)(pk2.x >> 16);
                da_q[1] = h2f((unsigned short)(pk2.y & 0xFFFF));
                sa_q[1] = (int)(short)(pk2.y >> 16);
                // r15: the dmv correction term is rank-1 in (token, od-col) —
                // the row-side product da*sa is shared by the od-col pair of
                // each C fragment, so fold it once per row (16 FMUL/chunk)
                // instead of once per C value (64 FMUL/chunk). The dsv term
                // and the per-chunk scale application are unchanged.
                const float dma[2] = { da_q[0] * (float)sa_q[0],
                                       da_q[1] * (float)sa_q[1] };
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        const float da = da_q[l >> 1];
                        const int idx = (g * 2 + nh) * 4 + l;
                        sum[idx] += da * dsv[nh][l & 1] * (float)clow[g][nh][l];
                        sum[idx] += dma[l >> 1] * dmv[nh][l & 1];
                    }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < 8; g++)
        #pragma unroll
        for (int nh = 0; nh < 2; nh++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = i0 + g * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od)
                    C[(size_t)i * od + j] = sum[(g * 2 + nh) * 4 + l];
            }
#endif // __CUDA_ARCH__ >= 800
}


// ─── mmq RAW-NIBBLE kernel (Direction-A "NB" variant, docs §11) ───────────
// 64 tokens x 128 od, KD=8 native (one full 256-k super-block per k-tile in
// the raw qs plane), 8 warps x 16 od-rows, sum[32]. The weight B tile is the
// RAW 2-nibbles/byte qs plane (O*128 = 16,384 B), so the block totals
// 45,056 B -> 2 blocks/SM on GB10. B fragments are assembled in-loop by
// reading the packed nibbles (NO staging-ALU expansion, NO ldmatrix for B).
// Numerics are exact to the wide kernel: the mma consumes the UNSIGNED 0..15
// nibble as the int8 B operand, and the fp32 two-term rank-1 rescale
// (d*sc*nib - dmin*m) is applied per chunk — never the (nib - m) fold.
//
// The A path is byte-identical to mmq_raw_wide_nt_kernel (r20 split-phase
// staging + r22 XOR swizzle, 32B rows) but over T=64 (4 A-frag groups).
// The B unpack was validated standalone against the wide kernel's
// ldmatrix B-fragment: for chunk sg, od-row jj, lane l:
//   reg0 byte j = nibble(sg&1 ? hi : lo) of qs[(sg>>1)*32 + (l&3)*4 + j]
//   reg1 byte j = nibble(sg&1 ? hi : lo) of qs[(sg>>1)*32 + 16 + (l&3)*4 + j]
// (0x0F0F0F0F low, (v>>4)&0x0F0F0F0F high; upper nibble zero => positive int8).

extern "C" int launch_mmq_raw_wide_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, cudaStream_t stream, int kd
) {
    (void)type_id;
    // 16-chain layout: 128-token x 128-od block tile. r14: qb8 slot-major
    // 48B stride (ldmatrix-for-B) + packed scales. KD=8 totals 98,304B and
    // KD=4 73,728B — both inside the ~99KB opt-in cap, 1 block/SM. The
    // attr/launch results are checked: an over-cap request used to fail
    // SILENTLY (r7 phantom 2124). #147 routes both through the shared named
    // helpers, so a failure says which call and which instantiation.
    dim3 grid((nt + 127) / 128, (od + 127) / 128);
    if (kd <= 4) {
        const int smem = 4 * MMQ_WBI * 32 + 4 * MMQ_WBI * 8
                       + 8 * MMQ_WBJ * MMQ_WBQ + 2 * 4 * MMQ_WBJ * 4;
        const char* const kname = "mmq_raw_wide_nt_kernel<4>";
        if (!minfer_smem_optin("attr:mmq_raw_wide_kd4", kname,
                               reinterpret_cast<const void*>(&mmq_raw_wide_nt_kernel<4>), smem))
            return 0;
        minfer_launch_prelude("launch:mmq_raw_wide_kd4", kname);
        mmq_raw_wide_nt_kernel<4><<<grid, 256,
                                    minfer_launch_smem("launch:mmq_raw_wide_kd4", smem),
                                    stream>>>(w, q8, c, nt, od, id);
        if (!minfer_launch_ok_opt("launch:mmq_raw_wide_kd4", kname)) return 0;
    } else {
        const int smem = 8 * MMQ_WBI * 32 + 8 * MMQ_WBI * 8
                       + 8 * MMQ_WBJ * MMQ_WBQ + 2 * 8 * MMQ_WBJ * 4;
        const char* const kname = "mmq_raw_wide_nt_kernel<8>";
        if (!minfer_smem_optin("attr:mmq_raw_wide_kd8", kname,
                               reinterpret_cast<const void*>(&mmq_raw_wide_nt_kernel<8>), smem))
            return 0;
        minfer_launch_prelude("launch:mmq_raw_wide_kd8", kname);
        mmq_raw_wide_nt_kernel<8><<<grid, 256,
                                    minfer_launch_smem("launch:mmq_raw_wide_kd8", smem),
                                    stream>>>(w, q8, c, nt, od, id);
        if (!minfer_launch_ok_opt("launch:mmq_raw_wide_kd8", kname)) return 0;
    }
    return 1;
}

// #147: the terminal raw-narrow launcher (the last resort of the MMQ dispatch)
// now returns 1 = launched and accepted, 0 = refused. A 0 is an `Err` at the
// Rust caller, never a silent launch over an un-opted-in dynamic smem.
extern "C" int launch_mmq_raw_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, cudaStream_t stream, int kd
) {
    (void)type_id; // q4_K only in the first cut
    dim3 grid((nt + 63) / 64, (od + 63) / 64);
    if (kd <= 4) {
        const int smem = 2 * 4 * MMQ_BI * 40 + 2 * MMQ_BI * 144
                         + 2 * 2 * 4 * MMQ_BI * 4;
        const char* const kname = "mmq_raw_nt_kernel<4>";
        if (!minfer_smem_optin("attr:mmq_raw_nt_kd4", kname,
                               reinterpret_cast<const void*>(&mmq_raw_nt_kernel<4>), smem))
            return 0;
        minfer_launch_prelude("launch:mmq_raw_nt_kd4", kname);
        mmq_raw_nt_kernel<4><<<grid, 256,
                               minfer_launch_smem("launch:mmq_raw_nt_kd4", smem), stream>>>(
            w, q8, c, nt, od, id);
        return minfer_launch_ok("launch:mmq_raw_nt_kd4", kname) ? 1 : 0;
    }
    const int smem = 2 * 8 * MMQ_BI * 40 + 2 * MMQ_BI * 144
                     + 2 * 2 * 8 * MMQ_BI * 4;
    const char* const kname = "mmq_raw_nt_kernel<8>";
    if (!minfer_smem_optin("attr:mmq_raw_nt_kd8", kname,
                           reinterpret_cast<const void*>(&mmq_raw_nt_kernel<8>), smem))
        return 0;
    minfer_launch_prelude("launch:mmq_raw_nt_kd8", kname);
    mmq_raw_nt_kernel<8><<<grid, 256, minfer_launch_smem("launch:mmq_raw_nt_kd8", smem), stream>>>(
        w, q8, c, nt, od, id);
    return minfer_launch_ok("launch:mmq_raw_nt_kd8", kname) ? 1 : 0;
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// This file owns the template instantiations, so the address-taking must
// happen here (a cross-TU template reference is nvcc #20280-D and can fail
// to link).
extern "C" void minfer_prewarm_mmq_raw_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, (mmq_raw_nt_kernel<8>));
    MINFER_PREWARM_ONE(a, (mmq_raw_wide_nt_kernel<8>));
}
