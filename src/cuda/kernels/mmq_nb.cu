// src/cuda/kernels/mmq_nb.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"

template <int KDR>
__global__ void __launch_bounds__(256) mmq_raw_nb_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id
) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_nb_sh[];
    // Single-buffer sync-staged, KD=8 totals 43,008 B -> 2 blocks/SM:
    //   qa8     [KDR][64][32]   chunk q8 planes (r22 XOR swizzle, r20 split)
    //   sda_q   [KDR][64]         (d f16 | ssum i16) packed, one uint32 per
    //                             token, Q-MAJOR: within kd the 4 token groups
    //                             for a given lane's pair q are CONTIGUOUS
    //                             (uint32 idx = kd*64 + q*8 + g*2 + half), so a
    //                             lane reads its whole per-chunk d/ssum set with
    //                             TWO LDS.128 (32 LDS.64 -> 16 LDS.128 / k-tile).
    //   qb_raw  [128][128]        raw GGUF qs plane (2 nibbles/byte, full
    //                             super-block per od-row)
    //   sds     [KDR][128] float2 (d | dmin*m): r15 rank-1 rescale terms
    uint8_t* qa8 = mmq_nb_sh;
    uint32_t* sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_NBI * 32);
    uint8_t* qb_raw = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_NBI);
    float2* sds = reinterpret_cast<float2*>(qb_raw + MMQ_NBJ * 128);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Each warp owns a private 16-od-row slice and reads the FULL 64-token
    // tile (B fragments warp-exclusive, A fragments warp-shared).
    const int i0 = blockIdx.x * MMQ_NBI;
    const int j0 = blockIdx.y * MMQ_NBJ;
    const int j0w = warp * 16;
    const int nb32 = id >> 5;
    const int nchunk = nb32;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    float sum[32] = {0.0f};   // [g=4][nh=2][l=4] C accumulators (fp32)

#define RAW_STAGE_NB(kt)                                                      \
    do {                                                                       \
        /* ---- A: r20 split-phase (LDG batch then STS) + r22 XOR swizzle ----*/\
        {                                                                      \
            unsigned av[KDR * 2];   /* 8 words x 64 tok x KDR / 256 thr */     \
            unsigned short dv[2];   /* KDR * NBI/256 = 2 f16 d words */        \
            unsigned sv[2];                                                     \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),             \
                          kd = x / (8 * MMQ_NBI);   /* KDR=8, 8*NBI = 512 */  \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned v = 0;                                                \
                if (tok < nt && c < nchunk)                                    \
                    v = *(const unsigned*)(q8x                                 \
                        + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4);          \
                av[i] = v;                                                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 2; ++i) {                                      \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_NBI - 1), kd = x / MMQ_NBI;  /* x/NBI */ \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned short d16 = 0; unsigned ss = 0;                       \
                if (tok < nt && c < nchunk) {                                  \
                    const uint8_t* src = q8x + ((size_t)tok * nb32 + c) * 40;  \
                    d16 = *(const unsigned short*)src;                         \
                    ss = (unsigned)(short)*(const int*)(src + 36);             \
                }                                                              \
                dv[i] = d16; sv[i] = ss;                                       \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),             \
                          kd = x / (8 * MMQ_NBI);                               \
                const int R = kd * MMQ_NBI + r;                                \
                *(unsigned*)(qa8 + (size_t)(R & ~3) * 32                       \
                    + (size_t)(((((R & 3) << 1) + (u >> 2))                    \
                                ^ ((R >> 2) & 7)) << 4)                        \
                    + (size_t)(u & 3) * 4) = av[i];                            \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 2; ++i) {                                      \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_NBI - 1), kd = x / MMQ_NBI;             \
                const int g = r >> 4, t = r & 15, q = t & 7, half = t >> 3;    \
                const int rg = g >> 1, gsel = g & 1;                           \
                /* conflict-free: region=g/2 block, q*16B stride, gsel*8B */  \
                *(unsigned*)(sda_q + (size_t)kd * MMQ_NBI                       \
                              + rg * 32 + q * 4 + gsel * 2 + half) =           \
                    dv[i] | (sv[i] << 16);                                     \
            }                                                                  \
        }                                                                      \
        /* ---- B: bulk raw qs super-block copy (r18-style, no staging ALU) */ \
        {                                                                      \
            const int sb = ((kt) * KDR) >> 3;                                  \
            for (int off = threadIdx.x; off < MMQ_NBJ * 8; off += blockDim.x) {\
                const int jj = off >> 3, c8 = off & 7;                         \
                const int j = j0 + jj;                                         \
                uint4 v = make_uint4(0, 0, 0, 0);                              \
                if (j < od && sb < nsb)                                        \
                    v = *(const uint4*)(W + (size_t)j * ((size_t)nsb * 144)    \
                        + (size_t)sb * 144 + 16 + (size_t)c8 * 16);            \
                /* doc 99 P1 XOR swizzle (the A-plane r22 trick applied to  */\
                /* B): rows are 128 B = the full 32-bank span, so unswizzled*/\
                /* rows alias the same banks and the compute's 8-row reads  */\
                /* 4-way-conflict (ncu: 40% excessive shared wavefronts).   */\
                /* Scramble the 16-B chunk index by the row; the compute    */\
                /* read applies the same XOR, so the bytes are unchanged.   */\
                *(uint4*)(qb_raw + (size_t)jj * 128                            \
                          + (size_t)(c8 ^ (jj & 7)) * 16) = v;                 \
            }                                                                  \
        }                                                                      \
        /* ---- B: SDS per-(chunk, od-row) rank-1 rescale terms ---- */        \
        for (int x = threadIdx.x; x < MMQ_NBJ * KDR; x += blockDim.x) {        \
            const int r = x % MMQ_NBJ, kd = x / MMQ_NBJ;                       \
            const int j = j0 + r, c = (kt) * KDR + kd;                         \
            float dv = 0.0f, mv = 0.0f;                                        \
            if (j < od && c < nchunk) {                                        \
                const uint8_t* blk = W + (size_t)j * ((size_t)nsb * 144)       \
                    + (size_t)(c >> 3) * 144;                                  \
                const float d = h2f(*(const uint16_t*)blk);                    \
                const float dmin = h2f(*(const uint16_t*)(blk + 2));           \
                uint8_t sc, m;                                                 \
                get_scale_min_k4(c & 7, blk + 4, &sc, &m);                     \
                dv = d * (float)sc;                                            \
                mv = -(dmin * (float)m);                                       \
            }                                                                  \
            sds[(size_t)kd * MMQ_NBJ + r] = make_float2(dv, mv);               \
        }                                                                      \
    } while (0)

    // r22: precomputed swizzled A-frag byte offsets (identical to wide kernel).
    const unsigned l12m = (unsigned)(lane & 12) * 32;
    const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                             ^ ((lane >> 2) & 3)) << 4;
    unsigned G[4];   // only 4 token groups at T=64
    #pragma unroll
    for (int g = 0; g < 4; g++)
        G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);

    RAW_STAGE_NB(0);

    for (int kt = 0; kt < nktile; ++kt) {
        if (kt > 0) RAW_STAGE_NB(kt);
        __syncthreads();

        #pragma unroll
        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)kd * MMQ_NBI * 32;

            // A fragments: 4 independent 16-token groups (T=64), r22 G[].
            int a[4][4], b[2][2];
            #pragma unroll
            for (int g = 0; g < 4; g++) {
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
            // B fragments: raw-nibble in-loop unpack (validated == wide kernel
            // ldmatrix). reg0 = qs[(sg>>1)*32 + (l&3)*4 + 0..3],
            //            reg1 = qs[(sg>>1)*32 + 16 + (l&3)*4 + 0..3].
            {
                const int p = sg >> 1, is_hi = sg & 1, lm3 = lane & 3;
                const unsigned M = 0x0F0F0F0Fu;
                #pragma unroll
                for (int nh = 0; nh < 2; nh++) {
                    const int jj = j0w + nh * 8 + (lane >> 2);
                    // doc 99 P1: mirror the staging XOR. Logical 16-B chunks
                    // p*2 and p*2+1 of row jj map to ((p*2) ^ ph) and
                    // ((p*2+1) ^ ph) — XOR each chunk index separately (it
                    // does NOT distribute over the +1).
                    const int ph = jj & 7;
                    const uint8_t* qs0 =
                        qb_raw + (size_t)jj * 128
                        + (size_t)((p * 2) ^ ph) * 16;
                    const uint8_t* qs1 =
                        qb_raw + (size_t)jj * 128
                        + (size_t)((p * 2 + 1) ^ ph) * 16;
                    const uint32_t* q0 = (const uint32_t*)(qs0 + lm3 * 4);
                    const uint32_t* q1 = (const uint32_t*)(qs1 + lm3 * 4);
                    uint32_t v0 = *q0, v1 = *q1;
                    b[nh][0] = (int)(is_hi ? ((v0 >> 4) & M) : (v0 & M));
                    b[nh][1] = (int)(is_hi ? ((v1 >> 4) & M) : (v1 & M));
                }
            }
            int clow[4][2][4];
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) clow[g][nh][l] = 0;
            // 8 independent mma chains per thread per chunk (4 A-frags x 2
            // B-frags), all C fragments live simultaneously.
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    mmq_mma_k32(clow[g][nh], a[g], b[nh]);

            // rescale: exact r15 two-term rank-1 fold (d*sc, -dmin*m).
            float dsv[2][2], dmv[2][2];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const float4 sc4 = *(const float4*)(sds
                    + (size_t)kd * MMQ_NBJ + j0w + nh * 8 + (lane & 3) * 2);
                dsv[nh][0] = sc4.x; dsv[nh][1] = sc4.z;
                dmv[nh][0] = sc4.y; dmv[nh][1] = sc4.w;
            }
            // r31: Q-major sda repack — one uint32 per token; the group-region
            // split (g/2 region block, q*16B stride) makes each warp LDS.128
            // read 8 unique 16B words at 16B stride = bank-conflict-free.
            // s0 = groups 0,1, s1 = groups 2,3 (TWO LDS.128 instead of the old
            // four LDS.64). Same values, same per-chunk application points.
            const uint32_t* sda_blk = sda_q + (size_t)kd * MMQ_NBI
                                      + (size_t)(lane >> 2) * 4;
            const uint4 s0 = *(const uint4*)(sda_blk);
            const uint4 s1 = *(const uint4*)(sda_blk + 32);
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                float da_q[2];
                int sa_q[2];
                const unsigned w0 = g == 0 ? s0.x : (g == 1 ? s0.z : (g == 2 ? s1.x : s1.z));
                const unsigned w1 = g == 0 ? s0.y : (g == 1 ? s0.w : (g == 2 ? s1.y : s1.w));
                da_q[0] = h2f((unsigned short)(w0 & 0xFFFF));
                sa_q[0] = (int)(short)(w0 >> 16);
                da_q[1] = h2f((unsigned short)(w1 & 0xFFFF));
                sa_q[1] = (int)(short)(w1 >> 16);
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
    for (int g = 0; g < 4; g++)
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

// --- P6 r34: NB kernel with the A-side layout transform relocated into a
// quantize-transpose prepass (MINFER_MMQ_A_TRANSPOSE=1). Byte-identical
// compute to mmq_raw_nb_kernel (same qa8/sda_q smem content, same ldmatrix
// fragment reads, same rescale) — only the A STAGING differs: the qs plane and
// the packed d|ssum are emitted PRE-TRANSPOSED (quantize_q8_0_pad40_t) so the
// per-(kt, warp) A reads become contiguous bulk LDG->STS with no per-element
// index math (the r22 XOR swizzle and the r31 q-major sda repack are baked into
// the prepass layout). The B (weight) + SDS staging is unchanged.
// r59: DSC=true stages the per-(chunk, od-row) rescale terms
// float2(d*sc, -(dmin*m)) from the registration-time W_dsc f32-pair plane
// (chunk-major, 16-B cp.async stream) instead of decoding them in-staging
// (get_scale_min_k4 + 2 h2f per (chunk, od-row) — the r58 spec's one real
// q4_K staging-ALU residual). DSC=false keeps the scalar decode verbatim.
template <int KDR, bool DSC>
__global__ void __launch_bounds__(256) mmq_raw_nb_bt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ W_dsc,
    const uint8_t* __restrict__ qa8g,
    const uint8_t* __restrict__ sdag, float* __restrict__ C,
    int nt, int od, int id, int nchunk,
    float* __restrict__ Cpart, int ksplit
) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_nb_sh[];
    // doc 91: two staging buffer sets — tile kt+1's bulk LDG->STS (and the
    // DSC cp.async sds stream) is prefetched into the other buffer while
    // tile kt computes, hiding the per-tile staging latency that serialised
    // the single-buffer loop (the small-M floor: ntb=1 leaves only od/NBJ
    // blocks, so no other block hides the stall either). Pure data-movement
    // restructure: fragments, operand values and accumulation order are
    // unchanged, so results are bitwise identical to the single-buffer
    // kernel.
    constexpr size_t BUF = (size_t)KDR * MMQ_NBI * 32      // qa8
                         + (size_t)KDR * MMQ_NBI * 4       // sda_q
                         + (size_t)MMQ_NBJ * 128           // qb_raw
                         + (size_t)8 * MMQ_NBJ * 8;        // sds
    uint8_t* qa8;
    uint32_t* sda_q;
    uint8_t* qb_raw;
    float2* sds;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i0 = blockIdx.x * MMQ_NBI;
    const int j0 = blockIdx.y * MMQ_NBJ;
    const int j0w = warp * 16;
    const int nb32 = id >> 5;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    // doc 92: K-split — grid.z slots each own a contiguous tile range and
    // write fp32 partials; ksplit == 1 keeps the single-pass C write. Each
    // slot sums its range in ascending kt order (same per-slot order as the
    // unsplit kernel) and the reduce kernel sums slots in fixed z order,
    // so results are run-to-run bit-stable.
    const int per = ksplit > 1 ? (nktile + ksplit - 1) / ksplit : nktile;
    const int kt_lo = (int)blockIdx.z * per;
    const int kt_hi = min(kt_lo + per, nktile);

    float sum[32] = {0.0f};   // [g=4][nh=2][l=4] C accumulators (fp32)

#define RAW_STAGE_NB_BT(kt)                                                    \
    do {                                                                       \
        /* ---- A: bulk LDG->STS of the pre-transposed qa8/sda (no math) ----*/\
        {                                                                      \
            const size_t qbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_QASZ; \
            for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16;       \
                 off += blockDim.x)                                            \
                ((uint4*)(qa8))[off] = ((const uint4*)(qa8g + qbase))[off];    \
            const size_t sbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_SDASZ; \
            for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 4) / 16;        \
                 off += blockDim.x)                                            \
                ((uint4*)(sda_q))[off] = ((const uint4*)(sdag + sbase))[off];  \
        }                                                                      \
        /* ---- B: bulk raw qs super-block copy (r18-style, no staging ALU) */\
        {                                                                      \
            const int sb = ((kt) * KDR) >> 3;                                  \
            for (int off = threadIdx.x; off < MMQ_NBJ * 8; off += blockDim.x) {\
                const int jj = off >> 3, c8 = off & 7;                         \
                const int j = j0 + jj;                                         \
                uint4 v = make_uint4(0, 0, 0, 0);                              \
                if (j < od && sb < nsb)                                        \
                    v = *(const uint4*)(W + (size_t)j * ((size_t)nsb * 144)    \
                        + (size_t)sb * 144 + 16 + (size_t)c8 * 16);            \
                /* doc 99 P1 XOR swizzle (the A-plane r22 trick applied to  */\
                /* B): rows are 128 B = the full 32-bank span, so unswizzled*/\
                /* rows alias the same banks and the compute's 8-row reads  */\
                /* 4-way-conflict (ncu: 40% excessive shared wavefronts).   */\
                /* Scramble the 16-B chunk index by the row; the compute    */\
                /* read applies the same XOR, so the bytes are unchanged.   */\
                *(uint4*)(qb_raw + (size_t)jj * 128                            \
                          + (size_t)(c8 ^ (jj & 7)) * 16) = v;                 \
            }                                                                  \
        }                                                                      \
        /* ---- B: SDS per-(chunk, od-row) rank-1 rescale terms ---- */        \
        /* r59: W_dsc f32 plane (registration-time precompute; chunk-major     \
         * layout plane[c*od + j] = float2(d*sc, -(dmin*m))) removes the       \
         * per-(chunk,row) get_scale_min_k4 branch + 2 h2f converts + 2        \
         * multiplies from the staging critical path (the r56 q6_K mechanism  \
         * applied to the OTHER 63% of bt busy). od even (registration gate)  \
         * => a row PAIR is fully valid or fully beyond od (src-size           \
         * zero-fill); nchunk % 8 == 0 (launch gate) with KDR=8 =>             \
         * c0d+kdd < nchunk always, exactly like the r56 q6_K plane.           \
         * Null plane = DSC=false scalar path, byte-identical. */             \
        if (DSC) {                                                             \
            const int c0d = (kt) * KDR;                                        \
            const int nc2 = MMQ_NBJ / 2;  /* 16-B chunks (2 float2) per kd */  \
            for (int g = threadIdx.x; g < KDR * nc2; g += blockDim.x) {        \
                const int kdd = g / nc2, mm = g % nc2;                         \
                const int j = j0 + 2 * mm;                                     \
                const bool full = (j + 1 < od);                                \
                gemm_cp16(                                                     \
                    (__half*)(void*)(sds + (size_t)kdd * MMQ_NBJ + 2 * mm),    \
                    (const __half*)(const void*)(W_dsc                         \
                        + ((size_t)(c0d + kdd) * (size_t)od + (size_t)j) * 8), \
                    full);                                                     \
            }                                                                  \
            gemm_cp_commit();                                                  \
        } else {                                                               \
        for (int x = threadIdx.x; x < MMQ_NBJ * KDR; x += blockDim.x) {        \
            const int r = x % MMQ_NBJ, kd = x / MMQ_NBJ;                       \
            const int j = j0 + r, c = (kt) * KDR + kd;                         \
            float dv = 0.0f, mv = 0.0f;                                        \
            if (j < od && c < nchunk) {                                        \
                const uint8_t* blk = W + (size_t)j * ((size_t)nsb * 144)       \
                    + (size_t)(c >> 3) * 144;                                  \
                const float d = h2f(*(const uint16_t*)blk);                    \
                const float dmin = h2f(*(const uint16_t*)(blk + 2));           \
                uint8_t sc, m;                                                 \
                get_scale_min_k4(c & 7, blk + 4, &sc, &m);                     \
                dv = d * (float)sc;                                            \
                mv = -(dmin * (float)m);                                       \
            }                                                                  \
            sds[(size_t)kd * MMQ_NBJ + r] = make_float2(dv, mv);               \
        }                                                                      \
        }                                                                      \
    } while (0)

    const unsigned l12m = (unsigned)(lane & 12) * 32;
    const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                             ^ ((lane >> 2) & 3)) << 4;
    unsigned G[4];
    #pragma unroll
    for (int g = 0; g < 4; g++)
        G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);

    // doc 91: double-buffer only when the grid is M-starved (ntb == 1, the
    // small-M regime): there the staging latency of tile kt+1 has no other
    // block to hide behind and pipelining pays. With ntb >= 2 the extra
    // buffer set halves SM residency (84 KB vs 42 KB per block) and cost
    // prefill ~10% (measured), so the single-buffer r59 sequence is kept.
    // doc 92: with a K-split active the grid already has block-level
    // parallelism to spare — keep the single-buffer 42 KB footprint for its
    // higher SM residency (4 vs 2 blocks/SM) and drop the prefetch.
    const bool dbuf = nt <= MMQ_NBI && ksplit == 1;

    // prologue: stage the slot's first tile into the buffer the regime owns
    // (double-buffered: the kt_lo parity buffer; single-buffer: buffer 0)
    qa8 = dbuf ? mmq_nb_sh + (size_t)(kt_lo & 1) * BUF : mmq_nb_sh;
    sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_NBI * 32);
    qb_raw = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_NBI);
    sds = reinterpret_cast<float2*>(qb_raw + MMQ_NBJ * 128);
    RAW_STAGE_NB_BT(kt_lo);

    for (int kt = kt_lo; kt < kt_hi; ++kt) {
        if (dbuf) {
            // prefetch tile kt+1 into the OTHER buffer while tile kt computes
            if (kt + 1 < kt_hi) {
                uint8_t* nb = mmq_nb_sh + (size_t)((kt + 1) & 1) * BUF;
                qa8 = nb;
                sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_NBI * 32);
                qb_raw = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_NBI);
                sds = reinterpret_cast<float2*>(qb_raw + MMQ_NBJ * 128);
                RAW_STAGE_NB_BT(kt + 1);
            }
        } else if (kt > kt_lo) {
            // original r59 sequence: restage into the single buffer (the
            // first tile was staged by the prologue)
            RAW_STAGE_NB_BT(kt);
        }
        // r59 visibility rule: tile kt's DSC cp.async group must be complete
        // to THIS thread before the barrier publishes it cross-thread; in the
        // double-buffered regime tile kt+1's group (already issued) may stay
        // in flight (wait_group 1). DSC=false: no groups are issued, no-op.
        qa8 = dbuf ? mmq_nb_sh + (size_t)(kt & 1) * BUF : mmq_nb_sh;
        sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_NBI * 32);
        qb_raw = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_NBI);
        sds = reinterpret_cast<float2*>(qb_raw + MMQ_NBJ * 128);
        if (DSC) {
            if (dbuf && kt + 1 < kt_hi) gemm_cp_wait1(); else gemm_cp_wait0();
        }
        __syncthreads();

        #pragma unroll
        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)kd * MMQ_NBI * 32;

            int a[4][4], b[2][2];
            #pragma unroll
            for (int g = 0; g < 4; g++) {
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
            {
                const int p = sg >> 1, is_hi = sg & 1, lm3 = lane & 3;
                const unsigned M = 0x0F0F0F0Fu;
                #pragma unroll
                for (int nh = 0; nh < 2; nh++) {
                    const int jj = j0w + nh * 8 + (lane >> 2);
                    // doc 99 P1: mirror the staging XOR. Logical 16-B chunks
                    // p*2 and p*2+1 of row jj map to ((p*2) ^ ph) and
                    // ((p*2+1) ^ ph) — XOR each chunk index separately (it
                    // does NOT distribute over the +1).
                    const int ph = jj & 7;
                    const uint8_t* qs0 =
                        qb_raw + (size_t)jj * 128
                        + (size_t)((p * 2) ^ ph) * 16;
                    const uint8_t* qs1 =
                        qb_raw + (size_t)jj * 128
                        + (size_t)((p * 2 + 1) ^ ph) * 16;
                    const uint32_t* q0 = (const uint32_t*)(qs0 + lm3 * 4);
                    const uint32_t* q1 = (const uint32_t*)(qs1 + lm3 * 4);
                    uint32_t v0 = *q0, v1 = *q1;
                    b[nh][0] = (int)(is_hi ? ((v0 >> 4) & M) : (v0 & M));
                    b[nh][1] = (int)(is_hi ? ((v1 >> 4) & M) : (v1 & M));
                }
            }
            int clow[4][2][4];
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) clow[g][nh][l] = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    mmq_mma_k32(clow[g][nh], a[g], b[nh]);

            float dsv[2][2], dmv[2][2];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const float4 sc4 = *(const float4*)(sds
                    + (size_t)kd * MMQ_NBJ + j0w + nh * 8 + (lane & 3) * 2);
                dsv[nh][0] = sc4.x; dsv[nh][1] = sc4.z;
                dmv[nh][0] = sc4.y; dmv[nh][1] = sc4.w;
            }
            const uint32_t* sda_blk = sda_q + (size_t)kd * MMQ_NBI
                                      + (size_t)(lane >> 2) * 4;
            const uint4 s0 = *(const uint4*)(sda_blk);
            const uint4 s1 = *(const uint4*)(sda_blk + 32);
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                float da_q[2];
                int sa_q[2];
                const unsigned w0 = g == 0 ? s0.x : (g == 1 ? s0.z : (g == 2 ? s1.x : s1.z));
                const unsigned w1 = g == 0 ? s0.y : (g == 1 ? s0.w : (g == 2 ? s1.y : s1.w));
                da_q[0] = h2f((unsigned short)(w0 & 0xFFFF));
                sa_q[0] = (int)(short)(w0 >> 16);
                da_q[1] = h2f((unsigned short)(w1 & 0xFFFF));
                sa_q[1] = (int)(short)(w1 >> 16);
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
    for (int g = 0; g < 4; g++)
        #pragma unroll
        for (int nh = 0; nh < 2; nh++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = i0 + g * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od) {
                    if (ksplit == 1)
                        C[(size_t)i * od + j] = sum[(g * 2 + nh) * 4 + l];
                    else
                        Cpart[((size_t)blockIdx.z * nt + i) * od + j] =
                            sum[(g * 2 + nh) * 4 + l];
                }
            }
#undef RAW_STAGE_NB_BT
#endif // __CUDA_ARCH__ >= 800
}

// r59: w_dsc != 0 selects the DSC=true instantiation (the registration-time
// W_dsc f32-pair plane; the SDS staging is a cp.async stream); w_dsc == 0
// keeps the r34 in-kernel scalar decode. Returns 0 (clean fallback) on any
// cap/mismatch.
// doc 92: deterministic K-split reduce — sums the per-slot fp32 partial
// planes in fixed ascending z order (run-to-run bit-stable, capture-replay
// parity safe; atomicAdd would not be).
__global__ void mmq_ksplit_reduce_kernel(
    const float* __restrict__ parts, float* __restrict__ C,
    int total, int ksplit
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)total) return;
    float acc = parts[idx];
    for (int z = 1; z < ksplit; ++z)
        acc += parts[(size_t)z * (size_t)total + idx];
    C[idx] = acc;
}

extern "C" int launch_mmq_raw_nb_bt_nt(
    int type_id, const uint8_t* w, const uint8_t* w_dsc, const uint8_t* qa8g,
    const uint8_t* sdag, float* c, int nt, int od, int id, int nchunk,
    cudaStream_t stream, int kd, float* cpart, int ksplit
) {
    (void)type_id;
    if (kd != 8) return 0;
    if (qa8g == 0 || sdag == 0) return 0;
    // doc 91: the second staging buffer set only for the M-starved regime
    // (ntb == 1, matching the kernel's dbuf flag) — prefill keeps the
    // original 42 KB footprint and its higher SM residency.
    // doc 92: refine the requested ksplit so every z-slot owns a non-empty
    // tile range (an empty slot would leave garbage partials for the reduce)
    int ks = ksplit > 1 ? ksplit : 1;
    {
        const int nktile_all = (nchunk + 7) / 8;
        if (ks > nktile_all) ks = nktile_all;
        if (ks < 1) ks = 1;
        const int per = (nktile_all + ks - 1) / ks;
        ks = (nktile_all + per - 1) / per;
    }
    const bool dbuf_smem =
        (nt + MMQ_NBI - 1) / MMQ_NBI <= 1 && ks <= 1;
    const int smem = (dbuf_smem ? 2 : 1) * (8 * MMQ_NBI * 32   // qa8
                   + 8 * MMQ_NBI * 4    // sda_q (one uint32 per token)
                   + MMQ_NBJ * 128      // qb_raw
                   + 8 * MMQ_NBJ * 8);  // sds (float2 = 8B)
    dim3 grid((nt + MMQ_NBI - 1) / MMQ_NBI, (od + MMQ_NBJ - 1) / MMQ_NBJ, ks);
    // r59: DSC is a template constant, so each instantiation keeps only its
    // own SDS path (the cp.async plane stream vs the scalar decode) — the
    // r53 pattern (no runtime branch in staging).
    const bool dsc = w_dsc != 0;
    // #147: the opt-in's own return value decides; the pre-#147 code read
    // `cudaGetLastError()` here, which treated ANY latch (from anywhere) as
    // "smem/reg cap" and discarded the error code and its origin.
    const char* const kname =
        dsc ? "mmq_raw_nb_bt_kernel<8,true>" : "mmq_raw_nb_bt_kernel<8,false>";
    const void* kfn = dsc ? reinterpret_cast<const void*>(&mmq_raw_nb_bt_kernel<8, true>)
                          : reinterpret_cast<const void*>(&mmq_raw_nb_bt_kernel<8, false>);
    if (!minfer_smem_optin("attr:mmq_raw_nb_bt", kname, kfn, smem)) return 0;
    minfer_launch_prelude("launch:mmq_raw_nb_bt", kname);
    if (dsc) {
        mmq_raw_nb_bt_kernel<8, true><<<grid, 256,
                                        minfer_launch_smem("launch:mmq_raw_nb_bt", smem),
                                        stream>>>(w, w_dsc, qa8g, sdag, c, nt, od, id, nchunk,
                                                  cpart, ks);
    } else {
        mmq_raw_nb_bt_kernel<8, false><<<grid, 256,
                                         minfer_launch_smem("launch:mmq_raw_nb_bt", smem),
                                         stream>>>(w, w_dsc, qa8g, sdag, c, nt, od, id, nchunk,
                                                   cpart, ks);
    }
    if (!minfer_launch_ok_opt("launch:mmq_raw_nb_bt", kname)) return 0;
    if (ks > 1) {
        const int total = nt * od;
        // #162: the reduce is a SEPARATE site token so the gate can arm it
        // alone — arming the kernel's token refuses the launch and returns 0
        // before this line, which would make the reduce site unreachable.
        minfer_launch_prelude("launch:mmq_raw_nb_bt_ksplit", "mmq_ksplit_reduce_kernel");
        mmq_ksplit_reduce_kernel<<<(total + 255) / 256, minfer_launch_block("launch:mmq_raw_nb_bt_ksplit", 256), 0, stream>>>(
            cpart, c, total, ks);
        if (!minfer_launch_ok_opt("launch:mmq_raw_nb_bt_ksplit", "mmq_ksplit_reduce_kernel")) return 0;
    }
    return 1;
}


extern "C" int launch_mmq_raw_nb_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, cudaStream_t stream, int kd
) {
    (void)type_id;
    // 64-token x 128-od block tile, KD=8 native. Raw qs plane + single-buffer
    // staging = 43,008 B (r31 q-major sda repack shrinks sda_q 4,096 ->
    // 2,048 B) >>> 2 blocks/SM on GB10. KD!=8 is inapplicable to the
    // raw-nibble variant (the qs plane encodes a FULL 256-k super-block), so
    // clean-fallback (return 0) to the wide kernel. smem/reg guards return 0
    // on any cap failure (never silently launch over cap).
    if (kd != 8) return 0;
    const int smem = 8 * MMQ_NBI * 32   // qa8
                   + 8 * MMQ_NBI * 4    // sda_q (one uint32 per token)
                   + MMQ_NBJ * 128      // qb_raw
                   + 8 * MMQ_NBJ * 8;   // sds (float2 = 8B)
    dim3 grid((nt + MMQ_NBI - 1) / MMQ_NBI, (od + MMQ_NBJ - 1) / MMQ_NBJ);
    // #147: the opt-in's own return value decides; the pre-#147 code read
    // `cudaGetLastError()` here, which treated ANY latch (from anywhere) as
    // "smem/reg cap" and discarded the error code and its origin.
    if (!minfer_smem_optin("attr:mmq_raw_nb", "mmq_raw_nb_kernel<8>",
                           reinterpret_cast<const void*>(&mmq_raw_nb_kernel<8>), smem))
        return 0;
    minfer_launch_prelude("launch:mmq_raw_nb", "mmq_raw_nb_kernel<8>");
    mmq_raw_nb_kernel<8><<<grid, 256, minfer_launch_smem("launch:mmq_raw_nb", smem), stream>>>(
        w, q8, c, nt, od, id);
    if (!minfer_launch_ok_opt("launch:mmq_raw_nb", "mmq_raw_nb_kernel<8>")) return 0;
    return 1;
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// This file owns the template instantiations, so the address-taking must
// happen here (a cross-TU template reference is nvcc #20280-D and can fail
// to link).
extern "C" void minfer_prewarm_mmq_nb_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, (mmq_raw_nb_bt_kernel<8, true>));
    MINFER_PREWARM_ONE(a, (mmq_raw_nb_bt_kernel<8, false>));
    MINFER_PREWARM_ONE(a, (mmq_raw_nb_kernel<8>));
}
