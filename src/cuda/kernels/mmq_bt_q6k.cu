// src/cuda/kernels/mmq_bt_q6k.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"

// `mmq_ksplit_reduce_kernel` lives next to the NB kernels (mmq_nb.cu) because
// `launch_mmq_raw_nb_bt_nt` is its other caller. A plain `__global__` may be
// launched across translation units without `-rdc`, so this TU only declares it.
__global__ void mmq_ksplit_reduce_kernel(const float* __restrict__ parts,
                                         float* __restrict__ C, int total, int ksplit);


// --- P6 r38: q6_K BT kernel (raw-byte, expanded int8 B) ---------------------
// q6_K uses mma.m16n8k16 (KSPLIT=2) because a 32-k chunk spans TWO 16-element
// sub-blocks with DIFFERENT scales (sc[2c%16], sc[(2c+1)%16]); the per-half int
// accumulators are rescaled separately with dsc0/dsc1 (single-term: no dmin).
// The B tile is EXPANDED to centered int8 (-32..31, 256 B/row, the ql+qh
// recombination + -32 centering all leave the hot loop — the r21/r22/r31
// "keep index math out of the loop" lesson) and the A side is the exact BT bulk
// LDG->STS of the pre-transposed qa8/sda (weight-type-agnostic).
// r53 bundle (EXP=true): the expansion itself was hoisted to registration — a
// dense centered-int8 plane W_exp (od x id, row stride = id, super-block
// stride = 256) built by expand_q6k_dense — so the B staging is a pure
// cp.async bulk copy (explicit PTX) with no recomb ALU and no ql/qh reads.
// EXP=false keeps the r41 in-kernel expand (raw 210-B layout / W_exp miss).
__device__ __forceinline__ int expand_q6_elem(const uint8_t* ql, const uint8_t* qh, int elem) {
    int m  = elem & 31;
    int it = elem >> 7;
    int n  = elem & 127;
    int ql_idx   = it * 64 + (n & 63);
    int ql_shift = (n >> 6) * 4;          // 0 or 4 (low/high nibble)
    int qh_idx   = it * 32 + m;
    int qh_shift = ((n >> 5) & 3) * 2;    // 0,2,4,6 (2-bit field)
    int v = ((ql[ql_idx] >> ql_shift) & 0x0F)
          | (((qh[qh_idx] >> qh_shift) & 0x03) << 4);
    return v - 32;
}

template <int KDR, bool EXP>
__global__ void __launch_bounds__(256, 3) mmq_raw_nb_bt_q6k_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ W_exp,
    const uint8_t* __restrict__ W_dsc,
    const uint8_t* __restrict__ qa8g,
    const uint8_t* __restrict__ sdag, float* __restrict__ C,
    int nt, int od, int id, int nchunk, int bstride,
    float* __restrict__ Cpart, int ksplit
) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_q6k_sh[];
    // r39: DOUBLE-BUFFERED staging — two copies of every per-kt plane so kt+1's
    // global->smem expansion (the ql+qh recomb) overlaps kt's compute, hiding the
    // B-staging latency that left r38 latency-bound. Layout per buffer b below.
    uint8_t* qa8 = mmq_q6k_sh;                                     // [2][KDR*NBI*32]
    uint8_t* sda_q = qa8 + 2 * KDR * MMQ_NBI * 32;                 // [2][KDR*NBI*4]
    uint8_t* qb_exp = sda_q + 2 * KDR * MMQ_NBI * 4;               // [2][NBJ*KDR*32]
    float2* sds = reinterpret_cast<float2*>(qb_exp + 2 * MMQ_NBJ * KDR * 32);
    const int qa8_stride = KDR * MMQ_NBI * 32;
    const int sdaq_stride = KDR * MMQ_NBI * 4;
    const int qbexp_stride = MMQ_NBJ * KDR * 32;
    const int sds_stride = KDR * MMQ_NBJ;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i0 = blockIdx.x * MMQ_NBI;
    const int j0 = blockIdx.y * MMQ_NBJ;
    const int j0w = warp * 16;
    const int nsb = (id >> 5) >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    // doc 92: K-split, same contract as the q4_K BT kernel — each grid.z
    // slot owns a contiguous tile range, sums it in ascending kt order and
    // writes fp32 partials; the shared reduce kernel sums slots in fixed
    // z order (run-to-run bit-stable).
    const int per = ksplit > 1 ? (nktile + ksplit - 1) / ksplit : nktile;
    const int kt_lo = (int)blockIdx.z * per;
    const int kt_hi = min(kt_lo + per, nktile);

    float sum[32] = {0.0f};   // [g=4][nh=2][l=4]

#define RAW_STAGE_Q6K_BT(kt, b)                                                \
    do {                                                                       \
        uint8_t* qa8b = qa8 + (size_t)(b) * qa8_stride;                        \
        uint8_t* sdaqb = sda_q + (size_t)(b) * sdaq_stride;                    \
        uint8_t* qbexpb = qb_exp + (size_t)(b) * qbexp_stride;                 \
        float2* sdsb = sds + (size_t)(b) * sds_stride;                         \
        /* ---- A: r56 cp.async bulk staging of the pre-transposed qa8/sda --*/\
        /* (r45's mechanism on top of r53: the sync LDG->STS exposed its      */\
        /* global latency at the top of every staging phase; cp.async hands  */\
        /* it to the async unit and the group wait below hides it under the  */\
        /* previous tile's compute. Bytes identical - the plane is always    */\
        /* full: the prepass zero-fills the padded rows.)                    */\
        {                                                                      \
            const size_t qbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_QASZ; \
            for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16;       \
                 off += blockDim.x)                                            \
                gemm_cp16((__half*)(void*)(qa8b + (size_t)off * 16),           \
                          (const __half*)(const void*)(qa8g + qbase            \
                                                       + (size_t)off * 16),    \
                          true);                                               \
            const size_t sbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_SDASZ; \
            for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 4) / 16;        \
                 off += blockDim.x)                                            \
                gemm_cp16((__half*)(void*)(sdaqb + (size_t)off * 16),          \
                          (const __half*)(const void*)(sdag + sbase            \
                                                       + (size_t)off * 16),    \
                          true);                                               \
        }                                                                      \
        /* ---- B: KDR*32-chunk super-block window (half-super at KDR=4) --- */\
        {                                                                      \
            const int sb = ((kt) * KDR) >> 3;                                  \
            const int cbase = ((kt) * KDR) & 7;   /* chunk offset in sb */     \
            if (EXP) {                                                         \
                /* r53 bundle: the ql+qh recomb + -32 centering ran ONCE at    \
                 * registration (expand_q6k_dense -> dense centered-int8 plane \
                 * W_exp: od x id, row stride = id, super-block stride = 256), \
                 * so the staging is a pure cp.async bulk copy (explicit PTX)  \
                 * from W_exp — no recomb ALU, no register round-trip, no      \
                 * ql/qh reads (r44's -10.9% kernel-cycles mechanism), and the \
                 * copy latency hides under compute via the group wait (r45's  \
                 * -10.2% mechanism, applied to the B side). Dense index:      \
                 * W_exp + j*id + sb*256 + cbase*32 (16B-aligned: id is a      \
                 * multiple of 256 on this path). Rows beyond od zero-fill via \
                 * the cp.async src-size qualifier (gemm_cp16 full=0). */      \
                const int nc = (KDR * 32) / 16;   /* 16B chunks per row */     \
                const int ncopy = MMQ_NBJ * nc;                                \
                for (int g = threadIdx.x; g < ncopy; g += blockDim.x) {        \
                    const int jj = g / nc, cc = g % nc;                        \
                    const int j = j0 + jj;                                     \
                    const bool full = (j < od) && (sb < nsb);                  \
                    const uint8_t* src = W_exp + (size_t)j * id                \
                        + (size_t)sb * 256                                     \
                        + (size_t)(cbase * 32 + cc * 16);                      \
                    gemm_cp16(                                                 \
                        (__half*)(void*)(qbexpb + (size_t)jj * (KDR * 32)      \
                                                 + cc * 16),                   \
                        (const __half*)(const void*)src, full);                \
                }                                                              \
                /* r56: the commit moved to the end of RAW_STAGE so ONE group */\
                /* per kt covers A + B + dsc together.                       */\
            } else if ((bstride & 15) == 0) {                                  \
                /* r41: 16-elem group expand via uint4 ql+qh global loads.     \
                 * The padded 224B block stride is 16-aligned, so a group's    \
                 * ql run [it0*64+gg*16 .. +16) and qh run [it0*32+(gg&1)*16..) \
                 * each load in ONE uint4 instead of 16 per-byte LDGs — cuts   \
                 * the B-expand global-load instruction count and its L1TEX    \
                 * scoreboard exposure ~16x. Element-for-element identical to  \
                 * expand_q6_elem (gate-1 validator check). Per group (gg=g&3, \
                 * cbase in {0,2,4,6}): it0=(cbase>>2)&1; qsh=((cbase>>1)&1)*4; \
                 * qh_shift=qsh+((gg>>1)&1)*2; ql_shift=qsh.                    \
                 * v[b] = ((ql[it0*64+gg*16+b]>>qsh)&0xF)                      \
                 *        | ((qh[it0*32+(gg&1)*16+b]>>qh_shift)&3)<<4) - 32. */\
                const int it0 = (cbase >> 2) & 1;                              \
                const int qsh = ((cbase >> 1) & 1) * 4;                        \
                const int ng = MMQ_NBJ * (KDR * 32) / 16;                      \
                for (int g = threadIdx.x; g < ng; g += blockDim.x) {           \
                    const int jj = g / 4, gg = g & 3;                          \
                    const int j = j0 + jj;                                     \
                    uint8_t* out = qbexpb + (size_t)jj * (KDR * 32) + gg * 16; \
                    if (j < od && sb < nsb) {                                  \
                        const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)\
                            + (size_t)sb * bstride;                            \
                        const uint4 qlv = *(const uint4*)(blk + it0*64 + gg*16);\
                        const uint4 qhv = *(const uint4*)(blk + 128 + it0*32    \
                                          + (gg & 1) * 16);                    \
                        const int qhs = qsh + ((gg >> 1) & 1) * 2;             \
                        const uint32_t qx = qlv.x, qy = qlv.y, qz = qlv.z, qw = qlv.w;\
                        const uint32_t hx = qhv.x, hy = qhv.y, hz = qhv.z, hw = qhv.w;\
                        _Pragma("unroll")                                      \
                        for (int e = 0; e < 16; e++) {                         \
                            const int sidx = e >> 2;                           \
                            const int sh = (e & 3) * 8;                        \
                            const uint32_t qsel = (sidx == 0) ? qx : (sidx == 1) ? qy\
                                              : (sidx == 2) ? qz : qw;         \
                            const uint32_t hsel = (sidx == 0) ? hx : (sidx == 1) ? hy\
                                              : (sidx == 2) ? hz : hw;         \
                            const uint8_t qb_ = (uint8_t)((qsel >> sh) & 0xFF);\
                            const uint8_t hb_ = (uint8_t)((hsel >> sh) & 0xFF);\
                            out[e] = (uint8_t)((((qb_ >> qsh) & 0xF)           \
                                | (((hb_ >> qhs) & 3) << 4)) - 32);            \
                        }                                                      \
                    } else {                                                   \
                        _Pragma("unroll")                                      \
                        for (int e = 0; e < 16; e++) out[e] = 0;               \
                    }                                                          \
                }                                                              \
            } else {                                                           \
                for (int x = threadIdx.x; x < MMQ_NBJ * (KDR * 32); x += blockDim.x) {\
                    const int jj = x / (KDR * 32), bec = x % (KDR * 32);       \
                    const int j = j0 + jj;                                     \
                    const int elem = cbase * 32 + bec;   /* super-block elem */\
                    int v = 0;                                                 \
                    if (j < od && sb < nsb) {                                  \
                        const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride) \
                            + (size_t)sb * bstride;                            \
                        v = expand_q6_elem(blk, blk + 128, elem);              \
                    }                                                          \
                    qbexpb[(size_t)jj * (KDR * 32) + bec] = (uint8_t)v;         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* ---- B: dsc pair (d*sc[2c%16], d*sc[(2c+1)%16]) per (chunk,row) ----*/\
        /* r56: W_dsc f32 plane (registration-time precompute; chunk-major    */\
        /* layout plane[c*od + j] = float2(d*sc0, d*sc1)) turns the scalar    */\
        /* blk[192+..]/blk[208] loads + I2F (a leading r43 residual stall     */\
        /* post-r53) into a contiguous 16-B cp.async stream inside the same   */\
        /* per-kt commit group. Null plane (raw weights / alloc failure /     */\
        /* odd od) = the r41 scalar path, byte-identical.                     */\
        {                                                                      \
            const int c0d = (kt) * KDR;                                        \
            if (W_dsc != nullptr) {                                            \
                const int nc2 = MMQ_NBJ / 2; /* 16-B chunks (2 float2) per kd */\
                for (int g = threadIdx.x; g < KDR * nc2; g += blockDim.x) {    \
                    const int kdd = g / nc2, m = g % nc2;                      \
                    const int j = j0 + 2 * m;                                  \
                    /* od even (registration gate) => a pair is either fully   */\
                    /* valid or fully beyond od (src-size zero-fill).          */\
                    const bool full = (j + 1 < od);                            \
                    gemm_cp16(                                                 \
                        (__half*)(void*)(sdsb + (size_t)kdd * MMQ_NBJ          \
                                                  + 2 * m),                    \
                        (const __half*)(const void*)(W_dsc                     \
                            + ((size_t)(c0d + kdd) * (size_t)od + (size_t)j) * 8), \
                        full);                                                 \
                }                                                              \
            } else {                                                           \
        for (int x = threadIdx.x; x < MMQ_NBJ * KDR; x += blockDim.x) {        \
            const int r = x % MMQ_NBJ, kd = x / MMQ_NBJ;                       \
            const int j = j0 + r, c = (kt) * KDR + kd;                         \
            float dsc0 = 0.0f, dsc1 = 0.0f;                                    \
            if (j < od && c < nchunk) {                                        \
                const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)   \
                    + (size_t)(c >> 3) * bstride;                              \
                const float d = h2f(*(const uint16_t*)(blk + 208));            \
                const int s0 = 2 * (c & 7);                                    \
                dsc0 = d * (float)(int8_t)blk[192 + s0];                       \
                dsc1 = d * (float)(int8_t)blk[192 + s0 + 1];                   \
            }                                                                  \
            sdsb[(size_t)kd * MMQ_NBJ + r] = make_float2(dsc0, dsc1);          \
        }                                                                      \
            }                                                                  \
        }                                                                      \
        gemm_cp_commit();                                                      \
    } while (0)

    const unsigned l12m = (unsigned)(lane & 12) * 32;
    const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                             ^ ((lane >> 2) & 3)) << 4;
    unsigned G[4];
    #pragma unroll
    for (int g = 0; g < 4; g++)
        G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);

    RAW_STAGE_Q6K_BT(kt_lo, 0);
    __syncthreads();

    int buf = 0;
    for (int kt = kt_lo; kt < kt_hi; ++kt, buf ^= 1) {
        // Overlap kt+1's global->smem staging with kt's compute: stage into the
        // OTHER buffer (buf^1) while reading buffer buf (the mmq_nt<7,2> pipeline).
        if (kt + 1 < kt_hi) {
            RAW_STAGE_Q6K_BT(kt + 1, buf ^ 1);
            // r53 (EXP) / r56: two groups are pending (kt's and kt+1's); wait
            // until only kt+1's remains — group(kt), the cp.async copies (r56:
            // A + dsc too, not just B) for `buf`, has landed, while buf^1's
            // copies stay in flight under kt's compute (in-order group
            // completion).
            gemm_cp_wait1();
        } else {
            gemm_cp_wait0();  // last tile: drain every outstanding group
        }
        __syncthreads();  // r53/r56: cross-thread visibility of the kt
                          // buffer's async copies before compute

        const uint8_t* qa8c = qa8 + (size_t)buf * qa8_stride;
        const uint32_t* sdaqc = reinterpret_cast<const uint32_t*>(sda_q + (size_t)buf * sdaq_stride);
        const uint8_t* qbexpc = qb_exp + (size_t)buf * qbexp_stride;
        const float2* sdsc = sds + (size_t)buf * sds_stride;

        #pragma unroll
        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const uint8_t* qat = qa8c + (size_t)kd * MMQ_NBI * 32;

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
            // B-frag: straight int8 read from the expanded plane. Each mma.k16
            // uses b[nh][0] (k=0..15, sub 2c) or b[nh][1] (k=16..31, sub 2c+1).
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const int jj = j0w + nh * 8 + (lane >> 2);
                const uint8_t* qs = qbexpc + (size_t)jj * (KDR * 32) + (size_t)kd * 32;
                b[nh][0] = *(const int*)(qs + (lane & 3) * 4);
                b[nh][1] = *(const int*)(qs + 16 + (lane & 3) * 4);
            }
            int clow[4][2][4], chigh[4][2][4];
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) { clow[g][nh][l] = 0; chigh[g][nh][l] = 0; }
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++) {
                    mmq_mma_k16(clow[g][nh], a[g], b[nh][0]);       // low-16, sub 2c
                    mmq_mma_k16(chigh[g][nh], a[g] + 2, b[nh][1]);  // high-16, sub 2c+1
                }

            const uint32_t* sda_blk = sdaqc + (size_t)kd * MMQ_NBI
                                      + (size_t)(lane >> 2) * 4;
            const uint4 s0 = *(const uint4*)(sda_blk);
            const uint4 s1 = *(const uint4*)(sda_blk + 32);
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                float da_q[2];
                const unsigned w0 = g == 0 ? s0.x : (g == 1 ? s0.z : (g == 2 ? s1.x : s1.z));
                const unsigned w1 = g == 0 ? s0.y : (g == 1 ? s0.w : (g == 2 ? s1.y : s1.w));
                da_q[0] = h2f((unsigned short)(w0 & 0xFFFF));
                da_q[1] = h2f((unsigned short)(w1 & 0xFFFF));
                #pragma unroll
                for (int nh = 0; nh < 2; nh++) {
                    const float4 sc4 = *(const float4*)(sdsc
                        + (size_t)kd * MMQ_NBJ + j0w + nh * 8 + (lane & 3) * 2);
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        const float da = da_q[l >> 1];
                        const float dsc0 = (l & 1) ? sc4.z : sc4.x;
                        const float dsc1 = (l & 1) ? sc4.w : sc4.y;
                        const int idx = (g * 2 + nh) * 4 + l;
                        // two separate accumulations (per-16 sub-block), matching
                        // the host reference's per-half += (see mmq_nt<7,2>).
                        sum[idx] += da * dsc0 * (float)clow[g][nh][l];
                        sum[idx] += da * dsc1 * (float)chigh[g][nh][l];
                    }
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
                    if (ksplit == 1)
                        C[(size_t)i * od + j] = sum[(g * 2 + nh) * 4 + l];
                    else
                        Cpart[((size_t)blockIdx.z * nt + i) * od + j] =
                            sum[(g * 2 + nh) * 4 + l];
            }
#undef RAW_STAGE_Q6K_BT
#endif // __CUDA_ARCH__ >= 800
}
// P6 r39: q6_K BT launcher (double-buffered). The B tile is the EXPANDED centered
// int8 plane (KDR*32 B/row) + the dsc float2 scale plane. KDR=2 stages a QUARTER
// super-block per kt, but two buffers (2xA + 2xB) pipeline kt+1's global->smem
// expansion under kt's compute while keeping the footprint at 29,696 B -> 2 blocks/SM
// (the r38 figure; full-double-buffer at KDR=4 would be 59,392 B -> 1 block/SM and
// was excluded on r38's occupancy evidence). r53: w_exp != 0 selects the
// pre-expanded-B instantiation (B staging = cp.async bulk copy from the dense
// W_exp plane); w_exp == 0 keeps the r41 in-kernel expand. Returns 0 (clean
// fallback to the generic mmq_nt<7,2>) on any cap/mismatch.
extern "C" int launch_mmq_raw_nb_bt_q6k_nt(
    int type_id, const uint8_t* w, const uint8_t* w_exp, const uint8_t* w_dsc,
    const uint8_t* qa8g, const uint8_t* sdag, float* c, int nt, int od, int id,
    int nchunk, int bstride, cudaStream_t stream, int kd, float* cpart,
    int ksplit
) {
    (void)type_id;
    if (kd != 8) return 0;
    if (qa8g == 0 || sdag == 0) return 0;
    constexpr int KDR = 2;
    const int smem = 2 * KDR * MMQ_NBI * 32   // qa8  (double-buffered)
                   + 2 * KDR * MMQ_NBI * 4    // sda_q (double-buffered)
                   + 2 * MMQ_NBJ * KDR * 32   // qb_exp (double-buffered)
                   + 2 * KDR * MMQ_NBJ * 8;   // sds  (double-buffered)
    // doc 92: refine the requested ksplit so every z-slot owns a non-empty
    // tile range (KDR=2 tiles here)
    int ks = ksplit > 1 ? ksplit : 1;
    {
        const int nktile_all = (nchunk + 1) / 2;
        if (ks > nktile_all) ks = nktile_all;
        if (ks < 1) ks = 1;
        const int per = (nktile_all + ks - 1) / ks;
        ks = (nktile_all + per - 1) / per;
    }
    dim3 grid((nt + MMQ_NBI - 1) / MMQ_NBI, (od + MMQ_NBJ - 1) / MMQ_NBJ, ks);
    // r53: EXP is a template constant, so each instantiation keeps only its
    // own B path (the cp.async copy vs the r41 recomb) — no runtime branch.
    const bool exp = w_exp != 0;
    // #147: the opt-in's own return value decides (see the q4_K launcher).
    const char* const kname =
        exp ? "mmq_raw_nb_bt_q6k_kernel<2,true>" : "mmq_raw_nb_bt_q6k_kernel<2,false>";
    const void* kfn = exp ? reinterpret_cast<const void*>(&mmq_raw_nb_bt_q6k_kernel<KDR, true>)
                          : reinterpret_cast<const void*>(&mmq_raw_nb_bt_q6k_kernel<KDR, false>);
    if (!minfer_smem_optin("attr:mmq_raw_nb_bt_q6k", kname, kfn, smem)) return 0;
    minfer_launch_prelude("launch:mmq_raw_nb_bt_q6k", kname);
    if (exp) {
        mmq_raw_nb_bt_q6k_kernel<KDR, true><<<grid, 256,
                                              minfer_launch_smem("launch:mmq_raw_nb_bt_q6k", smem),
                                              stream>>>(w, w_exp, w_dsc, qa8g, sdag, c, nt, od, id,
                                                        nchunk, bstride, cpart, ks);
    } else {
        mmq_raw_nb_bt_q6k_kernel<KDR, false><<<grid, 256,
                                               minfer_launch_smem("launch:mmq_raw_nb_bt_q6k", smem),
                                               stream>>>(w, w_exp, w_dsc, qa8g, sdag, c, nt, od, id,
                                                         nchunk, bstride, cpart, ks);
    }
    if (!minfer_launch_ok_opt("launch:mmq_raw_nb_bt_q6k", kname)) return 0;
    if (ks > 1) {
        const int total = nt * od;
        // #162: the reduce is a SEPARATE site token so the gate can arm it
        // alone — arming the kernel's token refuses the launch and returns 0
        // before this line, which would make the reduce site unreachable.
        minfer_launch_prelude("launch:mmq_raw_nb_bt_q6k_ksplit", "mmq_ksplit_reduce_kernel");
        mmq_ksplit_reduce_kernel<<<(total + 255) / 256, minfer_launch_block("launch:mmq_raw_nb_bt_q6k_ksplit", 256), 0, stream>>>(
            cpart, c, total, ks);
        if (!minfer_launch_ok_opt("launch:mmq_raw_nb_bt_q6k_ksplit", "mmq_ksplit_reduce_kernel")) return 0;
    }
    return 1;
}



// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// This file owns the template instantiations, so the address-taking must
// happen here (a cross-TU template reference is nvcc #20280-D and can fail
// to link).
extern "C" void minfer_prewarm_mmq_bt_q6k_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, (mmq_raw_nb_bt_q6k_kernel<2, true>));
    MINFER_PREWARM_ONE(a, (mmq_raw_nb_bt_q6k_kernel<2, false>));
}
