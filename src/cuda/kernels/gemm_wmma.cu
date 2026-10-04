// src/cuda/kernels/gemm_wmma.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"

extern "C" {

// ─── 8m: prefill dequant-to-f16 + wmma HGEMM ────────────────────────────
// Prefill (nt >= 16) routes quantized matmuls through ONE tiled
// tensor-core GEMM instead of the decode-shaped kernels whose
// grid.y = nt re-streamed the whole weight matrix once per token
// (7B q4_k_m @2K: 30.7 tok/s vs llama.cpp MMQ 3401). Weights are
// dequantized to f16 once per call into a scratch buffer, activations
// converted to f16, then C[nt, od] = A[nt, id] · B[od, id]^T via
// 16x16x16 wmma with f32 accumulation. Gated on id % 32 == 0 (block
// math + 16B-aligned uint4 tile loads).

// type_id mapping for launch_dequant_f16 (Rust side passes it):
// 0=q8_0 1=q4_0 2=q4_1 3=q5_0 4=q5_1 5=q4_K 6=q5_K 7=q6_K

__global__ void dequant_q8_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 34;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    const int8_t* q = (const int8_t*)(blk + 2);
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) o[i] = __float2half(d * float(q[i]));
}

__global__ void dequant_q4_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 18;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    const uint8_t* q = blk + 2;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    // minfer Q4_0 stores round(v/d) + 8 (same -8 offset as the matmuls).
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[i] = __float2half(d * (float(q[i] & 0x0F) - 8.0f));
        o[i + 16] = __float2half(d * (float(q[i] >> 4) - 8.0f));
    }
}

__global__ void dequant_q4_1_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 20;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    const uint8_t* q = blk + 4;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[i] = __float2half(d * float(q[i] & 0x0F) + m);
        o[i + 16] = __float2half(d * float(q[i] >> 4) + m);
    }
}

__global__ void dequant_q5_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 22;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    // 22-byte blocks are only 2-byte aligned: assemble qh from two u16
    // loads — a u32 load at blk+2 misaligns for even g
    // (cudaErrorMisalignedAddress 716; latent until 8p's bitparity test
    // exercised Q5_0 prefill GEMM for the first time).
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
    const uint8_t* qs = blk + 6;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1) - 16.0f;
        float hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1) - 16.0f;
        o[j] = __float2half(d * lo);
        o[j + 16] = __float2half(d * hi);
    }
}

__global__ void dequant_q5_1_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 24;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint32_t qh = *reinterpret_cast<const uint32_t*>(blk + 4);
    const uint8_t* qs = blk + 8;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1);
        float hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1);
        o[j] = __float2half(d * lo + m);
        o[j + 16] = __float2half(d * hi + m);
    }
}

__global__ void dequant_q4_k_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nsub = id / 32; // 8 sub-blocks per 256 super-block
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nsub) return;
    int row = (int)(g / nsub), s = (int)(g % nsub);
    int sp = s / 8, sub = s % 8;
    const uint8_t* blk = w + ((long long)row * (id / 256) + sp) * Q4KB;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    int j = sub / 2, half = sub % 2;
    const uint8_t* q = blk + 16 + j * 32;
    __half* o = out + (long long)row * id + s * 32;
    float ds = d * float(scb), dmm = dmin * float(mb);
    #pragma unroll
    for (int l = 0; l < 32; l++) {
        float nib = half ? float(q[l] >> 4) : float(q[l] & 0x0F);
        o[l] = __float2half(ds * nib - dmm);
    }
}

__global__ void dequant_q5_k_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nsub = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nsub) return;
    int row = (int)(g / nsub), sidx = (int)(g % nsub);
    int sp = sidx / 8, sub = sidx % 8;
    const uint8_t* blk = w + ((long long)row * (id / 256) + sp) * Q5KB;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    int ci = sub >> 1, hi = sub & 1;
    const uint8_t* q4 = blk + 48 + ci * 32;
    const uint8_t* qh = blk + 16;
    __half* o = out + (long long)row * id + sidx * 32;
    float ds = d * float(scb), dmm = dmin * float(mb);
    #pragma unroll
    for (int l = 0; l < 32; l++) {
        float nib = hi ? float(q4[l] >> 4) : float(q4[l] & 0x0F);
        float wv = nib + 16.0f * float((qh[l] >> sub) & 1);
        o[l] = __float2half(ds * wv - dmm);
    }
}

__global__ void dequant_q6_k_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out,
    int od, int id, int block_stride
) {
    int nsub = id / 16; // 16-element units, 16 per 256 super-block
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nsub) return;
    int row = (int)(g / nsub), s = (int)(g % nsub);
    int sp = s / 16, sub = s % 16;
    const uint8_t* blk = w + ((long long)row * (id / 256) + sp) * block_stride;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    int n = sub / 8, rem = sub % 8, tt = rem / 2, gq = rem % 2;
    int ql_off = n * 64 + (tt % 2) * 32 + gq * 16;
    int qh_off = n * 32 + gq * 16;
    int sc_idx = n * 8 + tt * 2 + gq;
    __half* o = out + (long long)row * id + sp * 256 + n * 128 + tt * 32 + gq * 16;
    float dsc = d * float(sc[sc_idx]);
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        int nib = (tt < 2) ? (ql[ql_off + r] & 0x0F) : (ql[ql_off + r] >> 4);
        int q2 = (qh[qh_off + r] >> (tt * 2)) & 3;
        o[r] = __float2half(dsc * float((nib | (q2 << 4)) - 32));
    }
}

// f32 activations -> f16 (one kernel, elementwise).
__global__ void convert_f32_f16_kernel(
    const float* __restrict__ x, __half* __restrict__ out, long long n
) {
    // P1: 8 elements per thread (2x float4 -> 4x half2) instead of one
    // scalar element — 8x fewer transactions on the same traffic.
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base + 7 < n) {
        float4 a = *reinterpret_cast<const float4*>(x + base);
        float4 b = *reinterpret_cast<const float4*>(x + base + 4);
        __half2* o = reinterpret_cast<__half2*>(out + base);
        o[0] = __floats2half2_rn(a.x, a.y);
        o[1] = __floats2half2_rn(a.z, a.w);
        o[2] = __floats2half2_rn(b.x, b.y);
        o[3] = __floats2half2_rn(b.z, b.w);
    } else if (base < n) {
        for (long long i = base; i < n; i++) out[i] = __float2half(x[i]);
    }
}

// 8m②: cp.async global→shared staging (sm_80+). The synchronous load
// stalled every warp on the L2 round trip each 32-k step (~31 TFLOPS
// measured); async copies overlap the k+32 tile fetch with the k compute.
__device__ __forceinline__ void gemm_load_tile_sync(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* As, __half* Bs,
    int n0, int m0, int k0, int nt, int od, int id
) {
    int r = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id;
    int n = n0 + r;
    if (n < nt && k_ok) {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) =
            *reinterpret_cast<const uint4*>(A + (long long)n * id + k0 + c4);
    } else {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
    int m = m0 + r;
    if (m < od && k_ok) {
        *reinterpret_cast<uint4*>(Bs + r * 32 + c4) =
            *reinterpret_cast<const uint4*>(B + (long long)m * id + k0 + c4);
    } else {
        *reinterpret_cast<uint4*>(Bs + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
}

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void gemm_load_tile_async(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* As, __half* Bs,
    int n0, int m0, int k0, int nt, int od, int id
) {
    // 256 threads: 64 rows x 4 aligned 16B chunks (8 halves each).
    int r = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id; // id % 8 == 0 gate keeps chunks inside the row
    int n = n0 + r;
    gemm_cp16(As + r * 32 + c4, A + (long long)n * id + k0 + c4, n < nt && k_ok);
    int m = m0 + r;
    gemm_cp16(Bs + r * 32 + c4, B + (long long)m * id + k0 + c4, m < od && k_ok);
}

// P2: B-panel loader for TM-row od tiles (TM > 64 needs a second pass of
// 64 rows; A always stages TN = 64 rows).
__device__ __forceinline__ void gemm_load_b_async(
    const __half* __restrict__ B, __half* Bs,
    int m0, int k0, int od, int id, int tm
) {
    int r0 = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id;
    for (int rep = 0; rep < tm / 64; rep++) {
        int r = r0 + rep * 64;
        int m = m0 + r;
        gemm_cp16(Bs + r * 32 + c4, B + (long long)m * id + k0 + c4, m < od && k_ok);
    }
}

// synchronous B loader for pre-sm80 builds lives AFTER the sm_80 guard
// (sm_75 is still a build target and compiles the fallback paths).
#endif // __CUDA_ARCH__ >= 800

__device__ __forceinline__ void gemm_load_b_sync(
    const __half* __restrict__ B, __half* Bs,
    int m0, int k0, int od, int id, int tm
) {
    int r0 = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id;
    for (int rep = 0; rep < tm / 64; rep++) {
        int r = r0 + rep * 64;
        int m = m0 + r;
        if (m < od && k_ok) {
            *reinterpret_cast<uint4*>(Bs + r * 32 + c4) =
                *reinterpret_cast<const uint4*>(B + (long long)m * id + k0 + c4);
        } else {
            *reinterpret_cast<uint4*>(Bs + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// sync B loader lives outside the sm_80 guard: the pre-sm80 fallback paths
// of gemm_f16_nt_kernel_t reference it (sm_75 is still a build target).

// A template cannot have C linkage: pause the extern "C" block around the
// templated GEMM.
}

// AF32 A staging, mirror scheme: cp.async the F32 k-tile into a smem
// mirror (16B = 4 f32 chunks; async again — the v1 synchronous global
// loads stalled every k-tile and measured -8%), then convert
// smem->smem f32->f16 right before compute. Requires id % 8 == 0.
template <int KS>
__device__ __forceinline__ void gemm_mirror_a32(
    float* Am, int bbuf, const float* A32, int TN,
    int n0, int k0, int nt, int id, int tid
) {
#if __CUDA_ARCH__ >= 800
    for (int c = tid; c < TN * KS / 4; c += blockDim.x) {
        int r = (c * 4) / KS, d = (c * 4) % KS;
        int n = n0 + r;
        gemm_cp16(reinterpret_cast<__half*>(Am + bbuf * TN * KS + r * KS + d),
                  reinterpret_cast<const __half*>(A32 + (long long)n * id + k0 + d),
                  n < nt && k0 + d < id);
    }
#endif // pre-sm80 callers use the inline synchronous staging instead
}

// P4: stage the A (TN rows) and B (TM rows) k-tiles [k0, k0+KS) into the
// double-buffered dynamic-smem regions. Chunk-linear mapping: chunk c
// covers 8 consecutive halves, r = c*8/KS, d = c*8%KS (KS % 8 == 0).
// AF32: A arrives as f32 activations and converts on stage (P6), so the
// separate convert_f32_f16 pass disappears.
template <int TM, int KS, int TN, bool AF32 = false>
__device__ __forceinline__ void gemm_stage_ab(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* As, __half* Bs, float* Am, int bbuf,
    int n0, int m0, int k0, int nt, int od, int id, int tid
) {
#if __CUDA_ARCH__ >= 800
    if (AF32) {
        gemm_mirror_a32<KS>(Am, bbuf, reinterpret_cast<const float*>(A), TN,
                            n0, k0, nt, id, tid);
    } else {
        for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
            int r = (c * 8) / KS, d = (c * 8) % KS;
            int n = n0 + r;
            gemm_cp16(As + bbuf * TN * KS + r * KS + d,
                      A + (long long)n * id + k0 + d, n < nt && k0 + d < id);
        }
    }
    for (int c = tid; c < TM * KS / 8; c += blockDim.x) {
        int r = (c * 8) / KS, d = (c * 8) % KS;
        int m = m0 + r;
        gemm_cp16(Bs + bbuf * TM * KS + r * KS + d,
                  B + (long long)m * id + k0 + d, m < od && k0 + d < id);
    }
#endif // pre-sm80 callers use the inline synchronous staging instead
}

// C[nt, od] = A[nt, id] · B[od, id]^T. 64 x TM output tiles (TM = 64
// baseline, 128 halves the B-panel re-reads through L2 and the per-k-step
// barrier count), k-step 32, double-buffered shared staging, 8 warps (each
// owns 32 nt rows x TM/4 od cols as 2 x TM/64 f32 fragment pairs). f32
// accumulation. Tails: nt/od masked at store, k-tail zero-filled (id % 8
// == 0 keeps the uint4 chunk loads aligned).
template <int TM, int KS, bool AF32 = false>
__global__ void gemm_f16_nt_kernel_t(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, int nt, int od, int id
) {
    using namespace nvcuda;
    constexpr int TN = 64;
    constexpr int ODC = TM / 64;  // od 16-col fragments per warp row-half
    constexpr int KHC = KS / 32;  // k-half PAIRS per staged tile (each loop iteration consumes fa[0]+fa[2] = 2 k-halves)
    // dynamic smem: TM=128 + KS=64 needs 56KB — over the 48KB static cap
    extern __shared__ __align__(16) uint8_t smem_raw[];
    __half* As = reinterpret_cast<__half*>(smem_raw); // 2 x TN*KS halves
    // AF32 adds a 2 x TN*KS f32 mirror right after As (cp.async target)
    float* Am = reinterpret_cast<float*>(As + 2 * TN * KS);
    __half* Bs = reinterpret_cast<__half*>(
        smem_raw + 2 * TN * KS * 2 + (AF32 ? 2 * TN * KS * 4 : 0)); // 2 x TM*KS
    float* Cs = reinterpret_cast<float*>(Bs + 2 * TM * KS); // NW x 256 f32

    const int tid = threadIdx.x;
    const int NW = blockDim.x >> 5;   // warps: 8 for TM<=128, 16 for TM=256
    int warp = tid >> 5;
    int wm = warp >> 1;               // od chunk of this warp (TM/(NW/2) cols)
    int wn = warp & 1;                // nt sub-tile: 2 x 32 rows
    // blockIdx.x = nt tile, blockIdx.y = od tile: consecutive blocks share
    // the same od-tile's B panel (64 rows x id f16, ~0.5MB) in L2, so the
    // f16 weight matrix streams from DRAM ~once instead of nt/64 times.
    int m0 = blockIdx.y * TM;
    int n0 = blockIdx.x * TN;
    const int ob = wm * (TM / (NW >> 1)); // od row base of this warp's chunk

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2][ODC];
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int oc = 0; oc < ODC; oc++) wmma::fill_fragment(fc[j][oc], 0.0f);

    int buf = 0;
#if __CUDA_ARCH__ >= 800
    // stage A (64 rows) + B (TM rows) for k = 0
    gemm_stage_ab<TM, KS, TN, AF32>(A, B, As, Bs, Am, 0, n0, m0, 0, nt, od, id, tid);
    gemm_cp_commit();
#else
    {
        if (AF32) { // pre-sm80: synchronous global->smem convert (no cp.async)
            for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                int r = (c * 8) / KS, d = (c * 8) % KS;
                int n = n0 + r;
                uint4 h = make_uint4(0u, 0u, 0u, 0u);
                if (n < nt && d < id) {
                    const float4 f0 = *reinterpret_cast<const float4*>(
                        reinterpret_cast<const float*>(A) + (long long)n * id + d);
                    const float4 f1 = *reinterpret_cast<const float4*>(
                        reinterpret_cast<const float*>(A) + (long long)n * id + d + 4);
                    __half2 p0 = __floats2half2_rn(f0.x, f0.y);
                    __half2 p1 = __floats2half2_rn(f0.z, f0.w);
                    __half2 p2 = __floats2half2_rn(f1.x, f1.y);
                    __half2 p3 = __floats2half2_rn(f1.z, f1.w);
                    h = make_uint4(*reinterpret_cast<unsigned*>(&p0),
                                   *reinterpret_cast<unsigned*>(&p1),
                                   *reinterpret_cast<unsigned*>(&p2),
                                   *reinterpret_cast<unsigned*>(&p3));
                }
                *reinterpret_cast<uint4*>(As + r * KS + d) = h;
            }
        } else {
            for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                int r = (c * 8) / KS, d = (c * 8) % KS;
                int n = n0 + r;
                if (n < nt && d < id) {
                    *reinterpret_cast<uint4*>(As + r * KS + d) =
                        *reinterpret_cast<const uint4*>(A + (long long)n * id + d);
                } else {
                    *reinterpret_cast<uint4*>(As + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        for (int c = tid; c < TM * KS / 8; c += blockDim.x) {
            int r = (c * 8) / KS, d = (c * 8) % KS;
            int m = m0 + r;
            if (m < od && d < id) {
                *reinterpret_cast<uint4*>(Bs + r * KS + d) =
                    *reinterpret_cast<const uint4*>(B + (long long)m * id + d);
            } else {
                *reinterpret_cast<uint4*>(Bs + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();
    }
#endif
    for (int k = 0; k < id; k += KS, buf ^= 1) {
#if __CUDA_ARCH__ >= 800
        if (k + KS < id) {
            gemm_stage_ab<TM, KS, TN, AF32>(A, B, As, Bs, Am, buf ^ 1, n0,
                                            m0, k + KS, nt, od, id, tid);
            gemm_cp_commit();
            // wait until the CURRENT tile landed (one group may stay in flight)
            gemm_cp_wait1();
        } else {
            gemm_cp_wait0();
        }
        __syncthreads();
#else
        if (k + KS < id) {
            if (AF32) { // pre-sm80: synchronous convert into the next buffer
                for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                    int r = (c * 8) / KS, d = (c * 8) % KS;
                    int n = n0 + r;
                    uint4 h = make_uint4(0u, 0u, 0u, 0u);
                    if (n < nt && k + KS + d < id) {
                        const float4 f0 = *reinterpret_cast<const float4*>(
                            reinterpret_cast<const float*>(A) + (long long)n * id + k + KS + d);
                        const float4 f1 = *reinterpret_cast<const float4*>(
                            reinterpret_cast<const float*>(A) + (long long)n * id + k + KS + d + 4);
                        __half2 p0 = __floats2half2_rn(f0.x, f0.y);
                        __half2 p1 = __floats2half2_rn(f0.z, f0.w);
                        __half2 p2 = __floats2half2_rn(f1.x, f1.y);
                        __half2 p3 = __floats2half2_rn(f1.z, f1.w);
                        h = make_uint4(*reinterpret_cast<unsigned*>(&p0),
                                       *reinterpret_cast<unsigned*>(&p1),
                                       *reinterpret_cast<unsigned*>(&p2),
                                       *reinterpret_cast<unsigned*>(&p3));
                    }
                    *reinterpret_cast<uint4*>(As + (buf ^ 1) * TN * KS + r * KS + d) = h;
                }
            } else {
                for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                    int r = (c * 8) / KS, d = (c * 8) % KS;
                    int n = n0 + r;
                    if (n < nt && k + KS + d < id) {
                        *reinterpret_cast<uint4*>(As + (buf ^ 1) * TN * KS + r * KS + d) =
                            *reinterpret_cast<const uint4*>(A + (long long)n * id + k + KS + d);
                    } else {
                        *reinterpret_cast<uint4*>(As + (buf ^ 1) * TN * KS + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
            for (int c = tid; c < TM * KS / 8; c += blockDim.x) {
                int r = (c * 8) / KS, d = (c * 8) % KS;
                int m = m0 + r;
                if (m < od && k + KS + d < id) {
                    *reinterpret_cast<uint4*>(Bs + (buf ^ 1) * TM * KS + r * KS + d) =
                        *reinterpret_cast<const uint4*>(B + (long long)m * id + k + KS + d);
                } else {
                    *reinterpret_cast<uint4*>(Bs + (buf ^ 1) * TM * KS + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();
#endif
        if (AF32) {
            // mirror[buf] landed with the wait above: convert smem->smem
            for (int c = tid; c < TN * KS / 4; c += blockDim.x) {
                int r = (c * 4) / KS, d = (c * 4) % KS;
                float4 f = *reinterpret_cast<float4*>(
                    &Am[buf * TN * KS + r * KS + d]);
                __half2 p0 = __floats2half2_rn(f.x, f.y);
                __half2 p1 = __floats2half2_rn(f.z, f.w);
                *reinterpret_cast<__half2*>(As + buf * TN * KS + r * KS + d) = p0;
                *reinterpret_cast<__half2*>(As + buf * TN * KS + r * KS + d + 2) = p1;
            }
            __syncthreads();
        }
        // fa[n-half][k-half]; fb[k-half] per od chunk. Both k halves of each
        // 32-slice must accumulate (the v1 bug: only the first 16 k's were
        // multiplied); fb's k offset is +16 ELEMENTS (one k-half), not +16
        // rows.
#pragma unroll
        for (int kh = 0; kh < KHC; kh++) {
            wmma::load_matrix_sync(fa[0], &As[buf * TN * KS + wn * 32 * KS + kh * 32], KS);
            wmma::load_matrix_sync(fa[1], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32], KS);
            wmma::load_matrix_sync(fa[2], &As[buf * TN * KS + wn * 32 * KS + kh * 32 + 16], KS);
            wmma::load_matrix_sync(fa[3], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32 + 16], KS);
#pragma unroll
            for (int oc = 0; oc < ODC; oc++) {
                wmma::load_matrix_sync(fb[0], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32], KS);
                wmma::load_matrix_sync(fb[1], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32 + 16], KS);
                wmma::mma_sync(fc[0][oc], fa[0], fb[0], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[1], fb[0], fc[1][oc]);
                wmma::mma_sync(fc[0][oc], fa[2], fb[1], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[3], fb[1], fc[1][oc]);
            }
        }
        __syncthreads();
    }

    int lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
        for (int oc = 0; oc < ODC; oc++) {
            wmma::store_matrix_sync(Cs + warp * 256, fc[j][oc], 16, wmma::mem_row_major);
            int nb = n0 + wn * 32 + j * 16, mb = m0 + ob + oc * 16;
            for (int e = lane; e < 256; e += 32) {
                int n = nb + (e >> 4), m = mb + (e & 15);
                if (n < nt && m < od)
                    C[(long long)n * od + m] = Cs[warp * 256 + (e >> 4) * 16 + (e & 15)];
            }
        }
    }
}

// ─── prefill-GEMM dynamic shared memory: one formula, checked opt-ins ────
//
// The dynamic-smem requirement of one `gemm_f16_nt_kernel_t` instantiation is
// the kernel's own byte layout (see its definition): the As tile (2*TN*KS
// halves), the AF32 f32 mirror Am (2*TN*KS floats, AF32 only), the Bs tile
// (2*TM*KS halves) and the Cs store (NW*256 floats). TN = 64 and NW =
// blockDim.x/32 = 8 at every launch site (256 threads). The launcher
// (`launch_gemm_f16`) and the production opt-in (`gemm_smem_optin` below, called
// from the launcher's `GEMM_ONE`) MUST read the same number, so this formula is
// the single source. (Issue #145, history: the **eager** sweep removed by #218
// used a stale copy of this formula that assumed a 512-thread TM=256 launch and
// always added the AF32 mirror, so it asked for 131072 B for
// `gemm_f16_nt_kernel_t<256,64,true>` — more than the device's
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` (101376 B on GB10/sm_121). The
// rejected `cudaFuncSetAttribute` return value was never read, and the latched
// `cudaErrorInvalidValue` later surfaced as a phantom kernel-launch error.)
static size_t gemm_dynamic_smem_bytes(int tm, int ks, bool af32) {
    const size_t tn = 64, nw = 8;
    return (2 * tn * ks + 2 * (size_t)tm * ks) * 2
           + (af32 ? (size_t)2 * tn * ks * 4 : 0)
           + nw * 256 * 4;
}

// The launchable prefill-GEMM set, listed **once**. The fatbin lookup below, the
// production pre-warm entry (#223) and the #218 test seam all expand this list,
// so an instantiation that is compiled cannot be silently missed by a caller's
// own copy — the class of drift the #145/#218 records are about.
#define MINFER_GEMM_OPTIN_SET(X)                                               \
    X(64, 32, false) X(64, 32, true)                                           \
    X(64, 64, false) X(64, 64, true)                                           \
    X(128, 32, false) X(128, 32, true)                                         \
    X(128, 64, false) X(128, 64, true)                                         \
    X(256, 32, false) X(256, 32, true)                                         \
    X(256, 64, false) X(256, 64, true)

// The compiled `gemm_f16_nt_kernel_t<tm, ks, af32>` instantiation, or null for
// a combination that is not in the fatbin.
#define GEMM_FN_FOR(TM, KS, AF)                                                \
    if (tm == (TM) && ks == (KS) && af32 == (AF))                              \
        return reinterpret_cast<const void*>(&gemm_f16_nt_kernel_t<TM, KS, AF>);
static const void* gemm_f16_fn_for(int tm, int ks, bool af32) {
    MINFER_GEMM_OPTIN_SET(GEMM_FN_FOR)
    return nullptr;
}
#undef GEMM_FN_FOR

// #218: how many times a prefill-GEMM dynamic-smem opt-in was attempted while
// the launch stream was inside a capture window. The design invariant is that
// this **never** happens: the attribute is decided on the first (uncaptured)
// launch and cached per instantiation, so a capture-window launch re-reads the
// cached answer. `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window`
// asserts the counter stays zero; `MINFER_TEST_CAPTURE_WARMUP=1` (the
// documented test seam) drives capture onto the first run and makes it fire.
static int g_gemm_smem_in_capture = 0;
extern "C" int gemm_smem_optin_in_capture_count() { return g_gemm_smem_in_capture; }

// The **production** opt-in for one prefill-GEMM instantiation, decided through
// the shared #147 helper. The launcher runs per layer per prefill, so an
// admitted or refused answer is recorded per instantiation and never re-asked
// on the hot path. Returns true when the >48 KiB dynamic smem is admitted and
// the launch may proceed.
//
// The template parameters are **the instantiation's own** (`TM`, `KS`, `AF32`),
// not a deduced function-pointer type: every `gemm_f16_nt_kernel_t` shares one
// signature, so a `template <typename K>` cache here would be a *per-signature*
// cache and the first admitted instantiation would answer for all the others
// (the #218 coverage gate found exactly that: `<64,64,true>` at 73728 B cached
// "admitted" for `<128,64,false>` at 57344 B, which was never set). `static
// state` must stay inside a per-`(TM,KS,AF32)` instantiation.
//
// #223 restored the **eager pre-warm**: `CudaState::try_new` drives this same
// production function once per process, for every launchable `(tm, ks, af32)`,
// from the earliest point in the process (context creation) where no stream can
// hold a capture window yet — so "the attribute is set outside any window" holds
// by construction, not by inference. This function is still the lazy per-launch
// opt-in too (defence in depth): it caches the admitted/refused answer per
// instantiation, so a launch whose pre-warm was skipped by the control env
// (`MINFER_NO_GEMM_PREWARM=1`) or by a regression still decides on its first
// uncaptured launch. The two remaining emergent mechanisms — capture opens in
// `cudaStreamCaptureModeThreadLocal` and only from the **third** run of a
// `(uid, range)` key (both recorded next to the capture trigger in
// `graph/cuda_backend.rs`) — are now defence in depth behind the pre-warm, not
// the primary guarantee. `stream` is used only to observe (and count) the
// never-case; the pre-warm passes 0 because no window can exist yet.
template <int TM, int KS, bool AF32>
static bool gemm_smem_optin(const char* site, const char* kernel_name, size_t bytes,
                            cudaStream_t stream) {
    static int state = 0;  // 0 = undecided, 1 = admitted, -1 = refused
    const bool injected = minfer_test_call_fails(site);
    // The 48 KiB default cap admits it — unless the test knob asks this site to
    // fail, because then the failure path is what is under test.
    if (bytes <= 48 * 1024 && !injected) return true;
    if (state != 0 && !injected) return state == 1;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (stream != 0 && cudaStreamIsCapturing(stream, &cs) == cudaSuccess
        && cs != cudaStreamCaptureStatusNone) {
        // The #218 gate asserts this stays zero: the attribute is set before the
        // window opens, never inside it.
        g_gemm_smem_in_capture++;
    } else {
        cudaGetLastError();  // a failed query must not latch
    }
    const bool ok = minfer_smem_optin(
        site, kernel_name,
        reinterpret_cast<const void*>(gemm_f16_nt_kernel_t<TM, KS, AF32>), (int)bytes);
    if (!injected) state = ok ? 1 : -1;  // a test answer is never cached
    return ok;
}

// Introspection for the Rust gates. `gemm_smem_need` is the single-source
// formula; `opted_in` is the device's own answer read back through
// `cudaFuncGetAttributes`; `limit` is the queried device opt-in limit.
extern "C" int gemm_prefill_smem_limit() { return minfer_optin_limit(); }
extern "C" size_t gemm_smem_need(int tm, int ks, int af32) {
    return gemm_dynamic_smem_bytes(tm, ks, af32 != 0);
}
extern "C" int gemm_smem_opted_in(int tm, int ks, int af32) {
    const void* fn = gemm_f16_fn_for(tm, ks, af32 != 0);
    if (fn == nullptr) return 0;
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, fn) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return a.maxDynamicSharedSizeBytes
                   >= (int)gemm_dynamic_smem_bytes(tm, ks, af32 != 0)
               ? 1
               : 0;
}

// Drive the production per-instantiation opt-in for one compiled combination.
// Returns 1 when admitted, 0 when refused (named by `minfer_smem_optin`), -1
// when the combination is not in the fatbin. Both the pre-warm (#223) and the
// test seam below go through here, so they read the same dispatch and the same
// cache.
static int gemm_smem_optin_decision(int tm, int ks, bool a, size_t bytes,
                                    const char* site, cudaStream_t stream) {
#define GEMM_OPTIN_CASE(TM, KS, AF)                                            \
    if (tm == (TM) && ks == (KS) && a == (AF))                                 \
        return gemm_smem_optin<TM, KS, AF>(                                    \
                   site, "gemm_f16_nt_kernel_t<" #TM "," #KS "," #AF ">",      \
                   bytes, stream)                                              \
                   ? 1                                                         \
                   : 0;
    MINFER_GEMM_OPTIN_SET(GEMM_OPTIN_CASE)
#undef GEMM_OPTIN_CASE
    return -1;
}

// #223: the **production** eager pre-warm entry, called once per process from
// `CudaState::try_new` for every combination the Rust side enumerates. It drives
// `gemm_smem_optin`, so the attribute is set through the one production path and
// the per-instantiation cache the launcher later reads is already populated.
//
// Outcomes:  1 = the attribute is in force (set now, or already cached);
//            0 = refused — `minfer_smem_optin` already named the instantiation,
//                the requested bytes, the call, the device limit and
//                `cudaGetErrorName`, and cleared the latch;
//           -1 = the combination is not in the fatbin (the caller skips it);
//           -2 = **skipped deliberately**: the request exceeds the device's own
//                `cudaDevAttrMaxSharedMemoryPerBlockOptin`, so the attribute was
//                never called — the call could only return
//                `cudaErrorInvalidValue` and the instantiation cannot launch on
//                this device at all (`minfer_smem_optin` printed the reason).
// Not gated on `stream`: there is none at context creation, and the whole point
// of the placement is that no capture window can exist.
extern "C" int gemm_prefill_smem_prewarm_one(int tm, int ks, int af32) {
    const bool a = af32 != 0;
    const size_t bytes = gemm_dynamic_smem_bytes(tm, ks, a);
    const int decision = gemm_smem_optin_decision(tm, ks, a, bytes, "prewarm:gemm_f16", 0);
    if (decision != 0) return decision;  // 1 admitted, -1 not compiled
    // Refused. Re-derive *why* so the caller can report a deliberate skip
    // separately from a real attribute failure; the report itself (with the
    // formula, the limit and the error name) came from `minfer_smem_optin`.
    const int limit = minfer_optin_limit();
    return (limit > 0 && bytes > (size_t)limit) ? -2 : 0;
}

// #218 test seam: drive the **production** `gemm_smem_optin` (template, cache
// and all) for one compiled instantiation, so the #145-derived coverage gate
// reads the decision production makes rather than a mirror of it. Returns 1
// when admitted, 0 when refused/skipped, -1 when the combination is not in the
// fatbin. Test-only: production reaches the same function through
// `launch_gemm_f16`'s `GEMM_ONE` and through `gemm_prefill_smem_prewarm_one`.
extern "C" int gemm_prefill_smem_optin_one_for_test(int tm, int ks, int af32, void* stream) {
    const bool a = af32 != 0;
    const size_t bytes = gemm_dynamic_smem_bytes(tm, ks, a);
    return gemm_smem_optin_decision(tm, ks, a, bytes, "test:gemm_smem_optin",
                                    reinterpret_cast<cudaStream_t>(stream));
}

// Test injection (issue #145): latch a REAL `cudaErrorInvalidValue` by asking
// for more dynamic shared memory than the device admits — the exact pre-fix
// init request — and deliberately NOT clear it, so the Rust gate can prove
// `CudaState::sync` reports a latched error as latched (and clears it) instead
// of blaming the kernel that just ran. Never called by production code.
extern "C" int cuda_test_latch_oversized_smem() {
    int dev = 0, limit = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaError_t e = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(&gemm_f16_nt_kernel_t<128, 32, false>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, limit + 4096);
    return (int)e;  // left latched on purpose
}

extern "C" {

void launch_dequant_f16(
    int type_id, const uint8_t* w, __half* out,
    int od, int id, int block_stride, cudaStream_t stream
) {
    int block = 256;
    long long total;
    switch (type_id) {
        case 7: total = (long long)od * (id / 16); break;            // q6_K
        default: total = (long long)od * (id / 32); break;           // all others
    }
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    switch (type_id) {
        case 0:
            minfer_launch_prelude("launch:dequant_f16__q8_0", "dequant_q8_0_f16");
            dequant_q8_0_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q8_0", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q8_0", "dequant_q8_0_f16");
            break;
        case 1:
            minfer_launch_prelude("launch:dequant_f16__q4_0", "dequant_q4_0_f16");
            dequant_q4_0_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q4_0", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q4_0", "dequant_q4_0_f16");
            break;
        case 2:
            minfer_launch_prelude("launch:dequant_f16__q4_1", "dequant_q4_1_f16");
            dequant_q4_1_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q4_1", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q4_1", "dequant_q4_1_f16");
            break;
        case 3:
            minfer_launch_prelude("launch:dequant_f16__q5_0", "dequant_q5_0_f16");
            dequant_q5_0_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q5_0", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q5_0", "dequant_q5_0_f16");
            break;
        case 4:
            minfer_launch_prelude("launch:dequant_f16__q5_1", "dequant_q5_1_f16");
            dequant_q5_1_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q5_1", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q5_1", "dequant_q5_1_f16");
            break;
        case 5:
            minfer_launch_prelude("launch:dequant_f16__q4_k", "dequant_q4_k_f16");
            dequant_q4_k_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q4_k", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q4_k", "dequant_q4_k_f16");
            break;
        case 6:
            minfer_launch_prelude("launch:dequant_f16__q5_k", "dequant_q5_k_f16");
            dequant_q5_k_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q5_k", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q5_k", "dequant_q5_k_f16");
            break;
        default:
            minfer_launch_prelude("launch:dequant_f16__q6_k", "dequant_q6_k_f16");
            dequant_q6_k_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q6_k", block), 0, stream>>>(w, out, od, id, block_stride);
            minfer_launch_ok("launch:dequant_f16__q6_k", "dequant_q6_k_f16");
            break;
    }
}

void launch_convert_f16(
    const float* x, __half* out, long long n, cudaStream_t stream
) {
    long long grid = (n / 8 + 255) / 256;
    if (grid > 2147483647LL) grid = 2147483647LL;
    minfer_launch_prelude("launch:convert_f16", "convert_f32_f16_kernel");
    convert_f32_f16_kernel<<<(int)grid, minfer_launch_block("launch:convert_f16", 256), 0, stream>>>(x, out, n);
    minfer_launch_ok("launch:convert_f16", "convert_f32_f16_kernel");
}

// #147: returns 1 when the launch was issued and accepted, 0 when the
// dynamic-smem opt-in failed or the launch itself returned an error — either
// way the call is named here and the caller (`prefill_gemm_f16_inner`) turns
// the 0 into an `Err`, never a silent launch into a checked error.
int launch_gemm_f16(
    const __half* a, const __half* b, float* c,
    int nt, int od, int id, cudaStream_t stream, bool af32
) {
    // P2: od-tile width (128 default = halved B re-reads; MINFER_GEMM_TM=64
    // reverts to the 8m② baseline for A/B).
    static int tm = -1;
    if (tm < 0) {
        const char* e = getenv("MINFER_GEMM_TM");
        int v = e ? atoi(e) : 128;
        tm = (v <= 64) ? 64 : (v >= 256) ? 256 : 128;
    }
    // KS = staged k-width per tile. KS=64 halves the barriers per FLOP but
    // measured -38% (56KB dynamic smem halves resident blocks on GB10);
    // KS=32 (8m2 baseline) stays the default. MINFER_GEMM_K64=1 re-tries 64.
    static int ks = -1;
    if (ks < 0) {
        const char* e = getenv("MINFER_GEMM_K64");
        ks = (e && atoi(e)) ? 64 : 32;
    }
    // #145: the single-source formula (AF32-aware). The old inline copy dropped
    // the AF32 f32 mirror, so `launch_gemm_f32a` declared 16384 B less than the
    // kernel's own `Bs`/`Cs` offsets need at KS=32 — an out-of-declaration
    // shared-memory access that "worked" only because the block's smem happened
    // to be carved where nothing else wrote.
    const size_t dyn_smem = gemm_dynamic_smem_bytes(tm, ks, af32);
    // #147: one site token per (af32) variant: the opt-in and the launch are
    // separate checks with separate injections.
    const char* const attr_site = af32 ? "attr:gemm_f16_a32" : "attr:gemm_f16_f16";
    const char* const launch_site = af32 ? "launch:gemm_f16_a32" : "launch:gemm_f16_f16";
    int ok = 0;
#define GEMM_ONE(TM_, KS_, AF_)                                                        \
    do {                                                                               \
        const char* const nm = "gemm_f16_nt_kernel_t<" #TM_ "," #KS_ "," #AF_ ">";     \
        if (gemm_smem_optin<TM_, KS_, AF_>(attr_site, nm, dyn_smem, stream)) {        \
            minfer_launch_prelude(launch_site, nm);                                    \
            gemm_f16_nt_kernel_t<TM_, KS_, AF_>                                        \
                <<<grid, 256, minfer_launch_smem(launch_site, dyn_smem), stream>>>(    \
                    a, b, c, nt, od, id);                                              \
            ok = minfer_launch_ok(launch_site, nm) ? 1 : 0;                            \
        }                                                                              \
    } while (0)
#define GEMM_LAUNCH(TM_, KS_)                                                          \
    do {                                                                               \
        dim3 grid((nt + 63) / 64, (od + TM_ - 1) / TM_);                               \
        if (af32) {                                                                    \
            GEMM_ONE(TM_, KS_, true);                                                  \
        } else {                                                                       \
            GEMM_ONE(TM_, KS_, false);                                                 \
        }                                                                              \
    } while (0)
    if (tm >= 128) {
        if (ks >= 64)
            GEMM_LAUNCH(128, 64);
        else
            GEMM_LAUNCH(128, 32);
    } else {
        if (ks >= 64)
            GEMM_LAUNCH(64, 64);
        else
            GEMM_LAUNCH(64, 32);
    }
#undef GEMM_LAUNCH
#undef GEMM_ONE
    return ok;
}

// P6: A arrives as f32 activations; converts inside the kernel on stage.
// #147: returns `launch_gemm_f16`'s own result (1 = launched and accepted) so
// the af32 path's failed launch is checked by its caller too.
int launch_gemm_f32a(
    const float* a, const __half* b, float* c,
    int nt, int od, int id, cudaStream_t stream
) {
    const int ok =
        launch_gemm_f16(reinterpret_cast<const __half*>(a), b, c, nt, od, id, stream, true);
    // The launch error (and the opt-in failure) is named inside
    // `launch_gemm_f16` now (#147); this stays for the opt-in synchronous
    // fault probe.
    if (getenv("MINFER_A32_SYNC") && ok) {
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "minfer/cuda: af32 gemm ASYNC FAULT nt=%d od=%d id=%d: %s\n",
                    nt, od, id, cudaGetErrorString(e));
            cudaGetLastError();  // named here, cleared here
        }
    }
    return ok;
}
}
