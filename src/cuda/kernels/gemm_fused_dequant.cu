// src/cuda/kernels/gemm_fused_dequant.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"

extern "C" {


// ─── 8p: fused dequant-in-GEMM ────────────────────────────────────────────
// 8m's two-pass path dequantized W to an f16 scratch (288 ms on 7B @2K)
// before every GEMM and streamed that f16 matrix back from DRAM. This
// kernel keeps the identical 64x64 wmma tile structure but dequantizes B
// tiles in-register from the RAW quantized bytes (a Q4_K 64-row k-tile
// reads ~1.5 KB of quantized data + headers instead of 4 KB of f16 — and
// the separate dequant pass disappears). A still comes from the f16
// activation scratch (launch_convert_f16 stays; a fused f32→f16 A load is
// future work). Each bqa_* mirrors its dequant_*_f16 kernel's element math
// and __float2half rounding EXACTLY, so fused and two-pass results are
// bit-identical (asserted in cuda_prefill_fused_b_bitparity).
//
// Requires id % 256 == 0 (host-side gate): the 8-element runs never
// straddle a 32-sub-block, and K-quant super-block boundaries stay aligned.

__device__ __forceinline__ void bqa_q8_0(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 34) + (long long)(e0 >> 5) * 34;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    int b = e0 & 31;
    #pragma unroll
    for (int l = 0; l < 8; l++)
        dst[l] = __float2half(d * float((int8_t)blk[2 + b + l]));
}

__device__ __forceinline__ void bqa_q4_0(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 18) + (long long)(e0 >> 5) * 18;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    int b = e0 & 31;
    // minfer Q4_0 stores round(v/d) + 8 (same -8 offset as the matmuls);
    // element e uses byte blk[2 + (e & 15)]: lo nibble for e < 16.
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[2 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        dst[l] = __float2half(d * (nib - 8.0f));
    }
}

__device__ __forceinline__ void bqa_q4_1(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 20) + (long long)(e0 >> 5) * 20;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    int b = e0 & 31;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[4 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        dst[l] = __float2half(d * nib + m);
    }
}

__device__ __forceinline__ void bqa_q5_0(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 22) + (long long)(e0 >> 5) * 22;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    // 22-byte blocks are only 2-byte aligned: assemble qh from two u16
    // loads (a plain u32 load at blk+2 misaligns for even block indices —
    // cudaErrorMisalignedAddress, caught by cuda_prefill_fused_b_bitparity).
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
    int b = e0 & 31;
    // element e: nibble qs[e & 15] (lo for e < 16), high bit qh >> e.
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[6 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        float v = nib + 16.0f * float((qh >> e) & 1) - 16.0f;
        dst[l] = __float2half(d * v);
    }
}

__device__ __forceinline__ void bqa_q5_1(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 24) + (long long)(e0 >> 5) * 24;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6) << 16);
    int b = e0 & 31;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[8 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        float v = nib + 16.0f * float((qh >> e) & 1);
        dst[l] = __float2half(d * v + m);
    }
}

__device__ __forceinline__ void bqa_q4_k(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 8) * 144) + (long long)(e0 >> 8) * 144;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    int eb = e0 & 255;
    int sub = eb >> 5, l0 = eb & 31;
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    float ds = d * float(scb), dmm = dmin * float(mb);
    int half = sub & 1;
    const uint8_t* q = blk + 16 + (sub >> 1) * 32;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int bi = l0 + l;
        float nib = half ? float(q[bi] >> 4) : float(q[bi] & 0x0F);
        dst[l] = __float2half(ds * nib - dmm);
    }
}

__device__ __forceinline__ void bqa_q5_k(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 8) * 176) + (long long)(e0 >> 8) * 176;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    int eb = e0 & 255;
    int sub = eb >> 5, l0 = eb & 31;
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    float ds = d * float(scb), dmm = dmin * float(mb);
    int ci = sub >> 1, hi = sub & 1;
    const uint8_t* q4 = blk + 48 + ci * 32;
    const uint8_t* qh = blk + 16;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int bi = l0 + l;
        float nib = hi ? float(q4[bi] >> 4) : float(q4[bi] & 0x0F);
        float wv = nib + 16.0f * float((qh[bi] >> sub) & 1);
        dst[l] = __float2half(ds * wv - dmm);
    }
}

__device__ __forceinline__ void bqa_q6_k(
    const uint8_t* w, int row, int id, int e0, int bstride, __half* dst
) {
    // bstride: 210 raw or 224 padded (7e② repack keeps intra-block layout).
    const uint8_t* blk = w + (long long)row * ((id >> 8) * bstride) + (long long)(e0 >> 8) * bstride;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    int eb = e0 & 255;
    int n = eb >> 7, rem = eb & 127, tt = rem >> 5, gq = (rem >> 4) & 1;
    int ql_off = n * 64 + (tt & 1) * 32 + gq * 16;
    int qh_off = n * 32 + gq * 16;
    float dsc = d * float(sc[n * 8 + tt * 2 + gq]);
    int r0 = eb & 15;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int r = r0 + l;
        int nib = (tt < 2) ? (ql[ql_off + r] & 0x0F) : (ql[ql_off + r] >> 4);
        int q2 = (qh[qh_off + r] >> (tt * 2)) & 3;
        dst[l] = __float2half(dsc * float((nib | (q2 << 4)) - 32));
    }
}

// One 64-row B tile k-slice: thread = (row, 8-element chunk); the B side
// dequantizes raw bytes in-register, the A side vector-loads f16.
__device__ __forceinline__ void gemm_qb_load_tile(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    __half* As, __half* Bs,
    int n0, int m0, int k0, int nt, int od, int id,
    int type_id, int q6_stride
) {
    int r = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    int n = n0 + r;
    if (n < nt) {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) =
            *reinterpret_cast<const uint4*>(A + (long long)n * id + k0 + c4);
    } else {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
    int m = m0 + r;
    if (m < od) {
        int e0 = k0 + c4;
        __half* dst = Bs + r * 32 + c4;
        switch (type_id) {
            case 0: bqa_q8_0(W, m, id, e0, dst); break;
            case 1: bqa_q4_0(W, m, id, e0, dst); break;
            case 2: bqa_q4_1(W, m, id, e0, dst); break;
            case 3: bqa_q5_0(W, m, id, e0, dst); break;
            case 4: bqa_q5_1(W, m, id, e0, dst); break;
            case 5: bqa_q4_k(W, m, id, e0, dst); break;
            case 6: bqa_q5_k(W, m, id, e0, dst); break;
            default: bqa_q6_k(W, m, id, e0, q6_stride, dst); break;
        }
    } else {
        *reinterpret_cast<uint4*>(Bs + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
}

// C[nt, od] = A[nt, id] · dequant(W[od, id])^T — same tile/warp structure,
// fragment layout, and store masking as gemm_f16_nt_kernel; only the B
// staging differs (raw bytes dequantized in-register). Synchronous
// double-buffered loads: cp.async cannot convert or dequantize.
__global__ void gemm_qb_nt_kernel(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    float* __restrict__ C, int nt, int od, int id,
    int type_id, int q6_stride
) {
    using namespace nvcuda;
    __shared__ __half As[2][64 * 32];
    __shared__ __half Bs[2][64 * 32];
    __shared__ float Cs[8][16 * 16];

    int warp = threadIdx.x >> 5;
    int wm = warp >> 1;
    int wn = warp & 1;
    int m0 = blockIdx.y * 64;
    int n0 = blockIdx.x * 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2];
    wmma::fill_fragment(fc[0], 0.0f);
    wmma::fill_fragment(fc[1], 0.0f);

    int buf = 0;
    gemm_qb_load_tile(A, W, As[0], Bs[0], n0, m0, 0, nt, od, id, type_id, q6_stride);
    __syncthreads();
    for (int k = 0; k < id; k += 32, buf ^= 1) {
        if (k + 32 < id)
            gemm_qb_load_tile(A, W, As[buf ^ 1], Bs[buf ^ 1], n0, m0, k + 32,
                              nt, od, id, type_id, q6_stride);
        // fa: [n-block 0/1] x [k-half 0/1]; fb: [k-half 0/1] (same as 8m).
        wmma::load_matrix_sync(fa[0], &As[buf][wn * 32 * 32], 32);
        wmma::load_matrix_sync(fa[1], &As[buf][(wn * 32 + 16) * 32], 32);
        wmma::load_matrix_sync(fa[2], &As[buf][wn * 32 * 32 + 16], 32);
        wmma::load_matrix_sync(fa[3], &As[buf][(wn * 32 + 16) * 32 + 16], 32);
        wmma::load_matrix_sync(fb[0], &Bs[buf][wm * 16 * 32], 32);
        wmma::load_matrix_sync(fb[1], &Bs[buf][wm * 16 * 32 + 16], 32);
        wmma::mma_sync(fc[0], fa[0], fb[0], fc[0]);
        wmma::mma_sync(fc[1], fa[1], fb[0], fc[1]);
        wmma::mma_sync(fc[0], fa[2], fb[1], fc[0]);
        wmma::mma_sync(fc[1], fa[3], fb[1], fc[1]);
        __syncthreads();
    }

    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        wmma::store_matrix_sync(Cs[warp], fc[j], 16, wmma::mem_row_major);
        int nb = n0 + wn * 32 + j * 16, mb = m0 + wm * 16;
        for (int e = lane; e < 256; e += 32) {
            int n = nb + (e >> 4), m = mb + (e & 15);
            if (n < nt && m < od)
                C[(long long)n * od + m] = Cs[warp][(e >> 4) * 16 + (e & 15)];
        }
    }
}

void launch_gemm_qb_nt(
    const __half* a, const uint8_t* w, float* c,
    int nt, int od, int id, int type_id, int q6_stride, cudaStream_t stream
) {
    dim3 grid((nt + 63) / 64, (od + 63) / 64);
    minfer_launch_prelude("launch:gemm_qb_nt", "gemm_qb_nt_kernel");
    gemm_qb_nt_kernel<<<grid, minfer_launch_block("launch:gemm_qb_nt", 256), 0, stream>>>(a, w, c, nt, od, id, type_id, q6_stride);
    minfer_launch_ok("launch:gemm_qb_nt", "gemm_qb_nt_kernel");
}

}
