// src/cuda/kernels/mmvq_q6k.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"

// ─── D3b-1b: software-pipelined q6_K MMVQ (tall rows, npair > 256) ─────────
// Same mapping and arithmetic as q6_k_q8_mmvq_v2; both units' loads issue
// before either accumulates so the second unit's weight latency leaves the
// critical path. Bitwise-identical (see the file comment).
struct Q6kUnitRegs {
    uint4 qla, qlb, qha, qhb;
    const uint32_t* xw;
    uint32_t shift;
    int g;
    float d, sc0, sc1, d8;
};

__device__ __forceinline__ void q6k_unit_load(
    int u, const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ x8row,
    int blk_stride, Q6kUnitRegs* r
) {
    const int kbx = u >> 3, pair = u & 7;
    const uint8_t* blk = wrow + (size_t)kbx * blk_stride;
    r->d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    r->sc0 = (float)(int8_t)blk[192 + 2 * pair];
    r->sc1 = (float)(int8_t)blk[192 + 2 * pair + 1];
    const int chunk = pair >> 2, g = pair & 3;
    r->shift = 2 * g;
    r->g = g;
    r->qla = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32);
    r->qlb = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32 + 16);
    r->qha = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32);
    r->qhb = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32 + 16);
    const uint8_t* x8 = x8row + (size_t)u * Q8PB;
    r->d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
    r->xw = reinterpret_cast<const uint32_t*>(x8 + 4);
}

__device__ __forceinline__ void q6k_unit_acc(const Q6kUnitRegs* r, float& acc) {
    const uint32_t qls[8] = {r->qla.x, r->qla.y, r->qla.z, r->qla.w,
                             r->qlb.x, r->qlb.y, r->qlb.z, r->qlb.w};
    const uint32_t qhs[8] = {r->qha.x, r->qha.y, r->qha.z, r->qha.w,
                             r->qhb.x, r->qhb.y, r->qhb.z, r->qhb.w};
    const uint32_t shift = r->shift;
    int dot0 = 0, dot1 = 0;
    #pragma unroll
    for (int v = 0; v < 4; v++) {
        const uint32_t wl0 = qls[v], wl1 = qls[v + 4];
        const uint32_t wh0 = qhs[v], wh1 = qhs[v + 4];
        const uint32_t nib0 = (r->g < 2) ? (wl0 & 0x0F0F0F0F) : ((wl0 >> 4) & 0x0F0F0F0F);
        const uint32_t nib1 = (r->g < 2) ? (wl1 & 0x0F0F0F0F) : ((wl1 >> 4) & 0x0F0F0F0F);
        const uint32_t hi0 = ((wh0 >> shift) & 0x03030303) << 4;
        const uint32_t hi1 = ((wh1 >> shift) & 0x03030303) << 4;
        const int vi0 = __vsubss4((int)(nib0 | hi0), 0x20202020);
        const int vi1 = __vsubss4((int)(nib1 | hi1), 0x20202020);
        dot0 = __dp4a(vi0, (int)r->xw[v], dot0);
        dot1 = __dp4a(vi1, (int)r->xw[v + 4], dot1);
    }
    // textually identical to the v2 accumulation statement (same contraction)
    acc += r->d8 * r->sc0 * r->d * (float)dot0 + r->d8 * r->sc1 * r->d * (float)dot1;
}

__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2_pf(
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
    const uint8_t* wrow = weights + (size_t)row * row_stride;

    float acc = 0.0f;
    const int u0 = threadIdx.x;
    const int u1 = u0 + 256;
    if (u0 < npair) {
        Q6kUnitRegs r0, r1;
        q6k_unit_load(u0, wrow, x8row, blk_stride, &r0);
        const bool two = u1 < npair; // npair > blockDim here (dispatch-gated)
        if (two) q6k_unit_load(u1, wrow, x8row, blk_stride, &r1);
        q6k_unit_acc(&r0, acc);
        if (two) q6k_unit_acc(&r1, acc);
    }
    mmvq_block_reduce(acc, output, od, t);
}

// ─── D4-4 L1: dense split-plane (dpl) q6_K decode MMVQ ─────────────────────
// Sibling plane built at registration (q6k_dpl map, keyed by the padded
// weight's device pointer): per row [ql: nbe*128][qh: nbe*64][sc: nbe*16]
// [d: nbe*2] at a 16B-aligned row stride — 210B of content per 256-elem
// block, zero 224B pad sectors, every uint4 load 16B-aligned. Per-unit
// values are byte-identical to the padded layout and the unit/accumulation
// order is unchanged, so outputs are bitwise-identical (probe
// /tmp/d4/probe_l1_dpl.cu: memcmp-equal vs the padded kernels on the
// ffn_down + lm_head shapes; kernels −17%/−16.7% mean there).

__device__ __forceinline__ void q6k_unit_load_dpl(
    int u, const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ x8row,
    int nbe, Q6kUnitRegs* r
) {
    const int kbx = u >> 3, pair = u & 7;
    const uint8_t* ql_row = wrow;
    const uint8_t* qh_row = wrow + (size_t)nbe * 128;
    const uint8_t* sc_row = wrow + (size_t)nbe * 192;
    const uint8_t* d_row  = wrow + (size_t)nbe * 208;
    r->d = h2f(*reinterpret_cast<const uint16_t*>(d_row + (size_t)kbx * 2));
    r->sc0 = (float)(int8_t)sc_row[(size_t)kbx * 16 + 2 * pair];
    r->sc1 = (float)(int8_t)sc_row[(size_t)kbx * 16 + 2 * pair + 1];
    const int chunk = pair >> 2, g = pair & 3;
    r->shift = 2 * g;
    r->g = g;
    const uint8_t* qlp = ql_row + (size_t)kbx * 128 + chunk * 64 + (g & 1) * 32;
    r->qla = *reinterpret_cast<const uint4*>(qlp);
    r->qlb = *reinterpret_cast<const uint4*>(qlp + 16);
    const uint8_t* qhp = qh_row + (size_t)kbx * 64 + chunk * 32;
    r->qha = *reinterpret_cast<const uint4*>(qhp);
    r->qhb = *reinterpret_cast<const uint4*>(qhp + 16);
    const uint8_t* x8 = x8row + (size_t)u * Q8PB;
    r->d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
    r->xw = reinterpret_cast<const uint32_t*>(x8 + 4);
}

__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2_pf_dpl(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int nbe
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int row_stride = (nbe * 210 + 15) & ~15;
    const int npair = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;
    const uint8_t* wrow = weights + (size_t)row * row_stride;

    float acc = 0.0f;
    const int u0 = threadIdx.x;
    const int u1 = u0 + 256;
    if (u0 < npair) {
        Q6kUnitRegs r0, r1;
        q6k_unit_load_dpl(u0, wrow, x8row, nbe, &r0);
        const bool two = u1 < npair;
        if (two) q6k_unit_load_dpl(u1, wrow, x8row, nbe, &r1);
        q6k_unit_acc(&r0, acc);
        if (two) q6k_unit_acc(&r1, acc);
    }
    mmvq_block_reduce(acc, output, od, t);
}

__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2_dpl(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int nbe
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int row_stride = (nbe * 210 + 15) & ~15;
    const int npair = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < npair; u += 256) {
        Q6kUnitRegs r;
        q6k_unit_load_dpl(u, weights + (size_t)row * row_stride, x8row, nbe, &r);
        q6k_unit_acc(&r, acc);
    }
    mmvq_block_reduce(acc, output, od, t);
}

__global__ void q6_k_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int QKK = 256;
    const int NR0 = 2;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nbe = (id + QKK - 1) / QKK;
    int row_stride = nbe * Q6KB;

    const uint8_t* w0 = weights + (r0 + 0) * row_stride;
    // NR0 = 2 with odd od: the last group's row 1 does not exist — alias it
    // to row 0 (in-bounds); the write guard discards the sum (7e review)
    const bool row1_ok = (r0 + 1) < od;
    const uint8_t* w1 = row1_ok ? weights + (r0 + 1) * row_stride : w0;
    const float* y = acts + t * id;

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (int ib = lane_id; ib < nbe; ib += WARP) {
        const uint8_t* blk0 = w0 + ib * Q6KB;
        const uint8_t* blk1 = w1 + ib * Q6KB;

        float bd0 = h2f(*reinterpret_cast<const uint16_t*>(blk0 + 208));
        float bd1 = h2f(*reinterpret_cast<const uint16_t*>(blk1 + 208));

        const uint8_t* ql0 = blk0;
        const uint8_t* ql1 = blk1;
        const uint8_t* qh0 = blk0 + 128;
        const uint8_t* qh1 = blk1 + 128;
        const int8_t* sc0 = (const int8_t*)(blk0 + 192);
        const int8_t* sc1 = (const int8_t*)(blk1 + 192);
        const float* yb = y + ib * QKK;

        // 7e②: the y side (the hot cache-resident stream) loads float4
        // groups instead of 32-float-strided scalars; ql/qh stay per-byte —
        // Q6KB = 210 is not 16-byte aligned, so vector weight loads would
        // need a repacked layout (possible follow-up).
        for (int n = 0; n < 2; n++) {
            #pragma unroll
            for (int g = 0; g < 2; g++) {
                // l stays in [0,32): two 16-element groups per half; the four
                // terms ys[0/32/64/96] use scales sc[n*8 + g + {0, 2, 4, 6}]
                // (si = l/16 + n*8 in the scalar formulation, is = g).
                const float* yb_n = yb + n * 128 + g * 16;

                float p00 = 0.0f, p01 = 0.0f, p02 = 0.0f, p03 = 0.0f;
                float p10 = 0.0f, p11 = 0.0f, p12 = 0.0f, p13 = 0.0f;

                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    const int l = g * 16 + v * 4;
                    float4 ys0 = *reinterpret_cast<const float4*>(yb_n + 0);
                    float4 ys1 = *reinterpret_cast<const float4*>(yb_n + 32);
                    float4 ys2 = *reinterpret_cast<const float4*>(yb_n + 64);
                    float4 ys3 = *reinterpret_cast<const float4*>(yb_n + 96);

                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const int b = l + r;
                        int qh0_b = qh0[b];
                        int qh1_b = qh1[b];
                        int q0_0 = ((int)(ql0[b] & 0xF) | ((qh0_b & 3) << 4)) - 32;
                        int q1_0 = ((int)(ql1[b] & 0xF) | ((qh1_b & 3) << 4)) - 32;
                        int q0_1 = ((int)(ql0[b + 32] & 0xF) | (((qh0_b >> 2) & 3) << 4)) - 32;
                        int q1_1 = ((int)(ql1[b + 32] & 0xF) | (((qh1_b >> 2) & 3) << 4)) - 32;
                        int q0_2 = ((int)(ql0[b] >> 4) | (((qh0_b >> 4) & 3) << 4)) - 32;
                        int q1_2 = ((int)(ql1[b] >> 4) | (((qh1_b >> 4) & 3) << 4)) - 32;
                        int q0_3 = ((int)(ql0[b + 32] >> 4) | (((qh0_b >> 6) & 3) << 4)) - 32;
                        int q1_3 = ((int)(ql1[b + 32] >> 4) | (((qh1_b >> 6) & 3) << 4)) - 32;

                        // component index is r (the element within the
                        // float4), not v — the v selector was the 7e② bug
                        // (3/4 of the y values were never read).
                        const float* c0 = reinterpret_cast<const float*>(&ys0);
                        const float* c1 = reinterpret_cast<const float*>(&ys1);
                        const float* c2 = reinterpret_cast<const float*>(&ys2);
                        const float* c3 = reinterpret_cast<const float*>(&ys3);
                        const float y0 = c0[r];
                        const float y1 = c1[r];
                        const float y2 = c2[r];
                        const float y3 = c3[r];
                        p00 += float(q0_0) * y0;
                        p01 += float(q0_1) * y1;
                        p02 += float(q0_2) * y2;
                        p03 += float(q0_3) * y3;
                        p10 += float(q1_0) * y0;
                        p11 += float(q1_1) * y1;
                        p12 += float(q1_2) * y2;
                        p13 += float(q1_3) * y3;
                    }
                    yb_n += 4;
                }
                int si = n * 8 + g;
                sumf0 += bd0 * (float(sc0[si + 0]) * p00 + float(sc0[si + 2]) * p01
                              + float(sc0[si + 4]) * p02 + float(sc0[si + 6]) * p03);
                sumf1 += bd1 * (float(sc1[si + 0]) * p10 + float(sc1[si + 2]) * p11
                              + float(sc1[si + 4]) * p12 + float(sc1[si + 6]) * p13);
            }
            ql0 += 64; ql1 += 64;
            qh0 += 32; qh1 += 32;
        }
    }

    sumf0 = warp_reduce_sum(sumf0);
    sumf1 = warp_reduce_sum(sumf1);
    if (lane_id == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
    }
}
extern "C" {

void launch_q6_k_q8_mmvq_v2_pf(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int blk_stride, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q6_k_q8_mmvq_v2_pf", "q6_k_q8_mmvq_v2_pf");
    q6_k_q8_mmvq_v2_pf<<<grid, minfer_launch_block("launch:q6_k_q8_mmvq_v2_pf", 256), 0, stream>>>(weights, acts8, output, od, id, nt, blk_stride);
    minfer_launch_ok("launch:q6_k_q8_mmvq_v2_pf", "q6_k_q8_mmvq_v2_pf");
}

// D4-4 L1: dense split-plane (dpl) decode launchers — see the kernel comments.
void launch_q6_k_q8_mmvq_v2_pf_dpl(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int nbe, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q6_k_q8_mmvq_v2_pf_dpl", "q6_k_q8_mmvq_v2_pf_dpl");
    q6_k_q8_mmvq_v2_pf_dpl<<<grid, minfer_launch_block("launch:q6_k_q8_mmvq_v2_pf_dpl", 256), 0, stream>>>(weights, acts8, output, od, id, nt, nbe);
    minfer_launch_ok("launch:q6_k_q8_mmvq_v2_pf_dpl", "q6_k_q8_mmvq_v2_pf_dpl");
}

void launch_q6_k_q8_mmvq_v2_dpl(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int nbe, cudaStream_t stream
) {
    dim3 grid(od, nt);
    minfer_launch_prelude("launch:q6_k_q8_mmvq_v2_dpl", "q6_k_q8_mmvq_v2_dpl");
    q6_k_q8_mmvq_v2_dpl<<<grid, minfer_launch_block("launch:q6_k_q8_mmvq_v2_dpl", 256), 0, stream>>>(weights, acts8, output, od, id, nt, nbe);
    minfer_launch_ok("launch:q6_k_q8_mmvq_v2_dpl", "q6_k_q8_mmvq_v2_dpl");
}

void launch_q6_k_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 2, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    minfer_launch_prelude("launch:q6_k_f32_matmul", "q6_k_f32_matmul");
    q6_k_f32_matmul<<<grid, minfer_launch_block("launch:q6_k_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q6_k_f32_matmul", "q6_k_f32_matmul");
}
}
