// ─── Flash-style windowed attention: the fast explicit-span prefill (#359) ───
//
// The E1 `attn_span` read path had one kernel family before this file: the
// correctness reference `kernel_gqa_attn_window_f32/_f16` in attn_window.metal
// (#44), which runs **one threadgroup per (query, KV head)** and resolves the
// window `[lo, hi)` per query. That is the honest reference but it is 2.7x-10.4x
// slower than the causal prefill `attn_flash_prefill` (#315), because it stages
// K/V with scalar loads and computes one dot per lane instead of tiling with
// simdgroup_matrix.
//
// This family is the fast sibling. It is a **copy** of the causal
// `kernel_flash_attn_blk_*` family (fa_prefill.metal) — same Q=8 x C=64
// simdgroup tiling, same online softmax, same f16 shared-memory staging — whose
// code is deliberately left byte-untouched (its measured numbers are a
// contract). The only change is the mask: instead of the causal
// `[0, positions[t] + 1)` it reads each query's explicit
// `[window[t], window[nt + t])` range.
//
// The threadgroup processes the **global** range `[lo_min, hi_max)` of its
// queries. The host advances K/V by `lo_min` cells for the tail pad and passes
// `lo_min` so the kernel masks each query to its own window; the mask is the
// only correctness guard, exactly as the causal kernel's inline mask is. A
// windowed prefill therefore does the same tile work the causal prefill does,
// with any extra blocks masked out. The window mode's own dispatch
// (`src/graph/metal_backend.rs`, the `explicit_span` arm) selects this family
// only for `nt > 1` and `hd` in {64, 128}; every other shape keeps the
// correctness kernels.
//
// `window` carries the `attn_span` I32 input stored as `f32::from_bits` (compute
// graph rule 4), bound as `constant int *` exactly like the causal kernels'
// `positions`. A `kv_map`-sized window is a different layout and stays on
// `kernel_gqa_attn_map_f32/_f16` (#362).

// ─── hd = 64 f32 ────────────────────────────────────────────────────────────
// shmem (7168 B): sq[512 half] | so[512 f32] | ss[1024 f32]
kernel void kernel_flash_attn_window_blk_f32(
    device const float * q         [[buffer(0)]],
    device const float * k         [[buffer(1)]],
    device const float * v         [[buffer(2)]],
    device const float * pad       [[buffer(3)]],   // [2][64][nkt] K-tail then V-tail
    device       float * out       [[buffer(4)]],
    constant    int    * window    [[buffer(5)]],   // [lo_0..lo_{nt-1}, hi_0..hi_{nt-1}]
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],  // hi_max - lo_min
    constant    int    & lo_min    [[buffer(12)]],  // absolute first cell
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int Q   = 8;
    constexpr int C   = 64;
    constexpr int NSG = 4;
    constexpr int DK  = 64;
    constexpr int NW  = 32;
    constexpr int NQ  = Q / NSG;      // 2
    constexpr int SH  = 2 * C;        // 128
    constexpr int DK4 = DK / 4;       // 16
    constexpr int DK8 = DK / 8;       // 8
    constexpr int PV  = 64;           // PAD2(DV, 64)
    constexpr int PV4 = PV / 4;       // 16
    constexpr int PV8 = PV / 8;       // 8
    constexpr int NC  = (C / 8) / NSG; // 2
    constexpr int NO  = PV8 / NSG;    // 2
    constexpr float MINF = 65504.0f;

    const int iq1 = (int)tgpig.x * Q;
    const int iq2 = (int)tgpig.y;
    const int nblk = (nkv + C - 1) / C;

    const int nqt  = nh * hd;
    const int nkt  = nk * hd;
    const int hk   = iq2 / (nh / nk);
    const int hoff = hk * hd;

    const int tx = (int)tiisg;

    // shmem layout (bytes): sq[0..1024) | so[1024..3072) | ss[3072..7168)
    threadgroup half  * sq = (threadgroup half  *)shmem;
    threadgroup float * so = (threadgroup float *)(shmem + 256);
    threadgroup float * ss = (threadgroup float *)(shmem + 768);

    threadgroup half4  * sq4 = (threadgroup half4  *)sq;
    threadgroup float4 * so4 = (threadgroup float4 *)so;
    threadgroup float2 * ss2 = (threadgroup float2 *)ss;

    // load Q heads into shared memory (each simdgroup loads NQ queries)
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *)(q + (iq1 + j) * nqt + iq2 * hd);
        for (int i = tx; i < DK4; i += NW) {
            sq4[j * DK4 + i] = (iq1 + j < nt) ? half4(q4[i]) : (half4)0.0f;
        }
    }

    // zero so + ss
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] = (float4)0.0f;
        for (int i = tx; i < SH; i += NW) ss[j * SH + i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (short jj = 0; jj < NQ; ++jj) { S[jj] = 0.0f; M[jj] = -FLT_MAX / 2; }

    for (int ic0 = 0; ic0 < nblk; ++ic0) {
        const int ic = ic0 * C;
        const bool partial = (ic + C > nkv);
        const int pos0 = ic;
        device const float * ksrc =
            partial ? (pad + hoff) : (k + lo_min * nkt + ic * nkt + hoff);
        device const float * vsrc =
            partial ? (pad + C * nkt + hoff) : (v + lo_min * nkt + ic * nkt + hoff);

        // ── Q*K^T ──
        {
            threadgroup const half  * pq = sq;
            threadgroup       float * ps = ss + sgitg * 8;
            device     const float * pk = ksrc + sgitg * (8 * nkt);
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8 mq[2];
                simdgroup_float8x8 mk[2];
                #pragma unroll
                for (short i = 0; i < DK8 / 2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
                    simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);
                    simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }
                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8 * (NSG * nkt);
                ps += 8 * NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── online softmax (explicit window mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const bool qvalid = (iq1 + j < nt);
            // An invalid query (the Q-tile's tail padding) masks nothing: give it
            // the full range so no lane produces inf and the softmax stays finite.
            const int lo = qvalid ? window[iq1 + j] : lo_min;
            const int hi = qvalid ? window[nt + iq1 + j] : (lo_min + (int)nkv);
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = lo_min + pos0 + 2 * tx;
            s2[0] += (kpos0 >= lo && kpos0 < hi) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= lo && kpos0 + 1 < hi) ? 0.0f : -MINF;
            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
            const float ms = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[j * (SH / 2) + tx] = vs2;
            for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── O += P * V ──
        {
            simdgroup_float8x8 lo[NO];
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            {
                device const float * pv = vsrc + 8 * sgitg;   // dim offset 8*sgitg
                for (short cc = 0; cc < C / 8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                    simdgroup_float8x8 mv[2];
                    simdgroup_load(mv[0], pv + 0 * NSG, nkt, 0, false);
                    simdgroup_load(mv[1], pv + 8 * NSG, nkt, 0, false);
                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
                    pv += 8 * nkt;
                }
            }
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── store to global ──
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// ─── hd = 64 f16-K/V ────────────────────────────────────────────────────────
kernel void kernel_flash_attn_window_blk_f16(
    device const float * q         [[buffer(0)]],
    device const half *  k         [[buffer(1)]],
    device const half *  v         [[buffer(2)]],
    device const half *  pad       [[buffer(3)]],
    device       float * out       [[buffer(4)]],
    constant    int    * window    [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
    constant    int    & lo_min    [[buffer(12)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int Q   = 8;
    constexpr int C   = 64;
    constexpr int NSG = 4;
    constexpr int DK  = 64;
    constexpr int NW  = 32;
    constexpr int NQ  = Q / NSG;
    constexpr int SH  = 2 * C;
    constexpr int DK4 = DK / 4;
    constexpr int DK8 = DK / 8;
    constexpr int PV  = 64;
    constexpr int PV4 = PV / 4;
    constexpr int PV8 = PV / 8;
    constexpr int NC  = (C / 8) / NSG;
    constexpr int NO  = PV8 / NSG;
    constexpr float MINF = 65504.0f;

    const int iq1 = (int)tgpig.x * Q;
    const int iq2 = (int)tgpig.y;
    const int nblk = (nkv + C - 1) / C;

    const int nqt  = nh * hd;
    const int nkt  = nk * hd;
    const int hk   = iq2 / (nh / nk);
    const int hoff = hk * hd;

    const int tx = (int)tiisg;

    threadgroup half  * sq = (threadgroup half  *)shmem;
    threadgroup float * so = (threadgroup float *)(shmem + 256);
    threadgroup float * ss = (threadgroup float *)(shmem + 768);

    threadgroup half4  * sq4 = (threadgroup half4  *)sq;
    threadgroup float4 * so4 = (threadgroup float4 *)so;
    threadgroup float2 * ss2 = (threadgroup float2 *)ss;

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *)(q + (iq1 + j) * nqt + iq2 * hd);
        for (int i = tx; i < DK4; i += NW) {
            sq4[j * DK4 + i] = (iq1 + j < nt) ? half4(q4[i]) : (half4)0.0f;
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] = (float4)0.0f;
        for (int i = tx; i < SH; i += NW) ss[j * SH + i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (short jj = 0; jj < NQ; ++jj) { S[jj] = 0.0f; M[jj] = -FLT_MAX / 2; }

    for (int ic0 = 0; ic0 < nblk; ++ic0) {
        const int ic = ic0 * C;
        const bool partial = (ic + C > nkv);
        const int pos0 = ic;
        device const half * ksrc =
            partial ? (pad + hoff) : (k + lo_min * nkt + ic * nkt + hoff);
        device const half * vsrc =
            partial ? (pad + C * nkt + hoff) : (v + lo_min * nkt + ic * nkt + hoff);

        // ── Q*K^T ──
        {
            threadgroup const half  * pq = sq;
            threadgroup       float * ps = ss + sgitg * 8;
            device     const half  * pk = ksrc + sgitg * (8 * nkt);
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8 mq[2];
                simdgroup_half8x8 mk[2];
                #pragma unroll
                for (short i = 0; i < DK8 / 2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
                    simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);
                    simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }
                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8 * (NSG * nkt);
                ps += 8 * NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── online softmax (explicit window mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const bool qvalid = (iq1 + j < nt);
            const int lo = qvalid ? window[iq1 + j] : lo_min;
            const int hi = qvalid ? window[nt + iq1 + j] : (lo_min + (int)nkv);
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = lo_min + pos0 + 2 * tx;
            s2[0] += (kpos0 >= lo && kpos0 < hi) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= lo && kpos0 + 1 < hi) ? 0.0f : -MINF;
            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
            const float ms = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[j * (SH / 2) + tx] = vs2;
            for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── O += P * V ──
        {
            simdgroup_float8x8 lo[NO];
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            {
                device const half * pv = vsrc + 8 * sgitg;
                for (short cc = 0; cc < C / 8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                    simdgroup_half8x8 mv[2];
                    simdgroup_load(mv[0], pv + 0 * NSG, nkt, 0, false);
                    simdgroup_load(mv[1], pv + 8 * NSG, nkt, 0, false);
                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
                    pv += 8 * nkt;
                }
            }
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// ─── hd = 128 f32 ───────────────────────────────────────────────────────────
// shmem (10240 B): sq[1024 half] | so[1024 f32] | ss[1024 f32].
kernel void kernel_flash_attn_window_blk_hd128_f32(
    device const float * q         [[buffer(0)]],
    device const float * k         [[buffer(1)]],
    device const float * v         [[buffer(2)]],
    device const float * pad       [[buffer(3)]],
    device       float * out       [[buffer(4)]],
    constant    int    * window    [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
    constant    int    & lo_min    [[buffer(12)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int Q   = 8;
    constexpr int C   = 64;
    constexpr int NSG = 4;
    constexpr int DK  = 128;
    constexpr int NW  = 32;
    constexpr int NQ  = Q / NSG;      // 2
    constexpr int SH  = 2 * C;        // 128
    constexpr int DK4 = DK / 4;       // 32
    constexpr int DK8 = DK / 8;       // 16
    constexpr int PV  = 128;          // PAD2(DV, 64)
    constexpr int PV4 = PV / 4;       // 32
    constexpr int PV8 = PV / 8;       // 16
    constexpr int NC  = (C / 8) / NSG; // 2
    constexpr int NO  = PV8 / NSG;    // 4
    constexpr float MINF = 65504.0f;

    const int iq1 = (int)tgpig.x * Q;
    const int iq2 = (int)tgpig.y;
    const int nblk = (nkv + C - 1) / C;

    const int nqt  = nh * hd;
    const int nkt  = nk * hd;
    const int hk   = iq2 / (nh / nk);
    const int hoff = hk * hd;

    const int tx = (int)tiisg;

    // shmem layout (bytes): sq[0..2048) | so[2048..6144) | ss[6144..10240)
    threadgroup half  * sq = (threadgroup half  *)shmem;
    threadgroup float * so = (threadgroup float *)(shmem + 512);
    threadgroup float * ss = (threadgroup float *)(shmem + 1536);

    threadgroup half4  * sq4 = (threadgroup half4  *)sq;
    threadgroup float4 * so4 = (threadgroup float4 *)so;
    threadgroup float2 * ss2 = (threadgroup float2 *)ss;

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *)(q + (iq1 + j) * nqt + iq2 * hd);
        for (int i = tx; i < DK4; i += NW) {
            sq4[j * DK4 + i] = (iq1 + j < nt) ? half4(q4[i]) : (half4)0.0f;
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] = (float4)0.0f;
        for (int i = tx; i < SH; i += NW) ss[j * SH + i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (short jj = 0; jj < NQ; ++jj) { S[jj] = 0.0f; M[jj] = -FLT_MAX / 2; }

    for (int ic0 = 0; ic0 < nblk; ++ic0) {
        const int ic = ic0 * C;
        const bool partial = (ic + C > nkv);
        const int pos0 = ic;
        device const float * ksrc =
            partial ? (pad + hoff) : (k + lo_min * nkt + ic * nkt + hoff);
        device const float * vsrc =
            partial ? (pad + C * nkt + hoff) : (v + lo_min * nkt + ic * nkt + hoff);

        // ── Q*K^T ──
        {
            threadgroup const half  * pq = sq;
            threadgroup       float * ps = ss + sgitg * 8;
            device     const float * pk = ksrc + sgitg * (8 * nkt);
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8 mq[2];
                simdgroup_float8x8 mk[2];
                #pragma unroll
                for (short i = 0; i < DK8 / 2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
                    simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);
                    simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }
                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8 * (NSG * nkt);
                ps += 8 * NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── online softmax (explicit window mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const bool qvalid = (iq1 + j < nt);
            const int lo = qvalid ? window[iq1 + j] : lo_min;
            const int hi = qvalid ? window[nt + iq1 + j] : (lo_min + (int)nkv);
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = lo_min + pos0 + 2 * tx;
            s2[0] += (kpos0 >= lo && kpos0 < hi) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= lo && kpos0 + 1 < hi) ? 0.0f : -MINF;
            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
            const float ms = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[j * (SH / 2) + tx] = vs2;
            for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── O += P * V ──
        {
            simdgroup_float8x8 lo[NO];
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            {
                device const float * pv = vsrc + 8 * sgitg;
                for (short cc = 0; cc < C / 8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                    simdgroup_float8x8 mv[4];
                    simdgroup_load(mv[0], pv + 0 * NSG, nkt, 0, false);
                    simdgroup_load(mv[1], pv + 8 * NSG, nkt, 0, false);
                    simdgroup_load(mv[2], pv + 16 * NSG, nkt, 0, false);
                    simdgroup_load(mv[3], pv + 24 * NSG, nkt, 0, false);
                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
                    simdgroup_multiply_accumulate(lo[2], vs, mv[2], lo[2]);
                    simdgroup_multiply_accumulate(lo[3], vs, mv[3], lo[3]);
                    pv += 8 * nkt;
                }
            }
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// ─── hd = 128 f16-K/V ───────────────────────────────────────────────────────
kernel void kernel_flash_attn_window_blk_hd128_f16(
    device const float * q         [[buffer(0)]],
    device const half *  k         [[buffer(1)]],
    device const half *  v         [[buffer(2)]],
    device const half *  pad       [[buffer(3)]],
    device       float * out       [[buffer(4)]],
    constant    int    * window    [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
    constant    int    & lo_min    [[buffer(12)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int Q   = 8;
    constexpr int C   = 64;
    constexpr int NSG = 4;
    constexpr int DK  = 128;
    constexpr int NW  = 32;
    constexpr int NQ  = Q / NSG;
    constexpr int SH  = 2 * C;
    constexpr int DK4 = DK / 4;
    constexpr int DK8 = DK / 8;
    constexpr int PV  = 128;
    constexpr int PV4 = PV / 4;
    constexpr int PV8 = PV / 8;
    constexpr int NC  = (C / 8) / NSG;
    constexpr int NO  = PV8 / NSG;
    constexpr float MINF = 65504.0f;

    const int iq1 = (int)tgpig.x * Q;
    const int iq2 = (int)tgpig.y;
    const int nblk = (nkv + C - 1) / C;

    const int nqt  = nh * hd;
    const int nkt  = nk * hd;
    const int hk   = iq2 / (nh / nk);
    const int hoff = hk * hd;

    const int tx = (int)tiisg;

    threadgroup half  * sq = (threadgroup half  *)shmem;
    threadgroup float * so = (threadgroup float *)(shmem + 512);
    threadgroup float * ss = (threadgroup float *)(shmem + 1536);

    threadgroup half4  * sq4 = (threadgroup half4  *)sq;
    threadgroup float4 * so4 = (threadgroup float4 *)so;
    threadgroup float2 * ss2 = (threadgroup float2 *)ss;

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *)(q + (iq1 + j) * nqt + iq2 * hd);
        for (int i = tx; i < DK4; i += NW) {
            sq4[j * DK4 + i] = (iq1 + j < nt) ? half4(q4[i]) : (half4)0.0f;
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] = (float4)0.0f;
        for (int i = tx; i < SH; i += NW) ss[j * SH + i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (short jj = 0; jj < NQ; ++jj) { S[jj] = 0.0f; M[jj] = -FLT_MAX / 2; }

    for (int ic0 = 0; ic0 < nblk; ++ic0) {
        const int ic = ic0 * C;
        const bool partial = (ic + C > nkv);
        const int pos0 = ic;
        device const half * ksrc =
            partial ? (pad + hoff) : (k + lo_min * nkt + ic * nkt + hoff);
        device const half * vsrc =
            partial ? (pad + C * nkt + hoff) : (v + lo_min * nkt + ic * nkt + hoff);

        // ── Q*K^T ──
        {
            threadgroup const half  * pq = sq;
            threadgroup       float * ps = ss + sgitg * 8;
            device     const half  * pk = ksrc + sgitg * (8 * nkt);
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8 mq[2];
                simdgroup_half8x8 mk[2];
                #pragma unroll
                for (short i = 0; i < DK8 / 2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
                    simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);
                    simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }
                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8 * (NSG * nkt);
                ps += 8 * NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── online softmax (explicit window mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const bool qvalid = (iq1 + j < nt);
            const int lo = qvalid ? window[iq1 + j] : lo_min;
            const int hi = qvalid ? window[nt + iq1 + j] : (lo_min + (int)nkv);
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = lo_min + pos0 + 2 * tx;
            s2[0] += (kpos0 >= lo && kpos0 < hi) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= lo && kpos0 + 1 < hi) ? 0.0f : -MINF;
            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
            const float ms = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[j * (SH / 2) + tx] = vs2;
            for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── O += P * V ──
        {
            simdgroup_float8x8 lo[NO];
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            {
                device const half * pv = vsrc + 8 * sgitg;
                for (short cc = 0; cc < C / 8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                    simdgroup_half8x8 mv[4];
                    simdgroup_load(mv[0], pv + 0 * NSG, nkt, 0, false);
                    simdgroup_load(mv[1], pv + 8 * NSG, nkt, 0, false);
                    simdgroup_load(mv[2], pv + 16 * NSG, nkt, 0, false);
                    simdgroup_load(mv[3], pv + 24 * NSG, nkt, 0, false);
                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
                    simdgroup_multiply_accumulate(lo[2], vs, mv[2], lo[2]);
                    simdgroup_multiply_accumulate(lo[3], vs, mv[3], lo[3]);
                    pv += 8 * nkt;
                }
            }
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}
