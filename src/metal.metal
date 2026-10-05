// ─── Flash attention for prefill (nt > 1) — port of llama kernel_flash_attn_ext_blk
// (the legacy simdgroup_matrix flash, docs/METAL_OPTIMIZATIONS.md §4.3.1). Fixed-shape:
// NSG=4, Q=8, C=64, DK=DV=64. Grid (ceil(nt/8), nh), 128 threads (32 lanes x 4
// simdgroups). Each threadgroup computes Q=8 query tokens x ALL KV for head h
// (GQA head hk = h/gqa is baked into the K/V base). Faithful llama transcription
// with two GPU-safety deviations: the causal mask is computed inline (no
// mask/pad pre-pass kernels), and the PARTIAL last KV block (nkv % 64 != 0) is
// read from the [2][64][nkt] tail-pad buffer filled by kernel_kv_tail_pad
// (padded rows are zero + masked to -MINF, so they never contribute).
//
// shmem (7168 B): sq[512 half] | so[512 f32] | ss[1024 f32]
kernel void kernel_flash_attn_blk_f32(
    device const float * q         [[buffer(0)]],
    device const float * k         [[buffer(1)]],
    device const float * v         [[buffer(2)]],
    device const float * pad       [[buffer(3)]],   // [2][64][nkt] K-tail then V-tail
    device       float * out       [[buffer(4)]],
    constant    int    * positions [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
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
        const int pos0 = partial ? (nkv - C) : ic;
        // K/V source: direct cache rows (K at ic*nkt + head hoff) or the tail pad.
        device const float * ksrc = partial ? (pad + hoff) : (k + ic * nkt + hoff);
        device const float * vsrc = partial ? (pad + C * nkt + hoff) : (v + ic * nkt + hoff);

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

        // ── online softmax (causal + pad mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int qpos = (iq1 + j < nt) ? positions[iq1 + j] : (int)nkv - 1;
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = pos0 + 2 * tx;
            s2[0] += (kpos0 >= 0 && kpos0 <= qpos) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= 0 && kpos0 + 1 <= qpos) ? 0.0f : -MINF;
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

// f16-K/V variant (MINFER_CACHE_TYPE=f16): reads the half K/V cache and tail pad.
// Q is ALWAYS f32 (llama reads Q as float4 regardless of the KV cache type — the
// graph only casts K/V to f16, llama-graph.cpp:2457-2463), so this kernel keeps
// the f32 Q input and only switches the global K/V/pad operands + the simdgroup
// K/V tile types to half8x8.
kernel void kernel_flash_attn_blk_f16(
    device const float * q         [[buffer(0)]],
    device const half *  k         [[buffer(1)]],
    device const half *  v         [[buffer(2)]],
    device const half *  pad       [[buffer(3)]],
    device       float * out      [[buffer(4)]],
    constant    int    * positions [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
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
        const int pos0 = partial ? (nkv - C) : ic;
        device const half * ksrc = partial ? (pad + hoff) : (k + ic * nkt + hoff);
        device const half * vsrc = partial ? (pad + C * nkt + hoff) : (v + ic * nkt + hoff);

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

        // ── online softmax (causal + pad mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int qpos = (iq1 + j < nt) ? positions[iq1 + j] : (int)nkv - 1;
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = pos0 + 2 * tx;
            s2[0] += (kpos0 >= 0 && kpos0 <= qpos) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= 0 && kpos0 + 1 <= qpos) ? 0.0f : -MINF;
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

    // ── store to global ──
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// hd=128 (7B) variant of kernel_flash_attn_blk_f32. Same structure as the hd=64
// kernel above (faithful llama kernel_flash_attn_ext_blk transcription) with
// the §4.3.3 constant deltas: DK=DV=128, DK4=DV4=32, DK8=16, PV=128, PV4=32,
// PV8=16, NO=4 (the P*V loop uses 4 mv[]/lo[] accumulators — llama's DV>64
// branch split differently but the same math). shmem (10240 B):
// sq[1024 half] | so[1024 f32] | ss[1024 f32].
kernel void kernel_flash_attn_blk_hd128_f32(
    device const float * q         [[buffer(0)]],
    device const float * k         [[buffer(1)]],
    device const float * v         [[buffer(2)]],
    device const float * pad       [[buffer(3)]],   // [2][64][nkt] K-tail then V-tail
    device       float * out       [[buffer(4)]],
    constant    int    * positions [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
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
        const int pos0 = partial ? (nkv - C) : ic;
        // K/V source: direct cache rows (K at ic*nkt + head hoff) or the tail pad.
        device const float * ksrc = partial ? (pad + hoff) : (k + ic * nkt + hoff);
        device const float * vsrc = partial ? (pad + C * nkt + hoff) : (v + ic * nkt + hoff);

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

        // ── online softmax (causal + pad mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int qpos = (iq1 + j < nt) ? positions[iq1 + j] : (int)nkv - 1;
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = pos0 + 2 * tx;
            s2[0] += (kpos0 >= 0 && kpos0 <= qpos) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= 0 && kpos0 + 1 <= qpos) ? 0.0f : -MINF;
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

    // ── store to global ──
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// hd=128 f16-K/V variant (MINFER_CACHE_TYPE=f16): reads the half K/V cache and
// tail pad; Q stays f32 (same as the hd=64 f16 kernel — the graph only casts
// K/V to f16).
kernel void kernel_flash_attn_blk_hd128_f16(
    device const float * q         [[buffer(0)]],
    device const half *  k         [[buffer(1)]],
    device const half *  v         [[buffer(2)]],
    device const half *  pad       [[buffer(3)]],
    device       float * out      [[buffer(4)]],
    constant    int    * positions [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
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
        const int pos0 = partial ? (nkv - C) : ic;
        device const half * ksrc = partial ? (pad + hoff) : (k + ic * nkt + hoff);
        device const half * vsrc = partial ? (pad + C * nkt + hoff) : (v + ic * nkt + hoff);

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

        // ── online softmax (causal + pad mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int qpos = (iq1 + j < nt) ? positions[iq1 + j] : (int)nkv - 1;
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = pos0 + 2 * tx;
            s2[0] += (kpos0 >= 0 && kpos0 <= qpos) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= 0 && kpos0 + 1 <= qpos) ? 0.0f : -MINF;
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

    // ── store to global ──
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// Copy the last partial KV block (nkv % 64 != 0) into the [2][64][nkt] flash-prefill
// tail pad: virtual rows [nkv-64, nkv) from the real cache (head offset hoff, both
// K and V), rows outside [0, nkv) zeroed — the causal+pad mask hides them. Grid
// (nkt, 64), one thread per (dim, virtual row). Handles f32 (e=4) or f16 (e=2)
// cache via the `f16` flag.
kernel void kernel_kv_tail_pad(
    device const char * ksrc  [[buffer(0)]],
    device const char * vsrc  [[buffer(1)]],
    device       char * pad   [[buffer(2)]],
    constant    int    & nkv   [[buffer(3)]],
    constant    int    & nkt   [[buffer(4)]],
    constant    int    & f16   [[buffer(5)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    const int d   = (int)tgpig.x;
    const int t   = (int)tgpig.y;
    const int e   = f16 ? 2 : 4;
    const int pos = nkv - 64 + t;
    const bool valid = pos >= 0 && pos < nkv;
    const int dst = (t * nkt + d) * e;
    if (valid) {
        const int src = (pos * nkt + d) * e;
        for (int b = 0; b < e; ++b) {
            pad[dst + b] = ksrc[src + b];
            pad[64 * nkt * e + dst + b] = vsrc[src + b];
        }
    } else {
        for (int b = 0; b < e; ++b) {
            pad[dst + b] = 0;
            pad[64 * nkt * e + dst + b] = 0;
        }
    }
}

kernel void kernel_gqa_attn_combine_f32(
    device const float * partial [[buffer(0)]],
    device       float * o       [[buffer(1)]],
    constant    int    & nh       [[buffer(2)]],
    constant    int    & hd       [[buffer(3)]],
    constant    int    & nt       [[buffer(4)]],
    constant    int    & n_chunks [[buffer(5)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]]
) {
    int t = (int)tgpig.x;
    int h = (int)tgpig.y;
    if (t >= nt || h >= nh) return;

    int pbase = (t * nh + h) * n_chunks * (2 + hd);

    // merged running max over the partials
    float m = -INFINITY;
    for (int c = 0; c < n_chunks; c++) {
        m = max(m, partial[pbase + c * (2 + hd) + 0]);
    }
    if (m == -INFINITY) {
        // no partial had data (nkv==0) — not reachable for real heads, but
        // write zeros rather than NaN (exp(-INF - -INF)) for GPU safety.
        device float * ohead = o + t * (nh * hd) + h * hd;
        for (int d = tiisg; d < hd; d += 32) ohead[d] = 0.0f;
        return;
    }

    float l = 0.0f;
    float acc[256];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        int cbase = pbase + c * (2 + hd);
        float e = exp(partial[cbase + 0] - m);
        l += partial[cbase + 1] * e;
        for (int d = tiisg; d < hd; d += 32) {
            acc[d] += partial[cbase + 2 + d] * e;
        }
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    device float * ohead = o + t * (nh * hd) + h * hd;
    for (int d = tiisg; d < hd; d += 32) {
        ohead[d] = acc[d] * inv;
    }
}

