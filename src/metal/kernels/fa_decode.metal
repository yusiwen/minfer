// ─── Flash attention (decode nt==1) — port of llama kernel_flash_attn_ext_vec
// (option C): single-simdgroup fixed-shape port, DK=DV=64, NE=2, C=32,
// NWG=n_chunks, NSG=1. Each threadgroup computes an online-softmax PARTIAL
// {M, S, O[hd]} over the strided KV chunks {iwg, iwg+n_chunks, ...}×C — the
// SAME partial format as kernel_gqa_attn_partial_f32, so
// kernel_gqa_attn_combine_f32 merges them unchanged.
//
// GPU-safety (deadlock discipline):
//  - No per-lane early returns. `if (ic >= nkv) break` depends only on
//    tgpig/lane-independent values → all 32 lanes break together.
//  - No `continue`. Out-of-range KV lanes are masked to -MINF_MAXHALF (inline,
//    lane-local) so exp() yields ~0; the read is clamped to nkv-1 (in-bounds,
//    value ignored).
//  - All lanes reach every threadgroup_barrier. NSG=1 fixed: no cross-simdgroup
//    reduce, no threadgroup_barrier in the reduce phase (llama's `r` loop runs
//    only for NSG>1).
//  - The uniform d-loop for the acc reduction is NOT needed here — each lane
//    writes its own so4[tiisg] float4 slot (ty==0 lanes, 16 slots = hd/4).
//  - hd==64 is required (DK/DV fixed); host layer_gpu guards hd==64 &&
//    hd%4==0 before dispatching (else falls back to the split-attention path).

kernel void kernel_flash_attn_ext_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int DK  = 64;
    constexpr int NE  = 2;
    constexpr int C   = 32;
    constexpr int NW  = 32;
    constexpr int NL  = NW / NE;   // 16
    constexpr int DK4 = DK / 4;    // 16
    constexpr float MINF_MAXHALF = 65504.0f; // half max; -MINF_MAXHALF ~= -INF mask

    const int t   = (int)tgpig.x;
    const int h   = (int)tgpig.y;
    const int iwg = (int)tgpig.z;
    if (t >= nt || h >= nh) return;

    const int nkv = positions[t] + 1;
    const int gqa = nh / nk;
    const int hk  = h / gqa;
    const int stride_kv  = nk * hd;         // f32 elements per token row
    const int hk4        = hk * hd / 4;     // head offset in float4

    const int tx = (int)tiisg % NL;   // 0..NL-1 (DK4 dim)
    const int ty = (int)tiisg / NL;   // 0..NE-1 (token lane)

    // shmem layout (f32): sq4[DK4 float4] | ss[C] | so4[NW float4]
    // (no sm[] array: the partial-chunk mask is computed inline below so every
    //  lane reads/writes only its own registers — no cross-lane threadgroup
    //  access outside the two barrier-protected ss[] handoffs)
    threadgroup float4 * sq4 = (threadgroup float4 *)shmem;
    threadgroup float  * ss  = shmem + DK4 * 4;
    threadgroup float4 * so4 = (threadgroup float4 *)(ss + C);

    // load Q head into shared memory (DK4 float4)
    device const float4 * q4 = (device const float4 *)(q + t * (nh * hd) + h * hd);
    for (int i = (int)tiisg; i < DK4; i += NW) sq4[i] = q4[i];
    // zero ss and this lane's O slot
    for (int i = (int)tiisg; i < C; i += NW) ss[i] = 0.0f;
    so4[tiisg] = (float4)0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    // KV chunk loop: chunks iwg, iwg+n_chunks, ... each of C tokens
    for (int ic0 = iwg; ; ic0 += n_chunks) {
        int ic = ic0 * C;
        if (ic >= nkv) break;

        // Q*K^T
        float mqk[C / NE];
        for (int cc = 0; cc < C / NE; ++cc) {
            int token = ic + NE * cc + ty;
            if (token >= nkv) token = nkv - 1; // clamped read, value masked out
            device const float4 * pk = (device const float4 *)
                (k + token * stride_kv) + hk4 + tx;
            float4 qv = sq4[tx] * pk[0];
            mqk[cc] = qv.x + qv.y + qv.z + qv.w;
            // simdgroup reduce over the DK4 lanes (tx): full-head dot
            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
            // broadcast the reduced value from lane NL*ty
            mqk[cc] = simd_shuffle(mqk[cc], NL * ty);
        }
        // store scaled score (+ partial-chunk mask) in ss[2*tx+ty] == token
        // ic+2tx+ty; out-of-range lanes get -MINF_MAXHALF (exp() ~= 0 contribution).
        // Mask is computed inline (lane-local) so no threadgroup memory race.
        ss[NE * tx + ty] = mqk[tx] * scale
                         + ((ic + NE * tx + ty < nkv) ? 0.0f : -MINF_MAXHALF);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        {
            const float m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) so4[tiisg] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = O + (Q*K^T)*V
        {
            float4 lo = (float4)0.0f;
            for (int cc = 0; cc < C / NE; ++cc) {
                int token = ic + NE * cc + ty;
                if (token >= nkv) token = nkv - 1; // clamped read, value masked out
                device const float4 * pv4 = (device const float4 *)
                    (v + token * stride_kv) + hk4 + tx;
                lo += pv4[0] * ss[NE * cc + ty];
            }
            // merge the NE=2 ty lanes (token ic+2cc and ic+2cc+1)
            lo += simd_shuffle_down(lo, 16);
            if (ty == 0) so4[tiisg] += lo;
        }
    }

    // write partial (same layout as partial_f32): {M, S, O[hd]} per (t,h,iwg)
    int pbase = ((t * nh + h) * n_chunks + iwg) * (2 + hd);
    if (tiisg == 0) {
        partial[pbase + 0] = M;
        partial[pbase + 1] = S;
    }
    if (ty == 0) {
        float4 acc = so4[tiisg];
        partial[pbase + 2 + tx * 4 + 0] = acc.x;
        partial[pbase + 2 + tx * 4 + 1] = acc.y;
        partial[pbase + 2 + tx * 4 + 2] = acc.z;
        partial[pbase + 2 + tx * 4 + 3] = acc.w;
    }
}

// F16 KV cache variant of kernel_flash_attn_ext_f32: K/V read from a half cache
// (2 bytes/elem) and converted to f32 when dotted. Partials + combine are f32
// and shared with the f32 variant (kernel_gqa_attn_combine_f32).
kernel void kernel_flash_attn_ext_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int DK  = 64;
    constexpr int NE  = 2;
    constexpr int C   = 32;
    constexpr int NW  = 32;
    constexpr int NL  = NW / NE;
    constexpr int DK4 = DK / 4;
    constexpr float MINF_MAXHALF = 65504.0f;

    const int t   = (int)tgpig.x;
    const int h   = (int)tgpig.y;
    const int iwg = (int)tgpig.z;
    if (t >= nt || h >= nh) return;

    const int nkv = positions[t] + 1;
    const int gqa = nh / nk;
    const int hk  = h / gqa;
    const int stride_kv  = nk * hd;         // half elements per token row
    const int hk4        = hk * hd / 4;

    const int tx = (int)tiisg % NL;
    const int ty = (int)tiisg / NL;

    threadgroup float4 * sq4 = (threadgroup float4 *)shmem;
    threadgroup float  * ss  = shmem + DK4 * 4;
    threadgroup float4 * so4 = (threadgroup float4 *)(ss + C);

    device const float4 * q4 = (device const float4 *)(q + t * (nh * hd) + h * hd);
    for (int i = (int)tiisg; i < DK4; i += NW) sq4[i] = q4[i];
    for (int i = (int)tiisg; i < C; i += NW) ss[i] = 0.0f;
    so4[tiisg] = (float4)0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    for (int ic0 = iwg; ; ic0 += n_chunks) {
        int ic = ic0 * C;
        if (ic >= nkv) break;

        float mqk[C / NE];
        for (int cc = 0; cc < C / NE; ++cc) {
            int token = ic + NE * cc + ty;
            if (token >= nkv) token = nkv - 1;
            device const half4 * pk = (device const half4 *)
                (k + token * stride_kv) + hk4 + tx;
            float4 kv4 = float4(pk[0]);
            float4 qv = sq4[tx] * kv4;
            mqk[cc] = qv.x + qv.y + qv.z + qv.w;
            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
            mqk[cc] = simd_shuffle(mqk[cc], NL * ty);
        }
        ss[NE * tx + ty] = mqk[tx] * scale
                         + ((ic + NE * tx + ty < nkv) ? 0.0f : -MINF_MAXHALF);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) so4[tiisg] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            float4 lo = (float4)0.0f;
            for (int cc = 0; cc < C / NE; ++cc) {
                int token = ic + NE * cc + ty;
                if (token >= nkv) token = nkv - 1;
                device const half4 * pv4 = (device const half4 *)
                    (v + token * stride_kv) + hk4 + tx;
                lo += float4(pv4[0]) * ss[NE * cc + ty];
            }
            lo += simd_shuffle_down(lo, 16);
            if (ty == 0) so4[tiisg] += lo;
        }
    }

    int pbase = ((t * nh + h) * n_chunks + iwg) * (2 + hd);
    if (tiisg == 0) {
        partial[pbase + 0] = M;
        partial[pbase + 1] = S;
    }
    if (ty == 0) {
        float4 acc = so4[tiisg];
        partial[pbase + 2 + tx * 4 + 0] = acc.x;
        partial[pbase + 2 + tx * 4 + 1] = acc.y;
        partial[pbase + 2 + tx * 4 + 2] = acc.z;
        partial[pbase + 2 + tx * 4 + 3] = acc.w;
    }
}

// HD=128 variant of kernel_flash_attn_ext_f32 (llama vec dk128_dv128): DK=DV=128,
// NE=1, C=32, NW=32, NL=32. With NE=1 every lane is a DK4=DV4=32 dim lane
// (tx=tiisg, ty=0) covering one float4 of the 128-dim head, so:
//  - the QK^T dot is a single simd_sum over all 32 lanes (full-head reduce,
//    llama vec NE==1 branch) — no shuffle_down tree / broadcast needed;
//  - the online softmax has no ty lanes to merge (all lanes run the ms/so4
//    scaling; so4[tiisg] is this lane's own O slot);
//  - the O accumulator is per-lane over all KV tokens (no cross-lane merge);
//  - every lane writes its own partial O slice (32 slots = hd/4).
// Same partial format {M, S, O[hd]} + combine kernel as the hd==64 variant.
//
// GPU-safety: identical discipline to kernel_flash_attn_ext_f32 — no per-lane
// early return, no `continue`, all lanes reach every threadgroup_barrier, and
// the (t,h,iwg) guard returns uniformly.
kernel void kernel_flash_attn_ext_hd128_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int DK  = 128;
    constexpr int NE  = 1;
    constexpr int C   = 32;
    constexpr int NW  = 32;
    constexpr int NL  = NW / NE;   // 32
    constexpr int DK4 = DK / 4;    // 32
    constexpr float MINF_MAXHALF = 65504.0f;

    const int t   = (int)tgpig.x;
    const int h   = (int)tgpig.y;
    const int iwg = (int)tgpig.z;
    if (t >= nt || h >= nh) return;

    const int nkv = positions[t] + 1;
    const int gqa = nh / nk;
    const int hk  = h / gqa;
    const int stride_kv  = nk * hd;         // f32 elements per token row
    const int hk4        = hk * hd / 4;     // head offset in float4

    const int tx = (int)tiisg % NL;   // 0..NL-1 (DK4 dim) == tiisg (NE=1)

    // shmem layout (f32): sq4[DK4 float4] | ss[C] | so4[NW float4]
    threadgroup float4 * sq4 = (threadgroup float4 *)shmem;
    threadgroup float  * ss  = shmem + DK4 * 4;
    threadgroup float4 * so4 = (threadgroup float4 *)(ss + C);

    device const float4 * q4 = (device const float4 *)(q + t * (nh * hd) + h * hd);
    for (int i = (int)tiisg; i < DK4; i += NW) sq4[i] = q4[i];
    for (int i = (int)tiisg; i < C; i += NW) ss[i] = 0.0f;
    so4[tiisg] = (float4)0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    // KV chunk loop: chunks iwg, iwg+n_chunks, ... each of C tokens
    for (int ic0 = iwg; ; ic0 += n_chunks) {
        int ic = ic0 * C;
        if (ic >= nkv) break;

        // Q*K^T — NE=1: each lane holds one float4 of the 128-dim head, so the
        // per-token dot reduces over all 32 lanes (simd_sum broadcasts).
        float mqk[C / NE];
        for (int cc = 0; cc < C / NE; ++cc) {
            int token = ic + cc; // NE=1, ty=0
            if (token >= nkv) token = nkv - 1; // clamped read, value masked out
            device const float4 * pk = (device const float4 *)
                (k + token * stride_kv) + hk4 + tx;
            float4 qv = sq4[tx] * pk[0];
            mqk[cc] = qv.x + qv.y + qv.z + qv.w;
            mqk[cc] = simd_sum(mqk[cc]);
        }
        // store scaled score (+ partial-chunk mask) in ss[tx] == token ic+tx
        ss[tx] = mqk[tx] * scale
               + ((ic + tx < nkv) ? 0.0f : -MINF_MAXHALF);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        {
            const float m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            so4[tiisg] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = O + (Q*K^T)*V — NE=1: each lane accumulates its own 4 output dims
        // over all KV tokens; no cross-lane merge needed.
        {
            float4 lo = (float4)0.0f;
            for (int cc = 0; cc < C / NE; ++cc) {
                int token = ic + cc;
                if (token >= nkv) token = nkv - 1; // clamped read, value masked out
                device const float4 * pv4 = (device const float4 *)
                    (v + token * stride_kv) + hk4 + tx;
                lo += pv4[0] * ss[cc];
            }
            so4[tiisg] += lo;
        }
    }

    int pbase = ((t * nh + h) * n_chunks + iwg) * (2 + hd);
    if (tiisg == 0) {
        partial[pbase + 0] = M;
        partial[pbase + 1] = S;
    }
    float4 acc = so4[tiisg];
    partial[pbase + 2 + tx * 4 + 0] = acc.x;
    partial[pbase + 2 + tx * 4 + 1] = acc.y;
    partial[pbase + 2 + tx * 4 + 2] = acc.z;
    partial[pbase + 2 + tx * 4 + 3] = acc.w;
}

// F16 KV cache variant of kernel_flash_attn_ext_hd128_f32: K/V read from a half
// cache (2 bytes/elem) and converted to f32 when dotted. Partials + combine are
// f32 and shared with the f32 variant (kernel_gqa_attn_combine_f32).
kernel void kernel_flash_attn_ext_hd128_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int DK  = 128;
    constexpr int NE  = 1;
    constexpr int C   = 32;
    constexpr int NW  = 32;
    constexpr int NL  = NW / NE;   // 32
    constexpr int DK4 = DK / 4;    // 32
    constexpr float MINF_MAXHALF = 65504.0f;

    const int t   = (int)tgpig.x;
    const int h   = (int)tgpig.y;
    const int iwg = (int)tgpig.z;
    if (t >= nt || h >= nh) return;

    const int nkv = positions[t] + 1;
    const int gqa = nh / nk;
    const int hk  = h / gqa;
    const int stride_kv  = nk * hd;         // half elements per token row
    const int hk4        = hk * hd / 4;

    const int tx = (int)tiisg % NL;   // 0..NL-1 (DK4 dim) == tiisg (NE=1)

    threadgroup float4 * sq4 = (threadgroup float4 *)shmem;
    threadgroup float  * ss  = shmem + DK4 * 4;
    threadgroup float4 * so4 = (threadgroup float4 *)(ss + C);

    device const float4 * q4 = (device const float4 *)(q + t * (nh * hd) + h * hd);
    for (int i = (int)tiisg; i < DK4; i += NW) sq4[i] = q4[i];
    for (int i = (int)tiisg; i < C; i += NW) ss[i] = 0.0f;
    so4[tiisg] = (float4)0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    for (int ic0 = iwg; ; ic0 += n_chunks) {
        int ic = ic0 * C;
        if (ic >= nkv) break;

        float mqk[C / NE];
        for (int cc = 0; cc < C / NE; ++cc) {
            int token = ic + cc;
            if (token >= nkv) token = nkv - 1;
            device const half4 * pk = (device const half4 *)
                (k + token * stride_kv) + hk4 + tx;
            float4 kv4 = float4(pk[0]);
            float4 qv = sq4[tx] * kv4;
            mqk[cc] = qv.x + qv.y + qv.z + qv.w;
            mqk[cc] = simd_sum(mqk[cc]);
        }
        ss[tx] = mqk[tx] * scale
               + ((ic + tx < nkv) ? 0.0f : -MINF_MAXHALF);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            so4[tiisg] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            float4 lo = (float4)0.0f;
            for (int cc = 0; cc < C / NE; ++cc) {
                int token = ic + cc;
                if (token >= nkv) token = nkv - 1;
                device const half4 * pv4 = (device const half4 *)
                    (v + token * stride_kv) + hk4 + tx;
                lo += float4(pv4[0]) * ss[cc];
            }
            so4[tiisg] += lo;
        }
    }

    int pbase = ((t * nh + h) * n_chunks + iwg) * (2 + hd);
    if (tiisg == 0) {
        partial[pbase + 0] = M;
        partial[pbase + 1] = S;
    }
    float4 acc = so4[tiisg];
    partial[pbase + 2 + tx * 4 + 0] = acc.x;
    partial[pbase + 2 + tx * 4 + 1] = acc.y;
    partial[pbase + 2 + tx * 4 + 2] = acc.z;
    partial[pbase + 2 + tx * 4 + 3] = acc.w;
}

