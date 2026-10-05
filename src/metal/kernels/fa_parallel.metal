// ─── Parallel prefill attention (P1 2026-08-11) ──────────────
// The classic kernel_gqa_attn_f32 at prefill is latency-bound: grid (nt,nk),
// each threadgroup loops the KV sequentially with 2 barriers/tile (~24K
// barriers at nt=430) → measured ~100 ms (48% of prefill, ~25x llama's
// attention). This 3-pass replacement is fully parallel (no threadgroup
// barriers):
//   pass 1 kernel_attn_scores:  scores[t][h][kv] = dot(q[t][h][0..hd], k[kv][hk*hd..]) * scale
//   pass 2 kernel_softmax_attn: masked softmax over kv per (t,h) row (in-place)
//   pass 3 kernel_attn_output:  out[t][h][0..hd] = Σ_kv softmax[t][h][kv] * v[kv][hk*hd..]
// GQA: each query head h uses KV group hk = h/gqa.

// pass 1: scores. Grid: (nt*nh) threadgroups of 256 threads — one threadgroup
// per (t,h) row, threads split across the nkv scores. Each thread computes one
// score = dot(q[t][h][0..hd], k[kv][hk*hd..]) * scale.
kernel void kernel_attn_scores(
    device const float * q    [[buffer(0)]],  // [nt][nh*hd]
    device const float * k    [[buffer(1)]],  // [nkv][nkt]
    device       float * scores [[buffer(2)]], // [nt][nh][nkv]
    constant    int    & nh    [[buffer(3)]],
    constant    int    & hd    [[buffer(4)]],
    constant    int    & nkv   [[buffer(5)]],
    constant    int    & nt    [[buffer(6)]],
    constant    int    & gqa   [[buffer(7)]],
    constant    int    & nkt   [[buffer(8)]],
    constant    float  & scale [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {
    const int th = (int)tgpig.x;   // t*nh+h
    const int t = th / nh;
    const int h = th % nh;
    if (t >= nt) return;
    const int hk = h / gqa;
    device const float * qh = q + t * nh * hd + h * hd;
    // thread i computes scores[th][i] (i in 0..nkv, 256 threads)
    const int kv = (int)tpitg.x;
    if (kv >= nkv) return;
    device const float * kh = k + kv * nkt + hk * hd;
    float s = 0.0f;
    for (int d = 0; d < hd; d++) s += qh[d] * kh[d];
    scores[th * nkv + kv] = s * scale;
}

// pass 3: out[t][h][0..hd] = Σ_kv softmax[t][h][kv] * v[kv][hk*hd..hd].
// Grid: (nt*nh) threadgroups of 256 threads — one per (t,h), threads split
// across the hd output dims (hd<=256 for Qwen).
kernel void kernel_attn_output(
    device const float * scores [[buffer(0)]],  // [nt][nh][nkv] (softmaxed)
    device const float * v      [[buffer(1)]],  // [nkv][nkt]
    device       float * out    [[buffer(2)]],  // [nt][nh*hd]
    constant    int    & nh    [[buffer(3)]],
    constant    int    & hd    [[buffer(4)]],
    constant    int    & nkv   [[buffer(5)]],
    constant    int    & nt    [[buffer(6)]],
    constant    int    & gqa   [[buffer(7)]],
    constant    int    & nkt   [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {
    const int th = (int)tgpig.x;   // t*nh+h
    const int t = th / nh;
    const int h = th % nh;
    if (t >= nt) return;
    const int hk = h / gqa;
    const int d = (int)tpitg.x;   // 0..hd-1
    if (d >= hd) return;
    device const float * sc = scores + th * nkv;
    device const float * vh = v + hk * hd + d;
    float acc = 0.0f;
    for (int kv = 0; kv < nkv; kv++) acc += sc[kv] * vh[kv * nkt];
    out[t * nh * hd + h * hd + d] = acc;
}

kernel void kernel_softmax_attn(
    device       float * scores [[buffer(0)]],   // [nt*nh][nkv] (scores already scaled)
    constant    int    * positions [[buffer(1)]],
    constant    int    & nkv    [[buffer(2)]],
    constant    int    & nt     [[buffer(3)]],
    constant    int    & nh     [[buffer(4)]],
    threadgroup float * shmem  [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int ntg = 256; // dispatched threads per threadgroup (8 simdgroups)
    const int th = (int)tgpig.x;   // row = t*nh + h
    const int t = th / nh;
    const int vl = positions[t] + 1;   // valid KV length for this token
    device float * row = scores + th * nkv;

    if (sgitg == 0) { shmem[tiisg] = -INFINITY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 1: max over valid positions (scores already scaled)
    float m = -INFINITY;
    for (int i = 32*sgitg + tiisg; i < vl; i += ntg) {
        m = max(m, row[i]);
    }
    m = simd_max(m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) { shmem[sgitg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = shmem[tiisg];
    m = simd_max(m);

    // pass 2: exp(sum) and write normalized + masked
    // (re-init shmem to 0 — the max pass left per-simdgroup maxes in it)
    if (sgitg == 0) { shmem[tiisg] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (int i = 32*sgitg + tiisg; i < vl; i += ntg) {
        float e = exp(row[i] - m);
        row[i] = e;
        s += e;
    }
    s = simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) { shmem[sgitg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = shmem[tiisg];
    s = simd_sum(s);

    const float inv = 1.0f / s;
    for (int i = 32*sgitg + tiisg; i < nkv; i += ntg) {
        if (i < vl) row[i] *= inv;
        else row[i] = 0.0f;
    }
}

