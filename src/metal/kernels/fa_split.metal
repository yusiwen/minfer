// ─── KV-parallel split attention (nt==1 decode) ──────────────
// The decode bottleneck (measured ~48% of per-token time, grows with KV): for
// nt==1 the classic kernel uses a grid of only (1, nk) threadgroups that loop
// the KV tiles SEQUENTIALLY (latency-bound, GPU underutilized). This two-pass
// split parallelizes the KV dimension:
//   pass 1  kernel_gqa_attn_partial_f32: grid (nt, nk, n_chunks) — each TG
//           computes an online-softmax PARTIAL (mx, S, acc[hd]) for its KV
//           chunk [c*cs, min(nkv,(c+1)*cs)). Same tile/barrier structure as the
//           classic kernel, but each TG loops only its chunk.
//   pass 2  kernel_gqa_attn_combine_f32: grid (nt, nh) — reads the n_chunks
//           partials, merges with the standard max/exp/l-sum, writes output.
// GPU safety: pass 1 preserves the uniform-loop + valid-head + no-early-return
// patterns; pass 2 is a pure elementwise kernel (no shared memory, no barriers).

kernel void kernel_gqa_attn_partial_f32(
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
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t     = (int)tgpig.x;
    int hk    = (int)tgpig.y;
    int chunk = (int)tgpig.z;
    if (t >= nt || hk >= nk) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int h0  = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h  = valid_head ? h0 : 0;

    // chunk bounds: chunk c covers [c*cs, min(nkv,(c+1)*cs)), cs = ceil(nkv/P).
    // Empty chunks (kv_start >= nkv) produce an empty partial (mx=-INF, S=0,
    // acc=0) that the combine ignores via exp(-INF - m) == 0.
    int cs = (nkv + n_chunks - 1) / n_chunks;
    int kv_start = chunk * cs;
    int kv_end   = min(nkv, kv_start + cs);

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;
    int hd4       = hd / 4;   // hd % 4 == 0 is guarded upstream (layer_gpu)

    device const float4 * qhead4 = (device const float4 *)(q + t * stride_q + h * hd);

    threadgroup float4 * k_tile4 = (threadgroup float4 *)shmem;
    threadgroup float4 * v_tile4 = (threadgroup float4 *)(shmem + Bc * hd);

    float mx = -INFINITY;
    float S = 0.0f;
    // Vectorized float4 accumulator: hd<=256 => at most 64 float4s. Kept small
    // (64 floats for hd=64) so the compiler can keep it in REGISTERS — the
    // scalar dynamic-indexed float acc[256] landed in per-thread LOCAL memory,
    // which was the long-context attention bottleneck (per-thread serial DRAM
    // RMWs that don't improve with more parallelism).
    float4 acc4[64];
    for (int d4 = 0; d4 < hd4; d4++) acc4[d4] = 0.0f;

    int n_tiles = (kv_end - kv_start + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int ks = kv_start + tile_idx * Bc;
        int tile_sz = min(Bc, kv_end - ks);

        int total4 = tile_sz * hd4;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total4; i += tgsz) {
            int ki = ks + i / hd4;
            int di = i % hd4;
            device const float4 * k4 = (device const float4 *)(k + ki * stride_kv + hk * hd);
            device const float4 * v4 = (device const float4 *)(v + ki * stride_kv + hk * hd);
            k_tile4[i] = k4[di];
            v_tile4[i] = v4[di];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float4 * kj4 = k_tile4 + j * hd4;
                dot = 0.0f;
                for (int d4 = 0; d4 < hd4; d4++) {
                    float4 qv = qhead4[d4] * kj4[d4];
                    dot += qv.x + qv.y + qv.z + qv.w;
                }
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d4 = 0; d4 < hd4; d4++) acc4[d4] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float4 * vj4 = v_tile4 + j * hd4;
                for (int d4 = 0; d4 < hd4; d4++) acc4[d4] += e * vj4[d4];
                S += e;
            }
            mx = new_mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);

    // partial layout: [t][h][chunk] = {mx, S, acc[hd]}  (contiguous per chunk)
    int pbase = ((t * nh + h) * n_chunks + chunk) * (2 + hd);
    if (valid_head) {
        if (tiisg == 0) {
            partial[pbase + 0] = mx;
            partial[pbase + 1] = S;
        }
        // UNIFORM d loop (all 32 lanes step together, so simd_sum reduces the
        // SAME component across lanes) — a per-lane divergent loop over the
        // acc4 elements would make simd_sum reduce mismatched values.
        for (int d = 0; d < hd; d++) {
            float4 a4 = acc4[d / 4];
            float val;
            switch (d % 4) {
                case 0: val = simd_sum(a4.x); break;
                case 1: val = simd_sum(a4.y); break;
                case 2: val = simd_sum(a4.z); break;
                default: val = simd_sum(a4.w); break;
            }
            if (tiisg == 0) partial[pbase + 2 + d] = val;
        }
    }
}

// F16 KV cache variant of kernel_gqa_attn_partial_f32: K/V read from a half
// cache (2 bytes/elem) and converted to f32 (float4) when staged into the
// threadgroup tiles. The partials + combine are f32, so kernel_gqa_attn_combine_f32
// is shared. Enabled via MINFER_CACHE_TYPE=f16.
kernel void kernel_gqa_attn_partial_f16(
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
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t     = (int)tgpig.x;
    int hk    = (int)tgpig.y;
    int chunk = (int)tgpig.z;
    if (t >= nt || hk >= nk) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int h0  = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h  = valid_head ? h0 : 0;

    int cs = (nkv + n_chunks - 1) / n_chunks;
    int kv_start = chunk * cs;
    int kv_end   = min(nkv, kv_start + cs);

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;
    int hd4       = hd / 4;

    device const float4 * qhead4 = (device const float4 *)(q + t * stride_q + h * hd);

    threadgroup float4 * k_tile4 = (threadgroup float4 *)shmem;
    threadgroup float4 * v_tile4 = (threadgroup float4 *)(shmem + Bc * hd);

    float mx = -INFINITY;
    float S = 0.0f;
    float4 acc4[64];
    for (int d4 = 0; d4 < hd4; d4++) acc4[d4] = 0.0f;

    int n_tiles = (kv_end - kv_start + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int ks = kv_start + tile_idx * Bc;
        int tile_sz = min(Bc, kv_end - ks);

        int total4 = tile_sz * hd4;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total4; i += tgsz) {
            int ki = ks + i / hd4;
            int di = i % hd4;
            device const half4 * k4 = (device const half4 *)(k + ki * stride_kv + hk * hd);
            device const half4 * v4 = (device const half4 *)(v + ki * stride_kv + hk * hd);
            k_tile4[i] = float4(k4[di]);
            v_tile4[i] = float4(v4[di]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float4 * kj4 = k_tile4 + j * hd4;
                dot = 0.0f;
                for (int d4 = 0; d4 < hd4; d4++) {
                    float4 qv = qhead4[d4] * kj4[d4];
                    dot += qv.x + qv.y + qv.z + qv.w;
                }
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d4 = 0; d4 < hd4; d4++) acc4[d4] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float4 * vj4 = v_tile4 + j * hd4;
                for (int d4 = 0; d4 < hd4; d4++) acc4[d4] += e * vj4[d4];
                S += e;
            }
            mx = new_mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);

    int pbase = ((t * nh + h) * n_chunks + chunk) * (2 + hd);
    if (valid_head) {
        if (tiisg == 0) {
            partial[pbase + 0] = mx;
            partial[pbase + 1] = S;
        }
        for (int d = 0; d < hd; d++) {
            float4 a4 = acc4[d / 4];
            float val;
            switch (d % 4) {
                case 0: val = simd_sum(a4.x); break;
                case 1: val = simd_sum(a4.y); break;
                case 2: val = simd_sum(a4.z); break;
                default: val = simd_sum(a4.w); break;
            }
            if (tiisg == 0) partial[pbase + 2 + d] = val;
        }
    }
}

