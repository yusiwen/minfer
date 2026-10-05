// ─── KV cache store ──────────────────────────────────────────
// Scatters nt new K/V rows into the persistent KV cache at positions[].

 kernel void kernel_store_kv_f32(
    device const float * src [[buffer(0)]],
    device       float * dst [[buffer(1)]],
    constant    int    & nkt [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int j = tid.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = src[t * nkt + j];
}

// F16 KV cache variant: stores f32 K/V rows into a half cache (2 bytes/elem),
// halving attention memory bandwidth (matches llama.cpp's F16 cache).
kernel void kernel_store_kv_f16(
    device const float * src [[buffer(0)]],
    device       half  * dst [[buffer(1)]],
    constant    int    & nkt [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int j = tid.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = half(src[t * nkt + j]);
}

// ─── Fused bias-add + RoPE + KV-store (nt==1 decode) ──────────
// One kernel replaces add_bias×3 + rope×2 + store_kv×2 (7 dispatches → 1).
// bqkv layout (nt==1): [q: 0..nqt][k: nqt..nqt+nkt][v: nqt+nkt..nqt+2nkt].
// Applies the per-section bias, RoPE to the q/k sections, and stores k/v into
// the KV cache (f32 or f16 per kv_is_f16). Bit-identical to the 7 separate
// kernels: bias then rope on the same values, same store addresses.
// Thread mapping: one thread per (head, d<half_dim) rope pair for q and k,
// one thread per v element → grid = nqt/2 + nkt/2 + nkt.

kernel void kernel_attn_bias_rope_store(
    device       float * bqkv [[buffer(0)]],
    device const float * bias_q [[buffer(1)]],
    device const float * bias_k [[buffer(2)]],
    device const float * bias_v [[buffer(3)]],
    device        void * kv_k [[buffer(4)]],
    device        void * kv_v [[buffer(5)]],
    constant        int & nqt [[buffer(6)]],
    constant        int & nkt [[buffer(7)]],
    constant        int & hd  [[buffer(8)]],
    constant      float & freq_base [[buffer(9)]],
    constant      float & freq_scale [[buffer(10)]],
    constant        int & pos [[buffer(11)]],
    constant        int & rope_style [[buffer(12)]],
    constant        int & kv_is_f16 [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = (int)tid;
    if (u >= total) return;

    if (u < qpairs) {
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int i0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int i1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        float x0 = bqkv[i0] + bias_q[i0];
        float x1 = bqkv[i1] + bias_q[i1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        bqkv[i0] = x0 * cs - x1 * sn;
        bqkv[i1] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kpairs) {
        const int u2   = u - qpairs;
        const int head = u2 / half_dim;
        const int d    = u2 % half_dim;
        const int base = head * hd;
        const int j0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int j1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        const int k0 = nqt + j0, k1 = nqt + j1;
        float x0 = bqkv[k0] + bias_k[j0];
        float x1 = bqkv[k1] + bias_k[j1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        float r0 = x0 * cs - x1 * sn;
        float r1 = x0 * sn + x1 * cs;
        bqkv[k0] = r0; bqkv[k1] = r1;
        if (kv_is_f16) {
            ((device half *)kv_k)[pos * nkt + j0] = half(r0);
            ((device half *)kv_k)[pos * nkt + j1] = half(r1);
        } else {
            ((device float *)kv_k)[pos * nkt + j0] = r0;
            ((device float *)kv_k)[pos * nkt + j1] = r1;
        }
    } else {
        const int j  = u - qpairs - kpairs;
        const int vi = nqt + nkt + j;
        float v = bqkv[vi] + bias_v[j];
        bqkv[vi] = v;
        if (kv_is_f16) {
            ((device half *)kv_v)[pos * nkt + j] = half(v);
        } else {
            ((device float *)kv_v)[pos * nkt + j] = v;
        }
    }
}

// ─── Fused decode QKV with per-head Q/K RMSNorm (Qwen3, no bias) ───
// Analogue of kernel_attn_bias_rope_store but WITHOUT the attention biases
// (Qwen3 has none). The per-head RMSNorm is applied by the separate
// rms_norm_256 dispatches BEFORE this kernel is launched (the concat buffer's
// q/k sections are normed in place); this kernel only does RoPE on q/k and
// stores K/V into the persistent KV regions. Same section layout as the
// bias+rope+store kernel: q = bqkv[0..nqt], k = bqkv[nqt..nqt+nkt],
// v = bqkv[nqt+nkt..]. Grid: nqt/2 + nkt/2 + nkt, 256 threads.
kernel void kernel_attn_rope_store(
    device       float * bqkv [[buffer(0)]],
    device        void * kv_k [[buffer(1)]],
    device        void * kv_v [[buffer(2)]],
    constant        int & nqt [[buffer(3)]],
    constant        int & nkt [[buffer(4)]],
    constant        int & hd  [[buffer(5)]],
    constant      float & freq_base [[buffer(6)]],
    constant      float & freq_scale [[buffer(7)]],
    constant        int & pos [[buffer(8)]],
    constant        int & rope_style [[buffer(9)]],
    constant        int & kv_is_f16 [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = (int)tid;
    if (u >= total) return;

    if (u < qpairs) {
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int i0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int i1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        float x0 = bqkv[i0];
        float x1 = bqkv[i1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        bqkv[i0] = x0 * cs - x1 * sn;
        bqkv[i1] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kpairs) {
        const int u2   = u - qpairs;
        const int head = u2 / half_dim;
        const int d    = u2 % half_dim;
        const int base = head * hd;
        const int j0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int j1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        const int k0 = nqt + j0, k1 = nqt + j1;
        float x0 = bqkv[k0];
        float x1 = bqkv[k1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        float r0 = x0 * cs - x1 * sn;
        float r1 = x0 * sn + x1 * cs;
        bqkv[k0] = r0; bqkv[k1] = r1;
        if (kv_is_f16) {
            ((device half *)kv_k)[pos * nkt + j0] = half(r0);
            ((device half *)kv_k)[pos * nkt + j1] = half(r1);
        } else {
            ((device float *)kv_k)[pos * nkt + j0] = r0;
            ((device float *)kv_k)[pos * nkt + j1] = r1;
        }
    } else {
        const int j  = u - qpairs - kpairs;
        const int vi = nqt + nkt + j;
        float v = bqkv[vi];
        bqkv[vi] = v;
        if (kv_is_f16) {
            ((device half *)kv_v)[pos * nkt + j] = half(v);
        } else {
            ((device float *)kv_v)[pos * nkt + j] = v;
        }
    }
}

kernel void kernel_gqa_attn_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t  = (int)tgpig.x;
    int hk = (int)tgpig.y;
    if (t >= nt || hk >= nk) return;

    int nkv  = positions[t] + 1;
    int gqa  = nh / nk;
    // Do NOT return early for heads beyond nh: all simdgroups in the
    // threadgroup must reach every threadgroup_barrier below, or the GPU
    // deadlocks when nh % nk != 0. Invalid heads run the loop with a dummy
    // head index but skip the output write.
    int  h0         = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h          = valid_head ? h0 : 0;

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;

    device const float * qhead = q + t * stride_q + h * hd;
    device       float * ohead = o + t * stride_q + h * hd;

    threadgroup float * k_tile = shmem;
    threadgroup float * v_tile = shmem + Bc * hd;

    float mx = -INFINITY;
    float S = 0.0f;
    float acc[256];
    for (int i = 0; i < hd; i++) acc[i] = 0.0f;

    int n_tiles = (nkv + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int kv_start = tile_idx * Bc;
        int tile_sz  = min(Bc, nkv - kv_start);

        int total = tile_sz * hd;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total; i += tgsz) {
            int ki = kv_start + i / hd;
            int di = i % hd;
            k_tile[i] = k[ki * stride_kv + hk * hd + di];
            v_tile[i] = v[ki * stride_kv + hk * hd + di];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            // All 32 lanes of the simdgroup execute the SAME iteration count so
            // simd_max below never runs across divergent lanes (a divergent
            // simd_max includes stale register values from exited lanes, which
            // corrupts the online-softmax running max for partial tiles).
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float * kj = k_tile + j * hd;
                dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += qhead[d] * kj[d];
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d = 0; d < hd; d++) acc[d] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float * vj = v_tile + j * hd;
                for (int d = 0; d < hd; d++) acc[d] += e * vj[d];
                S += e;
            }
            mx = new_mx;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);
    for (int d = 0; d < hd; d++) acc[d] = simd_sum(acc[d]);

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    if (valid_head) {
        for (int d = tiisg; d < hd; d += 32) {
            ohead[d] = acc[d] * inv;
        }
    }
}

// F16 KV cache variant of kernel_gqa_attn_f32: K/V are read from a half cache
// (2 bytes/elem, matching llama.cpp) and converted to f32 when staged into
// threadgroup tiles. Enabled via MINFER_CACHE_TYPE=f16 (default is f32).
kernel void kernel_gqa_attn_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t  = (int)tgpig.x;
    int hk = (int)tgpig.y;
    if (t >= nt || hk >= nk) return;

    int nkv  = positions[t] + 1;
    int gqa  = nh / nk;
    // Do NOT return early for heads beyond nh: all simdgroups in the
    // threadgroup must reach every threadgroup_barrier below, or the GPU
    // deadlocks when nh % nk != 0. Invalid heads run the loop with a dummy
    // head index but skip the output write.
    int  h0         = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h          = valid_head ? h0 : 0;

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;

    device const float * qhead = q + t * stride_q + h * hd;
    device       float * ohead = o + t * stride_q + h * hd;

    threadgroup float * k_tile = shmem;
    threadgroup float * v_tile = shmem + Bc * hd;

    float mx = -INFINITY;
    float S = 0.0f;
    float acc[256];
    for (int i = 0; i < hd; i++) acc[i] = 0.0f;

    int n_tiles = (nkv + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int kv_start = tile_idx * Bc;
        int tile_sz  = min(Bc, nkv - kv_start);

        int total = tile_sz * hd;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total; i += tgsz) {
            int ki = kv_start + i / hd;
            int di = i % hd;
            k_tile[i] = float(k[ki * stride_kv + hk * hd + di]);
            v_tile[i] = float(v[ki * stride_kv + hk * hd + di]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float * kj = k_tile + j * hd;
                dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += qhead[d] * kj[d];
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d = 0; d < hd; d++) acc[d] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float * vj = v_tile + j * hd;
                for (int d = 0; d < hd; d++) acc[d] += e * vj[d];
                S += e;
            }
            mx = new_mx;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);
    for (int d = 0; d < hd; d++) acc[d] = simd_sum(acc[d]);

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    if (valid_head) {
        for (int d = tiisg; d < hd; d += 32) {
            ohead[d] = acc[d] * inv;
        }
    }
}

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

