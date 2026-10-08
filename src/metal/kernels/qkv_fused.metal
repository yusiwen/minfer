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

// Packed Q8_0 KV cache variant of kernel_gqa_attn_f32 (C4 S2b Metal twin, issue
// #310): K/V are packed Q8_0 cells (`kvformat::KvFormat::Q8_0`) and are
// dequantized block-by-block as they are staged into the threadgroup tiles. The
// tiling, online softmax and reduction order are identical to the f32 kernel;
// only the load changes, so a query reads exactly the f32 kernel's value up to
// the Q8_0 block rounding. `row_bytes` is one cell's word-padded byte width.
kernel void kernel_gqa_attn_q8_0(
    device const float * q        [[buffer(0)]],
    device const uchar * k        [[buffer(1)]],
    device const uchar * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    uint   & row_bytes [[buffer(10)]],
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
            int e  = hk * hd + di;
            k_tile[i] = dequant_q8_0_kv_elem(k, ki, row_bytes, e);
            v_tile[i] = dequant_q8_0_kv_elem(v, ki, row_bytes, e);
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

