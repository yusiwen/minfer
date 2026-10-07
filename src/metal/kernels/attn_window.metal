// ─── Windowed GQA attention: the E1 `attn_span` read path (issue #44, G5a) ───
//
// The causal attention kernels (`kernel_gqa_attn_f32/_f16`, the flash / split /
// prefill families) each derive their window from `positions`: token `t` may see
// the cells `[0, positions[t] + 1)`. That is only correct while a run starts at
// cell 0 (`position == cell`); a batched forward carrying several sequences, or
// a single sequence whose run starts at a non-zero cell, needs the window the
// explicit `attn_span` input names (E1): one `[lo, hi)` pair per query, `lo` at
// `window[t]` and `hi` at `window[nt + t]`.
//
// These two kernels are the read side of that layout. They mirror
// `kernel_gqa_attn_f32/_f16` **exactly** — same Bc=32 tiling, same online
// softmax, same reduction order — and change only where the K/V window starts
// and ends: the base pointer is advanced by `lo` rows and the length becomes
// `hi - lo`. A single sequence at cell 0 whose span is causal (`[0, pos+1)`)
// therefore computes the same bytes as the causal kernel; the arithmetic is not
// re-derived. Dispatch is described in `src/metal/ops.rs::gqa_attn_window_f32`.
//
// `window` carries the `attn_span` I32 input stored as `f32::from_bits` (compute
// graph rule 4), which is why it is bound as `constant int *` exactly like the
// causal kernels' `positions`. A `kv_map`-sized window (a list of `(cell, len)`
// runs) is a *different* layout and is read by the sibling kernels below
// (issue #362) — reading it as `(lo, hi)` would attend to the wrong rows
// silently (risk 1 of the #44 plan).
kernel void kernel_gqa_attn_window_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * window   [[buffer(4)]],
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

    // E1: the query's window is `[lo, hi)` cells. `lo`/`hi` are uniform across
    // the threadgroup (they depend only on `t` and `hk`), so the early return on
    // an empty window cannot strand a `threadgroup_barrier`.
    int lo  = window[t];
    int hi  = window[nt + t];
    int nkv = hi - lo;
    if (nkv <= 0) return;

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
    // Advance into the window's first cell; every index below is window-relative,
    // exactly as the causal kernel's cell-0 window.
    device const float * kwin = k + lo * stride_kv + hk * hd;
    device const float * vwin = v + lo * stride_kv + hk * hd;

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
            k_tile[i] = kwin[ki * stride_kv + di];
            v_tile[i] = vwin[ki * stride_kv + di];
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

// F16 KV cache variant of kernel_gqa_attn_window_f32: K/V are read from a half
// cache (2 bytes/elem, matching llama.cpp) and converted to f32 when staged into
// threadgroup tiles. The E1 window resolution is identical; only the load
// changes. Dispatch selects this when the engine's KV format is f16.
kernel void kernel_gqa_attn_window_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * window   [[buffer(4)]],
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

    int lo  = window[t];
    int hi  = window[nt + t];
    int nkv = hi - lo;
    if (nkv <= 0) return;

    int gqa  = nh / nk;
    int  h0         = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h          = valid_head ? h0 : 0;

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;

    device const float * qhead = q + t * stride_q + h * hd;
    device       float * ohead = o + t * stride_q + h * hd;
    // The f16 region addresses one cell every `stride_kv` half elements, so this
    // is the same `lo`-row advance in half units.
    device const half * kwin = k + lo * stride_kv + hk * hd;
    device const half * vwin = v + lo * stride_kv + hk * hd;

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
            k_tile[i] = float(kwin[ki * stride_kv + di]);
            v_tile[i] = float(vwin[ki * stride_kv + di]);
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

// ─── C8b S4: the set-valued `kv_map` read path (issue #362) ─────────────────
//
// A sequence that reads part of its prefix in place has a window that is not one
// contiguous range: its shared prefix `[0, r)` lives in the donor's cells and its
// private run `[r, p]` in its own. The `kv_map` input names it as a zero-padded
// list of `KV_MAP_MAX_SPANS` `(cell, len)` runs per query. These two kernels are
// the read side of that layout: same Bc = 32 tiling, same online softmax, same
// reduction order as `kernel_gqa_attn_window_f32/_f16`; only the K/V row that a
// flat window index names changes — a run walk instead of `lo + ki`. The row
// resolution mirrors CUDA's `attn_map_nkv` / `kv_cell`
// (`src/cuda/kernels/common.cuh:391-452`).
//
// They are siblings on purpose: the one-range window kernels' instruction stream
// is a measured contract (#315) and is not touched. Like the span kernels, the
// per-query window length is uniform across the threadgroup (it depends only on
// `t`), so the early return cannot strand a `threadgroup_barrier`.
constant int KV_MAP_MAX_SPANS = 4; // mirrors kvcache::KV_MAP_MAX_SPANS

// The number of rows query `t`'s window names — the sum of its runs' lengths.
static inline int kv_map_nkv(constant const int * map, int t) {
    const int base = t * KV_MAP_MAX_SPANS * 2;
    int n = 0;
    for (int r = 0; r < KV_MAP_MAX_SPANS; r++) n += map[base + r * 2 + 1];
    return n;
}

// The arena row (cell) the flat window index `ki` names. The runs are walked in
// order; `ki` is below their total, so the walk always returns.
static inline int kv_map_cell(constant const int * map, int t, int ki) {
    const int base = t * KV_MAP_MAX_SPANS * 2;
    int off = ki;
    for (int r = 0; r < KV_MAP_MAX_SPANS; r++) {
        const int len = map[base + r * 2 + 1];
        if (off < len) return map[base + r * 2] + off;
        off -= len;
    }
    return map[base]; // unreachable: the runs' lengths sum to nkv
}

kernel void kernel_gqa_attn_map_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * map      [[buffer(4)]],
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

    // The query's window length is the runs' total; uniform across the
    // threadgroup, so the early return cannot strand a barrier.
    int nkv = kv_map_nkv(map, t);
    if (nkv <= 0) return;

    int gqa  = nh / nk;
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
            int cell = kv_map_cell(map, t, ki);
            k_tile[i] = k[cell * stride_kv + hk * hd + di];
            v_tile[i] = v[cell * stride_kv + hk * hd + di];
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

// F16 KV cache variant of kernel_gqa_attn_map_f32: K/V are read from a half
// cache and converted to f32 when staged into threadgroup tiles. The run
// resolution is identical; only the load changes.
kernel void kernel_gqa_attn_map_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * map      [[buffer(4)]],
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

    int nkv = kv_map_nkv(map, t);
    if (nkv <= 0) return;

    int gqa  = nh / nk;
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
            int cell = kv_map_cell(map, t, ki);
            k_tile[i] = float(k[cell * stride_kv + hk * hd + di]);
            v_tile[i] = float(v[cell * stride_kv + hk * hd + di]);
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
