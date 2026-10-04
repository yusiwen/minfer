// src/cuda/kernels/attention_decode.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


// ─── GQA Attention (online softmax, 32 threads/head/token) ───
// q/k/v/o layout: [nt][nh][hd]; k/v stored as [nkv][nk][hd]

// 8b: GQA attention over an f16 KV cache — exact structural mirror of
// gqa_attn_f32 (same online softmax, same reductions); the ONLY difference
// is the K/V load mechanics: half4 → float4 conversions, f32 accumulation
// everywhere (Metal pl_gqa_attn_f16 precision class).

template <bool CAUSAL, bool MAP>
__global__ void gqa_attn_f32_f16kv(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ o,
    const int* bound,
    int nh, int nk, int hd,
    float scale, int nt
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= nt || h >= nh) return;

    int row0, nkv;
    attn_extent<CAUSAL, MAP>(bound, t, nt, row0, nkv);
    int gqa = nh / nk;
    int hk = h / gqa;
    int ne_q = nh * hd;
    int stride_kv = nk * hd;

    const float* qhead = q + t * ne_q + h * hd;
    float* ohead = o + t * ne_q + h * hd;

    int tid = threadIdx.x;
    int hd4 = hd / 4;
    const float4* q4 = reinterpret_cast<const float4*>(qhead);

    // Online softmax with persistent accumulators
    const int NE = 2;
    const int C = WARP * NE;

    float mx = -INFINITY;
    float S = 0.0f;
    float4 oc[32];
    #pragma unroll
    for (int i = 0; i < hd4; i++) oc[i] = make_float4(0, 0, 0, 0);

    for (int batch = 0; batch < nkv; batch += C) {
        float s0 = -INFINITY, s1 = -INFINITY;
        int kv0 = batch + tid * NE;
        int kv1 = kv0 + 1;

        if (kv0 < nkv) {
            const __half* krow = k + (size_t)kv_cell<MAP>(bound, t, row0, kv0) * stride_kv + hk * hd;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = h4_to_f4(krow + i * 4);
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s0 = d * scale;
        }
        if (kv1 < nkv) {
            const __half* krow = k + (size_t)kv_cell<MAP>(bound, t, row0, kv1) * stride_kv + hk * hd;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = h4_to_f4(krow + i * 4);
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s1 = d * scale;
        }

        float batch_mx = fmaxf(s0, s1);
        // Warp-level max reduction
        for (int off = 16; off > 0; off >>= 1)
            batch_mx = fmaxf(batch_mx, __shfl_xor_sync(0xFFFFFFFF, batch_mx, off));
        float new_mx = fmaxf(mx, batch_mx);
        float corr = expf(mx - new_mx);

        float e0 = expf(s0 - new_mx);
        float e1 = expf(s1 - new_mx);

        #pragma unroll
        for (int i = 0; i < hd4; i++) oc[i].x *= corr, oc[i].y *= corr, oc[i].z *= corr, oc[i].w *= corr;
        S *= corr;

        if (kv0 < nkv) {
            const __half* vrow = v + (size_t)kv_cell<MAP>(bound, t, row0, kv0) * stride_kv + hk * hd;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = h4_to_f4(vrow + i * 4);
                oc[i].x += e0 * vv.x; oc[i].y += e0 * vv.y;
                oc[i].z += e0 * vv.z; oc[i].w += e0 * vv.w;
            }
        }
        if (kv1 < nkv) {
            const __half* vrow = v + (size_t)kv_cell<MAP>(bound, t, row0, kv1) * stride_kv + hk * hd;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = h4_to_f4(vrow + i * 4);
                oc[i].x += e1 * vv.x; oc[i].y += e1 * vv.y;
                oc[i].z += e1 * vv.z; oc[i].w += e1 * vv.w;
            }
        }
        S += e0 + e1;
        mx = new_mx;
    }

    // Warp-level reduction of S and oc
    S = warp_reduce_sum(S);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        oc[i].x = warp_reduce_sum(oc[i].x);
        oc[i].y = warp_reduce_sum(oc[i].y);
        oc[i].z = warp_reduce_sum(oc[i].z);
        oc[i].w = warp_reduce_sum(oc[i].w);
    }

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float4* o4 = reinterpret_cast<float4*>(ohead);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        o4[i].x = oc[i].x * inv;
        o4[i].y = oc[i].y * inv;
        o4[i].z = oc[i].z * inv;
        o4[i].w = oc[i].w * inv;
    }
}

// ─── 8d: split-K decode attention (flash-decoding) ───────────
// The single-warp-per-(token,head) online-softmax kernel leaves the GPU idle
// at nt == 1 (28 warps total at 7B) — nsys showed 48% of the 7B decode step
// at 2K ctx. Pass 1 scans SPLITS KV chunks in parallel (fixed grid, device-
// side range split so CUDA Graph capture stays valid); pass 2 merges the
// partial (mx, S, oc) results. Partial layout: [SPLITS][nh][pstr] floats,
// pstr = (4+hd+3)&~3 — oc starts at float offset 4 so the float4 writes are
// 16-byte aligned. Scratch is a fixed-size state buffer (nh/hd are graph
// constants), grown during warmup — never inside a capture window.
//
// R-followup rewrite (dim-parallel lanes). nsys on the previous version
// showed: (a) the hd-wide float4 oc[32] accumulator is runtime-indexed
// (hd is a kernel argument) so it lives in LOCAL MEMORY — ~80 MB of local
// traffic per layer, re-read+re-written on every online-softmax rescale;
// (b) each lane walked whole K/V rows with 4-byte loads — 64 scattered
// sector requests per row (12.5% sector utilization), L1-bandwidth bound;
// (c) only 224 single-warp blocks (~4.7 warps/SM). Net: ~150 us/layer at
// 7B @2K (~28 GB/s effective on a 4.3 MB K+V read) — the entire @2K
// decode gap to llama.cpp (28 x 151 us = 4.2 ms of a ~25.8 ms step) — and
// it got MONOTONICALLY worse with more splits (148 -> 172 -> 419 -> 609 us
// for 8/16/32/64) because more resident warps thrash L1 with the local
// oc arrays. Now each lane owns 4 fixed dims: the accumulator is ONE
// float4 in registers (zero spill, hd <= 128 enforced by the dispatch),
// every K/V access is a perfectly-coalesced row instruction, and the row
// dot is a warp reduction. Rows run in batches of 4 with BOTH the K and the
// V rows of each batch staged into registers before the serial online-softmax
// chain starts (D2 — the inline-V form exposed a full memory latency per row;
// see docs/CUDA_OPTIMIZATION.md P7 D2). Idle splits still write an mx=-INF/S=0 partial
// that the combine weights to zero; the [SPLITS][nh][pstr] layout is
// unchanged (combine untouched).

#define ATTN_SPLITS 32

// C4 S2b: `kv_ld4<KV>` and its two specialisations are gone — the single load
// idiom is `kv4<LAYOUT>` (defined with the layout tag above), so both the split
// body and the general kernel read a KV row the same way.

// D3-4 L1: the incumbent D2-staged 1-warp body, refactored into a device
// function so the incumbent kernel and the hybrid dispatch (below) share ONE
// source. The math is byte-identical to the pre-refactor kernel (same rows,
// same order, same per-row ops; only the index setup moved to the callers).
//
// C4 S2b: `LAYOUT` + `row_bytes` replace `typename KV` + `stride_kv`. The f32 and
// f16 instantiations issue the same loads as before (`row_bytes = nk * hd * 4` is
// the byte form of the old `stride_kv`), and the same cells are named in the same
// order — only the address arithmetic moved into `kv_row` / `kv4`.
template <int LAYOUT, bool CAUSAL, bool MAP, bool Q8DP4A = false, bool Q8WIDE = false>
__device__ __forceinline__ void attn_split_1w_body(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ partial,
    const int* __restrict__ bound, int qt,
    int row0, int nkv, int sp, int h,
    int nh, int nk, int hd, float scale, int pstr, size_t row_bytes, int lane_id
) {
    const int SPLITS = ATTN_SPLITS;
    int chunk = (nkv + SPLITS - 1) / SPLITS;
    int lo = sp * chunk;
    int hi = min(nkv, lo + chunk);

    int gqa = nh / nk;
    int hk = h / gqa;
    // Each lane owns 4 consecutive dims (hd % 4 == 0 and hd <= 128 are
    // enforced by the dispatch); lanes with d0 >= hd are idle but keep
    // participating in the warp reductions (zero contribution).
    int d0 = lane_id * 4;
    bool live = d0 < hd;
    const float4 q4 = live ? *reinterpret_cast<const float4*>(q + h * hd + d0)
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // #186: when the Q8_0 K dot accumulates in `int` (`__dp4a`), the lane's four
    // query values are quantized once against their 32-element block's `amax`.
    // The eight lanes that share a block reduce with three `shfl_xor` offsets
    // (4/2/1 stay inside the 8-lane group), so every lane gets the block's
    // scale; K is then read as int8 and never converted to float. `qscale` is
    // the query's block scale, `qk` its packed int8 (zero for idle lanes).
    int qk = 0;
    float qscale = 0.0f;
    if (Q8DP4A) {
        float amax = live ? fmaxf(fmaxf(fabsf(q4.x), fabsf(q4.y)), fmaxf(fabsf(q4.z), fabsf(q4.w)))
                          : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        qscale = amax / 127.0f;
        const float qid = (qscale != 0.0f) ? (1.0f / qscale) : 0.0f;
        if (live) {
            signed char c0 = (signed char)(int)fminf(127.0f, fmaxf(-128.0f, rintf(q4.x * qid)));
            signed char c1 = (signed char)(int)fminf(127.0f, fmaxf(-128.0f, rintf(q4.y * qid)));
            signed char c2 = (signed char)(int)fminf(127.0f, fmaxf(-128.0f, rintf(q4.z * qid)));
            signed char c3 = (signed char)(int)fminf(127.0f, fmaxf(-128.0f, rintf(q4.w * qid)));
            qk = ((int)(unsigned char)c0) | ((int)(unsigned char)c1 << 8) |
                 ((int)(unsigned char)c2 << 16) | ((int)(unsigned char)c3 << 24);
        }
    }

    float mx = -INFINITY, S = 0.0f;
    float4 oc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int base = lo; base < hi; base += 4) {
        int nr = min(4, hi - base); // warp-uniform
        // C8b S4: a map's runs are long, so four consecutive rows almost always
        // sit inside one run. Resolve the batch's first row (and how many rows its
        // run holds) once, then every other row is an add; a run boundary *inside*
        // the batch falls back to the walk, so the rows are still named in linear
        // order. For the contiguous modes `left >= 4` and this is the same `base +
        // j` arithmetic as before (the template folds the fallback away).
        int cell[4];
        {
            int left;
            const int at = kv_cell_left<MAP>(bound, qt, row0, base, left);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                cell[j] = (j < left) ? (at + j) : kv_cell<MAP>(bound, qt, row0, base + j);
        }
        // D2: stage BOTH K and V for the whole 4-row window before the first
        // softmax step. All 8 row loads then issue back-to-back and their
        // latency overlaps the serial chain; the old form relied on the
        // compiler hoisting the inline V loads, which it does not do across
        // the shfl/softmax dependency chain (D1 probe: −11% hot-L2, D2 probe:
        // −42% cold-DRAM vs inline V; bitwise-identical — same rows, same
        // order, same per-row ops, only the load scheduling changes).
        float4 k4[4], v4[4];
        int ki4[4];
        float kd4[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const bool use = live && j < nr;
            if (Q8DP4A) {
                // The packed K word and its block scale; no float conversion.
                if (use) {
                    kv4_q8_0_packed<Q8WIDE>(
                        kv_row(k, cell[j], row_bytes), hk * hd + d0, ki4[j], kd4[j]);
                } else {
                    ki4[j] = 0;
                    kd4[j] = 0.0f;
                }
            } else {
                k4[j] = use
                    ? kv4<LAYOUT, Q8WIDE>(kv_row(k, cell[j], row_bytes), hk * hd + d0)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            v4[j] = use
                ? kv4<LAYOUT, Q8WIDE>(kv_row(v, cell[j], row_bytes), hk * hd + d0)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= nr) break; // warp-uniform: all lanes exit together
            // Full-row dot: this lane's 4-dim partial, then a warp reduction
            // so every lane holds the row's complete dot (uniform softmax).
            float d;
            if (Q8DP4A) {
                // `qk`/`ki4` carry the two int8 quads; `qscale * kd4` is the
                // pair of block scales, applied per lane because block scale is
                // a per-lane (per-32-element-block) value inside the row.
                d = (float)__dp4a(qk, ki4[j], 0) * (qscale * kd4[j]);
            } else {
                d = q4.x * k4[j].x + q4.y * k4[j].y + q4.z * k4[j].z + q4.w * k4[j].w;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                d += __shfl_xor_sync(0xFFFFFFFF, d, off);
            float s = d * scale;
            float nmx = fmaxf(mx, s);
            float corr = expf(mx - nmx);
            float e = expf(s - nmx);
            S = S * corr + e;
            mx = nmx;
            if (live) {
                float4 vv = v4[j];
                oc.x = oc.x * corr + e * vv.x;
                oc.y = oc.y * corr + e * vv.y;
                oc.z = oc.z * corr + e * vv.z;
                oc.w = oc.w * corr + e * vv.w;
            }
        }
    }

    // No cross-lane reduction needed: each lane owns distinct dims.
    float* dst = partial + ((size_t)sp * nh + h) * pstr;
    if (lane_id == 0) {
        dst[0] = mx;
        dst[1] = S;
    }
    if (live) {
        *reinterpret_cast<float4*>(dst + 4 + d0) = oc; // 16B-aligned via pstr
    }
}

template <int LAYOUT, bool CAUSAL, bool MAP, bool Q8DP4A = false, bool Q8WIDE = false>
__global__ void gqa_attn_split_partial(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ partial,
    const int* bound,
    int nh, int nk, int hd, float scale, int pstr, size_t row_bytes,
    int rpw_gate
) {
    // E1b: `bound` is `positions` when CAUSAL (nt == 1 for this decode path) and
    // the `[lo, hi)` span otherwise; row0 is 0 and nkv is `positions[0] + 1` in
    // the causal instantiation, so its arithmetic is the pre-E1 code.
    int row0, nkv;
    attn_extent<CAUSAL, MAP>(bound, 0, 1, row0, nkv);
    // D3-4 L1 dual-kernel dispatch: when rpw_gate > 0 and the 4-warp kernel
    // owns this nkv (rpw >= rpw_gate), exit before touching anything — the
    // hybrid kernel writes the partial. The branch is nkv-uniform across the
    // whole grid (bound[0] is launch-wide), so this stays replay-safe,
    // and for every nkv the incumbent path takes, the arithmetic in the body
    // is unchanged (bitwise; dump-memcmp gated). A packed cache passes
    // `rpw_gate = 0` (the hybrid body is f16-typed), so this branch folds away.
    if (rpw_gate > 0) {
        const int chunk0 = (nkv + ATTN_SPLITS - 1) / ATTN_SPLITS;
        if (((chunk0 + 3) >> 2) >= rpw_gate) return;
    }
    attn_split_1w_body<LAYOUT, CAUSAL, MAP, Q8DP4A, Q8WIDE>(q, k, v, partial, bound, 0, row0, nkv,
                                            blockIdx.x, blockIdx.y, nh, nk, hd, scale, pstr,
                                            row_bytes, threadIdx.x);
}

__global__ void gqa_attn_split_combine(
    const float* __restrict__ partial,
    float* __restrict__ o,
    int nh, int hd, int pstr
) {
    int h = blockIdx.y;
    int i = threadIdx.x; // hd threads
    if (i >= hd) return;
    float gmx = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        gmx = fmaxf(gmx, partial[((size_t)sp * nh + h) * pstr]);
    float S = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        const float* p = partial + ((size_t)sp * nh + h) * pstr;
        float w = expf(p[0] - gmx);
        S += p[1] * w;
        acc += p[4 + i] * w;
    }
    o[h * hd + i] = (S > 0.0f) ? acc / S : 0.0f;
}

// ─── doc 94: batched split attention for the verify shapes (1 < nt <= 16) ────
// The greedy identity (spec-draft output token-for-token equal to sequential
// decode) requires a verify batch's attention logits to be bitwise-equal to
// the nt=1 decode path at every position. The incumbent nt>1 kernel
// (gqa_attn_f32_f16kv) walks the keys with a different reduction schedule
// than the decode split path, which flips argmax on near-ties. These batched
// variants reuse attn_split_1w_body VERBATIM (same per-token nkv =
// positions[t]+1, same 32-split chunking, same combine merge order), so a
// verify batch's partials and merged output are bitwise-equal to running the
// decode kernel at each position. Scope: the 1-warp incumbent body only —
// the rpw>=16 hybrid (nkv >= 1921) is not batched, so the identity guarantee
// covers nkv < 1921 (the batteries run at 512-640). Grid z = nt keeps the
// launch static per captured graph; per-token nkv comes from positions[t]
// device-side, so replay stays capture-safe.
// C4 S2b: still instantiated for F32/F16 only. A packed cache does **not** take
// this path (it routes to `gqa_attn_f32`), because this kernel exists for
// spec-verify's bitwise identity with sequential decode and a Q8_0 session
// refuses a draft rather than claim an unmeasured identity.
template <int LAYOUT, bool CAUSAL, bool MAP>
__global__ void gqa_attn_split_partial_bt(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ partial,
    const int* bound,
    int nh, int nk, int hd, float scale, int pstr, size_t row_bytes, int nt
) {
    const int t = blockIdx.z;
    int row0, nkv;
    attn_extent<CAUSAL, MAP>(bound, t, nt, row0, nkv);
    attn_split_1w_body<LAYOUT, CAUSAL, MAP, false>(
        q + (size_t)t * nh * hd, k, v,
        partial + (size_t)t * ATTN_SPLITS * nh * pstr,
        bound, t, row0, nkv,
        blockIdx.x, blockIdx.y, nh, nk, hd, scale, pstr, row_bytes, threadIdx.x);
}

__global__ void gqa_attn_split_combine_bt(
    const float* __restrict__ partial,
    float* __restrict__ o,
    int nh, int hd, int pstr
) {
    const int t = blockIdx.z;
    const size_t base = (size_t)t * ATTN_SPLITS * nh * pstr;
    int h = blockIdx.y;
    int i = threadIdx.x;
    if (i >= hd) return;
    float gmx = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        gmx = fmaxf(gmx, partial[base + ((size_t)sp * nh + h) * pstr]);
    float S = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        const float* p = partial + base + ((size_t)sp * nh + h) * pstr;
        float w = expf(p[0] - gmx);
        S += p[1] * w;
        acc += p[4 + i] * w;
    }
    o[(size_t)t * nh * hd + h * hd + i] = (S > 0.0f) ? acc / S : 0.0f;
}

template <int LAYOUT>
static void launch_gqa_attn_split_batched_kv(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* bound, int mode,
    int n_head, int n_head_kv, int hd, float scale, int pstr, size_t row_bytes, int nt,
    cudaStream_t stream
) {
    // E1b/C8b S4: `bound` is positions (causal), the [lo, hi) span, or the
    // (cell, len) runs of a `kv_map`; the mode picks the instantiation.
    // C4 S2b: instantiated for F32/F16 only — a packed cache routes to
    // `gqa_attn_f32` instead (see the dispatch cut in `cuda_backend.rs`).
    if (mode == ATTN_WIN_MAP) {
        minfer_launch_prelude("launch:gqa_attn_split_batched_kv__partial_map", "gqa_attn_split_partial_bt<LAYOUT,false,true>");
        gqa_attn_split_partial_bt<LAYOUT, false, true><<<dim3(ATTN_SPLITS, n_head, nt), minfer_launch_block("launch:gqa_attn_split_batched_kv__partial_map", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, row_bytes, nt
        );
        minfer_launch_ok("launch:gqa_attn_split_batched_kv__partial_map", "gqa_attn_split_partial_bt<LAYOUT,false,true>");
    } else if (mode == ATTN_WIN_SPAN) {
        minfer_launch_prelude("launch:gqa_attn_split_batched_kv__partial_span", "gqa_attn_split_partial_bt<LAYOUT,false,false>");
        gqa_attn_split_partial_bt<LAYOUT, false, false><<<dim3(ATTN_SPLITS, n_head, nt), minfer_launch_block("launch:gqa_attn_split_batched_kv__partial_span", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, row_bytes, nt
        );
        minfer_launch_ok("launch:gqa_attn_split_batched_kv__partial_span", "gqa_attn_split_partial_bt<LAYOUT,false,false>");
    } else {
        minfer_launch_prelude("launch:gqa_attn_split_batched_kv__partial_causal", "gqa_attn_split_partial_bt<LAYOUT,true,false>");
        gqa_attn_split_partial_bt<LAYOUT, true, false><<<dim3(ATTN_SPLITS, n_head, nt), minfer_launch_block("launch:gqa_attn_split_batched_kv__partial_causal", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, row_bytes, nt
        );
        minfer_launch_ok("launch:gqa_attn_split_batched_kv__partial_causal", "gqa_attn_split_partial_bt<LAYOUT,true,false>");
    }
    minfer_launch_prelude("launch:gqa_attn_split_batched_kv__combine", "gqa_attn_split_combine_bt");
    gqa_attn_split_combine_bt<<<dim3(1, n_head, nt), minfer_launch_block("launch:gqa_attn_split_batched_kv__combine", hd), 0, stream>>>(
        partial, o, n_head, hd, pstr
    );
    minfer_launch_ok("launch:gqa_attn_split_batched_kv__combine", "gqa_attn_split_combine_bt");
}

extern "C" int launch_gqa_attn_split_batched_f16kv(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* bound, int mode,
    int n_head, int n_head_kv, int hd, float scale, int pstr, int nt,
    cudaStream_t stream
) {
    launch_gqa_attn_split_batched_kv<KV_LAYOUT_F16>(
        q, k, v, o, partial, bound, mode, n_head, n_head_kv, hd, scale, pstr,
        (size_t)n_head_kv * hd * 2, nt, stream);
    return 1;
}

extern "C" int launch_gqa_attn_split_batched_f32kv(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* bound, int mode,
    int n_head, int n_head_kv, int hd, float scale, int pstr, int nt,
    cudaStream_t stream
) {
    launch_gqa_attn_split_batched_kv<KV_LAYOUT_F32>(
        q, k, v, o, partial, bound, mode, n_head, n_head_kv, hd, scale, pstr,
        (size_t)n_head_kv * hd * 4, nt, stream);
    return 1;
}

// ─── D3-4 L1: hybrid rpw dispatch for hd==128 f16-KV decode split attention ───
// D3a (reverted) built the llama fattn-vec-style 4-warp kernel and measured
// the rows-per-warp pathology: rpw = ceil(ceil(nkv/32)/4) = 26 / 13 / 1 at
// 14B @3254 / 7B @1641 / tg128 — the 32-row window idles 59-75% of lanes
// below rpw≈16 (+64% kernel at 7B @1641) while it wins at rpw=26 (73.4 →
// 62.1 µs/layer in-situ nsys, −13.9%). Dispatch (D3-4 L1, dual-kernel
// self-gating — see the f16kv launcher): BOTH kernels launch for hd==128 and
// each re-reads positions[0] per replay; exactly one is live per nkv:
//
//   rpw >= H4W_MIN_RPW (16, i.e. nkv >= 1921): this 4-warp fattn-vec-style
//   body (D3a code, probe-verified ≤1.3e-7 vs CPU; tolerance-gated class).
//   rpw <  16: the incumbent 32-thread D2-staged kernel (bitwise incumbent
//   arithmetic via the shared attn_split_1w_body device function; it
//   early-exits when this kernel owns the nkv).
//
// Running the 1-warp body inside 128-thread blocks was measured +78% kernel
// at 7B @1641 (35.4 vs 19.8 µs) — a 128-thread block caps the SM at 12
// working warps (1536/128) vs the incumbent's 24-32 — hence the dual-kernel
// form. The branch is nkv-uniform and grid/block nkv-independent, so
// CUDA-graph capture/replay is unaffected and per-nkv output stays
// deterministic. D3a numerics record: docs/CUDA_OPTIMIZATION.md §2D D3a +
// /tmp/d3/D3A_FINDINGS.md (kernel-level parity ≤1.3e-7, argmax hard gate,
// greedy 0/10 diverged).

#define H4W_NTHREADS 128 // 4 warps per block
#define H4W_MIN_RPW 16   // 4-warp body only when rows/warp amortize the window

// 8 halves (one uint4) vs two float4 Q slices -> 8-dim partial dot
__device__ __forceinline__ float h4w_dot8(const uint4 ka, const float4 q0, const float4 q1) {
    const __half2* h = reinterpret_cast<const __half2*>(&ka);
    float2 a = __half22float2(h[0]);
    float2 b = __half22float2(h[1]);
    float2 c = __half22float2(h[2]);
    float2 d = __half22float2(h[3]);
    return q0.x * a.x + q0.y * a.y + q0.z * b.x + q0.w * b.y
         + q1.x * c.x + q1.y * c.y + q1.z * d.x + q1.w * d.y;
}

// 8-lane subgroup sum (butterfly inside 8-lane segments)
__device__ __forceinline__ float h4w_subgroup_sum8(float v) {
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off, 8);
    return v;
}

// uint4 of 8 halves -> two float4
__device__ __forceinline__ void h4w_h8_to_f8(const uint4 u, float4& f0, float4& f1) {
    const __half2* h = reinterpret_cast<const __half2*>(&u);
    float2 a = __half22float2(h[0]);
    float2 b = __half22float2(h[1]);
    float2 c = __half22float2(h[2]);
    float2 d = __half22float2(h[3]);
    f0 = make_float4(a.x, a.y, b.x, b.y);
    f1 = make_float4(c.x, c.y, d.x, d.y);
}

template <bool MAP>
__device__ __forceinline__ void attn_split_h4w_body(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ partial,
    const int* __restrict__ bound, int qt,
    int row0, int nkv, int sp, int h,
    int nh, int nk, int hd, float scale, int pstr
) {
    const int gqa = nh / nk;
    const int hk = h / gqa;
    const size_t stride_kv = (size_t)nk * hd;

    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;  // warp id
    const int t = lane & 7;          // 16-dim slice (hd=128 -> 8 slices)
    const int g = lane >> 3;         // row slot within a 4-row pass

    // Same device-side range split as the 1-warp kernel (identical [sp] rows,
    // so the combine sees the same split partitioning).
    const int chunk = (nkv + ATTN_SPLITS - 1) / ATTN_SPLITS;
    const int lo = sp * chunk;
    const int hi = min(nkv, lo + chunk);
    // Balanced contiguous stripes: warp w owns rows [lo + w*rpw, +rpw).
    const int rpw = (chunk + 3) >> 2;
    const int wlo = lo + w * rpw;
    const int wend = min(hi, wlo + rpw);

    // Q slice dims [16t, 16t+16): replicated across the 4 subgroups of the
    // warp (llama keeps the same per-thread Q copy per subgroup).
    const float4* qp = reinterpret_cast<const float4*>(q + (size_t)h * hd + 16 * t);
    const float4 qc0 = qp[0], qc1 = qp[1], qc2 = qp[2], qc3 = qp[3];

    // Finite mx base: exp(-INF - base) == 0 exactly, and exp(base - base) == 1
    // for idle warps — no NaN paths anywhere (the old kernel reached the same
    // states by skipping its loop; here empty stripes keep the base).
    float mx = -1e38f;
    float S = 0.0f;
    float4 oc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 oc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 oc2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 oc3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    __shared__ float probs[H4W_NTHREADS]; // per-warp 32-float prob stage
    float* pw = probs + w * 32;

    for (int b = wlo; b < wend; b += 32) {
        const int wl = min(32, wend - b); // rows in this window (warp-uniform)
        float kq = -INFINITY;             // this lane's row score
        float mx_new = mx;
        const int np = min(8, wl);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            if (p >= np) break;
            const int row = b + g * 8 + p;
            float d = 0.0f;
            if (row < wend) {
                const __half* krow =
                    k + (size_t)kv_cell<MAP>(bound, qt, row0, row) * stride_kv + hk * hd + 16 * t;
                const uint4 ka = *reinterpret_cast<const uint4*>(krow);
                const uint4 kb = *reinterpret_cast<const uint4*>(krow + 8);
                d = h4w_dot8(ka, qc0, qc1) + h4w_dot8(kb, qc2, qc3);
            }
            // The subgroup reduce runs for ALL lanes: a shfl_sync with the full
            // mask deadlocks when subgroups diverge on row validity (found by
            // the standalone probe at nkv=3), so the guard selects the score
            // AFTER the reduction.
            float s = h4w_subgroup_sum8(d) * scale;
            if (row >= wend) s = -INFINITY;
            mx_new = fmaxf(mx_new, s);
            if (t == p) kq = s; // lane (g,t) keeps subgroup g's row (g,p)
        }
        // Cross-subgroup window max (llama: offsets nthreads_KQ..WARP_SIZE).
        #pragma unroll
        for (int off = 8; off < 32; off <<= 1)
            mx_new = fmaxf(mx_new, __shfl_xor_sync(0xFFFFFFFFu, mx_new, off));
        const float wsc = expf(mx - mx_new); // one rescale per 32-row window
        mx = mx_new;
        kq = expf(kq - mx); // invalid rows: kq=-INF -> exact 0
        S = S * wsc + kq;
        oc0.x *= wsc; oc0.y *= wsc; oc0.z *= wsc; oc0.w *= wsc;
        oc1.x *= wsc; oc1.y *= wsc; oc1.z *= wsc; oc1.w *= wsc;
        oc2.x *= wsc; oc2.y *= wsc; oc2.z *= wsc; oc2.w *= wsc;
        oc3.x *= wsc; oc3.y *= wsc; oc3.z *= wsc; oc3.w *= wsc;
        __syncwarp();                    // previous window's prob reads done
        pw[lane] = kq;                   // prob of row b + g*8 + t
        __syncwarp();
        // V accumulation: 8 passes of 4 rows, subgroup g on row b+4p+g; every
        // lane multiplies its 16-dim slice (4 redundant copies per warp,
        // summed in the epilogue). Loads are predicated off for rows past the
        // window end — their staged prob is 0 AND the KV bytes behind them are
        // never written, so both the load and the FMA must be masked.
        const int nvp = min(8, (wl + 3) >> 2);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            if (p >= nvp) break;
            const int row = b + 4 * p + g;
            const float pr = pw[4 * p + g];
            float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 v1 = v0, v2 = v0, v3 = v0;
            if (row < wend) {
                const __half* vrow =
                    v + (size_t)kv_cell<MAP>(bound, qt, row0, row) * stride_kv + hk * hd + 16 * t;
                h4w_h8_to_f8(*reinterpret_cast<const uint4*>(vrow), v0, v1);
                h4w_h8_to_f8(*reinterpret_cast<const uint4*>(vrow + 8), v2, v3);
            }
            oc0.x += pr * v0.x; oc0.y += pr * v0.y; oc0.z += pr * v0.z; oc0.w += pr * v0.w;
            oc1.x += pr * v1.x; oc1.y += pr * v1.y; oc1.z += pr * v1.z; oc1.w += pr * v1.w;
            oc2.x += pr * v2.x; oc2.y += pr * v2.y; oc2.z += pr * v2.z; oc2.w += pr * v2.w;
            oc3.x += pr * v3.x; oc3.y += pr * v3.y; oc3.z += pr * v3.z; oc3.w += pr * v3.w;
        }
    }

    // ── block epilogue: LSE-merge the 4 warp states, write ONE partial ──
    // (w,g) copy of dim d lands at vkq_s[w*512 + g*128 + d]: lane (g,t) stores
    // dims [16t,+16) at offset w*512 + g*128 + 16t, so the final per-dim sum
    // is a stride-128 walk — bank-conflict-free across threads.
    __shared__ __align__(16) float vkq_s[4 * 512];
    __shared__ float mx_sh[4];
    __shared__ float s_sh[4];
    float Sw = warp_reduce_sum(S);
    if (lane == 0) {
        mx_sh[w] = mx;
        s_sh[w] = Sw;
    }
    __syncthreads();
    const float gmax = fmaxf(fmaxf(mx_sh[0], mx_sh[1]), fmaxf(mx_sh[2], mx_sh[3]));
    const float wsc = expf(mx - gmax); // idle warp: exp(-1e38 - gmax) == 0
    oc0.x *= wsc; oc0.y *= wsc; oc0.z *= wsc; oc0.w *= wsc;
    oc1.x *= wsc; oc1.y *= wsc; oc1.z *= wsc; oc1.w *= wsc;
    oc2.x *= wsc; oc2.y *= wsc; oc2.z *= wsc; oc2.w *= wsc;
    oc3.x *= wsc; oc3.y *= wsc; oc3.z *= wsc; oc3.w *= wsc;
    float* vs = vkq_s + (w * 512 + g * 128 + 16 * t);
    reinterpret_cast<float4*>(vs)[0] = oc0;
    reinterpret_cast<float4*>(vs)[1] = oc1;
    reinterpret_cast<float4*>(vs)[2] = oc2;
    reinterpret_cast<float4*>(vs)[3] = oc3;
    __syncthreads();
    float* dst = partial + ((size_t)sp * nh + h) * pstr;
    const int tid = threadIdx.x;
    if (tid < hd) {
        float acc = 0.0f;
        #pragma unroll
        for (int w2 = 0; w2 < 4; w2++)
            #pragma unroll
            for (int g2 = 0; g2 < 4; g2++)
                acc += vkq_s[w2 * 512 + g2 * 128 + tid];
        dst[4 + tid] = acc;
    }
    if (tid == 0) {
        float st = 0.0f;
        #pragma unroll
        for (int w2 = 0; w2 < 4; w2++)
            st += s_sh[w2] * expf(mx_sh[w2] - gmax);
        dst[0] = gmax;
        dst[1] = st;
    }
}

// D3-4 L1: 4-warp fattn-vec-style body in its own kernel; live only when
// rpw >= H4W_MIN_RPW (the incumbent 32-thread kernel owns smaller rpw — see
// the f16kv launcher for the dual-kernel dispatch rationale). All-exit
// otherwise; the branch is nkv-uniform (positions[0] is launch-wide), so
// CUDA-graph capture/replay stays correct and per-nkv output is deterministic.
template <bool CAUSAL, bool MAP>
__global__ void __launch_bounds__(H4W_NTHREADS, 8)
gqa_attn_split_partial_hybrid(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ partial,
    const int* bound,
    int nh, int nk, int hd, float scale, int pstr
) {
    int row0, nkv;
    attn_extent<CAUSAL, MAP>(bound, 0, 1, row0, nkv);
    const int chunk = (nkv + ATTN_SPLITS - 1) / ATTN_SPLITS;
    if (((chunk + 3) >> 2) < H4W_MIN_RPW) return;
    attn_split_h4w_body<MAP>(q, k, v, partial, bound, 0, row0, nkv, blockIdx.x, blockIdx.y,
                             nh, nk, hd, scale, pstr);
}

// C4 S2b: the general kernel is the one a packed cache uses for every nt > 1
// (the verify band and prefill — see the dispatch cuts in the plan). `LAYOUT` +
// `row_bytes` select the load; the softmax/reduction schedule is untouched, so
// F32 keeps its pre-C4 instruction stream byte for byte.
template <int LAYOUT, bool CAUSAL, bool MAP>
__global__ void gqa_attn_f32(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ o,
    const int* bound,
    int nh, int nk, int hd,
    float scale, size_t row_bytes, int nt
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= nt || h >= nh) return;

    int row0, nkv;
    attn_extent<CAUSAL, MAP>(bound, t, nt, row0, nkv);
    int gqa = nh / nk;
    int hk = h / gqa;
    int ne_q = nh * hd;

    const float* qhead = q + t * ne_q + h * hd;
    float* ohead = o + t * ne_q + h * hd;

    int tid = threadIdx.x;
    int hd4 = hd / 4;
    const float4* q4 = reinterpret_cast<const float4*>(qhead);

    // Online softmax with persistent accumulators
    const int NE = 2;
    const int C = WARP * NE;

    float mx = -INFINITY;
    float S = 0.0f;
    float4 oc[32];
    #pragma unroll
    for (int i = 0; i < hd4; i++) oc[i] = make_float4(0, 0, 0, 0);

    for (int batch = 0; batch < nkv; batch += C) {
        float s0 = -INFINITY, s1 = -INFINITY;
        int kv0 = batch + tid * NE;
        int kv1 = kv0 + 1;

        if (kv0 < nkv) {
            // C4 S2b: `kv_row` + `kv4<LAYOUT>` — the same 4-element groups the
            // old `float4*` walk named, addressed in bytes so a packed cell works.
            const char* krow = kv_row(k, kv_cell<MAP>(bound, t, row0, kv0), row_bytes);
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = kv4<LAYOUT>(krow, hk * hd + i * 4);
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s0 = d * scale;
        }
        if (kv1 < nkv) {
            const char* krow = kv_row(k, kv_cell<MAP>(bound, t, row0, kv1), row_bytes);
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = kv4<LAYOUT>(krow, hk * hd + i * 4);
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s1 = d * scale;
        }

        float batch_mx = fmaxf(s0, s1);
        // Warp-level max reduction
        for (int off = 16; off > 0; off >>= 1)
            batch_mx = fmaxf(batch_mx, __shfl_xor_sync(0xFFFFFFFF, batch_mx, off));
        float new_mx = fmaxf(mx, batch_mx);
        float corr = expf(mx - new_mx);

        float e0 = expf(s0 - new_mx);
        float e1 = expf(s1 - new_mx);

        #pragma unroll
        for (int i = 0; i < hd4; i++) oc[i].x *= corr, oc[i].y *= corr, oc[i].z *= corr, oc[i].w *= corr;
        S *= corr;

        if (kv0 < nkv) {
            const char* vrow = kv_row(v, kv_cell<MAP>(bound, t, row0, kv0), row_bytes);
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = kv4<LAYOUT>(vrow, hk * hd + i * 4);
                oc[i].x += e0 * vv.x; oc[i].y += e0 * vv.y;
                oc[i].z += e0 * vv.z; oc[i].w += e0 * vv.w;
            }
        }
        if (kv1 < nkv) {
            const char* vrow = kv_row(v, kv_cell<MAP>(bound, t, row0, kv1), row_bytes);
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = kv4<LAYOUT>(vrow, hk * hd + i * 4);
                oc[i].x += e1 * vv.x; oc[i].y += e1 * vv.y;
                oc[i].z += e1 * vv.z; oc[i].w += e1 * vv.w;
            }
        }
        S += e0 + e1;
        mx = new_mx;
    }

    // Warp-level reduction of S and oc
    S = warp_reduce_sum(S);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        oc[i].x = warp_reduce_sum(oc[i].x);
        oc[i].y = warp_reduce_sum(oc[i].y);
        oc[i].z = warp_reduce_sum(oc[i].z);
        oc[i].w = warp_reduce_sum(oc[i].w);
    }

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float4* o4 = reinterpret_cast<float4*>(ohead);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        o4[i].x = oc[i].x * inv;
        o4[i].y = oc[i].y * inv;
        o4[i].z = oc[i].z * inv;
        o4[i].w = oc[i].w * inv;
    }
}

// #141/#162: the #147 checked-launch helpers are defined further down (with the
// MMQ launchers); their C++-linkage forward declarations live at the top of this
// file, before the first launch site.

// ====================================================================
// extern "C" launch wrappers (called from Rust via FFI)
// ====================================================================

extern "C" {

void launch_gqa_attn_f32_f16kv(
    const float* q, const void* k, const void* v, float* o,
    const int* bound, int mode,
    int n_head, int n_head_kv, int hd,
    float scale, int nt, cudaStream_t stream
) {
    int block_sz = 32; // one warp per (token, head)
    dim3 block(block_sz, 1, 1);
    dim3 grid(nt, n_head, 1);
    // C8b S4: the window mode picks the instantiation, so the causal path keeps
    // its pre-E1 instruction stream (E1b's rule, now three modes).
    switch (mode) {
        case ATTN_WIN_MAP:
            minfer_launch_prelude("launch:gqa_attn_f32_f16kv__map", "gqa_attn_f32_f16kv<false,true>");
            gqa_attn_f32_f16kv<false, true><<<grid, minfer_launch_block("launch:gqa_attn_f32_f16kv__map", block), 0, stream>>>(
                q, (__half*)k, (__half*)v, o, bound,
                n_head, n_head_kv, hd, scale, nt);
            minfer_launch_ok("launch:gqa_attn_f32_f16kv__map", "gqa_attn_f32_f16kv<false,true>");
            break;
        case ATTN_WIN_SPAN:
            minfer_launch_prelude("launch:gqa_attn_f32_f16kv__span", "gqa_attn_f32_f16kv<false,false>");
            gqa_attn_f32_f16kv<false, false><<<grid, minfer_launch_block("launch:gqa_attn_f32_f16kv__span", block), 0, stream>>>(
                q, (__half*)k, (__half*)v, o, bound,
                n_head, n_head_kv, hd, scale, nt);
            minfer_launch_ok("launch:gqa_attn_f32_f16kv__span", "gqa_attn_f32_f16kv<false,false>");
            break;
        default:
            minfer_launch_prelude("launch:gqa_attn_f32_f16kv__causal", "gqa_attn_f32_f16kv<true,false>");
            gqa_attn_f32_f16kv<true, false><<<grid, minfer_launch_block("launch:gqa_attn_f32_f16kv__causal", block), 0, stream>>>(
                q, (__half*)k, (__half*)v, o, bound,
                n_head, n_head_kv, hd, scale, nt);
            minfer_launch_ok("launch:gqa_attn_f32_f16kv__causal", "gqa_attn_f32_f16kv<true,false>");
            break;
    }
}

void launch_gqa_attn_split_f16kv(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* bound, int mode,
    int n_head, int n_head_kv, int hd,
    float scale, int pstr, cudaStream_t stream
) {
    // D3-4 L1: hd == 128 (Qwen2.5/Qwen3 decode shapes) dual-kernel
    // self-gating dispatch on rpw = ceil(ceil(nkv/ATTN_SPLITS)/4):
    // rpw >= H4W_MIN_RPW (nkv >= 1921) -> the 4-warp fattn-vec-style kernel;
    // rpw < 16 -> the incumbent 32-thread D2-staged kernel. Both launches are
    // static (grid/block nkv-independent), so CUDA-graph capture/replay is
    // unaffected; each kernel re-reads positions[0] on every replay and
    // exactly one is live for the current nkv (the rpw branch is
    // nkv-uniform). The dud launch costs ~1-2 us/layer but keeps the
    // small-rpw shapes on the incumbent geometry — running the 1-warp body
    // inside 128-thread blocks caps the SM at 12 working warps (1536/128)
    // and measured +78% kernel at 7B @1641 (35.4 vs 19.8 us, nsys). Other
    // head dims (incl. the hd=8 parity fixtures) keep the single incumbent
    // launch (rpw_gate=0).
    // C8b S4: `mode` selects the window instantiation (causal / span / map). The
    // hybrid's rpw gate reads `nkv` off the window, which this path resolves for
    // its single query, so the branch stays launch-wide for all three modes.
    if (hd == 128) {
        if (mode == ATTN_WIN_MAP) {
            minfer_launch_prelude("launch:gqa_attn_split_f16kv__map_h4w", "gqa_attn_split_partial<KV_LAYOUT_F16,false,true>");
            gqa_attn_split_partial<KV_LAYOUT_F16, false, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__map_h4w", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 2, H4W_MIN_RPW);
            minfer_launch_ok("launch:gqa_attn_split_f16kv__map_h4w", "gqa_attn_split_partial<KV_LAYOUT_F16,false,true>");
            minfer_launch_prelude("launch:gqa_attn_split_f16kv__hybrid_map", "gqa_attn_split_partial_hybrid<false,true>");
            gqa_attn_split_partial_hybrid<false, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__hybrid_map", H4W_NTHREADS), 0, stream>>>(
                q, (const __half*)k, (const __half*)v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr);
            minfer_launch_ok("launch:gqa_attn_split_f16kv__hybrid_map", "gqa_attn_split_partial_hybrid<false,true>");
        } else if (mode == ATTN_WIN_SPAN) {
            minfer_launch_prelude("launch:gqa_attn_split_f16kv__span_h4w", "gqa_attn_split_partial<KV_LAYOUT_F16,false,false>");
            gqa_attn_split_partial<KV_LAYOUT_F16, false, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__span_h4w", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 2, H4W_MIN_RPW);
            minfer_launch_ok("launch:gqa_attn_split_f16kv__span_h4w", "gqa_attn_split_partial<KV_LAYOUT_F16,false,false>");
            minfer_launch_prelude("launch:gqa_attn_split_f16kv__hybrid_span", "gqa_attn_split_partial_hybrid<false,false>");
            gqa_attn_split_partial_hybrid<false, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__hybrid_span", H4W_NTHREADS), 0, stream>>>(
                q, (const __half*)k, (const __half*)v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr);
            minfer_launch_ok("launch:gqa_attn_split_f16kv__hybrid_span", "gqa_attn_split_partial_hybrid<false,false>");
        } else {
            minfer_launch_prelude("launch:gqa_attn_split_f16kv__causal_h4w", "gqa_attn_split_partial<KV_LAYOUT_F16,true,false>");
            gqa_attn_split_partial<KV_LAYOUT_F16, true, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__causal_h4w", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 2, H4W_MIN_RPW);
            minfer_launch_ok("launch:gqa_attn_split_f16kv__causal_h4w", "gqa_attn_split_partial<KV_LAYOUT_F16,true,false>");
            minfer_launch_prelude("launch:gqa_attn_split_f16kv__hybrid_causal", "gqa_attn_split_partial_hybrid<true,false>");
            gqa_attn_split_partial_hybrid<true, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__hybrid_causal", H4W_NTHREADS), 0, stream>>>(
                q, (const __half*)k, (const __half*)v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr);
            minfer_launch_ok("launch:gqa_attn_split_f16kv__hybrid_causal", "gqa_attn_split_partial_hybrid<true,false>");
        }
    } else if (mode == ATTN_WIN_MAP) {
        minfer_launch_prelude("launch:gqa_attn_split_f16kv__map", "gqa_attn_split_partial<KV_LAYOUT_F16,false,true>");
        gqa_attn_split_partial<KV_LAYOUT_F16, false, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__map", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 2, 0);
        minfer_launch_ok("launch:gqa_attn_split_f16kv__map", "gqa_attn_split_partial<KV_LAYOUT_F16,false,true>");
    } else if (mode == ATTN_WIN_SPAN) {
        minfer_launch_prelude("launch:gqa_attn_split_f16kv__span", "gqa_attn_split_partial<KV_LAYOUT_F16,false,false>");
        gqa_attn_split_partial<KV_LAYOUT_F16, false, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__span", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 2, 0);
        minfer_launch_ok("launch:gqa_attn_split_f16kv__span", "gqa_attn_split_partial<KV_LAYOUT_F16,false,false>");
    } else {
        minfer_launch_prelude("launch:gqa_attn_split_f16kv__causal", "gqa_attn_split_partial<KV_LAYOUT_F16,true,false>");
        gqa_attn_split_partial<KV_LAYOUT_F16, true, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__causal", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 2, 0);
        minfer_launch_ok("launch:gqa_attn_split_f16kv__causal", "gqa_attn_split_partial<KV_LAYOUT_F16,true,false>");
    }
    minfer_launch_prelude("launch:gqa_attn_split_f16kv__combine", "gqa_attn_split_combine");
    gqa_attn_split_combine<<<dim3(1, n_head), minfer_launch_block("launch:gqa_attn_split_f16kv__combine", hd), 0, stream>>>(
        partial, o, n_head, hd, pstr
    );
    minfer_launch_ok("launch:gqa_attn_split_f16kv__combine", "gqa_attn_split_combine");
}

void launch_gqa_attn_split_f32kv(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* bound, int mode,
    int n_head, int n_head_kv, int hd,
    float scale, int pstr, cudaStream_t stream
) {
    if (mode == ATTN_WIN_MAP) {
        minfer_launch_prelude("launch:gqa_attn_split_f32kv__map", "gqa_attn_split_partial<KV_LAYOUT_F32,false,true>");
        gqa_attn_split_partial<KV_LAYOUT_F32, false, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f32kv__map", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 4, 0);
        minfer_launch_ok("launch:gqa_attn_split_f32kv__map", "gqa_attn_split_partial<KV_LAYOUT_F32,false,true>");
    } else if (mode == ATTN_WIN_SPAN) {
        minfer_launch_prelude("launch:gqa_attn_split_f32kv__span", "gqa_attn_split_partial<KV_LAYOUT_F32,false,false>");
        gqa_attn_split_partial<KV_LAYOUT_F32, false, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f32kv__span", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 4, 0);
        minfer_launch_ok("launch:gqa_attn_split_f32kv__span", "gqa_attn_split_partial<KV_LAYOUT_F32,false,false>");
    } else {
        minfer_launch_prelude("launch:gqa_attn_split_f32kv__causal", "gqa_attn_split_partial<KV_LAYOUT_F32,true,false>");
        gqa_attn_split_partial<KV_LAYOUT_F32, true, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_f32kv__causal", 32), 0, stream>>>(
            q, k, v, partial, bound,
            n_head, n_head_kv, hd, scale, pstr, (size_t)n_head_kv * hd * 4, 0);
        minfer_launch_ok("launch:gqa_attn_split_f32kv__causal", "gqa_attn_split_partial<KV_LAYOUT_F32,true,false>");
    }
    minfer_launch_prelude("launch:gqa_attn_split_f32kv__combine", "gqa_attn_split_combine");
    gqa_attn_split_combine<<<dim3(1, n_head), minfer_launch_block("launch:gqa_attn_split_f32kv__combine", hd), 0, stream>>>(
        partial, o, n_head, hd, pstr
    );
    minfer_launch_ok("launch:gqa_attn_split_f32kv__combine", "gqa_attn_split_combine");
}

// C4 S2b: the packed decode path. One 1-warp split-K launch per window mode, with
// `rpw_gate = 0` — the hybrid 4-warp body (`attn_split_h4w_body`) takes
// `const __half*` and is not converted, so it is not offered for a packed cell.
// `row_bytes` is `KvFormat::Q8_0.row_bytes(nkt)` (whole f32 words).
//
// #186: the K dot is accumulated in `int` (`__dp4a`, see
// `kv4_q8_0_packed`/`attn_split_1w_body<...,Q8DP4A>`) instead of the
// convert-based `kv4<Q8_0>` load. `dp4a` is the caller's answer, resolved once per
// process on the Rust side (`cuda::q8_kv_dp4a_enabled`, `MINFER_NO_DP4A_Q8_KV=1`
// for the incumbent arm) and passed as a value so a captured decode graph cannot
// see it change.
//
// #202: `wide` is the second, independent arm — the K and V four-quant groups are
// fetched with two 16-bit loads instead of four byte loads (`q8_0_load4_wide`),
// which halves the L1 request count for the same bytes. It is only meaningful on
// the `dp4a` arm (the packed accessor); the convert arm keeps its byte loads for
// the V side and only differs in arithmetic. `cuda::q8_kv_wide_enabled`
// (`MINFER_NO_Q8_KV_WIDE=1` for the incumbent byte-load control) is resolved once
// per process for the same captured-graph reason as `dp4a`.
void launch_gqa_attn_split_q8_0(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* bound, int mode,
    int n_head, int n_head_kv, int hd,
    float scale, int pstr, size_t row_bytes, int dp4a, int wide, cudaStream_t stream
) {
    if (mode == ATTN_WIN_MAP) {
        if (dp4a && wide) {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__map_wide", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,true,true,true>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, false, true, true, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__map_wide", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__map_wide", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,true,true,true>");
        } else if (dp4a) {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__map_dp4a", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,true,true>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, false, true, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__map_dp4a", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__map_dp4a", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,true,true>");
        } else {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__map", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,true>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, false, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__map", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__map", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,true>");
        }
    } else if (mode == ATTN_WIN_SPAN) {
        if (dp4a && wide) {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__span_wide", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,false,true,true>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, false, false, true, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__span_wide", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__span_wide", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,false,true,true>");
        } else if (dp4a) {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__span_dp4a", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,false,true>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, false, false, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__span_dp4a", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__span_dp4a", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,false,true>");
        } else {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__span", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,false>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, false, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__span", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__span", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,false,false>");
        }
    } else {
        if (dp4a && wide) {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__causal_wide", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,true,false,true,true>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, true, false, true, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__causal_wide", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__causal_wide", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,true,false,true,true>");
        } else if (dp4a) {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__causal_dp4a", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,true,false,true>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, true, false, true><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__causal_dp4a", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__causal_dp4a", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,true,false,true>");
        } else {
            minfer_launch_prelude("launch:gqa_attn_split_q8_0__causal", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,true,false>");
            gqa_attn_split_partial<KV_LAYOUT_Q8_0, true, false><<<dim3(ATTN_SPLITS, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__causal", 32), 0, stream>>>(
                q, k, v, partial, bound,
                n_head, n_head_kv, hd, scale, pstr, row_bytes, 0);
            minfer_launch_ok("launch:gqa_attn_split_q8_0__causal", "gqa_attn_split_partial<KV_LAYOUT_Q8_0,true,false>");
        }
    }
    minfer_launch_prelude("launch:gqa_attn_split_q8_0__combine", "gqa_attn_split_combine");
    gqa_attn_split_combine<<<dim3(1, n_head), minfer_launch_block("launch:gqa_attn_split_q8_0__combine", hd), 0, stream>>>(
        partial, o, n_head, hd, pstr
    );
    minfer_launch_ok("launch:gqa_attn_split_q8_0__combine", "gqa_attn_split_combine");
}

// C4 S2b: the general nt > 1 kernel, now layout-tagged. `row_bytes` is the
// packed cell's byte width under Q8_0 and `nk * hd * {4,2}` under f32/f16.
void launch_gqa_attn_f32(
    const float* q, const void* k, const void* v, float* o,
    const int* bound, int mode, int layout, int nh, int nk, int hd,
    float scale, size_t row_bytes, int nt, cudaStream_t stream
) {
    dim3 block(WARP, 1, 1); // 32 threads per block (1 warp)
    dim3 grid(nt, nh, 1);
    // #162: one source site (`gqa_attn_f32`), nine instantiations. The kernel
    // name is stringized from the macro arguments, so the report names the exact
    // `gqa_attn_f32<layout,causal,map>` the dispatch picked. Required: the sticky
    // makes `execute_node` return `Err` on a failed launch.
    #define GQA_F32_LAYOUT_CASE(L, C, M) \
        do { \
            const char* const kn = "gqa_attn_f32<" #L "," #C "," #M ">"; \
            minfer_launch_prelude("launch:gqa_attn_f32", kn); \
            gqa_attn_f32<L, C, M><<<grid, minfer_launch_block("launch:gqa_attn_f32", block), 0, stream>>>( \
                q, k, v, o, bound, nh, nk, hd, scale, row_bytes, nt); \
            minfer_launch_ok("launch:gqa_attn_f32", kn); \
        } while (0)
    switch (layout) {
        case KV_LAYOUT_F32:
            if (mode == ATTN_WIN_MAP) GQA_F32_LAYOUT_CASE(KV_LAYOUT_F32, false, true);
            else if (mode == ATTN_WIN_SPAN) GQA_F32_LAYOUT_CASE(KV_LAYOUT_F32, false, false);
            else GQA_F32_LAYOUT_CASE(KV_LAYOUT_F32, true, false);
            break;
        case KV_LAYOUT_F16:
            if (mode == ATTN_WIN_MAP) GQA_F32_LAYOUT_CASE(KV_LAYOUT_F16, false, true);
            else if (mode == ATTN_WIN_SPAN) GQA_F32_LAYOUT_CASE(KV_LAYOUT_F16, false, false);
            else GQA_F32_LAYOUT_CASE(KV_LAYOUT_F16, true, false);
            break;
        default:
            if (mode == ATTN_WIN_MAP) GQA_F32_LAYOUT_CASE(KV_LAYOUT_Q8_0, false, true);
            else if (mode == ATTN_WIN_SPAN) GQA_F32_LAYOUT_CASE(KV_LAYOUT_Q8_0, false, false);
            else GQA_F32_LAYOUT_CASE(KV_LAYOUT_Q8_0, true, false);
            break;
    }
    #undef GQA_F32_LAYOUT_CASE
}
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// This file owns the template instantiations, so the address-taking must
// happen here (a cross-TU template reference is nvcc #20280-D and can fail
// to link).
extern "C" void minfer_prewarm_attention_decode_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, (gqa_attn_f32_f16kv<true, false>));
    MINFER_PREWARM_ONE(a, (gqa_attn_f32_f16kv<false, false>));
    MINFER_PREWARM_ONE(a, (gqa_attn_f32_f16kv<false, true>));
}
