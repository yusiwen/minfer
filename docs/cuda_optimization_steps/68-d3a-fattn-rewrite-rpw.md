# 68 · D3a — the 4-warp fattn-vec-style split-attention rewrite (REVERTED) + tolerance-gate calibration

> **Result**: the rewrite itself was vetoed — 7B @1641 split kernel 21.1 →
> 34.7 µs (**+64%**, the rows-per-warp pathology), 14B @3254 73.4 → 68.4 µs
> (−6.9%) but the wall clock only −0.95% (inside the ±2% noise band) — both
> bars failed, and the whole thing was reverted per the r44 precedent. **The
> session's durable output is the tolerance-gate calibration**: a
> kernel-level 1e-7 numerical difference drifts end-to-end logits by O(0.4)
> on 28–48-layer models; the "end-to-end max\|Δlogits\| ≤ 1e-3" gate proposed
> at D3-1 is unsatisfiable for **any** rewrite that changes accumulation
> order (retired); the usable gate set = kernel-vs-CPU ≤1e-4 (real
> outlier-magnitude data) + argmax HARD gate (top-2 margin > 0.1) + greedy
> −n 256 ×5 seeds + temp-0.8 sampling contrast + suite + interleaved A/B.
> **Commit**: no repo change (the experiment patch is kept at
> `/tmp/d3/d3a_kernel_patch.diff`, full measurement record
> `/tmp/d3/D3A_FINDINGS.md`; the docs-only commit is `a05af20`). **Date**:
> 2026-09-07.

## 1. Background — where things stood

The D-series teardown campaign reached its third stage. D1 attributed 100%
of 7B @1641 decode's KV-scaling wall to the single split-attention kernel
`gqa_attn_split_partial` (34.1 µs/launch, 76.5% long_scoreboard) and proved
the ATTN_SPLITS sweep a dead end (changing the split count reorders float
summation, no longer bitwise). D2 landed the first win with "explicit K+V
register staging": all 8 loads of the 4-row window issued early, kernel
34.1 → 19.4 µs, wall clock +2.0% (a bitwise-class change, cheap gates).

Then D3-1 finished dismantling the 14B @3254 decode wall, where the
attention term is bigger: split kernel **73.4 µs/layer, still 50% above the
48.9 µs byte floor**. D3b's bitwise MMVQ levers (1a/1b/1c/2) cleared the
GEMM side's low fruit, but the attention residual was the next big target.
While dismantling, D3-1 had also drafted a tolerance gate for future
non-bitwise levers: "end-to-end max\|Δlogits\| ≤ 1e-3". Nobody had validated
that draft — this doc is its first combat test, and it lost (see §5.2).

Lever 2's direction was written in the campaign plan long before: **change
decode split attention from the 1-warp serial structure to llama.cpp
`fattn-vec`'s 4-warp structure** (source-verified @ca3d5a3e1). The chain of
assumptions at the time: the incumbent kernel is "serially dense" — every
lane works on every row, 32 threads carrying all 128 dims; llama's 4-warp
structure hands rows to 4 warps in 32-row windows, each window doing only
one online-softmax rescale, with shorter latency chains and more warps per
SM (D3a measured `__launch_bounds__(128,8)` → 32 warps/SM vs the incumbent's
24). It looked like winning on both ends. D3a's task was to port this
structure onto minfer's f16-KV decode path and measure what it is worth.

Where things stall without this step: the 14B @3254 attention residual (73.4 − 48.9 ≈ 24.5 µs/layer × 48 layers ≈ 1.2 ms/step) was the largest known single item, and D2's "issue-point hoisting" trick was already spent; the remaining levers are all structural.

## 2. Principle — the GPU mechanism: the rows-per-warp pathology

### 2.1 The geometry of the two generations

**The incumbent 1-warp kernel** (D2-staged): grid
`dim3(ATTN_SPLITS=32, n_head)`, each block 32 threads (1 warp) covering one
split's `chunk = ceil(nkv/32)` rows × `hd` dims. Each lane permanently owns
4 consecutive dims (`d0 = lane_id * 4`) and, for **every row**, does one
"4-dim dot + full-warp butterfly reduce + online-softmax update". However
many rows, every lane is busy — that is "serially dense": no idle slots, at
the price of a per-row reduce dependency chain (D2 relieved the load side
with early issue, but the softmax chain itself remains row-by-row).

**D3a's 4-warp structure (fattn-vec style)**: 128 threads (4 warps) per
block; K/V stream straight from global (never into smem); Q lives in
registers (16 dims per thread = 4 × float4); the warp's four 8-lane
subgroups each claim a row (butterfly sums within a subgroup); each warp
handles a 32-row window and does **exactly one** online-softmax rescale per
window; probabilities stage into a 128-float smem row, and a final 4-warp
LSE-merge writes one partial.

### 2.2 The pathology: rpw decides slot utilization

The key parameter is **rows-per-warp**:

```
chunk = ceil(nkv / 32)          # rows assigned to each split
rpw   = ceil(chunk / 4)         # split evenly over 4 warps → rows per warp
```

The 32-row window's lane-slot mapping is fixed: 8 passes × 4 subgroups,
subgroup g claiming window rows `8g..8g+7`. That is, **a window has exactly
32 lane slots**, and when the actual live row count is rpw the slot
utilization = rpw/32:

| Shape | nkv | chunk | rpw | Slot utilization | Consequence |
|---|---|---|---|---|---|
| 14B @3254 | 3254 | 102 | **26** | 81% | no pathology, 4-warp wins |
| 7B @1641 | 1641 | 52 | **13** | 41% (59% idle) | subgroups 2–3 idle as whole groups |
| 7B tg128 | 128 | 4 | **1** | 8% | 3 of 4 warps exit outright; the live warp uses 8/32 lanes |

And the per-block fixed cost is amortized over rpw rows: a 4-warp block's Q
preload is 128 threads × 64 B = **8 KB** (the incumbent 1-warp only 32 ×
16 B = 512 B), plus the epilogue's syncthreads/staging and partial
write-out. At rpw=13 these costs spread over 13 rows; at rpw=1 over 1 row —
total collapse. Conversely the incumbent kernel has no such problem:
0.58–0.83 waves fully resident, every lane busy on every row. **The
fattn-vec structure has net gain only when rpw = ceil(ceil(nkv/32)/4) ≳
16** — this doc's first conclusion, which later became D3-4 L1's dispatch
threshold (nkv ≥ 1921) directly.

### 2.3 Why a kernel win may not be a wall-clock win

In one 14B @3254 decode step attention is only 7.4%. So a kernel −6.9% folds into roughly +0.5% on the wall clock — below the +1.5% session bar and inside the ±2% A/B noise band. "A win on a kernel that is not the wall" does not reach the wall (r45's old conclusion) — the veto's second leg.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Grid untouched**: `dim3(ATTN_SPLITS, n_head)`, independent of nkv →
  CUDA-graph capture/replay unaffected. This is D1's iron rule: the grid
  must not depend on `positions`.
- **Occupancy first**: `__launch_bounds__(128, 8)` compiles to REG 64 /
  STACK 0 / SMEM 8960 B = **32 warps/SM** (incumbent 24). smem is only the
  probs row (128 floats) + the epilogue's `vkq_s` (4×512 floats) +
  `mx_sh`/`s_sh`.
- **Q all in registers**: each thread holds its own 16-dim slice (4 ×
  float4, redundantly replicated across the 4 subgroups — llama's same
  trick); K/V stream straight from global, saving smem K/V staging
  bandwidth.
- **Dispatch keyed only on hd==128**: f16-KV + hd==128 (the decode shape of
  Qwen2.5/Qwen3) takes the new kernel; hd≠128 keeps the incumbent (parity
  fixtures undisturbed).
- **LSE-merge epilogue**: the 4 warps each hold their own (mx, S, oc)
  state, merged through smem into one partial — the partial layout stays
  exactly the incumbent kernel's, so the combine kernel needs no change.

### 3.2 Key code

D3a's patch lives on in the current tree inside the "hybrid kernel body"
(when D3-4 L1 landed, it installed the D3a body verbatim as
`gqa_attn_split_partial_hybrid`), so the excerpts below come from the
**current tree** `src/cuda_kernels.cu`; D3a's original form of the day
(`gqa_attn_split_partial_h4w` as the sole hd==128 path) is in
`/tmp/d3/d3a_kernel_patch.diff`.

**before — the incumbent 1-warp body (D2-staged, row-level softmax chain)**:

```cuda
// All of the 4-row window's K+V issued early (D2), then row-by-row online-softmax updates
#pragma unroll
for (int j = 0; j < 4; j++) {
    if (j >= nr) break; // warp-uniform: all lanes exit together
    // full-row dot: this lane's 4 dims + full-warp butterfly, once per row
    float d = q4.x*k4[j].x + q4.y*k4[j].y + q4.z*k4[j].z + q4.w*k4[j].w;
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
        oc.y = oc.y * corr + e * vv.y;   // one rescale (corr) per row
    }
}
```

**after — the h4w body: warp stripes + 32-row windows (excerpted from the
current tree's `attn_split_h4w_body`)**:

```cuda
// Balanced contiguous stripes: warp w owns rows [lo + w*rpw, +rpw).
const int rpw = (chunk + 3) >> 2;
const int wlo = lo + w * rpw;
const int wend = min(hi, wlo + rpw);

__shared__ float probs[H4W_NTHREADS]; // per-warp 32-float prob stage
float* pw = probs + w * 32;

for (int b = wlo; b < wend; b += 32) {
    const int wl = min(32, wend - b); // rows in this window (warp-uniform)
    ...
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        if (p >= np) break;
        const int row = b + g * 8 + p;      // subgroup g claims 8 rows
        float d = 0.0f;
        if (row < wend) {
            const __half* krow = k + row * stride_kv + hk * hd + 16 * t;
            const uint4 ka = *reinterpret_cast<const uint4*>(krow);
            const uint4 kb = *reinterpret_cast<const uint4*>(krow + 8);
            d = h4w_dot8(ka, qc0, qc1) + h4w_dot8(kb, qc2, qc3);
        }
        float s = h4w_subgroup_sum8(d) * scale;  // 8-lane butterfly
        if (row >= wend) s = -INFINITY;
        mx_new = fmaxf(mx_new, s);
        if (t == p) kq = s;                 // lane (g,t) keeps row (g,p)'s score
    }
    // window max across subgroups (llama: offsets nthreads_KQ..WARP_SIZE)
    for (int off = 8; off < 32; off <<= 1)
        mx_new = fmaxf(mx_new, __shfl_xor_sync(0xFFFFFFFFu, mx_new, off));
    const float wsc = expf(mx - mx_new);    // one rescale per window
    mx = mx_new;
    kq = expf(kq - mx);
    S = S * wsc + kq;
    ...
}
```

**LSE-merge epilogue** (4 warp states → one partial):

```cuda
__shared__ __align__(16) float vkq_s[4 * 512];
__shared__ float mx_sh[4];
__shared__ float s_sh[4];
float Sw = warp_reduce_sum(S);
if (lane == 0) { mx_sh[w] = mx; s_sh[w] = Sw; }
__syncthreads();
const float gmax = fmaxf(fmaxf(mx_sh[0], mx_sh[1]), fmaxf(mx_sh[2], mx_sh[3]));
const float wsc = expf(mx - gmax); // idle warp: exp(-1e38 - gmax) == 0
// ...the four float4 accumulators each multiply wsc, then write to vkq_s by (w, g) slot...
// finally sum across the 4 warps per dim (stride-128 walk, bank-conflict-free), write dst
```

### 3.3 Pitfalls

**The shfl_sync deadlock (this doc's single biggest lesson, now in
Appendix-B)**. The first version put the subgroup reduction inside the
row-validity guard:

```cuda
if (row < wend) {                       // wrong form: lanes diverge
    d = ...;
    s = h4w_subgroup_sum8(d) * scale;   // __shfl_xor_sync(0xFFFFFFFF, ...)
}
```

`__shfl_xor_sync(0xFFFFFFFF, v, off, 8)`'s mask names all 32 lanes, but the
8-lane subgroup's butterfly only needs its own subgroup present — when
different subgroups disagree on `row < wend` (inevitable when rpw does not
divide the window; e.g. rpw=13's second window has only 5 rows), some lanes
never reach the shuffle point, the names in the mask can never be redeemed,
and the warp spins at the hardware level. **Symptom**: the standalone probe
hung at nkv=3; in `cargo test` it presented as a GPU-spin hang. **The fix**
(the comment at current-tree lines 3050–3055 is its tombstone): compute
contributions conditionally (invalid rows default 0.0f), run the reduction
unconditionally for all lanes, then mask invalid rows to `-INFINITY`
afterwards:

```cuda
float s = h4w_subgroup_sum8(d) * scale;  // unconditional: all lanes present
if (row >= wend) s = -INFINITY;          // mask afterwards
```

**Finite initial values**. An idle warp / empty stripe's `mx` must not be
`-INFINITY` (`exp(-INF - (-INF))` produces NaN); the h4w body uses `-1e38f`
as the base: `exp(-1e38 - gmax) == 0` holds exactly, so an empty stripe's
warp contributes exactly zero.

**V-side loads and FMAs must be masked together**. Rows at the window tail
have probability exactly 0, but the KV bytes behind them were never written
— "probability 0 so the product is 0" does not hold; what gets read is a NaN
from garbage bytes. When the load predicate turns off, the FMA must turn off
too (current-tree lines 3074–3078 comment).

## 4. Verification

This kernel was numerically **all-green** — every gate passed, which is why
it later became the tolerance-gate calibration vehicle. What each gate
defends:

- **kernel-level parity probe (defends against math errors)**: h4w vs the
  CPU reference swept over all nkv 3..4096 (including 14B's exact in-situ
  shapes and every chunk boundary) ≤ **1.3e-7**; same inputs vs the
  incumbent kernel ≤ 8.9e-8; on **real outlier-magnitude data** (residual
  \|q\|~50, V outliers ±127) new-vs-CPU 6.5e-5 vs old-vs-CPU 3.8e-5 — the
  same error class.
- **in-situ per-layer TRACE (defends against layout/indexing errors)**:
  NO_CUDA_GRAPH + MINFER_TRACE capturing per-node attn outputs (14B):
  prefill attention bitwise; decode per-layer deltas L0 2.4e-7, **L1–L11
  bitwise 0.0** (the f16-KV store is a noise gate — sub-ULP reordering
  noise is quantized away at each layer's K/V write-back), L12+ 1e-3..5e-1
  (noise crosses the f16 rounding boundary and is amplified through
  outlier-dim cancellation).
- **argmax HARD gate (defends against sampling flips)**: every dump step's
  argmax byte-identical, with top-2 margins 4.95 / 10.97, far from the 0.1
  threshold.
- **greedy × seeds + sampling contrast (defends against distribution
  drift)**: greedy −n 256 × 5 seeds × both models, 0/10 diverged by a single
  token; the temp 0.8 seed-7 sampling contrast fully identical.
- **suite (defends against the regression surface)**: 169/0/3 + the FA trio
  + split-decode parity (the parity test drives the new kernel with the
  hd=128/n_ctx 4200 shape — reverted along with the code).
- **interleaved A/B (defends against window drift)**: same-window 3×
  interleaved medians.

**Two calibration findings about the dump gate** (later written into D3b's
aliasing note):

1. decode node dumps (`node{2,3,5,8,11}_decode`) read **aliased pool slots**
   (node11 = kv_load's dump is only 20 KB, not the 16.7 MB KV region) —
   node-level diff noise comes from slot aliasing, not values.
2. The KV-region diff between `-n 1` and `-n 2` dumps is a **pre-existing
   per-step row rewrite**: reproducible pre-vs-pre on the incumbent binary
   too (layer-10 K's rows 17–20 change between decode steps 1→2). Rule:
   gate on logits + the final step's KV, and use **the same -n on both
   sides**.

## 5. Results

### 5.1 The veto numbers and mechanism

nsys (NO_CUDA_GRAPH, bench −n 8 averaged):

| Shape | incumbent | h4w | Δ |
|---|---|---|---|
| 14B @3254 split kernel | 73.4 µs | 68.4 µs | **−6.9%** (= 71.5% of the 48.9 µs byte floor, was 67%) |
| 7B @1641 split kernel | 21.1 µs | 34.7 µs | **+64%** |

Wall clock (same-window interleaved 3× medians, tok/s):

| Shape | pre | post | Δ | bar | Verdict |
|---|---|---|---|---|---|
| 14B @3254 | 21.04 | 20.84 | −0.95% | ≥ +1.5% | fail (a −6.9% kernel is worth ~+0.5% wall: attention is 7.4% of the step) |
| 7B @1641 | 47.53 | 45.91 | **−3.4%** | ≥ 47.9 | fail (tight clusters, same direction as the kernel's +64%) |
| 7B tg128 | — | — | noise-level | — | the rpw=1 pathology crushes the theoretical gain |

**Veto mechanism (why reverted, when a retry is worthwhile)**: the rpw
pathology is structural — the 32-row window's lane-slot mapping is frozen
(8 passes × 4 subgroups); whenever rpw < ~16, idle slots and the diluted
per-block fixed costs eat everything. This is not tunable; the geometry does
not fit the shape. The revert followed the r44 precedent (a parity-green,
wall-clock sub-bar kernel change is not kept). **Retry conditions** are on
record: multi-warp decode attention must first solve the rpw pathology —
either a subgroup-dense row mapping (letting live rows fill all 32 slots) or
a runtime switch back to the 1-warp body by rpw. The latter is the form
D3-4 L1 adopted (doc 69).

### 5.2 The durable output: tolerance-gate calibration

The kernel was numerically all-green (probe 1.3e-7), but the "end-to-end
max\|Δlogits\| ≤ 1e-3" gate proposed at D3-1 failed:

```
end-to-end logits max|Δ| = 0.376 (14B, 48 layers) vs 0.389 (7B, 28 layers)
```

The two depths are the **same magnitude** ⇒ this is a depth-independent
error class, not a bug. Mechanism: a kernel-level 1e-7 accumulation-order
difference is quantized and amplified layer by layer through f16 rounding
boundaries (L12+ already at 5e-1), reaching O(0.4) at the logits.
**Conclusion: on 28–48-layer models, "end-to-end logits ≤ 1e-3" is
unsatisfiable for any rewrite that changes accumulation order** (the
r50/r57 lesson generalized to decode).

The usable tolerance-gate package (all demonstrated green on this session's
experimental kernel):

1. kernel-level parity vs CPU ≤ ~1e-4, and it must be validated on **real
   outlier-magnitude** data;
2. the **argmax HARD gate**: every dump step's argmax byte-identical with
   top-2 margin > 0.1 (measured margins 4.95 / 10.97);
3. greedy −n 256 × 5 seeds × both models with 0/10 single-token divergence +
   the temp 0.8 sampling contrast;
4. suite + FA trio + split-decode parity (new shapes must cover the new
   dispatch path);
5. the same-window interleaved A/B bar.

This gate set was adopted directly by D3-4 L1's h4w arm (doc 69), D3-6, and
D3-7 2b.

## 6. Lessons

1. **Shared shape parameters decide a structural rewrite's life or death**:
   the 4-warp fattn-vec structure's efficiency is a staircase function of
   rpw = ceil(ceil(nkv/32)/4), winning only at ≥16 — before porting any
   "rows-to-warps" kernel, compute the target shape's rows-per-warp; do not
   write code first.
2. **`__shfl_sync`'s mask is a whole-warp promise**: with 0xFFFFFFFF in the
   mask, all 32 lanes must reach the point; any lane-divergent guard must
   move out of the reduction, into "compute contributions conditionally,
   reduce unconditionally, mask afterwards".
3. **A kernel-level win is not a wall-clock win**: a −6.9% kernel that is
   7.4% of a step folds to ~+0.5%; before crossing the bar, compute "what
   fraction of the whole step is this kernel".
4. **An end-to-end logits tolerance is a property of the error class, not of
   bugs**: any accumulation-order change necessarily amplifies to O(0.4)
   across dozens of f16 store layers — tolerance gates must be pinned to
   discrete invariants like argmax/greedy/sampling, not to absolute
   thresholds on continuous quantities.

---
← 67 · [Index](./README.md) · 69 →
