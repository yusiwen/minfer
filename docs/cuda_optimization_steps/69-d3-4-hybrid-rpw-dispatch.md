# 69 · D3-4 — L1 hybrid rpw dispatch (LANDED) + L2 window-prefetch pipelining (REVERTED) + long-prompt dump calibration

> **Result**: **L1 landed** (`22336b2`) — the dual-kernel self-gating rpw
> dispatch: 14B @3254 split kernel 72.1 → 62.11 µs (**−13.9%**) plus a 1.5 µs
> dud launch, wall clock 21.20 → **21.33** (**+0.61%**, SEP), all three 7B
> guards held, and the 1-warp arm is bitwise. **L2 reverted** — window-level
> K/V prefetch inside the h4w body: 62.1 → 66.46 µs (**+7%**); occupancy
> halved (32→16 warps/SM) + wave growth (3.33→4.44) outweighed the shortened
> chain; at 79% of the 48.9 µs byte floor the kernel is bytes+tail-bound, not
> chain-bound. Also archived as calibrations: two pre-existing behaviors —
> long-prompt dump non-determinism and the CLI n_ctx headroom.
> **Commit**: `22336b2` (neither L2 nor the in-kernel fallback form was
> committed). **Date**: 2026-09-07.

## 1. Background — where things stood

D3a (doc 68) ported llama's fattn-vec-style 4-warp split attention; the
kernel was numerically all-green but hit the rows-per-warp pathology:
rpw = ceil(ceil(nkv/32)/4) is 26 at 14B @3254 (winning 6.9%), 13 at 7B @1641
(**+64%**), and 1 at tg128 (collapse). After the full revert, one clear
rescue path remained, already written into D3a's record: **"the 4-warp path
serves only dense chunks; at rpw < ~16 fall back to the D2-staged 1-warp
body within the same kernel — the branch is nkv-uniform and replay-safe."**

Meanwhile the window's other bottlenecks did not sit still. Baseline anchors (same-window interleaved 3× medians, pre side): 14B tg128 23.81→22.96-class (drifting), @3254 21.90→21.20; 7B tg128 50.90→50.28, @1641 49.32→48.78. The sglang co-tenant was resident throughout and the window drifts ±2% — so **every judgment must be a same-window interleaved A/B**; cross-session absolutes are not comparable (r59b's old rule).

D3-1's teardown also left a "distance account": at 14B @3254 the attention residual (73.4 − 48.9 µs byte floor) is the largest single item, and D3-4's brief made L2 (window-level K/V prefetch) the main lever: 62.1 µs (after L1 landed) is 79% of the floor, the K phase issues 8 serial load→reduce per 32-row window, the V phase 7 load→FMA — if D2's issue-point early-issue trick moved to window granularity, it should in theory harvest a large part of the remaining 21%.

So D3-4 did three things in one session: L1 landed D3a's legacy in dispatch form; L2 tested the brief's prefetch hypothesis; and two metrological traps (long-prompt dump, CLI n_ctx) were calibrated and archived along the way.

## 2. Principle — the GPU mechanism: the dual-kernel self-gate and the economics of a dud launch

### 2.1 Why "launch both kernels" is replay-safe

The dispatch condition is `rpw = ceil(ceil(nkv/32)/4) ≥ 16` (i.e.
nkv ≥ 1921). Both roads:

- **grid shape is nkv-independent**: both kernels use the static
  `dim3(ATTN_SPLITS=32, n_head)` grid (h4w has `H4W_NTHREADS=128` threads,
  the incumbent 32) — the captured launch configuration never changes, so
  replay is legal.
- **liveness is recomputed per execution on device**: every kernel re-reads
  `positions[0]` for nkv at every execution, computes rpw itself, and then
  **exactly one** kernel works while the other early-exits wholesale. The
  branch condition is uniform across the whole grid (`positions[0]` is a
  launch-level scalar) — no inter-warp divergence, per-nkv output
  deterministic.
- **the cost**: one extra dud launch per layer, measured ~1.3–1.5 µs. In
  exchange, each geometry serves its own rpw range without polluting the
  other.

### 2.2 Why the first cut (in-kernel fallback) was vetoed: block geometry is destiny

The first cut put both bodies in one 128-thread kernel: rpw ≥ 16 takes the
h4w body, otherwise the 1-warp body (only the first 32 threads work).
Bitwise all-green (7B @1845-token dump identical per file), but 7B @1641
measured **35.4 vs 19.8 µs (+78%, nsys)**. The mechanism is not math but
geometry:

```
GB10 caps every SM at 1536 threads.
A 128-thread block → at most 1536/128 = 12 blocks per SM,
of which only 1 warp per block works on the 1-warp arm → 12 working warps/SM.
A 32-thread block → 1536/32 = 48 block slots; the incumbent form runs 24–32 working warps/SM.
```

The 1-warp body is **serially dense**; its throughput model is "working
warps per SM" — a 128-thread block cuts its resident warp count by more than
half. **The body did not change; the configuration killed it**. This is
direct evidence for "geometry, not math": any shared-kernel form that
"stuffs the 1-warp body into a bigger block" need not be tried again.

### 2.3 L2's hypothesis and counter-hypothesis: when prefetch is not free

L2's hypothesis chain: 62.1 µs = 79% × the 48.9 µs floor ⇒ 21% is
harvestable chain overhead; D2 proved issue-point early issue is bitwise and
free (when registers are plentiful). The counter-hypothesis (known only
afterwards): **this kernel already presses against ~131 KB/SM of in-flight
loads (≈ 30× the latency-BW product)** — its in-flight bytes were already
oversaturated; the bottleneck is the byte count itself (the composition of
the L2 5× re-reads) + the wave tail, not each chain's depth. To fit the
prefetch ring's buffers into registers, `__launch_bounds__`'s minBlocks must
drop from 8 to 4 (register budget 64→128), and occupancy goes 32→16
warps/SM, waves 3.33→4.44 — **the occupancy tax for shortening the chain is
1:2**, and a shortened chain cannot save a bytes-bound kernel.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Dual kernels, shared device function**: the incumbent 1-warp body was
  extracted into `attn_split_1w_body` (math and indexing byte-identical,
  only the index setup moved to the caller), and the incumbent kernel and
  the hybrid dispatch share that one source — the 1-warp arm's bitwise
  property is guaranteed structurally by "the same body".
- **The h4w body reuses D3a verbatim**: the probe already verified ≤1.3e-7
  vs CPU and the tolerance class is calibrated; no reinvention.
- **The gate lives on device, not host**: the host cannot know each decode
  step's nkv in advance (that would need a sync); the kernel reads
  `positions[0]` and gates itself — zero host-side logic, zero sync.
- **Dual launch only when hd==128**; other head dims (including the hd=8
  parity fixture) keep the single incumbent launch (`rpw_gate=0`), leaving
  the parity surface undisturbed.

### 3.2 Key code

**The incumbent kernel's rpw_gate early exit** (current tree `src/cuda_kernels.cu`):

```cuda
template <typename KV>
__global__ void gqa_attn_split_partial(..., int rpw_gate) {
    // D3-4 L1 dual-kernel dispatch: when rpw_gate > 0 and the 4-warp kernel
    // owns this nkv (rpw >= rpw_gate), exit before touching anything — the
    // hybrid kernel writes the partial. The branch is nkv-uniform across the
    // whole grid (positions[0] is launch-wide), so this stays replay-safe,
    // and for every nkv the incumbent path takes, the arithmetic in the body
    // is unchanged (bitwise; dump-memcmp gated).
    if (rpw_gate > 0) {
        const int nkv0 = positions[0] + 1;
        const int chunk0 = (nkv0 + ATTN_SPLITS - 1) / ATTN_SPLITS;
        if (((chunk0 + 3) >> 2) >= rpw_gate) return;   // h4w owns this nkv
    }
    attn_split_1w_body<KV>(q, k, v, partial, positions[0] + 1, ...);
}
```

**The hybrid kernel's dual gate**:

```cuda
#define H4W_NTHREADS 128 // 4 warps per block
#define H4W_MIN_RPW 16   // 4-warp body only when rows/warp amortize the window

__global__ void __launch_bounds__(H4W_NTHREADS, 8)
gqa_attn_split_partial_hybrid(const float* q, const __half* k, const __half* v,
                              float* partial, const int* positions, ...)
{
    const int nkv = positions[0] + 1;
    const int chunk = (nkv + ATTN_SPLITS - 1) / ATTN_SPLITS;
    if (((chunk + 3) >> 2) < H4W_MIN_RPW) return;      // rpw < 16 → exit
    attn_split_h4w_body(q, k, v, partial, nkv, blockIdx.x, blockIdx.y, ...);
}
```

**The launcher's dual launch** (`launch_gqa_attn_split_f16kv`):

```cuda
if (hd == 128) {
    gqa_attn_split_partial<__half><<<dim3(ATTN_SPLITS, n_head), 32, 0, stream>>>(
        q, (const __half*)k, (const __half*)v, partial, positions,
        n_head, n_head_kv, hd, scale, pstr, H4W_MIN_RPW);      // 1-warp arm
    gqa_attn_split_partial_hybrid<<<dim3(ATTN_SPLITS, n_head), H4W_NTHREADS, 0, stream>>>(
        q, (const __half*)k, (const __half*)v, partial, positions,
        n_head, n_head_kv, hd, scale, pstr);                    // h4w arm
} else {
    gqa_attn_split_partial<__half><<<dim3(ATTN_SPLITS, n_head), 32, 0, stream>>>(
        q, ..., /*rpw_gate=*/0);                                // single launch
}
gqa_attn_split_combine<<<dim3(1, n_head), hd, 0, stream>>>(partial, o, ...);
```

The two kernels write the same `partial` region, but per nkv exactly one is
alive — combine is unaware the dispatch exists. The dud arm's whole-grid
early exit is a launch-level scalar branch under both 32/128-thread blocks,
no divergence cost, only the ~1.3–1.5 µs launch itself.

### 3.3 Pitfalls

- **The in-kernel fallback's +78%** (§2.2): bitwise-green ≠ landable. The
  1-warp body is sensitive to block size; any shared-block form must compute
  working-warps/SM before speaking.
- **The `attn_split_1w_body` extraction must preserve the math byte for
  byte**: the extraction left "each split's row-range computation" in the
  caller and the body takes only `(nkv, sp, h, ...)` — same rows, same
  order, same row-level ops, only the index setup moved. The dump-memcmp
  gate confirmed the 1-warp arm's output is bit-identical to pre (7B @1845:
  71/71 files identical, see §4).
- **A co-tenant outlier on the pre side**: 7B @1641's pre series contained
  one 44.40 co-tenant outlier rep — the median held, and the record
  explicitly annotates the outlier's attribution rather than silently
  dropping it.

## 4. Verification

- **The 1-warp arm's bitwise dump gate (defends against "the shared
  extraction changed the math")**: 7B @1845-token prompt, logits
  prefill+decode, all KV, decode nodes — 71/71 files byte-identical; the 3
  `node{3,5,8}_prefill` diffs are D3b's already-calibrated slot-aliasing
  trio, reproducible pre-vs-pre.
- **The h4w arm's tolerance class (D3a's gate set, doc 68 §5.2)**: 7B @2800
  (rpw=25, h4w territory) max\|Δlogits\| 0.309 (the calibrated 0.39 class);
  argmax identical every dump step, margin 0.716 (HARD gate > 0.1); upper KV
  layers show the f16 noise pattern (kv0–7 bitwise, kv8–27 drifting on the
  decode side).
- **greedy × seeds + sampling contrast (defends against distribution
  drift)**: greedy −n 256 × 5 seeds × both models, h4w-regime prompts:
  exactly one divergence each at the regime entry point (1/256 = 0.4% < 2%),
  coherent continuation afterwards (no repeated degradation); the temp 0.8
  seed-7 contrast identical.
- **suite (defends against the regression surface)**: 169/0/3, with the
  parity test extended by one hd=128/n_ctx-4200 shape whose pos0 sweep
  crosses exactly the rpw 15/16 dispatch boundary (nkv 1920/1921) — **parity
  coverage on both sides of the dispatch boundary**.
- **guards (defend against regressions at other shapes)**: 7B ≥ 49.0 /
  ≥ 47.9 and 14B tg128 ≥ 22.7 all held.
- **nsys kernel level (defends against "wall-clock noise hiding the kernel
  truth")**: NO_CUDA_GRAPH, bench −n 8 averaged over the last 384.

## 5. Results

### 5.1 L1 (LANDED, `22336b2`)

Kernel level (nsys, NO_CUDA_GRAPH):

| Shape | pre | post | Δ |
|---|---|---|---|
| 14B @3254 split | 72.1 µs | 62.11 µs | **−13.9%** (plus a 1.5 µs dud launch) |
| 7B @1641 split | — | 20.7 µs + 1.3 µs dud | the incumbent path's min equals pre → the body undisturbed |

(In the D3a era the same shape was 73.4 → 68.4 — the hybrid's h4w arm is
faster than D3a's full form, because small-rpw shapes no longer pollute the
same bin.)

Wall clock (same-window interleaved 3× medians, tok/s):

| Shape | pre | post | Δ | Verdict |
|---|---|---|---|---|
| 14B @3254 | 21.20 | **21.33** | **+0.61%** | SEP: min-new 21.25 > max-base 21.23 |
| 14B tg128 | 22.96 | 22.94 | −0.09% | guard ≥ 22.7 held |
| 7B tg128 | 50.28 | 50.20 | −0.16% | guard ≥ 49.0 held |
| 7B @1641 | 48.78 | 48.68 | −0.20% | guard ≥ 47.9 held (pre contained a 44.40 co-tenant outlier) |

The distance account (post-L1, 14B @3254): minfer 21.33 t/s = 46.88 ms/step
vs llama 24.32 = 41.12 ms → still 5.76 ms short (−12.3%). Known lever list:
attention residual (62.1−48.9)×48 = 0.63 ms; D3-1's three MMVQ stragglers
(attn_v-q6K 0.28 + ffn_down-q6K 0.49 + output-head 0.44) = 1.21 ms; D3c
elementwise fusion ≈ 1.0 ms (projected +2.1% wall, not implemented this
session). Together ≈ 2.84 ms = 49% of the gap → landing all of them reaches
only ~22.6 t/s (0.93×); the remaining ~2.9 ms is the matmul account (D3-1's
wall-effective 194.9 vs llama 207.6 GB/s) + launch-structure slack. 7B:
tg128 1.016× (ahead), @1641 0.985× — the 7B decode campaign is de facto
closed.

### 5.2 L2 (REVERTED, uncommitted)

patch_l2.py: K-phase software pipeline (+2 uint4 per thread) + a 4-deep
V-bulk ring (+8 uint4), `__launch_bounds__` minBlocks 8→4 traded for
register budget (64→128 cap).

| Metric | pre (h4w) | post | Δ |
|---|---|---|---|
| 14B @3254 h4w kernel | 62.1 µs | 66.46 µs | **+7%** |
| occupancy | 32 warps/SM | 16 warps/SM | halved |
| waves (1280 blocks) | 3.33 | 4.44 | +33% |

**Veto mechanism**: at the saturation point of 131 KB/SM in-flight loads
(≈30× the latency-BW product), shortening a single chain's depth produces no
time — the time lives in the byte count (the L2 5× re-read composition) and
the wave tail. D2's "issue-point moves are free" holds only when registers
are plentiful; at a 64-reg budget any pipeline funding is a 1:2 occupancy
trade. The bitwise gates were never reached (the revert came before them).
**Retry conditions**: prefetch has room for discussion only after the L2
re-read byte count itself is cut (a bytes-side lever, e.g. GQA q-head
batching — which D3-6 later showed is also not the residual) or a
register-free staging is found.

### 5.3 Metrology calibrations (RECORDED, no code)

- **long-prompt dump non-determinism**: at prompts ≥ 2.8K tokens the
  `MINFER_GRAPH_DUMP` PREFILL-phase files (all `kv*_prefill`,
  `logits_prefill`, prefill nodes) are non-deterministic even pre-vs-pre
  (wholesale, garbage-magnitude — an aliasing problem in the dump read
  path); decode-phase dumps stay deterministic. Rule: for long prompts the
  dump gate must be **anchored pre-vs-pre on the exact shape, and gate only
  decode-phase files**.
- **CLI n_ctx headroom**: prompts above the default n_ctx 4096 leave zero
  generation headroom (the `position N exceeds n_ctx N` panic; the record
  cites graph.rs:367, the current-tree line has drifted to 418); bench
  unaffected. Until n_ctx sizing is fixed, use shapes with
  `prompt + n ≤ 4096` for long-prompt greedy gates.

## 6. Lessons

1. **One dud launch per layer (~1.4 µs) is fair rent for "geometry
   self-gating"**: a device-side nkv-uniform branch + static grid preserves
   replay, far cheaper than a host-side synchronized decision.
2. **A bitwise-green kernel can still be +78%**: the shared-kernel fallback
   changed the block geometry (working warps/SM), not the math — a body is
   sensitive to "how big a block it lives in".
3. **Occupancy and chain depth trade 1:2** (under a 64-reg budget): adding a
   pipeline ring to a kernel whose in-flight bytes are already saturated
   buys a chain shortening that cannot beat halved warp residency.
4. **A dump gate's determinism must be calibrated per phase and per shape**:
   at long prompts even pre-vs-pre prefill-phase dumps are non-deterministic
   — gate only on the phase proven deterministic.

---
← 68 · [Index](./README.md) · 70 →
