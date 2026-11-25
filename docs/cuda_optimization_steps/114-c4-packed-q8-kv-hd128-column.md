# 114 · #848: the hd-128 column measured — and the four ratios #842 quotes (MEASURED)

> **Result**: on one tree, one box and one command, the packed/f16 column now reads: hd 64
> decode **0.9712x** (q8_0 245.74 vs f16 238.67 tok/s) and prefill **1.2451x** (2 180.6 vs
> 2 715.0); hd 128 decode **1.0043x** (158.04 vs 158.72) and prefill **1.0796x** (8 147 vs
> 8 796). The hd-128 pair is the column's **first current-tree measurement** — the figures it
> replaces (1.01x decode, 1.04x prefill) came from a 2026-09-27 tree and from the [#144](https://github.com/yusiwen/minfer/issues/144)
> round. Decode at hd 128 is **parity within 0.5%**, and the kernel trace accounts for it
> line by line: the packed arm runs the one-warp body (**42.18 µs/layer**) because
> `attn_split_h4w_body` takes `const __half*`, while f16 runs the 4-warp hybrid
> (**39.65 µs/layer** + a 0.80 µs early-exit launch), and the packed epilogue adds
> **+0.29 µs/layer** — **+55.4 µs of a 6.30 ms step**, against a −30.3 µs cross-arm
> difference in weight matmuls the KV format cannot touch. So the geometry [#848](https://github.com/yusiwen/minfer/issues/848) wants to
> fix is worth **≤0.9% of a decode step**, not a residual. The prefill half is a different
> cause: at hd 128 *both* arms take the FA (flash-attention) prefill, and the packed arm's
> tile staging makes `fa_prefill_kv` **+44.8%** per launch (460.56 → 666.78 µs), which is
> 18.8 ms of the 18.6 ms prefill delta per 2 048-token prompt.
> **Bar**: pre-registered on [#848](https://github.com/yusiwen/minfer/issues/848) on 2026-11-25 before these runs — decode *better than*
> the pre-E1 1.01x (falsified if ≥1.01x), prefill 1.00–1.10x, with the decision rule.
> **MET** on both cells: 1.0043x and 1.0796x.
> **Commit**: `c08589d` (no repo change — a measurement and this record).
> **Date**: 2026-11-25. **Box**: dgxspark (aarch64, GB10 sm_121, driver 580.178.04,
> CUDA 13.0, nsys 2025.3.2).

## 1. Background — where things stood

Record [111](./111-c4-packed-decode-epilogue-and-two-cell.md) and record
[112](./112-c4-packed-decode-four-cells-per-warp.md) closed the **hd-64** decode residual:
1.239x → 1.0139x → 0.9705x, the last of them by keeping four KV (key-value) cells per warp.
Both records are explicit that their lever is `hd == 64`-gated, and neither measured the
other column.

That other column has been quoted for two months from measurement trees that no longer
exist. Decode: **1.01x**, 136.66 vs 136.73 tok/s on Qwen3-0.6B Q8_0 ([record
108](./108-c4-dp4a-packed-q8-kv-cuda.md), 2026-09-27). Prefill: **1.04x** from the [#144](https://github.com/yusiwen/minfer/issues/144)
round ([record 107](./107-c4-packed-q8-kv-cuda.md)). Four levers have landed since — the
dp4a K dot, #186's wide arm, E1's epilogue rewrite, E2's cell packing — and none of them was
measured here.

[#848](https://github.com/yusiwen/minfer/issues/848) filed the reason the pair cannot simply be reprinted: the two arms do not run the
same body. `attn_split_h4w_body` takes `const __half*` and "is not converted, so it is not
offered for a packed cell" (`src/cuda/kernels/attention_decode.cu`, the
`launch_gqa_attn_split_q8_0` comment), so at hd 128 the f16 arm runs a 4-warp hybrid over
more rows per warp while the packed arm is stuck on the 32-split one-warp body. A ratio
between two bodies is a fact about the shipping code — which is what a reader of the book
actually runs — but it is not a statement about how close the *implementations* are, and the
ticket is right that reprinting it as "current" hides which of the two it is.

[#842](https://github.com/yusiwen/minfer/issues/842) is the book's carrier: Chapter 10 · Lesson 5 quotes the packed Q8_0 residual, and its
own agent asked for the four ratios — hd 64 and hd 128, decode and prefill — with the
command, the device and the commit, rather than for a code change. This record is that
deliverable, plus the kernel attribution that says what the remaining hd-128 gap *is*.

## 2. Principle — why hd 128 cannot inherit the hd-64 levers

Three separate mechanisms decide this column, and they do not share a gate.

**The decode geometry.** `hd == 64` is where E2 lives: 32 lanes × 4 dims = 128, so a warp
carrying one KV cell has 16 live lanes, and records 111/112 spend them on a second and then
a third and fourth cell. At hd 128, 32 × 4 = 128 is exact — every lane is already doing a
cell — so the lever has nothing to spend and is gated off (`hd == 64` dispatch).

The f16 arm at hd 128 has a *different* body available: `gqa_attn_split_partial_hybrid`,
the four-warp hybrid that splits rows between warps and merges them, gated on
`rpw >= H4W_MIN_RPW` (rows per warp ≥ 16, i.e. nkv ≥ 1 921). At the measured shape
(`-p 2048 -n 128`, nkv 2 049 → 2 176) that gate is open, so f16 launches **two** kernels per
layer — the hybrid that works and the incumbent that exits before touching memory (0.80 µs,
the "dud launch" the code accepts to keep the grid static for CUDA-graph capture). The packed
launch keeps `rpw_gate = 0`, so it launches one kernel per layer, the one-warp body, and
never gets the hybrid.

**The arithmetic that makes this a parity result anyway.** Per layer, the decode step is
about 228 µs: four weight matmuls (`q8_0_f32_matmul` 30.6 + `q8_0_p32_q8_mmvq` 13.5) at
~176 µs, attention at ~45 µs, the rest norms and copies. The packed cell reads 1.88x fewer
KV bytes than f16 (99.4 vs 187 MB per step at hd 64; the same ratio holds here), so a body
4% slower per layer still lands within half a percent of f16 at the step level. That is the
whole finding: **the byte advantage is already buying back the geometry disadvantage**, and
what is left over is smaller than the cross-arm noise in the trace.

**The prefill geometry.** The FA prefill template takes a layout parameter — `fa_prefill_kv`
exists as `<(bool)1, (bool)0, (int)1>` (f16) and `<(bool)1, (bool)0, (int)2>` (packed) —
because a packed cell cannot be read as `__half` tiles; its blocks are staged (dequantized)
into f16 tiles first. The gate is `nt >= 2 && hd == 128 && layout != f32`, so at hd 128 the
**packed arm does take the FA prefill**, at the price of that staging; at hd 64 neither arm
does, and the packed prefill falls back to the per-element-dequantizing body — which is
[#845](https://github.com/yusiwen/minfer/issues/845) and is why the hd-64 prefill penalty (1.245x) is three times the hd-128 one (1.080x).
The two prefill penalties have nothing in common except the format.

## 3. Implementation — no code; the measurement recipe

### 3.1 The bench matrix

Both models, both KV formats, three **alternating** rounds — the arm order flips each round
(`f16, q8_0` then `q8_0, f16` then `f16, q8_0`) so a monotone clock or thermal drift cannot
land on one arm:

```text
MINFER_CACHE_TYPE=f16   minfer bench -p 2048 -n 128 -r 3 --n-ctx 4096 <model>
MINFER_CACHE_TYPE=q8_0  minfer bench -p 2048 -n 128 -r 3 --n-ctx 4096 <model>
```

- hd 64: `qwen2.5-0.5b-instruct-q4_0.gguf` (2 715 / 2 181 tok/s `pp2048`, 238.67 / 245.74 `tg128`);
- hd 128: `Qwen3-0.6B-Q8_0.gguf` (8 796 / 8 147 `pp2048`, 158.72 / 158.04 `tg128`).

Each `bench` rep is a mean over its own 3 measured reps after a 2 000 ms warmup; the number
this record reports per round is that rep's `ts_mean` from `-o json`, and the reported figure
is the **median of the three rounds**. The binary was rebuilt for the run
(`cargo build --release --features cuda`, 2m59s, `sm_75…sm_121` + `compute_121`) so the thing
that was timed and the commit this record names are the same tree.

### 3.2 The kernel attribution, and its trap

nsys 2025.3.2, `--cuda-graph-trace=node`, one `-p 2048 -n 128 -r 1` run per arm, medians over
**10 752 launches per kernel** (384 decode forwards × 28 layers).

The trap is worth recording because the first attempt produced a plausible and wrong answer.
The decode loop replays a **captured CUDA graph**, and `nsys` does not decode graph nodes by
default: without `--cuda-graph-trace=node` the kernel table holds 1 070 rows spanning
**0.015 s of a 4.2 s run** — roughly two forwards — while the memcpy and runtime tables span
the whole run. Every median in that report is then a median over the wrong population, and
nothing about it looks broken: the numbers are the right order of magnitude. The check is the
**span**, not the row count.

### 3.3 Pitfalls

- **Two bodies, two launch counts.** f16 issues `gqa_attn_split_partial` (early exit, 0.80 µs)
  *and* `gqa_attn_split_partial_hybrid` (the work, 39.65 µs) per layer; the packed arm issues
  one `gqa_attn_split_partial<(int)2,…>` (42.18 µs). Summing only the "real" kernel of each arm
  understates the f16 side by the dud launch; summing the per-layer totals is what reconciles
  with the step.
- **Cross-arm kernel noise is the size of the effect.** The weight matmuls are the same code
  in both arms — the KV format is not an input — and they still read 30.62 vs 30.85 µs
  (`q8_0_f32_matmul`) and 13.50 vs 13.54 (`q8_0_p32_q8_mmvq`). That is ±0.7%, and it is
  −30.3 µs/step, which is larger than the step delta being explained. Any attribution below
  that floor has to be stated as "within trace noise", not as a mechanism.
- **Prefill passes are counted, not guessed.** `fa_prefill_kv` appears 364 times (f16) and 336
  (q8_0) across four prefill passes: 91 vs 84 launches per prompt, and 91 × 206.22 µs =
  18.8 ms against the 18.6 ms delta in `pp2048`. The reconciliation is what makes the
  "staging cost" claim evidence rather than an attribution story.

## 4. Verification

No source line changed, so this record's gates are the ones that make its **numbers**
trustworthy, not the ones that would defend a kernel:

- **the tree**: `cargo build --release --features cuda` at `c08589d` succeeded and the binary
  is the one timed; `git status` clean apart from the never-committed `reports/`;
- **the cell bytes**: the packed `q8_0` cell this column measures is #112's — nothing in this
  step writes or reads a different layout, so the numerics class is #111/#112's
  (max |Δlogit| 2.4795, argmax 0.5201, greedy 9-of-9), unchanged and not re-measured here;
- **reproducibility across sessions**: hd-64 decode read **0.9712x** here against record 112's
  **0.9705x**, and its f16 arm read 238.67 vs 236.18 tok/s — a session ~1% faster on *both*
  arms, with the ratio preserved to 0.07%. That is the control that says the ratio is the
  portable quantity and the tok/s figures are not;
- **doc gates**: `build_book.sh`, `check_docs_links`, `check_doc_line_anchors`,
  `check_doc_counts`, `check_doc_hashes`, `check_doc_code_copies`, `check_book_coverage`,
  `check_book_links`, `check_env_vars`, `check_support_matrix` and `check_anchor_drift`
  against `origin/master`, all green on the record commit.

## 5. Results

### 5.1 The four cells

Medians of three alternating rounds; every round's `ts_mean` in brackets, f16 first.

| | decode (`tg128`, tok/s) | ratio | prefill (`pp2048`, tok/s) | ratio |
|---|---|---|---|---|
| **hd 64**, Qwen2.5-0.5B q4_0 | 238.67 [239.84, 238.67, 238.53] vs 245.74 [246.22, 245.74, 245.02] | **0.9712x** | 2 714.98 [2 736.73, 2 714.98, 2 676.52] vs 2 180.56 [2 180.56, 2 192.53, 2 167.26] | **1.2451x** |
| **hd 128**, Qwen3-0.6B Q8_0 | 158.72 [158.72, 158.61, 158.90] vs 158.04 [158.04, 158.17, 157.96] | **1.0043x** | 8 795.89 [8 795.89, 8 824.28, 8 614.43] vs 8 147.13 [8 387.11, 8 147.13, 8 103.41] | **1.0796x** |

Step times: hd 64 4.190 → 4.069 ms (−121 µs); hd 128 6.3004 → 6.3275 ms (**+27.1 µs**).

Two caveats belong beside the table, not after it.

- **The hd-64 prefill ratio is the least stable cell.** This session read 1.2451x where
  records 111/845 read 1.232x/1.224x, and the f16 arm alone moved +1.5% (2 715 vs 2 675).
  Interleaving cancels drift *within* a session, not between sessions: the honest statement
  is **1.23–1.25x at 2k**, not 1.2451x.
- **Every ratio is one model, one prompt length, one context.** The 8k hd-64 decode row
  (0.906x, one round) and the hd-128 column were not measured at 8k.

### 5.2 What the hd-128 decode residual is made of

nsys medians, µs per layer, at nkv 2 049–2 176:

| kernel | f16 | packed q8_0 | delta |
|---|---|---|---|
| attention body (the working launch) | **39.65** (`gqa_attn_split_partial_hybrid`) | **42.18** (`gqa_attn_split_partial<(int)2,…>`) | **+2.53** |
| attention body (the early-exit launch) | 0.80 | — (not launched) | −0.80 |
| `gqa_attn_split_combine` | 2.82 | 2.78 | −0.04 |
| QKV epilogue / KV store | 1.34 (`attn_rope_store_f32`) | 1.63 (`attn_rope_store_q8_0`) | **+0.29** |
| **per layer, total** | **44.61** | **46.59** | **+1.98** |
| **per step, × 28 layers** | | | **+55.4 µs** |

Against the +27.1 µs the bench measured, the weight matmuls contribute **−30.3 µs** of
cross-arm noise (they do not read the KV format; see §3.3), leaving +25.1 µs predicted versus
+27.1 µs measured. The attention share of the step is 44.6 µs of ~228 (19.6%), which is why a
+1.98 µs/layer difference of two different bodies lands inside half a percent of the step.

The epilogue's +0.29 µs/layer is the *residue* of E1, and it is worth stating plainly: at hd 64
E1 took 21.6 → 1.92 µs/layer, and the same kernel at hd 128 costs 1.63 µs/layer against the
f16 store's 1.34. The 4-thread serial loop E1 removed was a **thread-mapping** cost, and one
thread per (head, block) at hd 128 is already 8 heads × 4 blocks = 32 threads — a full warp —
so the mapping E1 introduced changes much less here than it did where four threads were
carrying 128 elements.

### 5.3 What the hd-128 prefill residual is made of

| kernel | f16 | packed q8_0 | delta |
|---|---|---|---|
| `fa_prefill_kv` | 460.56 µs (364 launches) | 666.78 µs (336 launches) | **+206.22 (+44.8%)** |

Both arms take the FA prefill here — the packed one through the `<(int)2>` layout
instantiation, which stages each packed block into f16 tiles before the tensor-core tile
maths. 91 launches per 2 048-token prompt × 206.22 µs = 18.8 ms, against the 18.6 ms the
`pp2048` figures differ by (232.8 vs 251.4 ms). This is a **different defect** from [#845](https://github.com/yusiwen/minfer/issues/845)'s
(at hd 64 there is no FA prefill at all), it had no ticket before this measurement, and it
is filed as [#858](https://github.com/yusiwen/minfer/issues/858) with the three shapes the
evidence favours — stage once per layer, dequantize inside the tile maths, or write the 8%
down as the price of 1.88x fewer bytes.

## 6. Lessons

1. **A ratio between two bodies is still the number a reader runs — print it, and print the
   body.** "1.01x at hd 128" was not wrong for two months, it was *undocumented*; the fix is
   one clause naming the geometry, not a withheld figure.
2. **hd-gating is invisible in a result.** E1 (hd-independent) moved hd 128 by ~0.6 pp, E2
   (hd-64-gated) moved hd 64 by 22 pp. A lever's record must say which shapes it cannot see,
   or its successor will assume it saw them all.
3. **`nsys` silently profiles 1% of a CUDA-graph run unless told to decode nodes.** Check the
   span of the activity table, not the number of rows, and never trust a median whose
   population was not counted.
4. **Reconcile or say "within noise".** Kernel deltas must be summed, multiplied by the layer
   count, and compared with the step; a 0.7% cross-arm difference in a kernel the change does
   not touch is the size of a parity claim, and naming it is what keeps the claim honest.
5. **A penalty can hide in a body that exists.** The hd-128 prefill gap was never a
   missing kernel — both arms take the FA prefill — it was 45% of staging inside the
   second instantiation. "Which body runs" found #845; "what that body pays for the
   format" found this one.

← [113 · #844: the packed decode takes the QKV fusion on Metal, and the rope becomes bit-equal](./113-c4-packed-decode-metal-fused-epilogue.md) · [Index](./index.md)
