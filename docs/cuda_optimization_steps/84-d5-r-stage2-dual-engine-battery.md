# 84 · D5-R stage 2 — same-window dual-engine battery (LANDED)

> **Result**: 14B+0.5B q4_0, d=2, greedy n=128, 3 interleaved reps, one window: minfer **34.0 / 40.6 tok/s (1.33× / 1.59×)** vs serial 25.5; llama.cpp in the same window **38.8 / 49.0 (1.64× / 2.08×)** vs serial 23.7. minfer reaches **88% (prose) / 83% (code)** of llama's absolute speculative speed; gate ≥1.2× **PASS**. The whole gap decomposes to the verify marginal — closing it to llama's ~2.5 ms/row recovers 1.62×.
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. Background — where things stood

Stage 1 (doc 83) landed the spec loop and measured 1.34×/1.58× on single
runs. That satisfies the plan's stage-② gate arithmetically, but a campaign
claim needs the full protocol: multiple prompts, interleaved A/B against
**both** engines in one thermal/co-tenant window, acceptance reporting, and
an explicit decomposition of the remaining gap to llama.cpp. Doc 81 §4.3's
corrected llama battery (1.86× on its window) is the external reference;
this stage re-measures it in the same window as minfer.

## 2. Principle — what the battery must decide

One number decides the stage: minfer's speculative speedup relative to
llama.cpp's **in the same window**. Serial base rates are engine-dependent
(minfer's serial decode is ~8% faster here), so ratio-vs-own-serial is the
per-engine metric and absolute tok/s is the user metric. The gap between
the engines' ratios must be attributable to a measurable component before
stage ③ spends session time on it — the plan's candidate is the verify
marginal (cost per added verify row): minfer C_T(3)−C_T(1) = 56.6−39.0 =
**8.8 ms/row**; llama ≈ 47−42 = **2.5 ms/row** (doc 81 §4.3).

## 3. Implementation

### 3.1 Protocol

Same protocol as doc 81 §4.3's corrected battery, extended with the minfer
cells and interleaved across engines: 3 reps × {prose, code} × 4 cells —
minfer-off, llama-base, minfer-d2, llama-d2 — greedy, n=128, seed 42,
chat template, `-ngl 99/-ngld 99`, llama-cli `b10665-ca3d5a3e1` with the
explicit `--spec-type draft-simple` (the doc 81 errata's lesson), minfer
`--greedy --spec-draft … --spec-draft-n 2`. Per-cell wall time ~60 s
(loads dominate); one window, no other GPU work in between.

### 3.2 Measurement notes

- minfer acceptance comes from the round stats on stderr (deterministic
  under greedy: identical every rep). llama-cli `-st` prints no draft
  statistics; llama-side acceptance cites doc 81's `speculative-simple`
  measurement (p ≈ 0.74 at 14B).
- Reps are medians-of-3; spreads were ≤1% on every cell except llama d=8-
  class cells in doc 81 (not re-run here).

## 4. Results

14B q4_k_m + 0.5B q4_0 draft, d=2, medians of 3 interleaved reps, one window:

| cell | prose tok/s | code tok/s | speedup |
|---|---|---|---|
| minfer serial | 25.5 | 25.5 | 1.00× |
| minfer d=2 | 34.0 | 40.6 | **1.33× / 1.59×** |
| llama base | 23.7 | 23.6 | 1.00× |
| llama d=2 | 38.8 | 49.0 | **1.64× / 2.08×** |

- **Gate ≥1.2×: PASS** (1.33× / 1.59×; gate met on both prompts).
- Absolute: minfer d=2 reaches **34.0 vs llama 38.8** (88%) on prose and
  **40.6 vs 49.0** (83%) on code — while minfer's serial is *faster* than
  llama's (25.5 vs 23.7).
- Acceptance: minfer per-proposal 51.6% (prose) / 73.1% (code); tokens/round
  2.03 / 2.46. Code lands at doc 81's p≈0.74; prose is diluted by the
  near-tie flaps (doc 83 §3.4) and prompt dependence.
- Window note: llama d=2 prose measured 44.5 (1.86×) in doc 81's window vs
  38.8 (1.64×) here — a ~13% window shift on the speculative cell with a
  stable base (23.9 → 23.7); both engines' cells are internally tight
  (≤1%), so the same-window ratios are the trustworthy ones.

**Decomposition.** Round cost = verify(2) + 2·C_D + repairs + eager ≈
56.6 + 5.8 + 1.15 + 1.2 ≈ 64.7 ms over 2.03 tokens = 31.9 ms/token → 25.5/
31.9×25.5 ≈ 1.25×… measured 1.33× (the serial window rate is 25.5 → the
round model is within ~5%). If minfer's verify marginal fell to llama's
2.5 ms/row (nt=3 verify ≈ 44 ms), the round drops to ≈ 49.9 ms → 24.6
ms/token → **1.62× — llama's ratio, recovered**. That is the entire stage
③/④ prize, priced: ~8.8 → ~2.5 ms/row.

## 5. What was learned

- **Same-window interleaving matters more than rep count**: the llama
  speculative cell moved 13% between windows while its base moved <1% —
  ratio-only comparisons across windows would misattribute the drift to
  the engine.
- The gap to llama is **not** in drafting, acceptance machinery, or loop
  overhead (minfer's per-round non-verify cost is within ~1 ms of llama's
  structure); it is the verify row marginal, now measured from both sides.
- Acceptance dilution from numerics flaps is real but second-order: at
  p=0.52 prose the speedup still cleared the gate; recovering p→0.74 via
  nt-invariant accumulation is a ~+10% speedup on prose-class prompts
  (stage ④ candidate, doc 83 §3.4).

## 6. Next steps

Stage ③: ncu attribution of the verify marginal on the nt=3/nt=9 rounds —
candidates priced by this battery: (a) multi-MMVQ extension to nt=9–16
(the doc 82 kernels cap at acc[8]); (b) small-M GEMM tiles for nt 9+; (c)
attention query-tiling (KV read once per nt rows vs per row); (d) graph
capture for the fixed verify shapes (kills the ~1.2 ms eager overhead);
(e) the nt-invariant accumulation that would also restore exact greedy
identity (doc 83 §3.4). Deliverable: a per-item ms ledger before any
kernel work.
