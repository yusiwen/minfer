# 88 · D5-R stage 5 — the final dual-engine battery and the campaign verdict (LANDED)

> **Result**: same-window medians, 14B+0.5B q4_0 d=2: minfer **35.7 / 42.5 tok/s = 1.42× / 1.68×** vs serial 25.2; llama **37.7 / 48.4 = 1.65× / 2.10×**. minfer at **95% (prose) / 88% (code)** of llama's absolute speculative speed. d=8 measured 0.63× e2e — the doc-87 retirement confirmed end-to-end. Graph-capture prize verified already collected (48.08 captured vs 50.15 eager). D5-R closes: 4 stages landed, 1 measured negative, 1 target retired by measurement, 1 open kernel question filed.
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. Background — where things stood

Stages ①–④b landed the loop, the battery protocol, the per-kernel ledger,
the attention fix, and the measured negative on multi-MMQ extension (docs
83–87). Two items remained: verify the ledger's last assumed prize (launch
idle → CUDA-graph capture) and re-run the dual-engine battery on the
④a-updated binary — the campaign's final record.

## 2. Principle — what "done" means for a measurement campaign

A campaign stage is done when its gate is met or its gate is proven
miscalibrated by measurement. The two open questions were (a) whether the
2.4 ms/forward launch idle from the doc-85 ledger was still uncollected
prize, and (b) whether the ④a gains hold in the full same-window protocol.
(a) is answerable with an A/B of the existing capture machinery — it was
never new work, only a verification.

## 3. Implementation

### 3.1 The capture A/B

`minfer specverify -p 512 -r 7` with the default (R3-B prefill capture ON)
vs `MINFER_NO_PREFILL_CAPTURE=1`: **C_T(3) = 48.08 vs 50.15 ms** — the
existing `graph_replay` machinery already collects ~2.1 ms/forward, and
has been on for every number in docs 84–87. The doc-85 "graph capture"
prize was therefore already banked; the residual 4% nsys idle is the
per-forward synchronize + logits readback inherent to the eager sampling
loop.

### 3.2 The battery

The doc-84 protocol unchanged: 3 interleaved reps × {prose, code} × 4
cells, greedy, n=128, one window; d=8 e2e measured once on the same binary
for the record.

## 4. Results

| cell | prose tok/s | code tok/s | speedup |
|---|---|---|---|
| minfer serial | 25.2 | 25.3 | 1.00× |
| minfer d=2 | 35.7 | 42.5 | **1.42× / 1.68×** |
| llama base | 22.9 | 23.0 | 1.00× |
| llama d=2 | 37.7 | 48.4 | **1.65× / 2.10×** |

- Absolute: minfer at **95% (prose) / 88% (code)** of llama's speculative
  speed, while minfer's serial decode is ~10% faster.
- Progression across the campaign (prose/code): 1.33×/1.59× (doc 84) →
  **1.42×/1.68×** (④a attention) on the same protocol; acceptance
  unchanged (50.8% / 73.1%), so the gain is pure verify-cost reduction,
  as the ledger prescribed.
- d=8 e2e: **16.1 tok/s (0.63×)** — the doc-87 retirement (0.71× projected
  on prose-class acceptance) confirmed on the real prompt. The ≥1.5× d=8
  target is retired: at C_T(9) = 86.4 the round cost (~112 ms) cannot beat
  serial below p ≈ 0.62, and only the unexplained MMQ small-M gap
  (65 vs ~30 ms, ncu-blocked) could move it.
- Remaining gap to llama (prose 5%, code 12%): the code gap tracks the
  verify row marginal still above llama's (4.05 vs ~1.2 ms/row within the
  multi kernel — a kernel-family property); the prose gap adds the
  near-tie acceptance dilution (50.8% vs 73.1% code), which needs
  nt-invariant accumulation — both filed, neither cheap.

## 5. What was learned

- **One ledger item was already paid for**: before scheduling work, A/B the
  flag that controls it — the capture machinery from R3-B had silently
  banked the launch-overhead prize two campaigns ago.
- The campaign's shape is the honest one: 4 landed stages, 1 measured
  negative (④b), 1 retired target (d=8), 1 open question (MMQ small-M) —
  every gate either met or invalidated by measurement, none left asserted.
- The remaining distance to llama is now attributable to two named,
  bounded causes (verify row slope within the multi kernel; near-tie
  acceptance dilution), both requiring deeper kernel work (ncu
  permissions, nt-invariant accumulation) rather than dispatch changes.

## 6. Next steps (post-campaign)

Open leads, in expected-value order: (1) ncu on the small-M MMQ path once
counter permissions exist (the 2×-over-floor gap prices at ~35 ms at
nt=9 and ~8 ms at nt=3); (2) nt-invariant per-row accumulation — recovers
prose acceptance to ~0.7 and restores exact greedy identity (doc 83 §3.4);
(3) conversation/server modes behind the existing CLI surface. The
speculative feature ships: `--spec-draft <model> --spec-draft-n <2|8>`.
