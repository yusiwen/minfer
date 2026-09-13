# 93 · Draft-quant swap is a mixed knob; greedy identity does NOT hold — the nt-invariance campaign has its motivating measurement

> **Result**: two measurements from the post-doc-92 follow-up. (1) Swapping the 0.5B draft between cached quants moves acceptance by ±3 pts per cell with opposite signs — q4_k_m takes code to 75.5% (+3.0% e2e, 44.2 tok/s) but drops prose to 48.5% (−2%); q5_k_m loses everywhere. The draft is already near its floor standalone (q4_0 at 583.8 tok/s ≈ 1.34× the weight-stream floor), so draft *speed* has little headroom — the earlier 2.9 ms/token inference was wrong. (2) The decisive one: **spec-draft greedy output is NOT identical to sequential greedy output** — both cells diverge (prose within ~8 tokens, code at line 19). Since the suite proves the batched and single matmuls are bitwise-equal, the divergence must originate in the target-side kernels that change shape with the verify batch — attention/softmax (and possibly norm/rope) — flipping argmax on near-ties. That is the motivating measurement for the nt-invariance campaign: make every target-side kernel bitwise-invariant to nt, with the identity test as its acceptance criterion.
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. Draft-quant swap (d=2, -n 128, same window, both cells)

| draft | prose acc / tok·s⁻¹ | code acc / tok·s⁻¹ | standalone 0.5B decode |
|---|---|---|---:|
| q4_0 (default) | 51.6% / 35.9 | 73.1% / 42.9 | 583.8 tok/s |
| q4_k_m | 48.5% / 35.2 | **75.5% / 44.2 (+3.0%)** | 634.8 tok/s |
| q5_k_m | 51.6% / 33.2 | 68.5% / 40.7 | 541.6 tok/s |

Reading: quantizing the same draft differently reshuffles its token choices by ULP-scale margins, and acceptance — a count of near-ties — moves by ±3 pts with opposite signs per cell. q4_k_m is the better draft for code-shaped workloads and the worse one for prose; q5_k_m's extra draft cost buys nothing. No default change is warranted from a mixed result; the recommendation is workload-dependent (q4_k_m for code sessions). The draft-cost picture is corrected: standalone q4_0 decode runs at ~1.34× its weight floor, so "optimize the draft kernels" has no meaningful prize — earlier round-arithmetic that implied 2.9 ms/token double-counted repair rounds.

## 2. The identity test and what it implies

Method: same prompts, `--greedy -n 128`, sequential target decode vs spec-draft (q4_0, d=2); diff the generated text after stripping stats lines.

- prose: diverges within ~8 generated tokens ("The history begins…" vs "The story begins…")
- code: diverges at line 19 (identical through the memoized Fibonacci body)

After the first flip the continuations are independent samples of the near-tie neighborhood, so the diff size says nothing about the flip count — the first-divergence position is the datum. The implication chain is tight:

1. The target's greedy choices should not depend on *how many tokens are verified together* — same KV, same weights, greedy argmax.
2. The suite's `cuda_multi_token_matmul_bitwise` proves batched and single **matmuls** are bitwise-equal.
3. Therefore the flip originates in a target-side kernel whose numerics change with the verify batch: the batched verify attention/softmax (nt=3 queries per block vs the decode kernel's 1), and secondarily norm/rope if their batched forms reorder reductions.
4. Each flip is simultaneously (a) a possible acceptance/quality deviation from the true greedy path, and (b) evidence of the uncontrolled ±3 pt acceptance jitter the quant swap just displayed.

## 3. Proposed campaign: verify-path nt-invariance

Goal: `spec-draft --greedy` output **token-for-token identical** to sequential decode on a fixed battery, by making the verify batch's kernel numerics bitwise-equal to the nt=1 path. Scope, in expected-difficulty order:

1. **First-divergence localization**: a token-stream diff (not text diff) between sequential and spec runs, per prompt — establishes which position and, with `MINFER_DUMP`/trace, which node flips.
2. **Attention/softmax**: make the batched verify attention produce bitwise-identical logits to the decode attention for the same query position (same key iteration order and online-softmax block schedule). The flash-decode split-K machinery makes this the hard part; the existing bitwise-test harness is the safety net.
3. **Norm/rope/sampler sweep**: per-row reductions are likely already invariant (grid-per-row forms); prove it with per-node A/B dumps, fix any that are not.
4. **Acceptance criterion**: the identity test battery (this document's method, several prompts, both cells) passes byte-identical; then re-run the doc-88 acceptance battery to measure whether sequential-faithful acceptance differs from the batched numbers (direction unknown, magnitude ±1–2 pts).

Payoff: speculative decoding becomes *exactly* transparent (bitwise-equivalent to sequential greedy) — a correctness guarantee competitors rarely state — and the acceptance statistics stop carrying quantization-jitter noise. Speed-wise it is neutral by construction; the d=8 door (code-p 0.755 needed vs 0.731) may open or close by up to ±2 pts as a side effect.

## 4. Verification for this document

- Three-quant e2e sweep: the table above (both cells, -n 128, same session window).
- Standalone 0.5B decode per quant: the table above (-n 64, greedy).
- Identity test: the diff commands in §2, stats lines stripped; both cells diverge.
