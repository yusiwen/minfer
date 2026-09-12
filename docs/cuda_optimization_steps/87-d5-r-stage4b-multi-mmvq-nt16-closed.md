# 87 · D5-R stage 4b — can multi-MMVQ beat the padded GEMM at nt 9–16? (CLOSED — measured negative)

> **Result**: No — within the acc-array MMVQ design family, nt 9–16 cannot beat the padded MMQ GEMM: token-groups-of-8 land at **parity (85.0 vs 84.9 ms matmul)** because group ≥ 1 re-streams the row's weights from DRAM (L2 cannot hold the streamed rows), and a single-pass acc[16] variant **regresses to 111 ms** (register spill). The doc-82 GEMM boundary at nt ≥ 9 stands; the doc-85 ledger's 4.05 ms/row extrapolation beyond one group was wrong. Dispatch reverted and verified bitwise (suite 179 green, incl. the multi-token bitwise test); d=8 economics unchanged and re-scoped.
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. Background — where things stood

The doc 85 ledger's item ②: at nt=9 the matmuls leave multi-MMVQ (cap
nt ≤ 8) for the padded GEMM `mmq_raw_nb_bt` (M=9 < tile 16), costing
84.9 ms where the multi row slope projected ~72.7 — a ~14 ms prize that
would make d=8 speculative decoding viable. This stage implements the
extension and tests the projection.

## 2. Principle — why the extension should have worked, and why it can't

The multi-MMVQ kernel assigns one block per weight row; 256 threads stride
the row's sub-blocks, and `acc[8]` accumulates one output per token with
the weight chunk loaded once per thread-tile — weights stream once
regardless of nt **as long as every token's accumulator is live during
that single pass**. Eight accumulators per thread is the register budget's
comfort limit (doc 82). For nt = 9–16 there are exactly two designs inside
this family:

1. **Token groups of 8** (acc[8] reused per group): group g > 0 must
   re-read the row's weights. The hope was L2: a q4_K row is ~414 KB, and
   GB10's L2 is 126 MB — but the working set is not one row, it is the
   whole weight tensor streaming through 2880 concurrent blocks
   (~1.2 GB); by the time group 1's block revisits a chunk, it is evicted.
   Group re-reads therefore come from DRAM: the weight stream is paid
   once per group — the very cost the family exists to avoid.
2. **A single pass with acc[16]**: 16 live accumulators + 8 weight
   registers per thread overflow the register budget at
   `__launch_bounds__(256)`; nvcc spills to local memory, and local-memory
   traffic lands on the same DRAM the kernel was trying to save
   (measured below).

## 3. Implementation

- `q4_k_q8_mmvq_multi` (v1), `q4_k_q8_mmvq_v2_multi`, `q6_k_q8_mmvq_v2_multi`
  gained the token-group loop (group of 8, `tmax = min(8, nt − t0)`,
  `mmvq_block_reduce_multi` takes a `t0` output-row offset). For nt ≤ 8 the
  loop runs exactly one group with the original accumulation order —
  bitwise-identical outputs, which the `cuda_multi_token_matmul_bitwise`
  test asserts.
- The doc-82 GEMM gate (`nt >= 9`) was temporarily raised for
  Q4_K/Q6_K (`gemm_min_nt = 17`) to route verify shapes to the multi path,
  and an acc[16] single-pass pair (`*_multi16`) was measured; both were
  **reverted** after the numbers below.
- q5_K's multi kernels were left at 8 tokens (its dispatch gate keeps
  `nt <= 8`), so no q5_K shape can silently take an 8-row kernel at nt > 8.

## 4. Results

14B q4_k_m, KV 512, specverify medians (matmul ms from the doc-85 nsys
method):

| variant | C_T(3) | C_T(9) | matmul @ nt=9 |
|---|---|---|---|
| baseline (MMQ GEMM, doc 82 boundary) | 48.0 | 86.4 | 84.9 |
| token groups of 8 | 48.1 | 88.0 | 85.0 — **parity** |
| acc[16] single pass | 48.4 | **111.3** | spills — **−25 ms worse** |

The doc-85 "multi row slope 4.05 ms/row" was measured across nt 1→3 —
**within one group**, where the weight stream is genuinely shared. Beyond
8 tokens each additional group pays a full weight stream (~30 ms), i.e.
the group marginal is ~3.8 ms/row *plus* ~30 ms per group boundary — which
is why the GEMM (one padded pass, 6.1 ms/row) matches it exactly at nt=9.

**d=8 economics (re-scoped).** With C_T(9) = 86.4 standing, a d=8 round
costs ≈ 86.4 + 8×2.9 + overhead ≈ 112 ms. At prose-class acceptance
(p ≈ 0.52, tokens/round ≈ 2.0) that is a 0.7× loss; even code-class
(p ≈ 0.73, ≈ 3.5 tokens/round) reaches only ~1.25×. **d=8 is not viable
at the current verify curve and the stage-⑤ ≥1.5× d=8 target is retired
honestly**: the real nt=9 prize is the MMQ kernel's own distance from the
weight-stream floor (65.1 ms for q4_K at M=9 vs the ~30 ms one-pass floor
— a 2× gap whose cause needs ncu, permission-blocked), not the MMVQ/GEMM
choice.

**The bitwise safety net worked.** The first group-extension build passed
timing but failed `cuda_multi_token_matmul_bitwise` + parity tests: a
variable rename collision (`g` = group index shadowing the sub-block's
`g` in the q6_K mapping) silently mis-selected quantization pieces.
Acceptance in an end-to-end run dropped 50.8% → 38.4% before the suite
caught it. The rename (`gq`) restored bitwise equality (179 green) and
the measured acceptance.

## 5. What was learned

- **Extrapolating a per-row slope across a kernel's structural boundary
  (here: accumulator lifetime) is how measurement plans go wrong** — the
  ledger should annotate which region each slope was measured in.
- Two negative results with numbers (parity, spill regression) close a
  design branch permanently; the doc-82 boundary was not a limitation to
  fix but a measured equilibrium.
- The bitwise multi-token test is the campaign's cheapest insurance —
  it caught in seconds what an acceptance-rate regression would have
  attributed to "prompt noise".

## 6. Next steps

Stage ④c: CUDA-graph capture for the fixed verify shapes (~2.4 ms/round
launch idle per doc 85) — the last unpriced item that moves d=2. Stage ⑤:
the dual-engine re-battery at d=2 (llama references 1.64×/2.08×
same-window), with d=8 recorded as measured-infeasible at the current
verify curve and the MMQ small-M gap (65 vs ~30 ms) filed as the open
kernel question that would change that.
