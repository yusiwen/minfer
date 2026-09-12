# 90 · D5-R follow-up — R-rows-per-block implemented, measured, reverted (menu item 1 closed)

> **Result**: the doc-89 §6.1 fix (~1.7 ms/row predicted) delivered **~0 at the d=2 shape** (C_T(3) 47.5–48.1 across runs vs the 48.04 baseline; acceptance stable at 51.6%/73.1%) and was reverted to restore the strict bitwise net. The falsification is the finding: in the real chain, activation rows are **L2-hot from the producer kernel**, so the re-read term priced from the isolated ablation does not exist at nt=3, and lifting lane utilization (31%→62% at R=2) did not move the needle either — **the nt=3 chain residual is per-row arithmetic throughput itself**, removable only by small-M tensor-core tiles. Side products kept: the cold-L2 bench (env-gated test), a measured nt≥6 flatten (attn nt=8: 126.9→94.4 µs — relevant only if a d≥4 shape ever ships), and two sharpened gotchas.
> **Commit**: this document's commit (bench test retained; kernels byte-identical to doc 88 state). **Date**: 2026-09-12.

## 1. Background — the hypothesis under test

Doc 89 localized the verify row marginal (4.5 ms/row) and priced three
fixes, the largest being R output-rows per block (~1.7 ms/row): one
activation pass shared by R weight rows cuts the re-read term R× and
lifts lane utilization at id=5120 (npair=80 < 256 threads). This session
implemented it and measured it.

## 2. Principle — what the implementation had to decide

Thread mapping `r = tid % R`, `u = tid/R + (256/R)·k` keeps one output
row per thread (acc[8] — no acc[16]-style pressure, the doc-87 trap) and
makes the R lanes of a chunk load identical activation addresses (one
transaction). The block reduction becomes a warp shuffle over same-r
lanes (offsets R..16 are multiples of R) plus one smem slot per warp —
all sound. The unavoidable cost: the block-level sum order changes, so
batched results stop being bitwise-identical to the single-token kernels
and the doc-82 bitwise test must drop to tolerance for these kernels —
weakening the campaign's cheapest safety net. A change that only breaks
even at the production shape cannot pay for that.

## 3. Implementation

Both v2 multi kernels (q4_K, padded q6_K) were rewritten as
`template<int R>` with the mapping above and the staged reduction;
launchers picked R adaptively. The bitwise test was relaxed to a 1e-3
tolerance for exactly the two restructured kernels (v1/q5_K stayed
strict), with NaN/±inf cases accepted only on exact match — random-byte
test weights make f16 scales hit NaN/inf patterns, and the old bitwise
assert had been silently passing on identical NaN payloads.

## 4. Results

| config | C_T(3) | note |
|---|---:|---|
| baseline (doc-82 design) | 48.04 / 48.08 / 48.28 | |
| R=8 everywhere | 52.40 | nt=3 loses badly |
| R=8 nt≥4, R=2 nt≤3 | 47.79 / 48.12 / 47.52 | neutral-to-noise |
| R=8 nt≥4, R=4 nt≤3 | 48.50 | block parallelism beats utilization |
| reverted | 48.28 | strict bitwise restored |

Isolated-bench detail (the useful part):

- **R=8 flattened the large-nt regime**: attn nt=8 126.9→94.4 µs,
  ffn_up 358→223 µs — the accelerating marginal (doc 89) is gone. But at
  nt=3 R=8 pays ~+9 µs/matmul: 8× fewer blocks cannot hide the DRAM
  weight-stream latency.
- **q6_K down regressed at R=8 for every nt** (~+40%): with 210 B row
  blocks, the R-lane weight pattern reads 8 scattered rows per warp —
  transaction overhead the q4_K layout (144 B, 64 B chunks) does not hit
  as hard.
- **R=2 at nt=3 moved nothing** despite halving activation traffic and
  doubling lane utilization → both terms were over-priced by the isolated
  ablation; in-chain the activation rows are resident in L2 from the
  producing kernel, so re-reading them is nearly free.
- Acceptance end-to-end: 51.6% prose / 73.1% code (was 50.8/73.1) — the
  ULP reordering does not disturb the near-tie statistics.

## 5. What was learned

- **Isolated-kernel ablations price terms that the chain gives away for
  free.** The doc-89 act-traffic term was measured on a kernel whose
  weights the compiler had eliminated and whose L2 saw only activations;
  in the real chain the producer leaves those rows hot. Chain nsys and
  isolated bench must be read together before pricing a fix.
- **Block-parallel latency hiding dominates utilization at small nt.**
  The 69%-idle-lanes defect (doc 89) is real but not binding at nt=3 —
  the weight-stream latency is hidden by block count, not lane count.
- **The nt=3 residual is arithmetic throughput per row** (dp4a chains on
  the rows' 80-lane groups) — irreducible in the dp4a design family.
  Doc 89's menu re-prices: small-M tensor-core mma is the only remaining
  lever of size; chain hygiene de-prices (the norm/quant bucket scales
  linearly with nt and the fuse saves only ~0.1 ms/row: measured nt=1
  fused 0.67 ms/row vs nt=3 unfused 0.55 ms/row — already honest).
- The `-r < 5` specverify gate exits silently before touching CUDA — two
  plausible-looking but empty nsys profiles were produced before checking
  the captured kernel count.
- Random-byte test weights + f16 scales: bitwise asserts can pass on
  matched NaN payloads — a tolerance rewrite must handle NaN/±inf
  explicitly or it fails on identical non-finite values.

## 6. Next steps

The gap menu collapses to one lever of size: **small-M tensor-core MMQ**
(mma tiles of M=4/8 for nt 2–8, replacing both the per-row-dp4a multi
kernels and the padded M=16 GEMM) — the llama-class endgame that removes
the row marginal, the utilization defect, and the nt≥9 cliff in one
design. It is a full kernel project (new weight-layout plumbing for
small-M fragments); everything cheaper has now been measured. The cold-L2
bench stays as the regression instrument for that work.
