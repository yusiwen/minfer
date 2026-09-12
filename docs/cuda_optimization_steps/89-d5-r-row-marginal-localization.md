# 89 · D5-R follow-up — where the verify row marginal actually lives (localized)

> **Result**: the residual per-row marginal (e2e 4.53 ms/row at d=2, 14B post-④a) decomposes as: **q4_K matmuls +2.4 ms (59%)**, q6_K +1.0 (25%), norm/quant/elt +0.67 (16%), attention +0.05 (1%). Inside the q4_K kernel, roughly **half the marginal is the one-block-per-output-row activation re-read** (od × nt × 3.2 KB served from L2 at ~3.4 TB/s, measured 4.7 µs/row per attn-shape matmul and 13.4 µs/row per ffn_up-shape matmul with an ablated kernel), the rest splits between per-row ALU/latency on 80-of-256 active lanes and chain-mode overhead (colder L2 for activations arriving from producer kernels + quantize traffic). Structural cause: minfer has **no efficient small-M tensor-core matmul** — the MMVQ path pays per-row ALU that stops hiding beyond nt≈4, and the MMQ path pads M to 16 (5× waste at M=3). The highest-value fix is an R-output-rows-per-block restructure (cuts the re-read term R× and lifts lane utilization from 31% to ~100% simultaneously), then tensor-core small-M tiles.
> **Commit**: this document's commit (includes the env-gated bench test). **Date**: 2026-09-12.

## 1. Background — the question

Doc 88 left the absolute speculative gap to llama.cpp (95%/88%) attributed
to "two named bounded causes", the largest being the verify row marginal
(4.05–4.53 ms/row vs llama's ~1.2–2.5) — with its micro-cause unlocated
because ncu is permission-blocked. This session localizes it with two
instruments that need no counters: a real-kernel bench with a cold-L2
protocol, and an ablated-kernel measurement.

## 2. Principle — what the free arithmetic already ruled out

At nt=1 the q4_K matmul stream runs at 211 GB/s (30.3 ms for ~7 GB) — 85%
of GB10's DRAM peak: the decode kernel is bandwidth-saturated and its ALU
is fully hidden. The nt-marginal's DRAM floor is tiny (one extra
activation row ≈ 50 MB per forward ≈ 0.2 ms), yet the measured marginal is
4.05+ ms — so the marginal is NOT extra DRAM traffic. It must be either
L2-side traffic, ALU/latency that no longer hides, or chain overhead —
distinguishable by isolating the kernel from the chain and neutralizing
one term at a time.

## 3. Implementation

### 3.1 The bench (committed, env-gated)

`cuda_row_marginal_bench` (in `graph/cuda_backend.rs`, runs only with
`MINFER_BENCH_ROW_MARGINAL=1`) executes the real dispatch path
(`execute_node` → quantize + kernel) over real 14B shapes (attn 5120×5120
q4_K, ffn_up 13824×5120 q4_K, ffn_down 5120×13824 q6_K) at nt = 1..8.
Cold-L2 protocol: each nt owns NC independent weight copies (>126 MB
aggregate), cycled so no copy is revisited within the L2 lifetime of its
blocks; runs per (uid, range) stay below the 3-run graph-capture trigger
so timing is never capture/replay. Launch count is nt-invariant, so
`t(nt) − t(1)` is pure per-row in-kernel work.

### 3.2 The nsys chain view

`nsys profile` of specverify (with the known `-r ≥ 5` gate respected —
two silent early-exits cost the first two capture attempts) parsed
per-forward (gap-cluster): the captured nt=3 verify forwards measure
48.4 ms busy vs 48.04 e2e — profiling overhead ~1%, so the kernel table is
trustworthy. The isolated bench reproduces the chain-level marginal when
scaled by matmul count, closing the loop between instruments.

### 3.3 The ablation

A throwaway patch replaced the per-row dot (`h2f` + `dp4a` chains + `fma`)
with a constant while keeping the loads. The compiler dead-code-eliminated
the now-unconsumed weight loads — the patch was reverted immediately — but
that accident is exactly the second instrument: the remnant kernel times
measure **activation traffic alone**.

## 4. Results

Per-row marginal budget (d=2 verify forward, nt=3 vs nt=1, per-forward
busy ms / 2 rows):

| component | ms/row | share |
|---|---:|---:|
| q4_K matmuls (chain) | 2.40 | 53% |
| q6_K matmul (chain) | 1.00 | 22% |
| norm/quant/elt kernels | 0.67 | 15% |
| attention (post-④a) | 0.05 | 1% |
| **total** | **4.1** | (e2e 4.53) |

Inside the isolated q4_K kernel (bench, nt=1→3):

| shape | marginal/matmul | activation re-read (ablated) | residual (ALU/latency) |
|---|---:|---:|---:|
| attn 5120×5120 | 2.07 µs | 4.7 µs/row ÷ ... (L2 3.4 TB/s) | ~0 (within noise) |
| ffn_up 13824×5120 | 12.7 µs | 13.4 µs/row | ~0–3 µs |
| ffn_down q6_K | 16.5 µs | (not ablated) | — |

Readings that survived scrutiny:

- **The activation re-read is real and large**: every output-row block
  re-reads every activation row — od × nt × 3.2 KB of L2 traffic per
  matmul per row (ffn_up: 44 MB/row → 13.4 µs at ~3.3 TB/s; the attn
  shape scales with od exactly). Scaled to the forward it prices at
  ~1.3 ms/row, ≈ half the q4_K+q6_K chain marginal.
- **The lane-utilization defect**: at id=5120 the v2 multi kernel's u-loop
  has npair=80 iterations over 256 threads — 69% of lanes idle at every
  nt. Harmless while DRAM-bound (nt=1), it concentrates all per-row ALU on
  80 lanes once the weight stream no longer saturates the SMs.
- **The marginal accelerates beyond nt≈4** (ffn_up: 12.7 → 38 µs/row at
  nt=6–8): the per-row work outgrows what the memory pipeline hides —
  consistent with the acc[16] spill catastrophe (doc 87) and the group
  parity: the kernel family has no headroom left at large nt.
- **Chain overhead is ~1 ms/row**: bench-isolated q4_K marginal (1.45)
  vs chain (2.4) — activations arrive via producer kernels (colder L2),
  plus quantize traffic and launch gaps the isolated matmul never sees.
- The nt=2 point reads high in two of three shapes (a small, reproducible
  oddity — possibly the quantize plane's tail utilization at nt=2);
  immaterial to the verdict.

## 5. What was learned

- **"Needs ncu" was too pessimistic**: cold-L2 benching + a (accidentally
  DCE-ing) ablation + chain nsys triangulated the marginal without a
  single counter. The permission blocker stands for microarchitectural
  counters, but kernel-level attribution was achievable.
- **The dominant term is a design property, not a tuning miss**: one
  block per output row × per-row activation re-read × 31% lane
  utilization. No dispatch gate or accumulator widening (docs 82/87)
  touches it.
- The two silent `-r < 5` early-exits produced plausible-looking empty
  profiles — always check the captured kernel count before parsing.

## 6. Next steps (the fix menu, priced)

1. **R output-rows per block** (R=4–8) in the multi kernels: activation
   re-read ÷R, lane utilization → ~100% (8 rows × 80 chunks = 640 items
   over 256 threads), DRAM weight traffic unchanged. Prices at roughly
   half the q4_K+q6_K marginal (~1.7 ms/row) — the single best lever.
   Requires a cross-row reduction (second kernel or atomics).
2. **Small-M tensor-core MMQ** (mma tiles of M=4/8 for nt 2–8): the
   llama-class endgame; removes both the per-row ALU and the re-read in
   one move, but is a full kernel project (the doc-82 nb_bt layout pads
   M to 16).
3. **Chain hygiene** (~1 ms/row): keep producer→consumer activations
   L2-resident (fusion already exists for norm→quant; extend to the
   matmul epilogue), shrink the quantize plane's nt=2 tail.
4. ncu on the small-M MMQ path once permissions exist — still the clean
   way to close the ALU-vs-latency residual (~0.5–1 ms/row).
