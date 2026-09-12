# 81 · D5-1a — the verify-batch gate measured end-to-end: no amortization at any nt, D5 closed (LANDED · gate FAIL)

> **Result**: the pre-registered D5-0 gate FAILED decisively. End-to-end
> target verify cost at KV depth 512 (7B q4_k_m, CUDA): C_T(1)=18.33 ms,
> C_T(3)=106.0 ms → per-token amortization **0.52×** vs the required ≥ 2.5×
> (C_T(3) would have had to be ≤ 22.1 ms). The full curve shows why rescue is
> impossible: every batched step nt=2…8 costs a flat **~35 ms per token**
> (zero amortization — weights are re-streamed per row), and the first
> efficient regime only starts at nt≥16 (56.7 ms total = 3.1× the
> weight-streaming floor), which speculative verify (nt=d+1 ≤ 9) can never
> reach. Per the stop rule pre-registered in D5-0 and the plan: **the D5
> speculative-decoding campaign is CLOSED**.
> **Commit**: this one (specverify instrument + this doc). **Date**: 2026-09-10.

## 1. Background — where things stood

D5-0 (doc 80) built the speculative-decoding cost model from measured
pieces: target C_T(1)=18.42 ms (7B q4_k_m CUDA, from `bench` tok/s), draft
C_D=2.92 ms (0.5B q4_0), real acceptance p≈0.68–0.70 (llama.cpp
`speculative-simple`). Break-even at d=2 needed p*=0.73 — measured p fell
short — UNLESS the target's nt=3 verify batch achieved the "BT-MMQ
amortization": D4-1 had measured a 2.7× per-token win at nt=4, and the D5-0
projection at that anchor was a marginal 1.04–1.05× (ceiling ~1.2×). Two
hypotheses disagreed about nt=3 — linear interpolation said 2.28× (lose),
the "same-tile-as-M=4" tile-regime step hypothesis said 2.7× (win) — and doc
80's verdict was therefore *conditional*: measure the nt=3 amortization
BEFORE building any loop plumbing; < 2.5× → STOP with the negative
documented.

That measurement is this doc. It is the first end-to-end measurement of
minfer's batched verify step; every prior amortization number was a
kernel-level micro-bench.

## 2. Principle — what the gate asks, and what the machine can answer

A verify pass is ONE target forward at `nt = d+1`, `n_out = d+1`. The gate
quantity is the per-token amortization:

```
amortization(nt) = nt · C_T(1) / C_T(nt)      required: ≥ 2.5× at nt=3
⇔ C_T(3) ≤ 3 · C_T(1) / 2.5 = 3 · 18.33 / 2.5 = 22.1 ms
```

The physical ceiling any engine could hope for: at nt=3 the GEMMs must
stream the same ~4.4 GiB of weights once (C_T(1) is exactly that stream:
18.33 ms ≈ 4.36 GiB at ~240 GB/s effective), so the best conceivable
C_T(3) is ≈ C_T(1) + small marginal → amortization ceiling ≈ 3.0×. The 2.5×
bar therefore demands a near-zero-marginal verify: the batched forward must
cost almost the same as the single-token forward. That is what "the D4-1
amortization" was supposed to deliver — and what the measurement below shows
the graph-level dispatch never does.

## 3. Implementation

### 3.1 The instrument — `minfer specverify`

New subcommand (`src/spec_verify.rs`), the only code D5-1a lands. It drives
the existing generic primitive `forward_graph_cached(tokens, positions,
n_out)` — at `nt > 1` the CUDA backend already dispatches the batched path
(MMQ GEMM + full attention; the decode-only MMVQ / FusedQKV / FusedFFN /
split-K flash-decoding paths are all `nt == 1`-gated), so the measurement
needs no engine change.

Protocol (campaign methodology):

1. prefill `P` tokens (default 512, untimed) — fills KV so every later
   attention read hits an initialized slot;
2. per `nt` phase: 3 untimed warmups (absorbing the one-time graph rebuild —
   `GraphParams.n_tokens` is part of the reuse identity), then `r` timed
   steps at fixed positions `[P-nt..P)`, `n_out = nt`, medians reported;
3. two passes (nt ascending, then descending) — the second is the drift
   check; GB10 does not support `nvidia-smi -lgc` clock locking, so drift is
   handled by medians and the reversal, as in prior steps.

Probe switches used for the attribution below: `MINFER_SPECVERIFY_NTS`
(phase list), `MINFER_SPECVERIFY_NOUT=1` (force prefill-style tail-row
output).

### 3.2 What the first run showed, and the five probes

First measurement: C_T(3)=106 ms — not "unamortized" but *pathological*
(>3× the naive 3·C_T(1) bound). Before trusting it, five alternative
explanations were tested and eliminated:

| Hypothesis | Probe | Outcome |
|---|---|---|
| CUDA-graph launch asymmetry (nt=1 replays a graph, nt=3 runs eager) | `MINFER_NO_CUDA_GRAPH=1` | C_T(1) 18.3→19.5 ms; C_T(3) unchanged → not launch overhead |
| Attention pathology at deep KV | `-p 64` vs `-p 512` | C_T(3) 102 vs 106 ms → not depth |
| The FA prefill kernel at tiny query counts | `MINFER_NO_FA_PREFILL=1` | unchanged → not the FA kernel |
| The `n_out=nt` output path (no tail-row reduction, lm_head at M=3) | `MINFER_SPECVERIFY_NOUT=1` | 106→96 ms → output path is ~10 ms, not the cause |
| A fluke / clock drift | reverse pass, 15–40 reps | stable to ±2% → real |

## 4. Verification — the measured pricing curve

7B q4_k_m, CUDA, KV depth 512, medians of 15–40 timed reps (`/tmp/d51a/`):

| nt | C_T(nt) | per token | per-token amortization |
|---|---|---|---|
| 1 | 18.33 ms | 18.33 ms | 1.00× |
| 2 | 72.71 ms | 36.35 ms | 0.50× |
| 3 | 106.03 ms | 35.34 ms | **0.52×** (gate needs ≥ 2.5×) |
| 4 | 139.87 ms | 34.97 ms | 0.52× |
| 5 | 174.38 ms | 34.88 ms | 0.53× |
| 6 | 207.43 ms | 34.57 ms | 0.53× |
| 8 | 277.50 ms | 34.69 ms | 0.53× |
| 16 | 56.67 ms | 3.54 ms | 5.17× |
| 64 | 64.88 ms | 1.01 ms | 17.99× |

Three facts, each fatal to a different assumption:

1. **nt=2…8: per-token cost is FLAT at ~35 ms — zero amortization.** The
   batched path re-streams the weights for every row (C_T grows linearly,
   ~35 ms per added token). It is also **1.9× slower per token than the
   nt=1 MMVQ decode path** — the batched dispatch is not merely unamortized,
   it is a worse kernel. The D5-0 linear interpolation (2.28× at nt=3) and
   the tile-regime step hypothesis (2.7×) are both refuted: there is no
   regime near M=4 at all.
2. **The real tile-regime step sits at M≥16, not M=4.** nt=16 costs 56.7 ms
   TOTAL — less than nt=8 (277.5 ms) — because only from M=16 (the mma
   M-tile) does the batched GEMM stream weights once per forward. But even
   that regime runs at 3.1× the weight-streaming floor (56.7 vs 18.3 ms),
   and speculative verify at d≤8 can never reach M=16. Padding the verify
   batch up to the good regime (nt=3→16) would give 3·18.33/56.7 = **0.97×**
   — still under break-even even before the draft cost.
3. **The kernel-level anchor never existed end-to-end.** D4-1's 2.7×@nt=4
   was a BT-GEMM micro-bench; the graph around it (dispatch shape, weight
   re-reads at small M) delivers 0.52×. This is the D5-0 lesson
   "interpolation is not measurement" landing a level deeper: *kernel*
   numbers are not *graph* numbers either.

Gate arithmetic, final: required C_T(3) ≤ 22.1 ms, measured 106.0 ms —
**FAIL by 4.8×**. Even the best measurable batched point (nt=16) cannot beat
1.0× per-token amortization. No dispatch fix within the current kernel
family can close a 4.8× gap; the win the gate demanded (verify ≈ free
marginal cost on top of one weight stream) does not exist in this engine's
batched path, and the D4-1 anchor that justified looking does not survive
contact with the full graph.

**Campaign decision**: the stop rule pre-registered in D5-0 and
`SPECULATIVE-DECODING-PLAN.md` ("< 2.5× → STOP, document the negative, close
the campaign after the primitive") triggers. D5 is closed after D5-1a; no
`Speculator` trait, no KV rollback, no loop plumbing will be built. The
instrument (`specverify`) remains as the primitive's usable residue.

### 4.1 External reference — llama.cpp itself lands at 1.00×

The gate's arithmetic is only as good as its input costs, so the same model
pair was rerun through llama.cpp's own mature speculative implementation as
an outside check (llama-cli build `b10665-ca3d5a3e1`, CUDA: 7B q4_k_m
`-ngl 99` + 0.5B q4_0 `-md -ngld 99`, greedy `--temp 0 -n 128 -t 20 -s 42`,
medians of 3 interleaved reps, prose and code prompts; raw log
`/tmp/d51a/llama/battery.txt`):

| config | prose t/s (median) | code t/s (median) | speedup vs base |
|---|---|---|---|
| baseline (no draft) | 47.8 | 47.6 | 1.00× |
| draft `--spec-draft-n-max 2` | 47.7 | — | 1.00× |
| draft `--spec-draft-n-max 8` | 47.7 | 47.5 | 1.00× |
| draft `--spec-draft-n-max 16` | 47.2 | — | 0.99× |

llama.cpp's speculative decoding is a **wash on this box** — not a
regression of their implementation but the same physics doc 80 projected:
with C_T(nt) near the weight-streaming floor (their batched verify is
efficient), the win is capped at ~1.04–1.2×, and the per-drafted-token
draft-model cost eats all of it at measured acceptance p≈0.69 (long drafts
slightly negative). Two conclusions:

- minfer's D5 negative is not an engine defect of ambition — the strategy
  itself has no headroom on GB10 at these costs; the external reference
  confirms doc 80's ceiling from the outside.
- The minfer-specific finding stands unchanged: llama.cpp loses only the
  ~0% residual, while minfer's batched path would *lose 2×* before drafting
  even starts (the 0.52× graph-level gate) — the dispatch fix remains
  worthwhile for any future multi-token feature, just not for D5.

The llama.cpp-side mechanism behind the reference numbers — the quantized
`mul_mat` dispatch chain (MMVQ for ne11 ≤ 8 with tokens-in-registers, MMQ
for ne11 ≥ 9 with M-tiles floored at 8; "M never enters the grid as a
weight-multiplier dimension") and the point-by-point contrast with minfer's
`grid(od/4, nt)` hole — is documented in
[`LLAMA-CPP-MMQ-ANALYSIS.md` §12](../LLAMA-CPP-MMQ-ANALYSIS.md).

### 4.2 14B replication + post-fix gate re-read (2026-09-12, after doc 82)

The external battery was replicated on the 14B pair after doc 82 restored
the batching invariant (Qwen2.5-14B-Instruct Q4_K_M + 0.5B q4_0 draft, same
protocol as §4.1, 3 interleaved reps; a root-owned sglang server was
co-resident but idle — `--sleep-on-idle`, 44 GB reserved, no compute in the
window):

| config | prose t/s (median) | code t/s (median) | speedup vs base |
|---|---|---|---|
| baseline (no draft) | 23.9 | 24.0 | 1.00× |
| draft `--spec-draft-n-max 2` | 23.9 | 24.1 | 1.002× |
| draft `--spec-draft-n-max 8` | 23.8 | 23.7 | 0.992× |
| draft `--spec-draft-n-max 16` | 24.2 | 24.0 | 1.006× |

llama.cpp's speculative decoding is a wash on the 14B too — the doc 80
ceiling argument is target-scale-invariant on GB10.

minfer's verify gate re-measured post-doc-82 (`specverify -p 512 -r 40`,
median of both passes): C_T(1) = 42.15 ms, C_T(3) = 59.14 ms, C_T(8) =
91.56 ms → amortization(nt=3) = 2.14× (the pre-fix 7B shapes measured
0.52×), amortization(nt=8) = 3.68×, marginal cost ≈ 7.1 ms/token — weight
traffic is nt-independent at 14B as well. The pre-registered gate still
fails (2.14 < 2.5): the residual per-token marginal is attention +
q8-quantize + dp4a compute, exactly the doc 82 §5 cost model. D5 stays
closed, now with external confirmation at a second target scale.

Instrument errata (fixed this session): the `specverify` JSON's
`amortization_nt3`/`amortization_nt5` fields shipped computing
`C_T(1)/C_T(nt)` — missing the `nt` factor of the §2 definition, which made
the reported metric unsatisfiable by construction (≤ 1.0). The doc 81/82
tables were computed from the raw medians with the correct
`nt·C_T(1)/C_T(nt)` formula and are unaffected; raw medians are unchanged
by the fix.

## 5. Lessons

- **The gate did its job — 90 minutes of measurement against days of
  plumbing.** The conditional-go from doc 80 was honest precisely because it
  named the one number that could kill it; that number killed it.
- **Kernel micro-benchmarks are not engine facts.** The entire D5 bet rested
  on a 2.7× that lived inside one GEMM kernel. The graph-level truth is
  0.52× — a 5× divergence between the level where the number was measured
  and the level where it was needed. Future campaign anchors must be
  measured at the level they will be consumed.
- **A batched path that re-streams weights per row is worse than no batching
  at all.** minfer's nt>1 GEMM dispatch (built for prefill shapes) silently
  degrades to per-row weight streaming at small M — 1.9× worse per token
  than the nt=1 MMVQ path. Any future multi-token decode feature (batched
  verify, beam search, parallel sampling) inherits this cliff; fixing it
  means an M≥16-or-nothing shape gate in the dispatch.
- **GB10 has no lockable clocks** (`nvidia-smi -lgc` rejects) — medians plus
  a reversed second pass remain the drift defense, and spread stayed ±2%
  across 40-rep phases.

Artifacts: `/tmp/d51a/*.json` (ephemeral; key numbers inlined above).

← 80 · [Index](./README.md)
