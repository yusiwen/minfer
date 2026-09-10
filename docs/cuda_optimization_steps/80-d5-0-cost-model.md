# 80 · D5-0 — speculative-decoding cost model: measured baseline, acceptance, and the d=2 gate (MEAS-ONLY)

> **Result**: draft-simple speculative decoding on GB10 is **conditionally
> viable in the d=2 regime only**, and the go/no-go now rests on one measurable
> number. Measured: 7B q4_k_m target 54.3 tok/s, 0.5B q4_0 draft 342.2 tok/s
> (CUDA) / 73.3 tok/s (CPU); real greedy acceptance of the 0.5B-on-7B pair
> (via llama.cpp `speculative-simple`) p ≈ **0.68–0.70** across prose and code
> prompts and both draft quants. Break-even requires per-position acceptance
> p\* = 0.73 (d=2) / 0.81 (d=4) / 0.90 (d=8) — measured p is below the line at
> every d **unless** the target's nt=d+1 verify batch achieves the BT-MMQ
> amortization. At d=2 the requirement is **≥ 2.5× amortization at nt=3**
> (the D4-1 anchor is 2.7× at nt=4); projected outcome at the anchor is
> 1.04–1.05×, ceiling ~1.2×. The CPU-draft cross-device variant is **dead by
> measurement** (73.3 vs 54.3 tok/s = 1.35×, no break-even at any p or d).
> **Commit**: docs only (measurement only; artifacts in `/tmp/d50/`,
> ephemeral, key numbers inlined here). **Date**: 2026-09-10.

## 1. Background — where things stood

The D4-4 session closed the decode-kernel campaign with the note that
"nt=4 → BT-MMQ 2.7× cheaper/step = the spec-decode foundation" (step doc 08's
0.14-wave collapse at M=1 is the same fact read from the other side). The D5
plan ([`SPECULATIVE-DECODING-PLAN.md`](../SPECULATIVE-DECODING-PLAN.md)) made
D5-0 the go/no-go gate: measure the cost model before building any engine
plumbing. Three questions had to be answered with numbers, not priors:

1. what does the draft model actually cost per token on this machine (same-GPU
   and cross-device)?
2. what acceptance rate does a real 0.5B-on-7B same-family pair sustain?
3. what does the target's verify batch have to cost for the round to win?

## 2. Principle — the GPU mechanism

One speculative round with draft length `d` and per-position conditional
acceptance `p`:

- **draft phase**: `d+1` serial `nt==1` decode steps on the draft model
  (latency-bound; on the same GPU they are pure added latency);
- **process re-eval**: the draft must ingest the accepted target tokens to
  sync its KV (llama's measured `≈ 2(1+d)` draft token-evaluations factor);
- **verification**: ONE target forward at `nt = d+1` — every row needs logits
  (`n_out = d+1`), and because the batch rides the decode graph's BT-MMQ
  path, its per-token cost is NOT `C_T(1)`: this is exactly the regime where
  D4-1 measured 2.7× per-token amortization at nt=4;
- **output**: `1 + E[a]` tokens, `E[a] = Σ_{i=1..d} p^i` for constant `p`.

Round cost ≈ `C_T(d+1) + 2(d+1)·C_D(1)`; break-even against the baseline
`1/C_T(1)` solves for the required `p` given measured `C_T`, `C_D` — or for
the required verify amortization given measured `p`. Both directions are used
below. The subtlety the gate turns on: **where between M=1 (0.14-wave
collapse) and M=4 (2.7×) does M=3 land?** Linear interpolation of the win
says ~2.28× (below the 2.5× requirement); tile-regime step behavior could
give the full 2.7× (M=3 and M=4 may hit the identical tile configuration).
Only a measured `C_T(3)` settles it.

## 3. Implementation

No minfer code changed. The measurement harness:

1. **minfer cost battery** (3 interleaved rounds, `-p 0 -n 128 -r 1`, medians,
   GPU idle at 0% util, backend engagement confirmed by the CPU-vs-CUDA
   spread below):

   ```bash
   ./target/release/minfer bench -p 0 -n 128 -r 1 -o json <gguf>          # CUDA
   MINFER_DISABLE_CUDA=1 ./target/release/minfer bench ... <gguf>          # CPU
   ```

2. **acceptance measurement** — reused llama.cpp's binary instead of building
   the minfer loop first (the acceptance rate is an engine-independent
   property of the model pair at greedy: draft argmax vs target argmax along
   the target's own trajectory):

   ```bash
   llama-speculative-simple -m <7B q4_k_m> -md <0.5B draft> \
     -ngl 99 -ngld 99 --spec-type draft-simple --spec-draft-n-max <d> \
     --temp 0 -n 256 -p <prompt>
   ```

3. **cost model** (`/tmp/d50/cost_model.py`, ephemeral): break-even solver in
   both directions + projection grid; key formulas inlined above and in §5.

## 4. Verification

- **Backend engagement**: 0.5B q4_0 reads 342.2 tok/s (CUDA) vs 73.3 (CPU,
  20-thread Grace) — the 4.7× spread confirms the CUDA path was measured.
- **Baseline cross-check**: llama-bench tg128 on the same 7B = 49.57 ± 0.09
  tok/s; minfer's 54.3 = **1.096×**, inside the campaign's post-r60 band
  (1.074×/1.018× were the D4-4 numbers) — the baseline is trustworthy.
- **Acceptance stability**: per-position conditional `p` back-solved from the
  aggregate rates is consistent across draft quant (q4_0 42.6% vs q4_k_m
  48.9% aggregate at d=4 → p ≈ 0.685 vs 0.68) and workload (prose 42.6% vs
  code 44.1% aggregate) — no cherry-picked regime.
- **Model**: the greedy acceptance measured on llama.cpp transfers to minfer
  because minfer's 7B greedy output already matches llama's (campaign gate)
  and the draft is the same family; residual engine differences affect the
  verify cost (modeled separately), not the acceptance.

## 5. Results

Measured medians (3 interleaved rounds, tg128, GB10, quiet):

| config | tok/s |
|---|---|
| 7B q4_k_m CUDA (target) | **54.3** (C_T(1) = 18.42 ms) |
| 0.5B q4_0 CUDA | **342.2** |
| 0.5B q4_0 CPU | 73.3 |
| 0.5B q4_k_m CUDA | 365.0 |
| 0.5B q5_k_m CUDA | 321.8 |

Measured acceptance (llama.cpp, greedy, 0.5B q4_0 draft unless noted):

| d | prompt | aggregate accept | conditional p |
|---|---|---|---|
| 2 | prose | 140/238 = 58.8% | ≈ 0.69 |
| 4 | prose | 164/385 = 42.6% | ≈ 0.685 |
| 8 | prose | 173/692 = 25.0% | ≈ 0.65 |
| 4 | prose (q4_k_m draft) | 172/352 = 48.9% | ≈ 0.68 |
| 4 | code | 83/188 = 44.1% | ≈ 0.69 |

Break-even (minfer measured costs, D4-1 anchor 2.7×/token at nt=4,
saturating; draft cost = 2(d+1) token-evals):

| d | required p\* | measured p | verdict at anchor |
|---|---|---|---|
| 2 | 0.73 | 0.68–0.69 | **marginal lose** → needs verify amortization ≥ **2.5×** at nt=3 |
| 4 | 0.81 | 0.685 | lose; required amortization ≥ 4.5× at nt=5 — implausible |
| 8 | 0.90 | 0.65 | lose decisively |

Projected d=2 wall-clock vs the 54.3 tok/s baseline, as a function of the
(measured-later) nt=3 verify amortization:

| amortization @nt=3 | p=0.68 | p=0.69 | p=0.75 (optimistic) |
|---|---|---|---|
| 2.0× | 0.87× | 0.88× | 0.94× |
| 2.7× (anchor) | **1.04×** | **1.05×** | 1.12× |
| 3.5× | 1.18× | 1.20× | 1.28× |

**Verdict**: conditional go, d=2 only. The single gate variable is minfer's
measured `C_T(3)` — the D5-1 phase is re-ordered to build the
`n_out = d+1` verify-batch primitive FIRST and measure it before any of the
loop plumbing. Stop rule: if measured nt=3 amortization < 2.5×, the campaign
stops after the primitive with the negative documented. Honest ceiling even
on success: **~1.05–1.2×** decode tok/s — the risk the plan pre-registered.

## 6. Lessons

- **Measure the gate variable with someone else's binary**: the acceptance
  rate — the number the whole gate hangs on — came from llama.cpp's
  `speculative-simple` in minutes, with zero minfer code. Building the loop
  first to "find out" would have been days for the same number.
- **The cross-device fallback died by measurement, not argument**: 73.3 vs
  54.3 tok/s (1.35×) kills the CPU-draft idea at any acceptance — an
  assumption ("draft several times faster") that survived until it met a
  number.
- **Interpolation is not measurement**: the linear interpolation of the BT-MMQ
  win (nt=3 ≈ 2.28×) sits BELOW the 2.5× requirement while the tile-regime
  step hypothesis puts nt=3 at the full 2.7×. The two hypotheses disagree
  about the entire fate of the campaign, and only the primitive measurement
  arbitrates. This is why D5-1's first deliverable is a number, not a feature.
- **Acceptance is workload-stable within a model family** (p ≈ 0.65–0.69
  across prose/code/draft-quant here) — but it decays hard with depth
  (aggregate 58.8% → 25.0% from d=2 to d=8), which is why deeper drafts lose
  twice: more draft steps AND a worse mix.

← 79 · [Index](./README.md)
