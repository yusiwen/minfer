# 95 · Adaptive Draft Depth (`--spec-draft-adaptive`)

**Status**: ✅ done (2026-09-15). All four throughput gates pass with margin, identity battery 4/4
byte-identical, suite 185 green. Commit: see git log.

## 1. Goal

Static speculative decoding fixes the draft depth `d` per run. Prose wants `d=2` (acceptance
collapses with depth), code wants more — but the best static depth was chosen per *cell*, not per
*workload*. This campaign adds an online controller that picks `d` per round from measured
per-depth acceptance and per-nt verify cost, with one hard constraint inherited from doc 94: the
output must stay **byte-identical to sequential decode**.

Pre-registered gates (user-approved): adaptive ≥ 95% of the best static per cell
(prose q4_0 ≥ 34.1, code q4_0 ≥ 41.7, prose q4_K_M ≥ 33.8, code q4_K_M ≥ 44.3 tok/s),
identity battery 4/4 with the controller on, suite green.

## 2. Design

Per round the engine computes

```
E(d)   = 1 + p1 + p1·p2 + … + p1···pd          (expected committed tokens)
cost(d) = V(d+1) + d · t_draft                  (verify round + draft loop, ms)
pick    = argmax_d E(d) / cost(d)
```

with three online estimators:

- **Per-depth acceptance** `p_j`: Laplace-smoothed beta mean (β(0.5, 1.5)) over observed rounds.
  An **unobserved depth inherits the depth-1 rate** — optimism that is exactly right for the flat
  acceptance curve of code-heavy text and merely optimistic for collapsing prose, so exploration
  is *emergent* (no probe schedule): a high depth-1 rate pulls unobserved depths up until real
  samples say otherwise.
- **Verify cost** `V(nt)`: min over the last 4 samples per nt. The min is the steady-state
  estimator for a deterministic kernel; an EWMA let capture/warm-up spikes poison the curve and
  lock the controller shallow (observed in the first prototype).
- **Draft cost** `t_draft`: same min-of-4 window.
- **Hysteresis**: a depth switch requires a 10% score advantage. Without it, 2–3 early unlucky
  deep rounds read a true-0.85 depth as 0.14 via the beta mean and the pick collapses to d=1 and
  starves that depth forever (observed on code q4_K_M: dmean 1.06, 33.8 tok/s).
- **Identity cap**: the controller's depth cap is `min(d_max, 7)`. d=8 → verify nt=9 crosses into
  the BT-MMQ GEMM family, which is *not* bitwise-identical to the MMVQ family (§4). A static
  `--spec-draft-n 8` stays available for max throughput and is documented as not identity-safe.

CLI: `--spec-draft-adaptive` (boolean). The cap defaults to 8 (= engine cap 7) unless
`--spec-draft-n` is set explicitly. Without the flag, behavior is unchanged (static `d`, default 2).

## 3. Controller evolution (measured mis-steps, kept for the record)

| Version | Failure mode | Fix |
|---|---|---|
| EWMA (α=0.15/0.2) + fixed 0.75 prior for unseen depths | Shallow lock-in on code: unseen deep depths trusted the 0.75 prior while the real rate was 0.82 flat; capture spikes poisoned V(d) | Optimism rule + min-of-4 cost window |
| Optimism + EWMA, probe every 16 rounds | Prose paid a deep probe every 16 decisions forever (−5..8% on −n 128) | Probe schedule deleted entirely — optimism makes exploration emergent |
| Beta means, no hysteresis | code q4_K_M collapsed to dmean 1.06 after 3 unlucky deep trials (33.8 tok/s) | 10% switch margin + optimism floor for < 8 trials |

Unit tests (`src/spec.rs`, 4): flat-high acceptance → cap; collapsing acceptance → d ≤ 2;
horizon clamp → d = min(cap, horizon); prohibitive observed deep-verify cost → backs off.

## 4. The identity boundary: verify nt ≤ 8 is bitwise, nt = 9 is not

The adaptive battery first failed at −n 200 (prose/p3/p4 diverged; code identical). Position-wise
token tracing (`MINFER_TOKEN_TRACE`, doc-93 methodology) found the first divergence at pos 143:
an **accepted** draft token whose verify-row argmax had margin 3.25 — not a near-tie flip, a
materially different logit row.

Layer bisection via the graph dump (extended to write **numbered logits files and both K and V
regions per layer**; new `GraphAllocator::copy_kv_to_cpu`) showed **all 48 layers' K and V at the
divergent position bitwise-identical** between the sequential and spec runs. So the entire hidden
state is bitwise-equal through every layer — the divergence is in the **transient lm_head logits**
of the verify forward only, which has no downstream KV write. That is consistent with exactly one
seam: verify nt=9 dispatches matmuls to the **BT-MMQ** path (`prefill_mmq`), whose accumulation
structure differs from the single/multi-MMVQ family used at nt ≤ 8. The BT QKV/FFN results happen
to be bitwise-stable (KV proof above); the lm_head row is tolerance-class.

The kernel-level nt sweeps in `cuda_multi_token_matmul_bitwise` and
`cuda_verify_attention_nt_invariance` were extended from the historical nt=3 probe to
**{3, 5, 8}** — all bitwise — which pins the safe boundary at **verify nt ≤ 8 (d ≤ 7)** and the
break exactly at nt=9. Doc 94's identity claim therefore holds *per kernel family*; the battery
now runs adaptive (which visits nt 2..8) as the standing check.

## 5. Measurement hygiene lessons

- **GPU exclusivity is a gate prerequisite.** The first clean-build measurement round ran while
  an unrelated `sglang::scheduler` held 47 GB and compute: adaptive read 13.3 tok/s (prose) with
  148 ms rounds, and one identity run hung past a 600 s timeout. `nvidia-smi` first, always.
- **A CLI bug silently invalidated the first depth sweep.** The `spec_d_max` resolution was gated
  on the adaptive flag, so every static `--spec-draft-n 3..7` run actually ran d=8; the sweep's
  "d=3..7 DIFFERS" was one data point (d=8) repeated six times. Fixed; the true sweep shows
  d=3..7 IDENTICAL, d=8 DIFFERS.
- **Depth switches are cheap.** `MINFER_REBUILD_TRACE` (new, env-gated) measures the single-slot
  `GraphCache` rebuild at **~1.0–1.4 ms** per switch (build+assign+alloc; CUDA capture follows the
  existing 3-run protocol). The multi-graph-cache idea was priced out — flapping costs < 1%.

## 6. Results (GB10, Qwen2.5-14B q4_K_M target, −n 128, greedy, two runs each)

| Cell | best static (doc 94) | adaptive (this doc) | gate | verdict |
|---|---|---|---|---|
| prose, q4_0 draft | 35.9 (d=2) | **36.0** (dmean 2.1–3.0) | ≥ 34.1 | ✅ ≥ static |
| code, q4_0 draft | 43.9 (d=8) | **46.5** (dmean 3.4–4.1) | ≥ 41.7 | ✅ +5.9% |
| prose, q4_K_M draft | 35.6 (d=2) | **35.6** (dmean 2.1) | ≥ 33.8 | ✅ = static |
| code, q4_K_M draft | 46.6 (d=8) | **48.4** (dmean 3.4–3.5) | ≥ 44.3 | ✅ +3.9% |

The controller did not merely track the best static depth — it found a better operating point
(dmean ≈ 3.5) that the static sweep had never measured, beating static d=8 on both code cells
while staying identity-safe. Identity battery (adaptive vs sequential, −n 200): prose / code /
Romeo-Juliet / TCP-UDP **4/4 byte-identical**. Suite: 185 passed, 0 failed (181 + 4 controller
tests).

## 7. Instruments added

- `MINFER_TOKEN_TRACE=<path>` — `pos<TAB>token_id` per committed token in both the sequential and
  spec paths (the aligned-comparison tool; found the trace's own off-by-one and then the real
  divergence).
- `MINFER_REBUILD_TRACE=1` — per-rebuild timing in `forward_cached`.
- `MINFER_GRAPH_DUMP` upgrades: numbered logits files (`logits_{tag}_{n}.f32`), both K and V
  regions per layer, `GraphAllocator::copy_kv_to_cpu`.

## 8. Deferred

- Making the nt=9 BT lm_head bitwise-equal to MMVQ (would unlock d=8 under the identity): priced
  into 战役 96's profile-first work — the per-kernel stall data decides whether it is worth it.
- Marginal-cost pick refinement (the total-score comparison is conservative about extending depth
  when the marginal verify cost per token keeps dropping with nt).
- 战役 97 (conversation/server integration) inherits doc 94's penalty-window contract and this
  campaign's identity cap.
