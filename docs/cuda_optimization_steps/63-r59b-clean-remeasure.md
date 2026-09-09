# 63 · r59b — clean re-measurement + baseline-contamination correction (measurement round)

> **Result**: finalized numbers 3590.8 vs 3232.0 = **+11.1%** (replacing r59's
> on-the-spot +26.2%; the co-tenant-tax attribution is voided); headline figure
> **~3581 tok/s = 1.080×** llama-bench 3323.29 @pp3314 — the campaign uses this
> pair from here on. No code changes.
> **Commit**: `074ca94` (docs). **Date**: 2026-09-06 (Session F wrap-up, same day).

## 1. Background — where things stood

After r59 landed the W_dsc plane it measured a +26.1/+26.2% interleaved
series in the co-tenant window, and explained "the baseline reads only
2836/2843, below the landed record of 3219.6" as a −12% co-tenant tax. Both
numbers were written into the master table at the time (row 73).

But +26.2% is an order of magnitude above every same-family lever the
campaign had landed (q6_K's W_dsc plane +2.35%, the W_exp bundle +5.03%),
and the direct ncu evidence for r59's mechanism change (staging-phase decode
ALU disappearing) is ffn_down −34.6%, gate/up −35% — kernel-level truths,
but extrapolating them across the whole prefill cannot support +26%. And the
"co-tenant tax" explanation was never independently verified: no anchor, no
tax. Session F paused for a dedicated measurement audit — this round wrote
not one line of engine code, yet rewrote row 73's Δ column and left the
campaign its most important protocol rule.

Two questions: (1) is this window trustworthy (does the co-tenant actually
tax anything)? (2) which code was r59's baseline binary actually built from?

## 2. Principle — the GPU mechanism: the instrument theory of A/B measurement

### 2.1 The delta is a quotient of two instruments

The same-window paired A/B reading is `new_base / base_base`. The campaign's
conventions guarantee "same window" (removing machine drift), but one
implicit premise was never checked: **that each binary really is the code it
claims to be**. The baseline binary is a measuring instrument — when it is
not built from the code you think it is, the numerator and denominator still
exist, but the value is fiction. And contamination is **multiplicative**: a
baseline deficient by −12.5% inflates every delta by ~1.125×, and the bigger
the change, the bigger the absolute inflation.

### 2.2 Behavioral anchoring: the instrument must be checked against a known answer

The only way to verify an instrument is to have it measure a **known answer**. Two anchors:

- **Recorded anchor**: some historical binary/config has published medians
  (e.g. the r58-era clean record 3219.6, llama-bench's 3324.42);
- **Rebuild anchor**: `git worktree` a clean rebuild of the baseline commit —
  bypassing /tmp-snapshot provenance entirely, generating the instrument
  straight from source.

A mismatch on either condemns the baseline — this is not new measurement, it is putting test weights on the instrument.

### 2.3 Why an idle co-tenant should theoretically collect no tax

The "co-tenant tax" intuition comes from resource contention (SM time, memory
bandwidth, L2 capacity). A 0%-util process that merely sits resident in memory
participates in none of these: its pages lie in DRAM, it occupies no SMs, no
bandwidth, and its L2 lines get evicted normally. The only theoretical tax is
a negligible physical term. So "an idle resident collecting a 12% tax" fails
mechanically — the correct suspect is the instrument, not the neighbor. r59b
turned this expectation into a measured conclusion (§4).

### 2.4 The three failure modes of measurement (the campaign's own history)

r59b's rules were not written in a vacuum — all three failure modes have priors in the campaign:

1. **Window drift**: the same machine drifts −9% to +38% between sessions (the
   r12–r25 era recorded in footnote 2). Countermeasure: same-window
   interleaved pairing — the two readings of each A/B round must be produced
   interleaved within one time window; absolute values across sessions are
   not comparable.
2. **Co-tenant load**: a **live** co-tenant is a real tax — r55's baseline
   sanity read 3144.4–3151.4 under a live co-tenant vs 3181 on a quiet
   machine. **An idle resident is not** (proven in §3.1). Countermeasure:
   distinguish "a neighbor holding bandwidth" from "a neighbor lying in
   memory".
3. **Binary drift**: this doc's protagonist — you think you are measuring
   A vs B, but you are actually measuring B vs C. The first two modes are
   covered by the interleaved-pairing protocol; this one had no defense
   before r59.

What the three modes share: all contaminate the delta **without producing any
anomalous signal** — readings are self-consistent, variance is normal, the
mechanism evidence (ncu/nsys) is all real. The only defense is proactive
anchoring.

## 3. Implementation: three audit rounds

### 3.1 Round 1: window validation (the co-tenant's innocence proof)

Two things on the machine carrying the 46 GB sglang co-tenant:

1. **Utilization sampling**: 15 samples, all 0% util — the co-tenant is an idle resident;
2. **Two independent anchors**:
   - llama-bench re-run: 3323.29 ± 3.08, **0.03%** from the clean-machine-era 3324.42;
   - the **known binary** behind the 3219.6 record re-measured: **3217.0** (−0.08%).

Both anchors calibrated → **an idle resident produces no measurable tax**. The
window is clean, the "co-tenant tax" hypothesis loses its footing, and
suspicion turns to the instrument.

### 3.2 Round 2: binary-drift test (the smoking gun)

r59's session baseline was a /tmp snapshot (`/tmp/minfer_pre_r59`), which per
the session record should have been "pre-r58 baseline code". Three "same
baseline code" binaries compared in the same window:

| Binary | Reading (tok/s) | Delta vs the healthy anchor 3217 |
|---|---:|---:|
| `/tmp/minfer_pre_r58` (r58 session's baseline snapshot) | 3217.0 | — (healthy) |
| `/tmp/minfer_pre_r59` (r59 session's baseline snapshot) | **2824.7** | **−12.2%** |
| Fresh worktree rebuild of the same commit | 3232.0 | +0.5% (healthy) |

`minfer_pre_r58` and `minfer_pre_r59` claim to be the same code yet differ by
−12.2% in behavior; the fresh rebuild is healthy. The conclusion is unique:
**the r59 session mistook the r58-delta binary (the "new" A/B build carrying
the −12.6% transplant) for its baseline snapshot**. The fingerprint matches:
−12.2% ≈ the −12.6% measured in r58's A/B — the snapshot contained that build.

### 3.3 Round 3: finalized measurement (all rebuilt anchors)

Both baseline and HEAD were cleanly rebuilt from source, interleaved in the
same window:

- **fresh HEAD rebuild**: 3590.8 median (all 10 runs within 3553.8–3591.5);
- **fresh baseline rebuild**: 3232.0;
- delta = **+11.1%** (corroborating the +11.5% on the "r58-era clean record"
  basis — self-consistent when the denominator is the 3219.6 record value);
- combined median (10 HEAD runs aggregated) **3580.7 → the headline ~3581 tok/s**;
- same-window vs-llama: 3590.8 / 3323.29 = **1.080×** (minfer ahead);
- the memory two-mode census reproduced incidentally: +1454 MB (57189 vs
  55735 MiB, minus the resident co-tenant set), matching r59's +1456 MB —
  the mechanism-change side evidence closes.

### 3.4 The multiplicative anatomy of the contamination

Decomposing r59's on-the-spot readings against the finalized numbers makes
the multiplicative contamination obvious:

| Quantity | r59 on-the-spot (contaminated) | r59b finalized (clean) |
|---|---:|---:|
| New build reading | 3574.7/3588.8 | 3590.8 (fresh HEAD rebuild) |
| Baseline reading | 2836.3/2843.2 | 3232.0 (fresh worktree rebuild) |
| delta | +26.1/+26.2% | **+11.1%** |

The numerator is nearly identical in both windows (3588.8 co-tenant vs
3590.8 clean — an idle neighbor is again harmless); **all the inflation comes
from the denominator**: contaminated 2843.2 vs healthy 3232.0 = −12.0%,
interlocking with the binary-drift test's −12.2% and the r58 transplant's A/B
delta −12.6%. A −12% denominator defect amplifies a true +11.1% into +26% —
**a delta is the product of the numerator's truth and the denominator's fiction**.

## 4. Verification (this audit's own gates)

- **Fingerprint match**: the contaminated −12.2% nearly coincides with the
  r58 transplant's A/B delta −12.6% — a verdict requires an independent
  source explaining the wrong value, not just another mystery.
- **Two anchors cross-confirming**: the llama-bench anchor (external program)
  and the record anchor (the campaign's own historical binary) matched
  independently at 0.03%/0.08% — the window conclusion transfers only when
  both instruments are healthy.
- **Rebuild reproduction**: the fresh-worktree baseline 3232.0 falls in the
  healthy band (3217–3232), ruling out "the code itself regressed".
- **Orthogonal-quantity cross-check**: the +1454 MB memory delta matches
  r59's +1456 MB — the correction overturns only the baseline, not the
  mechanism evidence (ncu/nsys kernel numbers are same-binary before/after
  differences, unaffected by contamination).

## 5. Results

- **row 73 corrected**: +26.2% → **+11.1%**; the "co-tenant tax −12%"
  attribution is voided (footnote 1 permanently marked). The master table's
  Perf column now only admits same-window anchored values. The r59 section's
  original text (including the reading series) is preserved with a CORRECTION
  note added in place — history is not erased; the correction is overlaid as
  a bound annotation so later readers see the full shape of the error.
- **campaign headline finalized**: 7B pp3314 **~3581 tok/s** (combined median
  3580.7), **1.080×** vs llama-bench 3323.29 @pp3314 — the "verified 1.080×
  path" crowned at r60 refers to this pair of numbers.
- **Protocol output** (row 74's one-line lesson): **every A/B baseline must
  first be behaviorally anchored in the same window** — re-measure a binary
  with a known record, or `git worktree`-rebuild the baseline commit — before
  the delta is trustworthy.
- **Corollary**: idle co-tenancy is equivalent to clean; **never infer a
  "co-tenant tax" without an anchor**. Most of the earlier r12–r25-era "box
  drift −9% to +38%" confusion would have been avoided by this one rule.

## 6. Lessons

1. **The baseline binary is a measuring instrument, not scenery** — calibrate
   before every A/B (re-measure a known-record binary, or rebuild in a
   worktree); three minutes once, saves an entire round of wrong attribution.
2. **Multiplicative contamination**: with the baseline deficient by −12.5%, a
   true +11.1% reads as +26% — the bigger a delta is hyped, the more suspect
   the denominator first.
3. **An idle neighbor collects no tax**: 0% util × 15 samples + two-anchor
   verification is permanent; when a window misbehaves, check the instrument
   before the neighbor.
4. **/tmp snapshots have no provenance**: a snapshot's filename carries no
   code identity; any cross-session binary must be behaviorally anchored or
   rebuilt outright before use.

---
← 62 · [Index](./README.md) · 64 →
