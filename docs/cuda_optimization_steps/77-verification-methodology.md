# 77 · Verification Methodology & Transferable Lessons (the campaign's master gate system)

> **Result**: 12 sessions and ~60 optimization levers, every landing/veto adjudicated by the same gate chain — 3 parity tests + greedy byte-for-byte identity + interleaved A/B medians + the suite + the ncu/nsys evidence protocol.
> **Commit**: no repo change (this doc expands `docs/CUDA_OPTIMIZATION.md` Appendix B).
> **Date**: 2026-08-30 → 2026-09-09.

## 1. Background — why a fixed gate set is needed

This campaign had one recurring scenario: an optimization runs fast and correct in
isolated kernel tests, but end to end it is either a **phantom gain** (the fast path
is never actually taken) or **correctness erosion** (a floating-point summation order
changed; the parity simulator cannot see it, but the sampling chain can amplify it
into divergence).

The matrix: kernel-level correct ≠ end-to-end correct; fast ≠ truly fast (baseline
drift, co-tenant interference, silent fallbacks). Without a fixed gate chain, every
lever would invent its verification on the spot, and a method invented on the spot
is precisely blind to the defect class you most need it for.

So from R1 onward, every lever shares the same gate chain, each gate defending one
defect class. This doc first covers the gate chain itself, then the tool protocol
(the GB10 specifics of ncu/nsys/SASS), and finally collects the transferable rules
the campaign sedimented.

## 2. Principle — the gate chain and the defect classes it defends

### 2.1 Gate 1: the parity trio (numeric correctness)

Three independent calls before every landing:

| test | form | defends |
|---|---|---|
| `cuda_prefill_mmq` | 1/0, 8 quant types × 8 shapes swept against the host reference | the quant kernels' numeric path |
| `cuda_prefill` | 7/0 | the prefill graph end to end |
| `cuda_fa_prefill_attention_parity` | 1/0 | the attention kernel |

The 1e-3 tolerance is **informative**: a nibble-layout error (misaligned unpacking)
shows up as a ~1e0-magnitude deviation, while legitimate f32 rounding noise is only
~1e-5. So the 1e-3 tolerance is not "lowering the bar" — it is a discriminative
window that separates bug classes from noise classes.

### 2.2 Gate 2: greedy byte-for-byte identity (end-to-end correctness)

`-n 32 --greedy --seed 42` against prompt2k on the pre-change binary; the token
stream must be byte-for-byte identical.

It defends against **graph-level/memory-level corruption** the parity fixtures miss
— two real cases: r52's rms kernel out-of-bounds write (OOB), r58's smem buffer-1
cross-write. Such defects may happen to be invisible in kernel-output tests, but
once any layer is polluted the whole generation sequence necessarily diverges.

**The FA exception** (important): a tile-size change necessarily changes the
grouping order of floating-point accumulation (the r50/r57 lesson); even with
parity all green, greedy will diverge at some step. For changes of this "naturally
breaks byte-for-byte identity" class, the gate is swapped for a whole **calibrated
tolerance package** (see §2D D3a/D3-6: kernel-level vs CPU ≤1e-4 on realistic
outlier data, the argmax hard gate, the rp=1.0 greedy identity stream + sampler
knife-edge attribution).

### 2.3 Gate 3: interleaved A/B medians (performance truth)

3×/5× same-window pairing, warmup, alternating order; **headline numbers require
distribution separation** (min-new > max-base); the whole-prefill landing bar is
+1.5% (relative to the re-measured baseline, calibrated at r24).

The alternating order is the key design: the GPU is shared (sglang is co-tenanted
on this machine); running one side back-to-back disguises window drift as a trend.
Paired medians cancel the co-tenant noise.

### 2.4 Gate 4: the suite and co-tenant flakes

The suite grew from 166/0/3 at the campaign's start to 174/0/3 (gate 1's byte-level
tests were progressively promoted to permanent tests). Tests that flake occasionally
under a co-tenanted window are adjudicated with an isolated `--exact` rerun —
**a flaky test is either proven isolated or fixed, never silently retried to green**.

### 2.5 Gate 5: the ncu/nsys/SASS protocol (GB10 specifics)

This GB10 (DGX Spark, sm_121) has several unavoidable tool pitfalls:

- **ncu must go through `sudo -n env LD_LIBRARY_PATH=...`**: plain sudo strips
  environment variables, and ncu silently profiles the legacy path (the r56
  lesson — you think you are measuring the new kernel but are measuring the old one);
- **GB10/GB20B has no `dram__*`/`launch__grid_size`/shared-sector counters** —
  bandwidth rooflines can only be derived from `lts__t_sectors_aperture_device`
  (L2 sector count × 32 B) plus byte counts (established at r55);
- **ncu serializes replay; nsys is the wall-clock authority**: ncu's per-kernel
  times are distorted under serialization and are used only for structural metrics
  like occupancy/sectors/residency;
- **PC-sampling's attribution rule**: `--page source` attributes a stall to the
  **consumer instruction waiting on it**, not the instruction producing the bytes
  (r20/r43) — do not read the causality backwards;
- **SASS first**: read `cuobjdump -sass` before writing any lever — cp.async in
  SASS is `LDGSTS.E.BYPASS.128` (grep LDGSTS to verify the compiler actually
  emitted it, r45); ptxas `-Xptxas -v` reports the register/spill/occupancy
  budget (r40 used it to confirm the 3rd resident block).

## 3. Implementation — the transferable rules that sedimented out

Every rule below was paid for with real campaign losses; the source is in
parentheses.

**Baseline anchoring (r59b)**: every A/B baseline must be behaviorally anchored in
the same window — re-measure a known binary with a historical record, or rebuild
the baseline commit from a worktree. Idle co-tenancy = clean equivalence; without
anchoring you may not attribute performance fluctuation to the co-tenant tax.
One of r59's +Δs was proven by r59b to be a polluted baseline; the clean
re-measurement corrected the number.

**Liveness checks (r53/r54)**: an optimization with "a fallback as safety net"
must have a counter/label for "did the fast path actually go live" — neither
parity nor greedy can see a fast path that is **never taken**. Distinguish an
intentional fallback (`exp=off`) from an accidental one (`fallback!`).

**Tile size vs greedy identity (r50/r57)**: for kernels sensitive to accumulation
order, strict byte-for-byte identity is satisfiable only for changes that "preserve
the accumulation order" — once the tile size changes, ULP regrouping necessarily
happens, and green parity is not enough. This rule directly spawned D3a's
calibrated tolerance package.

**The pipeline value formula (r58, the mirror of r45)**: a staging mechanism's
wall-clock value = what it removes − its granularity cost. Replacing expensive
work (q6_K r39/r53/r56) → +13/+5/+2.35%; replacing cheap copies (q4_K r58) →
−12.6%. "Not dead, waiting for its scenario" is valid only while the mechanism's
cost model holds.

**Roofline before code (r55)**: derive the byte-traffic lower bound first (when
there is no `dram__*`, use sector count × 32 B); if even a perfect kernel cannot
pass the bar, skip the implementation. D4-2's Lever A (the llama L2 prefetch port)
was this rule's pure-inference veto — closed without writing a single line of code.

**Buy occupancy first, then save instructions (r13→r25→r28/r29)**: at 1 block/SM
an instruction surplus is real but wall-clock inert; buy the occupancy first and
the same instruction reduction starts paying (+2.6/+2.8%). At 3 blocks/SM, 4 B of
spill is irrelevant (r40).

**The compiler already did it (r30/r32/r33)**: read the SASS before writing a
lever — if ptxas has already scheduled a source-level rearrangement, changing it
is a SASS-level no-op. r33's "line-by-line port" of the llama inner loop died
exactly here: the SASS was completely identical, so of course the performance was too.

**Stall mass conservation (r20/r21)**: fix one bottleneck and the stall moves to
the next (latency → lg_throttle → wait) — strike in stages in that order; do not
expect one fatal blow.

**Layout-transform locality (r34)**: promote the layout transform to a prepass
(or fold it straight into the producer) instead of adapting inside every tile
consumer — moving the transform out of the kernel itself is worth +9.72%.

**Mechanisms compound across the wall (r53/r56)**: two levers individually
wall-clock-neutral (one reducing WORK, one reducing WAIT) compose almost additively
once the first one unblocks the bottleneck.

**Attribute to the consumer (r42/r43)**: the stall counter tells you the resource;
PC-sampling tells you the instruction **waiting on it** — cut the latency where it
is exposed, not where the bytes move.

**Wall decompositions expire (r37→r47)**: re-attribute the whole wall after each
line converges; the hidden tax (q6_K prepass's +31.6 ms) must be weighed against
its gain. D4-1 overturning D3-8's "matmul aggregation 2.9 ms" attribution is this
rule's decode edition.

**Phantom results (r8, P5·3, r52a)**: silent fallback / attribute-set failure /
OOM masking all produce "fast and wrong" timings and fake errors — guard trips
must report loudly, and whether the fast path fired must have evidence.

**Shared-machine etiquette (2026-08-31)**: a kernel OOM under pool exhaustion
kills **someone else's** workload (it happened once); bare allocation probes are
forbidden; check `free -g` before the suite; while sglang is serving, run only
single-process 7B-scale benches.

## 4. Verification

This doc is the verification system itself and has no independent object to
verify. The way it is "verified": across the 12 sessions, not one REVERTED
decision was ever proven afterwards to have been the wrong revert, and the single
defect that slipped through (D4-2 B0: 7B losing 13.5% of its down-proj compute)
is precisely this system's **blind-spot case** — when both sides of the A/B share
the same bug, the comparison is bit-identical. The rules added from it:
cross-binary comparison must use the `-n 1` first-step dump (token cascades
pollute all subsequent KV), and D3-6's sampler knife-edge attribution gate
(the `--repeat-penalty 1.0` identity stream = the clean kernel-numerics gate).

## 5. Results

The gate chain's final form evolved with the campaign:

- R1 (2026-08-31): parity ×3 + greedy-32 + interleaved A/B + suite 166 —
  the four-piece set finalized;
- r24 (2026-09-01): the +1.5% whole-prefill landing bar calibrated;
- r50/r57 (2026-09-05): the applicability boundary of byte-for-byte identity was
  delimited, spawning the calibrated tolerance package;
- r59b (2026-09-06): the baseline-anchoring rule established;
- D3a (2026-09-07): the decode tolerance gate package calibrated (the argmax
  hard gate, sampler knife-edge attribution, kernel-level ≤1e-4 on outlier data);
- D4-2 (2026-09-09): the `-n 1` first-step dump rule + closing the cross-binary
  comparison blind spot.

## 6. Lessons

1. **The gate chain's value is the matrix of defect classes it defends** — every
   gate can state "what I defend" in one sentence; a gate that cannot is ritual,
   not verification.
2. **All isolated "fast and correct" evidence is a suspect** — liveness, baseline,
   co-tenancy, fallback: only after these four suspects are eliminated one by one
   is a headline allowed.
3. **Magnitude discrimination matters more than tolerance values**: the 1e0 vs
   1e-3 vs 1e-5 deviation magnitudes directly identify the bug class.
4. **A revert is not a failure**: the campaign's 12 negative-result levers are
   each archived with their mechanism, and later sessions (e.g. D4-1's design
   doc) reused these veto evidences directly, saving at least three rounds of
   duplicate construction.
5. **On a shared machine no performance number is "naked"** — every conclusion is
   a difference after paired anchoring, not an absolute value.

---

← 76 · [Index](./README.md) →
