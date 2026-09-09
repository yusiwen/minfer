# 24 · r19: Weight L2 residency — `__ldg` imperceptible, persisting window catastrophic (REVERTED)

> **Result**: both levers built in the same round. The `__ldg` read-only path: neutral (KD=4 median +2.0%, KD=8
> sunk in noise) — the weight tiles were already being re-read from L2, and L2 is not the constraint (SOL
> 37–46%). The `MINFER_MMQ_L2WIN=1` persisting window (hitRatio 1.0):
> **−50%, 6/6 consistent** — the 12.8–34 MB of resident rows per weight squeeze the C stores (37 MB per GEMM),
> activations and KV out of the normal L2, and every re-mark churns the carveout.
> The bar ≥ +5% decisively missed; both levers reverted.
> **Commit**: `072dd9a` (record commit; the code left no separate commit with the revert — the originally cited revert
> anchor HEAD `384b3d9` is an unresolvable pre-amend twin, see the master-table footnote-3 pattern).
> **Date**: 2026-09-03.

## 1. Background — where things stood

r19 shares r18's day and session, is item 2 of the Phase-7 tasks (task-2), and steps directly on
r18's autopsy conclusion. r18 had just proven: cutting the B-expansion ALU buys nothing, with SM%
motionlessly stuck at 30–34, "the latency binding the kernel is elsewhere". But r18's revert record left
one door open — the EB/SB materialization machinery was fully preserved, **"as the precursor for any future L2-residency
experiment"**. r19 puts that L2 direction directly on the table: without waiting for the pre-expanded planes, ask on the raw
weight byte span itself — **when the same weight bytes are re-read over and over, do they actually come from
DRAM or L2? Can they be forced to stay resident in L2?**

The problem's magnitude deserves computing first. The wide kernel's tile is 128 tokens × 128 od; the B (weight)
panel is blocked by od column, and every x-block (token direction) must re-read the **same** B panel completely
once more. At nt=2630 the x-block count = ⌈2630/128⌉ = 21 — **the same weight bytes are
re-read 21× within a single launch**; adding the cross-launch re-reads within a layer, the total repeated access to
weight bytes is substantial.

The campaign history had in fact probed this question sideways twice:

- r7–r8's x-tile lesson said "B-traffic reduction is a dead lever in the face of L2-resident re-reads"
  (L2 absorbed the B re-reads);
- r13's counter forensics measured our per-MAC L2 read traffic at only 1.6× llama.cpp's
  — far below the 1.67× instruction-count gap.

But both are indirect evidence. r19 wanted direct intervention: use CUDA's two levels of L2 control
(the read-only path hint, the persisting access policy window) to try turning "weight bytes resident in L2" from
accident into policy.

The session environment remained harsh: box absolute values ran ~9% below the r15/r17 sessions, and the day drifted violently
(the KD=8 baseline's three interleaved runs gave 780.0 / 1061.7 / 1252.4 — a ±25% spread). This sets
r19's metrological situation: **only signals whose effect size exceeds the ±25% noise band are judgeable**. In hindsight
this constraint mattered enormously — it is what made "−50%" the only judgeable signal in the whole round.

## 2. Principle — the GPU mechanism

The two levers' mechanisms are entirely different; audit them separately.

### 2.1 Lever one: `__ldg` (the read-only data path)

`__ldg` routes a load through the non-coherent read-only path, and on that basis the compiler can emit
`LDG.E.CI`-class instructions and cache the data on the path optimized for read-only data. It matters for performance
only under two premises:

1. The data really is read-only (r19's targets — the weight bytes within the wide kernel's lifetime — satisfy this);
2. **The compiler didn't already know it**.

The latter is key: the wide kernel's weight pointers are declared `const __restrict__`, and nvcc for such
pointers already tends to emit non-coherent loads — if it already does, `__ldg` is a pure
no-op. Also, it only affects cache-path selection and **changes no byte counts**: if the bottleneck is not
the source of the weight bytes (DRAM vs L2) at all, the hint has nothing to push on.

### 2.2 Lever two: the L2 persisting window (`cudaAccessPolicyWindow`)

This is CUDA's L2 residency mechanism. Set an access policy window on a stream:

| Parameter | r19's value | Semantics |
|---|---|---|
| `base_ptr` / `num_bytes` | the raw q4_K weight byte span (12.8–34 MB scale) | the address range the window covers |
| `hitRatio` | **1.0** | accesses inside the window are marked hitProp with this probability |
| `hitProp` | `Persisting` | hit lines can only be evicted by other persisting accesses or an explicit reset |
| `missProp` | `Streaming` | unmarked accesses take the streaming (low-priority residency) path |

Alongside, `cudaLimitPersistingL2CacheSize` (the persisting carveout) must be raised to
its maximum — r19 does this once at per-process init; the window is set before each wide-kernel launch.

### 2.3 What hitRatio 1.0 does, in arithmetic

The three L2 working sets of a single GEMM:

- Weights (the object the window marks): 12.8–34 MB per weight; hitRatio 1.0 = **every line
  request gets the residency mark**;
- C output: nt × od × 4 B; at nt=2630, od=3584 that is ≈ **37 MB/GEMM** of write
  traffic, on the Normal path;
- A activations and KV traffic: smaller than C, but competing on the Normal side all the same.

The carveout is raised to the driver-allowed ceiling (`persistingL2CacheMaxSize`), while the resident set the window demands
is the same size as the carveout or larger — the result has two layers.

**Layer one: capacity hijacking.** A fixed slice of L2 is monopolized by weight lines; the C stores'
37 MB, activations, KV can only circle in the remaining normal L2, and the miss rate climbs.

**Layer two (more hidden): re-mark churn.** When the window-marked weight set exceeds the
carveout, newly accessed weight lines must evict old persisting lines to become persisting
themselves — "residency" degenerates into a high-frequency mark/evict cycle, every line shuttling in and out of the
carveout, with L2 control-path metadata overhead stacked on top. The mechanism itself is deterministic — it deterministically does
the wrong thing.

### 2.4 The premise both levers shared and should have checked first

**Is this kernel's L2 actually tight?** ncu's Speed-of-Light reading had been sitting there all along:
L2 Cache Throughput 37–46% SOL. A resource below half saturation cannot be the source of a −50% or +5%
no matter how its residency policy is optimized. Half of r19's value is measuring this
veto; the other half is nailing the "read SOL first, then act" ordering into the methodology.

## 3. Implementation

> **Forensics note**: r19's code vanished with the revert (the variant survives in artifacts like `/tmp/patch_p7_t2.py` and
> `/tmp/cuda_kernels_p7t2.cu`, which do not travel with the repo). This section's code excerpts are the
> **target load sites** these two levers aimed at on the
> current tree (`src/cuda_kernels.cu` wide kernel
> `RAW_STAGE` macro); the levers themselves are reconstructed from the record's narration — their form is only a few lines of API calls,
> which the narration reproduces exactly.

### 3.1 Design choices (why this shape and not another)

**The read-only hint hits only the two weight-side load classes.** Weights are read-only within the kernel's lifetime; the activation
side is not — `__ldg`'s targets are limited to: (a) the B-staging super-block `uint4`
bulk reads (the mainstay of the 144 B raw rows), (b) the scale-header `uint16` reads (the d/dmin
f16 headers). This is parity-neutral by construction (a cache-path hint changes no values).

**The window targets the raw weight span, set before every launch.** The set being re-read 21× is the
raw W byte span itself, so the window should aim at it; `cudaAccessPolicyWindow` is a stream
(stream) attribute, not a process attribute, so the set point is before each wide-kernel launch — which incidentally guarantees
only the MMQ wide kernel's launches are affected and no other work on the stream is hit by mistake. The carveout raise
is a per-process `cudaDeviceSetLimit`.

**hitRatio 1.0 first is hypothesis testing, not engineering tuning.** r19's question structure is
"does weight residency in L2 help at all" — the cleanest experiment pushes the hypothesis to the extreme (the whole weight set
resident), and only if significantly positive comes back to selective variants; if significantly negative, the whole family closes. In hindsight, the +5%
bar combined with the ±25% noise band means: **intermediate variants (hitRatio ~0.25) are simply unjudgeable in this
noise environment**; the extreme test is actually the only metrologically meaningful first step.

**The env-var gate `MINFER_MMQ_L2WIN=1`.** The window mechanism has global side effects (the carveout
is a process-level resource), so it must default off and be enabled explicitly — which also lets A/B toggle via
env var on the same binary.

### 3.2 Key code

**Lever one's aiming point** — the weight-side loads in the wide kernel's staging (current tree
`src/cuda_kernels.cu`; r19 adds `__ldg` to these two read classes):

```cuda
/* B-staging superblock bulk read (__ldg target a): */
if (j < od && sb < nsb) {
    const uint8_t* src =
        W + (size_t)j * ((size_t)nsb * 144)      // weight row stride 144 B/superblock
          + (size_t)sb * 144 + 16 + p * 32;      // +16 skips the f16 d/dmin header
    v0 = *(const uint4*)(src);                   // → __ldg(const uint4*)
    v1 = *(const uint4*)(src + 16);
}

/* scale-header read (__ldg target b): */
float d    = h2f(*(const uint16_t*)blk);         // → __ldg(const uint16_t*)
float dmin = h2f(*(const uint16_t*)(blk + 2));
```

**Lever two itself** (reconstructed per the record; the whole logic is these few lines):

```cuda
// once per process (at init):
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                   persistingL2CacheMaxSize);
// before each wide-kernel launch (when MINFER_MMQ_L2WIN=1):
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr  = /* base address of this weight's raw byte span */;
attr.accessPolicyWindow.num_bytes = /* od * nsb * 144 (12.8–34 MB scale) */;
attr.accessPolicyWindow.hitRatio  = 1.0f;        // ← the disaster parameter, proven after the fact
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                       &attr);
```

### 3.3 Pitfalls

- **Parity passed in one shot, on both sides** — cache hints and access policy windows change no values;
  window-on and window-off at both KD=4/KD=8 depths all green. r19 has no correctness story, only a performance story.
- **The noise band itself is this doc's biggest pit**: the KD=8 baseline's three interleaved runs
  780.0 / 1061.7 / 1252.4 make a median meaningless on its own; `__ldg`'s +2% (KD=4)
  is below the judgeable threshold in this environment. Lesson: a lever whose effect size is smaller than the session noise band
  cannot be judged even if measured — estimate effect sizes first, then order the experiments.
- **"The mechanism works" and "the mechanism helps" are two different things**: the window delivered a 6/6-consistent −50%
  — extremely deterministic. Without the same-session three-arm interleave (baseline / `__ldg` /
  `__ldg`+L2WIN), a deterministic large negative effect could have been misread as an environment problem.

## 4. Verification

- **Parity two-way gate (window on/off × KD=4/KD=8)** — defends against "a cache-policy change quietly moved the
  values": an access policy window should not affect bits in theory; that must be proven by measurement.
- **Same-session three-arm 3× interleaved A/B (baseline / `__ldg` / `__ldg`+L2WIN)** — defends against
  machine drift and single-arm hallucination; the ±25% noise band makes any effect smaller than it unjudgeable, and makes the −50%
  conclusion unusually solid.
- **6/6 consistency check** (−50% reproduced in all three repeats of both the KD=4 and KD=8 groups)
  — a large negative effect's credibility comes not from its magnitude but from reproduction consistency.
- **ncu SOL reading (L2 Cache Throughput 37–46%)** — provides the mechanism explanation for `__ldg`'s
  neutrality: the resource is unsaturated, so optimizing its residency is moot.

## 5. Results

**Wall clock** (7B @2630 tok, same-session 3× interleaved; that day's box noise was extreme, the KD=8 baseline
spanning ±25%):

| Config | baseline (1e0dded state) | r19 `__ldg` | r19 `__ldg` + L2WIN |
|---|---|---|---|
| wide KD=8 | 780.0 / 1061.7 / 1252.4 | 700.7 / 1278.6 / 1282.3 | **453.6 / 535.4 / 548.2 (−50%)** |
| wide KD=4 | 1195.8 / 1144.4 / 1240.0 | 1201.9 / 1219.7 / 1229.9 (+2.0% median) | **557.6 / 575.4 / 577.2 (−50%)** |

**Verdict**: bar ≥ +5% (relative to the task-1 state); the only signal in the whole round exceeding the noise band is
−50%. `__ldg` neutral (+2.0% KD=4 median, no consistent KD=8 gain); the persisting
window catastrophic and 6/6 consistent. Both levers reverted to the then-HEAD (cmp-verified).

**Veto mechanism** (each lever stands independently):

1. `__ldg`: the weight tiles were already being re-read from L2 (r13: per-MAC L2 read traffic only
   1.6× llama's), and L2 throughput 37–46% SOL says it is not the constraint; moreover under
   `const __restrict__` nvcc most likely already emitted non-coherent loads — a "read-only hint"
   is literally a no-op once the compiler already holds the information.
2. L2WIN: the mechanism works completely (deterministic 6/6), but with the hitRatio 1.0 + weight set ≥
   carveout parameter combination the direction is negative — the resident set eats the carveout, the C stores'
   37 MB/activations/KV are squeezed into the remaining normal L2, and every re-mark churns the carveout.

**Under what future conditions a retry is worthwhile**: the record names three untested cheap variants (the budget
was exhausted at the time) —

- (a) a selective window at hitRatio ~0.25 (resident only some lines, leaving L2 for normal traffic);
- (b) windowing only small weights (q/k/v-type weights whose sets ≤ carveout, churn-free objects);
- (c) per-layer `cudaCtxResetPersistingL2Cache` (preventing cross-layer mark accumulation).

The shared higher-level premise: **first find a domain where L2 really is the constraint** (shapes/kernels with SOL near
saturation); otherwise every L2-residency optimization is optimizing a bottleneck that does not exist. r18's preserved
EB/SB machinery and this doc's named variant (b) are a natural pair — "pre-expanded plane + selective
residency" remains an unclosed road.

## 6. Lessons

1. **A "read-only hint" is a no-op once the compiler already knows**: `const __restrict__` pointers
   most likely already take non-coherent loads — before adding the hint, confirm what information it changes.
2. **Read SOL before ordering levers**: the L2 37–46% reading could have killed this direction before any code was
   written — optimizing an unsaturated resource is a zero-sum game of luck.
3. **A persisting window needs selectivity**: hitRatio 1.0 applied to a working set ≥ carveout
   turns a shared cache into a churning private region — the mechanism works
   deterministically, including deterministically biting back.
4. **A lever whose effect size is below the noise band is unjudgeable**: ±25% session noise makes a +2% "micro-win"
   meaningless — experiment ordering should do the effect-size arithmetic first and put the judgeable extreme hypotheses up front.

---

← 23-r18-load-time-b-preexpansion · [Index](./README.md) · 25-r20-split-phase-a-staging →
