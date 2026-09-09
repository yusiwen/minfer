# 25 · r20: Split-phase A staging — attribute first, shoot second: the first hit (LANDED)

> **Result**: 7B whole-prefill KD=4 1230.4 → 1317.7 (**+7.1%**), KD=8
> 1275.8 → 1319.9 (**+3.5%**), 6/6 reproducible; ncu (matched nt=512 q-proj):
> long_scoreboard 6.22 → 2.92 (−53%), kernel duration 632.4 → 555 µs (−12.2%),
> warp instructions −16%; the freed stall share migrated to lg_throttle 0.33 → 2.38. Although
> the ≥1350 bar was not reached, it landed on the strength of the complete mechanism — the turning point where the campaign moved
> from "guessing levers" into "attribute first" mode.
> **Commit**: `5ac8917` (code + record in the same commit). **Date**: 2026-09-03.

## 1. Background — where things stood

2026-09-03 afternoon, the third lever round of the same day. The morning's r18 (B pre-expansion) and
r19 (L2 residency) had been reverted one after the other, plus the previous day's r17 (warp remap); the campaign was already holding
**four independent null results**, which together pointed at a judgment that kept being repeated but never named:
SM% stuck at 30–34, the kernel is stall-bound, "the latency binding the kernel is elsewhere". But
where "elsewhere" is, nobody knew.

r13 had given a coarse localization five rounds earlier ("duration tracks instruction count 1:1"),
but its stall table proved afterwards to be **doubly distorted** — it used the pre-r14 old kernel,
and compared misaligned launch points, nt=2630 vs llama's nt=512. The attribution infrastructure
itself was untrustworthy — that is the structural cause of the three nulls in a row before this: every lever was a shot fired on an
uncalibrated map.

This round's starting question was asked differently. llama.cpp's `mul_mat_q` vs minfer's
`mmq_raw_wide_nt_kernel` at **exactly the same occupancy** (2.00 active
warps/cyc/sched, 8 warps / 4 schedulers) have a per-scheduler issue rate of **0.42 vs
0.26**. Same occupancy, same tensor work (IMMA counts exactly equal), yet a 1.6× issue-rate gap
— meaning the gap is not in resource allocation but in **what the warps spend their cycles on**.

That question can be answered directly with warp-state sampling: the sampler periodically snapshots each resident warp's
PC and stall reason, turning "where did the cycles go" into a readable table. r20's session
flow thereby rewrote the campaign methodology: **do matched-shape attribution first, calibrate the stall
structure clearly, then decide what to move**. The matching conditions tightened to the same layer's q-proj GEMM, nt=512,
id=od=3584, taking the launch points on both sides and comparing them one by one (llama `mul_mat_q<12,128,0>` grid
(48,1,1) + fixup (48,4,1); minfer grid (4,28) = 112 blocks ≈ 2.33 waves).
Falsify first, localize second, fix third — that order itself is one of this doc's main outputs.

## 2. Principle — the GPU mechanism

### 2.1 First establish how to read the attribution

ncu's PC-sampling (warp-state sampling) records the state of resident
warps sampled per SM cycle: if a warp cannot issue because some instruction's dependency is not ready, the sample lands on
**that instruction's PC**, with the stall reason attached. How to read the common reasons:

| Stall reason | Meaning |
|---|---|
| `long_scoreboard` | waiting on a register produced by a long-latency instruction (global load) |
| `short_scoreboard` | same, but the producer is a short-latency instruction (smem, shared values) |
| `wait` | fixed-latency instructions' (most ALU) waiting slots |
| `barrier` | waiting for other warps at `bar.sync` |
| `not_selected` | eligible to issue but the scheduler picked another warp (a good sign — there is slack) |
| `lg_throttle` | LSU input queue/issue port saturated (request rate overloaded) |
| `math_pipe_throttle` | compute pipeline backpressure |

The critical reading (this doc's core lesson): **the sample lands on the "consumer" instruction — the one
waiting on the dependency — not on the "producer"**. When an `LDG` is followed by the `STS` depending on it, the warp stalls on
the `STS`, with reason `long_scoreboard`. Reading the table as "the store is slow" gets it entirely wrong;
the store is only where the load's latency surfaces.

### 2.2 The matched nt=512 comparison table

Per-issue-active warp ratios; llama taken from launch 0 of its 8 launches (launch 6
verified consistent):

| Metric | minfer `<KDR=4>` | minfer `<KDR=8>` | llama `<12,128>` |
|---|---:|---:|---:|
| duration q-proj (µs) | 632.4 | 609.6 | 263.6 |
| issue /cyc/sched | 0.16–0.26 | 0.20 | **0.42** |
| eligible /cyc | 0.22–0.28 | 0.28 | 0.64 |
| warps active /cyc | 2.00 | 2.00 | 2.00 |
| warp inst / tile (k) | 499 | 488 | 356 |
| IMMA mma-inst | 1,605,632 | 1,605,632 | 1,605,632 |
| **long_scoreboard** | **6.22** | 5.76 | **1.15** |
| wait | 0.63 | 0.57 | 0.58 |
| barrier | 0.26 | 0.17 | 0.19 |
| not_selected | 0.41 | 0.40 | 0.50 |
| lg_throttle | 0.33 | 0.60 | 0.09 |
| LDS bank conflicts | 3,211,264 | 3,211,264 | 42 |

This table rules out four candidate explanations in one stroke:

- **Barrier structure**: 4 barriers/256 k on both sides — the "their staging is less
  synchronized" hypothesis is false at the source level;
- **Tensor work**: IMMA 1,605,632 identical across all three;
- **Occupancy**: 2.00 warps/cyc the same;
- **Launch shape**: the stream-k vs 3-wave difference measured as a wash; llama's fixup
  +34 µs counted in does not change the conclusion.

The only remaining significant difference is long_scoreboard: 6.22 vs 1.15, **97%** of the total named-stall
difference — minfer's warps spend ~86% of their resident cycles on global load
latency, llama only ~24%.

### 2.3 Using PC sampling to pin long_scoreboard onto an instruction

The top stall site holds **16.7% of all** stalls: the **first
STS** in the A-qs staging loop. The mechanism taken apart:

The old staging loop interleaves "load → store"; after the compiler unrolls 4-deep into batches,
each batch = 4 `LDG` + 4 dependent `STS`. The warp issues 4 `LDG`s, and the immediately following
`STS_0` must wait for `LDG_0`'s registers to return — the A-activation read is an 8-sector scatter pattern
(40 B per token row: `d(2B) | qs(32B) | ssum(4B)`, 8 lanes each reading one 4 B
word, sector efficiency 62.5%), one round trip ~600 cycles. So every batch pays one full
memory latency, and batches serialize against each other.

Compute the per-thread batching: at KD=4 the A-qs side has 16 load words
(`128·KDR·8 / 256 thr` = KDR·4) → 4 batches of depth 4; each batch sleeps ~600 cycles with
an in-flight depth of only 4. The aggregate in-flight volume of 8 warps/SM cannot cover a 600-cycle latency, and the issue rate collapses to
0.16–0.26. **Each warp's MLP (memory-level parallelism, the number of in-flight memory requests)
depth is clamped to 4 by the dependency chain** — that is the microscopic composition of long_scoreboard 6.22.

### 2.4 The fix's shape is uniquely determined by the attribution result

Since what binds is "dependency scheduling" rather than bytes, instruction count, or barriers, the minimal fix is to split the
phases: first issue **all** staging global loads into a register array (one k-tile's worth of
deep, independent `LDG`s), then write smem in one go. Addresses, traffic, instruction counts **unchanged item by item** —
the only change is the dependency schedule: the warp now sleeps once per k-tile, at the first STS (by then it has
issued 16 in-flight loads, MLP depth 4×), instead of once per batch. This is the cleanest class of diff in experiment design:
any delta can only be attributed to the scheduling itself.

And one expectation that must be stated upfront: **this class of fix moves the constraint; it does not eliminate it**.
Once the latency is hidden, the constraint that ranked second surfaces — measured here it migrated to
`lg_throttle` (LSU queue saturation, 0.33 → 2.38): staging **latency** and **request
rate** are serially co-bound. r21 will prove this conservation with a "fix the request side only" experiment.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**Why not cp.async / double buffering?** The family history tried both: cp.async double
buffering (`784786d`) measured neutral — but that experiment's interpretation needs correcting today: it moved the same
bytes and kept the same "move while computing" interleaved structure, measuring the "prefetch depth" variable; the
attribution pointed at **dependency scheduling**. The split-phase plan introduces no new transfer mechanism, adds no smem,
adds no barriers — it is the only candidate that maps one-to-one onto the attribution result.

**Why a register array?** The old code actually had 4-deep unrolling too — the problem is that within each unrolled batch
loads and stores interleave, and batches serialize against each other on the first STS. The register array
`av[KDR*4]` splits "issuing loads" and "consuming loads" into two independent loops; the compiler naturally issues
the load loop as a whole and consumes it as a whole in the store loop; `_Pragma("unroll")` guarantees the enumeration is not
folded back into the interleaved form.

**Why the A side only?** The attribution's top site (16.7%) is in the A-qs loop; B-side staging
at KDR=4 has the restage-skip guard (two k-tiles share one super-block),
so its diluted stall share is an order of magnitude lower. Fix where the attribution points — this also lets the +7.1% gain
be booked cleanly to the A side.

**Why nt=512 for the matched capture?** Both before and after use the same launch point, the same shape
comparison (q-proj, id=od=3584), guaranteeing the longsb/duration before/after difference is not an artifact of shape drift —
r13's lesson (nt=2630 vs llama nt=512 double distortion) converts directly into
this round's experimental discipline.

### 3.2 Key code

First the layout context of the objects being operated on (current tree `src/cuda_kernels.cu` wide-kernel smem comment;
the two A-side blocks are r20's targets):

```cuda
//   qa8   [KDR][128] x 32B  chunk qs only (d/ssum in sda_q) ...
//   sda_q [KDR][128] x 8B   (d f16 | ssum i16) packed, uint2-tiling ...
```

**Before — the interleaved LDG→STS chain** (`git show 5ac8917` deletion side, the `RAW_STAGE`
macro's A section):

```cuda
/* Old: loads and stores interleave — every 4-deep batch pays one full
 * memory latency at its first STS (PC sampling: that STS alone held
 * 16.7% of all warp stalls). */
for (int x = threadIdx.x; x < MMQ_WBI * KDR * 8; x += blockDim.x) {
    int u = x % 8, r = (x / 8) % MMQ_WBI, kd = x / (8 * MMQ_WBI);
    int tok = i0 + r, c = (kt) * KDR + kd;
    unsigned v = 0;
    if (tok < nt && c < nchunk)
        v = *(const unsigned*)(q8x + ((size_t)tok * nb32 + c) * 40
                               + 4 + u * 4);                     // LDG (scattered 4B words)
    *(unsigned*)(qa8 + ((size_t)kd * MMQ_WBI + r) * 32 + u * 4) = v; // STS (dependent)
}
for (int x = threadIdx.x; x < MMQ_WBI * KDR; x += blockDim.x) {   // d/ssum isomorphic
    int r = x % MMQ_WBI, kd = x / MMQ_WBI;
    ...
    d16 = *(const unsigned short*)src;               // LDG.16
    ss   = (unsigned)(short)*(const int*)(src + 36); // LDG.32
    *(unsigned*)(sda_q + ...) = d16 | (ss << 16);    // STS (dependent)
}
```

**After — split-phase** (`5ac8917` addition side; the current tree keeps it to this day, r22 only rewrote
the store addresses' swizzle):

```cuda
/* r20: split-phase A staging. The old interleaved LDG->STS chains
 * stalled the warp at the first store of every 4-deep unroll batch
 * (PC-sampled: the leading STS held 16.7% of all warp stalls = one
 * full memory latency per batch, ~4 batches per kt). Issue ALL
 * global loads into registers first - one deep independent LDG batch
 * per warp per kt - then store to smem. Identical addresses, traffic
 * and instruction count; only the dependency schedule changes. */
{
    unsigned av[KDR * 4];          /* qs words: 128*KDR*8 / 256 thr */
    unsigned short dv[KDR / 2];    /* d f16 words: 128*KDR / 256 */
    unsigned sv[KDR / 2];          /* ssum words */
    _Pragma("unroll")
    for (int i = 0; i < KDR * 4; ++i) {                 // phase 1: issue loads only
        const int x = threadIdx.x + i * 256;
        const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),
                  kd = x >> 10;    /* x/(8*MMQ_WBI) */
        const int tok = i0 + r, c = (kt) * KDR + kd;
        unsigned v = 0;
        if (tok < nt && c < nchunk)
            v = *(const unsigned*)(q8x
                + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4);
        av[i] = v;                                      // independent batch of depth KDR*4
    }
    /* ... dv/sv isomorphic load loop ... */
    _Pragma("unroll")
    for (int i = 0; i < KDR * 4; ++i) {                 // phase 2: stores only
        const int x = threadIdx.x + i * 256;
        const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),
                  kd = x >> 10;
        *(unsigned*)(qa8 + ((size_t)kd * MMQ_WBI + r) * 32 + u * 4)
            = av[i];                                    // consume the already-arrived registers
    }
    /* ... dv|ss merged-word store loop ... */
}
```

The address enumeration between the two loops is identical item by item (the same linearization `x = threadIdx.x + i*256`),
the stores are still the same 4 B/8 B writes — anything outside the diff (B side, compute section,
barrier count) untouched.

### 3.3 Pitfalls

- **The attribution table's reading trap**: PC sampling records the stall on the STS, and the first instinct is "optimize
  the store" — e.g. widen the store, merge writes. Exactly backwards: the store is the victim;
  the producer chain (interleaved scheduling + scattered loads) is the pathology. This doc's fix touches not one
  store instruction's form.
- **r13's old stall table is not reusable**: pre-r14 kernel + nt=2630 vs nt=512
  launch-point misalignment, doubly distorted. Attribution must be redone on the **post-fix kernel, matched shape** —
  an old map is more dangerous than no map.
- **Register pressure is this plan's natural risk**: `av[16] + dv[2] + sv[2]` at KD=4
  costs roughly a dozen extra registers; the kernel already runs at 1 block/SM occupancy, and
  register spills would eat the gain directly. The post-landing ncu showed no spill growth
  (warp-inst −16% is a net reduction), so the risk was covered by measurement.
- **Hunt down where the "freed stalls" went**: after longsb −53%, without looking at lg_throttle it is easy to
  misread it as "53% of the same problem remains"; in fact the constraint migrated — the next round's lever
  (r21's merged uint4) was derived from exactly that reading.

## 4. Verification

- **parity green KD=4 + KD=8** — split-phase only reorders independent smem writes; the end state is byte-identical;
  the parity gate confirms the "schedule change" did not quietly become a "value change".
- **Greedy token stream byte-identical** — the end-to-end behavior gate, defending against a kernel-level diff amplifying over
  multi-step inference.
- **Suite 166/0/3** — the cross-shape/cross-model regression matrix, defending the long-tail shapes outside the matched shape
  (short prompts, nt boundaries) against enumeration-difference breakage.
- **6/6 interleaved A/B reproducible** — +7.1%/+3.5% must hold repeatedly on a box drifting ±9% that day to be bookable
  (the same measurement discipline as r18/r19).
- **Matched nt=512 before/after ncu (longsb 6.22→2.92, duration
  632.4→555 µs, warp-inst −16%, lg_throttle 0.33→2.38)** — the mechanism gate:
  the named stall must fall as predicted, and the freed share must have a findable destination; a mismatch in direction or
  magnitude means the attribution was wrong.
- **The exclusion list (barrier density, IMMA count, occupancy, launch shape, memory bytes all
  measured flat)** — the attribution's control group: without these "exclusions", "long_scoreboard is the
  carrier" is only a correlation.

## 5. Results

**Wall clock** (7B whole-prefill, same-session interleaved, 6/6 reproducible):

| Config | before | after | Δ |
|---|---|---|---|
| wide KD=4 | 1230.4 | **1317.7** | **+7.1%** |
| wide KD=8 | 1275.8 | **1319.9** | **+3.5%** |

**Kernel level** (matched nt=512 q-proj): duration 632.4 → 555 µs (−12.2%);
warp-inst −16%; long_scoreboard 6.22 → 2.92 (−53%; the ~86% resident share on global-load
latency falls sharply); lg_throttle 0.33 → 2.38 (the constraint migrated to the
request-rate side). Against llama's same-shape 263.6 µs (+34 µs fixup): the per-kernel
gap converges from 2.4× to ~2.0×.

**The verdict on the bar**: the session bar ≥1350 was not reached (1317.7 < 1350), but this step
landed. Unlike r18/r19's "miss the bar, revert", r20's evidence structure is complete:

1. The named stall was cut in half as predicted;
2. 6/6 reproducible;
3. The causal chain between mechanism and gain is closed;
4. The residual constraint (lg_throttle/request rate) was already picked up by the next experiment (r21).

The bar is a heuristic, not a mechanism — when the mechanism evidence is complete, the bar yields. This is also the prelude to the campaign's later
"+1.5% relative bar" replacing the absolute bar (formally calibrated in r24).

**Follow-ups directly connected to this doc**: the lg_throttle reading directly spawned r21 (merged block-linear
A staging, targeting sector efficiency and request count — reverted, proving stall-mass conservation);
the split-phase structure itself became the wide kernel's permanent shape; later r22 (swizzle) and r34
(prepass-ization) are both built on it.

## 6. Lessons

1. **Attribute first, shoot second**: the r17/r18/r19 triple-null were all guesses on uncalibrated maps; one
   matched-shape PC-sampling session pinned 97% of the named-stall difference onto a single site,
   and the fix hit in one shot for +7.1%. An attribution session costs far less than a round of blind trials.
2. **PC-sampling names the consumer instruction**: the sites in a stall table answer "who is waiting",
   never "who is slow" — reading the LDG's latency as the STS's fault sends you to optimize the wrong
   instruction.
3. **A schedule-only diff is the cleanest experiment**: when addresses, traffic, and instruction counts are unchanged item by item,
   any delta can only come from the dependency schedule — variable isolation needs no statistics, only construction.
4. **A fix moves the constraint without eliminating it**: the longsb → lg_throttle migration shows
   staging latency and request rate are serially co-bound; before celebrating, bring back the "next binder" reading
   (r21 immediately proved the other side of this conservation).

---

← 24-r19-weight-l2-residency · [Index](./README.md) · 26-r21-coalesced-block-linear-a →
