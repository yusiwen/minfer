# 30 · r25 — SASS opcode census; the unroll is wall-inert (MEAS-ONLY + REVERTED)

> **Result**: the census attributes the +25.2% instruction surplus **100% to supporting instruction classes** — integer ALU
> +69,465/tile (77%), fp32 rescale FMUL +15.2k, conversions +14.3k; IMMA and FFMA
> are **equal item-for-item with llama** (114,688 FFMA/tile on both sides). The accompanying kd-unroll fix cut
> integer ALU −38% and total instructions −9.7% (surplus 89.8k → 46.6k/tile, halved), but the wall clock moved only
> **+0.37/+0.49%** — below the +1.5% bar calibrated in r24, reverted; **the census itself is this round's deliverable**.
> Paradigm verdict: the kernel is **issue/occupancy-bound** (98 KB smem → 1 block/SM →
> ~2 warps/sched → latency unhidden), not instruction-count-bound.
> **Commit**: `8658f1b` (docs-only record; the unroll attempt's code, like r24's, never became a code commit). **Date**: 2026-09-04.

> **Forensics note (STYLE rule 0)**: r25 is a "measure + attempt + revert" round; `8658f1b` contains only the
> 87-line `docs/CUDA_OPTIMIZATION.md` record; the census numbers were reproduced verbatim in
> `docs/LLAMA-CPP-MMQ-ANALYSIS.md` §7 (this doc quotes that table directly and notes the source). Code excerpts come from the
> **current tree**: the wide kernel's 1-block/SM budget comment, and the NB kernel's kd loop as landed by r29
> (the same unroll lever's final home, for comparison).

## 1. Background — where things stood

r24 had just closed the scheduling-structure family and recalibrated the landing bar to relative +1.5%. At this point the campaign had two layers of evidence for
"why it is slow": r13's counter forensics said the 3× gap's carrier is the **per-MAC
warp instruction stream** (10.14 vs 6.06 M/GMAC, not bytes), and r20's stall table localized that stream's
latency exposure to long_scoreboard (6.22 vs 1.15, 97% of the named-stall surplus).
But r13's table had one defect, named when r20 re-reviewed it: **the two sides' shapes mismatch** (the pre-r14 kernel
ran nt-2630, llama ran nt-512) — "double-distorted". Attribution had reached the stall,
but not yet the **instructions themselves**: which SASS classes make up the surplus? Math (mma)?
Movement (LDG/STS/LDSM)? Or glue (address arithmetic, predicates, loop control)?

This question decides the next lever's life or death: if the surplus is in mma, it is a decomposition problem;
if in movement, a layout problem (r14/r20/r22 had already taken three rounds); if in supporting instructions,
it is a "glue per MAC" problem — cuttable, but r17 had already rehearsed "cut it and the wall doesn't move" once.
r25 answers both questions at once: the surplus is real (+25.2%, precise to the class), and at the current occupancy
cutting it is wall-inert — **and explains why**. That explanation (issue/occupancy-bound)
directly spawned r28's Direction-A design.

## 2. Principle — the GPU mechanism

### 2.1 How the census was done: the per-opcode-class counting chain

The tool is ncu's **thread-granularity per-opcode metrics** (the `sass_thread_inst_executed_op_*`
family, `pred_on` counts): each SASS instruction class is counted at **thread** granularity; divide by 32
to get **warp instructions** (the /32 conversion was cross-verified against the warp counters). Steps:

1. **Match the GEMM**: both sides run layer-0 q-proj, nt=512, id=od=3584. On minfer's side
   `mmq_raw_wide_nt_kernel` grid (4,28) = 112 128×128 tiles; on llama's side
   `mul_mat_q<12,128,0>` grid (48,1,1) + fixup (48,4,1) (stream-k, equivalent
   to the same GEMM's 112 tiles).
2. **Sum by class, normalize by tile**: each class's total divided by the tile count = per-tile warp instruction count.
   The ledger reconciles: ours `smsp__inst_executed.sum` 49,900,928 → **445,544
   warp-inst/tile**; theirs 39,844,864 → **355,758**; surplus **+89,786/tile
   (+25.2%)**.
3. **The reconciliation gate**: the gap between the classes' sum and `smsp__inst_executed.sum` converges to **~1.4%** —
   the anchor of the census's credibility, defending against "class definitions under- or double-counting".

### 2.2 The census table (reproduced verbatim from `docs/LLAMA-CPP-MMQ-ANALYSIS.md` §7)

| SASS class (ncu opcode metric) | ours/tile | theirs/tile | delta/tile | ratio | reading |
|---|---:|---:|---:|---:|---|
| integer ALU (IADD3/IMAD/LEA/SHF/SEL/ISETP/LOP3) | **113,552** | 44,087 | **+69,465** | 2.58× | ← 77% of the surplus |
| FP32 FMUL (rescale) | 72,592 | 57,344 | +15,248 | 1.27× | dequant-rescale |
| conversion (I2FP/F2I) | 72,600 | 58,254 | +14,346 | 1.25× | int-mma→fp32 |
| misc (NOP/CS2R) | 13,336 | 5,851 | +7,485 | 2.28× | loop/init |
| control-flow (BRA/isync) | 3,584 | 1,160 | +2,424 | 3.09× | loop control |
| uniform datapath (UR) | 1,808 | 55 | +1,753 | 33× | uniform regs |
| FP32 FFMA (rescale/accum) | 114,688 | 114,688 | +0 | 1.00× | **exactly equal** |
| bit (LOP3/PRMT/SHF) | 8 | 456 | −448 | 0.02× | (theirs higher) |
| fp16 HADD2/HFMA path | 15,232 | 36,400 | −21,168 | 0.42× | (theirs higher) |
| memory (LDG/STS/LDS/LDSM) | 31,816 | 36,836 | −5,020 | 0.86× | (theirs higher) |

Accompanying per-tile memory-family counts: global_ld ours 6,944 / theirs 6,384;
**LDSM ours 8,064 / theirs 1,792 (4.5×)**; shared_ld ours 8,960 /
theirs 19,346 (**ours is lower instead**); shared_st ours 5,376 / theirs 7,730.

### 2.3 Three structural readings

- **The compute side is exactly MAC-bound**: IMMA (r20: 1,605,632 equal on both sides) and FFMA
  (114,688/tile equal on both sides) match item for item — the mma work, decomposition, and accumulation depth
  are all aligned; compute was never the gap.
- **The movement side is cheaper on ours**: shared_ld below llama's, LDSM 4.5× theirs — r14's
  ldmatrix B path is exactly "fewer, wider smem ops". "The surplus comes from moving bytes" is rejected.
- **The surplus = 100% supporting instructions**: integer ALU (address/predicate/loop arithmetic) +69.5k holds
  77%, and fp32 rescale (FMUL + I2FP, ~29.6k combined) holds most of the rest. (An honest annotation
  attached: llama's fp16 classes are higher, and the "llama uses half2 scales" reading is still marked interpreted rather than source-verified in the analysis doc's
  §Corrections — the census records counts, not facts about the opponent's source.)

### 2.4 The mechanism of "wall-inertness": issue arithmetic at 1 block/SM

**Wall-inert** = the instruction stream genuinely shortens and the wall clock does not budge. It has a precise
mechanism, with every number in r20/r25's counters:

- The wide kernel KD=8's smem budget is **98,304 B → 1 block/SM** (inside the
  `~99KB opt-in cap`, but doubling would breach it); 256 threads per block = 8 warps, 4 schedulers →
  **~2 resident warps per scheduler**.
- r20's table: `warps_active` **2.00 vs 2.00** (equal residency), but issue/cyc/sched
  **0.20–0.26 vs 0.42** and eligible **0.28 vs 0.64** — llama squeezes 2× the issue out of the same
  2 resident warps. Where the difference lies: our 2 warps spend most of their resident cycles
  stalled on long_scoreboard (2.92, post-r20; llama 1.15) — **when both warps
  wait on the same load's return, the scheduler holds no eligible warp and the issue slots idle**.
- So cutting instructions cannot change the wall clock: the wall is decided by
  **latency-chain length ÷ available overlap (warp count)**, not by total instruction count. Issue slots are only 25–26% used (≪100%),
  showing "instructions too dense" is not the constraint at all — **the constraint is too few resident warps**. ∂wall/∂inst ≈ 0.

The verdict yields a falsifiable prediction: **the same instruction cut should start paying once occupancy is bought
(2 blocks/SM → ~4 warps/sched)**. r29 cashed that prediction (+2.80%), and r28's job was to buy the occupancy.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Census first, lever second**: attribute the instruction stream first, then decide what to cut — r17's lesson
  (blind per-MAC instruction cuts paid 0 wall clock) upgraded into procedure: this time there is a per-class ledger before the cut.
- **The matched-shape protocol**: the root cause of r13's table's distortion was the two sides' different nt; r25 made "same layer,
  same nt, both binaries, tile counts aligned (112 = 112)" the census's precondition gate.
- **Lever choice: the kd-unroll**: within the integer ALU's composition, each chunk's select/base/
  bounds (`is_hi`, smem base, bound compares) are loop-invariant yet recomputed every iteration —
  `#pragma unroll` folds them into compile-time constants, the cheapest targeted cut,
  and it changes no accumulation order (bit-identical by construction).

### 3.2 Key code

The tested lever itself is one `#pragma unroll` line on the wide kernel's kd loop (attempted then reverted,
never committed). Its final home in the campaign is r29 — **the same lever line landed on the NB kernel**.
The current tree's `mmq_raw_nb_kernel` kd loop (`src/cuda_kernels.cu:6333-6341`;
the current-tree form includes r29's unroll; it was absent when r28 landed):

```cuda
    for (int kt = 0; kt < nktile; ++kt) {
        if (kt > 0) RAW_STAGE_NB(kt);
        __syncthreads();

        #pragma unroll                    // ← exactly the line r29 landed: the kd loop unrolled,
        for (int kd = 0; kd < KDR; kd++) { //   is_hi/smem base/bounds all become compile-time constants
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)kd * MMQ_NBI * 32;
```

The wide kernel the census targeted and its 1-block/SM budget (`src/cuda_kernels.cu:7200-7202`):

```cuda
    // 16-chain layout: 128-token x 128-od block tile. r14: qb8 slot-major
    // 48B stride (ldmatrix-for-B) + packed scales. KD=8 totals 98,304B and
    // KD=4 73,728B — both inside the ~99KB opt-in cap, 1 block/SM.
```

The post-unroll-fix reconciliation arithmetic (the derivation of the halved surplus):

```text
int-ALU class cut    = 0.38 × 113,552        ≈ 43,150 warp-inst/tile
total instruction cut = 43,150 / 445,544      = −9.7%            (matches measurement)
total surplus          = 89,786 − 43,150       ≈ +46.6k/tile      ("surplus halved")
int-ALU surplus        = 69,465 − 43,150       ≈ +26.3k/tile      (still the largest class)
wall clock             = +0.37 / +0.49%        <  +1.5% bar       ⇒ REVERTED
```

### 3.3 Pitfalls

- **The shape-mismatch re-accounting trap**: r13's table's nt-2630 vs nt-512 distorted the per-tile comparison;
  any "per-tile normalized" census must first align tile counts (112 = 112),
  otherwise the surplus number itself is an artifact.
- **ncu is a structural authority only**: on GB10 ncu serializes replays and per-kernel times are distorted
  (doc 77's methodology §2.5) — the census reads **counts** (inst/occupancy classes); all wall-clock
  verdicts go to interleaved A/B measurement; the two evidence sets are never mixed.
- **"Below the bar" ≠ "measured for nothing"**: without r24's freshly calibrated relative
  bar, +0.37/+0.49% would have been read as "a small positive in the noise" and wrongly landed. The bar's value cashed in for the first time here:
  vetoing a real but inconsequential improvement.

## 4. Verification

- **Reconciliation gate**: the per-class sum vs `smsp__inst_executed.sum` differs by ~1.4%; the thread→warp
  /32 conversion cross-verified against the warp counters — defends against class-definition under- or double-counting.
- **Matching gate**: same layer-0 q-proj, same nt=512, both binaries, tile counts aligned —
  defends against r13-style shape artifacts.
- **Interleaved A/B measurement**: the unroll fix +0.37/+0.49% (read against r24's relative bar) —
  defends against the intuition-swap of "counters better = wall better".
- **Identity gate**: the unroll changes no accumulation order; parity/greedy still ran as usual (the attempt itself
  was reverted; the identity gate's meaning is confirming "the negative result was not bought with numeric damage").

## 5. Results

- **The census (the deliverable)**: surplus +89,786 warp-inst/tile (+25.2%), 100% supporting
  instruction classes (int ALU 77%); IMMA/FFMA equal item-for-item on both sides; the movement side lower on ours.
  Per-tile wall 211 µs vs 113 µs (+34 µs fixup) — **a ≈1.4×
  occupancy-bound residual** (211 / (113+34) ≈ 1.44).
- **The unroll attempt (reverted)**: int ALU −38%, total instructions −9.7% (surplus 89.8k →
  46.6k/tile), wall clock +0.37/+0.49% — below the +1.5% bar.
- **Veto mechanism**: at 1 block/SM and ~2 warps/sched, instruction cuts are wall-inert —
  the root cause of the idling issue slots is too few resident warps, not too many instructions. **Retry conditions**:
  occupancy ≥ 2 blocks/SM. That condition was satisfied by r28's Direction-A kernel,
  and r29 immediately cashed +2.80% on the same lever (int ALU −25%, total instructions −6.5%).
- **The paradigm verdict** (this round's most important output): the kernel is issue/occupancy-bound,
  not instruction-count-bound. r28's design doc §11.3 quotes this verdict directly for a
  reverse bet: "since a −38% integer cut cannot move the wall clock by even 0.5%, an instruction increase of a few
  percentage points should also be wall-inert — **provided the occupancy lever actually fires**".

## 6. Lessons

1. **Attribute the instruction stream before cutting it** — the per-class ledger (with its reconciliation gate) turns "where to cut" from
   a guess into a reading; after cutting, also check whether the cut class **stands on the critical path**.
2. **Issue arithmetic can predict wall-inertness**: issue 0.25 vs 0.42 @ warps_active 2.00 = 2.00
   says the issue slots are not the constraint; look at occupancy before deciding whether to save instructions.
3. **Occupancy and instruction cuts are ordered, not independent**: buy occupancy first
   (r28) and only then do instruction cuts become visible (r29) — the same lever, reordered, went from +0.4% to +2.8%.
4. **A MEAS-ONLY round's deliverable can be a ledger**: the census table was repeatedly cited afterwards by r29 (the NB kernel's re-census) and
   §8/§9's design comparisons; its compound interest exceeds most landed patches.

---
← [29 · r24 scheduling-structure ladder](29-r24-scheduling-ladder.md) · [Index](./README.md) · [31 · r28 raw-nibble NB kernel](31-r28-nb-kernel-2blocks.md) →
