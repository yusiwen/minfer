# 35 · r32 — The finite lever sweep: two regions fenced off (REVERTED)

> **Result**: a SASS region census of the post-r31 NB kernel (2,617 instructions
> total) fenced off both of the integer-ALU surplus's remaining "cuttable
> candidates" — the staging's kt-independent addressing was already hoisted into
> the prolog by ptxas (source-level lever dead, no measurement needed), and the
> epilogue write-back widening measured **+0.46%** (noise) with a structurally
> capped dynamic share of **~0.4%**. Both reverted, cmp-verified = HEAD. The NB
> kernel's integer-ALU surplus thereby reached **the compiler floor**.
> **Commit**: `153d28c` (docs-only record commit; both code experiments were
> completed, measured, and reverted in the local tree — never landed as code
> commits). **Date**: 2026-09-04.

## 1. Background — where things stood

After r28 swapped in the raw-nibble NB kernel (2 blocks/SM, +2.56%), this line ate two positive gains in a row: r29's kd-loop unroll (+2.80%, integer ALU −25%, total instructions −6.5%) and r31's q-major sda scale-read repack (+1.07%, 1424.10 → 1439.40 tok/s, longsb 24.61 → 21.46%). One side conclusion of r29 deserves singling out: **the purely instruction-count cuts that were "wall-inert" in the r25 era started paying out at 2 blocks/SM** — occupancy unlocks instruction cuts, not the other way around. That kept "keep hunting cuttable instructions" reasonable on September 4.

But the levers were visibly thinning. r31 itself was a **sub-bar landing below the +1.5% bar** (the bar was calibrated by r24's scheduling-ladder experiment: wall-clock changes below it are indistinguishable from interleaved A/B noise) — mechanism real, magnitude already brushing the top of the noise band. The earlier r30 was a sharper warning: the SWAR word-granular unpack measured +0.54% (noise), and the SASS comparison showed **r29's unroll had long since induced ptxas to CSE every raw word** — writing in source a version "the compiler already generates" can only add overhead.

By the time r32 started, the books left by r30/r31 read: **integer-ALU surplus 1.598 e-3/MAC, about 2.1× llama's (0.751 e-3)**, attributed three ways — A-frag LDSM (claimed irreducible), staging index math, epilogue. r32's problem statement was deliberately modest: **is there any source-level cuttable component left in this surplus?** If no, the "instruction count" axis closes as a whole and the next round of hypotheses (r33's loop-organization theory) gets a clean start. The method was a **finite lever sweep**: a SASS region census to apportion the kernel's instructions, then one decisive experiment per remaining candidate — falsify what can be falsified, cap what can be capped.

The archival situation matches doc 15's r10: `git show 153d28c --stat` contains only `docs/CUDA_OPTIMIZATION.md` +49 lines and `docs/LLAMA-CPP-MMQ-ANALYSIS.md` +32 lines — the code changes were reverted the same day after measurement, so **the numbers and SASS evidence exist only in the record commit**. Every piece of "current-tree code" cited here is a before form that survived the revert.

The master table's row 46 gives the verdict: "staging addressing already hoisted by ptxas; run-once epilogue cannot clear a bar" — two regions, two kinds of death (proven dead by SASS vs capped by measurement), demonstrating both forms of "fencing off".

## 2. Principle — the GPU mechanism

**The region census — apportioning 2,617 instructions.** The toolchain is r25's SASS census (`cuobjdump -sass`), but classified by **execution region** instead of opcode — r25 answered "which instruction types are over-represented", r32 answers "where do the extra instructions live". The post-r31 `mmq_raw_nb_kernel<8>`'s 2,617 instructions split into four segments (the four regions total 2,585; 32 strays):

| Region | Count | Execution frequency | Contents |
|---|---:|---|---|
| prolog + stage0 | 500 | once per block | index base materialization, first k-tile's staging (incl. stage0's staging share) |
| per-kt in-loop staging | 455 | once per k-tile | A-side LDG batch + swizzle STS + sda repack (~21% — the "staging = 21% of kernel instructions" cited later in r34) |
| per-kt compute-kd | 1,515 | once per k-tile | ldmatrix A-frags + B unpack + 64 IMMA + rescale (the hot path) |
| run-once epilogue | 115 | once per block | 32 scalar STG write-backs + guards |

Statically staging is 455/2617 ≈ 17%; adding stage0's staging share gives the recorded ~21% — the largest nominally "possibly cuttable" block on the books.

**Region → lever mapping.** The census's value is turning "where else can we cut" into a finite list: prolog/stage0 is the product of hoisting, "optimized" by definition; compute-kd is the hot path just harvested by r28–r31 (ldmatrix A-frags = the claimed-irreducible A-frag LDSM; B unpack = proven compiler-CSE'd at r30; 64 IMMA = the campaign's reason to exist; rescale = r15's fp semantics contract); staging's index math and the epilogue are the only two regions that "look untouched" — hence r32's two candidates.

**Swizzle/repack — the staging region's two protagonists.** The index math r32 examined is not casual: r22's XOR swizzle (`(((R&3)<<1 + (u>>2)) ^ ((R>>2)&7)) << 4`) lands each 8×8 ldmatrix tile's 8 rows on mutually conflict-free bank groups — a conflicted LDSM would turn "read A-frags" into a serialization hotspot (r22 once precomputed all 8 tile offsets to zero the address ALU per ldmatrix); r31's q-major region-split repack converges each warp's per-chunk scale reads into two LDS.128s (8 16-B words, 16 B stride, zero conflicts). The staging region's instruction count is the price paid for **conflict-free consumption** — r32 proves that price cannot be cut further at source level (ptxas has hoisted everything hoistable), and r34 will prove it can be **moved wholesale**.

**Dynamic-share arithmetic for run-once regions.** Static count is not dynamic share: the epilogue runs once per block, staging + compute once per k-tile. For the 7B GEMM with hidden = 3584: `nchunk = id/32 = 112`, `KDR = 8` → `nktile = 14`. The dynamic share is

```
115 / (115 + (455 + 1515) × 14) = 115 / 27,695 ≈ 0.42%
```

That is where the record's "epilogue ~0.4%" comes from (prolog/stage0 likewise). The implication is hard: **deleting a run-once region entirely has a theoretical ceiling below 0.5% — it can never touch the +1.5% bar**. The epilogue experiment knew this ceiling from the start — it was measured to get one clean data point and to verify the premise "ptxas really does not vectorize scalar STGs".

**ptxas's hoist mechanism: uniform registers.** The evidence chain is SASS-level. The A-side staging address has two parts: a kt-independent term (the A-token base `(i0+r)*nb32`) and a kt-dependent term (the k-tile stride). ptxas **materializes the former once in the prolog** and loads the latter into a **uniform register** (UR — a per-warp bank of uniform scalar registers, present since Volta, carrying loop invariants identical across the warp, outside the per-thread register file); the in-loop STS target reads `STS [R57+UR11+0x400..]`: R57 is the prolog-computed base, UR11 carries the only per-kt term, 0x400 is a constant offset. **The prolog's and the loop body's STS targets are byte-for-byte isomorphic** — the SASS definition of "the compiler already did it". The av loads likewise merge into 3 base registers + immediate offsets (`4 + kd*40` — in the native pad40 chunk qs starts at byte 4, with a 40 B kd stride). The remaining per-kt addressing is intrinsic (the k-tile is genuinely varying), not a hoistable index chain.

**The minimal SASS forensics workflow.** r32's forensics loop, recorded verbatim: `cuobjdump -sass <binary>` exports the target kernel's disassembly → split the instruction stream into regions by `MMQ.`-prefixed labels or register usage (the prolog's signature is "index math executed once"; the loop body is delimited by `BRA`/labels) → per region, count instructions and inspect operand sources (`UR*` uniform registers = hoisted loop invariants). The whole "staging is hoisted" ruling took under half an hour from export to reading — replacing a pointless round of rewrite + compile + measure.

**Why store widening stops at float2.** The mma `m16n8k32` C-fragment lane map fixes the row/col coordinates of the 8 output points each thread holds: for each (g, nh) combination the thread has one point at row `iA` and one at `iA+8`, with adjacent columns `(j, j+1)` (`l&1` is the column low bit; see §3.2). So per combination a thread can assemble **one pair of adjacent-column 8 B float2s** (one pair per row) — `STG.64` available; the two float2s are a full row apart, no contiguous 16 B pair, `STG.128` not available. Alignment is not the problem: `j0w + nh*8` is a multiple of 8 and `(lane&3)*2` is even, so the pair start is always an even column and 8 B alignment holds. 8 B is this fragment geometry's physical ceiling, and boundary blocks (od/nt not divisible by 128/64) must keep a scalar tail. As for "why doesn't ptxas do it automatically": vectorizing scalar stores requires proving the two STGs' addresses are adjacent, aligned, and side-effect-free in between — the induction across l iterations is not free for ptxas, and here it chose conservatism.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

The two candidates were vetted in different orders. **Staging (455 instructions, ~21%)** is the biggest block on the books, but r30's lesson is forensics before action — it took the pure SASS-forensics route, concluded source-level no-op, and **wrote no code at all**. **The epilogue** was measured even though its ~0.4% dynamic cap was known: it is the cleanest vehicle for "halving the STG count", one experiment answering two questions at once (does ptxas really not vectorize? how much does the widened guard ALU cost?) — both answers matter for every later kernel, and the 0.4% ceiling made it a low-risk probe.

The widening experiment's shape was **float2 interior + scalar tail**: divisible interior tiles take the `STG.64` path, boundary blocks keep the scalar path — trading one runtime branch (dual-path) for halving the interior's store count. That shape choice is itself one of r32's propositions under test: **in a run-once region, is a branch-for-width trade worth it?**

### 3.2 Key code

First the staging side — the complete object the SASS proved "ptxas has already hoisted". `RAW_STAGE_NB`'s A side has three phases (r20 split-phase: LDG batch → scale reads → STS write-back):

```cuda
// src/cuda_kernels.cu:6242-6253 (RAW_STAGE_NB's LDG batch — the before form, still alive today)
_Pragma("unroll")
for (int i = 0; i < KDR * 2; ++i) {
    const int x = threadIdx.x + i * 256;
    const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),
              kd = x / (8 * MMQ_NBI);   /* KDR=8, 8*NBI = 512 */
    const int tok = i0 + r, c = (kt) * KDR + kd;
    unsigned v = 0;
    if (tok < nt && c < nchunk)
        v = *(const unsigned*)(q8x                     // ← native pad40: 40 B per (token, chunk)
            + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4); // base (i0+r)*nb32 + kt term + 4 + u*4
    av[i] = v;                                          //   — SASS: 3 base registers + immediate offsets
}
```

In source, every address explicitly contains `kt`; the SASS ruling is that the base term is materialized in the prolog and the kt stride term goes into UR11. Next the STS write-back — where r32 had hoped to find a lever on this index chain:

```cuda
// src/cuda_kernels.cu:6268-6277 (RAW_STAGE_NB's qa8 write-back — A-side XOR swizzle)
_Pragma("unroll")
for (int i = 0; i < KDR * 2; ++i) {
    const int x = threadIdx.x + i * 256;
    const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),
              kd = x / (8 * MMQ_NBI);
    const int R = kd * MMQ_NBI + r;
    *(unsigned*)(qa8 + (size_t)(R & ~3) * 32                       // ← group base: the kt-independent part
        + (size_t)(((((R & 3) << 1) + (u >> 2))                    // ← XOR swizzle index math
                    ^ ((R >> 2) & 7)) << 4)
        + (size_t)(u & 3) * 4) = av[i];
}
```

Intuitively `R` contains both kd (per-kt) and r (kt-independent) — an index chain recomputed every iteration. The SASS ruling: `R`'s kt-independent component is materialized in the prolog, the per-kt component goes into UR11, and the XOR/shift parts survive instruction-for-instruction but with hoisted operands — **everything hoistable has been hoisted; what remains is intrinsic per-kt addressing**, and a source-level staging rewrite has no actionable lever.

The sda-side repack write (r31's region-split formula) belongs to the census's staging region too:

```cuda
// src/cuda_kernels.cu:6278-6288 (RAW_STAGE_NB's sda write — r31 q-major region split)
_Pragma("unroll")
for (int i = 0; i < 2; ++i) {
    const int x = threadIdx.x + i * 256;
    const int r = x & (MMQ_NBI - 1), kd = x / MMQ_NBI;
    const int g = r >> 4, t = r & 15, q = t & 7, half = t >> 3;
    const int rg = g >> 1, gsel = g & 1;
    /* conflict-free: region=g/2 block, q*16B stride, gsel*8B */
    *(unsigned*)(sda_q + (size_t)kd * MMQ_NBI                // ← kd term: per-kt (goes into UR)
                  + rg * 32 + q * 4 + gsel * 2 + half) =     // ← region-split: kt-independent (hoistable to prolog)
        dv[i] | (sv[i] << 16);                               //   f16 d | i16 ssum packed into one u32
}
```

(This packing formula is reused byte-for-byte in r34's prepass — see doc 37.)

The epilogue's before form still lives in the current tree (the widening was reverted) — `mmq_raw_nb_kernel`'s write-back, 4 g × 2 nh × 4 l = 32 (i,j) points, one scalar `STG.E` per point:

```cuda
// src/cuda_kernels.cu:6434-6444 (mmq_raw_nb_kernel epilogue — the before of r32's widening experiment)
#pragma unroll
for (int g = 0; g < 4; g++)
    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            const int i = i0 + g * 16 + (l >> 1) * 8 + (lane >> 2);   // row: l-high bits + lane-high bits
            const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1); // col: (lane&3)*2 + l-low bit
            if (i < nt && j < od)
                C[(size_t)i * od + j] = sum[(g * 2 + nh) * 4 + l];    // ← 32 scalar STG.E
        }
```

Read the column map: `(lane & 3) * 2 + (l & 1)` — the same lane's `l=0/1` points are **adjacent columns** (stride 1), so each pair merges into an 8 B float2; `l=2/3` differ in row by 8 (`(l>>1)*8`), another pair in the other row. The widened version (never committed) added a divisibility branch for the interior: SASS result `32 STG.E → 16 STG.E.64 + 24 STG.E` — stores really did halve, but the dual-path guard pushed **static integer ALU up** (IMAD 55 → 74, LEA.HI.X 8 → 24), at ptxas 111 regs / 0 spill.

### 3.3 Pitfalls

- **"Static instruction count down" is not the objective function.** In the widening experiment int ALU actually rose (the dual-path guard), and even had it fallen, run-once instructions are irrelevant to the wall — r25's wall-inert conclusion was only ever inverted for the per-kt hot path × 2 blocks/SM combination (r29); the epilogue is not on the hot path.
- **A store widening's ceiling is decided by the fragment lane map, not by desire.** STG.128 needs the thread to hold 16 contiguous output bytes, and the mma C-fragment's row distribution (`iA`/`iA+8`) excludes that from the start. Draw the lane map before setting the widening target and you save an entire experimental round.
- **A census's region split must fix execution frequency first.** The same static instruction carries a dynamic weight differing by nktile (=14) between a "once per block" and a "once per kt" region — classifying by opcode (r25) cannot see this; classifying by region can.
- **The forensics duty when experiment code is never committed.** Same as r10: post-hoc re-inspection of the widened source is impossible; what can be re-inspected is only the SASS numbers in the record commit and the before form in the current tree. A docs-only commit must describe "what changed" well enough that a reader can mentally reconstruct it.

## 4. Verification

- **SASS region census (cuobjdump)**: defends against fake levers — see what the compiler already generates before acting; the r30 pattern made institutional.
- **SASS comparison (widening experiment)**: `32 STG.E → 16 STG.E.64 + 24 STG.E` plus the IMAD/LEA counts — confirms the change touches only the write-back region and quantifies the guard's ALU cost.
- **Parity 1/0 + greedy-32 byte-identity**: defends against "the widened stores corrupting output / boundary blocks written wrong" — the dual-path branch is exactly where boundary mistakes hide.
- **Interleaved 4-pair A/B** (baseline 1441.5 → 1448.15, alternated within one window): defends against co-tenant drift reading +0.46% of noise as signal.
- **cmp-verified = HEAD revert check**: confirms the experiment tree is byte-identical to HEAD — the negative result carries no residue.

## 5. Results

**The staging lever: dead at the source level, no measurement needed.** SASS proves the A-token base is materialized in the prolog, the per-kt term lives in a uniform register, and the av loads merge into 3 bases + immediate offsets — a source-level staging rewrite is a no-op. This is the second instance of r30's "compiler already did it" pattern, this time confirmed by SASS rather than source intent.

**The epilogue lever: measured, capped, reverted.** The SASS store count halved as predicted (32 → 16×64-bit + 24 scalar), the guard ALU rose (IMAD 55 → 74, LEA.HI.X 8 → 24); the interleaved 4-pair measurement gave **+0.46%** (1441.5 → 1448.15) — inside the noise band, and the ~0.4% structural ceiling (the §2 dynamic-share arithmetic) means it can never clear the bar. Reverted, cmp-verified = HEAD.

**The total ledger.** The NB kernel's remaining integer-ALU surplus = A-frag LDSM (irreducible) + intrinsic sda/sds scale decode + fp rescale (FFMA/FMUL/I2FP already at parity/deficit) — **there is no source-level integer-ALU cut left**.

**The lever's reincarnation.** The "staging instructions cannot be cut" death sentence is valid only for **source-level rewrites**: the next day, r34 moved the staging's entire transform component out of the kernel (a quantize prepass pre-transpose), and that 21% of index math vanished wholesale in the bt kernel — the bottleneck was not cut away, it was **relocated**. r32's census (the two numbers: staging 21%, epilogue ~0.4%) is exactly the map r34 used to aim that cut (doc 37).

**Veto mechanism and retry conditions**: a run-once region's dynamic share = static count / (per-kt count × nktile); in a GEMM kernel with nktile ≫ 1 this quotient is always under 1%. Only when the write-back itself becomes a per-tile hot path (e.g. f16 written directly to C, or a tile-geometry change making the epilogue scale with kt) is it worth revisiting this lever. The staging side has exactly one retry condition: a shape that changes the staging's instruction **composition** (not its count) — precisely the direction r34 picked up.

## 6. Lessons

1. **Compute the dynamic-share quotient before writing code**: a run-once region's benefit ceiling = static count / (per-kt count × nktile); a region with a quotient < 1% does not deserve an experiment.
2. **Forensic ptxas's hoisting ability in SASS, not in source intuition**: the uniform register is how it expresses "loop invariant"; an index chain that looks recomputed every iteration may already be split into prolog + UR.
3. **A store widening's width ceiling is written in the fragment lane map**: confirm the thread's output contiguity first, then set the STG.64/128 target.
4. **A census's split dimension decides what it can see**: splitting by opcode (r25) finds instruction-type surplus; splitting by region (r32) finds frequency structure — the same 2,617 instructions, but only the second split exposes "21% in staging, 0.4% in epilogue".
5. **A closing sweep's value is turning open questions into answered ones**: r32 spent a day fencing off two levers so r33's hypothesis could be tested single-variable — negative results queue into the record, and only then is the road clean for positive ones.

← 34-r31-qmajor-sda-repack · [Index](./README.md) · 36-r33-hybrid-inner-loop →
