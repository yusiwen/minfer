# 34 · r31 — q-major sda scale-read repack: a sub-bar positive gain caught by conflict analysis (LANDED)

> **Result**: 7B q4_K whole-prefill 1424.10 → 1439.40 tok/s (**+1.07%**, median of 45 samples; range
> +0.49 ~ +2.38) — below the +1.5% bar, but the mechanism is ncu-confirmed (long_scoreboard 24.61% →
> 21.46%, mio_throttle −0.92 pp), register-neutral (111 → 109 regs / 0 spill), smem 45,056 → 43,008 B —
> **landed as a sub-bar positive gain**. SASS: LDS.64 32 → 0, LDS.128 16 → 32 (scale path 48 → 32
> conflict-free LDS/k-tile). The first, naive q-major layout had a 2-way bank conflict and reached only
> +0.57%; conflict analysis caught it and produced the group-region split.
> **Commit**: `851a896` (+ `76d495a` docs). **Date**: 2026-09-04.

## 1. Background — where things stood

r30's revert left behind a pinned residual-attribution list: the NB kernel's integer-ALU surplus
versus llama (1.598 e-3/MAC) and the shared-path stalls (mio_throttle 15.4%, long_scoreboard 19.0%)
come from four classes — **A-frag LDSM, sda/sds scale reads, staging index arithmetic, epilogue**. The
B-unpack class had been proven to sit at the compiler's floor (the r30 pattern), so the next item on
the list was the sda scale reads.

sda is the A side's per-token scale data: each 32-token chunk's q8 quantization block carries an f16
`d` (the dequant scale) and an i16 `ssum` (the sum of the block's q8 values), packed together into one
8-byte pair for consumption by the two-term rescale after the mma. The role of `ssum` deserves a
sentence: a q8 block's dot product splits into two terms, "the q8-value contribution + the block-sum
contribution"; the latter is multiplied by the A-side scale (`dma = da * sa`, sa being ssum) and
combined with the B-side min term (dmv = −dmin·m) to form the correction term of r15's two-term rank-1
fold. That is why each token's (d, ssum) must be **available as a pair at the rescale point** — they
are the inputs to two scalar corrections of the same mma accumulator value, and any reordering on the
read side must preserve that pairing. The r28/r29-era layout was one `uint2` (8 B) per token; the read
side issued **4 LDS.64 per chunk** (one per token group) — 32 LDS.64/lane per k-tile across the 8
chunks — the largest family of shared instructions on the scale path.

The SASS-first gate established at r30 "passed" a lever for the first time here instead of vetoing it:
`cuobjdump -sass` showed ptxas had **not** merged those 4 LDS.64 (each kd issues its own at a 0x40
stride) — the opposite of r30's B-unpack (where the compiler had done everything); here was a genuine
compiler blind spot, so the lever had footing. The motivation was twofold: replace 32 narrow loads
with 16 wide ones (saving MIO issue slots), and land every load on a conflict-free bank distribution
(compressing the stall class).

The other piece of context is landing-bar politics: after r29 the wall bar is +1.5%. sda reads are
only a few percent of the kernel's instruction stream, so this lever would most likely miss the bar —
**whether to land a measured sub-bar result** became the question this step had to answer head-on.

## 2. Principle — the GPU mechanism

**sda/sds distinction**: `sda` = the A side's per-token packed `(d f16 | ssum i16)` plane; `sds` = the
B side's per-(chunk, od-row) `(d·sc | −dmin·m)` float2 (the B-side term of r15's two-term rank-1
rescale). This step touches only sda — sds reads are already at maximum width (LDS.128, float4), at
their own floor.

**Old layout** (the r28 original): `sda_q` holds one `uint2` per token, word index `kd*128 + g*16 +
q*2 + half` (g = token group 0..3, q = within-group pair 0..7, half = the 8-token half-group 0..1).
The read side issues one LDS.64 per (kd, g) reading 2 adjacent words. The raw material for wide loads
was all there: the same lane's 4 groups' words sit 16 B apart — ptxas failed to see it.

First, pin down the old layout's "charge sheet" precisely: its LDS.64 is **not conflicted**. Within
one phase of an LDS.64, the 16 lanes present only 4 unique 8-B words (q = lane>>2; the 4 lanes of a
group broadcast), 4 broadcasts per phase, zero conflicts. The problem is purely **instruction count**
— each load moves only 8 B, 32 of them per k-tile; the wide-load merging opportunity (the same lane's
16 B adjacency across groups) was always there, but the per-g read order gave ptxas no foothold. So
this step's benefit model is "instruction-count halving first, conflict removal second" — which
determines the direction of the region-split derivation below.

**Naive q-major, first version (the rejected shape)**: lay out the 4 groups' words each lane needs,
`[q][g0..g3]`, contiguously — 32 B per lane, 32 B stride within a warp. Issuing LDS.128, one phase (8
lanes) reads 8 16-B words at byte offsets 0, 32, 64, …, 224:

```
bank(word start) = (byte_offset/4) mod 32
  → 0, 8, 16, 24, 0, 8, 16, 24
  → 4 bank quads each hit by 2 words = 2-way conflict
```

A 32 B stride is exactly 1/4 of the bank space (128 B); 8 words stomp 4 quads twice each — every
LDS.128 splits into two waves, and the wide-load benefit is cut in half. Measured: only +0.57%; the
conflict analysis explains why.

**Group-region split (the landed shape)**: the word index becomes `kd*64 + rg*32 + q*4 + gsel*2 +
half`, where `rg = g/2` (the 4 groups split into two 32-word regions) and `gsel = g&1`. Within a
region, `q*4` gives a 16 B stride and `gsel*2 + half` gives the 4 word slots inside a 16 B word-group.
A lane reads its full per-chunk `(d|ssum)` set with two LDS.128s (s0 = groups 0,1; s1 = groups 2,3):
each instruction has the warp read 8 unique 16-B words at byte offsets 0, 16, 32, …, 112 → banks 0-3,
4-7, …, 28-31 — **each of the 32 banks exactly once**, zero conflicts. The write side is just as clean
— one staging warp decomposed by lane (g = lane>>4, half = (lane>>3)&1, q = lane&7) produces this bank
sequence for its 32 4-B writes:

```
lane  0.. 7 → gsel=0, half=0 → word slot q*4   → banks 0,4,8,12,16,20,24,28
lane  8..15 → gsel=0, half=1 → word slot q*4+1 → banks 1,5,9,13,17,21,25,29
lane 16..23 → gsel=1, half=0 → word slot q*4+2 → banks 2,6,10,14,18,22,26,30
lane 24..31 → gsel=1, half=1 → word slot q*4+3 → banks 3,7,11,15,19,23,27,31
```

— 32 banks hit exactly once each; the whole warp's writes complete in a single instruction, single
transaction. Both the read and write sides are conflict-free.

**Semantic invariant**: only the storage order moves; the math does not. Every word's value is
unchanged (`d | ssum<<16`), the consumption sites are unchanged (still used per (g, half) in r15's
two-term fold), and the r22 qa8 swizzle is untouched — so the correctness gate can demand
bit-identical results.

**The occupancy side effect is positive**: sda_q shrinks from 4,096 B to 2,048 B, total smem 45,056 →
43,008 B; 2 blocks/SM is kept with a thicker margin; registers 111 → 109 (the per-g address arithmetic
becomes one uint4 read + constant selection). The smem ledger can be re-verified item by item: qa8
8×64×32 = 16,384 B, sda_q 2,048 B, qb_raw 128×128 = 16,384 B, sds 8×128×8 = 8,192 B — total 43,008 B,
matching the kernel comment and the launcher's formula; substituting the old sda_q's 4,096 B back in
gives exactly the pre-change 45,056 B.

## 3. Implementation

### 3.1 Design choices (two layouts, one conflict analysis)

- **Why naive q-major was tried first**: `[q][g0..g3]` contiguous at 32 B/lane is
the most intuitive arrangement of "one lane's data together", and the write-side indexing is
simplest. It fails at the warp dimension: layout correctness is per-lane, bank behavior is per-warp —
the intuitive arrangement buries the warp conflict inside its 32 B stride.
- **Conflict analysis before integration**: the bank re-check was done only
after the naive version measured +0.57% (positive but suspiciously weak), which found the 2-way
conflict. Run in the other order — analyze first, measure second — the lesson would have saved one
integration.
- **The region-split derivation direction**: the target read order is "an
LDS.128's 8 unique words spread across all 32 banks"; working backwards, the word layout must have a
16 B intra-warp stride; 16 B × 8 = 128 B = one region; 4 groups do not fit → split into two regions
(rg = g/2), each LDS.128 handling two groups.
- **Two LDS.128s, not one**: each lane needs 8 words per chunk (4 groups × 2
halves) = 32 B — exactly two LDS.128s; the naive version also issued two. The only difference is the
intra-warp stride (32 B → 2-way conflict, 16 B → zero conflict). **Identical instruction count,
different address pattern** — the entire wall-clock difference comes from the conflict waves. This is
the minimal specimen of "layout is performance": the instruction counters see no difference; the bank
analysis and the wall clock do.
- **sds untouched**: it is already at the LDS.128 floor; touching it would be a
lever with no mechanism (the r30 pattern applied preemptively).

### 3.2 Key code

**Write side** (inside the staging macro, `src/cuda_kernels.cu` current tree; in `git show 851a896`
changed from uint2/token-pair tiling to region-split):

```cuda
// before r31 (a − line of 851a896): uint2 per token, g-major
// *(unsigned*)(sda_q + ((size_t)kd * MMQ_NBI + (r >> 4) * 8 + (r & 7)) * 2
//              + ((r >> 3) & 1)) = dv[i] | (sv[i] << 16);

// after r31 (current tree): region-split, the warp's 32 writes cover all 32 banks once each
for (int i = 0; i < 2; ++i) {
    const int x = threadIdx.x + i * 256;
    const int r = x & (MMQ_NBI - 1), kd = x / MMQ_NBI;
    const int g = r >> 4, t = r & 15, q = t & 7, half = t >> 3;
    const int rg = g >> 1, gsel = g & 1;
    /* conflict-free: region=g/2 block, q*16B stride, gsel*8B */
    *(unsigned*)(sda_q + (size_t)kd * MMQ_NBI
                  + rg * 32 + q * 4 + gsel * 2 + half) =
        dv[i] | (sv[i] << 16);          // value unchanged: d f16 | ssum i16
}
```

**Read side** (two LDS.128s per chunk replace four LDS.64s; current tree):

```cuda
// before r31: one LDS.64 per (kd, g), 32 per k-tile
// const uint2 pk2 = *(const uint2*)(sda_q
//     + (size_t)kd * MMQ_NBI * 2 + g * 16 + (lane >> 2) * 2);

// after r31: s0 = groups 0,1; s1 = groups 2,3 — 16 B warp stride, zero conflicts
const uint32_t* sda_blk = sda_q + (size_t)kd * MMQ_NBI
                          + (size_t)(lane >> 2) * 4;   // q*16B stride
const uint4 s0 = *(const uint4*)(sda_blk);       // rg=0: g0h0,g0h1,g1h0,g1h1
const uint4 s1 = *(const uint4*)(sda_blk + 32);  // rg=1: g2, g3, each half
#pragma unroll
for (int g = 0; g < 4; g++) {
    float da_q[2]; int sa_q[2];
    const unsigned w0 = g == 0 ? s0.x : (g == 1 ? s0.z : (g == 2 ? s1.x : s1.z));
    const unsigned w1 = g == 0 ? s0.y : (g == 1 ? s0.w : (g == 2 ? s1.y : s1.w));
    da_q[0] = h2f((unsigned short)(w0 & 0xFFFF));    // consumption identical point-for-point to the old layout
    sa_q[0] = (int)(short)(w0 >> 16);
    da_q[1] = h2f((unsigned short)(w1 & 0xFFFF));
    sa_q[1] = (int)(short)(w1 >> 16);
    ...
}
```

### 3.3 Pitfalls

- **Layout-correct ≠ warp-correct**: every lane of the naive q-major got the
right data; what was broken was the warp-level bank distribution of the 32 B stride. A shared-memory
layout review must do both layers: the per-lane value mapping + the per-warp bank trace of the
accesses.
- **Positive but suspiciously weak = a mechanism problem signal**: a number like
+0.57% ("right direction, limping magnitude") deserves the question "why"; only after the conflict
analysis answered it did region-split reach +1.07%. Without asking, a real lever gets sold at half
price.
- **Handling a flaky suite**: this round's suite once came out 164/2 flaky; it
was recorded only after a rerun came back all green (166/0/3) — a flaky run is rerun-confirmed,
neither counted nor ignored outright.
- **Shrink one region, fix every derived pointer**: after sda_q went from
`uint2`/token to `uint32`/token, the `qb_raw` base formula had to change from `sda_q + KDR * MMQ_NBI *
2` to `* 1` in lockstep, and the launcher's smem formula from `* 8 → * 4` — miss any one of them and
the staging writes overrun/corrupt the adjacent plane (off by exactly one region; parity will
certainly explode, but this class of error is best caught in diff review, not waiting for the
cross-check). "Shrinking one smem region" is the classic three-site coupled change.
- **uint4 alignment is a property the layout gives you**: the new read side is a
`uint4` load and needs 16 B natural alignment; the region split's `q*4`-word offset (= q×16 B)
provides exactly that. The old layout needed only 8 B alignment; switching to `uint4` reads directly
on the old indices would send half the accesses across 16 B boundaries — alignment constraints belong
in the layout design, not left to runtime.
- **ptxas's blind spots are selective**: at r30 it had already done the B-unpack
CSE; at r31 it did not merge the sda loads (the 0x40 stride was right there) — both "the compiler
already did it" and "it didn't" must be verified point by point in SASS, never extrapolated by
intuition.

## 4. Verification

- **Parity (NB-active) 1/0**: the strongest gate for a layout reorder — any
(g, q, half) misalignment is ~1e0 scale while f32 rounding is 1e-5 scale; one test tells them apart.
- **greedy-32 byte-identical**: the greedy 32-token output is byte-identical —
confirms "same values, same consumption sites"; the rescale math and summation order were untouched.
- **SASS reconciliation**: LDS.64 32 → 0, LDS.128 16 → 32 (scale path 48 → 32
per k-tile, all conflict-free layout) — the mechanism cashes out at the compilation-artifact level.
- **ncu**: long_scoreboard 24.61% → 21.46%, mio_throttle 16.53% → 15.61%
(−0.92 pp) — the stall class really was compressed, and no new stall class appeared.
- **Resource ledger**: 109 regs / 0 spill (−2), smem 43,008 B — occupancy stays
2 blocks/SM with no regression.
- **Suite 166/0/3** (including one 164/2 flaky rerun-confirmed).
- **45-sample large-N A/B**: a small effect (+1.07%) is unresolvable at the
5-pair protocol's scale (r29's protocol size); only a 45-sample median + range (+0.49 ~ +2.38) could
lift the signal out of the noise band.

## 5. Results

| Metric | r29 baseline | after r31 |
|---|---|---|
| whole-prefill (45-sample median) | 1424.10 | **1439.40 (+1.07%)** |
| sda reads (SASS, per k-tile/lane) | 32 × LDS.64 | **16 × LDS.128 (0 LDS.64)** |
| long_scoreboard | 24.61% | **21.46%** |
| mio_throttle | 16.53% | **15.61%** |
| regs / spill | 111 / 0 | **109 / 0** |
| smem | 45,056 B | **43,008 B** (sda_q 4,096 → 2,048) |

+1.07% is below the +1.5% bar; **the ruling to land it rests on**: (1) the mechanism is doubly
confirmed by ncu and SASS — it genuinely eliminated an entire stall-contributing class (32 narrow
scale reads) rather than being coincidental positive noise; (2) a zero-regression surface — registers,
smem, and occupancy are all neutral or better; (3) the direction is stackable — the scale-read class
belongs to the same family as the later r35 (sds predecode, REVERTED), and this step's floor is that
step's starting point. The ruling is recorded in the master table's "LANDED (sub-bar)" status: **a
sub-bar but mechanism-confirmed positive gain may be kept when it "compresses some stall class with no
regression" — the bar-decision itself must be recorded explicitly, not left for posterity to
excavate**.

Two post-hoc notes. First, the 43,008 B smem thickened the 2-blocks/SM margin, but the NB kernel was
never pushed to 3 blocks afterwards — the q6_K line's r40 later proved the 3rd resident block is
bought with `__launch_bounds__` register trade-offs, not smem subtraction; the NB 2-block equilibrium
held until the campaign's end. Second, an effect of +1.07% magnitude established the 45-sample median
protocol — every later sub-bar candidate (r32/r33 etc.) used the "large sample + mechanism
confirmation" double referee, with the 5-pair protocol reserved for candidates above +2%.

## 6. Lessons

1. **A shared layout passes two reviews**: the per-lane value mapping (parity's
job) + the per-warp bank trace (conflict analysis's job) — naive q-major lost at the second review,
and only a suspiciously weak measurement exposed it.
2. **The SASS-first gate is bidirectional**: at r30 it vetoed a lever the
compiler had already exhausted; at r31 it passed a real lever inside a compiler blind spot — the
gate's value is turning "compiler behavior" from guesswork into evidence.
3. **Landing sub-bar requires the trio**: mechanism confirmation (ncu/SASS) +
zero regression (regs/smem/occupancy) + an explicitly recorded bar decision; missing any one, a
sub-bar positive gain should be reverted.
4. **Small effects need large samples**: a wall-clock effect of ~+1% is
unresolvable inside the 5-pair protocol's noise band; a 45-sample median + range is the right
measuring instrument for this class of lever.

← [33-r30-swar-unpack](33-r30-swar-unpack.md) · [Index](./README.md) · [35-r32-finite-lever-sweep](35-r32-finite-lever-sweep.md) →

