# 27 · r22 — qa8 XOR swizzle (LANDED); d/ssum fold reverted separately

> **Result**: shared-load conflict counter op_ld 16,859,136 → **0**; KD=8 1311.8 → 1329.6
> tok/s (**+1.4%**, 3/3 pairs positive); KD=4 −0.5% (noise band). The d/ssum stream
> fold attempted in the same round: both enumeration variants **−19–20%**, reverted.
> **Commit**: `5b40058` ("cuda(mmqr): r22 — qa8 XOR swizzle kills the A-side LDSM
> 2-way (op_ld 16.86M -> 0, KD=8 +1.4%); d/ssum stream fold NEGATIVE (reverted)").
> **Date**: 2026-09-03.

## 1. Background — where things stood

r21 (the previous doc) had just delivered an expensive verdict: after changing the A fetch to coalesced block-linear, sectors −28.6% and
lg_throttle −90%, yet the wall clock instead −2.3%/−0.7% — **stall-mass conservation**; neither the byte side nor the queue side is the
binder. The natural next question: on the real binder, the "instruction stream/issue", is there a lever that
**actually deletes work** (rather than reshuffling it)?

r22's Step 0 was pure attribution: on the q-proj, nt=2630, KD=4 shape, break the shared-access
conflict counters apart. GB10 has no dedicated ldmatrix conflict metric (ldmatrix's bank conflicts
fold into the `op_ld`/shared-load wavefront counts), but the ledger balances:

- **op_ld conflicts 16,859,136 = the A-side qa8 ldmatrix's 2-way conflicts** — 4.74 M
  `ldmatrix.x4` × 4 phases each, each 2-way phase paying one extra wavefront, summing exactly onto
  op_ld;
- **op_st 6.5 M is the qb8 staging store + sds conflicts** — untouched this round.

Mechanically this comes from the qa8 smem plane's **32B row stride** (§2 expands). The key judgment: conflicts are
**pure wavefront waste** — the same bytes, the same instruction count, each ldmatrix paying double the
MIO wavefronts; fixing it is "deleting work" rather than "moving work", fundamentally different from r21's reshuffle. And r21's
lesson already pointed the way: the fix must **not add address ALU per load** — which determined r22's final
shape as "each thread precomputes all 8 offsets once".

## 2. Principle — the GPU mechanism

### 2.1 The mechanical definition of a bank conflict

Shared memory is split into **32 banks, each 4B wide**; an address lands in bank
`(addr/4) mod 32`. In one memory instruction, if the 32 lanes' addresses hit mutually distinct
banks, one wavefront completes; two lanes hitting the same bank is a 2-way conflict, and that
phase splits into 2 serial wavefronts — **time ×2, bytes unchanged**. ldmatrix is a warp-level instruction: 32
lanes each supply a row address, `.x4` takes one 8×8 b16 matrix per phase across 4 phases, conflicts counted per phase.

### 2.2 Why the qa8 plane is 2-way

The A-side smem plane is `qa8[KDR][128 tokens] × 32B` (the qs plane of each token per 32-k
chunk). ldmatrix takes A fragments per m16n8k32's standard distribution: lanes 0–7 supply matrix 0's 8 rows,
lanes 8–15 supply matrix 1… rows taken by adjacent phases are **4 rows apart**. And the row stride is 32B: 4 rows = 128B =
32 banks × 4B, **wrapping exactly back to bank phase zero**. So row addresses "4 rows apart" all collide on the same
bank phase — 2-way per phase, one extra wavefront per phase per ldmatrix.x4, cumulatively
16.86 M extra wavefronts.

```text
No swizzle: address delta between row r and row r+4 = 4 × 32B = 128B ≡ 0 (mod 32 banks×4B)
           → the 8 row addresses within an ldmatrix phase collide pairwise → 2-way

With swizzle: the 8 16B granules within a 128B super-row are placed by XOR permutation
           gr(row, h) = ((row&3)*2 + h) ^ ((row>>2)&7)
           4 rows apart ⇔ (row>>2)&7 differs (bit 2 flips) ⇒ 8 rows → 8 distinct phases
```

A concrete comparison (the 4 rows of super-row group 0 plus group 1's first row; granule number = in-row half
×2 + in-group row ×1, then the XOR mask): without swizzle, granule 0 of rows 0/4/8/12 all sit at
byte offsets {0, 128, 256, 384} — all ≡ 0 (mod 128B), same bank phase; with swizzle
they are scattered by masks 0/1/2/3 into 4 different granule slots, and the phase's 8 lanes each land on a
distinct phase.

### 2.4 Why these 16.86 M wavefronts are worth fixing

r20/r21's stall chain already showed: this kernel's issue slots are a scarce resource, and every MIO-side wavefront has to
compete with LDG/STS issue. One 2-way-conflicted ldmatrix phase = the same bytes crossing the LSU pipe **twice**
— 16.86 M extra wavefronts are pure issue-slot rent. It also satisfies two "worth fixing" criteria:
**(a) an exact zero is reachable** (the conflict counter can be verified to 0, not "reduced a bit"); **(b) the fix can
amortize the ALU cost outside the loop** (r21's lesson absorbed head-on).

### 2.3 The XOR swizzle: putting 8 granules into 8 phases

r22 makes every **4 consecutive 32B rows form a 128B super-row**, with the 8 16B
granules inside the super-row placed by XOR permutation:

```text
gr(row, h) = ((row&3)*2 + h) ^ ((row>>2)&7)
  row   : the low 2 bits of the tile-global row number R (which row within the super-row)
  h     : which 16B granule within the row (0/1)
  row>>2: the super-row group number, which sets the XOR mask
```

Two row numbers 4 apart always differ in `(row>>2)&7` (bit 2 flips) → their granule placements are
scrambled by different masks → ldmatrix's 8 row addresses per phase scatter into **8 distinct bank phases**
→ conflicts to zero, with **zero smem growth** (only a permutation inside the 128B). The store side and the
ldmatrix side use **the same mapping**: whichever slot the data lands in is the slot it is read from — the two ends just have to agree.

## 3. Implementation

### 3.1 Design choices: verify standalone first, then integrate "computed once"

Two deliberate orderings:

1. **Standalone first**: first write an independent test kernel to verify the swizzle mapping — 131,072
   LDSM instructions' conflicts fell from 524,288 to **0**, and the A fragments' lane→(row, word) distribution matches the old layout
   exactly (distribution unchanged is what makes bitwise parity possible). Only after the mechanism measured clean did we touch the real kernel.
2. **Offset precompute**: r21's lesson (added address ALU eats the gain) directly determined the shape — the 8
   ldmatrix addresses are **lane-invariant** per thread (they only shift with the kd base), so
   `G[0..7]` is computed once outside the loop; inside the loop each ldmatrix keeps only **one IADD**, fewer than the
   baseline's ~3; registers actually fell below baseline (129/145 vs 141/149, zero spill).

### 3.2 Key code

**Store side** (staging writes place bytes by the same mapping; today's tree's form, with the r20 split-phase
structure kept and only qa8's store address changed — the comment is r22's on-site record):

```cuda
/* src/cuda_kernels.cu — wide-tile RAW_STAGE macro, qa8 store (today's tree) */
/* r22: r20's split-phase A staging is kept verbatim; only the qa8
 * store address gained the XOR swizzle. The d/ssum stream fold
 * (single 9-word-per-chunk pass) was tried and REVERTED: the old
 * scattered d/ssum loads are L1 hits (the qs pass of the same
 * chunks has the lines resident), so folding saves little sector
 * traffic while the flat enumeration costs ALU + branchy batches
 * (-19% wall, see docs r22). */
const int R = kd * MMQ_WBI + r;
*(unsigned*)(qa8 + (size_t)(R & ~3) * 32                 /* 4-row group base */
    + (size_t)(((((R & 3) << 1) + (u >> 2))              /* granule number within the group */
                ^ ((R >> 2) & 7)) << 4)                  /* XOR permutation × 16B */
    + (size_t)(u & 3) * 4) = av[i];                      /* 4B word within the group */
```

**Load side** (each thread precomputes `G[0..7]` once; one IADD per ldmatrix inside the loop):

```cuda
/* src/cuda_kernels.cu — wide-tile kernel, before the main loop (today's tree) */
// r22: precomputed swizzled A-frag byte offsets. The 8 ldmatrix
// addresses per chunk are lane-invariant except for the kd base:
// addr = qat + G[g], G[g] = g*512 + (lane&12)*32
//      + (((lane&3)*2 + (lane>>4&1)) ^ (lane>>2&3) ^ ((g&1)*4)) << 4.
// One IADD per ldmatrix (below baseline's 3), and the granule XOR
// gives every ldmatrix phase 8 distinct bank phases (the 32B row
// stride is 2-way conflicted).
const unsigned l12m = (unsigned)(lane & 12) * 32;
const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                         ^ ((lane >> 2) & 3)) << 4;
unsigned G[8];
#pragma unroll
for (int g = 0; g < 8; g++)
    G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);
```

```cuda
/* consumption point: one ldmatrix.x4 per 16-token group, address = qat + G[g] */
for (int g = 0; g < 8; g++) {
    const uint8_t* p = qat + G[g];          /* the only per-instruction cost: one IADD */
    unsigned r0_, r1_, r2_, r3_;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
        : "r"((unsigned)__cvta_generic_to_shared(p)));
    a[g][0] = (int)r0_; a[g][1] = (int)r1_;
    a[g][2] = (int)r2_; a[g][3] = (int)r3_;
}
```

The read-side formula and the write-side formula are two expansions of the same mapping, item by item:

| Read-side term | Write-side counterpart | Meaning |
|---|---|---|
| `g * 512` | — (carried by the `qat` base via kd) | base of the g-th 16-token A-frag group: 16 rows × 32B = 512B per group |
| `(lane & 12) * 32` | `(R & ~3) * 32` | byte base of the 4-row subgroup the lane sits in (row stride 32B) |
| `(lane & 3) * 2` | `(R & 3) << 1` | in-group row number (low 2 bits) × 2: selects one of the row's two 16B granules |
| `(lane >> 4) & 1` | `u >> 2` | the row's 16B half h (byte 0 vs byte 16) |
| `(lane >> 2) & 3` \| `(g & 1) * 4` | `(R >> 2) & 7` | the XOR mask: the in-group row number's high 2 bits ∥ group parity (carry bit) |

Two key invariants: **(1) `G[g]` is a loop invariant per thread** — once the lane is fixed, its (row, half) in the
m16n8k32 A-fragment distribution is fixed, and the 8 groups are only base shifts, so the whole table is
computed once before the main loop; **(2) the kd base does not enter `G[g]`** — the consumption point carries kd via
`qat = qa8 + kd * MMQ_WBI * 32`, and `G[g]` describes only the in-group geometry; that is the
precise meaning of "the 8 ldmatrix addresses are lane-invariant, shifting only with the kd base".

### 3.3 Pitfalls: the d/ssum fold is r21's echo

In the same round r22 tried Lever 2: folding d/ssum's separate scattered small round into the qs's big enumeration
("one pass sweeps a chunk's 9 words"). **Both enumeration variants were −19–20% wall clock**, three mechanisms stacked:

1. **What is saved are L1-hit duplicate requests**: the so-called "scattered" d/ssum loads land on the same
   cache line the qs big pass just fetched — an L1 hit; the sector traffic merging saves was nearly free to begin with;
2. **The flat enumeration's cost**: mod-9 address ALU + branchy store batches, tearing apart r20's carefully
   kept "one deep LDG batch per warp";
3. **Register blowout**: ptxas hit 255 regs + 112 B spill in the folded form.

This is r21's "stall-mass conservation" replayed at micro scale: **a request-count "saving" paid for in ALU and
batch depth always loses**. The fold was reverted, the swizzle kept — one positive and one negative in the same commit, booked
separately.

## 4. Verification

- **Standalone conflict test**: 131,072 LDSM instructions' conflicts 524,288 → 0, defending against "mapping derived wrong,
  swizzle wasted";
- **A-fragment distribution consistency** (standalone cross-check): defends against "the swizzle changed the lane→data mapping" —
  distribution unchanged is the precondition for bitwise parity;
- **Parity (KD=4 / KD=8 all green) + greedy-32 token identity** (vs the default binary, only
  the timing lines differ): defends against values being polluted by the layout change;
- **ncu pair (op_ld → 0, op_st unchanged)**: proves the fixed counter is exactly the attributed one and the
  conflicts were not pushed onto the store side;
- **smem size check (73,728 / 98,304 B unchanged)**: defends against the "zero growth" promise being quietly broken by
  alignment;
- **3× same-window interleaved A/B** (vs the pre-change binary, 3/3 pairs positive): defends against cross-session drift.

## 5. Results

| Metric | before | after | Δ |
|---|---:|---:|---|
| op_ld (shared-load conflict wavefronts) | 16,859,136 | **0** | zeroed |
| op_st (qb8/sds store conflicts) | 6.5 M | 6.5 M | untouched (not this round's target) |
| registers / spill | 141 / 149 | 129 / 145 | below baseline |
| smem | 73,728 / 98,304 B | same | zero growth |
| wall clock KD=8 (default path) | 1311.8 | 1329.6 | **+1.4%** (3/3) |
| wall clock KD=4 | — | — | −0.5% (noise band) |

**LANDED** (default path positive + mechanism closed loop: the attributed counter exactly zeroed). The combined ≥1350 bar
was not reached — booked under the stop conditions as a "split result": swizzle kept, fold reverted.

In campaign coordinates, this +1.4% is the close of the r20/r21/r22 "staging attribution series" —
r20 proved scheduling can win (+7.1%), r21 proved the byte/request side is not the binder (reshuffling actually lost),
r22 proved the MIO-conflict side can be zeroed and cashed into wall clock. Together the three calibrate "the space left to
squeeze on the A side" down to the instruction stream itself, directly pushing r24/r25 toward scheduling structure (negative) and the SASS
census (the issue/occupancy verdict), with r28's occupancy rewrite then taking over. A +1.4% lever's campaign
value lies mainly in **what question it closed**, not in the number itself.

**The d/ssum fold's veto mechanism**: what it saves are L1-hit duplicate small requests (nearly free), and the price is
mod-9 ALU, branchy batches, torn-up deep LDG batches, and register blowout (255 regs + 112 B spill).
**Retry conditions**: only worth another look when d/ssum becomes a true main-memory scatter (L1 no longer hits) and the enumeration adds no
per-word ALU; the later r34 (quantize-transpose prepass) changed the A-side layout at the root, and this line closed
naturally.

## 6. Lessons

1. **Generate addresses once per thread, not once per load**: a lane-invariant offset table (`G[g]`)
   compresses the swizzle's ALU cost to one IADD per ldmatrix — the key for a "reshuffle"-class change to gain rather
   than lose.
2. **Prefer levers with an exact zero reachable**: conflict-class counters terminate at exactly 0 (op_ld → 0 is verifiable),
   a far cleaner evidence chain than reshuffle-class levers' "reduced a bit".
3. **Attribute first; make the ledger balance to death**: 16.86 M conflicts = 4.74 M LDSM.x4 × 4 phases × 2-way
   extra wavefronts — make the counter equal the mechanism before touching anything.
4. **Book positive and negative results separately within one commit**: r22 = the LANDED swizzle + the REVERTED
   fold; reporting them as one "round" misleads both reuse and revert.

---
← [26-r21-coalesced-block-linear-a](26-r21-coalesced-block-linear-a.md) · [Index](./README.md) · [28-r23-f16-wall-decomposition](28-r23-f16-wall-decomposition.md) →
