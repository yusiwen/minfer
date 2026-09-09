# 19 · r14 — B fragments via ldmatrix + widened scale reads (LANDED)

> **Result**: wide kernel KD=4 1036 → 1225 tok/s (+18.5%), KD=8 1273 (+23–30%); the same
> kernel's ncu duration 3.632 → 2.378 ms (−34.5%), shared-load instructions −64%, LDS
> conflicts −47%, warp instructions −6.8%; the per-GMAC gap to llama narrows from 2.6× to
> 70.4 vs 41.1 µs/GMAC (1.7×).
> **Commit**: `c64cd99`. **Date**: 2026-09-03.

## 1. Background — where things stood

r13's counter forensics had just classified the case: the gap's carrier is the **per-MAC
warp instruction stream** (10.14 M vs 6.06 M per GMAC, 1.67×), and the three hypotheses —
bytes, bank conflicts, store efficiency — were killed simultaneously by the two counter-
evidence patches. The conclusion landed on the compute loop's shared-read instruction count
— so the next cut belongs on shared reads, and it must satisfy the two constraints r13 drew:
**add no staging ALU whatsoever** (r18 would later prove staging-ALU-class substitutions are
wall-clock ineffective at SM% ~30), and **add no new global traffic** (x-tile's −9% already
demonstrated the price of moving bytes around).

Inventorying the wide kernel's (r12's landed 16-chain tile, by now the MMQ path's
performance carrier) compute-loop shared-read streams: A fragments moved to
`ldmatrix.m8n8.x4` at r12 (8 per chunk), but **B fragments were still 4 scalar LDS.32**; the
od-column scales were two streams of 8 LDS.32 each (the post-DCE per-lane execution count;
record basis: 8 LDS.32 per minitile); A-side d/ssum was 2 LDS.32 per (chunk, g) (16 per
chunk). llama.cpp's B fragments are fed by ldmatrix — precisely the not-yet-ported half of
what r9 flagged at the time as "their pre-arranged mma-fragment B layout, producible at
weight-load time".

r14's target list was therefore explicit: change B fragments from "move bytes, then read
them one scalar LDS at a time" to ldmatrix fragments; widen the scale and d/ssum reads. The
constraints were equally explicit: all three changes live on the **read side of the compute
loop**, and the staging side changes only address arithmetic (which slot an element is
written to), adding no expansion multiply-adds — i.e. "fewer/wider smem ops with zero
staging-ALU growth". The narrow kernel is not the performance path and was left alone (r16
later confirmed "narrow is not the perf path" formally).

## 2. Principle — the GPU mechanism

**What ldmatrix is.** `ldmatrix.sync.aligned.m8n8.x4.shared.b16` is a warp-level
shared→register matrix move: one instruction loads four 8×8 b16 matrices, with the 32 lanes
**each supplying one row address** (lanes 0–7 → matrix 0's 8 rows, 8–15 → matrix 1, and so
on); each lane receives 4 32-bit registers whose contents land in the mma operand's fragment
distribution. Against scalar LDS: one `LDS.32` moves 4 bytes with addresses computed per
lane; `ldmatrix.x4` moves 512 bytes in one instruction and the addresses occupy only the 32
lanes' addressing slots. The difference is not bytes (the total is identical) — it is **MIO
queue entries and issue slots**, which r13 had just proven to be the binding resource.

**Why one ldmatrix.x4 can replace 4 LDS.32.** mma.m16n8k32's B operand is 16 rows (two 8-row
minitiles) × 32 k of int8, each lane holding 2 32-bit words (8 bytes). The old code used 2
LDS.32 per minitile, 4 for both; the data those 4 move is exactly four 8×8 b16 matrices —
matrices 0/1 = od rows 0–7's k first/second halves, matrices 2/3 = od rows 8–15's. As long
as the bytes in smem sit in the **row-address distribution ldmatrix expects**, one x4
performs the identical data movement with a register distribution exactly matching what the
mma needs (r14's session verified this equivalence standalone — see §4).

**The 48 B slot stride's bank arithmetic.** smem bank conflicts are decided by the 4-byte-
granularity bank phase. With row addresses laid out by slot, row r's starting bank =
`(stride/4 × r) mod 32`. The old layout's 32 B stride gives `8r mod 32 = 0`: all 8 rows slam
the same phase (an 8-way conflict — the same disease seen on the A side at r12); the new 48
B stride gives `12r mod 32`, and r = 0..7 yields

```
{0, 12, 24, 4, 16, 28, 8, 20}  — 8 distinct phases, zero conflicts
```

48 = 3×16 preserves 16 B alignment (a ldmatrix row-address requirement), at the price of
each slot growing 32 B → 48 B (32 B of content + 16 B pad), the qb8 plane growing 32,768 →
49,152 B (+16 KB, KDR-independent — qb8 holds one super-block). KD=8's block total goes
81,920 → 98,304 B, still 1 block/SM and **nearly filling the ~99 KB opt-in cap** — which
explains why this kernel never had plane-widening room again.

**The same arithmetic for widening the scale / d:ssum reads.** The C fragment's epilogue
consumes only the od column pair (j, j+1) and token pair (t, t+8) per lane:
- sds packs each row's two scales (d | dmin·m) into a `float2`, so adjacent rows join into
  one `float4` and one LDS.128 serves a minitile — lanes 0–3's four float4 addresses sit 16
  B apart covering a contiguous 64 B, conflict-free.
- sda_q is retilted to `[KDR][16 g][8 q]` as uint2: the two packed u32 of the token pair (t,
  t+8) a C fragment needs sit adjacent, read by one LDS.64 (the old code: two LDS.32 with
  addresses 16 B apart).

Together, per (warp, chunk) the shared-read instruction count drops from ~30 (the scalar
streams of 4 LDS.32 + 16 LDS.32 + 16 LDS.32, plus 8 A-side ldmatrix) to 9 ldmatrix + 10 wide
LDS — the origin of the next ncu's −6.8% warp-inst and −64% shared-load against r13's
instruction-stream reading. **The key asymmetry**: r17/r25 later proved that pure "cut
support instructions" is wall-clock ineffective at 1 block/SM, yet r14 immediately gained
+18.5% — because r14 cut **shared-read entries in the MIO queue**, hitting precisely the
LDG→STS→LDS latency chain r13's forensics identified, not the SM-side int ALU.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**slot-major (sg-major), not row-major.** One B-fragment ldmatrix addresses **a single chunk
(sg)**'s 16 rows × 2 16 B halves — laying slots out as `[8 sg][128 od-row][48B]` puts those
16×48 B inside one sg's contiguous 6,144 B region, leaving every per-lane address term but
sg loop-invariant (only `+ sg * (WBJ*WBQ)` moves per chunk). The old row-major `[128 row][8
sg][32B]` scattered the 16 rows across the same row's 8 sg segments — irrelevant for scalar
reads, but ldmatrix's 32 row addresses would have to cross segments. **Why not pick some
other width than 48 B**: 44/48/52 all give distinct phases, but only multiples of 16 keep
ldmatrix row alignment — 48 is the distinct-phase solution nearest 32 among 16 B-aligned
widths.

**sds packed as (d | dmin·m) rather than two planes.** Two planes mean any widened read has
to stitch across planes — two LDS on different planes can never merge into one. Packed as
float2, the width follows naturally from the epilogue's consumption granularity (column
pairs): one float4 = row j's and row j+1's float2 each.

**sda_q as uint2, not uint4.** The first cut used uint4 tiling and wrote straight past the
sda_q plane into the neighboring qb8 (see §3.3) — a consumer's C fragment needs exactly
**one pair** of tokens, so uint2 is the consumption granularity's exact width; uint4 spends
smem budget feeding a nonexistent fourth consumer.

### 3.2 Key code

The excerpts below all come from the **current tree's** `mmq_raw_wide_nt_kernel` in
`src/cuda_kernels.cu` (r14's changes survive verbatim after the adjacent r20/r22/r15
changes; line numbers measured against the current tree).

**(a) smem layout and `MMQ_WBQ` (current tree 5861–5888)** — the 48 B slot's definition and
the four planes' arrangement:

```cuda
#define MMQ_WBI 128
#define MMQ_WBJ 128
#define MMQ_WBQ 48  // padded per-(sg,row) qb8 slot: 16B-aligned, 12r mod 32
...
    //   qb8   [8][128][48]      B sub-blocks pre-expanded to per-k int8,
    //                           SLOT-MAJOR (sg-major); 48B row stride puts
    //                           every ldmatrix row on a distinct bank phase
    //                           (raw nibbles 0..15; 128 od-rows per tile)
    //   sds   [KDR][128] float2 (d | dmin*m): one float4 load serves the
    //                           (j, j+1) od-col pair per minitile
    uint8_t* qa8 = mmq_raw_sh;
    uint32_t* sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_WBI * 32);
    uint8_t* qb8 = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_WBI * 2);
    float2* sds = reinterpret_cast<float2*>(qb8 + 8 * MMQ_WBJ * MMQ_WBQ);
```

The old version (`c64cd99`'s deletion side) was `qb8 [128][8][32]` + separate `sds`/`sdm`
f32 planes + a KD=8 total of 81,920 B — the new layout pushes the block to 98,304 B (+16 KB,
still 1 block/SM).

**(b) the staging side's slot-major expansion (current tree 5986–6015)** — the write side
changes only address arithmetic, no ALU growth; the uint4 group's low/high nibbles each
write two 16 B granules, the low half (pair p's sub-block 2p) and high half (2p+1) in two
adjacent sg rows:

```cuda
        if (((kt) * KDR & 7) == 0) {                     // at KDR=4, reused across kt
        for (int x = threadIdx.x; x < MMQ_WBJ * 4; x += blockDim.x) {
            const int r = x >> 2, p = x & 3;
            const int j = j0 + r, sb = ((kt) * KDR) >> 3;
            uint4 v0 = make_uint4(0,0,0,0), v1 = make_uint4(0,0,0,0);
            if (j < od && sb < nsb) {
                const uint8_t* src = W + (size_t)j * ((size_t)nsb * 144)
                                  + (size_t)sb * 144 + 16 + p * 32;
                v0 = *(const uint4*)(src);               // 32B qs read as-is
                v1 = *(const uint4*)(src + 16);
            }
            const unsigned M = 0x0F0F0F0Fu;
            uint8_t* dst = qb8 + (size_t)(p * 2) * (MMQ_WBJ * MMQ_WBQ)
                         + (size_t)r * MMQ_WBQ;          // slot-major 48B row pitch
            *(uint4*)(dst)      = make_uint4(v0.x & M, v0.y & M,
                                             v0.z & M, v0.w & M);
            *(uint4*)(dst + 16) = make_uint4(v1.x & M, v1.y & M,
                                             v1.z & M, v1.w & M);
            uint8_t* dst1 = dst + MMQ_WBJ * MMQ_WBQ;     // high-nibble neighbor row
            *(uint4*)(dst1)     = make_uint4((v0.x >> 4) & M, ...);
            *(uint4*)(dst1 + 16) = make_uint4((v1.x >> 4) & M, ...);
        }
        }
```

**(c) the compute side: B fragments before/after**. Before (`c64cd99`'s deletion side), 4
scalar LDS.32 per (warp, chunk):

```cuda
// B fragments: 2 minitiles of the warp's private 16 od-rows;
// the staged bytes are already per-k int8 in element order
// (slot sg of the row), so both words are plain smem loads.
#pragma unroll
for (int nh = 0; nh < 2; nh++) {
    const int jr = j0w + nh * 8 + (lane >> 2);
    const uint8_t* rb8 = qb8 + (size_t)jr * 256 + sg * 32;
    b[nh][0] = *(const int*)(rb8 + 4 * (lane & 3));      // LDS.32 ×2/minitile
    b[nh][1] = *(const int*)(rb8 + 16 + 4 * (lane & 3));
}
```

After (current tree 6085–6104), **one** `ldmatrix.x4`, the per-lane address loop-invariant
except for sg:

```cuda
            // B fragments: ONE ldmatrix.x4 serves both 8-od-row
            // minitiles (matrices 0/1 = od-rows 0-7 at k-halves 0/1,
            // matrices 2/3 = od-rows 8-15). reg_i of lane L = matrix_i row
            // L/4, bytes (L%4)*4 — the exact mma.m16n8k32 B-operand
            // distribution the plain LDS pattern produced. Per-lane address
            // parts are loop-invariant; only the sg term moves per chunk.
            {
                const uint8_t* rb8 = qb8
                    + (size_t)sg * (MMQ_WBJ * MMQ_WBQ)
                    + (size_t)(j0w + (lane >> 4) * 8 + (lane & 7)) * MMQ_WBQ
                    + (size_t)((lane >> 3) & 1) * 16;
                unsigned b0_, b1_, b2_, b3_;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(b0_), "=r"(b1_), "=r"(b2_), "=r"(b3_)
                    : "r"((unsigned)__cvta_generic_to_shared(rb8)));
                b[0][0] = (int)b0_; b[0][1] = (int)b1_;
                b[1][0] = (int)b2_; b[1][1] = (int)b3_;
            }
```

The 32 lanes' row-address distribution: lanes 0–7 → od rows j0w+0..7's first 16 B (k first
half), lanes 8–15 → the same 8 rows' second 16 B (k second half), lanes 16–23/24–31 → od
rows 8–15's first/second halves. The four 8×8 matrices land in `b0_..b3_`, matching the old
4-LDS register layout one for one — the precondition for a bitwise "relayout-only" change.

**(d) the epilogue read side: sds float4 and sda_q uint2 (current tree 6124–6142)**:

```cuda
            // od-col scales: one float4 per minitile serves the (j, j+1)
            // column pair the C fragment consumes (float2-packed at staging).
            float dsv[2][2], dmv[2][2];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const float4 sc4 = *(const float4*)(sds
                    + (size_t)kd * MMQ_WBJ + j0w + nh * 8 + (lane & 3) * 2);
                dsv[nh][0] = sc4.x; dsv[nh][1] = sc4.z;   // rows j and j+1's d
                dmv[nh][0] = sc4.y; dmv[nh][1] = sc4.w;   // rows j and j+1's dmin*m
            }
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                float da_q[2];
                int sa_q[2];
                // token pair (t, t+8) in one LDS.64 (uint2 tiling)
                const uint2 pk2 = *(const uint2*)(sda_q
                    + (size_t)kd * MMQ_WBI * 2 + g * 16 + (lane >> 2) * 2);
                da_q[0] = h2f((unsigned short)(pk2.x & 0xFFFF));
                sa_q[0] = (int)(short)(pk2.x >> 16);
                da_q[1] = h2f((unsigned short)(pk2.y & 0xFFFF));
                sa_q[1] = (int)(short)(pk2.y >> 16);
```

The before side was `dsv[2][8]/dmv[2][8]` full enumeration + the `t4` twin scalar reads (the
deletion-side shape in §3.2(c); record basis: scales 8 LDS.32 per minitile → 2 LDS.128,
sda_q 16 LDS.32 per chunk → 8 LDS.64). The launcher changed only the smem-size expression
(`8 * MMQ_WBJ * MMQ_WBQ` replacing `MMQ_WBJ * 256`, current tree 7195–7221); both KD
settings remain inside the ~99 KB cap.

### 3.3 Pitfalls

1. **The first cut's uint4 tiling overflowed and polluted qb8**. Retilting the sda_q plane
   as uint4 wrote past the plane's `KDR·1024 B` boundary and smashed the head of the
   neighboring qb8 — the symptom was not off-by-one results but whole regions of data
   invalidated. Localization came by bisect: two earlier bugs were fixed first (a dropped
   `j0w` term, a word-offset misalignment), parity stayed red, and only then did the plane
   overflow surface. Switching to the consumption-exact uint2 passed on the first try.
2. **The verification order for relayout-class changes**. ldmatrix's register distribution
   must equal the old scalar reads' one-for-one for "change the layout, not the semantics"
   to hold — r14's discipline was standalone verification first (confirming the ldmatrix
   distribution == the mma B-operand distribution), then wiring into the kernel; once in,
   parity went green on the first build.
3. **The smem budget is a hard ceiling**. After +16 KB, KD=8 sits at 98,304 B of ~99 KB —
   this kernel never had room to add a plane again (r17's A-side padding attempt and
   everything after had to subtract within this budget).

## 4. Verification

- **Standalone distribution verification**: ldmatrix.x4's lane→register distribution was
  verified offline first to equal the mma.m16n8k32 B-operand distribution (defends: layout
  right but registers misassigned — the result "looks like a fixed permutation off", easily
  misdiagnosed as a scale error).
- **Parity gate (both KD=4 and KD=8 pass)**: the change touches only byte placement and read
  patterns, not math, so bitwise parity is required (defends: semantic drift introduced by
  the relayout).
- **greedy-32 identity**: the 32-token greedy sequence byte-identical (defends: summation-
  order or sampling-chain drift invisible to parity's short samples).
- **suite 166/0/3**: full regression (defends: the launcher's smem-size change breaking
  other paths).
- **narrow kernel control**: narrow untouched, its readings noise — proving the gains really
  come from this wide-kernel change rather than machine-state drift (defends: polluted A/B
  attribution).
- **The ncu evidence chain**: warp-inst −6.8%, shared-load inst −64%, LDS conflicts −47%,
  duration −34.5% — four readings interlocking; the instruction stream was cut, the
  conflicts cleared, and the time cashed per r13's linear law (defends: wall-clock gains
  from mechanisms unrelated to this one).

## 5. Results

| Metric | before | after | Δ |
|---|---|---|---|
| wide KD=4 whole-machine wall | 1036 tok/s | **1225 tok/s** | **+18.5%** |
| wide KD=8 whole-machine wall | ~1000-class | **1273 tok/s** | **+23–30%** |
| Kernel duration (ncu, same session) | 3.632 ms | **2.378 ms** | **−34.5%** |
| shared-load instructions | — | — | **−64%** |
| LDS bank conflicts | — | — | **−47%** |
| warp instructions | — | — | −6.8% |
| per-GMAC duration | 107.7 µs (r13) | **70.4 µs** | vs llama 41.1 (1.7×) |

narrow control noise; parity green at both depths; suite 166/0/3; greedy-32 identity.
Against r13's prediction: the instruction stream (especially shared-read entries) was cut by
more than half and the wall narrowed to a 1.7× ratio per the linear law — r13's
"instruction-stream-bound" conclusion cashed in its predictive power on its first formal
application. r15 (rank-1 rescale, +1.9%) and r20 (split-phase A staging, +7.1%/+3.5%) then
kept stacking on this kernel; the A-side d/ssum epilogue was subsequently rewritten by r15
(the rank-1 fold after current-tree 6143 is outside this doc's scope).

## 6. Lessons

1. **In the stall-bound regime, "fewer/wider smem ops + zero staging-ALU growth" is the
   lever class that cashes into wall clock** — r13 classified it, r14 cashed it, and r17/r25
   proved that outside this regime (SM-side int-ALU class) the same effort buys ~0%.
2. **A relayout-only change's correctness is designed, not tested**: prove the new layout's
   register distribution standalone-equal to the mma operand distribution first; only then
   does parity qualify as a "green on first build" gate rather than a debugging tool.
3. **A widened read's width is decided by consumption granularity**: the C fragment consumes
   column/row pairs, so float4 and uint2 are exactly enough; wider than consumption (the
   first cut's uint4) only overflows the plane, smashes the neighbor, and buys nothing.
4. **Compute the bank phases before touching a layout**: the 48 B stride's `12r mod 32`
   eight distinct phases are the sufficient condition for zero conflicts — one line of
   arithmetic like this can veto a doomed 8-way-conflict scheme before any code is written.

---

← 18 · [Index](./README.md) · 20 →
