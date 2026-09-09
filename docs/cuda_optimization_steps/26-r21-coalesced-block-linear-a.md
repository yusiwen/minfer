# 26 · r21 — Coalesced block-linear A staging: stall-mass conservation (REVERTED)

> **Result**: A-side fetch sector count 79.35 → 56.67 M (−28.6%), lg_throttle 1.42 → 0.14 (−90%) — but `wait` +110%, `short_scoreboard` +126%, warp instruction count +11.5%, wall clock **−2.3% / −0.7% (negative)**, reverted.
> **Commit**: `3c009cc` (record/docs commit). **Date**: 2026-09-03.
>
> **Code provenance note**: r21's code changes were fully reverted and **have no directly reachable code commit** —
> `3c009cc` contains only the 64-line record for `docs/CUDA_OPTIMIZATION.md`. Per the forensics protocol, this doc's
> "before" code comes from the current tree (the split-phase staging shape r20 landed; the store
> addresses on today's tree carry r22's XOR swizzle beyond the r21-era baseline, noted inline), and "after" is given as record narration +
> addressing arithmetic — no code is fabricated.

## 1. Background — where things stood

This is the middle of the q4_K MMQ campaign (Era C). r20 (the previous doc) had just taken the
campaign's first clear scheduling-level victory with split-phase A staging: tearing apart the serial chain where
"the first STS of every 4-deep unrolled batch stalls on the global load
latency", issuing all global loads into the register array first and then writing smem in one go — addresses, traffic,
instruction counts all unchanged, only the dependency schedule changed. KD=4 1230.4 → 1317.7 (+7.1%), KD=8 1275.8 →
1319.9 (+3.5%), 6/6 reproducible, and it landed even without reaching the ≥1350 bar.

But r20's ncu evidence also handed over two "not dead yet" leads:

1. **The A fetch is still 8-sector scattered**. The A-side source layout is per-token pad40 blocks
   (`d(2B) | qs(32B) | ssum(4B)`, pitch 40B), and staging enumerates "one 4B word per thread";
   one warp step covers 4 non-adjacent 32B segments; measured sector efficiency only **62.5%**
   (the top stall site holds 16.7% of all stalls, each 4-deep batch eating about one ~600-cycle memory latency).
2. **The freed stalls immediately squeezed in somewhere else**: long_scoreboard 6.22 → 2.92 (−53%),
   yet lg_throttle surged from 0.33 to 2.38. r20's lesson's own words: "staging latency and request
   count are serially co-bound" — fix one, and the other becomes the new binder.

r21's hypothesis is therefore very direct: **if the binder is request quality (sector efficiency) and request count
(L2TEX queue depth), then change the A fetch to coalesced block-linear access** — the warp steps through contiguous
per-token regions, moving one granule per 16B `uint4`, with d/ssum folded into the same pass along the way.
In theory sector efficiency → 100%, load instruction count ÷4, and lg_throttle's queue pressure should fall with it.
This is a frontal assault on item (1) of r20's leftover list.

The overall coordinates of the moment: the wide-tile raw kernel at ~1320 tok/s (KD=8), llama-bench at the same anchor
3401, bar ≥1350. Every percentage point of the campaign had to be attribution-driven — r21 is the decisive experiment for
"is r20's remaining gap really a sector/request problem".

## 2. Principle — the GPU mechanism

### 2.1 The source layout's geometry ledger

The q8_0-quantized activation plane `q8x` is stored per token, one pad40 block per token per 32-k chunk:

```text
token t, chunk c:  [d: 2B][qs: 32B @+4][ssum: 4B @+36]   block pitch = 40B
```

The r20-shape staging enumeration (see the §3.1 code): `x = tid + i*256`, `u = x & 7` (the u-th
4B word within the block), `r = (x>>3) & 127` (token row within the tile). That is, **each thread fetches 4B**;
32 threads in a warp = one (r, u=0..7) 32B qs row × 4 adjacent r's — the global address is
`(tok*nb32 + c)*40 + 4 + u*4`, and rows of adjacent r's are 40B apart. One warp step issues 4
32B transactions landing on 4 different cache lines, and each 40B block's qs(4..35) + ssum(36..39)
straddles two 32B sectors, with the d field's header in between — the combined measured efficiency is 62.5%.

### 2.2 r21's coalescing plan and theoretical gains

r21 flips the enumeration: **the warp steps through contiguous per-token regions, each lane fetching one 16B `uint4` at a time**.
Each (token, chunk)'s qs plane is 32B = two 16B granules; one warp step = 32 lanes ×
16B = 512B contiguously covering 4 blocks' qs; d(2B) and ssum(4B) are folded into smem in the same store pass.
Three theoretical gains:

- **Sector efficiency → ~100%**: the 16B granularity divides the qs plane exactly, no more half-empty
  sectors from the 40B pitch;
- **Load instruction count ÷4**: one `uint4` covers 4 4B words, so the instruction count for the same bytes drops
  linearly (lg_throttle is essentially the L2TEX queue full — queue slots are counted per request, so requests drop 4×);
- **d/ssum folding**: in the old shape d/ssum was a separate round of scattered small loads; after coalescing that second round
  disappears.

On paper this is a change where "every line should win". r21's verdict value lies precisely here: **it won every line, yet lost the wall
clock** — that is exactly the mechanism this doc leaves behind.

### 2.3 Stall-mass conservation

A warp issues at most one instruction per cycle; the kernel's wall clock is decided by "whichever resource stalls the warp first".
Moving stalls from long_scoreboard (data latency) to lg_throttle (queue full) was r20;
r21 wanted to move stalls away from lg_throttle, premised on **the freed issue slots not being immediately
refilled by the next binder**. Coalesced staging saves load instructions and requests, but to compute where each 16B
granule goes, the enumeration itself pays new address ALU and branches — if these added instructions plus
the reshuffled, shallower load batches (deep LDG batches chopped up) refill the issue slots, the wall clock does not move or even regresses.
r21's measurement was the latter ending.

## 3. Implementation

### 3.1 before: r20's split-phase staging (current tree, real code)

Below is the wide-tile kernel (`mmq_raw` family) A-side staging as it stands today. The r20-introduced "all LDGs into
registers first, then unified STS" structure is preserved as-is (the three sections
`av[KDR*4]` / `dv[]` / `sv[]`); the only difference from the r21-era baseline is that qa8's store addresses got
r22's XOR swizzle (noted in the comment) —
r21 at the time faced the un-swizzled flat address `qa8 + R*32 + u*4`:

```cuda
/* src/cuda_kernels.cu — wide-tile raw MMQ kernel, RAW_STAGE macro A side (today's tree) */
{
    unsigned av[KDR * 4];          /* qs words: 128*KDR*8 / 256 thr */
    unsigned short dv[KDR / 2];    /* d f16 words: 128*KDR / 256 */
    unsigned sv[KDR / 2];          /* ssum words */
    _Pragma("unroll")
    for (int i = 0; i < KDR * 4; ++i) {              /* phase 1: all qs loads */
        const int x = threadIdx.x + i * 256;
        const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),
                  kd = x >> 10;    /* x/(8*MMQ_WBI), 8*128 = 1024 */
        const int tok = i0 + r, c = (kt) * KDR + kd;
        unsigned v = 0;
        if (tok < nt && c < nchunk)
            v = *(const unsigned*)(q8x
                + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4);   /* 4B scatter */
        av[i] = v;                                     /* accumulate into registers first */
    }
    /* …d/ssum likewise loaded into dv[]/sv[] first (the other phase-1 batch)… */
    _Pragma("unroll")
    for (int i = 0; i < KDR * 4; ++i) {              /* phase 2: unified STS */
        const int x = threadIdx.x + i * 256;
        const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),
                  kd = x >> 10;
        const int R = kd * MMQ_WBI + r;
        *(unsigned*)(qa8 + (size_t)(R & ~3) * 32
            + (size_t)(((((R & 3) << 1) + (u >> 2))      /* r22 swizzle */
                        ^ ((R >> 2) & 7)) << 4)
            + (size_t)(u & 3) * 4) = av[i];
    }
    /* …sda_q's d|ssum packed store… */
}
```

Key point: addresses, traffic, and instruction counts are word-for-word identical to r20's landing (except the store-side swizzle); the dependency
schedule is "one batch of deep LDG → one batch of STS".

### 3.2 after: r21's block-linear enumeration (record narration)

r21's shape (reverted, no code survives; its structure restored per the record):

- **Enumeration granularity 4B → 16B**: each lane handles one 16B `uint4`, the warp step covering contiguous
  per-token qs regions (one warp step = 512B = 4 blocks' qs planes);
- **d/ssum folded into the same pass**: when the qs loads land and store, d and ssum are written along the way, eliminating the separate
  d/ssum scattered small round;
- **Split-phase kept**: still the "load first, store second" phase structure, with only the enumeration and addresses changed.

Three parity fixes (quoted as-is from the record; all are the index accounts easiest to get wrong when rewriting an
enumeration):

1. Each lane's **slice count** must be recomputed at the new granularity (at 16B granularity, 256 threads × K slices must exactly tile
   the `128×KDR×32B` smem mirror);
2. The granule index is derived from `lane`, not from the old 4B word's `x` decomposition;
3. The global token index must **add back `i0`** (row number within the tile ≠ global token number) — this bug
   is exposed only by the nt=256 sweep shape (at small shapes i0=0 happens to be harmless), a textbook case of
   "single-shape testing missing an addressing bug".

Alongside came a **byte-exact CPU simulator**: on the CPU, replay both enumerations' smem write
sequences byte by byte, confirming the old and new shapes produce exactly the same smem mirror — verifying
"did the layout rewrite change the data" off the GPU.

### 3.3 Pitfalls

- **The i0 loss only surfaced at nt=256**: the single-point nt (the main measured shapes like 2630/2659) happens to start from tile 0,
  so `i0=0` and the lost global token base never fires; only the sweep shape reveals it. The lesson, distilled:
  addressing-class changes must pass a shape sweep.
- **The index accounting of enumeration rewrites**: once granularity changes, the "thread → (kd, r, u)" three-dimensional decomposition must be fully re-derived;
  two of the three parity fixes were this class of decomposition error.
- **ptxas will not preserve batch depth for you**: the old 4-deep unroll gave each warp one coherent deep LDG batch;
  the coalesced enumeration has wider single loads, but the reshuffled batch structure and boundary checks made the issue shape worse
  (see §5's `wait`/`short_scoreboard` rebound).

## 4. Verification

- **CPU byte-exact simulator**: defends against "the layout rewrite quietly changed the smem mirror" — old and new enumerations
  compared byte by byte, confirmed equivalent.
- **Shape sweep parity (nt=256 included)**: defends against i0/slice-count-class addressing bugs that fire only at
  specific shapes (this doc's fix #3 is what it caught).
- **ncu counter pair (sectors / lg_throttle / wait / short_scoreboard / warp-inst)**:
  defends against "looking at only one wall-clock number" — all of this doc's mechanism evidence comes from this pair.
- **Same-window A/B interleaved measurement**: defends against cross-session drift; the wall clock −2.3%/−0.7% is the median difference of
  adjacent same-binary pairs.

## 5. Results

| Metric | before (r20 shape) | r21 coalesced shape | Δ |
|---|---:|---:|---|
| l1tex sectors (A-side fetch) | 79.35 M | 56.67 M | **−28.6%** |
| lg_throttle (stall/issue-active) | 1.42 | 0.14 | **−90%** |
| mio / math throttle | present | gone | — |
| `wait` (barrier-class stalls) | — | — | **+110%** |
| `short_scoreboard` (smem dependency) | — | — | **+126%** |
| total warp instructions | — | — | **+11.5%** |
| kernel duration | — | — | **+10.9%** |
| wall clock (same-window A/B) | — | — | **−2.3% / −0.7%** |

**Veto mechanism**: all three "should-win" accounts cashed in (sectors, request count, and queue stalls all improved massively),
yet the wall clock is negative. The freed issue slots did not become useful work; they were refilled by the reshuffled, shallower load
batches (`wait`/`short_scoreboard` rebound) and the enumeration's own added instructions (warp-inst +11.5%) —
**stall-mass conservation**: sector efficiency and request counts are not the binder; the warp instruction stream is
(r25 later corrected "instruction count" to "issue/occupancy", but the direction agrees: the MIO/byte side is not this
phase's ceiling). The code was fully reverted.

**Under what future conditions a retry is worthwhile**: only when the instruction stream already matches the opponent's and the issue slots
genuinely have slack can "fewer, wider requests" convert into wall clock. The campaign actually took another road — r34 moved the entire A-side
layout transform out of the kernel (quantize-transpose prepass), so the staging problem was **eliminated** rather
than optimized; r51/r52 went further and had the producers emit the already-quantized plane directly. Looking back at r21, it tried to optimize within
the premise of "per-token 40B scattered source" — and the premise itself was later changed.

## 6. Lessons

1. **Stall-mass conservation**: fixing the current binder only makes room for the next one — predicting (and measuring)
   "who picks up the freed slots" is what completes an attribution.
2. **Counters winning ≠ the wall clock winning**: sector/request/queue-class metrics improving while the wall clock regresses means they are not on the
   critical path; measure the next stall before writing the next lever.
3. **Deep load batches are an asset**: when re-enumerating, the old shape's implicit "one deep independent LDG batch per warp"
   gets torn apart unintentionally — the issue shape is an explicit design object, not a byproduct.
4. **Addressing rewrites must pass a shape sweep**: bugs like a lost `i0` base are symptom-free on main measured shapes
   where i0=0; the byte-exact simulator and the shape sweep are two gates, both indispensable.

---
← [25-r20-split-phase-a-staging](25-r20-split-phase-a-staging.md) · [Index](./README.md) · [27-r22-qa8-xor-swizzle](27-r22-qa8-xor-swizzle.md) →
