# 23 · r18: Load-time B pre-expansion — staging becomes a bulk copy (W_exp's debut, REVERTED)

> **Result**: parity green on first build; wall clock KD=8 +0.9% (noise band), KD=4 −19% (median,
> variance out of control); kernel duration KD=8 −4.5%, KD=4 +2.1%; warp instruction count only −0.3%; the price
> +5.8 GB of VRAM. The ≥1350 tok/s bar decisively missed → reverted; the EB/SB machinery preserved as
> a prior asset for later L2-residency / pre-expansion experiments.
> **Commit**: `0a26b35` (record commit; the code left no separate commit with the revert — the originally cited HEAD
> `1e0dded` is an anchor that was swallowed by an amend and no longer resolves, see master-table footnote 3).
> **Date**: 2026-09-03.

## 1. Background — where things stood

2026-09-03, the P6 q4_K MMQ campaign at mid-game. The wide kernel (`mmq_raw_wide_nt_kernel`
) had just taken two structural landings in a row: r12's 16-chain warp tile + ldmatrix
(441 → 1020–1058, +2.3×), r14's B-fragment ldmatrix-ization + widened scale reads
(→ 1225 @KD=4 / 1273 @KD=8), and r15's rank-1 term2 rescale added another +1.9%
(KD=8 1295). But the front was beginning to show fatigue: r16's port of the rank-1 fold into the narrow kernel bought only
a noise-level +1.7%; r17's pure index remap measured +0.9%/+1.0%, landed in the noise band and was reverted.

The campaign's provisional bar stood at **≥1350 tok/s**, and the best config was stuck at 1295 — a gap of only
~4%, yet three consecutive steps harvested only noise. More important than the numbers was the qualitative judgment: r13 had concluded
(per-MAC warp instruction count is the first predictor, 10.14 vs 6.06 M/GMAC), but the IMMA
counts were dead even on both sides — what was slow was not the tensor cores' work, but the supporting instructions and the latency structure. r17's
revert tightened the line further: **with SM% stuck at ~30, pure instruction removal buys no wall clock**. The bottleneck
was classified as "latency/stall structure" — something was making the 8 resident warps wait.

r18's hypothesis came from there (labeled in the record as the Phase-7 task-1 hypothesis): the stall
structure's suspect was **the B-side staging round trip**. Each super-block's staging is
two-phase — first move the 144 B q4_K raw super-block (16 B header + 128 B packed
nibbles) into smem, then use ALU to unpack/expand it in-loop into qb8 `[8][128][48]`
(one int8 per k). This "raw moved in → ALU expand" segment happens between compute phases,
and at 1 block/SM with 8 resident warps there is no other warp to fill the gap.

There was also an earlier seed: r9, while decoding the llama.cpp reference implementation, had written "the remaining lever is
llama's pre-arranged mma-fragment B layout, which **can be produced at weight-load time**" —
r18 was the first attempt to cash that sentence. (The day's box absolute values ran ~9% below the r15/r17 sessions;
this doc takes only same-session interleaved A/B relative values.)

## 2. Principle — the GPU mechanism

First quantify what "pre-expansion" is replacing. The wide kernel's thread block covers 128 od rows × 128 tokens;
B-side staging works at super-block granularity (256 k, 8 sub-blocks of 32 k); the
per-(super-block, od-row) cost:

| Phase | Bytes/ALU |
|---|---|
| read raw | 144 B (128 B qs nibbles + 16 B header d/dmin/scale codewords) |
| ALU unpack | each 32 B word group split into high/low nibbles, `& 0x0F0F0F0F` mask |
| write smem | 256 B of payload into a 384 B padded slot (48 B/slot × 8; the excess is ldmatrix alignment padding) |
| scale side | one header parse per chunk + `get_scale_min_k4` table lookup → `(d·sc, −dmin·m)` float2 |

The key arithmetic is **how much of the instruction stream this ALU occupies** — r18's ncu gave the after-the-fact answer:
the entire expansion is ~0.1% of the per-block warp stream (total warp instructions differ by only −0.3% before/after).
Hoping to speed things up by "cutting the staging ALU" runs into a lever ceiling of 0.1% itself; the real
bet was on **dependency structure**: the two-phase raw→qb8 staging strings a smem dependency chain through
"move → expand → compute", and at 1 block/SM this chain has nowhere to hide.

The pre-expansion plan reshapes the bet into another form — generate two planes **once at load time**:

- **EB plane** (expanded B): per-k int8, layout `[sb][od][8][32]`, 256 B per
  (super-block, od-row), with slot bytes kept as element-ordered raw nibbles
  0..15 — **byte-for-byte equal to the old in-loop expansion's output**. The core layout trick: one
  k-tile's 128 rows × 8 slots assemble exactly into **one contiguous 32 KB global span**,
  so staging degenerates into 2048 × 16 B LDG/STS pure bulk copies with zero unpack ALU.
- **SB plane** (scales of B): the per-(chunk, od-row) `(d·sc, −dmin·m)`
  float2 pairs, `[chunk][od]` layout, replacing the per-chunk header parse + table lookup.

The cost is VRAM: EB ≈ 1 B per weight element (raw q4_K is 0.5 B/element), and with SB roughly
`od·id + 8·od·id/32` bytes — measured on 7B q4_k_m as **+5.8 GB** (free VRAM
26.1 → ~20 GB of 130.6 total, verified per-tensor actual values). And the staging bytes actually
grow: the bulk copy reads **256 B/row** (EB's 8×32 B) while the ALU path reads only
**144 B/row** (the raw super-block) — B-side global read bytes ~2×. This is a
trade of "bytes for ALU + for dependency depth": r13/r17's evidence points at latency (the KD=8 kernel's
−4.5% confirms the latency gain exists), while KD=4's failure shows the price on the other side of the trade
(see §5).

## 3. Implementation

> **Forensics note**: r18's code vanished with the revert; the complete variant survives only in record commit `0a26b35`'s
> narration and in `/tmp` artifacts (`/tmp/patch_p7_t1.py` etc., lost on machine reboot).
> The before excerpt is taken from the **current tree**, i.e. the original path it replaced; the after is
> reconstructed from the record's layout and byte-for-byte contract narration.

### 3.1 Design choices (why this shape and not another)

**Expand at load time, not in the kernel.** The expansion is a per-weight invariant, while the
in-loop version re-expands every super-block once per block that touches the weight — moving it to
load time replaces hot-path repetitive labor with a one-time, off-hot-path cost, the same family as
8p's persistent f16 cache.

**Two planes, not one mixed plane.** Nibble expansion and scale decoding have different
lifecycles (qb8 changes per super-block — at KDR=4 two consecutive k-tiles share one
super-block and can skip the re-move; sds changes per k-tile). Splitting the layouts lets EB achieve the
"one k-tile = one contiguous 32 KB" bulk-copy form; SB's `[chunk][od]` aligns directly
with the staging loop's enumeration order.

**Byte-for-byte contract alignment, not layout reshuffling.** EB's slot bytes are defined as
element-ordered raw nibbles 0..15 — **byte-for-byte identical to the old in-loop expansion's
output**. Getting the layout change to "smem end-state byte-identical" first leaves parity with only
one variable, copy correctness — and indeed it was green on the first build, with none of r14's bisect debugging.

**Registry + fallback, not a hard switch.** `register_weight_q4k_expanded` registers the EB/SB
buffers into the new `mmq_expanded` registry keyed by the raw
`wptr`; the wide kernel launch **requires** the planes to exist and falls back to raw-W narrow-kernel staging if
missing; the raw tensor itself stays
registered and other paths are unaffected.

### 3.2 Key code

**Before — the in-loop B expansion r18 replaces** (current tree `src/cuda_kernels.cu`,
the B section of the wide kernel's `RAW_STAGE` macro; since r18 this section had its A side
rewritten again by r20/r22, but the B-side core is unchanged):

```cuda
/* B: expand ONE 256-k super-block to per-k int8 AT STAGING - raw
 * nibble values 0..15 ... At KDR=4 two consecutive k-tiles share the
 * super-block: restage only when this k-tile starts a new super-block. */
if (((kt) * KDR & 7) == 0) {                       // restage-skip guard
for (int x = threadIdx.x; x < MMQ_WBJ * 4; x += blockDim.x) {
    const int r = x >> 2, p = x & 3;               // od-row, nibble-pair
    const int j = j0 + r, sb = ((kt) * KDR) >> 3;
    uint4 v0 = make_uint4(0,0,0,0), v1 = make_uint4(0,0,0,0);
    if (j < od && sb < nsb) {
        const uint8_t* src = W + (size_t)j * ((size_t)nsb * 144)
                          + (size_t)sb * 144 + 16 + p * 32;   // raw qs
        v0 = *(const uint4*)(src);
        v1 = *(const uint4*)(src + 16);
    }
    const unsigned M = 0x0F0F0F0Fu;                // ← exactly what r18 eliminates
    uint8_t* dst = qb8 + (size_t)(p * 2) * (MMQ_WBJ * MMQ_WBQ)
                 + (size_t)r * MMQ_WBQ;            //   mask unpack + reordered writes
    *(uint4*)(dst)      = make_uint4(v0.x & M, v0.y & M,     // 384 B padded
                                     v0.z & M, v0.w & M);    //   smem slot
    /* ... the high-nibble half likewise written to dst1 = dst + MMQ_WBJ*MMQ_WBQ ... */
} }
```

r18's after (reconstructed per the record): the staging B section no longer reads `W` and no longer has the `& M`
unpack; it does pure 16 B copies from the EB plane at k-tile base addresses — one super-block
is exactly 128 od-rows × 8 slots × 32 B = 32 KB contiguous, each thread enumerating one of the 2048
16 B moves, `LDG.128 → STS.128`, zero ALU.

The scale section likewise — this per-chunk parse:

```cuda
for (int x = threadIdx.x; x < MMQ_WBJ * KDR; x += blockDim.x) {
    int r = x % MMQ_WBJ, kd = x / MMQ_WBJ;
    int j = j0 + r, c = (kt) * KDR + kd;
    float dv = 0.0f, mv = 0.0f;
    if (j < od && c < nchunk) {
        const uint8_t* blk = W + (size_t)j * ((size_t)nsb * 144)
                          + (size_t)(c >> 3) * 144;
        float d    = h2f(*(const uint16_t*)blk);            // f16 d
        float dmin = h2f(*(const uint16_t*)(blk + 2));      // f16 dmin
        uint8_t sc, m;
        get_scale_min_k4(c & 7, blk + 4, &sc, &m);          // 6-bit codeword table lookup
        dv = d * (float)sc;  mv = -(dmin * (float)m);
    }
    sds[(size_t)kd * MMQ_WBJ + r] = make_float2(dv, mv);
}
```

is replaced by a direct read of the SB plane's `float2` — `(d·sc, −dmin·m)` is already a load-time
product.

**Load side** (reconstructed): one `expand_q4k_kernel` device-side launch +
stream sync per tensor, producing EB/SB and registering them; at wide-kernel launch the raw `wptr` is looked
up to decide between the pre-expanded path and the fallback.

### 3.3 Pitfalls

- **No layout pit was stepped in — because the contract was nailed down first.** r14 had once silently
  rewritten qb8 due to a uint4 tiling overrun (caught only by bisect); r18's EB was defined as "byte-for-byte equal to the old expansion output",
  locking the layout variable away in advance, and parity was green on the first build.
- **A hidden test-coverage trap was plugged in advance**: the expanded buffers were also registered into the parity fixture;
  otherwise the fallback logic would have let the wide kernel quietly bypass the new path in tests — what was measured
  would have been the fallback branch.
- **KD=4's variance is itself a signal**: three interleaved runs produced an outlier like 816.2 (median
  −19%, variance out of control), meaning not a stable slowdown but hitting some resource boundary (the record
  attributes it to the staging-byte excess, see §5 reading 2).

## 4. Verification

- **8-shape parity sweep (KD=4 + KD=8 + default + narrow-kernel control) green on first build**
  — defends against the layout/stride-error class of "runs but computes wrongly" regression, and incidentally confirms the fallback branch
  works.
- **Parity fixture registers the expanded buffers** — defends against the false green of
  "fallback logic means the new path was never tested".
- **3 same-session interleaved A/B runs (baseline binary vs r18 binary) + narrow-kernel control**
  — defends against cross-session machine drift and global environment noise (both narrow-kernel pairs are in the noise band, showing
  no environment-level discontinuity; the day's box ran ~9% low, making interleaved relative values the only trustworthy source).
- **ncu single-launch profiling (q-proj, launch 1, nt 2630, od=id=3584,
  grid (21,28), per-GMAC = 33.78e9)** — separates the two opposite-direction effects of
  "latency gain" and "byte cost", grounding the wall-clock conclusion in kernel-level mechanism.

## 5. Results

**Wall clock** (7B @2630 tok, same-session 3× interleaved medians):

| Config | baseline | r18 expanded-B | Δ |
|---|---|---|---|
| wide KD=8 (default) | 1170.9 / 1164.7 / 1155.5 | 929.8 / 1181.7 / 1175.0 | **+0.9%** (noise band) |
| wide KD=4 | 1145.7 / 1141.3 / 1130.6 | 1071.6 / 816.2 / 907.5 | **−19% median, high variance** |
| narrow kernel (control) | 463.3 / 429.9 | 444.5 / 441.5 | noise |

**Kernel level** (ncu, q-proj):

| Metric | KD=4 baseline | KD=4 r18 | KD=8 baseline | KD=8 r18 |
|---|---|---|---|---|
| warp instructions | 8.68 M | 8.65 M (−0.3%) | 8.49 M | 8.46 M (−0.3%) |
| duration | 2.288 ms | 2.335 ms (+2.1%) | 2.155 ms | **2.059 ms (−4.5%)** |
| Compute (SM) | 31.1% | 30.4% | 32.4% | 33.7% |
| Memory Throughput | 44.2% | 36.5% | 46.2% | 43.0% |

Three readings (summarizing the record's own text):

1. **The staging-ALU cut is instruction-neutral**: the expansion ALU was only ever ~0.1% of the
   warp stream — r15/r17's "cutting instructions at SM% 30 is useless" rule thereby extends to staging
   instructions.
2. **The latency gain is real but small and config-dependent**: KD=8's −4.5% kernel duration translates to only
   +0.9% wall clock (the GEMM is only one of ~200 launches, and only the q4_K matmul benefits); KD=4
   instead went +2.1% at the kernel level — the bulk copy reads 256 B/row while the ALU path reads 144 B (~2×
   staging bytes), and on KD=4 the excess outweighed the removed latency.
3. **The fourth independent confirmation that the stall structure is not in the B expansion**: SM% does not move (30–34),
   and removing the B-expansion work did not shrink the stall — the latency binding the kernel is elsewhere (in hindsight,
   exactly the A-side staging, r20's subject).

**Veto mechanism**: the bar ≥1350 was decisively missed, and at every level the reason is structural rather than measurement
fluctuation — no gain at the instruction level (ALU share 0.1%), a reversal at the kernel level on KD=4 (the bytes-for-latency trade nets
a loss at shallow staging depth), the wall clock diluted by "one launch among many + a single weight type benefits",
plus the permanent +5.8 GB VRAM price. Reverted to the then-HEAD (cmp-verified).

**Under what future conditions a retry is worthwhile**: only when the EB plane gains a consumer that is
**not staging**, changing the trade's denominator — the record's explicit candidate is "L2-residency experiments on pre-expanded
weights" (this is exactly why the EB/SB machinery was fully preserved). That prophecy was later partially
cashed: the pre-expansion route revived on q6_K as the W_exp plane and landed (from r43 on), but hit
a dense-plane stride mismatch along the way — a story for later, see docs 46/47.

## 6. Lessons

1. **Staging-ALU cuts extend r17's rule**: with SM% stuck at ~30, cutting instructions
   (staging instructions included) buys no wall clock — first ask what percentage of the instruction stream the operation
   occupies (this time: 0.1%).
2. **"Eliminating work" at a byte cost is a trade, not a gain**: −0.1% ALU for ~2×
   staging bytes nets −19% at shallow staging depth (KD=4).
3. **The arithmetic of kernel-level wins being diluted by the wall must be done in advance**: −4.5% kernel → +0.9% wall; the dilution
   factor = the kernel's share of the wall × the share of weight types that benefit; when the bar is set at wall-clock level,
   compute the lever's global ceiling first.
4. **A reverted machine can be a valuable asset**: the EB/SB materialization machinery was fully preserved, becoming
   the direct precursor of the later L2 experiments and even the q6_K W_exp route — a revert decision and asset preservation
   are not in conflict.

---

← 22-r17-wide-warp-remap · [Index](./README.md) · 24-r19-weight-l2-residency →
