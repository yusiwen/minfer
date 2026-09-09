# 12 · r5–r6 re-ranking + structural rewrite spec (MEAS-ONLY + REVERTED)

> **Result**: both queued levers re-measured negative — KD=4 re-test 427 vs 438 tok/s, and the
> 4-warp 32×32 warp tile 399 tok/s with a parity hole (zero cells in `mmq_w80`) — all reverted;
> the wall decomposition was also corrected (the 352 ms q4_K dequant pass runs at LOAD time and
> is not inside the prefill wall at all). r6 then landed the execution spec: raw-byte smem
> staging + dequant at mma time — it became the execution contract for the 30+ rounds of work
> that followed in Era C.
> **Commit**: `1e0673f`, `491eb5c` (both docs commits; the code under test (KD=4 switch, 4-warp
> tile, dequant vectorization) was reverted right after measurement and never became its own
> commit). **Date**: 2026-09-01.

## 1. Background — where things stood

As P5 wrapped up (around 2026-09-01), minfer's CUDA engine stood here: the default f16 path,
pushed by the TM=128 big tile (P5·2) and the FA softmax balancing (P5·3), had reached 2340–2370
tok/s @2K — 1.44×/1.43× against llama-bench's 3401 @2K; the decode line (R2 MMVQ weight-
streaming rework, R4 split-attention dim-parallel rewrite) had already pushed decode to a seesaw
against llama.cpp. **Prefill was the remaining main gap, and prefill's next direction had to be
a binary choice: keep pressing the f16 path's wall, or pull the opt-in int8 MMQ line (R1,
`40e97c9`) up from 441 tok/s.**

R1's position was awkward. It was parity-clean and structurally correct (int8 tensor-core mma +
q8_0 activations), but its throughput was only ~6.1 TMAC/s: the f16 GEMM itself ran about 24
TMAC/s and llama.cpp's MMQ about 30. That is, R1 trailed the f16 GEMM by 4× and llama.cpp's MMQ
by 5× — and when R1 landed, that 8×-scale gap **had never been attributed** (the master table's
R1 row, verbatim: "the 8× gap was unprofiled"). Every overt lever on the r1–r4 incremental line
(MMQ landing, MMVQ streaming rework, split-attention rewrite) had been harvested; the MMQ line
stalled on the question "what is the next lever?"

Two candidate levers were queued at the time, both "obvious" changes to the R1 kernel shape:

- **KD=4**: drop the number of 32-k chunks resident in each double-buffer half from 8 to 4,
  halving the smem footprint and raising occupancy from 1 block/SM to 2 — the textbook occupancy
  lever;
- **4-warp 32×32 warp tile**: change the warp tile from 32(i)×16(j) to 32×32 so each fragment
  word feeds 2× the mma, halving smem traffic — the textbook data reuse lever.

A third "obvious" lever was even more tempting: **the 352 ms q4_K dequant pass**. In the one-
shot CLI prefill nsys timeline of the time there was a ~352 ms w16 fill that looked like forty
percent of the 890 ms total — vectorizing it seemed like free money.

What r5–r6 did was knock down each of these three "obvious" ideas one by one, then write the
only road left standing into an executable spec. **The cost of not doing this is concrete**:
without re-measuring KD=4 and the 4-warp tile, the team would keep investing on a wrong
assumption ("occupancy/reuse is the gap's source"); without correcting the dequant pass's
attribution, the team would optimize a pass that isn't inside the wall; and without a spec, a
structural rewrite of that size would be attempted with no parity gate — which is exactly how
r5's 4-warp tile broke parity on the spot.

## 2. Principle — the GPU mechanism

**Depth vs occupancy: why KD=4 was slower.** R1's word-staging kernel keeps this cadence per
k-tile: `[dequant ALU + smem expansion + __syncthreads] → mma section`. The staging section is a
block-wide barrier-synchronized serial section — all 8 warps must finish their unpack before
anyone crosses the barrier. At KD=8 you pay that cadence once per 256-k; KD=4 doubles the
cadence count, so the fixed overhead amortized per weight byte (barriers, addressing, scale
reads) rises. Going from 1 to 2 blocks/SM does let another block's compute section overlap with
this block's staging section, but for this kernel the overlap gain cannot buy back the doubled
cadence — the master table's verdict is **"staging-depth amortization dominates"**. 427 vs 438:
the occupancy lever lost by 2.5%. The transferable conclusion: **the depth-vs-occupancy trade-
off is a property of the kernel class** — the heavier the staging section (more ALU, denser
barriers), the more valuable depth amortization is; once r7's raw-byte kernel cut the staging
section down to pure cp.async, the trade-off point would move (the r6 spec wrote this explicitly
as "raw staging shifts the tradeoff", requiring both KD depths to be re-measured).

**4-warp 32×32: why the reuse lever broke on correctness.** Changing the warp tile from 32×16 to
32×32 has two theoretical gains: each fragment word (one smem read) feeds 2× the mma, and the
B-side fragment count stays fixed while coverage doubles, halving smem traffic. The cost: 256
threads must be cut into a 4×2 sub-block grid, staging partitioning changes from a single loop
to loopified, and B fragments go from 2 to 4 — the whole fragment mapping had to be re-derived
by hand. The result: 399 tok/s (9% slower than 438) **and** a `mmq_w80` parity failure: some C
cells got **zero contribution** (zero cells) — not numeric drift, but a hand-written mapping
that missed some cells' contribution paths entirely. What the reuse lever saved on "more mma per
word" was outweighed by the index ALU and scheduling losses the mapping complexity introduced;
and the parity hole showed the complexity had already passed what could be hand-verified at the
time.

**Wall decomposition: what inside the 890 ms is movable.** r5 re-read the one-shot CLI prefill
with nsys and split the wall into: GEMM ~600 ms + convert 56 + fa 54 + swiglu 51 + add 19 + host
gaps. Three secondary conclusions:

- swiglu re-measured at ~257 GB/s = bandwidth peak — already at roofline, no implementation
  headroom left;
- convert (the f32→f16 activation conversion) at 56 ms is an inherent tax of the f16 path;
- **the 352 ms dequant pass runs at LOAD time** (the w16 warm phase one-shot dequantizes q4_K
  into the f16 weight cache) and is not inside the prefill wall. Vectorizing it measured a null
  delta — the scalar version was already warp-coalesced, and it runs on the cold-start path,
  invisible inside the wall.

GEMM is therefore the only material lever: everything else is either one-shot (dequant), at
roofline (swiglu), or small (fa/add). The source record puts it as "GEMM ~85% is the only
material lever" — meaning that after host gaps and incompressible items, nearly all actionable
time is GEMM.

**The dequant-at-mma-time principle.** The spec's core insight: the nibble unpack's int ALU does
not have to finish during staging — in the gaps between tensor-core `mma.m16n8k32` issue slots,
the INT32 pipeline is largely idle. Resident the B side in smem as **raw bytes** (the 144 B Q4KB
super-block: d[2] + dmin[2] + scales[12] + qs[128], nibbles unexpanded, un-centered), and
staging degenerates into pure cp.async copies; move the unpack into the mma loop, one shift/mask
per 32-bit word, `__vsubss4` for centering — the ALU then overlaps tensor-core issue instead of
serializing ahead of the barrier.

New concepts in this doc: **pad40** = the 40 B q8_0 32-element block layout (d 2B + qs 32B @4 +
ssum 4B @36, padded to 40 overall); **Q4KB super-block** = q4_K's 256-k weight block (144 B);
**KDR** = the number of 32-k chunks resident per double-buffer half (KD=8 means 256-k); **word
staging** = R1's approach — expanding nibbles into int8/words in smem during staging.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

r5's two negative results marked the end of the road called "keep looking for staging knobs on
the R1 shape": occupancy (KD=4) and reuse (32×32) had both been tried and both lost. There was
still no profiler (ncu would not work on this device until r13), so the gap could not be located
with counters; what could be established was that the common bottleneck was in the
**per-32-k-chunk inner loop overhead** (the rescale FMA chain, scale smem reads, sync cadence) —
a property of R1's staging structure itself, not of any single knob. So r6's choice: stop
looking for knobs and **write an execution spec for a structural rewrite**, pinning the entire
next phase (later r7 through r34) to a design.

The spec's seven points, each with an explicit "why":

1. **smem holds RAW bytes only**: A side 40 B pad40 per (token, 32-k chunk) (2×uint4 qs + uint2
   d/ssum) — the pad40 layout is itself a raw format, so fragment words can be read straight out
   of it; B side 144 B Q4KB per (row, 256-k super-block) (9×uint4). At KD=8 the footprint is A
   2×8×64×40 = 41 KB + B 2×64×144 = 18.4 KB + scales ≈ 62 KB → 1 block/SM; KD=4 ≈ 31 KB → 2–3
   blocks/SM. Both depths re-measured — r5 had already shown this trade-off moves.
2. **staging = pure cp.async 16B chunks** (A: 2×16 + 8 B; B: 9×16), commit_group per (kt, buf);
   **zero dequant ALU** on the staging path.
3. **In-register dequantization inside the mma loop**: the math matches `mmq_stage_b` TYPE==5
   (`get_scale_min_k4` per (row, sub), nibble unpack, `__vsubss4`), so the ALU overlaps tensor-
   core issue.
4. **Warp tile stays 32(i)×16(j) × 8 warps on 64×64** — the 4-warp 32×32 variant measured slower
   and broke parity; the spec says outright "do not retry".
5. **Land as `mmq_raw_nt_kernel<TYPE>` beside `mmq_nt_kernel`**, entering prefill_mmq behind the
   `MINFER_MMQ_RAW=1` gate — R1 stays untouched, always A/B-able and revertible; the first cut
   covers q4_K only (79% of MMQ time), with q6_K's KSPLIT=2 structure left for later.
6. **Three-stage parity gate**: `cuda_prefill_mmq_parity` gains a raw-mode arm (same host
   reference) → 7B greedy token identity vs the f16 path → quiet-window A/B (vs R1 MMQ and vs
   the default f16). Parity before performance, and the order is non-negotiable — r5's parity
   hole is the cautionary tale for getting this order backwards.
7. **The quantize pass (129 ms) is follow-up work once the GEMM wins**: it is limited by convert
   bandwidth (~87–102 GB/s) and the serial fmaxf chain's latency; the approach is tree-reduce
   amax + register-packed stores, target ~60–70 ms.

The spec also wrote down two tiers of expected payoff: at GEMM = f16-parity (≥24 TMAC/s), the
MMQ path (quantize ~130 + GEMM ~600) drops the 56 ms convert → ~2670 tok/s; at llama-parity
(GEMM ~480 ms) → wall ~634 ms → ~3250 tok/s (conversion per the spec's anchors at the time).

### 3.2 Key code

**The spec verbatim** (`git show 491eb5c -- docs/CUDA_OPTIMIZATION.md`, design body excerpted):

```markdown
### MMQ structural rewrite — execution spec (P6 r6, for next session)

Goal: mmq GEMM 6.1 TMAC/s (23 ms per ffn_gu call) -> >=24 (f16-GEMM
parity) or ~30 (llama.cpp parity). ...

Design (llama.cpp mmq structure; q4_K first = 79% of MMQ time):
1. smem holds RAW bytes only: A per (token, 32-k chunk) = 40 B pad40
   (2x uint4 qs + uint2 d/ssum); B per (row, 256-k super-block) =
   144 B Q4KB (9x uint4). KD = 8 chunks (256-k) per double buffer.
   Footprint: A 2x8x64x40 = 41 KB + B 2x64x144 = 18.4 KB + scales
   ~ 62 KB -> 1 block/SM at KD=8; KD=4 -> ~31 KB -> 2-3 blocks/SM
   (re-measure both; r5 showed depth beats occupancy for the word-
   staging kernel, raw staging shifts the tradeoff).
2. Staging = pure cp.async 16B chunks (A: 2x16 + 8 B; B: 9x16),
   commit_group per (kt, buf); NO dequant ALU in the staging path.
3. The mma loop dequants IN REGISTERS from smem raw bytes (same math
   as mmq_stage_b TYPE==5: get_scale_min_k4 per (row, sub), nibble
   unpack, __vsubss4) so the ALU overlaps tensor-core issue instead of
   serializing before __syncthreads.
4. Warp tile stays 32(i)x16(j) x 8 warps on 64x64 (the 4-warp 32x32
   variant measured slower AND broke parity — do not retry).
5. Land as mmq_raw_nt_kernel<TYPE> beside mmq_nt_kernel; gate
   MINFER_MMQ_RAW=1 through prefill_mmq so R1 stays intact; q4_K only
   in the first cut (q6_K KSPLIT=2 later).
6. Parity: extend cuda_prefill_mmq_parity with a raw-mode arm (same
   host reference), then the 7B greedy-token-identity check vs the f16
   path, then quiet-window A/B vs R1 MMQ and vs the default f16 path.
7. Quantize pass (129 ms) is follow-up work once the GEMM wins: ...
   tree-reduce amax + register-packed stores, target ~60-70 ms.
```

**The R1 word-staging that the spec replaced** (the q4_K branch of `mmq_stage_b` in the current
tree `src/cuda_kernels.cu`; after r15 dmin is precomputed as ds/dm pairs and the nibble words
stay unsigned — the same branch at r6's time did `__vsubss4` centering in staging; the mechanism
is identical: dequant ALU and expanded words both live in the staging section, serialized ahead
of the barrier):

```cuda
} else if constexpr (TYPE == 5) {   // q4_K: value = d·s·nib − dmin·m (nib unsigned)
    int sb = c >> 3, s = c & 7;
    const uint8_t* blk = W + (size_t)j * ((nb32 >> 3) * 144) + (size_t)sb * 144;
    #pragma unroll
    for (int w = 4 * half; w < 4 * half + 4; w++) {
        uint32_t N = *(const uint32_t*)(blk + 16 + (s >> 1) * 32 + 4 * w);
        uint32_t nib = (s & 1) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
        qb[r * MMQ_WS + w] = (int)nib;          // ← nibble expanded into int words resident in smem
    }
    if (half == 0) {
        uint8_t sc, m;
        get_scale_min_k4(s, blk + 4, &sc, &m);  // ← scale decode also sits in the staging section
        ds[r] = h2f(*(const uint16_t*)blk) * (float)sc;
        dm[r] = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);
    }
```

Segment by segment: each (row, chunk) unit reads a 144 B Q4KB block from global memory, does 4
words of nibble shift/mask, writes 4 expanded words into smem, and then the half==0 thread
decodes the scale — this whole set of ALU and smem writes happens before `__syncthreads`, pure
up-front serial cost ahead of the mma section. The raw-byte scheme compresses this into "a 9×16
B cp.async copy", deferring all dequantization to registers inside the mma loop.

### 3.3 Pitfalls

- **The parity hole's shape was "zero cells"**: the 4-warp tile's `mmq_w80` failure was not
  numeric deviation (the max-diff-82.9 kind of partial-sum misalignment) but entire output cells
  never being written — the hand-written fragment mapping had coverage holes. The lesson from
  this class of error went into spec point 6: structural changes must pass a parity gate first,
  and the parity gate must be designed to expose "zero contributions" (per-cell comparison
  against the same host reference).
- **The attribution trap in wall decomposition**: in the one-shot CLI prefill's nsys timeline,
  the LOAD-phase w16 fill (352 ms) and the prefill wall (890 ms) sit on the same trace; not
  slicing by phase makes it look like an in-wall cost. r5's commit message first wrote "~352ms
  of the 890ms wall"; the spec commit 13 minutes later corrected it to "runs at LOAD time, NOT
  inside the Prefill wall" — the correction was grounded in the vectorized dequant's measured
  null delta.
- **Negative results must not be extrapolated**: KD=4 was negative for the word-staging kernel,
  but the spec did not write it up as a universal conclusion — it wrote "raw staging shifts the
  tradeoff" — and r7's raw kernel re-measuring KD indeed gave a different answer (KD=8 472 vs
  KD=4 440; depth still wins, but the meaning of the margin had changed).

## 4. Verification

- **`cuda_prefill_mmq_parity` sweep**: every quantization sub-shape compared per-cell against
  the same host reference — the 4-warp tile's `mmq_w80` zero cells were caught by exactly this
  gate (defends: fragment-mapping coverage holes / hand-written indexing bugs).
- **suite 169/0**: all green after the reverts (defends: the reverts themselves introducing a
  regression).
- **nsys phase slicing**: the one-shot CLI prefill's w16-fill split measurement, used to
  attribute the dequant pass to LOAD (defends: prioritizing an out-of-wall cost as if it were
  in-wall).
- **Default-path isolation**: everything was measured under the `MINFER_MMQ=1` opt-in; the
  default f16 path was untouched the whole time (2320–2370 tok/s unchanged) (defends:
  measurement changes polluting the production path).

## 5. Results

All measurement/revert, no landed code:

- **KD=4**: 427 vs 438 tok/s — R1-era conclusion holds; occupancy 1→2 blocks/SM cannot buy back
  the staging-depth amortization loss.
- **4-warp 32×32 tile**: 399 tok/s (9% slower than 438) + `mmq_w80` parity zero cells —
  reverted, and marked "do not retry" in the spec.
- **Vectorized q4_K dequant**: null delta — reverted (it is not inside the wall; the scalar
  version was already warp-coalesced).
- **The r6 spec**: became Era C's execution contract. r7–r8 landed the raw kernel per points 1–5
  (472 tok/s), point 7 was cashed in early by r7–r8 (quantize 129 → 74 ms), and point 3's
  "dequant at mma time" plus point 6's parity gate carried through all subsequent MMQ kernel
  work.
- **Veto mechanism**: the retry conditions for KD=4/4-warp are that the staging structure
  changes first (raw-byte conversion zeroes the staging section's ALU, reshuffling the
  depth/occupancy trade-off); the precondition for retrying dequant vectorization is that it
  returns to inside the wall (i.e. abandoning the w16 cache path) — neither happened later, so
  the verdicts stand.

## 6. Lessons

1. Re-reading a profile must be done against the **current** code and phase slicing — the 352 ms
   dequant pass was never inside the wall, and one mis-attributed number nearly defined the
   whole optimization direction.
2. Write an execution spec with parity gates before any large rewrite; the spec must state what
   NOT to retry (the 4-warp tile's "do not retry" saved every later re-argument cost).
3. Depth vs occupancy has no universal answer — it is a function of the staging section's
   weight; when the staging structure changes, re-measure instead of inheriting the old
   conclusion.
4. Parity gate before performance gate: zero-cell-class structural errors should be caught
   before any meaningful timing measurement happens.

---

← 11 · [Index](./README.md) · 13 →
