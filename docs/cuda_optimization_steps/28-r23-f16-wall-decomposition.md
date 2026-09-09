# 28 · r23 — f16-path whole-graph wall decomposition + FA_TKV occupancy raise (MEAS-ONLY + REVERTED)

> **Result**: default f16 path (2659-token prefill, quiet window 2285 tok/s) whole-graph decomposition:
> **GEMM 74%** (gate+up 37.8% @ mem SOL 85% / down 27.5% / q+o 7.5%), FA 7.2%,
> convert 6.0%, swiglu 5.6% — llama's split gives the byte-width gap as 59.5 vs 39.5 µs/GMAC.
> The FA_TKV 64→32 raise: occupancy 16.7 → 32.68%, kernel −6.7%, but wall clock **−0.3% <
> +3% bar**, reverted.
> **Commit**: `e8c348d` (docs/record commit; the FA_TKV raise's code was reverted, **no reachable code
> commit** — this doc's FA code comes from the current tree, i.e. the shape after r48's FAP2 landed, noted inline).
> **Date**: 2026-09-03.

## 1. Background — where things stood

r21/r22 had probed the MMQ wide kernel's A side to the bottom in two consecutive rounds: reshuffling lost (stall-mass conservation), deleting conflicts won
(+1.4%), KD=8 stuck at ~1330 tok/s. But one increasingly awkward fact: **MMQ is still
opt-in to this day** (`MINFER_MMQ=1`) — what users run by default is the f16 path (8p's resident f16 weight cache +
dequant-in-GEMM), ~2285–2370 tok/s (quiet window 2285), vs llama-bench at the same anchor
1.43×.

The campaign had invested eleven rounds (r12–r22) in the MMQ kernel, yet "how much can MMQ actually win" had always been an extrapolation rather than a
measured wall-clock decomposition. r23 decided to pause adding levers and first do two things:

1. **A complete whole-graph wall decomposition of the default f16 path** — nsys per-launch bucketing + ncu SOL
   cross-checks, charging every millisecond to an arithmetic class. This answers "how much wall clock is the MMQ campaign's premise (the byte-width gap)
   worth", and also "what other big non-MMQ items remain on the f16 path".
2. **Raise FA's occupancy in passing**. In the decomposition FA is 7.2% with occupancy only 16.7%
   (the `FA_TKV=64` set in the 8n era pushes smem past a single SM's one-block headroom). A one-line define,
   FA_TKV 64→32, halves the KV tile's smem and doubles the block count — a cheap
   pilot testing whether "occupancy for wall clock" holds on FA.

This round's nature is **measurement first**: the decomposition itself (MEAS-ONLY) is the main deliverable, and the FA_TKV raise is a
piggybacked experiment.

## 2. Principle — the GPU mechanism

### 2.1 Arithmeticizing the campaign premise

llama's MMQ and our f16 path do the same MACs; the difference is **how many bytes flow per weight**:

```text
f16 path:  B side 16 bit per weight (f16)
q4_K MMQ:  B side ~4.5 bit per weight (raw nibble stream + per-super-block scale/dmin)
           + A side 8 bit per token per k (q8) — same class on both sides
```

Measured split (quiet window, same anchor): llama's prefill GEMMs total 597 ms (**39.5 µs/GMAC**);
our f16 path GEMMs 59.5 µs/GMAC + 73 ms of f32→f16 convert pass (llama pays ~0,
its quantization happens in-kernel). **59.5 / 39.5 ≈ 1.5×/GMAC** — not the 16/4.5 = 3.5× theoretical extreme,
because MMQ also pays nibble expansion, scale rescaling, and int8 mma's extra supporting instructions; but at
gate+up's 85% mem SOL, the B-stream byte width is the source of the 1.5×. This is the entire MMQ
campaign's founding premise, pinned to wall-clock numbers for the first time. Corollary: **if the MMQ GEMMs reach llama's
GEMM level, the f16 default path should reach ~2900 tok/s** — the target of every MMQ round after
r23.

### 2.2 The gains and costs of halving FA_TKV

8n's FA kernel caches `FA_TKV × hd` K/V tiles per block; the smem bulk is KV staging
(the precise figure from r46's later audit: 69.38 KB total → 1 block/SM → occupancy 16.64%,
~2 warps/scheduler, nowhere for latency to hide). TKV 64→32 halves the KV-side bytes (~43.8 KB) →
2 blocks/SM → occupancy 32.68%.

But the other side of the gain is **the per-tile fixed cost paid twice as often**: the total KV column count is unchanged, so halving the tile means the
k loop iterates ×2, each iteration paying the fixed overhead of staging + barriers + the softmax rescale chain.
What occupancy buys is latency hiding; what the fixed cost eats is the actual gain — r23's result (kernel −6.7%,
wall −0.3%) is the record of these two nearly cancelling.

### 2.3 The hidden mine of shrinking the tile: lane masks

The wmma accumulator fragments' lane→(row, col) mapping is derived from the tile geometry. After TKV is halved, each
warp's accumulator fragments go from 4 to 2, and lanes 16–31's **in-tile column numbers** no longer
coincide with the old geometry — the validity mask must contain both the **local bound** (`c0/c1 < FA_TKV`) and the **global bound**
(`kt + c0 < kv_end`); missing either, the tail tile feeds garbage scores from out-of-range columns into the softmax.
r23's integration was caught mid-way by the verification system with exactly one such real bug (the record's own words: "TKV=32 lanes 16–31
must be masked by `c0/c1 < FA_TKV`, not just `kt+c0 < kv_end`").

## 3. Implementation

### 3.1 The decomposition method: per-launch bucketing + SOL cross-checks

- **Full nsys trace**: the 2659-token prefill's default f16 path, per-launch timing bucketed by arithmetic class
  (gate/up, down, q/o, k/v, FA, convert, swiglu, add/rms/rope, host gap);
- **ncu SOL**: spot-check a representative kernel per class against its throughput ceiling (mem SOL, SM busy),
  defending against the misreading "nsys bucketed correctly, but that kernel itself runs unsaturated";
- **Co-tenant pollution cleaning**: outlier launches caused by co-tenant load within the window are removed; all conclusions re-verified on the quiet window
  (2285 tok/s reference).

**Graph-structure discovery** (an unexpected harvest before the decomposition): the prefill is **27 full layers + one nt=1
TAIL** — q6_K's lm_head (1.45 TMAC = 9.6% of the whole model's MACs) runs only once in the tail,
already "tail-priced"; the graph **contains no** [nt, vocab] logits GEMM in the wall clock. This overturns the
intuition "lm_head is a prefill bulk item" and means prefill optimization only needs to watch the 27-layer body.

### 3.2 The FA_TKV raise: one define and one mask

The raise itself is `#define FA_TKV 64` → `32` (all downstream sizes symbolic, no other changes); the real
work was the mask fix (§2.3). The raise's code was reverted and survives in no commit; **today's
tree's FA kernel is already the post-r48-FAP2 shape**, but the "local + global double predicate" mask structure survives verbatim and
can serve as a living specimen of this fix class:

```cuda
/* src/cuda_kernels.cu — fa_prefill_f16kv, today's tree (post-r48 FAP2 shape) */
float mnew0 = -INFINITY, mnew1 = -INFINITY;
#pragma unroll
for (int q = 0; q < FA_TKV / 16 * 4; q++) {
    /* valid = causal (kv <= query pos) AND within the stored KV range
     * (rows >= kv_end are zero-staged and must NOT contribute). */
    bool v0 = (gcol[q] <= qpos0) && (gcol[q] < kv_end);   /* global bound */
    bool v1 = (gcol[q] <= qpos1) && (gcol[q] < kv_end);
    if (v0) mnew0 = fmaxf(mnew0, sm[q]);
    if (v1) mnew1 = fmaxf(mnew1, sm1_[q]);
}
```

Today's tree tile geometry (`FA_TKV 32` is the resident value; r50's attempt at 32→16 was rejected, and r57 states
"stays 32"):

```cuda
/* src/cuda_kernels.cu:3936 — today's tree FA tile width */
#define FA_TKV 32
```

Worth spelling out the causality: r23/r46's two independent "64→32 raises" were both reverted for missing the wall-clock bar, yet today's
tree has TKV=32 — it entered and solidified as the new kernel's geometry in r48's FAP2 rewrite (register-resident softmax, deleting the S/P smem round trip
entirely, 69.38 → 34.82 KB). **The occupancy wall was ultimately climbed not by shrinking the tile but by deleting another block of smem**; r23's pilot supplied the evidence that
"2 blocks/SM is worth wanting", and r48 supplied the path that pays no fixed-cost tax.

### 3.3 Pitfalls

- **The lane mask bug**: see §2.3 — shrink the tile geometry and validity masks derived from the old geometry quietly
  fail; the global bound does not stop lanes whose "local column is out of range but global column still legal".
- **Co-tenant outliers**: a decomposition done on a polluted window skews every bucket's share; outlier removal +
  quiet-window re-verification are hard steps (after r59b this became the whole campaign's standard protocol).
- **The misleading nature of "FA is 7.2%"**: a small share does not mean a small lever — when r47 later re-decomposed in the converged regime,
  FA was already the 10.2% #1 structural residual. Shares travel with the baseline (decompositions have a shelf life).

## 4. Verification

- **Bucket-conservation check**: the sum of the arithmetic classes' launch times ≈ nsys wall clock (GPU-busy basis),
  defending against bucket omissions or double counting;
- **ncu SOL cross-check**: each class's representative kernel's mem/SM utilization reconciled against the physical expectation of
  "should it saturate" (gate/up 85% mem SOL = reasonable saturation; down's 20%/MAC asymmetry is an
  anomaly, booked as such);
- **Quiet-window re-verification**: all shares re-measured on a co-tenant-free window;
- **The FA_TKV raise's full gate set**: parity + greedy identity + interleaved A/B — it was exactly this gate set that
  caught §2.3's mask bug mid-integration (a real bug stopped by the gates, not by the naked eye).

## 5. Results

### 5.1 f16-path whole-graph decomposition (2659-token prefill, quiet window 2285 tok/s)

| Arithmetic class | share of wall | Notes |
|---|---:|---|
| gate+up GEMM | **37.8%** | mem SOL 85% (saturated) |
| down GEMM | **27.5%** | 20%/MAC slower than gate/up — cause of the asymmetry unknown, booked |
| q / o GEMM | 7.5% | — |
| k / v GEMM | 1.4% | — |
| FA attention | **7.2%** | occupancy 16.7% (1 block/SM) |
| convert f32→f16 | **6.0%** | llama pays ~0 (quantization in-kernel) |
| swiglu | 5.6% | DRAM peak |
| add / rms / rope | ~5% | — |
| host gaps | 0.8% | — |

**GEMMs total 74%** — the f16 path's wall is GEMM. q6_K layers carry **no** per-MAC
penalty on the f16 kernel (the w16 resident cache smooths out the type difference), i.e. the f16 path is insensitive to weight type; all of MMQ's gain
space comes from byte width. Against llama's split (GEMM 597 ms @ 39.5 µs/GMAC vs our
59.5 µs/GMAC + 73 ms convert): **the prefill gap = the GEMM byte-width gap**; the raw q4_K
B stream (4.5 bit/w) at 85% SOL overwhelms f16 (16 bit/w) at ~1.5×/MAC — the campaign premise
holds, and yields the quantitative target: **MMQ GEMMs to llama's level → ~2900 tok/s**.

### 5.2 The FA_TKV 64→32 raise (REVERTED)

| Metric | before (TKV=64) | after (TKV=32) | Δ |
|---|---:|---:|---|
| occupancy | 16.7% | **32.68%** | 1 → 2 blocks/SM |
| FA kernel | — | — | **−6.7%** |
| whole-prefill wall clock | — | — | **−0.3%** (bar +3%) |

**Veto mechanism**: occupancy doubled and the kernel got 6.7% faster, yet the wall moved only 0.3% — **the fixed cost of the
k loop's doubled iteration count ate the latency-hiding gain**; and FA held only 7.2% of the wall at the time, so a −6.7%
kernel amortizes to noise level across the whole graph. Reverted (but the mask-fix knowledge gained en route was booked). FA's 2.5×/layer
gap to llama is **structural**: llama's FA keeps 128-wide KV tiles — shrinking the tile is not
the answer to that gap.

**Retry conditions**: tile/occupancy-class levers are worth touching again only when (a) FA dominates the wall clock and
(b) the per-tile fixed cost has been cut (r48's register softmax is exactly (b)). Two re-tests confirmed this verdict:
r46 (TKV 64→32 + S/P padding, −11% kernel / +0.27% wall) and r50 (TKV 32→16,
the occupancy gain offset by sync overhead + breaking bitwise identity);
r48 FAP2 reached 2 blocks/SM by "deleting the S/P smem" and landed +5.6% — same wall, different road.

## 6. Lessons

1. **Byte-width arithmetic can adjudicate the campaign premise before any kernel exists**: the ledger of 59.5 vs 39.5 µs/GMAC +
   4.5 vs 16 bit/w is worth more than any single-point kernel experiment done first.
2. **Decompositions have a shelf life**: re-run after each convergence (r47 therefore re-judged FA from a 7.2% "small item" to the
   #1 structural residual); hidden taxes (like convert's 6.0%) must appear in the same table as gains.
3. **Shrink the tile geometry and the lane masks must be re-derived**: local bound + global bound are two predicates; missing either is
   silent numeric pollution of the tail tile (r46/r50 inherited this lesson).
4. **Occupancy is not a free lunch**: the fixed cost of ×2 iterations offsets latency hiding; cut the fixed cost first
   (r48), then talk tile size.

---
← [27-r22-qa8-xor-swizzle](27-r22-qa8-xor-swizzle.md) · [Index](./README.md) · [29-r24-scheduling-ladder](29-r24-scheduling-ladder.md) →
