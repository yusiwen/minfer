# 32 · r29 — NB kd-loop unroll: 2 blocks/SM lets integer-ALU pruning move the wall clock for the first time (LANDED)

> **Result**: 7B q4_K whole-prefill 1387.9 → 1426.8 tok/s (**+2.80%**, interleaved 5/5 positive) — a
> one-line `#pragma unroll` change: integer ALU −25% (49.3M → 36.8M warp instructions), total inst
> −6.5%, 123 regs / 0 spill, the 2-blocks/SM occupancy preserved intact. r25 measured the same pruning
> at 1 block/SM: +0.37/+0.49%, wall-clock inert.
> **Commit**: `bfe6bba`. **Date**: 2026-09-04.

## 1. Background — where things stood

Era C's target had been fixed at r6: lift the q4_K MMQ GEMM from 6.1 TMAC/s to ≥24 (f16-path parity)
and ideally ~30 (llama.cpp parity). r25 ran a SASS opcode-class census that reconciled minfer's per-
tile warp instruction stream against llama.cpp's: 445,544 vs 355,758 total, a +25.2% surplus — and
that surplus is **100% supporting instructions**: integer ALU +69.5k/tile (77% of it), fp32 rescale
FMUL +15.2k, conversions +14.3k, while IMMA and FFMA are exactly equal on both sides (114,688
FFMA/tile on each). In other words: not one extra instruction computes a MAC; everything extra is
"the overhead of carrying the MACs".

At the time, r25 also tried the kd-unroll in passing: integer ALU −38%, total inst −9.7% (the
surplus halved), but the wall clock moved only +0.37/+0.49% — below the +1.5% landing bar — and it
was reverted. The conclusion then was: the wide kernel's 98 KB smem admits only 1 block/SM, ~2 warps
per scheduler, latency completely unhidden — the kernel is issue/occupancy-bound, not instruction-
count-bound. **The instruction stream was pruned, but the issue slots were never waiting for it.**

r28 inverted that verdict: the one occupancy lever r13–r25 had never touched was smem itself. The
new Direction-A kernel `mmq_raw_nb_kernel` squeezes the B-side smem to raw-packed (a 2-nibbles/byte
qs plane), accepts a small in-loop B-unpack cost, and drops smem to 45,056 B → **2 blocks/SM**.
Result +2.56% (1375.2 → 1410.4), with ncu confirming warps_active 16.17 ≈ 4.04 warps/scheduler,
long_scoreboard 2.92 → 2.03, issue_active 25 → 37.36%. Occupancy had been bought.

The NB kernel's own pedigree is worth recording: its B-fragment nibble layout was not guessed — r28
derived it from the wide kernel's validated ldmatrix path and ran a standalone byte-equivalence
check before integration (all 8 sgs × 32 lanes × 4 registers, 0 mismatches); at landing it is
double-gated (`MINFER_MMQ_RAW_NB=1` and kd==8, with a clean fallback to the wide kernel for K dims
that are not multiples of 8), the wide kernel remains the default raw path, and the two are byte-
equivalent. That is, r29's measurement subject is a kernel already wrapped in two independent
verifications — instruction pruning no longer needs to worry about layout correctness. That is the
precondition that let it "land in one line".

Landing-bar context: the wall bar stays at +1.5%, judged on 7B whole-prefill wall clock. The NB
kernel serves the KD=8 shape of q4_K MMQ, so a kernel-level gain must be amplified by its share of
the wall to clear the bar — r28's +2.56% showed the amplification factor is sufficient, and same-
order kernel-level improvements were worth continuing to mine.

So r29's question became natural: **once occupancy is fixed, is r25's "wall-inert" instruction
scissors alive again?** The census was re-run on the NB kernel first — and the answer still had the
original shape: integer ALU 2.85× llama per MAC (2.142 vs 0.751 e-3/MAC), FFMA still level on both
sides; stalls concentrate on the shared path (mio_throttle 18.89% + long_scoreboard 16.80%). The
same class of surplus, the same lever — it only needed to be verified once.

## 2. Principle — the GPU mechanism

The NB kernel's K-dimension main loop runs `KDR=8` 32-token chunks per k-tile (`kd = 0..7`). Each
chunk's iteration body does five things, and four of them have addresses that depend on the runtime
value of `kd`:

1. A fragments: one `ldmatrix.x4` for each of the 4 16-token groups, base `qat = qa8 + kd*64*32`;
2. B unpack: read 2 raw 32-bit words from `qb_raw`; `p = sg>>1` (sg = `c&7`) selects which pair of
16-k groups, `is_hi = sg&1` selects low or high nibble;
3. 8 independent mma chains (4 A-frags × 2 B-frags);
4. sds scale read: a float4 from `sds + kd*128 + ...`;
5. sda scale read: from `sda_q + kd*64 + ...` (sda/sds are the two sides' A/B scale data planes — A
side per-token d|ssum, B side per-row d·sc|−dmin·m, fully defined in r31 — here you only need to
know they are shared data read on every iteration).

Without unrolling, all these offsets are runtime integer arithmetic: `sg = c&7`, `p = sg>>1`, `is_hi
= sg&1`, the multiply-add chain of the `kd * constant` bases — every LDS/LDSM address hangs off the
end of this integer dependency chain, and every step on the chain is a source of the 77% surplus
from r25's census. `#pragma unroll` lays the 8 iterations out statically, and each chunk's `is_hi`
selection, smem bases, and boundary checks all become **compile-time constants**: the address
arithmetic collapses into immediate offsets and the integer chain disappears; at the same time ptxas
can see through, for the first time, that "adjacent chunk pairs read the same pair of raw words"
(even kd pairs share the same `p`), and CSEs the raw-word load across chunks — which is exactly the
SWAR equivalent that r30 goes on to test.

Why does the wall clock move this time? The occupancy ledger is unchanged: 256 threads × 2 blocks =
16 warps/SM = 4 warps per scheduler. In the r25 era there were only ~2 warps per scheduler, idle
issue slots were the norm, and instruction count was not the bottleneck; now, under 4-warp issue
pressure, **the supporting instructions themselves start competing for issue slots** — every
instruction pruned lets a real IMMA/FFMA issue one step earlier. Occupancy is the precondition for
instruction pruning; the order cannot be reversed. The unroll also adds no register pressure (the
compiler merely folds constants; it does not need more live registers), so 123 regs / 0 spill is
preserved intact — that is the guarantee that occupancy is not bitten back.

Dissect one chunk's iteration body into an instruction list and the location of the surplus is
obvious. Per chunk per lane: 4 ldmatrix.x4 on the A side, 2 LDS.32 on the B side, 8 independent mmas
(the full 4 A-frag × 2 B-frag combination), two LDS.128 for sds (float4, the nh=0/1 od rows), four
LDS.64 for sda, plus the rescale FFMA/FMUL pairs. The MAC portion (mma + rescale) is level with
llama class by class — the census's own words were "114,688 FFMA/tile, equal on both sides" — and
the extra integer ALU is all on the address side: each iteration recomputes the kd-dependent offset
for each of the four bases qat/qb_raw/sds/sda, plus the bit ops and boundary checks around
sg/p/is_hi. After unrolling, this arithmetic collapses into immediates and disappears from the SASS
instruction stream outright.

The issue-slot accounting closes too: the int-ALU class is 77% of the surplus, about ~19% of the
whole instruction stream; cutting 25% of it ≈ ~4.8% of the whole stream, the same order as the
measured total inst −6.5% (including the knock-on effect of address folding). Under 4-warp-per-
scheduler issue pressure, that ~5% release of issue slots lands precisely in the gaps of the IMMA-
dense segments — matching the +2.80% wall-clock move.

## 3. Implementation

### 3.1 Design choices (the candidate ladder: three vetted, two shot down, one landed)

After the census re-run, r29 arranged the candidate levers into a ladder, every one verified before
acting:

- **(a) PRMT nibble extraction — REFUTED**. A standalone sm_120 micro-benchmark showed `PRMT` still
needs shift+mask alongside it to extract a nibble and cannot beat the existing SHF+LOP3 combination;
the census also showed the kernel was already emitting 0 PRMT. The instruction that looks better on
paper does not exist on the hardware.
- **(b) Software pipelining of the B raw-word load — NEUTRAL, reverted**. Moving the B raw-word load
one stage ahead into double buffering measured wall-neutral, and registers inflated 123 → 177. The
occupancy ledger could be computed before the measurement: GB10 has 64K registers per SM; 123 regs ×
512 threads (2 blocks) ≈ 63K, already nearly full; 177 regs × 512 ≈ 91K, and the second block would
inevitably be squeezed out — even with a neutral wall clock, the register ledger alone is enough to
veto. Risking occupancy for a neutral gain: voted down on both counts.
- **(c) LDSM A-fragments — already in place**. The r14/r22 legacy; the NB kernel was born with it.
- **LANDED: the kd-loop `#pragma unroll`** — r25's scissors, a one-line change; the census is its
source of legitimacy, the zero register increment is its safety margin.

The essential reason for choosing it: it is the only lever on the ladder whose "mechanism is census-
confirmed and whose cost is structurally zero". The other two were either falsified by the hardware
or carried an occupancy side effect.

### 3.2 Key code

The change itself is one line (`git show bfe6bba -- src/cuda_kernels.cu`: 1 insertion). Below is the
landed form (current tree `src/cuda_kernels.cu`), annotated with which quantities become constants
because of the unroll:

```cuda
// src/cuda_kernels.cu — mmq_raw_nb_kernel main K loop (current tree, post-r29)
for (int kt = 0; kt < nktile; ++kt) {
    if (kt > 0) RAW_STAGE_NB(kt);
    __syncthreads();

    #pragma unroll                      // ← r29: this line only (bfe6bba)
    for (int kd = 0; kd < KDR; kd++) {
        const int c = kt * KDR + kd;
        if (c >= nchunk) break;         // boundary guard kept; predicated per iteration after unroll
        const int sg = c & 7;           // ← compile-time constant (kd statically known)
        const uint8_t* qat = qa8 + (size_t)kd * MMQ_NBI * 32;  // ← folded to an immediate offset
        ...
```

The part that really eats integer ALU is inside the iteration body. After unrolling, the B unpack's
`p/is_hi` and both load bases, the A LDSM `G[g]` offsets, and the sds/sda `kd*` offsets are all
fixed at compile time:

```cuda
        // B fragments: raw-nibble in-loop unpack (after unroll, p/is_hi are constants,
        // and the two chunk tiers (2p, 2p+1) read the same word pair → ptxas cross-chunk CSE, see r30)
        {
            const int p = sg >> 1, is_hi = sg & 1, lm3 = lane & 3;
            const unsigned M = 0x0F0F0F0Fu;
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const int jj = j0w + nh * 8 + (lane >> 2);
                const uint8_t* qs = qb_raw + (size_t)jj * 128;
                const uint32_t* q0 = (const uint32_t*)(qs + p * 32 + lm3 * 4);
                const uint32_t* q1 = (const uint32_t*)(qs + p * 32 + 16 + lm3 * 4);
                uint32_t v0 = *q0, v1 = *q1;
                b[nh][0] = (int)(is_hi ? ((v0 >> 4) & M) : (v0 & M));
                b[nh][1] = (int)(is_hi ? ((v1 >> 4) & M) : (v1 & M));
            }
        }
```

### 3.3 Pitfalls

- **The un-warmed ~1206 outlier**: the first batch of interleaved measurements produced one outlier
at ~1206 tok/s, traced to a GPU power-state artifact under the un-warmed harness (clock ramp not
settled); it vanished once warmup was added. From then on the campaign's A/B interleaved protocol
always includes warmup — check the measurement environment before suspecting the code.
- **unroll coexisting with `break`**: the loop body contains `if (c >= nchunk) break`, and `#pragma
unroll` still takes effect (the 8 iterations are laid out statically and the guard is predicated per
iteration); the boundary does not need to be rewritten for a "constant trip count" — do not
complicate tail-block logic for unrollability's sake.
- **Do the software-pipelining register accounting first**: candidate (b)'s threat of 177 regs to 2
blocks/SM should have been visible before measuring; a "neutral + register-inflating" lever is a
liability in an occupancy-sensitive kernel.
- **The cost structure of a "one-line change"**: the diff is 1 line (+`#pragma unroll`), but the
forensics took three rounds — the census re-run, the PRMT micro-benchmark, the software-pipelining
trial. This class of lever's real cost is in the burden of proof, not the edit; an unroll without
the proof is a gamble, not an optimization.

## 4. Verification

- **Parity (NB-active) 1/0**: `cuda_prefill_mmq` cross-check passes — defends against the B-fragment
mapping being broken by the unroll's reordering (a layout error is ~1e0 scale, f32 rounding is 1e-5
scale; one test tells them apart).
- **greedy-32 byte-identical**: the greedy 32-token output is byte-identical — defends against any
change in floating-point summation order (the unroll only touches address arithmetic and must not
touch summation order).
- **ptxas ledger**: 123 regs / 0 spill, warps_active 3.94 — defends against register inflation
knocking 2 blocks/SM back to 1, which would destroy this lever's entire precondition.
- **Suite 166/0/3**: the campaign-wide full regression gate (the same standard for the r28/r30/r31
steps). The unroll touches only one function in the NB kernel; the suite gates the whole engine's
behavior surface — defends against "local optimization, global regression".
- **Interleaved 5-pair (with warmup)**: +2.80%, 5/5 positive; adjacent pairs +1.88% — defends
against single-point noise and machine drift. 5/5 positive has probability 2⁻⁵ ≈ 3% under the "no
real difference" null hypothesis — sign-test-level directional evidence, not just a nice-looking
mean.

## 5. Results

| Metric | r28 baseline | after r29 |
|---|---|---|
| whole-prefill (interleaved 5-pair median) | 1387.9 | **1426.8 (+2.80%)** |
| integer ALU (warp instructions) | 49.3M | **36.8M (−25%)** |
| total inst | — | **−6.5%** |
| regs / spill | 123 / 0 | 123 / 0 (unchanged) |
| warps_active (per scheduler) | ~4.04 | 3.94 (2 blocks/SM kept) |

The control group is the same lever on r25's wide kernel (1 block/SM): int ALU −38%, total inst
−9.7%, wall-clock only +0.37/+0.49%. **The same scissors, once occupancy moved 1 → 2 blocks/SM, went
from inert to +2.80%** — r25's census conclusion ("wall-inert ≠ the class does not matter") and
r28's occupancy conclusion converge here: the two levers are not independent items but ordered
moves.

One more output that does not enter the comparison table but did enter the follow-up agenda: the
cross-chunk CSE that unrolling lets ptxas perform (the B raw words are read-once — loaded once per
kd-pair, shared by the lo/hi uses) is clearly visible in the SASS. That became r30's test subject —
whether the SWAR word-granular unpack proposal still had headroom (answer: none; see the next doc).

Baseline-convention note: this step's 1387.9 → 1426.8 holds only within the same session window;
r31's control baseline reads 1424.10 rather than 1426.8, because machine state drifts across
sessions (the master-table reading convention: all A/B numbers are measured **interleaved within the
same window**; absolute values must not be subtracted across windows).

## 6. Lessons

1. **Occupancy unlocks instruction pruning, not the other way around**: before the issue slots are
contended, supporting instructions are free; buy occupancy first, then cut instructions — in the
wrong order, both come up empty.
2. **The census is the lever's source of legitimacy**: reconcile first to confirm which class is
over-represented, then operate on that class — what separates r29 from r25 is not better pruning but
better occupancy.
3. **Vet the candidate ladder in layers**: the micro-benchmark falsified PRMT (the hardware shortcut
does not exist), the register accounting vetoed software pipelining (an occupancy side effect), the
census supported the unroll — every rejection has a concrete mechanism, not guesswork.
4. **Interleaved A/B must include warmup**: the un-warmed outlier was a power-state artifact, not a
code signal.

---
← [31-r28-nb-kernel-2blocks](31-r28-nb-kernel-2blocks.md) · [Index](./README.md) · [33-r30-swar-unpack](33-r30-swar-unpack.md) →


