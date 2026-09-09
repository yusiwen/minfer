# 29 · r24 — The scheduling-structure ladder (REVERTED; the +1.5% whole-prefill landing bar calibrated here)

> **Result**: all three scheduling-structure rungs negative — tile-order swizzle: A-hot transposed −2.3%,
> G-grouped −2.4% (default x-fastest/B-hot 1370.0 optimal); persistent blocks −3.3%
> (the non-persistent u-loop also −3.3%, persistent occ=1/occ=2 ≈ identical). All reverted.
> This round's real durable output is **calibration**: the whole-prefill landing bar changed from an absolute ≥1350 tok/s to
> **relative ≥ +1.5% (vs the same-window re-measured baseline)** — used ever since by r25/r28/r31/r46/r55/D3a.
> **Commit**: `d90b3e9` (docs-only record; r24's experiment code never became a code commit, see below). **Date**: 2026-09-04.

> **Forensics note (STYLE rule 0)**: r24's ladder code (the swizzle bijection + persistent grid) is an
> in-session "implement → measure → revert, never committed" artifact — `d90b3e9` contains only the 64-line
> `docs/CUDA_OPTIMIZATION.md` record, and a full git-history search finds no r24 code commit. This doc's code
> excerpts are therefore taken from the **current tree** throughout (Grep + Read bounded ranges): the current-tree code shows what the ladder
> acted on (the wide kernel's grid geometry and tile walk), while the ladder itself is presented as **clearly-labeled narrative pseudocode**.

## 1. Background — where things stood

By the time r23 closed, the q4_K MMQ line's shape was: `mmq_raw_wide_nt_kernel` (128 tok × 128 od wide
tile, 8 warps × 32×32 warp tile, r14's ldmatrix B fragments, r20's split-phase A
staging, r22's qa8 XOR swizzle), with the same-window baseline at KD=8 in the **~1330 tok/s** band
(r20 landed 1317.7→1319.9, r22 1329.6). r13–r23, eleven rounds, turned over both axes —
"how many instructions are issued per MAC" and "where the stalls land": r17 proved pure per-MAC instruction cuts pay
~0 wall clock at SM% ~30, r21 proved stall-mass conservation, r23 decomposed the whole f16-path wall and
confirmed GEMM ~85% is the only lever.

One last **untried structural family remained: scheduling** — touching neither instructions, nor bytes, nor dependencies, only
"which blocks run when". It has three natural rungs:

1. **tile-order swizzle**: change the block-index→tile mapping (a bijection on the grid) so adjacent
   blocks share different operands;
2. **persistent blocks**: one block resident per SM, a software loop walking the tile list, eliminating
   the wave-quantization tail;
3. **full stream-k**: split along K + a fixup pass write-back (llama's plan; r13 had already measured the launch-shape-level
   equivalent: wash + fixup +34 µs).

Meanwhile a metrological problem exploded this round: the box **drifted faster**. The KD=8 re-measured reading was
**~1385–1391**, while the recorded band was ~1330 (r20/r22's landing numbers). This is not progress —
master-table footnote 2 states it plainly: the r12–r25 rounds interleaved on boxes drifting −9% to +38%;
**only same-window deltas mean anything**. But this directly sentenced the then-current working bar: the campaign had been using
**absolute ≥1350 tok/s** as the landing line (r18 "Bar ≥1350 decisively missed", r20
"landed despite missing the ≥1350 bar", r22 "the combined ≥1350 bar not
reached") — with the baseline itself already at 1385+, a zero-work binary could "pass the line".
The absolute bar's premise (a stable box) had collapsed, the bar had to be recalibrated, and r24 happened to be the
round with "no result to protect" — the most suitable round for doing exactly this.

## 2. Principle — the GPU mechanism

### 2.1 Why the default order is B-hot: the grid geometry's arithmetic

The wide-tile kernel's grid is two-dimensional: `blockIdx.x` is the **token tile** (i0) and
`blockIdx.y` is the **od tile** (j0) — CUDA unrolls x fastest, so **consecutive blocks on the
same od column share the same B (weight) panel**, while each block's A (activation) token tile is
distinct. That is "default x-fastest / B-hot":

```text
grid = (ceil(nt/128), ceil(od/128))
  blockIdx.x = token tile  (x, fastest)  → consecutive blocks: same j0, new i0
  blockIdx.y = od tile     (y, slow)     → the weight panel changes only when y changes
```

Instantiating the layer-0 q-proj GEMM (nt=512, od=id=3584): grid (4, 28) = **112
blocks ≈ 2.33 waves** (at ~48 scheduling slots per SM, 112/48 = 2.33; llama's same GEMM uses
a stream-k grid (48,1,1) + fixup (48,4,1)). Each od column has 4 blocks taking turns re-reading
the same B (per kt phase a qb8 panel of 128 od × 8 chunks × 48 B ≈ 49 KB, 258 KB over the full K
range) — small panel, short reuse window, **naturally absorbed by L2**; A, meanwhile, is a fresh
token tile per block, the streaming side. The x-tile round (row 26) already measured the two streams' sizes:
**A re-read 327 MB vs B re-read 152 MB, A at 2.1:1**.

### 2.2 Why the swizzle can only lose

Tile-order swizzle changes only the **temporal locality window**; the total re-read volume is a schedule-invariant quantity
(how many bytes each tile must read is decided by the tile geometry). Three candidates:

- **A-hot transposed** (x/y swapped): consecutive blocks share A tiles. But A is the dominant stream at 2.1× —
  making the dominant stream "shared-resident" stretches its reuse window into conflict with the other waves,
  while B was already free (L2-resident); there is no gain to trade. Measured −2.3%.
- **G-grouped**: a third bijection (group-clustered traversal), likewise creating no new operand sharing; measured
  −2.4% (the master-table row records −2.3/−4.7%, a cross-window basis).
- **Default B-hot 1370.0**: this window's optimum. Mechanism: the weight panel is small (the key difference between MMQ and an ordinary GEMM —
  q4_K weights are 4.5 bit/weight, so the 128-od B panel is far smaller than the 128-token
  A panel), the L2 residency cost ≈ 0, and the exclusive A stream stays one-step-one-tile, naturally pipelined.

The structural reason behind the conclusion: **at 1 block/SM and ~2 warps/sched occupancy, the kernel is
latency-bound, not byte-bound** (r21's "stall-mass conservation" already falsified the byte side once).
Scheduling can change only L2 temporal locality, and both streams' temporal locality under the default order is already
no bottleneck — there is nothing separable.

### 2.3 Why persistent blocks have no tail to eliminate

A persistent grid = launch exactly num_sms resident blocks, each software-looping over a strided tile
list. It pays off only under two premises: (a) a wave-quantization tail exists —
the fractional part of ceil(waves) leaves the last batch of SMs idling; (b) the per-block fixed cost (smem zeroing,
attr setup) is a significant share. Here 112 blocks ≈ 2.33 waves, but the GPU's launch is **pipelined,
not lockstep waves** — as soon as a block of the previous grid exits, the next grid's block fills in immediately;
the 0.33-wave "tail" was never idle. With no tail, persistent merely replaces the overlap the hardware block
scheduler does for free with a software loop doing it itself, plus the loop's address arithmetic.
Measured: **the non-persistent u-loop alone is −3.3%** (merely changing one-launch-one-tile into looping over tiles),
and persistent occ=1 / occ=2 ≈ identical — no difference even along the occupancy dimension.

### 2.4 Why stream-k (rung 3) was vetoed before any code

Full stream-k splits along K across blocks, and the partial sums need a fixup pass re-ordering
**floating-point accumulation** (llama pays +34 µs for this). This campaign's hard gate is greedy-32 byte-for-byte
identity — a floating-point re-order necessarily flips some argmax blade at some step. So rung 3 is not "never got
around to it" but **vetoed a priori by the gate**: r24's bijection rungs were deliberately chosen as bit-identical shapes;
stream-k is the only one mathematically guaranteed to break identity, not worth spending implementation cost to hit a known wall.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **A bijection, not a reordered loop**: rung 1 uses the env var `MINFER_MMQ_RAW_SCHED` to select the block
  index → tile mapping function. The bijection guarantees each tile is computed exactly once and the output memory image is bit-for-bit
  identical — correctness holds by construction, and measurement only needs to pass the performance gates. Far cleaner than adding a "schedule mode"
  branch inside the kernel: zero change to the hot loop.
- **Rung 2 split into two variables**: first measure the non-persistent u-loop (changing the tile walk into a loop,
  still one-block-one-tile semantics), then persistent (num_sms resident + strided
  list). −3.3% appeared at the first step, and adding persistence changed nothing — the two variables cleanly
  isolated: the cost comes from loop-ification itself, not from persistence.
- **Register-identity check**: the persistent rewrite's ptxas register allocation matches the original's
  (register-identical per ptxas), ruling out the confound "the negative came from register pressure" —
  the −3.3% is the pure scheduling-structure cost.
- **Using r24 to calibrate the bar**: precisely because all three rungs were expected negative, this round was pure measurement-framework
  practice — same-window interleaved A/B, distribution-separation reading (later hardened by doc 77's methodology into
  "min-new > max-base"). See §5.2.

### 3.2 Key code (current tree; what the ladder acted on)

The ladder acts on the wide kernel's grid geometry and tile walk. The current tree's `launch_mmq_raw_wide_nt`
grid mapping and smem budget (`src/cuda_kernels.cu:7200-7205`):

```cuda
    // 16-chain layout: 128-token x 128-od block tile. r14: qb8 slot-major
    // 48B stride (ldmatrix-for-B) + packed scales. KD=8 totals 98,304B and
    // KD=4 73,728B — both inside the ~99KB opt-in cap, 1 block/SM. The
    // attr/launch results are checked: an over-cap request used to fail
    // SILENTLY (r7 phantom 2124).
    dim3 grid((nt + 127) / 128, (od + 127) / 128);
```

The kernel-side tile walk (`src/cuda_kernels.cu:5892-5902`) — x is the token tile, y the od tile,
exactly the mapping the swizzle wanted to rearrange:

```cuda
    // Each warp owns a private 16-od-row slice and reads the FULL 128-token
    // tile: B fragments become warp-exclusive, A fragments are warp-shared.
    const int i0 = blockIdx.x * MMQ_WBI;      // token tile  (x, fastest)
    const int j0 = blockIdx.y * MMQ_WBJ;      // od tile     (y, slow)  = B-hot
    const int j0w = warp * 16;
    const int nb32 = id >> 5;
    const int nchunk = nb32;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    float sum[64] = {0.0f};   // [g][nh][l]: 8 A-frags x 2 B-frags x 4 C regs
```

The ladder's own shape (**narrative pseudocode, never committed** — `d90b3e9` is a docs-only commit):

```text
# Rung 1 — tile-order swizzle (a bijection on the grid, selected by MINFER_MMQ_RAW_SCHED)
sched = env("MINFER_MMQ_RAW_SCHED")            # default "x" (B-hot)
(x, y)  = (blockIdx.x, blockIdx.y)
match sched:
    "x"    -> tile = (x, y)                    # default: B-hot, measured optimal 1370.0
    "a"    -> tile = (y, x)                    # A-hot transposed, −2.3%
    "g"    -> tile = group_order(x, y)         # G-grouped, −2.4%
# bijection ⇒ each tile exactly once ⇒ output bit-identical

# Rung 2 — persistent blocks
for t in tiles(stride = gridDim, offset = blockIdx):   # u-loop, −3.3%
    compute_tile(t)                                    # occ=1 / occ=2 ≈ identical
# ptxas: register allocation identical to the original

# Rung 3 — full stream-k: not implemented. An fp accumulation re-order ⇒ greedy identity necessarily breaks.
```

### 3.3 Pitfalls

- **The absolute anchor failed silently**: nobody "broke" anything — the box drifting faster turned the ≥1350 absolute bar into
  something passable at zero cost. The lesson: an anchor must live in the same frame as the reading (same-window relative
  quantities); otherwise the bar loses its force without anyone noticing.
- **"struct family untried" ≠ "struct family promising"**: the scheduling family has no separable resource on a
  latency-bound, 1 block/SM kernel — working through §2's arithmetic (the A/B re-read ratio, wave
  count, tail existence) before acting predicts it; half of r24's value is turning that prediction into a measured record.
- **The u-loop has no free lunch**: even without persistence, merely changing "one block one tile" into
  "a block looping over tiles" is −3.3% — the hardware scheduler's work should not be replicated in software.

## 4. Verification

- **bit-identical (rung 1)**: the bijection guarantees by construction that the output memory image is bit-for-bit identical;
  greedy-32 identity therefore holds automatically (still run once as a regression confirmation). It defends against
  "a schedule change quietly skipping/recomputing some tile".
- **Interleaved A/B medians (uniform across the three rungs)**: same window, warmed up, alternating order, read against
  the same re-measured baseline. It defends against co-tenancy/thermal drift disguising the box's drift as a trend — this round's
  box drift is precisely why this reading protocol exists.
- **Register identity (rung 2)**: ptxas output comparison, ruling out the register-pressure confound.
- **A priori gate veto (rung 3)**: the greedy identity gate sentenced stream-k before any code was written —
  the cheapest use of a verification gate is "before building".

## 5. Results

### 5.1 The three rungs' numbers (same-window interleaved, vs the re-measured baseline)

| rung | shape | result | verdict |
|---|---|---|---|
| 1 | swizzle `MINFER_MMQ_RAW_SCHED` | default B-hot **1370.0** optimal; A-hot **−2.3%**; G-grouped **−2.4%** | REVERTED |
| 2 | persistent blocks | u-loop **−3.3%**; persistent occ=1/occ=2 ≈ identical | REVERTED |
| 3 | full stream-k | not implemented — vetoed a priori by the greedy identity gate | CLOSED |

The box's re-measured band: KD=8 ~1385–1391 (recorded band ~1330). Veto mechanism: the scheduling family has no separable resource on a
kernel with "1 block/SM + latency-bound + L2 already absorbing B reuse";
**retry conditions** — (a) a real quantization tail appears in the grid (large-tile shapes with tile count < SM count);
(b) the per-block fixed cost is visible relative to per-tile work (very short K); (c) the operand re-read ratio
inverts to B-dominant (only then does the A-hot order have a theoretical gain); (d) stream-k only if the numeric gate is replaced by a
tolerance regime (never happened in this campaign).

### 5.2 This round's real output: calibrating the +1.5% relative bar

Why can a **reverted round** calibrate a bar? Because calibration needs exactly a
"disinterested" measurement-framework rehearsal, and r24 is exactly that:

1. **The absolute bar's death certificate**: the baseline at 1385+ already passed ≥1350 by itself — the absolute anchor's premise
   (a stable box) was falsified. The bar had to become a **same-window relative quantity**.
2. **Resolution demonstrated**: this round's three rungs' −2.3% / −2.4% / −3.3% were all unambiguously judged negative under
   same-window interleaved A/B — the protocol has clean resolution for ≥2% effects,
   with the noise band clearly narrower than that magnitude. The landing line is set at **+1.5%**: above the noise band (readings
   reliable), below typical structural levers' magnitude (a real lever can pass: r28 +2.56%, r29 +2.80%),
   and high enough to block "sub-noise positives with an unproven mechanism".
3. **The reading discipline, distilled**: same-window pairing, alternating order, distribution separation (min-new > max-base),
   later hardened by doc 77's methodology into §2.3 — "the whole-prefill landing bar is +1.5%
   (relative to the re-measured baseline, calibrated in r24)".

Every round since reads inside this frame: r25's instruction cut +0.37/+0.49% → **below the bar,
reverted**; r28's Phase-2 gate 4 wrote in black and white "≥ +1.5% over the re-measured
baseline (**r24 convention**)"; r31's +1.07% was below the bar but mechanism-confirmed by ncu and
landed after explicit recording as real-but-sub-bar; r46's FA +0.27% → reverted; r55 even used a
roofline argument to derive "even a perfect kernel could not pass the bar" and skipped implementation; D3a listed it alongside
the ±2% A/B noise as veto grounds. In one sentence: **r24's levers all died, but they bought the
common yardstick used by the 30+ rounds since.**

## 6. Lessons

1. A bar must live in the same frame as its reading: an absolute anchor holds only while the box is stable; once the box drifts it becomes a
   ritual — relative (same-window paired) quantities are the only cross-session comparable thing.
2. A "failed" round's durable output need not be code: baseline re-measurement + resolution demonstration +
   reading discipline are calibrations only a disinterested round can fix accurately.
3. A scheduling-structure lever's premise is "separable temporal locality or a quantization tail exists" — compute the
   re-read ratio, wave count, and launch pipelining first; most swizzle/persistent
   attempts can be predicted negative before any code.
4. A verification gate's cheapest use is before implementation: stream-k's floating-point re-order being incompatible with greedy
   identity is an a priori fact — not one line of code should be written for it.

---
← [28 · r23 f16 wall decomposition](28-r23-f16-wall-decomposition.md) · [Index](./README.md) · [30 · r25 SASS opcode census](30-r25-sass-opcode-census.md) →
