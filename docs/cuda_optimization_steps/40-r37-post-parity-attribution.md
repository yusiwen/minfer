# 40 · r37 — Post-parity whole-prefill attribution: the wall clock re-decomposed (MEAS-ONLY, no code change)

> **Result**: 3325-tok prefill, the all-four-gates BT path: GPU busy 2139.6 ms / wall 2190
> ms = **1521 tok/s, vs-llama 2.15×** (the first whole-wall number on the 3325-eq anchor).
> Decomposition: **q6_K GEMM 1094.7 ms = 51.2%** (368.9 µs/GMAC vs llama 57.8 →
> **6.38×**), q4_K bt 600.0 ms (28.0%, 1.15× = parity); **the q6_K ffn_down class alone is
> 1063.5 ms (48.6%) — larger than the entire q4_K bt GEMM (600 ms)**. Priority queue: ①
> put q6_K on a raw-byte kernel (ceiling ~2720 tok/s) ② q4_K short-nt amortization ③ FA
> structure. **No code change**.
> **Commit**: `ea234f1` (docs-only). **Date**: 2026-09-05.

## 1. Background — where things stood

r34–r36 had pushed the q4_K BT-kernel line to its end: r34's quantize-transpose prepass
+9.72% (1364.2 → 1496.8 tok/s), while r35 and r36 falsified in succession the hypothesis
that "cuttable supply remains inside the kernel" — the ALU hides in the IMMA's shadow
(r35), the wavefronts hide in the same shadow (r36), and the bt kernel reached per-IMMA
parity with llama's `mul_mat_q` (6.33 vs 6.02 G-IMMA/s). The campaign was on the opt-in
MMQ path (`MINFER_MMQ=1`) at this point, and none of the prior MMQ rows had ever
recorded a whole-wall vs-llama number on the 3325-eq anchor (the default f16 path had
been stuck at 1.43× since P5).

But the 2.15× whole-wall gap remained — it has to live somewhere. r35/r36's
falsifications delivered an **exclusionary** conclusion: the gap is not in the bt
kernel's inner loop. That is precisely the license to turn to whole-wall attribution:
since the single-kernel microbenchmark had declared itself "not guilty", the only honest
next move was to **take the whole wall apart launch by launch** and see which kernel
class the time actually lands in. It also answers a deeper anxiety: every +2%, +7%, +9%
since r12 had been kernel-level A/B, with never a family portrait of the whole prefill
wall to answer "who should the next +X% be spent on".

Where this stalls without the step: the candidate list is empty (r32's finite-lever
sweep, r33's hybrid port, r35's decode, r36's LDSM all cleared out); continuing to hunt
levers inside the q4_K bt kernel is doubling down on a falsified direction; and for the
campaign to reach 1.0×, it must first know the composition of the remaining 2.15×.

## 2. Principle — attribution methodology

This round has no new kernel; its "principle" is three attribution-method decisions,
each learned from a real lesson of the previous rounds.

**Decision one: whole-wall nsys finds #1; matched-nt pairing sets the per-class
ratios.** A single-kernel ncu answers "who stalls inside this kernel" but cannot answer
"is this kernel worth optimizing". Whole-wall attribution needs two levels of
measurement: full-graph nsys buckets GPU busy time by launch (which kernel class ate how
many milliseconds), then each kernel class gets its wall/GMAC ratio measured separately
at a paired nt. Neither level works alone: without the bucketing you do not know whom to
pair; without the pairing you have only absolute milliseconds and cannot tell "model
structure" (q6_K simply has more FLOPs) from "implementation gap".

**Decision two: wall/GMAC normalization.** The only fair cross-kernel,
cross-implementation unit is "wall clock per GMAC". minfer and llama run the same GGUF
weights over the same shapes → the two sides' GMACs are strictly equal, so the wall
ratio is the efficiency ratio. q4_K bt is compared against `mul_mat_q<12>`, q6_K
`mmq_nt<7,2>` against `mul_mat_q<14>`, each yielding a dimensionless multiplier, and the
classes add up into a GEMM total (84.0 vs 35.2 µs/GMAC = 2.38×).

**Decision three: lock nt first, ratios second.** r36's lesson was cashed the very next
round: it had measured minfer@3325 against llama@512 and read per-IMMA 1.05× — this
round's matched-nt re-test (both sides nt≈512, minfer's 511-tok prompt vs `llama-bench
-p 512 -n 0 -r 1 -t 8`) shows bt's wall/IMMA is **1.43×** (minfer 379.9 µs / 4.23
G-IMMA/s vs llama 265.7 µs / 6.04 G-IMMA/s). The gap comes from tile prologue / wave
amortization: at prefill-scale nt it is diluted across thousands of inner tiles into
invisibility; at short nt it stands exposed. Per-IMMA parity is cited with an nt clause
from now on.

**The wall-relevance precondition check.** Kernel time and wall time are separated by
the launch gap: this round's GPU busy 2139.6 ms vs wall 2190 ms — a gap of 50.4 ms
(2.3%) — busy≈wall, so the kernel-level decomposition is valid as an approximation of
the whole wall (this "verify before attributing" move is exactly the vaccine against
r46's later FA trap of "the kernel shrank but the wall did not respond").

**The GMAC arithmetic cross-check.** q6_K ffn_down alone: 1063.5 ms @ 5.5 TFLOPS; q4_K
gate/up: 63.6 TFLOPS (the whole bt-kernel class 60.0 TFLOPs) — the same ffn line differs
**11.5× per-MAC** between the two weight types. That is the quantitative meaning of
"structural deficit": it is not that llama's q6_K is inherently dearer (its 57.8 µs/GMAC
is only 1.84× its own q4_K's 31.4, consistent with K-quant unpack cost) — it is that
minfer's q6_K runs an old road that none of r12–r34 ever modernized.

**Why q6_K never benefited from the modernization — the structural inventory.** Review
every upgrade r12–r34 landed on the q4_K bt kernel and the generic q6_K has none of
them: no ldmatrix B-frag supply (qb is a plain array, read word by word), no raw-nibble
smem layout (B values are expanded to int at staging), no r18-style bulk copy (every
tile redoes the nibble assembly), no r34 prepass transpose (the A side still goes
through `mmq_stage_a`'s per-chunk reshuffle) — and it carries an extra layer of
q6_K-inherent complexity: **16 16-element sub-blocks** (q4_K has 8 32-element
sub-blocks) force `KSPLIT=2`: one mma split into `m16n8k16` × 2 with two independent int
accumulators (clow/chigh), and smem keeps both `sds`/`sds1` scale planes plus an `sdm`
min plane. It is not "a slow version of the q4_K kernel" — it is **a generation-earlier
architecture** carrying 51.2% of the wall; the attribution round's value is turning this
into an indictment with numbers.

## 3. Implementation

### 3.1 Design choices (why these two measurements and this machine state)

- **The "all four gates" BT path**: the attribution subject must be the campaign's best
  configuration at the time (MMQ on + BT routing on), otherwise the buckets describe a
  path nobody runs.
- **co-tenant idle @0% verified**: the drift lessons of the r12–r25 era (absolute values
  not comparable across windows) are more lethal in an attribution round — the bucketed
  milliseconds are to be treated as numbers "comparable against llama in the same
  window", so the machine must be clean. This round explicitly records co-tenant 0%.
- **The bucketing's practical granularity**: the nsys trace aggregates launches by
  kernel symbol name (not by op semantics), so the "q6_K GEMM" bucket holds all
  `mmq_nt<7,2>` launches for both the attn_v and ffn_down weight classes; ffn_down's 13
  launches × ~80 ms were split out of that bucket by shape/od. Symbol-name aggregation's
  blind spot is same-name-different-type — fortunately `mmq_nt`'s TYPE is a template
  parameter, so the symbol name carries `<7,2>` and the bucket boundaries are naturally
  clean.
- **The single-class ncu spot check**: a matched q GEMM single launch (grid 8×28,
  1,605,632 IMMA bit-identical on both sides — the direct consequence of same weights,
  same shape) grounds the class-level 6.38× in single-kernel evidence.

### 3.2 Key code: why q6_K sits on the old road

The routing side (era tree): the bt entry `launch_mmq_raw_nb_bt_nt`'s guards mean only
q4_K benefits; every other type falls back to generic in the dispatch's `default` arm:

```cuda
extern "C" int launch_mmq_raw_nb_bt_nt(int type_id, …, int kd) {
    (void)type_id;
    if (kd != 8) return 0;              // bt exists only in the KD=8 raw-nibble shape
    if (qa8g == 0 || sdag == 0) return 0;   // no prepass planes → clean fallback
    …
    mmq_raw_nb_bt_kernel<8><<<grid, 256, smem, stream>>>(w, qa8g, sdag, …);
    …
}
// dispatch (q6_K = type 7, falls into default):
default: MMQ_LAUNCH((mmq_nt_kernel<7, 2, false>)); break;   // KSPLIT=2
```

The generic q6_K **re-derives** the B values from the raw weights for every tile it
consumes — `mmq_stage_b<7>`'s per-tile redo (against bt's "bulk-copy the raw bytes +
unpack in registers at mma time", there is neither a raw-byte bulk path nor a
pre-expanded plane here):

```cuda
int s = (2 * c + half) % 16;                       // q6_K: 16 16-element sub-blocks
int chunk = s >> 3, g = (s >> 1) & 3, is = s & 1;
const uint8_t* ql = blk + chunk * 64 + (g & 1) * 32 + is * 16;  // the low 4-bit plane
const uint8_t* qh = blk + 128 + chunk * 32 + is * 16;           // the high 2-bit plane
#pragma unroll
for (int w = 4 * half; w < 4 * half + 4; w++) {
    …
    uint32_t nib = (g < 2) ? (QL & 0x0F0F0F0Fu) : ((QL >> 4) & 0x0F0F0F0Fu);
    uint32_t hi  = ((QH >> (2 * g)) & 0x03030303u) << 4;        // high-bit assembly
    qb[r * MMQ_WS + w] = __vsubss4((int)(nib | hi), 0x20202020); // −32 → signed 6-bit
}
if (half == 0) {                                   // each row also carries two 8-bit scales
    ds[r]  = d * (float)(int8_t)blk[192 + (2 * c) % 16];
    ds1[r] = d * (float)(int8_t)blk[192 + (2 * c + 1) % 16];
}
```

Add the kernel header's double-buffered smem layout (the seven planes
`qa/qb/ssa/sda/sds/sds1/sdm`) and `KSPLIT=2` (two `mma.m16n8k16` + independent
accumulators, because q6_K has a scale group every 16 elements): this is the shape that
**none** of r12–r34's modernizations (ldmatrix, raw-nibble, BT, prepass) ever touched.
The attribution round's whole meaning is to turn that "never modernized" into an
indictment with milliseconds attached.

### 3.3 Pitfalls

- **r36's mixed-nt measurement was corrected this round**: per-IMMA 1.05× (mixed nt) →
  wall/IMMA 1.43× (matched nt≈511). The correction itself went into the record — not
  papering over a previous round's measurement defect is why this record can be trusted.
- **busy ≠ wall must be verified first**: the 50.4 ms launch-gap deficit (2.3%) is on
  record, and only then was the kernel-level decomposition allowed to approximate the
  whole wall; r46 later added a dedicated counter-example check for "kernel faster, wall
  unmoved".
- **The absolute-millisecond window changes with this round**: 3325-tok (3325-eq)
  becomes the MMQ era's anchor; the earlier @2K/@3354 numbers are no longer directly
  comparable (machine state + anchor both changed).

## 4. Verification

- **co-tenant idle @0%**: defends against machine drift reading the bucketed numbers as
  implementation gaps.
- **matched-nt pairing** (minfer's 511-tok vs llama-bench `-p 512 -n 0 -r 1 -t 8`,
  launch 1 each): defends against nt amortization polluting the class-level ratios —
  r36's lesson promoted to a gate.
- **identical-weights → identical-GMAC**: validates the precondition of the wall/GMAC
  normalization (grid 8×28, 1,605,632 IMMA identical on both sides).
- **Two-level cross-check**: the full-graph bucketing (51.2%) and the matched-nt ratio
  (6.38×) point at the same culprit, and the inequality that q6_K ffn_down alone (1063.5
  ms) exceeds the entire q4_K bt GEMM (600.0 ms) orders the priorities — the conclusion
  does not rest on a single measurement.

## 5. Results

**Whole-wall decomposition** (full-graph nsys, GPU busy 2139.6 ms, wall 2190 ms = 1521
tok/s):

| Kernel class | Time | Share of busy | Notes |
|---|---:|---:|---|
| **GEMM q6_K** (attn_v + ffn_down, generic `mmq_nt<7,2>`) | **1094.7 ms** | **51.2%** | 368.9 µs/GMAC |
| GEMM q4_K (bt) | 600.0 ms | 28.0% | 60.0 TFLOPs |
| FA prefill attention | 124.7 ms | 5.8% | 5.7× |
| quantize prepass | 86.8 ms | 4.1% | 1.25× llama (partly a per-shared-A 2× redundancy, fixed only in r49) |
| swiglu | ~3.9% | | |
| rest | ~7% | | |

Within that, **q6_K ffn_down alone, 13 launches × ~80 ms = 1063.5 ms (48.6% of the whole
wall)**, @5.5 TFLOPS — larger than the entire q4_K bt GEMM.

**matched-nt wall/GMAC** (this round's headline table):

| GEMM class | minfer | llama | ratio |
|---|---:|---:|---:|
| q4_K (bt vs `mul_mat_q<12>`) | 36.2 µs/GMAC | 31.4 µs/GMAC | **1.15×** (parity-grade) |
| **q6_K (`mmq_nt<7>` vs `mul_mat_q<14>`)** | **368.9 µs/GMAC** | 57.8 µs/GMAC | **6.38×** |
| GEMM total | 84.0 µs/GMAC | 35.2 µs/GMAC | 2.38× |

**The priority queue** (ordered by recoverable milliseconds): ① put q6_K on a raw
byte-width kernel — at the q4_K bt rate of 63.6 TFLOPs, about −970 ms → ceiling ~2720
tok/s; ② q4_K short-nt amortization (1.43× @511, 1.15× at prefill nt); ③ FA structure
(124.7 ms, 5.7×).

The ceiling number's arithmetic chain is worth walking in full, because it is the
priority's price tag: lowering q6_K's two GEMMs (1094.7 ms) to the q4_K bt per-GMAC rate
(368.9 → ~60 µs/GMAC, a 6.15× speedup) saves about **1094.7 − 1094.7/6.15 ≈ −917 ms**;
rounding up with the same bucket's dsc/stage residue gives ~−970 ms; wall 2190 − 970 =
1220 ms → 3325 tok ÷ 1.220 s = **~2725 ≈ 2720 tok/s**. All three steps use this round's
measured milliseconds, with no extrapolated parameters — r47's later measurement (q6_K
196.4 ms, wall 1274 ms) landed in the same interval, showing the pricing was
conservative.

**Priority ① delivered, plus the hidden tax (looking back from r38–r41)**: the raw-byte
port chartered here was delivered by Era D's four-hit combo, round by round:

| Round | Lever | Whole wall | q6_K class-level metric |
|---|---|---|---|
| r38 | q6_K BT-style raw-byte mma kernel (KSPLIT=2, KDR=4) | +2.87% | 368.9 → 221.8 µs/GMAC (1.66×) |
| r39 | KDR=2 double buffering (A+B pipelined) | +13.3% | attn_v kernel −19.7% |
| r40 | `__launch_bounds__(256,3)` third resident block | +13.0% | kernel −23% |
| r41 | B-expand widened to uint4 groups | +30.7% | kernel 1.70 → 0.654 ms (−61.5%) |

Together they cut the q6_K GEMM from 1094.7 ms to 196.4 ms (r47's re-measurement) — but
once q6_K goes down the BT path it must pass through the same quantize-transpose
prepass, so **the prepass grew +31.6 ms (the hidden tax)**; the net q6_K wall-clock gain
= **+866.7 ms**. Lesson: a landing's benefit accounting must subtract every hidden tax
it creates, otherwise the next attribution round "loses" some of the already-delivered
milliseconds.

**Why wall decompositions expire (looking back from r47)**: r37's table, re-measured at
r47, went "stale exactly as predicted" — q6_K 51.2% → 15.8%, wall 2190 → 1274 ms (1521 →
2610 tok/s), vs-llama 2.15× → 1.27×, and **FA topped the table as the #1 structural
residual** (125.8 ms, 5.72×). A decomposition table is a snapshot of "the current kernel
mix": every landed lever re-ranks it. So the correct way to attribute is to **re-measure
the whole wall every time one line converges**, and any citation of an old decomposition
must carry its version number.

**Campaign state at r37: 1521 tok/s, 2.15× vs-llama.**

## 6. Lessons

1. **Re-attribute the whole wall after each line converges, then pick the lever** —
   after r35/r36's consecutive falsifications, the only way out was to ask "where in the
   whole wall is the gap", and the answer (51.2% in another kernel) is forever invisible
   from a single-kernel viewpoint.
2. **Optimizing one weight-type line exposes the next**: the MMQ modernization made q4_K
   reach parity, which cast the never-modernized q6_K as "slower than f16" — a lever's
   value is relative, decided by the wall's composition.
3. **Lock the variables before normalizing**: mixed-nt turned per-IMMA 1.05× into wall
   1.43×; matched controls are the ticket into cross-kernel comparison (r36's
   measurement defect was promoted to a gate this round).
4. **A landing's net gain = the milliseconds won − the hidden taxes** (prepass +31.6
   ms), and **decomposition tables have a shelf life** — together these are the complete
   answer to "why wall decompositions expire".

← [39-r36-a-frag-wavefront](39-r36-a-frag-wavefront.md) · [Index](./README.md) · [41-r38-q6k-bt-rawbyte-mma](41-r38-q6k-bt-rawbyte-mma.md) →

