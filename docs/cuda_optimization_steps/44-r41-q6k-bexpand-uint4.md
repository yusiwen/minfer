# 44 · r41 — q6_K B-expand widened to uint4 group loads (LANDED)

> **Result**: whole-prefill 1979.9 → 2605.2 tok/s (**+30.7%**, 7B q4_k_m @pp3314-eq, same-window A/B median); q6_K attn_v GEMM kernel 1.70 → 0.654 ms (**−61.5%**); the `long_scoreboard` (L1TEX) share of warp time 85.5% → 33.6%. The q6_K line's biggest single-step lever to that point.
> **Commit**: `b891e1b` (code, `src/cuda_kernels.cu` +58/−10) + `aa82e8f` (record). **Date**: 2026-09-05.

## 1. Background — where things stood

r37's whole-prefill attribution swapped the target for the whole campaign: the MMQ redesign had brought q4_K-class weights to near llama.cpp, but q6_K was still on a path slower than f16 — the q6_K
GEMM alone ate 1094.7 ms, 51.2% of the entire prefill wall clock, at 6.38×/GMAC efficiency. That is: the next doubling of the default path lay not in q4_K's tile shape but in "building q6_K a real
kernel".

So P6 landed three steps in a row, all on `mmq_raw_nb_bt_q6k_kernel`:

- **r38** (`75aabb9`): the BT-style raw-byte kernel. q6_K is not 8 32-element sub-blocks but 16 16-element sub-blocks, and one 32-k chunk spans two sub-blocks with different scales, so it uses
  `mma.m16n8k16` (KSPLIT=2) + an independent dsc rescale per half; the B tile is **expanded** into centered int8 (−32..31) at staging, moving the recomb (nibble + 2-bit field assembly + −32) out of
  the hot loop. +2.87%.
- **r39** (`f2b9e54`): KDR=2 double buffering — two copies of each A/B staging plane, kt+1's global→smem expansion overlapping kt's compute. +13.3%.
- **r40** (`65ecef7`): `__launch_bounds__(256, 3)` forcing a third resident block. The 0-spill dogma was falsified (80 regs + 4 B spill beats 87 regs
  at 2 blocks), +13.0%, reaching 2015.6 tok/s.

At r41's start the occupancy lever was spent (3 blocks/SM, 80 regs), yet the kernel was still latency-bound: compute at only 27%, No-Eligible as high as 74.5%. First on r40's candidate-lever list
was the B-expand's fetch style — and ncu's Warp State data pointed the finger exactly there: **85.5% of warp-stall cycles were `long_scoreboard`** (13.7 of the 16.0 cy/inst CPIStall), the
characteristic signature of L1TEX memory latency.

Where this step's absence would stall: three resident blocks had already pushed "use other warps to cover this warp's wait" to its limit, but the bulk of `long_scoreboard` is a **within-warp**
serial dependency chain — a load issued by this warp is consumed by this warp's next instruction, and the latency has nowhere to hide. More scheduling tuning, more tile tuning, would all skirt the
real bottleneck.

**The CPIStall accounting.** Of ncu's Warp State average 16.0 cy per inst of issue spacing, 13.7 cy are booked to `long_scoreboard` — i.e. 85.5% of each warp's issue time is spent "waiting for an
L1TEX round trip". After r40 the measurement was 18.12 warps/SM (37.74% occupancy): resident warps were not scarce, but 74.5% No-Eligible says they lacked **issuable instructions** — every recomb
hangs on its own load's result. Occupancy can only hide **cross-warp** latency; a chain of 32-level byte loads feeding one ALU stays exposed level by level inside the warp.

## 2. Principle — the GPU mechanism

**What B-expand is.** A q6_K 256-element super-block is packed into 210 B (224 B after padded registration). The in-block byte map:

```
offset   0 …… 127        128 …… 191       192 …… 207      208-209   210-223(padded)
content  ql: low nibbles  qh: per-element  sc[0..15]:      d: f16    14 B padding
         (256 4-bit       high 2-bit       the 16 sub-     scale     (for alignment)
         nibbles, 2 elems fields (64 B,    blocks' i8
         per byte)        4 elems per byte) scales)
```

An element's value = the 4-bit nibble in `ql` (low) | the corresponding 2-bit field in `qh` << 4, then −32 centered overall. The trouble is byte sharing: one ql byte serves two elements (nibble
shift 0/4), one qh byte serves four elements (2-bit field shift 0/2/4/6) — the same byte is sliced at 2-bit steps across elements. The BT kernel's smem B plane stores the **expanded centered int8**
(1 B per element, range −32..31) and `mma` eats int8 directly; so the whole recomb happens at staging:

```
v = ((ql[i] >> shift) & 0x0F) | (((qh[j] >> shift2) & 0x03) << 4)   // then −32
```

**The bottleneck's arithmetic.** At KDR=2 one kt covers 64 elements per row; the B tile is MMQ_NBJ=128 rows × 64 elements, and the kernel dispatches in 16-element groups: `ng = 128 ×
(2×32)/16 = 1024` groups, spread over 256 threads — **exactly 4 groups per thread per kt**. Before widening, each group issues **32 `LDG.E.U8`** (16 ql + 16 qh), and the SASS shows staging sits at
the top of the kt loop with load results consumed immediately by recomb→STS — the full L1TEX round-trip latency of every byte load is exposed on the critical path. That is the 13.7
cy/inst `long_scoreboard`: not a bandwidth shortage, but **strings of serially-awaited short loads**.

**Why uint4 cures it.** The padded 224 B row pitch = 14×16, so any block base `blk = W + j·(nsb·224) + sb·224` is 16-aligned; the offsets of the group's two runs can be enumerated straight from the
code — the ql run `blk + it0·64 + gg·16` (it0∈{0,1}, gg∈{0..3}) lands on {0,16,32,48,64,80,96,112}, the qh run `blk + 128 + it0·32 + (gg&1)·16` lands on {128,144,160,176} — all multiples of 16,
with the qh run strictly inside the qh region (128..191). When alignment holds, 16 contiguous bytes are **one `LDG.E.128` (uint4)**:

| | Loads per group | Per thread per kt | Whole block per kt |
|---|---|---|---|
| Before widening | 32× `LDG.E.U8` | 4 groups × 32 = 128 | 32,768 |
| After widening | 2× `LDG.E.128` | 4 groups × 2 = 8 | 2,048 |

A roughly **16×** cut in load instructions. Scoreboard events fall roughly proportionally with instruction count, while the recomb shift/mask ALU now operates on 32-bit wide words in registers (4
bytes per word, four words per uint4; `_Pragma("unroll")` lets ptxas expand the byte selection into a static `BFI/SHF` sequence) — ALU volume is nearly unchanged.

**Register neutrality is the key to composability.** The widening introduces only 4 uint4 (16 registers) of transient occupancy, and the ptxas landing point stays **80 regs / 4 B spill** — not one
register of the 3 blocks/SM budget that r40 bought with +50% resident warps is touched. A pure load-width change can stack with the occupancy lever only if it demands no repayment of the register
budget; that is also why it is cheaper than "deeper software pipeline"-class schemes (the latter already paid smem doubling once in r39).

**Which structure the change lands in.** r39's double buffering keeps two copies of each of a kt's four staging planes; the smem budget can be computed from the layout at the kernel head (KDR=2,
MMQ_NBI=64, MMQ_NBJ=128):

```
per buffer copy:
  qa8    [KDR·NBI·32]      = 2·64·32  = 4096 B   (A: q8 plane, swizzled)
  sda_q  [KDR·NBI·4]       = 2·64·4   =  512 B   (A: packed d|ssum)
  qb_exp [NBJ·KDR·32]      = 128·2·32 = 8192 B   (B: expanded centered int8 ← r41's target)
  sds    [KDR·NBJ]·float2  = 2·128·8  = 2048 B   (the dsc plane)
  subtotal                          14,848 B × 2 copies = 29,696 B
```

These 29,696 B are exactly the source of r39's record "KDR=2 hits the same 29,696 B" — r41's widening **adds not one byte of smem**; the `qb_exp` plane keeps its size, only the instructions
filling it get fewer. In-block division of labor: 256 threads = 8 warps, each warp owns 16 consecutive od rows (`j0w = warp*16`), and B-expand's `ng` dispatch is split linearly across threads, so
the same group's ql/qh uint4s always land on the same thread — the recomb therefore completes entirely in registers, with no cross-thread exchange.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Widen inside the kernel rather than going back to load-time pre-expansion.** r18 once moved B pre-expansion to load time and was vetoed for its +5.8 GB cost; r41 does not repeat that —
  widening changes only fetch width; the plane's byte size is untouched. (The load-time pre-expansion direction was later re-realized on q6_K by r53 in the form of the `W_exp` plane — a different
  trade: +1.52 GB to make staging a pure copy.)
- **`(bstride & 15) == 0` as the runtime gate.** Real models always register with the padded 224 B layout (16-aligned, gate open); the raw 210-B layout used by tests (210 = 13×16+2, unaligned)
  keeps the scalar path. One gate guarantees two things at once: the **alignment safety** of uint4 accesses and **bit equivalence** (the two paths agree element by element).
- **A closed form per group, not per-element `expand_q6_elem` calls.** Map `(cbase, gg)` to `it0/qsh/qh_shift`, and derive all 16 outputs from the 2 uint4 wide words via register shifts. The
  closed form was verified against the per-element version before landing: **512,000 elements, 0 mismatches**.

### 3.2 Key code

**Where the change sits in the main loop.** The r41-era kt main loop is r39's double-buffer shape: at the top of each iteration, stage the next tile first (filling `buf^1`), then run this tile's mma
compute on `buf` —

```cuda
// Loop skeleton of the r41 era (the gemm_cp_wait* lines still on the tree are
// r45/r53/r56's cp.async additions; at r41 time this spot had only a plain __syncthreads)
RAW_STAGE_Q6K_BT(0, 0);
__syncthreads();
for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
    if (kt + 1 < nktile) RAW_STAGE_Q6K_BT(kt + 1, buf ^ 1);  // top: stage the next tile
    __syncthreads();
    /* … KDR rounds of ldmatrix + mma.m16n8k16 + dsc rescale on buf … */
}
```

The SASS confirms the staging macro sits at the **top** of the iteration, ahead of compute issue — exactly r41's gain shape: the string of short loads queues once at the loop head, the mma compute
then starts, whereas before widening the same stretch was 128 byte loads queued one by one and consumed one by one by recomb.

**Before — the scalar per-byte path** (the shape all q6_K took before r41; today it survives on the tree as the raw 210-B fallback). `expand_q6_elem` addresses each element independently, one byte
at a time:

```cuda
// src/cuda_kernels.cu — scalar expansion (2 byte reads per element)
__device__ __forceinline__ int expand_q6_elem(const uint8_t* ql, const uint8_t* qh, int elem) {
    int m  = elem & 31;
    int it = elem >> 7;
    int n  = elem & 127;
    int ql_idx   = it * 64 + (n & 63);
    int ql_shift = (n >> 6) * 4;          // 0 or 4 (low/high nibble)
    int qh_idx   = it * 32 + m;
    int qh_shift = ((n >> 5) & 3) * 2;    // 0,2,4,6 (2-bit fields)
    int v = ((ql[ql_idx] >> ql_shift) & 0x0F)
          | (((qh[qh_idx] >> qh_shift) & 0x03) << 4);
    return v - 32;
}
```

The per-element staging branch that calls it (2 byte LDGs per element — the very source of the 85.5% `long_scoreboard`):

```cuda
// before: 16 elements per group = 32 LDG.E.U8, results consumed immediately by recomb
for (int x = threadIdx.x; x < MMQ_NBJ * (KDR * 32); x += blockDim.x) {
    const int jj = x / (KDR * 32), bec = x % (KDR * 32);
    const int j = j0 + jj;
    const int elem = cbase * 32 + bec;   /* element index within the super-block */
    int v = 0;
    if (j < od && sb < nsb) {
        const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)
            + (size_t)sb * bstride;
        v = expand_q6_elem(blk, blk + 128, elem);
    }
    qbexpb[(size_t)jj * (KDR * 32) + bec] = (uint8_t)v;
}
```

**After — r41's uint4 group expansion** (the tree's `(bstride & 15) == 0` branch, i.e. the live path for real models):

```cuda
// r41: 16-elem group expand via uint4 ql+qh global loads.
// padded 224B stride is 16-aligned → one uint4 each for the ql and qh runs, replacing 32 byte LDGs
const int it0 = (cbase >> 2) & 1;
const int qsh = ((cbase >> 1) & 1) * 4;
const int ng = MMQ_NBJ * (KDR * 32) / 16;
for (int g = threadIdx.x; g < ng; g += blockDim.x) {
    const int jj = g / 4, gg = g & 3;
    const int j = j0 + jj;
    uint8_t* out = qbexpb + (size_t)jj * (KDR * 32) + gg * 16;
    if (j < od && sb < nsb) {
        const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)
            + (size_t)sb * bstride;
        const uint4 qlv = *(const uint4*)(blk + it0*64 + gg*16);      // ql: one LDG.E.128
        const uint4 qhv = *(const uint4*)(blk + 128 + it0*32          // qh: one LDG.E.128
                          + (gg & 1) * 16);
        const int qhs = qsh + ((gg >> 1) & 1) * 2;
        const uint32_t qx = qlv.x, qy = qlv.y, qz = qlv.z, qw = qlv.w;
        const uint32_t hx = qhv.x, hy = qhv.y, hz = qhv.z, hw = qhv.w;
        _Pragma("unroll")
        for (int e = 0; e < 16; e++) {                                 // recomb entirely in registers
            const int sidx = e >> 2;                                   // which word holds byte e
            const int sh = (e & 3) * 8;
            const uint32_t qsel = (sidx == 0) ? qx : (sidx == 1) ? qy
                              : (sidx == 2) ? qz : qw;
            const uint32_t hsel = (sidx == 0) ? hx : (sidx == 1) ? hy
                              : (sidx == 2) ? hz : hw;
            const uint8_t qb_ = (uint8_t)((qsel >> sh) & 0xFF);
            const uint8_t hb_ = (uint8_t)((hsel >> sh) & 0xFF);
            out[e] = (uint8_t)((((qb_ >> qsh) & 0xF)                   // low nibble
                | (((hb_ >> qhs) & 3) << 4)) - 32);                    // high 2 bits, −32 centered
        }
    } else {
        _Pragma("unroll")
        for (int e = 0; e < 16; e++) out[e] = 0;                       // zero-fill out-of-range rows
    }
}
```

Side-by-side: the load side goes from "2 byte LDGs per element" to "2 128-bit LDGs per group"; the addressing side absorbs the scalar version's `it/ql_idx/qh_idx` divisions into the group
coordinates `gg = g & 3`, `jj = g / 4` closed forms; the consumption side goes from "one shift chain per byte" to "expansion shifts over four bytes per 32-bit word". Total dispatch volume is
unchanged (still `MMQ_NBJ × KDR×32` elements); what changes is **the width and count of each memory access**.

Where `bstride` comes from — the launch side decides by registration layout, and the same kernel serves both layouts:

```rust
// src/cuda.rs — prefill MMQ launch side: block_stride chosen by Q6_K registration layout
let block_stride: i32 = if ttype == TensorType::Q6_K && padded_q6k {
    224                                    // 7e② padded repack → the r41 uint4 gate opens
} else {
    210                                    // raw GGUF → the scalar fallback arm
};
```

Inside the kernel, `(bstride & 15) == 0` says at a glance which arm to take: real models (padded) get the uint4 path, and the 210 B raw test layout falls back to scalar automatically — one SASS
serves both layouts, no second kernel needed.

### 3.3 Pitfalls

- **The SASS placement trap (foreshadowing r43).** Double-buffered staging sits at the top of the kt loop, and after widening all uint4s still issue ahead of compute. The first recomb ALU
  (LOP3/SHF) still waits out the whole load round trip — what r41 cut was the **instruction count**, not the **length of the latency chain**. r43's PC-sampling then quantified this residual: the
  recomb consumer side still held 45% of the remaining stall.
- **uint4 alignment is not a soft constraint.** `*(const uint4*)(blk + ...)` requires 16 B alignment in hardware; a misaligned address faults outright. Alignment is guaranteed by the padded
  stride arithmetic (224 ≡ 0 mod 16, all in-group offsets multiples of 16) — but only for the padded layout. That is why the gate must exist, not as a "testing convenience".
- **The nibble/2-bit bookkeeping is easy to get wrong.** The ql nibble choice (low/high 4 bits) and the qh 2-bit field (shifts 0/2/4/6) vary with the `(cbase, gg)` combination; one wrong `>>` in
  the closed form is a silent bit flip. The closed form was first compared element-by-element against the scalar version on the host for 512,000 elements, and only entered the kernel at 0
  mismatches.
- **Keep and test both arms of the gate.** Widening holds only for the padded layout, but the raw 210-B layout is the one parity testing uses — if the raw path were made to error out for
  convenience, the parity gate could only ever run on padded. Coexisting arms let the parity gate verify once each under the two fetch shapes — raw (`expand_q6_elem` per-element expansion) and
  padded (uint4 closed-form expansion) — against the same reference output: layout and fetch shape get tested as two separated variables.

## 4. Verification

- **Gate-1 closed-form verifier**: the uint4 expansion's output vs `expand_q6_elem` element by element, 512,000 elements 0 mismatches — defends against algebra errors in the closed form (the
  bookkeeping trap above).
- **Parity dump (raw + padded) 1/0**: kernel output bit-identical to the CPU reference — defends against layout/alignment assumptions drifting on the real registration path.
- **greedy-32 byte-identical**: the end-to-end decoded token stream unchanged — defends against the cumulative class of errors where "the numbers look right but the decode trajectory drifted".
- **suite 166/0/3**: full graph-path regression — defends against the change spilling into branches beyond q6_K.
- **ncu Warp State before/after**: `long_scoreboard` 85.5% → 33.6% (13.7 → 3.6 cy), L1/TEX throughput 14.8% → 33.6%, compute 27.0% → 37.8% — the mechanism's evidence chain: the widening's gain
  really did land on the stall that was attributed.

## 5. Results

| Metric | before | after |
|---|---|---|
| whole-prefill (7B q4_k_m, same-window A/B median) | 1979.9 | **2605.2 (+30.7%)** |
| q6_K attn_v GEMM kernel | 1.70 ms | **0.654 ms (−61.5%)** |
| `long_scoreboard` | 85.5% (13.7 cy/inst) | **33.6% (3.6 cy)** |
| L1/TEX throughput | 14.8% | 33.6% |
| compute (pipe utilization) | 27.0% | 37.8% |
| registers / spill | 80 / 4 B | **80 / 4 B (unchanged)** |

vs-llama advanced from 1.65× after r40 to **1.27×**. This is the q6_K line's biggest single-step lever (against r39's +13.3% and r40's +13.0%), register-neutral, and with zero memory cost. The
remaining 33.6% at this point pointed mainly at the dsc `d·sc` reads (unmerged narrow loads) plus KSPLIT=2's intrinsic overhead — this attribution directly begat r42 (widening the dsc reads) and
r43 (PC-sampling precise attribution).

**This step's place in the campaign**: at r38's start q6_K's per-GMAC cost was 368.9 µs/GMAC, and the three steps r39/r40/r41 narrowed it down to this step's −61.5% kernel time; the q6_K GEMM was
no longer a "slower than f16" burden. But whose the 33.6% residual stall was — the answer of the moment (narrow dsc reads) got only the consumer side of the 26 percentage points right — r42 used a
zero-gain experiment to veto the width theory, and r43's PC-sampling finally broke it into the actionable list recomb 45% / A-staging 28% / dsc consumers 26%. r41's "register-neutral" property also
verified, on the r40/r41 combination, that levers can stack; that lesson was cited repeatedly in the later cp.async series (r45/r53).

(Comparison note: the 2015.6 recorded in the r40 chapter is that session window's absolute value; r41's A/B re-anchored at 1979.9 within its own session window. Cross-window absolute values are not
comparable — read only same-window deltas.)

## 6. Lessons

1. **The q6_K line's biggest single-step lever was load width, not scheduling**: byte-granularity global loads in a staging loop are a textbook L1TEX scoreboard factory — 32 serially consumed
   `LDG.E.U8` cannot be hidden even at three resident blocks.
2. **Register-neutral levers compose with occupancy levers**: check the ptxas landing point before starting; a widening that repays no register debt is free, and anything that does repay (a deeper
   pipeline, more buffers) must be priced against it.
3. **Widening cuts instruction count, not the latency chain**: the first consumer ALU still eats the full load round trip — the residual stall needs PC-sampling to find the consuming instruction
   before it can be broken down further (r43).

---
← [43 · r40 third resident block](43-r40-third-resident-block.md) · [Index](./README.md) · [45 · r42 stage-wide dsc scale read](45-r42-stage-wide-dsc-read.md) →
