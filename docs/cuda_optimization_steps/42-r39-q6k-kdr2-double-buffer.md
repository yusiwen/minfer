# 42 · r39 — q6_K KDR=2 double-buffer (LANDED)

> **Result**: 7B whole-prefill 1568.7 → 1777.5 tok/s (+13.3%); attn_v q6_K kernel −19.7% (2,549,248 → 2,046,848 ns).
> **Commit**: `f2b9e54`. **Date**: 2026-09-05.

## 1. Background — where things stood

r38 (the previous doc) had just given q6_K a BT-style raw-byte mma kernel: matched-nt unit cost 368.9 → 221.8 µs/GMAC, whole-prefill +2.87%. But r38's verification data left one conspicuous hook: **the attn_v kernel was still latency-bound, with compute at only 16.7%** — most of the SM's issue slots were spinning idle; the kernel was waiting on data, not computing.

Waiting on what? r38's main loop was single-buffered:

```
for (kt...) {
    if (kt > 0) RAW_STAGE_Q6K_BT(kt);   // stage the kt panel this iteration needs
    __syncthreads();                     // whole block waits until staging is complete
    ... mma compute(kt) ...              // only then compute
    __syncthreads();                     // loop-tail fence: protects the single buffer from next round's overwrite
}
```

Staging (the A-side bulk uint4 copy + the B-side ql/qh bitfield expansion + the dsc reads) is squeezed between two `__syncthreads()`: every warp's LDG latency and expansion ALU are **fully exposed** — no compute overlaps them. All warps stage together, wait together for the slowest one, and only then start computing together. Compute at 16.7% is the direct reading of this serial structure: within one kt period, the staging segment is far longer than the compute segment.

The fix was validated once back in r20 (that was the A-side split-phase staging lesson: "the gap carrier is long_scoreboard in the LDG→STS chains"); this time the same idea moves to the q6_K kernel's **entire staging layer**, pipelining the B-side ALU expansion along with it: **double buffering** — two copies of every per-kt plane, with kt+1's expansion written into the other buffer while kt computes. The code comments spell out that this is the pipeline scheme the generic kernel `mmq_nt<7,2>` already had, now ported to the q6_K-specific kernel.

The cost of the serial structure can be computed from r38's shape (in its KDR=4 form, per block per kt iteration; derived from the kernel structure, not a measured breakdown): the staging segment must move the A-side 9,216 B (qa8 8,192 + sda_q 1,024) as uint4s, expand the B-side 128 rows × 128 elements = 16,384 elements (each one ql `LDG.U8` + one qh `LDG.U8` + bitfield ALU), and read 512 dsc pairs; the compute segment is 8 warps × 4 kd × 16 `mma.m16n8k16` = 512 mmas plus the epilogue. The two segments are of the same magnitude, and the serial structure makes them **add** into the critical path — compute at 16.7% means staging holds an overwhelming share of that path.

Only one question remains: where does the smem come from.

## 2. Principle — the GPU mechanism

### 2.1 The fence economics of double buffering

Single-buffered (r38), the critical path per kt:

```
[all warps] stage(kt)  →  barrier  →  [all warps] compute(kt)  →  barrier
     ~LDG latency + expansion ALU        0 overlap              mma
```

Double-buffered (r39):

```
prologue: stage(0 → buf0); barrier
loop kt:  stage(kt+1 → buf^1)   ── concurrent with the line below ──▶  compute(kt on buf)
          (LDG latency hidden under the mma issue stream)
          barrier   ← the loop-tail fence does double duty (see §2.3)
```

Drawn as a timeline (each cell is one critical-path segment; the two rows are concurrent activities on the same SM):

```
r38 single-buffer:  ─[stage 0]─🚧─[compute 0]─🚧─[stage 1]─🚧─[compute 1]─🚧─
r39 double-buffer:  ─[stage 0]─🚧─[compute 0]─🚧─[compute 1]─🚧─[compute 2]─🚧─
                          [stage 1 ]─[stage 2 ]─[stage 3 ]      ← riding on the compute segments
```

In r38's timeline, stage and compute alternate in exclusive possession; in r39 the compute segments stretch out (covering the stages), and total time ≈ stage(prologue) + Σ compute, not Σ(stage + compute).

The essence of the gain is **pure overlap**: staging's workload is not one byte smaller, compute's workload is not one byte smaller; what changed is only that the two no longer wait on each other. This is r20's conclusion replayed at the staging layer — latency is not eliminated, it is hidden inside issue slots that were idle anyway. The compute 16.7% r38 left behind shows the idle slots were plentiful, so the room for overlap to cash in was large.

A counterexample worth contrasting: the cp.async double buffering tried on the q4_K raw kernel in the r13 era (`784786d`) measured ~0 and was vetoed — at that time the kernel was L2-throughput bound, and re-timing the same bytes bought nothing. The same mechanism (double-buffered staging) is a dead lever on a throughput-bound kernel and +13% on a latency-bound kernel: **the lever is determined by the kernel's bottleneck regime, not by the mechanism itself**. The compute 16.7% r38 left behind was the ticket that said "this is a latency regime" before r39 even started.

### 2.2 The smem budget: price both planes

Double buffering is not free: smem doubles. The form r38 landed was **KDR=4 single-buffer = 29,696 B** (qa8 8192 + sda_q 1024 + qb_exp 16384 + sds 4096). Keeping KDR=4 and double-buffering directly gives every term ×2 = **59,392 B = 1 block/SM** — exactly the cliff r38 had just measured-vetoed with KDR=8 (1097.8 tok/s). The smem budget must price the A and B **planes together**; looking only at the B expanded plane (qb_exp, the largest term at 16384 B) creates the illusion that "there is still headroom".

The solution is to halve KDR to buy buffers: **KDR=2 × double buffer**. Per-plane accounting (MMQ_NBI=64, MMQ_NBJ=128; qa8 = KDR·NBI·32, sda_q = KDR·NBI·4, qb_exp = NBJ·KDR·32, sds = KDR·NBJ·8):

| Plane | r38: KDR=4 single buffer | Vetoed: KDR=4 double buffer | **r39: KDR=2 double buffer** |
|---|---|---|---|
| qa8 | 8,192 | 16,384 | 8,192 |
| sda_q | 1,024 | 2,048 | 1,024 |
| qb_exp | 16,384 | 32,768 | 16,384 |
| sds | 4,096 | 8,192 | 4,096 |
| **Total** | **29,696** | **59,392** | **29,696** |

**29,696 B — exactly the same footprint as r38's KDR=4 single buffer, 2 blocks/SM preserved**, but each iteration now gains real compute/staging overlap. The cost of KDR dropping 4→2: each kt advances only 64 k (two 32-chunks), kt iterations double and mmas per iteration halve — iteration overhead rises slightly and A/B plane reuse drops slightly. Trading the overlap gain against this dilution, the account is positive (§5 measures +13.3%).

### 2.3 How to place the fences

Double buffering saves one of the two per-iteration fences (plus one in the prologue), but both duties of the remaining fence must hold:

1. **Separate compute(kt−1, buf^1) from stage(kt+1, buf^1)** — staging writes exactly the buffer the previous iteration's compute read, so the write must not start until every warp has finished reading;
2. **Separate stage(kt+1, buf^1) from compute(kt+1, buf^1)** — compute reads exactly the buffer this iteration's stage wrote, so the read must not start until every warp has finished writing.

One `__syncthreads()` at the loop tail carries both: it is simultaneously the rendezvous for "the previous round's compute is fully done" (allowing the next round's stage to reuse the buffer) and the rendezvous for "this round's stage is fully done" (allowing the next round's compute to read). Drop either duty and you have a data race — which also explains why the "double-buffer B only" variant in §3.1 never even left the gate on correctness grounds. Fence accounting compared:

| Form | Fences per iteration | Staging vs compute |
|---|---|---|
| r38 single buffer | 2 (after stage + after compute) | serial: all warps stage together, wait together, compute together |
| r39 double buffer | 1 + 1 in the prologue | overlapped: stage(kt+1) rides on compute(kt) |

Note the gain is not "one fence fewer" per se (that is only tens of cycles per iteration); it is that the interval between fences changes from mutually exclusive to concurrent.

### 2.4 The vetoed variant: double-buffer B only (KDR=4)

The intuitive plan is to keep KDR=4 and give only the largest plane (qb_exp) a second copy, halving the doubling cost. **It fails on correctness alone**: with the A side still single-buffered, stage(kt+1)'s A-side bulk copy would overwrite A while compute(kt) is still reading it — A and B staging happen atomically inside the same macro (A first, then B), so pipelining requires doubling everything, and doubling everything at KDR=4 is the 59,392 B cliff. The variant dies on §2.2's "price both planes together" before any performance test.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

1. **Port an existing scheme rather than invent a new one**: the generic kernel `mmq_nt_kernel<7,2,0>` has long been a double-buffered pipeline; the q6_K-specific kernel copies the same `buf ^= 1` skeleton, reducing correctness risk.
2. **KDR 4→2 rather than hard-doubling smem**: the arithmetic in §2.2 — the same 29,696 B buys overlap, not depth; r38's KDR=8 regression (1097.8) was a data point from 20 minutes earlier, leaving zero room for illusions about 59,392 B.
3. **Pipeline A and B together**: the A side is a bulk uint4 LDG→STS (the r34 prepass transport), the B side is per-byte bitfield expansion + dsc reads; both live in the same RAW_STAGE macro and both move into the second buffer.
4. **Hold the register budget at 0 spill**: double buffering introduces per-plane stride constants and a buffer selector; ptxas recorded 85 → 87 regs, still 0 spill, and the 2 blocks/SM occupancy premise is untouched.

### 3.2 Key code

The excerpts below all come from r39 commit `f2b9e54`'s diff to `src/cuda_kernels.cu` (+77/−29).

**smem planes ×2 and per-buffer stride**:

```cuda
// r39: DOUBLE-BUFFERED staging — two copies of every per-kt plane so kt+1's
// global->smem expansion (the ql+qh recomb) overlaps kt's compute, hiding the
// B-staging latency that left r38 latency-bound. Layout per buffer b below.
uint8_t* qa8    = mmq_q6k_sh;                          // [2][KDR*NBI*32]
uint8_t* sda_q  = qa8    + 2 * KDR * MMQ_NBI * 32;     // [2][KDR*NBI*4]
uint8_t* qb_exp = sda_q  + 2 * KDR * MMQ_NBI * 4;      // [2][NBJ*KDR*32]
float2*  sds    = reinterpret_cast<float2*>(qb_exp + 2 * MMQ_NBJ * KDR * 32);
const int qa8_stride    = KDR * MMQ_NBI * 32;
const int sdaq_stride   = KDR * MMQ_NBI * 4;
const int qbexp_stride  = MMQ_NBJ * KDR * 32;
const int sds_stride    = KDR * MMQ_NBJ;
```

The base addresses of all four planes reserve two copies, and the macro selects the buffer via `b * stride` — the staging macro's body itself is unchanged word for word (A's uint4 copy, B's `expand_q6_elem` expansion, the dsc read all as-is); only the destination pointers become the `+ (size_t)(b) * stride` offset versions:

```cuda
#define RAW_STAGE_Q6K_BT(kt, b)                                                \
    do {                                                                       \
        uint8_t* qa8b   = qa8    + (size_t)(b) * qa8_stride;                   \
        uint8_t* sdaqb  = sda_q  + (size_t)(b) * sdaq_stride;                  \
        uint8_t* qbexpb = qb_exp + (size_t)(b) * qbexp_stride;                 \
        float2*  sdsb   = sds    + (size_t)(b) * sds_stride;                   \
        /* A: bulk LDG->STS of the pre-transposed qa8/sda (no math)      */    \
        ... ((uint4*)(qa8b))[off]   = ((const uint4*)(qa8g + qbase))[off];     \
        ... ((uint4*)(sdaqb))[off]  = ((const uint4*)(sdag + sbase))[off];     \
        /* B: expand KDR*32-chunk super-block ...                              \
        ... qbexpb[...] = (uint8_t)v;  /* expand_q6_elem, same as r38 */       \
```

**Main loop: prologue + interleave**. r38's `for { if(kt>0) stage(kt); barrier; compute }` is rewritten as:

```cuda
RAW_STAGE_Q6K_BT(0, 0);
__syncthreads();

int buf = 0;
for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
    // Overlap kt+1's global->smem expansion with kt's compute: stage into the
    // OTHER buffer (buf^1) while reading buffer buf (the mmq_nt<7,2> pipeline).
    if (kt + 1 < nktile) RAW_STAGE_Q6K_BT(kt + 1, buf ^ 1);

    const uint8_t*    qa8c   = qa8    + (size_t)buf * qa8_stride;
    const uint32_t*   sdaqc  = ... sda_q  + (size_t)buf * sdaq_stride;
    const uint8_t*    qbexpc = qb_exp + (size_t)buf * qbexp_stride;
    const float2*     sdsc   = sds    + (size_t)buf * sds_stride;

    for (int kd = 0; kd < KDR; kd++) {          // KDR=2: two 32-chunks per kt
        ...
        const uint8_t* qat = qa8c + (size_t)kd * MMQ_NBI * 32;
        ... ldmatrix A-frag / int8 B-frag / 16× mma.m16n8k16 / two += rescales ...
    }
    __syncthreads();   // loop-tail fence: the two duties of §2.3 (a plain fence at r39 time)
}
```

Key points: every compute-side pointer becomes the per-buffer `*c` version (in the diff, the three single-line hunks at 5451/5470/5483 are just reference replacements like `sds`→`sdsc` in the epilogue); the loop-tail `__syncthreads()` stays — it is now the only in-loop fence. The current tree (`src/cuda_kernels.cu` lines 6905-6929) still carries this skeleton; r53/r56 merely layered cp.async group waits (`gemm_cp_wait1`) on top, which this doc does not expand on.

Change-surface accounting: `src/cuda_kernels.cu` +77/−29 lines and `src/cuda.rs` 2 lines — the genuinely "new" parts are only three: the smem plane table, the macro signature gaining the `(kt, b)` pair, and the main loop's interleave structure; the staging macro's body (uint4 copy, `expand_q6_elem`, dsc read) and the compute body are untouched word for word. **Minimal diff surface = minimal verification surface**: the first explanation for parity/greedy going all-green is that the arithmetic path changed zero bits.

**Launcher: KDR 4→2 + smem expression ×2**:

```cuda
constexpr int KDR = 2;
const int smem = 2 * KDR * MMQ_NBI * 32   // qa8  (double-buffered)
               + 2 * KDR * MMQ_NBI * 4    // sda_q (double-buffered)
               + 2 * MMQ_NBJ * KDR * 32   // qb_exp (double-buffered)
               + 2 * KDR * MMQ_NBJ * 8;   // sds  (double-buffered)
```

The dispatch gate, the `(id/32)%8==0` whole-super-block alignment check, and the return-0 clean fallback on failure are all unchanged; the `cuda.rs` diff is only 2 lines (1+/1−, comment-only).

### 3.3 Pitfalls

1. **The half-pipeline of "double-buffer only the big plane"**: KDR=4 + B-only double buffering looks like it saves 12 KB, but in reality compute(kt) reads A while stage(kt+1) overwrites A — a data race, vetoed on correctness alone (§2.4). The minimal complete unit of pipelining is "all inputs of one kt iteration", not "the largest plane".
2. **The smem quote illusion**: looking only at the B expanded plane (16 KB) suggests ample headroom; adding back A (8 KB), sda_q (1 KB), and sds (4 KB) reveals KDR=4 double buffer = 59,392 B. Budgets must be priced per "whole-iteration input".
3. **Watch small register creep**: the stride constants ×4 plus the buffer selector took ptxas from 85 → 87 regs. 0 spill survived, but r40 would prove how thin this margin was (87 → 80, squeezing hard for 3 blocks/SM, is exactly where a 4 B spill got squeezed out).
4. **Two mechanical traps when restructuring staging**: the prologue stage, moved out of the loop, must be immediately followed by `__syncthreads()` (prevents a first-iteration race); typed-pointer strides scale by element, not byte (`sds` is `float2*`, so +1 = 8 B, hence `sds_stride = KDR * MMQ_NBJ` without the ×8) — both are silent killers of the "structure right, offsets wrong" kind; recount the fence accounting point by point and re-derive each stride.

## 4. Verification

- **parity 1/0**: logits deviation vs the baseline path within the 1e-3 gate — double buffering changes only buffer assignment and fence placement, no arithmetic order, so in theory it should be near-bitwise; the 1/0 record confirms no read/write race pollution from a misplaced fence (defends against losing either of §2.3's two duties).
- **greedy byte-identical**: the greedy output stream matches the pre-change one (defends against argmax knife-edge flips).
- **ptxas**: 87 regs / 0 spill (defends against register pressure breaking 2 blocks/SM).
- **suite 166/0/3**: full regression; notation passed/failed/skipped (defends against other quantization paths being collateral damage — this diff touched the RAW_STAGE macro; q4_K's same-named macro is independent, but the suite is the final "didn't break anyone else" evidence).
- **A/B interleaved 3/3**: same-window, same-binary pairing (defends against machine-drift fake deltas).

Why parity should be near-bitwise: double buffering changes no arithmetic — the expansion formula, the mma sequence, the two `+=` of the rescale, the accumulation order are all as they were; what changes is only which of the two buffers data lands in and where the fences sit. As long as the fence accounting is right (§2.3), the output is bit-identical to single buffering. So this round's parity gate is not really defending against arithmetic regressions but against **read/write races from misplaced fences** — errors of that kind feature sporadic dirty data, and the parity 1/0 and greedy byte-identical gates catch it together.

- **nsys kernel timing**: attn_v kernel 2,549,248 → 2,046,848 ns (defends against attribution errors like "the wall clock moved but really some other kernel got slower").

## 5. Results

| Metric | before → after | Note |
|---|---|---|
| whole-prefill (7B, same-window A/B median) | 1568.7 → 1777.5 tok/s (**+13.3%**) | master row 53 |
| attn_v q6_K kernel (nsys) | 2,549,248 → 2,046,848 ns (**−19.7%**) | direct evidence the exposed latency was overlapped |
| kernel compute share | 16.7% (r38) → 21.5% | idle issue slots reduced, but still far from saturated |
| ptxas | 85 → 87 regs, 0 spill | the cost of the double-buffer pointers |
| smem | 29,696 B (unchanged) | KDR 4→2 × double buffer = same footprint |
| vs llama.cpp (3325-eq anchor) | 2.13× → 1.87× | the engine got faster, so the relative multiple falls |

Three readings:

1. **+13.3% ≫ r38's +2.87%**: r38 bought occupancy (2 blocks/SM); r39 bought overlap. Occupancy cannot save exposed latency while the structure is fence-serial — more warps, but within each iteration all warps wait on staging together. Overlap hides the staging segment directly inside the compute segment: a structural deletion of time.
2. **The ratio between the attn_v kernel's −19.7% and the wall clock's +13.3%** is also self-consistent: attn_v is one of q6_K's two big tensors, and a kernel-level −19.7% amortized over whole-prefill with a discount lands at the +13.3% order of magnitude.
3. **compute 21.5% is still a hook**: once the overlap cashed in, the next bottleneck shows itself — latency's source shifts from "serial staging" to "per-byte B-expansion loads" (32 per-byte LDG.U8), which r41's uint4 widening will take up; and occupancy still sits at 2 blocks/SM, where r40's third resident block will also take a share. Both lines (r39→r41 latency, r39→r40 occupancy) start from this step's profile data.

**Baseline drift note**: r38 landed reporting 1561.9; this doc's in-window baseline is 1568.7 — absolute values from different session windows of the same code are not comparable (§0 table-reading convention), so deltas are always taken from same-window A/B pairs (+13.3%), never subtracted across windows.

**q6_K line postscript**: after this doc the occupancy and load-width lines run in parallel — r40 third resident block (+13.0%), r41 B-expand uint4 widen (+30.7%, kernel 1.70 → 0.654 ms); after r41 the q6_K line sat at only 1.27× vs llama, finally closing in the r45–r53 cp.async bundle. This doc's smem 29,696 B and fence skeleton lived on past r53 (the current tree's lines 6905-6929 are still the `RAW_STAGE(0,0)` + `buf ^= 1` skeleton).

## 6. Lessons

1. **Once occupancy is bought, the next lever is pipelining, not a deeper tile**: +2.87% (occupancy) was followed directly by +13.3% (overlap) — same kernel, same tile, same smem.
2. **The gain is "pure overlap"**: workloads unchanged, arithmetic unchanged; only the arrangement of buffers and fences changes. This is also why its parity risk is extremely low (no numeric order changes).
3. **Price the smem budget per "whole-iteration input", A and B planes together**; a half-pipeline (doubling only the big plane) is disqualified on correctness alone.
4. **Porting an existing scheme beats inventing one**: the `mmq_nt<7,2>` pipeline skeleton was reused as-is, concentrating correctness risk on a single fence accounting.

---
[← 41 · r38 q6_K BT-style raw-byte mma](41-r38-q6k-bt-rawbyte-mma.md) · [Index](./README.md) · [43 · r40 third resident block](43-r40-third-resident-block.md) →
