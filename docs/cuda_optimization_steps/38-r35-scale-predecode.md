# 38 · r35 — sda scale predecode: a total SASS win, a wall-clock tie (REVERTED)

> **Result**: 7B q4_K BT-kernel prefill **1493.2 → 1486.3 tok/s (−0.46%, inside the noise
> band; bar +1.5%)**. At the SASS level the lever fully cashed out — `SHF` 64→0, `HADD2`
> 64→10, net **−130 instructions** — yet the wall clock did not move: the decode
> instructions had been executing in the IMMA's shadow all along; cutting them frees no
> critical resource.
> **Commit**: `6112db3` (docs record only; the code change was reverted, `cmp-verified`
> identical to HEAD, no trace in the current tree). **Date**: 2026-09-05 (the work ran
> late on 09-04 into the morning of 09-05).

## 1. Background — where things stood

r34 (2026-09-04) had just landed the quantize-transpose prepass, P6's largest
single-mechanism gain: +9.72% (1364.2 → 1496.8 tok/s @ the 3354-tok window). Its
mechanism moved the A-side layout transform wholesale out of the kernel —
`quantize_q8_0_pad40_t` pre-produces the qa8 / packed-d|ssum planes transposed into the
mma consumption layout, and `mmq_raw_nb_bt_kernel`'s A staging degenerated from
"per-element index math + swizzle" to a bulk LDG→STS. r32's census had said staging was
21% of kernel instructions; after r34 that path is "zero-math".

So where is the residual? r35's first step was a **new four-region instruction census**
of the post-r34 BT kernel: prolog+stage0 387 / in-loop staging 286 / **compute-kd 1514**
/ epilogue 117. compute-kd is 66% — the only heavyweight region. And inside compute-kd,
the largest "nameable, cuttable" block is the **sda d|ssum decode**: roughly 64 `SHF`
sign-extensions + 64 `HADD2`s (f16→f32) + 64 `I2FP`s (i32→f32) per kt — all spent
restoring, on the spot, the `d` (f16, low 16 bits) and `ssum` (i16, high 16 bits) packed
into one u32 into the f32/i32 operands the mma rescale needs.

Hence r35's hypothesis: **the decode is pure ALU, and the prepass runs once per token
while the kernel runs it per (block, kt)** — hoist it into the prepass and the compute
loop's each level nets ~192 fewer ALU instructions; the wall clock should move. The
intuition is not baseless: r22 had proven that "per-ldmatrix address ALU eats the freed
wavefronts", showing ALU had genuinely blocked progress in this class of kernel. Without
this step the census stops at an unverified named candidate; with it, win or lose, the
boundary of the "composition residual" gets drawn tighter (the other candidate, A-frag
LDSM, is left for r36).

## 2. Principle — the GPU mechanism

First dissect the "decode" chain to SASS grain. In the A-side scale plane, each (token,
chunk) is one u32:

```
bit  0..15 : d     (f16 bit pattern, the quantization scale)
bit 16..31 : ssum  (i16, the sum of the chunk's 32 q8 values, used by the rank-1 term)
```

The rescale needs `d` as f32 and `ssum` as i32. Recovering them from the packed u32 is
three SASS instructions: `SHF.R.S32.HI` (arithmetic-shift the high half-word into a
sign-extended i32), `HADD2.F32` (f16→f32 conversion, taking one of HADD2's dual
half-word lanes), and `I2FP` (i32→f32). All of them issue on the FP/INT pipe.

But the compute loop's master is the tensor pipe: per kt a warp issues **64 IMMA**
(`mma.m16n8k32.s8`) — the throughput mainstay confirmed repeatedly since r12. The key is
that on GB10's SMs the FP/INT pipe and the tensor pipe are **parallel issue resources**
— once the 64 IMMA fill the tensor pipe, the compiler scheduler stuffs the
data-independent ALU (no dependency on the mmas) into the idle issue slots between them.
These instructions **occupy slots that would otherwise idle** and lengthen no critical
path. r35, after the fact, named this phenomenon "the IMMA shadow": instructions in the
shadow are free.

The predecode scheme's benefit and costs are both easy to compute:

- **Benefit**: 64 SHF + 64 HADD2 + 64 I2FP ≈ 192 fewer ALU per kt in the compute loop.
- **Cost 1**: the scale plane goes from 4 B/token/chunk to 8 B (an f32 d plane + an i32
  ssum plane), smem 43,008 → 45,056 B, and the read side's LDS instruction count rises
  (one packed u32 plane becomes two planes).
- **Cost 2**: addressing two planes occupies more registers than one packed plane.

If the ALU pipe is the bottleneck, benefit > cost; if the IMMA pipe is the bottleneck
and the ALU is in the shadow, benefit = 0 while the costs are paid in full — exactly the
two worlds this experiment had to distinguish.

## 3. Implementation

### 3.1 Design choices (why this shape)

- **Change the prepass, not the B side.** d/ssum are A-side (activation-quantization)
  properties, and the prepass `quantize_q8_0_pad40_t` already computes them — merely
  packed as f16|i16. The hoisting direction is natural: once per token in the prepass vs
  once per (block, kt) in the kernel.
- **The layout skeleton stays.** Keep r31's q-major region-split addressing (already
  proven bank-conflict-free); only swap "one packed u32 plane" for "[d f32][ssum i32]
  two planes", growing each chunk's smem scale region from 256 B to 512 B.
- **Budget first, act second.** The r34 kernel's smem = qa8 (8×64×32 = 16,384) + sda_q
  (8×64×4 = 2,048) + qb_raw (128×128 = 16,384) + sds (8×128×8 = 8,192) = **43,008 B**;
  the scale plane's +2,048 B brings 45,056 B, still inside the 2-blocks/SM dynamic smem
  budget. On registers, r34 is 103 regs / 0 spill and the planar addressing was expected
  to add ~10 — as long as it stays under 128, 2 blocks/SM survives (256 thr × 128 regs ×
  2 blocks = 65,536, exactly the per-SM register-file ceiling).

### 3.2 Key code

**BEFORE (an era-tree excerpt; after r35's revert the current tree matches this)** — the
decode segment in the compute loop that r35 wanted to delete (`mmq_raw_nb_bt_kernel`,
executed per kt):

```cuda
const uint32_t* sda_blk = sda_q + (size_t)kd * MMQ_NBI
                          + (size_t)(lane >> 2) * 4;
const uint4 s0 = *(const uint4*)(sda_blk);
const uint4 s1 = *(const uint4*)(sda_blk + 32);
#pragma unroll
for (int g = 0; g < 4; g++) {
    float da_q[2];
    int sa_q[2];
    const unsigned w0 = g == 0 ? s0.x : (g == 1 ? s0.z : (g == 2 ? s1.x : s1.z));
    const unsigned w1 = g == 0 ? s0.y : (g == 1 ? s0.w : (g == 2 ? s1.y : s1.w));
    da_q[0] = h2f((unsigned short)(w0 & 0xFFFF));  // f16 d → f32   (HADD2.F32)
    sa_q[0] = (int)(short)(w0 >> 16);              // i16 ssum → i32 (SHF.R.S32.HI)
    da_q[1] = h2f((unsigned short)(w1 & 0xFFFF));
    sa_q[1] = (int)(short)(w1 >> 16);              //                (I2FP at the dma multiply)
    const float dma[2] = { da_q[0] * (float)sa_q[0],
                           da_q[1] * (float)sa_q[1] };
    /* … rank-1 + main-term rescale, multiplied by dsv/dmv and accumulated into sum[] … */
}
```

r35's AFTER shape (reverted; narrated from the record): on the prepass side, the packing
segment above

```cuda
// Packed d|ssum (r31 Q-major region split of the old sda_q).
const int g = r >> 4, t15 = r & 15, q = t15 & 7, half = t15 >> 3;
const int rg = g >> 1, gsel = g & 1;
size_t sbase = ((size_t)tb * nchunk + b) * MMQ_A_SDASZ
               + (rg * 32 + q * 4 + gsel * 2 + half) * 4;
__half dh = __float2half(d);
uint16_t dbits = *reinterpret_cast<uint16_t*>(&dh);
*reinterpret_cast<uint32_t*>(ysda + sbase) =
    (uint32_t)dbits | ((uint32_t)(uint16_t)ssum << 16);
```

became a direct write of two planes (`*(float*)` for d, `*(int*)` for ssum — 8
B/token/chunk, with f32/i32 exact representations of the quantized values), and the
kernel side read straight into `da_q[]/sa_q[]` — the `h2f`/sign-extension trio vanishing
from the source.

### 3.3 Pitfalls

- **The smem budget was being squeezed a third time.** r31 had already compressed sda_q
  from 4,096 to 2,048 B, and r35 added 2,048 B back — an "smem for ALU" trade needs a
  budget table; 45,056 B is not far from the 2-blocks/SM ceiling, and any further
  inflation would push occupancy down (the r38-era KDR=4 lesson is the same mechanism in
  reverse).
- **Deleted instructions grow back elsewhere.** SASS showed `LDS.128` 32→48: with one
  packed u32 plane becoming two, each (kt, lane) issues more scale reads. "Net effect"
  must be read in both directions — this round netted −130, but structurally the LDS
  increase is half of the later tie explanation.
- **Registers rose instead of falling** (103 → 113). Planar addressing added arithmetic;
  ptxas spent 10 more registers — still under the 128 red line, but the "deleting code =
  saving registers" intuition does not hold.

## 4. Verification

- **Parity dump 1/0 + greedy-32 byte-identical**: an f32 d plane replacing "f16 pack →
  in-kernel h2f" is theoretically bit-exact (the f32 value is `__half2float`'s exact
  expansion), but the unchanged rescale multiply order had to be proven — defends
  against numeric-path drift.
- **SASS census (cuobjdump) before/after**: proves the change actually landed in the
  SASS (SHF 64→0, HADD2 64→10, LDS.128 32→48, net −130) — defends against the r30-style
  "cancelled out by the compiler's CSE; wrote an air lever"; r30's lesson is precisely
  to read the SASS before concluding, and this time the SASS really did change.
- **5-round interleaved A/B, median taken**: defends against co-tenant drift reading
  noise as signal (the campaign's standard practice before r59b).
- **`cmp-verified` = HEAD**: confirms the post-revert working tree is byte-identical to
  the pre-change state.

## 5. Results (with the veto mechanism)

| Layer | before → after | Verdict |
|---|---|---|
| SASS `SHF` | 64 → **0** | lever cashed out |
| SASS `HADD2` | 64 → 10 | lever cashed out |
| SASS `LDS.128` | 32 → **48** | cost cashed out |
| SASS net instructions | 2304 → 2174 (−130) | lever cashed out |
| ptxas | 113 regs / 0 spill, smem 45,056 B, 2 blocks/SM | budget held |
| Parity / greedy-32 | 1/0 / byte-identical | numerics clean |
| **Wall (5-round median)** | **1493.2 → 1486.3 tok/s (−0.46%)** | **inside the noise band, NEUTRAL** |

**Veto mechanism**: 128 int/fp ALU instructions deleted, the wall clock moved 0.0%, and
16 more LDS.128 were paid. The only self-consistent explanation: these decode
instructions were scheduled in the **idle FP/INT issue slots** between the 64-per-kt
IMMA (the IMMA shadow) all along — they never competed with the tensor pipe for any
resource; delete what is free and of course the wall does not move, and the LDS increase
was absorbed by same-band noise. The compute loop is therefore **not ALU-bound**, and
the residual is the **inherent combination** of A-frag LDSM consumption + fp rescale —
closing the loop with r32/r33's conclusions; the "decode instruction class" is closed as
a wall-clock lever from here on.

**When a retry is worthwhile**: (a) if some tiling rework makes the loop no longer
tensor-bound (e.g. enlarging A-frag reuse so IMMA pressure drops and the ALU surfaces),
predecode becomes a candidate again; (b) the B side's analogous decode (d/dmin per
super-block) later took a different road — r56/r59 pre-multiplied `d·sc` into a
registration-time f32 plane (W_dsc), and that one succeeded because it paired with the
cp.async pipeline and the B-side decode genuinely competed with the mma — the same word
"predecode", but completely different criteria (in-shadow or not).

## 6. Lessons

1. **Instructions hiding in the tensor-core's shadow are free** — cut only work that
   competes with the mma pipe; cutting shadow instructions merely vacates idle issue
   slots.
2. **Instruction count is not the wall clock**: −130 SASS = 0.0% wall; ask "which pipe
   do these instructions run on, and is that pipe saturated" before deciding to act.
3. **"Delete instructions" levers tend to grow bytes back elsewhere** (LDS.128 32→48) —
   a census must always read the net effect; a single class's decrease may be a
   transfer, not an elimination.
4. **A total SASS win + a wall-clock tie is a high-value measurement**: it narrows the
   composition residual to the LDSM/IMMA + fp-rescale body itself, directly framing
   r36's wavefront-economics question.

← [37-r34-quantize-transpose-prepass](37-r34-quantize-transpose-prepass.md) · [Index](./README.md) · [39-r36-a-frag-wavefront](39-r36-a-frag-wavefront.md) →

