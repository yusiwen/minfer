# 37 · r34 — The quantize-transpose prepass: layout-transform locality (LANDED, +9.72%)

> **Result**: the A-side layout transform (the r22 XOR swizzle + r31 q-major sda
> repack index math) moves out of the mma kernel's per-tile staging into a per-GEMM
> quantize prepass — llama.cpp's `quantize_mmq_q8_1` design. New
> `quantize_q8_0_pad40_t` (bit-identical, transposed layout, zero-filled pad tokens)
> + `mmq_raw_nb_bt_kernel` (A staging degenerates to bulk LDG→STS with no per-element
> index math): **1364.2 → 1496.8 tok/s (+9.72%, interleaved 4-pair all positive, no
> interval overlap)**, prepass 0.405 vs 0.446 ms (**0.908× — smaller, not bigger**),
> bt kernel **103 regs / 0 spill** (NB 109 — the direct evidence that the staging
> index math left the kernel). P6's largest single-mechanism gain since r28.
> **Commit**: `ba977bf` (`src/cuda.rs` +135, `src/cuda_kernels.cu` +309). **Date**: 2026-09-05.

## 1. Background — where things stood

After two consecutive negative results (r32/r33, docs 35/36), the line stood at a thoroughly cleaned-up crossroads. r32 had sealed the "instruction count" axis: staging addressing already hoisted by ptxas, epilogue structurally capped, remaining integer-ALU surplus = A-frag LDSM + intrinsic sda/sds decode + fp rescale, all "compiler floor". r33 had sealed the "loop organization" axis: source-level reordering produces byte-identical SASS. Together the two verdicts point at the only exit — **to change ptxas's output, the DAG itself must change**. r33's scope caveat wrote that "way of changing" as a will: flip the mma operand orientation to A=weights (eliminating the per-tile activation A-frag LDSM), or move the A-side layout transform out of the kernel. The former was out of budget; r34 took the latter — a llama-faithful narrow slice.

The problem itself was restated in §11.14 as **layout-transform locality**. Look at what the minfer NB kernel's staging does: for every (64-token block, od-tile column) combination it reads the activations' qs/d/ssum out of the native token-major 40 B chunks, then writes them into the mma-consumption smem layout via the **per-element index math** of the r22 XOR swizzle and the r31 q-major repack. And the grid is `(nt/64, od/128)` — **the same 64-token A tile gets restaged once per od-tile column**: 28 times per buffer for the od=3584 projections (q/o/down), and 148 times for od=18944 gate/up. The transform's index math scales linearly with the restage count, while the transform itself is od-independent — pure repeated labor.

llama.cpp never had this problem: its quantize-side `quantize_mmq_q8_1` **pre-transposes** the activations into exactly the layout the mma kernel consumes, inside the prepass (both the transpose and the intra-block shuffle happen on the write side), so the kernel's A staging is a near-trivial copy. Its planes are `y_qs` (qs) / `y_dm` (packed scale half-words) — one-to-one with minfer's new `yqs`/`ysda` planes; the activation-side trivial LDS r33 quoted ("faster than load_ldmatrix") reads exactly this pre-transposed plane. r32's census priced the contrast: staging = **21% of kernel instructions** (455 per-kt) — r34 relocates the "transform" share of that 21%.

One more anchor: whole-prefill was still on the ~1440 tok/s plateau (the r31 window), while the campaign target (set at r6) was f16 parity ~24 TMAC/s / llama parity ~30. r34 is among the last levers of the "q4_K kernel itself" — a week later r37's attribution would show q6_K is the new wall (51.2% of wall), but that is another line's (Era D's) business.

## 2. Principle — the GPU mechanism

**Layout-transform locality — splitting the two costs.** Each byte of A-side staging costs two things:

- **Copy**: moving bytes from global to smem. No ALU, pure LSU, and repeated reads are absorbed by cache — paid again per restage, but cheap.
- **Transform**: computing where each byte lands in smem. The XOR swizzle's shifts/xors/masks, the q-major repack's region indices — pure ALU, register-occupying, **scaling linearly with the restage count**.

Before r34 the two frequencies were bound together: transform count = copy count = restage count (the od/128 column count). r34 moves the transform into the prepass so it is paid once per buffer; the copy stays in the kernel but degenerates into a `uint4` bulk copy:

```text
restages (per A buffer) = grid.y = ceil(od / 128)
  od = 3584 (q / o / down) → 28 times        od = 18944 (gate/up) → 148 times
transform cost: old = once × grid.y; new = once × 1     ← the multiplication r34 removes
copy cost: both sides = once × grid.y (old: a 455-instruction mixed stream; new: pure uint4 copies)
```

**Transform count ÷28 for the od=3584 projections, and the in-kernel index math disappears wholesale**. As for re-reading the same plane across od-tile columns: it is the same class as the B side's weight re-reads (r19 proved that class is absorbed by L2 and creates no DRAM traffic) — not a new cost.

**The plane layout's arithmetic.** The planes the new prepass emits are the exact global mirrors of the smem layout: one swizzled 2048 B qs block per (64-token block, chunk) (`[ntb][nchunk][2048]`), one 256 B packed d|ssum block (`[ntb][nchunk][256]`). Against the old path: native pad40 is 40 B per (token, chunk) (4 B d + 4 B ssum + 32 B qs), so each block-chunk read 64×40 = **2560 B** and wrote smem 2048 + 256 = **2304 B**; the new plane is directly **2304 B** per block-chunk — d and ssum packed into one u32 (f16 d + i16 ssum) halves the scale bytes from 8 → 4 B/token/chunk. At nt=3354, id=3584: the planes are ≈ 11.6 MiB (qs) + 1.45 MiB (sda) ≈ **13 MiB** — a one-time per-GEMM activation-buffer cost.

**The bulk copy's issue arithmetic.** Each k-tile's A-segment copy: qs side `KDR × NBI × 32 / 16` = 8×64×32/16 = **1024 uint4s** (16 KB), sda side 8×64×4/16 = **128 uint4s** (2 KB); 256 threads issue 4-5 `LDG.128 → STS.128` each, with contiguous per-thread addresses = perfectly coalesced. Against the old path's same segment: 455 mixed instructions with XOR/shift/mask. The two call sites `RAW_STAGE_NB_BT(0)` / `if (kt > 0) RAW_STAGE_NB_BT(kt)` are identical to the NB kernel's — **r20 split-phase double buffering as before** (prefetch the next tile while computing the current); only the stage macro's internals change.

**Why the prepass does not grow — it shrinks.** The quantize body itself (8×float4 reads, the amax tree, `rintf/clamp`, the ssum sum) is output-layout-independent — bit-identical in both versions. Only the write side changes: 8 u32 qs writes + 1 u32 sda write per thread, using **the same** addressing formula as the old smem STS (the same swizzle routine) with the target switched from smem to global. Packing additionally halves the written scale bytes. Measured **0.405 vs 0.446 ms (0.908×)** — the transpose carries no tax; it nets a small saving.

**The register mechanism — why regs drop.** The per-kt staging index math (the swizzle's XOR/shifts, the repack's region indices) is a real source of register pressure: their live ranges span the LDG batch and the STS write-back. With the transform out of the kernel, this set of intermediates vanishes — **109 → 103 regs, 0 spill kept, smem 43,008 B unchanged → 2 blocks/SM kept**. Occupancy unmoved while per-tile instructions shrink: a clean positive combination.

**The consumption-side contract — what layout ldmatrix demands.** Why must the old staging XOR-swizzle, and why must the new plane replicate it byte for byte? Because the consumer's `ldmatrix.sync.aligned.m8n8.x4` reads 8-row × 16 B tiles from smem, and if the 8 rows' addresses crowd into the same bank group, one ldmatrix tears into multiple serial replays. r22's swizzle (the inter-row XOR `(R>>2)&7` term) exists precisely to stagger adjacent tiles' rows across banks; the kernel-side `G[4]` precompute (r22: each tile's start computed once) then moves the address ALU from every ldmatrix into the prolog. r34 does not touch one hair of this contract — it only moves "who arranges the bytes into this layout" from the kernel's per-tile staging to the prepass's once-per-buffer. The layout is unchanged, so the consumer's read pattern need not change; the byte-exactness validator guards exactly this contract.

**Consumer-side invariance is the proposition's boundary.** A-frags are still read from smem by ldmatrix — and ldmatrix does not consume an arbitrary layout but the **m8n8 bank-friendly row-staggered layout**, which is why the swizzle existed (r22). r34 changes neither the consumption layout nor the B weight staging, the SDS fold, the mma loop, or the fp32 write-back. The only thing that changes is "how the qa8/sda_q tiles are produced" — guaranteeing any wall-clock change attributes to the staging alone (the "eliminate LDSM" half-step of the r33 caveat was **not** done).

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**A new kernel; the old one untouched.** The minimal-intrusion shape for swapping the A-side supply is a copied `mmq_raw_nb_bt_kernel` (bt = bulk-transposed) with `mmq_raw_nb_kernel` kept alive as-is. More than conservatism: the old kernel becomes the **A/B integrity control** — its SASS in the new binary must be byte-identical to the old binary's, proving the A/B difference can only come from the bt kernel itself. Forensic check: `git show ba977bf --stat` shows `src/cuda_kernels.cu` +309 lines; `quantize_q8_0_pad40_t`, the `MMQ_A_*` constants, `mmq_raw_nb_bt_kernel`, and `RAW_STAGE_NB_BT` were all introduced by that commit; the current tree's A segment is still its introduced form (the later r59 DSC plane touched only the B-side sds staging).

**Gate shape**: `MINFER_MMQ_A_TRANSPOSE=1`, paired with `MINFER_MMQ_RAW_NB=1` (current-tree dispatch: `if nb && at && kd == 8`). The opt-in experimental gate makes rollback free; this gate later carried a building — r49 added shared-A dedup on top, r51/r52 folded quantization into producers, r59 added the W_dsc plane, and r60 flipped the default on with the whole gate set. One detail: the native pad40 quantize (`quantize_q8_0_pad40`) was not deleted — after r34 it fires only when the bt path is unavailable (the dispatch comment's own words: "the native q8 buffer is only filled on the (rare) bb-bt fallback below").

**Plane layout = mirror of the smem layout** is the design's load-bearing wall: only when the global plane's byte order matches the smem target's byte order exactly can staging degenerate into an index-math-free bulk copy. To that end the prepass's write side embeds the old STS's swizzle routine directly (byte-for-byte replication; see §3.3).

**Pad-token zero fill**: blocks whose tail has fewer than 64 tokens emit all-zero pad rows (d=0, ssum=0, qs=0) — the transposed plane is thus independent of buffer-reuse history (deterministic), and the write-back's `i < nt` guard guarantees pad rows never land in C. "Zero fill instead of skipping writes" also keeps the bulk copy length constant (`KDR*NBI*32` bytes exactly), so the kernel needs no length special-casing for tail blocks.

**The launcher** keeps the native quantize's shape (a 1-D grid covering all (pad-token, chunk) pairs):

```cuda
// src/cuda_kernels.cu:3536-3545 (launch_quantize_q8_0_pad40_t — introduced in r34, unchanged)
void launch_quantize_q8_0_pad40_t(
    const float* x, uint8_t* yqs, uint8_t* ysda,
    int dim, int nt, int nchunk, int ntb, cudaStream_t stream
) {
    long long total = (long long)ntb * MMQ_A_BLK * nchunk;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    quantize_q8_0_pad40_t<<<(int)grid, block, 0, stream>>>(x, yqs, ysda, dim, nt, nchunk, ntb);
}
```

At nt=3354, id=3584: `total = 53 × 64 × 112 = 379,904` threads → 1,484 256-thread blocks, one launch covering the whole plane; each thread writes 9 u32s (8 qs + 1 sda) — exactly the native quantize's write granularity, only the target address formula differs.

### 3.2 Key code

**The prepass kernel** (`src/cuda_kernels.cu`, introduced in r34; current-tree version — after r51, the producer-fused path reuses exactly this quantize body):

```cuda
// src/cuda_kernels.cu:753-810 (quantize_q8_0_pad40_t — excerpt)
__global__ void quantize_q8_0_pad40_t(
    const float* __restrict__ x,
    uint8_t* __restrict__ yqs,    // [ntb][nchunk][2048] ← the swizzled qs plane
    uint8_t* __restrict__ ysda,   // [ntb][nchunk][256]  ← the packed d|ssum plane
    int dim, int nt, int nchunk, int ntb
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pad = ntb * MMQ_A_BLK * nchunk;      // one thread per (pad-token, chunk)
    if (tid >= total_pad) return;
    int t = tid / nchunk;
    int b = tid % nchunk;
    const int r = t & (MMQ_A_BLK - 1);  // local token 0..63
    const int tb = t >> 6;              // 64-token block index
    // …quantize body: 8×float4 reads + amax tree + rintf/clamp + ssum (bit-identical to
    //   quantize_q8_0_pad40; pad tokens (t >= nt) emit all zeros, keeping the plane deterministic)…
    // Swizzled qs write — byte-identical to the NB kernel's old smem staging
    //   (qa8 + (R&~3)*32 + ((((R&3)<<1 + (u>>2)) ^ ((R>>2)&7)) << 4) + (u&3)*4).
    const int t4 = r & 3, grp = r & ~3;
    const int xswz = (r >> 2) & 7;
    size_t qbase = ((size_t)tb * nchunk + b) * MMQ_A_QASZ + grp * 32;
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        const int off = (((t4 * 2 + (u >> 2)) ^ xswz) << 4) + (u & 3) * 4;  // ← the same swizzle
        *reinterpret_cast<uint32_t*>(yqs + qbase + off) = packed[u];         //   just written into global
    }
    // Packed d|ssum (r31 Q-major region split of the old sda_q).
    const int g = r >> 4, t15 = r & 15, q = t15 & 7, half = t15 >> 3;
    const int rg = g >> 1, gsel = g & 1;
    size_t sbase = ((size_t)tb * nchunk + b) * MMQ_A_SDASZ
                   + (rg * 32 + q * 4 + gsel * 2 + half) * 4;
    __half dh = __float2half(d);
    uint16_t dbits = *reinterpret_cast<uint16_t*>(&dh);
    *reinterpret_cast<uint32_t*>(ysda + sbase) =
        (uint32_t)dbits | ((uint32_t)(uint16_t)ssum << 16);   // f16 d + i16 ssum = 4 B
}
```

The bt kernel's header comment (r34's original, `src/cuda_kernels.cu:6448-6455`) states the design's boundary in one breath:

```cuda
// --- P6 r34: NB kernel with the A-side layout transform relocated into a
// quantize-transpose prepass (MINFER_MMQ_A_TRANSPOSE=1). Byte-identical
// compute to mmq_raw_nb_kernel (same qa8/sda_q smem content, same ldmatrix
// fragment reads, same rescale) — only the A STAGING differs: the qs plane and
// the packed d|ssum are emitted PRE-TRANSPOSED (quantize_q8_0_pad40_t) so the
// per-(kt, warp) A reads become contiguous bulk LDG->STS with no per-element
// index math (the r22 XOR swizzle and the r31 q-major sda repack are baked into
// the prepass layout). The B (weight) + SDS staging is unchanged.
```

**After — the bt kernel's A staging** (the A segment of `RAW_STAGE_NB_BT`; introduced in r34, the A path unchanged to this day; the current tree's DSC/B-side sds is r59's later story):

```cuda
// src/cuda_kernels.cu:6486-6498 (RAW_STAGE_NB_BT — A side)
/* ---- A: bulk LDG->STS of the pre-transposed qa8/sda (no math) ----*/
{
    const size_t qbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_QASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16;        \
         off += blockDim.x)                                             \
        ((uint4*)(qa8))[off] = ((const uint4*)(qa8g + qbase))[off];     // ← pure uint4 copy
    const size_t sbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_SDASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 4) / 16;         \
         off += blockDim.x)                                             \
        ((uint4*)(sda_q))[off] = ((const uint4*)(sdag + sbase))[off];   // ← same, no index math
}
```

**Before — the NB kernel's A staging** (`RAW_STAGE_NB`, LDG batch + swizzle STS, excerpting the STS write-back; the full contrast is doc 35 §3.2):

```cuda
// src/cuda_kernels.cu:6268-6277 (RAW_STAGE_NB — the index math every tile paid before r34)
_Pragma("unroll")
for (int i = 0; i < KDR * 2; ++i) {
    const int x = threadIdx.x + i * 256;
    const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),
              kd = x / (8 * MMQ_NBI);
    const int R = kd * MMQ_NBI + r;
    *(unsigned*)(qa8 + (size_t)(R & ~3) * 32                       // ← these three addressing lines
        + (size_t)(((((R & 3) << 1) + (u >> 2))                    //   all vanish in the
                    ^ ((R >> 2) & 7)) << 4)                        //   bt kernel
        + (size_t)(u & 3) * 4) = av[i];
}
```

**The Rust-side dispatch** (`src/cuda.rs`; the comments are r34's original, the cache logic is r49/r52's later story):

```rust
// src/cuda.rs:3288-3307 (prefill_mmq's routing branch, excerpt)
// P6 r34: relocate the A-side layout transform out of the mma
// kernel into a quantize-transpose prepass (llama.cpp's design).
// Under MINFER_MMQ_A_TRANSPOSE=1 the activations are emitted
// PRE-TRANSPOSED (quantize_q8_0_pad40_t) so the NB kernel's A
// staging is a bulk LDG->STS; …
let mut nb_ok = false;
if nb && at && kd == 8 {
    let nchunk = (id / 32) as i32;
    let ntb = ((nt as i64 + 63) / 64) as i32;
    // r49: cache-backed transposed A-quantize prepass (reuses
    // the previous same-A matmul's qa8g/sdag when consecutive).
    let (qa8g, sdag) = self.mmq_quantize_transposed(
        x as *const f32, id as i32, nt as i32, nchunk, ntb, stream,
    );
    // …(launch mmq_raw_nb_bt_kernel; null planes fall back to the NB kernel)…
}
```

### 3.3 Pitfalls

- **The swizzle routine must be replicated byte for byte, with the formula written into the comment.** The qs write side's comment embeds the full formula — because its correctness criterion is not "looks right" but "byte-identical to the old smem staging's product". A byte-exactness validator was written for exactly this: 9 shapes (including nt values that are not multiples of 64) comparing the planes against the old path's smem content, 0 mismatches. The non-multiple-of-64 nt values are deliberate: the tail blocks' all-zero pad rows, the grp/tb out-of-range index behavior, and the bt kernel's write-back guard are only truly exercised on tail blocks.
- **The ncu census failed this round.** `ncu` injection failed for both `mmq_raw_nb_kernel` and `mmq_raw_nb_bt_kernel` ("Unknown Error on device 0" — a platform/toolchain limitation), so the instruction census was unobtainable. The landing was not blocked: the mechanism is corroborated by three independent pieces of evidence — the register drop (109→103, the static evidence that the staging index math left the kernel), the prepass timing (0.908×), and the wall clock (+9.72%).
- **"Transpose" is easy to read as a cost.** Intuition says writing an extra transposed plane should be taxed; in fact 0.908×. The lesson: do not estimate cost from a name — the transform's ALU did not grow (the same swizzle formula), and the scale packing even saved 256 B/block-chunk of writes.
- **The discipline of not deleting the old kernel.** `mmq_raw_nb_kernel` stays as the control and fallback path (serving normally when the planes are missing or the gate is off), which gives the A/B integrity gate a checkable anchor. The bt kernel's B/SDS segments are verbatim identical to the NB kernel's — maintaining two copies is the price of duplication, exchanged for the comparability of "the difference is only in the A segment": any B-side drift in a SASS diff is immediately suspicious.

## 4. Verification

- **byte-exactness validator (9 shapes, including non-multiples-of-64 nt)**: defends against "one XOR of the replicated swizzle written wrong" — the transposed planes must be byte-identical to the old smem staging, the foundation of the entire "supply swap only" proposition. The 0 mismatches cover every (tb, chunk) index corner: multi-block, tail-block pad rows, and the 64-token alignment boundaries.
- **plain NB kernel SASS byte-identity (across the old and new binaries)**: A/B integrity — proves the control was untouched and the performance difference can only come from the bt kernel.
- **ptxas resource audit**: bt 103 regs / 0 spill, smem 43,008 B unchanged — defends against an occupancy regression (2 blocks/SM must hold) and doubles as static mechanism evidence.
- **`cuda_prefill_mmq` parity 1/0 + greedy-32 token identity (matching the f16 default path)**: defends against the layout relocation changing any numerics.
- **Prepass timing contrast (0.405 vs 0.446 ms)**: defends against "the transpose hiding its tax in the prepass" — the wall-clock gain must be proven not traded away on the quantize side.
- **Suite 166/0/3**: defends against cross-shape regressions.

In the campaign's gate numbering (the `ba977bf` commit message's own words): **Gates 1-6,8 green** — byte-exactness, parity, greedy identity, suite, interleaved A/B, and the resource audit all green; the only absentee is the ncu census (platform injection failure), substituted by the three alternative pieces of evidence. This "gate numbers + absence declaration" combination later became the docs-commit standard signature.
- **Interleaved 4-pair A/B (7B q4_k_m @3354-token prefill, same window)**: 1364.2 → 1496.8, every pair positive with no interval overlap — defends against co-tenant/drift reading the signal as noise.

## 5. Results

| Metric | before | after |
|---|---|---|
| whole-prefill (7B q4_k_m @3354 tok, 4-pair median) | 1364.2 tok/s | **1496.8 tok/s (+9.72%)** |
| quantize prepass | 0.446 ms (native pad40) | **0.405 ms (0.908×)** |
| mma kernel registers | 109 regs / 0 spill | **103 regs / 0 spill** |
| smem / occupancy | 43,008 B / 2 blocks/SM | unchanged |
| suite | — | 166/0/3 |

+9.72% is the P6 line's largest single-mechanism gain since r28 (the NB kernel landing, +2.56%) — and its source is not a faster mma or fewer instructions but **moving one job out of the wrong place**. One anchor warning: the r31 window recorded 1439.40 tok/s, yet r34's window baseline is 1364.2 — **absolute values are not comparable across sessions** (machine state / co-tenant load); this step's credibility rests entirely on the same-window interleaved 4-pair A/B (the later "r59b lesson" codified this rule). The master table's vs-llama column records "—": before r37 the campaign ran on the opt-in MMQ path with no whole-prefill vs-llama record to cite (the §0 reading convention).

**The mechanism chain, replayed.** Read the three steps in sequence: r32's census drew the map (staging 21%, epilogue ~0.4%, compute hot path uncuttable) → r33 falsified "reordering saves instructions" (SASS identity) → r34 changed the question ("must this 21% be paid inside the kernel?" — answer: no). +9.72% did not fall from the sky: it is the composite of the map (r32), elimination (r33), and the designed relocation (r34) — the campaign methodology's most complete specimen of "negative results paving the way for positive ones".

**A correction to the mechanism attribution**: r33 called the residual "composition" — r34 proves the word is not monolithic. The staging component inside "composition" is actually a **layout-transform locality** problem and can be moved away; the A-frag LDSM consumption and fp rescale remain intrinsic (r32's conclusion stands as written). The residual was split smaller again.

**Where the line went next**: r35 (sda scale predecode into the prepass, −0.46%, REVERTED — instructions hiding in the IMMA's shadow are not worth cutting), r36 (A-frag wavefront economics, falsified in its H1), r37 (post-parity attribution: the q6_K GEMM is 51.2% of the wall — with MMQ having made q4_K fast, the ball passed to the q6_K line) — Era C closes here, and Era D's q6_K/FA/prepass lines take the baton. This doc's plane layout `[ntb][nchunk][2048/256]` remains the foundation of the entire A-side stack to this day.

**Follow-ups** (each with its own doc): r49 discovered q/k/v and gate/up share one A, cutting prepass launches from per-GEMM to a deduplicated 193 → 110 (118.4 → 83.9 ms); r51/r52 folded quantization directly into the rms/swiglu producers (the prepass eventually 10.1 ms, with mode 2 not even writing the f32 output); r54/60 pushed the whole gate set to default-on.

## 6. Lessons

1. **Paying a layout transform once vs paying it restage-count times is a multiplication-level difference**: transform cost = per-instance cost × the grid.y column count; wider-od projections amplify harder (gate/up is 148×, not 28×). r6's spec principle "transform out of the hot loop" — this is its complete application on the A side.
2. **Split copy and transform into separate ledgers**: byte copies can be absorbed by L1/L2 and cost zero ALU; index math is pure ALU and occupies registers — once the transform leaves the kernel, the remaining copy degenerates into a bulk copy and the register pressure vanishes with it (109→103).
3. **The consumption layout is the contract; the supply location is a degree of freedom**: the smem layout ldmatrix demands cannot change by one byte (r22's swizzle stands), but "who arranges it, at what frequency" is renegotiable — correctness is backstopped by the byte-exact validator.
4. **ncu being unavailable does not block a landing**: the register audit, the prepass timing, and the wall clock are three independent pieces of evidence sufficient to corroborate the mechanism — a profiler is one source of evidence, not a precondition for landing.
5. **Leave a SASS-identical control behind for the replaced thing**: the old kernel surviving intact = the A/B integrity gate gets its anchor for free; "new kernel + old kernel frozen" is the cheapest experimental design.

← 36-r33-hybrid-inner-loop · [Index](./README.md) · 38-r35-scale-predecode →
