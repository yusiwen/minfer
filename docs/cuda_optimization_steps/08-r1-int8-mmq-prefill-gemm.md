# 08 · R1 — int8 MMQ prefill GEMM (opt-in): the parity-first strategy (LANDED)

> **Result**: a self-built int8 tensor-core MMQ kernel (64×64×256 tile) behind `MINFER_MMQ=1`, parity fully green (8 types × 8 shapes, max diff < 1e-3; greedy 7B token-identical to the f16 path); performance 155 (co-tenant) / 412 (quiet) / 441 (r7–r8 window re-measure) tok/s vs the f16 path's 630–880 / 1460 — parity clean but ~8× slow (vs llama ~24 TMAC/s), a gap unattributable in that window (ncu refused by the device). It is the scaffold of the r7+ raw-line campaign: the 441 → 3590.8 = **8.1×** campaign arc starts here.
> **Commit**: `40e97c9`. **Date**: 2026-08-31.

## 1. Background — where things stood

After R3 located the small models' fixed overhead, prefill's main contradiction returned to the
GEMM itself. At that point the engine's prefill weight path was 8p's **resident f16 cache**: all
quantized weights are dequantized to f16 once at load and laid flat in VRAM (the ≥2 GB gate),
and the GEMM reads the f16 panels on wmma. This path was working on 7B — by P5's end, @2K
2340–2370 tok/s, 1.43× vs llama-bench 3401 — but it carried two structural costs:

1. **The weight traffic starts out multiplied**. An f16 panel costs 64 B per 32 elements; raw
   q4_0 blocks are 18 B, q8_0 34 B, K-quants 18–34 B per 32-k block — resident f16 makes
   weight DRAM traffic 2–4× the raw bytes, and the panel is the operand the GEMM re-reads
   over and over.
2. **Dequant is a one-time tax, but memory is a permanent tax**. The 2–4 GB of extra resident
   plane buys only "reads fast", while for llama.cpp this plane does not exist at all — its
   MMQ (matrix-multiplication with quantized weights) feeds int8 straight to the tensor
   cores: weights stay raw nibbles, activations are quantized to q8, int8×int8 accumulates
   exactly in int32, and block scales are corrected per block outside the mma.

That llama.cpp's prefill is fast — this is a core link of it. For minfer to close the 1.43× gap
there is no way around standing this pipeline up. But at the time llama.cpp's MMQ internals had
not yet been dissected the way they are today (that dissection only later became
`docs/LLAMA-CPP-MMQ-ANALYSIS.md`), so R1's shape was: **implement llama's MMQ math in a
self-built kernel on minfer's own 8p tile skeleton** — not a line-by-line port. This step's
strategic significance outweighs its performance significance: nail down the **correctness** of
"int8 pipeline + per-block rescale" behind an opt-in switch first, so the later campaign (the
raw-byte line from r7 on) can chase speed on a parity-green foundation.

Where things would stick without this step: the f16 path's weight-traffic floor locks the GEMM's
byte budget; without a parity gate, any int8 attempt faces two unknowns at once — "is it right"
and "is it fast" — and events soon proved the first version was destined to be slow.

## 2. Principle — the GPU mechanism

### 2.1 The int8 tensor-core mma (s8s8s32)

R1's workhorse instruction is `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`: one
warp-collective call completes an **M=16 × N=8 × K=32** matrix-multiply fragment, A/B operands
int8, accumulator int32. Three key properties:

- **Exactness**. Integer multiply-add has no rounding — Σ w·q within one 32-k block is exact and
  associative in int32 (nibble grid |w|≤127 × activations |q|≤127; the in-block magnitude is
  far within range). This means **all floating-point work reduces to the block-scale
  correction multiplies**, and the parity tolerance only needs to cover the ordering
  differences of those few f32 multiply-adds, not the dot product itself. That is the
  mathematical basis for daring to set the later parity gate at 1e-3.
- **Warp collectivity: there is profit only at M≥16**. `mma.m16n8k32`'s fragment layout is
  fixed: A's 16×32 is spread across the 32 lanes with row = groupID (lane/4) and k = the quad;
  C is 4 int32 registers per lane. **M=16 is the instruction's hard size** — run a
  decode-shaped nt=1 problem on it and 15/16 of the rows are padding, with the per-instruction
  cost undiminished. This exactly explains the engine's two-tier dispatch: decode takes MMVQ
  (dp4a per-thread dot products, the scheme 8e landed), and only prefill (large nt) takes
  MMQ's mma pipeline. (The D series later quantified this as "MMQ-at-M=1 = 0.14-wave
  collapse", with nt=4 the crossover where BT-MMQ pays — but the geometric reason was already
  written into the kernel comments at R1 time.)
- **The throughput gear**. On Ampere-class and later SMs, the int8 tensor pipe's per-cycle MAC
  count is on the order of 4× f16's; stacked on "weights stay raw bytes", MMQ earns both the
  compute gear and the traffic at once.

R1's tile family is **64×64×256**: block tile 64 tokens × 64 od rows (`MMQ_BI=64`, `MMQ_BJ=64`),
staging 8 32-k chunks at once along k (`MMQ_KD=8`, i.e. 256-k depth, llama.cpp ITER_K style);
warp tile 32 tokens × 16 rows, 8 warps (`wm = warp>>1` covering 4×16 rows, `wn = warp&1`
covering 2×32 tokens) exactly tiling 64×64. The shared-memory ledger:

```
qa:  2 × KD × BI×WS int  = 2×8×64×9 ×4B = 36,864 B   ← A fragments (WS=9: 8 data + 1 pad)
qb:  same                  = 36,864 B                ← B fragments
ssa: 2 × KD × BI int      =  4,096 B                ← activation block int sums (for the rank-1 term)
sda/sds/sds1/sdm: 4 × 2×KD×BJ f32 = 16,384 B        ← block scales (q6_K dual sub-scales)
─────────────────────────────────────────────────
Total ≈ 94,208 B ≈ 94 KB → 1 block/SM on sm_121 (GB10)
```

### 2.2 The q8_1-style activation pipeline

llama.cpp's MMQ quantizes activations into **q8_1**: each 32-element block stores 32 int8s + f16
scale d + one int32 **in-block integer sum s**. Why a sum beyond the dot product? Because
min-carrying weight types (q4_1/q5_1/K-quant) have values `w_i = ds·nib_i − dmin`, so:

```
Σ w_i·q_i = ds·Σ nib_i·q_i − dmin·Σ q_i
            └── acc computed by int mma ──┘   └── dmin × activation block sum (rank-1 term)
```

The activation block sum the rank-1 term needs must be produced in the **same pass** as
quantization (summing in passing during quantization is the cheapest). R1's q8 pipeline is an
isomorphic implementation of llama's q8_1: activations are quantized once per launch into
**pad40** blocks (40 B per 32-element block), and the 4 slack bytes written at offset 36 are
exactly the block's int sum — during staging that one word travels into shared memory together
with the quants (the kernel's `ssa[]`), zero extra reads outside the mma. This was also the
design prototype later contrasted with llama's `quantize_mmq_q8_1` in the r34 quantize-transpose
prepass; llama's own step of folding this quantization into the GEMM prologue (the q8_1
pipeline) remained on record as a "step-function next" all the way to the campaign's close.

K-quant nibbles stay **UNSIGNED** (llama's unpack_scales trick): the grid is 0..15 or 0..31, the
integer part is non-negative, and the scale/offset pair `(d·s, −dmin)` is applied per 32-k
sub-block; q6_K's scales live on 16-element sub-blocks and one 32-k chunk spans two → each chunk
runs **two m16n8k16s** with separate int accumulators, multiplied by `(d·sc0, d·sc1)`
respectively.

### 2.3 Why this shape was "destined to be slow first but worth making correct first"

The bandwidth ledger: A side 40 B per chunk per token (pad40 q8), B side 16–34 B per chunk per
row of raw weight bytes, against the f16 cache path's 64 B/row — **weight traffic cut 2–4×
outright**. But R1's staging is synchronous (`cp.async` cannot carry quantized bytes —
quantization must happen before the move), so each chunk's load latency is exposed directly on
the critical path — the kernel comments recorded the numbers: staging one 32-k chunk at a time
gives 2.5 TMAC/s, and batching 8 chunks amortizes it 8×. This "staging depth vs exposed latency"
contradiction is the target the whole later raw line (r12/r14/r20/r34) would shoot at; R1 stood
the target up rather than breaking through it on the spot.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Self-built tile, ported math**. Not a line-by-line transplant of llama.cpp: the tile
  skeleton reuses 8p's wmma GEMM structure (8 warps, consecutive blocks sharing one od-tile
  weight panel's L2 locality), and the mma fragment layout strictly follows the PTX ISA
  documentation (C's `get_i/get_j` mapping uses the set llama.cpp has production-verified).
  The reasoning: porting the math contract (per-block rescale, unsigned nibbles, q6_K dual
  accumulators) gets parity; porting the code would drag llama's launch/stream-k structure in
  with it, and that part was not yet dissected at the time.
- **Weights stay RAW**. No dequant, no w16 cache — the B-side staging unpacks nibbles straight
  into shared memory. This is the fundamental fork from the 8p route, and the source of the
  traffic advantage.
- **Types as templates**. Three axes, `<int TYPE, int KSPLIT, bool HAS_OFF>`: TYPE 0–7 covers
  all 8 supported quant types; KSPLIT=1 (one m16n8k32 per chunk, types 0–6) vs KSPLIT=2 (two
  m16n8k16s + separate accumulators, q6_K only); HAS_OFF = min-carrying types take the rank-1
  term.
- **KD=8 vs KD=4**: ~94 KB deep staging (1 block/SM) measured faster than KD=4 (2 blocks/SM) —
  under co-tenant load, depth beat occupancy (a conclusion r5–r6 would flip once more; see
  later).
- **Landed as opt-in**. Enabled only with `MINFER_MMQ=1`; the default stays f16 — in this state
  MMQ is ~3.5× slower, and parity-first does not mean a default switch. `mma.m16n8k32` does
  not exist on sm_75, so compile-time `__CUDA_ARCH__ >= 800` falls straight back to the f16
  path.
- **q6_K does not use a per-16 loop**. The naive cut of mma per 16-element sub-block is 4×
  slower end to end — q6_K carries 7B's ffn_down + lm_head, far too much of the wall. k32
  staging + dual m16n8k16 preserves full throughput.

### 3.2 Key code

**The design contract** (`src/cuda_kernels.cu`, commit `40e97c9`, excerpt):

```cuda
// ─── R1: int8 MMQ prefill GEMM ────────────────────────────────────────
// llama.cpp's MMQ math structure on minfer's 8p tile skeleton. Activations
// are quantized to q8_0 once per launch (pad40 blocks; the kernel writes the
// per-block int sum into the 4 slack bytes at offset 36), weights stay RAW —
// no f16 dequant pass, no w16 cache. A tiled mma.m16n8k32 (s8) GEMM
// accumulates one 32-k int chunk per step; the int C fragment is rescaled
// per (token, row, k-block) with the block-scale products and, for
// min-carrying types, a rank-1 offset term (weight min × activation block
// sum) — exactly llama.cpp's per-block correction, so results sit within
// f32 rounding of the CPU q8_0-activation dot path.
//
// K-quants keep their nibbles UNSIGNED in the int GEMM and carry the min
// term separately (llama.cpp's unpack_scales trick — the nibble grid is
// 0..15/0..31, so the "integer part" is non-negative and the scale/offset
// pair (d·s, −dmin·m) is applied per 32-k sub-block). q6_K's scales live on
// 16-element sub-blocks: each 32-k chunk spans two of them, so the chunk
// runs as TWO m16n8k16 mmas (low/high k halves) with separate int
// accumulators, rescaled by (d·sc0, d·sc1).
```

**The int8 mma inlines** — note the accumulator C folded in place (`"+r"(d[i])`), 4 int32s per
lane:

```cuda
__device__ __forceinline__ void mmq_mma_k32(int* d, const int* a, const int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mmq_mma_k16(int* d, const int* a, int b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}
```

**The main loop skeleton** (`mmq_nt_kernel`, excerpt) — smem layout, double buffering, fragment
assembly, rescale:

```cuda
template <int TYPE, int KSPLIT, bool HAS_OFF>
__global__ void __launch_bounds__(256) mmq_nt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id, int bstride
) {
#if __CUDA_ARCH__ >= 800   // mma.m16n8k32 (s8) needs sm_80+; sm_75 keeps f16
    extern __shared__ int mmq_sh[];
    int* qa  = mmq_sh;                               // [2][KD][BI*WS]
    int* qb  = qa + 2 * MMQ_KD * (MMQ_BI * MMQ_WS);  // [2][KD][BJ*WS]
    int* ssa = qb + 2 * MMQ_KD * (MMQ_BJ * MMQ_WS);  // [2][KD][BI] activation block sums
    float* sda  = reinterpret_cast<float*>(ssa + 2 * MMQ_KD * MMQ_BI);
    float* sds  = sda + 2 * MMQ_KD * MMQ_BI;         // weight scales [2][KD][BJ]
    float* sds1 = sds  + 2 * MMQ_KD * MMQ_BI;        // q6_K's second 16-sub
    float* sdm  = sds1 + 2 * MMQ_KD * MMQ_BI;        // min (the rank-1 term)

    float sum[16] = {0.0f};   // [nh][h][l]: 2 B-frags × 2 A-frags × 4 C regs
    // ... double buffer: stage block 0 → inside the loop, stage kt+1 before computing kt ...
    for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
        if (kt + 1 < nktile) MMQ_STAGE_TILE(kt + 1, buf ^ 1)
        for (int kd = 0; kd < MMQ_KD; kd++) {
            // A fragments: 2×16-token halves; B fragments: 2×8-row halves; 4 mmas per chunk
            int a[2][4], b[2][2];
            int clow[2][2][4], chigh[2][2][4];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r0 = (i0w + h * 16 + (lane >> 2)) * MMQ_WS + (lane & 3);
                const int r1 = (i0w + h * 16 + 8 + (lane >> 2)) * MMQ_WS + (lane & 3);
                a[h][0] = qat[r0]; a[h][1] = qat[r1];
                a[h][2] = qat[r0 + 4]; a[h][3] = qat[r1 + 4];
            }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    if constexpr (KSPLIT == 1) {
                        mmq_mma_k32(clow[nh][h], a[h], b[nh]);
                    } else {                          // q6_K: low/high k halves split into two
                        mmq_mma_k16(clow[nh][h], a[h], b[nh][0]);
                        mmq_mma_k16(chigh[nh][h], a[h] + 2, b[nh][1]);
                    }
                }
            // per (token, row, k-block) rescale: value = ds·int (+ dm·sa), all × da
            const float da_q[4] = { sdat[i0w + lane / 4], sdat[i0w + 8 + lane / 4],
                                    sdat[i0w + 16 + lane / 4], sdat[i0w + 24 + lane / 4] };
            // ... after the dsv/dmv reads: sum[idx] += da·(ds·acc + dm·sa) (when HAS_OFF)
```

**q6_K's unsigned-nibble B fragment assembly** — the 6-bit grid 0..63 subtracts 32 per byte into
the signed domain; `__vsubss4` does the 4-byte SIMD subtract in one instruction:

```cuda
uint32_t nib = (g < 2) ? (QL & 0x0F0F0F0Fu) : ((QL >> 4) & 0x0F0F0F0Fu);
uint32_t hi  = ((QH >> (2 * g)) & 0x03030303u) << 4;
qb[r * MMQ_WS + w] = __vsubss4((int)(nib | hi), 0x20202020);  // −32/byte
```

**The parity test's reference frame** (current tree
`src/graph/cuda_backend.rs::cuda_prefill_mmq_parity` — this gate lives on today, still the hard
gate for every MMQ change). The test comment's own words give the tolerance rationale:

```rust
    // dot math (the structure llama.cpp's MMQ implements): int8×int8 dots
    // are exact on both sides and the block scales are f16→f32 on both
    // sides; only accumulation order differs, so 1e-3 absolute leaves
    // orders of magnitude of headroom over f32 rounding while still failing
    // loudly on any fragment-layout or unpacking mistake. All 8 types ×
    // {odd tile edges, 2 super-blocks}; q6_K in both registered layouts.
```

The reference implementation is the CPU-side per-32-block q8_0-activation dot product:

```rust
        // reference: CPU q8_0-activation dot math, per 32-block:
        //   out += da · (ds · Σ w_i·q_i + dm · Σ q_i)
        // q6_K carries 16-element sub-scales → two halves per 32-block.
        for b in 0..nb {
            let blk = &x[t * id + b * 32..t * id + b * 32 + 32];
            let am = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = am / 127.0;
            da[b] = half::f16::from_f32(d).to_f32(); // f16 rounding, as the GPU kernel stores it
            let di = if d != 0.0 { 1.0 / d } else { 0.0 };
            for (i, v) in blk.iter().enumerate() {
                let qi = (*v * di).round_ties_even();
                qi = qi.clamp(-128.0, 127.0);
                q[b * 32 + i] = qi as i32;
                sa[b] += qi as i64;                  // block sum: the rank-1 term's reference
            }
        }
```

### 3.3 Pitfalls

- **Synchronous staging's exposed latency**. Quantized bytes cannot ride `cp.async` (it moves
  but does not unpack), so the first version synchronized once per chunk, exposing global-load
  latency in full on the mma critical path — 2.5 TMAC/s. Batching 8 chunks (KD=8) amortized it
  8× to reach usable. This pitfall defined the shape of the entire campaign that followed.
- **The q6_K per-16 loop is a 4× trap**. Cutting the chunk into k16 along "the scales live on
  16-element sub-blocks" doubles the mma count and halves the staging rhythm; end to end it is
  4× slower. The correct shape is k32 staging + one m16n8k16 each for the low/high k halves,
  separate accumulators, each paired with its own (d·sc0, d·sc1).
- **The K-quant signed domain**. Nibbles stay unsigned and min goes through the rank-1 term,
  rather than pre-converting nibbles to signed before the mma — the former keeps `(d·s,
  −dmin)` in the rescale to be applied per sub-block, and q6_K needs only one `__vsubss4` SIMD
  subtract to move the 6-bit grid into the signed domain. This "B-frag contract" was later
  reused verbatim in r28's NB kernel.
- **The profiling channel was sealed**. On GB10 in that window ncu reported `ERR_NVGPUCTRPERM`
  (device counter permission), so the ~8× performance gap had **no counter evidence at all** —
  only TMAC/s estimates. r13's counter forensics had to wait for the channel to be repaired;
  this is also one of the premises that made the "parity first, speed later" strategy sound:
  while the gap is unattributed, the only certainly-correct asset is parity itself.

## 4. Verification

- **`cuda_prefill_mmq_parity`**: an 8-type × 8-shape sweep (odd tile edges, 2 super-blocks, q6_K
  in both registered layouts all covered), max diff < 1e-3. Defends against fragment-layout
  errors (misaligned lane→(row,col) mapping) and nibble-unpacking errors — the integer dots
  are exact on both sides, so 1e-3's entire margin belongs to the f32 rescale's ordering
  differences, and any "real error" fails loudly instead of hiding in noise.
- **Greedy 7B ≡ the f16 path, token for token**: end-to-end comparison against the in-service
  f16 path. Defends against regressions in the integration surfaces beyond the kernel —
  dispatch, epilogue, type registration.
- **Suite all green**: the existing CPU/Metal/f16-CUDA paths unbroken.
- **Interleaved A/B measurement**: same binary, `MINFER_MMQ=1/0` alternated — 155 vs 630–880
  under co-tenant, 412 vs 1460 on a quiet machine; the r7–r8 window re-measured **the same
  code** at 441. Defends against "mistaking the co-tenant tax for a code tax" — this gate
  later matured into r59b's "baseline behavior anchoring" rule.

## 5. Results

Kernel level and wall-clock level (7B q4_k_m, DGX Spark GB10):

- **Performance**: the MMQ path at 155 tok/s (sglang ~96% co-tenant) / 412 (quiet) / 441 (r7–r8
  window re-measure); the f16 path in the same windows 630–880 / 1460. I.e. the opt-in state
  is ~3.5× slower, ~2.9 TMAC/s vs llama ~24 on the quiet baseline — **a ~8× gap,
  unattributable in that window** (ncu `ERR_NVGPUCTRPERM`). Fixed overhead ~0.6 ms per prefill
  (launch + quantization).
- **Shape conclusions**: KD=8 (1 block/SM deep staging) beat KD=4 (2 blocks/SM) under load; the
  per-chunk latency exposure of synchronous staging was the largest single item (2.5 TMAC/s →
  KD=8 amortizes 8×).
- **The landing**: `MINFER_MMQ=1` opt-in, the default stays f16 (a 3.5× regression cannot be the
  default). The master table's status reads **LANDED (opt-in; superseded by raw line)** — from
  r7 on, the raw-byte line rebuilt the kernel on this scaffold (raw-byte smem, ldmatrix, 2
  blocks/SM, the quantize-transpose prepass … until r60 flipped it default-on).
- **What it bought**: the parity gate has been the hard gate for every MMQ change since R1 —
  none of the dozens of levers after r7 needed to reinvent a correctness standard; the
  campaign arc of master-table footnote 4, **441 → 3590.8 = 8.1×**, starts at this row's 441.
  The slow R1 is the only step in the entire campaign where "performance did not matter",
  because what it delivered was not speed but **a foundation on which speed could be chased
  with confidence**.

## 6. Lessons

1. **A parity-clean opt-in lands even when slow**: with the correctness scaffold in place first,
   the speed campaign has a stable right/wrong standard — the raw line's 8.1× walked on this
   gate the whole way.
2. **Re-measure the same code in a quiet window before concluding**: 155 (co-tenant) → 412
   (quiet) → 441 (re-measure); judge the environment before judging the code.
3. **A kernel you cannot profile can only be estimated**: with the counter channel sealed,
   TMAC/s is the only clue; the counter forensics after the channel was repaired (r13)
   changed the direction of the entire campaign.
4. **Port the math contract, not the code**: implementing llama's per-block rescale +
   unsigned-nibble contract on one's own tile skeleton gets parity; a line-by-line port drags
   the un-dissected structures in with it.

---
← [07 · R3 small-model overhead](./07-r3-small-model-overhead.md) · [Index](./README.md) · [09 →](./09-r2-mmvq-weight-streaming.md)
