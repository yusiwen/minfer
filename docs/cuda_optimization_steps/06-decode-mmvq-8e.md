# 06 · 8e/8e② — decode MMVQ: dp4a integer dot products + the llama.cpp launch table (LANDED)

> **Result**: 7B decode **+37%** (q4_K); kernel-level +74–77% per matmul (194–207 vs 112–117 GB/s, L2-defeated microbenchmark); q6_K/q5_K followed + an od·id ≥ 24M shape gate.
> **Commit**: `b7b8e73` (8e: the q4_K MMVQ reversal), `1298cb2` (8e follow-up: q6_K/q5_K kernels), `1d28235` (the shape gate into graph dispatch). **Date**: 2026-08-30.

## 1. Background — where things stood

8o (chapter 04) removed the one-time decode-start stall, but decode's steady state still ran on
the f32-activation row-wise kernel. This path has an earlier dark history: in the P4 draft era
before Phase 7 (`docs/CUDA_OPTIMIZATION.md` Part IV), a "tiled quantized matmul" had already
been tried once, the verdict then being **negative return**, with a seemingly authoritative
explanation — "116 GB/s = the platform's streaming bandwidth limit" — so the direction was
buried, and llama.cpp's corresponding design was misread as "shared-memory tiling with Stream-K
decomposition".

8e began with doubt about that buried verdict, and the evidence for the doubt took only one
control experiment: **an empty kernel that computes nothing and only reads weights reaches 252.7
GB/s on GB10** — 93% of the theoretical 273 GB/s. So 116 GB/s could not possibly be a platform
limit; it had to be the kernel's own geometry.

The disease was immediately clear. The old f32-activation kernel used a 2 warps × 4 rows layout,
each lane serially processing an entire 144-byte block — on the 7B ffn_down shape that is only
~28K threads in flight across the whole grid. GB10's LPDDR latency hides behind concurrency: 28K
threads cannot spread enough outstanding loads, and the kernel actually ran at ~46% of platform
bandwidth.

llama.cpp's decode dot-product path `mul_mat_vec_q` (MMVQ) is a different geometry, and a
**parameterized** one: its `MMVQ_PARAMETERS_*` gives a launch-parameter table per GPU generation
— the GB10 entry is 8 warps, one output row per block. Porting that geometry as-is was 8e's
reversal: within a single day, the old "negative return + platform limit" verdict was overturned
and 7B decode gained +37%. Row 6's lesson follows from it: **port the launch-table parameters,
not just the math** — half of the original mistake lived outside the math.

## 2. Principle — the GPU mechanism

**dp4a** (DOT4-accumulate, an integer instruction since sm_61+): `__dp4a(a, b, c)` completes 4
pairs of int8 products and accumulates into a 32-bit integer in one instruction — `a0·b0 + a1·b1
+ a2·b2 + a3·b3 + c`. For quantized dot products it is a natural building block: one uint32
register holds 4 bytes (4 int8 activations, or 4 unpacked 4-bit nibble components), and one dp4a
does 4 multiply-adds. The integer pipeline's throughput far exceeds an f32 chain, and it keeps
weight traffic at the quantized byte width (q4_K 0.5 B/weight) — the price is that activations
must become integers too.

**The q8_0 activation quantization path** (decode side): at nt==1 the activation row x (f32, id
elements) is quantized into q8_0 in 32-element blocks before entering the matmul: per block `d =
max|x|/127`, `q_e = rintf(x_e/d)` clamped to [-128, 127]. This matches the CPU path's
established convention (§Core Conventions: all CPU matmuls consume Q8_0 activations) — CUDA
decode moving from f32 activations to q8 activations is essentially pulling minfer back into
llama.cpp's isomorphic design. minfer uses the **pad40 layout**: each block is 40 bytes = `[f16
d][2B pad][32B int8 payload]` (plus a 4 B sum of quantized values at the block tail, used by the
prefill MMQ's min-term correction; the MMVQ kernel reads only d and the payload and never sees
that word). The pad's job is to land the int8 payload on 4-byte alignment — dp4a's uint32 reads
require it.

The min term also has a non-obvious trick. A Q4_K dequantized value = `d·s·nib − dm·m` (s/m are
the sub-block scale/min, d/dm the super-block's), so:

```
W·x = Σ_sub-blocks [ d·s·Σ_e(nib_e·x_e) − dm·m·Σ_e(x_e) ]
```

The second term needs not the weight sum but the **activation sum** Σx. In the kernel it is
accumulated per sub-block with one `__dp4a(0x01010101, xa, sx)` — four ones times four
activation bytes is an incremental update of Σx, done in one line, with no separate bsums pass.

**The MMVQ geometry** (what the launch table contains): llama.cpp's `mul_mat_vec_q` for GB10 is
**8 warps (256 threads) per block, one block computing one output row**, with that row's (block,
32-element sub-block) units round-robined across the 256 threads and reduced by a block-wide
reduction at the end. The concurrency arithmetic: 7B ffn_down (od 3584, one row per block) =
3584 blocks × 256 threads ≈ **917K threads in flight**, versus the old kernel's ~28K — 32× the
concurrency, and LPDDR latency is finally covered by outstanding loads. This is exactly what
"launch-table parameters are part of the design" means: the same math spread as 8 warps ×
one-row-per-block versus 2 warps × four-rows-per-block — the former +74–77%, the latter stuck at
46% of platform bandwidth.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Copy llama.cpp's structure; invent nothing.** 8e's first commit title says "re-examined" —
  what was reversed was our own old verdict, and the method was porting `mul_mat_vec_q`'s
  block geometry, unit mapping, and reduce structure as-is, aligning only the K-quant
  block-format details (q4_K's scale/min layout, q6_K's 16-element units) to GGUF semantics.
- **One kernel per type**: one each for q4_K / q6_K / q5_K (`1298cb2`), sharing the block shape
  and reduce while each owning its own nibble/byte unpacking code — the K-quant super-block
  formats differ enough that forcing an abstraction would sacrifice readability.
- **The shape gate lives in dispatch** (`1d28235`): MMVQ does not win everywhere. Measured: at
  od·id < ~24M elements it actually loses (od 512 → 4.5× slower, 896 → 3.0×, 2048×4864 →
  1.66×) — with too few rows each thread gets only 1–2 units and q5/q6's uncoalesced byte
  loads are exposed; large shapes win (7B ffn_down 3584×18944 → 1.5× faster, lm_head
  152064×3584 → 1.4×). The q4_K gate uses `id ≥ 2048 && id % 32 == 0` (below id 2048 the
  margin shrinks to launch-latency noise), the q5_K/q6_K gate uses `od·id ≥ 24M`. Tensors
  outside the gate keep the padded-f32 kernel — **small tensors stay on the safe path**, the
  "dispatch gated by shape" recorded in row 6.
- **The q8 scratch's capture constraint**: `buf_q8_decode` is sized by id, constant within a
  graph, grown during warmup and never inside a capture window — CUDA Graph replay must
  contain no allocation events; this is 7d's existing capture/replay discipline applied in
  this step.

### 3.2 Key code

The q4_K MMVQ kernel (`src/cuda_kernels.cu`, `q4_k_q8_mmvq`; some unpacking details elided in
the middle):

```c
__global__ void __launch_bounds__(256) q4_k_q8_mmvq(   // 8 warps = launch table
    const uint8_t* __restrict__ weights, const uint8_t* __restrict__ acts8,
    float* __restrict__ output, int od, int id, int nt
) {
    const int row = blockIdx.x;                        // one block = one row
    const int nbe = (id + 255) / 256;
    const int row_stride = nbe * Q4KB;
    const int nsub = (id + 31) / 32;
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nsub; u += 256) {    // unit round-robin
        const int blk_i = u >> 3, sub = u & 7;
        const uint8_t* blk = weights + (size_t)row * row_stride + blk_i * Q4KB;
        const float d  = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        uint8_t s8, m8;
        get_scale_min_k4(sub, blk + 4, &s8, &m8);
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 16 + (sub >> 1) * 32);
        const bool lo = (sub & 1) == 0;                // even subs take the low nibble, odd the high
        const uint8_t* x8 = x8row + (size_t)u * Q8PB;  // pad40 q8 activation block
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
        int dot = 0, sx = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {                  // 32 bytes = 8×uint32
            const uint32_t w = qw[v];
            const int n = lo ? (int)(w & 0x0F0F0F0F) : (int)((w >> 4) & 0x0F0F0F0F);
            const int xa = (int)xw[v];                 // 4 activation int8s
            dot = __dp4a(n, xa, dot);                  // 4 multiply-adds/instruction
            sx  = __dp4a(0x01010101, xa, sx);          // Σx, for the min term
        }
        acc += d8 * ((float)s8 * (float)d * (float)dot
                    - (float)m8 * (float)dm * (float)sx);
    }
    // …shfl_xor 8-warp tree reduction, thread 0 writes output[t*od + row]…
}
```

The dispatch gate (`src/cuda.rs`, the Q5_K branch; the comment carries the crossover data
measured at the time):

```rust
// 8e follow-up: decode (nt == 1) joins the MMVQ structure
// (dp4a over q8 activations, one row per 256-thread block).
// Shape gate measured on-device (dbg micro-bench, padded f32
// vs mmvq): od*id < ~24M elements loses (od 512 → 4.5x
// slower, 896 → 3.0x, 2048x4864 → 1.66x) because 1-2 units
// per thread expose the uncoalesced q5/q6 byte loads; large
// shapes win (7B ffn_down 3584x18944 → 1.5x faster, lm_head
// 152064x3584 → 1.4x). MINFER_NO_KQ_MMVQ=1 forces f32.
if nt == 1 && od * id >= 24_000_000 && !Self::no_kq_mmvq() {
    self.q5_k_decode_mmvq(wptr, x, out, od, id, nt);
    Ok(())
} else {
    launch!(launch_q5_k_f32_matmul)
}
```

The q8_0 activation quantization body (per-block quantization in the pad40 layout; D3-5 later
moved this body verbatim into the rms/swiglu producers for fusion — the quantization math is
exactly 8e's standalone kernel):

```c
__device__ __forceinline__ void quantize_pad40_block(
    const float* __restrict__ src, uint8_t* __restrict__ dst
) {
    float4 sv[8];                                       // 32 floats = 8×float4
    #pragma unroll
    for (int v = 0; v < 8; v++)
        sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
    float am = 0.0f;                                    // max|x| within the block
    for (int v = 0; v < 8; v++)
        am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                             fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
    float d = am / 127.0f;
    float di = (d != 0.0f) ? 1.0f / d : 0.0f;
    *reinterpret_cast<__half*>(dst) = __float2half(d);  // [f16 d]
    uint32_t packed[8];
    for (int v = 0; v < 8; v++) {                       // rintf + clamp → int8
        const float* e = &sv[v].x;
        uint32_t p = 0;
        for (int j = 0; j < 4; j++) {
            int q = int(rintf(e[j] * di));
            q = max(-128, min(127, q));
            p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);
        }
        packed[v] = p;
    }
    for (int v = 0; v < 8; v++)                         // payload lands at offset 4:
        *reinterpret_cast<uint32_t*>(dst + 4 + 4 * v) = packed[v];  // 4B aligned
    *reinterpret_cast<uint32_t*>(dst + 36) = /* Σq, for MMQ; not read by MMVQ */;
}
```

### 3.3 Pitfalls

- **"Platform limit" was the wrong attribution.** The original 8e's 116 GB/s was recorded as the
  platform's streaming limit, fossilizing a kernel-geometry problem into a physical constant.
  One data point — a read-only empty kernel at 252.7 GB/s (93% of theoretical bandwidth) —
  dismantled it. The lesson, mechanized: for any "XX GB/s = the limit" conclusion, run a
  ceiling probe first.
- **A test generator's block-layout bug impersonating a kernel bug** (`1298cb2`). When q6_K
  followed, tests failed for a while and the investigation suspected the kernel had wrong high
  bits ("wrong high bits"); the final finding: **the kernel draft was correct — it was the
  test's generator writing d at offset 0** — a Q6_K block is `ql[128] + qh[64] + sc[16]`, 208
  bytes, and only then the 2-byte d (block 210 B). The generator did not lay out data per the
  real layout, so the bit comparisons went all red, naturally.
- **The small-shape reversal**. MMVQ's round-robin assumes "enough units per thread to amortize
  overhead"; with few rows (od 512) each thread is left with 1–2 units and the latency of
  q5/q6's uncoalesced byte loads is exposed directly — 4.5× slower than padded-f32. The shape
  gate (24M / id≥2048) is not conservative decoration; it is the measured crossover.
- **Allocation inside a capture window**. If the q8 scratch grows during capture it breaks
  replay; handled by the "grow fully in warmup, constant within the graph" discipline (§3.1).

## 4. Verification

- **The L2-defeated microbenchmark (bench8e2)**: 194–207 GB/s vs 112–117 GB/s on the 7B shapes —
  L2 warmth deliberately flushed before measuring, defending against fake bandwidth from
  "reading L2, not DRAM".
- **Dispatch equivalence** (`1d28235`): in-gate shapes through graph dispatch are bit-identical
  to direct kernel invocation (bit-exact 0.0000) — defends against "the shapes entering
  through the gate actually running a different kernel/parameters than the directly-invoked
  shapes that were verified"; out-of-gate sub-gate shapes (attn_v, 0.5B ffn_down) verify the
  kernel itself by direct invocation.
- **The full CUDA suite**: 158 passed, 0 failed (as of `1d28235`).
- **`MINFER_NO_KQ_MMVQ=1`** forces the f32 path back — the standing valve for A/B and
  regression. (A semantics note: MMVQ vs the padded-f32 kernel is not a bitwise relationship —
  q8 activation quantization vs reading f32 directly are different numerical semantics
  classes; what it aligns with is the CPU path's Q8_0 activation convention. The later
  D-series campaign built the mature gate set of argmax/greedy/A/B for exactly this kind of
  "quantization semantics switch" — see the decode chapters after 06.)

## 5. Results

- **Row 6**: 7B decode **+37%** (q4_K); q6_K/q5_K followed and landed; the vs-llama column reads
  "—" (no whole-prefill comparison was recorded at the time; decode-vs-llama became an
  accounting item only in the D-series era).
- **Kernel level**: +74–77% per matmul (194–207 vs 112–117 GB/s, L2-defeated); the old
  f32-activation kernel's 46% of bandwidth → the MMVQ structure pushed the 7B shapes to ~200+
  GB/s.
- **The shape gate**: od·id ≥ 24M (q5_K/q6_K) / id ≥ 2048 (q4_K) — small tensors outside the
  gate keep padded-f32. D3-7 2b later lowered the q6_K gate to 4M for the attn_v class (a
  story for another chapter).
- **The long tail**: this dp4a MMVQ skeleton became the foundation of all later decode work —
  R2's weight-streaming rework (tg128 +6.9%), D3b-1b's pipelining, D4-4's dense split-plane,
  D3-5's fused-producer A quantization, and D3-8's FusedQKV concat equivalence argument all
  stack on this structure. After D4-4, 7B decode tg128 is 51.2 (1.074× vs llama) — the
  starting point was this step's +37%.

## 6. Lessons

1. **Port the launch-table parameters, not just the math** (row 6's own words): block geometry,
   warps, and unit mapping are half the design — with the same dp4a math, 2 warps ×
   four-rows-per-block stalls at 46% of bandwidth while 8 warps × one-row-per-block gains
   +74–77%.
2. Run a read-only ceiling probe before believing any "bandwidth = platform limit" conclusion —
   the 116 GB/s "limit" was dismantled in one stroke by a 252.7 GB/s empty kernel.
3. For a latency-bound kernel, fix concurrency first (28K → 917K threads in flight), then talk
   bytes — 32× concurrency is not tuned into existence; it is rearranged into existence by
   geometry.
4. When a K-quant test goes red, check the generator's block layout first (d at offset 208, not
   0) before suspecting the kernel — fixture bugs and kernel bugs are fixed in completely
   different ways.

---
← [05 · persistent f16 weight cache (8p)](./05-persistent-f16-cache-8p.md) · [Index](./README.md) · [07 →](./07-r3-small-model-overhead.md)
