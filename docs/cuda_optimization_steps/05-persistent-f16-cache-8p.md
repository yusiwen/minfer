# 05 · 8p — Persistent f16 weight cache + fused dequant-in-GEMM (LANDED)

> **Result**: 7B q4_k_m @2K prefill 1201 → ~1400–1500 tok/s (+17–24%); decode unchanged (40.8 vs 40.6 A/B). The price is 2 B/element (~8.6 GB on 7B).
> **Commit**: `2992f57` (docs record `b9e7a91`). **Date**: 2026-08-31.

## 1. Background — where things stood

8m/8m② (row 2) unified the prefill GEMM into a 64×64 wmma f16 tensor-core kernel, taking 7B @2K
from 30.7 → 294 → 1204 tok/s. But this path carries a hidden tax: wmma consumes an f16 B
operand, while the GGUF weights are quantized (q4_K 0.5 B/weight, q8_0 1 B/weight, …). So 8m's
two-pass GEMM re-dequantizes W into an f16 scratch **on every matmul call**, and the GEMM then
reads that scratch.

How expensive is "every call"? Measured at 7B q4_k_m @2K: **288 ms/call** — dequantizing the 4.4
GB of quantized weights (read 4.4 GB + write ~8.8 GB f16) plus scheduling for all 28 layers × 7
matrices. The 1204 tok/s prefill forward itself is no longer slow, but every forward first pays
this fixed 288 ms tax. It was also one of the main components of the then ~2.8× gap to llama.cpp
on prefill.

Dequantization has an obvious asymmetry: **weights are immutable after load** (reuse consistency
guarded by `weights_version`), yet the two-pass GEMM re-derives their values on every forward.
The only defense of "dequant per call" is saving VRAM — holding an f16 copy on 7B costs +2
B/element ≈ **+8.6 GB**. 8p was built around this time-vs-memory triangle:

1. **Persistent f16 cache (the default path)**: each weight is dequantized once at load, the f16
   copy stays resident in VRAM, and the GEMM reads it directly — the per-call 288 ms goes to
   zero.
2. **Fused dequant-in-GEMM (the alternate path, `MINFER_FUSED_B=1`)**: no cache is built; inside
   the GEMM, B tiles are dequantized in registers from the raw quantized bytes — zero extra
   VRAM, but slower at large nt (§2), kept as the fallback for memory-constrained settings.
3. **A size gate (`W16_ENABLE_BYTES` = 2 GB)**: the cache is enabled only when total quantized
   matmul weight bytes ≥ 2 GB (7B q4_k_m's 4.4 GB passes; the 0.5–1.5B test fixtures do not
   and keep their pre-8p footprint — the reason is the suite OOM pitfall in §3.3).

This combination still serves today: the "f16 w16-cache" under §1.1's `MINFER_MMQ=0` legacy f16
escape path (~2353 tok/s, ~20.5 GB — about 11 GB heavier than the MMQ default path) is exactly
the skeleton this step built.

## 2. Principle — the GPU mechanism

Let W be quantized weights (q4_K-class, 0.5 B/element), id×od elements, f16 copy 2 B/element.
The relative per-forward costs of the three shapes:

- **Dequant per call (pre-8p)**: the dequant pass reads 4.4 GB of quantized bytes + writes 8.8
  GB of f16 scratch, which the GEMM then reads — **the dequant pass alone is one full-weight
  DRAM sweep**, 288 ms/call. The scratch's write traffic (8.8 GB) is generated for nothing,
  and every GEMM B-panel re-read hits this scratch.
- **Persistent f16 cache (8p default)**: the dequant's read+write each happen **once** (at
  load); from then on every forward's B panel reads only the f16 copy. The full 288 ms per
  call is saved, in exchange for +8.6 GB resident. The GEMM's byte footprint is unchanged (the
  same f16 as before); what is saved is the dequant ALU plus the wholesale in/out of the
  intermediate scratch.
- **Fused dequant-in-GEMM (`MINFER_FUSED_B=1`)**: B tiles are staged into smem as quantized
  bytes (q4_K 0.5 B/element, a quarter of f16) and dequantized to f16 in registers before
  entering wmma. The cheapest in bytes and zero VRAM delta, but **the dequant ALU becomes part
  of the kernel's inner loop**. What matters is the re-read count: with 64×64 tiles, the same
  B panel is re-read by every m0 (a tile along nt) — at nt=2048 that is 32 passes, i.e. 32
  rounds of dequant ALU in the fused shape; the f16 cache shape's "re-read" is only a byte
  re-read (with L2 backing it), no ALU. This is why the commit measured fused as **slower than
  the cp.async f16 GEMM at large nt** ("every nt tile re-dequantizes the B panel"), while in
  the nt==1 decode context fused is the one with the byte advantage (decode later went exactly
  the quantized-byte-stream + MMVQ/MMQ route — see 06).

Why the cache key can be a **device pointer**: `register_weight` reuses the same device copy for
same-name same-size registrations and never frees on replace — so a registered weight's device
pointer is stable for the process lifetime and the keys of `w16_cache: HashMap<usize /*wptr*/,
(CudaPtr, usize)>` never dangle. Another cash-in of the "weights are immutable" invariant (04's
registration idempotence lock leans on the same invariant).

The arithmetic of the alignment problem (Q5_0's latent crash, the protagonist of §3.3): a Q5_0
block is 22 bytes (2 B f16 `d` + 4 B `qh` bit plane + 16 B nibble `qs`). Block g sits at `base +
22g`; the u32 load of `qh` at `blk+2` is 4-byte aligned iff `22g + 2 ≡ 0 (mod 4)`, i.e. g is
odd. **Every even block is misaligned** — CUDA's scalar loads require natural alignment,
violation is `cudaErrorMisalignedAddress` (error code 716) and the kernel fails outright. Q5_1's
block is 24 bytes and `24g + 4 ≡ 0 (mod 4)` always holds, so q5_1's u32 loads were always safe —
alignment is a function of "stride × g + offset", not of provenance.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **The cache is the default and fused is the escape hatch, not the reverse.** The target
  scenario (7B @2K prefill) is large nt, long-running inference — the dequant cost amortizes
  over countless forwards and one-time materialization wins outright; the VRAM price is
  acceptable on GB10's unified memory pool. The fused kernel's value is in memory-constrained
  settings, kept as the "zero VRAM delta" alternative.
- **The gate lives in the loader's warm pass** (`W16_ENABLE_BYTES = 2 << 30`), not in runtime
  adaptation: the total bytes of quantized matmul weights are known the moment the model
  loads, making the check O(1); and "should we occupy 8.6 GB more" is fundamentally a
  model-level decision that should not fluctuate with load.
- **Dequantization happens at load (`warm_w16`), not lazily at the first GEMM**: the load path
  already registers weight by weight, so warming in passing lets the very first forward run at
  full speed; the price is one extra (one-time) segment of load time.
- **Two env vars, one direction each**: `MINFER_NO_W16CACHE=1` turns the cache off and falls
  back to per-call scratch; `MINFER_FUSED_B=1` turns fused on. Both A/B and regression have
  valves.
- **The guard added from R1 on**: the int8 MMQ prefill GEMM (R1, later made default via r60)
  streams raw quantized bytes directly, so the f16 cache is dead weight while it is active —
  the loader's warm condition gains `!cuda.mmq_active()` (visible in the current tree), and
  MMQ models no longer pay 8.6 GB for nothing.

### 3.2 Key code

First site: the loader-side warm gate (`src/models/qwen2/loader.rs`; the Qwen3 loader carries
the same 50-line change):

```rust
let warm_bytes: usize = warm
    .iter()
    .filter_map(|(t, _)| t.as_ref())
    .map(|t| t.data.len())              // total bytes of quantized matmul weights
    .sum();
// R1: with the int8 MMQ prefill GEMM active the f16 cache would be
// dead weight (MMQ streams raw quantized bytes) — skip the warm pass.
let warmable = warm_bytes >= crate::cuda::W16_ENABLE_BYTES && !cuda.mmq_active();
if warmable {
    cuda.enable_w16_cache();
}
for (t, name) in &warm {
    if warmable {
        if let Some(t) = t {
            cuda.warm_w16(name, t);     // dequantize once at load, f16 stays resident
        }
    }
}
```

The constant comment on `W16_ENABLE_BYTES` states the 7B/fixture boundary outright
(`src/cuda.rs`):

```rust
/// 8p: warm the f16 weight cache only for models whose quantized matmul
/// weights total at least this much (7B q4_k_m = 4.4 GB warms; the 0.5-1.5B
/// test fixtures do not, keeping their footprint at pre-8p levels).
pub const W16_ENABLE_BYTES: usize = 2 << 30;
```

Second site: the alignment fix forced out by the bitparity test — `dequant_q5_0_f16`
(`src/cuda_kernels.cu`, post-fix form):

```c
__global__ void dequant_q5_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 22;          // 22-byte block, stride not a multiple of 4
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    // 22-byte blocks are only 2-byte aligned: assemble qh from two u16
    // loads — a u32 load at blk+2 misaligns for even g
    // (cudaErrorMisalignedAddress 716; latent until 8p's bitparity test
    // exercised Q5_0 prefill GEMM for the first time).
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
    const uint8_t* qs = blk + 6;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1) - 16.0f;
        float hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1) - 16.0f;
        o[j] = __float2half(d * lo);
        o[j + 16] = __float2half(d * hi);
    }
}
```

Two u16 loads (addresses `22g+2` and `22g+4` are both even, so 2-byte alignment holds)
reassemble the original u32 — the value is identical and alignment is restored. The same fix
lands on the fused path's block-level assembly helper `bqa_q5_0` (`bqa_q5_1` was unified into
the two-u16 form while at it, even though the 24-byte block's u32 was safe already):

```c
__device__ __forceinline__ void bqa_q5_0(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 22)
                             + (long long)(e0 >> 5) * 22;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    // 22-byte blocks are only 2-byte aligned: assemble qh from two u16
    // loads (a plain u32 load at blk+2 misaligns for even block indices —
    // cudaErrorMisalignedAddress, caught by cuda_prefill_fused_b_bitparity).
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
    int b = e0 & 31;
    #pragma unroll
    for (int l = 0; l < 8; l++) {          // 8 elements from nibble + high bit plane
        int e = b + l;
        uint8_t byte = blk[6 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        float v = nib + 16.0f * float((qh >> e) & 1) - 16.0f;
        dst[l] = __float2half(d * v);
    }
}
```

Third site: the fused GEMM body's skeleton (`gemm_qb_nt_kernel`; 8m's 64×64 wmma structure
unchanged, the B panel's staging switched to "quantized bytes into smem + register
dequantization"):

```c
__global__ void gemm_qb_nt_kernel(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    float* __restrict__ C, int nt, int od, int id,
    int type_id, int q6_stride
) {
    __shared__ __half As[2][64 * 32];
    __shared__ __half Bs[2][64 * 32];        // dequantized f16 B tile (double buffer)
    ...
    int buf = 0;
    gemm_qb_load_tile(A, W, As[0], Bs[0], n0, m0, 0, nt, od, id, type_id, q6_stride);
    __syncthreads();
    for (int k = 0; k < id; k += 32, buf ^= 1) {
        if (k + 32 < id)
            gemm_qb_load_tile(A, W, As[buf ^ 1], Bs[buf ^ 1], n0, m0, k + 32,
                              nt, od, id, type_id, q6_stride);
        // same as 8m: fa[4] × fb[2] wmma pipeline, fc[2] accumulation
        wmma::load_matrix_sync(fb[0], &Bs[buf][wm * 16 * 32], 32);
        ...
        wmma::mma_sync(fc[0], fa[0], fb[0], fc[0]);
        wmma::mma_sync(fc[1], fa[1], fb[0], fc[1]);
        __syncthreads();
    }
```

`gemm_qb_load_tile` is the type-dispatched B assembly (one `bqa_*` branch per quant type, 8
total); its dequant math/rounding is bit-aligned with the standalone dequant kernel — the
precondition for fused passing the bitparity gate.

### 3.3 Pitfalls

- **The latent alignment crash (this chapter's main pitfall)**: `dequant_q5_0_f16`'s u32 load at
  `blk+2` is aligned only for odd blocks (§2's arithmetic). It "lived happily" because the
  CUDA prefill GEMM had never been run on Q5_0 weights before — any Q5_0 model on the prefill
  GEMM would crash deterministically (`cudaErrorMisalignedAddress` 716). What caught it was
  not a Q5_0 user but 8p's own newly written bitparity test (full enumeration of 8 types × 2
  super-block configurations). The fix is two-u16 load assembly, synchronized across three
  sites (the dequant kernel + `bqa_q5_0`/`bqa_q5_1`).
- **A performance feature pushed the test suite off a cliff (memory, not numerics)**: warming
  blindly for the small fixtures below 7B would cost each fixture +1–2 GB — and the suite
  keeps multiple loaded models co-resident in one oversubscribed CUDA pool, so
  later-registered models would **probabilistically OOM** while uploading weights. Hence the
  `W16_ENABLE_BYTES = 2 GB` gate: "a perf feature can break tests via footprint rather than
  via numerics". The gate keeps small models at their exact pre-8p footprint.
- **Bit-level equivalence means "the same rounding path", not "the same math"**: fused's
  dequantization must run in exactly the same order as the standalone dequant kernel — the
  same `__float2half` rounding, the same wmma accumulation — or the bitparity gate means
  nothing. The test even lays out each block's benign `d` (and min-type values) explicitly as
  f16 bytes, so random bytes cannot produce NaN/Inf that would disturb the bit comparison.
- **A pointer as the cache key = inheriting an ownership invariant**: `w16_cache` keying on
  device pointers presumes that `register_weight` reuses the same device copy for same-name
  same-size and does not free on replace. That presumption is written in the field comment —
  whoever changes the registration semantics in the future hits the comment first.

## 4. Verification

- **`cuda_prefill_fused_b_bitparity`** (new, `src/graph/cuda_backend.rs`): fused vs the legacy
  two-pass must be **bit-identical** across 8 quant types × 2 super-block configurations.
  Defends against: drift between fused's register dequantization and the standalone dequant
  kernel's rounding path (any "equivalent rewrite" of nibble order or scaling order gets
  caught). This is the test that snagged the Q5_0 alignment crash.
- **The legacy path's reference frame**: the two-pass path itself is verified by the existing
  `cuda_prefill_f16_gemm_parity` (against a reference implementation) — one end of the
  bitparity chain must be pinned first.
- **Decode A/B**: 40.8 vs 40.6 — defends against "prefill optimization hurting decode" (the
  cache changes the weight-resolution path; decode's kernel dispatch should be untouched).
- **Dual env valves**: `MINFER_NO_W16CACHE=1` (fall back to per-call scratch) and
  `MINFER_FUSED_B=1` (the fused alternate) make both alternative paths independently
  re-testable.

## 5. Results

- **Prefill (7B q4_k_m @2K)**: 1201 → ~1400–1495 tok/s (+17–24%; row 5 records ~1400–1500, with
  the Δ column "~4.7× vs 8m" — anchored on 8m's landing at 294 tok/s, 1400/294 ≈ 4.7×; before
  8m②'s cp.async staging the baseline was 294).
- **vs llama**: ~2.3× (the §0 reading convention: the early 8m–8p rows' multiples use the
  llama-bench 3401 @2K figure as denominator).
- **Component level**: the 288 ms per-forward dequant pass → 0 (paid once at load).
- **Decode**: unchanged (40.8 vs 40.6 A/B) — the cache serves only the prefill GEMM.
- **VRAM**: +2 B/element, ~8.6 GB on 7B q4_k_m; the fused alternate (`MINFER_FUSED_B=1`) adds
  nothing but is slower at large nt, positioned as the memory-constrained escape hatch.
- Later evolution: once R1's int8 MMQ path became the default, the f16 cache retreated to being
  the skeleton of the `MINFER_MMQ=0` escape path (§1.1: ~2353 tok/s, ~20.5 GB) — the
  infrastructure this step built outlived the entire MMQ campaign.

## 6. Lessons

1. One-time materialization of immutable weights beats per-call recomputation (row 5's own
   words: load-time materialization beats per-call dequant) — the test is always "will this
   value change", never "is computing it once expensive".
2. The bitparity test pays for itself: the full 8-type × 2-configuration enumeration caught, on
   merge day, a latent bug that would have crashed every Q5_0 prefill (row 5's own words: a
   bitparity test pays for itself immediately).
3. Alignment of strided access is the arithmetic of `stride × g + offset (mod width)` — a u32
   load on 22-byte blocks is aligned only for odd blocks; the fix is assembling from narrower
   naturally-aligned loads.
4. Performance features need "a size gate + an exit valve": footprint side effects reach the
   test suite as probabilistic OOM, which is harder to attribute than numeric errors.

---
← [04 · decode-start CPU stalls (8o)](./04-decode-start-stall-8o.md) · [Index](./README.md) · [06 · decode MMVQ (8e)](./06-decode-mmvq-8e.md)
