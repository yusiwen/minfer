# CPU Inference Path Bottleneck Analysis

This document details all performance bottlenecks in minfer's CPU inference
path, compares each with llama.cpp's approach, and provides prioritized
optimization recommendations.

## Current State

CPU inference on Qwen2-0.5B (AVX2, i7-1260P): ~21 tok/s decode.
llama.cpp on the same model: ~60-80 tok/s. The gap is ~3-4x.

---

## Bottleneck #1: Scalar Hotspots — Operations Without SIMD

**Impact:** ~30% of total CPU time across element-wise operations.
**Difficulty:** Low (existing AVX2 functions in `vec_ops.rs` are already written
but never called from the forward path).

### RMSNorm — Purely Scalar f64

**Location:** `src/models/qwen2/forward.rs`, lines 221-229

```rust
fn rms_norm(x: &[f32], eps: f32, out: &mut [f32], n: usize, d: usize,
            w: Option<&[f32]>) {
    for t in 0..n {
        let row = &x[t * d..(t + 1) * d];
        let dst = &mut out[t * d..(t + 1) * d];
        let mut ss = 0.0f64;
        for i in 0..d {
            ss += (row[i] as f64) * (row[i] as f64);  // scalar f64
        }
        let sc = 1.0 / ((ss / d as f64) as f32 + eps).sqrt();
        match w {
            Some(w) => {
                for i in 0..d { dst[i] = row[i] * sc * w[i]; }  // scalar
            }
            None => {
                for i in 0..d { dst[i] = row[i] * sc; }         // scalar
            }
        }
    }
}
```

**Problems:**
1. No SIMD — every operation is scalar f64 (2x slower per element than f32
   AVX2, which has 8-wide f32 lanes vs 4-wide f64).
2. Three separate passes: (1) sum-of-squares, (2) compute scale, (3) apply
   scale × weight.
3. Called **twice per layer** (attention norm + FFN norm) = 48 times for a
   24-layer model.
4. `vec_ops::rms_norm_f32` exists with partial SIMD but is **never called**
   from the forward path. It also has a bug (scales `y` before copying `x`
   into it).

**llama.cpp approach:** Does not use explicit SIMD for RMSNorm either — relies
on compiler auto-vectorization. But critically, llama.cpp **fuses RMSNorm with
the subsequent weight multiply** into a single pass (`rms_norm_mul_fused`),
avoiding one full memory read+write.

**Fix:** Replace the scalar loop with AVX2 operations. The sum-of-squares can
use `_mm256_loadu_ps` + `_mm256_mul_ps` + `_mm256_add_ps`. The output can use
fused multiply. Alternatively, pre-allocate a buffer and call `vec_ops`
functions directly.

### Residual Add — Scalar Loop

**Location:** `src/models/qwen2/forward.rs`, lines 102 and 117

```rust
for i in 0..hidden.len() { hidden[i] += bn[i]; }
```

**Problems:**
- Pure scalar loop over `nt * ne` elements (e.g., 4096 f32 values).
- `vec_ops::vec_add_f32` has a proper AVX2 path (lines 236-271) but is
  **never called** here.
- Runs twice per layer = 48 times total.

**Fix:** Replace with `vec_ops::vec_add_f32(nt * ne, hidden, hidden, bn)` or
equivalent AVX2 loop.

### Gate × Up Element-wise Multiply — Scalar

**Location:** `src/models/qwen2/forward.rs`, line 115

```rust
for i in 0..len { bg[i] *= bf[i]; }
```

**Problems:**
- `len = nt * nf` (e.g., 11008 elements). Pure scalar.
- No SIMD vectorization.
- Runs once per layer = 24 times.

**Fix:** Replace with AVX2 multiply loop: load 8 floats from each, multiply,
store.

---

## Bottleneck #2: RoPE — Redundant Trigonometric Computation

**Impact:** ~5% of total CPU time.
**Difficulty:** Low.

### Current Implementation

**Location:** `src/models/qwen2/forward.rs`, lines 235-245

```rust
fn apply_rope(x: &mut [f32], pos: &[usize], nh: usize, hd: usize,
              fb: f32, freq_scale: f32) {
    let half = hd / 2;
    let mut freqs = [0.0f32; 64];
    for i in 0..half {
        freqs[i] = freq_scale / fb.powf((2 * i) as f32 / hd as f32);
    }
    for t in 0..pos.len() {
        let p = pos[t] as f32;
        for h in 0..nh {                          // ← every head
            let b = t * nh * hd + h * hd;
            for i in 0..half {
                let th = p * freqs[i];
                let (sn, cs) = th.sin_cos();      // ← recomputed per head!
                let (i0, i1) = (b + i, b + i + half);
                let (x0, x1) = (x[i0], x[i1]);
                x[i0] = x0 * cs - x1 * sn;
                x[i1] = x0 * sn + x1 * cs;
            }
        }
    }
}
```

**Problem:** The cos/sin values depend only on `position × frequency_index`,
**not on the head**. But the code recomputes `sin_cos()` for every head:
- 32 heads × 64 frequency bins = 2048 `sin_cos` calls per token per layer
- Called twice per layer (Q and K) = 4096 calls per token
- `sin_cos()` costs ~50-100 cycles each

**llama.cpp approach:** `ggml_rope_cache_init` precomputes cos/sin values into
a per-position cache. All heads share the same cache. The rotation loop then
just does multiply-add with cached values — no trig calls in the inner loop.

**Fix:** Precompute `sin[pos * freq]` and `cos[pos * freq]` arrays before the
head loop:

```rust
// Precompute once per position
let mut sin_cache = vec![0.0f32; half];
let mut cos_cache = vec![0.0f32; half];
for i in 0..half {
    let th = p * freqs[i];
    let (sn, cs) = th.sin_cos();
    sin_cache[i] = sn;
    cos_cache[i] = cs;
}
// Then use cached values in the head loop
for h in 0..nh {
    for i in 0..half {
        let (sn, cs) = (sin_cache[i], cos_cache[i]);  // no trig call
        // ...
    }
}
```

This reduces `sin_cos` calls from `nh × half` to just `half` per position.

---

## Bottleneck #3: Redundant Activation Quantization

**Impact:** ~5% of total CPU time.
**Difficulty:** Medium.

### Current Implementation

**Location:** `src/kernel.rs`, lines 59-67

```rust
pub fn cpu_quant_matmul_f32(w, x, out, od, id, nt) {
    let nbe = id / 32;
    let mut qb = vec![0u8; nt * nbe * Q8B];   // heap alloc every call
    crate::avx2::quantize_row_q8_0_buf(x, nt, id, &mut qb);
    cpu_quant_matmul(w, &qb, out, od, id, nt)
}
```

**Location:** `src/models/qwen2/forward.rs`, lines 87-91

```rust
crate::kernel::quant_matmul_f32_batch(&mut [
    (l.wq.as_ref().unwrap(), &mut bq, nqt),
    (l.wk.as_ref().unwrap(), &mut bk, nkt),
    (l.wv.as_ref().unwrap(), &mut bv, nvt),
], &bn, nt, ne);
```

The batch matmul calls `cpu_quant_matmul_f32` independently for each weight.
All three (Wq, Wk, Wv) share the same activation `bn`, but each call:
1. Allocates a new Q8_0 buffer on the heap
2. Quantizes `bn` from f32 to Q8_0

Result: the same `bn` is quantized **3 times** per layer, with 3 heap
allocations.

**llama.cpp approach:** Uses `Q8_0x4` interleaved format — quantizes 4 rows
simultaneously into an interleaved layout. The quantized data is then reused
across multiple weight matrix multiplications.

**Fix:** Quantize the activation once before the batch, then pass the pre-
quantized Q8_0 buffer to each matmul:

```rust
// Quantize once
let mut q8_buf = vec![0u8; nt * nbe * Q8B];
quantize_row_q8_0_buf(&bn, nt, ne, &mut q8_buf);
// Reuse for all three matmuls
dot_q4_0_q8_0_batch(wq, &q8_buf, &mut bq, ...);
dot_q4_0_q8_0_batch(wk, &q8_buf, &mut bk, ...);
dot_q4_0_q8_0_batch(wv, &q8_buf, &mut bv, ...);
```

---

## Bottleneck #4: Q4_K / Q6_K Dot Products — No AVX2

**Impact:** Depends on model quantization type. For Q4_K models, this affects
**all** matmul operations (~70% of total time).
**Difficulty:** High (requires SIMD bit manipulation).

### Current Implementation

**Location:** `src/avx2.rs`, lines 201 and 290

```rust
TensorType::Q4_K => dot_q4_k_q8_0_scalar(w, x, nb),
TensorType::Q6_K => dot_q6_k_q8_0_scalar(w, x, nb),
```

Only Q4_0 and Q8_0 have AVX2 implementations. Q4_K and Q6_K fall through to
pure scalar code with:
- Bit-by-bit nibble extraction with `if j % 2 == 0` branches per element
- 6-bit scale unpacking with bitwise operations per subblock
- 256 scalar multiply-accumulates per dot product call

**llama.cpp approach:** All quantization types have AVX2 paths:

```cpp
// Q4_K AVX2: denibble() unpacks 32 nibbles in one SIMD operation
static inline __m256i denibble(const uint8_t *p) {
    __m128i x = _mm_loadu_si128((const __m128i *)p);
    return _mm256_and_si256(_mm256_set1_epi8(15),
        _mm256_insertf128_si256(
            _mm256_castsi128_si256(x),
            _mm_srli_epi16(x, 4), 1));
}
```

Combined with `_mm256_sign_epi8` trick for signed dot product and
`_mm256_maddubs_epi16` for 8-bit integer multiply-add.

**Fix:** Implement AVX2 paths for Q4_K (highest priority) and Q6_K. The Q4_K
block format has 8 subblocks of 32 elements each, with super-block scale and
subblock scales. The key operations to SIMD-ize:
1. Scale unpacking (6-bit min/max scales from super-block)
2. Nibble denibbling (same `denibble` trick as llama.cpp)
3. Integer dot product via `maddubs` + `madd`

---

## Bottleneck #5: Attention — Strided KV Access + No Flash Attention

**Impact:** ~10-15% of total CPU time (grows with context length).
**Difficulty:** Medium.

### Current Implementation

**Location:** `src/models/qwen2/forward.rs`, lines 247-270

```rust
fn gqa_attn(q, ka, va, pos, nt, nkv, nh, nk, hd, out, scrs) {
    for h in 0..nh {
        for t in 0..nt {
            for kv in 0..vl {
                // K cache access: stride = nk * hd (e.g., 4KB)
                let ks = kv * nk * hd + hk * hd;
                let s = vec_dot_f32(hd, &q[qs..], &ka[ks..]) * sc;
                scrs[kv] = s;
            }
            // Softmax: 3 separate passes
            vec_soft_max_f32(vl, &mut scrs[..vl], mx);   // pass 1: exp
            vec_scale_f32(vl, &mut scrs[..vl], 1.0/s);   // pass 2: normalize
            // V accumulation: per-position function call
            for kv in 0..vl {
                vec_muladd_f32(hd, scrs[kv], &va[ks..], &out[qs..]);
            }
        }
    }
}
```

**Problems:**

1. **Strided KV cache access:** K/V are stored as `[nkv][nk * hd]`. Accessing
   consecutive KV positions for one head requires jumping `nk * hd` floats
   (e.g., 8 × 128 × 4 = 4096 bytes). Each `vec_dot_f32` reads only 128 floats
   (512 bytes), so 87.5% of each fetched cache line is wasted.

2. **Three-pass softmax:** Scores are computed (pass 1), then exp + sum
   (pass 2), then normalized (pass 3), then V accumulated (pass 4). Four
   separate traversals of the score/output arrays.

3. **Per-position function call overhead:** `vec_muladd_f32` is called once per
   KV position with only `hd=128` elements. The function call overhead is
   non-trivial relative to the work done.

**llama.cpp approach:** Tiled Flash Attention on CPU:
- Q, K, V processed in tiles (e.g., 32 query rows × 64 KV positions)
- K is **packed transposed** (`K_f32[dk][kv]`) so the KV dimension is
  contiguous for SIMD
- **Online softmax**: single pass computes scores, finds max, applies
  correction, and accumulates V — no separate softmax pass
- Uses `simd_gemm` with 6×2 microkernel for the Q*K^T and softmax*V
  multiplications within each tile
- Multi-chunk parallelism: KV dimension split across threads, partial results
  reduced at the end

**Fix (incremental):**
1. Merge softmax into a single-pass online softmax (same as GPU path)
2. Precompute cos/sin cache for RoPE (already covered in Bottleneck #2)
3. Consider transposing K cache for better access pattern

---

## Bottleneck #6: KV Cache — Full f32 Storage

**Impact:** 2-4× memory bandwidth waste during attention.
**Difficulty:** Medium.

### Current Implementation

**Location:** `src/cache.rs`

```rust
pub struct KVCacheLayer {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    // ...
}
```

For a model with `n_kv_head=8`, `head_dim=128`, `max_seq_len=2048`:
- Per layer: `2 × 2048 × 8 × 128 × 4 bytes = 16 MB`
- 24 layers: **384 MB** of KV cache

During attention, the entire K and V cache is streamed through the CPU for
each head. Full f32 means 2× the bandwidth of f16, 4× the bandwidth of
quantized KV.

**llama.cpp approach:** Supports f16 KV cache (`--cache-type-k f16`), reducing
memory by half. Also supports quantized KV cache (Q8_0, Q4_0) for 4×
reduction.

**Fix:** Add f16 KV cache option. The attention dot product would need to
handle f16 K/V (load + widen to f32 for computation).

---

## Bottleneck #7: Per-Layer Heap Allocations

**Impact:** ~1-2% of total CPU time.
**Difficulty:** Low.

### Current Implementation

**Location:** `src/models/qwen2/forward.rs`, line 104

```rust
let ffn_in = bf[..nt * ne].to_vec();   // 16KB alloc per layer
```

**Location:** `src/kernel.rs`, line 64

```rust
let mut qb = vec![0u8; nt * nbe * Q8B];  // ~4KB alloc per matmul
```

Total: ~200+ heap allocations per forward pass (24 layers × ~8 allocs each).

**Fix:** Pre-allocate scratch buffers outside the layer loop and reuse them.
The `ffn_in` clone exists because `bf` is simultaneously used as output for
`ffn_up` — this can be resolved by using a separate pre-allocated buffer.

---

## Bottleneck #8: Matmul Loop Ordering

**Impact:** Minor for decode (nt=1), significant for prefill (nt>1).
**Difficulty:** Medium.

### Current Implementation

**Location:** `src/kernel.rs`, lines 69-136

```rust
for o in 0..od {                    // outer: output dimension
    let wrow = &wb[o * ws..(o+1) * ws];
    for t in 0..nt {                // inner: token
        out[t * od + o] = dot_q4_0_q8_0(wrow, &x[t * nb * Q8B ..]);
    }
}
```

**Problems:**
- Weight row is reused across tokens (good for cache).
- But output write `out[t * od + o]` is strided by `od` between tokens.
- For prefill, Q8_0 activation slices are accessed with stride `nb * Q8B`
  between tokens — cache-unfriendly.
- No loop tiling or blocking for cache efficiency.

**llama.cpp approach:** Tiled GEMM with register blocking. The `tinyBLAS`
kernel computes 4×N tiles of the output simultaneously, keeping accumulators
in AVX2 registers. K is packed transposed for contiguous access.

**Fix:** For decode (nt=1), the current ordering is fine. For prefill, consider
loop tiling: process blocks of output dims × blocks of tokens together.

---

## Bottleneck #9: Dead/Buggy Code in vec_ops

**Location:** `src/vec_ops.rs`, lines 402-430

```rust
pub fn rms_norm_f32(n: usize, y: &mut [f32], x: &[f32], eps: f32) {
    let mut sum_sq = 0.0f64;
    for i in 0..n { sum_sq += (x[i] as f64) * (x[i] as f64); }
    let scale = 1.0 / ((sum_sq / n as f64) as f32 + eps).sqrt();
    vec_scale_f32(n, y, scale);       // ← scales y BEFORE x is copied!
    if y.as_ptr() != x.as_ptr() {
        vec_cpy_f32(n, y, x);
        vec_scale_f32(n, y, scale);   // ← scales again after copy
    } else {
        vec_scale_f32(n, y, scale);   // ← scales a third time
    }
}
```

**Problems:**
- Never called from the forward path.
- When `y != x`: scales `y` (uninitialized), copies `x` over it, scales again.
  First scale is wasted work on garbage data.
- Sum-of-squares is scalar f64.

**Fix:** Either fix and use this function, or delete it and use the
self-contained version in `forward.rs` (but add SIMD to that one).

---

## Optimization Priority Matrix

| Priority | Bottleneck | Fix | Expected Gain | Difficulty |
|----------|-----------|-----|---------------|------------|
| **P0** | #1 Scalar hotspots | Replace with vec_ops AVX2 calls | 2-3× for these ops (~10% total) | Low |
| **P0** | #2 RoPE redundant trig | Precompute cos/sin cache | 32× fewer sin_cos calls (~5% total) | Low |
| **P1** | #3 Redundant quantization | Quantize once, reuse Q8_0 buffer | Eliminate 2/3 of quant work (~5% total) | Medium |
| **P1** | #5 Attention | Online softmax single pass | Reduce 4 passes to 1 (~10% total) | Medium |
| **P2** | #4 Q4_K no AVX2 | Add AVX2 dot product | Huge if using Q4_K model | High |
| **P2** | #6 KV cache f32 | Add f16 KV cache option | 2× bandwidth reduction | Medium |
| **P3** | #7 Heap allocations | Pre-allocate scratch buffers | ~1-2% total | Low |
| **P3** | #8 Matmul loop order | Loop tiling for prefill | Only helps prefill, not decode | Medium |

## Summary: Path from 21 tok/s to llama.cpp Parity

The largest gains come from P0 fixes (scalar → SIMD, RoPE cache), which require
minimal code changes and could bring decode to ~30-40 tok/s. P1 fixes (quant
reuse, online softmax) add another ~10-15%. P2 fixes (Q4_K AVX2, f16 KV) are
needed for full parity with llama.cpp on specific model types.

The fundamental gap is that minfer's CPU path has many operations that were
written as simple scalar loops during initial development and never optimized,
while `vec_ops.rs` already contains AVX2 implementations that simply aren't
wired into the forward path.
