# Metal Backend Optimizations

This document details all optimizations applied to minfer's Metal GPU backend,
including the principles behind each change, the code modifications, and the
measured performance impact.

## Baseline

Before optimization, the Metal GPU backend had 4 correctness bugs and ran at
~130 tok/s on Qwen2-0.5B (decode). llama.cpp achieved ~300 tok/s on the same
model and hardware (Apple M4 Pro).

Final result: **334 tok/s** (2.6x improvement over the buggy GPU baseline).

---

## Phase 1: Correctness Bug Fixes

### Bug 1: RoPE freq_scale Not Applied

**Problem:** The `rope_freq_scale` hyperparameter was loaded from the GGUF file
but never passed to the Metal RoPE kernel. The frequency formula was:

```
freq = 1.0 / pow(freq_base, 2i / d)          // WRONG
freq = freq_scale / pow(freq_base, 2i / d)    // CORRECT
```

Without `freq_scale`, the rotary embeddings use wrong rotation angles, producing
garbage output for any model where `freq_scale != 1.0`.

**Fix:** Added `freq_scale` parameter through the entire call chain:
- `metal.metal`: `kernel_rope_f32` — new `buffer(5)` for freq_scale, positions
  shifted to `buffer(6)`
- `metal.rs`: `rope_f32()` and `layer_gpu()` — new `freq_scale: f32` parameter
- `forward.rs`: `apply_rope()` — passes `hp.rope_freq_scale`

### Bug 2: Output Bias Not Applied

**Problem:** Qwen2's output projection has a bias term (`output_b`) that was
loaded from GGUF but never added after the output matmul. This caused the final
logits to be offset, degrading output quality.

**Fix:**
- GPU path: `output_norm_gpu()` now accepts `output_b: Option<&Tensor>`, applies
  `add_bias_f32` after the output matmul
- CPU path: `forward.rs` adds `output_b` data after the output projection

### Bug 3: Softmax Max Initialization

**Problem:** The attention kernel initialized the running max to `-1e30f` instead
of `-INFINITY`. For the first KV position, the online softmax correction factor
`exp(old_max - new_max)` would compute `exp(-1e30 - s)` which underflows to 0,
losing the first attention weight entirely.

**Fix:** One-line change: `float mx = -1e30f` → `float mx = -INFINITY`

### Bug 4: Hardcoded Stack Array in Attention

**Problem:** The attention kernel used `float acc[128]` as a stack-allocated
output accumulator, limiting the head dimension to 128 floats. For models with
larger head dimensions, this would silently corrupt the stack.

**Fix:** Accumulate directly into the device output buffer `ohead` instead of a
local array. The output is normalized in-place at the end.

---

## Phase 2: Flash Attention (Online Softmax)

### Principle

Standard attention requires two passes over the KV cache:
1. Compute all Q*K dot products → find the max score
2. Compute softmax(Q*K) * V using the max for numerical stability

**Online softmax** (Flash Attention) merges this into a single pass by maintaining
a running max `m` and running sum `S`. When a new score exceeds the current max,
the output accumulator is corrected by a factor `exp(old_m - new_m)`:

```
for each KV position kv:
    s = dot(Q, K[kv]) * scale
    new_m = max(m, s)
    correction = exp(m - new_m)

    O *= correction          // rescale previous accumulation
    S = S * correction       // rescale denominator
    O += exp(s - new_m) * V[kv]
    S += exp(s - new_m)
    m = new_m

O /= S   // final normalization
```

This eliminates the redundant Q*K recomputation. The correction factor ensures
numerical equivalence with the two-pass algorithm.

### Additional: float4 Vectorized Memory Access

The KV cache stores `float` values. By casting to `float4*` and using Metal's
`dot(float4, float4)` intrinsic, we load 4 floats per memory transaction instead
of 1, quadrupling memory bandwidth utilization:

```metal
device const float4 * q4 = (device const float4 *)qhead;
device const float4 * k4 = (device const float4 *)(khead + kv * stride_kv);
float s = 0.0f;
for (int i = 0; i < hd4; i++) s += dot(q4[i], k4[i]);
```

### Result

130 tok/s → 151 tok/s (+16%)

---

## Phase 3: SIMD-Parallel Attention (Vec Kernel)

### Principle

The flash attention kernel from Phase 2 used **1 thread per (token, head)**.
For decode (nt=1, nh=14), this launches only 14 threads — leaving thousands of
GPU cores idle.

Inspired by llama.cpp's `vec` kernel, we parallelize across the KV dimension
instead of the head dimension:

- **1 threadgroup per (token, head)** with **32 threads** (1 simdgroup)
- Each thread handles `NE=2` KV positions per batch iteration
- 32 threads × 2 positions = 64 KV entries processed per batch
- Dot products are computed independently by each thread (each thread loads its
  own K/V vectors and computes the full Q*K dot product)
- `simd_max()` finds the batch-wide max score across all threads
- Since `simd_max` broadcasts its result to all lanes, the softmax weights
  `e0 = exp(s0 - new_mx)` are uniform across threads — each thread independently
  accumulates its own output slice
- After all batches, `simd_sum()` reduces the per-thread output accumulators
  into the final result

```
for batch in 0..nkv step C:
    // Each thread computes dot products for its own KV positions
    s0 = dot(Q, K[kv0]) * scale    // thread's KV position 0
    s1 = dot(Q, K[kv1]) * scale    // thread's KV position 1

    // SIMD-wide max across all threads' scores
    batch_mx = simd_max(max(s0, s1))
    new_mx = max(mx, batch_mx)     // broadcast to all lanes

    // Online softmax correction (uniform across threads)
    corr = exp(mx - new_mx)
    e0 = exp(s0 - new_mx)
    e1 = exp(s1 - new_mx)

    // Each thread accumulates its own output slice
    oc[i] *= corr
    oc[i] += e0 * V[kv0][i]
    oc[i] += e1 * V[kv1][i]

// Final reduction: sum across all threads
S = simd_sum(S)
oc[i] = simd_sum(oc[i])
```

### Dispatch Change

```rust
// Old: 1 thread per (token, head)
dispatch_2d(nt, nh, 1, 1);

// New: 32 threads per (token, head), using threadgroup_position_in_grid
dispatch_2d(nt, nh, 32, 1);
```

Kernel attributes changed from `thread_position_in_grid` to
`threadgroup_position_in_grid` + `thread_index_in_simdgroup`.

### Result

151 tok/s → 196 tok/s (+30%)

---

## Phase 4: SIMD-Parallel RMSNorm

### Principle

The original RMSNorm kernel used **1 thread per row**, processing all `d` elements
serially:

```metal
// OLD: 1 thread does d iterations
for (int i = 0; i < d; i++) ss += r[i] * r[i];
```

For d=1024, this is 1024 sequential multiply-adds per thread. During decode
(nt=1), only 1 thread is active per RMSNorm call — the GPU is ~0% utilized.
RMSNorm is called 3x per layer (attention norm, FFN norm, output norm) = 72
times total.

**Parallel approach** (borrowed from llama.cpp's `kernel_rms_norm_fuse_impl`):

- **1 threadgroup per row**, 32 threads (1 simdgroup) per threadgroup
- Each thread processes `d/4/32` float4 elements (for d=1024: 8 elements each)
- Partial sum-of-squares reduced via `simd_sum()` — a single hardware instruction
- All threads then compute the same `scale` and write output in parallel

```metal
// Each thread accumulates partial sum-of-squares
float ss = 0.0f;
for (int i = tpitg.x; i < d4; i += 32) {
    ss += dot(x4[i], x4[i]);    // float4 dot product
}
ss = simd_sum(ss);              // warp-level reduction

// All threads compute the same scale
float scale = 1.0f / sqrt(ss / (float)d + eps);

// Parallel output write with float4
for (int i = tpitg.x; i < d4; i += 32) {
    y4[i] = x4[i] * scale * w4[i];
}
```

### Dispatch Change

```rust
// Old: n threads total, 1 per row, threadgroup size 256
dispatch_1d(n, 256);

// New: n threadgroups (1 per row), 32 threads each
dispatch_2d(n, 1, 32, 1);
```

### Note on Multi-Simdgroup Reduction

We attempted 128 threads (4 simdgroups) with shared memory reduction, but
encountered correctness issues with cross-simdgroup `simd_sum` over
uninitialized shared memory slots. The 32-thread (1 simdgroup) version is
sufficient for decode workloads where only 1 row is processed — the bottleneck
is memory bandwidth, not compute, and 32 threads already saturate it.

### Result

196 tok/s → 334 tok/s (+70%)

---

## Phase 5: SwiGLU Fusion

### Principle

The FFN branch in Qwen2 (SwiGLU architecture) computes:

```
gate = matmul(ffn_gate, norm_out)
up   = matmul(ffn_up,   norm_out)
out  = silu(gate) * up
```

Originally this required **two separate GPU kernels**:
1. `silu_f32(bg_buf)` — in-place SiLU on gate output
2. `mul_f32(bg_buf, bf_buf, bg_buf)` — element-wise multiply

Each kernel reads and writes the full gate buffer (`nt * nf` floats). The second
kernel re-reads data that the first kernel just wrote.

**Fused kernel** (`kernel_swiglu_f32`) computes both operations in one pass:

```metal
// dst[i] = silu(gate[i]) * up[i]
float g = gate[tid];
dst[tid] = (g / (1.0f + exp(-g))) * up[tid];
```

Each element is read once from `gate` and `up`, computed, and written once to
`dst`. This eliminates:
- 1 kernel dispatch per layer (24 layers = 24 fewer dispatches)
- 1 full read+write pass over the gate buffer per layer

### Data Flow

```
Before: matmul(gate) → bg_buf →[silu]→ bg_buf →[mul × bf_buf]→ bg_buf → quantize
After:  matmul(gate) → bg_buf →[swiglu × bf_buf]→ bg_buf → quantize
```

The `dst` buffer aliases `gate` (same `bg_buf`), which is safe because each
thread reads `gate[tid]` before writing `dst[tid]`, and thread indices are
unique.

### Files Changed

| File | Change |
|------|--------|
| `metal.metal` | New `kernel_swiglu_f32` |
| `metal.rs` | New `pl_swiglu` pipeline + `swiglu_f32()` method |
| `metal.rs` `layer_gpu` | 2 calls → 1 call |

### Result

~312-334 tok/s (within noise of Phase 4; long-text generation improved from
166 → 186 tok/s due to reduced kernel launch overhead).

---

## Performance Summary

| Phase | Optimization | Decode (short) | Cumulative Gain |
|-------|-------------|----------------|-----------------|
| Baseline | GPU enabled + bug fixes | 130 tok/s | 1.0x |
| +2 | Flash Attention + float4 | 151 tok/s | 1.2x |
| +3 | SIMD-parallel attention | 196 tok/s | 1.5x |
| +4 | SIMD-parallel RMSNorm | 334 tok/s | 2.6x |
| +5 | SwiGLU fusion | 312-334 tok/s | 2.5x |

**Overall: 130 → 334 tok/s (2.6x)** on Qwen2-0.5B-Instruct, Apple M4 Pro.

Notes on measurement:
- minfer currently reports `(prompt + generated) / total_time` as its tok/s
  metric, which blends prefill and decode into a single number
- The pre-/post-optimisation numbers above used Qwen2-0.5B (older model);
  current benchmarks below use Qwen2.5-0.5B (same architecture, different
  tokenizer) — model-specific differences may account for minor variations

## Current vs llama.cpp (Q4_0, Qwen2.5-0.5B, Apple M4 Pro)

Measured 2026-07-31 … 2026-08-01 with the same model file and the same
`"hello"` prompt (chat template → 30 prompt tokens, 9 generated tokens):

| | llama.cpp | minfer (after P1 GEMM) | Gap |
|---|---|---|---|
| **Prefill** (30 tokens) | 1318 t/s (22.8 ms) | ~510 t/s (~59 ms) | **2.6×** |
| **Prefill** (70 tokens) | ~1318 t/s | ~870 t/s | ~1.5× |
| **Decode** (9 tokens) | 345 t/s (26.1 ms) | ~340 t/s¹ (~35 ms) | ~1.0× (blended) |

The prefill gap is largely a **batch-size / fixed-overhead** effect: at 70
tokens the GEMM reaches ~870 t/s (vs ~650 for the f32 multi kernel), while
short prompts (30 tokens) amortize the fixed per-forward overhead less well.
The simdgroup GEMM (P1) now beats the f32 multi kernel for nt ≥ 16 — see the
section below.

¹ minfer reports a blended rate = (30 + 9) / (prefill + decode time), which
counts prefill overhead in the denominator. Before P1 this was 281 t/s
(~48.9 ms); after P1 it is ~330-350 t/s. Pure decode-only is lower
(~200 t/s) because the prefill portion of the denominator inflates the
blended number less as generation lengthens.

## Prefill Gap: simdgroup GEMM — implemented, fixed, and SHIPPED (P1)

minfer's Q4_0 prefill kernel (`kernel_q4_0_f32_matmul_multi`) uses a
**scalar dot-product loop**:

```metal
for (int j = 0; j < 16; j++) {
    bs += (int(byte & 0x0F) - 8) * int(xq[j])
        + (int(byte >> 4) - 8) * int(xq[j + 16]);
}
```

llama.cpp's equivalent (`kernel_mul_mm_q4_0_f32`) uses
**`simdgroup_matrix`** — the hardware matrix‑multiply engine on Apple‑Silicon.

### P0 experiment (2026-08-01): simdgroup GEMM implemented, then reverted

A full simdgroup GEMM was implemented following llama.cpp's legacy
`kernel_mul_mm` structure (64×32 output tile, 4 simdgroups × 32 threads,
Q4_0 dequant staged into threadgroup memory, `simdgroup_half8x8` inputs →
`simdgroup_float8x8` accumulators, transposed store). It was verified
**correct** (per-layer cos ≥ 0.9999 vs CPU), but **slower than the simple
f32 multi kernel**:

| Prefill batch | f32 multi | simdgroup GEMM |
|---|---|---|
| 30 tokens | ~440 t/s | ~422 t/s |
| 70 tokens | ~895–1155 t/s | ~490–594 t/s |

The GEMM was removed.

### P1 (2026-08-01): faithful llama.cpp transcription — NOW SHIPPED

A fresh transcription of llama.cpp's `kernel_mul_mm_q4_0_f32` (legacy
simdgroup path) fixed **three bugs** in the P0 attempt:

1. **B-staging `by`/`bly`** — llama writes to the *raw* `(tiitg/NL1)/8` and
   `(tiitg/NL1)%8` (unclamped) sb positions, reading the activations from the
   clamped row. P0 used the clamped index for both, leaving out-of-range N-rows
   of the B tile uninitialized (stale/NaN threadgroup memory → garbage).
2. **Store transpose** — `simdgroup_store` with `transpose=false` is the
   *row-major* store (element [r][c] → dst[r*stride + c]); minfer's out is
   `[nt][od]` = C^T of llama's `[M][N]`. The direct store must use
   `C + 8*(i/4)*od + 8*(i%4)` with `transpose=false`, and the bc_out store
   `temp_str + 8*(i%4) + 8*NR0*(i/4)` with `transpose=false` (the bc_out
   copy reads back `temp_str[j*NR0 + m]` = C[N=j][M=m]).
3. **Barrier scope** — the multiply loop's `simdgroup_barrier(mem_flags::mem_none)`
   does not make the sa/sb writes (done by *other* simdgroups) visible to
   `simdgroup_load`. On M4 Pro this manifested as a **non-deterministic race** at
   od=4864 (gate/up) with nt%32 != 0. Fixed by making the **first** multiply
   barrier `mem_flags::mem_threadgroup` (the two subsequent ones stay `mem_none`).

Also: Q4_0 block layout in GGUF is **byte j = {element j (lo), element j+16
(hi)}**, not byte j/2 = {2j, 2j+1}. The llama `dequantize_q4_0` reads `qs` as
uint16 and the store maps `temp_a[i/4][i%4]` accordingly.

**Isolation test** (`tests/gemm_isolation.rs`, macOS): runs the kernel on
deterministic synthetic weights/acts at nt = 12/30/32/33 and asserts
(1) run-to-run determinism and (2) agreement with a scalar CPU reference
(≤ 2.5e-3, half-precision tolerance). All four nt values pass.

**Result** (Qwen2.5-0.5B, M4 Pro, `"hello"` prompt, 30-token chat prefill):

| Prefill batch | f32 multi | simdgroup GEMM | Gain |
|---|---|---|---|
| 12 tokens | ~150 t/s | ~88 t/s | (GEMM slower — small-batch overhead) |
| 30 tokens | ~460 t/s | ~510 t/s | +11% |
| 70 tokens | ~650 t/s | ~870 t/s | +34% |

The GEMM is dispatched for **nt ≥ 16** (below that the f32 multi kernel's
lower fixed overhead wins). `MINFER_GEMM=0` forces the f32 multi path for
A/B comparison.


## Decode Gap: Dispatch Overhead (1.9×) + F16 KV Cache

For 1‑token decode the matmul kernels are memory‑bound and the scalar
dot‑product loop is less of a bottleneck. The dominant delta comes from
two sources:

### 1. Q4_0 quantize + matmul — two dispatches per matmul (FIXED 2026-08-01)

minfer's Q4_0 path originally required a separate `quantize_q8_0` kernel
before every `q4_0_q8_0_matmul`:

```
Old Q4_0 path: [quantize_q8_0]  →  [q4_0_q8_0_matmul]   = 2 dispatches
Other types:   [f32_matmul]                              = 1 dispatch
```

The quantization is shared per norm output, not per matmul: 4 `quantize_q8_0`
dispatches per layer (attn_norm, attn_out, ffn_norm, swiglu) = **96 per
forward pass** (not 168 — corrected from an earlier draft).

**Root cause**: minfer's Q4_0 path mimicked llama.cpp's **CPU** backend
(`vec_dot_type = Q8_0`), but llama.cpp's **Metal** backend reads **f32
activations directly** for Q4_0 (verified: `mul_vec_q_n_f32_impl` casts
`src1` to `float*`, no Q8_0 quantization step anywhere in the Metal
mul_mat dispatch). minfer's Q4_0 Q8_0 path was the odd one out.

**Fix (P1)**: route Q4_0 through the existing f32 matmul path
(`matmul_on_gpu_buf` always calls `quant_matmul_f32_on_gpu_buf`), remove
the 4 `quantize_q8_0` calls in `layer_gpu` + 1 in `output_norm_gpu`, and
delete the now-dead `quantize_q8_0` method/pipeline/shader. Q4_0 now matches
llama.cpp Metal behaviour.

**Measured** (Q4_0, Qwen2.5-0.5B, M4 Pro, "hello" → 30 prompt / 9 gen):
- prefill 338 → ~440 t/s (+~30 %), blended generated 281 → ~340 t/s (+~20 %)
- bigger than the initial estimate: the f32 `block_q4_0_dot_y` kernel (interleaved
  ushort trick) is also faster per-dot than the Q8_0 int8 kernel, not just
  fewer dispatches. Primary value is alignment with llama.cpp Metal and code
  simplification, which unblocks the P0 GEMM work.

### 2. F32 KV cache vs llama.cpp's F16

minfer stores the key‑value cache in `float` (4 bytes per element);
llama.cpp uses `half` (2 bytes). This doubles the attention and store‑kv
memory bandwidth. The impact is small for 0.5B‑class models with short
contexts, but grows linearly with head‑dimension × sequence‑length.

## Files Modified

| File | Changes |
|------|---------|
| `src/metal.metal` | New/rewritten kernels: `kernel_gqa_attn_f32` (flash attention + SIMD vec), `kernel_rms_norm_f32` (parallel), `kernel_swiglu_f32` (fused), `kernel_rope_f32` (freq_scale fix). Additional kernels added later: `kernel_q5_1_f32_matmul` + `_multi`, `kernel_q5_k_f32_matmul` + `_multi`. P1 (2026-08-01): `kernel_quantize_q8_0` removed |
| `src/metal.rs` | New pipeline `pl_swiglu`, new method `swiglu_f32()`, updated `rms_norm()` dispatch, updated `rope_f32()`/`layer_gpu()` signatures, updated `output_norm_gpu()` for output bias. Q5_1/Q5_K pipeline states + dispatch added later. P1 (2026-08-01): Q4_0 routed through f32 path in `matmul_on_gpu_buf`, 5 `quantize_q8_0` calls removed, dead `quantize_q8_0` method + `pl_quantize_q8_0` pipeline removed |
| `src/models/qwen2/forward.rs` | `apply_rope()` freq_scale, output_b in CPU path, GPU path enabled. Per-layer CPU‑fallback added later (partial GPU work submitted on Q5_K layer failure) |
| `src/avx2.rs` / `src/kernel.rs` / `src/models/qwen2/forward.rs` | Q5_K formula fix (signed‑16 → unsigned) + qh‑indexing fix (`qh[p] bit s`), 2026-07-31 |

## Recently Completed (2026-07-31 … 2026-08-01)

| Item | Detail |
|------|--------|
| Q5_1 Metal kernel | `kernel_q5_1_f32_matmul` + `_multi` (F32 activation path, same structure as Q5_0) |
| Q5_K Metal kernel | `kernel_q5_k_f32_matmul` + `_multi` (176B/256‑elem super‑block, `qh[p] bit s` indexing, unsigned formula) |
| Q5_K_M GPU verified | Qwen2.5‑0.5B Q5_K_M full GPU: prefill ~240 t/s, decode ~250 t/s |
| Per‑layer CPU fallback | When `layer_gpu` fails for a Q5_K layer, the engine submits partial GPU work, downloads the hidden state, and resumes the remaining layers on CPU |
| Q5_K formula + qh‑indexing fixes | Two independent bugs in minfer's CPU Q5_K implementation — signed formula (`dl·(u-16)-ml`) corrected to unsigned (`dl·u-ml`), and qh high‑bit indexing corrected from `qh[sub·4+pos/8] bit pos%8` to `qh[pos] bit sub` (matching llama.cpp's quantizer layout) |
| **P1: Q4_0 → f32 activation path** | Q4_0 no longer Q8_0‑quantizes activations; routed through the existing f32 matmul kernel (matching llama.cpp Metal). Removed 4 `quantize_q8_0` calls in `layer_gpu` + 1 in `output_norm_gpu`, deleted the dead `quantize_q8_0` method/pipeline/shader. Decode ~+5‑10 %, minimal prefill change |

## Remaining Optimization Opportunities

Ranked by estimated impact‑to‑effort ratio for the current workload
(Q4_0, Qwen2.5‑0.5B, Apple M4 Pro).

| Priority | Item | Impact | Effort | Notes |
|:---:|------|--------|--------|-------|
| ~~P0~~ | ~~GEMM kernel: simdgroup_mm~~ | — | — | **Investigated 2026-08-01: implemented correctly but 2× slower than the f32 multi kernel for 0.5B-class models — reverted.** Would only pay off for 7B+ models / very long prefill batches |
| **P2** | **KV cache: F32 → F16** | 2× attention bandwidth | Small | Change cache allocation, `store_kv`, `gqa_attn` kernel (~30 lines) |
| **P3** | **Improve f32 multi kernel occupancy** | up to 2× prefill | Medium | The simple f32 kernel already reaches ~1000 t/s at 70 tokens; tune grid/tile/threads to close the gap to llama.cpp's 1318 t/s |
| P4 | Matmul + bias fusion | 1 dispatch/matmul | Medium | Merge `add_bias_f32` into matmul epilogue |
| P5 | Residual add + RMSNorm fusion | 2 dispatches/layer | Medium | Merge `add_f32` + `rms_norm` into one kernel |
| P6 | Element‑wise float4 vectorisation | 1‑2% | Small | `add_f32`, `mul_f32`, `silu_f32` still use scalar loads |
| P7 | RoPE parallelisation | ~1% | Small | Currently 1 thread per (token, head) |
| P8 | RoPE + store_kv fusion | 1 dispatch/layer | Small | Merge K's RoPE transform with KV cache scatter |
