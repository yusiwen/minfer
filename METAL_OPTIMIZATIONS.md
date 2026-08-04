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

## Current vs llama.cpp (Qwen2.5-0.5B, Apple M4 Pro)

Measured 2026-08-01 with the `"Tell me about Transformer architecture."` prompt
(chat template → 35 prompt tokens) and the `"hello"` prompt (→ 30 tokens).

> **Metric warning**: llama.cpp's `Generation:` is **pure** decode (only
> generation time in the denominator). minfer's `Generated:` is a **blended**
> rate = `(prompt + generated) / total_time`, which counts prefill in the
> denominator — the two are **not** directly comparable. All rates below are
> pure decode / pure prefill.

### Prefill (Q4_0 vs Q4_K_M / Q5_K_M)

| | llama.cpp | minfer Q4_0 | minfer Q4_K_M / Q5_K_M | Gap |
|---|---|---|---|---|
| **Prefill** (35 tokens) | ~1750 t/s | ~554 t/s | ~240 t/s | Q4_0 3.2×; K_M **7.3×** |
| **Prefill** (70 tokens) | ~2600 t/s | ~780 t/s | — | ~3.3× |

**Why Q4_K_M/Q5_K_M prefill is ~7× slower**: minfer's simdgroup GEMM
(`kernel_q4_0_mm_f32`) supports **Q4_0 only**. Q4_K_M's weights are
`q5_0 / q8_0 / q4_k / q5_1 / q6_k` — none use the GEMM, so they fall back to the
scalar f32 multi kernel (no `simdgroup_matrix`). llama.cpp ships a `kernel_mul_mm`
GEMM for **every** quant type. Even Q4_0 prefill (554 vs 1750) has a 3× gap
because the GEMM only dispatches for nt ≥ 16 and the f32 multi covers the rest.

### Decode (pure, single-token)

| Context (KV length) | minfer Q4_0 | llama.cpp Q4_0 | Gap |
|---|---|---|---|
| Short (~40) | ~187 t/s | ~375 t/s | **2.0×** |
| Long (~400) | ~86 t/s | ~279 t/s | **3.2×** |
| KV-growth degradation | 187 → 86 (**2.2×**) | 375 → 279 (1.34×) | — |

**Why decode slows 2.2× as context grows (vs llama's 1.34×)**: minfer's
attention re-reads the **entire KV cache** every decode step. The default cache
is **F32 (4 bytes/elem)** vs llama.cpp's **F16 (2 bytes)**, but an F16 cache
(`MINFER_CACHE_TYPE=f16`) was measured to *not* recover the degradation for the
0.5B model (decode is dispatch-latency-bound, not KV-bandwidth-bound — see the
Decode Gap section). The base ~3× decode gap is per-dispatch **encoding**
overhead (~24µs/kernel vs llama's ~7µs via multi-command-buffer parallel
encoding), not the dispatch count itself — see Decode Gap §0a.

¹ Blended-rate note: a short 9-token generation (30 prompt + 9 gen) reports
~340 t/s blended because the fast prefill dominates the denominator; long
generations converge to the pure decode rate (~86 t/s at ~400 KV).

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


## Decode Gap: Per-Dispatch Encode Cost (~24µs vs llama ~7µs) + KV Growth (2.2×) + F16 KV Cache

For 1‑token decode the matmul kernels are memory‑bound and the scalar
dot‑product loop is less of a bottleneck. The dominant delta comes from
three sources:

> **SUPERSEDED (2026-08-03, A1 dead-end):** the "per-dispatch encode cost
> ~24µs/kernel" premise below does NOT hold for the current code. Measuring the
> A1 parallel-encoding prototype with `MINFER_TIMING` showed the encode is only
> **~1 ms/step** (4 threads × 6 layers), and splitting the 24 layers across 4
> command buffers **regressed** decode (120-token gen: 1.67/1.08/1.43 s,
> nondeterministic, vs serial 0.93 s stable) — each extra command buffer adds
> GPU launch overhead, and the encode was already hidden behind GPU execution.
> The decode step is **GPU execution-bound** (~7 ms/token for Q4_0 on M4 Pro):
> 24 small per-layer kernels, attention re-reading a growing KV. A1 was reverted
> (2026-08-03); the next lever is GPU-side kernel fusion / fewer dispatches
> (e.g. fuse QKV projection into one kernel, larger threadgroups), NOT parallel
> command buffers. The `retain`/`release` fix for the metal crate's
> autoreleased cb/encoder objects (required for any cross-thread cb) is kept.

### 0. KV-cache-growth degradation (2.2× at ~400 KV — the biggest long-context factor)

Measured 2026-08-01 (Q4_0, pure decode):

| Context (KV length) | minfer | llama.cpp |
|---|---|---|
| Short (~40) | ~187 t/s | ~375 t/s |
| Long (~400) | ~86 t/s | ~279 t/s |
| Degradation | **2.2×** | 1.34× |

minfer's `kernel_gqa_attn_f32` re-reads the **entire KV cache** every decode
step. The default cache is **F32 (4 bytes/elem)** vs llama.cpp's **F16 (2 bytes)**.
llama.cpp degrades far less (its F16 cache + tighter attention inner loop).
This is why short "hello"-style generations look fast (~340 t/s blended) while
long generations collapse to ~86 t/s.

**Measured 2026-08-01**: switching to an F16 KV cache (`MINFER_CACHE_TYPE=f16`)
did **not** recover the degradation for the 0.5B model — decode stayed
~dispatch-latency-bound (attention ≈ 5% of decode work; the KV read is ≈ 0.5%
of the M4 Pro's ~200 GB/s bandwidth, so halving it saves ~0.5%, while the
 f16→f32 conversion in the attention kernel adds per-element overhead). F16 is
kept **opt-in** (matches llama.cpp's default) for larger models / very long
contexts where attention bandwidth actually dominates. **Note (2026-08-03)**: the
KV-parallel split attention now has a f16 partial variant (`kernel_gqa_attn_partial_f16`),
so `MINFER_CACHE_TYPE=f16` no longer falls back to the classic single-pass kernel
(200-token decode 1.60 → 0.95 s). The dominant long-context decode cost was the
per-token KV buffer reallocation (fixed 2026-08-03 via geometric growth), not KV
bandwidth.

### 0a. Per-dispatch encoding overhead — the REAL decode bottleneck (2026-08-01)

Measured dispatch counts for the same model (Qwen2.5-0.5B, flash_attn on):

| | llama.cpp | minfer |
|---|---|---|
| ggml graph nodes | **822** (runtime log) | — |
| actual Metal compute dispatches | **~490–530** (822 minus ~300 no-op view/permute/reshape nodes; K-RoPE fused into the KV store) | **~484** (`layer_gpu`: 20/layer × 24 + output 3 + embed 1) |
| decode time / step | ~3.6 ms | ~11.6 ms |
| **per-dispatch cost** | **~7 µs** | **~24 µs** |

**The dispatch COUNT is nearly identical — the gap is the per-dispatch cost.**

llama.cpp keeps per-dispatch overhead ~3× lower via:
1. **Multi-command-buffer parallel encoding** (`ggml-metal-context.m`): `n` extra
   threads encode disjoint node ranges into `n+1` command buffers concurrently,
   overlapping CPU encoding with GPU execution. minfer uses ONE command buffer,
   so all ~484 encodes are on the critical path and the semaphore wait blocks
   each decode step.
2. **Optimized encoding**: op params packed into a struct and set with a single
   `setBytes`; pipeline/buffer state reused across nodes. minfer issues
   `set_buffer`×3 + `set_bytes`×3 + `set_threadgroup_memory_length` +
   `dispatch_thread_groups` per kernel.
3. **K-RoPE fused into the KV cache store** (1 fewer dispatch/layer).

This corrects the earlier conclusion that "484 dispatches is the bottleneck" —
the count is not the problem; the per-dispatch encode cost is.

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

### 2. F32 KV cache vs llama.cpp's F16 — implemented as opt-in, measured no gain

minfer's default KV cache is `float` (4 bytes per element); llama.cpp uses
`half` (2 bytes). An F16 cache (`kernel_store_kv_f16` + `kernel_gqa_attn_f16`,
enabled via `MINFER_CACHE_TYPE=f16`) was implemented to halve attention/store‑kv
memory bandwidth. **Measured 2026-08-01**: ~3% **slower** than F32 on the 0.5B
model — decode is dispatch‑latency‑bound (attention ≈ 5% of decode work; the KV
read is ≈ 0.5% of ~200 GB/s bandwidth), so the bandwidth saving is negligible
and the f16→f32 conversion in the tile load adds per‑element overhead. F16
remains opt‑in for larger models / very long contexts where attention bandwidth
dominates.

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
| **GPU-hang safety fixes** | After a 2026-08-02 GPU hang (M4 Pro, AGXG16X) that froze the machine, three safety changes: (1) `submit()` now waits **bounded 10 s** (was `DISPATCH_TIME_FOREVER`), checks `MTLCommandBufferStatus`, and prints a **dispatch trace** (`MINFER_TRACE=1` records the last 16 op labels) — a GPU fault/hang now reports and exits instead of blocking forever (which previously let a single fault stall every Metal client incl. WindowServer → whole-machine freeze). (2) `kernel_gqa_attn_f32/f16` no longer `return` early for `h >= nh` before the `threadgroup_barrier` — a per-simdgroup early return deadlocks the GPU when `nh % nk != 0`; invalid heads now run the loop with a dummy index and skip the output write. (3) Runtime guards in `layer_gpu`/`output_norm_gpu`: `nh % nk == 0`, `hd <= 256` (the `acc[256]` array), `id % 32 == 0` — fall back to CPU instead of risking a GPU fault |
| **Decode bottleneck analysis (corrected 2026-08-03)** | The 2026-08-01 "per-dispatch encode cost ~24µs" premise was measured to be **wrong** for the current code: `MINFER_TIMING` shows the A1 parallel-encoding prototype encodes in **~1 ms/step**, and decode is **GPU-execution-bound** (~7 ms/token Q4_0). A1 (4 parallel command buffers) **regressed** decode (1.67/1.08/1.43 s vs serial 0.93 s, nondeterministic) and was reverted. The real lever is GPU-side kernel fusion / fewer dispatches, not parallel encoding |
| **Sampling pipeline rewrite** | `sampler.rs`: fixed `apply_top_p` double-softmax bug (excluded tokens now `-INFINITY`, logits not clobbered to probabilities), added `apply_repetition_penalty` (llama `repeat_penalty` on last 64 tokens), seeded `StdRng` (`--seed`, default 42), unified `sample()` entry. Defaults now match llama.cpp: `temp=0.8`, `top_p=0.95`, `repeat_penalty=1.1`. CLI flags `--temp/--greedy/--top-k/--top-p/--repeat-penalty/-n/--seed`. **Repeat penalty alone breaks the 0.5B model's greedy repetition loops** (previously hit the n_predict cap repeating; now stops at EOS). 7 new sampler unit tests |
| **F16 KV cache (opt-in, measured no gain)** | `MINFER_CACHE_TYPE=f16` (default f32): `kernel_store_kv_f16` + `kernel_gqa_attn_f16`, f16 allocation in `kv_ensure_layer`, f16→f32 in `sync_kv_to_cpu`. Correct (gen0 logits cos 0.9997), ~3% slower on 0.5B (dispatch-latency-bound) — kept opt-in for larger models. Attention isolation test now covers both `kernel_gqa_attn_f32` and `kernel_gqa_attn_f16` |
| Q5_1 Metal kernel | `kernel_q5_1_f32_matmul` + `_multi` (F32 activation path, same structure as Q5_0) |
| Q5_K Metal kernel | `kernel_q5_k_f32_matmul` + `_multi` (176B/256‑elem super‑block, `qh[p] bit s` indexing, unsigned formula) |
| Q5_K_M GPU verified | Qwen2.5‑0.5B Q5_K_M full GPU: prefill ~240 t/s, decode ~250 t/s |
| Per‑layer CPU fallback | When `layer_gpu` fails for a Q5_K layer, the engine submits partial GPU work, downloads the hidden state, and resumes the remaining layers on CPU |
| Q5_K formula + qh‑indexing fixes | Two independent bugs in minfer's CPU Q5_K implementation — signed formula (`dl·(u-16)-ml`) corrected to unsigned (`dl·u-ml`), and qh high‑bit indexing corrected from `qh[sub·4+pos/8] bit pos%8` to `qh[pos] bit sub` (matching llama.cpp's quantizer layout) |
| **P1: Q4_0 → f32 activation path** | Q4_0 no longer Q8_0‑quantizes activations; routed through the existing f32 matmul kernel (matching llama.cpp Metal). Removed 4 `quantize_q8_0` calls in `layer_gpu` + 1 in `output_norm_gpu`, deleted the dead `quantize_q8_0` method/pipeline/shader. Decode ~+5‑10 %, minimal prefill change |
| **GQA attention `simd_max` divergence fix** | `kernel_gqa_attn_f32` looped `for (j = tiisg; j < tile_sz; j += 32)` — lanes with `j >= tile_sz` exited early, so `simd_max(dot)` ran across divergent lanes with stale registers, corrupting the online-softmax running max for partial KV tiles (`nkv % 32 != 0`). Symptoms: coherent short replies but repetition loops on longer/37-token prefills, GPU prefill logits cos ≈ 0.83 vs CPU. **Fix**: uniform iteration count + `valid = j < tile_sz` mask (`dot = -INFINITY` for invalid lanes, `e = 0`). Result: prefill logits cos 0.83 → 0.999, gen0 0.96 → 0.999, the looping prompt now generates coherent text matching llama.cpp. Regression test `tests/gqa_attn_isolation.rs` |
| **Metal cb/encoder autorelease fix** | The `metal` crate returns **autoreleased** ObjC objects from `commandBuffer` / `newComputeCommandEncoder`. `cmd_buffer()` now explicitly `retain`s both and `MpsCommandBuffer::drop` releases them, so a cb created on any thread survives that thread's autorelease-pool drain (a background-thread cb previously got its encoder released without `endEncoding` → Metal assert + abort). Discovered while prototyping A1 parallel encoding; harmless for the serial path, required for any future threaded encoding |

## Remaining Optimization Opportunities

Ranked by estimated impact‑to‑effort ratio for the current workload
(Qwen2.5‑0.5B, Apple M4 Pro). **2026-08-01 update**: measured gaps are
prefill 3× (Q4_0) / 7× (Q4_K_M, Q5_K_M) vs llama.cpp and pure-decode 2× (short
context) / 3.2× (long context).

| Priority | Item | Impact | Effort | Notes |
|:---:|------|--------|--------|-------|
| **P0.5** | **Fused QKV + FFN gate/up matmuls (decode)** | **~5% decode (shipped 2026-08-03)** | Medium | Wq/Wk/Wv and ffn_gate/ffn_up are row-major-concatenated at load (`concat_rows` → `blk.{i}.attn_qkv`/`ffn_gu`) when types + input dim match. nt==1 decode runs ONE matmul/group (od=nqt+2·nkt, 2·nf) into `buf_bqkv`/`buf_bgu`; rope/store/swiglu read sections via `set_buffer` byte offsets. nt==1-only + exact-length `gpu_abort` guards. Byte-identical to separate path (`MINFER_NO_FUSE_QKV=1` A/B); layout locked in `gemm_isolation.rs::qkv_row_concat_layout`. Median 200-token decode 1.63→1.55 s (~5%) at a clean GPU state (the "~24%" figure was inflated by a GPU-state artifact). **Dispatch fusions are dead-ends** (store_kv_both + residual_rms_norm verified correct but no gain) |
| **P0.75** | **KV-parallel split attention (decode)** | **~32% decode (shipped 2026-08-03)** | High | attention was measured as the #1 decode bottleneck (~48%, grows with KV): for nt==1 `kernel_gqa_attn_f32` grids only (1, nk)=2 TGs looping the KV sequentially. New `kernel_gqa_attn_partial_f32` + `_f16` (grid (nt,nk,P), per-chunk online-softmax partials; f16 reads half K/V, converts to f32 float4 tiles) + shared `kernel_gqa_attn_combine_f32` (grid (nt,nh), merge pass, no shared mem/barriers). nt==1, both cache types. P adaptive `clamp(nkv/16,1,32)` (`MINFER_ATTN_CHUNKS`). Built in `gqa_attn_split_isolation` first (deterministic + cos 1.0 vs scalar at nkv up to 4097, f32 AND f16 variants), then A/B byte-identical to classic. Median 200-token decode 1.56→1.06 s (f32); f16 1.60→0.95 s |
| **P0.8** | **Attention float4 acc + adaptive chunks + KV geometric growth** | **~15% more decode + long-context (shipped 2026-08-03)** | Medium | float4-vectorized acc/dot/tile-loads in the partial kernel (scalar dynamic `acc[256]` was per-thread local memory); adaptive `n_chunks=clamp(nkv/16,1,32)`; `kv_ensure_layer` grows KV buffers ×2 instead of +1 row every decode token (was O(n²) CPU encode: 0.5ms@KV140 → 4.2ms@KV2510 → now 0.13ms). ⚠️ old_v clone typo polluted the V cache (Q4_K_M garbage) — A/B didn't catch it (both paths share the corrupted KV); caught against a known-good reference. Short 200-token decode 1.06→0.88 s; long-context (KV≈2510) 10.6→8 ms/token |
| ~~P0~~ | ~~GEMM kernel: simdgroup_mm~~ | — | — | **Investigated 2026-08-01: initially 2× slower and reverted, then re-transcribed (P1) with 3 bug fixes — now SHIPPED (see "Prefill Gap" section above): ~+11 % at 30 tokens, ~+34 % at 70 tokens for nt ≥ 16** |
| **P1** | **GEMM kernels for non‑Q4_0 quants** | **Q4_K_M ~300→~650, Q5_K_M ~330→~610, 1.5B Q4_K_M 48→442 t/s prefill (shipped 2026-08-03)** | Large | `kernel_q8_0_mm_f32`/`kernel_q5_0_mm_f32`/`kernel_q5_1_mm_f32` (32-elem blocks, drop-in Q4_0 GEMM + per-quant `dequant_*_16`) + `kernel_q4_k_mm_f32`/`kernel_q5_k_mm_f32`/`kernel_q6_k_mm_f32` (256-elem super-blocks, dequant il = (loop_k%256)/16 + il0). nt≥16 dispatch + 8 KB tg-memory guard. Faithful llama transcriptions (block_q6_K d at END; block_q5_K qh BEFORE qs). Isolation `non_q4_0_gemm_isolation` (deterministic, relerr<5e-3 vs CPU for all 6 + Q4_0) + A/B byte-identical vs MINFER_GEMM=0. 1.5B Q4_K_M (Q4_K/Q6_K weights) verified end-to-end ~9× prefill |
| **P2** | **KV cache: F32 → F16 (implemented, opt-in)** | ~0 for 0.5B; helps larger models / long context | Small | **Implemented 2026-08-01** as `MINFER_CACHE_TYPE=f16` (default f32). `kernel_store_kv_f16` + `kernel_gqa_attn_f16`. Measured F16 is ~3% **slower** on the 0.5B model — decode is dispatch-latency-bound, not KV-bandwidth-bound (attention ≈ 5% of decode work; KV read ≈ 0.5% of ~200 GB/s bandwidth). Kept opt-in for larger models where attention bandwidth dominates |
| **P3** | ~~Reduce per-dispatch encode cost + parallel command buffers~~ (A1 dead-end 2026-08-03) | — | — | ~~A1 (parallel command buffers)~~ **tested and REVERTED 2026-08-03**: encode is ~1 ms/step (not the ~24µs/kernel claim), so 4 parallel command buffers only added GPU launch overhead. A2 (packed `set_bytes`) shipped. The GPU-side fusions that replaced it are SHIPPED: fused QKV + FFN gate/up matmuls (P0.5, ~5% decode) and KV-parallel split attention (P0.75, ~32% decode, incl. the f16 partial variant) |
| ~~P4~~ | ~~Matmul + bias fusion~~ | — | — | Dead-end 2026-08-03: Qwen models have no attention biases, and dispatch-count reductions don't move decode (see P5) |
| ~~P5~~ | ~~Residual add + RMSNorm fusion~~ | — | — | **Dead-end 2026-08-03**: `residual_rms_norm` was implemented, verified correct, and REVERTED — no measured gain (1.79 vs 1.74 s). Dispatch-count reductions don't move decode |
| P6 | Element‑wise float4 vectorisation | 1‑2% | Small | `add_f32`, `mul_f32`, `silu_f32` still use scalar loads |
| P7 | RoPE parallelisation | ~1% | Small | Currently 1 thread per (token, head) |
| ~~P8~~ | ~~RoPE + store_kv fusion~~ | — | — | Dead-end 2026-08-03: same dispatch-reduction class as store_kv_both (reverted, no gain) |
