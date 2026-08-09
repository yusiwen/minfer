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

> **Metric warning (updated 2026-08-06)**: llama.cpp's `Generation:` is **pure**
> decode (only generation time in the denominator). minfer's `Generated:` is now
> ALSO pure decode (`generated / gen_time`) — aligned with llama.cpp; minfer's
> `Total:` line keeps the previous **blended** rate
> `(prompt + generated) / total_time` for comparison. All rates below are pure
> decode / pure prefill.

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

## Decode bottleneck is the CPU sampler, not the GPU (2026-08-06)

A side-by-side benchmark of the **same** Qwen2.5-0.5B-Instruct Q4_K_M GGUF
(`"Tell me about Transformer architecture."`, chat template → 35 prompt tokens)
on the M4 Pro showed the reported decode gap vs llama.cpp is mostly **CPU-side
sampling**, not GPU:

| Source | Config | Generated | Total time | Pure decode |
|---|---|---|---|---|
| llama.cpp (`Generation:`) | default sampling | ~330 (EOS) | — | **247.2 tok/s** |
| minfer | default sampling (top_k=40/top_p=0.95/temp=0.8) | 512 | 6.64-7.31 s | ~77-93 tok/s |
| minfer | `--greedy` (temp=0, GPU only) | 512 | 3.02 s | **~172 tok/s** |

The minfer rates above are the OLD **blended** caliber (`(prompt+gen)/total`,
pre-fix 2026-08-06); the pure-decode rate is higher — e.g. ~5.0-5.8 ms/token
(~180-208 tok/s greedy), and the default sampler adds ~7.6 ms/token on top.

### Per-token breakdown (identical `-n 128`, avg KV≈100)

| Sampler config | ms/token | delta vs greedy |
|---|---|---|
| greedy (temp=0) | 5.1 | — |
| + temperature (no top_k/top_p) | 5.9 | +0.8 |
| + top_k=40 | 12.6 | **+6.7** |
| + top_p=0.95 | 14.8 | **+2.2** |

The greedy-vs-sampled gap is reproducible across repeated runs (greedy
1.33-2.40 s vs sampled 6.79-7.31 s for 512 tokens) — **not** a GPU-state
artifact (unlike the earlier bimodal 0.80/1.15 s decode variance).

### Root cause

`sampler.rs` runs a full-vocab O(n·log n) sort **per token** on the critical
path. The decode loop is strictly serial (sample → print → forward), so the CPU
sampling time does not overlap with the GPU forward:

- `apply_top_k` (`sampler.rs:45-57`): `logits.to_vec()` (607 KB copy) +
  `sort_by` over **all 151,936** logits every token.
- `apply_top_p` (`sampler.rs:63-87`): full softmax + sort of **151,936**
  `(usize, f32)` tuples (2.4 MB allocation per token) even though top_k already
  reduced the candidates to ≤40.

llama.cpp's sampler is a **candidate-list chain** (verified in llama.cpp
`src/llama-sampler.cpp` @ 88b47a755): `top_k` (`llama_sampler_top_k_impl`,
:321) does a `std::partial_sort` — **O(n·log k), NOT a full sort** — then
shrinks the candidate array to k (`cur_p->size = k`), and every later sampler
(`top_p` :1355, temperature, dist) operates on the ≤k survivors. Its only
full-vocab work per token is a sequential fill of the candidate array from the
logits plus that one partial sort, ~0.5-1 ms. (Note: llama-cli runs this CPU
chain via `common_sampler_sample`; the newer GPU backend samplers
`llama_set_sampler`/`backend_apply` in this version are used by llama-server /
speculative, not llama-cli.) This is why llama.cpp sustains 247 tok/s with
default sampling while minfer's GPU-only rate is already close (~200 tok/s
greedy).

### Implications

1. The 3.3× "decode gap" users see with default settings is mostly sampler CPU
   time, NOT GPU. With `--greedy`, minfer reaches ~180-208 tok/s vs llama's
   ~247 — a genuine GPU gap of ~1.2-1.4× (dispatch count + f32 activations,
   tracked separately in this document).
2. GPU decode is **flat at ~5-6 ms/token up to KV≈550** (greedy -n 64/128/256/512:
   5.2/5.1/6.0/5.8 ms/token) — the split attention hides KV growth at these
   lengths, confirming the earlier attention work paid off.

### Fix — SHIPPED 2026-08-06 (`sampler.rs`)

- `apply_top_k`: full-vocab `sort_by` (O(n·log n)) + 607 KB copy replaced with
  `select_nth_unstable_by` on a copy to extract the k-th largest threshold in
  O(n), then a value mask pass. (Runs on a copy because the in-place variant
  reorders the array, which would corrupt the index→token mapping; the original
  `logits` is only masked, never moved.)
- `apply_top_p`: collects the finite top_k survivors (≤40) and does softmax +
  stable descending sort on just those (O(k·log k)); falls back to the old
  full-array path only when > 1024 candidates survive (top_k disabled).
- `sample_temperature`: skips the exp() call for masked (-INF) logits (still
  yields exp(-INF)=0 exactly) and skips zero-probability tokens in the final
  cumulative scan.
- `main.rs`: decode loop now moves the single-token `forward()` logits Vec in
  place instead of `logits_all[..n_vocab].to_vec()` (607 KB copy/token).

**Measured** (Qwen2.5-0.5B Q4_K_M, M4 Pro, default sampling top_k=40/top_p=0.95):

| Length | Before | After | |
|---|---|---|---|
| -n 128 (seed 42) | 1.46 s | **0.72 s** | 2.0× |
| -n 256 | ~3.35 s | ~1.7 s | ~2.0× |
| -n 512 | 6.64-7.31 s | **3.49-3.55 s** | ~2.0× |

Per-token default-sampling decode: ~12.6-14.8 ms → **~5.5-6.5 ms** (~150-200
tok/s), now close to the greedy GPU-only rate (~5 ms/token). The fixed-seed
output is **byte-identical** to the old sampler (all 7 `sampler.rs` tests pass;
the 5 failing `avx2::test_q4k_dot_*` are the unrelated pre-existing x86 bug).
Residual sampled-vs-greedy gap (~0.5-1.5 ms/token, noisy under GPU-state
variance) is the remaining ~0.5-1 ms CPU sampling serialized in the loop.

## Decode Gap (revised 2026-08-06): matmuls at ~130 GB/s — structural for nt==1

**Precise gap location** (Q4_K_M 0.5B, greedy decode, avg KV≈160, M4 Pro).
Subtractive profile via `MINFER_SKIP_ATTN/MATMULS/SMALL=1` env gates
(decode-only; centralized in `metal.rs::DecodeSkips`, OnceLock-cached env read
like MINFER_TRACE, so normal decode has ~zero overhead — each dispatch is gated
in its exact original position). Kept in `layer_gpu`/`output_norm_gpu`/
`forward.rs` for future profiling:

| Component | ms/token | % | Evidence |
|---|---|---|---|
| matmuls (QKV/O/GU/down/output, ~97 kernels) | **3.2** | 59% | full − skip_matmul |
| small element-wise (~216 kernels: norm/rope/bias/add/swiglu/store) | 0.9 | 17% | full − skip_small |
| attention (partial+combine, ~48 kernels) | 0.7 | 13% | full − skip_attn; **flat** vs KV (split attention works) |
| base infra (cb submit + sync + 607 KB logits download + encode) | 0.86 | 16% | all three skips |
| **full** | **5.4** | | |

**Root cause (final 2026-08-06): matmuls are the bottleneck at ~130 GB/s —
structural for nt==1, NOT fixable by dequant vectorization.** The matmul weight
sweep is **~392 MB/token** (Q5_0 173 + Q8_0 146 + Q6_K 43 + Q4_K 29 MB, parsed
from GGUF) → memory floor ~1.96 ms at ~200 GB/s. Measured **matmul-only time is
~3.0 ms/token ≈ 130 GB/s** (isolated via `MINFER_SKIP_ATTN + MINFER_SKIP_SMALL`,
base subtracted). The 436-kernel launch-overhead model was WRONG (a controlled
fusion showed tiny kernels cost ~2-3 µs each). The Q4_0 model decodes ~1.4×
faster than Q4_K_M (0.98 s vs 1.36 s / 256 tokens) — but that's mostly the
smaller Q4_0 weight sweep (~250 MB vs 392 MB), not a kernel-speed difference.

This **revises** the session's earlier conclusions:
1. "Matmul dequant inner loop is THE lever (~1.7 ms potential)" (2026-08-06,
   earlier) — vectorizing the Q5_0 kernel (the last scalar one) gained only
   **~0.08 ms (~2.6 % matmul)**, so the matmuls are NOT dequant-compute-bound;
   the ~130 GB/s is nt==1 small-grid launch/latency + ALU + occupancy.
2. "Per-kernel launch overhead ~10-15 µs is THE lever" (2026-08-06, earlier) —
   Phase 1 (bias+RoPE+store 7→1, 144 kernels) measured ~0.27 ms (~2 µs/kernel).

### Phase 1 — SHIPPED 2026-08-06: fused bias+RoPE+KV-store (7 → 1 kernel)

`kernel_attn_bias_rope_store` (metal.metal) + `attn_bias_rope_store` dispatch
+ `pl_attn_bsr` pipeline. Replaces add_bias×3 + rope×2 + store_kv×2 for nt==1
decode (fused QKV path), handling f32 and f16 KV caches. Gated on all three
biases present + even hd; `MINFER_NO_FUSE_BSR=1` falls back to the 7 kernels.
Verified **byte-identical** (f32 AND f16 caches, fixed seed). Measured: 1.420 →
**1.350 s median** (-n 256, greedy) = **~0.27 ms/token (~5 %)**.

### Q5_0 vectorized dot — SHIPPED 2026-08-06, small gain (the last scalar kernel)

The Q5_0 matmul was the **only** scalar dequant kernel left (all others — Q8_0 is
a llama `kernel_mul_mv_q8_0_f32_impl` translation, Q4_0/Q5_1 use the
interleaved-ushort dot). `kernel_q5_0_f32_matmul`/`_multi` now use the existing
`block_q5_0_dot_y` (ushort nibble + qh-bit trick, llama `block_q_n_dot_y` port),
matching the Q5_1 kernel structure. **Measured (MPS-asserted, corrected
methodology)**: matmul-only 3.09 → **3.05 ms/token (~1-2 %)** — a negligible
gain, no decode-level difference beyond noise. **Conclusion: the matmuls are NOT
primarily dequant-compute-bound** — the vectorization (the entire ALU-difference
between scalar and vectorized dequant) bought only ~0.04-0.08 ms, so the ~130
GB/s is a mix of nt==1 small-grid launch latency + ALU + occupancy, not fixable
by more dequant micro-opts.
> **Verification lesson (2026-08-06)**: this kernel initially LOOKED like a GPU
> throttle (~7 tok/s) — the duplicate `block_q5_0_dot_y` definition failed the
> Metal shader compile, MPS fell back to CPU silently, and the CPU time was
> misread as a hot GPU. Guard now: `tests` → `metal.rs::tests::metal_pipelines_compile`
> compiles every pipeline at `cargo test` (catches shader errors), and
> `scripts/bench.sh` asserts `MPS: GPU acceleration enabled` (and prefill ≥ 200
> tok/s in `--health`) before trusting any timing — the CPU-fallback signature is
> ~7 tok/s vs the healthy ~500+ tok/s prefill.

### Matmul bandwidth decomposition (2026-08-06 #1) — no per-matmul lever found

`metal.rs::tests::matmul_bandwidth_profile` batches the SAME nt==1 matmul dozens
of times in one command buffer and reports GB/s. **Warm results** (Qwen2.5-0.5B
dims): output q8_0 **~200-230 GB/s (bandwidth-bound ✓)**, GU ~170-250, QKV/O/down
~60-116. Small-od matmuls are ~2× below the floor — structural (small grids
can't saturate DRAM), NOT a fixable single kernel. Q5_0 is only ~1.3× slower
than Q4_0/Q5_1 at equal dims (the 5-bit format costs more ALU), so there is no
"slow quant" to optimize.
> **Measurement trap (2026-08-06)**: the FIRST timed run of each kernel was
> initially read as "Q5_0 is 5× slower than Q4_0" — a **cold-start/GPU-clock-ramp
> artifact** (the same kernel measured later: 23 → 107 GB/s). The test now warms
> each kernel and measures twice, reporting the warm value. Batched-cb numbers
> still vary ~2× run-to-run (60 vs 107 GB/s for the same kernel) — treat them as
> relative, never absolute.

### Corrected matmul GB/s and the residual plan

The actual matmul weight sweep is **~392 MB/token** (Q5_0 173 + Q8_0 146 + Q6_K
43 + Q4_K 29 MB, parsed from GGUF) — the earlier "278 MB / ~90 GB/s" was wrong.
Matmul-only is **3.05 ms/token ≈ 129 GB/s** (63 % of the ~200 GB/s floor;
bandwidth-bound would be ~1.96 ms). The ~1 ms to the floor is structural:
nt==1 single-token matmuls with small grids can't fully saturate DRAM bandwidth,
and the dequant ALU adds ~7 ops/byte even vectorized. **Dequant vectorization
has diminishing returns** (Q5_0: +2.6 % for a full kernel rewrite). Remaining
low-value options: Q6_K/Q4_K vectorization (~0.1-0.2 ms combined, same class as
Q5_0), per-kernel encode optimization (pack setBytes, ~0.1-0.2 ms CPU-side),
small-kernel fusions (residual+norm, swiglu epilogue — ~0.07-0.25 ms each).
Realistic decode ceiling with all of them: ~4.8-5.0 ms/token vs llama's ~4.05.

### llama.cpp per-op comparison (2026-08-06 #3) — no hidden matmul lever

This llama.cpp version (88b47a755) removed the old `ggml_perf` per-op table;
`--perf`/`-v` only exposes the aggregate graph stats. What IS verifiable:
- llama's Qwen2 decode graph = **822 nodes** (≈490-530 actual Metal dispatches after
  ~300 view/permute no-ops) — comparable to minfer's ~436.
- **Flash Attention is enabled** — llama runs ONE fused flash-attn kernel per
  layer; minfer's split attention is 2 kernels (partial + combine).
- llama eval ≈ 3.66 ms/token on this run.
- The common-quant matmul kernels (`kernel_q5_0_*`, `kernel_q8_0_*`, the
  `block_q*_dot_y` functions) are **line-for-line llama translations** — there is
  no faster llama matmul kernel to copy. (The newer `kernel_mul_mv_ext_*` kernels
  are selected only for nt=2-8 small batches, NOT single-token decode.)

### Fair A/B + wall-clock decomposition (2026-08-06 #4) — the gap is 100 % GPU

Same-session interleaved A/B (llama `-v --single-turn --perf` `predicted_per_token_ms`
vs minfer generation-only, -n 128, default sampling, 6 rounds, medians):
**llama 3.51 ms/token vs minfer 5.16 ms/token = 1.47× (minfer ~68 %)**. The earlier
"80-85 %" was a flawed cross-session comparison (minfer greedy GPU-only vs llama
full-sampling); the user's cross-session 1.9× included GPU-state noise.

`MINFER_TIMING=1` (added to main.rs/forward.rs, env-gated) decomposes minfer's
per-token wall-clock:

| Component | ms/token |
|---|---|
| CPU encode (all ~436 dispatches) | **0.13** |
| **GPU execution (submit-wait)** | **~4.3-4.6** |
| logits download (607 KB) | 0.08 |
| CPU sampling (default) | 0.43 |

**The entire gap vs llama is GPU execution** (minfer ~4.5 ms vs llama ~3.1 ms =
per-dispatch 10.3 µs vs 6.2 µs, with identical matmul kernels + comparable
dispatch count). The CPU-side hypotheses are DEAD: the per-dispatch encode is
~0.3 µs (0.13 ms total — "pack setBytes / parallel encode" is ~worthless), and
the sampler fix already cut sampling to 0.43 ms.

### What "structural" means — verified vs inferred (2026-08-06 #5)

"Structural" is an **inference, not a proven architectural inferiority**. What is
strictly VERIFIED:

1. The matmul kernel **source** is line-for-line identical (nt==1
   `mul_vec_q_n_f32_impl` / `block_q*_dot_y` translations).
2. Dispatch **count** is comparable (~436 vs ~490-530).
3. Per-dispatch GPU time differs (10.3 µs vs 6.2 µs).

Hypotheses tested and CLOSED (2026-08-06 #6, follow-up executed):

1. **Dispatch parameters — DISPROVEN.** llama's `mul_mv` (nt==1) pipeline config
   (`ggml-metal-impl.h` N_R0/N_SG) for Q5_0 = 4/2, Q8_0 = 2/4, Q6_K = 2/2,
   Q4_K = 2/2, plus the Q8_0 special grid (ne01/nr0, simdgroups cooperate) — ALL
   match minfer's `quant_matmul_f32_on_gpu_buf` exactly. Same kernel source +
   same dispatch config ⇒ the matmuls execute identically.
2. **Attention is NOT the main lever.** llama `-fa on` vs `-fa off`: 3.64 vs
   3.88 ms/token → flash attention saves llama only ~0.25 ms, and even with
   flash OFF llama (3.88 ms) is far faster than minfer (5.16 ms). Fusing
   minfer's split-attention combine is worth at most ~0.2-0.4 ms.
3. **Multi-command-buffer — NOT a lever.** llama's multi-cb
   (`ggml_metal_graph_compute`: n_main + n_cb threads) exists to HIDE CPU
   encoding under GPU execution — irrelevant to minfer (encode is 0.13 ms), and
   there is no cross-cb GPU overlap (same queue). `MINFER_SPLIT_CB=N` re-test
   (corrected methodology, MPS-asserted) REGRESSES linearly: single 0.67 s /
   split2 0.93 / split4 1.23 / split8 1.62 s (-n 128 greedy) — each extra cb
   adds ~0.13-0.25 s of submit+sync.

### Where the GPU gap actually is (conclusion)

With kernel source AND dispatch params matching llama, and multi-cb/attention
ruled out as the main cause, the ~1 ms GPU gap is in the per-kernel GPU
execution efficiency of the ~436 serial kernels in ONE command buffer. The
remaining candidate is the small elementwise/norm kernels (minfer is f32
everywhere; llama may process intermediates as f16, halving their traffic) and
the fundamental MPS serialization of a single large cb — a genuine structural
difference, not a micro-optimizable one. `MINFER_SPLIT_CB` is kept as an
env-gated debug tool (default off).

### Architecture-level optimization plan (2026-08-06 #7)

Since matmul kernel source AND dispatch params match llama exactly, the matmul
portion (~3 ms) is at llama's level; the ~1 ms gap is concentrated in the
**non-matmul kernels** (attention ~0.7 ms + small elementwise ~0.5 ms +
overhead). The per-dispatch "10.3 µs" is only an average — the real per-kernel
GPU distribution is unknown. The plan is measurement-driven (per this session's
lessons — every inference so far has been wrong without a real measurement):

**Step 0 (required): real per-kernel GPU timeline via `xctrace`.**
`xctrace record --template 'Metal System Trace' --launch ./target/release/minfer …`
for BOTH minfer and llama (same workload), then `xctrace export` to parse each
kernel's actual GPU time. This decides which lever matters (attention-dominated?
small-op-dominated? uniform launch overhead?). It also settles whether minfer's
subtractive "matmul-only 3.05 ms" overestimates the real matmul cost.

**Step 1 — candidate architecture-level changes (pick per the trace):**

| Lever | If the trace shows… | Change | Trade-off / risk |
|---|---|---|---|
| **1. Fused single-kernel flash attention** | attention partial/combine dominate non-matmul time | Port llama's `kernel_flash_attn_ext_blk` (ONE kernel/layer with KV-parallel tiles, f16 KV, simdgroup-optimized) replacing minfer's partial+combine (2→1 kernel/层, −24 kernels) | Keeps the KV-parallel structure the split was built for (no long-context regression). Moderate-high risk (new kernel + isolation). ~0.2-0.4 ms |
| **2. Faithful non-blocking multi-cb** | uniform per-kernel launch/serialization overhead | Split decode into N cbs, commit ALL without waiting (llama's pattern), wait once. The 2026-08-06 `MINFER_SPLIT_CB` test was BLOCKING (each submit waits) — NOT a faithful test | Low risk to test. Uncertain (sequential dependency → GPU still runs cbs in order) |
| **3. f16 intermediate activations** | small-op traffic dominates | Halve elementwise/attention tile traffic (f16 buffers + kernels) | Weak prior: llama's graph activations are ALSO f32 (only K/V → f16). Huge rewrite |
| **4. Accept the architecture floor** | uniform MPS serialization, no dominant component | Document ~68 % (1.47×) as minfer's Metal decode level | — |

**Step 2: implement + verify** per the established methodology (byte-identical
A/B, `scripts/bench.sh` MPS assertion, isolation tests).

**Honest expectation:** uncertain 0.2-1.0 ms; Step 0's trace converts the
"structural" inference into per-kernel fact before committing to a rewrite.

### Step 0 result (2026-08-06) — per-category GPU decomposition

**Reliable per-category GPU times** (`MINFER_TIMING` gpu(submit-wait) +
`DecodeSkips`, greedy, min of runs, -n 64):

| Category | ms/token | % |
|---|---|---|
| matmuls | **2.99** | 72 % |
| attention | 0.54 | 13 % |
| small elementwise | 0.52 | 12 % |
| **full GPU** | **4.18** | |

**matmuls (2.99 ms, 72 %) are identical to llama** (kernel source + dispatch
params verified) — the ~1 ms gap vs llama is in the **non-matmul kernels**
(attention 0.54 + small 0.52 = 1.06 ms vs llama's ~0.5 ms) plus kernel
serialization. This REFINES the target: attention fusion and small-op
efficiency, not the matmuls.

**xctrace limitation (recorded honestly):** the `Metal System Trace` template's
CLI export does NOT expose full per-kernel durations — `metal-gpu-execution-points`
underestimates kernel time (~0.28 ms/token vs the real 4.18 ms) and the shader
profiler intervals were not captured. The aggregate GPU-work for the same
workload was comparable (minfer 24.1 ms vs llama 24.0 ms total), consistent with
"same GPU work, gap is launch/idle/serialization", but per-kernel precision is
not obtainable from the CLI alone (needs the Xcode GUI on the .trace).

### Step 1 result (2026-08-06) — naive fused attention confirmed SLOWER

Before committing to the llama `kernel_flash_attn_ext_impl` port, the naive
"1 kernel/层" alternative was measured (minfer already has the classic
single-pass `kernel_gqa_attn_f32` via `MINFER_NO_SPLIT_ATTN=1`):

| Attention design | GPU/token (min, short KV) |
|---|---|
| split (partial+combine, 2 kernels/层) | **4.15 ms** |
| classic single-pass (1 kernel/层) | 4.80 ms (+0.65 ms) |

**Conclusion: the split-attention design (2026-08-03) is correct — a naive
1-kernel fusion is SLOWER.** llama's fused flash attention is fast because of its
`simdgroup_matrix` design and function constants, NOT because it is one kernel.
The only way to get a fast 1-kernel attention is a faithful port of
`kernel_flash_attn_ext_impl` (~600 lines, simdgroup matrices, per-shape function
constants) — a multi-day, high-risk effort for an expected ~0.3 ms (out of the
~1.4 ms gap). **Net assessment: no low-risk path remains to close the GPU gap in
this architecture; the remaining candidates (full flash port, f16 small ops) are
high-effort with modest/uncertain return.**

### Step 1 — Phase A validation gate result (2026-08-06): flash port is a dead-end

The full llama `kernel_flash_attn_ext_vec` port was gated on a Phase A experiment
(per the session's measure-first discipline). Verified:
1. **f16 KV does NOT help attention** — `MINFER_CACHE_TYPE=f16` attention GPU
   0.54 ms vs f32 0.52 ms (the existing `kernel_gqa_attn_partial_f16`; the f16
   cache is the core of llama's flash advantage, but minfer's 0.5B attention is
   not KV-read-bound — consistent with the 2026-08-03 f16 wall-clock finding).
2. **Chunk tuning is already optimal** — adaptive (`nkv/16`) gives 0.51 ms;
   chunks=8 is 0.52 ms, over-parallelizing regresses badly (chunks=32 → 1.1 ms,
   chunks=64 → 1.9 ms).
3. **Attention scales sub-linearly with KV** (0.51 → 0.56 ms for KV 163 → 291) —
   not KV-bandwidth-bound, so halving KV bytes (f16) and even the float4/stride
   structure have little headroom.
4. minfer's split attention already uses float4 acc + adaptive chunks +
   KV-parallel two-pass (structurally what llama's vec kernel does).

**Phase A verdict: STOP the flash port.** llama's decode flash is faster because
of its overall Metal backend maturity (f16 KV as the DEFAULT, simdgroup-heavy
prefill, multi-cb scheduling), not because the decode attention kernel alone is
transformative for this model. minfer's split attention is at its design limit
(~0.5 ms, 13 % of the 4.2 ms GPU).

### Final gap report (2026-08-06) — where the GPU gap actually is

**Verified chain:**
1. Same-session interleaved A/B: llama ~3.6-4.2 ms vs minfer ~5.1-5.2 ms per token
   at -n 64-256 (**1.2-1.4×**), widening to **1.7×** at -n 512 (minfer 6.7 vs llama
   3.9 ms).
2. minfer's pure GPU (`MINFER_TIMING` submit-wait) exceeds llama's TOTAL wall —
   the gap is 100 % GPU-side.
3. minfer's AVERAGE decode grows with context (5.05 → 6.7 ms at -n 64 → 512);
   llama's stays ~flat (~3.6-4.2 ms) — the KV-length attention-scaling component.

**Precise decomposition (short context -n 32, min of runs):**

| Component | minfer GPU | llama GPU | gap |
|---|---|---|---|
| matmuls | ~3.0 ms (bandwidth-bound, identical source+params) | ~3.0 ms | **~0** |
| **non-matmul** (attention + small elementwise + per-kernel serialization) | **~1.2 ms** (`no_matmul` config: full 4.18 − 3.0) | **~0.3 ms** (inferred) | **~0.9 ms** |

**The structural gap is 100 % in the NON-MATMUL kernels** — minfer's ~340 small
kernels (attention partial/combine, rms_norm×2, add×2, rope, store, swiglu) run
at ~1.2 ms vs llama's ~0.3 ms (~4×). The matmuls (the ~72 % "body" of the GPU)
are identical and contribute ~0 gap. This is the precise pinpoint of what was
previously described vaguely as "attention + small + serialization": the gap is
the small-kernel tail (f32 elementwise + ~340 single-cb serialized dispatches +
attention that grows with KV), NOT the matmuls.

**Honest uncertainty:** the ~1.2 ms non-matmul figure is reliable (the
`no_matmul` config, which isolates the non-matmul GPU without subtractive
noise). The attention-vs-small split within it has ±0.2 ms noise (subtractive
deltas), and llama's ~0.3 ms non-matmul is inferred (no per-op timing in this
llama version). The KV-growth component adds ~0.5 ms/token at -n 512.

**Bottom line:** ~0.9-1.0 ms structural GPU gap (plus ~0.5 ms KV-growth at long
context) = minfer's ~340 non-matmul kernels at ~4× llama's efficiency. This
reconciles the session's 7 closed hypotheses — no single fixable component
exists without an architecture-level rewrite (f16 small ops, flash attention,
fewer/serialized kernels); minfer's Metal decode is at this architecture's
practical limit.


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

> **2026-08-06 update**: the "dispatch fusions are dead-ends" rows below (P4/P5/P8
> and the P0.5 note) were concluded from measurements whose ±0.05 s noise hid
> ~0.1-0.25 ms gains. Phase 1 (bias+RoPE+store 7→1, SHIPPED 2026-08-06) measured
> **~0.27 ms/token (~5%)** and proved small fusions DO help, just much less than
> the launch-overhead model predicted. The authoritative gap analysis and next
> plan (optimize the Q4_K_M matmul dequant inner loop, ~1.7 ms potential) are in
> **"Decode Gap (revised 2026-08-06)"** above — the table below is the historical
> 2026-08-01/03 record.

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
| **P6** | **Element‑wise float4 vectorisation** | ~2% decode (shipped 2026-08-03) | Small | `kernel_add_f32`/`kernel_mul_f32`/`kernel_silu_f32`/`kernel_swiglu_f32`/`kernel_add_bias_f32` now process 4 elements/thread (float4) with a scalar tail; dispatch thread count ÷4. 200-token decode ~0.88 → ~0.80 s (fast GPU state) |
| **P7** | **RoPE parallelisation** | ~1% decode (shipped 2026-08-03) | Small | One thread per (dim, head, token) instead of one per (token, head); dispatch (half_dim, n_head, nt). Recomputes pow/cos/sin per dim but parallelizes the previously 14-thread Q rope |
| ~~P8~~ | ~~RoPE + store_kv fusion~~ | — | — | Dead-end 2026-08-03: same dispatch-reduction class as store_kv_both (reverted, no gain) |

## Follow-up Work (2026-08-03)

Current state: decode 0.87 s / 200 tokens (~230 t/s, Q4_0), prefill uses simdgroup
GEMMs for every quant type, GPU-safety audit fully closed (H1/H2/M1/M2 guarded).

| Priority | Item | Type | Notes |
|---|---|---|---|
| 1 | **Qwen2.5-7B Q4_K_M verification** | verification | **DONE 2026-08-03**: Split-GGUF support (multi-part `-0000X-of-0000Y.gguf`) SHIPPED, aligned with llama.cpp — `GgufModel`/`load_gguf_model` parse every part (each lists its own tensors), loader builds a merged tensor index reading each tensor from its own part; `download hf:repo:q4_k_m` fetches all parts (resume-safe via expected-size check); `get_i64` extended to Uint16/Int16/Uint8/Int8/Bool (llama's `split.count`/`split.no` are Uint16); cached-name resolution dedupes split parts to part 0. Verified end-to-end: 7B loads as 2 parts (4466 MB), GPU "The capital of France is Paris." / "I am Qwen, a large language model developed by Alibaba" at ~28-31 tok/s, CPU 0.8 tok/s; isolation 6 + bin 16 (14+2 split) passed, all models correct |
| 2 | ~~Decode micro-opt: P6 + P7~~ | ~~perf~~ | **SHIPPED 2026-08-03**: float4 element-wise kernels + parallel RoPE (~2-3 % decode; 200-token ~0.88 → ~0.80 s) |
| 3 | ~~**Q4_K AVX2 CPU dot-product fix**~~ | ~~correctness (x86)~~ | **RESOLVED 2026-08-06**: was a stale TEST reference, not a bug — the 5 failing `test_q4k_dot_*` compared against the OLD 16-bytes-per-subblock Q4_K layout. The `dot_q4_k_q8_0` implementation is correct (4-chunk deinterleave, matches llama `dequantize_row_q4_K`) and there is no AVX2 Q4_K path (scalar only). Both test references (`reference_dot`, `independent_dot_q4k`) fixed → all 29 bin tests pass. x86 CPU users were never affected |
| ~~4~~ | ~~Q4_1 GEMM~~ | ~~completeness~~ | **SHIPPED 2026-08-03**: `kernel_q4_1_mm_f32` (20 B/32-elem, `dequant_q4_1_16` d*q+m) — every quant type now has a simdgroup GEMM; verified in `non_q4_0_gemm_isolation` (8 GEMMs) |
| 5 | ~~**CPU sampler: top_k/top_p full-vocab sort**~~ | ~~perf (CPU)~~ | **SHIPPED 2026-08-06**: default sampling was 2.5x slower than `--greedy` (+7.6 ms/token; GPU decode is only ~5 ms/token — the llama.cpp 247 vs minfer 80 tok/s gap was mostly sampler CPU time, see "Decode bottleneck is the CPU sampler, not the GPU" 2026-08-06). Fix: top_k → `select_nth_unstable_by` (O(n)); top_p → softmax+sort only the ≤k survivors; temp → skip masked logits; main.rs drops the 607 KB logits copy. Byte-identical fixed-seed output; default decode ~12.6-14.8 → ~5.5-6.5 ms/token (~2×, 512 tokens 6.9 → 3.5 s) |
| 6 | ~~**Q5_0 vectorized matmul dot (last scalar kernel)**~~ | ~~perf (GPU)~~ | **SHIPPED 2026-08-06**: `kernel_q5_0_f32_matmul`/`_multi` now use the existing `block_q5_0_dot_y` (ushort nibble + qh-bit, llama `block_q_n_dot_y` port), matching Q5_1/Q4_0. Correct (deterministic greedy output matches known-good; f32+f16). **Measured (MPS-asserted) matmul-only 3.09 → 3.05 ms/token (~1-2 %) — the matmuls are NOT dequant-compute-bound**; real sweep is ~392 MB/token ≈ 129 GB/s (63 % of floor). Bandwidth decomposition (2026-08-06 #1): output ~220 GB/s (bandwidth-bound), small-od matmuls ~60-116 (structural small-grid, no fixable lever); the earlier "Q5_0 is 5× slower" was a cold-start artifact. ⚠️ This kernel's shader-compile failure (duplicate `block_q5_0_dot_y`) initially read as a GPU throttle — now guarded by `metal_pipelines_compile` test + `scripts/bench.sh` MPS assertion. Remaining low-value: Q6_K/Q4_K vectorization, encode opt, small fusions → realistic ceiling ~4.8-5.0 ms/token vs llama ~4.05. See "Decode Gap (revised 2026-08-06)" |
| 7 | ~~**Fused bias+RoPE+KV-store (7 → 1 kernel)**~~ | ~~perf (GPU)~~ | **SHIPPED 2026-08-06** (Phase 1): `kernel_attn_bias_rope_store` replaces add_bias×3 + rope×2 + store_kv×2 for nt==1 decode; f32+f16 KV caches; gated on biases+even-hd; `MINFER_NO_FUSE_BSR=1` fallback. Byte-identical A/B (both caches). ~0.27 ms/token (~5 %) — proved tiny kernels cost ~2-3 µs, i.e. the launch-overhead model was wrong and the real bottleneck is the matmul dequant inner loop (row 6) |
| 8 | ~~**Truly locate the GPU gap**~~ | ~~perf (GPU)~~ | **CLOSED 2026-08-06 #6**: fair A/B (#4): llama 3.51 vs minfer 5.16 ms/token (1.47×); MINFER_TIMING: gap is 100 % GPU. Follow-up executed — (1) dispatch params DISPROVEN (llama N_R0/N_SG for Q5_0/Q8_0/Q6_K/Q4_K all match minfer), (2) attention ~0.25 ms only (llama `-fa on` 3.64 vs off 3.88 ms; even non-flash llama beats minfer), (3) multi-cb regresses (`MINFER_SPLIT_CB=N`: 0.67 → 0.93/1.23/1.62 s; llama's multi-cb is CPU-encode hiding). Conclusion: the ~1 ms GPU gap is per-kernel execution of ~436 serial kernels in one cb (small ops f32 vs llama f16 + MPS serialization) — genuine structural difference, not micro-optimizable. See "Decode Gap (revised 2026-08-06)" #6 |
| 9 | **Architecture-level GPU gap plan (xctrace → fused flash / multi-cb / f16)** | perf (GPU) | Plan #7 (2026-08-06): matmul source+params match llama ⇒ gap is non-matmul kernels (attention 0.7 + small 0.5 + overhead). Step 0: `xctrace` per-kernel GPU timeline for minfer AND llama (decides the lever). Step 1 candidates: (1) port llama `kernel_flash_attn_ext_blk` (single fused flash, KV-parallel, −24 kernels, ~0.2-0.4 ms), (2) faithful non-blocking multi-cb (previous test was blocking), (3) f16 intermediates (weak prior), (4) accept. Step 2: implement+verify. See "Decode Gap (revised 2026-08-06)" #7 |
