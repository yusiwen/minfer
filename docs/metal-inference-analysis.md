# minfer Apple Silicon Metal Inference Path — Analysis Report

> Baseline comparison: llama.cpp ggml-metal (~10,000+ lines) vs minfer metal.rs + metal.metal (~1,700 lines)
> Date: 2026-07-27

---

## Issue Overview

| # | Issue | Severity | Impact | Status |
|---|-------|----------|--------|------|
| 1 | No Flash Attention — brute-force GQA | CRITICAL | O(n²) complexity, long contexts unusable | ⬜ |
| 2 | No matrix-matrix multiply — token-by-token prefill | CRITICAL | Prefill bandwidth waste by factor of N | ⬜ |
| 3 | Duplicate KV Cache (CPU + GPU), no coherence | CRITICAL | 2× KV memory, GPU/CPU switch data loss | ⬜ |
| 4 | Single CommandBuffer, no encode/execute overlap | HIGH | GPU idle during CPU encoding | ⬜ |
| 5 | Weight data full-copy, no zero-copy mapping | HIGH | 2× weight memory | ⬜ |
| 6 | CPU-side Q8_0 quantization blocking pipeline | HIGH | CPU bottleneck + extra upload cost | ⬜ |
| 7 | No non-Mac GPU path (Vulkan/MoltenVK) | HIGH | Intel Mac unusable for GPU inference | ⬜ |
| 8 | RoPE limited to Qwen2 style, missing YaRN/interleaved | MEDIUM | Blocks other architectures on GPU | ⬜ |
| 9 | Unaligned `float4` in RMSNorm shader | MEDIUM | Potential perf/portability issues | ⬜ |
| 10 | Mutex on buffer grow hot path | MEDIUM | Micro-optimization | ⬜ |
| 11 | Attention scale hardcoded | MEDIUM | Some models broken | ⬜ |
| 12 | Embedding lookup on CPU | MEDIUM | Large-vocab perf degradation | ⬜ |
| 13 | Output layer type restriction | MEDIUM | Q4_1/Q8_0 output falls back to CPU | ⬜ |
| 14 | Pipeline compilation at startup | LOW | Startup latency | ⬜ |
| 15 | Weight HashMap + Mutex hot-path lookups | LOW | Unnecessary lock overhead | ⬜ |
| 16 | No GPU trace/debug support | LOW | Debugging difficulty | ⬜ |
| 17 | No MTLResidencySet support | LOW | macOS 15+ may evict GPU memory | ⬜ |
| 18 | KV cache `Mutex<Vec<Buffer>>` per-layer locking | LOW | Replace with RwLock | ⬜ |

---

## Detailed Analysis

### CRITICAL #1 — No Flash Attention

**Files**: `metal.metal:710-794`

The current `kernel_gqa_attn_f32` uses one threadgroup per (token, head) pair to sequentially iterate over all KV entries. At 4096-token context, each head loops 4096 times — no tiling, no shared-memory K/V reuse.

**Specific bugs**:
- `float4 oc[32]` (line 745) hardcodes `hd/4 ≤ 32`, meaning **hd > 128 causes out-of-bounds access** (Qwen2-72B hd=128 is right at the limit; larger models crash).
- `C = 64` batch size means only 64 KV entries per inner batch. On long contexts, the exp/max recalculation overhead grows exponentially.
- No shared memory for K/V — each thread independently loads K/V from global memory with zero reuse across iterations.

**llama.cpp approach**: Tiered flash attention (single tiled kernel `flash_attn_ext` and vector reduction kernel `flash_attn_ext_vec`) with shared memory tiling, block-mask precomputation, padding alignment, and multi-workgroup partial sum reduction.

### CRITICAL #2 — No Matrix-Matrix Multiply

**Files**: `metal.rs:163-178`, `metal.metal:13-82`

minfer only has mat-vec kernels (`NR0=2-4` rows per threadgroup). During prefill with N tokens, it dispatches N separate threadgroup columns via `dispatch_2d(..., nt, ...)`. Each token re-reads the same weight matrix independently, wasting memory bandwidth by a factor of N.

**llama.cpp approach**:
- `mul_mm`: `simdgroup_matrix` operations, 128×64 block, for large-batch prefill.
- `mul_mv_ext`: optimized for small batch sizes (2-8 tokens).

### CRITICAL #3 — Duplicate KV Cache

**Files**: `cache.rs` vs `metal.rs:57-58`

Two independent KV caches are maintained:
- CPU side: `KVCache.layers[i].k/v` (`Vec<f32>`, `cache.rs:6-11`)
- GPU side: `kv_k`/`kv_v` (`Vec<metal::Buffer>`, `metal.rs:57-58`)

The GPU path writes KV via `store_kv_f32`, but the CPU KV cache is never updated. If the inference falls back from GPU to CPU path, KV data is lost.

### HIGH #4 — Single CommandBuffer, No Parallelism

**Files**: `metal.rs`, `forward.rs:62-75`

All layers are encoded into one `MpsCommandBuffer`, then `submit()` blocks until completion. The GPU sits idle while the CPU encodes.

**llama.cpp approach**: `dispatch_apply(n_cb, ...)` encodes multiple command buffers across CPU cores concurrently — GPU starts executing the first CB while the remainder is encoded. Plus `MTLDispatchTypeConcurrent` with memory-range tracking for intra-CB concurrency.

### HIGH #5 — Weight Data Full-Copy

**Files**: `metal.rs:466-478`, `loader.rs:183`

Every weight tensor is copied into a new `MTLBuffer` via `copy_nonoverlapping`, even though Apple Silicon has a unified memory architecture.

**llama.cpp approach**: `newBufferWithBytesNoCopy` directly maps mmap'd model file pages, zero-copy.

### HIGH #6 — CPU-Side Quantization Blocking

**Files**: `metal.rs:598`, `kernel.rs:81`

Before each GPU matmul dispatch, `quantize_row_q8_0_buf` runs on the CPU to quantize activations to Q8_0, then uploads to GPU. This is a synchronous bottleneck in the GPU pipeline.

### HIGH #7 — No Non-Mac GPU Path

**Files**: `metal.rs:16-17`, `main.rs:17-18`

`MpsStateInner` is entirely wrapped in `#[cfg(target_os = "macos")]`. There is no Vulkan/MoltenVK fallback for Intel Macs.

### MEDIUM #8 — RoPE Limited to Qwen2 Style

**Files**: `metal.metal:662-687`

The RoPE shader hardcodes the Qwen2 non-interleaved layout. It does not support the interleaved format used by LLaMA/Mistral, nor YaRN extended-context RoPE.

### MEDIUM #9 — Unaligned float4 in RMSNorm

**Files**: `metal.metal:570-586`

Reads `float4` from arbitrary addresses at `x + row * d`. While Metal tolerates misaligned `float4` reads with a performance penalty, some devices may fault. If `d % 16 != 0`, the reads are unaligned.

---

## Remediation PLAN

Four phases, ordered by priority:

### Phase 1 — Infrastructure Fixes (1-2 days)

- [ ] **P1.1** Zero-copy weight mapping
  - Use `newBufferWithBytesNoCopy` + `MTLResourceOptions::StorageModeShared`
  - Remove `copy_nonoverlapping` weight copies
  - Related: #5

- [ ] **P1.2** Remove hot-path Mutex overhead
  - Weight HashMap: use immutable reference (populated once, never modified after loading)
  - KV buffer Vec: use `&[metal::Buffer]` instead of `Mutex<Vec<...>>`
  - Related: #10, #15, #18

- [ ] **P1.3** Fix hardcoded attention scale
  - Read attention scale from model hparams, pass as shader parameter
  - Related: #11

- [ ] **P1.4** Fix RMSNorm float4 alignment
  - Add `d % 4 != 0` fallback to element-wise path in the shader
  - Related: #9

### Phase 2 — Core Performance Fixes (3-5 days)

- [ ] **P2.1** Implement Flash Attention kernel
  - Reference: llama.cpp `flash_attn_ext` — tiled Q×K^T + online softmax + tiled V weighted sum
  - Shared memory K/V reuse
  - Padding alignment and block mask support
  - Related: #1

- [ ] **P2.2** GPU-side Q8_0 quantization in pipeline
  - The `kernel_quantize_q8_0` shader already exists but is unused in the full-layer forward path
  - Change: f32 activations → upload to GPU → quantize on GPU → feed directly to matmul
  - Eliminate CPU quantization bottleneck
  - Related: #6

- [ ] **P2.3** Multi-CommandBuffer parallel encoding
  - Split layers across multiple CBs using `dispatch_apply`
  - Implement `MTLDispatchTypeConcurrent` + memory barriers
  - Related: #4

- [ ] **P2.4** Unify KV Cache storage
  - Store KV only on GPU side (`MTLBuffer`), remove CPU-side `Vec<f32>` KV
  - When CPU path is needed, read back KV from GPU
  - Related: #3

### Phase 3 — Operator Enhancements (3-5 days)

- [ ] **P3.1** Implement matrix-matrix multiply kernel
  - Reference: llama.cpp `mul_mm` — simdgroup_matrix + shared memory tiling
  - Used for large-batch prefill scenario
  - Related: #2

- [ ] **P3.2** Support RoPE variants (interleaved + YaRN)
  - Abstract RoPE style enum (Qwen2 / LLaMA / Mistral / YaRN)
  - Related: #8

- [ ] **P3.3** GPU-side embedding lookup
  - Implement `kernel_get_rows` — lookup rows from quantized embedding table by token ID
  - Related: #12

- [ ] **P3.4** Fix output layer type restriction
  - `output_norm_gpu` to support Q4_1, Q8_0, Q4_K, Q6_K
  - Related: #13

### Phase 4 — Robustness & Ecosystem (2-3 days)

- [ ] **P4.1** Lazy pipeline compilation
  - Compile pipelines on first use instead of at startup
  - Related: #14

- [ ] **P4.2** MTLResidencySet support
  - macOS 15+ ResidencySet to keep GPU memory wired
  - Related: #17

- [ ] **P4.3** GPU trace support
  - Add `MINFER_METAL_CAPTURE` env var + `MTLCaptureManager` integration
  - Related: #16

- [ ] **P4.4** Vulkan/MoltenVK fallback path (optional)
  - GPU acceleration for non-Apple-Silicon devices
  - Related: #7

---

## Progress Tracking

| Task | Phase | Est. | Status | Done | Notes |
|------|-------|------|--------|------|-------|
| P1.1 Zero-copy weights | 1 | 0.5d | ⬜ | - | |
| P1.2 Remove hot-path Mutex | 1 | 0.5d | ⬜ | - | |
| P1.3 Fix attention scale | 1 | 0.5d | ⬜ | - | |
| P1.4 Fix RMSNorm alignment | 1 | 0.5d | ⬜ | - | |
| P2.1 Flash Attention | 2 | 2d | ⬜ | - | Largest single task |
| P2.2 GPU-side quantization | 2 | 1d | ⬜ | - | |
| P2.3 Multi-CB parallelism | 2 | 1d | ⬜ | - | |
| P2.4 Unify KV Cache | 2 | 1d | ⬜ | - | |
| P3.1 Mat-mul kernel | 3 | 2d | ⬜ | - | |
| P3.2 RoPE variants | 3 | 1d | ⬜ | - | |
| P3.3 GPU embedding | 3 | 1d | ⬜ | - | |
| P3.4 Fix output layer | 3 | 0.5d | ⬜ | - | |
| P4.1 Lazy pipelines | 4 | 0.5d | ⬜ | - | |
| P4.2 ResidencySet | 4 | 0.5d | ⬜ | - | |
| P4.3 GPU trace | 4 | 0.5d | ⬜ | - | |
| P4.4 Vulkan fallback | 4 | 2d | ⬜ | - | Optional |
