# Backends — the three execution engines

minfer runs one declarative compute graph (build → assign → fuse → allocate →
execute, `docs/GRAPH-REFACTOR-PLAN.md`) on three interchangeable backends:

| | **CPU** | **Metal** | **CUDA** |
|---|---|---|---|
| Executor | `src/graph/cpu_backend.rs` | `src/graph/metal_backend.rs` | `src/graph/cuda_backend.rs` |
| Device layer | — (std threads) | `src/metal.rs` + `src/metal.metal` shaders | `src/cuda.rs` + `src/cuda_kernels.cu` |
| Platform | any | macOS (Apple GPU) | NVIDIA, **opt-in** `--features cuda` |
| Assign priority | last (always answers) | first on macOS | second, when built in |
| Activations | quantized to Q8_0 (Q8_K for K-quant weights) | read as f32 | f32; int8 MMQ for prefill |
| KV cache | f32 regions | f32 or f16 (`MINFER_CACHE_TYPE=f16`) | f32 or f16 |
| Async model | synchronous | one `MpsCommandBuffer` per split | stream + CUDA Graph capture/replay |
| Deep dives | [walkthrough 10](./inference_e2e_walkthrough/10-cpu-matmul-kernels.md), [11](./inference_e2e_walkthrough/11-attention-vecops-kv.md), [CPU optimizations](./CPU_OPTIMIZATIONS.md) | [walkthrough 14](./inference_e2e_walkthrough/14-metal-backend.md), [Metal optimizations](./METAL_OPTIMIZATIONS.md) | [walkthrough 15](./inference_e2e_walkthrough/15-cuda-backend.md), [backend plan](./CUDA-BACKEND-PLAN.md), [campaign](./CUDA_OPTIMIZATION.md) |

This page is the overview: what the backend contract is, how nodes land on a
backend, and where the three differ. The linked pages carry the per-backend
detail.

## 1. The contract: the `Backend` trait

Every backend implements one trait (`src/graph/backend.rs:21-95`). The
scheduler knows nothing about Metal or CUDA specifics — it only talks to this
surface:

| Method | Meaning |
|---|---|
| `supports_op(op, dtype)` | capability query, asked **per node at build time** |
| `supports_fused(fused)` | gates the fusion pass — fused IR nodes are only produced where a kernel exists (`fusion.rs:68`, `:112`) |
| `alloc_buffer` / `free_buffer` | the backend's own buffer pool, sized in `f32` elements |
| `alloc_fresh` | same, but bypasses the recycle free list — split-boundary staging needs ids whose physical contents are still referenced later in the same execute (`backend.rs:36-42`) |
| `execute_node(node, in_bufs, out_buf, kv_pair)` | run one node; `kv_pair` carries the layer's persistent (K, V) region ids for KV ops; the output may alias an input (in-place ops) |
| `read_host` / `write_host` | host access to a pool buffer — direct slices on CPU, staged transfers on GPU |
| `synchronize` | wait for async work: CPU no-op; Metal submits the pending command buffer; CUDA closes a capture window if one is open |
| `graph_replay` *(CUDA only)* | replay a previously captured CUDA Graph for a split (`backend.rs:91-94`, feature-gated; default no-op) |

Two supporting traits/rules ride along:

- `KvProvider::kv_pair(layer)` — each layer owns **two persistent KV regions**
  (K and V) that live in the backend's pool and survive graph rebuilds
  (`backend.rs:12-19`; allocator detail in
  [walkthrough 07](./inference_e2e_walkthrough/07-allocator-liveness-kv.md)).
- **The allocator is the single owner.** Backends own pools, but buffers are
  only created through the allocator's liveness pass; input buffers are never
  freed, and in-place aliasing is decided at allocation time, not by kernels.

## 2. How a node gets its backend

Assignment happens **once, at build time** — never mid-run:

1. `GraphAllocator::supports(op, dtype)` walks the registered backends
   **highest priority first: Metal → CUDA → CPU** (`src/graph/alloc.rs:139-152`).
   The first backend whose `supports_op` answers `true` wins that node. The CPU
   backend supports the full op set, so it always terminates the walk.
2. **GPU participation is gated on weights**: a GPU backend only claims ops
   once *all* weight tensors are registered on it; the gate fails → the run
   aborts with the actual values, never a silent CPU fallback
   (`docs/GPU_SAFETY.md`).
3. Whether any GPU took part is recorded in `CParams.gpu`, which is part of
   the graph-reuse identity — a run that switches between GPU and CPU gets a
   different `GraphParams` and therefore a rebuilt graph
   ([walkthrough 13](./inference_e2e_walkthrough/13-decode-loop-graph-reuse.md)).

Because assignment is per *node*, one forward pass can mix backends. The
scheduler cuts the node list into **splits** — maximal runs of the same
backend — and at every split boundary it synchronizes the previous backend and
copies split inputs across (`copy_across`, a host round trip through shared
memory; `src/graph/scheduler.rs:176`). Metal encodes one
`MpsCommandBuffer` per split and submits it at `synchronize`. Mechanically
this is
[walkthrough 08](./inference_e2e_walkthrough/08-scheduler-execute.md) §3.

The error contract at execution time mirrors the build-time gate:
**kernel-invariant violations return `Err` from `execute_node` — never a
silent fallback to another backend** (e.g. a KV-store position ≥ `n_ctx`, an
attention head geometry mismatch, a device-limit shortfall). `docs/GPU_SAFETY.md`
is the binding rules page for the GPU backends (bounded submit waits, no early
return past a `threadgroup_barrier`, device limits queried at runtime).

## 3. What differs between the three

### CPU — deterministic, zero-setup, bit-identical

- Quantized weights straight from the GGUF mmap; activations quantized to
  **Q8_0** (32-value blocks) or **Q8_K** (306-byte super-blocks) per matmul —
  the int8×int8 dot kernels are AVX2 / NEON+SDOT with scalar fallbacks
  ([walkthrough 10](./inference_e2e_walkthrough/10-cpu-matmul-kernels.md)).
- Thread parallelism is **ownership-based** (one matmul row / one attention
  head per worker), so results are bit-identical for any `--threads` value —
  the property the greedy-token verification gates rely on.
- KV regions are f32; scores are computed in f32.

### Metal — zero-copy on unified memory

- Weights register with `newBufferWithBytesNoCopy`: the GGUF bytes *are* the
  Metal buffer, no copy, on Apple Silicon's unified memory
  ([walkthrough 14](./inference_e2e_walkthrough/14-metal-backend.md)).
- Activations stay f32; a per-op dispatch matrix picks handwritten shaders
  (3 matmul tiers, 5 attention variants, rms_norm 2 widths) and decode
  fusions.
- Optional f16 KV regions halve attention bandwidth (`MINFER_CACHE_TYPE=f16`).

### CUDA — opt-in, campaign-tuned

- Built only with `--features cuda` (plain builds never touch nvcc);
  `--features cuda,cuda_static` links cudart statically for deployment
  (`docs/BUILD.md`).
- Prefill runs int8 **MMQ** tensor-core GEMMs; decode runs weight-streaming
  MMVQ kernels; attention is split-KV with a combine pass — the whole arc is
  the [CUDA optimization campaign](./CUDA_OPTIMIZATION.md) (r5–r60, D1–D4-4).
- Captures decode-shaped splits as **CUDA Graphs** and replays them
  (`graph_replay`; `MINFER_NO_CUDA_GRAPH=1` to disable) — the only backend
  with a capture/replay protocol on the trait.

### Fusion capability differs per backend

The fusion pass consults `supports_fused` before producing fused IR nodes, so
the same model builds a different graph per backend: Metal accepts
`SwiGLU` + `QKVBiasRopeStore` (`metal_backend.rs:289`); CUDA accepts `SwiGLU`
(`cuda_backend.rs:1303-1305`) and carries its own fused decode kernels from
the campaign (see the CUDA docs for the inventory). The decode fusions
(`Op::FusedQKV`, `Op::FusedFFN`) are env-revertable
(`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1`) and are part of the reuse
identity. Fused vs unfused is bit-identical; when comparing, the unfused path
must still run the FusionPass.

## 4. Support matrix and forcing a backend

- Quant formats per backend (including the Metal prefill GEMM dispatch
  window and CUDA MMQ notes): `docs/SUPPORT-MATRIX.md`. Short version:
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K everywhere; Q2_K/Q3_K/I-quants
  nowhere.
- `MINFER_DISABLE_MPS=1` — force the CPU backend on macOS.
- `MINFER_NO_NEON=1` (aarch64) — drop the CPU NEON layer to scalar (A/B lever).
- `MINFER_NO_CUDA_GRAPH=1`, `MINFER_CACHE_TYPE=f16|f32`,
  `MINFER_NO_FUSE_QKV` / `MINFER_NO_FUSE_FFN` — per-backend behavior levers.
- Which backend to expect: the startup banner and `MINFER_TRACE` /
  `MINFER_GRAPH_TRACE` show per-node assignments
  ([walkthrough 08](./inference_e2e_walkthrough/08-scheduler-execute.md) §4).

**Numerics across backends are not identical by design** — CPU quantizes
activations, GPUs read f32 (and CUDA prefill quantizes differently still), so
CPU-vs-GPU logits differ; every path is verified against its own reference
(greedy output equality with llama.cpp where noted in the support matrix).

## 5. Reading order

1. [walkthrough 08 — the scheduler](./inference_e2e_walkthrough/08-scheduler-execute.md) — splits, copies, execution.
2. Backend episodes of the walkthrough: [10](./inference_e2e_walkthrough/10-cpu-matmul-kernels.md) / [11](./inference_e2e_walkthrough/11-attention-vecops-kv.md) (CPU), [14](./inference_e2e_walkthrough/14-metal-backend.md) (Metal), [15](./inference_e2e_walkthrough/15-cuda-backend.md) (CUDA).
3. `docs/GPU_SAFETY.md` — the hard rules before touching GPU code.
4. Per-backend history: `docs/CPU_OPTIMIZATIONS.md`, `docs/METAL_OPTIMIZATIONS.md`, `docs/CUDA-BACKEND-PLAN.md` + `docs/CUDA_OPTIMIZATION.md`.
