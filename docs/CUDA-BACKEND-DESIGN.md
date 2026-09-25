# minfer CUDA Backend Design

How minfer runs the compute graph on an NVIDIA GPU: the `CudaBackend` graph executor, the `cuda.rs`
device layer, the kernel families it dispatches, the CUDA Graph capture/replay machine, and the
memory and safety rules that hold it together.

> **Status.** Landed. Phase 7 (7a–7e) shipped the backend, and the later campaigns (Phase 8, R/P5,
> the MMQ line, the decode campaign D1–D4, D5-R speculative decoding) built on it. Every mechanism
> described here is implemented in the tree.
> Baseline: `HEAD = 916a7d2` (2026-09-14).
>
> **Provenance.** This file was `docs/CUDA-BACKEND-PLAN.md`, written before Phase 7 as a plan. The
> section skeleton is preserved; the body has been rewritten in the present tense against the
> current code, and the plan-time "current state" / "placeholder" / v1-matrix text has been replaced
> by the landed design. Plan-era commit hashes that no longer resolve are noted where cited.
>
> **Related records.** This document is the backend *design* (what it is and how it fits the graph).
> The optimization campaign — every measured lever, accepted or reverted — lives in
> `docs/CUDA_OPTIMIZATION.md` (live state + §0 history table + Appendix A env-gate reference) and its
> expansion layer `docs/cuda_optimization_steps/` (one document per step). `docs/CUDA-TECH-PRIMER.md`
> explains the GPU techniques, `docs/GPU_SAFETY.md` holds the hard safety rules (CUDA section), and
> `docs/inference_e2e_walkthrough/15-cuda-backend.md` narrates the backend for a first-time reader.
> The graph contract this backend implements is `docs/COMPUTE-GRAPH-DESIGN.md` §3.5/§9.

---

## 1. Design Goals and Outcome

### 1.1 Goal

Implement `CudaBackend` (`src/graph/cuda_backend.rs`) as the third backend of the compute graph by
**wrapping** the existing `src/cuda.rs` device layer — not stubbing it — so that on a CUDA machine
the whole per-layer chain runs on the GPU through the standard
`build → assign → fuse → alloc → execute` pipeline, with the same correctness contract as CPU and
Metal: backend placement is decided at build time, kernel-invariant violations return `Err`, and
there is never a silent mid-run fallback.

### 1.2 Outcome

| Goal | Landed outcome | Evidence |
|---|---|---|
| `CudaBackend` wraps `cuda.rs` in the `Backend` trait | `cuda_backend.rs` implements the full trait (pool, name→ptr weights, per-op dispatch, host transfers, graph replay); the device layer keeps the registry, streams and kernel launchers | §2, §4.1–§4.2 |
| Per-op dispatch, not a whole-layer call | Every `Op` the model builders emit has a CUDA arm; the legacy `cuda.rs::layer_gpu` is no longer driven | §4.4 |
| Per-node placement decided at build time | `supports_op` + the model-level `weights_on_cuda` all-or-nothing gate feed `CParams.gpu`; the scheduler splits the graph | §4.3, §4.6 |
| Weights resident from load, no execution-time host copies | The loaders register every graph-referenced tensor by name at model load; `execute_node` resolves pointers from the registry | §2.3, §4.6 |
| CUDA Graph capture/replay | Decode splits capture after a 2-run warmup and replay as one launch; repeated identical-nt prefills capture too (default on since R3-B); keyed by `(uid, node range)` with pool-generation invalidation | §4.8 |
| Same correctness gates as Metal | Per-op parity tests, per-quant matmul parity, capture bit-parity, greedy-text equality vs CPU; CPU-vs-GPU logits differ by design (f32 activations vs Q8_0) and are compared with tolerance classes | §7 |
| Performance | Post-campaign default: 7B Q4_K_M whole-prefill ~3581 tok/s (1.080× llama.cpp same-window) and decode 51.2 tok/s tg128 (1.074×); the full ledger is `CUDA_OPTIMIZATION.md` §1 | §5, `CUDA_OPTIMIZATION.md` §1.1 |

### 1.3 Non-goals

- **Multi-GPU / tensor split** (llama.cpp's split machinery) — single-GPU engine.
- **FP16/BF16 activations and cuBLAS/cublasLt.** Prefill uses f16 weight tiles and int8 tensor-core
  MMQ, and the KV cache can be f16, but activations stay f32 and the matmul path is minfer's own.
- **IQ\*/Q2_K/Q3_K kernels** — not implemented and not planned without a model that needs them.
- **Windows** and a **self-hosted CUDA CI runner** (deferred; device-gated tests skip gracefully).
- **`graph_optimize`-style node reordering** — the graph topology is the builder's decision.

### 1.4 Related records

| Topic | Where |
|---|---|
| Optimization history, current state, env-gate reference | `docs/CUDA_OPTIMIZATION.md` |
| Per-step records (every lever, accepted/reverted/closed) | `docs/cuda_optimization_steps/` |
| GPU technique primer | `docs/CUDA-TECH-PRIMER.md` |
| Safety rules (capture windows, D2H staging, weight ownership) | `docs/GPU_SAFETY.md` (CUDA section) |
| Graph contract (IR, allocator, scheduler, backend trait) | `docs/COMPUTE-GRAPH-DESIGN.md` |
| Beginner narrative of this backend | `docs/inference_e2e_walkthrough/15-cuda-backend.md` |
| MMQ analysis, speculative decoding | `docs/LLAMA-CPP-MMQ-ANALYSIS.md`, `docs/SPECULATIVE-DECODING-PLAN.md` |

---

## 2. Architecture at a Glance

### 2.1 Three layers

| Layer | File | Role |
|---|---|---|
| Graph executor | `src/graph/cuda_backend.rs` | Implements `Backend`: device buffer pool, per-op dispatch, positions conversion, CUDA Graph state machine, trace staging, error contract |
| Device layer | `src/cuda.rs` | `CudaState` singleton: device probe, weight registry, the one stream + stream lock, `extern "C"` kernel launchers, CUDA Graph API, pinned/staging memory, MMQ caches and gate reads |
| Device tier table | `src/device_tier.rs` | cc-keyed tier rows (measured GB10 + llama.cpp-adopted consumer rows + GENERIC) resolved once at init; feeds the MMQ gate, smem feasibility and plane-VRAM budget checks. Design + status: `DEVICE-ADAPTATION-PLAN.md`, docs 105–106 |
| Kernels | `src/cuda_kernels.cu` | The `__global__` kernels (quantized matmul families, attention, norms, elementwise, KV store, embedding gather, quantize planes) |
| Build chain | `build.rs` | Opt-in `--features cuda`, nvcc/`-ccbin` probe, per-arch SASS/PTX (incl. native sm_121), cudart link + rpath |

The split is deliberate: `cuda.rs` is the only place that touches the CUDA runtime API, and
`cuda_backend.rs` is the only place that knows about graph nodes.

### 2.2 The `CudaBackend` surface

```rust
pub struct CudaBackend {
    state: &'static crate::cuda::CudaState,   // process-wide device singleton
    kv_f16: bool,                             // KV element type for this instance
    pool: Vec<CudaBuf>,                       // id -> { ptr, bytes }
    free: Vec<usize>,                         // byte-length-matched free list
    pool_gen: u64,                            // bumped on every pool allocation
    pos_scratch: *mut c_void,                 // device i32 positions plane
    pos_scratch_bytes: usize,
    pos_memo: Option<(usize, u64)>,           // one-execution-window conversion memo
    graph_execs: Vec<CapturedGraph>,          // instantiated graphs
    graph_runs: HashMap<(u64, (usize, usize)), u32>, // warmup counters
    capturing: Option<(u64, (usize, usize))>, // open capture window
    stream_guard: Option<MutexGuard<'static, ()>>,
    graphs_mode: GraphMode,                   // Enabled / Disabled
    prefill_capture: bool,                    // default ON (MINFER_NO_PREFILL_CAPTURE opts out)
    cap: crate::cuda::CaptureStaging,         // viz/trace async D2H staging
}
```

`new()` returns `None` when `CudaState::get()` fails (no device, or `MINFER_DISABLE_CUDA=1`, both
handled inside the device layer), which is how `GraphAllocator::enable_cuda` declines. The struct is
`unsafe impl Send/Sync`: raw device pointers are dereferenced only by the GPU, and mutation happens
only through `&mut self` (mirroring `MetalBackend`).

### 2.3 Weight residency and registry

Every graph-referenced tensor is uploaded once at model load and referenced by name afterwards:

- `CudaState::register_weight(name, data)` allocates and H2D-copies; the same name **and** size is a
  no-op (reloading the same GGUF must not leak a second device model), while a different size
  replaces the entry and deliberately leaks the stale buffer, because a live captured graph may
  still reference it. The leak is bounded by the number of distinct (architecture, tensor) shapes.
- `has_weight_of_size(name, raw_len)` is the size-aware gate used by the model wiring; it matches
  repacked weights by their **original** raw length, so a foreign-architecture entry reads as "not
  registered" and that model stays on CPU.
- `get_weight_ptr(name)` is the only lookup the executor uses.
- Quant-specific registrations: Q6_K uses a 224-byte-padded repack
  (`register_weight_q6k_padded`, enabling 16-byte `uint4` loads); Q8_0 optionally registers a p32
  split plane (`register_weight_q80_p32`); Q6_K/Q4_K optionally build pre-expanded / pre-decoded
  planes (`register_weight_q6k_exp`, `register_weight_q6k_dsc`, `register_weight_q4k_dsc`, and the
  D4-4 `{name}__dpl` dense split plane). Plane maps are keyed by the weight's device pointer and
  looked up inside `prefill_mmq`.
- A per-weight f16 dequant cache (`w16_cache`) is enabled by the loader only when quantized matmul
  weights exceed 2 GiB **and** MMQ is off; `MINFER_NO_W16CACHE=1` reverts.
- `ModelLoadGuard` (reentrant, process-wide) serializes loader registration so two models with
  same-named tensors cannot interleave, and real-model tests hold it across their forwards.

The graph-side CPU registry is separate: `register_graph_weights` registers into the allocator's CPU
backend, which is what a CPU split of a mixed graph executes from; the CUDA registry is filled by the
loaders at model-load time.

### 2.4 Streams, locking and capture windows

Everything runs on **one stream**, and stream capture is per-stream rather than per-thread — so a
process-wide `CudaState::stream_lock()` serializes stream use:

- Normal operation: each backend call takes the lock for its own enqueues (`stream_guard()`).
- While this backend has a capture window open, it **holds** the lock for the whole window (its own
  enqueues skip re-locking); any other thread's stream work would otherwise be recorded into the
  graph. Production runs a single engine thread, so the lock is uncontended.
- `synchronize` closes a window (`close_capture_or_sync`: end capture, instantiate, launch once,
  cache) or performs a plain stream sync. The sync also polls `cudaGetLastError`; since issue
  [#145](https://github.com/yusiwen/minfer/issues/145) it reports what it finds as a **latched API
  error observed by `cudaGetLastError`**, naming the `cudaGetErrorName` symbol and stating that it is
  *not* attributed to a kernel. The old message ("CUDA kernel launch error: 1") blamed whatever
  kernel had just run for an error an earlier call had latched. The latched error is still counted
  (`latched_api_error_count`) and cleared — never dropped — but the fix belongs at the call site that
  discarded the return value.

**The eager prefill-GEMM smem opt-in (issue [#145](https://github.com/yusiwen/minfer/issues/145)).**
A `gemm_f16_nt_kernel_t` instantiation whose dynamic shared memory exceeds the 48 KiB default must be
opted in with `cudaFuncSetAttribute(.., cudaFuncAttributeMaxDynamicSharedMemorySize, N)` **before**
anyone opens a capture window — `cudaFuncSetAttribute` is illegal under
`cudaStreamCaptureModeGlobal`, and a lazy first-use opt-in inside a window fails and poisons the
first captured launch. `CudaState::try_new` therefore calls `gemm_prefill_smem_init()` eagerly. The
number it requests is the kernel's own byte layout — `As 2*TN*KS halves + Am 2*TN*KS floats (AF32
only) + Bs 2*TM*KS halves + Cs NW*256 floats`, TN = 64, NW = `blockDim.x/32` = 8 — and
`gemm_dynamic_smem_bytes(tm, ks, af32)` is the single source that the launcher
(`launch_gemm_f16`) and the init both read; before #145 the init carried a stale copy of it while the
launcher carried a copy that dropped the AF32 mirror. Every call's return value is checked and a
failure is named once, at init (function, attribute, requested bytes, device limit,
`cudaGetErrorName`) and cleared there. A request that exceeds the device's own
`cudaDevAttrMaxSharedMemoryPerBlockOptin` is **skipped with the reason printed**, instead of called:
the call could only return `cudaErrorInvalidValue` (which `compute-sanitizer` counts) and the
instantiation cannot launch on that device at all. On GB10/sm_121 (limit 101376 B) that is exactly
one combination, `gemm_f16_nt_kernel_t<256,64,true>` at 122880 B.

### 2.5 Legacy surface

The pre-graph imperative path still compiles but is not driven by inference: `layer_gpu`,
`output_norm_gpu`, `init_kv_cache` + the `kv_k/kv_v/kv_size` slots, the `buf_*` persistent-slot pool
with `get_or_grow`/`upload_*`, the host-staged `quant_matmul_*` wrappers, and the old single-slot
capture API. Each carries a scoped `#[allow(dead_code)]` with a reason; the release build is
warning-free. The graph path owns KV through the allocator's persistent regions, so `init_kv_cache`
is bypassed entirely.

---

## 3. llama.cpp Reference Map

llama.cpp's CUDA backend (`ca3d5a3e1` at the time of the port) was the reference for the device
layer, the replay state machine and the kernel strategy. What was borrowed and what was deliberately
not:

| llama.cpp concept | minfer analog | Status |
|---|---|---|
| Backend interface (`graph_compute`, `synchronize`, async tensor set/get, `supports_op`) | `Backend` trait: per-node `execute_node` instead of a whole graph | Borrowed, reshaped |
| Weights placed in device buffers at load; ops follow their weights; never host-copied during execution | `register_weight` at load + name→ptr lookup in `execute_node` | Borrowed |
| Device buffer pool, alloc/free hot path | `CudaBackend` pool with a byte-length free list; `alloc_fresh` for split staging | Borrowed |
| Enqueue during compute; synchronize only at scheduler boundaries | `execute_node` launches on the one stream; `synchronize()` at split boundaries | Borrowed |
| CUDA Graph: warm up twice, capture the third run, replay keyed per graph, invalidate on pointer change | `(uid, node range)` key, 2-run warmup, `pool_gen` invalidation, launch-once at close | Borrowed |
| Capture disqualifiers (host syncs, arch floor, env off-switch) | No syncs/readbacks inside a window; `MINFER_NO_CUDA_GRAPH=1` | Borrowed |
| Replay requires stable buffer addresses across steps | `GraphCache` keeps the allocator (and pool pointers) alive across decode steps | Borrowed |
| int8 tensor-core quantized GEMM (MMQ) | The r-series MMQ line: pad40 transposed A planes, fused quantize producers, raw-byte NB-BT kernels | Borrowed in spirit, own kernels |
| Flash-attention tiling | decode split-KV attention, batched verify attention, FA-style tiled prefill attention | Borrowed in spirit, own kernels |
| Fusion pass patterns (`ggml_cuda_try_fuse`) | minfer's `FusionPass` + build-time fused nodes; CUDA implements `SwiGLU`, `FusedQKV`, `QkvBiasRopeStore`, `FusedFFN` | Diverged (graph-level fusion) |
| Multi-GPU split, NCCL, VMM pool, cuBLAS paths, `graph_optimize` reordering | — | Deliberately skipped |
| Abort-on-capture-failure | logs, disables graphs for the session, continues with direct launches | Diverged (chosen) |

The one llama.cpp idea still on the table as a step function is the **q8_1 GEMM-prologue fusion**
(see `CUDA_OPTIMIZATION.md` §1.4); everything else in the campaign is closed or sub-bar.

---

## 4. Design

### 4.1 `CudaBackend` lifecycle and state

`new()` resolves the device singleton, snapshots the process-wide KV element type
(`crate::cuda::kv_cache_is_f16()`), reads the two graph gates
(`MINFER_NO_CUDA_GRAPH=1` → `GraphMode::Disabled`; `prefill_capture` defaults ON unless
`MINFER_NO_PREFILL_CAPTURE=1`), and starts with an empty pool. `Drop` takes the stream guard (a
`cudaFree` implicitly syncs, so it must serialize against an open capture window), frees every pool
pointer, the positions scratch, and every captured graph exec.

State groups:

| Group | Fields | Lifecycle |
|---|---|---|
| Device + KV policy | `state`, `kv_f16` | fixed at construction |
| Pool | `pool`, `free`, `pool_gen` | grows on demand; `free_buffer` only recycles; `alloc_fresh` bypasses the list for split staging; `pool_gen` bumps on every allocation |
| Positions | `pos_scratch`, `pos_scratch_bytes`, `pos_memo` | grown on demand; the scratch pointer is embedded in captured execs, so growth bumps `pool_gen` to force re-capture |
| Capture | `graph_execs`, `graph_runs`, `capturing`, `stream_guard`, `graphs_mode`, `prefill_capture` | see §4.8 |
| Trace | `cap` | pinned async D2H staging, see §4.7 |

Pool rules worth restating because they carry correctness weight:

- **Exact byte-length reuse only** — `alloc_buffer` scans `free` for `pool[id].bytes == size * 4`.
- **`free_buffer` never frees** — a persistent KV region must survive rebuilds, and the pool keeps
  device memory for the next graph. Only `Drop` returns memory to the driver.
- **`alloc_fresh`** exists for split-boundary staging: ids in the free list are still referenced by
  `node_to_buf` and physically live during the execute that follows.
- **OOM is not a panic.** `cuda_malloc` logs and returns null; the null buffer fails cleanly at
  execute time (`ptr_of`). Panicking is forbidden because the backend may hold the process-wide
  stream lock, and a panic under that mutex would poison it for every other user.

### 4.2 Backend trait mapping

| Trait method | CUDA implementation |
|---|---|
| `name()` | `"cuda"` |
| `supports_op(op, dtype)` | §4.3 table |
| `supports_fused(fused)` | `matches!(fused, FusedOp::SwiGLU)` — the only variant in the enum |
| `alloc_buffer` / `free_buffer` / `alloc_fresh` | §4.1 rules |
| `execute_node(node, in_bufs, out_buf, kv_pair)` | wraps `execute_node_inner`; on `Err` with an open capture window it calls `abort_capture` first (§4.8) |
| `read_host(id)` | **`None`** — a staged D2H cannot return a borrowed `&[f32]` from `&self`; the allocator's `copy_to_cpu` arm calls `copy_to_host` instead |
| `write_host(id, data)` | size-checked H2D; small inputs go through the pinned async ring (§4.7) |
| `synchronize()` | clears the MMQ cache + positions memo, then `close_capture_or_sync()` |
| `graph_replay(uid, range, nt_hint)` | §4.8 state machine |

### 4.3 Eligibility

`supports_op` is f32-activation only (`dtype != DType::F32` ⇒ `false`) and answers:

- **Unconditionally supported**: `Input`, `Add`, `Mul`, `Silu`, `SwiGLU`, `RmsNorm`, `QkNorm`,
  `MatMul`, `Attn`, `KvcacheStore`, `KvcacheLoad`, `View`, `Reshape`, `Permute`, `GetRows`,
  `FusedQKV`, `QkvBiasRopeStore`, `FusedFFN`.
- **Conditional**: `RoPE` only for `RopeStyle::NonInterleaved` (the neox layout; the only style the
  supported architectures emit).
- **Everything else** (`Scale`, `Softmax`, `BatchMatMul`, `FusedQkvNorm`) stays on CPU.

Two layers of checks are deliberately *not* in `supports_op`:

1. **Weight/quant eligibility is a model-level, all-or-nothing gate** (`weights_on_cuda`, §4.6). A
   layer whose weights are not all registered in a kernel-supported type keeps the whole graph on CPU
   rather than creating a partial-GPU split. The whitelist for matmul weights is
   Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F32; the embedding (`tok_embd`) additionally has its own
   type list because it is gathered, not multiplied.
2. **Shape and feature invariants are enforced in `execute_node`** and return `Err` — never a silent
   fallback. Guards include: `transpose_b` unsupported; quantized matmul `id % 32 == 0`; `RmsNorm`/
   `QkNorm` dim a nonzero multiple of 4 (the float4 kernel); attention `nkt == n_head_kv * hd`,
   `hd == hd_kv`, `hd` a multiple of 4 in 1..=128, `n_head % n_head_kv == 0`; the fused decode nodes
   `nt == 1`; `RoPE` neox; `KvcacheStore`'s output buffer must be the K region; a missing declared
   norm weight is an error.

### 4.4 Execution dispatch

`execute_node_inner` opens with one stream-lock acquisition and a cache rule:

> The MMQ A-quantize memo is valid only across **consecutive** `MatMul`/`FusedFFN` nodes; every
> other node kind clears it (`clear_mmq_cache`). `FusedFFN` is in the preserve set because its input
> is the FFN-norm output whose pre-quantized plane the fused producer just recorded.

| Op | CUDA path | Picking conditions |
|---|---|---|
| `Input`, `KvcacheLoad` | no kernel | host-filled / the output *is* the persistent K region |
| `View`/`Reshape`/`Permute` | `copy_d2d` identity | — |
| `GetRows` + `Embed` meta | `embed_rows_on_gpu` (per weight type, incl. the padded Q6_K layout) | weight registered; type via the model gate |
| `GetRows` + no meta | `gather_rows_f32_on_gpu` | the G3 tail-row gather |
| `Add` / `Mul` | `add_f32` / `mul_f32` | input element counts must match |
| `Silu` | `copy_d2d` if not aliased, then `silu_f32` in place | in-place alias rule |
| `SwiGLU` | producer-fused `swiglu_quant_nw` (mode 2) → `swiglu_quant` → `swiglu_f32` | `rows >= 16 && dim % 256 == 0` plus the MMQ gate set and `MINFER_MMQ_A_FUSE` mode; plane OOM degrades mode 2 → 1 → unfused |
| `RmsNorm` | prefill producer-fused `rms_norm_quant_nw`/`rms_norm_quant` → decode `rms_norm_quant_on_gpu` → `rms_norm` | prefill `n >= 16 && d % 256 == 0`; decode `n == 1 && d % 32 == 0 && !MINFER_NO_DECODE_A_FUSE` |
| `QkNorm` | `rms_norm` with `d = hd` over the flat `[nt*nh, hd]` view | `hd % 4 == 0`, nonzero, divides the element count |
| `MatMul` | `matmul_f32_ptr_layout` + optional `add_bias_f32` | see the family table below |
| `RoPE` | `copy_d2d` if not aliased, then `rope_f32` | neox; `hd` even |
| `KvcacheStore` | `store_kv_f32` / `store_kv_f16`, or `store_kv_q8_0` for a packed cache (K then V), per `kv_layout` | `out_buf == k_id`; `nt = elems(K_in)/nkt`; the rows are the `cells` input (C6) — device data, not re-validated against `n_ctx` here (the allocator's `kv_cells_for_seq` and `fill_input_i32` own that). The packed store maps one thread to one `(row, 32-element block)` and uses the CPU's quantizer, so both backends write the same bytes |
| `Attn` | f32/f16: `nt == 1` → `gqa_attn_split`; `1 < nt <= 16` → `gqa_attn_split_batched`; `nt > 16` → `gqa_attn_f16kv` (FA prefill when `hd == 128 && !MINFER_NO_FA_PREFILL`, else legacy) or `gqa_attn_f32`. **q8_0** (C4 S2b): `nt == 1` → `gqa_attn_split_q8_0` (the same 1-warp split-K body, `rpw_gate = 0` — the hybrid 4-warp body is f16-typed); every `nt > 1` → `gqa_attn_f32` (the batched split kernel's bitwise-identity purpose is not claimed for Q8_0, and `fa_prefill_f16kv` is f16-typed shared-memory staging) | the attention guards of §4.3; the batched verify path is bitwise-equal per position. A packed cache must therefore refuse `--spec-draft` (`spec::SpecEngine::new`), and its prefill is correct but off the tuned FA route |
| `Attn` **windowed** (`explicit_span`) | the same entry points, instantiated with `CAUSAL = false`; `bound` carries `[lo, hi)` pairs (`bound[t]` = `lo`, `bound[nt + t]` = `hi`) instead of `positions`, and every per-row limit must come from `hi`. All three window modes are instantiated per layout | `cuda_windowed_attention_matches_causal_for_long_windows` sweeps `(nh, nk, hd)` × `n` × `start` × **both KV dtypes**; `cuda_map_window_matches_the_span_over_the_same_rows` sweeps all three layouts (f32/f16/q8_0) over the same rows. `fa_prefill_f16kv` used `bound[t]` (the window's `lo`) as the causal limit until 2026-09-19, which made every non-zero-start prefill attend to a single row — see `ARCHITECTURE-EXECUTION-PLAN.md` §14 row 0 |
| `FusedFFN` | concat `matmul_f32_ptr_layout` + in-place `swiglu_quant_off`/`swiglu_f32_off` | `nt == 1`; offset fuse when `n % 32 == 0` |
| `FusedQKV` | concat matmul over `[wq\|wk\|wv]` + `attn_bias_rope_store` (sources `[x, positions, cells]`) | `nt == 1`, neox, even `hd`; concat weight + 3 biases registered; KV pair present. `positions[0]` ropes q/k, `cells[0]` addresses the four KV writes (C6), so the node is valid for a run that does not start at cell 0 |
| `QkvBiasRopeStore` | `copy_d2d` for q + `attn_bias_rope_store` over three separate matmul outputs (sources `[q, k, v, positions, cells]`) | `nt == 1`, neox, even `hd`; 3 biases registered; same positions/cells split |
| anything else | `Err("cuda: op ... has no kernel ...")` | — |

**MatMul family selection** (`matmul_f32_ptr_layout` in `cuda.rs`):

- **Prefill GEMM** when `(nt >= 9 || MINFER_SMALL_M_GEMM=1) && id % 32 == 0 && !MINFER_NO_PREFILL_GEMM`
  and the type is quantized: the promoted int8 **MMQ** path (`MINFER_MMQ`, default on) — pad40
  pre-transposed A planes, fused producers, raw-byte NB-BT tensor-core kernels for q4_K/q6_K; else
  the f16 **wmma** GEMM (or the persistent f16 weight cache / `MINFER_FUSED_B` dequant-in-GEMM).
- **Decode / small batch**: per-type **MMVQ** (dp4a over q8_0 activations) for `nt == 1` with
  shape gates, `_multi` variants for `nt 2..8`, and f32-activation kernels otherwise. Q4_1/Q5_0/Q5_1
  have f32 kernels only; F32 weights use `f32_f32_matmul_vec`/`_scalar`.
- **Bias** is applied by `add_bias_f32` after the GEMM; its last argument is the **row count** `nt`
  (the kernel maps one block row per token).

### 4.5 Allocator and scheduler integration

- `GraphAllocator::supports` priority is **Metal → CUDA → CPU**; `enable_cuda()` mirrors
  `enable_metal()`. On a CUDA-only host the practical effect is CUDA first.
- **KV compaction (C3) is not a node op** — it is the `Backend::copy_cells` trait method, called by
  `GraphAllocator::kv_defrag` between forwards. CUDA implements it with `kv_move_rows`: one block,
  rows walked **ascending** with a `__syncthreads()` between them, because the contract is
  `dst_row <= src_row` with **overlapping** ranges (a compaction slides a run into the gap just below
  it) and device-to-device `cudaMemcpyAsync` is documented undefined for overlap. No staging buffer,
  no second pass. The launcher returns non-zero on a contract violation and the Rust side turns that
  into an `Err`, so the allocator fails the compaction **before** it renumbers any run. It runs on the
  backend's own stream, so it is ordered after the previous forward's kernels.
- KV regions are created by `ensure_kv` on the layer's assigned backend, so with CUDA assignment the
  per-layer K/V regions live in the CUDA pool and `KvProvider::kv_pair` returns pool ids that
  `execute_node` resolves to device pointers. `init_kv_cache` is bypassed.
- Cross-backend values go through the allocator's staging path: `copy_to_cpu` (pinned D2H) +
  `write_host` (pinned H2D) into a buffer allocated with `alloc_fresh` on the consumer's backend.
- The scheduler syncs at split boundaries (`sync_backend` → `CudaState::sync()`), which is also where
  a pending capture window closes. In practice the post-7e③ graph is a **single CUDA split** for
  Qwen2/Qwen3 (embed gather and tail gather are on device), so there are no per-step cross-backend
  copies at all; splits appear only in synthetic or intentionally mixed graphs.

### 4.6 Model wiring

Both architectures use the same shape (Qwen2 shown; Qwen3 mirrors it):

```
cuda_on  = CudaState::get().is_some() && weights_on_cuda(model)     // #[cfg(feature = "cuda")]
metal_on = metal_available() && weights_on_gpu(model)
CParams.gpu = metal_on || cuda_on
```

- **`weights_on_cuda`** is the all-or-nothing gate: every graph-referenced tensor must be registered
  (`has_weight_of_size`) and, for matmul/embedding tensors, of a kernel-supported type. On failure it
  prints `CUDA GATE: weight '<name>' (type <t>) has no CUDA kernel or is not registered` and the
  model runs entirely on CPU. Since 7e③ `tok_embd` is gated like every other weight (the embedding
  gather is on device); its own type list is F32/Q4_0/Q8_0/Q4_K/Q5_0/Q5_1/Q6_K/Q5_K.
- **Registration happens in the loader**, not in `register_graph_weights` (which fills the CPU
  registry only): `cuda.register_weight` per tensor, `register_weight_q6k_padded` for Q6_K,
  `register_weight_q80_p32` for the q8_0 split plane, and the `_exp`/`_dsc`/`__dpl` planes under
  their gates. The loaders also build the fused concat weights with `cuda::concat_rows` and register
  them: `blk.{i}.attn_qkv` (Qwen2 only — the Qwen3 fused-QKV path is Metal-only, so CUDA registers no
  Qwen3 attn_qkv) and `blk.{i}.ffn_gu` (both models, gated on `nf <= 16384`).
- **Concat availability**: Qwen2's `qkv_concat_available` / `gu_concat_available` use the
  **metadata-only** `cuda::concat_rows_feasible` probe on the CUDA arm, because rebuilding the concat
  bytes during graph construction measured a ~920 ms stall per decode graph build. Qwen3's
  `gu_concat_available` uses the eager `cuda::concat_rows` (its concat is built once per layer at
  load either way); Qwen3's `qkv_concat_available` has no CUDA arm.
- **Fused nodes per model**: Qwen2 builds `FusedQKV` (concat class) or `QkvBiasRopeStore` (mixed-quant
  class, CUDA-only) for decode QKV, and `FusedFFN` when `nf <= 16384`. Qwen3 builds `FusedFFN` but
  **not** its `FusedQkvNorm` on CUDA: that fused path is Metal-only (the Qwen3 concat probe has no
  CUDA arm), so CUDA Qwen3 runs the unfused `qk_norm` chain.
- **FusionPass** receives the CUDA backend in its `Vec<&dyn Backend>` and `backend_of` maps
  `Backend::Cuda` to the position found by name, so the SwiGLU rewrite is gated by CUDA's own
  `supports_fused`.
- The dump/debug tags in `forward_cached` are backend-agnostic (`MINFER_GRAPH_DUMP`;
  `MINFER_REBUILD_TRACE=1`, Qwen2 only). `MINFER_CUDA_DEBUG` is a device-layer trace for the legacy
  surface.

### 4.7 Memory, residency and staging

**Residency.** Weights are uploaded at load and never copied during execution; only activations,
positions and logits cross PCIe (tiny for decode). Quant-specific registrations trade device memory
for speed: the Q6_K padded repack (224-byte slots), the q8_0 p32 split plane (≈ +94% of that
tensor's bytes), the q6_K `W_exp` dense plane (~1.52 GB on 7B), the q4_K/q6_K `W_dsc` f32-pair planes
(~1.46 GB), and the D4-4 q6_K dense split plane. Each has an opt-out gate (§5, env-gate reference in
`CUDA_OPTIMIZATION.md` Appendix A); the promoted default spends ~3.27 GB to reach the 1.080× prefill
path.

**KV layout.** The persistent K/V regions keep their f32 IR shape, but the store/attention kernels
run one of three layouts, tagged by `crate::cuda::KV_LAYOUT_F32/F16/Q8_0` — the same `0/1/2` codes
`KvFormat` uses, and a host contract the kernels are templated on (`int LAYOUT` in
`cuda_kernels.cu`):

- `KV_LAYOUT_F32` — one f32 per element;
- `KV_LAYOUT_F16` — one f16 per element in the first half of the f32-shaped region
  (`set_kv_cache_type` auto-selects it when `n_layers × n_kv_embd >= 8192`, the 7B class, and
  `MINFER_CACHE_TYPE=f16` overrides). Its **snapshots are restorable since [#130](https://github.com/yusiwen/minfer/issues/130)**:
  the session container records the type in its header flags (`FLAG_F16`, C5 S3), so an
  auto-f16 model's `--slots-file` / `--session` companion is no longer refused on load;
- `KV_LAYOUT_Q8_0` — packed 34-byte Q8_0 blocks, one cell rounded up to whole f32 words
  (`MINFER_CACHE_TYPE=q8_0`, C4 S2b).

Every KV address is formed in **bytes**: `kv_row(base, cell, row_bytes)` names a cell and
`kv4<LAYOUT>(row, elem) -> float4` is the one load idiom — the old `float4` load for f32, the old
two-`__half2` pair for f16 (both bit-identical to the pre-C4 instantiations), and for Q8_0 the f16
scale plus four quants of block `elem/32`. A 4-element group never straddles a block because a KV
head's base is `hd`-aligned and `hd % 32 == 0` (`ensure_kv`'s packed-width check). The kernels take
`const void* k/v` plus `size_t row_bytes`, and the launchers take the layout as an `int`.

`CudaBackend` snapshots the process-wide policy into its `kv_layout` field at construction
(`crate::cuda::kv_cache_layout`), and `kv_row_bytes(nkt)` derives the stride (`nkt*4`, `nkt*2`, or
`KvFormat::Q8_0.row_bytes(nkt)`). The packed store is `store_kv_q8_0`, whose quantizer is the CPU's
step for step (`amax/127`, f16 scale, round-ties-even), so both backends store the same bytes.
Before C4 S2b this was a `bool` that mapped anything not exactly `f16` to f32 — which would have
addressed a packed region as f32 rows, the silent corruption the layout tag exists to make
impossible.

**Per-engine scope ([#99](https://github.com/yusiwen/minfer/issues/99)).** #99 made the KV *format*
per engine for the model, the graph builder (`CParams::kv_format`), the allocator and the CPU
kernels; **this device layout is deliberately still a per-load process policy.** The kernels read
`crate::cuda::KV_LAYOUT` themselves (not through `CudaBackend::kv_layout`), so the tag a weight
load / `set_kv_cache_layout` installs is what every KV kernel in the process runs — the same
process-wide shape `CudaState`'s other state has. Making the layout per-graph means threading it
through every launcher in `cuda.rs` and the captured-graph key, which is filed as its own follow-up;
the discipline for now is the documented serial device run (`scripts/cuda_test.sh`,
`scripts/real_model_gates.sh`). The loader keeps the two halves in step: `load_model_configured`
restates `KV_LAYOUT_Q8_0` when the resolved format is packed, so the region the builder sizes and
the kernel that addresses it cannot disagree within one engine.

**Host transfers.**

- **H2D fills** go through a lazy ring of 8 × 2 MiB pinned slots (`write_input_async`): the Rust slice
  is copied into a slot and the `cudaMemcpyAsync` is queued; same-stream ordering guarantees the
  consumer kernels see the data. A ring wrap retires in-flight copies with one stream sync; oversized
  inputs or a failed `cudaHostAlloc` fall back to a blocking copy.
- **D2H readbacks** use a single grow-on-demand pinned buffer (`PinnedBuf`), pre-grown to 4 MiB at
  first prefill because the lazy `cudaHostAlloc` showed up as a 0.78 ms tail malloc at the logits
  readback. `MINFER_NO_PINNED_READBACK=1` reverts to pageable copies.
- **Device memory is not host-readable by plain memcpy on GB10** — a host probe that dereferences a
  device pointer faults. All D2H goes through `cudaMemcpy` staging (`copy_to_host`).
- **Trace/viz** uses `CaptureStaging`: one pinned buffer (128 MiB ceiling) into which each captured
  node's output is queued as a stream-ordered async D2H right after its launch, drained with a single
  sync at the split boundary. Node outputs above the ceiling fall back to the per-node synchronous
  copy.

**Caches.** The MMQ A-quantize memo (consecutive-window rule above), the positions→i32 memo (keyed by
`(input buffer id, pool_gen)`, cleared in `synchronize`), and the optional persistent f16 weight
cache (`w16_cache`, enabled when quantized matmul weights ≥ 2 GiB and MMQ is off) are all
execution-window caches with explicit invalidation points rather than long-lived state.

### 4.8 CUDA Graph capture and replay

`graph_replay(uid, range, nt_hint)` is called by the scheduler once per split, before its node loop.
The state machine:

1. `graphs_mode != Enabled` → direct launches.
2. An open capture window of our own → direct launches (a nested replay would be CUDA-invalid; the
   graph is single-split today so this is unreachable).
3. A stored exec for `(uid, range)` with a **matching `pool_gen`** → `cudaGraphLaunch`; on launch
   failure, disable graphs for the session and fall back.
4. A stored exec with a **different `pool_gen`** → destroy it, drop the warmup counter, re-warm.
5. Warmup: executions 1 and 2 of a key run direct launches (llama.cpp warms up twice); a one-shot
   prefill never reaches capture.
6. On the third execution, if `nt_hint.map_or(true, |nt| nt == 1 || prefill_capture)`, the backend
   takes the process-wide stream lock and opens a capture window around the node loop. The gate means
   decode-shaped graphs always capture; prefill-shaped graphs capture only when `prefill_capture` is
   on (default **ON** since R3-B; `MINFER_NO_PREFILL_CAPTURE=1` opts out).
7. The window closes at the split's `synchronize` → `close_capture_or_sync`: end capture, instantiate,
   **launch once** so the step still produces output, cache the exec at the current `pool_gen`, then
   sync. A failure destroys the exec, logs loudly, and disables graphs for the session.

Replay correctness rests on stable addresses: pool ids never move memory, `copy_across` rewrites the
same staging buffers each step, and the positions scratch pointer is embedded in captured execs — so
growing it bumps `pool_gen` and forces re-capture.

Two interactions are part of the contract:

- **Trace/viz disables replay.** The scheduler skips `graph_replay` entirely while `MINFER_TRACE` or
  live viz capture is active, because per-node host readbacks inside a capture window are illegal.
- **A node error inside the window aborts it** (`abort_capture`): end capture without launching,
  destroy the exec, release the lock, disable graphs, sync. Later steps run direct-launch with graphs
  disabled; the aborted step's outputs were never produced and are consumed as-is — there is no
  poisoned-error mechanism, and the code says so explicitly.

### 4.9 GPU safety (CUDA edition)

The hard rules live in `docs/GPU_SAFETY.md` (CUDA section); this is how the backend implements them:

1. **Errors are errors.** Guards return `Err` naming the node and the blocking values; the scheduler
   aborts. There is no mid-run CPU fallback.
2. **No sync inside an active capture window.** The 7e② incident — a temporary sync wrapper that
   produced garbage only with graphs on — is the recorded reason. Debug reads go through the
   boundary.
3. **D2H always through staging** (GB10 device memory is not host-readable by plain memcpy).
4. **A latched error is never blamed on the kernel that just ran.** `sync()` polls
   `cudaGetLastError` + `cudaStreamSynchronize`; the first reports whatever an *earlier* call on the
   thread latched, so its message names the observer and the `cudaGetErrorName` symbol and says it is
   not attributed to a kernel (the error is counted and cleared, not dropped). The two origins behind
   the old phantom "kernel launch error: 1" were a rejected `cudaFuncSetAttribute` and
   `cudaGraphDestroy` called on a `cudaGraphExec_t` — both fixed at their call sites.
5. **A return value that gates a later launch is read where the call is made.** In particular the
   dynamic-smem opt-in: an over-limit request is skipped with the reason, any other failure is named
   and cleared at the call site. The sync poll is the backstop for a *missed* site, not the place to
   diagnose one; the sites still uncovered are listed in
   [#147](https://github.com/yusiwen/minfer/issues/147).
6. **Same-stream ordering is the async-fill contract**; the pinned ring syncs on wrap and never
   hands a slot back early.
7. **Weight-registry ownership**: name+size reuse, different-size replace with a deliberate, bounded
   leak (a live captured graph may still reference the old buffer).
8. **Device limits are queried at runtime** (SM count, compute capability, free memory); the only
   hardcoded shape knowledge is the compiled target list and the documented kernel invariants.

---

## 5. Implementation Phases

### 5.1 Phase 7 (7a–7e)

| Phase | Content | Status |
|---|---|---|
| 7a | Skeleton + wiring: `CudaBackend` struct/pool/trait impl, allocator arms, `enable_cuda`, `supports()` priority; `execute_node` handles only `Input` | ✅ |
| 7b | Per-op execution + parity tests: full dispatch; un-`allow` the used device-layer methods; launch error checks | ✅ |
| 7c | Model wiring + E2E: CUDA gate + `weights_on_cuda`, FusionPass backend index, legacy KV pre-alloc removal | ✅ |
| 7d | CUDA Graph capture/replay: `uid` population, device-derived attention bound, capture state machine, replay hook in the trait/scheduler | ✅ |
| 7e① | CPU-path residual diagnosed as a path-identity artifact (cross-backend f32 reduction order), not a bug; gates switched to greedy equality on CUDA builds | ✅ |
| 7e② | Vectorized q4_K/q6_K kernels + Q6_K padded repack: 7B decode 8.4 → 26.4 tok/s (3.1×) | ✅ |
| 7e③ | Embed + generic `GetRows` on device: prefill/decode become a single CUDA split (no cross-backend copies) | ✅ |
| 7e④ | F32×F32 matmul kernels; F32-weight models participate in CUDA | ✅ |
| 7e⑤ | `FusedFFN` on CUDA (`concat_rows` + `swiglu_f32_off`); `CParams.fuse_ffn` decoupled from the QKV gate; 0.5B +10%, Qwen3-0.6B +4% | ✅ |
| 7e⑥ | Async H2D input fill through pinned staging; Q8_0 prefill GEMM shape-gated (8c) | ✅ |
| 7e⑦ | Docs + cleanup: per-item `#[allow(dead_code)]` with reasons, release build warning-free | ✅ |

### 5.2 After Phase 7

The optimization campaign is indexed in `docs/CUDA_OPTIMIZATION.md` §0 and expanded in
`docs/cuda_optimization_steps/`. For the design record, its eras and what each one changed in the
backend or its graph integration:

| Era / workstream | Backend impact | Step docs | Representative commits |
|---|---|---|---|
| Phase 8 foundations (8m–8p, 8e) | wmma f16 prefill GEMM, FA-style tiled prefill attention, decode-start stall elimination, persistent f16 weight cache, decode MMVQ | 02–06 | `ba3f317`, `cdc6599`, `cb66fca`, `65b686c`, `2992f57`, `b7b8e73`, `1298cb2`, `1d28235` |
| Phase 8 correctness/coverage (8a–8q) | KV f16, shaped Q8_0 GEMM, split-K attention, Q5_K/Q5_1/Q5_0 kernels, the F32-matmul latent bug, llama baseline | 78, 79 | `f7b0036`, `69a27c5`, `a5af60f`, `b959ec9`, `acca28f`, `9f419f9` |
| R1–R4 + P5 | R1 int8 MMQ, R2 MMVQ weight streaming, R3-A1 single-split prefill (`tail_ids` at the graph head), R3-A2 pinned D2H, R3-B prefill capture default ON, R4 decode split attention rewrite | 07–11 | `40e97c9`, `6df3245`, `029a9a4`, `a213c89`, `761e236`, `70f57db`, `86ca78c` |
| q4_K MMQ line (r5–r37) | Staging-shape search → raw-byte NB kernels → quantize-transpose prepass; the int8 tensor-core prefill path | 12–40 | `d440d16`, `774a116`, `0957a08`, `bfe6bba`, `851a896`, `ba977bf` |
| q6_K + FA + promotion (r38–r60) | q6_K BT kernel, FAP2 register softmax, shared-A dedup, fused producers, `W_exp`/`W_dsc` planes; the verified gate set promoted default-on (1.080×) | 41–64 | `75aabb9`, `d38744d`, `87a75a3`, `cf1ed4b`, `910d967`, `83fee77`, `4cf7c74`, `36a481f`, `57edcf6` |
| Decode campaign D1–D4-4 | Split-K decode attention, hybrid rpw dispatch, fused decode A-quantize, D4-2 correctness fix, D4-4 q6_K dense split plane | 65–76 | `a5af60f`, `22336b2`, `3230b2b`, `b31084c`, `ffce151` |
| D5 → D5-R speculative decoding | New `forward_graph_cached` consumer: draft nt=1 chain + target verify at nt=d+1; verify attention bitwise-equal per position; adaptive depth | 80–104 | `a6b7cf3`, `c3d4bb1`, `0fe132f`, `5471680`, `b3e5dab` |

The retired `CUDA-FOLLOWUP-PLAN.md` was consolidated into step docs 78/79 (commit `ace6242`); its
residual open items are listed in `CUDA_OPTIMIZATION.md` §1.4.

---

## 6. Risks and Open Questions

The plan's original risk table, with its resolution:

| # | Risk | Status |
|---|---|---|
| 1 | Host-scalar `nk` baked into captured graphs → stale attention window | **Resolved**: the causal bound is derived from the device positions buffer inside the attention kernel; no host scalar crosses, and the v1 positions readback never existed in the graph path |
| 2 | Sync/readback inside a capture window corrupts capture | **Resolved**: replay is skipped under trace/viz; `abort_capture` handles node errors; the 7e② incident is recorded in `GPU_SAFETY.md` |
| 3 | `store_kv` layout vs allocator KV region layout mismatch | **Resolved**: KV roundtrip tests (`cuda_rope_kv_attn_roundtrip`, `cuda_kv_f16_roundtrip_attn`) |
| 4 | Legacy default-stream implicit sync is load-bearing | **Resolved**: explicit `sync()` before every D2H |
| 5 | Pool never shrinks → VRAM high-water mark | **Accepted**: same policy as CPU/Metal; `Drop` frees; documented |
| 6 | Q5_K/F32-weight models silently fall back to CPU | **Mostly resolved**: Q5_K/Q5_1/Q5_0 and F32 matmul kernels landed; the gate remains all-or-nothing and logs the failing tensor |
| 7 | RoPE kernel is neox-only | **Accepted**: guard + `Err`; all supported models are neox |
| 8 | No CUDA CI | **Open, deferred** (8h②): device-gated tests skip gracefully; GB10 is the reference bench |
| 9 | F32-activation CUDA vs Q8_0-activation CPU logits differ by design | **Accepted**: gates use greedy-text equality + tolerance classes, never cross-path bitwise |

Newer, still-standing items:

| Item | Status |
|---|---|
| End-of-capture failure leaves that step's outputs undefined (loudly logged, graphs disabled) | **Accepted**: effectively unreachable (no host syncs/readbacks in the window); there is deliberately no re-execution path |
| Weight-registry different-size replace leaks the stale buffer | **Accepted, bounded** by distinct (arch, tensor) shapes; freeing would break live captured graphs |
| Planes cost ~3.27 GB on 7B for the promoted prefill path | **Documented**, per-plane opt-out gates available |
| Open performance leads (q8_1 GEMM-prologue fusion, `rms_nw` roofline, small-od re-tile, fused `ffn_gu`, FA deep-opt) | Tracked in `CUDA_OPTIMIZATION.md` §1.4 and the step-doc index; all sub-bar or step-function |

---

## 7. Verification

### 7.1 Device test suite

`src/graph/cuda_backend.rs` carries the device suite (39 test functions); the device layer's own
gates live in `src/cuda.rs`. Both are device-gated: tests skip when no CUDA device is present.
Categories and the invariants they pin:

- **Capture / replay** — `cuda_graph_replay_bit_parity`, `cuda_prefill_shaped_graph_never_captures`,
  `cuda_multisplit_capture_bit_parity`, `cuda_prefill_capture_defaults_on`,
  `cuda_prefill_capture_bit_parity_pp16_pp300`, `cuda_capture_abort_on_error`,
  `cuda_graph_recaptures_on_pool_gen_change`, `cuda_graph_generation_replay_parity_real_model`,
  `cuda_capture_staging_order_and_fallback`; the device-layer gate
  `cuda_graph_exec_destroy_leaves_no_latched_error` (#145: destroying the `cudaGraphExec_t` must not
  latch an API error) and the env-gated `cuda_sync_surfaces_a_latched_error_as_latched`.
- **Pool / allocator / host transfer** — `cuda_pool_roundtrip`, `cuda_pinned_readback_roundtrip`,
  `copy_across_cpu_to_cuda_and_back`, `kv_persistent_regions_survive_realloc`,
  `cuda_scheduler_chain`, `cuda_row_marginal_bench` (env-gated bench).
- **Per-op parity** — elementwise (`cuda_elementwise_parity`), norms (`cuda_norm_parity`), matmuls
  (`cuda_matmul_parity`, `cuda_kquant_matmul_parity`, `cuda_q5_matmul_parity`, and the per-type MMVQ
  parity tests for q4_K/q6_K/q5_K), attention (`cuda_attn_split_decode_parity`,
  `cuda_verify_attention_nt_invariance`, `cuda_rope_kv_attn_roundtrip`, `cuda_kv_f16_roundtrip_attn`),
  embedding gather (`cuda_embed_getrows_parity`), fused FFN (`cuda_fused_ffn_parity`).
- **KV cell movement (C3/C7b)** — `cuda_copy_cells_moves_overlapping_rows_in_both_directions`
  (rows `[1, 4)` -> `[0, 3)` and then `[0, 3)` -> `[1, 4)`, i.e. two of three rows are read *and*
  overwritten in each direction; the whole buffer is compared against the bytes a copy through a
  temporary would produce. The kernel walks the rows in the order the overlap requires — ascending
  when the run slides down, descending when it slides up — and `kvcache::order_moves` fixes that
  order across several runs).
- **Prefill GEMM / MMQ / FA** — `cuda_prefill_mmq_parity`, `cuda_prefill_f16_gemm_parity`,
  `cuda_q4_0_prefill_q8_0_gemm_parity`, `cuda_fa_prefill_attention_parity` (note: causal,
  single-sequence, `start = 0` — the *windowed* FA mask is covered by
  `cuda_windowed_attention_matches_causal_for_long_windows`, which is what caught the
  `fa_prefill_f16kv` window-limit fault on 2026-09-19), the byte-exact plane
  tests (`cuda_q6k_exp_dense_byte_exact`, `cuda_q6k_dsc_dense_byte_exact`,
  `cuda_q4k_dsc_dense_byte_exact`), `cuda_prefill_fused_b_bitparity`, and
  `cuda_multi_token_matmul_bitwise` (one nt=3 forward bitwise-equal to three nt=1 forwards). The
  eager >48 KiB opt-in has its own gates in `src/cuda.rs`:
  `the_gemm_smem_formula_matches_the_kernel_layout` (pins `gemm_dynamic_smem_bytes` against the
  kernel's byte layout — the value-level arm, so a silent shrink is seen),
  `cuda_prefill_smem_optin_covers_every_launchable_instantiation` (every admitted >48 KiB request
  reads back opted in through `cudaFuncGetAttributes`, every over-limit one is skipped), and
  `the_latched_error_message_never_blames_a_kernel`.

Model-level CUDA coverage:
`cuda_conversation_multiturn_reuse` (Qwen2) asserts that an incremental multi-turn session reusing
the decode graph and appended KV produces the same turn-2 text as a fresh conversation that
re-prefills. `graph_logits_match_forward_real_model`'s comparison helper switches to greedy-token
equality when a CUDA device is present (the 7e① path-identity artifact). The Metal-only tests
(`fused_qkv_matches_unfused_decode`, `fused_qkv_norm_matches_unfused_decode`,
`graph_metal_*`) skip on a Linux CUDA build.

### 7.2 Acceptance gates

| Phase | Gate |
|---|---|
| 7a | alloc/copy roundtrip + `copy_across`; plain (non-CUDA) build untouched, zero nvcc |
| 7b | all per-op parity tests; 0.5B Q4_0 full model: CUDA greedy text == CPU greedy text |
| 7c | E2E table (0.5B / 0.6B / 7B) with throughput; disable-env negatives; graph reuse across decode steps |
| 7d | replay bit-parity; re-capture on `pool_gen` change; long-generation parity; graphs-off A/B identical greedy text |
| 7e+ | per-lever A/B, recorded in `CUDA_OPTIMIZATION.md` and the step docs |

The campaign's verification methodology — the five-gate chain (parity ×3, greedy-32 identity,
interleaved A/B medians with a +1.5% whole-prefill bar, the device suite, the ncu/nsys/SASS protocol)
and its transferable lessons — is `docs/cuda_optimization_steps/77-verification-methodology.md`. Read
it before running any A/B on this engine. Two standing measurement rules: never quote llama-bench
long-context throughput as an attention target without an ncu byte-count or a llama-cli recall
cross-check, and an end-to-end max|Δlogits| ≈ 0.38/0.39 is the inherent class of any
accumulation-order change — argmax + greedy divergence + A/B are the operative gates.

Suite baselines in the campaign records: the Phase-7e entries say "144/0 (CUDA parallel + single),
130/0 plain"; the decode campaign's later rows reach `187/0/3`. The non-CUDA baseline on the current
tree is 145 passed / 0 failed / 3 ignored. Run `cargo test --release` for the plain suite and
`cargo test --release --features cuda` on a device.

### 7.3 Issue #145 verification (GB10, sm_121, CUDA 13.0, driver 580.178.04)

`compute-sanitizer --tool memcheck` over the serial CUDA unit suite is the acceptance gate. Baseline
(before the fix): **36 API errors** — 26 `cudaGraphDestroy` (the wrong destructor for an exec, from
`graph_replay_step` and `Drop`), 9 `cudaGetLastError` observations, 1 `cudaFuncSetAttribute` — over
**490 passed / 0 failed / 31 ignored**. After: **0 errors** over **495 passed / 0 failed / 31
ignored** (the five new gates). `minfer bench` on the 0.5B Q4_K_M goes from printing
`CUDA kernel launch error: 1` between its two loops to printing none, with the eager opt-in's one
skipped instantiation named instead. The device limit the request had to fit is
`cudaDevAttrMaxSharedMemoryPerBlockOptin = 101376 B`; the rejected pre-fix request was 131072 B for
`gemm_f16_nt_kernel_t<256,64,true>` (the corrected request is 122880 B, still over the limit, hence
the deliberate skip). The mutation checks are recorded in
`docs/ARCHITECTURE-EXECUTION-PLAN.md` (C4 S2c).

---

## 8. Out of Scope / Future

- **Not planned** (revisit with a concrete need): cuBLAS/cublasLt, VMM pool, multi-GPU + peer copies,
  `graph_optimize`-style node reordering, Windows, self-hosted CUDA CI, IQ/Q2/Q3 quants.
- **Open leads, all sub-bar or step-function** (details in `CUDA_OPTIMIZATION.md` §1.4): the q8_1
  GEMM-prologue fusion (the identified step change); the `rms_nw` roofline (+0.5–1%); a wave re-tile
  for small-`od` classes (+0.3–0.8%, needs ≤85 registers); a fused `ffn_gu` concat (needs the G5
  `nf <= 16384` gate re-measured); FA deep-opt only with a numerics-order-preserving structure.
- **Inherited Phase-8 ledger**: the shape-dependent q6_K MMVQ idle-tail rule (not started), the CUDA
  CI runner (deferred), and the `/tmp/minfer_phase7/` ledger cleanup (awaiting a decision).
- **Landed since the original plan** and therefore no longer in this list: prefill CUDA-Graph capture,
  FA-style prefill attention, fused decode nodes on CUDA, pinned host buffers, f16 KV, Q5_K/Q5_1/Q5_0
  and F32 kernels.

