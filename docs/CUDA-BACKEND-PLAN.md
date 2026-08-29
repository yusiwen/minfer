# CUDA Graph Backend Plan (Phase 7)

> Status: **plan** (not started). Extends `docs/GRAPH-REFACTOR-PLAN.md` §9/§14
> Phase 7 with the concrete design. Prerequisite already landed: the CUDA build
> chain rework (`081cf24` — opt-in `--features cuda`, auto-detected nvcc +
> `-ccbin`, multi-arch SASS/PTX incl. native sm_121, cudart rpath).
> Reference material: llama.cpp master (`ca3d5a3e1`) ggml-cuda backend, and the
> two in-repo backend implementations (CPU / Metal).

---

## 1. Goal & Non-Goals

**Goal.** Implement `CudaBackend` (`src/graph/cuda_backend.rs`) as the third
backend of the compute graph, by *wrapping* the existing `src/cuda.rs` draft
(per §9: "wrap, do NOT stub") so that on a CUDA machine the whole per-layer
chain runs on GPU through the standard
`build → assign → fuse → alloc → execute` pipeline, with correctness gates
identical to the Metal phases (greedy text equality vs CPU at temp 0).

**Non-goals (v1).** Multi-GPU / tensor split (llama.cpp §6 machinery — out of
scope for a single-GPU engine); FP16/BF16 activations (kernels are f32-act);
Q5_0/Q5_1/Q5_K/IQ*/Q2_K/Q3_K kernels (do not exist in `cuda_kernels.cu`);
`Flash` attention mode changes; Windows validation; CUDA CI runners.

---

## 2. Current State (inventory)

### 2.1 What already works

| Piece | Location | Note |
|---|---|---|
| Build chain: opt-in feature, nvcc/`-ccbin` probe, arch test-and-skip sm_61…sm_121, cudart link + DT_RPATH | `build.rs` | landed `081cf24` |
| Device init: `MINFER_DISABLE_CUDA` off-switch, device probe, auto-pick highest CC, stream | `cuda.rs:266-307` | mirrors `MINFER_DISABLE_MPS` |
| Weight registry: `register_weight(name, &[u8])` → cudaMalloc + H2D; `has_weight` / `get_weight_ptr` | `cuda.rs:383-433` | loader already calls it (§2.3) |
| Copy helpers: sync/async H2D/D2H/D2D, `sync()` | `cuda.rs:469-536` | |
| Per-op kernels: matmul Q4_0/Q4_1/Q8_0/Q4_K/Q6_K (f32 act), rms_norm, rope, silu, swiglu, add, add_bias, mul, store_kv, gqa_attn, quantize_q8_0 | `cuda_kernels.cu:796-949` | all take trailing `cudaStream_t` |
| CUDA Graph primitives: `graph_begin_capture` / `graph_end_capture` / `graph_launch` | `cuda.rs:675-740` | capture on our stream, mode 1 (ThreadLocal) |
| Loader registration (qwen2 + qwen3): `cuda.register_weight` for Q4_0/Q4_1/Q8_0/Q4_K/Q6_K + F32 | `qwen2/loader.rs:221-237`, `qwen3/loader.rs:241-256` | Q5_* deliberately absent (no kernels) |
| Graph placeholders: `Backend::Cuda` enum, DOT color, JSON name | `graph/mod.rs:60-62`, `dot.rs:22`, `json.rs:131-136` | |
| `ComputeGraph.uid` reserved for CUDA-Graph caching | `graph/mod.rs:109-111` | never written yet |

### 2.2 Placeholder arms to replace (exact sites)

| Site | Today | Becomes |
|---|---|---|
| `alloc.rs:254` `alloc_in_pool` | `unreachable!("CUDA pool not implemented")` | pool alloc |
| `alloc.rs:269` `free_in_pool` | `{}` | free-list return |
| `alloc.rs:364` `fill_input_impl` | `Err("CUDA unavailable")` | `write_host` H2D |
| `alloc.rs:392` `copy_to_cpu` | `None` | sync D2H → `Vec<f32>` |
| `alloc.rs:434` `sync_backend` | `{}` | `CudaState::sync()` |
| `alloc.rs:459` `copy_across` dst | `Err("CUDA unavailable")` | generic path works |
| `scheduler.rs:241` dispatch | `Err("CUDA backend not implemented")` | real arm |
| `qwen2/graph.rs:396-411`, qwen3 mirror + `json.rs:72-75` | FusionPass `backend_of` maps CPU→0, Metal→1 | + CUDA index |
| `alloc.rs:91-102` `supports()` | Metal → CPU | Metal → CUDA → CPU |

### 2.3 Draft API triage (wrap vs drop)

- **Usable as-is** (the wrapper surface): `get/init`, `has_weight/register_weight/get_weight_ptr`, `stream`, `copy_to_device/copy_from_device/copy_device_to_device`, `sync`, `quant_matmul_f32_on_gpu`, `rms_norm`, `rope_f32`, `gqa_attn_f32`, `store_kv_f32`, `add_f32/add_bias_f32/mul_f32/silu_f32/swiglu_f32`, `graph_begin_capture/graph_end_capture/graph_launch`.
- **Legacy, not wrapped** (old imperative forward, no live callers): `layer_gpu`, `output_norm_gpu`, the 12 named activation slots + `get_or_grow`, `upload/download_hidden/positions/logits`, `init_kv_cache` + `kv_k/kv_v/kv_size`, `quant_matmul_q8/quant_matmul_f32(_batch)` (host-staged H2D→launch→sync→D2H per call). Keep compiled behind `#![allow(dead_code)]` until 7e cleanup; `main.rs:614-618` legacy KV pre-allocation is removed in 7c (it would waste up to ~1.2 GB VRAM on qwen3-4B-sized models and the allocator owns KV now).

---

## 3. llama.cpp Reference Map

What we borrow (llama.cpp file:line → minfer analog):

| llama.cpp concept | Where | minfer analog |
|---|---|---|
| Backend iface (`graph_compute`, `synchronize`, async tensor set/get, `supports_op`) | `ggml-backend-impl.h:106-141`, `ggml-cuda.cu:4575-4592` | `Backend` trait (`graph/backend.rs:21-58`) — per-node instead of per-graph |
| Weights placed in device buffers at load; ops follow their weights; **never host-copied during execution** | `llama-model.cpp:1747` | `register_weight` at loader (already wired) + name→ptr lookup in `execute_node` |
| Device buffer pool, alloc-free hot path | `ggml-cuda.cu:420-533`, `common.cuh:1167-1216` | `CudaBackend` pool with free-list (mirror `metal_backend.rs:288-304`) |
| `graph_compute` enqueues; sync only at scheduler boundaries | `ggml-cuda.cu:4301-4303`, `:2534-2540` | `execute_node` launches on the one stream; `synchronize()` = `cudaStreamSynchronize` |
| CUDA Graph replay: keyed per graph, **warmup ×2 then capture**, node-props memcmp (`uid` fast path + src-pointer/sha snapshot), `cudaGraphExecUpdate` whole-graph on change, re-instantiate on update failure | `ggml-cuda.cu:2585-2653`, `:4247-4304`, `common.cuh:1231-1262` | §4.8 state machine keyed on `(uid, node_range, pool_gen)` |
| Capture disqualifiers: ops needing host sync; arch < Volta; env off-switch | `ggml-cuda.cu:2547-2579`, `:4235-4240` | §4.8 constraints; `MINFER_NO_CUDA_GRAPH=1` |
| Replay precondition = **stable buffer addresses across steps** | `ggml-cuda.cu:2618-2621` | already encoded: `GraphCache` keeps the allocator (and pool ptrs) alive across decode steps (`cache.rs:9-12`) |
| Fusion pass (`ggml_cuda_try_fuse` patterns) | `ggml-cuda.cu:3277-4010` | out of scope v1; minfer's `FusionPass` + `supports_fused` plays the same role |

What we deliberately skip: multi-GPU split/NCCL, VMM pool, pinned host buffers
(optimization, 7e), cuBLAS paths (minfer has its own dot kernels), `graph_optimize`
QKV reordering, and llama.cpp's `CUDA_CHECK`-aborts-on-capture-failure policy —
minfer falls back to direct launches instead (§4.9).

---

## 4. Design

### 4.1 CudaBackend struct

```rust
// src/graph/cuda_backend.rs  (#[cfg(feature = "cuda")])
pub struct CudaBackend {
    state: &'static crate::cuda::CudaState,   // global singleton (cuda.rs:239)
    pool: Vec<CudaBuf>,        // id -> { ptr: *mut c_void, bytes: usize }
    free: Vec<usize>,          // free-list, byte-length matched (metal pattern)
    pool_gen: u64,             // bumped on every external (re)alloc pass; replay-invalidations
    weights_ok: bool,          // set by the model wiring gate
}
```

- `new() -> Option<Self>` = `CudaState::get()?` (device probe + `MINFER_DISABLE_CUDA` already inside `try_new`, `cuda.rs:266-277`); mirrors `MetalBackend::new()` (`metal_backend.rs:130-140`).
- `alloc_buffer(size)` (size = f32 elements): byte length `size*4`, free-list exact match else `cudaMalloc`; **never `cudaFree` on `free_buffer`** (cache, like CPU/Metal) — only `Drop` frees everything. Persistent KV regions survive rebuilds because the allocator only frees liveness-expired buffers (`alloc.rs:109-115`).
- `pool_gen += 1` whenever `alloc_graph` runs (allocator calls `begin_cycle()`-style hook, or gen is bumped inside `alloc_in_pool` on first alloc after a reset — implementation detail, see 7d).
- `unsafe impl Send/Sync` like Metal (`metal_backend.rs:75-76`) — raw pointers wrapped, all mutation behind `&mut self`.

### 4.2 Backend trait mapping

| Trait method | Implementation |
|---|---|
| `name()` | `"cuda"` |
| `supports_op(op, dtype)` | §4.3 matrix |
| `supports_fused(fused)` | `SwiGLU → true` (kernel exists); `QKVBiasRopeStore/BiasRope/BatchMatMul → false` |
| `alloc_buffer/free_buffer` | pool §4.1 |
| `execute_node(node, in_bufs, out_buf, kv_pair)` | §4.4 dispatch |
| `read_host(id)` | **`None`** — staged D2H cannot return a borrow (trait takes `&self`); the only consumer `copy_to_cpu` gets a direct-D2H arm instead (`alloc.rs:392`) |
| `write_host(id, data)` | `copy_to_device` (sync H2D), byte-length checked |
| `synchronize()` | `CudaState::sync()` (`cuda.rs:531`) |

### 4.3 `supports_op` matrix (v1)

All ops require `dtype == DType::F32` (activation dtype; weight type rides in
`MatMulMeta.weight_ttype`) — same convention as CPU (`cpu_backend.rs:64-66`)
and Metal (`metal_backend.rs:265-280`).

| Op | CUDA v1 | Kernel |
|---|---|---|
| `Input`, `KvcacheLoad`, `View/Reshape/Permute` | ✓ | no-op / identity D2D (`copy_device_to_device`) |
| `Add`, `Mul` | ✓ | `launch_add_f32`, `launch_mul_f32` |
| `Silu` (in-place) | ✓ | `launch_silu_f32` |
| `SwiGLU` | ✓ | `launch_swiglu_f32` |
| `RmsNorm` | ✓ | `launch_rms_norm_f32` (weight required) |
| `QkNorm` (qwen3 per-head) | ✓ | loop `n_head` × `launch_rms_norm_f32` with `w_ptr + h*hd` offset |
| `MatMul` (weights Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) | ✓ | `quant_matmul_f32_on_gpu` (`cuda.rs:768`) + optional `launch_add_bias_f32` |
| `MatMul` (F32 weights) | ✗ (7e candidate kernel) | — |
| `RoPE` | ✓ | `launch_rope_f32` — **neox-style only** (kernel has no style param; all supported models are neox; `Err` otherwise) |
| `Attn` (Gqa/Flash) | ✓ | `launch_gqa_attn_f32` with guards `hd_kv == hd` (kernel strides KV by `nk*hd`) |
| `KvcacheStore` | ✓ | 2 × `launch_store_kv_f32` (K, V) |
| `Embed`/`GetRows`, `Scale`, `Softmax`, fused decode ops | ✗ (stay CPU) | no kernels; `Embed`+tail-`GetRows` on CPU produce exactly 2 host round trips per forward (§6) |

Positions are `I32` inputs stored as `f32::from_bits` (`fill_input_i32`,
`alloc.rs:329-337`); kernels reinterpret the buffer as `*const i32` — bit-exact
for `|v| < 2^24`, same trick the CPU backend uses (`cpu_backend.rs:136-139`).

### 4.4 `execute_node` dispatch rules

- **Weights by name** → `get_weight_ptr`; missing → `Err("weight '..' not on GPU")` (Metal pattern, `metal_backend.rs:464-467`). Buffer ids → pool ptrs; unknown id → `Err`.
- **In-place alias rule** (Silu/RoPE may have `out_buf == in_bufs[0]`, `alloc.rs:201-229`): if not aliased, stage with D2D copy first (`copy_in` mirror of `metal_backend.rs:324-331`, `525-546`), then run the in-place kernel on `out_buf`.
- **`KvcacheStore`** requires `kv_pair` and `out_buf == k_id` (CPU contract, `cpu_backend.rs:130-132`); writes K into `k_id`, V into `v_id` from `srcs` (Metal pattern `metal_backend.rs:557-579`); `pos >= n_ctx` check needs positions on host → see Attn readback below (do one combined readback per node where needed).
- **`Attn`** needs `nk = max(positions)+1` on the host today (kernel arg, `cuda.rs:1005`): v1 does a sync D2H readback of the positions buffer (nt·4 bytes) inside `execute_node`, computes `nk` like the CPU backend (`cpu_backend.rs:394-398`), then launches. 7d removes this (§4.8) because a host→baked scalar breaks replay.
- **`KvcacheLoad`** is a view: `out_buf` *is* the K region — no launch (CPU: `cpu_backend.rs:365`).
- **Guards return `Err`, never fallback** (GPU_SAFETY rule, `docs/GPU_SAFETY.md:62-64`): missing meta, missing weight, `hd_kv != hd`, non-neox rope, unsupported op shape.
- Every launch site checks errors via the existing `cudaGetLastError` wrapper (`cuda_kernel_check`, `cuda.rs:92`) instead of ignoring them.

### 4.5 Allocator / scheduler integration

Replace the §2.2 placeholder arms; priority order in `supports()`
(`alloc.rs:91-102`): **Metal → CUDA → CPU** (mutually exclusive platforms in
practice, so order is nominal). `enable_cuda()` mirrors `enable_metal()`
(`alloc.rs:56-61`). `copy_across` works unchanged once `copy_to_cpu`/`write_host`
exist — cross-backend = host round trip (`alloc.rs:441-473`), exactly like CPU↔Metal.

KV regions: `ensure_kv`/`alloc_persistent` (`alloc.rs:275-294`) allocate the
per-layer K/V regions **on the layer's assigned backend** — with CUDA enabled
and layers assigned to CUDA, KV lives in the CUDA pool automatically.
`KvProvider::kv_pair` returns pool ids; `execute_node` resolves them to device
ptrs. The legacy `init_kv_cache` path is bypassed entirely.

### 4.6 Model wiring (qwen2 + qwen3, identical shape)

Mirror the Metal gate (`qwen2/graph.rs:356-379`, `qwen3/graph.rs:344-361`):

```text
cuda_on = cuda_available() && weights_on_cuda(model)     // #[cfg(feature = "cuda")]
metal_on = metal_available() && weights_on_gpu(model)    // existing
CParams.gpu = metal_on || cuda_on
weights_on_cuda: every graph-referenced weight has cuda.has_weight(name),
                 EXCLUDING tok_embd (Embed/GetRows stay on CPU in v1)
```

`weights_on_cuda` mirrors `weights_on_gpu` (`qwen2/graph.rs:501-549`). If any
non-excluded weight is missing (e.g. Q5_K weights, F32 `output`), `cuda_on` is
false → full CPU graph — the all-or-nothing rule avoids partial-GPU splits.
`cuda_available()` = `CudaState::get().is_some()` (mirror
`metal_backend.rs:1014-1016`). FusionPass gets the CUDA backend pushed into its
`Vec<&dyn Backend>` with an index in `backend_of` (`qwen2/graph.rs:396-411`,
qwen3 mirror, `json.rs:72-75`). Loader registration already exists (§2.1) —
nothing to do there for v1. Fused decode concat weights (`blk.{i}.attn_qkv`,
`ffn_gu`) are **not** registered on CUDA in v1 (no fused kernels to consume
them); `qkv_concat_available`/`gu_concat_available` stay macOS-only so the
builder never emits fused decode nodes for CUDA (`qwen2/graph.rs:230-267`).

### 4.7 Memory-correctness notes

1. **Sync before D2H.** `copy_from_device` is a synchronous `cudaMemcpy` on the
   NULL stream (`cuda.rs:492-501`) — correct today only because the legacy
   default stream implicitly synchronizes blocking streams (`cudaStreamCreate`,
   `cuda.rs:303`). The backend wraps all device reads with an explicit
   `state.sync()` first so correctness never hinges on implicit semantics (and
   the async-copy optimization in 7e becomes possible).
2. **No host copies of weights** — kernels take device ptrs from the registry;
   only activations/positions/logits cross PCIe (tiny for decode).
3. **Stream discipline**: one stream for everything (llama.cpp default flow is
   single-stream too, `common.cuh:1420`); no events needed in v1.
4. **Capture-window hygiene** (7d): while capturing, `execute_node` must not
   sync or read back — `debug_sync` (`cuda.rs:545`) and the Attn readback get a
   `capturing` guard; all launch args must be device-resident data.

### 4.8 CUDA Graph capture / replay (phase 7d — IMPLEMENTED, commit 7d)

llama.cpp's state machine, adapted (✅ = as designed; ⚠ = implementation note):

- ✅ **Key**: `(graph.uid, node_range, pool_gen)` per captured CUDA split.
  `ComputeGraph.uid` is populated in `GraphCache::replace_graph` (monotonic
  counter starting at 1; reuse keeps it — llama.cpp `ggml_graph_next_uid`
  semantics, `ggml.c:56-68`).
  `pool_gen` plays the role of llama.cpp's node-props memcmp: any pool
  (re)allocation invalidates the stored exec (checked lazily at replay time,
  before a stale exec could ever launch; pointers never move, so this is
  conservative).
- ✅ **Warmup ×2**: executions 1-2 of a `(uid, range)` run direct launches;
  the 3rd opens the capture window (llama.cpp `ggml-cuda.cu:4267-4286`).
  One-shot prefill graphs never reach capture — zero overhead there.
- ✅ **Capture**: `cudaStreamBeginCapture(ThreadLocal)` around the split's
  node loop; `EndCapture` + `cudaGraphInstantiate` at the split's
  `synchronize` (the window must close before the next split's host I/O).
  The captured launches do not execute during capture, so the backend
  **launches the instantiated graph once at close** — the capture run itself
  produces its outputs.
- ✅ **Replay**: subsequent executions with the same key call
  `cudaGraphLaunch` instead of the node loop; input staging buffers were
  H2D-filled before the split at stable addresses (`copy_across` rewrites
  them per step), so replay reads fresh data (`ggml-cuda.cu:2618-2621`).
- ✅ **Trait hook** (additive, default no-op — CPU/Metal untouched):
  `fn graph_replay(&mut self, uid: u64, range: (usize, usize)) -> bool`;
  the scheduler skips the CUDA split's node loop on `true`. ⚠ Replay is
  force-disabled while `MINFER_TRACE`/`--viz` capture is active (per-node
  host readbacks inside a capture window are illegal — they would corrupt
  the recorded graph).
- ✅ **Prerequisite kernel change: already in place.** `gqa_attn_f32` derives
  the per-token KV bound from the device-side `positions` buffer
  (`nkv = positions[t] + 1`); its `nk` parameter is the KV-head count (a
  stride), not the window — no host scalar is baked, and the v1 positions
  readback never existed in the graph path.
- ✅/⚠ **Failure policy**: `MINFER_NO_CUDA_GRAPH=1` forces off at backend
  construction; a begin-capture or replay-launch failure logs a warning and
  disables graphs for the session (inference continues with direct
  launches). ⚠ Deviation for end/instantiate failure: the recorded launches
  never executed, so that step's split cannot produce output — the backend
  logs loudly, disables graphs, and the step's outputs are undefined
  (effectively unreachable: the window contains only kernels + async D2D;
  host syncs and readbacks are outside it). Accepted rather than inventing
  a silent re-execution path.

Implementation notes (what the parallel test suite forced into existence):

- **Process-wide stream lock** (`CudaState::stream_lock`): stream capture is
  **per-stream, not per-thread** — while one backend holds an open window,
  any other thread's enqueues to the shared stream (fills, copies, launches,
  even another test's `execute`) would be recorded into that graph. The
  capturing backend holds the lock across its whole window (stored
  `MutexGuard`); its own enqueues skip re-locking; every other stream-touch
  (`execute_node`, `write_host`, `copy_to_host`, allocs/frees, plain sync)
  takes it per call. Production runs one engine thread — uncontended.
- **Weight-registry hardening** (`register_weight`): same name + size ⇒
  reuse the existing device copy (unit tests reload the same GGUF and were
  leaking a full device model per load — 100+ OOM aborts when the parallel
  suite finally exercised registration everywhere); different size ⇒ replace
  but never free (a live captured graph may still reference it; the leak is
  bounded by distinct shapes). The gate (`weights_on_cuda`) is size-aware, so
  a foreign-architecture entry reads as "not registered" and that model
  cleanly stays on CPU.
- **`ModelLoadGuard`** (reentrant, process-wide): loaders hold it while
  registering; real-model graph tests hold it across their forwards. Without
  it, a parallel load of another architecture lands mid-test and flips the
  CUDA gate between forwards — mixing CPU-allocated persistent KV regions
  with CUDA assignment (`node buffer on CPU but executing split is Cuda`).
  Pre-7d the parallel suite was green only because the init race made most
  test loads skip device registration entirely.
- **Capture-safe D2D**: `copy_device_to_device` switched from legacy-sync
  `cudaMemcpy` (illegal inside a window) to `cudaMemcpyAsync` — a landmine
  no current graph hits (View/Reshape/Permute + aliased in-place ops), but
  one code path away.

### 4.9 GPU safety (CUDA edition)

- Kernel-invariant violations → `Err` from `execute_node`; scheduler aborts
  (`scheduler.rs:230-242`); **no silent CPU fallback** (`GPU_SAFETY.md:62-64`).
- CUDA's fault model is friendlier than Metal's (no machine-freezing
  compositor): faults surface as errors on the *next* API call — hence the
  §4.4 `cudaGetLastError` checks after launches (`cuda_kernel_check`).
- `synchronize` = blocking `cudaStreamSynchronize` (bounded by kernel runtimes;
  no unbounded host wait possible on a healthy device). Document the deviation
  from Metal's 10 s bounded submit in `GPU_SAFETY.md` (7e doc update).
- Device limits (SM count, mem) queried at runtime (`cuda.rs:309-342`), never
  hardcoded — rule already satisfied by the draft.

---

## 5. Implementation Phases

### 7a — Skeleton + wiring (compiles, allocates, copies)

Files: new `src/graph/cuda_backend.rs`; `graph/mod.rs` (module decl);
`alloc.rs` (§4.5 arms + `enable_cuda` + `supports()`); `scheduler.rs:241`.

- CudaBackend struct/pool/trait impl; `execute_node` handles only `Input`, everything else `Err("cuda: op not implemented yet")`.
- Tests (device-gated: skip when no CUDA device, like Metal tests gate on macOS):
  pool alloc/write/read roundtrip; `copy_across` CPU↔CUDA both directions;
  KV persistent-region survival across `alloc_graph` re-runs.

### 7b — Per-op execution + parity tests

Files: `cuda_backend.rs` (full §4.4 dispatch); `cuda.rs` (un-`allow` the used methods; error checks at launch sites).

- Per-op parity tests (Phase-3 Metal pattern): elementwise/rms_norm bit-identical vs `vec_ops`; matmuls (each quant type) vs CPU Q8_0-activation reference within tolerance and vs themselves bit-identical; RoPE vs `cpu_rope`; KV store/load roundtrip incl. positions ≥ 1 (n_past growth); GQA attention vs `cpu_gqa_attn`; qwen3 `QkNorm` head-loop vs CPU.
- Whole-layer chain test: one layer's full node sequence on CUDA vs CPU (tolerance), then full `forward_graph` 0.5B Q4_0 CUDA-vs-CPU logits (f32-activation tolerance class) and **greedy text equality**.

### 7c — Model wiring + E2E

Files: `qwen2/graph.rs`, `qwen3/graph.rs` (cuda gate, `weights_on_cuda`,
FusionPass/backend_of); `json.rs:72-75`; `main.rs:614-618` (remove legacy KV
pre-alloc); `AGENTS.md` (GPU matrix row).

- E2E on GB10 (sm_121): qwen2.5-0.5B Q4_0, qwen3-0.6B Q8_0, qwen2.5-7B Q4_K_M — greedy output == CPU greedy output; prefill→decode→multi-turn session (KV growth, graph reuse across steps); tok/s vs CPU baseline (and vs the Metal numbers in AGENTS.md for context).
- Negative paths: `MINFER_DISABLE_CUDA=1` → CPU; missing device → CPU; Q5_K model → CPU (gate).

### 7d — CUDA Graph capture/replay

Files: `cache.rs` (uid population); `cuda_kernels.cu` (gqa_attn device-derived
bound, §4.8); `cuda_backend.rs` (capture state machine); `backend.rs` +
`scheduler.rs` (replay hook, default no-op); `cuda.rs` (expose capture safety
flag).

- Tests: replayed decode logits **bit-identical** to direct-launch runs; re-capture triggered by pool_gen change (params switch prefill↔decode); 200-token generation with growing KV (validates the nk fix); `MINFER_NO_CUDA_GRAPH=1` A/B; capture-failure injection → session fallback (still-correct output).

### 7e — Polish + optional perf (each item A/B measured, independently landable)

- ✅ **CPU-path residual (7e①, RESOLVED — not a bug)**: `graph_logits_match_forward_real_model` diff **0.449** (prefill) / **0.525** (decode) diagnosed as a *path-identity artifact*: since Phase 6 `model.forward` routes through the graph path, the test's "reference" is itself a graph — on CUDA builds it takes the CUDA graph (the manual side is CPU-only), so the diff is cross-backend f32 reduction-order noise (AGENTS §9 class), not a CPU-path defect. macOS's 0.0 was just "no CUDA feature → both sides CPU". Hypotheses (NEON split, Q8_K handling, worker pool) all disproved: the diff survived `MINFER_NO_NEON=1` and `set_cpu_threads(1)` unchanged. Greedy tokens agree on both steps (12095 / 11). Fix: `compare` mirrors the Metal precedent — strict `1e-3` on CPU-only builds, greedy-token equality when the engine side runs CUDA. Recorded in `docs/KNOWN-CPU-ISSUES-2026-08-29.md`.
- ✅ **Vectorized q4_K/q6_K kernels (7e②, 2026-08-29)**: 7B decode **8.4 → 26.4 tok/s (3.1×)**, 0.5B 241 tok/s (unchanged, q4_0 untouched). Three combined changes, each validated by a standalone nvcc A/B against the original kernel plus a new `cuda_kquant_matmul_parity` test (independent in-test dequant reference; q6_K previously had NO parity coverage — the gap that let a broken intermediate variant pass the suite): (1) **Q6_K padded registration** — `register_weight_q6k_padded` copies each 210-byte block into a 224-byte slot (224 = 14×16) so the weight stream can use 16-byte-aligned uint4 loads (the raw 210-byte stride forced 1-byte-per-instruction reads); dispatch via `matmul_f32_ptr_layout(.., padded_q6k)`; `has_weight_of_size` matches padded entries by their ORIGINAL raw length so the weights gate stays correct. (2) **Unit lane mapping for q4_K** — lanes own (row, super-block) pairs (NR0=4, 56 units/warp) instead of one block per lane: the 7B FFN shapes have nbe=14 blocks/row, which idled 18/32 lanes in the lane-per-block layout; measured 42 → 163 GB/s on the ffn_gu shape. (3) **float4 activation loads** in both kernels. Debugging lessons recorded: a temporary sync inside the matmul dispatch silently corrupted graph capture windows (garbage output whenever graphs were enabled) — never sync inside `execute_node`; and "faster but wrong" (the v-selector bug read ¼ of the y values) is a red flag for load-count changes, not a perf win.
- ✅ **Embed + GetRows on device (7e③, 2026-08-29)**: the embedding gather
  (Op::GetRows with Embed meta — `builder.embedding()` emits GetRows, there
  is no separate Op::Embed) and the generic f32 tail gather (G3, no meta)
  run as CUDA kernels, removing the prefill/decode CPU round trips around
  them — prefill and decode graphs are now a SINGLE CUDA split (no
  cross-backend copies at all). One thread per output sub-block per weight
  type (F32/Q8_0/Q4_0: 32-elem groups; Q4_K: 32-elem sub-blocks of the
  super-block, scale = sub-block index; Q6_K: 16-elem sub-blocks, with a
  `block_stride` parameter covering both the raw 210-byte and padded
  224-byte registrations). ids are I32-as-f32 BIT patterns read with
  `__float_as_int` (graph rule §4) — `__float2int_rn` would read the
  denormal float value and always yield 0. minfer Q4_0 stores
  `round(v/d) + 8` — the embed kernel needs the same `-8` offset as the
  matmuls and the CPU embed path (a plain `d*nibble` version passed the
  suite's q4_0-free coverage and produced garbage only on q4_0 models —
  re-run E2E per quant after touching a shared kernel; unit tests with a
  single quant do not cover the convention). `weights_on_cuda` now gates
  tok_embd like every other weight (embed-kernel types F32/Q4_0/Q8_0/
  Q4_K/Q6_K; Q4_1-embd models are unaffected — their Q5_K matmul weights
  already keep them CPU-only). Measured: prefill 300-token 7B prompt
  9.42→9.18 s (≈1.5% — matmul-dominated; the structural win is the
  single-split graph), decode unchanged (26.2 tok/s 7B, 248 tok/s 0.5B).
  Parity: `cuda_embed_getrows_parity` covers all 6 embed types incl. the
  padded Q6_K layout against `kernel::embed_tokens`, plus a model-shape
  q4_0 case (n_embd=896, ids up to 9625) and the generic gather.
- Remaining prefill perf work: FusedFFN (7e⑤) and F32×F32 matmul (7e④).
- ✅ **F32×F32 matmul kernel (7e④, 2026-08-29)**: `f32_f32_matmul_vec`
  (same unit lane mapping as the q4_K kernel — lanes own (row, 256-elem
  chunk) pairs, float4 loads; requires `id % 8 == 0`) plus
  `f32_f32_matmul_scalar` (thread per output element) for general dims.
  Dispatch: `TensorType::F32` arm in `matmul_f32_ptr_layout`; the 32-byte
  quant-block guard now exempts F32 weights; `matmul_ok` gates admit F32
  matmul weights, so unquantized / F32-head models participate in CUDA.
  Kernel lesson (2nd of the phase): in the unit mapping the unit's lane
  must stream its WHOLE chunk — a leftover `i = lane_id * 8` inside the
  unit silently computed only 1/64 of each dot (the same-lane-count
  structure from the quant kernels does not transfer). Validated by
  standalone nvcc A/B (bit-exact vs CPU double-accumulation at
  8×512×3) and the extended `cuda_kquant_matmul_parity` (aligned + odd-id
  cases). No cached F32-weight GGUF — E2E F32-model validation pending
  the first such model.
- ✅ **FusedFFN for CUDA (7e⑤, 2026-08-29)**: `cuda::concat_rows` (the
  `metal::concat_rows` analog — raw row-major concat of same-type,
  same-in-dim weights); loaders register `blk.{i}.ffn_gu` on CUDA (Q6_K
  concats go through the padded repack). New `swiglu_f32_off` kernel:
  in-place `buf[i] = silu(buf[i]) * buf[off+i]` over the concat matmul
  output (gate rows 0..nf, up rows nf..2nf); `Op::FusedFFN` supported and
  executed (concat matmul → offset swiglu; down reads rows 0..nf).
  Topology: `CParams.fuse_ffn` decouples the FFN fusion from the QKV
  fusion gate in BOTH models (mirrors Qwen3's existing intent) — the
  `MINFER_NO_FUSE_FFN` toggle is now part of the reuse identity, so it
  reliably forces a rebuild (Qwen2 previously keyed FFN fusion off
  `fuse_qkv`, a Metal-only gate that silently disabled fusion on CUDA).
  Qwen2/Qwen3 `gu_concat_available` gains the CUDA branch, enabling FFN
  fusion on CUDA for the first time. Parity: `cuda_fused_ffn_parity`
  (q4_K plain concat + q6_K padded concat vs host silu(gate·x)·(up·x),
  independent in-test dequant). E2E: 0.5B q4_0 226.7 → 249.6 tok/s
  (+10%), Qwen3-0.6B q8_0 182.4 → 190.0 tok/s (+4%); 7B unchanged
  (nf = 18944 > 16384 gate, unfused). Suites 144/0 (cuda parallel +
  single), plain 130/0.
- FusedQKV decomposition (concat matmul + bias/rope/store chain under one node) — only if it beats the unfused chain.
- ✅ **Async H2D input fill + pinned staging (7e⑥, 2026-08-29)**:
  `CudaState::write_input_async` — a lazy ring of 8 × 2 MiB
  `cudaHostAlloc` slots (`cudaFreeHost` on drop); `write_host` now copies
  the Rust slice into a pinned slot and queues `cudaMemcpyAsync` on the
  stream (returns before the copy lands; same-stream ordering guarantees
  the fill completes before the kernels that read it). Pageable
  `cudaMemcpy` had blocked the CPU until completion and forced the driver
  through its own bounce buffer. Ring wrap (>8 fills without a sync)
  triggers one stream sync before slot reuse — never hit in practice
  (per-step input fills are KB-scale). Sync fallback for >2 MiB fills or
  if pinned allocation fails. E2E unchanged-or-better (0.5B ~248,
  Qwen3-0.6B ~188 tok/s — H2D per step is KB-scale so the win is the
  removed CPU stall, µs; 7B n=128 21.7–23.6 tok/s, no regression).
  Suites 144/0 (cuda parallel + single), plain 130/0.
- Docs: `GRAPH-REFACTOR-PLAN.md` §17 Phase 7 row → ✅ (replace the stale "本机无 nvcc" blocker note); `GPU_SAFETY.md` CUDA section; prune `#![allow(dead_code)]` in `cuda.rs` to the still-legacy surface.

---

## 6. Risks & Open Questions

| # | Risk | Mitigation |
|---|---|---|
| 1 | Host-scalar `nk` baked into captured graphs → stale attention window | 7d kernel change (device-derived bound) + bit-parity + long-gen test |
| 2 | Sync/readback inside a capture window corrupts capture | `capturing` flag gates `debug_sync` + readbacks; window-hygiene audit checklist in 7d |
| 3 | `store_kv_f32` layout vs allocator KV region layout mismatch | dedicated 7b roundtrip test (region `[n_ctx][nkt]` row-major vs kernel `dst + pos*nkt` stride) |
| 4 | Legacy-default-stream implicit sync is load-bearing | explicit `sync()` before all D2H in the backend (§4.7.1) |
| 5 | Pool never shrinks → VRAM high-water mark | acceptable (same as CPU/Metal pools); document; `Drop` frees |
| 6 | Q5_K/F32-weight models silently fall back to CPU | gate is all-or-nothing by design; log the reason; 7e kernels lift it |
| 7 | RoPE kernel is neox-only | guard + `Err`; all supported models are neox |
| 8 | No CUDA CI | device-gated tests skip gracefully; GB10 is the reference bench; optional CUDA GitHub runner later |
| 9 | F32-activation CUDA vs Q8_0-activation CPU logits differ by design | same accepted divergence as Metal (AGENTS.md §9.9); gates use greedy-text equality + tolerance classes, not cross-path bitwise |

---

## 7. Verification Matrix (acceptance)

| Phase | Gate |
|---|---|
| 7a | alloc/copy roundtrip + copy_across tests pass (feature build); plain build untouched, zero nvcc |
| 7b | all per-op parity tests; 0.5B Q4_0 full-model: CUDA greedy text == CPU greedy text |
| 7c | 3-model E2E table (0.5B/0.6B/7B) with tok/s; disable-env negatives; graph reuse across decode steps (params-only, no rebuild) |
| 7d | ✅ replay bit-parity (`cuda_graph_replay_bit_parity`); re-capture on pool_gen change (`cuda_graph_recaptures_on_pool_gen_change`); 200-token generation, replay vs direct-launch bitwise parity (`cuda_graph_generation_replay_parity_real_model`); graphs-off A/B identical greedy text on 0.5B + 7B (CLI). Measured: 0.5B decode 200 → 236 tok/s (+18%), 7B decode 8.1 → 8.6 tok/s (+6% — launch overhead was the smaller share; kernel bandwidth is the 7e lever) |
| 7e | per-item A/B numbers appended to `docs/CUDA-BACKEND-PLAN.md` or `CUDA_OPTIMIZATION.md` |

---

## 8. Out of Scope / Future

Multi-GPU + peer copies (llama.cpp §6), FP16 activations + cuBLAS, FlashAttention
kernel, VMM pool, pinned-host buffers, CUDA graphs for prefill, Q5/IQ kernel
family, `graph_optimize`-style node reordering, CUDA CI runners, Windows.
