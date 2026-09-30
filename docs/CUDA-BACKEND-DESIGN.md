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
| Device layer | `src/cuda.rs` | `CudaState` context: device probe, weight registry, the per-instance stream binding (`bind_stream`/`stream`/`create_stream`), `extern "C"` kernel launchers, CUDA Graph API, pinned/staging memory, per-stream activation scratches, MMQ caches and gate reads |
| Device tier table | `src/device_tier.rs` | cc-keyed tier rows (measured GB10 + llama.cpp-adopted consumer rows + GENERIC) resolved once at init; feeds the MMQ gate, smem feasibility and plane-VRAM budget checks. Design + status: `DEVICE-ADAPTATION-PLAN.md`, docs 105–106 |
| Kernels | `src/cuda_kernels.cu` | The `__global__` kernels (quantized matmul families, attention, norms, elementwise, KV store, embedding gather, quantize planes) |
| Build chain | `build.rs` | Opt-in `--features cuda`, nvcc/`-ccbin` probe, per-arch SASS/PTX (incl. native sm_121), cudart link + rpath |

The split is deliberate: `cuda.rs` is the only place that touches the CUDA runtime API, and
`cuda_backend.rs` is the only place that knows about graph nodes.

### 2.2 The `CudaBackend` surface

```rust
pub struct CudaBackend {
    state: &'static crate::cuda::CudaState,   // process-wide device context
    stream: *mut c_void,                      // THIS instance's non-blocking stream (#188)
    kv_layout: i32,                           // KV layout tag for this instance
    pool: Vec<CudaBuf>,                       // id -> { ptr, bytes }
    free: Vec<usize>,                         // byte-length-matched free list
    pool_gen: u64,                            // bumped on every pool allocation
    pos_scratch: *mut c_void,                 // device i32 positions plane
    pos_scratch_bytes: usize,
    pos_memo: Option<(usize, u64)>,           // one-execution-window conversion memo
    graph_execs: Vec<CapturedGraph>,          // instantiated graphs
    graph_runs: HashMap<(u64, (usize, usize)), u32>, // warmup counters
    capturing: Option<(u64, (usize, usize))>, // open capture window (on `stream`)
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
- **The q4_K `W_dsc` plane is admitted for q4_K and only q4_K** (`src/q4k_dsc.rs`,
  `q4k_dsc_plane_admitted` — issue [#165](https://github.com/yusiwen/minfer/issues/165)). The rule
  has two halves: the *type* gate (`TensorType::Q4_K` is the one type `mmq_raw_nb_bt` dispatches the
  dsc template for, so the loader admits no other type) and the *payload* gate (`raw.len()` must be
  exactly `od * (id / 256) * 144`, q4_K's own block layout — **equality, not a lower bound**).
  `register_weight_q4k_dsc` re-checks the payload before the budget query and before
  `expand_q4k_dsc`, and `expand_q4k_dsc` itself returns `None` for a payload it cannot index, so a
  direct caller cannot bypass either. Why both: a q4_0 payload has exactly q4_K's bytes/element
  ratio (18/32 == 144/256), so the size check cannot refuse it; a q8_0 payload (34/32) is *longer*
  and would be misread as 144-byte q4_K super-blocks; and a future type with a *smaller* ratio (a
  2-bit K-quant: 84/256) is *shorter* than the row arithmetic needs, so the size check is what
  refuses it instead of reading past the tensor. The check cannot tell a q4_K payload from another
  type's bytes of the same length — that is the type gate's job.
- **The per-tensor registration dispatch is one shared rule** (`src/models/weight_reg.rs`,
  issue [#167](https://github.com/yusiwen/minfer/issues/167)). Both loaders call
  `register_cuda_weight` for every tensor the E5 plan puts on the device; it carries the whole
  contract: the quantized-type `matches!` set, the F16 raw branch, the F32 (1-D norms/biases vs
  2-D matmul weights) branch, the Q6_K padded repack, the q8_0 p32 plane, the q4_K `W_dsc` plane
  under `q4k_dsc_plane_admitted`, and the `clear_mmq_nb_bt_only` rule. The **decision**
  (`cuda_weight_reg`) is pure — no `CudaState`, no environment; the r59 dispatch gates are passed
  in — so CI's CPU job *runs* its tests, exactly like `src/q4k_dsc.rs`. Both loaders previously
  carried a copy of this block and the copies had drifted twice: the qwen3 copy had neither the
  f16 branch (#141) nor the q4_K dsc call (r59/#165), so an f16 Qwen3 fell to the CPU and a q4_K
  Qwen3 kept the in-kernel scalar dsc decode. The graph-side type gate is per architecture and
  must list the same types (`Qwen3Graph::weights_on_cuda` gained F16 in #167).
- A per-weight f16 dequant cache (`w16_cache`) is enabled by the loader only when quantized matmul
  weights exceed 2 GiB **and** MMQ is off; `MINFER_NO_W16CACHE=1` reverts.
- `ModelLoadGuard` (reentrant, process-wide) serializes loader registration so two models with
  same-named tensors cannot interleave, and real-model tests hold it across their forwards.

The graph-side CPU registry is separate: `register_graph_weights` registers into the allocator's CPU
backend, which is what a CPU split of a mixed graph executes from; the CUDA registry is filled by the
loaders at model-load time.

### 2.4 Streams, capture windows and the per-instance discipline

Everything the device path issues — kernels, `cudaMemcpyAsync` staging, events,
capture/replay, `synchronize` — runs on a **stream owned by the `CudaBackend`
instance**, not on a process-wide one. That is the #188 change:

- `CudaState` stays the process-wide **context**: the device, the name-keyed weight
  registry, the derived weight planes (`q6k_exp`/`q6k_dsc`/`q4k_dsc`/`w16_cache`) and
  the `device_memory`/`host_alloc` queries are genuinely context-scoped and shared.
- `CudaState::create_stream()` returns a fresh `cudaStreamNonBlocking` stream.
  `CudaBackend::with_layout` creates one per instance (no stream, no backend) and
  `Drop` destroys it after the pool, the scratches and the captured graphs.
- Every `CudaBackend` device operation starts with `let _bound = self.bind()`.
  `bind_stream` publishes the instance's stream in a thread-local;
  `CudaState::stream()` answers with it, so the ~60 launch/copy/event helpers in
  `cuda.rs` keep their signatures and still follow the instance. The stream
  consumers, named:

  | Consumer | How it gets its stream |
  |---|---|
  | kernel launchers (`cuda.rs`, the MMQ/attention/fused families) | `self.stream()` → the bound instance stream |
  | H2D input fill (`write_input_async`), D2D (`copy_device_to_device`), D2H staging (`copy_to_host_async`) | `self.stream()`; the pinned staging ring is keyed on the stream too |
  | events (`record_event`, `stream_wait_event`) and the F5 `copy_cross`/`await_cross` hooks | `self.stream()` via `enqueue_cross_host`/`take_cross`, both of which bind |
  | `cudaGraphLaunch` in `graph_replay_step`, `graph_begin_capture`, `graph_end_capture_to_exec` | `self.stream()` while the backend's own window is open |
  | `synchronize` / `state_sync` | `self.stream()` — one stream sync, counted per backend |
  | `copy_cells` (`kv_move_rows`) and `alloc_buffer`/`free_buffer` | the bound stream (the `cudaFree`/`cudaMalloc` themselves are context-wide) |
  | `Drop for CudaBackend` | binds, then frees pool/scratch/host/graph, then destroys its stream |
  | **weight registration** (`CudaState::register_weight`) | the **context** stream, never a backend's: an H2D copy queued on the context stream + `cudaStreamSynchronize(context)`, so it is stream-ordered and cannot be recorded into anybody's capture window |
  | **context-wide, stays shared** | `cudaMalloc`/`cudaFree`, `cudaMemGetInfo` (`device_memory`), `cudaHostAlloc`/`cudaFreeHost` (`host_alloc`/`host_free`), the weight registry and the derived planes, `cudaGetLastError` |

- **Activation scratch is per stream.** The `buf_hidden`/`buf_q8_prefill`/`buf_qa8_t`/
  `buf_q8_decode`/`buf_attn_partial`/… slots are now `StreamScratch`, a map keyed on
  the current stream, and the `MmqCache` memo is keyed the same way (a hit records a
  scratch pointer, so a shared memo would hand one engine's plane to another). The
  staging ring (`write_input_async`) is keyed on the stream for the same reason.
  Everything unbound — the legacy layer path, direct `CudaState` tests — keys on the
  context stream, which is why the #185 guard is **narrowed** to that path (§2.5).

**Capture mode.** Windows are opened with `cudaStreamCaptureModeThreadLocal`
(`graph_begin_capture`), not `cudaStreamCaptureModeGlobal`. Under Global another
thread's capture-unsafe driver call belongs to the window: it either invalidates the
capture (`cudaErrorStreamCaptureInvalidated`, 901) or faults inside the driver. The
recorded SIGSEGV ([#185](https://github.com/yusiwen/minfer/issues/185)) was exactly
that — one thread at `cuMemcpyHtoD_v2` under `register_weight` while another was at
`cuGraphInstantiateWithFlags` under `graph_end_capture_to_exec`. Thread-local mode
scopes invalidation to the capturing thread, so a registration on another thread is
benign. `MINFER_CUDA_CAPTURE_MODE=0|1|2` overrides the mode (relaxed / global /
thread-local) — it exists for the #188 probe's measurement, not for production.

**The #188 probe is the instrument.** `graph::cuda_backend::tests::
capture_window_on_one_thread_survives_a_weight_registration_on_another` opens a
capture window on one thread and issues a weight-registration copy on another while
the window is open, then closes it and checks both that `cudaStreamEndCapture`
returned `0` (never 901) and that the replayed graph produced the bytes it recorded.
It has two env knobs so the *mode* can be judged rather than assumed:
`MINFER_PROBE_STREAM=context` captures on the context (blocking) stream — the pre-#188
shared-stream model — and `MINFER_PROBE_LEGACY_MEMCPY=1` issues the registration with
the pre-#188 blocking `cudaMemcpy`. Measured on GB10 sm_121, 5 process runs per cell
(90 s watchdog; `crash` = the probe's own assertion failed), 2026-09-27:

| capture stream | registration | mode | result |
|---|---|---|---|
| context (blocking), shared | blocking `cudaMemcpy` (pre-#188) | global (1, pre-#188) | **5/5 hang** |
| context (blocking), shared | blocking `cudaMemcpy` | thread-local (2) | **5/5 hang** |
| context (blocking), shared | blocking `cudaMemcpy` | relaxed (0) | **5/5 hang** |
| instance (non-blocking) | stream-ordered (this PR) | global (1) | 5/5 pass |
| instance (non-blocking) | stream-ordered | **thread-local (2, adopted)** | **5/5 pass** |
| instance (non-blocking) | stream-ordered | relaxed (0) | **5/5 `end_code=901`** |

Three measured readings, none assumed:

1. **The blocking copy is a hard deadlock, independent of the mode.** A blocking
   `cudaMemcpy` is issued on the legacy null stream, which implicitly synchronizes with
   every **blocking** stream — including the one holding the open capture window, which
   by construction cannot complete until the host closes it. All three modes hang 5/5.
   So the mode is *not* the fix for the historical setup; a stream-ordered copy on a
   non-blocking instance stream is.
2. **Relaxed is ruled out by direct measurement.** With the structural fix in place,
   `relaxed` still returns `cudaErrorStreamCaptureInvalidated` (901) — the exact code
   the acceptance forbids — in 5/5 runs (`cudaMalloc` inside the window also fails,
   `CUDA: failed to allocate 16384 bytes`). It is not adopted.
3. **Thread-local is the mode.** With it, the probe passes 5/5 in both the pre-#188
   shared-stream cell (the deadlock aside) and the instance cell; `global` also passes
   the instance cell but is the mode that lets a *foreign* thread's driver call
   belong to the capture, which is the class this ticket exists to remove.

Two readings: the **mode** is what makes the historical shared-stream setup safe
(global is not viable; thread-local and relaxed both are, and thread-local keeps the
capturing thread's own mistakes fatal, so it is the one adopted), and the
**structural** change makes the mode irrelevant by removing the sharing. The
concurrent device gate
(`models::qwen2::graph::tests::two_cuda_engines_forward_concurrently_and_stay_bitwise_identical`)
is the positive half: two engines on two threads, two distinct streams, bitwise equal
to their serial references.

**The prefill-GEMM smem opt-in (#145, #147; lazy and gated since #218).**
A `gemm_f16_nt_kernel_t` instantiation whose dynamic shared memory exceeds the 48 KiB default must be
opted in with `cudaFuncSetAttribute(.., cudaFuncAttributeMaxDynamicSharedMemorySize, N)` — a launch
over an un-opted-in dynamic smem is rejected with `cudaErrorInvalidValue` and cannot succeed. The
number requested is the kernel's own byte layout — `As 2*TN*KS halves + Am 2*TN*KS floats (AF32 only)
+ Bs 2*TM*KS halves + Cs NW*256 floats`, TN = 64, NW = `blockDim.x/32` = 8 — and
`gemm_dynamic_smem_bytes(tm, ks, af32)` is the single source the launcher (`launch_gemm_f16`) reads;
before #145 an eager sweep carried a stale copy of it while the launcher carried a copy that dropped
the AF32 mirror. A request that exceeds the device's own
`cudaDevAttrMaxSharedMemoryPerBlockOptin` is **skipped with the reason printed**, instead of called:
the call could only return `cudaErrorInvalidValue` (which `compute-sanitizer` counts) and the
instantiation cannot launch on that device at all. On GB10/sm_121 (limit 101376 B) that is exactly one
combination, `gemm_f16_nt_kernel_t<256,64,true>` at 122880 B.

**The design is eager pre-warm at context creation + lazy per-launch opt-in (#223 restored the eager
half).** [#188](https://github.com/yusiwen/minfer/issues/188)
deleted `gemm_prefill_smem_init`'s `CudaState::try_new` call site with no mention in its commit message
or its docs commit; two later "dead code hygiene" commits annotated the orphan
`#[cfg_attr(not(test), allow(dead_code))]` instead of asking why a production-looking init had no
production caller; [#218](https://github.com/yusiwen/minfer/issues/218) removed the function and its
`checked`/`skipped` introspection, leaving the invariant tested but not **enforced** by production.
[#223](https://github.com/yusiwen/minfer/issues/223) put the runtime guarantee back **at the same
site**: `CudaState::try_new` calls the production entry `gemm_prefill_smem_prewarm_one(tm, ks, af32)`
once per process for every launchable combination (the `MINFER_GEMM_OPTIN_SET` X-macro in
`cuda_kernels.cu`, shared with the fatbin lookup and the test seam). The placement *is* the argument:
`try_new` runs under `CUDA.get_or_init`, before the state is published, before any `CudaBackend`
exists, and therefore before the per-instance stream `graph_begin_capture` needs — so "the attribute
is set outside any capture window" holds **by construction**, not by inference from the warmup count
or the capture mode. It is once per process, not once per backend. The entry drives the **same**
`gemm_smem_optin<TM,KS,AF32>` the launcher reads, so the pre-warm and the lazy path share one
per-instantiation cache and one `cudaFuncSetAttribute` site; a cache-keying regression therefore
cannot hide behind the pre-warm (it would leave the pre-warmed instantiations un-opted-in, which the
gates read back from the device). A successful pre-warm prints nothing; a failure or a deliberate
over-limit skip is named per instantiation, and `MINFER_NO_GEMM_PREWARM=1` is the documented control
that skips the loop (same-binary A/B and the "lazy path alone" gate arms).

The **lazy per-launch opt-in stays** as defence in depth: `gemm_smem_optin` (called from the
launcher's `GEMM_ONE`) invokes the shared `minfer_smem_optin` helper on an instantiation's **first**
launch and caches the answer in a function-local `static` per instantiation (a test
injection is never cached). The helper names the site, the instantiation, the attribute, the requested
bytes, the queried device limit and `cudaGetErrorName`, clears the latch, and the launcher **does not
launch** when the answer is false; its own `<<<>>>` error is read too (`minfer_launch_ok`) and returned
as 0, which `prefill_gemm_f16_inner` turns into an `Err`. The af32 wrapper `launch_gemm_f32a` returns
the same result. A request above `cudaDevAttrMaxSharedMemoryPerBlockOptin` is **skipped without calling
the attribute** on both paths, with the reason named. The same treatment covers every MMQ launcher
(`launch_mmq_nt`/`launch_mmq_raw_nt`/`launch_mmq_raw_nb_nt`/`launch_mmq_raw_nb_bt_nt`/
`launch_mmq_raw_nb_bt_q6k_nt`/`launch_mmq_raw_wide_nt`); the terminal two return 0 → `Err` at the
Rust caller, the fallback ones keep their documented `0 = clean fallback` contract. The operator's
signal stays the **first-launch site report**: a refused or skipped instantiation prints once, at the
launch site (`minfer_smem_optin`). The removed `checked`/`skipped` counters did not come back, and a
fully admitted pre-warm is silent.

**Why the invariant holds — the pre-warm by construction, then three defence-in-depth mechanisms.**
The historical claim was
that `cudaFuncSetAttribute` is illegal inside a capture window and poisons the context (error 700).
Nothing in the repository establishes whether that holds for the *adopted* mode (see the honest limit
below), so the design does not rely on the call being legal in a window; it relies on the call **never
happening in one**. The eager pre-warm makes that true by construction (above). The three emergent
mechanisms remain, now as the lazy path's fallback — called out at their sites (`graph_replay_step` in
`src/graph/cuda_backend.rs`, `gemm_smem_optin` in `src/cuda_kernels.cu`); any change to one is a
design change, not a tuning knob:

1. **The 3-run capture warmup** (`capture_warmup`, default 3): capture opens only from the third run
   of a `(uid, range)` key, and prefill-shaped graphs (`nt > 1`) capture by default since R3-B. An
   instantiation's first launch — the one that calls `cudaFuncSetAttribute` — therefore always runs
   uncaptured.
2. **`cudaStreamCaptureModeThreadLocal`** (#188's measured choice): the window belongs to the
   capturing thread, so a foreign thread's driver call cannot join it or invalidate it. Changing the
   mode must not silently change (1)'s guarantee.
3. **The per-instantiation cache**: the in-window launch re-reads the cached answer instead of asking
   the driver again, so the >48 KiB launch inside the window never calls the attribute. The cache is
   instantiated on the **`(tm, ks, af32)` template parameters**, not on a deduced `K`: every
   `gemm_f16_nt_kernel_t` shares one signature, and the pre-#218 `template <typename K>` gave the
   whole family one `static` (the #218 coverage gate found `<64,64,true>` answering for
   `<128,64,false>`, whose attribute had never been set). #223's pre-warm drives this same function,
   so the cache is exercised for every instantiation at context creation — the regression can no
   longer hide behind a sweep that bypassed the cache.

**The #218 gates pin that as observed behaviour.** `cuda_prefill_smem_optin_is_done_by_production`
(a real prefill forward in a fresh process; asserts the device's own `opted_in` read-back),
`cuda_prefill_smem_optin_refusal_fails_the_prefill` (the control arm:
`MINFER_TEST_CALL_FAIL=attr:gemm_f16_f16` makes the production prefill refuse the launch and name the
site), `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window` (a >48 KiB prefill captures,
replays bitwise, and `gemm_smem_optin_in_capture_count() == 0`), and
`cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation` (every launchable >48 KiB
instantiation reads back opted in through the production function). The `capture_warmup` test seam
(`MINFER_TEST_CAPTURE_WARMUP=1`) is the mutation lever for the counter.

**#223 adds the runtime guarantee's own gate.**
`issue223_tests::cuda_prefill_smem_prewarm_opts_in_every_launchable_instantiation_before_any_launch`
runs in a **fresh process** and asserts, immediately after `CudaState::init()` and before any kernel
launch, that every launchable >48 KiB instantiation already reads back opted in; a second fresh
process with `MINFER_NO_GEMM_PREWARM=1` asserts the negation, so the read-back is not always `1`. It is
the detector for the mutation the #218 gates cannot see — a pre-warm that skips one `(tm, ks, af32)`
(the lazy path simply opts it in on first launch, so the coverage/counter arms stay green). The four
#218 arms run their fresh-process children with `MINFER_NO_GEMM_PREWARM=1`, the documented control, so
their claims stay the lazy path's and the pre-warm cannot make them vacuous.

**Cost, measured on the real binary (2026-09-29, GB10 sm_121, CUDA 13.0, driver 580.178.04).** The
proxy in #223 used a synthetic kernel pair; the real prefill-GEMM fatbin is different, and the numbers
differ by ~15×: a pre-warm that is admitted in full is silent, and with `MINFER_OP_TIMING=1` the loop
reports **~2.2 ms** (median 2249 µs over 25 fresh processes, range 2126–2448) — the fatbin's one-time
module load, not the 152.9 µs the synthetic first `cudaFuncSetAttribute` cost. It is set-size
independent: looping over just `(128,64,false)` costs the same ~2.2 ms as looping over all twelve, so
it is one module finalization, not twelve. The **net** effect is still the proxy's conclusion: the
~2.2 ms **moves** rather than appears. `prewarm_prefill()`
(the r59 rider's `minfer_prewarm_kernels`, called at the end of Qwen2/Qwen3 weight registration,
before the first forward) already pushes that fatbin load into the startup path; with the pre-warm on
it costs ~2.3 ms, with the pre-warm off ~4.5 ms — the same ~2.2 ms, moved earlier. Controlled probe on
the real binary (fresh process, `MINFER_MMQ=0/1` × `MINFER_GEMM_K64=0/1`): the first prefill forward
is 2288–2314 µs with the pre-warm off and 78–120 µs with it on; `prewarm_prefill()` is 4.3–4.6 ms off
vs 2.2–2.4 ms on. The hot path is untouched: `minfer bench -p 2048 -n 128`, same binary, 7 interleaved
matched rounds, medians — `pp2048` 2546.55 vs 2544.34 t/s (**+0.09%**), `tg128` 236.30 vs 236.45 t/s
(**−0.06%**), bar ±1%. The full transcript is in the #223 record.

**The same-thread `ThreadLocal` in-window case — measured (2026-09-29).** The open question this
section used to carry — is `cudaFuncSetAttribute` legal inside a **same-thread**
`cudaStreamCaptureModeThreadLocal` window? — is now measured: the
`MINFER_TEST_CAPTURE_WARMUP=1` mutation arm drives the opt-in into the first, captured run, and on
GB10 sm_121 / CUDA 13.0 / driver 580.178.04 the call is **tolerated** — the attribute publishes, the
>48 KiB graph still captures, instantiates and replays bitwise-identically, and only
`gemm_smem_optin_in_capture_count()` moves. (The 2026-09-25 probe below measured the *Global* mode
instead; this one is the adopted mode.) The design nevertheless keeps the call out of the window:
that behaviour is not contractual across toolkits, and the counter gate is what makes "never set
inside a window" an observed property rather than a driver assumption. The full transcript is in the
#218 record.

**Measured correction (2026-09-25, CUDA 13.0 / driver 580.178.04 / sm_121).** A probe
(`/tmp/fix147_attr_capture_probe.cu`) shows the historical claim above no longer holds verbatim on
this runtime: `cudaFuncSetAttribute` returns `cudaSuccess` when called *inside* an open
`cudaStreamCaptureModeGlobal` window, both for the already-set value and for a new one. The eager
sweep that was in tree then has since been removed by #218 (it was already dead code — #188 had
dropped its caller), and because the launcher caches one answer per instantiation it does not re-ask
inside a window either.
> **#223 forward note (2026-09-29):** the eager half is back, as a **pre-warm through the lazy
> entry**, not as the old sweep: `CudaState::try_new` calls `gemm_prefill_smem_prewarm_one` for every
> launchable instantiation, i.e. the same `gemm_smem_optin` cache the launcher reads. Nothing in the
> measured driver behaviour above changes; the point of the placement is to stop depending on it (the
> attribute is set before any window can exist, by construction). On the real binary the loop's
> measured cost is ~2.2 ms (the fatbin's one-time module load), which `prewarm_prefill()` already paid
> during registration — net new startup cost ≈ 0, hot path unchanged. See §2.4 and the #223 record in
> `ARCHITECTURE-EXECUTION-PLAN.md`.

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

`with_layout()` resolves the device singleton, **creates the instance's own non-blocking stream**
(issue #188; a device that cannot give one gives no backend), snapshots the engine's KV element type
(`crate::cuda::layout_of`), reads the two graph gates (`MINFER_NO_CUDA_GRAPH=1` →
`GraphMode::Disabled`; `prefill_capture` defaults ON unless `MINFER_NO_PREFILL_CAPTURE=1`), and
starts with an empty pool. `Drop` binds the stream, frees every pool pointer, the positions scratch,
the cross-backend staging and every captured graph exec, then destroys the stream.

State groups:

| Group | Fields | Lifecycle |
|---|---|---|
| Device + KV policy | `state`, `kv_f16` | fixed at construction |
| Pool | `pool`, `free`, `pool_gen` | grows on demand; `free_buffer` only recycles; `alloc_fresh` bypasses the list for split staging; `pool_gen` bumps on every allocation |
| Positions | `pos_scratch`, `pos_scratch_bytes`, `pos_memo` | grown on demand; the scratch pointer is embedded in captured execs, so growth bumps `pool_gen` to force re-capture |
| Capture | `graph_execs`, `graph_runs`, `capturing`, `graphs_mode`, `prefill_capture` (all on the instance's own stream) | see §4.8 |
| Trace | `cap` | pinned async D2H staging, see §4.7 |

Pool rules worth restating because they carry correctness weight:

- **Exact byte-length reuse only** — `alloc_buffer` scans `free` for `pool[id].bytes == size * 4`.
- **`free_buffer` never frees** — a persistent KV region must survive rebuilds, and the pool keeps
  device memory for the next graph. Only `Drop` returns memory to the driver.
- **`alloc_fresh`** exists for split-boundary staging: ids in the free list are still referenced by
  `node_to_buf` and physically live during the execute that follows.
- **OOM is not a panic.** `cuda_malloc` logs and returns null; the null buffer fails cleanly at
  execute time (`ptr_of`). Panicking is forbidden because it would poison the shared scratch maps
  and the device-entry token (the legacy path) for every other user.

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
| `GetRows` + `Embed` meta | `embed_rows_on_gpu` (per weight type, incl. the padded Q6_K layout and, since #141, a dedicated f16 gather) | weight registered; type via the model gate |
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
- **f16 weights (#141)**: `f16_f32_matmul_vec` when `id % 8 == 0`, else `f16_f32_matmul_scalar` — for
  every `nt`, because f16 is not an MMQ *format* (MMQ streams quantized bytes) and the f16-wmma path
  is the `MINFER_MMQ=0` fallback for the quantized types, not an f16-weight kernel. The weight bytes
  stay 2 B/element **on the device**: there is no registration-time dequant to f32, so the memory the
  f16 file exists to save is actually saved (a 0.5B f16 GGUF registers 942.4 MiB of device weights;
  ~1.9 GiB if it were dequantized). The kernel converts in-register with `__half22float2` and FMA's
  against the f32 activations, so the accumulation is f32 like every other CUDA matmul. Both launchers
  read their own launch return through the #147 helpers and return non-zero → `Err`, rather than
  joining the unchecked `<<<>>>` sites of [#162](https://github.com/yusiwen/minfer/issues/162). The
  f16 **embedding** gather is `embed_rows_f16` (one thread per output element) — without it the
  all-or-nothing `weights_on_cuda` check would drop a converted f16 GGUF to the CPU over its
  `token_embd` alone.
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
  (`kvformat::auto_device_format` selects it when `n_layers × n_kv_embd >= 8192`, the 7B class, and
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

`CudaBackend` holds that `int` in its `kv_layout` field, and `kv_row_bytes(nkt)` derives the stride
(`nkt*4`, `nkt*2`, or `KvFormat::Q8_0.row_bytes(nkt)`). The packed store is `store_kv_q8_0`, whose
quantizer is the CPU's step for step (`amax/127`, f16 scale, round-ties-even), so both backends store
the same bytes. Before C4 S2b this was a `bool` that mapped anything not exactly `f16` to f32 — which
would have addressed a packed region as f32 rows, the silent corruption the layout tag exists to make
impossible.

**Per-engine scope ([#99](https://github.com/yusiwen/minfer/issues/99), completed by
[#153](https://github.com/yusiwen/minfer/issues/153)).** #99 made the KV *format* per engine for the
model, the graph builder (`CParams::kv_format`), the allocator and the CPU kernels; #153 finished the
device half. There is **no** `cuda::KV_LAYOUT` static any more: the engine's resolved `KvFormat` —
including the GPU auto policy, which `kvformat::resolve` now folds in from the model dims — reaches
`CudaBackend` through `GraphAllocator::set_kv_format` (and `enable_cuda` builds a fresh backend from
the same stamp), so `CudaBackend::kv_layout` is the only source the dispatch reads. `cuda::layout_of`
/ `format_of` are the one binding between `KvFormat` and the FFI tag. The launchers already took the
tag as an argument; what was process-wide was the value. Two engines in one process therefore run
their own layouts, and the captured-graph key carries the tag (below), so an exec instantiated for
one layout cannot replay for another. `models::load_model_configured` no longer restates anything.

**Metal is the remaining device-static.** `metal::kv_cache_is_f16` is still a process-wide
`OnceLock` its kernels read (no Mac here to re-plumb it; Metal is G5 for packed anyway), so the
*Metal* device run keeps the documented discipline.

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
3. A stored exec for `(uid, range)` with a **matching `pool_gen` and `kv_layout`** → `cudaGraphLaunch`;
   on launch failure, disable graphs for the session and fall back.
4. A stored exec with a **different `pool_gen`** (pointer layout may differ) **or a different
   `kv_layout`** (the recorded kernels were instantiated for the old tag — `store_kv_f16` vs
   `store_kv_q8_0`, the layout-tagged attention) → destroy it, drop the warmup counter, re-warm. #153
   added the `kv_layout` term; `CudaBackend::set_kv_layout` also invalidates eagerly when a stamp
   moves, so the lookup check is the second line of defence.
5. Warmup: executions 1 and 2 of a key run direct launches (llama.cpp warms up twice); a one-shot
   prefill never reaches capture.
6. On the third execution, if `nt_hint.map_or(true, |nt| nt == 1 || prefill_capture)`, the backend
   opens a capture window on **its own stream**, in `cudaStreamCaptureModeThreadLocal` (§2.4) — no
   process-wide lock, because no other backend shares this stream. The gate means decode-shaped
   graphs always capture; prefill-shaped graphs capture only when `prefill_capture` is on (default
   **ON** since R3-B; `MINFER_NO_PREFILL_CAPTURE=1` opts out).
7. The window closes at the split's `synchronize` → `close_capture_or_sync`: end capture, instantiate,
   **launch once** so the step still produces output, cache the exec at the current `pool_gen` and
   `kv_layout`, then sync. A failure destroys the exec, logs loudly, and disables graphs for the
   session.

Replay correctness rests on stable addresses: pool ids never move memory, `copy_across` rewrites the
same staging buffers each step, and the positions scratch pointer is embedded in captured execs — so
growing it bumps `pool_gen` and forces re-capture.

Two interactions are part of the contract:

- **Trace/viz disables replay.** The scheduler skips `graph_replay` entirely while `MINFER_TRACE` or
  live viz capture is active, because per-node host readbacks inside a capture window are illegal.
- **A node error inside the window aborts it** (`abort_capture`): end capture without launching,
  destroy the exec, disable graphs, sync. Later steps run direct-launch with graphs disabled; the
  aborted step's outputs were never produced and are consumed as-is — there is no poisoned-error
  mechanism, and the code says so explicitly.

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
   `cudaGraphDestroy` called on a `cudaGraphExec_t` — both fixed at their call sites by #145, and the
   remaining unchecked sites of the same class (every MMQ dynamic-smem opt-in and launch, the
   prefill-GEMM launcher's own launch, and `graph_end_capture_to_exec`'s `cudaGraphDestroy`) by
   #147. #162 then removed the class entirely: **every** `<<<>>>` in `cuda_kernels.cu` reads its own
   error, so a latched error at `sync()` is by construction an error no site read (a non-launch API
   call), never an unattributed launch.
5. **A return value that gates a later launch is read where the call is made.** Every dynamic-smem
   opt-in goes through `minfer_smem_optin`: an over-limit request is skipped with the reason, any
   other failure is named and cleared at the call site and the launch is **refused** (a launch over
   an un-opted-in dynamic smem cannot succeed). Every launch goes through `minfer_launch_ok`, whose
   immediately-following `cudaGetLastError` is a launch check because `minfer_launch_prelude`
   cleared (and reported) any latch that predates the launch. The sync poll is the backstop for a
   *missed* site, not the place to diagnose one; #147's `issue147_tests` gates inject a real failure
   at every one of the sites and assert the named report, the refusal and a clean latch.
   **#162 states the per-op severity in the helper** (§7.9): `minfer_launch_ok` is required — it
   records a sticky failure that `CudaBackend::execute_node` turns into an `Err` naming the site (one
   Rust-side check, not one per launcher, so the op never proceeds on a stale output) — while
   `minfer_launch_ok_opt` only names and clears for a path with a **documented fallback** (the MMQ
   fast paths, the fa-prefill smem fallback, and the int-returning launchers whose Rust caller already
   decides). Both levers are data: `minfer_launch_block` (an illegal block geometry) and
   `minfer_launch_smem` (an over-limit dynamic smem request) make the *real* launch fail.
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
  lazy >48 KiB opt-in (#218) has its own gates: `the_gemm_smem_formula_matches_the_kernel_layout`
  (pins `gemm_dynamic_smem_bytes` against the kernel's byte layout — the value-level arm, so a silent
  shrink is seen); `cuda_prefill_smem_optin_is_done_by_production` (a **real** prefill forward makes
  the device read back `opted_in == 1` for `gemm_f16_nt_kernel_t<128,64,false>`, in a fresh process so
  the "not opted in before" precondition is observable); the control arm
  `cuda_prefill_smem_optin_refusal_fails_the_prefill` (`MINFER_TEST_CALL_FAIL=attr:gemm_f16_f16`
  refuses the launch, and the site report names the call and `cudaErrorInvalidValue`);
  `cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation` (every launchable >48 KiB
  instantiation reads back opted in through production's own `gemm_smem_optin`, and every over-limit
  one is refused without a call); and `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window`
  (a >48 KiB prefill captures, replays bitwise, and `gemm_smem_optin_in_capture_count() == 0` proves
  the opt-in ran before the window opened). Plus `the_latched_error_message_never_blames_a_kernel`.
  #147's `issue147_tests` module adds
  `the_graph_destroy_failure_message_names_the_matching_destructor` and
  `the_injection_matcher_matches_only_the_named_site` (both pure) plus the three
  `cuda_issue147_*` deliberate-failure gates (device; env-gated behind `MINFER_TEST_ISSUE147=1`, which
  arms `MINFER_TEST_CALL_FAIL` per site): every dynamic-smem site names its failed opt-in and refuses
  the launch, every launch site names its failed `<<<>>>` and refuses, and the destroy site names a
  failed `cudaGraphDestroy` and leaves no latch.

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

### 7.2a Profiling on dgxspark (`ncu` / `nsys`)

The reusable recipe, so the next session does not re-derive it (recorded 2026-09-27, GB10 sm_121,
CUDA 13.0, driver 580.178.04):

- **`ncu` is installed but not on `PATH`.** The binary is
  `/usr/local/cuda-13.0/bin/ncu` (2025.3.1). `which ncu` finding nothing means the directory is not
  on `PATH`, not that the tool is missing — use the absolute path.
- **A normal user cannot collect counters here** — `RmProfilingAdminOnly: 1` makes `ncu` fail with
  `ERR_NVGPUCTRPERM`. **`sudo` collects** (passwordless on dgxspark); no module parameter change and
  no driver reload is needed. Record "collected as root; module parameter unchanged".
- **`sudo` changes `HOME` to `/root`,** so the model cache under `/home/yusiwen/.cache/minfer/models`
  is invisible to a `sudo`-launched binary. Pass absolute model paths
  (`MINFER_BATCH_TEST_MODEL=/home/yusiwen/.cache/...`, or the path on the command line) or use
  `sudo -E`. A "model not found" under `sudo` is this, not a missing file.
- **Build as the normal user first, then attach `ncu` to an already-built binary.** `ncu` does not
  write repository files, so `target/` stays `yusiwen`-owned; if a `sudo` run does leave a root-owned
  file, `chown` it back before the next `cargo` build.
- **Some metrics are `n/a` on GB10/sm_121** — `dram__bytes.sum` among them (the integrated-memory /
  DGX Spark form exposes no classic DRAM counters). Confirm a metric name exists with
  `ncu --query-metrics` first, prefer the SM / instruction / L1 / L2 families, and treat a metric you
  actually collected as the only evidence; `n/a` is not a number.
- **Collect targeted, not whole-run.** Counter replay is slow: filter with `-k regex:<kernel>` and
  bound it with `--launch-count`, rather than replaying a full `bench`.
- **A `-k` regex must match ncu's base kernel name, not the demangled signature.** ncu lists the
  available kernels by base name (`gqa_attn_split_partial`, no template arguments), so a regex that
  includes `<` — e.g. `-k 'regex:gqa_attn_split_partial<'` — matches nothing and ncu prints an
  "Available Kernels" list instead of collecting (`No kernels were profiled`). Anchor it
  (`-k 'regex:gqa_attn_split_partial$'`) so sibling kernels (`..._combine`, `..._bt`) are excluded
  rather than eating into `--launch-count`. Recorded by #202, which lost one collection to this.

`nsys` stays the cheap, always-available instrument for per-kernel durations
(`nsys profile --trace=cuda --cuda-graph-trace=node ...` — without `node`, kernels launched from a
replayed CUDA graph are traced as one graph and never appear individually, which silently hides the
whole decode path).

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

### 7.4 Issue #147 verification (GB10, sm_121, CUDA 13.0, driver 580.178.04)

The follow-on to #145: the remaining CUDA calls that discarded a return value gating a later launch or
allocation (the MMQ dynamic-smem opt-ins and the launches that follow them, the prefill-GEMM
launcher's own launch, and `graph_end_capture_to_exec`'s `cudaGraphDestroy`). Latent on this device —
the sanitizer was already clean — so the acceptance is that each site still cannot latch: **0 API
errors** before and after, **508 passed / 0 failed / 32 ignored** after (503 before; five new gates:
three pure, two device/env-gated), 0.5B and Qwen3-0.6B real-model sets **32 / 0** each, CPU suite
440 / 0 / 29 + 10 / 0 / 6 and CPU serial ignored 29 / 0 unchanged.

The deliberate-failure injection is `MINFER_TEST_CALL_FAIL` (site tokens, or `all`) plus
`MINFER_TEST_ISSUE147=1` to enable the gates: a named site performs its **real** call with a value that
fails — an attribute request one page over the queried device limit, a launch with 16 MiB more dynamic
smem than was opted in, or `cudaGraphDestroy` on the exec — so the "latch cleared" half is exercised for
real rather than through a synthetic return value. Both variables are unset in every default, bench and
`compute-sanitizer` run. The two probes the injection relies on are recorded in `/tmp`:
`fix147_attr_capture_probe.cu` (`cudaFuncSetAttribute` inside a Global capture window returns
`cudaSuccess` on this runtime — see §2.4) and `fix147_launch_fail_probe2.cu` (an over-limit dynamic-smem
launch is rejected by the launch call with `cudaErrorInvalidValue`, and the kernel never runs).
Mutation checks: every hardening reverted one at a time (eight per-site attribute guards, the gemm
opt-in guard, the shared launch check, the gemm message naming a wrong instantiation, the shared opt-in
admitting a failure, the Rust destroy read, the Rust formatter, the injection matcher), each failing
its gate, with both files restored byte-identically. The remaining 65 unchecked `<<<>>>` returns were
[#162](https://github.com/yusiwen/minfer/issues/162), now closed by §7.9; the full #147 record is
`docs/ARCHITECTURE-EXECUTION-PLAN.md` (C4 S2d).

---

### 7.5 Issue #141 verification (GB10, sm_121, CUDA 13.0, driver 580.178.04)

f16 weights on the device, the third item F6 (#49) left: the loader registered f32 and the supported
quants only, so an f16 GGUF fell to the CPU through the all-or-nothing `weights_on_cuda` gate even on
a CUDA build. The change is the registration branch (`TensorType::F16` → `register_weight` raw; it is
deliberately **not** folded into the quantized `matches!`, whose q4_K dsc-plane gate has no type
check), `matmul_f32_ptr_layout`'s f16 arm, `embed_rows_on_gpu`'s f16 arm, and `weights_on_cuda`'s
matmul/embed type sets.

**Verified** (`f141_f16_weights_run_on_the_cuda_device`, `#[ignore]`d in the real-model set):
`minfer convert` produced the file from the Qwen2.5-0.5B-Instruct HF checkpoint (994,156,352 B,
948 MiB, 290 tensors, every 2-D tensor f16). The gate asserts, in order: the model's device is
`Cuda`; the offload report says all 24 blocks + embed/output are on the device (942.4 MiB of device
weights); the scheduler assigns **169 f16 matmul nodes + 1 f16 embed node** to `Backend::CUDA`
(counted by walking the built graph's `CNode.backend`, so a silent CPU fallback fails here); then the
device logits against the same file's `Layers(0)` CPU run — max |Δlogit| **7.34e-5** (mean 1.26e-5)
against max |logit| 18.43, **4.0e-6** relative, greedy `[12095, 13, 1084, 374]` identical, asserted at
|Δ| ≤ 0.01 and relative ≤ 1e-3 (the bound is stated and printed by the gate). llama.cpp on the same
file (`--temp 0`) prints `Paris.`, the same continuation minfer gives on CPU and CUDA.

**Mutations (reverted; files restored byte-identical).** (a) the loader's f16 registration gated off →
the gate fails at `device()`: *left: Cpu, right: Cuda*; (b) the f16 arm removed from
`matmul_f32_ptr_layout` → the gate panics on the loud `Err` *"cuda: weight type F16 has no
f32-activation matmul kernel"*, not on a silent fallback; (c) the kernel's `__half22float2` replaced
with zeros for half of each 8-element chunk → the gate fails on the greedy continuation (a value-level
fault cannot pass the numeric comparison either). `compute-sanitizer --tool memcheck` over the CUDA
unit suite stays at **0 API errors**. The full record is `docs/ARCHITECTURE-EXECUTION-PLAN.md` (#141).

---

### 7.6 Issue #165 verification (GB10, sm_121, CUDA 13.0, driver 580.178.04)

The q4_K `W_dsc` plane's admission contract (§2.3 above). The defect was the qwen2 loader reaching
`register_weight_q4k_dsc` for every non-Q6_K type in its quantized `matches!`; the surface was a
plane built from another type's bytes that no kernel reads (the map is keyed on the q4_K weight's
device pointer), plus a latent out-of-bounds read for a future smaller-ratio type.

**Fixed** in `src/q4k_dsc.rs` (`q4k_dsc_plane_admitted` = type + exact payload; the qwen2 loader
calls it; `register_weight_q4k_dsc` re-checks the payload before the budget query and before
`expand_q4k_dsc`, which itself returns `None` for a payload it cannot index).

**Measured** (`q4dsc_planes()`, registry-by-name; before = the two gate halves reverted):

| Model | Before | After |
|---|---|---|
| cached 0.5B **q4_0** | 24 planes / 26 148 864 B | **0 / 0** |
| `/tmp/fix165/qwen2.5-0.5b-instruct-q8_0.gguf` (`minfer quantize` of the q4_0 0.5B) | 24 / 26 148 864 B | **0 / 0** |
| `Qwen3-0.6B-Q8_0.gguf` | 0 / 0 — the qwen3 loader never had the call | 0 / 0 |
| cached 0.5B **q4_k_m** (positive control) | — | **12 / 13 074 432 B** (exactly the index's admissible q4_K set) |

`cargo test --release --features cuda -- --test-threads=1`: **516 passed / 0 failed / 34 ignored**
(baseline 513 / 0 / 33). `FEATURES=cuda scripts/real_model_gates.sh`: **34 / 0** at both the 0.5B and
the Qwen3-0.6B-Q8_0 config. `compute-sanitizer --tool memcheck` over the serial unit suite **0 API
errors**, before (514/2/34, the two new gates failing on the pre-fix path) and after (516/0/34).

**Mutations (reverted; files restored byte-identical, `sha256sum`).** (a) type gate removed →
the pure wrong-type assertion and the real-model q4_0 gate fail (**24 planes / 26 148 864 B** against
0 expected); (b) exact payload equality weakened to `>=` → the pure *"one block long"* assertion and
the device gate's *"a q8_0 payload must not register a __q4dsc plane"* fail; (c) the plane registered
under a wrong name (`__q4dscX`) → the device gate fails at its **positive control**, proving the
"nothing registered" arms observe the plane's real registry entry. The full record is
`docs/ARCHITECTURE-EXECUTION-PLAN.md` (F6c).

**Honest scope.** The model #165 names (`Qwen3-0.6B-Q8_0`) does **not** reproduce the defect — it is
arch `qwen3`, whose loader has no `register_weight_q4k_dsc` call (0 planes before and after); the
qwen2 q8_0 arm is measured on a q8_0 file built here. And the size check cannot tell a q4_K payload
from another type's bytes of the same length (q4_0 shares q4_K's ratio exactly), which is why the
type gate is not redundant.

---

### 7.7 Issue #167 verification (GB10, sm_121, CUDA 13.0, driver 580.178.04)

The qwen3 loader's registration block had drifted from qwen2's twice: it had no r59
`register_weight_q4k_dsc` call (so a q4_K Qwen3 kept `mmq_raw_nb_bt`'s in-kernel scalar dsc decode)
and no `TensorType::F16` branch (#141, so an f16 Qwen3 model was dropped to the CPU); the qwen3
graph's `weights_on_cuda` was missing `F16` as well, so registering alone would not have been
enough. Fixed by §2.3's shared rule (`src/models/weight_reg.rs`; both loaders call
`register_cuda_weight`) plus `Qwen3Graph::weights_on_cuda`'s two F16 arms and
`CudaState::q4dsc_plane_for` (the same pointer-keyed lookup `mmq_raw_nb_bt` performs).

**Verified.** `f167_qwen3_q4k_registers_the_dsc_plane_exactly` loads a HuggingFace Q4_K_M
Qwen3-0.6B and asserts the registered `*__q4dsc*` set **equals** the GGUF index's admissible q4_K
set by name and count: **168 planes / 95 420 416 B** (168 q4_K among 29 other quantized 2-D
tensors), each found by the kernel's own `q4dsc_plane_for`, each non-null and distinct; a q8_0 and a
**q4_0** negative arm register **0** (q4_0 shares q4_K's bytes/element exactly, so it is the only
negative that can catch a type-gate bypass). `f167_f16_qwen3_weights_run_on_the_cuda_device` shows
28/28 blocks + embed/output on the device (1137.0 MiB), **197 f16 matmul + 1 f16 embed node**
assigned `Backend::CUDA`, and device-vs-CPU max |Δlogit| **8.92e-3** (mean 1.25e-3) / **4.46e-4**
relative at max |logit| 19.99, greedy `[12095, 13, 576, 6722]` identical — asserted at ≤ 0.05 and
≤ 1e-3. Qwen3's spread is ~100× the 0.5B f16 gate's (7.34e-5 / 4.0e-6) because it runs four norms
per layer and the CPU rms_norm (8-lane AVX2 FMA + f64 tail, `1/sqrt`) and the device rms_norm
(warp-shuffle f32, `rsqrtf`) differ in reduction order and reciprocal-sqrt form.

Suites: `cargo test --release --features cuda -- --test-threads=1` **521 passed / 0 failed / 36
ignored** (baseline 516 / 0 / 34); `FEATURES=cuda scripts/real_model_gates.sh` **36 / 0** at both
the 0.5B and the Qwen3-0.6B-Q8_0 config (baseline 34 / 0); `compute-sanitizer --tool memcheck` over
the serial unit suite **0 errors** (521 / 0 / 36 in 353.44 s); CPU `cargo test --release` **452 / 0
/ 32** unit + **10 / 0 / 6** integration, `PARALLEL=0 scripts/real_model_gates.sh` **32 / 0**.

**Mutations (reverted; `sha256sum` byte-identical).** (a) the F16 arm removed from
`cuda_weight_reg` → the pure f16 test fails and the device gate fails at *left: Cpu, right: Cuda*;
(b) F16 removed from `Qwen3Graph::weights_on_cuda`'s `matmul_t_ok` → the same failure; (c) the q4_K
admission bypassed (`q4k_dsc = gates && od % 2 == 0`) → the pure test fails and the real-model gate
fails on its **q4_0** negative arm (the q8_0 arm alone cannot catch it: the registry re-checks the
payload and refuses a longer one); (d) the plane forced off → the pure test fails and the gate
fails at 0 planes against 168 expected.

**Honest scope.** The r59 dsc **prefill win is not measured** — the gate proves the plane set is
exactly right and the kernel finds it, not a Qwen3 prefill speedup. The q4_K model is a community
Q4_K_M (the official Qwen Qwen3-0.6B GGUF repo ships only Q8_0) and the f16 model is
`llama-quantize --allow-requantize … F16` of the cached Q8_0; `minfer quantize --type f16` was
tried first and writes f16 **1-D norms**, which the engine cannot load (filed as
[#169](https://github.com/yusiwen/minfer/issues/169)). The plane gate needs
`/tmp/f167-work/qwen3-q4k.gguf` and skips (printing why) when it is absent.

---

### 7.8 Issue #169 verification — the norm weight type is part of the rms_norm invariant (GB10, sm_121, CUDA 13.0, driver 580.178.04, 2026-09-26)

The issue recorded this as a **latent** hazard: `CudaBackend::norm_weight` resolved the weight by
name and checked only that it was *registered*, not what it was. The CUDA `rms_norm` kernel indexes
the weight as `d` f32 elements (`d*4` bytes) regardless, so an f16 norm weight (2 B/element) — which
`minfer quantize --type f16` used to write for every 1-D tensor — would be read past its end. The CPU
panic was reached first there, so the device behaviour was not observable at the time.

`norm_weight(node, elems)` now resolves `CudaState::weight_size(name)` (the registered raw length,
the same convention `has_weight_of_size` uses for a padded Q6_K plane) and requires exactly
`elems * 4`, returning `Err` that names the registered length, the length the kernel reads, the node
and "f16-norm". Both callers pass the dim the kernel uses (`node.out_shape[0]` for `Op::RmsNorm`,
`*hd` for `Op::QkNorm`), so the check is the kernel's own geometry, not a name heuristic.

**Verified.** `cuda_norm_weight_size_is_part_of_the_invariant`: one 64-element norm graph, two
registered names, both valid float4 dims — the f32 arm (control) executes, the f16-sized arm must
`Err` and the test asserts the refusal names `128 B`, `256 B` and `f16-norm`. The mutation (check
forced off) makes the f16 arm **execute** and, under `compute-sanitizer --tool memcheck`,
`Invalid __global__ read of size 16 bytes` ×12 / `ERROR SUMMARY: 12 errors` — the read is real.
The CUDA unit suite is **526 passed / 0 failed / 37 ignored** with the gate in place (baseline
524 / 0 / 37, +1 CI tooling gate and +1 device gate), and `compute-sanitizer` over that suite reports
**0 errors**. The end-to-end half is #169's acceptance run: `minfer quantize --type f16` of the
cached `Qwen3-0.6B-Q8_0.gguf` now writes 1-D f32 / 2-D f16, the file is byte-identical to
`llama-quantize … F16`, and it runs with all 28 blocks + embed/output on the device (1137.0 MiB);
CPU-vs-CUDA argmax agrees at prefill and the two decode steps (max |Δlogit| 0.0178).

**Honest scope.** The gate is device-only (CI has no GPU); its arithmetic has no CI-covered pure
twin — it is one `usize` comparison against `elems * 4`, and the device arm is the gate. It does not
make an f16-norm model *loadable* on CUDA; registration still admits an f16 1-D weight, and the
refusal happens at execute time with the node named, which is the documented `Err`-not-fallback rule
of `docs/GPU_SAFETY.md`. The one lookup per norm node per execution was not benchmarked and no
timing claim is made.

---

### 7.9 Issue #162 verification — every `<<<>>>` reads its own launch error (GB10, sm_121, CUDA 13.0, driver 580.178.04, 2026-09-26)

The wider #147: the **104** sites in `src/cuda_kernels.cu` that enqueued a kernel and never read the
launch's error (`grep -c '<<<'` is 122; two are `<<<>>>` in prose comments, so the audited surface is
**120 sites in 76 `launch_*` owners** (the wrappers plus the static `launch_gqa_attn_split_batched_kv` helper). Each is now preceded by `minfer_launch_prelude(site, kernel)` and
followed by a `minfer_launch_ok` / `minfer_launch_ok_opt` read that names the `launch:` site, the
kernel instantiation and `cudaGetErrorName`, and clears the latch it named. The report tag is
`(#162/<site>)` (§4 rules 4–5).

**The decision per launcher is severity in the helper.** `minfer_launch_ok` (required) records a
sticky failure; `CudaState::take_launch_failure` drains it in `CudaBackend::execute_node` on **both**
arms and returns `Err` naming the site and the node — one Rust-side check, not 104 signature changes
— so the op never proceeds on a stale output. `minfer_launch_ok_opt` names and clears for a path with
a documented fallback. Of the 120 sites, **107 are required → `Err`** and **13 are documented
fallbacks**: `launch_fa_prefill_f16kv`'s three window modes (the launcher returns `-1` and the Rust
caller falls back to the legacy attention kernel), the four `launch_mmq_raw_nb*` / `_wide_nt`
launchers #147 already treated as clean fast-path fallbacks (including their k-split reduce sites,
which got their own `launch:*_ksplit` tokens so they can be armed alone), and `launch_kv_move_rows`
(whose non-zero return its `Result` caller already uses).

**The injection lever is shared geometry.** `minfer_launch_block(site, dim3|unsigned)` wraps the
block argument at every ordinary site; arming the site makes the block 4096 threads (over the
1024/block limit), so `<<<>>>` returns `cudaErrorInvalidValue` for real and the kernel never runs
(probed on sm_121 — both an over-limit block and an over-limit dynamic smem return
`cudaErrorInvalidValue`). The dynamic-smem launchers keep `minfer_launch_smem`. Adding a site's
coverage is therefore one string in the driver table.

**Two gates, one source and one runtime.** `scripts/check_cuda_launch_returns.py` parses the source
(comments and string/char literals blanked, so a commented-out occurrence is not a site) and, for
every `<<<`, requires a `launch:`-prefixed prelude before it in the same function, a
`minfer_launch_ok`/`_opt` after its statement naming the **same** token, and a real-failure lever in
the geometry; it prints the offending lines and exits 1. It runs in the `check-docs` CI job with
`--selftest` (five pass/fail cases) and `--check-fixture tests/fixtures/cuda_launch_sites.tsv` (the
committed site list; line numbers are documentation, owner/site/fragment are identity). The runtime
half is `cuda::issue162_tests` (`MINFER_TEST_ISSUE162=1`, five tests): the coverage test drives every
audited site through the branches that reach it, asserts armed set == observed set, the message names
the site, the fixture's kernel fragment and `cudaErrorInvalidValue`, `cudaGetLastError() == 0` after
every scenario, and the union of driven tokens equals the fixture; the severity test asserts a
required site sets the sticky and an `_opt` site does not; the positive control (knob off) actually
computes `1 + 2`; the node-level test makes a real `Op::Add` node fail with `Err` naming
`launch:add_f32`; and the Err-arm test drives an f16 matmul whose Rust wrapper returns `Err` (so the
sticky is pending when `execute_node_inner` errs) and asserts the drain still happened.

**Acceptance results.**

| check | before | after |
|---|---|---|
| `scripts/check_cuda_launch_returns.py` | **104** unchecked sites | **empty list** (120 / 120) |
| CUDA serial unit suite | **531 / 0 / 37** (master `09406ce`; the recorded 526 predated #173's +3 and #98's +2) | **536 / 0 / 37** (+5 device/env-gated tests) |
| `MINFER_TEST_ISSUE162=1 … issue162` | — | **5 / 0** |
| `compute-sanitizer --tool memcheck` over the test binary | 0 errors | **0 errors** over 536 |
| CUDA serial ignored, 0.5B / Qwen3-0.6B | 37 / 0 each | **37 / 0** each |

The sanitizer command is `compute-sanitizer --tool memcheck --target-processes all
target/release/deps/minfer-<hash> --test-threads=1` — wrapping the test binary, not `cargo` (wrapping
the whole wrapper script lets the tree launcher follow every build process and the harness stalls).

**Mutations (reverted; `src/cuda_kernels.cu` restored to `sha256 da2e00fb79442fcd03bd3618301b4014cd835f832bbafa4153b4af4283dcdbfb`).** Deleting the read at one single-site
family (`launch:add_f32`), one `switch` case (`launch:embed_rows__q4_k`) and one templated branch
(`launch:gqa_attn_split_f16kv__hybrid_causal`) each makes the audit exit 1 naming that line; making
`minfer_launch_ok` report but admit the launch, removing the sticky, removing `execute_node`'s drain,
and making `minfer_launch_block` a no-op each fail a device test rather than the audit (the
mutation-proof half of the pair); disabling the audit's lever check fails its selftest; changing a
fixture fragment fails `--check-fixture`. Transcripts in the closing comment on #162 and in
`docs/ARCHITECTURE-EXECUTION-PLAN.md`.

**Honest scope.** Nothing was failing on this device: the sanitizer was already 0, the production
paths are latent, and the evidence is that every site *can* be driven into a real
`cudaErrorInvalidValue` and then names itself. The source audit is static and cannot tell whether the
read runs; the device gate compares **sets** of site tokens, so where two source sites share a token
(the two `mmq_raw_nb_bt` kernel instantiations) it proves the token is reached, not both branches —
the audit, not the gate, guarantees each source site has its own read. The parser resolves a site
variable only through a plain `=` in the enclosing function, so `launch_gemm_f16`'s `launch_site`
ternary resolves to its first arm (`launch:gemm_f16_a32`), which is what the driver arms. The
sanitizer count and the device limit (101376 B) are dgxspark's; CI has no GPU, and its `check-docs`
job enforces only the source half.

### 7.10 Issue #189 verification — the S4 map-window A/B is a paired sign test with a value arm (GB10, sm_121, CUDA 13.0, driver 580.178.04, 2026-09-27)

**The defect.** `graph::cuda_backend::tests::cuda_map_window_costs_no_more_than_the_span_it_replaces` was a pure stopwatch. It interleaved matched rounds of the *span* window (`row0 + i`) and the *map* window (row resolved through a run list) and asserted the **median of 9 per-round ratios** `<= 1.25x`. It failed once in the parallel `#[ignore]`d device run (GB10 sm_121, CUDA 13.0, driver 580.178.04, 2026-09-26) on the prefill phase:

```text
[s4-ab] prefill nt=512 nkv=512 hd=128: span 89.9 / map 125.1 us/launch (median of 9 interleaved
rounds of 50); per-round ratios [0.632, 0.697, 0.989, 1.091, 1.398, 1.440, 1.466, 2.198, 6.695]
— median 1.398x
```

A median of 9 flips once 5 pairs are disturbed, and that run had exactly 5 above the bar. The bar itself was justified by #123 as "16 CPU spinners plus two concurrent CUDA attention loops" (median 1.079–1.145 prefill); the parallel `#[ignore]`d suite is a far heavier co-tenant (~38 device tests, several capturing CUDA graphs, one GPU). The verdict was about how loaded the box was, not about the kernel — the #154 class, not the #185 SIGSEGV.

**The fix, two halves.**

1. **A value arm first (gate contract rule 1).** Before any timing the gate asserts
   - a **one-row** map window at cell 512 returns exactly that row's V — an absolute value computed on the host, not a relation between the two modes (softmax over one key is exactly 1.0, so the kernel is the identity on V);
   - a **two-run** map window at cell 512 returns the span's bytes over the same rows, bit for bit — f32 KV at the decode shape (`nt = 1`), f16 KV at the FA-prefill shape (`nt = 512`). One run is indistinguishable to a resolver that reads `(cell, len)` as `(lo, hi)`; two runs, at a non-zero base, are not;
   - the map instantiation actually ran, by counted observation rather than the dispatch's own report: `testfail::note_checked("cuda_attn_map_window")` is bumped in the launcher (`CudaState::gqa_attn_split` / `gqa_attn_kv_prefill`, `src/cuda.rs`), and the gate resets it, runs one span call (counter stays 0) and one map call (counter 1). The bitwise arms say *what* was computed; the counter says the map path was the thing that computed it.
2. **A paired sign test (gate contract rule 4).** The timing verdict is the **count** of matched pairs whose map arm is above `1.25x` its own span arm, and the gate refuses only at **7 of 9** — the one-sided sign test at `alpha = 46/512 = 0.090`. A load spike that disturbs a minority (or even a bare majority) of pairs cannot decide it; the recorded failure's 5 does not; a doubled map cost moves all 9 and does. The bar is unchanged at 1.25x and the timed fixture is the pre-#189 one (constant K/V, one run at cell 0), so the recorded margins stay comparable; the round count is fixed in advance (`PAIRS = 9`) and every per-round ratio and the refusal count are printed.

**Measured — the gate alone, idle (GB10 sm_121, CUDA 13.0, driver 580.178.04, 2026-09-27; `cargo test --release --features cuda --bin minfer cuda_map_window_costs_no_more_than_the_span_it_replaces -- --ignored --nocapture --test-threads=1`).**

| phase | span / map µs/launch | per-round ratios | refusals at 1.25x | verdict |
|---|---|---|---|---|
| decode (nkv 2048, nh 28, nk 4, hd 128) | 24.6 / 25.0 | [0.871, 1.007, 1.011, 1.013, 1.014, 1.017, 1.044, 1.050, 1.051] | **0 / 9** | pass |
| prefill (nt 512, f16 KV) | 82.8 / 91.1 | [1.054, 1.089, 1.097, 1.099, 1.100, 1.100, 1.104, 1.104, 1.189] | **0 / 9** | pass |

**Measured — the parallel `#[ignore]`d configuration (the one that produced the failure), six runs (GB10 sm_121, CUDA 13.0, driver 580.178.04, 2026-09-27; `cargo test --release --features cuda --bin minfer -- --ignored --nocapture`, default parallel harness).** The gate's refusal counts per run, and the whole set's result:

| run | decode refusals | prefill refusals | worst per-round ratio seen | suite |
|---|---|---|---|---|
| 1 | 0 / 9 | 3 / 9 | 1.57 (prefill) | **39 passed / 0 failed / 0 ignored** |
| 2 | 1 / 9 | 3 / 9 | 2.07 (prefill) | **39 / 0 / 0** |
| 3 | 0 / 9 | 3 / 9 | 1.48 (prefill) | **39 / 0 / 0** |
| 4 | 1 / 9 | 2 / 9 | 5.09 (decode) | **39 / 0 / 0** |
| 5 | 0 / 9 | 0 / 9 | 1.15 (decode) | **39 / 0 / 0** |
| 6 | 2 / 9 | 3 / 9 | 4.09 (decode) | **39 / 0 / 0** |

The co-tenant moved up to 3 of 9 pairs above the bar (the recorded failure's 5 is inside the tolerance); a single disturbed pair reached **5.09x** in run 4 and the sign test still returned green. The whole set was **6 / 6 green**, including the `#154` batching gate that had also failed once in this configuration. Honest reading: in these six runs the co-tenant was lighter than in the recorded one — no run's *median* exceeded 1.25x — so they show the gate is not decided by the co-tenant, not that the new statistic rescued a median-red run. That case is the recorded distribution itself, replayed by the pure test `graph::cuda_backend::tests::the_s4_ab_statistic_absorbs_a_loaded_run_and_still_refuses_a_real_regression` (5 of 9 refusals pass at 7; the old median of the same ratios is 1.398x and red).

The parallel configuration is reachable without removing anything: after #188 the `src/device_entry.rs` guard covers only `CudaState::layer_gpu` (`src/cuda.rs` ~line 6900, `#[allow(dead_code)]` legacy surface), and `BackendScheduler::execute` and `register_cuda_weight` no longer take it. No guard change is part of this ticket.

**Mutation evidence (rule 3).** `MINFER_S4_AB_MAP_REPS=2` (`src/cuda.rs::s4_ab_map_reps`) issues every **map-mode** attention launch twice, so the gate's timed map arm pays twice the work — the reproducible form of #123's map-work doubling, and an implementation mutation rather than a test edit. With it armed, the same command on the same binary fails in the decode phase:

```text
[s4-ab] decode nkv=2048 nh=28 nk=4 hd=128: span 24.6 / map 51.7 us/launch (9 interleaved matched
pairs of 100); per-round ratios [1.706, 1.745, 2.052, 2.091, 2.095, 2.100, 2.102, 2.136, 2.358]
— 9/9 above 1.25x (sign test refuses at 7)
thread '...cuda_map_window_costs_no_more_than_the_span_it_replaces' panicked at
src/graph/cuda_backend.rs: the map window is above 1.25x the span in 9 of 9 matched pairs ...
FAILED
```

The seam is an env switch, so the unmutated run is the same binary with the variable unset (no source revert to check); `git diff` on the tree contains only the #189 change, no mutation residue. The value arm, the counter arm and the timing arm all still run under the mutation.

**The rest of the device suite (GB10 sm_121, CUDA 13.0, driver 580.178.04, 2026-09-27).** `scripts/cuda_test.sh` → **546 / 0 / 39** (was 545 / 0 / 39; +1 the pure statistic test `the_s4_ab_statistic_absorbs_a_loaded_run_and_still_refuses_a_real_regression`). `FEATURES=cuda scripts/real_model_gates.sh` → **39 / 0** for the 0.5B config and **39 / 0** for the Qwen3-0.6B config. `compute-sanitizer --tool memcheck --target-processes all <test binary> --test-threads=1` → **0 API errors** over 546 / 0 / 39. `cargo test --release` (CPU) → 465 / 0 / 33 unit + 10 / 0 / 6 integration, unchanged (the new test lives in the `cuda`-gated module). `python3 scripts/check_status.py --check` → exit 0.

**Honest scope.** Every number here is a local GB10 measurement; CI has no GPU, so its CUDA job only compiles the harness. The two-run value arms are the gate's *own* fixtures, not the graph path — the graph-path bitwise coverage stays with `cuda_map_window_matches_the_span_over_the_same_rows` (which sweeps f32/f16/q8_0 over one/two/three runs and both batch shapes). The `MINFER_S4_AB_MAP_REPS` seam is wired to the two launches this gate drives (`gqa_attn_split`, `gqa_attn_kv_prefill`), not to the batched split path. The bar 1.25x is inherited unchanged from #123; this ticket changed the statistic and added the value arm, and did not widen it.

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

