# llama.cpp Metal (MPS) End-to-End Inference Path — Reference Baseline

> This document records, line-by-line, llama.cpp's end-to-end GPU inference
> implementation on Apple Silicon (Metal / MPS backend), as the reference
> starting point for comparing minfer against llama.cpp. Each row gives the
> **execution order**, **purpose**, and **source location (`file:line`)**, plus
> a **minfer equivalent** comparison column.
>
> Scope: the full chain from "model weight loading" to "next-token logits
> readback" (including batch preparation, graph build, scheduler, Metal
> execution, KV cache), with the Metal backend internals recorded at the
> finest granularity. The CPU-side sampler is not expanded (non-Metal path).

## 0. Version & reading conventions

- **llama.cpp baseline**: `master @ 8a832e4bf` (2026-08-20). This revision uses
  the per-arch graph-build API (`src/models/` + `llm_graph_context`); the Metal
  backend spans 6 files under `ggml/src/ggml-metal/`.
- **minfer baseline**: `master @ ad040ff` (2026-08-20).
- Paths are relative to each repo root (llama.cpp / minfer).
- Comparison-column value convention:
  - **Match** = minfer has an equivalent implementation;
  - **N/A** = minfer has no such step (architectural difference);
  - **Similar** = functionally equivalent but structurally different (noted).

### llama.cpp Metal related files

| File | Role |
|---|---|
| `ggml/src/ggml-metal/ggml-metal.cpp` | Metal backend interface (buffer types, set/get tensor, graph_compute entry) |
| `ggml/src/ggml-metal/ggml-metal-device.m` | Low-level MTLDevice/MTLCommandQueue, encoder, buffer allocation, kernel library loading |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | Pipeline (kernel instance) lookup/compilation, op support table |
| `ggml/src/ggml-metal/ggml-metal-context.m` | Multi-command-buffer scheduling, graph compute, tensor set/get |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | Per-op encoding (encoder setup + kernel dispatch + fusion) |
| `ggml/src/ggml-metal/ggml-metal.metal` | Metal kernel source |
| `ggml/src/ggml-metal/ggml-metal-impl.h` | Quantized block structs, dequant functions, threadgroup constants, function-constant offsets |
| `ggml/src/ggml-backend.cpp` | Backend scheduler (split / alloc / compute) |
| `src/llama-graph.cpp` | Graph-build helpers (`build_*`, `llm_graph_context`) |
| `src/llama-context.cpp` | `decode` / `process_ubatch` / `graph_compute` / logits readback |
| `src/llama-kv-cache.cpp` | KV cache (allocation, slot lookup, in-graph write/read) |
| `src/llama-model.cpp` / `src/llama-model-loader.cpp` | Model loading & weight registration |
| `src/models/qwen2.cpp` | Qwen2 architecture graph build |

## 1. High-level overview (12 phases)

| # | Phase | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| P0 | Backend & scheduler init | Create Metal backend, MTLCommandQueue, kernel library, scheduler; pre-reserve compute buffers | `ggml-metal.cpp:689` `ggml-metal-context.m:84` `llama-context.cpp:581` `ggml-backend.cpp:1792` | `src/metal.rs:1090` (MpsState::try_new) |
| P1 | Model loading / weight registration | Allocate Metal buffers per tensor and upload quantized weights | `llama-model.cpp:1401` `llama-model-loader.cpp:1426` `ggml-metal-device.m:1631` | `src/models/qwen2/loader.rs` + `src/metal.rs:1302` (register_weight) |
| P2 | Batch preparation & microbatching | Split the API batch into micro-batches, reserve host output buffers | `llama-context.cpp:1635` `llama-batch.cpp:25` | `src/main.rs` (single batch, no split) "N/A" |
| P3 | Compute graph build (Qwen2) | Build the ggml compute graph (DFS topological order) | `src/models/qwen2.cpp:53` `llama-graph.cpp` `ggml.c:7188` | `src/models/qwen2/forward.rs:6` (imperative encode, not a graph) |
| P4 | Scheduler split & allocation | Assign nodes to backends, split into runs, gallocr allocation | `ggml-backend.cpp:1936`→`:1055` | "N/A" (single MPS backend, static buffers) |
| P5 | Scheduler compute | Per-split: copy inputs, call backend graph_compute | `ggml-backend.cpp:1594` `ggml-metal.cpp:535` | `forward.rs:88-134` (single CB, all layers) |
| P6 | Metal graph compute | Multi-command-buffer encode (main thread + n_cb workers) | `ggml-metal-context.m:438` `:663` | `src/metal.rs:1032` (submit, single CB) |
| P7 | Per-op encoding & concurrency/barrier | Filter empty nodes, concurrency check, insert memoryBarrier | `ggml-metal-ops.cpp:175` `device.m:513` | `src/metal.rs:321` (barrier), `:327` (dispatch_2d) |
| P8 | Per-op kernel dispatch | Per op type: set pipeline + args + threadgroups | `ggml-metal-ops.cpp:265-497` | `src/metal.rs:392` (quant_matmul_f32_on_gpu_buf) et al. |
| P9 | GPU kernel execution | Metal shader compute | `ggml-metal.metal` | `src/metal.metal` |
| P10 | KV cache | In-graph write (set_rows) & read (flash/matmul direct) | `llama-kv-cache.cpp:1301` `llama-graph.cpp:2800` | `src/metal.rs:866` (store_kv) + `src/cache.rs` |
| P11 | logits/embd readback | GPU→host copy | `ggml-metal-context.m:351` `llama-context.cpp:1854` | `src/metal.rs:2012` (output_norm_gpu then download_logits, now `n_out` rows / 608 KB) |

## 2. Detailed step table (end-to-end execution order)

### P0 Backend & scheduler init (once per process)

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 0.1 | Register Metal backend | Construct backend per device, call `ggml_metal_init` | `ggml-metal.cpp:689` | — |
| 0.2 | Create `struct ggml_metal` context | MTLDevice + shared MTLCommandQueue, load kernel library, create concurrent dispatch queue, fusion/concurrency flags | `ggml-metal-context.m:84-175` | `src/metal.rs:1090-1200` (MpsState singleton) — **2026-08-21: kernel library is now a build-time precompiled `.metallib` embedded in the binary (`build.rs`, llama's `-O3` flags, `newLibraryWithData`), with a runtime `newLibraryWithSource` fallback when the toolchain is absent** |
| 0.3 | Init low-level device | `MTLCreateSystemDefaultDevice` + `newCommandQueue`, probe capabilities (simdgroup_mm / unified_memory / bfloat / tensor) | `ggml-metal-device.m:714-760` | `src/metal.rs` (`new_device` capability probe) |
| 0.4 | **tensor-API gate** | `has_tensor` defaults to OFF for pre-M5/M6/A19/A20 (disabled on M4) | `ggml-metal-device.m:753-760` | "N/A" (llama itself doesn't use the tensor API on M4) |
| 0.5 | Create scheduler | `ggml_backend_sched_new` (backend array + gallocr + events) | `ggml-backend.cpp:1792` | "N/A" |
| 0.6 | Pre-reserve worst-case graphs | Reserve pp (prefill) and tg (decode) graphs | `llama-context.cpp:630-657` | "N/A" (static buffers) |

### P1 Model loading / weight registration (once per process)

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 1.1 | Load architecture tensors | Qwen2's `load_arch_tensors` creates all weight tensors | `src/models/qwen2.cpp:19-47` | `src/models/qwen2/loader.rs` |
| 1.2 | Per-layer device split | Layers 0..i_gpu_start-1 stay on CPU; the rest split to GPU devices by free memory | `llama-model.cpp:1314-1323` | "N/A" (all-on-GPU or MINFER_DISABLE_MPS all-CPU) |
| 1.3 | Allocate Metal buffers | `ggml_backend_alloc_ctx_tensors_from_buft` → `ggml_metal_buffer_init`; mmap path `ggml_metal_buffer_map` | `llama-model.cpp:1637` `ggml-metal-device.m:1631,1701` | `src/metal.rs` (register_part: ONE page-aligned `newBufferWithBytesNoCopy` per mmap'd part) — **2026-08-21: weights are (buffer, byte-offset) into the part buffer, llama's exact design** |
| 1.4 | Buffer storage mode | shared = `newBufferWithBytesNoCopy` (mmap/weights); private = `newBufferWithLength` | `ggml-metal-device.m:1668,1673` | `src/metal.rs` (mmap parts `StorageModeShared` NoCopy; scratch/KV buffers `StorageModeShared` copies) |
| 1.5 | Mark weight buffers | `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` | `llama-model.cpp:1657` | — |
| 1.6 | Upload weight data | mmap direct reference; non-mmap uses blit `set_tensor_async` | `llama-model-loader.cpp:1548,1558` `ggml-metal-context.m:307` | **2026-08-21: zero-copy — weights are Borrowed slices of the mmap'd GGUF (`Tensor.data: Cow<'static,[u8]>`) wrapped by `newBufferWithBytesNoCopy` at the part level; no memcpy anywhere** |

### P2 Batch preparation & microbatching (per decode)

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 2.1 | Init batch allocator | `llama_batch_allocr::init` | `src/llama-batch.cpp:25` | — |
| 2.2 | Split micro-batches | `memory->init_batch`, retry on failure (cache optimization) | `llama-context.cpp:1828-1856` `llama-kv-cache.cpp:698` | "N/A" (minfer single batch; `-n 0` = whole-segment prefill) |
| 2.3 | Reserve host output | `output_reserve` fixed-size logits/embd buffers | `llama-context.cpp:2032` | `forward.rs:141` (new logits vec each call) |
| 2.4 | Microbatch loop | Call `process_ubatch` per ubatch | `llama-context.cpp:1879-1900` | — |

### P3 Compute graph build (Qwen2)

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 3.1 | Entry | `llama_model::build_graph` → `build_arch_graph` | `llama-model.cpp:2457` | `src/models/qwen2/forward.rs:6` (forward) |
| 3.2 | Input embd | `build_inp_embd`: token ids + `ggml_get_rows(tok_embd, inp_tokens)` | `llama-graph.cpp:2284` | `src/metal.rs:1594` (embed_tokens_gpu, get_rows) — **2026-08-21: Q4_0 + Q4_K on GPU (was Q4_0-only; 7B/1.5B Q4_K embd fell back to CPU)** |
| 3.3 | Position input | `build_inp_pos` | `llama-graph.cpp:2373` | `src/metal.rs:1565` (upload_positions) |
| 3.4 | KV graph inputs | `build_attn_inp_kv` (k_idxs/v_idxs, mask, rotation tensors) | `llama-graph.cpp:2729` | `src/metal.rs` (store_kv uses pos_buf) |
| 3.5 | Per layer: attn_norm | `build_norm` (RMSNorm + Mul + optional Add) | `llama-graph.cpp:1556` | `src/metal.rs:614` (rms_norm) + `:648` (add) |
| 3.6 | Per layer: QKV | `build_qkv` (3× `build_lora_mm` = ggml_mul_mat for wq/wk/wv) | `llama-graph.cpp:1592` | `src/metal.rs:392` (3× quant_matmul) |
| 3.7 | Per layer: RoPE | `ggml_rope_ext` (once each for Q, K) | `src/models/qwen2.cpp:86-96` | `src/metal.rs:719` (rope_f32 ×2) |
| 3.8 | Per layer: KV write | `mctx_cur->cpy_k/cpy_v` → `ggml_set_rows` | `llama-graph.cpp:2800-2801` `llama-kv-cache.cpp:1301,1336` | `src/metal.rs:866` (store_kv) |
| 3.9 | Per layer: attention | `build_attn` → `build_attn_mha`: flash path `ggml_flash_attn_ext`; non-flash `mul_mat(k,q)`+`soft_max_ext`+`mul_mat(v,kq)` | `llama-graph.cpp:2517,2557` | `src/metal.rs:949` (attn_flash_prefill) / `:741` (gqa_attn_f32) |
| 3.10 | Per layer: wo + residual | `build_attn` inner `build_lora_mm(wo)` + `ggml_add` | `llama-graph.cpp:2677` `src/models/qwen2.cpp:110` | `src/metal.rs:392` (wo matmul) + `:648` (add) |
| 3.11 | Per layer: ffn_norm | `build_norm` | `src/models/qwen2.cpp:114` | `rms_norm` |
| 3.12 | Per layer: FFN | `build_ffn`: SILU-gated `mul(gate,up)` + `mul_mat(down)` | `llama-graph.cpp:1669` | `src/metal.rs:692` (swiglu) + `:392` (down matmul) — **2026-08-21: last layer runs on `n_out` rows only (llama's `get_rows` reduction, #34)** |
| 3.13 | Per layer: residual | `ggml_add` | `src/models/qwen2.cpp:127` | `add_f32` — **2026-08-21: last layer's both residuals on the tail `n_out` rows (`add_f32_off`, #34)** |
| 3.14 | Output norm | `build_norm` (result_norm) | `src/models/qwen2.cpp:137` | `src/metal.rs:2072` (rms_norm inside output_norm_gpu) — **2026-08-21: also output-rows-only (`n_out`), matching llama** |
| 3.15 | lm_head | `build_lora_mm(model.output)` + optional bias | `src/models/qwen2.cpp:145-150` | `src/metal.rs:2078` (output GEMM) — **2026-08-21: now output-rows-only (`n_out`), matching llama** |
| 3.16 | Node ordering | `ggml_build_forward_expand` → `ggml_build_forward_impl` → `ggml_visit_parents_graph` (DFS, parents before children) | `ggml.c:7188,7120` | "N/A" (minfer encodes imperatively in layer order) |

### P4 Scheduler split & allocation

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 4.1 | Split graph | `ggml_backend_sched_split_graph`: 5-pass node→backend assignment (weight's backend decides MUL_MAT), build splits, insert cross-backend tensor_copy | `ggml-backend.cpp:1055-1443` | "N/A" (all Metal) |
| 4.2 | Allocate memory | `ggml_gallocr_alloc_graph` (retry via `reserve_n` on failure) | `ggml-backend.cpp:1562-1585` | "N/A" (static buffers) |

### P5 Scheduler compute (per split)

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 5.1 | Copy split inputs | Copy cross-backend srcs to the split's backend (INPUT flag → sync copy; MoE copies only used experts; else async) | `ggml-backend.cpp:1555-1671` | "N/A" |
| 5.2 | Call backend compute | `ggml_backend_graph_compute_async` → Metal's `ggml_backend_metal_graph_compute` | `ggml-backend.cpp:1678` `ggml-metal.cpp:535` | `forward.rs:134` (cb.submit) |
| 5.3 | Event record | MTLEvent signal for multi-copy scenarios | `ggml-backend.cpp:1717-1721` | "N/A" |

### P6 Metal graph compute (multi-command-buffer scheme)

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 6.1 | Split work | `n_main = MAX(64, 0.1*n_nodes)`; first n_nodes_0 nodes encoded by main thread, rest split evenly by n_cb | `ggml-metal-context.m:445-466` | — |
| 6.2 | Main-thread encode | Create `cmd_bufs[n_cb]`, enqueue, `encode_async(n_cb)` | `ggml-metal-context.m:510-523` | — |
| 6.3 | Worker encode | `dispatch_apply(n_cb, d_queue, encode_async)` encodes remaining CBs concurrently | `ggml-metal-context.m:530-550` | — |
| 6.4 | encode_async block | Per CB: compute node range, `ggml_metal_op_init` → loop `ggml_metal_op_encode` → `ggml_metal_op_free` → commit | `ggml-metal-context.m:676-721` | — |
| 6.5 | Async return | graph_compute returns immediately (only capture mode waits + checks status) | `ggml-metal-context.m:557-611` | `src/metal.rs:1032` (submit blocks + 10 s cap) |
| 6.6 | Synchronize | `ggml_metal_synchronize`: wait + check all CB statuses, set `has_error` on failure | `ggml-metal-context.m:239-295` | `submit()` `MTLCommandBufferStatus` check |

> **n_cb value**: `ggml_backend_metal_set_n_cb(backend, 1)` (`ggml-metal.cpp:612,707`),
> `ggml_metal_set_n_cb` caps at `GGML_METAL_MAX_COMMAND_BUFFERS` (`context.m:665`).
> That is **2 CBs** (1 main + 1 worker). minfer uses a **single CB for all
> layers** (`forward.rs:76-134`).

### P7 Per-op encoding & concurrency/barrier model

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 7.1 | Create encoder | `ggml_metal_encoder_init`: `MTLDispatchTypeConcurrent` (when use_concurrency) or serial | `ggml-metal-ops.cpp:42` `ggml-metal-device.m:464` | `src/metal.rs:1324` (cmd_buffer: `new_compute_command_encoder`, serial) |
| 7.2 | Filter empty nodes | Skip empty / no-op nodes | `ggml-metal-ops.cpp:55-62` | "N/A" (minfer has no empty-node concept) |
| 7.3 | Op support check | `ggml_metal_device_supports_op` big switch | `ggml-metal-ops.cpp:201` `ggml-metal-device.m:1086` | quant-type checks in `layer_gpu` |
| 7.4 | Concurrency check | If the current node's read/write ranges conflict with existing `mem_ranges`, insert `memoryBarrierWithScope:MTLBarrierScopeBuffers` and clear ranges; else record ranges and run concurrently | `ggml-metal-ops.cpp:159-173,220-225` `ggml-metal-device.m:513` | `src/metal.rs:321` (barrier: **after every dispatch**) + `:333` |
| 7.5 | Op dispatch switch | Dispatch to `ggml_metal_op_*` by `node->op`; returns fusion count n_fuse | `ggml-metal-ops.cpp:265-497` | encode in fixed sequence inside `layer_gpu` |

> **Key difference**: llama uses `mem_ranges` for **dependency-aware barriers**
> (non-conflicting adjacent ops run concurrently in the same encoder,
> `MTLDispatchTypeConcurrent`); minfer inserts an unconditional barrier after
> every dispatch (`dispatch_2d` → `barrier()`, `src/metal.rs:333`).

### P8 Per-op kernel dispatch (forward-path ops)

| ggml_op | llama.cpp encoder | Selected kernel (variants) | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| `MUL_MAT` | `ggml_metal_op_mul_mat` (3-way selection, §3.2) | `kernel_mul_mm_*` / `kernel_mul_mv_ext_*` / `kernel_mul_mv_*` | `ops.cpp:2299-2541` | `src/metal.rs:392` (quant_matmul_f32_on_gpu_buf) + `:352` (gemm_dispatch) |
| `FLASH_ATTN_EXT` | `ggml_metal_op_flash_attn_ext` | `_kv_f16` / `_pad` / `_blk` / main kernel / `_vec` / `_vec_reduce` | `ops.cpp:2990-3492` | `src/metal.rs:949` (attn_flash_prefill) |
| `RMS_NORM` | `ggml_metal_op_norm` (fuses Mul+Add) | `kernel_rms_norm_fuse_impl` | `ops.cpp:3887-4006` `metal.metal:3181` | `src/metal.rs:614` (rms_norm) + separate `:648` (add) |
| `ROPE` | `ggml_metal_op_rope` | `kernel_rope_norm/neox/multi/vision` | `ops.cpp:4025-4126` `metal.metal:4664-4868` | `src/metal.rs:719` (rope_f32) |
| `ADD/SUB/MUL/DIV` | `ggml_metal_op_bin` (ADD fusion ×8) | `kernel_add` / `kernel_mul` (n_fuse specialization) | `ops.cpp:3578` | `src/metal.rs:648` (add_f32 single op) |
| `GET_ROWS` | `ggml_metal_op_get_rows` | `kernel_get_rows_q/_f` | `ops.cpp:1165` `metal.metal:10061` | `src/metal.rs:599` (embed_tokens_gpu → get_rows_q4_0) |
| `SET_ROWS` | `ggml_metal_op_set_rows` | `kernel_set_rows_*` | `ops.cpp:1210` `metal.metal:10156` | `src/metal.rs:866` (store_kv dedicated kernel) |
| `CPY/DUP/CONT` | `ggml_metal_op_cpy` | `kernel_cpy_t_t/_f32_q/_q_f32` | `ops.cpp:2078` `metal.metal:8023-8122` | "N/A" (f16 KV converted directly by store_kv) |

### P9 GPU kernel execution (key kernels)

| kernel | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|
| `kernel_mul_mm<...>` | simdgroup/tensor matmul (64×32 tile, §3.2 variants) | `metal.metal:10658-11040` (template + instantiations) | `src/metal.metal:758` (kernel_q4_0_mm_f32) and 7 more mm kernels |
| `kernel_mul_mv_*` | mat-vec (decode, per quant type) | `metal.metal:3847` (q4_0), `:8498` (q4_K), etc. | `src/metal.metal` `*_f32_matmul` kernels |
| `kernel_mul_mv_ext_*` | small-batch (ne11∈[2,8]) mat-mv | `metal.metal:4196` | "N/A" |
| `kernel_flash_attn_ext_kv_f16` | **quantized KV → f16 dequant pre-pass** (Q4_0/1, Q5_0/1, Q8_0) | `metal.metal:6328-6366` | "N/A" (minfer KV stores f32/f16 raw, `MINFER_CACHE_TYPE=f16`) |
| `kernel_flash_attn_ext_pad` | pad pre-pass for partial KV blocks | `metal.metal:6373` | `src/metal.metal: kernel_kv_tail_pad` (equivalent) |
| `kernel_flash_attn_ext_blk` | mask pre-pass (nqptg/ncpsg blocks) | `metal.metal:6445` | inline causal mask (`kernel_flash_attn_blk_f32`) |
| `kernel_flash_attn_ext` / `_impl` | flash attention main kernel (half8x8) | `metal.metal:6546,7184` | `src/metal.metal: kernel_flash_attn_blk_f32` |
| `kernel_flash_attn_ext_vec` / `_vec_reduce` | decode small-batch flash (half4x4, ne01<20) | `metal.metal:7411,7980` | `src/metal.metal: kernel_flash_attn_ext_f32` |
| `kernel_rms_norm_fuse_impl` | RMSNorm + Mul + Add fusion | `metal.metal:3181` | `rms_norm_256` + separate add |
| `kernel_soft_max*` | non-flash path softmax | `metal.metal:2011,2117` | "N/A" (inlined in flash; or a dedicated `softmax` kernel) |
| `kernel_rope_*` | RoPE | `metal.metal:4664-4868` | `src/metal.metal: kernel_rope_f32` |
| `kernel_get_rows_*` | embedding lookup | `metal.metal:10061,10092` | `src/metal.metal: kernel_get_rows_q4_0` |

### P10 KV cache

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 10.1 | KV tensor creation | `ggml_new_tensor_3d(ctx, type_k/v, n_embd_k_gqa, kv_size, n_stream)`, default **F16** | `llama-kv-cache.cpp:231-232` | `src/cache.rs` (GPU KV f16 auto for 7B class / f32 for small, `MINFER_CACHE_TYPE` overrides) |
| 10.2 | Per-layer device allocation | per-layer backend buft, allocate + clear | `llama-kv-cache.cpp:299,307` | `src/metal.rs` (KV buffer allocation) |
| 10.3 | Slot lookup | `find_slot`: ring-buffer cell range + k/v idx tensors | `llama-kv-cache.cpp:894` | `src/metal.rs: store_kv` writes by `pos_buf` |
| 10.4 | In-graph write | `cpy_k`/`cpy_v` → `ggml_set_rows` (K always cache-row indexed; V per FA/non-FA layout) | `llama-kv-cache.cpp:1301-1389` | `src/metal.rs:866` (store_kv, `nkt`/`nt` strides) |
| 10.5 | In-graph read | flash reads cache tensor directly; non-flash `mul_mat(k,q)`/`mul_mat(v,kq)` | `llama-graph.cpp:2807-2808,2491,2535` | `attn_flash_prefill` / `gqa_attn_f32` read KV buffers directly |

> **Layout**: llama KV = f16 `[nkv][nk*hd]`, token stride `nk*hd*elem` (after
> `llama-graph.cpp` permute, flash receives `nb11=nk*hd*elem`). minfer uses the
> same layout but f32 by default, f16 optional (`MINFER_CACHE_TYPE=f16`).

### P11 logits/embd readback

| # | Step | Purpose | llama.cpp location | minfer equivalent |
|---|---|---|---|---|
| 11.1 | Locate backend | `ggml_backend_sched_get_tensor_backend(t_logits)` | `llama-context.cpp:1948` | — |
| 11.2 | Async readback | `ggml_backend_tensor_get_async` → `ggml_metal_get_tensor_async`: `newBufferWithBytesNoCopy` wraps host memory + **blit encoder** GPU→host, queued into `cmd_bufs_ext` | `llama-context.cpp:1854` `ggml-metal-context.m:351-391` | `src/metal.rs` (`copy_from_gpu`: Shared buffer direct memcpy, no blit) — **2026-08-21: now `n_out×nv` (608 KB for single output; was 301 MB)** |
| 11.3 | Synchronize | before the next decode, `ggml_backend_sched_synchronize` waits for the blit | `ggml-backend.cpp` | `submit()` blocks + `download_logits` |

## 3. Supplementary mapping tables

### 3.1 Per-op dispatch switch (`ggml-metal-ops.cpp:265-497`)

Complete forward-path mapping (non-forward ops omitted):

| ggml_op | handler | fusion |
|---|---|---|
| CONCAT | `ggml_metal_op_concat` | — |
| ADD/SUB/MUL/DIV | `ggml_metal_op_bin` | ADD ×N (up to 8 consecutive ADDs → 1 dispatch); Snake/GEGLU specialization |
| ADD_ID | `ggml_metal_op_add_id` | — |
| SOFT_MAX | `ggml_metal_op_soft_max` | — |
| MUL_MAT | `ggml_metal_op_mul_mat` | — |
| MUL_MAT_ID | `ggml_metal_op_mul_mat_id` (MoE) | — |
| GET_ROWS / SET_ROWS | `op_get_rows` / `op_set_rows` | — |
| NORM / RMS_NORM | `ggml_metal_op_norm` | **RMSNorm + Mul(weight) + Add(bias)** in 1 kernel |
| ROPE / ROPE_BACK | `ggml_metal_op_rope` | — |
| FLASH_ATTN_EXT | `ggml_metal_op_flash_attn_ext` | QK^T + softmax + PV single kernel (+aux pad/blk/kv_f16/vec_reduce) |
| DUP / CPY / CONT | `ggml_metal_op_cpy` | — |
| SILU_BACK / GLU | `op_silu_back` / `op_glu` (training/gating) | — |

### 3.2 `MUL_MAT` 3-way kernel selection (`ggml-metal-ops.cpp:2336-2538`)

| Branch | Trigger condition | kernel / pipeline | threadgroups | llama.cpp location |
|---|---|---|---|---|
| ① mat-mv ext | src1=f32, `ne00%128==0`, src0 type in supported set, **ne11∈[2,8]** (K-quants need ne11∈[4,8]) | `kernel_mul_mv_ext_*` (nsg=2, nxpsg per ne00: 16/8/4) | (ne01/r0ptg, ne11/r1ptg, ne12·ne13), 32×nsg | `ops.cpp:2340-2439` `device.cpp:706` |
| ② simdgroup MM | non-transposed, `has_simdgroup_mm`, `ne00>=64`, `ne11>8` | `kernel_mul_mm_<t0>_<t1>` (function-constants `bc_inp/bc_out/ne12/ne13/r2/r3`) | (ne11/nr1, ne01/nr0, ne12·ne13), 32×nsg; **M4: nr0=64, nr1=32, nsg=4, smem 8192** (bc_out) | `ops.cpp:2440-2490` `device.cpp:739-799` |
| ③ mat-vec | otherwise | `kernel_mul_mv_*` (per quant type) | nsg/nr0 per type | `ops.cpp:2491-2538` `device.cpp:801+` |

**Function-constant offsets** (`ggml-metal-impl.h:99-100`): `FC_MUL_MV=600`,
`FC_MUL_MM=700`; MM uses 700-705 (bc_inp/bc_out/ne12/ne13/r2/r3).

**On M4, llama disables the tensor API** (§0.4) and actually uses branch ②'s
legacy `simdgroup_matrix` path — which is **level-for-level equivalent** to
minfer's `kernel_q4_k_mm_f32` (`src/metal.metal:4692`, 64×32 tile, 32×4
threads, 8192 B smem) (see minfer `docs/METAL_OPTIMIZATIONS.md §3.6`).

### 3.3 `FLASH_ATTN_EXT` variant selection (`ggml-metal-ops.cpp:2990-3492`)

| Test | Variant | Trigger condition |
|---|---|---|
| `use_vec` | `_vec` + `_vec_reduce` | `ne01 < 20 && ne00 % 32 == 0` (decode small batch, half4x4) |
| `use_kv_f16` | first `kernel_flash_attn_ext_kv_f16` dequantizes KV→f16 | KV type ∈ {Q4_0,Q4_1,Q5_0,Q5_1,Q8_0} (**new in #27390**) |
| `has_kvpad` | first `kernel_flash_attn_ext_pad` | `ne11 % ncpsg != 0` (KV not a multiple of ncpsg) |
| `has_mask` | first `kernel_flash_attn_ext_blk` | mask present (block pre-pass) |
| main kernel | `kernel_flash_attn_ext` (half8x8, nqptg=8/ncpsg=64, nsg=ne00>=512?8:4) | prefill (non-vec path) |

**Middle-buffer layout** (`ops.cpp:3055-3065`): after dst, in order `pad` →
`blk` → `tmp` → `kv_f16` (sizes computed by `ggml_metal_op_flash_attn_ext_extra_*`).

### 3.4 Fusion rule summary

| Fusion | llama.cpp | minfer |
|---|---|---|
| RMSNorm + Mul(weight) + Add(bias) | `kernel_rms_norm_fuse_impl` (`ops.cpp:3929-3974`) | not fused: `rms_norm` + separate `add_f32` |
| Consecutive ADD ×N | `op_bin` (`ops.cpp:3195+`) | single add_f32 |
| flash attention (QK^T+softmax+PV) | single kernel + aux passes | `attn_flash_prefill` (`src/metal.rs:949`) |
| KV write + RoPE | RoPE writes into the KV path (graph k/v expanded together) | `store_kv` dedicated kernel |
| GLU/SiLU | `op_glu` / `op_snake_fused` | `swiglu_f32` (single kernel) |
| mul_mat + bias / residual | **not fused** (bias/residual is a separate `kernel_add`) | same (separate add_f32 after wo) |

### 3.5 Multi-CB and encoder/barrier model comparison

| Item | llama.cpp | minfer |
|---|---|---|
| CB count | `n_cb=1` → 2 CBs (main thread 64 nodes + 1 worker) | 1 CB (all 28 layers + output) |
| Encode parallelism | `dispatch_apply` multi-thread concurrent encode | single-threaded sequential encode |
| encoder dispatch type | `MTLDispatchTypeConcurrent` (on by default, `GGML_METAL_CONCURRENCY_DISABLE` to off) | serial (`new_compute_command_encoder`) |
| barrier | dependency-aware (`mem_ranges` conflict only inserts `memoryBarrierWithScope`) | unconditional `memoryBarrierWithScope` after every dispatch |
| commit/wait | main thread returns async, `synchronize` waits explicitly; 10 s timeout guard | `submit()` blocks on completed handler (10 s timeout + status check) |

## 4. Key data structures

| Struct | Defined at | Purpose / key fields |
|---|---|---|
| `struct ggml_metal` (ggml_metal_t) | `ggml-metal-context.m:26` | device, library, `d_queue`, `n_cb`, `cmd_bufs[]` (`cmd_bufs[n_cb+1]`), `encode_async` block, `cmd_bufs_ext`, `cmd_buf_last`, `has_error` |
| `struct ggml_metal_device` | `ggml-metal-device.m:521` | `mtl_device`, `mtl_queue` (globally shared), `rsets` (residency sets), `library`, `props`, `addr_virt` |
| `struct ggml_metal_encoder` | `ggml-metal-device.m:460` | wraps `MTLComputeCommandEncoder` |
| `struct ggml_metal_library` | `ggml-metal-device.m:97` | `MTLLibrary` + cached `MTLComputePipelineState` map + lock; `newLibraryWithSource` (`device.m:234`) |
| `struct ggml_metal_buffer` | `ggml-metal-device.m` (`buffer_init:1631`) | `buffers[]` (`{id<MTLBuffer>, offs}`), `is_shared`, `rset` |
| `struct ggml_metal_op` | `ggml-metal-ops.cpp:28` | per-CB encode state: `enc`, `mem_ranges`, filtered `idxs[]`, fusion flags |
| `struct ggml_backend_sched` | `ggml-backend.cpp` (`sched_new:1792`) | backend array, `splits[]`, `galloc`, `node/leaf_backend_ids[]`, `events[b][c]`, `graph_copy` |
| `struct ggml_backend_sched_split` | `ggml-backend.cpp:1055+` | `{backend_id, i_start, i_end, n_inputs, inputs[], graph}` |
| `llm_graph_context` | `llama-graph.h` | `ctx0`, `gf`, hparams/cparams, sched, res |
| `llm_graph_result` | `llama-graph.h` | `t_inp_tokens`, `t_logits`, `t_embd`, `inputs[]`, compute ctx + `ggml_cgraph` |
| `ggml_metal_pipeline_with_params` | `ggml-metal-ops.h` | `{pipeline, nr0, nr1, nsg, smem}` — everything a single kernel dispatch needs |

## 5. minfer comparison notes (for later comparison work)

1. **Architecture difference**: llama.cpp = declarative ggml graph (topological
   nodes → scheduler → backend); minfer = imperative (`forward.rs` encodes
   layer-by-layer directly into a single MPS command buffer). No
   scheduler/allocator layer; `src/cache.rs` holds KV directly.
1b. **Output-rows reduction (2026-08-21)**: llama shrinks the graph to
   `n_outputs` rows after the last attention (`get_rows(cur, inp_out_ids)` +
   `get_rows(inpSA, inp_out_ids)`, `qwen2.cpp:106-108`) → the last layer's
   FFN + both residuals + final norm + lm_head all run on 1 row. minfer mirrors
   the FULL reduction: final norm + lm_head via `n_out` (#32), and the last
   layer's FFN + residuals via `layer_gpu(n_out, is_last)` (#34) — minfer's
   total graph work (≈6.26 TFLOP) now exactly equals llama's.
2. **Level-for-level equivalence proven** (minfer `docs/METAL_OPTIMIZATIONS.md`
   §3.4/§3.6): the prefill GEMM kernels (`kernel_mul_mm` vs
   `kernel_q*_mm_f32`) match at source/IR/smem/dispatch/runtime-compile level;
   this table's P8/P9 rows are the comparison anchors.
3. **Fusion gap**: llama's RMSNorm+Mul+Add, ADD×N, single-kernel flash fusion vs
   minfer's mostly-separate dispatches (§3.4) — the source of the
   per-layer dispatch-count difference in decode/prefill.
4. **KV format**: llama defaults f16 + optional quantized KV (since #27390, a
   `kv_f16` dequant pass); minfer auto-selects f16 for the 7B class / f32 for small models (#37,
   `MINFER_CACHE_TYPE=f16/f32` overrides), no quantized KV.
5. **Multi-CB**: llama 2-CB concurrent encode; minfer single CB all layers.
   Measured (minfer §3.6): llama's 2-CB split is **slower** in a pure-GEMM
   replay — not a speed source.
6. **Barrier**: llama dependency-aware; minfer barriers after every dispatch.
   Measured free (§3.6) — not a gap source.
