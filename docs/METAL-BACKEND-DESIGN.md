# minfer Metal Backend Design

How minfer runs the compute graph on an Apple GPU: the `MetalBackend` graph executor, the `metal.rs`
device/kernel layer, the `metal.metal` shaders, the command-buffer rhythm, and the memory and safety
rules that hold it together.

> **Status.** Landed. The backend arrived with compute-graph Phase 3, and the G1–G6 wiring passes
> plus the objc2 migration brought it to parity with the pre-graph imperative path. Every mechanism
> described here is implemented in the tree.
> Baseline: `HEAD = 62a0a3e` (2026-09-14).
>
> **Provenance.** New document. It mirrors the layout of `docs/CUDA-BACKEND-DESIGN.md` and is the
> design of record for the Metal backend; the optimization campaign and its measurements stay in
> `docs/METAL_OPTIMIZATIONS.md`.
>
> **Related records.** `docs/METAL_OPTIMIZATIONS.md` is the optimization history and current-state
> ledger (including §0.1, the graph-path integration status), `docs/METAL_OBJC-ECOSYSTEM.md` and
> `docs/METAL-OBJC2-MIGRATION-PLAN.md` cover the objc2 crate migration, `docs/LLAMA_METAL_E2E.md`
> is the llama.cpp Metal reference, `docs/GPU_SAFETY.md` holds the hard safety rules (Metal
> sections), and `docs/inference_e2e_walkthrough/14-metal-backend.md` narrates the backend for a
> first-time reader. The graph contract this backend implements is
> `docs/COMPUTE-GRAPH-DESIGN.md` §3.5/§7.

---

## 1. Design Goals and Outcome

### 1.1 Goal

Implement `MetalBackend` (`src/graph/metal_backend.rs`) as the macOS backend of the compute graph by
**wiring the existing per-op kernels** on `MpsState` (`src/metal.rs`) — no new kernels required — so
that on Apple Silicon the whole per-layer chain runs on the GPU through the standard
`build → assign → fuse → alloc → execute` pipeline, with the same correctness contract as CPU:
backend placement is decided at build time, kernel-invariant violations return `Err`, and there is
never a silent mid-run fallback.

### 1.2 Outcome

| Goal | Landed outcome | Evidence |
|---|---|---|
| `MetalBackend` implements the `Backend` trait over `MpsState` | Full trait: shared-memory buffer pool, `weight_buf` offsets, per-op dispatch, direct host views, split-scoped command buffer | §2, §4.1–§4.2 |
| One command buffer per split, submitted at boundaries | `cb()` creates it lazily, `synchronize()` submits it; `Drop` flushes a pending one | §2.4, §4.8 |
| Per-node placement decided at build time | `supports_op` + the model-level `weights_on_gpu` all-or-nothing gate feed `CParams.gpu`; the scheduler splits the graph | §4.3, §4.6 |
| Zero-copy weights | GGUF parts are wrapped with `newBufferWithBytesNoCopy`; weights are `(buffer, byte offset)` pairs; a warm-up read moves the ~44 ms first-touch page cost to model load | §2.3 |
| Attention dispatch matches the old path | G1 wires nt==1 and nt>1 to the flash/split/parallel/classic kernels with the same gates and env vars | §4.4, §4.6 |
| Decode fusions on Metal | G4 `FusedQKV`, G5 `FusedFFN`, G6 `FusedQkvNorm` (Qwen3) are built as single nodes | §4.4, §4.6 |
| Same correctness gates as CPU | Per-op parity tests, cross-backend copy tests, model-level CPU-vs-Metal logits / greedy equality, kernel isolation tests | §7 |
| Performance | Graph path at or above the old imperative path: 0.5B decode ~299–331 tok/s (KV440, G4+G5) and prefill pp440 ~3900–4000 tok/s; 7B decode ≈ parity; Qwen3-4B decode ~75.9 tok/s ≈ llama-Metal 79.7 | `METAL_OPTIMIZATIONS.md` §0.1 |

### 1.3 Non-goals

- **A whole-layer `layer_gpu` fast path.** The pre-graph imperative path was removed; `metal.rs` is
  the per-op device/kernel layer only (the name survives solely in the legacy CUDA code).
- **f16/bf16 activations.** Graph activations are f32; the f16 story is the KV cache
  (`MINFER_CACHE_TYPE`) and the flash-attention f16-KV kernel variants.
- **New kernels for the graph.** The graph path is wiring plus the G2 `rms_norm_256` selection and
  the decode `*_off` kernel variants that already existed.
- **Multi-GPU / device selection.** One integrated GPU per Mac.
- **Training or fine-tuning.**

### 1.4 Related records

| Topic | Where |
|---|---|
| Optimization history, current state, graph-path status, env gates | `docs/METAL_OPTIMIZATIONS.md` |
| objc 0.2 → objc2 crate migration (phases, gotchas, checklist) | `docs/METAL-OBJC2-MIGRATION-PLAN.md`, `docs/METAL_OBJC-ECOSYSTEM.md` |
| llama.cpp Metal (MPS) end-to-end path (reference baseline) | `docs/LLAMA_METAL_E2E.md` |
| GPU safety rules (barriers, bounded submit, capture windows) | `docs/GPU_SAFETY.md` |
| Graph contract (IR, allocator, scheduler, backend trait) | `docs/COMPUTE-GRAPH-DESIGN.md` |
| Beginner narrative of this backend | `docs/inference_e2e_walkthrough/14-metal-backend.md` |
| Kernel-level analyses | `docs/metal-inference-analysis.md`, `docs/multi-token-kernel-analysis.md` |
| Qwen3-4B vs llama.cpp (Metal) | `docs/PERF-QWEN3-4B-VS-LLAMACPP.md` |

---

## 2. Architecture at a Glance

### 2.1 Three layers

| Layer | File | Role |
|---|---|---|
| Graph executor | `src/graph/metal_backend.rs` | Implements `Backend`: shared-memory buffer pool, name→buffer-offset weight resolution, per-op dispatch, split-scoped `MpsCommandBuffer`, capture staging, error contract |
| Device/kernel layer | `src/metal.rs` | `MpsState` singleton: device/queue/library init, zero-copy weight registry, `MpsCommandBuffer` (encoder, barriers, submit), and one Rust method per op/kernel |
| Shaders | `src/metal.metal` | The Metal Shading Language kernels (norms, matmul tiers, attention variants, elementwise, KV store, get_rows, fused epilogues) |
| Build chain | `build.rs` | Runtime shader compilation (default) or a precompiled metallib (`MINFER_METALLIB_FILE`/`_PATH`); the objc2 framework links |

The split follows CUDA: `metal.rs` is the only place that touches Objective-C/Metal APIs, and
`metal_backend.rs` is the only place that knows about graph nodes.

### 2.2 The `MetalBackend` surface

```rust
pub struct MetalBackend {
    state: &'static MpsState,                 // process-wide MPS singleton
    pool: Vec<MetalBuffer>,                   // f32-element pool: id -> shared MTLBuffer
    free: Vec<usize>,                         // exact-byte-length free list
    staging: Vec<MetalBuffer>,                // trace/viz capture staging (blit targets)
    free_staging: Vec<usize>,
    cb_ptr: *mut MpsCommandBuffer<'static>,   // pending command buffer (null = none)
}
```

Pool buffers are `StorageModeShared` MTLBuffers — host and GPU see the same memory — so
`read_host`/`write_host` are direct memory views and a cross-backend copy is a host round trip, not a
staged transfer. The command buffer is stored as a leaked box pointer because
`MpsCommandBuffer` is `!Send`/`!Sync`; all access happens sequentially through `&self`/`&mut self`,
and the struct is `unsafe impl Send/Sync` on that basis.

### 2.3 Weight residency and registry

- **Zero-copy parts.** `MpsState::register_part` wraps an mmap'd GGUF part with
  `newBufferWithBytesNoCopy` (`StorageModeShared`). The base must be page-aligned (16 KiB on Apple
  Silicon); mmap guarantees it, and the code `debug_assert!`s it.
- **Per-weight offsets.** `register_weight(name, data)` records `(part buffer, byte offset)`; the
  executor resolves any weight through `state.weight_buf(name) -> Option<(MetalBuffer, u64)>` and
  passes the offset to the kernel. `MINFER_WEIGHT_COPY=1` forces a copy per weight for A/B.
- **Load-time warm-up.** The first GPU access to file-backed (mmap) pages costs a one-time
  page/TLB setup (~44 ms measured); a dummy full-buffer read at model load moves that cost out of
  the first prefill (`METAL_OPTIMIZATIONS.md` §0 Done #39).
- **KV element type.** `set_kv_cache_type(n_layers, n_kv_embd)` runs once at load: f16 when
  `n_layers × n_kv_embd >= 8192` (the 7B class; measured ~−1 ms/token at 2K context) or f32 for
  small models (f16 measured ~3% slower there, dispatch-latency-bound); `MINFER_CACHE_TYPE=f16|f32`
  overrides. `kv_cache_is_f16()` is the query the store/attention/fused kernels use.

### 2.4 Command buffers and submission

One `MpsCommandBuffer` is kept for the current split:

- `cb()` creates it on the first op of a split (a leaked box, so the returned reference is `'static`
  and callers can still touch the pool).
- Every dispatch helper ends with `memoryBarrierWithScope(MTLBarrierScope::Buffers)`, so kernels in
  one command buffer see each other's writes.
- `synchronize()` submits it; the scheduler calls that at split boundaries and once at the end.
- `Drop` submits a pending buffer so an unterminated encoder can never be left behind.
- `MpsCommandBuffer::submit()` commits with a dispatch-semaphore completion handler (avoiding the
  ~20 ms scheduler wakeup of `waitUntilCompleted`), **waits with a bound** and checks the command
  buffer status. On a failure it returns `Err`; the backend `expect`s it, and device-configuration
  problems go through `gpu_abort` instead.

### 2.5 Legacy surface

The pre-graph whole-layer `layer_gpu` path was removed when the graph became the default: there is
no `layer_gpu` function in `metal.rs` (the name survives only in the legacy CUDA code), and
`src/graph/metal_backend.rs` is the only live consumer of the device layer. What remains is a
`#[allow(dead_code)]` block in `metal.rs` holding the old-forward scaffolding and a few methods kept
for tests (e.g. `matmul_on_gpu_buf`); the loaders, the graph backend and the kernel tests are the
live callers. (The legacy `KVCache` type in `src/cache.rs` is likewise unused by the graph path; KV
lives in the allocator's persistent regions.)

---

## 3. llama.cpp Metal Reference Map

`docs/LLAMA_METAL_E2E.md` documents llama.cpp's Metal path end to end. What minfer borrowed, and
what it deliberately does not do:

| llama.cpp Metal concept | minfer analog | Status |
|---|---|---|
| Backend interface + scheduler splits | Graph `Backend` trait + `split_graph`; per-node `execute_node` | Borrowed, reshaped |
| Multi-command-buffer scheme with status tracking | One command buffer per split; submit at boundaries | Simplified |
| Per-op encoder + explicit memory barriers | One compute encoder per split, `memoryBarrierWithScope(Buffers)` after every dispatch | Borrowed |
| `MUL_MAT` three-way kernel selection (matrix/tile/vector) | `quant_matmul_f32_on_gpu_buf` tiers: simdgroup GEMM, `_multi`, single-token | Borrowed in spirit, own kernels |
| `FLASH_ATTN_EXT` variant selection (DK/DV, f16 KV) | `gqa_attn_flash` (nt==1) and `attn_flash_prefill` (nt>1) with `hd ∈ {64,128}` guards | Borrowed in spirit, own port |
| Fusion rules in the Metal backend | minfer fuses at the graph level: `FusionPass` SwiGLU + build-time `FusedQKV`/`FusedFFN`/`FusedQkvNorm` | Diverged (graph-level fusion) |
| Unified-memory weight buffers (mmap or copies) | `newBufferWithBytesNoCopy` over mmap'd GGUF parts, per-weight (buffer, offset) | Borrowed |
| KV cache element-type policy | `set_kv_cache_type` auto f16 for the 7B class | Borrowed |
| MPS `MPSGraph` / higher-level MPS APIs | — | Not used (all kernels are hand-written MSL) |
| Multi-GPU / device selection, command-buffer concurrency tuning | — | Out of scope |

---

## 4. Design

### 4.1 `MetalBackend` lifecycle and state

`new()` returns `None` when MPS is unavailable or `MINFER_DISABLE_MPS` is set (both handled inside
`MpsState::try_new`); otherwise it stores the `'static` singleton reference and starts with an empty
pool and a null command-buffer pointer. `MpsState::init()` runs once at model load.

Pool rules:

- **Exact byte-length reuse only** — `alloc_buffer` matches `size * 4` against the free list, else
  allocates a `StorageModeShared` f32 MTLBuffer.
- **`free_buffer` never releases** — the id goes back to the free list so persistent KV regions
  survive rebuilds; the MTLBuffer stays alive for the process.
- **`alloc_fresh` always allocates** — split-boundary staging must not be recycled out from under
  in-flight node buffers.
- **No generation counter.** Unlike CUDA there is no captured-exec cache, so no pointer
  invalidation machinery is needed.

Capture staging (`staging` / `free_staging`) exists only for trace/viz: `staging_alloc` reuses an
exact-length staging buffer or allocates one, `capture_split(src_ids)` encodes blits at the **end**
of the split's command buffer (after all kernels, so the staging holds this step's output) and
returns ids valid only after the next submit, `read_staging(id)` reads them back, and
`release_staging_all()` returns them to the free list after that split's readback.

In-place aliasing is the allocator's decision (sole consumer + same backend). The executor calls
`copy_in(dst, src)` **only when `in_bufs[0] != out_buf`** — the non-aliased case — so an in-place
kernel runs directly on `out_buf`, which for an aliased node is the producer's buffer.

`MINFER_OP_PROFILE=1` accumulates host encode time per op label and per-submit GPU wait; the first
submit prints a top-20 table, later submits print one line each; zero overhead when unset. `Drop`
submits any pending command buffer (never leaving an unterminated encoder) and prints the profile.

### 4.2 Backend trait mapping

| Trait method | Metal implementation |
|---|---|
| `name()` | `"metal"` |
| `supports_op(op, dtype)` | §4.3 |
| `supports_fused(fused)` | `matches!(fused, FusedOp::SwiGLU)` — the only variant in the enum |
| `alloc_buffer` / `free_buffer` / `alloc_fresh` | §4.1 |
| `execute_node(node, in_bufs, out_buf, kv_pair)` | opens the split's command buffer once, then dispatches; guards return `Err` |
| `read_host(id)` / `write_host(id, data)` | direct `&[f32]` views over shared memory (no staging, no copy) |
| `synchronize()` | `submit_pending()` |
| `graph_replay(..)` | not overridden (the trait default returns `false`) — Metal has no capture/replay path |

Alongside the trait, `MetalBackend` exposes the capture helpers (`capture_split`, `read_staging`,
`release_staging_all`) that the scheduler's trace/viz path calls, and the module exposes
`metal_available()`.

### 4.3 Eligibility

`supports_op` is:

| Op | Metal |
|---|---|
| `Input` | yes (any dtype) |
| `Add`, `Mul`, `Silu`, `RmsNorm`, `QkNorm`, `SwiGLU` | F32 |
| `MatMul` | F32 activation (the weight type rides in `MatMulMeta.weight_ttype`) |
| `GetRows`, `RoPE`, `Attn`, `KvcacheStore`, `KvcacheLoad` | F32 |
| `FusedQKV`, `FusedQkvNorm`, `FusedFFN` | F32 |
| `View`, `Reshape`, `Permute` | yes (identity copy) |
| `Scale`, `Softmax`, `BatchMatMul` | no (vocabulary only) |
| `QkvBiasRopeStore` | no — the mixed-quant decode epilogue is CUDA-only; on macOS the builder never emits it |

Two layers of checks are deliberately elsewhere:

1. **Weight-type eligibility is a model-level all-or-nothing gate** (`weights_on_gpu`, §4.6): either
   every graph-referenced weight is registered on the GPU, or the model runs entirely on CPU.
2. **Shape invariants are enforced in `execute_node`** and return `Err`: attention requires
   `nkt == n_head_kv * hd` (the kernel strides KV by `nk*hd`) and `hd == hd_kv` (it uses the query
   head dim); the KV pair must exist; a missing weight is an error; the fast attention paths are
   limited to `hd ∈ {64, 128}` and fall back to the classic kernel otherwise.

One Metal/CUDA divergence worth stating: `Op::RoPE` carries the style into the kernel
(`rope_style` 0 = non-interleaved/Qwen2, 1 = interleaved/LLaMA), so Metal supports both styles;
CUDA's `supports_op` gates RoPE to `NonInterleaved` only. All supported models are non-interleaved.

### 4.4 Execution dispatch

`execute_node` opens the split's command buffer once, then dispatches:

| Op | Metal path |
|---|---|
| `Input` | no-op (host-filled) |
| `Silu` | `copy_in` if not aliased, then `silu_f32` in place |
| `Add` / `Mul` | `add_f32` / `mul_f32` |
| `RmsNorm` | weight from `NormMeta`; `rms_norm_256` when `rms_norm_256_enabled()`, else `rms_norm` (a `None` weight selects the weightless kernel) |
| `QkNorm` | same kernels with `d = hd`, `n = len/hd` over the flat `[nt*nh, hd]` rows |
| `MatMul` | `quant_matmul_f32_on_gpu_buf` (tier below) + optional `add_bias_f32` |
| `GetRows` + `Embed` meta | `embed_tokens_gpu` (per-weight-type row gather + dequant) |
| `GetRows` + no meta | `get_rows_f32` (the G3 tail-row selection) |
| `RoPE` | `copy_in` if not aliased, then `rope_f32` with the node's `rope_style` |
| `SwiGLU` | `swiglu_f32` |
| `KvcacheStore` | `kv_pair` required; two `store_kv` calls (K then V); f32 or f16 by `kv_cache_is_f16()` |
| `KvcacheLoad` | no-op — the output buffer *is* the persistent K region |
| `Attn` | §4.4.1 |
| `View` / `Reshape` / `Permute` | `copy_in` when the output differs, else no-op |
| `FusedQKV` | concat matmul (`blk.{i}.attn_qkv`) + `attn_bias_rope_store`; `debug_assert!(nt == 1)` |
| `FusedFFN` | concat matmul (`blk.{i}.ffn_gu`, `od = 2*nf`) + in-place `swiglu_f32_off`; `debug_assert!(nt == 1)` |
| `FusedQkvNorm` | concat matmul + two in-place per-head `rms_norm[_256]` (q at offset 0, k at byte offset `nqt*4`) + `attn_rope_store`; `debug_assert!(nt == 1)` |
| `Scale` / `Softmax` / `BatchMatMul` | `Err("op ... unsupported on Metal (Phase 3)")` |
| `QkvBiasRopeStore` | `Err("op ... unsupported on Metal (CUDA-only)")` — reaching it is a scheduling invariant violation |

**MatMul tiers** (`quant_matmul_f32_on_gpu_buf`): a simdgroup **GEMM** kernel (64×32 tile,
128 threads, 8 KiB threadgroup scratch) when
`nt >= 2 && (od >= 2048 || nt >= 9) && gemm_enabled()`; otherwise the **`_multi`** kernel for
`nt > 1` (one threadgroup per two output rows); otherwise the **single-token** kernel. Every
supported quant type has the tiers that matter. Guards: K-quant `id % 256 != 0` aborts via
`gpu_abort`; the GEMM checks the threadgroup-memory request against the device limit queried at
init. `MINFER_GEMM=0` disables the GEMM tier for A/B.

**Aliasing.** Only `Silu`, `RoPE` and the view ops call `copy_in(dst, src)`, and only when the
allocator did *not* alias them; an aliased node runs its in-place kernel directly on `out_buf`.

#### 4.4.1 Attention dispatch

Pre-dispatch guards return `Err`: `nkt == n_head_kv * hd` (the classic kernel strides KV by
`nk*hd`), `hd == hd_kv` (it uses the query head dim), and the layer's KV pair must exist.

- **Decode (`nt == 1`)**: `flash_attn_enabled(hd)` → `gqa_attn_flash` (chunked, partials merged by the
  shared combine kernel); else `hd ∈ {64,128}` and `MINFER_NO_SPLIT_ATTN != "1"` →
  `gqa_attn_split_f32` (two-pass KV-parallel); else the classic `gqa_attn_f32`.
- **Prefill (`nt > 1`) with `hd ∈ {64,128}`**: `prefill_flash_enabled(hd)` → `attn_flash_prefill`
  (the llama `flash_attn_ext_blk` port, with the tail-pad kernel for a partial last KV block); else
  `matmul_attn_enabled()` → `attn_parallel_prefill` (3-pass scores → masked softmax → output); else
  the classic `gqa_attn_f32`.
- **Other head dims** always take the classic kernel.

The decode chunk count is `MINFER_ATTN_CHUNKS` or `((max_pos + 1 + 31) / 32).clamp(1, 16)` — one
chunk per 32 KV rows, capped at 16. `nkv` for the prefill kernels and the chunk count come from a
host read of the positions buffer; that is safe because positions are host-written input data, never
GPU-computed. (CUDA instead derives the bound on device so nothing host-side enters a captured
graph; Metal has no replay to protect, so the host read is free.)

### 4.5 Allocator and scheduler integration

- Assignment priority is **Metal → CUDA → CPU**; `enable_metal()` mirrors `enable_cuda()`.
- KV regions are created by `ensure_kv` on the layer's assigned backend, so a Metal-assigned layer's
  K/V live in the Metal pool and `kv_pair` resolves to pool ids the executor passes to the kernels.
- Cross-backend values are staged by the allocator (`alloc_fresh` on the consumer's backend) and
  copied through host memory — with shared MTLBuffers that is a plain `copy_nonoverlapping`.
- The scheduler calls `sync_backend` (→ `MetalBackend::synchronize`) at every backend change and
  after the last split, which is exactly when the split's single command buffer is submitted.
- Because Metal supports the whole Qwen2/Qwen3 op set (including the embedding and tail gathers), a
  normal forward is a **single Metal split**; CPU splits appear only for an op Metal refuses
  (`Scale`/`Softmax`) or in synthetic/mixed graphs — and the tests deliberately exercise the
  multi-split alternation.

### 4.6 Model wiring

```
metal_on = metal_available() && weights_on_gpu(model)   // #[cfg(target_os = "macos")]
CParams.gpu = metal_on || cuda_on
```

- **`weights_on_gpu` is registration-only**: it builds the exact list of weight names the graph reads
  (`tok_embd`, `output_norm`, `output`, `output_b`, and per layer `attn_norm`, `wq`, [`bq` Qwen2],
  `wk`, [`bk`], `wv`, [`bv`], `wo`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, plus Qwen3's
  `q_norm`/`k_norm`) and requires `mps.has_weight(name)` for every one. It is all-or-nothing: one
  missing name and the model runs entirely on CPU. Unlike CUDA's `weights_on_cuda` there is **no
  type whitelist here and no diagnostic print** — a Metal gate failure is silent.
- **Type support is enforced at loader registration**: `load_ti` registers a tensor only when its
  type is Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0, or F32. An unsupported type is simply never
  registered, so the gate fails and the model falls back to CPU. Names are namespaced (`mps.register_part`
  per mmap'd part; `{ns}{tensor}` for the registry) so a second model cannot collide with the first.
- **Concat weights** are built once at load with `metal::concat_rows` and registered: Qwen2 and
  Qwen3 both register `blk.{i}.attn_qkv` (all three of wq/wk/wv present, same type/dim,
  block-aligned) and `blk.{i}.ffn_gu`, the latter only when `nf <= 16384 && MINFER_NO_FUSE_FFN != "1"`
  (the 7B concat would otherwise hold ~2 GiB of weights no node reads).
- **Fused nodes per model**: Qwen2 builds `FusedQKV` for decode QKV; Qwen3 builds `FusedQkvNorm`
  (per-head Q/K norm, which the Qwen2 bias+rope+store kernel cannot express); both build `FusedFFN`
  when `nf <= 16384`. The mixed-quant `QkvBiasRopeStore` class is CUDA-only, so those layers keep
  the unfused chain on macOS. Gates: `CParams.fuse_qkv` / `fuse_ffn` (Qwen3's `fuse_qkv` is
  `metal_on`-only), so no CUDA presence enables a Metal-incompatible fusion.
- **FusionPass** gets `[cpu, metal, maybe cuda]` and a `backend_of` that maps CPU→0, Metal→1 and
  CUDA→the index found by name; the SwiGLU rewrite is gated by `supports_fused(SwiGLU)` and applies
  to the unfused FFN path (when `FusedFFN` is built, silu+mul are inside the fused kernel).

### 4.7 Memory, residency and staging

- **Unified memory changes the copy math.** Activations are `StorageModeShared` MTLBuffers, so
  host reads/writes are direct views and `copy_across` is a host round trip; there is no pinned
  staging ring and no `pool_gen`.
- **Weights are zero-copy** over the mmap'd GGUF parts, with the ~44 ms first-touch page cost paid
  at load by a dummy warm-up read (`METAL_OPTIMIZATIONS.md` §0 Done #39). `MINFER_WEIGHT_COPY=1` forces
  per-weight copies.
- **KV element type.** The persistent regions stay f32-shaped in the IR, but `set_kv_cache_type`
  picks f16 for the 7B class (KV-bandwidth-bound; measured ~−1 ms/token at 2K) and f32 for small
  models (f16 measured ~3% slower there); `MINFER_CACHE_TYPE` overrides.
- **Capture staging** exists only while trace/live capture is armed; per-split blits write node
  outputs into host-readable staging at the end of the command buffer, read back after submit, then
  released.
- **Device limits are queried once** at init (`maxThreadgroupMemoryLength`) and referenced by
  dispatch-time guards; the remaining hardcoded numbers are kernel-declared array sizes, documented
  in `GPU_SAFETY.md` §3.
- **Profiling.** `MINFER_OP_PROFILE=1` reports host-encode time per op and per-submit GPU wait, which
  is how the decode dispatch-cost story in `METAL_OPTIMIZATIONS.md` was measured.

### 4.8 Command buffers, submission and trace capture

One `MpsCommandBuffer` per split, submitted at boundaries, is the whole execution model:

- Every dispatch helper ends with `memoryBarrierWithScope(MTLBarrierScope::Buffers)`. Metal does not
  guarantee write visibility between dispatches in one compute encoder; the missing barrier caused
  intermittent last-2-token corruption on 1.5B/7B prefill before the 2026-08-19 fix.
- `submit()` commits with a dispatch-semaphore completion handler (avoiding the ~20 ms scheduler
  wakeup of `waitUntilCompleted`), waits with a **10 s bound**, and requires
  `MTLCommandBufferStatus::Completed`; otherwise it returns `Err` carrying the recent dispatch trace.
  The backend `expect`s the submit result; device-configuration problems go through `gpu_abort`
  (print the actual limits and exit) rather than degrading silently.
- **Trace/viz**: the scheduler pushes each non-KV Metal node's output buffer, and at the split end
  `capture_split` encodes the blits; after `sync_backend` submits, `flush_metal_captures` reads every
  staging buffer and records its stats. KV regions are skipped (a full region per layer would
  dominate the trace).
- **No graph replay.** Metal has no CUDA-Graph equivalent here; `graph_replay` is the trait default.
  `MINFER_METAL_CAPTURE=1` instead starts an `MTLCaptureManager` GPU trace at init for Xcode, and
  `MINFER_TRACE=1` records a 16-deep per-dispatch label ring used to diagnose a GPU fault
  (note: `MINFER_TRACE` also names the graph trace path in the CLI, which is a separate mechanism).

### 4.9 GPU safety (Metal edition)

`docs/GPU_SAFETY.md` holds the rules and the incident history; the backend implements them as:

1. **Bounded submit with a status check** — never `DISPATCH_TIME_FOREVER`, never a silent
   non-`Completed` status.
2. **No early return past a `threadgroup_barrier`** — the attention kernels were rewritten to run a
   dummy head instead of returning early, then skip the output write via a `valid_head` flag.
3. **Device limits queried at runtime** — threadgroup memory and thread limits come from the device
   at init; guards compare against the queried values.
4. **Barriers between dispatches** — a buffer written by one dispatch and read by the next needs the
   explicit scope barrier; a reused threadgroup-memory buffer needs a `threadgroup_barrier` between
   the last read and the first write.
5. **`Err` from `execute_node`, never a CPU fallback** — missing weight, bad shapes, missing KV
   regions, unsupported op.
6. **`gpu_abort` for configurations the GPU path cannot run** — dimension misalignment, device-limit
   overruns, kernel-array overflow: print the actual values and exit.
7. **Recurrence playbook** — reproduce with one app and a bounded `-n`; bisect with `MINFER_GEMM=0`
   and `MINFER_CACHE_TYPE=f32`; on a freeze, `spindump` over SSH and check the diagnostic reports.

Accepted and documented (not fixed): the audit's L1/L2 landmines, and the fact that a kernel-level
fault is not always attributable to a single dispatch (hence the `MINFER_TRACE` ring).

## 5. Implementation Phases

### 5.1 Graph-backend phases

Metal arrived as Phase 3 of the compute-graph rewrite and was then wired to parity with the
pre-graph path through the G-series:

| Phase | Content | Status |
|---|---|---|
| 3 | `MetalBackend` per-op adapter (`metal_backend.rs`) + cross-backend scheduling | ✅ |
| 4–6 | scheduler assign/split/execute, `FusionPass`, DOT/cache; Qwen2 graph build; imperative `forward.rs` deleted | ✅ |
| G1 | `Attn` dispatch mirrors the old path (flash / split / parallel / classic) with the same gates | ✅ `4e105ce` |
| G2 | `RmsNorm` selects `rms_norm_256` when enabled (~2× faster per dispatch) | ✅ `4e105ce` |
| G3 | `n_out` tail-row `GetRows` + two allocator liveness fixes | ✅ `d81af71` (docs `8d7cb38`) |
| G4 | Decode `Op::FusedQKV` (concat matmul + `attn_bias_rope_store`) | ✅ `bd28047` (docs `96404fb`) |
| G5 | Decode `Op::FusedFFN` (concat gate+up + in-place swiglu), gated `nf <= 16384` | ✅ `1dee1b5` (docs `ec922f1`) |
| G6 | Qwen3 `Op::QkNorm` + `Op::FusedQkvNorm` (per-head norm + no-bias rope/store) | ✅ `283c7d6`, `94d57ac`, `d5b8023` |
| objc2 migration | `metal` 0.28 / `block` + `vendor/block` patch → `objc2-metal` / `block2` / `objc2-foundation`, Phases 0–6 | ✅ `9e238bd`, `6a382a3`, `be3df55`, `ee9b65b` |

The graph-era details live in `docs/COMPUTE-GRAPH-DESIGN.md` §17 (Phases 1–11, deviations 18–26) and
`METAL_OPTIMIZATIONS.md` §0.1 / §4.3.

### 5.2 Optimization and cold-start record

> **Hash caveat.** The commit hashes printed in `docs/METAL_OPTIMIZATIONS.md` predate a repository
> history rewrite and no longer resolve. The hashes below are subject-matched equivalents from the
> current history; each was verified with `git cat-file -t`. Cite these, and the
> `METAL_OPTIMIZATIONS.md` section, together.

| Workstream | What it changed | Doc reference | Representative commits |
|---|---|---|---|
| Initial Metal port | Device/queue/encoder skeleton, the full kernel set, RMSNorm/GQA simdgroup parallelism, SwiGLU fusion | §3.1, §5.1 | `ac2cb3d`, `e7df395`, `6811da3`, `2a76d0c`, `b0819e7`, `2f484f8` |
| Correctness foundation | RoPE `freq_scale`, `output_b`, softmax max, dynamic `hd`, Q5_K formula, GQA `simd_max` partial-tile divergence, first isolation suites | §3.1 (#1–#5) | `df2da9f`, `c34b7d8`, `87fec18` |
| GPU-safety hardening | Bounded `submit()` + status check, dispatch-label trace ring, no early return past a barrier, runtime guards, autorelease retain fix | §3.1 (#6–#7), `GPU_SAFETY.md` §1–§4a | `ef6cde7`, `5f5a42d` |
| Decode fusions + split attention (old path) | Fused QKV/FFN decode matmuls, 2-pass KV-parallel split attention, float4 acc, adaptive chunks, KV geometric growth | §3.3 | `e5db3aa`, `39c2ba9`, `eb0f812`, `e031282` |
| GEMM (simdgroup) work | llama `kernel_mul_mm` port (64×32 tile, 4 simdgroups), per-quant GEMMs, hot-loop unroll, ik-loop `simdgroup_barrier`, the partial-tile race + `memoryBarrier` fix | §3.4 (#11/#12/#28/#29/#30) | `c34b7d8`, `d83bd25`, `5202548`, `e7be3d3`, `69a47d4`, `f52628c`, `e997b99` |
| Decode matmul layout ports | q6_K stride-2/float4 (72→209 GB/s), q4_K stride-4/`sc16` (7B decode ~51→~19.3 ms/token) | §3.3 (#27) | `36c9b01`, `b59c8c8` |
| Flash-attention ports | Decode `flash_attn_ext_vec` hd=64/hd=128, prefill `flash_attn_blk` hd=64/hd=128, tail-pad kernel | §3.3/§3.4 (#22/#24–#26), §5.5 | `c1177f7`, `b56b5dd`, `ac5e1ea`, `a78620c` |
| Parallel prefill attention + RMSNorm-256 | 3-pass barrier-free prefill attention; 256-thread RMSNorm; chunk cap 16; drop the KV→CPU sync | §3.4/§3.5 (#16/#17) | `35a9659`, `89aa1fa` |
| KV f16 | `store_kv_f16` + `_f16` attention kernels; auto-select f16 for the 7B class | §3.5 (#13/#37) | `0aad968`, `ff60ed7` |
| `n_out` tail-row reduction | Final norm + lm_head on output rows; last-layer FFN + both residuals on the tail rows | §3.7 (#32/#34) | `59a97f9`, `660f59e` |
| Cold-start axis | Embedded precompiled metallib, GGUF mmap + zero-copy weights, load-time warm-up read | §4.2 | `b7cb5bc`, `1032ca9`, `a5709e7` |
| Embedding coverage + GEMM threshold | GPU `get_rows` for all 8 quant types; adaptive GEMM dispatch `nt≥2 && (od≥2048 || nt≥9)` | §0 Done #33/#38/#40 | `d6058b1`, `8dfdfd8`, `8506a6d` |
| Prefill-GEMM investigation | Grid-shape probe, exact-shape replay, 7B decomposition, structural-equivalence audit, gap acceptance | §3.6 (§4.3.1–§4.3.10) | `6205f7e`, `9a201b5`, `c900127`, `89e2468` |
| Compute-graph IR + allocator | Phases 1–6 of the rewrite (IR/builder/allocator, CPU backend, Metal backend, scheduler/fusion/cache, Qwen2 graph build) | `COMPUTE-GRAPH-DESIGN.md` §17.1 | `a163a07`, `308cc74`, `be35b1b`, `8091b61`, `941d34f`, `e54070c` |

Numbers like `#27` are local to a `METAL_OPTIMIZATIONS.md` §0 table — always qualify them, because
the Done / To-do / Decided tables reuse the same numbers.

---

## 6. Risks and Open Questions

| # | Risk / question | Status |
|---|---|---|
| 1 | Cross-dispatch write visibility (2026-08-19 incident) | **Fixed**: scope barrier after every dispatch; rule recorded in `GPU_SAFETY.md` |
| 2 | Early return past a `threadgroup_barrier` in attention | **Fixed**: no early returns; invalid heads run the loop and skip the store |
| 3 | Device-limit guesses | **Fixed**: limits queried at init, never hardcoded |
| 4 | Concurrent MPS access under parallel tests flips fused/unfused greedy tokens | **Mitigated for tests**: `metal_test_lock()` + `MpsState::init()`; production runs one scheduler thread |
| 5 | Q8_0 multi-token matmul race (missing trailing `threadgroup_barrier`) | **Fixed**; pinned by `metal_prefill_determinism` |
| 6 | Metal gate failure is silent (no diagnostic like CUDA's `CUDA GATE:`) | **Open (diagnostics)**: a missing/unregistered weight drops the model to CPU with only the init line to explain it |
| 7 | Whole-layer `layer_gpu` reference path still referenced in comments/tests | **Open (cleanup)**: the function is gone from `metal.rs`; the `#[allow(dead_code)]` block and comments remain |
| 8 | 7B prefill ~10% behind the old path | **Accepted**: GEMM-bound; the GEMM transfers fully, and the prefill-GEMM investigation closed as "decided not to change" (`METAL_OPTIMIZATIONS.md` §3.6) |
| 9 | Warm-up / cold-start costs | **Tracked**: mmap first-touch solved by the load-time warm-up; remaining cold-start to-dos in `METAL_OPTIMIZATIONS.md` §4.2 |

---

## 7. Verification

### 7.1 Device test suite

`src/graph/metal_backend.rs` carries 14 graph-backend tests, all serialized by `metal_test_lock()`
and skipped with `MPS unavailable; skipping` when Metal is absent:

- **Elementwise / norm**: `metal_elementwise_matches_cpu` (silu+add, bit-for-bit),
  `metal_rmsnorm_matches_cpu`, `metal_rmsnorm_real_scale` (d=896, nt=8).
- **Cross-backend**: `metal_cross_backend_copy`, `metal_cross_backend_copy_large`,
  `metal_embed_then_rmsnorm_cross_backend`, `metal_multi_split_alternation`.
- **Matmul**: `metal_matmul_q8_matches_cpu` (real `wk` shape), `metal_matmul_q4_matches_reference`
  (layer-0 `wq`, [896, 896], vs a manual Q4_0×f32 reference).
- **KV + attention**: `metal_attn_kv_matches_cpu` (bit-exact), `metal_attn_kv_real_scale`
  (nh=14, nk=2, hd=64, nkt=128, nt=30), `metal_attn_decode_step` (nt=1 with 30 stored rows),
  `metal_store_after_gpu_op` (KV store whose K comes from a GPU op), `metal_store_real_dims`
  (n_ctx=32768).

Model-level Metal tests live with the models and skip on non-macOS:
`graph_metal_matches_cpu_logits`, `graph_metal_layer0_isolation`, `graph_metal_real_wk_matmul`
(Qwen2), `fused_qkv_matches_unfused_decode` (Qwen2, with the `metal_test_lock` rationale recorded),
`fused_qkv_norm_matches_unfused_decode` (Qwen3), `graph_metal_matches_llama_reference` (Qwen3; pins
the first 9 tokens of a 60-token byte-identical llama-Metal run) and `metal_prefill_determinism`
(Qwen3; the Q8_0 multi-token matmul race regression).

### 7.2 Kernel isolation suites

`tests/` carries four **macOS-only** integration binaries (`#![cfg(target_os = "macos")]`, so they do
not exist on Linux). Each builds its own deterministic inputs and embeds a scalar CPU reference —
none needs an external dump:

| File | Tests | Coverage |
|---|---|---|
| `tests/flash_attn_isolation.rs` | `flash_attn_ext_isolation` (hd 64 + hd 128), `flash_attn_matches_split` | decode flash vs a scalar online-softmax reference and vs the split path; partial/empty KV chunks, nt 1–2, nkv up to 4097 |
| `tests/flash_attn_blk_isolation.rs` | `flash_attn_blk_isolation` | prefill blk port (hd 64/128, NSG=4), partial last KV block via the tail-pad kernel, nt up to 200, GQA heads, f32 + f16 KV, vs classic |
| `tests/gqa_attn_isolation.rs` | `gqa_attn_isolation`, `gqa_attn_split_isolation`, `gqa_attn_split_timing` | classic + split attention vs a scalar reference, including the `nkv % 32 != 0` divergent-simd_max case |
| `tests/gemm_isolation.rs` | `gemm_isolation`, `qkv_row_concat_layout`, `non_q4_0_gemm_isolation`, `get_rows_q4_k_isolation`, `get_rows_multi_type_isolation` | GEMM determinism + correctness vs scalar, concat layout, per-type row-gather |

### 7.3 Acceptance gates

- **Bit-exactness where the math is identical**: elementwise and KV/attention round trips are
  checked bit-for-bit; matmul and norm are checked within float tolerance.
- **CPU-vs-Metal logits** are compared with tolerance plus `greedy_ref == greedy_gpu`, because the
  Metal path uses f32 activations while the CPU reference quantizes to Q8_0 (the ~18-magnitude
  difference is expected and documented in the test).
- **Fused-vs-unfused decode** must be bit-identical, with the unfused side running the FusionPass.
- **llama-reference oracle**: Qwen3's first 9 greedy tokens are pinned against llama-Metal.
- **Determinism**: `metal_prefill_determinism` and the isolation suites' repeat-run checks.

Suite baselines: `METAL_OPTIMIZATIONS.md` / `KNOWN-CPU-ISSUES-2026-08-29.md` record the
single-threaded and parallel runs (e.g. 152 passed / 3 ignored single-threaded main-bin, plus the
integration binaries green on device). Run `cargo test --release` on macOS for the device suites;
on Linux the Metal-only tests self-skip and the isolation binaries are empty.

---

## 8. Out of Scope / Future

- **Not planned**: MPSGraph / higher-level MPS APIs, multi-GPU, f16 activations, training.
- **Closed by measurement** (`METAL_OPTIMIZATIONS.md` §3.6/§4): the prefill GEMM gap (params match
  llama's; decided not to change), and the flash/split/parallel attention lineup.
- **Remaining research** (`METAL_OPTIMIZATIONS.md` §4.1/§4.2): cold-start items and the residual
  7B-prefill gap; see that document's roadmap section rather than duplicating it here.
- **Cleanup candidates**: the `#[allow(dead_code)]` legacy block/comments in `metal.rs`, and adding a
  Metal gate diagnostic to match CUDA's `CUDA GATE:` line (§6 #6/#7).

