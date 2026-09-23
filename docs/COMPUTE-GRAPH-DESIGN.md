# minfer Compute Graph Design

How minfer turns one forward pass into a declarative graph and executes it across CPU, Metal and
CUDA. This is the authoritative design and implementation record for `src/graph/`: the IR, the
builder, the allocator, the scheduler, the fusion rules, graph reuse, and the per-backend execution
mapping.

> **Status.** Landed. Every mechanism described here is implemented in the tree; the phase ledger and
> the deviations from the original plan are in [§17](#17-implementation-record).
> Baseline: `HEAD = 5471680` (2026-09-13); the working tree additionally carries the post-baseline
> fixes listed at the end of §17.3.
>
> **Provenance.** This file was `docs/GRAPH-REFACTOR-PLAN.md`, written before the rewrite as a plan.
> The section skeleton is preserved; the body has been rewritten in the present tense against the
> current code, and the plan-time "current state" descriptions have been replaced by the landed
> implementation. One historical caveat: the commit hashes in the plan-era phase table no longer
> resolve (the repository history was rewritten); the resolvable equivalents are noted in §17.
>
> **Related documents.** The end-to-end walkthrough
> (`docs/inference_e2e_walkthrough/05-graph-builder-ir.md` … `08-scheduler-execute.md`) narrates the
> same machinery line by line for a first-time reader; this document is the design of record and does
> not repeat that narrative. Backend-specific optimization history lives in
> `docs/CUDA-BACKEND-DESIGN.md`, `docs/CUDA_OPTIMIZATION.md`, `docs/cuda_optimization_steps/` and
> `docs/METAL_OPTIMIZATIONS.md`.

---

## 1. Design Goals

### 1.1 The problem

The pre-rewrite engine computed a forward pass imperatively: a hand-written `forward()` walked the
layers, dispatched each operation to a GPU "whole layer" fast path or a CPU fallback, and allocated
scratch buffers per step. That shape had four structural costs:

- **No reuse.** Topology was recomputed every decode step, together with the CPU scratch.
- **No explicit fusion.** Fusion existed (Metal `swiglu_f32`, `attn_bias_rope_store`, batched QKV
  matmul) but was hard-wired and invisible to the rest of the code.
- **Backend choice was runtime and per layer.** `layer_gpu()` decided inside the loop, and a support
  limitation could silently change the execution path mid-run.
- **New architectures meant rewriting the loop** (~620 lines of `qwen2/forward.rs`).

### 1.2 Goals and outcome

| Goal | Landed outcome | Evidence |
|---|---|---|
| Multi-backend dispatch decided before execution | Per-node assignment at build time via `GraphAllocator::supports` and each backend's `supports_op`/`supports_fused`; the scheduler partitions the assigned graph into splits | §3.4, §3.5 |
| Fusion as a first-class IR citizen | `FusionPass` rewrites `Mul(Silu(x), y)` → `SwiGLU`; decode additionally *builds* `FusedQKV`, `QkvBiasRopeStore`, `FusedFFN`, `FusedQkvNorm` as single nodes | §5 |
| Graph reuse with no rebuild | `GraphCache` compares `GraphParams` only; a decode step reuses graph + allocator + KV regions and refreshes input buffers | §6 |
| Extensible architectures | `ModelDef::build_graph` per architecture; Qwen2 and Qwen3 each own a `graph.rs`; the imperative `forward.rs` was deleted | §10, §13 |
| No silent fallback | `execute_node` returns `Result<(), String>`; kernel-invariant violations abort with the actual values, matching `docs/GPU_SAFETY.md` | §3.5, §7.4 |
| CUDA as a real backend, not a stub | `CudaBackend` wraps the existing `cuda.rs` device layer and preserves CUDA Graph capture/replay keyed by graph `uid` | §9 |

### 1.3 Non-goals

- **Multi-sequence batching on CPU.** The IR, the allocator and the attention kernels are
  sequence-aware (E1/E1b/E2: `seq_ids`, `attn_span`, per-sequence KV reservations, `Batch`), and the
  server composes a batch — but the default follows the **device** (E6: batches iff the model runs on
  CUDA, off on CPU/Metal; `MINFER_BATCH=0/1` forces either way), because batching measures **0.49x**
  the serial path on CPU while it is **1.9x** on the GB10 (E2's acceptance is device-dependent; see
  `ARCHITECTURE-EXECUTION-PLAN.md` §7 and the E6 record).
- **A generic ggml operator set.** `Scale`, `Softmax`, `View`, `Reshape`, `Permute`, `AttnMode::Mha`
  and `FusedOp::BatchMatMul` are present in the vocabulary but no supported architecture emits
  them; they are kept for parity and future use. `Op::FusedBiasRope` and its fusion rule were
  *removed* outright (the capability was never claimed — see §5.2).
- **Cross-vendor graph transpilation.** Each backend implements its own `execute_node`; there is no
  lowering pass.

### 1.4 Core invariants

These are the rules the rest of the document elaborates. Breaking one is a bug, not a tuning choice.

1. **KV positions are data, not structure.** `KvcacheStore`/`KvcacheLoad` carry only the layer index;
   the write row arrives through the `cells` input node (C6: `positions` is the token's index *within
   its sequence* and drives RoPE and the causal bound, while the allocator resolves `cells` — the
   arena row — from the run table; the two coincide only while a run starts at cell 0). The topology
   never depends on `n_past` — this is the precondition for decode reuse (llama.cpp `allow_reuse`
   behaves the same).
2. **Topology is a deterministic function of `GraphParams`.** Equal params ⇒ identical node sequence
   ⇒ the graph may be reused without rebuilding. The debug build asserts this structurally.
3. **Weights are named, not copied.** A node references its weight by name (`MatMulMeta.weight_name`);
   the backend resolves the name in its registry at execution time.
4. **The allocator is the single owner of buffers.** Backends own pools; the scheduler never allocates.
5. **Execution follows build order.** The builder appends sources before consumers, so node-id order
   is a valid topological order; the allocator's liveness uses the same order.
6. **Fusion never duplicates an existing fused kernel.** A rewrite is applied only when the target
   backend reports `supports_fused`.
7. **Errors are errors.** A backend that cannot execute a node returns `Err`; it never falls back to
   CPU mid-run.

---

## 2. Overall Architecture

### 2.1 Pipeline

```
 model.build_graph(&GraphParams)                models/<arch>/graph.rs
        │  ComputeGraph (pure IR, no execution)
        ▼
 scheduler.assign_backends(&graph, &alloc)      capability-driven, per node
        │
        ▼
 FusionPass::run(&graph, backends, backend_of)  SwiGLU rewrite (see §5)
        │
        ▼
 alloc.alloc_graph(&graph)                      liveness + persistent KV regions
        │
        ▼
 scheduler.execute(&graph, &alloc)              split → sync → copy → execute
        │
        ├── CpuBackend    (kernel.rs / vec_ops.rs)
        ├── MetalBackend  (metal.rs + metal.metal)
        └── CudaBackend   (cuda.rs + cuda_kernels.cu)
```

`GraphCache` owns the graph, the allocator and the last `GraphParams`; a decode step that reuses the
cached graph skips the first three stages and only refills inputs.

### 2.2 Module map

| Module | Role |
|---|---|
| `graph/mod.rs` | `ComputeGraph`, `CNode`, `DType`, `Backend`, `BufRef`, `PersistentBuf`, `topo_order` |
| `graph/ops.rs` | `Op`, `NodeMeta` and the per-op metadata structs, `AttnMode`, `FusedOp` |
| `graph/builder.rs` | `GraphBuilder` — the declarative construction API |
| `graph/params.rs` | `GraphType`, `CParams`, `GraphParams` (the reuse identity) |
| `graph/cache.rs` | `GraphCache` — params-only reuse, graph `uid`, allocator lifetime |
| `graph/backend.rs` | `Backend` trait, `KvProvider` |
| `graph/alloc.rs` | `GraphAllocator` — liveness, node→buffer map, KV regions, cross-backend staging |
| `graph/scheduler.rs` | `BackendScheduler` — `assign_backends`, `split_graph`, `execute` |
| `graph/fusion.rs` | `FusionPass` — pattern-matching rewrite |
| `graph/cpu_backend.rs` | CPU executor over `kernel.rs` / `vec_ops.rs` |
| `graph/metal_backend.rs` | Metal executor over `metal.rs` per-op methods |
| `graph/cuda_backend.rs` | CUDA executor over `cuda.rs`, including CUDA Graph capture/replay |
| `graph/dot.rs` | Graphviz DOT export |
| `graph/json.rs` | JSON export for the interactive visualizer (`viz/`) |

### 2.3 Scheduler policies

The original plan stated three policies; all three are landed, with two nuances.

1. **Attention and its KV regions share a backend.** A layer's two KV regions are created on the
   backend that first uses the layer (`GraphAllocator::ensure_kv`), and the layer's attention node
   resolves the same pair through `kv_pair(layer)`. Keeping them together avoids a per-step O(n_kv)
   host round trip.
2. **A GPU split shares one submission.** Metal accumulates kernels into one command buffer for the
   split and submits at the split boundary (`sync_backend`). CUDA can go further: a decode split may
   be replayed as a single captured CUDA Graph launch (§9.4).
   *Nuance:* the whole-layer `layer_gpu()` fast path the plan proposed keeping was **not** kept.
   Per-op execution through `MetalBackend`/`CudaBackend` is the only inference path; the legacy
   `cuda.rs::layer_gpu` survives as dead code, and the plan's "whole-layer fast path" fallback is
   unnecessary because assignment is decided at build time.
3. **Fusion is capability-driven, not forced.** `FusionPass` consults `supports_fused` per node, and
   the decode fusions are gated in `CParams` so they can be A/B-tested.
   *Nuance:* only the SwiGLU rewrite is actually accepted by a backend today; see §5.2.

### 2.4 Execution order and the correctness contract

`BackendScheduler::execute` walks the graph split by split, and within a split node id by node id.
This is the same contract ggml uses (`nodes[0..n_nodes]`): the builder guarantees that a producer
precedes its consumers, so a KV store runs before the attention that reads the KV view.

The allocator's liveness analysis deliberately uses the **same** order rather than the Kahn order
returned by `topo_order()`. `topo_order()` may move source-less nodes (e.g. `kv_load`) ahead of nodes
built before them; liveness computed on that order can consider an input dead while the scheduler has
not read it yet, and reuse its buffer — the G3 regression, recorded as deviation 22 in §17.

---

## 3. Core Data Structures

### 3.1 The IR (`graph/mod.rs`, `graph/ops.rs`)

```rust
pub type NodeId = usize;

pub enum DType { F32, F16, I32, Q8_0 }   // activations are F32; F16/I32/Q8_0 for inputs

pub enum Backend { CPU, Metal, Cuda }

pub struct BufRef { pub backend: Backend, pub id: usize }   // id inside that backend's pool
pub struct PersistentBuf { pub name: String, pub backend: Backend, pub id: usize }

pub struct CNode {
    pub id: NodeId,
    pub name: String,
    pub op: Op,
    pub src: Vec<NodeId>,
    pub out_shape: [usize; 4],
    pub out_dtype: DType,
    pub backend: Option<Backend>,     // decided by the scheduler
    pub meta: NodeMeta,
}

pub struct ComputeGraph {
    pub nodes: Vec<CNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub uid: u64,                     // CUDA Graph cache key; assigned by GraphCache
}
```

Shapes follow the llama.cpp convention: activations are feature-major `[d, nt, 1, 1]` (feature
fastest, tokens in dim 1), and a weight tensor carries GGUF metadata `[in, out]` while its memory is
`[out][in]` row-major. `MatMulMeta` therefore records `in_dim = shape[0]`, `out_dim = shape[1]`.

`DType::size()` is defined for all four variants, but only F32 activations and I32 inputs are
constructed by the supported models; F16 KV storage is a backend-internal concern, not an IR dtype.

#### The `Op` vocabulary

```rust
pub enum Op {
    Input,                                     // leaf, host-filled every step
    Add, Mul, Scale(f32), Silu,                // element-wise
    Softmax { dim: usize },                    // reduction (vocabulary only)
    RmsNorm { eps: f32 },                      // normalization
    QkNorm { hd: usize, nh: usize, eps: f32 }, // per-head RMSNorm (Qwen3 q/k norm)
    MatMul { transpose_b: bool },              // linear algebra
    GetRows,                                   // embedding lookup / tail-row selection
    RoPE { style: RopeStyle },                 // positional encoding
    Attn { mode: AttnMode, explicit_span: bool },  // attention (softmax fused inside the kernel);
                                               //   `explicit_span` = `positions` cannot bound the
                                               //   node (several sequences, or a window that does
                                               //   not start at cell 0), so only a backend that
                                               //   reads the explicit span may take it
    KvcacheStore { layer: usize },             // persistent KV write; the row comes from `cells`
    KvcacheLoad  { layer: usize },             // view of the persistent KV region
    View { offset: usize, shape: [usize; 4] },
    Reshape { shape: [usize; 4] },
    Permute { dims: [usize; 4] },
    SwiGLU,                                    // FusionPass output
    BatchMatMul,                               // planned, not emitted (single-output IR)
    FusedQKV { layer: usize },                 // decode: concat matmul + bias/rope/store
    QkvBiasRopeStore { layer: usize },         // decode mixed-quant: 3 matmuls + one epilogue
    FusedFFN,                                  // decode: gate|up concat matmul + in-place swiglu
    FusedQkvNorm { layer: usize },             // decode Qwen3: concat matmul + qk_norm + rope/store
}
```

`Op` derives a **full** `PartialEq` (payloads included). The production reuse decision does not use
it — it compares `GraphParams` only — but the debug structural check compares op payloads, shapes and
dependencies, so a payload that silently changed is caught.

The variants marked "vocabulary only" (`Scale`, `Softmax`, `View`, `Reshape`, `Permute`,
`BatchMatMul`, `AttnMode::Mha`) carry `#[allow(dead_code)]` and no supported architecture emits
them (§5.5). The KV rule from invariant 1 is visible directly in the payloads:
`KvcacheStore`/`KvcacheLoad` carry the layer index and nothing else.

#### Node metadata

`meta` uses a concrete enum rather than the plan's `Box<dyn Any + Send + Sync>` — it is `PartialEq`
(needed by the structural check), cannot panic on downcast, and keeps `CNode` `Clone`:

```rust
pub enum NodeMeta {
    None,
    MatMul(MatMulMeta), Norm(NormMeta), Rope(RoPEMeta), Attn(AttnMeta),
    Kvcache(KvcacheMeta), Embed(EmbedMeta),
    FusedQkv(FusedQkvMeta), QkvBiasRopeStore(QkvBiasRopeStoreMeta),
    FusedFfn(FusedFfnMeta), FusedQkvNorm(FusedQkvNormMeta),
}
```

| Meta | Carries | Consumed by |
|---|---|---|
| `MatMulMeta` | `weight_name`, `bias_name`, `weight_ttype`, `in_dim`, `out_dim` | backends pick the kernel by `weight_ttype` without holding the `Tensor` |
| `NormMeta` | optional weight/bias names (shared by `RmsNorm` and `QkNorm`) | CPU / Metal / CUDA |
| `RoPEMeta` | `freq_base`, `freq_scale`, `n_head`, `hd` | rope kernel |
| `AttnMeta` | `layer`, `n_head`, `n_head_kv`, `hd`, `hd_kv`, `nkt` (KV row stride), `scale` | attention; `layer` resolves `kv_pair`. The allowed cells are **data** (`attn_span`, E1), not a field |
| `KvcacheMeta` | `n_embd`, `n_head_kv` | KV region sizing / attention strides |
| `EmbedMeta` | `vocab_size`, `weight_name`, `weight_ttype` | embedding lookup |
| `FusedQkvMeta` | concat weight name, three bias names, `in_dim`, `nqt`, `nkt`, `hd`, `nh`, `nk`, rope params, `kv_elems` | `FusedQKV` |
| `QkvBiasRopeStoreMeta` | three bias names, `nqt`, `nkt`, `hd`, rope params, `kv_elems` | `QkvBiasRopeStore` |
| `FusedFfnMeta` | `gu_weight`, `weight_ttype`, `in_dim`, `nf` | `FusedFFN` |
| `FusedQkvNormMeta` | concat weight, `q_norm_name`, `k_norm_name`, dims, rope params, `kv_elems`, `eps` | `FusedQkvNorm` |

`ComputeGraph::topo_order()` is a Kahn sort used for validation; `capture_nt_hint()` (CUDA builds)
returns the first `MatMul` node's output row count for the capture gate.

### 3.2 Graph builder (`graph/builder.rs`)

`GraphBuilder` is an append-only factory: it assigns ids, records shapes/dtypes/meta and wires `src`
edges. It never computes and never allocates.

```rust
impl GraphBuilder {
    pub fn new() -> Self;
    pub fn node(&mut self, name: &str, op: Op, src: &[NodeId],
                out_shape: [usize; 4], out_dtype: DType, meta: NodeMeta) -> NodeId;

    pub fn input(&mut self, name: &str, shape: [usize; 4], dtype: DType) -> NodeId;

    // shape-aware convenience constructors
    pub fn embedding(&mut self, ids: NodeId, weight: &Tensor) -> NodeId;
    pub fn rms_norm(&mut self, x: NodeId, weight: Option<&Tensor>, eps: f32) -> NodeId;
    pub fn qk_norm(&mut self, x: NodeId, weight: Option<&Tensor>,
                   hd: usize, nh: usize, eps: f32) -> NodeId;
    pub fn matmul(&mut self, x: NodeId, w: &Tensor, bias: Option<&Tensor>) -> NodeId;
    pub fn matmul_by_name(&mut self, x: NodeId, weight_name: &str, ttype: TensorType,
                          out_dim: usize, in_dim: usize) -> NodeId;
    pub fn get_rows(&mut self, x: NodeId, ids: NodeId, out_shape: [usize; 4]) -> NodeId;
    pub fn rope(&mut self, x: NodeId, pos: NodeId, style: RopeStyle, meta: RoPEMeta) -> NodeId;
    pub fn silu(&mut self, x: NodeId) -> NodeId;
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId;
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId;
    pub fn softmax(&mut self, x: NodeId, dim: usize) -> NodeId;
    pub fn attn(&mut self, q: NodeId, kv: NodeId, pos: NodeId,
                mode: AttnMode, meta: AttnMeta) -> NodeId;  // src = [q, kv, pos, span]
    pub fn swiglu(&mut self, gate: NodeId, up: NodeId) -> NodeId;
    pub fn kvcache_store(&mut self, layer: usize, k: NodeId, v: NodeId,
                         pos: NodeId, n_ctx: usize) -> NodeId;
    pub fn kvcache_load(&mut self, layer: usize, n_embd: usize,
                        n_ctx: usize, n_head_kv: usize) -> NodeId;
    pub fn output(&mut self, node: NodeId);
    pub fn build(self) -> ComputeGraph;

    // decode fused constructors (§5.3)
    pub fn fused_qkv(&mut self, x: NodeId, pos: NodeId, layer: usize, meta: FusedQkvMeta) -> NodeId;
    pub fn qkv_bias_rope_store(&mut self, q: NodeId, k: NodeId, v: NodeId, pos: NodeId,
                               layer: usize, meta: QkvBiasRopeStoreMeta) -> NodeId;
    pub fn fused_ffn(&mut self, x: NodeId, meta: FusedFfnMeta) -> NodeId;
    pub fn fused_qkv_norm(&mut self, x: NodeId, pos: NodeId, layer: usize,
                          meta: FusedQkvNormMeta) -> NodeId;
}
```

Two shape rules are worth calling out because the fused nodes depend on them:

- `attn()` takes its output shape from `AttnMeta` (`n_head * hd`) rather than from the `q` input,
  because a fused QKV node's `q` handle actually points at a larger `q|k|v` concat buffer.
- `fused_qkv` / `fused_qkv_norm` output `[nqt + 2*nkt, nt]`; `fused_ffn` outputs `[2*nf, nt]`; the
  downstream matmul reads the rows it needs (`0..nqt` q for attention, `0..nf` for the FFN down
  projection).

### 3.3 Allocator (`graph/alloc.rs`)

```rust
pub struct GraphAllocator {
    cpu: CpuBackend,
    metal: Option<MetalBackend>,      // macOS
    cuda: Option<CudaBackend>,        // feature = "cuda"
    node_to_buf: HashMap<NodeId, BufRef>,
    cross: HashMap<NodeId, BufRef>,   // split-boundary staging for the current graph
    buf_alive: HashMap<(Backend, usize), usize>,
    kv: HashMap<usize, [BufRef; 2]>,  // per-layer [K, V] persistent regions
    pub persistent: Vec<PersistentBuf>,
}
```

`alloc_graph` runs on every build/rebuild and performs, in order:

1. **Release the previous graph's liveness buffers** into their pools (`buf_alive` and `node_to_buf`
   are cleared; the `cross` staging buffers are freed and re-materialized on the next execute).
   Persistent regions are *not* touched — they are the KV cache.
2. **Validate acyclicity** with `topo_order()`, then keep **build order** for everything else.
3. **Compute liveness.** `last_use[node]` starts at its exec index and is extended to the largest
   index of any consumer. `graph.outputs` get `last_use = order.len()`. **Inputs get the same
   treatment**: they are host-filled before execution, so reusing an input's buffer for another input
   would let the later fill clobber the earlier one (deviation 23).
4. **Count consumers** per node, for in-place alias safety.
5. **Walk in build order**, calling `sweep(i)` to return expired buffers, then allocate:
   - `KvcacheStore`/`KvcacheLoad` bind the node buffer to the layer's **K** region; `ensure_kv`
     allocates the never-freed `[K, V]` pair on first use for that layer/backend;
   - `FusedQKV` / `FusedQkvNorm` / `QkvBiasRopeStore` also `ensure_kv` (their kernels store K/V, with
     the size taken from `meta.kv_elems`) but keep a normal concat/epilogue output buffer;
   - `Silu`, `RoPE` and `QkvBiasRopeStore` alias their input buffer when the input's **sole
     consumer** is this node and both are on the same backend; otherwise they get a fresh buffer;
   - every other node gets a pooled buffer sized by `n_elements()`.

The in-place alias rule is what makes kernel-order execution correct without a host copy: a pending
GPU producer and this kernel read/write the same physical buffer. A cross-backend input is never
aliased (the producer already completed at the split boundary, so the staging copy is safe).

Host access and transfer helpers:

```rust
pub fn fill_input(&mut self, graph, name: &str, data: &[f32]) -> Result<(), String>;
pub fn fill_input_i32(&mut self, graph, name: &str, data: &[u32]) -> Result<(), String>;
pub fn get_buffer(&self, graph, id) -> Option<&[f32]>;          // CPU pools only
pub fn copy_to_cpu(&mut self, id) -> Option<Vec<f32>>;          // any backend
pub fn copy_kv_to_cpu(&mut self, layer) -> Option<(Vec<f32>, Vec<f32>)>;
pub fn sync_backend(&mut self, backend: Backend);
pub fn copy_across(&mut self, node_id, dst_backend) -> Result<(), String>;
pub fn cross_buffer(&self, node_id) -> Option<BufRef>;
pub fn alloc_persistent(&mut self, name, backend, size) -> BufRef;
```

I32 inputs (token ids, positions, tail ids) are stored as `f32::from_bits` bit patterns — exact for
`|v| < 2^24`, which covers every supported model's vocab and context. CPU and Metal read them back
with `to_bits()`; CUDA converts the buffer to a real `i32` plane on device, memoized per execution
window.

Cross-backend staging is deliberately **not** an overwrite of the node's canonical buffer.
`copy_across` allocates a staging buffer on the consumer's backend (once per graph, then rewritten on
each execute) and records it in `cross`; consumers on the producer's own backend keep reading the
canonical buffer. This is what keeps a reused graph re-executable.

`GraphAllocator` implements `KvProvider`, so `kv_pair(layer)` is the only way any executor reaches
the persistent regions — the V half in particular has no dedicated node and is reachable only
through this pair.

### 3.4 Scheduler (`graph/scheduler.rs`)

```rust
pub struct Split {
    pub backend: Backend,
    pub node_range: (usize, usize),   // [start, end) in graph.nodes
    pub inputs: Vec<NodeId>,          // produced by another backend, copied in
    pub outputs: Vec<NodeId>,         // consumed by another backend
}

impl BackendScheduler {
    pub fn assign_backends(&self, graph: &mut ComputeGraph, alloc: &GraphAllocator);
    pub fn split_graph(&self, graph: &ComputeGraph) -> Vec<Split>;
    pub fn execute(&self, graph: &ComputeGraph, alloc: &mut GraphAllocator) -> Result<(), String>;
}
```

- **`assign_backends`** asks `alloc.supports(op, dtype)` for each unassigned node and falls back to
  CPU via `.or(Some(Backend::CPU))`. `supports` consults Metal first (macOS), then CUDA
  (`feature = "cuda"`), then CPU. Nodes with an explicit backend keep it.
- **`split_graph`** scans nodes in order and cuts whenever the assigned backend changes, then derives
  cross-split edges: a `src` living in another split becomes an `input` of this split and an `output`
  of the producer's split. The result is a contiguous partition — there is no split-merging pass.
- **`execute`** flushes the previous backend, copies the new split's inputs across, then runs its
  nodes. Cross-backend inputs are resolved through `cross_buffer(src)` filtered to the executing
  backend, falling back to the canonical `node_buffer(src)`; this filtering matters when one value
  feeds two backends and only one staging copy exists.
- **KV resolution** happens before the backend borrow: `KvcacheStore`, `FusedQKV`,
  `QkvBiasRopeStore`, `FusedQkvNorm` and `Attn` (via `AttnMeta.layer`) all call `alloc.kv_pair(layer)`
  and pass it to `execute_node`.
- **Dead nodes are skipped.** Fusion can orphan a node (e.g. the `Silu` folded into `SwiGLU`); such a
  node has no buffer and is not executed.
- **Split mismatch is an error**, not a fallback: if a node's buffer is on a different backend than
  the executing split, `execute` returns an `Err` naming the node and both backends.

Diagnostics: `MINFER_GRAPH_TRACE=1` prints the split list and a per-op/per-backend node count.

The scheduler is also where the optional per-node data capture for the visualizer hooks in (§8.3) and
where CUDA Graph replay short-circuits a split's node loop (§9.4).

### 3.5 Backend trait (`graph/backend.rs`)

```rust
pub trait KvProvider {
    fn kv_pair(&self, layer: usize) -> Option<(usize, usize)>;   // (k_buf_id, v_buf_id)
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;                                       // diagnostics
    fn supports_op(&self, op: &Op, dtype: DType) -> bool;
    fn supports_fused(&self, fused: &FusedOp) -> bool;

    fn alloc_buffer(&mut self, size: usize) -> usize;             // from the recycle free list
    fn free_buffer(&mut self, id: usize);
    fn alloc_fresh(&mut self, size: usize) -> usize;              // bypasses the free list
    fn pool_len(&self) -> usize;                                  // E4: did the pool grow?

    fn execute_node(&mut self, node: &CNode, in_bufs: &[BufRef], out_buf: BufRef,
                    kv_pair: Option<(usize, usize)>) -> Result<(), String>;

    fn read_host(&self, id: usize) -> Option<&[f32]>;
    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String>;   // exact length
    fn write_host_window(&mut self, id: usize, offset: usize, data: &[f32])    // E4 S2: a window of
        -> Result<(), String>;                                                 // a class-sized buffer
    fn synchronize(&mut self);

    #[cfg(feature = "cuda")]
    fn graph_replay(&mut self, uid: u64, range: (usize, usize),
                    nt_hint: Option<usize>) -> bool { false }
}
```

Three deviations from the plan's sketch are worth stating explicitly:

- `execute_node` takes `&mut self` (the pool is mutated) and returns a `Result` (invariant 7).
- `kv_pair` is passed in rather than looked up: the backend cannot see the allocator's KV map, and
  the scheduler already resolves it.
- `alloc_fresh` exists because a split-boundary staging buffer must not be recycled during the same
  execute: the free list can hold ids whose physical contents are still referenced by `node_to_buf`
  and read later in the same pass. Fresh buffers re-enter the free list on the next `alloc_graph`.

`read_host` is a borrowed view, which CUDA cannot provide; CUDA returns `None` and exposes the real
device→host path through the allocator's `copy_to_cpu` (§9.5).

`graph_replay` is CUDA-only. Returning `false` may still have armed or entered capture mode as a side
effect (backend-internal warmup bookkeeping); the captured window is closed at the split's
`synchronize`. The default implementation is a no-op, so CPU and Metal drop the method entirely.

### 3.6 Parameters and the reuse cache (`graph/params.rs`, `graph/cache.rs`)

```rust
pub enum GraphType { Decode, Prefill }

pub struct CParams {
    pub n_ctx: usize,
    pub flash_attn: bool,
    pub gpu: bool,        // whether a GPU backend participates (assignment is part of topology)
    pub fuse_qkv: bool,   // G4 decode QKV fusion (MINFER_NO_FUSE_QKV=1 disables)
    pub fuse_ffn: bool,   // G5 decode FFN fusion (MINFER_NO_FUSE_FFN=1 disables)
}

pub struct GraphParams {
    pub n_tokens: usize,
    pub n_out: usize,          // tail rows (G3): part of topology, not just execution
    pub gtype: GraphType,
    pub cparams: CParams,
    pub weights_version: u64,  // reserved for weight reload / LoRA switch
}

pub struct GraphCache {
    graph: Option<ComputeGraph>,
    alloc: GraphAllocator,     // outlives rebuilds: holds the KV regions
    prev_params: Option<GraphParams>,
}
```

`GraphCache::try_reuse` compares `params_match` — `n_tokens`, `n_out`, `gtype`, `cparams`
(all of it, including `gpu`/`fuse_qkv`/`fuse_ffn`/`explicit_span`) and `weights_version`. It never
inspects the node sequence. A batch's *sequence count* is deliberately absent: it is data (A7/E2), and
a 1-sequence and a 2-sequence batch of the same shape share one graph. `replace_graph` assigns a fresh monotonic `uid` for CUDA Graph caching and leaves the
allocator in place. See §6 for the full reuse story.

`weights_version` is currently a constant `1` at both model call sites: there is no LoRA or weight
reload path yet, and no `weights_version()` trait method. The field and the `next_weights_version()`
counter exist so a future reload can break reuse by bumping it.

---

## 4. Model Graph Construction

Both supported architectures build the graph in `models/<arch>/graph.rs` through `GraphBuilder`. The
skeleton is shared; the differences are all in the projections and norms.

### 4.1 Qwen2 — prefill and decode

`Qwen2Graph::build` emits, in order:

1. **Inputs** — `token_ids` `[nt]` I32, `positions` `[nt]` I32, and (only when `n_out < nt`)
   `tail_ids` `[n_out]` I32. `tail_ids` is declared at the graph head, not next to its consumer: an
   input node in the middle of the graph would split a GPU run into extra CPU/CUDA boundaries, each
   with a stream sync and host round trip (R3-A1, `docs/CUDA_OPTIMIZATION.md`).
2. **Embedding** — `b.embedding(token_ids, tok_embd)`.
3. **Per layer** `il`:
   - `residual = h`; `normed = rms_norm(h, attn_norm, eps)`;
   - Q/K/V projections — one of three shapes (§4.2);
   - `attn_out = attn(q, kv, positions, mode, AttnMeta{ layer: il, ... })`, where `mode` is `Flash`
     when `cparams.flash_attn` else `Gqa`;
   - `wo = matmul(attn_out, wo)`;
   - residual add, with the G3 tail reduction on the last layer (§4.3);
   - `residual = h`; `normed = rms_norm(h, ffn_norm, eps)`;
   - FFN — fused or unfused (§4.2);
   - residual add.
4. **Output** — `rms_norm(h, output_norm)` → `matmul(normed, output, output_b)` →
   `b.output(logits)`.

For `nt > 1` the QKV and FFN blocks are always the unfused chains, because every fusion gate requires
`nt == 1`. Prefill therefore runs the `FusionPass`-rewritten `SwiGLU` in the FFN and nothing else.

### 4.2 The decode fusion branches

The Q/K/V construction is where topology differs materially between prefill and decode. For decode
(`nt == 1`) with a GPU participating and `CParams.fuse_qkv` on, and provided all three biases exist:

| Class | Condition | Graph |
|---|---|---|
| Concat (G4/D3-8 class 1) | `qkv_concat_available(wq, wk, wv)` — same quant type, same input dim, block-aligned, so the loader registered `blk.{i}.attn_qkv` | one `fused_qkv(normed, positions, il, ...)` node (the builder also wires `cells`), then `kvcache_load` for attention |
| Mixed quant (D3-8 class 2) | concat unavailable, CUDA present | three bias-less `matmul`s + one `qkv_bias_rope_store(q, k, v, positions, il, ...)` epilogue (plus `cells` from the builder); attention is wired to the epilogue node so `q`'s matmul has exactly one consumer and can be aliased in place |
| Unfused | any gate off (or a Metal-only mixed-quant layer) | `matmul`×3 (+bias) → `rope`×2 → `kvcache_store` → `kvcache_load` |

Class 2 is CUDA-only: without the `cuda` feature `qkv_epilogue_ok` is `false` and those layers keep
the unfused chain on Metal, bitwise-neutral against the pre-D3-8 behaviour. The FFN branch: for
decode with `CParams.fuse_ffn`, `gu_concat_available(ffn_gate, ffn_up)` and `nf <= 16384`, the
builder emits one `fused_ffn(normed, ...)` followed by the down matmul; otherwise it emits
`matmul(gate)`, `matmul(up)`, `silu`, `mul` and lets `FusionPass` fold the last two.

Two gates are measured decisions, not correctness ones:

- **`nf <= 16384`.** On the 7B class the concat matmul (`od = 2*nf ≈ 37888`) is slower than two
  separate matmuls with the decode `nt == 1` kernel, while on 0.5B the fusion is worth ~3%.
- **`concat_rows_feasible` on the CUDA Qwen2 path.** The probe is metadata-only because rebuilding
  the concat rows during graph construction cost ~920 ms per decode graph build (a ~1.9 GB probe);
  the loader has already built the concatenated weight, so the builder only needs to confirm it is
  feasible.

Both gates live in the builder, and `CParams.fuse_qkv` / `fuse_ffn` keep the A/B honest across graph
reuse.

### 4.3 G3 — the `n_out` tail-row reduction

Prefill computes logits for the last `n_out` tokens only (the CLI uses `n_out = 1`). Rather than
computing all `nt` rows and slicing, the builder inserts, after the last layer's `wo`:

```
cur_tail = get_rows(wo,       tail_ids, [n_embd, n_out])
res_tail = get_rows(residual, tail_ids, [n_embd, n_out])
h        = add(res_tail, cur_tail)
```

so the last layer's `ffn_norm`, gate/up/down, the residual adds and the `lm_head` matmul all run on
`n_out` rows. This mirrors llama.cpp's `ggml_get_rows(cur, inp_out_ids)` at the final layer. When
`n_out == nt` (decode), no `tail_ids` input and no reduction node are built. `GraphParams.n_out`
participates in the reuse identity precisely because it changes the topology.

The logits buffer is exactly `n_out * n_vocab` either way (G3-reduced, or `n_out == nt`), so the
extraction path does not clone a full-vocabulary row set.

### 4.4 Qwen3 differences

Qwen3 shares the skeleton and adds a per-head RMSNorm on Q and K (`attn_q_norm` / `attn_k_norm`) and
has no attention biases. Its head dim is decoupled from `n_embd / n_head` (128 vs 64 on 0.6B),
which drives the Q/K/V/wo widths, the RoPE dims, the attention scale and the KV row stride.

- Unfused: `qk_norm(q, q_norm, hd, nh, eps)` and `qk_norm(k, k_norm, hd, nk, eps)` after the three
  bias-less matmuls and before the rope. `QkNorm` treats the flat token-major buffer as a contiguous
  `[nt*nh, hd]` matrix and reuses the RMSNorm kernels with `d = hd`, `n = nt*nh`.
- Decode fused: `fused_qkv_norm(normed, positions, il, ...)` — one concat matmul, then per-head q/k
  norm, then a no-bias rope + K/V store pass. It exists as a separate op because the Qwen2
  `attn_bias_rope_store` kernel cannot express the per-head norm; the Qwen2 path is untouched.
  This fusion is **Metal-only** (the Qwen3 concat probe has no CUDA arm and `fuse_qkv` is gated on
  `metal_on`), so CUDA Qwen3 runs the unfused qk_norm chain.
- No `QkvBiasRopeStore` class exists for Qwen3 (there are no attention biases to fold).
- Qwen3 uses the same G3 tail reduction and the same `FusedFFN` gate.

### 4.5 Graph inputs

| Input | Shape | DType | Present when |
|---|---|---|---|
| `token_ids` | `[nt, 1, 1, 1]` | I32 | always |
| `positions` | `[nt, 1, 1, 1]` | I32 | always (sequence-relative: RoPE + the causal mask) |
| `cells` | `[nt, 1, 1, 1]` | I32 | when the graph writes or resolves KV (`KvcacheStore`, `FusedQKV`, `QkvBiasRopeStore`): the arena row per token (C6) |
| `tail_ids` | `[n_out, 1, 1, 1]` | I32 | `n_out < nt` (prefill with the G3 reduction) |

The prefill / decode call sites fill `tail_ids` with `[(nt - n_out) .. nt)`; decode leaves it absent.
Positions in the graph are always data: no node payload contains `n_past`, and the KV region is
allocated as `[n_embd, n_ctx]` with only its written prefix read at execution time.

---

## 5. Operator Fusion

minfer fuses in two distinct ways, and the distinction matters when reading the IR.

### 5.1 Two mechanisms

| Mechanism | When | Examples | Gated by |
|---|---|---|---|
| **Build-time fused node** | The builder knows the pattern statically at graph construction | `FusedQKV`, `QkvBiasRopeStore`, `FusedFFN`, `FusedQkvNorm` | `CParams.fuse_qkv` / `fuse_ffn` (+ weight availability, quant class, `nf` cap) |
| **`FusionPass` rewrite** | A peephole pass over the built graph, per node's assigned backend | `SwiGLU` (the only rule) | backend `supports_fused` |

Build-time fusion is preferred where the pattern spans weight registration (concat weights) or where
the fused node needs extra metadata (layer index, biases, rope parameters). The peephole pass covers
patterns that are cheap to recognize and safe to leave unfused — the unfused form is still correct,
just slower.

### 5.2 `FusionPass` rules

```rust
impl FusionPass {
    pub fn run(&self, graph: &mut ComputeGraph, backends: &[&dyn Backend],
               backend_of: &dyn Fn(&ComputeGraph, usize) -> Option<usize>) -> usize;
}
```

- **SwiGLU**: `Mul(Silu(x), y)` → `SwiGLU(x, y)`; recognized with `Silu` on either side. Gate: the
  mul node's backend reports `supports_fused(FusedOp::SwiGLU)` — true for CPU, Metal and CUDA.
- **FusedBiasRope (removed)**: the plan's `RoPE(Add(x, b), pos)` → `FusedBiasRope(x, b, pos)` rule
  was implemented but **no backend ever advertised `FusedOp::BiasRope`**, so the rewrite was
  unreachable and `Op::FusedBiasRope` could never be constructed. Rule, op and capability tag were
  removed; `FusedOp` now has a single variant (`SwiGLU`). The bias+rope work is covered by the
  build-time fused nodes instead (Metal/CUDA `attn_bias_rope_store`, which also does the KV store —
  a strictly stronger fusion).

The pass rewrites `op` and re-points `src`; orphaned producers (the old `Silu`) become dead nodes
and are skipped by the scheduler and the allocator. It returns the rewrite count, which the tests
use.

One CPU nuance: `CpuBackend::supports_fused` accepts `SwiGLU`, and its `Op::SwiGLU` execution is a
single pass (`vec_ops::vec_swiglu_f32`, `dst[i] = silu(gate[i]) * up[i]`). It is bit-identical to
the `vec_silu_f32` + `vec_mul_f32` pair it replaced (same formula, same per-element order) but does
not allocate the full-size intermediate buffer that pair needed.

Because fusion rewrites the IR, an unfused-vs-fused comparison must run `FusionPass` on the unfused
side too; otherwise `silu` + `mul` execute as two kernels and differ from the single swiglu kernel by
float noise (~1e-6, amplified at large values). The real forward path always runs the pass.

### 5.3 Build-time fused ops

| Op | Shape of the win | Where it executes |
|---|---|---|
| `FusedQKV` | replaces 3 matmul + 3 bias + 2 rope + 2 store (10 dispatches) with one concat matmul + one fused bias/rope/store kernel (2) | Metal and CUDA |
| `QkvBiasRopeStore` | mixed-quant layers that cannot share a concat matmul: the 3 matmuls stay, but 3 bias + 2 rope + 2 store become 1 epilogue pass | CUDA only (D3-8 class 2) |
| `FusedFFN` | replaces 2 matmul + silu + mul (4) with one concat matmul + one in-place swiglu (2); gate `nf <= 16384` | Metal and CUDA |
| `FusedQkvNorm` | Qwen3's concat matmul + per-head q/k norm + no-bias rope/store | Metal only |

All four are part of the topology, so their enable flags are part of `CParams` and therefore of the
reuse identity. Fused vs unfused output is bit-identical on the same backend (verified by
`fused_qkv_matches_unfused_decode` and its Qwen3 counterpart), and the fused kernels are
decode-only: `nt == 1` is a `debug_assert` plus a shape gate in each backend.

### 5.4 Deferred: `BatchMatMul`

The plan proposed folding three sibling matmuls sharing one activation into one `BatchMatMul` node,
to quantize the activation once on the CPU path. It is **not implemented**: the IR gives every node
exactly one output buffer, and `BatchMatMul` is inherently multi-output. Expressing it needs either a
multi-output node kind or a concat-plus-view encoding, and both are larger IR changes than the
measured CPU prefill gain justifies today. The `Op::BatchMatMul` variant remains in the IR with a
comment recording the reason; the corresponding `FusedOp::BatchMatMul` capability tag was removed
(no rule ever probed it).

### 5.5 Operator vocabulary not emitted

`Scale`, `Softmax`, `View`, `Reshape`, `Permute`, `AttnMode::Mha` and `BatchMatMul`
are represented in the IR for ggml parity but no supported architecture builds them: attention
kernels fuse their own softmax, the Qwen2/Qwen3 graphs need no view/reshape node, and the bias+rope
rewrite was removed (§5.2). `Scale` and `Softmax` are additionally unsupported by Metal and CUDA
at execution time.

### 5.6 Toggles

| Env var | Effect | Reuse impact |
|---|---|---|
| `MINFER_NO_FUSE_QKV=1` | disables the decode QKV fusion (`CParams.fuse_qkv = false`) | forces a rebuild, by design |
| `MINFER_NO_FUSE_FFN=1` | disables the decode FFN fusion (`CParams.fuse_ffn = false`) | forces a rebuild |
| `MINFER_DISABLE_MPS=1` | no Metal participation (`CParams.gpu` / assignment change) | forces a rebuild |
| `MINFER_GRAPH_TRACE=1` | prints splits + per-op/backend counts | none (diagnostic only) |
| `MINFER_TRACE`, `MINFER_GRAPH_DUMP` | per-node data capture / dumps (§8) | none |

### 5.7 Measured effect

| Change | Model | Effect |
|---|---|---|
| G4 `FusedQKV` | Qwen2.5-0.5B Q4_0 decode | ~269 → ~299 tok/s (+11% at KV 440); logits bit-identical |
| G4 `FusedQKV` | Qwen2.5-7B Q4_K_M decode | flat (Q4_K GEMM-bound) |
| G5 `FusedFFN` | Qwen2.5-0.5B Q4_0 decode | ~303 → ~312–331 tok/s (+3%) |
| G5 `FusedFFN` | Qwen2.5-7B Q4_K_M decode | gate off (`nf > 16384`), unchanged |

Numbers are from the phase ledger (§17); `docs/CPU_OPTIMIZATIONS.md`, `docs/METAL_OPTIMIZATIONS.md`
and `docs/CUDA_OPTIMIZATION.md` hold the full measurement context.

---

## 6. Graph Reuse Mechanism

### 6.1 The invariant

> The graph topology is a deterministic function of `GraphParams`. Equal params ⇒ identical
> topology ⇒ the cached graph and its buffers can be reused as-is; only input data is refreshed.

`n_past` (the KV position) is deliberately absent from `GraphParams`: it enters through the
`positions` input node. This is the same invariant as llama.cpp's `llm_graph_params::allow_reuse`.

### 6.2 Reuse flow

```rust
// GraphCache::try_reuse — params-only
match (&self.prev_params, &self.graph) {
    (Some(prev), Some(_)) if Self::params_match(prev, params) => {
        self.prev_params = Some(params.clone());
        true
    }
    _ => false,
}

// GraphCache::replace_graph — keep the allocator (KV lives there), assign a fresh uid
pub fn replace_graph(&mut self, mut graph: ComputeGraph, params: GraphParams) {
    graph.uid = NEXT_GRAPH_UID.fetch_add(1, Ordering::Relaxed);
    self.graph = Some(graph);
    self.prev_params = Some(params);
}
```

A caller that gets `false` from `try_reuse` builds a new graph, assigns backends, runs `FusionPass`,
calls `alloc_graph`, and stores it. The allocator object is never replaced, so the persistent KV
regions survive a prefill→decode rebuild — the transition that changes `n_tokens`/`gtype` and
therefore necessarily rebuilds.

### 6.3 What forces a rebuild

| Parameter | Why it changes topology |
|---|---|
| `n_tokens` | matmul output shapes, KV store shape, `n_out < nt` decision |
| `n_out` | decides whether the G3 tail reduction nodes exist |
| `gtype` | Decode vs Prefill, and the decode-only fused branches |
| `cparams.n_ctx` | sizes the persistent KV regions |
| `cparams.flash_attn` | selects `AttnMode::Flash` vs `Gqa` |
| `cparams.gpu` | backend assignment is a build-time decision |
| `cparams.fuse_qkv` / `fuse_ffn` | fused nodes are topology |
| `weights_version` | reserved for weight reload / LoRA switch (constant 1 today) |

### 6.4 Debug structural check

`GraphCache::verify_structural(graph)` is compiled in debug test builds only. It compares two graphs
built from equal params node by node (`op` with full payloads, `out_shape`, `src`) and returns
`false` on any difference. `cache.rs` has a test asserting that a graph with an extra node fails the
check — the guard against a non-deterministic builder silently corrupting reuse.

`cache.rs` also pins the fusion flags into the reuse identity with a test
(`fuse_flags_are_part_of_the_reuse_identity`): flipping `fuse_qkv` or `fuse_ffn` in either direction
must force a rebuild, so no one can drop the fields from `CParams` and silently break the A/B envs.

### 6.5 `uid`

`uid` is monotonic per process, starts at 1, and is assigned only by `replace_graph`; a reused graph
keeps its uid. CUDA uses it as the CUDA Graph cache key together with the node range and the token
hint (`ComputeGraph::capture_nt_hint` returns the first matmul's output row count, `None` for graphs
without matmuls).

---

## 7. CPU and Metal Backend Mapping

The full Metal backend design (device/kernel layer, dispatch, command buffers, model wiring, safety)
is `docs/METAL-BACKEND-DESIGN.md`; this section is the condensed mapping of the graph-facing surface.

### 7.1 CPU (`graph/cpu_backend.rs`)

`CpuBackend` is `{ buffers: Vec<Vec<f32>>, free: Vec<usize>, weights: HashMap<String, Tensor> }`.
`alloc_buffer` reuses the first free id whose length matches exactly and zero-fills it; `alloc_fresh`
always appends; `free_buffer` only returns the id to the list. Re-registering an existing weight is
skipped, and that skip is load-bearing: `Tensor` owns its bytes, so the model call sites' `clone()`
would deep-copy ~4.4 GB on 7B at every graph rebuild (~635 ms measured).

`supports_op` gates on `dtype == DType::F32` for **every** op, including `Input`. I32 input nodes are
therefore not "supported" by CPU in the capability sense and land on CPU only through the scheduler's
`or(Some(Backend::CPU))` default. `supports_fused` accepts only `SwiGLU`.

| Op | CPU execution |
|---|---|
| `Input` | no-op (host-filled) |
| `Add` / `Mul` | `vec_ops::vec_add_f32` / `vec_mul_f32` |
| `Silu` | `vec_ops::vec_silu_f32` in place |
| `Scale` | copy + `vec_scale_f32` |
| `RmsNorm` / `QkNorm` | per-row `rms_norm_fused_f32` (or `rms_norm_f32`); `QkNorm` uses `d = hd` |
| `MatMul` | F32 weight → `vec_ops::mat_mul_f32`; quantized → `kernel::cpu_quant_matmul_f32` (Q8_0 activations on the fly); optional bias added scalar |
| `GetRows` | embedding path (`kernel::embed_tokens`) or generic gather |
| `RoPE` | local `cpu_rope` (copy then transform) |
| `Softmax` | dims 0/1 only; other dims are an `Err` |
| `SwiGLU` | `vec_ops::vec_swiglu_f32`, one pass (`dst[i] = silu(gate[i]) * up[i]`) |
| `KvcacheStore` | per-token copy into the K and V regions; `pos >= n_ctx` is an `Err` |
| `KvcacheLoad` | no-op — the node buffer *is* the K region |
| `Attn` | `cpu_gqa_attn`, parallelized over heads via `kernel::par_for` |
| `View`/`Reshape`/`Permute` | identity copy |
| any fused op it does not support | `Err("op ... unsupported on CPU ...")` |

Aliased inputs (`Silu`, `RoPE`) are snapshotted into a local `Vec` before the pool is split, so the
kernel reads the producer's values even though output and input are the same buffer. A recorded
regression (8a②) is that an F32-weight matmul with `nt > 1` must produce token-major `[nt][od]`
output; the earlier `[od][nt]` layout was silently wrong for prefill only.

### 7.2 Metal (`graph/metal_backend.rs`)

`MetalBackend` holds `{ state: &'static MpsState, pool, free, staging, free_staging, cb_ptr }`. Its
pool buffers are shared-mode f32 MTLBuffers, so `read_host`/`write_host` are direct memory views.
`staging`/`free_staging` are capture-only buffers used by `capture_split`. The backend holds **no
weight map**: every op resolves `state.weight_buf(name) -> (MetalBuffer, byte offset)` — weights are
registered in `MpsState` as zero-copy `newBufferWithBytesNoCopy` slices over the mmap'd GGUF parts
(`MINFER_WEIGHT_COPY=1` forces a copy). One `MpsCommandBuffer` per split is submitted by
`synchronize()`; `Drop` flushes a pending buffer.

`supports_op` accepts `Input` at any dtype and F32 for the element-wise, norm, matmul, rope,
attention, KV, `GetRows` and decode-fused ops; `View`/`Reshape`/`Permute` are accepted.
`supports_fused` is `matches!(fused, FusedOp::SwiGLU)` — the only fusion rule that exists (§5.2).

| Graph op | MpsState method / kernel |
|---|---|
| `Add` / `Mul` | `add_f32` / `mul_f32` |
| `Silu` | `silu_f32` (in place) |
| `RmsNorm` | `rms_norm_256` when `rms_norm_256_enabled()` (G2; `MINFER_NO_RMS_256` opt-out), else `rms_norm` |
| `QkNorm` | same kernels with `d = hd`, `n = len/hd` |
| `MatMul` | `quant_matmul_f32_on_gpu_buf`; per-ttype `*_f32_matmul` (nt==1) / `*_multi` (nt>1) / GEMM (`nt >= 2 && (od >= 2048 || nt >= 9) && gemm_enabled()`); optional `add_bias_f32` |
| `GetRows` | `embed_tokens_gpu` (per-quant kernels) or `get_rows_f32` for the G3 gather |
| `RoPE` | `rope_f32` (copy_in first when not aliased) |
| `SwiGLU` | `swiglu_f32` |
| `Attn` | G1 dispatch (below) |
| `KvcacheStore` | two `store_kv` calls (K then V); f32 or f16 KV selected by `kv_cache_is_f16()` / `MINFER_CACHE_TYPE` |
| `KvcacheLoad` | no-op (view of the K region) |
| `FusedQKV` | concat `quant_matmul_f32_on_gpu_buf` + `attn_bias_rope_store` (3 bias + q/k rope + K/V store in one pass) |
| `FusedFFN` | concat matmul + `swiglu_f32_off` (in-place on the concat buffer, gate at offset 0, up at `nf`) |
| `FusedQkvNorm` | concat matmul + two in-place per-head `rms_norm[_256]` (q at byte offset 0, k at `nqt*4`) + `attn_rope_store` |
| `Scale` / `Softmax` / `BatchMatMul` | `Err` (no kernel) |
| `QkvBiasRopeStore` | `Err` (CUDA-only epilogue) |

The `*_off` kernel variants (`swiglu_f32_off`, `rope_f32(off)`, `rms_norm(off_x, off_y)`,
`add_bias_f32(off)`, `store_kv(off)`, `attn_bias_rope_store(bias offsets)`) exist so a fused node can
operate on a section of one physical buffer without an extra copy.

#### Attention dispatch (G1)

Pre-dispatch guards return `Err` before anything is encoded: `nkt == n_head_kv * hd`,
`hd == hd_kv`, and the layer's `kv_pair` must exist.

For decode (`nt == 1`), first match wins:

1. `flash_attn_enabled(hd)` (`MINFER_NO_FLASH != "1" && hd ∈ {64,128}`) → `gqa_attn_flash`, with the
   `hd128` kernel variants.
2. else `hd ∈ {64,128}` and `MINFER_NO_SPLIT_ATTN != "1"` → `gqa_attn_split_f32` (partial + combine).
3. else → `gqa_attn_f32` (classic).

For prefill (`nt > 1`) with `hd ∈ {64,128}`:

1. `prefill_flash_enabled(hd)` (`MINFER_NO_PREFILL_FLASH != "1"`) → `attn_flash_prefill`.
2. else `matmul_attn_enabled()` (`MINFER_NO_MATMUL_ATTN != "1"`) → `attn_parallel_prefill`.
3. else → `gqa_attn_f32(..., nt)`.

Other head dims always use the classic kernel. The KV-parallel chunk count is `MINFER_ATTN_CHUNKS`
or `((max_pos + 1 + 31) / 32).clamp(1, 16)`. This mirrors the pre-graph path exactly, which is why
G1 was bitwise-neutral; the fast paths live in kernels that were already isolated-tested.

### 7.3 In-place execution and the aliasing rule

`Silu` and `RoPE` execute in place whenever the allocator aliased their input (sole consumer, same
backend). `FusedFFN`'s swiglu runs in place on the concat buffer; `QkvBiasRopeStore` runs in place on
`q`'s buffer. The backends assume the allocator did the safety analysis — `execute_node` does not
re-check consumer counts. Metal's `copy_in` snapshots only the non-aliased case; CPU snapshots every
aliased input because its kernels are written as slice operations.

The hard rule that came out of Phase 3: **never host-copy a buffer with pending GPU work.** A
per-node host readback inside a split with an open command buffer reads stale data. This surfaced as
an all-zero KV region and garbled output before the in-place aliasing rule was introduced.

### 7.4 Error contract

Kernel-invariant violations (unsupported shape, device-limit guard failure, missing weight) return
`Err(String)` naming the node and the actual values, and abort the run. There is no "delete the node
and fall back to CPU" path: `assign_backends` already decided the backend before execution, and on
Metal the submit itself panics with the status when a bounded 10 s wait fails. This is the
graph-level expression of `docs/GPU_SAFETY.md`; `metal::gpu_abort` remains the device-configuration
escape hatch that refuses to risk a GPU fault and exits.

---

## 8. Graph Export and Debugging

### 8.1 DOT (`graph/dot.rs`)

`ComputeGraph::dump_dot(w)` writes a Graphviz digraph: one node per `CNode` labelled with its name and
op, filled by backend (Metal light blue, CPU light yellow, CUDA light green, unassigned white), edges
from `src`, and double-circle markers for inputs and outputs.

```
./target/release/minfer --dump-graph /tmp/g.dot <model.gguf> "hello"
dot -Tpng /tmp/g.dot -o /tmp/g.png
```

The export path does not reuse the runtime cache: `main.rs` rebuilds the graph via
`json::build_runtime_graph` (`build_graph` → enable GPU → `assign_backends` → `FusionPass`), so the
DOT and JSON exports contain the fusion and backend assignment the live run used. Both flags exit
immediately after writing (before decode).

### 8.2 JSON for the visualizer (`graph/json.rs`)

`export_graph_json` / `ComputeGraph::export_json` emits `{format:"minfer-graph", version:1, model,
kind:"prefill"|"decode", inputs, outputs, nodes:[{id,name,op,detail,shape,dtype,backend,src,meta}]}`.
`op_name`, `op_detail` and `meta_json` cover every `Op` and `NodeMeta` variant, including the fused
ones. The schema reserves per-node `stats` / `values` fields for trace data, so the page treats their
absence as "no data for this node in this step".

```
./target/release/minfer --dump-graph-json /tmp/g.json <model.gguf> "hello"
./target/release/minfer viz <model.gguf>          # page + live SSE (default port 8081)
```

### 8.3 Per-node data capture

With `MINFER_TRACE=<path>` the scheduler records, after each node executes, the node's output stats
and a downsampled value sample; `viz`'s live mode uses the same machinery over SSE. The capture is
backend-aware:

- **CPU** outputs are read directly from the pool.
- **Metal** outputs are blitted into staging after all of the split's kernels and read back once the
  split's command buffer is submitted — one submit per split, never a per-node GPU flush.
- **CUDA** outputs are queued as stream-ordered D2H copies into pinned capture staging and drained
  with a single sync at the split boundary; tensors above the staging ceiling fall back to a per-node
  synchronous copy.
- **Input nodes** are host-filled, so they are read directly without a sync; `KvcacheLoad` has no
  execution at all and is skipped.
- **KV regions** are skipped (a full region per layer would dominate the trace), so the visualizer
  shows them as "no data".

CUDA Graph replay is disabled while capture is on: a host readback inside a capture window would
corrupt the recorded graph.

The trace JSON is `{format:"minfer-trace", version:1, model, prompt, phases:[{kind, graph, steps:
[{token, text, logits_top, nodes:[{id, dtype, stats, values, stride, n}]}]}]}`. `viz/README.md`
documents the page, the SSE endpoints (`GET /viz/graph`, `GET /viz/events`, `POST /viz/run`) and the
`MINFER_VIZ_DIR` sample directory.

### 8.4 Diagnostics and env vars

| Env var / flag | Effect |
|---|---|
| `--dump-graph <path>` | Graphviz DOT export, then exit |
| `--dump-graph-json <path>` | JSON export for `viz/`, then exit |
| `MINFER_TRACE=<path>` | per-node real-data trace (JSON) |
| `MINFER_GRAPH_TRACE=1` | split list + per-op/backend node counts on stderr |
| `MINFER_GRAPH_DUMP=<dir>` | logits / per-layer KV dumps (read in the model graph modules) |
| `MINFER_REBUILD_TRACE` | logs each graph rebuild (Qwen2) |
| `MINFER_DISABLE_MPS=1` | force CPU participation off for Metal |
| `MINFER_NO_FUSE_QKV` / `MINFER_NO_FUSE_FFN` | disable the decode fusions (A/B) |
| `MINFER_VIZ_DIR` | directory for viz samples (default `viz`) |
| `MINFER_BENCH_WARMUP_MS` | `bench` warmup budget |

> **Resolved (2026-09): export/trace fusion gate.** The export/trace paths used to gate `fuse_qkv`
> on `metal_on` only while the Qwen2 runtime gates it on `metal_on || cuda_on`, so a CUDA-only run
> exported a graph without the `FusedQKV` nodes the runtime built — diverging node ids. Both paths
> (and the model's own `CParams` construction) now share
> `graph::json::preview_fuse_flags(nt, metal_on, cuda_on)`, and `viz/README.md` no longer claims QKV
> fusion is Metal-only. The gate is pinned by
> `graph::json::tests::preview_fuse_flags_include_cuda_only_runs`.

---

## 9. CUDA Backend (design level)

> The kernel-level optimization campaign — MMQ (int8 quantized GEMM), MMVQ (quantized
> matrix-vector), FA (flash attention) tiling, weight-plane prepasses, occupancy work and the
> per-step measurements — is documented in `docs/CUDA-BACKEND-DESIGN.md`,
> `docs/CUDA_OPTIMIZATION.md` and `docs/cuda_optimization_steps/`. This section covers only what the
> graph design depends on.

### 9.1 Structure

`CudaBackend` (`src/graph/cuda_backend.rs`, `feature = "cuda"`) wraps the device layer in
`src/cuda.rs` (`CudaState`, kernel launchers, weight registry, CUDA Graph API). It implements the same
`Backend` trait as CPU and Metal: its own device buffer pool, `execute_node` dispatch, host
read/write, and `synchronize`. `CudaState` is a process-wide `OnceLock` singleton reached through
`CudaState::get()`; `CudaBackend::new()` returns `None` when no device is present (or
`MINFER_DISABLE_CUDA`), and the allocator then declines to enable CUDA. The legacy whole-layer
`cuda.rs::layer_gpu` path is no longer driven by inference; the graph backend is the only path.

Device work is serialized through `CudaState::stream_lock()`, held around every enqueue except while
this backend itself owns an open capture window.

### 9.2 Eligibility

`supports_op` is f32-activation only (`dtype != DType::F32` ⇒ `false`). Unconditionally supported:
`Input`, `Add`, `Mul`, `Silu`, `SwiGLU`, `RmsNorm`, `QkNorm`, `MatMul`, `Attn`, `KvcacheStore`,
`KvcacheLoad`, `View`, `Reshape`, `Permute`, `GetRows`, `FusedQKV`, `QkvBiasRopeStore`, `FusedFFN`.
`RoPE` is supported for `RopeStyle::NonInterleaved` only. `FusedQkvNorm` is not supported (Qwen3's
fused path is Metal-only). `supports_fused` accepts only `FusedOp::SwiGLU`.

Weight-quant eligibility is a **model-level all-or-nothing gate**, not a per-op one:
`Qwen2Graph::weights_on_cuda` / `Qwen3Graph::weights_on_cuda` requires every matmul weight to be one
of Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F32 and to be registered on the device; on any failure
CUDA is not enabled for that model and `CParams.gpu` records the decision. Kernel-invariant checks
that cannot be made at build time (head dims, alignment, `nt == 1` for fused nodes, rope style) run
inside `execute_node` and return `Err` — never a silent CPU fallback.

### 9.3 Execution dispatch

`execute_node` maps each op to a CUDA path selected by shape and weight type. Representative mapping
(the full table is `docs/CUDA-BACKEND-DESIGN.md` §4.4 and the per-round records):

| Op | CUDA path |
|---|---|
| `GetRows` | embedding kernels per quant type, or `gather_rows_f32` for the G3 tail reduction |
| `Add` / `Mul` / `Silu` / `View`-family | device kernels (identity D2D copy for views) |
| `RmsNorm` / `QkNorm` | float4 rms_norm; the MMQ producer-fused variant when a following GEMM can consume a pre-quantized activation plane |
| `SwiGLU` | `swiglu_f32`, or the producer-fused `swiglu_quant` when the consumer is an MMQ GEMM |
| `MatMul` | prefill: int8 MMQ (`MINFER_MMQ`, default on) or f16 wmma GEMM; decode/small-batch: per-quant MMVQ (`*_decode_mmvq`, `_multi` for nt 2..8) or f32 kernels |
| `Attn` | `gqa_attn_split` (decode split-KV), `gqa_attn_split_batched` (2..16 tokens, per-position bitwise-equal), `gqa_attn_f32`/`_f16kv` (prefill; FA tiled prefill for hd 128) |
| `KvcacheStore` | `store_kv_f32` / `store_kv_f16`; the output buffer must be the K region |
| `FusedQKV` | concat matmul + `attn_bias_rope_store` (decode, neox, even hd) |
| `QkvBiasRopeStore` | in-place `q` + `attn_bias_rope_store` (mixed-quant class 2) |
| `FusedFFN` | concat matmul + in-place offset swiglu (decode) |
| any op with no kernel | `Err("cuda: op ... has no kernel ...")` |

Two implementation details matter to the graph contract: the positions buffer is converted to a real
`i32` plane on device and memoized per execution window (the causal bound is derived device-side, so
no host scalar crosses — a precondition for capture), and the MMQ activation-quantize memo is
invalidated at every non-matmul node.

### 9.4 CUDA Graph capture/replay

`graph_replay(uid, range, nt_hint)` is the graph-level hook the original plan asked for: a previously
captured decode split replays as one captured CUDA Graph launch, and the scheduler skips that split's
node loop (`continue`). Key properties:

- The cache key is `(uid, node range)`, with `capture_nt_hint` gating capture to decode-shaped graphs
  so a prefill graph is not captured accidentally. Prefill capture is **default ON** since R3-B
  (`MINFER_NO_PREFILL_CAPTURE=1` opts out; `MINFER_CAPTURE_PREFILL=1` is accepted but redundant).
  `MINFER_NO_CUDA_GRAPH=1` disables capture entirely.
- Executions 1 and 2 of a key run direct launches (warmup); on the 3rd the backend takes the stream
  lock and begins capture. The window is closed by `synchronize`, which ends capture, instantiates
  and **launches once** so the step still produces output; later executions replay.
- Pointers inside a captured window must stay stable. Pool ids never move memory, but any allocation
  bumps `pool_gen`; a captured exec whose `pool_gen` differs is destroyed and re-captured.
- Capture is disabled under `MINFER_TRACE` / live viz, because per-node host readbacks inside the
  window are illegal. `abort_capture` ends a window without launching when a node returns `Err`, and
  disables graphs for the session (the recorded launches never executed, so that split's outputs are
  invalid).
- Replay is refused while this backend already has an open capture window.

### 9.5 Buffer pool, staging and readback

`alloc_buffer` matches exact byte sizes in a free list and bumps `pool_gen` (both on reuse and on
`cudaMalloc`); `free_buffer` only recycles — persistent KV regions survive rebuilds and the pool
keeps device memory. `alloc_fresh` always allocates and bypasses the free list, which is what the
allocator uses for split-boundary staging. OOM is not a panic (a null buffer fails later with a real
error) because the backend may be holding the stream lock.

`read_host` returns `None` for CUDA: a staged device→host copy cannot return a borrowed slice. The
real path is `copy_to_host`, called by the allocator's `copy_to_cpu`, which syncs and reads back
through a grow-on-demand pinned buffer (`MINFER_NO_PINNED_READBACK=1` reverts to a pageable copy).
Async H2D input fills use a small pinned ring. The trace path uses a dedicated pinned
`CaptureStaging` (ceiling 128 MiB) so the scheduler can enqueue one stream-ordered D2H per node and
drain them with a single sync at the split boundary.

---

## 10. ModelDef Trait

```rust
pub trait ModelDef: Send + Sync {
    fn forward(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache,
               n_out: usize, n_ctx: usize) -> Vec<f32>;
    fn as_any(&self) -> &dyn std::any::Any;

    fn build_graph(&self, params: &GraphParams) -> ComputeGraph;            // default: unimplemented!
    fn forward_graph(&self, tokens, positions, kv, n_out, n_ctx) -> Vec<f32>; // default: forward()
    fn forward_graph_cached(&self, tokens, positions, n_out, n_ctx,
                            cache: &mut GraphCache) -> Vec<f32>;            // default: unimplemented!

    fn format_chat(&self, messages: &[(String, String)]) -> String;
    fn special_tokens(&self) -> SpecialTokens;
    fn n_layer(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn n_embd_head(&self) -> usize;
    fn n_kv_embd(&self) -> usize;
    fn n_vocab(&self) -> usize;
    fn rope_style(&self) -> RopeStyle;
}
```

- `forward` is retained as the single-shot convenience entry and both models route it to the graph
  path: `QwenXGraph::forward` clamps `n_ctx` to `max_seq_len`, locks the process-global
  `graph_cache()`, and calls `forward_cached`. The `kv: &mut KVCache` argument is legacy and ignored —
  the KV lives in the graph allocator.
- `build_graph` is the immutable-topology constructor; each architecture implements it in
  `models/<arch>/graph.rs`.
- `forward_graph_cached` is the real primitive used by the CLI, the server, the conversation engine
  and `bench`. Callers that need an isolated KV (a server slot, a draft model) must own their
  `GraphCache`; the process-global one exists only for the single-shot CLI path.
- There is no `weights_version()` method: the value is the `GraphParams.weights_version` field, set to
  a constant `1` today.

Dispatch is `models/mod.rs::load_model_ns`, matching `general.architecture` (`qwen2`, `qwen3`) and
passing a weight-registry namespace. The namespace matters when two models are loaded in one process
(the D5-R draft): the registry is process-global and name-keyed, and a second model without a prefix
would collide and silently drop the primary model to CPU.

| Architecture | Graph module | Highlights |
|---|---|---|
| Qwen2 / Qwen2.5 | `models/qwen2/graph.rs` | biases; `FusedQKV` (concat or CUDA mixed-quant), `FusedFFN` |
| Qwen3 (dense) | `models/qwen3/graph.rs` | no biases; per-head q/k norm (`QkNorm`, `FusedQkvNorm` on Metal), `FusedFFN` |

---

## 11. Runtime Wiring

### 11.1 The reuse kernel

Every path funnels through `QwenXGraph::forward_cached`:

```
params = GraphParams { n_tokens, n_out, gtype, cparams, weights_version: 1 }
if !cache.try_reuse(&params) {
    graph = model.build_graph(&params)
    register_graph_weights(...)          // CPU/Metal/CUDA registration
    alloc.enable_metal() / enable_cuda() // per the same gates recorded in cparams.gpu
    scheduler.assign_backends(&mut graph, alloc)
    FusionPass::run(...)
    alloc.alloc_graph(&graph)
    cache.replace_graph(graph, params)
}
let (graph, alloc) = cache.current().unwrap()
alloc.fill_input_i32(graph, "token_ids", tokens)
alloc.fill_input_i32(graph, "positions", positions)
alloc.fill_input_i32(graph, "tail_ids", tail)   // only when n_out < nt
scheduler.execute(graph, alloc)?
logits = alloc.copy_to_cpu(graph.outputs[0])
```

`cparams.gpu` is `metal_on || cuda_on`, and `fuse_qkv`/`fuse_ffn` are `nt == 1 && gpu` minus their
env opt-outs, so the params carry the same decisions the builder will make.

### 11.2 CLI

The CLI computes `ctx = params.n_ctx.max(prompt_len)` once so prefill and decode size the same KV
regions, prefills with `model.forward(&input_ids, ..., 1, ctx)`, then decodes one token per step with
`model.forward(&[tok], &[pos], ..., 1, ctx)`. The prefill→decode transition changes `n_tokens` and
`gtype`, so it rebuilds once; the KV regions survive because the allocator lives in the cache
(deviation 14). Decode steps then reuse the same graph indefinitely.

### 11.3 Server and conversation

- `server/slot.rs` gives every slot its own `GraphCache` and `n_ctx_total / n_slots` context; a slot's
  cache is reset per request so a request cannot observe another request's KV.
- `server/chat.rs` routes all inference through a `catch_unwind`-guarded
  `forward_graph_cached` call, so a backend panic becomes a 500 instead of a dead worker.
- `conversation.rs` owns a `GraphEngine { model, cache, n_ctx }` for multi-turn sessions; the append-
  only KV is exactly the persistent regions in that cache.

### 11.4 Speculative decoding (D5-R)

Speculative decoding is a graph-reuse consumer built on the same primitive:

- the draft model runs a single-token `forward_graph_cached` chain against its own `GraphCache`;
- the target verifies a draft block of `d` tokens with **one** `forward_graph_cached` call at
  `nt = d + 1`, which builds or reuses a `Decode`-shaped graph for that token count;
- `specverify` and the greedy-identity tests compare the spec output against sequential decode
  byte-for-byte.

Because `n_tokens` changes as the draft length changes, the target graph rebuilds when the draft
depth changes; adaptive depth therefore pins the identity boundary deliberately
(`docs/SPECULATIVE-DECODING-PLAN.md`).

### 11.5 `bench`

`bench` uses a local `GraphCache`: it warms up the prompt forward for a time budget, then repeats
prompt forwards with identical params (the graph is reused; the KV is simply rewritten from position
0), and measures decode by one untimed prefill followed by timed single-token forwards.

---

## 12. Measured Results

Representative numbers, all from the phase ledger (§17) or the referenced documents. They are
environment-dependent (M4 Pro for Metal, GB10 for CUDA) and greedy-decoded unless noted; treat them
as evidence that the design works, not as a benchmark suite.

| Change | Model / path | Result |
|---|---|---|
| Graph vs imperative forward | Qwen2.5-0.5B Q4_0, prefill + decode | logits **max diff 0.000** (bit-identical), KV carried across steps |
| G1–G3 (attention dispatch, `rms_norm_256`, tail rows) | 0.5B decode KV206, Metal | ~122 → ~256 tok/s (2.1×, ≈ the old path) |
| G1–G3 | 0.5B prefill pp440, Metal | ~2530–2620 → ~3900–4000 tok/s (+55%) |
| G1–G3 | 7B Q4_K_M decode KV206 / prefill pp206 | ~32.5 → ~49 tok/s; prefill ~217 tok/s (−10% vs old; attention not the bottleneck) |
| G4 `FusedQKV` | 0.5B decode KV440 / KV550 | ~269 → ~299 tok/s (+11%) / 265 → 295; 7B flat |
| G5 `FusedFFN` | 0.5B decode | ~303 → ~312–331 tok/s (+3%); 7B gate off |
| CUDA Phase 7 (7e②) | 7B Q4_K_M decode | 8.4 → 26.4 tok/s (kernel vectorization) |
| CUDA Phase 8 / R / MMQ (r56, R4) | 7B @2K prefill / decode | ~3212 tok/s (≈1.035× llama.cpp) / ~43–45 tok/s — details in `docs/CUDA_OPTIMIZATION.md` |

Test surface: **83 `#[test]`** across `src/graph/*.rs` (CUDA backend 40, Metal 14, CPU 6, allocator 5,
cache 4, mod 4, builder/scheduler 3 each, fusion 2, dot/json 1 each; `ops.rs`, `params.rs`,
`backend.rs` have none directly), plus 2 in `vec_ops` for the CPU SwiGLU kernel. The model graph
modules add their own end-to-end tests. Not all 83 compile in a single configuration because the
Metal and CUDA backends are platform/feature-gated.

---

## 13. Module Inventory

The plan's file-change manifest, replaced by the landed inventory.

| File | Lines | Role |
|---|---:|---|
| `src/graph/mod.rs` | 273 | IR types, `topo_order`, `capture_nt_hint` |
| `src/graph/ops.rs` | 319 | `Op`, `NodeMeta`, metadata structs |
| `src/graph/builder.rs` | 496 | `GraphBuilder` |
| `src/graph/params.rs` | 71 | `GraphType`, `CParams`, `GraphParams` |
| `src/graph/cache.rs` | 235 | `GraphCache`, `uid`, structural check |
| `src/graph/backend.rs` | 95 | `Backend`, `KvProvider` |
| `src/graph/alloc.rs` | 795 | liveness allocator, KV regions, staging |
| `src/graph/scheduler.rs` | 519 | assign / split / execute, capture hook |
| `src/graph/fusion.rs` | 151 | `FusionPass` |
| `src/graph/cpu_backend.rs` | 916 | CPU executor |
| `src/graph/metal_backend.rs` | 2081 | Metal executor |
| `src/graph/cuda_backend.rs` | 6662 | CUDA executor + graph capture |
| `src/graph/dot.rs` | 80 | DOT export |
| `src/graph/json.rs` | 357 | JSON export, `preview_fuse_flags` |
| `src/models/qwen2/graph.rs` | 2253 | Qwen2 build + weights + tests |
| `src/models/qwen3/graph.rs` | 1132 | Qwen3 build + weights + tests |

`src/models/qwen2/forward.rs` was **deleted** in Phase 6; the imperative path no longer exists.
`src/metal.rs` and `src/cuda.rs` remain the per-op kernel/device layers that the graph backends wrap;
`src/cuda_kernels.cu` holds the CUDA kernels. `src/cache.rs` keeps the legacy `KVCache` type, which the
graph path ignores (the allocator owns KV).

---

## 14. Implementation Timeline

The plan's build order, with landed status:

| Phase | Content | Status |
|---|---|---|
| 1 | IR + buffer infrastructure: `mod.rs`, `ops.rs`, `builder.rs`, `alloc.rs` | ✅ |
| 2 | CPU backend: `backend.rs` + `cpu_backend.rs` (+ minimal executor) | ✅ |
| 3 | Metal fine-grained backend: `metal_backend.rs` + cross-backend scheduling | ✅ |
| 4 | Scheduling + fusion + debugging: `scheduler.rs`, `fusion.rs`, `dot.rs`, `cache.rs`, `params.rs` | ✅ |
| 5 | Qwen2 graph construction: `models/qwen2/graph.rs` | ✅ |
| 6 | Wiring + cleanup: `ModelDef` routes to the graph path, `forward.rs` deleted | ✅ |
| 7 | CUDA backend wrapping `cuda.rs`, preserving CUDA Graph | ✅ |
| 8 | Verification: old/new logits, 7B GPU, `--dump-graph` | ✅ |
| 9 | Metal wiring optimization G1/G2/G3 + allocator liveness fixes | ✅ |
| 10 | G4 decode QKV fusion (`Op::FusedQKV`) | ✅ |
| 11 | G5 decode FFN fusion (`Op::FusedFFN`) | ✅ |
| 12+ | Post-plan workstreams (Qwen3, CUDA phases, viz, spec decode, bench) — see §17.2 | ✅ |

---

## 15. Design Decisions, Invariants and Risks

### 15.1 Standing decisions

These were deviations from the original plan that are now deliberate design:

1. **`NodeMeta` is a concrete enum**, not `Box<dyn Any + Send + Sync>`: `PartialEq` for the structural
   check, no downcast panic, `CNode: Clone`. The cost is that a new metadata kind edits the enum.
2. **`Op::Input` is a node kind**, so inputs are visible in the IR and in `ComputeGraph::inputs`.
3. **The allocator is the single owner of every buffer.** Backends own pools; the scheduler is a pure
   orchestrator (`assign_backends` queries allocator capabilities, execution goes through
   `alloc.cpu_mut()` / `metal_mut()` / `cuda_mut()`).
4. **In-place ops alias their input** when it is their sole consumer on the same backend. This is what
   makes kernel-order execution correct without host copies.
5. **Execution follows build order**, and so does liveness.
6. **Inputs are never freed by liveness**, exactly like outputs.
7. **Fusion flags are part of the reuse identity.** `CParams` carries `gpu`, `fuse_qkv`, `fuse_ffn`;
   toggling any of them forces a rebuild, which is what makes the env A/Bs valid.
8. **Fused decode nodes are built, not pattern-matched.** The builder knows the layer, the biases and
   the concat weights; the peephole pass only handles the backend-generic SwiGLU rewrite.

### 15.2 Risks and status

| Risk | Status |
|---|---|
| KV position coupling (the original plan's fatal flaw) | Resolved: positions are input data; no topology depends on `n_past` |
| Reuse decision too loose / missing payloads | Resolved: params-only comparison plus a debug structural check; fusion flags pinned by a test |
| Per-node host round trips across backends | Resolved: buffer-id interface; cross-backend copies only at split boundaries |
| Lost `n_out` optimization | Resolved: G3 `GetRows` tail reduction; `n_out` is in the reuse identity |
| Metal per-op dispatch overhead | Resolved: one command buffer per split; G1 fast-path dispatch is bitwise-neutral and ~2.1× on 0.5B decode |
| Cross-backend KV copy | Resolved: KV regions are created on, and resolved from, the layer's backend |
| Double fusion | Resolved: `supports_fused` gates the pass; build-time fused ops bypass it entirely |
| GPU safety rules broken | Resolved: `execute_node` returns `Err`; `docs/GPU_SAFETY.md` rules are enforced in the guards |
| CUDA functionality regression | Resolved: CUDA Graph capture/replay preserved and default-on for decode |
| `BatchMatMul` fusion | **Open by design**: deferred (single-output IR); recorded in §5.4 |
| `FusedBiasRope` rule | Removed: no backend ever claimed the capability; the op, rule and tag are gone (§5.2) |
| Dump/trace fusion gate differs from runtime on CUDA | Resolved: shared `preview_fuse_flags` gate + test (§8.4) |

---

## 16. References

### Internal

| Topic | Where |
|---|---|
| IR / builder / scheduler / allocator walkthrough | `docs/inference_e2e_walkthrough/05-graph-builder-ir.md` … `08-scheduler-execute.md` |
| Architecture overview, adding an architecture | `docs/ARCHITECTURE.md` |
| Backend overview (CPU/Metal/CUDA, feature gates) | `docs/BACKENDS.md` |
| CUDA backend design + Phase 7 record | `docs/CUDA-BACKEND-DESIGN.md` |
| CUDA optimization history + env-gate reference | `docs/CUDA_OPTIMIZATION.md`, `docs/cuda_optimization_steps/` |
| Metal optimization plans | `docs/METAL_OPTIMIZATIONS.md` |
| CPU optimization record | `docs/CPU_OPTIMIZATIONS.md` |
| GPU safety rules | `docs/GPU_SAFETY.md` |
| Speculative decoding (D5) | `docs/SPECULATIVE-DECODING-PLAN.md` |
| Graph visualizer (JSON, trace, live SSE) | `viz/README.md` |
| Debug dumps | `docs/debug-dump.md` |
| Qwen2 / Qwen3 support | `docs/QWEN2-SUPPORT.md`, `docs/QWEN3-SUPPORT-PLAN.md` |

### llama.cpp analogues

| Concept | Reference |
|---|---|
| Graph IR | `ggml/src/ggml-impl.h` — `ggml_cgraph` |
| Reuse decision (params only, no `n_past`) | `src/llama-graph.h` — `llm_graph_params::allow_reuse` |
| Graph construction context | `src/llama-graph.h` — `llm_graph_context` |
| Result container / reuse | `src/llama-graph.h` — `llm_graph_result`, `can_reuse()` |
| Backend split graph | `ggml/src/ggml-backend.cpp` — `split_graph()` |
| Per-split execution | `ggml/src/ggml-backend.cpp` — `compute_splits` |
| Tail rows | `src/models/llama.cpp` — `ggml_get_rows(cur, inp_out_ids)` |
| In-place rope/silu, swiglu split | `ggml` op semantics; minfer's `*_off` kernels mirror them |

---

## 17. Implementation Record

### 17.1 Phase ledger

The G-phase rows below are the plan-era record. Their original commit hashes no longer resolve
because the repository history was rewritten; the resolvable documentation commits are
`8d7cb38` (G1–G3), `96404fb` (G4), `ec922f1` (G5). Post-plan work is in §17.2.

| Phase | Content | Status | Evidence |
|---|---|---|---|
| 1 | IR infrastructure (`mod.rs`, `ops.rs`, `builder.rs`, `alloc.rs`) | ✅ | unit tests: topo sort / cycle detection, op payload equality, chained liveness reuse, parallel-chain isolation, input fill, persistent KV sharing |
| 2 | CPU backend (`backend.rs`, `cpu_backend.rs`, minimal executor) | ✅ | graph vs manual computation (matmul+bias+silu+scale), rms_norm, embedding+rope, KV store/load + GQA attention |
| 3 | Metal backend + cross-backend scheduling | ✅ | 15 tests: element-wise / norm / Q4_0+Q8_0 matmul vs reference, cross-backend copies, multi-split alternation, KV + decode, layer-0 full-Metal vs CPU bit-identical; CLI GPU output fluent |
| 4 | Scheduler + fusion + diagnostics (`scheduler.rs`, `fusion.rs`, `dot.rs`, `cache.rs`, `params.rs`) | ✅ | fusion apply/gate, DOT format, cache reuse semantics, split boundaries |
| 5 | Qwen2 graph construction | ✅ | real model logits: graph vs imperative **max diff 0.000** (prefill + decode, KV across steps) |
| 6 | Wiring + cleanup: `forward.rs` deleted, `ModelDef` on the graph path, `--graph` a compatibility no-op | ✅ | full suite green after deletion; CLI output consistent |
| 7 | CUDA backend wrapping `cuda.rs` (7a–7e: skeleton, per-op dispatch, graph integration + staging, CUDA Graph, kernel/path optimizations) | ✅ | GB10: 7B Q4_K_M decode 8.4 → 26.4 tok/s; test suites pass; see `docs/CUDA-BACKEND-DESIGN.md` |
| 8 | Verification: old/new logits, 7B GPU, `--dump-graph` export, graph path default | ✅ | 0.5B Q4_0 bit-identical; 7B Q4_K_M GPU fluent (~42 tok/s at the time); 437-node DOT |
| 9 (G1–G3) | Metal attention dispatch, `rms_norm_256`, `n_out` tail rows; two allocator liveness fixes | ✅ | 0.5B decode 2.1×; prefill +55%; 7B decode ≈ old; tail-reduction per-node comparison 0.000 |
| 10 (G4) | Decode QKV fusion `Op::FusedQKV` (concat matmul + bias/rope/store), flags in `CParams`, env revert | ✅ | 0.5B decode +11%; fused vs unfused logits 0.000 (0.5B + 7B); `fused_qkv_matches_unfused_decode` |
| 11 (G5) | Decode FFN fusion `Op::FusedFFN` (gate+up concat + in-place swiglu), `nf <= 16384` gate | ✅ | 0.5B decode +3%; 7B unchanged (gate off); fused vs unfused logits 0.000 |

### 17.2 Post-plan workstreams

After G5 (2026-08-22) the graph path gained 12 workstreams (101 commits through HEAD `5471680`):

| Workstream | Graph impact | Representative commits |
|---|---|---|
| Chat API / server / conversation plumbing | explicit `n_ctx`, per-slot `GraphCache`, conversation engine cache | `f8f1124`, `55296f2`, `5866d36`, `99f2188`, `98f05bb`, `921bb4c`, `b6c9473` |
| Qwen3 dense on the graph path | second architecture: `QkNorm`, `FusedQkvNorm`, Metal fixes; KV sized by `--n-ctx` | `283c7d6`, `94d57ac`, `d5b8023`, `6756d42` |
| Viz live + CLI subcommands | scheduler capture/live events, JSON export, `viz`/`serve` subcommands | `eac8180`, `c95df6a`, `163cb6c`, `5b22353`, `2a70e35`, `071b5f6`, `ed5fc9b` |
| Metal objc2 migration | backend internals re-homed; no topology change | `6a382a3`, `be3df55`, `ee9b65b`, `94d57ac` |
| CUDA Phase 7a–7e | `cuda_backend.rs`, per-op dispatch, staging, CUDA Graph, FusedFFN port, async H2D | `dfa3516`, `ad1512d`, `8fb88f8`, `4fcd0d8`, `7123adb`, `92ad586`, `8d2ee6f`, `78d410a`, `f54f721`, `082095c` |
| CPU graph correctness | `tail_ids`, buffer-pool race, cross-buffer handling, token-major F32 matmul, fusion flags in the reuse identity | `2ed3eb1`, `b849601`, `058ea97` |
| CUDA Phase 8 campaign (8b–8p) | f16 KV, prefill GEMMs, split-K decode attention, wmma/FA prefill, MMVQ; **graph-side 8o** removes ~1.6 s per-rebuild CPU stalls | `f7b0036`, `69a27c5`, `a5af60f`, `b959ec9`, `eb24054`, `b7b8e73`, `1298cb2`, `1d28235`, `acca28f`, `ba3f317`, `cb66fca`, `65b686c`, `2992f57` |
| CUDA R1–R4 + capture defaults | int8 MMQ, MMVQ weight streaming, pinned D2H, prefill capture default-on, R3-A1 `tail_ids` at the graph head, decode split attention | `40e97c9`, `6df3245`, `761e236`, `a213c89`, `70f57db`, `86ca78c`, `4d8c666`, `f45c241` |
| MMQ campaign r34–r60 | no topology change; weight-plane precompute + producer-fused A-quantize; gates default-on | `87a75a3`, `cf1ed4b`, `910d967`, `83fee77`, `4cf7c74`, `36a481f`, `57edcf6` |
| D3-x CUDA decode | decode attention dispatch, D3-5/D3-7, **G4 `FusedQKV` ported to CUDA** (class 1 concat + class 2 mixed-quant epilogue) | `22336b2`, `3230b2b`, `92b0712`, `b3b6077`, `3857633`, `a448a4a`, `9f419f9` |
| `bench` subcommand | harness around `forward_graph_cached` (pp/tg, warmup budget) | `646f22e`, `cdc1e21`, `62b9b90` |
| Speculative decoding D5 → D5-R | draft nt=1 chain + target verify at nt=d+1; `specverify`; adaptive depth; conversation/server integration; draft weight namespacing | `3fd0880`, `a6b7cf3`, `90ef8ea`, `8d7ce50`, `c3d4bb1`, `0fe132f`, `39eceaa`, `c26c114` |

### 17.3 Deviations from the plan

Items 1–26 are the plan-era deviations (kept, lightly updated). Items 27+ are post-plan changes that
a reader of the original plan would not find there.

1. **`NodeMeta` is a concrete enum**, not `Box<dyn Any + Send + Sync>`.
2. **`Op::Input` leaf node**; `ComputeGraph::inputs` records the ids.
3. **`rope()` carries `style`**, and **`kvcache_store`/`load` carry `n_ctx`** so the persistent region
   has a fixed length; only the written prefix is read at execution time.
4. **(Superseded by 20)** The first KV layout was one contiguous `[K | V]` block per layer.
5. **`attn()` takes a `pos` input**: the causal mask needs `pos[t] + 1`, matching llama.cpp's
   idx-dependent mask.
6. **Metadata expansion** beyond the plan's sketch: `EmbedMeta.weight_name`,
   `RoPEMeta.n_head/hd`, `AttnMeta.nkt/scale/layer`, plus the four fused metas.
7. **`Backend::execute_node` takes `&mut self`** (the pool is mutated), and the **allocator is the
   single owner of the pool** (deviation 11).
8. **I32 inputs are stored as `f32::from_bits` bit patterns** (exact for `|v| < 2^24`), filled through
   `fill_input_i32`.
9. **The CPU backend supports F32-weight matmul** through `vec_ops::mat_mul_f32`; the quantized path
   quantizes activations to Q8_0 on the fly.
10. **`BatchMatMul` fusion deferred**: the single-output IR cannot express a multi-output node.
11. **Allocator single ownership (final form)**: the scheduler is a pure orchestrator; execution goes
    through `alloc.cpu_mut()`/`metal_mut()`/`cuda_mut()`.
12. **`GraphParams` is the entire reuse basis**; `n_past` is explicitly absent; `weights_version` is
    reserved (constant 1 today).
13. **Weight layout is the GGUF layout**: metadata `[in, out]`, memory `[out][in]` row-major, so
    `od = shape[1]`, `id = shape[0]`.
14. **The KV persistent region survives graph rebuilds**: `GraphCache` owns the allocator, so a
    prefill→decode rebuild swaps only the graph.
15. **The scheduler executes in build order**, which guarantees a KV store precedes the attention
    that reads it; fusion-orphaned nodes are skipped.
16. **G3 tail rows implemented**: `GetRows(wo/residual, tail_ids)` after the last layer's `wo`;
    `forward.rs`'s old "compute all, slice the tail" no longer exists.
17. **`forward.rs` deleted in Phase 6**; `embed_tokens` moved to `kernel.rs`; `--graph` is a
    compatibility no-op.
18. **In-place op buffer aliasing** (the key Phase-3 fix): `Silu`/`RoPE` alias their input when it is
    the sole consumer on the same backend; this fixed the all-zero-KV corruption.
19. **Backend configuration is part of the reuse identity** (`CParams.gpu`): MPS/CUDA initialization
    changes force a rebuild.
20. **Each layer's KV is two independent persistent regions** (`kv.{layer}.k`, `kv.{layer}.v`),
    exposed through `kv_pair(layer)`; the plan's contiguous `[K | V]` was replaced.
21. **Metal G1/G2 wiring**: attention dispatches by `nt`/`hd` to flash/split/parallel/classic with the
    same gating and env vars as the old path; `RmsNorm` selects `rms_norm_256` when enabled.
22. **Liveness must match the scheduler's execution order (G3 fix)**: using `topo_order()` for liveness
    let a later node reuse an input the scheduler had not read yet (logits off by 21.79). Liveness now
    uses build order; `topo_order()` only validates acyclicity.
23. **Input buffers are never freed (G3 fix)**: inputs are host-filled before execution, so two inputs
    sharing a liveness block let the later fill clobber the earlier one.
24. **G1–G3 measured** (M4 Pro, greedy): 0.5B decode KV206 ~122 → ~256 tok/s; prefill pp440 ~3900–4000
    tok/s (+55%); 7B decode ~49 tok/s; 7B prefill ~217 tok/s (−10%). Greedy output per token identical
    to pre-G1.
25. **`FusedQKV` (G4)**: decode builds one node (concat matmul + `attn_bias_rope_store`); `attn()`
    takes its output shape from `AttnMeta`, not from the (larger) fused q buffer; `CParams.fuse_qkv`
    is in the reuse identity; CPU never builds it.
26. **`FusedFFN` (G5)**: decode builds concat gate+up matmul + in-place offset swiglu; the down matmul
    reads rows `0..nf`; **gated `nf <= 16384`** because the 7B concat measured slower. Debugging
    lesson: an unfused comparison must run `FusionPass`, or two-kernel silu+mul vs one swiglu kernel
    differ by ~1e-6.

Post-plan additions:

27. **Qwen3 on the graph path**: `Op::QkNorm` (per-head RMSNorm) and `Op::FusedQkvNorm` (concat
    matmul + qk-norm + no-bias rope/store). The fused variant is Metal-only; CUDA Qwen3 runs the
    unfused qk_norm chain.
28. **D3-8 class 2 (`Op::QkvBiasRopeStore`)**: mixed-quant decode layers (e.g. Q6_K `attn_v` among
    Q4_K q/k) run three separate matmuls plus one bias/rope/store epilogue, CUDA-only. On Metal those
    layers keep the unfused chain, bitwise-neutral.
29. **G4 `FusedQKV` ported to CUDA** (class 1 concat) — `Op::FusedQKV` is no longer Metal-only.
30. **CUDA Graph replay**: `Backend::graph_replay` + `ComputeGraph::capture_nt_hint`, keyed by
    `(uid, node range)`, invalidated by `pool_gen`; two direct-launch warmups, capture on the third;
    prefill capture default-on. The scheduler disables replay under trace/live capture.
31. **`alloc_fresh`** added to the `Backend` trait for split-boundary staging that must not be
    recycled mid-execute.
32. **R3-A1**: `tail_ids` is declared at the graph head, not beside its consumer, to avoid extra
    CPU/CUDA split boundaries. **R3-A2**: the logits buffer is exactly `n_out * n_vocab`, so the
    per-step full-logits clone was dropped.
33. **`weights_version` is static `1`**: there is no `weights_version()` trait method and no LoRA path
    yet; the counter exists but is unused.
34. **Export/debug surface**: `--dump-graph`, `--dump-graph-json`, `MINFER_TRACE` (P2) and the viz
    live path (P3) are graph consumers added after the plan.
35. **Runtime consumers of `forward_graph_cached`**: the server's per-slot caches, the conversation
    engine and the D5-R draft/verify loops.
36. **Export/trace fusion gate (fixed)**: the export and trace paths used to gate `fuse_qkv` on Metal
    only while the Qwen2 runtime used `metal_on || cuda_on`, so CUDA previews lacked `FusedQKV` and
    node ids diverged from live events. Both call sites now share
    `graph::json::preview_fuse_flags`, pinned by a unit test (§8.4).

Post-baseline additions (working-tree changes made while writing this document, after `HEAD`
`5471680`):

37. **CPU single-pass SwiGLU**: `vec_ops::vec_swiglu_f32` replaces the `vec_silu_f32` +
    `vec_mul_f32` + full-size temporary pair in `CpuBackend::execute_node`. Bit-identical to the
    pair (same formula and per-element order); pinned by `vec_ops::tests::swiglu_matches_silu_then_mul`.
38. **Dormant `FusedBiasRope` removed**: `Op::FusedBiasRope`, `FusionPass::fuse_bias_rope`, the
    `FusedOp::BiasRope` tag and the never-probed `FusedOp::BatchMatMul`/`FusedOp::QKVBiasRopeStore`
    tags are gone; `FusedOp` now has a single variant. Metal's `supports_fused` no longer advertises
    a tag whose op it cannot execute (§5.2/§5.4).
39. **Shared preview gate**: `graph::json::preview_fuse_flags(nt, metal_on, cuda_on)` is the single
    source for the `--dump-graph*` / `MINFER_TRACE` preview's fusion flags (item 36).

### 17.4 Test surface and baseline

83 inline `#[test]` functions live in `src/graph/*.rs` (distribution in §12), plus 2 in `vec_ops` for
the CPU SwiGLU kernel. The CUDA backend
dominates because capture/replay needs bit-parity tests
(`cuda_graph_replay_bit_parity`, `cuda_prefill_shaped_graph_never_captures`,
`cuda_multisplit_capture_bit_parity`, `cuda_graph_recaptures_on_pool_gen_change`, and the prefill
capture parity harness). The model graph modules add end-to-end tests that compare graph and
reference logits.

Two plan-era baseline notes are now historical: the `attn_parallel_realdata_correctness` test needs a
real dump under `/tmp/dp3` (environment-dependent, a pre-existing failure), and the "46 pass / 1 fail"
figure dates from Phase 1 — later rows of the ledger record larger suites (80/81 for G3–G5, 144+130
for the CUDA phases). Run `cargo test --release` for the current number; a bare
`cargo test --release` does not compile the CUDA backend unless `--features cuda` is passed.

