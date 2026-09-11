# 05 · Graph build — the IR and the builder

> **Stage**: tokenizer + template (04) → **graph build (IR)** → backend
> assignment + fusion (06). By this point the prompt is a list of integer
> token ids and every weight is registered by name in the allocator — but
> **nothing has been computed yet**. This stage writes the entire transformer
> forward pass as one pure data structure: a declarative compute graph that
> later stages assign to hardware, fuse, allocate, and execute.
> **Code**: `src/graph/mod.rs` (`CNode` :84, `ComputeGraph` :107,
> `topo_order` :154), `src/graph/ops.rs` (`Op` :40, `NodeMeta` :166),
> `src/graph/builder.rs` (`GraphBuilder` :15, method family :58–421),
> `src/models/qwen2/graph.rs` (`Qwen2Graph::build` :38, `forward_cached`
> :403), `src/graph/params.rs` (`GraphParams` :52), `src/graph/cache.rs`
> (`try_reuse` :47) — lines verified at commit `e7fa0da`.

## 1. Background — where this stage sits

After doc 04 the engine holds two things: a `Vec<u32>` of token ids — one
integer per piece of your prompt, as produced by the byte-pair-encoding
tokenizer — and a loaded model whose weight tensors (the quantized matrices
from the GGUF file) are registered by name in the graph allocator. A forward
pass must now turn those ids into **logits**: one floating-point score per
vocabulary entry, from which the sampler will pick the next token. That takes
hundreds of math operations — normalizations, matrix multiplies, rotations,
attention — arranged in a very specific order, repeated for each of the
model's transformer layers.

The engine does **not** run those operations as a hand-written loop, at least
not any more. Instead it first *describes* the whole computation as a data
structure called a **compute graph** — a directed acyclic graph where each
vertex ("node") is one math operation and each edge means "this node's output
is that node's input". The adjective *declarative* is the point: building the
graph states **what** should be computed, never **how** or **when**. The
"how" (which CPU instructions, which GPU kernel) is decided later, per node,
by the backend scheduler (doc 06). The "when" is the node order itself
(doc 08).

Why go through this indirection? Four concrete payoffs, each of which gets a
full section in §3.3:

1. **Reuse.** Decoding is a loop: one forward pass per generated token, often
   hundreds of them. If the graph is a pure function of a small parameter
   struct (`GraphParams`), identical parameters mean an identical graph — so
   the engine builds it once and replays it for every decode step, refreshing
   only the input *data* (the new token id, the new position).
2. **Global decisions.** "Which backend runs this node?" and "which buffers
   can share memory?" need the *complete* picture of the computation before
   anything runs — and a data structure you can walk is exactly that picture.
3. **Optimization as rewriting.** Pattern-optimizations (fusing `silu∘mul`
   into one operation) become a small graph-rewrite pass instead of scattered
   special cases inside a forward loop.
4. **Observability.** The graph can be printed (`--dump-graph` exports
   Graphviz DOT; `--dump-graph-json` exports JSON for the web visualizer), so
   you can literally *see* one forward pass.

There was an older design: `models/qwen2/forward.rs`, an imperative per-layer
loop that computed as it went and hard-coded its GPU fallbacks. It is gone —
deleted in Phase 6 of the graph refactor (commit `6af12a4`) only after the
graph path reproduced its logits bit-identically, prefill and decode
(`docs/GRAPH-REFACTOR-PLAN.md` §17, Phases 5–6 and 8). Its shape survives as
a historical record in `docs/ARCHITECTURE.md` Appendix A.

So this document is the heart of the engine: the transformer forward pass,
written as a data structure. Doc 06 walks it and gives every node a backend;
doc 07 gives every node a buffer; doc 08 finally executes it.

## 2. Principle — how it works and why

### 2.1 The vocabulary: graph, node, edge, IR

Some terms we will use constantly:

- **Compute graph** — the data structure describing one forward pass. Each
  node is one operation ("multiply these two matrices"), each **edge** is a
  data dependency ("node 13's output feeds node 14"). No cycles are allowed:
  information flows one way, input to output.
- **IR** — *intermediate representation*, a compiler word for "a program
  expressed as data, sitting between the source and the machine". minfer's IR
  is `ComputeGraph`: a Rust struct you can traverse, compare, rewrite, and
  export. llama.cpp has the same idea (`ggml_cgraph`); minfer mirrors it.
- **Node** (`CNode`) — one operation plus everything the rest of the engine
  needs to know about it: which operation (an `Op` enum value with **full
  parameters**), which other nodes it reads (`src`, the edges), the shape and
  element type of its output, and (only later, doc 06) which backend it runs
  on.
- **Topological order** — an ordering of nodes where every node appears after
  all of its inputs. The builder produces this *by construction*: it only
  lets you reference nodes that already exist. "Append sources before
  consumers" is one sentence, and it buys an enormous amount: no sort is ever
  needed, and executing nodes in list order (doc 08) is automatically
  correct.
- **Backend** — one compute device and its kernels: `CPU` (AVX2/NEON SIMD),
  `Metal` (Apple GPU), `Cuda` (NVIDIA GPU). Each node gets *assigned* one,
  but that happens after this stage.
- **Buffer** — a chunk of memory holding one node's output; a `BufRef` is
  just a handle (which backend's pool, which slot). Allocated in doc 07; the
  graph only records shapes.

### 2.2 The three data structures

The whole IR lives in `src/graph/mod.rs` (273 lines including tests). You
have met the `CNode` fields in §2.1; what `ComputeGraph` itself adds is the
node list plus two id sets — `inputs` (the leaves filled with external data
each step) and `outputs` (the logits) — and a `uid`, a monotonic id assigned
when the graph is cached, which the CUDA backend keys its replay cache on.
A `BufRef { backend, id }` is just a handle for "buffer `id` in that
backend's pool" — created only in doc 07.

### 2.3 Operations carry full payloads

The `Op` enum is not just a tag like "MatMul". Every variant carries the
parameters that make it *this* operation and no other: `RmsNorm { eps }`,
`MatMul { transpose_b }`, `RoPE { style }`, `KvcacheStore { layer }`,
`FusedQKV { layer }`. Node-level extras (which weight tensor, which bias,
frequency base) ride along in a `NodeMeta` enum. Two consequences:

- Two graphs can be compared **structurally** — node by node, payload by
  payload. `Op` and `NodeMeta` both derive `PartialEq` for exactly this, and
  a debug/test check (`GraphCache::verify_structural`) asserts that rebuilding
  a graph with identical parameters yields an identical node sequence — the
  tripwire guarding the reuse invariant of §2.7.
- The dumped graph is self-describing: the DOT export prints
  `RmsNorm { eps: 1e-6 }` and `KvcacheStore { layer: 23 }`, so you can read a
  forward pass off the page.

The full op list is wider than what Qwen2 emits (`Softmax`, `View`,
`Permute`, `Scale`… exist for ggml parity), but the live graph uses a compact
set — see the census in §2.5.

### 2.4 The builder: an append-only factory

`GraphBuilder` (496 lines) is the only way to construct a graph. Its methods
come in two kinds:

- `input(name, shape, dtype)` — declare a leaf node whose data arrives from
  outside each step. It records the id in `graph.inputs` and nothing else.
- The operation family — `embedding`, `rms_norm`, `qk_norm` (Qwen3),
  `matmul`, `get_rows`, `rope`, `silu`, `add`, `mul`, `swiglu`, `softmax`,
  `attn`, `kvcache_store`/`kvcache_load`, plus the decode-fusion constructors
  `fused_qkv` / `qkv_bias_rope_store` / `fused_ffn` / `fused_qkv_norm`. Each
  one computes the output shape from its inputs' shapes (so shapes propagate
  through the graph automatically), wraps the parameters into an `Op` +
  `NodeMeta`, appends the node, and returns its id.

The crucial property is stated in the module's first lines: *"The builder is
pure: it only appends nodes to the graph, it never computes."* Building a
graph allocates a few `Vec`s and does integer shape arithmetic — no matmul
runs, no GPU is touched, no buffer exists yet. Purity is what makes the reuse
invariant (§2.7) even statable.

One naming convention to keep straight, because it recurs in every excerpt:
GGUF stores weight metadata as `[in, out]` (input dim first) but memory
layout is `[out][in]` row-major, and activations are token-major `[nt][d]`
(`nt` = token count, `d` = features) — so `matmul`'s output shape is
`[w.shape[1], nt]`: output dim, then token count.

### 2.5 The main event: one forward pass, node by node

`Qwen2Graph::build` (`src/models/qwen2/graph.rs:38`) is where the forward
pass is actually written down. The real thing below is from the Qwen2.5-0.5B
model (24 layers, hidden 896, 14 query heads of dim 64, 2 KV heads — more on
that below, FFN width 4864, vocabulary 151936), prefilled with a 30-token
prompt. Node ids are the real ones from a `--dump-graph` export.

```
token_ids(0)  positions(1)  tail_ids(2)      <- input leaves (filled per step)
      \          |              \______ (consumed at the LAST layer only)
       v        v
    embed(3) GetRows             h = one token_embd row per id, [896, nt]
       |
  == layer 0 ==================================================================
   rms_norm(4) <- h                                  "attn_norm", eps 1e-6
     |-- matmul(5) blk.0.attn_q     [896 -> 896]   Q = Wq @ normed (+ bias)
     |-- matmul(6) blk.0.attn_k     [896 -> 128]   K = Wk @ normed (+ bias)
     +-- matmul(7) blk.0.attn_v     [896 -> 128]   V = Wv @ normed (+ bias)
   rope(8)  <- n5, positions                        rotate Q by position
   rope(9)  <- n6, positions                        rotate K by position
   kv_store.0(10) <- n9(K), n7(V), positions        write into layer-0 KV region
   kv_load.0(11)                                    view of that region (no edges in)
   attn(12) <- n8(Q), n11(KV), positions            GQA attention, scale 1/sqrt(64)
   matmul(13) blk.0.attn_output    [896 -> 896]     project heads back
   add(14) <- embed(3), n13                          <-- FIRST residual add
   rms_norm(15)                                      "ffn_norm"
   matmul(16) blk.0.ffn_gate      [896 -> 4864]
   matmul(17) blk.0.ffn_up        [896 -> 4864]
   silu(18)  <- n16                                  (orphaned by fusion, see §2.6)
   SwiGLU(19) <- n16, n17                            silu(gate) * up (fused form)
   matmul(20) blk.0.ffn_down      [4864 -> 896]
   add(21) <- n14, n20                               <-- SECOND residual add
  == layers 1..22 identical (18 nodes each) ==================================
  == layer 23 (the last): as above, plus two GetRows nodes (428, 429) =========
   rms_norm(438)                                     final norm
   matmul(439) output (lm_head)   [896 -> 151936] -> OUTPUT logits
```

Reading it top to bottom is reading the forward pass. What a beginner should
take away is the *shape* of a transformer layer, which the IR makes
unmistakable:

- **An attention block**: normalize, then three matrix multiplies produce
  Q (query), K (key), V (value) — attention's three roles; the positions are
  mixed in by RoPE (Rotary Position Embedding, which rotates Q and K so that
  attention scores depend on token distance); the fresh K and V are written
  into the layer's **KV cache** region (persistent memory holding every past
  token's keys and values — the thing that makes decoding cheap, doc 11);
  then one attention operation reads Q plus the whole KV region, and one
  final matrix multiply projects the result back to the hidden width.
- **An FFN block** (feed-forward network): normalize, two wide matrix
  multiplies (gate, up), the SwiGLU activation — `silu(gate) * up`, where
  SiLU is the smooth gate function `x·sigmoid(x)` — then a narrow matrix
  multiply back down.
- **Two residual adds** — `h = h + attention(...)` and `h = h + ffn(...)`.
  Each block computes a *correction* to the running hidden state instead of
  replacing it; the add nodes are the only places the hidden state is
  rewritten.

Nodes 4–21 are layer 0 (0–2 are the input leaves, 3 the embedding); layers 1
through 22 repeat the same 18-node pattern (layer 23, with the two extra
`get_rows` of §2.8, spans nodes 418–437); node 438 is the final norm and 439
the lm_head. The full prefill graph is **440 nodes** (measured: `--dump-graph`
on 0.5B, 30-token prompt, n_out = 1). The complete census:

| Op kind | Count | Where |
|---|---|---|
| `MatMul` | 169 | 7 per layer (q,k,v,wo,gate,up,down) × 24 + lm_head |
| `RmsNorm` | 49 | 2 per layer × 24 + 1 final |
| `RoPE` | 48 | 2 per layer (q, k) |
| `Add` | 48 | 2 residual adds per layer |
| `SwiGLU` | 24 | 1 per layer (rewritten from silu+mul) |
| `Silu` | 24 | orphans left behind by that rewrite |
| `KvcacheStore` / `KvcacheLoad` | 24 / 24 | 1 each per layer |
| `Attn` | 24 | 1 per layer |
| `GetRows` | 3 | embed + 2 tail-row selects |
| `Input` | 3 | token_ids, positions, tail_ids |

A note on **GQA** — grouped-query attention, the Qwen2 trick visible in the
shapes above. Full multi-head attention stores a key and a value per query
head: with 14 heads of dim 64, that is 14 × 64 = 896 numbers per token per
layer, per K *and* V. Qwen2.5-0.5B instead shares each key/value among a
group of 7 query heads: only `n_head_kv = 2` KV heads exist, so K and V are
128-wide (`n_kv_embd = 128`), and the KV cache shrinks 7×. The graph
expresses this in plain shapes — the wk/wv matmuls output 128, not 896 — and
the `attn` node's metadata carries the mapping (`n_head`, `n_head_kv`, head
dims, the KV row stride `nkt`, the scale `1/sqrt(64)`).

### 2.6 Prefill vs decode: same skeleton, two fusion classes

A forward pass is built for exactly one `n_tokens`:

- **Prefill** (`nt > 1`): all prompt tokens flow through together; the KV
  store writes a block of positions and attention reads the fresh prefix.
- **Decode** (`nt == 1`): one token at a time — same 18-node skeleton, the
  shapes just narrow to `[.., 1]`.

Decode is where minfer buys speed with two *build-time* fusions — extra
builder constructors that emit fewer, bigger nodes. They are decided inside
`build` itself (gated on `nt == 1` and a GPU backend), not by a later pass:

- **`FusedQKV`** (the concat class): instead of `3 matmul + 3 bias-add +
  2 rope + 2 store` = **10 kernel dispatches** per layer, one concat matmul
  against a pre-concatenated `blk.{i}.attn_qkv` weight, then one fused kernel
  applying the three biases, roping Q and K, and storing K/V — **2
  dispatches**. Attention reads Q from offset 0 of the concat buffer.
- **`FusedFFN`**: instead of `gate matmul + up matmul + silu + mul` = **4
  dispatches**, one concat matmul against `blk.{i}.ffn_gu` plus one in-place
  SwiGLU pass — **2 dispatches**. Measured on the 0.5B: decode ~269 → ~299
  tok/s (+~11%) for QKV, ~303 → ~312–331 (+~3%) for FFN
  (`GRAPH-REFACTOR-PLAN.md` §17, Phases 10–11).

Both fusions are **gated by measurement, not ideology**: FusedFFN is built
only when `nf ≤ 16384`, because on the 7B model (`nf = 18944`, so the concat
matmul's output is 37 888 wide) the single wide matmul measured *slower* than
two narrow ones — 42.5 vs 46.7 tok/s — and the gate turns it off there.
Mixed-quantizer layers (e.g. Q6_K attention-V among Q4_K q/k) cannot share a
concat weight at all; for those, a second class (`qkv_bias_rope_store`) keeps
three separate matmuls and fuses only the epilogue. Qwen3 has its own variant,
`FusedQkvNorm`, which additionally folds the per-head Q/K RMSNorm that Qwen3
requires: `3 matmul + 2 qk_norm + 2 rope + 2 store → 2`.

Also note the `SwiGLU`/`Silu` rows in the census: on the *unfused* path the
builder still emits `silu` then `mul`, and a later pass (doc 06) rewrites the
`mul` into `SwiGLU` in place, orphaning the `silu` node. The orphan stays in
the node list — the scheduler skips nodes without buffers — which is why both
counts are 24. In the DOT dump you can literally see the stale edge
`n16 → silu(18)` alongside the fused `n16 → SwiGLU(19)`.

### 2.7 The reuse invariant: topology = f(GraphParams)

Here is the sentence the whole design stands on: **the graph topology is a
deterministic function of `GraphParams`** — equal parameters produce an
identical graph, node for node. `GraphParams` (`src/graph/params.rs:52`)
contains exactly `n_tokens`, `n_seqs`, `n_out` (tail rows, §2.8), `gtype`
(prefill or decode), `cparams` (context size, flash-attn flag, GPU
participation, the two fusion toggles), and `weights_version`.

Conspicuously absent: `n_past` — how many tokens are already in the KV cache.
That number changes on *every decode step*, and it is deliberately not part
of the structure. Where does the position live instead? In an **input node**:
the `positions` leaf (node 1) is filled with fresh values before every
execution, and the KV store/load and RoPE nodes read it as data. Positions
are data, not structure — the one invariant everything else protects
(`ops.rs:3-6` states it; §3.3 Q5 asks "what breaks without it").

Given that invariant, reuse is a six-field comparison: `GraphCache::try_reuse`
compares the new `GraphParams` against the cached one; equal means the graph —
and the allocator, with its persistent KV regions — are reused as-is, and only
the input data is refreshed. During a 500-token generation the prefill graph
is built once, the decode graph is built once (the first decode step, because
`nt` changed 30 → 1), and the remaining ~500 steps replay it.

### 2.8 The n_out tail: shrink the graph to the rows you sample

One forward pass produces logits for *some* tokens, but generation only ever
samples the **last** token (single sequence). Computing lm_head — the final
`[896 → 151936]` matmul — for all 30 prompt tokens costs
30 × 896 × 151936 ≈ **4.1 × 10⁹** multiply-accumulates; for the last token
only, ≈ **1.4 × 10⁸**. minfer, like llama.cpp's `inp_out_ids`, draws the line
further up: after the **last** layer's attention output projection, two
`GetRows` nodes select the tail `n_out` rows — of the attention output *and*
of the residual — so the last FFN block, both final residual adds, the final
norm, and lm_head all run on `n_out` rows instead of `nt`. That is nodes
428/429 in the dump; with `n_out = 1` the entire output stack is one row.
The row indices are the `tail_ids` **input node** (node 2) — again data, not
structure — filled with `[nt-n_out .. nt)` before execution. Measured impact
on 0.5B prefill: ~3900–4000 tok/s, ~+55% over the full-nt graph (plan §17,
deviation 16).

One subtlety worth admiring: `tail_ids` is declared at the **head** of the
graph, next to the other inputs, though its consumers sit at the very end.
The code comment explains why (`graph.rs:55-60`): a mid-graph input node
would split execution into extra backend boundary segments, each costing
full-stream syncs and host round trips on the GPU path. Node order is not
semantics — only the edges are — so declaration position is free to optimize
execution, not readability.

## 3. Implementation

### 3.1 Data in / data out

**In** (from doc 04 and doc 03):

- `tokens: &[u32]` — the tokenized prompt (prefill) or the single generated
  token (decode). Becomes the `token_ids` input node, `[nt, 1, 1, 1]`, `I32`.
- `positions: &[usize]` — the KV slot of every token: `0..len` for prefill,
  the running position per decode step. Becomes the `positions` input node,
  same shape, `I32`.
- The loaded model: hparams (layer count, head counts, FFN width, eps,
  RoPE base/scale) and the weight tensors — which the builder references
  **by name only** (`MatMulMeta.weight_name = "blk.7.attn_q.weight"`), not
  by pointer. The graph stays pure data; the allocator resolves names to
  buffers at execution time.
- `GraphParams` — the full set of structure-deciding knobs (§2.7).

**Out**:

- A `ComputeGraph`: 440 nodes (0.5B prefill with n_out = 1), `inputs = [0, 1,
  2]`, `outputs = [439]` (the lm_head matmul). No buffers, no backends, no
  numbers — nothing computed.

Two layout conventions the shapes encode (from `docs/ARCHITECTURE.md` §4.5):
weights are metadata `[in, out]` but memory row-major `[out][in]` — so a
matmul's output dim is `w.shape[1]`; activations are shape `[d, nt, 1, 1]`
with token-major memory `[nt][d]`; and integer inputs (`I32`) are stored
bit-exactly inside f32 buffers as `f32::from_bits` patterns
(`fill_input_i32`, doc 07) so one buffer pool serves both dtypes.

A sense of scale for the persistent parts the graph merely *declares*: each
layer's K region is `[n_kv_embd=128, n_ctx=4096]` f32 = 2 MB; K + V per layer
= 4 MB; 24 layers ≈ **101 MB** of KV the allocator must keep alive across
every rebuild (doc 07 owns that machinery).

### 3.2 Key code

#### The node and the graph (`src/graph/mod.rs:82-115`)

```rust
/// Single compute node.
#[derive(Debug, Clone)]
pub struct CNode {
    pub id: NodeId,
    pub name: String,
    pub op: Op,                 // operation + full payload
    pub src: Vec<NodeId>,       // input dependencies (the edges)
    pub out_shape: [usize; 4],  // output shape [d, nt, 1, 1] convention
    pub out_dtype: DType,       // output element type
    /// Backend assigned by the scheduler (Phase 4); None = undecided.
    pub backend: Option<Backend>,
    pub meta: NodeMeta,         // weight names, rope/attn params, ...
}

/// Compute graph: topologically ordered node sequence + input/output sets.
#[derive(Debug, Clone, Default)]
pub struct ComputeGraph {
    pub nodes: Vec<CNode>,
    pub inputs: Vec<NodeId>,    // input nodes that need external filling
    pub outputs: Vec<NodeId>,   // output nodes (logits, etc.)
    /// Graph identifier for reuse detection (CUDA Graph caching etc.).
    /// Populated by `GraphCache::replace_graph` (monotonic per process);
    /// a reused graph keeps its uid.
    pub uid: u64,
}
```

Note what is *absent*: no buffers, no data pointers, no backend — decisions
for later stages. The IR records only structure and shape; the
`backend: Option<Backend>` slot exists so doc 06 can fill it in place, and
at this stage it is `None` everywhere.

Topological order is not maintained — it is *implied*. The `topo_order()`
doc comment (`src/graph/mod.rs:151-153`) says it outright: *"The builder
appends sources before consumers, so `nodes` is already topologically
ordered; this validates the invariant and returns a stable order (used by
the allocator)."* Because a node can only reference ids the builder has
already returned, `nodes` is sorted the moment it is built. `topo_order()`
still runs a full Kahn pass — but its real job is validation (cycle
detection, dangling ids), and its result is deliberately *not* used as the
execution order (a G3 bug story told in doc 07).

#### Operations with payloads (`src/graph/ops.rs:36-101`, abridged)

```rust
/// Operator type. Implements full `PartialEq` (payloads included) so debug
/// builds can verify graph-rebuild structural identity; the production graph
/// reuse decision is params-only (see docs/GRAPH-REFACTOR-PLAN.md §6).
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Leaf input node (token ids, positions, KV idx, ...). Filled externally
    /// each step via `GraphAllocator::fill_input`; never part of the topology.
    Input,

    Add,
    Mul,
    Silu,
    // ... Scale(f32), Softmax { dim } — ggml-parity vocabulary, no live
    //     architecture emits them today ...
    RmsNorm {
        eps: f32,
    },
    QkNorm {
        hd: usize,
        nh: usize,
        eps: f32,
    },
    MatMul {
        transpose_b: bool,
    },
    GetRows,
    RoPE {
        style: RopeStyle,
    },
    Attn {
        mode: AttnMode,
    },
    // ---- KV cache (persistent external buffer; positions are data) ----
    KvcacheStore {
        layer: usize,
    },
    KvcacheLoad {
        layer: usize,
    },
    // ... fused: SwiGLU, FusedBiasRope, FusedQKV{layer},
    //     QkvBiasRopeStore{layer}, FusedFFN, FusedQkvNorm{layer}
```

Two things to notice. First, the KV ops carry **only the layer index** — no
position, no length. The module doc at `ops.rs:3-6` states the invariant:
*positions are data, injected via the `positions` input node, so the graph
topology never depends on `n_past`*. Second, the `PartialEq` derive is
annotated with its purpose: structural identity checks for the reuse
invariant.

Weights and other per-node parameters travel in `NodeMeta`, a concrete enum
(`ops.rs:160-178`) rather than the plan's original `Box<dyn Any>` — a
recorded deviation that buys `PartialEq`, no downcast panics, and `Clone`
nodes. The workhorse is `MatMulMeta { weight_name, bias_name,
weight_ttype, in_dim, out_dim }`: the graph references the weight **by
name** (`"blk.7.attn_q.weight"`), and the backend resolves that name to a
registered buffer at execution time. That indirection is what keeps the IR
pure data — and it is why doc 03 registered every weight under exactly these
GGUF names.

The builder's only real primitives are `node(name, op, src, out_shape,
out_dtype, meta)` — which appends a `CNode` with `id = graph.nodes.len()` and
`backend: None`, and returns that id — and `input(name, shape, dtype)`, which
creates an `Op::Input` leaf with no sources, records its id in
`graph.inputs`, and is otherwise an ordinary node (its doc comment already
names the payoff: *"so `n_past`/positions changes never force a graph
rebuild"*). Everything else in the 496-line file is sugar over `node`.

#### The builder in practice (`src/graph/builder.rs:127-145`)

A representative convenience method, `matmul`:

```rust
    pub fn matmul(&mut self, x: NodeId, w: &Tensor, bias: Option<&Tensor>) -> NodeId {
        let out = w.shape[1] as usize;   // GGUF: out dim = shape[1]
        let nt = self.graph.nodes[x].out_shape[1];
        let name = format!("matmul_{}", w.name);
        self.node(
            &name,
            Op::MatMul { transpose_b: false },
            &[x],
            [out, nt, 1, 1],             // output width, then token count
            DType::F32,
            NodeMeta::MatMul(MatMulMeta {
                weight_name: w.name.clone(),
                bias_name: bias.map(|b| b.name.clone()),
                // ... weight_ttype, in_dim = w.shape[0], out_dim = w.shape[1]
            }),
        )
    }
```

Shapes propagate automatically: the output width comes from the weight, the
token count from the input. The bias is not a separate node — it is a name in
the metadata, and each backend's matmul kernel applies it inline (a small
taste of how aggressively this IR avoids node spam). `transpose_b` is
`false` everywhere in the live graph: minfer stores weights row-major
`[out][in]` and never needs the transposed form; the flag exists for ggml
vocabulary parity.

And the KV pair (`src/graph/builder.rs:368-389`, store side — trimmed):

```rust
    /// Write this step's K/V into the layer's persistent KV region at the
    /// positions carried by `pos`. `n_ctx` sizes the persistent region.
    pub fn kvcache_store(
        &mut self,
        layer: usize,
        k: NodeId,
        v: NodeId,
        pos: NodeId,
        n_ctx: usize,
    ) -> NodeId {
        let n_embd = self.graph.nodes[k].out_shape[0];
        // shape mirrors the persistent region so the allocator can size it
        self.node(&format!("kv_store.{layer}"), Op::KvcacheStore { layer },
                  &[k, v, pos], [n_embd, n_ctx, 1, 1],
                  DType::F32, NodeMeta::Kvcache(/* ... */))
    }
```

The store node reads three sources — the rotated K, the raw V, and the
positions input — and its declared output shape `[n_kv_embd, n_ctx]` is the
allocator's sizing contract for the persistent region. `kvcache_load` is even
simpler: **no sources at all** (it is a view of the region, which is why
`kv_load` nodes have no incoming edges in the DOT dump), carrying
`{layer}` again and nothing positional.

#### The main event, part 1: inputs and the QKV branch (`src/models/qwen2/graph.rs:53-131`)

```rust
        let inp_ids = b.input("token_ids", [nt, 1, 1, 1], crate::graph::DType::I32);
        let inp_pos = b.input("positions", [nt, 1, 1, 1], crate::graph::DType::I32);
        // G3 tail-row reduction input, declared at the graph HEAD (not beside
        // its consumers at the last layer): an input node mid-graph splits the
        // forward into extra CPU/CUDA boundaries (2 full-stream syncs + host
        // round-trip copies per step on the split path). R3-A1,
        // docs/CUDA_OPTIMIZATION.md Part III. Node order is not semantics —
        // the consumers below just reference the handle.
        let tail_ids = (params.n_out < nt).then(|| {
            b.input(
                "tail_ids",
                [params.n_out, 1, 1, 1],
                crate::graph::DType::I32,
            )
        });

        let mut h = b.embedding(inp_ids, model.tok_embd.as_ref().unwrap());
```

Then, inside the `for (il, l) in model.layers.iter().enumerate()` loop, after
`rms_norm`, the Q/K/V projection has **three build-time classes**, selected by
a gate that reads only `nt`, `GraphParams`, and the layer's own tensors:

```rust
            let fuse_qkv = nt == 1
                && params.cparams.gpu
                && params.cparams.fuse_qkv
                && l.bq.is_some()
                && l.bk.is_some()
                && l.bv.is_some();
            let (q, kv) = if fuse_qkv && Self::qkv_concat_available(&l.wq, &l.wk, &l.wv) {
                let qkv = b.fused_qkv(
                    normed,
                    inp_pos,
                    il,
                    FusedQkvMeta {
                        qkv_weight: format!("blk.{il}.attn_qkv"),
                        // ... biases, dims, rope params, kv_elems ...
                    },
                );
                // q lives at concat offset 0 (rows 0..nqt); K/V went into the
                // persistent regions via the fused store — read them back.
                let kv = b.kvcache_load(il, nkt, n_ctx, nk);
                (qkv, kv)
            } else if /* mixed-quant class: 3 matmuls + qkv_bias_rope_store */ {
                // ...
            } else {
                let q = b.matmul(normed, l.wq.as_ref().unwrap(), l.bq.as_ref());
                let k = b.matmul(normed, l.wk.as_ref().unwrap(), l.bk.as_ref());
                let v = b.matmul(normed, l.wv.as_ref().unwrap(), l.bv.as_ref());
                let q = b.rope(q, inp_pos, hp.rope_style, RoPEMeta { /* ... */ });
                let k = b.rope(k, inp_pos, hp.rope_style, RoPEMeta { /* ... */ });
                b.kvcache_store(il, k, v, inp_pos, n_ctx);
                let kv = b.kvcache_load(il, nkt, n_ctx, nk);
                (q, kv)
            };
```

This is where "the graph is data" pays its first visible dividend: the
*decision* about decode fusion is an ordinary `if` in a pure function, and
each branch emits a different topology. A different `GraphParams` (nt = 1 vs
30, or `MINFER_NO_FUSE_QKV=1` in the environment) yields a different graph —
and because the deciding values all live in `GraphParams`, the difference is
exactly reproducible.

#### The main event, part 2: attention, tail rows, FFN (`src/models/qwen2/graph.rs:195-263`, abridged)

```rust
            // attention
            let attn_out = b.attn(
                q,
                kv,
                inp_pos,
                mode,
                AttnMeta {
                    layer: il,
                    n_head: nh,
                    n_head_kv: nk,
                    hd,
                    hd_kv,
                    nkt,             // KV row stride = n_kv_embd
                    scale: attn_scale,   // 1/sqrt(hd), loader.rs:36-38
                },
            );

            // output projection + residual
            let wo = b.matmul(attn_out, l.wo.as_ref().unwrap(), None);
            let is_last = il == model.layers.len() - 1;
            if is_last && params.n_out < nt {
                let tail_ids = tail_ids.expect("tail_ids input declared when n_out < nt");
                let cur_tail = b.get_rows(wo, tail_ids, [ne, params.n_out, 1, 1]);
                let res_tail = b.get_rows(residual, tail_ids, [ne, params.n_out, 1, 1]);
                h = b.add(res_tail, cur_tail);
            } else {
                h = b.add(residual, wo);
            }
```

```rust
            let fuse_gu = nt == 1
                && params.cparams.gpu
                && params.cparams.fuse_ffn
                && Self::gu_concat_available(&l.ffn_gate, &l.ffn_up)
                && nf <= 16384;      // 7B: concat matmul measured slower
            // ... rms_norm(ffn_norm) ...
            let ffn_out = if fuse_gu {
                let gu = b.fused_ffn(normed, FusedFfnMeta { /* ... */ });
                b.matmul(gu, l.ffn_down.as_ref().unwrap(), None)
            } else {
                let gate = b.matmul(normed, l.ffn_gate.as_ref().unwrap(), None);
                let up = b.matmul(normed, l.ffn_up.as_ref().unwrap(), None);
                let g = b.silu(gate);
                let sw = b.mul(g, up);
                b.matmul(sw, l.ffn_down.as_ref().unwrap(), None)
            };
            h = b.add(residual, ffn_out);
```

After the loop, two lines finish the pass: `rms_norm(h, output_norm)` then
`matmul` against `model.output` (lm_head), marked with `b.output(logits)`,
and `b.build()` hands back the finished `ComputeGraph`. Two details deserve a
pause: the attention node takes the **positions input as a third source** —
attention needs it for causal masking (token `t` may attend to positions
`0..=pos[t]`; builder comment at `builder.rs:325-327`) — and on the last
layer the *residual* is also narrowed by a second `get_rows`, because both
sides of the add must shrink or the shapes would disagree.

#### The reuse check (`src/graph/cache.rs:47-64`)

```rust
    /// Params-only reuse check. On success the previously stored graph is
    /// reused without rebuilding (caller then refreshes input data).
    pub fn try_reuse(&mut self, params: &GraphParams) -> bool {
        match (&self.prev_params, &self.graph) {
            (Some(prev), Some(_)) if Self::params_match(prev, params) => {
                self.prev_params = Some(params.clone());
                true
            }
            _ => false,
        }
    }

    fn params_match(a: &GraphParams, b: &GraphParams) -> bool {
        a.n_tokens == b.n_tokens
            && a.n_seqs == b.n_seqs
            && a.n_out == b.n_out
            && a.gtype == b.gtype
            && a.cparams == b.cparams
            && a.weights_version == b.weights_version
    }
```

No node walk, no comparison of 440 nodes — six field comparisons, because
§2.7's invariant makes them sufficient. The caller
(`Qwen2Graph::forward_cached`, `models/qwen2/graph.rs:472-518`) shows the
whole production loop in one glance: `try_reuse`; if it fails, `build` →
register weights → `assign_backends` (doc 06) → `FusionPass` (doc 06) →
`alloc_graph` (doc 07) → store in the cache with a fresh `uid`; then execute
(doc 08) and refresh input data. Notice that `CParams.fuse_qkv`/`fuse_ffn`
are read from the environment at params construction (`MINFER_NO_FUSE_QKV=1`
etc.), so the A/B toggles force rebuilds through the front door — a test
(`fuse_flags_are_part_of_the_reuse_identity`) guards exactly that.

### 3.3 Design choices (why this shape and not another)

**Q1: Why a declarative graph instead of the obvious imperative loop?**
The loop is genuinely simpler to write — the old `forward.rs` did exactly
that, and it worked. Four things it could not do well:

1. *Decode-step reuse.* With a graph, one token of generation = fill 2–3
   input buffers + walk the node list; the expensive part (build → assign →
   fuse → allocate) runs once per distinct `GraphParams`.
2. *Backend assignment before execution.* Deciding "this node runs on Metal"
   requires the global picture *before* running anything, so splits and
   cross-backend copies can be planned (doc 06/08). A loop that computes as
   it goes must decide mid-flight — which is how the old design ended up with
   per-layer fallback heuristics and surprise host round trips
   (ARCHITECTURE.md Appendix A.3).
3. *Fusion as a rewrite pass.* `Mul(Silu(x), y) → SwiGLU` is a few lines of
   pattern matching over the IR (`fusion.rs:39-87`), applied wherever a
   backend has the kernel. In an imperative loop the same optimization is
   hand-woven into the control flow of every model's forward code.
4. *Observability.* `--dump-graph` / `--dump-graph-json` are ~40 lines each
   because the forward pass *is* data (dot.rs:13-55). The old loop had to be
   reverse-engineered from breakpoints.

The cost is real but bounded: 273 + 322 + 496 lines of IR/builder, and every
model expresses its forward as builder calls (Qwen2's build function is ~240
of its 2244 lines). In exchange, the old `forward.rs` was deleted outright
once the graph path matched it bit-for-bit (Phase 6, plan §17: "after
deletion full suite 78 pass; CLI default-path output consistent with the old
implementation"). Nothing regression-tested was lost.

**Q2: Why must building be pure / side-effect-free?**
Because reuse *requires* determinism: `try_reuse` compares six parameters and
then skips building entirely — safe only if the skipped build would have
produced the same graph. Purity is what makes "same params ⇒ same graph" even
meaningful, and the debug structural check enforces it (rebuild and compare,
node for node). Purity also makes building *cheap and repeatable*: the dump
path builds a second graph just to export it (`json.rs::build_runtime_graph`),
no GPU context is touched, and there is no hidden state to invalidate. The
moment someone reaches into the allocator or the GPU during `build`, the
invariant becomes unverifiable.

**Q3: Why does `Op` carry full payloads?**
So that graphs are *comparable* and *self-describing*. Comparison is concrete:
`GraphCache::verify_structural` walks two graphs asserting `op == op` per
node, and the unit test `op_partial_eq_compares_payloads` (`mod.rs:255-264`)
pins the semantics — `RmsNorm{eps: 1e-5} ≠ RmsNorm{eps: 1e-6}`,
`KvcacheLoad{layer: 2} ≠ {layer: 3}`. Without payloads in the equality, two
graphs differing only in an eps or a layer index would look identical and the
tripwire would be blind. Self-description is the DOT dump of §2.5 —
`KvcacheStore { layer: 23 }` printed on the node — and the same payloads drive
execution: the scheduler resolves the layer's persistent K/V regions from
`KvcacheStore{layer}`, and backends read `AttnMeta.scale`/`nkt` directly.

**Q4: Why are KV store/load explicit graph nodes at all?** The KV cache is
"just memory" — the old design hid it inside a cache object. Making store and
load *nodes* buys three things. First, **ordering for free**: execution
follows build order, and the store node is built before the attention that
loads — so "this step's K/V are written before attention reads them" is
list order, guaranteed by the executor rather than by discipline
(doc 08; ARCHITECTURE.md §4.5.5). Second, **the allocator sees the truth**:
the store node's declared shape `[n_kv_embd, n_ctx]` sizes the persistent
region, and the load node's absence of edges marks the region as alive —
which is how KV regions survive graph rebuilds while ordinary buffers are
recycled by liveness (doc 07). Third, **backends resolve locality**: the
`layer` index lets any backend find the sibling K/V regions
(`kv_pair(layer)`) — attention and its KV stay on the same backend by
construction, with no per-token KV drain.

**Q5: What breaks if topology secretly depends on `n_past`?**
Everything in §2.7, concretely. Suppose the build baked the cache length into
node shapes — say attention reading a KV view of `[n_kv_embd, n_past + nt]`:

- `params_match` (which does not compare `n_past`) would call two *different*
  graphs "equal" — step 5 would silently reuse step 4's graph, reading the
  wrong rows. And if you instead added `n_past` to `GraphParams`, `try_reuse`
  would fail on *every* decode step: the graph rebuilds every token, the
  allocator recomputes its liveness mapping every token, and the `uid`
  churns, defeating the CUDA Graph replay cache keyed on it — decode
  throughput collapses to build-and-allocate speed.
- The debug structural check would catch the mismatch (equal params,
  different topology ⇒ panic) — but only in test builds; production takes
  the silent path.

This is why the invariant is repeated in the three places a reader will trip
over it — the `ops.rs` module doc, the `params.rs` module doc ("`n_past`
deliberately absent: it is execution data"), the builder's `input` doc
comment — and asserted by the unit test `kv_nodes_carry_layer_only`
(`builder.rs:470-484`), the smallest expression of "positions are data".

### 3.4 Pitfalls & invariants

1. **KV positions are data, not structure.** `KvcacheStore/Load` carry only
   `{layer}`; positions arrive via the `positions` input node. Topology must
   never branch on `n_past` (Q5). Guarded by `kv_nodes_carry_layer_only`.
2. **Topological by construction.** A node can only reference ids already
   returned by the builder; `topo_order()` validates rather than sorts.
   Execution order = build order (doc 08) — which is also why a KV store
   built before its attention is guaranteed to run before it.
3. **Node order is not semantics — the edges are.** `tail_ids` is declared at
   the graph head but consumed at the last layer (`graph.rs:55-60`), purely
   to keep input nodes from fragmenting execution into extra backend
   boundary segments. When reading `build`, follow `src`, not position.
4. **Fused and unfused are both real graphs, and they must agree.** The
   fusions are gated by `GraphParams` fields, so both topologies are
   first-class, A/B-able via environment variables, and bit-identical in
   output (asserted by `fused_qkv_matches_unfused_decode`). One recorded
   lesson (plan §17.26): when comparing fused vs unfused, the *unfused* path
   must still run the FusionPass — an unfused graph executed without it
   computes silu+mul as two kernels, differing from the single SwiGLU kernel
   by ~1e-6 float noise that amplifies at large magnitudes.
5. **Fusion orphans stay in the node list.** After `Mul→SwiGLU` rewrites, the
   old `Silu` node has no consumers and gets no buffer; the executor skips
   bufferless nodes (`scheduler.rs:228-233`: "dead nodes … are skipped, not
   executed"). Do not panic at a node count that exceeds the number of
   "real" operations.
6. **Attention's output shape comes from metadata, not from Q.** With
   `FusedQKV`, Q is a slice of a wider concat buffer, so `attn` sizes its
   output from `AttnMeta` (`builder.rs:336-339`). Shapes in this IR are
   declared facts, not derived ones.
7. **GPU participation and fusion toggles are part of the reuse identity**
   (`CParams.gpu`, `fuse_qkv`, `fuse_ffn`): they change the topology, so
   they must (and do) change the reuse decision — a test locks this in after
   it was nearly broken (cache.rs:135-140).
8. **Weights are referenced by name, registered elsewhere** (doc 03): a
   `MatMulMeta.weight_name` with no registered weight is an execution-time
   error — the IR cannot check it, and purity forbids it from trying.

## 4. Observe & verify

- `--dump-graph <path>` — exports the forward pass as Graphviz DOT (nodes
  colored by assigned backend, inputs/outputs as double circles) and exits;
  the run prints the node count. The §2.5 census came from
  `MINFER_DISABLE_MPS=1 ./target/release/minfer <model> "Hello"
  --dump-graph /tmp/g.dot` (440 nodes) and the same with `--no-template` and
  a one-token prompt (437, the decode graph).
- `--dump-graph-json <path>` — the same graph as JSON for `viz/` (web
  visualizer); the JSON also carries the `GraphParams` that produced it.
- `minfer viz <model>` — live graph page with per-node data over SSE, built
  by the same `build_runtime_graph` helper.
- `MINFER_TRACE=<path>` — per-node real-data trace during a run; the bridge
  from this doc's structure to doc 08's execution.
- `MINFER_GRAPH_DUMP=<dir>` — dumps per-node logits/KV outputs of the live
  graph (any build) for offline comparison.
- Tests (all in the current tree): `topo_order_validates_chain` /
  `topo_order_detects_cycle` / `op_partial_eq_compares_payloads`
  (`graph/mod.rs`), `builder_creates_topo_sorted_graph` /
  `kv_nodes_carry_layer_only` (`graph/builder.rs`), the reuse quartet
  `reuse_requires_equal_params` / `fuse_flags_are_part_of_the_reuse_identity`
  / `allocator_survives_rebuild` / `structural_check_detects_different_graph`
  (`graph/cache.rs`), and the model-level `graph_logits_match_forward_real_model`
  / `tail_reduction_matches_full_nt` / `fused_qkv_matches_unfused_decode`
  (`models/qwen2/graph.rs`). Historical acceptance bar: graph logits
  **max diff 0.000e0** vs the old imperative path, prefill and decode
  (plan §17, Phases 5 and 8).

## 5. Cross-references

- [04 — Tokenizer and chat template](04-tokenizer-template.md): produces the
  token ids this stage wraps into the `token_ids` input node.
- [06 — Backend assignment and fusion](06-assign-fusion.md): walks this graph
  and fills every `backend: None`; runs the FusionPass whose `Mul∘Silu →
  SwiGLU` rewrite orphaned the silu nodes in the census.
- [07 — Allocator, liveness, KV regions](07-allocator-liveness-kv.md): turns
  shapes into buffers, shares memory by liveness, and owns the persistent
  per-layer KV regions this stage merely declared.
- [08 — Scheduler + execute](08-scheduler-execute.md): executes nodes in
  build order — the guarantee behind "store before the attention that reads
  it".
- [09 — Prefill forward path](09-prefill-forward-path.md) and
  [13 — Decode loop + graph reuse](13-decode-loop-graph-reuse.md): the two
  callers of `forward_cached` — where `GraphParams` comes from and how the
  params-only reuse behaves across hundreds of steps.
- [11 — Attention + vec ops + KV](11-attention-vecops-kv.md): the kernels
  behind `RmsNorm`, `RoPE`, `Attn`, and the KV store/load — including how
  positions drive masking and cache writes.
- [03 — Model dispatch and weights](03-model-dispatch-weights.md): registered
  every weight under the names this stage's metadata references.
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §4 — the verified design summary
  this doc expands (§4.6 has the mermaid layer diagram); Appendix A preserves
  the imperative design this replaced.
- [`docs/GRAPH-REFACTOR-PLAN.md`](../GRAPH-REFACTOR-PLAN.md) §3–6 — the
  design record (IR, builder, fusion rules, reuse) and §17 — the
  phase-by-phase implementation log with the measured fusion/tail numbers
  quoted above and the full deviation list.
- [`docs/LLAMA-COMPUTE-GRAPH.md`](../LLAMA-COMPUTE-GRAPH.md): the llama.cpp
  equivalent (`ggml_cgraph`, `llm_graph_context`, `inp_out_ids`) minfer
  mirrors. `viz/README.md`: the visualizer data format; `docs/GLOSSARY.md`:
  backstop definitions (GQA, RoPE, SwiGLU, liveness…).

← [04 — Tokenizer and chat template](04-tokenizer-template.md) · [Index](./README.md) · [06 — Backend assignment and fusion](06-assign-fusion.md) →
