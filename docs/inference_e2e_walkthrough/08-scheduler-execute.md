# 08 · The scheduler: splits, synchronization, and execute

> **Stage**: graph allocated ([07 — Memory allocation: liveness and the KV regions](07-allocator-liveness-kv.md)) →
> **this stage: the graph finally does work** →
> the forward produces logits ([09 — Prefill: the first forward](09-prefill-forward-path.md)).
>
> **Code**: `src/graph/scheduler.rs` — `BackendScheduler::split_graph` (L73) and
> `execute` (L123); the `Backend` trait contract in `src/graph/backend.rs`; the
> split-boundary helpers in `src/graph/alloc.rs` (`sync_backend` L542,
> `copy_across` L575); the three `execute_node` implementations in
> `src/graph/cpu_backend.rs`, `src/graph/metal_backend.rs`,
> `src/graph/cuda_backend.rs`. All line numbers verified against commit
> `e7fa0da` (the current HEAD).

## 1. Background — where this stage sits

Everything so far in Act 2 of this walkthrough has been *bookkeeping*. Doc 05
built a **compute graph**: a list of nodes, one per math operation of the
transformer, each describing *what* to compute but computing nothing. Doc 06
**assigned** every node to a backend — CPU, Metal, or CUDA — by asking each
backend "can you do this op?" in priority order. Doc 07 **allocated** buffers:
every node's output has a home, memory is shared between nodes whose lifetimes
don't overlap, and each layer owns two persistent KV (KV = key/value, the
attention memory of the model) regions. What exists as this stage begins is:
a `ComputeGraph` whose nodes are in **build order** (the order the builder
appended them, sources before consumers); a `GraphAllocator` owning one
**buffer pool** per backend, with a `node → buffer` map covering every live
node; and the graph's input buffers already filled with this step's data
(token ids, positions — doc 07). Nothing has been computed yet.

This stage is where the math happens. The **scheduler** walks the node list,
hands each node to the backend it was assigned to, and — the new, subtle part
— makes sure that when a node reads its inputs, those inputs actually contain
the values their producers wrote. On the CPU that is almost trivial: a
function computes, returns, the result is in memory. On a GPU it is *not*
trivial, because a GPU is an **asynchronous** device: asking it to do work and
getting the result are two separated events in time. Most of this document is
about managing that gap.

Two terms we will use constantly. **Scheduling** is deciding *where* work runs
(which backend — doc 06) and *in what order, with what synchronization* (this
document). A **split** is minfer's unit of scheduling: a maximal *contiguous*
run of nodes all assigned to the same backend. Without this stage's
synchronization the model would not merely be slow — it would be *wrong*:
attention would read KV rows before they were written, and the logits buffer
would be read while the GPU was still filling it. The scheduler's whole job is
"never read a value before its producer finished writing it", across three
very different devices.

## 2. Principle — how it works and why

### 2.1 Execution is a walk over a list that is already in the right order

The scheduler does **not** run a sort before executing. It walks
`graph.nodes[0..n]` in index order — **build order** — and executes each node
on its assigned backend. Why is that correct? Because of how doc 05's builder
works: every time the builder creates an operation node, the node's sources
already exist as earlier nodes, so every consumer sits after all of its
producers. An ordering with that property has a name: a **topological order**
of the graph (a **DAG** — directed acyclic graph, a dependency network with no
cycles — is the data structure here).

minfer keeps the formal sort only as a checker: `execute` opens with
`debug_assert!(graph.topo_order().is_ok(), ...)` (`scheduler.rs` L124–125; a
Kahn sort, `src/graph/mod.rs` L154). In release builds nothing is re-sorted —
the walk *is* the order. §3.3 returns to this "one order everywhere" decision
with the bug that proved it.

Build order buys two guarantees for free:

1. **A KV store node always executes before the attention node that reads the
   KV it wrote.** `kvcache_store`/`kvcache_load` being *explicit nodes* (doc
   05) makes this an ordinary data-dependency instead of hidden control flow:
   attention's `src` list contains the load node, the load node sits after the
   store node, so the walk order enforces write-before-read with no special
   case anywhere.
2. **The allocator's liveness analysis uses the same order.** Doc 07 computes
   "when is this buffer's last read?" over node ids 0..n. If the executor and
   the allocator disagreed about the order, the allocator would recycle a
   buffer while execution still needed it. That disagreement actually
   happened — the G3 bug, logits off by 21.79 — and the fix was to make both
   sides use build order (§3.3, §3.4).

### 2.2 Three backends, two execution styles

The three backends differ fundamentally in *when* work happens relative to the
call that requests it:

- **CPU** — `execute_node` computes immediately and returns with the result
  already in the buffer. **Synchronous**: the call's return means the work is
  done.
- **Metal** (Apple GPU) and **CUDA** (NVIDIA GPU) — `execute_node` only
  *records* work and returns immediately. **Asynchronous**: the call's return
  means "the work has been *enqueued*", nothing more.

For Metal, "recording" means appending a kernel launch to a **command
buffer**: a list of GPU commands the CPU builds up in memory, which the GPU
executes only after the CPU explicitly submits it (minfer keeps **one command
buffer per split**). For CUDA, recording means launching a kernel on a
**stream**: an ordered queue of GPU work; kernels on one stream run in launch
order, and the launch call returns long before the kernel finishes. The third
word is **synchronize**: wait until every piece of work previously enqueued on
this backend has *finished* — the mirror image of "submit" (hand the batch to
the GPU).

Why be asynchronous at all? Because encoding is cheap and GPU execution is
long: while the GPU chews through kernel #37, the CPU can already be encoding
kernel #38. Synchronizing after *every* node would idle one side or the other
at each step. Batching a whole split into one submission keeps both busy: a
0.5B decode step costs ~3.9 ms end-to-end (~256 t/s, the G1–G3 figures in
`docs/METAL_OPTIMIZATIONS.md` L55) — one submit per split pays the CPU↔GPU
round-trip cost once per step, not once per node.

Asynchrony creates exactly one hazard, and the scheduler exists to police it:
**a host read of a buffer the GPU has not finished writing returns stale
data**. Every place the scheduler touches GPU-written memory is therefore
placed *after* a synchronize.

### 2.3 Splits: contiguous runs are submission units

`split_graph` scans the node list in build order and cuts it every time the
assigned backend changes. Each piece is a `Split`:

```rust
// src/graph/scheduler.rs L22-32
pub struct Split {
    pub backend: BackendTag,
    /// Node id range [start, end) in graph.nodes.
    pub node_range: (usize, usize),
    /// Nodes whose source values live on another backend (copied in).
    pub inputs: Vec<NodeId>,
    /// Nodes consumed by a later split on another backend (copied out).
    pub outputs: Vec<NodeId>,
}
```

A small ASCII example — three nodes assigned CPU → Metal → CPU:

```
nodes:     0(input x, CPU)   1(silu, Metal)   2(add s+x, CPU)
split 0:   [0..1) CPU          outputs: [0]  (x is read by the Metal split)
split 1:   [1..2) Metal        inputs:  [0]  outputs: [1]
split 2:   [2..3) CPU          inputs:  [1, 0]
boundary 0→1: sync CPU (no-op) · copy x  CPU→Metal
boundary 1→2: sync METAL (submit+wait) · copy silu-out Metal→CPU · (x already CPU: no copy)
end:          sync CPU (no-op)
```

(That is the `split_on_backend_change` unit test, `src/graph/scheduler.rs`
L487–503: it asserts `splits[1].inputs == vec![0]`,
`splits[2].inputs == vec![1, 0]`.)

Why **contiguous** runs, rather than "all nodes of backend X, wherever they
sit"? Both answers are about *count*: each boundary costs a synchronize plus
copies, and contiguity makes the boundary count equal the number of backend
*alternations* along the list — the minimum possible for a given assignment.
A split is also the natural submission unit for an async backend: one command
buffer encodes `end − start` kernel launches and pays one submit + one wait at
the boundary (per-node units would multiply round trips). §3.3 revisits this
choice.

Consequences for the two trivial shapes: an **all-CPU graph is one split**
(no backend change → no boundaries → no copies, no syncs — `prev_backend`
never differs), and an **all-Metal graph is likewise one split**. In practice
minfer runs are overwhelmingly single-split: Metal and CUDA are enabled
**all-or-nothing** per model — a GPU backend runs only when *every* graph
weight is registered on it (`Qwen2Graph::weights_on_gpu`; "same all-or-nothing
rule as Metal", `src/graph/cuda_backend.rs` L1296–1297). Mixed multi-split
graphs are real but rare — they occur when one backend rejects an op the other
supports (CUDA accepts only the non-interleaved RoPE style, L1298) — and the
code still handles them matter-of-factly (`scheduler.rs` L243–258 analyzes
"x goes to a Metal silu split and a CPU add split").

### 2.4 The split protocol

`execute` is a loop over splits, and each iteration follows the same four-beat
protocol:

1. **Sync the previous backend** (`alloc.sync_backend(pb)`). Everything the
   previous split enqueued is now *finished*: its output buffers hold final
   values and are safe to read from the host.
2. **Copy this split's cross-backend inputs** (`alloc.copy_across(inp,
   split.backend)` for each `inp` in `split.inputs`). Each copy is a **host
   round trip**: read the producer's buffer to host memory, then write host
   memory into a **staging buffer** (scratch memory holding a value in
   transit) in the consumer backend's pool.
3. **Run the split's nodes in order** via the backend's `execute_node`. CPU
   computes inline; Metal encodes into the split's command buffer; CUDA
   launches onto its stream.
4. **Flush** — at the *next* boundary (or the very end), sync this backend
   and read back any observability captures queued during the split.

After the loop, the same protocol runs once more with no copies: sync the last
backend, so that when `execute` returns `Ok(())` *every* buffer in the graph —
including the logits — holds its final value and is safe to read. That is the
guarantee doc 09's prefill code relies on when it reads the logits right after
`execute`.

One definition borrowed from GPU_SAFETY explains why beat 1 must precede beat
2: a **barrier** is an explicit ordering guarantee making writes from one
piece of GPU work visible to work that follows. Across backends, minfer's
barrier is the split-boundary synchronize (submit + bounded wait); *inside*
one Metal encoder, the backend inserts `memoryBarrierWithScope` between
dispatches, because Metal does not promise cross-dispatch write visibility on
its own (`docs/GPU_SAFETY.md` §3). The scheduler never thinks about the second
kind; the backend does.

### 2.5 The error contract: abort, never silently fall back

Every `execute_node` returns `Result<(), String>`. When a kernel's
preconditions do not hold — a dimension mismatch, an unregistered weight, a
KV position beyond capacity — the backend returns `Err(...)`, the scheduler
propagates it with `?`, and the whole run aborts. It **never** quietly
re-runs the node on the CPU. The reason is numerical, not aesthetic:

- CPU matmuls quantize activations to Q8_0 on the fly (a per-32-value integer
  block format with one scale, doc 10); the GPU backends read activations as
  f32. The two paths produce *different numbers by design* (AGENTS.md rule 9,
  ARCHITECTURE.md §4.5 invariant 6). A mid-run fallback would splice
  Q8_0-rounded numbers into an otherwise f32 stream — every downstream value
  subtly wrong, with nothing to notice but a degraded, hard-to-trace output.
- Backend placement is a **build-time decision** (doc 06, via `supports_op`);
  execution-time failures are invariant violations — bugs in the assignment
  or the kernel — and bugs should be loud. `docs/GPU_SAFETY.md` §2.3 and CUDA
  rule 1 (L203) state this as a hard rule. The call site makes "loud" literal:
  `sched.execute(graph, alloc).unwrap();` (`src/models/qwen2/graph.rs`
  L549–550) turns an `Err` into a panic, and the server wraps the forward so
  an unexpected abort becomes a 500 instead of killing the worker thread
  (`src/server/chat.rs` L234–247).

## 3. Implementation

### 3.1 Data in / data out

**In:**

- `graph: &ComputeGraph` — nodes in build order, each with `op`, `src` (ids
  of producer nodes), `out_shape`/`out_dtype`, `backend: Option<BackendTag>`
  (set by doc 06's `assign_backends`), and `meta` (weight names, RoPE/attention
  parameters).
- `alloc: &mut GraphAllocator` — owns the per-backend buffer pools; maps every
  live node to a buffer (`node_to_buf`); holds the two persistent KV regions
  per layer (resolved by `kv_pair(layer)` → `(k_id, v_id)`); holds the `cross`
  staging map for cross-backend copies. Input buffers are already host-filled
  (`fill_input_i32`, doc 07).

**Out:**

- Every allocated node buffer contains its computed value; the KV regions
  contain this step's K/V rows at the positions given by the `positions`
  input; the graph's output buffers (logits) are readable on the host.
- The `Result` verdict: `Ok(())` means "all splits ran and all backends
  synced"; `Err(msg)` means an invariant failed and the run is aborted.

Shapes worth keeping concrete (Qwen2.5-0.5B: `n_embd = 896`, vocab 151936,
f32 = 4 B): a hidden-state buffer for one decode token is 896 × 4 = 3,584 B
(~3.5 KB); for a 512-token prefill it is ~1.8 MB; the logits buffer is
151936 × 4 B ≈ 608 KB (the size
`docs/cuda_optimization_steps/07-r3-small-model-overhead.md` L79 works with).
Cross-backend copies move buffers of exactly these sizes — which is why "how
many splits exist" is a performance question, not just a correctness one.

### 3.2 Key code

#### The split walk — boundaries first

The heart of `execute` is one loop over the splits, with the boundary protocol
in its first lines:

```rust
// src/graph/scheduler.rs L176-189
for split in &splits {
    if let Some(pb) = prev_backend {
        if pb != split.backend {
            // 1. flush the previous backend's async work
            alloc.sync_backend(pb);
            // 1b. staged Metal/CUDA captures are valid now — read back
            flush_metal_captures(graph, alloc, &mut staged, trace_on, live_on);
            flush_cuda_captures(graph, alloc, &mut cuda_caps, trace_on, live_on);
            // 2. copy this split's inputs across backends
            for &inp in &split.inputs {
                alloc.copy_across(inp, split.backend)?;
            }
        }
    }
```

Beat by beat: `sync_backend` dispatches on the backend tag (CPU is a no-op —
`src/graph/alloc.rs` L542–562; Metal submits the pending command buffer; CUDA
closes any capture window and stream-syncs). Only after that may the captures
be read (their staging writes just landed) and `copy_across` run (it reads the
producer's buffer host-side — guaranteed final now). If the previous split was
on the same backend, none of this runs: same-pool buffers need no copies and
no sync.

#### The CUDA replay hook — skip the whole split

Before the node loop, one backend-specific fast path: a captured CUDA Graph (a
recording of *every kernel launch of this split*, replayable as a single
launch) can replace the entire node walk:

```rust
// src/graph/scheduler.rs L194-213 (cfg attributes abridged)
let replayed = if capture {
    false
} else {
    match split.backend {
        BackendTag::Cuda => {
            let c = alloc.cuda_mut().ok_or("CUDA backend not enabled")?;
            c.graph_replay(graph.uid, split.node_range, graph.capture_nt_hint())
        }
        _ => false,
    }
};
if replayed {
    // the captured launch covers every node of this split (the
    // boundary sync of the NEXT split still closes it out)
    prev_backend = Some(split.backend);
    continue;
}
```

`true` = "I handled this split": the scheduler `continue`s past all of the
split's nodes, and the replayed work — stream-ordered like any other launch —
is drained by the *next* boundary's sync. The replay is keyed on
`(graph.uid, split.node_range)`: `uid` is the graph's identity for reuse
(`src/graph/mod.rs` L111–115), so step 50 replays the recording made at step 3.
And `capture` (trace/viz per-node readback) forces `replayed = false`: a
readback inside a recorded capture would be recorded *into* the graph and
corrupt it — the same rule as GPU_SAFETY CUDA rule 2. The capture mechanics
live in `graph_replay_step` (`src/graph/cuda_backend.rs` L184–244):
executions 1–2 of a split run direct (warmup), the 3rd records, later ones
replay; the `nt_hint` gate (`capture_nt_hint`, `src/graph/mod.rs` L126–132)
restricts capture to decode-shaped graphs. Docs 14/15 cover the rest; this
hook is all the scheduler sees.

#### The node loop — three skip rules, then dispatch

```rust
// src/graph/scheduler.rs L225-241 (input capture under trace omitted)
if node.is_input() {
    continue; // data pre-filled by the allocator
}
// dead nodes (no consumers, not outputs) get no buffer — the
// fusion pass can orphan them (e.g. silu folded into SwiGLU);
// they are skipped, not executed
let Some(br) = alloc.node_buffer(id) else {
    continue;
};
if br.backend != split.backend {
    let op_full = format!("{:?}", node.op);
    let op = op_full.split(['(', '{']).next().unwrap_or("?");
    return Err(format!(
        "node {id} ({op}) buffer on {:?} but executing split is {:?} (assignment/alloc mismatch)",
        br.backend, split.backend
    ));
}
```

Three ways a node can be *not executed*, in order. First, **it is an input** —
no math; the allocator filled its buffer before `execute` was called (doc 07's
`fill_input_i32`); under trace/viz capture its (host-written, therefore
current) data is recorded at the top of the loop, before the `continue`.
Second, **it has no buffer** — the allocator only maps nodes with consumers or
output status; when doc 06's fusion pass folded `Mul(Silu(x), y)` into
`SwiGLU`, the original `silu` node may survive in the list as a dead orphan,
and it is skipped, not executed — build order stays intact and the dead weight
costs one map lookup. Third, **it is an error**: a buffer on a different
backend than the split it landed in means assignment and allocation disagree —
a bug, and it gets a descriptive `Err` naming the node and both backends
(`split_graph` derives splits from `node.backend`, so this "cannot happen";
if it ever does, it fails loudly instead of writing a buffer with the wrong
kernel).

#### Input resolution: which buffer is "the" input?

For each of the node's `src` producers, the scheduler picks the buffer the
kernel should read:

```rust
// src/graph/scheduler.rs L242-258
let mut in_bufs = Vec::with_capacity(node.src.len());
for &s in &node.src {
    // a cross-backend staging copy (split boundary) takes
    // precedence only when it was made FOR this split's backend:
    // a node feeding two different backends leaves one stale
    // cross-buffer (e.g. x goes to a Metal silu split and a CPU
    // add split — cross_buffer(x) ends up Metal), which must not
    // be read by the CPU consumer. Otherwise fall back to the
    // node's canonical buffer (already on the split's backend if
    // no copy was needed for it).
    let sbr = alloc
        .cross_buffer(s)
        .filter(|cb| cb.backend == split.backend)
        .or_else(|| alloc.node_buffer(s))
        .ok_or_else(|| format!("node {s} has no allocated buffer"))?;
    in_bufs.push(sbr.id);
}
```

The rule in one sentence: prefer a cross-backend staging copy *if one was made
for this split's backend*, otherwise read the producer's own buffer. The
subtlety (the comment's scenario): `copy_across` keeps one staging buffer per
(node, destination) pair, so a producer feeding *two* consumer backends leaves
behind a staging buffer for one of them; a consumer on the *other* backend
must not read it. The `.filter` makes the stale entry invisible to the wrong
split.

#### Resolving kv_pair, then dispatching

KV ops need the layer's two persistent regions (doc 07). The scheduler
resolves them *before* borrowing the backend mutably, because the backend —
not the scheduler — knows which sibling buffer an op writes:

```rust
// src/graph/scheduler.rs L259-292 (cfg attributes abridged)
// resolve the layer's KV region pair BEFORE the mutable backend
// borrow (the backend needs it for KV store / attention)
let kv_pair = match &node.op {
    Op::KvcacheStore { layer } => alloc.kv_pair(*layer),
    Op::FusedQKV { layer } => alloc.kv_pair(*layer),
    Op::QkvBiasRopeStore { layer } => alloc.kv_pair(*layer),
    Op::FusedQkvNorm { layer } => alloc.kv_pair(*layer),
    Op::Attn { .. } => match &node.meta {
        NodeMeta::Attn(m) => alloc.kv_pair(m.layer),
        _ => None,
    },
    _ => None,
};
// NOTE: execution follows node id order (build order), which is
// the graph's topological order by construction.
match split.backend {
    BackendTag::CPU => alloc
        .cpu_mut()
        .execute_node(node, &in_bufs, br.id, kv_pair)?,
    BackendTag::Metal => {
        let m = alloc.metal_mut().ok_or("Metal backend not enabled")?;
        m.execute_node(node, &in_bufs, br.id, kv_pair)?;
    }
    /* ... Cuda arm: the same call on alloc.cuda_mut() ... */
}
```

This is the `Backend` trait contract in use — every backend implements the
same four-argument call, and `?` propagates any kernel-invariant failure
straight out of `execute` (§2.5). The trait itself:

```rust
// src/graph/backend.rs L49-70
fn execute_node(
    &mut self,
    node: &CNode,
    in_bufs: &[usize],
    out_buf: usize,
    kv_pair: Option<(usize, usize)>,
) -> Result<(), String>;
/* ... read_host / write_host ... */
/// Wait for async work to complete (CPU: no-op; Metal: submit the pending
/// command buffer). Called between splits and after the last split.
/// A backend that captured a CUDA Graph window for the current split must
/// close it here (capture records launches without executing them).
fn synchronize(&mut self);
```

`in_bufs[i]` is the pool-local id of the buffer for `node.src[i]`; `out_buf`
is the pool-local output id; `kv_pair` is `Some((k_id, v_id))` only for the
KV-touching ops listed above (attention uses it to find the V region while
iterating the K view — doc 11). Note what the contract does **not** contain:
no shapes-in, values-out signature — buffers are identified, not passed, and
the data never moves for a normal op.

The tail after the loop is the same boundary protocol one last time, minus the
copies (`src/graph/scheduler.rs` L346–353): sync the last `prev_backend`,
flush the staged captures, return `Ok(())`. Everything enqueued by the whole
graph is done; the logits sit on a host-readable buffer; the caller (doc 09)
can read them immediately.

#### What a copy across backends actually is

```rust
// src/graph/alloc.rs L575-582 + L608-611 (doc comment trimmed)
pub fn copy_across(&mut self, node_id: NodeId, dst_backend: Backend) -> Result<(), String> {
    let br = self
        .node_buffer(node_id)
        .ok_or_else(|| format!("node {node_id} has no buffer"))?;
    if br.backend == dst_backend {
        return Ok(());
    }
    /* one staging buffer per (node, dst) pair is reused when present, then: */
    let data = self
        .copy_to_cpu(node_id)
        .ok_or_else(|| format!("node {node_id} host read failed"))?;
    let new_id = self.alloc_fresh_in(dst_backend, data.len());
```

The function's own doc comment (L564–574) names the mechanism: "host round
trip through read_host/write_host (shared-memory GPU buffers make this a plain
memcpy both ways)", with the copy landing in the `cross` staging map — the
node's canonical buffer is left untouched, and consumers resolve it via
`cross_buffer` (the mechanism the previous excerpt showed). So: device → host
(a plain host-pointer read for Metal's shared-memory buffers, a staged
`cudaMemcpy`-style read for CUDA) → host → device (`write_host` into a fresh
staging buffer). The early `Ok(())` for same-backend pairs is why
`split.inputs` can be copied without pre-filtering: in the §2.3 example,
`copy_across(node 0, CPU)` is a no-op even though node 0 appears in split 2's
inputs. Staging buffers are allocated **fresh** (bypassing the free-list
recycle) because a recycled buffer could still be referenced by nodes that
execute later in the same run (`Backend::alloc_fresh` doc, `backend.rs`
L36–42).
#### CPU execution: one representative arm

`CpuBackend::execute_node` (`src/graph/cpu_backend.rs` L133) is a big `match`
on the node's op, dispatching to the handwritten kernels of `kernel.rs` /
`vec_ops.rs`. Before the match, two pieces of plumbing make every arm safe:
the KV store is handled first (it needs mutable access to *two* regions at
once, L143–177), and aliased in-place inputs (liveness reuse may map `out_buf`
onto an input — doc 07) are snapshotted so the output can be carved out of the
pool with `split_at_mut` (L179–201). Then the arms; the one that matters most
for the rest of this series is MatMul:

```rust
// src/graph/cpu_backend.rs L265-299 (bias loop abridged)
Op::MatMul { .. } => {
    let meta = match &node.meta {
        NodeMeta::MatMul(m) => m,
        other => return Err(format!("matmul node missing MatMulMeta: {other:?}")),
    };
    let w = self
        .weights
        .get(&meta.weight_name)
        .ok_or_else(|| format!("weight '{}' not registered", meta.weight_name))?;
    // llama.cpp/GGUF convention: metadata [in, out], memory [out][in]
    let od = w.shape[1] as usize; // output dim
    let id = w.shape[0] as usize; // input dim
    let nt = node.out_shape[1];
    if w.ttype == crate::tensor::TensorType::F32 {
        // plain f32 matmul: out[t*od+o] = dot(w[o], x[t])
        crate::vec_ops::mat_mul_f32(od, nt, id, out, w.data_f32(), ins[0]);
    } else {
        // quantized weight × f32 activations (Q8_0-quantized on the fly)
        kernel::cpu_quant_matmul_f32(w, ins[0], out, od, id, nt);
    }
    if let Some(bname) = &meta.bias_name { /* per-row bias add */ }
    Ok(())
}
```

Everything about the CPU path the later docs (10, 11) zoom into is visible
here: weights resolve by name out of the backend's own registry, the GGUF
`[in, out]` metadata convention fixes which shape index is which, and the
f32-vs-quantized fork is exactly where `cpu_quant_matmul_f32` (doc 10's
subject — quantized weight rows × on-the-fly Q8_0 activation blocks) takes
over. The elementwise and normalization ops are the same shape but simpler —
`Op::RmsNorm { eps }` loops rows and calls
`crate::vec_ops::rms_norm_fused_f32(d, dst, row, w.data_f32(), *eps)`
(L223–241); `Op::Silu` is one `vec_silu_f32` call. The two `Err` arms right
here show the error contract on CPU too: a missing `MatMulMeta` or an
unregistered weight aborts rather than improvising. (The KV store arm — the
code that *writes* the KV regions using `kv_pair` and the `positions` input —
is excerpted in doc 11, where it belongs.)

#### Metal execution: encode now, compute later

The Metal `execute_node` is structurally identical — same match, same arms —
but every arm ends in a *command-buffer encode*, not a computation:

```rust
// src/graph/metal_backend.rs L323-341 (abridged)
let cb = self.cb();          // the split's one command buffer
/* ... */
match &node.op {
    Op::Input => Ok(()),
    Op::Silu => {
        if in_bufs[0] != out_buf {
            self.copy_in(out_buf, in_bufs[0]);
        }
        let n = self.pool[out_buf].length() as usize / 4;
        cb.silu_f32(self.buf(out_buf), n);   // enqueue, don't compute
        Ok(())
    }
    /* ...add, mul, rms_norm, matmul, attn, ... all encode into cb... */
```

`self.cb()` (L146–157) lazily creates the split's single `MpsCommandBuffer`
on the first op and hands it to every subsequent op — that is the "one command
buffer per split" rule from AGENTS.md (rule 8) living in code. Each `cb.*`
call appends a kernel dispatch; nothing runs on the GPU yet. The actual
submission happens in `synchronize` (L1023–1024), which just calls
`submit_pending` (L159–186): take the box, `cb.submit()`, clear the pointer.

`submit()` (`src/metal.rs` L1935–1978) is where the GPU_SAFETY rules get
enforced. It commits the command buffer, then waits on a dispatch semaphore
with a **10-second timeout** and checks the final `MTLCommandBufferStatus`:
`Completed` → `Ok(())`; any other status → `Err` with the recent dispatch
labels attached; timeout → `Err("Metal command buffer timed out after 10s
(GPU hang)")`. The history is a real machine freeze (GPU_SAFETY §1,
2026-08-02): the old submit waited *forever* on the semaphore and never
checked status, so one GPU fault hung the whole process — and, because Metal
clients share the GPU, threatened the machine. Now a hang costs at most 10
seconds and produces a diagnosable error (the last 16 dispatch labels, recorded
when `MINFER_TRACE` is on, identify the faulting kernel family). Doc 14 shows
the full submit path.

#### What `synchronize` actually guarantees

Because the whole async story funnels into this one call, it is worth stating
its guarantee precisely. After `alloc.sync_backend(b)` returns for backend
*b*:

1. **Every node of every split already run on *b* has finished executing.**
   Metal: its pending command buffer was committed and its completion handler
   fired with status `Completed` (or the wait timed out → error). CUDA: the
   stream was synchronized (and any open capture window was closed by
   instantiating + launching the recorded work). CPU: nothing was ever
   pending.
2. **Therefore every buffer written by *b* holds its final value** — safe to
   read from the host (trace readbacks, `copy_to_cpu`), and safe for another
   backend to read *via a host round trip*: `copy_across` reads host-side
   precisely at this point in the protocol. Note the division of labor:
   synchronize does *not* copy across backends, and `copy_across` does *not*
   synchronize — the pairing is the scheduler's, and the invariant "never
   host-copy a GPU-pending buffer" (ARCHITECTURE.md §4.5 #4, from a real
   Phase-3 KV-corruption bug) is enforced by that ordering.
3. **It says nothing about backends never synchronized** — which is why the
   scheduler syncs on *every* boundary and once after the last split, and why
   a CPU-only build "never calls it" (the trait doc's note, `backend.rs`
   L62–64): a synchronous backend has no gap to close.

### 3.3 Design choices (why this shape and not another)

**Why execute in build order instead of re-sorting topologically?**
Four reasons stack up:

1. *It is already a topological order* — the builder appends sources before
   consumers (doc 05). Re-sorting would compute, per execution, a permutation
   the builder already guaranteed.
2. *One order everywhere is a correctness feature, not a convenience.* The
   allocator's liveness must predict exactly which buffers are still needed at
   each step of execution (doc 07). This is not hypothetical: the G3 bug was
   exactly that disagreement — the Kahn order moved source-less nodes (like
   `kv_load`) earlier, liveness concluded a buffer was dead before the
   build-order executor had read it, attention reused the residual buffer, and
   the tail `get_rows` read the attention output instead of the residual —
   logits off by **21.79** (`docs/GRAPH-REFACTOR-PLAN.md` §22, fixed in commit
   `8febf4c`). The fix was not "sort better"; it was "everyone uses build
   order".
3. *Store-before-attention for free.* With explicit KV nodes (doc 05) plus
   build order, the write-before-read property of KV needs no scheduling
   logic at all — it falls out of list order.
4. *It matches ggml.* llama.cpp executes `ggml_cgraph.nodes[0..n]` in order;
   the scheduler's module doc (L1–15) says so explicitly: "matching ggml,
   which executes nodes[0..n] in order".

**Why is a host round trip acceptable for cross-backend copies?** Because the
case is rare, the buffers are small, and the alternative is a lot of plumbing.
Splits are rare (§2.3: all-or-nothing GPU gates make single splits the norm;
typically zero to two cross-boundary copies per forward), and the values that
cross are small — a decode-step hidden state for 0.5B is 3.5 KB, and even a
512-token prefill's hidden state is ~1.8 MB, one memcpy each way on Metal
where GPU buffers live in shared host-visible memory (the `copy_across` doc:
"shared-memory GPU buffers make this a plain memcpy both ways"). The
alternative — device-to-device **peer-to-peer copies** — would need each
backend to expose cross-device primitives (and CPU↔GPU are not peer devices
at all; the host *is* the intermediary), plus allocator plumbing to track peer
mappings, for a code path that runs zero or a handful of times per forward.
The one real cost of the round trip — the sync it requires — is a cost the
correctness protocol already pays: nothing may read a GPU buffer before its
split syncs.

**Why must `execute_node` take `&mut self`?** Two independent reasons, both
visible in the code:

1. *The backend mutates its own state to do the work.* The CPU backend writes
   results into its buffer pool (`split_at_mut` over `self.buffers`,
   `src/graph/cpu_backend.rs` L191–201) and mutates the pool in the KV-store
   arm (L160–166). The backend trait doc says this outright (`backend.rs`
   L3–5: "`execute_node` takes `&mut self` (the CPU backend mutates its own
   pool)").
2. *Asynchronous backends mutate encoding state per call.* Metal appends to
   the split's command buffer and lazily creates it (`self.cb()` mutates
   `cb_ptr`, `metal_backend.rs` L146–157); CUDA tracks capture windows and
   launch bookkeeping. "Encode one op" is a stateful operation on the
   backend, not a pure function on the node.

The `&mut self` is also a *safety* choice: it is a compile error to execute a
node while something else borrows the backend's pool — which is why the
scheduler resolves `kv_pair` from the allocator *before* borrowing the backend
mutably (`scheduler.rs` L259–261).

**Why are splits *contiguous runs* rather than per-backend op groups?**
Because the boundary costs (sync + copies) scale with the number of
boundaries, not with the number of nodes moved — contiguity minimizes
boundaries for a given assignment (§2.3) and keeps `split_graph` a single
O(n) scan plus an O(n·src) pass to derive cross-split edges: no graph
reordering, no risk of changing execution order.

**Why derive cross-split inputs/outputs from `src` edges instead of
annotating them at assignment time?** Because assignment (doc 06) and the
fusion rewrites happen *before* this stage and both can change which nodes
feed which; deriving from the final `src` lists at split time keeps the
boundary set consistent with the graph actually being executed — which is what
lets the `split_on_backend_change` test (L487–503) assert the exact
input/output vectors.

### 3.4 Pitfalls & invariants

- **One order everywhere (invariant 5).** Execution, liveness, and the
  "buffer may be freed" logic all use build order; `topo_order()` only proves
  acyclicity. Origin: the G3 regression — 21.79 of logit drift traced to an
  executor/allocator order mismatch, fixed in `8febf4c` together with "inputs
  are never freed" (GRAPH-REFACTOR-PLAN §23: two inputs whose liveness-shared
  buffer got refilled by the *later* input's fill, clobbering the first — the
  `embedding_and_rope` case).
- **Never host-copy a GPU-pending buffer (invariant 4's corollary).** A host
  read of a buffer whose producer kernel is encoded-but-not-submitted reads
  stale bytes. Every host-side read in the split protocol sits after a
  boundary sync. Origin: the Phase-3 KV-corruption bug (ARCHITECTURE.md §4.5
  #4). The trace/viz capture code is arranged around the same trap: CPU
  outputs are read immediately (already final), Metal outputs are blitted to
  staging and read *after* the split's submit, CUDA outputs queue a
  stream-ordered async D2H drained at the boundary (`scheduler.rs` L293–330).
- **Dead fusion orphans are skipped, not executed.** A node without a buffer
  would crash a naive `pool[buf_id]` — the `let Some(br) =
  alloc.node_buffer(id) else { continue }` is a correctness requirement, not a
  nicety.
- **Stale cross-buffers.** One staging buffer per (node, dst backend) means a
  producer feeding two backends leaves an entry that is *wrong* for the other
  backend; the `.filter(|cb| cb.backend == split.backend)` at input resolution
  (L252–256) is the guard. Removing it reads valid-looking but wrong-split
  data — the kind of bug that survives small graphs and fails at multi-backend
  ones.
- **Kernel-invariant violations abort (invariant 6's enforcement).** Silent
  CPU fallback would quantize activations to Q8_0 mid-stream and corrupt the
  numerics (§2.5). The error path is: backend `Err` → scheduler `?` →
  `forward_graph_cached` `.unwrap()` → abort (or HTTP 500 in the server).
  Assignment is build-time; runtime surprises are bugs.
- **Bounded GPU waits, always.** Metal submits wait ≤10 s and check status
  (GPU_SAFETY §2.1); CUDA syncs poll `cudaGetLastError` plus stream state
  (GPU_SAFETY CUDA rule 4). The scheduler never blocks on a GPU without a
  timeout in the path beneath it.
- **No host readbacks inside a CUDA capture window.** The scheduler disables
  per-node capture (and therefore replay) whenever trace/viz capture is on
  (L190–193); a readback inside a recorded launch sequence corrupts the
  capture — GPU_SAFETY CUDA rule 2, learned from the 7e② "faster but wrong"
  incident. (The capture vectors `metal_srcs`/`staged`/`cuda_caps` are
  declared unconditionally with empty non-macOS/non-CUDA flush stubs,
  L167–175, L444–456, so no `#[cfg]` maze forks the main loop.)

## 4. Observe & verify

- **`MINFER_GRAPH_TRACE`** (any value) — at the top of `execute`
  (`scheduler.rs` L127–145) prints one line per split (`[graph] split 0:
  Metal nodes 0-412`) plus a per-op/per-backend node-count table: how many
  splits your model produced and which ops went where.
- **`MINFER_TRACE=<path>`** — per-node real-data trace for the web
  visualizer: after each node executes, its output buffer is analyzed
  (min/max/mean/abs-mean stats + downsampled values) and recorded per step
  (`src/trace.rs`; the hook is `record_node_data`, `scheduler.rs` L369–389).
  KV-region nodes are skipped (up to `n_embd × n_ctx` per layer — huge). Load
  the file at `viz/index.html` to watch the graph compute, node by node;
  details in `viz/README.md` (no doc 17 exists; that README is the reference).
- **Live viz** — `minfer viz <model>` serves the same graph page with a live
  SSE feed: `crate::live::enabled()` (`src/live.rs` L51–56) is checked once
  per `execute()` and flips the same `capture` flag, so nodes light up as
  each split executes. Again: `viz/README.md`.
- **`MINFER_OP_PROFILE`** — Metal-side: per-op host *encode* times plus
  per-submit GPU wait times (first submit prints a full table;
  `metal_backend.rs` L224–242). Shows the encode/execute split of §2.2 in
  real microseconds.
- **Tests** — `src/graph/scheduler.rs` L458–519 covers this stage end to end:
  `assign_all_cpu_and_single_split` (all-CPU ⇒ exactly one split, no cross
  edges), `split_on_backend_change` (CPU→Metal→CPU ⇒ three splits with the
  derived inputs/outputs of §2.3), and `execute_single_backend_graph`
  (computes `silu(x) + x` through the full allocate→fill→execute path). The
  backend suites pin cross-backend copies (`copy_across_cpu_to_cuda_and_back`,
  `cuda_backend.rs` L1477) and replay parity (`cuda_backend.rs` L6159).
- **`MINFER_GRAPH_DUMP=<dir>`** — dumps the logits and layer-0 KV after
  execute, for CPU-vs-GPU comparison of what this stage produced
  (`src/models/qwen2/graph.rs` L552).

## 5. Cross-references

- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §4.3 (the
  assign→fuse→alloc→execute pipeline), §4.5 (invariants 4–6), §5.1 (the
  `Backend` trait list), §5.3 (GPU safety mapping).
- [`docs/GPU_SAFETY.md`](../GPU_SAFETY.md) — the hard rules this stage
  enforces: bounded submit + status check (§2.1), `Err`-not-fallback (§2.3),
  no sync inside a capture window (CUDA rule 2), stream order as the async
  contract (CUDA rule 5).
- [`docs/GRAPH-REFACTOR-PLAN.md`](../GRAPH-REFACTOR-PLAN.md) §16/§22/§23
  (L948–955) — the G3 tail-row optimization and the two liveness-vs-order bugs
  (21.79 logit drift; input-buffer clobber) that made build order the one
  order.
- [05 — Graph build](05-graph-builder-ir.md): why the node list is topological
  by construction, and why KV ops are explicit nodes.
- [06 — Assign + fusion](06-assign-fusion.md): where `node.backend` comes from
  and which fused ops exist (the source of dead orphans).
- [07 — Allocator](07-allocator-liveness-kv.md): buffer pools, liveness
  sharing, the allocator side of the `cross` staging map, KV regions.
- [09 — Prefill](09-prefill-forward-path.md): the caller that reads logits
  immediately after `execute` returns — the consumer of this doc's guarantee.
- [10 — CPU matmul kernels](10-cpu-matmul-kernels.md) and
  [11 — Attention + vec ops + KV](11-attention-vecops-kv.md): inside
  `cpu_quant_matmul_f32`, and the KV-store/attention arms that the `kv_pair`
  contract feeds.
- [14 — Metal backend](14-metal-backend.md) /
  [15 — CUDA backend](15-cuda-backend.md): the full story of the
  one-command-buffer-per-split discipline and CUDA Graph capture/replay that
  this doc only hooks into.
- [`viz/README.md`](../../viz/README.md): the trace format, the live SSE view,
  and how to load a run into the graph visualizer.

← [07 — Memory allocation: liveness and the KV regions](07-allocator-liveness-kv.md) · [Index](./README.md) · [09 — Prefill: the first forward](09-prefill-forward-path.md) →
