# 07 · The allocator: liveness, buffer reuse, and the KV cache

> **Stage**: docs 05–06 built a compute graph and gave every node a backend →
> **this stage decides WHERE every intermediate tensor lives** (and owns the
> KV cache) → doc 08's scheduler then executes the graph through those buffers.
> **Code**: `src/graph/alloc.rs` (`GraphAllocator::alloc_graph`, `fill_input_i32`,
> `ensure_kv`, `copy_across`), `src/graph/cache.rs` (`GraphCache`),
> `src/graph/backend.rs` (`Backend` trait: pool + `alloc_fresh`, `KvProvider`),
> `src/models/qwen2/graph.rs::forward_cached` (input filling call site).

## 1. Background — where this stage sits

By the end of doc 06, the engine holds a **compute graph** — a pure data
structure that lists every math operation of one transformer forward pass as
`CNode`s ("compute nodes"): a `RmsNorm` node, three projection `MatMul` nodes,
a `RoPE` node, a fused `SwiGLU` node, and so on. Each node already knows *which
backend* will run it (CPU, Metal, or CUDA) and how big its output is
(`out_shape`, e.g. `[896, 1, 1, 1]` for one token's hidden state on
Qwen2.5-0.5B). What no node has is a place to put its result. The nodes are
pure description: "the silu of node 17". Silu of *what memory*?

This stage answers that. The **allocator** (`GraphAllocator` in
`src/graph/alloc.rs`) walks the graph once and hands every node a **buffer** —
a region of memory inside a backend's **pool**, addressed by a small handle
(`BufRef { backend, id }`). It also does two jobs that are easy to overlook
but are just as load-bearing:

- it **owns the KV cache**: the per-layer scratchpad that attention reads and
  writes (defined properly in §2.5), and
- it **fills the input buffers**: the token ids and positions your prompt was
  turned into in doc 04 are written into pool buffers here, before any kernel
  runs.

Why not just give every node its own fresh buffer and be done? Arithmetic
makes the naive version ugly fast. The 0.5B decode graph has **437 nodes**
(recorded in `docs/GRAPH-REFACTOR-PLAN.md` Phase 8), so naive allocation means
437 separate memory regions per backend — and for the GPU that is 437 driver
buffer objects to create, register, and keep alive. Worse, the two persistent
things (KV regions) must be sized *once* and survive; a throwaway
allocate-per-node scheme has no place to put them.

The saving observation is old and simple: a buffer's contents only matter
between the moment they are written and the moment they are last read. Node 17
(hello again, silu) writes its output; some later node reads it once; after
that the memory is dead weight that the next operation could reuse. Bookkeep
those windows — the **live ranges** — and buffers can be shared by operations
that never overlap in time. This is exactly what llama.cpp's `ggml_gallocr`
does, and `alloc.rs` says so in its first line: *"Mirrors llama.cpp's
`ggml_gallocr`: buffers are shared between nodes whose live ranges do not
overlap" (`src/graph/alloc.rs:1-10`).

But sharing memory is also where the two most instructive bugs of this codebase
happened: one where a copy read data the GPU had not produced yet (the whole
KV cache silently became zeros), and one where liveness was computed in a
different order than execution, so a buffer was recycled while its reader was
still waiting (logits off by 21.79). Both bugs, and the invariants that now
prevent them, are told in §3.4 — because "why is reuse safe *now*" is the
single best question you can ask about this stage.

## 2. Principle — how it works and why

### 2.1 Buffers, pools, and handles

Three terms, defined once and used everywhere after:

- A **buffer** is a contiguous region of memory holding `size` f32 numbers
  (4 bytes each). On the CPU backend a buffer is literally a `Vec<f32>`
  (`CpuBackend.buffers`, `src/graph/cpu_backend.rs:18-23`); on Metal it is an
  `MTLBuffer` the CPU and GPU can both see; on CUDA it is device memory.
- A **pool** is the backend's list of all its buffers, plus a **free list** of
  ids that are currently unused. Allocating means "find me a buffer of this
  size" — from the free list if one fits, otherwise create one.
- A **handle** (`BufRef`) is just `{ backend, id }` — which pool, which slot.
  Nobody outside the backend ever touches the memory through the id directly;
  the backend resolves `id → &mut [f32]` (`read_host`/`write_host`) or passes
  the id to its own kernels.

One deliberate simplification shapes everything: **every pool buffer is
f32-typed**. The allocator counts sizes in f32 elements (`Backend::alloc_buffer`
"allocate / release a buffer of `size` f32 elements", `backend.rs:32-34`),
Metal sizes buffers as `size * 4` bytes (`metal_backend.rs:296`), and weights
keep their quantized bytes elsewhere (registered by name in doc 03). One dtype
means one allocator, one copy path, one set of host-access functions — and,
as §2.6 shows, even integers ride along as f32 bit patterns.

The last pool property to internalize: **ids are stable**. Once buffer #7
exists, it is buffer #7 until the whole graph is torn down; a recycled id
keeps its size; nothing ever moves. That stability is what lets a GPU record
raw pointers into a captured kernel launch (CUDA Graph replay, doc 15) and
replay them later — the backend trait says it outright: "Implementations must
keep captured pointers stable (pool ids never move memory)"
(`backend.rs:79-80`).

### 2.2 Liveness: when a buffer's contents are precious

A node's output is **live** from the moment the node executes (first write)
until the moment its last consumer has executed (last read). That window is
its **live range**. **Liveness analysis** is just computing everyone's window.

The windows come from the graph structure, not from a clock. If node `h` is
consumed by nodes at execution positions 40 and 240, `h`'s live range is
`[40, 240]` — its buffer must hold `h`'s value through position 240 and not a
step longer. Two nodes whose ranges never overlap can safely share one buffer:
whoever comes first writes, its readers finish, and only then does the second
writer overwrite. A chain of ops shares beautifully:

```
exec position:   0     1      2      3      4      5
node:          input  silu   add    silu   add    output
live range:    [0,5]  [1,2]  [2,3]  [3,4]  [4,5]  [5,∞)

buffers in use at any moment: 2 (input + "current value")
naive:                        6 buffers
```

Every result only feeds the next op, so one scratch buffer ping-pongs with the
input buffer. Contrast a **parallel** shape — `a2 = add(silu(a0), a0)` and
`b2 = add(silu(b0), b0)` computed independently — where both branches are live
simultaneously and *must* get separate buffers. minfer's unit tests assert
exactly these two behaviors: `liveness_reuses_buffers_along_chain` (fewer
buffers than nodes) and `parallel_chains_do_not_share` (`alloc.rs:668-709`).

Two bookkeeping rules extend the basic window, and both exist because of
execution-order realities rather than graph theory:

1. **Graph outputs live forever** (well, until the scheduler copies them out).
   Logits are read *after* the whole graph ran, so their live range ends at
   `order.len()` — one past the last node.
2. **Graph inputs live forever too.** This is the subtler one, and it is a
   recorded bug fix (deviation 23, §3.4): inputs are filled on the host
   *before* execution starts, so a buffer that liveness would normally recycle
   for another input would get clobbered by the later fill. Inputs get the
   same "live to the end" treatment as outputs
   (`alloc.rs:204-210`).

### 2.3 The alloc_graph walk: intervals, sweep, free list

`alloc_graph` runs once per graph (re)build, after doc 06's assign + fusion
passes. The walk is a single forward pass with a running clock:

1. **Tear down the previous graph's liveness buffers** (`alloc.rs:164-177`):
   every id tracked in `buf_alive` goes back to its pool's free list; the
   `node_to_buf` map is cleared. Persistent regions (§2.5) are *not* in
   `buf_alive`, so they sail through untouched.
2. **Order check**: call `topo_order()` — but only to *validate* that the
   graph is acyclic. The order actually used is plain **build order**,
   `0..n_nodes` (`alloc.rs:179-186`). §3.4 explains why this one line is the
   tombstone of the G3 bug.
3. **Compute windows**: `exec[id] = i` gives each node its execution position;
   then one pass over all nodes raises `last_use[src]` to the latest consumer
   position. Finally outputs and inputs are pinned to `order.len()`
   (`alloc.rs:189-210`).
4. **Count consumers** per node (`alloc.rs:214-219`) — the safety input for
   in-place aliasing (§2.4).
5. **Walk nodes in order** (`alloc.rs:221-310`). For each node, first
   **sweep**: free every tracked buffer whose `last_use < i` — its readers
   have all been positioned earlier, so its contents are officially dead
   (`alloc.rs:410-422`). Then decide where this node's output lives:
   - KV store/load nodes → the layer's persistent K region (§2.5);
   - fused QKV nodes → their persistent regions *plus* an ordinary output
     buffer for the concatenated `q|k|v` result;
   - `Silu` / `RoPE` / `QkvBiasRopeStore` → try to **alias** the input buffer
     in place (§2.4);
   - everything else → `alloc_in_pool(backend, size)`, which asks the pool for
     a recycled buffer of exactly that size or creates a new one, then records
     `(backend, id) → last_use` in `buf_alive`.
6. Dead nodes get *no buffer at all*: `last_use[id] > i` is false when a node
   has no consumers (fusion orphans the `Silu` inside a fused `SwiGLU`), so no
   allocation happens, and the scheduler skips bufferless nodes
   (`scheduler.rs:228-233`).

The pool side of step 5 is where reuse actually happens
(`cpu_backend.rs:107-131`):

```rust
fn alloc_buffer(&mut self, size: usize) -> usize {
    if let Some(idx) = self
        .free
        .iter()
        .position(|&id| self.buffers[id].len() == size)
    {
        let id = self.free.swap_remove(idx);
        self.buffers[id].fill(0.0);   // recycled: zero it, so stale data can't leak
        return id;
    }
    self.buffers.push(vec![0.0f32; size]);
    self.buffers.len() - 1
}
```

Note the **exact-size match**: recycling only takes a free buffer whose length
equals the request. That keeps the bookkeeping trivial (a buffer's size never
changes) at the cost of occasionally missing a "big enough" free buffer — a
deliberate trade: first-fit-with-growth would save a few allocations but
complicates every downstream size assertion.

How much does all this save? A hand tally of the 0.5B prefill graph (24
layers, per-layer node list in `docs/ARCHITECTURE.md` §4.6) with a 440-token
prompt makes it concrete. Every hidden-width buffer holds `896 × 440 × 4 B ≈
1.58 MB`; every FFN-width buffer `4864 × 440 × 4 B ≈ 8.6 MB`. Naive
per-node allocation would put ≈55 MB of activation buffers per layer × 24
layers ≈ **1.3 GB** of live-at-build-time buffers on the heap. With liveness,
the *peak simultaneous* set is roughly six hidden-width buffers plus three
FFN-width ones — around **35–40 MB**, about 30× less, and the pool only ever
holds as many buffers as the peak demanded. For decode (nt = 1) the byte
savings are small (a hidden buffer is 3.5 KB), but the buffer *count* still
drops from 437 to a couple dozen — which is what matters for GPU buffer
objects.

### 2.4 In-place aliasing: Silu and RoPE write into their input

An **alias** means two nodes share the *same* buffer on purpose, not by
recycling accident. `Silu` (the sigmoid-linear unit activation) and `RoPE`
(rotary position embedding, which rotates pairs of numbers by an angle derived
from the token's position) are elementwise transforms with a special property:
their output has the same shape as their input, and their input has *no other
reason to keep its old value* if nothing else reads it. So instead of

```
q_buf ──(read)──▶ rope kernel ──(write)──▶ q_rope_buf   [2 buffers, 2 passes over memory]
```

the allocator maps the rope node's output to the *input's* buffer:

```
q_buf ──▶ rope kernel reads and overwrites q_buf in place   [1 buffer]
```

`alloc.rs:263-299` implements this, guarded by exactly two conditions — the
input's **sole consumer** is this op (`n_consumers[src] == 1`, from step 4
above) and the input lives **on the same backend**. Both guards are load
bearing. If another node also reads the input, overwriting it destroys data
that reader still needs. If the input is on another backend, "just use the
input's buffer" would mean running your kernel against memory in *another
device's pool* — and the cross-backend case gets its own treatment (§2.7),
which is precisely where the Phase-3 bug lived (§3.4).

When aliasing applies, the aliased input's live range is extended to cover the
aliasing op's consumers (`alloc.rs:292`) — the buffer now carries *two*
logical tensors' worth of deadlines, and liveness must respect the later one.

Why bother? Three reasons, in decreasing order of "wow":

1. **Correctness on GPU.** This is the surprising one. On Metal, one split's
   kernels are *encoded* into a command buffer and only *submitted* at the
   split boundary (doc 08/14). If the allocator instead made rope read a
   *host-side copy* of its input, that copy would read a GPU buffer whose
   producing kernel is still queued, not run — stale data. Aliasing keeps the
   read/write *inside the same command buffer in kernel order*, which is
   always coherent. This is ARCHITECTURE.md invariant 4's "never host-copy a
   GPU-pending buffer" rule, and it was learned the hard way (§3.4).
2. **Memory traffic.** Each avoided alias-copy is a full pass over the
   activation. The prefill graph runs two RoPEs per layer (Q and K) whose
   inputs have sole consumers, so aliasing skips (896 × 440 + 128 × 440) × 4 B
   ≈ 1.8 MB of copy per layer — about 43 MB of pure memcpy per 440-token
   prefill across 24 layers. (The FFN silu copy is skipped too on the fused
   path, but there the whole `Silu` node is folded into `SwiGLU`, so it is
   fusion's win, not aliasing's.)
3. **Parity with llama.cpp**, which executes rope and silu in place for the
   same reasons (`alloc.rs:273-274`).

The model-side code cooperates with the rule. In the mixed-quant QKV decode
path, the builder deliberately wires attention to the epilogue node *so that*
the q matmul's buffer has exactly one consumer and can alias
(`models/qwen2/graph.rs:135-136`: "Attention is wired to the epilogue node so
q's matmul buffer has exactly one consumer (in-place alias rule, §5)").

### 2.5 The KV cache as persistent regions

Time for the term this doc has been promising. A **KV cache** is the
transformer's memory of tokens it has already processed: for each layer and
each past token, the attention mechanism's **K** (key) and **V** (value)
vectors (what those *are* is doc 11's business; here they are just tensors
named K and V). Autoregressive generation works by *appending* each new
token's K/V to this notepad and letting attention read the whole accumulated
prefix — that is why decode is cheap per token. **KV cache preview** (doc 01's
phrase) means deciding *how big* that notepad is before anything is written.

minfer's allocator owns it as **persistent regions**: each layer gets two
buffers, K and V, each sized `n_kv_embd × n_ctx` f32 elements, allocated the
first time any node of that layer mentions the layer and then **never freed
and never recycled** (`alloc.rs:382-403`). `n_kv_embd` is the KV width —
128 for Qwen2.5-0.5B (2 KV heads × head-dim 64), 1024 for Qwen3-4B — and
`n_ctx` is the context budget from the CLI (`--n-ctx`, default 4096). The
store/load node shapes carry the size (`builder.rs:376-388` builds the store
node with shape `[n_embd, n_ctx, 1, 1]`, "shape mirrors the persistent region
so the allocator can size it").

The byte arithmetic you should carry around:

```
one layer  : 2 regions × n_kv_embd × n_ctx × 4 B
0.5B       : 2 × 128  × 4096 × 4 B =  4.2 MB/layer  × 24 layers ≈ 100 MB
Qwen3-4B   : 2 × 1024 × 4096 × 4 B = 33.6 MB/layer  × 36 layers ≈ 1.2 GB
             ...at n_ctx = 40960 (10× the tokens):              ≈ 12.1 GB  (!)
```

That 12.1 GB is not hypothetical — it is the recorded lesson of
`docs/PERF-QWEN3-4B-VS-LLAMACPP.md` §2, retold from the sizing side in §3.3
below.

Why "persistent, never recycled"? Two lifetimes matter, and both are longer
than one graph execution:

1. **Across forward calls.** The whole point of a KV cache is to survive
   between steps: token 50's attention must read tokens 0–49's K/V, written
   during *previous* forward calls. A liveness-recycled buffer would be
   overwritten by the very next matmul.
2. **Across graph rebuilds.** Prefill (many tokens) and decode (one token)
   have different `GraphParams`, so the prefill→decode transition *rebuilds
   the graph* — new nodes, new `node_to_buf` mapping (doc 13). The allocator
   object, though, is the *same* object (that is the GraphCache design, §2.8),
   so `ensure_kv` finds the existing pair and returns it untouched. Zero
   copies: the KV the prefill just wrote is exactly where decode's attention
   will read it. Deviation 14 records this as the analogue of llama.cpp's KV
   living in the memory context rather than the graph's buffer set.

Mechanically, the K region does double duty as the store node's *output
buffer* (`alloc.rs:226-229`: "the node's buffer = the K region"), and the CPU
executor enforces that contract (`cpu_backend.rs:146-148`: "KV store out
buffer must be the K region"). The V region is a sibling the kernel reaches
through the `kv_pair` handle (§3.2, excerpt 8). The load node executes as a
no-op — it is a *view* of the K region (`cpu_backend.rs:381`).

### 2.6 Filling inputs: why f32 buffers, and the I32 bit-pattern ride

The graph declares three inputs for a prefill (`models/qwen2/graph.rs:53-67`):
`token_ids` `[nt,1,1,1]`, `positions` `[nt,1,1,1]`, and (when the tail-row
optimization is active) `tail_ids` — all typed `DType::I32` in the IR. Yet
every pool buffer is f32 (§2.1). The bridge is `fill_input_i32`
(`alloc.rs:436-446`): each u32 is packaged as `f32::from_bits(v)` — a pure
bit reinterpretation, *not* a numeric conversion — and written into the input
node's buffer via the backend's `write_host`. On the consumer side the
kernels run the inverse, `x.to_bits()`, recovering the exact integer:

| consumer | code |
|---|---|
| embedding row gather (CPU) | `ins[0][t].to_bits()` → token id (`cpu_backend.rs:326`) |
| generic get_rows (CPU) | `ins[1][t].to_bits() as usize` (`cpu_backend.rs:311`) |
| RoPE positions (CPU) | `ins[1][t].to_bits() as usize` (`cpu_backend.rs:352`) |
| attention positions (CPU) | `ins[2][t].to_bits() as usize` (`cpu_backend.rs:411-415`) |
| CUDA kernels | device-side `__float_as_int` in one pass (`cuda_kernels.cu:2483-2497`) |

Why this trick at all? Because of the uniform-pool decision. The alternatives
were a second, integer-typed pool per backend (double the allocator state,
double the copy paths, and a special-case `alloc_buffer(size, dtype)` in
every backend) or converting integers to their float *values* (which is exact
only for small integers and lossy in surprising ways). Riding the bits keeps
one pool and is lossless. The safety envelope recorded in the code — "exact
for |v| < 2^24" (`alloc.rs:437`, `cpu_backend.rs:2-4`) — is generous
headroom: 2²⁴ = 16,777,216, and real data sits far inside it — the largest
vocabulary here is 151,936 token ids, and contexts top out in the tens of
thousands (the biggest `n_ctx` in the perf tables, 65,536, is clamped to the
model's 40,960-token `max_seq_len` before it reaches the allocator). Within
that envelope the pattern is robust even if some stage ever treated the
contents as a float *value* instead of bits.

The CUDA note is worth savoring because it shows the constraint pushing back:
the decode kernels need *raw int32*, but converting on the host would need a
sync (and would break CUDA Graph replay, doc 15). So a tiny device kernel
`f32_bits_to_i32` reinterprets the bits *on the GPU*, "fully device-side, so
the per-layer path needs no host sync (and stays CUDA-Graph-replayable)"
(`cuda_kernels.cu:2485-2487`).

Filling happens at a strict moment: after `alloc_graph`, *before* the
scheduler runs (`models/qwen2/graph.rs:520-536`: `cache.current()` → three
`fill_input_i32` calls). That ordering is exactly why inputs must be pinned
out of the recycling pool (§2.2 rule 2) — the fills would otherwise fight
each other over a shared buffer before any node had executed (§3.4, bug 2b).

### 2.7 Cross-backend staging and alloc_fresh

When doc 06's assignment puts a producer on Metal and its consumer on CPU, the
scheduler inserts a **split boundary**: sync the previous backend, then copy
the consumer's inputs across (`scheduler.rs:176-188`). The copy lands in the
allocator's `copy_across`, which routes through `copy_to_cpu` (a host round
trip — Metal/CUDA buffers here are CPU-visible, so this is a plain memcpy)
and then `write_host` into a buffer on the *destination* pool
(`alloc.rs:575-636`).

That destination staging buffer must be **fresh** — `alloc_fresh_in`
(`alloc.rs:336-358`) — never drawn from the recycle free list. The trait
comment is the design record (`backend.rs:36-42`):

```rust
/// Allocate a buffer that bypasses the recycle free list. Split-boundary
/// staging needs this: at execute time the free list holds ids whose
/// physical contents are still referenced by node_to_buf and get
/// read/written later in the same execute — recycling one would clobber
/// in-flight data. Fresh buffers enter the normal free list on
/// free_buffer (at graph rebuild), where liveness recycling is safe.
```

Unpacking that: during the *build loop*, the free list is safe to draw from
because the sweep clock (§2.3 step 5) advances monotonically through liveness
order — anything freed is dead from that position onward, and every later
allocation is also later in execution. But a split boundary is an
*out-of-band* allocation: it happens at execution position P, with the free
list frozen in whatever state the build left it. The list can still contain a
buffer whose last reader sits at position ≥ P (nothing after it in the build
happened to want that size), and a staging write at P would clobber data that
execution has not consumed yet. Fresh allocation sidesteps the whole question
by never consulting the list.

Staging buffers are one-per-(node, destination backend) per graph: the first
execute allocates, every later execute of the reused graph just rewrites the
same buffer ("no per-step allocation", `alloc.rs:28-34`). They are freed at
the next rebuild (`alloc.rs:171-177`) — the "at graph rebuild" moment the
trait comment mentions, where returning them to the normal free list *is*
safe because the next build's monotonic sweep re-establishes the invariant
from scratch.

### 2.8 GraphCache: the allocator outlives the graph

The final principle is an ownership decision that makes §2.5 possible.
`GraphCache` (`src/graph/cache.rs`) is a tiny struct with three fields: the
current graph, **the allocator**, and the params the graph was built for
(`cache.rs:24-28`). Reuse is decided by `try_reuse`, which compares
`GraphParams` only — `n_tokens`, `n_seqs`, `n_out`, `gtype`, `cparams`
(including `n_ctx`, the GPU flag, and the fusion toggles), `weights_version`
(`cache.rs:47-64`). Equal params ⇒ the topology is deterministic ⇒ reuse the
graph and just refresh input data (§2.6). Mismatched params ⇒ the caller
builds a new graph and `replace_graph` swaps it in — **keeping the allocator**
(`cache.rs:66-73`).

That is the entire reason the KV regions survive: the regions live inside the
allocator, the allocator lives inside the cache, and rebuilds replace only the
graph. The unit test `allocator_survives_rebuild` pins this contract with a
planted persistent region (`cache.rs:203-217`).

## 3. Implementation

### 3.1 Data in / data out

**In:**

- A `ComputeGraph` fresh from doc 06: nodes with `backend: Some(_)` assigned,
  fusion applied (some nodes orphaned, some replaced by `FusedQKV`/`SwiGLU`
  style ops), shapes and dtypes final.
- `CParams.n_ctx` — riding inside `GraphParams` — which sizes every KV
  region (§2.5).
- Registered weights, already inside the backends' registries (doc 03) — the
  allocator's CPU pool is the same object weight registration went through
  (`alloc.rs:135-137` delegates to `self.cpu.register_weight`).
- Host data for inputs: `&[u32]` token ids, positions, tail ids
  (`models/qwen2/graph.rs:522-536`).

**Out:**

- `node_to_buf: HashMap<NodeId, BufRef>` — the answer to "where does node N's
  output live". The scheduler consumes it for every node of every split
  (`scheduler.rs:231-257`).
- `kv: HashMap<layer, [BufRef; 2]>` + `persistent: Vec<PersistentBuf>` — the
  KV regions with stable names like `"kv.7.k"` / `"kv.7.v"`, exposed to
  backends through the `KvProvider` trait (`backend.rs:12-19`,
  `alloc.rs:646-650`).
- `cross: HashMap<NodeId, BufRef>` — split-boundary staging copies, filled
  lazily during the first execute and rewritten on later ones.
- Filled input buffers, ready before the scheduler's first node.

**Shapes at this stage** (Qwen2.5-0.5B, decode step, CPU path): inputs
`[1,1,1,1]` f32-carried I32; hidden-width buffers `[896,1,1,1]` = 3.5 KB;
FFN-width `[4864,1,1,1]` = 19.5 KB; KV regions `[128, 4096, 1, 1]` = 2.1 MB
each, two per layer, 24 layers ≈ 100 MB total. All f32.

### 3.2 Key code

**Excerpt 1 — the allocator's fields** (`src/graph/alloc.rs:21-41`). Every
map below reappears in the walk; the comments record the ownership rules.

```rust
pub struct GraphAllocator {
    cpu: CpuBackend,
    #[cfg(target_os = "macos")]
    metal: Option<super::metal_backend::MetalBackend>,
    #[cfg(feature = "cuda")]
    cuda: Option<super::cuda_backend::CudaBackend>,
    node_to_buf: HashMap<NodeId, BufRef>,
    /// Cross-backend copies for the CURRENT graph (split-boundary staging):
    /// node → buffer on the consuming split's backend. NOT part of the node's
    /// canonical assignment — node_to_buf must stay re-executable (a remap
    /// would break the next execute of a reused graph, whose producing split
    /// would find its buffer on another backend). The same staging buffer is
    /// rewritten on every execute (no per-step allocation).
    cross: HashMap<NodeId, BufRef>,
    /// (backend, pool id) → last exec index it stays alive until
    buf_alive: HashMap<(Backend, usize), usize>,
    /// per-layer KV persistent regions: [k, v]
    kv: HashMap<usize, [BufRef; 2]>,
    /// All persistent regions (never freed).
    pub persistent: Vec<PersistentBuf>,
}
```

Note the deliberate separation of `node_to_buf` (canonical, re-executable)
from `cross` (staging). A naive design would *move* a node's buffer to the
consuming backend — which would break execute #2 of a reused graph, when the
producing split needs its buffer back where it was.

**Excerpt 2 — liveness in build order, with inputs and outputs pinned**
(`alloc.rs:179-210`). This is the code that bug G3 rewrote; the comment is
the tombstone.

```rust
// The scheduler executes nodes in BUILD order (node id order — the
// builder appends sources before consumers), so liveness must use the
// same order: topo_order() can reorder srcless nodes (kv_load) ahead,
// which would let a later consumer's buffer reuse clobber an input the
// scheduler has not yet read (G3 tail get_rows regression). Validate
// acyclicity, but keep build order.
graph.topo_order()?;
let order: Vec<NodeId> = (0..graph.n_nodes()).collect();
let n = graph.n_nodes();

let mut exec = vec![0usize; n];
for (i, &id) in order.iter().enumerate() {
    exec[id] = i;
}
let mut last_use = exec.clone();
for (i, &id) in order.iter().enumerate() {
    for &s in &graph.node(id).src {
        if last_use[s] < i {
            last_use[s] = i;
        }
    }
}
for &o in &graph.outputs {
    last_use[o] = order.len();
}
// Inputs are filled on the host BEFORE execution starts, so every
// input buffer is live at fill time; liveness (which tracks execution
// order) must never reuse an input's buffer for another input — the
// later fill would clobber the earlier one. Treat inputs like outputs.
for &i in &graph.inputs {
    last_use[i] = order.len();
}
```

`last_use` starts as each node's own position, so a node with no consumers
(a fusion orphan) has `last_use == exec` and will get no buffer; a node read
by many consumers ends at the latest reader.

**Excerpt 3 — the main walk: sweep, then per-node decision**
(`alloc.rs:221-239`, KV arm; the generic arm at 301-309 is three lines of
"alloc if alive").

```rust
for (i, &id) in order.iter().enumerate() {
    self.sweep(i);
    let node = graph.node(id);
    let backend = node.backend.unwrap_or(Backend::CPU);
    match node.op {
        Op::KvcacheStore { layer } | Op::KvcacheLoad { layer } => {
            let pair = self.ensure_kv(layer, backend, node.n_elements());
            // the node's buffer = the K region
            self.node_to_buf.insert(id, pair[0]);
        }
        Op::FusedQKV { layer } => {
            // fused decode QKV: also needs the layer's persistent KV
            // regions (the kernel stores K/V), but its output is a
            // normal concat buffer (q|k|v), not the K region.
            let kv_elems = match &node.meta {
                NodeMeta::FusedQkv(m) => m.kv_elems,
                _ => node.n_elements(),
            };
            self.ensure_kv(layer, backend, kv_elems);
            if last_use[id] > i { /* … ordinary buffer for the concat … */ }
        }
```

`sweep(i)` (`alloc.rs:410-422`) collects every `buf_alive` entry whose deadline
passed (`al < i`), removes it, and hands the id to `free_in_pool` — which
pushes it onto the backend's free list. Nothing is *deallocated*; "free" here
means "return to the recycling pool", which is why the next `alloc_in_pool`
of the same size is a zero-cost reuse (plus one zero-fill on CPU).

**Excerpt 4 — the in-place alias arm** (`alloc.rs:273-299`). The two guards
and the live-range extension, exactly as argued in §2.4.

```rust
// In-place elementwise transforms: alias the input buffer
// (llama.cpp executes rope/silu in place). Same-backend
// aliasing avoids a host-side copy between a pending GPU
// producer and this kernel — it reads/writes the buffer the
// producer wrote, in kernel order. Cross-backend inputs get
// a fresh buffer: the producer completed before the split
// boundary, so the backend's host copy is safe there.
if last_use[id] > i {
    let in_ref =
        self.node_to_buf.get(&node.src[0]).copied().ok_or_else(|| {
            format!("in-place op src buffer missing (node {id})")
        })?;
    // alias only when the input's sole consumer is this op
    // (in-place overwrites the input) AND it is on the same
    // backend
    if in_ref.backend == backend && n_consumers[node.src[0]] == 1 {
        self.node_to_buf.insert(id, in_ref);
        // the aliased input must stay alive through this
        // node's consumers
        last_use[node.src[0]] = last_use[node.src[0]].max(last_use[id]);
    } else {
        let size = node.n_elements();
        let pid = self.alloc_in_pool(backend, size);
        self.buf_alive.insert((backend, pid), last_use[id]);
        self.node_to_buf.insert(id, BufRef { backend, id: pid });
    }
}
```

The `else` branch matters as much as the `if`: a cross-backend or
multi-consumer input silently falls back to a normal buffer. Aliasing is an
optimization with strict preconditions, never an assumption.

**Excerpt 5 — persistent region creation** (`alloc.rs:382-403`).

```rust
/// Per-layer KV persistent regions (K and V), created on first use on the
/// layer's assigned backend.
fn ensure_kv(&mut self, layer: usize, backend: Backend, size: usize) -> [BufRef; 2] {
    if let Some(&pair) = self.kv.get(&layer) {
        return pair;
    }
    let k = self.alloc_persistent(&format!("kv.{layer}.k"), backend, size);
    let v = self.alloc_persistent(&format!("kv.{layer}.v"), backend, size);
    self.kv.insert(layer, [k, v]);
    [k, v]
}

/// Allocate a persistent (never-freed) region on a backend.
pub fn alloc_persistent(&mut self, name: &str, backend: Backend, size: usize) -> BufRef {
    let id = self.alloc_in_pool(backend, size);
    self.persistent.push(PersistentBuf {
        name: name.to_string(),
        backend,
        id,
    });
    BufRef { backend, id }
}
```

Two quiet details: the pair is created *once per layer per process* — the
`if let Some` early-return is what makes rebuilds zero-copy (§2.5) — and
`alloc_persistent` never touches `buf_alive`, so no sweep can ever free it.
The region is also sized on **first use only**: if a later graph asked for a
different size, it would silently get the old buffer — one reason `n_ctx`
must stay consistent across a run (§3.3, question 3).

**Excerpt 6 — I32 input filling** (`alloc.rs:436-446` plus the routing tail
of `fill_input_impl`, `alloc.rs:454-465`).

```rust
/// Fill an I32 input (token ids / positions). Stored as `f32::from_bits`
/// patterns — exact for |v| < 2^24.
pub fn fill_input_i32(
    &mut self,
    graph: &ComputeGraph,
    name: &str,
    data: &[u32],
) -> Result<(), String> {
    let bits: Vec<f32> = data.iter().map(|&v| f32::from_bits(v)).collect();
    self.fill_input_impl(graph, name, &bits)
}
```

```rust
let id = graph.inputs.iter().copied()
    .find(|&i| graph.node(i).name == name)
    .ok_or_else(|| format!("no input node named '{name}'"))?;
let br = self.node_buffer(id)
    .ok_or_else(|| format!("input '{name}' has no buffer (not allocated)"))?;
match br.backend {
    Backend::CPU => self.cpu.write_host(br.id, data),
    // … Metal / CUDA arms call the same write_host on their pools …
```

Inputs are found **by name**, not position — the graph is rebuilt between
prefill and decode, so node ids may shift, but the names `"token_ids"` /
`"positions"` / `"tail_ids"` are stable API.

**Excerpt 7 — the copy that must be fresh** (`alloc.rs:608-635`, the tail of
`copy_across`).

```rust
let data = self
    .copy_to_cpu(node_id)
    .ok_or_else(|| format!("node {node_id} host read failed"))?;
let new_id = self.alloc_fresh_in(dst_backend, data.len());
// write into the destination backend pool
match dst_backend {
    Backend::CPU => self.cpu.write_host(new_id, &data)?,
    // … Metal / CUDA arms identical …
}
self.cross.insert(
    node_id,
    BufRef { backend: dst_backend, id: new_id },
);
```

`copy_to_cpu` is safe *here* — this code only runs at a split boundary,
*i.e.* after the producing split was synchronized (§2.7). The same host copy
performed *inside* a split, against an unsubmitted command buffer, is the
Phase-3 bug (§3.4).

**Excerpt 8 — how backends receive the KV regions** (`backend.rs:12-19` and
the scheduler's resolution, `scheduler.rs:261-271`).

```rust
pub trait KvProvider {
    /// (k_buf_id, v_buf_id) of a layer's persistent regions on this pool.
    fn kv_pair(&self, layer: usize) -> Option<(usize, usize)>;
}
```

```rust
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
```

`execute_node` takes `kv_pair: Option<(usize, usize)>` alongside the ordinary
input ids (`backend.rs:49-55`) — the K/V regions are *not* the node's `src`
inputs; they are process-lifetime siblings only KV-aware ops know about. The
CPU store kernel shows the split-brain clearly (`cpu_backend.rs:143-175`):
K is written through `out_buf` (which the allocator guaranteed is the K
region), V through the sibling id, both reached with `split_at_mut` for
disjoint mutable borrows, and positions decoded from the I32 input with
`to_bits` (`cpu_backend.rs:152-155`) — with a hard error if a position
exceeds `n_ctx` (`cpu_backend.rs:169-171`), never a silent overflow.

**Excerpt 9 — the pool's two remaining flavors** (`cpu_backend.rs:121-131`;
`alloc_buffer` was already shown in §2.3, so this is just its siblings).

```rust
fn free_buffer(&mut self, id: usize) {
    if !self.free.contains(&id) { self.free.push(id); }   // "free" = recycle
}
fn alloc_fresh(&mut self, size: usize) -> usize {
    // never recycled from the free list (see Backend::alloc_fresh)
    self.buffers.push(vec![0.0f32; size]);
    self.buffers.len() - 1
}
```

(Metal's pool is the same shape with `MTLBuffer` lengths in bytes,
`metal_backend.rs:292-314`, except recycled buffers are *not* re-zeroed —
kernels fully overwrite their outputs, and the driver zero-fills only new
allocations.)

**Excerpt 10 — GraphCache: params-only reuse, allocator kept**
(`cache.rs:47-73`).

```rust
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

/// Store a freshly built graph. The allocator is kept (KV regions persist);
/// its liveness mapping is recomputed by the caller via `alloc_graph`.
pub fn replace_graph(&mut self, mut graph: ComputeGraph, params: GraphParams) {
    graph.uid = NEXT_GRAPH_UID.fetch_add(1, Ordering::Relaxed);
    self.graph = Some(graph);
    self.prev_params = Some(params);
}
```

Note what is *absent* from `params_match`: `n_past` (how many tokens are
already in the KV cache). Positions are **data, not structure** —
ARCHITECTURE.md invariant 1 — which is the precondition for the whole reuse
scheme: if topology depended on `n_past`, every decode step would rebuild.

### 3.3 Design choices (why this shape and not another)

**Why does the allocator own the backend pools? Why is the scheduler a pure
orchestrator?** (`GRAPH-REFACTOR-PLAN.md` deviation 11.) Three forces point
the same way. *One id space*: buffer ids appear in `node_to_buf`, in split
input lists, in kernel launches, and in captured CUDA Graphs; if two
components each held a pool, every id would need a "whose?" qualifier and
every bug a suspect. *One lifetime*: pools must live exactly as long as the
cached graph (rebuilding them per step would re-create hundreds of GPU buffer
objects per token); the cache owns the graph, so the cache owns the pools,
through the allocator. *One registration path*: weights land in the same CPU
pool object (`register_weight`), which is how "does this node's weight live on
the GPU?" becomes a simple registry query during assignment (doc 06). The
scheduler keeps only orchestration logic — assign, split, copy, run — and
borrows the backends mutably through `alloc.cpu_mut()` / `alloc.metal_mut()`
at execution time (`scheduler.rs:274-292`).

**Why is buffer reuse safe here when it broke twice?** Because each bug was a
missing *precondition*, not a flaw in liveness itself, and the fixes wrote the
preconditions into the code:

1. Reuse is only sound if liveness is computed in **the same order the
   executor runs the nodes**. The G3 bug computed it in Kahn topological
   order while execution used build order (§3.4). Now both are build order —
   literally `0..n_nodes` — so "dead after position i" means the same thing to
   both components.
2. Reuse must respect **who fills memory outside the node walk**. Inputs are
   host-filled before execution; outputs are read after. Both classes are
   pinned to `order.len()` and never recycled (§2.2).
3. Reuse must respect **who allocates outside the build loop**. Split-boundary
   staging bypasses the free list via `alloc_fresh` (§2.7).
4. In-place sharing (aliasing) is a *stronger* claim than reuse — two live
   tensors, one buffer — so it carries its own extra guards: sole consumer,
   same backend (§2.4).

The general lesson: sharing memory is safe exactly when every writer's
schedule is known and every reader is accounted for in one order. minfer now
has that schedule (build order) and that accounting (last_use + pins + fresh
staging), enforced by code, comments, and the unit tests of §4.

**Why size KV by `n_ctx` and not the model's `max_seq_len`?** The regions are
allocated *once* and their size is `n_kv_embd × n_ctx` — so `n_ctx` is the
single biggest memory decision in the process, and for Qwen3-4B the wrong
answer was 12.1 GB: the single-shot CLI used to pass `max_seq_len = 40960`
straight through, giving `36 layers × 2 regions × 40960 × 1024 × 4 B = 12.1
GB` of Metal shared buffers for a 10-token prompt. The damage was not resident
memory (peak RSS was identical, ~2.1 GB, at 4096 and 40960) but the Metal
driver's one-time first-submit setup, which scales with total buffer bytes:
289 ms at n_ctx 40960 vs 106 ms at 4096 — a 3× tax on the *first* token
(`docs/PERF-QWEN3-4B-VS-LLAMACPP.md` §2). The fix put the choice in the CLI's
hands (`--n-ctx`, default 4096; doc 01 covered that side) and clamped it:
`main.rs` computes `ctx = params.n_ctx.max(input_ids.len())` — a long prompt
must never overflow the notepad — and the model clamps again with
`n_ctx.min(max_seq_len)` (`src/models/qwen2/graph.rs:392`, `main.rs:744-749`).
One more consistency requirement hides here: because `ensure_kv` sizes on
*first use only* (excerpt 5), prefill and decode must pass the **same** `n_ctx`
so the regions created during prefill are correctly sized for every decode
step — the comment "Computed ONCE so prefill and decode size the same KV
regions" (`main.rs:748`) pins that.

**Why are inputs f32 buffers at all?** Because the pool is uniform and the
two numeric paths agree on f32 as the interchange format: GPU backends read
f32 activations directly (convention #1 in `AGENTS.md`; CPU quantizes
activations to Q8_0 *at the matmul*, inside the kernel), so f32 is already
the lingua franca of every buffer. Integer inputs ride as bit patterns
(§2.6). The alternative — per-dtype pools — would multiply allocator state,
copy paths, and backend code for the sake of two `[nt]`-element integer
buffers per graph; the bit-pattern trick costs one `from_bits`/`to_bits` pair
per element and one explanatory comment.

### 3.4 Pitfalls & invariants

**Bug 1 — the Phase-3 KV-corruption bug (never host-copy a GPU-pending
buffer).** After doc 06's assignment, a GPU-resident layer's RoPE input
sometimes needed a copy: the original allocator materialized cross-backend and
in-place inputs through a host `copy_in`. On Metal, though, one split's
kernels are *encoded* into an `MpsCommandBuffer` as they execute — and only
*submitted* at the split boundary (`metal_backend.rs:146-157`,
`metal_backend.rs:1023-1025`). A host copy enqueued mid-split therefore read
the buffer's *old* contents: freshly allocated Metal memory, i.e. **zeros**.
The copy captured zeros, RoPE dutifully rotated them, `KvcacheStore` wrote
them into the layer's persistent region — and the whole KV region was zeros,
so every attention read garbage and the output was unintelligible. The
recorded fix (`GRAPH-REFACTOR-PLAN.md` deviation 18; ARCHITECTURE.md invariant
4): same-backend in-place ops **alias** their input (the read and the write
happen inside the same command buffer, in kernel order, so coherence is
guaranteed by the GPU's own queue), and cross-backend inputs get a **fresh
buffer** — safe, because the producer split was synchronized at the boundary
before any copy runs (excerpt 7). After the fix, all 437 nodes of the 0.5B
graph ran correct on a single command buffer. The invariant, verbatim from
the architecture doc: *"Never host-copy a GPU-pending buffer: a host `copy_in`
of a producer that is encoded but not submitted reads stale data."*

**Bug 2 — the G3 liveness-order bug (liveness must follow execution
order).** The allocator originally computed liveness over `topo_order()` — a
Kahn topological sort — while the scheduler executes in build order. Both are
valid topological orders, but they are not the *same* order: Kahn's queue
front-loads every source-less node (inputs, `kv_load`) and can reorder two
independent nodes relative to each other. In the G3 tail-shrink graph, that
reordering made the allocator believe the residual stream `h` was dead
*earlier* than execution would prove — so when the attention node was
allocated (after `h`'s supposed last use, in Kahn order), the sweep had
already recycled `h`'s buffer to it. Execution then ran in build order: the
attention node wrote its output into what was still `h`'s buffer, and the
later `get_rows(h)` read the attention output instead of the residual —
logits off by **21.79** (`GRAPH-REFACTOR-PLAN.md` deviation 22). The fix is
excerpt 2: call `topo_order()?` purely to reject cycles, then compute
liveness over `0..n_nodes` — the order the scheduler actually runs. (Small
forensics note: the doc comment on `topo_order` still says "used by the
allocator" (`graph/mod.rs:151-153`) — a stale leftover; `alloc.rs:179-185` is
authoritative.)

**Bug 2b — input buffers are never freed.** Same fix series, complementary
rule (deviation 23): inputs are host-filled *before* execution, but liveness
only tracks consumers *during* execution — so `token_ids` (last consumer: the
embedding, position 1) looked dead long before `positions` was filled, the
two inputs' buffers were reconciled into one, and the later fill clobbered
the earlier (recorded as "the `embedding_and_rope` regression: token_ids
overwritten by positions"). Every embedding then gathered garbage rows. Fix:
`last_use[i] = order.len()` for all inputs (excerpt 2's final loop) — the
cost is a few dozen bytes pinned per graph; the benefit is that fill order no
longer matters.

**The remaining invariants, in one list** (each traceable to a §2 section):

- Aliasing requires sole-consumer **and** same-backend; everything else
  allocates normally (§2.4).
- Inputs and outputs are pinned to the end of the execution; persistent
  regions are outside `buf_alive` entirely (§2.2, §2.5).
- Split-boundary staging always allocates fresh; it rejoins the free list only
  at rebuild (§2.7).
- KV positions are data: the region is sized `n_kv_embd × n_ctx`, and a
  position ≥ `n_ctx` is a loud error, not an overflow (`cpu_backend.rs:169-171`,
  plus the pre-flight assert `maxp < n_ctx` in `models/qwen2/graph.rs:415-419`).
- Dead nodes get no buffer and the scheduler skips them — so adding an op the
  fusion pass orphans cannot corrupt memory, it just does nothing
  (`scheduler.rs:228-233`).

## 4. Observe & verify

- **`cargo test` — the allocator's own unit tests** (`src/graph/alloc.rs:652-778`):
  `liveness_reuses_buffers_along_chain` and `parallel_chains_do_not_share`
  assert the two liveness behaviors of §2.2; `kv_regions_two_per_layer`
  asserts store and load share the K region, V is a sibling, and exactly two
  persistent regions exist for one layer; `cycle_graph_allocation_fails`
  proves the acyclicity check is live. `src/graph/cache.rs:203-217` pins
  "persistent regions survive rebuilds". Filter with
  `cargo test liveness` / `cargo test kv_regions`.
- **`MINFER_TRACE=/tmp/t.json ./target/release/minfer model.gguf "Hello"`** —
  records per-node real data for the viz page; input nodes appear
  host-filled, and you can watch a buffer's contents change across the nodes
  that share it.
- **`MINFER_GRAPH_DUMP=/tmp/d …`** — dumps logits and the KV regions after a
  run; the KV dump is exactly the persistent regions of §2.5, so zeros there
  would reproduce bug 1's symptom.
- **`--dump-graph` / `--dump-graph-json`** — re-runs build → assign → fusion
  and exports the 437-node graph with backend colors; the node ids it shows
  are the build order the allocator's liveness uses.
- **`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1`** — flips the fusion
  toggles, which changes `cparams`, which forces a graph rebuild with the
  *same* allocator — a hands-on way to watch KV regions survive a rebuild
  (§2.8) while the node/buffer mapping is recomputed.
- **Greedy equivalence checks** — the recorded acceptance for both bug fixes:
  `--temp 0` output identical pre/post fix, and fused-vs-unfused decode
  logits diff 0.000 (`GRAPH-REFACTOR-PLAN.md` deviations 18, 22-24, 25-26).

## 5. Cross-references

- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §4.3 (pipeline position),
  §4.4 (GraphCache), §4.5 (the invariants this doc expanded, esp. 1, 2, 4, 5),
  §7 (KV cache summary) — the compressed version of this stage.
- [`docs/GRAPH-REFACTOR-PLAN.md`](../GRAPH-REFACTOR-PLAN.md) §3.3 (original
  allocator design — note where the implementation diverged: per-backend
  pools, build-order liveness, two regions instead of one `[K|V]` block),
  §17 deviations 11 (pool ownership), 14 (allocator survives rebuilds), 18
  (aliasing fix), 20 (two regions per layer), 22-23 (the G3 liveness fixes).
- [`docs/PERF-QWEN3-4B-VS-LLAMACPP.md`](../PERF-QWEN3-4B-VS-LLAMACPP.md) §2 —
  the 12 GB `n_ctx` lesson with its measurements.
- [05 — Graph build (IR)](05-graph-builder-ir.md) — where the node list,
  input names, and KV node shapes come from.
- [06 — Backend assignment and fusion](06-assign-fusion.md) — upstream:
  decides *which* backend each buffer must be allocated on, and creates the
  orphan/fused node shapes the allocator must handle.
- [08 — The scheduler: splits, copies, execution](08-scheduler-execute.md) —
  downstream: consumes `node_to_buf`/`cross`/`kv_pair`, runs the split
  boundaries whose staging rules this doc motivated.
- [11 — Attention + vec ops + KV](11-attention-vecops-kv.md) — what K and V
  actually mean and how attention reads the written prefix.
- [13 — Decode loop + graph reuse](13-decode-loop-graph-reuse.md) — the
  prefill→decode rebuild that the persistent regions are designed to survive.

← [06 — Backend assignment and fusion](06-assign-fusion.md) · [Index](./README.md) · [08 — The scheduler: splits, copies, execution](08-scheduler-execute.md) →
