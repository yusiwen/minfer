# 06 · Assign + fusion — every node picks an engine, then patterns fold

> **Stage**: [05 — Graph build: the IR and the builder](05-graph-builder-ir.md) → **assign
> backends + fuse op patterns** → [07 — Memory allocation: liveness and the KV regions](07-allocator-liveness-kv.md)
> (row 6 of the README master table: the graph leaves its "pure IR" form here and becomes
> executable, but nothing has been allocated or run yet).
> **Code**: `src/graph/scheduler.rs::assign_backends` (line 60) · `src/graph/alloc.rs::supports`
> (line 140) · `src/graph/backend.rs` (the `Backend` trait) ·
> `cpu_backend.rs` / `metal_backend.rs` / `cuda_backend.rs` (`supports_op`, `supports_fused`) ·
> `src/graph/fusion.rs::run` (line 23) · call site `src/models/qwen2/graph.rs::forward_cached`
> (line 472, mirrored in `qwen3/graph.rs`) · `src/graph/params.rs` (`CParams`).

## 1. Background — where this stage sits

Doc 05 left us with a `ComputeGraph`: a plain data structure that lists every math operation of
the transformer as a node, with edges pointing from each node to the nodes that produce its
inputs. Nothing has been computed; nothing has been allocated; the graph is a *description*, not
a program. In compiler language this description is called an **IR (intermediate
representation)** — a neutral, engine-independent way of writing down "what should be computed"
so that several different engines can later agree on how to compute it.

This stage performs the two transformations that turn that description into something an
engine can actually run:

1. **Backend assignment.** Every node gets a **backend** — one of the engine implementations
   that can execute a node: `CPU`, `Metal` (macOS GPU), or `Cuda` (`--features cuda`, NVIDIA
   GPU). A backend is more than a pile of math code: it owns its own **buffer pool** (its own
   memory for intermediate results), its own **kernels** (the small, hand-tuned functions that
   actually crunch numbers — a matmul kernel, a RoPE kernel, …), and the machinery to schedule
   that work on its device. Assignment answers one question per node: *who computes this?*
2. **Fusion.** A **rewrite pass** walks the graph and looks for small, fixed *patterns* of nodes
   that some backend knows how to compute in a single kernel — a **fused op**. When both the
   pattern matches *and* the node's assigned backend says "I have a kernel for that", the pass
   replaces the pattern with one fused node. This stage ships one pattern: `Mul(Silu(x), y)`
   folds into a single `SwiGLU` node. (The plan's second pattern, `RoPE(Add(x, b))` →
   `FusedBiasRope`, was removed once it became clear no backend would claim the capability —
   the bias+rope work ships as the builder-built decode nodes; see §2.3.)

Both transformations happen **once per graph build**, not once per token. The engine builds the
graph on the first forward (and on any rebuild), runs assign → fusion immediately after, and
then stores the finished graph in the `GraphCache`. Every decode step afterwards reuses that
exact graph and only refreshes its input *data* (doc 13). So the cost of assignment and fusion
is paid a handful of times per run, while their benefit — cheaper execution — is paid off on
every single forward.

Why does this stage exist at all? Two reasons, one per transformation.

*Without backend assignment*, the engine would have to pick an engine *while running* — and the
tempting version of that ("this kernel failed on the GPU, let's quietly redo it on the CPU") is
exactly what minfer forbids. Silent mid-run fallback makes results backend-dependent in
surprising ways, hides kernel bugs, and makes execution non-deterministic. minfer's convention
(documented in `docs/GPU_SAFETY.md`) is the opposite: the decision is made once, at build time,
and if a kernel invariant is violated at run time the run **aborts with an error** rather than
falling back.

*Without fusion*, the graph would execute every tiny operation as its own kernel launch. Each
launch has a fixed cost (on a GPU it means encoding and submitting work; on the CPU it means a
thread-pool dispatch), and each intermediate result is written to memory and read back. Elementwise
ops like SiLU and Mul are dominated by exactly those memory round trips — the arithmetic itself
is trivial. Fusing the two-launch, three-memory-trip pattern into one kernel removes a full
write + read of the intermediate buffer and one launch, per layer, per forward. §2.3 does that
arithmetic with real byte counts.

One piece of vocabulary before we dive in: a **dispatch** (or **launch**) is the act of handing
one kernel invocation to an engine — on the CPU, waking the thread pool for one operation; on
Metal, recording the kernel into a **command buffer** (a batched to-do list for the GPU; doc 08
covers submission). Dispatches are individually cheap but never free, and decode (one token per
forward) multiplies their cost by every layer of the model.

## 2. Principle — how it works and why

### 2.1 Capability-driven assignment: ask each engine, in priority order

The policy that assigns backends is one line long (we'll read it in §3.2):

> for every unassigned node: the first backend — trying **Metal, then CUDA, then CPU** — whose
> `supports_op(op, dtype)` answers "yes" gets the node; if nobody answers, the node goes to CPU.

`supports_op` is a **capability query**: a pure yes/no function from `(operation, data type)` to
`bool`, implemented *by each backend about itself*. The scheduler never consults a master table
of "which ops run where"; it just asks the engines in order. The **priority order is the order
of the checks** inside the allocator's `supports` function — Metal is asked first because, when
available, it is the fastest engine; CUDA second; CPU last, and CPU is also the landing pad
(`or(Some(CPU))`) when nothing else is enabled.

`dtype` (**data type**) matters here: nodes carry `out_dtype` (f32 for activations; i32 inputs
are stored as f32 bit patterns — a doc 05 fact). A backend that only has f32 kernels answers
"no" for anything else, and the query result reflects what that backend can *actually* execute,
not what would be nice to execute.

Two facts make GPU assignment possible at all, and both are decided before this stage runs:

- **Weights must already live on the GPU.** A matmul node assigned to Metal is useless if the
  weight tensor it needs is only on the host. The model layer therefore gates GPU participation
  all-or-nothing: `metal_on = metal_available() && Self::weights_on_gpu(model)` — either every
  matmul weight is registered on the GPU registry, or the whole graph stays on CPU
  (`src/models/qwen2/graph.rs:425-427`; the CUDA arm is the same shape at line 435). This is why
  doc 03 (weight registration) is a *precondition* of this stage.
- **The user can opt out.** `MINFER_DISABLE_MPS=1` makes `metal_available()` false
  (`src/metal.rs:2032`), so the priority question falls straight through to CPU.

And one design rule makes the assignment trustworthy: **it is a build-time decision, full stop.**
Three concrete consequences, all visible in the code:

1. **No mid-run fallback, ever.** Once a node is assigned, its buffer lives in that backend's
   pool and its kernel is that backend's kernel. If a kernel invariant is violated at execution
   (a shape mismatch, an unregistered weight, a device-limit problem), `execute_node` returns
   `Err` and the run aborts — `docs/GPU_SAFETY.md` rule 1: *"Kernel-invariant violations return
   `Err` from `execute_node` — never a silent CPU fallback."* The CPU backend even has an
   explicit arm that turns "I have no kernel for this fused op" into a loud error
   (`cpu_backend.rs:433-441`, quoted in §3.4). A quiet fallback would mask the bug the guard
   exists to catch.
2. **Splits are deterministic.** The scheduler partitions the graph into *splits* — maximal
   runs of consecutive nodes on the same backend (doc 08) — and copies data across backends only
   at split boundaries. Because every node's backend is fixed before execution, the split map
   and the copy set are pure functions of the graph. Nothing about them can wobble run to run.
3. **Participation is recorded in the reuse identity.** `CParams.gpu` stores "did a GPU take
   part in this graph?" Because graph reuse is params-only (equal `GraphParams` ⇒ rebuild
   skipped), the flag makes a backend toggle — MPS becoming available mid-session, or the env
   var flipping between runs — *force a rebuild* instead of silently reusing a graph whose
   backends are wrong. The comment on the field says exactly this
   (`params.rs:19-21`, quoted in §3.2).

### 2.2 What fusion is, and why elementwise fusion is a bandwidth story

**Fusion** means replacing a fixed pattern of small operations with one operation that computes
the same result. The pattern this stage actually folds today is the FFN activation of every
Qwen2/Qwen3 layer:

```
gate = matmul(normed, W_gate)        # [nt, nf]
up   = matmul(normed, W_up)          # [nt, nf]
g    = silu(gate)                    # [nt, nf]   SiLU(x) = x · sigmoid(x)
sw   = g * up                        # [nt, nf]   (Op::Mul)
out  = matmul(sw, W_down)            # [nt, d]
```

The `silu`-then-`mul` pair is the pattern `Mul(Silu(x), y)`; the fused form is one node
`Op::SwiGLU(gate, up)` — the same acronym the model cards use for this activation
(*Swi**GLU*** = SiLU-gated GLU).

Why is this worth a pass? Look at what the two kernels do to memory. Call `A` the size of one
intermediate buffer, `A = nt · nf · 4` bytes in f32. Elementwise kernels read their inputs and
write their output in full; they do ~one multiply per 4 bytes moved. That ratio (work per byte)
is called **arithmetic intensity**, and it is so low that the *memory traffic* — not the math —
sets the runtime. This is what "**bandwidth-bound**" means: you could make the arithmetic ten
times faster and barely notice.

Count the traffic of the unfused pair, per layer, per forward:

| kernel | reads | writes |
|---|---|---|
| `silu` | gate: `A` | silu-out: `A` |
| `mul` | silu-out: `A`, up: `A` | out: `A` |
| **total** | **3A** | **2A** |

The fused `swiglu` kernel reads gate + up and writes out once:

| kernel | reads | writes |
|---|---|---|
| `swiglu` | `2A` | `A` |
| **total** | **2A** | **A** |

Fusion removes **one full write plus one full read of the intermediate buffer = 2A bytes per
layer**, and one dispatch. Now put real numbers on it (Qwen2.5-0.5B: hidden 896, intermediate
`nf = 4864`, 24 layers):

- **Prefill** (say a 512-token prompt, `nt = 512`): `A = 512 × 4864 × 4 B ≈ 9.96 MB`, so 2A ≈
  **19.9 MB saved per layer → ≈ 478 MB of memory traffic avoided in one prefill**, plus 24
  fewer dispatches. On a GPU moving tens of GB/s for scattered elementwise work, that is real
  time.
- **Decode** (`nt = 1`): `A = 19.4 KB`, so 2A ≈ 39 KB per layer → under 1 MB across the model —
  *bandwidth is irrelevant here*. The decode win is the **dispatch**: every Metal pass costs a
  host-side encode (`MINFER_OP_PROFILE=1` prints this cost per op; metal_backend.rs:224-242),
  and at one token per forward there is nothing else to hide it behind.

That decode asymmetry is why minfer has *two* fusion mechanisms, and keeping them apart is the
clearest way to understand this stage:

- **The FusionPass** (this doc) folds *general* patterns — the SwiGLU pair — on **any** graph,
  prefill or decode, CPU or GPU. It is small, safe, and always runs.
- **The builder's decode fusions** (doc 05 introduced them; this doc explains the division of
  labor) go much further but only for `nt == 1` on GPU: `Op::FusedQKV` replaces 3 matmul + 3
  bias + 2 rope + 2 KV-store dispatches with 2 dispatches (**10 → 2**, measured ~+11% decode
  throughput on 0.5B), and `Op::FusedFFN` replaces 2 matmul + silu + mul with 2 dispatches
  (**4 → 2**, ~+3%; `docs/COMPUTE-GRAPH-DESIGN.md` §17 rows G4/G5). Those are *built* as single
  nodes because they need special kernels with unusual shapes (a concat weight
  `blk.{i}.attn_qkv`, an in-place offset swiglu), not because a pattern matcher found them.

### 2.3 The rewrite, node by node

The SwiGLU rewrite on one FFN fragment. Note what the pass does *not* do: it does not delete the
old `silu` node — it merely stops referencing it. The node becomes an **orphan** (no consumers),
and the *next* stage (the allocator, doc 07) gives orphans no buffer, so the executor skips them
(§3.4). The pass itself stays a pure op-substitution:

```
BEFORE (silu and mul are two nodes, two intermediate buffers)

  normed ──► MatMul(ffn_gate) ──► Silu ──┐
                                         ├──► Mul ──► MatMul(ffn_down)
  normed ──► MatMul(ffn_up) ─────────────┘

AFTER (silu folded into SwiGLU; the Silu node is orphaned)

  normed ──► MatMul(ffn_gate) ───────────┐
                                         ├──► SwiGLU ──► MatMul(ffn_down)
  normed ──► MatMul(ffn_up) ─────────────┘

  Silu out: [nt, nf] buf written+read     Silu: still in the node list, but
  Mul  out: [nt, nf] buf written          orphaned → no buffer → never runs
```

The matcher's exact logic (`fusion.rs:48-79`, read in full in §3.2):

1. Find a node whose op is `Op::Mul` with exactly two sources.
2. Check whether either source is an `Op::Silu` (the pattern may arrive as `Mul(Silu(x), y)` or
   `Mul(y, Silu(x))` — both are handled; the *other* operand becomes `up`).
3. Ask the **assigned backend of the Mul node**: `supports_fused(FusedOp::SwiGLU)?`
4. If yes: rewrite the Mul node in place — op becomes `Op::SwiGLU`, sources become
   `[gate, up]`. Shapes are unchanged (`SwiGLU`'s output shape equals the Mul's), so nothing
   downstream moves.

The plan's second pattern, `RoPE(Add(x, b), pos)` → `Op::FusedBiasRope(base, bias, pos)`, is
**gone**. It was implemented and gated on `supports_fused(FusedOp::BiasRope)`, but no backend ever
claimed that capability (CPU, Metal and CUDA each list only `SwiGLU`), so the rewrite was dormant
by construction and no `Op::FusedBiasRope` node could exist. Rather than keep an unreachable arm,
the rule, the op and the capability tag were removed. The bias+rope work it described is covered by
the builder-built decode nodes: Metal/CUDA `attn_bias_rope_store` folds 3 biases + 2 ropes + 2 KV
stores into one kernel — a strictly more aggressive version of the same idea. The invariant that
matters is the *gating* (fusion can never invent an op its backend did not claim), not this
particular pattern: a future backend with a standalone bias+rope kernel would re-add the rule
together with its capability tag.

For contrast, the *builder-built* decode fusion (not this pass) collapses ten nodes into two:

```
DECODE (nt == 1, GPU, fuse_qkv on) — built by GraphBuilder, not by FusionPass

BEFORE:  MatMul(wq) ─ Add(bq) ─ RoPE ─┐
         MatMul(wk) ─ Add(bk) ─ RoPE ─┼─ KvcacheStore   ⇒  10 dispatches / layer
         MatMul(wv) ─ Add(bv) ────────┘
AFTER:   FusedQKV(concat matmul W_qkv) ── attn_bias_rope_store   ⇒  2 dispatches / layer
```

### 2.4 The gating story: who may fuse what

The pass never decides alone. Every rewrite is gated by the *target backend's own* answer to
`supports_fused(&FusedOp)`, and the three capability tables currently read:

| backend | `supports_fused` claims | consequence for the pass |
|---|---|---|
| CPU | `SwiGLU` | silu+mul folds on CPU too — one node, executed as one single-pass vector kernel (§3.4) |
| Metal | `SwiGLU` | silu+mul folds (the kernel is `swiglu_f32`) |
| CUDA | `SwiGLU` | silu+mul folds |

Why does the *decoder-side* `FusedQKV`/`FusedFFN` not appear in this table, even though they are
fused ops? Because they are not produced by this pass at all. The division of labor:

- **The builder** decides *topology*: when `CParams.fuse_qkv` / `fuse_ffn` are on (decode, GPU,
  concat weights available, `nf ≤ 16384` for FFN), it emits `Op::FusedQKV` / `Op::FusedFFN`
  nodes *directly* (`models/qwen2/graph.rs:93-105` and `236-262`). When they are off it emits
  the decomposed chain — **and the FFN branch deliberately builds `silu` + `mul` rather than a
  `swiglu` node**, with the in-code comment "built as silu+mul so the fusion pass folds it"
  (line 228). One builder, two shapes, one downstream pass.
- **The FusionPass** decides *local rewrites* on whatever graph it is handed — model-built or
  test-built — using only pattern + capability.

This split is what makes **double fusion impossible** by construction. Double fusion would mean
fusing an already-fused node again — e.g. wrapping `FusedQKV` in another pattern. It cannot
happen here, for three independent reasons:

1. The pass's patterns match only *decomposed* ops (`Mul`, `Silu`). Fused nodes
   (`SwiGLU`, `FusedQKV`, `FusedFFN`, …) are not triggers, so a rewritten graph matches nothing
   the second time around — the pass is **idempotent**.
2. The pass runs **exactly once** per build, at a single call site right after
   `assign_backends` (§3.2). There is no loop that could re-run it on its own output.
3. Fused nodes produced by the builder contain no sub-nodes to fold — `FusedQKV` is one node
   whose "insides" live inside a Metal/CUDA kernel, invisible to a graph pattern matcher.

And the reason gating matters at all: a fusion applied to a backend without the kernel would
produce an op *no* engine can execute — the graph would abort at run time (loudly, per the
convention above, but still uselessly). Gating at the source means the graph only ever contains
ops its assigned backend claimed.

### 2.5 The env toggles: fusion as a first-class A/B experiment

Both builder fusions are controlled by environment variables read at graph-build time
(`models/qwen2/graph.rs:462-467`):

```
MINFER_NO_FUSE_QKV=1  →  CParams.fuse_qkv = false  →  builder emits the decomposed QKV chain
MINFER_NO_FUSE_FFN=1  →  CParams.fuse_ffn = false  →  builder emits matmul + silu + mul + matmul
```

`fuse_qkv`/`fuse_ffn` are fields of `CParams`, which is part of `GraphParams`, which is the
*entire* input to graph reuse. Flip either env var and three things follow, mechanically:

1. The reuse check `try_reuse` fails (params differ) → the graph is **rebuilt** with the new
   topology. Toggling fusion can never leave a stale fused graph in place.
2. The rebuilt graph has different nodes (fused vs decomposed) — the two graphs are a valid A/B
   pair, and the experiment is cheap: no process restart, no re-loading of weights.
3. The comparison is **bit-identical**, not approximately equal. Fused kernels were written to
   produce exactly the same bits as the decomposed chain; the G4/G5 records measured
   fused-vs-unfused logit differences of **0.000** on 0.5B and 7B
   (`COMPUTE-GRAPH-DESIGN.md` §17 G4/G5), and the test `fused_qkv_matches_unfused_decode`
   asserts it (§4).

That bit-identity has one famous footnote — the `~1e-6 noise lesson` (plan doc deviation 26):
during bring-up, a test ran the *unfused* graph **without** the FusionPass, so `silu` and `mul`
executed as two separate kernels; the result differed from the fused path by ~1e-6 relative
noise (two float roundings vs one, amplified across large intermediate values) and the
bit-identity assertion "failed" for a reason that had nothing to do with the fused kernels.
The resolution is a rule, not a workaround: **the real forward path always runs FusionPass** —
the only difference between the A and B sides is which nodes the *builder* emitted, never
whether the pass ran. An unfused graph that skipped the pass would not be "more primitive"; it
would be *wrong as a reference*.

## 3. Implementation

### 3.1 Data in / data out

**In:** the freshly built `ComputeGraph` (doc 05) — every node carries `op` (with full payloads),
`src` (input node ids), `out_shape`/`out_dtype`, and `backend: Option<Backend>` where `None`
means "undecided" (`src/graph/mod.rs:92`). Plus the allocator's backend registry: which backends
were *enabled* (Metal enabled on macOS when available and all weights registered; CUDA when the
feature is on and a device exists).

**Out:** the same graph object, mutated in place:

- every node has `backend: Some(…)`,
- some `Mul` nodes are now `Op::SwiGLU` nodes with re-routed `src = [gate, up]`,
- nothing else: no node is added or removed, no shape changes, build order is untouched. The
  orphaned `Silu` nodes are still in the list — they only die at allocation time.

The subsequent stages consume exactly this: `alloc_graph` (doc 07) uses the backends to pick
pools and gives orphans nothing; `split_graph`/`execute` (doc 08) uses the backends to cut
splits and dispatch.

**Where the stage physically runs:** inside the model's `forward_cached` — the first place a
forward needs a graph. The sequence there is `try_reuse → build → register weights → enable
backends → assign_backends → FusionPass → alloc_graph → replace_graph`
(`src/models/qwen2/graph.rs:472-517`). Note the pipeline lives in *model* code, not the
scheduler: the scheduler provides `assign_backends`, but the model orchestrates, because only
the model knows whether its weights made it onto the GPU.

### 3.2 Key code

**The whole assignment policy** — `src/graph/scheduler.rs:58-69`:

```rust
/// Assign every node to the best backend that supports it (capability
/// driven via the allocator's backend registry).
pub fn assign_backends(&self, graph: &mut ComputeGraph, alloc: &GraphAllocator) {
    for node in &mut graph.nodes {
        if node.backend.is_some() {
            continue; // keep explicit assignments
        }
        node.backend = alloc
            .supports(&node.op, node.out_dtype)
            .or(Some(BackendTag::CPU));
    }
}
```

Three lines of policy, each deliberate. The loop is order-independent — each node is asked on
its own, so the result cannot depend on graph traversal order. The `continue` keeps any explicit
assignment (tests use it; the production builder assigns none). The `.or(Some(CPU))` fallback
makes the assignment *total*: every node leaves with a backend even if `supports` returned
`None`. That cannot hide an unsupported op — the CPU backend's `execute_node` has an explicit
`Err` arm for ops it has no kernel for (§3.4) — it just guarantees the *abort names the right
node* instead of crashing on an `Option` unwrap.

**Where the priority actually lives** — `src/graph/alloc.rs:139-157`:

```rust
/// Which backend supports this op/dtype (highest priority first).
pub fn supports(&self, op: &Op, dtype: crate::graph::DType) -> Option<Backend> {
    #[cfg(target_os = "macos")]
    if let Some(m) = &self.metal {
        if m.supports_op(op, dtype) {
            return Some(Backend::Metal);
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(c) = &self.cuda {
        if c.supports_op(op, dtype) {
            return Some(Backend::Cuda);
        }
    }
    if self.cpu.supports_op(op, dtype) {
        return Some(Backend::CPU);
    }
    None
}
```

The "registry" is just three `Option` fields on the allocator; `enable_metal()` / `enable_cuda()`
populate them (model code calls them only when its own weight gates passed, §2.1). Priority is
the *order of the if-blocks* — Metal first, CUDA second, CPU last. The `#[cfg]` gates mean a
non-macOS build literally contains no Metal question to ask, and a non-CUDA build no CUDA one.

**The contract every backend implements** — `src/graph/backend.rs:21-30`:

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    /// Op support by (op, dtype). `supports_fused` gates the fusion pass
    /// (Phase 4) so fused IR nodes are only produced when a kernel exists.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool;
    fn supports_fused(&self, fused: &FusedOp) -> bool;
    // ... buffer pool, execute_node, host read/write, synchronize
}
```

**A backend's self-description, CPU** — `src/graph/cpu_backend.rs:73-105`:

```rust
fn supports_op(&self, op: &Op, dtype: DType) -> bool {
    if dtype != DType::F32 {
        return false;
    }
    matches!(
        op,
        Op::Input
            | Op::Add
            | Op::Mul
            | Op::Scale(_)
            | Op::Silu
            | Op::Softmax { .. }
            | Op::RmsNorm { .. }
            | Op::QkNorm { .. }
            | Op::MatMul { .. }
            | Op::GetRows
            | Op::RoPE { .. }
            | Op::Attn { .. }
            | Op::KvcacheStore { .. }
            | Op::KvcacheLoad { .. }
            | Op::SwiGLU
            | Op::View { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
    )
}

fn supports_fused(&self, fused: &FusedOp) -> bool {
    // The SwiGLU rewrite IS applied to CPU nodes (CPU is first in the
    // fusion pass's backend list); `Op::SwiGLU` below executes it as a
    // single pass. bias+rope and batch-matmul are not fused (batch QKV
    // quantize-sharing is a Phase 5+ win).
    matches!(fused, FusedOp::SwiGLU)
}
```

The CPU claims the whole per-layer vocabulary — everything is F32-only, and the `matches!` list
*is* the CPU's capability table. Note `Op::SwiGLU` is claimed but `FusedQKV`/`FusedFFN` are not:
decode fusions are GPU-only (§2.3). The `supports_fused` answer is `true` for `SwiGLU`, so the
pass *does* fold on CPU; what "fused" means there is a single-pass kernel, shown next.

**What a "fused" SwiGLU means on CPU** — `src/graph/cpu_backend.rs:373-379`:

```rust
Op::SwiGLU => {
    // silu(gate) * up, one pass: bit-identical to the old
    // vec_silu_f32 + vec_mul_f32 pair (same formula and
    // per-element order) without its full-size temp buffer.
    crate::vec_ops::vec_swiglu_f32(out.len(), out, ins[0], ins[1]);
    Ok(())
}
```

One *node*, one pass (`vec_ops::vec_swiglu_f32`), no intermediate buffer. The earlier form ran
`vec_silu_f32` then `vec_mul_f32` through a full-size scratch copy (`out.to_vec()`); the fused
version computes `silu(gate[i]) * up[i]` in one loop and is bit-identical to that pair (same
formula, same per-element order — pinned by `vec_ops::tests::swiglu_matches_silu_then_mul`).

**Metal's table, with the decode fusions and the CUDA-only epilogue negative** —
`src/graph/metal_backend.rs:265-290`:

```rust
fn supports_op(&self, op: &Op, dtype: DType) -> bool {
    match op {
        Op::Input => true,
        Op::Add | Op::Mul | Op::Silu | Op::RmsNorm { .. } | Op::QkNorm { .. } | Op::SwiGLU => {
            dtype == DType::F32
        }
        Op::MatMul { .. } => {
            matches!(dtype, DType::F32) // activations are f32; weight type in meta
        }
        Op::GetRows | Op::RoPE { .. } | Op::Attn { .. } => dtype == DType::F32,
        Op::KvcacheStore { .. } | Op::KvcacheLoad { .. } => dtype == DType::F32,
        Op::FusedQKV { .. } | Op::FusedQkvNorm { .. } | Op::FusedFFN => dtype == DType::F32,
        Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => true,
        Op::Scale(_) | Op::Softmax { .. } | Op::BatchMatMul => false,
        // Mixed-quant decode QKV epilogue (D3-8 class 2) is CUDA-only; on
        // Metal the graph builder never emits it (qkv_epilogue_ok = false
        // without `--features cuda`), so it is never assigned here.
        Op::QkvBiasRopeStore { .. } => false,
    }
}

fn supports_fused(&self, fused: &FusedOp) -> bool {
    // swiglu_f32 is the only fusion-pass kernel. The bias+rope+store
    // capability is a build-time fused node (FusedQKV/FusedQkvNorm), not a
    // FusionPass target, so it is not advertised here.
    matches!(fused, FusedOp::SwiGLU)
}
```

The negative arms are design statements, not gaps: `Op::QkvBiasRopeStore => false` documents that
the mixed-quant decode epilogue belongs to CUDA only, and `Op::BatchMatMul => false` is a deferred
vocabulary entry. Metal *does* claim the builder's decode fusions (`FusedQKV`, `FusedQkvNorm` —
the Qwen3 per-head-norm variant — and `FusedFFN`), which is what makes the builder's `fuse_qkv`
gate safe.

**CUDA's table differs where its kernels differ** — `src/graph/cuda_backend.rs:supports_op`,
trimmed to the interesting arms:

```rust
/// Capability matrix (docs/CUDA-BACKEND-DESIGN.md §4.3): the full per-layer
/// chain runs on CUDA, including the embedding/tail gather (7e③) and the
/// decode fusions FusedQKV/QkvBiasRopeStore/FusedFFN. Scale, Softmax,
/// BatchMatMul and the Qwen3-only FusedQkvNorm have no kernels and stay on
/// the CPU backend; weight-quant eligibility is the model-level
/// all-weights-registered gate. RoPE is gated to the neox
/// (non-interleaved) layout — the only style the supported architectures
/// emit.
fn supports_op(&self, op: &Op, dtype: DType) -> bool {
    if dtype != DType::F32 {
        return false;
    }
    match op {
        Op::Input | Op::Add | Op::Mul | Op::Silu | Op::SwiGLU | /* … */
        Op::MatMul { .. } | Op::Attn { .. }
        | Op::KvcacheStore { .. } | Op::KvcacheLoad { .. } | /* … */
        Op::GetRows
        | Op::FusedQKV { .. }
        | Op::QkvBiasRopeStore { .. }
        | Op::FusedFFN => true,
        Op::RoPE { style } => matches!(style, RopeStyle::NonInterleaved),
        _ => false,
    }
}
```

The instructive line is `RoPE`: capability
can be *payload-conditional* — CUDA rotates only the `NonInterleaved` style, so a hypothetical
interleaved-RoPE node would silently route to CPU instead of producing wrong numbers. This is
`supports_op` earning its keep as a per-node query rather than a per-backend yes/no.

**The fusion pass, part 1 — dispatch** — `src/graph/fusion.rs:20-35`:

```rust
/// Run all supported fusions over the graph. Returns the number of nodes
/// rewritten. `backend_for` returns the backend a node is assigned to
/// (used to gate fusions per backend capability).
pub fn run(
    &self,
    graph: &mut ComputeGraph,
    backends: &[&dyn Backend],
    backend_of: &dyn Fn(&ComputeGraph, usize) -> Option<usize>, // node id -> backend index
) -> usize {
    let mut n = 0;
    n += self.fuse_swiglu(graph, backends, backend_of);
    n
}
```

The pass takes the backends *as trait objects* plus a closure mapping node id → index into that
slice. The indirection exists because backends are stored as `Option` fields on the allocator in
cfg-conditional order; the closure is built at the call site where that layout is known. The
`None` case of the closure means "unassigned" — and unassigned nodes are never fused (the gate
treats `None` as `false`, §2.4's promise again).

**The fusion pass, part 2 — the SwiGLU matcher** — `src/graph/fusion.rs:48-79` (inside
`fuse_swiglu`):

```rust
for id in 0..n {
    if !matches!(graph.node(id).op, Op::Mul) {
        continue;
    }
    let mul = graph.node(id);
    if mul.src.len() != 2 {
        continue;
    }
    let (s, y) = (mul.src[0], mul.src[1]);
    let is_silu = |x: usize| matches!(graph.node(x).op, Op::Silu);
    let (silu_in, gate, up) = if is_silu(s) {
        (graph.node(s).src[0], graph.node(s).src[0], y)
    } else if is_silu(y) {
        (graph.node(y).src[0], graph.node(y).src[0], s)
    } else {
        continue;
    };
    let _ = silu_in;
    // gate the fusion on the mul node's backend capability
    let ok = match backend_of(graph, id) {
        Some(bi) => backends[bi].supports_fused(&FusedOp::SwiGLU),
        None => false,
    };
    if !ok {
        continue;
    }
    // replace Mul with SwiGLU(gate, up)
    new_ops[id] = Some(Op::SwiGLU);
    if let Some(node) = graph.nodes.get_mut(id) {
        node.src = vec![gate, up];
    }
    replaced += 1;
}
```

Mechanics worth noticing: the rewrite is collected into a `new_ops` staging vector and applied
after the scan, so the matcher never walks a half-mutated graph. The rewrite keeps the *Mul's*
node id and the *Mul's* backend — consumers of the Mul (the down matmul) keep pointing at the
same id and need no edits. The new `src = [gate, up]` re-routes dataflow: `gate` is the Silu
node's *input* (skipping the dead Silu — the `silu_in` binding holds the same value, which is
why the source discards it with `let _ =`), `up` is the other operand. Shapes never change
because `SwiGLU`'s output shape equals the Mul's by definition.

**The second matcher is gone.** The old `fuse_bias_rope` (`RoPE(Add(x, b))` → `Op::FusedBiasRope`)
and its `FusedOp::BiasRope` capability tag were removed (§2.3): no backend claimed the capability,
so the matcher was unreachable code. `run()` now composes exactly one rewrite.

**The call site that wires it all together** — `src/models/qwen2/graph.rs:472-517`, trimmed:

```rust
if !cache.try_reuse(&params) {
    let mut graph = Self::build(model, &params);
    let sched = BackendScheduler::new();
    {
        let alloc = cache.alloc();
        Self::register_graph_weights(model, alloc);
        #[cfg(target_os = "macos")]
        if metal_on { alloc.enable_metal(); }
        #[cfg(feature = "cuda")]
        if cuda_on { alloc.enable_cuda(); }
        sched.assign_backends(&mut graph, alloc);
        // fusion pass gated per node's assigned backend
        let backends: Vec<&dyn Backend> = { /* [cpu] + maybe metal, maybe cuda */ };
        let cuda_idx = backends.iter().position(|b| b.name() == "cuda");
        FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
            Some(crate::graph::Backend::CPU) => Some(0),
            Some(crate::graph::Backend::Metal) => Some(1),
            Some(crate::graph::Backend::Cuda) => cuda_idx,
            _ => None,
        });
        alloc.alloc_graph(&graph).unwrap();
    }
    cache.replace_graph(graph, params);
}
```

This is the stage's place in the pipeline, verbatim: reuse check → build → enable → **assign →
fuse** → allocate → store. The whole block is skipped when `try_reuse` succeeds — assignment and
fusion run *once* per `GraphParams`, which is why they can afford to be thorough. (`Qwen3Graph`
mirrors this block at `src/models/qwen3/graph.rs:419-446`.)

**The reuse identity that makes toggles work** — `src/graph/params.rs:17-35`:

```rust
/// Runtime parameters that affect graph construction.
///
/// `gpu` records whether the GPU backend participates — the backend assignment
/// is part of the built graph, so a change (e.g. MPS init between runs) must
/// force a rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CParams {
    pub n_ctx: usize,
    pub flash_attn: bool,
    pub gpu: bool,
    /// G4 decode QKV fusion enabled (part of the topology: toggling
    /// `MINFER_NO_FUSE_QKV` must force a rebuild).
    pub fuse_qkv: bool,
    /// G5 decode FFN gate+up fusion enabled (part of the topology: toggling
    /// `MINFER_NO_FUSE_FFN` must force a rebuild). Decoupled from `fuse_qkv`
    /// so A/B-ing one fusion does not flip the other.
    pub fuse_ffn: bool,
}
```

and the comparison that consumes it — `src/graph/cache.rs:47-64`:

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
```

`a.cparams == b.cparams` is derived `PartialEq` over *all six fields* — `gpu`, `fuse_qkv`,
`fuse_ffn` included. That one derived impl is why every toggle in this doc is a rebuild instead
of a silent stale-graph reuse. The dedicated test
`fuse_flags_are_part_of_the_reuse_identity` (cache.rs:141-170) flips each flag and asserts
non-reuse.

**Where fusion orphans die** — `src/graph/scheduler.rs:225-233` (doc 08's executor, forward
pointer):

```rust
if node.is_input() {
    continue; // data pre-filled by the allocator
}
// dead nodes (no consumers, not outputs) get no buffer — the
// fusion pass can orphan them (e.g. silu folded into SwiGLU);
// they are skipped, not executed
let Some(br) = alloc.node_buffer(id) else {
    continue;
};
```

The comment is the contract between this stage and the next two: the fusion pass is allowed to
leave garbage nodes in the graph *because* the allocator (doc 07) refuses to give dead nodes
buffers and the executor (doc 08) skips bufferless nodes. Cleaning them out of the node list
would mean renumbering every node id — the riskier operation by far.

### 3.3 Design choices (why this shape and not another)

**Q1: Why is `supports_op` a per-backend query instead of a global capability table?**

The obvious alternative is a central table — `SUPPORTS: [(Op, Backend, bool); N]` — that the
scheduler reads. It looks simpler and it rots faster: every new backend must be threaded into
the table, every new op must be threaded into the table, and the two edits happen in different
files with no compiler help linking them. The trait inverts that: a **capability table is
replaced by a type**. Adding the CUDA backend meant implementing `Backend` (including
`supports_op`/`supports_fused`) and registering it in the allocator — the scheduler's
assignment code has not changed for it (the CUDA enable call is cfg-gated model-side, and the
`supports` function gained one symmetric if-block). The compiler enforces completeness: a
backend that forgets to answer for an op answers `false` via its own `match` exhaustiveness
(and the enum's missing-arm error), never "table out of date". It also lets capability be
*stateful in a legitimate way* — Metal's yes depends on which kernels its GPU supports at
runtime, CUDA's RoPE answer depends on the payload style — which a static table cannot express.

**Q2: Why does fusion run *after* assignment?**

Because gating needs an answer to "fused for *whom*?", and only assignment provides it. The
pass's gate is literally `backend_of(mul_node) → supports_fused(...)`. Fuse before assignment
and the pass would have to guess a backend (fusing against Metal and CUDA "just in case") or
fuse unconditionally (producing a fused node its eventual owner cannot execute — aborting the
run later, or worse, tempting someone to add a fallback). Assignment-first means each node's
rewrite is decided by the engine that will actually run it; the graph can never contain a fused
op its owner didn't claim. There is a bonus: the order is safe in the other direction too,
because every fused op the pass can produce (`Op::SwiGLU`) is itself in all three backends'
`supports_op` — so had assignment somehow run again after fusion, it would be a no-op. The
pipeline relies on the ordering, not on that coincidence.

**Q3: Why is the fusion pass a graph→graph rewrite instead of fusing inside the kernels?**

Four reasons, in descending order of weight:

1. **Kernels stay single-purpose.** `swiglu_f32` exists because one Metal kernel computes it;
   the CPU instead runs two vec passes inside one node. Both decisions are *backend-internal*
   and can change (the plan doc calls for a single-pass CPU kernel) without touching the IR,
   the scheduler, or any other backend. Fusion-in-kernel would instead mean every backend
   re-implements pattern detection over raw buffers.
2. **The IR stays inspectable.** After this stage the graph still prints as nodes with names,
   ops and backends — which is what `--dump-graph-json`, the DOT export, the P2 trace and the
   live viz server show (§4). Debugging "why is decode slow" becomes reading a graph, not
   decoding backtrace soup.
3. **Verification becomes A/B on graphs.** Fused vs unfused is two `GraphParams` and one env
   var, executed through the same scheduler, compared bit-for-bit (§2.5). If fusion happened
   opaquely inside kernels, the "unfused" reference would be unreachable — there would be
   nothing to compare against.
4. **Downstream stages see the truth.** The allocator sizes and shares buffers for the *fused*
   graph (no buffer for the dead Silu), the splitter draws splits around the *fused* node, and
   the executor dispatches exactly what was planned. A kernel-internal fusion would make all
   three plan against a graph that lies.

**Q4: Why are fused ops (and `gpu`) part of the reuse identity?**

Reuse exists so decode steps don't rebuild the graph; it is *safe* only if equal params
guarantee an identical graph. Fusion decisions are topology: a fused decode graph and an
unfused one are different graphs by any structural measure. Since `try_reuse` compares params
*only* (deliberately — node-by-node structural comparison was rejected as unnecessary given
deterministic building, `COMPUTE-GRAPH-DESIGN.md` §6), the fusion switches must ride along in
those params or the cache could hand back a graph that contradicts the current configuration:
you'd set `MINFER_NO_FUSE_QKV=1` for your A/B run and the engine would quietly reuse the fused
graph — the experiment would show "no difference" and teach you nothing. Determinism and
A/B-ability are the same requirement viewed from two sides: identical params ⇒ identical fusion
decisions ⇒ identical topology ⇒ reuse is sound *and* toggles are observable.

**Two smaller choices worth naming:**

- *Why is priority hardcoded as if-block order instead of a sorted backend list?* With at most
  three backends, an ordered list adds an indirection (sort keys, cfg-conditional membership)
  to save nothing; the if-chain makes the priority readable at a glance and cfg-gates compile
  away cleanly. The cost — a new backend edits `supports()` once — is the same edit the
  registry needs anyway.
- *Why does the pass keep orphans instead of removing them?* Removal renumbers node ids, which
  invalidates every stored `src`, every output id, the trace bookkeeping — all to save a
  skipped iteration at execution. Leaving them costs one `continue` per orphan (§3.2, last
  excerpt) and zero risk.

### 3.4 Pitfalls & invariants

- **Assignment is total and final.** Every node exits this stage with a backend, and no stage
  after this may change it. Execution errors are `Err` + abort: the CPU backend's arm for ops
  it cannot run says so in its error text —
  `cpu_backend.rs:433-441`:

  ```rust
  Op::BatchMatMul
  | Op::FusedQKV { .. }
  | Op::QkvBiasRopeStore { .. }
  | Op::FusedQkvNorm { .. }
  | Op::FusedFFN => Err(format!(
      "op {:?} unsupported on CPU (fusion not enabled for it)",
      node.op
  )),
  ```

  ("fusion not enabled for it" — i.e. if you ever see this error, a fused node reached a backend
  that never claimed it: a gating bug, not a runtime condition. `docs/GPU_SAFETY.md` rule 1 is
  the convention; this arm is the CPU-side enforcement.)
- **The fusion pass must run — on every path, including the "unfused" one.** The deviation-26
  lesson (§2.5): a graph that skips FusionPass is not a valid reference for bit-identity
  experiments, because two-kernel silu+mul differs from one-kernel swiglu by ~1e-6 float noise.
  That is why the JSON export path (`json.rs::build_runtime_graph`, lines 50-99) re-runs
  assign + FusionPass — so a dumped graph matches what actually executed — and why the test
  helper in `fused_qkv_matches_unfused_decode` runs the pass on both sides.
- **Double-fusion is impossible by construction** (§2.4): patterns match only decomposed ops,
  the pass is idempotent, it runs once per build, and builder-fused nodes have no graph-visible
  insides. If you add a third fusion pattern, preserve all four properties.
- **The pass has exactly one rule.** `RoPE(Add)` → `FusedBiasRope` was removed (§2.3) because no
  backend ever claimed the capability. If a future backend gains a standalone bias+rope kernel,
  re-add the rule together with its `FusedOp` tag and a test that pins the negative case — the
  gating design is the invariant, and the removed rule is the worked example of what *not* to
  advertise.
- **Fusion never changes shapes or node count.** Orphans stay, ids stay, shapes stay. Anything
  that *does* change the node set (decode fusions) happens in the builder, where shapes are
  computed with full context (`FusedQKV`'s `[nqt+2·nkt, 1]` output, `FusedFFN`'s `[2·nf, 1]`).
  Keep that boundary: pattern pass = local substitution, builder = structural change.
- **GPU feasibility is a precondition, not a per-node property.** `supports_op(Metal)` says
  nothing about whether the *weights* are resident; that is the model-level all-or-nothing gate
  (`weights_on_gpu`, qwen2/graph.rs:630) feeding `metal_on`, feeding `enable_metal()`. A
  backend enabled without its weights would abort at the first matmul with "weight not
  registered" — loud, but avoidable.
- **In-place aliasing interacts with fusion only indirectly** (full story in docs 07/08): the
  fused `SwiGLU` reads gate and up as separate inputs on GPU, while `Op::Silu` and `Op::RoPE`
  are the ops with in-place aliasing rules. Fusion *removes* Silu nodes (orphaning them), which
  slightly reduces the number of aliased buffers — a quiet benefit for the allocator.

## 4. Observe & verify

- **`MINFER_GRAPH_TRACE=1`** prints the split map and a per-op/per-backend node census at
  execute time (`scheduler.rs:127-144`), e.g. `op SwiGLU  backend Metal  x24` — the direct
  output of this stage: 24 folded nodes, one per layer, on their assigned backend.
- **`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1`** switch the builder's decode fusions off
  for one run; combined with the trace (or the decode throughput line) they are the A/B switch,
  and the rebuild they force is the mechanism §2.5 describes.
- **`--dump-graph-json <file>` / `--dump-graph <dot>`** rebuild the graph through
  `json.rs::build_runtime_graph` — explicitly *"build → assign → FusionPass"*
  (json.rs:50-68) — so the exported node list shows the fused ops (`"swiglu"`, `"fused_qkv"`)
  and each node's backend, i.e. a picture of this stage's output. `MINFER_TRACE` /
  the `viz` server show the same graph live, with per-node data.
- **`MINFER_OP_PROFILE=1`** (Metal) prints host-encode time per op after the run
  (metal_backend.rs:36-39, 224-242) — the dispatch cost that fusion removes, made visible.
- **`MINFER_DISABLE_MPS=1`** forces `metal_on = false` and shows the priority chain collapsing
  to CPU (everything in the trace census becomes `backend CPU`).
- **Tests** (all `cargo test`): `fusion.rs` — `swiglu_fusion_applies_when_backend_supports`
  (rewrite happens, `src == [gate, up]`), `swiglu_fusion_skipped_when_backend_does_not_support`
  (unassigned node ⇒ no fusion); `vec_ops::tests::swiglu_matches_silu_then_mul` (the CPU
  single-pass kernel is bit-identical to the old silu+mul pair);
  `cache.rs::fuse_flags_are_part_of_the_reuse_identity` (flag flip ⇒ no reuse);
  `qwen2/graph.rs::fused_qkv_matches_unfused_decode` (fused nodes present iff gated on, logits
  **bit-identical**, plus per-layer output comparison); the tail-reduction test asserts the
  SwiGLU node exists and consumes the gate/up matmuls (qwen2/graph.rs:1497-1502) — the pass is
  exercised on every graph-level test run, not just the fusion unit tests.

## 5. Cross-references

- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §4.3 (the assign → fuse → alloc → execute
  pipeline), §4.5 (invariants 4–5: aliasing, dead nodes), §5.1–5.3 (Backend trait, selection
  rules, GPU safety) — the condensed version of this doc.
- [`docs/COMPUTE-GRAPH-DESIGN.md`](../COMPUTE-GRAPH-DESIGN.md) §5 (fusion rules incl. the
  deferred BatchMatMul), §7 (Metal per-op mapping — where `Op::SwiGLU`'s kernel comes from),
  §17 rows G4/G5 (decode fusion measurements: 10→2 and 4→2 dispatches, +11%/+3% on 0.5B,
  fused-vs-unfused diff 0.000) and deviation 26 (the ~1e-6 FusionPass-must-run lesson).
- [`docs/GPU_SAFETY.md`](../GPU_SAFETY.md) — the Err-not-fallback convention this stage's
  build-time assignment exists to honor.
- Doc 05 (builder: where decomposed vs fused topologies come from) · doc 07 (allocator: what
  happens to fusion orphans) · doc 08 (scheduler: splits and cross-backend copies that
  assignment makes deterministic) · doc 13 (decode loop: why assign+fuse run once, not per
  token) · doc 14/15 (Metal/CUDA: the kernels behind the capability tables).

← [05 — Graph build: the IR and the builder](05-graph-builder-ir.md) · [Index](./README.md) · [07 — Memory allocation: liveness and the KV regions](07-allocator-liveness-kv.md) →
