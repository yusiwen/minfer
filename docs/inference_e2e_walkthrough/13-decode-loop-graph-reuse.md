# 13 · The decode loop and graph reuse

> **Stage**: the sampler picks the first token (doc 12) → **this stage: the
> autoregressive decode loop — one token per forward, with the compute graph
> rebuilt once and then reused for every step** → the same loop runs unchanged
> on the GPU backends (docs 14–15, which swap out the kernels *inside* the
> loop, never the loop itself).
> **Code**: `src/main.rs` (decode loop (`main.rs:1727-1791`, the `while generated.len() < params.n_predict` loop), the repeat window `REPEAT_LAST_N` (`conversation.rs:426`), loop setup
> `main.rs:1553-1566` (the decode-loop setup), `is_stop_token` `main.rs:1868-1870`), the cached forward
> `src/models/qwen2/graph.rs::forward_cached` (`src/models/qwen2/graph.rs:450-466`), the reuse
> decision `src/graph/cache.rs::try_reuse` `try_reuse` (`cache.rs:69-85`), the params
> `src/graph/params.rs` (`GraphParams` `params.rs:88-97`), the allocator's
> persistent KV regions `src/graph/alloc.rs` (`alloc_graph` `alloc.rs:513`,
> `ensure_kv` `alloc.rs:995`), the per-step execution
> `BackendScheduler::execute` (`scheduler.rs:137-473`), and the
> multi-turn session `src/conversation.rs` (`user_turn` `conversation.rs:757-764`,
> `generate_assistant_with_logits` `conversation.rs:1271-1278`) — lines verified at
> commit `15fa45c`.

## 1. Background — where this stage sits

By the end of doc 12, one full forward pass has happened. The prompt was
rendered, tokenized, and pushed through the compute graph in a single
**prefill** step (doc 09): every layer's math ran over all `nt` prompt tokens
at once, each layer's K and V projections were written into the graph
allocator's persistent **KV regions** (doc 07), and the graph returned the
logits — one score per vocabulary entry — for the *last* prompt token only.
The sampler then turned those 151,936 scores into exactly one token id. What
the process now holds is a curious half-finished sentence: the prompt tokens,
which the model has *read*, and one new token, which the model has *written*
but not yet read.

This document is about the loop that finishes the sentence. Language models
generate text **autoregressively** — *auto* ("self") + *regressive* ("moving
backward"): each new token is produced by feeding the model everything so far,
including the tokens the model itself just wrote. So the engine enters a loop:

1. take the token sampled last step (for the first step, the last prompt
   token's logits were already produced by prefill),
2. run one forward pass whose only input is that single token, at the next
   free slot in the KV cache,
3. sample one token from the resulting logits,
4. append it, print it, and check whether generation should stop.

Step 2 is a full transformer forward — 24 layers, every weight matrix — but
for **one** token instead of `nt`. That single-token forward is called a
**decode step**, and the loop that repeats it is the **decode loop**. In
`src/main.rs` it is a plain Rust `while` loop (`main.rs:1727-1791`, the decode loop); everything
dramatic about LLM inference — chatbots streaming words one at a time — is
this loop running hundreds of times per second.

Two facts make this loop interesting enough for its own document.

**Fact one: a decode step is cheap in a very specific way.** Every matmul
still reads its *entire* weight matrix (there is only one token's activation
to multiply against each row, so nothing about the weights can be skipped) —
that makes decode **memory-bound**: its speed is set by how fast RAM streams
weights to the compute units, not by how fast they multiply (doc 10 §2.1
walks the arithmetic). But everything else shrinks: the matmul *arithmetic*
drops from `nt` token-rows to 1, and attention (doc 11) drops from the
quadratic all-pairs work of prefill to a single query scanning the KV window
built so far. A decode step costs roughly "one prefill token's worth of
matmul work plus one attention scan of `nkv` cached keys" — no more.

**Fact two: the graph is rebuilt exactly once, then reused for free.** Recall
from docs 05–08 what one forward costs *besides* the math: build a few hundred
`CNode` structs describing the topology, walk them to assign each a backend,
run the fusion pass, and run the liveness allocator to hand every node a
buffer. Doing all of that *per decode step* would be pure overhead — the
topology does not change from step to step. minfer avoids it with a design
rule that shapes the whole codebase: **the graph topology is a deterministic
function of `GraphParams`, and `n_past` (how much KV is cached) is not one of
those params — it is input *data***. Equal params ⇒ identical graph ⇒ the
cached graph, its backend assignment, its buffers, and above all its KV
regions can be reused as-is; the only thing that changes per step is a handful
of bytes written into the input nodes. The reuse decision itself is a
six-field struct comparison in `GraphCache::try_reuse` (`cache.rs:69-85`) —
no graph traversal, no node-by-node diff.

That rule is why the KV regions must live *outside* the graph, in an allocator
owned by the cache (doc 07's contract). The prefill graph (built for `nt`
tokens) and the decode graph (built for 1 token) are *different* graphs —
different buffer shapes, a different `gtype`, decode-only fused ops on the
GPU — so switching from prefill to decode does force one rebuild. The rebuild
throws away the node list and the liveness mapping, but the allocator that
owns the KV regions survives, and the regions are addressed by position, not
by step. All the prompt's context is still there, and the very first decode
step reads it.

The loop also decides **when to stop**. Three gates live in (or beside) the
loop: the sampled token is the end-of-text sentinel `is_stop_token` (`main.rs:1749-1751`),
the generated *byte stream* now ends with a user-supplied stop string
`match_stop_suffix` (`main.rs:1761-1768`), or the `-n` token cap is reached (the `while` condition
itself, the loop head (`main.rs:1727`). Doc 12 introduced these; here we see them as the
loop's control flow.

Finally, there is a longer-timescale version of the same trick: a multi-turn
chat session (`--cnv`, `src/conversation.rs`) keeps one token stream in the KV
across *whole conversations*. Each user turn prefills only the **delta** — the
few tokens of the new message — at the next free position, and the decode loop
resumes on the same cached graph. A ten-turn conversation never re-prefills
the previous nine turns; it pays O(delta) per turn instead of O(full history).
Same invariant, larger loop: positions are data, the KV is append-only, and
the graph just keeps getting reused.

What would break without this stage? Nothing downstream exists: no second
token, no streaming, no chat. And without *graph reuse*, every one of the
hundreds of decode steps would pay the build → assign → fuse → allocate tax
again — the plan document estimated the wasted work as "recomputes topology
every step + CPU scratch reallocation" and made parameter-deterministic reuse
a headline goal of the rewrite (`docs/COMPUTE-GRAPH-DESIGN.md` §1, "Graph
reuse" row). The rest of this document walks the loop one excerpt at a time:
the per-step data flow (§2, §3.1), the code of the loop and the reuse
machinery (§3.2), why it is built this way (§3.3), and the traps the design
has to defuse (§3.4).

## 2. Principle — how it works and why

### 2.1 The loop in one picture

Here is one run of `minfer model.gguf "Hello"` with `-n 4`, annotated with
which graph the engine used at each moment:

```text
 prompt "Hello" ──tokenize──▶ [ids: nt tokens]

 ┌────────────────────────────────────────────────────────────────────┐
 │ PREFILL (one call, doc 09)                                         │
 │   graph: built for n_tokens = nt        (GraphType::Prefill)       │
 │   inputs: token_ids = ids, positions = [0, 1, .., nt-1]            │
 │   KV regions after: nt slots filled per layer                      │
 │   output: logits for the LAST token          ← 607 KB of f32      │
 └────────────────────────────────────────────────────────────────────┘
                    │ sample → t₁                     (doc 12)
                    ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │ DECODE step 1                                                      │
 │   REBUILD here (n_tokens nt→1, gtype Prefill→Decode) — exactly     │
 │   once per run; KV regions + buffer pools survive the rebuild      │
 │   inputs: token_ids = [t₁], positions = [nt]                       │
 │   KV: layer ℓ writes t₁'s K/V at slot nt, attention reads slots    │
 │        0..=nt (nkv = nt+1 — the window grows by one)               │
 │   output: logits for t₂'s sampling                                 │
 └────────────────────────────────────────────────────────────────────┘
                    │ sample → t₂
                    ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │ DECODE step 2, 3, …   SAME graph, zero rebuilds                    │
 │   try_reuse(params) → true every step                              │
 │   inputs refreshed: token_ids = [tᵢ], positions = [nt+i-1]         │
 │   one scheduler walk per step (one split walk per token)           │
 └────────────────────────────────────────────────────────────────────┘
                    │ sample → t₄
                    ▼
        stop gate: EOS? stop string? generated == n_predict?
```

Two structural things to notice before any code:

- **The loop consumes logits that already exist.** Each iteration samples
  *first*, from the logits the *previous* iteration's forward produced (or
  from prefill's logits on iteration one), and only then runs the forward
  that produces logits for the *next* iteration. The forward is the last
  statement of the body — `ModelDef::forward` (`main.rs:1782`), not the first.
- **The rebuild happens inside the first `forward`, invisibly.** The loop
  calls `model.forward(...)` on every step; the cache check
  (`try_reuse`) is inside `forward_cached`. The loop never knows whether a
  given call rebuilt or reused — from the loop's point of view every step is
  "fill two small inputs, execute, get logits".

### 2.2 Why a decode step is cheap relative to prefill

The run header prints two rates for a reason — prefill throughput and decode
throughput differ by an order of magnitude, and it is worth being precise
about where the savings come from. Take Qwen2.5-0.5B as the worked example
(`d_model = 896`, 24 layers, 14 query heads / 2 KV heads × head-dim 64, so
`nkt = 128` KV channels per layer, `n_vocab = 151,936`; doc 10 uses the same
model).

**Budget 1 — weight streaming (unchanged, and dominant).** Every matmul of
every layer reads its whole weight matrix per forward, whether it multiplies
it by 128 token rows or by 1. That is ~0.3 GB of quantized weights streamed
from RAM per decode step, so on a ~60–100 GB/s memory system the floor is a
few milliseconds per token no matter what the loop does — doc 10 §2.1 calls
this "the decode physics". Graph reuse cannot shrink this budget; it only
removes the overhead *around* it.

**Budget 2 — matmul arithmetic (shrinks by `nt`).** The multiply-add work of
a matmul scales with the number of token rows. Prefill of a 128-token prompt
does 128× the arithmetic of a decode step for the identical weights. This is
why prefill shows high tok/s (the cost is amortized over many tokens) while
decode shows the raw weight-streaming rate: the plan document records ~100
tok/s CPU decode for 0.5B (`docs/COMPUTE-GRAPH-DESIGN.md` §12, first row) —
the loop is running as fast as RAM allows.

**Budget 3 — attention (shrinks from quadratic to one scan).** Prefill
attention is causal all-pairs work over `nt` queries (doc 11); decode has
**one** query, whose scan a *data* window bounds: the allocator resolves it
from the KV cell store and hands it to the kernel as the `attn_span` /
`kv_map` input (E1, C8b). The CPU path decodes that window — `decode_window`
(`cpu_backend.rs:683`):

```rust
                // E1/C8b S2: the allowed cells are an explicit input, not a bound
                // derived from `positions` — that is what lets a batch hold several
                // sequences, and (S2) what lets one query's window be a list of runs.
                // The allocator resolved it from the cell store; here it is only
                // validated and decoded.
                let (runs, off) = decode_window(ins[3], nt, n_ctx)?;
```

No field of the graph knows or cares how large the cache has grown; the
kernel reads exactly the cells the window names out of a region allocated at
the full `n_ctx` from the start — the whole reason a fixed graph can serve a
growing cache (§2.4). The cost grows linearly with context: at `nkv = 1024`
a decode step reads ≈ 24 MB of K/V (`2 × 24 layers × 128 × 1024 × 4 B`), and
~96 MB at `n_ctx = 4096`. The position-derived `nkv = max_pos + 1` predates
E1 (PR #3, 2026-09-17) and now lives only on Metal (`metal_backend.rs:1433-1434`).

**What reuse removes.** Build + assign + fuse + allocate are per-*graph*
costs, not per-token math. Reuse means a decode step pays only: two small
input fills (a 4-byte token id and a 4-byte position), one scheduler walk
over the cached graph, and the math. The plan document's expected-gains
table puts it plainly: decode's gain is "skips topology rebuild and CPU
scratch allocation" (`docs/COMPUTE-GRAPH-DESIGN.md` §12).

### 2.3 One rebuild, then none: the params identity

Every `forward` call constructs a `GraphParams` value and asks the cache
whether it may reuse. Trace the three calls of one run:

| Call | `n_tokens` | `n_out` | `gtype` | `cparams.gpu` | `fuse_qkv`/`fuse_ffn` | Cache verdict |
|---|---|---|---|---|---|---|
| prefill (nt=23) | 23 | 1 | Prefill | false (CPU build) | false / false | miss (empty cache) → build |
| decode step 1 (nt=1) | 1 | 1 | Decode | false | false / false | miss (params differ) → **rebuild** |
| decode step 2, 3, … (nt=1) | 1 | 1 | Decode | false | false / false | **hit → reuse** |

The interesting row is the second. The prefill and decode graphs are *genuinely
different programs*, which is why the rebuild is honest work and not a
technicality:

- every buffer's shape is `[nt, …]` — activations are 23 rows vs 1 row
  (doc 07's layouts), so the whole liveness mapping must be redone;
- the prefill graph contains the **G3 tail reduction** — an extra `tail_ids`
  input node and two `GetRows` nodes that cut the last layer's FFN and the
  lm_head down to the `n_out` output rows (`src/models/qwen2/graph.rs:78-84`, `src/models/qwen2/graph.rs:250-251`).
  In decode `n_out == nt == 1`, so those nodes don't exist at all;
- on a GPU build, the **decode fusions** apply only when `nt == 1`
  (`src/models/qwen2/graph.rs:120-125`), and the same gate in the `CParams` construction
  (`src/models/qwen2/graph.rs:578-584`): `fuse_qkv`. `Op::FusedQKV` merges 3 matmuls + 3 biases
  + 2 RoPEs + 2 KV stores into one kernel. Different node set ⇒ different graph.

And then the third row is the payoff: step 2's params and step 3's params are
*equal* — same `n_tokens`, same `n_out`, same `gtype`, same `cparams` — so
every remaining decode step of the run takes the reuse branch and does zero
structural work. Within one generation run there is **exactly one rebuild**
(prefill → first decode) and **zero** after it.

It is worth being explicit about what does *not* appear in the comparison:
the token ids, the positions, and `n_past`. Those are execution data, injected
into input nodes after the reuse check (`src/models/qwen2/graph.rs:649-672`). The plan
document records this as the design's founding correction: an earlier
sketch encoded `n_past` into the KV-cache op, which "causes the decode
topology to change at every step and structurally breaks graph reuse"
(`docs/COMPUTE-GRAPH-DESIGN.md`, revision note 1). Moving the position from
the graph's *structure* into its *data* is what makes row 3 of the table
possible at all.

### 2.4 Why a params-only comparison is enough to decide reuse

The claim that carries the whole design: **if the params are equal, the
topology is equal** — so comparing six scalar/enum fields is a sound
substitute for comparing graphs. For that to be sound, each field must be
*genuinely load-bearing* — it must be able to change the node sequence. It
is worth checking each one `GraphParams` (`params.rs:88-97`), defined below in §3.2):

- **`n_tokens`** — every activation buffer's shape and loop trip counts
  derive from it; also selects the per-layer QKV build path (`nt == 1`
  enables the decode fusions, `src/models/qwen2/graph.rs:120-125`).
- **~~`n_seqs`~~** — *deleted in E2 (A7 closure).* A batch's sequence count is
  data, like `n_past`: the one topology decision it can force — `explicit_span`,
  the explicit attention window — lives in `cparams` and is derived from the KV
  reservations. Carrying the count here bought nothing and cost a rebuild
  whenever batching changed shape with the topology unchanged.
- **`n_out`** — decides whether the G3 `tail_ids` input + tail `GetRows`
  nodes exist (`n_out < nt`) and how many rows the output buffer has.
- **`gtype`** — Decode vs Prefill; today it is redundant with
  `n_tokens == 1`, but it names the *intent* (llama.cpp's
  `llm_graph_params` carries the same distinction) and keeps the door open
  for graph shapes that are not a function of token count alone.
- **`cparams`** — the runtime knobs that reach into topology:
  `n_ctx` (sizes the KV regions and the RoPE/attention metadata),
  `flash_attn` (selects the attention node's mode), **`gpu`** (backend
  assignment is part of the built graph — a GPU that initialized between two
  calls must force a rebuild (`params.rs:33`, the `gpu` field), and the fusion gates
  `fuse_qkv` / `fuse_ffn` (the A/B env toggles must reliably force a
  rebuild, hence they live inside `cparams` and inside the equality check —
  the test `fuse_flags_are_part_of_the_reuse_identity` pins this,
  `verify_structural` (`cache.rs:134`).
- **`weights_version`** — the future LoRA/reload hook: bumped whenever
  weights change, invalidating every cached graph (`params.rs:96-97`, the `weights_version` field).

If topology were *not* a pure function of these, a params hit would reuse a
graph that was subtly wrong for the new inputs — the worst kind of bug,
because it looks like a numerics problem. The defense is a debug-build-only
second line of checks: `GraphCache::verify_structural`
`verify_structural` (`cache.rs:134-145`) compares two graphs built from equal params node by node
(op, shape, dependencies) and is wired into tests, so a *non-deterministic*
builder — the one way params-equality could lie — is caught in CI rather
than in production. In release builds the six-field comparison is all there
is, and it is enough *because* the builder is deterministic by contract
(`src/models/qwen2/graph.rs:37-38`: "deterministic in `params` — the reuse invariant").

This is a deliberate about-face from an earlier plan sketch that built the
new graph first and compared node sequences ("build then compare") — which
can never skip the rebuild, defeating the purpose. The plan document
records the correction: "no graph build, no node-sequence comparison; the
graph topology is a deterministic function of the parameters — the same
invariant as llama.cpp `allow_reuse()`" (`docs/COMPUTE-GRAPH-DESIGN.md` §6).

### 2.5 What survives a rebuild vs what is recomputed

The rebuild between prefill and decode (and between turns in a chat session)
redraws a sharp line. On the "recomputed" side, everything that *describes
the graph*; on the "survives" side, everything that *holds data*:

| Survives the rebuild | Recomputed on rebuild |
|---|---|
| The `GraphAllocator` itself (it lives inside `GraphCache`, `cache.rs:38-45`) | The node list (`Self::build`, `src/models/qwen2/graph.rs:605`) |
| Registered weights (registered once by name; `register_weight`, `alloc.rs:374`) | Backend assignment (`assign_backends`, `src/models/qwen2/graph.rs:622`) |
| **The per-layer KV regions** — `kv.{ℓ}.k` / `kv.{ℓ}.v`, allocated once at full `n_ctx` size and never freed (`ensure_kv`, `alloc.rs:995-1002`; `alloc_graph` explicitly frees only liveness buffers, `alloc_graph` (`alloc.rs:513-519`)) | The fusion pass (`FusionPass::run`, `src/models/qwen2/graph.rs:630-634`) |
| Backend buffer *pools* (freed liveness buffers return to their pool; the memory is recycled, not released) | The node→buffer mapping (`alloc_graph` clears `node_to_buf`, `alloc.rs:519`) |
| The monotonic graph `uid` of the *reused* graph (a rebuilt graph gets a fresh uid — which is exactly what invalidates a stale CUDA Graph capture, `replace_graph` (`cache.rs:106-114`) + §3.4) | Cross-backend staging buffers (keyed by `(graph uid, node, backend)` and surviving a rebuild *and* a re-map — E4 S3; `alloc_graph` clears only the in-flight `cross_pending` set, `alloc_graph` (`alloc.rs:513-528`)) |

The first row is the one that matters most: the allocator is a field of the
cache, not a local of the forward function, so a rebuild cannot take the KV
cache with it. `cache.rs`'s module comment states the contract in two
sentences (`cache.rs:9-12`, the "survives graph rebuilds" module comment): the allocator "lives inside the cache and
**survives graph rebuilds**: the persistent KV regions are exactly the KV
cache, so a prefill→decode transition (different `n_tokens`/`gtype` ⇒
rebuild) must not lose them. Only the node/buffer mapping is recomputed on
rebuild." Doc 07 covers the region mechanics; this document cares about the
*consequence* — the decode step after a rebuild finds the prompt's context
already in place, at the same addresses, and simply appends.

### 2.6 Multi-turn conversation: append-only KV, delta prefill

A chat session (`--cnv`) is the decode loop wearing a longer timeline. The
session keeps a host-side mirror of the KV contents — `stream_tokens`, a
plain `Vec<u32>` — plus a write cursor `current_pos` that always equals its
length (`conversation.rs:399-401`, the `stream_tokens`/`current_pos` fields); the real-model test asserts this
invariant (`conversation/tests/real_model.rs:90-91`, the real-model assert). The KV region itself is
**position-addressed** (slot `p` of layer ℓ's K region holds the K vector of
the token at position `p`), which makes the whole session strategy possible:

- **Append-only.** Turn 2 does not rewrite turn 1's slots. `user_turn`
  renders only the *delta* — the new user message wrapped in the template's
  turn separator — tokenizes it, and prefills it at positions
  `current_pos .. current_pos + delta.len()` (`conversation.rs:949-952`, the delta prefill).
  The engine then decodes the assistant reply with the same loop as before.
  Each turn costs O(delta), never O(history).
- **Same graph, more rebuilds.** Each turn boundary flips `n_tokens` from 1
  back to `delta_len` (Prefill) and then to 1 again (Decode) — a rebuild
  per flip. That is two cheap rebuilds per turn against one very expensive
  re-prefill of the whole history; the KV regions and buffer pools survive
  every flip (§2.5).
- **Rollback without erasing.** `/regen` rewinds `current_pos` to
  `turn_pos` (the start of the last turn's delta) and regenerates
  (`conversation.rs:965-977`, the `regen_turn` rollback). The rolled-back slots in the KV region are
  now *stale but never read*: attention reads only the window the cursor
  bounds. Regeneration simply overwrites those slots as
  it appends. (A full `/clear` or a template mismatch falls back to
  `rehydrate_full` — reset the cache, re-render everything, re-prefill
  once, `rehydrate_full` (`conversation.rs:518`).)
- **Seams kept consistent.** If a turn ended without an end-of-turn token,
  the next turn first inserts the EOT token into the KV at the cursor
  (`conversation.rs:771-776`, the EOT insert) so the region keeps matching what the chat
  template's canonical render would have produced — the module doc calls
  this the §5.4 KV-consistency invariant (`conversation.rs:16-19`, the module's KV-consistency note).

### 2.7 What stops generation, and where that is decided

Three gates, three different owners, all on the sampler's output before or
instead of the next forward (§3.2 walks the code):

1. **End-of-generation token.** The sampled id equals the model's `eos` or
   `<|im_end|>` (`is_stop_token`, `main.rs:1868-1870`; the ids come from
   `special_tokens()` (`main.rs:1560`). Decided in the loop
   `is_stop_token` (`main.rs:1749-1751`); the token is *not* appended and *not* fed to the
   graph — the run just ends. (Conversation mode deliberately does the
   opposite: it writes the EOG token into the KV before breaking, to keep
   the region matching the template's canonical next-turn render —
   (`conversation.rs:1330-1343`, the EOG write to the KV).)
2. **Stop strings (`--stop`).** Byte-level suffix match over the *entire*
   generated byte stream, so a stop string split across token boundaries is
   still caught by `match_stop_suffix` (`main.rs:1761-1768`;
   doc 12 §3 owns the matcher). The matching bytes are truncated out of the
   kept text; already-printed bytes stay in the terminal, llama.cpp-style.
3. **The `-n` cap.** The `while` condition `generated.len() < params.n_predict`
   the loop head (`main.rs:1727`), default 512 in `GenParams::default` (`main.rs:110-112`). In conversation mode a
   fourth gate joins: the write cursor reaching `n_ctx` stops cleanly
   instead of overflowing the KV regions (`conversation.rs:1314-1318`, the cursor guard).

Note the asymmetry between the gates: EOS and stop strings `break` *before*
the forward at the bottom of the body, so no forward is wasted on a token
nobody reads. The `-n` cap, checked only at the loop head, lets the final
iteration's forward run — one discarded forward per full-length run, the
price of a simple loop condition (§3.4).

## 3. Implementation

### 3.1 Data in / data out

Per decode step, the engine touches a remarkably small amount of *changing*
data. Everything else — weights, graph, KV regions, buffers — was set up once
and is only read:

| Data | Type / shape | Where it comes from | Where it goes |
|---|---|---|---|
| `token_ids` input | `I32`, shape `[1, 1, 1, 1]` (one token) | step 1: the token sampled from prefill's logits; every later step: the previous iteration's `sampled.token_id` | `fill_input_i32` writes it into the input node's buffer (`src/models/qwen2/graph.rs:651`) |
| `positions` input | `I32`, shape `[1, 1, 1, 1]` | `current_pos` (`main.rs:1561`) — starts at `input_ids.len()`, incremented once per step at `current_pos` (`main.rs:1790`) | same, `src/models/qwen2/graph.rs:653`; the KV store uses it as the write slot, attention as the last readable slot |
| K/V regions (per layer) | `f32`, `[nkt][n_ctx]` (128 × 4096 = 2 MiB per region for Qwen2.5-0.5B at `--n-ctx 4096`) | allocated once at first use (`ensure_kv`, `alloc.rs:995-1002`); contents: prefill's prompt + every generated token so far | step ℓ's store writes slot `position`; step ℓ+1's attention reads slots `0..=nkv-1` |
| logits output | `f32`, `[n_vocab]` = 151,936 × 4 B ≈ 607 KB | the graph's output buffer (`graph.outputs[0]`, copied to host at `src/models/qwen2/graph.rs:746-758`) | moved into the loop's `logits` variable (`main.rs:1782-1783`, the `logits = model.forward(...)` assignment) for the next sample |
| `prev_tokens` window | `Vec<u32>`, ≤ 64 ids | prompt tail + generated tokens — `recent_window` (`main.rs:1568, 1752-1757`) | the sampler's repeat/frequency/presence penalties (doc 12) |
| `generated` | `Vec<u32>` | pushed per step (`main.rs:1752`, the `generated.push(...)` call) | stop checks, final stats; decode_bytes streams it to stdout |

Two details of this table deserve unpacking.

**The I32 bit-pattern trick.** The allocator's buffers are `f32` pools —
every node reads and writes `f32` slices, whatever its logical type. Token
ids and positions are integers. Rather than special-case integer buffers,
`fill_input_i32` stores each `u32` *bit pattern* reinterpreted as an `f32`
value `fill_input_i32` (`alloc.rs:1901`), and the kernels that consume these inputs
(attention, KV store) convert back with `f32::to_bits() as usize`
(`cpu_backend.rs:334-338`, the store's position decode; `:919-920`, attention's). This is exact for values below 2²⁴ —
vocabulary ids and positions never come close — and it keeps one uniform
buffer format across the whole graph (AGENTS.md "Compute Graph" rule 4).
`f32::from_bits(v)` does no rounding at all; it is a `transmute`, not an
arithmetic conversion, which is why a round trip through the buffer is
lossless.

**One forward call per step, five hidden phases.** The loop's single
`model.forward(...)` — `ModelDef::forward` (`main.rs:1782`) — expands inside `forward_cached` into:
build params (six fields) → try_reuse (usually a hit) → fill 2–3 small
inputs → one `scheduler.execute` walk → copy the output buffer back. On a
rebuild step the same call additionally runs build → register → assign →
fuse → alloc. The loop code has no idea any of that exists — which is the
point of the `ModelDef::forward` facade `ModelDef::forward` (`models/mod.rs:222`).

One name in the call needs a sentence for honesty: `forward`'s signature *used*
to take a `&mut KVCache` (`main.rs` passed a `kv_cache` created at load). On the
graph path that legacy type was always *ignored* — the callee bound it as `_kv`
and `forward_cached` never took it — so [#252] deleted the parameter, the
`KVCache` type and the construction rather than keep a dead API shape. The real
KV lives in the allocator's persistent regions (doc 07), and the call now reads
`forward(tokens, positions, n_out, n_ctx)`.

[#252]: https://github.com/yusiwen/minfer/issues/252

### 3.2 Key code

#### The loop's setup (`src/main.rs:1553-1568`, the decode-loop locals)

Before the loop starts, a handful of locals are established — each one is a
hand the loop plays with on every iteration:

```rust
    let mut logits = last_logits;
    if trace_on {
        crate::trace::begin_phase("decode");
    }
    let gen_start = Instant::now(); // pure-decode start (llama "Generation" caliber)
    let mut generated: Vec<u32> = Vec::new();
    let special = model.special_tokens();
    let mut current_pos = input_ids.len();

    // Seeded RNG for reproducible sampling; recent-token window for the
    // penalties (llama.cpp repeat_last_n default = 64), seeded with the prompt
    // tail so the first generated tokens are penalized too.
    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    const REPEAT_LAST_N: usize = 64;
    let mut prev_tokens = sampler::recent_window(&input_ids, REPEAT_LAST_N);
```

- `logits` starts as **prefill's last-token logits** — the loop's first
  iteration samples from them without running a forward first. This is the
  delayed-forward shape of §2.1 made concrete.
- `current_pos = input_ids.len()`: the first generated token will be written
  at KV slot `nt` — the slot *after* the prompt. The KV regions were sized
  for `n_ctx` slots, and `ctx = params.n_ctx.max(input_ids.len())`
  (`main.rs:1452-1453`, the `ctx` computation) guaranteed at prefill time that the prompt fits; from here
  the cursor only ever increments.
- `special` carries the model's stop-token ids (eos, and `<|im_end|>` when
  the model defines one) — gate 1 of §2.7.
- `rng` is seeded once (`--seed`, default 42): the whole run's sampling is a
  deterministic function of the seed, which is what makes minfer's
  llama.cpp-comparison claims testable.
- `prev_tokens` seeds the 64-token penalty window with the *prompt's tail*,
  so even the first sampled token is subject to repeat penalties — doc 12's
  `recent_window` helper.

The stop-string machinery follows immediately — `stop_refs` (`main.rs:1570-1582`): user
`--stop` strings are copied to byte vectors (`stop_bytes`/`stop_refs`), and
two cursors track output — `full`, every generated byte ever, and `emitted`,
how many of those bytes have already been flushed to stdout. The pair exists
because a stop string may straddle token boundaries: earlier tokens were
already printed before anyone could know a stop string was forming.

#### The decode loop, first half — sample and stop (`src/main.rs:1727-1772`, through the stop-string check)

```rust
    while generated.len() < params.n_predict {
        t0 = std::time::Instant::now();
        let sampled = match sampler::sample_with_config_grammar(
            &mut logits,
            &sampler_cfg,
            &prev_tokens,
            &mut mirostat,
            &mut grammar_state,
            &mut rng,
        ) {
            Ok(s) => s,
            Err(e) => {
                // The loud stop the grammar contract requires: the emitted text
                // is a valid prefix, nothing illegal is appended.
                eprintln!("\n[grammar] {e}");
                break;
            }
        };
        if timing {
            t_samp += t0.elapsed().as_secs_f64();
        }

        if is_stop_token(sampled.token_id, &special) {
            break;
        }
        generated.push(sampled.token_id);
        token_trace(current_pos, sampled.token_id);
        prev_tokens.push(sampled.token_id);
        if prev_tokens.len() > REPEAT_LAST_N {
            prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
        }

        // Stop-string detection on the FULL byte stream before emitting.
        full.extend_from_slice(&tokenizer.decode_bytes(&[sampled.token_id]));
        if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
            full.truncate(cut);
            if cut > emitted {
                hi.feed(&full[emitted..]);
                emitted = full.len();
            }
            break;
        }
        if emitted < full.len() {
            hi.feed(&full[emitted..]);
            emitted = full.len();
        }
```

Reading it as a state machine, one iteration:

1. **Gate: the cap.** The `while` condition is gate 3: a full quota exits
   without touching anything.
2. **Sample.** `sample_with_config_grammar` (doc 12) consumes *and mutates*
   `logits` (hence `&mut`): the config's penalties and DRY, then the grammar
   mask, then top-k → top-p → temperature (or mirostat) → one seeded draw.
   Out comes one token id; the loop never inspects logits itself, and the
   607 KB of scores are noise to everyone but the sampler. A `SampleError`
   (no token the grammar allows) is the loud stop the excerpt handles: the
   loop prints it and breaks, never emitting an arbitrary token.
3. **Gate: EOS.** `is_stop_token` (`main.rs:1868-1870`) checks `eos`/`im_end`;
   a hit breaks *before* the token is pushed — the sentinel is not text.
4. **Append.** The token enters `generated` and the 64-entry sliding
   penalty window, drained from the front.
5. **Gate: stop strings** — the tail of the excerpt: the token's bytes join
   `full`, and `match_stop_suffix` (`main.rs:1760-1768`) tests whether it
   now *ends* with a stop string; on a hit the text is truncated before the
   match, else the new bytes are flushed to stdout (`main.rs:1769-1772`).

Only when all three gates pass does the iteration continue to the forward —
the loop never runs the transformer for a token it has already decided to
discard.

#### The decode loop, second half — forward and advance (`src/main.rs:1774-1791`, the forward call and cursor advance)

```rust
        // forward() returns n_out*nv logits (n_out=1 for single-token decode,
        // exactly n_vocab), so move the Vec in place instead of copying 607 KB/token.
        if trace_on {
            let text = String::from_utf8_lossy(&tokenizer.decode_bytes(&[sampled.token_id]))
                .into_owned();
            crate::trace::set_token(sampled.token_id, &text);
        }
        t1 = std::time::Instant::now();
        logits = model.forward(&[sampled.token_id], &[current_pos], 1, ctx);
        if trace_on {
            crate::trace::attach_step(&logits);
        }
        if timing {
            t_fwd += t1.elapsed().as_secs_f64();
            n_tok += 1;
        }
        current_pos += 1;
    }
```

The heart of the whole document is one line:

```rust
logits = model.forward(&[sampled.token_id], &[current_pos], 1, ctx);
```

Four arguments, and every one of them is either constant across the whole
loop or a single value that changes: the input is a **one-element slice**
containing last iteration's sampled token; the position is a **one-element
slice** containing the cursor; `n_out` is 1 (we need logits for exactly one
token); `ctx` is the constant that sized the KV regions. Nothing here says
"decode" or "step 37 of 500" — the loop is identical whether it is the first
or the four-hundredth step, and the graph machinery underneath treats it as
just another forward whose params happen to be unchanged. The `MINFER_TIMING`
arrows around it (`t1`, `t_fwd`) split the per-token wall clock into
sampling time and forward time — §4 uses this to show that the loop overhead
is microseconds against the forward's milliseconds.

The `trace_on` block records the token and the logits for the viz trace
(§4); note it is hoisted out of the loop as a bool — `trace_on` (`main.rs:1456-1457`) so
the steady-state loop does one env-var read fewer per step. And
`current_pos += 1` is the KV cursor's whole life: prefill filled slots
`0..nt`, this line claims slot `nt`, `nt+1`, … one per step, until either a
stop gate fires or (in conversation mode) a guard stops it before the
regions overflow.

`is_stop_token` itself is the smallest function in the pipeline
`is_stop_token` (`main.rs:1868-1870`):

```rust
fn is_stop_token(id: u32, special: &models::SpecialTokens) -> bool {
    id == special.eos || Some(id) == special.im_end
}
```

`im_end` is an `Option` because not every model defines `<|im_end|>` (Qwen2.5
and Qwen3 chat models do). Two ids, one boolean — but this function is the
model's only "voice": everything else about stopping is user policy (stop
strings, `-n`), while this is the model saying "I'm done".

#### Inside the forward: building the params — `GraphParams` (`src/models/qwen2/graph.rs:541-597`)

The loop's `forward` call lands in `forward_cached`, which first expresses
"what kind of graph does this step need?" as a plain data value:

```rust
        let params = GraphParams {
            n_tokens: nt,
            n_out,
            gtype: if nt == 1 {
                GraphType::Decode
            } else {
                GraphType::Prefill
            },
            cparams: CParams {
                n_ctx,
                flash_attn: false,
                explicit_span,
                kv_map,
                gpu: metal_on || cuda_on,
                gpu_layers: if metal_on || cuda_on {
                    model.offload.plan.gpu_layers
                } else {
                    0
                },
                fuse_qkv: nt == 1
                    && (metal_on || cuda_on)
                    && (cuda_on || !explicit_span)
                    && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1"),
                fuse_ffn: nt == 1
                    && (metal_on || cuda_on)
                    && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1"),
                kv_format: model.kv_format,
            },
            weights_version: 1,
        };
```

Everything the reuse decision will ever need is assembled here, before the
cache is consulted (comments elided): `gtype` is derived from `nt`; the new
`explicit_span` / `kv_map` pair selects the attention window *layout* — both
are topology, both derived from the KV reservations, never from `n_past` —
and `gpu` (`params.rs:33`) records whether a GPU backend participates, a
*param* because backend assignment is baked into the graph, so "Metal became
available between two calls" must force a rebuild. `gpu_layers` carries the
E5 offload plan (its assignment is topology too), and `kv_format`
(`params.rs:61-67`, C4) sizes every KV cell, so a graph built for one format
never serves another. The fusion gates are the subtlest part: runtime env
vars (`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1`) change which `Op`s
the graph contains, so they live in `CParams` — flipping one mid-run fails
the comparison and rebuilds with the new state. That is how an *environment
variable* safely joins a build cache, and the two gates are separate so that A/B-ing one cannot flip the other.

Before this struct is built, `forward_cached` has already done two quiet
checks worth noting (`src/models/qwen2/graph.rs:507-521`): it asserts every position is
below `n_ctx` — out-of-range positions would write past the KV regions, so
the failure is a loud panic, not silent corruption (`src/models/qwen2/graph.rs:507-514`) —
and it probes GPU availability (`metal_on` / `cuda_on`), which feeds the
`gpu` field above.

#### The reuse decision itself `GraphCache` (`src/graph/cache.rs:38-64`)

With params in hand, the cache is asked one question
(`src/models/qwen2/graph.rs:599`): `if !cache.try_reuse(&params).expect("re-map
onto a cached graph") { … }`. Here is the whole machinery:

```rust
    pub fn try_reuse(&mut self, params: &GraphParams) -> Result<bool, String> {
        let Some(pos) = self
            .graphs
            .iter()
            .position(|(p, _)| Self::params_match(p, params))
        else {
            return Ok(false);
        };
        let (_, graph) = self.graphs.remove(pos);
        …(the `Err` carries the failed re-map back to the caller)…
        self.alloc.alloc_graph(&graph)?;
        self.graphs.insert(0, (params.clone(), graph));
        self.reuses += 1;
        Ok(true)
    }
```

That is the entire cache: an MRU `Vec` of at most `MAX_CACHED_GRAPHS = 8`
`(GraphParams, ComputeGraph)` pairs, one per distinct shape
(`cache.rs:36-45`) — a session alternates a handful, so they all fit. Six
field comparisons (`params_match`, `cache.rs:95-101`) replace rebuilding a
few hundred nodes, re-assigning backends, re-running fusion, and re-walking
liveness. A miss means no cached graph has these params; a hit moves that
graph to the front and re-maps the allocator onto it, which is the one step
that can fail — hence the `Result`, which the caller `.expect`s (loud panic).

The `GraphParams` and `CParams` types behind the comparison
`CParams` (`src/graph/params.rs:30-63`) and `GraphType` (`params.rs:19`) carry exactly the fields
argued for in §2.4:

```rust
pub struct GraphParams {
    pub n_tokens: usize,
    /// Number of output (tail) rows: the last layer's FFN + lm_head run on the
    /// last `n_out` rows only (llama `inp_out_ids`). Part of the topology —
    /// a change forces a rebuild.
    pub n_out: usize,
    pub gtype: GraphType,
    pub cparams: CParams,
    /// Bumped by the model whenever weights change (LoRA switch, reload).
    pub weights_version: u64,
}
```

and inside `CParams`: `n_ctx`, `flash_attn`, `gpu`,
`fuse_qkv`, `fuse_ffn` (`src/graph/params.rs`) — each documented there with the
reason it belongs in the identity. The module's opening comment is the
invariant in one breath (`params.rs:1-7`, the module comment): these are "the ONLY inputs to
graph reuse … `n_past` (KV position) is deliberately absent: it is
execution data."

#### The rebuild branch `forward_cached` (`src/models/qwen2/graph.rs:456`, `599-645`)

When the comparison fails, the five-phase pipeline of docs 05–08 runs, and
its result is stored back into the same cache:

```rust
        if !cache
            .try_reuse(&params)
            .expect("re-map onto a cached graph")
        {
            let rebuild_t0 = std::time::Instant::now();
            let trace_rb = std::env::var("MINFER_REBUILD_TRACE").map_or(false, |v| v == "1");
            let mut graph = Self::build(model, &params);
            let sched = BackendScheduler::new();
            {
                let alloc = cache.alloc();
                Self::register_graph_weights(model, alloc);
                …(the `#[cfg]` enable_metal / enable_cuda blocks, doc 07)…
                alloc.set_offload_plan(Some(model.offload.plan));
                sched.assign_backends(&mut graph, alloc);
```

…(the middle of the block is the fusion pass's backend list, which comes from
the allocator's registry view rather than a hand-built `Vec` — `src/models/qwen2/graph.rs:623-629`)…

```rust
                let backends: Vec<&dyn Backend> = alloc.fusion_backends();
                FusionPass::new().run(&mut graph, &backends, &|g, id| {
                    g.node(id)
                        .backend
                        .and_then(|b| alloc.fusion_backend_index(b))
                });
                alloc.alloc_graph(&graph).unwrap();
            }
            cache.replace_graph(graph, params);
        }

        let (graph, alloc) = cache.current().unwrap();
```

Three details turn this from "a rebuild" into "a rebuild that preserves the
session", and the `.expect("re-map onto a cached graph")` above makes a
failed re-map a loud panic, never a silent rebuild:

- `cache.alloc()` — the allocator is *borrowed from the cache*; `register_graph_weights`
  re-registers weights by name, idempotently (doc 03/07), and `set_offload_plan` (E5)
  puts the block plan in force *before* assignment. `alloc_graph` frees the old
  liveness mapping but keeps the persistent KV regions (§2.5, below).
- The pipeline is the *full* one — assign, fusion (backend-gated per node),
  allocate — because a rebuilt graph must be indistinguishable from a
  fresh-process graph; determinism (§2.4) is what makes that sound.
  `MINFER_REBUILD_TRACE=1` prints `rebuild_t0`'s build+assign+alloc ms.
- `replace_graph` (`cache.rs:106-114`) stores the new node list and params,
  and stamps the graph with a fresh monotonic `uid` — `replace_graph` (`cache.rs:106-107`) —
  the CUDA Graph cache keys captures by uid, so a new topology naturally
  invalidates the old capture while a *reused* graph keeps its uid and its
  replay (doc 15).

Then, on *both* paths (rebuilt or reused), the inputs are refreshed:

```rust
        // refresh input data (positions/ids are data, not topology)
        let ids: Vec<u32> = tokens.to_vec();
        alloc.fill_input_i32(graph, "token_ids", &ids).unwrap();
        let pos: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        alloc.fill_input_i32(graph, "positions", &pos).unwrap();
        …(E1/E2: `fill_batch_inputs` resolves every query's window)…
        alloc
            .fill_batch_inputs(graph, batch)
            .unwrap_or_else(|e| panic!("batch inputs: {e}"));
        // G3: the last-layer tail-row reduction reads `tail_ids` (filled when
        // the graph was built with n_out < nt, i.e. prefill)
        if graph
            .inputs
            .iter()
            .any(|&i| graph.node(i).name == "tail_ids")
        {
            alloc.fill_input_i32(graph, "tail_ids", &out_rows).unwrap();
        }
```

This is "positions are data" in executable form: the same code runs for a
23-token prefill (23 positions) and for decode step 400 (one position,
value 422) — the graph is never told which situation it is in; it reads the
buffers. `fill_batch_inputs` (`src/models/qwen2/graph.rs:661-663`) resolves
each sequence's window first; the `tail_ids` fill is conditional and takes
the batch's `out_rows` (`src/models/qwen2/graph.rs:671`) — that input *only
exists in prefill graphs* (`n_out < nt`), so its presence is data too.

Finally `sched.execute(graph, alloc)` runs one scheduler walk
(`src/models/qwen2/graph.rs:686`), and the output buffer is copied back as the returned
logits (`src/models/qwen2/graph.rs:746-758`). With G3 active the output buffer already holds
exactly `n_out × n_vocab` values, so the return is either the buffer itself
or a truncated copy — never a full-`nt` logits matrix (doc 09 covered the
prefill-side benefit; in decode `n_out == nt == 1`, so the buffer is one
row regardless).

#### Why the KV survives: the allocator's two kinds of memory (`alloc_graph` (`src/graph/alloc.rs:513-519`), `alloc_persistent` (`alloc.rs:1080-1089`))

The claim everywhere above is that a rebuild "keeps the KV". The mechanism
is that the allocator distinguishes two kinds of buffers, and only one kind
is freed on rebuild. First, `alloc_graph` — which runs on *every* rebuild —
clears only the liveness-managed mappings:

```rust
    /// Runs on every graph (re)build: previous liveness buffers are released
    /// back to their pools, while **persistent regions (KV cache) survive** —
    /// they are the KV cache and must persist across prefill→decode rebuilds.
    pub fn alloc_graph(&mut self, graph: &ComputeGraph) -> Result<(), String> {
        let prev: Vec<(Backend, usize)> = self.buf_alive.keys().copied().collect();
        for (b, id) in prev {
            self.free_in_pool(b, id);
        }
        self.buf_alive.clear();
        self.node_to_buf.clear();
        // F5: a rebuild starts a fresh execution — no staging copy can be in
        // flight across it (the boundary either completed or failed loudly).
        self.cross_pending.clear();
```

Note the `free_in_pool` call: freed buffers *return to their backend's pool*
rather than being deallocated — the prefill graph's big activation buffers
are immediately available to serve the decode graph's smaller ones, which is
why a rebuild does not reallocate GPU memory or fragment the pools (doc 07's
pool mechanics). `cross_pending.clear()` only forgets the in-flight copies:
the staging buffers themselves are keyed by `(graph uid, node, backend)` and
survive both a rebuild and a re-map — re-creating them per switch would leak,
as staging is `alloc_fresh`, which never recycles from the free list
(`alloc.rs:520-528`). The KV regions are *never in* `buf_alive`: `ensure_kv`
creates them on first use into a separate `persistent` list nothing touches:

```rust
    fn ensure_kv(…, n_embd: usize, row_elems: usize, n_ctx: usize) -> Result<[BufRef; 2], String> {
        …(the C4 packed-width checks)…
        let elems = row_elems * n_ctx;
        if let Some(region) = self.kv.get(layer) {
            …(elems / backend / packed checks)…
            return Ok([region.k, region.v]);
        }
        let k = self.alloc_persistent(&format!("kv.{layer}.k"), backend, elems);
        let v = self.alloc_persistent(&format!("kv.{layer}.v"), backend, elems);
        self.kv
            .insert(layer, k, v, n_embd, row_elems, n_ctx, packed);
        Ok([k, v])
    }

    /// Allocate a persistent (never-freed) region on a backend.
    pub fn alloc_persistent(&mut self, name: &str, backend: Backend, size: usize) -> BufRef {
        let id = self.alloc_exact_in_pool(backend, size);
        self.persistent.push(PersistentBuf {
            name: name.to_string(),
            backend,
            id,
        });
        BufRef::own(backend, id, size)
    }
```

`ensure_kv` is the "allocate once, then look up forever" pattern: the first
graph that touches layer ℓ's KV creates both regions at the *full*
`n_ctx`-sized extent (`elems = row_elems × n_ctx` — 2 MiB per region for
0.5B f32 at `--n-ctx 4096`; a packed Q8_0 region counts packed words, not
f32 elements), and every later graph — including the decode graph of every
subsequent step — just gets the same `BufRef`s back (`alloc.rs:1043-1069`,
the existing-region early return), allocated exact into the never-freed
`persistent` list by `alloc_persistent` (`alloc.rs:1078-1087`), during
`alloc_graph`'s walk of `KvcacheStore`/`KvcacheLoad` (`alloc.rs:670-681`: the store
node's buffer *is* the K region; V is its sibling) — `n_ctx` is a `CParams` field
because `kv_elems: nkt * n_ctx` (`src/models/qwen2/graph.rs:152`) fixes its extent.

This is the exact contract doc 07 promised and the decode loop depends on:
**the KV cache is not a structure the graph owns — it is two never-freed
buffers per layer that the graph borrows by position.** The graph can be
thrown away and rebuilt at every step boundary without touching a single
cached K or V value.

#### One split walk per token (`src/graph/scheduler.rs:219-285`, the per-split boundary and walk)

Reuse also means the execution machinery runs the same cached plan every
step. `execute` `split_graph` (`scheduler.rs:87`) splits the graph into contiguous
same-backend ranges once per call, then walks them. The per-split boundary
handling is where cross-backend sync and copies happen — and on a CPU-only
run there is exactly one split, so none of it fires:

```rust
        for split in &splits {
            if let Some(pb) = prev_backend {
                if pb != split.backend {
                    // 1. retire the previous backend's async work (#138)
                    alloc.retire_backend(pb);
                    // 1b. staged Metal/CUDA captures are valid now — read back
                    flush_metal_captures(graph, alloc, &mut staged, trace_on, live_on);
                    flush_cuda_captures(graph, alloc, &mut cuda_caps, trace_on, live_on);
                    // 2. copy this split's inputs across backends
                    for &inp in &split.inputs {
                        alloc.copy_across(graph.uid, inp, split.backend)?;
                    }
                }
            }
```

and inside each split, the per-node walk (`scheduler.rs:305-326`, the node loop):

```rust
            for id in split.node_range.0..split.node_range.1 {
                let node = graph.node(id);
                if capture && node.is_input() {
                    // inputs are host-filled before execute — no pending GPU
                    // work, so reading them here is always current
                    if let Some(br) = alloc.node_buffer(id) {
                        if let Some(d) = read_host_buffer(alloc, br.backend, br.id)
                            .and_then(|d| window_of(br, d))
                        {
                            record_node_data(node, d, trace_on, live_on);
                        }
                    }
                }
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

The comment at `scheduler.rs:184-186` (the one-step-per-`execute()` note) is
the decode-relevant summary: capture happens "one step per `execute()` (prefill
= 1 step, each decode forward = 1)". So "one split walk per token" is literal:
each decode step re-walks the split list and executes every node in build
order — the *kernels* run fresh every step (new data!), but the *plan* (which
nodes, which backends, which buffers, which splits) is the cached graph's.
`copy_across` only *enqueues* (F5/#138): the single wait is issued at the
consumer's first read, and the capture read above is windowed (`window_of`,
`scheduler.rs:483`). The KV ops resolve their layer's regions by index —
`kv_pair` (`scheduler.rs:360-370`) — then dispatch to the backend's
`execute_node` (`scheduler.rs:398`); a GPU build replays a captured split by
`uid` (doc 15 owns that).

#### The one structural difference: prefill's G3 tail — `tail_ids` (`src/models/qwen2/graph.rs:252-258`)

It is worth seeing the actual node-level difference that forces the
prefill→decode rebuild — the G3 tail-row reduction, which exists only in
prefill graphs:

```rust
            let wo = b.matmul(attn_out, l.wo.as_ref().unwrap(), None);
            let is_last = il == model.layers.len() - 1;
            // G3: reduce to the tail n_out rows BEFORE the last layer's FFN
            // (llama `ggml_get_rows(cur/inpSA, inp_out_ids)` at
            // qwen2.cpp:106-108) — ffn_norm, gate/up/down, swiglu, both
            // residuals and lm_head all run on n_out rows only. The tail_ids
            // input itself is declared at the graph head (see there).
            if is_last && params.n_out < nt {
                let tail_ids = tail_ids.expect("tail_ids input declared when n_out < nt");
                let cur_tail = b.get_rows(wo, tail_ids, [ne, params.n_out, 1, 1]);
                let res_tail = b.get_rows(residual, tail_ids, [ne, params.n_out, 1, 1]);
                h = b.add(res_tail, cur_tail);
            } else {
                h = b.add(residual, wo);
            }
```

Prefill computes attention and projections for all `nt` tokens (it must —
every token's K/V goes into the cache), but only the *last* token's logits
are wanted. So after the last layer's attention output projection, two
`GetRows` nodes gather just the `n_out` tail rows (`tail_ids` is filled
with `nt-1, …, nt-n_out` at `src/models/qwen2/graph.rs:671`), and the remaining FFN +
residual + lm_head run on `n_out` rows instead of `nt`. With `n_out = 1`
and a 2000-token prompt, that saves a 2000-row lm_head GEMM — a 151,936
column × 2000 row output — per run. In decode, `n_out == nt == 1`, the
`if` is false, and the graph simply has no tail nodes at all. Same builder,
one branch, two different programs — the honest reason the params
comparison must fail across the prefill→decode boundary.

#### Multi-turn conversation: the delta prefill (`src/conversation.rs:949-952`, the delta prefill call)

The conversation engine (`GraphEngine`, `conversation.rs:113-117`) wraps the
same `forward_graph_cached` with a *session-private* cache — turn 9 of a
chat reuses turn 8's decode graph, not just this turn's. When the user
submits a new message, `user_turn` prefills only the delta:

```rust
        // 5. Record the user message, set the rollback point, prefill the delta.
        self.messages
            .push(("user".to_string(), Some(input.to_string())));
        self.turn_pos = self.current_pos;
        let positions: Vec<usize> =
            (self.current_pos..self.current_pos + delta_toks.len()).collect();
        let logits = engine.forward(&delta_toks, &positions, 1);
        self.stream_tokens.extend_from_slice(&delta_toks);
        self.current_pos += delta_toks.len();
        // The penalty window is reseeded **after** the delta is appended (llama.cpp feeds prompt tokens into the sampler window).
        self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);
```

Compare with doc 09's first prefill: positions `0..nt` there, positions
`current_pos..current_pos + delta` here — the *same* graph builder, the
same forward, just a later window of slots. `turn_pos` records the delta's
start so `/regen` can rewind to it (§2.6). After this call the cache holds
a prefill graph for `delta_len` tokens; the loop's first decode forward
rebuilds to a 1-token graph — two rebuilds per turn, both cheap, both
KV-preserving.

The conversation decode loop —
`generate_assistant_with_logits` (`conversation.rs:1271-1389`) is the same sample → gates → forward shape, with
two additions the single-shot CLI doesn't need. First, a context guard
before sampling (`conversation.rs:1314-1318`, the cursor guard): if `current_pos >= n_ctx`, the
turn ends "cleanly" — reported as `hit_n_predict` — because writing at slot
`n_ctx` would overflow the regions (the single-shot path instead relies on
`forward_cached`'s position assert, `src/models/qwen2/graph.rs:507-514`). Second, the EOG
token is *written into the KV* before breaking:

```rust
            if self.is_eog(sampled.token_id) {
                // The EOG must be written to the KV: the canonical render carries the EOG marker after the assistant message
                // (§5.4), and llama.cpp also decodes the EOG before stopping. Not writing it would break the invariant.
                stopped_by_eog = true;
                self.prev_tokens.push(sampled.token_id);
                if self.prev_tokens.len() > REPEAT_LAST_N {
                    self.prev_tokens
                        .drain(0..self.prev_tokens.len() - REPEAT_LAST_N);
                }
                self.stream_tokens.push(sampled.token_id);
                let _ = engine.forward(&[sampled.token_id], &[self.current_pos], 1);
                self.current_pos += 1;
                break;
            }
```

The single-shot loop treats EOS as pure control flow — break, drop the
token. The conversation loop cannot: the session's `stream_tokens` mirror
must equal what the chat template would render for the *next* turn, and a
canonical render contains the end marker after each assistant message. So
the EOG token is pushed, decoded through the graph at the cursor (its K/V
land in the regions), and only then does the turn end. Two stop policies,
one principle each: the CLI stops the *text*; the session keeps the *KV
mirror* canonical (`conversation.rs:16-19` — the module's §5.4 note — documents the invariant this
preserves).

### 3.3 Design choices (why this shape and not another)

**Params-only comparison, not build-then-compare.** The original plan
sketch built the new graph and diffed node sequences against the cached one
(`docs/COMPUTE-GRAPH-DESIGN.md` §6 records the correction). That design can
never skip the build — the expensive half of the rebuild — so it saves
nothing. The shipped design inverts the burden of proof: the *builder* is
required to be deterministic in `GraphParams` (each architecture's `build`
doc-comment carries the invariant, `src/models/qwen2/graph.rs:37-38`), and once that
contract holds, six field comparisons are a complete decision procedure.
The failure mode of the contract (a non-deterministic builder) is covered
in debug builds by `verify_structural` (`cache.rs:134-145`), which
re-derives the graph and compares node-by-node — a test-only safety net
that keeps the production path comparison-free.

**Rebuild once, don't parameterize the graph.** An obvious alternative to
"two graphs, one rebuild": a single graph shape that serves both prefill
and decode — e.g. always build for `[n_ctx]` token rows and mask off the
unused ones, or mutate the `n_out` tail in place between phases. Both lose.
A `[n_ctx]`-wide graph would run every activation at the full context width
(4096 rows of scratch instead of 1 for decode) and forfeit the G3 tail
optimization. In-place mutation of shapes would make the *graph* stateful —
the exact property the reuse design is trying to eliminate — and would need
its own invalidation logic, i.e. a second, ad-hoc cache protocol. One
params-derived rebuild per phase boundary is measured in microseconds and
keeps every graph immutable and self-describing.

**The allocator lives in the cache, not in the forward call.** The
`GraphCache` struct holds `graph` *and* `alloc` side by side
`GraphCache` (`cache.rs:38-45`) precisely so that one can be replaced without the other.
If the allocator were created per forward (or per rebuild), every rebuild
would zero the KV cache and chat would forget itself at every turn
boundary; if it were a global keyed by nothing, two concurrent sessions
(server mode, `OPENAI-CHAT-API-PLAN.md`) would share and corrupt each
other's regions. Hence the three ownership tiers that exist in the tree:
the CLI's plain mode uses a process-wide static cache (`graph_cache()`,
`src/models/qwen2/graph.rs:1014-1019` — one run, one session); the conversation engine holds a
session-private cache (`conversation.rs:113-117`, the `GraphEngine` `cache` field); the server hands each slot
its own cache via `forward_graph_cached` (`models/mod.rs:288-298`). Same
mechanism, scoped ownership.

**Positions as data — the founding decision.** The plan's first revision
note tells the story of the alternative (`docs/COMPUTE-GRAPH-DESIGN.md`
revision note 1): an early design encoded `n_past` into the KV-cache op,
meaning the decode topology changed *every step* and reuse was structurally
impossible. The fix was to make the KV a persistent external buffer
addressed by position, with the position injected through an input node —
the design llama.cpp's `allow_reuse` also relies on. Everything else in
this document (params-only comparison, KV-survives-rebuild, append-only
sessions) is downstream of that one move. It also explains the odd-looking
window derivation in §2.2: there is no "cache length" field anywhere in the
engine, because *the query window is data*: it is decoded at execution time
(`decode_window`, `cpu_backend.rs:683`).

**Sample-first loop shape.** The loop samples from logits that already
exist and runs its forward at the *end* of the body. The alternative —
forward first, sample after — would need a dummy forward before the loop
(what would it compute?) or a special first iteration. Sampling first also
gives the stop gates a natural home *before* the forward: a stop decision
costs zero forward time. The one asymmetry (§3.4) is the discarded final
forward when the run ends via `-n` rather than via a stop gate.

**KV regions sized up front, not grown per step.** `ensure_kv` allocates
`nkt × n_ctx` f32 on first touch — the full context window — even though
prefill may use 30 slots. Growing per step (realloc at slot 129, 257, …)
would mean periodic huge copies and, on GPU, buffer re-creation; sizing
once at `n_ctx` means a step's KV write is a plain `copy_from_slice` into
existing memory (`cpu_backend.rs:377-378`, the KV `copy_from_slice`). The cost is bounded by
`--n-ctx`, which the CLI deliberately decouples from the model's
`max_seq_len` (`main.rs:1448-1451` cites the multi-GB over-allocation and
first-submit Metal tax this avoids — `docs/PERF-QWEN3-4B-VS-LLAMACPP.md`
§2). The cursor-bounded window (§2.2) is what makes the slack harmless:
unread region contents beyond it are never touched by attention.

**Two stop policies for EOG, not one.** §3.2's conversation excerpt shows
EOG being *written* to the KV before breaking, while the CLI's loop drops
it. The difference is not inconsistency but the invariant each loop must
keep. The CLI's obligation ends at the printed text; the session's
obligation extends to a KV mirror that the *next* turn will rely on — and
the next turn's canonical template render contains the end-of-message
marker. Breaking the mirror would surface later as subtly wrong context
(the model would effectively "see" a conversation missing its turn
boundaries). llama.cpp makes the same choice — the comment that cites it is
at `conversation.rs:1331-1332`.

### 3.4 Pitfalls & invariants

**1. A position past `n_ctx` is corruption, so every layer of the stack
guards it.** The cursor is incremented per step with no loop-level
ceiling in single-shot mode (only `-n` bounds the run), so the defense is
layered: `forward_cached` asserts `max(positions) < n_ctx` before anything
runs (`src/models/qwen2/graph.rs:507-514` — "fail loudly instead of corrupting memory");
the CPU KV-store kernel bounds-checks each slot and returns `Err` — never
a silent clamp (`cpu_backend.rs:358-360`, the `p >= n_ctx` refusal); the conversation loop checks the
cursor *before* sampling and stops the turn cleanly
(`conversation.rs:1314-1318`, the cursor guard). Three guards, one rule: an out-of-range KV
write would overwrite *another position's* cached vector — the resulting
nonsense would look like a model-quality bug, which is why it must crash
instead.

**2. In-place ops and GPU-pending buffers: never host-copy a KV region's
producer mid-flight.** RoPE and the fused KV-store ops alias their input
buffers in place (sole-consumer rule, doc 07); on a GPU backend the
"buffer" may be un-submitted device memory. Host-copying it at the wrong
moment was the Phase-3 KV-corruption bug (recorded in
`docs/COMPUTE-GRAPH-DESIGN.md` and AGENTS.md rule 5): the copy read stale
bytes and wrote them back over freshly stored K/V. The scheduler's split
boundaries (the boundary's own close-out wait before any cross-backend read,
`sync_backend` (`scheduler.rs:466-471`) are the only sanctioned sync points — which is
also why the decode loop's *reuse* discipline matters: no code path between
steps touches buffers out-of-band.

**3. Liveness and fills must follow build order.** `alloc_graph` computes
buffer lifetimes in node-id (build) order, not topological order, because
`topo_order()` may reorder src-less nodes (like KV loads) ahead of nodes
the scheduler reads first — the G3 tail regression
(`alloc.rs:530-536`, the build-order comment). Related decode-side rule: input buffers are treated
as live for the whole step — `last_use` (`alloc.rs:555-561`) — so that a liveness reuse
can never clobber `token_ids` after it was filled but before its consumer
ran. On a rebuild, `node_to_buf` is cleared and re-derived — the input
*fill* in `forward_cached` happens *after* `alloc_graph` has produced the
new mapping (`src/models/qwen2/graph.rs:635` before `src/models/qwen2/graph.rs:649-653`), so fills always
land in the buffers the scheduler will read.

**4. Stale-but-unread KV after rollback.** `/regen` rewinds the cursor
without erasing the region (`conversation.rs:965-977`, the `regen_turn` rollback), so slots past the
cursor hold abandoned tokens. This is safe *only* because attention reads a
window that ends at the cursor: the span input names the cells per head
(`cpu_backend.rs:1111-1140`, the per-head `vl` window) and the store overwrites slot `p` on the
next append. The invariant "the cursor is the truth; region contents past
it are garbage" is what makes rollback O(1) — but it means *nothing* may
read a region by extent (only by position), or it would see the garbage.

**5. Fusion toggles are part of the identity — A/B tests must keep the
FusionPass.** `fuse_qkv`/`fuse_ffn` live inside `CParams`
(`src/models/qwen2/graph.rs:578-584`), so `MINFER_NO_FUSE_QKV=1` at any point forces a
rebuild with the unfused node set — reliable A/B. The flip side: the fused
and unfused graphs must be *bit-identical* in output, and the unfused path
must still run the FusionPass (AGENTS.md rule 7) so that the only
difference between the two runs is the QKV fusion itself. The test
`fuse_flags_are_part_of_the_reuse_identity` (`cache/tests.rs:36`) pins the
identity half.

**6. Graph uid discipline.** A reused graph keeps its `uid`; a rebuilt one
gets the next monotonic value `replace_graph` (`cache.rs:106-114`). The CUDA backend's
replay cache is keyed by uid — `graph_replay` (`scheduler.rs:286-304`, doc 15) — so the pairing
is load-bearing: same uid ⇒ same topology ⇒ replay is valid; new uid ⇒ the
old capture is orphaned (and simply never requested again). Breaking this —
say, reusing a uid across a rebuild — would replay the wrong graph's
captured launches.

**7. The discarded final forward.** When a run ends by reaching `-n`, the
last iteration's forward has already run at the bottom of the body and its
logits are dropped by the loop condition. One wasted forward per
full-length run (a few milliseconds) buys a loop with a single trivial exit
condition. The stop gates that *can* save it (EOS, stop strings) do, since
they break before the forward — so the cost only applies to runs that end
by quota, and those are exactly the runs where the model would have kept
going anyway.

**8. `n_ctx` is computed once, before prefill, for both phases.**
`ctx = params.n_ctx.max(input_ids.len())` (`main.rs:1452-1453`, the `ctx` computation) with the comment
"Computed ONCE so prefill and decode size the same KV regions"
(`main.rs:1452`, the "Computed ONCE" comment). If prefill and the first decode step disagreed on
`ctx`, they would disagree on `CParams`, the params comparison would fail,
and — worse — `ensure_kv` would have sized regions from the prefill graph
that the decode graph considers too small. One variable, computed early,
keeps the whole run inside one region geometry.

**9. The field that looked like cheap insurance — and was not.**
`GraphParams` used to carry `n_seqs`, "so that adding batching later cannot
silently reuse a single-sequence graph for a two-sequence call". E2 added the
batching and found the reasoning backwards: what a multi-sequence batch changes
about the *topology* is which attention instantiation is built, and that is
`cparams.explicit_span`, decided by the KV reservations. The count itself is
data. Keeping it meant a 2-sequence and a 1-sequence batch of the same shape
described the same graph and still rebuilt it — measured (uid 3 → 4) by
`sequence_count_is_data_not_topology` in `models/qwen2/graph.rs`, which now
pins the reuse instead. An identity field earns its place only if some topology
decision reads it.

## 4. Observe & verify

The decode loop and the reuse machinery are unusually observable — most of
the evidence is on stderr of any plain run.

- **The run header itself.** `Prefill: N tokens in …` vs
  `Generated: M tokens in … (tok/s)` (`main.rs:1479-1483, 1854-1858`, the two rate headers) — two
  deliberately different calibers (pure-decode rate vs blended rate,
  `main.rs:1851-1853`, the pure-decode-vs-blended caliber comment). The gap between the two numbers *is* this
  document's §2.2 argument, measured.
- **`MINFER_TIMING=1`.** Splits the per-token wall clock into sampling vs
  forward milliseconds — `t_samp`/`t_fwd` (`main.rs:1584-1587, 1800-1804`). Expect sampling in the
  tens of microseconds and forward in the milliseconds — the loop's own
  overhead is the difference between the two.
- **`MINFER_GRAPH_TRACE=1`.** The scheduler prints the split layout and a
  per-op/backend census *once per `execute` call* (`scheduler.rs:164-182`, the `MINFER_GRAPH_TRACE` print)
  — i.e. once per token on stderr. A CPU-only run shows a single split; a
  Metal run shows the decode graph's split boundary and the fused-op
  census. Watching it repeat N times for N tokens is the loop made
  visible.
- **`MINFER_TRACE=<path>`.** Records a per-node data trace; the CLI marks
  the prefill/decode phase boundary — `begin_phase` (`main.rs:1459`, `:1556`), tags each decode
  step with its sampled token (`set_token`, `main.rs:1690-1693`), and
  attaches both the prefill graph and a *separate decode graph* JSON at the
  end — `export_graph_json` (`main.rs:1826-1844`). Opening the trace in `viz/` shows the two
  topologies side by side — the G3 tail nodes present in one and absent in
  the other.
- **`--dump-graph <dot>` / `--dump-graph-json <path>`.** Exits after
  exporting the graph the runtime *would* use — labeled `"decode"` when the
  prompt is one token and `"prefill"` otherwise (`main.rs:1521-1525, 1545-1548`, the `kind` label).
  Compare a prefill dump against a decode dump (feed a 1-token prompt) to
  see exactly which nodes the rebuild adds or removes.
- **`MINFER_GRAPH_DUMP=<dir>`.** Writes `logits_decode.f32` plus every
  layer's `kv{ℓ}_decode.f32` *per step, overwritten in place*
  (`src/models/qwen2/graph.rs:688-744`). Each file is the full persistent region (fixed
  2 MiB for 0.5B at `--n-ctx 4096`), so across steps you watch the *valid
  prefix* of `kv0_decode.f32` grow — 512 B of new K values per step
  (128 f32) — which is the append-only KV made tangible; the per-step
  logits dumps let you diff GPU vs CPU decode runs layer by layer.
- **Unit tests.** `src/graph/cache.rs` covers the reuse contract directly:
  `reuse_requires_equal_params` (`cache/tests.rs:126` — n_tokens / gtype /
  weights_version changes all force rebuilds),
  `allocator_survives_rebuild` (`cache/tests.rs:159` — a persistent region
  registered before a rebuild is still there after),
  `fuse_flags_are_part_of_the_reuse_identity` (`cache/tests.rs:36`), and
  `structural_check_detects_different_graph` (`cache/tests.rs:176`). On the
  model side, `forward_cached_isolates_kv_between_caches`
  `forward_cached_isolates_kv_between_caches` (`src/models/qwen2/graph/tests/batching.rs:742`) proves two caches don't share KV. The
  conversation state machine is tested without a model via a mock engine:
  `second_turn_appends_only_delta` (`conversation/tests/turns.rs:48`) asserts the
  delta prefill is smaller than the first turn's, and
  `context_fill_during_decode_stops_cleanly` (`conversation/tests/overflow.rs:22`)
  pins the cursor-vs-`n_ctx` guard.
- **`bench -n N <model>`.** Runs decode for a fixed `N` and reports the
  pure-decode rate in md/csv/json — the reproducible version of the
  `Generated:` line, and the tool the plan document's §12 expectations
  were written against.

A five-minute experiment that exercises the whole document: run once with
`MINFER_GRAPH_TRACE=1` and a two-word prompt, and count the split-layout
lines — prefill once, then one per generated token, all with *identical*
layouts (same graph reused). Then rerun with `--stop` set to a word early
in the output and watch the trace stop mid-stream without a final forward.

## 5. Cross-references

- [05 — Graph build: the IR and the builder](05-graph-builder-ir.md) — the
  builder whose determinism-in-`GraphParams` contract makes params-only
  reuse sound; the census of what a build actually constructs.
- [07 — The allocator: liveness, buffer reuse, and the KV cache](07-allocator-liveness-kv.md) —
  the other half of this document: `ensure_kv`/`alloc_persistent` and the
  pools that freed liveness buffers return to; doc 07 defines the
  "persistent regions" contract, this doc consumes it.
- [08 — The scheduler: splits, synchronization, and execute](08-scheduler-execute.md) —
  the split walk this document reduces to "one walk per token"; split
  boundaries, cross-backend copies, and sync rules in full.
- [09 — Prefill: the first forward](09-prefill-forward-path.md) — the
  other phase of the loop: same `forward_cached` entry point, `nt` tokens,
  the G3 tail, and the first logits.
- [11 — Attention, vec ops, and the KV cache](11-attention-vecops-kv.md) —
  the math inside the `nkv` window: how one query scans the cached keys,
  and why the store/load pair needs positions.
- [12 — Sampler: from logits to one chosen token](12-sampler.md) — the
  function the loop calls first each iteration, and the `match_stop_suffix`
  matcher behind stop strings.
- [14 — The Metal backend](14-metal-backend.md) /
  [15 — The CUDA backend](15-cuda-backend.md) — this loop unchanged, with
  different `execute_node` implementations; doc 15's CUDA Graph replay is
  keyed by the graph `uid` this doc's cache mints (§3.4 #6).
- [`docs/COMPUTE-GRAPH-DESIGN.md`](../COMPUTE-GRAPH-DESIGN.md) — §6 is the
  design record of params-only reuse (including the rejected
  build-then-compare sketch); the revision notes record the
  positions-as-data correction; §11/§12 sketch the loop refactor and its
  measured expectations.
- [`docs/CLI-CONVERSATION-PLAN.md`](../CLI-CONVERSATION-PLAN.md) — the
  multi-turn session state machine this doc summarized in §2.6: the §5.4
  KV-consistency invariant, delta rendering, rollback, and overflow
  policy.
- [`docs/PERF-QWEN3-4B-VS-LLAMACPP.md`](../PERF-QWEN3-4B-VS-LLAMACPP.md) —
  why `n_ctx` (and hence the KV regions) is decoupled from the model's
  `max_seq_len`.
- [`docs/USAGE.md`](../USAGE.md) — the flags the loop consumes:
  `-n`, `--stop`, `--seed`, `--cnv`, `--session`.
- [GLOSSARY](../GLOSSARY.md) — autoregressive, KV cache, nkv, prefill,
  decode, and the rest of the series' vocabulary in one place.

← [12 — The sampler: from logits to a token](12-sampler.md) · [Index](./README.md) · [14 — The Metal backend](14-metal-backend.md) →

