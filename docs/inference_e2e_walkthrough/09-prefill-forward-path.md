# 09 · Prefill: the first forward

> **Stage**: the graph machinery is ready — build, assign, fuse, allocate,
> execute (docs 05–08) → **this stage: the CLI actually calls it, for the whole
> prompt at once, and gets logits back** → the math inside the graph
> ([10 — CPU matmul kernels](10-cpu-matmul-kernels.md),
> [11 — Attention + vec ops + KV](11-attention-vecops-kv.md)) and the first
> sample ([12 — Sampler](12-sampler.md)).
> **Code**: `src/main.rs` prefill block (L737–769, timing prints L763–769 and
> L1000–1017), the call chain `src/models/mod.rs::forward` (L26) →
> `src/models/qwen2/mod.rs::forward` (L33) → `src/models/qwen2/graph.rs::forward`
> (L384) → `forward_cached` (L403), `src/graph/params.rs` (`GraphParams` L52),
> `src/trace.rs::begin_phase` (L70) — lines verified at commit `e7fa0da`.

## 1. Background — where this stage sits

Docs 01–04 got your typed prompt turned into a list of integer token ids.
Docs 05–08 built the machinery that can run a transformer *forward pass* — a
**forward pass** is one trip through the whole network: embeddings, then every
transformer layer (attention + feed-forward), then a final projection —
without computing anything. The machinery is a **compute graph**: a data
structure that lists every math operation as a node, knows which backend
(CPU, Metal, CUDA) runs each node, where each result lives in memory, and in
what order the nodes execute. What it does *not* have yet is data to chew on
and a caller. This document is that caller: the ~30 lines of `main.rs` that
say "here are the prompt's token ids, please run the graph once over all of
them, and give me the answer."

The answer has a name: **logits**. The last operation of the network — the
`lm_head` matrix multiply — produces one floating-point score per entry of the
model's **vocabulary** (the list of all tokens the model knows; about 151,936
for the Qwen models). A score is not a probability yet — it is an unnormalized
"how plausible does each next token look" number, bigger = more plausible. The
sampler (doc 12) turns these scores into one chosen token. For a 512-token
prompt the raw scores occupy 151,936 × 4 bytes ≈ 608 KB — roughly half a
megabyte of "opinions" per forward.

This specific forward is called **prefill**: the model reads *all* prompt
tokens in **one** graph execution, computing every token's hidden state and —
importantly — writing every token's K/V (key/value, the attention memory) into
the **KV cache**, the per-layer notepad of past attention states. Contrast
that with **decode**, every later step of generation: *one* new token per
forward, reusing the cached graph. The prefill/decode split is the two-act
structure of the whole engine, and most of this document is about the
decisions the boundary forces: why all prompt tokens go through together, why
only the last token's logits are wanted, why the context size is fixed once
for both phases, and why the logits come back as an owned `Vec<f32>` rather
than a borrowed slice.

What would break without this stage working as it does? Three things. If the
prompt went through token-by-token, the engine would stream every weight byte
from memory once *per token* instead of once *per prompt* — prefill would be
hundreds of times more memory traffic for the same arithmetic. If earlier
positions computed full logits too, the biggest matrix in the model would run
512 times more work than needed. And if prefill and decode disagreed about
the KV region size, the cache the prefill just filled would have to be copied
— or silently misplaced — before the first decode step could read it. Each of
these is a design decision with real arithmetic behind it; §2 and §3.3 walk
them.

One more orientation point. The single-shot CLI path this doc follows is the
*simplest* caller: one sequence, one user prompt, one sample per step. The
conversation REPL (`--cnv`) and the HTTP server (`serve`) call the very same
`forward` with their own bookkeeping (append-only KV, per-slot caches); doc 13
returns to them. Everything here — the prefill block, the params it builds,
the logits it receives — is the shared spine of all three.

## 2. Principle — how it works and why

### 2.1 Four words, defined once: forward, prefill, decode, context

- **Forward pass** — one execution of the graph: tokens in, hidden states
  transformed layer by layer, logits out. minfer runs one forward per
  `GraphParams` shape; the graph itself is cached and replayed (docs 05/13).
- **Prefill** — the *first* forward of a run, feeding all prompt tokens at
  once (`n_tokens` = prompt length). It computes two things at once: the
  hidden state of every prompt position (needed so the last position's
  prediction is informed by all of them), and the K/V entries for every
  prompt position (the KV cache's initial content).
- **Decode** — every forward after that: one token in (the one just sampled),
  whose K/V *appends* to the cache, and whose logits pick the next token.
  Same graph skeleton, `n_tokens` = 1.
- **Context** — how many token slots the KV cache has room for (`n_ctx`, the
  CLI flag `--n-ctx`, default 4096). It is a *memory* reservation, not a
  speed knob: it decides how long a conversation can grow before the notepad
  is full.

### 2.2 Why a positions vector exists at all

The prefill block starts with one unassuming line:

```rust
let positions: Vec<usize> = (0..input_ids.len()).collect();
```

Prompt token `i` gets position `i`. Why carry that in a vector — can't the
code just *know* "the i-th token of the prompt is at position i"? It can't,
and the reason is the central invariant of the graph design (docs 05/07):
**KV positions are data, not structure**. The graph's topology never depends
on *where* in the cache a token lands. Instead, every execution fills a
`positions` input buffer, and three different kernels read it as ordinary
data:

- the `kvcache_store` nodes use it to decide *which KV rows to write*
  (position `p` writes row `p` of the layer's K and V regions);
- the `attn` node uses it for **causal masking** — token `t` may only attend
  to positions `0..=pos[t]` (the builder's own words: "`pos` carries the
  per-token write positions (I32 input), needed for causal masking
  (`vl = pos[t]+1`)", `builder.rs:325-327`);
- the `rope` nodes use it as the rotation angle's input (position determines
  the angle — doc 11).

This indirection is what lets *one* graph topology serve prefill *and* decode:
prefill fills `positions` with `0, 1, 2, …, nt-1`; the decode loop fills it
with one running number (`current_pos`, starting at the prompt length,
`main.rs:840`, incrementing per step, `main.rs:940`). If positions were
instead baked into the topology (say, attention shaped to read exactly
`n_past + nt` cache rows), the graph would have to be rebuilt on *every*
decode step, because `n_past` changes every step — and the params-only reuse
scheme of doc 05 §2.7 would collapse. The positions vector is the price of
reusability, paid in a few bytes of input data.

There is a second, subtler payoff: decoupling *slot* from *loop iteration*
means the slot need not equal "how many tokens came before in this call". The
conversation path (doc 13) re-prefills a new user turn at positions that
continue from the previous turn; the server (one `GraphCache` per slot) gives
each slot its own position range. Same graph, different data.

### 2.3 One forward for the whole prompt — the parallelism argument

The obvious naive design is: run the graph once per prompt token, feeding it
tokens `0..=i` each time, i.e. token-by-token prefill. minfer (like
llama.cpp) does the opposite: **one** forward with `n_tokens = nt`. The
reason is arithmetic about what a matmul *is*.

Every weight matrix in the model is read from memory during any forward —
that part is unavoidable; the weights are the model. The question is how much
*useful work* each byte of weight buys. In a decode-shaped forward (`nt = 1`),
each weight byte participates in exactly one multiply-accumulate for the one
token being processed: the forward is dominated by *reading the weights*, not
by math. In a prefill-shaped forward (`nt = 512`), the same weight byte is
reused for all 512 columns of the activation matrix — 512 multiply-accumulates
per weight element. Same memory traffic, 512× the arithmetic. In roofline
language: decode is **memory-bandwidth-bound** (the bottleneck is streaming
the weights), prefill is **compute-bound** (the bottleneck is the ALUs/SIMD
lanes, which are now saturated). That single ratio — arithmetic per weight
byte scales with `nt` — is why prefill throughput is measured in thousands of
tokens/second and decode in hundreds, on the same hardware, and docs 10/14
build their kernel strategies directly on this split.

Token-by-token prefill would throw that away: 512 sequential decode-shaped
forwards stream the weights 512 times. Concretely for Qwen2.5-0.5B (weights
total ≈ 0.5 × 10⁹ elements, mostly in quantized matmul weights): one batched
512-token prefill reads each weight element once and does ≈ 5 × 10¹¹
multiply-accumulates; the token-by-token version does the same math but pays
512× the weight traffic, and *then* still could not beat the batched version
even with infinite bandwidth, because each of the 512 walks also repeats the
per-node dispatch overhead of a 440-node graph (doc 05's census) 512 times.

Two more dividends of batching, beyond bandwidth:

1. **The KV cache is filled in one pass.** The `kvcache_store` nodes write
   all `nt` positions contiguously (`0..nt`) in the same execution that
   computes attention over them. Token-by-token would interleave 512 store
   passes with 512 attention reads, with no benefit — attention for token
   `i` needs exactly the rows `0..=i`, which are all available in the batched
   version the moment each layer's store node runs (build order guarantees
   store-before-attention, doc 08).
2. **One build/assign/fuse/allocate for the prompt.** Each distinct
   `GraphParams` pays the graph-build pipeline once (doc 05 §2.7). One
   prefill forward = one build; the decode graph that follows is a *second*
   build (because `n_tokens` changed), and then hundreds of decode steps
   replay it for free.

The costs of batching are real but bounded: activation buffers scale with
`nt` (a hidden-width buffer on 0.5B is 896 × nt × 4 B ≈ 1.8 MB at
`nt = 512`, doc 07 §2.3), and attention grows quadratically (every token
scores against every earlier token, `O(nt²)` per layer — doc 11). Both are
why real engines *chunk* very long prompts into batches; minfer's CLI keeps
the simple one-shot shape and relies on `n_ctx` clamping (§2.5) to keep the
problem bounded.

### 2.4 Why only the last token's logits (`n_out = 1`)

The prefill forward's fourth argument is `n_out`, and the CLI passes `1`
(`main.rs:757`). This is the tail-row optimization whose *topology* doc 05
§2.8 built; here is the *why*.

A transformer is **causal**: token `t`'s hidden state is computed only from
tokens `0..=t`. A consequence that surprises people the first time: the
forward pass could, in principle, produce a "next-token prediction" for
*every* prompt position — position 17's logits predict token 18, and so on.
But during prefill we already *know* what token 18 is: it is sitting in the
prompt (positions 18..nt-1 are the prompt itself). Training uses those
predictions as the learning signal; **inference does not need them**. The
only position whose continuation is genuinely unknown is the **last** one.
So the CLI requests `n_out = 1`: one row of logits, for one token, the last.

What does `n_out = 1` change? Doc 05 §2.8 drew the line not at lm_head but
one layer earlier: after the last layer's attention output projection, two
`get_rows` nodes select the tail `n_out` rows of the attention output *and*
of the residual. Everything after that — the last FFN block, the final
residual add, the final RMSNorm, and the lm_head — runs on `n_out` rows
instead of `nt`. The row indices themselves arrive via a third input node,
`tail_ids`, filled with `[nt-n_out .. nt)` before execution — again data, not
structure. The saving is the largest single-matrix win in the graph: for a
30-token 0.5B prompt, full-`nt` lm_head costs 30 × 896 × 151936 ≈ 4.1 × 10⁹
multiply-accumulates; the last-row-only version costs 1.4 × 10⁸ — 30× less on
the widest matrix in the model (doc 05 §2.8 measured the whole-graph effect
at ~+55% prefill throughput).

Worth pausing on why `n_out = 1` is *safe* here and not a general-purpose
setting: it is a property of the **single-sequence CLI**. The engine keeps
`n_out` as a `GraphParams` field precisely because other callers want more —
a trainer computing loss over all positions would pass `n_out = nt` (and the
graph would keep every row), and the graph's own fallback (`forward_cached`,
`graph.rs:613-626`) handles the `n_out == nt` case with the same code path.
The CLI's choice is the degenerate, cheapest corner of a general mechanism.

### 2.5 Sizing the context once for both phases

Right before the forward call, `main.rs` computes:

```rust
let ctx = params.n_ctx.max(input_ids.len());
```

and passes that single number to *every* forward of the run — prefill at
`main.rs:757`, each decode step at `main.rs:932`. The comment above it (L744–748)
records the three facts that make this the right shape:

1. `n_ctx` **sizes the graph's KV regions** — the allocator carves each
   layer's K and V regions as `n_kv_embd × n_ctx` f32 elements, once (doc 07
   §2.5). It is *not* the model's `max_seq_len`: using that unconditionally
   was the 12 GB lesson of `docs/PERF-QWEN3-4B-VS-LLAMACPP.md` §2 (a 10-token
   prompt reserving 12.1 GB of KV and paying a 3× first-token Metal submit
   tax).
2. It **never shrinks below the prompt length** (`max`): the prompt itself
   needs `nt` KV slots, positions `0..nt`, and the decode loop continues at
   `current_pos = input_ids.len()` (`main.rs:840`) — if `ctx < prompt len`,
   the very first decode position would overflow the regions.
3. The model **clamps again** to `max_seq_len` (`n_ctx.min(max_seq_len)`,
   `graph.rs:392`): a prompt longer than the model was trained for is capped
   at the model's own limit, the same way llama.cpp clamps. The two clamps
   chain: CLI requests → `max(prompt)` floors it → `min(max_seq_len)` caps
   it. One number survives.

The byte arithmetic for that number (per token of context headroom; the
regions are f32):

```
per token, per layer : 2 regions (K + V) × n_kv_embd × 4 B
Qwen2.5-0.5B         : 2 × 128  × 4 B = 1 KB/layer × 24 layers =  24 KB/token
  at n_ctx 4096      : 24 KB × 4096 ≈ 96 MB total (doc 07 §2.5's ≈100 MB)
Qwen3-4B             : 2 × 1024 × 4 B = 8 KB/layer × 36 layers = 288 KB/token
  at n_ctx 4096      : 288 KB × 4096 ≈ 1.2 GB   (doc 01 §2.4; doc 07 §2.5)
  at max_seq_len 40960 (the old bug):          ≈ 12.1 GB  (!)
```

Now the *once* part, which is the design question hiding in the comment's
last line ("Computed ONCE so prefill and decode size the same KV regions").
Suppose prefill passed `max(4096, prompt=512)` = 4096 but the decode loop
recomputed with some other value. Two failure modes, both bad:

- **A larger decode `n_ctx` would be silently ignored.** `ensure_kv` sizes
  each region on **first use only** and returns the existing pair thereafter
  (doc 07, excerpt 5: "The region is also sized on *first use only*: if a
  later graph asked for a different size, it would silently get the old
  buffer"). The decode graph would *declare* bigger regions (`kvcache_store`
  shape `[n_kv_embd, n_ctx]`) but write into the small ones — an
  out-of-bounds write waiting for a long generation.
- **A smaller decode `n_ctx` would trip the pre-flight assert** —
  `forward_cached` asserts `max(positions) < n_ctx` (`graph.rs:415-420`) —
  after the prefill already filled positions `0..512`. Loud, but a crash the
  design made unnecessary.

And there is a third reason that is about *work*, not safety: `n_ctx` lives
inside `CParams`, which is part of the reuse identity (`params_match`,
doc 07 §2.8). Any change forces a graph rebuild. During a rebuild the graph
is replaced but the allocator — and with it the KV regions and their freshly
written prefill contents — is kept (doc 07 §2.5). So identical `n_ctx` buys
the best possible outcome: the prefill→decode transition rebuilds the graph
(because `n_tokens` changed 512 → 1), the regions survive untouched, and
decode's very first attention reads exactly the rows prefill wrote. Zero
copies, zero refills.

### 2.6 What comes back: an owned `Vec<f32>`, moved not copied

`forward` returns `Vec<f32>` — the logits of the last `n_out` tokens,
concatenated in token order. For `n_out = 1` that is exactly one row of
`n_vocab` f32 values: 151,936 × 4 B ≈ 608 KB on 0.5B/Qwen3. (The code
comment says "607 KB"; 151,936 × 4 = 607,744 bytes — same number, rounding
choice.)

Where the bytes travel is worth tracing, because it explains the *owned*
return type. Inside `forward_cached`, after `execute` returns, the logits
live in the graph's output buffer — a pool buffer owned by the allocator
(doc 07). The engine copies them out exactly once:

```rust
let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
// R3-A2: the buffer is always exactly n_out*nv (G3-reduced, or
// n_out == nt) — skip the redundant full-logits clone.
if logits.len() == n_out * nv { logits } else { logits[..n_out * nv].to_vec() }
```

(`graph.rs:617-625`.) That one copy is unavoidable: the pool buffer will be
overwritten by the *next* forward's lm_head (the allocator pins it only for
the duration of one execute, doc 07 §2.2), so the caller must own its own
copy before the next step. What the design *avoids* is every copy after
that: `copy_to_cpu` already returns a `Vec` sized exactly `n_out × nv` (the
`else` branch — the slice-and-`to_vec()` second copy — is dead on the
G3-reduced path; the comment records its removal as R3-A2), and from there
the value travels by **move**, Rust's zero-byte ownership transfer:

```rust
let logits = model.forward(&input_ids, &positions, &mut kv_cache, 1, ctx); // owned Vec
let last_logits: Vec<f32> = logits;      // move, main.rs:758
...
let mut logits = last_logits;            // move into the decode binding, main.rs:833
...
let sampled = sampler::sample_with_penalties(&mut logits, ...); // borrow, mutate in place
...
logits = model.forward(&[sampled.token_id], &[current_pos], ...); // move again, main.rs:932
```

The decode loop's comment (`main.rs:924-925`) states the stakes: *forward()
returns n_out\*nv logits (n_out=1 for single-token decode, exactly n_vocab),
so move the Vec in place instead of copying 607 KB/token.* At ~300 decode
tokens/second, an extra 608 KB copy per token would be ~180 MB/s of pure
`memcpy` — a measurable tax on a loop that is already memory-bound. The
borrowed-slice alternative (`forward` returning `&[f32]` into the pool
buffer) is not viable for a second, harder reason: the borrow would pin the
allocator for as long as the caller holds the logits, but the very next
statement needs `&mut` access to run the next forward — the borrow checker
forbids the loop outright. And even ignoring the checker, the pointed-to
buffer is recycled by the next execute; a held slice would read *garbage*
(the next token's logits, or whatever the pool put there). Owned-at-the-
boundary is the minimal-copy, borrow-checker-friendly shape: exactly one copy
per forward (pool → caller), zero after that. Doc 13 picks this thread up
from the decode side.

## 3. Implementation

### 3.1 Data in / data out

**In** (what the prefill block holds when it calls `forward`):

- `input_ids: Vec<u32>` — the tokenized prompt from doc 04; becomes the
  `token_ids` input node, shape `[nt, 1, 1, 1]`, typed `I32` (carried as f32
  bit patterns, doc 07 §2.6).
- `positions: Vec<usize>` — `0..nt` (§2.2); becomes the `positions` input
  node, same shape.
- `n_out = 1` — the tail-row count (§2.4).
- `ctx = max(params.n_ctx, nt)` — the KV region width for the whole run
  (§2.5); `params.n_ctx` defaults to 4096 (`main.rs:74`).
- `kv_cache` — a legacy `KVCache` object created at `main.rs:657` and passed
  as `&mut kv_cache`. On the graph path it is **ignored** (the parameter is
  named `_kv` at `graph.rs:388`); the graph owns KV in its persistent
  regions (doc 03 covers why the object still exists — the type predates the
  graph refactor and remains the pre-graph API shape).
- The model itself: hparams and weight tensors, registered by name in the
  allocator's registries (doc 03).

**Out:**

- `Vec<f32>` of `n_out × n_vocab` = `1 × 151936` logits — the last prompt
  token's unnormalized scores over the vocabulary (§2.6). On the graph path
  this is one `copy_to_cpu` out of the output buffer, owned by the caller.
- **A filled KV cache** — the second, invisible output: every layer's K and
  V regions now hold rows `0..nt` (written by the `kvcache_store` nodes),
  which every subsequent decode step reads. Nothing is returned for it; the
  regions live in the `GraphCache`'s allocator and simply persist (doc 07).
- A wall-clock `prefill_time` covering the whole call — including the
  one-time graph build, backend assignment, allocation, and (on Metal) the
  first-submit setup cost (§3.2, timing calibers).

### 3.2 Key code

#### The prefill block (`src/main.rs:737-769`)

```rust
    // === Prefill ===
    let infer_start = Instant::now();
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    // forward() computes logits for only the LAST n_out tokens (n_out=1 here:
    // single sequence, only the final token is sampled). llama.cpp does the same
    // via ggml_get_rows(inp_out_ids) at the last layer, shrinking the lm_head
    // to n_outputs rows — saves the full-nt output GEMM + logits download.
    // n_ctx (--n-ctx, default 4096) sizes the graph KV regions — NOT the
    // model's max_seq_len, which would allocate 12 GB+ and pay a first-submit
    // Metal tax (docs/PERF-QWEN3-4B-VS-LLAMACPP.md §2). It never shrinks below
    // the prompt length, and the model's forward clamps it to max_seq_len.
    // Computed ONCE so prefill and decode size the same KV regions.
    let ctx = params.n_ctx.max(input_ids.len());
    // P2 trace (MINFER_TRACE=<path>): the scheduler records per-node data; the
    // CLI marks phase boundaries and attaches tokens/logits for the page.
    let trace_on = crate::trace::enabled();
    if trace_on {
        crate::trace::begin_phase("prefill");
    }
    let logits = model.forward(&input_ids, &positions, &mut kv_cache, 1, ctx);
    let last_logits: Vec<f32> = logits;
    if trace_on {
        crate::trace::attach_step(&last_logits);
    }

    let prefill_time = infer_start.elapsed();
    println!(
        "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        input_ids.len(),
        prefill_time.as_secs_f64(),
        input_ids.len() as f64 / prefill_time.as_secs_f64()
    );
```

Segment by segment: L738 starts the prefill stopwatch (after tokenization —
tokenize cost is *not* in the prefill number); L739 builds the positions
vector (§2.2); L740–748 is the comment that documents `n_out` (and its
llama.cpp analogue `inp_out_ids`), the `n_ctx` sizing policy, and the
compute-once rule; L749 the double-clamped context (§2.5); L753–756 opens
the `prefill` trace phase (only when `MINFER_TRACE` is set — one env read
per run, hoisted); **L757 is the stage itself** — one call, all prompt
tokens, `n_out = 1`; L758–761 renames the result and (for the trace page)
attaches the logits' top-5 to the prefill step; L763–769 prints the prefill
caliber.

#### The call chain — four hops, one delegation each

`main.rs:757` calls through the architecture-agnostic trait object
(`Box<dyn ModelDef>`), so the static type knows nothing about Qwen. Each hop
adds exactly one concern:

```
main.rs:757   model.forward(&input_ids, &positions, &mut kv_cache, 1, ctx)
  │           trait method (models/mod.rs:26-33): the architecture-agnostic
  │           signature; doc comment: "the legacy `kv` arg is ignored" on the
  ▼           graph path; callers must guarantee positions[i] < n_ctx
qwen2/mod.rs:33-42   impl ModelDef for Qwen2Model
  │           one line: graph::Qwen2Graph::forward(self, tokens, positions, kv, n_out, n_ctx)
  ▼           (qwen3/mod.rs mirrors this identically for Qwen3)
qwen2/graph.rs:384-395   Qwen2Graph::forward
  │           n_ctx = n_ctx.min(model.hparams.max_seq_len)  ← 2nd clamp
  │           locks the process-global GraphCache (graph_cache())
  ▼           delegates to forward_cached — the CLI wrapper; server code
              calls forward_cached directly with a slot-scoped cache
qwen2/graph.rs:403-626   forward_cached — the real work (below)
```

The two trait-level aliases on the way are worth one glance
(`models/mod.rs:48-75`): `forward_graph` and `forward_graph_cached` expose
the same two implementations at trait level — `forward_graph_cached` is the
server/multi-slot entry point (it takes an explicit `&mut GraphCache`
instead of using the process-global one). The CLI goes through plain
`forward` and lands in the same `forward_cached`; there is one forward-pass
implementation, not one per caller.

#### `forward_cached`: the build → fill → execute → read pipeline
(`src/models/qwen2/graph.rs:403-470`, `520-550`, `617-625` — abridged)

```rust
    pub fn forward_cached(model: &Qwen2Model, tokens: &[u32], positions: &[usize],
                          n_out: usize, n_ctx: usize, cache: &mut GraphCache) -> Vec<f32> {
        let nt = tokens.len();
        debug_assert!(n_out <= nt);
        // Out-of-range positions would write past the KV regions (which are
        // sized n_kv_embd * n_ctx): fail loudly instead of corrupting memory.
        if let Some(&maxp) = positions.iter().max() {
            assert!(maxp < n_ctx, "position {maxp} exceeds n_ctx {n_ctx} ...");
        }
        /* metal_on / cuda_on: device present AND every weight registered —
           all-or-nothing GPU participation (docs 03/14/15) */
        let params = GraphParams {
            n_tokens: nt,
            n_seqs: 1,
            n_out,
            gtype: if nt == 1 { GraphType::Decode } else { GraphType::Prefill },
            cparams: CParams { n_ctx, n_batch: nt, flash_attn: false,
                               gpu: metal_on || cuda_on,
                               fuse_qkv: nt == 1 && (metal_on || cuda_on) && ...,
                               fuse_ffn: nt == 1 && (metal_on || cuda_on) && ... },
            weights_version: 1,
        };
        if !cache.try_reuse(&params) {
            /* build → register weights → assign_backends → FusionPass →
               alloc_graph → cache.replace_graph (docs 05-08) */
        }
        let (graph, alloc) = cache.current().unwrap();
        // refresh input data (positions/ids are data, not topology)
        alloc.fill_input_i32(graph, "token_ids", &ids).unwrap();
        alloc.fill_input_i32(graph, "positions", &pos).unwrap();
        if graph.inputs.iter().any(|&i| graph.node(i).name == "tail_ids") {
            let tail = ((nt - n_out)..nt).collect::<Vec<u32>>();
            alloc.fill_input_i32(graph, "tail_ids", &tail).unwrap();
        }
        let sched = BackendScheduler::new();
        sched.execute(graph, alloc).unwrap();
        /* MINFER_GRAPH_DUMP block omitted (logits + KV dumps for debugging) */
        let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
        if logits.len() == n_out * nv { logits } else { logits[..n_out * nv].to_vec() }
    }
```

This is the whole "call side of the graph machinery" in one function, and it
is worth reading as five beats:

1. **Pre-flight assert** (L415–420): every position must be `< n_ctx` — the
   loud version of the KV-overflow check that the store kernel also enforces
   (doc 07 §3.2, excerpt 8).
2. **GraphParams construction** (L438–470): six fields, all derived from the
   call's arguments plus device availability. This is the prefill graph's
   birth certificate — for a 512-token CPU prompt: `n_tokens = 512`,
   `n_seqs = 1`, `n_out = 1`, `gtype = Prefill`, `cparams = { n_ctx: 4096,
   n_batch: 512, flash_attn: false, gpu: false, fuse_qkv: false,
   fuse_ffn: false }`. Note the fusion flags are `nt == 1 && gpu`: the
   decode fusions of doc 05 §2.6 are **off** during prefill by construction —
   they pay off at `nt = 1` only, and their being params-derived is what
   keeps fused and unfused graphs reproducible (and A/B-able via
   `MINFER_NO_FUSE_QKV=1`).
3. **Reuse-or-build** (L472–518): `try_reuse` compares the six fields
   against the cached graph (doc 05 §2.7); on mismatch — which for the CLI
   happens exactly twice, at the prefill call and at the first decode call —
   the full build → assign → fuse → allocate pipeline runs, and
   `replace_graph` swaps the graph in *keeping the allocator* (hence the KV
   regions).
4. **Fill inputs, execute** (L520–550): the three input buffers are
   overwritten with this step's data (by *name* — node ids shift between
   rebuilds, names do not), then the scheduler walks the nodes (doc 08).
   `execute` returning `Ok(())` is the guarantee that every buffer — logits
   included — holds its final value (doc 08 §2.4).
5. **Read the output** (L617–625): one `copy_to_cpu` of the output buffer,
   returned as an owned `Vec` (§2.6).

#### `GraphParams` → topology: what each field changes

Doc 05 established the mapping; here is the prefill-relevant digest, with
the field's home in `params.rs`:

| Field (`params.rs`) | Prefill value (CLI) | Topology effect (doc 05 §) |
|---|---|---|
| `n_tokens` | prompt length | every activation shape's `nt`; decode-fusion gates read `nt == 1` |
| `n_seqs` | 1 | batch dimension placeholder (single sequence) |
| `n_out` | 1 | tail `get_rows` pair after the last attention projection (§2.8); the whole output stack runs on 1 row |
| `gtype` | `Prefill` | part of the reuse identity; with `nt` it names the graph class (§2.6) |
| `cparams.n_ctx` | `max(--n-ctx, prompt)` | KV store/load node shape `[n_kv_embd, n_ctx]` → region size (doc 07 §2.5) |
| `cparams.gpu` | false (CPU) / true | backend-assignment eligibility — `Cuda`/`Metal` claim nodes only when every weight is registered |
| `cparams.fuse_qkv` / `fuse_ffn` | false (`nt > 1`) | decode-only fused topologies; prefill builds the plain matmul+rope+store skeleton |
| `weights_version` | 1 | invalidates reuse if weights change (future LoRA/reload hook) |

(`src/graph/params.rs:10-63`; `GraphType` L10–15, `CParams` L22–48,
`GraphParams` L50–63. The module doc's first line is the invariant: these
are "the ONLY inputs to graph reuse".)

#### The trace phase boundary (`src/trace.rs:70-80`, CLI at `main.rs:753-761`)

```rust
/// CLI: mark the start of a phase (prefill / decode). Repeated calls within the
/// same phase are no-ops; a kind change starts a new phase.
pub fn begin_phase(kind: &str) {
    let mut t = trace().lock().unwrap();
    match t.phases.last() {
        Some(p) if p.kind == kind => {}
        _ => t.phases.push(Phase { kind: kind.into(), steps: Vec::new(), graph: None }),
    }
}
```

`begin_phase("prefill")` opens a `Phase` record; the scheduler then appends
one `Step` per `execute()` call (`begin_step`, called from the scheduler —
prefill is *one* step, each decode forward one more), and
`attach_step(&last_logits)` (L759–761, `trace.rs:130-145`) staples the
prefill logits' top-5 `(token_id, probability)` pairs onto that step. The
JSON export (`trace.rs::finish`) embeds the prefill-phase graph, so the viz
page can show the whole prefill execution node by node. Note the phase
*naming* lives in the CLI, not the engine: the engine has `GraphType`, the
trace has human-facing labels.

#### The timing calibers (`src/main.rs:763-769`, `837`, `1000-1017`)

Three stopwatches, three printed calibers — and the differences between them
are the point:

```rust
    let prefill_time = infer_start.elapsed();          // L763 (started L738)
    println!("Prefill: {} tokens in {:.2}s ({:.1} tok/s)", ...);   // prompt / prefill wall

    let gen_start = Instant::now();                    // L837, "pure-decode start"
    ...
    let gen_time = gen_start.elapsed();                // L1000
    let total_time = infer_start.elapsed();            // L1001
    // Pure-decode rate (generated tokens / decode time) — matches llama.cpp's
    // "Generation:" caliber. The "Total:" line below keeps the previous blended
    // caliber (prompt+generated / prefill+decode) for comparison.
    println!("Generated: {} tokens in {:.2}s ({:.1} tok/s)", ...); // L1006-1011
    println!("Total:     {} tokens in {:.2}s ({:.1} tok/s)", ...); // L1012-1017
```

- **Prefill** = prompt tokens ÷ wall time of the single prefill forward. It
  *includes* the one-time costs of the run: graph build, backend assignment,
  allocation, and on Metal the first-submit setup (the 3× first-token tax of
  doc 07 §3.3). Long prompts amortize it; a 1-token prompt measures mostly
  setup.
- **Generated** = generated tokens ÷ decode-only wall time (`gen_start` L837
  → L1000). This is the llama.cpp/llama-bench **"Generation"** caliber:
  steady-state speed, the number people mean by "tok/s". It includes the
  sampler and streaming write per token — which is why `MINFER_TIMING`
  exists to split it further (§4).
- **Total** = (prompt + generated) ÷ (prefill + decode) — the blended
  caliber llama.cpp prints as its "Total" line; dominated by whichever phase
  has more tokens.

Why split the calibers at all, rather than one honest number? Because the two
phases are bottlenecked by *different* resources, and one blended number
hides both. Prefill is **compute-bound** (§2.3): every weight byte is reused
for `nt` multiply-accumulates, so throughput scales with ALU/SIMD throughput
and kernel efficiency — GPUs love it, and that is where int8 MMQ prefill
(doc 15) pays. Decode is **memory-bound**: each token streams the entire
weight set from memory to do only `2 × n_params` flops with it, so
throughput tracks memory bandwidth, not arithmetic. A hardware or kernel
change (say, faster dot products) moves prefill a lot and decode barely; a
memory-side change (quantization, bandwidth) moves decode and barely touches
prefill. Reporting them separately is what makes such measurements
interpretable — and `minfer bench` (§4) adopts llama-bench's `pp`/`tg` test
split for exactly this reason (`bench.rs:1-12`: "pp\<P\>: prefill-only …
tg\<T\>: prefill P context tokens (untimed setup), then time the decode").

### 3.3 Design choices (why this shape and not another)

**Q1: Why process ALL prompt tokens in ONE forward instead of token-by-token?**
§2.3 gave the bandwidth argument; the full tally has four legs:

1. *Amortized weight traffic.* One batched forward reads every weight byte
   once and reuses it for all `nt` tokens' math (each weight element feeds
   `nt` multiply-accumulates instead of 1). Token-by-token multiplies the
   dominant cost of decode-shaped work by `nt` for zero extra information.
2. *One graph build.* The build → assign → fuse → allocate pipeline runs
   once per distinct `GraphParams`; token-by-token would run it `nt` times
   (or force one graph to serve growing `nt`, which violates the
   topology-=-f(params) reuse identity).
3. *KV filled in one pass.* The `kvcache_store` nodes write positions
   `0..nt` contiguously in the same execution; build order guarantees each
   layer's store precedes its attention (doc 08 §2.1). The batched attention
   is also the natural shape for the causal-mask kernel: token `t` reads
   rows `0..=pos[t]` of a region that is already there.
4. *Parallelism inside the kernels.* Wide matmuls (`[out, nt]` outputs,
   doc 05 §2.4's shape convention) give SIMD lanes and GPU threadgroups
   `nt` columns of independent work — the difference between a GEMV (one
   token: bandwidth-bound) and a GEMM (many tokens: compute-bound). This is
   precisely why prefill on CUDA uses a different kernel family (int8 MMQ)
   than decode (MMVQ), doc 15.

The alternative was rejected on measurement, not taste: llama.cpp made the
same choice, and minfer's own fusion A/B numbers (doc 05 §2.6) show how even
*decode*-side batching decisions are gated by measurements — prefill batching
is the same discipline applied where the win is 100×, not 10%.

**Q2: Why only the last token's logits?** §2.4 gave the causality argument;
the design-shaped summary: for an autoregressive *generator*, every prompt
position except the last has a known continuation (it is in the prompt), so
computing logits for them buys nothing the CLI can use. The `n_out` mechanism
generalizes (any tail count; `n_out = nt` restores full logits and is the
code's own fallback path, `graph.rs:613-626`), so choosing `1` is a *caller
policy*, not an engine limitation — the engine still offers every row to
callers who want them (trainers, scoring tools). The placement of the cut
*after the last attention projection* (not at lm_head) is doc 05's §2.8
refinement: everything downstream of the tail select — last FFN, final norm,
lm_head — shrinks with it, and the measured effect was ~+55% prefill
throughput on 0.5B.

**Q3: Why is `n_ctx` sized once for both phases?** §2.5 gave the failure
modes; the principle underneath: the KV regions are **process-lifetime
state**, allocated on first use inside an allocator that outlives every
graph (doc 07 §2.8). Their size is therefore a *run-level* decision, not a
*phase-level* one — and the only run-level facts available are the CLI flag
and the prompt length. Sizing per phase would either silently reuse the
first phase's regions (too small → corruption on later positions) or force a
region reallocation (a full KV copy, or a loss of everything prefill just
wrote — the cache *is* the run's memory). One number, computed once from
`max(CLI, prompt)` and capped by the model, is the smallest contract that
makes "prefill fills, decode appends" work with zero copies.

**Q4: Why does `forward` return owned `Vec<f32>` instead of a borrowed
slice?** §2.6 gave the mechanics; the ownership-shaped summary:

1. *The pool buffer's lifetime is shorter than the caller's need.* The
   logits buffer is pinned for one execute (doc 07 §2.2); the next forward
   overwrites it. A borrowed return would hand the caller a reference into
   memory whose contents are dead by the time the loop iterates.
2. *The borrow checker agrees.* Holding `&[f32]` into the allocator's pool
   while calling `forward` again (which needs `&mut GraphCache`) is a
   compile error — the loop shape *requires* ownership at the boundary.
3. *One copy is the minimum anyway.* The data must leave the pool before
   the next execute; `copy_to_cpu` produces the owned `Vec` at exactly
   `n_out × nv` size (R3-A2 removed the second, slice-`to_vec` copy that a
   non-shrunk logits buffer would have needed). After that, every handoff —
   prefill binding → decode binding → sampler borrow → next forward's move
   — transfers 8 bytes of pointer/len/cap, never the 608 KB payload. The
   recorded alternative, copying 607 KB/token at ~300 tok/s, is ~180 MB/s of
   pure memcpy inserted into the engine's most memory-sensitive loop.

### 3.4 Pitfalls & invariants

- **Positions must satisfy `positions[i] < n_ctx` — always.** The preflight
  assert (`graph.rs:415-420`) and the store kernel's hard error
  (`cpu_backend.rs:169-171`, doc 07) both enforce it, because a bad position
  is an out-of-bounds write into a persistent region. The CLI satisfies it
  structurally: positions are `0..nt` and `ctx ≥ nt` by the `max` clamp.
- **`n_ctx` is a run-level constant.** Passing a different `n_ctx` to decode
  than to prefill breaks the run (§2.5, Q3): either silently (regions sized
  on first use) or loudly (the preflight assert). This is why the comment
  says "computed ONCE" — and why `main.rs:932` passes the *same* `ctx`
  binding, not a recomputation.
- **The legacy `kv_cache` argument is dead on this path.** Passing `&mut
  KVCache` keeps the pre-graph API shape alive (doc 03); nothing reads it in
  `forward_cached`. Do not "optimize" it away without a breaking API change —
  the trait signature is the compat surface for the server's gradual
  migration (`models/mod.rs:23-25` documents the contract).
- **The prefill timer includes one-time setup.** Comparing `Prefill:` lines
  across runs of different prompt lengths (or across builds with different
  fusion env toggles) mixes steady-state prefill cost with build/first-submit
  cost. For steady-state numbers use `minfer bench` (warmup + reps,
  `bench.rs:292`), not the CLI's single-shot print.
- **`n_out` is part of the reuse identity.** Changing it rebuilds the graph
  (it changes the tail topology). The CLI never does mid-run; the server
  could, and the params comparison handles it (`params_match`, doc 07
  §2.8) — but a caller that flips `n_out` per step would rebuild per step.
- **Prefill and decode graphs are two graphs, one cache.** The transition
  costs exactly one rebuild (first decode step); the KV regions and the
  allocator survive it (doc 07 §2.5). If decode's first step ever seems to
  "lose" prefill's cache, the bug is in region identity (layer index,
  backend, dtype) — not in the params scheme, which is what the
  `allocator_survives_rebuild` test pins (doc 07 §4).

## 4. Observe & verify

- **The three printed calibers** — any run prints `Prefill: … tok/s`,
  `Generated: … tok/s`, `Total: … tok/s` (§3.2). On 0.5B CPU expect prefill
  in the thousands of tok/s and decode in the hundreds — the compute-bound vs
  memory-bound split of §2.3, visible in two numbers.
- **`MINFER_TIMING=1`** — decomposes the decode side one level further
  (`main.rs:863-868`): per token it times the sampler call (`t_samp`,
  around `sample_with_penalties`, L884–898) and the `forward` call
  (`t_fwd`, L931–939 — "CPU encode + GPU exec + logits download" per the
  comment), then prints one line, e.g. `[MINFER_TIMING] over 512 tokens:
  sample 0.02 ms/tok (1.2%), forward 3.30 ms/tok (98.8%)` (L949–953). One
  paragraph is all it needs: it exists to answer "is my decode time
  *inference* or *sampling/streaming*?" — on every model in the support
  matrix the answer is overwhelmingly forward, which is why the kernel docs
  (10/14/15) own the optimization story.
- **`MINFER_TRACE=/tmp/t.json`** — the trace records a `prefill` phase
  (opened by `begin_phase`, §3.2) with one step per execute; the step carries
  per-node buffer stats for all 440 nodes and the prefill logits' top-5
  (`attach_step`). Load at `viz/index.html` or `minfer viz`.
- **`MINFER_GRAPH_DUMP=/tmp/d`** — writes `logits_prefill.f32`
  (`n_out × n_vocab` lef32 values — the exact `Vec` this doc is about) plus
  per-node and per-layer KV dumps (`graph.rs:554-611`), so CPU-vs-GPU logits
  comparisons start from this stage's output.
- **`--dump-graph / --dump-graph-json`** — rebuilds and exports the prefill
  graph (the same `GraphParams` the runtime used, doc 05 §4): the `n_out`
  tail rows are visible as the two `get_rows` nodes before the last FFN, and
  the node count differs from the decode graph's (440 vs 437 on 0.5B).
- **`minfer bench`** — the measurement-grade version of the calibers:
  `pp<P>` (prefill-only) and `tg<T>` (prefill untimed + decode timed) rows
  with llama-bench-style mean/stdev over reps (`bench.rs:1-12`, greedy
  sampling for reproducibility). Use it instead of single-shot prints for
  any number you intend to compare.
- **Tests** — `graph_logits_match_forward_real_model` (graph path vs the old
  imperative path, the acceptance bar for this whole pipeline),
  `tail_reduction_matches_full_nt` (the `n_out` mechanism of §2.4: reduced
  and full-logits graphs agree on the tail rows), and the reuse quartet in
  `graph/cache.rs` (params-only identity) — all cited with locations in
  doc 05 §4.

## 5. Cross-references

- [01 — CLI args and model resolution](01-cli-args-model-resolution.md)
  §2.4 — the `--n-ctx` side of the double clamp and the 288 KB/token
  Qwen3-4B arithmetic quoted in §2.5.
- [03 — Model dispatch and weights](03-model-dispatch-weights.md) — why the
  legacy `KVCache` object still exists and how weights got registered under
  the names the graph references.
- [04 — Tokenizer and chat template](04-tokenizer-template.md) — produces
  the `input_ids` this stage feeds in.
- [05 — Graph build (IR)](05-graph-builder-ir.md) — the topology this stage
  triggers: §2.6 prefill-vs-decode classes, §2.7 topology = f(GraphParams),
  §2.8 the `n_out` tail rows.
- [06 — Backend assignment and fusion](06-assign-fusion.md) and
  [07 — Allocator, liveness, KV regions](07-allocator-liveness-kv.md) — the
  two middle steps `forward_cached` runs on a cache miss; §2.5 of 07 is the
  KV-region sizing this doc's `ctx` controls.
- [08 — The scheduler: splits, copies, execution](08-scheduler-execute.md) —
  the `execute` call inside `forward_cached`, and the sync guarantee that
  makes the immediately-following `copy_to_cpu` safe.
- [10 — CPU matmul kernels](10-cpu-matmul-kernels.md) — inside the
  compute-bound side: quantized weight × Q8_0 activations, the GEMM shape
  that makes batching profitable.
- [11 — Attention + vec ops + KV](11-attention-vecops-kv.md) — the consumers
  of the `positions` input: causal masking, RoPE angles, KV store/load.
- [12 — Sampler](12-sampler.md) — the first reader of the returned logits
  Vec; [13 — Decode loop + graph reuse](13-decode-loop-graph-reuse.md) — the
  other caller of `forward`, the rebuild at `nt` 512→1, and the logits-move
  continuation of §2.6.
- [14 — Metal backend](14-metal-backend.md) / [15 — CUDA backend](15-cuda-backend.md) —
  the GPU inside the same call; 15's int8 MMQ prefill is the compute-bound
  argument taken to its kernel-level conclusion.
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §3 (pipeline + "timing is
  dual-caliber"), §4.7 (prefill vs decode table) — the compressed version of
  this stage; [`docs/PERF-QWEN3-4B-VS-LLAMACPP.md`](../PERF-QWEN3-4B-VS-LLAMACPP.md)
  §2 — the `n_ctx` over-allocation measurement behind the main.rs comment.

← [08 — The scheduler: splits, copies, execution](08-scheduler-execute.md) · [Index](./README.md) · [10 — CPU matmul: quantized weights × Q8_0 activations](10-cpu-matmul-kernels.md) →
