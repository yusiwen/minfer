# 04 · 8o — Killing the CPU stall at decode start (LANDED)

> **Result**: first decode step 724 → 35 ms (~20×); steady-state decode rate unchanged.
> **Commit**: `65b686c` ("perf(graph): 8o — kill ~1.6 s one-time decode-start CPU stalls (4.4 GB weight re-clone + 1.9 GB concat probe per rebuild)"). **Date**: 2026-08-30.

## 1. Background — where things stood

Phase 7 (row 1) had just raised the CUDA backend: resident weight registry, per-op dispatch,
CUDA Graph capture/replay. 8m/8m② (row 2) and 8n (row 3) had pushed prefill from 30.7 tok/s to
1204 tok/s and attention from 176 ms/layer down to 8.5 ms/layer. The prefill line finally looked
respectable — naturally, decode was next.

But the moment decode was touched, it showed: **the first decode step waits 724 ms before
emitting the first token**, while steady-state decode steps are far faster. This is not a kernel
problem — in nsys the GPU has no kernel queued during that window — it is the host side doing
two things at graph switch that it should never have done.

Background mechanism: minfer's inference runs through a "declarative compute graph" — built
deterministically from `GraphParams` (including `n_tokens`); different parameters mean a rebuild
and a new cached graph. prefill (nt = prompt length) and decode (nt == 1) differ in parameters,
so the prefill→decode switch necessarily triggers one decode graph rebuild. That rebuild should
have been purely structural (node list + buffer planning), but at the time it:

1. **Re-registered every weight on the CPU backend**. The model-side call sites did `t.clone()`
   per weight before handing it to `register_weight`, and `Tensor`'s byte payload is a
   `Cow::Owned` — `clone()` is a full deep copy. All of 7B q4_k_m's matmul weights are ~4.4
   GB, memcpy'd on pure CPU at every graph (re)build, measured as a **~635 ms** pure-CPU
   stall (no CUDA calls, no kernels — just memcpy).
2. **Answered a feasibility question with a "build it and throw it away" probe**. When building
   the decode graph, the engine must decide whether the 28 ffn gate/up weight pairs can be
   concatenated and registered as `blk.{i}.ffn_gu` (the precondition of the G5 FFN fusion);
   the CUDA branch of the time called the eager `concat_rows` — to obtain a can-it-be-done
   answer of type `Option<Vec<u8>>`, it actually re-concatenated ~1.9 GB (7B's 28 gate/up
   pairs), then used only the metadata and dropped the bytes outright. Measured **~920 ms**.

Together, ~1.6 s of one-time startup stall (the commit title's framing). For interactive use,
this 1.6 s lands on a user who has already finished waiting for prefill, and it reads as "the
first word is especially slow"; for bench it pollutes the first-step timing. 8o's job was to
turn both segments into zero cost.

## 2. Principle — the GPU mechanism

This step's "mechanism" is not on the GPU but in the host↔GPU pipeline relationship: **during
the encode phase the GPU is idle**. CUDA Graph capture only happens after the graph rebuild,
weight resolution, and launch arguments are all ready; while the host rebuilds the graph the GPU
has nothing to do. So every 1 ms of needless host work becomes first-step latency 1:1.

The byte arithmetic of the two stalls (7B q4_k_m, matmul weights ~4.4 GB):

- **The weight deep copy**: a `Vec<u8>` memcpy is one read + one write ≈ 8.8 GB of DRAM traffic.
  Single-threaded memcpy runs at ~7 GB/s, so a 4.4 GB copy ≈ 630 ms — matching the measured
  ~635 ms. A pure host memory-bandwidth problem, completely unrelated to the GPU.
- **The eager concat probe**: a freshly allocated ~1.9 GB output buffer (page-faulted on first
  touch) + ~1.9 GB read of inputs + ~1.9 GB write of output ≈ 3.8 GB of traffic, measured ~920
  ms. Yet the question it answers — "are these tensors' types, block structures, and row byte
  counts compatible, and can they be concatenated row-wise" — **requires only reading each
  tensor's `ttype` and `shape`, a few hundred bytes in total**.

The key insight: the `Clone` semantics of `Cow<'static, [u8]>`. The `#[derive(Clone)]`
implementation for `Cow` is: cloning `Borrowed(b)` copies only the reference (O(1)), while
cloning `Owned(v)` deep-copies the whole `Vec` (O(n)). minfer's weight tensors took the `Owned`
path at the time, so a casual `t.clone()` in the model code was a GB-scale memcpy on GB-scale
objects like weights. The type system raises no error, the tests don't fail — it is just slow.

Equally key is the idempotence argument: **weights are immutable after load** (reuse consistency
is guarded by `GraphParams.weights_version`; any future weight change bumps the version first
and forces a rebuild), so a duplicate same-name registration necessarily carries identical data
— skipping it is safe. That is what makes the `contains_key` early-exit lock legitimate.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Do not eliminate the rebuild — only eliminate the duplicated labor inside it.** That
  different prefill/decode parameters trigger a rebuild is orthogonal to the reuse design (the
  same invariant as llama.cpp's `allow_reuse`); touching it is risky, and "rebuild =
  structural work" is the right shape. So the fix turns the two big pieces of work inside the
  rebuild into no-ops, rather than adding a rebuild cache.
- **Idempotent registration rather than making clone cheap by reference.** The alternative was
  switching `Tensor`'s weights to `Borrowed` (mmap zero-copy) so clone becomes cheap — but
  that touches the loader's ownership structure, a wide blast radius. The early-exit lock
  lands in one line: the first registration proceeds as before, and later same-name
  registrations skip.
- **The feasibility probe and the real concatenation must share "the same precondition"**. The
  real concatenation happens once, in the loader (concatenated and registered as
  `blk.{i}.ffn_gu` at load); graph building merely asks "can it be concatenated". The two
  paths' decisions must match branch by branch, or you get the silent mismatch of "the probe
  says yes, the loader concatenated a different layout". So the new function mirrors
  `concat_rows`' precondition checks item by item, and this is written into the doc comment as
  a maintenance contract.
- **No speculative generalization**: the probe is simply `concat_rows_feasible` — no traits, no
  callbacks; the two precondition lists exist explicitly side by side, locked to each other by
  comments. Small, visible duplication drifts less than an abstraction.

### 3.2 Key code

First site: the registration idempotence lock in `src/graph/cpu_backend.rs` (10 lines added):

```rust
/// Register a weight tensor by name (Phase 6 wires this from the model).
pub fn register_weight(&mut self, name: &str, t: Tensor) {
    // Skip re-registration of an already-known weight: Tensor carries its
    // bytes as Cow::Owned, so the `t.clone()` at the model call sites
    // deep-copies the full weight set (~4.4 GB on 7B) on EVERY graph
    // (re)build — measured as a ~635 ms pure-CPU stall at the
    // prefill→decode graph switch (no CUDA calls, no kernels). Model
    // weights are immutable after load (weights_version guards any future
    // change), so a same-name registration always carries the same data.
    if self.weights.contains_key(name) {
        return;
    }
    self.weights.insert(name.to_string(), t);
}
```

The `Tensor` definition (current tree `src/tensor.rs`) explains why the call site's `t.clone()`
is so expensive:

```rust
#[derive(Clone)]
pub struct Tensor {
    pub ttype: TensorType,
    pub shape: [i64; 4],      // ne[0..3]: number of elements per dimension
    pub strides: [usize; 4],  // nb[0..3]: stride in bytes per dimension
    pub data: std::borrow::Cow<'static, [u8]>,  // ← when Owned, clone() = full memcpy
    pub name: String,
}
```

Second site: the metadata-only feasibility probe added to `src/cuda.rs` (the core of the 34
added lines):

```rust
pub fn concat_rows_feasible(tensors: &[&Tensor]) -> bool {
    if tensors.len() < 2 {
        return false;
    }
    let tt = tensors[0].ttype;
    if tensors.iter().any(|t| t.ttype != tt) {
        return false;
    }
    let bq = quant_block_q(tt);          // elements per block
    let bb = quant_block_bytes(tt);      // bytes per block
    if bb == 0 {
        return false;
    }
    let ne0 = tensors[0].shape[0] as usize;
    if tensors.iter().any(|t| t.shape[0] != ne0 as i64) {
        return false;
    }
    if ne0 % bq != 0 {
        return false;
    }
    let row = (ne0 / bq) * bb;           // compressed bytes per row
    tensors
        .iter()
        .all(|t| t.data().len() == row * (t.shape[1] as usize))
}
```

Compared clause by clause against the eager `concat_rows`' preconditions: ≥2 tensors, identical
quantization type, identical `ne0` divisible by the block size, and each tensor's actual byte
length equal to `rows × row bytes`. The final data-length check subsumes the eager version's
trailing `out.len() != rows * row` defense — so both functions give the same can-it-be-done
answer for the same input.

Third site: the call site in `src/models/qwen2/graph.rs`, before → after:

```rust
// before — concatenate 1.9 GB for one bool and throw it away:
crate::cuda::concat_rows(&[fg, fu]).is_some()
// after — reads metadata only:
crate::cuda::concat_rows_feasible(&[fg, fu])
```

### 3.3 Pitfalls

- **The pitfall is language semantics itself**: `Cow`'s derived `Clone` is an O(n) deep copy on
  the `Owned` variant. Whoever writes `t.clone()` is thinking "copy a handle" and gets a 4.4
  GB memcpy. No compile-time or runtime signal whatsoever — only a timer can catch it.
- **The probe had side effects**: `concat_rows(&[fg, fu]).is_some()` looks perfectly harmless
  but actually allocates a GB-scale buffer. The lesson: when you only want a bool, do not call
  a function that returns data — no matter how convenient.
- **The sync risk of two precondition lists**: `concat_rows_feasible` and `concat_rows` are two
  hand-written checks, and the comment states explicitly "must mirror concat_rows'
  preconditions; the data-length check covers the trailing total-length defense". This is a
  deliberate trade — a comment lock instead of an abstraction merge; the price is that any
  future change to `concat_rows` must be mirrored here.

## 4. Verification

This step is a startup-path/host-side change that touches no numeric path, and the source record
sets up no dedicated bitwise-dump gate for it. Its gates are behavioral-equivalence gates:

- **Steady-state decode rates unchanged** (row 4 records "all decode rates unchanged") — defends
  against the "the work we skipped was actually needed" class of regression: if the skipped
  registration or probe had had a needed side effect, the decode step itself would change.
- **The equivalence argument**: the registration early-exit reuses the very `Tensor` registered
  by the same load (ownership unchanged, no copies); the probe changes only "whether bytes get
  constructed", never the returned verdict (the branch-by-branch mirror of §3.2).
- The end-to-end metric "first decode step 724 → 35 ms" is itself the most direct verification:
  it is the observable surface of this very bug.

## 5. Results

- **End to end**: first decode step **724 → 35 ms** (~20×), steady-state decode rates unchanged
  (row 4's Δ records 20×).
- **Component level** (segmented timing): weight re-clone ~635 ms → 0; eager concat probe ~920
  ms → 0; the combined ~1.6 s one-time startup stall disappears (the commit title's framing).
  A note on framing: the 635/920 ms figures are re-assembly numbers from segment-timing the
  rebuild path, while 724 → 35 ms is the end-to-end metric over the "first decode step" window
  — the two are not additive over the same measurement window, and row 4's conclusion defers
  to the latter.
- No new kernels introduced, no numeric path changed; the comparison target is the engine's own
  baseline (before/after, same machine, same model).

## 6. Lessons

1. `Cow::Owned` turns `clone()` into a deep copy — every `clone()` on a startup path deserves an
   audit in bytes (row 4's own words: audit clones on startup paths).
2. Never answer a can-it-be-done question with a "build the result and throw it away" function —
   a feasibility check should read metadata only.
3. Weights are immutable after load (`weights_version` guards changes), so registration can be
   made idempotent; idempotent registration turns the rebuild's duplicate registrations into
   no-ops wholesale.
4. The end-to-end "first decode step" time is the sentinel metric for startup-path regressions —
   when it moves, always check the host side first, not the kernels.

---
← [03](./03-fa-tiled-prefill-attention-8n.md) · [Index](./README.md) · [05 →](./05-persistent-f16-cache-8p.md)
