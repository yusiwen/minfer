# minfer Architecture Execution Plan

**Status:** Phase A in progress (campaign branch, tickets A0–A8).
**Companion to:** `docs/ARCHITECTURE-ROADMAP.md` (what is missing, why, and how it
is ranked). This document is the *how*: phase-by-phase tickets with
deliverables, acceptance criteria and dependencies.
**Baseline:** `HEAD = f32daa7` (2026-09-16).

## 0. Decisions already taken

| Decision | Consequence for this plan |
|---|---|
| **Metal is out of scope this round.** | No ticket here edits `src/graph/metal_backend.rs`, `src/metal.rs` or `src/metal.metal`. Every phase records what it defers into **Phase G (Metal alignment)**. |
| **Dead reuse-identity fields: option (a).** | Delete `CParams.n_batch`; keep `GraphParams.n_seqs` marked *reserved for item 3*. A7 is unblocked — rationale in §8. |
| **The campaign starts with Phase A (A0–A8).** | Phases B–G are planned but out of scope until A's exit criteria are met. |

## 1. Standing rules

Every ticket is rejected if it breaks one of these. They restate the project's
own invariants rather than inventing new ones.

1. **Params-only reuse.** Anything that changes graph topology must join
   `GraphParams` / `CParams` and be compared in `GraphCache::params_match`
   (`src/graph/cache.rs:57-64`); `n_past` never enters the identity.
2. **No silent fallback.** A kernel-invariant violation returns `Err` from
   `execute_node` with the actual values; assignment is decided at build time.
3. **Identity gate.** Any change to a kernel or an execution path is A/B'd
   against the existing path. Bitwise-identical is the default bar; where that
   is impossible, the ticket must name the tolerance class and its cause
   (the project's existing example: `nt ≤ 8` bitwise, `nt = 9` tolerance-class).
4. **GPU safety.** Bounded waits, status checks, device limits queried at
   runtime, no hardcoded device constants.
5. **Index docs move with the code.** `README.md`, `AGENTS.md`,
   `docs/SUPPORT-MATRIX.md` and `docs/ARCHITECTURE-ROADMAP.md` are updated in
   the same commit as the change they describe.
6. **Deferred-Metal marking.** A ticket whose cross-backend design changes
   Metal's behaviour must add a Phase G line *in the same commit*.

## 2. Verification matrix (what this box can prove)

This machine is a **DGX Spark (GB10), aarch64 Linux, CUDA 13.0**
(`/usr/local/cuda-13.0`), with cached Qwen2.5-0.5B/7B/14B and Qwen3-0.6B GGUFs.

| Backend | Build | Run/verify | Note |
|---|---|---|---|
| CPU (aarch64 NEON+SDOT) | ✅ | ✅ | Primary correctness net here |
| CUDA (sm_121) | ✅ | ❌ **unavailable** (A0, 2026-09-16) | Builds; at runtime `cudaGetDeviceCount` returns err 304 (OS/driver call failed) → `CUDA: no CUDA devices found`, graceful CPU fallback |
| Metal | ❌ | ❌ | macOS-only code, not compilable here → Phase G |
| x86 AVX2 / AVX-512 | ❌ | ❌ | Item 11 needs an x86 box or CI |

**A0 verdict (2026-09-16): CUDA is compile-only in this environment.** Every
CUDA-touching ticket's acceptance is therefore "compiles + reviewer-inspected",
never "measured here"; the CPU path is the only runtime net. Design
consequence, already applied in A3: where a guard is needed on all three
backends, put it in the **backend-agnostic** layer (the allocator) rather than
in `cuda_backend.rs` — that keeps the fix fully verified on CPU and needs no
unverifiable GPU code.

## 3. Phase A — instrument, then hazard removal

*Why first:* the roadmap's §5 calls items 5, 6, 13, 26–28 "cheap and
independent". This plan pulls **item 23 (op matrix) and item 24 (CI) to the
front of that batch** — they are the instrument that keeps Phases B–E honest,
and one of them (the op/dtype/backend matrix) would have caught several of the
roadmap §4 defects automatically.

| ID | Item | Title | Effort | Status |
|---|---|---|---|---|
| A0 | — | CUDA access spike on this box | S | ✅ done — **unavailable** (compile-only) |
| A1 | 23 | Op × dtype × backend correctness matrix | M | |
| A2 | 24 | CI: test on Linux/CPU, build on CUDA, keep macOS build | S | |
| A3 | 5 | KV bounds guard + `ensure_kv` size check | S | ✅ done |
| A4 | 6 | Server worker panic isolation | S | |
| A5 | 27 | Re-key cross-backend staging by `(node, dst_backend)` | S | |
| A6 | 28 | Remove CPU per-op allocations | S | |
| A7 | 26 | Dead identity fields | S | ✅ done |
| A8 | 13 | Guard symmetry (docs half + CUDA `FusedQkvNorm`) | S | |

### A0 — CUDA access spike — **DONE (2026-09-16): unavailable**
- **Verdict:** `cargo build --release --features cuda` succeeds (1m23s, targets
  `sm_75…sm_121`, PTX `compute_121`), but at runtime
  `cudaGetDeviceCount` returns err 304 and the engine logs
  `CUDA: no CUDA devices found (cudaGetDeviceCount err 304, count 0)` then
  `CUDA: not available, using CPU fallback`. The CPU path is unaffected
  (Qwen3-0.6B Q8_0: 120 tok/s prefill, 64.7 tok/s decode).
- **Consequence:** the CUDA half of every later ticket is compile-verified only.

### A1 — Op × dtype × backend matrix  · item 23 · M
- **Files:** new `tests/op_matrix.rs` (or `src/graph/*` test module), driven by a
  small table of (op, dtype, backend) cases.
- **Deliverable:** every op in `Backend::supports_op` exercised on every backend
  that claims it, compared against the CPU reference; failures reported as a
  matrix, not as individual panics.
- **Acceptance:** the matrix runs under `cargo test`; the current op-set
  asymmetry (roadmap §4 defect 5) shows up as an explicit row rather than a
  surprise; ≥ 1 real defect is either found or proven absent.
- **Deps:** A0 for the CUDA rows (CPU rows can land first).

### A2 — CI  · item 24 · S
- **Files:** `.github/workflows/ci.yml`.
- **Deliverable:** three jobs — Linux/CPU `cargo test`, macOS `cargo build`
  (unchanged), CUDA `cargo build --features cuda`. Tests that need a model stay
  `#[ignore]`d.
- **Acceptance:** a deliberately broken commit fails the Linux job.

### A3 — KV bounds guard + `ensure_kv` size check  · item 5 · S — **DONE**
- **Files:** `src/graph/alloc.rs` (only — see the deviation note).
- **Deliverable, part 1:** `ensure_kv` now records the element count and the
  backend each region was allocated for, and returns `Err` when a later graph
  asks for a different size or assigns the layer elsewhere. Before this, the
  early return silently handed back a wrongly sized region; the CPU backend
  then failed with a position error and the GPU backends wrote out of bounds.
- **Deliverable, part 2 (deviation from the ticket as written):** the
  `pos < n_ctx` guard was **not** added to `cuda_backend.rs`. Because A0 found
  CUDA unverifiable here, the guard went into
  `GraphAllocator::fill_input_i32` — the single point where positions become
  graph data. It is structural (an I32 input consumed by a KV-writing or
  attention op is bounded by the graph's `n_ctx`, taken from the `kv_load`
  node), so it covers **all three backends including Metal without touching
  `metal.rs`**, costs one O(nt) host scan, and is fully testable on CPU.
  `token_ids` is deliberately exempt (vocabularies exceed `n_ctx`).
- **Acceptance:** `kv_region_size_change_is_a_loud_error`,
  `position_beyond_n_ctx_is_rejected`, `token_ids_are_not_bounded_by_n_ctx`
  (all fail before this change); `cargo test --release` 155 passed / 0 failed.
- **Defers to G:** nothing — the Metal path is covered by the same allocator
  guard. Phase G keeps only the *style* asymmetry (`debug_assert!` vs `Err`).

### A4 — Worker panic isolation  · item 6 · S
- **Files:** `src/server/chat.rs:454-518`.
- **Deliverable:** the worker runs each job under `catch_unwind`; a panic emits
  `StreamEvent::Err` for that job, logs it, and the loop continues. Slot state
  returns to `Idle`.
- **Acceptance:** a test that injects a panicking job gets an error response and
  the *next* request succeeds.

### A5 — Staging map re-key  · item 27 · S
- **Files:** `src/graph/alloc.rs:34`, `:592-653`; `src/graph/scheduler.rs:252-255`.
- **Deliverable:** `cross` keyed by `(NodeId, Backend)`; the consumer-side
  backend filter becomes unnecessary and is removed.
- **Acceptance:** existing scheduler tests pass; a synthetic graph feeding one
  node to two foreign backends stages correctly (new test).

### A6 — CPU per-op allocations  · item 28 · S
- **Files:** `src/graph/cpu_backend.rs:157-158` (K/V source clone), `:195`
  (`Vec<&[f32]>` per node).
- **Deliverable:** borrow instead of clone; reuse a scratch `Vec` across nodes.
- **Acceptance:** CPU decode tok/s on Qwen2.5-7B Q4_K_M does not regress;
  output bit-identical; fewer allocations per token (measured, not asserted).

### A7 — Dead identity fields  · item 26 · S
- **Decision taken: option (a)** — delete `CParams.n_batch`, keep `GraphParams.n_seqs`
  marked *reserved for item 3* (rationale in §8).
- **Files:** `src/graph/params.rs`, `src/graph/cache.rs:57-64`,
  `src/graph/json.rs:40`, and the `n_batch` construction sites in
  `src/models/*/graph.rs`.
- **Deliverable:** `n_batch` gone from `CParams`, from `params_match` and from
  the graph JSON export; `n_seqs` carries a comment naming the item that will
  read it.
- **Acceptance:** no occurrence of `n_batch` remains in `src/graph/`; every
  field compared by `params_match` has at least one reader; `cargo test` green.

### A8 — Guard symmetry  · item 13 · S
- **Files:** `docs/SUPPORT-MATRIX.md` (op × backend column), `src/graph/cuda_backend.rs`.
- **Deliverable:** `SUPPORT-MATRIX.md` gains a per-backend op column so a
  platform-dependent decode path is visible; CUDA gains `FusedQkvNorm` **or**
  the matrix records that it does not have it.
- **Acceptance:** the asymmetry in roadmap §4 defect 5 is either fixed or
  documented; A1's matrix agrees with the table.
- **Defers to G:** the Metal half (`debug_assert!` → `Err`; the weightless
  RMSNorm fallback).

## 4. Phase B — persistent server context

*Why second:* it is independent of the KV redesign, and it is the first
user-visible win — a chat client that resends the conversation stops
re-prefilling it every turn.

| ID | Item | Title | Effort |
|---|---|---|---|
| B1 | 4 | Reproduce the doc-97 contamination | S |
| B2 | 4 | Slot cache retention + prefix check | M |
| B3 | 4 | Measurement + docs | S |

### B1 — Reproduce contamination
- **Deliverable:** a failing test that drives the server (or `worker_loop`)
  through two requests on one slot with different prompts and shows the stale-row
  read documented in `chat.rs:483-490`.
- **Acceptance:** the test fails on today's code for the *documented* reason,
  or the ticket reports that the documented reason does not reproduce and the
  real cause is something else. **Do not build B2 on an unverified cause.**

### B2 — Retention + prefix reuse
- **Deliverable:** the slot keeps its `GraphCache`; a new request reuses the
  cached KV only when the new prompt's first `cached_len` tokens equal the
  cached sequence (append-only cases); otherwise it prefills from position 0
  over the same regions. No cell machinery yet.
- **Acceptance:** B1's test passes; the slot no longer re-allocates KV per
  request; a second turn that extends the first prefills only the new tokens;
  `cudaMemGetInfo`-style allocation churn (or an instrumented counter) drops.
- **Risk:** if append-only reuse cannot be made safe without cells, stop and
  promote Phase C1 ahead of B2.

### B3 — Measurement
- **Deliverable:** prefill tokens and time-to-first-token per turn, before/after,
  on a fixed multi-turn conversation; recorded in `docs/ARCHITECTURE-ROADMAP.md`
  §2.5 or the OpenAI plan.
- **Acceptance:** numbers are interleaved A/B, not sequential (the campaign's
  own clock-drift rule).

## 5. Phase C — KV cell store (item 1)

Five sub-steps, each keeping the tree green. C3 depends on Phase D.

| ID | Title | Effort |
|---|---|---|
| C1 | `KvCache` with cells, single implicit sequence (behaviour-preserving) | L |
| C2 | `seq_rm` / `seq_add`: prefix truncation + context shift | L |
| C3 | Defragmentation (cell copy) — **needs D1** | M |
| C4 | Quantized KV (item 21) | M |
| C5 | State save/restore for session persistence | M |

### C1 — Cell store, one sequence
- **Deliverable:** per-layer cell arenas with an owner set; the KV store node
  resolves `position → cell` on the host and passes an index array to the
  kernel; attention still derives its bound from `positions`, so behaviour is
  unchanged.
- **Acceptance:** bitwise-identical greedy output vs the pre-change binary on
  the full smoke set (0.6B Q8_0, 7B Q4_K_M, 14B Q4_K_M) at several context
  lengths; the graph topology is unchanged (no new topology-affecting params).
- **Defers to G:** nothing — the Metal backend keeps the old regions until G.

### C2 — Removal and shift
- **Deliverable:** `seq_rm` (drop a range) and `seq_add` (shift positions),
  surfaced as graph-level operations; `conversation.rs:597-732` overflow switches
  from "drop the oldest turns + full re-render" to a context shift.
- **Acceptance:** the conversation overflow test passes; a turn that overflows
  no longer re-prefills the whole conversation; the shift path is a **named
  tolerance class** with the reason recorded (positions change ⇒ RoPE inputs
  change ⇒ not bitwise).
- **Defers to G:** n/a.

### C3 — Defragmentation
- **Deliverable:** a cell-copy operation that compacts the arena; triggered when
  fragmentation exceeds a threshold.
- **Deps:** D1 (a copy needs either a view or an explicit copy op).
- **Acceptance:** node-count and arena-utilisation counters before/after; output
  bit-identical.

### C4 / C5
- C4: Q8_0 KV first, behind `MINFER_CACHE_TYPE`, gated to where the kernels
  support it; acceptance = named tolerance class (dequantisation is not
  bitwise) plus a memory-footprint measurement.
- C5: write/read the KV state to a file; acceptance = a session resumed from
  disk produces the same continuation as one that stayed in memory.

## 6. Phase D — IR expressiveness (item 7)

| ID | Title | Effort |
|---|---|---|
| D1 | Strided views with allocator-known aliasing + multi-output nodes | L |
| D2 | Re-express one decode fusion as a composition (proof) | M |
| D3 | Decide the fate of the four hand-written fused ops | S |

- **D1 acceptance:** `Op::View` becomes zero-copy (a test asserts the allocator
  maps a view onto its parent's buffer); the allocator's liveness understands
  `view_src`; existing graphs unchanged and bitwise.
- **D2 acceptance:** `FusedFFN` re-expressed as `MatMul` + views + in-place
  `SwiGLU`; the hand-written node stays behind an env gate for A/B; bitwise
  identity; no decode regression beyond a recorded budget.
- **D3:** with D2 proven, decide per fusion whether to keep the hand-written
  node (performance) or delete it (simplicity). MoE (item 17) is unblocked here.

## 7. Phase E — batching, then memory policy

| ID | Item | Title | Effort |
|---|---|---|---|
| E1 | 2 | IR `seq_id` + explicit attention masks (CPU + CUDA) | L |
| E2 | 3 | Batch composition + continuous batching; make `n_seqs` real | XL |
| E3 | 10 | Chunked prefill: make `n_batch` real | M |
| E4 | 8 | Allocator reserve/assign split + size classes + memory accounting | L |
| E5 | 9 | Layer-offload budget (`n_gpu_layers` equivalent) | L |

- **E1 acceptance:** the mask is an explicit input, not a derivation from
  `positions`; a two-sequence test proves no cross-attention; single-sequence
  output is bitwise unchanged.
- **E2 acceptance:** aggregate throughput at `--n-slots 4` materially exceeds
  the serial baseline on a fixed workload; the `n_seqs` field is either real or
  deleted (closes A7 if it was kept).
- **E4/E5** are what make a model that does not fit in VRAM runnable at all.

## 8. Note — the dead identity fields (A7 rationale)

`CParams.n_batch` and `GraphParams.n_seqs` live in the two structs that define
the graph-reuse identity (`src/graph/params.rs`), and
`GraphCache::params_match` compares both (`src/graph/cache.rs:57-64`). A change
in either forces a full graph rebuild.

They are *dead* because nothing reads them:

- `n_seqs` is set to `1` at every construction site
  (`models/qwen2/graph.rs:443`, `models/qwen3/graph.rs:382`, `graph/json.rs:31`,
  …). No builder, kernel or allocator consults it.
- `n_batch` is set to `n_tokens` at every site (`models/qwen2/graph.rs:452`,
  …) — it is a second copy of a field that already exists, so it can never
  carry independent information.

Both are therefore compared for equality but can never differ: they are inert.

**Why it is worth a ticket.** They read as evidence that the engine supports
multi-sequence batches and prefill chunking, which it explicitly does not
(`docs/COMPUTE-GRAPH-DESIGN.md` §1.3 lists multi-sequence batching as a
non-goal). The concrete hazard is the next implementer: `n_seqs` sitting in the
reuse identity looks load-bearing, so E2 might start by "just setting it",
while every real decision point — attention masking, KV addressing, allocator
liveness — has no notion of sequence ids. Deleting or annotating the fields
forces that realisation at the point where it is cheap.

**Options.**

| Option | What changes | Trade-off |
|---|---|---|
| **(a) Recommended** — delete `n_batch`, keep `n_seqs` with an explicit "reserved for item 3" comment | One field gone | E3 will reintroduce `n_batch` with its real meaning (a chunk size chosen by the scheduler, not `n_tokens`); keeping today's field would make the old meaning the default. `n_seqs` keeps its exact future name and semantics, so nothing is lost by keeping it. |
| (b) Keep both, comment as reserved | Nothing | Minimal churn, but two inert fields remain in the identity and the comment is easy to ignore. |
| (c) Delete both, re-add in E2/E3 | Two fields gone | Cleanest now, but `n_seqs` must be re-added as part of E2 — and E2 is already the largest change in the plan. |

**Decision (2026-09-16): option (a).** Delete `n_batch` in A7; keep `n_seqs`
with a comment naming item 3 as its future reader.

## 9. Phase F — independent tracks

Can run in parallel with A–E by a different workstream.

| ID | Item | Title | Effort | Box needed |
|---|---|---|---|---|
| F1 | 11 | AVX2/AVX-512 dots for the K-quants + weight repacking | L | **x86** |
| F2 | 15 | GBNF-style grammar + JSON-schema constrained decoding | M | this box |
| F3 | 16 | Sampler set: min-p, typical, XTC, DRY, mirostat, logit bias | M | this box |
| F4 | 12 | Backend registry (drop the compile-time enum) | M | this box |
| F5 | 14 | Async cross-backend copy + events | M | this box (CUDA) |
| F6 | 22 | Quantizer tooling (`convert-hf-to-gguf`, `quantize`, `split`) | L | this box |
| F7 | 19/20 | Chat-template fidelity + tokenizer generality | M | this box |

F1 is the only item in this plan that **cannot be verified on this machine**
(aarch64): it needs an x86 box or a new CI runner. It is also the largest
single CPU win, so it should be scheduled against hardware availability, not
against the critical path.

## 10. Phase G — Metal alignment round (deferred by decision)

Nothing in this plan edits Metal. This phase collects the debt so it is not
forgotten:

| ID | Origin | Work |
|---|---|---|
| G1 | A3 | `pos < n_ctx` guard in Metal's `KvcacheStore` |
| G2 | A8 | `debug_assert!` → `Err` for `FusedFFN`/`FusedQKV`/`FusedQkvNorm` `nt == 1` |
| G3 | A8 | Remove the silent weightless-RMSNorm fallback (`metal_backend.rs:403-414`, `:457-468`) |
| G4 | A8 | CUDA/Metal op-set asymmetry: decide whether Metal gains `QkvBiasRopeStore` |
| G5 | C1/C2/E1 | Port the cell store and the explicit attention mask to Metal |
| G6 | E4 | Adopt the reserve/assign allocator split in Metal's pool |
| G7 | METAL-OBJ | Re-run the Metal gap/parity measurements after G2–G3, since both change a kernel path |

Entry condition: a machine that can build and run the Metal backend. Exit
condition: `SUPPORT-MATRIX.md`'s per-backend op column matches `supports_op` on
all three backends, with A1's matrix green.

## 11. Sequencing

```
Phase A  ├─ A0 ─ A1 ─┬─ A3 ─ A4 ─ A5 ─ A6 ─ A7 ─ A8 ──────────►  (A8 CUDA half)
         └─ A2 ──────┘
Phase B  ├─ B1 ─ B2 ─ B3                          (starts once A0/A1 exist)
Phase C  ├─ C1 ─ C2 ────────────────► C3 ─ C4 ─ C5        (C3 needs D1)
Phase D  ├────────── D1 ─ D2 ─ D3 ──────────────►         (D unlocks MoE/MLA)
Phase E  ├──────────────────── E1 ─ E2 ─ E3 ─ E4 ─ E5
Phase F  └─ F2 F3 F4 F5 F6 F7 (parallel)        F1 = needs x86
Phase G  └────────────────────────────────────────────►  (needs a Mac)
```

**Critical path:** A0 → A1 → C1 → C2 → E1 → E2.
**Deliberate exception to the roadmap's ordering:** A1/A2 run *before* the
hazard-removal tickets, because they are the instrument that proves those
tickets and everything after them.

## 12. What "done" means per phase

| Phase | Done when |
|---|---|
| A | `cargo test` green on Linux/CPU; A1's matrix green (or every red row explained); A0's CUDA verdict recorded; **each hazard ticket has a test that fails before and passes after**. |
| B | A multi-turn conversation prefills only the new turns; the contamination test passes; numbers recorded interleaved. |
| C | Cell store lands bitwise; shift is a documented tolerance class; quantized KV behind its gate; session save/restore round-trips. |
| D | A view is provably zero-copy; one hand-written fusion is replaced by a composition, bitwise. |
| E | Two sequences can be batched without cross-attention; `--n-slots 4` beats serial; `n_batch` chunks prefill; an over-VRAM model runs with layer offload. |
| G | Three backends agree with the op matrix and the support table. |
