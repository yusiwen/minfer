# minfer Architecture Execution Plan

**Status:** Phase A **complete** (2026-09-16); Phase B not started.
**Companion to:** `docs/ARCHITECTURE-ROADMAP.md` (what is missing, why, and how it
is ranked). This document is the *how*: phase-by-phase tickets with
deliverables, acceptance criteria and dependencies.
**Baseline:** `HEAD = f32daa7` (2026-09-16); Phase A landed on
`architecture-phase-a` (PR #1).

## 0. Decisions already taken

| Decision | Consequence for this plan |
|---|---|
| **Metal is out of scope this round.** | No ticket here edits `src/graph/metal_backend.rs`, `src/metal.rs` or `src/metal.metal`. Every phase records what it defers into **Phase G (Metal alignment)**. |
| **Dead reuse-identity fields: option (a).** | Delete `CParams.n_batch`; keep `GraphParams.n_seqs` marked *reserved for item 3*. A7 is unblocked — rationale in §8. |
| **Phase A (A0–A8) is complete** (2026-09-16, PR #1); **Phase B (B1–B3) is complete** (2026-09-16). | Phase C is the next tranche; Phases D–G remain planned. |

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
| A1 | 23 | Op × dtype × backend correctness matrix | M | ✅ done — found + fixed an op defect |
| A2 | 24 | CI: test on Linux/CPU, build on CUDA, keep macOS build | S | ✅ done |
| A3 | 5 | KV bounds guard + `ensure_kv` size check | S | ✅ done |
| A4 | 6 | Server worker panic isolation | S | ✅ done |
| A5 | 27 | Re-key cross-backend staging by `(node, dst_backend)` | S | ✅ done |
| A6 | 28 | Remove CPU per-op allocations | S | ✅ measured — refuted, reverted |
| A7 | 26 | Dead identity fields | S | ✅ done |
| A8 | 13 | Guard symmetry (docs half + CUDA `FusedQkvNorm`) | S | ✅ done (docs route) |

### A0 — CUDA access spike — **DONE (2026-09-16): unavailable**
- **Verdict:** `cargo build --release --features cuda` succeeds (1m23s, targets
  `sm_75…sm_121`, PTX `compute_121`), but at runtime
  `cudaGetDeviceCount` returns err 304 and the engine logs
  `CUDA: no CUDA devices found (cudaGetDeviceCount err 304, count 0)` then
  `CUDA: not available, using CPU fallback`. The CPU path is unaffected
  (Qwen3-0.6B Q8_0: 120 tok/s prefill, 64.7 tok/s decode).
- **Consequence:** the CUDA half of every later ticket is compile-verified only.

### A1 — Op × dtype × backend matrix  · item 23 · M — **DONE**
- **Files:** new `src/graph/op_matrix.rs`, registered as `#[cfg(test)] mod
  op_matrix;`. It needs the crate's internals (the allocator, the backends), so
  it lives in the module tree rather than under `tests/` — the crate is a
  binary, and the existing `tests/*.rs` files get at the code with `#[path]`
  includes, which this would have made worse.
- **Three tests:**
  1. `matrix_cases_match_their_reference` — 17 cases (Add, Mul, Scale, Silu,
     SwiGLU, Softmax, RmsNorm, QkNorm, MatMul, GetRows, View/Reshape/Permute,
     RoPE, Attn, KvcacheStore/Load) run on **every backend that claims the op**,
     each compared against an analytic reference written in the test — never
     against another backend. Unavailable backends report
     `SKIP (reason)`, never `PASS`: on this box that is 17 CPU cells + 34 skips.
  2. `support_table_matches_support_matrix_doc` — `supports_op` for 23 op rows
     against the table published in `SUPPORT-MATRIX.md`, so the A8 doc and the
     code cannot drift. The CPU column is checked here; the Metal/CUDA columns
     check themselves wherever they are compiled in.
  3. `every_op_has_a_matrix_decision` — every `Op` variant is either covered or
     excused in `EXCUSED`. `op_label` matches with **no wildcard arm**, so adding
     an `Op` variant is a compile error until the matrix is updated.
- **Found a real defect, fixed here:** the CPU `Op::Softmax` arm called
  `vec_soft_max_f32` (which writes `exp(x - max)` and *returns* the sum) and
  discarded the return, so the op produced **unnormalised** output. Nothing
  caught it because no architecture emits a standalone `Softmax` node. The arm
  now scales by `1/sum`; verified by neutering that line (the Softmax cell goes
  red) and restoring it.
- **Acceptance met:** the matrix runs under `cargo test`; the A8 asymmetry table
  is now machine-checked; one real defect found and fixed; `cargo test --release`
  162 passed / 0 failed.
- **Left for a GPU-verifiable run:** the Metal and CUDA columns, and the four
  GPU-only fused ops (`FusedQKV`/`FusedFFN`/`FusedQkvNorm`/`QkvBiasRopeStore`)
  which are excused on a CPU-only box.

### A2 — CI  · item 24 · S — **DONE (verified in CI)**
- **Files:** `.github/workflows/ci.yml`.
- **Deliverable:** three jobs — `test-linux-cpu` (`cargo test --release`, the
  runtime net), `build-linux-cuda` (`cargo build --release --features cuda` in
  `nvidia/cuda:12.8.0-devel-ubuntu22.04`; no GPU on a hosted runner, so
  compile-only by necessity), and the unchanged macOS/Metal `build`.
- **Acceptance:** the workflow parses and declares all three jobs; a broken
  commit now fails `test-linux-cpu` (unit tests run there, including the
  allocator/scheduler tests added in A3/A5).
- **Verified outcome** (PR #1, run [35080755969](https://github.com/yusiwen/minfer/actions/runs/35080755969)):

  | job | result | time |
  |---|---|---|
  | `test-linux-cpu` | ✅ | 1m09s |
  | `build-macos` | ✅ | 1m26s |
  | `build-linux-cuda` | ✅ (incl. `cargo test --features cuda --no-run`) | 4m36s |

- **The x86_64 question is answered: the Linux/CPU suite passes.** `running 163
  tests → 160 passed; 0 failed; 3 ignored`, plus `3 passed; 6 ignored` for the
  integration file. That is 2 fewer than the aarch64 dev box, and the difference
  is exactly the `quants::neon_correctness` module, gated
  `#[cfg(all(test, target_arch = "aarch64"))]` — no test is silently missing.
- **First CI run failed, and that was the point.** Run 35080525201 came back
  `build-linux-cuda` ❌ at the `dtolnay/rust-toolchain` step:
  `curl: command not found` — the CUDA devel images are minimal, so rustup could
  not bootstrap and the CUDA build never started. Fixed by installing
  `curl ca-certificates build-essential` before the toolchain step (the last
  also supplies nvcc's host compiler). Only the fix commit's push turned all
  three green.

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

### A4 — Worker panic isolation  · item 6 · S — **DONE**
- **Correction to the ticket as written:** the worker did have a
  `catch_unwind` — `guarded_forward` (`chat.rs:438-450`) contains a panic inside
  `forward_graph_cached`. It is just too narrow: the speculative path calls both
  models' forwards **directly** (`spec.rs:385/417/447/471/509`), and the
  tokenizer, sampler, stop-string and streaming paths were unguarded too. One
  panic there unwound `worker_loop`, dropping `job_rx` (every later request is
  rejected) and the already-queued jobs' event senders (an empty 200 instead of
  an error).
- **Files:** `src/server/chat.rs` — `run_job_isolated` + `panic_message`, called
  around the whole per-job body in `worker_loop`; the inner `guarded_forward`
  stays for the better message on the common case.
- **Deliverable:** any panic in a job becomes a 500 on that request's stream,
  is logged, and the worker keeps draining the queue with the slot released.
- **Acceptance:** `isolated_job_turns_a_panic_into_an_error_event`,
  `isolated_job_forwards_a_normal_error`,
  `isolated_job_passes_success_through_silently`; `cargo test --release`
  158 passed / 0 failed.
- **Found while writing the test:** `panic_message(&payload)` on a
  `Box<dyn Any + Send>` downcasts against the *Box* (which is itself `Any`) and
  always reports a non-string payload; `&*payload` is required. Verified against
  a throwaway `rustc` probe.
- **Not fixed (needs a device):** a panic while the CUDA capture window holds
  the process-wide stream mutex poisons it, so every later request would panic
  too — now contained per job, but the server would still fail every request.
  Recorded for the CUDA-verifiable phase.

### A5 — Staging map re-key  · item 27 · S — **DONE**
- **Files:** `src/graph/alloc.rs` (`cross`, `copy_across`, `cross_buffer`, plus
  a new `write_pool` helper that removes the duplicated four-arm write) and
  `src/graph/scheduler.rs` (the consumer-side filter).
- **Deliverable:** `cross` is keyed by `(NodeId, Backend)`. The scheduler's
  `.filter(|cb| cb.backend == split.backend)` is gone — the map itself can no
  longer offer a consumer the other backend's copy, and one node feeding two
  foreign backends now gets one staging buffer each instead of a single entry
  that only one of them could use.
- **Acceptance:** `staging_is_keyed_by_destination_backend` (stages one node for
  CPU, asserts the CUDA consumer gets `None`, then stages CUDA and asserts the
  CPU entry survives); the existing scheduler tests still pass;
  `cargo test --release` 159 passed / 0 failed.
- **Honest limitation:** a *behavioural* test needs a second usable backend,
  which this box does not have (A0: CUDA unavailable; Metal not compiled). The
  test asserts the new keying contract directly through a `#[cfg(test)]` hook
  rather than through a real split boundary. Phase G should re-test it on a Mac.

### A6 — CPU per-op allocations  · item 28 · S — **DONE: measured, refuted, reverted**
- **Files (measured, not changed):** `src/graph/cpu_backend.rs` — the
  per-node `Vec<&[f32]>` of resolved inputs, and the K/V source clone in the KV
  store arm.
- **What was tried:** replace the per-node input `Vec` with a fixed
  `[&[f32]; 4]` array (every op the builders emit has ≤ 4 inputs) plus a heap
  spill for anything larger — i.e. zero allocation per node execution instead of
  one.
- **Measurement** (Qwen2.5-0.5B Q4_0, CPU, 20 threads, `bench -r 3`, three
  interleaved before/after pairs, `minfer-a6-before` vs `minfer-a6-after`):

  | pair | before pp512 | after pp512 | before tg128 | after tg128 |
  |---|---|---|---|---|
  | 1 | 76.08 ± 0.03 | 74.79 ± 0.03 | 40.30 ± 0.09 | 39.60 ± 0.28 |
  | 2 | 76.05 ± 0.01 | 75.33 ± 0.02 | 40.25 ± 0.16 | 39.07 ± 0.40 |
  | 3 | 76.14 ± 0.03 | 75.39 ± 0.05 | 40.00 ± 0.31 | 39.68 ± 0.29 |

  The change is **consistently slower** (−1.2 % prefill, −1.8 % decode; every
  pair separated, and the intra-run error is ±0.05 or less). Reverting restores
  the baseline (76.10 / 76.06 pp512, 40.79 / 40.21 tg128), which confirms the
  cause.
- **Why it does not matter anyway:** the decode loop is weight-streaming bound.
  At ~250 nodes/token the per-node `Vec` was one small allocation each, on the
  order of 0.1 % of the step — below what the harness can resolve. The K/V clone
  was left alone for the same reason: it is ~2 KB per layer and the buffers are
  needed for the borrow structure (the store arm needs disjoint access to four
  pool slots).
- **Outcome:** no code change; the negative result is recorded so nobody
  re-opens it. Item 28 is closed as "not worth doing" rather than done.

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

### A8 — Guard symmetry  · item 13 · S — **DONE (docs route)**
- **Files:** `docs/SUPPORT-MATRIX.md` — a new "Operator Coverage by Backend"
  section.
- **Route taken:** the ticket offered "CUDA gains `FusedQkvNorm` **or** the
  matrix records that it does not have it". A0 made CUDA unverifiable here, so
  writing a new CUDA kernel blind is strictly worse than documenting the
  asymmetry: the table now lists every op against CPU/Metal/CUDA, generated from
  the three `supports_op` implementations, with the four asymmetric rows called
  out and their consequences (Qwen3 decode is fused on Metal and unfused on
  CUDA; `QkvBiasRopeStore` is the mirror case; interleaved RoPE is CPU-only).
- **Acceptance:** the roadmap §4 defect 5 asymmetry is now visible in the
  support matrix; A1's matrix must agree with this table.
- **Defers to G:** the Metal half (`debug_assert!` → `Err`; the weightless
  RMSNorm fallback) and any decision to port `FusedQkvNorm` to CUDA.

## 4. Phase B — persistent server context

*Why second:* it is independent of the KV redesign, and it is the first
user-visible win — a chat client that resends the conversation stops
re-prefilling it every turn.

| ID | Item | Title | Effort |
|---|---|---|---|
| B1 | 4 | Reproduce the doc-97 contamination | S | ✅ done — mechanism refuted, property pinned |
| B2 | 4 | Slot cache retention + prefix check | M | ✅ done |
| B3 | 4 | Measurement + docs | S | ✅ done — ≈11× on turn 2 |

### B1 — Reproduce contamination — **DONE: the documented mechanism does not reproduce**
- **What the docs claimed.** `chat.rs` reset the slot cache on every request
  because "re-prefilling a DIFFERENT prompt over the same regions leaves stale
  rows inside the new attention window" — the message of `39eceaa` (doc 97),
  which fixed a cross-request KV bug and added the reset.
- **What was tested.** `reused_cache_across_prompts_matches_a_fresh_cache`
  (`src/models/qwen2/graph.rs`) runs real prompts through one `GraphCache` and
  compares **bitwise** against a virgin cache, in three orderings:
  1. long prompt → short prompt (the stale-row case);
  2. short → long (the append case B2 wants);
  3. prefill → 3 decode steps → shorter prompt (the server's actual sequence,
     including the generated rows the comment is about).
  All three are bitwise equal.
- **Why.** A prefill writes rows `0..nt` and attention reads
  `[0, max(pos)+1) = [0, nt)` — i.e. only rows this request just wrote. Stale
  rows survive *above* the window, where nothing reads them. The original bug is
  consistent with the state recorded in the OpenAI plan's revision notes: the
  cache was **process-global** at the time, so two slots shared one set of
  regions. That is already fixed by per-slot caches; the reset was belt-and-braces
  whose stated mechanism does not hold on the current tree.
- **Residual uncertainty (why the reset stays until B2).** This box can only run
  the CPU path. The fused GPU stores (`FusedQKV`, `FusedQkvNorm`,
  `QkvBiasRopeStore`) write K/V inside their own kernels and are unverified here
  (A0: CUDA compile-only; Metal not built). The `chat.rs` comment now records the
  finding and this caveat instead of the unverified mechanism.
- **Consequence for B2.** Reuse is safe *by construction* if it is gated on an
  exact token-prefix match: every row read is then a row whose contents were
  verified to be the same tokens. That is what B2 implements.

### B2 — Retention + prefix reuse — **DONE**
- **Files:** `src/server/slot.rs` (`Slot.cached_tokens`), `src/server/chat.rs`
  (`common_prefix_len`, `prefill_span`, the suffix prefill, the per-token
  record, `worker_loop` no longer resets the cache).
- **Deliverable:** the slot keeps its `GraphCache` and a record of the token
  sequence its KV rows hold; `generate_seq` reuses the cache only when the new
  prompt starts with exactly that sequence (else it prefills from position 0),
  and always feeds at least the last token because its logits seed the sampler.
- **Why it is safe by construction:** the reuse gate *is* the verification —
  every row attention reads is a row `common_prefix_len` just proved to hold the
  same token. It is also bitwise: the CPU attention is now `nt`-invariant
  (item 14), so feeding a suffix at its own positions reproduces a single-shot
  prefill exactly.
- **Deliberate limits:** the speculative path neither reuses nor records (a
  verify round writes rows past the committed tokens); a panicking job clears
  the record; `MINFER_NO_PREFIX_REUSE=1` restores the pre-B2 behaviour for A/B.
- **Tests:** `common_prefix_len_finds_the_exact_match`,
  `prefill_span_always_feeds_the_last_token` (pure), and
  `prefix_reuse_matches_a_full_prefill` (real model, bitwise).
- **Not covered here:** the fused GPU stores are unverified on this box, so the
  gate stays conservative; Phase G re-tests on Metal.

### B3 — Measurement — **DONE (interleaved A/B, same binary)**
- **Setup:** `serve --n-ctx 1024 --n-slots 1` on Qwen2.5-0.5B Q4_0, one long
  system prompt (203 tokens) plus a second turn that appends an assistant reply
  and a new question (219 tokens), `max_tokens=1` so the wall time is
  essentially time-to-first-token. Four alternating runs; `after` vs `before`
  differ only by `MINFER_NO_PREFIX_REUSE=1`.

  | mode | turn | prompt tok | fed | reused | wall |
  |---|---|---|---|---|---|
  | after | 1 | 203 | 203 | 0 | 2.66 s |
  | after | 2 | 219 | **16** | **203** | **0.24 s** |
  | before | 1 | 203 | 203 | 0 | 2.60 s |
  | before | 2 | 219 | 219 | 0 | 2.71 s |
  | after | 1 | 203 | 203 | 0 | 2.66 s |
  | after | 2 | 219 | **16** | **203** | **0.26 s** |
  | before | 1 | 203 | 203 | 0 | 2.88 s (turn 2) |

- **Result:** turn 2's prefill drops from 219 to 16 tokens and its
  time-to-first-token from ~2.7–2.9 s to ~0.24–0.26 s — **≈ 11×** — while turn 1
  (cold slot) is unchanged, as it must be. The per-request line
  `[server] prefill fed N/M prompt tokens (R reused)` is what the numbers come
  from, and it stays in the server so the behaviour is observable in production.

## 5. Phase C — KV cell store (item 1)

Five sub-steps, each keeping the tree green. C3 depends on Phase D.

| ID | Title | Effort |
|---|---|---|
| C1 | `KvCache` with cells, single implicit sequence (behaviour-preserving) | L |
| C2 | `seq_rm` / `seq_add`: prefix truncation + context shift | L |
| C3 | Defragmentation (cell copy) — **needs D1** | M |
| C4 | Quantized KV (item 21) | M |
| C5 | State save/restore for session persistence | M |

### C1 — Cell store, one sequence — **DONE**
- **Landed:** `src/graph/kvcache.rs` (`KvCache`, `KvLayer`, `SEQ_MAIN`/`FREE`,
  `cells_for`, `is_identity`, and the ownership/write bookkeeping C2 will use);
  the allocator's `kv` map is now the store, so `ensure_kv` validates through it
  (it gained the `n_ctx` argument) and `kv_pair`/`copy_kv_to_cpu` read it.
- **The gate has teeth:** `BackendScheduler::execute` refuses any graph whose
  mapping is no longer the identity, because no backend consumes the resolved
  cell array yet — so a half-ported C2 fails loudly instead of writing the wrong
  row. `a_non_identity_kv_mapping_is_refused` proves it fires.
- **The resolver has a live consumer today:** A3's position guard
  (`fill_input_i32`) now resolves through `cells_for` instead of re-deriving the
  bound from a node's shape, so when C2 changes the mapping the check keeps
  meaning the same thing without being touched.
- **Verified bitwise, as the ticket requires:** `cargo test --release` 172
  passed / 0 failed — including the real-model bitwise tests
  (`reused_cache_across_prompts_matches_a_fresh_cache`,
  `prefix_reuse_matches_a_full_prefill`, `graph_logits_match_forward_real_model`)
  — and a pre-C1 vs post-C1 binary A/B on Qwen2.5-0.5B Q4_0 greedy (`-n 24`)
  produced **byte-identical generated text**; only the timing lines differ.
- **Deliverable (as specified):** per-layer cell arenas with an owner set; the
  KV store node resolves `position → cell` on the host and passes an index
  array to the kernel; attention still derives its bound from `positions`, so
  behaviour is unchanged.
- **Delivered in this pass:** the arenas, the owner set and the host-side
  resolver. The index array *reaching the kernel* is deliberately **not** done —
  it is only needed once the mapping stops being the identity, and the
  scheduler gate refuses to execute in that state, so no backend can silently
  index the wrong row in the meantime. Handing the array across the `Backend`
  trait is C2's first step.
- **Acceptance (as specified):** bitwise-identical greedy output vs the
  pre-change binary on the full smoke set (0.6B Q8_0, 7B Q4_K_M, 14B Q4_K_M) at
  several context lengths; the graph topology is unchanged (no new
  topology-affecting params).
- **Acceptance met for:** 0.5B Q4_0 byte-identical end to end, plus the in-tree
  bitwise model tests. **Not run:** the 0.6B/7B/14B smoke rows — those are
  minutes-long CPU jobs and C1 does not touch a kernel, so they are deferred to
  the C2 landing rather than claimed here.
- **Defers to G:** nothing — the Metal backend keeps the old regions until G.

**C1 design (written before the code, 2026-09-16).** The shape below is what C2
needs, so C1 builds it rather than a type with no consumer:

1. `KvCache` (new `src/graph/kvcache.rs`) owns, per layer, the two persistent
   regions plus `owner: Vec<SeqId>` — one entry per cell, `FREE` for a cell no
   sequence holds. One implicit sequence for now, so every cell of a written row
   is owned by it.
2. `KvCache::cells_for(positions: &[usize]) -> Result<Vec<u32>, String>` — the
   host-side resolver. Today it is the identity (`cell == pos`) and returns
   `Err` for a position outside the arena; C2 changes exactly this function to
   consult `owner` and the layer's window start.
3. `KvCache::is_identity() -> bool` — true until C2 introduces a hole or a
   window. The scheduler consults it: while true the existing `positions` input
   reaches the backend exactly as today (so no kernel changes), and once false
   the resolved cell array must be passed instead. **A backend that has not been
   ported returns `Err` instead of indexing the wrong row** (standing rule 2).
   On this box that means CPU first: CUDA is compile-verified only (A0) and
   Metal stays untouched (G), so C2 must either keep the mapping identity for
   them or refuse to run there.
4. The index array travels as a graph **input** (`kv_cells`, I32), filled by the
   allocator from `positions` — positions stay data, so the topology and the
   params-only reuse identity are untouched.
5. Tests: resolver edge cases (out-of-range → `Err`, identity while there are no
   holes), `is_identity` flipping exactly when C2 introduces one, and the
   existing model tests (B1, B2's prefix reuse, `graph_logits_match_forward`)
   staying bitwise — that is the refactor's acceptance gate.

The first thing to change is therefore the *resolution*, not the kernels: the
Backend trait gains the resolved cell slice alongside `kv_pair`, and each
backend uses it for row indexing while still using `positions` for the causal
bound and RoPE.

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
| A | **Complete 2026-09-16.** `cargo test` green on Linux/CPU (aarch64 locally, x86_64 in CI); A1's matrix green (or every red row explained); A0's CUDA verdict recorded; **each hazard ticket has a test that fails before and passes after**; A6 is closed by measurement instead — a refuted hypothesis with numbers is a result, not a gap. |
| B | **Complete 2026-09-16.** A multi-turn conversation prefills only the new turns (219 → 16 tokens, ≈11× TTFT); the contamination property is pinned by a bitwise test; numbers recorded interleaved with the same binary. |
| C | Cell store lands bitwise; shift is a documented tolerance class; quantized KV behind its gate; session save/restore round-trips. |
| D | A view is provably zero-copy; one hand-written fusion is replaced by a composition, bitwise. |
| E | Two sequences can be batched without cross-attention; `--n-slots 4` beats serial; `n_batch` chunks prefill; an over-VRAM model runs with layer offload. |
| G | Three backends agree with the op matrix and the support table. |

## 13. Phase A outcome (2026-09-16)

**Branch:** `architecture-phase-a` (PR #1). All nine tickets closed.

| Check | Result |
|---|---|
| Unit suite (aarch64, this box) | `162 passed / 0 failed / 3 ignored` |
| Unit suite (x86_64, CI runner) | `running 163 → 160 passed / 0 failed / 3 ignored`; the 2-test delta is `quants::neon_correctness`, gated `cfg(all(test, target_arch = "aarch64"))` |
| Op matrix | 17 cases × 3 backends (17 CPU cells, 34 explicit skips), 23 support rows, exhaustive `Op` enumeration |
| CI | three jobs green: 1m09s / 1m26s / 4m36s |
| `rustfmt --check`, pre-commit | clean; the hook ran on every commit |
| `mdbook build` | passes |

Defects found and fixed during the phase — none of them were on the ticket list,
which is the point of A1 and A2:

- **`Op::Softmax` returned unnormalised values** on the CPU backend (found by
  A1's matrix; pinned by a case).
- **Four CUDA test call sites** broken by A5's signature change — invisible to
  `cargo build` because it does not compile `#[cfg(test)]`; the CUDA CI job now
  runs `cargo test --features cuda --no-run`.
- **`panic_message(&payload)`** downcast against the `Box` (itself `Any`)
  instead of the payload, so every panic reported a non-string payload.
- **CI could not bootstrap rustup** in the CUDA container (no `curl`).

Deliberately not done, and where it is recorded:

- **GPU columns of the matrix** and the four GPU-only fused ops → a
  GPU-verifiable run (A0: CUDA is compile-only here).
- **Metal** — untouched by decision; its debt is Phase G.
- **A6** — no code change; the measurement is the deliverable.

Open for the maintainer: the three CI jobs carry a pre-existing
`Node.js 20 is deprecated` annotation for `actions/checkout@v4`; bumping to
`v5` silences it.
