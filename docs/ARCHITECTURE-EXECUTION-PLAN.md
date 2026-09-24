# minfer Architecture Execution Plan

**Status:** Phase A **complete** (9/9, 2026-09-16); Phase B **complete** (3/3,
2026-09-16); Phase C **complete (8/8)** — C1, C2, **C3**, **C4 (quantized Q8_0 cache,
CPU)**, **C5 (session save/restore)**, **C6 (logical positions)**, **C7 (+C7b)** and
**C8 (cross-sequence cell sharing)** are done. C6 merged 2026-09-20 as `001b8cc`;
**C7 landed 2026-09-20** (the partition is elastic, and growth moves runs in **both**
directions, so a busy neighbour above the slot no longer blocks it); **C8** split into
**C8a** (shared prefill, duplicated rows: no IR change) and **C8b** (paged sharing: a
block map and a gather in every attention kernel — S1a/S1b/S2/S3/S4/S5 landed, closed on
CPU and CUDA, Metal's share path at G5); **C4** and **C5** landed 2026-09-22 (C4's fused
dots and the CUDA/Metal kernels are [#87](https://github.com/yusiwen/minfer/issues/87);
the CLI/server surfaces C5 enables are [#89](https://github.com/yusiwen/minfer/issues/89)).
Phase D **3/3** (**D1 done**: views, multi-output via `split_parts`, D2, D3); Phase E
**7/7** (E1, E1b, E2, **E3**, **E4**, **E5**, E6 all done); Phase F **0/8** (F1 needs x86); Phase G
**scheduled** — after the CUDA
KV path, not before it (device claims need a Mac; CI's `build-macos` is the compile
check). **Phase C is complete (8/8); next: E4/E5 (allocator, then layer offload) and the Metal
round G1–G3/G5 (on a Mac).** The order is
deliberate: the Metal KV port (G5) comes **after** the CUDA arena stops changing shape
(C7, C7b, C8), so those semantics are written into Metal once. Per-ticket evidence is in
each phase's record and in the §14 open-risks table.
**Companion to:** `docs/ARCHITECTURE-ROADMAP.md` (what is missing, why, and how it
is ranked). This document is the *how*: phase-by-phase tickets with
deliverables, acceptance criteria and dependencies.
**Baseline:** `HEAD = f32daa7` (2026-09-16); Phase A landed on
`architecture-phase-a` (PR #1). This status was refreshed against `master =
75f14b7` (2026-09-22, the C5 merge); it is refreshed with every PR that lands a ticket.

**Issue links.** Work tracked on GitHub carries its issue link in its table row
(prose sections carry it in the heading), and the record written when that work lands
keeps the link. Tickets that predate the tracker — Phase A and B, C1–C3, C6/C7/C7b,
C8a, D1–D3, E1/E1b/E2/E6 — have no issue, and their record here is the only one; the
tickets below that *do* have one are exactly the open list in the
[tracker](https://github.com/yusiwen/minfer/issues). Doc debt and test health are filed
the same way: [#62](https://github.com/yusiwen/minfer/issues/62) (`docs/USAGE.md`
staleness). The docs build was [#63](https://github.com/yusiwen/minfer/issues/63) and is
**closed**: CI job `check-docs` builds the book with the same toolchain Docs deploys with
and runs `scripts/check_docs_links.py`, so a renamed file now fails on the PR that renames
it instead of rotting silently (it found four dead links the day it landed). Test health was [#82](https://github.com/yusiwen/minfer/issues/82) — the three
`#[ignore]`d real-model tests that failed on `master` — and it is **closed**: two were
writing into a directory nothing created, and the third asserted a *model behaviour* (the
greedy 0.5B stopping on EOG within 16 tokens) instead of the engine's rule that ties
`need_insert_eot` to the stream.

## 0. Decisions already taken

| Decision | Consequence for this plan |
|---|---|
| **Metal is out of scope this round.** | No ticket here edits `src/graph/metal_backend.rs`, `src/metal.rs` or `src/metal.metal`. Every phase records what it defers into **Phase G (Metal alignment)**. |
| **Dead reuse-identity fields: option (a), then (c).** | A7 deleted `CParams.n_batch`; E2 then **deleted** `GraphParams.n_seqs` too — item 3 landed and showed the sequence count is data, not topology (A7 closed, rationale in §8). |
| **Phase A (A0–A8) is complete** (2026-09-16, PR #1); **Phase B (B1–B3) is complete** (2026-09-16, PR #1); **Phase C's C1 and C2 are complete** (2026-09-16, PR #2); **E1 and its CUDA half (E1b) are complete** (2026-09-17, PR #3) — E1b was compile-verified and SASS-checked then, and is **device-verified** since 2026-09-18 (see its record); **E2 landed and is closed** (2026-09-17, PR #4; **re-measured on the GPU 2026-09-18**): mechanism in, A7 closed by deleting `n_seqs`, acceptance **refuted on CPU (0.49x) and met on GPU (1.9x)** — the sign of the effect is a property of the device. **The CUDA device is available from 2026-09-18** (A0 superseded); E1b's windowed attention is device-verified and its causal path is timing-neutral, and the A1 matrix's CUDA column now runs on hardware. | The next work is **C3 + D1 increment 3** (the explicit cell-copy op C3 needs, and multi-output nodes — also the MoE/MLA prerequisite), then **C4/C5**; **E6 settled the batching default** (device-aware: batches iff the model runs on CUDA, off on CPU/Metal, `MINFER_BATCH=0/1` to force either way), and **§14 row 0 closed the GPU-batching correctness blocker** behind it (the f16 windowed FA prefill mask, fixed 2026-09-19); a CPU `nt>1` decode kernel (F1 family) remains the only CPU route to the throughput claim; Phases D–G remain planned. |

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
| CUDA (sm_121) | ✅ | ✅ **device since 2026-09-18** | GB10 (121.6 GiB, driver 580.178.04, CUDA 13.0). Device-gated tests are still **local-only**: CI has no GPU, so its CUDA job only compiles the harness (F-campaign note in §14 row 1) |
| Metal | ❌ | ❌ | macOS-only code, not compilable here → Phase G |
| x86 AVX2 / AVX-512 | ❌ | ❌ | Item 11 needs an x86 box or CI |

**A0 verdict (2026-09-16) — superseded (2026-09-18).** It read "CUDA is
compile-only in this environment", and for two days every CUDA ticket's acceptance
was "compiles + reviewer-inspected". The reading was wrong: those probes ran under
an agent file sandbox whose Landlock rules denied `open()` on `/dev/nvidia*` even
though the nodes exist and are world-writable, so `cuInit` failed with err 304 for
a reason unrelated to the driver. The device has been available since 2026-09-18,
and CUDA work is now **measured** on it — E1b's windowed attention, E2's 1.9x, and
the f16 windowed-prefill fix recorded in §14 row 0 all carry device evidence.

The design consequence worth keeping is about *coverage*, not capability: CI has no
GPU (§14 row 1), so a device-gated assertion is a local, manual run. Where a guard
can live in the **backend-agnostic** layer (the allocator) instead of
`cuda_backend.rs`, put it there so CI still proves it — that reasoning is
independent of whether a device happens to be present (applied in A3, and again by
C3 below).

## 3. Phase A — instrument, then hazard removal

*Why first:* the roadmap's §5 calls items 5, 6, 13, 26–28 "cheap and
independent". This plan pulls **item 23 (op matrix) and item 24 (CI) to the
front of that batch** — they are the instrument that keeps Phases B–E honest,
and one of them (the op/dtype/backend matrix) would have caught several of the
roadmap §4 defects automatically.

| ID | Item | Title | Effort | Status |
|---|---|---|---|---|
| A0 | — | CUDA access spike on this box | S | ✅ done — but the verdict is **superseded (2026-09-18)**: the device is available; "unavailable" was an agent-sandbox artefact (§2) |
| A1 | 23 | Op × dtype × backend correctness matrix | M | ✅ done — found + fixed an op defect |
| A2 | 24 | CI: test on Linux/CPU, build on CUDA, keep macOS build | S | ✅ done |
| A3 | 5 | KV bounds guard + `ensure_kv` size check | S | ✅ done |
| A4 | 6 | Server worker panic isolation | S | ✅ done |
| A5 | 27 | Re-key cross-backend staging by `(node, dst_backend)` | S | ✅ done |
| A6 | 28 | Remove CPU per-op allocations | S | ✅ measured — refuted, reverted |
| A7 | 26 | Dead identity fields | S | ✅ done |
| A8 | 13 | Guard symmetry (docs half + CUDA `FusedQkvNorm`) | S | ✅ done (docs route) |

### A0 — CUDA access spike — **SUPERSEDED (2026-09-18): the device is available**
- **Original verdict (2026-09-16):** `cargo build --release --features cuda`
  succeeds (1m23s, targets `sm_75…sm_121`, PTX `compute_121`), but at runtime
  `cudaGetDeviceCount` returns err 304 and the engine logs
  `CUDA: no CUDA devices found (cudaGetDeviceCount err 304, count 0)` then
  `CUDA: not available, using CPU fallback`. The CPU path was unaffected
  (Qwen3-0.6B Q8_0: 120 tok/s prefill, 64.7 tok/s decode).
- **Correction (2026-09-18).** The *agent's execution sandbox* is enough to
  produce every symptom A0 recorded, on a device that works. Under the harness's
  default file sandbox (Landlock)
  every `open("/dev/nvidia*")` returns `EACCES` even though the nodes are
  `crw-rw-rw-`, so `cuInit` fails with 304 and NVML prints
  `Failed to initialize NVML: Unknown Error` — **the exact signature A0 recorded**.
  With the sandbox widened (2026-09-18, after a reboot that also cleared a
  driver-upgrade state the maintainer had flagged): `NVIDIA GB10`, `sm_121`,
  121.6 GiB, driver 580.178.04, CUDA 13.0, `cuInit` → `CUDA_SUCCESS`.
  Every A0-era probe was therefore run inside a sandbox that cannot reach a GPU
  **even when the GPU is healthy**, so those probes could not establish anything
  about the driver's state: "no device" was unsupported rather than merely
  pessimistic. (The maintainer recalls a driver upgrade without a reboot at the
  time; that account and the sandbox are both consistent with the record, and
  only the sandbox is reproducible today — which is why the correction is to
  *re-run* the verification, not to assume it always would have passed.) What is
  certain now: the CUDA half of every ticket between A0 and this session was
  compile-verified only, and in this session it is **device-verified** (see the
  E1b record's device section).
- **Consequence for the plan:** tickets that were closed "compile-verified only"
  because of A0 are re-opened *as verification*, not as code: E1b's kernels, the
  A1 matrix's CUDA column, and the CUDA-side test suite all get their first
  execution on hardware below.

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
- **CUDA column executed on hardware (2026-09-18).** With the sandbox corrected
  (A0) the CUDA column finally ran, and every cell it claimed passed: Add, Mul,
  Silu, SwiGLU, RmsNorm, QkNorm, MatMul, GetRows, View, Reshape, Permute, RoPE,
  Attn, KvcacheStore, KvcacheLoad (Scale/Softmax stay `SKIP (Cuda does not claim
  …)`, which the support table asserts). Running it exposed four harness defects,
  all fixed here — the kind that only a device can show:
  1. the CUDA column silently depended on **test order**: `CudaState::get()` is
     `None` until something calls `init()`, which `main` does at startup and the
     model tests do through the loader, so a *filtered* run reported
     `SKIP (no CUDA device)` while a full-suite run used the device. The harness
     now initialises the state itself (in `backend_claims` and `run_on`).
  2. the synthetic weights were registered only on CPU (`alloc.register_weight`
     delegates to the CPU pool; in the product the *loader* registers each weight
     into `CudaState`), so every weight op failed with "weight not registered on
     CUDA". The harness now registers them into `CudaState` too.
  3. `Attn` (hd 2) and `QkNorm` (hd 2) used a head dim the CUDA kernels reject
     (`must be a nonzero multiple of 4`) — cells that could only ever pass on
     CPU. Both fixtures now use hd 4.
  4. the `KvcacheStore`/`Load` case wrote two of its four region rows and expected
     the rest to be **zero**: true for a fresh CPU pool, false for device memory.
     It now writes every cell, so the expectation is defined on every backend.
- **Acceptance met for the CUDA column too (2026-09-18):** on GB10 (sm_121) the
  matrix is 17 CPU cells + 17 CUDA cells with no `FAIL` in either column.
- **Recorded limitation (2026-09-18):** `support_table_matches_support_matrix_doc`
  compares the code against a **mirror table hard-coded in Rust** and only *names*
  `SUPPORT-MATRIX.md` in its failure message — it does not parse the markdown. A
  stale row label therefore survives a green suite (it did: the `Attn` row still
  said `multi_seq` after the field was renamed to `explicit_span`). Parsing the
  published table would make the drift impossible rather than merely visible; it
  is a hardening item for A1/A8, not a defect.

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

### A7 — Dead identity fields  · item 26 · S — **DONE, closed in E2**
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
- **E2 closure (the second half of the ticket).** Keeping `n_seqs` *reserved*
  was the right call only until item 3 landed. It did — and the field turned out
  not merely unread but **redundant**: the only topology decision a
  multi-sequence batch can force is the attention instantiation, and
  `CParams.explicit_span`, derived from the KV reservations (the authority on
  where each window starts), already carries it. E2 therefore **deleted**
  `GraphParams.n_seqs`, satisfying the acceptance by the "or deleted" branch.
  Measured justification (`sequence_count_is_data_not_topology`,
  `models/qwen2/graph.rs`): with the field present, a 2-sequence batch and a
  1-sequence batch with the same `n_tokens`/`n_out`/`gtype`/`explicit_span`
  rebuilt the graph (uid 3 → 4); without it the same graph is reused **and** its
  logits are bitwise-identical to a fresh single-sequence forward. The test is
  permanent, so the claim cannot silently regress.

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
| C2 | `seq_rm` / `seq_add`: prefix truncation + context shift — **DONE** | L |
| C3 | Defragmentation (cell copy) — **needs D1** | M |
| C4 | Quantized KV (item 21) · [#42](https://github.com/yusiwen/minfer/issues/42) | M |
| C5 | State save/restore for session persistence · [#43](https://github.com/yusiwen/minfer/issues/43) | M |

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
  trait was planned as C2's first step; C2 landed without it because it removes
  rows *physically* and therefore never leaves the identity mapping (see the C2
  record below) — so the hand-off stays available, not urgent.
- **Acceptance (as specified):** bitwise-identical greedy output vs the
  pre-change binary on the full smoke set (0.6B Q8_0, 7B Q4_K_M, 14B Q4_K_M) at
  several context lengths; the graph topology is unchanged (no new
  topology-affecting params).
- **Acceptance met for:** 0.5B Q4_0 byte-identical end to end, plus the in-tree
  bitwise model tests. **Deferred at the time:** the 0.6B/7B/14B smoke rows —
  minutes-long CPU jobs and C1 does not touch a kernel, so they were left to the
  C2 landing rather than claimed here. **Discharged at C2** (2026-09-16): the
  pre-C2 (C1) binary and the C2 binary generate **byte-identical text** greedily
  (`-n 16`, `-t 8`, single-shot) on Qwen3-0.6B Q8_0, Qwen2.5-7B Q4_K_M and
  Qwen2.5-14B Q4_K_M; only the timing lines differ.
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
   On this box that means CPU first: CUDA was compile-verified only at the time
   (A0 — **superseded 2026-09-18**, the device is available) and Metal stays
   untouched (G), so C2 must either keep the mapping identity for them or refuse
   to run there.
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

### C2 — Removal and shift — **DONE**
- **Deliverable:** `seq_rm` (drop a range) and `seq_add` (shift positions),
  surfaced as graph-level operations; `conversation.rs` overflow switches
  from "drop the oldest turns + full re-render" to a context shift.
- **Acceptance:** the conversation overflow test passes; a turn that overflows
  no longer re-prefills the whole conversation; the shift path is a **named
  tolerance class** with the reason recorded (positions change ⇒ RoPE inputs
  change ⇒ not bitwise).
- **Defers to G:** n/a.

#### C2 record (2026-09-16)

**What landed.** `GraphAllocator::kv_rm(start, len, &KvRope)` removes KV rows
`[start, start + len)` from every layer and re-bases the rows after them by
`-len`, re-roping their K; `kv_shift(drop, rope)` is the `start == 0` case
(`KvCache::after_rm` / `after_shift` do the bookkeeping). `rope_shift_kv` now
takes a **signed** delta so a rotation can be undone (the tests use that).

The removal is **physical**, which is the whole point: `cell == pos` survives, so
`is_identity()` stays true, C1's scheduler gate stays satisfied, and no backend
gains a kernel. The work is a host-side memmove plus one re-rope pass per layer,
both through the existing `copy_kv_to_cpu` / `write_host` pair — which is also
why CUDA/Metal need no change here (`copy_kv_to_cpu` has no Metal arm, so a
Metal session falls back loudly rather than shifting wrongly; the port is G5).

`Engine::kv_rm(start, len)` is the conversation-facing hook (`Err` by default,
so a mock or a non-shiftable backend is handled explicitly). `Conversation`
plans the drop on a copy of the message list, resolves the dropped turn's token
span from the chat template, and **verifies both boundaries against the KV
stream** before touching anything; an unverifiable boundary or an engine that
refuses falls back to the exact drop-and-re-render path, with the reason logged.
`MINFER_NO_CONTEXT_SHIFT=1` forces that path.

**The retained rows keep the context they were computed with.** This is the
ticket's named tolerance class, and its cause is not rounding. Re-roping is
exact to the rope tolerance (measured `max|Δ| = 2.38e-7`, relative `1.14e-7`, by
`rope_shift_matches_roping_at_the_new_position`), but a shifted row's *value* is
whatever it was when it was written — it attended to the tokens that were later
dropped. Measured on Qwen2.5-0.5B Q4_0 with the `a`/`b`/`c` probe (drop `a`,
keep `b`, continue with `c`):

| Layer | 0 | 1 | 5 | 12 | 23 |
|---|---|---|---|---|---|
| retained-K `max|Δ|` vs a fresh `b` prefill | 1.5e-5 | 2.55 | 7.80 | 8.26 | 3.81 |

Last-token logits after continuing with `c`: `max|Δ| = 16.75` vs a fresh `b+c`
prefill, and the argmax changes. **This is inherent**: recomputing the retained
rows is exactly the re-prefill the shift exists to avoid. llama.cpp's context
shift (`llama_kv_cache_seq_rm` + `seq_add` + `llama_kv_cache_update`) has the
same property. What C2 guarantees instead is exactness of the *mechanism*, and
that is what the tests assert bitwise:

- removing a **tail** range leaves the retained head byte-identical, so
  continuing from it is **bitwise** equal to a fresh prefill of that head
  (`kv_rm_is_exact_and_the_window_shift_is_a_named_tolerance_class`);
- removing a **middle** range copies V byte-for-byte, leaves `[0, start)`
  untouched, and moves K by exactly the re-rope (undone to `< 1e-4` per layer);
- removing everything written empties the arena, `len == 0` is a no-op, and
  removing past the end is an `Err`, not a silent truncation.

**Model-path A/B (discharging C1's deferral).** The pre-C2 binary (HEAD at C1)
and this build produce **byte-identical generated text** greedily (`-n 16`,
`-t 8`, single-shot, CPU) on Qwen3-0.6B Q8_0, Qwen2.5-7B Q4_K_M and
Qwen2.5-14B Q4_K_M — only the timing lines differ — which is the evidence that
adding the removal path did not perturb the model path.

**Conversation measurement.** `context_shift_real_model_measurement` (0.5B
Q4_0, `n_ctx = 192`, 12 turns with a system prompt) overflows on turns 8–11 and
shifts on every one:

```
turn 8:  dropped 22 KV rows at 17 (2 messages), prefill 14 tokens instead of 185
turn 9:  dropped 23 KV rows at 17 (2 messages), prefill 14 tokens instead of 180
turn 10: dropped 23 KV rows at 17 (2 messages), prefill 14 tokens instead of 175
turn 11: dropped 23 KV rows at 17 (2 messages), prefill 14 tokens instead of 170
```

— a **~13× cut in prefill tokens** per overflowing turn, `start = 17` (the
system prompt is retained and not even re-roped), no full re-render, and the
answers stay correct on the shifted window (`Stockholm,`/`Athens,`/`Warsaw,`/
`Lisbon,` for Sweden/Greece/Poland/Portugal). `prefill_tokens` is the observable
that makes "no longer re-prefills the whole conversation" checkable, so the
ticket's acceptance is a test, not a claim.

**Two pre-existing bugs the C2 tests found and fixed** (both outside the ticket
but blocking it, and both user-visible in `--cnv`):

1. The **first `user_turn` never prefilled pre-existing messages** — it wrote
   only the new message's delta, so the `--system` prompt never reached the KV
   while still being rendered into every later delta. The first turn now
   prefills the whole canonical render (`first_user_turn_prefills_the_system_prompt`).
2. A **pending EOT was written before the overflow check**, so a full context
   with a reply that had not reached EOG called `forward` at `position == n_ctx`
   and tripped the KV-region guard. The EOT is now deferred into the overflow
   path (it belongs at the end of the stream, so it commutes with the removal),
   and `ContextFull` keeps it pending for the next attempt.

**Not done here.** The *logical* mapping (a hole or a window with `cell != pos`,
which is what would let a backend keep the old rows in place) is deliberately
still refused by C1's scheduler gate — nothing consumes the resolved cell array
yet, and the physical removal makes it unnecessary for the sliding-window case.
C3's defragmentation, C4's quantized KV and C5's state save/restore are
untouched. The **server** path is untouched too: a full slot is still reported
as `finish_reason: "length"` and relies on B2's prefix-matched reuse, so a
server-side shift policy (and the `n_keep`-style rule that protects a system
prompt) belongs with E2's batching work.

### C3 — Defragmentation
- **Deliverable:** a cell-copy operation that compacts the arena; triggered when
  fragmentation exceeds a threshold.
- **Deps:** D1 (a copy needs either a view or an explicit copy op).
- **Acceptance (as resolved, 2026-09-19):** node-count and arena-utilisation
  counters before/after; the moved bytes are exact — `V` verbatim and `K` exactly
  `rope_shift_kv(old, delta)`, both asserted — and a mid-session compaction leaves
  the continuation's **greedy token** intact (asserted; a wrong re-rope flips it).
  Bit-identical *logits* were **not** claimable before C6 (a cell move shifted every
  RoPE angle, so the tail moved by the amplified rounding of that shift — `§14`
  row 9 carries the four probes that attributed it). **C6 landed the construction
  that removes it**: `positions` are sequence-relative and the allocator resolves
  `cells`, so a compaction now changes no angle and the continuation is bitwise
  (gated by `a_compaction_between_steps_keeps_the_continuation` and the inverted
  offset tests).
- **Follow-up: [C6](#c6--logical-positions-positions--cells)** does exactly this —
  sequence-relative `positions` plus an allocator-resolved `cells` input — which
  makes a cell move change nothing (a token's RoPE angle is its index within its
  sequence) and drops C3's K re-rope. When C6 lands, this ticket's acceptance
  tightens from the named amplified-rounding class back to **bit-identical logits**.

#### C3 design (written before the code, 2026-09-19)

**The problem the arena has.** `KvCache::reserve_seq` is first-fit over
`[0, n_ctx)`: it needs a *contiguous* free run of `cap` cells, while
`release_seq` frees whatever run a sequence held. Two sequences that finish out
of admission order therefore leave a hole in the middle, and the next admission
can fail with `no free run of N cells` while the arena has far more than `N`
free cells. That is fragmentation, and it is the one failure mode of E2's
per-sequence reservations that no amount of capacity fixes.

**Scope.** C3 compacts **downward**: live runs are packed from cell 0 in
ascending old-start order, each keeping its `cap`, so free space coalesces at the
top of the arena. Compaction never changes a run's `cap`, never reallocates a KV
region (the copy is *within* the same K/V buffer) and never changes a query's
logical position — but it does change the *cell* a sequence's rows live in, so
the caller that reserved the run must be told (the contract below).

**Where the copy executes.** The K/V regions are backend buffers (host memory on
CPU, device memory on CUDA), so the copy is a backend operation:
`Backend::copy_cells(&mut self, dst: BufRef, src: BufRef, rows, elems_per_cell)`
— a new trait method, implemented by CPU and CUDA in this ticket and refused
**loudly** by Metal (Phase G), which keeps standing rule 2: a backend that cannot
move cells fails the request instead of skipping the compaction.

The contract is `dst_row <= src_row` (downward only) and the implementation must
be correct for **overlapping** ranges, which the two obvious implementations are
not:
- CPU: `copy_within` (memmove semantics) is exactly right.
- CUDA: device-to-device `cudaMemcpyAsync` is documented **undefined** for
  overlapping ranges, so the kernel is one block that walks rows in **ascending**
  order with a `__syncthreads()` between rows. Ascending is the safe direction
  when `dst <= src` (the row a write could clobber is already copied), and a
  row's `elems_per_cell` (<= `n_kv_embd`) parallelises across the block's
  threads. No staging buffer and no second pass.

**The planner is pure.** `KvCache::compaction_plan(need)` is a host-side function
over the run table with no backend calls: it returns the moves (`seq`, `from`,
`to`, `rows`) that compact the live runs, or — with `Some(n)` — the shortest prefix
of that plan which opens a free run of `n` cells (`None` compacts fully). `rows`
is the sequence's **written** length (`last owned cell + 1 - start`), not `cap`:
unwritten cells hold no data worth copying, and moving them would shuffle another
sequence's stale bytes. Because the planner is pure, its policy is unit-tested
without a device — that is the half CI can prove.

**The contract with the caller.** Only whoever reserved a run caches its `start`:
E2's server keeps one per slot (`SlotState.start`) and passes `start +
current_pos` as the store position, so a move must be reported.
`GraphAllocator::kv_defrag(need)` returns the applied moves plus before/after
stats, and `BatchEngine` applies the new `start` to the slots it owns; everything
else follows, because the graph's window (`attn_span`) is *derived* from
`seq_slot` on every forward. A caller that ignores the report is not silently
wrong in a hard-to-find way: its next store would write to the old cells and the
span check would reject the query.

**Trigger.** `GraphAllocator::kv_reserve_seq_with_defrag` retries once through a
compaction when first-fit fails, unless `MINFER_NO_KV_DEFRAG` (presence-checked)
disables it — the A/B gate standing rule 3 requires: the same workload must
produce identical output with and without the compaction, and the counters must
show what it bought. The plain `kv_reserve_seq` stays pure, so a caller that keeps
no run bookkeeping (the single-sequence path, the graph tests) is unaffected; the
defrag-capable variant returns the moved runs precisely because a caller that
*does* keep one (E2's server) has to follow them.

**Counters (the ticket's other deliverable).** `KvArenaStats` carries `n_ctx`,
reserved/owned/free cells, the number of **free runs** (the "node count" C3
compacts), the largest free run, the live run count, and cumulative
`defrags`/`cells_moved`. The acceptance test records them before and after; the
same struct is what F8 (observability) will export.

**Increments.**
1. Planner + counters + `KvCache` bookkeeping (`apply_moves`) with unit tests —
   pure host-side logic, so CI proves it.
2. `Backend::copy_cells` (CPU + CUDA), `GraphAllocator::kv_defrag`, the
   reservation retry and the `MINFER_NO_KV_DEFRAG` gate.
3. Integration: fragment the arena on purpose (`--n-slots N`, prompts of
   different lengths) so an admission needs the compaction, and assert the
   server's replies stay bit-identical to the serial path, plus a device run for
   the CUDA copy.

**Deferred (recorded, not silently skipped).** Compaction does not *grow* a
sequence's run, does not evict, does not move a sequence up to make room for a
larger one, and does not run inside a forward (it is a between-forwards,
host-driven operation, so CUDA Graph capture is unaffected). Compacting the
arena *per layer* separately is also out of scope: one plan moves every layer
together, which is what keeps `owner[]` consistent across layers.

#### C3 record (2026-09-19) — increments 1 and 2

Landed: the planner, the counters and the bookkeeping (increment 1), and the copy
path (increment 2) — `Backend::copy_cells` on CPU (`copy_within`, i.e. memmove) and
on CUDA (a new `kv_move_rows` kernel: one block, ascending rows, a barrier between
them, no staging buffer), Metal refusing loudly (G5),
`GraphAllocator::kv_defrag` + `kv_reserve_seq_with_defrag` + the
`MINFER_NO_KV_DEFRAG` gate, and the server applying the returned moves to its slot
starts.

Evidence:

- CPU end to end: `kv_defrag_moves_the_bytes_and_opens_the_run` fragments a real
  16-cell arena (K/V regions allocated by a `kvcache_store` graph), asserts the
  8-cell reservation is **refused**, compacts, then checks the *bytes* at the new
  cells, the counters (`free_runs` 2 -> 1, `defrags`/`cells_moved`), and that the
  refused reservation now fits — and repeats the fragmentation so the retry
  helper has to hand the moves back. CPU suite: 207 passed / 0 failed / 5 ignored.
- Device: `cuda_copy_cells_moves_overlapping_rows_down` moves rows `[1, 4)` to
  `[0, 3)` — two of the three rows are read *and* overwritten — compares the whole
  buffer, and checks that the downward-only at the time (C7b later added the upward direction — see §5) contract is refused before a launch.
  CUDA suite: 255 passed / 0 failed / 5 ignored.
- Coverage split, stated rather than implied: the allocator's **CPU** arm is proven
  end to end and the CUDA *primitive* is device-proven, but "the allocator drives
  the CUDA copy" is glue that only increment 3's integration run exercises.

Increment 3 (2026-09-19) added the part that makes a compaction *correct* rather
than merely well-bookkept, and the model-level gate:

- **A compaction must re-rope the K rows it moves.** `positions` are cells today,
  so RoPE angles are absolute: a K row copied from cell `from` to cell `to` still
  carries the angle of `from`, and the next decode step attends to it at the wrong
  relative angle. `kv_defrag(need, rope)` now re-ropes the moved K rows through the
  model's own `rope_shift_kv` — the C2 path, one host pass per moved run, with
  **delta = `from - to`** (C2's convention: `rope_shift_kv(d)` means
  `new_pos = old_pos - d`). V carries no rotation and moves verbatim.
- **The gate caught its own sign bug.** `a_compaction_between_steps_keeps_the_continuation`
  (real 0.5B: prefill + one step at a non-zero start, release the holder below it,
  compact, then one more step) first failed on its argmax assertion because the
  delta was written as `to - from`; the allocator's *unit* test had passed anyway,
  because it computed its expectation with the same wrong sign. That is the whole
  argument for a behavioural gate next to a byte-level one.
- **The offset-alone effect, measured while gating.** The control run differs from
  the compacted one only in the subject's cell offset, and that alone moves the
  logits by **2.6% relative** (0.43 absolute) on the 0.5B *before* any compaction;
  the compaction's own contribution is the remaining ~0.2pp. A uniform RoPE shift is
  attention-neutral and V is unrotated, so an unidentified op is offset-sensitive.
  This is now §14 row 9 — it is why the C3 acceptance is a surviving greedy token
  plus byte-exact K/V rather than bit-identical logits, and it is the floor the
  compaction test prints.

Coverage: the planner, the counters and the bookkeeping are unit-tested; the
compaction's bytes, ownership and counters are asserted end to end on CPU
(`kv_defrag_moves_the_bytes_and_opens_the_run`, with the K re-rope computed
independently through `rope_shift_kv`); the CUDA copy primitive is device-proven
(`cuda_copy_cells_moves_overlapping_rows_down`); and the continuation gate runs the
real model. Still open: driving the compaction from the **server's** dynamic runs
(today every slot is reserved once at startup and packed, so the trigger cannot fire
there), and the logical-positions change that would remove the re-rope and the
offset sensitivity altogether.

### C4 — Quantized KV cache, Q8_0 first · [#42](https://github.com/yusiwen/minfer/issues/42) — **DONE (CPU: S1 + S2) 2026-09-23**

**Why.** f32 (and the GPU's f16-in-an-f32-region) is all the cache could be, so context
length was bounded by KV memory with no way to trade precision for it. Q8_0 is the first
**packed** layout: one cell is `ceil(n_kv_embd/32 * 34)` bytes instead of `4 * n_kv_embd`,
which is a real footprint reduction — unlike f16, whose regions stay f32-shaped and buy
bandwidth, not memory.

**What S1 landed** (`src/graph/kvformat.rs` is the new authority):

- `KvFormat { F32, F16, Q8_0 }` with a **strict** `MINFER_CACHE_TYPE` parser and a device
  policy (`resolve`, pure so CI covers the matrix): an unknown value is refused on every
  device — CUDA used to read anything that was not `f16` as f32, which is exactly the
  silent fallback the ticket forbids — and `q8_0` is refused loudly on CUDA and Metal,
  whose attention kernels address f32/f16 rows. `f16` on CPU stays the documented f32
  (a process whose env is set for a GPU run must not fail a CPU one); the load path
  (`models::load_model_ns`) turns any refusal into a failed load with the reason printed.
- **Packed rows stay addressable by the C3/C8b machinery.** A cell is rounded up to a
  whole number of f32 words, so `elems / n_ctx` is still one cell's width and
  `copy_cells` (the copy-on-write shift, the compaction) keeps moving rows verbatim with
  no change at all. `KvcacheMeta::row_elems` carries the cell width, `ensure_kv` sizes the
  region by it, and the node's shapes stay *logical* (`[n_kv_embd, n_ctx]`), which is what
  the store's K/V input means.
- **CPU store quantizes, CPU attention reads the packed blocks** — S1 dequantized the
  window into a scratch; S2's fused read is below. The CUDA/Metal kernels are still
  [issue #87](https://github.com/yusiwen/minfer/issues/87).

**Acceptance, as measured** (CPU, `cargo test --release`, 2026-09-22):

- *Footprint*: the real-model gate prints the persistent regions, f32 against q8_0 —
  0.5B (n_kv_embd 128): **6 291 456 B → 1 671 168 B (3.76× smaller)**; Qwen3-0.6B Q8_0
  (n_kv_embd 1024, head dim 128): **58 720 256 B → 15 597 568 B (3.76×)**.
- *Tolerance class* (never bitwise — the store rounds every K/V cell): at the reference's
  argmax the logits move by ≤ **1.0** (measured 0.60 on the 0.5B, 0.55 on Qwen3), and over
  the whole 152k-way vector by ≤ **3.0** (measured 2.50, ≤ 8 % of the 37.8 spread). The
  greedy continuation agreement is *reported*, not asserted: the CLI diverges at the 5th
  token on a chat-templated 0.5B prompt, which is the format's honest cost, not a defect.
- *The format itself is pinned one level down*: `a_packed_kv_region_answers_like_the_f32_one_and_is_smaller`
  asserts a stored cell is **bitwise** the Q8_0 quantizate of the row it was given (and
  that the f32 region holds the row verbatim), plus the 3× footprint and an attention
  comparison over three causal queries. The gate was mutation-checked: disabling the
  packed read path turns that comparison into max |Δ| = 3.2e38.
- *Refusals*: `MINFER_CACHE_TYPE=q8_0` on this box (CUDA available) ends the load with
  `minfer: MINFER_CACHE_TYPE=q8_0 is not supported on cuda yet: …`, and a typo
  (`banana`) with `… is not a KV cache type (f32, f16, q8_0); refusing rather than
  silently running with f32`. `ensure_kv` backstops both: a packed width on a non-CPU
  backend, and a width Q8_0 cannot express, are `Err` where the region is sized.

**Coverage, stated rather than implied.** The packed layout, the parser/policy matrix, the
store-exactness check, the refusal and the region accounting are unit-tested and run in
CI. The real-model gate is `#[ignore]`d (it flips a process-wide policy) and was run on
both cached models. Not verified here: any packed region on a GPU (by design, S1 refuses
it), and the fused dots' performance (S2).

**C4 S2 — the fused read, the quantize-aware shift, and what is handed off (2026-09-23).**

[#87](https://github.com/yusiwen/minfer/issues/87) named three items. The first two are
CPU-only and land here; the GPU kernels are the follow-up at the end of this record.

- **The fused read** (`attn_heads_q8`, `src/graph/cpu_backend.rs`). S1 dequantized the union
  of the batch's windows into a reusable f32 scratch — two allocations plus a write-then-read
  pass per attention node, per layer, per forward — and ran the unchanged f32 kernel over it.
  S2 reads the packed blocks directly: the K score is `dot_q8_0_q8_0` between the stored K
  blocks and the **Q8_0-quantized query row** (llama.cpp's form for a quantized cache), and V
  accumulates out of the cell block by block (`kvformat::accumulate_q8_0_row`). No scratch,
  no second pass. `MINFER_NO_FUSED_Q8_KV` (presence-checked) restores S1's path — the A/B
  standing rule 3 asks for — and the S1 path stays as the fallback for a `hd_kv` narrower
  than one Q8_0 block.
- **The quantize-aware shift.** `kv_rm`/`kv_shift` used to refuse a packed region because
  they re-rope K in f32 and a packed cell is not f32 rows. S2 keeps the property S1 already
  relied on — a cell is a whole number of words, so the survivors move **verbatim**, which is
  why `copy_cells` and the CoW/compaction paths needed no change at all — and then maps each
  survivor through `kvformat::map_q8_0_cells`: dequantize → re-rope → requantize. Stated
  honestly, the shifted K is `quantize(rope(dequantize(quantize(row))))`: C2's re-rope class
  composed with C4's packing class.
- **The store allocates once per node, not once per row** (`kvformat::pack_q8_0_cell_into`):
  a 2048-token prefill packs ~98k cells across 24 layers, and the per-cell `vec!` was the
  packed store's one avoidable cost.

**Acceptance, as measured** (CPU aarch64 on the GB10 box, 20 threads, Qwen2.5-0.5B Q4_0,
`n_ctx` 4096, `minfer bench -r 3`, greedy):

| config | tg128 (ctx 512) | tg64 (ctx 2048) | pp512 | pp2048 |
|---|---|---|---|---|
| f32 KV | 57.96 | 47.00 | 182.94 | 173.43 |
| q8_0 fused (S2) | **59.77** | **48.63** | 182.52 | 164.06 |
| q8_0 S1 scratch | 51.50 | 37.26 | 187.54 | 174.67 |

- *Decode is the measured effect*: the fused read is **1.16× (ctx 512)** and **1.31× (ctx
  2048)** faster than S1's dequantizing read, which is what the deferred item was for. Against
  f32 the packed cache went from **0.89× / 0.79×** (a real regression, which is why "does not
  regress by more than a named factor" was the acceptance) to **1.03× (both contexts)**. A
  second run at ctx 2048 read 174.89/50.20 f32, 162.10/50.22 fused, 156.62/37.66 S1 — the
  same decode ratios; the f32 baseline itself moves 47.0–50.2 between runs, so only
  within-run ratios are quoted.
- *Prefill is not measurably affected*: the three configs overlap inside their own stddev
  (3–6 tok/s).
- *Tolerance class*: the real-model gate's whole-vector bound moved 3.0 → 4.0 because the
  fused read adds a **new term** — the query's own Q8_0 quantization. Measured on the same
  gate: **3.029 fused vs 2.505 S1** over the vector, and at the reference's argmax **0.646 vs
  0.604**; the decoding-relevant bound stays 1.0. `MINFER_NO_FUSED_Q8_KV=1` reproduces the S1
  column, so the delta is attributable to the term rather than to run-to-run noise.
- *The shift*: after a 4-row shift of a 20-token prefill, the **first** decode step — the one
  whose input *is* the shifted context — is 2.466 from the f32 shift (0.064 at the argmax) on
  the fused path and 3.165 (1.074) on S1's; the greedy continuation agrees 4/8 and 1/8
  respectively, *reported*, because a flipped token makes the two runs different sequences
  from that step on. (An earlier version of this gate compared the **last** step instead and
  read 13.9 for a difference that is 0.06 at the decision point — the measurement, not the
  code, was wrong.)
- *Gates, each mutation-checked*, one level below the model:
  `the_fused_q8_read_matches_the_dequantizing_reference` (two KV heads whose K rows differ
  strongly, so a wrong head block cannot pass; using head 0's blocks for every head fails
  with max |Δ| 1.238 of a 1.313 spread — the shipped path measures 6.0e-5) and
  `a_packed_physical_shift_moves_v_verbatim_and_requantizes_k` (V verbatim **compared as
  bits** — a packed word is an f16 scale plus int8 quants, so as f32 it is frequently NaN and
  `==` on it is never true; K exactly the Q8_0 quantizate of the re-roped row; the tail
  zeroed; skipping the re-rope fails at row 0). The shift gate's first fixture filled only
  `positions` and left the builder's own `cells` input at zero, so all three rows went to
  cell 0 and the gate passed on zeros: that was found by mutating the re-rope and watching
  the mutant survive. The fixture now fills `cells` and asserts every row is non-zero.
- *Suites*: CPU **282 passed / 0 failed / 15 ignored** (was 280/0/15), green both with and
  without `MINFER_NO_FUSED_Q8_KV`.

**Handed off, still on [#87](https://github.com/yusiwen/minfer/issues/87):** the **CUDA and
Metal kernels** — `MINFER_CACHE_TYPE=q8_0` is still refused on both, and the refusals stay
loud. CUDA is the one that matters for throughput, and it is a kernel project rather than a
read-path change: `kv_ld4<KV>` and `stride_kv` address *elements*, so a packed cell needs
byte-based addressing plus a block-dequantizing load in every attention kernel (three window
modes each), a Q8_0 store, and the fused decode QKV epilogue's own store. Metal stays at G5
by the round's own decision.

**C4 S2b — the CUDA Q8_0 kernels: the plan of record (2026-09-24, design only).**

The CPU half of #87 landed (S2a). The device half is a kernel project, not a read-path change, so
it lands as its own increment against the same issue. This is the map it starts from: where the
CUDA backend touches a KV region, the design that covers those sites, and the dispatch cuts that
keep the first increment bounded. It came from walking the three files below at `a5f7961` + S2a,
so the line numbers are the ones to read, not to trust.

**What is there today.** No CUDA code knows `KvFormat`. The layout is one process-wide `bool`
(`cuda.rs`'s `KV_F16`), and every kernel addresses a cell as `row * nkt` *elements* of a
`float*`/`__half*` — there is no byte addressing anywhere, so a 34-byte-per-32-element cell
cannot be expressed by any current kernel signature. `set_kv_cache_type` maps anything that is
not exactly `f16` to false, so flipping the load gate before the kernels exist would silently
mean f32.

**The design.** A layout tag plus byte-addressed row accessors in `cuda_kernels.cu`:

- `KV_LAYOUT_F32 / KV_LAYOUT_F16 / KV_LAYOUT_Q8_0`, `kv_row(base, cell, row_bytes)`, and
  `kv4<LAYOUT>(row, elem) -> float4` as the one load idiom: f32 is the old `float4` load, f16 the
  old two-`__half2` pair (both bit-identical to today's instantiations), and Q8_0 reads block
  `elem/32`'s f16 scale plus four quants at `2 + elem%32`. A 4-element group never straddles a
  block, because a KV head's base is `hd`-aligned and `hd % 32 == 0` — the property
  `ensure_kv`'s packed-width check already enforces;
- kernels take `const void* k/v` + `size_t row_bytes` instead of a typed pointer plus
  `stride_kv = nk * hd`, templated on `int LAYOUT` rather than `typename KV`;
- `store_kv_q8_0`: one thread per (row, 32-element block), with the CPU store's own quantizer
  (`amax/127`, f16 scale, `round_ties_even`), so both backends store the same bytes.

**The access sites to convert** (`src/cuda_kernels.cu`, plus the host rows below):

| Site | Lines | What it is |
|---|---|---|
| `kv_ld4<KV>` specialisations | 2935–2950 | the split body's two loads |
| `attn_split_1w_body<KV,…>` | 2957–3053 (3013, 3016) | K+V, `cell[j]`-indexed, `hd/4` dims per lane |
| `gqa_attn_split_partial<KV,…>` | 3055–3083 | decode (nt == 1) |
| `gqa_attn_split_partial_bt<KV,…>` | 3120–3137 | spec-verify (1 < nt ≤ 16) |
| `gqa_attn_f32<CAUSAL,MAP>` | 3456–3575 (3500/3510/3535/3544) | the general kernel (nt > 16 prefill) |
| `store_kv_f32` / `store_kv_f16` | 2530–2569 | the store both dtypes use |
| `attn_bias_rope_store_f32` | 2590–2667 (2649–2665) | the fused decode epilogue's K/V store |
| `gqa_attn_f32_f16kv`, `attn_split_h4w_body`, `fa_stage_kv_async`, `fa_prefill_f16kv` | 2781–2900, 3273–3431, 4327–4372, 4374–4648 | the f16-specialised paths (see the cuts) |
| `kv_move_rows` | 8703–8720 | the compaction mover — a plain word copy, so a packed cell (a whole number of f32 words) needs **no change**; the host already passes the cell's word count |
| host launchers | `cuda.rs` 349–897 (FFI), 4481–4755 (`CudaState`), 5393–5488 (store/epilogue) | `typename KV` / `f16_kv: bool` become a layout tag |
| executor | `cuda_backend.rs` 26/114/133 (the field), 1075–1109 (store), 1111–1262 (dispatch), 1510–1541 (`copy_cells`), 154–156 (test setter), + 41 `kv_f16` test sites | the format decides store, dispatch and the move stride |
| format plumbing | `cuda.rs` 1456–1478, `kvformat.rs` 124–133 (`supports`), `alloc.rs` 867–939 (`ensure_kv`'s packed refusal) and 2035–2059 (`kv_element_format`), `models/{qwen2,qwen3}/loader.rs` 462–464 | one authority: the device reads the format the loader already resolved |

**The dispatch cuts that keep the first increment bounded** — build-time choices
(`no silent fallback`), each stated when the format is enabled:

- **decode (nt == 1)** takes the split-K path through the converted 1-warp body; its hybrid 4-warp
  dispatch (`hd == 128`, `nkv >= 1921`) is f16-typed and is skipped for Q8_0 (`rpw_gate = 0`);
- **1 < nt ≤ 16** routes to `gqa_attn_f32` instead of the batched split kernel, which exists for
  spec-verify's *bitwise* identity contract with sequential decode. A speculative session must
  therefore **refuse** a packed cache loudly (a draft keeps its own KV, and the two reduction
  schedules would no longer agree) — consistent with S2a, which already refuses a draft in the
  session container;
- **prefill (nt > 16)** routes to `gqa_attn_f32`: the FA path (`fa_prefill_f16kv`, f16-typed
  shared-memory staging) is not offered for Q8_0 in this increment, so the prefill is correct but
  off its tuned path;
- **the fused decode epilogue** (`Op::FusedQKV` / `Op::QkvBiasRopeStore`) is not built for a
  packed cache — the model builders' `layer_gpu` gate gains `&& !packed`, and a Q8_0 decode takes
  the unfused bias/rope/store chain through the converted store kernel.

**Acceptance** (the issue's, made concrete): `cuda_map_window_matches_the_span_over_the_same_rows`
(`cuda_backend.rs:3678`) extended to Q8_0 rows — it writes its own region contents and calls
`exec_ids` directly, so it needs a packed cell encoder and a packed-word region size; a Q8_0
store→attention round trip (`cuda_kv_f16_roundtrip_attn`, `:5150`, is the pattern); `copy_cells`
under the packed stride (`cuda_f16_kv_cell_move_strides_by_row_bytes`, `:3946`); the real-model
gate `a_packed_kv_cache_answers_like_the_f32_one` run with the `cuda` feature on GB10 (CI has no
GPU, and its `MINFER_C4_MODEL` arm covers a second model); and an A/B against f16 on the same
model/context with the numbers recorded. The honest expectation, stated before the work: the win
on the device is **memory** (3.76× less than f32, 1.88× less than f16); whether decode bandwidth
also improves depends on the block-dequant instruction count, so the bar against f16 is "no worse
than a named factor", not "faster".

**Why it is not in this increment.** ~10 kernel sites, 3 launchers, ~15 host sites and 41 test
sites, each needing an nvcc iteration and — for the gates — a serial device run. It is recorded
here rather than half-wired. [Metal's half stays at G5](https://github.com/yusiwen/minfer/issues/44).

### C5 — Session save and restore · [#43](https://github.com/yusiwen/minfer/issues/43) — **DONE 2026-09-22**

**Why.** A session's KV rows, ownership and run table lived only in memory, so every
restart re-prefilled the whole context.

**What landed:**

- **`src/graph/kvsession.rs` is the container.** An 8-byte magic, a version, flags, the
  backend tag and the shape (`n_layer`, `n_ctx`, `n_embd`, `row_elems`), then one K/V blob
  per layer as raw little-endian pool words, then the bookkeeping, then an FNV-1a
  checksum. It streams both ways: a save never holds a second copy of a 6 MB–1 GB arena,
  and a load materializes one layer at a time.
- **`KvCache::session_state` / `restore_session`** is the bookkeeping half — the owner
  table, `n_used`, the run table (`SeqSlot`: start, cap, shared prefix, written extent),
  each sequence's span list, `identity` and the C3/C8b counters — kept separate so it
  round-trips in unit tests without a device. `restore_session` validates before it
  applies: arena capacity, the owner table's length, every reservation and span inside the
  arena, and every live sequence carrying a span list.
- **`GraphAllocator::kv_save` / `kv_load`** move the bytes through the backends'
  `read_host`/`write_host` (CUDA included) and enable the pool a file names if it does not
  exist yet, because a restore happens before the first graph — which is also how the CUDA
  run of the gate found the gap. The header records the KV element type
  (`f32`/`f16`/`q8_0`), so a file written under one width cannot be resumed under another.
- **A failed load is a no-op.** `kv_load` runs `kvsession::verify` — a full pass over the
  header, every layer's declared length, the bookkeeping, the checksum and end-of-file —
  **before** `ensure_kv` creates a single region. Every refusal path is asserted to leave
  no arena behind (`kv_n_used(0).is_none()`).
- **Truncation is caught by construction**, not only by the checksum: every read is exact
  against a length the header fixes, so a short file fails wherever it stops with
  "truncated — the file ends inside …". A version bump, a bad magic, trailing bytes, an
  unknown flag, a header whose packed flag and cell width disagree, and a file whose
  backend / `n_ctx` / `n_embd` / element type does not describe this run are each refused
  with the reason.

**Acceptance, as measured** (2026-09-22):

- *Container, in CI*: 9 unit tests — a round trip including the run table, a packed
  session, truncation, a version mismatch, a flipped byte (checksum), trailing bytes, a
  foreign file, a header whose flag and width disagree, and a writer short a layer.
- *Allocator, in CI*: the region bytes **and** the run table survive a save and a load
  into a **fresh** allocator, the restored table resolves the same cells, and four refusal
  paths (another `n_ctx`, another row width, another backend, another element type) plus a
  truncated file each leave the allocator untouched.
- *Real model* (0.5B q4_0, `#[ignore]`d, run on the **CPU and on CUDA**): prefill, save
  (24 layers / 256 cells / 5 written / 6 316 748 bytes), drop the cache, restore into a
  fresh one, then 8 greedy steps against the session that never left memory — **max
  |Δlogit| = 0**, i.e. bitwise, which is the strongest form of "the same continuation".
- Handed off: resuming the CLI's `--session` (which still re-prefilled its history JSON)
  from this container, and an E2 slot-table snapshot for the server, are
  [#89](https://github.com/yusiwen/minfer/issues/89).

**C5 S2 — the CLI resumes, the host state rides along (2026-09-24).**

The container described the KV *rows*; the rows belong to a host state, and a restore that
brought one back without the other would be a different session. S2 closes that gap for the
CLI and records what the server still needs.

- **The container gained a host-state section (version 2).** An opaque, length-prefixed blob
  after the bookkeeping and **inside the checksum** (`KvSessionWriter::set_host`,
  `KvSessionReader::finish` → `KvSessionBody`), so the two halves travel together or not at
  all; a version-1 file has no such section and is refused loudly, which is exactly what the
  caller's fallback path is for. `kv_save_with_host`/`kv_load_with_host` carry it through the
  allocator; `kv_save`/`kv_load` stay as the no-host wrappers every existing gate uses.
- **`Conversation::snapshot` / `restore_snapshot`** is the host half: the message list,
  `stream_tokens`, `current_pos`, `turn_pos`, `prev_tokens` and `need_insert_eot`, as a
  versioned JSON blob (`SNAPSHOT_VERSION`), with validation before it is applied — a snapshot
  whose `current_pos` disagrees with the token mirror, or that does not fit this run's
  `n_ctx`, is refused rather than half-applied.
- **`--session FILE` (under `--cnv`) writes `FILE.kv` on exit and resumes it on start.**
  Everything that could make a resume wrong falls back to re-rendering the JSON **with the
  reason printed**: another `n_ctx`/model/`MINFER_CACHE_TYPE`, a history the user edited
  between runs (the snapshot and the JSON must agree), an unknown snapshot version, a
  missing companion, an engine that cannot hand its KV to the host (a speculative session —
  the draft keeps its own KV — and the mocks), and any file the container itself refuses.

**Acceptance, as measured** (2026-09-24, Qwen2.5-0.5B Q4_0, CPU, `--n-ctx 1024`,
1530-character history, greedy, `-n 8`):

- *Continues alike*: turn 2 in a **new process** resumed from the companion answers
  `The answer to 2+2 is` — byte-identical to the same turn in the process that never left
  memory, **and** to a run with the companion moved away (the re-seed path). So the resume is
  behaviour-preserving, not merely fast.
- *And prefills nothing at startup*: the resumed run prints
  `resumed 2 message(s) and 379 KV row(s) from …seed.json.kv (25 268 624 bytes) — 0 tokens
  prefilled`. Wall clock to the identical continuation: **0.47 s resumed vs 2.36 s re-seeded**
  (5.0×), the difference being the ~370-token history the JSON path re-renders.
- *Mismatches are loud*: `--n-ctx 512` against the 1024-cell companion prints
  `KV session: the file describes a 1024-cell arena, this run has 512 (--n-ctx); re-seeding
  the history instead` and continues correctly.
- *Gates, each mutation-checked*: `a_resumed_snapshot_prefills_nothing_and_continues_alike`
  (the restored run issues the **same engine calls, at the same positions**, as the in-memory
  run — forgetting `prev_tokens` in `restore_snapshot` fails it),
  `the_host_state_round_trips_and_is_covered_by_the_checksum` (a flipped byte inside the host
  blob is refused; dropping the host from the writer fails it),
  `a_version_1_file_is_refused_so_the_caller_can_re_seed`,
  `a_snapshot_that_contradicts_the_host_mirror_is_refused`, and
  `an_engine_without_a_kv_refuses_the_session_calls`. Suites: CPU **287 passed / 0 failed /
  15 ignored** (was 282/0/15).
- *Not verified here*: a CUDA or Metal session companion (the container is backend-tagged and
  the CPU path is what this box measured; the CUDA half of C5's own gate was run at S1).

**Still open on [#89](https://github.com/yusiwen/minfer/issues/89): the server slot
snapshot.** `server::batch` keeps a per-slot table (`seq`, `start`, `cap`, the in-flight
`Run`) that admission rebuilds from scratch, so a restart drops every in-flight conversation
even though the rows are recoverable in the same container. That needs the slot table and
each slot's request state in the host blob, a restore path in admission, and a refusal for a
snapshot taken under another `--n-slots`/`n_ctx` — a server-lifecycle increment, so it is
handed off rather than half-wired here.

### C6 — Logical positions (`positions` ≠ cells)

**Why.** §14 row 9 closed the cell-offset sensitivity by measurement: a sequence's
logits' tail changes when its run moves because `positions` is *both* the RoPE angle
and the KV cell index, so a cell move shifts every rotation. The intervention
experiment proved that RoPE is the **only** entry (`offset_divergence_is_caused_by_the_rope_rounding_alone`:
injecting run A's 48 rope outputs into run B makes the logits bitwise identical),
and the sweep showed the effect saturates at a ~1e-6 distributed perturbation
(`a_distributed_rope_perturbation_saturates_the_logits_tail`). Two consequences
drive this ticket: C3's compaction needs a K re-rope (and can never be
bit-identical), and any future cross-slot sharing inherits the same coupling.

**Semantics.** `positions` becomes what a caller naturally has — the token's index
*within its sequence* — and the allocator resolves, per token, the KV row it must
be written to:

| Input | Meaning | Consumed by |
|---|---|---|
| `positions` | sequence-relative token index | RoPE (q and k), the causal bound |
| `cells` (new) | the row `KvCache` resolves for `(seq, position)` | `KvcacheStore` and the fused decode QKV family (`Op::FusedQKV`, `Op::QkvBiasRopeStore`) |
| `attn_span` | unchanged: the `[lo, hi)` **cell** range | attention |

The single-sequence case is unchanged by construction: its run starts at cell 0,
so `cells[t] == positions[t]`, and the classic path stays bitwise.

**Invariants / gates.**
1. every existing single-sequence test stays bitwise (regression);
2. batched forward == the same sequences run one at a time, **same layout**,
   bitwise (the existing gate, migrated to relative positions);
3. **a compaction between steps leaves the logits bitwise identical** — the literal
   acceptance C3 could not claim before, and the reason this ticket exists;
4. a backend that cannot resolve `cells` refuses the node (`Err`, no silent
   fallback); Metal is out of reach by construction because it already refuses
   multi-sequence attention (`supports_attn_span() == false`), so it never sees a
   non-zero run start and keeps `positions == cells` (recorded as G5, no Metal code
   change in this ticket);
5. decode timing on GB10 does not regress (the fused QKV family is the only hot
   path this touches, ported in S3 with its own A/B).

**Change surface.**
- **IR**: `GraphBuilder::kvcache_store` wires its row input to a new `cells` node
  (created like `attn_span`); the fused QKV family (`FusedQKV`,
  `QkvBiasRopeStore`, `FusedQkvNorm`) is gated off while `explicit_span` is set
  (S1) and takes `cells` in S3.
- **Allocator**: `fill_batch_inputs` / `fill_attn_inputs` / `fill_seq_ids` stop
  treating `positions` as cells — `own_range`, `kv_note_used` and `attn_span`
  resolve through the run table — and fill `cells`.
- **KV store**: `KvCache::cells_for` (today the identity) and `attn_span` (today
  requires `position ∈ run` and `owner[position] == seq`) become the resolver;
  `check_positions_bound` / `check_attn_span` check the *cells* form.
- **Models**: `forward_batch`'s `explicit_span` decision and positions stay as they
  are semantically; the store wiring is inside `kvcache_store`.
- **Server**: `positions = slot.start + current_pos` becomes `current_pos`;
  `SlotState.start` drops to reporting/diagnostics.
- **Backends**: the `KvcacheStore` arms need **no change** — they already write at
  the rows their third input names; only that input's producer changes. This is
  what makes S1 landable with the suite green on all three backends.
- **C3**: `kv_defrag(need, rope)` loses the rope parameter, and the `KvRope` plumbing
  added for the re-rope goes away.

**Staged plan** (each step: local CPU suite, plus the CUDA suite when it touches
CUDA, then a docs progress update, then a commit on `feat/logical-positions`).

| Step | Content | Gate |
|---|---|---|
| S0 | this design, the roadmap/status corrections | docs build (`check-docs`) |
| S1 | `cells` wiring + allocator/kvcache resolver + model/server switch + fused QKV gated off under `explicit_span` | invariants 1–2, 4; CPU+CUDA suites |
| S2 | compaction without a re-rope; the bit-identity gate | invariants 1–3 |
| S1∪S2 | *merged in execution:* the re-rope removal is **not** optional once S1 lands — S1 makes the stored K rows' angles sequence-relative, so a compaction that still re-roped them by `from - to` would rotate them *away* from the correct angle. The first S1 run showed exactly that: `a_compaction_between_steps_keeps_the_continuation` failed with a flipped greedy token until `kv_defrag` stopped re-roping. Semantics switch and re-rope removal must therefore land in the **same** commit. | as above |
| S3 | CUDA fused QKV family takes `cells`; the gate re-enabled for CUDA — **landed**, plus a server position bug the gate exposed | invariant 5 + fused-vs-unfused bitwise + capture regression |
| S4 | docs closure (roadmap §2.4, AGENTS rules, design docs) | docs build (`check-docs`) |
| S5 | PR, CI four jobs green with zero annotations, rebase merge | CI |

> The `docs build` gate is the CI job **`check-docs`** (`.github/workflows/ci.yml`):
> `mdbook build` with the deployed toolchain plus `scripts/check_docs_links.py`. It did not
> exist when S0/S4 ran — the docs were checked by hand then — and it landed with
> [#63](https://github.com/yusiwen/minfer/issues/63); the rows above name it so the record
> and the gate agree.

#### C6 progress (2026-09-19)

**S0 done** (`71b86da`). **S1 + S2 landed** — the semantics switch and the re-rope
removal went in as one commit, as the merged-step row above requires. Suites:
**CPU 213 passed / 0 failed / 5 ignored**, **CUDA 261 passed / 0 failed / 5 ignored**.

What S1 changed (production):

- `GraphBuilder::cells_input`; `kvcache_store` creates and consumes it, so callers
  no longer pass a row buffer (26 call sites migrated);
- `KvCache::attn_span` resolves `cell = run.start + position`;
  `GraphAllocator::kv_cells_for_seq` fills `cells`, with the classic
  single-sequence identity fallback when no run is reserved;
- `fill_batch_inputs` / `fill_attn_inputs` / `fill_seq_ids` use relative positions,
  and `fill_attn_inputs` only resolves `cells` when the graph has that input (a
  rope-only fixture must not need a KV arena);
- qwen2/qwen3 gate the fused QKV family off while `explicit_span` is set (S3 ports
  it for CUDA), and `server/batch.rs` passes relative positions;
- `kv_defrag` no longer re-ropes and the `KvRope` plumbing is gone (S2).

What the tests became, i.e. the new gates: the minimal hand-built attention graph
asserts **bitwise** invariance to the cell placement (it used to be "within the
rotation's rounding"); the four offset-sensitivity tests invert into "a cell
placement changes nothing" (logits, per-layer V, per-node `q`/`k`/attention); and
the perturbation probe was **deleted**, because its premise — that the offset
effect needs explaining — is gone.

**S3 done** — the CUDA fused decode QKV family takes `cells`:

- `attn_bias_rope_store_f32` gained a `cells` parameter: `pos` rotates q/k and
  `row = cells[0]` addresses the four KV store writes. The launcher, the FFI
  declaration and `CudaState::attn_bias_rope_store` carry it; the builder wires
  the shared `cells` input into `fused_qkv` (sources `[x, pos, cells]`) and
  `qkv_bias_rope_store` (`[q, k, v, pos, cells]`), so no model call site changed;
- the qwen2 gate becomes `(cuda_on || !explicit_span)`: CUDA keeps the fused
  chain under an explicit span, Metal keeps the pre-C6 gate (it has no
  explicit-span attention at all, G5). Qwen3 needs nothing: its fused family is
  Metal-only, so no CUDA arm existed to port. The hand fixtures in
  `cuda.rs::d38_probe_tests` pass their positions buffer as `cells` — identity
  by construction, which is the case they assert.

Suites: **CPU 213 passed / 0 failed / 5 ignored**, **CUDA 261 passed / 0 failed /
5 ignored**. End-to-end gate on GB10 (`Qwen2.5-0.5B` Q4_0, `--n-slots 2
--n-ctx 1024`, one short + one long request submitted together): `cap = n_ctx/2 =
512`, so the long request ran in **slot 1** (server log) and, once the short one
answered in a token, **alone** — a single-token (`nt == 1`) step in a run whose
start is not 0, which is exactly the path the S1 gate had closed. Fused vs
`MINFER_NO_FUSE_QKV=1` produced **byte-identical** completions (`sha1
36c55ed81464`, 319/319 expected words): the `cells` port and the S1/S2
`KvcacheStore` path agree on a non-zero run start.

**The gate also caught a real server bug, now fixed:** `BatchEngine::submit_on`
still built positions as `start + i` — the pre-C6 "positions are cells"
semantics — so a request placed in a non-zero-start slot fed position 512 into a
512-cell run and `kv_cells_for_seq` rejected it ("past sequence 2's reserved
run"), rejecting the job. Both the batch prefill path and `current_pos` were
already relative; only this entry point had not been migrated, and no earlier
test exercised a second slot's run (the CPU suite batches nothing by default, so
every server test ran the single-sequence path). It now passes `feed_from..nt`.

Regression coverage: `server_batch_matches_serial_and_is_faster` pins each of four
requests to the slot it would occupy through `submit_on`, so it drives runs with
non-zero starts — it is `#[ignore]`d only because CI has no cached model, and it
must be run locally for any change to positions or the run table:
`cargo test --release -- --ignored server_batch_matches_serial`. It passes both on
CPU (byte-equality of batched vs serial, 1.43x) and on CUDA (structural
assertions, 1.28x).

One measurement note worth keeping: an end-to-end A/B of the fusion on the
*batched* decode path cannot isolate this gate, because `fuse_qkv` requires
`nt == 1` — a step carrying four streams is unfused in **both** configurations.
The measurement that matters is the single-stream step in a non-zero-start run
above; a 4-slot throughput A/B (master, 7B Q4_K_M, 4 concurrent requests) showed
the fused and unfused chains within noise (92.6 vs 92.7 tok/s best case), which
is expected for a bandwidth-bound model and is why the port's value is
correctness and parity, not throughput.

The two pre-C6 experiments that established the attribution above — the RoPE
injection (EXP1) and the distributed perturbation sweep (EXP2) — are archived with
their code and measured numbers in `experiments/logical-positions/`.

**Not in this ticket:** sharing one cell range across sequences (`owner[cell]`
becomes a set/refcount — the real cross-slot prefix reuse, which this ticket makes
possible), C4 (quantized KV) and C5 (state save/restore) — both should follow this,
because each would otherwise multiply the addressing surface.
- **Explicitly not changed:** C2's `kv_rm`/`kv_shift` re-rope stays: it changes
  *positions*, not cells, so the angles genuinely have to move. Only the
  *compaction* re-rope disappears.

### C7 — Dynamic runs: one request may use the whole arena · follows C6 · M · [#59](https://github.com/yusiwen/minfer/issues/59)

**Why (user-visible).** `BatchEngine::new` hands every slot a fixed
`cap = n_ctx_total / n_slots` up front, and `submit_on` rejects anything whose
`prompt + generation` exceeds that cap (`prompt of N tokens exceeds slot context of
M`). With `--n-slots 4 --n-ctx 8192`, a single 6000-token request is refused while
three slots sit idle and their rows are free: the **arena** is sized for the total,
the **partition** is the limit. This is the most user-visible consequence of the
fixed reservation and the reason C3's "server-side dynamic-run trigger" was recorded
as a follow-up instead of being closed with C3.

**What.** Make the partition elastic instead of fixed:
- a slot may grow past its initial cap when the admitted request needs it, via
  `kv_reserve_seq_with_defrag(seq, need)` — C3 already plans the move, copies the
  rows through `Backend::copy_cells` and returns the moves;
- the shrinking slots are re-reserved around the growth, and **every caller that
  cached a run `start`** (E2's `SlotState`) applies the returned moves — the contract
  `kv_defrag` already documents. C6 is what makes this safe: a moved row keeps its
  sequence-relative position, so no other slot's logits change and nothing re-ropes;
- the repartition happens at one serialization point (admission), so no in-flight
  step can observe a half-moved arena;
- a genuinely full arena still rejects loudly with the existing message shape — one
  request never silently shrinks another's context.

**Gates.** (1) `--n-slots 4 --n-ctx 8192` serves one request of ~8k tokens while the
other slots are idle (the literal requirement); (2) after a grow, each idle slot's
next request is **bitwise identical** to the same request on a fresh engine; (3) the
4-slot batched-vs-serial byte-equality test stays green
(`server_batch_matches_serial_and_is_faster`, run locally with `--ignored`); (4) the
CUDA batched-decode throughput stays within noise of the recorded numbers.

**Not in this ticket:** sharing (C8) — C7 only makes the partition elastic, two
sequences still never look at the same cell.

#### C7 increment 1 (2026-09-20) — the partition is elastic

**What landed.** `BatchEngine` now sizes a slot from the request instead of from the
startup partition: `wanted_cells_from` (pure, unit-tested) asks for
`prompt + max_tokens + 1` for a bounded request and `prompt + slot cap + 1` for an
unbounded one, clamped to the arena and never below the prompt. When that exceeds the
slot's `cap`, `ensure_slot_capacity` reclaims the **idle** runs above it (they hold at
most B2's prefix hint, which the slot re-prefills when it is next used), then
re-reserves this slot's run **at its existing `start`** and re-owns its written prefix;
the moves the KV store returns are applied to every slot's cached `start`
(`apply_run_moves`, the contract C3 documented).

No new KV primitive was needed — `kv_release_seq`, `kv_reserve_seq_with_defrag` and
`kv_own_range` compose into "grow in place" — and no backend changed, because the store
already writes at the allocator-resolved `cells` (C6). A reservation that would land
anywhere other than where the rows physically are is refused and the slot is left to
re-prefill: the one outcome C6 exists to prevent (reading rows at an offset they do not
have) cannot happen. The HTTP layer's request bound moved from a slot's share to the
whole arena (`AppState::n_ctx`); the serial path (batching off) keeps its own
slot-sized bound, because its graph region really is that size.

**The boundary, and why it is no longer padded.** `tick` forwards the committed token
*before* `advance` can report `length`, so a generation that reaches its cap used to
perform one more forward at the cell after its last token — a whole weight pass whose
logits are discarded, plus a cell nothing reads. The reservation was therefore padded
with one cell, and a reservation of exactly `prompt + max_tokens` rejected that batch
(`kv_cells_for_seq: position N is past sequence S's reserved run`).

That padding is gone: `advance` now decides **before committing** whether another token
can still be used (the request's token budget, and `current_pos + 1 < cap`), and ends the
turn instead of committing one that could only be discarded. `wanted_cells_from` is
exact — `prompt + max_tokens`, clamped to the arena — and the top-of-`advance` bound is
the safety net it always was. The regression is a request with no token budget
(`max_tokens = -1`) on a run sized to its prompt plus 48 cells: it must end with
`length` and exactly 48 tokens, because a forward past the run is rejected by
`kv_cells_for_seq` — before this change that request failed instead of finishing.

**Gates.** (1) `wanted_cells_plans_from_the_request` — the policy, no model needed;
(2) `a_long_request_may_use_the_whole_arena` — a 302-token prompt on `n_ctx = 366` with
four slots (91 cells each) is served, and **both** admission paths (`submit` /
`prefill_group` and `submit_on`, the one the server uses) produce a continuation
byte-identical to a one-slot engine that needs no reclaim; (3) the four-slot
batched-vs-serial byte-equality test still passes; (4) CPU and CUDA suites green;
(5) on GB10, `--n-slots 4 --n-ctx 8192` with a 2054-token prompt logs
`slot 0: capacity 2048 -> 2071 cells ... (released 3 idle slot(s) above)` and answers
the same bytes as `--n-slots 1`, which needs no reclaim at all. Gate (5) is the point
of the ticket: C6 makes a moved row arithmetic-free, so where the partition puts a
sequence cannot show up in its output.

**C7b — both directions, so a busy neighbour is not a wall (2026-09-20).** The
increment-1 caveat is gone: `Backend::copy_cells` and CUDA's `kv_move_rows` kernel now
move rows **up** as well as down (the kernel walks one row at a time with a barrier,
descending when the run slides up), `apply_moves` accepts `to > from` and frees the
vacated cells in either direction (guarded by the sequence's own stamp, so a plan can
never clear another sequence's ownership), and `KvCache::set_cap` grows or shrinks a
reservation and returns the plan — shrinking below a sequence's written rows is
refused, and so is a growth the arena cannot hold with the other reservations. The
allocator's `kv_set_cap_with_defrag` copies the rows and then renumbers, like
`kv_defrag`.

The subtle half is **order**: wherever ranges overlap, a destination must never land on
a row that has not been copied yet. `kvcache::order_moves` is the single rule — upward
moves top-down, downward bottom-up, upward first — and both the data copy and the
bookkeeping follow it (the owner table is a mirror of where the rows live, so they have
to travel together). The unit test caught exactly this: applying the plan in its raw
ascending order moved one sequence's stamps through cells another had already
overwritten.

**Gates.** (1) `growing_a_run_pushes_the_runs_above_it_up` — a pure test of the plan,
the upward move and the owner stamps; (2) `a_resize_refuses_to_eat_rows_or_overcommit_the_arena`;
(3) the CPU and CUDA twins that used to pin "upward is refused" now pin the *result* of
an overlapping upward move (a memmove's output) — the CUDA one on GB10;
(4) on GB10 the CUDA twin pins the upward *kernel* path (the bytes a memmove would
produce). Suites: CPU **216** passed, CUDA **264** passed (0 failed, 6 ignored each).

**The `cells` bound is its own rule (2026-09-21) — [#60](https://github.com/yusiwen/minfer/issues/60).**
`check_positions_bound` classified any
I32 input consumed by a KV-writing op as positions, and `cells` — which now feeds the same
ops (C6) — was measured by that rule. It cannot false-reject (a cell always indexes the
arena, which is exactly `n_ctx` cells), but the two bounds only coincide *because* a cell
equals its position, which is the coincidence C8 removes. `cells` now has its own bound and
its own message (`input 'cells': cell N is past the M-cell arena`), pinned by a test that
also covers the no-arena case.

**End-to-end (2026-09-20) — the engine scenario is verified.** Four slots at
`--n-ctx 8192`: a short request, then a long generation on another slot, then a
2148-token prompt admitted while that generation is still running. On GB10 the server
logs `slot 0: capacity 2048 -> 2157 cells for a request wanting 2157 (released 1 idle
slot(s); 2 run(s) moved)` — two runs moved, one of them the live generation above — and
on CPU (forced with `MINFER_BATCH=1`, which makes the comparison deterministic) the same
scenario moves one run. In **both** runs the live neighbour's continuation is
**byte-identical to the same request served alone**: its rows travelled up under it and
its answer did not change, which is exactly what C6 + C7b promise. The grown request
itself is compared structurally (a shared 38-byte opening with its one-slot baseline),
because it shares the batch with the live request and therefore takes the windowed
(`explicit_span`) attention path while the baseline takes the causal one — a named
tolerance class, not a bitwise property.

That gate was missed twice for an environmental reason worth recording: `cargo test
--release` overwrites `target/release/minfer` with a CPU-only build, so the server logged
`batching: off (device cpu)` and served the *serial* path while the script believed it was
testing the engine. The script now asserts `batching: on` **and** the device line in the
log before it measures anything — a gate that cannot check its own preconditions is not a
gate.


### C8 — Cross-sequence cell sharing (`owner` → set/refcount) · follows C7 · L · [#41](https://github.com/yusiwen/minfer/issues/41)

**Why.** A prefix shared by several sequences (a system prompt on every slot; B2's
prefix reuse, but *across* slots) is duplicated today: each slot stores its own copy
and pays its own prefill, so N sharers cost N× the memory and N× the prefill for the
same tokens. llama.cpp's unified cache shares one cell between sequences through a
per-cell sequence set (`seq[i]` is a bitset, with `seq_cp`/`seq_rm`/`seq_keep` over
it); this ticket is that capability, scoped to what minfer's server needs.

**What.** `KvCache::owner[cell]` becomes a set (bitset or refcount); `attn_span` and
`kv_cells_for_seq` accept "owned by any of these sequences"; a new
`kv_seq_cp(src, dst, range)` shares a prefix's cells with another sequence; `kv_rm`
frees a cell only when its last owner drops it; and the store must not write *into* a
shared prefix — the first divergent token copies the shared cell it would have
overwritten into a private one (copy-on-write). The CoW rule is the delicate part: a
store that wrote through a shared row would corrupt every sharer, so it is either
implemented or refused, never ignored.

**Gates.** (1) two slots sharing a prefix produce continuations **bitwise identical**
to the same two slots run with private copies (CPU first; CUDA may take a named
tolerance class only if a kernel shape differs); (2) removing one sharing sequence
leaves the other's logits bitwise; (3) a store that would write into a shared cell
copies it or fails loudly; (4) `KvArenaStats` counts a shared cell once, not once per
owner (memory-footprint measurement).

**Depends on:** C7 (elastic runs — shared prefix plus growth is what the server
actually needs); C1's owner table and C6's resolver are the hooks.


#### C8 design (2026-09-21) — the read path, not the refcount, is the cost

**Why.** A prefix used by several sequences (a system prompt on every slot, a conversation
resumed twice) is prefilled and stored once per sequence. B2 reuses a prefix *within* a
slot; across slots there is no reuse at all. `KvCache::owner[cell]` holds a single `SeqId`,
so a cell belongs to exactly one sequence.

**The constraint that shapes everything.** A sequence's rows are one **contiguous run**, and
the read path depends on that: `cells[t] = start + position`, and attention reads a single
`[lo, hi)` span from `attn_span`. The *write* path is already general — the allocator hands
the backend a per-token `cells` vector, so a store may target any row (C6) — but a **read**
cannot follow a sequence whose rows are not contiguous. Now let two sequences share a prefix
and then diverge: only one of them can keep a contiguous layout (its tail sits right after
the prefix); every other sharer's rows become `[0, p) ∪ [private, ...)` — two ranges. So true
sharing is not "a refcount in the cell store"; it is a **paged read path** (a per-sequence
block map and a gather in the attention kernels of each backend). That is the whole cost of
this ticket, and it is why it was estimated L rather than M.

**Two increments, because the cheap half is independent of the read path.**

| Step | Content | Benefit | Cost |
|---|---|---|---|
| **C8a** | *Shared prefill, duplicated rows*: a slot that matches another slot's prefix copies its K/V rows with `copy_cells` instead of re-running the forward, then diverges privately. | Removes the N× **prefill** for a shared system prompt — the latency that is actually paid on the first turn. No IR change, no read-path work, works on every backend that already copies cells (CPU, CUDA; Metal behind G5). | Memory stays N×: each slot holds its own copy. |
| **C8b** | *True sharing*: `owner[cell]` becomes a refcount; the allocator resolves a per-sequence **block map** (64-cell blocks) into the per-token `cells` vector the write path already accepts; attention gains a block gather (`supports_attn_span()` is replaced by a block-map capability), CPU first, then CUDA; Metal stays behind its gate (G5). | The blocks are stored once, so memory follows the sharing, on top of C8a's prefill win. | Every kernel that reads a window needs the gather; `attn_span` is replaced (or supplemented) by the map; compaction and the arena counters must understand refcounts. |

**Invariants / gates.**
1. C8a: a prefix copied from another slot produces continuations **byte-identical** to the
   same prefix re-prefilled, on CPU and on CUDA (the same named-tolerance caveat as any
   device comparison).
2. C8a: the copy must be measurably cheaper than the prefill it replaces (it is the point).
3. C8b: two sharers are byte-identical to the same two sequences with private copies.
4. C8b: `kv_rm` frees a block only when its last owner drops it; a store into a shared block
   copies it (copy-on-write) or fails loudly — never writes through.
5. C8b: `KvArenaStats` counts a shared block once, and a compaction moves it once (not once
   per owner); the memory saving is measured.

**C8a increment 1 (2026-09-21) — the copy primitive.** `GraphAllocator::kv_copy_prefix(src, dst, rows)`
copies the first `rows` written K/V rows of one sequence's run into another's (per layer,
through `Backend::copy_cells`, which handles either direction — C7b) and hands `dst` their
ownership via `own_range`. It refuses a missing run, a source with fewer written rows than
requested, and a destination that reserved fewer cells than that, and it is a no-op for zero
rows or a same-run copy. Unit coverage is the region-free half of those refusals; the data
path and the written/capacity bounds are covered by the S3 gate, because they need real KV
regions and the honest test for a copy is "the answer does not change".

The engine is **not** wired to it yet. S2 does that at admission: placement already prefers
the slot with the best-matching prefix, but only among *idle* slots and only for the slot's
own rows — what is missing is that the donor may be **another** slot (possibly busy, and its
rows are stable while it generates).

**C8a increment 2 (2026-09-21) — admission uses it.** The reuse source is now every slot, not
just the admitting slot's own rows: when another slot holds the longer match, its written rows
are copied into this slot's run (the donor may be *busy* — its rows are stable for the duration
of the copy) and only the suffix is prefilled. The reuse is clamped to leave one token to feed,
exactly as the slot's own reuse is, because a forward with nothing to run produces no logits.
A failed copy falls back to the slot's own cache and prefills the rest — never a wrong-row read.

The test-only counter `prefix_rows_copied` makes the copy observable, so the gate asserts both
that it happened and that it changed nothing: the same prompt served via a copy and on a private
single-slot run answers **byte-identically** (CPU, where the comparison is exact; a device
comparison takes the usual named-tolerance caveat). **C8a cost gate (2026-09-21, GB10).** With a 1450-token prompt on `--n-slots 2 --n-ctx 8192` and
the donor held **busy** (a second request generating), the same prompt served through the copy
costs **0.013 s** against **0.467 s** with prefix reuse switched off
(`MINFER_NO_PREFIX_REUSE=1`) — **36.6×** — and both answers are identical. Two scenario traps had
to be fixed before the number meant anything, and both are easy to repeat: (a) if the sharing
slot is *idle*, the engine simply places the request **there** and reuses its own cache (B2), so
no copy happens at all — the donor has to be busy; (b) if the donor's own request needs more than
its share of the arena, C7's growth reclaims the idle slot the measurement needs, leaving one slot
and self-defeating the scenario. Each copy is reported for an operator as
`[server] slot N: copied R prefix row(s) from slot M`.

**Order.** C8a first: it delivers the user-visible half with the machinery that already
exists (C3's row copy + C7's moves) and its gate is byte-equality. C8b is then a read-path
project, and it can be scheduled on its own evidence rather than on this ticket's estimate.

**Not in this ticket:** C4 (quantized KV) and C5 (state save/restore) — each multiplies the
addressing surface this ticket touches, which is why C6 preceded C7 for the same reason.


#### C8b design (2026-09-21) — a block map for the read path

**What sharing needs that C8a does not.** C8a duplicated the rows so the destination could keep
one contiguous run. Sharing for real means a block exists once, several sequences refer to it,
and a sequence's rows become a **list of spans** rather than one run
(`cells[t] = span_of(t).start + offset`). The write path needs nothing new — since C6 it takes an
arbitrary per-token `cells` vector. The **read path is the whole change**: `attn_span` hands each
query a single `[lo, hi)` range, while a sharing sequence's window is a set of ranges.

**Data.** `KvCache` gains (a) a per-cell **refcount** at block granularity (64 cells — small
integers, and the unit that compaction moves and the stats count), and (b) a per-sequence **span
list** covering `[0, written)`. A sequence that shares nothing has a one-entry span list, which is
the property that keeps this incremental: **its code path stays today's, bitwise.**

**IR — additive, never a replacement.** A graph whose sequences share blocks gets a `kv_map` input
(per query, the sequence's spans) alongside `attn_span`; attention uses the map when the input is
present and the span otherwise. `attn_span`'s single range is load-bearing — the `CAUSAL` fast path
and Metal's refusal (G5) both rest on it — so it is not generalized in place.

**Allocator.** `kv_cells_for_seq` and `attn_span` resolve through the span list. A store that would
land in a shared block takes a **private row for that token** (`kv_private_row_for(seq, t)`): the
allocator decides *before* the forward, so no backend learns that sharing exists and a shared block
is never written through. That is the copy-on-write rule, and it is why the store path did not have
to change.

**Backends.** CPU attention gains the span-list gather first (it is the reference), then CUDA's
`gqa_attn_*` inner loop, whose KV walk becomes (block, offset); Metal stays behind G5 and refuses
the map, as it already refuses the explicit span.

**Why S2 does not split the way S1 did (2026-09-21, found while sizing it).** S1 split cleanly
because its two consumers already existed — the write resolver and `attn_span` — so each half was a
live path with a bitwise gate. S2's halves are coupled instead. Sharing means one cell has several
owners, and the store's ownership representation is *per cell*: `Layer::owner` is a `Vec<SeqId>`,
`written_rows` scans it contiguously for `owner[cell] == seq`, and `attn_span`'s written check
compares it against the query's sequence. A shared prefix cannot be expressed without changing what
"written by this sequence" means (block refcounts plus the span list as the read-side authority),
and until the read path gathers over several spans a sharing sequence would resolve a **wrong**
window. So the store half and the read half have to land together: staging them separately would
add machinery nothing reads, which is the shape A7 deleted. Concretely S2 = block refcounts (64
cells), `kv_seq_cp`, refcount-aware release/`kv_rm`, and a compaction that moves a shared block once
and renumbers every sharer's span list — **plus** the additive `kv_map` input, the CPU gather, and
admission wired to `kv_seq_cp`, accepted by gate 2 below.

**Gates.**
1. S1 changed nothing observable (S1a + S1b landed: both resolvers read the list, and the suites
   stayed bitwise); the refcounts arrive in S2 together with the readers that justify them.
2. Two sharers answer byte-identically to the same two sequences with private copies (CPU; CUDA
   takes the usual named tolerance).
3. `kv_rm` frees a block only when its last owner drops it, and a store into a shared block either
   takes a private row or fails loudly — never writes through.
4. `KvArenaStats` counts a shared block once; a compaction moves it once and renumbers **every**
   sharer's span list; the memory saving is measured.

**Why S1 is a *used* generalization, not dead bookkeeping.** The first draft of this plan put the
refcounts in S1 and the span list in S2. That is the shape A7 deleted: the campaign's own note says
an identity field earns its place only if some *topology* decision reads it — "a future feature will
need it" is not enough, and `n_seqs` was removed for exactly that reason. Refcounts are only read by
the sharing that S2 introduces, so they move to S2 and S1 becomes the span list **plus the two
resolvers that read it** (`kv_cells_for_seq` and `attn_span`). Those are live paths for every
request, which is what makes S1's gate meaningful: a single-entry span list must reproduce today's
answers bit for bit, and any slip shows up in the existing suites rather than in a field nobody
consults.

**C8b S1a landed (2026-09-21) — the write path resolves through spans.** `KvCache` carries a
per-sequence span list `(position base, first cell, length)`, and `GraphAllocator::kv_cells_for_seq`
resolves every store row through `KvCache::cell_of` instead of `slot.start + position`. Every path
that changes a run's `start`/`cap` or drops the run republishes the list (`refresh_spans`: reserve,
release, resize, and each relocation `apply_moves` performs), and a position outside the list is a
**loud error**, not a fallback to the contiguous form — which is what makes a missed maintenance
point visible rather than silent. Today the list always holds exactly one entry, so nothing
observable changed: the full suites keep their counts (CPU 218 / CUDA 267, 0 failed) and every
byte-equality gate still passes — the four-slot batched-vs-serial test, the C7 boundary case, and
the C8a prefix copy. The unit test pins the resolution against `start + position`, including a real
compaction move and a release.

S1b **landed 2026-09-21** — the read path (`attn_span`) now resolves through the same list. A
sequence's window is still one contiguous `[lo, hi)` range, which is all the current input layout
can carry, so S1b resolves only the single-span case and **refuses a multi-span sequence loudly**
(a window that is not one range needs S2's `kv_map`). With one span the answer is identical by
construction: `lo` is the span's first cell and `hi` is `min(span end, query cell + 1)`, which for
`rel < length` is exactly the old `start + rel + 1`. The unit tests pin both halves — the window is
compared against the old arithmetic for a run that does **not** start at cell 0 (where a
`cell == position` slip could hide), and a hand-written two-span list proves the refusal. Full
suites after S1b: CPU **221 passed / 0 failed / 7 ignored** (two tests added; the S1a figures of
218 CPU / 267 CUDA were each one under the state they described — that state measures 219 / 268),
CUDA **270 passed / 0 failed / 7 ignored**, and the
byte-equality gates (batched-vs-serial, C7 boundary, C8a prefix copy) unchanged.

**C8b S2 landed 2026-09-21 — sequences share a prefix in place.** A sequence's address space is now
its span list: a `SharedPrefix { cell, rows }` it reads from a donor plus its private run, which holds
positions `[rows, rows + cap)`. `KvCache::share_prefix` establishes that by pointer — no bytes copied —
and refuses loudly what it cannot express: an empty share, an unwritten donor range, a destination that
already shares or has written rows of its own, itself, and a donor prefix that is not one contiguous
cell range (the caller then falls back to C8a's copy). `KvCache::attn_map` resolves a query's window as
`KV_MAP_MAX_SPANS` `(cell, len)` runs, and `CParams.kv_map` selects that layout for the window input
instead of `attn_span`'s single range: an input's size is topology, so the difference is fixed at build
time, and a size that matches neither form is refused rather than guessed. The CPU kernel gathers the
runs (`decode_window` + `cpu_gqa_attn_runs`, with `cpu_gqa_attn` kept as the one-range wrapper); CUDA's
windowed arm refuses a window that is not `2 * nt` (S4 ports the gather), and both models ask for a map
only on CPU. Admission uses it: at C8a's reuse site `kv_share_prefix` replaces `kv_copy_prefix` where
the device can gather, so the arena holds one copy of those bytes, and every other device keeps the
copy. **Gate 2 holds**: sharing answers byte-identically to a private run, on the same gate that
validated the copy (`a_prefix_copied_from_another_slot_answers_identically`), and
`server_batch_matches_serial_and_is_faster` still passes with the sharing path live. Suites: CPU 228 /
CUDA 277, 0 failed.

**Two deliberate departures from the design above.** (1) **No block refcounts.** Occupancy
(`reserve_seq`, `free_runs`) and the written count are derived from the span lists: a released donor's
rows stay taken for as long as a sharer's spans name them, they come back when the sharer drops them,
and a compaction moves each run's own rows and renumbers every sharer's pointer. A per-block refcount
would be a derived cache of exactly that union and could drift from it — the shape A7 deleted — and
the four gates it was meant to serve hold without it. (2) **The layout rides the window input's size**
rather than a new op field. A `kv_map` flag on `Op::Attn` would have touched every construction site
across three backends, the models and the tests for no behavioural gain; the size *is* the topology
here, and every backend that cannot gather refuses loudly instead of misreading pairs.

S3 then adds copy-on-write (`kv_private_row_for`) for a store that would land in a shared block; S4
ports the gather to CUDA's `gqa_attn_*` loop; S5 is Metal behind G5.

**C8b S3 landed (2026-09-22) — a store inside a shared prefix copies the row first.**
`KvCache::private_row_for(seq, t)` plans a **copy-on-write** and `apply_private_row` books it, with
`GraphAllocator::kv_private_row_for` driving the data half through `Backend::copy_cells`; both fill entry
points (`fill_batch_inputs`, `fill_attn_inputs`) run it **before the first cell is resolved**, and the
store resolver (`kv_cells_for_seq`) now refuses a position inside the share outright. A sharing
sequence's run holds positions `[shared.rows, shared.rows + cap)`, so a store at `t < shared.rows` gives
the share up from `t` on: `shared.rows` drops to `t` and the rows the sequence already wrote shift **up**
by `d = old_base - t` inside the same run. That arithmetic is what keeps the change small — the span list
stays at **two entries** (the remaining share plus the run), the owner table mirrors the move with one
`copy_within`, and `written` is unchanged: the sequence's readable positions are still `[0, written)`,
only `private_written` grows by `d`. `KvArenaStats` gains `cows`/`cow_cells` so a gate can *see* the
mechanism run, and `GraphAllocator::kv_cell_of` is the read-side twin of the store resolver (a caller
snapshotting a sharing sequence's rows cannot use the store one, which refuses those positions by
design).

**Why the run is rebased rather than given a fresh row per token.** The design's `kv_private_row_for
(seq, t)` says "a private row for that token", and the honest way to give it one is to move the run's
base, not to hand out unrelated cells: a per-token fresh cell would put `d` one-length spans in the span
list for a `d`-token divergence, while `kv_map` carries `KV_MAP_MAX_SPANS = 4` runs — so any real
divergence would fail the read path — and a *second* run per sequence is a data-model change (`SeqSlot`
is one run) that every relocation path would have to learn. Growing the run **downward** instead of
shifting (`[start - d, …)`) needs `d` free cells immediately below it, and C7's compaction packs free
space *upward*, so that placement is usually blocked. The shift needs no arena space at all, only
`cap - private_written >= d`, and `copy_cells` is overlap-safe (C7b), so it is one move inside one run.

**The room is not assumed.** The shift needs `cap >= written - t`, and the server's sizing guarantees it:
a run is sized for the whole previous request (`prompt + max_tokens`, and generation stops at
`max_tokens`), so it covers the shared rows as well as the private ones — `cap >= rows + w >= written - t`.
A hand-sized run without that room gets a **loud `Err`** (gate 3's second arm), never a write-through.

**Why the pre-pass is its own loop.** The shift renumbers the run's cells, so a cell resolved before it
would be stale; the resolver's refusal is what makes a missed maintenance point visible instead of
silent. After the copy the share is `[0, t)` — still two spans, or one when `t = 0` and the share
disappears — so `CParams.kv_map` is unaffected: the builder saw a share at build time, and `attn_map`
handles one entry as readily as two.

**Gate 3 holds.** Always-run coverage: three `KvCache` unit tests (the plan; its refusals — a run with no
room, and a plan applied to a state it was not made from; the owner table and the spans; and a sequence
that copies **twice** as it diverges earlier each time, ending with the share gone) plus one allocator
test on real regions that refuses a shared position through the store resolver, drives the copy, checks
K/V byte-for-byte at the moved cells, checks the donor's four rows are unchanged, and drives a second
copy-on-write through `fill_attn_inputs`. The real-model gate
(`a_store_inside_a_shared_prefix_takes_a_private_row`, ignored like the others) shares 16 rows between
two slots and then serves slot 1 a prompt matching only their first three tokens: the request is served
(the resolver would otherwise refuse it), `cows > 0` proves the copy ran, the answer is byte-identical to
the same prompt on an engine with nothing to share, and the **donor's** rows — 17 positions × K/V × 24
layers — are byte-identical after it. That last check is the one with teeth: disabling the pre-pass *and*
the resolver's guard makes the store write through, and the gate fails on exactly that comparison
("the diverging request wrote through the shared prefix"). Gate 3's `kv_rm` half (`occupied()` derived
from the span lists) landed with S2. Suites: CPU 233 / CUDA 282, 0 failed.

**C8b S5 landed (2026-09-22) — Metal refuses both window layouts, and C8b closes.**
Metal derives every query's window from `positions` (the pre-E1 form) and has no
cell-store read path, so **both** explicit layouts — `attn_span`'s one `[lo, hi)`
pair per query and `kv_map`'s `(cell, len)` runs — are refused. Two layers say so:
`Backend::supports_attn_span` (the trait default, now an explicit override on Metal)
keeps such a node off the backend at assignment time, and the `Op::Attn` arm of
`execute_node` returns a loud `Err` if one ever arrives — the alternative would be
computing a *causal* window from `positions` and attending to the wrong rows, which is
the silent-wrong this project refuses. Nothing else changes for Metal: the models ask
for a map only where `Device::gathers_attn_map()` says a kernel reads one (CPU, CUDA),
so on a Mac the server keeps C8a's copy path — and since Metal's `copy_cells` is also
refused until G5, admission logs the failed copy and prefills, which is the same loud
fallback it has always taken. S5's gate is the **compile check** (Metal is
`cfg(target_os = "macos")`, so CI's `build-macos` is the only compile this box cannot
do) plus this record; the device claims stay [#44](https://github.com/yusiwen/minfer/issues/44)'s, on a Mac.

That closes C8b: S1a/S1b (the span list and its two resolvers), S2 (sharing in place +
the CPU gather), S3 (copy-on-write), S4 (the CUDA gather in every kernel) and S5
(Metal's refusal). The ticket C8 itself is **closed on CPU and CUDA** — a prefix is
stored once and shared, and a store into it never writes through — with Metal's share
path moving to **G5**, where the cell store is ported.

**C8b S4 landed (2026-09-22) — every CUDA attention path gathers the map.**
`attn_span`'s single range became three *window modes* selected by the **size** of
the node's window input (C8b S2's departure 2): `positions` (causal), one `[lo, hi)`
pair per query (`attn_span`), or `KV_MAP_MAX_SPANS` `(cell, len)` runs per query
(`kv_map`). Each mode is a separate template instantiation — `MAP` joins `CAUSAL` as
a compile-time flag for the same reason (E1b: the causal kernels must compile to
exactly their pre-E1 instructions) — and a size that matches neither is refused
rather than mis-strided. The row walk resolves a *linear window index* through the
run list (`kv_cell`), and the key insight that keeps every kernel's mask valid is
that a map window is a **prefix** of the sequence's address space (every run before
the query's own, plus its own row), so the existing `index < limit` form stays exact
with `limit` = the run total. Covered: the split-K 1-warp decode body (the sharing
slot's decode step), the batched split path (`1 < nt <= 16`), the 4-warp hybrid, the
legacy per-(token, head) kernels (f16 and f32 KV), and **FA prefill**, whose staging
loop now resolves each linear index through the runs of the tile's widest query (its
list is a superset of the others' — the one-sequence-per-tile precondition the span
path already relied on for its tile-wide min/max).

**The first A/B found a real cost, and the fix was to stop resolving per row.** The
map window measured **1.513x** the span at the 7B decode shape (nkv = 2048, nh = 28,
nk = 4, hd = 128: 24.6 vs 37.2 µs/launch) — the run walk ran per staged row. A map's
runs are long, so a four-row batch almost always sits inside one: resolving the
batch's first row once (plus how many rows its run still holds) and adding for the
rest brought it to **24.6 µs — 1.001x** the span, with a straddling batch still
walking. The same measurement on the **prefill** path (nt = 512, hd = 128, f16 KV)
is **1.726 → 1.872 ms (1.1x)** now that FA gathers too; before it, FA had to be
skipped for a map, which the A/B priced at **170x** (301 ms of legacy per-token
attention) — that number is why the FA port is part of this ticket rather than a
follow-up.

**Admission switches on one authority.** `Device::gathers_attn_map()` (CPU, CUDA) is
read by both the models' graph builders (which ask for the `kv_map` input) and the
server's admission (which shares a prefix in place instead of copying it), so the
share and the window layout cannot disagree; `MINFER_NO_KV_SHARE=1` forces the copy
as the A/B gate for the share itself.

**Gate 2 holds, bitwise.** Always-run: `cuda_map_window_matches_the_span_over_the_same_rows`
compares a map window against the span over the *same bytes* for two KV dtypes across
one/two/three runs, at both prefill- and **decode-shaped** batches (a single token at
the window's end — the case a one-token-at-position-0 draft of this test could not
reach, and the real-model gate caught it missing) — 72 cases, all bitwise. On GB10,
the three real-model gates (C8a's copy, E2's batched-vs-serial, and S3's
copy-on-write) pass with sharing live, on an **f32-KV** 0.5B and an **f16-KV,
hd = 128** Qwen3-0.6B — the two combinations that matter, since FA prefill and the
half-width cell stride only exist on the second.

**Two pre-existing bugs this ticket's gate found, both fixed here.** (1) The server's
decode position was **off by one**: `advance` incremented `current_pos` before the
forward that writes the row, so the first token after a prefill was stored at
`nt + 1` and position `nt` was never written — the next step's attention read a stale
row whose content came from the arena's history, which made answers depend on
allocation (the gate's two identical share runs disagreed). The increment now happens
where the row exists (`tick`, next to the mirror's push), with the assertion
corrected to match. (2) **f16 KV cell moves strode by f32 elements**: `copy_cells`'s
`elems_per_cell` is a count of f32, while an f16 cache stores a row as `nkt` halves
(`nkt / 2` f32), so every moved row walked twice as far and landed in the wrong cell
— a compaction *or* a copy-on-write on an f16 device silently corrupted the arena.
`CudaBackend::copy_cells` now halves the stride for f16, pinned by
`cuda_f16_kv_cell_move_strides_by_row_bytes` (always-run, and it fails without the
fix). Neither bug could be seen by the existing gates: both compare server-against-
server, and every CUDA gate ran on an f32-KV model.

Suites: CPU 233 / CUDA 284, 0 failed, 9 ignored.

S5 is Metal behind G5.

**Risks.** (1) The attention inner loop changes on every backend — correctness *and* timing, so each
backend gets its own A/B. (2) The map must stay additive, or the `CAUSAL` path and Metal regress.
(3) CoW's per-token private rows must not fragment the arena beyond what C7's growth can absorb;
the stats gate watches exactly that.

| Step | Content | Gate |
|---|---|---|
| S1 | Per-sequence **span list** + the resolvers (`kv_cells_for_seq`, `attn_span`) reading through it — single-entry in practice, so nothing observable changes | no behaviour change: suites stay bitwise |
| S2 | Block-granular **refcounts** + `kv_seq_cp` (share a prefix) + the CPU attention gather | gate 2 on CPU |
| S3 | Copy-on-write: `kv_private_row_for` at store resolution — **landed 2026-09-22**: the share shrinks to the first position this forward writes and the run's own rows shift up inside it (two spans, no arena space needed), the store resolver refuses a shared position, and a run without room is a loud `Err` | gate 3 on CPU + the real-model donor-byte check |
| S4 | CUDA attention gather + device A/B — **landed 2026-09-22**: three window modes (causal / span / map) selected by the window input's size, the `MAP` flag threaded through every attention kernel including FA prefill, and the batch-level row resolution that makes the gather free | gate 2 on GB10 (bitwise, f32 *and* f16 KV) + the decode/prefill A/B |
| S5 | Metal behind G5 — **landed 2026-09-22**: both explicit window layouts refused (`supports_attn_span` keeps them off the backend; the `Op::Attn` arm backstops with a loud `Err`), admission keeps C8a's copy, docs closed | compile check (CI `build-macos`) + G5 record |


## 6. Phase D — IR expressiveness (item 7)

| ID | Title | Effort |
|---|---|---|
| D1 | Strided views with allocator-known aliasing + multi-output nodes — **DONE (2026-09-19), increments 1–3**: exact views are zero-copy and offset/partial windows work on CPU and CUDA (`BufRef` carries offset+len through the `Backend` trait; Metal is exact-only), and the "multi-output node" is `GraphBuilder::split_parts` — one owning node plus one `Op::View` per part, so the graph stays single-output and no backend changed. The design's sketched `Op::SplitParts` was **not** added: a split op over one input is just views of that input. MoE/MLA remain model work (producer ops, weights, routing) | L |
| D2 | Re-express one decode fusion as a composition (proof) — **DONE (2026-09-19)**: `FusedFFN` as concat `MatMul` + two partial windows + in-place `SwiGLU`; CUDA takes the composition, Metal keeps the node until G5; env gate `MINFER_FFN_NODE=1` for the A/B; three models byte-identical, decode cost **≤0.8%** (recorded) | M |
| D3 | Decide the fate of the four hand-written fused ops — **DONE (2026-09-19)**: `FusedFFN` keeps the default (the proven composition is 0.3–0.8% slower on CUDA and Metal needs the node until G5); the QKV family keeps (no composition proof); `QkvBiasRopeStore` is a fallback epilogue, not a redundancy. Policy rule + gate clarity recorded | S |

- **D1 acceptance:** `Op::View` becomes zero-copy (a test asserts the allocator
  maps a view onto its parent's buffer); the allocator's liveness understands
  `view_src`; existing graphs unchanged and bitwise.

#### D1 design (written before the code, 2026-09-19)

**What D1 is actually for.** C3's own text allows either route ("a copy needs
either a view or an explicit copy op"), so views are **not** on C3's critical
path; what needs them is **D2** (a decode fusion re-expressed as a composition:
`FusedQKV` leaves `q|k|v` in one concat buffer, and the composition feeds
attention from three *windows* of it), and what needs multi-output nodes is
**MoE/MLA** (item 17). The diagram's "C3 needs D1" is therefore softer than it
looks; the honest dependency is D2 → D1.

**Today (measured in the code, not assumed).** `Op::View`/`Reshape`/`Permute` are
**copies**: the CPU arm is `out.copy_from_slice(ins[0])`, i.e. the IR has no
aliasing at all. The one alias that exists is the *in-place elementwise* rule in
`GraphAllocator::alloc_graph` (Silu/RoPE): when the input's sole consumer is the
op and both are on the same backend, the op's output **is** the input's buffer
(`node_to_buf[op] = node_to_buf[src]`) and the input's `last_use` is extended past
it. `BufRef` is `{ backend, id }` — no offset — and the `Backend` trait hands the
backends plain ids (`in_bufs: &[usize]`, `out_buf: usize`). An *offset* view
therefore cannot be expressed without threading offsets through the trait, all
three backends and their call sites (the A5 lesson: a trait-signature change
silently skips the CUDA test call sites unless CI compiles them).

**Three increments.**

1. **Exact views — landed here.** `CNode` gains `view: Option<ViewAlias>` where
   `ViewAlias { src: NodeId, offset: usize }` (the field carries `offset` from the
   start so the IR shape does not change again later), the builder marks
   `View`/`Reshape`/`Permute` as views of their single source, the allocator maps
   `node_to_buf[view] = node_to_buf[parent]` and extends the parent's liveness
   through the `view_src` chain, and the view kernels become no-ops (the alias
   *is* the output). The backend trait is untouched: an exact view is the same
   buffer id. Loud refusals: `offset != 0` in this increment, a missing parent, a
   parent on another backend, or a window that would not fit inside the parent.
2. **Offset views.** `BufRef` gains `offset: usize`, the trait passes `BufRef`,
   and each backend adds the offset where it resolves an id (CPU's
   `split_at_mut` plumbing, CUDA/Metal's `ptr_of`). This is what D2 needs.
3. **Multi-output nodes** (`CNode` grows a list of outputs), which is what MoE
   and MLA need.

**Acceptance for increment 1** (the D1 acceptance above, narrowed to what is
landed): a view is zero-copy (a unit test asserts the allocator maps it onto the
parent's buffer id), liveness understands `view_src` (the parent is not recycled
while the view is live, and is freed after), and existing graphs are unchanged —
the real-model bitwise tests and the op matrix are the evidence.

**Increment 2 record (2026-09-19).** `BufRef` now carries `offset` and `len`
(the window, not just a buffer id), the `Backend` trait passes references instead
of raw ids, and the allocator maps a view to `parent.window(offset, len)` after
checking the window fits inside the parent:

- Every backend applies the offset where it resolves a buffer: CPU slices each
  input/output window out of the pool (the `split_at_mut` plumbing is otherwise
  unchanged), CUDA adds `offset * 4` bytes in a new `ptr_of_ref` and takes element
  counts from the window (`in_bufs[k].len`, which also corrected `copy_d2d`'s size
  check), and the host paths are window-aware — `copy_to_cpu` returns exactly the
  window.
- Metal is **exact views only**, and that is a capability boundary rather than an
  oversight: its kernels take a buffer and a length with no element offset, so a
  window would silently read the wrong bytes. `supports_op(Op::View{offset})` is
  `offset == 0` there, the allocator backstops the partial-window case (which
  `supports_op` cannot see, since the parent's length is not in the op), and
  `SUPPORT-MATRIX.md` gains the asymmetric row; G5 is where Metal would learn
  offsets. That path is compile-verified by CI's macOS job alone, and the job
  earned its keep on this change: it caught a `BufRef` passed to a Metal debug
  `Display` format, which no Linux build can see.
- Evidence: a new op-matrix case, `View offset` (a partial window at offset 2 over
  an 8-element parent, expected `[3,4,5,6]`), passes on **CPU and CUDA** and is
  skipped on Metal; two allocator tests pin the mapping
  (`(id, offset, len) == (parent, 2, 4)`) and the new refusal boundary (a window
  past the parent); the device-gated CUDA test call sites were re-pointed through
  a test-only `exec_ids` shim that derives each reference's length from the pool
  (the A5 lesson: a trait change skips them unless the test harness is compiled);
  and the real-model bitwise gates are unchanged.

**Increment 1 record (2026-09-19).** Landed as designed:

- `CNode.view: Option<ViewAlias>` (with `offset` present from the start), set
  automatically for `View`/`Reshape`/`Permute` in `GraphBuilder::node` — the one
  construction point, so hand-built graphs and the op matrix get it too.
- `GraphAllocator::alloc_graph` maps such a node onto its parent's buffer and
  extends the parent's liveness through the `view_src` chain
  (`extend_through_views`), which is also now used by the in-place branch (an
  in-place op on a view writes through to the buffer it windows).
- The three view kernels became **no-ops** on all backends, with the aliasing
  asserted instead of assumed: CPU `debug_assert_eq!` on the pointers, CUDA/Metal
  return `Err` rather than copying if the output is not the source's buffer
  (standing rule 2 — the previous arms performed a silent identity copy).
- Refusals, all loud: `offset != 0` (increment 2), a partial window (its element
  count differs from the parent's, which would make `copy_to_cpu` read the whole
  parent), a cross-backend view, a missing parent buffer.
- Evidence: three new tests in `graph::alloc::tests`
  (`a_view_aliases_its_parent_buffer_and_keeps_it_alive`,
  `views_that_need_more_than_exact_aliasing_are_refused`,
  `in_place_on_a_view_extends_the_parents_liveness`); the op matrix's
  View/Reshape/Permute cells now exercise the alias on CPU **and** CUDA; and the
  real-model bitwise gates (B2, C2, prefix reuse) are unchanged: CPU 195 passed /
  0 failed, `--features cuda` 241 / 0. No architecture emits these ops yet, so
  the payoff is D2's: it is the machinery a `FusedQKV`-as-composition needs, one
  increment short of the offsets it will actually use.

**Risks.** Aliasing changes who may write whose bytes: a view's consumer can now
write into the parent's buffer (in-place ops on a view), so the existing
"sole consumer + same backend" rule and the transitive liveness extension are both
load-bearing, and chains (view of a view, in-place on a view) need their own
tests. A backend that cannot express an alias must refuse loudly rather than
copy silently (standing rule 2). And `copy_to_cpu`/`fill_input` on a view now read
the parent's bytes — which is the point, but it is also why the op matrix's
View/Reshape/Permute cells are the regression net for it.
- **D2 acceptance:** `FusedFFN` re-expressed as `MatMul` + views + in-place
  `SwiGLU`; the hand-written node stays behind an env gate for A/B; bitwise
  identity; no decode regression beyond a recorded budget.

#### D1 increment 3 design (written before the code, 2026-09-19) — multi-part outputs via views

The ticket row above says "multi-output nodes". Before giving `CNode` a second
output — which would touch the allocator, the scheduler, all three backends and the
JSON/DOT/trace exporters, for a capability only MoE would exercise — consider what
D1 increments 1–2 already bought: an output can be an **offset view** of another
buffer (`BufRef { offset, len }`, `CNode::view`) and liveness follows views
(`extend_through_views`). A node that produces several tensors therefore needs one
output *buffer*, not several outputs: its parts are views over it, and consumers
bind to those views exactly as they bind to any other producer's output.

**Landing correction (2026-09-19, before the code): no new op is needed, and none
was added.** The sketch above proposed `Op::SplitParts`, but a *split* op taking
one input is either an identity copy or — what it really is — a set of windows over
its input, which `Op::View` already expresses. What the increment needs is the
**helper**: `GraphBuilder::split_parts(owner, sizes) -> Vec<NodeId>` emits one
`Op::View` per part (cumulative offsets, the owner's trailing dims, the owner's
dtype), so:

- a node whose one kernel writes several logical tensors keeps writing **one**
  output buffer and the parts are ordinary graph tensors — the graph stays
  single-output, no backend grows a second-output path, and the parts inherit the
  owner's liveness through the existing `extend_through_views`;
- a `(values, indices)` pair needs no second output *dtype*: an indices part is
  i32 in f32 bit patterns (rule 4), which is what lets it drive `Op::GetRows`;
- and the shape is already in production — D2's `fused_ffn_composition` is a concat
  matmul whose gate/up halves are consumed through two such windows.

The property that makes this the right shape: the graph stays **single-output**, so
no allocator, scheduler or backend grows a second-output path that only MoE will
ever use, and the parts are ordinary graph tensors (so `MINFER_TRACE`, the JSON
export and DOT already show them).

#### D1 increment 3 record (2026-09-19)

Landed as `GraphBuilder::split_parts` (builder helper: no new `Op`, no kernel, no
backend change — `Op::View`'s arm is a debug assertion because the allocator maps
a part onto the owner's buffer). Two tests, both CPU-runnable:

- `split_parts_maps_every_part_onto_the_owners_buffer` — structural: a real producer
  (a concat matmul) split into `[3, 3]`; each part's `BufRef` is
  `(owner.id, offset, len)` with cumulative offsets, and the record is a view
  (`CNode::view`), which is what extends the owner's liveness.
- `split_parts_parts_feed_independent_consumers` — functional, in the shape MoE
  routing needs: one owner buffer holding four value rows plus two indices, split
  into `[8, 2]`, with the **indices part driving `Op::GetRows`** over the values
  part. The gathered rows are asserted against the Rust-side expectation, so the
  mechanism is proven end to end (one owner, two parts, two consumers, no copy).

What this closes and what it does not: the **IR** blocker is gone — a producer can
feed several consumers with independently bindable tensors without a second output
— and it is closed without touching the allocator, the scheduler or any backend.
MoE and MLA still need their own work (expert weights, routing, the grouped GEMM;
latent projections, per-head latents), which is model work rather than IR work, and
is recorded as such instead of implied to be unblocked.

What increment 3 does **not** do — correcting "D1 unblocks MoE/MLA" a second time,
after D1's own design already scaled it back: MoE still needs its architecture work
(expert weights, routing, the grouped GEMM) and MLA needs its latent projections.
Increment 3 removes the *IR* blocker and proves the mechanism with a kernel any
model can use; it is not a MoE implementation.

#### D2 design (written before the code, 2026-09-19)

**What `FusedFFN` is today (read from the code, not remembered).** The model
builders emit it only for a GPU decode step: `nt == 1 && cparams.gpu &&
cparams.fuse_ffn && gu_concat_available(...) && nf <= 16384`
(`models/qwen2/graph.rs:243`, qwen3 likewise). Its output shape is
`[2 * nf, nt]` — a *concatenated* gate|up buffer — and its consumer (the `down`
matmul) reads **rows `0..nf`**, where the fused kernel leaves `silu(gate) * up`.
CPU never sees the op: `cpu_backend` returns `Err` for `Op::FusedFFN` ("fusion
not enabled for it"), and the GPU gate is what keeps it off the CPU path.

**The composition, and why it is bitwise-comparable.** With D1 increment 2 in
place, the same graph is expressible as:

```
MatMul(x, gu_concat_weight) -> concat [2*nf, nt]
View { offset: 0,  shape: [nf, nt] } -> gate      (a partial window of concat)
View { offset: nf, shape: [nf, nt] } -> up        (the second window)
SwiGLU(gate, up) -> out                            (in place, into the gate window)
```

`out` occupies the **same bytes** the fused node leaves its result in (rows
`0..nf` of the concat), so the `down` matmul and everything downstream are
unchanged and the two paths are comparable element by element. That is the whole
point of doing D2 *after* D1: before offset views, the gate/up windows could only
be copies.

**What must change.**

1. A builder path that emits the composition (`fused_ffn_composition`), with the
   hand-written `Op::FusedFFN` kept behind an env gate for the A/B — the plan's
   wording, and the reason D3 exists: the node is not deleted until the
   composition is *measured*.
2. The allocator's in-place rule must accept `Op::SwiGLU` writing into its first
   input's window. The two inputs are windows of the *same* pool buffer at
   different offsets, which the CPU's aliasing snapshot already handles (it
   clones each aliased input); the rule and a test for the two-input case are what
   is missing.
3. A per-backend capability decision, and it is not uniform:
   - **CPU** never emits FFN fusion (`cparams.gpu` is required), so nothing
     changes there;
   - **CUDA** takes the composition (MatMul and SwiGLU are both long-supported
     ops, views landed in D1 increment 2);
   - **Metal keeps the hand-written node**: the composition's views are *partial*
     windows at a non-zero offset, which Metal's exact-view rule refuses until G5
     (`SUPPORT-MATRIX.md`'s D1 row). Falling back must be a *build-time* choice
     per backend, not a runtime copy.

**Evidence plan (the increment this round did not reach).** Bitwise A/B on the
device with a real model: one binary, the env gate selecting node vs composition,
comparing the token stream and the logits; a decode tokens/s measurement for the
recorded budget (the composition adds one `2*nf` f32 read+write per layer per
token for the SwiGLU pass, and it may pick a different GEMM kernel than the fused
special case — that is exactly what the budget is for); the real-model bitwise
gates and the op matrix unchanged; `cargo test --release --features cuda` green.

**Increment record (2026-09-19) — landed.** `GraphBuilder::fused_ffn_composition`
builds the composition (`matmul_by_name` over the concatenated gate|up weight →
`View{offset: 0}` gate → `View{offset: nf}` up → `Op::SwiGLU`), and:

- the allocator's in-place set accepts `Op::SwiGLU` **only when its first input is
  a view** — the fusion pass's own `SwiGLU` (CPU, non-view) keeps its separate
  buffer, so no existing graph moves;
- the models pick per backend: **CUDA takes the composition**, **Metal keeps the
  hand-written node** (partial/offset windows are refused there until G5 — a
  build-time choice, never a runtime copy), and `MINFER_FFN_NODE=1` forces the
  node anywhere, which is the A/B gate;
- a CI-verifiable test (`ffn_composition_swiglu_aliases_the_gate_window`) pins the
  structure and, with no device, that the `SwiGLU`'s buffer **is** the concat
  buffer at offset 0 — the same bytes the fused node writes.

**Evidence (device, GB10, greedy, equal work).** Same-mode repeat is the
determinism control (identical), and the comparison strips the timing lines:

| Model | Hand-written node | Composition | Generated text |
|---|---|---|---|
| Qwen2.5-0.5B Q4_0, n=64 | 516.3 tok/s | 513.6 tok/s | **byte-identical** |
| Qwen3-0.6B Q8_0, n=32 | 228.9 tok/s | 228.2 tok/s | **byte-identical** |
| Qwen2.5-7B Q4_K_M, n=64 | 199.7 tok/s | 198.2 tok/s | **byte-identical — but vacuous, see below** |

**Correction (2026-09-19, D3 round).** The 7B row is *vacuous*: the fusion gate is
`nf <= 16384` and the 7B's `nf` is 18944, so that model never uses `FusedFFN` and
`MINFER_FFN_NODE=1` selected the same unfused path as the default — the two runs
were identical by construction, not by evidence. The meaningful rows are the 0.5B
(`nf` 4864) and Qwen3-0.6B, both of which do fuse. The 7B's 0.3%-faster reading is
noise, and the recorded budget therefore rests on the 0.5B/Qwen3 numbers (0.3–0.8%
slower). Anything measured on a model above the gate says nothing about D2.

So the acceptance is met: re-expressed as a composition, identical greedy output
(the project's established proxy for bitwise), and a **recorded budget of <=0.8%
decode** (measured 0.3–0.75% slower, consistently — the separate SwiGLU pass costs
one `2*nf` f32 read+write per layer per token, and it is not offset by the fused
epilogue on this device). Suites: CPU 197 / 0, `--features cuda` 243 / 0.

**Method note, because it cost a false alarm.** The first A/B hashed *stdout*,
which includes the `Prefill:`/`Generated:` timing lines that differ every run, and
therefore reported `DIFFER` on all three models. The same-mode repeat caught it:
if a mode does not reproduce itself, the comparison is measuring the harness. Any
future A/B of this kind must strip those lines (or compare token ids) and keep the
repeat control.

**What D3 inherits.** The composition is proven and *slightly* slower on CUDA,
which is exactly the trade D3 decides: keep the hand-written node (and its
device-specific kernels) or delete it for simplicity. It cannot be deleted
outright regardless — Metal needs it until G5.

**Risks.** The budget is the honest risk: if the separate SwiGLU pass costs more
than the fused epilogue saves on this device, D2's outcome is "re-expressed,
measured slower, node stays" — which is a *result*, not a failure, and is why D3
is a decision ticket rather than a deletion. The second risk is the reverse of
D1's: a composition that *works* everywhere invites deleting the hand-written
node, and the CUDA-specific kernels (fused epilogue, MMQ tiling) exist for
reasons the composition cannot express; D3 owns that call.
- **D3:** with D2 proven, decide per fusion whether to keep the hand-written
  node (performance) or delete it (simplicity). MoE (item 17) is unblocked here.

#### D3 record (2026-09-19) — the fate of the four hand-written fusions

**Decisions.**

| Op | Decision | Evidence / why |
|---|---|---|
| `FusedFFN` | **keep as the default**; the composition stays behind `MINFER_FFN_COMPOSITION=1` | D2 measured the composition 0.3–0.8% slower on CUDA on the models that actually fuse (0.5B `nf` 4864, Qwen3-0.6B; the 7B never fuses — `nf` 18944 > the 16384 gate — and its row is recorded as vacuous), and **Metal requires the node until G5**, since a partial window at a non-zero offset is refused there. Deleting it would cost performance on CUDA and break Metal. |
| `FusedQKV` | **keep** — no deletion decided | Both GPU backends support it and it carries the bias+rope+store epilogue; no composition proof exists. A D2-style proof is expressible now (concat `MatMul` → three windows → in-place rope → store via `Op::KvcacheStore`) but unmeasured, so deleting it would be an unmeasured simplification that also has to hold on Metal. |
| `FusedQkvNorm` | **keep**, same as `FusedQKV` | Adds per-head Q/K RMSNorm inside the epilogue (Qwen3); a composition would need a norm on a window — expressible, still unproven. |
| `QkvBiasRopeStore` | **keep — it is not a redundant fusion** | It is the *fallback epilogue* for the mixed-quant case: three separate q/k/v matmuls without bias, plus one combined pass (bias×3 + rope×2 + store×2). Deleting it would slow that path, not simplify anything. |

**The policy rule (reusable).** A hand-written fusion may be deleted only when all
four hold: (1) a composition exists in the builder; (2) it is byte-identical on
**every** backend that emits the fusion, not just the one measured; (3) its measured
cost is within a **recorded budget** on those backends; (4) no backend needs the
node for lack of a composition primitive. Otherwise the node stays and its A/B gate
is documented. `FusedFFN` fails (3) on CUDA and (4) on Metal → kept. The QKV family
fails (1) → kept.

**Gate clarity (the follow-through).** Two classes of environment gate exist and
mean different things:

- `MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` — **disable the fusion** entirely
  (build the plain `MatMul` + rope/store, or gate+up+silu+mul, path). Unchanged.
- `MINFER_FFN_COMPOSITION=1` — keep fusing, but build the fusion as the
  **composition** (`MatMul` + gate/up windows + in-place `SwiGLU`) instead of the
  hand-written node. This is the D2 proof path and the A/B reference; it is refused
  with a warning on a backend without offset views (Metal until G5; CPU, which
  never fuses in the first place).

The choice is a pure function (`models::ffn_composition`) unit-tested across the
matrix, so CI covers it without a GPU — the same shape as E6's `batch_mode`.

**MoE prerequisite, honestly.** Phase D's note says MoE is unblocked here; it is
**not**. MoE and MLA need **multi-output nodes** — D1's third increment, which is
not landed (D1 has delivered exact views and offset/partial windows). Nothing in D3
changes that.

**Evidence run.** With the new default, the node and the composition still produce
byte-identical greedy text on the models that fuse: Qwen2.5-0.5B Q4_K_M (n=32) and
Qwen3-0.6B Q8_0 (n=32), both `IDENTICAL`. Suites: CPU 198 passed / 0 failed / 5
ignored, `--features cuda` 244 / 0 / 5.

## 7. Phase E — batching, then memory policy

| ID | Item | Title | Effort |
|---|---|---|---|
| E1 | 2 | IR `seq_id` + explicit attention masks (CPU) — **DONE** | L |
| E1b | 2 | CUDA attention kernels read `attn_span` — **DONE, device-verified (2026-09-18)** (window test passes on GB10; causal-path timing unchanged) | M |
| E2 | 3 | Batch composition + continuous batching — **mechanism landed; CPU acceptance refuted and accepted, GPU acceptance MET (1.9x)**; opt-in at the time (`MINFER_BATCH=1` — **E6 later made the default device-aware**); A7 closed by **deleting** `n_seqs` — **ticket closed** | XL |
| E3 | 10 | Chunked prefill: make `n_batch` real · [#45](https://github.com/yusiwen/minfer/issues/45) — **DONE (2026-09-22, see the record)** | M |
| E4 | 8 | Allocator reserve/assign split + size classes + memory accounting · [#55](https://github.com/yusiwen/minfer/issues/55) — **DONE (2026-09-23)**: S1 the accounting + the size-class ladder + the feasibility gate; S2 the pools allocate at the class, the length contract moved to `BufRef`, two latent liveness bugs fixed; S3 split reservation (the slot table) from assignment — a rebuild re-maps without touching the pool, CUDA's `pool_gen` stops moving, and `GraphCache` holds one graph per `GraphParams` (a switch re-maps instead of rebuilding) | L |
| E5 | 9 | Layer-offload budget (`n_gpu_layers` equivalent) · [#46](https://github.com/yusiwen/minfer/issues/46) — **DONE (2026-09-23)**: S1 the layer-granular plan (`--gpu-layers`/`MINFER_GPU_LAYERS`), per-block weight registration and placement, the startup report, a verified mixed CPU+CUDA run; S2 the `auto` fit — per-block weight bytes from the GGUF index, the pure `fit_blocks` prefix search against a weight budget (`MINFER_GPU_MEM`, else three quarters of device free), a quarter held back for KV/activations | L |
| E6 | 3 (follow-up) | **Device-aware batching default** — **DONE (2026-09-19)**: `MINFER_BATCH` unset now batches iff the model's forwards run on CUDA, serial otherwise; `=1`/`=0` force it either way; the decision is a pure unit-tested function. Refetched 1.97x on the 7B with **no** environment variable (see the record) | S |

#### E5 record, S2 (2026-09-23) — the offload plan as a fit, not a ceiling

**Why.** S1 shipped the placement machinery and a knob, but the knob was an explicit ceiling: the
user had to know the model's block count and guess how many fit. The ticket's "budget knob" is
the other half — *compute* the split from the device's memory, which is what a machine whose free
memory is smaller than the model actually needs.

**What S2 landed.**

- **`auto`** as a third request (`MINFER_GPU_LAYERS=auto` / `--gpu-layers auto`), parsed strictly
  with the rest (`OffloadRequest::parse`: a decimal count, `auto`, or unset; anything else is a
  refused load).
- **The byte table comes from the index.** `block_weight_bytes` sums `GgufTensorInfo::nbytes` per
  block (`block_of` on the tensor name) *before anything is loaded* — the decision cannot wait for
  a measurement, because the registration filter **is** the plan. `nbytes` is now the shared
  arithmetic (the loader slices the part with the same method), so the fit and the loader cannot
  disagree about a tensor's size.
- **The fit is a pure prefix search** (`fit_blocks(budget, per_block, reserve)`): the largest
  `k` with `reserve + Σ per_block[0..k] ≤ budget`. A prefix, not a knapsack — the plan is
  `0..gpu_layers`, and a gap would put a CPU block between two device blocks for nothing; the walk
  stops at the first block that does not fit, which is a deliberate, documented conservatism.
- **The budget** (`weight_budget`): `MINFER_GPU_MEM=<MiB>` when set, else **three quarters of what
  the device reports free** — the same default E4's feasibility gate uses, so the fit and the gate
  that later checks the activation pool are talking about one number. A quarter of the weight
  budget is then held back as `reserve` for the KV arenas and the activation pool: both are sized
  per graph at forward time, so no load-time fit can measure them; a prompt that needs more is
  refused by the E4 activation gate (loudly) instead of quietly swapping.
- **The report explains the fit** (`auto_source`): `offload: 5 of 24 blocks on cuda … (40.0 MiB of
  device weights; auto: 5 of 24 blocks fit — weights budget 64 MiB, 16 MiB reserved for
  KV/activations; MINFER_GPU_MEM=64 MiB)` — the decision and the numbers behind it.
- `device_free_bytes()` is the one place that asks the device (CUDA: `cudaMemGetInfo`); Metal's
  wrapper reports no free-bytes number yet, so on macOS `auto` needs `MINFER_GPU_MEM` and otherwise
  fits nothing (the default and an explicit count still work there).

**Acceptance, as measured** (0.5B q4_0 on GB10):

- `an_auto_offload_plan_fits_the_budget` (ignored, real model): with `MINFER_GPU_MEM=64` the fit is
  a **strict prefix** — 5 of 24 blocks, 40.0 MiB of device weights registered, i.e. inside the
  48 MiB the budget left for weights — the report names `auto` and the cap, and the model's
  **four greedy steps match the all-CPU run**; with no cap the same request selects **all 24
  blocks**, so an `auto` default cannot silently under-offload a device that fits the model.
- The pure matrix (4 tests, no device): the request spelling (`auto` case-insensitive, garbage
  refused with the alternatives named, `plan()` refusing `auto` so a caller cannot skip the fit),
  the prefix search (exact boundary, reserve eating the budget, empty/zero-size tables, the
  "a big block stops the walk even though smaller ones follow" conservatism), the budget
  (three quarters, explicit cap wins, no device → nothing fits, garbage refused) and the report
  text (device vs cap vs no budget).
- **Mutation-checked**: making `fit_blocks` ignore the budget fails the auto gate on its first
  assertion (`a 64 MiB budget must be a strict prefix, got gpu_layers: 24`), and S1's gate covers
  the placement rule on the same code path (removing it fails the mixed run; forcing
  `allows_weight` true fails the device-bytes assertion).
- Suites: CPU **280 passed / 0 failed / 15 ignored**; CUDA (GB10, serial) **333 / 0 / 16**; the
  `#[ignore]`d set serially **15 passed / 0 failed**.

**Honest scope.** The fit measures **raw tensor bytes**; the auxiliary device copies the loader
builds while loading (the fused `attn_qkv`/`ffn_gu` concats, the padded Q6_K layout, the q8_0 p32
split, the q4_K dsc pair) are not in the per-block table — the quarter held back and E4's loud
activation gate are what keep the estimate honest, and an under-estimate ends as a refusal, never
as a silent overcommit. The reserve is a fixed quarter rather than a function of `n_ctx`/`n_batch`
(which are runtime choices): a long-context request on a tight budget may therefore be refused by
the gate even though `auto` accepted the weights — the message names the numbers, and
`--gpu-layers`/`MINFER_GPU_MEM` are the knobs. Metal has no free-bytes query yet, and the fit is
per **allocator**, like every other budget.

#### E4 record, S3 (2026-09-23) — split reservation from assignment, then hold several graphs

**Why.** S1 and S2 left the ticket's two structural items. The allocator's walk **freed** the
previous graph's buffers and re-allocated them, so even a same-shape rebuild went through the
backend's pool — and CUDA's `pool_gen`, which moves on every allocate/free, is exactly what
invalidates its captured graphs (`graph_replay_step` re-captures when the generation changed).
Separately, `GraphCache` held **one** graph, so a server alternating a 1-wide and an N-wide
decode step rebuilt on every step, and a chunked prefill paid E3's measured "one forward's fixed
overhead per chunk" again on every repeat request.

**What S3 landed.**

- **Reserve, then assign.** A released *classed* buffer no longer goes back to the backend: it
  goes to a reservation table (`slots: HashMap<(Backend, class-elements), BTreeSet<pool-id>>`),
  and `alloc_class_in_pool` takes the **smallest idle id** of the class, allocating a new buffer
  only when the class has nothing idle. `buf_class` marks which allocations are classed; staging
  (exact-sized) keeps the backend's own free list. A rebuild therefore re-maps: the pool is not
  touched, and the same topology gets the same slots. The smallest-id-first order is
  load-bearing and deterministic — liveness releases buffers in `HashMap` order, so the first
  (LIFO) version handed a rebuilt graph *different* ids each time, which the gate caught
  immediately (`left: [(0, 2), (1, 1), (2, 0)] / right: [(0, 0), (1, 1), (2, 2)]`).
- **`GraphCache` holds one graph per `GraphParams`** (MRU first, `MAX_CACHED_GRAPHS = 8`).
  `try_reuse` matches **any** cached graph, makes it current and re-maps the allocator onto it
  (`alloc_graph`: liveness + slot assignment) — no build, no assign pass, no fusion pass.
  `stats()` reports `(builds, reuses)` for the gate; `cached_graphs()` is the depth.
- **Cross-backend staging is per graph**: the map is keyed by `(graph uid, node, backend)` and
  survives a re-map. Node ids restart per graph, so the old `(node, backend)` key would have let
  a switch reuse another graph's buffer; and re-creating the entries per switch leaked, because
  staging is `alloc_fresh` (never recycled) — the CUDA suite found it as `CUDA: OOM allocating 4
  bytes` after ~200 generate steps in the capture-parity gate. Entries of evicted graphs stay
  allocated (a handful of boundary-sized buffers, bounded by the cache's depth).
- The reservation is visible: `MemoryReport::{idle_slots, reserved_classes}`, and
  `CpuBackend::alloc_count` is the CPU twin of CUDA's `pool_gen`.

**Acceptance, as measured.**

- `a_rebuild_remaps_instead_of_reallocating` (CPU): the same graph **and** a neighbour in the
  same class (896x16 / 896x17, both class 16384) re-map — `n_cpu_allocs` and `n_cpu_buffers`
  unchanged, the node → slot mapping identical; a shape that leaves the class does reserve.
- `a_released_buffer_stays_reserved_and_idle`: `pool_bytes`, `live_bytes` and `idle_slots` are
  exactly what they were after a rebuild, and the reservation reports at least one class.
- `a_rebuild_does_not_touch_the_device_pool` (CUDA, GB10): `pool_gen` is unchanged across a
  same-shape rebuild **and** across a same-class different shape, and the mapping is identical —
  i.e. a captured graph survives a rebuild.
- `switching_between_cached_graphs_re_maps_instead_of_rebuilding` (CPU): two shapes built, four
  switches, `stats() == (2, 4)`, zero pool allocations during the switches, and each graph maps
  to the same slots every time it is switched in.
- `a_repeated_chunked_prefill_stops_rebuilding` (real 0.5B, ignored): request 1 → **3 builds /
  5 reuses**; request 2 (same prompt, same chunk size) → **3 / 9**, i.e. the second request
  builds nothing. Before S3 each chunk forward of the second request was a fresh build.
- **Mutation-checked**: returning classed buffers to the backend instead of the reservation
  fails the re-map gate; making `try_reuse` match only the previous graph fails the switch gate.
- Suites: CPU **276 passed / 0 failed / 14 ignored**; CUDA (GB10, serial) **329 / 0 / 15**; the
  `#[ignore]`d set serially **14 passed / 0 failed**.

**Honest scope.** The reservation is per *allocator* (per `GraphCache`), not per process: two
caches (two models, or the spec-decode pair) each hold their own slots, exactly as they always
held their own pools. Staging entries of an evicted graph are not reclaimed (`cross` is a
`HashMap` the allocator does not prune against the cache); the cost is a few boundary-sized
buffers per evicted graph, and a prune keyed on the live uids is the obvious follow-up if the
cache ever grows past a handful of graphs. Metal shares the code path but was compile-checked
only (`build-macos`).

#### E5 record, S1 (2026-09-23) — a layer-granular offload plan

**Why.** Device participation was one all-or-nothing check over the whole model
(`Qwen2Model::device` asked "is every weight registered on the GPU?"), so a model that does not
fit in device memory could not run at all — the roadmap's own example being "7B Q8_0 will not
fit (7.2 GB weights alone)". Two things were missing: **granularity** (nothing in the IR said
which block a node belonged to, so nothing could decide "these blocks here, the rest there") and
a **knob** to choose the split.

**What S1 landed.**

- **`src/graph/offload.rs`** — `OffloadPlan { gpu_layers, n_layers }` and the *pure* resolver
  `OffloadRequest::plan(env, n_layers, device_available)` (the E6 `batch_mode` pattern: CI covers
  the matrix with no GPU). `--gpu-layers N` (CLI) and `MINFER_GPU_LAYERS=N` (environment) spell
  it; unset means "every block a device can hold" — the pre-E5 behaviour — and anything that is
  not a block count fails the load loudly. Blocks `0..gpu_layers` run on the device; the tensors
  **outside** any block (the embedding, the final norm, `lm_head`) follow the device only when
  *every* block is offloaded, so a partial plan never has to fit the two largest tensors
  (llama.cpp's `n_gpu_layers > n_layer` convention, written down once).
- **The plan is one number read in three places, which must agree:**
  1. the **loader** registers a tensor on the device only when `OffloadPlan::allows_weight(name)`
     says so — the block comes from the registry name (`block_of`, `{ns}blk.{i}.…`), including
     the fused `blk.{i}.attn_qkv` / `blk.{i}.ffn_gu` concat copies. Registering a non-offloaded
     block would spend exactly the device memory the plan exists to save;
  2. the **builder** stamps `CNode.layer` (`GraphBuilder::set_layer`, called once per block by
     both model builders) and gates the device-only fused forms on
     `layer_gpu = gpu && il < gpu_layers`;
  3. the **assignment pass** (`BackendScheduler::assign_backends` →
     `GraphAllocator::supports_for(op, dtype, layer)`) never offers the device for a block past
     the plan, and `CParams.gpu_layers` carries the plan into the **reuse identity** — the
     assignment is topology, so a different plan must rebuild.
- **Verification and reporting.** After registering, the load asks `device()` whether the
  offloaded blocks can actually run there; if not, the plan drops to CPU-only with a printed
  reason (never a silent partial offload). `offload_report()` prints the startup line E5 asks
  for, with the device memory the offloaded weights measured.
- **A mixed plan refuses a KV session**: a session file is one arena with one backend tag (C5),
  so `kv_load` refuses before reading the file. `kv_save` already refused mixed layers.

**Acceptance, as measured.**

- **A chosen split runs** (`a_partial_offload_runs_the_rest_on_the_cpu`, 0.5B q4_0 on GB10):
  `--gpu-layers 4` loads with 4 blocks on CUDA and 20 on the CPU, the startup line reads
  `offload: 4 of 24 blocks on cuda, 20 on cpu; embed/output on cpu (32.0 MiB of device weights;
  --gpu-layers 4)`, and **four greedy steps match the all-CPU run** (CPU and device logits differ
  by design, rule 9, so tokens are the honest comparison). The device holds only the offloaded
  blocks: `device_bytes >= Σ(blocks 0..4)` and `< Σ(all blocks)`.
- **The boundaries copy** (the same gate): the built decode graph has **40 nodes on CUDA and 368
  on the CPU**, cut into **7 splits (3 device, 4 CPU)**, and both a device→CPU and a CPU→device
  boundary carry non-empty `inputs` — the scheduler's cross-backend copies are what make the
  mixed graph executable at all. (The split count is not one per block: a block's device nodes
  are contiguous with its neighbours' whenever the nodes between them are on the device too, so
  the claim asserted is the alternation plus the copies, not a count.)
- **Placement is a gate, not a hint** (`the_offload_plan_keeps_late_blocks_off_the_device`,
  CUDA): with `gpu_layers = 2/4`, `supports_for(...)` answers `Cuda` for block 0, `CPU` for block
  2, `CPU` for a node outside any block, and `Cuda` for an unblocked node only under a full plan.
- **The unit matrix** (5 tests, no device): unset/empty/garbage spellings, clamping, the CLI form
  beating the environment, the "unblocked tensors need a full plan" rule, the weight-name filter
  (`blk.3` yes, `draft.blk.3.attn_qkv` yes, `blk.4` no, `blk.0attn`/`token_embd` no), and the
  report text.
- **The builder contract** (`set_layer_tags_the_nodes_created_after_it`): nodes created after
  `set_layer(Some(2))` carry `Some(2)`, views inherit it, and the tag clears.
- **A mixed plan cannot resume a session** (the refusal precedes the file read — the test's path
  does not exist).
- Suites: CPU **273 passed / 0 failed / 13 ignored**; CUDA (GB10, serial) **325 passed / 0 failed
  / 14 ignored**; the `#[ignore]`d set serially **13 passed / 0 failed** on the CPU build (the
  mixed-run gate skips there — no device — and passes on the device, above).
- **CLI, end to end** (0.5B on GB10): `--gpu-layers 4` prints the line above and generates;
  `--gpu-layers 0` prints `offload: cpu only — 0/24 blocks on the device (--gpu-layers 0)`;
  `MINFER_GPU_LAYERS=6` prints the environment as the source; `MINFER_GPU_LAYERS=banana` fails the
  load with the reason instead of being ignored.
- **Mutation-checked**: with the placement gate removed from `supports_for`, the mixed run fails
  (a non-offloaded block reaches the device and its weights are not there); with
  `allows_weight` forced true, the device-bytes assertion fails (the device would hold the whole
  model, which is what the plan exists to prevent).
- **Honest scope**: this box's device has ~128 GB, so "a model larger than device memory" cannot
  be *staged* here. The gate forces the split with the knob and asserts the device holds only the
  offloaded blocks — the knob is exactly how the constraint is expressed; the automatic fit is
  S2 below. Metal takes the same code path but is compile-checked only (`build-macos`).

**What S1 does not do** (it stays on [#46](https://github.com/yusiwen/minfer/issues/46)): the
**automatic fit** — consuming E4's memory accounting to put "as many blocks as the budget allows"
on the device, which is what turns the knob into a policy (and what a device whose free memory is
smaller than the model needs); per-block `tensor_split`-style tuning; and a per-layer map for the
server's batching default (the server sees "the device participates", which is right for a mixed
plan but does not distinguish it).

#### E4 record, S2 (2026-09-23) — the pools allocate at the class, and what that exposed

**Why.** S1 ended with the ladder in the plan and the accounting but **exact pools**: two shapes in
one class still got a buffer each, so a rebuild with a slightly different `nt` grew the pool, and the
ticket's "recycled buffers come from the same size classes (no silent growth)" was not claimed. The
obstacle was that a pooled buffer is no longer the same thing as the node it serves: the pool buffer
is `class_size(n)`, while the node's real length is `n`, and every consumer that used a *physical*
length had to be told.

**What S2 landed.**

- **The pool request is the class** — `alloc_class_in_pool` asks for `class_size(size)`. Because the
  free list matches lengths exactly and every buffer of a class has the same length, the second shape
  in a class now finds the first one's buffer. The persistent KV regions still come through
  `alloc_exact_in_pool`: a cell's width is a layout contract (`row_elems`), not a tuning knob.
- **The length contract lives in `BufRef`** (new `Backend::write_host_window(id, offset, data)`): a
  fill is checked against the node's *logical* `BufRef::len` and written at the reference's offset,
  and `write_host` keeps the exact-length contract for the persistent regions and staging.
  `get_buffer` returns the reference's window, not the physical slice.
- **The capture reads are windowed** (`scheduler::window_of`): the CPU readback, the staged Metal
  readback and the CUDA `capture_enq`/`capture_drain` path all fed `data.len()` to
  `trace::analyze` as `n_total`, so a class-rounded buffer would have reported 256 elements for a
  1-element `token_ids` input and changed every viz statistic.
- **`pool_bytes` is what the pool holds** (`Backend::pool_len` is the probe): it only grows when the
  pool creates a buffer, so a recycled class buffer is not charged again. S1 charged every request,
  which made the report grow on every rebuild (and the budget gate refuse graphs that fit). Staging
  buffers are now charged too (exact, but resident and live until the next rebuild frees them).
- **Two latent bugs, both found by the rounding and both fixed here:**
  1. **A liveness extension did not move the pool's deadline.** `buf_alive` is written from
     `last_use` at allocation; an in-place alias and a D1 view extend `last_use` later, and `sweep`
     reads only `buf_alive`, so a buffer could be handed to another node while the alias still read
     it. `extend_through_views` now reports the nodes whose liveness grew and `extend_buffer_alive`
     bumps their buffers' deadlines. D1's view branch had this wrong twice over: it called the walk
     with `from = the view` and `to = the view's own last use`, so the first check ended the walk —
     **the parent extension never ran at all**; it now walks from the parent.
  2. **An input could take a buffer the walk released.** Inputs are host-filled *before* execution,
     so the previous owner's write (during execution) lands after the fill: `seq_ids` took the
     `matmul K` buffer in the 0.5B decode graph and read `-8.47` where its cell index belonged. All
     input buffers are now placed **before** the walk, when the free list still holds only the
     previous graph's buffers.

**Acceptance, as measured.**

- `a_rebuild_inside_one_class_reuses_the_pool`: two rebuilds inside one class (896×16 = 14336 and
  896×17 = 15232, both class 16384) leave `pool_bytes` **unchanged** and add **zero** pool buffers;
  a shape that leaves the class (896×20) does reserve more. Mutation-checked against exact pools and
  against per-request `pool_bytes` accounting.
- `a_fill_must_match_the_nodes_logical_length`: a 3-element input occupies one class, a 3-element
  fill is accepted, and 2- or 4-element fills are refused naming both numbers; the refused fills
  wrote nothing and `get_buffer` reads exactly 3 elements. Mutation-checked (drop the check).
- `an_input_never_takes_a_buffer_the_walk_released` and
  `a_view_keeps_its_parents_buffer_alive_through_later_consumers`: two synthetic execute-and-compare
  graphs whose values change if the buffer is recycled early. Both mutation-checked (remove the input
  pre-pass; extend from the view again).
- Suites: CPU **265 passed / 0 failed / 12 ignored**, CUDA on GB10 (serial,
  `scripts/cuda_test.sh`) **316 passed / 0 failed / 13 ignored** (+4 over S1's 312).
- Real-model: the **nine** non-ignored real-model graph tests that the un-migrated rounding
  broke (0.5B q4_0, 24 layers) — `graph_logits_match_forward_real_model`,
  `a_two_sequence_batch_matches_two_single_sequence_forwards`,
  `offset_sensitivity_is_narrowed_to_multi_query_attention` and the rest — pass again, and the
  whole CPU suite is green a second time with `MINFER_BATCH_TEST_MODEL` pointed at
  `Qwen3-0.6B-Q8_0.gguf` (f16 KV, the second cache-width path). The `#[ignore]`d real-model set,
  run **serially** (`cargo test --release --bin minfer -- --ignored --test-threads=1`), is
  **12 passed / 0 failed** — the same as on master; in parallel it is red, on master too, because
  the C4 packed-cache gate sets the process-wide KV format mid-run (issue
  [#99](https://github.com/yusiwen/minfer/issues/99), and now documented in `AGENTS.md`).
  `MINFER_TRACE` on the 0.5B reports `n = 30` for the `token_ids` input (the prompt's length);
  with the readback window removed it reports the class's **256**, which is the mutation
  evidence for the capture fix.

**Findings filed while doing S2** (both pre-existing, neither is caused by the allocator):
[#98](https://github.com/yusiwen/minfer/issues/98) — `ComputeGraph::topo_order` counts in-degree
per source entry but decrements once per node, so a graph with a repeated source (`add(x, x)`)
is rejected as a cycle, and `alloc_graph` calls it on every build;
[#99](https://github.com/yusiwen/minfer/issues/99) — the process-wide KV format above.

**What S2 does not do** (it stays on [#55](https://github.com/yusiwen/minfer/issues/55)): the
**reserve/assign re-map** (a reserved region a rebuild re-maps without touching the device — the
literal "split reservation from assignment", which Metal's G6 adopts) and the **multi-graph cache**
(§14 row 3, which is what removes E3's per-chunk rebuild). Cross-boundary staging is charged to
`pool_bytes` but is still allocated at its exact length, and backend-internal scratch (Metal capture
staging, CUDA `positions` scratch) is outside the report.

#### E4 record, S1 (2026-09-22) — account first, allocate second

**Why.** Roadmap §2.3 listed three consequences of an allocator that places and hands out
memory in one step: an exact-match pool ends up with one set of buffers per shape, nothing
can answer "will this fit?" before the device is touched (CUDA surfaced it as a null pointer
at execute time), and peak memory was not a number anyone could read.

**What S1 landed.**

- **`src/graph/allocplan.rs`** — the size-class ladder and a *pure* plan.
  `class_size(elems)`: powers of two up to 16 KiB (4096 elements), then multiples of 16 KiB,
  floor 1 KiB; a class never wastes more than one 16 KiB step. `AllocPlan::plan` takes the
  `(size, first use, last use)` intervals and simulates the pool's own reuse rule (a buffer
  freed at step `f` may serve an interval whose first use is after `f`), reporting
  `reserved_bytes`, `live_peak_bytes`, `buffers` and `reused`.
- **Accounting** — `GraphAllocator::memory_report(backend)` returns
  `MemoryReport { weights_bytes, pool_bytes, live_bytes, peak_live_bytes, budget }` plus
  `headroom_bytes()`. `pool_bytes` is the pool's high-water mark (the pools never return
  memory), `live_bytes` is what is handed out right now, and the peak is tracked as the
  build proceeds. `weights_bytes` comes from the backend (a new `Backend` trait method with
  a 0 default; CPU sums its registry, CUDA sums the device registry — Metal inherits the
  default until G6 adopts the split).
- **The feasibility gate** — `alloc_in_pool` is fallible and checks
  `weights + pooled + this allocation (at its class size)` against the backend's budget
  *before* the pool is asked for anything. The default budget is the backend's own answer:
  CUDA's current `cudaMemGetInfo` free bytes with a quarter held back (new
  `CudaState::device_memory` / `device_free_bytes`); CPU and Metal are unbounded unless
  `set_memory_budget` sets one (tests, and a future offload policy). The refusal names the
  numbers: weights, pooled bytes, this request, budget, all in MiB.

**Acceptance, as measured** (in CI, no device needed):

- a 4095-byte budget against a 4 KiB activation is refused with
  `out of CPU memory: … MiB of weights + … MiB of pooled buffers + … MiB for this activation
  exceeds the … byte budget (… MiB)`, and **the pool is untouched** (`n_cpu_buffers() == 0`,
  `pool_bytes == 0`) — the gate runs before the first backend call;
- the same graph fits at 1 MiB, and the report then shows `0 < live_bytes <= pool_bytes`,
  `peak_live_bytes >= live_bytes` and `headroom_bytes() < budget`;
- weights and activations are **one comparison**: a budget covering the registered weight
  but not the activation is still refused, and one byte more is accepted;
- the ladder is mutation-checked in both directions (the roundings above, plus the pure plan
  tests: two shapes in one class share, overlapping lifetimes do not, the plan is
  order-independent) and the gate itself is mutation-checked (disabling the comparison fails
  `a_graph_that_cannot_fit_is_refused_with_its_numbers`).

**What S1 does *not* do** (it stays on [#55](https://github.com/yusiwen/minfer/issues/55)):

- **The pools still allocate exact sizes**, so two shapes in one class do not yet *share* —
  the ladder is the plan's and the accounting's view, and the ticket's "recycled buffers come
  from the same size classes (no silent growth)" is **not** claimed yet. Rounding the pools
  is not a one-line change: every consumer's length contract has to move to the owning
  `BufRef` at the same time (the host read/write checks, the trace/dump capture, the
  scheduler's staging copy and the CUDA `copy_to_host` all return a physical length today).
  That is the reserve/assign re-map, and it is the next increment.
- **The reserve/assign re-map itself** (a reserved region a rebuild re-maps into without
  touching the device) and the **multi-graph cache** (which §14 row 3 points at, and which is
  what removes E3's per-chunk graph rebuild) are S2 as well.
- The pools still never return memory to the host/device (documented, accepted debt in
  `docs/CUDA-BACKEND-DESIGN.md`); the accounting now makes it visible as
  `pool_bytes` vs `live_bytes`.

#### E3 record (2026-09-22) — a prefill in chunks, and what runs between them

**Why.** A prefill was **one** forward over the whole prompt. Two consequences: activation
memory scaled with the prompt (`GraphParams.n_tokens` sizes every activation buffer), and a
long prompt blocked every other slot's decode for its whole duration — the batch worker is
single-threaded, so `admit` returning after the full prefill meant the other slots' tokens
simply waited.

**Rule.** `prefill_chunks(from, total, chunk)` splits the fed suffix into spans of at most
`chunk` tokens, in order, with the remainder last (so the final forward carries the tail row
whose logits the request samples from); `chunk == 0` is "off" — one span, the pre-E3 path —
and a suffix that already fits is never split. `MINFER_N_BATCH` (default
`DEFAULT_PREFILL_CHUNK = 2048`) sets it, and `--n-batch`-style plumbing is the same setter:
`BatchEngine::set_prefill_chunk`. The server prints the outcome at startup, like the batching
mode, because a default nobody can see is a default nobody can debug.

**Interleaving is the point.** Between chunks — never before the first or after the last —
the prefill runs the same decode step `serve_loop` would have run next, if any other slot has
a token waiting (`has_pending_decode`). That is safe by construction: the chunk's rows are
already written, the slot has no `Run` yet, and `finish` keeps a completed slot's rows (B2
reuse) rather than compacting, so nothing moves under the in-flight prefill.

**Why 2048 as the default**: it is a *no-op* for every prompt that fits it — identical
forwards, identical graph, identical timing — so the change cannot regress the common case,
while a longer prompt gets bounded memory and interleaving. The cost of the split is real and
measured, and it is one **forward's fixed overhead per chunk** (the weights re-stream and the
graph re-fills: it is keyed on `n_tokens`, so a chunked prefill rebuilds once per chunk —
§14 row 3; E4's multi-graph cache is what removes that):

| prompt | forwards | CPU | CUDA (GB10) |
|---|---|---|---|
| 98 tokens, chunk 24 | 1 → 5 | 484 → 485 ms (**1.004x**) | 29 → 57 ms (**2.0x**) |
| 514 tokens, chunk 171 | 1 → 4 | 2676 → 2691 ms (**1.005x**) | 107 → 119 ms (**1.11x**) |

The small-chunk case is the worst one (five weight passes for 98 tokens); at the default, a
4096-token prompt is two forwards, i.e. one extra fixed cost on a prefill of that size.

**Acceptance, as measured** (`cargo test --release --bin minfer -- --ignored`, 98-token
prompt, chunk 24):

- **Bounded**: the chunked run issued **5** prefill forwards whose largest `nt` was **24**;
  the unchunked run issued **1** forward of `nt = 98`. The bound is `prefill_stats()`, an
  observable, not a claim about buffers.
- **Equal**: the prefill's tail-row logits are **bitwise identical** on CPU (max |Δ| = 0) and
  within the named cross-shape class on CUDA (**0.218**, class 1.0 — CUDA's prefill tiles by
  `nt` and quantizes activations to int8); the CPU continuation is equal byte for byte.
- **Interleaved** (48×"buffalo " on slot 1 already decoding, a ~4× chunk prompt on slot 0):
  the chunked prefill ran **3** decode steps for the other slot and it emitted **3 bytes**
  during the prefill call; with chunking off the same call ran **0** steps and the slot
  gained **0** bytes — the A/B is deterministic, not a timing race.
- Both gates are mutation-checked: disabling the interleave fails the second
  (`ticks 0, bytes 0`), and making `prefill_chunks` never split fails the first
  (`1 forwards for a 98-token prompt at chunk 24`).

**Coverage, stated rather than implied.** The chunk plan and the env parser are pure and run
in CI; the two real-model gates are `#[ignore]`d (they need the cached 0.5B) and were run on
the CPU **and** on CUDA (GB10). Not in this increment: a `--n-batch` CLI flag (the setter and
`MINFER_N_BATCH` are the surface), and true **mixed** prefill+decode batches — the decode
steps between chunks are the same `tick` the worker already runs, one per chunk, which bounds
the stall without yet sharing a weight pass between a chunk and a decode row.

#### E6 record (2026-09-19) — a device-aware default

**Why it needed its own change.** E2 measured the *sign* of the batching effect
per device (CPU 0.49x, GPU 1.9x) and, correctly, left the default with the
measured-better path on the box it could measure. With the GPU available that
stopped being the honest default: a CUDA server was leaving a ~2x win on the
table behind an environment variable almost nobody would set.

**Rule.** `MINFER_BATCH` unset -> batched iff the model's forwards run on
**CUDA**; `=1` -> batched (also the way to batch on CPU); `=0` -> serial; any
other value warns and falls back to the device default. Metal is excluded *by
construction*, not caution: the batched path needs `attn_span` and Metal refuses
that node (`supports_attn_span()` is false there), so batching a Metal server
would fail loudly instead of serving; it waits for G5.

**The decision has one authority.** "The device participates" used to be
recomputed inside each model's `forward_batch` (`metal_available() &&
weights_on_gpu`, `CudaState::get().is_some() && weights_on_cuda`). It is now
`Qwen2Graph::device` / `Qwen3Graph::device` (an E6 addition), which the builder
derives `CParams.gpu` from **and** the server derives its default from — so the
two cannot disagree, and a partial weight registration (device present, weights
not all registered) correctly keeps the server serial. `ModelDef::device()`
exposes it as a `Device::{Cpu, Metal, Cuda}` with a CPU default.

**The decision is pure and tested without a device**, which matters because CI has
no GPU: `server::chat::batch_mode(requested, device)` is a pure function and
`batch_mode_follows_the_device_and_honours_the_override` covers the matrix
(unset / "1" / "0" / invalid x cpu / metal / cuda). Every server now also prints
the outcome at startup (`[server] batching: on|off (device cuda|cpu|metal; ...)`),
so the default is observable rather than inferred.

**Device verification (2026-09-19).** CPU build: unset -> `off (device cpu)`,
`=1` -> `on`, `banana` -> warning + `off`. CUDA build on the GB10: unset ->
`on (device cuda)`, `=0` -> `off`. Acceptance re-measured with the **default**,
no environment variable: 7B Q4_K_M, `--n-slots 4`, four identical prompts,
`max_tokens=16`, equal work — **0.66 s batched (default) vs 1.30 s with
`MINFER_BATCH=0` = 1.97x**, matching E2's 1.9x (1.314 vs 0.686) within noise.
Suites: CPU 192 passed / 0 failed, `--features cuda` 238 / 0.

- **E1 acceptance:** the mask is an explicit input, not a derivation from
  `positions`; a two-sequence test proves no cross-attention; single-sequence
  output is bitwise unchanged.
- **E2 acceptance:** aggregate throughput at `--n-slots 4` materially exceeds
  the serial baseline on a fixed workload; the `n_seqs` field is either real or
  deleted (closes A7 if it was kept).
**E2 design (written before the code, 2026-09-17).** `--n-slots N` does not
serve concurrently today: each `Slot` owns its own `GraphCache` and the worker is
serial (`server/slot.rs`), so `N` only divides the context budget
(`n_ctx_slot = n_ctx / n_slots`) — four slots buy nothing. E1 made the *attention*
side sequence-aware; E2 has to make the *bookkeeping and the serving loop*
sequence-aware:

1. **`KvCache` gains reservations.** A sequence is a `SeqId` owning a contiguous
   cell run: `reserve_seq(seq, cap)` (first-fit over free cells, `Err` when the
   arena cannot fit it), `release_seq(seq)`, and `seq_range(seq) -> (start, cap)`
   from the reservation rather than from a scan of `owner`. Ownership keeps its
   C1 meaning — it marks *written* cells — and becomes per-sequence:
   `own_range(seq, from, to)`, replacing today's `own_prefix(SEQ_MAIN, n)` which
   would clobber a second sequence's cells.
2. **The single-sequence path is the `cap == n_ctx` special case.** `SEQ_MAIN`
   takes the whole arena, so `seq_range` is `(0, n_ctx)` and `attn_span` still
   yields `[0, pos + 1)` — the existing model path stays bitwise, which is the
   refactor's acceptance gate.
3. **A batch is data.** `Batch { tokens, positions, seq_ids, n_out }` with
   `n_seqs` = distinct ids; the graph is built per `(n_tokens, n_seqs)` (both
   already in `GraphParams`), and the allocator fills `seq_ids`/`attn_span` from
   the batch exactly as E1 does for one sequence. `n_seqs` becomes a real field
   (it now gates the `multi_seq` op flag and the graph identity) — A7's "real or
   deleted" question answered with *real*.
   - **As implemented (outcome, 2026-09-17):** the second half of that sentence
     did not survive contact. `n_seqs` was made real for one commit, then the
     flag it was said to gate moved to `CParams.explicit_span` (the *reservation*
     is the authority on the window, not the count), and a test showed the count
     now changed nothing about the topology while still forcing a rebuild. The
     field was **deleted** — A7's question answered with *deleted*, the other
     branch of the same acceptance clause. See §8 and the A7 ticket.
4. **The server batches decode steps.** One shared cache for the batch; each
   active slot holds a reservation and its token stream; one forward per step
   carries every ready slot's next token (`nt = ready slots`, `n_seqs = ready
   slots`), so one weight pass serves all of them. Prefill stays per-slot in this
   increment (mixing prefill into a decode batch is E3's chunked prefill), and a
   decode batch is capped at 16 tokens so it rides the batched split-attention
   path on CUDA (`fa_prefill`'s tile must not span two sequences — documented in
   E1b).
5. **Tests and measurement.** Bitwise: a multi-sequence decode batch must equal
   the same sequences run one at a time on the CPU reference. Resolver: two
   reservations do not overlap, `release_seq` frees exactly its rows, `Err` when
   the arena is full. Serving: aggregate throughput at `--n-slots 4` against the
   same workload run serially (the ticket's acceptance), reported as a ratio with
   the workload recorded.

**Not in E2:** mixed prefill+decode batches and chunked prefill (E3), moving a
sequence's cells when the arena fragments (C3 needs D1; E2 reserves a slot's
budget up front and fails loudly instead), layer offload (E5), Metal (G5), and
CUDA runtime verification (deferred under A0; **done 2026-09-18 for E1b and for C2's CUDA arm** — see the sweep below).

**E2 progress (2026-09-17).** Step 1 of the design landed: `KvCache` holds
per-sequence reservations (`SeqSlot { start, cap }`, `reserve_seq` first-fit,
`release_seq`, `seq_slot`) and per-sequence ownership (`own_range`, replacing
`own_prefix`'s hard-coded sequence 0), and `attn_span` resolves a query's window
from its sequence's reservation — with a **written-row check** (`owner[pos] ==
seq`), so a window can never include rows nobody wrote. A reservation is not
ownership: reserving cells does not let a query attend to them, which is what
keeps a batch's unwritten rows out of attention. `after_rm`/`after_shift` now
refuse a multi-sequence cache explicitly (C2's shift is the single-sequence
sliding window; a general move is C3).

The single-sequence path is the `cap == n_ctx` special case: `own_prefix`
reserves the whole arena for `SEQ_MAIN`, so `attn_span` still returns
`[0, pos + 1)` and the real-model bitwise tests are the refactor's gate — they
did not move (`cargo test --release` 183 → 184 passed / 0 failed, the +1 being
the new reservation test). The allocator exposes the E2-facing surface
(`kv_reserve_seq`/`kv_release_seq`/`kv_seq_slot`/`kv_own_range`).

**E2 progress, step 2 (2026-09-17).** The batch entry point landed:
`graph/batch.rs` defines `Batch { tokens, positions, seq_ids }` with `groups()`
(contiguous runs), `out_rows()` (one logits row per sequence for a batch; the
last `n_out` rows for a single sequence) and `check()`, which **refuses an
interleaved batch** — the CPU path could tolerate it, but CUDA stages a query
tile against one window (E1b), so an interleaved batch would be right on CPU and
wrong on CUDA; that is the silent divergence the project refuses.
`ModelDef::forward_batch` (default: refuse) runs one forward over a batch, and
`forward_graph_cached` is now its one-sequence case, which is why the classic
path's tests did not move. `GraphAllocator::fill_batch_inputs` marks each
sequence's written rows, fills `seq_ids` and resolves the span in one call, and
`kv_set_capacity` lets a caller reserve before the first `alloc_graph`.

**The flag had to change meaning.** E2 makes a *single* sequence start at a
non-zero cell (a slot's reserved run), and CUDA's causal instantiation derives
`nkv = positions[t] + 1` — correct only when the window starts at cell 0. So
`Op::Attn { multi_seq }` became `Op::Attn { explicit_span }`, meaning “positions
alone cannot bound this node”: more than one sequence **or** a window that does
not start at 0. The model derives it from the KV reservations
(`kv_seq_slot(seq).start != 0`), never from `n_past`, and it lives in `CParams`
because it selects a kernel instantiation (two graphs per shape at most, exactly
like the fusion flags). The classic path — one sequence, `SEQ_MAIN`, start 0 —
keeps the causal instantiation, so E1b's SASS/performance argument still holds,
and Metal refuses the flagged node as before (G5).

**Evidence gate (`a_two_sequence_batch_matches_two_single_sequence_forwards`).**
Both sides use the *same* KV layout (sequence 7 at `[0, la)`, sequence 9 at
`[la, la + lb)`), so the only difference is one forward carrying two sequences
versus two forwards carrying one each: the batched prefill of both prompts and
the batched decode step are **bitwise equal** to the per-sequence runs on
Qwen2.5-0.5B Q4_0, and the fixture is discriminating (the two sequences' argmaxes
differ). `cargo test --release` 184 → 189 passed / 0 failed.

**E2 progress, step 3 (2026-09-17): the server batches, and the measurement says
not to default to it.** `server/batch.rs` implements the design: one shared arena
with one reservation per slot (`BatchEngine`, `n_ctx_total` rows split
`n_slots` ways), a slot keeping its KV and its `cached_tokens` across requests so
B2's prefix reuse still works, a submit path that pre-fills one request
(reuse-aware admission: among idle slots it picks the one whose KV holds the
longest prompt prefix), and a `tick` that carries every ready slot's next token
in one `forward_batch` and then samples, commits and streams per slot. A
speculative session keeps the per-slot caches and the run-to-completion loop
(doc 94/97's identity contract is per request), and a panic in a batched forward
fails the whole batch loudly, because the arenas may be half-written.

The measurement, end to end through the HTTP server with `--n-slots 4` and
`max_tokens` reached on every request (so both sides do equal work):

| Workload | Serial | Concurrent | Ratio |
|---|---|---|---|
| Qwen2.5-0.5B Q4_0, prefix reuse on | 136 tok / 4.47 s | 143 tok / 5.08 s | **0.88x** |
| Qwen2.5-0.5B Q4_0, `MINFER_NO_PREFIX_REUSE=1` | 136 tok / 6.36 s | 143 tok / 5.70 s | 1.12x |
| Qwen2.5-7B Q4_K_M, prefix reuse on | 32 tok / 16.50 s | 32 tok / 18.42 s | **0.49x** |

At the engine level (same slot layout on both sides, 4 requests × 16 tokens) the
batched *forward* is 1.45x on the 0.5B and **1.00x** on the 7B.

**Verdict: the acceptance is not met, and the two causes are measurable.**
(1) The CPU decode kernels' `nt > 1` path is not more efficient per token — the
7B's batched forward is exactly as fast as four serial forwards, so batching buys
nothing there (the 0.5B, whose decode is launch/compute bound rather than weight
bandwidth bound, gains 1.45x). (2) Concurrency *forfeits* B2's cross-request
prefix reuse: each concurrent request needs its own KV home and starts cold,
which on a model with a slow prefill (the 7B pays ~6 s for a 34-token chat
prompt) dominates — hence 0.49x despite the neutral forward. Where decode is
weight-bandwidth bound (a GPU) batching is the standard win, but nothing here can
verify that (A0), so `MINFER_BATCH=1` is **opt-in** and the default stays with the
measured-better serial path, exactly as A6 was reverted on measurement.

**E2 progress, step 4 (2026-09-17): batching the prefills too, and the trace that
closes the diagnosis.** `BatchEngine::admit` now places a group of arriving
requests and combines their prefills into **one** forward when the group fits
`MAX_PREFILL_BATCH` (per request otherwise, so the graph width does not churn;
with a CUDA device the prefills stay per request, because `fa_prefill`'s query
tile must not span two sequences — E1b). `MINFER_BATCH_TRACE=1` prints per-forward
timings, which is how the numbers below were obtained.

On Qwen2.5-7B Q4_K_M, `--n-slots 4`, four identical prompts, `max_tokens=4`, equal
work (16 tokens each side), reuse on:

```
serial:     7.65 s   (1 full prefill 4936 ms + 3 reused prefills ~140 ms + decode)
concurrent: 17.12 s  (batched prefill: 3 prompts, 129 tokens, 14788 ms + decode)
```

- **Prefill batching is exactly neutral on this CPU**: 129 tokens in one forward
  cost 14788 ms = 3 x 4936 ms, three separate prefills to the millisecond. The
  prefill is compute-bound, so sharing one weight pass buys nothing here (it is
  the case a bandwidth-bound device would change).
- **The whole gap is B2's prefix reuse**: the serial path re-feeds 1 token per
  request after the first (42 of 43 reused, ~140 ms), while four concurrent
  requests need four KV homes and each pays the full 4936 ms. Concurrency cannot
  reuse across slots without a cell copy — C3's operation, which needs D1.
- **Decode batching is neutral on this model**: a 4-wide step costs 4.0x a
  single-token step (the engine measurement: 1.00x on the 7B, 1.45x on the 0.5B).

So `--n-slots 4` cannot "materially exceed" the serial baseline on this box: for
identical prompts the serial path is ~3x cheaper on prefills that batching cannot
recover, and for distinct prompts the two tie (neutral prefill batching + neutral
decode batching). The remaining route to the acceptance on CPU is a `nt > 1`
decode kernel that actually exploits the shared weight read (the F1 family); on a
bandwidth-bound device batching is the standard win, and the trace above is what a
GPU re-measurement should compare.

Still to come in E2: nothing is left to *build* for the deliverable; what remains
is the acceptance, which this box cannot demonstrate (the step-4 trace above shows
why, and `MINFER_BATCH_TRACE=1` is the instrument for re-measuring elsewhere).

**E2 progress, step 5 (2026-09-17): A7's second half — `n_seqs` deleted.** The
ticket's second acceptance clause is "the `n_seqs` field is either real or
deleted (closes A7 if it was kept)". Step 1 had made it real (the `multi_seq`
flag), step 3 moved that flag to `CParams.explicit_span` for a reason that has
nothing to do with the count — a slot's *reservation* decides whether positions
alone can bound the window — leaving `n_seqs` set on every path and read by none.
Rather than "keep it reserved" a second time, the question was measured: a
2-sequence batch and a 1-sequence batch with the same `n_tokens`, `n_out`,
`gtype` and `explicit_span` describe the **same topology**, and with the field in
the identity they still rebuilt (uid 3 → 4, `sequence_count_is_data_not_topology`).
The field is therefore **deleted** from `GraphParams` (and from `params_match`,
the JSON export and every construction site); the builder's "more than one
sequence involved" test now reads the batch. The test that exposed the rebuild is
permanent and also asserts the reused graph's logits are bitwise-identical to a
fresh single-sequence forward, so the deletion is pinned from both sides. A7 is
now fully closed: both fields the A7 note called dead are gone. One incidental
dead field went with it — `BatchEngine`'s `Run.finish`, set to `None` and never
read (the finish reason is a parameter of `finish()`), removed with the build
warning it produced.

**E2 progress, step 6 (2026-09-18): the acceptance is MET on the GPU.** A0's
"no device" verdict turned out to be an artefact of the agent sandbox, not the
machine, so the re-measurement the closure note asked for was run on the GB10 —
7B Q4_K_M, `--n-slots 4`, four **identical** prompts, `max_tokens=16`, equal work
on both sides (64 tokens each), CUDA confirmed active in both server logs:

| Mode | Wall clock | Ratio |
|---|---|---|
| serial (default) | 1.314 s | — |
| `MINFER_BATCH=1` | **0.686 s** | **1.9x** |

The trace explains where it comes from, and confirms the E1b constraint still
holds on device: with a CUDA device the prefills stay per request
(`prefill_batch_ok()` is false, because `fa_prefill_f16kv` tiles a query against
one window) — 83 ms for the first 40-token prompt, 41 ms for each of the other
three — so the whole win is the **batched decode**. That used to be an inference
from the totals; `MINFER_BATCH_TRACE=1` now also prints the decode step itself,
and the re-run shows it directly: **15 steps of 4 sequences (plus one of 3 and one
of 1) for the 64 tokens, at 6.5–6.7 ms/token**, against ~19 ms/token for the
serial path (1.23 s of decode for 64 tokens) — ~2.9x per decoded token, which the
four full prefills the batched side pays turn into the 1.9x end to end. On CPU the
same workload was 0.49x (step 3): the sign of the effect is a property of the
device, exactly as the closure predicted.

**Device verification sweep before merging (2026-09-18).** Everything in PR #3
and PR #4 that had been left unverified for lack of a device was re-run on the
GB10, and the ones that could not be *asserted* were made assertable:

| Item | Result |
|---|---|
| E1b windowed kernels (`cuda_two_sequences_do_not_cross_attend`) | passes on device, after fixing its `hd = 2` fixture and `read_host` read |
| E1b causal vs windowed, same rows | new gate, bitwise equal on device (closes the gap above) |
| E1b causal vs windowed, **both KV dtypes** (`f32` and `f16`) | `cuda_windowed_attention_matches_causal_for_long_windows`: **22/22 bitwise equal** on device after fixing `fa_prefill_f16kv`'s windowed row mask (§14 row 0) — the f16 prefill case (`hd = 128, n = 34, start = 64`) was the one the fix is about |
| Server batching, four **different** prompts, 7B Q4_K_M (f16 KV), `--n-slots 4` | after the fix: batched **and** serial both 4/4 correct and identical (before it: batched = slot 0 right + slots 1-3 derailed, e.g. `0.1555555555555555555555`) |
| E2 batched decode windows with a real model | `batch_order_does_not_change_a_sequences_logits` passes bitwise on device |
| E2 engine acceptance (`server_batch_matches_serial_and_is_faster`, ignored) | passes: 1.32x at 0.5B, all four continuations share a non-empty prefix |
| E2 server, 7B Q4_K_M, `--n-slots 4` | 1.9x, 15 four-wide decode steps traced |
| Wider windowed batch | `--n-slots 8`, 8 concurrent prompts: 11 eight-wide steps, ~1.2–1.9 ms/token, all eight answers distinct |
| Qwen3 (the other model E1/E2 touched) multi-sequence | `--n-slots 2` on Qwen3-0.6B Q8_0: 12 two-wide steps, both answers correct — the path had no test coverage on any backend before this |
| CUDA Graph capture under batching | no capture failure or self-disable in any run |
| C2's conversation path on device (adjacent: merged in PR #2, but it shares the KV cell store) | `context_shift_real_model_measurement` passes on GPU: incremental prefills 30 then 14 tokens/turn, and the physical removal + re-rope shift takes 185 -> 14 prefill tokens with correct replies throughout |
| `conversation_real_model_smoke` (ignored, model-behaviour assertion) | **pre-existing red**: fails identically on master + device at the same `need_insert_eot` assertion, so it is not from these PRs |
| `dump_real_q4k_tensor` / `dump_real_q5k_tensor` (ignored) | **pre-existing red** debug dumps (they compare against llama.cpp artifacts); unrelated to these PRs and not used as gates |
| Full CUDA suite | 236 -> 237 passed / 0 failed / 5 ignored |
| Full CPU suite | 191 passed / 0 failed / 5 ignored |

**E2 closed (2026-09-17, maintainer decision; re-measured 2026-09-18).** With the
mechanism landed, A7 closed, and the throughput acceptance refuted *with* its two
causes measured (steps 3–4), the maintainer accepted the refutation for the CPU
and E2 was closed as **mechanism landed, CPU throughput acceptance refuted — needs
a bandwidth-bound device**, with the serial path kept as the default and
`MINFER_BATCH=1` as the documented opt-in. Step 6 then supplied the
bandwidth-bound device and the acceptance **holds there (1.9x)**: the CPU result
stands as the reason the default stays serial *on CPU*, and a device-aware default
(enable batching automatically when a CUDA device participates) is now a
supportable follow-up rather than a guess — it needs its own ticket, since it
changes the server's default behaviour. This follows the A6 precedent (a ticket may close on
a measured negative result, recorded so nobody re-opens it). The follow-on work
the decision names is C3/D1 (cross-slot prefix reuse via a cell copy, which is
what the 0.49x gap is actually made of) and C4/C5; a CPU `nt > 1` decode kernel
(the F1 family) is the only CPU-side route to the original throughput claim and is
*not* part of E2. A GPU re-measurement should re-run the step-4 trace before
concluding anything about batching on device — with a CUDA device the prefills
stay per request (E1b), so the comparison there starts from a different baseline.

- **E4/E5** are what make a model that does not fit in VRAM runnable at all.

**E1 design (written before the code, 2026-09-17).** The bound a query may attend
over is derived from `positions` today (`cpu_backend.rs`: `let vl = pos[t] + 1`,
and the same derivation on device in CUDA). That is correct only while one
sequence owns every written cell, and it is the one thing that makes E2
impossible: a batch holding two sequences would let each one attend to the
other. E1 replaces the derivation with **data**:

1. **Two new inputs, not new topology.** `seq_ids` (I32 `[nt]`, one sequence id
   per query token) and `attn_span` (I32 `[2*nt]`, the allowed cell range
   `[lo, hi)` per query). Both are `Op::Input` leaves filled per step by the
   allocator, exactly like `positions` — so `GraphParams`/`CParams` and the
   params-only reuse identity do not move.
2. **`KvCache` resolves the span.** The cell store already owns `owner[cell]`;
   E1 adds `seq_range(seq) -> Option<(start, len)>` and
   `attn_span(seq_ids, positions) -> Result<Vec<u32>, String>`, which returns
   `lo = start`, `hi = min(start + len, pos + 1)`. Ownership must be contiguous
   per sequence — that is the invariant the resolver asserts instead of
   silently producing a wrong bound, and the reason a *range* is enough.
3. **`Op::Attn` consumes it, and declares when positions would be wrong.**
   `Op::Attn { mode }` becomes `Op::Attn { mode, multi_seq }`. The kernels always
   read the span (single code path, no "derive or read" branch); `multi_seq`
   exists so a backend that has not been ported **refuses** the op instead of
   falling back to the positions derivation — Metal (untouched, Phase G) is that
   backend. `n_seqs > 1` is what sets the flag, giving `GraphParams.n_seqs` its
   first reader (E2 is what will make it a batch).
   - **As built:** the flag is named `explicit_span` (`Op::Attn { mode,
     explicit_span }`), and it is set from the *batch and its KV reservations* —
     "more than one sequence involved, or a window that does not start at cell 0"
     — never from `GraphParams`. E2 deleted `n_seqs` for exactly that reason
     (§8); this paragraph is the E1-era design that predicted otherwise.
4. **Bitwise for one sequence.** With a single sequence, `lo = 0` and
   `hi = min(n_used, pos + 1) = pos + 1`, so the CPU kernel's loop bounds,
   reduction length and accumulation order are unchanged — the existing
   real-model bitwise tests (`graph_logits_match_forward_real_model`,
   `reused_cache_across_prompts_matches_a_fresh_cache`, the C2 tests) are the
   gate, not a new tolerance class.
5. **Tests.** The cell store resolves two sequences in one arena (and errors on
   non-contiguous ownership); an op-level two-sequence graph — one query per
   sequence, distinctive V rows — must reproduce the single-sequence result
   **bitwise**, which is what "no cross-attention" means; A1's op matrix gains
   the `multi_seq` cell so the Metal refusal is recorded rather than assumed.
6. **CUDA.** Both attention kernels (`gqa_attn_split` and the non-split path)
   take the span; the I32 input reaches the device through the existing
   capture-safe `positions_i32` conversion. This box had no device at the time
   (A0 — **superseded 2026-09-18**), so the CUDA half could only be compile-verified then — see the landing record below: it is
   deferred to **E1b** rather than changed blind, and the assignment gate keeps
   CUDA correct (single-sequence) in the meantime.

**Not in E1:** batch composition and continuous batching (E2 — nothing composes
several sequences into one forward yet, so the model path fills `seq_ids` with
`SEQ_MAIN` and stays a single-sequence caller); chunked prefill (E3); a full
per-cell mask, which only a layout with holes needs (C3/D1 — a range covers
every layout the engine can currently produce); Metal (G5).

#### E1 record (2026-09-17)

**What landed.** `seq_ids` and `attn_span` are IR inputs (`Op::Input` leaves,
filled per step like `positions`); `Op::Attn { mode }` became
`Op::Attn { mode, multi_seq }` with sources `[q, kv, pos, span]` — `positions`
stays at index 2 because the Metal and CUDA arms read it there, and the span is
the new fourth input. `KvCache::seq_range` / `attn_span` resolve each query's
`[lo, hi)` from per-cell ownership (start) and its position (causal end), and
`GraphAllocator::fill_attn_inputs` is the single call that records how far the
forward writes, fills the ids and resolves the span, so the IR's ids and the
kernel's window cannot drift apart. The CPU kernel now walks `lo..hi` instead of
`0..pos[t] + 1`.

**The refusal has one definition.** `Backend::supports_attn_span` (default
`false`) plus `graph::backend_takes` decide assignment; `GraphAllocator::supports`
and A1's op matrix both call it, so a backend that still derives its bound from
positions is never handed a `multi_seq` node — and the op matrix gained an
explicit asymmetric row for it.

**Bitwise for one sequence, as specified.** With one sequence `lo = 0` and
`hi = min(n_used, pos + 1) = pos + 1`, which is exactly the old derivation, so
the loop bounds, reduction length and accumulation order are unchanged. Unit
suite `179 → 183 passed / 0 failed`; the real-model bitwise tests
(`graph_logits_match_forward_real_model`, `prefix_reuse_matches_a_full_prefill`,
`reused_cache_across_prompts_matches_a_fresh_cache`, the C2 tests) never moved,
and the pre-E1 vs E1 binaries generate byte-identical greedy text on
Qwen2.5-0.5B Q4_0, Qwen3-0.6B Q8_0, Qwen2.5-7B Q4_K_M and Qwen2.5-14B Q4_K_M
(only timing lines differ).

**No cross-attention (`two_sequences_do_not_cross_attend`).** One arena, two
sequences: sequence 0 owns row 0, sequence 1 owns row 2, queries at positions 0
and 2 with spans `[0, 1)` and `[2, 3)`. The K/V values are chosen so a leak
*changes the answer*: query 1 scores 1.0 against sequence 0's key, so a window
that wrongly started at 0 would return `[0.73, 0.27]` instead of sequence 1's
`V = [0, 1]`. The test asserts the exact window and the output, and the resolver
has its own store-level tests (`two_sequences_resolve_to_disjoint_windows`, plus
`Err` on non-contiguous ownership and on an empty window).

**The CUDA half is deferred, and why.** The ticket says "CPU + CUDA". CUDA's
attention kernels still compute `positions[t] + 1` (six of them:
`gqa_attn_f32_f16kv`, `gqa_attn_f32`, the split partial/combine pairs,
the batched variants and the flash-attention prefill path at
`cuda_kernels.cu:4194`). A window with `lo > 0` changes what the split-K chunking
covers, so the port is not mechanical; and this box had **no device at the time**
(A0 — **superseded 2026-09-18**: E1b is now device-verified), which made it the
one class of change that could not be verified then — a mistake would
silently corrupt *single-sequence* GPU output that is known-good today. So: CUDA
keeps its existing behavior, `supports_attn_span()` stays `false` for it (the
trait default), the assignment gate refuses it a `multi_seq` node, and the port
is ticket **E1b**. E1's acceptance is therefore met on CPU and **not** met on
CUDA — recorded here rather than claimed, and **the deferral was agreed with the
maintainer on 2026-09-17** rather than taken unilaterally.

**Not in E1:** batch composition and continuous batching (E2 — nothing composes
several sequences into one forward yet, so the model path fills `seq_ids` with
`SEQ_MAIN` and stays a single-sequence caller); chunked prefill (E3); a full
per-cell mask, which only a layout with holes needs (C3/D1 — a range covers
every layout the engine can currently produce); Metal (G5); the CUDA port (E1b).

#### E1b record (2026-09-17) — CUDA's windowed attention

**Shape.** Every attention kernel is now `template <bool CAUSAL>`. `CAUSAL`
(the existing behaviour) keeps `nkv = positions[t] + 1` and a row base of 0; the
windowed instantiation reads the `[lo, hi)` pair (`bound[t]`, `bound[nt + t]`)
and indexes rows as `row0 + j`. Row arithmetic is the *only* difference — loop
trip counts, split chunking, merge order and the per-row op order are untouched,
which is what preserves the doc-94 bitwise identity between the verify batch and
sequential decode. Kernels: `attn_split_1w_body`, `attn_split_h4w_body`,
`gqa_attn_split_partial`, `_hybrid`, `_bt`, `gqa_attn_f32_f16kv`, `gqa_attn_f32`
and `fa_prefill_f16kv` (whose tile-wide extent is `max(hi)`/`min(lo)` so a query
tile must not span two sequences — E2's composition keeps a sequence
contiguous). `Op::Attn { multi_seq }` picks the pointer (positions vs span) *and*
the instantiation host-side, so a causal node never touches the span input;
`supports_attn_span()` is now `true` for CUDA, and the op matrix's asymmetric row
flips accordingly.

**The performance constraint, with evidence.** The requirement was "do not
affect existing CUDA performance", and there was no device to measure it on
(`cuInit` → 304, re-confirmed 2026-09-17: no seccomp, no container, the
580.178.04 module loaded, nodes present — all of it an artefact of the agent
sandbox, see A0; the wall-clock half of this section was added 2026-09-18). So the
evidence at the time was the generated code:
both revisions compiled with the project's own nvcc flags (`-O3`,
`-gencode arch=compute_121,code=sm_121`) and `cuobjdump -sass` compared per
kernel:

- **instruction counts identical** for every causal instantiation —
  `gqa_attn_split_partial` 448/416, `_hybrid` 1664, `_bt` 464/416,
  `gqa_attn_f32_f16kv` 1952, `gqa_attn_f32` 1520, `fa_prefill_f16kv` 2104 (the
  pre-E1b numbers);
- **opcode histograms identical** for `gqa_attn_split_partial[float]` and
  `gqa_attn_split_partial_hybrid[half]`; the others differ only by
  `LDG.E → LDG.E.CONSTANT` (the `bound` parameter is `const __restrict__`, so the
  loads take the read-only path — a caching upgrade, not added work) and by
  `MOV`/`CS2R`/`NOP` register-allocation substitutions of equal count.

No kernel gained an instruction, none lost one. What is *not* verified is
wall-clock time on a GPU: the windowed path is unreachable until E2 composes
batches, and the first GPU session should re-check both the numbers and the
timing (recorded in the status line).

**Tests.** `cuda_two_sequences_do_not_cross_attend` mirrors the CPU test at the
backend level (device-gated: it compiles here and skips without a device), and
the existing CUDA attention tests — including `cuda_verify_attention_nt_invariance`,
the doc-94 identity — call the causal instantiation as before.

**Device verification (2026-09-18) — the first execution on hardware.** With the
sandbox corrected (A0), E1b's windowed path was run on the GB10 it was written
for, and it is correct and free:
- **The E1b test could not have passed anywhere.** As written it used `hd = 2`,
  which the CUDA attention kernels reject (`attention head dim 2 outside the
  kernel's supported range (multiple of 4, 1..=128)`), and it read its result with
  the trait's `read_host`, which is `None` on CUDA by design (device memory cannot
  be borrowed). Both are fixed — `hd = 4`, `copy_to_host` — and the test now
  **passes on the device**: token 0 returns `V(0)`, token 1 returns `V(2)`, so a
  window that started at 0 (the pre-E1b derivation) would fail it. This is the
  window arithmetic itself, checked row by row.
- **The claim "no CPU behaviour change" holds**: E1b's diff touches only
  `src/cuda.rs`, `src/cuda_kernels.cu`, `src/graph/cuda_backend.rs` and the
  `op_matrix` support-table test, and the CPU suite is green (191 passed / 0
  failed / 5 ignored).
- **The claim "no causal-path performance change" now has wall-clock evidence**
  next to the SASS instruction counts: pre-E1b (`82f5109`) vs this branch, 7B
  Q4_K_M, same prompt, greedy, on GB10 — prefill 529.8 → 530.3 tok/s, decode
  51.2 → 51.1 tok/s, identical output text. The windowed instantiations cost
  nothing when they are not selected, exactly as the opcode histograms said.
- **The multi-sequence path is right with a real model.** New gate
  `batch_order_does_not_change_a_sequences_logits`: two sequences in one batch
  (batch shape, KV layout, history and graph all held fixed) must reproduce each
  sequence's logits **bitwise** when their order in the batch is swapped. It
  passes on the device — and it is only bitwise because the window follows the
  sequence and its reservation, never the row index.
- **Still open on the timing side (2026-09-18):** the *bitwise* equivalence is
  now asserted, but the windowed instantiation's **cost at the same width** was
  never measured — the GPU numbers in this record compare a 4-wide windowed step
  against a 1-wide causal step (6.5 vs ~19 ms/token), which mixes the width and
  the instantiation. A same-width A/B (windowed vs causal over identical rows,
  timed) is the remaining question, and it needs a device.
- **The causal-vs-windowed gap is closed (2026-09-18).** The record previously
  left this open: the windowed test exercises the row arithmetic but never the
  causal pointer against it. `cuda_causal_and_windowed_agree_on_the_same_rows`
  now runs the *same* queries, K/V and rows through both instantiations — a
  window that starts at cell 0, so `positions[t] + 1` and the explicit span name
  the same rows — and asserts the outputs are **bitwise equal** on the device. It
  passes on GB10, so the equivalence no longer rests on the SASS identity alone.
- **The f16 half of the windowed path was untested until 2026-09-19.** That gate
  forced `cb.kv_f16 = false` (f32 KV), while the production default is f16 whenever
  `n_layers * n_kv_embd >= 8192` (`cuda::set_kv_cache_type`): the 7B (14336) runs
  f16, the 0.5B (3072) f32. The sweep now loops over **both** dtypes and at once
  caught a real fault in `fa_prefill_f16kv`'s windowed row mask (`bound[t]` is the
  window's `lo`, not its `hi`) — plan §14 row 0 has the root cause, the fix and the
  end-to-end server evidence. With the fix, 22/22 cases (11 shapes x 2 dtypes) are
  bitwise equal on device, including the `hd = 128, n = 34, start = 64` prefill case
  the fault lived in.

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

**Follow-up (E2, the rounding-out of this note).** Option (a) assumed `n_seqs`
would be *made real* by item 3. Item 3 landed and the assumption was wrong in an
instructive way: the sequence count is **data**, exactly like `n_past`, and never
needed to be in the identity at all. What item 3 did add to the identity is
`CParams.explicit_span` — the one decision (which attention instantiation to
build) that depends on how a batch is *composed*, derived from the KV
reservations rather than from a count. So the field was deleted in E2 instead of
being populated: option (c), chosen late, with the measurement that makes the
case (`sequence_count_is_data_not_topology`). The general lesson this note now
records: an identity field earns its place only if some *topology* decision
reads it — "a future feature will need it" is not enough, because a redundant
field costs a rebuild every time it changes shape with the topology unchanged.

## 9. Phase F — independent tracks

Can run in parallel with A–E by a different workstream.

| ID | Item | Title | Effort | Box needed |
|---|---|---|---|---|
| F1 | 11 | AVX2/AVX-512 dots for the K-quants + weight repacking · [#56](https://github.com/yusiwen/minfer/issues/56) | L | **x86** |
| F2 | 15 | GBNF-style grammar + JSON-schema constrained decoding · [#47](https://github.com/yusiwen/minfer/issues/47) | M | this box |
| F3 | 16 | Sampler set: min-p, typical, XTC, DRY, mirostat, logit bias · [#48](https://github.com/yusiwen/minfer/issues/48) | M | this box |
| F4 | 12 | Backend registry (drop the compile-time enum) · [#57](https://github.com/yusiwen/minfer/issues/57) | M | this box |
| F5 | 14 | Async cross-backend copy + events · [#58](https://github.com/yusiwen/minfer/issues/58) | M | this box (CUDA) |
| F6 | 22 | Quantizer tooling (`convert-hf-to-gguf`, `quantize`, `split`) · [#49](https://github.com/yusiwen/minfer/issues/49) | L | this box |
| F7 | 19/20 | Chat-template fidelity + tokenizer generality · [#50](https://github.com/yusiwen/minfer/issues/50) | M | this box |
| F8 | 25 | **Metrics/observability** (`/metrics`, KV occupancy, queue depth, per-op timing under a flag, graceful drain). Item 25 was the only member of the A-era batch (items 23/24/26/27/28 -> A1/A2/A7/A5/A6) with no ticket; it is independent of the critical path, hence this table · [#51](https://github.com/yusiwen/minfer/issues/51) | M | this box |

F1 is the only item in this plan that **cannot be verified on this machine**
(aarch64): it needs an x86 box or a new CI runner. It is also the largest
single CPU win, so it should be scheduled against hardware availability, not
against the critical path.

## 10. Phase G — Metal alignment round (**scheduled**; device claims need a Mac)

Metal is a first-class target — it is the default backend on macOS and a plain
`cargo build --release` builds it — so this phase is **scheduled, not deferred**.
Only the *device* half waits: CI's `build-macos` job already compile-checks any
Metal change, and per the standing rule a Metal-only change is recorded as
compile-verified or left behind a build-time gate when it cannot be run here.

**Order and rationale.** G1–G3 first: each is small, independent of the KV semantics,
and removes a way Metal can be *wrong* (missing guard, `debug_assert!` on a release
path, a silent weightless-RMSNorm fallback). Then **G5 after C7/C8**, deliberately:
porting the cell store before the arena becomes elastic (C7) and shareable (C8) would
mean writing the same semantics into Metal twice. G4/G6/G7 follow G5.

| ID | Origin | Work | Position |
|---|---|---|---|
| G1 | A3 | `pos < n_ctx` guard in Metal's `KvcacheStore` · [#38](https://github.com/yusiwen/minfer/issues/38) | now |
| G2 | A8 | `debug_assert!` → `Err` for `FusedFFN`/`FusedQKV`/`FusedQkvNorm` `nt == 1` · [#39](https://github.com/yusiwen/minfer/issues/39) | now |
| G3 | A8 | Remove the silent weightless-RMSNorm fallback (`metal_backend.rs:403-414`, `:457-468`) · [#40](https://github.com/yusiwen/minfer/issues/40) | now |
| G5 | C1/C2/E1 | Port the cell store, the KV removal/shift and the explicit attention span to Metal (`supports_attn_span()` becomes true; today `copy_kv_to_cpu` has no Metal arm, so a Metal session re-renders instead of shifting, and a multi-sequence batch is refused outright) · [#44](https://github.com/yusiwen/minfer/issues/44) | after C8 |
| G4 | A8 | CUDA/Metal op-set asymmetry: decide whether Metal gains `QkvBiasRopeStore` · [#52](https://github.com/yusiwen/minfer/issues/52) | after G5 |
| G6 | E4 | Adopt the reserve/assign allocator split in Metal's pool · [#53](https://github.com/yusiwen/minfer/issues/53) | after G5 |
| G7 | METAL-OBJ | Re-run the Metal gap/parity measurements after G2–G3 (and again after G5), since each changes a kernel path · [#54](https://github.com/yusiwen/minfer/issues/54) | last |

**G5 acceptance** (on a Mac; the CPU/CUDA equivalents are the gates already in the
suite): two sequences do not cross-attend, bitwise; a mid-session compaction is
bit-identical; the C2 context shift matches CPU; and `MINFER_BATCH` unset may then
batch on Metal, which is what E6's Metal exclusion is waiting for.

Entry condition: G1–G3 need only a machine that builds Metal (CI's `build-macos`);
G5's device claims need a Mac. Exit condition: `SUPPORT-MATRIX.md`'s per-backend op
column matches `supports_op` on all three backends, with A1's matrix green.

## 11. Sequencing

```
Phase A  ├─ A0 ─ A1 ─┬─ A3 ─ A4 ─ A5 ─ A6 ─ A7 ─ A8 ──────────►  (A8 CUDA half)
         └─ A2 ──────┘
Phase B  ├─ B1 ─ B2 ─ B3                          (starts once A0/A1 exist)
Phase C  ├─ C1 ✔ ─ C2 ✔ ─────► C3 ✔ ─ C6 ✔ ─ C7 ✔ ─ C7b ✔ ─ C8a ✔ ─ C8b(S1a ✔ S1b ✔ S2 ✔ S3 ✔ S4 ✔ S5 ✔) ─ C4 ✔ (S1, CPU; S2 = #87) ─ C5 ✔   (C3 needed D1; the CUDA path first, per the 2026-09-20 decision)
Phase D  ├────────── D1 ─ D2 ─ D3 ──────────────►         (D unlocks MoE/MLA)
Phase E  ├──────────────────── E1 ✔ ─ E2 ✔ ─ E3 ✔ ─ E4 ─ E5        (E1b ✔ device-verified; E2 closed: CPU 0.49x, GPU 1.9x)
Phase F  └─ F2 F3 F4 F5 F6 F7 (parallel)        F1 = needs x86
Phase G  └─────────────────► G1 ─ G2 ─ G3 ─ G5 ─ G4 ─ G6 ─ G7   (after C8; G1–G3 compile-verified in CI; device claims need a Mac)
```

**Critical path:** A0 → A1 → C1 → C2 → E1 → E2 → C3 → C6 → C7 → C8 → G5.
**Deliberate exception to the roadmap's ordering:** A1/A2 run *before* the
hazard-removal tickets, because they are the instrument that proves those
tickets and everything after them.

## 12. What "done" means per phase

| Phase | Done when |
|---|---|
| A | **Complete 2026-09-16.** `cargo test` green on Linux/CPU (aarch64 locally, x86_64 in CI); A1's matrix green (or every red row explained); A0's CUDA verdict recorded; **each hazard ticket has a test that fails before and passes after**; A6 is closed by measurement instead — a refuted hypothesis with numbers is a result, not a gap. |
| B | **Complete 2026-09-16.** A multi-turn conversation prefills only the new turns (219 → 16 tokens, ≈11× TTFT); the contamination property is pinned by a bitwise test; numbers recorded interleaved with the same binary. |
| C | Cell store lands bitwise; shift is a documented tolerance class; **a single request may use the whole arena while the other slots are idle, even with a busy run above it (C7 + C7b ✔), and one cell range can be shared across sequences, counted once (C8)**; quantized KV behind its gate; session save/restore round-trips. |
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

## 14. Known gaps and open risks (2026-09-18)

Everything the campaign *measured* is recorded where it happened (the phase records
above). This section exists because a risk that lives only in a chat message is
not recorded at all: it lists what is **known to be unfinished or unverified**, in
one place, so the next session does not have to rediscover it.

| # | Gap / risk | Kind | Home / next step |
|---|---|---|---|
| 0 | **FIXED 2026-09-19, same day (found in the D3 round): the GPU-batched server corrupted concurrent responses** — root cause, fix and evidence at the end of this cell. On GB10, `MINFER_BATCH=1` with `--n-slots N` returns **one correct reply and N-1 identical copies of `0.15555555555555`** (N = 2, 3, 4 all reproduce; `MINFER_BATCH=0` serial gives 4/4 correct). The engine-level gate `server_batch_matches_serial_and_is_faster` **passes** with all four replies sane, so the batched *graph/engine* path is fine and the fault is in the **server** path (or in a layout it produces). The server's per-row trace shows `slot.start = n_ctx_total / n_slots`, so slots are reserved as huge runs (e.g. 4096 rows each at `--n-ctx 8192 --n-slots 2`) and every failing slot is one with a **large absolute KV offset** — while slot 0 (offset 0) is always the correct one. The engine test uses a 512-row arena, so it never exercises those magnitudes: **the server's layout, not the kernel tests, is what lacks coverage**. Ruled out so far: prefix reuse misfiring (logs show `40/40 prompt tokens, 0 reused` for every slot), admission placing two jobs on one slot (the `taken` map is per-slot and the traces show distinct `seq`/`slot` per row), and D2's new in-place `SwiGLU` (the corruption reproduces identically with `MINFER_FFN_NODE=1`, i.e. the hand-written node). Not yet pinned: whether the trigger is the *magnitude* of the offset, the span/kernel arithmetic at large row bases, or `n_ctx` itself — the follow-up experiment (vary `n_ctx` with `--n-slots 2`) was cut short by a harness `wait` bug, not by the code. **Narrowed the same day (D3 round 2):** it is **not** the layout magnitudes — `--n-ctx 128` (slot start 64) reproduces it — and **not** the engine: `server_batch_matches_serial_and_is_faster` pointed at the 7B (`MINFER_BATCH_TEST_MODEL`) gives four sane replies at `--n-slots 4`, while the *server* with the same model and two slots gives one sane and one garbage. It is **model-dependent**: Qwen2.5-7B Q4_K_M reproduces, Qwen2.5-0.5B Q4_K_M does not. CUDA Graph capture and prefill capture are **ruled out** (`MINFER_NO_CUDA_GRAPH=1` and `MINFER_NO_PREFILL_CAPTURE=1` both reproduce). The server's and the engine test's remaining structural difference is **staggered admission**: the server admits requests as they arrive, so its step sequence mixes widths (the trace shows a 1-sequence step, then 2-sequence steps), while the engine test admits all four before its first tick and only ever runs 4-wide steps. That was the leading hypothesis, and it is now **ruled out**: the ignored engine test grew a `stagger` arm that reproduces the server's pattern faithfully (admit one request, tick, then admit the rest, so the step sequence mixes widths) and with the 7B at `--n-slots 4` every staggered reply is as sane as its simultaneous counterpart (leading bytes shared: 55/41/66/33, same as the simultaneous run). So mixed step widths are not the trigger, and the engine test now pins that pattern either way. What remains between the server and the engine test: the *inputs* (the server renders the chat template, the test feeds raw `encode(prompt)`; the sampling params come from the request), the engine configuration (`n_ctx` 2048 vs 512, `n_slots` 2 vs 4), and the fact that the server keeps running across waves. **Root-caused to the windowed-attention kernel (same day, third round).** Bisecting the inputs did it, and the answer is none of the candidates above: with the engine test made configurable (`MINFER_BATCH_TEST_{SLOTS,CTX,TEMPLATED}`) and, crucially, made to render the prompt **exactly as the server does** (`template::render_messages` with the GGUF template + generation prompt — `model.format_chat` is a different path and produced a 13-token prompt where the server's is 34, so the first bisect compared unequal inputs):

| Config (7B Q4_K_M, GB10) | Result |
|---|---|
| 2 slots / 2048, **5-token raw** prompts | all four sane (batched *and* serial) |
| 2 slots / 2048, **34-token templated** prompts | slot 0 sane; slot 1 garbage in **both** batched and serial |
| 4 slots / 512, **34-token templated** prompts | slot 0 sane; slots 1-3 garbage, *identically*, in **both** batched and serial |

So it is **not** the server, **not** batching, **not** the slot layout or the offset magnitude, and **not** the model per se — it is the **prompt length combined with a non-zero start**. Slot 0 (start 0) takes the *causal* attention instantiation and is always right; every slot with a non-zero start takes the **windowed** instantiation, and it produces garbage once the window is ~34 rows rather than ~5. That is consistent with everything measured: model-dependent because the two models select different kernel variants (hd 128/n_kv 4 vs hd 64/n_kv 2), unaffected by batching (the kernel is chosen by `explicit_span` = "start != 0", not by batch width), and invisible to E1b's device tests, which only ever exercised tiny windows (a synthetic `hd = 4` fixture, and ~7-token sequences in the order-invariance gate).

**RETRACTED (same day, fifth round): the windowed kernel is NOT at fault — the
"minimal reproduction" above was a bug in its own harness.** `run` was called twice
(causal, then windowed) while it drew q/k/v from a **single advancing LCG declared
outside the closure**, so the two calls compared *different* random data. The one
assertion that kept passing, `causal == V(row 0)`, is a single-key-softmax identity
that holds whatever the data are — which is exactly why the failure looked like
"the windowed path returns the wrong numbers". Two probes settled it:

- A device `printf` in both instantiation entry points (`gqa_attn_split_partial`
  and `gqa_attn_split_partial_bt`) printed, for `n = 1, start = 0`: causal
  `bound[0]=0 -> row0=0, nkv=1` and windowed `bound[0]=0, bound[1]=1 -> row0=0,
  nkv=1` — **identical**, so the windowed path derived the right window from the
  right cells and the divergence had to be in the data itself.
- With the LCG re-seeded from the shape *inside* `run` (never from `start` or
  `explicit`, which are precisely what the two calls differ in), the whole sweep is
  **bitwise equal**: 11 cases over `(nh, nk, hd)` = (1,1,4)/(1,1,128)/(2,2,64)/
  (4,4,128), `n` = 1/2/16/34, `start` = 0/1/64/256. The test is un-`#[ignore]`d and
  is now the E1b/E2 gate it was meant to be (device run 2026-09-19: 11/11 bitwise
  equal, 249 other tests filtered out).

Consequences recorded here rather than silently dropped:

- The "root-caused to the windowed-attention kernel" paragraph above is
  **withdrawn**, and with it the **retraction of E1b's device evidence**:
  `cuda_causal_and_windowed_agree_on_the_same_rows` was called degenerate; it is
  weak on its own (one-hot queries make the output score-insensitive) but nothing
  contradicts it, and its note now says so instead of blaming the kernel.
- What the fourth round got right and keeps: the engine test *did* compare unequal
  inputs to the server's (`model.format_chat`'s 13-token prompt vs the server's
  34-token render), and it now renders the prompt exactly as the server does
  (`chat_template_from_gguf` + `template::render_messages`) — that part of the
  bisect stands and is an improvement to the test.
- **The server blocker itself is therefore unexplained again**, and the earlier
  chain of eliminations must be re-read with that in mind: it is *not* prefix
  reuse, admission placement, D2's in-place SwiGLU, layout magnitude, CUDA Graph or
  prefill capture, staggered admission, model-specific kernel variants, or the
  windowed instantiation. The reproduction is re-run on the current build (below)
  to establish whether the symptom still exists at all.

**Root cause found and fixed (same day, sixth round): the f16-KV FA prefill
kernel's windowed row mask.** The server blocker is real and still reproduced on
the current build — 4 different prompts, `--n-ctx 2048 --n-slots 4`, greedy: the
batched run returned slot 0 correct and slots 1-3 derailed (slot 3 literally
`0.1555555555555555555555`, the value in the original report), while
`MINFER_BATCH=0` returned 4/4 correct. Two facts broke it open:

1. **The dtype gate.** `cuda::set_kv_cache_type` chooses f16 KV whenever
   `n_layers * n_kv_embd >= 8192` — true for the 7B (28 x 512 = 14336), false for
   the 0.5B (24 x 128 = 3072). That *is* the recorded "model dependence". The new
   gate forced `cb.kv_f16 = false`, so **no f16 windowed path was ever tested**.
2. **The sweep extended to both dtypes** went red immediately at exactly one
   point: `kv_f16=true (4,4,128) n=34 start=64 -> DIVERGES, first=(512, ...)`.
   Index 512 is the first element of token 1's output — token 0 was bitwise
   correct, so the fault was per-row, not per-tile.

`nt >= 2 && hd == 128 && !MINFER_NO_FA_PREFILL` routes to `fa_prefill_f16kv`, whose
per-row limit was `qpos = bound[t]`. With an explicit span `bound[t]` is the
window's **`lo`**, not the causal upper bound, so every row kept only the `lo`
column (`win_lo <= col <= lo`). Token 0 came out right because its window *is*
`{lo}`. That is why the 7B's per-slot prefill (34-44 tokens, non-zero start) fed
the rest of the network from corrupted hidden states: its KV *and* its logits were
wrong from the first prefill, so every decoded token was garbage, while slot 0
(start 0, causal) stayed correct. The 0.5B never reproduced because it runs f32 KV,
whose prefill kernel (`gqa_attn_f32`) had the windowed mask right.

The fix is the exclusive per-row limit
`qlim = CAUSAL ? bound[t] + 1 : bound[nt + t]` (four mask sites). For the causal
instantiation `col <= bound[t]` and `col < bound[t] + 1` are the same integer
comparison, so the pre-E1 instantiation keeps its codegen and output; only the
windowed instantiation changes behaviour.

**Evidence after the fix:** the sweep is **22/22 bitwise equal** (11 shapes x both
KV dtypes, including the `n=34, start=64, hd=128` prefill case); the server
reproduction returns distinct, correct replies in both modes; the full CUDA suite
passes on the device. The engine acceptance gate
(`server_batch_matches_serial_and_is_faster`, 7B, 4 slots, `--n-ctx 2048`, templated
prompts) passes too — and it passed *before* the fix as well, which is itself the
lesson: its assertion is "batched text == serial text", and both paths ran the same
faulty prefill, so a **shared** fault is invisible to a differential gate. Only a
kernel-level sweep over the KV dtypes (plus the live server A/B with *different*
prompts) could see this one. What this also says about the earlier rounds: the single
*broad* structural fact they established still holds and was the useful half —
**the server exercises non-zero-start prefills that the engine test's 512-row
single-sequence arena does not**, which is why the gate now sweeps `start = 64..256`
and both dtypes.

Earlier text (kept for the record of how the diagnosis narrowed): a focused device test of the **windowed** instantiation with a *long* window at a non-zero start (34-64 rows) against the causal instantiation over the same rows — that should pin the exact kernel variant (the split/rows-per-warp bodies and the `_bt` variants are the candidates) and give a minimal reproduction, then the fix. The engine test as it stands is the end-to-end reproduction and now renders the server's prompt, so it fails the moment the bug is present and passes when it is fixed. Note also that this *corrects* the earlier "server-only, engine is fine" conclusion: that comparison used 5-token prompts on the engine side and 34-token ones on the server side. **Impact: E6 made batching the default on CUDA, so this is the default behaviour on a GPU server today**; E2's "GPU acceptance 1.9x" measurement inspected only request 0's text, so its *timing* stands but its *correctness* was never checked per request — that record is corrected here. **Immediate mitigation: moot, and it would not have worked.** Reverting E6's CUDA default (back to opt-in) was proposed while the cause was unknown; the f16 prefill fault lives in the *per-slot prefill*, which non-batched serving also performs (slots 1..n start at a non-zero KV cell), so `MINFER_BATCH=0` was affected too — it merely produced plausible-looking text instead of derailed text. The default is left on and is correct as of the fix recorded below. |
| 1 | **CI has no GPU.** The CUDA job only compiles the harness, so every device-gated test is a local, manual run — which is exactly how six device-only test bugs survived to 2026-09-18. | process | A **self-hosted runner on this DGX Spark** would put `cargo test --features cuda` into CI; nothing else does. Until then, anyone changing CUDA code must run it by hand and say so. |
| 2 | **The windowed instantiation's cost at equal width is unmeasured** (`cuda_causal_and_windowed_agree_on_the_same_rows` proves equality, not speed). | measurement | E1b record; needs a device A/B over identical rows at one width. |
| 3 | **A varying batch width rebuilds the graph** (`GraphCache` holds one graph at a time), so a server alternating 1-wide and N-wide decode steps re-allocates. | design | **DONE (E4 S3, 2026-09-23)**: the cache holds one graph per `GraphParams` and a switch re-maps the allocator onto it (liveness + slots, no build); a repeated chunked prefill builds nothing the second time (`a_repeated_chunked_prefill_stops_rebuilding`: 3 builds/5 reuses, then 3/9). |
| 4 | **The op matrix's support table does not parse `SUPPORT-MATRIX.md`** — it checks a Rust mirror, so a stale doc row stays green (it did, for `multi_seq`). | test hardening | A1/A8; recorded in the A1 record. |
| 5 | ~~**Batching is opt-in** even where it is measured faster.~~ **Closed by E6 (2026-09-19)**: the default follows the device (CUDA on, CPU/Metal off), with `=1`/`=0` as the override. | product | done |
| 6 | **`conversation_real_model_smoke` and `dump_real_q4k/q5k_tensor` are red** (ignored tests). Attribution done: the first fails identically on master + device, the others are pre-existing debug dumps. They are not gates, but a red ignored test is easy to mistake for noise. | pre-existing | Either fix their assertions/artifacts or mark them clearly in their doc comments; not caused by any PR in this campaign. |
| 7 | **Roadmap item 25 (metrics/observability) had no ticket** — the only orphan from the A-era batch. | planning | **F8**, added with this section. |
| 8 | **F1 (AVX2 K-quant dots) and all of Phase G need different hardware** (x86 / a Mac). They cannot be started, let alone verified, on this box. | hardware | Sequencing §11; F1 is the largest single CPU win. |
| 9 | **A sequence's logits' tail depended on its absolute arena offset — resolved by C6 (2026-09-19)** (found while gating C3). The pre-C6 measurements stand and are what justified the fix: at cell 0 vs cell 8 the max |Δ| over the vocabulary was 2.6% relative with the greedy token unchanged and the run deterministic; the hand-built `q`/`k`/`v` → rope → store → attn graph is exact to ≤ 1.2e-7 at the model's own shape *and* equal for a 1-cell and an 8-cell offset; the arena layout is irrelevant (a split reservation is bit-identical per layer); `positions` had exactly four consumers per layer (96 = 24×4); a layer bisect put the entry at layer 0's attention output; and the **rope-injection intervention** proved the entry is RoPE alone (injecting run A's 48 rope outputs into run B made the logits bitwise identical), while a distributed ~1e-6 rope perturbation already saturates the tail (0.44 vs 0.43) with the greedy token stable from 1e-6 to 1e-2. **C6 removed the coupling** — `positions` are sequence-relative and the allocator resolves `cells` — so a cell move changes no angle: the offset tests now assert bitwise equality, C3's acceptance tightens from the named amplified-rounding class to bit-identical, and a compaction no longer re-ropes. Three method notes earned here: a zero from a perturbation probe means nothing without a loud control; **a control validates the path, not the equivalence of the perturbation** (a single element nudged by 1e-5 is not the offset's distributed 1.5e-5 — the earlier "refutation" was an over-read); and **an intermediate buffer may only be read immediately after its own node runs** (`graph.outputs` does not extend liveness). | measurement | Done by C6; the logical-positions design, its gates and the CUDA fused-op port (S3) are in §5. |
