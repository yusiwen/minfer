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
**7/7** (E1, E1b, E2, **E3**, **E4**, **E5**, E6 all done); Phase F **7/8** (F2, F3, **F4**, **F5**, **F6**, F7,
F8 done; F1 needs x86); Phase G
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

### C4 — Quantized KV cache, Q8_0 first · [#42](https://github.com/yusiwen/minfer/issues/42) — **DONE (CPU: S1 + S2a; CUDA: S2b) 2026-09-24**

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
CI. The real-model gate is `#[ignore]`d (at this record's time it flipped a process-wide KV policy;
since [#99](https://github.com/yusiwen/minfer/issues/99) the format is per engine and it no longer
mutates shared state — see the #99 record below) and was run on
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

**C4 S2b — the CUDA Q8_0 kernels: landed 2026-09-24 ([#87](https://github.com/yusiwen/minfer/issues/87)).**
The subsection below is the plan of record as it was written *before* the kernels; the design
decisions, the measured results and the honest scope are recorded after it, so the plan and the
outcome can be read against each other.

**The plan of record (2026-09-24, written before the first line of kernel code).**

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
gate `a_packed_kv_cache_answers_like_the_f32_one` run with the `cuda` feature on GB10 (since
[#123](https://github.com/yusiwen/minfer/issues/123) that gate is CPU-forced on a CUDA build until
these kernels land, so this acceptance includes re-pointing it at the device; CI has no
GPU, and its `MINFER_C4_MODEL` arm covers a second model); and an A/B against f16 on the same
model/context with the numbers recorded. The honest expectation, stated before the work: the win
on the device is **memory** (3.76× less than f32, 1.88× less than f16); whether decode bandwidth
also improves depends on the block-dequant instruction count, so the bar against f16 is "no worse
than a named factor", not "faster".

**S2b design decisions, frozen before the first line of kernel code (2026-09-24).** The plan above
is the map; these are the choices it leaves open, decided up front so the implementation cannot
drift into a silent fallback:

1. **Layout tag and accessors.** `KV_LAYOUT_F32 = 0 / KV_LAYOUT_F16 = 1 / KV_LAYOUT_Q8_0 = 2` are
   `#define`s in `src/cuda_kernels.cu`, and **`0/1/2` are a host contract**: the Rust side stores the
   same codes (the registry's `KvFormat` discriminants) and every launcher takes the tag as an
   `int`. Two accessors are the *only* place a KV address is formed:
   `kv_row(const void* base, int64_t cell, size_t row_bytes)` (byte address of a cell) and
   `kv4<LAYOUT>(const char* row, int elem) -> float4` (the single load idiom). `F32` is the old
   `float4` load and `F16` the old two-`__half2` pair, both **bit-identical** to today's
   `kv_ld4<float>` / `kv_ld4<__half>`; `Q8_0` reads block `elem/32`'s f16 scale plus four quants at
   `2 + elem%32`. A 4-element group never straddles a block because a KV head's base is `hd`-aligned
   and `hd % 32 == 0` — the property `ensure_kv`'s packed-width check already enforces.
2. **No typed pointer survives.** Every attention kernel that can see a packed row takes
   `const void* k, const void* v` plus `size_t row_bytes`, and templates on `int LAYOUT` instead of
   `typename KV`. The `row_bytes == nk * hd * 4` value the f32 path passes is the same arithmetic as
   the old `stride_kv * sizeof(KV)`, so the f32/f16 instruction streams are unchanged.
3. **Dispatch cuts, each stated when the format is enabled** (build-time; a packed row never reaches
   a kernel that would address it as f32):
   - decode `nt == 1` → the converted split-K 1-warp body (`gqa_attn_split_partial`), with
     `rpw_gate = 0`: the hybrid 4-warp dispatch is f16-typed (`attn_split_h4w_body` takes
     `const __half*`) and is not converted in this increment;
   - `1 < nt <= 16` → `gqa_attn_f32`. The batched split kernel (`gqa_attn_split_partial_bt`) exists
     for spec-verify's **bitwise** identity contract with sequential decode; rather than claim that
     contract without measuring it, a **speculative session refuses a packed cache loudly** (the
     draft already keeps its own KV, and C5 already refuses a draft session);
   - prefill `nt > 16` → `gqa_attn_f32`. The FA path (`fa_prefill_f16kv`, f16-typed shared-memory
     staging and tensor-core QK^T) is not offered for Q8_0, so the prefill is **correct but off its
     tuned path** — measured and reported, not hidden;
   - the fused decode epilogue (`Op::FusedQKV` / `Op::QkvBiasRopeStore`) is **not built** for a
     packed cache: the model builders' `layer_gpu` gate gains `&& !packed`, so a Q8_0 decode runs
     the unfused bias → rope → store chain through the converted `store_kv_q8_0`.
4. **The store is the CPU's quantizer, byte for byte.** `store_kv_q8_0` maps one thread to one
   `(row, 32-element block)`, computes `amax`, `d = amax/127` as an f16 (nearest-even), and
   `round_ties_even` for each quant — the same three steps `quants::quantize_row_q8_0_into` uses —
   and writes `d` then the 32 quants into the packed cell at 34-byte stride. Both backends therefore
   store the same bytes for the same f32 row, which is what makes the CPU/device Q8_0 comparison a
   layout check rather than a tolerance question.
5. **The gate flips only after the kernels exist, and through the registry.** `READS_PACKED_KV`
   stays `false` until every dispatch cut above is in place; then it becomes `true` for CUDA and
   `KvFormat::supports(Cuda)` follows automatically (`registry::reads_packed_kv` is the one
   authority). `CudaBackend` gains an `int` layout field fed by the *resolved* `KvFormat`, so
   `MINFER_CACHE_TYPE=q8_0` can never again mean "f32 with a packed region".
6. **Acceptance bar, named before measuring.** The device win is **memory**: a Q8_0 cell is `34/128`
   bytes per element versus `4` (f32) and `2` (f16), i.e. **3.76x less than f32 and 1.88x less than
   f16**. Whether *speed* also improves depends on the block-dequant instruction count in the load
   path, so the bar against f16 is stated as a factor, not as "faster": **Q8_0 decode tokens/s must
   be no worse than `1/1.30` of the f16 decode rate on the same model and context** (the same 1.30x
   already recorded for the CPU's fused Q8_0 read at ctx 2048), and the prefill is reported as its
   own number because it takes the untuned `gqa_attn_f32` route.

**Why it was not in the S2a increment.** ~10 kernel sites, 3 launchers, ~15 host sites and 41 test
sites, each needing an nvcc iteration and — for the gates — a serial device run. It was recorded
here rather than half-wired. [Metal's half stays at G5](https://github.com/yusiwen/minfer/issues/44).

**What actually landed (2026-09-24).** The real counts are close to the estimate: **8 kernel
sites**, **3 launchers** (`launch_gqa_attn_split_q8_0`, the layout-tagged `launch_gqa_attn_f32`,
`launch_store_kv_q8_0`) plus two re-templated existing ones, ~25 host sites (`cuda.rs`'s FFI +
`CudaState` methods, `cuda_backend.rs`'s store/attention/fused-epilogue dispatch, `copy_cells`, the
registry entry), 4 new gates and 2 strengthened ones. The two files the plan expected not to change
did not: `kv_move_rows` is a plain word copy (verified — the packed cell's word count is what the
host already passes, and the Q8_0 move gate pins it), and the CPU path is untouched.

- **Layout and accessors, exactly as designed.** `KV_LAYOUT_F32/F16/Q8_0` are `0/1/2` on both sides
  of the FFI; `kv_row` + `kv4<LAYOUT>` are the one address/load idiom. The f32 and f16 kernels
  compile to their pre-C4 instructions (same loads, same cells, same order).
- **The gate flipped through the registry.** `BackendCaps::reads_packed_kv` became `true` for CUDA
  and `KvFormat::supports(Cuda)` followed automatically; `CudaBackend` carries an `int kv_layout`
  fed by the resolved `KvFormat`, so the pre-S2b `bool` that mapped anything but `f16` to f32 can no
  longer turn a packed region into f32 rows.
- **The store is the CPU's quantizer, byte for byte.** `cuda_q8_0_store_matches_the_cpu_quantizer`
  asserts the device's packed words equal `kvformat::pack_q8_0_cell`'s for two block widths, and
  that unwritten cells stay zero. Mutation: `amax / 126` instead of `/ 127` fails it at `nkt=64 row
  0`.
- **All three acceptance gates, mutation-checked.** `cuda_map_window_matches_the_span_over_the_same_rows`
  now sweeps **f32/f16/q8_0**; `cuda_kv_q8_0_roundtrip_attn` (store → attention) and
  `cuda_q8_0_kv_cell_move_strides_by_row_bytes` (`copy_cells` under the packed stride) are the two
  new device gates. `compute-sanitizer --tool memcheck` over all three reports **no memory error**
  (the 2 pre-existing CUDA API errors it did report were root-caused in S2c below, [#145](https://github.com/yusiwen/minfer/issues/145)). Mutations: dropping the Q8_0 block base in `kv4` left the map-window gate
  **green** (every comparison there is *between* modes over the same bytes, so a value-level fault
  shifts both sides together — the F6 lesson) and made the round-trip and the **real-model device
  arm** fail (max |Δlogit| 2.48 → 26.34 of a 37.8 spread); halving the packed stride in
  `copy_cells` fails the move gate; flipping `READS_PACKED_KV` back to `false` makes the device arm
  refuse the region loudly. The map-window gate was **strengthened**: it now also compares one
  single-row Q8_0 window against the dequantized V cell (max |Δ| 1.71 under the block-base
  mutation), so it cannot pass on consistency alone.

**The A/B against f16, measured on GB10 (sm_121)** — the bar was named before the work as "Q8_0
decode no worse than `1/1.30` of f16" and **it is not met against the fused f16 baseline**:

| model / config | f16 (default policy) | q8_0 | factor |
|---|---|---|---|
| Qwen2.5-0.5B q4_0, `pp2048` @ n_ctx 4096 | 2693.54 tok/s | 2171.85 tok/s | 1.24x slower |
| Qwen2.5-0.5B q4_0, `tg128` @ n_ctx 4096 | 239.76 tok/s | 162.46 tok/s | **1.48x slower** |
| …f16 with the fused epilogue cut (`MINFER_NO_FUSE_QKV=1`) | 203.58 tok/s | — | the Q8_0 kernel's own cost is **1.25x** |
| Qwen3-0.6B Q8_0 (hd 128), `pp2048` | 8604.56 tok/s | 562.76 tok/s | **15.3x slower** |
| Qwen3-0.6B Q8_0 (hd 128), `tg128` | 137.85 tok/s | 122.21 tok/s | 1.13x slower |

Read honestly: decode on the 0.5B misses the named 1.30x because ~1.18x of the gap is the **stated
cut** (no fused QKV epilogue for a packed cache — the unfused chain adds launches on a model whose
decode is launch-bound) and the remaining **1.25x** is the packed load itself (`kv4<Q8_0>` costs
four int8 converts and four multiplies where f16 costs two `__half2` converts; there is no dp4a
packed dot in this increment). On Qwen3-0.6B, where the fused epilogue is not on the f16 default
path in the same way, decode is 1.13x — inside the bar. The prefill is a different story: at hd 128
the f16 path is `fa_prefill_f16kv` (**8604** tok/s) and the packed path is the general kernel
(**563** tok/s), i.e. the packed cache is correct but off its tuned route by 15x. **The win is
memory, as stated up front**: measured on both models, the f32/f16 region is 6 291 456 B and the
Q8_0 region 1 671 168 B — **3.76x smaller than f32 and, against f16's actual 2 B/element payload
(3 145 728 B), 1.88x smaller.**

**Acceptance results.**

- *Real model, device arm*: `a_packed_kv_cache_answers_like_the_f32_one` runs **both** arms — the
  CPU one (kept, coverage on every build) and, on a CUDA build with a device, one that asserts
  `device() == Cuda`. 0.5B: f32 6 291 456 B vs q8_0 1 671 168 B (3.76x), CPU max |Δlogit| 3.03 /
  argmax 0.65, CUDA 2.48 / 0.60 of a 37.8 spread; the physical-shift arm: CPU 2.47 / 0.064, CUDA
  3.34 / 0.42 — inside the bounds the CPU gate had already fixed.
- *Suites*: CPU `cargo test --release` **432 / 0 / 28** unit + **10 / 0 / 6** integration
  (unchanged); CPU serial ignored **28 / 0**; CUDA serial unit **490 / 0 / 31** (was 486/0/31: three
  new device gates + one new `cuda.rs` layout test; the packed gate now also runs on the device);
  CUDA serial ignored 0.5B **31 / 0**; Qwen3-0.6B **30 / 1**, the one failure the then-pre-existing
  [#130](https://github.com/yusiwen/minfer/issues/130) f16 session-container gap (closed in C5 S3:
  **31 / 0**).
- *The Q8_0 session container works.* `a_session_resumed_from_disk_continues_bitwise` with
  `MINFER_CACHE_TYPE=q8_0` on the device prints `live KV format: q8_0`, saves 24 layers / 256 cells
  / 5 written / 1 696 464 B and continues **bitwise** (max |Δlogit| = 0). The container's
  `FLAG_PACKED` bit is what encodes it; f16 had no flag then — the gap
  [#130](https://github.com/yusiwen/minfer/issues/130) closed in C5 S3 (`FLAG_F16`), below.

**Handed off after S2b, still on [#87](https://github.com/yusiwen/minfer/issues/87):** the packed
**fused decode epilogue** (a block-quantizing store inside `attn_bias_rope_store` would recover the
1.18x on the 0.5B) and a **dp4a packed K dot** (the 1.25x), then the FA prefill on packed cells
(the 15x at hd 128). [Metal's half stays at G5](https://github.com/yusiwen/minfer/issues/44).

### C4 — S2c: the latched CUDA API errors behind the phantom "kernel launch error" · [#145](https://github.com/yusiwen/minfer/issues/145) + [#128](https://github.com/yusiwen/minfer/issues/128) — **DONE 2026-09-25**

**Why.** `minfer bench` on a CUDA build printed `CUDA kernel launch error: 1` between its two loops —
pre-existing on master, not an S2b regression — and `compute-sanitizer --tool memcheck` over the unit
suite reported **36** CUDA API errors. `CudaState::sync` polls `cudaGetLastError`, which reports
whatever an *earlier* call on the thread latched, so an API error from several operations ago was
printed as a failure of the kernel that had just run.

**Root cause: two origins, both unchecked return values.**

1. **`gemm_prefill_smem_init` (1 of the 36; [#145](https://github.com/yusiwen/minfer/issues/145)).**
   The eager dynamic-smem opt-in looped over every `gemm_f16_nt_kernel_t<tm, ks, af32>` and asked
   `cudaFuncSetAttribute(.., cudaFuncAttributeMaxDynamicSharedMemorySize, N)` for a stale `N`: its
   formula assumed a 512-thread `TM=256` launch and always added the AF32 mirror, so
   `gemm_f16_nt_kernel_t<256,64,true>` requested **131072 B** against GB10/sm_121's
   `cudaDevAttrMaxSharedMemoryPerBlockOptin` of **101376 B**. The rejected call's return value was
   never read, so the error latched. The *same* formula existed a second time, inline in
   `launch_gemm_f16`, and **that** copy dropped the AF32 mirror — the launcher declared 16384 B less
   than the kernel's own `Bs`/`Cs` offsets need at the default `KS=32`, an out-of-declaration
   shared-memory access that only worked because the block's smem happened to be carved where nothing
   else wrote.
2. **`cudaGraphDestroy` on a `cudaGraphExec_t` (26 of the 36; [#128](https://github.com/yusiwen/minfer/issues/128)).**
   `CudaState::graph_destroy` is the only destroy path for an exec handle and called the *graph*
   destructor. It returned `cudaErrorInvalidValue`, leaked the exec, and the latch surfaced at the
   next sync — **the actual source of the bench line**. The remaining 9 errors were
   `cudaGetLastError` observations of one of the two.

**What landed.**

- **One formula, read twice.** `gemm_dynamic_smem_bytes(tm, ks, af32)` is the single source for the
  kernel's byte layout (`As 2*TN*KS halves + Am 2*TN*KS floats [AF32] + Bs 2*TM*KS halves + Cs
  NW*256 floats`, TN = 64, NW = `blockDim.x/32` = 8 at every launch site); `launch_gemm_f16` and
  `gemm_prefill_smem_init` both call it.
- **The eager opt-in is checked and deliberate.** Every `cudaFuncSetAttribute` return value is read;
  a failure is named once, at init, with the function, attribute, requested bytes, device limit and
  `cudaGetErrorName`, and cleared there. A request above the **queried** device limit is skipped
  without calling it, with the reason printed — on GB10 that is `gemm_f16_nt_kernel_t<256,64,true>`
  (122880 B > 101376 B), which cannot launch on this device at all. `CudaState::try_new` reports the
  counts.
- **`sync` attributes honestly.** `latched_api_error_message` names the observer
  (`cudaGetLastError`), the `cudaGetErrorName` symbol and the code, and states it is not attributed
  to a kernel; the error is counted (`latched_api_error_count`) and cleared — still visible, never
  dropped. `debug_sync`'s label follows.
- **`graph_destroy` calls `cudaGraphExecDestroy`**; the `cudaGraph_t` / `cudaGraphExec_t` distinction
  is stated at the extern and the call site, and a failure is named there and cleared.

**Acceptance results (GB10, sm_121, CUDA 13.0, driver 580.178.04).**

| check | before | after |
|---|---|---|
| `compute-sanitizer --tool memcheck` over the serial CUDA unit suite | **36** API errors (26 `cudaGraphDestroy`, 9 `cudaGetLastError`, 1 `cudaFuncSetAttribute`) | **0** errors |
| CUDA serial unit suite | 490 / 0 / 31 | **495 / 0 / 31** (five new gates) |
| CUDA serial ignored, 0.5B config | 31 / 0 | **31 / 0** |
| CUDA serial ignored, Qwen3-0.6B Q8_0 | 30 / 1 | **30 / 1** at this commit — only [#130](https://github.com/yusiwen/minfer/issues/130), closed in C5 S3 (**31 / 0**) |
| `minfer bench -p 64 -n 8 -r 2`, 0.5B Q4_K_M | prints `CUDA kernel launch error: 1` between the two loops | no such line; the one skipped opt-in named instead |
| CPU `cargo test --release` | 432 / 0 / 28 unit + 10 / 0 / 6 integration | **unchanged** |
| CPU serial ignored | 28 / 0 | **28 / 0** |

**The five new gates, all mutation-checked.**

- `the_latched_error_message_never_blames_a_kernel` (pure) — the message names `cudaGetLastError` and
  `cudaErrorInvalidValue` and does **not** contain "kernel launch". Mutation: the old message fails it.
- `the_gemm_smem_formula_matches_the_kernel_layout` (pure) — pins `gemm_dynamic_smem_bytes` against
  the kernel's byte layout for all 12 `(tm, ks, af32)` combinations. This is the *value-level* arm the
  F6 lesson asks for: a mode-vs-mode comparison is blind to a shrunk formula (launcher and init shrink
  together, so a consistency check stays green). Mutation: dropping the AF32 term fails it.
- `cuda_prefill_smem_optin_covers_every_launchable_instantiation` (device) — every >48 KiB request the
  device admits reads back opted in through `cudaFuncGetAttributes().maxDynamicSharedSizeBytes`; every
  over-limit one is skipped, never called. Mutation: `continue`-ing one admitted combination fails it.
- `cuda_graph_exec_destroy_leaves_no_latched_error` (device) — capture → instantiate → destroy
  returns `true` and leaves the latch at 0. Mutation: `cudaGraphDestroy` fails it. The gate asserts
  the destroy **call's own result** (`graph_destroy` now returns `bool`), not just the latch: the
  failure is named and cleared inside `graph_destroy`, so a latch-only assertion passed with the
  wrong destructor — the first cut of this gate passed for the wrong reason and was fixed here.
- `cuda_sync_surfaces_a_latched_error_as_latched` (device; env-gated behind
  `MINFER_TEST_LATCH_ERROR=1` because it *deliberately latches a real API error*, which a
  `compute-sanitizer` run must not see) — sync reports the injected error once and clears it.
  Mutation: dropping the report fails it.

The >48 KB capture path stays covered by the existing prefill gates — prefill capture is ON by default
(R3-B) and `cuda_prefill_capture_defaults_on`, `cuda_prefill_capture_bit_parity_pp16_pp300`,
`cuda_multisplit_capture_bit_parity` and the real-prefill
`cuda_graph_generation_replay_parity_real_model` are all green in the 495 / 0 / 31 run.

**The deliberate-failure check.** The pre-fix over-limit request was forced back with the
sanitizer-clean skip bypassed, so `cudaFuncSetAttribute` really failed: the init printed
`cudaFuncSetAttribute(gemm_f16_nt_kernel_t<256,64,true>,
cudaFuncAttributeMaxDynamicSharedMemorySize, 131072 B) failed: cudaErrorInvalidValue (1); device
opt-in limit 101376 B`, and `cuda_prefill_smem_optin_covers_every_launchable_instantiation` failed
(failures = 1). Reverted; the reverted `src/cuda_kernels.cu` is byte-identical to the committed one
(`sha256sum`, in the closing comment on #145).

**Honest scope.** The device limit — and therefore *which* instantiation is skipped — is measured on
GB10/sm_121 only; the decision is a runtime query and the skip is printed, so another device skips a
different subset. `MINFER_GEMM_TM=256` + `MINFER_GEMM_K64=1` + the f32-A path is the only combination
that lands on the skipped instantiation, and it had no working opt-in before either. The remaining
unchecked CUDA calls in the same file are enumerated in
[#147](https://github.com/yusiwen/minfer/issues/147) rather than silently fixed here.

### C4 — S2d: the remaining unchecked CUDA attribute / launch / destroy returns · [#147](https://github.com/yusiwen/minfer/issues/147) — **DONE 2026-09-25**

**Why.** S2c ([#145](https://github.com/yusiwen/minfer/issues/145)) fixed the two latched-error
origins `compute-sanitizer` could still see and left the CUDA unit suite at **0 API errors** — but the
audit it performed listed a whole family of call sites that still *discard a return value which gates a
later launch or allocation*, the class that let [#122](https://github.com/yusiwen/minfer/issues/122),
[#128](https://github.com/yusiwen/minfer/issues/128) and #145 hide in the first place. None was known
to fail on sm_121, so this is latent hardening, not a live bug: the value is that the next driver,
device or tile-config change cannot turn one of them into another phantom "kernel launch error".

**The re-derived site list (verified against the tree, not the ticket).** The S2c table named six rows;
re-deriving them against the post-#145 source gives **eight code sites**, one of which the table
understated (the wide-NT launcher already read its own return but reported nothing) and two of which
were already fixed by #145 (the eager `gemm_prefill_smem_init` opt-in and `graph_destroy`'s
`cudaGraphExecDestroy`). What remained, and how each is closed:

| site | was | now |
|---|---|---|
| `launch_mmq_raw_nt` (`kd <= 4` and `kd > 4`) | `cudaFuncSetAttribute`'s return discarded; the launch followed unconditionally; the `void` launcher reported nothing and the caller could not tell | `minfer_smem_optin` reads and names the call, the following launch is refused (`return 0`), and the launcher's own `<<<>>>` error is read too (`minfer_launch_ok`); `launch_mmq_raw_nt` returns `int` and the Rust caller turns a 0 into an `Err` |
| `launch_mmq_nt` (`MMQ_LAUNCH`, one call per quant type) | same, as a macro | the macro branches on the named opt-in, refuses, and returns the launch's own result; the launcher returns `int` → `Err` |
| `launch_mmq_raw_nb_nt` | the return was not read; a following `cudaGetLastError()` treated *any* latch as "smem/reg cap, return 0", discarding the code and its origin | the opt-in's own return decides (`attr:mmq_raw_nb`), the launch's own error is named (`launch:mmq_raw_nb`), the 0-fallback contract is unchanged |
| `launch_mmq_raw_nb_bt_nt` / `..._q6k_nt` | same post-hoc `cudaGetLastError()` idiom, attribution by position | the same named opt-in + launch check (`attr:…`, `launch:…`), cleared at the site |
| `launch_mmq_raw_wide_nt` (`kd <= 4` and `kd > 4`) | the return *was* read, but a failure cleared the latch silently (no name, no value) | routed through the shared named helper: site, instantiation, requested bytes, device limit, `cudaGetErrorName` |
| `gemm_smem_optin` + `launch_gemm_f16` | the helper printed a failed opt-in but the caller launched anyway; `launch_gemm_f16` returned `void`, so its own launch was unchecked (`launch_gemm_f32a` checked it via a post-hoc `cudaGetLastError`) | `gemm_smem_optin` returns the admitted/refused answer (cached per instantiation), the launcher refuses on false, reads its own launch error and returns `int`; `launch_gemm_f32a` forwards it; `prefill_gemm_f16_inner` returns `Err` |
| `graph_end_capture_to_exec` | `cudaGraphDestroy(graph)`'s return discarded | read, named with `cudaGetErrorName`, cleared at the site; the legacy `graph_end_capture` gets the same read |

**The mechanics (one place, no per-launcher copy).** `cuda_kernels.cu` gained a small block of shared
helpers — `minfer_smem_optin` (skip an over-limit request **without calling**, otherwise call once, name
the function/attribute/bytes/device-limit/`cudaGetErrorName`, clear the latch, return false → refuse),
`minfer_launch_prelude` (report a latch that *predates* the launch, so the post-launch read is a launch
check and not attribution by position), `minfer_launch_smem`, and `minfer_launch_ok` (name the launch's
own error and clear it) — plus `minfer_site_fail_*` introspection for the gates. `minfer_smem_optin`
queries `cudaDevAttrMaxSharedMemoryPerBlockOptin` once and skips a request above it, which is what keeps
`compute-sanitizer` clean (calling it would only observe `cudaErrorInvalidValue`).

**Two signature changes, both followed through.** `launch_mmq_raw_nt` and `launch_mmq_nt` went
`void → int`, and `launch_gemm_f16` / `launch_gemm_f32a` went `void → int` (1 = launched and accepted,
0 = refused). The Rust callers read the result and return `Err` — a failed launch is an error, never a
silent pass over an unwritten output. The *fallback* launchers keep their documented `0 = clean
fallback` contract, so which kernel runs is unchanged. Nothing numeric or dispatch-related moved.

**A measured correction to a documented belief.** `docs/CUDA-BACKEND-DESIGN.md` said
`cudaFuncSetAttribute` is illegal under `cudaStreamCaptureModeGlobal` and that a lazy opt-in inside a
window fails. A probe on CUDA 13.0 / driver 580.178.04 / sm_121
(`/tmp/fix147_attr_capture_probe.cu`) shows the call now returns `cudaSuccess` inside an open Global
capture window (both for the set value and a new one), so the eager init is defence-in-depth rather
than the only working form; the launcher caches one answer per instantiation, so it is not re-asked on
the hot path either way. Recorded in the design doc.

**Acceptance results (GB10, sm_121, CUDA 13.0, driver 580.178.04).**

| check | before | after |
|---|---|---|
| CUDA serial unit suite | 503 / 0 / 32 | **508 / 0 / 32** (five new gates: three pure, two device/env-gated) |
| `compute-sanitizer --tool memcheck` over the serial CUDA unit suite | **0** API errors over 503 (349.46 s) | **0** errors over 508 |
| CUDA serial ignored, 0.5B config | 32 / 0 | **32 / 0** |
| CUDA serial ignored, Qwen3-0.6B Q8_0 | 32 / 0 | **32 / 0** |
| CPU `cargo test --release` | 440 / 0 / 29 unit + 10 / 0 / 6 integration | **unchanged** |
| CPU serial ignored | 29 / 0 | **29 / 0** |
| `scripts/check_docs_links.py` | 940 links / 184 files | **940 / 184** |

**The five new gates, all mutation-checked.**

- `the_graph_destroy_failure_message_names_the_matching_destructor` (pure) — the Rust formatter names
  `cudaGraphDestroy`, the `cudaGraph_t from cudaStreamEndCapture` handle, the `cudaGetErrorName` symbol
  and the ticket, and must **not** name `cudaGraphExecDestroy` or "kernel launch" (a gate that only
  asserts "a message appeared" cannot see a wrong message). Mutation: naming the exec destructor fails it.
- `the_injection_matcher_matches_only_the_named_site` (pure) — `all`, exact comma-separated tokens,
  whitespace-tolerant, and **no substring rule in either direction** (`small` must not arm
  `attr:mmq_nt`; `attr:mmq_nt_extra` must not either). Mutation: the substring form fails it.
- `cuda_issue147_attribute_sites_name_the_call_and_refuse_the_launch` (device; env-gated behind
  `MINFER_TEST_ISSUE147=1` because it makes a **real** CUDA call fail) — for each of the ten
  dynamic-smem tokens the injected over-limit `cudaFuncSetAttribute` is reported once, at that site,
  with the API, the attribute, the device limit, a request above it, the exact instantiation and
  `cudaErrorInvalidValue`; the launcher returns 0; `take_last_error()` is 0. Then a positive control
  (knob off) launches and leaves no latch.
- `cuda_issue147_launch_sites_name_the_call_and_refuse_the_launch` (device; env-gated) — the same for
  the ten launch tokens (an over-limit dynamic-smem `<<<>>>`, which the launch call itself rejects with
  `cudaErrorInvalidValue` — probed before use, `/tmp/fix147_launch_fail_probe2.cu` — so the kernel never
  runs), plus a positive control.
- `cuda_issue147_graph_destroy_failure_is_named_and_the_exec_survives` (device; env-gated) — the
  injection re-creates #145's bug at this site (`cudaGraphDestroy(exec)`), the site clears the latch,
  and the valid exec is still returned and still destroyable (a failed graph destroy leaks only the
  graph handle, so refusing the exec would be wrong).

**Mutation evidence.** Every hardening was reverted one at a time and the corresponding gate re-run;
each reverted version failed, the file was restored, and the restored files were byte-identical
(`sha256sum`, in the closing comment on #147). Round 1 (per-site attribute guards): A1
`attr:mmq_raw_nt_kd4`, A2 `_kd8`, A3 the `mmq_nt` macro, A4 `attr:mmq_raw_nb`, A5 `attr:mmq_raw_nb_bt`,
A6 `attr:mmq_raw_nb_bt_q6k`, A7 `attr:mmq_raw_wide_kd4`, A8 `_kd8` — all eight trivially failed the
attribute gate at that site. Round 2: A9 the gemm opt-in guard, B the shared `minfer_launch_ok` check,
C the gemm message naming a wrong instantiation (the "prints a wrong message" check), D
`minfer_smem_optin` reporting but admitting the launch (the shape a message-only assertion would miss —
the gate's `ret == 0` catches it), E the Rust destroy read discarded, F the Rust formatter naming the
wrong destructor, G the injection matcher weakened to a substring.

**Honest scope.** The over-limit skips and the device limit are measured on GB10/sm_121 only; the
decision is a runtime query and every skip is printed, so another device skips a different subset. All
deliberate-failure evidence is device-gated and driven by test-only env knobs, which invert a site's
*own* answer rather than exercising a genuinely failing driver call in production — the injection makes
the call fail for real (over-limit attribute / over-limit dynamic smem / wrong destructor), so the
"latch cleared" half is exercised for real, but the production paths remain latent by construction. The
MMQ launch check is shared by the `mmq_nt` and raw-NT launchers, so the per-launch *check* was mutated
once (B) rather than per launcher; every launch token's own report was observed by the gate. Other
`void` launchers in the file — **65** of them (`launch_dequant_f16`, `launch_convert_f16`,
`launch_gemm_qb_nt`, every store/rope/attention/MVQ/MVQ-multi wrapper) — still do not read their own
`<<<>>>` error and were left alone: a failure there is currently reported by the next hardened launch's
prelude or by `sync` as a latched API error, and converting all of them is filed as
[#162](https://github.com/yusiwen/minfer/issues/162) rather than smuggled in here. And the pre-#147
fallback semantics
are preserved deliberately: `launch_mmq_raw_nb_nt` / `_nb_bt_nt` / `_q6k_nt` / `_wide_nt` still return 0
on any opt-in refusal, so the dispatch falls to the next kernel — the failure is named, not fatal,
because a fallback is the designed behaviour and changing it would change which kernel runs.

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
  (`f32`/`f16`/`q8_0`) in its flags word (`0`/`FLAG_F16`/`FLAG_PACKED` — C5 S3, #130), so a
  file written under one width cannot be resumed under another.
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

**C5 S2b — the server's slot snapshot (2026-09-24).**

`server::batch` keeps a per-slot table (`seq`, `start`, `cap`, `cached_tokens`) that admission
rebuilt from scratch, so a restart dropped every slot's context even though the rows were
recoverable in the same container.

**The design decision, stated because the issue's wording admits two readings.** A `Run` holds
the response channel to a client that a restart has already disconnected, plus its RNG and the
row's logits — none of it survives a process boundary, and "resuming" a stream nobody is reading
would be a fiction. So the snapshot carries the **context**, not the in-flight request: each
slot's reservation and the token sequence its rows hold. A request that arrives after the restart
with the same prefix is admitted onto the restored rows and prefills only its own delta — B2's
cross-request prefix reuse, with the rows coming from disk. That is the property the acceptance
measures ("resume the slot, and get the same continuation").

- **`SlotRow` / `SlotsSnapshot`** (`server/batch.rs`) are the table, as versioned JSON in the
  container's opaque host section — the same mechanism S2a added, so the two halves cannot drift
  apart and a version this build does not know is a refusal.
- **`BatchEngine::save_slots(path)`** writes the table *and* the arena through
  `kv_save_with_host`; **`load_slots(path, model)`** loads it back, validates it against this run
  (another `--n-slots`/`--n-ctx`, another model, another KV element type, a table whose sequence
  ids or written extents disagree with the arena it rode in with), and installs the slots.
- **When it is written: after every completed request** (`finish`). That is when a slot's context
  is stable and it is the last moment the rows are known-good — so a server killed *without*
  warning still resumes the conversations that had finished, which shutdown-time saving would not
  give. The cost is one arena write per completed request (`--slots-file` prints the size at
  startup; ~12 MiB for the 0.5B/512-row fixture), which is why the flag is opt-in.
- **`--slots-file <PATH>`** (CLI) → `server::run` → the worker, which loads at startup and
  rewrites on completion, printing what it resumed or why it started empty. The **serial**
  (non-batched) path has no shared arena to snapshot and says so loudly instead of writing
  nothing quietly; the batched engine is the one with cross-request prefix reuse to restore.

**Acceptance, as measured** (2026-09-24, Qwen2.5-0.5B Q4_0, CPU, 512 rows over 2 slots):

- *Resumes without re-prefilling*: the cold run feeds `5/5` prompt tokens; the restored engine
  feeds **`1/5`** (4 reused) for the same prompt, and its next turn — a continuation carrying the
  whole conversation — feeds `6/11` (5 reused). A re-render would have fed all 11.
- *Same continuation*: the restored run's generated text and token count equal the run that never
  stopped (`a_slot_snapshot_resumes_the_context_without_re_prefilling`, `#[ignore]`d, on the
  cached model).
- *Refusals, naming both numbers*: a 2-slot snapshot into a 1-slot engine (`--n-slots`), and a
  512-row snapshot into a 1024-row server (`--n-ctx`).
- *Gate, mutation-checked*: dropping the token mirror from `load_slots` makes the resumed prompt
  feed `5/5` again and fails the gate. Suites: CPU **288 passed / 0 failed / 16 ignored** (was
  287/0/15).
- *Sharing*: `kvformat.rs` gains `expect_for(model, n_ctx)` / `backend_of(device)`, and the CLI's
  `--session` engine now uses them too, so "what this file must match" has one definition.

**C5 S3 — the container encodes the f16 element type · [#130](https://github.com/yusiwen/minfer/issues/130) — DONE 2026-09-25.**

The header already carried the KV element type in its flags word, but only Q8_0 had a bit
(`FLAG_PACKED`). An **f16** region therefore wrote `flags == 0` and read back as **f32**, so
`kv_load`'s `header.format != live` check refused the file its own writer had just produced — on
any CUDA box whose model crosses the f16 auto-policy threshold (Qwen3-0.6B: `28 × 1024 = 28672 ≥
8192`), for `--slots-file` and `--session` alike. The real-model gate showed it as *"the file was
written with the f32 KV element type, this run uses f16"*.

**The design decision: a flag bit, not a format field.** `FLAG_F16 = 1 << 1` is added, and
`flags_of` / `format_of_flags` are each other's exact inverse. A format *field* would have had to
either renumber `FLAG_PACKED` — changing the meaning of every Q8_0 file already on disk — or move
the type elsewhere in the header, shifting the byte layout a version-2 reader is already parsing;
both are the silent misread the container exists to prevent. A new bit keeps the flag word's layout
(and every existing file's meaning), and the compatibility story follows for free: a **pre-#130
build** reading an f16 file sees an unknown bit and refuses it **loudly** at the unknown-flags check
instead of decoding the region as f32, while a **pre-#130 f32 file** (`flags == 0`) still loads as
f32. The two element-type bits are **mutually exclusive** — `packed | f16` describes two
incompatible cell layouts, so it is refused by name, never resolved by preferring one bit.
**No version bump**: `VERSION` stays 2, because a bump is for a layout change and this is an
additive flag whose older-reader behaviour is a loud refusal — exactly as `FLAG_PACKED` was when it
landed.

**What landed.**

- `src/graph/kvsession.rs`: `FLAG_F16`, `KNOWN_FLAGS`, the `flags_of` / `format_of_flags` pair, the
  writer encoding the format into the flag word, and the reader's unknown-bit + mutual-exclusion
  refusals before it decodes.
- Six new gates, five in `kvsession.rs` (no device — the CI-covered half) and one in `alloc.rs`
  (`kv_save` → `kv_load` under an f16 policy): the f16 round trip, the flag-word sweep over all
  three formats (encode and decode), the both-bits refusal, the unknown-bit refusal, the legacy
  `flags == 0` → f32 load, and the allocator-level f16 round trip.

**Measured** (GB10 sm_121, CUDA 13.0, serial).

| check | before | after |
|---|---|---|
| `kvsession` unit tests | 11 | **16** |
| CPU `cargo test --release` | 432 / 0 / 28 unit + 10 / 0 / 6 integration | **438 / 0 / 28** + 10 / 0 / 6 |
| CPU serial ignored | 28 / 0 | **28 / 0** |
| CUDA serial unit suite | 495 / 0 / 31 | **501 / 0 / 31** |
| CUDA serial ignored, 0.5B config | 31 / 0 | **31 / 0** |
| CUDA serial ignored, Qwen3-0.6B Q8_0 | 30 / 1 | **31 / 0** — `a_slot_snapshot_resumes_the_context_without_re_prefilling` resumes its own snapshot |

**Mutation checks** (each reverted byte-identically, `sha256sum`): breaking the encode (f16 → `0`)
fails the flag sweep, the container f16 round trip and the allocator f16 round trip (435/3);
breaking the decode (dropping the `FLAG_F16` branch) fails the same three; removing the
mutual-exclusion check fails only the both-bits gate — and it fails on the **message**, because the
width check would otherwise refuse the file for a different reason (that is the "passes for the
wrong reason" hazard this gate's assertion closes); widening `KNOWN_FLAGS` to `!0` fails only the
unknown-bit gate; and encoding f32 as `FLAG_F16` fails the legacy gate plus the two f32 gates.
`rustfmt --edition 2021 --check` clean on both files (stable rustfmt 1.9.0 — the pinned 1.97.1
toolchain has no `rustfmt` component here; CI runs no fmt job).

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
- `device_memory()` (`src/models/mod.rs`, from `CudaState::device_memory`) is the one place that
  asks the device (CUDA: `cudaMemGetInfo`), and since #122 it returns an explicit
  `allocplan::DeviceMemory` (`Reported` / `QueryFailed` / `NoDevice`) rather than an
  `Option<usize>` where a *failure* and "no device" both looked like a small number. A failed
  query refuses the `auto` request with the real CUDA error name instead of fitting 0 blocks;
  Metal's wrapper reports no free-bytes number yet, so on macOS `auto` needs `MINFER_GPU_MEM` and
  otherwise fits nothing (the default and an explicit count still work there).

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
  CUDA's current free bytes with a quarter held back, resolved through the pure
  `allocplan::budget_decision` over an explicit `allocplan::DeviceMemory` outcome — a
  **failed** query is not a number (it falls back to weights-only accounting with its CUDA
  error named once; see the S4 record below). CPU and Metal are unbounded unless
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

#### E4 record, S4 (#122, 2026-09-24) — a failed device query is not a zero budget

**What was wrong.** `GraphAllocator::memory_budget` derived CUDA's default budget as
`CudaState::device_free_bytes() / 4 * 3`, and `CudaState::device_memory` **discarded the
`cudaMemGetInfo` return code**:

```rust
let (mut free, mut total) = (0usize, 0usize);
unsafe { cudaMemGetInfo(&mut free, &mut total) };   // return code dropped
(free, total)
```

On failure `free` stayed `0`, so the budget became `Some(0)` and the E4 feasibility gate
refused **every** later device allocation with

```
out of Cuda memory: 553 MiB of weights + 0 MiB of pooled buffers + 0 MiB for this
activation exceeds the 0 byte budget (0 MiB); reduce --n-ctx/--n-batch, …
```

The refusal was correct *given a 0 budget*; the defect was that a failed query was
indistinguishable from "the device is full", and the message blamed the budget instead of
naming the cause. `weights_bytes()` shared the class in the other direction: it summed the
registry through `.map(…).unwrap_or(0)`, so a **poisoned** lock silently reported **0
weights** — the fail-*open* twin, which under-charges the same comparison.

**Root cause, and what actually latched the error.** Instrumenting the call
(`eprintln!` of the return code plus `cudaGetErrorName`/`cudaGetErrorString`) gave

```
rc=700 name=cudaErrorIllegalAddress desc=an illegal memory access was encountered free=0 total=0
```

`cudaErrorIllegalAddress` is a **sticky** error: once a kernel in the context performs an
illegal access, every later CUDA call in the process — `cudaMemGetInfo` included — returns
700 until the process ends. The origin was then located with
`compute-sanitizer --tool memcheck` over the failing two-test subset
(`conversation_real_model_smoke` + `cuda_map_window_costs_no_more_than_the_span_it_replaces`):
**4 227 errors**, of which 28 were

```
Invalid __global__ read of size 4 bytes
    at void fa_prefill_f16kv<(bool)0, (bool)0>(…)+0xe70
    by thread (64,0,0) in block (0,0,0)
    Access to 0xf654445fff00 is out of bounds
    and is 249 bytes after the nearest allocation at 0xf654445ffe00 of size 8 bytes
    … Host Frame: CudaState::gqa_attn_f16kv
    … Host Frame: cuda_map_window_costs_no_more_than_the_span_it_replaces
```

`fa_prefill_f16kv` indexes `q` as `nt` token rows of `nh * hd`
(`q[t * nh * hd + h * hd + d]`, `t < nt`), but the timing gate's **prefill A/B reused the
decode phase's single-row `qb` (`nh * hd` elements)** while calling the entry with
`nt = 512`, so the kernel walked up to ~7 MB past the buffer. When those device pages
happened to be mapped the read was silent garbage; when the heap layout left a hole
unmapped it faulted and the context was gone for the rest of the process. Which of the two
happened is a **test-order artifact** — the latch needs the conversation test(s) *and* the
map-window test ahead of the next E4 allocation (subset matrix, same binary:
`map` alone → 1 passed, no 700; `map+packed` → rc 700 count 0; `ctx+map+packed` → rc 700) —
but the **masking** is not: any sticky or fatal CUDA error from any cause (a different
kernel bug, a driver hiccup, an OOM in an earlier call) made `cudaMemGetInfo` fail and
handed the user a message about a 0-byte budget.

So both readings of the bug are true, and both are fixed: **(a)** a production robustness
bug — a failed query silently became a 0 budget with a misleading reason; **(b)** a
test-isolation artifact — a fixture passed an under-sized `q`, and only the test ordering
decided whether that became a visible fault. The production path sizes `q` as `nt` rows
(the `Attn` node's input), so the kernel indexing itself is correct.

**The fix.**

1. `allocplan::DeviceMemory { Reported { free, total }, QueryFailed { code, name },
   NoDevice }` makes the query's outcome a **type**; `CudaState::device_memory` checks the
   return code and names it with a new `cudaGetErrorName` extern, and
   `device_free_bytes() -> Option<usize>` is `None` on failure.
2. `allocplan::budget_decision(explicit, &DeviceMemory) -> BudgetDecision { budget, note }`
   is the **pure** mapping: an explicit `set_memory_budget` wins; a **reported** read keeps
   the pre-existing `free / 4 * 3` byte for byte (a genuine `free == 0` still refuses); a
   **failed** query falls back to weights-only accounting (`Some(usize::MAX)`) with a note
   naming `cudaErrorIllegalAddress (700)`, printed **once per process**; no device state is
   unbounded and silent. The fallback lets the backend's own allocation be the authority —
   it reports the real error if the context is genuinely unusable — instead of turning a
   broken accounting query into a total outage.
3. E5's fit shares the defect through `device_free_bytes()`, where a failure became
   `Some(0)` → `weight_budget → 0` → `fit_blocks(0, …)` → **0 device blocks planned** while
   the startup line said `device free 0 MiB (three quarters of it, …)`. `models::device_memory()`
   now returns the three-way outcome, and `weight_budget` **refuses** `auto` with the real
   CUDA error (offering `MINFER_GPU_MEM` as the escape hatch) rather than planning around a
   non-measurement; `auto_source` can no longer render an unmeasured `device free 0 MiB`.
   "No device at all" keeps the documented Metal behaviour (fit nothing).
4. `weights_from_lock(lock, what, sum)` recovers a **poisoned** registry (append-only, so
   the map behind the poison is still valid) and says so, instead of reporting 0 bytes.
5. `MemoryReport::budget_is_bounded()` / `headroom_bytes()` treat the unbounded sentinel as
   *unbounded*, and `kv_snapshot_from` omits the `minfer_memory_budget_bytes` /
   `headroom_bytes` gauges for it, so the metrics surface never publishes a number that was
   not measured.
6. The other memory queries whose return code was discarded in the same neighbourhood: the
   init banner now reports a failed read instead of printing `0 MB`; the `w16_cache`
   memory-pressure valve became fail-*closed* (`rc != 0` skips the optional cache, where
   `rc == 0 && …` used to fall through and allocate it); `plane_budget_ok` keeps the
   conservative "planes off" decision but prints the error name once instead of silently
   reading a failure as "not enough memory".
7. The trigger is fixed at both ends: the fixture allocates its own `nt`-row `qb2` (the
   timing assertions are untouched — this is a buffer-size fix, not a weakened gate), and
   the CUDA `Op::Attn` arm refuses a `q` input shorter than `n_tokens * n_head * hd`
   before the kernel can read past it (the `BufRef::len` contract of rule 13), so this
   class of mistake is a loud `Err` rather than a corrupted context.

**Measured before / after** (GB10, sm_121, CUDA 13.0, driver 580.178.04; all runs serial,
`--test-threads=1`).

| Gate | Before (`eeba0d0`) | After |
|---|---|---|
| Minimal repro (issue #122's 4 filters) | 3 passed / 1 failed, `exceeds the 0 byte budget` | 3 passed / 1 failed, the failure is the [#87](https://github.com/yusiwen/minfer/issues/87) q8_0-on-CUDA refusal |
| Full `#[ignore]`d serial set (CUDA) | 5 passed / 14 failed, **26** × `exceeds the 0 byte budget` | **20 passed / 2 failed**, **0** × that message (17/2 before the #47 F2 rebase added three tests to the set) |
| `compute-sanitizer --tool memcheck` on the latching subset | 4 227 errors, 28+ `Invalid __global__` faults | **11 errors, 0 kernel-memory faults** (6 `cudaGraphDestroy` + 4 `cudaGetLastError` from `graph_replay_step`, 1 `cudaFuncSetAttribute` at init — filed as [#128](https://github.com/yusiwen/minfer/issues/128)) |
| `cargo test --release` (CPU) | ~340 passed / 0 failed / 18 ignored + 3/0/6 | **382 passed / 0 failed / 21 ignored + 3/0/6** on the rebased tip (346/0/18 before the F2 rebase; the +6 here are the new gates) |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` (CPU, no `cuda` feature) | 12 passed / 0 failed (a stale count in `AGENTS.md`) | **18 passed / 0 failed** |

**The two residual failures were both #123's, and are fixed there.** At this record's time
`a_packed_kv_cache_answers_like_the_f32_one` was CPU-only by its own docstring and was refused on
CUDA (it *failed alone* the same way before this change once the budget was healthy), and
`a_partial_offload_runs_the_rest_on_the_cpu` was second-hand damage: the packed gate set the
**process-wide** KV format to `q8_0` and panicked at the #87 refusal before its
`set_kv_format(F32)` restore ran, so the next test built a `q8_0`-sized CUDA KV region and was
refused (issue #99's mechanism). Neither was a budget failure — the 0-byte-budget string was
absent from the run. The **test-hygiene record (#123)** below fixes both: the packed gate is
CPU-forced and its format restoration is panic-safe (twice over: the normal path *and* a `Drop`
guard), so the serial CUDA set is **22 passed / 0 failed**.

**Acceptance, as measured (pure, CI-covered).**

- `a_failed_device_query_is_not_a_zero_budget`: a `QueryFailed { code: 700, name:
  "cudaErrorIllegalAddress" }` decision is **not** `Some(0)`; the budget is unbounded, the
  note names the error **and** the code, and the note contains neither `0 MiB` nor
  `0 byte budget`.
- `a_reported_free_read_keeps_the_three_quarters_default`: `free = 4 MiB → 3 MiB`, and the
  integer rounding `4 000 001 → 3 000 000`, i.e. the happy path is the pre-#122 number.
- `a_measured_zero_free_read_is_still_a_zero_budget`: a real `free == 0` still refuses, with
  no excuse attached.
- `no_device_state_and_an_explicit_budget_are_unchanged`: `NoDevice` is unbounded and
  silent; an explicit budget is taken as given on every outcome.
- `a_failed_device_query_refuses_an_auto_fit`: E5's `auto` returns `Err` naming
  `cudaErrorIllegalAddress (700)`, offers `MINFER_GPU_MEM`, and never quotes a fabricated
  `0 MiB`; the explicit cap still plans.
- `the_weight_budget_prefers_the_explicit_cap` / `the_auto_source_names_what_the_fit_decided`:
  the pre-existing matrix, adapted to the three-way outcome, plus "an unmeasured device is
  not `device free 0 MiB`".
- `a_poisoned_registry_does_not_report_zero_weights`: a mutex poisoned by a panic is
  recovered and reports its real 18 bytes, not 0.
- **Mutation check** (all three reverted before landing): making `budget_decision` return
  `Some(0)` on `QueryFailed`, making `weight_budget` return `Ok(0)`, and making
  `weights_from_lock` return 0 on a poisoned lock each make their gate fail —
  `0 passed; 3 failed`.

**Honest scope.**

- **Metal is not exercised**: no Mac here. `DeviceMemory::NoDevice` is the Metal arm and
  keeps `auto` fitting nothing without `MINFER_GPU_MEM`; CI's `build-macos` job compiles the
  Metal backend, nothing more. `MINFER_DISABLE_CUDA` / no-device behaviour is unchanged and
  unit-tested through `NoDevice`.
- The **trigger** (the under-sized fixture `q`) is a *test* defect. The device evidence that
  production is unaffected is structural — the graph builder sizes the `Attn` input as
  `nt * n_head * hd` and the real-model gates pass — not a device A/B of a fixed
  production buffer, because there was no production buffer to fix. The trigger's
  test-order dependence is measured (subset matrix + one sanitizer run), not inferred.
- The `compute-sanitizer` numbers are a **memcheck A/B**, not a production measurement: the
  sanitizer changes allocation layout and timing, so the map-window timing assert fails
  under it (1.405x) for [#123](https://github.com/yusiwen/minfer/issues/123)'s reasons. The
  kernel-fault count (28+ → 0) is the part that matters.
- The **SIGTERM drain path** and the rest of the F8 wiring are untouched.
- The **residual `a_partial_offload` failure** is attributed to #123/#99 by mechanism and by
  the error string; it is not fixed here, per the ticket's "do not fix #123 here".
- No Metal run, no full non-ignored CUDA suite run: the ticket's gates are the `#[ignore]`d
  serial set plus the CPU suites, all run as above.

**Follow-ups.** [#123](https://github.com/yusiwen/minfer/issues/123) (the two residual
failures: the CPU-only packed gate and the load-sensitive timing margin),
[#87](https://github.com/yusiwen/minfer/issues/87) (no CUDA q8_0 KV kernel),
[#128](https://github.com/yusiwen/minfer/issues/128) (the API-level
`cudaErrorInvalidValue` findings: `cudaGraphDestroy` on an exec handle, which leaks the
exec, and the eager `cudaFuncSetAttribute` opt-in failing at init).

#### Test-hygiene record (#123, 2026-09-24) — the serial `#[ignore]`d set goes green on a CUDA build

**What was wrong.** The documented CUDA command
`cargo test --release --features cuda --bin minfer -- --ignored --test-threads=1` could never be
green, for two reasons that have nothing to do with the code under test.

1. **A documented CPU-only gate ran anyway.** `a_packed_kv_cache_answers_like_the_f32_one` says
   in its own docstring that a CUDA box refuses `q8_0` by design
   ([#87](https://github.com/yusiwen/minfer/issues/87)), and the default offload request let the
   device claim the model, so it failed **alone** with the `ensure_kv` refusal
   (`KV region for layer 0 would live on Cuda, which has no kernel that reads a packed q8_0
   region …`).
2. **A load-sensitive timing margin.** `cuda_map_window_costs_no_more_than_the_span_it_replaces`
   asserted `p_map <= p_span * 1.25` from **one** 20-launch block per mode, span first and map
   second. Load arriving during the map block had nothing to absorb it: a loaded GB10 measured
   1.267x (2.232 vs 1.761 ms), and a rerun of the same binary passed.

**Collateral (this is #122's record, above).** When the packed gate panicked at the #87 refusal it
never reached its `set_kv_format(F32)` restore, so the **process-wide** KV format stayed `q8_0`
and the next test in the serial set — `a_partial_offload_runs_the_rest_on_the_cpu` — sized its
CUDA KV region for the wrong format and was refused too (the mechanism of
[#99](https://github.com/yusiwen/minfer/issues/99)). The baseline at `bac8440` was therefore
**20 passed / 2 failed**, both failures carrying the identical #87 string.

**The fix.**

1. The packed gate is **device-aware without losing coverage**: it loads its model with
   `OffloadRequest::Layers(0)` (`--gpu-layers 0`, the established all-CPU configuration) and
   asserts `model.device() == Device::Cpu`. On a CUDA build the gate therefore **executes** the
   packed path CPU-forced instead of being skipped — the choice that keeps the most real coverage —
   and on a CPU build nothing changes. It prints the reason it is CPU-only, naming #87.
2. A `KvFormatGuard` snapshots the process-wide format before the gate flips it and restores it on
   `Drop`, so a panic **anywhere** in the gate cannot leak `q8_0` into the next test. The normal
   path's explicit restore stays; the guard is the panic-safe backstop. (Per-engine format is #99
   and was deliberately not implemented here.)
3. The timing gate interleaves the two modes' rounds (span, map, span, map, …) and asserts the
   **median of the per-round `map/span` ratios** — a matched pair per round, robust to up to
   `rounds / 2` disturbed rounds. Decode: 9 rounds × 100 launches; prefill: 9 rounds × 50 launches
   (the old prefill form had no round structure at all), each launch group with a 3-launch warm-up.
   The threshold is **unchanged at 1.25x** — the statistic is the fix, not a wider margin — and the
   gate prints every per-round ratio and the medians it used.

**Measured** (GB10 sm_121, CUDA 13.0, driver 580.178.04; all runs serial, `--test-threads=1`).

| Gate | Before (`bac8440`) | After |
|---|---|---|
| Packed gate alone (CUDA build) | 0 passed / 1 failed, the #87 refusal | **1 passed**, `[c4] CPU-only by construction …`; max \|Δlogit\| 3.0289, region 3.76x smaller — identical to the CPU build |
| Full `#[ignore]`d serial set (CUDA, 0.5B) | **20 passed / 2 failed** (both #87) | **22 passed / 0 failed**, ten consecutive runs (5 + 5) of the same binary |
| Full `#[ignore]`d serial set (CUDA, Qwen3-0.6B config) | — | 21 passed / 1 failed; the one failure was an **unrelated** C5 defect (a session could not encode an f16 element type), filed as [#130](https://github.com/yusiwen/minfer/issues/130) and **closed 2026-09-25** (C5 S3: **31 / 0**) |
| Timing gate, decode ratio | 1.001 / 1.001 / 1.006 / 1.001 / 1.001 (5 runs) | 1.001–1.004 (6 idle runs), 1.001–1.018 (6 loaded runs) |
| Timing gate, prefill ratio | 1.088 / 1.087 / 1.092 / **1.021** / 1.090 (5 runs) | 1.087–1.107 (6 idle), 1.079–1.145 (6 loaded) |
| `cargo test --release` (CPU) | 382 / 0 / 21 + 3 / 0 / 6 | **382 / 0 / 21 + 3 / 0 / 6** (unchanged) |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` (CPU) | green | **21 passed / 0 failed**, with the packed gate executing |

The pre-#123 prefill sample `1.021` is the old statistic's failure mode from the other side: a
spike hit the span block, the ratio went *down*, and the gate would have passed for the wrong
reason. Load for the "loaded" rows: 16 CPU spinners plus two concurrent processes running
`cuda_verify_attention_nt_invariance` in a loop. Individual per-round ratios reached **8.60x** under
that load; the median absorbed them (worst prefill median 1.145, worst decode median 1.018).

**Mutation checks** (all reverted before landing).

- Re-breaking the device-awareness (the pre-#123 `load_model`) makes the packed gate red again with
  the exact #87 refusal: **21 passed / 1 failed** — and the collateral test stays green, because the
  guard now restores the format on the panic path.
- Removing the guard as well restores the `bac8440` baseline exactly: **20 passed / 2 failed**, both
  with the #87 refusal. So both halves — the CPU-forced load and the guard — are load-bearing.
- Doubling the map path's work in the timing gate's prefill closure trips the assert at **2.190x**
  (182.2 vs 83.3 µs/launch), so the gate is live. Since the intrinsic ratio is ~1.09, a uniform map
  regression of ≥ ~15% crosses 1.25x and trips it.

**Honest scope.**

- **Metal is not exercised**: no Mac here; CI's `build-macos` job compiles the Metal backend only.
  The packed gate's CPU-forcing is backend-agnostic, so it holds there too.
- **[#99](https://github.com/yusiwen/minfer/issues/99) and [#87](https://github.com/yusiwen/minfer/issues/87) remain open.** #99
  (per-engine KV format) was explicitly out of scope; the guard is a test-local mitigation, not the
  format-ownership fix. #87 (a device kernel that reads a packed q8_0 region) is *why* the gate is
  CPU-forced: when it lands, the gate's `device() == Cpu` assertion will fire and it should be
  re-pointed at the device.
- The Qwen3-0.6B configuration's serial set was 21/1 for an unrelated pre-existing reason — the C5
  container had no F16 flag, so a CUDA f16 session was refused on load
  ([#130](https://github.com/yusiwen/minfer/issues/130), **closed 2026-09-25**: `FLAG_F16`, C5 S3);
  that test failed **alone** with no #123 code in its path, so it was not a #123 regression.
- The decode threshold could in principle be tighter than 1.25x (its intrinsic ratio is ~1.00). It
  is left at the pre-existing 1.25x on purpose, so the ticket removes flakiness without weakening
  the gate; it is not so wide that a real regression escapes (the 2.190x mutation trips, and the
  arithmetic above puts the detection floor at ~15%).

**Follow-ups.** [#87](https://github.com/yusiwen/minfer/issues/87) (device packed-q8_0 attention),
[#99](https://github.com/yusiwen/minfer/issues/99) (per-engine KV format — **landed 2026-09-25**;
its record below removes the global, and with it this record's item-2 `KvFormatGuard`), and
[#130](https://github.com/yusiwen/minfer/issues/130) (the f16 KV session round trip, found while
gating this ticket).

#### Test-infrastructure record (#99, 2026-09-25) — the KV format is per engine, and the ignored gate set stops lying in parallel

**What was wrong.** The `#[ignore]`d real-model gate set was red whenever the harness ran it in
parallel. Measured on the CPU build at `a756419`: **19 passed / 9 failed** parallel against **28
passed / 0 failed** serially. Every failure had one shape —

```text
KV region for layer 0 was allocated with 57600 elements but 15300 are requested
(n_ctx changed on a live GraphCache; the regions are persistent)
```

— with `57600 / 15300 = 3.765`, exactly the Q8_0 packing ratio. One cause:
`models::qwen2::graph::tests::a_packed_kv_cache_answers_like_the_f32_one` (the C4 packed gate,
device-aware since [#123](https://github.com/yusiwen/minfer/issues/123)) called
`kvformat::set_kv_format(KvFormat::Q8_0)` for its measurement runs. The format was a **process
global** read by `GraphBuilder::new` and `CpuBackend::new`, so while the gate ran any other test
building a graph sized its KV nodes for the packed format while its own `GraphCache` — or the model
it compared against — had been sized under another one; `ensure_kv` refused the mismatch, correctly
(the regions are persistent). #123's `Drop` guard made the *serial* set deterministic by restoring
the format on a panic, but it could not remove the hazard for a test running concurrently; the
parallel red set was the reason the E4 S2 gate run was ambiguous (see the S2 record above).

**The fix (the real one, not the guard).** The format is now a property of the **engine**:

- `models::load_model_configured(gguf, ns, offload, cache_type)` resolves `MINFER_CACHE_TYPE`
  once against `device()` and stamps the answer on the loaded model
  (`ModelDef::kv_format` / `set_kv_format`). `load_model_with` / `load_model_ns` are the
  environment-backed wrappers; the explicit argument is what a test passes instead of mutating the
  environment (which is process-global too).
- `CParams::kv_format` carries it into the build and is **part of the reuse identity**, so a cached
  graph is never reused across formats; `Qwen2Graph::build` / `Qwen3Graph::build` call
  `GraphBuilder::set_kv_format(params.cparams.kv_format)`, and `GraphBuilder::new` defaults to `F32`
  instead of reading a global.
- `GraphAllocator::set_kv_format` gives the **CPU** kernels the same answer
  (`CpuBackend`'s field, which the registry's `kv_format` hook and `kv_element_format` read);
  `forward_batch` calls it once per forward with `model.kv_format`.
- `spec::SpecEngine::new` takes the **target's** format as an argument instead of reading the
  global; the graph JSON exporter (`graph/json.rs`) takes it from the model.
- `graph/kvformat.rs`'s `KV_FORMAT` static, `set_kv_format` / `kv_format` and `KvFormat::from_code`
  are **deleted**, and the obsolete `the_process_wide_format_can_be_redecided` unit test with them
  (the CPU unit count therefore moves **438 → 437 passed**, same 28 ignored).

**The C4 gate now proves coexistence instead of mutating shared state.** It loads **two engines per
arm** — one resolved `f32`, one `q8_0` — through `load_model_configured`, in a CPU arm and (on a
CUDA build with a device) a device arm, and asserts each engine's `kv_format()`, its backend, the
3x-smaller region and the two logit tolerances (unchanged bounds). The `KvFormatGuard` is gone —
nothing process-wide is left to restore. The device arm still sets the one process-wide tag #99 left
in place (`cuda::KV_LAYOUT`, read by the device kernels themselves) under a small
`DeviceLayoutGuard` that restores it on a panic.

**The CUDA device half is deliberately not done.** `cuda.rs` holds the layout in a process-wide
`KV_LAYOUT` that the launchers read directly (not through `CudaBackend::kv_layout`), so a per-engine
device path would need every launcher and the captured-graph key threaded; a per-instance field
alone would silently still read the global. Filed as
[#153](https://github.com/yusiwen/minfer/issues/153); `docs/CUDA-BACKEND-DESIGN.md`'s KV-layout
section states the scope, and the device run keeps the documented serial discipline.

**The entry point.** `scripts/real_model_gates.sh` is the one command for the set: it defaults to
`--test-threads=1` (required on a device) and takes `PARALLEL=1` (CPU-only parallel) and
`FEATURES=cuda`. Documented in `AGENTS.md` rule 11 + the real-model-gates bullet and in
`docs/BUILD.md` §Tests.

**Measured** (CPU build, this box; `--bin minfer` for the gate set).

| run | before (`a756419`) | after (rebased on `09ce9e7`, #121 included) |
|---|---|---|
| `cargo test --release --bin minfer -- --ignored` (parallel) | 19 passed / **9 failed** | **28 passed / 1 failed** |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` | 28 passed / 0 failed | **29 passed / 0 failed** |
| `cargo test --release` (unit + integration) | 438 / 0 / 28 + 10 / 0 / 6 | **438 / 0 / 29 + 10 / 0 / 6** |
| C4 gate alone (`[c4]` print) | — | cpu: f32 6 291 456 B vs q8_0 1 671 168 B (**3.76x**), max \|Δlogit\| **3.0289** of a 37.79 spread; shifted max \|Δlogit\| **2.4662** |

The unit count is unchanged because two opposite moves cancel: #99 deletes the obsolete
`the_process_wide_format_can_be_redecided` test (the global it asserted is gone) and #121 adds one
(`38` -> `37` from #99, `+1` from #121). The ignored set grew by #121's saturation gate, hence 28 ->
29 serial.

The single remaining parallel failure is **not** a KV failure and not a #99 regression:
`server::batch::tests::server_batch_matches_serial_and_is_faster` asserts a **wall-clock** relation
(`t_serial > t_batch`) from two sequential whole-workload measurements, so under a loaded parallel
harness the first-measured phase absorbs the start-up wave — measured **15.60s batched vs 11.54s
serial** on the rebased tree (pre-rebase: 17.59 vs 9.94, then 18.39 vs 9.75), while the serial set
passes it. All nine KV-region failures are gone. The load-sensitive assertion is the same class #123
fixed for the CUDA map-window gate and is filed as
[#154](https://github.com/yusiwen/minfer/issues/154); until it is robust, the **serial** invocation
is the documented entry point.

**Mutation check** (run before the #121 rebase; reverted byte-identically, `sha256sum -c` on all
three files). Re-introducing the #99 shape — a process-global format read by `GraphBuilder::new`,
flipped by the C4 gate per run — makes the parallel set **20 passed / 8 failed**, every failure the
`KV region … was allocated with N elements but M are requested` string with the 3.765x ratio. So the
parallel gate set is what detects the interference, and the per-engine path is what removes it.

**Honest scope.**

- **Metal is not exercised** (no Mac; CI's `build-macos` job compiles the backend only). The
  per-engine plumbing is backend-agnostic; Metal's `kv_format` hook still reads
  `metal::kv_cache_is_f16`, and its packed format is G5 on
  [#44](https://github.com/yusiwen/minfer/issues/44) either way.
- **The device (CUDA) parallel run is still not green and is not claimed.** `CudaState` is a
  process-wide singleton (MMQ memo, captured graph execs, stream state — issue
  [#64](https://github.com/yusiwen/minfer/issues/64)) and the device KV layout tag is process-wide
  on top of that; `cargo test --release --features cuda -- --ignored` without `--test-threads=1`
  remains the wrong command, and `scripts/real_model_gates.sh` keeps it serial. Measured serially on
  this box (GB10 sm_121, CUDA 13.0): the unit suite is **501 passed / 0 failed / 32 ignored**, and
  the ignored set is **32 passed / 0 failed** in both the 0.5B (f32 KV) and the Qwen3-0.6B (f16 KV)
  configurations — unchanged by this increment except that one obsolete unit test is gone.
- An **explicit `f16`** cache type still lets the *device* layout follow
  `set_kv_cache_type`'s auto policy (the pre-C4 split the loader comment records); the builder's
  f32/f16 region shapes are identical, so that is not a sizing hazard. A packed (`q8_0`) resolution
  is still restated explicitly on the device, as before.
- The `forward_graph` (non-cached) path uses the process-global `graph_cache()`; two *models* with
  different formats driving that one cache would still be refused by `ensure_kv`. That is the
  pre-existing single-cache-per-process design, not the format global, and the server/CLI paths
  that matter pass a `GraphCache` explicitly.

**Follow-ups.** [#153](https://github.com/yusiwen/minfer/issues/153) (the CUDA per-graph layout),
[#154](https://github.com/yusiwen/minfer/issues/154) (the load-sensitive batching timing gate).

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

#### E2 follow-up record (#121, 2026-09-25) — a rejected job is answered, not dropped

**The defect.** F8's `minfer_jobs_dropped_total` ([#51](https://github.com/yusiwen/minfer/issues/51))
exposed it (see the F8 record's *"defect found while gating this"*): `serve_loop`
calls `admit` on **every** pass, busy or not, and `admit` consumed the `Job` by
value — so a job it could not place (`no idle slot`) had its
`mpsc::Sender<StreamEvent>` dropped **without an event**. The handler read the
closed channel as a *completed* answer: non-streaming `collect_response` returned
`Ok(("", "stop", 0))` → **HTTP 200 with empty content**; streaming sent an empty
SSE stream followed by `[DONE]`. A client could not tell a dropped request from a
model that produced nothing, and the request was silently lost.

Reproduced on this box (CPU, cached 0.5B Q4_0, `MINFER_BATCH=1 --n-slots 1`, two
concurrent `max_tokens=200` requests, B sent ~0.4 s after A,
`scripts`-free python client — see the PR):

| | before | after |
|---|---|---|
| A | `200`, 976 chars, `finish=length`, 200 completion tokens | unchanged |
| B, non-streaming | `200`, **0 chars**, `finish=stop`, `completion_tokens=0` | `503`, `{"error":{"code":503,"message":"no idle slot","type":"unavailable_error"}}` |
| B, streaming | `200`, SSE = role chunk + `[DONE]`, no error frame | `200`, SSE = role chunk + `data: {"error":{"code":503,…}}` + `[DONE]` |
| `minfer_jobs_dropped_total` | 1 | 1 |

**Decision: reject loudly; queueing is a feature ([#150](https://github.com/yusiwen/minfer/issues/150)).**
The batched worker admits on every pass instead of holding a backlog, so `--n-slots N`
bounds concurrent requests and a request arriving while all N are busy is refused. Queueing
it would turn `admit`'s contract inside out (the caller would own a retry loop **and** a
bound) for a behaviour nobody asked for, and the rejection is already the measured design:
`queue_depth` is a near-zero transient, `minfer_jobs_dropped_total` is the honest signal,
and this ticket's whole point is that the signal must reach the client. The serial path
(`MINFER_BATCH=0`) already *queues* — its one worker pulls a job at a time from the same
channel — so the two paths document their difference instead of one of them silently
losing a request. [#150](https://github.com/yusiwen/minfer/issues/150) tracks making the
batched path queue with a bound.

**What landed.** `src/server/batch.rs`: `reject(job, e)` sends `StreamEvent::Err` through
the job's own sender **before** the sender drops and returns the error so the caller can
still count the rejection; the three paths that give up on a job go through it —
`admit`'s no-idle-slot branch, `admit`'s failed-group-prefill branch (the install loop is
its last, infallible step, so nothing was installed) and `submit_on`'s errors (invalid
slot, busy slot, prompt over the slot context, failed prefill forward). `serve_loop` still
counts the returned `Err`s, so F8's counter keeps its meaning. `src/server/types.rs`:
`ApiError` gains `Clone` (the error is sent and returned); `unavailable` was already the
503 constructor (`status: 503`, `error_type: "unavailable_error"`), rendered by
`server::error_response` (non-streaming) and `server::to_event` (streaming). A misplaced
`submit_on` doc comment that sat above `set_prefill_chunk` was moved back to its function
in passing.

**Audit of every sender-drop path.** `worker_loop_serial`'s `no idle slot` and
mirostat-refusal branches **already** sent their refusals (`git log -S 'no idle slot' --
src/server/chat.rs` puts both at Phase 2–6, before F8), and the `no idle slot` branch is
unreachable in practice anyway: the serial loop pulls one job at a time, so every slot is
idle when it looks. `BatchEngine::fail` sends; `finish` sends `Finish`;
`run_job_isolated` turns a job panic into a 500. The handler's own `job_tx.send` failure
and the drain refusal answer `503` directly, before a `Job` exists. Two residual shapes
were found and **not** changed: (a) `tick`'s failed forward returns `Err` before any
per-run `fail`, so the affected runs keep `needs_forward` and `serve_loop` retries the same
batch forever (a *stuck* sender, not a dropped one) — filed as
[#151](https://github.com/yusiwen/minfer/issues/151); (b) a panic outside
`guarded_forward_batch` in `serve_loop` would drop `pending`'s senders, but every
model-calling path is guarded.

**Gates.** `a_rejected_job_answers_503_and_an_sse_error_frame` (CI, no model) drives the
real `reject` + `collect_response` + `error_response` + `stream_response` and asserts the
503 and the SSE error frame. `a_job_rejected_for_want_of_a_slot_is_answered_with_503`
(`#[ignore]`, real model, one slot, two jobs queued before the loop starts) asserts one
request is served (`Finish`, tokens > 0) and the other gets **exactly one**
`StreamEvent::Err` (status 503, `no idle slot`) and nothing else, with
`jobs_dropped_total` exactly 1. The old F8 round-2 gate kept both receivers dropped, so it
counted the rejection but could not see the empty `200` — the new gate reads them.

**Mutation checks.** (a) `reject` replaced by `drop(job)` (the pre-fix silent drop): the CI
gate fails with *"a rejected job must not read as a completed empty answer: (\"\",
\"stop\", 0)"*, the ignored gate with *"a job's channel closed with 0 event(s) — the #121
silent drop is back (the handler would answer HTTP 200 with empty content)"*. (b)
`jobs_dropped_total.fetch_add(dropped, …)` → `fetch_add(0, …)`: the ignored gate fails
*"exactly one job could not be placed on the one slot: left: 0, right: 1"*; the CI gate
still passes, because the counter is not its property. Both reverted;
`sha256sum src/server/batch.rs` equals the pre-mutation value, byte-identical.

**Verification (2026-09-25).**

| Command | Result |
|---|---|
| `cargo test --release` (CPU) | **439 / 0 / 29** unit + **10 / 0 / 6** integration |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` (CPU) | **29 / 0** (baseline 28 / 0) |
| `cargo test --release --features cuda -- --test-threads=1` (GB10 sm_121) | **502 / 0 / 32** unit + **10 / 0 / 6** integration (baseline 501 / 0 / 31) |
| … `--ignored --test-threads=1`, 0.5B f32 KV | **32 / 0** (baseline 31 / 0) |
| … `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf --ignored --test-threads=1` | **32 / 0** (baseline 31 / 0) |
| `rustup run stable rustfmt --edition 2021 --check` on the changed `.rs` | clean (rustfmt 1.9.0-stable) |
| `python3 scripts/check_docs_links.py` | **940 relative links / 184 files** (unchanged) |

The counts move by exactly this ticket's two gates (+1 unit pass, +1 `#[ignore]`d
real-model gate); the F8 serve-loop gate's own numbers (peak running 2, `1 dropped`) are
unchanged.

**Docs.** `USAGE.md` § *Slot saturation* (the decision, both status shapes, the counter);
`FEATURES.md` and `OPENAI-CHAT-API-PLAN.md` § *Slot Lifecycle* state the batched/serial
split (the plan's old *"defer the task in the request queue"* line was never implemented
by the batched path and is now corrected); `AGENTS.md`'s `server/batch.rs` bullet carries
the contract and the new counts. `ARCHITECTURE-ROADMAP.md` is untouched: no roadmap gap
closes here.

#### E2 follow-up record (#151, 2026-09-25) — a failed decode step answers its batch

**The defect.** The mirror of [#121](https://github.com/yusiwen/minfer/issues/121): there the
sender was dropped without an event; here it was **never dropped and never answered**.
`BatchEngine::tick` built one batch from every run with `needs_forward` set and called
`guarded_forward_batch(...)?` — the `?` returned **before** the per-row loop that takes each
`needs_forward` and before `advance`/`fail`, and `serve_loop` only logged
`[server] step failed: …`. So every run kept `needs_forward = Some(tok)`, the next pass rebuilt
the **same** batch and retried the **same** forward. A deterministic failure (a kernel-invariant
violation, an E4 activation-budget refusal, an `ensure_kv` format mismatch, a panic caught by
`guarded_forward_batch`) failed forever at 100% CPU; no client was ever told, `in_flight` stayed
≥ 1, so a graceful drain ran out its whole `MINFER_DRAIN_MS` deadline.

Reproduced on this box (CPU, no model — a test double whose `forward_batch` panics, driven
through the **real** `guarded_forward_batch` and the real `serve_loop`, with the handler's
`InFlight` guard held until the stream ends and a 500 ms drain window):

| | before | after |
|---|---|---|
| `serve_loop` returns | **never** (3 s window) | yes |
| decode forwards attempted | **955 907 in 3 s** (≈319 k/s — the spin) | 2 (1 prefill + 1 decode) |
| `StreamEvent::Err` to the client | **0** | **1** (status 500) |
| `Finish` | 0 | 0 |
| `in_flight` after the drain deadline | **1** (stuck) | 0 |
| `running` after the step | 1 | 0 |

**Decision: answer every row of the failed batch, once, and do not retry.** The forward is one
weight pass, so the failure belongs to every row it carried — not to a slot, and not to a run
that was not in it. `tick` already builds `rows: Vec<usize>` (the slot index per batch row) while
collecting the pending tokens; that list **is** the membership test, so a run whose
`needs_forward` was unset (already sampled) or that did not fit `MAX_BATCH` is untouched. Each
affected run goes through the existing `fail`, which is exactly the shape
[#121](https://github.com/yusiwen/minfer/issues/121)'s `reject` established per job: one
`StreamEvent::Err` through the run's **own** sender, the run taken (slot freed) and
`cached_tokens` cleared — a forward that failed part-way may have written rows the mirror does
not describe, and reusing that prefix is the one thing a failure must not lead to. Clearing the
slot is also what stops the retry: with `needs_forward` gone the next `tick` builds a different
batch (or none). **No retry is a choice, not an oversight**: every reachable class here is
deterministic and request-fatal, and a caught panic may have left the shared arena half-written,
so trying again can only spin (the defect) or read a corrupted arena. The failure is answered for
**these runs only** — nothing is latched on the server, so a later request builds a fresh batch.
A transient failure would therefore still cost these particular requests their answer: that is
the honest trade against the old infinite retry, and a genuinely transient class (if one ever
appears) belongs in the backend as a retry around the device op. The error attribution is
`ApiError::server` (`500 server_error`) — the **step** failed; `503 unavailable_error` stays the
saturation refusal of #121/#150.

**What landed.** `src/server/batch.rs`: `tick`'s forward is a `match`; its `Err` arm calls
`fail_batch(&rows, &e)` and then still returns `Err(e)`, so `serve_loop` keeps its "step failed"
signal while every affected run has been answered. `fail_batch` (`rows` empty ⇒ no-op) loops over
`rows` calling `fail`, then prints one line naming how many runs it answered and that their slots
were released. No behavior change on the success path, and no new test-only seam in the engine:
the gate's injection is a `ModelDef` double, so the real `guarded_forward_batch` → `ApiError::server`
path is what runs.

**Gates.** `a_failed_decode_forward_answers_a_single_run_and_releases_its_slot` and
`a_failed_decode_forward_answers_every_row_in_the_batch` (both plain `#[test]`, **CI** — no model
on disk): the double's `forward_batch` panics and counts attempts; each gate asserts one
`StreamEvent::Err` (500, `server_error`, non-empty message) and no `Finish`/`Text` per affected
run, the sender closed and the slot free, `cached_tokens` cleared, and — after a second `tick` —
that the attempt count did **not** grow (the bounded no-retry assertion, so a broken build fails
instead of hanging). The multi-slot gate also installs a live run with `needs_forward = None` and
asserts it survives with its prefix intact and its channel open. The real-model batched-vs-serial
gates keep their counts (the new gates need no model).

**Mutation checks.** (a) the pre-fix early `return` (no `fail_batch`): both gates fail with
*"exactly one error, not zero and not a retry: left: 0, right: 1"* (and the multi-slot one with
*"slot 0: exactly one error"*). (b) answer the run but do not take it / clear it (the slot not
released): both fail with *"the run's sender is dropped: the slot is released"* / *"slot 0:
released"*. Both reverted; `sha256sum src/server/batch.rs` equals the pre-mutation value,
byte-identical (`725827f6…`).

**Verification (2026-09-25).**

| Command | Result |
|---|---|
| `cargo test --release` (CPU) | **440 / 0 / 29** unit + **10 / 0 / 6** integration (baseline 438 / 0 / 29) |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` (CPU) | **29 / 0** (unchanged) |
| `cargo test --release --features cuda -- --test-threads=1` (GB10 sm_121) | **503 / 0 / 32** unit + **10 / 0 / 6** integration (baseline 501 / 0 / 32) |
| … `--ignored --test-threads=1`, 0.5B f32 KV | **32 / 0** (unchanged) |
| … `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf --ignored --test-threads=1` | **32 / 0** (unchanged) |
| `rustup run stable rustfmt --edition 2021 --check` on the changed `.rs` | clean (rustfmt 1.9.0-stable) |
| `python3 scripts/check_docs_links.py` | **940 relative links / 184 files** (unchanged) |

The CPU/CUDA unit counts move by exactly this ticket's two gates (+2). The `#[ignore]`d set does
not move (the new gates are CI-covered). The baseline at `ee4d0e1` measured **438 / 0 / 29** unit
on CPU, one below the `#121` record's literal 439 — the #94 count-drift class, recorded here
rather than rewritten into another ticket's dated record.

**Docs.** `AGENTS.md`'s `server/batch.rs` bullet carries the mirror case next to #121's;
`OPENAI-CHAT-API-PLAN.md` § *Slot Lifecycle* step 5 and the *Error Types* table state the failed
step (one 500 per affected run, no retry) and its note no longer claims the batched path defers;
`FEATURES.md`'s `serve` bullet names it. `ARCHITECTURE-ROADMAP.md` is untouched: no roadmap gap
closes here — this is robustness inside an existing row (item 3, *Batch composition + continuous
batching*), not a new capability. [#154](https://github.com/yusiwen/minfer/issues/154) (the
wall-clock flake of `server_batch_matches_serial_and_is_faster` under the parallel harness) is
left as-is on purpose; the ignored set is run serially.

#### Test-infrastructure record (#154, 2026-09-25) — the batching gate's throughput verdict is a median over interleaved rounds

**The defect.** `server::batch::tests::server_batch_matches_serial_and_is_faster` (an `#[ignore]`d
real-model gate, `src/server/batch.rs`) measured the two whole workloads **once, sequentially**
(batched then serial) and asserted the wall-clock relation `t_serial > t_batch`. Under the parallel
`--ignored` harness the first-measured phase absorbs the start-up wave, so the verdict was a property
of the load, not of the code. Measured at `e1ac17f` on this box (CPU build, 0.5B q4_0, four
requests, `max_tokens = 16`):

| run | batched | serial | ratio |
|---|---|---|---|
| parallel `--ignored` (first) | **21.20s** | **9.95s** | **0.47x** (fail; the set was 28 passed / 1 failed in 34.36s) |
| serial, same binary | 0.72s | 1.07s | 1.50x (pass; gate alone 2.68s) |

The correctness half was never at fault — batched vs serial output (and the staggered-admission arm)
already compared byte-for-byte on CPU and passed in both runs.

**The fix (the #123 shape).** The timed rounds are now **interleaved** — `run_batched`, then
`run_serial`, repeated `rounds` times — so each ratio is a matched pair measured next to each other
on the same machine state, and the assertion is on the **median of the per-round `serial/batched`
ratios** (the location estimate that tolerates up to `rounds / 2` disturbed rounds). This is the
shape [#123](https://github.com/yusiwen/minfer/issues/123) gave
`cuda_map_window_costs_no_more_than_the_span_it_replaces`. `median` and a factored
`assert_replies_match` helper replace the two inline comparison loops; every per-round ratio, both
time medians and the verdict median are printed, so a loaded box's result is auditable.

**Statistic, threshold, cost.** The threshold is **unchanged at 1.0x** ("must not be slower"); the
statistic is what changed, and no margin was invented. `rounds = 7` (env
`MINFER_BATCH_TEST_ROUNDS`) is the smallest odd count whose median tolerates three disturbed rounds.
The **timed** rounds use `timing_tokens = 8` (env `MINFER_BATCH_TEST_TIMING_TOKENS`) while the
**correctness** comparison stays a separate full-length (`max_tokens = 16`) pair, so a shorter timing
workload cannot weaken the byte-equality. The gate alone is **8.92s** on an idle box (2.68s before)
and the whole parallel set **~41-44s** (34.36s before). The ticket's cost note ("one pair is ~27s")
describes the *loaded* harness, where a full-length pair reached ~31s; the idle pair is ~1.8s, which
is what made 7 full-length rounds affordable-ish and 7 shorter ones clearly so.

**Verification (2026-09-25, CPU build, this box; `--bin minfer` for the gate set).**

| Command | Result |
|---|---|
| parallel `--ignored`, **before** | 28 passed / **1 failed** — 21.20s batched vs 9.95s serial = **0.47x** |
| serial `--ignored`, **before** | 29 passed / 0 failed |
| parallel `--ignored`, after run 1 | **29 / 0**; per-round ratios [1.335, 1.804, 1.776, 1.434, 1.420, 1.416, 1.368]; **median 1.420x** |
| parallel `--ignored`, after run 2 | **29 / 0**; ratios [1.350, **0.923**, 1.640, 1.342, 1.404, 1.398, 1.379]; **median 1.379x** |
| parallel `--ignored`, after run 3 | **29 / 0**; ratios [1.647, 1.360, 1.481, 1.472, 1.468, 1.396, 1.348]; **median 1.468x** |
| `scripts/real_model_gates.sh` (new default → parallel) | **29 / 0**; ratios [1.204, 1.787, 1.837, 1.805, 1.709, 1.824, 1.818]; **median 1.805x** |
| parallel `--ignored` + 16 CPU spinners | the set 28 / **1** — **this gate passed** at **median 2.159x** (ratios [1.099, 2.046, 2.142, 2.159, 2.182, 2.274, 2.286]); the one failure is a **different**, deadline-based gate, filed as [#158](https://github.com/yusiwen/minfer/issues/158) |
| serial `--ignored`, after | 29 passed / 0 failed |
| mutation: the timed batched arm runs `timing_tokens * 2` | **fails** at **median 0.805x** (ratios 0.784–0.842, every round < 1.0); reverted byte-identically, `sha256sum src/server/batch.rs` = `15b717e1…` |
| `cargo test --release` (CPU) | **440 / 0 / 29** unit + **10 / 0 / 6** integration (unchanged — no new test) |
| `cargo test --release --features cuda -- --test-threads=1` (GB10 sm_121) | **503 / 0 / 32** unit + **10 / 0 / 6** integration (unchanged) |
| … `--ignored --test-threads=1`, 0.5B f32 KV | **32 / 0** |
| … `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf --ignored --test-threads=1` | **32 / 0** |
| `rustup run stable rustfmt --edition 2021 --check` on the changed `.rs` | clean (rustfmt 1.9.0-stable; the pinned 1.97.1 toolchain has no rustfmt component) |
| `python3 scripts/check_docs_links.py` | **940 relative links / 184 files** (unchanged) |

**The wrapper default.** `scripts/real_model_gates.sh` used to default to `--test-threads=1` on every
build, because the device needs it (`CudaState` is process-wide, issue
[#64](https://github.com/yusiwen/minfer/issues/64)). With #154 landed, the only CPU reason left is
gone too, so the wrapper now defaults to **serial when `FEATURES` includes `cuda` and to the parallel
form otherwise** (`PARALLEL=1`/`0` overrides, with a warning if parallel is forced on a CUDA build).
The parallel CPU form is where the #154 gate's robustness is actually exercised, so it is the default
CPU command rather than an opt-in — the fix would otherwise be invisible to whoever runs the wrapper.

**Mutation check.** Doubling the **timed batched arm's** work (`timing_tokens * 2`, the natural
"removes the batching win" injection for this statistic) makes the gate fail at median **0.805x** with
all seven rounds below 1.0 — the median is not blind. It was reverted byte-identically (`sha256sum -c`
on `15b717e1…`). The three plain parallel runs double as the robustness check from the other side: run
2's round 0 measured **0.923x** (a load spike landing on one matched pair), and the median still
returned 1.379x.

**A gate that passes for the wrong reason — found, filed, not fixed here.** The extreme-load run (16
extra CPU spinners) failed `published_metrics_move_as_requests_are_served` at *"the long request
finished inside the deadline"* — it asserts a 180s wall-clock deadline on a 64-token generation, which
is the same *class* of load-dependence as #154 but not the same statistic (an absolute deadline, not a
ratio), and it is green in the plain parallel runs. Filed as
[#158](https://github.com/yusiwen/minfer/issues/158) rather than widened here.

**Docs.** `AGENTS.md`'s real-model-gates bullet no longer says the parallel CPU set is 28 / 1 or that
the wrapper defaults to serial; `docs/BUILD.md` § *Tests* states the per-feature default and the new
counts; the wrapper's own comments carry the same story. `ARCHITECTURE-ROADMAP.md` is untouched: this
is robustness inside the existing test-infrastructure/batching rows (item 3, *Batch composition +
continuous batching*), not a new capability — the same reason the #151 record gives. The stale
"E2's acceptance … take materially less wall time" paragraph that sat on the `serve_on` helper (it
described this gate, not the helper) moved onto the gate, where the new statistic is stated.

#### Test-infrastructure record (#158, 2026-09-25) — the F8 metrics gate bounds work, not wall-clock seconds

**The defect.** `server::batch::tests::published_metrics_move_as_requests_are_served` (an
`#[ignore]`d real-model gate, `src/server/batch.rs`) drove its two requests with **absolute
wall-clock deadlines** — a 120s loop for the warm 8-token request and a 180s loop for the long one —
and asserted `!engine.busy()` when the loop ended. The deadline is not a property of the code: on
this 20-core box, running the whole **parallel** `--ignored` set with **16 extra CPU spinners** made
the long request legitimately exceed 180s, and the gate panicked at `src/server/batch.rs:3609` with
*"the long request finished inside the deadline"*. Measured before the fix: the set was **28 passed /
1 failed in 435.11s** (460s wall with the spinners), green without them (29 / 0). This is the same
*class* as [#154](https://github.com/yusiwen/minfer/issues/154) — a verdict that is a property of the
load, not of the code — but an **absolute deadline** no statistic can absorb, unlike #154's ratio.
[#123](https://github.com/yusiwen/minfer/issues/123) is the third member of the class (the CUDA
map-window gate, which needed a *justified* margin rather than no margin).

**The fix: a bound on progress.** `BatchEngine` gained a monotone `work_units` counter, advanced
once per row a decode forward wrote and once per token `advance` committed. The invariant that makes
it a progress signal: a `tick` that leaves the engine busy must have moved it — if no forward ran,
then some slot's `advance` returned `Continue`, and `Continue` commits exactly one token (every other
`advance` outcome ends the run and takes it). The gate's two loops became two calls to a shared
`drive_by_work(engine, model, tok, rx, budget, what)` helper that steps until idle, asserts the
counter moved on every step that left the engine busy, and caps the step count. **No wall-clock bound
remains in the gate**, so there is no bare literal to justify and no env knob to add: a slow box runs
the same steps, only for longer, while a wedged engine trips the work assertion on the stalling step.

**The step budget.** `step_budget(prompt, max_tokens) = 4 * (prompt + max_tokens + 8)`
(`STEP_BUDGET_MARGIN = 4`, `STEP_BUDGET_SLACK = 8`): one decode forward and one sample per answer
token plus one prefill forward per chunk, times a deliberately loose margin. It is loose because the
bound must catch an engine that *cannot* terminate, never one that is merely slow — a false negative
hangs the suite, a false positive is the flaky gate this ticket removes. Measured on the 0.5B q4_0
(this box, 2026-09-25): the warm request (8-token prompt, `max_tokens = 4`) took **4 steps** against a
budget of **80** (20x); the long one (120-token prompt, `max_tokens = 64`) took **64** against **768**
(12x). The gate prints both counts on every run.

**Verdict, before and after (CPU build, this box, 16 extra CPU spinners on a 20-core machine).**

| Command | Before | After |
|---|---|---|
| parallel `--ignored` + 16 CPU spinners | **28 / 1** in **435.11s** — the gate panicked at `batch.rs:3609` *"the long request finished inside the deadline"* | **29 / 0** in **424.17s** (438s wall) |
| plain parallel `--ignored` | 29 / 0 | **29 / 0** in 36.60s |
| serial `--ignored` | 29 / 0 | **29 / 0** in 38.47s |

**Mutation check.** Making `tick` return `Ok(())` without forwarding or committing (the natural
"wedge the engine" injection) makes the gate fail on **step 1** of the warm request — *"the warm
request: the engine is wedged — step 1 left it busy without advancing the work counter (still 0)"* —
in **0.18s**, where the old deadline would have waited the full 120s to report the same thing. Reverted
byte-identically: `sha256sum src/server/batch.rs` back to `99a608e3…`.

**The audit.** Every real-model gate shape was checked for an absolute deadline used as its *only*
failure signal (`grep` for `Instant::now()` + `Duration::from_secs` across `src/` and `tests/`). The
two #158 deadlines were the only ones; what remains is genuinely different and filed as
[#160](https://github.com/yusiwen/minfer/issues/160):

- `src/server/batch.rs:2925` (`serve_loop_publishes_the_queue_and_running_depth`) — the feeder's 120s
  poll terminator is a **redundant backstop**: the verdict is the downstream `peak_running > 0` and
  queue-arithmetic assertions, and the worker runs on the main thread, so the deadline cannot rescue a
  real wedge.
- `tests/conversation_cli.rs:41` — `run_cli`'s **1800s** child-process kill for the `#[ignore]`d
  real-model CLI sessions: a **cross-process hang guard** (a child exposes no in-process progress
  counter, and 1800s is ~10-100x the legitimate scalar-CPU runtime).
- `tests/backend_registry_cli.rs:27` — `run_cli`'s fixed **60s** kill: a **cross-process hang guard**,
  and not a real-model run (every case points at a nonexistent model path).
- `src/server/mod.rs:827` — a CI unit test's 10s ceiling on a 50ms bounded drain (200x margin), not a
  real-model gate.
- Five `while engine.busy()` stepper loops in the other `#[ignore]`d server gates have **no** bound at
  all (a wedge hangs the suite rather than false-failing it); #158's helpers make hardening them
  mechanical, and #160 tracks it.

**Verification (2026-09-25, this box; `--bin minfer` for the gate set).**

| Command | Result |
|---|---|
| parallel `--ignored` + 16 CPU spinners, **before** | **28 / 1** in 435.11s (the gate panicked at `batch.rs:3609`) |
| parallel `--ignored` + 16 CPU spinners, **after** | **29 / 0** in 424.17s |
| plain parallel `--ignored`, after | **29 / 0** in 36.60s |
| serial `--ignored` (`PARALLEL=0`), after | **29 / 0** in 38.47s |
| mutation: `tick` returns without advancing | **fails** on step 1 (*"the engine is wedged … (still 0)"*), 0.18s; reverted byte-identically (`sha256sum` = `99a608e3…`) |
| `cargo test --release` (CPU) | **440 / 0 / 29** unit + **10 / 0 / 6** integration (unchanged — no new test) |
| `cargo test --release --features cuda -- --test-threads=1` (GB10 sm_121) | **503 / 0 / 32** unit + **10 / 0 / 6** integration (unchanged) |
| … `--ignored --test-threads=1`, 0.5B f32 KV | **32 / 0** |
| … `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf --ignored --test-threads=1` | **32 / 0** |
| `rustup run stable rustfmt --edition 2021 --check src/server/batch.rs` | clean (rustfmt 1.9.0-stable; the pinned 1.97.1 toolchain has no rustfmt component) |
| `python3 scripts/check_docs_links.py` | **940 relative links / 184 files** (unchanged) |

**Docs.** `AGENTS.md`'s real-model-gates bullet and `docs/BUILD.md` § *Tests* carry #158 next to #154;
this record is the per-ticket entry, cross-referencing #154 and #123 as the three members of the
load-dependent-verdict class. `ARCHITECTURE-ROADMAP.md` is untouched: this is robustness inside the
existing test-infrastructure/batching rows (item 3, *Batch composition + continuous batching*), not a
new capability — the same reason the #154 and #151 records give.

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
| F2 | 15 | GBNF-style grammar + JSON-schema constrained decoding · [#47](https://github.com/yusiwen/minfer/issues/47) — **DONE 2026-09-24** · follow-ups [#125](https://github.com/yusiwen/minfer/issues/125) (refused constructs), [#126](https://github.com/yusiwen/minfer/issues/126) (mask cost) | M | this box |
| F3 | 16 | Sampler set: min-p, typical, XTC, DRY, mirostat, logit bias · [#48](https://github.com/yusiwen/minfer/issues/48) — **DONE 2026-09-24** | M | this box |
| F4 | 12 | Backend registry (drop the compile-time enum) · [#57](https://github.com/yusiwen/minfer/issues/57) — **DONE 2026-09-24** · the per-device KV-format capability [#87](https://github.com/yusiwen/minfer/issues/87) needs is now a **used** registry field (`BackendCaps::reads_packed_kv`), not a hardcoded CPU test | M | this box |
| F5 | 14 | Async cross-backend copy + events · [#58](https://github.com/yusiwen/minfer/issues/58) — **DONE 2026-09-24** · CUDA's boundary copy is an `cudaMemcpyAsync` D2H into a pinned slab plus an event, waited on once at a documented synchronization point; the CPU is a registered synchronous no-op and Metal declines (unported) · follow-ups [#137](https://github.com/yusiwen/minfer/issues/137) (Metal), [#138](https://github.com/yusiwen/minfer/issues/138) (true overlap) | M | this box (CUDA) |
| F6 | 22 | Quantizer tooling (`convert-hf-to-gguf`, `quantize`, `split`) · [#49](https://github.com/yusiwen/minfer/issues/49) — **DONE 2026-09-24** · a GGUF v3 *writer* (`gguf_write.rs`), byte-exact weight encoders (`quantize.rs`), an HF converter that passes the strict loader (`convert.rs`), the three subcommands (`tooling.rs`), and a real download size check — follow-ups [#140](https://github.com/yusiwen/minfer/issues/140) (K-quant encoders), [#141](https://github.com/yusiwen/minfer/issues/141) (f16 on the device), [#142](https://github.com/yusiwen/minfer/issues/142) (bf16) | L | this box |
| F7 | 19/20 | Chat-template fidelity + tokenizer generality · [#50](https://github.com/yusiwen/minfer/issues/50) — **DONE 2026-09-24** · follow-ups [#132](https://github.com/yusiwen/minfer/issues/132) (NFC + the remaining pre-tokenizer rules) and [#133](https://github.com/yusiwen/minfer/issues/133) (`--chat-template`, `strftime_now`) | M | this box |
| F8 | 25 | **Metrics/observability** (`/metrics`, KV occupancy, queue depth, per-op timing under a flag, graceful drain). Item 25 was the only member of the A-era batch (items 23/24/26/27/28 -> A1/A2/A7/A5/A6) with no ticket; it is independent of the critical path, hence this table · [#51](https://github.com/yusiwen/minfer/issues/51) — **DONE 2026-09-24**, both real-model gates **device-verified on GB10 sm_121 2026-09-24**; the serial `#[ignore]`d set it left red (**#123**) is green as of 2026-09-24 (**22 passed / 0 failed**) | M | this box |

F1 is the only item in this plan that **cannot be verified on this machine**
(aarch64): it needs an x86 box or a new CI runner. It is also the largest
single CPU win, so it should be scheduled against hardware availability, not
against the critical path.

### F3 — Sampler set (#48) — **DONE 2026-09-24**

**What landed.** `src/sampler.rs` gained a `SamplerConfig` (the pre-F3 knobs plus
`min_p`, `typical_p`, `xtc_probability`/`xtc_threshold`, the DRY group,
`mirostat`/`mirostat_tau`/`mirostat_eta`/`mirostat_m`, and `logit_bias`), a
`MirostatState { mu }` that the caller owns (mirostat is the one sampler with
cross-token state), and one pipeline entry point:

```text
logit bias → penalties → DRY → (greedy shortcut) → top-k → typical → top-p →
min-p → XTC → temperature | mirostat v1/v2
```

The order is llama.cpp's `common_sampler_init` chain. Every filter is a pure
function with its own boundary tests: `apply_min_p` (p = 0/1, ties, empty/one
token, "the argmax always survives"), `apply_typical` (p = 1 off, p = 0 keeps
one, dominated distributions, ties), `apply_xtc` (probability 0/threshold > 0.5
off — *and no RNG draw*, fewer than two candidates, the exact exclusion set),
`apply_dry` (the reverse Z-algorithm plus the restart-sequence cap, an empty
history, a window shorter than `allowed_length`, the exponential scale, the
single-token-breaker exemption), `sample_mirostat_v2` / `sample_mirostat_v1`
(the `mu` update direction, `mu <= 0` never emptying the set, the documented
degenerate `s_hat = 1` rule), and `apply_logit_bias` (positive and negative).

**Surface.** `--min-p --typical --xtc-probability --xtc-threshold
--dry-multiplier --dry-base --dry-allowed-length --dry-penalty-last-n
--dry-sequence-breakers --mirostat --mirostat-tau --mirostat-eta --mirostat-m
--logit-bias` on the CLI; the matching optional fields on the OpenAI request
(`min_p`, `typical_p`, `xtc_probability`, `xtc_threshold`, `dry_*`,
`dry_sequence_breakers`, `mirostat`, `mirostat_tau`, `mirostat_eta`,
`mirostat_m`, `logit_bias`). `SamplerConfig::validate` refuses every
nonsensical value at the boundary (CLI startup / HTTP `400`), and
`validate_logit_bias` rejects a token id outside the vocabulary once it is
known — nothing is clamped or silently ignored.

**Acceptance measurements.** `cargo test --release`: the F3 gate
`default_config_is_bit_identical_to_the_old_path` runs 64 steps through both
`sample_with_penalties` and the new pipeline with one seed and asserts the token
sequences are equal; `default_pipeline_matches_the_pinned_pre_f3_sequence`
asserts the same 64 tokens as a sequence captured from `master` *before* the
change (`[5, 54, 21, 54, 105, …]`, seed 42). A mutation check (forcing
`min_p = 0.5` into the default config) fails the pinned gate, and reverting makes
it pass — so the gate can fail. Full-suite counts are in the F3 record commit /
PR (unit + integration lines).

**Honest scope.** (a) DRY sequence breakers are token-id sequences
(`--dry-sequence-breakers 198;13,2`), not llama.cpp's strings: mapping a string
breaker to the overlapping token sequences needs
`get_overlapping_token_sequences` against the vocabulary, a tokenizer port left
as a follow-up. (b) Speculative decoding (`--spec-draft`) carries the pure F3
filters but **not** mirostat — a verify round samples several rows from one
shared RNG, so `mu` has no faithful home there; the CLI and the server refuse
the combination loudly. (c) In mirostat mode the temperature is ignored
(mirostat's `mu` truncation subsumes it, as in llama.cpp, which sets
`temp = 1.0`); `temp == 0` still wins as the greedy shortcut. (d) The
conversation/KV session snapshot does not carry `mu` (it does not carry the RNG
either): a resumed session restarts `mu` at `2 * tau`, which affects sampling
only, never the KV.

### F2 — Grammar and JSON-schema constrained decoding (#47) — **DONE 2026-09-24**

**What landed.** `src/grammar.rs` (new) holds the whole engine, designed in
[`GRAMMAR-DESIGN.md`](./GRAMMAR-DESIGN.md) and implemented against that
contract; `src/sampler.rs` gained the mask's position in the one pipeline.

*GBNF subset.* Rules, string literals (`\n \r \t \\ \" \xNN \uNNNN`),
character classes with negation, `.`, grouping, alternation, `*` `+` `?`
`{m}` `{m,}` `{m,n}` (upper bound ≤ 1024), `#` comments. Refused loudly, each
with the offending token: `\d`/`\w`/`\p{...}`, an empty or reversed class, a
repetition whose upper bound is below its lower bound or above the cap, a
missing `root`, a duplicate rule name, an undefined reference, and left
recursion (detected while compiling every rule's epsilon closure, so it is a
startup error rather than a generation-time hang).

*JSON-Schema subset.* `type` (string or array), `enum`, `const`,
`properties`/`required`/`additionalProperties` (bool or schema), `items`,
`prefixItems`, `minItems`/`maxItems`, `string`, integer bounds (inclusive and
exclusive, ranges compiled by digit decomposition), `number`, `boolean`,
`null`, `anyOf`/`oneOf`, `$defs`/`definitions` + local `$ref` (recursive
schemas work). Refused loudly: `pattern`, `format`, `minLength`/`maxLength`,
non-integer numeric bounds on `number`, `multipleOf`, `propertyNames`,
`patternProperties`, `dependent*`, `uniqueItems`, `contains`, `allOf`, `not`,
`if`/`then`/`else`, remote or nested `$ref`, `items` as an array (the draft-07
tuple form), and a `required` name that is not declared.

*Engine.* One flat program per rule (`Cp`/`Class`/`Any`/`Split`/`Jump`/`Call`/
`Ret`), a stack of `{rule, pc}` frames, a bounded **set** of nondeterministic
stacks (64) with a stack-depth bound (256) that turns left recursion into an
error, codepoint matching with a carried partial-UTF-8 buffer, EOG allowed only
at an accepting state with no pending bytes, and an empty-piece token never
allowed. The mask is a packed bitset per token id, cached per state (bounded at
64) and computed through a per-call DFA-style transition memo.

*Pipeline.* `SamplerConfig.grammar: Option<Arc<Grammar>>` is the compiled,
immutable object; the mutable automaton state is per run
(`GrammarState`), exactly like `MirostatState`. The one pipeline is now

```text
logit bias → penalties → DRY → [GRAMMAR MASK] → greedy shortcut → top-k →
typical → top-p → min-p → XTC → temperature | mirostat v1/v2
```

The position is the argument: everything before the mask only *shifts* logits
(finite additions cannot lift a masked `-inf`), and everything after it only
*removes* candidates or reweights survivors, so a forbidden token can never win
— including under `--greedy`, which is why the mask is before the shortcut. The
mask consumes no RNG and writes nothing the other stages read, so mirostat's
`mu` and DRY are unperturbed for the same token sequence.
`sample_with_config_grammar` returns `SampleError::{NoAllowedToken, Grammar}`:
an empty allowed set is a loud stop, never an arbitrary token, and a configured
grammar with no run state is an error rather than a silent fallback.

*Surfaces.* CLI `--grammar <FILE>` / `--grammar-str <GBNF>` / `--json-schema
<FILE>` / `--json-schema-str <JSON>` (mutually exclusive, compiled once after
the vocabulary loads, refused at startup); `--cnv` resets the automaton per
assistant turn; `--spec-draft` + a grammar is refused. Server
`response_format: {"type":"json_object"}` and `{"type":"json_schema",
"json_schema":{"name":…,"schema":{…}}}` plus the `grammar` GBNF extension field
— `grammar` together with a non-text `response_format` is a `400`, and so is an
unsupported construct (compiled on the handler side, before a slot is taken).
Batch slots and serial requests each build their own state.

**Measured acceptance.** Greedy, seed 42, 0.5B q4_0 (f32 KV) and Qwen3-0.6B
Q8_0 (f16 KV), schema `{name: string, age: integer 0..150}`:

| Gate | 0.5B q4_0 | Qwen3-0.6B Q8_0 |
|---|---|---|
| schema run parses (`serde_json::from_str`) | `{"age": 25, "name": "John"}` | `{"age": 25, "name": "John Doe"}` |
| `response_format: json_object` (server, 80 tok) parses | `{"field1": "value1", "field2": "value2"}` | — |
| empty case: `max_tokens 0` → `""` / `"length"` | ✅ | ✅ |
| empty case: `root ::= ""` → `""` / `"stop"`, EOG only | ✅ | ✅ |
| max-length (8 tok) → `Grammar::accepts_prefix` ✅, full parse ✗ (as asserted) | `{"name": "John Doe` | `{"name": "John",` |
| mask cost per **new state** (151,936-token vocab) | 5.4 ms (16 states, 86 ms) | 5.4 ms |
| bitwise no-grammar path | pinned 64-token sequence, `test_default_pipeline_matches_the_pinned_pre_f2_sequence` | same |

The two defects the gates found are recorded in `GRAMMAR-DESIGN.md` §9: a
partial-UTF-8 token accepted where the automaton could never finish it (found by
the real-model parse gate, reproduced over HTTP as `"a\uFFFD"` for
`{"grammar":"root ::= \"ab\""}`), and the mask's first version costing 71.3 ms
per state (13× the memoized cost). Four mutations were run against the new
gates and each made one fail: an off-by-one in the sampler's mask index, a
disabled partial-completion check, an ignored `minItems`, and a disabled EOG
allowance.

**Honest scope.** Object properties are accepted in *declaration order* (a
subset of the schema's language: extras may be interleaved only where no
required property is pending); `oneOf` is compiled as `anyOf`; only
integer-valued numeric bounds are compiled (`number` + bounds is refused);
`pattern`/`minLength`/`maxLength` and `multipleOf` are refused. The mask is
O(vocabulary) per new state — 5.4 ms on a 151k vocabulary, cached per state — so
a long constrained generation still pays it once per unseen state. All
acceptance is CPU-only: CUDA/Metal are not touched by this ticket (the mask is
host-side, before any device work), so the CUDA `build-linux-cuda` job compiles
the change but no device run exercises it. Two follow-up issues carry what was
deliberately left out: [#125](https://github.com/yusiwen/minfer/issues/125) (the
GBNF/JSON-Schema constructs the compiler refuses: `pattern`/`minLength`/
`maxLength`, real-valued numeric bounds and `multipleOf`, property permutations,
an exact `oneOf`, `allOf`/`not`, `\d`/`\w`/`\p{...}`) and
[#126](https://github.com/yusiwen/minfer/issues/126) (index the vocabulary by
first codepoint to cut the residual mask cost).

### F7 — Chat-template fidelity and tokenizer generality (#50) — **DONE 2026-09-24**

**What landed.** `src/template.rs` renders the model's own
`tokenizer.chat_template` again, and `src/tokenizer.rs` makes
`tokenizer.ggml.pre` authoritative and replaces the silent byte fallback. The
accepted/refused sets, the loud refusal and the reference behind every gate are
the design-first artifact
[`CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`](./CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md).

*Templates.* minijinja 2.21.0's own extension point
(`Environment::set_unknown_method_callback`) closes the gap — **no dependency
change**, and `minijinja-contrib::pycompat` was considered and rejected as a new
dependency with different edge semantics. The implemented Python `str` methods
are `strip`/`lstrip`/`rstrip` (a *character set*, as CPython), `split`/`rsplit`
with `maxsplit`, `startswith`/`endswith`, `replace`, `lower`/`upper`/`title`/
`capitalize`, `join`, `find`/`rfind`/`count`, plus a `raise_exception` function
for templates that refuse their own input. Qwen3's template therefore renders,
including its `<think>`-block split and re-emission — the behaviour
`QWEN3-SUPPORT-PLAN.md §5 gotcha #9` recorded as lost. Everything else is
refused **loudly**: the error names the construct and the template line and
states that the engine will not substitute a generic prompt. `validate()` runs at
load, so the CLI exits before inference and `serve`/`viz` refuse to start (before
the worker thread is spawned); a per-request failure is an HTTP `400`. The ChatML
fallback survives for exactly one case — a GGUF with no
`tokenizer.chat_template` at all, announced once on stderr — and callers that
passed `""` for a missing template (which rendered the *empty* prompt) are fixed.
`format_single`/`Conversation` propagate the refusal as a turn error.

*Tokenizer.* `tokenizer.ggml.pre` selects the pre-tokenization rule: `qwen2`
(alias `deepseek-r1-qwen`) and `qwen35`, hand-written splitters ported from
llama.cpp's `unicode_regex_split_custom_*` — hand-written because the Rust
`regex` crate has no lookahead and `\s+(?!\S)` is load-bearing for whitespace
runs. Every other value, including a missing key, refuses the load, as do a
`tokenizer.ggml.model` that is not `gpt2`, an empty merge table, and a
vocabulary missing any of the 256 byte tokens. The silent `unwrap_or(0)` is a
checked byte fallback (one token per byte), and the "whole piece is in the
vocabulary" BPE shortcut is gone: it produced a *different* split than the
reference on Qwen3.5, where a vocabulary entry is not reachable through merges.

**Accepted / refused, in one line each.** Templates: all minijinja constructs,
the Python `str` methods above, `raise_exception`, slices with a step
(`messages[::-1]`); refused — any other method or an unparseable construct, each
naming itself. Tokenizers: byte-level BPE with `pre` = `qwen2`/`qwen35`;
refused — SentencePiece/unigram/WordPiece, `ignore_merges` and multi-regex
pre-tokenizers (`llama3`, `default`, `deepseek-*`, `falcon`, `starcoder`, …),
`byte_encode = false`, and a missing/unknown `pre`.

**Measured acceptance.**

| Gate | Evidence |
|---|---|
| Template rendering, per supported model | `model_templates_render_byte_for_byte`: 4 models × 7 cases byte-for-byte against transformers 5.17.0 (`tests/fixtures/chat/*.json`) |
| The artifact the engine loads | `gguf_template_renders_like_the_reference` (`#[ignore]`d): the GGUF's own template renders the same bytes for 4 models × 7 cases, including multi-turn, system-message, generation-prompt and `<think>`-reasoning cases |
| Loud refusal | 4 unit gates; restoring the pre-F7 fallback (mutation M2) fails all four with `unwrap_err() on an Ok`, and reverting makes them pass |
| Pre-tokenizer rules | `pre_tokenizer_split_matches_the_reference`: `qwen2` + `qwen35`, 52 corpus entries each, == CPython `regex` on the model's own `tokenizer.json` pattern |
| Token ids | `token_ids_match_the_reference` (`#[ignore]`d): 52 entries × 5 cached models byte-for-byte — Qwen2.5-0.5B/7B/14B and Qwen3-0.6B against transformers, Qwen3.5-0.8B against llama.cpp `llama-tokenize` on the same GGUF; transformers and llama.cpp agree on all 52 entries where both exist |
| Byte fallback | `byte_fallback_emits_one_token_per_byte`; mutation M5 (emit id 0 again) fails it `[0,0]` vs `[1097,1098]` |
| Mutation checks | M1 `py_trim` ignores the char set → rendering gate red; M2 restore the ChatML fallback → refusal gates red; M3 `is_number` broken → split gate red; M4 unknown `pre` silently defaults → refusal gate red; M5 silent id 0 → fallback gate red; M6 merge order inverted → id gate red (`[1519, 654, 78, …]` vs `[9707, 11, 1879, 0]`). All six reverted |
| Full suite | `cargo test --release`: **393 passed / 0 failed / 23 ignored** + 3 / 0 / 6 (baseline 382/0/21 + 3/0/6; +11 unit gates, +2 `#[ignore]`d real-model gates) |
| Real-model serial set | `cargo test --release --bin minfer -- --ignored --test-threads=1`: **23 passed / 0 failed** on the CPU build (baseline 21/0) — the F2 grammar and F3 pinned-sampler gates included |

**Honest scope.** (a) The tokenizer does not apply the HF normalizer (NFC):
llama.cpp does not either for BPE, and every corpus entry is NFC-normalized;
composed-vs-decomposed input can therefore differ from transformers, and a
normalization table is a follow-up ([#132](https://github.com/yusiwen/minfer/issues/132)). (b) Beyond `qwen2`/`qwen35` no
pre-tokenizer is implemented; each unsupported value is refused by name, and the
classic GPT-2 rule was dropped from the design rather than shipped ungated
(GPT-2's `tokenizer.json` has no `Split` pattern to reference). (c) SentencePiece
/ unigram / WordPiece models are out of scope — this is a byte-level BPE.
(d) The template reference is the model's *published* `tokenizer_config.json`;
the GGUF copies of Qwen2.5's and Qwen3's templates differ from it textually (a
converter-escaped tool-call string; for Qwen3 an older but equivalent revision),
so the real-model gate asserts the rendered bytes, and the fixture records both
hashes. (e) Qwen3.5 (`qwen35` arch) is *not* a supported architecture; only its
tokenizer is used, as the qwen35 reference. (f) Templates that call
`strftime_now` or use filters minijinja lacks are refused by name rather than
supported. (g) The byte-for-byte and id gates need the cached GGUFs, so they are
manual/local (`#[ignore]`d) evidence; the CI-runnable gates are the fixture-
template rendering, the split fixtures, the str-method/semantics fixtures, the
loud refusal and the mutation-checked unit gates. (h) Nothing here is
device-adjacent: the CUDA and macOS CI jobs are compile-only for this change.

### F8 — Metrics and observability (#51) — design (2026-09-24)

**Scope.** Four independent pieces, landed together because they share one
registry: a Prometheus text `/metrics` endpoint, live KV/arena occupancy, queue
depth, and per-op timing behind a flag. A fifth piece — graceful drain — is the
risky one and is designed separately below.

**Where the numbers come from.** The HTTP side and the worker side are two
threads with different reach: the handler owns the job sender and the tokenizer,
the worker owns the model, the `GraphCache` and the `BatchEngine`. A model
reading cannot happen on the handler thread (it would need a lock on the worker),
and a queue reading cannot happen on the worker thread (the channel backlog is
only visible from the sender side). So the design is one `Arc<ServerMetrics>` of
**relaxed atomics** written by both sides and read by the renderer:

* the handler writes `requests_total`, `requests_rejected_total` and
  `in_flight` (the drain surface);
* the worker writes the queue/engine counters (`accepted`, `admitted`,
  `running`) and, after every step, a **published snapshot** of the allocator's
  occupancy.

`minfer_queue_depth = accepted - admitted` is the quantity neither side can see
alone: it is everything between the handler's `send` and the engine's slot
assignment (the channel backlog plus the worker's `pending` deque). It is
monotone per request and converges to 0 when the worker catches up; a
transient overshoot cannot happen because `admitted` only ever counts a job that
`accepted` already counted.

**KV occupancy is a live reading, not a startup snapshot.** `BatchEngine` gains
`publish_metrics()`, called from `serve_loop` after each tick and after each
admission group. It copies `GraphAllocator::memory_report(backend)` (weights /
pool / live / peak live / budget / reservation depth), the arena shape
(`kv_layer_count()` x `kv_n_ctx()`, `kv_region_bytes()`, `kv_is_packed()`) and
the C3 counters (`kv_arena_stats()`) into the atomics. `memory_report` is a
handful of `HashMap` lookups, so publishing per step is cheaper than the step
itself; the alternative — reporting only `MemoryReport` and leaving the arena
shape to the startup log — was rejected because the plan's E4 record says "the
same struct is what F8 will export", and a snapshot taken at startup is exactly
what the ticket says it must not be.

**Per-op timing is measured at one choke point, and the flag is off by default.**
`BackendScheduler::execute` already walks every node and dispatches it to its
backend; the timer wraps that dispatch (`Instant::now()` only when the flag is
on, one `record()` after). The alternative — instrumenting each of
`cpu_backend` / `metal_backend` / `cuda_backend` separately — was rejected: it
triplicates the code, still cannot separate kernel time from its prologue, and
would leave the three backends' definitions of "an op" free to drift. The honest
scope this buys is written down: the number is **dispatch + execution of one
node**, and split-level syncs and cross-backend copies are not attributed.

Storage is a fixed `[(name, AtomicU64 nanos, AtomicU64 calls)]` table with an
exhaustive `op_name(&Op) -> &'static str`, so a new `Op` variant is a compile
error rather than a metric that silently disappears. `MINFER_OP_TIMING` is
presence-checked (the repo's convention for opt-in flags, like `MINFER_TRACE`),
resolved once into an atomic so the hot path is a relaxed load.

**Graceful drain (the risky piece).** Today `server::run` has no signal handling
and the worker is a detached thread blocked on its job channel, and a naive
`join()` can hang forever because in-flight HTTP handlers hold an `Arc<AppState>`
clone and therefore keep `job_tx` open. The design is therefore **bounded first,
observable second, join third**:

1. The handler increments `in_flight` when a job is accepted, and its
   **response guard** decrements it when the response is finished (body sent, SSE
   stream closed, or the client gone). `in_flight` is thus "requests the HTTP
   side still owes a response to" — which is what `axum::serve`'s graceful
   shutdown actually waits on. A client that disconnects mid-answer releases its
   count while the worker is still decoding, so step 4 adds a bounded wait for the
   worker itself; that is the case a single counter would otherwise lose.
2. On `SIGINT`/`SIGTERM`, a task sets `draining` (so handlers refuse new work
   with `503` instead of queueing behind a shutdown) and signals
   `axum::serve`'s graceful shutdown, which stops accepting connections.
3. `server::run` then waits for the serve future **up to `MINFER_DRAIN_MS`**
   (default 30000). On timeout it reports the still-in-flight count to stderr and
   into `minfer_drain_abandoned_requests`, and returns — the process exits and
   the detached worker dies with it. It never joins unboundedly.
4. Only if the graceful path completed inside the deadline does it wait (again
   bounded) for the worker, which by then has seen every sender drop.

The alternative — `worker.join()` after `axum::serve` returns — is the trap the
ticket names: an SSE client that never disconnects keeps a handler clone alive,
`job_tx` never drops, and the join blocks for as long as the client likes. The
CLI default is untouched: with no signal the loop waits forever, exactly as
before.

**Acceptance.** `/metrics` rendering is a pure function with boundary tests;
`/metrics` is driven over a real HTTP connection on an ephemeral port; the
occupancy/queue numbers are published and read back in a real-model engine run;
`MINFER_OP_TIMING` off leaves the table empty and on accumulates (and the
computation's output is unchanged either way); the drain is bounded and reports
what was left. At least one new gate is mutation-checked.

**What landed (2026-09-24).** All five pieces, in three commits: the design note,
`feat(graph): F8 per-op timing behind MINFER_OP_TIMING`, and
`feat(server): F8 /metrics, live KV/queue depth, bounded drain`
([#51](https://github.com/yusiwen/minfer/issues/51)).

*Surface.* `GET /metrics` (Prometheus text 0.0.4) on the OpenAI server, served
from its own router state so a scrape needs neither the tokenizer nor the job
channel and keeps answering while the model is busy. Metric families and units:

| Family | Type | Unit | Meaning |
|---|---|---|---|
| `minfer_requests_total` | counter | requests | accepted and queued for the worker |
| `minfer_requests_completed_total` | counter | requests | response finished (body sent, stream closed, client gone) |
| `minfer_requests_rejected_total` | counter | requests | refused before queueing (draining / worker gone) |
| `minfer_requests_in_flight` | gauge | requests | accepted, not yet finished — the drain surface |
| `minfer_jobs_dropped_total` | counter | jobs | the worker could not place the job in a slot |
| `minfer_queue_depth` | gauge | jobs | `accepted - admitted`: channel backlog + the worker's `pending` deque |
| `minfer_worker_pending_jobs` | gauge | jobs | the worker's own deque right now |
| `minfer_requests_running` | gauge | requests | occupying an engine slot right now |
| `minfer_prompt_tokens_total`, `minfer_completion_tokens_total` | counter | tokens | prompt / delivered completion tokens |
| `minfer_completion_tokens_per_second` | gauge | tokens/s | generated tokens/s over a trailing 16 s window |
| `minfer_draining` | gauge | 0/1 | a shutdown was requested |
| `minfer_drain_abandoned_requests` | gauge | requests | still in flight when the drain deadline expired |
| `minfer_memory_{weights,pool,live,peak_live,budget,headroom}_bytes` | gauge | bytes | E4 `MemoryReport`; budget/headroom **omitted** when unbounded |
| `minfer_memory_{idle_slots,reserved_classes}` | gauge | count | the E4 S3 reservation table's depth |
| `minfer_kv_{layers,rows,region_bytes}` | gauge | count / count / bytes | the arena's shape |
| `minfer_kv_packed` | gauge | 0/1 | packed Q8_0 cells vs f32/f16 words (C4) |
| `minfer_kv_{reserved,owned,shared,free}_cells`, `minfer_kv_{free_runs,sequences}` | gauge | cells / runs | C3/C8b arena occupancy |
| `minfer_kv_{defrags,cells_moved,cows,cow_cells}_total` | counter | count / cells | C3 compaction and C8b copy-on-write |
| `minfer_op_seconds_total{op=…}`, `minfer_op_calls_total{op=…}` | counter | seconds / count | per-op, **present only when `MINFER_OP_TIMING` is set** |

`minfer_completion_tokens_per_second` is the one family the issue names that is not
a plain counter: it is a **trailing 16-second window** (see the honest scope), so a
scrape gets a throughput without a Prometheus server, and `rate()` on the counters
remains available for anyone who wants a different window.

*Flags.* `MINFER_OP_TIMING` (presence-checked) turns on per-op timing.
`MINFER_DRAIN_MS` (default `30000`) bounds the graceful drain. Both are
documented in `docs/USAGE.md`.

*Landed as* [#120](https://github.com/yusiwen/minfer/pull/120) on the branch
`feat/f8-metrics`.

*Drain.* SIGINT/SIGTERM → set `draining` → `axum::serve` graceful shutdown →
wait for the serve future at most `MINFER_DRAIN_MS` via `bounded_drain` → on a
clean drain, a second bounded wait for the worker → otherwise log and record the
still-in-flight count and return. No unbounded join, and with no signal the loop
waits forever as before. The `--slots-file` snapshot is untouched (it is still
written on every completed request inside `BatchEngine::finish`).

Two mechanisms stop new work, and it is worth naming which does what: the
graceful shutdown **closes the listener**, so a brand-new connection after the
signal gets a connection error; the `draining` check in the handler then returns
`503` for a request that arrives on an **already-accepted** connection (a
keep-alive client, or one accepted just before the signal). The `503` is
therefore a backstop rather than the primary switch, and it is the one the
automated gate drives directly.

**Measured acceptance.** `cargo test --release`: **340 passed / 0 failed /
18 ignored** (unit) and **3 passed / 0 failed / 6 ignored** (integration) — a
delta of **+27 passed, +2 ignored** from the pre-F8 baseline (313/0/16 and 3/0/6,
at `911ce2c`). The 27 new non-ignored tests are the rendering boundary set, the
HTTP round-trips, the drain bound, the in-flight guard, the deadline parse, the
timing gates, and the three token-accounting gates (the trailing window's bucket
arithmetic and decay, the counters, and the rendered family). The 2 new ignored tests are the real-model gates, and the whole
ignored set run serially is **18 passed / 0 failed**
(`cargo test --release --bin minfer -- --ignored --test-threads=1`):
`published_metrics_move_as_requests_are_served` (24 layers, 128 rows,
3 145 728 B of KV region, `running` 0 → 1 → 0) and
`serve_loop_publishes_the_queue_and_running_depth` (the real serve loop; peak
running 2, and the deterministic one-slot round dropping exactly 1 job). CI's
CUDA step, `cargo test --release --features cuda --no-run`, is clean locally too
(nvcc 13.0, no device needed).

The first version of the serve-loop gate was **itself** flaky and the failure was
worth recording: it sampled for a peak every millisecond while serving 2-token
answers that finish inside one sampling interval, so "the peak was 0" was true of
the sampler, not the worker; and simply raising `max_tokens` with `n_ctx = 128`
made a request try to grow to the whole arena, which released the other slot's
reservation and killed both requests. The gate now asks for 128-token answers over
`n_ctx = 1024`, and only stops after it has actually *seen* `running > 0`.
`queue_depth` is deliberately **not** peak-asserted: `serve_loop` calls `admit` on
every iteration and a job is placed or rejected within that pass, so the non-zero
window after a send is shorter than a sampler's interval — the arithmetic is gated
purely and the loop's drain through `accepted - queue_depth == n`.

**Issue #51's own acceptance, item by item.** (i) *"Metrics are exposed in a
standard scrapeable format"* — Prometheus text 0.0.4 with `# HELP`/`# TYPE` per
family, asserted well-formed by a gate. (ii) *"and are correct while slots are
admitted, grown and released"* — a real-model round drives exactly that
transition: `minfer_kv_sequences` goes **2 → 1 → 1 → 2** while `reserved_cells`
goes **256 → 184** and `owned_cells` **120 → 183**, with the engine logging
`slot 0: capacity 128 -> 184 cells for a request wanting 184 (released 1 idle
slot(s); 0 run(s) moved)`; `minfer_requests_running` is 0 → 1 → 0 across it. (iii)
*"A drain stops accepting new requests and finishes the running ones"* — the
handler refuses with `503` once `draining` is set, and the bounded drain waits for
the in-flight set (manual SIGTERM run below: 1 in flight → `drain complete`).
(iv) *"Per-op timing is off by default and does not change results when on"* —
gated bitwise through the scheduler, plus a greedy CLI A/B. (v) `tokens/s` — the
counter pair and the trailing-window rate, verified live on **both** response
paths: non-streaming → `prompt 32 / completion 32 / 2.000 tokens/s`, streaming →
`64 / 64 / 4.000` (32 and 64 generated tokens over the 16-second window).

**End-to-end run (manual, on this box, CPU, Qwen2.5-0.5B Q4_0, `MINFER_BATCH=1`).**
A live `minfer serve` on port 18099 with `MINFER_OP_TIMING=1` and
`MINFER_DRAIN_MS=8000`: before any request `/metrics` reports zeros and no timing
family; one 16-token chat completion then shows `minfer_requests_total 1`,
`minfer_requests_completed_total 1`, `minfer_requests_in_flight 0`,
`minfer_kv_layers 24`, `minfer_kv_rows 512`, `minfer_kv_region_bytes 12582912`
(= 24 × 2 × 512 × 128 × 4, the f32 KV of the 24-layer model at `--n-ctx 512`),
`minfer_memory_weights_bytes 422782464`, and a per-op breakdown of exactly the ops
that ran (`matmul` 0.538 s / 2704 calls, `attn` 0.034 s / 384,
`rms_norm`, `rope`, `swiglu`, `add`, `get_rows`, `kvcache_store`,
`kvcache_load`; `silu` and `softmax` are absent because the kernels fuse them).
A second request was started, SIGTERM sent mid-flight with
`MINFER_DRAIN_MS=8000`: `[server] SIGTERM: draining — refusing new requests;
1 request(s) in flight, up to 8000 ms to finish` → `drain complete: every
in-flight request finished` → `worker stopped; exiting`. Repeating with
`MINFER_DRAIN_MS=150` and a 400-token request produced the **forced** path:
`drain deadline (150 ms) reached with 1 request(s) still in flight; abandoning
them and exiting`, with the process gone shortly after. A greedy 12-token run of
the same prompt with and without `MINFER_OP_TIMING=1` produced the identical
generated text (`The capital of France is Paris. It is the largest city`) — only
the throughput lines differed — which is the flag's end-to-end "off changes
nothing computed" claim.

**Mutation checks (the gates can fail).** (a) Rendering the queue-depth sample
from the worker's pending gauge instead of `accepted - admitted` fails
`both_threads_write_into_one_registry` and the HTTP
`metrics_endpoint_renders_over_http`. (b) Mapping `Op::MatMul` to `GetRows`'
index fails `op_names_agree_with_op_index` with "two variants map to index 9".
Both were reverted; the shared test gate is poison-tolerant so a mutation produces
one named failure rather than a cascade of `PoisonError`s.

**A defect found while gating this (pre-existing, not F8's).** The new
`minfer_jobs_dropped_total` counter made it visible: `serve_loop` calls `admit`
every iteration, and `admit` consumes the `Job`, so a job rejected with "no idle
slot" has its event sender dropped without an error event — the handler then
answers **`200` with empty content**, which a client cannot tell from an empty
generation. Reproduced on this box (CPU, 0.5B Q4_0, `MINFER_BATCH=1`,
`--n-slots 1`, two concurrent `max_tokens=200` requests): A `200` with 1014
chars and `finish=length`; B `200` with 0 chars, `finish=stop`,
`completion_tokens=0`, plus `[server] job rejected: no idle slot` in the log. It
is **out of F8's scope** (it changes `admit`'s contract and the error semantics of
both serve paths), so it was written up as a follow-up:
[#121](https://github.com/yusiwen/minfer/issues/121). **Fixed 2026-09-25 by #121** —
`admit`/`submit_on` now answer a job they cannot place through `reject`, so the
same reproduction reads `B 503 {"error":{"code":503,"message":"no idle slot",…}}`
(non-streaming) or an SSE error frame (streaming); the E2 follow-up record in §7
has the before/after transcript, the decision (reject loudly, not queue), the two
new gates and their mutation checks.

**Device verification (2026-09-24, `NVIDIA GB10` sm_121, CUDA 13.0 / nvcc
V13.0.88, driver 580.178.04).** Every measurement above is CPU-only, so the two
real-model gates were re-run on the GPU in an isolated worktree (`.worktrees/f8gpu`,
branch `verify/f8-gpu-gates`, from `master` `3616570`), leaving the CPU numbers
untouched. Build: `CUDA_HOME=/usr/local/cuda-13.0
PATH=/usr/local/cuda-13.0/bin:$PATH cargo build --release --features cuda` →
`targets sm_75,sm_80,sm_86,sm_87,sm_88,sm_89,sm_90,sm_100,sm_103,sm_110,sm_120,sm_121;
PTX compute_121` (`build.rs`'s auto-detected list; `cuobjdump --list-elf` confirms a
`sm_121` cubin and `--list-ptx` a `compute_121` PTX section in the binary). The
device really participates: the load banner reads `CUDA: using NVIDIA GB10 (SM 12.1,
124544 MB, 48 SMs)` / `CUDA: device tier DGX Spark GB10 (Measured, mmq true)` /
`CUDA: GPU acceleration enabled`, plus E5's `offload: all 24 blocks + embed/output
on cuda (403.2 MiB of device weights; default)`, where the control
(`MINFER_DISABLE_CUDA=1`) reads `CUDA: disabled by MINFER_DISABLE_CUDA` and
`CUDA: not available, using CPU fallback` (`=0` disables too —
presence-checked). A greedy 8-token CLI A/B on one prompt measured **1182.0 tok/s
prefill / 208.5 tok/s decode on the device vs 194.4 / 59.2 with the device
disabled** (6.1x / 3.5x) — the difference the gate numbers themselves cannot show.

Both gates were run serially on the device (the `CudaState` singleton is
process-wide, so the serial form is the only honest one) on the default 0.5B (f32
KV) **and** on `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf` (f16 KV, hd 128 —
the only combination that reaches FA prefill):

| Gate | Configuration | Result | Numbers it printed |
|---|---|---|---|
| `serve_loop_publishes_the_queue_and_running_depth` | 0.5B f32 KV, device | ok | peak running 2, 24 layers, 25 165 824 B region, 262 owned cells, 1 dropped |
| `serve_loop_publishes_the_queue_and_running_depth` | 0.5B f32 KV, `MINFER_DISABLE_CUDA=1` | ok | peak running 2, 24 layers, 25 165 824 B, 266 owned, 1 dropped |
| `serve_loop_publishes_the_queue_and_running_depth` | Qwen3-0.6B f16 KV, device | ok | peak running 2, 28 layers, 234 881 024 B, 262 owned, 1 dropped |
| `published_metrics_move_as_requests_are_served` | 0.5B f32 KV, device | ok | 24 layers, 128 rows, 3 145 728 B, owned 11, idle_slots 10, reserved_classes 5; 2→1→1→2, 256→184, 120→183 |
| `published_metrics_move_as_requests_are_served` | 0.5B f32 KV, `MINFER_DISABLE_CUDA=1` | ok | same shape, `reserved_classes` 4 |
| `published_metrics_move_as_requests_are_served` | Qwen3-0.6B f16 KV, device | ok | 28 layers, 128 rows, 29 360 128 B, owned 11, idle_slots 17, reserved_classes 6; 2→1→1→2, 256→184, 120→183 |

The *asserted* numbers are **device-independent by construction** — `model.n_layer()`,
`n_ctx` and the KV element width, plus the arena bookkeeping — so the device and the
control agree on them, and that agreement is deliberately **not** offered as
evidence. The device evidence is the in-test banner/offload line and the CLI A/B
throughput above. Two non-asserted gauges did differ (`owned_cells` 262 vs 266 in
the serve-loop gate, `reserved_classes` 5 vs 4 in the metrics gate), both incidental
— the greedy stop point and the per-backend size-class ladder — not independent
proof.

The F8 path that genuinely runs on the device is per-op timing, because
`BackendScheduler::execute` dispatches CUDA through the same wrapper. On the device
with `MINFER_OP_TIMING=1` a `serve` plus one 8-token chat completion renders **11 op
families / 26 lines**, including the CUDA-only fused forms `fused_qkv` (72 calls) and
`fused_ffn` (72), alongside `matmul` 316, `rms_norm` 196, `add` 192, `attn` 96,
`kvcache_load` 96, `rope` 48, `kvcache_store` 24, `swiglu` 24 and `get_rows` 6; the
same server with the flag unset renders **0** `minfer_op_` lines, so the flag is off
by default on the device too. The computed text is unchanged: a greedy CLI A/B
(`--greedy --seed 42 -n 12`) produced `The capital of France is Paris. It is the
largest city` byte-identically with the flag on and off (and across a repeated
off run), and the same chat request served with and without the flag returned
identical content.

**The full `#[ignore]`d set on this CUDA build was red, and not because of F8.**
`cargo test --release --features cuda --bin minfer -- --ignored --test-threads=1`
gave **5 passed / 14 failed** at `3616570` (deterministic across two runs), all 14
failing with the same refusal: the E4 default CUDA budget collapsed to **0 bytes**
once the conversation and map-window tests had run in one process — the
`cudaMemGetInfo` return code was discarded, `free` stayed 0, and the gate then refused
every later allocation — while `/proc/meminfo` sampled during the minimal reproduction
showed 118.6–119.8 GB of 121 GB still available. **That is fixed**
([#122](https://github.com/yusiwen/minfer/issues/122)): the same command at
`eeba0d0` + the S4 record below is now **17 passed / 2 failed** (**20 passed / 2 failed**
on the F2-rebased tip, which added three tests to the set), and the 0-byte-budget
message appears **0** times. The two residual failures were both the packed-cache gate's
[#87](https://github.com/yusiwen/minfer/issues/87) cause, tracked in
[#123](https://github.com/yusiwen/minfer/issues/123): first
`a_packed_kv_cache_answers_like_the_f32_one` itself (CPU-only by its own docstring,
refused on CUDA), then `a_partial_offload_runs_the_rest_on_the_cpu`, because the packed
gate sets the **process-wide** KV format to `q8_0` and panics before restoring `f32`, so
the next test's CUDA KV region is sized for the wrong format (the #99 mechanism, made
visible by #87's refusal now being the first failure instead of the budget). The
map-window gate's 1.25x timing margin (a loaded GB10 exceeded it once, 1.267x) is the
third #123 item and **passed** in this run. All three are filed, and none reproduces with
either F8 gate run alone, which is why the device claim below is scoped to those two
gates plus the op-timing path. **All three are fixed in
[#123](https://github.com/yusiwen/minfer/issues/123)** (its test-hygiene record lives in
the E4 section, above): on the 0.5B configuration the serial set is **22 passed / 0
failed**, ten consecutive runs of the same binary, and the timing gate's verdict no
longer depends on the box being quiet — its median-of-ratios statistic stayed below
1.15x even with 16 CPU spinners and two concurrent CUDA attention loops. The
Qwen3-0.6B configuration's set was 21/1, its one failure a separate C5
defect filed as [#130](https://github.com/yusiwen/minfer/issues/130) (**closed 2026-09-25**:
the container's `FLAG_F16` bit, C5 S3).

**Honest scope.** (a) Every measurement in the sections *above* this block was taken
on a **CPU-only build**: that worktree was built with `cargo build --release`,
without `--features cuda`, so the server's banner reads `device cpu` and the CUDA
backend is not compiled in at all (the outer tree's CUDA binary was left untouched).
This ticket does not need a device and the wiring is backend-agnostic — the timing
hook is in the shared scheduler and `kv_snapshot_from` reads whatever backend the
model reports — so the plan-level claim was made without one. The two real-model
gates and the timing path are **device-verified** in the block above (2026-09-24,
GB10); everything else here still carries no CUDA/Metal *runtime* claim. The CUDA
compile path is covered locally with `cargo test --release --features cuda --no-run`
(nvcc 13.0, no device needed) and by CI's `build-linux-cuda`; the macOS compile path
is CI's `build-macos` only. (b) Per-op timing measures the **scheduler's per-node
dispatch** — the one choke point CPU/Metal/CUDA share — so it includes the
backend's prologue and excludes split-level syncs, cross-backend staging copies,
allocator liveness and `fill_input`; there is no kernel-only timer. (c) There is
no histogram and no latency accounting (no `_bucket`/`_sum`,
no time-to-first-token): the counters and the trailing-window `tokens/s` the issue
asks for are present, but a latency histogram would need a bucket policy the issue
did not specify. The rate is a **trailing 16-second window**, not a lifetime
average — a lifetime average is not a throughput, since an idle server would keep
reporting its startup burst — and it is a dedicated one-writer bucket store (no
lock, no allocation). Tokens are recorded where the response is *produced*, so the
batched and serial paths are covered uniformly and nothing is plumbed through the
engine; a client that disconnects before its answer is complete is not counted,
because those tokens were never delivered. (d) The serial (non-batched) path has one `GraphCache` per slot, so
`/metrics` reports the arena of the slot that served the last request rather than
a sum; the batched path reports its single shared arena in full. (e) Queue depth
is a subtraction of two relaxed counters, so a scrape can transiently read 0
while a job is in the channel; it is exact in the steady state and saturating,
never negative. (f) The Metal path is compile-checked by CI's
`build-macos` only (there is no Mac here); the CUDA path is compile-checked both by
CI's `build-linux-cuda` and locally (see (a)), and the timing hook itself is
exercised on CPU here and on the GB10 (the device block above), including the
CUDA-only fused ops. (g) The signal path is **not in the automated suite** — only
`bounded_drain`, the deadline parse and the draining `503` are. The end-to-end
runs above were performed by hand on this box and are recorded here as manual
evidence, not as a CI gate; wiring a `SIGTERM` into a test would mean a
subprocess and a port, which this ticket did not take on.

### F4 — Backend registry (#57) — **DONE 2026-09-24**

**What landed.** `src/graph/registry.rs` (new) is the backend registry, and the
compile-time `enum Backend { CPU, Metal, Cuda }` is gone. `Backend` is now a
`Copy`/`Hash`/`Eq`/`Ord` **handle** over a fixed, configuration-independent id
space (`cpu = 0`, `metal = 1`, `cuda = 2`) — the ids are a KV-session
file-format contract, so they are appended, never renumbered. The design-first
artifact is [`BACKEND-REGISTRY-DESIGN.md`](./BACKEND-REGISTRY-DESIGN.md); the
contract is summarized in `COMPUTE-GRAPH-DESIGN.md` §3.6, and the module map and
the `Extending → New backend` recipe in `AGENTS.md` were rewritten to match.

*The registry.* One `BackendEntry` per backend, registered by its own module at
startup, carrying the name, the **assignment priority** (Metal 300, CUDA 200,
CPU 100 — the pre-F4 statement order inside `supports_for`, now an explicit
number), the capability matrix (`supports_op` / `supports_fused` /
`supports_attn_span` / `reads_packed_kv`) and the hooks that reach the pool
(`pool`/`pool_mut`, `host_read`, `kv_format`, lazy `enable`, `unavailable`).
Each backend's capability matrix moved to module-level functions that its
`impl Backend` methods forward to, so the registry's answer and the trait's
answer are one authority (asserted by a gate).

*What stopped matching on the enum.* The allocator's twelve dispatch sites
(allocate / free / fresh, `pool_len`, `weights_bytes`, `write_host_window`,
`write_host`, host read, `synchronize`, `copy_cells`, the KV element format, the
session `enable`) are trait calls on the entry's pool hook, with the per-`cfg`
fallback arms gone; `supports_for` walks `registry().by_priority()`;
`scheduler::execute` is `alloc.pool_mut(split.backend)`; the fusion wiring in
`graph/json.rs` and both model graphs is `alloc.fusion_backends()` +
`alloc.fusion_backend_index()` (the hand-built `[cpu, metal?, cuda?]` vector and
its `name() == "cuda"` position lookup are gone from three call sites);
`kvsession`'s tag table is `Backend::index()` / `from_index()`; the JSON and DOT
exporters, the op-matrix harness and `kvformat::KvFormat::supports` read the
handle/registry. `models::Device` gained `Device::backend()`, the one bridge
between the two id spaces (`kvsession::backend_of` now delegates to it).

*The two orders.* **Identity** (`Backend::index`) fixes the on-disk KV-session
tag, the exported graph's backend index, `Ord` and the fusion backend list;
**priority** (the numbers above) is the assignment preference. Nothing depends
on `HashMap` iteration order: the table is a fixed-size array indexed by id,
`by_priority()` sorts by `(Reverse(priority), id)` so no two entries can tie, and
the filter is a `[bool; 3]`. Both orders and the numbers are pinned per
configuration by a gate.

*The name surface.* `--backend <name>` (repeatable; comma-separated values
accepted; extracted from `argv` **before** any subcommand dispatch, so `serve`,
`viz`, `bench` and `specverify` all honour it) and `MINFER_BACKENDS=<csv>`;
accepted names `cpu`, `metal`, `cuda`; trimmed, case-insensitive; the flag wins
over the environment; unset is the pre-F4 behaviour. The request is a **fence**:
it removes backends from participation and never adds one. `cpu` is always
admitted — it is the universal fallback a graph must always be assignable to —
so `--backend cpu` is the useful spelling (force the CPU for every device) and
`--backend cuda` means "the device when it can take the node, the CPU
otherwise". The fence is read from **one** filter in two places, which is what
keeps them from disagreeing: `Qwen2Graph::device` / `Qwen3Graph::device` (so a
fenced device never causes a device-only fused node to be built) and
`GraphAllocator::supports_for` (so a node is never placed on a pool whose
weights were never registered).

*Refusals.* Three classes, three messages, all before the model is even
resolved:

```text
unknown backend 'gpu2'; known backends are: cpu, metal, cuda
backend 'cuda' is known but not compiled into this build: the CUDA backend is compiled only with --features cuda
backend 'cuda' is compiled in but not available on this machine: no CUDA device is available, or CUDA is disabled (MINFER_DISABLE_CUDA)
```

Stage 1 (names, purely — CI-covered) runs at the top of `main`; stage 2
(availability) runs once the device layer is up and, on every path that will run
a model (`run`/`serve`/`viz` in `main`, `bench` and `specverify` in their own
`run`), **before** the model path is resolved, so a missing file or a bad GGUF
cannot preempt it. Only a backend the request **named** is checked at stage 2:
the default request means "whatever this build can use", so
`MINFER_DISABLE_CUDA=1` / `MINFER_DISABLE_MPS=1` keep meaning "run on the CPU".
That distinction is a behaviour-preservation gate of its own — the first cut of
stage 2 checked every allowed backend and turned `MINFER_DISABLE_CUDA=1` into a
startup refusal, which the gate caught.

**Feature gates.** The *registered set* is the compile-time one (CUDA only under
`--features cuda`, Metal only on macOS) and is pinned per configuration, but the
**names** are unconditional: `metal` on Linux and `cuda` on a default build give
the accurate "not compiled into this build" refusal rather than "unknown". A
backend that is compiled out is never silently treated as absent.

**#87.** `BackendCaps::reads_packed_kv` is the per-device KV-format capability
query, and it is **used by this ticket's own code**, not reserved:
`GraphAllocator::ensure_kv`'s packed-region refusal and `KvFormat::supports`
both read it, replacing `backend != Backend::CPU` and
`matches!(device, Device::Cpu)`. The region-sizing gate and the C4 format gate
now read one field; [#87](https://github.com/yusiwen/minfer/issues/87) is the
work that flips CUDA's and Metal's value and adds their kernels. No follow-up
issue is filed for it here.

**Measured acceptance.**

| Command | Result |
|---|---|
| `cargo test --release` (CPU) | **399 passed / 0 failed / 23 ignored** unit (baseline 393; +5 registry, +1 allocator fence) and **10 / 0 / 6** integration (baseline 3; the new `tests/backend_registry_cli.rs` is 7) |
| `cargo test --release --features cuda --test backend_registry_cli` | **7 passed / 0 failed** (exercises the `#[cfg(feature = "cuda")]` branch of the disabled-device gate) |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` (CPU) | **23 passed / 0 failed** (baseline 23) |
| `CUDA_HOME=/usr/local/cuda-13.0 … cargo build --release --features cuda` | exit 0 |
| `cargo test --release --features cuda --bin minfer -- --ignored --test-threads=1` (GB10 sm_121, cached 0.5B) | **25 passed / 0 failed**. The ticket's "22" predates F7's two reference gates: `0602eb2` has 26 `#[ignore]` attributes, 2 of them in the macOS-only `src/metal.rs`, i.e. **24 runnable here**, and this ticket adds the 25th (the fence gate below) |
| `cargo test --release --features cuda --bin minfer -- --test-threads=1` (GB10, whole unit suite) | **452 passed / 0 failed / 25 ignored** |
| `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf cargo test --release --features cuda --bin minfer -- --ignored --test-threads=1` | **24 passed / 1 failed** at this commit, the one failure `server::batch::tests::a_slot_snapshot_resumes_the_context_without_re_prefilling` with *"the file was written with the f32 KV element type, this run uses f16"* — the then-pre-existing [#130](https://github.com/yusiwen/minfer/issues/130) C5 defect, **closed 2026-09-25** (C5 S3: **31 / 0**), which the record above documents as this configuration's 21/1 (the count moved by the same +3: F7's two gates and this ticket's) |
| registry / CLI gates | `names_resolve_and_unknown_names_are_refused`, `the_registered_set_and_priority_order_are_pinned`, `the_name_surface_fences_devices_and_keeps_cpu`, `the_packed_kv_capability_is_the_registrys_answer`, `registry_caps_match_the_backend_trait`, `alloc::tests::a_fresh_allocator_inherits_the_runs_backend_filter`, `tests/backend_registry_cli.rs` (7, incl. `a_disabled_device_is_not_refused_unless_it_was_named`) |
| `rustfmt --edition 2021 --check` on the 19 changed `.rs` | clean (stable's rustfmt 1.9.0: the pinned 1.97.1 toolchain has no `rustfmt` component installed here — CI runs no fmt job) |
| `python3 scripts/check_docs_links.py` | **935 links resolve in 183 files** (baseline 930 / 182) |

**Mutation checks (all reverted, all restored byte-identical).**

| Mutation | Gate that failed | Observed |
|---|---|---|
| (a) an unknown name falls back to the default backend | `names_resolve_and_unknown_names_are_refused` + `the_name_surface_fences_devices_and_keeps_cpu` | `called 'Result::unwrap_err()' on an 'Ok' value: CPU`, and `Ok(BackendFilter { allowed: [true, false, false], … })` |
| (b) swap the CUDA and CPU priorities | `the_registered_set_and_priority_order_are_pinned` | `left: [("cpu", 200), ("cuda", 100)]` vs `right: [("cuda", 200), ("cpu", 100)]` |
| (b2) perturb one priority value (200 → 201), order unchanged | same gate | `left: [("cuda", 201), ("cpu", 100)]` vs `right: [("cuda", 200), ("cpu", 100)]` |

(b2) exists because the expected numbers are deliberately **literals**: deriving
them from the `PRIORITY_*` constants would have made the value half of the gate
vacuous (only a reordering would have failed).

**Manual evidence (device, GB10; not a CI gate).** With the CUDA build,
`MINFER_DISABLE_CUDA=1 minfer --backend cuda <model>` and the `MINFER_BACKENDS`
spelling both print the stage-2 message; the same with `serve`, `viz`, `bench`
and `specverify`; `--backend metal` on Linux prints the not-compiled message;
`--backend gpu2` prints the unknown-name message. The fence itself was checked
end-to-end: `minfer --backend cpu … "The capital of France is"` and
`MINFER_DISABLE_CUDA=1 minfer …` on the **same CUDA build** produce
byte-identical stdout (prompt + 8 greedy tokens) apart from the timing lines —
i.e. naming `cpu` selects exactly the path the pre-existing disable flag selects.

**Honest scope.** (a) **Metal is compile-only**: there is no Mac here, so the
Metal entry's `pool`/`host_read`/`kv_format`/`enable` hooks, its `unavailable`
probe and its refusal text are covered by CI's `build-macos` job and by
`rustfmt`, and by nothing that runs. (b) Behaviour preservation rests on the
existing suites plus the new order/name gates; the parts a suite would not
notice are pinned explicitly — the identity order (the on-disk session tag, the
exporters), the priority order *and* numbers, `Ord`, the `Debug` spelling, and
the equality of the registry's capability matrix with the trait's. (c) One
deliberate, **unreachable** behaviour delta: `fill_input` / `write_pool` name a
disabled backend in an `Err` where the old code panicked with
`expect("… pool not enabled")`; assignment can never select a backend whose pool
is not enabled, so no suite path reaches it. (d) `copy_kv_to_cpu` keeps its
pre-F4 CPU/CUDA-only shape (a CPU identity-debug helper) rather than widening to
Metal through the new hook — widening is not this ticket's job. (e) One
per-backend branch remains on purpose: the `MINFER_TRACE`/viz capture path, where
each backend captures through its own mechanism (a borrowed read, a Metal blit,
an async D2H into pinned CUDA staging) — that is machinery, not a capability.
(f) The registry makes the backend set **data, not pluggable**: there is no
`dlopen` path, so a Vulkan/remote backend is registered by editing the crate, not
by dropping in a library (`ROADMAP` §2.6 keeps that distinction). (g) The CPU
numbers above are the *final* tree (after the rustfmt pass and the stage-2 fix);
the CUDA numbers are from the same final tree.

### F5 — Async cross-backend copies and events (#58) — **DONE 2026-09-24**

**What landed.** The scheduler's split boundary is now **two registered phases**
instead of one blocking host round trip. `GraphAllocator::copy_across` (phase A,
*enqueue*) resolves the destination staging buffer, marks the entry pending, and
calls the **source backend's** `BackendEntry::copy_cross` hook — the F4 rule
applies: the allocator never asks "is this the CPU?", it asks the entry.
`GraphAllocator::await_cross` (phase B, *wait*) calls the entry's `await_cross`,
clears the pending flag and counts the wait. The scheduler walks `Split::inputs`
once for the copies and once for the waits, before any node of the consuming
split runs; the consumer resolves a staged input through the new
`GraphAllocator::cross_input`, which **refuses** a still-pending entry with a
loud `Err` naming the missing wait. The registry contract, the per-backend table
and the enumerated synchronization points are in
[`BACKEND-REGISTRY-DESIGN.md`](./BACKEND-REGISTRY-DESIGN.md) §11; the
scheduler-side summary is `COMPUTE-GRAPH-DESIGN.md` §3.4.

*The hot path, defined and measured.* The ticket is about **the split-boundary
staging copies** — the per-`Split::inputs` transfers `execute` makes when a value
produced on one backend is consumed on another; they run on every forward, on the
critical path of every partially offloaded decode step. The copies that are
legitimately host-visible and therefore **not** in scope are enumerated in the
registry doc §11.1 (logits readback, KV-session save, debug/trace dumps,
weight/tokenizer load, `fill_input`) so "zero blocking boundary copies" is not
read as "zero device→host copies anywhere". Before F5 a single CUDA→host boundary
input cost **two** host stalls — a full stream synchronization inside
`copy_to_host` and then a blocking `cudaMemcpy` D2H — which is why the counters
below count both.

*Instrumentation.* `graph/copystats.rs` holds the per-allocator counters
(`copies`, `waits`, `blocking_host_copies`, `async_host_copies`, `event_syncs`,
`stream_waits`) and the `MINFER_SYNC_COPIES=1` switch that restores the pre-F5
path as the bitwise reference (with `set_sync_for_test` as its programmatic
form). Device-level twins: `CudaBackend::blocking_readback_count()` (actual
blocking `cudaMemcpy` D2H calls) and `cuda::stream_sync_count()` (host stalls).

*The device layer.* `src/cuda.rs` gains the event primitives (`record_event`,
`wait_event`, `stream_wait_event`, `event_destroy`) and the async transfer
(`copy_to_host_async`, `host_alloc`/`host_free` for the pinned slabs) — every
failure is a loud `Err` naming the `cudaGetErrorName`, never a silently missing
synchronization. `CudaBackend` owns a small grow-on-demand pinned-slab pool and
the pending-copy table; a slab is released when its event has been waited on, and
`Drop` destroys the events and frees the slabs.

**Measured acceptance (GB10 sm_121 unless noted).**

| Command | Result |
|---|---|
| `cargo test --release` (CPU) | **405 passed / 0 failed / 23 ignored** unit (baseline 399; +2 `copystats`, +2 allocator, +2 scheduler) and **10 / 0 / 6** integration (unchanged) |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` (CPU) | **23 passed / 0 failed** (unchanged — every F5 gate that needs a boundary is device-gated or in the unit suite) |
| `cargo test --release --features cuda -- --test-threads=1` (GB10, whole unit suite) | **459 passed / 0 failed / 26 ignored** (baseline 452 / 0 / 25; +6 CPU-visible tests, +1 new device test, +1 new `#[ignore]`d real-model gate) |
| `cargo test --release --features cuda --bin minfer -- --ignored --test-threads=1` (0.5B) | **26 passed / 0 failed** (baseline 25 / 0, +the F5 real-model gate) |
| `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf … --ignored --test-threads=1` | **25 passed / 1 failed** at this commit — the single failure was `server::batch::tests::a_slot_snapshot_resumes_the_context_without_re_prefilling` with *"the file was written with the f32 KV element type, this run uses f16"*, i.e. the then-pre-existing [#130](https://github.com/yusiwen/minfer/issues/130), not this ticket (**closed 2026-09-25**, C5 S3: **31 / 0**). The F5 gate itself passes in that configuration |
| `rustfmt +stable --edition 2021 --check` on the 10 changed `.rs` | clean (rustfmt **1.9.0-stable**; the pinned 1.97.1 toolchain has no `rustfmt` component here, as F4 found — CI runs no fmt job) |
| `python3 scripts/check_docs_links.py` | **935 links resolve in 183 files** (baseline 935 / 183) |

**The numbers the ticket asked for.** The real-model gate
(`async_cross_copies_never_block_and_stay_bitwise_identical`) runs the same
4-of-24-block offload graph 7 forwards (1 prefill + 6 decode) in both modes and
compares every step's logits:

| | async (default) | sync (`MINFER_SYNC_COPIES=1`, the pre-F5 path) |
|---|---|---|
| staging copies (`copies`) | 56 | 56 |
| boundary waits (`waits`) | 56 | 56 |
| **blocking device→host copies** | **0** | **7** |
| async device→host copies | 7 | 0 |
| event waits (host, the documented point) | 7 | 0 |
| device-level blocking readbacks | **0** | 7 |
| full stream syncs | **27** | **34** |
| max \|Δlogit\| vs the other mode | **0** (bitwise) | — |

So the before/after is: **7 → 0 blocking device→host copies** and **34 → 27 full
stream synchronizations** (−7, exactly one per staged device→host copy) over 7
forwards, with byte-identical logits. The cheap device gate
(`cuda_backend::tests::a_split_graph_waits_once_per_staged_copy_and_stays_bitwise`)
reports the same shape on a 3-node CPU→CUDA→CPU graph: async `copies=2, waits=2,
blocking=0, async_host=1, event_syncs=1, readbacks=0, syncs=1`; sync `copies=2,
waits=2, blocking=1, readbacks=1, syncs=2`.

**Mutation evidence (reverted; the tree was restored byte-identical).** The
boundary's phase-B loop was removed from `execute`:

```text
graph::cuda_backend::tests::a_split_graph_waits_once_per_staged_copy_and_stays_bitwise
  panicked: called `Result::unwrap()` on an `Err` value: "staged cross-backend input 0
  for Cuda was read before its boundary wait: every copy_across owes one await_cross (F5, #58)"

models::qwen2::graph::tests::async_cross_copies_never_block_and_stay_bitwise_identical
  panicked: called `Result::unwrap()` on an `Err` value: "staged cross-backend input 3
  for Cuda was read before its boundary wait: every copy_across owes one await_cross (F5, #58)"
```

**Which half of the missing-wait gate is deterministic, and which is not.** The
*loud refusal* above is deterministic: the pending flag left by the missing wait
is turned into an `Err` by `cross_input` on the consumer path, on the first read.
The *counter* half (`all_copies_awaited()`, i.e. `copies == waits`) is also
deterministic. The *bitwise comparison against the synchronous reference* is the
probabilistic half: a dropped event wait **can** also corrupt bytes, but whether
it does depends on timing, so it is reported as evidence of equality between the
two supported modes, never as the failure mode of the mutation. That is why the
gate carries both.

**Honest scope.** (a) **Metal is unported, deliberately**: there is no Mac here
and no macOS toolchain, so its `copy_cross` **declines** (`Ok(false)`) and the
allocator's synchronous host round trip handles it exactly as before F5 — no
half-written blit/event code that nothing can compile. A Metal source's copies
therefore still count as `blocking_host_copies`; the port is
[#137](https://github.com/yusiwen/minfer/issues/137). (b) **True overlap did not
land and is not claimed**: the split loop is strictly sequential (enqueue, then
wait), so there is no independent work for a copy to overlap with, and a
host-side consumer must wait by definition. What the substrate buys today is that
the transfers are enqueued back to back and the redundant per-copy stream syncs
disappear — the 34 → 27 measurement above. Deferring a wait to the consumer's
first use is [#138](https://github.com/yusiwen/minfer/issues/138). (c) **Only
CUDA has an async device path**, and only the CUDA→host direction (a device→device
staging copy is unreachable — `copy_across` early-returns when source and
destination backends match; CUDA→Metal on a macOS+CUDA build declines and stays
synchronous, the pre-F5 behaviour). (d) The CPU is a **synchronous no-op by
construction** — it has no device memory — and the gate that says so is
`scheduler::tests::the_cpu_path_never_enters_the_cross_copy_machinery`: a
CPU-only graph is one split, both modes are bitwise identical and every counter
stays zero. (e) The **latency** claim is a count, not a timing: the per-copy full
`cudaStreamSynchronize` is gone (measured as a sync count) and no blocking
`cudaMemcpy` is issued; no wall-clock speedup is claimed, because at a boundary
the consumer immediately needs the bytes and the dominant cost is unchanged.
(f) The end-to-end **counter** assertion needs a real cross-backend boundary, so
it runs on the CUDA device (`#[ignore]`d); CI (CPU-only) covers the
allocator-level missing-wait invariant plus the graph-level refusal through
`execute` with an injected pending entry
(`scheduler::tests::a_staged_boundary_input_cannot_be_consumed_before_its_wait`),
and the CPU no-op/bitwise gate. The mutation's CUDA failure output above is the
record of the part CI cannot reach.

### F6 — Quantizer tooling: convert, quantize, split (#49) — **DONE 2026-09-24**

**What landed.** Four new modules and the subcommands that use them:

- `src/gguf_write.rs` — the GGUF v3 **writer**. The reader in `src/gguf.rs` is
  the contract: magic/version/n_tensors/n_kv, the KV encoding for every one of
  the 13 `GgufType`s (including arrays of strings and of the numeric types), the
  tensor index (`name`, `n_dims`, `ne`, `type`, `offset`), `ggml_pad` alignment
  before the data section and after every tensor, and the multi-part convention
  (`split.no` / `split.count` / `split.tensors.count`, `{stem}-NNNNN-of-MMMMM.gguf`).
  `write_split` assigns tensors to parts greedily by padded size, never splits a
  tensor, gives every part the full metadata (so each parses standalone), and
  preserves the global tensor order so the merged index equals the single-file
  index; a one-part assignment is written as a plain single file with no
  `split.*` keys at all. The writer validates shapes/names/payload sizes and
  refuses a wrong-length payload instead of shifting every later tensor.
- `src/quantize.rs` — the **weight encoders**. `q4_0`, `q4_1`, `q5_0`, `q5_1`,
  `q8_0` (plus the `f16`/`f32` element casts), implemented as llama.cpp's
  `quantize_row_*_ref` (the CPU reference), with `QuantTarget::parse` refusing
  every other GGUF type **by name** ("known GGUF type but minfer has no weight
  encoder for it … writing it would emit wrong weights").
- `src/convert.rs` — the **HF → GGUF converter** (safetensors parsed as a
  length-prefixed JSON header plus raw bytes; `serde_json` only, no Python and
  no ML framework) and `QuantizePlan` (re-encode an existing GGUF, with the
  llama.cpp "1-D tensors stay f32" rule and the tied-embedding → q8_0 policy).
  The metadata it writes is what the F7 strict tokenizer/template loader reads:
  `tokenizer.ggml.model = gpt2`, `.pre = qwen2`, the full `vocab_size` token
  array with llama.cpp's token types, merges, special ids, `add_bos_token` and
  `tokenizer.chat_template`. Unknown architecture, tensor name or dtype is a
  refusal.
- `src/tooling.rs` — `convert` / `quantize` / `split` and the F6 gates.
- One engine change the acceptance forced: **f16 weights now run**. The CPU
  graph path had no f16 weight dispatch at all (only f32 and the quants), so
  `vec_ops::mat_mul_f16` decodes a weight row at a time and `cpu_backend`
  dispatches it for `Op::MatMul`, and `Op::GetRows` decodes f16 embedding rows.
  Without it "a converted model produces the same logits" was impossible for the
  f16 output the ticket names.
- `download::check_downloaded_size` — the second acceptance line. `http_download`
  used to ignore the expected size it was passed; now a downloaded file whose
  length differs from the remote's is an error and is **removed**, so it cannot
  be mistaken for a cached complete file.

**Measured acceptance (CPU; GB10 sm_121 for the CUDA row).** The references and
tolerances are stated in `docs/GGUF-TOOLING.md` §4.

| Command | Result |
|---|---|
| `cargo test --release` (CPU) | **432 passed / 0 failed / 28 ignored** unit (baseline 405 / 0 / 23; +27 F6 tests, +5 F6 real-model gates) and **10 / 0 / 6** integration (unchanged) |
| `cargo test --release --bin minfer -- --ignored --test-threads=1` (CPU) | **28 passed / 0 failed** (baseline 23; +5 F6 real-model gates) |
| `cargo test --release --features cuda -- --test-threads=1` (GB10 sm_121) | **486 passed / 0 failed / 31 ignored** (baseline 459 / 0 / 26) |
| `cargo test --release --features cuda --bin minfer -- --ignored --test-threads=1` (0.5B) | **31 passed / 0 failed** (baseline 26 / 0, +the 5 F6 gates) |
| … with `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf` | **30 passed / 1 failed** at this commit — the single failure is the then-pre-existing [#130](https://github.com/yusiwen/minfer/issues/130) (`the file was written with the f32 KV element type, this run uses f16`), as at the baseline; **closed 2026-09-25** (C5 S3: **31 / 0**) |
| the F6-produced q8_0 file on the device | `CUDA GATE: …` is **not** printed; `offload: all 24 blocks + embed/output on cuda (500.8 MiB of device weights)`, 1214.7 tok/s prefill, greedy `Paris.` — the same text as `MINFER_DISABLE_CUDA=1`. The **f16** file on the same build prints `weight 'token_embd.weight' (type F16) has no CUDA kernel or is not registered` and `running on CPU` ([#141](https://github.com/yusiwen/minfer/issues/141)) |
| HF → GGUF vs `convert_hf_to_gguf.py` (Qwen2.5-0.5B-Instruct, bf16 → f16) | **290/290 tensor payloads byte-identical** (sha256 per tensor); metadata equivalent for every loader-read value (llama.cpp also writes the cosmetic `general.size_label`, and key order differs) |
| minfer vs llama.cpp f16 file, logits after a 4-token greedy continuation | **bitwise identical** (`assert_eq!` over 151,936 logits) and identical greedy text `[12095, 13, 1084, 374]` |
| rewrite a cached GGUF with the writer | 291/291 tensors byte-identical, metadata key-for-key equal, logits **bitwise identical** |
| split the cached 0.5B into 4 parts | merged index **exactly** the single-file index (name/shape/type/nbytes, in order); logits **bitwise identical**; missing part / wrong `split.no` / filename-vs-`split.count` mismatch each fail the load |
| `minfer quantize` vs `llama-quantize` on the same f16 source | **q4_0, q4_1, q5_0, q5_1, q8_0 each 290/290 tensors byte-identical** |
| f16 → q8_0 end-to-end | greedy continuation identical; max \|Δlogit\| **0.481** (mean 0.082) against max \|logit\| 18.43 — stated bound ≤ 1.0 absolute |
| download size gate | pure test plus an end-to-end local HTTP server (correct `206` resume accepted; a server that ships the whole body for a range is rejected and the file removed) |
| `rustfmt +stable --edition 2021 --check` on the 8 changed/new `.rs` | clean (rustfmt **1.9.0-stable**; the pinned 1.97.1 toolchain has no `rustfmt` component here — CI runs no fmt job) |
| `python3 scripts/check_docs_links.py` | **940 links resolve in 184 files** (baseline 936 / 183; the new doc + its registration) |

**The FPE detail worth recording.** The first q4_0 encoder differed from
`llama-quantize` in **12 of 64512 bytes** on one tensor — every difference a
single nibble off by one. The cause was not the algorithm but the *compilation*:
llama.cpp's reference is compiled with `-ffp-contract=fast`, so `x*id + 8.5f`
becomes an FMA on aarch64/x86, while Rust's two separate operations rounded
twice. `f32::mul_add` reproduces it and the difference went to zero. Q8_0 uses
a single multiply and matched without it. A quantizer that is "close" is a wrong
file, which is why the gate is per-tensor byte equality.

**Mutation evidence (reverted; the tree was restored byte-identical).** Each new
gate was broken in the way it guards and observed to fail:

| Mutation (reverted after each run) | Gate that failed (observed output) |
|---|---|
| writer pads between tensors to 16 while the index declares 32 | `gguf_write::tests::tensor_index_offsets_match_the_parser_requirement` — `left: 880, right: 896`: the file is 16 bytes short of the layout its own index declares |
| tensors assigned to split parts in reverse order | `gguf_write::tests::split_writes_parts_the_reader_merges_back` — `left: 3, right: 1`, part 0 holds `t2` |
| `split.count` written as 1 for a 3-part split | the same test panics inside `load_gguf_model`: *split count mismatch: filename implies 3 parts, split.count = 1* |
| q4_0 encoder without `mul_add` | `f6_quantize_encoder_is_byte_identical_to_llamacpp` — *tensor `blk.0.attn_k.weight` payload differs* (the 12 bytes) |
| `check_downloaded_size` returns `Ok` unconditionally | `a_wrong_size_file_is_refused_and_a_right_size_file_is_accepted` **and** `http_download_resumes_a_partial_file_and_size_checks_it` both FAILED (the oversized / partial file is accepted) |

The reverts were byte-identical (`sha256sum` of each mutated file before and
after matches `HEAD`). Two of the gates had to be **strengthened to catch their
mutation**, which is the point of the exercise: the first padding gate only
checked the index the writer *declares* (the parser recomputes the same offsets
and never reads past the last tensor, so a short inter-tensor pad parsed fine),
so the test now also asserts `file length == data_offset + size` — for the
single file and for each split part; and the first reverse-order mutation landed
in `validate_specs`' loop rather than `split_assignment`'s, so it proved
nothing until the anchor was made specific. The table above is the re-run
output against the committed tests.

**Honest scope.** (a) **K-quants cannot be written.** The encoders are the five
legacy types; `q4_K`/`q5_K`/`q6_K` and every I-quant are refused by name
([#140](https://github.com/yusiwen/minfer/issues/140)). Reading them is
unchanged. (b) **f16 weights are CPU-only.** The CPU path now decodes them, but
the Metal/CUDA weight registration accepts f32 and the supported quants only, so
an f16 model runs the CPU path on a device build and that path is slow
(measured ~3 tok/s prefill on the 0.5B); the device row therefore exercises a
`minfer quantize`-produced q8_0 file ([#141](https://github.com/yusiwen/minfer/issues/141)).
(c) **bf16 output is refused** ([#142](https://github.com/yusiwen/minfer/issues/142));
f32 preserves every bf16 value exactly, so the conversion itself loses nothing.
(d) The exactness claims are named per step: f16/f32 copies and f16→f32,
bf16→f32 are bit-exact; **bf16→f16 is exact in the mantissa but can overflow**
(no saturation, so an out-of-range value becomes inf rather than a wrong finite
weight); **f32→f16 is not exact**. (e) The HF reference is llama.cpp's converter
output (byte-identical weights) plus, for logits/text, llama.cpp's own run —
transformers was installed but not used as the logit reference, because the
f16-vs-f16 byte comparison against llama.cpp's converter is the stronger claim.
(f) The five real-model gates are `#[ignore]`d (they need the checkpoint and/or
the cached 0.5B); CI covers the writer/encoder/converter/download unit and
local-HTTP gates.

### F6b — f16 weights on the device backends + a vectorized CPU f16 dot (#141) — **DONE 2026-09-25**

**What landed.** The third item F6 left open, plus the discovery that the fused
concat registration was already fine but the type gate was not.

- **CUDA**: `f16_f32_matmul_vec` / `f16_f32_matmul_scalar` in `cuda_kernels.cu`
  (the same NR0/NSG unit mapping and token-in-block loop as the f32 kernels;
  `__half22float2` + FMA, accumulation in f32) and `embed_rows_f16` (one thread
  per output element). `matmul_f32_ptr_layout` gained the `F16` arm and
  `embed_rows_on_gpu` the f16 gather; the loader registers `TensorType::F16`
  **raw** — deliberately its own branch, not folded into the quantized
  `matches!`, whose q4_K dsc-plane gate has no type check and would expand
  misinterpreted bytes from an f16 tensor into a plane no kernel reads
  ([#165](https://github.com/yusiwen/minfer/issues/165)). Both launchers read
  their own launch return through the #147 helpers and return non-zero, so a
  failed launch is an `Err` at the call site instead of joining the 65 unchecked
  `<<<>>>` sites of [#162](https://github.com/yusiwen/minfer/issues/162).
- **The design choice was a native kernel over dequant-at-registration**, and
  the reason is the memory trade the ticket asks to state: an f16 GGUF exists to
  halve the weight stream, and dequantizing to f32 on the device would give most
  of it back — 0.5B: 948 MiB of device weights as f16 against ~1.9 GiB as f32
  (14 GiB → 28 GiB for a 7B). The measured offload line for the 0.5B f16 file is
  942.4 MiB. f16 is not an MMQ *format* (MMQ streams quantized bytes and the
  f16-wmma GEMM is the `MINFER_MMQ=0` fallback for the *quantized* types), so an
  f16 prefill runs the f32-activation kernel at every `nt` rather than the int8
  GEMM.
- **Metal is a deliberate refusal, not a partial port.** The loader does not
  register f16 there, so `Qwen2Graph::weights_on_gpu` fails its all-or-nothing
  check and an f16 GGUF prints the loader's *"weights are not usable there —
  running on CPU"* line. Registering a weight type no kernel can consume would
  make the device claim true while the op ran the wrong (or no) kernel, which is
  exactly what that gate exists to prevent; a Metal f16 matmul/embed kernel
  cannot be verified from this box (no Mac; CI's `build-macos` compiles the crate
  and nothing runs it). Filed as [#164](https://github.com/yusiwen/minfer/issues/164).
- **CPU**: `vec_ops::dot_f16_f32` (AVX2 `F16C` `_mm256_cvtph_ps` / aarch64
  baseline NEON `FCVTL` `vcvt_f32_f16`, f64 scalar oracle) and
  `vec_ops::decode_f16_row`, with `mat_mul_f16` decoding each weight row once for
  `nt > 1` and folding every token into it, and the row loop going to the shared
  worker pool (`kernel::par_for`, the same `MIN_PARALLEL_MACS` threshold the
  quantized matmul uses). The SIMD loops run the same FMA tree in the same order
  as `vec_dot_f32`, so the f16 dot is **bit-identical** to `vec_dot_f32` over the
  decoded row — which is what makes the row loop safe to reorder and to thread:
  `n` never changes a value, asserted. `MINFER_NO_F16_ROWB=1` keeps F6's
  per-(row, token) shape as the A/B control.

**Measured acceptance (GB10 CPU; GB10 sm_121 for the CUDA rows).**

| Command | Result |
|---|---|
| `cargo test --release` (CPU) | **445 passed / 0 failed / 30 ignored** unit (baseline 440 / 0 / 29; +5 unit tests, +1 `#[ignore]`d device gate) and **10 / 0 / 6** integration (unchanged) |
| `PARALLEL=0 scripts/real_model_gates.sh` (CPU, serial) | **30 passed / 0 failed** (baseline 29 / 0) |
| `PARALLEL=1 scripts/real_model_gates.sh` (CPU, parallel) | **30 passed / 0 failed** (baseline 29 / 0) |
| CPU f16 prefill, before → after (34-token prompt, medians of 3 interleaved rounds per mode) | **3.2 → 207 tok/s** (10.72 s → 0.16 s). The vectorized dot alone is 25.3 tok/s; the row blocking alone measured within noise; the pool is the rest. Decode 2.2 → ~10 tok/s |
| `cargo test --release --features cuda -- --test-threads=1` (GB10 sm_121) | **513 passed / 0 failed / 33 ignored** (baseline 508 / 0 / 32) + **10 / 0 / 6** integration |
| `FEATURES=cuda scripts/real_model_gates.sh` (0.5B config) | **33 passed / 0 failed** (baseline 32 / 0) |
| `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf FEATURES=cuda scripts/real_model_gates.sh` | **33 passed / 0 failed** (baseline 32 / 0) |
| `compute-sanitizer --tool memcheck` over the serial CUDA unit suite | **0 API errors** over 513 passed / 33 ignored (357.69 s) |
| The f16 file on the device (`minfer convert --outtype f16` on the Qwen2.5-0.5B-Instruct checkpoint, 994,156,352 B) | offload report *"all 24 blocks + embed/output on cuda (942.4 MiB of device weights)"*; **169 f16 matmul + 1 f16 embed** nodes assigned `Backend::CUDA`; device vs CPU max \|Δlogit\| **7.34e-5** (mean 1.26e-5), 4.0e-6 relative, max \|logit\| 18.43; greedy `[12095, 13, 1084, 374]` identical on both |
| the same file under llama.cpp (`--temp 0`) | `Paris.` — the same greedy continuation minfer gives on CPU and CUDA |
| `rustup run stable rustfmt --edition 2021 --check` on the 6 changed `.rs` | clean (rustfmt **1.9.0-stable**; the pinned 1.97.1 toolchain has no `rustfmt` component here — CI runs no fmt job) |
| `python3 scripts/check_docs_links.py` | **940 links resolve in 184 files** (unchanged — no new file, only absolute issue URLs) |
| `cargo check` / `cargo rustc --emit=obj` for `x86_64-unknown-linux-gnu` | the AVX2+F16C path type-checks and **codegens** (the cross build reaches the link stage, where no `cc` cross-linker exists); running it is CI's ubuntu job |

**The gate asserts placement directly, not through a timing.**
`f141_f16_weights_run_on_the_cuda_device` (in `src/tooling.rs`, `#[ignore]`d):
(1) `Qwen2Graph::device()` is `Cuda`; (2) the offload report says all 24 blocks +
embed/output are on the device; (3) every `F16` matmul / embedding node the
**scheduler assigns** is `Backend::CUDA`, counted by walking the built graph's
`CNode.backend` — so a registered-but-unsupported type that `supports_op` routes
to the CPU fails here instead of quietly measuring the CPU; (4) then the logits
against the same file's `Layers(0)` CPU run, at \|Δ\| ≤ 0.01 and ≤ 1e-3 relative
(measured 7.34e-5 / 4.0e-6), greedy identical.

**Mutation evidence (reverted; the files restored byte-identical).**

| Mutation | Gate that failed (observed output) |
|---|---|
| the loader's f16 registration gated off | `f141_f16_weights_run_on_the_cuda_device` — *left: Cpu, right: Cuda* at assertion (1) |
| the f16 arm removed from `matmul_f32_ptr_layout` | the same gate, panicking on the loud `Err("cuda: weight type F16 has no f32-activation matmul kernel …")` — not a silent fallback |
| `__half22float2` replaced with zeros for half of each 8-element chunk in `f16_f32_matmul_vec` | the same gate — *greedy continuation differs between device and CPU*, so a value-level fault cannot pass either |
| `dot_f16_f32`'s SIMD match replaced by a direct scalar call (path reporting intact) | `vec_ops::tests::f16_dot_uses_the_vectorized_path` — *f16_dot_path() reports Neon but the SIMD dot never ran (0 -> 0)* |
| the SIMD branch removed from `decode_f16_row` only | the same test — *…but the SIMD row decode never ran (1 -> 1)* |

The last two are why the vectorization gate does not read `f16_dot_path()`
alone: the SIMD entry points (dot **and** row decode) bump a test-only
thread-local counter, so a dispatch that *reports* the SIMD path while running
the scalar dot still fails. Reading the path function by itself was the first
version of this gate, and the mutation above passed it — a textbook "assertion
on something the code under test clears".

**Two findings worth recording.**

1. **The gate had to load under a namespace, and the reason is a real hazard.**
   The CUDA weight registry is process-global and name-keyed, and
   `register_weight` reuses a same-name+same-size device copy. The other
   real-model gates in the `#[ignore]`d set load the cached q4_k_m 0.5B under the
   default `ns=""`, which shares **121 f32 norm/bias names and their byte sizes**
   with the converted f16 file — but not their values. The f16 gate therefore
   failed *only in the full serial set* (device greedy `[3110, 31139, 47, 34369]`
   against the CPU's `[12095, 13, 1084, 374]`) and passed when run alone: the
   device arm was computing with the other file's norms. Loading both arms under
   `ns="f141:"` (the loader's own documented remedy for a second model's
   name-keyed entries) fixes it. This is the process-global hazard
   [#64](https://github.com/yusiwen/minfer/issues/64) describes, and the shape of
   the failure — a gate that passes alone and fails in the set — is worth
   remembering.
2. **The cached `qwen2.5-0.5b-instruct-q4_k_m.gguf` is not weight-identical to
   the `Qwen/Qwen2.5-0.5B-Instruct` HF checkpoint.** Its f32 norms differ
   (`blk.0.attn_norm.weight[0]` = `-0.046875` in the HF-derived f16 file — the
   exact bf16 value in the safetensors — against `-0.082947` in the cached
   file), which is what turned finding 1 into a wrong-weights comparison rather
   than a last-bit one. No gate compares across the two files, so nothing is
   red; a future gate that does must not assume they are the same weights.

**Honest scope.** (a) **Metal has no f16 weight kernels**, so f16 is refused
there and falls to the CPU loudly ([#164](https://github.com/yusiwen/minfer/issues/164));
that is a stated policy, not an untested implementation. (b) The device gate's
tolerance is a **backend** tolerance: both paths compute f32 activations against
f16 weights (an f16 weight has no integer form, so the CPU does *not* quantize
its activations the way it does for the quantized types), so what remains is
accumulation order plus the attention exp/softmax kernel — measured 4.0e-6
relative, bounded at 1e-3 with ~250x headroom. (c) The CPU f16 path is
vectorized and pooled but **not blocked over the K dimension**, and the row
blocking that is there measured neutral on this model; a K-tiled kernel that
reuses a weight row across a token *tile* without re-reading it is not
implemented. (d) `x86_64` codegen is verified by cross-`cargo`, not by running
the AVX2 kernel — CI's ubuntu job is the run. (e) The row-blocked and direct
forms are asserted bit-identical on this box; the argument that they must be
(identical FMA tree and order) is also why the SIMD/scalar comparison uses a
tolerance rather than bit equality. (f) The pre-existing finding filed with F6b
is **fixed in F6c** ([#165](https://github.com/yusiwen/minfer/issues/165)): the
q4_K dsc plane was built for non-q4_K types with passing geometry.

### F6c — the q4_K dsc plane is gated on q4_K, with a payload contract (#165) — **DONE 2026-09-25**

**What landed.** [#165](https://github.com/yusiwen/minfer/issues/165), found while porting #141
(F6b) and deliberately kept out of that branch (which is why f16 was not folded into the loader's
quantized `matches!`).

- The qwen2 loader built the r59 `W_dsc` f32-pair plane inside the `else` of
  `if ttype == TensorType::Q6_K`, so the registration gate was reached for **every** non-Q6_K type
  the enclosing `matches!` admitted (Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q8_0), with no type check on the
  plane. Any of them whose geometry passed (`id % 256 == 0`, `od` even) had 144-byte q4_K
  super-blocks decoded out of its bytes and uploaded — a plane **no kernel reads** (the `q4k_dsc`
  map is keyed on the q4_K weight's device pointer and its only consumer is `mmq_raw_nb_bt`), at
  device-memory and host-CPU cost per tensor.
- The fix is one pure rule with two load-bearing halves, `src/q4k_dsc.rs::q4k_dsc_plane_admitted`:
  `ttype == TensorType::Q4_K` **and** `raw.len() == od * (id / 256) * 144` — q4_K's own block
  layout, an **equality and not a lower bound**. The qwen2 loader calls it;
  `register_weight_q4k_dsc` re-checks the payload before the budget query and before the host
  expansion, and `expand_q4k_dsc` itself returns `None` for a payload it cannot index, so a direct
  caller cannot bypass either. The rule is a free function in a **non-`cuda`-gated** module
  precisely so CI's CPU job *runs* its tests; the CUDA job only compile-checks the device modules.
- Why both checks. q4_0's bytes/element equals q4_K's exactly (18/32 == 144/256), so the size check
  is blind to the difference — the type gate is the only thing that can refuse it. A q8_0 payload
  (34/32) is *longer* than the row arithmetic needs and was misread; the size check refuses it. A
  future type with a *smaller* ratio (a 2-bit K-quant: 84/256) is *shorter* and is refused instead
  of read past the tensor — the latent OOB #165 names. What the size check **cannot** do is tell a
  q4_K payload from another type's bytes of the same length; that is the type gate's job.

**Measured acceptance (GB10, sm_121, CUDA 13.0, driver 580.178.04).** The "before" numbers are the
real pre-fix code path (the two gate halves reverted, everything else identical).

| Criterion | Before | After |
|---|---|---|
| `q4dsc_planes()` after loading the cached 0.5B **q4_0** (qwen2 arch) | **24 planes / 26 148 864 B** | **0 / 0** |
| `q4dsc_planes()` after loading `/tmp/fix165/qwen2.5-0.5b-instruct-q8_0.gguf` (a qwen2 q8_0 built by `minfer quantize` from the cached 0.5B q4_0) | **24 / 26 148 864 B** | **0 / 0** |
| `q4dsc_planes()` after loading `Qwen3-0.6B-Q8_0.gguf` — the model #165 names | **0 / 0**: the qwen3 loader never had the call (honest scope below) | 0 / 0 |
| the cached 0.5B **q4_k_m** (the q4_K positive control, end to end) | — | **12 planes / 13 074 432 B**, exactly the GGUF index's admissible q4_K set |
| `cargo test --release --features cuda -- --test-threads=1` | — | **516 passed / 0 failed / 34 ignored** (baseline 513 / 0 / 33) |
| `FEATURES=cuda scripts/real_model_gates.sh` (0.5B and Qwen3-0.6B-Q8_0 configs) | — | **34 passed / 0 failed** both (baseline 33 / 0) |
| `compute-sanitizer --tool memcheck` over the serial CUDA unit suite | **0 API errors** over 514 passed / 2 failed / 34 ignored (the two new gates failing on the pre-fix path; 344.62 s) | **0 errors** over 516 passed / 0 failed / 34 ignored (349.38 s) |
| `cargo test --release` (CPU) | — | **447 passed / 0 failed / 30 ignored** unit (baseline 445 / 0 / 30; +2 pure `q4k_dsc` tests) + **10 / 0 / 6** integration |
| `PARALLEL=0 scripts/real_model_gates.sh` (CPU) | — | **30 passed / 0 failed**, unchanged (the new `#[ignore]`d gate is `cuda`-gated) |
| `rustup run stable rustfmt --edition 2021 --check` on the changed `.rs` | — | clean (rustfmt **1.9.0-stable**; the pinned 1.97.1 toolchain has no `rustfmt` component, CI runs no fmt job) |
| `python3 scripts/check_docs_links.py` | — | **940 links resolve in 184 files**, unchanged |

**Gates.** Two pure tests in `src/q4k_dsc.rs` (CI's CPU job): the q8_0-length and wrong-type
refusals with a q4_K positive control and a Q5_K wrong-type control, and the short/long/empty
payload refusals. `graph::cuda_backend::tests::cuda_q4dsc_plane_is_q4k_only` (device): the q4_K
control registers `{name}__q4dsc{od}x{id}` and the **same** `q4dsc_planes()` query the refusals use
sees exactly that plane, while the q8_0-length and one-block-short payloads add **nothing** — a
query blind to planes could not see the control either. The `#[ignore]`d
`cuda_real_model_registers_q4dsc_planes_only_for_q4k` loads a real model and asserts the registered
plane set **equals** the GGUF index's admissible q4_K set (0 for q4_0/q8_0, 12 for q4_k_m).

**Mutations (reverted; every file restored byte-identical, `sha256sum`).**
(a) **type gate removed** (`q4k_dsc_plane_admitted` drops `ttype == Q4_K`): the pure wrong-type
assertion fails (*"the type gate must refuse a q8_0 weight regardless of its length"*) and the
real-model gate fails on the 0.5B q4_0 at **24 planes / 26 148 864 B** against 0 expected — the
mutation reproduces #165 exactly.
(b) **size validation weakened** (exact equality → `payload_bytes >= want`): the pure
*"one block long"* assertion fails and the device gate fails at
*"a q8_0 payload must not register a __q4dsc plane"* — so the gate really tests exactness, not a
lower bound.
(c) **wrong plane name** (`__q4dsc` → `__q4dscX`): the device gate fails at its *positive control*
(*"the q4_K payload must register f165q4k1024x3072__q4dsc1024x3072"*), which is what proves the
"nothing registered" arms observe the plane's real registry entry rather than passing vacuously.

**Honest scope.** (a) **The model #165 names does not reproduce the defect**: `Qwen3-0.6B-Q8_0` is
arch `qwen3` and goes to `src/models/qwen3/loader.rs`, which has **no** `register_weight_q4k_dsc`
call at all — its plane count is 0 before and after (measured). The ticket's "~22 MB on
Qwen3-0.6B-Q8_0" is the qwen2 loader's arithmetic applied to the qwen3 model's `ffn_down` shape; the
defect is qwen2-loader-only. The qwen2 q8_0 arm is measured on a q8_0 file built here with
`minfer quantize` (from the cached q4_0 0.5B, since a K-quant source is not re-quantizable) and both
qwen2 arms are named in the table. (b) **The size check cannot tell a q4_K payload from another
type's bytes of the same length** — q4_0 shares q4_K's ratio exactly, which is why the type gate is
not redundant; the pair is the contract, and only the pair is tested. (c) The `#[ignore]`d real-model
gate assumes the default full offload plan (all blocks fit), which holds for every cached small
model it runs on; a *partial* plan would register fewer planes than the GGUF index implies. (d)
**A separate loader divergence is filed, not fixed**:
[#167](https://github.com/yusiwen/minfer/issues/167) — the qwen3 loader lacks both the q4_K dsc
plane this record gates and the f16 registration branch #141 gave qwen2, so a q4_K Qwen3 runs the
in-kernel scalar dsc decode and an f16 Qwen3 model drops to the CPU on CUDA (read from the loader,
not device-verified — no f16 Qwen3 GGUF is cached here).

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
