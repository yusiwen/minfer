# minfer Architecture Execution Plan

**Status:** Phase A **complete** (9/9, 2026-09-16); Phase B **complete** (3/3,
2026-09-16); Phase C **complete** (8/8) — C1, C2, **C3**, **C4 (quantized Q8_0 cache,
CPU)**, **C5 (session save/restore)**, **C6 (logical positions)**, **C7 (+C7b)** and
**C8 (cross-sequence cell sharing)** are done. C6 merged 2026-09-20 as `001b8cc`;
**C7 landed 2026-09-20** (the partition is elastic, and growth moves runs in **both**
directions, so a busy neighbour above the slot no longer blocks it); **C8** split into
**C8a** (shared prefill, duplicated rows: no IR change) and **C8b** (paged sharing: a
block map and a gather in every attention kernel — S1a/S1b/S2/S3/S4/S5 landed, closed on
CPU and CUDA, Metal's share path at G5); **C4** and **C5** landed 2026-09-22 (C4's fused
dots and the CUDA/Metal kernels are [#87](https://github.com/yusiwen/minfer/issues/87);
the CLI/server surfaces C5 enables are [#89](https://github.com/yusiwen/minfer/issues/89)).
Phase D **complete** (3/3) (**D1 done**: views, multi-output via `split_parts`, D2, D3); Phase E
**complete** (7/7) (E1, E1b, E2, **E3**, **E4**, **E5**, E6 all done); Phase F **in progress** (7/8) (F2, F3, **F4**, **F5**, **F6**, F7,
F8 done; F1 needs x86); Phase G
**scheduled** (0/7) — after the CUDA
KV path, not before it (device claims need a Mac; CI's `build-macos` is the compile
check). **Next: the Metal round G1–G3/G5 (on a Mac) and F1 (needs x86).** The order is
deliberate: the Metal KV port (G5) comes **after** the CUDA arena stops changing shape
(C7, C7b, C8), so those semantics are written into Metal once. Per-ticket evidence is in
each phase's record and in the §14 open-risks table.
**Companion to:** `docs/ARCHITECTURE-ROADMAP.md` (what is missing, why, and how it
is ranked). This document is the *how*: phase-by-phase tickets with
deliverables, acceptance criteria and dependencies.
**Baseline:** `HEAD = f32daa7` (2026-09-16); Phase A landed on
`architecture-phase-a` (PR #1). This status was refreshed against `master =
f663750` (2026-09-26, the #175 merge); it is refreshed with every PR that lands a ticket.
**Derived facts:** the phase counters above, the `next:` sentence and the two commit
ids are not hand-maintained prose — [`docs/status.toml`](./status.toml) is the source of
truth, and `scripts/check_status.py --check` (CI job `check-docs`) fails when this block
disagrees with it, naming the file, the line and both values. Edit the source, not the
counter. The same checker reads `AGENTS.md`'s suite counts: `--check-live` compares the
one CI-verifiable box against the `test-linux-cpu` log, while the CUDA / real-model /
sanitizer rows are labelled *recorded measurements* because CI has no GPU. The
per-ticket ✔ marks in the §11 diagram are **not** derivable from `(done, total)` and are
deliberately out of scope for the checker.

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

## 2. Verification matrix (what dgxspark can prove)

`dgxspark` — the box every recorded measurement in this plan was taken on — is a
**DGX Spark (GB10), aarch64 Linux, CUDA 13.0**
(`/usr/local/cuda-13.0`), with cached Qwen2.5-0.5B/7B/14B and Qwen3-0.6B GGUFs.
Measured records name it absolutely, never as "this box" (gate contract rule 5).

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
| A0 | — | CUDA access spike on dgxspark | S | ✅ done — but the verdict is **superseded (2026-09-18)**: the device is available; "unavailable" was an agent-sandbox artefact (§2) |
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
  it lives in the module tree rather than under `tests/` — the crate is a binary
  target, so a file under `tests/` is a **separate crate** that cannot name the
  binary's items at all (there is no `lib` target to link against). The test must
  therefore be an in-crate module.
- **Three tests:**
  1. `matrix_cases_match_their_reference` — 17 cases (Add, Mul, Scale, Silu,
     SwiGLU, Softmax, RmsNorm, QkNorm, MatMul, GetRows, View/Reshape/Permute,
     RoPE, Attn, KvcacheStore/Load) run on **every backend that claims the op**,
     each compared against an analytic reference written in the test — never
     against another backend. Unavailable backends report
     `SKIP (reason)`, never `PASS`: on dgxspark that is 17 CPU cells + 34 skips.
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
- **Container base moved (2026-09-30):** the image tag above is now
  `nvidia/cuda:12.8.2-devel-ubuntu24.04` — deliberately the same image the x64
  release job builds in (`.github/workflows/release-build.yml`, `linux-x64-cuda`).
  The 22.04 tag was glibc 2.35 / gcc 11 against the artifact's 2.39 / gcc 13, so
  the compile check ran on a base no artifact was ever built on. The rest of
  this record stands as written.

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
  which dgxspark does not have (A0: CUDA unavailable; Metal not compiled). The
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
- **Not covered here:** the fused GPU stores are unverified on dgxspark, so the
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
   On dgxspark that means CPU first: CUDA was compile-verified only at the time
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
- *Refusals*: `MINFER_CACHE_TYPE=q8_0` on dgxspark (CUDA available) ends the load with
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

**Handed off after S2b:** the packed **fused decode epilogue**, a **dp4a packed K dot** and the
**FA prefill on packed cells** — taken up as [#144](https://github.com/yusiwen/minfer/issues/144)
and recorded in the next subsection (items 1 and 3 landed; item 2 stays open).
[Metal's half stays at G5](https://github.com/yusiwen/minfer/issues/44).

### C4 — #144: the packed fused decode epilogue and the packed FA prefill · [#144](https://github.com/yusiwen/minfer/issues/144) — **DONE (items 1 + 3) 2026-09-26**

**Why.** C4 S2b's packed cache is correct and 3.76x smaller than f32, but three paths were off their
tuned route: a Q8_0 decode ran the unfused bias/rope/store chain (the builders' `layer_gpu` gate
carried `&& !packed`), `kv4<Q8_0>` paid four int8 converts where f16 paid two `__half2`, and an hd-128
packed prefill could not enter `fa_prefill_f16kv`'s f16 shared-memory staging, so it fell to the
general layout-tagged kernel — 15.3x off on Qwen3-0.6B `pp2048`. This ticket took the first and third
(a dp4a K dot is a numerics change with its own accuracy statement; it is filed separately).

**What landed.**

1. **The packed fused decode epilogue.** `attn_bias_rope_store_q8_0` keeps the f32/f16 epilogue's q
   section (bias + rope in place, one thread per pair) and re-maps K and V to **one thread per
   (head, 32-element block)**: the thread computes all 32 K roped values itself (from the unroped row
   plus the pair partner `d <-> d + hd/2`) or the 32 V bias-added values, and hands them to
   `q8_0_quantize_block` — the *same* device quantizer `store_kv_q8_0` now calls, factored out so the
   two cannot drift. Neither K nor V is written back (both buffers are dead in the fused classes) —
   which is also why the packed epilogue has no rope race. The builders dropped `&& !b.kv_is_packed()`
   from the Qwen2 family's `fuse_qkv` gate. `FusedQkvMeta` / `QkvBiasRopeStoreMeta` /
   `FusedQkvNormMeta` gained a `row_elems` field (the packed cell width, `KvFormat::row_elems(nkt)`)
   and the allocator sizes the region from it: without that, the first packed fused graph died with
   `the node declares 34 words per cell but the q8_0 layout packs one cell of 512 elements into 136`.
2. **The packed FA prefill.** `fa_prefill_f16kv` became `fa_prefill_kv<CAUSAL, MAP, LAYOUT>`: the
   staging is the only layout-dependent part (`kv8_q8_0` dequantizes each packed 32-element block
   into the same f16 tile, one 16-byte smem store per 8 elements); the tensor-core QK^T, the
   fragment-resident softmax and P·V are untouched. `gqa_attn_kv_prefill` serves f16 and Q8_0 from
   one call and keeps the general layout-tagged kernel as the documented fallback; the launcher's
   failed-smem arm and `MINFER_NO_FA_PREFILL=1` both reach it. The chokepoint bumps
   `testfail::note_checked("cuda_fa_prefill_q8_0")`, so the gate can prove the packed prefill took
   the FA route rather than silently falling back.

**The A/B protocol and the numbers.** All numbers: GB10 sm_121, `cargo build --release --features cuda`,
`minfer bench -p 2048 -n 128 --n-ctx 4096 -o json`, `MINFER_CACHE_TYPE` pinned per arm, **5 interleaved
rounds, medians**, 2026-09-26. The bars were named *before* measuring (gate contract rules 3 and 5);
the full protocol lives in
[`CUDA_OPTIMIZATION.md`](./CUDA_OPTIMIZATION.md) and
[`cuda_optimization_steps/107`](./cuda_optimization_steps/107-c4-packed-q8-kv-cuda.md).

The re-measured baseline (master `85c712e`, its own binary) reproduced the S2b table within a few
percent: 0.5B q4_0 `tg128` 237.37 f16 / 161.49 q8_0, `MINFER_NO_FUSE_QKV=1` f16 200.80 (a 1.182x cut —
the ticket's 1.18x); Qwen3-0.6B Q8_0 `pp2048` 8323.24 f16 / 564.47 q8_0.

| item | bar (named first) | measured | verdict |
|---|---|---|---|
| 1 packed fused epilogue | 0.5B `tg128` q8_0 ≥ 1.15x the unfused packed chain | 170.11 vs 161.63 same-binary = **1.052x** (vs the baseline binary 161.49 = 1.053x); f16 gap 1.470x → 1.393x | **landed, bar missed** |
| 3 packed FA prefill | Qwen3-0.6B `pp2048` q8_0 ≥ 0.5x the f16 arm | **8231.05** vs 564.57 same-binary = **14.58x**; vs the f16 arm 8540.83 = **0.964x**, i.e. the 14.7x factor became **1.038x** | **landed** |
| 1/3 no-regression | 0.5B `pp2048` q8_0 (hd 64: FA does not apply) flat | 2144.18 vs 2139.10 = 1.002x | flat |

**Item 1's miss is a result, not a failure.** The ticket's "~1.18x" was the fusion cut measured on the
**f16-weight** arm; on the q4_0 packed arm the same binary's fused-vs-unfused A/B is 1.052x. The
saving is the ~6 launches/layer the epilogue removes (~13 µs/layer at 24 layers ≈ the 0.32 ms/token
the two arms differ by), so the launch-overhead component is what this cut actually buys; the residual
1.39x f16-to-packed gap is the packed *load* (`kv4<Q8_0>`'s four converts) and the 1-warp split-K
decode body's `rpw_gate = 0`, which the dp4a item targets.

**Verification.** The two new device gates and the five named Q8_0 gates are green:
`cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer` (new: byte-exact K/V against the CPU quantizer at
`pos = 0`, a value q check at `pos = 7`, plus V byte-exact), `cuda_q8_0_fa_prefill_attention_parity`
(new: 100-token hd-128 prefill against the CPU attention over the same packed bytes, 5e-3 class,
**max err 4.8e-4**, plus the `note_checked` observation arm),
`cuda_q8_0_store_matches_the_cpu_quantizer`, `cuda_kv_q8_0_roundtrip_attn`,
`cuda_q8_0_kv_cell_move_strides_by_row_bytes`, the Q8_0 arm of
`cuda_map_window_matches_the_span_over_the_same_rows`, and `cuda_fa_prefill_attention_parity`
(the f16 route, unchanged, max err 2.8e-4). Mutations: reverting the fused kernel's block offset
(`d = blk*32 + i` → `d = i`) turns the byte arm red (it *found* exactly that bug during development);
shifting `kv8_q8_0`'s block base (`elem >> 5` → `elem >> 4`) turns the FA parity red (max err 3.33);
making the packed prefill skip the FA launch leaves the parity arm green (max err 5.8e-5) but turns the
observation arm red — which is the whole reason it exists. `MINFER_TEST_ISSUE162=1` drives all 124
audited launch sites (six new: three `fa_prefill_kv` layouts × modes and the packed epilogue) and is
green.

**Deliberately not taken.** The **dp4a packed K dot** (item 2): a per-head-quantized query against the
packed K accumulates in `int`, which is a numerics change needing its own accuracy statement and a
re-measured real-model tolerance — filed as a follow-up rather than assumed. **Qwen3's packed fused
QKV** (`Op::FusedQkvNorm`) also waits: that op has no CUDA kernel at all (it is Metal + CPU today), so
the `!packed` gate there is not what keeps Qwen3's decode off the device; its 1.13x packed gap is
unrelated to this ticket's cuts.

### C4 — #186: the dp4a packed Q8_0 K dot · [#186](https://github.com/yusiwen/minfer/issues/186) — **DONE 2026-09-27**

**Why.** [#144]'s residual was 1.393x on Qwen2.5-0.5B q4_0 `tg128` (170.11 q8_0 vs 236.97 f16),
and [#144] names two components: the 1-warp split-K body's `rpw_gate = 0` **plus** the
`kv4<KV_LAYOUT_Q8_0>` load. Only the load is addressable by a dp4a K dot. The trap is that a
whole-kernel q8_0-vs-f16 gap would mix the two: at hd 128 the f16 arm takes the 4-warp hybrid body
while the packed arm cannot. **hd 64** (the 0.5B) is the shape where both take the identical
`gqa_attn_split_partial<LAYOUT,true,false>` on the same grid, so the delta there is the load alone.

**The bar, named before measuring.** The load-attributable share of the Q8_0 decode step ≥ 10%.
Reasoning: only one of the residual's two components is addressable; a free load recovers at most
`share / 1.393`; dp4a removes only the K side's converts and multiplies, adds a query quantize, and
still pays the memory traffic; [#144]'s landed item 1 was 5.3% on this arm, the measured floor for
"worth landing". **Measured 20.3%** (nsys, real decode, node tracing: the incumbent packed partial
kernel is 67424 ns median per layer/token vs f16's 18528, = 1.182 ms of a 5.818 ms/token step).
Bar cleared.

**What landed.** `attn_split_1w_body<LAYOUT,CAUSAL,MAP,Q8DP4A>` (the `Q8DP4A` default `false`, so
the f32/f16 and verify instantiations are untouched) quantizes each lane's four query values against
its 32-element block's `amax` (the eight lanes sharing a block reduce it with three `shfl_xor`), reads
the K quants as `int8` through `kv4_q8_0_packed`, accumulates `__dp4a(qk, ki, 0)` and scales by
`d_q * d_k` once per block. V still dequantizes; only K changes. The launcher
`launch_gqa_attn_split_q8_0` takes an `int dp4a` and picks the instantiation;
`cuda::q8_kv_dp4a_enabled()` resolves `MINFER_NO_DP4A_Q8_KV=1` once per process (the same-binary A/B
control), and the Rust call site bumps `testfail::note_checked("cuda_q8_kv_dp4a")` only when it
launched the dp4a arm.

**The numbers** (GB10 sm_121, `cargo build --release --features cuda`,
`minfer bench -p 2048 -n 128 --n-ctx 4096 -o json`, 5 interleaved matched rounds, medians,
2026-09-27):

| arm | 0.5B q4_0 tg128 | Qwen3-0.6B Q8_0 tg128 | 0.5B pp2048 |
|---|---|---|---|
| incumbent (`MINFER_NO_DP4A_Q8_KV=1`) | 171.95 | 123.98 | 2149.22 |
| dp4a | **193.30** (1.124x) | **136.66** (1.102x) | 2150.37 (flat) |

The 0.5B f16 arm in the same round is 240.03, so the packed/f16 residual went 1.396x → **1.242x**;
on Qwen3-0.6B the f16 arm is 136.73, i.e. the packed decode now matches f16. The isolated probe
(same 1-warp geometry, load only) cuts the delta from +96% to +40% at hd 64, and `ncu` (collected as
root, the module parameter unchanged) names the residual: dp4a removes 494984 instructions (−15.3%)
and leaves the L1 load-sector count unchanged at 792904 — 1.71x f16's, while its L2 read sectors are
0.57x f16's — so the remaining 1.242x is the 34-byte block layout's L1 request count, not DRAM
traffic and no longer the arithmetic.

**Tolerance, re-measured not assumed.** `a_packed_kv_cache_answers_like_the_f32_one`'s CUDA arm reads
max |Δlogit| **2.479504** of a 37.82 spread, at the argmax **0.552662**, greedy 9/9 (incumbent arm:
2.479504 / 0.596050 / 9-of-9) — inside the ≤4.0 / ≤1.0 class. The max is unchanged because it is the
`nt = 512` **prefill** step's; the decode steps' deltas moved and nsys on the gate itself names
`gqa_attn_split_partial<(int)2,(bool)1,(bool)0,(bool)1>` under the default and `...,(bool)0>` under
the control, so the class is not "both arms ran the old path".

**Verification and refusal-to-over-claim.** The dp4a arm was added to
`cuda_kv_q8_0_roundtrip_attn` (two cells through an explicit span, an exactly Q8_0-representable
query, plus the observation-counter arm); the first version used one cell and was **vacuous** — a
mutated block base still passed, which is how the two-cell form was found. Mutation: `elem >> 5` →
`elem >> 4` → red at max |Δ| = 0.35126442. **Prefill/verify is deliberately not taken**: the
`nt > 1` general kernel also reads `kv4<Q8_0>` and measures 1.32x off its f16 arm on the 0.5B
prefill attention, but that is a different kernel (per token *and* head query, no shared block scale)
and not the ticket's residual — named rather than assumed. Full record:
[`cuda_optimization_steps/108`](./cuda_optimization_steps/108-c4-dp4a-packed-q8-kv-cuda.md).

### C4 — #202: the packed Q8_0 KV cell's L1 request count · [#202](https://github.com/yusiwen/minfer/issues/202) — **DONE 2026-09-27 (counter-only; throughput bar missed)**

**Why.** [#186]'s ncu attribution named the residual: the packed decode arm issued **1.71x** f16's L1
load sectors (~792904 vs 463008) with 0.57x its L2 sectors, because a 34-byte Q8_0 block's 4-element
group at `blk + 2 + (elem & 31)` is 4-byte aligned only on odd blocks, so `kv4<Q8_0>` spends four
`s8` loads plus a scattered scale load per group. The ticket listed three candidates and said to prefer
the one with no layout change if it clears the bar (a split plane or a 36-byte block would move the CPU
store/read path, `copy_cells`, `map_q8_0_cells`, the FA-prefill staging and the C5 session version).

**What landed.** The no-layout candidate: `34k + 2 + 4m` is **always 2-byte aligned** (`34k` is even
for every `k`, and `4m` is), so two `unsigned short` loads replace the four `s8` loads for the same
bytes. `q8_0_load4<WIDE>` / `kv4<LAYOUT,WIDE>` / `kv4_q8_0_packed<WIDE>` / a `Q8WIDE` template
parameter on `attn_split_1w_body`, and a third arm in `launch_gqa_attn_split_q8_0`, selected once per
process by `cuda::q8_kv_wide_enabled()` (`MINFER_NO_Q8_KV_WIDE=1` is the same-binary control). Three
new launch sites, driven by the #162 test and added to the audit fixture. **No layout, CPU,
copy-stride, session-format or `map_q8_0_cells` change.**

**The bar, named before measuring.** (a) packed/f16 L1 load-sector ratio ≤ 1.30x; (b) same-binary
`tg128` ≥ +2% over the #186 dp4a baseline 193.30, `pp2048` within 2%. **Measured: (a) met with margin
— 792904 → 455840, ratio 1.712x → 0.9845x, instructions −2.17%; (b) NOT met — 193.13 → 193.65
(+0.27%, 5 interleaved medians), the kernel itself only 1.4–1.7% faster (nsys 40320 → 39744 ns
median).**

**Honest result: a partial refutation.** The L1 request count moves exactly as the mechanism predicted
and the decode step does not follow, so the 1.23x packed/f16 residual is **not** L1-request-bound —
not L2 traffic (0.564x), not instruction count (1.10x), not load sectors (0.9845x). Landed
counter-only, byte-identical (the `nt > 1` general kernel keeps the byte form; only the dp4a decode
arm opts in), with the latency hypothesis (per-block scale load + `cvt` + V dequant on a 1-warp
block's critical path) recorded for the next attribution rather than claimed. Mutation: swapping the
two `u16` halves in `q8_0_load4_wide` turns `cuda_kv_q8_0_roundtrip_attn` red at max |Δ| = 2.5326836.
Full record: [`cuda_optimization_steps/109`](./cuda_optimization_steps/109-c4-packed-q8-kv-l1-request.md).

### Test infrastructure — #207: the CUDA count-row drift · [#207](https://github.com/yusiwen/minfer/issues/207) — **DONE 2026-09-27**

**Why.** The CUDA rows are `live_check = false` recorded measurements (CI has no GPU), but they count
the **same test binary plus the device-gated tests**: a CPU-only ticket that adds a feature-independent
test moves both. [#140]'s K-quant encoder tests (7 passed / 1 ignored) and [#142]'s bf16 writer +
CPU-path tests (7 passed / 2 ignored) did exactly that and left the row reading 548 / 0 / 39 while a
device run on the branch reads **562 / 0 / 42** and both CUDA real-model rows read **42 / 0**; this was
the second consecutive occurrence.

**What landed.** Mechanism (c), chosen over a `pending` field and a rule line: `docs/status.toml`'s
recorded rows may carry `projection_key` / `projection_box` / `projection_base_passed`, and
`scripts/check_status.py --check` **prints** (never fails on) `passed + (cpu_now − base)` when the CPU
twin has moved. The relation is inexact — the device-gated tests move independently — which is exactly
why it is a hint and not a comparison; the point is that a stale recorded row cannot look current
without a projection beside it. The unit/sanitizer rows project from `cpu-unit (aarch64)` and the two
real-model rows from `cpu-real-model (aarch64)`, each with that row's `passed` at this measurement as
the base, so all four hints are silent today and fire the moment a CPU ticket adds tests.
`--selftest` gained a moved/not-moved pair (12/12 cases).



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

> **#218 forward note (2026-09-29):** the **eager** sweep this record describes is gone.
> [#188](https://github.com/yusiwen/minfer/issues/188) deleted `gemm_prefill_smem_init`'s
> `CudaState::try_new` call site with no mention in any commit message or document, and
> [#218](https://github.com/yusiwen/minfer/issues/218) removed the orphaned function (and the
> `checked`/`skipped` introspection) instead of leaving it `allow(dead_code)`-annotated. The
> per-launch opt-in this record already describes — `gemm_smem_optin` → the shared
> `minfer_smem_optin`, reached by `launch_gemm_f16` — is now the whole mechanism; the invariant it
> must uphold (the attribute is in force **before** a capture window opens, never set inside one) is
> upheld by the 3-run capture warmup (`capture_warmup`), `cudaStreamCaptureModeThreadLocal` and the
> per-instantiation cache, and gated by `cuda_prefill_smem_optin_is_done_by_production`, the control
> arm `cuda_prefill_smem_optin_refusal_fails_the_prefill`, the coverage arm
> `cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation`, and
> `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window` — see the #218 record in
> §test-infrastructure. `gemm_dynamic_smem_bytes` remains the one formula; the "read twice" wording
> above is historical (the launcher reads it once).

> **#223 forward note (2026-09-29):** the eager half is back, but **not as the sweep this record
> describes**. `CudaState::try_new` now calls `gemm_prefill_smem_prewarm_one(tm, ks, af32)` for every
> launchable combination — a thin dispatcher onto the **same** production `gemm_smem_optin` cache the
> launcher reads — so there is one opt-in mechanism and one `cudaFuncSetAttribute` site, not a second
> sweep with its own copy of the formula (the pre-#145 bug class). The placement is the argument: at
> `try_new` no `CudaBackend` (and so no capture window) can exist, so the attribute is set outside any
> window by construction. The `checked`/`skipped` counters and the startup banner are still gone; a
> successful pre-warm is silent, and each failure or deliberate over-limit skip is named per
> instantiation. The lazy path stays as defence in depth, and the #218 gates keep their claims by
> running their children against the documented control `MINFER_NO_GEMM_PREWARM=1`. On the real binary
> the loop costs ~2.2 ms (the fatbin's one-time module load), which `prewarm_prefill()` already paid —
> net new startup cost ≈ 0, hot path within ±1% — see the #223 record in §test-infrastructure.

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

### Every `<<<>>>` reads its own launch error · [#162](https://github.com/yusiwen/minfer/issues/162) — **DONE 2026-09-26**

**Why.** The #147 audit enumerated the wider case: **104** `<<<>>>` sites in `src/cuda_kernels.cu`
(the 65 wrappers in the ticket plus multi-site families and two wrappers the #147 list had not
counted) enqueued a kernel and never read the launch's error. A launch that failed for real —
an illegal grid/block shape, an out-of-resources configuration, a stale context — latched the error,
and it surfaced at `CudaState::sync` or at the *next* hardened site's pre-launch check as a *latched
API error* (#145's honest label) with **no indication of which launch produced it**. `grep -c '<<<'`
on the tree is **122**; two of those are the `<<<>>>` in prose comments, so the audited surface is
**120 sites in 76 `launch_*` owners** (the wrappers plus the static `launch_gqa_attn_split_batched_kv` helper).

**What landed.**

- **Every site reads its own error, through the shared helper.** Each `<<<>>>` is preceded by
  `minfer_launch_prelude(site, kernel)` (which reports any *pre-existing* latch as not this launch's)
  and followed by a `minfer_launch_ok` / `minfer_launch_ok_opt` read that clears the latch it named.
  The site token starts with `launch:` and the report names the **kernel instantiation** and
  `cudaGetErrorName` — e.g. `kernel launch gemm_f16_nt_kernel_t<128,32,true> failed:
  cudaErrorInvalidValue (1) — the launch is refused (#162/launch:gemm_f16_a32)`.
- **The decision per launcher is severity in the helper, not a signature change.** `minfer_launch_ok`
  is **required**: it records a sticky failure that `CudaBackend::execute_node` — **one** Rust-side
  check (`CudaState::take_launch_failure`), not 104 signature changes and 104 Rust `Err` arms — drains
  on **both** arms and turns into an `Err` naming the site and the node, so the op never proceeds with
  a stale output (and the drain keeps a stale record from blaming the next node).
  `minfer_launch_ok_opt` names and clears **without** the sticky for a path with a documented
  fallback. Split of the 120 sites: **107 required → `Err`** (67 launchers — the matmul, elementwise,
  rope/store-KV, embedding, dequant/convert, MMVQ, attention and split-attention families, plus the
  MMQ terminal launchers `launch_mmq_raw_nt` / `launch_mmq_nt` and the #147 "no later gate" ones) and
  **13 documented fallbacks** (6 launchers — `launch_fa_prefill_f16kv`'s three window modes fall back
  to the legacy attention kernel; `launch_mmq_raw_nb_nt` / `_nb_bt_nt` (kernels + k-split reduce) /
  `_q6k_nt` (kernels + k-split reduce) / `_wide_nt` are #147's clean fast-path fallbacks;
  `launch_kv_move_rows` returns non-zero to its `Result` caller). The choice per family is a comment
  at each family in `cuda_kernels.cu`.
- **The injection lever is shared geometry, so a site's coverage is data.** `minfer_launch_block(site,
  dim3|unsigned)` replaces the block argument at every ordinary site; when `MINFER_TEST_CALL_FAIL`
  names the site the block becomes 4096 threads (over the 1024/block limit) and the launch itself
  returns `cudaErrorInvalidValue` for real, the kernel never runs — probed on GB10/sm_121. The
  dynamic-smem launchers keep #147's `minfer_launch_smem` lever. Adding a site's token is therefore a
  one-line string in the driver, not a bespoke mechanism.
- **A scripted audit, wired into CI.** `scripts/check_cuda_launch_returns.py` parses the source
  (comments and string literals blanked, so a commented-out occurrence is not a site) and requires,
  for **every** `<<<`: an enclosing `minfer_launch_prelude("<site>", …)` before it, an
  `minfer_launch_ok`/`_opt("<site>", …)` after its statement naming the **same** token, a token that
  starts with `launch:`, and a real-failure lever (`minfer_launch_block` or `minfer_launch_smem`) in
  the launch geometry. It prints its result and exits 1 on any offending line. It runs in the
  `check-docs` job (python3 present; the CUDA container's is not guaranteed) together with its own
  `--selftest` and a `--check-fixture` against
  `tests/fixtures/cuda_launch_sites.tsv`, the committed site list (line, owner, token, kernel
  fragment) the device gate compares its driven set against.
- **Two k-split reduce sites are separate tokens** (`launch:mmq_raw_nb_bt_ksplit`,
  `launch:mmq_raw_nb_bt_q6k_ksplit`). They share the launcher with the kernel site, whose `_opt`
  read returns 0 before the reduce; arming the kernel's token would leave the reduce site
  unreachable, so the reduce has its own prelude/read/token and the driver arms it alone.

**Acceptance results** (GB10/sm_121, CUDA 13.0, driver 580.178.04; serial device runs).

| check | before | after |
|---|---|---|
| `scripts/check_cuda_launch_returns.py` on `src/cuda_kernels.cu` | **104** unchecked sites | **empty list**: 120 / 120 sites read their own error and carry a lever |
| CUDA serial unit suite (`scripts/cuda_test.sh`) | **531 / 0 / 37** (master; the recorded 526 predated #173's +3 and #98's +2) | **536 / 0 / 37** (+5 gates) |
| `MINFER_TEST_ISSUE162=1` device gate: every audited site driven and named | — | **5 / 0** tests; the coverage test's driven set equals the 118-token fixture |
| `compute-sanitizer --tool memcheck` over the serial CUDA unit suite | **0** API errors | **0** errors over 536 |
| CUDA serial ignored, 0.5B config | 37 / 0 | **37 / 0** |
| CUDA serial ignored, Qwen3-0.6B Q8_0 | 37 / 0 | **37 / 0** |

The sanitizer row's command is
`compute-sanitizer --tool memcheck --target-processes all target/release/deps/minfer-<hash> --test-threads=1`
(wrapping the **test binary**, not `cargo` — the cargo/test build tree is not instrumented and
`--target-processes all` around the whole `bash scripts/cuda_test.sh` chain stalls the harness; the
binary form is the one that completes, in 350.17 s).

**Mutation checks (each reverted; `src/cuda_kernels.cu` restored to
`sha256 da2e00fb79442fcd03bd3618301b4014cd835f832bbafa4153b4af4283dcdbfb` byte-identically).** Full
transcripts in the closing comment on #162; the list:
(a) deleting the read at one single-site family (`launch_add_f32`) fails the audit **and** the device
coverage test (the armed site reports nothing); (b) the same at one `switch` case
(`launch:embed_rows__q4_k`) and (c) at one branch of a templated family
(`launch:gqa_attn_split_f16kv__hybrid_causal`); (d) making `minfer_launch_read` report but **admit**
the launch (return true) is caught by the severity test's `assert_eq!(rc, -1)` on the fa-prefill
fallback — a message-only assertion misses it; (e) the message naming a **wrong** instantiation is
caught by the fixture's fragment check; (f) the sticky removed from `minfer_launch_ok` is caught by
`cuda_issue162_required_sites_set_the_sticky_opt_sites_do_not` and by the node-level test; (g)
`execute_node`'s unconditional drain removed is caught by
`cuda_issue162_the_err_arm_also_drains_the_sticky` (an f16 matmul whose Rust wrapper returns `Err`
leaves the sticky pending, so a drain only on the `Ok` arm would blame the next node); (h)
`minfer_launch_block` made a no-op is caught because no armed site fails and every armed set observes
the empty set; (i) an `_opt` site that also sets the sticky is caught by the severity test's
`take_launch_failure().is_none()`; (j) the audit's lever check disabled is caught by the selftest's
"read but no lever" case; (k) the fixture's kernel fragment changed is caught by `--check-fixture`.
An **equivalent mutant** is recorded too: deleting the trailing `cudaGetLastError()` in
`minfer_launch_read` changes nothing (the read's own `cudaGetLastError` already resets the latch), and
the gate correctly stays green — the trailing call is belt-and-braces, not the clear.

**Honest scope.** Nothing was failing on sm_121 before or after: the sanitizer was already 0, so the
production paths remain **latent** and the evidence is that every site *can* be shown to refuse and
name a **real** failing launch. The injection is a test-only knob (`MINFER_TEST_CALL_FAIL` +
`MINFER_TEST_ISSUE162=1`, unset in every default, bench and sanitizer run) that drives the site's own
geometry illegal; it does not exercise a genuine driver fault. The **source audit is static**: it
proves the read and the lever are *written*, not that they run — a `<<<` inside a string literal or a
macro the parser cannot resolve is reported, never silently accepted, but the parser resolves a site
variable only through a plain `=` assignment in the enclosing function (the `launch_site` ternary of
`launch_gemm_f16` resolves to its first arm, which is why the driver arms `gemm_f16_a32`). The device
gate's coverage assertion compares **sets** of site tokens; where two source sites share one token
(the two `mmq_raw_nb_bt` kernel instantiations) the gate proves the token is reached, not that both
branches were — the audit, not the gate, is what guarantees each source site has its own read. The
`compute-sanitizer` and real-model rows are recorded measurements on dgxspark (CI has no GPU); the
x86_64 CPU row is unaffected because no pure-Rust test was added.

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
  the CPU path is what dgxspark measured; the CUDA half of C5's own gate was run at S1).

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
  resolve through the run table — and fill `cells`. (History: E1's test-only
  `fill_attn_inputs` and the `kv_note_used` → `own_prefix` C1 remnant were deleted
  in [#228](https://github.com/yusiwen/minfer/issues/228); `fill_batch_inputs` is
  the one production fill entry point.)
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
  rope-only fixture must not need a KV arena). *(History: `fill_attn_inputs` was
  deleted in [#228](https://github.com/yusiwen/minfer/issues/228), which moved its
  corpus onto `fill_batch_inputs`; the rope-only branch survives as the
  `#[cfg(test)]`-scoped `GraphAllocator::fill_attn_inputs_without_cells`.)*
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

*(Forward note, 2026-09-29: `kv_own_range` in the sentence above is
`GraphAllocator::kv_own_range`, which
[#232](https://github.com/yusiwen/minfer/issues/232) deleted as dead surface — it was a
one-line forwarder to `KvCache::own_range`, the function production actually calls (through
`fill_batch_inputs` → `own_positions`, and through `kv_copy_prefix`). The sentence records
what this step landed with; it is not a live API.)*

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
store resolver (`kv_cells_for_seq`) now refuses a position inside the share outright. (As of
[#228](https://github.com/yusiwen/minfer/issues/228) there is **one** entry point — the production
`fill_batch_inputs` — and the resolver's refusal is a belt-and-braces check it cannot reach with a
shared position; the `fill_attn_inputs` half of that sentence was true when S3 landed.) A sharing
sequence's run holds positions `[shared.rows, shared.rows + cap)`, so a store at `t < shared.rows` gives
the share up from `t` on: `shared.rows` drops to `t` and the rows the sequence already wrote shift **up**
by `d = old_base - t` inside the same run. That arithmetic is what keeps the change small — the span list
stays at **two entries** (the remaining share plus the run), the owner table mirrors the move with one
`copy_within`, and `written` is unchanged: the sequence's readable positions are still `[0, written)`,
only `private_written` grows by `d`. `KvArenaStats` gains `cows`/`cow_cells` so a gate can *see* the
mechanism run, and `GraphAllocator::kv_cell_of` is the read-side twin of the store resolver (a caller
snapshotting a sharing sequence's rows cannot use the store one, which refuses those positions by
design).

*(Forward note, 2026-09-30: `kv_cell_of` is **test-only** as of
[#236](https://github.com/yusiwen/minfer/issues/236) — `#[cfg(test)] pub(crate)`, not
`allow(dead_code)`, because it never had a non-test caller. `git log -S 'kv_cell_of' --all` over
`src/` names only `d43e716` (which introduced the forwarder **with** this doc), `c9bbcbd` (the test
call) and `54f6de0` (the test-module extraction), so no production call site was ever deleted. The
"caller snapshotting a sharing sequence's rows" the sentence above names is served two other ways in
production: a sharer's rows are read as **windows** (`KvCache::attn_map` → the `kv_map` input, C8b
S2/S4) and a whole-run snapshot goes through the C5 container (`kv_save`/`kv_save_with_host`, which
stores the whole arena). The one consumer is the test-side observation instrument `kv_rows_of`
(`server::batch::tests`), driving the C8b S3 gate
`a_store_inside_a_shared_prefix_takes_a_private_row`; see the #236 record in
§test-infrastructure. The sentence records what S3 landed with; it is not a live API.)*

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
copy-on-write through the fill entry point (`fill_batch_inputs` since
[#228](https://github.com/yusiwen/minfer/issues/228); `fill_attn_inputs` when this gate landed). The
real-model gate
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
`cfg(target_os = "macos")`, so CI's `build-macos` is the only compile dgxspark cannot
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
- **Honest scope**: dgxspark's device has ~128 GB, so "a model larger than device memory" cannot
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
is rejected as a cycle, and `alloc_graph` calls it on every build; **fixed 2026-09-26** (record
below);
[#99](https://github.com/yusiwen/minfer/issues/99) — the process-wide KV format above.

**What S2 does not do** (it stays on [#55](https://github.com/yusiwen/minfer/issues/55)): the
**reserve/assign re-map** (a reserved region a rebuild re-maps without touching the device — the
literal "split reservation from assignment", which Metal's G6 adopts) and the **multi-graph cache**
(§14 row 3, which is what removes E3's per-chunk rebuild). Cross-boundary staging is charged to
`pool_bytes` but is still allocated at its exact length, and backend-internal scratch (Metal capture
staging, CUDA `positions` scratch) is outside the report.

#### Graph record (#98, 2026-09-26) — a repeated source is one edge, not two

**The defect.** `ComputeGraph::topo_order` counted in-degree **per source entry**
(`for &s in &node.src { indeg[node.id] += 1 }`) but released it **once per node**
(`if self.nodes[v].src.contains(&u) { indeg[v] -= 1 }`). A consumer that lists the same predecessor
twice therefore never reached in-degree 0 and the validator reported a false cycle — on the minimal
`add(x, x)` graph, `cycle detected: 1/2 nodes ordered` — even though the DAG was legal.

**Reachability.** `GraphBuilder::add`/`mul` pass `&[a, b]` straight through with no dedup, so
`b.add(x, x)` (`2 * x` written as an addition) is buildable; `GraphAllocator::alloc_graph` validates
with `topo_order()?` on every build, so the graph could not be allocated at all (`BackendScheduler`
only had a `debug_assert!`, so allocation was the hard failure). Found while writing E4 S2's
input-buffer gate, which worked around it with two distinct inputs.

**The fix.** The in-degree pass now counts each *distinct* predecessor once
(`!node.src[..j].contains(&s)`), so both passes share one notion of "u is a predecessor of v". A
duplicate source is one edge read twice, not two edges. The release pass is unchanged, and duplicate
sources stay **legal** by design — the issue's intent is that `add(x, x)` works, not that it is
rejected.

**The tests.** `topo_order_accepts_a_repeated_source` (the order covers every node, source first)
and `a_repeated_source_allocates_and_executes_as_two_reads` (end-to-end through `alloc_graph` +
`BackendScheduler::execute`, asserting the **value**: `add(x, x)` = `2x` and `mul(x, x)` = `x²` on
concrete inputs, so a graph that dropped the second read fails instead of passing an `is_ok()`). The
control arm, `topo_order_detects_cycle`, now asserts the exact message
`cycle detected: 0/2 nodes ordered`: a fix that simply stopped detecting cycles fails it. The E4 S2
gate `an_input_never_takes_a_buffer_the_walk_released` keeps its two-input form on purpose (its
property is input placement, not the duplicate-source path) and carries a comment naming #98 as the
reason it was written that way.

**Mutation evidence (rule 3).** Reverting the in-degree pass to the per-entry form (dropping the
`!node.src[..j].contains(&s)` guard) and running both new tests:

```
$ cargo test --release repeated_source
test graph::alloc::tests::a_repeated_source_allocates_and_executes_as_two_reads ... FAILED
test graph::tests::topo_order_accepts_a_repeated_source ... FAILED
thread '...a_repeated_source...' panicked at src/graph/alloc.rs:4123:37:
add(x, x) must allocate, got: cycle detected: 1/2 nodes ordered
thread '...topo_order_accepts_a_repeated_source' panicked at src/graph/mod.rs:337:36:
add(x, x) is acyclic: "cycle detected: 1/2 nodes ordered"
test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 493 filtered out
exit=101
```

Restored byte-for-byte (`diff -q` clean).

**Adjacent audit (the `contains`-vs-occurrence asymmetry).** The only other `src.contains(` in
`src/graph/` is `GraphAllocator`'s "is this input consumed by a KV-indexing node" existence test —
a duplicate does not change existence. The liveness `last_use` pass takes a `max` over source
entries, so a duplicate is idempotent. `n_consumers` **does** count per entry, but over-counting
only makes the in-place rule's `== 1` test stricter: it refuses an alias and keeps a private buffer,
which is conservative and never a wrong read (a comment now says so next to the count).
`extend_through_views` / `extend_buffer_alive` walk the single `view.src` chain and `max` a deadline,
so no source list is involved. The fusion pass reads `mul.src[0]` / `src[1]`; a duplicate is
arithmetically preserved because `SwiGLU(gate, up) = silu(gate) * up` is exactly the
`Mul(Silu(gate), up)` it replaces. The scheduler pushes one input buffer per source entry (so
`add(x, x)` really reads the buffer twice) and dedups cross-split `inputs` by `contains` (existence
again). `cache.rs`'s reuse identity compares `src` vectors element-wise, so duplicates are
deterministic. No other defect found; no follow-up issue needed.

**Counts (rule 4).** `cargo test --release` on dgxspark (aarch64), 2026-09-26: unit **462 passed / 0
failed / 33 ignored** (was 460; +2 for the two new tests), integration **10 / 0 / 6**. The
`x86_64 (CI runner)` row moves by the same +2 (458 → 460); `test-linux-cpu`'s `--check-live`
confirms it against its own log, and `AGENTS.md` and `docs/status.toml` carry both rows.

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
`DeviceLayoutGuard` that restores it on a panic. **Both of those device-only parts are gone as of
[#153](https://github.com/yusiwen/minfer/issues/153)** (the guard deleted, the tag per engine — see
the #153 record below); the sentences above describe the state at #99's landing.

**The CUDA device half was deliberately not done at #99 and landed as
[#153](https://github.com/yusiwen/minfer/issues/153).** `cuda.rs` held the layout in a process-wide
`KV_LAYOUT` that the launchers read directly (not through `CudaBackend::kv_layout`), so a per-engine
device path needed the launchers and the captured-graph key threaded; a per-instance field alone
would silently still have read the global. The #153 record below states what is now per-engine and
what is not.

**The entry point.** `scripts/real_model_gates.sh` is the one command for the set: it defaults to
`--test-threads=1` (required on a device) and takes `PARALLEL=1` (CPU-only parallel) and
`FEATURES=cuda`. Documented in `AGENTS.md` rule 11 + the real-model-gates bullet and in
`docs/BUILD.md` §Tests.

**Measured** (CPU build, dgxspark; `--bin minfer` for the gate set).

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
  [#64](https://github.com/yusiwen/minfer/issues/64)); `cargo test --release --features cuda --
  --ignored` without `--test-threads=1` remains the wrong command, and `scripts/real_model_gates.sh`
  keeps it serial. Measured serially on dgxspark (GB10 sm_121, CUDA 13.0): the unit suite is **501
  passed / 0 failed / 32 ignored**, and the ignored set is **32 passed / 0 failed** in both the 0.5B
  (f32 KV) and the Qwen3-0.6B (f16 KV) configurations — unchanged by this increment except that one
  obsolete unit test is gone. (**#153 removed the second reason** — the device KV layout tag is per
  engine now — and re-measured the parallel form; see the #153 record below for the fresh counts.)
- An **explicit `f16`** cache type still lets the *device* layout follow
  `set_kv_cache_type`'s auto policy (the pre-C4 split the loader comment records); the builder's
  f32/f16 region shapes are identical, so that is not a sizing hazard. A packed (`q8_0`) resolution
  is still restated explicitly on the device, as before. (**#153 folded the auto policy into the
  engine's resolved format and deleted `set_kv_cache_type`; the loader no longer restates anything.**)
- The `forward_graph` (non-cached) path uses the process-global `graph_cache()`; two *models* with
  different formats driving that one cache would still be refused by `ensure_kv`. That is the
  pre-existing single-cache-per-process design, not the format global, and the server/CLI paths
  that matter pass a `GraphCache` explicitly.

**Follow-ups.** [#154](https://github.com/yusiwen/minfer/issues/154) (the load-sensitive batching
timing gate); [#153](https://github.com/yusiwen/minfer/issues/153) (the CUDA per-graph layout) was
its own follow-up and landed — its record follows.

#### #153 record — the CUDA KV layout is per engine, and the captured-graph identity carries it

**What was wrong.** #99 made the KV *format* per engine for the model, the graph builder, the
allocator and the CPU kernels, but left the CUDA side on a process-wide `static KV_LAYOUT: AtomicI32`
in `cuda.rs`. `CudaBackend` snapshotted it into `kv_layout` at construction, yet the value came from
the static (and `entry()`'s `kv_format` hook read the static directly), so a per-instance value would
have been a half-wire: two engines loaded with different formats would still have run the
last-loaded layout, and the loader had to *restate* the tag for `q8_0`
(`models::load_model_configured`). The honest-scope sentence in `AGENTS.md` rule 11 said exactly
this and is the caveat this ticket retires.

**The one authority is now the engine's resolved format.** `KvFormat::resolve` takes the model dims
and folds in the GPU's own auto policy (`auto_device_format`: f16 for the 7B class, f32 for small
models; the CPU stays f32) — it used to live only in `cuda::set_kv_cache_type`, which is deleted
along with `kv_cache_layout` / `kv_cache_is_f16` / `set_kv_cache_layout` / `set_kv_cache_f16`.
`models::load_model_configured` stamps the one answer on the engine, and:

- `CParams::kv_format` carries it into the builder (region width) and the reuse identity, as before;
- `GraphAllocator::set_kv_format(format)` stamps the CPU backend **and**, if it exists, the CUDA
  backend's tag (`cuda::layout_of(format)` → `KV_LAYOUT_F32/F16/Q8_0`);
- `GraphAllocator::enable_cuda` builds a fresh `CudaBackend::with_layout(...)` from that stamp, so a
  backend created after the stamp cannot revert to a default;
- the registry's `kv_format` hook answers from `a.cuda().kv_format()`, so a KV session's header
  element type is the engine's, and `server::batch`'s row-width snapshot reads
  `GraphAllocator::kv_format` instead of the process static.

The kernels already took the tag as a launcher argument (`gqa_attn_split`, the prefill/verify
attention, `store_kv_f32/f16/q8_0`, `attn_bias_rope_store`); what was missing was that the value came
from a global. `cuda::layout_of` / `format_of` are the only bindings between `KvFormat` and the FFI
codes, and the layout tag is exhaustive over the enum.

**The captured-graph identity carries the layout.** `CapturedGraph` gained a `kv_layout` field, and
`graph_replay_step`'s lookup refuses an exec whose recorded tag no longer matches the backend's —
it is destroyed and the 3-run warmup restarts, exactly like a `pool_gen` change. `set_kv_layout`
invalidates eagerly on a change as the first line of defence. The enforcement point is
`graph/cuda_backend.rs::graph_replay_step` (the `pool_gen` / `kv_layout` comparison), with the
identity recorded in `close_capture_or_sync`.

**The device gate.** `models::qwen2::graph::tests::two_cuda_engines_with_different_kv_layouts_run_interleaved`
loads an `f32` engine and a `q8_0` engine **before either runs**, prefills and then decodes them
interleaved on two live `GraphCache`s, and asserts four things: each cache's `CudaBackend::kv_layout`
is the layout its engine named (and the two differ); the packed regions are ≥ 3x smaller; the packed
engine's logits stay in the C4 class of the f32 engine's (at the argmax ≤ 1.0, tail ≤ 4.0); and each
engine's interleaved logits are **bitwise** its own solo logits (isolation). Interleaved, not
threaded: `CudaState` is a process-wide singleton and the capture path holds a process-wide stream
lock, so two OS threads would serialize on that lock anyway — the form the ticket allows. A unit gate
(`cuda_graph_recaptures_on_kv_layout_change`) pins the capture identity without the model, and
`alloc::tests::set_kv_format_stamps_the_cuda_layout_per_engine` pins the stamp → backend path.

**Measured** (GB10 sm_121, CUDA 13.0, 2026-09-26).

- `scripts/cuda_test.sh`: **539 passed / 0 failed / 38 ignored** (was 536 / 0 / 37; #153 adds two
  device unit gates and one pure `kvformat` gate, and moves the two `cuda::kv_dtype_tests` to the
  `KvFormat` ↔ tag mapping they now assert).
- `FEATURES=cuda scripts/real_model_gates.sh`: **38 passed / 0 failed** in both the 0.5B (f32 KV) and
  the Qwen3-0.6B (f16 KV, `hd` 128) configurations (was 37 / 0; the new two-engine gate is
  `#[ignore]`d).
- The two-engine gate alone prints
  `[153] two live CUDA engines, interleaved 8 decode steps: tags f32=0 q8_0=2; regions f32 6291456 B
  vs q8_0 1671168 B (3.76x smaller); interleaved packed-vs-f32 max |Δlogit| = 2.479504 of a 37.821205
  spread, at the argmax 0.59605026; interleaved-vs-solo drift 0 / 0`.
- Full `cargo test --release --features cuda -- --ignored` (**integration targets included**, which is
  what the ticket's acceptance line names): serial `--test-threads=1` is **green** — the 38-test bin
  set plus the 6-test `conversation_cli` set, 0 failed. Parallel (default harness) is **still red**,
  with fresh counts and a fresh reason (below).
- `compute-sanitizer --tool memcheck --target-processes all` over the CUDA unit suite:
  **0 API errors** (539 / 0 / 38).

**The parallel `--ignored` answer (restated, not a shrug).** The parallel run remains the wrong
command, and the reason is now measured rather than inherited: it is **not** the KV layout any more.
Two runs of the same command:

- run A: **33 passed / 5 failed**, failures `cuda_map_window_costs_no_more_than_the_span_it_replaces`
  (a load-sensitive timing gate — its own per-round ratios ranged 0.63–2.20 with median 1.398x) and
  `server_batch_matches_serial_and_is_faster` (the known [#154] timing gate), plus three
  process-global-state failures: `conversation_real_model_smoke` with
  `cudaErrorStreamCaptureInvalidated (901)` inside a capture window,
  `async_cross_copies_never_block_and_stay_bitwise_identical` comparing the **process-wide**
  `cuda::stream_sync_count()` (4160 async vs 728 sync across concurrent tests), and
  `an_auto_offload_plan_fits_the_budget` mutating the **process-wide** `MINFER_GPU_MEM` env;
- run B: **SIGSEGV (signal 11)** after two unrelated failures — the failure set is not stable.

Every mechanism is the process-wide `CudaState` singleton (one stream, one capture window, one
MMQ/pool state — issue [#64](https://github.com/yusiwen/minfer/issues/64)) and the process-wide
counters/env a few gates read, not the KV format: the layout is per engine now, and no failure names
a KV region or a layout mismatch. `scripts/cuda_test.sh` and
`FEATURES=cuda scripts/real_model_gates.sh` keep the device set serial.

[#154]: https://github.com/yusiwen/minfer/issues/154

**Honest scope.** The CUDA tag is per engine; **Metal's `kv_cache_is_f16` is still process-wide** (its
kernels read it, there is no Mac here to change it on — G5 for packed). The gate interleaves two
engines in one process, which is the ticket's allowed form; it does not prove two *threads* can drive
two CUDA engines concurrently, and the `compute-sanitizer` run is still the whole CUDA unit suite
serially.

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
(`kv_reserve_seq`/`kv_release_seq`/`kv_seq_slot`/`kv_own_range`). *(Since
[#228](https://github.com/yusiwen/minfer/issues/228) the classic case is
`fill_batch_inputs`'s own `reserve_seq(seq, n_ctx)` when a group has no run, and
`own_prefix` is test-only. Since [#232](https://github.com/yusiwen/minfer/issues/232),
2026-09-29: `kv_own_range` was deleted from that surface — a one-line forwarder with no
production caller; `KvCache::own_range` is the production-used function, and the `alloc`
tests drive it directly.)*

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

So `--n-slots 4` cannot "materially exceed" the serial baseline on dgxspark: for
identical prompts the serial path is ~3x cheaper on prefills that batching cannot
recover, and for distinct prompts the two tie (neutral prefill batching + neutral
decode batching). The remaining route to the acceptance on CPU is a `nt > 1`
decode kernel that actually exploits the shared weight read (the F1 family); on a
bandwidth-bound device batching is the standard win, and the trace above is what a
GPU re-measurement should compare.

Still to come in E2: nothing is left to *build* for the deliverable; what remains
is the acceptance, which dgxspark cannot demonstrate (the step-4 trace above shows
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
`0..pos[t] + 1`. *(E1's `fill_attn_inputs` was never called by production: it was
deleted in [#228](https://github.com/yusiwen/minfer/issues/228) and its whole test
corpus now drives `fill_batch_inputs`, the call `forward_cached`/`forward_batch`
themselves make.)*

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
covers, so the port is not mechanical; and dgxspark had **no device at the time**
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

Reproduced on dgxspark (CPU, cached 0.5B Q4_0, `MINFER_BATCH=1 --n-slots 1`, two
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

Reproduced on dgxspark (CPU, no model — a test double whose `forward_batch` panics, driven
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
of the load, not of the code. Measured at `e1ac17f` on dgxspark (CPU build, 0.5B q4_0, four
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

**Verification (2026-09-25, CPU build, dgxspark; `--bin minfer` for the gate set).**

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
(dgxspark, 2026-09-25): the warm request (8-token prompt, `max_tokens = 4`) took **4 steps** against a
budget of **80** (20x); the long one (120-token prompt, `max_tokens = 64`) took **64** against **768**
(12x). The gate prints both counts on every run.

**Verdict, before and after (CPU build, dgxspark, 16 extra CPU spinners on a 20-core machine).**

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

**Verification (2026-09-25, dgxspark; `--bin minfer` for the gate set).**

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

#### Test-infrastructure record (#160, 2026-09-27) — the remaining server-gate steppers bound work, not wall-clock seconds

**The defect.** [#158](https://github.com/yusiwen/minfer/issues/158)'s audit left two shapes standing. The one this ticket was filed for: five
`#[ignore]`d server gates drove the engine with `while engine.busy() { engine.tick(…) }` and
**neither a progress assertion nor a step budget**, so a wedge inside `BatchEngine::tick` hung the
suite forever instead of failing it (#158's own F8 metrics gate kept its work bound). The other
shape: three absolute wall-clock bounds whose *verdict* does not depend on the clock. Both are
review findings, not reproduced failures — the set was green before and after; the evidence below is
**injected** wedges, not an observed hang.

**What now bounds each site.** Every stepper goes through one shared `WorkBound`
(`src/server/batch.rs`), the single implementation of the invariant #158 introduced: a `tick` that
leaves the engine busy must have advanced `BatchEngine::work_units` (if no forward ran, some slot's
`advance` returned `Continue`, and `Continue` commits exactly one token), and the step count must
stay inside `step_budget(prompt, max_tokens)`. `drive_by_work` is now a thin wrapper over it.

| site (gate) | before | now |
|---|---|---|
| `run_batched` (`server_batch_matches_serial_and_is_faster`, `a_long_request_may_use_the_whole_arena`) | `while !queue.is_empty() \|\| engine.busy()` | `WorkBound` with `step_budget(Σ prompts, n · cap)` |
| `run_serial` (same two gates) | `while done.is_none()` | one `WorkBound` per request |
| `serve_on` (`a_prefix_copied_from_another_slot_answers_identically`, `a_store_inside_a_shared_prefix_takes_a_private_row`) | `while engine.busy()` | `WorkBound` |
| `a_slot_snapshot_resumes_the_context_without_re_prefilling` | three `while X.busy()` loops | three `WorkBound`s (cold / resumed / delta), each with its own budget |
| `a_chunked_prefill_answers_like_an_unchunked_one` | `while engine.busy()` in both arms | `WorkBound`, the arm's own name in the message |
| `a_repeated_chunked_prefill_stops_rebuilding` | `while engine.busy()` in both calls | `WorkBound` |
| `a_long_prefill_keeps_another_slot_decoding` | `for _ in 0..3 { tick }` (already finite) | the same three steps through `WorkBound`, so a wedge fails on the step that wedged it |
| `published_metrics_move_as_requests_are_served` (#158) | `drive_by_work` | unchanged behaviour, now on the shared `WorkBound` |

`step_cap` folds an unbounded request (`max_tokens < 0`, which ends on its context bound) into the
budget. No wall-clock number was added anywhere as a failure signal, and no bar was widened.

**The wall-clock bounds that remain are named backstops, not verdicts.**

- `serve_loop_publishes_the_queue_and_running_depth`'s feeder terminator is now
  `FEEDER_POLL_BACKSTOP` (120s) with its reasoning at the call site: the verdict is the downstream
  `peak_running > 0` / queue arithmetic, and the worker runs on the test's own thread, so a real
  wedge hangs that join regardless — which is exactly why #158 left it and #160 classifies it a
  redundant backstop. `peak_running` is observed within milliseconds, so it is not load-sensitive.
- `tests/conversation_cli.rs`'s `run_cli` child-process ceiling (1800s for the `#[ignore]`d
  real-model sessions, 30s for the no-model cases) and `tests/backend_registry_cli.rs`'s 60s are
  cross-process hang guards — a child exposes no in-process progress counter. Both are now
  **env-overridable** through `MINFER_CLI_WATCHDOG_SECS` (unset/unparsable/zero keeps the caller's
  number), with the reasoning recorded at each `run_cli`.
- `src/server/mod.rs`'s `elapsed < Duration::from_secs(10)` is untouched: a CI unit test's 200x
  ceiling over a 50ms bounded drain, not a real-model gate.

**The seam (rule 3).** `MINFER_TEST_TICK` (`src/server/batch.rs::tick_seam`, read once per process
through a `OnceLock`, `Off` when unset or on any unrecognised value) injects the two arms of a
bounded drive into the **real** `tick`: `wedge` returns before forwarding or committing anything (the
work counter freezes), `spin` advances `work_units` without ever completing a run (the counter keeps
moving along a path that cannot terminate). It is an **environment switch** — the mutation runs need
no source revert, so `git diff` stayed clean throughout.

**Mutation evidence — the wedge arm (rule 3).** Command shape (CPU build, aarch64, dgxspark,
2026-09-27): `MINFER_TEST_TICK=wedge cargo test --release --bin minfer -- --ignored --exact <gate>
--nocapture`. Every hardened gate fails on **step 1** with
*"the engine is wedged — step 1 left it busy without advancing the work counter (still 0)"*
(`src/server/batch.rs:3756`):

| gate | wall clock | what the message names |
|---|---|---|
| `published_metrics_move_as_requests_are_served` (#158 precedent) | **0.30s** | the warm request |
| `a_prefix_copied_from_another_slot_answers_identically` | **0.21s** | the slot-scoped request |
| `a_store_inside_a_shared_prefix_takes_a_private_row` | **0.29s** | the slot-scoped request |
| `a_long_request_may_use_the_whole_arena` | **1.68s** | the batched run |
| `server_batch_matches_serial_and_is_faster` | **0.30s** | the batched run |
| `a_slot_snapshot_resumes_the_context_without_re_prefilling` | **0.22s** | the cold run |
| `a_chunked_prefill_answers_like_an_unchunked_one` | **0.69s** | the unchunked prefill |
| `a_repeated_chunked_prefill_stops_rebuilding` | **0.71s** | the repeated chunked prefill |
| `a_long_prefill_keeps_another_slot_decoding` | **0.45s** | the short run |

The listed time is the whole process (start + 0.5B load + the failure); libtest's own per-test time is
0.15–1.61s. Every run's `test result:` line is `FAILED. 0 passed; 1 failed` — the assertion, not a
timeout, ends it.

**Mutation evidence — the step-budget arm.** `MINFER_TEST_TICK=spin` over the whole `#[ignore]`d set
with the two `serve_loop` gates skipped is **10 failed / 21 passed in 12.45s** (CPU build, dgxspark,
2026-09-27; `cargo test --release --bin minfer -- --ignored --nocapture --skip
serve_loop_publishes_the_queue_and_running_depth --skip
a_job_rejected_for_want_of_a_slot_is_answered_with_503`). Each bounded stepper stops exactly one step
past its budget, and the budget arm is the one that fires:

| gate | the budget arm |
|---|---|
| `published_metrics_move_as_requests_are_served` | 81 steps exceeded the 80-step budget |
| `a_prefix_copied_from_another_slot_answers_identically` | 85 > 84 |
| `a_store_inside_a_shared_prefix_takes_a_private_row` | 133 > 132 |
| `a_long_request_may_use_the_whole_arena` | 1273 > 1272 |
| `server_batch_matches_serial_and_is_faster` | 369 > 368 |
| `a_slot_snapshot_resumes_the_context_without_re_prefilling` | 69 > 68 |
| `a_chunked_prefill_answers_like_an_unchunked_one` | 457 > 456 |
| `a_repeated_chunked_prefill_stops_rebuilding` | 441 > 440 |
| `a_long_prefill_keeps_another_slot_decoding` | **not the budget**: the work counter does move under `spin`, and its priming loop is a fixed three steps, so the gate's own *"slot 1 must have emitted something before the long prefill"* assertion fires instead |

The tenth failure is `the_seam_fails_the_batch_forward_without_a_bespoke_mock`: the #171 gate asserts a
`tick` reaches the forward and reports the injected `500`, and under any `MINFER_TEST_TICK` injection
`tick` returns before the forward, so it fails on its own `.expect_err`. It is reported for
completeness, not a bounded stepper.

**The bound that could not be made to fail fast — honest scope.** Two `#[ignore]`d gates step the
engine through the **production** `serve_loop` on the test's own thread:
`serve_loop_publishes_the_queue_and_running_depth` and
`a_job_rejected_for_want_of_a_slot_is_answered_with_503`. A persistent wedge keeps the engine busy,
so `serve_loop` keeps ticking and the test thread never returns — there is nowhere for a test
assertion to run. Measured with `MINFER_TEST_TICK=wedge … --exact` under an outer `timeout 30`: the
process is killed at **exit 124** (it hangs), which is the shape #158's audit already recorded and
#160's table classifies as a redundant backstop. Making those two wedge-proof needs a *production*
liveness bound in `serve_loop` (a counted consecutive-no-progress limit) or the engine's work counter
published to `ServerMetrics` so a spawned worker can be observed; either is a production change this
ticket's scope fence excludes. Filed as [#196](https://github.com/yusiwen/minfer/issues/196).

**Verification (2026-09-27, CPU build, dgxspark (aarch64); `--bin minfer` for the gate set).**

| Command | Result |
|---|---|
| `cargo test --release` | **465 passed / 0 failed / 33 ignored** unit + **10 / 0 / 6** integration (unchanged — no test added or removed) |
| `scripts/real_model_gates.sh` (parallel, 0.5B) | **33 / 0** in 35.08s (42.9s wall) |
| `PARALLEL=0 scripts/real_model_gates.sh` (serial, 0.5B) | **33 / 0** in 36.67s (36.8s wall) |
| `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf PARALLEL=0 scripts/real_model_gates.sh` | **33 / 0** in 46.39s (46.6s wall) |
| `MINFER_TEST_TICK=wedge` per gate (9 gates) | each **FAILED** on step 1, 0.21–1.68s wall (table above) |
| `MINFER_TEST_TICK=spin` (set, minus the two `serve_loop` gates) | **10 failed / 21 passed** in 12.45s (table above) |
| `MINFER_TEST_TICK=wedge` on the two `serve_loop` gates, outer `timeout 30` | **exit 124** — hangs, recorded as the limit above ([#196](https://github.com/yusiwen/minfer/issues/196)) |
| `rustup run stable rustfmt --edition 2021 --check` on the changed `.rs` | clean (rustfmt 1.9.0-stable; the pinned 1.97.1 toolchain has no rustfmt component) |

**Docs.** `docs/GATE-CONTRACT.md` rule 4's closing *"[#160](https://github.com/yusiwen/minfer/issues/160) tracks the remaining unbounded steppers"*
is replaced with the outcome (the steppers are bounded; the remaining watchdogs are named backstops;
the seam is `MINFER_TEST_TICK`), and §3 documents the seam next to `MINFER_TEST_CALL_FAIL`.
`docs/BUILD.md` § *Tests* records the same and the mutation lever. `ARCHITECTURE-ROADMAP.md` is
untouched: this is robustness inside the existing test-infrastructure/batching rows, not a new
capability — the same reason the #154, #151 and #158 records give.

#### Test-infrastructure record (#196, 2026-09-27) — `serve_loop` carries a counted no-progress bound, so a wedge answers its clients and ends

**The defect.** [#160](https://github.com/yusiwen/minfer/issues/160) made every
`while engine.busy()` stepper in the `#[ignore]`d server gates fail fast under an injected wedge —
except two, for a structural reason:

- `server::batch::tests::serve_loop_publishes_the_queue_and_running_depth`
- `server::batch::tests::a_job_rejected_for_want_of_a_slot_is_answered_with_503`

Both drive the engine through the **production** `serve_loop` on the test's own thread. A persistent
wedge in `BatchEngine::tick` leaves the engine busy forever, `serve_loop` keeps ticking, and the test
thread never returns — there is nowhere for an assertion to run. #160 measured
`MINFER_TEST_TICK=wedge … --exact <gate>` under an outer `timeout 30` as **exit 124** (a hang); the
feeder's `FEEDER_POLL_BACKSTOP` cannot rescue it, because the worker is on the same thread. In
production the same shape is [#151](https://github.com/yusiwen/minfer/issues/151)'s client-visible
failure one level up: the loop spins at 100% CPU, no client is ever told, `in_flight` stays ≥ 1, and
a graceful drain burns its whole `MINFER_DRAIN_MS` deadline.

**What landed — a production no-progress guard in `serve_loop`.** `src/server/batch.rs`:

- `STALL_STEP_LIMIT: u64 = 64` — the number of **consecutive** steps that may leave the engine busy
  without advancing `BatchEngine::work_units`. The healthy maximum is **0**: a `tick` that leaves the
  engine busy has either forwarded a decode row or committed a token through `advance`'s `Continue`,
  and both increment the counter (#158's invariant; the gates' `WorkBound` asserts it per step, and
  this is the same invariant enforced in production). 64 is deliberately loose rather than tuned: a
  future engine change that legitimately defers work for a handful of steps is not mistaken for a
  stall, and 64 no-op steps cost well under a millisecond, so a real stall still ends immediately
  against the previous "spins forever". It is a **count** (steps), never a wall-clock number
  (rule 4).
- On the 64th consecutive no-progress step the loop: counts `minfer_worker_stalled_total`; answers
  **every live run exactly once** with `ApiError::server("the worker stalled")` (500 `server_error`)
  through the existing `fail` machinery — new `BatchEngine::fail_all`, the `fail_batch` shape without
  a row list, since a wedged step may have built no batch; answers **every queued-but-unadmitted
  job** (the worker's `pending` deque plus the channel) with the same terminal error through #121's
  `reject`, so its sender never drops into the silent empty `200` — new `reject_queued`; zeroes
  `worker_pending`, publishes metrics, logs, and **breaks** the loop. The queued jobs leave the
  queue, so they are counted in `jobs_admitted_total` **and** `jobs_dropped_total`, keeping
  `accepted - admitted` a true queue depth.
- The count is **reset** wherever the loop legitimately moves: on `blocking_recv` (an idle wait is
  not a spin), on any step that advanced `work_units`, and on #151's `Err` step (it answered its
  batch and released its slots; `work_units` does not move there because the increment sits after
  the successful forward). The last reset is deliberate: without it, a saturated server whose every
  forward fails deterministically — a state #151 already handles correctly, one 500 per batch —
  would accumulate no-progress steps and be declared stalled, turning a wrong-but-answering server
  into a stopped one.
- Ending the loop drops `job_rx`, so a later request gets the existing
  `503 unavailable_error("server shutting down")` (`requests_rejected_total`), not a hang.

The condition is observable next to the other `AtomicU64`s: `ServerMetrics::worker_stalled_total`
and the `minfer_worker_stalled_total` family in the `/metrics` rendering. It is the guard's own
signal — a count of stall events — and deliberately not [#157](https://github.com/yusiwen/minfer/issues/157)'s
general terminal-error accounting.

**Gates.** Both `#[ignore]`d gates now assert `worker_stalled_total == 0`, the property the mutation
breaks. `serve_loop_publishes_the_queue_and_running_depth` also prints the terminal error each
client received, and its feeder leaves as soon as the counter moves, so the assertion runs at once
instead of waiting out `FEEDER_POLL_BACKSTOP` (before that the mutated gate failed correctly but
only after 120.19s). `a_job_rejected_for_want_of_a_slot_is_answered_with_503` checks the counter
after reading the channels, so a wedge surfaces as the error the client actually got. Two plain
`#[test]`s cover the answer machinery in CI, with no model:
`the_stall_answers_every_live_run_exactly_once` (three live runs and an idle slot: one 500 each,
slot freed, prefix cleared, idle slot untouched) and
`the_stall_answers_every_queued_job_exactly_once` (one job in the deque, two in the channel: one 500
each, senders closed).

**Mutation evidence (rule 3).** `MINFER_TEST_TICK=wedge` is an environment switch, so no source
revert is involved and `git diff` stayed clean throughout. CPU build, dgxspark (aarch64),
2026-09-27, `MINFER_TEST_TICK=wedge cargo test --release --bin minfer -- --ignored --exact <gate>
--nocapture`, **no outer `timeout`**:

| gate | exit | wall (whole process) | what failed | the client's answer |
|---|---|---|---|---|
| `serve_loop_publishes_the_queue_and_running_depth` | **101** | **0.29s** | *"the worker tripped its counted no-progress bound … left: 1, right: 0"* | two runs, one `500 the worker stalled` each |
| `a_job_rejected_for_want_of_a_slot_is_answered_with_503` | **101** | **0.18s** | *"a rejected job is unavailable: the worker stalled"* (`left: 500, right: 503`) | the served run got `500 the worker stalled` |

Both logs also carry `[server] the worker stalled: 64 consecutive steps left the engine busy without
advancing its work counter (N run(s), 0 queued job(s) answered with 500); stopping the worker`. A
whole-set wedge run with **no `--skip`** is **21 passed / 12 failed in 12.04s** (the nine bounded
steppers, the two `serve_loop` gates, and the #171 seam gate that cannot reach its forward under any
injection) — against #160's `exit 124` for the same two gates under an outer `timeout 30`. The
unmutated gates on the same commands are **pass, exit 0** (0.5B: 3.09s / 0.21s; Qwen3-0.6B: 3.69s /
0.17s).

**The `spin` arm still needs the two gates skipped — honest scope.** `MINFER_TEST_TICK=spin`
advances `work_units` without ever completing a run, so by construction a step that keeps "moving"
cannot trip a count of no-progress steps; the two `serve_loop` gates would still spin (they have no
step budget, and adding one is exactly the shape #196 offers as its second, rejected alternative).
The #160 command is unchanged:

| Command (2026-09-27, CPU build, dgxspark (aarch64)) | Result |
|---|---|
| `MINFER_TEST_TICK=spin cargo test --release --bin minfer -- --ignored --skip serve_loop_publishes_the_queue_and_running_depth --skip a_job_rejected_for_want_of_a_slot_is_answered_with_503` | **21 passed / 10 failed** in 12.21s (the #160 budget arm, unchanged) |

**Verification (2026-09-27, CPU build, dgxspark (aarch64)).**

| Command | Result |
|---|---|
| `cargo test --release` | **467 passed / 0 failed / 33 ignored** unit + **10 / 0 / 6** integration (baseline 465/0/33; the two CI stall gates) |
| the two gates `--ignored --exact`, unmutated, 0.5B | **1 / 0** each, 3.09s / 0.21s |
| the two gates `--ignored --exact`, unmutated, Qwen3-0.6B | **1 / 0** each, 3.69s / 0.17s |
| `scripts/real_model_gates.sh` (parallel, 0.5B) | **33 / 0** in 35.96s |
| `PARALLEL=0 scripts/real_model_gates.sh` (serial, 0.5B) | **33 / 0** in 38.80s |
| `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf scripts/real_model_gates.sh` (parallel) | **33 / 0** in 42.48s |
| `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf PARALLEL=0 scripts/real_model_gates.sh` (serial) | **33 / 0** in 46.95s |
| `MINFER_TEST_TICK=wedge`, whole `#[ignore]`d set, no `--skip` | **21 passed / 12 failed** in 12.04s, no hang |
| `MINFER_TEST_TICK=spin`, two `serve_loop` gates skipped | **21 passed / 10 failed** in 12.21s |
| `rustup run stable rustfmt --edition 2021 --check src/server/batch.rs src/server/metrics.rs` | clean (rustfmt 1.9.0-stable; the pinned 1.97.1 toolchain has no rustfmt component) |
| `python3 scripts/check_status.py --check` | exit 0 |
| `python3 scripts/check_docs_links.py` | **968 relative links / 186 markdown files**, exit 0 |

**Honest limits.**

- The bound is on the **loop**, not inside a single `tick`: a `tick` that never returns (or spins
  internally without returning to the loop) still has no in-process bound — only a cross-process
  watchdog can bound that, and none claims to here.
- The guard **stops the worker**. Requests after the stall get `503 server shutting down` until the
  process is restarted; that is the deliberate trade against spinning at 100% CPU with clients left
  hanging, and `minfer_worker_stalled_total` is the signal to alert on. A self-restart is out of
  scope.
- `STALL_STEP_LIMIT = 64` cannot be validated by observation on the healthy path (it never reaches
  1); it is slack, chosen by the argument above, not a measured constant.
- The `spin` arm above: a step that advances the work counter is not a stall by this guard's
  definition, so the two gates remain skipped in that mutation run. Filed as
  [#198](https://github.com/yusiwen/minfer/issues/198) — a production bound for the
  moving-but-non-terminating step, which must not false-positive a legitimately long generation.
- The CUDA unit row of `docs/status.toml` (546, GB10 sm_121, 2026-09-27) was **not** re-measured:
  this ticket's two new CI tests are feature-independent, so a CUDA build would read **+2**; the row
  stays the dated device record and is not claimed to include them (the GPU rows were out of scope).
- `AGENTS.md` no longer carries a per-module `server/batch.rs` bullet (the *Docs* lines of the #121
  and #151 records predate that refactor); the server contract now lives in `FEATURES.md`,
  `USAGE.md` and `OPENAI-CHAT-API-PLAN.md`, which are the files updated here.

**Docs.** `docs/GATE-CONTRACT.md` rule 4's [#160](https://github.com/yusiwen/minfer/issues/160) paragraph now ends with the resolution (the
`serve_loop` gates are wedge-proof; `STALL_STEP_LIMIT`; `minfer_worker_stalled_total`;
`FEEDER_POLL_BACKSTOP` demoted to a last resort). `docs/BUILD.md` § *Tests* and
`scripts/real_model_gates.sh`'s mutation note record the same. Every `/metrics` enumeration
(`USAGE.md` § *Metrics and observability*, `FEATURES.md`'s `serve` bullet,
`OPENAI-CHAT-API-PLAN.md` § *Slot Lifecycle*, the F8 table above) names the new counter.
`AGENTS.md`'s suite counts carry the +2 (467 aarch64 / 465 x86_64).
`ARCHITECTURE-ROADMAP.md` is untouched: no roadmap gap closes here — this is robustness inside the
existing test-infrastructure/batching rows, the same reason the #154, #151, #158 and #160 records
give.

#### Test-infrastructure record (#218, 2026-09-29) — the prefill-GEMM dynamic-smem opt-in is lazy, per-instantiation, and gated

**The finding.** [#188](https://github.com/yusiwen/minfer/issues/188) deleted the production call site
of `gemm_prefill_smem_init` — the **eager** dynamic-smem opt-in in `CudaState::try_new` — and said so
nowhere: the commit message has no `smem`/`shared`/`attribute`/`#145` token, the same PR's docs commit
rewrote `CUDA-BACKEND-DESIGN.md` (+174/−74) without mentioning it, and the implementation commit
touched six `.rs` files and no documentation. Two later dead-code-hygiene commits then annotated the
orphan `#[cfg_attr(not(test), allow(dead_code))]`: `f5a956a` under the banner *"55 items are reached
only from `#[cfg(test)]` code"*, and `58379c0` left it because the `#145` gate calls it. The
instrument that should have surfaced an orphaned production path instead silenced the diagnostic, and
the `#145` gate `cuda_prefill_smem_optin_covers_every_launchable_instantiation` kept certifying a
function production no longer called.

**The decision (plan B).** Do not reintroduce the eager init. Keep the lazy per-launch path
(`gemm_smem_optin`, reached from `launch_gemm_f16`), **remove** the orphan rather than rename it, and
make the invariant explicit and gated. The invariant: the >48 KiB attribute is in force **before** a
capture window opens and is never set **inside** one. Three mechanisms uphold it — the 3-run capture
warmup (`capture_warmup`, default 3), `cudaStreamCaptureModeThreadLocal` (#188's measured choice),
and the per-instantiation cache — and all three are now written down at their sites
(`graph_replay_step`'s comment, `gemm_smem_optin`'s comment, `CUDA-BACKEND-DESIGN.md` §2.4) so the
next person who changes the capture mode or the warmup count reads why. The `#145` gate was
re-pointed at the production function and renamed
`cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation`; the `checked`/`skipped`
introspection and the `gemm_prefill_smem_init` symbol are gone, and the remaining introspection is a
`#[cfg(test)] extern "C"` block, so no non-test build carries even a declaration for it (the
`allow(dead_code)` pattern this ticket exists to end).

**The bug the first gate run found.** The `gemm_smem_optin` cache was **not** per-instantiation. It
was `template <typename K>` and every `gemm_f16_nt_kernel_t<TM,KS,AF32>` shares one *signature*, so
`K` deduced to one function-pointer type and `static int state` was one cache for the whole family.
The re-pointed coverage gate found it on its first run: `<64,64,true>` (73728 B) answered "admitted"
for `<128,64,false>` (57344 B), whose attribute was never set — the device still reported its
49152 B default for the 57344 B request. #218 made the cache genuinely per-instantiation
(`template <int TM, int KS, bool AF32>`), which is what its comment always claimed. It is latent in
production today (only one `(tm, ks, af32)` is used per process — both tile env knobs are read once —
so nothing mixed families), but it is exactly the "unwritten, untested" property this ticket exists
to end.

**The gates.**

| gate | value it asserts | the precondition that keeps it non-vacuous |
|---|---|---|
| `cuda_prefill_smem_optin_is_done_by_production` | after a **real** prefill forward, the device's own `cudaFuncGetAttributes().maxDynamicSharedSizeBytes` read-back says `<128,64,false>` (57344 B) is opted in | it runs in a **fresh process** (`src/cuda/test_child.rs`) and asserts `opted_in == 0` *before* the forward — the tile env and the attribute are both process-scoped, so "before" is only observable in a process no earlier launch has touched |
| `cuda_prefill_smem_optin_refusal_fails_the_prefill` | the control arm: `MINFER_TEST_CALL_FAIL=attr:gemm_f16_f16` makes the production prefill **refuse** the launch and the site report name `cudaFuncSetAttribute` + `cudaErrorInvalidValue` | a **second** child process, so the per-instantiation cache cannot have answered already; env-gated behind `MINFER_TEST_ISSUE218=1` (it makes a real call fail, as #147/#162's gates are) |
| `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window` | a >48 KiB prefill-shaped graph **is captured** (`captured_count() == 1`) and replays bitwise over 5 steps, and `gemm_smem_optin_in_capture_count() == 0` — the opt-in ran before the window, never inside it | fresh process; `gemm_smem_need > 48 KiB` asserted; `captured_count() == 1` so an uncaptured path cannot pass; `opted_in == 0` before; `MINFER_TEST_CAPTURE_WARMUP=1` is the test-only seam that makes it fail |
| `cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation` (the re-pointed `#145` gate) | every launchable >48 KiB instantiation reads back opted in through the **production** `gemm_smem_optin`; every over-limit one is refused without a call | it drives the production function (via `gemm_prefill_smem_optin_one_for_test`), not a mirror; the over-limit combination is a separate negative arm |

**Mutation evidence (rule 3), GB10 sm_121, 2026-09-29.**
1. `gemm_smem_optin` answers `true` without calling `minfer_smem_optin` (C++, one line). The coverage
   gate goes red at `<64,64,true>` (*"cudaFuncGetAttributes reports its maxDynamicSharedSizeBytes
   below that"*); the real-prefill gate's child goes red with `minfer/cuda: kernel launch
   gemm_f16_nt_kernel_t<128,64,false> failed: cudaErrorInvalidValue (1) — the launch is refused
   (#162/launch:gemm_f16_f16)` and then *"cuda: prefill GEMM (f16): launch_gemm_f16 refused the
   launch … no kernel ran"*; the capture gate's child goes red on the same refused launch. One
   mutation, three gates.
2. `MINFER_TEST_CAPTURE_WARMUP=1` (the documented seam, from the parent — the harness passes it
   through on purpose): the child's whole gate runs — the capture happens, replays bitwise, and
   `opted_in == 1` — and only `gemm_smem_optin_in_capture_count()` trips:
   *"assertion `left == right` failed: the smem opt-in must be performed before a capture window
   opens, never inside one — this is the load-bearing part of the design; left: 1, right: 0"*.
3. `prefill_gemm_f16_inner`'s `if (launched == 0) return Err(..)` arm removed (Rust): the control
   arm's `expect_err` goes red — *"the injected attribute failure must refuse the >48 KiB prefill
   launch: ()"*.

**Honest limit — now measured, and it still does not license removing the warmup.** Mutation 2 is
also the experiment the #218 issue said had never been run: it drives `cudaFuncSetAttribute` into an
open **same-thread `cudaStreamCaptureModeThreadLocal`** window. On this runtime the call is
**tolerated** — the attribute publishes, the capture completes, instantiates and replays
bitwise-identically; only the counter moves. So on GB10 sm_121 / CUDA 13.0 / driver 580.178.04 an
in-window opt-in would work. The design nevertheless keeps the call out of the window, because the
historical failure is real on other toolkits, the 2026-09-25 probe measured the *Global* mode (not
the adopted one), and the whole point of the #218 gate is to pin the property as an *observed*
invariant rather than rely on driver behaviour. The counter gate is what makes that true regardless
of what the driver tolerates.

**Counts (rule 5).** `scripts/cuda_test.sh` → **565 / 0 / 42**, GB10 sm_121, 2026-09-29
(was 562 / 0 / 42; +3 device gates: the two fresh-process `issue218_tests` arms and the captured-graph
arm). `compute-sanitizer --tool memcheck --target-processes all <test binary> --test-threads=1` →
**0 API errors** over 565 / 0 / 42 (one aggregated `ERROR SUMMARY` for the whole process tree — the
fresh-process children are followed too). The two real-model configurations were re-run and stay
green: `FEATURES=cuda scripts/real_model_gates.sh` → **42 / 0** (0.5B config) and the same command
with `MINFER_BATCH_TEST_MODEL=~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` →
**42 / 0**, GB10 sm_121, 2026-09-29.

**Docs.** `CUDA-BACKEND-DESIGN.md` §2.4 (the eager claim replaced by the lazy path, the three
load-bearing mechanisms, the gate list, the measured in-window experiment);
`cuda_optimization_steps/02-wmma-f16-prefill-gemm-8m.md` and this file's #145 record carry dated
forward notes (history is not rewritten); `inference_e2e_walkthrough/15-cuda-backend.md` no longer
says `gemm_prefill_smem_init()` runs eagerly; `docs/GATE-CONTRACT.md` rule 1 gains the reusable
lesson from this ticket — *a gate must exercise the production entry point, not a test-only helper
that mirrors it* — with the #145 gate as the instance. `docs/status.toml` and `AGENTS.md` carry the
new CUDA counts.

**Process lesson.** A dead-code campaign treated "an item with no production caller that looks like
production" as a fact to annotate rather than a question to ask. `allow(dead_code)` on a
test-reachable production-looking item is a **deferred question**; the fix is to list such items for
review. Future hygiene work should treat the annotation as a lead, not a resolution.

> **#223 forward note (2026-09-29):** the "tested, not enforced" gap this record names is closed.
> `CudaState::try_new` now runs the eager pre-warm through the production `gemm_smem_optin`, at the
> earliest point in the process, where no capture window can exist yet; the lazy path and the three
> mechanisms this record lists remain as defence in depth. The four gates above keep their claims by
> running their children with `MINFER_NO_GEMM_PREWARM=1` (the documented control and the "lazy path
> alone" arm); the runtime guarantee has its own fifth gate. See the #223 record below.

#### Test-infrastructure record (#223, 2026-09-29) — the eager prefill-GEMM smem pre-warm is back, as a runtime guarantee through the lazy entry

**The gap #218 left.** #218 made the prefill-GEMM dynamic-smem invariant explicit, gated and correctly
cached, but production still upheld it only by *emergent* means: the 3-run capture warmup,
`cudaStreamCaptureModeThreadLocal`, and the fact that the cache happens to be consulted on the first
launch. The attribute was a **tested** property, not an **enforced** one — [#188](https://github.com/yusiwen/minfer/issues/188)
had deleted the eager caller and nobody noticed.

**The design (plan A).** `CudaState::try_new` drives the production pre-warm once per process for
every launchable `(tm, ks, af32)`, at the earliest point in the process. The placement *is* the
argument: `try_new` runs under `CUDA.get_or_init`, before the state is published and before any
`CudaBackend` — the only thing that can hold a capture window — can exist, so "the attribute is set
outside any window" holds **by construction**. Crucially it is **not** the old `gemm_prefill_smem_init`
sweep: the Rust side enumerates the set and each entry goes through the new production
`gemm_prefill_smem_prewarm_one` → `gemm_smem_optin<TM,KS,AF32>` — the same per-instantiation cache the
launcher reads. One mechanism, one `cudaFuncSetAttribute` site, one copy of `gemm_dynamic_smem_bytes`.
The C++ `MINFER_GEMM_OPTIN_SET` X-macro is now the single list of the launchable set (fatbin lookup,
pre-warm and the #218 test seam all expand it). The lazy per-launch opt-in stays as defence in depth,
and the three #218 mechanisms are demoted to defence in depth behind the pre-warm. A request above
`cudaDevAttrMaxSharedMemoryPerBlockOptin` is skipped without calling the attribute (reason named); a
failure is named per instantiation by `minfer_smem_optin`; the `checked`/`skipped` counters and the
startup banner did not come back — a fully admitted pre-warm is silent. `MINFER_NO_GEMM_PREWARM=1` is
the documented control.

**The fifth gate.** `issue223_tests::cuda_prefill_smem_prewarm_opts_in_every_launchable_instantiation_before_any_launch`
asserts, in a fresh process immediately after context creation and before any launch, that every
launchable >48 KiB instantiation already reads back opted in. Non-vacuity: a second fresh process with
`MINFER_NO_GEMM_PREWARM=1` asserts the negation (the read-back is capable of answering 0), the child
launches nothing before the check, and the assertion names the load-bearing
`gemm_f16_nt_kernel_t<128,64,false>` (57344 B). This gate exists for a mutation the #218 arms cannot
see: **skip one `(tm, ks, af32)` in the pre-warm and the lazy path simply opts it in on first launch**,
so the coverage/counter arms stay green. The four #218 arms now run their fresh-process children with
`MINFER_NO_GEMM_PREWARM=1`, which is where their pre-#223 `opted_in == 0` preconditions are observable;
their claims are unchanged (they are the lazy-path-alone arm) and they remain the cache-keying
detector.

**Mutation evidence (rule 3), GB10 sm_121, 2026-09-29.** Drop `(128, 64, 0)` from `GEMM_PREWARM_SET`
(the Rust production list; replaced by a duplicate so the array still type-checks) and run the new
gate: the pre-warmed child fails with *"immediately after context creation and before any kernel
launch, gemm_f16_nt_kernel_t<128,64,false> (57344 B > 48 KiB) must already read as opted in … left: 0,
right: 1"*. Under the **same** mutation
`cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation` stays **green** (its child skipped
the pre-warm, so the lazy path opts the dropped instantiation in on first launch) — the transcript that
justifies the fifth gate. The over-limit skip is visible in every pre-warmed child's stderr:
`cudaFuncSetAttribute(gemm_f16_nt_kernel_t<256,64,true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
122880 B) SKIPPED: the request exceeds cudaDevAttrMaxSharedMemoryPerBlockOptin (101376 B) …`.

**Performance, measured (rule 5) — the real binary disagrees with the #223 proxy by ~15×, and the net
is still zero.** Date/device/command: 2026-09-29, GB10 sm_121, CUDA 13.0, driver 580.178.04,
`MINFER_OP_TIMING=1 ./target/release/minfer bench -p 8 -n 1 -r 1`, 25 fresh processes.

| bar (named before measuring) | proxy in #223 | measured | verdict |
|---|---|---|---|
| pre-warm loop's own duration | ≲ 0.2 ms (first attr call 152.9 µs) | **median 2249 µs** (range 2126–2448, n=25) | above the stated bar — it is the fatbin's one-time module load, not 152.9 µs; set-size independent (n=1 ≈ n=12 ≈ 2.2 ms). The range is **warm-clock only** — see the cold row below |
| pre-warm loop's own duration, first (**cold / idle-clock**) invocation | same one-time work, no separate bar named | **~14 526 µs** (~6× the warm median; the three consecutive runs were 14526, 2157, 2364 µs) | the 2126–2448 range is **not** the worst case: the module load is clock/state dependent, and its first cold run measured ~14.5 ms ([#225](https://github.com/yusiwen/minfer/issues/225)) |
| `minfer bench -p 2048 -n 128` `tg128` | within ±1% | ON 236.30 vs OFF 236.45 t/s (**−0.06%**) | pass (7 interleaved matched rounds, medians, same binary) |
| `minfer bench -p 2048 -n 128` `pp2048` | within ±1% | ON 2546.55 vs OFF 2544.34 t/s (**+0.09%**) | pass |
| startup (model load → first token) | net new ≈ 10 µs | **net ≈ 0**: the ~2.2 ms **moves** into the existing `prewarm_prefill()` module load | the proxy's qualitative reading survives. The ≈ 0 net is **coupled**: it holds only while a later step pays that same module load — move `prewarm_prefill()` after the first launch, or remove it, and the pre-warm's loop becomes ~2.2 ms of net-new startup cost |

The "the ~150 µs moves rather than appears" reading **did** survive on the real path, with a different
magnitude. Controlled fresh-process probe on the same binary (the tiny `gemm_f16_nt_kernel_t` fixture,
`MINFER_MMQ` 0/1 × `MINFER_GEMM_K64` 0/1): first prefill forward **2288–2314 µs** with the pre-warm
off vs **78–120 µs** with it on; `prewarm_prefill()` (the r59 rider's `minfer_prewarm_kernels`, called
at the end of Qwen2/Qwen3 weight registration) **4.3–4.6 ms** off vs **2.2–2.4 ms** on. So the ~2.2 ms
is a one-time fatbin module load that the startup path already pays; the pre-warm only decides *where*
— at `try_new` instead of at the end of registration. The end-to-end CLI phase measurement is
consistent with that but cannot resolve the net to better than a few ms: paired `cuda_ready`
(mid-context-creation) +2.53 ms, `forward_ms` +0.05 ms, model-load phase noise ±20 ms. The honest
residual: if the module load ever stops being paid by `prewarm_prefill()` (e.g. that rider is removed),
the pre-warm's loop becomes ~2.2 ms of **net new** startup cost, so the two are coupled and the next
person to touch `prewarm_prefill` must know.

> **#225 forward note (2026-10-04):** the table's 2126–2448 range is a **warm-clock** measurement of
> the one-time fatbin **module load**, not a bound. An independent three-run check under the same
> command (2026-09-29, GB10 sm_121) saw the first, cold / idle-clock invocation take **14 526 µs**
> (~6× the warm median; the next two runs 2157 µs and 2364 µs), so the range above is warm-only and
> the load's magnitude is clock/state dependent. `CUDA-BACKEND-DESIGN.md` §2.4's cost table now
> carries both readings. The coupling is load-bearing: the ≈ 0 net holds **only while a later step
> pays the same module load** — today `prewarm_prefill()`, otherwise the first launch from the fatbin.

**Counts (rule 5).** `scripts/cuda_test.sh` → **566 / 0 / 42**, GB10 sm_121, 2026-09-29 (was 565 / 0 /
42; +1 device gate, `cuda::issue223_tests`). `compute-sanitizer --tool memcheck --target-processes all
<test binary> --test-threads=1` → **0 API errors** over 566 / 0 / 42. The two real-model configurations
were re-run and stay green: `FEATURES=cuda scripts/real_model_gates.sh` → **42 / 0** (0.5B config) and
with `MINFER_BATCH_TEST_MODEL=~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` →
**42 / 0**, GB10 sm_121, 2026-09-29.

**Docs.** `CUDA-BACKEND-DESIGN.md` §2.4 now states the restored design (eager pre-warm at context
creation + lazy per-launch opt-in), demotes the three mechanisms to defence in depth, names the fifth
gate, and carries the measured cost table; `cuda_optimization_steps/02-wmma-f16-prefill-gemm-8m.md`
and this file's #145 and #218 records carry dated forward notes (history is not rewritten);
`inference_e2e_walkthrough/15-cuda-backend.md` still describes only `prewarm_prefill()`, which is
unchanged. `docs/status.toml` and `AGENTS.md` carry the new CUDA counts.

#### Test-infrastructure record (#228, 2026-09-30) — the E1 test corpus moves onto the production `fill_batch_inputs`

**The finding (S1 of [#227](https://github.com/yusiwen/minfer/issues/227)).**
`GraphAllocator::fill_attn_inputs` (`alloc.rs:1724` on master `b5a9d5f`) was a
production-shaped entry point with `#[cfg_attr(not(test), allow(dead_code))]` and a
doc that claimed *"the model path and hand-built graphs both use it"*. The model path
does not: `forward_batch` and `forward_cached` — the latter literally
`forward_batch(&Batch::single(tokens, positions), …)` (`models/qwen2/graph.rs:458-465`)
— call `fill_batch_inputs`, a **separate implementation** of the same job. The
coverage was therefore inverted: E2 had **one** test call site
(`qwen2::graph::tests:2068`) while E1, which production never runs, had **15**. A
change to E2's COW ordering, reservation or ownership recording could leave the suite
green — the "mirrored helper" hazard of [GATE-CONTRACT §1](GATE-CONTRACT.md).

**Classification of the 15 E1 call sites (step 1 done before any rewriting).** Every
one asks the same question — can the scenario be a `Batch`? — and the answer is
**yes for 14**; the exception is the rope-only fixture.

| file:line (master) | driving test | graph has `cells`? | disposition |
|---|---|---|---|
| `alloc/tests.rs:423` | `kv_session_round_trips_the_rows_and_the_run_table` | yes | moved → `Batch::new(_, _, [SEQ, SEQ])` |
| `alloc/tests.rs:558` | `an_f16_session_round_trips_through_save_and_load` | yes | moved → `Batch::new` |
| `alloc/tests.rs:794` | `a_copy_on_write_moves_the_rows_and_never_writes_through` | yes | moved → `Batch::new(_, _, [2])` (not `Batch::single`: that names `SEQ_MAIN`, which holds no run here) |
| `cpu_backend/tests.rs:377` | `embedding_and_rope` | **no** | **exception** — see below |
| `cpu_backend/tests.rs:436` | `kvcache_store_load_and_attn_roundtrip` | yes | moved → `Batch::single` |
| `cpu_backend/tests.rs:507` | `a_packed_kv_region_answers_like_the_f32_one_and_is_smaller` | yes | moved → `Batch::single` |
| `cpu_backend/tests.rs:678` | `a_packed_physical_shift_moves_v_verbatim_and_requantizes_k` | yes | moved → `Batch::single`; the redundant post-execute `kv_note_used(nt)` deleted with it |
| `qwen2/graph/tail_tests.rs:203` | `tail_reduction_matches_full_nt` (`run_keep`) | yes | moved → `Batch::single` |
| `qwen2/graph/tail_tests.rs:403` | `fused_qkv_matches_unfused_decode` | yes | moved → `Batch::single` |
| `qwen2/graph/tail_tests.rs:498` | `fused_qkv_matches_unfused_decode` | yes | moved → `Batch::single` |
| `qwen2/graph/tests.rs:2907` | `graph_logits_match_forward_real_model` (`run_prefill_decode`) | yes | moved → `Batch::single` |
| `qwen2/graph/tests.rs:2955` | `graph_logits_match_forward_real_model` (decode step) | yes | moved → `Batch::single(&[next], &[nt])` |
| `qwen2/graph/tests.rs:3107` | `graph_metal_layer0_isolation` (macOS) | yes | moved → `Batch::single` |
| `qwen2/graph/tests.rs:3132` | `graph_metal_layer0_isolation` (macOS) | yes | moved → `Batch::single` |
| `qwen3/graph/tests.rs:406` | `metal_prefill_determinism` (macOS) | yes | moved → `Batch::single` |

**The no-`cells` case exists — and it is a no-op.** `cpu_backend::tests::embedding_and_rope`
builds an embedding + RoPE graph with **no** `kvcache_store`, so it has none of
`seq_ids` / `cells` / `kv_map` / `attn_span`. E1's `has_cells` guard made its call
there a **complete no-op**, and E2 cannot express the shape at all: with no KV node
there is no arena (`n_ctx() == 0`), and `fill_batch_inputs`'s per-group implicit
reservation is `reserve_seq(seq, 0)`, which is refused. So this is the one site that
keeps the shape, as `#[cfg(test)]`-scoped
`GraphAllocator::fill_attn_inputs_without_cells` — the `has_cells == false` branch and
nothing else, `debug_assert`ed to reject a `cells` graph, with the doc naming
`embedding_and_rope`. It is `#[cfg(test)]`-scoped, not `allow(dead_code)`-silenced, so
a non-test build cannot see it.

**What was deleted, and the honest disposition of each item that lost its only
caller.**

- `GraphAllocator::fill_attn_inputs` — deleted with its header comment (it was not an
  entry point).
- `GraphAllocator::kv_note_used` (`alloc.rs:1240`) — after the move its last caller was
  the redundant `cpu_backend::tests:683` line; that line is gone, so it had **nothing**
  and was deleted (a dead item behind an `allow` is exactly the pattern Core
  Convention 5 exists to stop). Its doc claimed "the model calls this after a forward
  with `max(positions)+1`" — false; production records the extent through
  `KvCache::own_positions`.
- `KvCache::own_prefix` (`kvcache.rs:582`) — `kv_note_used` was its last non-test
  caller, so it is now **test-only**; it keeps its `#[cfg_attr(not(test),
  allow(dead_code))]` and its doc now names the four `kvcache::tests` that drive it.
- **`kv_cells_for_seq` is *not* an orphan — the #228 premise (inherited from #227)
  was tested and falsified.** The premise was that its only non-test callers are E1
  and S3's `kv_cell_of`, so it is production-unreachable yet unannotated, "silent only
  because both its callers are annotated", and therefore missed by an
  annotation-grep census. The reachability chain says otherwise:
  `models/qwen2/graph.rs:660` / `qwen3/graph.rs:567` → `fill_batch_inputs` →
  **`fill_seq_ids` → `kv_cells_for_seq`** (the `has("cells")` branch), and every model
  graph has a `cells` input because `kvcache_store` always creates one
  (`models/qwen2/graph.rs:217`). It was verified at runtime, not by grep: an
  env-gated `eprintln!` at the top of `kv_cells_for_seq` plus
  `cargo test --release models::qwen2::graph::tests::graph_logits_match_forward_real_model`
  → **4 hits**, test green (probe reverted). So it is production-reachable, has no
  annotation because rustc is right, and **the annotation-grep/transitive-closure
  worry does not apply to this subgraph: `kv_cells_for_seq` needs no change.** S1 and
  S3 are still one subgraph (joined through `kv_cells_for_seq`), but the closure
  direction is inverted, and that inversion is the result #227's census method needs.
  The genuinely dead chain is the smaller one:
  `fill_attn_inputs` → `kv_note_used` → `own_prefix` (E1 and `kv_note_used` deleted;
  `own_prefix` now test-only with its doc naming the tests). Its documented role is
  unchanged; only the "two fill entry points" comment was corrected.
- Residual, not fixed (out of this ticket's scope, recorded): `kv_cells_for_seq`'s
  `classic` branch (`arena_stats().sequences == 0` → `cell == position`) is now
  reachable only from `alloc/tests.rs`; production always reserves before
  `fill_seq_ids` runs, because `fill_batch_inputs`'s first loop covers every group.

**Mutation evidence (rule 3), CPU aarch64, 2026-09-30.** The production path must be
the thing that goes red. Mutation: in `fill_batch_inputs`, own the **wrong** positions —
`own_positions(seq, positions.iter().map(|p| p + 1))` — so the ownership recording is
off by one while everything still compiles (a plain "skip `own_positions`" mutation
does *not* compile: the item would become dead and `#![cfg_attr(not(test),
deny(warnings))]` fires, which is itself evidence the recording is load-bearing).

```
cargo test --release
test result: FAILED. 478 passed; 3 failed; 36 ignored; 0 measured; 0 filtered out
  graph::alloc::tests::kv_session_round_trips_the_rows_and_the_run_table
    src/graph/alloc/tests.rs:445: assertion `left == right` failed: positions 0..4 are written
      left: 5   right: 4
  graph::cpu_backend::tests::a_packed_physical_shift_moves_v_verbatim_and_requantizes_k
    src/graph/cpu_backend/tests.rs:722: assertion `left == right` failed: one row removed
      left: 3   right: 2
  models::qwen2::graph::tests::kv_rm_is_exact_and_the_window_shift_is_a_named_tolerance_class
    src/models/qwen2/graph/tests.rs:2676: assertion `left == right` failed: only A survives removing B
      left: 8   right: 7
```

Two of the three are **moved** E1 call sites; the third,
`kv_rm_is_exact_and_the_window_shift_is_a_named_tolerance_class`, reaches
`fill_batch_inputs` through the real production caller `forward_graph_cached` — the
strongest form of the point, because the mutation is visible to production's own path and
not only to test-driven fills. Mutation reverted; `grep -n MUTATION src/graph/alloc.rs`
empty and `git diff` clean of it.

**Counts (rule 5).** No `#[test]` was added or removed, so the suite counts are
unchanged and **no CUDA row was re-measured**: `cargo test --release`, box
`dgxspark (aarch64, GB10 sm_121)`, 2026-09-30 → **481 / 0 / 36** unit + **10 / 0 / 6** integration (the same
as the recorded CPU row). `cargo fmt --all --check` clean; the non-test build
warning-free with and without `--features cuda` (the one `warning:` line on the CUDA
build is `build.rs`'s pre-existing `cargo:warning=` target list, not a rustc
diagnostic).

**Limits.** (1) The macOS-only arms (`graph_metal_layer0_isolation`,
`fused_qkv_matches_unfused_decode`, `metal_prefill_determinism`) are type-checked only
by the CI `build-macos` job here — no Mac. (2) The x86_64 CPU row cannot be computed
locally; it is unchanged by construction (same test count) and the CI `test-linux-cpu`
log is its source. (3) The no-`cells` helper's claim is weak by nature ("filling a
graph with no KV input resolves nothing and touches no arena"); it is kept because the
ticket asks for exactly that shape, not because it catches a bug.

#### Test-infrastructure record (#232, 2026-09-29) — `GraphAllocator::kv_own_range` deleted; the tests drive `KvCache::own_range`

**The finding (S2 of [#227](https://github.com/yusiwen/minfer/issues/227)).**
`GraphAllocator::kv_own_range` (`alloc.rs:1645-1650` on master `f4d3390`) was a one-line
forwarder — `self.kv.own_range(seq, from, to)` — carrying
`#[cfg_attr(not(test), allow(dead_code))]` and a doc that claimed a production role:
*"Mark cells `[from, to)` as written by `seq` in every layer (E2's batched forwards write
several sequences per step)."* Production does not call it; its only callers were five
sites in `src/graph/alloc/tests.rs`.

**One implementation, not two.** Unlike S1's `fill_attn_inputs`, the wrapper and production
share **one** implementation: the inner, unannotated `KvCache::own_range`
(`kvcache.rs:536`). It is production-reachable through two paths that do not go through the
wrapper:

- `fill_batch_inputs` (`alloc.rs:1697` on master) → `KvCache::own_positions`
  (`kvcache.rs:574`) → `own_range`; production reaches `fill_batch_inputs` from
  `models/qwen2/graph.rs:660` and `models/qwen3/graph.rs:567` (`forward_batch`).
- `GraphAllocator::kv_copy_prefix` (`alloc.rs:1488`) → `own_range`; its production caller
  is `server/batch.rs:821`.

*(Correction to [#232]'s body and [#227]'s S2 text: they place `alloc.rs:1488` inside
`kv_private_row_for`. It is not — `kv_private_row_for` (`alloc.rs:1516`) drives
`KvCache::private_row_for`/`apply_private_row` and never calls `own_range`. Line 1488 is
the tail of `kv_copy_prefix`. The conclusion — production does not call the wrapper — is
unaffected, but the reachable path is the batch prefix-copy, not the copy-on-write.)*

**The disposition (b): delete, and move the call sites to the production spelling.** The
wrapper and its doc comment are deleted. All **five call sites** — in **four** test
functions — now spell `a.kv.own_range(...)` / `alloc.kv.own_range(...)`:

| test | call sites (master) | disposition |
|---|---|---|
| `kv_session_round_trips_the_rows_and_the_run_table` | `:433` | `a.kv.own_range(SEQ, 0, 4)` |
| `an_f16_session_round_trips_through_save_and_load` | `:570` | `a.kv.own_range(SEQ, 0, 2)` |
| `kv_defrag_moves_the_bytes_and_opens_the_run` | `:629` | `alloc.kv.own_range(seq, …)` |
| `a_copy_on_write_moves_the_rows_and_never_writes_through` | `:739`, `:742` | `alloc.kv.own_range(1, 0, 4)` / `(2, 4, 6)` |

`src/graph/alloc/tests.rs` is a child module of the type's module (`alloc.rs:2735`
`#[cfg(test)] mod tests;`) and `kv` is a private field (`alloc.rs:174`), so the direct field
access compiles **as predicted** — the fallback (`#[cfg(test)] impl GraphAllocator` inside
`alloc/tests.rs`) was not needed. **Every assertion is unchanged**; the diff is the five
spellings plus the deleted wrapper. The issue's "five tests" is five *call sites* in four
test functions — no test was dropped or weakened.

**Mutation evidence (rule 3), CPU aarch64, 2026-09-29.** The mutation is in the
**production-used** `KvCache::own_range`: an off-by-one in the written extent —
`Some(pos) => written = written.max(pos + 1)` → `written.max(pos)`.

```
cargo test --release graph::alloc::tests::
test result: FAILED. 34 passed; 3 failed; 0 ignored; 480 filtered out

---- graph::alloc::tests::a_copy_on_write_moves_the_rows_and_never_writes_through ----
panicked at src/graph/alloc/tests.rs:743:47:
called `Result::unwrap()` on an `Err` value:
  "share_prefix: sequence 1 has written 3 rows, 4 requested"

---- graph::alloc::tests::kv_session_round_trips_the_rows_and_the_run_table ----
panicked at src/graph/alloc/tests.rs:445:5:
assertion `left == right` failed: positions 0..4 are written
  left: 3   right: 4

---- graph::alloc::tests::kv_defrag_moves_the_bytes_and_opens_the_run ----
panicked at src/graph/alloc/tests.rs:662:5:
assertion `left == right` failed
  left: 3   right: 4
```

Three of the four moved tests go red — the required "at least one" with margin. The fourth
(`an_f16_session_round_trips_through_save_and_load`) stays green under this mutation because
its `written` is already raised to the asserted value by `fill_batch_inputs`'s
`own_positions`; only the owner-table half of its `own_range` call is load-bearing, and that
half is what the copy-on-write test asserts. Mutation reverted; `grep -rn MUTATION src/`
empty and `git diff` clean of it (`src/graph/kvcache.rs` has no diff).

**Counts (rule 5).** No `#[test]` was added or removed, so no row moves and **no CUDA row was
re-measured**: `cargo test --release`, box `dgxspark (aarch64, GB10 sm_121)`, 2026-09-29 → **481 / 0 / 36**
unit + **10 / 0 / 6** integration (the recorded CPU row). `cargo fmt --all --check`
clean; `python3 scripts/check_status.py --check` exits 0; the non-test build warning-free
with and without `--features cuda` (`cargo build --release` and `cargo build --release
--features cuda`; the one `warning:` line on the CUDA build is `build.rs`'s pre-existing
`cargo:warning=` target list, not a rustc diagnostic).

**Limits.** (1) No CUDA device is used here: the `--features cuda` line is a compile, not a
run; the CUDA *unit* row is unchanged by construction (no `#[test]` moved) and is not
re-measured. (2) The x86_64 CPU row cannot be computed locally; it is unchanged by
construction and the CI `test-linux-cpu` log is its source. (3) The moved call sites
certify `KvCache::own_range`'s owner-table and written-extent halves only as far as those
tests assert them; the f16 session test does not depend on the mutated `written` arm.

#### Test-infrastructure record (#236, 2026-09-30) — `GraphAllocator::kv_cell_of` is test-only; production reads a sharer's rows as `kv_map` windows

**The finding (S3 of [#227](https://github.com/yusiwen/minfer/issues/227)).**
`GraphAllocator::kv_cell_of` (`alloc.rs:1829` on master `691a9d2`) was a one-line forwarder —
`self.kv.cell_of(seq, pos)` — carrying `#[cfg_attr(not(test), allow(dead_code))]` and a doc that
claimed a production role: *"…this answers 'where would a reader look?', which is what a caller
snapshotting a sharing sequence's rows needs."* No such caller exists, and none was deleted:
`git log -S 'kv_cell_of' --all` over `src/` names exactly three commits — `d43e716` (C8b S3 2/4,
which introduced the forwarder **and** that doc), `c9bbcbd` (C8b S3 3/4, the test call) and
`54f6de0` (the test-module extraction). Its only call site is `server/batch/tests.rs:514` inside
`kv_rows_of` (`:502`), the observation instrument of the C8b S3 gate
`a_store_inside_a_shared_prefix_takes_a_private_row` (`:385`).

**The consumer class the doc named is served two other ways — neither needs position→cell.**
(1) Reading a sharing sequence's rows **for attention** goes through *windows*:
`GraphAllocator::fill_seq_ids` → `KvCache::attn_map` (`kvcache.rs:803`) → the `kv_map` input, which
the CPU and CUDA attention kernels gather (C8b S2/S4). It returns **runs**, not per-position cells.
(2) Whole-run snapshots go through the C5 container: `BatchEngine::save_slots` (`server/batch.rs:442`)
calls `kv_save_with_host` (`alloc.rs:2172`), which stores the **whole arena** with the slot table as
its host section — no position→cell mapping anywhere. C8b is closed (S1a–S5 landed 2026-09-21/22),
so no pending slice was meant to consume the accessor. The `kv` field it forwards into is private
(`alloc.rs:174`), which is why the tests cannot bypass the wrapper and call `KvCache::cell_of`
directly from `server::batch::tests`; it is the forwarder, not `KvCache::cell_of`, that lacks a
production caller.

**The disposition (b): `#[cfg(test)] pub(crate)`, in place.** The annotation and the doc are replaced
on the item itself (the `pub(crate) fn` is at `alloc.rs:1840` after the change), matching
[#228](https://github.com/yusiwen/minfer/issues/228)'s precedent
(`fill_attn_inputs_without_cells` is `#[cfg(test)]`-scoped in place). The equivalent alternative —
a `#[cfg(test)] impl GraphAllocator` in `src/graph/alloc/tests.rs` — was rejected: `alloc/tests.rs`
is unit-test scaffolding for `alloc`, and this method's one consumer is in
`server::batch::tests`, so putting the type's surface in a different module's test file would move
the boundary without moving the caller. **No call site changed** — an inherent method resolves
wherever its `impl` lives, and `pub(crate)` keeps the cross-module caller compiling. `#[cfg(test)]`
is strictly stronger than `allow(dead_code)`: the method **does not exist** in a non-test build, so
a later cleanup cannot leave a production-looking orphan behind. `KvCache::cell_of` itself stays
`pub` and production-used (the store resolver `kv_cells_for_seq` resolves through it); only the
wrapper is gated.

**Correction to an earlier record's wording.** The #228 record lists "S3's `kv_cell_of`" among
`kv_cells_for_seq`'s *non-test* callers while it reports the premise it then falsifies.
`kv_cell_of` forwards to `KvCache::cell_of` (`kvcache.rs:512`) and never called
`kv_cells_for_seq`; the falsification that record reports — `kv_cells_for_seq` **is**
production-reachable through `fill_seq_ids` — is unaffected, and as of this record `kv_cell_of`
is itself test-only. The #228 text is left as written (it is a historical record); this is the
dated forward correction for it.

**Bar named before measuring.** No test loses its assertion, and the driver stays sensitive:
`cargo test --release` stays 481 / 0 / 36 unit + 10 / 0 / 6 integration, the ignored S3 gate is
green on the unmutated tree, and a `cell_of` mutation must be visible to `kv_rows_of`.

**Mutation evidence (rule 3), box `dgxspark (aarch64, GB10 sm_121)`, 2026-09-30.** Two mutations of
the **production-used** `KvCache::cell_of` (`kvcache.rs:512`), each run as
`cargo test --release server::batch::tests::a_store_inside_a_shared_prefix_takes_a_private_row -- --ignored --nocapture`,
each reverted. Baseline (unmutated) run: `1 passed; 0 failed; 0 ignored; 516 filtered out`.

*(a) The span offset, `cell + (pos - base)` → `+ 1`.* The gate goes red — but at the **answer** arm,
not through the instrument:

```
thread 'server::batch::tests::a_store_inside_a_shared_prefix_takes_a_private_row' panicked at
src/server/batch/tests.rs:492:5:
assertion `left == right` failed: the shared run must answer like the shape-matched copied one
  left: ".\nA. the a\nB."
 right: "\nA. Rome\nB. Naples"
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 516 filtered out; finished in 1.50s
```

Every `kv_rows_of`-driven assertion **passed** under it, and that is structural, not luck:
`kv_rows_of` resolves its cells through the *same* mutated `cell_of` the store uses, so a consistent
offset cancels in a before/after snapshot
(`assert_eq!(kv_rows_of(donor), donor_before)`) and in an A/B of two identical runs
(`assert_eq!(dst_a, dst_b)`). The arm that caught it compares the sharing run's **generated answer**
against the shape-matched copied run — generation reads through `attn_map`, which never calls
`cell_of`, so the store's shifted cells surface as a wrong answer instead of a shifted snapshot. So
the ticket's own example is evidence that the *gate* is
sensitive, but **not** that the row-snapshot instrument is; it is recorded here rather than dropped,
and the instrument is exercised by (b).

*(b) The span cover, `pos < base + len` → `pos + 1 < base + len`* — the last position of every span
resolves to `None`. This one the instrument refuses itself:

```
thread 'server::batch::tests::a_store_inside_a_shared_prefix_takes_a_private_row' panicked at
src/server/batch/tests.rs:515:36:
no cell for sequence 2 position 2
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 516 filtered out; finished in 0.60s
```

`tests.rs:515` is `kv_rows_of`'s own
`.unwrap_or_else(|| panic!("no cell for sequence {seq} position {p}"))` — the failing assertion
**is** the instrument, reached through `kv_cell_of` → `KvCache::cell_of` while it snapshots the
diverging run's rows, before the answer arm can run. So at least one `kv_rows_of`-driven refusal
fails under a mutation of the production `cell_of`, and the row snapshot is load-bearing. Both
mutations reverted; `git diff src/graph/kvcache.rs` empty and `grep -rn MUTATION src/` empty.

**Counts (rule 5).** No `#[test]` was added or removed, so no row moves and **no CUDA row was
re-measured**: `cargo test --release`, box `dgxspark (aarch64, GB10 sm_121)`, 2026-09-30 → **481 / 0 / 36**
unit + **10 / 0 / 6** integration (the recorded CPU row). `cargo fmt --all --check` clean;
`python3 scripts/check_status.py --check` exits 0; the non-test build warning-free with and without
`--features cuda` (`cargo build --release` and `cargo build --release --features cuda`; the one
`warning:` line on the CUDA build is `build.rs`'s pre-existing `cargo:warning=` target list, not a
rustc diagnostic).

**Docs.** `AGENTS.md` rule 1's last sentence no longer claims a production consumer (it now says
test-only, names the `kv_map`/`kv_save*` production paths and the one consumer); the C8b S3
paragraph above carries a dated forward note; `docs/COMPUTE-GRAPH-DESIGN.md` was re-checked and
never made the claim.

**Limits.** (1) No CUDA device is used here: the `--features cuda` line is a compile, not a run; the
CUDA *unit* row is unchanged by construction and is not re-measured. (2) The x86_64 CPU row cannot
be computed locally; it is unchanged by construction and the CI `test-linux-cpu` log is its source.
(3) The S3 gate is `#[ignore]`d (it needs the cached 0.5B model), so the mutation transcripts come
from `-- --ignored`, not from the default `cargo test --release` run. (4) The mutation that reaches
the instrument is the span-**cover** one, not the span-**offset** one the ticket names; which arm
each kills is stated above, and neither is presented as the other.


#### Test-infrastructure record (#238, 2026-10-01) — the bucket-C test-only wrappers become `#[cfg(test)]`, in place

**The ticket.** [#238](https://github.com/yusiwen/minfer/issues/238) is T1 of the `allow(dead_code)` census. Its bucket **C** is "dead in the
production `--features cuda` build, reached from test code, with at least one test caller **outside**
the item's own module subtree". Those items cannot move into their module's `tests.rs` — that is
bucket **B** / T2 ([#239](https://github.com/yusiwen/minfer/issues/239)) — so the retirement is an explicit test-only scope at the item:

```diff
-    #[cfg_attr(not(test), allow(dead_code))]
-    pub fn foo(…)
+    /// Test-only (#238): driven by `…::tests::…`; `#[cfg(test)]` keeps it out of production builds.
+    #[cfg(test)]
+    pub(crate) fn foo(…)
```

`#[cfg(test)]` is strictly stronger than `allow(dead_code)`: the item **does not exist** in a
non-test build, so a later cleanup cannot leave a production-looking orphan behind (the #218 shape
[GATE-CONTRACT.md](./GATE-CONTRACT.md) §1 asks about). The item stays where it is because its callers are in other
modules: moving the surface into a different module's test file would relocate the boundary without
relocating the caller — the same reasoning [#236](https://github.com/yusiwen/minfer/issues/236)'s record gives for `kv_cell_of`.

**Reconciliation: the three tallies were 26, 71 and 56.** Re-derived from the raw captures
(`minfer-allow-census/cuda.jsonl`, `tests_cuda.jsonl`), taking **every** span of every `dead_code`
diagnostic whose file starts with `src/`:

| unit | count |
|---|---|
| distinct `(file, line)` dead sites, `cargo check --release --features cuda` | **314** (99 diagnostics) |
| … still dead under `--tests --features cuda` → bucket **A** ([#242](https://github.com/yusiwen/minfer/issues/242)) | **142** |
| … test-reachable (314 − 142) → 68 sites on B items + 71 on C items + 33 `impl`/`mod` header spans that rustc reports without an item identity | **172** |

At census **item** level (`census.json`, keyed by `(file, line, name)`) the cuda build has **260**
dead items: A **121** (89 code / 32 shape), B **68** (58 / 10), C **71** (56 code / 15 shape). The
three numbers the tracker carried are three different units, not three measurements:

- **26** — the number of `dead_code` *diagnostics* whose **primary** span is a bucket-C item. The
  earlier analysis took each diagnostic's primary span only, so a grouped diagnostic collapsed a
  whole dead `impl` into one row. This record reproduces 26 exactly with that rule.
- **71** — the item-level bucket-C count (`cuda_C` in `census.json`), i.e. 56 code + 15 shape.
- **56** — the **code**-only C set, which is what T1 lists. `tickets/T1.md` carries exactly 56 item
  bullets (a naive `grep -c '^- \`` reads 57: one Acceptance bullet also starts with a backtick).

**Per-item verification, and what a re-read changed.** Every item was read at its call site before
it was touched; a name match is not a call site. The census's six recorded collision overrides
(`snapshot`, `plan`, `cuda_backend::new`, `submit`, `sample`, `source`) all held. Six more were found
here, and one of them changed a recorded call list without changing a bucket:

- `CudaBackend::kv_format` vs `GraphAllocator::kv_format` / `ModelDef::kv_format`: the census's three
  recorded test callers for `graph/cuda_backend.rs:314` are the other two (the registry hook is
  `fn(&GraphAllocator)`, `registry.rs:241`). The real caller is `graph/alloc/tests.rs:1526`
  (`alloc.cuda().unwrap().kv_format()`), so the item stays C.
- `optiming::record` vs `TimingSink::record`: the census's `optiming/tests.rs:142` caller is the
  sink's method; the free function's one caller is `graph/scheduler/tests.rs:208`.
- `GgufContext::get_val_bool` vs `GgufKv::get_val_bool`: the four non-test matches (`gguf.rs:770`,
  `:1752`, `:1849`, `main.rs:2398`) are the inner type's method; the context's is test-only.
- `CpuBackend::pool_len` vs `BackendTrait::pool_len`: `GraphAllocator::pool_len_of` calls the trait
  method; the inherent method's only caller is `n_cpu_buffers`.
- `Tensor::new` vs `Vec::new()` / `String::new()` / `OnceLock::new()`: the census's ~200 "other
  callers" are all other types; the one real caller is `graph/builder/tests.rs:6`.
- `GraphAllocator::supports` vs `KvFormat::supports`: `kvformat::resolve` calls the format's.

**No item moved from C to A or C to B**, but four of the 56 are not gated here, each with its reason
written at the item (or, for the two (a) verdicts, in the tracker):

| item | verdict |
|---|---|
| `CudaState::matmul_f32_ptr` (`cuda.rs:3888`) | **blocked**: its caller `CudaState::quant_matmul_f32_on_gpu` (`cuda.rs:5325`) is bucket **A** and still compiled into production (part of [#240](https://github.com/yusiwen/minfer/issues/240)'s legacy wrapper layer). Gating the callee would leave the production CUDA build naming a function that does not exist. Deferred to [#240](https://github.com/yusiwen/minfer/issues/240)/[#242](https://github.com/yusiwen/minfer/issues/242). |
| `CpuBackend::pool_len` (`cpu_backend.rs:113`) | **blocked**: its only caller `GraphAllocator::n_cpu_buffers` (`alloc.rs:2358`) is bucket **B** ([#239](https://github.com/yusiwen/minfer/issues/239)) and stays compiled, with its own `allow`, until T2 moves it. Deferred to [#239](https://github.com/yusiwen/minfer/issues/239). |
| `ModelDef::as_any` (`models/mod.rs:211`) | **(a) question**: the doc says "downcast helper for the graph path's weight registration"; `models::weight_reg` never downcasts (its decision is a pure `(ttype, geometry)` predicate) and every `as_any()` call site under `src/` is in a `#[cfg(test)]` module. A production-looking trait method whose documented caller does not exist — reported, not silenced. |
| `ModelDef::offload` (`models/mod.rs:294`) | **(a) question**: the doc (and the qwen2/qwen3 module docs) says it "hands the plan to the graph builder", but the builder reads `model.offload.plan` (`models/qwen2/graph.rs:556`) and nothing outside tests calls the accessor. Same class, same disposition. |

Both (a) items are defaulted/required **trait** methods, so `#[cfg(test)]` would also change the
trait's public surface and (for `as_any`) both impls — a decision for [#244](https://github.com/yusiwen/minfer/issues/244)'s shape/API class
rather than a mechanical retirement.

**Stale docs corrected at the item.** `CudaState::stream_is_capturing` and its FFI declaration
`cudaStreamIsCapturing` both claimed a "registration path's refusal / inventory" consumer. No such
caller exists: the production capture bookkeeping is the per-instance `CudaBackend::capturing` field,
and the pair's only caller is the #188 probe in `graph/cuda_backend/tests.rs`. Their docs now say so.

**Bar named before measuring.** No `#[test]` is added or removed, and no assertion moves — the tests
keep their own assertions and only the resolution of the item changes. The bar: the non-test build
stays warning-free with and without `--features cuda` (the crate carries
`#![cfg_attr(not(test), deny(warnings))]`, so an exposed callee or a broken call site is a hard
failure), the suite counts stay where the recorded rows put them, and one mutation per module group
must be visible to the test that drives the scoped item.

**Mutation evidence (rule 3), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01.** One representative
item per module group; each mutation applied, the named test run with `--exact`, then reverted and
`git status --porcelain <file>` empty. All 16 went red:

| group (item) | mutation | test (all FAILED) | first failing line |
|---|---|---|---|
| conversation (`Conversation::start`) | `prefill_tokens = toks.len() + 1` | `conversation::tests::first_turn_full_render_and_eog` | `tests.rs:209`: left 22 / right 21 |
| cuda (`format_of`) | `KV_LAYOUT_F16 => F32` | `cuda::kv_dtype_tests::the_layout_tag_is_the_format_discriminant` | `kv_dtype_tests.rs:14`: left F32 / right F16 |
| gguf (`get_arr_n`) | `get_ne() + 1` | `gguf_write::tests::every_metadata_type_round_trips_through_the_parser` | `tests.rs:58`: left 3 / right 2 |
| grammar (`accepts`) | `Err(_) => true` | `grammar::tests::gbnf_literals_classes_and_dot` | `tests.rs:83` |
| graph/alloc (`get_buffer`) | `if true { return None }` | `graph::alloc::tests::fill_and_read_input` | `tests.rs:356` |
| graph/builder (`swiglu`) | inputs `&[up, gate]` | `graph::builder::tests::swiglu_builder_and_meta` | `tests.rs:91`: left `[1, 0]` / right `[0, 1]` |
| graph/cache (`stats`) | `builds + 1` | `graph::cache::tests::switching_between_cached_graphs_re_maps_instead_of_rebuilding` | `tests.rs:96`: left (3, 0) / right (2, 0) |
| graph/copystats (`delta`) | `copies: self.copies` | `graph::copystats::tests::the_two_phases_are_counted_separately_and_the_delta_is_exact` | `tests.rs:28`: left copies 5 / right 2 |
| graph/cpu_backend (`causal_span`) | `(0, p)` instead of `(0, p + 1)` | `graph::cuda_backend::tests::cuda_rope_kv_attn_roundtrip` (**CUDA**) | `graph/cuda_backend/tests.rs:689` |
| graph/cuda_backend (`stream_sync_count`) | return `0` | `graph::cuda_backend::tests::stream_sync_counts_are_per_backend_not_process_wide` (**CUDA**) | `tests.rs:8317`: left 0 / right 1 |
| graph/kvcache (`GraphAllocator::kv_clear_identity`) | no-op | `graph::scheduler::tests::a_non_identity_kv_mapping_is_refused` | `tests.rs:85` |
| graph/kvformat (`pack_q8_0_cell`) | `dst *= 2.0` after packing | `graph::kvformat::tests::a_packed_cell_round_trips_within_the_q8_0_block_error` | `tests.rs:172` |
| graph/registry (`is_unfiltered`) | `all` → `any` | `graph::registry::tests::the_name_surface_fences_devices_and_keeps_cpu` | `tests.rs:186` |
| optiming (`record`) | drop the record | `graph::scheduler::tests::a_concurrent_graph_load_cannot_move_a_private_sink` | `tests.rs:242`: left 256 / right 257 |
| tensor (`Tensor::from_data`) | zero the payload | `graph::op_matrix::matrix_cases_match_their_reference` | `op_matrix.rs:800` |
| testfail (`checked`) | `map_or(0, |_| 1)` | `graph::scheduler::tests::the_execute_chokepoint_is_observable` | `tests.rs:131`: left 1 / right 2 |

The full suite run is the baseline these deltas are read against (below). The CUDA rows ran through
the device build; the rest are CPU rows.

**Counts (rule 5).** No `#[test]` was added or removed, so no row moves and **no CUDA row is
re-measured**: `cargo test --release`, box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01 → **481 / 0 /
36** unit + **10 / 0 / 6** integration (the recorded CPU row, unchanged);
`bash scripts/cuda_test.sh`, same box, 2026-10-01 → **566 / 0 / 42** (the recorded CUDA row,
unchanged; the suite was run to confirm the device build still links the `#[cfg(test)]` items it now
gates). `cargo check --release` and `cargo check --release --features cuda` both exit 0 with no rustc
diagnostic (the one CUDA `warning:` line is `build.rs`'s pre-existing `cargo:warning=` target list);
`cargo fmt --all --check` clean; `python3 scripts/check_status.py --check` exits 0.

**Docs.** The [#235](https://github.com/yusiwen/minfer/issues/235) family issue gets a comment recording which of its four wrappers this ticket
closed (`kv_clear_identity`, `kv_shift`, `kv_n_used` — `kv_cells_for` is bucket A and stays with
[#242](https://github.com/yusiwen/minfer/issues/242)); [#238](https://github.com/yusiwen/minfer/issues/238) gets the reconciliation, the four deferrals and the patch that brings its body in
line; [#239](https://github.com/yusiwen/minfer/issues/239)/[#240](https://github.com/yusiwen/minfer/issues/240)/[#242](https://github.com/yusiwen/minfer/issues/242) keep their rows.

**Limits.** (1) The macOS-only modules (`metal.rs`, the Metal half of `graph/metal_backend.rs`) are
not compiled on Linux, so their annotations are unchanged and unjudged — the macOS CI job is the only
gate (`src/models/mod.rs`'s `as_any` is reached from `metal/mmap_align_test.rs`, which is why it
appears in this bucket at all). (2) The item-level classification still rests on textual test-caller
resolution for the *module* of a caller (liveness is rustc's); the six new collisions above are the
residue, each now verified by reading. (3) `Tensor::new`'s only consumer is the `f32_tensor` helper
in `graph/builder/tests.rs`, whose assertions read fields the helper sets itself (name, shape) — no
mutation of `new`'s body is observable through it, so the tensor group's mutation is on
`Tensor::from_data` (same `impl Tensor`, same file), and that is a real gap in the helper's coverage,
recorded rather than papered over. (4) `--features debug_dump` was not built by the census or here.

#### Test-infrastructure record (#239, 2026-10-01) — the same-module test-only items move into their module's `tests.rs`

**The ticket.** [#239](https://github.com/yusiwen/minfer/issues/239) is T2 of the `allow(dead_code)` census. Its bucket **B** is
"dead in the production `--features cuda` build, reached from test code, and **every** test caller
sits inside the item's own module subtree". That is what makes the retirement a *move* rather than
#238's in-place gate: an inherent method goes into a `#[cfg(test)] impl Type` block in that module's
`tests.rs` (an inherent impl may live in a child module — the cuda build proves it), a free
function/const/FFI declaration goes in as a `#[cfg(test)]` item there. After the move nothing in a
non-test build knows the item exists, which is strictly stronger than `#[cfg(test)] pub(crate)`
in place: there is no production-file line left for a later cleanup to orphan.

**Reconciliation: the ticket's "26" is not a B unit.** Re-derived from the raw captures
(`minfer-allow-census/cuda.jsonl`, `tests_cuda.jsonl`) with the same rule T1 used — **every** span of
every `dead_code` diagnostic whose file starts with `src/`:

| unit | count |
|---|---|
| distinct `(file, line)` dead sites, `cargo check --release --features cuda` | **314** (99 diagnostics) |
| … still dead under `--tests --features cuda` → bucket **A** ([#242](https://github.com/yusiwen/minfer/issues/242)) | **142** |
| … test-reachable (314 − 142) → **68** sites on B items + **71** on C items + **33** `impl`/`mod` header spans rustc reports without an item identity | **172** |

At census **item** level (`census.json`, keyed by `(file, line, name)`) the cuda build has **260** dead
items: A **121** (89 code / 32 shape), **B 68 (58 / 10)**, C **71** (56 / 15). The three numbers the
tracker circulated are therefore:

- **68** — the item-level bucket-B count (`cuda_B`), i.e. 58 code + 10 shape.
- **58** — the **code**-only B set, which is T2's list. `tickets/T2.md` carries exactly 58 item
  bullets (a naive `grep -c '^- \`'` reads 59 because one Method/Sentence line also matches);
  `census.json` and the list agree name for name.
- **26 — does not reproduce for B, and is not a third unit.** Re-deriving it the only way it can be
  read (*how many of the 99 diagnostics have their **primary** span on a bucket-B item*) gives **45**
  (42 code + 3 shape), not 26. **26** reproduces *exactly* as the number of diagnostics whose primary
  span is a bucket-**C** item — which is T1's headline unit ([#238](https://github.com/yusiwen/minfer/issues/238)). The
  ticket body's "26 (diagnostics whose primary span is a bucket item)" is a mis-attribution of T1's
  number to T2, and its body is patched to say so. (For completeness, the README's *collapsed* B = 28
  reproduces as **27** diagnostics whose every `src/` span is a B item; the one-diagnostic difference
  is a diagnostic that also carries an `impl` header span.)

**Per-item verification, and what a re-read changed.** Every item was read at its call site before it
was touched. The census's collision overrides all held for these items (`snapshot`, `plan`,
`cuda_backend::new`, `submit`, `sample` were checked again: the recorded cross-module matches really
are other items — `TimingSink::snapshot`, `OffloadRequest::plan`, `Vec::new`, the Metal
`CommandBuffer::submit`, `server::metrics`' own `sample`). Five items did **not** survive as a plain
move:

| item | verdict |
|---|---|
| `KvCache::set_owner` (`kvcache.rs:322`) | **(a) question**: its doc said "C1 uses it from `own_range`", but `own_range` (`kvcache.rs:517`) writes `l.owner[cell] = seq` inline and never calls it. A helper whose documented production caller does not exist is the #218 shape — *reported, not moved* (the annotation stays), for [#244](https://github.com/yusiwen/minfer/issues/244). |
| `OffloadPlan::all_on_device` (`offload.rs:44`) | **(a) question**: its doc said it is "what an unset request resolves to when a device is available", but `OffloadRequest::plan` (`offload.rs:187`) and `resolve` (`offload.rs:232`) build `OffloadPlan { gpu_layers, n_layers }` inline (they must — they clamp first). Same class, same disposition. |
| `StreamScratch::slot` (`cuda.rs:1300`) | **blocked**: bucket B by the census (its only *test* caller is `cuda::d35_probe_tests`), but the bucket-**A** legacy wrappers `upload_hidden` / `upload_positions` / `download_logits` / `get_positions_buf` still call it **and are still compiled**, so it cannot leave the production file. Deferred to [#240](https://github.com/yusiwen/minfer/issues/240) / [#242](https://github.com/yusiwen/minfer/issues/242), with the reason at the item. |
| `device_entry.rs`'s 4 items (`DEVICE_ENTRY`, `DeviceHolder`, `DeviceEntry`, `enter`) | **blocked**: `enter` is still called by the bucket-**A** `CudaState::layer_gpu` (`cuda.rs:7051`), which is compiled in the cuda build. Deferred to [#241](https://github.com/yusiwen/minfer/issues/241), which deletes the guard. |
| `CudaBackend::new` (`cuda_backend.rs:200`) | **reclassified B→C, and still moved**: the census calls it B because `is_test_path` only recognises `tests.rs` / `*_tests.rs`, but it also has a cross-module test caller, `graph::op_matrix` (a `#[cfg(test)] mod`). An inherent method's impl may live in a child module, so moving it into `cuda_backend/tests.rs` with `pub(crate)` keeps `op_matrix` compiling while production loses the constructor — verified by the `--features cuda` check and the device suite. |

Three further stale doc claims were found on items that **were** moved (their own docs do not assert
a current consumer, so they are not (a) items) and are recorded at the item and in the tracker:
`GraphAllocator::set_memory_budget`'s "a future offload policy uses it" (the E5 S2 policy landed and
resolves its budget through `MINFER_GPU_MEM` + `allocplan::weight_budget`), `KvCache::note_written`'s
`// E2 surface` note (E2 records written extents through `own_positions` /
`fill_batch_inputs`), and `allocplan.rs`'s module doc ("the plan is checked against a per-backend
budget before that loop runs" — the gate is `alloc_in_pool`, not `AllocPlan::plan`). The module doc
keeps its wording and gains an (a) note rather than being quietly rewritten.

**The moves: 51 items, 18 module groups.** One item per line stayed free; the map is

| module (`tests.rs`) | moved items | visibility |
|---|---|---|
| `conversation` | `Conversation::snapshot` | private |
| `cuda` (**new `src/cuda/tests.rs`**) | 13 test-only FFI declarations (`cuda_test_latch_oversized_smem`, `minfer_site_fail_*` ×8, `minfer_site_hist_{len,site,name,msg}`), `latched_api_error_count`, `CudaState::take_last_error` | `pub(super)` — the callers are sibling `*_tests` modules under `cuda` |
| `device_tier` | `IDENTITY_BATCH_BOUND`, `mmvq_batch_limit`, `mmvq_cap` | private (the const is a standalone item, not a field/variant, so it can move; `mmvq_cap` was its only production-side reader) |
| `grammar` | `GrammarState::pending_bytes` | private |
| `graph/alloc` | `set_memory_budget`, `n_cpu_buffers`, `n_mapped_buffers` | private |
| `graph/allocplan` | `AllocPlan`, `plan`, `live_peak`, `DeviceMemory::free_bytes` | private |
| `graph/cache` | `cached_graphs` | private |
| `graph/cpu_backend` | `weight` | private |
| `graph/cuda_backend` | `new`, `exec_ids` + `elems` (the `#[cfg(test)]` shim and its helper) | `pub(crate)` on `new` (the `op_matrix` caller), private otherwise |
| `graph/kvcache` | `own_prefix`, `note_written`, `after_shift`, `seq_range`, `private_written` | private |
| `graph/mod` | `DType::size` | private |
| `graph/registry` | `names` | private |
| `graph/scheduler` | `with_timing` | private |
| `sampler` | `apply_repetition_penalty`, `sample_with_penalties`, `sample` | private (`sample_with_penalties` has no test caller of its own; it is reached only through `sample`, so it moved with it) |
| `server/batch` | `idle_slots`, `submit`, `prefill_stats`, `prefill_fed`, `interleaved_ticks` | private |
| `server/metrics` | `completion_tokens_per_second` | private |
| `template` | `PyValue::to_json` | private |
| `tokenizer` | `PreTokenizer::gguf_name`, `Tokenizer::decode` | private |

The visibility rule: private when the only caller is that one `tests.rs`; `pub(super)` when the
callers are several test files under the same module (`cuda`); `pub(crate)` when a `#[cfg(test)]`
module elsewhere in the crate calls it (`CudaBackend::new` from `graph::op_matrix`). Each moved item
carries a doc note naming the test that drives it (machine-checked against the named test's module
and file) and the sentence "Test-only (#239)".

**T1's `pool_len` deferral is closed.** With `GraphAllocator::n_cpu_buffers` now in
`graph/alloc/tests.rs`, `CpuBackend::pool_len` has no production caller left, so it became
`#[cfg(test)] pub(in crate::graph)` — the narrowest spelling that reaches `graph::alloc::tests`
(`pub(crate)` would also work; the narrower one is preferred). [#238](https://github.com/yusiwen/minfer/issues/238) gets a comment saying so.
T1's other deferral, `CudaState::matmul_f32_ptr`, stays with [#240](https://github.com/yusiwen/minfer/issues/240)/[#242](https://github.com/yusiwen/minfer/issues/242): its caller
`quant_matmul_f32_on_gpu` is still compiled.

**Bar named before measuring.** No `#[test]` is added or removed and no assertion changes — the moved
items are byte-identical bodies, so the claim each test makes is unchanged by construction. The bar:
the non-test build stays warning-free with and without `--features cuda` (the crate carries
`#![cfg_attr(not(test), deny(warnings))]`), the recorded suite counts do not move, and one mutation
per module group must be visible to the test that drives the moved item.

**Mutation evidence (rule 3), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01.** One representative
item per module group that actually moved; each mutation applied, the named test run, then reverted
and the file verified byte-identical. **All 18 went red**:

| group (item) | mutation | test (all FAILED) | first failing line |
|---|---|---|---|
| conversation (`Conversation::snapshot`) | `messages: Vec::new()` | `conversation::tests::a_resumed_snapshot_prefills_nothing_and_continues_alike` | `tests.rs:301` |
| cuda (`latched_api_error_count`) | return `0` (`MINFER_TEST_LATCH_ERROR=1`) | `cuda::issue145_tests::cuda_sync_surfaces_a_latched_error_as_latched` (**CUDA**) | `issue145_tests.rs:230` |
| device_tier (`mmvq_cap`) | drop the `.min(IDENTITY_BATCH_BOUND)` | `device_tier::tests::caps_clamp_to_the_identity_bound` | `tests.rs:85` |
| grammar (`pending_bytes`) | return `0` | `grammar::tests::token_advancement_handles_partial_utf8` | `tests.rs:514` |
| graph/alloc (`pool_len`, through the moved `n_cpu_buffers`) | return `0` | `graph::alloc::tests::a_rebuild_remaps_instead_of_reallocating` | `tests.rs:1607` |
| graph/allocplan (`free_bytes`) | `Some(*free + 1)` | `graph::allocplan::tests::a_reported_free_read_keeps_the_three_quarters_default` | `tests.rs:51` |
| graph/cache (`cached_graphs`) | return `0` | `graph::cache::tests::switching_between_cached_graphs_re_maps_instead_of_rebuilding` | `tests.rs:97` |
| graph/cpu_backend (`weight`) | return `None` | `graph::cpu_backend::tests::f32_matmul_nt2_token_major` | `tests.rs:249` |
| graph/cuda_backend (`new`) | return `None` | `graph::cuda_backend::tests::cuda_pool_roundtrip` (**CUDA**) | `tests.rs:108` |
| graph/kvcache (`seq_range`) | `Ok(None)` | `graph::kvcache::tests::two_sequences_resolve_to_disjoint_windows` | `tests.rs:151` |
| graph/mod (`DType::size`) | `F32 => 8` | `graph::tests::dtype_size` | `tests.rs:101` |
| graph/registry (`names`) | return `["metal", "cpu", "cuda"]` | `graph::registry::tests::names_resolve_and_unknown_names_are_refused` | `tests.rs:111` |
| graph/scheduler (`with_timing`) | force `TimingMode::Global` | `graph::scheduler::tests::a_concurrent_graph_load_cannot_move_a_private_sink` | `tests.rs:230` |
| sampler (`apply_repetition_penalty`) | no-op | `sampler::tests::test_repeat_penalty_reduces_repeated` | `tests.rs:18` |
| server/batch (`idle_slots`) | return `0` | `server::batch::tests::a_failed_decode_forward_answers_a_single_run_and_releases_its_slot` | `tests.rs:1955` |
| server/metrics (`completion_tokens_per_second`) | return `0.0` | `server::metrics::tests::token_counters_and_the_trailing_rate` | `tests.rs:277` |
| template (`PyValue::to_json`) | encode an int as a string | `template::tests::python_str_methods_match_cpython` | `tests.rs:263` |
| tokenizer (`Tokenizer::decode`) | return `String::new()` | `tokenizer::tests::decode_bytes_reverses_byte_encoding` | `tests.rs:65` |

The `offload` group has no row: its only bucket-B item, `all_on_device`, is an (a) finding and was
not moved.

**Two coverage gaps, recorded rather than papered over.** (1) `allocplan::live_peak`'s result is not
observable at all: `AllocPlan::plan` computes `peak` in its main loop (that loop never subtracts, so
`peak == reserved_bytes` on exit) and then `peak.max(live_peak(…))` cannot exceed it — mutating
`live += class_bytes(classes[i])` to `live += 0` left `the_live_peak_is_not_the_reserved_total`
**green**. The group's mutation is therefore on a sibling in the same move, `DeviceMemory::free_bytes`.
This is a real gap in `live_peak`'s coverage and also a hint that the function is dead arithmetic —
for [#244](https://github.com/yusiwen/minfer/issues/244) to decide. (2) `pool_len`'s *value* is only ever compared with itself in
`a_rebuild_remaps_instead_of_reallocating`, so a constant offset (`+1`) is invisible; the observable
mutation is `0`, which fails that test's `buffers > 0` start assertion.

**Counts (rule 5).** No `#[test]` was added or removed, so no row moves: `cargo test --release`, box
`dgxspark (aarch64, GB10 sm_121)`, 2026-10-01 → **481 / 0 / 36** unit + **10 / 0 / 6** integration (the
recorded CPU row); `bash scripts/cuda_test.sh`, same box, 2026-10-01 → **566 / 0 / 42** (the recorded
CUDA row — run to confirm the device build still resolves the moved FFI declarations, the moved
`CudaBackend::new` and the `#[cfg(test)] extern "C"` block the new `cuda/tests.rs` introduces).
`cargo check --release` and `cargo check --release --features cuda` both exit 0 with no rustc
diagnostic (the one CUDA `warning:` line is `build.rs`'s pre-existing `cargo:warning=` target list);
`cargo fmt --all --check` clean; `check_status.py --check`, `check_docs_links.py` and
`check_source_layout.py` all clean (the new `src/cuda/tests.rs` is named by its `#[cfg(test)] mod
tests;`).

**Docs.** [#239](https://github.com/yusiwen/minfer/issues/239) gets the reconciliation, the five non-moves and the mutation table as a comment and
its body is patched (the 26 → 45/68/58 reconciliation); [#238](https://github.com/yusiwen/minfer/issues/238) gets the `pool_len` closure;
[#240](https://github.com/yusiwen/minfer/issues/240)/[#241](https://github.com/yusiwen/minfer/issues/241)/[#242](https://github.com/yusiwen/minfer/issues/242) keep the two blocked groups; [#244](https://github.com/yusiwen/minfer/issues/244) gets the two (a)
items and the three stale doc claims.

**Limits.** (1) The macOS-only modules (`metal.rs`, the Metal half of `graph/metal_backend.rs`) are
not compiled on Linux and were not touched — the macOS CI job is their only gate. (2) The item-level
B/C classification rests on textual resolution of a caller's *module* (liveness is rustc's); the one
residue found here (`CudaBackend::new`, called from the `#[cfg(test)] mod op_matrix`) is exactly the
limit, and it was handled by moving with `pub(crate)`. (3) `--features debug_dump` was not built.
(4) The two (a) verdicts are *questions*, not findings of fact about intent: in both cases the doc may
simply be stale rather than a caller having been deleted, and either resolution (call it, or delete
it, or say test-only) belongs to [#244](https://github.com/yusiwen/minfer/issues/244), not here.

#### Test-infrastructure record (#240/#241, 2026-10-01) — the dead legacy `CudaState` wrapper layer and the `device_entry` guard are deleted

**The tickets.** [#240](https://github.com/yusiwen/minfer/issues/240) is T3a of the `allow(dead_code)` census: delete the
legacy `CudaState` wrapper layer — 23 methods and 18 backing fields that no build configuration calls.
[#241](https://github.com/yusiwen/minfer/issues/241) is T3b: the legacy `device_entry` guard is unreachable and #185's
remaining token should be retired. They landed as **one PR** because they are compile-coupled, not
because they were convenient to batch.

**Why one PR: the coupling is a compile error, not a preference.** `CudaState::layer_gpu` is the
**only** caller of `has_weight`, `debug_sync`, `kv_ensure_layer`, `get_positions_buf`, `matmul_on_gpu`,
`quant_matmul_q8` and `quant_matmul_f32_on_gpu`, and the only reader of fifteen `buf_*`/`kv_*` fields.
Deleting them while `layer_gpu` stays is `E0599`. The reverse is also true: `src/device_entry.rs` opened
with `#![cfg_attr(not(feature = "cuda"), allow(dead_code))]`, so in the cuda build there is no allow —
with `layer_gpu` gone, `enter`/`DeviceEntry`/`DEVICE_ENTRY` have no caller and `deny(warnings)` fails the
build. A partial #240 that "skips `layer_gpu`" can therefore only delete the sixteen uncoupled leaf
items and cannot close either of the deferrals T1/T2 left here; that split was rejected after the
coupling was traced, and the user approved deleting the guard, whose only caller is itself dead.

**Per-item disposition.** Every item was read at its definition, its cross-module matches checked, and
its history read (`git log -S`) before it was touched.

| item | disposition |
|---|---|
| `stream_lock`, `clear_launch_failure`, `upload_hidden`, `download_hidden`, `upload_positions`, `init_kv_cache`, `get_kv_size`, `download_logits`, `graph_available`, `graph_end_capture`, `graph_launch`, `quant_matmul_f32_batch`, `quant_matmul_f32`, `output_norm_gpu` | **deleted** (uncoupled leaf; no caller in any build) |
| `buf_logits` (readers: `download_logits`/`quant_matmul_f32`/`output_norm_gpu`), `decode_graph_exec` (`graph_available`/`graph_end_capture`/`graph_launch`) | **deleted** with their initializers |
| `layer_gpu` (bucket A; dead in every build) | **deleted** ([#241](https://github.com/yusiwen/minfer/issues/241)'s item, absorbed by the coupling above) |
| `has_weight`, `debug_sync`, `kv_ensure_layer`, `get_positions_buf`, `matmul_on_gpu`, `quant_matmul_q8`, `quant_matmul_f32_on_gpu` | **deleted** once `layer_gpu` went (it was their only caller) |
| `buf_hidden`, `buf_bn`, `buf_bq`, `buf_bk`, `buf_bv`, `buf_ba`, `buf_bf`, `buf_bg`, `buf_q8_bn`, `buf_q8_ba`, `buf_positions`, `kv_k`, `kv_v`, `kv_size` | **deleted** with their initializers (only `layer_gpu` read them) |
| `tier` | **deleted**: never read from `self`; the name-level matches are `device_tier::Selection`'s own `tier` in the two select arms (the collision the census note flags). Its doc said "Direct consumers arrive with the batch-cap activation (plan §14 R8)", which is the pattern the plan's own A7 note rejects — "a future feature will need it" is not enough. The effective gate production reads is `tier_mmq`; the R8 work re-adds the field from the selection it already computes. |
| `cc` | **kept, reported**: read by `#[cfg(test)] CudaState::cc()` (driven from `graph/cuda_backend/tests.rs`), so it is test-reachable and not a deletion. Its non-test-dead annotation is [#243](https://github.com/yusiwen/minfer/issues/243)'s tightening, the same shape T1/T2 handled for their items. |
| `stream_wait_event` / the FFI declaration `cudaStreamWaitEvent` | **kept**: the device→device staging copy it is the mechanism for. [#138](https://github.com/yusiwen/minfer/issues/138) landed 2026-10-04 (the F5 S2 record) **without creating a reachable caller**: the only pair that could express a device destination needs two device backends, `copy_across` early-returns on a same-backend pair, and Metal declines phase A. So the item is still dead in every compilable configuration and the census brief requires it be named. |
| `src/device_entry.rs` (`DEVICE_ENTRY`, `DeviceHolder`, `DeviceEntry`, `enter`) + its `mod` declaration and its test file | **deleted**: the guard exists for a caller that does not exist (#188 had already narrowed it to `layer_gpu`). |
| `cuda_debug_enabled` + the `CUDA_DEBUG` `OnceLock` | **newly dead**, identified here and deleted in the same commit: `debug_sync` was its only reader. |

Two doc claims were read because a doc that names a production caller is the #218 *lost-caller* shape:
`clear_launch_failure`'s `Err`-arm claim (`graph/cuda_backend.rs::execute_node` drains
`take_launch_failure()` on both arms instead — the doc was stale) and `init_kv_cache`'s "must be called
before the first forward pass" (superseded by the graph path, which owns KV through
`GraphAllocator::kv_pair`). Both items are deleted; no lost caller was found, so nothing is reported as
(a) for these two. No `extern "C"`/`#[no_mangle]` item is in the set and none of the names is a symbol
in `src/cuda_kernels.cu` or `build.rs`; nothing `memcpy`s a `CudaState`.

**The two deferrals this closes.** `StreamScratch::slot` ([#239](https://github.com/yusiwen/minfer/issues/239)'s deferral, pinned by the
four bucket-A wrappers `upload_hidden`/`upload_positions`/`download_logits`/`get_positions_buf`) now
lives in `src/cuda/tests.rs` as `pub(super)`, the visibility the rest of that file uses for the sibling
`cuda::*_tests` modules. `CudaState::matmul_f32_ptr` ([#238](https://github.com/yusiwen/minfer/issues/238)'s deferral, pinned by `quant_matmul_f32_on_gpu`)
is now `#[cfg(test)] pub(crate)` — its callers are the device gates in `graph::cuda_backend::tests`, a
`#[cfg(test)]` module in another file, so T1's in-place form (not T2's move) is the correct one.

**The one intentional coverage decrease.** `src/device_entry/tests.rs` held a single test,
`device_entry::tests::the_legacy_unbound_path_is_exclusive_across_threads_and_re_entrant_on_one`, which
asserted the guard's cross-thread exclusion and per-thread re-entrancy with mutation evidence. It is
deleted with the guard, **by decision** (recorded on [#241](https://github.com/yusiwen/minfer/issues/241)): the guard's only caller is
dead, so there is no engine property left for the assertions to be about. This is the only place a test
is removed in this ticket, and no other test loses an assertion. It is why the CPU rows move 481 → 480
(aarch64) and 479 → 478 (CI), and the CUDA rows 566 → 565.

**Evidence form: the census re-run (this ticket's version of mutation evidence).** There is no gate to
mutate — the deleted code has no caller, so no test observes it, and "break the implementation and watch
a gate go red" has nothing to break. For a deletion ticket the observable claim is instead "the dead set
shrank by exactly what was removed, and nothing else became dead", and that is *measured*. Method,
identical on both sides: `strip.py` in a scratch worktree (every `allow(dead_code)` replaced by a
`//STRIPPED` comment, line numbers preserved), then
`cargo check --release --features cuda --message-format=json` (the maximal configuration on Linux,
because `src/cuda.rs` is `#[cfg(feature = "cuda")] mod`), counting **every** `src/` span of every
`dead_code` diagnostic. Liveness is rustc's, never a name grep.

| tree | dead `src/` sites | dead items | `src/cuda.rs` sites |
|---|---|---|---|
| `e9bf4c5` — the durable census' recorded baseline (reproduced here from `minfer-allow-census/cuda.jsonl`) | **314** | 260 | 53 |
| `79c9837` — this branch's parent | **191** | **156** | 53 |
| `dc550a9` — after the uncoupled deletions | **175** | **140** | 37 |
| `2fbaf00` — the final tree | **143** | **109** | 7 |

The drop from the parent is **48 sites / 47 items**, and the before/after item-name diff has an **empty
"new" side**: zero newly-dead items. The 314 → 191 drop between the recorded baseline and this branch's
parent is **not** this ticket: [#238](https://github.com/yusiwen/minfer/issues/238)/[#239](https://github.com/yusiwen/minfer/issues/239) turned bucket-C items into `#[cfg(test)]` and moved bucket-B items
into `tests.rs`, and an item that is not compiled into a non-test build leaves the dead set entirely.
Truncation check: both captures re-run under `RUSTFLAGS=--cap-lints=warn` differ from the
`deny(warnings)` captures by **symmetric difference 0** (cpu and cuda), so `deny(warnings)` did not
truncate the count.

The seven `src/cuda.rs` sites left are outside this ticket's list and are **named, not silently left**:
`stream_wait_event` + its FFI declaration `cudaStreamWaitEvent` ([#138](https://github.com/yusiwen/minfer/issues/138)), `cc`
([#243](https://github.com/yusiwen/minfer/issues/243)), and `cudaGetErrorString`/`cuda_error_string`, `minfer_site_hist_reset`,
`stream_sync_count` — dead before this ticket too (they are in the baseline set, not the diff), so they
belong to [#242](https://github.com/yusiwen/minfer/issues/242)/[#243](https://github.com/yusiwen/minfer/issues/243)/[#244](https://github.com/yusiwen/minfer/issues/244)'s sweep, not to #240.

**Bar named before measuring.** No `#[test]` is added or removed **except** the guard's own, removed by
decision (above), so no assertion outside it changes. The bar: the non-test build stays warning-free
with and without `--features cuda` and without adding a single `allow(dead_code)`; the suite rows move
by exactly the one removed test (480/0/36 + 10/0/6 CPU, 565/0/42 CUDA); and the census dead-site count
drops by at least the number of items deleted, with zero newly-dead sites.

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01.**
`cargo test --release` → **480 / 0 / 36** unit + **10 / 0 / 6** integration.
`bash scripts/cuda_test.sh` → **565 / 0 / 42**.
`compute-sanitizer --tool memcheck --target-processes all <test binary> --test-threads=1` →
**0 API errors** over 565 / 0 / 42.
`cargo check --release`, `cargo check --release --features cuda` and
`cargo check --release --tests --features cuda` all exit 0; the only `warning:` line is `build.rs`'s
pre-existing `cargo:warning=` target list. `cargo fmt --all --check` clean; `check_status.py --check`,
`check_docs_links.py` and `check_source_layout.py` clean (the deleted `src/device_entry/tests.rs` and
the `mod device_entry;` line go together, so no `.rs` is left undeclared). The four rows that share the
removed test are updated together: `docs/status.toml`'s CPU rows (481 → 480, 479 → 478), the CUDA unit
row (566 → 565) and the sanitizer row (566 → 565), each with its `projection_base_passed`; `AGENTS.md`
carries the same numbers and the changelog sentence.

**Docs.** [#240](https://github.com/yusiwen/minfer/issues/240) is closed with the evidence comment; [#241](https://github.com/yusiwen/minfer/issues/241) carries the absorption and the deletion
(not silent); [#239](https://github.com/yusiwen/minfer/issues/239) gets the `slot` closure, [#238](https://github.com/yusiwen/minfer/issues/238) the `matmul_f32_ptr` closure; [#244](https://github.com/yusiwen/minfer/issues/244) gets
`cc`'s annotation and the two stale-doc findings as decisions; [#243](https://github.com/yusiwen/minfer/issues/243) gets `cc`; [#185](https://github.com/yusiwen/minfer/issues/185) is closed with
the guard's removal and what remains of its question. The live doc claims that named the removed knob
(`MINFER_CUDA_DEBUG` in `docs/CUDA-BACKEND-DESIGN.md`, the walkthrough and the tutorial) and the
`#189` record's sentence about the guard covering `layer_gpu` are updated in place; the historical
`#185`/`#188` records keep their wording.

**Limits.** (1) The macOS-only modules were not touched — the `build-macos` CI job is their only gate.
(2) The counted census is a `cargo check`: a "caller" is a compile-time reference, not a verified
runtime exercise; the suites above are the independent runtime check. (3) `--features debug_dump` was
not built, as in the census itself. (4) The real-model gate sets were **not** re-run here (they need
the cached models and were unaffected — no test in that set was touched); their rows are unchanged and
still dated 2026-09-27. (5) `cc`'s tightening and the residual dead items are named above and left to
[#242](https://github.com/yusiwen/minfer/issues/242)/[#243](https://github.com/yusiwen/minfer/issues/243)/[#244](https://github.com/yusiwen/minfer/issues/244).

#### Test-infrastructure record (#242, 2026-10-01) — the remaining no-caller items are deleted

**The ticket.** [#242](https://github.com/yusiwen/minfer/issues/242) is T3c of the `allow(dead_code)` census: bucket **A** — an item
rustc reports dead in the maximal build (`--release --features cuda`) that is **still** dead with
`--tests --features cuda`, i.e. nothing reaches it at all. It is the last of the census' *code* items
after [#238](https://github.com/yusiwen/minfer/issues/238)/[#239](https://github.com/yusiwen/minfer/issues/239) (buckets B/C, test-only) and
[#240](https://github.com/yusiwen/minfer/issues/240)/[#241](https://github.com/yusiwen/minfer/issues/241) (the legacy CUDA wrapper layer and the `device_entry` guard), and
it closes [#235](https://github.com/yusiwen/minfer/issues/235)'s last row (`kv_cells_for`).

**Reconciled numbers — the tracker was patched to this run.** The ticket body carried the `e9bf4c5`
item list, so it was re-derived and the body rewritten:

| tree | dead `src/` sites | dead items | bucket A code | bucket A shape |
|---|---|---|---|---|
| `e9bf4c5` — the durable census' recorded baseline | 314 | 260 | 89 | 32 |
| `79c9837` — after [#238](https://github.com/yusiwen/minfer/issues/238)/[#239](https://github.com/yusiwen/minfer/issues/239) | 191 | 156 | — | — |
| `5147d2c` — this branch's parent (after [#240](https://github.com/yusiwen/minfer/issues/240)/[#241](https://github.com/yusiwen/minfer/issues/241)) | **143** | **109** | **65** | 15 |
| `8d3f54a` — this branch's final tree | **71** | **47** | **3** | 15 |

Method, identical on both sides: `strip.py` in a scratch worktree (every `allow(dead_code)` replaced
by a `//STRIPPED` comment, line numbers preserved), then `cargo check --release --features cuda
--message-format=json`, counting **every** `src/` span of every `dead_code` diagnostic — rustc groups
a dead `impl` into one diagnostic, so a primary-span-only read undercounts ~2.6×. The
`--tests --features cuda` capture is the A-vs-B/C discriminator, and the `--cap-lints=warn` twins
prove the `deny(warnings)` capture was not truncated (symmetric difference **0** on both cpu and
cuda). The branch deletes **62 code items**; the counted drop is **72 sites / 62 items**, exactly the
number deleted.

**Per-file disposition (62 census items; every item read at its definition, its cross-module name
matches read rather than name-matched, and `git log -S` read where its doc claimed a caller).**

| file | items | what was deleted and why it was safely dead |
|---|---|---|
| `src/cache.rs` | 6 | `KVCacheLayer::{store, get_k, get_v, clear, store_multi}` and `KVCache::clear` — the graph path owns KV in the allocator; the type survives only as `forward`'s vestigial `&mut KVCache` argument (its fields are the shape items below) |
| `src/cuda.rs` | 4 | the `cudaGetErrorString` FFI declaration + `cuda_error_string` (production names errors through `cuda_error_name`), the dead `minfer_site_hist_reset` FFI declaration (the `.cu` defines and calls its own), and the process-wide `stream_sync_count` with its `STREAM_SYNCS` static and the increment in `CudaState::sync` ([#185](https://github.com/yusiwen/minfer/issues/185) moved every gate to the backend's own counter) |
| `src/gguf.rs` | 22 | `GgufType::type_name`; `GgufContext::{get_version, get_alignment, get_kv_type, get_val_u8…get_val_f64, get_val_str, get_val_data, get_n_tensors, find_tensor, get_tensor_offset, get_tensor_name, get_tensor_type, get_tensor_size, dump_metadata}` (main.rs has its own dumper). The surviving accessors are production- or test-reached, so the `impl GgufContext` `#[allow(dead_code)]` went with them |
| `src/graph/mod.rs` | 2 | `ComputeGraph::{node_mut, n_elements}` |
| `src/graph/alloc.rs` | 2 | `GraphAllocator::{kv_cells_for, reset_cross_stats}` |
| `src/graph/backend.rs` | 1 (+3 impls) | the `Backend::name` trait method and its cpu/cuda/metal impls — the live name surface is `registry::Backend::name(self)`, the handle's own method |
| `src/graph/params.rs` | 1 | `next_weights_version` (the `GraphParams::weights_version` field stays in the reuse identity) |
| `src/models/mod.rs` | 1 | `ModelDef::format_chat` |
| `src/models/qwen2/mod.rs` | 1 (+1 impl) | `format_chatml` and its `format_chat` impl |
| `src/models/qwen3/mod.rs` | 2 (+1 impl) | `format_chatml`, the inherent `Qwen3Model::n_layer`, and the `format_chat` impl |
| `src/tensor.rs` | 20 | `from_data_with_strides`, `nelements`, `nrows`, `ncols`, `data_mut`, `data_f32_mut`, the eight `data_q*`/`data_q*_mut` byte accessors, `get_f32`, `set_f32`, `copy_from`, `reshape`; with them the `impl Tensor` allow went |
| `src/server/batch/tests.rs` | — | the test mock's `format_chat` (forced by the trait change; it was an `unreachable!()`, so no assertion changes) |

**Named keeps, not silent leftovers.** `cudaStreamWaitEvent` + `CudaState::stream_wait_event` stay
reserved for the F5 device→device staging copy, which [#138](https://github.com/yusiwen/minfer/issues/138) (landed 2026-10-04) found unreachable — see the F5 S2 record. `ModelDef::forward_graph` stays because its
only caller is `models::qwen2::graph::tests`'s `#[cfg(target_os = "macos")]` block
(`tests.rs:3021`): the Linux census cannot compile it, `cargo test` on a Mac would not compile if the
method were deleted, and CI's macOS job (`cargo build --release`) would not catch the break. That is
this ticket's **one platform-cfg blind spot** in bucket A, found by reading the call site rather than
by the oracle. The 15 bucket-A **shape** items (`KVCacheLayer`'s fields, `Vendor::{Amd,Mthreads,
Apple}`, `AttnMode::Mha`, `SampledToken::logit`, `Slot::id`, `Tokenizer::{id_to_score,id_to_type}`,
`HfCheckpoint::order`) are [#244](https://github.com/yusiwen/minfer/issues/244)'s decisions; `cc`'s
annotation (2 of the 5 residual `src/cuda.rs` sites) is [#243](https://github.com/yusiwen/minfer/issues/243)'s.

**No newly-dead items.** The before/after diff keyed on `(name, kind, container)` has an **empty
"new" side**: 0 newly-dead. The after-set is a strict subset of the before-set, so deletion unmasked
no transitive dead code — the `impl GgufContext` and `impl Tensor` allow-removals were the two places
that could have exposed some, and neither did.

**Findings — every doc that named a caller was stale; no lost caller.** `GgufType::type_name` ("only
exercised by tests/debug tooling today") had no test caller, only the dead `dump_metadata`;
`ComputeGraph::node_mut` ("fusion pass / debug tooling") is never called by the fusion pass;
`ComputeGraph::n_elements` ("assertion / debug helper") by no assertion; `kv_cells_for` ("the
resolver's C2 consumers are the backends") names a wrapper C2 never used — the backends read the
resolved `cells` input through `kv_cells_for_seq` ([#235](https://github.com/yusiwen/minfer/issues/235)); `reset_cross_stats` ("a gate that wants
an absolute number") by no gate — every F5 gate reads `cross_stats().delta(before)`;
`next_weights_version` ("Phase 6 wires the model to bump it") was introduced by the Phase-4 commit
and never gained a caller (`git log -S`), i.e. the A7 "a future feature will need it" pattern;
`cuda_error_string` ("for diagnostics that want the prose form") by no diagnostic after
[#122](https://github.com/yusiwen/minfer/issues/122); `Backend::name` ("part of the Backend API surface") by nothing — all `.name()`
calls are on the registry handle; `ModelDef::format_chat`/`format_chatml` ("kept as the fallback
implementation") by nothing — `template.rs` is the path; `Qwen3Model::n_layer` ("stays as a
concrete-type helper") by no concrete-type caller; the `impl Tensor` "complete ggml_tensor interface
(used by tests / debug tooling)" note and the `impl GgufContext` "public raw API surface (tests /
debug tooling)" note likewise. None is the [#218](https://github.com/yusiwen/minfer/issues/218) *lost-caller* shape — no earlier cleanup
removed a caller that this ticket then deleted — so nothing here is referred to
[#244](https://github.com/yusiwen/minfer/issues/244) as a decision; the two bucket-**B** doc-vs-caller items [#239](https://github.com/yusiwen/minfer/issues/239) reported there
(`KvCache::set_owner`, `OffloadPlan::all_on_device`) are outside this ticket's bucket and untouched.

**Evidence form: the census re-run (this ticket's version of mutation evidence, as in #240).** There
is no gate to mutate — the deleted code has no caller, so no test observes it. The observable claim
is instead "the dead set shrank by exactly what was removed and nothing else became dead", and that
is *measured* on both sides (table above): sites 143 → 71, items 109 → 47, the item drop equal to the
62 deleted, and a before/after item diff whose "new" side is empty. The one test-adjacent edit is
`stream_sync_counts_are_per_backend_not_process_wide`'s mutation note, which named the deleted
process-wide counter as its mutation; it now names the still-available mutation (bump a process-wide
static in `state_sync` and read it) and the test's two assertions are unchanged.

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01.**
`cargo test --release` → **480 / 0 / 36** unit + **10 / 0 / 6** integration.
`bash scripts/cuda_test.sh` → **565 / 0 / 42**.
`compute-sanitizer --tool memcheck --target-processes all <test binary> --test-threads=1` →
**0 API errors** over 565 / 0 / 42. No count row moves: no `#[test]` was added or removed, and every
count above is the same number the #240/#241 record left. `cargo check --release`,
`cargo check --release --features cuda` and `cargo check --release --tests --features cuda` all exit
0; the non-test builds are warning-free and the only `warning:` line is `build.rs`'s pre-existing
`cargo:warning=` target list. `cargo fmt --all --check` clean; `check_status.py --check`,
`check_docs_links.py` and `check_source_layout.py` clean.

**Bar named before measuring.** No `#[test]` is added, removed or weakened: the bar is "no test loses
an assertion; every deletion is a no-caller item", and the evidence form is the census before/after
(the deleted code has no gate to mutate, so a mutation-checked gate would be a claim about nothing).

**Limits.** (1) The macOS-only modules and tests are not compiled here; `ModelDef::forward_graph` is
the one bucket-A item that lives only behind `target_os = "macos"`, and it was kept for that reason —
`build-macos` compiles the library, not the macOS-gated tests, so nothing in CI would have caught a
deletion of it. (2) `--features debug_dump` was not built; every bucket-A name was grepped against
its gated code (`src/dump.rs`, `src/quants.rs`, `src/main.rs`) and none appears there. (3) The
counted census is a `cargo check`: a "caller" is a compile-time reference, not a verified runtime
exercise; the suites above are the independent runtime check. (4) The real-model gate sets were not
re-run (no test in that set was touched); their rows remain dated 2026-09-27.

#### Test-infrastructure record (#243, 2026-10-01) — the config-gated dead-code annotations name their configuration

**The ticket.** [#243](https://github.com/yusiwen/minfer/issues/243) is T4 of the `allow(dead_code)`
census: bucket **D** — an item rustc reports dead in the build **without** `--features cuda` and live
**with** it. Its annotation must name that configuration
(`#[cfg_attr(not(feature = "cuda"), allow(dead_code))]`), not blanket-silence the item everywhere.
The same rule covers an item dead only in the non-test build (`CudaState::cc`, whose one reader is a
`#[cfg(test)]` accessor), and the sentence this ticket adds to Core Convention 5 states it once.

**Reconciled numbers — the tracker was patched to this run.** The ticket body carried the `e9bf4c5`
item list, so the census was re-derived on `212e748` (after [#242](https://github.com/yusiwen/minfer/issues/242)) and the body rewritten:

| tree | dead `src/` sites | dead items | bucket D |
|---|---|---|---|
| `e9bf4c5` — the durable census' recorded baseline | 314 | 260 | 22 |
| `79c9837` — after [#238](https://github.com/yusiwen/minfer/issues/238)/[#239](https://github.com/yusiwen/minfer/issues/239) | 191 | 156 | — |
| `5147d2c` — after [#240](https://github.com/yusiwen/minfer/issues/240)/[#241](https://github.com/yusiwen/minfer/issues/241) | 143 | 109 | — |
| `212e748` — this branch's parent (after [#242](https://github.com/yusiwen/minfer/issues/242)) | **71** | **47** | **22** |
| this branch's final tree | **71** | **47** | **22** |

Method, identical on both sides: `strip.py` in a scratch worktree (every `allow(dead_code)` replaced
by a `//STRIPPED` comment, line numbers preserved), then `cargo check --release --message-format=json`
and `cargo check --release --features cuda --message-format=json`, taking **every** `src/` span of
every `dead_code` diagnostic (rustc groups a dead `impl` into one diagnostic, so a primary-span-only
read undercounts ~2.6×). Bucket D is the set difference: items in the CPU capture that are not in the
`--features cuda` capture. The `--tests --features cuda` capture is the dead-in-the-maximal-build
discriminator; `src/cuda.rs` and `src/graph/cuda_backend.rs` do not exist without the feature, so
their dead items are outside bucket D by construction.

**Per-item disposition (all 22).**

| disposition | items |
|---|---|
| tightened to `#[cfg_attr(not(feature = "cuda"), allow(dead_code))]` | `GraphAllocator::kv_format` (`src/graph/alloc.rs`) — the only bucket-D item still bare |
| already precise, module-level form, no edit | `DeviceKey`, `Provenance`, `TIERS`, `GENERIC`, `llama_key`, `Selected`, `select`, `select_forced`, `family_row`, `select_by_key` — `#[cfg_attr(not(feature = "cuda"), allow(dead_code))] mod device_tier;` (`src/main.rs`); `OffloadPlan::allows_weight` — `#[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))] pub mod offload;` (`src/graph/mod.rs`); `CudaWeightReg` + `cuda_weight_reg` — `#![cfg_attr(not(feature = "cuda"), allow(dead_code))]` (`src/models/weight_reg.rs`); `q4k_dsc_payload_bytes` + `q4k_dsc_payload_ok` + `q4k_dsc_plane_admitted` — the same inner form (`src/q4k_dsc.rs`) |
| already precise, item-level form, no edit | `DeviceMemory::{Reported.free, QueryFailed.code}` — `#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]` (`src/graph/allocplan.rs`) |
| explicitly kept, naming [#244](https://github.com/yusiwen/minfer/issues/244) | `Vendor`, `QClass`, `DeviceTier` (`src/device_tier.rs`) — their bare item-level `#[allow(dead_code)]` is load-bearing for [#244](https://github.com/yusiwen/minfer/issues/244)'s never-constructed members in the **cuda** build (`Vendor::{Amd,Mthreads,Apple}`, `QClass::Other`, `DeviceTier::{source,mmvq_batch_default,mmvq_batch_by_type}`). Tightening the containing item to `not(feature = "cuda")` would expose exactly those members and break the warning-free cuda build; giving each its own note is [#244](https://github.com/yusiwen/minfer/issues/244)'s decision, so this ticket does not pre-empt it and a comment records the coupling on that issue. |

`CudaState::cc` is not bucket D (it lives in the feature-gated `src/cuda.rs`): its only reader is the
`#[cfg(test)] CudaState::cc()` accessor, so its honest form is `#[cfg_attr(not(test), allow(dead_code))]`
— the configuration where it is unused is the test build's complement, not a feature. The
`#[cfg(test)]` accessor and its `graph/cuda_backend/tests.rs` caller are untouched. The three
remaining residual `src/cuda.rs` dead sites (`cudaStreamWaitEvent` + `CudaState::stream_wait_event`,
the F5 device→device wait) are intentionally untouched, pending open
[#138](https://github.com/yusiwen/minfer/issues/138).

**The reverse direction does not exist here.** Seven items are reported in the cuda capture but not
the CPU one, all in `src/device_tier.rs`: `Vendor::{Amd, Mthreads, Apple}`, `QClass::Other`,
`DeviceTier::{source, mmvq_batch_default, mmvq_batch_by_type}`. They are not "dead only with cuda":
they are never constructed or read in **either** configuration. The CPU build reports the enclosing
enum/struct as unused and never descends to its members, so the members never appear in its dead set.
A `#[cfg_attr(feature = "cuda", allow(dead_code))]` on them would silence the cuda report of an item
that is dead in both — the over-claim this ticket exists to remove. They are [#244](https://github.com/yusiwen/minfer/issues/244)'s
shape items, and the census bucket definition has to be read as the *difference* between the two
captures, not as "whichever capture names an item".

**Evidence form: the stripped-oracle identity (this ticket's version of mutation evidence, as in #240/#242).**
There is no gate to mutate — the diff is two attributes and two comments, and an attribute-only change
has no observable behaviour to break. The falsifiable claim is instead that tightening changes the
*diagnostics* by exactly nothing, and that each `cfg_attr` names the configuration rustc proved dead:

| capture | before | after | symmetric difference |
|---|---|---|---|
| `cargo check --release` (CPU) | 59 items / 80 sites / 38 diagnostics | 59 / 80 / 38 | **0** |
| `cargo check --release --features cuda` | 47 / 71 / 24 | 47 / 71 / 24 | **0** |
| `cargo check --release --tests --features cuda` | 23 / 34 / 16 | 23 / 34 / 16 | **0** |

Not one item was newly exposed and not one was hidden: the after-set is the before-set. (The
`:296 → :300` line move of `kv_format` is the four-line doc-comment addition; the item key is
`(file, name, kind)`.)

**Non-truncation of the required captures, re-proved on this tree.** `#![cfg_attr(not(test),
deny(warnings))]` turns the lint into an error, so the build aborts after the lint pass; the census
uses a twin build per configuration in which that one crate-level line is commented out (the cheap
equivalent of the durable census' `RUSTFLAGS=--cap-lints=warn`, which would force an nvcc re-run for
every target). Required vs twin item sets: CPU 59 = 59, cuda 47 = 47, symmetric difference **0** in
both phases — so neither capture is truncated.

**Per-item config split (the proof each `cfg_attr` is precise, not decorative).**

| item | `cfg_attr` covers | dead there? | dead in the other configuration? |
|---|---|---|---|
| `GraphAllocator::kv_format` | `not(feature = "cuda")` | CPU: **yes** | `--features cuda`: **no** |
| `CudaState::cc` (field) | `not(test)` | non-test cuda: **yes** | `--tests --features cuda`: **no** |

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01.**
`cargo test --release` → **480 / 0 / 36** unit + **10 / 0 / 6** integration.
`bash scripts/cuda_test.sh` → **565 / 0 / 42**.
`cargo check --release`, `cargo check --release --features cuda` and `cargo check --release --tests
--features cuda` all exit 0; the non-test builds are warning-free and the only `warning:` line is
`build.rs`'s pre-existing `cargo:warning=` target list. No count row moves: no `#[test]` was added or
removed, so `docs/status.toml` and the `AGENTS.md` rows are unchanged. `cargo fmt --all --check`
clean; `check_status.py --check`, `check_docs_links.py` and `check_source_layout.py` clean.

**Bar named before measuring.** No `#[test]` is added, removed or weakened, and the change is
attributes plus comments only: the bar is "the stripped-oracle dead set is unchanged in each of the
three configurations, and each tightened annotation is dead in exactly the configuration it names" —
stated before the after-capture was taken and measured by the tables above.

**Limits.** (1) The macOS-only modules are not compiled here, but no bucket-D item is behind
`target_os = "macos"` in the sense that would matter: `--features cuda` is the maximal Linux
configuration and the CPU capture is the other side of the difference. (2) `--features debug_dump`
was not built; neither the tightened item nor any bucket-D name appears in `src/dump.rs`. (3) The
counted census is a `cargo check`: a "caller" is a compile-time reference, so `kv_format`'s liveness
with cuda rests on `cuda_backend::entry`'s `kv_format` hook being compiled in that configuration
(the suite above is the independent runtime check). (4) `CudaState::cc`'s test build is
`--tests --features cuda`, not the macOS test build; the `#[cfg(test)]` accessor is
platform-independent.

#### Test-infrastructure record (#244, 2026-10-01) — the dead-code decisions get a verdict, and the unambiguous subset lands

**The ticket.** [#244] is T5, the last ticket of the `allow(dead_code)` census: the items rustc
still reports on the stripped oracle after T1–T4, which need a decision rather than a deletion.
It is also the ticket the earlier records handed their findings to — [#239]'s two lost callers
(`KvCache::set_owner`, `OffloadPlan::all_on_device`), its vacuous `live_peak` gate and its
`Tensor::new` coverage gap; [#240]/[#241]'s stale-doc items; [#242]'s ~12 stale-doc notes; and
[#243]'s three `src/device_tier.rs` container blankets.

**Reconciled lineage — the oracle re-run on `0a81d47`.** Method identical on both sides: `strip.py`
in a scratch worktree (every `allow(dead_code)` replaced by a `//STRIPPED` comment, line numbers
preserved), then `cargo check --release`, `cargo check --release --features cuda` and
`cargo check --release --tests --features cuda`, each with `--message-format=json`, taking **every**
`src/` span of every `dead_code` diagnostic (a primary-span-only read undercounts ~2.6× because
rustc groups a dead `impl` into one diagnostic). The scratch worktree was pinned to the repo's
`1.97.1` toolchain (`RUSTUP_TOOLCHAIN` is set to `stable` in this shell; the census unsets it).

| tree | cuda dead sites | cuda dead items | CPU sites/items | tests-cuda sites/items |
|---|---|---|---|---|
| `e9bf4c5` — the durable census' baseline | 314 | 261 | — | — |
| `79c9837` — after [#238]/[#239] | 191 | 157 | — | — |
| `5147d2c` — after [#240]/[#241] | 143 | 110 | — | — |
| `212e748` — after [#242] | 71 | 48 | — | — |
| `0a81d47` — this branch's parent (after [#243]; re-run here) | **71** | **48** | 80 / 60 | 34 / 23 |
| this branch's final tree | **57** | **39** | 67 / 51 | 22 / 15 |

**A one-item parser bug the re-run exposed, fixed in the durable `census.py`.** `RopeStyle::Interleaved
= 1` carries an explicit discriminant, and the item-name regex (`VARIANT_RE`) accepted only
`(`/`{`/`,`/end after the identifier — so the variant's own row was dropped while its *span* still
counted. The durable captures were re-parsed after the fix: `e9bf4c5` is **261** cuda / **198** CPU
items, not 260 / 197 — exactly one more in each non-test capture. The sites columns are span-based
and unchanged, and the historical rows above are the recorded ones **plus that one** (the fix is
deterministic and the variant is present in every tree). The cuda set is therefore **41 shape + 7
code** items, not the 40 + 7 [#242] reconciled, and the item this recovery adds —
`RopeStyle::Interleaved` — is one of [#244]'s own (its bare `#[allow(dead_code)]` is now
`#[cfg_attr(not(test), allow(dead_code))]`, the test that constructs it being `graph::op_matrix`).

The 7 code items are `cudaStreamWaitEvent` + `CudaState::stream_wait_event` (reserved for the F5 device→device staging copy — still unreachable after [#138](https://github.com/yusiwen/minfer/issues/138)),
`KvCache::set_owner`, `OffloadPlan::all_on_device`, `ModelDef::{as_any, forward_graph, offload}`.

**Verdict table (all 48).**

| verdict | items | landed? |
|---|---|---|
| **delete** (internal, never read in any build, no API contract) | `SampledToken::logit`; `Slot::id`; `KVCacheLayer::{k,v,size,max_size,dim}` + `KVCache::layers` | **yes** |
| **wire the lost caller** (small, behaviour-preserving) | `OffloadPlan::all_on_device` (called by `resolve`'s unset-request arm) | **yes** |
| **keep + honest note, precise annotation** | `DType::{F16,Q8_0}`; `Op::{Scale,Softmax,Reshape,Permute,BatchMatMul}`; `AttnMode::Mha`; `RopeStyle::Interleaved`; `TimingMode::{Off,Private}`; `OffloadRequest::AutoWithBudget`; `HfCheckpoint::order` (deferred to [#209]); `Tokenizer::{id_to_score,id_to_type}`; `Vendor::{Amd,Mthreads,Apple}`; `QClass::Other`; `DeviceTier::{source,mmvq_batch_default,mmvq_batch_by_type}`; `TurnOutcome::{text,stopped_by_eog,stopped_by_string}`; `Tokenizer::{special_tokens,im_start,im_end}`; `BackendCaps::{supports_op,supports_fused,supports_attn_span}`; `BackendEntry::name`; `ModelDef::{as_any,forward_graph,offload}` | **yes** (annotation + note; the membership questions are escalated) |
| **keep, nothing to change** | `CudaState::cc` (already `not(test)` from [#243]); `cudaStreamWaitEvent` + `stream_wait_event` (bare, dead in every build — [#138](https://github.com/yusiwen/minfer/issues/138) landed 2026-10-04 and did not create a reachable caller) | n/a |
| **escalate, do not land** | `BackendCaps` **authority**; `ModelDef::as_any`; `ModelDef::offload`; `KvCache::set_owner`; the deferred `DType`/`Op`/`RopeStyle` variant **membership**; removing `ModelDef::forward`'s `&mut KVCache` | **no** |

Two annotation-shape corrections ride along: `Op::View`'s `#[allow(dead_code)]` was **unnecessary**
(D1's `GraphBuilder::split_parts` constructs it, and the stripped oracle does not report it), so it
was dropped; and the `ModelDef` trait-level blanket became four member-level annotations
(`not(test)` for `as_any`/`offload`, `any(not(test), not(target_os = "macos"))` for `forward_graph`)
so `forward`/`kv_format`/… stay checked.

**The three `device_tier.rs` container blankets (#243's acceptance item).** `Vendor`, `QClass` and
`DeviceTier` lost their item-level `#[allow(dead_code)]`; each member rustc named in the cuda build
now carries its own: `Vendor::{Amd,Mthreads,Apple}` bare (never constructed in **either**
configuration — the CPU build reports the enclosing enum and never descends), `QClass::Other` and
`DeviceTier::{source,mmvq_batch_default,mmvq_batch_by_type}` `not(test)` (constructed/read by
`device_tier::tests`). The non-test cuda build stays warning-free, which is what #243 could not
achieve without deciding these members first.

**The vacuous gate (#239's finding), fixed rather than removed.** `AllocPlan::plan`'s in-loop `live`
never decreased, so `peak` always equalled `reserved_bytes` and `peak.max(live_peak(..))` could
never observe `live_peak`; `the_live_peak_is_not_the_reserved_total` stayed green under a mutation
of `live_peak`'s arithmetic. The dead in-loop accumulation is **gone**; `live_peak` now takes its
peak between the add and the remove pass (an interval whose `first == last` counts for its one
step), and the test adds the shape where the two numbers genuinely differ — two **different** classes
whose lifetimes never overlap (both reserved, one live at a time) — on top of the existing
same-class-singletons and two-classes-overlapping cases. No assertion was removed or weakened; two
were added to an existing test, so no count row moves. **Mutation:** `live += class_bytes(classes[i])`
→ `live += 0` fails `the_live_peak_is_not_the_reserved_total` at `allocplan/tests.rs:171` (before
this ticket the same mutation left it green). The `[#244]` module doc also replaces the stale
sentence that claimed the plan is the feasibility gate: the gate is `GraphAllocator::alloc_in_pool`.

**The `Tensor::new` coverage gap (#239's finding), closed.** Its only consumer, `f32_tensor` in
`graph/builder/tests.rs`, asserted neither the strides nor the allocation, so the body was
unobserved. The helper now asserts `strides[0..4]`, `nbytes()` and `data.len() == nbytes()` for the
f32 shapes it is called with. **Mutation:** `tensor.strides[1] = 0` fails
`builder_creates_topo_sorted_graph` at the new assertion.

**Census before/after (the evidence form for annotation-only work).**

| capture | before (items / sites) | after (items / sites) | newly dead |
|---|---|---|---|
| `cargo check --release` (CPU) | 60 / 80 | 51 / 67 | **0** |
| `cargo check --release --features cuda` | **48 / 71** | **39 / 57** | **0** |
| `cargo check --release --tests --features cuda` | 23 / 34 | 15 / 22 | **0** |

The cuda-set difference is **exactly** the nine items the ticket deleted or wired — the six
`KVCache` fields, `SampledToken::logit`, `Slot::id` and `OffloadPlan::all_on_device` — and every
other item is the same item at a different line. No item was newly exposed (a deletion can unmask a
transitive callee, so this is checked, not assumed) and none was hidden. Item identity is
`(file, name, kind)` (with the parser fix above); the line-keyed diff is misleading because the
doc/annotation edits move lines.

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01.**
`cargo test --release` → **480 / 0 / 36** unit + **10 / 0 / 6** integration.
`bash scripts/cuda_test.sh` → **565 / 0 / 42**.
`cargo check --release` and `cargo check --release --features cuda` exit 0 with **0 diagnostics**.
The test build (which does not carry `deny(warnings)`) has an **unchanged warning multiset**:
79 diagnostics before and after, with no new, removed or re-counted message; the only difference is
the source line of one pre-existing `unused variable: pos` in `graph/builder/tests.rs`, shifted by
the assertions this ticket added above it. No count row moves: no `#[test]` was added, removed or
weakened, so `docs/status.toml` and the `AGENTS.md` rows are unchanged. `cargo fmt --all --check`,
`check_status.py --check`, `check_docs_links.py` and `check_source_layout.py` are clean.

**Bar named before measuring.** No item is load-bearing (the landed code changes are deletions and
one constructor call whose result is identical), so the bar is the census identity plus the two test
mutations: "the after-set is the before-set minus exactly the deleted/wired items", "the
`live_peak` mutation is red", "the `Tensor::new` mutation is red" — stated before the after-capture
and the two mutation runs, and measured by the tables above.

**Escalated — reported on [#244], not landed.** (1) `BackendCaps` authority: the three
`supports_*` fields are written by every `entry()` but read only by tests; the trait methods call the
same module-level functions, so the answers cannot diverge, but the registry field is not the
production read path. Options: make the trait/assignment read the caps (the design's intent, a real
refactor), drop the three fields (small, removes the mirror), or keep them as the test-asserted
mirror (what the code does). (2) `ModelDef::as_any` — no `src/` downcast outside `#[cfg(test)]`;
keep as the test harness' handle or delete it and restructure that harness. (3) `ModelDef::offload`
— the builders read `model.offload.plan` directly; wire them through the trait method or delete it.
(4) `KvCache::set_owner` — `own_range` clamps where `set_owner` errors, so wiring is not
behaviour-preserving; wire behind the clamp, delete it with its loud check, or keep it test-only.
(5) The deferred `DType`/`Op`/`RopeStyle` variant membership (keep the vocabulary or delete the
variants). (6) Removing `ModelDef::forward`'s legacy `&mut KVCache` parameter (the type is now an
empty marker kept so the signature does not move). Each has a recommendation on the ticket; none
changes behaviour, so none belongs in a dead-code cleanup silently.

> **#252 forward note (2026-10-02):** escalation (6) landed. `src/cache.rs`, its `mod cache;`
> declaration, the `&mut KVCache` argument of `ModelDef::{forward,forward_graph}`, the `KVCache::new`
> call in `main.rs` and the tests that constructed one only for the signature are all gone; the
> parameter was never read by any path (the graph allocator owns KV). See the [#252] record in
> §test-infrastructure.

**Doc corrections landed.** `docs/BACKEND-REGISTRY-DESIGN.md` §3 + §10 carry the `BackendCaps`
correction above; `src/models/{qwen2,qwen3}/mod.rs` no longer claim `ModelDef::offload()` hands the
plan to the builders (they read the field); `src/graph/allocplan.rs`'s module doc no longer claims
the plan is the feasibility gate. `clear_launch_failure`/`init_kv_cache` and the [#242] stale-doc
items were already corrected or deleted by [#240]/[#242], and were re-checked here: no live doc
still names them as callers. AGENTS.md Core Convention 5 gains one sentence: a deliberately retained
**deferred** item says what would construct or read it.

**Limits.** (1) The macOS-only modules (`metal.rs`, the Metal half of `graph/metal_backend.rs`) are
not compiled on Linux, so their annotations are outside this census — the macOS CI job is the only
gate, and the `ModelDef::forward_graph` cfg is written to be true on the macOS test build, where its
one caller lives. (2) `--features debug_dump` was not built; no #244 item appears in `src/dump.rs`.
(3) The census is a `cargo check`: a "caller" is a compile-time reference, so `all_on_device`'s new
liveness rests on `resolve` being compiled in every build (the offload unit tests are the
independent runtime check). (4) The `device_tier.rs` member annotations are asserted warning-free in
the two non-test builds and the cuda test build, not on a macOS host (the module is CUDA-only).
(5) The one-item parser fix above is applied to the durable `census.py` but is **not** in the repo:
it does not change any site count, and it corrects item counts only.

[#138]: https://github.com/yusiwen/minfer/issues/138
[#209]: https://github.com/yusiwen/minfer/issues/209
[#238]: https://github.com/yusiwen/minfer/issues/238
[#239]: https://github.com/yusiwen/minfer/issues/239
[#240]: https://github.com/yusiwen/minfer/issues/240
[#241]: https://github.com/yusiwen/minfer/issues/241
[#242]: https://github.com/yusiwen/minfer/issues/242
[#243]: https://github.com/yusiwen/minfer/issues/243
[#244]: https://github.com/yusiwen/minfer/issues/244

#### Test-infrastructure record (#244 escalations, 2026-10-01) — the five decided verdicts land, and decision 6 becomes [#252]

**The ticket.** [#244]'s six escalations each needed a decision rather than a dead-code deletion.
Decision 5 (keep the deferred `DType`/`Op`/`RopeStyle` vocabulary, each member naming what would
construct it) had already landed in [#251]; this branch lands 1–4 and 2, and decision 6 — retiring
the legacy `KVCache` and `ModelDef::forward`'s `&mut KVCache` parameter — is a **trait-signature
refactor, not dead-code cleanup**, so it was filed separately as [#252] with the verified inventory
rather than smuggled into this ticket.

**Verdict table.**

| # | item | verdict | why |
|---|---|---|---|
| 1 | `BackendCaps::{supports_op, supports_fused, supports_attn_span}` | **delete the three fields** | every `entry()` wrote them and production read none: the trait methods forward to the backend module's free function / constant, and assignment reads the trait. The registry keeps `reads_packed_kv`, the one field production reads (a *format* question answered without a `&self`, §8 of the registry design) |
| 2 | `ModelDef::as_any` | **keep, doc fixed** | its callers are `#[cfg(test)]` only (the harness deliberately holds `Arc<dyn ModelDef>`); no `src/` production path downcasts, and there is no weight-registration consumer — the doc no longer claims one |
| 3 | `ModelDef::offload` | **delete the trait method + both impls** | the graph builders and the assignment pass read the concrete `model.offload.plan` field; the method's only readers were tests |
| 4 | `KvCache::set_owner` | **delete it and its two test assertions** | no production consumer, and it was **not** behaviour-equivalent to `own_range` (which truncates out-of-range silently where `set_owner` returned `Err`), so wiring it in would have changed behaviour — option (a) was rejected on that ground |
| 5 | deferred `DType`/`Op`/`AttnMode`/`RopeStyle` vocabulary | already landed in [#251] | verified only: every kept member names what would construct it |
| 6 | `ModelDef::forward`'s legacy `&mut KVCache` | **escalated as [#252]** | a trait-signature refactor with its own acceptance criteria; the parameter is provably unread |

> **#252 forward note (2026-10-02):** the escalation landed as its own PR — decision 6 is no longer
> pending. The `KVCache` type, the `mod cache;` declaration and the `&mut KVCache` parameter are
> deleted, with **zero behaviour change** (every implementation ignored the argument, and both graph
> entry points already bound it as `_kv`). The [#252] record in §test-infrastructure carries the
> evidence.

**The caps≡trait gate was repointed, not deleted.** `registry_caps_match_the_backend_trait` used to
compare `Backend::CPU.caps().supports_op` (a registry field) against the trait method. With the
fields gone, the same test now compares **`cpu_backend::supports_op` / `supports_fused` /
`SUPPORTS_ATTN_SPAN` against the trait method** — i.e. the authority the field merely mirrored, so
the property the gate was written for (the registry's answer and the trait's answer cannot disagree)
is still asserted, and the two sides are now the forwarding pair rather than a field and the code it
pointed at. `names_resolve_and_unknown_names_are_refused` had three `caps().supports_*` assertions in
its "an unregistered handle claims nothing" block; they were repointed to the remaining field
(`reads_packed_kv`), which is the only capability an unregistered handle can now be asked about.
**Mutation:** making `CpuBackend::supports_op` diverge from `cpu_backend::supports_op` (return `true`
unconditionally) fails the test — `panicked at src/graph/registry/tests.rs:319: assertion left ==
right failed: Input F16, left: false, right: true`; reverted.

**The removed-assertion note (decision 4).** `Cache::set_owner`'s only callers were the two
assertions in `graph::kvcache::tests::own_prefix_and_release_round_trip`:
`c.set_owner(0, 1, FREE).unwrap()` + the `owner[1] == FREE` check (the release half) and
`assert!(c.set_owner(0, 9, SEQ_MAIN).is_err())` (the loud out-of-range check). **Two assertions are
removed with the method**, which is the same discipline [#240]/[#241] used for the `device_entry`
guard's test: the deleted method's check has no production consumer, and the deliberate production
contract is `own_range`'s silent clamp, so keeping a test that pins the opposite (erroring) contract
would pin behaviour the engine does not promise. No `#[test]` was added or removed; the test is
renamed `own_prefix_round_trip` so its name does not overclaim, and it still covers `own_prefix`'s
owner table + `n_used`. The two `ModelDef::offload` deletions removed no assertion — the five test
call sites (`qwen2::graph::tests` ×3, `tooling::tests` ×4 uses) now read `model.offload.plan`, and
the one call inside a `&dyn ModelDef` closure takes the plan as an explicit argument because the plan
surface is the concrete field.

**Census before/after (same method as [#244]: `strip.py` in a scratch worktree at the pinned
`1.97.1`, three `--message-format=json` captures, every `src/` span of every `dead_code`
diagnostic).** "sites" is the number of such spans; "items" the deduplicated `(file, name, kind)`
set, the same two columns the previous records use.

| capture | before (items / sites) | after (items / sites) | newly dead |
|---|---|---|---|
| `cargo check --release` (CPU) | 51 / 67 | **46 / 60** | **0** |
| `cargo check --release --features cuda` | **39 / 57** | **34 / 50** | **0** |
| `cargo check --release --tests --features cuda` | 15 / 22 | 15 / 22 | **0** |
| `cargo check --release --tests` (CPU) | 7 / 9 | 7 / 9 | **0** |

The cuda set is **33 shape + 6 code → 30 shape + 4 code**, and the difference is exactly the five
deleted items: the three `BackendCaps` **shape** fields (`supports_op`, `supports_fused`,
`supports_attn_span`) and the two **code** items (`KvCache::set_owner`, `ModelDef::offload`). The
site column drops by 7 rather than 5 because two of the removed spans are container spans rustc had
grouped into the now-gone diagnostics (`pub struct BackendCaps {` and `impl KvCache {`), which move
or vanish with the grouping, not with an item. **No item was newly dead** (checked, not assumed: a
deletion can unmask a transitive callee), and the cuda/CPU `--cap-lints=warn` captures have
symmetric difference 0 in both the before and the after run, so neither required capture is a
truncated lint pass.

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-01.**
`cargo test --release` → **480 / 0 / 36** unit + **10 / 0 / 6** integration;
`bash scripts/cuda_test.sh` → **565 / 0 / 42** — all three numbers unchanged, because no `#[test]`
was added or removed (the cap test was repointed, the two `set_owner` assertions were dropped from an
existing test). `cargo check --release` and `cargo check --release --features cuda` exit 0 with **0
diagnostics** (the non-test build's `deny(warnings)` gate). The test build's warning multiset is
**unchanged**: `cargo check --release --tests` 68 warning diagnostics / 35 distinct messages before
and after, `cargo check --release --tests --features cuda` 79 / 38 before and after, with no new,
removed or re-counted message. No count row in `docs/status.toml` or `AGENTS.md` moves, so none was
edited.

**Bar named before measuring.** The removed items are not load-bearing (three write-only fields and
two methods with test-only callers; the landed code changes are deletions plus test reads of the
same concrete field), so the bar is the census identity plus the one forwarding mutation: "the
after-set is the before-set minus exactly the five deleted items, with zero newly dead", "the
`caps`-divergence mutation is red", and "the suite counts are unchanged" — stated before the
after-capture and the mutation run, and measured by the tables above. The mutation rule is applied
where it bites (the repointed gate); for the pure deletions there is nothing to mutate, and what was
verified instead is the census identity and the green suites.

**Limits.** (1) Metal is not compiled on Linux, so the `metal_backend::entry()` edit and the
`ModelDef::as_any`/`forward_graph` annotations are asserted by the macOS CI job, not here. (2) The
census is a `cargo check`: a "caller" is a compile-time reference, so the claim that the three
deleted fields had no production reader rests on rustc plus the `git grep` of `.caps()` (the only
remaining reads are `reads_packed_kv`). (3) `--features debug_dump` was not built; no #244 item
appears in `src/dump.rs`. (4) The `--tests --features cuda` row is the maximal test build; the CPU
`--tests` row is reported alongside it for completeness.

[#244]: https://github.com/yusiwen/minfer/issues/244
[#251]: https://github.com/yusiwen/minfer/pull/251
[#252]: https://github.com/yusiwen/minfer/issues/252

#### Test-infrastructure record (#256, 2026-10-02) — the 25 bare `allow(dead_code)` verdicts land, and the stripped oracle does not move

**The ticket.** [#256] is the last Linux-side work on the `allow(dead_code)` population: the 18
sites whose bare `allow` is redundant because the item is **live** (delete the annotation), the 7
that are dead only outside `cfg(test)` (tighten to `#[cfg_attr(not(test), allow(dead_code))]`), and
the 4 stale comments the deletions exposed. It lands as [PR #257]. It is the Layer-1 prerequisite
[#254] names for its shape ratchet ("resolve the 40 bare/other `allow(dead_code)` sites … so Layer
1's rule starts from a clean baseline"). The two macOS-only sites (`src/metal.rs:971`, `:2029`) stay
with [#255], which needs a Mac; the 9 bare-**correct** deferred sites (`HfCheckpoint::order` —
[#209] owns apply-or-drop — the two `cuda.rs` items [#138](https://github.com/yusiwen/minfer/issues/138) reserved and left dead,
`Vendor::{Amd,Mthreads,Apple}`, `tokenizer.rs::{id_to_score,id_to_type}`, `AttnMode::Mha`) stay bare,
because bare is the precise spelling for an item dead in *every* compilable configuration.

**Method, identical on both sides.** `strip_dead_code.py` in a scratch worktree
(`.worktrees/256-oracle`): every `allow(dead_code)` — bare, `cfg_attr` and inner `#![…]` — replaced
by a line-preserving `//STRIPPED` comment. Then four `cargo check --release --message-format=json`
captures with `RUSTFLAGS=--cap-lints=warn` on the pinned `1.97.1` toolchain (`RUSTUP_TOOLCHAIN` is
`stable` in this shell, so it is unset for every run): `cpu`, `cuda` (`--features cuda`),
`tests_cpu` (`--tests`) and `tests_cuda` (`--tests --features cuda`). **Sites** are every `src/` span
of every `dead_code` diagnostic (a primary-span-only read undercounts ~2.6×); **items** are every
member named in a diagnostic's message, keyed `(file, name, kind)` — "variants `Scale`, `Softmax`, …
are never constructed" contributes five items, and a line move does not read as a change.

**The baseline is the recorded one, byte-for-byte.** The four `f47e4c3` captures are *diff-identical*
to the captures the 36-site table was built from (`/home/yusiwen/minfer-36/captures/*.dead.json` on
the maintainer's box), so the before column is that measurement rather than a restatement of it.

| capture | before sites/items | after sites/items | item Δ | per-file site-count Δ |
|---|---|---|---|---|
| `cargo check --release` (CPU) | 60 / 46 | 60 / 46 | **0** | none |
| `cargo check --release --features cuda` | **50 / 34** | **50 / 34** | **0** | none |
| `cargo check --release --tests` | 22 / 15 | 22 / 15 | **0** | none |
| `cargo check --release --tests --features cuda` | 22 / 15 | 22 / 15 | **0** | none |

**Every diagnostic is unchanged, not only every `dead_code` one.** The complete JSON diagnostic
multiset (level, code, message) is *equal* before and after in all four configurations: 32 `cpu` (all
`dead_code`), 17 `cuda`, 73 `tests_cpu` (61 warnings + 12 `dead_code`) and 86 `tests_cuda` (74 + 12).
The stripped-annotation population falls **61 → 43** (36 bare + 25 `cfg_attr` → 11 bare + 32
`cfg_attr`).

**A — the 18 deletions.** Each `allow` was redundant: the item is live in every configuration that
compiles it.

| site on `f47e4c3` | item | the live reader |
|---|---|---|
| `src/cuda.rs:551,553,555,557,559` | `minfer_launch_fail_{pending,site,name,code,clear}` | via `CudaState::take_launch_failure` → `graph/cuda_backend.rs:2137` `execute_node` (production, cuda) |
| `src/cuda.rs:3329` | `CudaState::take_launch_failure` | the same `execute_node` read; the #162 gates (`src/cuda/issue162_tests.rs:954…`) are the other callers |
| `src/cuda.rs:1346` | `test_call_failure_requested` | production `graph_end_capture_to_exec` (`src/cuda.rs:3403`, the `destroy:graph_destroy` injection) |
| `src/cuda.rs:2965` | `CudaState::get_or_grow` | ~24 non-test `impl CudaState` call sites (MMQ / attention / f16 scratch), e.g. `matmul_f32_ptr_layout` ← `cuda_backend.rs:1205/1346/1440` |
| `src/graph/alloc.rs:321` | `GraphAllocator::enable_cuda` | `graph/json.rs:94`, `models/qwen2/graph.rs:614`, `models/qwen3/graph.rs:534`, and the `cuda_backend::entry` `enable` hook (`:2034`) |
| `src/graph/alloc.rs:351` | `GraphAllocator::cuda` | the registry hooks `pool`/`host_read` (`cuda_backend.rs:2011/2017`) |
| `src/graph/backend.rs:139` | `Backend::synchronize` | `GraphAllocator::sync_backend` (`alloc.rs:2356`) ← `scheduler.rs:221/417` |
| `src/graph/copystats.rs:63` | the `impl CrossCopyStats` blanket | the `impl` is **empty** in every non-test build (`delta` has carried `#[cfg(test)]` since [#238]) |
| `src/graph/kvcache.rs:62` | `KvLayer::owner` | C2's resolver, and C1's own `release_seq` (`kvcache.rs:363`) ← `alloc.rs:1623` |
| `src/graph/kvcache.rs:66` | `KvLayer::n_used` | C2's `after_rm` / `kv_rm` (`kv_rm` ← `conversation.rs:189/258`) |
| `src/graph/kvcache.rs:281` | `KvCache::iter` | `alloc.rs`'s `kv_rm` (`:1121`) and `kv_save_with_host` |
| `src/gguf.rs:1681` | `MmapFile::_file` | **attribute line only** — the field stays; rustc ignores `_`-prefixed fields, so the `allow` silenced nothing |
| `src/models/qwen2/mod.rs:48` | `Qwen2Model::n_layer` | the concrete `&Qwen2Model` caller at `models/qwen2/graph.rs:714` |
| `src/download/mod.rs:295` | `HfSibling::size` | `download/mod.rs:274` `.and_then(\|s\| s.size)` (the `l.size` at `:365` is `OllamaLayer::size` — a name collision, not this field) |

Eight of the 18 are live **only under `--features cuda`** (`enable_cuda`, `cuda`, the five launch-fail
FFI decls and `take_launch_failure`), which is why the CUDA capture and the CUDA warning-free build
are load-bearing here: deleting them without the CUDA check would leave the CPU CI green while the
CUDA `deny(warnings)` build broke — the trap this series exists to close.

**B — the 7 tightenings, with the split that justifies the cfg.** All seven items behave identically:
dead in both non-test configurations, alive in both test configurations, so `not(test)` is exact and
any other cfg (`not(feature = "cuda")`) or a bare `allow` would be wrong.

| item | cpu (non-test) | cuda (non-test) | tests-cpu | tests-cuda | constructed in test code by |
|---|---|---|---|---|---|
| `DType::F16` (`graph/mod.rs`) | dead | dead | alive | alive | `graph/tests.rs` (`:102`, `:120`) |
| `DType::Q8_0` (`graph/mod.rs`) | dead | dead | alive | alive | `graph/tests.rs` (`:104`, `:122`) |
| `Op::Scale` (`graph/ops.rs`) | dead | dead | alive | alive | `op_matrix.rs` (`:262`, `:555`, `:677`), `optiming/tests.rs` |
| `Op::Softmax` (`graph/ops.rs`) | dead | dead | alive | alive | `op_matrix.rs` (`:573`, `:679`), `builder.rs`'s `#[cfg(test)] fn softmax` |
| `Op::Reshape` (`graph/ops.rs`) | dead | dead | alive | alive | `op_matrix.rs` (`:416`, `:625`, `:701`, `:886`) |
| `Op::Permute` (`graph/ops.rs`) | dead | dead | alive | alive | `op_matrix.rs` (`:432`, `:633`, `:704`, `:895`) |
| `Op::BatchMatMul` (`graph/ops.rs`) | dead | dead | alive | alive | `op_matrix.rs` (`:706`, `:937`) |

Each keeps the note naming what would construct it in production (a kernel taking an f16 activation,
an IR node exposing the quantized buffer, a builder scaling in place, …); the `not(test)` cfg is the
part that says *when* it is unused. The mutation direction that matters was checked: a wrong cfg is
exactly what this split would expose, and the stripped after-captures keep all seven out of both test
builds and in both non-test ones.

**The `copystats` coupling, stated because the deletion is only conditionally safe.** The blanket was
on `impl CrossCopyStats`, whose one member (`delta`) already carries `#[cfg(test)]` ([#238]), so the
`impl` is empty in the shippable build and an empty inherent impl draws no diagnostic. Deleting the
blanket is therefore safe *only while `delta` keeps its `#[cfg(test)]`*: if someone un-gates `delta`,
the non-test build must fail — which is the correct outcome, and the reason the blanket is deleted
rather than narrowed (the member-level gate is already the precise spelling).

**The `gguf` attribute-only note.** `src/gguf.rs` `MmapFile::_file` is the one site where "delete the
annotation" means the **attribute line only**: `_file` is the RAII handle that keeps the fd alive for
the mapping's lifetime and must stay. rustc ignores `_`-prefixed fields for `dead_code`, so the
attribute silenced nothing — the stripped capture reports no `gguf.rs` diagnostic at all, which is
the baseline's confirmation rather than a reasoning claim.

**Warning table — the un-stripped twin.** The same four commands on both trees, each analysed from
its own `--message-format=json` capture, with `RUSTFLAGS=--cap-lints=warn` so the crate's
`cfg_attr(not(test), deny(warnings))` cannot truncate the pass. Counting rustc `compiler-message`
diagnostics (so a build script's `cargo:warning=` is not mixed in with them):

| command | `f47e4c3` diagnostics (`dead_code` / other) | `8c3f74b` | multiset |
|---|---|---|---|
| `cargo check --release` | 0 (0 / 0) | **0 (0 / 0)** | equal (empty) |
| `cargo check --release --features cuda` | 0 (0 / 0) | **0 (0 / 0)** | equal (empty); the run's one `warning:` line is `build.rs`'s pre-existing `cargo:warning=` target list, the same line [#243] recorded |
| `cargo check --release --tests` | 68 (7 / 61) | **68 (7 / 61)** | equal |
| `cargo check --release --tests --features cuda` | 79 (5 / 74) | **79 (5 / 74)** | equal |

Both production configurations are warning-free — no rustc diagnostic of any kind. The two `--tests`
configurations are **not** warning-free on `f47e4c3` either: 61/74 of their diagnostics are the
test-code `unused import` / `unused variable` set and 7/5 more are `dead_code` on test helpers, the
"separate, tracked cleanup" Core Convention 5 explicitly scopes out of the non-test gate. These
counts differ from the stripped capture's 73/86 above only because stripping the annotations exposes
5/7 more dead items; the shared 61/74 non-`dead_code` warnings are the same set in both. (An
un-stripped `--tests --features cuda` cargo run therefore prints 81 `warning:` lines: 79 diagnostics
plus cargo's `generated 79 warnings` summary; `--tests` prints 69 = 68 + 1.) The acceptance property
this ticket can honestly claim is therefore **"no new warning and no new `dead_code` in any of the
four configurations"**, which the multiset equality above proves directly; "the test build is
warning-free" is not a property master has, and this record does not claim it.

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-02.**
`cargo test --release` → **480 / 0 / 36** unit + **10 / 0 / 6** integration.
`bash scripts/cuda_test.sh` → **565 / 0 / 42**. No `#[test]` was added, removed or weakened, so no
count row moves and `docs/status.toml` / the `AGENTS.md` rows — including
`projection_base_passed` — are untouched. `cargo fmt --all --check` clean; `check_status.py
--check`, `check_docs_links.py` and `check_source_layout.py` clean.

**Bar named before measuring.** No `#[test]` is added, removed or weakened and the diff is 25
attribute lines plus 4 comments, so the bar is: *the stripped-oracle `(file, name, kind)` item set
and the per-file site-count histogram are unchanged in all four configurations, and the un-stripped
warning multiset is unchanged in all four* — stated before the after-captures and the twin builds
were run, and measured by the two tables above.

**Mutation evidence takes the annotation-shape form here.** There is no behavioural gate to mutate:
an attribute-only diff has no observable effect, and the falsifiable claims are (a) deleting a
redundant `allow` changes no diagnostic once *all* annotations are stripped, and (b) tightening keeps
the item dead in exactly the configuration the `cfg_attr` names. Claim (a) is measured by the
before/after oracle identity (0 item Δ, 0 per-file site Δ, equal diagnostic multisets); claim (b) is
mutation-checkable in the direction that matters and was checked — the after-captures' per-item split
is dead/dead/alive/alive for all 7, which a wrong cfg (e.g. `not(feature = "cuda")`, or a bare
`allow`) would have broken.

**Limits.** (1) macOS is not compiled here: `src/metal.rs`'s two sites are [#255]'s, the macOS half of
`graph/metal_backend.rs` and the macOS branches of the loaders are invisible to all four captures,
and this ticket touches none of them. (2) `--features debug_dump` was not built; no #256 site lives
in `src/dump.rs`. (3) The oracle is a `cargo check`: "live" is a compile-time reference, not runtime
reachability — `Qwen2Model::n_layer` is live precisely because a `MINFER_GRAPH_DUMP` branch
references it, which is the right rule for a warning-free build and not a statement about hot paths.
(4) `cuda_static` was not built (it changes linking only). (5) The cross-platform gate is CI's
`build-macos` plus the three build jobs; the local aarch64 runs above are this box's.

[#138]: https://github.com/yusiwen/minfer/issues/138
[#209]: https://github.com/yusiwen/minfer/issues/209
[#238]: https://github.com/yusiwen/minfer/issues/238
[#243]: https://github.com/yusiwen/minfer/issues/243
[#254]: https://github.com/yusiwen/minfer/issues/254
[#255]: https://github.com/yusiwen/minfer/issues/255
[#256]: https://github.com/yusiwen/minfer/issues/256
[PR #257]: https://github.com/yusiwen/minfer/pull/257


#### Test-infrastructure record (#254, 2026-10-02) — the dead-code ratchet: an annotation-shape guard in `check-docs`, and a stripped oracle in the two building jobs

**The ticket.** The #218 → [#227] → #228/#236/#238–#244 series established that an
`#[allow(dead_code)]` is not a lint silence but a **liveness root**: rustc hides the annotated
item *and its whole call chain*, so one stray annotation can hide a large dead subgraph while
`deny(warnings)` stays quiet (the minimal case the census was built on: without the allow rustc
reports both `struct NeverBuilt is never constructed` and `fn annotated_root is never used`; with
it, neither). The series measured the real dead set by stripping every annotation and reading
rustc's own `dead_code` diagnostics, but that census lived in a scratch script outside the
repository. [#254] productises it as two mechanical layers; it lands as [PR #258]. The workflow
keeps its **seven-job shape**: Layer 1 rides in `check-docs` (no build), Layer 2 is the **last**
step of `test-linux-cpu` (`--config cpu`) and `build-linux-cuda` (`--config cuda`) — the two jobs
that already pay for their compile, so the strip pass is one more `cargo check` over a warm
`target/`.

**Layer 1 — the shape guard (`scripts/check_dead_code_annotations.py`).** It enforces the
annotation *shape*, not liveness, and its docstring says so — that is what keeps it honest. `R1`
a bare `#[allow(dead_code)]` is rejected unless it is one of the 11 sites that predate the ratchet
(`GRANDFATHERED_BARE`, keyed `path:item`; **a stale entry is a failure**, so fixing a site forces
its entry out — the two `src/metal.rs` entries must go when [#255] resolves them). `R2` a
`cfg_attr` must name a cfg predicate: `#[cfg_attr(dead_code, allow(dead_code))]` names the lint,
not a configuration. `R3` the annotated item needs a *reason marker* in its own comment block — a
ticket reference (`#254`), a named consumer/configuration (`tokenizer::tests`,
`not(feature = "cuda")`), or one of the deferred-use phrases [#244] established. `R4` the
rejection text points a test-only item at `#[cfg(test)]` / `tests.rs`, which needs no annotation.
`R3` is a *presence* test: the checker can see that a reason was written, never that it is true.
`--selftest` pins 14 cases (13 shapes plus the stale-entry rule) and runs in `check-docs`.

**Seven annotations had no reason under `R3`.** Each gets a one-line doc-comment reason — no
code and no attribute change — and all seven are reported here because the ticket asks for them:

| site | item | the reason added |
|---|---|---|
| `src/conversation.rs:329` | `TurnOutcome::text` | read only by `conversation::tests`; the streaming path hands each delta to the caller as it is produced |
| `src/device_tier.rs:82` | `QClass::Other` | points at the enum note above: `device_tier::tests` constructs it; a caller classifying a non-K-quant weight type would in production |
| `src/device_tier.rs:106` | `DeviceTier::mmvq_batch_default` | read by `device_tier::tests`; the batch-cap activation (plan §14 R8) would read it |
| `src/device_tier.rs:109` | `DeviceTier::mmvq_batch_by_type` | read by `device_tier::tests`; the batch-cap activation would read the per-class override |
| `src/graph/ops.rs:22` | `AttnMode::Mha` | what would construct it: a multi-head-attention builder (`n_head_kv == n_head`) |
| `src/graph/ops.rs:136` | `Op::Permute` | the deferred note: a builder that wants a transposed view rather than a copied/reshaped buffer |
| `src/graph/ops.rs:143` | `Op::BatchMatMul` | the `FusedOp::BatchMatMul` note: the single-output IR cannot express a batched matmul |

(`mmvq_batch_default`'s doc already mentioned “kernels”, which the marker test happens to accept;
its reason was added anyway so the two fields of the pair read the same.)

**Layer 2 — the oracle ratchet (`scripts/check_dead_code_oracle.py` +
`docs/dead-code-baseline.toml`).** It copies the tree to a scratch directory (the checked-out
tree is never modified), replaces **every** `allow(dead_code)` — bare, `cfg_attr`, a combined list
(dropping only `dead_code`) and the inner `#![...]` form — with a line-preserving marker, then
runs `cargo check --release [--features cuda] --message-format=json` with
`RUSTFLAGS=--cap-lints=warn` (never `deny(warnings)`, which turns the lint into an error and can
truncate the pass). It takes **every `src/` span** of every `dead_code` diagnostic — the primary
span alone undercounts ~2.6×, the first census's error — and every `(name, kind)` the diagnostic's
message names (`fields `a`, `b` …` is two `field` items, `variants `X`, `Y` …` two `variant`
items), and compares that set with the manifest. An **addition fails**, printed with `file:line`,
the diagnostic and a paste-ready entry; a **removal** is informational; a file move is a note.
The strip cannot pass silently: an unrecognised spelling (a multi-line attribute, a
`clippy::dead_code` path), a surviving `dead_code` in a code line, a zero-annotation strip, or a
capture with no `dead_code` diagnostic at all is an exit-2 infrastructure error.

The scratch copy shares the calling tree's `target/` by default, and that is the point: on a warm
tree the cpu configuration costs **1.5 s** (only the crate is rechecked) while the cuda one — whose
non-test artifacts the job has not built — costs about two minutes over the job's already-warm
dependency cache. Four CI steps, seconds-to-minutes inside jobs that compile the tree anyway.

**Counts, both architectures.** `sites` is every `src/` span of every `dead_code` diagnostic;
`items` is every `(name, kind)`:

| configuration | `dgxspark (aarch64, GB10 sm_121)` | `--target x86_64-unknown-linux-gnu` |
|---|---|---|
| `cargo check --release` (cpu) | **32 diagnostics / 60 sites / 46 items** | 32 / 60 / 46 |
| `cargo check --release --features cuda` | **17 / 50 / 34** | 17 / 50 / 34 |

Both columns reproduce the [#256] record's recorded cpu **60 / 46** and cuda **50 / 34** exactly,
and the two architectures are **identical at the `(name, kind)` level**. The x86_64 column is a
local cross-check the box can make because `cargo check` needs no x86 linker — the rustc invocation
carries `--target x86_64-unknown-linux-gnu` and `target_arch="x86_64"` — but it is not a run on
the runner. Since the sets agree, the manifest is the union of one set and carries **no `arch`
field**; the authoritative architecture validation is the PR's own CI run, which passes no
`--target` (the first run's oracle output is the confirmation this record's table is checked
against).

**Confirmed on the runner.** The PR's first CI run is green 7/7 with zero code annotations (the
single `build-macos` annotation is GitHub's platform `notice` about macOS runner capacity). On the
**x86_64** runner `test-linux-cpu` printed `stripped 43 annotation(s) … dead_code diagnostics: 32
src/ spans (sites): 60 items: 46 … baseline [cpu]: 46 entry/entries — 0 addition(s), 0 removal(s);
PASS` in 17 s, and `build-linux-cuda` printed `stripped 43 … 17 … 50 … 34 … 34 entry/entries —
0 addition(s), 0 removal(s); PASS` in 2 m 47 s over its warm dependency cache. Those are the same
sets as the aarch64 column, so no `arch` field was needed and the manifest needed no second seed;
`check-docs` printed `annotation shapes clean (11 grandfathered bare site(s))`, the Layer-1
selftest's `14 cases pass` and the oracle's `strip, item and capture cases pass`.

**The manifest (`docs/dead-code-baseline.toml`).** 46 `[[cpu]]` + 34 `[[cuda]]` entries, each
`name` / `kind` / `file` / `reason`, plus a top-level `macos = "unjudged"`. The update rule is
written into the file: an addition is a decision — make the item live, delete it, or add the entry
**in the same PR** with a one-line reason — and the checker prints the entry to paste; a removal
is informational. `--print-toml` regenerates the block (reusing the existing reasons), and
`--selftest` pins the strip, the item derivation and the capture parser in `check-docs`, so
regression in the strip is caught without cargo.

**The blind spots, written into the manifest rather than discovered later.** (1) macOS is not
compiled on Linux: a macOS-only module is invisible, and a cross-platform item whose only caller
sits in a `#[cfg(target_os = "macos")]` **test** block *looks* dead here while it is live there —
`ModelDef::forward_graph` is exactly that case, and its reason names the test
(`models::qwen2::graph::tests::graph_metal_matches_cpu_logits`). `build-macos` only type-checks,
so `macos = "unjudged"` records the gap and the two `src/metal.rs` sites stay with [#255]. (2) The
oracle is a `cargo check`: “live” is a compile-time reference, not runtime reachability. (3)
`--features debug_dump` and `cuda_static` are not covered.

**Fixture evidence — both layers can fail.** Layer 1: appending a bare
`#[allow(dead_code)] fn layer1_bare_allow_probe() {}` to a scratch copy makes
`check_dead_code_annotations.py` exit 1 with two problems (the bare form; no reason); swapping it
for a reason-less `#[cfg_attr(not(test), allow(dead_code))]` exits 1 with the “no reason” problem
alone; removing the probe restores the clean run. Layer 2: a synthetic `oracle_mutation_probe`
function added to a scratch copy makes the oracle exit 1 in **both** configurations, naming
`oracle_mutation_probe (fn) at src/vec_ops.rs:1349` and printing the manifest entry; pasting that
entry into the scratch manifest's `[[cpu]]` array makes the same command pass
(`47 items, 47 entries — 0 addition(s), 0 removal(s)`), which is the update path in one diff.
Neither fixture is committed.

**Bar named before measuring.** Layer 1: *the shape audit passes on the tree as it stands, and
each fixture mutation exits non-zero*. Layer 2: *the stripped oracle reproduces the recorded cpu
60 / 46 and cuda 50 / 34 and the manifest compares equal (0 additions, 0 removals) in both
configurations on both architectures*. Both were stated before the after-runs and are met by the
tables above.

**Mutation evidence takes the fixture form here.** There is no behavioural gate to break: an
attribute-only diff plus two new checkers have no runtime effect, so the falsifiable claims are
that Layer 1 rejects the two shapes it forbids and that Layer 2 fails on a newly hidden item. Both
are the transcripts above, and the oracle's own strip is mutation-checked by construction — the
`oracle_mutation_probe` run strips **44** annotations (43 + the probe) and the item set moves 46 →
47 exactly.

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-02.** `cargo test --release` →
**480 / 0 / 36** unit + **10 / 0 / 6** integration; `bash scripts/cuda_test.sh` → **565 / 0 / 42**
(unit), integration **7 + 3 passed / 0 failed / 6 ignored**. No `#[test]` was added, removed or
weakened — the diff is two scripts, a TOML manifest, CI steps and seven doc comments — so no count
row moves and `docs/status.toml` / the `AGENTS.md` rows are untouched. `cargo fmt --all --check`,
`check_status.py --check`, `check_docs_links.py` and `check_source_layout.py` are clean.

**Limits.** (1) The x86_64 column is a cross-compile, not a run on the runner; the CI job is the
confirmation. (2) The one-line `reason` strings are as good as the annotation notes they reuse:
the oracle proves an item is *dead*, never that the reason for keeping it is still true. (3) A
`dead_code` diagnostic whose only span is outside `src/` is ignored by construction — this
configuration compiles the crate, not `tests/`. (4) A *new* annotation that hides a large subgraph
is caught the next time the oracle runs, not by Layer 1; that division of labour is the design.

[#227]: https://github.com/yusiwen/minfer/issues/227
[PR #258]: https://github.com/yusiwen/minfer/pull/258

#### Test-infrastructure record (#252, 2026-10-02) — the legacy `KVCache` and `ModelDef::forward`'s `&mut KVCache` parameter are deleted

**The ticket.** [#252] is [#244]'s decision 6, spun out because it is a **trait-signature refactor**
rather than a dead-code cleanup: a dead-code ticket may delete an item, but it may not move a public
trait's signature. [#244] had already deleted `KVCache`'s storage, leaving `src/cache.rs` a 31-line
storage-free marker whose only job was to keep `ModelDef::forward`'s `kv: &mut KVCache` argument
spelled the same. The argument was **provably unread**: the graph path owns KV in the allocator's
persistent regions, both graph entry points already bound the parameter as `_kv`
(`models/qwen2/graph.rs:435`, `models/qwen3/graph.rs:380`), and every implementation only threaded it
through. [#252] lands as [PR #259].

**What was deleted, and the proof the parameter was dead.** The inventory was re-grepped on the
branch point (`bef4bd9`), not trusted from the ticket: `git grep -n 'KVCache' src/` plus
`git grep -n '\.forward(' src/` names exactly the sites below, and nothing else.

| what | site before | change |
|---|---|---|
| the marker type | `src/cache.rs` (31 lines: `pub struct KVCache;`, `KVCache::new`) | **file deleted** |
| its declaration | `src/main.rs:13` `mod cache;` | deleted (with the file, so `check_source_layout.py` stays consistent) |
| the trait declaration | `src/models/mod.rs:207` (`kv` at `:211`) | `kv: &mut KVCache` removed; doc comment updated |
| the default `forward_graph` | `src/models/mod.rs:264` (`kv` at `:268`) | `kv` removed from the signature and the delegation |
| Qwen2 impls | `src/models/qwen2/mod.rs:56`/`:97` (`kv` at `:60`/`:101`) | both removed; `use crate::cache::KVCache;` gone |
| Qwen3 impls | `src/models/qwen3/mod.rs:48`/`:89` (`kv` at `:52`/`:93`) | both removed; `use crate::cache::KVCache;` gone |
| graph entry points | `src/models/qwen2/graph.rs:435`, `src/models/qwen3/graph.rs:380` | `_kv: &mut KVCache` removed from `forward`; imports gone |
| CLI construction | `src/main.rs:1227` (`KVCache::new`) | deleted, with the then-unused `n_kv_embd`/`n_layer` locals |
| CLI forwards | `src/main.rs:1478`, `:1788` | argument dropped |
| test construction | `src/models/qwen2/graph/tests.rs:2978/3019/3023` | `KVCache::new` calls and arguments dropped |
| test mock | `src/server/batch/tests.rs:1794` (`FailingForward::forward`) | `_kv` parameter dropped |

`src/conversation.rs` was verified rather than assumed: `GraphEngine::forward`
(`src/conversation.rs:43`) is its **own** 3-argument method and the model calls go through
`forward_graph_cached` (`:237`), neither of which ever took a `KVCache` — its diff is empty.
`src/graph/cache.rs` (`pub mod cache;` at `src/graph/mod.rs:29`, the `GraphCache`) is untouched; only
the **other**, legacy `cache` module went. `Cargo.toml`, `src/graph/mod.rs` and every `GraphCache`
call site are unchanged.

**Zero behaviour change — stated before the suites ran.** No implementation read the argument (both
graph entry points bound it `_kv`), `KVCache` had no fields since [#244], and nothing else names the
type. The edit is therefore signature-only: the graph every caller builds, the KV the allocator owns,
the token stream and the logits are bit-identical. The suite counts are the check — no `#[test]` was
added, removed or weakened, so the rows below are unchanged rather than re-baselined.

**Warning table — four configurations, before and after.** Same method as [#256]: four
`cargo check --release --message-format=json` captures on the pinned `1.97.1` toolchain with
`RUSTFLAGS=--cap-lints=warn` (the crate's `cfg_attr(not(test), deny(warnings))` must not truncate the
pass), counting rustc `compiler-message` diagnostics (a build script's `cargo:warning=` is not mixed
in) and comparing the **whole `(level, code, message)` multiset** by sha256, not only counts:

| command | before `bef4bd9` | after (this PR) | multiset |
|---|---|---|---|
| `cargo check --release` | 0 (0 `dead_code` / 0 other) | **0 (0 / 0)** | equal (empty) — sha `e3b0c442…` both |
| `cargo check --release --features cuda` | 0 (0 / 0) | **0 (0 / 0)** | equal (empty) — sha `e3b0c442…` both |
| `cargo check --release --tests` | 68 (7 / 61) | **68 (7 / 61)** | equal — sha `cc1f8468…` both |
| `cargo check --release --tests --features cuda` | 79 (5 / 74) | **79 (5 / 74)** | equal — sha `6b87d17c…` both |

Both non-test configurations are warning-free in the strict sense (the plain `cargo build --release`
and `cargo build --release --features cuda`, without `--cap-lints`, both exit 0 under
`deny(warnings)`; the cuda run's one `warning:` line is `build.rs`'s pre-existing `cargo:warning=`
target list). The two `--tests` configurations carry #256's recorded 68/79 pre-existing test-code
warnings, which Core Convention 5 explicitly scopes out. The honest claim is therefore **"no new
warning in any of the four configurations"**, and the sha equality is stronger than the count
equality: not one message was added, removed or moved.

**The ratchet's first real exercise — it said PASS, 0 additions, 0 removals.** [#254]'s stripped
oracle (`scripts/check_dead_code_oracle.py`) ran in both configurations against
`docs/dead-code-baseline.toml`:

```
[cpu]  stripped 43 annotation(s); dead_code diagnostics: 32  src/ spans (sites): 60  items: 46
       baseline [cpu]: 46 entry/entries — 0 addition(s), 0 removal(s); PASS
[cuda] stripped 43 annotation(s); dead_code diagnostics: 17  src/ spans (sites): 50  items: 34
       baseline [cuda]: 34 entry/entries — 0 addition(s), 0 removal(s); PASS
```

Those are **exactly** the recorded cpu 60/46 and cuda 50/34 that [#254]/[#256] measured, and the
manifest needed no edit — no entry added, none removed, no `arch` field. That the ratchet is silent
here is itself the finding, and it is the expected one: the oracle strips `allow(dead_code)`
annotations and compiles, and `KVCache` carried **no annotation** — it was *live by reference*, kept
alive by the very signature this PR moves. The ratchet's subject is newly **hidden** code; this PR
deletes a live-by-reference item, so its item set is unchanged. The manifest never listed `KVCache`
(`grep -n KVCache docs/dead-code-baseline.toml` is empty), which is why there is no removal to
report. The two runs also confirm the oracle survives a file deletion and a `mod` removal without an
infrastructure error (the "every `.rs` is declared" half of `check_source_layout.py` is what would
have caught a half-done deletion).

**Counts (rule 5), box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-02.**
`cargo test --release` → **480 / 0 / 36** unit + **10 / 0 / 6** integration;
`bash scripts/cuda_test.sh` → **565 / 0 / 42** unit + **7 + 3 passed / 0 failed / 6 ignored**
integration; `PARALLEL=0 scripts/real_model_gates.sh` (CPU) → **36 / 0**.
No `#[test]` was added, removed or weakened, so no count row moves and `docs/status.toml`, the
`AGENTS.md` rows and `projection_base_passed` are untouched. `cargo fmt --all --check`,
`check_status.py --check`, `check_docs_links.py`, `check_source_layout.py` and
`check_dead_code_annotations.py` are clean.

**Docs.** `AGENTS.md`'s layout block loses its `cache.rs` line; `docs/ARCHITECTURE.md` loses the
module-table row and its "remains only as CLI plumbing" sentence; `docs/COMPUTE-GRAPH-DESIGN.md` §10's
trait sketch loses `kv` and the other two `src/cache.rs` mentions are corrected;
`docs/METAL-BACKEND-DESIGN.md`'s legacy-surface paragraph is past-tensed;
`docs/CPU_OPTIMIZATIONS.md`'s P2/P4 "current state" lines stop citing the deleted file/parameter;
`docs/OPENAI-CHAT-API-PLAN.md`'s revision note and slot structure say the ignored argument was
deleted; the walkthrough docs 01/03/09/13 carry dated forward notes (history is not rewritten); and
the move of this record closes the two `#252` forward notes in the [#244] and #244-escalations
records above.

**Bar named before measuring.** No `#[test]` is added, removed or weakened and the parameter was
never read, so the bar is: *the four warning multisets are unchanged, the stripped-oracle
`(site, item)` set is unchanged in both configurations, and the three suite counts are unchanged* —
stated before the after-captures, the oracle runs and the suites, and measured by the tables above.

**Mutation evidence takes the warning-multiset form here.** A signature refactor has no runtime
behaviour to mutate and no new gate to break — the falsifiable claim is "the edit is inert", and the
sharpest inertness test available is that the compiler sees **exactly** the same message multiset in
all four configurations, which cannot hold if the deletion moved a live path (it would surface as a
new `dead_code`/`unused` diagnostic) or left a dangling reference (it would surface as a resolved
error). The oracle's *stripped* captures are the second arm: if `KVCache` had been merely hidden
rather than removed, or if the deleted parameter had a reader left behind, the `(site, item)` set
would have moved; it did not, in either configuration. A behavioural mutation is not available
because there is no behaviour in the diff, and inventing one would be the "passed for the wrong
reason" failure `docs/GATE-CONTRACT.md` warns about.

**Limits.** (1) macOS is not compiled here: `graph_metal_matches_cpu_logits` is the one test that
called `forward_graph`, and its call site is inside a `#[cfg(target_os = "macos")]` **test** block,
so this box proves the edit compiles on Linux and CI's `build-macos` job (a type-check, not a test
run) must prove the macOS half compiles. (2) The oracle is a `cargo check`: "live" is a compile-time
reference, as [#254] states. (3) `--features debug_dump` and `cuda_static` were not built; neither
names the removed type. (4) The `--tests` warnings are the two configurations' pre-existing sets
(68/79), not a claim that master's test build is warning-free.

[PR #259]: https://github.com/yusiwen/minfer/pull/259

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
| F2 | 15 | GBNF-style grammar + JSON-schema constrained decoding · [#47](https://github.com/yusiwen/minfer/issues/47) — **DONE 2026-09-24** · follow-ups [#125](https://github.com/yusiwen/minfer/issues/125) (refused constructs), [#126](https://github.com/yusiwen/minfer/issues/126) (mask cost) | M | dgxspark |
| F3 | 16 | Sampler set: min-p, typical, XTC, DRY, mirostat, logit bias · [#48](https://github.com/yusiwen/minfer/issues/48) — **DONE 2026-09-24** | M | dgxspark |
| F4 | 12 | Backend registry (drop the compile-time enum) · [#57](https://github.com/yusiwen/minfer/issues/57) — **DONE 2026-09-24** · the per-device KV-format capability [#87](https://github.com/yusiwen/minfer/issues/87) needs is now a **used** registry field (`BackendCaps::reads_packed_kv`), not a hardcoded CPU test | M | dgxspark |
| F5 | 14 | Async cross-backend copy + events · [#58](https://github.com/yusiwen/minfer/issues/58) — **DONE 2026-09-24** · CUDA's boundary copy is an `cudaMemcpyAsync` D2H into a pinned slab plus an event, waited on once at a documented synchronization point; the CPU is a registered synchronous no-op and Metal declines (unported) · follow-ups [#137](https://github.com/yusiwen/minfer/issues/137) (Metal), [#138](https://github.com/yusiwen/minfer/issues/138) (deferred wait — **DONE 2026-10-04**, see the F5 S2 record) | M | dgxspark (CUDA) |
| F6 | 22 | Quantizer tooling (`convert-hf-to-gguf`, `quantize`, `split`) · [#49](https://github.com/yusiwen/minfer/issues/49) — **DONE 2026-09-24** · a GGUF v3 *writer* (`gguf_write.rs`), byte-exact weight encoders (`quantize.rs`), an HF converter that passes the strict loader (`convert.rs`), the three subcommands (`tooling.rs`), and a real download size check — follow-ups [#140](https://github.com/yusiwen/minfer/issues/140) (K-quant encoders), [#141](https://github.com/yusiwen/minfer/issues/141) (f16 on the device), [#142](https://github.com/yusiwen/minfer/issues/142) (bf16) | L | dgxspark |
| F7 | 19/20 | Chat-template fidelity + tokenizer generality · [#50](https://github.com/yusiwen/minfer/issues/50) — **DONE 2026-09-24** · follow-ups [#132](https://github.com/yusiwen/minfer/issues/132) (NFC + the remaining pre-tokenizer rules) and [#133](https://github.com/yusiwen/minfer/issues/133) (`--chat-template`, `strftime_now`) | M | dgxspark |
| F8 | 25 | **Metrics/observability** (`/metrics`, KV occupancy, queue depth, per-op timing under a flag, graceful drain). Item 25 was the only member of the A-era batch (items 23/24/26/27/28 -> A1/A2/A7/A5/A6) with no ticket; it is independent of the critical path, hence this table · [#51](https://github.com/yusiwen/minfer/issues/51) — **DONE 2026-09-24**, both real-model gates **device-verified on GB10 sm_121 2026-09-24**; the serial `#[ignore]`d set it left red (**#123**) is green as of 2026-09-24 (**22 passed / 0 failed**) | M | dgxspark |

F1 is the only item in this plan that **cannot be verified on dgxspark**
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
| `minfer_worker_stalled_total` | counter | events | the worker's counted no-progress bound tripped and ended the loop, answering every live and queued request `500` (**added by [#196](https://github.com/yusiwen/minfer/issues/196)**) |
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

**End-to-end run (manual, on dgxspark, CPU, Qwen2.5-0.5B Q4_0, `MINFER_BATCH=1`).**
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
generation. Reproduced on dgxspark (CPU, 0.5B Q4_0, `MINFER_BATCH=1`,
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
runs above were performed by hand on dgxspark and are recorded here as manual
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
first use landed as [#138](https://github.com/yusiwen/minfer/issues/138)
(2026-10-04; the F5 S2 record is below). (c) **Only
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

### F5 S2 — the cross-backend wait is deferred to the consumer (#138) — **DONE 2026-10-04**

**The ticket.** [#138](https://github.com/yusiwen/minfer/issues/138) is F5's overlap follow-up: F5
enqueued one staging copy per `Split::inputs` entry and then waited on **every** one of them at the
split entry, so no copy could overlap anything, and the boundary also paid a full
`cudaStreamSynchronize` to retire a producer whose work the copies were already stream-ordered
behind.

**What landed.**

- `src/graph/scheduler.rs` — the boundary calls `GraphAllocator::retire_backend(previous)` instead of
  `sync_backend(previous)`, its phase-B loop is gone, the node loop resolves each source through
  `GraphAllocator::cross_input_ready`, and `drain_cross_pending()` runs after the last split.
- `src/graph/backend.rs` — a new `Backend::retire`, whose default body is `synchronize` (Metal's
  boundary *is* a submission, so it still blocks there) with the reason the CUDA override is safe.
- `src/graph/cuda_backend.rs` — `CudaBackend::retire` closes the capture window and drops the MMQ
  memoization **without** the `cudaStreamSynchronize`; `close_capture_or_sync(block)` is that one
  difference; `CudaBackend::cross_inflight_peak()` is the device-side overlap metric.
- `src/graph/alloc.rs` — `cross_input_ready` (the deferred wait), `drain_cross_pending`,
  `retire_backend`, and a `copy_across` that is idempotent while an entry is in flight.
- `src/graph/copystats.rs` — `CrossCopyStats::deferred_waits`.

**The reading of the ambiguous acceptance line, recorded here and in the issue comment.** "A boundary
with several staged inputs waits **once**, at the first actual use, not once per input at the split
entry" is read as *each staged entry's single wait is issued at its first use*, not as *one wait per
boundary*: F5's documented contract is one phase-B call per phase-A copy
(`BACKEND-REGISTRY-DESIGN.md` §11.2), and the ticket's fourth acceptance line keeps
`GraphAllocator::cross_input`'s refusal of a pending entry, which a per-boundary collapse of the wait
count would have to weaken. The counters show the deferral: `deferred_waits == copies` on the
deferred path.

**The finding the deferral exposed (fixed in the same commit).** With the wait deferred, the same
`(graph, node, destination)` can be staged by **two** boundaries of one execution while the first
copy is still in flight — a state unreachable under F5, which always awaited before re-enqueuing.
Re-issuing it duplicated the transfer *and* pushed a second `CrossPending` record into `CudaBackend`
that the allocator's single pending key would never wait on: a leaked pinned slab and event. Measured
on the 0.5B gate before the fix: **56 copies / 47 waits**, 18 in-flight re-requests and 6 drains of 4.
`copy_across` now returns early while its entry is pending (same staging buffer, one unchanged source
node), so the contract holds again and `copies` means *unique transfers per execution*.

**Measured acceptance (box `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04).** The bar was named from
the F5 record (34 → 27 stream syncs, 7 → 0 blocking D2H). On this box the *pre-change* tree at
`6b6d94f` measures **21** stream syncs for the same gate and mode (the ticket's 27 is the
2026-09-24/27 measurement), so the same gate's printed output is the before/after:

| | F5 (measured at `6b6d94f`) | #138 |
|---|---|---|
| staging copies (`copies`) | 56 | 47 (9 in-flight re-requests deduped) |
| waits (`waits`) | 56 | 47 |
| of which deferred to a consumer read | 0 | 35 |
| blocking device→host copies | 0 | 0 |
| device-level blocking readbacks | 0 | 0 |
| **full stream syncs** | **21** | **0** |
| max \|Δlogit\| vs `MINFER_SYNC_COPIES=1` | 0 | **0** |

The cheap device gate (`cuda_backend::tests::a_split_graph_waits_once_per_staged_copy_and_stays_bitwise`,
3 nodes CPU → CUDA → CPU) reads: async `copies=2 waits=2 deferred=2 blocking=0 async_host=1
event_syncs=1 readbacks=0 syncs=0` against the F5 reading `syncs=1`; sync mode `copies=2 waits=2
blocking=1 readbacks=1 syncs=1` (F5: 2).

The new overlap gate
(`cuda_backend::tests::a_boundary_with_several_staged_inputs_defers_its_waits`) makes the device split
produce **two** values the CPU split consumes, with an independent CPU node between the boundary and
the first staged read. Two deterministic device-side metrics:

- **in-flight copies** — the pinned-slab high-water mark is **2** for the deferred boundary, while the
  F5 enqueue-then-wait discipline, driven by hand on the same buffers in the same test, cannot exceed
  **1** (the wait releases the slab before the next copy takes one). This is the "enqueuing copy N+1
  while copy N is in flight" measurement.
- **host stalls** — `stream_syncs` is **0** for the deferred boundary and **2** for the synchronous
  reference over the same two device→host copies; `copies == waits == 3` and
  `deferred_waits == 3`, bitwise against the reference.

The gate also asserts the dedup directly: two `copy_across` calls for one pending entry count as
**one** copy and owe **one** wait.

**Mutation evidence (reverted; the tree was restored byte-identical).** The node loop's resolver was
replaced with a bare `cross_input` — i.e. the deferral removed, the F5 read path restored:

```text
graph::cuda_backend::tests::a_split_graph_waits_once_per_staged_copy_and_stays_bitwise
  panicked: "staged cross-backend input 0 for Cuda was read before its boundary wait: every
  copy_across owes one await_cross (F5, #58)"
graph::cuda_backend::tests::a_boundary_with_several_staged_inputs_defers_its_waits
  panicked: "staged cross-backend input 0 for Cuda was read before its boundary wait: …"
graph::scheduler::tests::a_staged_boundary_input_is_waited_on_at_its_first_use
  panicked: "staged cross-backend input 1 for CPU was read before its boundary wait: …"
models::qwen2::graph::tests::async_cross_copies_never_block_and_stay_bitwise_identical
  panicked: "staged cross-backend input 3 for Cuda was read before its boundary wait: …"
```

**Suite counts.** `cargo test --release` **481 / 0 / 36** unit + **10 / 0 / 6** integration (baseline
480 / 0 / 36; `graph::alloc::tests::the_drain_waits_on_a_staged_entry_nothing_read` is the new
feature-independent gate). `scripts/cuda_test.sh` **567 / 0 / 42** (baseline 565 / 0 / 42; the same
alloc gate plus `cuda_backend::tests::a_boundary_with_several_staged_inputs_defers_its_waits`; no
`#[ignore]`d test moved). `FEATURES=cuda scripts/real_model_gates.sh` **42 / 0** on the cached 0.5B
and **42 / 0** with `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf`. `cargo build --release
--features cuda` and `cargo test --release --features cuda --no-run` are clean, and
`cargo fmt --all --check` is clean on the pinned 1.97.1 toolchain.

**Honest scope.** (a) **Metal is still unported** ([#137](https://github.com/yusiwen/minfer/issues/137)):
its `copy_cross` declines, `Backend::retire`'s default keeps its boundary blocking, and its copies
still count as blocking — no half-written blit/event code. (b) **The `cudaStreamWaitEvent`
device-consumer arm is not reached**: the ticket's first bullet asks for it, but the only pair that
could express a device destination would need two device backends, `copy_across` early-returns on a
same-backend pair, and CUDA→Metal (macOS + CUDA) declines phase A — so a call site would be
unreachable code, exactly what Core Convention 5 asks about rather than silences. The *observable*
requirement of that bullet — "a device consumer does not block the host" — is met by the CPU→device
direction, whose fill is already stream-ordered on the destination pool's own stream (§11.3). The two
grandfathered bare sites therefore **stay dead in every configuration**, and this PR leaves
`GRANDFATHERED_BARE` and `docs/dead-code-baseline.toml` unchanged; the stripped oracle confirms it
below. (c) **True cross-split overlap is still not claimed** — the split loop remains sequential;
what the deferral buys is that the wait happens where the data is read, not where it was produced.

**The dead-code ratchet, measured.** `python3 scripts/check_dead_code_annotations.py` reports the
same **11** grandfathered bare sites (this PR adds no annotation and tightens none), and the stripped
oracle run with `RUSTFLAGS=--cap-lints=warn` and `RUSTUP_TOOLCHAIN` unset reports **0 additions** in
both configurations — the `(name, kind)` set difference against
`docs/dead-code-baseline.toml` is empty, so neither site became live and neither entry went stale.

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
(c) bf16 output was refused at F6 and **landed in
[#142](https://github.com/yusiwen/minfer/issues/142)** (`--outtype bf16`, 2-D
bf16 / 1-D f32, round-to-nearest-even, byte-identical per tensor to
`llama-quantize --pure <f32>.gguf … BF16`), together with the CPU bf16 weight
path; CUDA and Metal still do not register bf16
([#208](https://github.com/yusiwen/minfer/issues/208)).
(d) The exactness claims are named per step: f16/f32 copies and f16→f32,
bf16→f32 are bit-exact; **bf16→f16 is exact in the mantissa but not in the
exponent range** — it can overflow to inf, and below f16's smallest normal it
rounds onto the subnormal grid (measured: 123 024 values on the 0.5B checkpoint,
[#142](https://github.com/yusiwen/minfer/issues/142)); **f32→f16 is not exact**
and neither is **f32→bf16** (RNE). (e) The HF reference is llama.cpp's converter
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
  cannot be verified from dgxspark (no Mac; CI's `build-macos` compiles the crate
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
forms are asserted bit-identical on dgxspark; the argument that they must be
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
**A separate loader divergence was filed here, and is fixed in F6d**
([#167](https://github.com/yusiwen/minfer/issues/167)): at the time the qwen3 loader lacked both
the q4_K dsc plane this record gates and the f16 registration branch #141 gave qwen2, so a q4_K
Qwen3 ran the in-kernel scalar dsc decode and an f16 Qwen3 model dropped to the CPU on CUDA (read
from the loader then; device-verified in F6d, 2026-09-25).

### F6d — the qwen3 loader shares qwen2's registration rule: f16 + the q4_K dsc plane (#167) — **DONE 2026-09-25**

**What landed.** [#167](https://github.com/yusiwen/minfer/issues/167), the loader divergence F6c filed
((d) above). Each loader carried its own copy of the `#[cfg(feature = "cuda")]` per-tensor
registration block, and the copies had drifted twice:

- the qwen3 copy had **no `register_weight_q4k_dsc` call** (r59), so a q4_K Qwen3 weight kept
  `mmq_raw_nb_bt`'s in-kernel scalar dsc decode — correct, but the r59 prefill win was not available
  to Qwen3 even though the same kernel consumes both models;
- it had **no `TensorType::F16` branch** (#141), so an f16 Qwen3 weight was never registered and the
  all-or-nothing `weights_on_cuda` gate dropped the whole model to the CPU even on a build with the
  f16 kernels. Registering would not have been enough on its own: the **qwen3 graph's own type gate**
  (`Qwen3Graph::weights_on_cuda` → `matmul_t_ok`/`embed_t_ok`) was missing `F16` as well.

**The shared rule.** `src/models/weight_reg.rs` now owns the dispatch. `cuda_weight_reg` is the pure
decision — no `CudaState`, no environment (the r59 dispatch gates are passed in), so CI's CPU job
*runs* its tests exactly like `src/q4k_dsc.rs` — and the `cuda`-gated `register_cuda_weight` carries
it out. Both `models/qwen2/loader.rs` and `models/qwen3/loader.rs` call it for every tensor the E5
plan puts on the device, so the type coverage has one authority and cannot diverge by edit. It
carries: the quantized set, the F16 raw branch, F32 (1-D norms/biases vs 2-D matmul weights), the
Q6_K padded repack, the q8_0 p32 split plane, the q4_K `W_dsc` plane under
`q4k_dsc_plane_admitted`, and the `clear_mmq_nb_bt_only` rule. The **Metal** per-tensor blocks were
deliberately *not* folded in: Metal's admitted set is different and an f16 weight must stay refused
there ([#164](https://github.com/yusiwen/minfer/issues/164)), so the extraction is CUDA-only and
Metal's per-loader blocks are untouched (CPU/Metal behaviour unchanged is part of the acceptance).
`Qwen3Graph::weights_on_cuda` gained `TensorType::F16` in `matmul_t_ok` and `embed_t_ok`, matching
qwen2's list; `CudaState::q4dsc_plane_for` was added so a gate can read the same pointer-keyed map
`mmq_raw_nb_bt` reads.

**Audit finding folded in.** The extracted F32 arm used to test `tensor.shape.len() == 2`, but
`Tensor::shape` is a `[i64; 4]`, so the test was **always false** in both loaders and the
NB-BT-only flag was never cleared for a 2-D f32 matmul weight. The shared rule takes the real rank
(`gguf_write::ggml_n_dims`) instead. It can only disable the mode-2 skip-write MMQ optimization
(never change a result), and no model in the gate set has a 2-D f32 matmul weight — pinned by
`only_a_2d_f32_weight_clears_the_nb_bt_flag`.

**Test models.**

| Model | Provenance | Size |
|---|---|---|
| `/tmp/f167-work/qwen3-q4k.gguf` | HuggingFace **`unsloth/Qwen3-0.6B-GGUF`** `Qwen3-0.6B-Q4_K_M.gguf` (the official `Qwen/Qwen3-0.6B-GGUF` repo ships only Q8_0); a real Q4_K_M mix — 168 q4_K + 29 other quantized 2-D tensors | 396 705 472 B |
| `/tmp/f167-work/qwen3-f16.gguf` | `llama-quantize --allow-requantize <cached `Qwen3-0.6B-Q8_0.gguf`> … F16` — 2-D f16, 1-D f32 | 1 198 182 048 B |
| `/tmp/f167-work/qwen3-q4_0.gguf` | `minfer quantize --type q4_0` of the cached Q8_0 — the **ratio-equal** negative control (q4_0's bytes/element is exactly q4_K's) | 419 245 728 B |
| `/tmp/f167-work/qwen3-f16-minfer-quantize.gguf` | `minfer quantize --type f16` of the cached Q8_0 — **not usable**, see below | 1 198 050 976 B |

`minfer quantize --type f16` was tried first, as the ticket suggested, and does **not** work: it is a
pure element cast, so it writes the 1-D norms as f16 too, and the engine's f16 contract requires 1-D
**f32** (the CPU RMSNorm reads `Tensor::data_f32`, which asserts F32). Loading that file panics at
`src/tensor.rs:277` — `data_f32 on 'blk.0.attn_norm.weight' of type F16`. The new gate asserts
1-D-stays-f32 (which is what caught it) and the f16 model was then produced with `llama-quantize`,
which follows llama.cpp's "except 1d tensors" rule. The tooling gap is filed as
[#169](https://github.com/yusiwen/minfer/issues/169).

**Measured acceptance (GB10, sm_121, CUDA 13.0, driver 580.178.04).**

| Criterion | Result |
|---|---|
| `f167_qwen3_q4k_registers_the_dsc_plane_exactly` — planes vs the GGUF index | **168 planes / 95 420 416 B**, exactly the index's admissible set (168 q4_K, 29 other quantized 2-D); the kernel's own pointer-keyed lookup finds every one; the q8_0 and **q4_0** negative arms register **0** |
| `f167_f16_qwen3_weights_run_on_the_cuda_device` | `device() == Cuda`; 28/28 blocks + embed/output on the device (1137.0 MiB); **197 f16 matmul + 1 f16 embed node** assigned `Backend::CUDA`; device-vs-CPU max \|Δlogit\| **8.92e-3** (mean 1.25e-3) / **4.46e-4** relative at max \|logit\| 19.99; greedy `[12095, 13, 576, 6722]` identical, asserted at ≤ 0.05 and ≤ 1e-3 |
| `cargo test --release --features cuda -- --test-threads=1` | **521 passed / 0 failed / 36 ignored** (baseline 516 / 0 / 34) |
| `FEATURES=cuda scripts/real_model_gates.sh` (0.5B and Qwen3-0.6B-Q8_0 configs) | **36 passed / 0 failed** both (baseline 34 / 0) |
| `compute-sanitizer --tool memcheck` over the serial CUDA unit suite | **0 errors** over 521 passed / 0 failed / 36 ignored (353.44 s) |
| `cargo test --release` (CPU) | **452 passed / 0 failed / 32 ignored** unit (baseline 447 / 0 / 30; +5 pure `weight_reg` tests, +2 `cuda`-gated ignored) + **10 / 0 / 6** integration, unchanged |
| `PARALLEL=0 scripts/real_model_gates.sh` (CPU) | **32 passed / 0 failed** (baseline 30 / 0; the two new `#[ignore]`d gates no-op and pass on a CPU build) |
| `rustup run stable rustfmt --edition 2021 --check` on the changed `.rs` | clean (rustfmt **1.9.0-stable**; the pinned toolchain has no `rustfmt` component, CI runs no fmt job) |
| `python3 scripts/check_docs_links.py` | **940 relative links in 184 files**, unchanged |

The device-vs-CPU spread is ~100× the 0.5B f16 gate's (7.34e-5 / 4.0e-6) because Qwen3 runs **four**
norms per layer (`attn_norm` + per-head `q_norm`/`k_norm` + `ffn_norm`) and the CPU rms_norm (8-lane
AVX2 FMA plus an f64 tail, then `1/sqrt`) and the device rms_norm (warp-shuffle f32, then `rsqrtf`)
differ in reduction order and reciprocal-sqrt form; the greedy continuation and the argmax agree, and
a wrong f16 row would move the logits by O(1).

**Gates.** Five pure tests in `src/models/weight_reg.rs` (CI's CPU job *runs* them): the F16 arm;
the q4_K dsc decision with each of its four gates refused independently (r59 gates off, odd `od`,
the q4_0 type control, a longer q8_0 payload) plus a q4_K positive control; the quantized arm's
pre-#167 dispatch (q6_K padded, q8_0 p32, every other quant raw + flag-clear, q4_K never clearing);
the 1-D vs 2-D f32 rank rule; and a three-way variant discriminator. Two `#[ignore]`d real-model
gates: the q4_K plane set (exact names/count against the GGUF index, the kernel's pointer map per
weight, distinct non-null buffers, and the two negatives) and the f16 device path (placement asserted
per node, then the numeric comparison). The plane gate calls `CudaState::init()` itself, so running it
alone does not silently skip for lack of a device — a skip-shaped pass was found and removed during
the mutation campaign.

**Mutations (reverted; files restored byte-identical, `sha256sum`).**
(a) the F16 arm removed from `cuda_weight_reg` → the pure f16 test fails and the device gate fails at
*left: Cpu, right: Cuda*; (b) `F16` removed from `Qwen3Graph::weights_on_cuda`'s `matmul_t_ok` → the
same device-gate failure, so the graph type gate is separately load-bearing; (c) the q4_K admission
bypassed (`q4k_dsc = gates && od % 2 == 0`) → the pure test fails and the real-model gate fails on
its **q4_0** negative arm (planes registered for a ratio-equal type); the q8_0 arm alone could *not*
catch this, because the registry's own payload re-check refuses a longer payload — which is exactly
why the q4_0 arm was added; (d) the plane forced off → the pure test fails and the real-model gate
fails with 0 registered against 168 expected. Each mutation was re-run after the rustfmt pass and
reverted with `sha256sum -c` OK.

**Honest scope.** (a) The **prefill win the dsc plane buys is not measured here** — the gate proves
the plane exists, contains the right weights, and that the kernel's pointer-keyed lookup finds each;
the r59 dsc prefill A/B on Qwen3 is not run. (b) The q4_K test model is a community quant
(unsloth) because the official Qwen Qwen3-0.6B GGUF repo has only Q8_0; it is a real Q4_K_M mix, not
a uniform q4_K, which is stronger for the type gate but means the per-tensor set is that file's.
(c) The f16 test model is `llama-quantize`'s F16 of the cached Q8_0, so its weights are a q8_0
re-quantization, not an HF bf16/f16 checkpoint; it preserves the 2-D-f16/1-D-f32 shape contract and
is the same class of file #141 gated. (d) The f16 device gate's absolute bound (0.05) is looser than
f141's (0.01) for the stated Qwen3 reason; the relative bound stays 1e-3. (e) An f16 Qwen3 file with
**f16 1-D norms** is not loadable (CPU panic; on CUDA `norm_weight` has no type gate, so the kernel
would read f16 bytes as f32) — not device-verified because the CPU panic is reached first; that
tooling gap is [#169](https://github.com/yusiwen/minfer/issues/169). (f) `minfer quantize
--type f16` was left unchanged: fixing it is #169, out of this loader-focused ticket.

### F6e — `quantize --type f16` keeps 1-D tensors f32, and the CUDA norm weight type gate (#169) — **DONE 2026-09-26**

**What landed.** [#169](https://github.com/yusiwen/minfer/issues/169), the tooling gap F6d filed when
its new "1-D stays f32" assertion rejected the model `minfer quantize --type f16` had just written.
`QuantizePlan::plan` treated the f16 target as a pure element cast (`let keep = is_quant && (ne[1] <= 1
|| !row_ok)`), so **every** tensor — the 1-D norms/biases included — was converted to f16. The
engine's f16 weight path requires 1-D **f32**: the CPU RMSNorm reads the weight through
`Tensor::data_f32`, which asserts `F32`, and neither `mat_mul_f16` nor the f16 embedding decode has
an f16-norm sibling, so the tool could not run a file it had itself produced.

The predicate is now one arm per target, matching llama.cpp's `tensor_allows_quantization` (which
returns `tensor->type` for `ggml_n_dims < 2`): a **quant** target keeps its `1-D or unaligned-row`
rule, **f16** keeps 1-D (both `minfer convert --outtype f16` and `llama-quantize … F16` write those
tensors f32, and that is the contract the engine reads), and **f32** converts everything. The kept
tensors are reported through the same `preserved` list; the CLI message now names the target's actual
reason (`1-D` for f16, `1-D or row length not a multiple of N` for a quant target) instead of
printing a block-size clause that is meaningless at f16's block size 1.

The **latent CUDA hazard** the issue recorded is fixed and gated too. `CudaBackend::norm_weight` used
to check only that the name was registered; it now compares the **registered byte length**
(`CudaState::weight_size`, the same raw-length convention as `has_weight_of_size`) against the `d*4`
bytes the rms_norm kernel indexes and returns `Err` naming both lengths. Without it an f16 norm would
be launched into a `d*4`-byte read out of a `d*2` buffer — the "kernel-invariant violation is a
refusal, never a silent wrong path" rule of `docs/GPU_SAFETY.md`, and the mutation below shows the
read is a real invalid access, not a theoretical one.

**Gates.** (1) `tooling::tests::quantize_f16_keeps_1d_f32_and_encodes_2d_f16` — CI-covered on a
miniature f16-shaped source written by the real writer. Three arms (f16, f32, q8_0) assert the type
of **every** tensor *as the written file declares it*, against a want computed from the source
spec's rank (never from `plan`), and that a preserved 1-D tensor carries the source's own non-zero
bytes; the f32 control differs in the property under test (its 2-D tensors must come back f32), so
neither arm can carry the other. The preserved list is asserted against the source's 1-D set.
(2) the ignored `f6_quantize_end_to_end_stays_within_the_stated_bound` gained an f16 arm on the real
f16 file: 1-D f32 / 2-D f16 read back, the preserved count, and a **bitwise** f16→f16 logit
equality (every source value is representable, so `assert_eq!` is the honest claim).
(3) `graph::cuda_backend::tests::cuda_norm_weight_size_is_part_of_the_invariant` — the device gate:
the same 64-element norm graph and `d`, two names both registered and both valid float4 dims, so the
f32 arm (control) can only pass and the 2-byte-per-element arm can only fail through the length
check; the refusal's text must name both lengths and "f16-norm".

**Measured acceptance (GB10, sm_121, CUDA 13.0, driver 580.178.04, 2026-09-26).** The source is the
cached `Qwen3-0.6B-Q8_0.gguf` (639 446 688 B); the reference is
`llama-quantize --allow-requantize <src> /tmp/fix169/llama-f16.gguf F16`.

| Criterion | Result |
|---|---|
| **Before** `minfer quantize --type f16` (the bug) | 1 198 050 976 B; **113 1-D F16** (`output_norm.weight` + `blk.*.{attn,ffn}_norm.weight` + `blk.*.attn_{q,k}_norm.weight`) and 197 2-D F16; loading panics at `src/tensor.rs:277` — `data_f32 on 'blk.0.attn_norm.weight' of type F16` |
| **After** `minfer quantize --type f16` | 1 198 182 048 B; **113 1-D F32 + 197 2-D F16**; `sha256 6341ef3a7287cad1e42b5910cb3db4ee587ae2eff76f48f953fef63c00e26667`; **byte-identical** (`cmp`) to the `llama-quantize … F16` reference, same size |
| per-tensor type set vs `llama-quantize … F16` | **310 / 310 agree** on (name → type), and every `ne` shape too |
| `minfer quantize --type f32` | **310 / 310 F32** (2 390 150 816 B), no preserved list |
| the f16 file on **CPU** (`--backend cpu`, greedy, 13-token prompt) | runs; prefill 13 tokens in 0.10 s, 16 generated tokens in 1.11 s; text `<think>\nOkay, the user asked for the capital of France. Let me think` |
| the f16 file on **CUDA** (default backend) | `all 28 blocks + embed/output on cuda (1137.0 MiB of device weights)`; same 16-token greedy text; prefill in 0.09 s, 16 tokens in 0.22 s |
| CPU-vs-CUDA logits (`MINFER_GRAPH_DUMP`) | argmax **equal** at prefill / decode_13 / decode_14; max \|Δlogit\| **0.0178 / 0.0169 / 0.0132** (6.0e-4 / 4.9e-4 / 4.8e-4 relative) |
| CPU `cargo test --release` | **457 passed / 0 failed / 33 ignored** unit (baseline 456 / 0 / 33; +1 gate) + **10 / 0 / 6** integration |
| `PARALLEL=0 scripts/real_model_gates.sh` and the default parallel form | **33 / 0** each |
| `cargo test --release --features cuda -- --test-threads=1` | **526 passed / 0 failed / 37 ignored** (baseline 524 / 0 / 37; +1 CI gate +1 device gate) |
| `FEATURES=cuda scripts/real_model_gates.sh` | **37 / 0** at the 0.5B config and **37 / 0** at the Qwen3-0.6B config |
| `compute-sanitizer --tool memcheck` over the CUDA unit suite | **0 errors** over 526 / 0 / 37 |
| `rustup run stable rustfmt --edition 2021 --check` on the changed `.rs` | clean (rustfmt **1.9.0-stable**; the pinned toolchain has no `rustfmt` component and CI runs no fmt job) |
| `python3 scripts/check_docs_links.py` | **957 relative links in 185 files** |

**Mutations (reverted; `sha256sum -c` byte-identical).** (a) `QT::F16 => false` in `plan` (the
pre-fix behaviour, 1-D converted again) → `quantize_f16_keeps_1d_f32_and_encodes_2d_f16` fails at the
**value** assertion `F16: tensor blk.0.attn_norm.weight came back f16, expected f32 (rank 1-D)`. (b)
the CUDA length check forced off (`if false && got != Some(want)`) → the device gate fails
(`an f16 norm weight must be refused before the launch: ()`, i.e. the f16 arm **executed**), and
under `compute-sanitizer` that run reports `Invalid __global__ read of size 16 bytes` ×12 /
`ERROR SUMMARY: 12 errors` — the `d*4` read out of a `d*2` buffer, device-verified.

**Honest scope.** (a) The byte-identity with `llama-quantize` is a stronger result than the ticket
asked for (per-tensor types) but it holds only for this source/target pair: an f16 source whose 1-D
tensors were already f16 would be **kept** as f16 by both tools (llama.cpp returns `tensor->type`;
neither converts a 1-D tensor to f32), so the "1-D is f32" contract is a property of the files the
producers write, not of an arbitrary GGUF. (b) The miniature source in the CI gate is synthetic
(zero-ish payloads, one architecture key) — the shape rule is exercised, not a real architecture's
tensor mix; the real-file half is the `#[ignore]`d arm and the acceptance run above. (c)
`output.weight` is absent from this tied Qwen3 model, so the tied-embedding policy is not exercised
by the acceptance run (it is covered by the existing q4_0 byte-parity tests and unchanged here). (d)
The CUDA norm gate is device-only (CI has no GPU); its CI-covered half is the `weight_size`
arithmetic exercised through the pure path, and the device arm is run here on GB10. (e) The
`norm_weight` change adds one size lookup per norm node at execute time; no timing was measured and
none is claimed.

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
| Unit suite (aarch64, dgxspark) | `162 passed / 0 failed / 3 ignored` |
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
| 8 | **F1 (AVX2 K-quant dots) and all of Phase G need different hardware** (x86 / a Mac). They cannot be started, let alone verified, on dgxspark. | hardware | Sequencing §11; F1 is the largest single CPU win. |
| 9 | **A sequence's logits' tail depended on its absolute arena offset — resolved by C6 (2026-09-19)** (found while gating C3). The pre-C6 measurements stand and are what justified the fix: at cell 0 vs cell 8 the max |Δ| over the vocabulary was 2.6% relative with the greedy token unchanged and the run deterministic; the hand-built `q`/`k`/`v` → rope → store → attn graph is exact to ≤ 1.2e-7 at the model's own shape *and* equal for a 1-cell and an 8-cell offset; the arena layout is irrelevant (a split reservation is bit-identical per layer); `positions` had exactly four consumers per layer (96 = 24×4); a layer bisect put the entry at layer 0's attention output; and the **rope-injection intervention** proved the entry is RoPE alone (injecting run A's 48 rope outputs into run B made the logits bitwise identical), while a distributed ~1e-6 rope perturbation already saturates the tail (0.44 vs 0.43) with the greedy token stable from 1e-6 to 1e-2. **C6 removed the coupling** — `positions` are sequence-relative and the allocator resolves `cells` — so a cell move changes no angle: the offset tests now assert bitwise equality, C3's acceptance tightens from the named amplified-rounding class to bit-identical, and a compaction no longer re-ropes. Three method notes earned here: a zero from a perturbation probe means nothing without a loud control; **a control validates the path, not the equivalence of the perturbation** (a single element nudged by 1e-5 is not the offset's distributed 1.5e-5 — the earlier "refutation" was an over-read); and **an intermediate buffer may only be read immediately after its own node runs** (`graph.outputs` does not extend liveness). | measurement | Done by C6; the logical-positions design, its gates and the CUDA fused-op port (S3) are in §5. |

#### Test-infrastructure record (#171, 2026-09-26) — one home for the gate contract, and one failure-injection seam

**The problem.** The rules this campaign's gate work produced existed only as narrative inside two
`AGENTS.md` bullets, interleaved with per-ticket history and the counts those tickets moved, so every
agent re-derived them. Mutation checking in particular cost a bespoke mock per ticket (#147's
`MINFER_TEST_CALL_FAIL`, #145's `MINFER_TEST_LATCH_ERROR`, #151's `FailingForward`, #167's switches),
so "break it and watch the gate fail" was skipped.

**What landed.** The rules are stated once in [`GATE-CONTRACT.md`](./GATE-CONTRACT.md) — five of them
(assert the value, not a relation between two code paths; a control arm that differs in the property
under test; mutation evidence; work, not seconds; a device number with its provenance) — each with the
precedent that produced it, and the two `AGENTS.md` bullets keep only the point-of-use
command/default/counts plus a pointer. The failure-injection seam is
[`src/testfail.rs`](../src/testfail.rs): one presence-checked switch (`MINFER_TEST_CALL_FAIL`, the
#147 name and semantics; exact-token comma list, `all` for every Rust site) with chokepoints at the
batched forward (`forward_batch`), the allocator pool entry (`alloc_in_pool`), the backend execute
entry (`execute_node`) and the weight registrar (`register_weight`), plus the device-side
`launch:*`/`attr:*` sites of #147. The observation half is `testfail::note_checked(site)` /
`checked(site)`, bumped by the chokepoint itself, so a gate proves the path ran instead of reading the
dispatch's own answer — the shape #141's f16 vectorization gate needed (`F16_SIMD_PATH_CALLS`). #147's
matcher `injection_names_site` moved into `testfail.rs`, so there is one matcher and its exact-token
tests now run in CI's CPU job instead of only on a `--features cuda` build.

**Migration of the older knobs.** `MINFER_TEST_CALL_FAIL` keeps its name and semantics (the contract
the #147 gates depend on). #145's `MINFER_TEST_LATCH_ERROR` is **left in place**: it *enables* a
deliberately-latching device gate, it does not select a chokepoint, so folding it into the token list
would change what it means. #151's `FailingForward` mock stays in its CI gate (no model on a hosted
runner), and the seam replaces it on the real path below. #167's gates use the pure registration rule
and needed no switch.

**Verification (2026-09-26, dgxspark: 20-core CPU, GB10 sm_121, CUDA 13.0).**

| Command | Result |
|---|---|
| `cargo test --release` (CPU) | **456 passed / 0 failed / 33 ignored** unit + **10 / 0 / 6** integration |
| `PARALLEL=0 scripts/real_model_gates.sh` (CPU, serial) | **33 passed / 0 failed** (37.59 s) |
| `scripts/real_model_gates.sh` (CPU, parallel) | **33 passed / 0 failed** (33.98 s) |
| `scripts/cuda_test.sh` (GB10) | **524 passed / 0 failed / 37 ignored** |
| `FEATURES=cuda scripts/real_model_gates.sh` (0.5B) | **37 passed / 0 failed** |
| `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf FEATURES=cuda scripts/real_model_gates.sh` | **37 passed / 0 failed** |
| `compute-sanitizer --tool memcheck` over the serial CUDA unit suite | **0 API errors** over 524 passed / 37 ignored (351.89 s) |
| `cargo test --release --bin minfer -- --ignored --exact server::batch::tests::the_seam_fails_the_batch_forward_without_a_bespoke_mock --test-threads=1 --nocapture` | **1 passed**; prints `[#171] forward_batch seam: one 500, slot released, no retry, 1 observed forward entries` |
| `MINFER_TEST_CALL_FAIL=forward_batch cargo test --release --bin minfer -- --ignored --exact server::batch::tests::server_batch_matches_serial_and_is_faster --test-threads=1` | **FAILED** — one environment variable reproduces #151's mutation and trips an existing gate (`[testfail] deliberate panic injected at site 'forward_batch'`) |
| `cargo test --release --bin minfer -- testfail::tests::the_seam_is_off_by_default` with `requested()` ignoring the environment | **FAILED** at `assertion failed: !requested("all")` (reverted, `sha256sum` identical) |
| `cargo test --release --bin minfer -- the_execute_chokepoint_is_observable` with `note_checked()` a no-op | **FAILED** `left: 0, right: 2` (reverted, `sha256sum` identical) |
| `rustup run stable rustfmt --edition 2021 --check` on every changed `.rs` | clean (`rustfmt 1.9.0-stable`) |
| `python3 scripts/check_docs_links.py` | **955 relative links resolve in 185 markdown files** |

The CUDA unit count moves 521 → **524** passed while gaining **five** tests, because #147's
`the_injection_matcher_matches_only_the_named_site` was cuda-gated and moved to the always-compiled
`testfail` module: −1 cuda-only, +4 always, +1 `#[ignore]`d real-model gate.

**Honest scope.** A script can require that mutation evidence is *present*; it cannot check that it is
true. The seam removes the cost, not the discipline. The observation counter is thread-local, so a gate
that runs the engine on another thread must read it there. The device-side matcher is necessarily a
second implementation in C++ (a kernel cannot call into Rust); the two are documented as exact-token
identical and the Rust half is the one unit-tested. The still-unbounded `while engine.busy()` steppers
remain [#160](https://github.com/yusiwen/minfer/issues/160), and the count-consistency check that would
enforce rule 5's provenance remains [#94](https://github.com/yusiwen/minfer/issues/94).

#### Test-infrastructure record (#173, 2026-09-26) — the op-timing gate reads its own sink, not a shared table

**The problem.** `graph::scheduler::tests::op_timing_does_not_change_the_result_but_does_accumulate`
(F8/`#51`) took the process-global `optiming::gate()` mutex, forced the timing flag on, and asserted that
a later quiet run left the process-global per-op table unchanged. The mutex serialized the *flag* flips
only; it could not stop another test from **executing a graph** while the flag was on, so that test's
records landed in the same two atomics between the gate's snapshots. Observed once on 2026-09-26 in a full
parallel CPU suite (heavy device jobs also on the box): `left: 2, right: 1` at the "a run with the flag off
records nothing" assertion. A verdict that depends on what else happens to run beside it is not a gate —
this is [#173](https://github.com/yusiwen/minfer/issues/173), the same isolation class `#99` fixed for the
KV format.

**What landed.** The accumulators moved into `optiming::TimingSink` — the same fixed
`[(nanos, calls)]` table, now an instance rather than a `static`. The engine keeps one process-global
sink (`record` / `snapshot` / `reset` delegate to it, so `/metrics` and every other consumer are
unchanged), and `BackendScheduler` carries a `TimingMode` chosen at **construction**:

- `Global` (default): the production policy — resolve `MINFER_OP_TIMING` once per `execute` and record
  into the process-global sink when it is on;
- `Off`: never read the clock;
- `Private { sink: Arc<TimingSink>, enabled: bool }`: record into a caller-owned sink, gated by a
  caller-owned flag — the isolation seam.

`force()`, `GATE` and `gate()` existed only to paper over the shared table and are deleted. The gate now
asserts **values** from the sink its own scheduler wrote: exactly one `silu` and one `add` after the
timing-on run (the `small_graph` is input → silu → add, and an input is never dispatched), and an empty
sink after the timing-off run. A new control test,
`a_concurrent_graph_load_cannot_move_a_private_sink`, runs `4 × 64` real `execute`s into the **shared
global** sink on four scoped threads while asserting eight private sinks stay at exactly one `silu` +
one `add` each, then asserts the shared sink's total moved to `1 + 4 × 64` (the extra one through the free
`record`). The load is bounded by work, every thread is joined by `scope`, and there are no sleeps or
clocks (rule 4).

Why `Private` carries its own `enabled` instead of the recommended `Private(Arc<TimingSink>)` that always
records: the "flag off records nothing" arm has to read the sink the scheduler *would* write, or the arm
is refused by the mode rather than by the flag (rule 2). With an always-recording private sink, the off
arm could only assert the global table — the shared state this ticket removes.

**The stale-doc defect.** The module doc claimed the property was "pinned by
`op_timing_off_by_default_leaves_the_table_empty`"; no such test has ever existed. The doc now names the
real ones — `op_timing_flag_is_presence_checked_and_off_when_unset`, `off_mode_never_records`,
`global_mode_follows_the_flag`, and the scheduler gate — and the stale `optiming::gate` references in
`graph/copystats.rs` are corrected.

**Mutation evidence (rule 3).** The failure-injection seam does not fit — the mutation ignores a flag, it
does not fail a call — so this is the one-line implementation edit the rule allows. Dropping the `enabled`
gate from `TimingMode::resolve_for`'s `Private` arm (one line: `enabled.then(|| sink.as_ref())` →
`Some(sink.as_ref())`) is "the scheduler records while the flag is off":

```
$ cargo test --release --bin minfer -- \
    graph::scheduler::tests::op_timing_does_not_change_the_result_but_does_accumulate --exact
test graph::scheduler::tests::op_timing_does_not_change_the_result_but_does_accumulate ... FAILED

thread '...' panicked at src/graph/scheduler.rs:706:9:
a run with timing off records nothing: [OpTimingEntry { name: "add", calls: 1, nanos: 43584 },
  OpTimingEntry { name: "silu", calls: 1, nanos: 7056 }]

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 492 filtered out; finished in 0.00s
error: test failed, to rerun pass `--bin minfer`
$ echo $?
101
```

Reverted byte-for-byte (`diff -q` clean; `src/optiming.rs` sha256
`4f20689df7ea803ea703c96893725fd3914d7d9c2a911e4f864e9d92b9409d3f` both sides).

**Counts (rule 5).** All on dgxspark (20-core aarch64), `cargo test --release`:

| Run | Unit | Integration |
|---|---|---|
| idle, run 1 | **460 passed / 0 failed / 33 ignored** | **10 / 0 / 6** |
| idle, run 2 | **460 passed / 0 failed / 33 ignored** | **10 / 0 / 6** |
| under 24 CPU spinners (20 cores) | **460 passed / 0 failed / 33 ignored** (95.89 s) | **10 / 0 / 6** |
| the two gate tests ×20 under the same load | **2 passed / 0 failed** every iteration | — |

The delta is **+3**: `optiming` loses `the_gate_switches_the_scheduler_path_on_and_off` (it tested the
deleted `force`) and gains `off_mode_never_records`, `global_mode_follows_the_flag` and
`private_mode_uses_its_own_sink_and_flag` (−1, +3); the scheduler replaces one gate with the rewritten
gate plus the concurrent-load control (+1). The `x86_64 (CI runner)` row moves by the same +3
(455 → **458**), which `test-linux-cpu`'s `--check-live` confirms against its own log; `AGENTS.md` and
`docs/status.toml` carry both rows.

**Audit of the other global-table readers (acceptance item 4).** `server::metrics::tests::
op_timing_family_renders_seconds_with_nanosecond_resolution` builds `MetricsSnapshot.ops` by hand and
never reads the process table, so it is outside the fault class. The two tests that render a live
`ServerMetrics::snapshot()` (`the_token_families_render`, `both_threads_write_into_one_registry`) do read
the global table incidentally, but assert only token/KV families and well-formedness — an op family a
concurrent execution adds cannot change their verdict. No follow-up is needed: the only shared-table
writer left in the suite is the `#173` control test itself, which resets the sink before and after.

**Honest scope.** The new gate proves that a scheduler's verdict reads only its own sink, and the control
test proves a concurrent global load cannot move it; it does **not** prove that any *other* future test
will keep its hands off the process table. That is now a rule a reviewer checks, not a property CI can
see. The removed `force()` also removes the suite's only way to flip the cached process flag — the
production `Global` path is exercised by the pure `resolve_for` tests and the (unset) flag's rule, not by
an in-process toggle; a real `MINFER_OP_TIMING=1` end-to-end run is still the device/manual evidence
recorded under F8.

#### CUDA test-infrastructure record (#185, 2026-09-26) — the parallel device suite's SIGSEGV is the global capture window racing an unlocked weight copy

**The observation.** A parallel `cargo test --release --features cuda -- --ignored`
(`--test-threads` = 20 cores, 38 tests) on GB10 sm_121 / CUDA 13.0 / driver 580.178.04 either
segfaulted or produced an unstable failure set. Reproduced in this session on the same box and binary
(`target/release/deps/minfer-d35a4f8f70a3b190 --ignored`, 2026-09-26):

| runs | SIGSEGV (rc 139) | completed runs' failures | failure sets |
|---|---|---|---|
| 6 (bare, with `timeout 300`) | **3 / 6** | 4, 3, 9 failed | three distinct sets |
| 10 (under gdb, `handle SIGSEGV stop print nopass`) | **2 / 10** | — | — |

Across the 16 runs the failing members moved over `conversation::tests::*`, `server::batch::tests::*`,
`models::qwen2::graph::tests::{a_packed_kv_cache…, an_auto_offload_plan…, async_cross_copies…,
two_cuda_engines…}` and `graph::cuda_backend::tests::cuda_map_window…` — five distinct sets once the
ticket's own two runs are counted. Not one of them is a wrong *value*: every one is either a 901
capture invalidation, a process-global assertion, or a timing relation. `ulimit -s` is 8192 KiB
(no guard-page frame in the backtraces); `compute-sanitizer --tool memcheck` is 0 errors on the
serial suite, so a device memory error was not the candidate.

**The backtrace.** Two independent gdb crashes stopped at the **same faulting instruction** inside
the driver:

```
Thread N "models::qwen2::" received signal SIGSEGV, Segmentation fault.
0x0000ffffab24da8c in ?? () from /lib/aarch64-linux-gnu/libcuda.so.1
#6  cuMemcpyHtoD_v2
#9  cudaMemcpy
#10 <minfer::cuda::CudaState>::register_weight
#11 minfer::models::weight_reg::register_cuda_weight
#12 minfer::models::qwen2::loader::load_tensor
#13 minfer::models::qwen2::loader::load
#14 minfer::models::load_model_configured
#15 tests::a_packed_kv_cache_answers_like_the_f32_one              (crash 1)
     tests::two_cuda_engines_with_different_kv_layouts_run_interleaved   (crash 2)
```

and, at that instant, a **second thread** inside the same driver:

```
#8  cuGraphInstantiateWithFlags
#10 cudaGraphInstantiate
#11 <minfer::cuda::CudaState>::graph_end_capture_to_exec
#12 <CudaBackend as minfer::graph::backend::Backend>::synchronize
#13 <minfer::graph::scheduler::BackendScheduler>::execute
#14 <minfer::models::qwen2::graph::Qwen2Graph>::forward_batch
#15 minfer::server::chat::guarded_forward_batch
#16 <minfer::server::batch::BatchEngine>::submit_on
#17 tests::a_long_prefill_keeps_another_slot_decoding       (crash 2's co-tenant)
```

One thread holds an open **`cudaStreamCaptureModeGlobal`** capture window
(`cudaStreamBeginCapture(stream, 1)`) and is instantiating it; the other issues a plain, *not*
stream-ordered, blocking `cudaMemcpy` (H2D) from the weight-registration path.

A **third** crash — captured once the two-chokepoint guard below was already in place — is a second
unguarded entry rather than the same one, and it is why the guard has a third chokepoint:

```
Thread 5 "graph::cuda_bac" received signal SIGSEGV
#5  cuLaunchKernel
#8  __device_stub__gqa_attn_split_combine
#9  launch_gqa_attn_split_f32kv
#10 graph::cuda_backend::tests::cuda_map_window_costs_no_more_than_the_span_it_replaces
```

That gate drives `CudaBackend`/`CudaState` **directly** (no `scheduler::execute`), so no
scheduler-level guard can see it; 3 of 8 gdb runs crashed there after the register-weight path was
refused. It now takes the device token itself.

**The determination: (a), the documented capture-stream race.**
`CudaState::stream_lock()` serializes the paths that take it, and `graph_replay_step` holds it across
the capture window — but `CudaState::register_weight` takes no lock at all, and a Global-mode capture
is invalidated by another thread's non-capturable driver call. The benign mode of that race is the
`cudaErrorStreamCaptureInvalidated (901)` the run logs are full of; the recorded mode is the driver
faulting instead of returning. (b) *teardown/drop race*: **ruled out** — neither crash has a
`Drop`, `cudaFree`, `cudaFreeHost`, `cudaGraphExecDestroy` or `graph_destroy` frame; the faulting
frame is a live forward `cudaMemcpy` and the co-tenant is `cudaGraphInstantiate`, not a destructor.
(c) *memory error the serial path only avoids by timing*: **ruled out as the root** — the destination
pointer is the one `cudaMalloc` returned two lines earlier in the same function, no minfer `unsafe`
passes a wild pointer, and `compute-sanitizer` is clean serially. (iv) *stack overflow*: ruled out —
the fault is inside libcuda several frames down, not at a guard page. What makes the crash bite the
serial path too is not a timing accident but the *sharing*: any future two-thread device use
(a second engine thread, a parallel harness) re-creates it.

**What landed (the loud refusal #185 asks for).** `src/device_entry.rs` owns a process-wide,
re-entrant-per-thread token for "inside the CUDA device path". `BackendScheduler::execute` takes it
for a whole execution when the graph has a **`CUDA` split** (the capture window's lifetime; the test
is the split set, not the allocator's state, so a CPU-only graph on a CUDA-enabled allocator is not
refused), and `models::weight_reg::register_cuda_weight` takes it for each registration (the copy
that crashes).
A second **thread** is refused with the operation it was doing, the operation already inside, the
mechanism, the evidence and the remedy (`--test-threads=1` / `scripts/cuda_test.sh`) — **before any
driver call**. The module is feature-independent and pure on purpose, so the CPU CI job runs its
test; the same reason `models::weight_reg::cuda_weight_reg` keeps its decision pure.

**Verification that it refuses rather than crashes (rule 5 numbers).** The same parallel
`--ignored` command, GB10 sm_121, 2026-09-26, as the guard grew (before → 3 / 6 bare runs and 2 / 10
gdb runs SIGSEGV'd with no guard at all):

| guard | bare runs | SIGSEGV | gdb runs | SIGSEGV | named refusals per bare run |
|---|---|---|---|---|---|
| two chokepoints (`execute` + registration) | 6 | 1 / 6 | 8 | 3 / 8 (all `cuLaunchKernel` from `cuda_map_window…`) | 25, 23, 23, 25, 27, 0 |
| **three (landed, + the direct-driver gate)** | **10** | **0 / 10** | **6** | **0 / 6** | 24, 23, 23, 30, 23, 24, 24, 23, 23, 21 |

Every refusal is the `device_entry` message; no run reached the driver concurrently. That the
*intermediate* state still crashed is the useful part: the guard converted the two backtraced
mechanisms into refusals and exposed the third, which is a direct-driver test the scheduler cannot
see. The serial configuration is untouched: `scripts/cuda_test.sh`, GB10 sm_121, 2026-09-26 →
**544 / 0 / 38** (was 541 / 0 / 38; +3 tests: the guard's own test, the explicit-`auto`-budget test
and the per-backend stream-sync gate), and the sanitizer stays at **0 errors** over the same 544.

**Mutation evidence (rule 3).** Deleting the `Some(holder) if holder.thread != me` arm from
`device_entry::enter` — i.e. letting every thread in — makes
`device_entry::tests::the_device_path_is_exclusive_across_threads_and_re_entrant_on_one` fail at
`a second thread must be refused`. The per-backend counter's gate is mutated by making
`CudaBackend::stream_sync_count` return `crate::cuda::stream_sync_count()`, which fails
`stream_sync_counts_are_per_backend_not_process_wide` on its second assertion.

**What does not change, and where it is filed.** The *race* is not fixed: the guard refuses the
configuration instead of making it correct, and it is a **chokepoint, not a structural exclusion** —
a caller that reaches `CudaBackend::execute_node`/`synchronize`/`graph_replay_step`, or
`CudaState::register_weight` directly rather than through those two entry points, still runs
unguarded (as does `Drop for CudaBackend`'s frees, which the pre-existing `stream_guard` serializes
unless the backend is mid-capture). Per-instance streams/capture contexts, or extending the
capture discipline to every un-ordered device call (`register_weight`'s `cudaMalloc`/`cudaMemcpy`,
`cudaMemGetInfo`, `cudaHostAlloc`/`cudaFreeHost`, the `Drop` frees), is
[#188](https://github.com/yusiwen/minfer/issues/188), with both backtraces attached. The S4
map-window timing gate's co-tenant sensitivity (median 1.398x once, in the parallel run; decode half
0.916x and the whole test green in the sibling run) is the #154 class, not the crash:
[#189](https://github.com/yusiwen/minfer/issues/189).

**Two test-hygiene defects, fixed here (they are why the failure set moved).**
`async_cross_copies_never_block_and_stay_bitwise_identical` read the **process-wide**
`cuda::stream_sync_count()` (run A: the async arm counted 4160 stalls against the synchronous arm's
728 — a foreign test's syncs inside the delta). The count now lives on the `CudaBackend`
(`stream_syncs`, bumped by the one `state_sync` helper) and both F5 gates read it there, exactly as
`blocking_readbacks` and `copystats`' accumulators already did; the process-wide function stays for
the "host stalls in this process" figure and its doc now says a gate must not read it. And
`an_auto_offload_plan_fits_the_budget` mutated the process-global `MINFER_GPU_MEM`; it now uses
`OffloadRequest::AutoWithBudget(64)` — an explicit argument, the repo's convention since #99/#153 —
and its second arm uses the device's measured free bytes, so the gate neither reads nor mutates the
environment.


#### CUDA concurrency record (#188, 2026-09-27) — the stream, the capture window and the activation scratch become per `CudaBackend`

**The defect #188 names.** [#185](https://github.com/yusiwen/minfer/issues/185) recorded the parallel
`#[ignore]`d device suite's SIGSEGV as `cudaStreamBeginCapture(stream, 1)` — `cudaStreamCaptureModeGlobal` —
racing a weight-registration `cudaMemcpy` that does not take the stream lock. Under Global semantics a
driver call from **another** thread belongs to the capture window, so it either invalidates it
(`cudaErrorStreamCaptureInvalidated`, 901: the benign mode, all over the run logs) or faults inside the
driver (the recorded mode: `cuMemcpyHtoD_v2` under `register_weight` while another thread sat in
`graph_end_capture_to_exec → cudaGraphInstantiate`). The reason one window could even be "the" window was
structural: `static CUDA: OnceLock<Option<CudaState>>` gave the process **one stream**, so two engines
could not capture independently at all. #185's `device_entry` token made the configuration a loud refusal;
it did not make it correct.

**The instrument, built first (gate contract rule 5 needs a number, and the raw crash rate was too low to
judge a fix).** `graph::cuda_backend::tests::capture_window_on_one_thread_survives_a_weight_registration_on_another`
opens a capture window on thread B, records a device→device copy into it, signals thread A, has A perform a
weight registration **while the window is open**, then closes the window and checks both that
`cudaStreamEndCapture` returned `0` (never 901; `cuda::last_capture_end_code`) and that the replayed graph
produced the bytes the window recorded (the wrong value is seeded into `dst` first, so the value arm cannot
pass on the setup — rule 1). Two env knobs make it the experiment: `MINFER_PROBE_STREAM=context` captures on
`CudaState`'s own (blocking) stream — the pre-#188 shared-stream model — and
`MINFER_PROBE_LEGACY_MEMCPY=1` uses the pre-#188 blocking `cudaMemcpy`;
`MINFER_CUDA_CAPTURE_MODE=0|1|2` picks relaxed/global/thread-local. GB10 sm_121, 2026-09-27, 5 process runs
per cell (90 s watchdog):

| capture stream | registration | mode | result |
|---|---|---|---|
| context (blocking, shared) | blocking `cudaMemcpy` (pre-#188) | global (1, pre-#188) | **5/5 hang** |
| context (blocking, shared) | blocking `cudaMemcpy` | thread-local (2) | **5/5 hang** |
| context (blocking, shared) | blocking `cudaMemcpy` | relaxed (0) | **5/5 hang** |
| instance (non-blocking) | stream-ordered (landed) | global (1) | 5/5 pass |
| instance (non-blocking) | stream-ordered | **thread-local (2, adopted)** | **5/5 pass** |
| instance (non-blocking) | stream-ordered | relaxed (0) | **5/5 `end_code=901`** |

**The mode finding, measured rather than assumed.** (i) The blocking copy is a hard **deadlock** against a
capture window on a blocking stream — the null stream implicitly synchronizes with every blocking stream,
including the one that cannot complete until the host closes it — in **all three modes**. So the mode is
*not* what fixes the historical setup; a stream-ordered copy on a per-instance non-blocking stream is.
(ii) With the structural fix in place, `relaxed` still returns **901** in 5/5 runs (the probe's exact
forbidden outcome; `cudaMalloc` inside the window also fails), so relaxed is ruled out by measurement.
(iii) `global` passes the instance cell but is the mode that lets a foreign thread's call belong to the
capture — the class this ticket exists to remove — so the adopted mode is
**`cudaStreamCaptureModeThreadLocal` (2)**, which bounds invalidation to the capturing thread.

**The structural change (both halves the ticket asks for).** `CudaBackend` now owns a
`cudaStreamNonBlocking` stream (`CudaState::create_stream`, `#[188]`; created in `with_layout`, destroyed
in `Drop` after the pool/scratch/host/graph teardown). `crate::cuda::bind_stream` publishes it in a
thread-local for the duration of every backend device operation, and `CudaState::stream()` answers with
it — so the ~60 launch/copy/event/capture/replay/synchronize helpers keep their signatures and follow the
instance: the launchers, the pinned H2D ring (`write_input_async`), D2D (`copy_device_to_device`), the F5
`copy_cross`/`await_cross` event path, `cudaGraphLaunch`/`graph_begin_capture`/`graph_end_capture_to_exec`,
`state_sync`, `copy_cells` (`kv_move_rows`), `alloc_buffer`/`free_buffer` and `Drop`. **Per stream too**:
the `buf_*` activation scratches became `StreamScratch` (a map keyed on the current stream), the `MmqCache`
memo is keyed the same way (a hit records a scratch pointer, so a shared memo would hand one engine's
plane to another), and the pinned staging ring is keyed on the stream. **Weight registration**
(`CudaState::register_weight`) no longer issues a blocking `cudaMemcpy` at all: it queues the H2D copy on
the **context** stream and waits on that stream, so it can never be recorded into a backend's window.
Context-wide calls stay shared and are named as such: `cudaMalloc`/`cudaFree`, `cudaMemGetInfo`,
`cudaHostAlloc`/`cudaFreeHost`, the weight registry and the derived planes. The process-wide
`stream_lock`/`stream_guard` is gone — the capture window is the backend's own `capturing` field.
Finally, the device tests that drive `cb.state.*` kernels **directly** (`store_kv_q8_0`,
`gqa_attn_split*`, the q5_k/q6_k decode parities, the map-window A/B) now bind the backend's stream:
a context-stream launch read back through an instance-stream sync was a race the shared stream used to
hide, and the serial suite caught exactly two of them
(`cuda_capture_staging_order_and_fallback`, `cuda_verify_attention_nt_invariance`) before the binding was
added.

**The negation: #185's guard is narrowed, and its test updated rather than deleted.** `device_entry`'s
token no longer covers `BackendScheduler::execute` or `weight_reg::register_cuda_weight` — both are
per-instance/stream-ordered now. It stays on the one path that still reaches `CudaState`'s helpers without
a backend to bind: `CudaState::layer_gpu`, the legacy per-layer path, which drives the context-keyed
`buf_*` scratches (the graph path and `main` do not call it). The module's docs and its test now state
that scope, and the test asserts the refusal messages name the narrowed mechanism (`unbound`) and the one
remaining path (`layer_gpu`). The **positive** property that replaces the refusal is the concurrent gate
below.

**The concurrent gate.** `models::qwen2::graph::tests::two_cuda_engines_forward_concurrently_and_stay_bitwise_identical`
runs two CUDA engines (f32 and q8_0) on two OS threads, both caches alive across a barrier so the forwards
genuinely overlap, and compares each engine's logits **bitwise** (`max |Δ| == 0`) against its own serial
reference. It also asserts the two live backends hold **different** device stream pointers. The verdict is a
value, not a timing, so the S4 map-window co-tenant gate ([#189](https://github.com/yusiwen/minfer/issues/189))
does not decide it. GB10 sm_121, 2026-09-27:
`[188] two CUDA engines forwarding on two threads (8 decode steps each): streams 0xe8f738039e60 vs
0xe8f72c039de0; concurrent-vs-serial drift 0 / 0`.

**Before/after on the parallel `#[ignore]`d configuration (`<test binary> --ignored`, default
parallel, GB10 sm_121, 2026-09-27).** "Before" is master `440178b` with #185's guard call sites removed
— i.e. the crashing configuration the guard was landed to suppress. "After" is this PR (which no longer
routes the graph path or registration through the guard at all). Bare, 6 runs each:

| | exit-ok | SIGSEGV | runs with test failures | failure set |
|---|---|---|---|---|
| before (guard removed) | **0 / 6** | **1 / 6** (rc 139, core dumped) | 6 / 6 (3, 5, 10, —, 5, 4 failures) | unstable: `900`/`901` launch refusals across `server::batch`, `conversation`, `a_session_resumed…` |
| after (this PR) | **4 / 6** | **0 / 6** | 2 / 6 (1 each) | `cuda_map_window_costs…` (#189 timing gate) and `server_batch_matches_serial_and_is_faster` (#154 timing gate) |

Under gdb (`gdb -batch -ex 'run --ignored' -ex 'thread apply all bt'`, 3 runs each): before **1 / 3**
SIGSEGV (faulting inside `libcuda.so.1`), after **0 / 3**. The "after" run also **runs** the configuration instead of refusing it: 39
tests pass in parallel, and the only failures left are the two known co-tenant timing gates — #189 (out
of scope by the ticket) and #154's batch ratio.

**Verification (rule 5 numbers).** All GB10 sm_121, 2026-09-27.
`bash scripts/cuda_test.sh` → **545 / 0 / 39** (was 544 / 0 / 38; +1 the probe, +1 the concurrent gate,
which is `#[ignore]`d). `cargo test --release` (CPU, dgxspark) → **465 / 0 / 33** unit + **10 / 0 / 6**
integration, unchanged — the renamed `device_entry` test keeps the count.
`compute-sanitizer --tool memcheck --target-processes all <test binary> --test-threads=1` →
**0 API errors** over the same 545 / 0 / 39. The real-model device set
(`FEATURES=cuda scripts/real_model_gates.sh`) → **39 / 0** for the 0.5B config and **39 / 0** for the
Qwen3-0.6B config (was 38 / 0; +1 the concurrent gate).

**Mutation evidence (rule 3).** (a) *Mode*: running the probe with `MINFER_CUDA_CAPTURE_MODE=0` (relaxed)
makes it fail 5/5 with `end iteration 0: the capture window was invalidated (code 901 =
cudaErrorStreamCaptureInvalidated)` — the probe detects exactly the outcome the acceptance forbids, and it
is why relaxed was not adopted. (b) *Per-instance stream*: replacing `CudaBackend::with_layout`'s
`state.create_stream()` with `state.stream()` (the pre-#188 shared stream) makes the concurrent gate die
with **SIGSEGV (signal 11)** — the pre-#188 crash reproduced by one line — instead of reaching its
`two different device streams` assertion.

**What did not change / honest scope.** The legacy `layer_gpu` path and the direct `CudaState` scratch
helpers (`upload_hidden`, `download_logits`, …) still share the context stream and its scratch; they are
`#[allow(dead_code)]` legacy surface (the graph path is the production path) and the guard keeps them to
one thread at a time. Metal is untouched. The device suite's own default stays `--test-threads=1`
(`scripts/cuda_test.sh`): the two new gates are parallel-safe, but other device tests still mutate
process-global state (`MINFER_GPU_MEM` history, timing gates) and the wrapper's serial default is not this
ticket's to change.

#### Test-infrastructure record (#189, 2026-09-27) — the S4 map-window A/B is a paired sign test with a value arm

**The defect.** `graph::cuda_backend::tests::cuda_map_window_costs_no_more_than_the_span_it_replaces` — the S4 device half of #123's fix — was a **pure stopwatch**: it interleaved matched rounds of the span window (`row0 + i`) and the map window (row resolved through a run list) and asserted the **median of 9 per-round ratios** `<= 1.25x`. It failed once in the parallel `#[ignore]`d device run (GB10 sm_121, CUDA 13.0, driver 580.178.04, 2026-09-26) on the prefill phase:

```text
[s4-ab] prefill nt=512 nkv=512 hd=128: span 89.9 / map 125.1 us/launch (median of 9 interleaved
rounds of 50); per-round ratios [0.632, 0.697, 0.989, 1.091, 1.398, 1.440, 1.466, 2.198, 6.695]
— median 1.398x
a map prefill costs 1.398x the span it replaces
```

Five of the nine matched pairs were above the bar — exactly the count a median of nine flips at. The decode half passed in the same run (median 0.916x) and the whole test passed in a sibling run. The 1.25x bar had been justified by #123 against "16 CPU spinners plus two concurrent CUDA attention loops" (median 1.079–1.145 prefill); the parallel `#[ignore]`d suite is a far heavier co-tenant (~38 device tests, several capturing CUDA graphs, one GPU). The verdict was about the machine, not the kernel — the #154 class, and not the #185 SIGSEGV that #188 fixed structurally.

**The fix — the value arm first (rule 1).** Before any timing the gate now asserts, on its own fixture (rows `[512, 2560)` carry distinct K/V; the map window's base cell is 512):

- a **one-row** map window returns exactly that row's V, bit for bit against the host's input row — an absolute oracle, not a mode-vs-mode relation;
- a **two-run** map window returns the span's bytes over the same rows, bit for bit — f32 KV at the decode shape and f16 KV at the FA-prefill shape. One run at cell 0 is indistinguishable to a resolver that reads `(cell, len)` as `(lo, hi)`; two runs at a non-zero base are not;
- `testfail::note_checked("cuda_attn_map_window")` — bumped in the launchers `CudaState::gqa_attn_split` / `gqa_attn_kv_prefill` — is 0 after a span call and 1 after a map call. The observation is counted rather than read from the dispatch's own report (the contract's "observation half"), and it is paired with the bitwise arms so "the map kernel ran" and "it resolved the span's rows" are separate facts.

**The fix — a paired sign test (rule 4).** The timing verdict is the **count** of matched pairs whose map arm is above `1.25x` its own span arm; the gate refuses only at **7 of 9** (`PAIRS = 9`, fixed in advance), the one-sided sign test at `alpha = 46/512 = 0.090`. A minority of disturbed rounds can no longer decide the verdict, and neither can the bare majority the recorded run had; a doubled map cost moves all nine. The bar is unchanged at 1.25x and the timed fixture is the pre-#189 one (constant K/V, one run at cell 0), so the recorded margins stay comparable. Every per-round ratio and the refusal count are printed.

**Verification (rule 5 numbers).** All GB10 sm_121, CUDA 13.0, driver 580.178.04, 2026-09-27.

| Command | Result |
|---|---|
| gate alone: `cargo test --release --features cuda --bin minfer cuda_map_window_costs… -- --ignored --nocapture --test-threads=1` | decode 24.6 / 25.0 µs/launch, **0/9** refusals; prefill 82.8 / 91.1 µs/launch, **0/9** |
| parallel `#[ignore]`d: `cargo test --release --features cuda --bin minfer -- --ignored --nocapture`, **6 runs** | **6 × 39 passed / 0 failed / 0 ignored**; gate refusals 0–3 per phase; worst single pair 5.09x |
| mutation: `MINFER_S4_AB_MAP_REPS=2` + the gate-alone command | **0 passed / 1 failed**, decode 51.7 vs 24.6 µs/launch = **9/9** above 1.25x |
| `scripts/cuda_test.sh` | **546 / 0 / 39** (was 545 / 0 / 39; +1 the pure statistic test) |
| `FEATURES=cuda scripts/real_model_gates.sh` (0.5B, then Qwen3-0.6B) | **39 / 0** and **39 / 0** |
| `compute-sanitizer --tool memcheck --target-processes all <test binary> --test-threads=1` | **0 API errors** over 546 / 0 / 39 |
| `cargo test --release` (CPU, dgxspark) | **465 / 0 / 33** unit + **10 / 0 / 6** integration, unchanged |
| `python3 scripts/check_status.py --check` | exit 0 |

**Before/after on the parallel configuration.** The #188 record's post-fix state had 2 of 6 parallel `#[ignore]`d runs with one failure each — this gate once and `server_batch_matches_serial_and_is_faster` (#154) once. With this ticket's statistic the same configuration is **6 / 6 green** (39 / 0 each). Honest reading: in these six runs the co-tenant was lighter than in the recorded one — no run's *median* exceeded 1.25x — so the six runs prove the gate is not decided by the co-tenant, while the recorded distribution (5 of 9 above the bar, median 1.398x) is replayed by the new pure test `graph::cuda_backend::tests::the_s4_ab_statistic_absorbs_a_loaded_run_and_still_refuses_a_real_regression`, which asserts the sign test passes it **and** that the old median of the same ratios is red.

**The stale #185 premise.** The ticket text says the parallel `--ignored` device run is "refused loudly (`src/device_entry.rs`)" and asks for the guard to be removed for the test session. After #188 (master `fe1b7cd`) that guard covers only the legacy unbound `CudaState::layer_gpu` path — `crate::device_entry::enter` has no other call site in `src/` (`grep -rn 'device_entry::enter' src/` names only `src/cuda.rs`, plus a doc-comment mention in `models/qwen2/graph.rs`) — and `BackendScheduler::execute` and `register_cuda_weight` no longer take it. The parallel configuration was therefore run **as-is, with no guard change**; it is the same narrowing #188's record already measured.

**Mutation evidence (rule 3).** `MINFER_S4_AB_MAP_REPS=2` (`src/cuda.rs::s4_ab_map_reps`) issues every **map-mode** attention launch twice, so the gate's timed map arm pays twice the work — the reproducible form of #123's map-work doubling, and an implementation seam rather than a test edit. Armed, the gate fails with **9/9** pairs above the bar:

```text
[s4-ab] decode nkv=2048 nh=28 nk=4 hd=128: span 24.6 / map 51.7 us/launch (9 interleaved matched
pairs of 100); per-round ratios [1.706, 1.745, 2.052, 2.091, 2.095, 2.100, 2.102, 2.136, 2.358]
— 9/9 above 1.25x (sign test refuses at 7)
thread '…cuda_map_window_costs_no_more_than_the_span_it_replaces' panicked: the map window is above
1.25x the span in 9 of 9 matched pairs … FAILED
```

The mutation is an env switch, so the unmutated run is the same binary with the variable unset — there is no source mutation to revert, and `git diff` on the tree carries only the #189 change.

**Docs.** `docs/CUDA-BACKEND-DESIGN.md` gains §7.10 (the statistic, the bar, the value arm, the dated device tables); `AGENTS.md`'s CUDA counts bullet and `docs/status.toml` move 545 / 39 → 546 / 39 for the pure statistic test; the gate's doc comment (which claimed "It no longer needs an otherwise quiet box" — the claim #189 refutes) now states the sign test and the value arm.

#### Test-infrastructure record (#261 step 0, 2026-10-04) — the source-layout plan and the four-layer convention

**What landed.** A docs-only PR (`9174644`, [#269](https://github.com/yusiwen/minfer/pull/269)) that fixes
the *shape* of the campaign splitting `src/cuda.rs`, `src/cuda_kernels.cu`, `src/metal.rs` and
`src/metal.metal`: `docs/SOURCE-LAYOUT-PLAN.md` (new — the decision, the target tree, the per-backend file
tables, the tooling changes, the documentation-anchor plan and the interaction table for the 18 affected
open issues), its mdBook chapter, the four-layer convention in `AGENTS.md` (L1 runtime / L2 launch /
L3 `<backend>/kernels/` / L4 executors; device-first, one polymorphic seam), the backend-layer section in
`docs/ARCHITECTURE.md` — including the rule that a shared `common` needs two real implementations, of
which `allocplan::DeviceMemory` is the only candidate today (second implementation: [#53](https://github.com/yusiwen/minfer/issues/53)) —
and a `docs/BACKENDS.md` footnote. Four of the six stale claims folded into
[#219](https://github.com/yusiwen/minfer/issues/219) are fixed here; `register_weight`'s prose and the
`src/cuda.rs:3471` banner naming the deleted `CudaCommandBuffer` ride with step 1
([#262](https://github.com/yusiwen/minfer/issues/262)), which is the step that moves that code.

**Verification (rule 5 numbers).** `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04, in the step's worktree:

| Command | Result |
|---|---|
| `python3 scripts/check_docs_links.py .` | **989 relative links resolve in 189 markdown files**, exit 0 (base `5386a1c`: 985 / 188) |
| `python3 scripts/check_status.py --check` | prose agrees with `docs/status.toml` (7 phases, 7 count rows: 1 live-checkable, 6 recorded measurements); exit 0 |
| CI run 37197027339 | **7 / 7 green** (`check-docs`, `test-linux-cpu`, `build-linux-cuda`, `build-macos`, `check-viz`, `check-pr-body`, `lint-workflows`), zero code annotations — the single `build-macos` annotation is GitHub's runner-capacity notice |

**Mutation evidence (rule 3).** Two mutations in the worktree, each reverted: a broken relative link at
`docs/BACKENDS.md:18` (`./SOURCE-LAYOUT-PLAN-NOPE.md`) → `check_docs_links` prints
`BROKEN LINK  docs/BACKENDS.md:18: ./SOURCE-LAYOUT-PLAN-NOPE.md` and
`1 of 989 relative links do not resolve (in 189 markdown files)`, exit 1; a perturbed copy of the manifest
(`passed = 478` → `479`, prose untouched) → `check_status` prints
`AGENTS.md:92: cpu-unit (x86_64 (CI runner)) passed: prose says '478', docs/status.toml says '479'`, exit 1.

**Deliberately out of scope.** No `src/` file changed. The plan document's pre-split line ranges are
rewritten by the step that moves the code; its two `docs/CUDA-BACKEND-DESIGN.md` line citations were
replaced by section references in the follow-up that carries this record, because step 0 itself shifted
them — the same class of rot this campaign exists to remove.

#### Test-infrastructure record (#267 step 6, file 1, 2026-10-04) — the long test files start moving: `graph/cuda_backend/tests.rs` becomes a parent plus 11 topic files

**What landed.** The crate's longest test file, `src/graph/cuda_backend/tests.rs` (8,603 lines; 89 `fn`
definitions in all — 74 top-level items, which are 61 `#[test]` of which **one** is `#[ignore]`d plus 13
helpers, and 15 nested helpers: 12 inside test bodies and 3 in the test-only `impl CudaBackend` shim),
is split by
op family into
`src/graph/cuda_backend/tests/<topic>.rs`, each named by a `mod` declaration in the now-106-line parent
([#267](https://github.com/yusiwen/minfer/issues/267), PR [#275](https://github.com/yusiwen/minfer/pull/275)).
It is a pure move: every top-level `fn`/`const` item is byte-identical apart from rustfmt, the 61 test
*leaf* names are the same set, and nothing was added, removed, renamed, re-gated or relaxed. The topic
file is `staging` (4 tests), `pool` (2), `elementwise` (4), `matmul` (6), `mmvq` (3), `prefill` (4),
`weights` (6), `kv` (9), `attention` (4), `attn_window` (6) and `capture` (13); the shared fixtures
(`device`, `pool`, `assert_close`), the `use` lines and the test-only `impl CudaBackend` shim stay in
the parent, which every topic reaches through `use super::*;`. `src/graph/cuda_backend.rs` keeps its
`#[cfg(test)] mod tests;` untouched — only the parent gained child `mod`s, and the layout checker's
declaration walk reaches `tests/<topic>.rs` through `module_dir()`.

**This is the shape the guard exists for.** A test file no `mod` names is never compiled, so its tests
silently stop running while `cargo test` and CI stay green. `scripts/check_source_layout.py` rule 2 is
the only thing that catches it, and this is the first `src/` tree in the repo where a test module is a
*directory*; `AGENTS.md`'s test-file paragraph records the convention in one sentence.

**Verification (rule 5 numbers).** `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04, in the step's worktree,
with the pre-move baseline taken in that same worktree first:

| Command | Result |
|---|---|
| `scripts/cuda_test.sh` | **567 / 0 / 42** (pre-move baseline in the same worktree: 567 / 0 / 42) |
| `cargo test --release` | **481 / 0 / 36** unit + **10 / 0 / 6** integration (baseline 481 / 0 / 36 + 10 / 0 / 6) |
| `python3 scripts/check_source_layout.py` | `src obeys the layout rules`, exit 0 |
| `python3 scripts/check_doc_line_anchors.py` | exit 0 (4 `OUT-OF-RANGE` anchors re-pointed at the moved tests) |
| `python3 scripts/check_docs_links.py`, `scripts/check_status.py --check` | exit 0 each |
| `cargo fmt --all --check` | clean |
| CI run on PR [#275](https://github.com/yusiwen/minfer/pull/275) | **7 / 7 green**, zero code annotations |

**Mutation evidence (rule 3).** `mod staging;` → `// mod staging;` in the parent, then reverted. The
layout checker exits **1** and names the orphan:
`src/graph/cuda_backend/tests/staging.rs: not reachable from src/main.rs — no \`mod\` declaration names it, so it is never compiled (tests in it would silently not run)`.
The CUDA unit row drops **567 → 563** passed (the four `staging` tests), 42 ignored unchanged — so the
tests really did move out of `tests.rs` into a file that only the declaration compiles.

**Docs swept.** `AGENTS.md` (the one-sentence convention, in the existing test-file paragraph),
`docs/SOURCE-LAYOUT-PLAN.md` (§6.4 gains the `tests.rs` → `tests/<topic>.rs` mapping row; the §Status
table's step-6 row), and the four live anchors in `docs/cuda_tutorial/05-kernels-attention-host.md`,
which now name `cuda_kv_f16_roundtrip_attn` at `cuda_backend/tests/kv.rs:1011` and
`cuda_graph_replay_bit_parity` at `cuda_backend/tests/capture.rs:274`. The `#238`/`#239` test-only
annotations in `src/graph/{cuda_backend,alloc,copystats,builder,cpu_backend}.rs`, `src/q4k_dsc.rs` and
`src/server/batch/tests.rs` are re-anchored to the deeper module path (`…::tests::<topic>::<leaf>`);
`src/cuda.rs` is deliberately left alone because #262 owns it in a parallel worktree, and re-anchoring
its four `#238` paths is the one follow-up.

**What the brief had wrong (the tree wins).** The ticket and the step-6 blueprint both said the file has
"88 `fn` items = 61 `#[test]` + 1 `#[ignore]`d test + 26 helpers". Measured on the pre-split file: `61
#[test]`, of which **2** are `#[ignore]`d (`cuda_map_window_costs_no_more_than_the_span_it_replaces` at
`#[ignore = "timing: needs a CUDA device"]`, and `cuda_real_model_registers_q4dsc_planes_only_for_q4k`)
plus 13 top-level helpers = **74 top-level `fn` items**; 12 further helper `fn`s are nested inside test
bodies and 3 inside the test-only `impl CudaBackend` shim, for **89 `fn` definitions** in all. The "88"
counted those nested helpers as items and double-counted the ignored test. The numbers that matter — 61
tests, 2 ignored — are unchanged, and `docs/status.toml` is not edited: the suite counts do not move.

**Deliberately out of scope.** The other eight files of #267 are untouched here, largest first next:
`src/models/qwen2/graph/tests.rs` (3,399), `src/server/batch/tests.rs` (2,629),
`src/graph/alloc/tests.rs` (1,833), `src/tooling/tests.rs` (1,669), `src/sampler/tests.rs` (1,232),
`src/conversation/tests.rs` (1,156), `src/graph/kvcache/tests.rs` (1,134), and
`src/cuda/issue162_tests.rs` (1,186) last, after #263 changes its launch-fixture column assertion.

#### Test-infrastructure record (#267 step 6, file 2, 2026-10-04) — `models/qwen2/graph/tests.rs` becomes a parent plus five topic files

**What landed.** `src/models/qwen2/graph/tests.rs` (3,399 lines, **21 `#[test]` — 7 of them
`#[ignore]`d** device/real-model gates — plus 6 top-level helpers) is split by topic into
`src/models/qwen2/graph/tests/{cuda_kv,offload_copy,kv_reuse,batching,real_model}.rs`, each named by a
`mod` declaration in the now-111-line parent (PR
[#277](https://github.com/yusiwen/minfer/pull/277), part of [#267](https://github.com/yusiwen/minfer/issues/267)).
A pure move: every top-level `fn`/`const` item is byte-identical apart from rustfmt and the 21 test names
are the same set. The topics are `cuda_kv` (4: the packed cache, two engines with different KV layouts,
the concurrent bitwise gate, a session resumed from disk), `offload_copy` (3: E5 partial/auto offload and
the F5 async cross copies), `kv_reuse` (4: cache/prefix reuse, compaction, physical `kv_rm`/`kv_shift`),
`batching` (6: batch composition, offset sensitivity, sequence-count independence) and `real_model` (4:
logits parity + the Metal gates). The 6 shared fixtures (`cached_model_path`, `max_delta`,
`cross_shape_tolerance`, `assert_across_shapes`, `argmax`, `compare`) stay in the parent.
`src/models/qwen2/graph.rs` keeps its `#[cfg(test)] mod tests;` untouched.

**Verification (rule 5 numbers).** `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04, in the worktree:

| Command | Result |
|---|---|
| `scripts/cuda_test.sh` | **567 / 0 / 42** |
| `cargo test --release` | **481 / 0 / 36** unit + **10 / 0 / 6** integration |
| `python3 scripts/check_source_layout.py` | `src obeys the layout rules`, exit 0 |
| `python3 scripts/check_doc_line_anchors.py` | exit 0 (the walkthrough anchor was `OUT-OF-RANGE` before the re-point) |
| `cargo fmt --all --check` | clean |
| CI on PR [#277](https://github.com/yusiwen/minfer/pull/277) | **7 / 7 green** |

**Mutation evidence (rule 3).** `mod batching;` → `// mod batching;`: the layout checker exits **1** with
`src/models/qwen2/graph/tests/batching.rs: not reachable from src/main.rs …`, and the CPU unit row drops
**481 → 475** (the six `batching` tests), 36 ignored unchanged. Reverted.

**Docs swept.** The live `models::qwen2::graph::tests::<leaf>` prose gains its topic segment in
`AGENTS.md`, `docs/CUDA-BACKEND-DESIGN.md`, and the `#238`/`#244` annotations in
`src/graph/cuda_backend.rs`, `src/graph/alloc.rs` and `src/models/mod.rs`; the walkthrough anchor in
`docs/inference_e2e_walkthrough/13-decode-loop-graph-reuse.md` now names
`forward_cached_isolates_kv_between_caches` at `models/qwen2/graph/tests/batching.rs:742`. The
`ARCHITECTURE-EXECUTION-PLAN.md` records themselves are frozen and keep their pre-split anchors.

**Deliberately out of scope.** The remaining six files of #267: `src/server/batch/tests.rs` (2,629),
`src/graph/alloc/tests.rs` (1,833), `src/tooling/tests.rs` (1,669), `src/sampler/tests.rs` (1,232),
`src/conversation/tests.rs` (1,156), `src/graph/kvcache/tests.rs` (1,134), and
`src/cuda/issue162_tests.rs` (1,186) last, after #263.

#### Test-infrastructure record (#267 step 6, file 3, 2026-10-04) — `server/batch/tests.rs` becomes a parent plus seven topic files

**What landed.** `src/server/batch/tests.rs` (2,630 lines, **21 `#[test]` — 13 of them `#[ignore]`d**
device/real-model gates — plus 19 helpers, two structs and three `impl` blocks) is split by topic into
`src/server/batch/tests/{kv_sharing,slots,prefill,batching,stall,http,metrics}.rs`, each named by a `mod`
declaration in the now-356-line parent (PR
[#278](https://github.com/yusiwen/minfer/pull/278), part of [#267](https://github.com/yusiwen/minfer/issues/267)).
The topics are `kv_sharing` (4: a copied prefix, the copy-on-write store, the planned cells, the
whole-arena request), `slots` (2: the table round trip and a resumed snapshot), `prefill` (5: chunk size,
chunked-vs-unchunked, the interleaved ticks), `batching` (1: the batched-vs-serial verdict), `stall` (5:
the injected `FailingForward` double and the one-answer-per-run gates), `http` (2: the 503 and the SSE
error frame) and `metrics` (2: queue/running depth and the counter deltas). The parent keeps the shared
fixtures (`cached_model`, `Reply`, `sampling_params`, `run_batched`, `run_serial`, the
`STEP_BUDGET_*`/`WorkBound` cluster the helpers themselves call, and the `impl BatchEngine` accessors).
`src/server/batch.rs` keeps its `#[cfg(test)] mod tests;` untouched.

**The one non-comment edit.** The file's only `super::super::` reference —
`super::super::chat_template_from_gguf`, which meant `server::chat_template_from_gguf` at the old module
depth — becomes `crate::server::chat_template_from_gguf`. A topic file sits one module deeper, so the
relative path would otherwise resolve to `batch::`; the absolute path names the same item. This is the
only line of test text that differs from a pure move, and it is called out in the PR body too. (The
`super::serve_loop` references are unaffected: the parent's `use super::*;` re-exports `batch::serve_loop`
into the `tests` module.)

**Verification (rule 5 numbers).** `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04, in the worktree:

| Command | Result |
|---|---|
| `cargo test --release` | **481 / 0 / 36** unit + **10 / 0 / 6** integration |
| `scripts/cuda_test.sh` | **567 / 0 / 42** |
| `python3 scripts/check_source_layout.py` | `src obeys the layout rules`, exit 0 |
| `python3 scripts/check_doc_line_anchors.py`, `check_docs_links.py`, `check_status.py --check` | exit 0 each |
| `cargo fmt --all --check` | clean |
| CI on PR [#278](https://github.com/yusiwen/minfer/pull/278) | **7 / 7 green** |

**Mutation evidence (rule 3).** `mod stall;` → `// mod stall;`: the layout checker exits **1** with
`src/server/batch/tests/stall.rs: not reachable from src/main.rs …`, and the CPU unit row drops
**481 → 477** passed and **36 → 35** ignored — the four running `stall` gates and the one `#[ignore]`d
one, which is why the ignored count is the sharper half of this mutation. Reverted.

**Docs swept.** `docs/SOURCE-LAYOUT-PLAN.md` (§6.4 now carries the qwen2 and batch rows, so the three
landed splits are all in the mapping table; the §Status step-6 row reads files 1–3) and this record. No
live document anchors `server/batch/tests.rs` by line number, and the parent's own `#239` annotations now
name the child module (`server::batch::tests::<topic>::<leaf>`, five sites).

**Deliberately out of scope.** The remaining five files of #267: `src/graph/alloc/tests.rs` (1,833),
`src/tooling/tests.rs` (1,669), `src/sampler/tests.rs` (1,232), `src/conversation/tests.rs` (1,156),
`src/graph/kvcache/tests.rs` (1,134), and `src/cuda/issue162_tests.rs` (1,186) last, after #263.

#### Test-infrastructure record (#267 step 6, file 4, 2026-10-04) — `graph/alloc/tests.rs` becomes a parent plus six topic files

**What landed.** `src/graph/alloc/tests.rs` (1,833 lines, 42 `#[test]` — 3 of them `#[ignore]`d or
`#[cfg]`-gated — plus 4 helpers and a test-only `impl GraphAllocator`) is split by allocator concern into
`src/graph/alloc/tests/{backend_fence,views,liveness,kv_arena,staging,budget}.rs`, each named by a `mod`
declaration in the now-72-line parent (PR
[#279](https://github.com/yusiwen/minfer/pull/279), part of [#267](https://github.com/yusiwen/minfer/issues/267)).
The topics are `backend_fence` (4: F4's fence, the E5 offload plan and the per-engine KV format),
`views` (8: D1 zero-copy windows, their liveness and `split_parts`), `liveness` (4: reuse along a chain,
parallel chains, input fill and cycles), `kv_arena` (12: regions, C5 sessions, C3 defrag, C8b S3
copy-on-write and the C6/C7 cell bounds), `staging` (4: the F5 destination key, the pending copy and the
#138 drain) and `budget` (10: E4/E4-S3 accounting, the length contract and rebuild re-mapping). The
parent keeps `chain` and the `impl GraphAllocator` accessors; `view_graph` moves with `views` and
`tensor_f32` with `budget`, each used by that topic only. `src/graph/alloc.rs` keeps its
`#[cfg(test)] mod tests;` untouched.

**The non-comment edits.** Ten `super::super::` paths — `DType`, `CNode`, `ops::NodeMeta` and
`kvformat::KvFormat`, all of which meant `graph::…` at the old module depth — become absolute
`crate::graph::…` paths, because a topic file sits one module deeper and `super::super` would otherwise
resolve to `alloc::`. (`super::kv_defrag_enabled_from` is unaffected: the parent's `use super::*;`
re-exports it into the `tests` module.) These are the only test-text differences from a pure move.

**Verification (rule 5 numbers).** `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04, in the worktree:

| Command | Result |
|---|---|
| `cargo test --release` | **481 / 0 / 36** unit + **10 / 0 / 6** integration |
| `scripts/cuda_test.sh` | **567 / 0 / 42** |
| `python3 scripts/check_source_layout.py` | `src obeys the layout rules`, exit 0 |
| `python3 scripts/check_doc_line_anchors.py`, `check_docs_links.py`, `check_status.py --check` | exit 0 each |
| `cargo fmt --all --check` | clean |
| CI on PR [#279](https://github.com/yusiwen/minfer/pull/279) | **7 / 7 green** |

**Mutation evidence (rule 3).** `mod staging;` → `// mod staging;`: the layout checker exits **1** with
`src/graph/alloc/tests/staging.rs: not reachable from src/main.rs …`, and the CPU unit row drops
**481 → 477** passed (the four staging gates), 36 ignored unchanged. Reverted.

**Deliberately out of scope.** The remaining four files of #267: `src/tooling/tests.rs` (1,669),
`src/sampler/tests.rs` (1,232), `src/conversation/tests.rs` (1,156), `src/graph/kvcache/tests.rs`
(1,134), and `src/cuda/issue162_tests.rs` (1,186) last, after #263.

#### Test-infrastructure record (#267 step 6, file 5, 2026-10-04) — `tooling/tests.rs` becomes a parent plus seven topic files

**What landed.** `src/tooling/tests.rs` (1,669 lines, **14 `#[test]` — 11 of them `#[ignore]`d**
real-model/device gates — and 14 helpers) is split by tooling concern into
`src/tooling/tests/{parse,f16_encode,f6_roundtrip,f141_device,f167_qwen3,quantize_bounds,bf16}.rs`,
each named by a `mod` declaration in the now-166-line parent (PR
[#280](https://github.com/yusiwen/minfer/pull/280), part of [#267](https://github.com/yusiwen/minfer/issues/267)).
The topics are `parse` (2: the size parser and the split stem rule), `f16_encode` (1: the f16 writer's
1-D/2-D contract), `f6_roundtrip` (3: the llama.cpp rewrite, HF-conversion and split references),
`f141_device` (1), `f167_qwen3` (2), `quantize_bounds` (4: the end-to-end bounds and the byte-identical
encoder) and `bf16` (2). The parent keeps the shared fixtures (`env_path`, `work_dir`, `cached_qwen05`,
`PROMPT`, `logits_greedy`, `logits_greedy_on`, `logits_greedy_on_qwen3`,
`assert_tensor_payloads_equal`); `miniature_f16_source_specs`/`tensor_of` move with `f16_encode` and
`bf16_vs_f16_weight_value_diffs` with `bf16`, each used by that topic only. `src/tooling.rs` keeps its
`#[cfg(test)] mod tests;` untouched.

**Verification (rule 5 numbers).** `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04, in the worktree:

| Command | Result |
|---|---|
| `cargo test --release` | **481 / 0 / 36** unit + **10 / 0 / 6** integration |
| `scripts/cuda_test.sh` | **567 / 0 / 42** |
| `python3 scripts/check_source_layout.py` | `src obeys the layout rules`, exit 0 |
| `python3 scripts/check_doc_line_anchors.py`, `check_docs_links.py`, `check_status.py --check` | exit 0 each |
| `cargo fmt --all --check` | clean |
| CI on PR [#280](https://github.com/yusiwen/minfer/pull/280) | **7 / 7 green** |

**Mutation evidence (rule 3).** `mod parse;` → `// mod parse;`: the layout checker exits **1** with
`src/tooling/tests/parse.rs: not reachable from src/main.rs …`, and the CPU unit row drops
**481 → 479** passed (the two `parse` gates), 36 ignored unchanged. (A topic whose gates are all
`#[ignore]`d would move only the ignored count; `parse` was chosen because it moves the running half.)
Reverted.

**Deliberately out of scope.** The remaining three files of #267 under 1,300 lines —
`src/sampler/tests.rs` (1,232), `src/conversation/tests.rs` (1,156), `src/graph/kvcache/tests.rs`
(1,134) — and `src/cuda/issue162_tests.rs` (1,186) last, after #263.

#### Test-infrastructure record (#267 step 6, files 6–8, 2026-10-04) — the four sub-1,300-line files land in one PR, a commit apiece

**What landed.** The three remaining files under the ticket's 1,300-line grouping threshold are split
in one PR ([#281](https://github.com/yusiwen/minfer/pull/281), part of
[#267](https://github.com/yusiwen/minfer/issues/267)), one commit per file, each keeping its shared
fixtures in `tests.rs` and declaring one `mod <topic>;` per topic file:

| file | before | after | tests |
|---|---:|---|---:|
| `src/sampler/tests.rs` | 1,232 | a 140-line parent + `{greedy_topk,penalties,stops,minp_typical,xtc,dry,mirostat,bias_validate,defaults,grammar}.rs` | 47 |
| `src/conversation/tests.rs` | 1,156 | a 209-line parent + `{turns,regen,spec,snapshot,overflow,real_model}.rs` (the mock engine and 12 fixtures stay up) | 27 |
| `src/graph/kvcache/tests.rs` | 1,134 | a 122-line parent + `{cells,spans,sharing,resize,defrag}.rs` | 33 |

All three are pure moves: every item is byte-identical apart from rustfmt and the test-name sets are
unchanged. None of the three files contained a `super::super::` reference, so no test-body path needed
the absolute rewrite that #278 and #279 carried. The two live walkthrough anchors for
`conversation/tests.rs` are re-pointed at `conversation/tests/turns.rs:48` and
`conversation/tests/overflow.rs:22` in the same commit.

**Verification (rule 5 numbers).** `dgxspark (aarch64, GB10 sm_121)`, 2026-10-04, in the worktree:

| Command | Result |
|---|---|
| `cargo test --release` | **481 / 0 / 36** unit + **10 / 0 / 6** integration |
| `scripts/cuda_test.sh` | **567 / 0 / 42** |
| `python3 scripts/check_source_layout.py` | `src obeys the layout rules`, exit 0 |
| `python3 scripts/check_doc_line_anchors.py`, `check_docs_links.py`, `check_status.py --check` | exit 0 each |
| `cargo fmt --all --check` | clean |
| CI on PR [#281](https://github.com/yusiwen/minfer/pull/281) | **7 / 7 green** |

**Mutation evidence (rule 3).** One mutation per file, each reverted — the checker names the orphan and
the CPU row drops: `mod dry;` in `sampler/tests.rs` → `src/sampler/tests/dry.rs` and **481 → 476**;
`mod overflow;` in `conversation/tests.rs` → `src/conversation/tests/overflow.rs` and **481 → 475**;
`mod spans;` in `graph/kvcache/tests.rs` → `src/graph/kvcache/tests/spans.rs` and **481 → 475**.

**What is left.** `src/cuda/issue162_tests.rs` (1,186 lines) is the last file of #267 and is deliberately
not split here: it carries the launch-fixture column assertion (`assert_eq!(f.len(), 4)`) that Step 2
(#263) changes, so it moves after #263 merges.
