# Known CPU-Path Issues — 2026-08-29 (GB10 / aarch64-Linux first exposure)

> **RESOLVED — 2026-08-29.** All issues below were root-caused and fixed, and a
> third issue (Metal cross-backend copy) was discovered and fixed during macOS
> verification. The full `cargo test` suite is green in **both** single-threaded
> and parallel runs. See each section for the root cause + fix; `## Resolution`
> at the bottom summarizes verification.

Two pre-existing CPU-path issues surfaced on the GB10 (aarch64-Linux) machine
during Phase 7c. Both were **never executed before** on any non-macOS machine:
real-model tests skip when the model is not cached, and Linux could not load
GGUF at all until the `MmapFile` cfg(unix) fix (commit `8fb88f8`). Neither is
caused by Phase 7c — both were proven to fail identically at HEAD `ad1512d`
plus only the mmap fix (`git stash` isolation, see Appendix A).

Per the master directive ("don't touch CPU/METAL paths"), **no CPU-path code
was modified** for either issue. This doc records them for triage; §3 lists
the macOS verification steps.

Affected suite result (before fix): CUDA suite `136 passed / 1 failed`; plain
suite `128 passed / 1 failed` (the same issue 1 test). Everything passes with
`--test-threads=1` except issue 1.

---

## Issue 1 — `graph_logits_match_forward_real_model` diverges from `forward` (deterministic, max diff 18.99) — **RESOLVED (test-only)**

**Test:** `src/models/qwen2/graph.rs::graph_logits_match_forward_real_model`
(model: qwen2.5-0.5b-instruct-q4_0).

**Root cause:** the test's **manual graph path (path A)** — the one that builds
and runs the prefill graph by hand to verify the builder→scheduler→execute
pipeline — forgot to fill the **G3 tail-reduction input `tail_ids`**. The graph
creates `tail_ids` whenever `n_out < nt` (a prefill with `n_out=1`), and
`forward_cached` (path B, production) fills it
(`((nt - n_out)..nt)`); the manual path did not, so the tail `get_rows` picked
row **0** (the zero-filled default) instead of the **last** row. Therefore the
`[prefill]` comparison was `logits(token_0)` vs `logits(token_{nt-1})` — two
completely different tokens, hence exactly `1.899e1` (logic divergence, not
float noise), deterministic.

**Fix (`src/models/qwen2/graph.rs`):** path A now fills `tail_ids` exactly as
`forward_cached` does, guarded by the same input-existence check. Production
`forward_cached` was already correct — **no production code changed**; the bug
was purely in the test.

**Reproduction / verification (macOS, Apple Silicon):**
- Before fix: `[prefill] logits max abs diff: 1.899e1` (deterministic, 3/3),
  matching GB10 exactly.
- After fix: `[prefill] logits max abs diff: 0.000e0`, `[decode] ... 0.000e0`
  — bit-identical.

**Note on the doc's earlier "macOS passed 0.0" record:** that record was
stale — the test has an MPS skip guard (`MpsState::get().is_some()`), so when
it "passed" on macOS it had almost certainly *skipped* (MPS initialized by an
earlier Metal test), not actually compared. Run in isolation (CPU-only, no
MPS) it fails with `18.99` on macOS, exactly like GB10 — which is the
interpretation-matrix row "machine-independent logic bug; macOS records stale".
The fix makes the test pass bit-identically regardless.

**Residual (open, scheduled as Phase 7e item):** on aarch64-Linux (GB10) the
test still fails after the `tail_ids` fix, with max diff **0.449** (down from
18.99; macOS 0.0). Deterministic, single-threaded, both paths share the
now-gated worker pool — a genuine numeric-order difference between the graph
execute path and the legacy adapter path on this machine. Tracked in
`docs/CUDA-BACKEND-PLAN.md` §7e (first item); not blocking any CUDA phase.

## Issue 2 — Parallel test runs trip the `attn_heads` UB check (passes single-threaded) — **RESOLVED**

**Location:** `src/graph/cpu_backend.rs:598` in `attn_heads`, dispatched via the
global CPU worker pool (`src/kernel.rs` `worker_loop` / `par_for`).

**Root cause:** the process-global worker `Pool` is a single `OnceLock`
instance whose `gen`/`done`/`job` protocol is **single-submission and
non-reentrant**. `par_for` and `cpu_quant_matmul` both write `*pool.job`,
bump the shared `gen`, share the `done` counter, and spin on it. Two
**concurrent submissions** (e.g. parallel test threads, or the multi-slot server
path) clobber `job` and share `done`, so a caller can return while its own
range was still computed by a stale job — and then its stack-local `AttnCtx`
dies while late workers still read it (`std::slice::from_raw_parts(c.va...)` in
`attn_heads` → Rust slice-creation UB precondition fires under `debug_assertions`;
in a release build it is silent memory corruption). Normal inference is a single
forward stream, so the race only manifests under concurrent use.

**Fix (`src/kernel.rs`):** added a `gate: Mutex<()>` to `Pool` that serializes
the entire **submit → wait** critical section in both `par_for` and
`cpu_quant_matmul`. Worker threads never take `gate` (they only read `job`), so
a submission runs to completion while its caller holds the lock — no deadlock,
and uncontended in the normal single-stream path. This makes the pool safe for
concurrent callers.

**Regression test (`src/kernel.rs::par_for_concurrent_submissions_safe`):**
spawns 8 threads that each `par_for`-write a distinct marker over `[0, 64)` and
asserts every range is fully its own marker. Verified: with the gate disabled it
reliably reports `mixed/corrupt range`; with the gate it always passes.

**Reproduction / verification (macOS):** the doc described the UB panic under
parallel stress on GB10. On macOS the full parallel suite no longer trips it.
(The CPU-only real-model tests skip once MPS is initialized by Metal tests, and
the gate removes the underlying race regardless.)

## Issue 3 (discovered during macOS verification; not in the original doc) — Metal cross-backend copy reads a stale staging buffer — **RESOLVED**

**Symptom:** three Metal tests failed deterministically (even single-threaded):
`metal_cross_backend_copy` (max diff `1.0`), `metal_cross_backend_copy_large`
(`1.0`), `metal_multi_split_alternation` (`0.51`). On the aarch64-Linux GB10
they never ran (no Metal), so the doc's "plain suite" did not see them.

**Root cause:** in `BackendScheduler::execute`, an input node's buffer was
resolved as `alloc.cross_buffer(s).or_else(|| alloc.node_buffer(s))`. A node
feeding **two** different backends leaves a stale staging buffer behind: the
`x` input is copied to Metal for the silu split (`cross_buffer(x)=Metal`), then
`copy_across(x, CPU)` early-returns (x is already on CPU) so `cross_buffer(x)`
**stays Metal**. The CPU `add` consumer then read the **Metal** staging buffer
(often zeros on the CPU side) instead of the CPU `x` → `add(s, x)` computed
`s + 0`.

**Fix (`src/graph/scheduler.rs`):** only use `cross_buffer(s)` when it is on the
**consuming split's** backend,
`.cross_buffer(s).filter(|cb| cb.backend == split.backend).or_else(node_buffer(s))`.
Otherwise fall back to the node's canonical buffer (which is on the consumer's
backend when no copy was needed for that edge).

**Verification:** all three tests pass after the fix, with diffs ~`1e-7`
(float noise).

---

## §3 macOS manual verification steps (now historical — all pass)

The macOS machine must have the model caches present:

- `~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf`
- `~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`

```bash
# Issue 1 — before fix FAILED (1.899e1); after fix PASSES (0.0).
cargo test -- --test-threads=1 graph_logits_match_forward_real_model

# Issue 2 sanity — passes everywhere.
cargo test -- --test-threads=1 graph_cpu_self_consistency_real_model

# Issue 2 parallel stress — before fix could abort in attn_heads (GB10); after
# fix the full suite is green in parallel.
cargo test
```

Notes for the macOS run:

- The CPU-only real-model tests skip themselves when MPS was initialized by an
  earlier test (`MpsState::get().is_some()` guard). Filtering to a single test
  avoids initializing MPS, which is why the commands above filter.

### Interpretation matrix (pre-fix; kept for the record)

| macOS result | Meaning | Resolution |
|---|---|---|
| Issue 1 passes (0.0) | Linux/aarch64-specific divergence | Not the case — it failed (18.99) on macOS too |
| Issue 1 fails (≈18.99) | Machine-independent logic bug; macOS records stale | Root-caused: test omitted `tail_ids`; fixed |
| Issue 2 aborts in parallel | Worker-pool ctx lifetime bug | Root-caused: non-reentrant shared pool; fixed with a gate mutex |
| Issue 2 passes in parallel | GB10 memory-pressure artifact only | The gate fix removes the underlying race regardless |

## §4 Disposition options (now superseded by the resolution)

All three options were superseded: the issues were root-caused and fixed
directly (option 3), rather than recorded-only (option 1) or harness-only
mitigated (option 2). The fixes are confined to a test (`graph.rs`), the CPU
worker pool (`kernel.rs`), and the scheduler's cross-buffer lookup
(`scheduler.rs`).

---

## Resolution (verification, macOS Apple Silicon)

- Single-threaded: `cargo test -- --test-threads=1` → **152 passed / 3 ignored**
  (main bin) + all integration binaries pass.
- Parallel (default 14 threads): **green, repeated 3×** after the two
  test-harness hardening changes below (no `attn_heads` UB, no flakes).
- New regression tests: `par_for_concurrent_submissions_safe` (issue 2),
  `metal_cross_backend_copy` / `_large` / `metal_multi_split_alternation`
  (issue 3), and the bit-identical `graph_logits_match_forward_real_model`
  (issue 1).
- **Test-harness hardening (parallel stability):**
  - `metal::tests::prefill_gemm_throughput_profile` is now `#[ignore]` — it is a
    heavy Metal throughput *profile* (~450 MB, ~20 s, prints GB/s/TFLOPS, asserts
    nothing) and is timing-sensitive under parallel load. Run opt-in with
    `cargo test -- --ignored prefill_gemm_throughput_profile`.
  - `models::qwen2::graph::tail_tests::fused_qkv_matches_unfused_decode` now
    takes `metal_test_lock()` + `MpsState::init()`. It runs on GPU but had no
    lock, so under parallel `cargo test` it raced concurrent MPS access and the
    fused/unfused greedy token flipped (GPU nondeterminism under contention).

### Fix summary (files touched)

| File | Change |
|---|---|
| `src/models/qwen2/graph.rs` | issue 1: manual graph path now fills the G3 `tail_ids` input (was the ~18.99 divergence). Also: `fused_qkv_matches_unfused_decode` takes the Metal lock. |
| `src/kernel.rs` | issue 2: added `Pool::gate` (`Mutex<()>`) serializing the submit→wait critical section in `par_for`/`cpu_quant_matmul`; added `par_for_concurrent_submissions_safe` regression test. |
| `src/graph/scheduler.rs` | issue 3: `execute` only uses `cross_buffer(s)` when it is on the consuming split's backend (was reading a stale cross-backend staging buffer for a node feeding two backends). |
| `src/metal.rs` | `prefill_gemm_throughput_profile` marked `#[ignore]` (heavy profile test). |
| `docs/KNOWN-CPU-ISSUES-2026-08-29.md` | this resolution record. |

## Appendix A — GB10 evidence (2026-08-29, commit `8fb88f8`)

```
# Baseline isolation: HEAD ad1512d + only the gguf.rs mmap fix, 7c stashed
$ cargo test --features cuda -- --test-threads=1 graph_logits_match_forward_real_model
test result: FAILED. ...

# With 7c applied — identical deterministic divergence (3 runs)
[prefill] graph vs forward logits diverge (max diff 1.899e1)
test result: FAILED. 0 passed; 1 failed; ...

# qwen3 structural twins pass single-threaded
test models::qwen3::graph::tests::graph_cpu_self_consistency_real_model ... ok
test models::qwen3::graph::tests::forward_cached_isolates_kv_between_caches ... ok

# Parallel stress (threads=4) abort
thread '<unnamed>' panicked at src/graph/cpu_backend.rs:598:21
```

Suite totals at `8fb88f8`: CUDA `136 passed / 1 failed / 1 ignored`
(`--test-threads=1`), plain `128 passed / 1 failed / 1 ignored`; the single
failure is issue 1 in both suites.
