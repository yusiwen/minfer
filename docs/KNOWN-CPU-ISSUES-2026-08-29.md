# Known CPU-Path Issues — 2026-08-29 (GB10 / aarch64-Linux first exposure)

Two pre-existing CPU-path issues surfaced on the GB10 (aarch64-Linux) machine
during Phase 7c. Both were **never executed before** on any non-macOS machine:
real-model tests skip when the model is not cached, and Linux could not load
GGUF at all until the `MmapFile` cfg(unix) fix (commit `8fb88f8`). Neither is
caused by Phase 7c — both were proven to fail identically at HEAD `ad1512d`
plus only the mmap fix (`git stash` isolation, see Appendix A).

Per the master directive ("don't touch CPU/METAL paths"), **no CPU-path code
was modified** for either issue. This doc records them for triage; §3 lists
the macOS verification steps.

Affected suite result: CUDA suite 136 passed / 1 failed; plain suite 128
passed / 1 failed (the same issue 1 test). Everything passes with
`--test-threads=1` except issue 1.

---

## Issue 1 — `graph_logits_match_forward_real_model` diverges from the legacy forward (deterministic, max diff 18.99)

**Test:** `src/models/qwen2/graph.rs::graph_logits_match_forward_real_model`
(model: qwen2.5-0.5b-instruct-q4_0).

**What it compares:**

- Path A (direct graph): the test manually builds the prefill graph
  (`n_out=1`), assigns, fuses, allocates, fills inputs, executes via
  `BackendScheduler`, reads `graph.outputs[0]`.
- Path B (legacy adapter): `Qwen2Model::forward(&ids, &positions, &mut kv_f,
  1, n_ctx)` — since Phase 6 this routes through the graph path with a legacy
  `KVCache` adapter (writes the legacy cache into the graph's KV regions,
  executes, copies back).

On macOS this test passed with max diff 0.0 (bit-identical, per the AGENTS
model matrix). On GB10 it fails **deterministically**: max diff **1.899e1**
identical across runs (3/3 reproductions, `--test-threads=1`).

**Evidence that the graph path (A) itself is correct:**

- `graph_cpu_self_consistency_real_model` (qwen3, graph vs graph) **passes**.
- Real generation E2E on CPU produces coherent text and matches CUDA greedy
  output byte-for-byte on 0.5B Q4_0 and 0.6B Q8_0.
- 18.99 is a logic-divergence magnitude (misaligned layer input), not
  float noise (~1e-2).

**Hypothesis (not root-caused):** the divergence is inside path B's legacy
KV adapter chain, or in a call-path difference such as one side using the
kernel.rs worker pool while the other does not. Note the test previously
contained a stale slice (`logits[(nt-1)*nv..nt*nv]` on a G3-reduced 1-row
output) that could never pass once the model was cached anywhere — fixed in
`8fb88f8` (test-only); the 18.99 divergence is downstream of that fix.

**Impact:** test-only. Production inference uses a single path and is correct.

## Issue 2 — Parallel test runs trip the `attn_heads` UB check (passes single-threaded)

**Location:** `src/graph/cpu_backend.rs:598` in `attn_heads`, dispatched via
the global CPU worker pool (`src/kernel.rs:273` `worker_loop`):

```rust
std::slice::from_raw_parts(c.va.add(kv * c.nkt + vs_base), c.hd_kv)
```

**Reproduction:** run the full suite with any parallelism ≥ 4
(`cargo test --features cuda -- --test-threads=4`); a thread aborts at the
line above (Rust slice-creation UB precondition check fires). With
`--test-threads=1` the entire suite passes, deterministically.

**Trigger conditions:** the heavy real-model tests allocate ~12 GB KV pools
each (qwen3-0.6B builds regions for `max_seq_len=40960`); several running
concurrently create extreme memory pressure and change worker timing.

**Analysis (code read, not root-caused):** `attn_heads` receives
`ctx: *const ()` with the documented contract "ctx points to a live AttnCtx
for the duration" (cpu_backend.rs:557). The precondition firing under load is
consistent with that lifetime contract being violated under contention
(closure executed after the ctx died / a pool slot was recycled → garbage
`c.va`). In a release build (UB checks off) the same race could be silent
corruption — that is the real risk. Daily inference shows no errors; only
stressed parallel test runs expose it.

**Impact:** test-harness stability + a latent hazard in the CPU worker pool.
Not hit by normal inference workloads in any observed run.

---

## §3 macOS manual verification steps

The macOS machine must have the model caches present:

- `~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf`
- `~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`

Run at the commit that introduced this doc (or later), on master:

```bash
# Issue 1 — expect "max diff 1.899e1" + FAILED on GB10.
# On macOS: pass (0.0) => Linux-specific; same 1.899e1 failure => the macOS
# pass record was stale (test had not really run there either).
cargo test --features cuda -- --test-threads=1 graph_logits_match_forward_real_model

# Issue 2 — sanity: single-threaded passes everywhere (also run on macOS).
cargo test --features cuda -- --test-threads=1 graph_cpu_self_consistency_real_model

# Issue 2 — parallel stress (default threads). On GB10 this aborts in
# attn_heads; macOS behavior decides whether the hazard is machine-specific.
cargo test --features cuda 2>&1 | tail -5
```

Notes for the macOS run:

- The CPU-only real-model tests skip themselves when MPS was initialized by
  an earlier test (`MpsState::get().is_some()` guard). Filtering to a single
  test avoids initializing MPS, which is why the commands above filter.
- If issue 1 reproduces on macOS with the same 18.99, the divergence is
  machine-independent logic (legacy KV adapter or worker-pool call-path
  difference) and predates the GB10 work entirely.

### Interpretation matrix

| macOS result | Meaning | Suggested disposition |
|---|---|---|
| Issue 1 passes (0.0) | Linux/aarch64-specific divergence (NEON dispatch or pool path) | Root-cause on the Linux machine in a dedicated CPU-fix branch |
| Issue 1 fails (≈18.99) | Machine-independent logic bug; macOS records stale | Root-cause anywhere; higher priority |
| Issue 2 aborts in parallel | Hazard not GB10-specific; worker-pool ctx lifetime bug | Fix kernel.rs ctx lifetime (needs approval — hot CPU path) |
| Issue 2 passes in parallel | GB10 memory-pressure artifact only | Serialize the heavy tests in the harness; document |

## §4 Disposition options (decided after macOS verification)

1. **Record only (current state).** No CPU code touched; suites run with
   `--test-threads=1` where the real-model tests participate.
2. **Test-harness-only mitigation** for issue 2: serialize the real-model
   tests (macOS already uses a `metal_test_lock` pattern). Zero CPU-path risk.
3. **Root-cause both** in a dedicated branch, requiring explicit approval to
   touch CPU code (kernel.rs / cpu_backend.rs / the legacy KV adapter).

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
