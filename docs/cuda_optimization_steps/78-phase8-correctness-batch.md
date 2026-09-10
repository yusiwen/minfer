# 78 · Phase-8 correctness & engineering-debt batch — 8a review fixes, 8h①, 8i (LANDED, ledger closed)

> **Result**: 11 Phase-8 review findings fixed (`961f696`); the F32-weight E2E
> exercise (8a②) caught a latent CPU bug — `vec_ops::mat_mul_f32` wrote
> token-TRANSPOSED output for nt > 1; rebuild-gate, multi-split-capture and
> multiturn-reuse tests added; device suite grew 147 → 158 across the batch.
> The last open item (8a① macOS Metal regression run) was hardware-blocked at
> the time and **closed on 2026-09-10** on an Apple M4 Pro — §3.2.
> **Commits**: `961f696` (review batch), `789e64b`→`b849601` (batch-1 + 8g①),
> `60e9cc1` (8h① + 8i). **Date**: 2026-08-29 (8a① closed 2026-09-10).

## 1. Background — where things stood

Phase 7 had just landed (the CUDA graph backend, 7a–7e — see doc 01). Phase 8
opened deliberately with debt instead of performance: an independent review of
`4fcd0d8..5cbb4ca` surfaced 11 findings, and the 7e leftovers still had open
verification items. The ordering rationale is structural: every later
optimization session gates its landings on the device suite and on greedy
token-identity checks, so any hole here silently weakens every gate downstream.
A correctness batch is cheap; a perf campaign built on a leaking suite is not.

Where things stall without this step: the capture-window lifecycle had error
paths that could abort mid-window, several kernels lacked weight-READ row
guards, and there was no end-to-end F32-weight model coverage at all (the F32
kernels from 7e④ had parity fixtures but never ran a real model) — exactly the
kind of gap that hides a real bug forever.

## 2. Principle — why these fixes take this shape

This is a batch document, so there is no single mechanism; the common threads:

- **Lifecycle pairing**: every CUDA Graph capture window needs abort, replay,
  and error paths that agree with each other (capture-window abort on
  `execute_node` errors; the replay-vs-open-window guard).
- **Guard the reads, not just the writes**: kernels that read weight rows
  (q4_K/q6_K raw + padded, f32_vec) got row guards — a malicious/corrupt
  shape must fail loudly, not read out of bounds.
- **Bookkeeping must track reality**: `pos_scratch` pool_gen bump, ring-wrap
  reset race, stale `padded_weights` flag, pinned-ring alloc logging — all
  cases where cached state outlived the condition that justified it.
- **Env flips must be part of the identity**: `MINFER_NO_FUSE_FFN` /
  `MINFER_NO_FUSE_QKV` have to flip `CParams`, otherwise an A/B comparison
  silently reuses the wrong graph (the "A/B footgun" class — 8a③ closes it
  permanently with a test).

## 3. Implementation

### 3.1 The 8a review batch (`961f696`) — 11 findings, all fixed

From the independent Phase-8 review of `4fcd0d8..5cbb4ca`:

1. Capture-window abort on `execute_node` errors (no half-open windows).
2. Replay-vs-open-window guard (a replay must not interleave with an open
   capture).
3. Kernel weight-READ row guards (q4_k/q6_k raw + padded, f32_vec).
4. `pos_scratch` pool_gen bump (stale positions buffer across pool
   generations).
5. Ring-wrap reset race in the pinned ring.
6. Stale `padded_weights` flag.
7. Pinned-ring alloc logging (diagnosability).
8. Metal `ffn_gu` loader gate (~1.99 GiB concat on 7B — memory-only, no perf
   intent).
9. A stray 7 MB trace file removed from the tree.
10. Dump/trace/viz `fuse_ffn` gating aligned with the engine's actual gate.

Suites re-verified after the batch: cuda 147/0, plain 133/0, fmt clean; 7B and
0.5B end-to-end greedy output coherent.

### 3.2 Batch-1 (`789e64b`→`b849601`): 8a③ + 8g① + 8a②

- **8a③ (rebuild-gate unit test) — DONE.**
  `fuse_flags_are_part_of_the_reuse_identity` asserts that flipping
  `MINFER_NO_FUSE_FFN` (and `MINFER_NO_FUSE_QKV`) changes `CParams`, i.e.
  `GraphCache::try_reuse` returns false in both directions.

- **8g① (decode-only capture gate) — DONE**, shipped with this batch:
  `ComputeGraph::capture_nt_hint()` + a decode-only capture gate in
  `graph_replay_step`. Before it, the scheduler ran the 3-run capture protocol
  for EVERY CUDA split, so a repeated identical-nt prefill (the server/slot
  scenario) would silently start capturing a ~437-node graph.
  `cuda_prefill_shaped_graph_never_captures` runs a prefill-shaped (nt=8)
  graph 4× and asserts zero captures. (Productization followed in 8g②/R3-B —
  doc 07.)

- **8a② (F32-weight GGUF E2E) — DONE, and it caught a real CPU bug.** 7e④'s
  F32×F32 kernels had parity coverage but no end-to-end model (none cached).
  A byte-exact GGUF→F32 converter (numpy) produced test models, and
  qwen2.5-0.5B-F32 produced GARBAGE on CPU while the same weights kept as
  Q4_0 ran fine. Root cause: `vec_ops::mat_mul_f32` wrote token-TRANSPOSED
  output for nt > 1 — decode (nt==1) was accidentally correct, which is why
  nothing had ever caught it. After the fix, both F32 models (0.5B +
  qwen3-0.6B) produce identical greedy text on CPU and CUDA. Regression test:
  `f32_matmul_nt2_token_major`.

- **8a① (macOS Metal regression run) — CLOSED 2026-09-10** (Apple M4 Pro,
  Metal 4, `cargo build --release`, rustc 1.92.0). The qwen2 FFN-fusion gate
  was decoupled from `fuse_qkv` to `CParams.fuse_ffn` in 7e⑤ (mirroring
  Qwen3's existing intent), and `961f696` additionally gated the Metal
  `ffn_gu` loader registration on the same condition (nf ≤ 16384 +
  `MINFER_NO_FUSE_FFN`) — the two edits were Metal-relevant but had never run
  on a Mac. Verification, all three checks green:
  1. **Post-vs-pre-change greedy identity** — the pre-change tree
     (`78d410a`, = `f54f721~1`: FFN fusion keyed off `fuse_qkv`, `ffn_gu`
     registered unconditionally) was built in a worktree and A/B'd against the
     current tree, same prompt/seed/`--greedy`: **byte-identical** on 0.5B
     q4_k_m ×2 prompts, 0.5B q4_0 ×1, 7B q4_k_m ×2 (64 tokens each), all on
     Metal.
  2. **`MINFER_NO_FUSE_FFN` A/B** — fused vs unfused (unfused still running
     the FusionPass, per the AGENTS.md comparison rule) is **byte-identical**
     on the same 5 model/prompt pairs, so the decoupled gate does not change
     numerics. This also closes the pre-change footgun: before 7e⑤ the toggle
     was read inline in the build gate but was *not* part of the reuse
     identity (`fuse_flags_are_part_of_the_reuse_identity` now asserts it).
  3. **0.5B still fuses on Metal** — `MINFER_TRACE` decode graph: 24 ×
     `fused_ffn` nodes with `weight: blk.{i}.ffn_gu`, `nf: 4864`,
     `backend: metal`; the 7B decode graph has 0 × `fused_ffn` and 28 ×
     `swiglu` (nf = 18944 > 16384, gate closed, exactly as intended). The
     fused run succeeding at all proves the loader registration happened —
     execution aborts if any weight the graph reads is unregistered.
  - Doc correction: the nf quoted for the 0.5B gate check was 2944; the actual
    `blk.0.ffn_gate.weight` out-dim is **4864** (`[896, 4864]`, well under the
    16384 gate). Conclusion unchanged.
  - Suites on the same Metal machine: `cargo test --release` **169 passed /
    0 failed / 11 ignored** = 155 bin + 14 integration (`conversation_cli` 3,
    `gemm_isolation` 5, `flash_attn_isolation` 2, `flash_attn_blk_isolation` 1,
    `gqa_attn_isolation` 3 — the four Metal isolation suites ran on device and
    are green). The 11 ignored are the env-dependent helpers (real-data dumps,
    throughput profiling, and the `conversation_cli` model-dependent cases),
    unchanged from before this check. `cargo fmt --check` clean.

  The interim state (2026-08-29 → 2026-09-10) stayed on the open ledger rather
  than being silently dropped; this entry is that ledger item being closed.

### 3.3 8h① — stale docs marked SUPERSEDED (`60e9cc1`)

`CUDA_OPTIMIZATION.md` / `CUDA_PROBLEMS.md` (the pre-Phase-7 records) got
SUPERSEDED banners pointing at the current plans, with the absorbed ideas
named: cuBLAS → 8k evaluation, MMQ tiling → 8e (later reversed into the MMVQ
win), GPU quantize → 8c. (2026-09 consolidation note: both legacy docs have
since been retired — `CUDA_PROBLEMS.md`'s surviving conclusions live in
CUDA_OPTIMIZATION.md Appendix C.)

### 3.4 8h② / 8h③ — deferred, deliberately

- **8h② (optional CUDA CI runner) — DEFERRED**: device-gated tests skip
  gracefully today; a self-hosted GB10 runner would keep the suite honest on
  every commit, but requires standing runner infrastructure (a wired-up
  machine + runner registration) — not achievable from a dev session. The
  158-test device suite runs green locally.
- **8h③ (temp files)**: the Phase-7 ledger (`/tmp/minfer_phase7/TEMPS.md`) is
  closed; cleanup awaits the user's decision (no auto-delete policy).

### 3.5 8i — graph integration test debts (`60e9cc1`)

1. **Multi-split capture** — `cuda_multisplit_capture_bit_parity`: a
   CUDA → CPU (Softmax) → CUDA graph yields two CUDA splits; both capture
   (`captured_count == 2`) and replay bit-identical to direct launches.
2. **Multi-turn conversation** — `cuda_conversation_multiturn_reuse` (q4_0
   0.5B, device): turn-2 incremental (append-only KV + reused decode graph)
   vs turn-2 rehydrated from history (fresh graphs + full re-prefill) produce
   IDENTICAL greedy text. `ConversationSpec` now derives Clone; note that
   device tests must call `CudaState::init()` themselves (`get()` only reads
   the singleton).
3. **Slot loop** — covered by the same test: both paths run the GraphCache
   prefill→decode alternation the OpenAI server slot uses; a dedicated
   axum-level test remains out of scope (needs a live HTTP harness).

## 4. Verification

- Device suite progression across the batch: cuda 147/0 → 155/0 → 158/0 (the
  later steps in doc 79 added their own tests); plain suite 133/0; fmt clean.
- `fuse_flags_are_part_of_the_reuse_identity` — defends the A/B workflow
  (an env flip that doesn't change the reuse identity would compare the same
  graph twice).
- `cuda_prefill_shaped_graph_never_captures` — defends the "prefill never
  captures" invariant that 8c later relies on for capture-safety.
- `f32_matmul_nt2_token_major` — pins the fixed nt>1 F32 matmul layout.
- `cuda_multisplit_capture_bit_parity` / `cuda_conversation_multiturn_reuse`
  — integration-level bit-parity and KV-continuity guards.
- 7B/0.5B E2E greedy coherence after every commit in the batch.

## 5. Results

A correctness-only tree: no perf deltas intended or claimed. The concrete
outcomes are the 11 review fixes, the fixed nt>1 F32 CPU matmul, three new
permanent tests, and the completed Phase-8 ledger (8a③/8g①/8a②/8h①/8i DONE;
8h② deferred; 8h③ user-decision; 8a① closed 2026-09-10, §3.2). The batch also
unblocked the rest of Phase 8: 8c's capture-safety argument ("prefill never
captures since 8g①") and the 8f model-coverage work both stand on this
foundation.

## 6. Lessons

1. **Run the correctness batch FIRST** — every later gate (parity, greedy
   identity, suite) inherits its strength from this layer.
2. **"Add the missing E2E coverage" is a test that can fail**: the F32 model
   exercise caught a latent, year-class CPU bug that decode-only usage had
   hidden forever (nt>1 transpose).
3. **Env toggles must join the reuse identity** — otherwise A/B comparisons
   silently compare identical graphs; assert it, don't remember it.
4. **Hardware-blocked verification stays on the open ledger** — never silently
   dropped, and never assumed to be free once the hardware appears. 8a① sat
   blocked for 12 days and, on being re-run (2026-09-10), was green on all
   three checks — but that verdict had to be *produced*, not inferred from
   "the change looked memory-only".

← 77 · [Index](./README.md) · 79 →
