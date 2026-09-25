# minfer — AI Agent Context

Pure-Rust LLM inference engine written from scratch (~4400 LOC), llama.cpp-inspired, 0 ML framework deps.
Qwen2/Qwen2.5 + Qwen3 (dense) · CPU + Metal (macOS) + CUDA (opt-in) · GGUF v3.
Inference runs through a **declarative compute graph** (builder → scheduler → per-backend kernels) — design + implementation record: `docs/COMPUTE-GRAPH-DESIGN.md`.

This file is the always-loaded index. Deep dives live in `docs/` (index at the bottom) — don't duplicate them here.

## Code Search (ccc)

For "where / how is X implemented", prefer [ccc](https://cocoindex.io/cocoindex-code/) semantic search over grep:

- minfer: `ccc search "<query>"` (or the `ccc` MCP tool). llama.cpp reference ($HOME/git/reading/llama.cpp): `ccc-llamacpp` MCP tool, or `cd` there and run `ccc search`.
- Structural: `ccc grep '<pattern>'` (e.g. `ccc grep 'fn \NAME(\(A*\))' src/`); narrow with `--lang` / `--path`.
- Indexes auto-refresh; if stale: `ccc index` / `ccc search --refresh`.

## Layout

```
src/
├── main.rs          # CLI + inference loop (prefill → autoregressive decode)
├── graph/           # ★ compute graph — the inference core (files below)
├── gguf.rs          # GGUF v3 parser (~2100 lines, largest file)
├── gguf_write.rs    # F6 (#49): the GGUF v3 *writer* — header/metadata/tensor-index
│                    #   encoding for every GgufType, ggml_pad alignment, per-tensor
│                    #   byte layout, and the split convention (`split.no`/`split.count`,
│                    #   `{stem}-NNNNN-of-MMMMM.gguf`). `GgufWriter` + `write_single` /
│                    #   `write_split`; the parser in gguf.rs is the contract, asserted
│                    #   write→parse in the module's unit tests
├── quantize.rs      # F6 (#49): weight-encoding targets + encoders, byte-identical to
│                    #   llama.cpp's `quantize_row_*_ref` (q4_0/q4_1/q5_0/q5_1/q8_0 +
│                    #   f16/f32 casts), their dequantize/blocks, and the loud refusal of
│                    #   every type without an encoder (K-quants are readable, not writable)
├── convert.rs       # F6 (#49): HuggingFace Qwen2 checkpoint → GGUF (safetensors parsed
│                    #   with serde_json only), the strict-loader metadata (tokenizer
│                    #   arrays/chat template/hparams), and `QuantizePlan` (re-encode an
│                    #   existing GGUF); unknown arch/tensor/dtype are refusals
├── tooling.rs       # F6 (#49): the `convert` / `quantize` / `split` subcommands and the
│                    #   F6 real-model gates
├── block.rs         # quantized block types (repr(C), ggml-common.h layout)
├── quants.rs        # AVX2 / NEON+SDOT dot kernels + Q8_0/Q8_K quantization
├── kernel.rs        # quantized matmul dispatch + CPU scalar fallbacks
├── vec_ops.rs       # RMSNorm, RoPE, Softmax, SiLU
├── tensor.rs        # 4D Tensor (shape/strides/data)
├── cache.rs         # legacy KV cache type (graph path owns KV in the allocator)
├── dump.rs          # debug dump module (--features debug_dump)
├── tokenizer.rs     # byte-level BPE tokenizer (self-contained, from GGUF metadata);
│                    #   `tokenizer.ggml.pre` selects the pre-tokenization rule
│                    #   (qwen2 / qwen35, hand-written splitters — the `regex`
│                    #   crate has no lookahead), and an unknown or missing value,
│                    #   an empty merge table, a non-`gpt2` model or a vocabulary
│                    #   missing any of the 256 byte tokens refuses the load; the
│                    #   old silent `unwrap_or(0)` is a checked byte fallback
├── sampler.rs       # SamplerConfig pipeline: penalties → DRY → grammar mask (F2) → top-k →
│                    #   typical → top-p → min-p → XTC → temperature | mirostat v1/v2, plus
│                    #   logit bias (#48)
├── grammar.rs       # F2 (#47): GBNF parser + pushdown automaton + JSON-schema front end; the
│                    #   per-state token mask and its per-run state (docs/GRAMMAR-DESIGN.md)
├── template.rs      # chat templates (minijinja 2.21.0 + a Python-`str`-method hook
│                    #   installed through `set_unknown_method_callback`), so the
│                    #   model's own template renders — including Qwen3's think-block
│                    #   split. A template that cannot be compiled or rendered is a
│                    #   LOUD error naming the construct and line, never a silent
│                    #   ChatML fallback (which survives only for a GGUF with no
│                    #   `tokenizer.chat_template`); validated at load, so the CLI
│                    #   exits and `serve` refuses to start (F7/#50,
│                    #   docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md)
├── conversation.rs  # multi-turn session (append-only KV; overflow drops the oldest
│                    #   turn's KV range + re-ropes the tail — C2; MINFER_NO_CONTEXT_SHIFT=1 re-renders)
├── server/          # OpenAI-compatible HTTP server (axum); `batch.rs` = continuous
│                    #   batching (E2/E6: default follows the device — on for CUDA, off
│                    #   for CPU/Metal; 0.49x on CPU, 1.97x on GPU; MINFER_BATCH=0/1 overrides);
│                    #   #121: a job the engine cannot place is **answered**, never dropped —
│                    #   `admit`/`submit_on` send `StreamEvent::Err` through the job's own
│                    #   sender via `reject` before it is dropped, so a saturated server gives
│                    #   `503` (non-streaming) / an SSE error frame (streaming) and
│                    #   `minfer_jobs_dropped_total` +1 instead of an empty `200` (queueing
│                    #   instead is #150); the serial path already sent its refusal and still
│                    #   queues in the channel — the mirror case is #151: a **failed decode
│                    #   forward** answers every run whose row was in that one weight pass
│                    #   (the `rows` list is the membership test) through its own sender with
│                    #   `ApiError::server` (500), clears its `cached_tokens` and frees the
│                    #   slot, so the worker stops rebuilding the same batch and retrying the
│                    #   same forward forever; it does not retry (the reachable failures are
│                    #   deterministic, and a panic may have left the arena half-written) and a
│                    #   run whose row was not in the batch is untouched
│                    #   E3: a prefill is fed in chunks of `MINFER_N_BATCH` tokens (default
│                    #   2048 — a no-op below it, 0 = one forward per prefill), and the other
│                    #   slots take their decode step between chunks, so a long prompt does
│                    #   not stall them
│                    #   C7/C7b: the per-slot partition is elastic — a request that needs
│                    #   more than its share reclaims idle capacity, and the planner moves
│                    #   whatever is in the way in either direction, so a grown run keeps
│                    #   its rows while a busy neighbour moves out of the way; C8a: a request
│                    #   whose prefix another slot already computed copies those rows instead
│                    #   of prefilling them (donor may be busy; C8b S2 shares them in place on
│                    #   CPU and copies elsewhere, and S3 copies a shared row out of the donor
│                    #   as soon as a later request diverges inside the shared prefix)
│                    #   F8: `metrics.rs` = the shared `Arc<ServerMetrics>` of relaxed
│                    #   atomics behind `GET /metrics` (Prometheus text): live KV/arena
│                    #   occupancy republished by the worker after every step, queue depth
│                    #   (`accepted - admitted`, the one number neither thread sees alone),
│                    #   running/in-flight counts, token counters + a trailing-window
│                    #   `tokens/s`, and per-op seconds under
│                    #   `MINFER_OP_TIMING` (off by default; `optiming.rs` wraps the
│                    #   scheduler's per-node dispatch, so it includes a backend's prologue
│                    #   and excludes split syncs/copies). SIGINT/SIGTERM drain: refuse new
│                    #   work, let in-flight responses finish up to `MINFER_DRAIN_MS`
│                    #   (default 30000), then log and exit — never an unbounded worker join
├── download/mod.rs  # HuggingFace + Ollama auto-download
├── metal.rs + metal.metal  # MPS kernels + shaders (graph backend: graph/metal_backend.rs)
├── cuda.rs          # CUDA device layer, feature-gated (graph backend: graph/cuda_backend.rs);
│                    #   `device_memory() -> allocplan::DeviceMemory`, so a failed
│                    #   `cudaMemGetInfo` names its `cudaGetErrorName` instead of reading 0 (#122);
│                    #   F5 (#58): the event + async host-transfer primitives the split
│                    #   boundary's staging copy uses (`record_event`/`wait_event`/
│                    #   `stream_wait_event`/`copy_to_host_async`/`host_alloc`), plus
│                    #   `stream_sync_count()` — the host-stall counter the F5 gate reads
├── device_tier.rs   # cc-keyed CUDA device-tier table + selector (docs 105-106)
└── models/          # ModelDef trait + per-arch mod/graph/loader (qwen2/, qwen3/)
```

`src/graph/`: `mod.rs` ComputeGraph/CNode · `ops.rs` Op enum + NodeMeta · `builder.rs` GraphBuilder · `copystats.rs` F5 (#58): the split boundary's counters (`copies`/`waits`/`blocking_host_copies`/`async_host_copies`/`event_syncs`) and the `MINFER_SYNC_COPIES` synchronous reference · `alloc.rs` liveness allocator + persistent KV regions (E4 S1: `memory_report` + a feasibility gate; S2: the pools allocate at the size class, inputs are placed before the walk and the length contract is `BufRef::len`; S3: split reservation (the slot table) from assignment, so a rebuild re-maps; S4 #122: the budget resolves through `allocplan::budget_decision` over an explicit `DeviceMemory`, so a failed device query is not a 0-byte budget, and a poisoned weights lock recovers its entries instead of reporting 0) · `allocplan.rs` the size-class ladder + the pure allocation plan (E4) + `DeviceMemory`/`budget_decision` (the explicit device-memory outcome) · `offload.rs` the layer offload plan (E5: `--gpu-layers`/`MINFER_GPU_LAYERS`, the weight filter and the startup report) · `kvcache.rs` cell store + removal/shift + compaction (C1/C2/C3) + per-sequence span list, in-place prefix sharing (C8b S1/S2) and its copy-on-write store rule (S3) · `kvformat.rs` the KV storage format + the `MINFER_CACHE_TYPE` gate (C4: `f32|f16|q8_0`, strict; `q8_0` is packed and read by the CPU and CUDA kernels — Metal is G5) · `kvsession.rs` the versioned KV session container (C5) · `registry.rs` the backend registry (F4: the `Backend` handle — a fixed id space, **not** an enum — the name-keyed table of entries carrying priority + caps + the pool/host-read/kv-format/enable hooks, the `--backend`/`MINFER_BACKENDS` fence, and the three startup refusals; **F5 adds the boundary's two phases** — `copy_cross` (enqueue; CUDA = `cudaMemcpyAsync` D2H into a pinned slab + `cudaEventRecord`, CPU = the synchronous host round trip, Metal declines) and `await_cross` (the wait; CUDA = `cudaEventSynchronize` for a host consumer / `cudaStreamWaitEvent` for a device one, CPU = a documented no-op)) · `backend.rs` Backend trait · `cpu_backend.rs` / `metal_backend.rs` / `cuda_backend.rs` executors · `scheduler.rs` assign → split → execute · `fusion.rs` SwiGLU fusion · `cache.rs` + `params.rs` params-only graph reuse (E4 S3: one graph per `GraphParams`, a switch re-maps) · `dot.rs` DOT export · `json.rs` graph JSON export for viz.

## Build & Run

```bash
cargo build --release                             # CPU + Metal; never touches nvcc
cargo build --release --features cuda             # + CUDA backend (needs nvcc)
cargo build --release --features cuda,cuda_static # + static cudart (no libcudart.so dep)
cargo build --release --features debug_dump       # + MINFER_DUMP_DIR per-layer dumps

./target/release/minfer <model.gguf> "hello"                       # run (graph path; --graph accepted for compat)
./target/release/minfer info <model>                               # tensor names/types/shapes
./target/release/minfer bench [-p N] [-n N] [-r N] [-o md|csv|json] <model>
MINFER_DISABLE_MPS=1 ./target/release/minfer <model> "hello"       # force CPU
./target/release/minfer --backend cpu <model> "hello"              # F4: fence by name (or MINFER_BACKENDS=cpu)
MINFER_GRAPH_DUMP=/tmp/d  ./target/release/minfer <model> "hello"  # graph logits/KV dump (any build)
MINFER_TRACE=/tmp/t.json  ./target/release/minfer <model> "hello"  # per-node real-data trace for viz/
./target/release/minfer --gpu-layers 8 <model> "hello"             # E5: 8 blocks on the device, the rest on CPU
./target/release/minfer viz <model>                                # viz server (page + live SSE)
MINFER_OP_TIMING=1 ./target/release/minfer serve <model>           # F8: per-op timing in /metrics (off by default)
MINFER_DRAIN_MS=5000 ./target/release/minfer serve <model>         # F8: bound the SIGINT/SIGTERM drain (default 30000)
curl -s http://127.0.0.1:8080/metrics                              # F8: Prometheus text snapshot

# F6 tooling (#49) — HF → GGUF, quantize, split (docs/GGUF-TOOLING.md)
./target/release/minfer convert <hf-model-dir> out.gguf [--outtype f16|f32] [--split-max-size N]
./target/release/minfer quantize in.gguf out.gguf --type q4_0 [--split-max-size N]
./target/release/minfer split in.gguf <out-dir> --max-size 200M [--stem NAME]
```

- CUDA test suite on a real GPU: `scripts/cuda_test.sh` (i.e. `cargo test --release --features cuda -- --test-threads=1`; CI has **no** GPU — its CUDA job only compiles the harness — so this is the only way to exercise the device-gated tests; on this box, last full run after [#121](https://github.com/yusiwen/minfer/issues/121), [#99](https://github.com/yusiwen/minfer/issues/99) and [#151](https://github.com/yusiwen/minfer/issues/151) (2026-09-25): **503 passed / 0 failed / 32 ignored**, GB10 sm_121 — #121's two gates (one CI transport gate and one `#[ignore]`d real-model saturation gate) and #151's two CI gates on the C5 S3 record's 501, less the one obsolete `the_process_wide_format_can_be_redecided` unit test #99 deleted; the Qwen3-0.6B configuration's real-model set is green at **32 / 0**). `CudaState::sync` reports a `cudaGetLastError` latch as a **latched API error with its real origin**, never as a kernel launch (C4 S2c); the eager prefill-GEMM smem opt-in checks each `cudaFuncSetAttribute` return value and skips an over-limit request with the reason. The real-model gates should be run twice: the cached 0.5B (f32 KV) **and** `MINFER_BATCH_TEST_MODEL=~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (f16 KV, hd 128 — the only combination that reaches FA prefill and the half-width cell stride; two pre-existing bugs hid behind the f32-only runs). Local GPU runs must rebuild the CLI *with* the feature (`cargo build --release --features cuda`): a plain `cargo test --release` overwrites `target/release/minfer` with a CPU-only build, which silently measures the CPU. `MINFER_DISABLE_CUDA` is presence-checked — `=0` disables CUDA. **Run it serially**: the device state is a process-wide singleton (`CudaState`) whose MMQ memo / captured graph execs / stream state every test shares, so the parallel harness can make one test perturb another's measurement — a parallel-only determinism failure is a harness artifact unless it reproduces serially (issue [#64](https://github.com/yusiwen/minfer/issues/64)). The `#[ignore]`d subset of this command is **32 passed / 0 failed** on the CUDA build (0.5B config, measured 2026-09-25 after #121).
- Real-model gates (the `#[ignore]`d set): one command — `scripts/real_model_gates.sh`
  (`FEATURES=cuda` selects the device build; `PARALLEL=1`/`0` forces the parallel/serial form). The
  wrapper defaults to **serial on a device build** because the device state is a process-wide
  singleton (`CudaState` — its MMQ memo / captured graph execs / stream state are shared by every
  test), so the parallel harness can make one test perturb another's measurement (issue
  [#64](https://github.com/yusiwen/minfer/issues/64)); on a **CPU-only** build that reason does not
  exist, the *KV-format* reason is gone too (since
  [#99](https://github.com/yusiwen/minfer/issues/99) made the KV storage format **per engine**, the
  parallel harness no longer sizes one gate's KV regions from another gate's format), and the last
  CPU-parallel failure was fixed by [#154](https://github.com/yusiwen/minfer/issues/154) — so the
  wrapper defaults to the **parallel** form there. Measured 2026-09-25 on this box (CPU): **29 passed
  / 0 failed** serial, **29 passed / 0 failed** in parallel (3 plain harness runs; before #99: 19
  passed / 9 failed parallel against 28 passed / 0 failed serial, every one of the nine a KV-region
  width mismatch; before #154 the parallel set was 28 passed / 1 failed).
  [#154](https://github.com/yusiwen/minfer/issues/154) fixed that remaining failure:
  `server_batch_matches_serial_and_is_faster` used to assert a **wall-clock** relation between two
  sequential whole-workload measurements, so a loaded parallel harness let the first-measured phase
  absorb the start-up wave (measured 21.20s batched vs 9.95s serial = **0.47x** in parallel against
  **1.50x** serially). It now interleaves matched rounds of the two modes (`batched, serial` × 7) and
  asserts the **median of the per-round `serial/batched` ratios** > 1.0, printing every ratio and the
  median — the same shape #123 gave the CUDA map-window gate. The median was **1.379–1.468x** over the
  three plain parallel runs (individual rounds as low as 0.923x) and **2.159x** under 16 extra CPU
  spinners; the mutation that doubles the timed batched arm trips it at **0.752x**. The correctness
  comparison is still a separate full-length (16-token) pair, byte-for-byte on CPU.
  [#158](https://github.com/yusiwen/minfer/issues/158) removed the last load-dependent verdict in this
  set: `published_metrics_move_as_requests_are_served` bounded a real-model run by absolute wall-clock
  deadlines (`Instant::now() + Duration::from_secs(120/180)`, asserting `!engine.busy()`), so on this
  20-core box under the parallel set with 16 extra CPU spinners it panicked at *"the long request
  finished inside the deadline"* — measured **28 passed / 1 failed in 435.11s**, green without the
  spinners. It now bounds **work, not seconds**: `BatchEngine::work_units` (one unit per decode-forward
  row plus one per committed token) must advance on every `tick` that leaves the engine busy, and
  `step_budget(prompt, max_tokens) = 4 * (prompt + max_tokens + 8)` caps the step count — measured
  **4 steps / 80** and **64 / 768** on the 0.5B. Under the same 16 spinners the set is **29 passed / 0
  failed in 424.17s**; the mutation that makes `tick` return without advancing trips the work assertion
  on step 1 (0.18s, against the old 120s wait). The audit found no other real-model gate whose only
  failure signal is an absolute deadline — the rest are process-hang watchdogs or a redundant poll
  backstop — and the still-unbounded `while engine.busy()` stepper loops in the other `#[ignore]`d
  server gates are filed as [#160](https://github.com/yusiwen/minfer/issues/160).
  On a CUDA box the set is **32 passed / 0 failed** (0.5B config, GB10 sm_121) and the Qwen3-0.6B
  configuration's set is **32 / 0**; both are still run with `--test-threads=1`. **#123 made the C4
  packed-cache gate device-aware; C4 S2b made it exercise the device; #99 made it per-engine**: it
  loads **two engines per arm** (f32 and q8_0) through `models::load_model_configured` instead of
  flipping a process global, so it is itself the proof that two formats coexist in one process. It runs
  a CPU arm (`--gpu-layers 0`, coverage on every build) and, on a CUDA build with a device, one that
  asserts `device() == Cuda`; each arm asserts its own backend *and* each engine's `kv_format()`, so a
  silent CPU fallback or a mis-resolved format fails loudly instead of reporting a CPU number as a
  device one. The device arm still sets the one process-wide layout tag #99 left in place
  (`cuda::KV_LAYOUT`, which the device kernels read), under a `Drop` guard that restores it if the gate
  panics; the old per-process `kvformat` global and its guard are gone. The map-window timing gate
  (`cuda_map_window_costs_no_more_than_the_span_it_replaces`) asserts the median of interleaved
  per-round ratios (9 rounds per mode) instead of two sequential sums, so its verdict no longer
  depends on a quiet box; its correctness sibling `cuda_map_window_matches_the_span_over_the_same_rows`
  sweeps **f32/f16/q8_0** and additionally compares one single-row Q8_0 window against the
  dequantized cell, because every mode-vs-mode comparison there is blind to a value-level fault
  (mutation-checked: dropping the Q8_0 block base left it green until that arm was added). The
  Qwen3-0.6B configuration's set is **32 passed / 0 failed** (measured 2026-09-25): the f16 KV
  element type is now a **flag** in the session header, so
  `a_slot_snapshot_resumes_the_context_without_re_prefilling` round-trips its own snapshot instead
  of being refused with *"the file was written with the f32 KV element type, this run uses f16"*
  ([#130](https://github.com/yusiwen/minfer/issues/130), closed). A **Q8_0** session round-trips
  the same way: `a_session_resumed_from_disk_continues_bitwise` with `MINFER_CACHE_TYPE=q8_0`
  prints `live KV format: q8_0`, saves 24 layers / 256 cells / 1 696 464 B (vs 6 316 748 B under
  f32) and continues bitwise (max |Δlogit| = 0), because the container's `FLAG_PACKED` bit encodes
  it — and an **f16** session now does too, via `FLAG_F16` (mutually exclusive with `FLAG_PACKED`;
  an unknown flag bit and a header claiming both are refused loudly, and a pre-#130 `flags == 0`
  file still loads as f32).
- Sandboxed agent shells: if `nvidia-smi` reports `Failed to initialize NVML: Unknown Error` and `cuInit` returns 304 while `/dev/nvidia*` exists, the *file sandbox* (Landlock) is denying `open()` with `EACCES` even on `crw-rw-rw-` nodes — that is **not** evidence of a broken driver. Check with a widened sandbox before recording "no device" (A0's probes could not see the GPU either way, so "no device" was unsupported).
- Batching default (E6): `chat::batch_mode(requested, model.device())` — pure and unit-tested, so CI covers the matrix. `ModelDef::device()` (`Device::{Cpu,Metal,Cuda}`) is the single authority for "the device participates", shared with the graph builder's `CParams.gpu`.
- Full CLI + options: `docs/USAGE.md` (stale in places — [#62](https://github.com/yusiwen/minfer/issues/62)). CUDA build details (ccbin pinning, GPU arch coverage, cudart linking): `docs/BUILD.md`.
- Multi-part GGUF: written by `minfer split` (F6) as `{stem}-NNNNN-of-MMMMM.gguf` with `split.no`/`split.count`/`split.tensors.count`; entry is part 0, all parts parsed into one merged tensor index (part order preserved, so the merged index equals the single-file index), and download resume is size-checked (`download::check_downloaded_size` refuses a wrong-length file and removes it).
- **Parallel work in a nested worktree**: `.worktrees/<scope>` (git-ignored, see `.gitignore`) keeps
  the session workspace writable when the file policy is `workspace-write` — a worktree *beside* the
  root is outside it and cannot be written to, and this repo's branch is often already checked out
  elsewhere. `scripts/agent_worktree.sh new|rm|list` wraps the flow; `git worktree remove --force
  .worktrees/<scope>` deletes the tree **including its `target/`**. Each worktree keeps its **own**
  `target/`: never point `CARGO_TARGET_DIR` at the outer tree, because a plain `cargo test --release`
  there overwrites the outer `target/release/minfer` (the CPU-vs-CUDA measurement trap above). The
  outer tree's `git clean -fdx` skips a nested worktree (git sees an embedded repository); never use
  `-ff` while one exists. Use absolute paths in every command: the session's relative paths still
  resolve against the outer root.

## Support

- Quants (CPU + GPU): **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K**. Not supported: Q2_K/Q3_K/I-quants. Full matrix incl. CUDA notes: `docs/SUPPORT-MATRIX.md`.
- Activations: CPU quantizes to Q8_0 on the fly (Q8_K for K-quant weights); GPU backends read f32 (CUDA prefill uses int8 MMQ).
- Verified models (CPU + graph-GPU, greedy output matches llama.cpp where noted in docs): Qwen2.5-0.5B Q4_0/Q4_K_M/Q5_K_M · Qwen2.5-7B Q4_K_M · Qwen3-0.6B Q8_0 · Qwen3-4B Q4_K_M (KV sized by `--n-ctx`, see `docs/PERF-QWEN3-4B-VS-LLAMACPP.md`) · DeepSeek-R1-Distill-Qwen-1.5B (needs the tokenizer special-token match).
- Chat templates (F7/#50): the GGUF `tokenizer.chat_template` is rendered by
  minijinja with a Python-`str`-method hook; the reference renderings live in
  `tests/fixtures/chat/` (generated by transformers 5.17.0 from each model's own
  `tokenizer_config.json`, provenance in the fixtures). A template that cannot be
  rendered refuses the load; tokenizers whose `tokenizer.ggml.pre` is not
  `qwen2`/`qwen35` refuse the load too. Token-id equality with the reference
  (transformers, or llama.cpp on the same GGUF for Qwen3.5) is gated by
  `tests/fixtures/tokenizer/ids_*.json` plus the
  `token_ids_match_the_reference` `#[ignore]`d test.
- Qwen3.5 (`qwen35` arch) is **not** a supported architecture; its GGUF is used
  only as the `qwen35` pre-tokenizer/id reference.
- f16 **weights** (F6/#49): a converted f16 GGUF loads and runs on the CPU path —
  `Op::MatMul` decodes one f16 weight row at a time (`vec_ops::mat_mul_f16`) and
  `Op::GetRows` decodes f16 embedding rows; 1-D norms/biases stay f32 because the
  converter writes them f32 (llama.cpp's rule). The Metal/CUDA registration still
  accepts f32 and the supported quants only, so an f16 model runs the CPU path even
  on a device build, and that path is slow (~3 tok/s prefill on the 0.5B):
  [#141](https://github.com/yusiwen/minfer/issues/141). `minfer quantize` supports
  q4_0/q4_1/q5_0/q5_1/q8_0 (byte-identical to `llama-quantize`), f16 and f32, and
  refuses every type without an encoder by name ([#140](https://github.com/yusiwen/minfer/issues/140)).

## GPU Safety

Read `docs/GPU_SAFETY.md` before touching Metal/CUDA code. Hard rules: `submit()` waits bounded + checks status (never blocks forever); no early return past a `threadgroup_barrier`; device limits queried at runtime, never hardcoded; guard failures abort with actual values. In the graph, **kernel-invariant violations return `Err` from `execute_node` — never a silent CPU fallback**; backend assignment is decided at build time.

## Compute Graph — core rules

Inference = build `ComputeGraph` → assign backends → fuse → allocate → execute; one graph per `GraphParams`, reused across decode steps. Full design: `docs/COMPUTE-GRAPH-DESIGN.md`.

1. **KV positions are data, not structure** — topology never depends on `n_past` (precondition for decode reuse). So is the **allowed attention window**: `attn_span` (E1) carries each query's `[lo, hi)` cell range, resolved by `KvCache` from its per-sequence **span list** — `(position base, first cell, length)`, C8b S1a/S1b — instead of `start + position`; the list is derived from the run and republished on every reserve/release/resize/relocation, and the read path **refuses a multi-span sequence loudly** (one contiguous range is all the input layout carries; a set-valued window (a shared prefix plus a private run) goes through the `kv_map` input instead — C8b S2/S4: the CPU **and CUDA** kernels gather it; Metal refuses both layouts until G5, `supports_attn_span` keeping such a node off it and its `Op::Attn` arm backstopping with a loud `Err`); `Op::Attn { explicit_span }` marks a node that `positions` cannot bound (more than one sequence, or a window that does not start at cell 0), and only a backend with `supports_attn_span()` (CPU and CUDA; Metal is G5) may take it; `graph/batch.rs` composes such batches and `forward_graph_cached` is its one-sequence case. A store may **never** write into a prefix a sequence reads in place: `kv_cells_for_seq` refuses a position inside the share, and both fill entry points run the **copy-on-write** first (`GraphAllocator::kv_private_row_for`, C8b S3) — the share shrinks to the first position that forward writes and the run's own rows shift up inside it, so the address space stays at two spans and no arena space is needed; `GraphAllocator::kv_cell_of` is the read-side twin, for a caller that snapshots a sharing sequence's rows (the store resolver refuses those positions by design). A backend expresses its window as one of **three modes**, selected by the *size* of the window input (causal `positions`, one `[lo, hi)` pair per query, or `KV_MAP_MAX_SPANS` `(cell, len)` runs) — C8b S4 made CUDA's attention kernels take all three as separate template instantiations (`MAP` joins `CAUSAL` so the causal path's instructions are unchanged), resolving a linear window index through the runs; a map window is a *prefix* of the sequence's address space, which is why every kernel's existing `index < limit` mask stays exact.
2. Each layer owns **two persistent KV regions** (K/V) via `kv_pair(layer)`; they survive rebuilds (allocator lives in `GraphCache`). `kvcache.rs` tracks the owner of every cell (C1), can drop a row position in place (C2), can compact the arena (C3), lets a sequence read another's prefix in place (C8b S2) and copies a row out of that prefix the moment a store would land on it (C8b S3). **C6 split position from cell**: `positions` is a token's index *within its sequence* (what RoPE rotates by), and the allocator resolves `cells` — the row to write — from the run's span list (`kv_cells_for_seq`, `attn_span`); the two coincide only while a run starts at cell 0, which is why the single-sequence path is bitwise unchanged. The fused decode QKV family (`Op::FusedQKV`, `Op::QkvBiasRopeStore`) consumes `cells` as well — `positions` ropes, `cells` stores — which is how CUDA keeps its fused chain under an explicit span (`(cuda_on || !explicit_span)`; Metal keeps the pre-C6 gate, G5), and whoever feeds a batch must pass *sequence-relative* positions, never `run.start + pos`. A *physical* `kv_rm`/`kv_shift` (C2) changes **positions**, so it still re-ropes; a **compaction** (C3) changes only cells, so the rows move verbatim (`Backend::copy_cells`: CPU `copy_within`; CUDA's `kv_move_rows` kernel — one row at a time with a barrier, **ascending when the run slides down and descending when it slides up** (C7b), no staging buffer, because overlapping device-to-device `cudaMemcpy` is undefined; Metal refuses it, G5) and `kv_defrag(need)` takes no rope. Whoever copies must use `kvcache::order_moves`: upward moves top-down, downward bottom-up, upward first — a destination may never land on a row that has not been copied yet. Whoever caches a run's `start` — E2's server keeps one per slot — must apply the moves `GraphAllocator::kv_defrag` / `kv_reserve_seq_with_defrag` return.
3. **Reuse is params-only**: `GraphParams` (`CParams.gpu`, `CParams.gpu_layers` and `CParams.kv_format`) deterministically fixes the topology; `GraphCache::try_reuse` compares params only.
4. Weight layout = GGUF: metadata `[in, out]`, memory row-major `[out][in]`; activations token-major `[nt][d]`. I32 inputs stored as `f32::from_bits` via `fill_input_i32`.
5. In-place ops (`Silu`, `RoPE`) alias their input buffer (sole consumer + same backend only). **Never host-copy a GPU-pending buffer** (Phase-3 KV-corruption bug).
6. Execution follows build order (valid topo order); allocator liveness uses the same order, not `topo_order()` (G3 regression); input buffers are never freed.
7. Decode fusions: `Op::FusedQKV` (concat matmul + bias/rope/store) and `Op::FusedFFN` (gate+up concat + swiglu) — gated, and part of the reuse identity (`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` disable the fusion; `MINFER_FFN_COMPOSITION=1` builds the proven D2 *composition* instead of the hand-written node — see plan §D3). Fused vs unfused is bit-identical; when comparing, the unfused path MUST run the FusionPass.
8. Backends own their buffer pools; the allocator is the single owner. **The scheduler's split boundary has two phases (F5, [#58](https://github.com/yusiwen/minfer/issues/58))**: it *enqueues* one cross-backend staging copy per `Split::inputs` entry (`GraphAllocator::copy_across` → the source entry's `copy_cross` hook) and then *waits* on each of them (`GraphAllocator::await_cross` → the entry's `await_cross` hook), before the consuming split runs. A CUDA source's device→host transfer is an `cudaMemcpyAsync` into a pinned slab plus a `cudaEventRecord`, waited on once by `cudaEventSynchronize` — the one documented host block; the CPU's two hooks are a synchronous host round trip and a documented no-op (no device memory, so no transfer to overlap); Metal declines and keeps the synchronous path until it is ported and verified on a Mac (`docs/BACKEND-REGISTRY-DESIGN.md` §11). One wait per copy is a *contract*: the consumer reads through `GraphAllocator::cross_input`, which refuses a staged entry whose wait has not been issued. One Metal command buffer per split. True cross-split *overlap* is a follow-up, not a claim: the split loop is sequential, so the substrate only removes the redundant per-copy stream syncs.
9. CPU quantizes activations to Q8_0, GPU reads f32 — CPU-vs-GPU logits differ by design; compare each path against its own reference.
10. **A node's one output can be several tensors.** `GraphBuilder::split_parts(owner, sizes)` exposes contiguous parts of it as independent `Op::View` aliases (D1 increment 3), so the graph stays single-output and no backend gains a second-output path. An indices part is i32 in f32 bit patterns (rule 4) — that is how a `(values, indices)` producer drives `Op::GetRows` (the MoE-routing shape). D2's `fused_ffn_composition` already consumes a concat matmul this way.
11. **The KV storage format is a gate, never a guess (C4), and it is per engine (#99).** `MINFER_CACHE_TYPE` is parsed strictly into `KvFormat { f32, f16, q8_0 }` (`graph/kvformat.rs`, the single authority) and resolved once per load against the device in `models::load_model_configured` (the env-backed `load_model_ns` / `load_model_with` wrappers read the variable; the explicit cache-type argument is what the tests use instead of mutating the environment). The answer is stored **on the loaded engine** (`ModelDef::kv_format`/`set_kv_format`) and reaches the graph through `CParams::kv_format` (part of the reuse identity: the builder stamps each KV node's `KvcacheMeta::row_elems` from it) and the CPU kernels through `GraphAllocator::set_kv_format`. There is deliberately **no** process-wide `kvformat` global any more: when there was one, the C4 packed gate flipped it and every other test building a graph in the same process sized its regions for the wrong format, which made the `#[ignore]`d gate set red under the parallel harness (issue [#99](https://github.com/yusiwen/minfer/issues/99)). **Honest scope: the device (CUDA/Metal) layout is still process-wide** — `cuda::KV_LAYOUT` / `metal::kv_cache_is_f16` are read by the kernels themselves, so a device run keeps the serial discipline and making that tag per-graph is filed as its own follow-up. An unknown value fails the load on **every** device (CUDA used to read anything but `f16` as f32), and `q8_0` fails on a backend whose attention kernel has no packed read — the answer is the registry's `BackendCaps::reads_packed_kv`, which is **true for the CPU and CUDA** (C4 S1/S2a and S2b) and **false for Metal**, which stays at G5 on [#44](https://github.com/yusiwen/minfer/issues/44) ([#87](https://github.com/yusiwen/minfer/issues/87) is the CUDA/Metal kernel ticket; its CUDA half landed in C4 S2b). CUDA's `KV_LAYOUT_F32/F16/Q8_0` tag and its `kv_row`/`kv4<LAYOUT>` byte-addressed load are the one idiom; the cuts are stated, not silent: decode through the split-K 1-warp body with `rpw_gate = 0` (the hybrid 4-warp body is f16-typed), the verify band and prefill through the general layout-tagged kernel (the f16-typed FA prefill is not offered), no fused decode QKV epilogue (the builders' `layer_gpu` gate carries `&& !packed`), and a **speculative session refuses a packed cache** (`spec::SpecEngine::new`) because the batched split kernel's bitwise identity with sequential decode is what its greedy contract rests on. A Q8_0 cell is a whole number of Q8_0 blocks rounded up to whole f32 words, so `elems / n_ctx` is still one cell's width and every `copy_cells` move (the CoW shift, the compaction) stays verbatim — on the device too, `copy_cells` passes that word count through unchanged; `KvcacheMeta::row_elems` carries that width while the node's shape stays *logical* (`[n_kv_embd, n_ctx]`, what the store's K/V input means), and `ensure_kv` refuses a packed width on a backend without the capability or a width Q8_0 cannot express. The store quantizes with **one** quantizer on both backends (C4 S2b's `store_kv_q8_0` is `amax/127`, f16 scale, round-ties-even — `quants::quantize_row_q8_0_into`'s three steps), and the attention reads the packed blocks **directly**: on the CPU the K score is a `dot_q8_0_q8_0` against the Q8_0-quantized query row and V accumulates out of the cell (`kvformat::accumulate_q8_0_row`) — S1's dequantize-into-a-scratch pass is gone, and `MINFER_NO_FUSED_Q8_KV` restores it for the A/B (measured 1.16x at ctx 512 / 1.31x at ctx 2048 over S1's read, and 1.03x of f32 where S1 was 0.79-0.89x) — while on CUDA the layout-tagged `kv4<KV_LAYOUT_Q8_0>` dequantizes each 4-element group from its block. A physical `kv_rm`/`kv_shift` **works** on a packed region: the survivors move verbatim (a cell is whole words) and each one is mapped through `kvformat::map_q8_0_cells` (dequantize → re-rope → requantize), i.e. C2's re-rope class composed with C4's packing class.
12. **A KV session is a file with a header, never a memory dump (C5).** `graph/kvsession.rs` writes a versioned, checksummed container (shape, backend, **KV element type — the header's flags word encodes it: `FLAG_PACKED` for Q8_0, `FLAG_F16` for f16, `0` for f32; the two element-type bits are mutually exclusive and an unknown bit is refused loudly, so a pre-#130 `flags == 0` file still loads as f32 and an older build refuses a newer f16 file instead of decoding it as f32** — one K/V blob per layer as pool words, then the owner table + run table + span lists + written extents, then — version 2, C5 S2 — an opaque, length-prefixed **host-state blob** inside the checksum: the KV rows belong to a host state, so the container carries both or neither and a version-1 file is refused loudly); `GraphAllocator::kv_save`/`kv_load` stream it through the backends' host I/O and enable the pool a file names if the graph has not been built yet. `kv_load` **verifies the whole file before it applies anything**, so a truncated, corrupted or foreign file is a no-op, and the header must describe the run that loads it (backend, `n_ctx`, `n_embd`, element type). `KvCache::restore_session` validates the bookkeeping — arena capacity, owner-table length, every reservation and span inside the arena, every live sequence carrying a span list — before it applies it. The real-model gate asserts the resumed session continues **bitwise** (max |Δlogit| = 0): the restored rows are the bytes the in-memory run wrote, so equality is the honest claim here. **S2 wires the CLI**: `--session FILE` (under `--cnv`) keeps the JSON history and writes `FILE.kv` beside it on exit; on start a matching companion is resumed with **no** prefill (the host state — messages, `stream_tokens`, `current_pos`, `turn_pos`, `prev_tokens`, `need_insert_eot` — rides in the container as a versioned JSON blob), and everything else (another `--n-ctx`/model/`MINFER_CACHE_TYPE`, an edited history, an unknown snapshot version, an engine that cannot hand its KV to the host) prints the reason and re-renders the JSON. **S2b covers the server**: `--slots-file <PATH>` snapshots the batched engine's slot table (each slot's reservation and the token sequence its rows hold, versioned JSON in the same host section) after every completed request, and resumes it at startup — a request whose prompt matches a restored slot's tokens prefills only its delta. The in-flight request is deliberately *not* in the snapshot (its response stream belongs to a client a restart has disconnected); the serial path has no shared arena and says so loudly.

13. **Memory is accounted before it is allocated, and the length contract lives in the `BufRef` (E4 S1 + S2).** `alloc_in_pool` is fallible: `weights (Backend::weights_bytes) + pooled + this allocation at its size class` is checked against the backend's budget *before* the pool is asked for anything, and the refusal names the numbers (weights / pooled / request / budget, in MiB). The default budget is the backend's own answer — CUDA's current free bytes with a quarter held back, resolved from an explicit `allocplan::DeviceMemory` outcome (`Reported` / `QueryFailed` / `NoDevice`) through the pure `allocplan::budget_decision`; a **failed** query is not a zero budget — it prints the real CUDA error name once and charges weights only, while a *measured* zero still refuses (issue [#122](https://github.com/yusiwen/minfer/issues/122)); CPU and Metal are unbounded unless `GraphAllocator::set_memory_budget` sets one. `memory_report(backend)` is the accounting surface: `pool_bytes` (what the pool **holds** — it only grows when the pool creates a buffer, so a recycled class buffer is not charged twice; `Backend::pool_len` is the probe), `live_bytes`, `peak_live_bytes`, `weights_bytes`, `budget`, `headroom_bytes()`. The class ladder lives in `graph/allocplan.rs` (`class_size`: powers of two to 16 KiB, then 16 KiB steps) and **the pools allocate at it**: `alloc_class_in_pool` asks for `class_size(size)`, so two shapes in one class share a buffer across a rebuild instead of growing the pool. Two rules follow, both of them latent bugs that rounding turned into real ones: (a) a node's **logical** length is `BufRef::len` — `fill_input` checks the data against it and writes through `Backend::write_host_window`, and every capture read is windowed (`scheduler::window_of`); `write_host` keeps the exact-length contract for the persistent KV regions and staging; (b) an input is host-filled **before** execution, so it must never take a buffer this build's `sweep` released — all inputs are placed before the walk — and a liveness extension (in-place alias, D1 view) must move the buffer's `buf_alive` deadline too, which is what `extend_through_views` → `extend_buffer_alive` does (D1's view branch used to extend the *view* itself, a no-op). **E4 S3 split reservation from assignment**: a released classed buffer goes to a reservation table (`slots`, keyed by `(backend, class)`) instead of back to the backend, and `alloc_class_in_pool` takes the smallest idle id of the class (deterministic — liveness releases in `HashMap` order, so a LIFO list would move a rebuilt graph's slots around). A rebuild therefore **re-maps**: `alloc_buffer`/`free_buffer` are not called at all, CUDA's `pool_gen` does not move (so its captured graphs survive the rebuild), and the same topology gets the same slots. `MemoryReport` carries the reservation's depth (`idle_slots`, `reserved_classes`) and `CpuBackend::alloc_count` is the CPU twin of `pool_gen` for the gate. On top of it `GraphCache` (E4 S3) holds one graph per `GraphParams` (MRU, `MAX_CACHED_GRAPHS`), so a switch is `try_reuse` → `alloc_graph` (liveness + slot assignment) with **no** build and no fusion pass: alternating decode widths and a repeated chunked prefill stop rebuilding (`stats()` reports builds vs reuses). Cross-boundary staging is charged to `pool_bytes` but still allocated exact, and backend-internal scratch (Metal capture staging, CUDA `positions` scratch) is not in the report. With S3 the ticket is complete — a rebuild re-maps, the reservation is reported, and several graphs share one allocator ([#55](https://github.com/yusiwen/minfer/issues/55) closed); the follow-on that used to be listed here, the automatic offload fit, lives on [#46](https://github.com/yusiwen/minfer/issues/46).

14. **Layer offload is one plan read in three places (E5).** `graph/offload.rs` owns `OffloadPlan { gpu_layers, n_layers }`: blocks `0..gpu_layers` run on the device and the rest on the CPU, and the tensors *outside* any block (the embedding, the final norm, `lm_head`) follow the device only when **every** block is offloaded (`device_holds_unblocked` — llama.cpp's `n_gpu_layers > n_layer` convention, stated once). The request is `--gpu-layers N` (CLI) or `MINFER_GPU_LAYERS=N` (environment; unset = every block a device can hold, i.e. the pre-E5 behaviour); `OffloadRequest::plan` resolves it **purely** (CI-tested), and a spelling that is not a block count is a refused load, never a guess. The three readers must agree, so they read the same number: (a) the **loader** registers a tensor on the device only when `OffloadPlan::allows_weight(name)` says so, and the block comes from the registry name (`block_of`: `{ns}blk.{i}.…`, including the fused `blk.{i}.attn_qkv` / `blk.{i}.ffn_gu` concat copies); (b) the **builder** stamps `CNode.layer` (`GraphBuilder::set_layer`, called once per block by the model builders) and gates the device-only fused forms on `layer_gpu = gpu && il < gpu_layers`; (c) the **assignment pass** (`BackendScheduler::assign_backends` → `GraphAllocator::supports_for(op, dtype, layer)`) never offers the device for a block past the plan. `CParams.gpu_layers` carries it into the reuse identity, because the assignment is topology. The load verifies the plan against what was registered and drops to CPU-only with a printed reason when the offloaded blocks' weights are not usable there, and `offload_report` prints where the blocks landed with the measured device bytes (E5's startup line). A mixed plan has KV regions on two backends, so `kv_load` refuses a session (one arena per file, C5). **S2 (the automatic fit)** adds the `auto` spelling: the loader measures each block's weight bytes from the **GGUF index** (`GgufTensorInfo::nbytes`, before anything is loaded — the filter *is* the plan, so the decision cannot wait for a measurement), fits the largest **prefix** into the weight budget (`fit_blocks`, pure: a prefix, not a knapsack, because the plan is `0..gpu_layers` and a gap would put a CPU block between two device blocks for nothing), and holds back a quarter for the KV arenas and the activation pool (those are sized per graph, so no load-time fit can measure them; a prompt that needs more is refused by the E4 activation gate rather than silently swapped). The budget is `MINFER_GPU_MEM=<MiB>` if set, else **three quarters of the device's free bytes** — the same default E4's feasibility gate uses, so the fit and the gate talk about one number (`weight_budget`). `auto` needs a device that reports free memory: CUDA does, Metal's wrapper does not yet, so on macOS `auto` fits nothing unless `MINFER_GPU_MEM` says otherwise (the device still participates under the default or an explicit count); a **failed** device query refuses `auto` with the real CUDA error name instead of reading it as "0 bytes free" and fitting 0 blocks (issue [#122](https://github.com/yusiwen/minfer/issues/122)).

## Core Conventions

1. CPU matmuls: quantized weight × Q8_0 activations (`dot_q*_q8_0()`); GPU reads f32 activations directly.
2. SIMD: AVX2 (x86) / NEON+SDOT (aarch64, inline asm) with scalar fallbacks; `MINFER_NO_NEON=1` forces scalar.
3. No ML frameworks — all ops handwritten; tensor data is raw `&[u8]`; GGUF padding via `ggml_pad()`.
4. Cross-backend: per-op assignment decided at build time via `supports_op` — offered in the backend registry's priority order (F4: Metal 300, CUDA 200, CPU 100; `docs/BACKEND-REGISTRY-DESIGN.md`) rather than in a hardcoded chain; guard failures abort — never silent mid-run fallback.

## Extending

**New architecture** (mirror `models/qwen2/` / `qwen3/`): create `models/<name>/{mod,graph,loader}.rs` with `HParams` + `LayerWeights`; dispatch in `models/mod.rs::load_model()`; build the graph with `GraphBuilder` — deterministic in `GraphParams` (reuse invariant); implement `ModelDef` (`forward`/`build_graph`/`forward_graph`/`as_any`); add a chat template if needed.

**New backend** (CUDA is the worked example — `docs/CUDA-BACKEND-DESIGN.md`; the registry contract is `docs/BACKEND-REGISTRY-DESIGN.md`): implement the `Backend` trait (`src/graph/backend.rs`: `supports_op`/`supports_fused`, buffer pool, `execute_node`, host read/write, `synchronize`) with its capability matrix as module-level free functions (the trait methods forward to them, so the registry's answer and the trait's answer are one authority); extend the fixed id space in `src/graph/registry.rs` (`Backend::<NAME>`, a `NAMES` entry, an id — the id is a KV-session file-format contract, so it is **appended**, never renumbered) and write the module's `entry()`/`register()`: priority, caps (`reads_packed_kv` is the per-format capability query), and the `pool`/`pool_mut`/`host_read`/`kv_format`/`enable`/`unavailable` hooks; call `register()` from `Registry::build` under its `#[cfg]` gate. **No consumer needs editing** — the allocator's dispatch, the scheduler's execute, the fusion wiring, the exporters and the KV-session tags all read the registry (`docs/BACKEND-REGISTRY-DESIGN.md` §4). Then register weights at load and gate execution on all-weights-registered, and record participation in `CParams.gpu` (`Qwen2Graph::device` — remember the `active_filter` fence, so `--backend cpu` keeps a device-only fused node out of the graph).

## Sampling

`sampler.rs` (#48): one `SamplerConfig` drives one pipeline — logit bias → penalties (repeat /
frequency / presence, last 64 tokens) → DRY → **grammar mask (F2)** → greedy shortcut
(`temp == 0`) → top-k → typical → top-p → min-p → XTC → temperature **or** mirostat v1/v2, seeded
`StdRng`. Every F3 knob defaults to a no-op, so the default path is bit-identical to the pre-#48
chain — pinned by `test_default_pipeline_matches_the_pinned_pre_f3_sequence`, and re-pinned through
the grammar-aware entry point by `test_default_pipeline_matches_the_pinned_pre_f2_sequence` (the
same sequence, captured from `master` before both changes).
`SamplerConfig::validate` refuses nonsensical values at CLI startup / HTTP 400 (never clamps
silently), and logit-bias token ids are checked against the vocabulary.
Mirostat's `mu` is caller-owned state (`MirostatState`: one per run / session / request / batch
slot); speculative decoding refuses mirostat (`--spec-draft`), because a verify round samples
several rows from one shared RNG. DRY sequence breakers are token-id sequences
(`--dry-sequence-breakers 198;13,2`); llama.cpp's string form needs a tokenizer port (follow-up).
CLI: `--temp --greedy --top-k --top-p --repeat-penalty --frequency-penalty --presence-penalty
--min-p --typical --xtc-probability --xtc-threshold --dry-multiplier --dry-base
--dry-allowed-length --dry-penalty-last-n --dry-sequence-breakers --mirostat --mirostat-tau
--mirostat-eta --mirostat-m --logit-bias --grammar --grammar-str --json-schema
--json-schema-str -n --seed -t`.

**Constrained decoding (F2, [#47](https://github.com/yusiwen/minfer/issues/47)).**
`src/grammar.rs` compiles a GBNF grammar (or a JSON Schema, through a generated
GBNF) into one pushdown automaton — a flat program per rule, a set of
`{rule, pc}` call stacks — and turns it into a per-state token bitset. The mask
is applied **inside** `sample_with_config_grammar`, after DRY and before the
greedy shortcut: every stage before it only shifts logits and every stage after
it only removes candidates, so a forbidden token can never be chosen, and the
mask consumes no RNG (mirostat/DRY are unperturbed). The compiled
`Arc<Grammar>` is per request; the mutable `GrammarState` is per run, exactly
like `MirostatState`. Token advancement is byte-level correct (a piece may be one
byte of a multi-byte character); a token whose pending bytes can never complete
to an accepted codepoint is rejected, EOG is legal only at an accepting state
with no pending bytes, and **no allowed token** is a loud stop
(`SampleError::NoAllowedToken`), never an arbitrary token. Unsupported GBNF or
schema constructs are startup/`400` refusals — never a silent guess; the
accepted subset and every refusal are catalogued in `docs/GRAMMAR-DESIGN.md`.
Server: `response_format` (`json_object` / `json_schema`) plus a `grammar`
extension field; the two together are a `400`. `--spec-draft` + a grammar is
refused (a verify round samples several rows from one automaton state).

## Dependencies

Core: `rand`, `regex`, `half`, `serde`+`serde_json`, `minijinja`. Server: `axum`/`tokio`/`tower-http`/`uuid`/… macOS: `objc2-*` family (2026-08-25 objc2 migration).

## Docs Index

All docs live in `docs/` (root keeps only `AGENTS.md` + `README.md`).

| Topic | Where |
|---|---|
| Architecture design (module map, pipeline, adding an arch) | `docs/ARCHITECTURE.md` |
| **End-to-end inference walkthrough (15-doc beginner series: CLI → GGUF → graph → kernels → backends)** | `docs/inference_e2e_walkthrough/` (index: `README.md`) |
| Compute graph design + implementation record | `docs/COMPUTE-GRAPH-DESIGN.md` |
| **Backend registry (F4/#57): the handle, the registered set, the two orders, the name surface, the three refusals; F5/#58: the async staging copy, the per-backend mechanism and the enumerated synchronization points (§11)** | `docs/BACKEND-REGISTRY-DESIGN.md` |
| llama.cpp compute-graph analysis | `docs/LLAMA-COMPUTE-GRAPH.md` |
| Metal optimization plans / gap analysis | `docs/METAL_OPTIMIZATIONS.md` |
| objc2 ecosystem + migration record | `docs/METAL_OBJC-ECOSYSTEM.md` |
| GPU safety conventions + audit | `docs/GPU_SAFETY.md` |
| CPU optimizations | `docs/CPU_OPTIMIZATIONS.md` |
| Metal backend design + implementation record (device layer, dispatch, command buffers, safety) | `docs/METAL-BACKEND-DESIGN.md` |
| CUDA backend design + implementation record (device layer, dispatch, capture, safety) | `docs/CUDA-BACKEND-DESIGN.md` |
| **CUDA optimization history (live status) + per-step records (incl. Phase 8)** | `docs/CUDA_OPTIMIZATION.md` + `docs/cuda_optimization_steps/` |
| CUDA / GPU technology primer (every technique explained) | `docs/CUDA-TECH-PRIMER.md` |
| Campaign glossary (every term/formula, classified into 7 layers) | `docs/GLOSSARY.md` |
| llama.cpp MMQ / speculative-decoding analyses | `docs/LLAMA-CPP-MMQ-ANALYSIS.md`, `docs/LLAMA-CPP-SPECULATIVE-ANALYSIS.md` |
| **Speculative decoding plan (D5, closed by measurement — doc 81)** | `docs/SPECULATIVE-DECODING-PLAN.md` |
| **Device adaptation layer (T-series, T1/T2 landed on master — plan + status)** | `docs/DEVICE-ADAPTATION-PLAN.md` |
| Qwen3 support plan (+ minijinja gotcha §5#9) | `docs/QWEN3-SUPPORT-PLAN.md` |
| **Chat templates + tokenizer pre-tokenization (F7/#50): accepted/refused constructs, the loud refusal, reference fixtures** | `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` |
| Qwen3-4B perf vs llama.cpp | `docs/PERF-QWEN3-4B-VS-LLAMACPP.md` |
| **Architecture roadmap (system layers: IR, scheduler, allocator, KV, batching, backends — gaps + prioritized backlog)** | `docs/ARCHITECTURE-ROADMAP.md` |
| **Architecture execution plan (phase-by-phase tickets, acceptance criteria, verification matrix)** | `docs/ARCHITECTURE-EXECUTION-PLAN.md` |
| **Grammar / JSON-schema constrained decoding (F2): accepted subset, refusals, mask position** | `docs/GRAMMAR-DESIGN.md` |
| Model support roadmap (which model families to port next) | `docs/MODEL-SUPPORT-ROADMAP.md` |
| OpenAI chat API plan | `docs/OPENAI-CHAT-API-PLAN.md` |
| CLI conversation plan | `docs/CLI-CONVERSATION-PLAN.md` |
| Inference-graph viz (`MINFER_TRACE` etc.) + end-to-end shape flow (`viz/e2e.html`) | `viz/README.md` |
| **GGUF tooling (F6/#49): the writer contract, `convert`/`quantize`/`split`, the supported/refused sets, the split convention, the references/tolerances** | `docs/GGUF-TOOLING.md` |
| Debug dump format | `docs/debug-dump.md` |
| Metal / multi-token kernel analyses | `docs/metal-inference-analysis.md`, `docs/multi-token-kernel-analysis.md` |
| Parameter audit, bug/debug notes, known issues | `docs/PARAMETER_AUDIT.md`, `docs/BUG-6-KV-CACHE-INDEXING.md`, `docs/DEBUGGING-*.md`, `docs/QWEN2.5-*.md`, `docs/KNOWN-CPU-ISSUES-2026-08-29.md` |
| Build / usage / support reference | `docs/BUILD.md`, `docs/USAGE.md`, `docs/SUPPORT-MATRIX.md`, `docs/FEATURES.md` |

**Open doc debt (GitHub issues):** [#62](https://github.com/yusiwen/minfer/issues/62) (`docs/USAGE.md` after
the elastic partition). The docs build was [#63](https://github.com/yusiwen/minfer/issues/63), **closed**: CI job
`check-docs` runs `scripts/build_book.sh` (the pinned mdBook toolchain plus the sha384-checked
Mermaid download) and `scripts/check_docs_links.py`, which fails
on a relative link whose target does not exist and names the file, line and target.
Test health was [#82](https://github.com/yusiwen/minfer/issues/82), **closed**: the three ignored real-model
tests that failed on master (two wrote into a directory nothing created; the third asserted the greedy 0.5B's stop
behaviour instead of the `need_insert_eot` invariant — a real-model test must assert the engine's rule, not the
model's mood). A finding from a gate run belongs here too — a red baseline makes every later gate run ambiguous.
