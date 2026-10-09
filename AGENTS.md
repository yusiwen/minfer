# minfer — AI Agent Context

Pure-Rust LLM inference engine written from scratch (~55k LOC of Rust, ~93k with tests), llama.cpp-inspired, 0 ML framework deps.
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
├── gguf.rs          # GGUF v3 parser
├── gguf_write.rs    # F6: GGUF v3 writer — header/metadata/tensor index, ggml_pad, split
├── quantize.rs      # F6: weight encoders (byte-identical to llama.cpp) + loud refusal
├── convert.rs       # F6: HF Qwen2 checkpoint → GGUF; strict-loader metadata; QuantizePlan
├── tooling.rs       # F6: convert / quantize / split subcommands
├── block.rs         # quantized block types (repr(C), ggml-common.h layout)
├── quants.rs        # AVX2 / NEON+SDOT dot kernels + Q8_0/Q8_K quantization — module decider
├── quants/          # dot_q4_0 · dot_q4_1 · dot_q5 · dot_q8_0 · kquant · quantize_q8_0 · quantize_q8_k · avx2 · neon (split by #264)
├── kernel.rs        # quantized matmul dispatch + scalar fallbacks; shared worker pool — module decider
├── kernel/          # dispatch · pool · embed (split by #264)
├── vec_ops.rs       # RMSNorm, RoPE, Softmax, SiLU; vectorized f16 weight dot — module decider
├── vec_ops/         # vec · rms_norm · rope · softmax · silu · f16 · bf16 · neon (split by #264)
├── tensor.rs        # 4D Tensor (shape/strides/data)
├── dump.rs          # debug dump module (--features debug_dump)
├── tokenizer.rs     # byte-level BPE from GGUF metadata; `tokenizer.ggml.pre` splitter
├── sampler.rs       # SamplerConfig pipeline: penalties → DRY → grammar → top-k/typical/top-p/min-p/XTC → temp | mirostat
├── grammar.rs       # F2: GBNF + JSON-schema → pushdown automaton → per-state token mask
├── template.rs      # chat templates (minijinja + Python-str hook); loud refusal, no ChatML fallback
├── conversation.rs  # multi-turn session (append-only KV; C2 overflow drops oldest + re-ropes)
├── server/          # OpenAI-compatible axum server; batch.rs = continuous batching, metrics.rs = /metrics
├── download/mod.rs  # HuggingFace + Ollama auto-download
├── metal.rs         # MPS module root: type defs + private dispatch primitives (split by #265)
├── metal/           # L1/L2: runtime.rs · encode.rs · ops.rs · policy.rs (split by #265)
│   └── kernels/     # L3 kernel sources: common.h · dequantize.h + 14 family .metal (split by #265)
├── cuda.rs          # CUDA device layer (L1), feature-gated (graph backend: graph/cuda_backend.rs); device_memory() → allocplan::DeviceMemory
├── cuda/            # L2 launch/dispatch: ffi_runtime.rs + methods.rs + methods/<family>.rs (split by #262)
├── cuda/kernels/    # L3 kernel sources: common.cuh + 17 .cu translation units, one per kernel family (split by #263)
├── device_tier.rs   # cc-keyed CUDA device-tier table + selector
└── models/          # ModelDef trait + per-arch mod/graph/loader (qwen2/, qwen3/)
```

Each device backend is organised as L1 runtime, L2 launch/dispatch, L3 kernel sources and L4 graph executor, and **only L4 is polymorphic** — `graph/backend.rs` + `graph/registry.rs` are the single device seam. The full layering rationale, the shared-`common` rule and the completed splits: [`docs/SOURCE-LAYOUT-PLAN.md`](docs/SOURCE-LAYOUT-PLAN.md).

`src/graph/`: `mod.rs` ComputeGraph/CNode · `ops.rs` Op + NodeMeta · `builder.rs` GraphBuilder · `scheduler.rs` assign → split → execute · `backend.rs` + `cpu_backend.rs`/`metal_backend.rs`/`cuda_backend.rs` executors · `registry.rs` backend registry (F4; F5's `copy_cross`/`await_cross`) · `alloc.rs` liveness allocator + persistent KV regions (E4) · `allocplan.rs` size-class ladder + pure plan + `DeviceMemory`/`budget_decision` · `offload.rs` layer offload plan (E5) · `kvcache.rs` cell store, removal/shift/compaction, span list, prefix sharing (C1–C3, C8b) · `kvformat.rs` KV format + `MINFER_CACHE_TYPE` gate (C4) · `kvsession.rs` versioned KV session container (C5) · `cache.rs`/`params.rs` params-only graph reuse · `fusion.rs` SwiGLU fusion · `copystats.rs` split-boundary counters · `batch.rs` batch composition · `dot.rs`/`json.rs` exporters.

Unit tests live beside their module as `<module>/tests.rs`, declared `#[cfg(test)] mod tests;` (a long module splits into `<module>/tests/<topic>.rs`, each named by a `mod <topic>;`). Enforced by `scripts/check_source_layout.py`: an inline `#[cfg(…test…)] mod … {` is rejected, and so is a `src/**.rs` no `mod` declaration names. The rationale and the measured build-hygiene note: [`docs/SOURCE-LAYOUT-PLAN.md`](docs/SOURCE-LAYOUT-PLAN.md).

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
./target/release/minfer convert <hf-model-dir> out.gguf [--outtype f16|bf16|f32] [--split-max-size N]
./target/release/minfer quantize in.gguf out.gguf --type q4_0 [--split-max-size N]
./target/release/minfer split in.gguf <out-dir> --max-size 200M [--stem NAME]
```

- CUDA test suite on a real GPU: `scripts/cuda_test.sh` (`cargo test --release --features cuda -- --test-threads=1`). CI has **no** GPU — its CUDA job only compiles the harness — so this is the only way to run the device-gated tests, and the device state is a process-wide singleton (`CudaState`), so **run it serially** ([#64](https://github.com/yusiwen/minfer/issues/64)). Local GPU runs must rebuild the CLI *with* the feature: a plain `cargo test --release` overwrites `target/release/minfer` with a CPU-only build, which silently measures the CPU. Per-instance streams and capture, the launch-return audit and the injection lever: [`docs/CUDA-BACKEND-DESIGN.md`](docs/CUDA-BACKEND-DESIGN.md) ("Running on a real device").
- Real-model gates (the `#[ignore]`d set): `scripts/real_model_gates.sh` (`FEATURES=cuda` selects the device build; `PARALLEL=1`/`0` forces the parallel/serial form). The wrapper defaults to **serial on a device build** (the `CudaState` singleton) and to the **parallel** form on a CPU-only build. Run the set twice: the cached 0.5B (f32 KV) **and** `MINFER_BATCH_TEST_MODEL=~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (f16 KV, hd 128 — the only combination that reaches FA prefill and the half-width cell stride). **The records are not kept here.** Every suite measurement — dated, box-labelled and machine-checked against `docs/status.toml` — lives in [`docs/TEST-BASELINES.md`](docs/TEST-BASELINES.md); write a new record there and update `docs/status.toml` in the same commit.
  Read [`docs/GATE-CONTRACT.md`](docs/GATE-CONTRACT.md) before writing or changing a gate: the five rules and the failure-injection seam (`MINFER_TEST_CALL_FAIL`, [#171](https://github.com/yusiwen/minfer/issues/171)). The PR-body shape (the two gate facts and the `Mac verification` record, [#335](https://github.com/yusiwen/minfer/issues/335)) is enforced by `scripts/check_pr_body.py` in the `check-pr-body` job ([#175](https://github.com/yusiwen/minfer/issues/175)): presence, not truth. Per-ticket history lives in the records, not here.
- Sandboxed agent shells: if `nvidia-smi` reports `Failed to initialize NVML: Unknown Error` and `cuInit` returns 304 while `/dev/nvidia*` exists, the *file sandbox* (Landlock) is denying `open()` with `EACCES` even on `crw-rw-rw-` nodes — that is **not** evidence of a broken driver. Check with a widened sandbox before recording "no device".
- Batching default (E6): `chat::batch_mode(requested, model.device())` — pure and unit-tested, so CI covers the matrix. `ModelDef::device()` (`Device::{Cpu,Metal,Cuda}`) is the single authority for "the device participates", shared with the graph builder's `CParams.gpu`.
- Full CLI + options: `docs/USAGE.md` (stale in places — [#62](https://github.com/yusiwen/minfer/issues/62)). CUDA build details (ccbin pinning, GPU arch coverage, cudart linking): `docs/BUILD.md`.
- Multi-part GGUF: written by `minfer split` (F6) as `{stem}-NNNNN-of-MMMMM.gguf`, all parts parsed into one merged tensor index. Details: [`docs/GGUF-TOOLING.md`](docs/GGUF-TOOLING.md) ("Multi-part GGUF files").
- **Nested worktrees**: put the tree *inside* the workspace — `.worktrees/<scope>`, via `scripts/agent_worktree.sh new|rm|list` (the `workspace-write` policy denies a worktree beside the root), and give it its **own** `target/` — never point `CARGO_TARGET_DIR` at the outer tree, or a plain `cargo test --release` there silently overwrites the outer `target/release/minfer` with a CPU-only build. The long-form rules and the per-failure records are in the global `~/.dsh/AGENTS.md` §"Parallel Work: Nested Worktrees", which this project's instruction budget currently omits from the session — read it explicitly when worktree work gets involved.

## Support

- Quants (CPU + GPU): **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K**. Not supported: Q2_K/Q3_K/I-quants. Full matrix incl. CUDA notes: `docs/SUPPORT-MATRIX.md`. The K-quant dots have AVX2/FMA and AVX-512/VNNI kernels on x86_64 (bitwise-identical to the scalar reference; [#56](https://github.com/yusiwen/minfer/issues/56)); weight repacking is the remaining F1 increment.
- Activations: CPU quantizes to Q8_0 on the fly (Q8_K for K-quant weights); GPU backends read f32 (CUDA prefill uses int8 MMQ).
- Verified models (CPU + graph-GPU, greedy output matches llama.cpp where noted in docs): Qwen2.5-0.5B Q4_0/Q4_K_M/Q5_K_M · Qwen2.5-7B Q4_K_M · Qwen3-0.6B Q8_0 · Qwen3-4B Q4_K_M (KV sized by `--n-ctx`, see `docs/PERF-QWEN3-4B-VS-LLAMACPP.md`) · DeepSeek-R1-Distill-Qwen-1.5B (needs the tokenizer special-token match).
- Chat templates (F7/#50): the GGUF `tokenizer.chat_template` is rendered by minijinja with a Python-`str`-method hook. A template that cannot be rendered refuses the load; a tokenizer whose `tokenizer.ggml.pre` is not `qwen2`/`qwen35` refuses the load too. Reference renderings live in `tests/fixtures/chat/` (transformers 5.17.0, provenance in the fixtures); token-id equality is gated by `tests/fixtures/tokenizer/ids_*.json` plus the `token_ids_match_the_reference` `#[ignore]`d test. Details: `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`.
- Qwen3.5 (`qwen35` arch) is **not** a supported architecture; its GGUF is used only as the `qwen35` pre-tokenizer/id reference.
- **f16 / bf16 weights** (F6/#49; CUDA [#141](https://github.com/yusiwen/minfer/issues/141)/[#208](https://github.com/yusiwen/minfer/issues/208), Metal [#164](https://github.com/yusiwen/minfer/issues/164)/[#208](https://github.com/yusiwen/minfer/issues/208)): both run on **CPU, CUDA and Metal** for both supported architectures; 2-D weights stay 2 B/element and promote in-register, **1-D norms/biases stay f32**, and bf16 conversion is round-to-nearest-even. File contract, byte-identity references and the measured device deltas: [`docs/SUPPORT-MATRIX.md`](docs/SUPPORT-MATRIX.md) ("f16 and bf16 weights").

## GPU Safety

Read `docs/GPU_SAFETY.md` before touching Metal/CUDA code. Hard rules: `submit()` waits bounded + checks status (never blocks forever); no early return past a `threadgroup_barrier`; device limits queried at runtime, never hardcoded; guard failures abort with actual values. In the graph, **kernel-invariant violations return `Err` from `execute_node` — never a silent CPU fallback**; backend assignment is decided at build time.

## Compute Graph — core rules

Inference = build `ComputeGraph` → assign backends → fuse → allocate → execute; one graph per `GraphParams`, reused across decode steps. Full design: `docs/COMPUTE-GRAPH-DESIGN.md`.

1. **KV positions are data, not structure** — topology never depends on `n_past` (the precondition for decode reuse); `positions` (what RoPE rotates by, and the causal bound) and `cells` (the arena row) are distinct, and a backend expresses its read window as causal `positions`, one `[lo, hi)` `attn_span`, or `KV_MAP_MAX_SPANS` runs. Elaboration: [`docs/KV-CACHE-DESIGN.md`](docs/KV-CACHE-DESIGN.md) §1.
2. Each layer owns **two persistent KV regions** (K/V) via `kv_pair(layer)`; they survive rebuilds, and the cell store owns ownership, removal, compaction and cross-sequence sharing — with **one** fill entry point that copy-on-writes before the first cell resolves. Elaboration: [`docs/KV-CACHE-DESIGN.md`](docs/KV-CACHE-DESIGN.md) §2.
3. **Reuse is params-only**: `GraphParams` (`CParams.gpu`, `CParams.gpu_layers`, `CParams.kv_format`) deterministically fixes the topology; `GraphCache::try_reuse` compares params only.
4. Weight layout = GGUF: metadata `[in, out]`, memory row-major `[out][in]`; activations token-major `[nt][d]`. I32 inputs are stored as `f32::from_bits` via `fill_input_i32`.
5. In-place ops (`Silu`, `RoPE`) alias their input buffer (sole consumer + same backend only). **Never host-copy a GPU-pending buffer** (Phase-3 KV-corruption bug).
6. Execution follows build order (a valid topo order); allocator liveness uses the same order, not `topo_order()` (G3 regression); input buffers are never freed.
7. Decode fusions: `Op::FusedQKV` (concat matmul + bias/rope/store) and `Op::FusedFFN` (gate+up concat + swiglu) — gated and part of the reuse identity (`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` disable them; `MINFER_FFN_COMPOSITION=1` builds the proven D2 *composition* instead — see plan §D3). Fused vs unfused is bit-identical; when comparing, the unfused path MUST run the FusionPass.
8. Backends own their buffer pools; the allocator is the single owner. **The scheduler's split boundary enqueues one cross-backend staging copy per `Split::inputs` entry and waits once per copy, at the consumer's first read** (F5, [#58](https://github.com/yusiwen/minfer/issues/58)). Elaboration: [`docs/BACKEND-REGISTRY-DESIGN.md`](docs/BACKEND-REGISTRY-DESIGN.md) §11.
9. CPU quantizes activations to Q8_0, GPU reads f32 — CPU-vs-GPU logits differ by design; compare each path against its own reference.
10. **A node's one output can be several tensors.** `GraphBuilder::split_parts(owner, sizes)` exposes contiguous parts of it as independent `Op::View` aliases (D1 increment 3), so the graph stays single-output and no backend gains a second-output path. An indices part is i32 in f32 bit patterns (rule 4) — that is how a `(values, indices)` producer drives `Op::GetRows` (the MoE-routing shape). D2's `fused_ffn_composition` consumes a concat matmul this way.
11. **The KV storage format is a gate, never a guess (C4), and it is per engine ([#99](https://github.com/yusiwen/minfer/issues/99))** — `MINFER_CACHE_TYPE` resolves once per load into `KvFormat { f32, f16, q8_0 }`, there is deliberately **no** process-wide global, and a backend must answer `reads_packed_kv` before a packed cell may exist (true on CPU, CUDA and Metal). Elaboration: [`docs/KV-CACHE-DESIGN.md`](docs/KV-CACHE-DESIGN.md) §3.
12. **A KV session is a file with a header, never a memory dump (C5)** — versioned and checksummed, verified whole before anything is applied, and refused outright for a mixed offload plan. Elaboration: [`docs/KV-CACHE-DESIGN.md`](docs/KV-CACHE-DESIGN.md) §4.
13. **Memory is accounted before it is allocated (E4)** — weights + pooled + this allocation at its size class is checked against the backend's budget *before* the pool is asked, and the refusal names the numbers. Elaboration: [`docs/MEMORY-POLICY-DESIGN.md`](docs/MEMORY-POLICY-DESIGN.md) §1.
14. **Layer offload is one plan read in three places (E5)** — the loader, the builder and the assignment pass must agree on `OffloadPlan { gpu_layers, n_layers }`, and the `auto` fit takes the largest *prefix*, never a knapsack. Elaboration: [`docs/MEMORY-POLICY-DESIGN.md`](docs/MEMORY-POLICY-DESIGN.md) §2.

## Core Conventions

1. CPU matmuls: quantized weight × Q8_0 activations (`dot_q*_q8_0()`); GPU reads f32 activations directly.
2. SIMD: AVX2 (x86) / NEON+SDOT (aarch64, inline asm) with scalar fallbacks; the K-quant dots also have an AVX-512/VNNI path on `avx512f/bw/dq/vl/vnni`. `MINFER_NO_NEON=1` (aarch64), `MINFER_NO_AVX2=1` (whole x86 quants SIMD layer) and `MINFER_NO_AVX512=1` (AVX-512 only, falls back to AVX2) force the lower path.
3. No ML frameworks — all ops handwritten; tensor data is raw `&[u8]`; GGUF padding via `ggml_pad()`.
4. Cross-backend: per-op assignment decided at build time via `supports_op` — offered in the backend registry's priority order (F4: Metal 300, CUDA 200, CPU 100; `docs/BACKEND-REGISTRY-DESIGN.md`) rather than in a hardcoded chain; guard failures abort — never silent mid-run fallback.
5. **The non-test build is warning-free, and gated** (`#![cfg_attr(not(test), deny(warnings))]` in `src/main.rs`); every dead-code `allow` must carry an allowed shape and a reason. The rules, the stripped oracle and the baseline: [`docs/GATE-CONTRACT.md`](docs/GATE-CONTRACT.md) §1 ("the dead-code annotation rules").

## Extending

**New architecture** (mirror `models/qwen2/` / `models/qwen3/`): create `models/<name>/{mod,graph,loader}.rs` with `HParams` + `LayerWeights`; dispatch in `models/mod.rs::load_model()`; build the graph with `GraphBuilder` — deterministic in `GraphParams` (reuse invariant); implement `ModelDef` (`forward`/`build_graph`/`forward_graph`/`as_any`); add a chat template if needed.

**New backend** (CUDA is the worked example — `docs/CUDA-BACKEND-DESIGN.md`; the registry contract is `docs/BACKEND-REGISTRY-DESIGN.md`): implement the `Backend` trait (`src/graph/backend.rs`) with its capability matrix as module-level free functions (the trait methods forward to them, so the registry's answer and the trait's answer are one authority); extend the fixed id space in `src/graph/registry.rs` (`Backend::<NAME>`, a `NAMES` entry, an id — the id is a KV-session file-format contract, so it is **appended**, never renumbered) and write the module's `entry()`/`register()`: priority, caps (`reads_packed_kv` is the per-format capability query), and the `pool`/`pool_mut`/`host_read`/`kv_format`/`enable`/`unavailable` hooks; call `register()` from `Registry::build` under its `#[cfg]` gate. **No consumer needs editing** — the allocator's dispatch, the scheduler's execute, the fusion wiring, the exporters and the KV-session tags all read the registry. Then register weights at load and gate execution on all-weights-registered, and record participation in `CParams.gpu` (`Qwen2Graph::device` — remember the `active_filter` fence, so `--backend cpu` keeps a device-only fused node out of the graph).

## Sampling

`sampler.rs` (#48): one `SamplerConfig` drives one pipeline — logit bias → penalties → DRY → **grammar mask (F2)** → greedy shortcut → top-k → typical → top-p → min-p → XTC → temperature **or** mirostat; every F3 knob defaults to a no-op, so the default path is bit-identical to the pre-#48 chain (pinned by `test_default_pipeline_matches_the_pinned_pre_f3_sequence` and its grammar-aware twin). `SamplerConfig::validate` refuses nonsensical values (never clamps), mirostat's `mu` is caller-owned, `--spec-draft` refuses mirostat, and `--spec-draft` + a grammar is refused. CLI flags and defaults: [`docs/USAGE.md`](docs/USAGE.md) ("Sampler pipeline and invariants"); the mask position and every refusal: [`docs/GRAMMAR-DESIGN.md`](docs/GRAMMAR-DESIGN.md).

## Dependencies

Core: `rand`, `regex`, `half`, `serde`+`serde_json`, `minijinja`. Server: `axum`/`tokio`/`tower-http`/`uuid`/… macOS: `objc2-*` family (2026-08-25 objc2 migration). Dev-only: `tokio`'s `io-util` (F8's `/metrics` test) and `sha2` (#205's fixture-manifest content check — a real hash is not something to hand-write in a test, and it is not in the production graph).

## Docs Index

All docs live in `docs/` (root keeps only `AGENTS.md` + `README.md`). **The table of contents is
[`docs/SUMMARY.md`](docs/SUMMARY.md)** — every chapter is listed there. This is the short router for the docs
an agent reaches for first.

| Reach for | Where |
|---|---|
| Architecture: module map, pipeline, adding an architecture | `docs/ARCHITECTURE.md` |
| End-to-end walkthrough (15 docs: CLI → GGUF → graph → kernels → backends) | `docs/inference_e2e_walkthrough/` |
| Compute graph design | `docs/COMPUTE-GRAPH-DESIGN.md` |
| KV cache: cells, the format gate, sessions | `docs/KV-CACHE-DESIGN.md` |
| Memory accounting and layer offload | `docs/MEMORY-POLICY-DESIGN.md` |
| Backend registry (F4/F5) + the async staging copy | `docs/BACKEND-REGISTRY-DESIGN.md` |
| CUDA / Metal backend design + records | `docs/CUDA-BACKEND-DESIGN.md`, `docs/METAL-BACKEND-DESIGN.md` |
| **The gate contract — read before writing or changing a gate** | `docs/GATE-CONTRACT.md` |
| Suite baselines (dated, box-labelled) | `docs/TEST-BASELINES.md` |
| GGUF tooling: convert / quantize / split (F6) | `docs/GGUF-TOOLING.md` |
| Grammar / JSON-schema constrained decoding (F2) | `docs/GRAMMAR-DESIGN.md` |
| Source layout plan (device-first layering) | `docs/SOURCE-LAYOUT-PLAN.md` |
| Support matrix · build · usage · features | `docs/SUPPORT-MATRIX.md`, `docs/BUILD.md`, `docs/USAGE.md`, `docs/FEATURES.md` |
| What is missing, and in what order | `docs/ARCHITECTURE-ROADMAP.md` |
| The phase-by-phase plan and the per-ticket history | `docs/ARCHITECTURE-EXECUTION-PLAN.md` |

**Doc gates.** The `check-docs` CI job runs `scripts/build_book.sh` (mdBook) and the checkers
`check_docs_links.py`, `check_status.py --check`, `check_doc_line_anchors.py`, `check_anchor_drift.py` and
`check_f6_fixtures.py`; what each can and cannot prove is in its own `--help` and in
[`docs/GATE-CONTRACT.md`](docs/GATE-CONTRACT.md) ("the doc gates"). Edit `docs/status.toml`, never a counter.
**Open doc debt:** [#62](https://github.com/yusiwen/minfer/issues/62) (`docs/USAGE.md`).