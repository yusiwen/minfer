# minfer — AI Agent Context

Pure-Rust LLM inference engine written from scratch (~4400 LOC), llama.cpp-inspired, 0 ML framework deps.
Qwen2/Qwen2.5 + Qwen3 (dense) · CPU + Metal (macOS) + CUDA (opt-in) · GGUF v3.
Inference runs through a **declarative compute graph** (builder → scheduler → per-backend kernels) — design + implementation record: `docs/GRAPH-REFACTOR-PLAN.md`.

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
├── block.rs         # quantized block types (repr(C), ggml-common.h layout)
├── quants.rs        # AVX2 / NEON+SDOT dot kernels + Q8_0/Q8_K quantization
├── kernel.rs        # quantized matmul dispatch + CPU scalar fallbacks
├── vec_ops.rs       # RMSNorm, RoPE, Softmax, SiLU
├── tensor.rs        # 4D Tensor (shape/strides/data)
├── cache.rs         # legacy KV cache type (graph path owns KV in the allocator)
├── dump.rs          # debug dump module (--features debug_dump)
├── tokenizer.rs     # BPE tokenizer (self-contained, from GGUF metadata)
├── sampler.rs       # repeat-penalty / top-k / top-p / temperature
├── template.rs      # chat templates (minijinja) — 2.21.0 has no `str` methods; Qwen3's template falls back to ChatML (docs/QWEN3-SUPPORT-PLAN §5#9)
├── conversation.rs  # multi-turn session (append-only KV)
├── server/          # OpenAI-compatible HTTP server (axum)
├── download/mod.rs  # HuggingFace + Ollama auto-download
├── metal.rs + metal.metal  # MPS kernels + shaders (graph backend: graph/metal_backend.rs)
├── cuda.rs          # CUDA device layer, feature-gated (graph backend: graph/cuda_backend.rs)
└── models/          # ModelDef trait + per-arch mod/graph/loader (qwen2/, qwen3/)
```

`src/graph/`: `mod.rs` ComputeGraph/CNode · `ops.rs` Op enum + NodeMeta · `builder.rs` GraphBuilder · `alloc.rs` liveness allocator + persistent KV regions · `backend.rs` Backend trait · `cpu_backend.rs` / `metal_backend.rs` / `cuda_backend.rs` executors · `scheduler.rs` assign → split → execute · `fusion.rs` SwiGLU/BiasRope fusion · `cache.rs` + `params.rs` params-only graph reuse · `dot.rs` DOT export · `json.rs` graph JSON export for viz.

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
MINFER_GRAPH_DUMP=/tmp/d  ./target/release/minfer <model> "hello"  # graph logits/KV dump (any build)
MINFER_TRACE=/tmp/t.json  ./target/release/minfer <model> "hello"  # per-node real-data trace for viz/
./target/release/minfer viz <model>                                # viz server (page + live SSE)
```

- Full CLI + options: `docs/USAGE.md`. CUDA build details (ccbin pinning, GPU arch coverage, cudart linking): `docs/BUILD.md`.
- Multi-part GGUF: entry is part 0, all parts parsed into one merged tensor index; download resume is size-checked.

## Support

- Quants (CPU + GPU): **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K**. Not supported: Q2_K/Q3_K/I-quants. Full matrix incl. CUDA notes: `docs/SUPPORT-MATRIX.md`.
- Activations: CPU quantizes to Q8_0 on the fly (Q8_K for K-quant weights); GPU backends read f32 (CUDA prefill uses int8 MMQ).
- Verified models (CPU + graph-GPU, greedy output matches llama.cpp where noted in docs): Qwen2.5-0.5B Q4_0/Q4_K_M/Q5_K_M · Qwen2.5-7B Q4_K_M · Qwen3-0.6B Q8_0 · Qwen3-4B Q4_K_M (KV sized by `--n-ctx`, see `docs/PERF-QWEN3-4B-VS-LLAMACPP.md`) · DeepSeek-R1-Distill-Qwen-1.5B (needs the tokenizer special-token match).

## GPU Safety

Read `docs/GPU_SAFETY.md` before touching Metal/CUDA code. Hard rules: `submit()` waits bounded + checks status (never blocks forever); no early return past a `threadgroup_barrier`; device limits queried at runtime, never hardcoded; guard failures abort with actual values. In the graph, **kernel-invariant violations return `Err` from `execute_node` — never a silent CPU fallback**; backend assignment is decided at build time.

## Compute Graph — core rules

Inference = build `ComputeGraph` → assign backends → fuse → allocate → execute; one graph per `GraphParams`, reused across decode steps. Full design: `docs/GRAPH-REFACTOR-PLAN.md`.

1. **KV positions are data, not structure** — topology never depends on `n_past` (precondition for decode reuse).
2. Each layer owns **two persistent KV regions** (K/V) via `kv_pair(layer)`; they survive rebuilds (allocator lives in `GraphCache`).
3. **Reuse is params-only**: `GraphParams` (+ `CParams.gpu`) deterministically fixes the topology; `GraphCache::try_reuse` compares params only.
4. Weight layout = GGUF: metadata `[in, out]`, memory row-major `[out][in]`; activations token-major `[nt][d]`. I32 inputs stored as `f32::from_bits` via `fill_input_i32`.
5. In-place ops (`Silu`, `RoPE`) alias their input buffer (sole consumer + same backend only). **Never host-copy a GPU-pending buffer** (Phase-3 KV-corruption bug).
6. Execution follows build order (valid topo order); allocator liveness uses the same order, not `topo_order()` (G3 regression); input buffers are never freed.
7. Decode fusions: `Op::FusedQKV` (concat matmul + bias/rope/store) and `Op::FusedFFN` (gate+up concat + swiglu) — gated, and part of the reuse identity (`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` to revert). Fused vs unfused is bit-identical; when comparing, the unfused path MUST run the FusionPass.
8. Backends own their buffer pools; the allocator is the single owner. The scheduler syncs + copies cross-backend at split boundaries; one Metal command buffer per split.
9. CPU quantizes activations to Q8_0, GPU reads f32 — CPU-vs-GPU logits differ by design; compare each path against its own reference.

## Core Conventions

1. CPU matmuls: quantized weight × Q8_0 activations (`dot_q*_q8_0()`); GPU reads f32 activations directly.
2. SIMD: AVX2 (x86) / NEON+SDOT (aarch64, inline asm) with scalar fallbacks; `MINFER_NO_NEON=1` forces scalar.
3. No ML frameworks — all ops handwritten; tensor data is raw `&[u8]`; GGUF padding via `ggml_pad()`.
4. Cross-backend: per-op assignment decided at build time via `supports_op`; guard failures abort — never silent mid-run fallback.

## Extending

**New architecture** (mirror `models/qwen2/` / `qwen3/`): create `models/<name>/{mod,graph,loader}.rs` with `HParams` + `LayerWeights`; dispatch in `models/mod.rs::load_model()`; build the graph with `GraphBuilder` — deterministic in `GraphParams` (reuse invariant); implement `ModelDef` (`forward`/`build_graph`/`forward_graph`/`as_any`); add a chat template if needed.

**New backend** (CUDA is the worked example — `docs/CUDA-BACKEND-PLAN.md`): implement the `Backend` trait (`src/graph/backend.rs`: `supports_op`/`supports_fused`, buffer pool, `execute_node`, host read/write, `synchronize`); register it in `GraphAllocator` (priority + sync/copy arms); register weights at load and gate execution on all-weights-registered; record participation in `CParams.gpu`.

## Sampling

`sampler.rs`: repeat-penalty (last 64 tokens) → top-k → top-p → temperature, seeded `StdRng`; defaults match llama.cpp (0.8 / 0.95 / 1.1). CLI: `--temp --greedy --top-k --top-p --repeat-penalty -n --seed -t`.

## Dependencies

Core: `rand`, `regex`, `half`, `serde`+`serde_json`, `minijinja`. Server: `axum`/`tokio`/`tower-http`/`uuid`/… macOS: `objc2-*` family (2026-08-25 objc2 migration).

## Docs Index

All docs live in `docs/` (root keeps only `AGENTS.md` + `README.md`).

| Topic | Where |
|---|---|
| Architecture design (module map, pipeline, adding an arch) | `docs/ARCHITECTURE.md` |
| **End-to-end inference walkthrough (15-doc beginner series: CLI → GGUF → graph → kernels → backends)** | `docs/inference_e2e_walkthrough/` (index: `README.md`) |
| Compute graph design + implementation record | `docs/GRAPH-REFACTOR-PLAN.md` |
| llama.cpp compute-graph analysis | `docs/LLAMA-COMPUTE-GRAPH.md` |
| Metal optimization plans / gap analysis | `docs/METAL_OPTIMIZATIONS.md` |
| objc2 ecosystem + migration record | `docs/METAL_OBJC-ECOSYSTEM.md` |
| GPU safety conventions + audit | `docs/GPU_SAFETY.md` |
| CPU optimizations | `docs/CPU_OPTIMIZATIONS.md` |
| CUDA backend design + implementation record (Phase 7a–7e) | `docs/CUDA-BACKEND-PLAN.md` |
| **CUDA optimization history (live status) + per-step records (incl. Phase 8)** | `docs/CUDA_OPTIMIZATION.md` + `docs/cuda_optimization_steps/` |
| CUDA / GPU technology primer (every technique explained) | `docs/CUDA-TECH-PRIMER.md` |
| Campaign glossary (every term/formula, classified into 7 layers) | `docs/GLOSSARY.md` |
| llama.cpp MMQ / speculative-decoding analyses | `docs/LLAMA-CPP-MMQ-ANALYSIS.md`, `docs/LLAMA-CPP-SPECULATIVE-ANALYSIS.md` |
| **Speculative decoding plan (D5, closed by measurement — doc 81)** | `docs/SPECULATIVE-DECODING-PLAN.md` |
| Qwen3 support plan (+ minijinja gotcha §5#9) | `docs/QWEN3-SUPPORT-PLAN.md` |
| Qwen3-4B perf vs llama.cpp | `docs/PERF-QWEN3-4B-VS-LLAMACPP.md` |
| Architecture roadmap | `docs/ARCHITECTURE-ROADMAP.md` |
| OpenAI chat API plan | `docs/OPENAI-CHAT-API-PLAN.md` |
| CLI conversation plan | `docs/CLI-CONVERSATION-PLAN.md` |
| Inference-graph viz (`MINFER_TRACE` etc.) | `viz/README.md` |
| Debug dump format | `docs/debug-dump.md` |
| Metal / multi-token kernel analyses | `docs/metal-inference-analysis.md`, `docs/multi-token-kernel-analysis.md` |
| Parameter audit, bug/debug notes, known issues | `docs/PARAMETER_AUDIT.md`, `docs/BUG-6-KV-CACHE-INDEXING.md`, `docs/DEBUGGING-*.md`, `docs/QWEN2.5-*.md`, `docs/KNOWN-CPU-ISSUES-2026-08-29.md` |
| Build / usage / support reference | `docs/BUILD.md`, `docs/USAGE.md`, `docs/SUPPORT-MATRIX.md`, `docs/FEATURES.md` |
