# minfer

A minimal local LLM inference engine built from scratch in Rust.

<p align="center">

[![Rust](https://img.shields.io/badge/lang-Rust-orange?logo=rust&logoColor=white)](#)
[![Rust edition 2021](https://img.shields.io/badge/edition-2021-informational)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)
[![Language](https://img.shields.io/github/languages/top/yusiwen/minfer)](#)
[![Conventional commits](https://img.shields.io/badge/conventional--commits-%E2%9C%93-green)](#)
[![Apple MPS](https://img.shields.io/badge/Apple-MPS-blue?logo=apple&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-opt--in-brightgreen)](#)
[![CI](https://img.shields.io/github/actions/workflow/status/yusiwen/minfer/ci.yml?branch=master&logo=githubactions&label=CI)](#)

</p>

**📚 Documentation** — the full project docs site (architecture, compute graph, backends, tooling): [yusiwen.cn/minfer](https://yusiwen.cn/minfer/)

![minfer CLI conversation (Qwen3-0.6B, Qwen3-0.6B-Q8_0)](docs/cli-conversation.png)

## Features

> Full detail: [docs/FEATURES.md](docs/FEATURES.md) — also on the [docs site](https://yusiwen.cn/minfer/FEATURES.html)

- **Declarative compute graph** — pure-IR graph + scheduler (llama.cpp-inspired), params-only reuse, DOT export
- **Interactive graph visualization** — `minfer viz <model>`: live SSE inference, per-node stats/heatmaps in the browser
- **GGUF v3 loader** — split multi-part, mmap'd weights shared zero-copy with the GPU
- **Self-contained BPE tokenizer** — from GGUF metadata, no tiktoken; special-token templates match llama.cpp exactly
- **CPU: AVX2 / NEON+SDOT** — all 8 quant dots SIMD'd + persistent thread pool
- **GPU: Metal** — fused flash attention, simdgroup GEMM, precompiled `.metallib`, f16 KV cache
- **GPU: CUDA** — default-on int8 tensor-core MMQ path: **~3581 tok/s @7B q4_K_m prefill = 1.080× llama.cpp**; CUDA Graph capture, flash attention, memory/speed opt-out gates
- **Qwen2 / Qwen3** — GQA, SwiGLU, RoPE, RMSNorm; Qwen3 decoupled head dim + per-head Q/K norm
- **Model download** — Hugging Face Hub / Ollama, resumable
- **Multi-turn conversation CLI** (`--cnv`) — incremental prefill, session persistence
- **OpenAI-compatible server** (`serve`) — `/v1/chat/completions` streaming, multi-slot
- **No ML framework** — pure Rust, minimal runtime deps, all kernels handwritten

> Supported quantization formats & model architectures:
> [docs/SUPPORT-MATRIX.md](docs/SUPPORT-MATRIX.md) — also on the
> [docs site](https://yusiwen.cn/minfer/SUPPORT-MATRIX.html)

## Interactive Web Visualization (viz/)

The inference compute graph can be viewed interactively in the browser. A toolbar switch
offers **two views of the same graph**:

- **Operators** — the layered tensor grid: one node = one operator, one edge = a tensor data
  flow. Nodes are colored by backend + data magnitude, with per-node tensor stats, heatmaps,
  and logits top-5.
- **Pipeline** — a semantic reasoning pipeline: one box = one function/stage
  (`Input → Embedding → [layer loop: RMSNorm → Attention → +Residual → RMSNorm → FFN → +Residual]`
  `→ Final RMSNorm → Logits → [Sampler]`), with collapsible layers, a `← Back` drill-down
  inspector, and a context-aware legend. Stages are derived from the exported graph (op + weight
  name), so it needs no extra instrumentation and works on the same structure/trace/live data.

Both views support playback animation and live inference over SSE:

![minfer inference graph visualization](docs/viz-demo.png)
![minfer inference pipeline visualization](docs/viz-demo2.png)

- **Live streaming**: `minfer viz <model.gguf>` (default port 8081; `--port N` to
  change) serves the page + live SSE from a single process.
- **Export a graph**: `minfer --dump-graph-json graph.json <model> "Hello"`; or
  pick a canned sample via the page's "Select a sample model" dropdown.

See **[viz/README.md](viz/README.md)** for the full user guide, the JSON format,
and all page features.

## Architecture

minfer is a pure-Rust LLM inference engine with no ML framework dependency.
The full design (module map, compute-graph pipeline, backend layering,
quantization layout, KV cache, adding a new architecture / backend) is
documented in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

At a glance: a GGUF v3 model is loaded and dispatched on `general.architecture`
to an implementation of the `ModelDef` trait; every forward call runs through a
**declarative compute graph** (`ComputeGraph`) — built once per `GraphParams`,
then reused params-only via `GraphCache` — while the scheduler assigns backends
per op (priority Metal → CUDA → CPU), applies pattern-based fusion, allocates
buffers with liveness analysis, and executes in per-backend splits:

```mermaid
flowchart LR
    subgraph GRAPH["build → assign → fuse → alloc → execute"]
        B1["GraphBuilder<br/>build_graph (pure IR)"] --> B2["assign backends<br/>Metal → CUDA → CPU"]
        B2 --> B3["fuse<br/>SwiGLU / BiasRope (gated)"]
        B3 --> B4["alloc<br/>liveness + persistent KV"]
        B4 --> B5["execute<br/>per split, cross-backend copies"]
    end

    A["CLI"] --> L["load GGUF<br/>metadata + quantized weights"]
    L --> C["tokenize prompt<br/>BPE + chat template"]
    C --> D["PREFILL<br/>graph forward, all prompt tokens"]
    D --> E["last-token logits"]
    E --> F{"DECODE loop"}
    F -->|sample| G["sample next token<br/>penalties → top-k → top-p → temp"]
    G -->|stop| H["text out"]
    G -->|continue| I["graph forward, 1 token<br/>KV persists in the allocator"]
    I --> F

    D -.->|"GraphCache: params-only reuse"| GRAPH
    I -.->|"GraphCache: params-only reuse"| GRAPH
```

Key invariants: **KV positions are data** (the graph topology never depends on
`n_past`, so decode steps reuse one graph); each layer owns **two persistent KV
regions** (K and V) resolved via `kv_pair(layer)`; in-place ops (`Silu`/`RoPE`)
alias their input buffer; matmuls follow the GGUF weight layout
(`od = shape[1]`, `id = shape[0]`).

Key modules: `src/graph/` (IR / builder / scheduler / backends / reuse cache),
`gguf.rs` (parser + mmap'd zero-copy loader), `models/qwen2/` (build_graph +
loader), `kernel.rs`/`quants.rs` (quantized matmul), `metal.rs`+`metal.metal` and
`cuda.rs`+`cuda_kernels.cu` (GPU kernels), `sampler.rs`/`tokenizer.rs`/
`template.rs` (sampling + tokenization + chat templates), `conversation.rs`
(multi-turn sessions), `server/` (HTTP). Supported quants:
Q4_0, Q4_1, Q8_0, Q4_K, Q6_K, Q5_0, Q5_1, Q5_K (CPU + Metal), F32/F16 norms &
biases.

## Performance

**CUDA — NVIDIA GB10 (DGX Spark, sm_121), default path (2026-09-08):**

| Model | Prefill (pp3314) | vs llama.cpp | Decode (tg128) | Decode @long KV | Device mem |
|-------|------------------|--------------|----------------|-----------------|------------|
| Qwen2.5-7B-Instruct Q4_K_M | **~3581 tok/s** | **1.080×** (llama-bench 3323.3, same shape) | **~51.2 tok/s** (**1.074×**) | **50.2 tok/s** @1.6K (**1.052×**) | ~10.4 GB |
| Qwen2.5-14B-Instruct Q4_K_M | **~1830 tok/s** | **1.12×** (llama-bench 1634, same window) | **24.6 tok/s** (**1.018×**) | 22.9 tok/s @3.3K (0.950×) | ~14.1 GB |

Decode numbers are same-window matched-anchor pairs against llama-bench
(ca3d5a3e1) on the shared GPU; window drift between sessions is ±2%.

The int8 tensor-core MMQ path is **default-on** in CUDA builds — ~3581 tok/s is
8.1× over the 441 tok/s where the path started, with every optimization step
(measurement, gates and commit) documented in the history table of
**[`docs/CUDA_OPTIMIZATION.md`](docs/CUDA_OPTIMIZATION.md)**.
Decode runs the dp4a MMVQ (q6_K on a dense split-plane layout, `MINFER_Q6K_DPL=0`
opt-out) + split-KV attention + fused-QKV kernels: 7B decode is **ahead of
llama.cpp at every measured context length**; 14B is ahead at short context with
the remaining ~5% gap at 3.3K KV in attention structure (the D/D4-series campaign
record is §2D).
`MINFER_MMQ=0` restores the legacy f16 path; `MINFER_MMQ_Q6K_EXP=0` /
`MINFER_MMQ_Q4K_DSC=0` trade ~6% prefill for ~3.3 GB of device memory.

**Metal — Apple M4 Pro (2026-08-21, compute-graph path):**

| Model | Prefill (pp499) | Decode (greedy) |
|-------|-----------------|-----------------|
| Qwen2.5-0.5B Q4_K_M | ~4460 tok/s | ~268 tok/s |
| Qwen2.5-0.5B Q4_0 | ~4775 tok/s | ~321 tok/s |
| Qwen2.5-1.5B Q4_K_M | ~1750 tok/s | ~153 tok/s |
| Qwen2.5-7B Q4_K_M | ~430 tok/s (pp31 ~250) | ~48 tok/s |

CPU (AVX2) reference: Qwen2-0.5B on i7-1260P ~27 tok/s prefill / ~21 tok/s
decode.

Metal prefill uses simdgroup GEMMs for every quant type (dispatched for
`nt ≥ 2 && (od ≥ 2048 || nt ≥ 9)`); decode uses fused QKV/FFN matmuls + a
KV-parallel split attention. See
**[`docs/METAL_OPTIMIZATIONS.md`](docs/METAL_OPTIMIZATIONS.md)**.

Decode optimizations on both GPU backends: CUDA Graph capture/replay (single
launch per decode step), full-layer GPU offload with zero-copy buffers,
on-GPU activation quantization (f32 → Q8_0), fused decode QKV/FFN chains,
flash attention with online softmax, and f16 KV cache for 7B-class models.

## Build

Requirements: **Rust** (edition 2021); a Nix flake devShell (`nix develop`) is
available. Typical builds:

```bash
cargo build --release                             # CPU + Metal (macOS)
cargo build --release --features cuda             # + CUDA backend (NVIDIA GPU)
cargo build --release --features cuda,cuda_static # CUDA with statically-linked cudart
cargo build --release --features debug_dump       # + per-node debug dumps
```

The full build reference — Metal `.metallib` precompilation, CUDA nvcc/host-
compiler auto-detection (`MINFER_CUDA_CCBIN`), GPU-arch coverage (sm_70…sm_121,
CUDA 12.8 vs 13 notes), and cudart linking details — lives in
**[`docs/BUILD.md`](docs/BUILD.md)**.

## Usage

Examples use the built binary (`cargo build --release` first — see
[Build](#build)); `cargo run --release -- …` works identically.

```bash
./target/release/minfer <model> [prompt] [OPTIONS]

# Cached model by name (no full path needed)
./target/release/minfer qwen2.5-0.5b-instruct-q4_0 "Hello"

# Auto-download from Hugging Face + run (quant auto-detected, splits included)
./target/release/minfer hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf "Hello"

# Interactive multi-turn conversation (REPL)
./target/release/minfer --cnv qwen2.5-0.5b-instruct-q4_0

# OpenAI-compatible HTTP server
./target/release/minfer serve qwen2.5-0.5b-instruct-q4_0

# llama-bench-style performance test (pp512 + tg128)
./target/release/minfer bench qwen2.5-0.5b-instruct-q4_0
```

The full CLI reference — model formats (local / `hf:` / `ollama:` / cached
names), all subcommands (`info`, `download`, `list`, `viz`, `bench`), every
sampling and conversation option — lives in
**[`docs/USAGE.md`](docs/USAGE.md)**.

## License

MIT
