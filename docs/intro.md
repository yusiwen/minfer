# minfer

A minimal local LLM inference engine built from scratch in Rust, modeled on
llama.cpp's `ggml_cgraph` + backend scheduler, with **zero ML framework
dependencies**.

- **Declarative compute graph** — inference builds a `ComputeGraph` (pure IR),
  then assigns backends, fuses ops, allocates and executes via a scheduler.
  Params-only graph reuse (decode steps skip reconstruction).
- **Backends** — CPU + Apple MPS (Metal); optional CUDA (feature-gated).
- **Models** — Qwen2 / Qwen2.5 / Qwen3 (dense), GGUF v3 format.
- **Tooling** — OpenAI-compatible server, CLI multi-turn conversation, an
  interactive web visualizer, and per-layer debug dumps.

## Explore

- [Architecture](ARCHITECTURE.md) — module map, pipeline, backend layering, adding an arch.
- [Compute Graph](GRAPH-REFACTOR-PLAN.md) — the declarative graph design + rewrite record.
- Backends: [Metal](METAL_OPTIMIZATIONS.md) · [CPU](CPU_OPTIMIZATIONS.md) · [GPU Safety](GPU_SAFETY.md).
- [Qwen3 Support](QWEN3-SUPPORT-PLAN.md) and the [Architecture Roadmap](ARCHITECTURE-ROADMAP.md).
- Tooling: [Debug Dump](debug-dump.md), web visualizer (`viz/`).

> **Build & run**
> ```bash
> cargo build --release
> ./target/release/minfer <model.gguf> "hello"
> ```
