# Features

This page expands the feature list from the [README](https://github.com/yusiwen/minfer) with the full detail. Performance numbers refer to Qwen2.5-7B-Instruct q4_K_m prefill on GB10 (sm_121) unless noted.

## Inference Core

### Declarative compute graph

Inference builds a `ComputeGraph` (pure IR) then assigns backends, fuses ops, allocates and executes via a scheduler — inspired by llama.cpp's `ggml_cgraph` + backend scheduler. Graph reuse is params-only (decode steps skip reconstruction), backend assignment is per-op, and the whole design is documented in [GRAPH-REFACTOR-PLAN.md](./GRAPH-REFACTOR-PLAN.md).

### Interactive graph visualization (`viz/`)

A zero-dependency browser page for the compute graph. `minfer viz <model>` serves the page, live SSE inference, per-node tensor stats/heatmaps and logits top-5 in one process; `--dump-graph-json` / `MINFER_TRACE` export graphs and real traces. See the [viz README](https://github.com/yusiwen/minfer/tree/master/viz) for the full user guide.

### GGUF loader

Parses GGUF v3 files (metadata + quantized tensors) with split multi-part support; weights are **mmap'd and shared zero-copy with the GPU**.

### Self-contained BPE tokenizer

Loaded directly from GGUF metadata — no external dependency on tiktoken. Special tokens (GGUF type 3/4 table plus `<|im_start|>`/EOS fallbacks) match as single IDs before BPE, so special-token templates (DeepSeek-R1's `<｜User｜>`/`<think>`, etc.) tokenize exactly like llama.cpp.

## Backends

### CPU — AVX2 / NEON+SDOT

All 8 quantized dot products as SIMD kernels (AVX2 on x86, NEON+SDOT via inline asm on Apple Silicon), plus a persistent row-parallel thread pool (`-t/--threads`). Qwen3-4B CPU decode runs ~52–58 tok/s on M4 Pro (vs 1.1 before the pool).

### GPU — Metal (Apple Silicon)

Flash attention (single fused kernel for decode + prefill), simdgroup GEMM prefill for every quant type, SIMD-parallel RMSNorm, float4-vectorized kernels, a build-time precompiled `.metallib` (no per-run shader compile), and auto-selected f16 KV cache for 7B-class models. Tracked in [METAL_OPTIMIZATIONS.md](./METAL_OPTIMIZATIONS.md).

### GPU — CUDA (NVIDIA, feature-gated `--features cuda`)

The performance headline of the project. The int8 tensor-core MMQ path is **default-on** in CUDA builds (opt-out per gate with `"0"`; `MINFER_MMQ=0` restores the legacy f16 path):

- **Default prefill: ~3581 tok/s** (7B q4_K_m @3314-token prompt) = **1.080× llama.cpp** (llama-bench 3323.3 same shape) — from 441 tok/s when the path first landed, an 8.1× campaign documented step-by-step in [CUDA_OPTIMIZATION.md](./CUDA_OPTIMIZATION.md) (75-step history table).
- Raw-nibble int8 `mma.m16n8k32` GEMMs for q4_K and q6_K with producer-fused activation quantization (rms-norm/swiglu emit the transposed q8 plane directly, skipping intermediate writes), registration-time weight-expansion planes (W_exp / W_dsc) staged by `cp.async`, and flash attention with register-resident softmax (2.43× kernel).
- CUDA Graph capture/replay for repeated identical-length prefills; decode uses the MMVQ weight-streaming path.
- Memory/speed knobs: the weight-expansion planes cost ~3.3 GB device for ~+6% prefill; `MINFER_MMQ_Q6K_EXP=0` / `MINFER_MMQ_Q4K_DSC=0` return the memory.

## Model Support

### Qwen2 / Qwen3 architectures

GQA attention, SwiGLU FFN, RoPE (Neox style), RMSNorm. Qwen3 adds the decoupled head dim and per-head Q/K RMSNorm (`attn_q_norm`/`attn_k_norm`, `Op::QkNorm`). Supported models include Qwen2.5 0.5B/7B, Qwen3 0.6B/4B, and DeepSeek-R1-Distill-Qwen-1.5B — see the matrix in [AGENTS.md](https://github.com/yusiwen/minfer/blob/master/AGENTS.md).

## User-Facing

### Model download

Auto-download from the Hugging Face Hub or the Ollama registry, with resume and cache-name resolution.

### Multi-turn conversation CLI (`--cnv`)

Append-only KV + incremental chat-template rendering: each turn only prefills the new message delta while the whole conversation accumulates in the KV cache. In-session commands (`/clear`, `/regen`, …), automatic overflow truncation, `--session` persistence. Plan: [CLI-CONVERSATION-PLAN.md](./CLI-CONVERSATION-PLAN.md).

### OpenAI-compatible HTTP server (`serve`)

`/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/health`; multi-slot with queued serial execution. Plan: [OPENAI-CHAT-API-PLAN.md](./OPENAI-CHAT-API-PLAN.md).

### Performance benchmark (`bench`)

`minfer bench <model>` runs llama-bench-style prefill (`pp<P>`) / decode (`tg<T>`) throughput tests on the active backend — mean ± stddev over reps after an untimed warmup, each rep from an empty KV context without a model reload — reported as a markdown/CSV/JSON table.

## Philosophy

**No external ML framework** — pure Rust; runtime deps are minimal (`rand`, `regex`, `half`, `serde`, `serde_json`, `minijinja`; `axum`/`tokio` only for the HTTP server). Attention, RMSNorm, RoPE, SiLU, Softmax and every quantized dot product are handwritten.
