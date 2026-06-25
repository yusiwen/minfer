# minfer — AI Agent Context

## Project Overview

minfer is a pure Rust LLM inference engine written from scratch (~4400 LOC), inspired by llama.cpp with 0 ML framework dependencies.  
Currently supports **Qwen2 architecture**, **CPU AVX2 (x86-64)** inference, and **GGUF v3 format**.

## Architecture at a Glance

```
src/
├── main.rs          # CLI + inference loop (prefill → autoregressive generation)
├── gguf.rs          # GGUF v3 parser (~1650 lines, largest file)
├── block.rs         # 20+ quantized block types (repr(C), matching ggml-common.h)
├── avx2.rs          # AVX2 dot product kernels + f32→Q8_0 quantization
├── vec_ops.rs       # SIMD vector ops (RMSNorm, RoPE, Softmax, SiLU)
├── tensor.rs        # 4D Tensor (shape/strides/data)
├── cache.rs         # KV Cache
├── tokenizer.rs     # BPE tokenizer (self-contained, loaded from GGUF metadata)
├── sampler.rs       # Greedy / Top-K / Top-P / Temperature sampling
├── template.rs      # ChatML / Llama3 / Mistral template rendering
├── download/mod.rs  # HuggingFace + Ollama model auto-download
└── models/
    ├── mod.rs       # ModelDef trait + factory dispatch
    └── qwen2/
        ├── mod.rs   # Qwen2Model + ModelDef implementation
        ├── forward.rs  # Forward pass (4-bit quantized inference)
        └── loader.rs   # GGUF weight loading
```

## Build & Run

```bash
cargo build --release

# Base model
./target/release/minfer ~/.cache/minfer/models/hf/Qwen2-0.5B-GGUF/qwen2-0.5b-q4_0.gguf "hello"

# Instruct model
./target/release/minfer ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf "hello"

# Auto-download
./target/release/minfer download hf:Qwen/Qwen2-0.5B-Instruct-GGUF:q4_0.gguf
```

No test framework, no lint/typecheck scripts (pure Rust, `cargo build` is the check).

## Core Conventions

1. **All weights stored as Q4_0, activations quantized to Q8_0 on-the-fly** — MatMul uses `Q4_0 × Q8_0` dot product
2. **AVX2 dispatch pattern**: all kernels use `is_x86_feature_detected!("avx2")` runtime detection + scalar fallback
3. **No ML frameworks** — Attention, RMSNorm, RoPE, SiLU, Softmax all handwritten loops
4. **Tensor data uses raw `&[u8]` interface** — avx2.rs dot products operate on byte slices, not structs
5. **GGUF padding rule**: `ggml_pad()`: `(x + n - 1) & !(n - 1)`

## Adding a New Architecture

1. Create `models/<name>/` with `mod.rs`, `forward.rs`, `loader.rs`
2. Add dispatch branch in `models/mod.rs::load_model()`
3. Define `HParams` and `LayerWeights` in `loader.rs`, read hyperparameters from GGUF KV
4. Implement forward pass in `forward.rs`
5. Implement `ModelDef` trait in `mod.rs`
6. Add template format support in `template.rs` if needed

## Dependencies

Only 5 external crates: `rand` (sampling), `regex` (BPE pre-tokenization), `half` (fp16), `serde+serde_json` (download API)
