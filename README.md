# minfer

A minimal local LLM inference engine built from scratch in Rust.

## Features

- **GGUF loader** — parses GGUF v3 files (metadata + quantized tensors)
- **Self-contained BPE tokenizer** — loaded directly from GGUF metadata,
  no external dependency on tiktoken
- **CPU: AVX2-accelerated** — Q4₀×Q8₀ and Q8₀×Q8₀ dot products via AVX2+FMA
- **GPU: Metal backend** — Apple Silicon acceleration with flash attention
  (online softmax), SIMD-parallel RMSNorm, float4 vectorized kernels
- **Qwen2 architecture** — GQA attention, SwiGLU FFN, RoPE (Neox style),
  RMSNorm
- **Model download** — auto-download from Hugging Face Hub or Ollama registry
- **No external ML framework** — pure Rust, only depends on `rand`, `regex`,
  `half`, `serde`, and `serde_json`

## Usage

```bash
cargo run --release -- <model> [prompt]
```

`<model>` can be a local path or an auto-download URI:

| URI format | Example |
|------------|---------|
| Local file | `~/models/qwen2.gguf` |
| HF Hub | `hf:Qwen/Qwen2-0.5B-GGUF:qwen2-0.5b-q4_0.gguf` |
| Ollama | `ollama:qwen2.5:0.5b` |

If `prompt` is omitted, reads from stdin.

**Examples:**

```bash
# Local model
cargo run --release -- ~/models/qwen2-0.5b-q4_0.gguf "What is the capital of France?"

# Auto-download from Hugging Face + run
cargo run --release -- hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf "Hello"

# List available GGUF files in a HF repo (without downloading)
cargo run --release -- download hf Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Pull from Ollama and create a symlink
cargo run --release -- download ollama qwen2.5:0.5b

# List locally cached models
cargo run --release -- list
```

## Performance

**Qwen2-0.5B-Instruct (Q4_0, 336 MB):**

| Backend | Hardware | Prefill | Decode |
|---------|----------|---------|--------|
| CPU (AVX2) | i7-1260P | ~27 tok/s | ~21 tok/s |
| Metal GPU | Apple M4 Pro | ~400 tok/s | ~330 tok/s |

GPU decode optimizations: flash attention (online softmax + SIMD-parallel
dot products), SIMD-parallel RMSNorm with float4 vectorization, quantized
matmul (Q4_0 × Q8_0) on GPU.

## Architecture

```
src/
├── main.rs        # Entry point, CLI, inference loop
├── gguf.rs        # GGUF format parser (v3) + KV helpers
├── block.rs       # Quantized block types + fp16 conversions
├── avx2.rs        # AVX2 dot product kernels + quantization
├── metal.rs       # Metal GPU state machine + kernel dispatch
├── metal.metal    # Metal compute shaders (attention, matmul, norm)
├── kernel.rs      # Quantized matmul dispatch (CPU/GPU bridge)
├── tensor.rs      # Tensor struct + data access
├── vec_ops.rs     # SIMD vector ops (RMSNorm, RoPE, softmax, SiLU)
├── cache.rs       # KV cache (shared, architecture-agnostic)
├── sampler.rs     # Greedy / temperature / top-k / top-p sampling
├── tokenizer.rs   # BPE tokenizer (self-contained, GGUF-backed)
├── template.rs    # Chat template detection + formatting
├── download/      # Model download from HF Hub & Ollama
│   └── mod.rs     # resolve() URI handler, curl-based HTTP, list_local()
└── models/        # Architecture-specific implementations
    ├── mod.rs     # ModelDef trait + load_model factory dispatch
    └── qwen2/     # Qwen2 implementation
        ├── mod.rs     # Qwen2Model + ModelDef impl
        ├── forward.rs # Forward pass (CPU + GPU paths)
        └── loader.rs  # Tensor loading from GGUF
```

## License

MIT
