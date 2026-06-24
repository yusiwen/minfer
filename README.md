# minfer

A minimal local LLM inference engine built from scratch in Rust.

## Features

- **GGUF loader** — parses GGUF v3 files (metadata + quantized tensors)
- **Self-contained BPE tokenizer** — loaded directly from GGUF metadata,
  no external dependency on tiktoken
- **AVX2-accelerated** — Q4₀×Q8₀ and Q8₀×Q8₀ dot products via AVX2+FMA
- **Qwen2 architecture** — GQA attention, SwiGLU FFN, RoPE (Neox style),
  RMSNorm
- **No external ML framework** — pure Rust, only depends on `rand`, `regex`,
  and `half`

## Usage

```bash
cargo run --release -- <model.gguf> [prompt]
```

If `prompt` is omitted, reads from stdin.

**Examples:**

```bash
# Base model (Qwen2)
cargo run --release -- ~/models/qwen2-0.5b-q4_0.gguf "What is the capital of France?"

# Instruct model (Qwen2.5) — auto-applies ChatML template from GGUF metadata
cargo run --release -- ~/models/qwen2.5-0.5b-instruct-q4_0.gguf "Hello"
```

## Performance

| Model | Q4₀ size | Prefill | Decode |
|-------|----------|---------|--------|
| Qwen2-0.5B | 336 MB | ~27 tok/s | ~21 tok/s |

Measured on a NUC12 (i7-1260P) with AVX2+FMA.

## Architecture

```
src/
├── main.rs        # Entry point, CLI, inference loop
├── gguf.rs        # GGUF format parser (v3)
├── block.rs       # Quantized block types + fp16 conversions
├── avx2.rs        # AVX2 dot product kernels + quantization
├── tensor.rs      # Tensor struct + data access
├── vec_ops.rs     # SIMD vector ops (RMSNorm, RoPE, softmax, SiLU)
├── model.rs       # Model/HParams structs + GGUF metadata extraction
├── forward.rs     # Transformer forward pass (24 layers, KV cache)
├── loader.rs      # Tensor loading from GGUF into Model
├── sampler.rs     # Greedy / temperature / top-k / top-p sampling
├── tokenizer.rs   # BPE tokenizer (self-contained, GGUF-backed)
└── template.rs    # Chat template detection + formatting
```

## License

MIT
