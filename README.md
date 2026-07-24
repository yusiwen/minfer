# minfer

A minimal local LLM inference engine built from scratch in Rust.

## Features

- **GGUF loader** — parses GGUF v3 files (metadata + quantized tensors)
- **Self-contained BPE tokenizer** — loaded directly from GGUF metadata,
  no external dependency on tiktoken
- **CPU: AVX2-accelerated** — Q4₀×Q8₀ and Q8₀×Q8₀ dot products via AVX2+FMA
- **GPU: CUDA backend** — NVIDIA GPU acceleration with CUDA Graph capture/replay
  for decode, full-layer GPU offload (zero-copy), automatic best-GPU selection
- **GPU: Metal backend** — Apple Silicon acceleration with flash attention
  (online softmax), SIMD-parallel RMSNorm, float4 vectorized kernels
- **Qwen2 architecture** — GQA attention, SwiGLU FFN, RoPE (Neox style),
  RMSNorm
- **Model download** — auto-download from Hugging Face Hub or Ollama registry
- **No external ML framework** — pure Rust, only depends on `rand`, `regex`,
  `half`, `serde`, and `serde_json`

## Supported Quantization Formats

minfer supports GGUF v3 files with the following quantized weight types.
Activation quantization uses Q8_0 (on-the-fly) for Q4_0 weights; all other
weight types work with f32 activations.

### Supported

| Type | Bits | Block | CPU | AVX2 | CUDA GPU | Metal GPU |
|------|------|-------|:---:|:----:|:--------:|:---------:|
| **Q4_0** | 4 | 18 B / 32 val | ✅ | ✅ | ✅ | ✅ |
| **Q4_1** | 4 | 20 B / 32 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q4_K** | 4 | 144 B / 256 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q6_K** | 6 | 210 B / 256 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q8_0** | 8 | 34 B / 32 val | ✅ | ✅ | ✅ | ✅¹ |
| **F32** | 32 | 4 B / 1 val | ✅ | — | ✅² | ✅² |

¹ Metal standalone `quant_matmul_f32` only handles Q4_0; Q4_1/Q4_K/Q6_K/Q8_0
require the `layer_gpu` full-layer offload path.  
² F32 weights (RMSNorm, biases) are supported on GPU but not for matmul.

**GPU grouping restriction** (CUDA and Metal): within one transformer layer,
all 7 weight matrices (WQ, WK, WV, WO, FFN Gate, FFN Up, FFN Down) must be
either **all in the Q4 group** (Q4_0 / Q4_1) or **all in the QK group**
(Q4_K / Q6_K). Mixed groups within a layer are rejected and fall back to CPU.

### Not Yet Supported

| Category | Types |
|----------|-------|
| Legacy Q5 | Q5_0, Q5_1 |
| K-quants | Q2_K, Q3_K, Q5_K, Q8_K |
| I-quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| Other | Q1_0, BF16, TQ1_0, TQ2_0, MXFP4, NVFP4 |

The GGUF parser can read metadata and compute tensor shapes for these types,
and block layout structs are defined in `src/block.rs` for size calculations,
but **no matmul kernel exists** — inference would fail at runtime.

## Supported Model Architectures

minfer currently supports **one** model architecture.

| Architecture | Variants | Status | Detection Key |
|-------------|----------|:------:|---------------|
| **Qwen2** | Qwen2, Qwen2.5 | ✅ Fully supported | `general.architecture = "qwen2"` |

### How Architecture Detection Works

minfer reads the `general.architecture` string from the GGUF metadata header.
Only the exact value `"qwen2"` (case-sensitive) is accepted. Any other value
produces a clear error:

```
Unsupported architecture: 'llama'
```

The loader will **not** silently misinterpret a non-Qwen2 model — it fails
immediately with a descriptive message. All model-agnostic components (BPE
tokenizer, Jinja2 chat template renderer, samplers) are ready for additional
architectures once the forward-pass code is added.

### Hyperparameter Keys

The Qwen2 loader reads GGUF keys from both `qwen2.*` and `llama.*` prefixes.
The `llama.*` fallback exists for compatibility with older GGUF converters that
used the `llama.` prefix as a de-facto standard for Llama-family hyperparameters.
This does **not** mean Llama architecture is supported.

### Adding a New Architecture

See `AGENTS.md` for a step-by-step guide. In brief:
1. Create `src/models/<name>/` with `mod.rs`, `forward.rs`, `loader.rs`
2. Add a `match` branch in `src/models/mod.rs::load_model()`
3. Define `HParams`, `LayerWeights`, and implement the `ModelDef` trait

Architectures that share Qwen2's tensor naming convention (LLaMA, Mistral, Phi)
should be relatively straightforward to port.

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

**Qwen2 / Qwen2.5 — 0.5B class (Q4_0, ~400 MB):**

| Backend | Hardware | Model | Prefill | Decode |
|---------|----------|-------|---------|--------|
| CPU (AVX2) | i7-1260P | Qwen2-0.5B | ~27 tok/s | ~21 tok/s |
| CUDA + Graph | RTX 2080 Ti | Qwen2.5-0.5B | ~593 tok/s | ~486 tok/s |
| Metal GPU | Apple M4 Pro | Qwen2-0.5B | ~400 tok/s | ~330 tok/s |

GPU decode optimizations: CUDA Graph capture/replay (single `cudaGraphLaunch`
per decode step), full-layer GPU offload with zero-copy buffers, on-GPU
activation quantization (f32 → Q8_0). Metal: flash attention (online softmax),
SIMD-parallel RMSNorm with float4 vectorization, Q4_0 × Q8_0 matmul.

## Architecture

```
src/
├── main.rs          # Entry point, CLI, inference loop
├── gguf.rs          # GGUF format parser (v3) + KV helpers
├── block.rs         # Quantized block types + fp16 conversions
├── avx2.rs          # AVX2 dot product kernels + quantization
├── cuda.rs          # CUDA GPU state, FFI bindings, graph capture
├── cuda_kernels.cu  # CUDA kernels (matmul, attention, element-wise ops)
├── metal.rs         # Metal GPU state machine + kernel dispatch
├── metal.metal      # Metal compute shaders (attention, matmul, norm)
├── build.rs         # CUDA kernel compilation + arch detection
├── kernel.rs        # Quantized matmul dispatch (CPU/GPU bridge)
├── tensor.rs        # Tensor struct + data access
├── vec_ops.rs       # SIMD vector ops (RMSNorm, RoPE, softmax, SiLU)
├── cache.rs         # KV cache (shared, architecture-agnostic)
├── sampler.rs       # Greedy / temperature / top-k / top-p sampling
├── tokenizer.rs     # BPE tokenizer (self-contained, GGUF-backed)
├── template.rs      # Chat template detection + formatting
├── download/        # Model download from HF Hub & Ollama
│   └── mod.rs       # resolve() URI handler, curl-based HTTP, list_local()
├── examples/
│   └── cuda_graph_diag.rs  # CUDA graph capture diagnostic tool
└── models/          # Architecture-specific implementations
    ├── mod.rs       # ModelDef trait + load_model factory dispatch
    └── qwen2/       # Qwen2 implementation
        ├── mod.rs       # Qwen2Model + ModelDef impl
        ├── forward.rs   # Forward pass (CPU + GPU paths)
        └── loader.rs    # Tensor loading from GGUF
```

## License

MIT
