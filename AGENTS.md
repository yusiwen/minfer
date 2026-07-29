# minfer — AI Agent Context

## Project Overview

minfer is a pure Rust LLM inference engine written from scratch (~4400 LOC), inspired by llama.cpp with 0 ML framework dependencies.  
Supports **Qwen2/Qwen2.5 architecture**, **CPU + Apple MPS (Metal) GPU** inference, and **GGUF v3 format**.

## Architecture at a Glance

```
src/
├── main.rs          # CLI + inference loop (prefill → autoregressive generation)
├── gguf.rs          # GGUF v3 parser (~1650 lines, largest file)
├── block.rs         # 20+ quantized block types (repr(C), matching ggml-common.h)
├── avx2.rs          # AVX2 dot product kernels + f32→Q8_0 quantization
├── kernel.rs        # Quantized matmul dispatch + CPU scalar fallbacks
├── vec_ops.rs       # SIMD vector ops (RMSNorm, RoPE, Softmax, SiLU)
├── tensor.rs        # 4D Tensor (shape/strides/data)
├── cache.rs         # KV Cache
├── dump.rs          # Debug dump module (gated by `--features debug_dump`)
├── tokenizer.rs     # BPE tokenizer (self-contained, loaded from GGUF metadata)
├── sampler.rs       # Greedy / Top-K / Top-P / Temperature sampling
├── template.rs      # ChatML / Llama3 / Mistral template rendering (minijinja)
├── download/mod.rs  # HuggingFace + Ollama model auto-download
├── metal.rs         # Apple MPS (Metal) GPU backend
└── models/
    ├── mod.rs       # ModelDef trait + factory dispatch
    └── qwen2/
        ├── mod.rs   # Qwen2Model + ModelDef implementation
        ├── forward.rs  # Forward pass (quantized inference)
        └── loader.rs   # GGUF weight loading + GPU registration
```

## Build & Run

```bash
cargo build --release

# Debug dumps (per-layer hidden states)
cargo build --release --features debug_dump

# Run (Q4_0 — works)
./target/release/minfer ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf "hello"

# Run (Q4_K_M — CPU path bug, see below)
./target/release/minfer ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf "hello"

# Skip chat template (raw prompt)
./target/release/minfer --no-template <model> "hello"

# Debug dump
MINFER_DUMP_DIR=/tmp/dump target/release/minfer --features debug_dump <model> "hello"

# Force CPU (disable MPS)
MINFER_DISABLE_MPS=1 target/release/minfer <model> "hello"

# Info (list tensor names/types/shapes)
./target/release/minfer info <model>

# Auto-download
./target/release/minfer download hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_0.gguf
```

## Debug Dump Files (`MINFER_DUMP_DIR`)

| File | Shape | Description |
|------|-------|-------------|
| `minfer_dump_embed_out.f32` | (nt, ne) | Token embedding output |
| `minfer_dump_layer0_bn.f32` | (nt, ne) | RMSNorm (attn_norm) output (layer 0 only) |
| `minfer_dump_layer0_bq.f32` | (nt, nqt) | Q projection output, pre-RoPE (layer 0 only) |
| `minfer_dump_layer0_bq_rope.f32` | (nt, nqt) | Q projection after RoPE (layer 0 only) |
| `minfer_dump_layer0_ba.f32` | (nt, ne) | Attention output before O_proj (layer 0 only) |
| `minfer_dump_layer0_attn_out.f32` | (nt, ne) | Hidden state after attention + residual |
| `minfer_dump_layer0_bg.f32` | (nt, nf) | FFN gate output before SiLU (layer 0 only) |
| `minfer_dump_layer0_swiglu.f32` | (nt, nf) | SwiGLU output (layer 0 only) |
| `minfer_dump_layer0_fd.f32` | (nt, ne) | FFN down projection output (layer 0 only) |
| `minfer_dump_layer{N}_out.f32` | (nt, ne) | Hidden state after layer N |
| `minfer_dump_last_norm.f32` | (nt, ne) | Final RMSNorm (output_norm) output |
| `minfer_dump_logits.f32` | (nt, nv) | Final logits |
| `minfer_dump_prompt.txt` | — | Rendered prompt text |
| `minfer_dump_q8_quant_verify.txt` | — | Q8_0 quantization verification |

Gen0 suffix (`_gen0.f32`) = first autoregressive generation step (single token).

## Quantization Support

### Working (verified)
- **Q4_0** — standard 4-bit, all weights, both architectures → **works correctly**
- **Q4_1** — 4-bit with min
- **Q8_0** — 8-bit
- **Q4_K** — 4-bit K-quant (super-block)
- **Q6_K** — 6-bit K-quant

### Partially working
- **Q4_K** — 4-bit K-quant super-block (CPU path: individual ops verified, but mixed Q4_K/Q6_K layers produce garbled output)
- **Q5_0** — 5-bit, individual ops verified correct, Q5_0×Q8_0 dot product implemented
- **Q5_K** — 5-bit K-quant (untested)

### Not supported (CPU)
- Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL, etc.
- **Q5_0 not supported in Metal GPU shaders** — always falls back to CPU

## Known Issues

### Q4_K_M CPU Path Bug (fixed)
Q4_K_M models use **alternating ffn_down types**: 12 layers with `Q4_K` (type 12) and 12 with `Q6_K` (type 14) for FFN down projection.

- **Root cause**: Q4_K/Q5_K nibble layout in `qs[128]` was misinterpreted. minfer assumed Q4_0-style layout (each byte's lo/hi nibble = same subblock's consecutive elements), but llama.cpp stores them cross-subblock (each byte's lo/hi nibble = element j of subblocks 2k and 2k+1). Fixed in `dot_q4_k_q8_0_scalar`, `cpu_q5_k_matmul_f32`, Q4_K embed, Q5_K embed.
- **Fix applied**: avx2.rs, kernel.rs, forward.rs — deinterleave nibbles properly before computing dot product.
- After fix: per-layer cos ≥ 0.9993 through all 24 layers, output matches llama CPU.

### Q5_0 not supported in Metal GPU shaders
Q5_0 weights are registered with MPS (for partial acceleration where possible), but the shader kernel doesn't handle Q5_0. When GPU path fails at `layer_gpu`, the engine gracefully falls back to CPU.

### KV Head Dimension (n_kv_embd)
Qwen2.5 models use **separate KV head dimensions** (e.g., Qwen2.5-0.5B: n_embd=896, n_head=14 → hd=64, n_kv_embd=128). The KV cache and attention functions now use `n_kv_embd` from `HParams` (read from K weight's ne[1]) instead of computing `n_head_kv * n_embd_head()`.

## Core Conventions

1. **Activations quantized to Q8_0 on-the-fly** — all CPU matmuls use `Q8_0` quantized activations. Q5_0 weights are handled via `dot_q5_0_q8_0()` (newly added).
2. **AVX2 dispatch pattern**: all kernels use `is_x86_feature_detected!("avx2")` runtime detection + scalar fallback (ARM Mac always uses scalar).
3. **No ML frameworks** — Attention, RMSNorm, RoPE, SiLU, Softmax all handwritten loops.
4. **Tensor data uses raw `&[u8]` interface** — avx2.rs dot products operate on byte slices, not structs.
5. **GGUF padding rule**: `ggml_pad()`: `(x + n - 1) & !(n - 1)`.

## Adding a New Architecture

1. Create `models/<name>/` with `mod.rs`, `forward.rs`, `loader.rs`
2. Add dispatch branch in `models/mod.rs::load_model()`
3. Define `HParams` (including `n_kv_embd`) and `LayerWeights` in `loader.rs`
4. Implement forward pass in `forward.rs`
5. Implement `ModelDef` trait in `mod.rs`
6. Add template format support in `template.rs` if needed

## Dependencies

Only 5 external crates: `rand` (sampling), `regex` (BPE pre-tokenization), `half` (fp16), `serde+serde_json` (download API), `minijinja` (template rendering).
