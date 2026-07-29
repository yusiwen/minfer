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
├── metal.rs         # Apple MPS (Metal) GPU backend + dispatch
├── metal.metal      # Metal GPU shaders (Q4_0/Q4_1/Q4_K/Q5_0/Q6_K/Q8_0 kernels)
└── models/
    ├── mod.rs       # ModelDef trait + factory dispatch
    └── qwen2/
        ├── mod.rs   # Qwen2Model + ModelDef implementation
        ├── forward.rs  # Forward pass (quantized inference, CPU + GPU fallback)
        └── loader.rs   # GGUF weight loading + MPS/CUDA GPU registration
```

## Build & Run

```bash
cargo build --release

# Debug dumps (per-layer hidden states)
cargo build --release --features debug_dump

# Run Q4_0 (GPU on Apple Silicon, CPU on other platforms)
./target/release/minfer ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf "hello"

# Run Q4_K_M (CPU — Q5_0 not optimized for Metal yet)
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
| `minfer_dump_layer{N}_out.f32` | (nt, ne) | Hidden state after layer N |
| `minfer_dump_layer{N}_attn_out.f32` | (nt, ne) | Hidden after attention + residual |
| `minfer_dump_layer0_bn.f32` | (nt, ne) | RMSNorm (attn_norm) output (layer 0 only) |
| `minfer_dump_layer0_bq.f32` | (nt, nqt) | Q projection output, pre-RoPE (layer 0 only) |
| `minfer_dump_layer0_bq_rope.f32` | (nt, nqt) | Q projection after RoPE (layer 0 only) |
| `minfer_dump_layer0_ba.f32` | (nt, ne) | Attention output before O_proj (layer 0 only) |
| `minfer_dump_layer0_bg.f32` | (nt, nf) | FFN gate output before SiLU (layer 0 only) |
| `minfer_dump_layer0_swiglu.f32` | (nt, nf) | SwiGLU output (layer 0 only) |
| `minfer_dump_layer0_fd.f32` | (nt, ne) | FFN down projection output (layer 0 only) |
| `minfer_dump_last_norm.f32` | (nt, ne) | Final RMSNorm (output_norm) output |
| `minfer_dump_logits.f32` | (nt, nv) | Final logits |
| `minfer_dump_prompt.txt` | — | Rendered prompt text |

Gen0 suffix (`_gen0.f32`) = first autoregressive generation step (single token).

## Quantization Support

### Working (verified)
- **Q4_0** — standard 4-bit, all weights, CPU + GPU → **works correctly**
- **Q4_1** — 4-bit with min
- **Q8_0** — 8-bit
- **Q4_K** — 4-bit K-quant super-block (nibble layout fixed)
- **Q6_K** — 6-bit K-quant
- **Q5_0** — 5-bit, CPU path works correctly, Q5_0×Q8_0 dot product implemented
- **Q5_K** — 5-bit K-quant, nibble layout fixed (CLI path verified, Q5_K_M model WIP)
- **Q5_1** — 5-bit with min (Q5_K_M model), CPU path implemented, formula verified

### Not supported (CLI)
- Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL, etc.

## GPU Support Matrix

| Quant | MPS (Metal) | CPU |
|-------|-------------|-----|
| Q4_0, Q4_1 | ✓ (Q8_0-activation path + f32 path) | ✓ |
| Q4_K, Q6_K | ✓ (f32 path) | ✓ |
| Q8_0 | ✓ (f32 path) | ✓ |
| **Q5_0** | ✓ (f32 path, qh unaligned-read bug fixed) | ✓ |
| **Q5_1** | ✗ | ✓ (formula verified, Q5_K_M model WIP) |
| Q5_K | ✗ | ✓ |
| F32 | ✓ (RMSNorm, biases, etc.) | ✓ |

## Model Support Matrix

| Model | CPU | MPS GPU | Notes |
|-------|-----|---------|-------|
| Q4_0 (qwen2.5-0.5b-instruct-q4_0) | ✓ | ✓ (361 tok/s) | All weights Q4_0 |
| Q4_K_M (qwen2.5-0.5b-instruct-q4_k_m) | ✓ (3.2s) | ✓ (226 tok/s) | Q5_0/Q8_0/Q4_K/Q6_K mixed |
| Q5_K_M (qwen2.5-0.5b-instruct-q5_k_m) | ✗ (乱码) | ✗ | Q5_1/Q8_0/Q5_K/Q6_K mixed, WIP |

## Known Issues

### Q4_K / Q5_K Nibble Layout (fixed)
Q4_K and Q5_K store `qs[128]` with cross-subblock nibble packing. Each byte's lo nibble belongs to subblock 2k and hi nibble to subblock 2k+1, aligned by element index. minfer originally assumed Q4_0-style layout (lo=sub_elem[j], hi=sub_elem[j+16]).

**Files affected and fixed:**
- `avx2.rs`: `dot_q4_k_q8_0_scalar`, `dot_q5_k_q8_0_scalar` — deinterleave before dot product
- `kernel.rs`: `cpu_q5_k_matmul_f32` — deinterleave before dequant
- `forward.rs`: Q4_K embed, Q5_K embed — deinterleave before dequant
- `metal.metal`: `kernel_q4_k_f32_matmul`, `kernel_q4_k_f32_matmul_multi` — deinterleave before dot product

### Q5_0 Metal GPU Kernel (fixed)
The Q5_0 Metal shader kernel is implemented in `metal.metal` (`block_q5_0_dot_y`, `kernel_q5_0_f32_matmul`, `kernel_q5_0_f32_matmul_multi`, `kernel_q5_0_debug`).

- **Root cause found & fixed**: `*(uint32_t *)(block + 2)` reads qh at a 2-byte aligned address — undefined behavior on ARM/Metal. Fixed by reading 4 individual bytes and combining. Verified via `kernel_q5_0_debug` (single-block dequant matches CPU bit-for-bit).
- **Verified working**: CPU vs GPU per-layer comparison shows all 24 layers with cos ≥ 0.998. Output matches CPU ("Hello! How can I assist you today?").

### Q5_1 (Q5_K_M model) — partially implemented
Q5_1 format: d(f16,2) + m(f16,2) + qh(u32,4) + qs(u8,16) = 24 bytes per 32 elements. Dequant: `val = d × unsigned_5bit + m` (no -16 offset — differs from Q5_0).

- **CPU path**: implemented (`TensorType::Q5_1`, `dot_q5_1_q8_0`, `embed_tokens` branch, `cpu_quant_matmul` dispatch). Unit test passes.
- **Status**: model loads and runs, produces reasonable norms (embed=0.31, l0_out=10.0), but output garbled. Root cause TBD.
- **MPS GPU**: not yet registered/implemented.

### KV Head Dimension (n_kv_embd)
Qwen2.5 models may use separate KV head dimensions (e.g., Qwen2.5-0.5B: n_embd=896, n_head=14 → hd=64, n_kv_embd=128). The KV cache and attention now use `n_kv_embd` from `HParams` (read from K weight's ne[1]) instead of computing `n_head_kv * n_embd_head()`. The attention function `gqa_attn` accepts separate `hd_kv` and `nkt` parameters for correct stride calculation.

## Core Conventions

1. **Activations quantized to Q8_0 on-the-fly** — all CPU matmuls use `Q8_0` quantized activations. Q5_0 weights use `dot_q5_0_q8_0()`; Q4_K uses `dot_q4_k_q8_0()`.
2. **AVX2 dispatch pattern**: all kernels use `is_x86_feature_detected!("avx2")` runtime detection + scalar fallback (ARM Mac always uses scalar).
3. **No ML frameworks** — Attention, RMSNorm, RoPE, SiLU, Softmax all handwritten loops.
4. **Tensor data uses raw `&[u8]` interface** — avx2.rs dot products operate on byte slices, not structs.
5. **GGUF padding rule**: `ggml_pad()`: `(x + n - 1) & !(n - 1)`.
6. **Metal GPU fallback**: when `layer_gpu` fails (unsupported weight type, shader error), the engine silently falls back to CPU via `run_cpu = true`.

## Adding a New Architecture

1. Create `models/<name>/` with `mod.rs`, `forward.rs`, `loader.rs`
2. Add dispatch branch in `models/mod.rs::load_model()`
3. Define `HParams` (including `n_kv_embd`) and `LayerWeights` in `loader.rs`
4. Implement forward pass in `forward.rs`
5. Implement `ModelDef` trait in `mod.rs`
6. Add template format support in `template.rs` if needed

## Dependencies

Only 5 external crates: `rand` (sampling), `regex` (BPE pre-tokenization), `half` (fp16), `serde+serde_json` (download API), `minijinja` (template rendering).
