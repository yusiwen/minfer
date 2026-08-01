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
├── download/mod.rs  # HuggingFace + Ollama auto-download + cached-name resolution
├── metal.rs         # Apple MPS (Metal) GPU backend + dispatch
├── metal.metal      # Metal GPU shaders (Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0 kernels)
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

# Run Q4_K_M (GPU on Apple Silicon when available)
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
- **Q5_K** — 5-bit K-quant, nibble + qh layout fixed, unsigned formula verified (Q5_K_M works CPU + Metal GPU)
- **Q5_1** — 5-bit with min (Q5_K_M model), CPU + Metal GPU verified (j+16 qh indexing matches llama-ARM)

### Not supported (CLI)
- Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL, etc.

## GPU Support Matrix

| Quant | MPS (Metal) | CPU |
|-------|-------------|-----|
| Q4_0, Q4_1 | ✓ (Q8_0-activation path + f32 path) | ✓ |
| Q4_K, Q6_K | ✓ (f32 path) | ✓ |
| Q8_0 | ✓ (f32 path) | ✓ |
| **Q5_0** | ✓ (f32 path, qh unaligned-read bug fixed) | ✓ |
| **Q5_1** | ✓ (F32 path) | ✓ |
| **Q5_K** | ✓ (f32 path, `kernel_q5_k_f32_matmul` + `_multi`) | ✓ |
| F32 | ✓ (RMSNorm, biases, etc.) | ✓ |

## Model Support Matrix

| Model | CPU | MPS GPU | Notes |
|-------|-----|---------|-------|
| Q4_0 (qwen2.5-0.5b-instruct-q4_0) | ✓ | ✓ (361 tok/s) | All weights Q4_0 |
| Q4_K_M (qwen2.5-0.5b-instruct-q4_k_m) | ✓ (3.2s) | ✓ (226 tok/s) | Q5_0/Q8_0/Q4_K/Q6_K mixed |
| Q5_K_M (qwen2.5-0.5b-instruct-q5_k_m) | ✓ | ✓ (~250 tok/s) | Q5_1/Q8_0/Q5_K/Q6_K, formula + qh indexing FIXED, Q5_K Metal kernel added — full GPU |

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

### Q5_K formula + qh indexing bugs — FIXED (2026-07-31)
**Q5_K_M garbled output had TWO root causes in minfer's Q5_K:**

**Bug 1 (formula)**: minfer used the Q5_0-style **signed** formula `dl*(u-16)-ml`. llama.cpp's Q5_K (both CPU `dequantize_row_q5_K` and Metal `dequantize_q5_K`, unchanged since Oct 2023) uses **unsigned** `dl*u - ml`.

**Bug 2 (qh high-bit indexing)**: minfer read the 5th bit as `qh[sub*4 + pos/8] >> (pos%8)`. The correct layout (from `quantize_row_q5_K_impl`, llama-quants.c:1829-1844) is **`qh[pos]` bit `sub`** — qh byte index = element position within the 32-elem subblock, bit index = subblock number (0-7).

**Evidence** (real `blk.2.ffn_down.weight` dequant):
- unsigned formula: mean≈0, symmetric ✓ vs signed-16: mean=-0.008 ✗
- correct qh indexing on llama's own swiglu: ffn_down cos=**0.99999902** vs llama; minfer's wrong indexing: 0.994

**Fixed in 3 files × 2 bugs**:
- `avx2.rs:dot_q5_k_q8_0_scalar` — formula + `(qh[k] >> s) & 1`
- `forward.rs:embed_tokens` Q5_K branch — formula + `(qh[j] >> sub) & 1`
- `kernel.rs:cpu_q5_k_matmul_f32` — formula + `(qh[j] >> sub) & 1`, `(qh[j+16] >> sub) & 1`

**MPS GPU kernel** (added for full GPU support): `metal.metal` `kernel_q5_k_f32_matmul` + `_multi` (176 B/256-elem superblock, `get_scale_min_k4` scales, `qh[p] bit s` high bits, unsigned formula). Dispatch in `metal.rs` + MPS registration in `loader.rs`. Must be placed AFTER `get_scale_min_k4` in metal.metal (Metal requires declaration before use).

**Why earlier verification missed it**: `scripts/verify_q5k_gate_up.py` reimplemented minfer's OWN (wrong) formula AND qh indexing, so "cos=1.0" only proved Python↔Rust self-consistency, not correctness vs llama.

**Result**: Q5_K_M per-layer now matches llama (blk.2 cos=0.99999894, layers 4-21 = 0.99999971, no collapse at 22-23). CPU and MPS both produce correct output.

### Q5_K_M — VERIFIED WORKING (2026-07-31)
- CPU: "Hello! How can I assist you today?" ✓
- **MPS GPU: full GPU support** ✓ — Q5_K Metal kernel (`kernel_q5_k_f32_matmul` + `_multi`) added, ~250 tok/s (prefill 282, gen 247). Per-layer GPU vs CPU cos ≥ 0.9999998 (layers 2-20), logits argmax matches llama.
- Q4_0 / Q4_K_M: no regression ✓
- llama.cpp dispatch for reference: Q5_1→Q8_1 activations (ARM `s` = fp16(d_q8·Σq8)), Q4_K/Q5_K/Q6_K→Q8_K, Q5_0/Q8_0→Q8_0. **These are the CPU backend** `vec_dot_type`s. The **Metal backend reads f32 activations directly for ALL types including Q4_0** (no Q8_0 quantization). Since P1 (2026-08-01) minfer's Metal Q4_0 also uses f32 activations, matching llama Metal. Q8_0 activations in minfer's CPU path match llama's CPU to ~1e-6.

### KV Head Dimension (n_kv_embd)
Qwen2.5 models may use separate KV head dimensions (e.g., Qwen2.5-0.5B: n_embd=896, n_head=14 → hd=64, n_kv_embd=128). The KV cache and attention now use `n_kv_embd` from `HParams` (read from K weight's ne[1]) instead of computing `n_head_kv * n_embd_head()`. The attention function `gqa_attn` accepts separate `hd_kv` and `nkt` parameters for correct stride calculation.

## Core Conventions

1. **Activations quantized to Q8_0 on-the-fly** — all CPU matmuls use `Q8_0` quantized activations. Q5_0 weights use `dot_q5_0_q8_0()`; Q4_K uses `dot_q4_k_q8_0()`. The **Metal backend reads f32 activations directly for all weight types** (Q4_0 included since P1), matching llama.cpp's Metal backend.
2. **AVX2 dispatch pattern**: all kernels use `is_x86_feature_detected!("avx2")` runtime detection + scalar fallback (ARM Mac always uses scalar).
3. **No ML frameworks** — Attention, RMSNorm, RoPE, SiLU, Softmax all handwritten loops.
4. **Tensor data uses raw `&[u8]` interface** — avx2.rs dot products operate on byte slices, not structs.
5. **GGUF padding rule**: `ggml_pad()`: `(x + n - 1) & !(n - 1)`.
6. **Metal GPU per-layer fallback**: when `layer_gpu` fails (e.g., unsupported weight type like `Raw`), the engine submits the partial GPU work, downloads the hidden state, and continues remaining layers on CPU. Q5_K is now GPU-supported.

## Adding a New Architecture

1. Create `models/<name>/` with `mod.rs`, `forward.rs`, `loader.rs`
2. Add dispatch branch in `models/mod.rs::load_model()`
3. Define `HParams` (including `n_kv_embd`) and `LayerWeights` in `loader.rs`
4. Implement forward pass in `forward.rs`
5. Implement `ModelDef` trait in `mod.rs`
6. Add template format support in `template.rs` if needed

## Dependencies

Only 5 external crates: `rand` (sampling), `regex` (BPE pre-tokenization), `half` (fp16), `serde+serde_json` (download API), `minijinja` (template rendering).
