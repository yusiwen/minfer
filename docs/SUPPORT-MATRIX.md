# Support Matrix

Supported quantization formats and model architectures. This page is the expanded version of the two support sections formerly in the README.

## Supported Quantization Formats

minfer supports GGUF v3 files with the following quantized weight types. The CPU backend quantizes activations on-the-fly (Q8_0 for the simple weight types, **Q8_K** — llama.cpp's format with precomputed per-subblock sums — for Q4_K/Q5_K/Q6_K); the GPU backends read f32 activations directly for their non-MMQ kernels, matching llama.cpp's Metal backend.

### Supported

| Type | Bits | Block | CPU | AVX2 | CUDA GPU | Metal GPU |
|------|------|-------|:---:|:----:|:--------:|:---------:|
| **Q4_0** | 4 | 18 B / 32 val | ✅ | ✅ | ✅ | ✅ |
| **Q4_1** | 4 | 20 B / 32 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q4_K** | 4 | 144 B / 256 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q5_0** | 5 | 22 B / 32 val | ✅ | ✅ | ✅ | ✅¹ |
| **Q5_1** | 5 | 24 B / 32 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q5_K** | 5 | 176 B / 256 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q6_K** | 6 | 210 B / 256 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q8_0** | 8 | 34 B / 32 val | ✅ | ✅ | ✅ | ✅¹ |
| **F32** | 32 | 4 B / 1 val | ✅ | — | ✅² | ✅² |

¹ Metal prefill uses a simdgroup GEMM for every quant type (dispatched when
`nt ≥ 2 && (od ≥ 2048 || nt ≥ 9)`); the scalar f32 multi kernels handle decode
(nt==1) and tiny small-od batches. The compute-graph `MetalBackend` dispatches
these kernels **per op** (`quant_matmul_f32_on_gpu_buf`), so every quant type
above runs on the GPU.
² F32 weights (RMSNorm, biases) are supported on GPU but not for matmul.

**CUDA notes**: prefill (`nt ≥ 16`) runs the default int8 tensor-core MMQ path
for the common quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K via the f16-wmma GEMM,
Q4_K/Q6_K via the raw-nibble int8 kernels — the promoted ~3581 tok/s path, see
[CUDA_OPTIMIZATION.md](./CUDA_OPTIMIZATION.md)); the f32-activation kernels
cover every type including Q5_1/Q5_K, and decode (nt==1) uses the dp4a MMVQ
kernels (Q4_K/Q5_K/Q6_K with shape gates). Q5_K requires `id % 32 == 0`
(tail-masking granularity).

**GPU grouping note**: the old whole-layer `layer_gpu` path required all 7
weight matrices in a layer to share one quant group (all-Q4 or all-QK) and fell
back to CPU otherwise. The compute-graph path (default) has **no such
restriction** — backend assignment is per op, so mixed-group layers run fully
on the GPU. `Raw` weights are not supported on GPU and select the CPU backend
for those ops.

### Not Yet Supported

| Category | Types |
|----------|-------|
| K-quants | Q2_K, Q3_K, Q8_K |
| I-quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| Other | Q1_0, BF16, TQ1_0, TQ2_0, MXFP4, NVFP4 |

Q5_K and Q5_1 are **fully supported on CPU and both GPU backends** — Q5_K_M
models run at full GPU speed.

## Supported Model Architectures

minfer currently supports **two** model architectures.

| Architecture | Variants | Status | Detection Key |
|-------------|----------|:------:|---------------|
| **Qwen2** | Qwen2, Qwen2.5, DeepSeek-R1-Distill-Qwen | ✅ Fully supported | `general.architecture = "qwen2"` |
| **Qwen3** | Qwen3 (dense: 0.6B–32B) | ✅ Fully supported (CPU + GPU) | `general.architecture = "qwen3"` |

Qwen3 support: dense architecture only (no MoE / hybrid-SWA / VL variants yet).
The dense models reuse the Qwen2 graph with two deltas — the head dim is read
from `qwen3.attention.key_length` (decoupled from `n_embd / n_head`) and Q/K go
through a per-head RMSNorm (`blk.{i}.attn_q_norm` / `attn_k_norm`) before RoPE.
See [QWEN3-SUPPORT-PLAN.md](./QWEN3-SUPPORT-PLAN.md) for the design +
verification record.

### How Architecture Detection Works

minfer reads the `general.architecture` string from the GGUF metadata header.
Only the exact values `"qwen2"` and `"qwen3"` (case-sensitive) are accepted. Any
other value produces a clear error:

```text
Unsupported architecture: 'llama'
```

The loader will **not** silently misinterpret a non-Qwen2 model — it fails
immediately with a descriptive message. All model-agnostic components (BPE
tokenizer, Jinja2 chat template renderer, samplers) are ready for additional
architectures once the graph construction (`build_graph`) is added.

### Hyperparameter Keys

The Qwen2 loader reads GGUF keys from both `qwen2.*` and `llama.*` prefixes.
The `llama.*` fallback exists for compatibility with older GGUF converters that
used the `llama.` prefix as a de-facto standard for Llama-family hyperparameters.
This does **not** mean Llama architecture is supported.

### Adding a New Architecture

See [AGENTS.md](https://github.com/yusiwen/minfer/blob/master/AGENTS.md) for a
step-by-step guide. In brief:

1. Create `src/models/<name>/` with `mod.rs`, `graph.rs`, `loader.rs`
2. Add a `match` branch in `src/models/mod.rs::load_model()`
3. Define `HParams`, `LayerWeights`, and implement the `ModelDef` trait
   (including `build_graph(&self, params) -> ComputeGraph`, which is
   deterministic in params — the graph-reuse invariant)

Architectures that share Qwen2's tensor naming convention (LLaMA, Mistral, Phi)
should be relatively straightforward to port.
