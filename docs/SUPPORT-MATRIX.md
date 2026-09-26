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
| **F16** | 16 | 2 B / 1 val | ✅³ | ✅³ | ✅⁴ | ❌⁵ |
| **F32** | 32 | 4 B / 1 val | ✅ | — | ✅² | ✅² |

¹ Metal prefill uses a simdgroup GEMM for every quant type (dispatched when
`nt ≥ 2 && (od ≥ 2048 || nt ≥ 9)`); the scalar f32 multi kernels handle decode
(nt==1) and tiny small-od batches. The compute-graph `MetalBackend` dispatches
these kernels **per op** (`quant_matmul_f32_on_gpu_buf`), so every quant type
above runs on the GPU.
² F32 weights (RMSNorm, biases) are supported on GPU but not for matmul.
³ F16 has no block: 2 B per element, so there is no integer dot to run. The CPU
dot is vectorized — AVX2 uses `F16C` (`_mm256_cvtph_ps`) and aarch64 uses
baseline NEON `FCVTL` (`vcvt_f32_f16`) — with an f64 scalar oracle/fallback
(`vec_ops::dot_f16_f32`, `f16_dot_path()`), and the multi-token prefill decodes
each weight row once and threads the row loop through the shared CPU pool
(#141). The AVX2 column marks the hand-written x86 kernel; NEON is folded into
CPU as in every other row.
⁴ CUDA decodes in-register (`f16_f32_matmul_vec` / `_scalar`, `__half22float2`)
and the embedding gather has its own f16 kernel — the weights stay 2 B/element
on the device, which is the point of the format. No MMQ route: MMQ streams
*quantized* bytes and f16 is not one of its formats, so an f16 prefill runs the
f32-activation kernel. **Both supported architectures** (Qwen2/Qwen2.5 and
Qwen3) use it: [#141](https://github.com/yusiwen/minfer/issues/141) landed the
registration branch and the graph type gate in the qwen2 loader/graph only, so
until [#167](https://github.com/yusiwen/minfer/issues/167) an f16 **Qwen3**
model fell to the CPU on a CUDA build even though these kernels existed; the
loaders now share one registration rule (`models::weight_reg`). The engine's f16
**file** contract is 2-D tensors f16 and 1-D norms/biases f32 (llama.cpp's rule;
`mat_mul_f16`/the f16 embed decode have no f16-norm sibling) — what `minfer
convert --outtype f16` writes.
⁵ **Metal refuses f16 weights**, so an f16 GGUF runs the CPU path there
(loudly, through the loader's all-or-nothing registration check). A registered
weight with no kernel would be a silent wrong path, which is exactly what that
check exists to prevent; the Metal f16 matmul/embed kernels are
[#162](https://github.com/yusiwen/minfer/issues/162).

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

### KV Cache Storage Type by Backend

`MINFER_CACHE_TYPE` picks the **KV cache** element type, which is a separate
axis from the weight type above (`graph/kvformat.rs` is the single authority,
and the answer for "can this backend read it" is the registry's
`reads_packed_kv`). A value the backend has no kernel for is **refused at load**,
never silently mapped to f32.

| `MINFER_CACHE_TYPE` | Cell | CPU | CUDA | Metal |
|---|---|:---:|:---:|:---:|
| `f32` (default) | 4 B/element, f32 | ✅ | ✅ | ✅ |
| `f16` | 2 B/element in the f32-shaped region | → f32 | ✅ | ✅ |
| `q8_0` | packed Q8_0 blocks, 34 B per 32 elements, cell padded to whole f32 words | ✅ (C4 S1+S2) | ✅ (C4 S2b) | ❌ — G5, [#44](https://github.com/yusiwen/minfer/issues/44) |

Notes:

- **`f16` on the CPU resolves to `f32`** — the CPU has no f16 KV kernel, and an
  env var set for a GPU run must not break a CPU one.
- **The default on CUDA/Metal is the model's own auto policy** (f16 when
  `n_layers × n_kv_embd ≥ 8192`, i.e. the 7B class, f32 for small models); the
  table's "default" row is the *region shape*, which f16 does not change.
- **Q8_0 is the packed one**: 3.76× smaller than f32 and 1.88× smaller than the
  f16 auto policy. On CUDA a Q8_0 decode runs the layout-tagged split-K kernel
  together with the **packed fused QKV epilogue** (`attn_bias_rope_store_q8_0`,
  #144), and a prefill at head dim 128 runs the **packed FA prefill** — its
  f16-tile staging dequantizes each packed block, so the tensor-core route is
  offered for a packed cell too (#144: Qwen3-0.6B `pp2048` 564.5 → 8231.1 tok/s).
  Still off their tuned route, and stated: the verify band (`1 < nt ≤ 16`) takes
  the general layout-tagged kernel, the hybrid 4-warp decode dispatch is
  f16-typed, and a **`dp4a` packed K dot** is a follow-up (it is a numerics
  change needing its own accuracy statement). The general layout-tagged kernel
  remains the fallback for every packed path. A **speculative** session refuses a
  packed cache outright (its greedy identity contract rests on the batched split
  kernel). See `docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 C4 #144 and
  `docs/cuda_optimization_steps/107-c4-packed-q8-kv-cuda.md`.
- **A Q8_0 cell width must be a whole number of 32-element blocks** (so `n_kv_embd
  % 32 == 0`, which every supported architecture satisfies); `ensure_kv` refuses
  anything else.

### Not Yet Supported

| Category | Types |
|----------|-------|
| K-quants | Q2_K, Q3_K, Q8_K |
| I-quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| Other | Q1_0, BF16, TQ1_0, TQ2_0, MXFP4, NVFP4 |

Q5_K and Q5_1 are **fully supported on CPU and both GPU backends** — Q5_K_M
models run at full GPU speed.

## Operator Coverage by Backend

`supports_op` decides at graph-build time which backend runs each node
(`docs/ARCHITECTURE.md` §5). This table is the contract, generated from the
three implementations — keep it in step with them.

| Operator | CPU | Metal | CUDA |
|---|:---:|:---:|:---:|
| `Input`, `KvcacheLoad`, `View`/`Reshape`/`Permute` | ✅ | ✅ | ✅ |
| `Add`, `Mul`, `Silu` | ✅ | ✅ | ✅ |
| `RmsNorm`, `QkNorm` | ✅ | ✅ | ✅ |
| `MatMul` | ✅ | ✅ | ✅ |
| `GetRows` (embedding, tail rows) | ✅ | ✅ | ✅ |
| `View` with `offset != 0` or a partial window (D1) | ✅ | ❌ | ✅ — Metal's kernels take a buffer and a length with no element offset, so it can express exact views only (G5); the allocator backstops the partial case, which `supports_op` cannot see |
| `Attn` | ✅ | ✅ | ✅ |
| `Attn` with `explicit_span` (a window that starts at a non-zero cell, or several sequences in one batch) | ✅ | ❌ | ✅ — Metal still derives the bound from `positions` (G5); CUDA's windowed instantiation (E1b) is **device-verified** on GB10 (sm_121), including a bitwise batch-order-invariance gate |
| `KvcacheStore` | ✅ | ✅ | ✅ |
| `SwiGLU` (fused) | ✅ | ✅ | ✅ |
| `RoPE` non-interleaved | ✅ | ✅ | ✅ |
| `RoPE` interleaved | ✅ | ✅ | ❌ |
| `FusedQKV` (decode) | ❌ | ✅ | ✅ |
| `FusedFFN` (decode) | ❌ | ✅ | ✅ |
| `FusedQkvNorm` (Qwen3 decode) | ❌ | ✅ | ❌ |
| `QkvBiasRopeStore` (mixed-quant decode) | ❌ | ❌ | ✅ |
| `Scale`, `Softmax`, `BatchMatMul` | ✅ / ✅ / ❌ | ❌ / ❌ / ❌ | ❌ / ❌ / ❌ |

Notes on the asymmetries — these are the rows where a model's decode path
differs by platform:

- **`FusedQkvNorm` is Metal-only.** Qwen3 decode on CUDA takes the unfused
  `QkNorm` path, which is numerically equivalent but issues more dispatches.
  Making CUDA fused is a Phase G / CUDA-verifiable ticket, not a correctness gap.
- **`QkvBiasRopeStore` is CUDA-only.** On Metal the mixed-quant decode layers
  keep the unfused bias+rope+store chain; the graph builder never emits the node
  there (`metal_backend.rs`'s `false` arm is a design statement, not a gap).
- **Interleaved RoPE is CPU-only.** Both loaders hard-code `NonInterleaved`
  today, so no shipped model hits this; a family that needs interleaved RoPE
  needs a loader change plus a CUDA kernel.
- **`Scale`/`Softmax` are CPU-only and unused.** Attention kernels fuse the
  softmax and carry the scale in `AttnMeta`, so no supported architecture emits
  either node.
- `BatchMatMul` is deferred everywhere (single-output IR) and nothing emits it.

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
