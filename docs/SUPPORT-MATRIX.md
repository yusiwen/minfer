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
| **F16** | 16 | 2 B / 1 val | ✅³ | ✅³ | ✅⁴ | ✅⁵ |
| **BF16** | 16 | 2 B / 1 val | ✅⁶ | — | ✅⁷ | ✅⁸ |
| **F32** | 32 | 4 B / 1 val | ✅ | — | ✅² | ✅² |

¹ Metal prefill uses a simdgroup GEMM for every quant type (dispatched when
`nt ≥ 2 && (od ≥ 2048 || nt ≥ 9)`); the scalar f32 multi kernels handle decode
(nt==1) and tiny small-od batches. The compute-graph `MetalBackend` dispatches
these kernels **per op** (`quant_matmul_f32_on_gpu_buf`), so every quant type
above runs on the GPU.
² F32 weights are supported on both GPUs: 1-D norms/biases through the norm kernels and 2-D matmul
weights through CUDA's `launch_f32_f32_matmul` and Metal's `kernel_f32_f32_matmul`
([#317](https://github.com/yusiwen/minfer/issues/317)). Before #317 an f32 weight on Metal had no
arm and silently ran the Q4_0 kernel.
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
⁵ Metal registers the raw 2 B/element f16 weights and promotes in-register:
`kernel_f16_f32_matmul` (`src/metal/kernels/f16.metal`) is the f32-activation
matmul and `kernel_get_rows_f16` the embedding gather, both selected by the
`TensorType::F16` arms of `quant_matmul_f32_on_gpu_buf` / `embed_tokens_gpu`
([#164](https://github.com/yusiwen/minfer/issues/164)). The weights stay half
width on the device — no registration-time f32 copy — and, like CUDA, an f16
prefill runs the f32-activation kernel, not a simdgroup GEMM. Both loaders admit
the type, so `weights_on_gpu`'s all-or-nothing check passes and the model is a
Metal model; 1-D norms/biases stay f32 (the file contract above), so an f16 norm
can never reach a `d*2` kernel buffer. Measured on `macbook (macOS 27.0.1, Apple
M4 Pro)` (2026-10-06) against the same file's CPU logits: max |Δlogit| 2.4e-3 on
the 0.5B and 7.9e-3 on Qwen3-0.6B (bar 0.05), with an identical greedy
continuation (`[12095, 11, 323, 432]` for Qwen2, `[12095, 13, 576, 6722]` for
Qwen3).
⁶ BF16 weights ([#142](https://github.com/yusiwen/minfer/issues/142)): the CPU
decodes one row at a time (`vec_ops::mat_mul_bf16`, exact
`f32::from_bits(bits << 16)`, then the same `vec_dot_f32` the f16 row path uses)
and the embedding rows in `Op::GetRows`. The decode is a left shift, so there is
no separate SIMD kernel to mark in the AVX2 column (the dot itself is the
vectorized `vec_dot_f32`). `minfer convert --outtype bf16` writes 2-D bf16 /
1-D f32 and is byte-identical to `llama-quantize --pure <f32>.gguf … BF16`
(docs/GGUF-TOOLING.md §4.1.1).
⁷ CUDA registers the raw 2 B/element bf16 words and promotes in-register —
`bf16_f32_matmul_vec` / `_scalar` (the `uint4` word load split by a `bits << 16`
shift, the f16 pair's exact sibling) and `embed_rows_bf16` — selected by the
`TensorType::BF16` arms of `matmul_f32_ptr_layout` / `embed_rows_on_gpu`
([#208](https://github.com/yusiwen/minfer/issues/208); [#141](https://github.com/yusiwen/minfer/issues/141) is the f16 template). The
decode is **exact** (`f32::from_bits(bits << 16)`), so unlike the quantized types
there is no rounding at all; the weights stay half width on the device — no
registration-time f32 copy — and, like f16, a bf16 prefill runs the
f32-activation kernel, not the int8 MMQ GEMM (MMQ streams quantized bytes and
bf16 is not one of its formats). Both loaders admit the type through the shared
`models::weight_reg::cuda_weight_reg` rule, so `weights_on_cuda`'s all-or-nothing
check passes for **both** supported architectures and the graph's `BF16` matmul /
embed nodes are assigned `Backend::CUDA`; 1-D norms/biases stay f32 (the file
contract above). bf16 does not fuse: `cuda::concat_rows` has no 2 B/element arm,
so the `attn_qkv` / `ffn_gu` concat copies are not registered and the unfused
matmul chain runs. Measured on a GB10 (2026-10-06, `dgxspark`): a 0.5B bf16 GGUF
registers 942.4 MiB of device weights (the same number as its f16 twin, i.e. the
2 B/element claim is real), 169 bf16 matmul + 1 embed nodes on CUDA, device-vs-CPU
max |Δlogit| **7.82e-5** absolute / **4.24e-6** relative (bar 0.01 / 1e-3) with an
identical greedy continuation `[12095, 13, 1084, 374]`.
⁸ Metal registers the raw 2 B/element bf16 words and promotes in-register:
`kernel_bf16_f32_matmul` (`src/metal/kernels/bf16.metal`) is the f32-activation
matmul and `kernel_get_rows_bf16` the embedding gather, both selected by the
`TensorType::BF16` arms of `quant_matmul_f32_on_gpu_buf` / `embed_tokens_gpu`
([#208](https://github.com/yusiwen/minfer/issues/208), the **Metal half**; the
CUDA half is footnote 7). Its own kernel, not a dtype flag on the f16 one — bf16
and f16 are different 2 B/element layouts, so a shared kernel would branch per
element in the hottest device kernel. The weights stay half width on the device
— no registration-time f32 copy — and, like CUDA/f16, a bf16 prefill runs the
f32-activation kernel, not a simdgroup GEMM. Both loaders' Metal arm
(`matches!(ttype, F32 | F16 | BF16)`) admits the type, so `weights_on_gpu`'s
all-or-nothing check passes and the model is a Metal model; 1-D norms/biases
stay f32 (the file contract above). bf16 does not fuse (the fused device forms
are CUDA-only). Measured on a Mac (2026-10-06, `macbook (macOS 27.0.1, Apple
M4 Pro)`) against the same file's CPU logits: 169 bf16 matmul + 1 embed nodes
all on `Backend::METAL`, 942.4 MiB of device weights, max |Δlogit| **1.889e-3**
absolute / **1.025e-4** relative (bar 0.05 / 5e-3), with an identical greedy
continuation `[12095, 13, 1084, 374]`.

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
| `q8_0` | packed Q8_0 blocks, 34 B per 32 elements, cell padded to whole f32 words | ✅ (C4 S1+S2) | ✅ (C4 S2b) | ❌ — Metal's kernels address f32/f16 rows; the packed read is [#310](https://github.com/yusiwen/minfer/issues/310) |

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
- **The KV *write/move* side** (`Backend::copy_cells` for C3 compaction / C8a
  prefix copy / C8b S3 copy-on-write, and `GraphAllocator::copy_kv_to_cpu` for the
  C2 shift and C5 sessions) is implemented on all three backends since #44 part
  (b): Metal moves rows one at a time with `MTLBlitCommandEncoder` in the
  overlap-safe order and reads its regions back through the registry `host_read`
  hook. A **physical shift of an f16 region** refuses loudly and is pinned by a gate
  ([#306](https://github.com/yusiwen/minfer/issues/306): the host round trip has
  no dequantize → re-rope → requantize map, so the CLI re-renders the retained
  window instead); the per-engine `kv_format` is what
  makes a Metal session describe the width its region really uses.

### Not Yet Supported

| Category | Types |
|----------|-------|
| K-quants | Q2_K, Q3_K, Q8_K |
| I-quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| Other | Q1_0, TQ1_0, TQ2_0, MXFP4, NVFP4 |

Q5_K and Q5_1 are **fully supported on CPU and both GPU backends** — Q5_K_M
models run at full GPU speed.

## Operator Coverage by Backend

`supports_op` decides at graph-build time which backend runs each node
(`docs/ARCHITECTURE.md` §5). This table is the contract, generated from the
three implementations — keep it in step with them.

The table's concrete **F32 `Op` rows** are pinned by
`graph::op_matrix::support_table_matches_support_matrix_doc`, which fails when a
backend's `supports_op` disagrees (each backend column is checked wherever it is
compiled in). Two kinds of row it *cannot* check, so they are prose plus their
own gates: the **composite rows** (one line spelling several ops) and the
**capability nuances that are not an `Op` field** — a *partial* `View` at offset
0 (the allocator backstops it; `supports_op` sees only the offset), the
set-valued `kv_map` attention window (`Device::gathers_attn_map`), and a packed
`q8_0` KV region (`BackendCaps::reads_packed_kv`).

| Operator | CPU | Metal | CUDA |
|---|:---:|:---:|:---:|
| `Input`, `KvcacheLoad`, `View`/`Reshape`/`Permute` | ✅ | ✅ | ✅ |
| `Add`, `Mul`, `Silu` | ✅ | ✅ | ✅ |
| `RmsNorm`, `QkNorm` | ✅ | ✅ | ✅ |
| `MatMul` | ✅ | ✅ | ✅ |
| `GetRows` (embedding, tail rows) | ✅ | ✅ | ✅ |
| `View` with `offset != 0` or a partial window (D1) | ✅ | ❌ | ✅ — Metal's kernels take a buffer and a length with no element offset, so it can express exact views only: a **standing design limit**, not a pending port (G5 landed the attention window and the KV cell store, *not* offset views — `Op::View { offset, .. } => *offset == 0`); it is what keeps the hand-written `Op::FusedFFN` on Metal (§D3). The allocator backstops the partial case, which `supports_op` cannot see |
| `Attn` | ✅ | ✅ | ✅ |
| `Attn` with `explicit_span` (a window that starts at a non-zero cell, or several sequences in one batch) | ✅ | ✅ | ✅ — the one-range `attn_span` window is read on all three backends (Metal's `kernel_gqa_attn_window_f32/_f16` landed in #44 part (a), and #44 part (b) gave Metal the matching write/move side so a batched and compacted multi-sequence run serves; CUDA's E1b instantiation is **device-verified** on GB10, including a bitwise batch-order-invariance gate). The set-valued `kv_map` window is read on all three backends too: CPU/CUDA always did, and Metal's sibling `kernel_gqa_attn_map_f32/_f16` landed in [#362](https://github.com/yusiwen/minfer/issues/362) (`Device::gathers_attn_map` is now true for Metal). A packed `q8_0` KV cache is still refused on Metal ([#310](https://github.com/yusiwen/minfer/issues/310)) |
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
- **`QkvBiasRopeStore` is CUDA-only — a recorded decision ([#52](https://github.com/yusiwen/minfer/issues/52)),
  not a gap.** It is the *mixed-quant* decode epilogue: q/k/v use different quant
  types (so they cannot share `FusedQKV`'s single concat weight), so three separate
  matmuls (no bias) feed one bias×3 + RoPE×2 + store×2 pass. CUDA fuses it (10
  dispatches → 4 per layer, −6); Metal keeps the unfused chain and the graph builder
  never emits the node there (`metal_backend.rs`'s `false` arm is a design statement).
  Porting would save **6 dispatches per mixed-quant layer — 84 per decode token on
  Qwen2.5-7B-Q4_K_M**, the realistic case, whose 14 of 28 layers carry `attn_v` as
  Q6_K against Q4_K q/k — with **no numerical difference** (`supports_op` is a
  build-time gate and the unfused chain is the reference). The whole forward's
  host-encode is ~0.2 ms against a ~20 ms/token decode, so those 84 dispatches are a
  sub-1% slice of decode time; the A/B of the *concat*-class fusion that removes more
  dispatches (`MINFER_NO_FUSE_QKV=1`, −8 on the same 14 layers) sits within run-to-run
  variance on `macbook (macOS 27.0.1, Apple M4 Pro)` (2026-10-06, five interleaved
  `bench -p 0 -n 128 -r 4` pairs: 48.06 vs 46.49 t/s means, individual pairs crossing
  zero), so a second kernel path and its bitwise gate are not earned by a ~1% ceiling.
  **Provenance and the rejected alternative, stated so the numbers are not misread.** The ~0.2 ms
  host-encode figure is `MINFER_OP_PROFILE=1` on that 7B — per-op host-encode **totals**, not
  per-label counts — and the 10 → 4 / 84-per-token dispatch counts are the CUDA D3-8 ledger applied
  to Metal's dispatch table; Metal has no per-op profiler, so they were not re-counted on the
  device. And the port is *cheap*: Metal already has the class-1 `attn_bias_rope_store` kernel, so
  the refused work is mainly a three-pointer binding — the decision rests on the measured ceiling,
  not on the size of the change.
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
