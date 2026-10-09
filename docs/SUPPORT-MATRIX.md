# Support Matrix

Supported quantization formats and model architectures. This page is the expanded version of the two support sections formerly in the README.

## Supported Quantization Formats

minfer supports GGUF v3 files with the following quantized weight types. The CPU backend quantizes activations on-the-fly (Q8_0 for the simple weight types, **Q8_K** — llama.cpp's format with precomputed per-subblock sums — for Q4_K/Q5_K/Q6_K); the GPU backends read f32 activations directly for their non-MMQ kernels, matching llama.cpp's Metal backend.

### Supported

| Type | Bits | Block | CPU | AVX2 | CUDA GPU | Metal GPU |
|------|------|-------|:---:|:----:|:--------:|:---------:|
| **Q4_0** | 4 | 18 B / 32 val | ✅ | ✅ | ✅ | ✅ |
| **Q4_1** | 4 | 20 B / 32 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q4_K** | 4 | 144 B / 256 val | ✅ | ✅⁹ | ✅ | ✅¹ |
| **Q5_0** | 5 | 22 B / 32 val | ✅ | ✅ | ✅ | ✅¹ |
| **Q5_1** | 5 | 24 B / 32 val | ✅ | ❌ | ✅ | ✅¹ |
| **Q5_K** | 5 | 176 B / 256 val | ✅ | ✅⁹ | ✅ | ✅¹ |
| **Q6_K** | 6 | 210 B / 256 val | ✅ | ✅⁹ | ✅ | ✅¹ |
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

⁹ The K-quant dots ([#56](https://github.com/yusiwen/minfer/issues/56), 2026-10-08)
have AVX2+FMA kernels (`src/quants/avx2.rs`) and AVX-512/VNNI variants
(`src/quants/avx512.rs`), dispatched **AVX-512 → AVX2 → scalar** at runtime
(`MINFER_NO_AVX512=1` drops to AVX2, `MINFER_NO_AVX2=1` to scalar — the x86
counterparts of `MINFER_NO_NEON`) and gated **bitwise** against the scalar
reference (`quants::avx2_correctness`). The AVX2 column marks the hand-written x86
kernel; NEON is folded into CPU as in every other row.

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
| `q8_0` | packed Q8_0 blocks, 34 B per 32 elements, cell padded to whole f32 words | ✅ (C4 S1+S2) | ✅ (C4 S2b) | ✅ (C4 S2b, #310 enabled) |

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
  the general layout-tagged kernel and the hybrid 4-warp decode dispatch is
  f16-typed. The **`dp4a` packed K dot is not a follow-up** — it landed in
  [#186](https://github.com/yusiwen/minfer/issues/186) (`cc19b4f`, 2026-09-27) as
  the packed decode route, and `docs/ARCHITECTURE-EXECUTION-PLAN.md` §C4 #186
  records it **DONE** with its tolerance class re-measured. The general
  layout-tagged kernel remains the fallback for every packed path. A
  **speculative** session refuses a packed cache outright (its greedy identity
  contract rests on the batched split kernel). See
  `docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 C4 #144 and
  `docs/cuda_optimization_steps/107-c4-packed-q8-kv-cuda.md`.
- **Metal ([#310](https://github.com/yusiwen/minfer/issues/310)) — enabled.**
  `kernel_store_kv_q8_0` writes the same bytes as the CPU quantizer. Two read mechanisms cover every
  attention shape, selected by the pure `crate::metal::packed_attn_route`: **mechanism A** reads packed
  cells natively in the decode flash family (`kernel_flash_attn_ext_q8_0` / `_hd128_q8_0`, `nt == 1`,
  `hd ∈ {64,128}`); **mechanism B** dequantizes the needed window into a transient **f32** stage
  (`kernel_dequant_kv_q8_0_to_f32`) and runs the unchanged f32 prefill / windowed-flash family. The
  classic `kernel_gqa_attn_q8_0` / `_window_q8_0` / `_map_q8_0` remain the fallback for a small/odd
  `hd`, an `nt == 1` explicit window and any `MINFER_NO_*` opt-out. `READS_PACKED_KV` is now **true**,
  so `MINFER_CACHE_TYPE=q8_0` loads and runs on Metal. The memory win is 3.76×; the measured speed
  (`macbook (macOS 27.0.1, Apple M4 Pro)`, 2026-10-08, `minfer bench -p 1024 -n 64 -r 3 --n-ctx 2048`,
  3 interleaved runs, medians; f16 baseline): Qwen3-0.6B `pp1024` 4844 → 4683 tok/s (0.967×) and `tg64`
  193.5 → 176.6 (0.913×), Qwen2.5-0.5B `pp1024` 6175 → 6154 (0.997×) and `tg64` 295.7 → 245.5 (0.830×);
  region sizes 58 720 256 → 15 597 568 B and 6 291 456 → 1 671 168 B (both 3.76×). The stage is f32, not
  f16: f16 staging's second rounding was measured to amplify to 16.9 logit delta on Qwen3-0.6B over 8
  decode steps, outside the inherited C4 class; f32 staging restores it (1.28 / 0.36) at parity speed.
  `docs/METAL-BACKEND-DESIGN.md` §4.4 records the mechanisms, the measurements and the gates.
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

## f16 and bf16 weights — the file contract and the measured device deltas

Moved here from `AGENTS.md` (its Support section keeps the one-line summary).

f16 **weights** (F6/#49; device half by [#141](https://github.com/yusiwen/minfer/issues/141) on CUDA, [#164](https://github.com/yusiwen/minfer/issues/164) on Metal): an f16 GGUF runs on **CPU, CUDA and Metal** for **both supported architectures** (Qwen2/Qwen2.5 and Qwen3). Both device loaders register the raw 2 B/element weights and the kernels promote in-register; the CUDA side shares one registration rule with the CPU (`models::weight_reg`, which also carries the q4_K `W_dsc` plane gate of [#165](https://github.com/yusiwen/minfer/issues/165)), while the Metal side is the loader's `matches!(ttype, F32 | F16 | BF16)` registration arm (the BF16 arm is #208's Metal half). `Op::MatMul` decodes one f16 weight row at a time (`vec_ops::mat_mul_f16`) and `Op::GetRows` decodes f16 embedding rows; **1-D norms/biases stay f32** — the file contract every producer writes (`minfer convert --outtype f16`, `llama-quantize … F16`, `minfer quantize --type f16`), and the CUDA norm path checks the registered byte length so an f16 norm cannot reach a `d*2` buffer ([#169](https://github.com/yusiwen/minfer/issues/169)). CUDA registers the raw 2 B/element weights and converts in-register (`f16_f32_matmul_vec` / `_scalar`, `embed_rows_f16`); an f16 prefill does **not** enter the int8 MMQ GEMM — it runs the f32-activation kernel. Metal is the same shape: `kernel_f16_f32_matmul` + `kernel_get_rows_f16` (`src/metal/kernels/f16.metal`) are its f32-activation matmul/embed, selected by the `TensorType::F16` arms of `quant_matmul_f32_on_gpu_buf` / `embed_tokens_gpu`, and its prefill runs that same f32-activation kernel (no simdgroup GEMM). `minfer quantize` supports q4_0/q4_1/q5_0/q5_1/q8_0 **and the K-quants q4_K/q5_K/q6_K** (byte-identical to `llama-quantize`; the K-quant reference is `llama-quantize --pure`, and `--type q4_K` writes one uniform type, not the `Q4_K_M` mixture — [#203](https://github.com/yusiwen/minfer/issues/203)), f16 and f32, and refuses every type without an encoder by name ([#140](https://github.com/yusiwen/minfer/issues/140)); a K-quant row that is not a multiple of 256 is demoted the way llama.cpp's `tensor_type_fallback` does it; `--type f16` keeps 1-D tensors f32, `--type f32` writes every tensor f32. The gate **inputs** are recorded: `tests/fixtures/f6-fixtures.json` carries one entry per cached artifact content (path, bytes, sha256, the exact producer command, the producer identity — minfer commit, or llama.cpp commit + compiler + effective `-ffp-contract` — date and an absolute box label), `scripts/check_f6_fixtures.py` audits that manifest in CI, verifies the whole cache where it exists, and — since [#345](https://github.com/yusiwen/minfer/issues/345) — re-runs a recorded producer with `--regenerate` and re-records the content identity it produces (idempotently; a `llama-quantize` reference, or a `minfer` entry whose recorded commit is not the one that runs, is refused by name rather than re-recorded), and every F6 gate verifies the fixture it resolves (`src/tooling/tests/f6_fixtures.rs`) so a stale or replaced reference refuses the run by name and digest instead of being compared against — [#205](https://github.com/yusiwen/minfer/issues/205).

bf16 **weights** ([#142](https://github.com/yusiwen/minfer/issues/142)): `minfer convert --outtype bf16` writes 2-D bf16 / 1-D f32, f32→bf16 being **round-to-nearest-even** (`ggml_compute_fp32_to_bf16`, NaN forced quiet; `general.file_type = 32` = `LLAMA_FTYPE_MOSTLY_BF16`). The CPU decodes one bf16 weight row at a time (`vec_ops::mat_mul_bf16`, exact `f32::from_bits(bits << 16)`, then the same `vec_dot_f32` the f16 row path uses) and `Op::GetRows` decodes bf16 embedding rows; `TensorType::BF16` is the new type the loader maps `GgmlType::BF16` to. **CUDA registers bf16 since [#208](https://github.com/yusiwen/minfer/issues/208)'s CUDA half** — `models::weight_reg::cuda_weight_reg` answers `Raw` for it (the f16 arm's sibling: 2 B/element, no f32 copy, `clear_nb_bt_only`), `Op::MatMul` takes the `TensorType::BF16` arm of `matmul_f32_ptr_layout` (`bf16_f32_matmul_vec` when `id % 8 == 0`, else `_scalar`; both shift `bits << 16` in-register, which is exact) and `Op::GetRows` the BF16 arm of `embed_rows_on_gpu` (`embed_rows_bf16`), and both loaders' CUDA branch admits the type through that one shared rule, so the all-or-nothing gate answers `Cuda` for **both** supported architectures. bf16 is not an MMQ format, so its prefill runs the f32-activation kernel; `cuda::concat_rows` has no 2 B/element arm, so the `attn_qkv`/`ffn_gu` concat copies are not registered and bf16 runs unfused. Measured on GB10 2026-10-06 (`dgxspark`, `scripts/cuda_test.sh` device gate `f208_bf16_weights_run_on_the_cuda_device`): 0.5B bf16 registers 942.4 MiB of device weights (the f16 twin's number), 169 bf16 matmul + 1 embed nodes all assigned `Backend::CUDA`, device-vs-CPU max |Δlogit| **7.82e-5** absolute / **4.24e-6** relative (bar 0.01 / 1e-3), greedy `[12095, 13, 1084, 374]` identical. **Metal registers it too since [#208](https://github.com/yusiwen/minfer/issues/208)'s Metal half** — `kernel_bf16_f32_matmul` + `kernel_get_rows_bf16` (`src/metal/kernels/bf16.metal`, the `pl_bf16_f32` / `pl_get_rows_bf16` pipelines) are its f32-activation matmul/embed, selected by the `TensorType::BF16` arms of `quant_matmul_f32_on_gpu_buf` / `embed_tokens_gpu`, and both loaders' Metal arm is now `matches!(ttype, F32 | F16 | BF16)`, so `weights_on_gpu` passes and both supported architectures answer `Device::Metal`. Measured on a Mac (`macbook (macOS 27.0.1, Apple M4 Pro)`, 2026-10-06, device gate `f208_bf16_weights_run_on_the_metal_device`): the 0.5B bf16 file's 169 bf16 matmul + 1 embed nodes are all assigned `Backend::METAL`, 942.4 MiB of device weights, device-vs-CPU max |Δlogit| **1.889e-3** absolute / **1.025e-4** relative (bar 0.05 / 5e-3), greedy `[12095, 13, 1084, 374]` identical. The writer is **byte-identical to `llama-quantize --pure <f32>.gguf … BF16`, 290/290 tensor payloads** (169 BF16 2-D, 121 F32 1-D; docs §4.1.1 — the reference is cast from the *f32* conversion, because bf16→f16 is lossy below 2^-14 and the f16 file cannot carry those values). On the bf16-source 0.5B, bf16-vs-f16 CPU logits differ by max **2.29e-5** / **1.24e-6** relative (not bitwise — the f16 file's 123 024 subnormal weight values are the whole difference) with an **identical greedy continuation** `[12095, 13, 1084, 374]`.

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
| `Attn` with `explicit_span` (a window that starts at a non-zero cell, or several sequences in one batch) | ✅ | ✅ | ✅ — the one-range `attn_span` window is read on all three backends (Metal's `kernel_gqa_attn_window_f32/_f16` landed in #44 part (a), and #44 part (b) gave Metal the matching write/move side so a batched and compacted multi-sequence run serves; CUDA's E1b instantiation is **device-verified** on GB10, including a bitwise batch-order-invariance gate). The set-valued `kv_map` window is read on all three backends too: CPU/CUDA always did, and Metal's sibling `kernel_gqa_attn_map_f32/_f16` landed in [#362](https://github.com/yusiwen/minfer/issues/362) (`Device::gathers_attn_map` is now true for Metal). A packed `q8_0` KV cache is read on Metal too since [#310](https://github.com/yusiwen/minfer/issues/310) (mechanisms A and B), alongside CPU and CUDA |
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

## Decisions governing this document

This page is the *current contract*; the decisions behind it are frozen in the ADR corpus:
- [ADR-0013](adr/0013-cpu-quantizes-activations-device-reads-f32.md) — The CPU quantizes activations to Q8_0; a device reads f32
- [ADR-0021](adr/0021-bf16-is-round-to-nearest-even-and-1d-stays-f32.md) — bf16 is a round-to-nearest-even cast, and 1-D tensors stay f32
