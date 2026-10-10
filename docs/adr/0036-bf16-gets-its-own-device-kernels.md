# 0036. bf16 weights get their own device kernels, not a dtype flag on the f16 ones

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #208

## Context

f16 weights were already on both devices when the second 2 B/element dtype arrived: bf16, the Metal half
of [#208](https://github.com/yusiwen/minfer/issues/208) (PR [#323](https://github.com/yusiwen/minfer/pull/323))
and the CUDA half (PR [#321](https://github.com/yusiwen/minfer/pull/321)). The two dtypes share a geometry
— two bytes per element, promoted in-register to f32 — which makes "reuse the f16 kernel behind a dtype
flag" the obvious economy, and it is the option both halves rejected.

The promotion is not the same operation, though: bf16 is f32's top 16 bits, so its promotion is a
**shift** — `b2f(bits) == __uint_as_float(bits << 16)`, exact, with no rounding at all — while f16's is a
different expansion of a different layout. The two documents state the resulting decision:
`docs/METAL-BACKEND-DESIGN.md` §4.4 ("Its own kernel, not a dtype flag on the f16 one — the same decision
#208's CUDA half made") and `docs/CUDA-BACKEND-DESIGN.md` §4.4 ("The launchers are their own").

## Decision

**Each device gets dedicated bf16 kernels and launchers; the f16 kernels are not parameterised by dtype.**

- **Metal**: `kernel_bf16_f32_matmul` + `kernel_get_rows_bf16`
  (`src/metal/kernels/bf16.metal`, the `pl_bf16_f32` / `pl_get_rows_bf16` pipelines built in `try_new` and
  listed in `build.rs`'s `SHADER_SOURCES`), dispatched by the `TensorType::BF16` arms of
  `quant_matmul_f32_on_gpu_buf` / `embed_tokens_gpu` — the f16 pair's geometry with an in-register
  `as_type<float>(bits << 16)` promotion, the device twin of `crate::block::bf16_to_f32`.
- **CUDA**: `bf16_f32_matmul_vec` when `id % 8 == 0` and `bf16_f32_matmul_scalar` otherwise, plus
  `embed_rows_bf16`, with **their own launchers** (`launch_bf16_f32_matmul` / `launch_embed_rows_bf16`),
  each reading its own launch return through the #147 helpers so a failed launch is an `Err` at the call
  site.
- **The registration stays shared**, which is what keeps two kernels from meaning two contracts: CUDA
  admits bf16 through `models::weight_reg::cuda_weight_reg`, and both loaders' Metal arm is
  `matches!(ttype, F32 | F16 | BF16)` — raw, 2 B/element, no f32 copy.

## Alternatives considered

- **A dtype flag (or a `template` parameter) on the f16 kernel.** Rejected, and the reason is recorded in
  the Metal document: bf16 and f16 are **different 2 B/element layouts**, so a shared kernel "would branch
  per element in the hottest device kernel". The branch would be paid on every element of every matmul to
  save one kernel's source, which is the wrong trade for the inner loop.
- **Dequantize bf16 to f32 and use the f32 path.** Rejected by the residency decision the same ticket
  carries (and ADR-0021 records for the file side): the weights stay raw 2 B/element — a 0.5B bf16 GGUF
  registers **942.4 MiB**, the f16 twin's number, where a dequantized copy would be ~1.9 GiB.
- **Route bf16 into the int8 MMQ GEMM.** Not an alternative at all, and the record says so: MMQ streams
  quantized bytes and bf16 is not a format, so a bf16 prefill does not enter it (ADR-0021's "not an MMQ
  format").

## Consequences

- The two kernels are separately gate-able, and both devices' exactness gates assert **bitwise** against
  the exact shift (`bf16_matmul_matches_the_exact_shift_reference` /
  `bf16_embed_gather_matches_the_reference` on Metal; the `cuda_bf16_*_matches_the_exact_shift_reference`
  pair on CUDA) — a wrong kernel, f16 or f32, is red. That is the payoff of the separation: the gate can
  name the dtype's own reference rather than a shared kernel's behaviour.
- The promotion being **exact** is why the accuracy bars are tight rather than f16-like: the real-model
  gates measure **1.889e-3 / 1.025e-4** relative on Metal (bar 0.05 / 5e-3) and **7.82e-5 / 4.24e-6** on
  CUDA (bar 0.01 / 1e-3), both with identical greedy continuations.
- Cost accepted: two kernels and two launcher pairs to maintain for one dtype, on each device. The shared
  registration rule is what keeps the cost from spreading into the loaders.
- bf16 pays one structural cost for the separate path: `cuda::concat_rows` has **no** 2 B/element arm, so
  the `attn_qkv` / `ffn_gu` concat copies are not registered and the bf16 graph runs the **unfused**
  matmul chain.

## References

- `docs/METAL-BACKEND-DESIGN.md` §4.4 — the Metal kernels, the rejection and the real-model gate.
- `docs/CUDA-BACKEND-DESIGN.md` §4.4 — the CUDA sibling, its launchers and its exactness gates.
- [ADR-0021](0021-bf16-is-round-to-nearest-even-and-1d-stays-f32.md) — the file-side bf16 decision
  (round-to-nearest-even, 1-D stays f32, not an MMQ format); [#208](https://github.com/yusiwen/minfer/issues/208)
  — the ticket; [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
