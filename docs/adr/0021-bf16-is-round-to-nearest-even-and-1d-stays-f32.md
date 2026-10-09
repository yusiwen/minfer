# 0021. bf16 is a round-to-nearest-even cast, and 1-D tensors stay f32

- Status: Accepted
- Date: 2026-09-27
- Issues: #142, #208, #209

## Context

`--outtype bf16` was refused, and the engine had no bf16 weight path: the GGUF type was parsed but
neither `mat_mul_*` nor the device kernels dispatched it, so writing a bf16 file would have produced
something the engine could not run. The ticket asked for round-to-nearest-even (RNE) writes and warned
that the accuracy bound *"is looser and must be stated"* — bf16 has 8 mantissa bits against f16's 10.

The interesting part of this decision is the **reference**, not the cast: bf16 output could have been
verified against a cast of the f16 file, and that was tried first.

## Decision

**bf16 output is a round-to-nearest-even `f32 → bf16` cast with a forced-quiet NaN, and 1-D tensors
stay f32 in the file.**

- `src/convert.rs::f32_to_bf16_bits`: NaN keeps the sign and top mantissa bits with the quiet bit set;
  otherwise the value round-to-nearest-even on the top 16 bits. `general.file_type` is 32
  (`LLAMA_FTYPE_MOSTLY_BF16`).
- `OutType::keeps_1d_f32()` is true for `F16 | Bf16`, and `tensor_type(ndims)` returns `GgmlType::F32`
  for `ndims <= 1`: 2-D weights are bf16, 1-D norms and biases are f32. This is llama.cpp's rule
  (`conversion/base.py` sets `data_type = F32` for `n_dims <= 1` on every file type).
- The byte reference is **the f32 conversion cast to bf16**, not the f16 file cast: `290/290 tensor
  payloads byte-identical` (169 BF16 2-D, 121 F32 1-D).

## Alternatives considered

- **Cast from the f16 source** (the first reference tried: `llama-quantize --pure <f16>.gguf … BF16`).
  **Rejected by measurement**: casting the f16 file cannot recover the **123 024 subnormal** values the
  f16 conversion already rounded, so 126 575 bytes across all 169 2-D tensors differ. The chosen
  reference is cast from the f32 conversion, which is exact for a bf16 source and therefore yields the
  f32→bf16 RNE projection of the checkpoint's own values.
- **The direct `convert_hf_to_gguf.py --outtype bf16` converter reference.** Not rejected on merit —
  **unavailable** here because it needs torch. The f32-source cast is a valid per-tensor byte
  reference but shares minfer's own RNE rule by construction, so the independent check remains open as
  [#209](https://github.com/yusiwen/minfer/issues/209).
- **Truncation instead of RNE.** **No rejected alternative is recorded.** The record states the RNE
  rule, its algorithm and its measurement against the reference; it never weighs truncation.
- **Keeping 1-D tensors in bf16.** No alternative is recorded for `convert` — the rule is stated as
  llama.cpp's. The sibling case is where the *bug* was: `quantize --type f16` had converted 1-D
  tensors to f16, and that regression was fixed by
  [#169](https://github.com/yusiwen/minfer/issues/169).

## Consequences

- A bf16 GGUF produced by `minfer convert` matches llama.cpp's cast byte for byte and runs — on the
  CPU from the start, and on CUDA and Metal since [#208](https://github.com/yusiwen/minfer/issues/208).
- bf16 is **not an MMQ format**, so it never enters the int8 MMQ GEMM (ADR-0013), and it does not
  fuse: `cuda::concat_rows` has no 2 B/element arm, so `blk.{i}.attn_qkv` / `blk.{i}.ffn_gu` are not
  registered and the census is the unfused 169 matmul + 1 embed shape.
- Registering the raw 2 B/element weights keeps the memory advantage: 942.4 MiB of device weights,
  the same number its f16 twin reports, where a dequantized copy would be ~1.9 GiB.
- Measured deltas, each against a stated bound: CPU bf16-vs-f16 logits 2.29e-5 / 1.24e-6 relative;
  CUDA device-vs-CPU **7.82e-5** absolute / **4.24e-6** relative against a bound of 0.01 / 1e-3;
  Metal **1.889e-3** / **1.025e-4** relative against 0.05 / 5e-3 — with identical greedy
  continuations.
- Exactness is per-step and the record says which step is not: f16/f32 copies and f16→f32, bf16→f32
  are bit-exact, but **bf16→f16 is exact in the mantissa and not in the exponent range** — it can
  overflow to inf, and below f16's smallest normal it rounds onto the subnormal grid (123 024 values
  on the 0.5B checkpoint).

## References

- `docs/GGUF-TOOLING.md` §4.1.1 — the bf16 writer contract and the byte-identity reference.
- `docs/SUPPORT-MATRIX.md` ("f16 and bf16 weights") — the file contract and the device deltas.
- `src/convert.rs` — `f32_to_bf16_bits`, `OutType::keeps_1d_f32`; `src/gguf_write.rs`.
- Commit `5f3f26e` (2026-09-27, "feat(tooling): bf16 output and a CPU bf16 weight path (#142)"); the
  device halves landed 2026-10-06 (#208).
