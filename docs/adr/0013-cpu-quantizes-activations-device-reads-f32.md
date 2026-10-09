# 0013. The CPU quantizes activations to Q8_0; a device reads f32

- Status: Accepted
- Date: 2026-06-24

## Context

The two paths were built for different bottlenecks. On the CPU, an int8×int8 inner loop is what
uses the dedicated dot-product silicon (AVX2 `vpmaddubsw`-class, NEON `SDOT`), and the CPU path
quantizes activations per matmul to `Q8_0`. `docs/CPU_OPTIMIZATIONS.md` states the governing
situation as *"during decode (nt=1), the inference is memory-bound, not compute-bound. The dominant
cost is loading weights from memory, not computation."*

On a device the arithmetic is already there in f32, and the activation is tiny next to the weight.
The `Q8_0`-activation dot is present in the very first commit (`95fd5d1`, 2026-06-24 —
`dot_q4_0_q8_0` in `src/avx2.rs`, `src/forward.rs`, `src/vec_ops.rs`), so this is a founding
decision, not a later optimization.

## Decision

**The two backend classes use different activation representations on purpose**, and each path is
verified against its own reference:

- **CPU**: activations are quantized to `Q8_0` on the fly per matmul (`dot_q*_q8_0()` in
  `src/quants/`), with `Q8_K` for K-quant weights.
- **Metal**: the kernels read f32 activations.
- **CUDA is the exception, on both phases.** Prefill runs the int8 MMQ GEMM (`MINFER_MMQ`, default
  on since r60; `MINFER_MMQ=0` restores the legacy f16 `wmma` GEMM), and decode's MMVQ path also
  quantizes activations to `q8`. So the precise invariant is **CPU-Q8_0 / Metal-f32**, with CUDA
  quantized on both prefill and decode.

The convention is stated in `AGENTS.md` ("CPU quantizes to Q8_0 on the fly … GPU backends read f32
(CUDA prefill uses int8 MMQ)") — note that the parenthetical is load-bearing, not a footnote.

## Alternatives considered

- **A GPU `Q8_0`-activation path in general.** Rejected for three recorded reasons
  (`docs/inference_e2e_walkthrough/14-metal-backend.md` §2.3): there is no int8 hardware to exploit
  (MSL exposes no `SDOT`-class int8×int8 instruction, and *"llama.cpp's Metal backend makes the same
  choice"*); decode bandwidth is bound by **weights**, not activations — for one `[896, 896]` Q4_0
  `attn_q` weight the weights are ≈451 KB against ≈3.5 KB of activations, i.e. **99.2% of the
  traffic**, so quantizing the activation saves ~0.6%; and the two paths are allowed to disagree.
  Metal measured the alternative directly: removing the `Q8_0` quantize for f32 activations gave
  **decode +5–10%** (`docs/METAL_OPTIMIZATIONS.md`, "Done" row 4).
- **Quantize the CPU activation once and reuse the `Q8_0` buffer across projections.**
  Implemented and **rejected by measurement**: 28.0 → 26.0 tok/s, a **−7% regression**, for three
  recorded reasons — added call overhead at `nt=1`, extra memory traffic for a separate
  quantization step, and worse inlining/register allocation.
- **CUDA's own alternative**: the legacy dequantize-to-f16 `wmma` prefill GEMM. It lost the default
  slot first because R1 was parity-clean but ~2.9 TMAC/s against llama.cpp's ~24 (the 8× gap was
  unprofiled), so MMQ shipped opt-in until the r34–r59 campaign; at r60 the promotion row records
  the escape (`MINFER_MMQ=0`) as **~11 GB heavier**, not lighter.

## Consequences

- CPU-vs-GPU logits differ **by design**, so each path is compared against its own reference: the
  CPU path is verified bit-for-bit against llama.cpp, the Metal path against its own f32 reference
  (`max diff < 1e-3`). A test that compared CPU logits to Metal logits would be testing the wrong
  invariant.
- The CPU pays a `Q8_0` quantize per matmul, and the −7% experiment shows sharing it does not pay at
  `nt=1`.
- On CUDA, MMQ buys ~1.080× over the legacy f16 path at the cost of **+3.27 GB** of weight-expansion
  planes, with two escape valves (`MINFER_MMQ_Q6K_EXP`, `MINFER_MMQ_Q4K_DSC`).
- **Contradiction recorded rather than smoothed over:** the convention *"GPU reads f32"* is not
  literally true for CUDA. `docs/CUDA_OPTIMIZATION.md` records MMVQ routing as quantizing activations
  to `q8`, and `AGENTS.md` qualifies its own line with "(CUDA prefill uses int8 MMQ)". A future
  editor should read the rule as CPU-Q8_0 / Metal-f32 / CUDA-quantized.

## References

- `docs/COMPUTE-GRAPH-DESIGN.md` — the Op × backend table ("quantized → `cpu_quant_matmul_f32`
  (Q8_0 activations on the fly)") and the deviations list.
- `docs/CPU_OPTIMIZATIONS.md` — the memory-bound finding and the rejected activation-reuse attempt.
- `docs/inference_e2e_walkthrough/14-metal-backend.md` §2.3 — the three reasons, with the traffic
  arithmetic.
- `docs/METAL_OPTIMIZATIONS.md` — the +5–10% measurement of the alternative.
- `docs/CUDA_OPTIMIZATION.md` — the MMQ gate, the expansion-plane cost and the r60 promotion.
- Commit `95fd5d1` (2026-06-24) — the CPU `Q8_0` dot is present from the first commit; the CUDA
  exception is `40e97c9` (2026-08-31, opt-in) and `57edcf6` (2026-09-06, default on).
