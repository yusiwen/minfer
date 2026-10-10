# 0032. The packed-KV staging window is f32, not f16

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #310

## Context

When a packed `Q8_0` KV region has to run through the f32 prefill / windowed-flash kernel families, Metal
does not edit those kernels (`docs/METAL-BACKEND-DESIGN.md` §4.4, "mechanism B"): a
`kernel_dequant_kv_q8_0_to_f32` dequantizes the needed window into a transient, **arena-addressed** scratch
buffer and the unchanged kernel runs against it with `f16 = false`. Because the stage is arena-addressed,
the windowed kernels keep their absolute `lo_min` addressing and the tail-pad kernel its relative offset, so
no kernel in those families is touched.

That leaves one question the surrounding work does not answer: **what element type is the staging buffer?**
It sits between two already-rounding steps — the weight's `Q8_0` quantization and, if the stage were f16,
the stage itself — and the answer is not obvious from the code, which is why it was measured.

## Decision

**The staging buffer is f32.** Mechanism B dequantizes the packed window to f32 and the f32 kernel family
consumes it unchanged, so a staged cell element is rounded exactly once (by the weight's own `Q8_0`
quantization) and never a second time by the stage.

`MINFER_PACKED_KV_STAGE=1` routes a shape that mechanism A would take through mechanism B, so the decode
A/B is measurable; the differential gate `metal_packed_decode_stage_matches_the_native_read` drives that
switch through a `#[cfg(test)]` thread-local rather than the environment.

## Alternatives considered

- **Stage to f16.** Rejected by measurement, and this is the whole substance of the decision: f16 staging
  rounds every already-dequantized cell element a **second** time, which is small per element but is
  amplified over a decode trajectory. On Qwen3-0.6B the interleaved f16-stage run measured **max |Δlogit|
  16.9** (at the argmax 9.46) against the f32 engine — over the same prompt where the classic f32-dequant
  packed path measures **1.29 / 0.43** and the f16 cache measures **0.011**. The prefill-step delta is tiny
  (**0.19**) but compounds; the mechanism is the decode trajectory's sensitivity to one error realization,
  not a kernel fault.
- **Stage to f32 but edit the f16 kernel family to accept it.** Rejected by mechanism B's design: the
  arena-addressed stage is what lets the windowed kernels keep absolute `lo_min` addressing and the
  tail-pad kernel its relative offset, so no kernel in either family is edited at all — an edit would
  reintroduce the `.metal` churn the mechanism exists to avoid.
- **Do not stage at all (use the classic packed kernels everywhere).** Not rejected on correctness — they
  remain the fallback — but it is the path the staging window exists to avoid for the fast prefill/window
  families; `MINFER_NO_*` opt-outs still select it.

## Consequences

- f32 staging restores the **classic / CUDA accuracy class** (`docs/SUPPORT-MATRIX.md`): the ignored
  real-model gate measures **1.28 / 0.36** on Qwen3-0.6B and **2.37 / 0.46** on Qwen2.5-0.5B, both inside
  the inherited **≤ 4.0 tail / ≤ 1.0 argmax** bars, and at parity speed.
- Cost accepted: the transient stage is 4 B/element instead of 2. It is a window, not the cache — the KV
  *regions* stay packed and 3.76× smaller than f32 — so the memory advantage of the packed cache is
  preserved while the second rounding is removed. The size and speed table is the capability's, recorded
  in §4.4 and the #310 record; this ADR owns the element-type choice.
- The measurement lever is a test-only thread-local, so the A/B cannot be flipped in production by
  accident — the same pattern as the capture-mode override (ADR-0031) and the launch-return injection
  lever (ADR-0030).

## References

- `docs/METAL-BACKEND-DESIGN.md` §4.4 — mechanism B, the f32-staging rationale and the measured bar.
- `docs/SUPPORT-MATRIX.md` — the accuracy class and the §"KV Cache Storage Type by Backend" contract.
- [#310](https://github.com/yusiwen/minfer/issues/310) — the packed-read ticket;
  [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
