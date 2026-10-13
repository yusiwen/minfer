# 0041. A row that cannot take the target's block is demoted, the legacy targets included

- Status: Accepted
- Date: 2026-10-13
- Issues: #204
- Corrects: ADR-0020

## Context

ADR-0020 recorded the legacy half of llama.cpp's `tensor_type_fallback` as an open residual, in its
own words: *"Not an alternative but a recorded residual: for **legacy** targets `minfer` keeps the
source type where llama.cpp demotes it (`q4_0` → `F16`), so an f32 source with such a row would
differ ([#204](https://github.com/yusiwen/minfer/issues/204), open)."* It was accurate. The plan
rule was `keep = 1-D || row_not_aligned`, so a 2-D tensor whose `ne[0]` was not a multiple of the
target's block was **copied verbatim** — the source type, whatever it was — while llama.cpp demotes
it. The K-quant half of the same table had been implemented by [#140](https://github.com/yusiwen/minfer/issues/140);
only the legacy five targets were left.

The two rules coincided on every fixture the gates held, which is why nothing failed: every source in
`~/.cache/minfer/f6-src/` is f16 and every 2-D row in it is a multiple of 32 (measured: 169 of 169
and 197 of 197 on the 0.5B and the 0.6B), so *"keep the source type"* and *"demote to F16"* were the
same bytes. The divergence needed a source that was **not** already f16.

It was measured on `dgxspark` (2026-10-13) against the reference build the record names, by making
one row unrepresentable: a copy of the recorded `qwen2.5-0.5b-instruct-f32.gguf` with
`blk.0.ffn_down.weight`'s `ne[0]` changed 4864 → 4863 and the following 287 tensor offsets shifted by
the same 3584 bytes, so both loaders still accept the file (they require a contiguous data section).

- `llama-quantize <patched> out q4_0` → `warning: blk.0.ffn_down.weight - ncols 4863 not divisible by
  32 (required for type q4_0) (WARNING: must use F16 due to unusual shape) -> falling back to f16`,
  exit 0. **289 of 290 tensors byte-identical** to minfer's file; the one difference was that tensor,
  written **f16** by llama.cpp and **f32** by minfer, and the file sizes differed by exactly the
  4 → 2 B/element change (8 714 496 B).
- A bf16 source behaved the same way with a third answer: llama.cpp wrote **f16**, minfer wrote
  **bf16**. The divergence was never about `f32`; it was *"preserve whatever the source was"*.
- The **tied embedding** is where the order of the two rules shows. With `token_embd.weight`'s
  `ne[0]` = 895, `llama-quantize` reported `ncols 895 not divisible by 32 (required for type q8_0)
  -> falling back to f16`: llama.cpp resolves its mixture to q8_0 **first** and applies the
  row-length rule **second**. minfer's plan asked the tensor for q8_0 but skipped the rule when the
  tied-embedding retarget fired, so a one-line fix would have written q8_0 — a different divergence
  rather than none.
- In the same shape minfer's CLI printed the tensor under *"tied embedding quantized at q8_0"* while
  writing f32: the report and the bytes disagreed, because `retargeted` was filled from the tied
  branch before the preserve branch decided the type.

## Decision

**1. The row-length rule covers every quant target, and it says what llama.cpp says.**
`QuantTarget::row_len_fallback` returns the demotion for all five legacy targets (`q4_0`, `q4_1`,
`q5_0`, `q5_1`, `q8_0`) as well as the three K-quants: a 256-element K-quant demotes to a legacy
32-block type (`Q4_K → Q5_0`, `Q5_K → Q5_1`, `Q6_K → Q8_0`), and a legacy 32-block target — with no
smaller block to demote to — routes straight to `F16`, which is llama.cpp's `default: if (qk_k <= 32)
return_type = target_type;` followed by its `ncols % block != 0 → F16` check in one step. `F16` and
`F32` return `None`: a 1-element block cannot fail to divide a row, and llama.cpp gates the whole
fallback on `ggml_is_quantized`.

**2. The rule is applied to the type the tensor is *asked* for.** A demotion is a property of the
target type, so the plan resolves the asked-for type first — including the tied-embedding q8_0
retarget — and demotes from *that*, exactly as llama.cpp's mixture runs before its fallback. A
demotion is never a copy of the source type: with an unrepresentable row an f32 source is not
written f32 and a bf16 source is not written bf16.

**3. The report follows the bytes.** `preserved` now means 1-D only, for every target, and the CLI
says so. A tied embedding that was demoted is reported as a demotion, never as the retarget it did
not get.

**4. Two evidence layers, because the property has two halves.** The *planner* rule is pinned in CI
by `src/tooling/tests/quantize_row_fallback.rs`, which writes miniature GGUFs whose rows no supported
architecture has and reads the types back out of the written file. The *byte* rule is pinned by an
`#[ignore]`d F6 gate against a recorded `llama-quantize` reference, on a source derived from a
recorded fixture.

## Alternatives considered

- **Leave it as ADR-0020's recorded residual.** Rejected. ADR-0020's decision is unconditional —
  *"Every quantized target is byte-identical, per tensor, to `llama-quantize`'s output"* — and this is
  a path `minfer quantize` can be handed today (any f32, bf16 or q8_0 source). A known divergence on
  a writable path is a defect, not a documented boundary, and "no supported model reaches it" is a
  statement about the models, not about the contract.
- **Fix the report and keep the bytes.** Rejected: a quantized file is a weight file, and the report
  is not the contract. This is the same reasoning ADR-0020 used to reject a tolerance.
- **Apply the rule to the requested target instead of the asked-for type.** Rejected by the
  measurement above: for a tied `q4_0` it writes q8_0 where llama.cpp writes f16. The two rules are
  ordered, and the order is observable.
- **Record a synthetic source in the fixture manifest, with a new producer kind.** Not rejected on
  merit, but not needed for the same evidence: the reference is the artifact that must be recorded,
  and the input is a pure function of a fixture that is already recorded. The gate derives it and is
  self-checking — a derivation that drifted would no longer reproduce the recorded reference.
  Introducing a fifth `producer_kind` would have widened `scripts/check_f6_fixtures.py` for no
  additional pin.
- **Leave the K-quant second step uncovered because it was not the reported bug.** Rejected: the same
  derived source makes a row that no 32-element block divides, which no existing fixture has, so the
  `F16` endpoint of the K-quant chain is exercised by the same arm at no extra cost.

## Consequences

- One rule for every quant target, and one meaning for `preserved`. `row_len_fallback` is a complete
  transcription of llama.cpp's `tensor_type_fallback` for the types this engine can write, so the
  next target added inherits the rule instead of repeating the omission.
- The CLI text changes: the preserved line no longer mentions a row length, and the demotion line
  names the block size the row missed (`row % 32 != 0` for a legacy target, `row % 256 != 0` for a
  K-quant) instead of the hard-coded 256.
- ADR-0020's text is **not** edited: its residual is corrected by this record, and its index row
  names the corrector, per ADR-0022.
- The CI arm is a planner test; the byte arm needs the fixture cache and a `llama-quantize` binary,
  so it runs in the `#[ignore]`d real-model set, not in CI. That split is the same one ADR-0020
  already draws for the byte-parity claim.
- **Residual, unchanged:** llama.cpp also preserves some tensors *by name* in
  `tensor_allows_quantization` (its output tensor when not quantized, `ffn_gate_inp`, `ssm_conv1d`, …).
  minfer's planner models rank and row length, not names. None of those names occurs in a supported
  architecture's tensor set, so this is out of scope rather than unnoticed.

## References

- `docs/GGUF-TOOLING.md` — §2.2 (the demotion table and the CLI report), §6 (the closed gap).
- [`docs/adr/0020-a-quantized-file-is-byte-identical-or-wrong.md`](./0020-a-quantized-file-is-byte-identical-or-wrong.md)
  — the byte-identity decision this completes, and the residual it recorded.
- `src/quantize.rs` — `QuantTarget::row_len_fallback`; `src/convert.rs` — `QuantizePlan::plan`;
  `src/tooling.rs` — the CLI report.
- `src/tooling/tests/quantize_row_fallback.rs` (the miniature-source arm) and `src/quantize/tests.rs`
  (the fallback table).
- `tests/fixtures/f6-fixtures.json` + `scripts/check_f6_fixtures.py` — the recorded reference the live
  arm compares against.
- Issues: [#204](https://github.com/yusiwen/minfer/issues/204) (this rule),
  [#140](https://github.com/yusiwen/minfer/issues/140) (the K-quant half).
