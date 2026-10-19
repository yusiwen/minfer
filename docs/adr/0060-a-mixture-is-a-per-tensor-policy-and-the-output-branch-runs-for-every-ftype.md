# 0060. A mixture is a per-tensor policy over the same encoders, and the `OUTPUT`/tied branch runs for every quantizing ftype

- Status: Accepted
- Date: 2026-10-19
- Issues: #203, #140
- Corrects: ADR-0020

## Context

ADR-0020 settled that a quantized file is byte-identical to `llama-quantize`'s or it is wrong, and
that the K-quant reference is `llama-quantize --pure`. Both still hold. Its *reason* for choosing
`--pure` was a gap, in its own words: *"the mixture planner is not implemented
([#203](https://github.com/yusiwen/minfer/issues/203), still open), so a `q4_K` file from
`llama-quantize` without `--pure` is not what this command produces."* It also recorded, as a
residual, that for legacy targets `minfer` kept the source type where llama.cpp demotes it — that
half was corrected by [ADR-0041](./0041-a-row-that-cannot-take-the-targets-block-is-demoted.md) once
the rule landed.

#203 landed the planner, and three facts it established are not derivable from the earlier record.

1. **The `OUTPUT`/tied-embedding branch is ftype-independent.** It is not part of the mixture: it
   runs for the five legacy targets too. minfer's pre-#203 rule asked for `q8_0` on a tied
   `token_embd.weight` unconditionally, which agrees with llama.cpp only when the requested type's
   block cannot divide the row. Measured 2026-10-19 on `dgxspark`: on `qwen2.5-0.5b` (hidden 896)
   llama.cpp also writes `q8_0`; on `qwen3-0.6b` (hidden 1024) `llama-quantize … q4_0` writes
   `q6_K` where `minfer` wrote `q8_0` — a byte difference in the shared embedding.
2. **Selection has to run before the row-length rule, and the branch's *own* row test decides where
   the demotion lands.** On the 0.5B the 12 `attn_v` bumped to `Q6_K` are demoted to `q8_0`
   (896 % 256 != 0), while the tied embedding is answered `q8_0` by the branch itself (896 % 256
   != 0) and is not demoted at all; on the 0.6B nothing is demoted. ADR-0041 had already fixed the
   order — mixture first, `tensor_type_fallback` second — but not this consequence of it.
3. **A mixture is a policy, not a file type.** Every tensor it writes is one of the concrete types the
   encoders already produce, so the byte-identity claim needs no new encoder; it needs the selection
   to be a pure function of the tensor's name, its row length and the model's block count.

## Decision

**At the date above, `minfer quantize --type q4_K_M`/`q5_K_M` reproduces `llama-quantize … q4_K`/`q5_K`
byte for byte on both recorded models** — 310/310 tensors on `qwen3-0.6b` and 290/290 on
`qwen2.5-0.5b`, with the 0.6B file identical whole-file as well (`sha256 a3ab8869…`, 396 704 416 B).
Four statements make that hold:

1. **Two families of target, and the bare K-quant spellings keep the uniform meaning.**
   `q4_K`/`q5_K` stay what `llama-quantize --pure` writes — the oracle the recorded
   `ref/*-q4_K.gguf` files hold — and the mixtures are the explicit `q4_K_M`/`q5_K_M`. llama.cpp's
   CLI aliases bare `q4_K` to `LLAMA_FTYPE_MOSTLY_Q4_K_M`, so this is a deliberate, documented
   spelling difference: the help text says it, and a spelling that produced a file yesterday produces
   the same bytes today.
2. **A mixture is a selection policy in [`src/quantize/mix.rs`](../../src/quantize/mix.rs), not an
   encoder.** `QuantTarget` stays total — one encoder per concrete type — and `QuantizePlan::plan`
   resolves every 2-D tensor to a concrete target before any weight is encoded: the 1-D rule, then
   `Target::select` (the `OUTPUT`/tied branch first, then the mixture's `use_more_bits` and
   fused-`qkv` bumps), then ADR-0041's row-length demotion applied to the type the tensor was
   *asked* for.
3. **The `OUTPUT`/tied branch runs for every quantizing ftype, and answers from the row length**:
   `q8_0` when the requested type's block cannot divide the row, `q6_K` when it can, and a `q8_0`
   request is its own answer. That half is not a mixture and cannot be deferred to one.
4. **`general.file_type` stays the ftype number (15/17) for both spellings**, exactly as
   `llama-quantize --pure` records it. The metadata describes the ftype, not the file's shape, so no
   reader may infer "uniform" from it.

Out of scope, and refused rather than approximated: `Q4_K_S`/`Q5_K_S`, imatrix, and the
`--tensor-type`/`--output-tensor-type` overrides.

## Alternatives considered

- **Make bare `q4_K` the mixture, as llama.cpp's CLI does.** Rejected: it silently changes what an
  existing spelling means, and the recorded `ref/*-q4_K.gguf` references — the oracle of the
  byte-parity gate — are `--pure` files. A user who typed `--type q4_K` before this change and one
  who types it after must get the same file.
- **Add the mixtures to `QuantTarget`.** Rejected: `quantize_row` matches on it exhaustively, so two
  more variants would force the encoder dispatch to answer for a *policy* — either a lie (encode at
  the base type) or a panic. A separate `Target` keeps the encoder total and puts the policy where
  its inputs (tensor name, row, block count) are.
- **Keep the old tied-embedding rule (`q8_0`).** Rejected by measurement: it matches llama.cpp only
  where the hidden size is not a multiple of 256, so the 0.6B's shared embedding came out `q8_0`
  against llama.cpp's `q6_K`.
- **Run the `OUTPUT` branch only for the mixtures.** Rejected: it is an ftype-independent rule in
  `llama_tensor_get_type_impl`, and implementing it only for `_M` would leave the legacy targets'
  claim wrong while looking finished.

## Consequences

- The F6 byte-parity gate gained a mixture arm, and a second gate compares the *picked types* tensor
  by tensor, so a planner defect and an encoder defect fail with different messages.
- Four references joined `tests/fixtures/f6-fixtures.json` — `q4_K_M` on both models, `q5_K_M` on the
  0.6B, and the 0.6B `q4_0` for the branch — each marked authoritative for the `dgxspark`
  `gcc 13.3.0 -ffp-contract=fast` build, because the K-quant search encoders are compiler-sensitive
  ([#334](https://github.com/yusiwen/minfer/issues/334),
  [#349](https://github.com/yusiwen/minfer/issues/349)).
- Two spellings now write the same `general.file_type`, so a reader cannot tell them apart from the
  metadata. That was already true of `llama-quantize --pure`, and the record says so rather than
  implying otherwise.
- The leftovers are named, not implied: `Q4_K_S`/`Q5_K_S`, imatrix and the per-tensor overrides are
  still absent and still refused by name.

## References

- [`docs/GGUF-TOOLING.md`](../GGUF-TOOLING.md) §2.2 (the targets and the demotion rule) and
  §4.2.1 (the reference procedure and the measured histograms); §6 lists what is still absent.
- [`src/quantize/mix.rs`](../../src/quantize/mix.rs) — the policy; [`src/convert.rs`](../../src/convert.rs)
  `QuantizePlan::plan` — the order; `src/tooling/tests/quantize_bounds.rs` — the two gates.
- llama.cpp's quantizer: `llama_tensor_get_type_impl`, `tensor_get_category`, `use_more_bits` and
  `tensor_type_fallback` in its own src/llama-quant.cpp (read at `458681e1d`, the build the recorded
  references come from; the 2026-10-19 references were produced by `ca3d5a3e1`, which reproduces
  those bytes).
