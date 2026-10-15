# 0044. The input normalizer follows the pre-tokenizer family's declaration

- Status: Accepted
- Date: 2026-10-15
- Issues: #564
- Corrects: ADR-0042

## Context

ADR-0042 decided that the tokenizer applies NFC to every text it turns into ids, on the
premise that the reference `tokenizer.json` files declare one. That premise was verified for
the three files the engine gated against at the time — `Qwen/Qwen3-0.6B@c1899de2`,
`Qwen/Qwen2.5-0.5B-Instruct@main` and `unsloth/Qwen3.5-0.8B@23c69c53`, all
`"normalizer": {"type": "NFC"}` — and the decision was written for the whole tokenizer
because **the GGUF carries no normalizer field**.

S3's third family breaks it. `unsloth/Llama-3.2-1B-Instruct@5a8abab4` declares
`"normalizer": null` (verified 2026-10-15; so do `unsloth/Llama-3.1-8B-Instruct` and
`unsloth/Llama-3.2-3B-Instruct`), and llama.cpp applies no normalizer for byte-level BPE
either. For `llama-bpe` minfer therefore normalized where **neither** reference does — the
same class of silent divergence S1 removed for the Qwen families, now in the opposite
direction.

Measured on `unsloth/Llama-3.2-1B-Instruct-GGUF` Q4_K_M (2026-10-15), `cafe` + U+0301:

| HuggingFace `tokenizers` 0.23.3 | llama.cpp `llama-tokenize` | minfer (unconditional NFC) |
|---|---|---|
| `[936, 1897, 54939]` | `[936, 1897, 54939]` | `[936, 59958]` |

The only per-model signal the loaded artifact gives is `tokenizer.ggml.pre`, which is
declared per family — and each family's reference `tokenizer.json` is exactly where the
normalizer is declared in the first place.

## Decision

**1. The normalizer is a property of the pre-tokenizer rule.** `PreTokenizer::applies_nfc()`
answers it — `qwen2`/`qwen35` → true, `llama-bpe` → false — and `Tokenizer::encode` /
`Tokenizer::encode_raw` go through one `normalized()` seam that consults it. A rule that
declares none hands the text to BPE exactly as it arrived.

**2. The mapping is verified, dated and recorded per family**, in the `applies_nfc` doc
comment and in the design record's rule table (§3.1), against the reference
`tokenizer.json` of the models that carry that `pre` value.

**3. A family that declares no normalizer gets decomposed fixture entries**, whose expected
ids are the reference's *decomposed* ids — so the property is gated in the same corpus walk
as everything else rather than by a special case.

**4. The GGUF's silence is a recorded residual.** A model whose `tokenizer.json` disagrees
with its rule's answer cannot be distinguished from one that agrees; the engine follows the
family because that is the declaration the artifact carries (see the alternatives).

## Alternatives considered

- **Keep the unconditional normalizer** (the status quo). Rejected: it makes `llama-bpe`
  differ from both references on decomposed input — the exact failure class S1 exists to
  remove, and a fixture corpus that is NFC-normalized would hide it.
- **Decide per model from the vocabulary** — normalize only when the composed form is in the
  vocabulary, or when the decomposed form is not. Rejected: a heuristic over the vocabulary
  replaces a declaration, and the two answers can both look plausible for the same text; the
  reference files state the answer outright.
- **Detect non-NFC input and refuse.** Rejected in ADR-0042 and still rejected: it needs the
  same Unicode tables, rejects input a consumer can legitimately send (macOS text is
  frequently NFD), and leaves the engine disagreeing with the model instead of agreeing.
- **Key it on the model identifier or architecture.** Rejected: minfer reads the GGUF, whose
  tokenizer metadata carries no model id, and the architecture (`llama` for a Llama-3.2 GGUF)
  is not the tokenizer's family — two architectures share a tokenizer.
- **A CLI flag or environment variable.** Rejected as the mechanism: a default that is wrong
  for a family and needs a flag to be right is not a contract. It remains available as a
  future escape hatch if a real model contradicts its family's declaration.

## Consequences

- For `llama-bpe`, minfer now returns what both references return for decomposed input
  (`[936, 1897, 54939]` for `cafe` + U+0301); for `qwen2`/`qwen35` nothing changes, and the
  decomposed-equals-composed property remains the control in the ignored real-model gate.
- The fixture corpus for a non-normalizing family carries the decomposed entries, so the
  reference ids — not a property assertion alone — are what a wrong answer fails.
- The rule table is the one home for both questions a `pre` value answers: which pattern, and
  whether the input is normalized. A new rule answers both when it lands.
- ADR-0042's text is **not** edited: its unconditional wording is corrected by this record,
  and its index row names the corrector (ADR-0022). Its mechanisms (generated tables, no new
  dependency, the CPython differential corpus, the conservative quick check) stand unchanged.
- **Residual, recorded:** a GGUF cannot express a per-model normalizer override. If a model
  is found whose `tokenizer.json` contradicts its family, that is a new decision (a GGUF
  field to honour, or a per-model table), not a silent heuristic.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.1 (the rule table, now with the
  normalizer column), §3.6 (the mechanism), §4 (the gates).
- [`docs/adr/0042-the-tokenizer-applies-the-models-declared-nfc-normalizer.md`](./0042-the-tokenizer-applies-the-models-declared-nfc-normalizer.md)
  — the record whose unconditional scope this corrects.
- `src/tokenizer.rs` — `PreTokenizer::applies_nfc`, `Tokenizer::normalized`.
- `src/tokenizer/nfc.rs` — the normalization itself, unchanged.
- `tests/fixtures/tokenizer/ids_llama-3.2-1b-instruct.json` — the decomposed corpus entries.
- Issues: [#564](https://github.com/yusiwen/minfer/issues/564) (this fix),
  [#132](https://github.com/yusiwen/minfer/issues/132) (S1, which introduced the normalizer;
  S3, whose rule exposed this).
