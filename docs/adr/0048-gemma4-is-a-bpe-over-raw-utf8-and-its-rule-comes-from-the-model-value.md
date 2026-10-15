# 0048. gemma4 is a BPE over raw UTF-8, and its rule comes from the model value

- Status: Accepted
- Date: 2026-10-15
- Issues: #132 (S4), #575
- Amends: ADR-0045

## Context

ADR-0045 made `tokenizer.ggml.model` a dispatch and fixed the reference rule
ADR-0046/ADR-0047 follow: *the implementation that defines the format is the
reference.* `model = gemma4` is the fifth value, and it is a BPE variant that
differs from the `gpt2` engine in four ways at once:

1. **No byte-level encoding** (`byte_encode = false`, `llama-vocab.cpp:529`): the
   merge loop runs on **raw UTF-8**, not on GPT-2's byte-to-unicode alphabet.
2. **The normalizer replaces spaces with `▁`** before anything else (the
   `tokenizer.json`'s `Replace(" " → "▁")`), and `escape_whitespaces = true`
   (`llama-vocab.cpp:2218`) — the vocabulary is written that way (137 541 of its
   262 144 entries start with `▁`).
3. **The pre-tokenization rule is derived from the model value**
   (`tokenizer_pre = "gemma4"`, `llama-vocab.cpp:2132`): a gemma4 GGUF carries no
   `tokenizer.ggml.pre` key, and the rule is `[^\n]+|[\n]+` — newline runs —
   because a merge lookup never spans a newline. A piece that is **entirely
   newlines** and is itself a vocabulary entry is emitted whole
   (`llama-vocab.cpp:630`, the PR-21343 fix), which is what makes `"\n\n\n"` one
   id instead of three byte tokens.
4. **The byte fallback is `<0xXX>`** (the same SPM-style tokens SentencePiece
   files carry): the vocabulary declares all 256 with type 6.

Measured 2026-10-15 on `unsloth/gemma-4-E2B-it-GGUF` (the target model's
`gemma-4-E2B-it-UD-IQ2_M.gguf`, and the 98 MB `MTP/mtp-gemma-4-E2B-it-Q8_0.gguf`
assistant module, whose tokenizer metadata is byte-identical — 262 144 tokens,
514 906 merges, 262 144 scores, 256 type-6 byte tokens, 16 type-3 and 7 type-4
specials; verified entry by entry): `model = gemma4`, no `pre`, no
`tokenizer.ggml.byte_encode`, `add_space_prefix = 0`.

## Decision

**1. `gemma4` is accepted as the fifth engine**, and its rule is taken from the
model value: `load` sets `pre = PreTokenizer::Gemma4` and does **not** read
`tokenizer.ggml.pre`, exactly as llama.cpp derives the pre-type
(`llama-vocab.cpp:2132`). The `PreTokenizer::Gemma4` split is the newline-run
split, and `applies_nfc()` is false.

**2. Spaces are escaped to `▁` before the split, and the merge loop runs on raw
UTF-8.** `bpe_encode_raw` is the `gpt2` engine's rank-ordered merge loop without
the byte-to-unicode step, and its fallback is one `<0xXX>` byte token per byte,
which `load` verifies covers all 256 values.

**3. The newline shortcut is implemented**: an all-newline piece that is a
vocabulary entry is emitted whole.

**4. The reference is the model's own `tokenizer.json`** (BPE with
`byte_fallback: true`, the `▁` `Replace` normalizer and the `Split(" ")`
pre-tokenizer that is a no-op once the spaces are gone) **and llama.cpp's
`llama-tokenize` on the target artifact**: the two agree on **all 54** corpus
entries, and minfer agrees with them on the same corpus.

**5. The committed fixture uses the 98 MB assistant module**, because the
target's UD (dynamic-quant) file is rejected by minfer's GGUF parser — a
separate defect, filed as [#575](https://github.com/yusiwen/minfer/issues/575).
The substitution is verified, not assumed: the two files' `tokenizer.ggml.tokens`,
`merges` and `scores` arrays are byte-identical (only `token_type` and the `eos`
role differ), and the corpus is gated twice — HuggingFace on both files, llama.cpp
on the target file — with 54/54 each.

## Alternatives considered

- **Refuse `gemma4`** until the model file parses. Rejected: the tokenizer is
  independent of the tensor layout, the finding is filed, and Gemma-4 is a live
  family with hundreds of thousands of downloads.
- **Run gemma4 through the `gpt2` engine.** Rejected: the byte-to-unicode step
  would rewrite every non-ASCII byte before the merge lookup, so the `▁`-prefixed
  entries could never match — a silent, large divergence.
- **Read the rule from `tokenizer.ggml.pre`** (the F7 contract). Rejected *for this
  model only*: the key is absent by construction and llama.cpp derives the rule
  from the model value; requiring it would refuse every gemma4 file. The
  derivation is recorded here and in `TokenizerModel::Gemma4`.
- **Adopt llama.cpp's name-based type override** (`<|tool_response>`, `</s>` and
  friends are forced to CONTROL). Rejected per ADR-0043 — a special token is what
  the file declares, never what its name looks like. On this artifact llama.cpp
  prints `control-looking token: 50 '<|tool_response>' was not control-type`, i.e.
  the file itself disagrees with the override; minfer follows the file, and a
  corpus entry containing that text would tokenize as text on minfer and as a
  control on llama.cpp. That divergence is recorded, not hidden.
- **Write the fixture against the target file** and skip the artifact switch.
  Rejected: it does not parse, and a fixture whose own gate cannot load the
  artifact would be worse than a documented substitution.

## Consequences

- Five engines are live; `rwkv`, `plamo2`, `whitespace`, `hybriddna` and `none`
  are still refused by name.
- The ignored gate covers **eleven** cached models across five engines.
- `PreTokenizer::Gemma4` is the first rule whose *split* does not use the
  letter/number/symbol arms: its `number_run`/`include_marks` answers are the
  unused arms of an exhaustive match, and the split is its own function.
- **Residuals, recorded:** (a) a UD-quant artifact does not load, filed as
  [#575](https://github.com/yusiwen/minfer/issues/575); (b) llama.cpp's
  name-based type override is deliberately not implemented (ADR-0043), so a text
  containing one of those names tokenizes differently; (c) the fixture's artifact
  is the assistant module, with the byte-identity of its tokenizer arrays to the
  target model's verified and recorded in the fixture's provenance.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.0 (the engine table), §3.4,
  §4 (the gates).
- [`docs/adr/0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md`](./0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md)
  — the dispatch this amends.
- `src/tokenizer.rs` — `TokenizerModel::Gemma4`, `PreTokenizer::Gemma4`,
  `gemma4_encode`, `bpe_encode_raw`.
- `llama-vocab.cpp:529`, `:630`, `:2132` and `:2218` (llama.cpp `@050dde50c`, the
  pinned reference checkout) — the ported behaviour.
- `tests/fixtures/tokenizer/ids_gemma-4-e2b.json` — the corpus, its reference and
  the cross-check.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132) (S4),
  [#575](https://github.com/yusiwen/minfer/issues/575) (the UD quant).
