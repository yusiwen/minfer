# 0045. The tokenizer engine is a dispatch on `tokenizer.ggml.model`

- Status: Accepted
- Date: 2026-10-15
- Issues: #132 (S4)

## Context

F7 built the tokenizer around one engine: byte-level BPE, selected by
`tokenizer.ggml.pre` (a pre-tokenization rule), driven by `tokenizer.ggml.merges` (merge
ranks) and `tokenizer.ggml.token_type` (the specials). Every GGUF whose
`tokenizer.ggml.model` was not `gpt2` was refused by name, deliberately: the engine could
not reproduce it, and approximating it with the wrong algorithm is worse than refusing.

That refusal covers most of the llama.cpp catalogue. `tokenizer.ggml.model = llama` is
SentencePiece — Llama-1/2, Mistral v0.1, TinyLlama, Phi-2 and a long tail of models — and
it is not BPE-with-ranks at all:

- the split order comes from **`tokenizer.ggml.scores`**, not from merge ranks. There is no
  `tokenizer.ggml.merges` array in an SPM GGUF;
- spaces are escaped to `U+2581` (`▁`) *before* the split, and the first segment is
  prefixed with one (the "add a dummy prefix" behaviour);
- unknown pieces fall back to declared `<0xXX>` byte tokens (type 6), not to a byte-level
  character map;
- `tokenizer.ggml.pre` is **irrelevant**: llama.cpp fixes the SPM flags from the model type
  (`add_space_prefix = true`, `clean_spaces = false`, `llama-vocab.cpp:2409`) and never
  pre-splits an SPM text; Llama-2-era GGUFs do not even carry the key.

Measured on `second-state/TinyLlama-1.1B-Chat-v1.0-GGUF` Q4_K_M (2026-10-15):
`tokenizer.ggml.model = llama`, **no** `pre`, no `merges`, 32 000 tokens of which 256 are
type 6 (`<0x00>`…`<0xFF>`), `bos` 1 `<s>`, `eos` 2 `</s>`, `unknown` 0 `<unk>`. minfer
refused the file outright; llama.cpp's `llama-tokenize` returns
`Hello, world!` → `[15043, 29892, 3186, 29991]`.

## Decision

**1. The engine is a dispatch, not a hint.** `TokenizerModel::from_gguf` reads
`tokenizer.ggml.model` and answers `Gpt2` (the value, or a missing key — the F7 behaviour)
or `Spm` (`llama`). Every other value is still a refusal **naming the value** and the two
engines that exist.

**2. `llama` is implemented as llama.cpp's SentencePiece session**, ported faithfully
(`llm_tokenizer_spm_session`, `llama-vocab.cpp:114`): symbols start as UTF-8 characters,
every adjacent pair whose concatenation is a vocabulary entry is queued with that entry's
**score**, the highest score merges first, a pair whose sides changed since it was queued is
stale and skipped, and the final walk emits a vocabulary entry where it can, re-splits
through the recorded pair where it cannot, and otherwise emits one `<0xXX>` byte token per
byte.

**3. The engine owns its prerequisites.** `Tokenizer::load` validates per engine: `gpt2`
requires a supported `pre`, non-empty `merges` and a byte-level-complete vocabulary; `llama`
requires a non-degenerate `scores` array and all 256 declared byte tokens, and requires
nothing of `pre`. A missing prerequisite is a load error, never a lossy fallback.

**4. The model-type defaults are the GGUF's, with the key as an override.**
`add_space_prefix` defaults to `true` for SPM (llama.cpp's model-type default) and
`tokenizer.ggml.add_space_prefix` overrides it. A special token resets the "previous segment
was special" state, so the segment after one is prefixed again.

**5. The reference for an SPM vocabulary is llama.cpp on the same artifact**, and it is
recorded in the fixture's `provenance.reference`. HuggingFace's `tokenizer.json` for
TinyLlama is a *BPE* rendering of the same vocabulary with its own normalizer — a different
code path, not a second opinion on this one.

**6. `Tokenizer::pre` becomes `Option<PreTokenizer>`.** `None` means the engine has no
pre-tokenization rule (SPM), and with it `applies_nfc()` is `false` — the SPM families this
engine accepts declare no normalizer, and llama.cpp applies none.

**7. The ignored real-model gate gains one model per engine**, and the fixture names its
family (`tokenizer_model`) so a corpus written for one engine cannot silently be run through
the other.

## Alternatives considered

- **Map SPM onto the existing BPE engine** — treat `scores` as ranks, skip the `▁` escape.
  Rejected: the two orders disagree, and the escape is not cosmetic (a `▁`-prefixed text is
  what the vocabulary was built over). The divergence would be silent and large.
- **Port HuggingFace's unigram Viterbi instead**, since SPM vocabularies are usually trained
  as unigram models. Rejected: the artifact does not carry the lattice, and the engine that
  is validated against it — llama.cpp — runs the score-merge session; matching a third
  implementation instead would make the gate unmeasurable against the tool that defines the
  format.
- **Require `tokenizer.ggml.pre` for SPM too**, or refuse a missing one. Rejected: the key
  does not participate in SPM tokenization, and requiring it would refuse every Llama-2-era
  GGUF for a reason that cannot change the output.
- **Accept the remaining model values in the same increment** (`bert`, `t5`, `rwkv`,
  `plamo2`, `whitespace`, `hybriddna`, `none`, `gemma4`). Rejected: each is a different
  algorithm (WordPiece, unigram with its own normalization, RWKV, …) with its own
  vocabulary shape and its own fixture; landing them without their own gates is exactly the
  approximation this dispatch refuses.
- **Detect the engine from the metadata shape** (`scores` present ⇒ SPM, `merges` ⇒ BPE).
  Rejected: the file states its engine; inferring it from which arrays happen to be non-empty
  is a heuristic over a declaration, and the two arrays coexist in some conversions.

## Consequences

- A `model = llama` GGUF loads and tokenizes instead of being refused; the corpus gate covers
  it against `llama-tokenize` byte for byte (54 entries, its own specials, decomposed entries
  included), and two CI-runnable unit tests cover the score merge, the `▁` prefix and the
  byte fallback without a cached model.
- The load path has two validated branches, and `pre` is now optional — every consumer that
  asked the tokenizer for its rule (`applies_nfc`, the fixtures' `pre` field) goes through
  the tokenizer instead.
- The design record's §3 describes both engines: a `pre` value answers the BPE questions, and
  a `model` value answers which engine asks them.
- **Residual, recorded:** the values that remain refused are refused *by name*, and the
  refusal lists what exists. Adding one means an ADR amendment (or a new record) plus its
  fixture, not a new branch in a match.
- SPM's `decode` unescapes `▁` to a space and a byte token to its byte; the leading space the
  tokenizer adds is part of the tokenization, exactly as in llama.cpp (`decode(encode(x))` is
  `" " + x` when the prefix is on).

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3 (both engines), §4 (the gates).
- `src/tokenizer.rs` — `TokenizerModel`, `Tokenizer::spm_encode`, `Tokenizer::resegment`.
- `llama-vocab.cpp:114` (llama.cpp `@050dde50c`, the pinned reference checkout) — the ported session.
- `tests/fixtures/tokenizer/ids_tinyllama-1.1b.json` — the SPM corpus and its reference.
- Issue [#132](https://github.com/yusiwen/minfer/issues/132) — S4.
