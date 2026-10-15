# 0047. The unigram engine reads the file's charsmap, and SentencePiece is its reference

- Status: Accepted
- Date: 2026-10-15
- Issues: #132 (S4)
- Amends: ADR-0045

## Context

ADR-0045 made `tokenizer.ggml.model` a dispatch; ADR-0046 added the third engine
and fixed the rule this record follows: *the implementation that defines the
format is the reference, even when it is not llama.cpp.*

`model = t5` is the fourth value and the third algorithm
(`LLAMA_VOCAB_TYPE_UGM`, `llm_tokenizer_ugm_session`, `llama-vocab.cpp:963`). A T5
GGUF is not segmented by merges, scores alone, or longest match. Measured
2026-10-15 on `Felladrin/gguf-flan-t5-small` Q8_0 (113 709 824 B):
`model = t5`, `pre = default`, 32 128 tokens, a full `scores` array, **no**
`merges`, `add_space_prefix = 1`, `remove_extra_whitespaces = 1`, eos 1 / unk 2 /
pad 0, and a **237 539-byte `tokenizer.ggml.precompiled_charsmap`**.

That charsmap is the pipeline's normalizer: the same XOR-compressed compact
double array (XCDA) SentencePiece compiles from its `NormalizerSpec`, carried in
the GGUF as bytes. An engine that ignores it tokenizes different text than the
model was trained on — and the whitespace flags around it decide where `▁` goes.

## Decision

**1. `t5` is accepted as the fourth engine.** `TokenizerModel::Ugm` owns its own
prerequisites: a non-degenerate `scores` array (the Viterbi's ordering), the
declared unknown id, and the trie over NORMAL / USER_DEFINED / UNUSED entries.
`pre`, merges and byte tokens do not participate.

**2. The precompiled charsmap is decoded and applied, not skipped.** The loader
reads the byte array, checks the four-byte XCDA blob size against the array
length, and splits it into the packed array and the NUL-terminated replacement
strings. `ugm_normalize_prefix` walks it exactly as llama.cpp does — `BASE[s] ^ c`
per input byte, LCHECK to confirm the edge, LEAF to mark a replacement — and falls
back to one unchanged code point when nothing matches. A corrupt table (an index
outside the array, an offset outside the replacements) **stops the match** rather
than panicking; `encode` cannot return an error and the loader has already
validated the header.

**3. The whitespace rules are the file's.** `add_space_prefix` (default `false`)
prepends one `▁` to text that does not start with whitespace, and
`remove_extra_whitespaces` collapses a run of whitespace into one `▁` **attached
to the following word** — which is why a trailing run produces no `▁` at all.
`treat_whitespace_as_suffix` is always `false` in the pinned reference, so the
suffix form is deliberately not implemented.

**4. The Viterbi is llama.cpp's, and its unknown transition is
`min_score - 10`.** At every code-point boundary every token that starts there is
a transition; when no token covers the whole code point the unknown token is one
too, at ten below the worst normal score. The walk back merges a run of
consecutive unknowns into a single token.

**5. The reference is the model's own SentencePiece model**
(`google/flan-t5-small/spiece.model` run by the `sentencepiece` package) — the
definition of this pipeline, and an **independent** implementation of both the
charsmap lookup and the normalizer. It agrees with llama.cpp's `llama-tokenize`
on **all 54** corpus entries, so here the port follows the definition exactly.
HuggingFace's `tokenizer.json` for the same model is *not* the reference, and that
is measured: its Metaspace pre-tokenizer does not implement
`remove_extra_whitespaces`, so it keeps a `▁` for trailing or all-whitespace input
where SentencePiece and llama.cpp drop it (**11 of 54** entries, every one of them
whitespace: `"trailing spaces  "` → `[5032, 53, 4856, 3]` on HF against
`[5032, 53, 4856]` from SentencePiece).

**6. The fixture's ids are the GGUF's.** The GGUF's first 32 000 pieces are
HuggingFace's verbatim; the last 128 differ by construction (the converter writes
`[PAD32000]`… where the tokenizer.json has `<extra_id_*>`). The corpus therefore
carries no `<extra_id_N>`, stays inside the shared range, and asserts the declared
specials separately — the same shape the WordPiece fixture uses.

## Alternatives considered

- **HuggingFace `tokenizer.json` as the reference.** Rejected, measured: its
  Metaspace rendering loses the `remove_extra_whitespaces` behaviour the artifact
  declares, so it would have forced minfer to keep whitespace the model does not
  see. The divergence is recorded instead, per entry.
- **Skip the charsmap** (treat the text as-is). Rejected: it is the normalizer the
  file carries; the fixture's decomposed entries (`café` and `cafe` + U+0301 reach
  the same ids) exist precisely because the charsmap folds them.
- **Refuse a GGUF that carries a charsmap.** Rejected: every T5-era file has one,
  so the engine would be a refusal with extra steps.
- **Validate the charsmap deeply at load** (every node reachable, every
  replacement NUL-terminated). Rejected for now: `load` checks the header and the
  blob bounds, and the walk cannot leave the array; a deeper check would be a
  parallel implementation of the format, recorded as a residual rather than
  written blind.
- **Implement `treat_whitespace_as_suffix`.** Not needed: the pinned reference
  never sets it (no GGUF key reaches it), so there is no artifact to gate it
  against.

## Consequences

- A `model = t5` GGUF loads and tokenizes; four engines are live, and `rwkv`,
  `plamo2`, `gemma4`, `whitespace`, `hybriddna` and `none` are still refused by
  name.
- The tokenizer gains one portable component — an XCDA decoder — with the
  hand-built one-mapping table in the unit tests as its pure witness, and the real
  charsmap (237 kB) exercised by the ignored gate.
- The real-model gate's normalization property is now three-way: NFC for the
  `qwen*` families, `BertNormalizer` for WordPiece, and *the charsmap folds both
  spellings* for `t5` — each asserted by the engine's own rule rather than by one
  assumption.
- The ignored gate covers **ten** cached models and four engines.
- **Residual, recorded:** `treat_whitespace_as_suffix` is not implemented (no
  artifact sets it); a corrupt charsmap degrades to "stop matching" rather than a
  refusal; the fixture's corpus excludes `<extra_id_N>` because the converter's
  tail padding makes those 128 ids artifact-specific.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.0 (the engine table), §3.4,
  §4 (the gates).
- [`docs/adr/0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md`](./0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md)
  — the dispatch this amends.
- `src/tokenizer.rs` — `TokenizerModel::Ugm`, `ugm_encode`, `ugm_normalize`,
  `ugm_normalize_prefix`, `VocabTrie`.
- `llama-vocab.cpp:963` and `:1071` (llama.cpp `@050dde50c`, the pinned reference
  checkout) — the ported session and normalizer.
- `tests/fixtures/tokenizer/ids_flan-t5-small.json` — the corpus and its reference.
- Issue [#132](https://github.com/yusiwen/minfer/issues/132) — S4.
