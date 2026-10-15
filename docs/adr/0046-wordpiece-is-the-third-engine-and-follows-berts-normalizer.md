# 0046. WordPiece is the third engine, and it follows BERT's normalizer

- Status: Accepted
- Date: 2026-10-15
- Issues: #132 (S4)
- Amends: ADR-0045

## Context

ADR-0045 made `tokenizer.ggml.model` a dispatch and landed the second engine,
named the remaining values as work that "lands with an ADR amendment (or a new
record) plus its fixture", and fixed the reference rule: *the engine that defines
the artifact's format is the reference*.

WordPiece (`model = bert`) is the third. It is what every BERT-family GGUF
carries — `bge-*`, `all-MiniLM`, `nomic-embed`, most embedding models — and like
SentencePiece it reads neither `merges` nor `scores`, and does not use
`tokenizer.ggml.pre`:

- each word is prefixed with the phantom `▁` and segmented by **greedy longest
  match** against the vocabulary (`llm_tokenizer_wpm_session::tokenize`,
  `llama-vocab.cpp:777`);
- the words come from BERT's BasicTokenizer (`preprocess`,
  `llama-vocab.cpp:827`): whitespace splits, control characters are dropped, and
  punctuation, an ASCII symbol or a Chinese character becomes a word of its own;
- two BertNormalizer options gate text rewriting:
  `tokenizer.ggml.normalizer.lowercase` and `.strip_accents`, both of which
  default to **`true`** (`llama-vocab.h:83`) with `strip_accents` following
  `lowercase` unless the file says otherwise (`llama-vocab.cpp:2635`).

Measured 2026-10-15 on `ggml-org/bge-small-en-v1.5-Q8_0-GGUF` (36 685 152 B):
`model = bert`, `pre = jina-v2-en` (irrelevant, and not implemented), 30 522
tokens of which 5 are type 3 (`[PAD]`/`[UNK]`/`[CLS]`/`[SEP]`/`[MASK]`), no
`merges`, no `scores`, no normalizer keys (so llama.cpp's defaults apply:
lowercase **on**, strip accents **on**), `unknown_token_id = 100`.

The GGUF's vocabulary is HuggingFace's `BAAI/bge-small-en-v1.5` vocabulary
**rewritten in place**: every word-initial entry gains a `▁` prefix, every `##`
continuation loses it, and the first 999 special/unused entries are untouched —
an id-preserving rewrite (verified entry by entry on 2026-10-15, 30 522/30 522).

## Decision

**1. `bert` is accepted as the third engine.** `TokenizerModel::Bert` owns its own
prerequisites: the `<unk>` role must be declared and in range (the engine's only
answer for a word it cannot segment), and `max_token_len` is the longest
vocabulary entry in bytes. It reads no `pre`, no merges and no scores, and
`Tokenizer::pre` stays `None` for it.

**2. The normalizer is BERT's, implemented from the existing NFC tables.**
`nfc::nfd` (shared with the NFC path: one UAX #15 decomposition, two forms) then
per-code-point: whitespace ends a word, `\p{C}` is dropped, **`\p{Mn}` is
dropped** when `strip_accents` is on, the code point is lowercased when
`lowercase` is on (the simple 1:1 mapping), and `\p{P}`, an ASCII symbol or a
Chinese character becomes its own word.

**3. The reference is HuggingFace `tokenizers` on the model's own
`tokenizer.json`** — WordPiece's definition — because the GGUF's vocabulary is
that file's vocabulary with the same ids (the rewrite above), so the two are
comparable id for id. The fixture records that reference and the equivalence.

**4. Two llama.cpp divergences are recorded rather than copied.** llama.cpp's
"NFD" (`unicode_cpts_normalize_nfd`, `unicode.cpp:1117`) maps each code point to
the **single first** code point of its canonical decomposition, and its accent
drop tests `\p{M}` (every mark) where BERT tests `Mn`:

| input | HF `tokenizers` (this engine) | llama.cpp `llama-tokenize` |
|---|---|---|
| `한국어 텍스트` | `[1469, 30006, 30021, 29991, 30014, 30020, 29999, 30008, 1467, 30009, 30020, 29997, 30017, 30003, 30017]` | `[1469, 29991, 29999, 1467, 29997, 30003]` |
| `क्षि नमस्ते` | `[1315, 29873, 29877, 1327, 29867, 29874, 29859]` | `[1315, 29873, 1327, 29867, 29874, 29859]` |

Both are lossy: llama.cpp turns a Hangul syllable into its leading jamo (the
vowel and coda disappear) and drops the spacing mark `Mc` (Devanagari U+093F)
along with the non-spacing ones. minfer decomposes fully and drops `Mn` only,
which is what BERT and every HuggingFace consumer expect.

**5. The corpus gate covers the engine, and the divergence is measured, not
assumed.** The committed fixture's 54 entries match HF byte for byte; the two
llama.cpp witnesses above are in the ADR and in the PR, and the unit tests pin
the same behaviour on a synthetic vocabulary without a cached model.

## Alternatives considered

- **Copy llama.cpp exactly** (1:1 decomposition, `\p{M}`). Rejected: it *loses*
  text — `한국어` becomes `ᄒᄀᄋ` — so minfer would return ids no HuggingFace
  user would predict, for the family whose reference files are HuggingFace's. The
  divergence is recorded instead, and the engine follows the algorithm's owner.
- **Use `llama-tokenize` as the reference** (as the SPM engine does). Rejected for
  the same reason: it is the only tool that runs a GGUF, but here it is a port
  with a known approximation, and the id-preserving vocabulary rewrite is what
  makes the definition's own implementation comparable.
- **Keep the NFC normalizer for WordPiece.** Rejected: `applies_nfc()` stays
  false — BERT's normalizer is NFD + mark dropping, not NFC — and the property the
  BPE/SPM families assert (composed ≠ decomposed) is the **opposite** here. The
  real-model gate branches on the engine accordingly.
- **Refuse a file that declares `strip_accents`** until NFD existed. Not needed:
  `nfc.rs` already carried the decomposition and combining-class tables (the NFC
  path uses them in the other direction), so `nfd()` is a few lines, and the
  differential corpus now gates it against `unicodedata.normalize("NFD", …)` for
  all 5 103 vectors.
- **Implement `clean_text`** (BERT's removal of control characters and the
  collapsing of whitespace). Rejected for now: `preprocess` already drops control
  characters, and the fixture's artifact cannot distinguish the rest — recorded as
  a residual rather than written blind.

## Consequences

- A `model = bert` GGUF loads and tokenizes; three engines are live, and every
  other value is still refused by name (`t5`, `rwkv`, `plamo2`, `gemma4`,
  `whitespace`, `hybriddna`, `none`).
- `NormalizerOptions` defaults to `true`/`true` — llama.cpp's struct defaults, and
  therefore what a converted BERT file that carries neither key gets.
- The NFC differential corpus carries NFD expectations too, and
  `nfd_matches_cpython_on_the_committed_corpus` gates them (5 103 vectors).
- The real-model gate branches on the engine for its normalization property
  instead of asking only `applies_nfc()`, so a WordPiece model asserts the fold
  (composed == decomposed) that BERT declares.
- **Residual, recorded:** `clean_text` is not implemented (no artifact in the gate
  distinguishes it); llama.cpp remains the reference for SPM while HuggingFace is
  the reference for WordPiece, and each fixture records which.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.0 (the engine table), §3.4
  (what remains), §4 (the gates).
- [`docs/adr/0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md`](./0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md)
  — the dispatch this amends with a third engine.
- `src/tokenizer.rs` — `TokenizerModel::Bert`, `wpm_encode`, `wpm_preprocess`,
  `NormalizerOptions`; `src/tokenizer/nfc.rs` — `nfd`.
- `llama-vocab.cpp:777` and `llama-vocab.cpp:827` (llama.cpp `@050dde50c`, the
  pinned reference checkout) — the ported session and normalizer.
- `tests/fixtures/tokenizer/ids_bge-small-en-v1.5.json` — the corpus and its
  reference.
- Issue [#132](https://github.com/yusiwen/minfer/issues/132) — S4.
