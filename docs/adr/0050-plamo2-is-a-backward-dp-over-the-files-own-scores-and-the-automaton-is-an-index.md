# 0050. plamo2 is a backward DP over the file's own scores, and llama.cpp's automaton is an index

- Status: Accepted
- Date: 2026-10-16
- Issues: #132 (S4)
- Amends: ADR-0045

## Context

ADR-0045 made `tokenizer.ggml.model` a dispatch; ADR-0046–ADR-0049 added four
engines and fixed the reference rule: *the implementation that defines the format is
the reference.* `model = plamo2` is the seventh value, and the first whose engine is
**llama.cpp's own construction** rather than a format an ecosystem defines:

- the vocabulary carries `tokenizer.ggml.scores` (100 032 entries here) and **no**
  merges — the file says how strongly each piece should be preferred;
- encoding is a **backward dynamic program over codepoints**, from the end of the
  text to the start, maximizing the sum of those scores
  (`llm_tokenizer_plamo2`, `llama-vocab.cpp:1360-1592`);
- the fallback for a codepoint no piece covers is **one `<0xXX>` byte token per
  UTF-8 byte** — never `<unk>`, and never a dropped character;
- a leading BOM is stripped, and the `pre` key does **not** participate (llama.cpp
  sets the pre-type from the model value; this artifact declares `pre = "default"`,
  which is not one of the four rules minfer accepts).

The interesting question was *how* llama.cpp selects candidates: it builds a suffix
automaton — 953 292 table rows over 193 510 token suffixes on this artifact — whose
state indexes the candidate set by the text's **suffix**, so a piece that locally
matches can in principle be unavailable when the state collapses.

## Decision

**1. The engine is the DP, not the automaton.** `plamo2_encode` walks the
vocabulary trie from each position (longest first), keeps the candidate that lowers
`scores[i]`, and takes the one-codepoint fallback row when nothing lowers it. The
semantics that matter are pinned explicitly:

| element | value / rule |
|---|---|
| objective | minimize `Σ -(score)`, i.e. maximize the summed score |
| transition | `s = scores[i + piece_length] - round(score × 1e4)` |
| tie | a **strict** `<`, so the first (longest) candidate wins |
| unusable score | non-finite → `INVALID_SCORE` (-20 000 000), skipped |
| fallback row | one codepoint, `UNKNOWN_SCORE` (-10 000 000) |
| fallback output | one `<0xXX>` byte token per UTF-8 byte (all 256 required at load) |
| BOM | a leading U+FEFF is stripped |

**2. The decision is measured, not assumed.** ADR-0046–ADR-0049 established the
reference rule; here the *reference is llama.cpp itself*, so the port was
cross-checked against a literal Python port of `llm_tokenizer_plamo2` (the
automaton included) and against `llama-tokenize` on the artifact:

| comparison | result |
|---|---|
| literal automaton port vs `llama-tokenize`, 54-entry corpus | **54 / 54** |
| automaton port vs this DP, 54-entry corpus | **54 / 54** |
| the two on 160 000 random strings (lengths 1–8) | **0 divergences** |
| the two on 8 constructed **state-collapse** cases (a real piece followed by a codepoint absent from all 100 032 tokens) | **8 / 8 identical** |

The fourth row is the one that makes this safe rather than corpus-lucky: random text
rarely reaches the collapsed state, so it was forced on purpose (U+1F600, U+E0001,
U+1D54F, U+0FD0, U+2E80 — each verified absent from the vocabulary) and the two
still agreed.

**3. The reference for the fixture is `llama-tokenize` on the artifact.** The engine
*is* llama.cpp's construction over the file's tokens and scores, so there is no
independent definition to prefer (as SentencePiece and the model's own tokenizer
were for T5 and RVKV).

**4. The committed artifact is an IQ1_S quant.** `mitmul/plamo-2-translate-GGUF`'s
`plamo-2-translate_IQ1_S.gguf` (2 714 053 408 B) is the smallest in the family and
is **not** a quant minfer can run; the tokenizer gate only needs the file to
*parse*, which #575's accounting fix made possible for exactly this class of file.
The fixture's provenance records that, so the artifact is not mistaken for a
runnable model.

## Alternatives considered

- **Port the suffix automaton literally.** Rejected as the landing shape: 953 292
  table rows and a 64-bit `(codepoint, suffix id)` map buy candidate-set indexing
  whose observable behaviour the direct DP reproduces on every input measured. The
  automaton remains the fallback design if a divergence ever appears.
- **Use `sentencepiece`'s Python model as the reference.** Rejected: the model's
  own runtime is not the thing that defines this engine — llama.cpp's DP over the
  file's `scores` is, and the two are free to differ in exactly the place this
  engine exists (tie-breaking and the fallback).
- **Treat the fallback as `<unk>`.** Rejected: the whole point of the engine's
  byte table is that no input is unrepresentable, and the artifact declares all 256.
- **Longest-match greedy** (what a BPE-style reading of the vocabulary suggests).
  Rejected: the scores decide, and a unit test pins a case where splitting beats the
  longer piece.
- **Require `pre` to be one of the four accepted rules.** Rejected: this artifact
  declares `"default"`, and llama.cpp ignores the key for this engine — refusing it
  would refuse every plamo2 file.

## Consequences

- Seven engines are live; `whitespace`, `hybriddna` and `none` are still refused by
  name (and `hybriddna` has no GGUF to gate against, `whitespace` no converter).
- The ignored id gate covers **thirteen** cached models across seven engines.
- `Tokenizer::id_to_score` gains its **first production reader**, so the
  `#[allow(dead_code)]` allowance that [#244] carried is removed. The dead-code
  oracle stays green at 41 entries (0 additions, 0 removals) — the field was
  already declared, not annotated.
- **Residual, recorded:** the DP is *observationally* equal to the automaton on the
  corpus, 160 000 random strings and the eight collapse witnesses — not proven equal
  in general. The gate is those three sets, so a divergence has somewhere to surface;
  the probe scripts and this ADR are the reproduction.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.0 (the engine table), §3.4,
  §4 (the gates).
- [`docs/adr/0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md`](./0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md)
  — the dispatch this amends.
- `src/tokenizer.rs` — `TokenizerModel::Plamo2`, `plamo2_encode`,
  `plamo2_score_int`, `VocabTrie::step`/`terminal`.
- `llama-vocab.cpp:1360-1592` (llama.cpp `@050dde50c`, the pinned reference
  checkout) — the ported behaviour.
- `tests/fixtures/tokenizer/ids_plamo-2-translate.json` — the corpus, its reference
  and the artifact note.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132) (S4),
  [#244](https://github.com/yusiwen/minfer/issues/244) (the field's allowance),
  [#575](https://github.com/yusiwen/minfer/issues/575) (the accounting fix that lets
  this artifact parse).
