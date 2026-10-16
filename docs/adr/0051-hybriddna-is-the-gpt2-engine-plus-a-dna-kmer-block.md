# 0051. hybriddna is the gpt2 engine plus a DNA k-mer block, and its vocabulary is escaped in place

- Status: Accepted
- Date: 2026-10-16
- Issues: #132 (S4)
- Amends: ADR-0045

## Context

ADR-0045 made `tokenizer.ggml.model` a dispatch; ADR-0046–ADR-0050 added five engines.
`model = hybriddna` is the eighth value, and it is **not a new algorithm**: llama.cpp maps
it to `LLAMA_VOCAB_TYPE_BPE` alongside `gpt2` (`llama-vocab.cpp:1986`) and only swaps the
*session* (`llama-vocab.cpp:3461`). What it adds is a single rule, for Carbon-3B
("HybridDNATokenizer", llama.cpp #23410):

- the vocabulary carries `<dna>`, `</dna>` and `<oov>`, and 4⁶ = 4096 six-base DNA k-mers
  whose spelling is suffixed with a **reserved marker U+E000** — that suffix is what keeps
  the k-mer `CCCCCC` distinct from the base vocabulary's BPE token `CCCCCC`;
- text inside `<dna>`…`</dna>` is **not** BPE'd: ASCII letters are upper-cased, the content
  is cut into 6-**byte** chunks, each chunk is looked up as a *marked* k-mer, a chunk that
  is not in the vocabulary (a non-ACGT base, a split multi-byte character) becomes `<oov>`,
  and a tail shorter than six bytes is padded with `A` before the lookup;
- text outside the tags is the plain `gpt2` engine, `pre` rule included;
- if any of the three tags is missing, llama.cpp falls back to plain BPE for everything;
- at load, llama.cpp **erases** the marker from `id_to_token` (keeping the marked spelling in
  the token→id map), so detokenization yields the bare DNA sequence.

Two claims of mine had to be corrected before this slice could be written, and the
correction is recorded on the issue: `hybriddna` **does** have a public artifact
(`HuggingFaceBio/Carbon-3B-GGUF`, 211 531 downloads — metadata probed here:
`model = hybriddna`, `general.architecture = llama`, 155 776 tokens, 151 387 merges,
`pre = qwen2`, the three tags, exactly 4096 marked k-mers), and the "no converter emits it"
claim about `whitespace` was too strong (the value exists for Jina models; no artifact
carrying it was found, and the public Jina GGUF this slice probed uses `gpt2` with
`pre = jina-v1-en`).

## Decision

**1. `hybriddna` is accepted and shares the `gpt2` load path.** `TokenizerModel::HybridDna`
resolves from the model key, and the loader's `Gpt2 | HybridDna` arm does the same work for
both (the `pre` rule through `PreTokenizer::from_gguf`, a non-empty merge table, and the
byte-level completeness check). The hybriddna-specific load step then:

- looks up the three tag ids; if all three exist, builds the k-mer table from `id_to_token`
  (bare k-mer → id) and **erases the U+E000 suffix from `id_to_token`**, leaving `vocab`
  (the token→id map) marked, exactly as llama.cpp does — the marked spelling is the lookup
  key, the bare one is what detokenization emits;
- if a tag is missing, leaves the engine as plain BPE.

**2. The session is a thin wrapper.** `hybriddna_encode` walks the `<dna>`…`</dna>` blocks
and delegates everything else to `encode_bpe`; `hybriddna_kmers` is the k-mer rule above,
working on **bytes** (so an ASCII-only upper-casing, a 6-byte chunk, and a byte-wise `A`
pad — the places where a Unicode-aware `to_uppercase` would change the chunking). A block
with no closing tag emits the opening tag and its k-mers and stops, without a closing tag —
llama.cpp's shape.

**3. The reference is llama.cpp on the artifact.** Carbon-3B's repository ships the
tokenizer's Python module and a `vocab.json` but no independent, runnable converter, and
the engine is llama.cpp's own construction over a marked vocabulary — so llama.cpp is the
definition, as it was for plamo2 (ADR-0050).

**4. The fixture gates the DNA rule explicitly.** The shared 54-entry corpus contains no DNA
text, so eight entries are appended: a tag block, a lower-case block, a tail shorter than
six bytes, a non-ACGT base, a block with no closing tag, text outside the tags, an all-`N`
block, and a block embedded between words. Without them the engine's whole reason to exist
would be untested.

## Alternatives considered

- **Treat `hybriddna` as `gpt2` with a new `pre` rule.** Rejected: the DNA rule is not a
  split rule — it changes *what is looked up* (marked k-mers) and emits tags the `pre` never
  sees.
- **Keep the marker in `id_to_token` and strip it at decode.** Rejected: llama.cpp's format
  contract is that the marker is invisible after load, and a decode-time strip would have to
  know the id list (it cannot distinguish a k-mer from a base token with the same spelling).
- **Do the marker erase before building the k-mer table.** Rejected: after the erase the
  marked spelling is gone from `id_to_token`, and the table's key *is* the marked spelling.
- **Unicode-aware upper-casing / character-count chunking.** Rejected: llama.cpp chunks
  bytes with `k = 6`; a character-based version would differ on any non-ASCII input inside a
  block, and on the tail pad.
- **Refuse the engine when a tag is missing.** Rejected: llama.cpp degrades to plain BPE, and
  a file without the tags is still a valid `gpt2` vocabulary.
- **Require the model key to be `gpt2`.** Rejected: the artifact says `hybriddna`, and the
  distinction decides whether the DNA rule runs.

## Consequences

- Eight engines are live; `whitespace` and `none` are still refused by name.
- The ignored id gate covers **fourteen** cached models across eight engines.
- `Tokenizer` gains a private `dna: Option<DnaData>` and the `DNA_KMER_MARKER` constant.
- **Residuals, recorded:** (a) inside a `<dna>` block a multi-byte character is split by
  *byte* chunking, exactly as llama.cpp does — its pieces are not in the vocabulary and
  become `<oov>`, which is llama.cpp's behaviour and not a loss to paper over; (b) the k-mer
  table is keyed by the bare 6-mer, so a vocabulary carrying two different ids for the same
  k-mer would keep the last one, matching `token_to_id`'s own last-wins construction;
  (c) `whitespace` still has no artifact, and the `jina-v1-en` pre rule its public GGUF uses
  is a separate gap (S3-shaped); (d) **llama.cpp's pinned revision has no input normalizer at
  all** (no NFC code in `src/`), while minfer follows the `pre` rule's declared NFC, so the
  one decomposed corpus entry (`cafe` + U+0301) is the input where the two differ. The fixture
  gates the NFC-stable spelling and its provenance says why; the decomposed entry stays gated
  for the qwen family, where the definition is HuggingFace's tokenizer (ADR-0044). This is a
  property of the reference, not a lenient gate: llama.cpp cannot be the reference for
  normalization-sensitive text on any engine.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.0 (the engine table), §3.4, §4 (the gates).
- [`docs/adr/0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md`](./0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md)
  — the dispatch this amends.
- `src/tokenizer.rs` — `TokenizerModel::HybridDna`, `hybriddna_encode`, `hybriddna_kmers`,
  `DnaData`, `DNA_KMER_MARKER`.
- `llama-vocab.cpp:1640` (the session), `:1986` (the type), `:2524` (the erase) at llama.cpp
  `@050dde50c`, the pinned reference checkout.
- `tests/fixtures/tokenizer/ids_carbon-3b.json` — the corpus (54 shared + 8 DNA) and its
  provenance.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132) (S4).
