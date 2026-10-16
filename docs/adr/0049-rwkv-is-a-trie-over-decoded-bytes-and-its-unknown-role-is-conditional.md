# 0049. rwkv is a trie over decoded bytes, and its unknown role is conditional

- Status: Accepted
- Date: 2026-10-16
- Issues: #132 (S4), #575
- Amends: ADR-0045

## Context

ADR-0045 made `tokenizer.ggml.model` a dispatch; ADR-0046–ADR-0048 added three
engines and fixed the reference rule: *the implementation that defines the format
is the reference.* `model = rwkv` is the sixth value, and it is the first whose
vocabulary entries do not stand for themselves:

- the vocabulary is **escaped** — `\xNN` is the byte 0xNN, `\t`/`\n`/`\r` are
  themselves, any other `\c` is `c` (`llama_unescape_rwkv_token`,
  `llama-vocab.cpp:1247`) — so a trie built from the *text* would never match the
  input; the decoded bytes are what the token stands for, and many of them are not
  valid UTF-8;
- there are **no merges, no scores and no `pre` rule**: the vocabulary is the
  tokenizer;
- encoding is a **greedy longest match** over that trie, and a byte no entry
  starts with emits the unknown token and advances one byte
  (`llm_tokenizer_rwkv_session`, `llama-vocab.cpp:1285`);
- decode is the same unescaping (`llama-vocab.cpp:3674`).

Measured 2026-10-16 on `latestissue/rwkv-6-world-1b6-gguf`: 65 536 tokens,
`token_type` declared, **no** `merges`, **no** `scores`, **no** `pre`, and **no**
`tokenizer.ggml.unknown_token_id` (bos/eos 0).

## Decision

**1. `rwkv` is accepted as the sixth engine.** `VocabTrie` gains byte-level entry
points (`insert_bytes`, `longest_prefix_bytes`, `starts_with_byte`); the loader
inserts every entry's **decoded** bytes, exactly as llama.cpp does.

**2. The unknown role is conditional, and the condition is checked.** llama.cpp
requires no `unknown_token_id` for RVKV and leaves it `LLAMA_TOKEN_NULL`; its
session would then push `-1`. This vocabulary decodes to a byte set that covers
every value — the path is unreachable — but minfer does not rely on that: `load` checks that **every byte value starts some entry**. If the role is
declared it is used (and range-checked); if it is not and every byte is covered,
the fallback is unreachable by construction; if it is not and a byte is
uncovered, the load is **refused**, naming the byte — never an invented id.

**3. The reference is the model author's own tokenizer**
(`RWKV/rwkv-6-world-1b6`'s `hf_rwkv_tokenizer.py` + `rwkv_vocab_v20230424.txt`) —
the definition of both the escaping and the greedy walk — and llama.cpp's
`llama-tokenize` on the same artifact agrees with it on **all 54** corpus entries
(`Hello, world!` → `[33155, 45, 40213, 34]`). minfer agrees with both.

**4. The committed artifact is the Q4_0 quant.** The repository's **Q2_K** file
trips minfer's tensor-offset accounting
(`tensor 'blk.0.attn_norm.weight' has offset 44040192, expected 36700160`) while
llama.cpp loads and tokenizes it — the second witness for
[#575](https://github.com/yusiwen/minfer/issues/575), which is where the diagnosis
and the discriminator (the Q4_0 file) are recorded. The Q4_0 file parses, so it is
the fixture's artifact; the tokenizer metadata is identical between the two.

## Alternatives considered

- **Use RWKV-4's `tokenizer.json` as the reference.** Rejected: it is a BPE
  rendering of a *different* vocabulary generation, like the WordPiece and unigram
  cases where the definition lives elsewhere.
- **Decode the escapes at encode time instead of at load.** Rejected: the trie is
  matched against raw input bytes; decoding per candidate would make the walk
  quadratic and still need the byte trie.
- **Drop a byte that starts no entry** (the way the BPE engines treat an unmatched
  piece). Rejected: RVKV's contract is one unknown token per unmatched byte, and
  dropping input silently is the failure class this engine exists to avoid.
- **Require `tokenizer.ggml.unknown_token_id` outright.** Rejected: no real RVKV
  file declares it, so the engine would refuse every model in the family. The
  conditional check keeps the refusal where it belongs — a vocabulary that really
  cannot represent a byte.
- **Copy llama.cpp's `-1`.** Rejected: an out-of-range id in the output is exactly
  the silent wrongness ADR-0043 and the byte-fallback gates exist to prevent.

## Consequences

- Six engines are live; `plamo2`, `whitespace`, `hybriddna` and `none` are still
  refused by name.
- The ignored gate covers **twelve** cached models across six engines.
- `VocabTrie` is now byte-oriented, which the unigram engine (UTF-8 entries) and
  the RWKV engine (arbitrary bytes) share.
- **Residuals, recorded:** (a) RVKV's `<unk>` (and every other entry) is in the
  trie, so a text containing `<unk>` matches that token, because llama.cpp inserts
  every entry too — a recorded consequence, not a divergence (a text containing
  `<unk>` matches that token on both engines); (b) the loader's tensor-offset accounting still rejects some quant
  families, filed as [#575](https://github.com/yusiwen/minfer/issues/575) with two
  witnesses and the Q4_0 discriminator; (c) the corpus carries no text with an
  uncovered byte, because no such byte exists in the vocabulary.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.0 (the engine table), §3.4,
  §4 (the gates).
- [`docs/adr/0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md`](./0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md)
  — the dispatch this amends.
- `src/tokenizer.rs` — `TokenizerModel::Rwkv`, `rwkv_encode`,
  `unescape_rwkv_token`, `VocabTrie::insert_bytes`.
- `llama-vocab.cpp:1247`, `:1285` and `:3674` (llama.cpp `@050dde50c`, the pinned
  reference checkout) — the ported behaviour.
- `tests/fixtures/tokenizer/ids_rwkv-6-world-1.6b.json` — the corpus, its
  reference and the artifact note.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132) (S4),
  [#575](https://github.com/yusiwen/minfer/issues/575) (the loader accounting).
