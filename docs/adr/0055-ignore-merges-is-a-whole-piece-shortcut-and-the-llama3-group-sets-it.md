# 0055. `ignore_merges` is a whole-piece shortcut, and the llama3 pre group sets it

- Status: Accepted
- Date: 2026-10-17
- Issues: #132 (S3's pre-rule family)

## Context

llama.cpp sets `ignore_merges = true` for its whole
`LLAMA_VOCAB_PRE_TYPE_LLAMA3` group (`llama-vocab.cpp:2155-2169`): `llama-bpe`,
`falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`, `lfm2`, `jina-v5-nano`. The name is
misleading — the flag does **one** thing, in the BPE piece routine
(`llama-vocab.cpp:626-630`):

```cpp
if (vocab.get_ignore_merges() && vocab.text_to_token(word) != LLAMA_TOKEN_NULL) {
    output.push_back(vocab.text_to_token(word));
    continue;                       // before the merge loop
}
```

A pre-split piece that is **itself** a vocabulary entry is emitted whole; every other
piece still goes through the merges. So it is a **whole-piece shortcut**, not "ignore
the merge table".

minfer was missing it for every value it already accepted (`llama-bpe` and `lfm2`
included). The existing id fixtures passed 54/54 either way, which proves the shortcut
does not change *those corpora* — not that it is absent: the two differ exactly on a
piece that is a whole vocabulary entry **and** unreachable through the merges.

## Decision

**1. The flag is a `Tokenizer` field, set from the pre rule.**
`PreTokenizer::ignores_merges()` is true for `LlamaBpe` and `Lfm2`; `load` copies it
into `Tokenizer::ignore_merges`, and `encode_bpe` does the whole-piece lookup before
`bpe_encode`.

**2. The five missing members of the group are accepted.** `falcon3`, `falcon-h1`,
`pixtral`, `midm-2.0` and `jina-v5-nano` are the **same regex** as `llama-bpe`, so
`PreTokenizer::from_gguf` maps them to `LlamaBpe` and the flag carries the difference —
they are not separate variants, unlike `lfm2` (which kept its own variant so the
fixture and refusals name the value the file carries).

**3. The regression evidence is the fourteen cached models, not a new fixture.** The
ignored id gate runs with the flag **on** for `llama-3.2-1b-instruct` and `lfm2-1.2b`
now, and both still match byte for byte — as does the rest of the set. A synthetic unit
test pins the behaviour the corpus cannot: a vocabulary where `hello` is an entry with
no merge path (`[1,2,3,3,4]` without the flag, `[5]` with it, `hel` still `[1,2,3]`),
plus the group membership (nine values set it, five do not).

## Alternatives considered

- **Treat the flag as "skip the merges entirely".** Rejected: the reference joins only a
  piece that is a whole entry; a piece that is not still needs the merge ranks.
- **Add five new `PreTokenizer` variants.** Rejected: the regex is identical, so five
  variants would be five copies of one rule with a boolean attached — the flag is the
  difference.
- **Land the aliases without the flag.** Rejected: that is the bug this ADR fixes for
  `llama-bpe`/`lfm2`; adding five more values to it would be worse.
- **Gate on a new id fixture only.** Rejected: the fourteen existing models are stronger
  evidence than one new file, and they are already cached.

## Consequences

- Eleven `pre` values are accepted: `qwen2`/`qwen35`/`llama-bpe` (+ its `llama3`/
  `llama-v3` aliases and the five group members)/`lfm2`/`gemma4`/`jina-v1-en`/
  `whitespace`.
- `ignore_merges` is now part of every byte-level BPE encode path; the fourteen-model
  gate is the regression net.
- **Residuals, recorded:** (a) two of the five new values now have an id fixture —
  `falcon3` (via ADR-0056's byte-reachability fix, #593) and `falcon-h1`
  (`tiiuae/Falcon-H1-0.5B-Instruct-GGUF`, Q2_K, 200 913 184 B); `pixtral`,
  `midm-2.0` and `jina-v5-nano` still need an artifact small enough to cache (the
  smallest Pixtral GGUF is ~7 GB); (b) the
  `add_bos` flag that comes with the same group is not implemented, like `add_sep`
  (ADR-0054) — minfer's text path adds no specials; (c) `ignore_merges` also appears at
  six load sites in llama.cpp; only the `LLAMA_VOCAB_PRE_TYPE_LLAMA3` group is covered
  here, and the others (`deepseek-v3`-style) come with their own slices.

## References

- `src/tokenizer.rs` — `Tokenizer::ignore_merges`, `PreTokenizer::ignores_merges`,
  `encode_bpe`, `PreTokenizer::from_gguf`.
- `llama-vocab.cpp:626-630`, `:2155-2169`.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132).
