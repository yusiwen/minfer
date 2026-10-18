# 0053. `whitespace` is the gpt2 engine behind a whitespace cut, and it has no artifact gate

- Status: Accepted
- Date: 2026-10-17
- Issues: #132 (S4)
- Amends: ADR-0045

## Context

`tokenizer.ggml.model = whitespace` is not a new algorithm: llama.cpp maps it to
`LLAMA_VOCAB_TYPE_BPE` alongside `gpt2` (`llama-vocab.cpp:1986`) and swaps only the
session, whose entire content is the segmenting — walk the code points, drop the
whitespace, and hand each remaining run to the byte-level BPE
(`llm_tokenizer_whitespace_session`). The value exists for Jina's embedding models
(llama.cpp #18756), and `pre = whitespace` also turns lower-casing off
(`llama-vocab.cpp:2231-2233`).

**The problem is the reference, not the rule.** No public GGUF carries the model
value: four Jina-family repositories were probed, and they use `gpt2` with
`pre = jina-v1-en` (the `-zh` models) or `bert` with `pre = jina-v2-en` (the `-en`
one). The usual fallback — write a synthetic GGUF and cross-check against
`llama-tokenize` on that same file — failed four times:

| attempt | file | result |
|---|---|---|
| vocabulary only | 5 024 B | rejected: `failed to load model` |
| + minimal hparams + `token_embd`/`output_norm` | 13 696 B | abort in `llama_model_load_from_file` |
| + `head_count_kv`, `layer_norm_rms_eps`, `rope_dimension_count`, `output.weight` | 22 144 B | abort |
| + the nine per-block tensors, `block_count = 1` | 25 280 B | abort, no message |

## Decision

**1. The value is accepted, and the rule is the split.** `TokenizerModel::Whitespace`
shares the `gpt2` load path (the `pre` rule, the merge table, the byte-completeness
check); `PreTokenizer::Whitespace` returns the **non-whitespace runs** of the input,
which is exactly the segmenting the session performs, so `encode_bpe` over those runs
is the session. Nothing else changes: no NFC, no digit grouping, no accent-mark
folding, and lower-casing stays off (minfer's normalizer default, matching the
`pre = whitespace` flag).

**2. The gate is the rule's unit tests, and the ADR says so.** Because neither a real
artifact nor a synthetic one could serve as the reference, the behaviour is pinned by
tests instead: the runs, the dropped whitespace, the tiling property (`concat(split(t))
== t` minus the whitespace), the empty and all-whitespace inputs, and the "no NFC, no
digit grouping" claims. This is a **weaker** gate than every other engine's and is
recorded as such rather than papered over.

**3. The synthetic-file route is recorded as blocked, not as impossible.** The four
attempts establish what the fallback needs (a genuinely loadable minimal model, not
just a vocabulary) and where it stopped (an abort with no message, most likely a
missing hparam or an arch-specific tensor). Whoever picks it up starts from
`/tmp/wsgen.py`'s shape and llama.cpp's `llm_load_tensors` requirements.

## Alternatives considered

- **Ship without the value.** Rejected: the rule is small and well-defined, and refusing
  a file whose only sin is a rare `model` value is the kind of gratuitous refusal
  ADR-0045 exists to avoid.
- **Guess a mapping (e.g. treat it as `gpt2` + no pre rule).** Rejected: that would BPE
  the whitespace into the pieces, which is precisely what the session exists to prevent.
- **Keep hunting for an artifact.** Rejected as a blocking condition: four repositories
  probed, none carries it; the value can be revisited if one appears.
- **Wait for a loadable synthetic model.** Rejected for now: four attempts, no message
  from the loader, and the value's behaviour is simple enough to pin by test.

## Consequences

- Nine `tokenizer.ggml.model` values are accepted; none is refused by name any more for
  the BPE family, and S4's engine list is complete.
- The ignored real-model gate is unchanged (**fourteen** models, eight engines): no
  artifact carries this value, so it has **no** row in it.
- **Residuals, recorded:** (a) the gate is unit tests only; (b) `pre = jina-v1-en` and
  `pre = jina-v2-en` remain unsupported pre rules — `jina-v1-en` is a plain GPT-2 rule
  plus `add_sep`, `jina-v2-en` is not even a value in the pinned llama.cpp, so it needs a
  reference before it can be implemented; (c) a real artifact, if one appears, would turn
  this into an id-gated engine like the rest.

## References

- `src/tokenizer.rs` — `TokenizerModel::Whitespace`, `PreTokenizer::Whitespace`.
- `llama-vocab.cpp:1986` (the type), `:2231` (the pre flag), `llm_tokenizer_whitespace_session`.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132) (S4),
  [#587](https://github.com/yusiwen/minfer/issues/587) (the checker this slice leaned on).
