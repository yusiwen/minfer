# 0054. `jina-v1-en` is the plain GPT-2 pre rule, not a Qwen variant

- Status: Accepted
- Date: 2026-10-17
- Issues: #132 (S3's pre-rule family)

## Context

`tokenizer.ggml.pre = jina-v1-en` is what the public Jina embedding GGUFs declare
(`second-state/jina-embeddings-v2-base-zh-GGUF`, `model = gpt2`, probed 2026-10-16).
llama.cpp maps the value to `LLAMA_VOCAB_PRE_TYPE_GPT2` **plus `add_sep = true`**
(`llama-vocab.cpp:2225-2227`), and that pattern is the plain GPT-2 one:

`(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`

It differs from every rule minfer already had (`qwen2`, `qwen35`, `llama-bpe`,
`lfm2`, `gemma4`, `whitespace`) in **four** places, which is why it is a variant and
not an alias:

| # | GPT-2 | the Qwen/Llama rules |
|---|---|---|
| 1 | a piece may start with **one space** (` ?\p{L}+`) | the letter arm consumes a preceding **non-letter** (`[^\r\n\p{L}\p{N}]?`) |
| 2 | digits group **without limit** (` ?\p{N}+`) | `\p{N}` (one) or `\p{N}{1,3}` (three) |
| 3 | the symbol arm has **no** trailing `[\r\n]*` | the symbol arm eats trailing newlines |
| 4 | **no** `\s*[\r\n]+` arm at all | newlines get their own piece |

## Decision

**1. `PreTokenizer::JinaV1En` is its own scan.** The shared scanner encodes the
Qwen/Llama shape; threading four flags through it would make every arm conditional,
so the GPT-2 rule is a separate walk with the same helpers (`is_letter`,
`is_number`).

**2. The subtle arm is `\s+(?!\S)`, and it is implemented as the regex behaves.** A
run of **two or more** whitespace characters followed by a non-space keeps its
**last** character for the next piece's ` ?` prefix (the lookahead fails at full
length and the engine backtracks); a **single** space before a non-space is its own
piece (the pattern's final `\s+`); a run at end of input goes whole. The first
implementation dropped the single-space case and looped forever — the unit test's
tiling assertion is what would have caught it, and now does.

**3. `add_sep` is recorded, not guessed.** llama.cpp appends the SEP role token for
this value at the tokenizer level (`llama_tokenize`'s special-token handling), not in
the pre-tokenization; minfer has no such flag and its text path adds no specials, so
the behaviour is unchanged for `encode`. If a caller ever needs it, this ADR is where
the flag belongs.

**4. The gate is the rule's unit tests, and an id fixture is the next step.** The
artifact exists (an 82 MB Q2_K Jina GGUF) and `llama-tokenize` understands the value,
so unlike ADR-0053's `whitespace` this rule **can** be upgraded to an id-gated engine
by generating a fixture from that file. That work is recorded as the follow-up rather
than claimed.

## Alternatives considered

- **Alias it to `qwen2`.** Rejected: the four differences above are exactly the
  inputs where an embedding tokenizer's pieces change (space-prefixed words, long
  digit runs, newline tails).
- **Alias it to `whitespace`.** Rejected: `whitespace` drops the whitespace entirely;
  GPT-2 keeps a space as a piece prefix.
- **Thread flags through the shared scanner.** Rejected: four conditionals in the
  hottest split path, with no gain over a 60-line separate walk.
- **Reject the value until the id fixture exists.** Rejected: the rule is
  well-defined and reference-backed, and refusing the Jina GGUFs over a missing
  fixture is the refusal ADR-0045 exists to avoid.

## Consequences

- Six pre rules are accepted; `jina-v1-en` is the first that is not in the Qwen/Llama
  family.
- Unit-tested: the space-prefixed letter and digit arms, two-space and
  double-newline runs, the contraction and symbol arms, an unlimited digit run, and a
  tiling assertion over eight inputs.
- **Residuals, recorded:** (a) no id fixture yet (the artifact and `llama-tokenize`
  are both available, so this is a generation task, not a blocker); (b) `add_sep` is
  not implemented; (c) `jina-v2-en` remains unsupported — it is not a value in the
  pinned llama.cpp at all, so it needs a reference first (ADR-0053's residual).

## References

- `src/tokenizer.rs` — `PreTokenizer::JinaV1En`.
- `llama-vocab.cpp:2225-2227` (the type and `add_sep`), `:2205`, `:3084` (the
  `jina-v2-*` group that does **not** include `-en`).
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132).
