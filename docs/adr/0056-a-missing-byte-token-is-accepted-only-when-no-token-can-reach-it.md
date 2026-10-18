# 0056. A missing byte token is accepted only when no token can reach it

- Status: Accepted
- Date: 2026-10-17
- Issues: #593, #132 (S3's pre-rule family)
- Amends: ADR-0045

## Context

ADR-0045 refused any byte-level BPE whose vocabulary does not carry all 256
single-byte tokens, on the ground that [`Tokenizer::bpe_encode`]'s byte fallback would
otherwise drop bytes silently. **That was stricter than its own justification.**
`Falcon3-1B-Instruct` (`general.architecture = llama`, `tokenizer.ggml.model = gpt2`,
`pre = falcon3`) omits **18** byte tokens, and llama.cpp loads and tokenizes it — so a
plain `llama`-architecture model minfer could run was refused at load.

Measured on that artifact (262 149 tokens, `GGUFReader`):

| set | bytes |
|---|---|
| missing byte tokens | `[0x09, 0x0A, 0x12, 0x17, 0x1E, 0xC0, 0xC1, 0xF5-0xFF]` (18) |
| never occur anywhere in the vocabulary | `[0xC0, 0xC1, 0xF5-0xFF]` (13) — invalid UTF-8 lead bytes |
| occur **and start** a vocabulary entry | `[0x09, 0x0A, 0x12, 0x17, 0x1E]` (5) — tab, newline, three control bytes |

The fallback is reached only by a piece that no token matches. A byte that never occurs
in the vocabulary has no token to fall back **to** — nothing is lost. A byte that
**starts** an existing token means a piece beginning with it is matched by that token
before the fallback is consulted. Neither case can drop a byte silently; both were
being refused.

## Decision

**1. The check refuses only the reachable set.** A missing byte token is accepted when
the byte never occurs in the vocabulary, or when it starts an existing token.
Otherwise the load still fails, with the same message, now naming the **reachable**
count and first byte.

**2. The rule is measured, not guessed, and it is general.** No hardcoded allow-list:
`Falcon3`'s five control bytes qualify because each starts a vocabulary entry, which is
a property of the file. The predicate is unit-free because the id gate proves it —
`Falcon3-1B`'s 54-entry fixture matches `llama-tokenize` byte for byte, and the cached
model set (`llama-3.2`, Qwen, LFM2, Gemma-4, RWKV, PLAMO-2, Carbon-3B, TinyLlama, BGE,
FLAN-T5) is unchanged.

**3. The id gate is the evidence, not a synthetic vocabulary.** A synthetic test would
have to re-implement the reachability argument it is supposed to check; the
fifteen-model gate exercises real vocabularies, including the one that motivated the
change.

## Alternatives considered

- **Hardcode the five control bytes as acceptable.** Rejected: it encodes one file's
  accident as a rule, and the reachability property is already checkable from the
  vocabulary.
- **Relax to a warning and a lossy fallback for any missing byte.** Rejected again (as
  in ADR-0053): it gives up the "no silent lossy fallback" promise globally to serve
  files that need only a narrower predicate.
- **Refuse and treat Falcon3 as unsupported.** Rejected: the model is a supported
  architecture and llama.cpp runs it; the refusal was the bug.
- **Accept any vocabulary with ≥ 161 byte tokens.** Rejected: a count is not a
  reachability argument.

## Consequences

- `Falcon3-1B-Instruct` loads, and its tokenizer matches the reference on 54 entries;
  the id gate grows from fourteen to **fifteen** models.
- **The Jina embedding GGUF also becomes loadable** (all 95 of its missing bytes start
  tokens). It is still not *runnable* — `jina-bert-v2` is not a supported architecture —
  and it still has no id fixture; ADR-0053's "unloadable by design" note is superseded
  on the loadability point only.
- **Residuals, recorded:** (a) the acceptance is silent at load — a byte-absent
  vocabulary prints nothing, and a warning channel does not exist yet; (b) the check
  runs once per load and scans the vocabulary (18 × 262 149 byte comparisons for
  Falcon3, well under a millisecond); (c) a vocabulary whose missing byte occurs
  **only inside** tokens it never starts is still refused — no artifact in the cached
  set does that, and the message names it if one appears.

## References

- `src/tokenizer.rs` — the `gpt2` load arm's completeness check; `Tokenizer::bpe_encode`.
- `tests/fixtures/tokenizer/ids_falcon3-1b-instruct.json` (54 entries, `llama-tokenize`
  on `bartowski/Falcon3-1B-Instruct-GGUF`'s Q2_K, 727 087 264 B).
- Issues: [#593](https://github.com/yusiwen/minfer/issues/593),
  [#132](https://github.com/yusiwen/minfer/issues/132).
