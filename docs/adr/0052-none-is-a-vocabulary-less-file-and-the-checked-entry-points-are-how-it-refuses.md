# 0052. `none` is a vocabulary-less file, and the checked entry points are how it refuses

- Status: Accepted
- Date: 2026-10-16
- Issues: #132 (S4), #585, #586
- Amends: ADR-0045

## Context

`tokenizer.ggml.model = no_vocab` (or `none`) means the model has **no vocabulary**
(`LLAMA_VOCAB_TYPE_NONE`, `llama.h:73`). llama.cpp's load branch
(`llama-vocab.cpp:1942-1961`) sets the type, leaves every special role `LLAMA_TOKEN_NULL`,
reads the optional `tokenizer.ggml.vocab_size` and resizes `id_to_token` to that many
**placeholder** entries ("adding %u dummy tokens"), then **returns early** — skipping the
token array, the merges, the `pre` rule and the tokenizer construction. Afterwards
`tokenize` **aborts** (`:3587`), `detokenize` returns 0, and every `is_*`/`text_to_token`/
`get_token_data`/`token_get_text`/`token_get_score` asserts (`:3131-3990`). The ids come
from the caller — that is the point for vision/audio/diffusion pipelines, and the declared
size exists so the runtime's vocabulary dimension matches the model's output tensor.

minfer refused the value at load, so such a file could not even be inspected (`minfer info`
failed in `Tokenizer::load`). In the same area, minfer mapped a **missing**
`tokenizer.ggml.model` to `gpt2`, while llama.cpp reads that key as **required**
(`ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model)`, no optional flag, `:1935`).

## Decision

**1. `none` and `no_vocab` are accepted, and the placeholder contract is built.** The load
branch fills `id_to_token` / `id_to_score` / `id_to_type` with `vocab_size` placeholders
(empty text, zero score, type 0) and no specials — the shape llama.cpp builds. `minfer info`
therefore works on such a file.

**2. Text↔id refuses through a checked entry point, and that is a deliberate divergence.**
`Tokenizer::encode_checked(text) -> Result<Vec<u32>, String>` and
`decode_checked(&[u32]) -> Result<Vec<u8>, String>` return a **named refusal** for `model =
none`; `encode`/`decode_bytes` keep their infallible contract for the engines that cannot
fail, and `encode_checked` delegates to `encode` for them (so nothing else changed). The CLI
routes its prompt encoding and its display decoding through the checked pair. llama.cpp
aborts on one side (`tokenize`) and silently returns 0 on the other (`detokenize`); minfer
refuses both by name, because an empty result is indistinguishable from "this text produced
no tokens" and a panic is what [#580] removed.

**3. A missing `tokenizer.ggml.model` is a refusal.** `TokenizerModel::from_gguf(None)` now
names the missing key instead of returning `Gpt2` ([#585]).

## Alternatives considered

- **Keep refusing `none` at load.** Rejected: the file cannot be inspected at all, and the
  type is a legitimate part of the format.
- **Accept it and let `encode` return an empty vector** (llama.cpp's `detokenize` shape).
  Rejected: silent emptiness is exactly the failure class the project refuses.
- **Abort/panic like `tokenize`.** Rejected: [#580] just replaced that pattern with a named
  refusal and a non-zero exit.
- **Default a missing model key to `gpt2`** (minfer's old behaviour). Rejected: llama.cpp
  requires the key, so a file without it is malformed; assuming `gpt2` turns a load error
  into a later, confusing vocabulary error.
- **Make `encode` itself fallible.** Rejected: it would touch every caller for a case only
  one engine has, so the checked pair is added alongside instead.

## Consequences

- Nine `tokenizer.ggml.model` values are accepted, `whitespace` is the only refusal left.
- `minfer info` works on a vocabulary-less GGUF; `minfer <file> "text"` exits non-zero with
  the loader's message (the [#580] contract).
- The dead-code gate is what proved the pair was wired: `decode_checked` had no caller until
  the CLI's display path was routed through it — a `deny(warnings)` build failure, not a
  review comment.
- **Residual:** a vocabulary-less model still has no *decode* path at all (llama.cpp emits
  nothing), so `minfer` refuses; if a real pipeline ever needs caller-supplied ids, the
  checked pair is where that contract would be expressed.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.0, §3.4, §4.
- `src/tokenizer.rs` — `TokenizerModel::None`, `encode_checked`, `decode_checked`;
  `src/main.rs` — `decode_token`.
- `llama-vocab.cpp:1935`, `:1942`, `:3131-3166`, `:3587`, `:3725`, `:3965-3990`;
  `llama.h:73`.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132) (S4),
  [#585](https://github.com/yusiwen/minfer/issues/585) (the required key),
  [#586](https://github.com/yusiwen/minfer/issues/586) (this contract),
  [#580](https://github.com/yusiwen/minfer/issues/580) (the refusal shape it reuses).
