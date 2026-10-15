# 0043. A special token is declared by the file, never inferred from its name

- Status: Accepted
- Date: 2026-10-15
- Issues: #132

## Context

The tokenizer matched special tokens as single ids before byte-level BPE, and the
set it matched was the GGUF `tokenizer.ggml.token_type` 3 (CONTROL) / 4
(USER_DEFINED) table **plus a name hardcode**: `<|im_start|>`, `<|im_end|>`, and any
EOS text starting with `<` ([#132](https://github.com/yusiwen/minfer/issues/132)
S2). The hardcode predates the type table (`b002cd8`, 2026-08-23, which added the
table for DeepSeek-R1's fullwidth-bar tokens) and was kept as a fallback for
converters that leave a control token typed NORMAL.

It fabricated tokens the vocabulary does not contain. Measured on
`unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` (Q4_K_M, the file's first 7 MB probed
2026-10-15; `tokenizer.ggml.pre = deepseek-r1-qwen`, all controls type 3/4,
`bos`/`eos`/`padding` declared): that vocabulary contains **neither** `<|im_start|>`
nor `<|im_end|>`, so the fallback mapped the literal string `<|im_start|>` to id
`0` and `<|im_end|>` to the EOS token `151643`. Two strings the model never had
became tokens; the chat-template path emits the model's own tokens, so the only
effect was on text that happens to spell them.

The reference does not do this. At the pinned checkout (`~/git/reading/llama.cpp` @
`050dde50c`), llama.cpp's parse-special cache is the predicate
`LLAMA_TOKEN_ATTR_CONTROL | USER_DEFINED | UNKNOWN`
(`llama-vocab.cpp:3012`), i.e. the type table; its *name* lists (EOT, EOM, the
EOG names and the six FIM names, `llama-vocab.cpp:2668-2820`, `:2850-2900`)
**force a type upward** with
`LLAMA_LOG_WARN("control-looking token: … was not control-type; this is probably a bug in the model")`,
and each list runs only when the matching role id is **not** declared
(`if (special_eot_id == LLAMA_TOKEN_NULL)`). Its declared roles
(`llama-arch.cpp:382-406`: bos, eos, eot, eom, unknown, seperator, padding, cls,
mask, the six fim ids and the deprecated prefix/suffix/middle aliases) are read for
generation-boundary and infill semantics, not to widen the cache.

## Decision

**1. The parse-special set is exactly what the file declares.** A token is special
when its declared `tokenizer.ggml.token_type` is `2` (UNKNOWN), `3` (CONTROL) or
`4` (USER_DEFINED) — llama.cpp's predicate verbatim, no name consulted — **or** when
a declared role names its id: the key table `SPECIAL_ROLE_KEYS` in
`src/tokenizer.rs`, transcribed from llama.cpp's role list including the deprecated
aliases and llama.cpp's own `seperator_token_id` spelling.

**2. A role declaration outranks a NORMAL type.** The file saying "this id is the
EOS / eot / fim_pad" is a declaration, not a guess, so that token matches as one id
even when the type table says NORMAL. This is the case the removed hardcode was
reaching for, now driven by data rather than by a Qwen-shaped name.

**3. Nothing is fabricated.** A role id must be in the vocabulary and its text
non-empty; a name the vocabulary lacks is never inserted. The matching order
(earliest position wins, longest text wins at the same position) is unchanged.

## Alternatives considered

- **Keep the hardcode** (the status quo). Rejected: it invents tokens the model does
  not have (measured above), and the pair it invents is Qwen-specific while the
  engine's supported set is wider.
- **Port llama.cpp's name lists** (~40 EOT/EOM/EOG/FIM names, forced only when the
  role is not declared). Rejected on two grounds. It is the reference's *guess* for
  a converter that failed to declare, and transcribing it re-creates exactly the
  name-shaped table this decision removes; and it drifts upstream (the pinned
  2026-09-07 list already carries a dozen entries newer than the original). The
  consequence is recorded rather than hidden: for a token llama.cpp recognizes **by
  name**, whose role the file does **not** declare and whose type is NORMAL,
  llama.cpp matches it and minfer BPE-splits it. A converter that wants the token
  matched declares it — that is what the role keys are for.
- **Treat every `tokenizer.ggml.*_token_id` key as a role.** Rejected: the key space
  is not restricted to vocabulary entries, so a future or model-specific key would
  silently widen the special set. The transcribed list is auditable against the
  reference and every entry is exactly a single vocabulary entry.
- **Include type 5 (UNUSED) or 6 (BYTE).** Rejected: llama.cpp's predicate excludes
  them, and 6 is the byte tokens in GGUFs that type them — matching those as
  specials would break byte-level BPE outright.
- **Refuse a GGUF whose control-looking name is typed NORMAL.** Rejected: a load
  must not depend on a name list, and the symptom of a mis-typed control token
  (a BPE split, visible in the ids) is not a reason to reject a model that
  otherwise works.

## Consequences

- The special set is auditable from the file: a type table plus a declared-role
  table, and no name anywhere in the path.
- `<|im_start|>` and friends are matched only when the file says so — exactly as in
  llama.cpp, and in every Qwen/DeepSeek GGUF measured (2026-10-15) they are type 3,
  so no supported model's tokenization changes.
- The measured defect disappears: no literal string is aliased to id `0` or to the
  EOS token, and the two `Tokenizer` fields that carried the fabricated ids
  (`im_start`, `im_end`) are gone.
- The derivation is gated through `Tokenizer::load` on synthetic GGUFs (the
  special-token derivation had no load-level test before), including a regression
  transcribed from the probed DeepSeek-R1 metadata and a control that a token
  declared nowhere stays ordinary text.
- **Residual, recorded:** llama.cpp's name lists still match a few tokens minfer
  splits (the case named in the alternatives). No supported model reaches it; a
  model that needs it gets a declaration, or a new decision.
- ADR-0019's text is not edited; the tokenizer contract it states is unchanged.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.2 (the contract), §4 (the gates).
- `src/tokenizer.rs` — `SPECIAL_ROLE_KEYS`, `Tokenizer::load`, `Tokenizer::encode`.
- `src/tokenizer/tests.rs` — the synthetic-GGUF gates and the DeepSeek-R1 transcription.
- [[#132]](https://github.com/yusiwen/minfer/issues/132) S2, and
  [#50](https://github.com/yusiwen/minfer/issues/50) (the F7 landing whose fallback
  this replaces).
