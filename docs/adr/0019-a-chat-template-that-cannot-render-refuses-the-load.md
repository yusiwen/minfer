# 0019. A chat template that cannot be rendered refuses the load

- Status: Accepted
- Date: 2026-09-24
- Issues: #50

## Context

`render_messages` / `render_template` compiled the GGUF `tokenizer.chat_template` into minijinja and,
**on any failure** — invalid syntax or a render error — printed a warning and returned
`fallback_chatml_messages`. For Qwen3 that path was *always* taken: its template uses Python
string-method syntax, and minijinja 2.21.0 exposes no `str` methods.

The consequence was silent, which is the part that matters: the template's think-block extraction,
tool-call formatting, multi-step-tool collapse and `enable_thinking` handling were all lost, and the
caller could not tell that a different prompt had been rendered. Two further silent paths existed: a
syntax failure at `Environment::add_template`, and a caller passing `""`, which renders the **empty
string** rather than ChatML.

## Decision

**A template that exists and cannot be rendered is a load-time refusal** — never a substituted
generic prompt. Exactly one case still gets ChatML, and it is not silent: a GGUF with **no**
`tokenizer.chat_template` at all (`Option::None`), which has no model-specific format to lose. That
path prints a one-line notice at startup and is documented in `docs/USAGE.md`.

- `src/template.rs` returns `Result<_, TemplateError>`; `template::validate` runs at load, so the CLI
  fails **before** inference and `serve` refuses to start.
- The refusal names the construct and the line, e.g. *"unsupported Python str method `splitlines`
  (template line 41); minfer refuses to fall back to a generic ChatML prompt"*.
- The mechanism is minijinja's `Environment::set_unknown_method_callback`, so Python-`str` methods are
  implemented rather than tolerated.
- Acceptance is byte-for-byte: `model_templates_render_byte_for_byte` over 4 models × 7 cases against
  transformers 5.17.0 (`tests/fixtures/chat/*.json`), with token ids pinned by
  `token_ids_match_the_reference` (52 entries × 5 cached models).

## Alternatives considered

- **A silent ChatML default on render failure.** Rejected: a template is refused loudly, never
  silently re-rendered as ChatML. The surviving ChatML case is explicitly the no-template-key case,
  announced and documented — the decision is "no **silent** fallback", not "no fallback".
- **`minijinja-contrib`'s `pycompat`**, which implements the same hook. Rejected as *"a new dependency
  and a superset with its own edge semantics"*: the engine's dependency set stays as it is
  (ADR-0007), and a superset's edge cases would be someone else's to change.
- **Upgrading minijinja.** Rejected: 2.21.0 provides the hook, so an upgrade is not required — and it
  would not by itself give Python semantics (`str.split` is not the `|split` filter, and Python's
  `lstrip` takes a *set* of characters).
- **A mix of the above** (the design record's option c). Recorded as not needed.

## Consequences

- The model's own published template renders for Qwen2.5 and Qwen3 — including Qwen3's `<think>`
  split — and a request-time render error maps to HTTP 400 on the server.
- Refusal is by name, at load: templates that call `strftime_now` or use filters this minijinja lacks
  are refused rather than half-supported. The tokenizer refuses any `tokenizer.ggml.pre` that is not
  `qwen2`/`qwen35` (including a missing key), a non-`gpt2` model, empty merges, or a vocabulary
  missing any of the 256 byte tokens.
- Accepted gaps: the Hugging Face normalizer (NFC) is **not** applied
  ([#132](https://github.com/yusiwen/minfer/issues/132)); the byte-for-byte and id gates need cached
  GGUFs, so they are `#[ignore]`d manual evidence rather than CI gates; and the GGUF copies of the
  Qwen2.5/Qwen3 templates differ **textually** from the published configs, so the real-model gate
  asserts rendered bytes and the fixture records both hashes.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — the accepted/refused constructs and the refusal text.
- `docs/QWEN3-SUPPORT-PLAN.md` §5 gotcha #9 — the root cause (Python `str` methods in the template).
- `src/template.rs`; `tests/fixtures/chat/` — the reference renderings and their provenance.
- Commit `7ef4616` (2026-09-24, "feat(f7): render the model's chat template and generalize the
  tokenizer (#50)"). Note: `29084de`, which a keyword search suggests, is a **CUDA docs commit** from
  2026-09-06, not this change.
