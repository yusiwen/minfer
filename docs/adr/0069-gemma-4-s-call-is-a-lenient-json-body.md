# 0069. Gemma 4's call is a lenient JSON body, and its arguments are serialized

- Status: Accepted
- Date: 2026-11-01
- Issues: #710
- Amends: ADR-0066 (the table gains a lenient-body block shape)

## Context

Gemma 4 writes its tool call as:

```
<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>
```

The name follows `call:`; the body is **lenient JSON**: unquoted keys and
`<|"|>` string delimiters, with numbers, booleans, null, arrays and nested
objects otherwise JSON's. transformers 5.17.0 ships a `gemma4` response-template
fallback (`_RESPONSE_TEMPLATE_FALLBACKS`) whose `json` parser applies
`unquoted_keys` and the `<|"|>` `string_delims`; llama.cpp has a matching
`common/parsers/gemma4.cpp`. minfer does not support the `gemma4` architecture
(the tokenizer engine exists; the model arch does not), so the row is reachable
through `--chat-template` only, or after arch support.

## Decision

**1. A row anchored on the piped tag and the call prefix.** `GEMMA4` anchors on
`<|tool_call>` and `call:`. The piped tag does not contain `<tool_call>` (the
`|` sits between `<` and `t`), so it does not collide with the Qwen2/Qwen3 or
Qwen3.5 rows; the row is first in the table for clarity.

**2. A `CallThenJson` block shape.** The parser reads the name between `call:`
and the first `{`, parses the lenient body into a `serde_json::Value`, and
expects `<tool_call|>` after it. A complete body at EOS without the close is a
usable call (the implicit-close rule); a body cut off is a named refusal. The
body parser is recursive (objects, arrays, strings, numbers, literals), not a
`str::find` on the close: a `<|"|>` string may contain the close literal.

**3. The lenient dialect.** Keys are bare `[A-Za-z0-9_]+` or `<|"|>`/standard
JSON strings; values are `<|"|>` strings (raw content, no escape processing, as
the reference's `string_delims` does), standard JSON strings (the reference's
`json.loads` accepts them), numbers, `true`/`false`/`null`, arrays and nested
objects. A partial token at the end of the buffer returns the "need more bytes"
status, so the streaming parser holds instead of refusing a call that is still
arriving.

**4. The arguments are serialized JSON text.** The body is not JSON the model can
hand back as bytes, so the parser serializes the value for OpenAI's `arguments`
string, the way the Qwen3.5 XML row does. Key order is the serializer's and
values are JSON values; the fixture compares parsed values, not bytes.

**5. The row is parse-only.** `grammar: None`. The existing schema compiler's
`j-object` produces **standard** JSON; this parser does not accept that as a
Gemma body, so the grammar and parser would disagree (the ADR-0063 decision-7
invariant). A Gemma-dialect grammar is #718; until then `tool_choice:
"required"`/forced on the template answers the named "can parse but cannot
constrain" 400.

**6. The other rows are untouched.** The Qwen/Mistral/Llama/Qwen3.5 fixtures and
grammars are unchanged; this is one more row plus one more block shape.

## Alternatives considered

- **Rewrite the body to standard JSON with a regex pre-pass and reuse
  `raw_value_len`.** Rejected: unquoted keys and `<|"|>` strings need
  context-aware scanning (a key inside a string, nested braces), and the
  reference's own behavior is the oracle; a recursive parser says what it
  accepts.
- **Use the `j-object` grammar for `tool_choice`.** Rejected: it produces
  standard JSON, which the Gemma parser would refuse; the grammar and parser must
  describe the same block.
- **Reject unquoted keys / only accept `<|"|>` strings.** Rejected: unquoted keys
  are the format; accepting standard JSON strings too matches the reference's
  `json.loads` transform and costs one branch.
- **Find the close with `str::find("<tool_call|>")`.** Rejected: the close
  literal can appear inside a `<|"|>` string; the parser must find the value's
  end first (the same reason the Qwen parser never finds its close by search).
- **Detect by the architecture.** Rejected: the template decides the syntax, and
  `--chat-template` can put this dialect on a Qwen checkpoint.

## Consequences

- A Gemma 4 call completion becomes `message.tool_calls` with JSON arguments
  where it used to be a 400; the HTTP gate forces the block and checks the
  serialized `{"city": "Paris"}`.
- `tool_choice` stays a named 400 for this row until #718.
- The lenient parser is one more block shape, not a second parser framework; its
  incomplete-token handling is covered by the streaming fold over the fixture.
- The fixture uses transformers' **released** `gemma4` fallback, a stronger
  reference than the authored Mistral/Ministral/Llama templates.
- Gemma 4 is still not a supported architecture; this is a template-format row,
  reachable through the override (or a future arch).

## References

- `src/server/toolcall/format.rs` — `GEMMA4`, `BlockShape::CallThenJson`.
- `src/server/toolcall.rs` — `call_then_json_step`, `GemmaJson`, `gemma_json`,
  `call_from_value`.
- `tests/fixtures/chat/parse_gemma4.json` — the released-fallback reference.
- `scripts/gen_tool_call_family_fixtures.py` — `--family gemma4` and `--check`.
- `tests/template_e2e.rs` — the forced block and the named 400.
- ADR-0063 (parser/grammar agreement), ADR-0066 (the table); issues #699, #710.
