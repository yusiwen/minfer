# 0067. Qwen3.5's XML call is parsed, and its arguments are serialized JSON

- Status: Accepted
- Date: 2026-11-01
- Issues: #708
- Amends: ADR-0066 (the table gains an XML block shape and a parse-only row)

## Context

ADR-0066 made a tool-call format a table row and left Qwen3.5's XML body as an
out-of-scope follow-up (#708). The format is:

```
<tool_call>
<function=get_weather>
<parameter=city>Paris</parameter>
</function>
</tool_call>
```

It is not JSON: the outer `<tool_call>` anchor matches, then the parser's
`raw_value_len()` sees `<function=…>` and answers a 400, because the Qwen2/Qwen3
row expects one JSON object. transformers 5.17.0 ships a `qwen3_5` response
template fallback and llama.cpp has a matching Qwen3-Coder/3.5 parser; both wrap
the same markup. minfer does not support the `qwen35` architecture, so the row is
reachable today through `--chat-template` on a supported model (or after arch
support lands).

## Decision

**1. One parse-only row, detected before the Qwen row.** `QWEN35` anchors on
`<tool_call>` **and** `<function=`, and sits first in the table; `<tool_call>`
without `<function=` keeps resolving to the Qwen2/Qwen3 row. The block's shape is
`BlockShape::XmlFunction { function_close, block_close }`, carrying the two
closing literals; the JSON rows are unchanged.

**2. The parser reads the markup, with the same incremental holdback.** The body
is `<function=NAME>` followed by repeated `<parameter=KEY>VALUE</parameter>`
entries. Values are the text between the tags, trimmed (transformers' `xml-inline`
does the same). A complete `</function>` with no `</tool_call>` at EOS is a usable
call, the implicit-close rule the JSON rows already use. A truncated parameter is
a named refusal: this is a **deliberate divergence** from transformers'
`xml-inline`, which returns an empty object for an unclosed parameter; the fixture
records the reference's acceptance and the divergence, and #708 requires the
refusal. A parameter containing the literal `</parameter>` is not supported (the
reference's tag regex has the same boundary).

**3. The arguments are serialized JSON text.** The XML values are strings, so the
parser collects them into an object and serializes it with `serde_json` for
OpenAI's `function.arguments` string. Unlike the JSON rows, this does **not** claim
the model's own bytes: there is no JSON in the completion to carry, and OpenAI's
shape is a string. Key order is the serializer's (the repo has `preserve_order`
off), and values are JSON strings; the fixture compares parsed values, not bytes.

**4. `tool_choice` stays a named 400.** The row sets `grammar: None`. An XML
grammar over the tool's JSON schema is a second grammar compiler, and the honest
first increment is the parse. `tool_choice: "required"`/forced on a Qwen3.5
template answers the handler's "can parse but cannot constrain" 400, never an
unconstrained generation.

**5. The existing Qwen fixture stays byte-identical.** The change is a new row and
a new block shape; the JSON rows, their grammars and the Qwen2/Qwen3 detection are
untouched.

## Alternatives considered

- **Leave the format a 400.** Rejected: #699's capability gap was that a client on
  another family gets no call at all; a parse-only row is strictly better, and the
  parser already has the seam.
- **Carry the raw XML body as `arguments`.** Rejected: OpenAI's `arguments` is a
  string holding JSON; a client would have to know the XML dialect. The issue says
  the parsed object is serialized back.
- **Accept the reference's `{}` for a truncated parameter.** Rejected: that is a
  half call, and #708's acceptance explicitly requires a named refusal. The
  divergence is recorded rather than hidden.
- **Build the XML `tool_choice` grammar now.** Rejected as scope: it needs an XML
  value/key grammar over the tool's JSON schema, i.e. a second compiler, and the
  parser/grammar-agreement rule (ADR-0063 decision 7) would apply to it. Parse-only
  first; the named 400 is the boundary.
- **Detect Qwen3.5 by the architecture.** Rejected: the template is the thing that
  tells the model which syntax to write, and `--chat-template` can put the XML on a
  Qwen2/Qwen3 checkpoint; detection stays on the template source.

## Consequences

- A completion in the Qwen3.5 XML format now returns `finish_reason:
  "tool_calls"` and `message.tool_calls` with JSON arguments, where it used to be
  a 400.
- The row is parse-only: `tool_choice` remains a named 400, and a client that
  needs the constraint waits for the XML grammar.
- The parser's XML branch is one more `BlockShape`, not a second parser; the JSON
  rows and their fixtures are untouched.
- The fixture uses transformers' **released** `qwen3_5` fallback, so it is a
  stronger reference than the authored Mistral/Llama templates; its one recorded
  divergence is the truncated parameter.
- Qwen3.5 is still not a supported architecture; this is a template-format row,
  reachable through the override (or a future arch).

## References

- `src/server/toolcall/format.rs` — `QWEN35`, `BlockShape::XmlFunction`.
- `src/server/toolcall.rs` — `xml_step`, `xml_call`.
- `tests/fixtures/chat/parse_qwen35.json` — the released-fallback reference.
- `scripts/gen_tool_call_family_fixtures.py` — `--family qwen3_5` and `--check`.
- `tests/template_e2e.rs` — the forced-block parse and the named 400.
- ADR-0063 (the parser/grammar agreement), ADR-0066 (the table); issues #614,
  #699, #708.
