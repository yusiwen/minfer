# 0071. Qwen3.5's XML call gets its own tool_choice grammar

- Status: Accepted
- Date: 2026-11-01
- Issues: #715
- Amends: ADR-0067 (decision 4: the row is no longer parse-only)

## Context

ADR-0067 made Qwen3.5's XML row **parse-only**: `tool_choice: "required"`/forced
answered "can parse but cannot constrain", because an XML grammar over the
tool's JSON Schema would be a second grammar compiler. The row's call is

```
<tool_call>
<function=get_weather>
<parameter=city>Paris</parameter>
</function>
</tool_call>
```

Every parameter value is text; `xml_step` stores it as a JSON **string**
(`Value::String`) and serializes the map for OpenAI's `arguments` (ADR-0067).
The name is between `<function=` and `>`; entries repeat as
`<parameter=KEY>VALUE</parameter>`; `</function>` and `</tool_call>` close.

## Decision

**1. `GrammarShape` is an enum, not one struct.** The five JSON rows carry
`GrammarShape::Json { open, between, close }` (their output is byte-for-byte
unchanged); Qwen3.5 carries `GrammarShape::Xml { function_open, function_close,
block_close, parameter_open, parameter_close }`. The block open is
`ToolFormat::open`; `forced_call_grammar` dispatches on the shape. Gemma 4 stays
`grammar: None`.

**2. One XML root alternative per selected tool.** The grammar is

```
root ::= "<tool_call>" j-ws "<function=" "<name>" ">" j-ws <entries> j-ws "</function>" j-ws "</tool_call>"
```

with a literal name fixed per alternative (a forced function selects one tool,
`"required"` selects every offered tool). The entries are
`"<parameter=KEY>" (value-rule) "</parameter>"` joined by `j-ws`; the wrapper and
separators are the `j-ws` rule the JSON rows already use.

**3. The schema supplies the keys and the text types, flat only.** A tool's
`parameters` must be an object schema; every declared property is emitted, in the
schema's order, and `required` is validated against `properties`. The value rule
is:

- `type: "string"` (no enum/const) → `[^<]*`: any XML text without a raw `<`
  (the escape check the issue asks for; the parser stores the text, it does not
  unescape).
- `type: "integer"` / `"number"` → the generic `j-integer` / `j-number` text.
- `type: "boolean"` → `"true" | "false"`.
- `enum` / `const` → the scalar's **text** (string literal, number text,
  `true`/`false`); a string containing `<` is refused (it could close the
  parameter), and object/array/null members are refused.
- Annotations (`description`, `title`, `default`, `examples`, `deprecated`,
  `readOnly`, `writeOnly`) are ignored.

**4. The parser's string values are a recorded limit.** The grammar constrains
the **text**, not the extracted JSON type: a number-typed property is emitted as
number text, and `xml_step` still stores it as a string (ADR-0067's contract).
The acceptance is keys plus value text, not JSON types; the ADR records the gap
rather than pretending the grammar retypes the parser.

**5. Unsupported constructs are the named 400.** Object/array/null values,
`$ref`, unions (`anyOf`/`oneOf`/`allOf`), string/number constraints
(`minLength`, `pattern`, `format`, `minimum`, …), unknown keywords, a top-level
non-object schema, a property name containing `<`/`>` or leading/trailing
whitespace, and an `enum` string containing `<` are all `Err` from
`forced_call_grammar`; the handler answers the same named 400 as an
uncompilable JSON schema. There is no fallback to an unconstrained call.

**6. The parser/grammar agreement.** The grammar's block is exactly what
`xml_step` reads: the same `<function=`/`>`/`<parameter=`/`</parameter>`/
`</function>`/`</tool_call>` literals, the same whitespace tolerance, the same
name and keys. A grammar-produced call parses with the existing `xml_step`; the
unit gate feeds a witness through `Grammar::accepts` and the parser fixture
stays green.

## Alternatives considered

- **Keep the row parse-only.** Rejected: #715 is the explicit follow-up, and the
  row's named 400 is a capability hole the table seam can now fill.
- **Reuse the JSON schema compiler's rules inside the XML.** Rejected: those
  rules emit quoted JSON strings (`j-string`), while the XML value is the string
  **content**; reusing them would make the grammar emit quotes the parser stores
  as data. The XML text rule is its own small builder, not a second general
  schema compiler.
- **Support nested objects/arrays as JSON text.** Rejected: the parser stores the
  whole text as a string, so a schema-typed object would be a lie; the honest
  boundary is 400 (as the issue allows).
- **Emit only required properties.** Rejected: an optional property would then be
  unreachable, and the model's call would be a strict subset; emitting all
  declared properties is a valid superset of the required-only call and keeps the
  grammar deterministic.
- **Unescape XML entities in the parser or the grammar.** Rejected: that changes
  ADR-0067's parse contract and the fixture; the grammar allows escaped entity
  text as literal text (no raw `<`), and the parser stores what it reads.
- **Let a string value contain `<`.** Rejected: the parser's close is the first
  `</parameter>`, and a value could contain it; the `[^<]*` class is the
  conservative agreement.

## Consequences

- `tool_choice: "required"`/forced on a Qwen3.5 template now returns the XML
  call; the handler's "can parse but cannot constrain" arm remains only for
  Gemma 4's lenient body (#718).
- A flat scalar schema works (the e2e's `city` enum); a nested or union schema is
  a named 400 before the job is queued.
- The arguments' values are still JSON strings (a pre-existing ADR-0067
  property); a client that needs typed numbers must parse them.
- The five JSON rows' grammar output is unchanged; their fixtures/tests are
  untouched.
- The XML grammar is small and local (`xml_call_alternative`, `xml_parameters`,
  `xml_value_rule`), not a general XML Schema compiler.

## References

- `src/server/toolcall/format.rs` — `GrammarShape::{Json,Xml}`, the QWEN35 row.
- `src/server/toolcall.rs` — `forced_call_grammar`, `xml_call_alternative`,
  `xml_parameters`, `xml_value_rule`, `xml_scalar_text`.
- `src/server/toolcall/tests.rs` — the XML closed-schema accept/refuse witness.
- `tests/template_e2e.rs` — the Qwen3.5 `tool_choice` success gate and the
  unsupported-schema 400.
- ADR-0063 (parser/grammar agreement), ADR-0066 (the table), ADR-0067 (the parse
  row); issues #614, #708, #715.
