# 0073. Gemma 4's lenient body gets a dialect tool_choice grammar

- Status: Accepted
- Date: 2026-11-02
- Issues: #718
- Amends: ADR-0069 (decision 5: the row is no longer parse-only)

## Context

ADR-0069 made Gemma 4's lenient-JSON row **parse-only**: `tool_choice` answered
"can parse but cannot constrain". The call is

```
<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>
```

with bare keys and `<|"|>` strings. The existing schema compiler emits **standard**
JSON (`{"city": "Paris"}`), not this dialect. (The parser accepts standard JSON
strings and quoted keys too — ADR-0069 decision 3 — but the template documents
the dialect, and the issue asks the grammar to emit it.)

## Decision

**1. A `GrammarShape::Gemma` variant.** `forced_call_grammar` dispatches a third
wrapper shape: `format.open` (`<|tool_call>`) + `call:` + the literal name + the
dialect body + `<tool_call|>`. The four `Json` rows and the `Xml` row are
unchanged; Gemma 4 is no longer `grammar: None`.

**2. A small dialect compiler, not the JSON schema compiler.** `GemmaCompiler`
maps the tool's `parameters` schema to a GBNF over the Gemma syntax:

- object members: `key` + `:` + value, comma-separated, all declared properties
  in schema order; `required` names validated; `additionalProperties` does not
  add keys (the grammar emits a subset of the schema, which is valid);
- keys: bare `[A-Za-z0-9_]+` when the name is an identifier, otherwise a
  `<|"|>` string; a name containing `<` is refused;
- values: `type: "string"` → the `g-string` rule (`<|"|>` + text without a raw
  `<` + `<|"|>`); integer → `j-integer`; number → `j-number`; boolean →
  `true|false`; null → `null`; array → `[` items `]` (`items`; absent means the
  global any-array rule); nested object → recursively;
- `enum`/`const` → a scalar Gemma literal (`<|"|>text<|"|>`, number text,
  boolean, null); object/array members are refused;
- `anyOf`/`oneOf` → the alternation of its branches; annotations are ignored.

**3. The global dialect rules are shared once.** `g-string`, `g-key`, `g-object`,
`g-array`, `g-value` describe the unconstrained dialect; the per-schema rules
are prefixed (`gemma-N-…`) so several tools share one grammar. The root is one
alternative per selected tool, as the JSON rows do.

**4. The tool parameters must be an object schema.** A top-level array, enum,
union, `items`, or a non-object `type` is the named 400; the arguments are an
object by OpenAI's shape. A `true`/empty schema compiles to the global any-object
rule (the grammar emits `{}`/a declared subset; it is valid).

**5. Unsupported constructs are the named 400.** `$ref`/`allOf`/`not`, string or
number constraints (`pattern`, `format`, `minLength`, `minimum`, …), object/array
enum values, unsafe property names, a circular schema that the compiler's
recursion cannot express, and unknown keywords are `Err` from
`forced_call_grammar`; the handler answers the named 400, never an unconstrained
call.

**6. The parser/grammar agreement.** The grammar emits exactly the block
`call_then_json_step` and `GemmaJson` read: `call:NAME{…}`, bare/quoted keys,
`<|"|>` strings, the close. The unit gate feeds a dialect witness through
`Grammar::accepts`; the fixture stays green and the e2e parses the constrained
call.

## Alternatives considered

- **Reuse the `j-object` grammar inside the Gemma wrapper.** The parser does
  accept standard JSON, so this would parse (correcting the issue's premise), but
  it would emit a syntax the template does not document. Rejected: `tool_choice`
  should produce the model's own dialect (ADR-0063's "the block the parser
  extracts" is the dialect), and #718 asks for bare keys/`<|"|>`.
- **Post-process the JSON compiler's rule text.** Rejected: keys are inlined as
  quoted JSON literals and strings as `j-string`; a textual replacement cannot
  distinguish a key literal from a string enum value.
- **Support only flat scalars.** Rejected: the issue explicitly asks for arrays
  and nested objects, and the parser already reads them.
- **Emit standard JSON because the parser accepts it.** Rejected as above: a
  grammar that forces a different syntax than the template is a second format,
  not the one the row names.
- **Allow any value for an omitted/empty schema.** The dialect's global
  `g-value` is used for unconstrained property values; the top-level empty schema
  uses the any-object rule, because the arguments must be an object.
- **Support `$ref`/`allOf`.** Rejected for this increment: a local-def
  resolution is the JSON compiler's territory; the named 400 is the boundary.

## Consequences

- `tool_choice: "required"`/forced on a Gemma 4 template now returns the dialect
  call; the "can parse but cannot constrain" arm is gone for all six rows.
- Nested objects/arrays and `anyOf`/`oneOf` are supported; string/number
  constraints, `$ref`/`allOf`, and object/array enum values are the named 400.
- The arguments are serialized JSON (ADR-0069); the grammar's dialect values are
  the parser's inputs, not the model's raw bytes.
- The `g-*` rules are small and local (`push_gemma_rules`, `GemmaCompiler`), not a
  general schema compiler; the JSON rows' rules and output are unchanged.
- The Gemma fixture, the Qwen/Mistral/Llama/Qwen3.5 fixtures and their tests are
  untouched.

## References

- `src/server/toolcall/format.rs` — `GrammarShape::Gemma`, the GEMMA4 row.
- `src/server/toolcall.rs` — `forced_call_grammar`, `push_gemma_rules`,
  `GemmaCompiler`, `gemma_schema_rules`, `gemma_key`, `gemma_literal`.
- `src/server/toolcall/tests.rs` — the dialect witness (flat and nested).
- `tests/template_e2e.rs` — the Gemma `tool_choice` success gate and the
  unsupported-`pattern` 400.
- ADR-0063 (parser/grammar agreement), ADR-0066 (the table), ADR-0069 (the parse
  row); issues #699, #710, #718.
