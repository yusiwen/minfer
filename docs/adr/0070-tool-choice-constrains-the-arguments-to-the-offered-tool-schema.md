# 0070. tool_choice constrains the arguments to the offered tool's schema

- Status: Accepted
- Date: 2026-11-01
- Issues: #713
- Amends: ADR-0066 (the grammar's argument rule is the tool's schema, not `j-object`)

## Context

ADR-0066's `forced_call_grammar` fixed the wrapper and the name and then used
the schema compiler's **generic** `j-object` for the arguments. That constrains
the arguments to *some* JSON object, not to the offered tool's `parameters`
schema: any key, any size. The family `tool_choice` e2e (#714) works with seed 0,
but a probe over seeds showed the generic object can run to a kilobyte and be cut
at `max_tokens`, which the parser then refuses — a property of the grammar, not
the row.

The schema compiler already exists for `response_format`/`grammar`
(`Grammar::from_json_schema`, `compile_schema_to_gbnf`); `tool_choice` did not
use it.

## Decision

**1. The arguments rule is the selected tool's schema.** `forced_call_grammar`
now takes the offered `tools` (the request's `ToolJson` array) and, for each
selected tool, converts its `function.parameters` to a `serde_json::Value` and
compiles it with a new `grammar::compile_schema_rules(schema, "args-i")`. The
generated rules are prefixed with `args-i`, so several tools' schemas share one
GBNF.

**2. One root alternative per tool.** The root is

```
root ::= <open> "<name>" <between> args-i <close>
       | <open> "<name2>" <between> args-j <close2>
```

The row's open/between/close literals and the literal name are the same
ADR-0066/ADR-0063 words; only the arguments rule changes. `"required"` selects
every offered tool, a forced function selects the one named (the handler already
rejects a forced name that is not offered). The parser still extracts exactly the
block the grammar produces (ADR-0063 decision 7).

**3. A schema that cannot be compiled is a named 400.** `forced_call_grammar`
returns `Result<Option<String>, String>`: `Ok(None)` is a row without a grammar
shape (the Qwen3.5 and Gemma 4 rows, unchanged), `Err(detail)` is a schema the
compiler refused (e.g. an unsupported `format`), which the handler answers as
`400 invalid_request` naming the construct. There is no fallback to `j-object`.

**4. A tool without `parameters` still gets an object.** A missing schema is
compiled as `{"type": "object"}`, not as an unconstrained JSON value: OpenAI's
arguments are an object, and this keeps the old object shape without pretending
to know a schema the client did not send.

**5. The schema compiler's closed objects are what bound the keys.** A schema
with `additionalProperties: false` produces no extra-key branch, so an extra key
is outside the language; `required` and the property types are compiled by the
existing compiler. The unit gate uses `Grammar::accepts` over a closed `city`
schema (a valid witness is accepted; an extra key is refused).

**6. The rows without a grammar are unchanged.** Qwen3.5's XML and Gemma 4's
lenient body still have `grammar: None`; `tool_choice` on them is the named
"can parse but cannot constrain" 400. The XML/Gemma grammars remain #715/#718.

## Alternatives considered

- **Keep the generic `j-object`.** Rejected: it is the bug #713 records — the
  arguments are unbounded and seed-dependent, and the tool's schema is dropped.
- **Hand-write a property list from the tool definition.** Rejected: it would
  duplicate the schema compiler and drift from `response_format`'s language;
  `parameters` is a JSON Schema and the compiler is the one interpretation.
- **Compile one `anyOf` schema with the wrappers embedded.** Rejected: the
  wrappers and the literal name are not JSON Schema; the root alternatives are
  the direct expression of "one block per offered tool".
- **Validate the arguments after generation.** Rejected: that is a `400` after a
  full generation, not a constraint; `tool_choice` promises the call.
- **Fall back to `j-object` when a schema cannot be compiled.** Rejected: the
  silent unconstrained substitution the ADR-0063 boundary forbids; the handler
  names the construct instead.
- **Emit one `args` rule for every tool.** Rejected: two tools may have
  different schemas; per-tool rules keep the name and its arguments paired.

## Consequences

- `tool_choice: "required"`/forced now bounds the arguments to the selected
  tool's schema; the family e2e tools add `required`, `additionalProperties:
  false` and a `maxLength` on the string so the toy model's call is bounded and
  the test asserts the only property is `city`.
- A schema the compiler cannot express is a `400` naming the construct, before
  the job is queued; it is no longer an unconstrained generation.
- The generic `j-object` remains part of `GENERIC_RULES` for other users
  (`GrammarSource::Json`, `response_format`), and the schema rules reference the
  same `j-*` prelude; nothing about `response_format` changes.
- The tool's `parameters` is converted through `serde_json` (key order sorted);
  JSON object order is not semantic, and the compiler's property order is what
  the grammar enforces.
- The unit gate's extra-key witness is a real language check (`accepts`), not a
  text search; the e2e checks the model's actual arguments for the three
  grammar-carrying rows.

## References

- `src/server/toolcall.rs` — `forced_call_grammar`, the root alternatives and
  the `args-i` rules.
- `src/grammar.rs` — `compile_schema_rules` (new), `GENERIC_RULES`,
  `compile_schema_to_gbnf`.
- `src/server/mod.rs` — the handler's `Ok(None)`/`Err` mapping.
- `src/server/toolcall/tests.rs` — the closed-schema accept/refuse witness.
- `tests/template_e2e.rs` — the three rows' `tool_choice` gates and the
  schema-error 400.
- ADR-0063 (parser/grammar agreement), ADR-0066 (the table); issues #614,
  #699, #713.
