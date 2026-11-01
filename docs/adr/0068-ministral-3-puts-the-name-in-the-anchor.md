# 0068. Ministral 3 puts the name in the anchor, and the grammar ends at the arguments

- Status: Accepted
- Date: 2026-11-01
- Issues: #709
- Amends: ADR-0066 (the table gains a name-in-anchor block shape)

## Context

ADR-0066 made the tool-call syntax a table row. The Mistral family has two
shapes, not one:

- Mistral Nemo / Magistral: `[TOOL_CALLS] [{"name": …, "arguments": …}]`, a JSON
  array of calls (#699; `common_chat_parse_mistral_nemo`);
- Ministral 3: `[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}` — the **name is
  the text between the open and `[ARGS]`**, and the JSON value after `[ARGS]` is
  the arguments object itself (#709; `common_chat_parse_ministral_3`).

The #699 row cannot represent the second shape: its anchors do not capture the
name, and its `name_key`/`args_key` describe a wrapper object the format does not
have. A `[TOOL_CALLS]` template that also contains `[ARGS]` must not be routed to
the array parser.

## Decision

**1. A row, excluded from the array row.** `MISTRAL3` anchors on `[TOOL_CALLS]`
**and** `[ARGS]`, and sits before `MISTRAL_NEMO`; the Nemo row keeps its
`without: ["[ARGS]"]`. A template with the array form has no `[ARGS]` and still
resolves to Nemo; a template with `[ARGS]` resolves to Ministral 3.

**2. A `NameThenArgs` block shape.** The parser reads the text before
`args_marker` (`[ARGS]`) as the name, trims it, parses the JSON value after the
marker as the arguments, and ends the call at the next `[TOOL_CALLS]` or EOS. One
call per block; repeated calls are separate blocks. A truncated arguments value
is a named refusal, and a partial next open is held until it resolves.

**3. The arguments keep the model's bytes.** The JSON value after `[ARGS]` is the
object the model wrote; a string value is decoded (OpenAI's arguments is text).
The row's `name_key`/`args_key` are unused by this shape, the same way the XML
row's are; the fields stay for the uniform row description.

**4. The `tool_choice` grammar ends at the arguments.** `GrammarShape` gains the
meaning "empty `close` = the arguments end the block": the grammar is
`[TOOL_CALLS]<name>[ARGS]<j-object>`, with the name fixed or enumerated and the
schema compiler's `j-object` for the arguments. The builder omits the space and
literal of an empty close; the other rows' grammar strings are unchanged. The
grammar produces the block the parser reads, so extraction and the constraint
agree by construction (ADR-0063 decision 7).

**5. Streaming is unchanged.** One `delta.tool_calls` start plus one arguments
fragment per call; the event→chunk mapping is shared with the other rows.

## Alternatives considered

- **Parse it with the array row and a special case.** Rejected: the body starts
  with a name, not `[`; the array parser would answer a malformed-JSON refusal or
  need a second parse inside the row.
- **Wrap the call as `{"name": …, "arguments": …}` in the grammar.** Rejected: the
  model does not write a wrapper, so the grammar and the parser would disagree
  about the block (the ADR-0063 decision 7 invariant).
- **Reuse `ObjectUntilNextOpen`.** Rejected: it expects a JSON value at the body's
  start; the name is outside JSON.
- **Keep one Mistral row and branch on `[ARGS]`.** Rejected: two shapes with
  different anchors and parsers are exactly what the table is for; `without`
  keeps detection unambiguous.
- **Capture the name with a regex at detection.** Rejected: detection is a
  literal scan (ADR-0066); the name capture happens at parse time.

## Consequences

- `[TOOL_CALLS]name[ARGS]{…}` completions become `message.tool_calls` with the
  name from the anchor and the arguments object's bytes, and `tool_choice:
  "required"`/forced returns that block (the toy-model e2e uses `seed: 0`; the
  arguments are a generic object).
- The array row is unchanged; the `without` exclusion is what keeps the two rows
  from stealing each other's detection.
- The grammar builder now treats an empty `close` as "no trailing literal"; the
  Qwen/Mistral/Llama grammar strings are byte-identical to before.
- The fixture is authored (transformers ships no Ministral 3 template), with
  llama.cpp's parser as the second oracle; the arguments are compared as values
  and, for objects, as the block's own bytes.
- A name containing `[ARGS]` is not supported (the marker splits the block); the
  format's names are identifiers.

## References

- `src/server/toolcall/format.rs` — `MISTRAL3`, `BlockShape::NameThenArgs`.
- `src/server/toolcall.rs` — `name_args_step`, `call_from_arguments`, the empty
  close in `forced_call_grammar`.
- `tests/fixtures/chat/parse_ministral3.json` — the authored reference.
- `scripts/gen_tool_call_family_fixtures.py` — `--family ministral3`, `--check`.
- `tests/template_e2e.rs` — the `tool_choice` gate on the row.
- ADR-0063 (parser/grammar agreement), ADR-0066 (the table); issues #699, #709.
