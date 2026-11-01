# 0066. A tool-call format is a table row

- Status: Accepted
- Date: 2026-11-01
- Issues: #699
- Amends: ADR-0063 (the `<tool_call>` format was three constants; the parser and the `tool_choice` grammar now read a row)

## Context

#614 shipped extraction and `tool_choice` for one format: `<tool_call>` around
`{"name": …, "arguments": …}`, with the literal tags, the two key names and the
grammar wrapper spread across `src/server/toolcall.rs` and
`src/server/mod.rs::chat_completions`. The follow-up (#699) has to cover other
template families, and their differences are not a different constant:

- Llama 3.1 writes `{"name": …, "parameters": …}` — a different arguments key;
- Mistral Nemo writes `[TOOL_CALLS] [{…}]` — an array of calls, one block, no
  close tag; Ministral 3 writes `[TOOL_CALLS]name[ARGS]{…}` — the name in the
  anchor;
- Qwen3.5 writes an XML-ish `<function=…><parameter=…>` body (tracked as #708);
- Gemma 4 writes `<|tool_call>call:name{…}` with unquoted keys and `<|"|>`
  strings (tracked as #710).

There is also no family metadata to read: a GGUF carries the chat template's
source and nothing that says "this is Mistral". Detection therefore has to read
the template, exactly as llama.cpp's auto-parser and transformers' response-template
fallbacks do — but transformers ships Mistral/Llama templates for none of those
families, so the fixture references for #699 are authored, not released.

## Decision

**1. One row per format.** `src/server/toolcall/format.rs` holds a `ToolFormat`
with: a stable id; the anchors that must all appear in the template for
detection; the literals that open and close a call; the object keys that carry
the name and the arguments; and an optional `GrammarShape` (the literals a
`tool_choice` GBNF wraps around the name rule and the schema compiler's
`j-object`). The parser reads the row; `forced_call_grammar` reads the same row,
so the constraint produces exactly the block the parser extracts (ADR-0063
decision 7).

**2. Detection is a literal scan of the template's source.** Jinja comments are
removed first — a format that appears only in `{# … #}` is documentation, not
something the template emits. Rows are ordered most specific first and the first
row whose anchors all appear wins. The rows this record ships are unambiguous;
an explicit CLI override is a later addition that does not change the table, and
an ambiguous or unrecognised template is a named `400` for `tool_choice` rather
than a guess. The end-to-end template that used to hide `<tool_call>` in a
comment now puts it in an unreachable `{% if false %}` branch, which keeps the
prompt bytes clean without relying on a comment.

**3. Parsing falls back to the Qwen row when detection finds nothing.** #614
decision 8 says parsing does not depend on `tools` having been sent, and the
existing end-to-end extraction test forces a `<tool_call>` block on a template
that has no marker. Parsing therefore uses the detected row when there is one
and the Qwen row otherwise; the fallback is **parsing only**. The `tool_choice`
constraint does not fall back: no row means a named `400`.

**4. A row with no grammar shape is a named `400`.** `forced_call_grammar`
returns `Option<String>`; `None` (a recognised format that cannot be constrained
yet) reaches the handler's named refusal. It is never an accept-and-ignore.

**5. Families are data, not parser branches.** The block shape lands with the
row that needs it: Qwen's "one JSON value then a literal close" today, Mistral's
array and Llama's "next open or EOS" with their rows in #699's second increment.
The state machine stays one incremental parser; it reads the row's open/close
literals and key names instead of branching per family.

**6. Out of scope for this record.** The XML body (#708), the name-in-anchor
`[ARGS]` form (#709) and Gemma 4's lenient JSON (#710) need shapes this first
row set does not have; they are separate records/issues, not hidden branches.

## Alternatives considered

- **Regex anchors with partial matching, as transformers does.** Rejected: Rust's
  `regex` has no partial-match API, and the parser's hand-rolled holdback is what
  makes a tag split across streaming chunks safe. Literal anchors cover the rows
  this record ships; a regex anchor can be added when a format needs one.
- **One parser branch per family.** Rejected: the differences are data (keys,
  literals, close rule), and branches would multiply the close/EOS logic in the
  one state machine that must stay incremental.
- **Interpret transformers' full `response_template`.** Rejected: it is a second
  parser/grammar language (`open_pattern`, `transform`, `join`, `transform_each`)
  for facts this engine can carry as a few literals; the fixture may still be
  authored in that form to get the reference.
- **Parse only when a row is detected.** Rejected: it would stop parsing the
  forced `<tool_call>` completion on the template that has no marker, a
  behaviour both #614 and its end-to-end test rely on.
- **An explicit `--tool-format` as the only selector.** Rejected for now: it asks
  every custom template to pass a flag, where the shipped rows are detectable
  from source. The override remains the escape hatch if a real template turns out
  ambiguous.
- **Detect by model architecture.** Rejected: the GGUF's architecture is not the
  template's family (a Qwen checkpoint can carry any template, and the
  `--chat-template` override is a first-class path), and the template is the
  thing that tells the model which syntax to write.

## Consequences

- `src/server/toolcall/format.rs` becomes the place later work obeys; a new
  family is a row plus its fixture and (when it has one) its grammar shape.
- The Qwen fixture is byte-identical: the row carries exactly the literals and
  keys the constants did. The only user-visible change is the `tool_choice`
  refusal's wording, which now names a missing/unknown format rather than a
  missing `<tool_call>` block.
- Detection is a heuristic. A template that emits two formats, or one whose body
  is not a JSON object, is not covered; the first is resolved by row order and
  recorded here, the second is a named `400`.
- The `tool_choice` grammar cannot cover every row: a `grammar: None` row is a
  named `400`, which is the honest boundary until a second grammar compiler (the
  XML case) exists.
- The row set is deliberately small. Qwen3.5, Ministral 3 and Gemma 4 stay open
  (#708, #709, #710) so this record does not become a partial parser for
  formats nobody can load yet.

## References

- `src/server/toolcall/format.rs` — the table, `ToolFormat`, `GrammarShape`,
  `detect`.
- `src/server/toolcall.rs` — the incremental parser and `forced_call_grammar`,
  both row-driven.
- `src/server/mod.rs` — detection, the parsing fallback, and the named refusals.
- `tests/template_e2e.rs` — the extraction and `tool_choice` end-to-end gates.
- `tests/fixtures/chat/parse_qwen2.5.json` — the unchanged Qwen reference.
- `scripts/adr-governs.toml` — the `ToolFormat` binding.
- ADR-0063 (the extraction this amends); issues #614, #699, #708, #709, #710.
