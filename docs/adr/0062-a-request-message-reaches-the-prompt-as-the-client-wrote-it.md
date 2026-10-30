# 0062. A request message reaches the prompt as the client wrote it, and a replayed tool call that reaches nothing refuses by name

- Status: Accepted
- Date: 2026-10-30
- Issues: #613
- Amends: ADR-0019 (its refusal contract now covers a message object, not only a request field)

## Context

ADR-0058 carried a request's `tools` into the prompt as the client wrote them, and left the
message object alone: `ChatMessage` was `{role, content}` and the render path rebuilt every
message as a two-key object. ADR-0058 recorded that gap as a known limit and named
[#613](https://github.com/yusiwen/minfer/issues/613) as the follow-up. Two consequences were
silent.

- **A replayed assistant turn that carries `tool_calls` could not be expressed.** The Qwen2.5
  template's `{%- for tool_call in message.tool_calls %}` branch therefore never executed, so a
  tool round-trip replayed by a client rendered the assistant's `content` instead of its
  `<tool_call>` block — a prompt no other implementation produces, and one the model was not
  trained on.
- **The client's key order inside a message was lost**, and not in the direction the ticket first
  guessed: `serde_json`'s map is a `BTreeMap` here (its `preserve_order` feature is deliberately
  off, ADR-0058 decision 3), so the rebuilt object emitted `{"content": …, "role": …}` whichever
  order the client wrote. A template that does `message | tojson` produced different bytes from
  transformers.

A third consequence was not in the ticket and surfaced while building it: with the carrier in
place, a template *without* a `tool_calls` branch (or a GGUF with no template at all) would
accept the call, render the assistant turn as its content, and answer `200` — the silent
substitution ADR-0019 forbids, one message deep.

## Decision

**1. A message is carried, not modelled.** The request's `messages` array is deserialized into
`template::ToolJson` — the same order-preserving carrier `tools` uses — and handed to the
template verbatim. The render core takes `&[ToolJson]`. A typed `tool_calls` field on
`ChatMessage` was rejected: it would drop `name` and any vendor key, and `| tojson` echoes
whatever structure the client sent.

**2. The client's order is the render order.** The object is serialized once, through the same
`Value::from_serialize` path `tools` uses (minijinja's `preserve_order`). `serde_json`'s
`preserve_order` stays off: it is process-wide and moves the JSON-Schema→GBNF grammar, which
ADR-0058 measured.

**3. Validation covers the fields the engine reads and the one it must refuse.** `role` must be
one of the five accepted roles, `content` a string (or `null`/absent, the tool-call-turn shape),
and a `tool_calls` array must be non-empty and carry OpenAI's shape —
`{id, type: "function", function: {name, arguments}}` with `arguments` a JSON **string**, because
the template echoes it through `tojson`. Anything else is a `400` naming
`messages[i].tool_calls[j]`. Every other key is passed through untouched; the engine does not
police what a template may read.

**4. A replayed call that reaches nothing is a refusal.** The check is mechanical, like the tools
one: render the conversation as given, then render it with every `tool_calls` key stripped, and
equal bytes mean the call reached nothing — `400`, kind `tool calls not rendered`, naming the
missing branch or the ChatML fallback. A stripped render that *fails outright* counts as "the
template depends on the call" (Qwen2.5 concatenates `message.content`, which is null for a
call-only turn). The check is scoped to `tool_calls`: a vendor key a template ignores is the
template's business, and refusing it would reject `name` and extensions that are legitimately
invisible.

**5. The `(role, content)` form remains the API for callers with no request JSON.** The CLI, the
conversation session and the single-turn path build the object through one adapter, in the key
order this engine has always emitted, so the 28 tools-free reference cases and `format_single`'s
two-render diff are untouched. Carrying an object is a request-path property, not a new
conversation model.

**6. Response-side extraction is still out of scope.** Parsing `<tool_call>` out of the model's
output is [#614](https://github.com/yusiwen/minfer/issues/614); when it lands, the response's
`tool_calls` should use the same carrier so a client can replay what it received without the
engine re-spelling it.

## Alternatives considered

- **Add `tool_calls: Option<Vec<ToolCall>>` to `ChatMessage`.** Rejected: it models one key and
  drops the rest, and `| tojson` would echo the engine's spelling of the call rather than the
  client's — the exact failure ADR-0058 decision 1 rejected for `tools`.
- **Enable `serde_json`'s `preserve_order` and keep a typed message.** Rejected on ADR-0058's
  measurement: the flag is process-wide and moves the grammar feature's accepted language.
- **Inspect the template text for a `tool_calls` branch.** Rejected for the same reason the tools
  check is mechanical: a branch can be guarded on a value this engine does not pass, or render an
  empty block, and only the output distinguishes those from a working branch.
- **Refuse every message key the template does not render.** Rejected: `name`, `tool_call_id` and
  vendor extensions are legitimately invisible to most templates; refusing them would break
  ordinary requests. The semantic loss that matters is a dropped call.
- **Render a branchless template as the assistant's content (the status quo).** Rejected:
  ADR-0019 — a `200` that silently drops the call is indistinguishable from the model refusing
  to use it.

## Consequences

- A replayed tool round-trip renders transformers' bytes: the tools fixture gains a fifth case
  (user → assistant with `tool_calls` and null content → tool → generation prompt), and the four
  earlier cases are byte-identical before and after.
- `ChatCompletionRequest.messages` is `Vec<template::ToolJson>`; a consumer that needs a field
  reads it from the object, and the `(role, content)` adapter exists only for callers with no
  request JSON.
- A client that replays a call against a template without a `tool_calls` branch now gets a `400`
  instead of a `200` with the call dropped — a deliberate behaviour change, and the same contract
  `tools` already has.
- A request carrying calls pays one extra render for the check (the tools check already paid
  two when tools are present).
- `tool_calls: []` is refused like `tools: []`: "no calls" is the absence of the field.
- Known limits, recorded rather than hidden: multimodal `content` arrays stay refused;
  `arguments` is carried as text and not JSON-parsed, so a malformed arguments string reaches the
  model as written; and extraction from the output remains #614.

## References

- `src/template.rs` — `render_at`, `render_checked_at`, `carries_tool_calls`,
  `strip_tool_calls`, `message_object`, `TemplateError::tool_calls_not_rendered`.
- `src/server/types.rs` — `messages`, `validate_message`, `validate_tool_call`.
- `src/server/mod.rs` — the handler's single render call.
- `tests/fixtures/chat/tools_qwen2.5-0.5b-instruct.json` + `scripts/gen_tools_fixture.py`.
- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` §2.7 and §4.
- Issues: [#613](https://github.com/yusiwen/minfer/issues/613); follow-up
  [#614](https://github.com/yusiwen/minfer/issues/614) (output extraction).
