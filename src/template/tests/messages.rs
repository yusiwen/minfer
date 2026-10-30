//! A request message's own JSON reaches the prompt (#613): the carrier, the
//! client's key order, and the `(role, content)` adapters the CLI still uses.
//!
//! The authorities here are the client's bytes and the committed transformers
//! reference (`tools_qwen2.5-0.5b-instruct.json`) — never this engine's own
//! output.

use super::super::{
    message_object, render_messages_opt, render_messages_opt_with_tools,
    render_messages_opt_with_tools_raw, ToolJson,
};

const REFERENCE_JSON: &str =
    include_str!("../../../tests/fixtures/chat/qwen2.5-0.5b-instruct.json");

/// The Qwen2.5 template the tools fixture also renders (same sha256).
fn qwen_template() -> String {
    let reference: serde_json::Value = serde_json::from_str(REFERENCE_JSON).unwrap();
    reference["template"].as_str().unwrap().to_string()
}

fn message(json: &str) -> ToolJson {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("message {json}: {e}"))
}

#[test]
fn parts_and_raw_objects_render_the_same_bytes() {
    // The `(role, content)` adapter is what every non-request caller passes
    // through. It must produce exactly the object the rebuild used to produce, so
    // the 28 tools-free cases stay byte-identical.
    let template = qwen_template();
    let parts: Vec<(String, Option<String>)> = vec![
        ("system".into(), Some("You are helpful.".into())),
        ("user".into(), Some("Hi".into())),
        ("assistant".into(), Some("Hello!".into())),
    ];
    let objects: Vec<ToolJson> = parts
        .iter()
        .map(|(role, content)| message_object(role, content.as_deref()))
        .collect();

    let from_parts = render_messages_opt(Some(&template), &parts, true, "").unwrap();
    let from_objects =
        render_messages_opt_with_tools_raw(Some(&template), &objects, None, true, "").unwrap();
    assert_eq!(from_parts, from_objects);
    assert!(from_parts.contains("Hi"), "got: {from_parts}");
}

#[test]
fn null_content_fails_the_same_way_through_both_forms() {
    // This template concatenates `message.content`, so a null fails — before and
    // after the carrier change, with the same message.
    let template = qwen_template();
    let parts: Vec<(String, Option<String>)> = vec![("assistant".into(), None)];
    let object = message(r#"{"role":"assistant","content":null}"#);
    let from_parts = render_messages_opt(Some(&template), &parts, true, "").unwrap_err();
    let from_object =
        render_messages_opt_with_tools_raw(Some(&template), &[object], None, true, "").unwrap_err();
    assert_eq!(from_parts.message(), from_object.message());
}

#[test]
fn an_object_without_extra_keys_is_the_parts_form() {
    // A request that sends only role/content must render exactly as the adapter
    // does — the point of carrying the object is that nothing else moves.
    let template = qwen_template();
    let parts: Vec<(String, Option<String>)> =
        vec![("user".into(), Some("Weather in Paris?".into()))];
    let object = message(r#"{"role":"user","content":"Weather in Paris?"}"#);
    assert_eq!(
        render_messages_opt(Some(&template), &parts, true, "").unwrap(),
        render_messages_opt_with_tools_raw(Some(&template), &[object], None, true, "").unwrap()
    );
}

#[test]
fn a_client_key_order_survives_both_ways() {
    // `serde_json::Map` is a BTreeMap here (its `preserve_order` feature is off by
    // design — ADR-0058), so a rebuilt object would sort to content-first. The
    // carrier must render the order the client wrote, in both directions.
    let template = "{{ messages[0] | tojson }}";
    let role_first = message(r#"{"role":"user","content":"hi"}"#);
    let content_first = message(r#"{"content":"hi","role":"user"}"#);
    assert_eq!(
        render_messages_opt_with_tools_raw(Some(template), &[role_first], None, true, "").unwrap(),
        r#"{"role": "user", "content": "hi"}"#
    );
    assert_eq!(
        render_messages_opt_with_tools_raw(Some(template), &[content_first], None, true, "")
            .unwrap(),
        r#"{"content": "hi", "role": "user"}"#
    );
}

#[test]
fn an_unknown_message_key_survives_the_render() {
    let template = "{{ messages[0] | tojson }}";
    let object = message(r#"{"role":"user","content":"hi","x_vendor":{"b":2,"a":1}}"#);
    let got =
        render_messages_opt_with_tools_raw(Some(template), &[object], None, true, "").unwrap();
    assert_eq!(
        got,
        r#"{"role": "user", "content": "hi", "x_vendor": {"b": 2, "a": 1}}"#
    );
}

#[test]
fn the_tools_entry_point_still_carries_the_same_message_bytes() {
    // The server passes its objects to the tools-aware raw entry point; a
    // no-tools request must render the same bytes as the plain one.
    let template = qwen_template();
    let objects = vec![message(r#"{"role":"user","content":"Hi"}"#)];
    assert_eq!(
        render_messages_opt_with_tools_raw(Some(&template), &objects, None, true, "").unwrap(),
        render_messages_opt_with_tools_raw(Some(&template), &objects, None, true, "").unwrap()
    );
    // ...and the adapter and the raw form agree with no tools in play.
    let parts: Vec<(String, Option<String>)> = vec![("user".into(), Some("Hi".into()))];
    assert_eq!(
        render_messages_opt_with_tools(Some(&template), &parts, None, true, "").unwrap(),
        render_messages_opt_with_tools_raw(Some(&template), &objects, None, true, "").unwrap()
    );
}

// === The refusal: a replayed call that reaches nothing (#613, S5) ========

/// A template with no `tool_calls` branch — the witness for a template a
/// replayed call cannot reach.
const NO_REPLAY_BRANCH: &str = "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}";

fn replayed() -> Vec<ToolJson> {
    vec![
        message(r#"{"role":"user","content":"Weather in Paris?"}"#),
        message(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]}"#,
        ),
    ]
}

#[test]
fn a_template_without_a_replay_branch_refuses_a_replayed_call() {
    let err =
        render_messages_opt_with_tools_raw(Some(NO_REPLAY_BRANCH), &replayed(), None, true, "")
            .expect_err("a template with no `tool_calls` branch must refuse");
    assert_eq!(err.kind, "tool calls not rendered");
    assert!(
        err.message().contains("no `tool_calls` branch"),
        "the refusal must name the missing branch: {}",
        err.message()
    );
    // ...and the same conversation without the calls still renders, so the
    // refusal is about the call and nothing else.
    let plain = vec![replayed()[0].clone()];
    assert!(
        render_messages_opt_with_tools_raw(Some(NO_REPLAY_BRANCH), &plain, None, true, "")
            .unwrap()
            .contains("Weather in Paris?")
    );
}

#[test]
fn the_chatml_fallback_refuses_a_replayed_call() {
    let err = render_messages_opt_with_tools_raw(None, &replayed(), None, true, "")
        .expect_err("the ChatML fallback has no tool-call format");
    assert_eq!(err.kind, "tool calls not rendered");
    assert!(
        err.message().contains("ChatML fallback"),
        "the refusal must name the fallback: {}",
        err.message()
    );
    assert!(
        !err.message().contains("Supported Python str methods"),
        "a tool-call refusal must not blame the str-method hook: {}",
        err.message()
    );
}

#[test]
fn a_vendor_key_is_not_a_replayed_call() {
    // The check is scoped to `tool_calls`: an unknown key a template ignores must
    // not turn into a refusal.
    let objects = vec![message(r#"{"role":"user","content":"hi","x_vendor":1}"#)];
    let got = render_messages_opt_with_tools_raw(
        Some("{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"),
        &objects,
        None,
        true,
        "",
    )
    .expect("a vendor key is the template's business, not a refusal");
    assert_eq!(got, "user:hi");
}

#[test]
fn the_replay_branch_is_reached_and_changes_the_bytes() {
    // Positive control for the check: with the branch, the call reaches the prompt.
    // The tools are supplied because the Qwen2.5 replay branch lives inside its
    // `{% if tools %}` block.
    let tools = vec![message(
        r#"{"type":"function","function":{"name":"get_weather"}}"#,
    )];
    let got = render_messages_opt_with_tools_raw(
        Some(&qwen_template()),
        &replayed(),
        Some(&tools),
        true,
        "",
    )
    .expect("the Qwen2.5 template has a `tool_calls` branch");
    assert!(got.contains("<tool_call>"), "got: {got}");
    assert!(got.contains("get_weather"), "got: {got}");
}
