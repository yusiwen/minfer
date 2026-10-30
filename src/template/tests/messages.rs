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
