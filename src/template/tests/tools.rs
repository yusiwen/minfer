//! Tools on the render path (#597): the transformers reference, CPython's
//! `tojson`, and the refusal when a template never puts the definitions into the
//! prompt.
//!
//! Three separate authorities are compared against, never this engine's own
//! output:
//!
//! * `tests/fixtures/chat/tools_qwen2.5-0.5b-instruct.json` — transformers
//!   5.17.0 rendering the Qwen2.5 template's `tools` branch.
//! * `tests/fixtures/chat/tojson_cpython.json` — CPython's `json.dumps`, which is
//!   what the same template's `| tojson` must emit.
//! * `tests/fixtures/chat/qwen2.5-0.5b-instruct.json` — the tools-free reference,
//!   whose template text the tools fixture must share.

use super::super::{civil_from_unix, environment_at};
use super::msgs;
use crate::template::{
    render_messages_opt, render_messages_opt_with_tools, render_messages_opt_with_tools_raw,
    ToolJson,
};
use minijinja::context;

const TOOLS_JSON: &str =
    include_str!("../../../tests/fixtures/chat/tools_qwen2.5-0.5b-instruct.json");
const TOJSON_JSON: &str = include_str!("../../../tests/fixtures/chat/tojson_cpython.json");
const REFERENCE_JSON: &str =
    include_str!("../../../tests/fixtures/chat/qwen2.5-0.5b-instruct.json");

// === CPython's `json.dumps`, which `| tojson` must reproduce ===============

#[derive(serde::Deserialize)]
struct ToJsonFixture {
    provenance: serde_json::Value,
    entries: Vec<ToJsonEntry>,
}

#[derive(serde::Deserialize)]
struct ToJsonEntry {
    /// Deserialized as `ToolJson`, so the fixture's document order survives
    /// `serde_json::Value`'s sorted map — that order is half the contract.
    value: ToolJson,
    /// `None` = the compact form; `Some(n)` = `json.dumps(indent=n)`.
    indent: Option<usize>,
    expected: String,
}

/// Render `value` through the real environment's `tojson` filter.
fn tojson(value: &ToolJson, indent: Option<usize>) -> Result<String, String> {
    let template = match indent {
        Some(n) => format!("{{{{ value | tojson(indent={n}) }}}}"),
        None => "{{ value | tojson }}".to_string(),
    };
    let mut env = environment_at(civil_from_unix(1_792_326_292));
    env.add_template("t", &template)
        .map_err(|e| e.to_string())?;
    env.get_template("t")
        .map_err(|e| e.to_string())?
        .render(context! { value => value })
        .map_err(|e| e.to_string())
}

#[test]
fn tojson_matches_cpython_json_dumps_entry_for_entry() {
    let fx: ToJsonFixture = serde_json::from_str(TOJSON_JSON).expect("tojson fixture");
    assert!(
        fx.provenance["reference"]
            .as_str()
            .unwrap()
            .contains("json.dumps"),
        "the fixture must name CPython as its authority"
    );
    assert_eq!(fx.entries.len(), 9, "fixture shrank");
    let mut sawn = 0;
    for (i, entry) in fx.entries.iter().enumerate() {
        let got =
            tojson(&entry.value, entry.indent).unwrap_or_else(|e| panic!("entry {i} refused: {e}"));
        assert_eq!(
            got, entry.expected,
            "entry {i}: tojson differs from CPython json.dumps({}, indent={:?})",
            entry.expected, entry.indent
        );
        sawn += 1;
    }
    assert_eq!(sawn, fx.entries.len());
}

#[test]
fn tojson_keeps_the_python_separators_and_leaves_html_characters_alone() {
    // The two divergences from minijinja's builtin `tojson`, asserted directly so
    // a switch back to it fails here and not only in the transformers gate:
    // compact separators and `\u003c`-style HTML escaping.
    let mut object = Vec::new();
    object.push(("type".to_string(), ToolJson::String("function".to_string())));
    object.push((
        "description".to_string(),
        ToolJson::String("a <b> & 'c' 检索".to_string()),
    ));
    let got = tojson(&ToolJson::Object(object), None).unwrap();
    assert_eq!(
        got, r#"{"type": "function", "description": "a <b> & 'c' 检索"}"#,
        "spaces after `,`/`:` and no HTML escaping are CPython's bytes"
    );
    assert!(!got.contains("\\u003c"), "got: {got}");
}

#[test]
fn tojson_refuses_a_keyword_it_cannot_honour_by_name() {
    // transformers' `tojson` accepts `sort_keys`/`ensure_ascii`/`separators`; this
    // engine implements neither and must say so rather than render other bytes.
    let err = tojson_with("{{ value | tojson(sort_keys=True) }}").unwrap_err();
    assert!(err.contains("sort_keys"), "got: {err}");
}

fn tojson_with(template: &str) -> Result<String, String> {
    let mut env = environment_at(civil_from_unix(1_792_326_292));
    env.add_template("t", template).map_err(|e| e.to_string())?;
    env.get_template("t")
        .map_err(|e| e.to_string())?
        .render(context! { value => ToolJson::String("x".to_string()) })
        .map_err(|e| e.to_string())
}

// === The ordered carrier ==================================================

#[test]
fn a_tool_definition_keeps_the_key_order_it_was_parsed_from() {
    // The whole point of `ToolJson`: `serde_json::Value` would sort this to
    // `{"a":2,"z":1}` and change the prompt.
    let parsed: ToolJson = serde_json::from_str(r#"{"z":1,"a":{"b":2,"a":3}}"#).unwrap();
    let keys: Vec<&str> = parsed
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(keys, ["z", "a"]);
    let inner = parsed.get("a").unwrap();
    let inner_keys: Vec<&str> = inner
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(inner_keys, ["b", "a"]);
    assert_eq!(
        serde_json::to_string(&parsed).unwrap(),
        r#"{"z":1,"a":{"b":2,"a":3}}"#
    );
}

// === The transformers reference ==========================================

#[derive(serde::Deserialize)]
struct ToolsFixture {
    model: String,
    revision: String,
    template_sha256: String,
    template: String,
    #[serde(default)]
    bos_token: Option<String>,
    provenance: serde_json::Value,
    cases: Vec<ToolsCase>,
}

#[derive(serde::Deserialize)]
struct ToolsCase {
    name: String,
    /// The client's own message objects (#613) — an assistant turn may carry
    /// `tool_calls`, which a `{role, content}` pair cannot express.
    messages: Vec<ToolJson>,
    tools: Vec<ToolJson>,
    add_generation_prompt: bool,
    expected: String,
}

fn tools_fixture() -> ToolsFixture {
    let fx: ToolsFixture = serde_json::from_str(TOOLS_JSON).expect("tools fixture");
    // Provenance stays attached, and the reference must be transformers.
    let prov = &fx.provenance;
    assert!(
        prov["reference"].as_str().unwrap().contains("transformers"),
        "the tools reference must name transformers"
    );
    assert!(
        prov["command"]
            .as_str()
            .unwrap()
            .contains("gen_tools_fixture.py"),
        "the regeneration command must be recorded"
    );
    assert!(!prov["date"].as_str().unwrap().is_empty());
    assert!(!fx.model.is_empty() && !fx.revision.is_empty());
    fx
}

#[test]
fn tools_render_byte_for_byte_against_the_transformers_reference() {
    let fx = tools_fixture();
    // Same template text as the committed tools-free reference: the two fixtures
    // cannot drift into rendering different templates.
    let reference: serde_json::Value = serde_json::from_str(REFERENCE_JSON).unwrap();
    assert_eq!(
        fx.template,
        reference["template"].as_str().unwrap(),
        "the tools fixture must render the same template as the tools-free reference"
    );
    assert_eq!(
        fx.template_sha256,
        reference["template_sha256"].as_str().unwrap()
    );

    let bos = fx.bos_token.as_deref().unwrap_or("");
    let mut cases = 0;
    for case in &fx.cases {
        assert!(
            !case.tools.is_empty(),
            "{}: a case without tools",
            case.name
        );
        // The reference is only meaningful if it actually rendered the tools
        // block, and only *proves* the escaped-character contract if it carries
        // the characters that HTML escaping would rewrite.
        assert!(
            case.expected.contains("<tools>") && case.expected.contains("</tools>"),
            "{}: the reference has no tools block",
            case.name
        );
        // The OpenAI key order must arrive in the prompt in that order: with a
        // sorted JSON map `"function"` would precede `"type"`.
        let type_at = case.expected.find("\"type\": \"function\"");
        let fn_at = case.expected.find("\"function\": {");
        assert!(
            matches!((type_at, fn_at), (Some(t), Some(f)) if t < f),
            "{}: the reference must render the client's key order",
            case.name
        );

        let got = render_messages_opt_with_tools_raw(
            Some(&fx.template),
            &case.messages,
            Some(&case.tools),
            case.add_generation_prompt,
            bos,
        )
        .unwrap_or_else(|e| panic!("{}/{} refused: {}", fx.model, case.name, e.message()));
        assert_eq!(
            got, case.expected,
            "{}/{}: rendered prompt differs from the transformers reference\n\
             got:      {got:?}\n\
             expected: {:?}",
            fx.model, case.name, case.expected
        );
        cases += 1;
    }
    assert_eq!(cases, 5, "all five reference cases must be checked");

    // The replayed tool round-trip (#613): the assistant turn's `tool_calls`
    // reached the template's replay branch, and the arguments string survived as
    // the JSON text the client sent.
    let replay = fx
        .cases
        .iter()
        .find(|c| c.name == "assistant_tool_call_replay")
        .expect("replayed tool-call case");
    assert!(
        replay.expected.contains("<tool_call>") && replay.expected.contains("</tool_call>"),
        "the reference must render the template's replay branch"
    );
    assert!(
        replay.expected.contains(r#"\"city\": \"Paris\""#),
        "the arguments string must reach the prompt as CPython tojson wrote it: {}",
        replay.expected
    );
    // The assistant turn is the template's `assistant` branch with a null content:
    // no fabricated `null` text either side of the block.
    assert!(
        !replay.expected.contains("assistant\nnull"),
        "a null content must not render as the text `null`"
    );

    // The case that pins the escaping contract: raw `<`, `&` and non-ASCII.
    let escaped = fx
        .cases
        .iter()
        .find(|c| c.name == "two_functions_nested_schema")
        .expect("nested-schema case");
    for needle in ["Search <docs> & notes", "检索 ünïcode", "\"default\": 0.5"] {
        assert!(
            escaped.expected.contains(needle),
            "the reference must keep {needle:?} verbatim: {}",
            escaped.expected
        );
    }
}

// === The tools contract: refuse by name, never drop silently =============

fn weather_tool() -> ToolJson {
    serde_json::from_str(
        r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}"#,
    )
    .unwrap()
}

#[test]
fn a_template_without_a_tools_branch_is_refused_by_name() {
    // A template that renders the messages and nothing else: byte-identical with
    // and without `tools`, so the definitions would reach nothing.
    let template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
    let tools = [weather_tool()];
    let err = render_messages_opt_with_tools(Some(template), &msgs(), Some(&tools), true, "")
        .expect_err("a template with no tools branch must refuse");
    assert_eq!(err.kind, "tools not rendered");
    assert!(
        err.message().contains("no `tools` branch"),
        "the refusal must name the missing branch: {}",
        err.message()
    );
    // ...and the same conversation without tools still renders, so the refusal
    // is about the definitions and nothing else.
    let plain = render_messages_opt(Some(template), &msgs(), true, "").unwrap();
    assert!(plain.contains("user: Hi"), "got: {plain}");
}

#[test]
fn the_chatml_fallback_refuses_tools_by_name() {
    // No template at all: the ChatML fallback has no tool format, and the
    // notice the loader prints describes the *template*, not a silently dropped
    // request field.
    let tools = [weather_tool()];
    let err = render_messages_opt_with_tools(None, &msgs(), Some(&tools), true, "")
        .expect_err("the ChatML fallback must refuse tools");
    assert_eq!(err.kind, "tools not rendered");
    assert!(
        err.message().contains("ChatML fallback"),
        "the refusal must name the fallback: {}",
        err.message()
    );
    assert!(
        !err.message().contains("Supported Python str methods"),
        "a tools refusal must not blame the str-method hook: {}",
        err.message()
    );
}

#[test]
fn no_tools_leaves_every_render_byte_identical() {
    // The pinned-`none` behaviour the existing 28-case gate relies on: passing no
    // tools is exactly the old call, including the empty-array reading (an empty
    // array carries no definitions to lose).
    let template = "{% if tools %}T{% endif %}{% for m in messages %}{{ m.role }}{% endfor %}";
    let before = render_messages_opt(Some(template), &msgs(), true, "").unwrap();
    let empty: [ToolJson; 0] = [];
    let after =
        render_messages_opt_with_tools(Some(template), &msgs(), Some(&empty), true, "").unwrap();
    assert_eq!(before, after);
    assert_eq!(before, "systemuserassistant");
}
