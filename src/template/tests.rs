//! `#[cfg(test)] mod tests` for `src/template.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

fn msgs() -> Vec<(String, Option<String>)> {
    vec![
        ("system".into(), Some("You are helpful.".into())),
        ("user".into(), Some("Hi".into())),
        ("assistant".into(), Some("Hello!".into())),
    ]
}

fn m(role: &str, content: &str) -> (String, Option<String>) {
    (role.to_string(), Some(content.to_string()))
}

#[test]
fn render_messages_with_jinja_template() {
    let tmpl = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
    let out = render_messages(tmpl, &msgs(), true, "<|endoftext|>").unwrap();
    assert_eq!(
        out,
        "<system>You are helpful.</system><user>Hi</user><assistant>Hello!</assistant><assistant>"
    );
}

#[test]
fn render_messages_null_content_emits_role_only() {
    // null content must not panic and must preserve the role marker.
    // (How the template renders a null is template-dependent — the contract
    // is that all messages are passed through.)
    let tmpl = "{% for m in messages %}[{{ m['role'] }}:{{ m['content'] }}]{% endfor %}";
    let mut msgs = msgs();
    msgs.push(("assistant".into(), None)); // tool-call turn
    let out = render_messages(tmpl, &msgs, false, "").unwrap();
    assert!(out.starts_with("[system:You are helpful.]"), "got: {out}");
    assert!(out.contains("[user:Hi]"), "got: {out}");
    assert!(out.contains("[assistant:Hello!]"), "got: {out}");
    assert!(
        out.contains("[assistant:"),
        "null-content role marker present: {out}"
    );
}

// === The loud-refusal contract (F7 #50, design §2.3) ===
//
// Before F7 both of these rendered a generic ChatML prompt and only printed
// a warning. The gate is that they are errors which *name* the construct;
// the mutation check (removing the unknown-method hook) makes the second
// test fail, and reverting makes it pass.

#[test]
fn invalid_template_is_a_loud_error() {
    let err = render_messages("{{ bad", &msgs(), true, "").unwrap_err();
    let msg = err.message();
    assert_eq!(err.kind, "syntax error");
    assert!(
        msg.contains("refuses to fall back"),
        "refusal must say so: {msg}"
    );
    assert!(
        !msg.contains("<|im_start|>"),
        "the refusal must not contain a rendered prompt: {msg}"
    );
    eprintln!("refusal (syntax error):\n{msg}");
}

#[test]
fn unsupported_str_method_is_a_loud_error_naming_the_construct() {
    let tmpl = "{% for m in messages %}{{ m['content'].splitlines() }}{% endfor %}";
    let err = render_messages(tmpl, &msgs(), false, "").unwrap_err();
    let msg = err.message();
    assert!(
        msg.contains("splitlines"),
        "the refusal must name the construct: {msg}"
    );
    assert!(
        msg.contains("refuses to fall back"),
        "refusal must say so: {msg}"
    );
    assert_eq!(
        err.line,
        Some(1),
        "the template line must be reported: {msg}"
    );
    // Shown by `cargo test -- --nocapture`; this is the user-facing text.
    eprintln!("refusal (unsupported method):\n{msg}");
}

#[test]
fn template_raise_exception_is_a_refusal_not_a_fallback() {
    let tmpl = "{{ raise_exception('this template refuses a null system prompt') }}";
    let err = render_messages(tmpl, &msgs(), false, "").unwrap_err();
    assert!(
        err.message().contains("refuses a null system prompt"),
        "the template's own refusal text must survive: {}",
        err.message()
    );
}

#[test]
fn validate_refuses_an_unrenderable_template_before_any_request() {
    assert!(validate("{{ messages[0]['content'] }}").is_ok());
    let err =
        validate("{% for m in messages %}{{ m['content'].splitlines() }}{% endfor %}").unwrap_err();
    assert!(err.message().contains("splitlines"), "{}", err.message());
}

#[test]
fn chatml_fallback_applies_only_when_there_is_no_template() {
    // Some(t) that cannot render -> refusal (tested above); None -> ChatML.
    let out = render_messages_opt(None, &msgs(), true, "").unwrap();
    assert_eq!(
        out,
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>assistant\n"
    );
    // A null content keeps the role marker and is not stringified.
    let mut msgs = msgs();
    msgs.push(("assistant".into(), None));
    let out = render_messages_opt(None, &msgs, false, "").unwrap();
    assert!(out.contains("<|im_start|>assistant\n<|im_end|>"), "{out}");
    assert!(!out.contains("None"), "null must not be stringified: {out}");
}

#[test]
fn render_template_single_user_unchanged() {
    // CLI path still renders a single user message via the template
    let tmpl = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:";
    let out = render_template(tmpl, "hello", true, "").unwrap();
    assert_eq!(out, "user: hello\nassistant:");
}

// === format_single (incremental diff rendering, CLI-CONVERSATION-PLAN.md §5.3) ===

/// Qwen2.5-style ChatML template (a newline after each message, ending with an assistant header).
const QWEN_CHATML: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

#[test]
fn format_single_diffs_only_new_user_message() {
    // History [system, user, assistant] + new user → delta contains only the new message
    // and the assistant header; it must not repeat the already-generated assistant content.
    let past = vec![
        m("system", "You are helpful."),
        m("user", "hi"),
        m("assistant", "Hello!"),
    ];
    let d = format_single(
        Some(QWEN_CHATML),
        &past,
        m("user", "what is 2+2?"),
        true,
        "",
    )
    .unwrap();
    assert!(
        d.prefix_matched,
        "prefix must match for a deterministic template"
    );
    assert_eq!(
        d.text,
        "\n<|im_start|>user\nwhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    );
    // Invariant: KV prefix + delta == fmt_new (canonical full rendering).
    // KV prefix = fmt_past minus the newline the template emits after the last message;
    // the model doesn't emit that newline after EOG, so compensation prepends it to delta.
    let fmt_past = render_messages(QWEN_CHATML, &past, false, "").unwrap();
    let kv_prefix = fmt_past.strip_suffix('\n').unwrap_or(&fmt_past);
    let mut all = past.clone();
    all.push(m("user", "what is 2+2?"));
    let fmt_new = render_messages(QWEN_CHATML, &all, true, "").unwrap();
    assert_eq!(format!("{kv_prefix}{}", d.text), fmt_new);
}

#[test]
fn format_single_no_trailing_newline_no_compensation() {
    // Template produces no trailing '\n' → no compensation.
    let tmpl = "{% for mm in messages %}[{{ mm['role'] }}:{{ mm['content'] }}]{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
    let past = vec![m("user", "hi"), m("assistant", "Hello!")];
    let d = format_single(Some(tmpl), &past, m("user", "Q"), true, "").unwrap();
    assert!(d.prefix_matched);
    assert_eq!(d.text, "[user:Q]<assistant>");
}

#[test]
fn format_single_empty_history_returns_full_new() {
    let d = format_single(Some(QWEN_CHATML), &[], m("user", "first"), true, "").unwrap();
    assert!(d.prefix_matched);
    let expect = render_messages(QWEN_CHATML, &[m("user", "first")], true, "").unwrap();
    assert_eq!(d.text, expect);
}

#[test]
fn format_single_prefix_mismatch_falls_back_to_full() {
    // Non-deterministic template (reverse) → fmt_new doesn't start with fmt_past → full text + flag.
    let tmpl = "{% for mm in messages|reverse %}[{{ mm['role'] }}]{% endfor %}";
    let past = vec![m("user", "hi"), m("assistant", "Hello!")];
    let d = format_single(Some(tmpl), &past, m("user", "Q"), false, "").unwrap();
    assert!(!d.prefix_matched);
    let mut all = past.clone();
    all.push(m("user", "Q"));
    let expect = render_messages(tmpl, &all, false, "").unwrap();
    assert_eq!(d.text, expect, "mismatch must return the full re-render");
}

#[test]
fn format_single_fallback_without_template() {
    // template=None → rendered via the ChatML fallback; same prefix/compensation semantics.
    let past = vec![m("user", "hi"), m("assistant", "Hello!")];
    let d = format_single(None, &past, m("user", "Q"), true, "").unwrap();
    assert!(d.prefix_matched);
    assert_eq!(
        d.text,
        "\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn format_single_null_content_history() {
    // An assistant message with content=None (tool-call turn) must not break the diff.
    let tmpl = "{% for mm in messages %}[{{ mm['role'] }}]{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
    let past = vec![
        m("user", "hi"),
        ("assistant".to_string(), None),
        m("user", "again"),
    ];
    let d = format_single(Some(tmpl), &past, m("assistant", "ok"), false, "").unwrap();
    assert!(d.prefix_matched);
    assert_eq!(d.text, "[assistant]");
}

// === Python str-method semantics, checked against CPython ================
//
// The expectation table is generated by CPython itself
// (/tmp/f7-make-fixtures.py; see the fixture's provenance) so the engine is
// never its own reference.

const PYSTR_JSON: &str = include_str!("../../tests/fixtures/chat/pystr_semantics.json");

fn py_arg(v: &serde_json::Value) -> PyArg {
    match v {
        serde_json::Value::String(s) => PyArg::Str(s.clone()),
        serde_json::Value::Number(n) => PyArg::Int(n.as_i64().unwrap()),
        serde_json::Value::Null => PyArg::None_,
        serde_json::Value::Array(a) => {
            PyArg::StrList(a.iter().map(|x| x.as_str().unwrap().to_string()).collect())
        }
        other => panic!("unsupported fixture argument {other}"),
    }
}

#[test]
fn python_str_methods_match_cpython() {
    let fx: serde_json::Value = serde_json::from_str(PYSTR_JSON).expect("pystr fixture");
    let entries = fx["entries"].as_array().expect("entries");
    assert!(entries.len() >= 24, "fixture shrank: {}", entries.len());
    for e in entries {
        let method = e["method"].as_str().unwrap();
        let recv = e["receiver"].as_str().unwrap();
        let args: Vec<PyArg> = e["args"].as_array().unwrap().iter().map(py_arg).collect();
        let got = py_str_method(method, recv, &args)
            .unwrap_or_else(|| panic!("{method} not implemented"))
            .unwrap_or_else(|e| panic!("{method}({recv:?}) errored: {e}"));
        assert_eq!(
            got.to_json(),
            e["expected"],
            "{recv:?}.{method}({args:?}) must match CPython"
        );
    }
}

#[test]
fn unknown_str_method_is_reported_as_unsupported() {
    assert!(py_str_method("splitlines", "a\nb", &[]).is_none());
    assert!(py_str_method("encode", "x", &[]).is_none());
}

#[test]
fn python_split_edge_cases() {
    // Empty separator is a Python ValueError, not a silent split.
    assert!(py_str_method("split", "abc", &[PyArg::Str(String::new())])
        .unwrap()
        .is_err());
    // maxsplit = 0 returns the whole string.
    assert_eq!(
        py_str_method("split", "a b c", &[PyArg::None_, PyArg::Int(0)])
            .unwrap()
            .unwrap()
            .to_json(),
        serde_json::json!(["a b c"])
    );
}

// === Reference renderings, per model (the ticket's first acceptance) =====
//
// Generated by transformers 5.17.0 from the model's own `chat_template` in
// its `tokenizer_config.json` (see each fixture's provenance). The engine
// renders the *same template text* here; the real-model gate in
// conversation.rs additionally renders the template the GGUF artifact
// carries and requires the same bytes.

const CHAT_FIXTURES: &[(&str, &str)] = &[
    (
        "qwen2.5-0.5b-instruct",
        include_str!("../../tests/fixtures/chat/qwen2.5-0.5b-instruct.json"),
    ),
    (
        "qwen2.5-7b-instruct",
        include_str!("../../tests/fixtures/chat/qwen2.5-7b-instruct.json"),
    ),
    (
        "qwen2.5-14b-instruct",
        include_str!("../../tests/fixtures/chat/qwen2.5-14b-instruct.json"),
    ),
    (
        "qwen3-0.6b",
        include_str!("../../tests/fixtures/chat/qwen3-0.6b.json"),
    ),
];

#[test]
fn model_templates_render_byte_for_byte() {
    for (slug, raw) in CHAT_FIXTURES {
        let fx: serde_json::Value =
            serde_json::from_str(raw).unwrap_or_else(|e| panic!("{slug} fixture: {e}"));
        // Provenance must stay attached to the fixture.
        for key in ["model", "revision", "template_sha256", "template"] {
            assert!(!fx[key].is_null(), "{slug}: fixture lacks {key}");
        }
        let prov = &fx["provenance"];
        assert!(
            prov["reference"].as_str().unwrap().contains("transformers"),
            "{slug}: reference generator must be named"
        );
        assert!(
            prov["template_source"]
                .as_str()
                .unwrap()
                .contains("tokenizer_config.json"),
            "{slug}: template source must be named"
        );
        assert!(!prov["command"].as_str().unwrap().is_empty());
        assert!(!prov["date"].as_str().unwrap().is_empty());

        let template = fx["template"].as_str().unwrap();
        let bos = fx["bos_token"].as_str().unwrap_or("");
        let cases = fx["cases"].as_array().unwrap();
        assert!(cases.len() >= 6, "{slug}: fixture shrank");
        let mut names = Vec::new();
        for case in cases {
            let name = case["name"].as_str().unwrap();
            names.push(name.to_string());
            let msgs: Vec<(String, Option<String>)> = case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (
                        m["role"].as_str().unwrap().to_string(),
                        m["content"].as_str().map(str::to_string),
                    )
                })
                .collect();
            let add_gen = case["add_generation_prompt"].as_bool().unwrap();
            let expected = case["expected"].as_str().unwrap();
            let got = render_messages(template, &msgs, add_gen, bos)
                .unwrap_or_else(|e| panic!("{slug}/{name} refused: {}", e.message()));
            assert_eq!(
                got, expected,
                "{slug}/{name}: rendered prompt differs from the transformers reference\n\
                 got:      {got:?}\n\
                 expected: {expected:?}"
            );
        }
        // The ticket asks explicitly for a multi-turn conversation, a system
        // message and a generation prompt: require all three in every fixture.
        assert!(names.iter().any(|n| n == "multi_turn"), "{slug}: {names:?}");
        assert!(
            names.iter().any(|n| n == "system_and_user"),
            "{slug}: {names:?}"
        );
        assert!(
            names.iter().any(|n| n.contains("generation_prompt")),
            "{slug}: {names:?}"
        );
    }
}

#[test]
fn qwen3_reasoning_turn_is_extracted_like_transformers() {
    // The construct that motivated F7: the template's own split/lstrip/rstrip
    // calls must produce transformers' output, not a ChatML prompt.
    let fx: serde_json::Value = serde_json::from_str(CHAT_FIXTURES[3].1).unwrap();
    let case = fx["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "reasoning_assistant_replay")
        .expect("fixture case");
    let expected = case["expected"].as_str().unwrap();
    assert!(
        expected.contains("<think>"),
        "the reference must exercise the think block: {expected:?}"
    );
    let msgs: Vec<(String, Option<String>)> = case["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["role"].as_str().unwrap().to_string(),
                m["content"].as_str().map(str::to_string),
            )
        })
        .collect();
    let got = render_messages(
        fx["template"].as_str().unwrap(),
        &msgs,
        case["add_generation_prompt"].as_bool().unwrap(),
        fx["bos_token"].as_str().unwrap_or(""),
    )
    .unwrap();
    assert_eq!(got, expected);
    assert!(
        expected.contains("\n  Let me count"),
        "the reference must show the normalized reasoning block: {expected:?}"
    );
}

// === The template the GGUF artifact carries (real-model gate) ===========
//
// The CI gate above renders the *published* `tokenizer_config.json`
// template. This gate renders the copy inside the GGUF the engine actually
// loads and requires the same bytes. For Qwen2.5 and Qwen3 those two texts
// differ (the converter rewrote a tool-call string and, for Qwen3, an older
// template revision with an equivalent `range(...)` loop instead of
// `messages[::-1]`), so the assertion is on the rendered bytes, which is
// what the ticket asks for.

#[test]
#[ignore = "requires the cached GGUF models (~/.cache/minfer/models)"]
fn gguf_template_renders_like_the_reference() {
    for (slug, raw) in CHAT_FIXTURES {
        let fx: serde_json::Value = serde_json::from_str(raw).unwrap();
        let rel = fx["gguf"].as_str().unwrap_or_default();
        let Some(path) = expand_home(rel) else {
            eprintln!("{slug}: no gguf path in the fixture");
            continue;
        };
        if !path.exists() {
            eprintln!("{slug}: {} not cached; skipping", path.display());
            continue;
        }
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let template = gguf.parts[0]
            .ctx
            .kv
            .iter()
            .find(|kv| kv.key == "tokenizer.chat_template")
            .map(|kv| kv.get_val_str(0).to_string())
            .unwrap_or_else(|| panic!("{slug}: the GGUF carries no chat template"));
        let bos = fx["bos_token"].as_str().unwrap_or("");
        let cases = fx["cases"].as_array().unwrap();
        for case in cases {
            let msgs: Vec<(String, Option<String>)> = case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (
                        m["role"].as_str().unwrap().to_string(),
                        m["content"].as_str().map(str::to_string),
                    )
                })
                .collect();
            let expected = case["expected"].as_str().unwrap();
            let got = render_messages(
                &template,
                &msgs,
                case["add_generation_prompt"].as_bool().unwrap(),
                bos,
            )
            .unwrap_or_else(|e| panic!("{slug}: GGUF template refused: {}", e.message()));
            assert_eq!(
                got,
                expected,
                "{slug}/{}: the GGUF's own template renders differently from the reference",
                case["name"].as_str().unwrap()
            );
        }
        eprintln!(
            "{slug}: {} cases render byte for byte from the GGUF template",
            cases.len()
        );
    }
}

fn expand_home(path: &str) -> Option<std::path::PathBuf> {
    if path.is_empty() {
        return None;
    }
    match path.strip_prefix("~/") {
        Some(rest) => Some(std::path::PathBuf::from(std::env::var_os("HOME")?).join(rest)),
        None => Some(std::path::PathBuf::from(path)),
    }
}
