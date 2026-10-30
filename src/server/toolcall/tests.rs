//! Unit tests for the tool-call parser (#614). See the parent module for the
//! contract; the transformers reference lives in
//! `tests/fixtures/chat/parse_qwen2.5.json` and is read by
//! `tests::the_transformers_reference_parses_identically`.

use super::*;

const CALL: &str =
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

fn parse(text: &str) -> (String, Vec<ToolCall>) {
    parse_completion(text).unwrap_or_else(|e| panic!("{text:?}: {}", e.message()))
}

fn parse_in_chunks(text: &str, size: usize) -> (String, Vec<ToolCall>) {
    let mut p = Parser::new();
    let mut events = Vec::new();
    for chunk in text.as_bytes().chunks(size) {
        // The parser takes &str; the test inputs are ASCII or split on char
        // boundaries by `str::get`, so a lossy decode is safe here.
        let s = std::str::from_utf8(chunk).unwrap();
        events.extend(p.feed(s));
    }
    events.extend(p.finish().expect("finish"));
    fold(events)
}

fn fold(events: Vec<Event>) -> (String, Vec<ToolCall>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    for e in events {
        match e {
            Event::Content(t) => content.push_str(&t),
            Event::Call(c) => calls.push(c),
        }
    }
    (content, calls)
}

#[test]
fn a_call_becomes_name_and_the_blocks_own_argument_bytes() {
    let (content, calls) = parse(CALL);
    assert_eq!(content, "");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    // The object is carried as written: spaces included.
    assert_eq!(calls[0].arguments, r#"{"city": "Paris"}"#);
}

#[test]
fn the_opening_tag_may_split_at_every_byte() {
    for size in 1..=CALL.len() {
        let (content, calls) = parse_in_chunks(CALL, size);
        assert_eq!(content, "", "chunk size {size}");
        assert_eq!(calls.len(), 1, "chunk size {size}");
        assert_eq!(calls[0].name, "get_weather", "chunk size {size}");
        assert_eq!(
            calls[0].arguments, r#"{"city": "Paris"}"#,
            "chunk size {size}"
        );
    }
}

#[test]
fn text_around_a_call_is_kept_and_the_template_separators_are_not() {
    let (content, calls) = parse(&format!("Let me check.\n{CALL}\nDone."));
    assert_eq!(content, "Let me check.Done.");
    assert_eq!(calls.len(), 1);

    // Two calls separated by a newline: the separator is not content.
    let (content, calls) = parse(&format!("{CALL}\n{CALL}"));
    assert_eq!(content, "");
    assert_eq!(calls.len(), 2);

    // Prose is unchanged, trailing whitespace included.
    let (content, calls) = parse("Hello, world!\n");
    assert_eq!(content, "Hello, world!\n");
    assert!(calls.is_empty());
}

#[test]
fn a_string_argument_is_decoded_and_an_object_keeps_its_bytes() {
    let block = r#"<tool_call>{"name": "f", "arguments": "{\"a\": 1, \"b\": 2}"}</tool_call>"#;
    assert_eq!(parse(block).1[0].arguments, r#"{"a": 1, "b": 2}"#);

    let block = r#"<tool_call>{"name": "f", "arguments": {"a": 1, "b": [2, 3]}}</tool_call>"#;
    assert_eq!(parse(block).1[0].arguments, r#"{"a": 1, "b": [2, 3]}"#);
}

#[test]
fn the_closing_tag_inside_the_arguments_is_not_a_close() {
    let block = r#"<tool_call>{"name": "f", "arguments": {"note": "</tool_call>"}}</tool_call>"#;
    let (content, calls) = parse(block);
    assert_eq!(content, "");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, r#"{"note": "</tool_call>"}"#);
}

#[test]
fn an_extra_key_in_the_call_object_is_ignored() {
    let block = r#"<tool_call>{"id": "call_1", "name": "f", "arguments": {"a": 1}}</tool_call>"#;
    assert_eq!(parse(block).1[0].name, "f");
}

#[test]
fn a_block_that_opens_and_never_closes_is_an_error() {
    // Open tag only.
    assert!(parse_completion("<tool_call>").is_err());
    // JSON cut off.
    assert!(
        parse_completion(r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Par"#)
            .is_err()
    );
    // JSON complete but no closing tag: the arguments are whole, so this is a
    // usable call (transformers closes the region implicitly at EOS too).
    let (content, calls) = parse(r#"<tool_call>{"name": "f", "arguments": {}}"#);
    assert_eq!(content, "");
    assert_eq!(calls.len(), 1);
    // A partial closing tag at EOS is the same case.
    assert_eq!(
        parse(r#"<tool_call>{"name": "f", "arguments": {}}</tool_cal"#)
            .1
            .len(),
        1
    );
    // The error names the block, so the handler's 400 is readable.
    let err = parse_completion("<tool_call>").unwrap_err();
    assert!(
        err.message().contains("tool-call block"),
        "{}",
        err.message()
    );
}

#[test]
fn a_malformed_call_object_is_an_error_not_an_empty_call() {
    for block in [
        r#"<tool_call>{"arguments": {}}</tool_call>"#,
        r#"<tool_call>{"name": "", "arguments": {}}</tool_call>"#,
        r#"<tool_call>{"name": 3, "arguments": {}}</tool_call>"#,
        r#"<tool_call>{"name": "f"}</tool_call>"#,
    ] {
        assert!(parse_completion(block).is_err(), "{block}");
    }
    // Garbage after the JSON is not a silent close either.
    assert!(
        parse_completion(r#"<tool_call>{"name": "f", "arguments": {}} garbage</tool_call>"#)
            .is_err()
    );
}

#[test]
fn unicode_and_html_characters_survive_in_the_arguments() {
    let block = r#"<tool_call>{"name": "f", "arguments": {"q": "检索 <b> & 'c'"}}</tool_call>"#;
    assert_eq!(parse(block).1[0].arguments, r#"{"q": "检索 <b> & 'c'"}"#);
}

// === The transformers reference ==========================================

const PARSE_JSON: &str = include_str!("../../../tests/fixtures/chat/parse_qwen2.5.json");

#[derive(serde::Deserialize)]
struct ParseFixture {
    provenance: serde_json::Value,
    cases: Vec<ParseCase>,
}

#[derive(serde::Deserialize)]
struct ParseCase {
    name: String,
    completion: String,
    expected: Option<ParseExpected>,
    error: Option<String>,
    /// A case where this engine deliberately accepts what the reference refuses;
    /// the reason is recorded in the fixture.
    diverges: Option<String>,
}

#[derive(serde::Deserialize)]
struct ParseExpected {
    content: Option<String>,
    tool_calls: Vec<ExpectedCall>,
}

#[derive(serde::Deserialize)]
struct ExpectedCall {
    name: String,
    arguments: serde_json::Value,
}

#[test]
fn the_transformers_reference_parses_identically() {
    let fx: ParseFixture = serde_json::from_str(PARSE_JSON).expect("parse fixture");
    assert!(
        fx.provenance["reference"]
            .as_str()
            .unwrap()
            .contains("transformers"),
        "the fixture must name transformers as its authority"
    );
    let mut seen = 0;
    for case in &fx.cases {
        let got = parse_completion(&case.completion);
        if let Some(reason) = &case.diverges {
            // Deliberate divergence: the reference's region scan splits at the
            // first `</tool_call>`, which sits inside a string here.
            assert!(!reason.is_empty());
            let (_, calls) = got.unwrap_or_else(|e| panic!("{}: {}", case.name, e.message()));
            assert_eq!(calls.len(), 1, "{}", case.name);
            assert_eq!(
                calls[0].arguments, r#"{"city": "</tool_call>"}"#,
                "{}",
                case.name
            );
            seen += 1;
            continue;
        }
        if case.error.is_some() {
            assert!(
                got.is_err(),
                "{}: the reference refused this completion, so this engine must too",
                case.name
            );
            seen += 1;
            continue;
        }
        let (content, calls) = got.unwrap_or_else(|e| panic!("{}: {}", case.name, e.message()));
        let expected = case
            .expected
            .as_ref()
            .unwrap_or_else(|| panic!("{}", case.name));
        assert_eq!(calls.len(), expected.tool_calls.len(), "{}", case.name);
        for (got, want) in calls.iter().zip(&expected.tool_calls) {
            assert_eq!(got.name, want.name, "{}", case.name);
            match &want.arguments {
                serde_json::Value::String(text) => {
                    assert_eq!(&got.arguments, text, "{}", case.name)
                }
                value => {
                    let parsed: serde_json::Value = serde_json::from_str(&got.arguments)
                        .unwrap_or_else(|e| panic!("{}: arguments are not JSON: {e}", case.name));
                    assert_eq!(&parsed, value, "{}", case.name);
                    assert!(
                        case.completion.contains(&got.arguments),
                        "{}: an object's arguments are the block's own bytes",
                        case.name
                    );
                }
            }
        }
        // The reference strips trailing whitespace from prose; this engine leaves a
        // prose completion untouched (asserted above), so content is compared only
        // where the reference has a call.
        if !expected.tool_calls.is_empty() {
            assert_eq!(
                content,
                expected.content.as_deref().unwrap_or(""),
                "{}",
                case.name
            );
        }
        seen += 1;
    }
    assert_eq!(seen, fx.cases.len());
    assert_eq!(fx.cases.len(), 12, "fixture shrank");
}

// === Streaming: the deltas rebuild the non-streaming message =============

/// Fold SSE payloads into (content, [(id, name, arguments)]), the way a client
/// assembles `delta` chunks.
fn fold_chunks(payloads: &[String]) -> (String, Vec<(String, String, String)>) {
    let mut content = String::new();
    let mut calls: Vec<(String, String, String)> = Vec::new();
    for payload in payloads {
        let v: serde_json::Value = serde_json::from_str(payload).expect("chunk json");
        let delta = &v["choices"][0]["delta"];
        if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
            content.push_str(c);
        }
        if let Some(list) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in list {
                let i = tc["index"].as_u64().unwrap() as usize;
                while calls.len() <= i {
                    calls.push((String::new(), String::new(), String::new()));
                }
                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    calls[i].0 = id.to_string();
                }
                if let Some(name) = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                {
                    calls[i].1 = name.to_string();
                }
                if let Some(args) = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                {
                    calls[i].2.push_str(args);
                }
            }
        }
    }
    (content, calls)
}

#[test]
fn streamed_deltas_rebuild_the_non_streaming_tool_calls() {
    use crate::server::types;
    let text = format!("Let me check.\n{CALL}\n{CALL}");
    let (want_content, want_calls) = parse_completion(&text).unwrap();
    let response = types::build_response(
        "id",
        "m",
        1,
        want_content.clone(),
        "tool_calls",
        1,
        1,
        Some(
            want_calls
                .iter()
                .map(types::ResponseToolCall::new)
                .collect(),
        ),
    );
    let response = serde_json::to_value(&response).unwrap();
    let message = &response["choices"][0]["message"];

    for split in [1usize, 2, 3, 5, 7, 4096] {
        let mut parser = Parser::new();
        let mut index = 0usize;
        let mut payloads = Vec::new();
        for chunk in text.as_bytes().chunks(split) {
            let text = std::str::from_utf8(chunk).unwrap();
            for event in parser.feed(text) {
                payloads.extend(types::chunks_for_tool_event(
                    "id", "m", 1, &mut index, &event,
                ));
            }
        }
        for event in parser.finish().unwrap() {
            payloads.extend(types::chunks_for_tool_event(
                "id", "m", 1, &mut index, &event,
            ));
        }

        let (content, calls) = fold_chunks(&payloads);
        assert_eq!(content, want_content, "split {split}");
        assert_eq!(
            calls.len(),
            message["tool_calls"].as_array().unwrap().len(),
            "split {split}"
        );
        for (i, (id, name, arguments)) in calls.iter().enumerate() {
            let want = &message["tool_calls"][i];
            assert!(id.starts_with("call_"), "split {split}: {id}");
            assert_eq!(name, want["function"]["name"].as_str().unwrap());
            assert_eq!(arguments, want["function"]["arguments"].as_str().unwrap());
        }
    }
}

// === The tool_choice constraint (#614) ==================================

fn byte_pieces() -> (Vec<Option<Box<[u8]>>>, Vec<bool>) {
    let mut pieces: Vec<Option<Box<[u8]>>> = Vec::new();
    for b in 0..=255u16 {
        pieces.push(Some(vec![b as u8].into_boxed_slice()));
    }
    pieces.push(Some(b"<eos>".to_vec().into_boxed_slice()));
    let n = pieces.len();
    let mut eog = vec![false; n];
    eog[n - 1] = true;
    (pieces, eog)
}

#[test]
fn the_forced_call_grammar_wraps_the_json_object_and_names_the_tool() {
    let grammar = forced_call_grammar(&["get_weather", "search_docs"], None);
    assert!(grammar.contains("j-object"), "the JSON prelude is reused");
    assert!(grammar.contains("\"get_weather\"") && grammar.contains("\"search_docs\""));
    let (pieces, eog) = byte_pieces();
    crate::grammar::Grammar::from_gbnf(&grammar, pieces, eog)
        .expect("the required-call grammar compiles");

    let grammar = forced_call_grammar(&["get_weather", "search_docs"], Some("get_weather"));
    assert!(grammar.contains("\"get_weather\""));
    assert!(
        !grammar.contains("\"search_docs\""),
        "a forced name must not offer the other tools"
    );
    let (pieces, eog) = byte_pieces();
    crate::grammar::Grammar::from_gbnf(&grammar, pieces, eog)
        .expect("the forced-call grammar compiles");
}
