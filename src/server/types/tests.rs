//! `#[cfg(test)] mod tests` for `src/server/types.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn parse_valid_request() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_k":20,"stop":["\n\n","User:"]}"#;
    let req = ChatCompletionRequest::parse(body).expect("parse");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.top_k, Some(20));
    match &req.stop {
        Some(StopCondition::Array(v)) => assert_eq!(v.len(), 2),
        _ => panic!("stop should be array"),
    }
}

#[test]
fn parse_stop_sequences_alias() {
    // Anthropic alias "stop_sequences" maps onto `stop`
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"stop_sequences":["a","b"]}"#;
    let req = ChatCompletionRequest::parse(body).expect("parse");
    match &req.stop {
        Some(StopCondition::Array(v)) => assert_eq!(v, &["a".to_string(), "b".to_string()]),
        _ => panic!("stop_sequences alias must populate stop"),
    }
}

#[test]
fn parse_rejects_malformed_json() {
    let err = ChatCompletionRequest::parse(br#"{"messages": ["#).unwrap_err();
    assert_eq!(err.status, 400);
    assert_eq!(err.error_type, "invalid_request_error");
}

#[test]
fn parse_rejects_empty_messages() {
    let err = ChatCompletionRequest::parse(br#"{"messages":[]}"#).unwrap_err();
    assert_eq!(err.status, 400);
}

#[test]
fn parse_rejects_unknown_role() {
    let err = ChatCompletionRequest::parse(br#"{"messages":[{"role":"pirate","content":"yo"}]}"#)
        .unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("pirate"));
}

#[test]
fn parse_rejects_array_content() {
    // multimodal content arrays are a documented non-goal -> 400, not a serde panic
    let err = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
    )
    .unwrap_err();
    assert_eq!(err.status, 400);
}

#[test]
fn resolve_applies_defaults() {
    let req =
        ChatCompletionRequest::parse(br#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    let p = req.resolve(1234).unwrap();
    assert_eq!(p.temp, DEFAULT_TEMP);
    assert_eq!(p.top_k, DEFAULT_TOP_K);
    assert_eq!(p.top_p, DEFAULT_TOP_P);
    assert_eq!(p.repeat_penalty, DEFAULT_REPEAT_PENALTY);
    assert_eq!(p.frequency_penalty, 0.0);
    assert_eq!(p.presence_penalty, 0.0);
    assert_eq!(p.max_tokens, MAX_TOKENS_UNLIMITED);
    assert_eq!(
        p.seed, 1234,
        "request seed missing -> caller-provided random default"
    );
    assert!(p.stop_strings.is_empty());
    // F3: every new knob's default is a no-op.
    assert_eq!(p.min_p, 0.0);
    assert_eq!(p.typical_p, 1.0);
    assert_eq!(p.xtc_probability, 0.0);
    assert_eq!(p.dry_multiplier, 0.0);
    assert_eq!(p.mirostat, crate::sampler::MirostatMode::Off);
    assert!(p.logit_bias.is_empty());
}

#[test]
fn resolve_honors_explicit_values() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.1,"max_tokens":50,"seed":7,"stop":"EOF","frequency_penalty":0.5,"presence_penalty":0.25}"#;
    let req = ChatCompletionRequest::parse(body).unwrap();
    let p = req.resolve(1234).unwrap();
    assert_eq!(p.temp, 0.1);
    assert_eq!(p.max_tokens, 50);
    assert_eq!(p.seed, 7);
    assert_eq!(p.stop_strings, vec!["EOF".to_string()]);
    assert_eq!(p.frequency_penalty, 0.5);
    assert_eq!(p.presence_penalty, 0.25);
}

#[test]
fn error_json_format_matches_llama_cpp() {
    let e = ApiError::exceed_context("prompt too long");
    let v: serde_json::Value = serde_json::from_str(&e.json()).unwrap();
    assert_eq!(v["error"]["code"], 400);
    assert_eq!(v["error"]["type"], "exceed_context_size_error");
    assert_eq!(v["error"]["message"], "prompt too long");
}

#[test]
fn response_json_shape() {
    let r = build_response("chatcmpl-x", "qwen", 1, "hi".into(), "stop", 3, 1, None);
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["total_tokens"], 4);
}

#[test]
fn chunk_builders() {
    let r = chunk_role("id", "m", 1);
    let v: serde_json::Value = serde_json::from_str(&r).unwrap();
    assert_eq!(v["object"], "chat.completion.chunk");
    assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
    assert!(v["choices"][0]["delta"]["content"].is_null());

    let c = chunk_content("id", "m", 1, "text");
    let v: serde_json::Value = serde_json::from_str(&c).unwrap();
    assert_eq!(v["choices"][0]["delta"]["content"], "text");

    let f = chunk_finish("id", "m", 1, "length");
    let v: serde_json::Value = serde_json::from_str(&f).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(v["choices"][0]["delta"].as_object().unwrap().is_empty());
}

#[test]
fn resolve_rejects_a_bad_mirostat_mode_or_logit_bias_key() {
    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"mirostat":3}"#,
    )
    .unwrap();
    let err = req.resolve(0).unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("mirostat"), "{}", err.message);

    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"logit_bias":{"not-an-id":1.0}}"#,
    )
    .unwrap();
    let err = req.resolve(0).unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("logit_bias"), "{}", err.message);
}

#[test]
fn resolve_and_validate_the_f3_surface() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"min_p":0.05,"typical_p":0.9,
        "xtc_probability":0.5,"xtc_threshold":0.1,"dry_multiplier":0.8,"dry_base":1.75,
        "dry_allowed_length":3,"dry_penalty_last_n":32,"dry_sequence_breakers":[[198],[13,2]],
        "mirostat":2,"mirostat_tau":4.0,"mirostat_eta":0.2,"mirostat_m":50,
        "logit_bias":{"198":-2.0,"42":1.5}}"#;
    let req = ChatCompletionRequest::parse(body).unwrap();
    let p = req.resolve(0).unwrap();
    assert_eq!(p.min_p, 0.05);
    assert_eq!(p.typical_p, 0.9);
    assert_eq!(p.dry_breakers, vec![vec![198u32], vec![13, 2]]);
    assert_eq!(p.mirostat, crate::sampler::MirostatMode::V2);
    // The map is resolved in a deterministic (sorted) order.
    assert_eq!(p.logit_bias, vec![(42u32, 1.5), (198u32, -2.0)]);
    assert!(p.validate(1000).is_ok());

    // Out-of-vocab id and an out-of-range value are 400s.
    let err = p.validate(100).unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("outside the vocabulary"));
    let bad = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"min_p":2.0}"#,
    )
    .unwrap()
    .resolve(0)
    .unwrap();
    assert!(bad.validate(1000).is_err());
}

// === F2 (#47): the structured-output surface ============================

#[test]
fn resolve_maps_response_format_and_the_grammar_extension() {
    // Absent / text -> unconstrained.
    let req =
        ChatCompletionRequest::parse(br#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert!(req.resolve(0).unwrap().grammar_source.is_none());
    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"text"}}"#,
    )
    .unwrap();
    assert!(req.resolve(0).unwrap().grammar_source.is_none());

    // json_object -> any JSON value.
    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
    )
    .unwrap();
    match req.resolve(0).unwrap().grammar_source {
        Some(crate::grammar::GrammarSource::AnyJson) => {}
        other => panic!("expected AnyJson, got {other:?}"),
    }

    // json_schema -> the schema, name and strict are accepted and ignored.
    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],
             "response_format":{"type":"json_schema","json_schema":
               {"name":"person","strict":true,"schema":{"type":"object"}}}}"#,
    )
    .unwrap();
    match req.resolve(0).unwrap().grammar_source {
        Some(crate::grammar::GrammarSource::Json(v)) => {
            assert_eq!(v["type"], "object");
        }
        other => panic!("expected Json, got {other:?}"),
    }

    // The GBNF extension field.
    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"grammar":"root ::= \"a\""}"#,
    )
    .unwrap();
    match req.resolve(0).unwrap().grammar_source {
        Some(crate::grammar::GrammarSource::Gbnf(g)) => assert_eq!(g, "root ::= \"a\""),
        other => panic!("expected Gbnf, got {other:?}"),
    }
}

#[test]
fn resolve_refuses_two_grammars_or_an_unknown_response_type() {
    // `grammar` + a non-text `response_format`: a 400, never a precedence rule.
    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"grammar":"root ::= .",
             "response_format":{"type":"json_object"}}"#,
    )
    .unwrap();
    let err = req.resolve(0).unwrap_err();
    assert_eq!(err.status, 400);
    assert!(
        err.message.contains("mutually exclusive"),
        "{}",
        err.message
    );

    // An unknown response_format type is refused by serde -> 400.
    let err = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"xml"}}"#,
    )
    .unwrap_err();
    assert_eq!(err.status, 400);

    // A json_schema entry without a schema is a 400.
    let err = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],
             "response_format":{"type":"json_schema","json_schema":{"name":"x"}}}"#,
    )
    .unwrap_err();
    assert_eq!(err.status, 400);
}

#[test]
fn compile_grammar_turns_an_unsupported_schema_into_a_400() {
    let tok = crate::tokenizer::Tokenizer::empty();
    let special = crate::models::SpecialTokens {
        eos: 0,
        im_end: None,
    };

    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],
             "response_format":{"type":"json_schema","json_schema":
               {"name":"x","schema":{"type":"string","pattern":"^a\\1"}}}}"#,
    )
    .unwrap();
    let mut p = req.resolve(0).unwrap();
    assert!(
        p.grammar.is_none(),
        "uncompiled until the vocabulary is known"
    );
    let err = p.compile_grammar(&tok, &special).unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("pattern"), "{}", err.message);
    assert!(
        p.grammar.is_none(),
        "a refused schema leaves no grammar behind"
    );

    // A supported schema compiles and lands in the sampler config.
    let req = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
    )
    .unwrap();
    let mut p = req.resolve(0).unwrap();
    p.compile_grammar(&tok, &special).expect("compiles");
    assert!(p.sampler_config().grammar.is_some());
}

// === #118: the two `dry_sequence_breakers` shapes ===========================

#[test]
fn dry_sequence_breakers_accepts_both_shapes_and_tells_them_apart() {
    let tok = crate::tokenizer::Tokenizer::byte_level_for_tests();

    // Token-id form: used as given, and the resolver is a no-op for it — it
    // never needs a vocabulary.
    let mut ids = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"dry_sequence_breakers":[[1010]]}"#,
    )
    .unwrap()
    .resolve(0)
    .unwrap();
    assert_eq!(ids.dry_breakers, vec![vec![1010u32]]);
    assert!(ids.dry_breaker_strings.is_empty());
    ids.resolve_dry_breakers(&crate::tokenizer::Tokenizer::empty())
        .expect("the token-id form needs no vocabulary");
    assert_eq!(ids.dry_breakers, vec![vec![1010u32]]);

    // String form: unresolved until the vocabulary is available, then the very
    // same sequence the token-id spelling above produced.
    let mut strings = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"dry_sequence_breakers":["\n"]}"#,
    )
    .unwrap()
    .resolve(0)
    .unwrap();
    assert!(
        strings.dry_breakers.is_empty(),
        "a breaker string is unresolved until the vocabulary is known"
    );
    assert_eq!(strings.dry_breaker_strings, vec!["\n".to_string()]);
    strings.resolve_dry_breakers(&tok).expect("resolves");
    assert_eq!(strings.dry_breakers, ids.dry_breakers);
}

#[test]
fn dry_sequence_breakers_rejects_an_empty_or_mixed_array() {
    // An empty array in either shape is a 400 — "no breakers" is the absent
    // field, matching llama.cpp's refusal of an empty array.
    let err = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"dry_sequence_breakers":[]}"#,
    )
    .unwrap()
    .resolve(0)
    .unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("must not be empty"), "{}", err.message);

    // A mixed array is neither shape, so it does not deserialize at all.
    let err = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"dry_sequence_breakers":["\n",[198]]}"#,
    )
    .unwrap_err();
    assert_eq!(err.status, 400);
}

#[test]
fn a_breaker_string_that_matches_no_token_is_a_400() {
    let mut p = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"dry_sequence_breakers":["\n"]}"#,
    )
    .unwrap()
    .resolve(0)
    .unwrap();
    let err = p
        .resolve_dry_breakers(&crate::tokenizer::Tokenizer::empty())
        .unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("overlaps no token"), "{}", err.message);
    assert!(
        p.dry_breakers.is_empty(),
        "a refused breaker leaves no sequence behind"
    );
}

// === Tool calling (#597) ==================================================
//
// The wire contract: the OpenAI shapes are accepted, everything this server
// cannot honour is a 400 that names the field, and a request without `tools`
// keeps the pinned `none` the byte-for-byte gates render.

const WEATHER_TOOL: &str = r#"{"type":"function","function":{"name":"get_weather","description":"Weather.","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}"#;

fn body_with_tools(extra: &str) -> String {
    format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],"tools":[{WEATHER_TOOL}]{extra}}}"#)
}

#[test]
fn tools_are_parsed_in_the_client_key_order() {
    let req = ChatCompletionRequest::parse(body_with_tools("").as_bytes()).expect("parse");
    let tools = req.tools.as_ref().expect("tools");
    assert_eq!(tools.len(), 1);
    let keys: Vec<&str> = tools[0]
        .as_object()
        .expect("object")
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(
        keys,
        ["type", "function"],
        "the client's order must survive the request parse"
    );
    // Unknown keys survive too: `| tojson` echoes the object, so a typed subset
    // would change the prompt.
    assert!(tools[0]
        .get("function")
        .unwrap()
        .get("description")
        .is_some());
}

#[test]
fn no_tools_leaves_the_field_absent() {
    let req = ChatCompletionRequest::parse(br#"{"messages":[{"role":"user","content":"hi"}]}"#)
        .expect("parse");
    assert!(req.tools.is_none());
    assert!(req.prompt_tools().expect("no tool_choice").is_none());
}

#[test]
fn parse_rejects_an_empty_tools_array() {
    let err = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"tools":[]}"#,
    )
    .unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("must not be empty"), "{}", err.message);
}

#[test]
fn parse_rejects_a_malformed_tool_by_index() {
    for (body, needle) in [
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":["get_weather"]}"#,
            "tools[0] must be an object",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"code_interpreter","function":{"name":"x"}}]}"#,
            "implements 'function' only",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"function":{"name":"x"}}]}"#,
            "no string `type`",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}"#,
            "no `function` object",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"  "}}]}"#,
            "function.name must be a non-empty string",
        ),
    ] {
        let err = ChatCompletionRequest::parse(body.as_bytes()).unwrap_err();
        assert_eq!(err.status, 400, "{body}");
        assert!(err.message.contains(needle), "{body} -> {}", err.message);
    }
}

#[test]
fn tool_choice_requires_tools() {
    let err = ChatCompletionRequest::parse(
        br#"{"messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}"#,
    )
    .unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("requires `tools`"), "{}", err.message);
}

#[test]
fn auto_and_absent_tool_choice_render_the_tools() {
    for extra in ["", r#","tool_choice":"auto""#] {
        let req = ChatCompletionRequest::parse(body_with_tools(extra).as_bytes()).expect("parse");
        let rendered = req.prompt_tools().expect("auto");
        assert_eq!(rendered.map(|t| t.len()), Some(1), "extra={extra:?}");
    }
}

#[test]
fn tool_choice_none_drops_the_tools_from_the_prompt() {
    // No output extraction exists, so "do not call a tool" can only be honoured
    // by not describing the tools: rendering them would invite a call this
    // server would return as ordinary content.
    let req = ChatCompletionRequest::parse(body_with_tools(r#","tool_choice":"none""#).as_bytes())
        .expect("parse");
    assert!(req.prompt_tools().expect("none").is_none());
}

#[test]
fn tool_choice_required_and_a_forced_function_are_resolved_to_a_constraint() {
    // #614: both stop being 400s. `required` renders the tools and asks the
    // handler for a grammar; a forced function names the one tool.
    let req =
        ChatCompletionRequest::parse(body_with_tools(r#","tool_choice":"required""#).as_bytes())
            .expect("parse");
    assert_eq!(
        req.resolve_tool_choice().unwrap(),
        ToolChoiceIntent::Required
    );
    assert!(
        req.prompt_tools().unwrap().is_some(),
        "required renders tools"
    );

    let req = ChatCompletionRequest::parse(
        body_with_tools(r#","tool_choice":{"type":"function","function":{"name":"get_weather"}}"#)
            .as_bytes(),
    )
    .expect("parse");
    assert_eq!(
        req.resolve_tool_choice().unwrap(),
        ToolChoiceIntent::Forced("get_weather".to_string())
    );
    assert!(req.prompt_tools().unwrap().is_some());

    // `auto` and `none` keep their old meaning.
    let req = ChatCompletionRequest::parse(body_with_tools("").as_bytes()).expect("parse");
    assert_eq!(req.resolve_tool_choice().unwrap(), ToolChoiceIntent::Auto);
    let req = ChatCompletionRequest::parse(body_with_tools(r#","tool_choice":"none""#).as_bytes())
        .expect("parse");
    assert_eq!(req.resolve_tool_choice().unwrap(), ToolChoiceIntent::None);
    assert!(req.prompt_tools().unwrap().is_none());
}

#[test]
fn a_forced_function_must_be_one_of_the_offered_tools() {
    let err = ChatCompletionRequest::parse(
        body_with_tools(r#","tool_choice":{"type":"function","function":{"name":"nope"}}"#)
            .as_bytes(),
    )
    .expect("parse")
    .resolve_tool_choice()
    .unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.message.contains("nope"), "{}", err.message);

    let err = ChatCompletionRequest::parse(
        body_with_tools(r#","tool_choice":{"type":"other","function":{"name":"get_weather"}}"#)
            .as_bytes(),
    )
    .expect("parse")
    .resolve_tool_choice()
    .unwrap_err();
    assert!(err.message.contains("other"), "{}", err.message);
}

#[test]
fn an_unknown_tool_choice_shape_is_a_body_error() {
    let err =
        ChatCompletionRequest::parse(body_with_tools(r#","tool_choice":"sometimes""#).as_bytes())
            .unwrap_err();
    assert_eq!(err.status, 400);
}

// === #613: a request message's own JSON ==================================

#[test]
fn parse_accepts_a_replayed_tool_call_and_keeps_every_key() {
    let body = br#"{"messages":[
        {"role":"user","content":"Weather?"},
        {"role":"assistant","content":null,"tool_calls":[
            {"id":"call_1","type":"function",
             "function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}
        ]},
        {"role":"tool","content":"18C","tool_call_id":"call_1"}
    ]}"#;
    let req = ChatCompletionRequest::parse(body).expect("a replayed round-trip parses");
    assert_eq!(req.messages.len(), 3);

    // The object's own key order is the client's, not the engine's.
    let keys: Vec<&str> = req.messages[1]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(keys, ["role", "content", "tool_calls"]);

    let crate::template::ToolJson::Array(calls) = req.messages[1].get("tool_calls").unwrap() else {
        panic!("tool_calls must stay an array");
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(crate::template::ToolJson::as_str),
        Some(r#"{"city": "Paris"}"#),
        "the arguments string is carried verbatim"
    );

    // A key the engine does not read (`tool_call_id`) still survives the parse.
    assert!(req.messages[2].get("tool_call_id").is_some());
}

#[test]
fn a_malformed_message_or_tool_call_is_a_400_naming_the_index() {
    // Every arm of `validate_message` / `validate_tool_call`, each naming the
    // message and — where it applies — the entry.
    let cases: &[(&str, &str)] = &[
        (r#"{"messages":["hi"]}"#, "messages[0] must be an object"),
        (
            r#"{"messages":[{"role":1,"content":"x"}]}"#,
            "messages[0] has no string `role`",
        ),
        (
            r#"{"messages":[{"role":"user","content":1}]}"#,
            "messages[0].content must be a string or null",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":{}}]}"#,
            "messages[0].tool_calls must be an array",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[]}]}"#,
            "must not be empty",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":["x"]}]}"#,
            "messages[0].tool_calls[0] must be an object",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{}"}}]}]}"#,
            "messages[0].tool_calls[0].id",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"other","function":{"name":"f","arguments":"{}"}}]}]}"#,
            "messages[0].tool_calls[0].type",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function"}]}]}"#,
            "messages[0].tool_calls[0] has no `function` object",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"","arguments":"{}"}}]}]}"#,
            "messages[0].tool_calls[0].function.name",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":{"city":"Paris"}}}]}]}"#,
            "messages[0].tool_calls[0].function.arguments must be a string",
        ),
    ];
    for (body, needle) in cases {
        let err = ChatCompletionRequest::parse(body.as_bytes()).expect_err(body);
        assert_eq!(err.status, 400, "{body}");
        assert!(
            err.message.contains(needle),
            "{body}\nexpected a message containing {needle:?}, got: {}",
            err.message
        );
    }
}

// === #614: tool calls in the response ===================================

#[test]
fn a_response_with_tool_calls_serialises_the_openai_shape() {
    let call = crate::server::toolcall::ToolCall {
        name: "get_weather".to_string(),
        arguments: r#"{"city": "Paris"}"#.to_string(),
    };
    let r = build_response(
        "id",
        "m",
        1,
        String::new(),
        "tool_calls",
        3,
        1,
        Some(vec![ResponseToolCall::new(&call)]),
    );
    let v = serde_json::to_value(&r).unwrap();
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], "");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(msg["tool_calls"][0]["type"], "function");
    assert!(
        msg["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .starts_with("call_"),
        "the model writes no id; the response synthesises one"
    );
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(
        msg["tool_calls"][0]["function"]["arguments"],
        r#"{"city": "Paris"}"#
    );
}

#[test]
fn a_prose_response_has_no_tool_calls_field() {
    let r = build_response("id", "m", 1, "hi".into(), "stop", 3, 1, None);
    let v = serde_json::to_value(&r).unwrap();
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());
}
