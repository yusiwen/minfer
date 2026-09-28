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
    let r = build_response("chatcmpl-x", "qwen", 1, "hi".into(), "stop", 3, 1);
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
               {"name":"x","schema":{"type":"string","pattern":"^a"}}}}"#,
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
