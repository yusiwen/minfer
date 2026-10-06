//! `#[cfg(test)] mod tests` for `src/server/chat.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::models::Device;

/// E6: the whole decision matrix, with no device involved — which is the
/// point, since CI has no GPU and the default it guards is what a GPU server
/// now gets.
#[test]
fn batch_mode_follows_the_device_and_honours_the_override() {
    // Unset: follow the device. CUDA batches (measured 1.9x); **Metal joined
    // CUDA in #44 part (b)** when the `copy_cells` / `copy_kv_to_cpu` arms
    // landed (the batching and server gates are green on the Mac). CPU still
    // does not (measured slower per token — the E2 record).
    assert_eq!(batch_mode(None, Device::Cuda), BatchMode::Batched);
    assert_eq!(batch_mode(None, Device::Metal), BatchMode::Batched);
    assert_eq!(batch_mode(None, Device::Cpu), BatchMode::Serial);

    // Explicit override wins on every device.
    for d in [Device::Cpu, Device::Metal, Device::Cuda] {
        assert_eq!(batch_mode(Some("1"), d), BatchMode::Batched, "{d:?}");
        assert_eq!(batch_mode(Some("0"), d), BatchMode::Serial, "{d:?}");
    }

    // A value that is neither 0 nor 1 (the caller warns) falls back to the
    // device's default rather than guessing an intent.
    for v in ["true", "yes", "2", ""] {
        assert_eq!(
            batch_mode(Some(v), Device::Cuda),
            BatchMode::Batched,
            "{v:?}"
        );
        assert_eq!(batch_mode(Some(v), Device::Cpu), BatchMode::Serial, "{v:?}");
    }
}

#[test]
fn prefill_span_always_feeds_the_last_token() {
    assert_eq!(prefill_span(10, 0), (0, 10), "cold slot: full prefill");
    assert_eq!(prefill_span(10, 4), (4, 6), "reuse a 4-token prefix");
    assert_eq!(
        prefill_span(10, 10),
        (9, 1),
        "fully cached: still one row for the logits"
    );
    assert_eq!(prefill_span(1, 0), (0, 1));
    assert_eq!(prefill_span(1, 1), (0, 1));
    assert_eq!(
        prefill_span(10, 99),
        (9, 1),
        "reuse cannot exceed the prompt"
    );
}

#[test]
fn common_prefix_len_finds_the_exact_match() {
    assert_eq!(common_prefix_len(&[], &[1, 2, 3]), 0, "nothing recorded");
    assert_eq!(common_prefix_len(&[1, 2], &[]), 0, "empty prompt");
    assert_eq!(common_prefix_len(&[1, 2], &[1, 2]), 2, "identical");
    assert_eq!(
        common_prefix_len(&[1, 2], &[1, 2, 3]),
        2,
        "the record is a true prefix of the prompt"
    );
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 9]), 1, "diverges early");
    assert_eq!(
        common_prefix_len(&[5], &[1, 2, 3]),
        0,
        "a different first token must force a full prefill"
    );
}

#[test]
fn stop_token_matches_eos_and_im_end() {
    let special = SpecialTokens {
        eos: 2,
        im_end: Some(7),
    };
    assert!(is_stop_token(2, &special));
    assert!(is_stop_token(7, &special));
    assert!(!is_stop_token(3, &special));
}

/// A panicking job must become a 500 on that request's stream — and the
/// worker must live on. Before the fix the panic unwound `worker_loop`, so
/// nothing was ever sent and the whole server degraded.
#[test]
fn isolated_job_turns_a_panic_into_an_error_event() {
    let (tx, mut rx) = mpsc::channel(4);
    let survived = run_job_isolated(&tx, || panic!("boom"));
    assert!(
        !survived,
        "a panicking job must report that it did not survive"
    );
    match rx.try_recv() {
        Ok(StreamEvent::Err(e)) => {
            let body = e.json();
            assert!(body.contains("boom"), "panic message lost: {body}");
            assert!(body.contains("server_error"), "wrong class: {body}");
        }
        _ => panic!("expected an Err event for the panicking job"),
    }
}

#[test]
fn isolated_job_forwards_a_normal_error() {
    let (tx, mut rx) = mpsc::channel(4);
    assert!(run_job_isolated(&tx, || Err(ApiError::server("nope"))));
    match rx.try_recv() {
        Ok(StreamEvent::Err(e)) => assert!(e.json().contains("nope")),
        _ => panic!("expected the error to be forwarded"),
    }
}

#[test]
fn isolated_job_passes_success_through_silently() {
    let (tx, mut rx) = mpsc::channel(4);
    assert!(run_job_isolated(&tx, || Ok(())));
    assert!(rx.try_recv().is_err(), "a clean job emits no error event");
}

// === F2 (#47): real-model grammar gates =================================
//
// The `#[ignore]`d family: CI has no cached GGUF, so these run on a box with
// the models — serially (`cargo test --release --bin minfer -- --ignored
// --test-threads=1`). The same set runs twice: the cached 0.5B (f32 KV) and,
// when `MINFER_BATCH_TEST_MODEL` points at it, the Qwen3-0.6B Q8_0.

fn cached_model() -> Option<std::path::PathBuf> {
    if let Ok(custom) = std::env::var("MINFER_BATCH_TEST_MODEL") {
        let p = std::path::PathBuf::from(custom);
        return p.exists().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/\
         qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    p.exists().then_some(p)
}

/// Greedy, seeded, small `max_tokens` — the deterministic reading the gates
/// compare against. The grammar is attached by the caller.
fn grammar_params(max_tokens: i64) -> SamplingParams {
    SamplingParams {
        temp: 0.0,
        top_k: 1,
        top_p: 1.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        typical_p: 1.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.5,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 64,
        dry_breakers: Vec::new(),
        mirostat: crate::sampler::MirostatMode::Off,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        mirostat_m: 100,
        logit_bias: Vec::new(),
        grammar_source: None,
        grammar: None,
        seed: 42,
        stop_strings: Vec::new(),
        max_tokens,
    }
}

/// One serial request through the real `generate` path; returns the text and
/// the finish reason.
fn run_serial(
    model: &dyn ModelDef,
    tok: &Tokenizer,
    params: &SamplingParams,
    prompt: &str,
) -> (String, String) {
    let input_ids = tok.encode(prompt);
    assert!(!input_ids.is_empty(), "the prompt must tokenize");
    let mut cache = GraphCache::new();
    let mut cached: Vec<u32> = Vec::new();
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
    generate(
        model,
        tok,
        &mut cache,
        &mut cached,
        512,
        &input_ids,
        params,
        &tx,
    )
    .expect("generate");
    drop(tx);
    let mut text = String::new();
    let mut reason = String::from("?");
    while let Some(ev) = rx.blocking_recv() {
        match ev {
            StreamEvent::Text(t) => text.push_str(&t),
            StreamEvent::Finish { reason: r, .. } => reason = r,
            StreamEvent::Err(e) => panic!("stream error: {}", e.message),
        }
    }
    (text, reason)
}

/// A JSON-schema-constrained greedy generation must parse, on whichever
/// model the test box has cached.
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn real_model_json_schema_generation_parses() {
    let Some(path) = cached_model() else {
        eprintln!("[f2] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let special = model.special_tokens();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150}
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });
    let mut params = grammar_params(48);
    params.grammar_source = Some(crate::grammar::GrammarSource::Json(schema));
    params
        .compile_grammar(&tok, &special)
        .expect("the schema compiles");
    let grammar = params.grammar.clone().expect("compiled");

    let t0 = std::time::Instant::now();
    let (text, reason) = run_serial(
        &*model,
        &tok,
        &params,
        "Give me a JSON object with a name and an age.\n",
    );
    let elapsed = t0.elapsed();
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("constrained output must parse: {e}\ntext={text:?}"));
    assert!(parsed.is_object(), "the schema is an object: {text:?}");
    assert!(parsed["name"].is_string(), "{text:?}");
    assert!(parsed["age"].is_u64(), "{text:?}");
    assert!(
        grammar.accepts(text.as_bytes()),
        "grammar:\n{}",
        grammar.source()
    );
    // The same prompt, same seed, no grammar: the delta is what the mask costs
    // (state-cached, so it is paid once per distinct state, not per token).
    let mut plain = grammar_params(48);
    plain.seed = params.seed;
    let t1 = std::time::Instant::now();
    let (plain_text, _) = run_serial(
        &*model,
        &tok,
        &plain,
        "Give me a JSON object with a name and an age.\n",
    );
    let plain_elapsed = t1.elapsed();
    eprintln!(
        "[f2] model={} schema: {reason} after {:.2}s -> {text:?}",
        path.display(),
        elapsed.as_secs_f64()
    );
    // The mask's own cost on the real vocabulary: drive the emitted text
    // token by token and time `mask` at each step (state-cached, so a
    // repeated state is a hash lookup).
    let ids = tok.encode(&text);
    let mut st = grammar.state();
    let t2 = std::time::Instant::now();
    for id in &ids {
        let _ = grammar.mask(&mut st).expect("mask");
        grammar.accept_token(&mut st, *id).expect("accept");
    }
    let mask_total = t2.elapsed();
    eprintln!(
        "[f2] mask: {} step(s) over {}-token vocab in {:?} ({:?}/step), cached states {}",
        ids.len(),
        grammar.n_vocab(),
        mask_total,
        mask_total / ids.len().max(1) as u32,
        st.cached_states()
    );
    eprintln!(
        "[f2] same prompt unconstrained: {:.2}s ({}) -> {plain_text:.60?}",
        plain_elapsed.as_secs_f64(),
        if plain_elapsed.as_secs_f64() > 0.0 {
            format!(
                "{:.2}x",
                elapsed.as_secs_f64() / plain_elapsed.as_secs_f64()
            )
        } else {
            "n/a".to_string()
        }
    );
}

/// The empty case: `max_tokens: 0` emits nothing; a grammar that accepts the
/// empty string allows only EOG at step 0 and stops with no bytes at all.
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn real_model_empty_generation_is_clean() {
    let Some(path) = cached_model() else {
        eprintln!("[f2] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let special = model.special_tokens();

    // (a) max_tokens = 0 with a schema: the loop never runs.
    let mut params = grammar_params(0);
    params.grammar_source = Some(crate::grammar::GrammarSource::Json(serde_json::json!({
        "type": "object",
        "additionalProperties": false
    })));
    params
        .compile_grammar(&tok, &special)
        .expect("the schema compiles");
    let (text, reason) = run_serial(&*model, &tok, &params, "Hello\n");
    assert!(
        text.is_empty(),
        "max_tokens 0 must emit nothing, got {text:?}"
    );
    assert_eq!(reason, "length");

    // (b) the grammar already accepts the empty string: only EOG is legal, so
    //     generation ends with zero tokens and zero bytes.
    let mut params = grammar_params(16);
    params.grammar_source = Some(crate::grammar::GrammarSource::Gbnf("root ::= \"\"".into()));
    params
        .compile_grammar(&tok, &special)
        .expect("the grammar compiles");
    let grammar = params.grammar.clone().expect("compiled");
    assert!(grammar.accepts(b""), "the grammar accepts the empty string");
    let (text, reason) = run_serial(&*model, &tok, &params, "Hello\n");
    assert!(
        text.is_empty(),
        "a completed grammar must emit nothing, got {text:?}"
    );
    assert_eq!(reason, "stop", "EOG ends the turn");
    eprintln!(
        "[f2] model={} empty case: reason={reason} text={text:?}",
        path.display()
    );
}

/// The max-length case: the length limit cuts the generation mid-grammar.
/// The honest assertion is not "the text parses" (a truncated object cannot)
/// but "every byte emitted left the automaton in a live state", i.e. the
/// output is a valid prefix.
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn real_model_max_length_output_is_a_valid_prefix() {
    let Some(path) = cached_model() else {
        eprintln!("[f2] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let special = model.special_tokens();

    let mut params = grammar_params(8);
    params.grammar_source = Some(crate::grammar::GrammarSource::Json(serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 3}
        },
        "required": ["name", "tags"],
        "additionalProperties": false
    })));
    params
        .compile_grammar(&tok, &special)
        .expect("the schema compiles");
    let grammar = params.grammar.clone().expect("compiled");

    let (text, reason) = run_serial(&*model, &tok, &params, "Fill in the JSON.\n");
    assert_eq!(reason, "length", "the cap must be what stopped it");
    assert!(
        grammar.accepts_prefix(text.as_bytes()),
        "the truncated output must still be a valid prefix: {text:?}\ngrammar:\n{}",
        grammar.source()
    );
    assert!(
        text.starts_with('{'),
        "the mask forces the object open: {text:?}"
    );
    assert!(
        serde_json::from_str::<serde_json::Value>(&text).is_err(),
        "8 tokens cannot complete this schema — the prefix assertion is the honest one: {text:?}"
    );
    eprintln!(
        "[f2] model={} max-length: {text:?} ({reason})",
        path.display()
    );
}
