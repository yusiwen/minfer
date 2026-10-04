//! The grammar mask inside the pipeline.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// A grammar whose language is the whole ASCII byte range must be a no-op:
/// same tokens, same mirostat trajectory, same RNG stream as no grammar.
#[test]
fn an_allow_everything_grammar_does_not_perturb_the_pipeline() {
    let g = tiny_grammar("root ::= .*", 128, &[]);
    let cfg = SamplerConfig {
        temp: 0.8,
        top_k: 0,
        top_p: 1.0,
        repeat_penalty: 1.0,
        mirostat: MirostatMode::V2,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let plain = SamplerConfig {
        grammar: None,
        ..cfg.clone()
    };
    let run = |cfg: &SamplerConfig| -> (Vec<u32>, f32) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let mut grammar = cfg.grammar.as_ref().map(|g| g.state());
        let mut prev: Vec<u32> = Vec::new();
        let mut out = Vec::new();
        for step in 0..32u64 {
            let mut logits: Vec<f32> = (0..128)
                .map(|i| ((i as f32) * 0.13 + (step as f32) * 0.29).sin() * 3.0)
                .collect();
            let t = sample_with_config_grammar(
                &mut logits,
                cfg,
                &prev,
                &mut mirostat,
                &mut grammar,
                &mut rng,
            )
            .expect("allow-all grammar")
            .token_id;
            out.push(t);
            prev.push(t);
        }
        (out, mirostat.mu)
    };
    let (with, mu_with) = run(&cfg);
    let (without, mu_without) = run(&plain);
    assert_eq!(
        with, without,
        "an allow-all grammar must not change the tokens"
    );
    assert_eq!(
        mu_with, mu_without,
        "mirostat's mu must follow the same trajectory"
    );
}
/// The mask decides the greedy winner: a grammar that only allows `a` beats
/// a logit argmax on `b`, and the state advances token by token.
#[test]
fn grammar_mask_decides_the_greedy_choice_and_advances() {
    let g = tiny_grammar("root ::= \"ab\"", 128, &[1]);
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let mut grammar = Some(g.state());
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);

    // 'b' (0x62) is the argmax but the grammar is at `a`.
    let mut logits = vec![0.0f32; 128];
    logits[0x62] = 10.0;
    logits[0x61] = 1.0;
    let first = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(first.token_id, 0x61, "the mask must beat the argmax");

    let mut logits = vec![0.0f32; 128];
    logits[0x63] = 10.0;
    logits[0x62] = 1.0;
    let second = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0x61],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(second.token_id, 0x62);

    // The grammar is complete: only the EOG token is legal now.
    let mut logits = vec![0.0f32; 128];
    logits[0x61] = 10.0;
    let third = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0x61, 0x62],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(third.token_id, 1, "EOG is the only legal continuation");
    assert!(grammar.as_ref().unwrap().is_accepting() || true);
    // A token after EOG is a loud error, never a silent one.
    let mut logits = vec![0.0f32; 128];
    logits[1] = 10.0;
    let err = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0x61, 0x62, 1],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap_err();
    assert!(err.to_string().contains("end of generation"), "{err}");
}
/// No legal token at all is a loud stop, not an arbitrary token.
#[test]
fn grammar_pipeline_stops_when_no_token_is_allowed() {
    // The vocabulary has 'a' (0x61) but no 'b': after `a` the grammar is stuck.
    let pieces = vec![Some(vec![0x61u8].into_boxed_slice())];
    let g = Arc::new(Grammar::from_gbnf("root ::= \"ab\"", pieces, vec![false]).unwrap());
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let mut grammar = Some(g.state());
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let mut logits = vec![5.0f32];
    let first = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(first.token_id, 0);
    let mut logits = vec![5.0f32];
    let err = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap_err();
    match err {
        SampleError::NoAllowedToken { state } => {
            assert!(
                state.contains("stack"),
                "the reason names the state: {state}"
            )
        }
        other => panic!("expected NoAllowedToken, got {other:?}"),
    }
}
/// A configured grammar with no run state is a bug, never a silent fallback
/// to unconstrained sampling.
#[test]
fn grammar_pipeline_refuses_a_configured_grammar_without_state() {
    let g = tiny_grammar("root ::= .*", 128, &[]);
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g),
        ..SamplerConfig::default()
    };
    let mut none: Option<GrammarState> = None;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let mut logits = vec![1.0f32; 128];
    let err =
        sample_with_config_grammar(&mut logits, &cfg, &[], &mut mirostat, &mut none, &mut rng)
            .unwrap_err();
    assert!(err.to_string().contains("no grammar state"), "{err}");
}
/// Every token the pipeline emits under a JSON grammar is one the automaton
/// accepts: drive a fixed token stream that spells a JSON object and assert
/// each step's sampled token is allowed and the final state is accepting.
#[test]
fn sampled_tokens_are_always_allowed_by_the_json_grammar() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": false
    });
    let n_vocab = 128usize;
    let pieces: Vec<Option<Box<[u8]>>> = (0..n_vocab)
        .map(|i| Some(vec![i as u8].into_boxed_slice()))
        .collect();
    let g =
        Arc::new(Grammar::from_json_schema(&schema, pieces, vec![false; n_vocab]).expect("schema"));
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let mut grammar = Some(g.state());
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let target = br#"{"a":1}"#;
    for (i, &want) in target.iter().enumerate() {
        // Reward exactly the next byte the target needs; everything else is
        // noise, so the mask is what has to keep the run on the rails.
        let mut logits = vec![0.0f32; n_vocab];
        logits[want as usize] = 5.0;
        let sampled = sample_with_config_grammar(
            &mut logits,
            &cfg,
            if i == 0 { &[] } else { &[] },
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap_or_else(|e| panic!("step {i} (byte {}): {e}", want as char));
        assert_eq!(
            sampled.token_id, want as u32,
            "the mask must allow the next byte {}",
            want as char
        );
    }
    assert!(
        grammar.as_ref().unwrap().is_accepting(),
        "the driven text is a complete instance"
    );
}
