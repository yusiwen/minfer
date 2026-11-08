//! #763 S6a: the engine facade as an **external** consumer sees it.
//!
//! These tests use `minfer::engine`, not `crate::`, so they exercise exactly the
//! surface a host gets: load a model, run a chat turn, drop everything, load
//! again. The toy fixture keeps them device-free and fast.

use minfer::conversation::TurnParams;
use minfer::engine::{Engine, SessionConfig};
use minfer::sampler::SamplerConfig;

fn toy() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/toy-qwen2-f32.gguf")
}

fn turn(n_predict: usize) -> TurnParams {
    TurnParams {
        n_predict,
        sampler: SamplerConfig::default(),
        stop_strings: Vec::new(),
    }
}

#[test]
fn a_host_loads_chats_drops_and_loads_again() {
    let path = toy();
    let first = {
        let engine = Engine::load(&path).expect("the facade loads the toy GGUF");
        assert!(engine.chatml_notice(), "the toy has no template");
        let mut session = engine.session(SessionConfig {
            n_ctx: 64,
            ..Default::default()
        });
        let mut out = Vec::new();
        let outcome = session
            .chat("hello", &turn(4), &mut |delta| out.extend_from_slice(delta))
            .expect("one chat turn");
        assert!(
            outcome.tokens_generated <= 4,
            "n_predict caps the turn: {}",
            outcome.tokens_generated
        );
        out
    };
    // The session and the engine are gone; a second load must work (the mapping
    // is released — #745's gate asserts the release itself).
    let engine = Engine::load(&path).expect("a second load works");
    assert_eq!(
        engine.model().n_layer(),
        1,
        "a fresh engine, not stale state"
    );
    let _ = first;
}

#[test]
fn two_sessions_on_one_engine_do_not_share_history() {
    let engine = Engine::load(&toy()).expect("load");
    let mut a = engine.session(SessionConfig {
        n_ctx: 64,
        ..Default::default()
    });
    let mut b = engine.session(SessionConfig {
        n_ctx: 64,
        ..Default::default()
    });
    let mut sink = |_d: &[u8]| {};
    a.chat("first", &turn(1), &mut sink).expect("a");
    assert_eq!(a.conversation().messages.len(), 2, "user + assistant");
    assert!(
        b.conversation().messages.is_empty(),
        "the second session starts empty"
    );
}

#[test]
fn the_external_generate_runs_the_whole_loop() {
    let engine = Engine::load(&toy()).expect("load");
    let ids = engine.tokenizer().encode("hi");
    let sampler = SamplerConfig::default();
    let params = minfer::engine::GenerateParams {
        tokens: &ids,
        n_ctx: 64,
        n_predict: 2,
        sampler: &sampler,
        seed: 42,
        stop_strings: &[],
        timing: false,
        spec_adaptive: false,
        spec_d_max: 2,
    };
    let mut out = Vec::new();
    let mut prefill = None;
    let outcome = engine.generate(
        &params,
        &mut |delta| out.extend_from_slice(delta),
        &mut |_logits, secs| prefill = Some(secs),
    );
    assert_eq!(outcome.prefill_tokens, ids.len());
    assert!(outcome.generated <= 2, "n_predict caps the run");
    assert!(prefill.is_some(), "the prefill hook fires once");
}
