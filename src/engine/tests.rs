//! `#[cfg(test)] mod tests` for `src/engine.rs`.
use super::*;

/// The facade's error is a value with a reason for every load step; the toy
/// fixture is the cheapest way to pin the successful half without a device.
#[test]
fn the_toy_model_loads_and_speaks_the_fallback_template() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
    let engine = Engine::load(&path).expect("the toy GGUF loads");
    assert!(
        engine.template().is_none(),
        "the toy fixture carries no template"
    );
    assert!(engine.chatml_notice(), "the fallback must be announced");
    assert_eq!(engine.model().n_layer(), 1);
    assert!(engine.tokenizer().vocab_size() > 0);
}

/// A missing file is a named error, not a panic (#763 S3's contract at the
/// facade boundary).
#[test]
fn a_missing_model_is_a_named_error() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("no-such-model.gguf");
    let err = Engine::load(&path)
        .err()
        .expect("a missing file must refuse");
    assert!(matches!(err, EngineError::Gguf(_)), "{err}");
}

/// #799: a generation inside the lib test binary, which the process-global
/// graph cache used to forbid — it sized the cache at the toy's `n_ctx` for
/// every real-model gate sharing this binary. Each engine now owns its cache,
/// so the isolation is structural, not a test-ordering accident.
#[test]
fn a_generation_inside_the_lib_borrows_no_process_global() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
    let engine = Engine::load(&path).expect("the toy GGUF loads");
    let ids = engine.tokenizer().encode("hi");
    let sampler = crate::sampler::SamplerConfig::default();
    let params = GenerateParams {
        tokens: &ids,
        n_ctx: 32,
        n_predict: 1,
        sampler: &sampler,
        seed: 42,
        stop_strings: &[],
        timing: false,
        spec_adaptive: false,
        spec_d_max: 2,
        cancel: None,
    };
    let outcome = engine.generate(&params, &mut |_| {}, &mut |_, _| {});
    assert_eq!(
        outcome.prefill_tokens,
        ids.len(),
        "the prefill ran, so the forward did too"
    );
}
