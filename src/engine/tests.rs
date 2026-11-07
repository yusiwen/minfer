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
