//! The defaults are a no-op: the pinned pre-F3 and pre-F2 pipelines.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// The F3 gate: with every new knob at its default, the config pipeline
/// reproduces the pre-F3 chain token-for-token over 64 steps with one RNG
/// seed (so it also proves no new sampler consumed an RNG draw).
#[test]
fn test_default_config_is_bit_identical_to_the_old_path() {
    let old = f3_sequence(|logits, prev| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42 + prev.len() as u64);
        pre_f3_chain(logits, prev, &mut rng)
    });
    let cfg = SamplerConfig {
        temp: 0.8,
        top_k: 40,
        top_p: 0.95,
        repeat_penalty: 1.1,
        ..SamplerConfig::default()
    };
    let new = f3_sequence(|logits, prev| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42 + prev.len() as u64);
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        sample_with_config(logits, &cfg, prev, &mut mirostat, &mut rng).token_id
    });
    assert_eq!(
        old, new,
        "the default config pipeline must be bit-identical to the pre-F3 chain"
    );
    assert_eq!(old.len(), 64);
}
/// The same gate against a sequence captured from `master` *before* this
/// change (`sample_with_penalties`, seed 42, the `f3_logits` stream). A
/// refactor that perturbs the default path fails here even if it stays
/// self-consistent.
#[test]
fn test_default_pipeline_matches_the_pinned_pre_f3_sequence() {
    const PINNED: [u32; 64] = [
        5, 54, 21, 54, 105, 69, 155, 36, 137, 1, 54, 35, 34, 85, 17, 103, 67, 16, 50, 0, 101, 133,
        66, 49, 115, 46, 82, 14, 98, 62, 98, 28, 78, 9, 12, 60, 45, 10, 77, 78, 26, 110, 44, 44,
        145, 57, 41, 7, 7, 39, 25, 41, 7, 91, 57, 21, 55, 38, 6, 72, 106, 37, 18, 121,
    ];
    // The pinned sequence was captured by advancing one RNG *across* steps
    // (the decode-loop shape), not reseeding per step.
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut cfg = SamplerConfig {
        temp: 0.8,
        top_k: 40,
        top_p: 0.95,
        repeat_penalty: 1.1,
        ..SamplerConfig::default()
    };
    cfg.min_p = 0.0;
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let mut prev: Vec<u32> = Vec::new();
    let mut out: Vec<u32> = Vec::new();
    for step in 0..64 {
        let mut logits = f3_logits(step);
        let t = sample_with_config(&mut logits, &cfg, &prev, &mut mirostat, &mut rng).token_id;
        out.push(t);
        prev.push(t);
    }
    assert_eq!(out, PINNED.to_vec());
}
/// The same pinned pre-F2 sequence, driven through the **new** entry point
/// with no grammar. `PINNED` was captured from `master` before the F2 change
/// (the F3 gate's array, which was itself captured pre-F3), so this proves
/// the added mask stage cannot perturb the unconstrained chain.
#[test]
fn test_default_pipeline_matches_the_pinned_pre_f2_sequence() {
    const PINNED: [u32; 64] = [
        5, 54, 21, 54, 105, 69, 155, 36, 137, 1, 54, 35, 34, 85, 17, 103, 67, 16, 50, 0, 101, 133,
        66, 49, 115, 46, 82, 14, 98, 62, 98, 28, 78, 9, 12, 60, 45, 10, 77, 78, 26, 110, 44, 44,
        145, 57, 41, 7, 7, 39, 25, 41, 7, 91, 57, 21, 55, 38, 6, 72, 106, 37, 18, 121,
    ];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let cfg = SamplerConfig {
        temp: 0.8,
        top_k: 40,
        top_p: 0.95,
        repeat_penalty: 1.1,
        min_p: 0.0,
        ..SamplerConfig::default()
    };
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let mut prev: Vec<u32> = Vec::new();
    let mut out: Vec<u32> = Vec::new();
    let mut grammar: Option<GrammarState> = None;
    for step in 0..64 {
        let mut logits = f3_logits(step);
        let t = sample_with_config_grammar(
            &mut logits,
            &cfg,
            &prev,
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .expect("no grammar can never fail")
        .token_id;
        out.push(t);
        prev.push(t);
    }
    assert_eq!(out, PINNED.to_vec());
}
