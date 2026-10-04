//! `#[cfg(test)] mod tests` for `src/sampler.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use rand::SeedableRng;

mod bias_validate;
mod defaults;
mod dry;
mod grammar;
mod greedy_topk;
mod minp_typical;
mod mirostat;
mod penalties;
mod stops;
mod xtc;
// === F3 (#48): the new sampler set ===

/// The default-config pipeline, and the pinned sequence it must reproduce.
fn f3_logits(step: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; 256];
    for (i, v) in logits.iter_mut().enumerate() {
        *v = ((i as f32) * 0.37 + (step as f32) * 0.11).sin() * 3.0 + ((i as f32) * 0.011).cos();
    }
    logits
}
/// 64 steps of the pipeline; `sample` is either the pre-F3 entry point or the
/// config one, so the two gates below share one driver.
fn f3_sequence(mut sample: impl FnMut(&mut Vec<f32>, &[u32]) -> u32) -> Vec<u32> {
    let mut prev: Vec<u32> = Vec::new();
    let mut out: Vec<u32> = Vec::new();
    for step in 0..64 {
        let mut logits = f3_logits(step);
        let t = sample(&mut logits, &prev);
        out.push(t);
        prev.push(t);
    }
    out
}
/// The pre-F3 chain, **reimplemented inline** so the bit-identity gate below
/// compares against an independent reference rather than the
/// `sample_with_penalties` wrapper (which now delegates to the new path and
/// would make the comparison vacuous). This is master's `sampler.rs` body,
/// verbatim: penalties -> greedy shortcut -> top-k -> top-p -> temperature.
fn pre_f3_chain(logits: &mut [f32], prev: &[u32], rng: &mut rand::rngs::StdRng) -> u32 {
    apply_penalties(logits, prev, 1.1, 0.0, 0.0);
    if 0.8f32 < 1e-6 {
        return sample_greedy(logits).token_id;
    }
    apply_top_k(logits, 40);
    apply_top_p(logits, 0.95);
    sample_temperature(logits, 0.8, rng).token_id
}
// === F2 (#47): the grammar mask in the pipeline =========================

/// A grammar over a synthetic vocabulary of `n_vocab` single-byte tokens.
fn tiny_grammar(src: &str, n_vocab: usize, eog: &[u32]) -> Arc<Grammar> {
    let pieces: Vec<Option<Box<[u8]>>> = (0..n_vocab)
        .map(|i| Some(vec![i as u8].into_boxed_slice()))
        .collect();
    let mut e = vec![false; n_vocab];
    for &i in eog {
        e[i as usize] = true;
    }
    Arc::new(Grammar::from_gbnf(src, pieces, e).expect("grammar compiles"))
}
// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `sampler.rs` (bucket B of the dead-code census —
// every test caller already lives in this module's subtree). `sample_with_penalties`
// has no test caller of its own; it is reached only through `sample`, so it moved
// with it.
// ────────────────────────────────────────────────────────────────────────────

/// Repetition penalty: penalize tokens that already appeared in `prev_tokens`.
/// `penalty == 1.0` disables. Positive logits are divided by the penalty
/// (reduced), negative logits are multiplied (pushed further down). This is
/// llama.cpp's `repeat_penalty` applied to the last `repeat_last_n` tokens.
///
/// Test-only (#239): driven by `sampler::tests::{test_repeat_penalty_reduces_repeated,
/// test_repeat_penalty_disabled_at_1, test_penalties_identity_with_old_repeat_only}`.
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    apply_penalties(logits, prev_tokens, penalty, 0.0, 0.0);
}
/// `temp < 1e-6` (greedy) skips the stochastic steps but still applies the
/// penalties.
///
/// Pre-F3 signature, kept for the pre-[`SamplerConfig`] tests. It builds a config
/// whose new knobs are all at their no-op defaults, so its output is bit-identical
/// to the pre-F3 chain (the `default_config_is_bit_identical_to_the_old_path` gate
/// pins this).
///
/// Test-only (#239): reached only through `sample` below.
pub fn sample_with_penalties<R: Rng>(
    logits: &mut [f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    prev_tokens: &[u32],
    rng: &mut R,
) -> SampledToken {
    let cfg = SamplerConfig {
        temp,
        top_k,
        top_p,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        ..SamplerConfig::default()
    };
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    sample_with_config(logits, &cfg, prev_tokens, &mut mirostat, rng)
}
/// Complete sampling pipeline with only the repeat penalty (frequency and
/// presence disabled) — the pre-F3 entry point.
///
/// Test-only (#239): driven by `sampler::tests::test_sample_pipeline_greedy_applies_penalty`
/// and the `f3_sequence` helper of the pinned-pipeline gates.
pub fn sample<R: Rng>(
    logits: &mut [f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    prev_tokens: &[u32],
    rng: &mut R,
) -> SampledToken {
    sample_with_penalties(
        logits,
        temp,
        top_k,
        top_p,
        repeat_penalty,
        0.0,
        0.0,
        prev_tokens,
        rng,
    )
}
