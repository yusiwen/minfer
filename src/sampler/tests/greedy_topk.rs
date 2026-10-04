//! The greedy shortcut, top-k, top-p and the seeded path.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn test_greedy_picks_max() {
    let logits = [1.0f32, 5.0, -2.0, 3.0];
    let s = sample_greedy(&logits);
    assert_eq!(s.token_id, 1);
}
#[test]
fn test_top_k_filters() {
    let mut logits = [1.0f32, 5.0, 2.0, 4.0];
    apply_top_k(&mut logits, 2);
    assert!(logits[1] > 0.0); // 5.0 kept
    assert!(logits[3] > 0.0); // 4.0 kept
    assert!(logits[0].is_infinite() && logits[0] < 0.0); // 1.0 masked
    assert!(logits[2].is_infinite() && logits[2] < 0.0); // 2.0 masked
}
#[test]
fn test_top_p_nucleus() {
    let mut logits = [1.0f32, 2.0, 3.0, 4.0];
    apply_top_p(&mut logits, 0.5);
    // only the top token (index 3) should remain non-masked
    let kept: Vec<usize> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_infinite() || v > 0.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        kept,
        vec![3],
        "only the most probable token should survive p=0.5"
    );
    // logits must NOT be overwritten with probabilities
    assert!(
        (logits[3] - 4.0).abs() < 1e-6,
        "raw logit preserved: {}",
        logits[3]
    );
}
#[test]
fn test_seeded_sampling_reproducible() {
    let mut logits1 = vec![0.0f32; 100];
    for (i, v) in logits1.iter_mut().enumerate() {
        *v = (i as f32) * 0.1;
    }
    let mut logits2 = logits1.clone();
    let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
    let s1 = sample_temperature(&mut logits1, 0.8, &mut rng1);
    let s2 = sample_temperature(&mut logits2, 0.8, &mut rng2);
    assert_eq!(s1.token_id, s2.token_id, "same seed must give same token");
}
#[test]
fn test_new_samplers_are_noops_at_their_defaults() {
    let raw = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5];
    let mut logits = raw;
    apply_min_p(&mut logits, SamplerConfig::default().min_p);
    apply_typical(&mut logits, SamplerConfig::default().typical_p);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    apply_xtc(
        &mut logits,
        SamplerConfig::default().xtc_probability,
        SamplerConfig::default().xtc_threshold,
        &mut rng,
    );
    apply_dry(
        &mut logits,
        &[1, 2, 3, 1, 2],
        SamplerConfig::default().dry_multiplier,
        SamplerConfig::default().dry_base,
        SamplerConfig::default().dry_allowed_length,
        SamplerConfig::default().dry_penalty_last_n,
        &[],
    );
    apply_logit_bias(&mut logits, &SamplerConfig::default().logit_bias);
    assert_eq!(logits, raw, "every new sampler must be off by default");
}
#[test]
fn test_temperature_zero_stays_greedy_with_every_new_filter_set() {
    // A greedy request must pick the argmax even with the whole new set
    // configured — the greedy shortcut keeps its pre-F3 position.
    let cfg = SamplerConfig {
        temp: 0.0,
        min_p: 0.9,
        typical_p: 0.1,
        xtc_probability: 1.0,
        xtc_threshold: 0.5,
        dry_multiplier: 5.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 64,
        mirostat: MirostatMode::V2,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        ..SamplerConfig::default()
    };
    let mut logits = [1.0f32, 7.0, 3.0, 2.0];
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let mut rng = rand::rngs::StdRng::seed_from_u64(21);
    let s = sample_with_config(&mut logits, &cfg, &[0, 1], &mut mirostat, &mut rng);
    assert_eq!(s.token_id, 1, "temp = 0 must stay greedy");
}
