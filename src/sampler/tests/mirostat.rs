//! Mirostat v1/v2: the mode parse, mu updates and the degenerate rule.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn test_mirostat_mode_parse() {
    assert_eq!(MirostatMode::parse(0), Ok(MirostatMode::Off));
    assert_eq!(MirostatMode::parse(1), Ok(MirostatMode::V1));
    assert_eq!(MirostatMode::parse(2), Ok(MirostatMode::V2));
    assert!(MirostatMode::parse(3).is_err());
    assert!(MirostatMode::parse(-1).is_err());
}
#[test]
fn test_mirostat_v2_truncates_and_updates_mu() {
    // A dominant head: with mu = 2*tau = 10 bits every tail token's surprise
    // (~14 bits) exceeds mu, so only the head survives and the step is
    // deterministic. Observed surprise 0 => mu rises by eta*tau.
    let mut mu = 10.0f32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let logits = [10.0f32, 0.0, 0.0, 0.0];
    let s = sample_mirostat_v2(&logits, &mut mu, 5.0, 0.1, &mut rng);
    assert_eq!(s.token_id, 0);
    assert!(
        (mu - 10.5).abs() < 1e-4,
        "mu must move by -eta*(0 - tau): {mu}"
    );
}
#[test]
fn test_mirostat_v2_never_empties_and_stays_finite() {
    // mu <= 0 would truncate everything; the at-least-one rule keeps the head.
    let mut mu = -1.0f32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let logits = [1.0f32, 0.0, 0.0];
    let s = sample_mirostat_v2(&logits, &mut mu, 5.0, 0.1, &mut rng);
    assert_eq!(s.token_id, 0, "the head must survive any mu");
    assert!(mu.is_finite());

    // A one-token candidate set cannot panic or divide by zero.
    let mut mu = 10.0f32;
    let s = sample_mirostat_v2(&[3.0f32], &mut mu, 5.0, 0.1, &mut rng);
    assert_eq!(s.token_id, 0);
    assert!(mu.is_finite() && (mu - 10.5).abs() < 1e-4);

    // An empty candidate set returns a defined value without touching mu.
    let mut mu = 10.0f32;
    let s = sample_mirostat_v2(&[], &mut mu, 5.0, 0.1, &mut rng);
    assert_eq!(s.token_id, 0);
    assert_eq!(mu, 10.0);
}
#[test]
fn test_mirostat_v1_bounds_and_degenerate_rule() {
    // Degenerate (one candidate): the documented rule pins s_hat = 1 => k = 1,
    // the argmax is sampled, and mu still moves by eta*tau.
    let mut logits = [1.0f32];
    let mut mu = 10.0f32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(13);
    let s = sample_mirostat_v1(&mut logits, &mut mu, 5.0, 0.1, 100, &mut rng);
    assert_eq!(s.token_id, 0);
    assert!(
        (mu - 10.5).abs() < 1e-4,
        "degenerate v1 must still update mu: {mu}"
    );

    // A wide mu keeps the whole distribution; the result is deterministic
    // for a fixed seed and mu stays finite and inside the documented range
    // (mu > 0 keeps the truncation non-degenerate).
    let mut logits = [2.0f32, 1.0, 0.0, -1.0];
    let mut mu = 10.0f32;
    let mut rng_a = rand::rngs::StdRng::seed_from_u64(99);
    let a = sample_mirostat_v1(&mut logits.clone(), &mut mu, 5.0, 0.1, 100, &mut rng_a);
    let mu_a = mu;
    let mut mu_b = 10.0f32;
    let mut rng_b = rand::rngs::StdRng::seed_from_u64(99);
    let b = sample_mirostat_v1(&mut logits, &mut mu_b, 5.0, 0.1, 100, &mut rng_b);
    assert_eq!(a.token_id, b.token_id);
    assert_eq!(mu_a, mu_b);
    assert!(mu_a.is_finite() && mu_a > 0.0);
    assert!((0..4).contains(&a.token_id));
}
