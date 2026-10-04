//! Repeat / frequency / presence penalties and their window.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn test_repeat_penalty_reduces_repeated() {
    // token 3 appears in prev; with penalty 2.0 its positive logit halves
    let mut logits = [1.0f32, 2.0, 3.0, 4.0];
    apply_repetition_penalty(&mut logits, &[3], 2.0);
    assert!(
        (logits[3] - 2.0).abs() < 1e-6,
        "positive logit should halve: {}",
        logits[3]
    );
    // greedy now picks token 2 (3.0) instead of 3
    let s = sample_greedy(&logits);
    assert_eq!(s.token_id, 2);

    // negative logit gets multiplied (more negative)
    let mut logits = [-4.0f32, -2.0, -1.0, -3.0];
    apply_repetition_penalty(&mut logits, &[3], 2.0);
    assert!(
        (logits[3] - -6.0).abs() < 1e-6,
        "negative logit should double: {}",
        logits[3]
    );
}
#[test]
fn test_repeat_penalty_disabled_at_1() {
    let mut logits = [1.0f32, 2.0, 3.0];
    let before = logits.clone();
    apply_repetition_penalty(&mut logits, &[0, 1], 1.0);
    assert_eq!(logits, before);
}
#[test]
fn test_sample_pipeline_greedy_applies_penalty() {
    let mut logits = [1.0f32, 2.0, 3.0, 4.0];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let s = sample(&mut logits, 0.0, 40, 0.95, 2.0, &[3], &mut rng);
    assert_eq!(
        s.token_id, 2,
        "greedy + penalty should avoid the penalized token"
    );
}
// === Phase 1 (OPENAI-CHAT-API-PLAN.md): frequency/presence penalties ===

#[test]
fn test_frequency_penalty_scales_with_count() {
    // token 3 appears twice in the window: logit -= 2 * 0.5 = 1.0
    let mut logits = [1.0f32, 2.0, 3.0, 4.0];
    apply_penalties(&mut logits, &[3, 3], 1.0, 0.5, 0.0);
    assert!(
        (logits[3] - 3.0).abs() < 1e-6,
        "2x0.5 subtracted: {}",
        logits[3]
    );
    // others untouched when repeat == 1.0
    assert_eq!(logits[0], 1.0);
    assert_eq!(logits[1], 2.0);
    assert_eq!(logits[2], 3.0);
}
#[test]
fn test_presence_penalty_applied_once() {
    // token 3 present (once or twice) => logit -= 0.8, no count scaling
    let mut a = [1.0f32, 2.0, 3.0, 4.0];
    apply_penalties(&mut a, &[3], 1.0, 0.0, 0.8);
    assert!((a[3] - 3.2).abs() < 1e-6, "presence once: {}", a[3]);
    let mut b = [1.0f32, 2.0, 3.0, 4.0];
    apply_penalties(&mut b, &[3, 3], 1.0, 0.0, 0.8);
    assert!(
        (b[3] - 3.2).abs() < 1e-6,
        "presence is per-token, not per-occurrence: {}",
        b[3]
    );
}
#[test]
fn test_freq_presence_then_repeat_penalty() {
    // llama.cpp order: subtract freq/presence, then apply repeat (÷ or ×)
    let mut logits = [4.0f32, -4.0, 0.0, 0.0];
    // token 0: 4.0 - 1*1.0(freq) - 1.0(presence) = 2.0, repeat 2.0 => 1.0
    apply_penalties(&mut logits, &[0], 2.0, 1.0, 1.0);
    assert!(
        (logits[0] - 1.0).abs() < 1e-6,
        "4 - 2 then /2: {}",
        logits[0]
    );
    // tokens not in the window are untouched
    assert_eq!(logits[1], -4.0);
    assert_eq!(logits[2], 0.0);

    // negative logit in the window: repeat multiplies (no freq/presence)
    let mut logits = [4.0f32, -4.0, 0.0, 0.0];
    apply_penalties(&mut logits, &[1], 2.0, 0.0, 0.0);
    assert!(
        (logits[1] - -8.0).abs() < 1e-6,
        "negative * repeat: {}",
        logits[1]
    );
    assert_eq!(logits[0], 4.0);
    assert_eq!(logits[2], 0.0);
}
#[test]
fn test_penalties_disabled_at_defaults() {
    let mut logits = [1.0f32, 2.0, 3.0, 4.0];
    let before = logits.clone();
    apply_penalties(&mut logits, &[1, 2, 3], 1.0, 0.0, 0.0);
    assert_eq!(logits, before, "repeat=1, freq=0, presence=0 is a no-op");
}
#[test]
fn test_penalties_identity_with_old_repeat_only() {
    // freq=presence=0 must reproduce apply_repetition_penalty exactly
    let mut a = [1.0f32, 2.0, 3.0, 4.0, -2.0, -5.0];
    let mut b = a.clone();
    apply_repetition_penalty(&mut a, &[3, 4], 2.0);
    apply_penalties(&mut b, &[3, 4], 2.0, 0.0, 0.0);
    assert_eq!(a, b, "combined pass must be identical to repeat-only");
}
#[test]
fn test_penalty_out_of_range_token_skipped() {
    let mut logits = [1.0f32, 2.0];
    apply_penalties(&mut logits, &[99, 99], 2.0, 1.0, 1.0);
    assert_eq!(logits, [1.0, 2.0]);
}
#[test]
fn test_recent_window_tail() {
    let tokens = [0u32, 1, 2, 3, 4, 5];
    assert_eq!(recent_window(&tokens, 3), vec![3, 4, 5]);
    assert_eq!(recent_window(&tokens, 64), vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(recent_window(&tokens, 0), Vec::<u32>::new());
    assert_eq!(recent_window(&[], 64), Vec::<u32>::new());
}
