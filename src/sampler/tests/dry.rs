//! DRY: the empty history, the exponential scale, breakers and restarts.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn test_dry_empty_history_and_short_window_are_noops() {
    let raw = [1.0f32, 2.0, 3.0];
    let mut logits = raw;
    apply_dry(&mut logits, &[], 2.0, 1.75, 2, 64, &[]);
    assert_eq!(logits, raw, "DRY with an empty history must be a no-op");

    // A window no longer than allowed_length cannot contain a repeat.
    let mut logits = raw;
    apply_dry(&mut logits, &[1, 2], 2.0, 1.75, 2, 64, &[]);
    assert_eq!(logits, raw);

    // Disabled multiplier / base / window are no-ops too.
    for (mult, base, window) in [(0.0f32, 1.75f32, 64usize), (2.0, 0.5, 64), (2.0, 1.75, 0)] {
        let mut logits = raw;
        apply_dry(&mut logits, &[1, 2, 1, 2], mult, base, 2, window, &[]);
        assert_eq!(logits, raw, "disabled DRY (m={mult} b={base} n={window})");
    }
}
#[test]
fn test_dry_penalizes_the_repeated_continuation() {
    // History [10, 11, 12, 10, 11]: the suffix "10 11" repeats, so token 12
    // would extend a length-2 repeat => exponent 0 => penalty = multiplier.
    let mut logits = [0.0f32; 16];
    logits[12] = 5.0;
    apply_dry(&mut logits, &[10, 11, 12, 10, 11], 2.0, 1.75, 2, 64, &[]);
    assert!(
        (logits[12] - 3.0).abs() < 1e-5,
        "token 12 must lose exactly the multiplier: {}",
        logits[12]
    );
    assert_eq!(logits[13], 0.0, "uninvolved tokens are untouched");
}
#[test]
fn test_dry_scales_exponentially_with_the_repeat_length() {
    // History [10, 11, 12, 13, 10, 11, 12]: "10 11 12" repeats, so token 13
    // extends a length-3 repeat => exponent 1 => penalty = multiplier * base.
    let mut logits = [0.0f32; 16];
    logits[13] = 9.0;
    apply_dry(
        &mut logits,
        &[10, 11, 12, 13, 10, 11, 12],
        2.0,
        1.75,
        2,
        64,
        &[],
    );
    let expected = 9.0 - 2.0 * 1.75;
    assert!(
        (logits[13] - expected).abs() < 1e-4,
        "exponent-1 penalty: {} vs {expected}",
        logits[13]
    );
}
#[test]
fn test_dry_restart_sequence_caps_the_repetition() {
    // Same history as the exponential case, but a breaker head at distance 1
    // bounds rep_limit to 1 < allowed_length => DRY stands down entirely.
    let mut logits = [0.0f32; 16];
    logits[13] = 9.0;
    apply_dry(
        &mut logits,
        &[10, 11, 12, 13, 10, 11, 12],
        2.0,
        1.75,
        2,
        64,
        &[vec![11]],
    );
    assert_eq!(logits[13], 9.0, "the restart sequence must suppress DRY");
}
#[test]
fn test_dry_single_token_breaker_is_exempt() {
    // A single-token breaker is meant to be repeated (a newline); DRY must
    // not penalise the token itself.
    let mut logits = [0.0f32; 16];
    logits[12] = 5.0;
    apply_dry(
        &mut logits,
        &[10, 11, 12, 10, 11],
        2.0,
        1.75,
        2,
        64,
        &[vec![12]],
    );
    assert_eq!(logits[12], 5.0);
}
