//! min-p and typical sampling, boundaries included.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn test_min_p_boundaries() {
    // p = 0 disables the filter.
    let mut logits = [1.0f32, 2.0, 3.0, 4.0];
    let before = logits;
    apply_min_p(&mut logits, 0.0);
    assert_eq!(logits, before, "min_p = 0 must be a no-op");

    // p = 1 keeps the argmax and its exact ties (max + ln(1) = max).
    let mut logits = [1.0f32, 4.0, 4.0, 2.0];
    apply_min_p(&mut logits, 1.0);
    assert_eq!(logits, [f32::NEG_INFINITY, 4.0, 4.0, f32::NEG_INFINITY]);

    // p = 0.5: ln(0.5) = -0.6931…, so 4.0 and 3.5 stay, 2.0 goes.
    let mut logits = [4.0f32, 3.5, 2.0, -8.0];
    apply_min_p(&mut logits, 0.5);
    assert_eq!(logits[0], 4.0);
    assert_eq!(logits[1], 3.5);
    assert!(logits[2].is_infinite() && logits[2] < 0.0);
    assert!(logits[3].is_infinite() && logits[3] < 0.0);

    // Degenerate inputs: empty, single token, all masked.
    let mut empty: [f32; 0] = [];
    apply_min_p(&mut empty, 0.5);
    let mut one = [7.0f32];
    apply_min_p(&mut one, 0.5);
    assert_eq!(one, [7.0]);
    let mut masked = [f32::NEG_INFINITY; 3];
    apply_min_p(&mut masked, 0.5);
    assert!(masked.iter().all(|v| *v == f32::NEG_INFINITY));
}
#[test]
fn test_min_p_never_empties_the_distribution() {
    // p > 1 would put the threshold above max; the argmax must survive
    // (llama.cpp's min_keep >= 1 guarantee), never an empty candidate set.
    let mut logits = [1.0f32, 9.0, 2.0];
    apply_min_p(&mut logits, 5.0);
    assert_eq!(logits, [f32::NEG_INFINITY, 9.0, f32::NEG_INFINITY]);
}
#[test]
fn test_typical_disabled_and_boundaries() {
    let raw = [1.0f32, 2.0, 3.0, 4.0];
    // typical_p = 1.0 disables.
    let mut logits = raw;
    apply_typical(&mut logits, 1.0);
    assert_eq!(logits, raw, "typical_p = 1 must be a no-op");

    // typical_p = 0 keeps exactly the single most typical token.
    let mut logits = raw;
    apply_typical(&mut logits, 0.0);
    let kept = logits.iter().filter(|v| **v > f32::NEG_INFINITY).count();
    assert_eq!(kept, 1, "typical_p = 0 keeps one token, got {logits:?}");

    // A dominated distribution: only the head is locally typical.
    let mut logits = [20.0f32, 0.0, 0.0, 0.0];
    apply_typical(&mut logits, 0.5);
    assert_eq!(
        logits[0], 20.0,
        "the dominant token must survive typical filtering"
    );
    assert!(
        logits[1..].iter().all(|v| *v == f32::NEG_INFINITY),
        "the tail must be cut: {logits:?}"
    );

    // Empty and one-token candidate sets are no-ops.
    let mut empty: [f32; 0] = [];
    apply_typical(&mut empty, 0.5);
    let mut one = [3.0f32];
    apply_typical(&mut one, 0.5);
    assert_eq!(one, [3.0]);
}
#[test]
fn test_typical_ties_keep_the_raw_logits_and_a_contiguous_prefix() {
    // Four equal logits: every score is identical, so the stable sort keeps
    // index order and p >= 0.5 keeps at least the first two.
    let mut logits = [1.0f32; 4];
    apply_typical(&mut logits, 0.5);
    let kept: Vec<usize> = logits
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > f32::NEG_INFINITY)
        .map(|(i, _)| i)
        .collect();
    assert!(kept.len() >= 2 && kept.len() <= 4, "kept {kept:?}");
    assert_eq!(kept[0], 0, "index order decides ties: {kept:?}");
    // Raw logits are never overwritten with probabilities.
    assert_eq!(logits[0], 1.0);
}
