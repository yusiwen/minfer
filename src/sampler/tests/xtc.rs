//! XTC: the probability-0 no-op, the exclusion rule and the two-candidate guard.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn test_xtc_disabled_at_probability_zero_and_consumes_no_rng_draw() {
    let raw = [1.0f32, 2.0, 3.0, 4.0];
    let mut a = raw;
    let mut b = raw;
    let mut r1 = rand::rngs::StdRng::seed_from_u64(7);
    let mut r2 = rand::rngs::StdRng::seed_from_u64(7);
    apply_xtc(&mut a, 0.0, 0.5, &mut r1);
    apply_xtc(&mut b, 1.0, 0.6, &mut r2); // threshold > 0.5 is llama.cpp's "empty" XTC
    assert_eq!(a, raw);
    assert_eq!(b, raw);
    // Neither call may have drawn from the RNG.
    let x: u64 = r1.gen();
    let y: u64 = r2.gen();
    let mut r3 = rand::rngs::StdRng::seed_from_u64(7);
    let z: u64 = r3.gen();
    assert_eq!(x, z, "disabled XTC must not disturb the RNG stream");
    assert_eq!(y, z, "threshold > 0.5 must not disturb the RNG stream");
}
#[test]
fn test_xtc_excludes_the_top_choices_but_keeps_one() {
    // Probabilities: 4.0 -> 0.644, 3.0 -> 0.237, 2.0 -> 0.087, 1.0 -> 0.032.
    // threshold 0.2 => indices 0 and 1 are above it, so pos_last = 1 and the
    // top choice alone is excluded (the *last* above-threshold one stays).
    let mut logits = [4.0f32, 3.0, 2.0, 1.0];
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    apply_xtc(&mut logits, 1.0, 0.2, &mut rng);
    assert!(
        logits[0] == f32::NEG_INFINITY,
        "the top choice must be excluded: {logits:?}"
    );
    assert_eq!(logits[1], 3.0, "the above-threshold survivor stays");
    assert_eq!(logits[2], 2.0);
    assert_eq!(logits[3], 1.0);

    // threshold 0.5 => only index 0 is above it, pos_last = 0, nothing dropped.
    let mut logits = [4.0f32, 3.0, 2.0, 1.0];
    apply_xtc(&mut logits, 1.0, 0.5, &mut rng);
    assert_eq!(
        logits.iter().filter(|v| **v > f32::NEG_INFINITY).count(),
        4,
        "no candidate may be dropped when only the head is above the threshold"
    );

    // threshold 0.0 => every candidate is above it, so all but the least
    // likely are excluded — one token always remains (never empty).
    let mut logits = [4.0f32, 3.0, 2.0, 1.0];
    apply_xtc(&mut logits, 1.0, 0.0, &mut rng);
    assert_eq!(
        logits.iter().filter(|v| **v > f32::NEG_INFINITY).count(),
        1,
        "one token must remain: {logits:?}"
    );
    assert_eq!(
        logits[3], 1.0,
        "the least likely above-threshold token stays"
    );
}
#[test]
fn test_xtc_needs_two_candidates() {
    // A single candidate cannot be excluded, and the RNG must not be drawn.
    let mut logits = [5.0f32];
    let mut r1 = rand::rngs::StdRng::seed_from_u64(3);
    apply_xtc(&mut logits, 1.0, 0.5, &mut r1);
    assert_eq!(logits, [5.0]);
    let x: u64 = r1.gen();
    let mut r2 = rand::rngs::StdRng::seed_from_u64(3);
    let y: u64 = r2.gen();
    assert_eq!(x, y);
}
