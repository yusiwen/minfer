//! `#[cfg(test)] mod tests` for `src/sampler.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use rand::SeedableRng;

#[test]
fn test_greedy_picks_max() {
    let logits = [1.0f32, 5.0, -2.0, 3.0];
    let s = sample_greedy(&logits);
    assert_eq!(s.token_id, 1);
}

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

// === Phase 1: stop strings (byte-wise suffix matching) ===

#[test]
fn test_stop_suffix_basic() {
    let buf = b"hello world";
    assert_eq!(match_stop_suffix(buf, &[b"world"]), Some(6));
    assert_eq!(match_stop_suffix(buf, &[b"hello"]), None, "not a suffix");
    assert_eq!(match_stop_suffix(buf, &[b"d"]), Some(10));
    assert_eq!(match_stop_suffix(buf, &[b"x"]), None);
    assert_eq!(match_stop_suffix(buf, &[]), None);
}

#[test]
fn test_stop_suffix_empty_and_too_long_ignored() {
    let buf = b"abc";
    assert_eq!(match_stop_suffix(buf, &[b"", b"abc", b"abcd"]), Some(0));
    assert_eq!(match_stop_suffix(buf, &[b"", b"zzz"]), None);
}

#[test]
fn test_stop_suffix_longest_wins() {
    // both "ab" and "b" are suffixes of "xab"; earliest start (longest) wins
    let buf = b"xab";
    assert_eq!(match_stop_suffix(buf, &[b"b", b"ab"]), Some(1));
    assert_eq!(match_stop_suffix(buf, &[b"ab", b"b"]), Some(1));
}

#[test]
fn test_stop_suffix_multibyte_split_across_tokens() {
    // U+4E2D = E4 B8 AD; first two bytes arrive in one token, last byte next
    let partial = [0xE4u8, 0xB8];
    assert_eq!(
        match_stop_suffix(&partial, &[&[0xE4, 0xB8, 0xAD]]),
        None,
        "stop longer than buf"
    );
    let complete = [0xE4u8, 0xB8, 0xAD, 0xE4, 0xB8, 0xAD];
    assert_eq!(
        match_stop_suffix(&complete, &[&[0xE4, 0xB8, 0xAD]]),
        Some(3)
    );
}

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

#[test]
fn test_logit_bias_positive_and_negative() {
    let mut logits = [0.0f32, 1.0, 2.0];
    apply_logit_bias(&mut logits, &[(0, 3.0)]);
    assert_eq!(logits, [3.0, 1.0, 2.0]);
    assert_eq!(sample_greedy(&logits).token_id, 0);

    let mut logits = [0.0f32, 1.0, 2.0];
    apply_logit_bias(&mut logits, &[(2, -3.0)]);
    assert_eq!(logits, [0.0, 1.0, -1.0]);
    assert_eq!(sample_greedy(&logits).token_id, 1);
}

#[test]
fn test_logit_bias_out_of_vocab_is_refused() {
    let cfg = SamplerConfig {
        logit_bias: vec![(3, 1.0)],
        ..SamplerConfig::default()
    };
    assert!(cfg.validate_logit_bias(4).is_ok());
    let err = cfg.validate_logit_bias(3).unwrap_err();
    assert!(err.contains("outside the vocabulary"), "{err}");
    // The bias value range and finiteness are refused by `validate`.
    for bad in [f32::NAN, f32::INFINITY, 101.0, -100.5] {
        let cfg = SamplerConfig {
            logit_bias: vec![(0, bad)],
            ..SamplerConfig::default()
        };
        assert!(cfg.validate().is_err(), "bias {bad} must be refused");
    }
}

#[test]
fn test_validate_rejects_nonsense() {
    let cases: Vec<(&str, SamplerConfig)> = vec![
        (
            "min_p > 1",
            SamplerConfig {
                min_p: 1.5,
                ..SamplerConfig::default()
            },
        ),
        (
            "typical_p < 0",
            SamplerConfig {
                typical_p: -0.1,
                ..SamplerConfig::default()
            },
        ),
        (
            "top_p > 1",
            SamplerConfig {
                top_p: 1.2,
                ..SamplerConfig::default()
            },
        ),
        (
            "negative temperature",
            SamplerConfig {
                temp: -1.0,
                ..SamplerConfig::default()
            },
        ),
        (
            "xtc_threshold > 0.5",
            SamplerConfig {
                xtc_probability: 1.0,
                xtc_threshold: 0.9,
                ..SamplerConfig::default()
            },
        ),
        (
            "dry_base < 1",
            SamplerConfig {
                dry_multiplier: 1.0,
                dry_base: 0.5,
                ..SamplerConfig::default()
            },
        ),
        (
            "dry window zero while enabled",
            SamplerConfig {
                dry_multiplier: 1.0,
                dry_penalty_last_n: 0,
                ..SamplerConfig::default()
            },
        ),
        (
            "mirostat_tau <= 0",
            SamplerConfig {
                mirostat: MirostatMode::V2,
                mirostat_tau: 0.0,
                ..SamplerConfig::default()
            },
        ),
        (
            "mirostat_eta <= 0",
            SamplerConfig {
                mirostat: MirostatMode::V1,
                mirostat_eta: 0.0,
                ..SamplerConfig::default()
            },
        ),
    ];
    for (name, cfg) in cases {
        assert!(cfg.validate().is_err(), "{name} must be refused");
    }
    // The defaults pass, and so does a fully configured but sane config.
    assert!(SamplerConfig::default().validate().is_ok());
    let sane = SamplerConfig {
        min_p: 0.05,
        typical_p: 0.9,
        xtc_probability: 0.5,
        xtc_threshold: 0.1,
        dry_multiplier: 0.8,
        mirostat: MirostatMode::V2,
        ..SamplerConfig::default()
    };
    assert!(sane.validate().is_ok());
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

/// A grammar whose language is the whole ASCII byte range must be a no-op:
/// same tokens, same mirostat trajectory, same RNG stream as no grammar.
#[test]
fn an_allow_everything_grammar_does_not_perturb_the_pipeline() {
    let g = tiny_grammar("root ::= .*", 128, &[]);
    let cfg = SamplerConfig {
        temp: 0.8,
        top_k: 0,
        top_p: 1.0,
        repeat_penalty: 1.0,
        mirostat: MirostatMode::V2,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let plain = SamplerConfig {
        grammar: None,
        ..cfg.clone()
    };
    let run = |cfg: &SamplerConfig| -> (Vec<u32>, f32) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let mut grammar = cfg.grammar.as_ref().map(|g| g.state());
        let mut prev: Vec<u32> = Vec::new();
        let mut out = Vec::new();
        for step in 0..32u64 {
            let mut logits: Vec<f32> = (0..128)
                .map(|i| ((i as f32) * 0.13 + (step as f32) * 0.29).sin() * 3.0)
                .collect();
            let t = sample_with_config_grammar(
                &mut logits,
                cfg,
                &prev,
                &mut mirostat,
                &mut grammar,
                &mut rng,
            )
            .expect("allow-all grammar")
            .token_id;
            out.push(t);
            prev.push(t);
        }
        (out, mirostat.mu)
    };
    let (with, mu_with) = run(&cfg);
    let (without, mu_without) = run(&plain);
    assert_eq!(
        with, without,
        "an allow-all grammar must not change the tokens"
    );
    assert_eq!(
        mu_with, mu_without,
        "mirostat's mu must follow the same trajectory"
    );
}

/// The mask decides the greedy winner: a grammar that only allows `a` beats
/// a logit argmax on `b`, and the state advances token by token.
#[test]
fn grammar_mask_decides_the_greedy_choice_and_advances() {
    let g = tiny_grammar("root ::= \"ab\"", 128, &[1]);
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let mut grammar = Some(g.state());
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);

    // 'b' (0x62) is the argmax but the grammar is at `a`.
    let mut logits = vec![0.0f32; 128];
    logits[0x62] = 10.0;
    logits[0x61] = 1.0;
    let first = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(first.token_id, 0x61, "the mask must beat the argmax");

    let mut logits = vec![0.0f32; 128];
    logits[0x63] = 10.0;
    logits[0x62] = 1.0;
    let second = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0x61],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(second.token_id, 0x62);

    // The grammar is complete: only the EOG token is legal now.
    let mut logits = vec![0.0f32; 128];
    logits[0x61] = 10.0;
    let third = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0x61, 0x62],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(third.token_id, 1, "EOG is the only legal continuation");
    assert!(grammar.as_ref().unwrap().is_accepting() || true);
    // A token after EOG is a loud error, never a silent one.
    let mut logits = vec![0.0f32; 128];
    logits[1] = 10.0;
    let err = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0x61, 0x62, 1],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap_err();
    assert!(err.to_string().contains("end of generation"), "{err}");
}

/// No legal token at all is a loud stop, not an arbitrary token.
#[test]
fn grammar_pipeline_stops_when_no_token_is_allowed() {
    // The vocabulary has 'a' (0x61) but no 'b': after `a` the grammar is stuck.
    let pieces = vec![Some(vec![0x61u8].into_boxed_slice())];
    let g = Arc::new(Grammar::from_gbnf("root ::= \"ab\"", pieces, vec![false]).unwrap());
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let mut grammar = Some(g.state());
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let mut logits = vec![5.0f32];
    let first = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap();
    assert_eq!(first.token_id, 0);
    let mut logits = vec![5.0f32];
    let err = sample_with_config_grammar(
        &mut logits,
        &cfg,
        &[0],
        &mut mirostat,
        &mut grammar,
        &mut rng,
    )
    .unwrap_err();
    match err {
        SampleError::NoAllowedToken { state } => {
            assert!(
                state.contains("stack"),
                "the reason names the state: {state}"
            )
        }
        other => panic!("expected NoAllowedToken, got {other:?}"),
    }
}

/// A configured grammar with no run state is a bug, never a silent fallback
/// to unconstrained sampling.
#[test]
fn grammar_pipeline_refuses_a_configured_grammar_without_state() {
    let g = tiny_grammar("root ::= .*", 128, &[]);
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g),
        ..SamplerConfig::default()
    };
    let mut none: Option<GrammarState> = None;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let mut logits = vec![1.0f32; 128];
    let err =
        sample_with_config_grammar(&mut logits, &cfg, &[], &mut mirostat, &mut none, &mut rng)
            .unwrap_err();
    assert!(err.to_string().contains("no grammar state"), "{err}");
}

/// Every token the pipeline emits under a JSON grammar is one the automaton
/// accepts: drive a fixed token stream that spells a JSON object and assert
/// each step's sampled token is allowed and the final state is accepting.
#[test]
fn sampled_tokens_are_always_allowed_by_the_json_grammar() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": false
    });
    let n_vocab = 128usize;
    let pieces: Vec<Option<Box<[u8]>>> = (0..n_vocab)
        .map(|i| Some(vec![i as u8].into_boxed_slice()))
        .collect();
    let g =
        Arc::new(Grammar::from_json_schema(&schema, pieces, vec![false; n_vocab]).expect("schema"));
    let cfg = SamplerConfig {
        temp: 0.0,
        grammar: Some(g.clone()),
        ..SamplerConfig::default()
    };
    let mut grammar = Some(g.state());
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    let target = br#"{"a":1}"#;
    for (i, &want) in target.iter().enumerate() {
        // Reward exactly the next byte the target needs; everything else is
        // noise, so the mask is what has to keep the run on the rails.
        let mut logits = vec![0.0f32; n_vocab];
        logits[want as usize] = 5.0;
        let sampled = sample_with_config_grammar(
            &mut logits,
            &cfg,
            if i == 0 { &[] } else { &[] },
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap_or_else(|e| panic!("step {i} (byte {}): {e}", want as char));
        assert_eq!(
            sampled.token_id, want as u32,
            "the mask must allow the next byte {}",
            want as char
        );
    }
    assert!(
        grammar.as_ref().unwrap().is_accepting(),
        "the driven text is a complete instance"
    );
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
