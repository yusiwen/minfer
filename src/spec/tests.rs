//! `#[cfg(test)] mod tests` for `src/spec.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use rand::SeedableRng;

#[test]
fn adaptive_picks_deep_when_acceptance_stays_high() {
    // code-like acceptance (~0.8 flat) with the doc-94 cost curve: the
    // deep-depth expected tokens amortize the verify round -> d_max wins.
    let mut ad = AdaptiveD::new(8);
    for _ in 0..40 {
        ad.observe_round(8, 8); // everything accepted at every depth
    }
    let d = ad.pick(64);
    assert_eq!(d, 8, "flat-high acceptance should pick the cap, got {d}");
}

#[test]
fn adaptive_picks_shallow_when_acceptance_collapses() {
    // prose-like: p1 ~0.5, deeper depths collapse — deep rounds pay the
    // C_T(9) premium for nothing -> the controller must stay shallow.
    let mut ad = AdaptiveD::new(8);
    for _ in 0..40 {
        ad.observe_round(8, 1); // only the first draft ever survives
    }
    let d = ad.pick(64);
    assert!(d <= 2, "collapsing acceptance should pick d<=2, got {d}");
}

#[test]
fn adaptive_respects_horizon_and_explores() {
    let mut ad = AdaptiveD::new(8);
    assert_eq!(ad.pick(0), 0, "spent horizon -> the d==0 fallback");
    assert_eq!(ad.pick(3), 3, "horizon caps the pick (prior curve)");
}

#[test]
fn adaptive_online_cost_observations_shift_the_pick() {
    // With flat-high acceptance but an absurd observed deep-verify cost,
    // the controller must back off the cap (online correction beats the
    // hardcoded prior).
    let mut ad = AdaptiveD::new(8);
    for _ in 0..30 {
        ad.observe_round(8, 8);
    }
    for _ in 0..30 {
        ad.observe_verify(9, 400.0); // 400 ms per nt=9 verify
    }
    let d = ad.pick(64);
    assert!(d < 8, "prohibitive deep-verify cost must back off, got {d}");
}

fn sampler() -> SpecSampler {
    SpecSampler {
        cfg: crate::sampler::SamplerConfig {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            ..crate::sampler::SamplerConfig::default()
        },
    }
}

/// Rows of d+1 tokens x nv logits; row i's argmax is exactly wants[i].
fn rows(wants: &[u32], nv: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; wants.len() * nv];
    for (i, &w) in wants.iter().enumerate() {
        v[i * nv + w as usize] = 2.0;
    }
    v
}

#[test]
fn full_accept_emits_d_plus_1() {
    let mut prev = vec![7u32];
    let mut rng = StdRng::seed_from_u64(42);
    let logits = rows(&[10, 11, 12], 32);
    // proposals match rows 0 and 1 (d=2); row 2 is the bonus.
    let (emitted, accepted) =
        accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0, 0);
    assert_eq!(emitted, vec![10, 11, 12]);
    assert_eq!(accepted, 2);
    // prev_tokens grew by the two accepted proposals + the bonus.
    assert_eq!(prev, vec![7, 10, 11, 12]);
}

#[test]
fn reject_at_row1_emits_prefix_plus_bonus() {
    let mut prev = vec![];
    let mut rng = StdRng::seed_from_u64(42);
    let logits = rows(&[10, 31, 12], 32); // row 1 disagrees with proposal 11
    let (emitted, accepted) =
        accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0, 0);
    assert_eq!(emitted, vec![10, 31]);
    assert_eq!(accepted, 1);
    assert_eq!(prev, vec![10, 31]);
}

#[test]
fn reject_at_row0_emits_bonus_only() {
    let mut prev = vec![];
    let mut rng = StdRng::seed_from_u64(42);
    let logits = rows(&[31, 11, 12], 32);
    let (emitted, accepted) =
        accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0, 0);
    assert_eq!(emitted, vec![31]);
    assert_eq!(accepted, 0);
}

/// The lazy rule: row 1's penalty window must include the accepted u1
/// from row 0 — repeat-penalty demotes a repeated token so the bonus
/// flips to the runner-up, which the serial path would also pick.
#[test]
fn lazy_penalty_window_sees_intra_round_tokens() {
    let mut prev = vec![];
    let mut rng = StdRng::seed_from_u64(42);
    let mut s = sampler();
    s.cfg.repeat_penalty = 1.5;
    // nv=4: row 0 argmax = 1 (accepted u1=1); row 1 raw argmax = 1 again,
    // runner-up = 2. With u1=1 in the penalty window, the penalized
    // row-1 sample must be 2, not 1.
    let mut logits = rows(&[1, 1], 4);
    logits[1 * 4 + 2] = 1.4; // runner-up wins once the 1.5 penalty demotes the repeat
    let (emitted, accepted) = accept_loop(&logits, &[1], &s, &mut prev, &mut rng, 0, 0, 0);
    assert_eq!(accepted, 1);
    assert_eq!(emitted, vec![1, 2]);
    assert_eq!(prev, vec![1, 2]);
}
