//! Logit bias and `SamplerConfig::validate` refusals.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
