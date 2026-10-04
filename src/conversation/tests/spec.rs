//! Speculative rounds inside a conversation.
//!
//! Split out of `src/conversation/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn spec_rounds_commit_batches_and_stop_at_eog() {
    // doc 97: the spec loop must mirror the plain loop's stream/window
    // semantics. The mock's spec batches replace the sampling: the seed
    // comes from the prefill spike, then each round commits one batch.
    let mut c = conv(512);
    // program[0] spikes the prefill logits → the turn's seed ('z');
    // the spares back the mock's per-forward logits afterwards.
    let mut eng = MockEngine::new(vec![b'z' as u32, EOS, EOS, EOS]);
    // The prefill consumes program[0] (its spike seeds the turn); the
    // remaining spikes back the mock's per-call logits if a plain forward
    // ever runs (it should not, beyond the EOG slot write).
    eng.spec_batches = vec![
        vec![b'a' as u32, b'b' as u32, b'c' as u32], // round 1 batch
        vec![IM_END],                                // round 2: EOG in-batch
    ]
    .into();
    let out = c
        .start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap()
        .unwrap();
    // The seed ('z', sampled from the prefill spike) is the first
    // committed token; the batch follows it.
    assert_eq!(out.text, "zabc");
    assert!(out.stopped_by_eog);
    assert!(!out.stopped_by_string);
    // The EOG is not counted (plain-loop mirror).
    assert_eq!(out.tokens_generated, 4);
    // The assistant message holds the batch text; the stream holds the
    // committed tokens plus the EOG.
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("zabc"));
    assert!(c.stream_tokens.contains(&(b'a' as u32)));
    assert!(c.stream_tokens.contains(&IM_END));
    // need_insert_eot stays false: the turn reached EOG.
    assert!(!c.need_insert_eot);
}
#[test]
fn spec_round_mid_batch_stop_string_truncates() {
    // The stop string lands mid-batch: committed tokens stop at the cut,
    // the stop tokens are not part of the canonical text.
    let mut cfgv = cfg();
    cfgv.stop_strings = vec!["bc".to_string()];
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![b'z' as u32, EOS, EOS]);
    eng.spec_batches = vec![vec![b'a' as u32, b'b' as u32, b'c' as u32, b'd' as u32]].into();
    let out = c
        .start(Some("hi"), &FakeCodec, &cfgv, &mut eng, &mut noop_emit())
        .unwrap()
        .unwrap();
    // The cut lands mid-batch (at 'c'): committed text stops before the
    // stop string, the batch tail ('d') is discarded uncommitted. The
    // tokens through the stop-completing one are counted (the plain loop
    // increments n_gen before its stop check — mirror exactly).
    assert_eq!(out.text, "za");
    assert!(out.stopped_by_string);
    assert!(!out.stopped_by_eog);
    assert_eq!(out.tokens_generated, 4);
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("za"));
}
