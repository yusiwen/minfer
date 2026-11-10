//! Cancelling a conversation turn (#800).
//!
//! The seam is [`TurnParams::cancel`] — the same `CancelToken` the single-shot
//! loop takes as `GenerateParams.cancel` — and the rule is the same: checked
//! once per decoded token and once per speculative round, honoured at a token
//! boundary (a forward pass cannot be interrupted), and reported as
//! `TurnOutcome::cancelled`, never as `hit_n_predict`.
//!
//! Split out of `src/conversation/tests.rs` (#267's layout rule); the fixtures
//! live in the parent module.

use super::*;
use crate::cancel::CancelToken;

#[test]
fn a_cancelled_turn_stops_between_tokens() {
    let mut c = conv(512);
    // 'H', 'i', then spares the cancel must prevent.
    let mut eng = MockEngine::new(vec![72, 105, 107, 108, 109]);
    let token = CancelToken::new();
    let mut tp = cfg();
    // A cap above the cancel point but below the program's length: without the
    // check the turn runs to `n_predict`, which is the mutation this pins.
    tp.n_predict = 4;
    tp.cancel = Some(token.clone());

    let mut emitted: Vec<u8> = Vec::new();
    let mut deltas = 0usize;
    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut |b| {
            deltas += 1;
            emitted.extend_from_slice(b);
            if deltas == 2 {
                token.cancel();
            }
        })
        .unwrap();

    assert!(out.cancelled, "the outcome names the cancellation");
    assert!(!out.hit_n_predict, "a cancel is not an n_predict hit");
    assert_eq!(out.tokens_generated, 2, "it stopped at the second token");
    assert_eq!(out.text, "Hi");
    assert_eq!(emitted, b"Hi", "the emitted bytes end at the cancel");
    // Every emitted token was forwarded, so the KV is exactly as long as the
    // stream — the invariant the next turn's positions depend on.
    assert_eq!(c.current_pos, c.stream_tokens.len());
    // The partial reply is history, and the turn owes an EOT (it never saw the
    // model's own end-of-turn), so the conversation can continue.
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("Hi"));
    assert!(c.need_insert_eot);
}

#[test]
fn a_cancelled_turn_leaves_the_next_turn_runnable() {
    let mut c = conv(512);
    // Turn 1: 'H', 'i' then a cancel. Turn 2: the EOT write (its logits are
    // never sampled), then the delta prefill, which spikes at EOG.
    let mut eng = MockEngine::new(vec![72, 105, 0, IM_END]);
    let token = CancelToken::new();
    let mut tp = cfg();
    tp.cancel = Some(token.clone());

    let mut deltas = 0usize;
    let first = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut |_b| {
            deltas += 1;
            if deltas == 2 {
                token.cancel();
            }
        })
        .unwrap();
    assert!(first.cancelled);
    // Every forward this turn took: the full render, the two emitted tokens
    // (the loop forwards a token after emitting it), and nothing else.
    let after_first = eng.calls.len();
    assert_eq!(after_first, 3);

    // A fresh token: a cancelled one never resets, and the host makes a new one
    // per turn.
    let mut tp2 = cfg();
    tp2.cancel = Some(CancelToken::new());
    let out = c
        .user_turn("Q", &FakeCodec, &tp2, &mut eng, &mut noop_emit())
        .unwrap();

    assert!(!out.cancelled, "the second turn is not cancelled");
    assert!(out.stopped_by_eog, "it reached the model's own end-of-turn");
    assert!(
        !c.need_insert_eot,
        "the pending EOT was written by this turn"
    );
    // The EOT write is the second turn's first forward (the delta prefill is
    // the one after it), which is what keeps the render canonical (§5.4).
    assert_eq!(
        eng.calls[after_first].0,
        vec![c.eot],
        "the cancelled turn's pending EOT is written first"
    );
}

#[test]
fn a_pre_cancelled_turn_generates_nothing() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![72, 105]);
    let token = CancelToken::new();
    token.cancel();
    let mut tp = cfg();
    tp.cancel = Some(token);

    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();

    assert!(out.cancelled);
    assert_eq!(out.tokens_generated, 0);
    assert_eq!(out.text, "");
    // An interrupted turn is still a turn: the empty reply is recorded and the
    // EOT is owed, so the next render stays well formed.
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some(""));
    assert!(c.need_insert_eot);
    assert_eq!(c.current_pos, c.stream_tokens.len());
}

#[test]
fn a_cancelled_continuation_stops_too() {
    let mut c = conv(512);
    // The continuation prefill pops the first spike; the decode loop pops the
    // second (its delta is emitted, then the cancel lands).
    let mut eng = MockEngine::new(vec![72, 105, 107]);
    let token = CancelToken::new();
    let mut tp = cfg();
    tp.cancel = Some(token.clone());

    let mut deltas = 0usize;
    let out = c
        .complete_tokens(
            &[b'x' as u32, b'y' as u32],
            &FakeCodec,
            &tp,
            &mut eng,
            &mut |_b| {
                deltas += 1;
                if deltas == 1 {
                    token.cancel();
                }
            },
        )
        .unwrap();

    assert!(out.cancelled, "a continuation takes the same seam");
    assert!(!out.hit_n_predict);
    assert_eq!(out.prefill_tokens, 2, "only the continuation's own tokens");
    assert_eq!(out.tokens_generated, 1);
    assert_eq!(c.current_pos, c.stream_tokens.len());
}

#[test]
fn a_cancelled_spec_turn_flushes_its_seed() {
    // doc 97: the spec loop commits whole batches, and its newest token (the
    // seed) sits at `current_pos` unwritten until the next round. A cancel at
    // the loop top must flush it — the cap exits there do the same — or the KV
    // would be a row short of the stream and the next turn's positions would
    // read the wrong rows.
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![b'z' as u32, EOS, EOS, EOS]);
    // Four batches of three: without the check the turn reaches the cap below,
    // which is the mutation this pins.
    eng.spec_batches = vec![
        vec![b'a' as u32, b'b' as u32, b'c' as u32],
        vec![b'd' as u32, b'e' as u32, b'f' as u32],
        vec![b'g' as u32, b'h' as u32, b'i' as u32],
        vec![b'j' as u32, b'k' as u32, b'l' as u32],
    ]
    .into();
    let token = CancelToken::new();
    let mut tp = cfg();
    tp.n_predict = 8;
    tp.cancel = Some(token.clone());

    let mut deltas = 0usize;
    let out = c
        .start(Some("hi"), &FakeCodec, &tp, &mut eng, &mut |_b| {
            deltas += 1;
            if deltas == 2 {
                token.cancel();
            }
        })
        .unwrap()
        .unwrap();

    assert!(out.cancelled, "the round-top check fired");
    assert!(!out.hit_n_predict);
    // A round commits its whole batch: the cancel is honoured at the round
    // boundary, and the seed ('c', the batch's last token) is flushed below.
    assert_eq!(out.text, "zabc", "the committed batch is kept");
    assert_eq!(
        c.current_pos,
        c.stream_tokens.len(),
        "the seed row was flushed"
    );
    assert!(c.need_insert_eot);
}
