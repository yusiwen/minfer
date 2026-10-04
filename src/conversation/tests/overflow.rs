//! C2 overflow: the error, the in-decode stop and the shift/truncate paths.
//!
//! Split out of `src/conversation/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn context_full_errors_before_prefill() {
    let mut c = conv(32); // a very small n_ctx
    let mut eng = MockEngine::new(vec![IM_END]);
    let long_input = "x".repeat(64);
    let err = c
        .user_turn(&long_input, &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap_err();
    assert!(matches!(err, ConvError::ContextFull { .. }));
    // State is not polluted: no forward calls at all
    assert!(eng.calls.is_empty());
    assert!(c.messages.is_empty());
}
#[test]
fn context_fill_during_decode_stops_cleanly() {
    // fallback([user hi], true) tokenizes to 21 tokens (special markers are 1 token each)
    let mut c = conv(30);
    // The first turn's delta takes 21; decoding fills all the way up to n_ctx=30
    let mut eng = MockEngine::new(vec![72, 105, 33, 34, 35, 36, 37, 38, 39, 40, IM_END]);
    let mut tp = cfg();
    tp.n_predict = 100;
    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(
        out.hit_n_predict,
        "context full should stop as cleanly as n_predict exhaustion"
    );
    assert!(c.current_pos <= 30);
    assert!(c.need_insert_eot);
}
// === Phase 3: overflow truncation + session persistence ===

/// C2: with a shiftable engine, an overflowing turn removes the dropped
/// turn's rows from the KV instead of re-rendering, so it prefills only its
/// own delta — and the resulting stream is still the canonical render of the
/// new message list (the §5.4 invariant), which is what makes the shortcut
/// safe to take.
#[test]
fn overflow_shift_removes_the_turn_and_prefills_only_the_delta() {
    let mut c = conv(50);
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.stream_tokens.len(), 44);

    let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
    eng3.shiftable = true;
    eng3.rows = c.current_pos; // the stub mirrors the session's written rows
    let out = c
        .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
        .unwrap();

    assert_eq!(out.dropped_turns, 1, "the oldest turn is dropped");
    assert_eq!(eng3.resets, 0, "a shift must not reset the cache");
    assert_eq!(eng3.shifts.len(), 1, "exactly one KV removal");
    let (start, len) = eng3.shifts[0];
    // "hi" and its reply are gone; Q's turn is still at the stream head.
    assert_eq!(start, 0, "Q's turn starts at the stream head");
    assert_eq!(
        len, 23,
        "the dropped turn's span, including its trailing newline"
    );
    // The overflow turn prefills its own delta, not the whole render.
    // (`c.messages` now ends with the new reply; the canonical prompt is the
    // messages as they were when the turn's prefill ran.)
    let full = fallback_full(&c.messages[..c.messages.len() - 1]);
    let delta = format_single(
        None,
        &c.messages[..c.messages.len() - 2],
        ("user".to_string(), Some("X".to_string())),
        true,
        "",
    )
    .expect("diff render");
    assert_eq!(
        out.prefill_tokens,
        FakeCodec.encode(&delta.text).len(),
        "only the overflow turn's own delta is prefilled"
    );
    assert!(
        out.prefill_tokens < full.len(),
        "the shift must not re-prefill the render ({} vs {})",
        out.prefill_tokens,
        full.len()
    );

    // The §5.4 invariant survives the removal: the stream is exactly the
    // canonical render of the final message list.
    assert_eq!(c.messages.len(), 4);
    assert_eq!(c.stream_tokens, [&full[..], &[IM_END]].concat());
    assert_eq!(c.current_pos, c.stream_tokens.len());
}
/// C2: a system prompt is *not* part of the removed region — the drop starts
/// after it, so the shift is a middle removal (`start > 0`), and the system
/// prompt's rows are not even re-roped.
#[test]
fn overflow_shift_keeps_the_system_prompt_in_place() {
    let mut spec = spec(60);
    spec.system_prompt = Some("sys".to_string());
    let mut c = Conversation::new(spec);
    // The system prompt is part of the first turn's delta; the boundary check
    // must still resolve it (ChatML renders it as its own block).
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    eng.shiftable = true;
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let sys_tokens = canonical(&[("system".into(), Some("sys".into()))]);
    let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
    eng3.shiftable = true;
    eng3.rows = c.current_pos; // the stub mirrors the session's written rows
    let out = c
        .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
        .unwrap();
    assert_eq!(out.dropped_turns, 1);
    assert_eq!(eng3.shifts.len(), 1, "the shift must fire");
    let (start, _) = eng3.shifts[0];
    assert_eq!(
        start,
        sys_tokens.len(),
        "the removed region starts right after the system prompt"
    );
    assert_eq!(
        &c.stream_tokens[..sys_tokens.len()],
        &sys_tokens[..],
        "the system prompt's tokens stay at the head"
    );
    assert_eq!(
        c.messages[0],
        ("system".into(), Some("sys".into())),
        "the system prompt survives the drop"
    );
}
#[test]
fn overflow_truncates_oldest_turns_and_rehydrates() {
    // n_ctx=50: turn1(22) + turn2(44) both fit; turn3's delta makes
    // current_pos + delta > 50 → drop the oldest user+assistant pair and fully re-render.
    let mut c = conv(50);
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.stream_tokens.len(), 22, "t1: 21-token render + EOG");
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.stream_tokens.len(), 44, "t2: +21 delta + EOG");

    // t3 triggers overflow: re-render (engine.reset) + drop 1 turn + full render
    let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
    let out = c
        .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
        .unwrap();
    assert_eq!(out.dropped_turns, 1, "oldest turn must be dropped");
    assert_eq!(eng3.resets, 1, "rehydrate must reset the engine cache");
    // Messages: the oldest [user hi, assistant] pair has been dropped
    assert_eq!(
        c.messages,
        vec![
            ("user".into(), Some("Q".into())),
            ("assistant".into(), Some("".into())),
            ("user".into(), Some("X".into())),
            ("assistant".into(), Some("".into())),
        ]
    );
    // KV = the full render after re-render (turn_pos reset to zero) + EOG
    let canon = FakeCodec.encode(&template::fallback_chatml_messages(
        &[
            ("user".into(), Some("Q".into())),
            ("assistant".into(), Some("".into())),
        ],
        false,
    ));
    assert_eq!(c.turn_pos, 0);
    let full = FakeCodec.encode(&template::fallback_chatml_messages(
        &[
            ("user".into(), Some("Q".into())),
            ("assistant".into(), Some("".into())),
            ("user".into(), Some("X".into())),
        ],
        true,
    ));
    assert_eq!(c.stream_tokens, [&full[..], &[IM_END]].concat());
    assert_eq!(
        c.stream_tokens.len(),
        canon.len() + full.len() - canon.len() + 1
    );
}
#[test]
fn overflow_single_message_still_errors() {
    let mut c = conv(20); // smaller than a single message's render length
    let mut eng = MockEngine::new(vec![]);
    let err = c
        .user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap_err();
    assert!(matches!(err, ConvError::ContextFull { .. }));
}
