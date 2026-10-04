//! Regeneration and clear: rollback, re-render and the assistant guard.
//!
//! Split out of `src/conversation/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn regen_rolls_back_and_regenerates() {
    let mut c = conv(512);
    // t1: EOG, placeholder; t2: 'P', EOG, placeholder; regen: 'W', EOG, placeholder
    let mut eng = MockEngine::new(vec![IM_END, EOS, 80, IM_END, EOS, 87, IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.messages.last().unwrap().0, "assistant");
    let turn2_start = c.turn_pos;

    // regen t2: roll back to turn2_start, replay t2's delta, generate 'W'
    let mut eng2 = MockEngine::new(vec![87, IM_END, EOS]); // 'W', EOG, placeholder
    let out = c
        .regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(out.text, "W");
    assert!(out.stopped_by_eog);
    // messages: user hi, assistant "", user Q, assistant W
    assert_eq!(c.messages.len(), 4);
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("W"));
    // t2's old content 'P' has been removed from the stream
    assert!(!c.stream_tokens.contains(&80));
    assert_eq!(c.turn_pos, turn2_start);
}
#[test]
fn regen_first_turn_uses_full_render() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(c.turn_pos, 0);

    // regen t1: turn_pos == 0 → full render (the engine's first call must be the full render, not a suffix)
    let mut eng2 = MockEngine::new(vec![88, IM_END, EOS]); // 'X', EOG, placeholder
    let out = c
        .regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(out.text, "X");
    let full = fallback_full(&[("user".into(), Some("hi".into()))]);
    assert_eq!(
        eng2.calls[0].0, full,
        "first call must be the full first-turn render"
    );
}
#[test]
fn regen_without_assistant_errors() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![]);
    let err = c
        .regen_turn(&FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap_err();
    assert!(matches!(err, ConvError::NothingToRegen));
}
#[test]
fn clear_resets_state_keeps_system() {
    let mut s = spec(512);
    s.system_prompt = Some("Be nice.".into());
    let mut c = Conversation::new(s);
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    c.clear();
    assert_eq!(c.messages, vec![("system".into(), Some("Be nice.".into()))]);
    assert!(c.stream_tokens.is_empty());
    assert_eq!(c.current_pos, 0);
    assert_eq!(c.turn_pos, 0);
    assert!(!c.need_insert_eot);
}
