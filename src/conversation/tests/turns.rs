//! Turn bookkeeping: render, EOG, deltas and the per-turn reseeds.
//!
//! Split out of `src/conversation/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn start_no_input_waits() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![]);
    let out = c
        .start(None, &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert!(out.is_none());
    assert!(c.messages.is_empty());
    assert_eq!(c.current_pos, 0);
}
#[test]
fn first_turn_full_render_and_eog() {
    let mut c = conv(512);
    // 'H','i', EOG, EOG-decode placeholder
    let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS]);
    let mut emitted: Vec<u8> = Vec::new();
    let out = c
        .start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut |b| {
            emitted.extend_from_slice(b)
        })
        .unwrap()
        .unwrap();
    assert!(out.stopped_by_eog);
    assert_eq!(out.text, "Hi");
    // First turn: full render (with generation prompt) + generation + EOG (§5.4: EOG enters the KV)
    let full = fallback_full(&[("user".into(), Some("hi".into()))]);
    assert_eq!(out.prefill_tokens, full.len());
    assert_eq!(c.stream_tokens, [&full[..], &[72, 105, IM_END]].concat());
    assert_eq!(
        c.messages,
        vec![
            ("user".into(), Some("hi".into())),
            ("assistant".into(), Some("Hi".into())),
        ]
    );
    assert!(!c.need_insert_eot);
    assert_eq!(emitted, b"Hi");
}
#[test]
fn second_turn_appends_only_delta() {
    let mut c = conv(512);
    // t1: EOG, EOG-decode placeholder; t2: EOG, EOG-decode placeholder
    let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let len_after_t1 = c.stream_tokens.len();
    let t2 = c
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert!(t2.stopped_by_eog);
    // delta = "\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n" (with newline compensation)
    let delta = FakeCodec.encode("\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(t2.prefill_tokens, delta.len());
    // Incrementality: only the delta + this turn's EOG are appended (hit immediately, no generated tokens)
    assert_eq!(c.stream_tokens.len(), len_after_t1 + delta.len() + 1);
    assert_eq!(
        &c.stream_tokens[len_after_t1..len_after_t1 + delta.len()],
        &delta[..]
    );
    assert_eq!(c.stream_tokens.last(), Some(&IM_END));
    // turn_pos points to the start of t2's delta (= the position after t1)
    assert_eq!(c.turn_pos, len_after_t1);
    // The assistant message is recorded (immediate EOG this turn → empty text)
    assert_eq!(c.messages.last().unwrap().1.as_deref(), Some(""));
}
#[test]
fn stop_string_truncates_and_sets_eot() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
    let mut tp = cfg();
    tp.stop_strings = vec!["!".to_string()];
    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(out.stopped_by_string);
    assert!(!out.stopped_by_eog);
    assert_eq!(out.text, "Hi");
    assert!(
        c.need_insert_eot,
        "stop-string termination → EOT needed before the next turn"
    );

    // The next turn inserts EOT first (the engine receives the [eot] call), then runs the delta.
    let mut eng2 = MockEngine::new(vec![999, IM_END]); // 999 = dummy for the EOT insertion
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(eng2.calls[0].0, vec![IM_END], "first call must insert EOT");
    assert_eq!(
        eng2.calls[0].1,
        vec![c.turn_pos - 1],
        "EOT written at the position before the delta"
    );
    // After EOT, before delta: the stream prefix == the canonical prefix (missing the trailing template newline)
    let canon = canonical(&[
        ("user".into(), Some("hi".into())),
        ("assistant".into(), Some("Hi".into())),
    ]);
    assert_eq!(c.stream_tokens[..canon.len() - 1], canon[..canon.len() - 1]);
}
#[test]
fn n_predict_exhaustion_sets_eot() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, 33, 34]);
    let mut tp = cfg();
    tp.n_predict = 2;
    let out = c
        .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(out.hit_n_predict);
    assert_eq!(out.text, "Hi");
    assert_eq!(out.tokens_generated, 2);
    assert!(c.need_insert_eot);
}
#[test]
fn prev_tokens_reseeded_per_turn() {
    let mut c = conv(512);
    // t1: 'H','i', EOG, placeholder; t2: EOG, placeholder
    let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS, IM_END, EOS]);
    c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    // After t1, prev_tokens = the last 64 stream tokens (including delta + generated + EOG)
    assert_eq!(
        c.prev_tokens.len(),
        c.stream_tokens.len().min(REPEAT_LAST_N)
    );
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    assert_eq!(
        c.prev_tokens.len(),
        c.stream_tokens.len().min(REPEAT_LAST_N)
    );
}
#[test]
fn eot_inserted_before_first_user_turn_after_no_eog() {
    // After start, the first turn ends via a stop string → the second user_turn inserts EOT first
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
    let mut tp = cfg();
    tp.stop_strings = vec!["!".to_string()];
    c.start(Some("hi"), &FakeCodec, &tp, &mut eng, &mut noop_emit())
        .unwrap();
    assert!(c.need_insert_eot);

    let mut eng2 = MockEngine::new(vec![999, IM_END]);
    c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert_eq!(eng2.calls[0].0, vec![IM_END]);
}
/// The very first `user_turn` has no KV yet, so it must prefill the whole
/// render — including anything already in `messages`, i.e. the `--system`
/// prompt. Before C2 the delta was prefilled on its own, which silently
/// dropped the system prompt from the KV.
#[test]
fn first_user_turn_prefills_the_system_prompt() {
    let mut sp = spec(128);
    sp.system_prompt = Some("be brief".to_string());
    let mut c = Conversation::new(sp);
    let mut eng = MockEngine::new(vec![IM_END]);
    let out = c
        .user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();

    let full = fallback_full(&[
        ("system".into(), Some("be brief".into())),
        ("user".into(), Some("hi".into())),
    ]);
    assert_eq!(
        c.stream_tokens,
        [&full[..], &[IM_END]].concat(),
        "the first turn's KV must be the canonical render + EOG"
    );
    assert_eq!(out.prefill_tokens, full.len());
    let sys = canonical(&[("system".into(), Some("be brief".into()))]);
    assert_eq!(
        &c.stream_tokens[..sys.len()],
        &sys[..],
        "the system prompt must reach the KV"
    );
}
