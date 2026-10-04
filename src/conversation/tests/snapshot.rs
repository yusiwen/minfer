//! The session snapshot: resume, refusal and the JSON round trip.
//!
//! Split out of `src/conversation/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// C5 S2's host-state half: a conversation restored from a session snapshot starts
/// with **nothing prefilled** and its next turn issues exactly the engine calls the
/// in-memory run's does (delta prefill + decodes — never a re-render of the history).
/// The KV bytes that make that legitimate are the C5 container's own gate.
#[test]
fn a_resumed_snapshot_prefills_nothing_and_continues_alike() {
    // The in-memory run: one turn, then the snapshot `--session` would write on exit.
    // The mock pops one program token per `forward` — including prefill calls — so the
    // program below is the one that makes turn 2 sample '!' on both sides.
    let mut base = conv(512);
    let mut eng = MockEngine::new(vec![72, 105, EOS, 33, 33, EOS]); // 'H','i',EOG,(EOG write),'!',EOG
    base.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let snap = base.snapshot();
    let json = base.snapshot_to_json();
    let parsed = Conversation::snapshot_from_json(&json).expect("the snapshot round-trips");
    assert_eq!(parsed, snap, "JSON must carry the snapshot losslessly");

    // The resumed run: a fresh conversation and a fresh (empty) engine, as a second
    // `--session` process would have.
    let mut resumed = conv(512);
    let mut eng2 = MockEngine::new(vec![33, EOS]);
    resumed.restore_snapshot(&parsed).expect("restore");
    assert!(
        eng2.calls.is_empty() && eng2.resets == 0,
        "restoring a snapshot must not touch the engine at all"
    );
    assert_eq!(resumed.messages, base.messages);
    assert_eq!(resumed.stream_tokens, base.stream_tokens);
    assert_eq!(resumed.current_pos, base.current_pos);
    assert_eq!(resumed.prev_tokens, base.prev_tokens);
    assert_eq!(resumed.need_insert_eot, base.need_insert_eot);

    // The next turn on both: same input, same engine calls, same output.
    let before = eng.calls.len();
    let out_base = base
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let base_calls: Vec<(Vec<u32>, Vec<usize>)> = eng.calls[before..]
        .iter()
        .map(|(t, p, _)| (t.clone(), p.clone()))
        .collect();
    let out_resumed = resumed
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    let resumed_calls: Vec<(Vec<u32>, Vec<usize>)> = eng2
        .calls
        .iter()
        .map(|(t, p, _)| (t.clone(), p.clone()))
        .collect();
    assert_eq!(
        base_calls, resumed_calls,
        "a resumed turn must issue the same forwards, at the same positions"
    );
    assert_eq!(out_base.text, out_resumed.text, "greedy tokens must agree");
    assert_eq!(out_resumed.text, "!");
    // And the first call is the turn's own delta, not the history re-rendered.
    let full_render = fallback_full(&base.messages);
    assert!(
        resumed_calls[0].0.len() < full_render.len(),
        "the resumed turn prefilled {} token(s) — a re-render would be {}",
        resumed_calls[0].0.len(),
        full_render.len()
    );
    assert_eq!(resumed.messages, base.messages);
    assert_eq!(resumed.stream_tokens, base.stream_tokens);
    assert_eq!(resumed.current_pos, base.current_pos);
}
/// C5 S2: a snapshot that contradicts the host mirror is refused *before* it is
/// applied — the caller then re-seeds, which is always safe.
#[test]
fn a_snapshot_that_contradicts_the_host_mirror_is_refused() {
    let mut c = conv(512);
    let good = ConversationSnapshot {
        messages: vec![("user".into(), Some("hi".into()))],
        stream_tokens: vec![1, 2, 3],
        current_pos: 3,
        turn_pos: 0,
        prev_tokens: vec![2, 3],
        need_insert_eot: false,
    };
    c.restore_snapshot(&good)
        .expect("a consistent snapshot applies");

    // current_pos must equal the token stream's length.
    let mut bad = good.clone();
    bad.current_pos = 4;
    let err = c.restore_snapshot(&bad).unwrap_err();
    assert!(err.contains("host mirror"), "{err}");
    // ... and must fit this run's n_ctx.
    let mut bad = good.clone();
    bad.stream_tokens = vec![1; 600];
    bad.current_pos = 600;
    let err = c.restore_snapshot(&bad).unwrap_err();
    assert!(err.contains("n_ctx"), "{err}");
    // ... and turn_pos may not run past it.
    let mut bad = good.clone();
    bad.turn_pos = 9;
    let err = c.restore_snapshot(&bad).unwrap_err();
    assert!(err.contains("turn_pos"), "{err}");

    // A JSON this build does not understand is `None`, never a guess.
    assert!(Conversation::snapshot_from_json("{}").is_none());
    assert!(Conversation::snapshot_from_json("not json").is_none());
    let bumped = json_with_version(&c.snapshot_to_json(), SNAPSHOT_VERSION + 1);
    assert!(Conversation::snapshot_from_json(&bumped).is_none());
    // The refused snapshots above must not have touched the conversation.
    assert_eq!(c.current_pos, 3);
    assert_eq!(c.stream_tokens, vec![1, 2, 3]);
}
/// C5 S2: an engine with no arena (the mocks, and any backend that cannot hand its KV
/// to the host) refuses both calls, so the CLI falls back to re-seeding instead of
/// pretending the session was resumed.
#[test]
fn an_engine_without_a_kv_refuses_the_session_calls() {
    let mut eng = MockEngine::new(vec![]);
    let path = std::path::Path::new("/tmp/minfer-c5s2-not-a-kv-file");
    assert!(eng.kv_save(path, b"{}").is_err());
    assert!(eng.kv_load(path).is_err());
}
#[test]
fn session_json_round_trip() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let json = c.messages_to_json();
    let parsed = Conversation::messages_from_json(&json).expect("parse");
    assert_eq!(parsed, c.messages);
    // null content is preserved (OpenAI-style object format)
    let with_null = vec![
        ("user".into(), Some("hi".into())),
        ("assistant".into(), None),
    ];
    let with_null_json = serde_json::json!([
        { "role": "user", "content": "hi" },
        { "role": "assistant", "content": null },
    ])
    .to_string();
    let j2 = Conversation::messages_from_json(&with_null_json).unwrap();
    assert_eq!(j2, with_null);
    // Invalid input → None
    assert!(Conversation::messages_from_json("not json").is_none());
}
#[test]
fn load_history_rehydrates_full_render() {
    let mut c = conv(512);
    let mut eng = MockEngine::new(vec![IM_END, EOS]);
    c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
        .unwrap();
    let saved = c.messages_to_json();

    // A new session loads the history → full re-render (render(messages, false))
    let mut c2 = conv(512);
    let mut eng2 = MockEngine::new(vec![IM_END, EOS]);
    let msgs = Conversation::messages_from_json(&saved).unwrap();
    c2.load_history(msgs, &FakeCodec, &mut eng2)
        .expect("load history");
    assert_eq!(c2.messages, c.messages);
    assert_eq!(c2.current_pos, c2.stream_tokens.len());
    let canon = canonical(&c.messages);
    assert_eq!(
        c2.stream_tokens, canon,
        "KV = canonical render of loaded history"
    );
    assert_eq!(c2.turn_pos, 0);

    // Continue the conversation after loading (the incremental delta continues from the re-rendered KV)
    let out = c2
        .user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
        .unwrap();
    assert!(out.stopped_by_eog);
    assert_eq!(c2.messages.len(), 4);
    assert!(c2.current_pos > c2.stream_tokens.len().saturating_sub(1));
    assert_eq!(c2.current_pos, c2.stream_tokens.len());
}
