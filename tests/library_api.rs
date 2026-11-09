//! #763 S6a: the engine facade as an **external** consumer sees it.
//!
//! These tests use `minfer::engine`, not `crate::`, so they exercise exactly the
//! surface a host gets: load a model, run a chat turn, drop everything, load
//! again. The toy fixture keeps them device-free and fast.

use minfer::conversation::TurnParams;
use minfer::engine::{Engine, SessionConfig};
use minfer::sampler::SamplerConfig;

fn toy() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/toy-qwen2-f32.gguf")
}

fn turn(n_predict: usize) -> TurnParams {
    TurnParams {
        n_predict,
        sampler: SamplerConfig::default(),
        stop_strings: Vec::new(),
    }
}

#[test]
fn a_host_loads_chats_drops_and_loads_again() {
    let path = toy();
    let first = {
        let engine = Engine::load(&path).expect("the facade loads the toy GGUF");
        assert!(engine.chatml_notice(), "the toy has no template");
        let mut session = engine.session(SessionConfig {
            n_ctx: 64,
            ..Default::default()
        });
        let mut out = Vec::new();
        let outcome = session
            .chat("hello", &turn(4), &mut |delta| out.extend_from_slice(delta))
            .expect("one chat turn");
        assert!(
            outcome.tokens_generated <= 4,
            "n_predict caps the turn: {}",
            outcome.tokens_generated
        );
        out
    };
    // The session and the engine are gone; a second load must work (the mapping
    // is released — #745's gate asserts the release itself).
    let engine = Engine::load(&path).expect("a second load works");
    assert_eq!(
        engine.model().n_layer(),
        1,
        "a fresh engine, not stale state"
    );
    let _ = first;
}

#[test]
fn two_sessions_on_one_engine_do_not_share_history() {
    let engine = Engine::load(&toy()).expect("load");
    let mut a = engine.session(SessionConfig {
        n_ctx: 64,
        ..Default::default()
    });
    let b = engine.session(SessionConfig {
        n_ctx: 64,
        ..Default::default()
    });
    let mut sink = |_d: &[u8]| {};
    a.chat("first", &turn(1), &mut sink).expect("a");
    assert_eq!(a.conversation().messages.len(), 2, "user + assistant");
    assert!(
        b.conversation().messages.is_empty(),
        "the second session starts empty"
    );
}

#[test]
fn the_external_generate_runs_the_whole_loop() {
    let engine = Engine::load(&toy()).expect("load");
    let ids = engine.tokenizer().encode("hi");
    let sampler = SamplerConfig::default();
    let params = minfer::engine::GenerateParams {
        tokens: &ids,
        n_ctx: 64,
        n_predict: 2,
        sampler: &sampler,
        seed: 42,
        stop_strings: &[],
        timing: false,
        spec_adaptive: false,
        spec_d_max: 2,
        cancel: None,
    };
    let mut out = Vec::new();
    let mut prefill = None;
    let outcome = engine.generate(
        &params,
        &mut |delta| out.extend_from_slice(delta),
        &mut |_logits, secs| prefill = Some(secs),
    );
    assert_eq!(outcome.prefill_tokens, ids.len());
    assert!(outcome.generated <= 2, "n_predict caps the run");
    assert!(prefill.is_some(), "the prefill hook fires once");
}

/// #796: a host cancels from inside its own delta callback — the shape a UI or a
/// deadline uses. The loop stops at the next token boundary and says so; it does
/// not run to `n_predict`, and it does not abandon a forward pass mid-flight.
#[test]
fn a_cancelled_generate_stops_at_the_next_token() {
    let engine = Engine::load(&toy()).expect("load");
    let ids = engine.tokenizer().encode("hi");
    let sampler = SamplerConfig::default();
    let cancel = minfer::engine::CancelToken::new();
    let params = minfer::engine::GenerateParams {
        tokens: &ids,
        n_ctx: 64,
        n_predict: 32,
        sampler: &sampler,
        seed: 42,
        stop_strings: &[],
        timing: false,
        spec_adaptive: false,
        spec_d_max: 2,
        cancel: Some(&cancel),
    };
    let mut deltas = 0usize;
    let mut out = Vec::new();
    let outcome = engine.generate(
        &params,
        &mut |delta| {
            deltas += 1;
            out.extend_from_slice(delta);
            if deltas == 1 {
                cancel.cancel();
            }
        },
        &mut |_logits, _secs| {},
    );
    assert!(outcome.cancelled, "the outcome names the cancellation");
    assert!(
        outcome.generated >= 1 && outcome.generated < 32,
        "the run stopped between tokens, not at n_predict: {}",
        outcome.generated
    );
    assert!(!outcome.hit_n_predict, "a cancel is not an n_predict hit");
    assert!(!out.is_empty(), "the partial answer was emitted");
}

/// #796: a pre-cancelled token decodes nothing, but the prefill still happens
/// and is reported — a host that cancels between a load and a run gets an
/// honest "0 of N tokens", not a silent no-op.
#[test]
fn a_pre_cancelled_generate_decodes_nothing() {
    let engine = Engine::load(&toy()).expect("load");
    let ids = engine.tokenizer().encode("hi");
    let sampler = SamplerConfig::default();
    let cancel = minfer::engine::CancelToken::new();
    cancel.cancel();
    let params = minfer::engine::GenerateParams {
        tokens: &ids,
        n_ctx: 64,
        n_predict: 8,
        sampler: &sampler,
        seed: 42,
        stop_strings: &[],
        timing: false,
        spec_adaptive: false,
        spec_d_max: 2,
        cancel: Some(&cancel),
    };
    let mut out = Vec::new();
    let outcome = engine.generate(
        &params,
        &mut |delta| out.extend_from_slice(delta),
        &mut |_logits, _secs| {},
    );
    assert!(outcome.cancelled, "the outcome names the cancellation");
    assert_eq!(outcome.generated, 0, "nothing is decoded");
    assert_eq!(outcome.prefill_tokens, ids.len(), "the prefill is reported");
    assert!(!outcome.hit_n_predict, "a cancel is not an n_predict hit");
    assert!(out.is_empty(), "nothing is emitted");
}

/// #795: a raw continuation prefills only its own tokens, and the session's KV
/// cursor says so.
#[test]
fn complete_prefills_only_the_new_tokens() {
    let engine = Engine::load(&toy()).expect("load");
    let mut session = engine.session(SessionConfig {
        n_ctx: 64,
        ..Default::default()
    });
    let mut sink = |_d: &[u8]| {};
    session.chat("hello", &turn(2), &mut sink).expect("chat");
    let before = session.n_past();
    assert!(before > 0, "the chat turn wrote rows");

    let ids = engine.tokenizer().encode(" world");
    assert!(!ids.is_empty(), "the fixture tokenizes the continuation");
    let out = session
        .complete(&ids, &turn(2), &mut sink)
        .expect("one continuation");
    assert_eq!(
        out.prefill_tokens,
        ids.len(),
        "only the continuation is prefilled; the chat's rows are reused"
    );
    assert!(
        session.n_past() >= before + ids.len(),
        "the cursor advanced by the continuation and its decode ({} -> {})",
        before,
        session.n_past()
    );
}

/// #795: splitting the same token stream across two calls is equivalent to one
/// call — the second prefill starts where the first stopped, and the KV rows make
/// the logits identical (both sessions start fresh, so the sampler seeds match).
#[test]
fn a_split_continuation_matches_a_single_one() {
    let engine = Engine::load(&toy()).expect("load");
    let ids = engine.tokenizer().encode("The capital of France is");
    assert!(ids.len() > 2, "split the prompt into two non-empty halves");
    let (head, tail) = ids.split_at(ids.len() / 2);

    let run = |chunks: &[&[u32]]| -> (String, usize) {
        let mut session = engine.session(SessionConfig {
            n_ctx: 64,
            ..Default::default()
        });
        let mut out = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            // The intermediate call only prefills; the last one decodes.
            let n = if i + 1 == chunks.len() { 4 } else { 0 };
            session
                .complete(chunk, &turn(n), &mut |d| out.extend_from_slice(d))
                .expect("continuation");
        }
        (String::from_utf8_lossy(&out).into_owned(), session.n_past())
    };

    let split = (head, tail);
    let (a_text, a_pos) = run(&[split.0, split.1]);
    let (b_text, b_pos) = run(&[&ids]);
    assert_eq!(a_text, b_text, "the split run decodes the same text");
    assert_eq!(
        a_pos, b_pos,
        "and writes the same number of KV rows ({} vs {})",
        a_pos, b_pos
    );
}

/// #795: a continuation is not history. However many raw tokens it wrote, the
/// next chat turn rebuilds the stream from the messages, so the session's cursor
/// afterwards depends on the history alone — run the same turn after
/// continuations of different lengths and the cursor must not move.
#[test]
fn a_continuation_does_not_leak_into_a_later_chat() {
    let engine = Engine::load(&toy()).expect("load");
    let mut sink = |_d: &[u8]| {};
    // Greedy: the assertion is about the KV the turn reads, not about which RNG
    // draws the continuation consumed.
    let greedy = |n: usize| TurnParams {
        n_predict: n,
        sampler: SamplerConfig {
            temp: 0.0,
            ..Default::default()
        },
        stop_strings: Vec::new(),
    };

    let run = |continuation: Option<usize>| -> (usize, usize, usize) {
        let mut session = engine.session(SessionConfig {
            n_ctx: 64,
            ..Default::default()
        });
        let mut sink = |_d: &[u8]| {};
        session.chat("hello", &greedy(2), &mut sink).expect("chat");
        if let Some(n) = continuation {
            let ids = engine.tokenizer().encode(" world, again and again");
            assert!(ids.len() >= n, "the fixture encodes {n} tokens");
            session
                .complete(&ids[..n], &greedy(2), &mut sink)
                .expect("continuation");
        }
        let before = session.n_past();
        session
            .chat("again", &greedy(2), &mut sink)
            .expect("chat after");
        (
            before,
            session.n_past(),
            session.conversation().messages.len(),
        )
    };

    let (control_before, control_after, control_msgs) = run(None);
    assert_eq!(control_msgs, 4, "chat, reply, chat, reply");

    // The control takes the delta path; a continued session is forced through the
    // full re-render, and the two tokenize the message boundary slightly
    // differently — so the comparable property is that the rebuilt cursor does not
    // depend on how many raw tokens the continuation wrote.
    let mut rebuilt = None;
    for n in [1usize, 2, 5] {
        let (before, after, msgs) = run(Some(n));
        assert!(
            before > control_before,
            "the continuation did write rows ({n} tokens, cursor {before} vs {control_before})"
        );
        assert_eq!(msgs, 4, "the continuation recorded no message ({n} tokens)");
        match rebuilt {
            None => rebuilt = Some(after),
            Some(first) => assert_eq!(
                after, first,
                "the rebuilt cursor must not depend on the continuation's length ({n} tokens)"
            ),
        }
    }
    assert!(rebuilt.is_some(), "at least one continuation ran");
    assert!(
        control_after > control_before,
        "the control's turn advanced its own cursor"
    );
}

/// #797: a host that already holds the model answers exactly like a path load —
/// same bytes, no temp file, byte-identical turn.
#[test]
fn a_model_loaded_from_bytes_answers_like_the_path_load() {
    let path = toy();
    let bytes = std::fs::read(&path).expect("read the fixture");
    let from_bytes = Engine::load_from_bytes(bytes).expect("load from bytes");
    let from_path = Engine::load(&path).expect("load from path");

    let greedy = TurnParams {
        n_predict: 4,
        sampler: SamplerConfig {
            temp: 0.0,
            ..Default::default()
        },
        stop_strings: Vec::new(),
    };
    let answer = |engine: &Engine| {
        let mut session = engine.session(SessionConfig {
            n_ctx: 64,
            ..Default::default()
        });
        let mut out = Vec::new();
        session
            .chat("hello", &greedy, &mut |d| out.extend_from_slice(d))
            .expect("one turn");
        out
    };
    let a = answer(&from_bytes);
    let b = answer(&from_path);
    assert!(!a.is_empty(), "the turn produced text");
    assert_eq!(a, b, "a bytes load answers byte-identically to a path load");
}
