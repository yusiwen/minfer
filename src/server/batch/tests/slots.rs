//! The slot table: its JSON round trip and a resumed snapshot.
//!
//! Split out of `src/server/batch/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// E3 gate helper: the bytes of `Text` a slot has been sent so far, drained
/// without blocking (the events are already queued by the engine).
/// C5 S2: the slot table is JSON in the container's host section — a version this
/// build does not know, or a shape it cannot read, is `None` (the caller refuses the
/// snapshot), never a half-parsed table.
#[test]
fn a_slot_table_round_trips_and_refuses_what_it_cannot_read() {
    let json = r#"{"version":1,"n_slots":2,"n_ctx_total":64,
        "slots":[{"seq":1,"start":0,"cap":32,"cached_tokens":[5,6,7]},
                 {"seq":2,"start":32,"cap":32,"cached_tokens":[]}]}"#;
    let snap = BatchEngine::slots_from_json(json).expect("parses");
    assert_eq!(snap.n_slots, 2);
    assert_eq!(snap.n_ctx_total, 64);
    assert_eq!(snap.slots[0].seq, 1);
    assert_eq!(snap.slots[0].cached_tokens, vec![5, 6, 7]);
    assert!(snap.slots[1].cached_tokens.is_empty());

    // Another version, a missing field, and a non-JSON blob are all `None`.
    let bumped = json.replace("\"version\":1", "\"version\":2");
    assert!(BatchEngine::slots_from_json(&bumped).is_none());
    assert!(BatchEngine::slots_from_json(r#"{"version":1,"n_slots":1}"#).is_none());
    assert!(BatchEngine::slots_from_json("not json").is_none());
}
/// C5 S2's acceptance on a real model: a snapshot written by one engine is resumed by
/// another with the history **not** re-prefilled, and the continuation matches the run
/// that never stopped. A snapshot from another `--n-slots` is refused loudly.
#[test]
#[ignore = "requires the cached 0.5B model and writes a snapshot file"]
fn a_slot_snapshot_resumes_the_context_without_re_prefilling() {
    let Some(path) = cached_model() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the slot-snapshot gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    #[cfg(feature = "cuda")]
    let _guard = crate::cuda::CudaState::model_load_guard();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let n_ctx = 512usize;
    let n_slots = 2usize;
    let file = std::env::temp_dir().join(format!(
        "minfer-c5s2-slots-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::remove_file(&file).ok();

    let prompt = tok.encode("The capital of France is");
    let second = tok.encode(" and the capital of Japan is");

    // Run one prompt to completion and let the engine write the snapshot.
    let (text_a, tokens_a, cold_fed) = {
        let mut a = BatchEngine::new(&*model, n_slots, n_ctx).expect("engine");
        a.set_slots_file(Some(file.clone()));
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
        a.submit(
            &*model,
            &tok,
            Job {
                input_ids: prompt.clone(),
                params: sampling_params(4),
                tx,
            },
        )
        .expect("submit");
        // #160: every stepper below is bounded by work (progress + budget).
        let mut bound = WorkBound::new(&a, step_budget(prompt.len(), 4), "the cold run");
        while a.busy() {
            a.tick(&*model, &tok).expect("tick");
            bound.step(&a);
        }
        let cold_fed = a.prefill_fed();
        assert_eq!(
            cold_fed,
            prompt.len(),
            "the cold run must feed the whole prompt"
        );
        let mut text = String::new();
        let mut tokens = 0usize;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Text(t) => text.push_str(&t),
                StreamEvent::Finish { tokens: n, .. } => tokens = n,
                _ => {}
            }
        }
        (text, tokens, cold_fed)
    };
    assert!(file.exists(), "the snapshot must be written on completion");

    // A fresh engine resumes it: the same prompt prefills **nothing**.
    let mut b = BatchEngine::new(&*model, n_slots, n_ctx).expect("engine");
    let (slots, bytes) = b
        .load_slots(&file, &*model)
        .unwrap_or_else(|e| panic!("load_slots: {e}"));
    assert_eq!(slots, n_slots);
    assert!(bytes > 0);
    let fed_before = b.prefill_fed();
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
    b.submit(
        &*model,
        &tok,
        Job {
            input_ids: prompt.clone(),
            params: sampling_params(4),
            tx,
        },
    )
    .expect("submit");
    let mut bound = WorkBound::new(&b, step_budget(1, 4), "the resumed run");
    while b.busy() {
        b.tick(&*model, &tok).expect("tick");
        bound.step(&b);
    }
    let warm_fed = b.prefill_fed() - fed_before;
    assert_eq!(
        warm_fed,
        1,
        "the snapshot holds the history, so only the query token is fed \
         (the cold run fed {cold_fed} of {} token(s))",
        prompt.len()
    );
    let mut text_b = String::new();
    let mut tokens_b = 0usize;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            StreamEvent::Text(t) => text_b.push_str(&t),
            StreamEvent::Finish { tokens: n, .. } => tokens_b = n,
            _ => {}
        }
    }
    assert_eq!(text_a, text_b, "the resumed continuation must match");
    assert_eq!(tokens_a, tokens_b);

    // ... and its *next* turn prefills only the delta, not the history: the
    // continuation carries the whole conversation, so a re-render would feed
    // `prompt + second` tokens.
    let mut continuation = prompt.clone();
    continuation.extend_from_slice(&second);
    let before_delta = b.prefill_fed();
    let (tx, _rx) = mpsc::channel::<StreamEvent>(1024);
    b.submit(
        &*model,
        &tok,
        Job {
            input_ids: continuation.clone(),
            params: sampling_params(2),
            tx,
        },
    )
    .expect("submit");
    let mut bound = WorkBound::new(&b, step_budget(second.len(), 2), "the delta run");
    while b.busy() {
        b.tick(&*model, &tok).expect("tick");
        bound.step(&b);
    }
    let fed_delta = b.prefill_fed() - before_delta;
    assert!(fed_delta > 0, "the delta still needs a forward");
    assert!(
        fed_delta < continuation.len(),
        "the next turn fed {fed_delta} of {} token(s) — the history came from the snapshot",
        continuation.len()
    );

    // A snapshot from another --n-slots is refused, loudly, naming both.
    let mut one = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
    let err = one
        .load_slots(&file, &*model)
        .expect_err("a 2-slot snapshot must not load into a 1-slot server");
    assert!(err.contains("2-slot"), "{err}");
    assert!(err.contains("--n-slots"), "{err}");
    // ... and one from another --n-ctx, too.
    let mut wide = BatchEngine::new(&*model, n_slots, n_ctx * 2).expect("engine");
    let err = wide
        .load_slots(&file, &*model)
        .expect_err("an n_ctx mismatch must be refused");
    assert!(
        err.contains(&n_ctx.to_string()) && err.contains(&(n_ctx * 2).to_string()),
        "the refusal must name both context lengths: {err}"
    );

    std::fs::remove_file(&file).ok();
}
