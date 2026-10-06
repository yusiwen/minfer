//! Prefill chunking: the chunk size, chunked-vs-unchunked answers and the step budget.
//!
//! Split out of `src/server/batch/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// E3: the chunk plan. Its boundaries are the whole contract — the last span is
/// the remainder, `0` means "off" (one span, the pre-E3 behaviour), and a suffix
/// that already fits is not split.
#[test]
fn prefill_chunks_split_a_suffix_by_the_chunk_size() {
    assert_eq!(prefill_chunks(0, 10, 0), vec![(0, 10)], "0 = chunking off");
    assert_eq!(prefill_chunks(0, 10, 10), vec![(0, 10)]);
    assert_eq!(
        prefill_chunks(0, 10, 16),
        vec![(0, 10)],
        "a suffix that fits stays one span"
    );
    assert_eq!(prefill_chunks(0, 10, 4), vec![(0, 4), (4, 8), (8, 10)]);
    assert_eq!(
        prefill_chunks(0, 8, 4),
        vec![(0, 4), (4, 8)],
        "an exact multiple must not emit an empty tail"
    );
    assert_eq!(
        prefill_chunks(3, 11, 4),
        vec![(3, 7), (7, 11)],
        "the reused prefix is not fed, so the split starts at `from`"
    );
    assert_eq!(prefill_chunks(5, 5, 4), Vec::<(usize, usize)>::new());
    assert_eq!(prefill_chunks(6, 5, 4), Vec::<(usize, usize)>::new());
    assert_eq!(prefill_chunks(0, 1, 8), vec![(0, 1)]);
    // Whatever the numbers, the spans tile `[from, total)` exactly and none is
    // wider than the chunk.
    for (from, total, chunk) in [(0usize, 100usize, 7usize), (0, 100, 1), (13, 91, 9)] {
        let spans = prefill_chunks(from, total, chunk);
        assert_eq!(spans.first().map(|s| s.0), Some(from));
        assert_eq!(spans.last().map(|s| s.1), Some(total));
        for (i, &(a, b)) in spans.iter().enumerate() {
            assert!(
                a < b && b - a <= chunk,
                "span {i} = ({a}, {b}) outside the chunk"
            );
            if i > 0 {
                assert_eq!(
                    spans[i - 1].1,
                    a,
                    "span {i} does not continue the previous one"
                );
            }
        }
    }
}
#[test]
fn the_prefill_chunk_size_comes_from_the_env_or_the_default() {
    assert_eq!(prefill_chunk_size(None), DEFAULT_PREFILL_CHUNK);
    assert_eq!(prefill_chunk_size(Some("")), DEFAULT_PREFILL_CHUNK);
    assert_eq!(prefill_chunk_size(Some(" 512 ")), 512);
    assert_eq!(prefill_chunk_size(Some("0")), 0, "0 is the off switch");
    assert_eq!(
        prefill_chunk_size(Some("banana")),
        DEFAULT_PREFILL_CHUNK,
        "a typo keeps the default rather than disabling chunking"
    );
    assert_eq!(prefill_chunk_size(Some("-1")), DEFAULT_PREFILL_CHUNK);
}
fn drain_text_len(rx: &mut mpsc::Receiver<StreamEvent>) -> usize {
    let mut n = 0;
    while let Ok(ev) = rx.try_recv() {
        if let StreamEvent::Text(t) = ev {
            n += t.len();
        }
    }
    n
}
/// E3 acceptance: a prompt several times the chunk size is served with **every**
/// prefill forward bounded by the chunk, and the continuation is the one the
/// unchunked path produced.
///
/// The comparison class is the repo's standing one: bitwise on CPU (its kernels
/// are per-token, so shape never enters the arithmetic) and a named tolerance on
/// CUDA, whose prefill GEMM tiles by `nt` and quantizes activations to int8. The
/// gate also asserts that the *unchunked* run really did exceed the chunk —
/// otherwise it would be proving nothing.
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn a_chunked_prefill_answers_like_an_unchunked_one() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the E3 equality gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode(&"buffalo ".repeat(96));
    let chunk = (ids.len() / 4).max(8);
    let n_ctx = ids.len() + 64;
    assert!(
        ids.len() > 4 * chunk / 2,
        "the prompt must be several chunks long: {} tokens, chunk {chunk}",
        ids.len()
    );

    // Returns the prefill's own tail-row logits (what the request samples its
    // first token from), the continuation, and the forward stats.
    let drive = |chunk: usize| -> (Vec<f32>, String, usize, usize) {
        let mut engine = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
        engine.set_prefill_chunk(chunk);
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
        engine
            .submit(
                &*model,
                &tok,
                Job {
                    input_ids: ids.clone(),
                    params: sampling_params(8),
                    tx,
                },
            )
            .expect("submit");
        let (max_nt, forwards) = engine.prefill_stats();
        let logits = engine.slots[0]
            .run
            .as_ref()
            .expect("the prefill installed a run")
            .last_logits
            .clone();
        // #160: bounded by work; the arm's name says which shape it drove.
        let what = if chunk == 0 {
            "the unchunked prefill"
        } else {
            "the chunked prefill"
        };
        let mut bound = WorkBound::new(&engine, step_budget(ids.len(), 8), what);
        while engine.busy() {
            engine.tick(&*model, &tok).expect("tick");
            bound.step(&engine);
        }
        let mut text = String::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Text(t) = ev {
                text.push_str(&t);
            }
        }
        (logits, text, max_nt, forwards)
    };

    let (plain_logits, plain, max_plain, fwd_plain) = drive(0);
    let (chunked_logits, chunked, max_chunked, fwd_chunked) = drive(chunk);
    eprintln!(
        "[e3] {}-token prompt: chunked {fwd_chunked} forward(s), max nt {max_chunked}; \
         unchunked {fwd_plain} forward(s), max nt {max_plain}",
        ids.len()
    );
    assert!(
        fwd_chunked >= 4,
        "the chunked run must really split ({fwd_chunked} forwards for a {}-token prompt \
         at chunk {chunk})",
        ids.len()
    );
    assert!(
        max_chunked <= chunk,
        "a prefill forward carried {max_chunked} tokens, over the {chunk} chunk"
    );
    assert!(
        max_plain > chunk,
        "the unchunked run carried only {max_plain} tokens, so this gate proves nothing"
    );
    assert!(!chunked.is_empty(), "the chunked run produced no text");
    // The comparison class is the repo's standing one: **bitwise** on CPU (its
    // kernels are per-token, so shape never enters the arithmetic) and a named
    // tolerance on a device, whose prefill tiles by `nt` — the same tokens at a
    // different width land on different scores. CUDA quantizes activations to
    // int8 (measured <= 0.37 absolute; 1.0 is the gross-error bound
    // `cross_shape_tolerance` uses). Metal reads f32 but its prefill GEMM and
    // flash attention reduce in a different block order per `nt`; the class is
    // the same one `cross_shape_tolerance` names for Metal (0.1, observed
    // <= 0.0153 there). Named before this gate's re-measurement; the measured
    // drift is 0.0087 (0.5B) / 0.0078 (Qwen3-0.6B). The *continuation* is
    // asserted only on CPU: on a degenerate repeated-token prompt a
    // sub-tolerance logit shift can flip an argmax, which is a fact about the
    // prompt, not about chunking.
    assert_eq!(
        plain_logits.len(),
        chunked_logits.len(),
        "logit widths differ"
    );
    let worst = plain_logits
        .iter()
        .zip(&chunked_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let on_cuda = {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::CudaState::get().is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    };
    let on_metal = {
        #[cfg(target_os = "macos")]
        {
            crate::metal::MpsState::get().is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    };
    let tol = if on_cuda {
        1.0
    } else if on_metal {
        0.1
    } else {
        0.0
    };
    let agree = plain
        .bytes()
        .zip(chunked.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "[e3] prefill logits: max |Δ| = {worst} (class {tol}); continuations agree on the \
         first {agree} bytes"
    );
    assert!(
        worst <= tol,
        "chunking moved the prefill's logits by {worst} (class {tol})"
    );
    if !on_cuda && !on_metal {
        assert_eq!(
            plain, chunked,
            "chunking changed the continuation (must be bitwise on CPU)"
        );
    }
}
/// E4 S3 acceptance on the real path: a **repeated** request with the same chunk pattern
/// stops rebuilding. The engine's `GraphCache` now keeps one graph per `GraphParams`, so
/// the second request's prefill chunks (same sizes) hit it — before S3 each chunk was a
/// fresh build, which is the "one forward's fixed overhead per chunk" the E3 record
/// measured. The observable is the cache's own (builds, reuses).
#[test]
#[ignore = "requires the cached 0.5B model"]
fn a_repeated_chunked_prefill_stops_rebuilding() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the E4 S3 rebuild gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode(&"buffalo ".repeat(96));
    let chunk = (ids.len() / 4).max(8);
    let n_ctx = ids.len() + 64;

    let mut engine = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
    engine.set_prefill_chunk(chunk);
    let mut submit_once = |engine: &mut BatchEngine| {
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
        engine
            .submit(
                &*model,
                &tok,
                Job {
                    input_ids: ids.clone(),
                    params: sampling_params(4),
                    tx,
                },
            )
            .expect("submit");
        let mut bound = WorkBound::new(
            engine,
            step_budget(ids.len(), 4),
            "the repeated chunked prefill",
        );
        while engine.busy() {
            engine.tick(&*model, &tok).expect("tick");
            bound.step(engine);
        }
        while rx.try_recv().is_ok() {}
    };

    submit_once(&mut engine);
    let (b1, r1) = engine.cache.stats();
    assert!(b1 > 0, "the first request built its chunk graphs");
    submit_once(&mut engine);
    let (b2, r2) = engine.cache.stats();
    eprintln!("[e4-s3] request 1: {b1} builds / {r1} reuses; request 2: {b2} / {r2}");
    assert_eq!(
        b2, b1,
        "the second request must build nothing: its chunk shapes are cached"
    );
    assert!(r2 > r1, "and it must hit the cache instead");
}
/// E3 acceptance: while a long prompt prefills, the slots already serving keep
/// taking their decode steps.
///
/// The interleaving is observed the way the ticket states it — as tokens emitted
/// by the *other* slot during the prefill call — with the A/B being chunking off,
/// where one forward means no decode step can fit inside the prefill at all.
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn a_long_prefill_keeps_another_slot_decoding() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the E3 interleaving gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    // Both prompts are repeats: neither is expected to EOG, which keeps the
    // scenario about scheduling rather than about the model's mood.
    let short = tok.encode(&"buffalo ".repeat(48));
    let long = tok.encode(&"buffalo ".repeat(192));
    let chunk = (long.len() / 4).max(8);
    let n_ctx = long.len() + 256;

    let drive = |chunk: usize| -> (usize, u64, usize) {
        let mut engine = BatchEngine::new(&*model, 2, n_ctx).expect("engine");
        engine.set_prefill_chunk(chunk);
        // Slot 1 is already serving when the long request arrives.
        let (tx1, mut rx1) = mpsc::channel::<StreamEvent>(4096);
        engine
            .submit_on(
                &*model,
                &tok,
                1,
                Job {
                    input_ids: short.clone(),
                    params: sampling_params(48),
                    tx: tx1,
                },
            )
            .expect("short submit");
        // #160: the three priming steps are a fixed bound, but they go
        // through the same work bound so a wedged step fails here, on the
        // step that wedged it, instead of one assertion later.
        let mut bound = WorkBound::new(&engine, step_budget(short.len(), 48), "the short run");
        for _ in 0..3 {
            engine.tick(&*model, &tok).expect("tick");
            bound.step(&engine);
        }
        // Drain what the three ticks produced: the next drain's *return* is then
        // exactly the bytes slot 1 gains while the long prefill runs (draining is
        // destructive, so nothing may be subtracted here).
        let before = drain_text_len(&mut rx1);
        assert!(
            before > 0,
            "slot 1 must have emitted something before the long prefill"
        );
        let (tx0, _rx0) = mpsc::channel::<StreamEvent>(4096);
        engine
            .submit_on(
                &*model,
                &tok,
                0,
                Job {
                    input_ids: long.clone(),
                    params: sampling_params(4),
                    tx: tx0,
                },
            )
            .expect("long submit");
        let gained = drain_text_len(&mut rx1);
        (gained, engine.interleaved_ticks(), engine.prefill_stats().0)
    };

    let (grew, ticks, max_nt) = drive(chunk);
    let (grew_off, ticks_off, max_nt_off) = drive(0);
    eprintln!(
        "[e3] long prefill: chunked -> {ticks} interleaved step(s), slot 1 gained {grew} \
         bytes, max prefill nt {max_nt}; chunking off -> {ticks_off} step(s), gained \
         {grew_off}, max nt {max_nt_off}"
    );
    assert!(max_nt <= chunk, "chunked prefill carried {max_nt} tokens");
    assert!(
        ticks > 0 && grew > 0,
        "a chunked prefill must keep the other slot decoding (ticks {ticks}, bytes {grew})"
    );
    assert_eq!(
        ticks_off, 0,
        "chunking off is one forward, so nothing can interleave"
    );
    assert_eq!(
        grew_off, 0,
        "with chunking off the other slot cannot advance during the prefill"
    );
}
