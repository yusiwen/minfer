//! `#[cfg(test)] mod tests` for `src/server/batch.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::models::ModelDef;
use std::time::Instant;

fn cached_model() -> Option<std::path::PathBuf> {
    // `MINFER_BATCH_TEST_MODEL` points the measurement at another cached
    // model (the 7B is where batching should pay: decode is weight-bandwidth
    // bound there, while the 0.5B's decode is kernel-compute bound).
    if let Ok(custom) = std::env::var("MINFER_BATCH_TEST_MODEL") {
        let p = std::path::PathBuf::from(custom);
        return p.exists().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    p.exists().then_some(p)
}

/// Drive `n` requests through the engine, returning each one's generated
/// tokens and the wall time. Requests are submitted together (the engine
/// admits what fits) and stepped until they all finish, which is what
/// continuous batching does.
#[derive(Debug, PartialEq)]
struct Reply {
    text: String,
    tokens: usize,
    reason: String,
}

fn sampling_params(max_tokens: i64) -> SamplingParams {
    SamplingParams {
        // Greedy, identical on both sides of the comparison.
        temp: 0.0,
        top_k: 1,
        top_p: 1.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        typical_p: 1.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.5,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 64,
        dry_breakers: Vec::new(),
        mirostat: crate::sampler::MirostatMode::Off,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        mirostat_m: 100,
        logit_bias: Vec::new(),
        grammar_source: None,
        grammar: None,
        seed: 7,
        stop_strings: Vec::new(),
        max_tokens,
    }
}

fn run_batched(
    model: &dyn ModelDef,
    tok: &Tokenizer,
    prompts: &[Vec<u32>],
    n_slots: usize,
    n_ctx: usize,
    max_tokens: i64,
    stagger: bool,
) -> (Vec<Reply>, f64) {
    let mut engine = BatchEngine::new(model, n_slots, n_ctx).expect("engine");
    let mut out: Vec<Option<Reply>> = (0..prompts.len()).map(|_| None).collect();
    let mut text: Vec<String> = vec![String::new(); prompts.len()];
    let mut pending: Vec<(usize, mpsc::Receiver<StreamEvent>)> = Vec::new();
    let mut queue: Vec<(usize, Job)> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
        queue.push((
            i,
            Job {
                input_ids: p.clone(),
                params: sampling_params(max_tokens),
                tx,
            },
        ));
        pending.push((i, rx));
    }
    let t0 = Instant::now();
    // #160: the whole drive is bounded by work, not by the clock. `sum of the
    // prompts` plus every request's own token cap is the most the engine can
    // legitimately consume; the budget's 4x margin absorbs the admission and
    // prefill steps, which do not show up in `work_units`.
    let total_prompt: usize = prompts.iter().map(|p| p.len()).sum();
    let budget = step_budget(total_prompt, prompts.len() * step_cap(max_tokens, n_ctx));
    let mut bound = WorkBound::new(&engine, budget, "the batched run");
    // Admit, then step until every request has finished.
    while !queue.is_empty() || engine.busy() {
        // `stagger` reproduces the **server's** admission pattern: requests
        // arrive while others decode, so `serve_loop` admits one per step and
        // the step sequence mixes widths (a 1-sequence step, then wider
        // ones). Without it, everything is admitted before the first tick and
        // every step has the same width — which is what this helper always
        // did, and why the blocker below hid from it.
        let mut admit = if stagger { 1 } else { engine.idle_slots() };
        while !queue.is_empty() && engine.idle_slots() > 0 && admit > 0 {
            let (i, job) = queue.remove(0);
            engine
                .submit(model, tok, job)
                .unwrap_or_else(|e| panic!("submit request {i}: {}", e.message));
            admit -= 1;
        }
        engine.tick(model, tok).expect("tick");
        bound.step(&engine);
        // Drain events; a finished request is reported by `Finish`, and its
        // text is the concatenation of the `Text` events.
        for (i, rx) in pending.iter_mut() {
            loop {
                match rx.try_recv() {
                    Ok(StreamEvent::Text(t)) => text[*i].push_str(&t),
                    Ok(StreamEvent::Finish { reason, tokens }) => {
                        out[*i] = Some(Reply {
                            text: std::mem::take(&mut text[*i]),
                            tokens,
                            reason,
                        });
                    }
                    Ok(StreamEvent::Err(e)) => panic!("request {i}: {}", e.message),
                    Err(_) => break,
                }
            }
        }
    }
    (
        out.into_iter()
            .map(|r| r.expect("every request finished"))
            .collect(),
        t0.elapsed().as_secs_f64(),
    )
}

/// The serial baseline: the same four requests, each served alone, **on the
/// slot it would occupy in the batched run** — a request's window offset is
/// part of its determinism (see [`BatchEngine::submit_on`]), so comparing
/// across offsets would measure arithmetic, not batching.
fn run_serial(
    model: &dyn ModelDef,
    tok: &Tokenizer,
    prompts: &[Vec<u32>],
    n_slots: usize,
    n_ctx: usize,
    max_tokens: i64,
) -> (Vec<Reply>, f64) {
    // One engine, so the baseline pays the same one-time graph builds as the
    // batched run; each request is placed on the slot it would occupy.
    let t0 = Instant::now();
    let mut out = Vec::new();
    let mut engine = BatchEngine::new(model, n_slots, n_ctx).expect("engine");
    for (i, p) in prompts.iter().enumerate() {
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
        engine
            .submit_on(
                model,
                tok,
                i,
                Job {
                    input_ids: p.clone(),
                    params: sampling_params(max_tokens),
                    tx,
                },
            )
            .unwrap_or_else(|e| panic!("submit request {i}: {}", e.message));
        // #160: this request's own drive is bounded by work (progress and the
        // step budget), exactly like the batched arm.
        let mut bound = WorkBound::new(
            &engine,
            step_budget(p.len(), step_cap(max_tokens, n_ctx)),
            "the serial baseline",
        );
        let mut text = String::new();
        let mut done = None;
        while done.is_none() {
            engine.tick(model, tok).expect("tick");
            bound.step(&engine);
            loop {
                match rx.try_recv() {
                    Ok(StreamEvent::Text(t)) => text.push_str(&t),
                    Ok(StreamEvent::Finish { reason, tokens }) => {
                        done = Some(Reply {
                            text: std::mem::take(&mut text),
                            tokens,
                            reason,
                        });
                    }
                    Ok(StreamEvent::Err(e)) => panic!("request {i}: {}", e.message),
                    Err(_) => break,
                }
            }
        }
        out.push(done.expect("finished"));
    }
    (out, t0.elapsed().as_secs_f64())
}

/// Whether a CUDA device participates in this process (`load_model` above
/// initialises the state when the model is loaded).
fn cuda_device_active() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::CudaState::get().is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// The E2 correctness half, shared by the batched and staggered arms: every
/// row of `got` must match its serial reference.
///
/// Byte-equality is a **CPU** property: both sides drive the same engine with
/// the same slot reservations (`submit_on` pins each request to the slot it
/// would occupy), so on CPU the arithmetic is identical. On a device it cannot
/// be — the batched step is `nt = 4` and the serial step `nt = 1`, and CUDA's
/// kernels tile by `nt` (measured drift 0.22–0.37 on logits; the plan records
/// it as a named tolerance class), so a greedy continuation may legitimately
/// diverge after a few tokens. Device runs therefore assert the *structural*
/// property that a wrong window would break immediately — the two
/// continuations must start identically — and report how far they track; the
/// window assignment itself is pinned bitwise on device by
/// `batch_order_does_not_change_a_sequences_logits` (same shape, same layout)
/// and `cuda_two_sequences_do_not_cross_attend`.
fn assert_replies_match(what: &str, got: &[Reply], serial: &[Reply]) {
    assert_eq!(got.len(), serial.len(), "{what}: row count differs");
    for (i, (b, s)) in got.iter().zip(serial).enumerate() {
        assert_eq!(
            b.reason, s.reason,
            "{what} request {i}: finish reason differs"
        );
        assert!(!b.text.is_empty(), "{what} request {i}: generated nothing");
        assert!(
            !s.text.is_empty(),
            "serial request {i}: generated nothing (the reference is empty)"
        );
        if cuda_device_active() {
            let common = b
                .text
                .bytes()
                .zip(s.text.bytes())
                .take_while(|(x, y)| x == y)
                .count();
            assert!(
                common > 0,
                "{what} request {i}: {b:?} and serial {s:?} diverge at the first byte on a \
                 device, which numerics cannot explain"
            );
            eprintln!(
                "[e2] {what} request {i}: {common} leading byte(s) shared on device; {:?} ({}) \
                 vs serial {:?} ({})",
                b.text, b.tokens, s.text, s.tokens
            );
        } else {
            assert_eq!(
                b.text, s.text,
                "{what} request {i}: {:?} ({}) vs serial {:?} ({})",
                b.text, b.tokens, s.text, s.tokens
            );
            assert_eq!(
                b.tokens, s.tokens,
                "{what} request {i}: token count differs"
            );
        }
    }
}

/// The location estimate #154's timing verdict uses (the upper median for an
/// even count, matching `cuda_backend::tests::median`).
fn median(v: &[f64]) -> f64 {
    assert!(!v.is_empty(), "median of an empty sample");
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// C8a: admit `prompt` on `slot` and drive the engine until it finishes, returning
/// the text the client would have streamed. (The E2 acceptance paragraph that used
/// to sit above this helper documents `server_batch_matches_serial_and_is_faster`
/// and now lives on it.)
fn serve_on(
    engine: &mut BatchEngine,
    model: &dyn ModelDef,
    tok: &Tokenizer,
    slot: usize,
    prompt: Vec<u32>,
    max_tokens: i64,
) -> String {
    let budget = step_budget(prompt.len(), step_cap(max_tokens, engine.n_ctx_total));
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
    engine
        .submit_on(
            model,
            tok,
            slot,
            Job {
                input_ids: prompt,
                params: sampling_params(max_tokens),
                tx,
            },
        )
        .expect("admit");
    let mut text = String::new();
    // #160: a bounded drive (progress + step budget), like the other steppers.
    let mut bound = WorkBound::new(engine, budget, "the slot-scoped request");
    while engine.busy() {
        engine.tick(model, tok).expect("tick");
        bound.step(engine);
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Text(t) = ev {
                text.push_str(&t);
            }
        }
    }
    while let Ok(ev) = rx.try_recv() {
        if let StreamEvent::Text(t) = ev {
            text.push_str(&t);
        }
    }
    text
}

/// C8a gate: a prompt served from *another* slot's rows must answer exactly as the
/// same prompt served on a private run, and the rows must actually be copied rather
/// than prefilled. The copy is what the counter proves; the equality is the whole
/// point — C6 makes the donor's different run start arithmetic-free, so a copied
/// prefix cannot change this slot's answer.
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn a_prefix_copied_from_another_slot_answers_identically() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the C8a gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let prompt = tok.encode("The capital of France is");
    let n_ctx = prompt.len() + 64;

    // Slot 0 computes the prompt; slot 1 must *copy* those rows.
    let mut engine = BatchEngine::new(&*model, 2, n_ctx).expect("engine");
    let first = serve_on(&mut engine, &*model, &tok, 0, prompt.clone(), 8);
    let before = engine.prefix_rows_copied();
    let second = serve_on(&mut engine, &*model, &tok, 1, prompt.clone(), 8);
    assert!(
        engine.prefix_rows_copied() > before,
        "slot 1 prefilled the prompt instead of copying slot 0's {} rows",
        prompt.len()
    );
    assert_eq!(first, second, "a copied prefix must not change the answer");

    // And the same prompt on a single-slot engine, which has nothing to copy.
    let mut alone = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
    let solo = serve_on(&mut alone, &*model, &tok, 0, prompt, 8);
    assert_eq!(
        second, solo,
        "the copied run must answer like a private one"
    );
}

/// C8b S3 gate: a request that diverges **inside** a prefix another slot
/// computed must take private rows there (copy-on-write) instead of storing
/// through the donor's cells.
///
/// The failure this ticket forbids is invisible from the sharer alone: a store
/// that wrote through the shared cells would still give the sharer the right
/// answer (it reads those same cells), and only the **donor** would be corrupted.
/// So the gate checks four things: the request is served at all (the store
/// resolver refuses a shared position, so a missing copy-on-write is a loud
/// error), the copy-on-write counter moved, the answer is the private run's, and
/// the donor's rows come out byte-identical.
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn a_store_inside_a_shared_prefix_takes_a_private_row() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the C8b S3 gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    // Long enough that the share has rows worth copying.
    let prompt = tok.encode(
        "You are a helpful assistant. Answer in one short sentence. The capital of France is",
    );
    assert!(prompt.len() > 8, "the prompt has to be worth sharing");
    let n_ctx = prompt.len() + 64;

    // A request on slot 1 matching only the prompt's first three tokens: its
    // prefill starts inside the shared prefix, so the store has to copy.
    let mut diverging = prompt[..3].to_vec();
    diverging.extend(tok.encode(" and the capital of Italy is"));
    assert_eq!(
        common_prefix_len(&prompt, &diverging),
        3,
        "the divergence point is the gate's input"
    );

    // One whole scenario — donor, sharer, then the diverging request — with the
    // share either on (the ticket's path) or replaced by C8a's **copy** (the
    // A/B). The copy variant is the baseline the answer is compared against
    // because it is *shape-matched*: the same requests, the same fed positions
    // and the same K/V bytes, with the donor's rows duplicated instead of
    // referenced. A one-slot baseline cannot be: on CUDA a prefill's GEMM shape
    // changes the K/V it computes, so a run that feeds a different number of
    // tokens answers differently for reasons that have nothing to do with
    // sharing — the first form of this gate passed on CPU and failed on CUDA for
    // exactly that reason.
    let mut scenario = |share: bool| -> (String, usize, usize, u64, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if !share {
            std::env::set_var("MINFER_NO_KV_SHARE", "1");
        }
        let mut engine = BatchEngine::new(&*model, 2, n_ctx).expect("engine");
        let (donor_seq, dst_seq) = (engine.slots[0].seq, engine.slots[1].seq);
        serve_on(&mut engine, &*model, &tok, 0, prompt.clone(), 8);
        serve_on(&mut engine, &*model, &tok, 1, prompt.clone(), 8);
        let shared_rows = engine
            .cache
            .alloc()
            .kv_seq_slot(dst_seq)
            .expect("slot 1 has a run")
            .shared
            .rows;
        let shared_cells = engine.cache.alloc().kv_arena_stats().shared_cells;
        let cows_before = engine.cow_stats().0;
        let donor_before = kv_rows_of(&mut engine, donor_seq);
        let answer = serve_on(&mut engine, &*model, &tok, 1, diverging.clone(), 8);
        let cows = engine.cow_stats().0 - cows_before;
        // The donor is still a live run — a reclaim would make the byte check
        // vacuous rather than wrong, which is the kind of silent pass this
        // assertion exists to prevent.
        assert!(engine.cache.alloc().kv_seq_slot(donor_seq).is_some());
        assert_eq!(
            kv_rows_of(&mut engine, donor_seq),
            donor_before,
            "the diverging request wrote through the shared prefix"
        );
        let dst_rows = kv_rows_of(&mut engine, dst_seq);
        if !share {
            std::env::remove_var("MINFER_NO_KV_SHARE");
        }
        (
            answer,
            shared_rows,
            shared_cells,
            cows,
            donor_before,
            dst_rows,
        )
    };

    let (shared_answer, shared_rows, shared_cells, cows, donor_rows, dst_a) = scenario(true);
    // Two identical share runs must agree byte for byte — the property that
    // caught the decode-position bug this gate first ran into: a skipped
    // position left an unwritten row that attention read, so the answer
    // depended on the arena's history.
    let (shared_answer2, _, _, _, _, dst_b) = scenario(true);
    assert_eq!(shared_answer, shared_answer2, "two share runs must agree");
    assert_eq!(dst_a, dst_b, "two share runs must write the same rows");
    assert!(
        shared_rows > 3,
        "slot 1 must read {shared_rows} rows in place for the gate to mean anything"
    );
    assert!(!donor_rows.is_empty(), "the donor must hold rows");
    // (1) The mechanism ran, (2) the rows really are shared in place on this
    // device (before S4 a CUDA run copied them), and (3) the donor survived it.
    assert!(cows > 0, "the store must have copied a private row");
    assert!(
        shared_cells >= shared_rows,
        "the prefix was copied, not shared, on device {}",
        model.device().name()
    );

    let (copied_answer, copied_rows, copied_cells, copied_cows, _, _) = scenario(false);
    assert_eq!(copied_rows, 0, "MINFER_NO_KV_SHARE must not share");
    assert_eq!(copied_cells, 0);
    assert_eq!(copied_cows, 0);
    // (4) And the answers agree: the shared run holds the same bytes in the same
    // order as the copied one, so a copy-on-write that moved the wrong rows (or
    // did not move them at all) shows up here.
    assert_eq!(
        shared_answer, copied_answer,
        "the shared run must answer like the shape-matched copied one"
    );
}

/// The K/V rows a sequence's written positions hold, one `Vec` per (layer, K or
/// V, position) in a deterministic order. Cells are resolved through the span
/// list on every call, so a relocation between two snapshots is not a difference;
/// a sharing sequence's shared rows are read from wherever they live.
fn kv_rows_of(engine: &mut BatchEngine, seq: SeqId) -> Vec<Vec<f32>> {
    let written = engine
        .cache
        .alloc()
        .kv_seq_slot(seq)
        .map_or(0, |s| s.written);
    let n_ctx = engine.cache.alloc().kv_n_ctx().max(1);
    let cells: Vec<usize> = (0..written)
        .map(|p| {
            engine
                .cache
                .alloc()
                .kv_cell_of(seq, p)
                .unwrap_or_else(|| panic!("no cell for sequence {seq} position {p}"))
        })
        .collect();
    // The region is over-allocated as f32 slots, but an f16 cache stores a row
    // as `nkt / 2` f32 slots (`store_kv_f16` indexes halves), so the window a
    // position covers is half as wide there. Reading it at the f32 width mixed
    // two rows per window and made this snapshot report differences in cells
    // nothing had written. #153: the answer is the **engine's** stamped format
    // (the same one its CUDA kernels run), not a process-wide device tag.
    let half_width = engine.cache.alloc().kv_format() == crate::graph::kvformat::KvFormat::F16;
    let mut out: Vec<Vec<f32>> = Vec::new();
    let mut layer = 0;
    while let Some((k, v)) = engine.cache.alloc().copy_kv_to_cpu(layer) {
        let row = if half_width {
            (k.len() / n_ctx / 2).max(1)
        } else {
            k.len() / n_ctx
        };
        for &cell in &cells {
            let at = cell * row;
            out.push(k[at..at + row].to_vec());
            out.push(v[at..at + row].to_vec());
        }
        layer += 1;
    }
    out
}

/// C7: the growth policy is pure, and it plans from the request rather than
/// from the startup partition.
#[test]
fn wanted_cells_plans_from_the_request() {
    // 4 slots over 2048 cells: 512 each. A prompt plus its answer that fit
    // ask for nothing new, so the common case never repartitions.
    assert_eq!(wanted_cells_from(100, 64, 512, 2048), 164);
    assert!(wanted_cells_from(100, 64, 512, 2048) <= 512);
    // An unbounded answer takes the slot's current capacity as headroom.
    assert_eq!(wanted_cells_from(100, -1, 512, 2048), 612);
    // A prompt past the partition asks for the prompt plus that headroom...
    assert_eq!(wanted_cells_from(1000, -1, 512, 2048), 1512);
    // ...clamped to the arena...
    assert_eq!(wanted_cells_from(1000, 100_000, 512, 2048), 2048);
    // ...and never below the prompt, so an impossible request stays the
    // caller's loud error instead of quietly shrinking into a smaller ask.
    assert_eq!(wanted_cells_from(4000, 8, 512, 2048), 4000);
}

/// C7 acceptance: a request whose prompt does not fit its share of the arena
/// is served anyway, and the repartition cannot change the answer.
///
/// Four slots over `n_ctx` give each slot `n_ctx/4`, and the prompt below
/// needs most of the arena, so it can only be served by reclaiming the idle
/// slots above it. Both admission paths are driven — `run_batched` goes
/// through `submit`/`prefill_group`, `run_serial` through `submit_on`, which
/// is the one the server uses — and both must agree with a one-slot engine,
/// which needs no reclaim at all. That equality is the C6 payoff: a moved row
/// keeps its sequence-relative position, so where the partition puts a
/// sequence cannot show up in its logits.
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn a_long_request_may_use_the_whole_arena() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the C7 gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode(&"buffalo ".repeat(300));
    let n_ctx = ids.len() + 64;
    assert!(
        ids.len() > n_ctx / 2,
        "the prompt must not fit a quarter-slot partition: {} tokens over {n_ctx} cells",
        ids.len()
    );
    let (group, _) = run_batched(&*model, &tok, &[ids.clone()], 4, n_ctx, 8, false);
    let (slot, _) = run_serial(&*model, &tok, &[ids.clone()], 4, n_ctx, 8);
    let (alone, _) = run_batched(&*model, &tok, &[ids], 1, n_ctx, 8, false);
    assert!(
        !group[0].text.is_empty() && !slot[0].text.is_empty(),
        "the long request must be served, not rejected ({} / {})",
        group[0].reason,
        slot[0].reason
    );
    assert_eq!(
        group[0].text, alone[0].text,
        "reclaiming idle capacity changed the continuation ({} vs {} tokens)",
        group[0].tokens, alone[0].tokens
    );
    assert_eq!(
        slot[0].text, alone[0].text,
        "the server's own admission path disagrees after a reclaim ({} vs {} tokens)",
        slot[0].tokens, alone[0].tokens
    );

    // #59: a request with no token budget is bounded by its run alone, and it must
    // stop *at* its last cell instead of forwarding one past it. Before the
    // commit-time check that forward was issued, `kv_cells_for_seq` rejected the
    // batch, and this request failed instead of finishing — so `reason == "length"`
    // with exactly the run's capacity in tokens is the regression test.
    let counter = tok.encode("Count slowly from 1 to 400, one number per line: 1,");
    let budget = 48;
    let (open_ended, _) = run_batched(
        &*model,
        &tok,
        &[counter.clone()],
        1,
        counter.len() + budget,
        -1,
        false,
    );
    assert_eq!(
        open_ended[0].reason, "length",
        "an unbounded request must end on the context bound, not fail: {:?}",
        open_ended[0]
    );
    assert_eq!(
        open_ended[0].tokens, budget,
        "it must use exactly the cells its run has"
    );
}

/// E2's acceptance, measured at the engine: four requests served as one decode
/// batch must generate exactly what four serial requests generate, and the
/// **median** of the per-round `serial/batched` ratios over interleaved rounds
/// must exceed 1.0 (#154).
///
/// Ignored by default like the other real-model tests:
///   cargo test --release --bin minfer -- --ignored server_batch --nocapture
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn server_batch_matches_serial_and_is_faster() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the batching measurement");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let texts = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Italy is",
        "The capital of Spain is",
    ];
    // Configurable so a bisect can put the *server's* exact configuration on
    // this path: the field bug of 2026-09-19 (plan §14) reproduces through the
    // server with the 7B and two slots, but not here with the 0.5B and four —
    // and these three knobs are the differences that are left.
    let templated = std::env::var("MINFER_BATCH_TEST_TEMPLATED").is_ok();
    let prompts: Vec<Vec<u32>> = texts
        .iter()
        .map(|p| {
            if templated {
                // Exactly what the server does (`server::mod`): the GGUF's
                // chat template, rendered with a generation prompt, then
                // tokenized. The legacy trait `format_chat` fallback (deleted
                // in #242) was a *different* path and produced a 13-token
                // prompt where the server's is 34 — which is why the first
                // bisect compared unequal inputs.
                let tpl = super::super::chat_template_from_gguf(&gguf.parts[0].data);
                let msgs = vec![("user".to_string(), Some(p.to_string()))];
                tok.encode(
                    &crate::template::render_messages_opt(
                        tpl.as_deref(),
                        &msgs,
                        true,
                        &tok.bos_text(),
                    )
                    .expect("chat template renders"),
                )
            } else {
                tok.encode(p)
            }
        })
        .collect();
    let n_ctx: usize = std::env::var("MINFER_BATCH_TEST_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let n_slots: usize = std::env::var("MINFER_BATCH_TEST_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let n_req = n_slots.min(prompts.len());
    let prompts: Vec<Vec<u32>> = prompts.into_iter().take(n_req).collect();
    let max_tokens = 16;
    // ---- #154: the throughput verdict is a median over interleaved rounds ----
    //
    // The pre-#154 form measured the two whole workloads **once, sequentially**
    // (batched, then serial) and asserted the wall-clock relation
    // `t_serial > t_batch`. That is not a property of the code: under the
    // parallel `--ignored` harness the first-measured phase absorbs the
    // start-up wave and the ratio can invert with nothing to absorb it.
    // Measured on dgxspark (CPU build) at `e1ac17f`: parallel **21.20s batched
    // vs 9.95s serial (0.47x)** — the only failure of the 29-gate set (28
    // passed / 1 failed) — while the same binary serially was **0.72s vs
    // 1.07s (1.50x)**.
    //
    // The rounds are now **interleaved** (batched, serial, batched, serial,
    // …), so each ratio is a matched pair measured next to each other on the
    // same machine state, and the assertion is on the **median of the
    // per-round `serial/batched` ratios** — the location estimate that
    // tolerates up to `rounds / 2` rounds a passing load spike disturbed.
    // Every sample and the median are printed so a loaded box's verdict is
    // auditable. This is the shape #123 gave
    // `cuda_map_window_costs_no_more_than_the_span_it_replaces`.
    //
    // Cost, and the round count. The correctness comparison below stays a
    // **full-length** pair (`max_tokens`); the timed rounds are a separate,
    // shorter workload so the gate's total wall clock stays bounded. On an
    // idle box the full-length pair is ~1.8s and the timing pair is
    // proportionally less; on the loaded parallel harness a full-length pair
    // reached ~31s. Seven rounds is the smallest odd count whose median
    // tolerates three disturbed rounds; `timing_tokens` halves the timed
    // workload's per-round cost without touching the correctness comparison.
    // Both knobs are env-overridable so a bisect can trade cost for samples.
    let rounds: usize = std::env::var("MINFER_BATCH_TEST_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let timing_tokens: i64 = std::env::var("MINFER_BATCH_TEST_TIMING_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    eprintln!(
        "[e2] config: {n_slots} slot(s), n_ctx {n_ctx}, {n_req} request(s), templated={templated}, \
         prompt len {}, correctness max_tokens {max_tokens}, {rounds} interleaved timing round(s) \
         at max_tokens {timing_tokens}",
        prompts[0].len()
    );

    // ---- correctness: the full-length pair, byte-comparable on CPU ----
    let (batched, t_batch) =
        run_batched(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens, false);
    let (serial, t_serial) = run_serial(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens);
    assert_replies_match("batched", &batched, &serial);
    // ---- the server's pattern: staggered admission (mixed step widths) ----
    let (staggered, t_stag) =
        run_batched(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens, true);
    assert_replies_match("staggered", &staggered, &serial);
    eprintln!(
        "[e2] staggered {:.2}s vs simultaneous {:.2}s for {n_slots} slots",
        t_stag, t_batch
    );
    let total: usize = serial.iter().map(|r| r.tokens).sum();
    assert!(total > 0, "the workload generated nothing");
    eprintln!(
        "[e2] {n_slots} slots: batched {t_batch:.2}s vs serial {t_serial:.2}s for {total} tokens \
         (per-request {:?} batched / {:?} serial)",
        batched.iter().map(|r| r.tokens).collect::<Vec<_>>(),
        serial.iter().map(|r| r.tokens).collect::<Vec<_>>()
    );
    eprintln!(
        "[e2] full-length single pair: {:.1} tok/s batched vs {:.1} tok/s serial = {:.2}x \
         (audit only, not the verdict)",
        total as f64 / t_batch,
        total as f64 / t_serial,
        t_serial / t_batch
    );

    // ---- throughput: interleaved matched rounds, verdict on the median ----
    let mut t_batches: Vec<f64> = Vec::with_capacity(rounds);
    let mut t_serials: Vec<f64> = Vec::with_capacity(rounds);
    let mut ratios: Vec<f64> = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let (_, tb) = run_batched(
            &*model,
            &tok,
            &prompts,
            n_slots,
            n_ctx,
            timing_tokens,
            false,
        );
        let (_, ts) = run_serial(&*model, &tok, &prompts, n_slots, n_ctx, timing_tokens);
        t_batches.push(tb);
        t_serials.push(ts);
        ratios.push(ts / tb);
    }
    let med_batch = median(&t_batches);
    let med_serial = median(&t_serials);
    let med = median(&ratios);
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "[e2] timing: {rounds} interleaved rounds at max_tokens {timing_tokens} — batched \
         {t_batches:?} s, serial {t_serials:?} s; per-round serial/batched ratios (round order) \
         {ratios:?}, sorted {sorted:?} — median {med:.3}x (medians {med_serial:.2}s serial vs \
         {med_batch:.2}s batched)"
    );
    // The threshold is the pre-#154 gate's "must not be slower": 1.0x, not a
    // named margin. What changed is the statistic (a median of matched pairs
    // instead of one sequential pair). On an idle CPU box the median is
    // ~1.5x, so the gate still trips on a mutation that halves the batched
    // arm's win (see the plan record's mutation check).
    assert!(
        med > 1.0,
        "batching must not be slower: median serial/batched {med:.3}x over {rounds} interleaved \
         rounds (per-round ratios {sorted:?}; medians {med_serial:.2}s serial vs {med_batch:.2}s \
         batched)"
    );
}
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
    // tolerance on CUDA, whose prefill tiles by `nt` and quantizes activations to
    // int8 — the same tokens at a different width land on different scores
    // (measured <= 0.37 absolute on this repo's fixtures; 1.0 is the gross-error
    // bound `cross_shape_tolerance` uses). The *continuation* is asserted only on
    // CPU: on a degenerate repeated-token prompt a sub-tolerance logit shift can
    // flip an argmax, which is a fact about the prompt, not about chunking.
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
    let tol = if on_cuda { 1.0 } else { 0.0 };
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
    if !on_cuda {
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
/// F8 (#51): `serve_loop` publishes the queue and running depth as it serves.
///
/// Two rounds, both driving the **real** serve loop with a real model:
///
/// 1. Two jobs on two slots: the feeder thread plays the HTTP handler's two
///    increments (`requests_total` on accept, `in_flight`), sends the jobs,
///    and watches the registry while the loop admits, steps and finishes
///    them. The gate asserts the running count was observed non-zero, that
///    every accepted job **left** the queue (`accepted - queue_depth == n`),
///    and that everything settles at zero.
///
///    `queue_depth` is deliberately **not** peak-asserted: `serve_loop` calls
///    `admit` on every iteration whether or not it is busy, so a job is
///    placed or rejected within one loop pass and its non-zero window is
///    shorter than a sampler's interval on dgxspark. Its arithmetic and its
///    saturation are gated purely in `server::metrics`, and the loop's *drain*
///    is gated here through `jobs_admitted_total`.
/// 2. Two jobs on **one** slot, both queued before the loop starts: the
///    "no idle slot" path is taken deterministically and its counter
///    (`jobs_dropped_total`) must fire — while the queue arithmetic still
///    balances, because a rejected job has also *left* the queue.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF).
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn serve_loop_publishes_the_queue_and_running_depth() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let Some(path) = cached_model() else {
        eprintln!("[f8] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let metric_source = |path: &str| -> Vec<u32> { tok.encode(path) };

    // --- round 1: two jobs, two slots, observed while running ---------------
    // 1024 rows over 2 slots = 512 a slot, far more than the 134 a 6-token
    // prompt asking for 128 tokens wants, so no admission has to repartition
    // the arena and both requests are served concurrently.
    let mut engine = BatchEngine::new(&*model, 2, 1024).expect("engine");
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let n = 2usize;
    // 128 tokens so the running window is seconds wide — far wider than the
    // sampler's 1 ms interval: "watched it run" must not be a race.
    let prompts: Vec<Vec<u32>> = (0..n)
        .map(|i| metric_source(&format!("Say the number {}.", i + 1)))
        .collect();
    let feeder_metrics = metrics.clone();
    let feeder = std::thread::spawn(move || {
        let mut rxs = Vec::new();
        for p in prompts {
            let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
            rxs.push(rx);
            job_tx
                .blocking_send(Job {
                    input_ids: p,
                    params: sampling_params(128),
                    tx,
                })
                .expect("send job");
            // The HTTP handler's side of the contract: accepted + in flight.
            feeder_metrics.requests_total.fetch_add(1, Ordering::SeqCst);
            feeder_metrics.in_flight.fetch_add(1, Ordering::SeqCst);
        }
        let mut peak_running = 0u64;
        // #160: `FEEDER_POLL_BACKSTOP` is **only a backstop**, not the gate's
        // failure signal. The verdict below is `peak_running > 0` plus the
        // queue arithmetic. Since #196 a wedge in `tick` no longer hangs the
        // join: `serve_loop` ends itself on its counted no-progress bound, so
        // the feeder also leaves as soon as it sees `worker_stalled_total`
        // move — the assertion that follows needs no deadline to run, and the
        // mutation run fails in seconds. What the terminator bounds is a
        // worker that never settles and never trips the bound; a healthy run
        // breaks out on `drained` within milliseconds.
        let deadline = Instant::now() + FEEDER_POLL_BACKSTOP;
        loop {
            let s = feeder_metrics.snapshot();
            peak_running = peak_running.max(s.running);
            let drained = s.running == 0 && s.queue_depth == 0 && s.requests_total >= n as u64;
            // Only stop once the run has actually been *seen* running: with
            // 256-token answers the window is seconds wide, so this is an
            // assertion about the worker, not about sampler luck.
            if (peak_running > 0 && drained)
                || s.worker_stalled_total > 0
                || Instant::now() > deadline
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // Drain the responses; the worker's `blocking_send` must never wedge.
        // #196: a healthy `serve_loop` never trips its no-progress bound, so
        // every response here is `Text`/`Finish`; the errors are printed so a
        // mutation run shows the terminal answer each client actually got.
        let mut errors: Vec<String> = Vec::new();
        for mut rx in rxs {
            while let Some(ev) = rx.blocking_recv() {
                if let StreamEvent::Err(e) = ev {
                    errors.push(format!("{} {} ({})", e.status, e.message, e.error_type));
                }
            }
        }
        if !errors.is_empty() {
            eprintln!(
                "[f8] serve_loop answered {} error(s): {}",
                errors.len(),
                errors.join("; ")
            );
        }
        peak_running
    });

    super::serve_loop(&*model, &tok, job_rx, &mut engine, &metrics);
    let peak_running = feeder.join().expect("feeder");

    let s = metrics.snapshot();
    assert_eq!(
        s.worker_stalled_total, 0,
        "the worker tripped its counted no-progress bound: a healthy serve_loop \
         always advances work on a step that leaves the engine busy (#196)"
    );
    assert_eq!(
        s.requests_total - s.queue_depth,
        n as u64,
        "every accepted job left the queue"
    );
    assert_eq!(s.jobs_dropped_total, 0, "two slots served two jobs");
    assert_eq!(s.running, 0, "nothing is running once the loop drains");
    assert_eq!(s.worker_pending, 0);
    assert!(
        peak_running > 0,
        "the running count was never observed non-zero"
    );
    // The KV reading is live, not a startup snapshot.
    assert_eq!(s.kv.layers, model.n_layer() as u64);
    assert!(s.kv.region_bytes > 0);

    // --- round 2: two jobs, one slot, both queued up front ------------------
    let mut one = BatchEngine::new(&*model, 1, 512).expect("engine");
    let one_metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (tx1, rx1) = mpsc::channel::<Job>(64);
    for p in [metric_source("Alpha"), metric_source("Beta")] {
        let (tx, _rx) = mpsc::channel::<StreamEvent>(1024);
        tx1.blocking_send(Job {
            input_ids: p,
            params: sampling_params(2),
            tx,
        })
        .expect("queue job");
        one_metrics.requests_total.fetch_add(1, Ordering::SeqCst);
    }
    drop(tx1); // no more senders: the loop drains and exits
    super::serve_loop(&*model, &tok, rx1, &mut one, &one_metrics);
    let s1 = one_metrics.snapshot();
    assert_eq!(
        s1.requests_total - s1.queue_depth,
        2,
        "a rejected job still left the queue"
    );
    assert_eq!(
        s1.jobs_dropped_total, 1,
        "the second job cannot fit one slot and must be counted as dropped"
    );
    assert_eq!(
        s1.worker_stalled_total, 0,
        "the worker tripped its counted no-progress bound on a run that must \
         simply serve one request and reject the other (#196)"
    );
    assert_eq!(s1.running, 0);
    assert_eq!(s1.worker_pending, 0);

    eprintln!(
        "[f8] serve_loop: peak running {peak_running}, layers {} region {} B owned {} \
         cells; one slot -> {} dropped",
        s.kv.layers, s.kv.region_bytes, s.kv.owned_cells, s1.jobs_dropped_total
    );
}

/// #121: the answer a rejected job gets, per transport.
///
/// Drives the real handler functions with the event [`reject`] produces:
/// `collect_response` must surface a **503** rather than read the closed
/// channel as a completed empty answer, and `stream_response` must emit a
/// `data:` **error frame** rather than an empty stream followed by `[DONE]`.
/// No model is involved, so this half runs in CI; the coupling — that a
/// saturated engine really calls `reject` — is the `#[ignore]`d real-model
/// gate below (`a_job_rejected_for_want_of_a_slot_is_answered_with_503`).
///
/// A plain `#[test]` with its own runtime: `reject` runs on the worker (a
/// non-async thread, as in production), and `blocking_send` correctly refuses
/// to block a runtime thread — so it is called *outside* `block_on`.
#[test]
fn a_rejected_job_answers_503_and_an_sse_error_frame() {
    use std::sync::Arc;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let job = |tx: mpsc::Sender<StreamEvent>| Job {
        input_ids: vec![1, 2, 3],
        params: sampling_params(4),
        tx,
    };

    // Non-streaming: the error reaches the handler as `Err(e)`, and `e` is a
    // 503 for the HTTP layer.
    let (tx, rx) = mpsc::channel::<StreamEvent>(4);
    let e = reject(job(tx), ApiError::unavailable("no idle slot"));
    assert_eq!(e.status, 503, "the engine rejects with 503");
    let from_stream = rt
        .block_on(crate::server::collect_response(rx))
        .expect_err("a rejected job must not read as a completed empty answer");
    assert_eq!(from_stream.status, 503, "{}", from_stream.message);
    assert!(
        from_stream.message.contains("no idle slot"),
        "the reason must survive: {}",
        from_stream.message
    );
    assert_eq!(
        crate::server::error_response(&from_stream).status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );

    // Streaming: the same event is an SSE error frame — not a silent close.
    // The `Sse` response itself is built inside the runtime (axum's keep-alive
    // arms a timer), while `reject` stays outside it.
    let (tx, rx) = mpsc::channel::<StreamEvent>(4);
    let _ = reject(job(tx), ApiError::unavailable("no idle slot"));
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let body = rt.block_on(async move {
        let resp = crate::server::stream_response(
            "chatcmpl-test",
            "test-model",
            0,
            rx,
            crate::server::InFlight::new(metrics.clone()),
            metrics,
            3,
        );
        axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("SSE body")
    });
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.contains("no idle slot"),
        "the SSE error frame is missing: {body}"
    );
    assert!(
        body.contains("\"code\":503"),
        "the frame must carry the status: {body}"
    );
    assert!(
        !body.contains("\"finish_reason\":\"stop\""),
        "a rejected request must not look finished: {body}"
    );
}

/// #121: a saturated engine **answers** the job it cannot place.
///
/// One slot, two jobs queued before the loop starts, so the rejection is
/// deterministic (the setup F8's round 2 already used) — but here both
/// receivers are kept and read. `A` is served (`Finish`, tokens > 0); `B`
/// gets **exactly one** `StreamEvent::Err` with status 503 and the
/// `no idle slot` message, and nothing else. `jobs_dropped_total` moves by
/// exactly one.
///
/// Before the fix `B`'s channel closed with no events at all, which the
/// handler rendered as HTTP 200 with empty content.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF).
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn a_job_rejected_for_want_of_a_slot_is_answered_with_503() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let Some(path) = cached_model() else {
        eprintln!("[f8] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let mut engine = BatchEngine::new(&*model, 1, 512).expect("engine");
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let mut rxs = Vec::new();
    for prompt in ["Alpha", "Beta"] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
        rxs.push(rx);
        job_tx
            .blocking_send(Job {
                input_ids: tok.encode(prompt),
                // Two tokens, so the served request finishes immediately and
                // the second job is rejected in the very first `admit` pass.
                params: sampling_params(2),
                tx,
            })
            .expect("queue job");
        // The HTTP handler's side of the contract (F8): accepted + in flight.
        metrics.requests_total.fetch_add(1, Ordering::SeqCst);
        metrics.in_flight.fetch_add(1, Ordering::SeqCst);
    }
    drop(job_tx); // no more senders: the loop drains and exits
    super::serve_loop(&*model, &tok, job_rx, &mut engine, &metrics);

    let s = metrics.snapshot();
    assert_eq!(
        s.jobs_dropped_total, 1,
        "exactly one job could not be placed on the one slot"
    );

    // Read both channels: one request was served, one was answered with an
    // error. Classify by content, not by index, so the gate cannot pass
    // because the two happened to be swapped — and name the silent-drop
    // shape explicitly, so the *mutation* that puts it back reports the
    // defect instead of a confusing "the served job must finish".
    let mut served = 0usize;
    let mut rejected = 0usize;
    for mut rx in rxs {
        let mut events = Vec::new();
        while let Some(ev) = rx.blocking_recv() {
            events.push(ev);
        }
        let err = events.iter().find_map(|ev| match ev {
            StreamEvent::Err(e) => Some(e),
            _ => None,
        });
        let finish = events.iter().find_map(|ev| match ev {
            StreamEvent::Finish { tokens, .. } => Some(*tokens),
            _ => None,
        });
        match (err, finish) {
            (Some(e), None) => {
                assert_eq!(
                    e.status, 503,
                    "a rejected job is unavailable: {}",
                    e.message
                );
                assert!(
                    e.message.contains("no idle slot"),
                    "the reason must reach the client: {}",
                    e.message
                );
                assert_eq!(
                    events.len(),
                    1,
                    "the rejected job gets exactly the error, then the channel closes"
                );
                rejected += 1;
            }
            (None, Some(tokens)) => {
                assert!(tokens > 0, "the served job produced {tokens} tokens");
                served += 1;
            }
            (Some(_), Some(_)) => panic!(
                "a job cannot be both rejected and finished ({} event(s))",
                events.len()
            ),
            (None, None) => panic!(
                "a job's channel closed with {} event(s) — the #121 silent drop is back \
                 (the handler would answer HTTP 200 with empty content)",
                events.len()
            ),
        }
    }
    assert_eq!(
        (served, rejected),
        (1, 1),
        "one slot serves one request and answers the other"
    );
    // #196: checked *after* the channels are read, so a wedge — which answers
    // the served run with `500 the worker stalled` instead of a `Finish` —
    // surfaces as the terminal error the client actually got, above, and this
    // is the assertion that names the guard itself.
    assert_eq!(
        s.worker_stalled_total, 0,
        "the worker tripped its counted no-progress bound: a one-slot run that \
         serves one job and rejects the other must never stall (#196)"
    );
}

/// #151's deterministic injection: a model whose **batch forward fails**.
///
/// `forward_batch` is the only forward the batched engine calls, and it cannot
/// return an error — a real failure is a panic that `guarded_forward_batch`
/// turns into `ApiError::server` (every refusal inside
/// `Qwen2Graph::forward_batch`, e.g. the `position exceeds n_ctx` invariant, is
/// exactly that shape). This double reproduces it: it counts attempts so the
/// gate can see a retry, and panics with a fixed message, so the failure
/// reaches `tick` through the **real** guard and the real error path. No model
/// on disk is involved, so the gate runs in CI.
struct FailingForward {
    attempts: std::sync::atomic::AtomicUsize,
}

impl FailingForward {
    fn new() -> Self {
        Self {
            attempts: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Batch forwards attempted. 1 after the failure and still 1 once the runs
    /// have been answered; an unbroken retry makes it grow.
    fn attempts(&self) -> usize {
        self.attempts.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl ModelDef for FailingForward {
    fn forward(
        &self,
        _tokens: &[u32],
        _positions: &[usize],
        _n_out: usize,
        _n_ctx: usize,
    ) -> Vec<f32> {
        unreachable!("the batched engine calls forward_batch, never forward")
    }

    fn forward_batch(
        &self,
        _batch: &crate::graph::batch::Batch,
        _n_out: usize,
        _n_ctx: usize,
        _cache: &mut GraphCache,
    ) -> Vec<f32> {
        self.attempts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        panic!("deterministic decode-forward failure (#151's gate injection)");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn kv_format(&self) -> crate::graph::kvformat::KvFormat {
        crate::graph::kvformat::KvFormat::F32
    }

    fn set_kv_format(&mut self, _format: crate::graph::kvformat::KvFormat) {}

    fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens {
            eos: 0,
            im_end: None,
        }
    }

    fn n_layer(&self) -> usize {
        1
    }
    fn n_head_kv(&self) -> usize {
        1
    }
    fn n_embd_head(&self) -> usize {
        1
    }
    fn n_kv_embd(&self) -> usize {
        1
    }
    fn n_vocab(&self) -> usize {
        4
    }
    fn rope_style(&self) -> crate::vec_ops::RopeStyle {
        crate::vec_ops::RopeStyle::NonInterleaved
    }
    fn rope_params(&self) -> (f32, f32) {
        (10_000.0, 1.0)
    }
}

/// Put a live request on `idx` whose next token is already committed and
/// waiting for a decode forward — the state `tick`'s batch builder looks for.
/// `cached_tokens` is pre-filled so the gate can see the failure clear it.
fn install_pending_run(
    engine: &mut BatchEngine,
    idx: usize,
    tx: mpsc::Sender<StreamEvent>,
    tok: u32,
) {
    let params = sampling_params(8);
    let cfg = params.sampler_config();
    let mirostat = crate::sampler::MirostatState::new(params.mirostat_tau);
    let rng = StdRng::seed_from_u64(params.seed);
    engine.slots[idx].cached_tokens = vec![1, 2, 3];
    engine.slots[idx].run = Some(Run {
        tx,
        params,
        cfg,
        mirostat,
        grammar_state: None,
        rng,
        prev_tokens: Vec::new(),
        stop_bytes: Vec::new(),
        full: Vec::new(),
        emitted: 0,
        completion_tokens: 0,
        current_pos: 3,
        last_logits: vec![0.0; 4],
        needs_forward: Some(tok),
        live_on: false,
    });
}

/// Drain a run's channel and report `(errors, finishes, text chunks, closed)`.
/// `closed` is true when the sender is gone — the observable that the slot was
/// released and nothing further will arrive.
fn read_run_channel(rx: &mut mpsc::Receiver<StreamEvent>) -> (Vec<ApiError>, usize, usize, bool) {
    let (mut errs, mut finishes, mut texts) = (Vec::new(), 0usize, 0usize);
    loop {
        match rx.try_recv() {
            Ok(StreamEvent::Err(e)) => errs.push(e),
            Ok(StreamEvent::Finish { .. }) => finishes += 1,
            Ok(StreamEvent::Text(_)) => texts += 1,
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => return (errs, finishes, texts, true),
        }
    }
    (errs, finishes, texts, false)
}

/// #151: a failed decode forward answers the **one** run whose row it carried,
/// instead of leaving it stuck.
///
/// The defect is an infinite retry, so the gate bounds itself instead of
/// hanging: it drives `tick` a second time and asserts the forward was
/// attempted **once**. With the pre-fix early `return`, the run keeps
/// `needs_forward`, so the second `tick` attempts the same forward again
/// (attempts == 2) and the client never hears.
#[test]
fn a_failed_decode_forward_answers_a_single_run_and_releases_its_slot() {
    let model = FailingForward::new();
    let mut engine = BatchEngine::new(&model, 1, 32).expect("engine");
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(8);
    install_pending_run(&mut engine, 0, tx, 7);
    assert_eq!(engine.running_slots(), 1);

    let err = engine
        .tick(&model, &Tokenizer::empty())
        .expect_err("the forward failed, so the step must report it");
    assert_eq!(
        err.status, 500,
        "a step failure is a server error: {}",
        err.message
    );
    assert_eq!(err.error_type, "server_error");

    let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
    assert_eq!(errs.len(), 1, "exactly one error, not zero and not a retry");
    assert_eq!(finishes, 0, "a failed run must not be finished");
    assert_eq!(texts, 0, "a failed run must not emit text");
    assert_eq!(
        errs[0].status, 500,
        "the run's own error: {}",
        errs[0].message
    );
    assert_eq!(errs[0].error_type, "server_error");
    assert!(
        !errs[0].message.is_empty(),
        "the client must be told why, not just that"
    );
    assert!(closed, "the run's sender is dropped: the slot is released");
    assert!(
        engine.slots[0].run.is_none(),
        "the slot must be free after the failure"
    );
    assert!(
        engine.slots[0].cached_tokens.is_empty(),
        "a half-written forward's prefix must not be reused"
    );
    assert_eq!(engine.idle_slots(), 1);
    assert!(!engine.busy(), "the failed request is no longer in flight");

    let attempts_after_failure = model.attempts();
    assert_eq!(attempts_after_failure, 1, "one forward for the one run");
    engine
        .tick(&model, &Tokenizer::empty())
        .expect("nothing is pending, so the next step is a no-op");
    assert_eq!(
        model.attempts(),
        attempts_after_failure,
        "the loop must not retry the failed forward"
    );
}

/// #151: a failed decode forward answers **every** run whose row was in the
/// batch, and leaves a run whose row was not.
///
/// Four slots: three with a pending token (one batch of three rows) and one
/// live run already sampled (`needs_forward = None`). The failure must answer
/// the three and touch nothing on the fourth — the membership rule, and what
/// keeps the fix from blaming a run the forward never carried.
#[test]
fn a_failed_decode_forward_answers_every_row_in_the_batch() {
    let model = FailingForward::new();
    let mut engine = BatchEngine::new(&model, 4, 64).expect("engine");
    let mut rxs = Vec::new();
    for (slot, tok) in [(0usize, 11u32), (1, 12), (2, 13)] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(8);
        install_pending_run(&mut engine, slot, tx, tok);
        rxs.push(rx);
    }
    // Slot 3 is live but has no committed-but-unwritten token: its row is not
    // in the batch, so this batch's failure is not its error.
    let (quiet_tx, mut quiet_rx) = mpsc::channel::<StreamEvent>(8);
    install_pending_run(&mut engine, 3, quiet_tx, 14);
    engine.slots[3]
        .run
        .as_mut()
        .expect("run installed")
        .needs_forward = None;
    assert_eq!(engine.running_slots(), 4);

    let err = engine
        .tick(&model, &Tokenizer::empty())
        .expect_err("the three-row batch forward failed");
    assert_eq!(err.status, 500);

    for (slot, rx) in rxs.iter_mut().enumerate() {
        let (errs, finishes, texts, closed) = read_run_channel(rx);
        assert_eq!(errs.len(), 1, "slot {slot}: exactly one error");
        assert_eq!(errs[0].status, 500, "slot {slot}: {}", errs[0].message);
        assert_eq!(errs[0].error_type, "server_error", "slot {slot}");
        assert_eq!(finishes, 0, "slot {slot}: no Finish");
        assert_eq!(texts, 0, "slot {slot}: no text");
        assert!(closed, "slot {slot}: released");
        assert!(engine.slots[slot].run.is_none(), "slot {slot}: free");
        assert!(engine.slots[slot].cached_tokens.is_empty());
    }
    // The run outside the batch survives it: still live, still its own sender,
    // still no event and its prefix untouched.
    assert!(
        engine.slots[3].run.is_some(),
        "a run outside the failed batch must be left alone"
    );
    assert_eq!(engine.slots[3].cached_tokens, vec![1, 2, 3]);
    let (q_errs, q_finishes, q_texts, q_closed) = read_run_channel(&mut quiet_rx);
    assert_eq!((q_errs.len(), q_finishes, q_texts), (0, 0, 0));
    assert!(!q_closed, "slot 3's run is still connected");

    let attempts_after_failure = model.attempts();
    assert_eq!(
        attempts_after_failure, 1,
        "one forward for the three-row batch"
    );
    // Drop the untouched run so the no-op step below cannot try to sample it
    // (this gate's tokenizer has no vocabulary); the retry assertion is about
    // the forward, which is what would have repeated.
    engine.slots[3].run = None;
    engine
        .tick(&model, &Tokenizer::empty())
        .expect("nothing is pending, so the next step is a no-op");
    assert_eq!(
        model.attempts(),
        attempts_after_failure,
        "the loop must not retry the failed forward"
    );
    assert!(!engine.busy());
}

/// #196: the stall path answers **every** live run exactly once.
///
/// `serve_loop`'s no-progress bound knows the engine is wedged but not which
/// batch wedged it, so [`BatchEngine::fail_all`] is its `fail_batch` without a
/// row list. Three live runs and one idle slot: each live run gets exactly one
/// `500 the worker stalled` (never zero — #121/#151's dropped-or-never-answered
/// defect — and never a second), is taken so the slot is free and no retry can
/// repeat the wedged batch, and has its cached prefix cleared. The idle slot is
/// untouched. No model is involved, so this half runs in CI.
#[test]
fn the_stall_answers_every_live_run_exactly_once() {
    let model = FailingForward::new();
    let mut engine = BatchEngine::new(&model, 4, 64).expect("engine");
    let mut rxs = Vec::new();
    for slot in [0usize, 1, 3] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(8);
        install_pending_run(&mut engine, slot, tx, 20 + slot as u32);
        rxs.push((slot, rx));
    }
    assert_eq!(engine.running_slots(), 3);

    let e = ApiError::server(WORKER_STALLED_MESSAGE);
    let answered = engine.fail_all(&e);
    assert_eq!(answered, 3, "every live run is answered");
    assert!(!engine.busy(), "no run is left occupying a slot");
    assert_eq!(engine.idle_slots(), 4);

    for (slot, mut rx) in rxs {
        let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
        assert_eq!(errs.len(), 1, "slot {slot}: exactly one terminal error");
        assert_eq!(
            (finishes, texts),
            (0, 0),
            "slot {slot}: the error is the whole answer"
        );
        assert_eq!(errs[0].status, 500, "slot {slot}: {}", errs[0].message);
        assert_eq!(errs[0].error_type, "server_error", "slot {slot}");
        assert_eq!(errs[0].message, WORKER_STALLED_MESSAGE, "slot {slot}");
        assert!(
            closed,
            "slot {slot}: the sender is dropped, so the slot is released"
        );
        assert!(engine.slots[slot].run.is_none(), "slot {slot}: free");
        assert!(
            engine.slots[slot].cached_tokens.is_empty(),
            "slot {slot}: a wedged engine's prefix must not be reused"
        );
    }
    assert!(engine.slots[2].run.is_none(), "the idle slot was untouched");
    assert_eq!(
        engine.fail_all(&e),
        0,
        "an already-idle engine has nothing to answer"
    );
}

/// #196: the stall path answers every **queued but unadmitted** job too.
///
/// Ending `serve_loop` drops a queued [`Job`] — and with it the only
/// `Sender<StreamEvent>` its handler listens on, which the handler reads as a
/// *completed* empty answer (#121's silent drop). [`reject_queued`] answers the
/// worker's own deque and whatever is still in the channel with the same
/// terminal error, exactly once each, before the loop ends. CI: no model.
#[test]
fn the_stall_answers_every_queued_job_exactly_once() {
    let (job_tx, mut job_rx) = mpsc::channel::<Job>(8);
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut rxs = Vec::new();
    for i in 0..3 {
        let (tx, rx) = mpsc::channel::<StreamEvent>(8);
        rxs.push(rx);
        let job = Job {
            input_ids: vec![1, 2, 3],
            params: sampling_params(2),
            tx,
        };
        if i == 0 {
            pending.push_back(job); // already in the worker's deque
        } else {
            job_tx.blocking_send(job).expect("queue job"); // still in the channel
        }
    }

    let e = ApiError::server(WORKER_STALLED_MESSAGE);
    let answered = reject_queued(&mut pending, &mut job_rx, &e);
    assert_eq!(answered, 3, "every queued job is answered");
    assert!(pending.is_empty(), "the worker's deque is drained");

    for (i, mut rx) in rxs.into_iter().enumerate() {
        let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
        assert_eq!(errs.len(), 1, "job {i}: exactly one terminal error");
        assert_eq!(
            (finishes, texts),
            (0, 0),
            "job {i}: never a Finish and never text"
        );
        assert_eq!(errs[0].status, 500, "job {i}: {}", errs[0].message);
        assert_eq!(errs[0].error_type, "server_error", "job {i}");
        assert_eq!(errs[0].message, WORKER_STALLED_MESSAGE, "job {i}");
        assert!(closed, "job {i}: the sender is dropped after the answer");
    }
}

/// #171 deliverable D: the seam replaces #151's bespoke mock, on the real path.
///
/// The gate above needs [`FailingForward`], a test-local `ModelDef`, because a
/// failed batch forward has no other deterministic trigger. The
/// failure-injection seam makes that mock one environment variable: this gate
/// loads the **real** cached 0.5B, arms `MINFER_TEST_CALL_FAIL=forward_batch`
/// for the scope, and asserts the #151 contract still holds — exactly one
/// `500` to the run's own sender, the slot released, no retry — plus that the
/// chokepoint's observation counter proves the real forward entry was reached.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF). It is run serially
/// by `scripts/real_model_gates.sh`; the transcript lives in the #171 record
/// because CI cannot run it.
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn the_seam_fails_the_batch_forward_without_a_bespoke_mock() {
    let Some(path) = cached_model() else {
        eprintln!("[#171] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    // Arm the seam for exactly this scope; the guard restores the process
    // value on drop, so a panicking gate cannot leave the switch on.
    let _seam = crate::testfail::InjectionGuard::arm("forward_batch");
    crate::testfail::reset_checked();

    let mut engine = BatchEngine::new(&*model, 1, 64).expect("engine");
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(8);
    // One committed token waiting for its decode forward — the state `tick`'s
    // batch builder looks for, so no prefill is involved.
    install_pending_run(&mut engine, 0, tx, 7);
    assert_eq!(engine.running_slots(), 1);

    let err = engine
        .tick(&*model, &tok)
        .expect_err("the injected forward failure must be reported");
    assert_eq!(
        err.status, 500,
        "a step failure is a server error: {}",
        err.message
    );
    assert_eq!(err.error_type, "server_error");

    let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
    assert_eq!(errs.len(), 1, "exactly one error, and no retry");
    assert_eq!(errs[0].status, 500, "{}", errs[0].message);
    assert_eq!((finishes, texts), (0, 0), "a failed run emits neither");
    assert!(closed, "the run's sender is dropped: the slot is released");
    assert!(engine.slots[0].run.is_none(), "the slot must be free");
    assert!(
        engine.slots[0].cached_tokens.is_empty(),
        "a half-written forward's prefix must not be reused"
    );
    assert!(!engine.busy());
    assert_eq!(
        crate::testfail::checked("forward_batch"),
        1,
        "the real batch-forward entry was reached exactly once before the seam fired"
    );

    // Disarm and prove the path is clean again: the next step is a no-op and
    // the observation counter does not move.
    drop(_seam);
    crate::testfail::reset_checked();
    engine
        .tick(&*model, &tok)
        .expect("nothing is pending, so the next step is a no-op");
    assert_eq!(crate::testfail::checked("forward_batch"), 0);
    eprintln!(
        "[#171] forward_batch seam: one 500, slot released, no retry, \
         {} observed forward entries",
        1
    );
}

/// #160: the `serve_loop_publishes_the_queue_and_running_depth` feeder's poll
/// terminator, named so the number is justified where it is used.
///
/// It is **not** a gate's failure signal: the verdict is the downstream
/// `peak_running > 0` and the queue arithmetic. Since #196 the worker also
/// stops itself, so this is no longer the only way out of the join: the feeder
/// leaves as soon as `worker_stalled_total` moves (the mutation run fails in
/// seconds), and it leaves on the healthy `drained` condition within
/// milliseconds. It is also not load-sensitive: `peak_running` is observed
/// within milliseconds of the first admission on every run. The value is
/// generous only because it is the last resort for a worker that drains but
/// whose published metrics never settle.
const FEEDER_POLL_BACKSTOP: std::time::Duration = std::time::Duration::from_secs(120);

/// The step budget for a real-model stepper loop (#158, shared by #160).
///
/// A legitimate run does one decode forward and one sample per answer token,
/// plus one prefill forward per chunk (with `MINFER_N_BATCH` low, one per
/// prompt token), so `MARGIN * (prompt + max_tokens + SLACK)` is far above
/// anything the engine can legitimately need. `MARGIN = 4` and `SLACK = 8` are
/// deliberately loose: this bound exists to catch an engine that *cannot*
/// terminate, never one that is merely slow, and a false negative hangs the
/// suite while a false positive is the flaky gate this ticket removes.
///
/// Measured on dgxspark (0.5B q4_0, GB10 host CPU, 2026-09-25): the warm
/// request (8-token prompt, `max_tokens = 4`) takes **4 steps** against a
/// budget of **80**, and the long one (120-token prompt, `max_tokens = 64`)
/// takes **64** against **768** — margins of 20x and 12x. The gates print both
/// counts on every run, so a workload that outgrows the budget says so.
const STEP_BUDGET_MARGIN: usize = 4;
const STEP_BUDGET_SLACK: usize = 8;

fn step_budget(prompt_tokens: usize, max_tokens: usize) -> usize {
    STEP_BUDGET_MARGIN * (prompt_tokens + max_tokens + STEP_BUDGET_SLACK)
}

/// #160: the answer tokens a request may legitimately produce, for
/// [`step_budget`]. An unbounded request (`max_tokens < 0`) ends on its
/// context bound, which is the cap the engine itself enforces.
fn step_cap(max_tokens: i64, n_ctx: usize) -> usize {
    if max_tokens < 0 {
        n_ctx
    } else {
        max_tokens as usize
    }
}

/// The per-step bound every stepper loop in this module shares (#158, made
/// the common shape by #160).
///
/// A `tick` that leaves the engine busy must have advanced
/// [`BatchEngine::work_units`]: if no forward ran, then some slot's `advance`
/// returned `Continue`, and `Continue` commits exactly one token (every other
/// `advance` outcome ends the run and takes it). A drive can violate that in
/// exactly two ways, and this type names the arm that catches each:
///
/// - the counter freezes while the engine stays busy — a **wedge**; [`step`]
///   panics on the step that wedged it (this is the arm `MINFER_TEST_TICK=wedge`
///   drives), or
/// - the counter keeps moving along a path that cannot terminate — only the
///   **step budget** catches that (the arm `MINFER_TEST_TICK=spin` drives).
///
/// Neither bound is a wall-clock number, so a loaded box runs the same steps
/// more slowly and still passes.
///
/// [`step`]: WorkBound::step
struct WorkBound<'a> {
    what: &'a str,
    budget: usize,
    steps: usize,
    work: u64,
}

impl<'a> WorkBound<'a> {
    fn new(engine: &BatchEngine, budget: usize, what: &'a str) -> Self {
        Self {
            what,
            budget,
            steps: 0,
            work: engine.work_units(),
        }
    }

    /// Record one step and assert it was honest. Call it immediately after
    /// every `tick` whose `busy()` state the drive is about to re-check.
    fn step(&mut self, engine: &BatchEngine) {
        self.steps += 1;
        let now = engine.work_units();
        assert!(
            !engine.busy() || now > self.work,
            "{}: the engine is wedged — step {} left it busy without advancing \
             the work counter (still {})",
            self.what,
            self.steps,
            self.work
        );
        self.work = now;
        assert!(
            self.steps <= self.budget,
            "{}: {} steps exceeded the {}-step budget — the run is not \
             progressing toward completion",
            self.what,
            self.steps,
            self.budget
        );
    }
}

/// #158: drive `engine` to idle with a bound on *work*, not wall-clock seconds.
///
/// Replaces the absolute deadlines this gate used (`Instant::now() +
/// Duration::from_secs(120/180)`), which a slow box could exceed while a
/// healthy engine ran on. Every `tick` that leaves the engine busy must
/// advance [`BatchEngine::work_units`] — a wedged engine (one whose `tick`
/// returns without forwarding or committing) trips that assertion on the
/// stalling step — and the step budget backstops an engine that keeps "moving"
/// along a path that cannot terminate. Neither bound is a wall-clock number, so
/// the verdict is a property of the code. Returns the number of steps it took.
fn drive_by_work(
    engine: &mut BatchEngine,
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    rx: &mut mpsc::Receiver<StreamEvent>,
    budget: usize,
    what: &str,
) -> usize {
    let mut bound = WorkBound::new(engine, budget, what);
    while engine.busy() {
        engine.tick(model, tokenizer).expect("tick");
        // The response channel is bounded, so every step's frames are drained:
        // a full channel would block the worker's `blocking_send` and stall the
        // step before the work assertion could see it.
        while rx.try_recv().is_ok() {}
        bound.step(engine);
    }
    bound.steps
}

/// F8 (#51): the published occupancy and running counts are a **live**
/// reading, not a startup snapshot — they move as requests are served.
///
/// Real-model gate (`#[ignore]`, like the rest of this module's measurement
/// tests: CI has no cached GGUF). Run it with
/// `cargo test --release --bin minfer -- --ignored --test-threads=1`.
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn published_metrics_move_as_requests_are_served() {
    let Some(path) = cached_model() else {
        eprintln!("[f8] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let n_slots = 2usize;
    let n_ctx = 128usize;
    let mut engine = BatchEngine::new(&*model, n_slots, n_ctx).expect("engine");
    let metrics = crate::server::metrics::ServerMetrics::new();

    // Before any forward: sequences are reserved, but no KV region exists yet
    // (the arena is sized by the first `alloc_graph`), so the layers gauge is
    // genuinely 0 — the reading tracks the arena, it does not assume one.
    engine.publish_metrics(&metrics);
    let cold = metrics.snapshot();
    assert_eq!(cold.kv.layers, 0, "no graph built yet, so no KV region");
    assert_eq!(cold.kv.sequences, n_slots as u64);
    assert_eq!(cold.running, 0);

    // Serve one request a step at a time and watch the numbers move.
    let prompt: Vec<u32> = (1..=8).collect();
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
    engine
        .submit_on(
            &*model,
            &tok,
            0,
            Job {
                input_ids: prompt.clone(),
                params: sampling_params(4),
                tx,
            },
        )
        .expect("submit");
    engine.publish_metrics(&metrics);
    let warm = metrics.snapshot();
    assert_eq!(
        warm.kv.layers,
        model.n_layer() as u64,
        "the first forward allocates every layer's KV region"
    );
    assert_eq!(warm.kv.rows, n_ctx as u64);
    assert!(warm.kv.region_bytes > 0, "a live arena has bytes");
    assert_eq!(warm.running, 1, "one slot holds a live request");
    // The admission prefilled the prompt, so those cells are already owned —
    // the gauge tracks the store, it does not wait for a decode step.
    assert!(
        warm.kv.owned_cells >= prompt.len() as u64,
        "the prefill wrote the prompt's {} cells (got {})",
        prompt.len(),
        warm.kv.owned_cells
    );

    // Step until the request finishes; `running` must fall back to 0 and the
    // owned-cell gauge must have grown. The bound is on work, not seconds
    // (#158): a loaded box runs the same steps more slowly and still passes,
    // while a wedged engine stops moving the counter and trips the assertion.
    let steps = drive_by_work(
        &mut engine,
        &*model,
        &tok,
        &mut rx,
        step_budget(prompt.len(), 4),
        "the warm request",
    );
    eprintln!("[f8] warm request: {steps} step(s)");
    assert!(!engine.busy(), "the warm request completed");
    engine.publish_metrics(&metrics);
    let done = metrics.snapshot();
    assert_eq!(done.running, 0, "no live request is running");
    assert!(
        done.kv.owned_cells >= warm.kv.owned_cells,
        "the run wrote at least the prompt's cells ({} -> {})",
        warm.kv.owned_cells,
        done.kv.owned_cells
    );
    assert_eq!(done.kv.layers, warm.kv.layers);

    // The snapshot is exactly what `/metrics` renders, so the endpoint sees
    // the live numbers too.
    let text = metrics.render();
    assert!(text.contains(&format!("minfer_kv_layers {}\n", model.n_layer())));
    assert!(text.contains(&format!("minfer_kv_rows {n_ctx}\n")));
    assert!(text.contains("minfer_requests_running 0\n"));

    // --- growth and release: the same gauges across an arena repartition -----
    // Issue #51's acceptance asks the metrics to be correct while slots are
    // "admitted, grown and released". A small arena forces a growth: the long
    // prompt needs more cells than its slot's share, so the engine releases an
    // **idle** slot's run and grows the asking one — both of which are visible
    // in `sequences` and `reserved_cells`.
    let mut growing = BatchEngine::new(&*model, 2, 256).expect("engine");
    let gm = crate::server::metrics::ServerMetrics::new();
    growing.publish_metrics(&gm);
    let g0 = gm.snapshot();
    assert_eq!(g0.kv.sequences, 2, "two slots hold a reservation up front");
    assert_eq!(g0.kv.layers, 0, "no arena before the first forward");
    assert_eq!(g0.kv.reserved_cells, 256, "128 cells a slot");
    assert_eq!(g0.running, 0);

    let long: Vec<u32> = (1..=120).collect();
    let (ltx, mut lrx) = mpsc::channel::<StreamEvent>(4096);
    growing
        .submit_on(
            &*model,
            &tok,
            0,
            Job {
                input_ids: long.clone(),
                params: sampling_params(64),
                tx: ltx,
            },
        )
        .expect("long submit");
    growing.publish_metrics(&gm);
    let g1 = gm.snapshot();
    assert_eq!(g1.running, 1, "the grown request is running");
    assert_eq!(g1.kv.layers, model.n_layer() as u64);
    assert!(
        g1.kv.sequences < g0.kv.sequences,
        "growing past its share releases an idle slot's run ({} -> {})",
        g0.kv.sequences,
        g1.kv.sequences
    );
    assert!(
        g1.kv.owned_cells >= 120,
        "the long prefill wrote its prompt ({} cells)",
        g1.kv.owned_cells
    );

    // Finish it, then ask for work again: the released slot re-reserves, so
    // the gauge comes back — and `running` is 0 at both ends. Same work bound
    // as the warm request, sized for this run's prompt and token budget.
    let steps = drive_by_work(
        &mut growing,
        &*model,
        &tok,
        &mut lrx,
        step_budget(long.len(), 64),
        "the long request",
    );
    eprintln!("[f8] long request: {steps} step(s)");
    assert!(!growing.busy(), "the long request completed");
    growing.publish_metrics(&gm);
    let g2 = gm.snapshot();
    assert_eq!(g2.running, 0);
    assert!(g2.kv.owned_cells >= 120);
    assert!(
        g2.kv.sequences >= g1.kv.sequences,
        "a released slot's reservation is gone until it is used again"
    );

    let short: Vec<u32> = (1..=8).collect();
    let (stx, _srx) = mpsc::channel::<StreamEvent>(1024);
    growing
        .submit_on(
            &*model,
            &tok,
            1,
            Job {
                input_ids: short,
                params: sampling_params(2),
                tx: stx,
            },
        )
        .expect("short submit");
    growing.publish_metrics(&gm);
    let g3 = gm.snapshot();
    assert_eq!(g3.running, 1);
    assert!(
        g3.kv.sequences > g2.kv.sequences,
        "the second slot's reservation is back ({} -> {})",
        g2.kv.sequences,
        g3.kv.sequences
    );

    eprintln!(
        "[f8] metrics: layers {} rows {} region {} B owned {} cells idle_slots {} \
         reserved_classes {}",
        done.kv.layers,
        done.kv.rows,
        done.kv.region_bytes,
        done.kv.owned_cells,
        done.kv.idle_slots,
        done.kv.reserved_classes
    );
    eprintln!(
        "[f8] growth/release: sequences {} -> {} -> {} -> {}, reserved_cells {} -> {}, \
         owned {} -> {}",
        g0.kv.sequences,
        g1.kv.sequences,
        g2.kv.sequences,
        g3.kv.sequences,
        g0.kv.reserved_cells,
        g1.kv.reserved_cells,
        g1.kv.owned_cells,
        g2.kv.owned_cells
    );
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `batch.rs` (bucket B of the dead-code census — every
// test caller already lives in this module's subtree). The `cb.submit()` matches
// in `metal/tests.rs` are the Metal `CommandBuffer`, a different item.
// ────────────────────────────────────────────────────────────────────────────

impl BatchEngine {
    /// Slots without a request (their reservation and KV stay).
    ///
    /// Test-only (#239): driven by the `run_batched` helper and by
    /// `server::batch::tests::{a_failed_decode_forward_answers_a_single_run_and_releases_its_slot,
    /// the_stall_answers_every_live_run_exactly_once}`.
    pub fn idle_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.run.is_none()).count()
    }

    /// Admit a single request: [`BatchEngine::admit`] with one job.
    ///
    /// Test-only (#239): driven by `server::batch::tests::a_slot_snapshot_resumes_the_context_without_re_prefilling`
    /// and the chunked-prefill gates.
    pub fn submit(
        &mut self,
        model: &dyn ModelDef,
        tokenizer: &Tokenizer,
        job: Job,
    ) -> Result<usize, ApiError> {
        self.admit(model, tokenizer, vec![job])
            .into_iter()
            .next()
            .expect("one job in, one answer out")
    }

    /// E3: `(largest nt any prefill forward carried, prefill forwards run)`. The
    /// activation-memory bound is `max_nt`, so the gate asserts on this rather than
    /// on a claim about buffers.
    ///
    /// Test-only (#239): driven by
    /// `server::batch::tests::{a_chunked_prefill_answers_like_an_unchunked_one,
    /// a_long_prefill_keeps_another_slot_decoding}`.
    pub fn prefill_stats(&self) -> (usize, usize) {
        (self.prefill_max_nt, self.prefill_forwards)
    }

    /// B2/C5 S2: prompt tokens this engine has fed to prefills (see `prefill_fed`).
    ///
    /// Test-only (#239): driven by
    /// `server::batch::tests::a_slot_snapshot_resumes_the_context_without_re_prefilling`.
    pub fn prefill_fed(&self) -> usize {
        self.prefill_fed
    }

    /// E3: decode steps run between the chunks of a prefill (0 with chunking off).
    ///
    /// Test-only (#239): driven by
    /// `server::batch::tests::a_long_prefill_keeps_another_slot_decoding`.
    pub fn interleaved_ticks(&self) -> u64 {
        self.interleaved_ticks
    }
}
