//! Slot-level prefix sharing: a copied prefix, the copy-on-write store, and the cells a request plans.
//!
//! Split out of `src/server/batch/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
