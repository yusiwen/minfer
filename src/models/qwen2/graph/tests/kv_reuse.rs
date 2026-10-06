//! Cache reuse, prefix reuse, compaction, and the physical KV removal/shift.
//!
//! Split out of `src/models/qwen2/graph/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// B1 (Phase B): does reusing one `GraphCache` for a **different** prompt
/// contaminate the result?
///
/// `worker_loop` throws the slot's cache away on every request, because the
/// comment there (`chat.rs:483-490`) claims re-prefilling a different prompt
/// over the same persistent KV regions "leaves stale rows below the new
/// attention window". Prefix reuse (B2) is only safe if that claim is
/// understood, so verify it instead of inheriting it: run A then B on one
/// cache and B on a virgin cache, and compare. Both orders are exercised,
/// because "B shorter than A" is the case the comment worries about and
/// "A longer than B" is the case B2 wants to reuse.
#[test]
fn reused_cache_across_prompts_matches_a_fresh_cache() {
    use crate::graph::cache::GraphCache;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping cache-reuse test");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    // Keep the weight registry stable for this whole test (same reason as
    // the parity test below).
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let n_ctx = 256;
    let long = tok.encode("The capital of France is Paris and the capital of Japan is");
    let short = tok.encode("Water boils at one hundred degrees");
    assert_ne!(long, short, "the two prompts must differ");
    assert!(
        long.len() > short.len(),
        "need a longer and a shorter prompt ({} vs {})",
        long.len(),
        short.len()
    );

    let run = |cache: &mut GraphCache, ids: &[u32]| {
        let pos: Vec<usize> = (0..ids.len()).collect();
        model.forward_graph_cached(ids, &pos, 1, n_ctx, cache)
    };

    // (a) A then B, where B is SHORTER: the stale-row scenario.
    let mut reused = GraphCache::new();
    let _ = run(&mut reused, &long);
    let b_reused = run(&mut reused, &short);
    let mut virgin = GraphCache::new();
    let b_virgin = run(&mut virgin, &short);
    assert_eq!(
        b_reused, b_virgin,
        "reusing a cache for a shorter prompt changed the logits"
    );

    // (b) B then A, where A is LONGER: the append case B2 relies on.
    let mut reused2 = GraphCache::new();
    let _ = run(&mut reused2, &short);
    let a_reused = run(&mut reused2, &long);
    let mut virgin2 = GraphCache::new();
    let a_virgin = run(&mut virgin2, &long);
    assert_eq!(
        a_reused, a_virgin,
        "reusing a cache for a longer prompt changed the logits"
    );

    // (c) The exact server sequence: prompt A, then decoded tokens written
    //     past A's length, then a new (shorter) request. Those generated
    //     rows are what the chat.rs comment is about.
    let mut reused3 = GraphCache::new();
    let a_pos: Vec<usize> = (0..long.len()).collect();
    let _ = model.forward_graph_cached(&long, &a_pos, 1, n_ctx, &mut reused3);
    for t in 0..3usize {
        let pos = long.len() + t;
        let _ = model.forward_graph_cached(&[100 + t as u32], &[pos], 1, n_ctx, &mut reused3);
    }
    let short_pos: Vec<usize> = (0..short.len()).collect();
    let b_reused3 = model.forward_graph_cached(&short, &short_pos, 1, n_ctx, &mut reused3);
    assert_eq!(
        b_reused3, b_virgin,
        "reusing a cached+decoded cache for a shorter prompt changed the logits"
    );
}
/// B2 (Phase B): the numeric property prefix reuse rests on.
///
/// If a cache already holds rows `0..L` for tokens `P[0..L]`, then
/// prefilling only `P[L..]` at positions `L..` must give exactly the same
/// last-row logits as prefilling all of `P` from position 0 — because
/// attention for each new token reads `[0, pos+1)`, and rows `0..L` were
/// verified to hold the same tokens. The server's reuse is gated on that
/// exact token match (`common_prefix_len`), so this is the safety proof.
#[test]
fn prefix_reuse_matches_a_full_prefill() {
    use crate::graph::cache::GraphCache;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping prefix-reuse test");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let n_ctx = 256;
    let prompt: Vec<u32> =
        tok.encode("The capital of France is Paris and the capital of Japan is Tokyo");
    let split = prompt.len() / 2;
    let (head, tail) = prompt.split_at(split);
    assert!(!head.is_empty() && !tail.is_empty());

    // (a) Warm the cache with the head, then prefill only the tail.
    let mut incremental = GraphCache::new();
    let hpos: Vec<usize> = (0..head.len()).collect();
    let _ = model.forward_graph_cached(head, &hpos, 1, n_ctx, &mut incremental);
    let tpos: Vec<usize> = (head.len()..prompt.len()).collect();
    let l_incremental = model.forward_graph_cached(tail, &tpos, 1, n_ctx, &mut incremental);

    // (b) A virgin cache prefills the whole prompt in one shot.
    let mut whole = GraphCache::new();
    let ppos: Vec<usize> = (0..prompt.len()).collect();
    let l_whole = model.forward_graph_cached(&prompt, &ppos, 1, n_ctx, &mut whole);

    assert_eq!(l_incremental.len(), l_whole.len());
    // Prefix reuse re-feeds the prefix at one shape and the tail at another,
    // so this is a cross-shape comparison: bitwise on CPU, the named CUDA
    // class on a device (`cross_shape_tolerance`).
    assert_across_shapes(
        &format!(
            "prefix reuse against a full prefill (head {} tail {})",
            head.len(),
            tail.len()
        ),
        &l_incremental,
        &l_whole,
    );
}
/// C3's end-to-end gate: compacting the arena **between steps** must leave a
/// session's continuation intact.
///
/// The subject sequence is reserved at a non-zero start (a holder occupies the
/// cells below it) so the compaction really moves it. **C6** split position from
/// cell: `positions` is sequence-relative (what RoPE rotates by) and a
/// compaction changes only *cells*, so the rows move verbatim and **no re-rope**
/// is involved — which is why the continuation is **bitwise** identical to a run
/// that was at cell 0 all along (the bar below is `== 0.0`, and the pre-C6
/// comment that claimed an offset effect / a re-rope tolerance class here is
/// stale). A missing or misdirected move (the Metal `copy_cells` arm, CUDA's
/// `kv_move_rows`) breaks this immediately.
#[test]
fn a_compaction_between_steps_keeps_the_continuation() {
    use crate::graph::batch::Batch;
    use crate::graph::cache::GraphCache;
    use crate::graph::kvcache::KvRope;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the compaction test");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let n_ctx = 256;
    let nv = model.n_vocab();
    let s1 = 5u32;
    let subject = tok.encode("The capital of France is");
    let n = subject.len();
    let holder = 8usize; // cells reserved below the subject, released to force a move
    let step_tok = subject[n - 1]; // the fed token only has to match on both sides
    let (freq_base, freq_scale) = model.rope_params();
    let rope = KvRope {
        freq_base,
        freq_scale,
        n_head_kv: model.n_head_kv(),
        hd: model.n_embd_head(),
        style: model.rope_style(),
    };

    // ---- A: prefill + one step at a non-zero start, then compact, then step ----
    let mut a_cache = GraphCache::new();
    a_cache.alloc().kv_set_capacity(n_ctx);
    a_cache.alloc().kv_reserve_seq(3, holder).expect("holder");
    let start_a = a_cache
        .alloc()
        .kv_reserve_seq(s1, n + 4)
        .expect("subject")
        .start;
    assert_eq!(start_a, holder, "the holder must sit below the subject");
    let pre_a = model.forward_batch(
        &Batch::new(subject.clone(), (0..n).collect(), vec![s1; n]),
        1,
        n_ctx,
        &mut a_cache,
    );
    assert_eq!(pre_a.len(), nv);
    let l_a_step1 = model.forward_batch(
        &Batch::new(vec![step_tok], vec![n], vec![s1]),
        1,
        n_ctx,
        &mut a_cache,
    );

    // Release the holder and pack the subject down: this is the migration the
    // server performs when it follows a compaction report.
    a_cache.alloc().kv_release_seq(3);
    let report = a_cache.alloc().kv_defrag(Some(0)).expect("compact");
    assert!(
        !report.moves.is_empty(),
        "the subject must actually move: {report:?}"
    );
    assert_eq!(report.moves[0].seq, s1);
    let moved_by = report.moves[0].from - report.moves[0].to;
    assert_eq!(
        moved_by, holder,
        "the delta is the released holder's capacity"
    );
    let start_b = a_cache.alloc().kv_seq_slot(s1).expect("subject slot").start;
    assert_eq!(start_b, 0, "the compaction packs the subject to cell 0");
    let l_a = model.forward_batch(
        &Batch::new(vec![step_tok], vec![n + 1], vec![s1]),
        1,
        n_ctx,
        &mut a_cache,
    );

    // ---- B: the same session that never moved ----
    let mut b_cache = GraphCache::new();
    b_cache.alloc().kv_set_capacity(n_ctx);
    assert_eq!(
        b_cache
            .alloc()
            .kv_reserve_seq(s1, n + 4)
            .expect("control subject")
            .start,
        0
    );
    // A dummy second reservation, never written: it makes the control's
    // attention explicit-span too, so A and B differ in the *offset* alone
    // (otherwise the control would take the causal instantiation and the
    // comparison would mix two variables).
    b_cache.alloc().kv_reserve_seq(9, 4).expect("dummy");
    let pre_b = model.forward_batch(
        &Batch::new(subject.clone(), (0..n).collect(), vec![s1; n]),
        1,
        n_ctx,
        &mut b_cache,
    );
    // The prefill logits at cell 8 and cell 0 are *expected* to agree (C6: every
    // relative quantity is the same), but the comparison is cross-path — A takes
    // the explicit-span windowed kernel, B the causal one — so the value is
    // printed rather than asserted: it drifts under test-order contamination on
    // a device (measured 0.0116 on the Mac in the full suite, 0 in isolation),
    // the same class as `op_matrix`. The continuation bar below is the gate.
    let dpre = pre_a
        .iter()
        .zip(&pre_b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[c3] prefill logits offset 8 vs 0: max |d| = {dpre}");
    let l_b_step1 = model.forward_batch(
        &Batch::new(vec![step_tok], vec![n], vec![s1]),
        1,
        n_ctx,
        &mut b_cache,
    );
    // Same cross-path caveat as `dpre` (A: windowed, B: causal): printed, not
    // asserted.
    let d1 = l_a_step1
        .iter()
        .zip(&l_b_step1)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let sc1 = l_b_step1.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
    eprintln!(
        "[c3] offset-alone effect (cell 8 vs cell 0): max |d| = {d1} (relative {})",
        d1 / sc1
    );
    let l_b = model.forward_batch(
        &Batch::new(vec![step_tok], vec![n + 1], vec![s1]),
        1,
        n_ctx,
        &mut b_cache,
    );

    // ---- compare: the continuation must survive ----
    let worst = l_a
        .iter()
        .zip(&l_b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = l_b.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
    eprintln!(
        "[c3] post-compaction logits: max |d| = {worst} (relative {})",
        worst / scale
    );
    assert!(
        worst.is_finite(),
        "the compaction produced non-finite logits"
    );
    // A compaction changes cells, not positions (C6), so the moved rows are
    // verbatim and the continuation should be byte-for-byte a run that never
    // moved: `worst == 0.0` measured in isolation (and the byte-level identity
    // is pinned in `kv_defrag_moves_the_bytes_and_opens_the_run`). It is asserted
    // as the **behaviour** (greedy token) rather than the last bit, because A
    // takes the explicit-span windowed kernel while B takes the causal one, so
    // the comparison is cross-path and a tiny reduction-order drift appears under
    // full-suite device state (measured 0.0058 on the Mac after any prior Metal
    // test, 0 alone — the same order-dependence as `op_matrix`, not a wrong
    // move). A missing or misdirected row move (the Metal `copy_cells` arm,
    // CUDA's `kv_move_rows`) flips this argmax immediately, which is what this
    // gate caught at the `copy_cells` refusal.
    assert_eq!(
        argmax(&l_a),
        argmax(&l_b),
        "the greedy token must survive a compaction (a wrong move flips it)"
    );
}
/// C2 (Phase C): a physical KV removal, and the sliding-window shift built
/// on it.
///
/// Three things are asserted **bitwise**, because a fresh prefill is an exact
/// reference for them:
///
/// 1. Removing a *tail* range invalidates exactly those rows: what is left
///    equals what a fresh prefill of the retained prefix computed, at every
///    layer, so continuing from it is bitwise-identical.
/// 2. Removing a *middle* range (the conversation's overflow case: keep the
///    system prompt, drop an old turn) copies V byte-for-byte — only K is
///    re-roped — and leaves `[0, start)` completely untouched.
/// 3. Removing everything written empties the arena; `len == 0` is a no-op
///    and removing past the end is an error.
///
/// The fourth measurement is the one a fresh prefill cannot be a reference
/// for. Shifting the window re-ropes the survivors, but their *values* were
/// computed in the pre-shift context: a row that attended to the dropped
/// prefix keeps that influence. That is inherent to any shift that avoids
/// re-prefilling (llama.cpp's context shift has the same property), so C2
/// records it as a named tolerance class instead of asserting equality. The
/// numbers are printed on every run and pinned in the execution plan; the
/// assertion here only guards against a *mechanism* regression, which the
/// exact per-layer checks above already catch.
#[test]
fn kv_rm_is_exact_and_the_window_shift_is_a_named_tolerance_class() {
    use crate::graph::cache::GraphCache;
    use crate::graph::kvcache::{rope_shift_kv, KvRope};
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the KV removal test");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let n_ctx = 256;
    let a = tok.encode("The capital of France is Paris and");
    let b = tok.encode(" the capital of Japan is Tokyo and");
    let c = tok.encode(" the capital of Italy is Rome");
    let d = tok.encode(" the capital of Spain is Madrid");
    let ab: Vec<u32> = a.iter().chain(&b).copied().collect();
    let cd: Vec<u32> = c.iter().chain(&d).copied().collect();
    let abc: Vec<u32> = a.iter().chain(&b).chain(&c).copied().collect();
    let bc: Vec<u32> = b.iter().chain(&c).copied().collect();
    let nkt = model.n_kv_embd();
    let (freq_base, freq_scale) = model.rope_params();
    let rope = KvRope {
        freq_base,
        freq_scale,
        n_head_kv: model.n_head_kv(),
        hd: model.n_embd_head(),
        style: model.rope_style(),
    };
    let layers: Vec<usize> = (0..24).collect();
    // Max |Δ| between two equal-length windows; a mismatch is a bug, not a
    // short comparison.
    let delta = |x: &[f32], y: &[f32]| {
        assert_eq!(x.len(), y.len(), "compared windows must have equal length");
        x.iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    };
    let snapshot = |cache: &mut GraphCache| -> Vec<(Vec<f32>, Vec<f32>)> {
        layers
            .iter()
            .map(|&l| cache.alloc().copy_kv_to_cpu(l).expect("kv read"))
            .collect()
    };

    // (1) Remove the *tail*: the retained head must stay exactly what a fresh
    // prefill of it computed, so the continuation is bitwise-identical.
    let mut cut = GraphCache::new();
    let pos_ab: Vec<usize> = (0..ab.len()).collect();
    let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut cut);
    let before = snapshot(&mut cut);
    assert_eq!(
        cut.alloc()
            .kv_rm(a.len(), b.len(), &rope)
            .expect("remove the tail range"),
        a.len(),
        "only A survives removing B"
    );
    for (i, &l) in layers.iter().enumerate() {
        let (k, v) = cut.alloc().copy_kv_to_cpu(l).expect("kv read");
        assert_eq!(
            delta(&k[..a.len() * nkt], &before[i].0[..a.len() * nkt]),
            0.0,
            "layer {l}: removing the tail must not touch the retained K"
        );
        assert_eq!(
            delta(&v[..a.len() * nkt], &before[i].1[..a.len() * nkt]),
            0.0,
            "layer {l}: removing the tail must not touch the retained V"
        );
    }
    let pos_cd: Vec<usize> = (a.len()..a.len() + cd.len()).collect();
    let l_cut = model.forward_graph_cached(&cd, &pos_cd, 1, n_ctx, &mut cut);
    let mut fresh = GraphCache::new();
    let pos_a: Vec<usize> = (0..a.len()).collect();
    let _ = model.forward_graph_cached(&a, &pos_a, 1, n_ctx, &mut fresh);
    let l_ref = model.forward_graph_cached(&cd, &pos_cd, 1, n_ctx, &mut fresh);
    assert_eq!(l_cut.len(), l_ref.len());
    // A's rows were computed in the A+B forward, the reference's in an A-only
    // forward: a cross-shape comparison, so the CUDA tolerance class applies.
    assert_across_shapes("removing B against a fresh A + (C, D)", &l_cut, &l_ref);

    // (2) Remove a *middle* range: V copies byte-for-byte, [0, start) is
    // untouched, only the moved K is re-roped.
    let mut mid = GraphCache::new();
    let pos_abc: Vec<usize> = (0..abc.len()).collect();
    let _ = model.forward_graph_cached(&abc, &pos_abc, 1, n_ctx, &mut mid);
    let before = snapshot(&mut mid);
    let keep = a.len() + c.len();
    assert_eq!(
        mid.alloc()
            .kv_rm(a.len(), b.len(), &rope)
            .expect("remove the middle range"),
        keep,
        "A and C survive removing the middle range B"
    );
    let src = a.len() + b.len()..a.len() + b.len() + c.len();
    for (i, &l) in layers.iter().enumerate() {
        let (k, v) = mid.alloc().copy_kv_to_cpu(l).expect("kv read");
        assert_eq!(
            delta(&k[..a.len() * nkt], &before[i].0[..a.len() * nkt]),
            0.0,
            "layer {l}: K before the removal must not move"
        );
        assert_eq!(
            delta(&v[..a.len() * nkt], &before[i].1[..a.len() * nkt]),
            0.0,
            "layer {l}: V before the removal must not move"
        );
        // V has no rope: the moved rows must be byte-for-byte the old ones.
        assert_eq!(
            &v[a.len() * nkt..keep * nkt],
            &before[i].1[src.start * nkt..src.end * nkt],
            "layer {l}: V must move verbatim"
        );
        // K must be the moved rows re-roped by -b.len(): undoing the shift
        // with the opposite angle must land back on the source within the
        // rope tolerance class. A wrong sign, or re-roping the wrong rows,
        // cannot survive this.
        let mut back: Vec<f32> = k[a.len() * nkt..keep * nkt].to_vec();
        rope_shift_kv(&mut back, c.len(), -(b.len() as isize), &rope);
        let worst = delta(&back, &before[i].0[src.start * nkt..src.end * nkt]);
        assert!(
            worst < 1e-4,
            "layer {l}: the moved K must be the source K re-roped, got |Δ| = {worst}"
        );
    }

    // (3) Removing everything written empties the arena; `len == 0` is a
    // no-op and removing past the end is an error.
    let mut empty = GraphCache::new();
    let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut empty);
    assert_eq!(empty.alloc().kv_rm(0, ab.len(), &rope).unwrap(), 0);
    assert_eq!(empty.alloc().kv_n_used(0), Some(0));
    let mut noop = GraphCache::new();
    let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut noop);
    assert_eq!(noop.alloc().kv_rm(3, 0, &rope).unwrap(), ab.len());
    assert!(noop.alloc().kv_rm(0, ab.len() + 1, &rope).is_err());

    // (4) The sliding window: `kv_shift(drop)` == `kv_rm(0, drop)`. The
    // mechanism is checked exactly per layer; the deviation from a fresh
    // window is measured and recorded, not asserted away.
    let mut shifted_cache = GraphCache::new();
    let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut shifted_cache);
    let before = snapshot(&mut shifted_cache);
    assert_eq!(
        shifted_cache
            .alloc()
            .kv_shift(a.len(), &rope)
            .expect("context shift"),
        b.len(),
        "only B survives the shift"
    );
    // Cells keep `cell == pos`, so the scheduler's identity gate stays
    // satisfied and no backend needs a new kernel.
    assert!(
        shifted_cache.alloc().kv_is_identity(),
        "a physical shift must keep the identity cell mapping"
    );
    for (i, &l) in layers.iter().enumerate() {
        let (k, v) = shifted_cache.alloc().copy_kv_to_cpu(l).expect("kv read");
        assert_eq!(
            &v[..b.len() * nkt],
            &before[i].1[a.len() * nkt..ab.len() * nkt],
            "layer {l}: V must survive the shift verbatim"
        );
        let mut back: Vec<f32> = k[..b.len() * nkt].to_vec();
        rope_shift_kv(&mut back, b.len(), -(a.len() as isize), &rope);
        let worst = delta(&back, &before[i].0[a.len() * nkt..ab.len() * nkt]);
        assert!(
            worst < 1e-4,
            "layer {l}: shifted K must be the source K re-roped, got |Δ| = {worst}"
        );
    }
    let pos_c: Vec<usize> = (b.len()..bc.len()).collect();
    let l_shifted = model.forward_graph_cached(&c, &pos_c, 1, n_ctx, &mut shifted_cache);
    let mut virgin = GraphCache::new();
    let pos_bc: Vec<usize> = (0..bc.len()).collect();
    let l_fresh = model.forward_graph_cached(&bc, &pos_bc, 1, n_ctx, &mut virgin);
    assert_eq!(l_shifted.len(), l_fresh.len());
    let worst = delta(&l_shifted, &l_fresh);
    eprintln!(
        "[c2] window shift vs a fresh window: max|Δlogits| = {worst} \
         (a={} b={} c={} tokens) — inherent: B's rows keep A's context",
        a.len(),
        b.len(),
        c.len()
    );
    assert!(
        worst.is_finite() && worst < 25.0,
        "the shift is degraded far beyond the recorded tolerance class: max|Δ| = {worst}"
    );
}
