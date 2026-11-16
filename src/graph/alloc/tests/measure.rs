//! #828: the allocation ledger and the defrag reclaim, exported as measurements.
//!
//! These two need more than a struct: the defrag reclaim must move **real region bytes**
//! (so it drives `GraphAllocator::kv_defrag`, which copies through `Backend::copy_cells`),
//! and the ledger is `GraphAllocator::memory_report` itself. Both stay on the **CPU**
//! pool, so neither needs a device: the CPU pool is host memory, and `memory_report(CPU)`
//! is where the engine's own weights + pool + liveness accounting is observable.
//!
//! Output protocol (#828): one line per measurement group, a JSON object after the
//! `measure: ` prefix. `cargo test --release measure -- --nocapture` prints them.
//!
//! | group | needs | what it answers |
//! |---|---|---|
//! | `kv_defrag` | nothing | fragmentation, the move plan, and that the bytes survived |
//! | `alloc_ledger` | `MINFER_MEASURE_MODEL` | weights/pool/live/peak for prefill and decode |
//!
//! **No environment mutation.** The model path is only *read*; the KV format is a
//! parameter (`CParams::kv_format` + `GraphAllocator::set_kv_format`), never
//! `MINFER_CACHE_TYPE`, because the environment is process-wide and tests run on parallel
//! threads (C4/#99 removed the process global for exactly this reason).

use super::*;

use crate::graph::backend::{Backend as BackendTrait, KvProvider};
use crate::graph::fusion::FusionPass;
use crate::graph::kvcache::SeqId;
use crate::graph::kvformat::KvFormat;
use crate::graph::params::{CParams, GraphParams, GraphType};
use crate::graph::scheduler::BackendScheduler;
use crate::models::ModelDef;
use serde_json::json;

/// One measurement line (see the protocol in the module docs).
fn emit(group: &str, value: serde_json::Value) {
    let mut obj = serde_json::Map::new();
    obj.insert(group.to_string(), value);
    println!("measure: {}", serde_json::Value::Object(obj));
}

/// A deterministic, cell-dependent byte pattern: cell `c`, element `e` holds
/// `base + c + e/10`, so a row that moved without its bytes (or moved twice) cannot pass
/// the checksum by accident.
fn pattern(base: f32, cells: usize, row: usize) -> Vec<f32> {
    (0..cells * row)
        .map(|i| base + (i / row) as f32 + (i % row) as f32 / 10.0)
        .collect()
}

/// A cheap, order-sensitive checksum of one cell row: K and V, element by element.
fn row_checksum(k: &[f32], v: &[f32], cell: usize, row: usize) -> u64 {
    let mut h: u64 = 0;
    for e in 0..row {
        h = h
            .wrapping_mul(31)
            .wrapping_add(k[cell * row + e].to_bits() as u64);
        h = h
            .wrapping_mul(31)
            .wrapping_add(v[cell * row + e].to_bits() as u64);
    }
    h
}

// ────────────────────────────────────────────────────────────────────────────
// defrag reclaim - chapter 18 (C3)
// ────────────────────────────────────────────────────────────────────────────

/// Three sequences interleaved over one arena, the middle one released, then a real
/// compaction: the free-run view before, the moves it planned, the free-run view after,
/// and a checksum of every written row on both sides.
///
/// `KvCache::compaction_plan` + `apply_moves` are pure, but the *claim* the book wants -
/// "a defrag moves bytes, never content" - is only true of the allocator's copy, so this
/// drives `GraphAllocator::kv_defrag` and reads the regions back through `kv_pair`.
#[test]
fn kv_defrag_reclaim() {
    const N_CTX: usize = 16;
    const ROW: usize = 4; // elements per cell (n_kv_embd)
    const RUN: usize = 4; // cells per sequence

    let mut b = GraphBuilder::new();
    let _positions = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, N_CTX);
    let load = b.kvcache_load(0, ROW, N_CTX, 1);
    b.output(load);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g).expect("allocate the one-layer arena");
    for seq in 1u32..=3 {
        let start = (seq as usize - 1) * RUN;
        let slot = alloc.kv_reserve_seq(seq, RUN).expect("first-fit places it");
        assert_eq!(slot.start, start, "three runs, bottom up");
        alloc.kv.own_range(seq, start, start + RUN);
    }
    let (k_id, v_id) = alloc.kv_pair(0).expect("one KV layer");
    alloc
        .write_pool(Backend::CPU, k_id, &pattern(0.0, N_CTX, ROW))
        .expect("write K");
    alloc
        .write_pool(Backend::CPU, v_id, &pattern(100.0, N_CTX, ROW))
        .expect("write V");

    // Every written row of the two survivors, by sequence position and by cell: the
    // cell is what moves, the checksum is what must not.
    fn written_rows(alloc: &mut GraphAllocator, seq: SeqId, written: usize) -> Vec<(usize, u64)> {
        let cells: Vec<usize> = (0..written)
            .map(|pos| {
                alloc
                    .kv
                    .cell_of(seq, pos)
                    .unwrap_or_else(|| panic!("sequence {seq} position {pos} does not resolve"))
            })
            .collect();
        let (k, v) = alloc
            .copy_kv_to_cpu(0)
            .expect("the CPU pool lends the region");
        cells
            .into_iter()
            .map(|cell| (cell, row_checksum(&k, &v, cell, ROW)))
            .collect()
    }
    let checksums = |rows: &[Vec<(usize, u64)>]| -> Vec<Vec<u64>> {
        rows.iter()
            .map(|run| run.iter().map(|&(_, sum)| sum).collect())
            .collect()
    };

    // Fragment on purpose: release the middle run, leaving [4, 8) and [12, 16) free.
    alloc.kv_release_seq(2);
    let before = alloc.kv_arena_stats();
    assert_eq!(
        (before.free_cells, before.free_runs, before.largest_free_run),
        (2 * RUN, 2, RUN),
        "8 free cells in two runs: capacity is not the constraint"
    );
    assert_eq!(
        (before.owned_cells, before.sequences),
        (2 * RUN, 2),
        "the released run's stamps went with it; two sequences remain"
    );
    assert!(
        alloc.kv_reserve_seq(4, 2 * RUN).is_err(),
        "first-fit refuses the 8-cell run the arena actually holds"
    );
    let survivors = [1u32, 3];
    let rows_before: Vec<Vec<(usize, u64)>> = survivors
        .iter()
        .map(|&seq| written_rows(&mut alloc, seq, RUN))
        .collect();
    assert_eq!(
        rows_before[1][0].0,
        2 * RUN,
        "before the compaction, sequence 3 starts at cell 8"
    );

    let report = alloc.kv_defrag(None).expect("compact the arena");
    let after = alloc.kv_arena_stats();
    let rows_after: Vec<Vec<(usize, u64)>> = survivors
        .iter()
        .map(|&seq| written_rows(&mut alloc, seq, RUN))
        .collect();

    // The move: run 3 slides down into the hole, so the free space coalesces at the top.
    assert_eq!(report.moves.len(), 1, "{report:?}");
    assert_eq!((report.moves[0].seq, report.moves[0].from), (3, 2 * RUN));
    assert_eq!((report.moves[0].to, report.moves[0].rows), (RUN, RUN));
    assert_eq!(report.rows_moved, RUN, "one written run moved, K and V");
    assert_eq!(
        rows_after[1][0].0, RUN,
        "sequence 3 now starts at cell 4 - the content check is not vacuous"
    );
    assert!(
        after.largest_free_run > before.largest_free_run,
        "the point of the compaction: {} -> {} contiguous free cells",
        before.largest_free_run,
        after.largest_free_run
    );
    assert_eq!((after.free_runs, after.largest_free_run), (1, 2 * RUN));
    assert_eq!(
        (after.owned_cells, after.sequences),
        (2 * RUN, 2),
        "the move renumbers cells; the written rows and the run table do not change size"
    );
    assert_eq!(
        (after.defrags, after.cells_moved),
        (before.defrags + 1, before.cells_moved + RUN as u64),
        "the counters C3's acceptance reads"
    );
    // The claim the book makes: the bytes moved with the rows, so no row changed value.
    assert_eq!(
        checksums(&rows_before),
        checksums(&rows_after),
        "a defrag moves bytes, never content (checksums per sequence per position)"
    );
    // Every sequence still resolves every written position in its *new* place.
    for &seq in &survivors {
        for pos in 0..RUN {
            assert!(
                alloc.kv.cell_of(seq, pos).is_some(),
                "sequence {seq} position {pos} must survive the compaction"
            );
        }
    }
    // And the reservation first-fit refused now fits in the opened tail.
    assert_eq!(
        alloc.kv_reserve_seq(4, 2 * RUN).expect("the tail").start,
        2 * RUN
    );

    emit(
        "kv_defrag",
        json!({
            "n_ctx": N_CTX,
            "n_embd": ROW,
            "layers": 1,
            "run_cells": RUN,
            "before": {
                "free_cells": before.free_cells,
                "free_runs": before.free_runs,
                "largest_free_run": before.largest_free_run,
                "sequences": before.sequences,
                "owned_cells": before.owned_cells,
            },
            "moves": report.moves.iter().map(|m| json!({
                "seq": m.seq, "from": m.from, "to": m.to, "rows": m.rows,
            })).collect::<Vec<_>>(),
            "rows_moved": report.rows_moved,
            "after": {
                "free_cells": after.free_cells,
                "free_runs": after.free_runs,
                "largest_free_run": after.largest_free_run,
                "defrags": after.defrags,
                "cells_moved": after.cells_moved,
            },
            "rows": survivors.iter().zip(rows_before.iter().zip(rows_after.iter()))
                .map(|(&seq, (b, a))| json!({
                    "seq": seq,
                    "cells_before": b.iter().map(|&(cell, _)| cell).collect::<Vec<_>>(),
                    "cells_after": a.iter().map(|&(cell, _)| cell).collect::<Vec<_>>(),
                    "checksums": b.iter().map(|&(_, sum)| sum).collect::<Vec<_>>(),
                    "checksums_unchanged": checksums(std::slice::from_ref(b)) == checksums(std::slice::from_ref(a)),
                }))
                .collect::<Vec<_>>(),
        }),
    );
}

// ────────────────────────────────────────────────────────────────────────────
// the allocator ledger - chapter 17 (E4)
// ────────────────────────────────────────────────────────────────────────────

/// Build a graph the way the CLI does — build, assign, fuse, allocate — so the ledger
/// describes a graph a run would actually execute, not a bare node vector.
fn build_and_allocate(
    model: &dyn ModelDef,
    params: &GraphParams,
    alloc: &mut GraphAllocator,
) -> ComputeGraph {
    let sched = BackendScheduler::new();
    let mut graph = model.build_graph(params);
    sched.assign_backends(&mut graph, alloc);
    {
        let backends: [&dyn BackendTrait; 1] = [alloc.cpu()];
        FusionPass::new().run(&mut graph, &backends, &|_, _| Some(0));
    }
    alloc
        .alloc_graph(&graph)
        .expect("the graph fits the CPU pool");
    graph
}

/// `memory_report(Backend::CPU)` for a prefill graph and, after the rebuild on the same
/// allocator, the decode graph: weights, pool (high-water), live and peak live, the idle
/// reservation table, and the persistent KV region the ledger does **not** include.
///
/// The book currently predicts peak memory with its own liveness model and observes the
/// kernel's RSS. This replaces both with the engine's own ledger.
///
/// `MINFER_MEASURE_MODEL` is only *read*. A test that needs a model returns early
/// when it is unset (not `#[ignore]`, so CI stays green and nothing silently stops
/// running) and prints a `skipped` line instead.
#[test]
fn alloc_ledger_prefill_and_decode() {
    let Some(path) = std::env::var("MINFER_MEASURE_MODEL")
        .ok()
        .filter(|p| !p.trim().is_empty())
    else {
        println!(
            "measure: {}",
            json!({
                "skipped": "alloc_ledger_prefill_and_decode needs MINFER_MEASURE_MODEL=<a GGUF path>",
            })
        );
        return;
    };

    // Keep the weight registry stable for the whole measurement: a parallel test loading
    // another architecture swaps same-named entries and would flip the device decision
    // mid-run.
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();

    let gguf = crate::gguf::load_gguf_model(std::path::Path::new(&path))
        .unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let model = crate::models::load_model(&gguf).unwrap_or_else(|e| panic!("load {path}: {e}"));

    let mut alloc = GraphAllocator::new();
    // The CPU ledger, and the CPU's own default format: the model's resolved format may be
    // a *device* policy (f16 auto-selected for the 7B class), which the CPU does not read.
    alloc.set_kv_format(KvFormat::F32);
    if let Some(q2) = model
        .as_any()
        .downcast_ref::<crate::models::qwen2::Qwen2Model>()
    {
        crate::models::qwen2::graph::Qwen2Graph::register_graph_weights(q2, &mut alloc);
    } else if let Some(q3) = model
        .as_any()
        .downcast_ref::<crate::models::qwen3::Qwen3Model>()
    {
        crate::models::qwen3::graph::Qwen3Graph::register_graph_weights(q3, &mut alloc);
    } else {
        panic!(
            "MINFER_MEASURE_MODEL={path} is not a supported architecture for this measurement \
             (qwen2 or qwen3): the ledger needs the model's weights registered on the CPU pool"
        );
    }

    let n_ctx = model.max_seq_len().min(4096);
    let n_tokens = 34.min(n_ctx);
    // The shape the CLI runs: one sequence, CPU graph, no fusion (both fusions are
    // device-only paths) and the CPU's KV format.
    let cparams = CParams {
        n_ctx,
        flash_attn: false,
        gpu: false,
        fuse_qkv: false,
        fuse_ffn: false,
        explicit_span: false,
        kv_map: false,
        gpu_layers: 0,
        kv_format: KvFormat::F32,
    };
    let params = |n_tokens: usize, gtype: GraphType| GraphParams {
        n_tokens,
        n_out: 1,
        gtype,
        cparams,
        weights_version: 1,
    };

    // ---- the session: one allocator, prefill then decode ----
    let prefill = build_and_allocate(
        model.as_ref(),
        &params(n_tokens, GraphType::Prefill),
        &mut alloc,
    );
    let session_prefill = alloc.memory_report(Backend::CPU);
    let prefill_nodes = prefill.nodes.len();

    // Same allocator, so the persistent KV regions survive the rebuild — the point of the
    // second measurement.
    let decode = build_and_allocate(model.as_ref(), &params(1, GraphType::Decode), &mut alloc);
    let session_decode = alloc.memory_report(Backend::CPU);
    let decode_nodes = decode.nodes.len();

    // E4's ordering: nothing is live that the pool does not hold, and the peak is at least
    // the live set. `pool_bytes` is a **high-water mark**: a rebuild re-maps onto reserved
    // slots, so it can only stay or grow.
    for (name, r) in [("prefill", session_prefill), ("decode", session_decode)] {
        assert!(
            r.weights_bytes > 0,
            "{name}: the model's weights must be registered on the CPU pool: {r:?}"
        );
        assert!(
            r.pool_bytes >= r.peak_live_bytes,
            "{name}: the pool holds the peak live set: {r:?}"
        );
        assert!(
            r.peak_live_bytes >= r.live_bytes,
            "{name}: the peak is at least the current live set: {r:?}"
        );
    }

    // ---- the two graphs, each on its own allocator ----
    //
    // `peak_live_bytes` is an allocator-wide high-water mark, not a per-graph number: after
    // the rebuild the report still carries the prefill's peak, because nothing resets it.
    // A fresh allocator per graph is what isolates one graph's peak. Weights are not
    // registered on these two (they are handles into the GGUF, and registration does not
    // change activation liveness), so the numbers here are activations only.
    let isolated = |n_tokens: usize, gtype: GraphType| {
        let mut fresh = GraphAllocator::new();
        fresh.set_kv_format(KvFormat::F32);
        let graph = build_and_allocate(model.as_ref(), &params(n_tokens, gtype), &mut fresh);
        (fresh.memory_report(Backend::CPU), graph.nodes.len())
    };
    let (prefill_only, prefill_only_nodes) = isolated(n_tokens, GraphType::Prefill);
    let (decode_only, decode_only_nodes) = isolated(1, GraphType::Decode);

    // What the two allocator kinds must, and must not, share: the weights are a property
    // of the session allocator (the isolated ones register none), and the graph sizes are
    // real node vectors, not an empty build.
    assert_eq!(
        session_prefill.weights_bytes, session_decode.weights_bytes,
        "both session reports describe the same registered weights"
    );
    assert_eq!(
        (prefill_only.weights_bytes, decode_only.weights_bytes),
        (0, 0),
        "the isolated allocators register no weights, so their numbers are activations"
    );
    assert!(
        prefill_nodes > 0 && decode_nodes > 0 && prefill_only_nodes > 0 && decode_only_nodes > 0,
        "a measured graph has nodes: session {prefill_nodes}/{decode_nodes}, isolated \
         {prefill_only_nodes}/{decode_only_nodes}"
    );
    // The reservation table is the reserve half of reserve/assign: a rebuild re-maps onto
    // slots, so it can only stay or grow.
    assert!(
        session_decode.idle_slots >= session_prefill.idle_slots
            && session_decode.reserved_classes >= session_prefill.reserved_classes,
        "the reservation table cannot shrink on a rebuild: prefill {} idle in {} classes, \
         decode {} idle in {} classes",
        session_prefill.idle_slots,
        session_prefill.reserved_classes,
        session_decode.idle_slots,
        session_decode.reserved_classes
    );
    // The CPU pool carries no device budget, so the report says so instead of publishing a
    // number that was never measured (#122).
    assert!(
        session_prefill.budget.is_none() && !session_prefill.budget_is_bounded(),
        "the CPU pool has no device budget: {:?}",
        session_prefill
    );

    // The ledger's peak is cumulative: the session's decode report carries whichever peak
    // is larger, which is exactly the prefill's on this model.
    assert_eq!(
        session_decode.peak_live_bytes,
        session_prefill
            .peak_live_bytes
            .max(decode_only.peak_live_bytes),
        "peak_live_bytes is the allocator's high-water mark, not this graph's: \
         session prefill {:?}, session decode {:?}, isolated decode {:?}",
        session_prefill,
        session_decode,
        decode_only
    );
    // A 1-token decode holds a smaller activation set than a 34-token prefill — this is the
    // number the book wants, and it is why the isolated runs exist.
    assert!(
        decode_only.peak_live_bytes < prefill_only.peak_live_bytes,
        "a 1-token decode must hold a smaller peak than a {n_tokens}-token prefill: \
         decode {decode_only:?} vs prefill {prefill_only:?}"
    );
    // The *pool* does not shrink on a rebuild (the ticket's "decode's pool is smaller than
    // prefill's" contradicts the field's own contract); it is a high-water mark.
    assert!(
        session_decode.pool_bytes >= session_prefill.pool_bytes,
        "the pool is the high-water mark and cannot shrink on a rebuild: decode {:?} vs \
         prefill {:?}",
        session_decode,
        session_prefill
    );

    // The region bytes the ledger leaves out: E4's `pool_bytes` counts classed activation
    // buffers, and the persistent KV regions are exact `alloc_persistent` allocations, so
    // `pool_bytes` does not carry them. The book needs both numbers or it understates a
    // session's memory by the whole cache.
    let kv_bytes = alloc.kv_region_bytes();
    assert!(
        kv_bytes > 0,
        "the decode graph must have kept the KV regions"
    );
    assert_eq!(alloc.kv_layer_count(), model.n_layer());
    assert_eq!(alloc.kv_n_ctx(), n_ctx);

    let ledger = |r: crate::graph::alloc::MemoryReport, nodes: usize| {
        json!({
            "nodes": nodes,
            "weights_bytes": r.weights_bytes,
            "pool_bytes": r.pool_bytes,
            "live_bytes": r.live_bytes,
            "peak_live_bytes": r.peak_live_bytes,
            "budget": r.budget,
            "budget_is_bounded": r.budget_is_bounded(),
            "idle_slots": r.idle_slots,
            "reserved_classes": r.reserved_classes,
        })
    };
    emit(
        "alloc_ledger",
        json!({
            "model": std::path::Path::new(&path).file_name().and_then(|n| n.to_str()).unwrap_or(""),
            "arch": if model.as_any().is::<crate::models::qwen2::Qwen2Model>() { "qwen2" } else { "qwen3" },
            "n_ctx": n_ctx,
            "n_tokens": n_tokens,
            "kv_format": KvFormat::F32.name(),
            // One allocator across both builds: the persistent regions survive, and the
            // pool/peak fields are high-water marks over the whole session.
            "session": {
                "prefill": ledger(session_prefill, prefill_nodes),
                "decode": ledger(session_decode, decode_nodes),
            },
            // One allocator per graph: each field is that graph's own number.
            "isolated": {
                "prefill": ledger(prefill_only, prefill_only_nodes),
                "decode": ledger(decode_only, decode_only_nodes),
            },
            "kv": {
                "region_bytes": kv_bytes,
                "layers": alloc.kv_layer_count(),
                "n_ctx": alloc.kv_n_ctx(),
                "packed": alloc.kv_is_packed(),
            },
        }),
    );
}
