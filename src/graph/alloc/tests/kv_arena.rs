//! The KV arena: regions, sessions, defrag, copy-on-write and the cell bounds.
//!
//! Split out of `src/graph/alloc/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// C8a S1: the prefix copy refuses what it cannot do instead of copying a row that
/// does not exist or writing past the destination's reservation.
///
/// These are the checks that run before any backend work. The data path, and the
/// written/capacity bounds (which need real KV regions), are covered by the S2 gate:
/// the same prompt served via a copy and via a re-prefill must produce byte-identical
/// continuations.
#[test]
fn copying_a_prefix_refuses_what_it_cannot_copy() {
    let mut a = GraphAllocator::new();
    a.kv_set_capacity(16);
    // Neither sequence holds a run yet.
    let err = a.kv_copy_prefix(0, 1, 4).unwrap_err();
    assert!(err.contains("holds no run"), "got: {err}");
    a.kv_reserve_seq(0, 8).unwrap();
    // The destination has no run.
    let err = a.kv_copy_prefix(0, 1, 4).unwrap_err();
    assert!(err.contains("holds no run"), "got: {err}");
    a.kv_reserve_seq(1, 4).unwrap();
    // Zero rows is a no-op, whatever the two runs are.
    a.kv_copy_prefix(0, 0, 0).expect("zero rows");
    // A source with nothing written is refused before any copy is attempted.
    let err = a.kv_copy_prefix(0, 1, 2).unwrap_err();
    assert!(err.contains("has written 0 rows"), "got: {err}");
}
/// C6/C7: `cells` is an absolute arena row, so it is bounded by the arena — not
/// by the per-sequence position rule it used to be checked with. The two bounds
/// coincide today, which is exactly why the difference is written down: sharing a
/// cell range across sequences is what stops them coinciding.
#[test]
fn the_cells_input_is_bounded_by_the_arena() {
    let mut b = GraphBuilder::new();
    let cells = b.input("cells", [1, 1, 1, 1], crate::graph::DType::I32);
    b.output(cells);
    let g = b.build();
    let mut alloc = GraphAllocator::new();
    alloc.kv_set_capacity(8);
    alloc.alloc_graph(&g).expect("alloc the graph");
    alloc
        .fill_input_i32(&g, "cells", &[7])
        .expect("the last cell of an 8-cell arena is valid");
    let err = alloc.fill_input_i32(&g, "cells", &[8]).unwrap_err();
    assert!(err.contains("cell 8 is past the 8-cell arena"), "{err}");
    // Without an arena there is nothing to bound against, and the resolver reports
    // that case itself when it runs.
    let mut fresh = GraphAllocator::new();
    fresh.alloc_graph(&g).expect("alloc the graph");
    fresh
        .fill_input_i32(&g, "cells", &[3])
        .expect("no arena, no bound to check");
}
#[test]
fn kv_regions_two_per_layer() {
    let mut b = GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, 1024);
    let load = b.kvcache_load(0, 16, 1024, 2);
    b.output(load);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    // C6: the store now also consumes a `cells` input, so node ids shifted —
    // look the nodes up by name instead of by position.
    let store = g.nodes.iter().position(|n| n.name == "kv_store.0").unwrap();
    let load = g.nodes.iter().position(|n| n.name == "kv_load.0").unwrap();
    // store and load share the K region; V is a sibling
    assert_eq!(alloc.node_buffer(store), alloc.node_buffer(load));
    let pair = alloc.kv_pair(0).unwrap();
    assert_eq!(alloc.node_buffer(store).unwrap().id, pair.0);
    assert_ne!(pair.0, pair.1);
    assert_eq!(alloc.persistent.len(), 2);
    assert_eq!(alloc.persistent[0].name, "kv.0.k");
    assert_eq!(alloc.persistent[1].name, "kv.0.v");
    // mapped buffers: positions/k/v/cells (4 liveness) + K region (shared) = 5
    assert_eq!(alloc.n_mapped_buffers(), 5);
}
/// C5 end to end on CPU: a session's region bytes **and** its run table
/// (reservation, ownership, written extent) survive a save and a load into a
/// **fresh** allocator, and the loaded arena resolves the same cells.
///
/// The refusals are the other half of the ticket: a file that does not describe
/// this arena is rejected loudly, and because `kv_load` verifies the whole file
/// before it applies anything, a rejected load leaves no arena behind.
#[test]
fn kv_session_round_trips_the_rows_and_the_run_table() {
    const N_CTX: usize = 16;
    const ROW: usize = 4;
    const SEQ: u32 = 7;
    let kk: Vec<f32> = (0..ROW * 2).map(|i| 0.25 + i as f32 * 0.5).collect();
    let vv: Vec<f32> = (0..ROW * 2).map(|i| 1.0 / (i as f32 + 1.0)).collect();

    let graph = |alloc: &mut GraphAllocator| -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let pos = b.input("positions", [2, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, N_CTX);
        let load = b.kvcache_load(0, ROW, N_CTX, 1);
        b.output(load);
        let g = b.build();
        alloc.kv_set_capacity(N_CTX);
        alloc.alloc_graph(&g).unwrap();
        g
    };

    // ---- the session that stays in memory ----
    let mut a = GraphAllocator::new();
    let ga = graph(&mut a);
    a.kv_reserve_seq(SEQ, N_CTX).unwrap();
    a.fill_input_i32(&ga, "positions", &[1, 3]).unwrap();
    // E2, the production spelling: the reservation above means the fill owns
    // positions 1 and 3 in sequence `SEQ` without reserving anything else (the
    // deleted E1 helper recorded a prefix on `SEQ_MAIN` here instead).
    a.fill_batch_inputs(&ga, &Batch::new(vec![0, 0], vec![1, 3], vec![SEQ, SEQ]))
        .unwrap();
    a.fill_input(&ga, "k", &kk).unwrap();
    a.fill_input(&ga, "v", &vv).unwrap();
    let mut sched = crate::graph::scheduler::BackendScheduler::new();
    sched.execute(&ga, &mut a).unwrap();
    a.kv.own_range(SEQ, 0, 4);
    let want_kv = a.copy_kv_to_cpu(0).unwrap();
    let want_stats = a.kv_arena_stats();
    let want_cells = a.kv_cells_for_seq(&[SEQ, SEQ], &[1, 3]).unwrap();

    let path = std::env::temp_dir().join(format!(
        "minfer-c5-alloc-{}-roundtrip.bin",
        std::process::id()
    ));
    let report = a.kv_save(&path).unwrap();
    assert_eq!(report.layers, 1);
    assert_eq!(report.cells, N_CTX);
    assert_eq!(report.written, 4, "positions 0..4 are written");
    assert_eq!(report.bytes, std::fs::metadata(&path).unwrap().len());

    // ---- a fresh allocator, restored from the file ----
    let expect = KvSessionExpect {
        backend: Backend::CPU,
        n_ctx: N_CTX,
        n_embd: ROW,
    };
    let mut b = GraphAllocator::new();
    let gb = graph(&mut b);
    let loaded = b.kv_load(&path, &expect).unwrap();
    assert_eq!(loaded, report);
    assert_eq!(
        b.copy_kv_to_cpu(0).unwrap(),
        want_kv,
        "the region bytes must be identical"
    );
    assert_eq!(b.kv_arena_stats(), want_stats);
    assert_eq!(
        b.kv_cells_for_seq(&[SEQ, SEQ], &[1, 3]).unwrap(),
        want_cells,
        "the restored run table must resolve the same cells"
    );
    assert_eq!(b.kv_n_used(0), a.kv_n_used(0));
    let _ = gb;

    // ---- refusals ----
    for (what, expect_bad, needle) in [
        (
            "a different n_ctx",
            KvSessionExpect {
                n_ctx: N_CTX + 1,
                ..expect
            },
            "-cell arena",
        ),
        (
            "a different row width",
            KvSessionExpect {
                n_embd: ROW + 4,
                ..expect
            },
            "KV rows",
        ),
        (
            "another backend",
            KvSessionExpect {
                backend: Backend::METAL,
                ..expect
            },
            "session",
        ),
    ] {
        let mut fresh = GraphAllocator::new();
        let err = fresh.kv_load(&path, &expect_bad).unwrap_err();
        assert!(err.contains(needle), "{what}: {err}");
        assert!(
            fresh.kv_n_used(0).is_none(),
            "{what}: a refused load must not create an arena"
        );
    }
    // A different KV element type for the same shape.
    let mut fresh = GraphAllocator::new();
    fresh
        .cpu_mut()
        .set_kv_format(crate::graph::kvformat::KvFormat::Q8_0);
    let err = fresh.kv_load(&path, &expect).unwrap_err();
    assert!(err.contains("element type"), "{err}");
    assert!(fresh.kv_n_used(0).is_none());

    // A truncated file: rejected, and (because `kv_load` verifies first) the
    // allocator is untouched.
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, &full[..full.len() / 2]).unwrap();
    let mut fresh = GraphAllocator::new();
    let err = fresh.kv_load(&path, &expect).unwrap_err();
    assert!(err.contains("truncated"), "{err}");
    assert!(fresh.kv_n_used(0).is_none());
    std::fs::remove_file(&path).ok();
}
/// #130: the same `kv_save` → `kv_load` round trip under an **f16** element
/// type — the policy a CUDA box's auto rule picks for the 7B class
/// (`n_layers * n_kv_embd >= 8192`). The store here is the CPU's (which copies
/// words), so this gates the *container* half with no device: the format the
/// allocator reports, the header's flags, the reader's decode and the
/// `header.format != live` check. Before `FLAG_F16` the load refused the file
/// the save had just produced. The device half is the Qwen3-0.6B real-model gate.
#[test]
fn an_f16_session_round_trips_through_save_and_load() {
    use crate::graph::kvformat::KvFormat;
    const N_CTX: usize = 8;
    const ROW: usize = 4;
    const SEQ: u32 = 3;
    let kk: Vec<f32> = (0..ROW * 2).map(|i| 0.5 + i as f32).collect();
    let vv: Vec<f32> = (0..ROW * 2).map(|i| 1.0 / (i as f32 + 1.0)).collect();

    let build = |alloc: &mut GraphAllocator| -> ComputeGraph {
        alloc.cpu_mut().set_kv_format(KvFormat::F16);
        let mut b = GraphBuilder::new();
        b.set_kv_format(KvFormat::F16);
        let pos = b.input("positions", [2, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, N_CTX);
        let load = b.kvcache_load(0, ROW, N_CTX, 1);
        b.output(load);
        let g = b.build();
        alloc.kv_set_capacity(N_CTX);
        alloc.alloc_graph(&g).unwrap();
        g
    };

    let mut a = GraphAllocator::new();
    let ga = build(&mut a);
    a.kv_reserve_seq(SEQ, N_CTX).unwrap();
    a.fill_input_i32(&ga, "positions", &[0, 1]).unwrap();
    // E2, the production spelling (see the f32 round trip above).
    a.fill_batch_inputs(&ga, &Batch::new(vec![0, 0], vec![0, 1], vec![SEQ, SEQ]))
        .unwrap();
    a.fill_input(&ga, "k", &kk).unwrap();
    a.fill_input(&ga, "v", &vv).unwrap();
    let mut sched = crate::graph::scheduler::BackendScheduler::new();
    sched.execute(&ga, &mut a).unwrap();
    a.kv.own_range(SEQ, 0, 2);
    let want_kv = a.copy_kv_to_cpu(0).unwrap();

    let path = std::env::temp_dir().join(format!("minfer-c5-alloc-{}-f16.bin", std::process::id()));
    let report = a.kv_save(&path).unwrap();
    // The bytes on disk name f16 — the flag the reader must decode.
    let bytes = std::fs::read(&path).unwrap();
    let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    assert_eq!(flags, 1 << 1, "the header must carry FLAG_F16");

    // A fresh allocator under the same policy resumes it, rows bit-identical.
    let expect = KvSessionExpect {
        backend: Backend::CPU,
        n_ctx: N_CTX,
        n_embd: ROW,
    };
    let mut b = GraphAllocator::new();
    let gb = build(&mut b);
    let loaded = b.kv_load(&path, &expect).unwrap();
    assert_eq!(loaded, report);
    assert_eq!(b.copy_kv_to_cpu(0).unwrap(), want_kv);
    let _ = gb;

    // An f32 reader refuses it by element type — not silently, and without
    // creating an arena.
    let mut c = GraphAllocator::new();
    c.cpu_mut().set_kv_format(KvFormat::F32);
    let err = c.kv_load(&path, &expect).unwrap_err();
    assert!(err.contains("element type"), "{err}");
    assert!(err.contains("f16"), "{err}");
    assert!(c.kv_n_used(0).is_none());
    std::fs::remove_file(&path).ok();
}
/// C3 end to end on CPU: a fragmented arena refuses an 8-cell reservation,
/// the compaction moves the real KV bytes *and* renumbers the runs, and the
/// reservation then fits — with the moved rows byte-identical at their new
/// cells.
#[test]
fn kv_defrag_moves_the_bytes_and_opens_the_run() {
    const N_CTX: usize = 16;
    const ROW: usize = 4; // elements per cell (n_kv_embd)
    let mut b = GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, N_CTX);
    let load = b.kvcache_load(0, ROW, N_CTX, 1);
    b.output(load);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g).unwrap();
    for seq in 1u32..=3 {
        assert_eq!(
            alloc.kv_reserve_seq(seq, 4).unwrap().start,
            (seq as usize - 1) * 4
        );
        alloc
            .kv
            .own_range(seq, (seq as usize - 1) * 4, seq as usize * 4);
    }
    let rope = crate::graph::kvcache::KvRope {
        freq_base: 10_000.0,
        freq_scale: 1.0,
        n_head_kv: 1,
        hd: ROW,
        style: crate::vec_ops::RopeStyle::NonInterleaved,
    };
    let (kid, vid) = alloc.kv_pair(0).unwrap();
    let pattern = |base: f32| -> Vec<f32> {
        (0..N_CTX * ROW)
            .map(|i| base + (i / ROW) as f32 + (i % ROW) as f32 / 10.0)
            .collect()
    };
    alloc
        .write_pool(crate::graph::Backend::CPU, kid, &pattern(0.0))
        .unwrap();
    alloc
        .write_pool(crate::graph::Backend::CPU, vid, &pattern(100.0))
        .unwrap();
    // Release the middle run: free runs [4,8) and [12,16), 8 cells, none 8 long.
    alloc.kv_release_seq(2);
    let before = alloc.kv_arena_stats();
    assert_eq!((before.free_cells, before.free_runs), (8, 2));
    assert!(alloc.kv_reserve_seq(4, 8).is_err());

    let report = alloc.kv_defrag(Some(8)).unwrap();
    assert_eq!(report.moves.len(), 1, "{report:?}");
    assert_eq!(report.moves[0].seq, 3);
    assert_eq!((report.moves[0].from, report.moves[0].to), (8, 4));
    assert_eq!(report.rows_moved, 4);
    assert_eq!((report.before.free_runs, report.after.free_runs), (2, 1));
    assert_eq!(report.after.largest_free_run, 8);
    assert_eq!((report.after.defrags, report.after.cells_moved), (1, 4));
    // C6: both K and V move **verbatim** — a cell move no longer changes any
    // rotation, because a token's angle is its index within its sequence.
    let (k_now, v_now) = alloc.copy_kv_to_cpu(0).unwrap();
    for e in 0..ROW {
        assert_eq!(v_now[4 * ROW + e], 108.0 + e as f32 / 10.0, "V element {e}");
    }
    let k_expect: Vec<f32> = (0..ROW).map(|e| 8.0 + e as f32 / 10.0).collect();
    for e in 0..ROW {
        assert_eq!(k_now[4 * ROW + e], k_expect[e], "K element {e}");
    }
    // The reservation that first-fit refused now fits, in the opened tail.
    assert_eq!(alloc.kv_reserve_seq(4, 8).unwrap().start, 8);
    // The same helper, on a fresh fragmentation: free the lowest run and the
    // 8-cell one, and ask for 12 contiguous cells. First-fit fails (the free
    // space is two runs); compacting the single survivor down opens the tail,
    // and the helper reports the move it relied on.
    alloc.kv_release_seq(1);
    alloc.kv_release_seq(4);
    let (slot, moves) = alloc.kv_reserve_seq_with_defrag(5, 12).unwrap();
    assert_eq!(
        slot.start, 4,
        "the retry packs the survivor down, then takes the tail"
    );
    assert_eq!(moves.len(), 1, "{moves:?}");
    assert_eq!((moves[0].seq, moves[0].from, moves[0].to), (3, 4, 0));
    // The helper also answers "it still does not fit" with the original
    // first-fit error, after compaction moved nothing.
    let err = alloc.kv_reserve_seq_with_defrag(6, 4).unwrap_err();
    assert!(err.contains("no free run of 4 cells"), "{err}");
}
#[test]
fn the_defrag_gate_is_off_only_when_the_flag_is_present() {
    assert!(super::kv_defrag_enabled_from(None));
    assert!(!super::kv_defrag_enabled_from(Some(std::ffi::OsStr::new(
        "1"
    ))));
}
/// C8b S3 end to end on CPU: a sequence that reads a prefix in place cannot
/// store into it, the copy-on-write moves **its own** rows up inside its run,
/// and the donor's cells come out of it byte-identical.
#[test]
fn a_copy_on_write_moves_the_rows_and_never_writes_through() {
    const N_CTX: usize = 16;
    const ROW: usize = 4; // elements per cell (n_kv_embd)
    let mut b = GraphBuilder::new();
    let _pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, N_CTX);
    let load = b.kvcache_load(0, ROW, N_CTX, 1);
    b.output(load);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g).unwrap();
    let (kid, vid) = alloc.kv_pair(0).unwrap();
    // Every cell holds the row index and the element, so a moved row is
    // identifiable wherever it lands.
    let pattern = |base: f32| -> Vec<f32> {
        (0..N_CTX * ROW)
            .map(|i| base + (i / ROW) as f32 * 100.0 + (i % ROW) as f32 / 10.0)
            .collect()
    };
    alloc
        .write_pool(crate::graph::Backend::CPU, kid, &pattern(0.0))
        .unwrap();
    alloc
        .write_pool(crate::graph::Backend::CPU, vid, &pattern(1000.0))
        .unwrap();
    // Sequence 1 computes four rows at [0, 4); sequence 2 reserves [4, 10),
    // reads those four in place, and writes two rows of its own at [4, 6).
    assert_eq!(alloc.kv_reserve_seq(1, 4).unwrap().start, 0);
    alloc.kv.own_range(1, 0, 4);
    assert_eq!(alloc.kv_reserve_seq(2, 6).unwrap().start, 4);
    assert_eq!(alloc.kv_share_prefix(1, 2, 4).unwrap(), 4);
    alloc.kv.own_range(2, 4, 6);

    // The store resolver refuses a position inside the share: that refusal is
    // what makes a write-through impossible rather than merely unlikely.
    let err = alloc.kv_cells_for_seq(&[2], &[1]).unwrap_err();
    assert!(err.contains("shared prefix"), "got: {err}");
    assert!(err.contains("C8b S3"), "got: {err}");

    let shift = alloc
        .kv_private_row_for(2, 1)
        .unwrap()
        .expect("position 1 must copy-on-write");
    assert_eq!((shift.base, shift.from, shift.to, shift.rows), (1, 4, 7, 2));
    assert_eq!(alloc.kv_arena_stats().cows, 1);
    // From position 1 on everything is private, and the rows that were at
    // [4, 6) kept their positions at [7, 9) — while position 0 still reads the
    // donor's cell 0.
    assert_eq!(alloc.kv.cell_of(2, 0), Some(0), "still shared");
    assert_eq!(
        alloc
            .kv_cells_for_seq(&[2, 2, 2, 2, 2], &[1, 2, 3, 4, 5])
            .unwrap(),
        vec![4, 5, 6, 7, 8]
    );
    // A store at 0 would still land in the donor's cells, so it is still
    // refused — the share is smaller, not gone.
    assert!(alloc.kv_cells_for_seq(&[2], &[0]).is_err());
    // The data moved verbatim (K and V), and the donor's four rows are what
    // they were — nothing wrote through them.
    let (k_now, v_now) = alloc.copy_kv_to_cpu(0).unwrap();
    for e in 0..ROW {
        let row = |cell: usize| cell * ROW + e;
        assert_eq!(k_now[row(7)], 400.0 + e as f32 / 10.0, "K at [7]");
        assert_eq!(k_now[row(8)], 500.0 + e as f32 / 10.0, "K at [8]");
        assert_eq!(v_now[row(7)], 1400.0 + e as f32 / 10.0, "V at [7]");
        assert_eq!(v_now[row(8)], 1500.0 + e as f32 / 10.0, "V at [8]");
        for cell in 0..4 {
            assert_eq!(
                k_now[row(cell)],
                cell as f32 * 100.0 + e as f32 / 10.0,
                "donor K cell {cell} changed"
            );
            assert_eq!(
                v_now[row(cell)],
                1000.0 + cell as f32 * 100.0 + e as f32 / 10.0,
                "donor V cell {cell} changed"
            );
        }
    }
    // Idempotent for a position that is already private.
    assert_eq!(alloc.kv_private_row_for(2, 3).unwrap(), None);
    assert_eq!(alloc.kv_arena_stats().cows, 1);
    // And a sequence with no run at all is a no-op, not an error — that is what
    // keeps the classic single-sequence path (which never reserves) untouched.
    assert_eq!(alloc.kv_private_row_for(9, 0).unwrap(), None);
    // The production entry point gets the same rule: `fill_batch_inputs` copies
    // for a batch that still names a shared position, and then resolves the
    // private cells. Position 0 is the last shared one, so this is the second
    // (and final) copy-on-write. A `Batch::single` would name `SEQ_MAIN` (0) and
    // try to reserve the whole arena for it — the wrong sequence, and the arena is
    // already full — so this spells the batch out for sequence 2.
    alloc
        .fill_batch_inputs(&g, &Batch::new(vec![0], vec![0], vec![2]))
        .unwrap();
    assert_eq!(alloc.kv_arena_stats().cows, 2);
    assert_eq!(alloc.kv.spans_of(2), &[(0, 4, 6)], "the share is gone");
    assert_eq!(alloc.kv.cell_of(2, 0), Some(4));
    assert_eq!(alloc.kv.cell_of(2, 5), Some(9));
    let (k_end, _) = alloc.copy_kv_to_cpu(0).unwrap();
    for e in 0..ROW {
        assert_eq!(
            k_end[9 * ROW + e],
            500.0 + e as f32 / 10.0,
            "the rows shifted up once more"
        );
    }
}
/// The KV regions are persistent across rebuilds (they ARE the cache), so a
/// graph that asks for a different `n_ctx` on the same allocator must be a
/// loud error, not a silent reuse of the older, smaller region.
#[test]
fn kv_region_size_change_is_a_loud_error() {
    fn kv_graph(n_ctx: usize) -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, n_ctx);
        let load = b.kvcache_load(0, 16, n_ctx, 2);
        b.output(load);
        b.build()
    }

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&kv_graph(1024)).unwrap();
    // Unchanged shape: reuse is fine (this is the decode-reuse path).
    alloc.alloc_graph(&kv_graph(1024)).unwrap();
    // Changed n_ctx on a live cache: must fail loudly.
    let err = alloc.alloc_graph(&kv_graph(2048)).unwrap_err();
    assert!(err.contains("KV region for layer 0"), "got: {err}");
}
/// The KV row input is `cells` (C6), and a cell is an index into the arena, so a
/// value at or past the arena is an out-of-bounds write on every backend that does
/// not re-check it (the GPU ones). It is bounded by the *arena* (C7/#60), not by the
/// per-sequence position rule — the two only coincide while a cell equals its
/// position. The guard lives in the allocator so all three backends share it.
#[test]
fn a_cell_beyond_the_arena_is_rejected() {
    let mut b = GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, 1024);
    let load = b.kvcache_load(0, 16, 1024, 2);
    b.output(load);
    let g = b.build();
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();

    // The last legal row is n_ctx - 1.
    alloc.fill_input_i32(&g, "cells", &[1023]).unwrap();
    let err = alloc.fill_input_i32(&g, "cells", &[1024]).unwrap_err();
    assert!(
        err.contains("cell 1024 is past the 1024-cell arena"),
        "got: {err}"
    );
}
/// The same guard must NOT bound `token_ids`: a vocabulary is routinely
/// larger than `n_ctx`.
#[test]
fn token_ids_are_not_bounded_by_n_ctx() {
    let mut b = GraphBuilder::new();
    let ids = b.input("token_ids", [1, 1, 1, 1], crate::graph::DType::I32);
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
    let emb = b.get_rows(k, ids, [16, 1, 1, 1]);
    let _store = b.kvcache_store(0, emb, v, 8);
    let load = b.kvcache_load(0, 16, 8, 2);
    b.output(load);
    let g = b.build();
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();

    alloc
        .fill_input_i32(&g, "token_ids", &[50_000])
        .expect("token ids are not positions");
    let err = alloc.fill_input_i32(&g, "cells", &[8]).unwrap_err();
    assert!(
        err.contains("cell 8 is past the 8-cell arena"),
        "got: {err}"
    );
}
/// E5: a KV session is one arena (the container carries a single backend tag), so a mixed
/// CPU/device plan cannot resume one. The refusal happens before the file is read — the
/// path below does not exist, which is how the test proves the order.
#[test]
fn a_mixed_offload_plan_refuses_to_resume_a_session() {
    use crate::graph::kvsession::KvSessionExpect;
    let expect = KvSessionExpect {
        backend: Backend::CPU,
        n_ctx: 16,
        n_embd: 8,
    };
    let mut alloc = GraphAllocator::new();
    alloc.set_offload_plan(Some(crate::graph::offload::OffloadPlan {
        gpu_layers: 2,
        n_layers: 4,
    }));
    let err = alloc
        .kv_load(std::path::Path::new("/nonexistent/e5-mixed.bin"), &expect)
        .unwrap_err();
    assert!(err.contains("offload plan"), "got: {err}");
    // An all-CPU or all-device plan proceeds to the file and fails there instead.
    for plan in [
        crate::graph::offload::OffloadPlan {
            gpu_layers: 0,
            n_layers: 4,
        },
        crate::graph::offload::OffloadPlan {
            gpu_layers: 4,
            n_layers: 4,
        },
    ] {
        alloc.set_offload_plan(Some(plan));
        let err = alloc
            .kv_load(std::path::Path::new("/nonexistent/e5-mixed.bin"), &expect)
            .unwrap_err();
        assert!(!err.contains("offload plan"), "{plan:?}: {err}");
    }
}
