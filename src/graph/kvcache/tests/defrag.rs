//! C3 compaction: the free-run view, the move plan and its application.
//!
//! Split out of `src/graph/kvcache/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn a_fresh_arena_is_one_free_run() {
    let c = cache(16);
    let st = c.arena_stats();
    assert_eq!(
        (st.free_cells, st.free_runs, st.largest_free_run),
        (16, 1, 16)
    );
    assert_eq!((st.reserved_cells, st.owned_cells, st.sequences), (0, 0, 0));
}
#[test]
fn fragmentation_refuses_a_run_the_arena_could_hold() {
    let mut c = cache(16);
    // A[0,4) B[4,4) C[8,4) D[12,4), all written.
    for seq in 1u32..=4 {
        place(&mut c, seq, 4);
    }
    // Free A and C: two 4-cell runs, 8 free cells, none of them 8 long.
    c.release_seq(1);
    c.release_seq(3);
    let st = c.arena_stats();
    assert_eq!(
        (st.free_cells, st.free_runs, st.largest_free_run),
        (8, 2, 4)
    );
    let err = c.reserve_seq(5, 8).unwrap_err();
    assert!(err.contains("no free run of 8 cells"), "{err}");
    // This is the failure C3 exists for: capacity is not the constraint.
    assert!(st.free_cells >= 8);
}
#[test]
fn the_plan_stops_as_soon_as_it_opens_the_run_it_needs() {
    let mut c = cache(16);
    for seq in 1u32..=4 {
        place(&mut c, seq, 4);
    }
    c.release_seq(1);
    c.release_seq(3);
    // Moving B down to 0 opens [4, 12): enough for 8 cells, so D stays put.
    assert_eq!(
        c.compaction_plan(Some(8)),
        vec![KvMove {
            seq: 2,
            from: 4,
            to: 0,
            rows: 4
        }]
    );
    // Without `need`, the whole live set is packed.
    assert_eq!(
        c.compaction_plan(None),
        vec![
            KvMove {
                seq: 2,
                from: 4,
                to: 0,
                rows: 4
            },
            KvMove {
                seq: 4,
                from: 12,
                to: 4,
                rows: 4
            },
        ]
    );
}
#[test]
fn applying_a_plan_opens_the_run_and_keeps_every_sequences_rows() {
    let mut c = cache(16);
    for seq in 1u32..=4 {
        place(&mut c, seq, 4);
    }
    c.release_seq(1);
    c.release_seq(3);
    let before = c.arena_stats();
    // The full plan packs both live runs, so the free tail is the whole top.
    let plan = c.compaction_plan(None);
    assert_eq!(c.apply_moves(&plan).unwrap(), 8, "4 written rows per run");
    let after = c.arena_stats();
    assert_eq!(after.largest_free_run, 8, "{after:?}");
    assert_eq!(after.free_runs, 1);
    assert_eq!(
        (after.defrags, after.cells_moved),
        (1, 16),
        "8 rows x 2 layers"
    );
    assert!(before.largest_free_run < 8);
    // The reservation that first-fit refused now fits, in the opened tail.
    assert_eq!(c.reserve_seq(5, 8).unwrap().start, 8);
    // Ownership travelled with each sequence; the vacated cells are FREE.
    let owner = &c.get(0).unwrap().owner;
    assert_eq!(owner[0..4], [2, 2, 2, 2]);
    assert_eq!(owner[4..8], [4, 4, 4, 4]);
    assert!(owner[8..16].iter().all(|&o| o == FREE));
    assert_eq!(c.seq_slot(4).unwrap().start, 4);
    assert_eq!(
        c.get(0).unwrap().n_used,
        8,
        "n_used must follow the rows down, not stay at the old top"
    );
    // The data the caller will copy is the sequence's live prefix.
    assert_eq!(c.written_rows(4), 4);
}
#[test]
fn an_unwritten_run_moves_its_reservation_without_copying_rows() {
    let mut c = cache(16);
    c.reserve_seq(1, 4).unwrap();
    c.reserve_seq(2, 4).unwrap();
    c.release_seq(1);
    let plan = c.compaction_plan(None);
    assert_eq!(
        plan,
        vec![KvMove {
            seq: 2,
            from: 4,
            to: 0,
            rows: 0
        }]
    );
    assert_eq!(c.apply_moves(&plan).unwrap(), 0);
    assert_eq!(c.seq_slot(2).unwrap().start, 0);
    assert_eq!(c.arena_stats().cells_moved, 0);
    assert_eq!(c.get(0).unwrap().n_used, 0);
}
#[test]
fn a_plan_that_would_overlap_a_live_run_is_refused() {
    let mut c = cache(16);
    place(&mut c, 1, 4); // [0, 4)
    place(&mut c, 2, 4); // [4, 8)
    let bad = vec![KvMove {
        seq: 2,
        from: 4,
        to: 2,
        rows: 4,
    }];
    let err = c.apply_moves(&bad).unwrap_err();
    assert!(err.contains("would overlap"), "{err}");
    // Nothing moved: the run table and the counters are untouched.
    assert_eq!(c.seq_slot(2).unwrap().start, 4);
    assert_eq!(c.arena_stats().defrags, 0);
    // A stale `from` and an upward move are refused too.
    let stale = vec![KvMove {
        seq: 2,
        from: 0,
        to: 0,
        rows: 0,
    }];
    assert!(c.apply_moves(&stale).unwrap_err().contains("the plan says"));
    // C7b: an upward move is legal now — what a plan may never do is land two
    // runs on the same cells, in either direction.
    let up_overlap = vec![
        KvMove {
            seq: 1,
            from: 0,
            to: 8,
            rows: 0,
        },
        KvMove {
            seq: 2,
            from: 4,
            to: 8,
            rows: 0,
        },
    ];
    assert!(c.apply_moves(&up_overlap).unwrap_err().contains("overlap"));
    // Rows past the run's cap are a bug, not a bigger copy.
    let too_many = vec![KvMove {
        seq: 2,
        from: 4,
        to: 0,
        rows: 5,
    }];
    assert!(c.apply_moves(&too_many).unwrap_err().contains("rows"));
}
