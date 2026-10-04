//! C8b prefix sharing: in-place reads, copy-on-write, map runs and the refusals.
//!
//! Split out of `src/graph/kvcache/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// C8b S2: sharing a prefix means the destination reads the donor's cells — one
/// copy of the bytes, two sequences — and keeps writing its own positions into
/// its own run. The two halves of the address space have to come back out of the
/// span list as one entry each.
#[test]
fn a_shared_prefix_is_read_in_place_and_written_past() {
    let mut c = cache(16);
    let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    assert_eq!(c.share_prefix(1, 2, 4), Ok(()));
    assert_eq!(
        c.spans_of(2),
        &[(0, donor.start, 4), (4, dst.start, 2)],
        "shared prefix then the private tail"
    );
    assert_eq!(c.cell_of(2, 0), Some(donor.start), "read in place");
    assert_eq!(c.cell_of(2, 3), Some(donor.start + 3));
    assert_eq!(c.cell_of(2, 4), Some(dst.start), "then its own run");
    assert_eq!(c.written_rows(2), 4, "the shared rows count as written");
    assert_eq!(c.private_written(2), 0, "but none of them are its own");
    // The sharer's own storage is not in its run's cells.
    assert_eq!(c.get(0).unwrap().owner[dst.start], FREE);

    c.own_positions(2, &[4, 5]).unwrap();
    assert_eq!(c.written_rows(2), 6);
    assert_eq!(c.private_written(2), 2);
    assert_eq!(c.get(0).unwrap().owner[dst.start], 2, "its own rows");
    assert_eq!(
        c.get(0).unwrap().owner[donor.start],
        1,
        "a shared row keeps the donor's stamp"
    );
    let stats = c.arena_stats();
    assert_eq!(stats.shared_cells, 4, "the saving is visible");
    assert_eq!(stats.reserved_cells, 6, "only the two private runs");
    // The donor is untouched by any of it and still reads its own rows.
    assert_eq!(c.spans_of(1), &[(0, donor.start, 4)]);
    assert_eq!(c.written_rows(1), 4);
}
/// C8b S3: a store inside a shared prefix takes a **private row** — the share
/// shrinks to the first position the forward writes, and the rows the sequence
/// already owns shift up inside its own run so they keep their positions. The
/// donor's cells are never named again by that position.
#[test]
fn a_store_inside_a_shared_prefix_takes_a_private_row() {
    let mut c = cache(16);
    let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let dst = c.reserve_seq(2, 4).unwrap(); // [4, 8)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    c.share_prefix(1, 2, 4).unwrap();
    c.own_positions(2, &[4, 5]).unwrap(); // two rows of its own at [4, 6)
    assert_eq!(c.spans_of(2), &[(0, donor.start, 4), (4, dst.start, 4)]);
    assert_eq!(c.private_written(2), 2);

    // Position 3 and everything after it is written privately; 4 is already
    // private, and an unknown sequence has nothing to copy.
    assert_eq!(c.private_row_for(2, 4), Ok(None), "already private");
    assert_eq!(c.private_row_for(7, 0), Ok(None), "no run, no share");
    let shift = c.private_row_for(2, 3).unwrap().expect("a copy is needed");
    assert_eq!(
        shift,
        KvShift {
            seq: 2,
            base: 3,
            from: dst.start,
            to: dst.start + 1,
            rows: 2,
        }
    );
    c.apply_private_row(&shift).unwrap();

    // The address space is the remaining share plus the whole run, whose base
    // dropped to the position the store starts at.
    assert_eq!(c.spans_of(2), &[(0, donor.start, 3), (3, dst.start, 4)]);
    for p in 0..3 {
        assert_eq!(c.cell_of(2, p), Some(donor.start + p), "still shared");
    }
    assert_eq!(c.cell_of(2, 3), Some(dst.start), "private from here on");
    assert_eq!(c.cell_of(2, 4), Some(dst.start + 1), "the row it moved up");
    assert_eq!(c.cell_of(2, 5), Some(dst.start + 2));
    assert_eq!(c.written_rows(2), 6, "no readable position was lost");
    assert_eq!(c.private_written(2), 3, "and the run holds three of them");
    // The owner table followed the move: [4, 6) -> [5, 7), and the vacated cell
    // is the new base's unwritten row.
    let owner = &c.get(0).unwrap().owner;
    assert_eq!(owner[dst.start], FREE, "vacated");
    assert_eq!(owner[dst.start + 1], 2);
    assert_eq!(owner[dst.start + 2], 2);
    assert_eq!(owner[donor.start], 1, "the donor keeps its own row");
    let stats = c.arena_stats();
    assert_eq!((stats.cows, stats.cow_cells), (1, 4), "2 rows x 2 layers");
    assert_eq!(stats.shared_cells, 3, "the saving shrank with the share");
    // A query at the copy-on-written row reads the private cells and the
    // remaining share — and the donor is untouched.
    let map = c.attn_map(&[2], &[4]).unwrap();
    assert_eq!(&map[..4], &[donor.start as u32, 3, dst.start as u32, 2]);
    assert_eq!(c.written_rows(1), 4);
    assert_eq!(c.spans_of(1), &[(0, donor.start, 4)]);
}
/// A sequence that diverges earlier than it did last time copies again: the share
/// only ever shrinks and the run only ever extends its own written range, so the
/// address space stays at most two spans however often this happens.
#[test]
fn a_copy_on_write_can_shrink_the_share_twice() {
    let mut c = cache(16);
    let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let dst = c.reserve_seq(2, 6).unwrap(); // [4, 10)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    c.share_prefix(1, 2, 4).unwrap();
    c.own_positions(2, &[4, 5]).unwrap();
    // First the store reaches position 3, then 1, then 0.
    let steps = [
        (3usize, 3usize, 4usize, 5usize, 2usize),
        (1, 1, 4, 6, 3),
        (0, 0, 4, 5, 5),
    ];
    for (t, base, from, to, rows) in steps {
        let shift = c.private_row_for(2, t).unwrap().expect("a copy is needed");
        assert_eq!(
            (shift.base, shift.from, shift.to, shift.rows),
            (base, from, to, rows)
        );
        c.apply_private_row(&shift).unwrap();
    }
    assert_eq!(c.spans_of(2), &[(0, dst.start, 6)], "the share is gone");
    assert_eq!(c.cell_of(2, 0), Some(dst.start));
    assert_eq!(c.cell_of(2, 4), Some(dst.start + 4));
    assert_eq!(c.cell_of(2, 5), Some(dst.start + 5));
    assert_eq!(c.written_rows(2), 6);
    assert_eq!(c.private_written(2), 6);
    assert_eq!(
        c.spans_of(1),
        &[(0, donor.start, 4)],
        "the donor is untouched"
    );
    assert_eq!(c.arena_stats().cows, 3);
}
/// The other end of the same rule: a store at position 0 gives the share up
/// entirely, and the sequence's whole run becomes its own.
#[test]
fn a_copy_on_write_to_the_first_position_drops_the_share() {
    let mut c = cache(16);
    let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let dst = c.reserve_seq(2, 6).unwrap(); // [4, 10)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    c.share_prefix(1, 2, 4).unwrap();
    c.own_positions(2, &[4, 5]).unwrap();
    let shift = c.private_row_for(2, 0).unwrap().expect("a copy is needed");
    assert_eq!((shift.base, shift.from, shift.to, shift.rows), (0, 4, 8, 2));
    c.apply_private_row(&shift).unwrap();
    assert_eq!(
        c.spans_of(2),
        &[(0, dst.start, 6)],
        "one run, no share left"
    );
    assert_eq!(c.cell_of(2, 0), Some(dst.start));
    assert_eq!(c.cell_of(2, 4), Some(dst.start + 4), "its own row, moved");
    assert_eq!(c.seq_slot(2).unwrap().shared.rows, 0);
    assert_eq!(c.written_rows(2), 6);
}
/// A sequence whose run has no room for the extra rows cannot take the store
/// privately: that is a loud refusal (gate 3), never a write into the donor's
/// cells.
#[test]
fn a_copy_on_write_refuses_a_run_without_room() {
    let mut c = cache(16);
    c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    c.share_prefix(1, 2, 4).unwrap();
    c.own_positions(2, &[4, 5]).unwrap(); // the run is exactly full
    let err = c.private_row_for(2, 0).unwrap_err();
    assert!(
        err.contains("cannot take position 0 privately"),
        "got: {err}"
    );
    assert!(err.contains("run of 2 cells"), "got: {err}");
    let err = c.private_row_for(2, 2).unwrap_err();
    assert!(
        err.contains("cannot take position 2 privately"),
        "got: {err}"
    );
    assert_eq!(c.spans_of(2).len(), 2, "the refusals changed nothing");
    assert_eq!(c.written_rows(2), 6);
    // A plan from a state the store is no longer in is refused rather than
    // applied to the wrong rows.
    let stale = KvShift {
        seq: 2,
        base: 3,
        from: dst.start + 1,
        to: dst.start + 2,
        rows: 1,
    };
    let err = c.apply_private_row(&stale).unwrap_err();
    assert!(err.contains("not in the state"), "got: {err}");
    let gone = KvShift {
        seq: 9,
        base: 0,
        from: 0,
        to: 1,
        rows: 1,
    };
    assert!(c
        .apply_private_row(&gone)
        .unwrap_err()
        .contains("holds no run"));
    let stats = c.arena_stats();
    assert_eq!((stats.cows, stats.cow_cells), (0, 0), "nothing ran");
}
/// A released donor must not hand its rows to somebody else while a sharer
/// still reads them (C8b S2 gate 3): occupancy comes from the span lists, so
/// the cells stay taken even though no slot reserves them any more.
#[test]
fn releasing_the_donor_keeps_the_shared_rows_taken() {
    let mut c = cache(16);
    let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    c.share_prefix(1, 2, 4).unwrap();
    assert_eq!(c.release_seq(1), 4, "the donor's run is what it freed");
    assert_eq!(c.cell_of(2, 0), Some(donor.start), "still reading the rows");
    assert_eq!(c.written_rows(2), 4);
    let held = donor.start..donor.start + 4;
    assert!(
        c.free_runs()
            .iter()
            .all(|&(s, l)| s + l <= held.start || s >= held.end),
        "the shared rows are not free: {:?}",
        c.free_runs()
    );
    // A third sequence cannot be placed on top of them.
    let err = c.reserve_seq(3, 16).unwrap_err();
    assert!(err.contains("no free run"), "got: {err}");
    // Once the sharer drops them they are free again.
    c.release_seq(2);
    assert_eq!(c.free_runs(), vec![(0, 16)]);
}
/// A compaction moves the donor's rows and **every sharer's pointer has to
/// follow** — the requirement the plan calls out for a shared block.
#[test]
fn a_shared_prefix_follows_the_donors_rows_when_they_move() {
    let mut c = cache(16);
    c.reserve_seq(0, 4).unwrap(); // [0, 4) — released below, opening the gap
    let donor = c.reserve_seq(1, 4).unwrap(); // [4, 8)
    let dst = c.reserve_seq(2, 2).unwrap(); // [8, 10)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[4, 5, 6, 7]);
    }
    c.share_prefix(1, 2, 4).unwrap();
    assert_eq!(c.spans_of(2), &[(0, 4, 4), (4, dst.start, 2)]);
    c.release_seq(0);
    let plan = c.compaction_plan(None);
    assert_eq!(plan.len(), 2, "both runs pack down: {plan:?}");
    c.apply_moves(&plan).unwrap();
    let moved_donor = c.seq_slot(1).unwrap();
    let moved_dst = c.seq_slot(2).unwrap();
    assert_eq!(moved_donor.start, 0, "the donor packed to the bottom");
    assert_eq!(moved_dst.start, 4);
    assert_eq!(
        moved_dst.shared.cell, 0,
        "the sharer's prefix followed the donor"
    );
    assert_eq!(c.cell_of(2, 0), Some(0));
    assert_eq!(c.cell_of(2, 4), Some(4), "and its own run moved with it");
    assert_eq!(c.spans_of(2), &[(0, 0, 4), (4, 4, 2)]);
}
/// What sharing cannot express is refused, never approximated: an empty share,
/// a donor that has not written that far, itself, a destination that already
/// shares or has written rows, and a donor prefix that is not one cell range.
#[test]
fn sharing_refuses_what_it_cannot_express() {
    let mut c = cache(16);
    let err = c.share_prefix(1, 2, 4).unwrap_err();
    assert!(err.contains("holds no run"), "got: {err}");
    c.reserve_seq(1, 4).unwrap(); // [0, 4)
    c.reserve_seq(2, 2).unwrap(); // [4, 6)
    c.reserve_seq(3, 2).unwrap(); // [6, 8)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    assert!(c.share_prefix(1, 2, 0).unwrap_err().contains("0 rows"));
    assert!(c
        .share_prefix(1, 2, 5)
        .unwrap_err()
        .contains("has written 4"));
    assert!(c
        .share_prefix(1, 1, 2)
        .unwrap_err()
        .contains("cannot share with itself"));
    c.share_prefix(1, 2, 4).unwrap();
    assert!(c
        .share_prefix(1, 2, 2)
        .unwrap_err()
        .contains("already shares"));
    // A destination that has written rows of its own would have them displaced.
    let own = c.reserve_seq(4, 2).unwrap(); // [8, 10)
    c.own_positions(4, &[0]).unwrap();
    let err = c.share_prefix(1, 4, 2).unwrap_err();
    assert!(err.contains("written 1 rows of its own"), "got: {err}");
    assert_eq!(c.cell_of(4, 0), Some(own.start), "and it still resolves");
    // Sequence 3 shares 4 rows and then writes two of its own at [6, 8):
    // positions [0, 6) are two cell ranges, so a 5-row share from it must
    // refuse instead of pointing the prefix across the gap.
    c.share_prefix(1, 3, 4).unwrap();
    c.own_positions(3, &[4, 5]).unwrap();
    assert_eq!(c.spans_of(3).len(), 2);
    let dst = c.reserve_seq(5, 2).unwrap(); // [10, 12)
    let err = c.share_prefix(3, 5, 5).unwrap_err();
    assert!(err.contains("one"), "got: {err}");
    assert!(err.contains("contiguous"), "got: {err}");
    assert_eq!(
        c.cell_of(5, 0),
        Some(dst.start),
        "the refusal changed nothing"
    );
}
/// C8b S2: the map is `attn_span` kept as a list — one run when a sequence shares
/// nothing, two when it reads a prefix in place, and a refusal (never a truncated
/// window) when the input cannot carry the runs.
#[test]
fn the_map_lists_a_querys_runs_and_refuses_to_truncate() {
    let mut c = cache(16);
    let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    c.share_prefix(1, 2, 4).unwrap();
    c.own_positions(2, &[4, 5]).unwrap();
    let map = c.attn_map(&[2, 2, 2], &[2, 4, 5]).unwrap();
    let k = KV_MAP_MAX_SPANS * 2;
    assert_eq!(
        &map[0..k],
        &[donor.start as u32, 3, 0, 0, 0, 0, 0, 0],
        "only the shared prefix is in reach at position 2"
    );
    assert_eq!(
        &map[k..2 * k],
        &[donor.start as u32, 4, dst.start as u32, 1, 0, 0, 0, 0],
        "position 4 crosses into the private run"
    );
    assert_eq!(
        &map[2 * k..3 * k],
        &[donor.start as u32, 4, dst.start as u32, 2, 0, 0, 0, 0]
    );
    // With one span the map and the span agree, which is what keeps the
    // single-span path bitwise.
    let span = c.attn_span(&[1], &[3]).unwrap();
    let single = c.attn_map(&[1], &[3]).unwrap();
    assert_eq!(&single[0..2], &[span[0], span[1] - span[0]]);
    assert_eq!(&single[2..], &[0u32; 6], "unused slots are zero-length");
    assert!(c
        .attn_map(&[2], &[6])
        .unwrap_err()
        .contains("has not been written"));
    assert!(c
        .attn_map(&[9], &[0])
        .unwrap_err()
        .contains("holds no cells"));
    // More runs than the input carries: written directly, because no store
    // mutation produces five spans yet.
    c.spans.insert(
        1,
        vec![(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1)],
    );
    c.note_written(1, 0, &[4]); // the fifth span's row
    let err = c.attn_map(&[1], &[4]).unwrap_err();
    assert!(err.contains("more than"), "got: {err}");
}
/// A physical removal (C2) renumbers cells across the whole arena, so it cannot
/// run while a prefix is shared — the rows it would move belong to someone else.
#[test]
fn a_shared_prefix_stops_a_physical_removal() {
    let mut c = cache(16);
    let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
    c.reserve_seq(SEQ_MAIN, 4).unwrap(); // [4, 8)
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[0, 1, 2, 3]);
    }
    c.share_prefix(1, SEQ_MAIN, 4).unwrap();
    c.release_seq(1);
    assert_eq!(c.cell_of(SEQ_MAIN, 0), Some(donor.start));
    let err = c.after_shift(1).unwrap_err();
    assert!(err.contains("shares a 4-row prefix"), "got: {err}");
}
