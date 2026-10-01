//! `#[cfg(test)] mod tests` for `src/graph/kvcache.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::Backend;

fn buf(id: usize) -> BufRef {
    BufRef::own(Backend::CPU, id, usize::MAX)
}

fn cache(n_ctx: usize) -> KvCache {
    let mut c = KvCache::new();
    c.insert(0, buf(1), buf(2), 4, 4, n_ctx, false);
    c.insert(1, buf(3), buf(4), 4, 4, n_ctx, false);
    c
}

#[test]
fn starts_identity_and_resolves_positions_verbatim() {
    let c = cache(8);
    assert!(c.is_identity(), "C1 is the identity by construction");
    assert_eq!(c.cells_for(0, &[0, 1, 7]).unwrap(), vec![0, 1, 7]);
    // Every layer resolves to the same cells (one implicit sequence).
    assert_eq!(c.cells_for(1, &[3]).unwrap(), vec![3]);
}

#[test]
fn a_position_past_the_arena_is_an_error() {
    let c = cache(8);
    let err = c.cells_for(0, &[8]).unwrap_err();
    assert!(err.contains(">= n_ctx 8"), "got: {err}");
    let err = c.cells_for(9, &[0]).unwrap_err();
    assert!(err.contains("no KV arena"), "got: {err}");
}

#[test]
fn ownership_follows_the_written_cells() {
    let c0 = cache(4);
    // Nothing written yet: every cell is free.
    assert!(c0.get(0).unwrap().owner.iter().all(|&o| o == FREE));
    assert_eq!(c0.get(0).unwrap().n_used, 0);

    let mut c = cache(4);
    let cells = c.cells_for(0, &[0, 1]).unwrap();
    c.note_written(SEQ_MAIN, 0, &cells);
    assert_eq!(
        c.get(0).unwrap().owner,
        vec![SEQ_MAIN, SEQ_MAIN, FREE, FREE]
    );
    assert_eq!(c.get(0).unwrap().n_used, 2, "used prefix tracks the writes");
    // note_written is per layer: layer 1 is untouched.
    assert!(c.get(1).unwrap().owner.iter().all(|&o| o == FREE));
}

#[test]
fn own_prefix_round_trip() {
    let mut c = cache(4);
    c.own_prefix(SEQ_MAIN, 3);
    assert_eq!(
        c.get(0).unwrap().owner,
        vec![SEQ_MAIN; 3]
            .into_iter()
            .chain([FREE])
            .collect::<Vec<_>>()
    );
}

#[test]
fn clearing_identity_is_the_backend_gate() {
    let mut c = cache(4);
    assert!(c.is_identity());
    c.clear_identity();
    assert!(
        !c.is_identity(),
        "once false, positions are no longer cells"
    );
}

#[test]
fn after_shift_renumbers_and_frees_the_tail() {
    let mut c = cache(4);
    c.own_prefix(SEQ_MAIN, 4);
    assert_eq!(c.after_shift(1).unwrap(), 3, "new n_used");
    for (_, l) in c.iter() {
        assert_eq!(l.n_used, 3);
        assert_eq!(l.owner, vec![SEQ_MAIN, SEQ_MAIN, SEQ_MAIN, FREE]);
    }
    // Removing past the end is an error, not a silent truncation.
    let err = c.after_shift(4).unwrap_err();
    assert!(err.contains("cannot remove 4 rows at 0 of 3"), "got: {err}");
    // A physical shift keeps the identity mapping: no backend hand-off.
    assert!(c.is_identity());
}

#[test]
fn after_rm_removes_a_middle_range_and_frees_the_tail() {
    let mut c = cache(6);
    c.own_prefix(SEQ_MAIN, 6);
    // Drop cells 2..4: [0,1] stay, [4,5] slide to [2,3], the tail frees.
    assert_eq!(c.after_rm(2, 2).unwrap(), 4, "new n_used");
    for (_, l) in c.iter() {
        assert_eq!(l.n_used, 4);
        assert_eq!(
            l.owner,
            vec![SEQ_MAIN, SEQ_MAIN, SEQ_MAIN, SEQ_MAIN, FREE, FREE]
        );
    }
    // Removing every written row empties the arena.
    assert_eq!(c.after_rm(0, 4).unwrap(), 0);
    for (_, l) in c.iter() {
        assert_eq!(l.n_used, 0);
        assert!(l.owner.iter().all(|&o| o == FREE));
    }
    assert!(c.is_identity(), "a physical removal keeps cell == pos");
}

/// E1: a single sequence's window is the one positions used to derive —
/// `[0, pos + 1)` — which is why the kernel change stays bitwise.
#[test]
fn a_single_sequence_resolves_to_the_causal_window() {
    let mut c = cache(8);
    c.own_prefix(SEQ_MAIN, 4);
    c.note_written(SEQ_MAIN, 0, &[0, 1, 2, 3]);
    // Query at position 2 may see cells 0..3; the last query sees 0..4.
    assert_eq!(
        c.attn_span(&[SEQ_MAIN, SEQ_MAIN], &[2, 3]).unwrap(),
        vec![0, 0, 3, 4],
        "lo row then hi row"
    );
}

/// E1's acceptance at the store level, on E2's reservations: two sequences
/// in one arena get two disjoint windows, so no query can see the other's
/// cells.
#[test]
fn two_sequences_resolve_to_disjoint_windows() {
    let mut c = cache(8);
    // Sequence 0 reserves cells 0..3, sequence 1 the next free run of 3.
    let a = c.reserve_seq(SEQ_MAIN, 3).unwrap();
    let b = c.reserve_seq(1, 3).unwrap();
    assert_eq!((a.start, a.cap), (0, 3));
    assert_eq!((b.start, b.cap), (3, 3), "first-fit after the first run");
    // Each sequence owns the rows it wrote (E1's window cap is the
    // reservation; the written-row check keeps unwritten rows out).
    for layer in [0usize, 1] {
        c.note_written(SEQ_MAIN, layer, &[0, 1, 2]);
        c.note_written(1, layer, &[3, 4, 5]);
    }
    assert_eq!(c.seq_range(0, SEQ_MAIN).unwrap(), Some((0, 3)));
    assert_eq!(c.seq_range(0, 1).unwrap(), Some((3, 3)));
    assert_eq!(c.seq_range(1, 1).unwrap(), Some((3, 3)), "layers agree");
    // A token of each sequence, each at its own last position.
    // C6: positions are sequence-relative; cell = start + position.
    let span = c.attn_span(&[SEQ_MAIN, 1], &[2, 2]).unwrap();
    assert_eq!(&span[..2], &[0, 3], "starts are each sequence's own");
    assert_eq!(&span[2..], &[3, 6], "ends are exclusive and causal");
    assert!(
        span[2] >= span[1],
        "sequence 1's window starts where sequence 0's ends"
    );
}

/// E2's reservations: disjoint, first-fit, released, and loud when the arena
/// cannot hold a sequence (no silent overlap, no moving).
#[test]
fn reservations_are_disjoint_first_fit_and_releasable() {
    let mut c = cache(8);
    assert_eq!(c.reserve_seq(0, 3).unwrap().start, 0);
    assert_eq!(c.reserve_seq(1, 3).unwrap().start, 3);
    // An existing reservation is returned unchanged when it fits.
    assert_eq!(c.reserve_seq(0, 2).unwrap().start, 0);
    // …and growth is refused, not silently relocated.
    let err = c.reserve_seq(0, 4).unwrap_err();
    assert!(err.contains("no growth"), "got: {err}");
    // The arena is full: 2 free cells cannot hold a third sequence of 3.
    let err = c.reserve_seq(2, 3).unwrap_err();
    assert!(err.contains("no free run of 3"), "got: {err}");
    // Releasing frees exactly its cells, and the next sequence reuses them.
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[3, 4, 5]);
    }
    assert_eq!(c.release_seq(1), 3);
    assert_eq!(c.seq_range(0, 1).unwrap(), None);
    assert!(
        c.get(0).unwrap().owner[3..6].iter().all(|&o| o == FREE),
        "released cells lose their owner"
    );
    assert_eq!(c.reserve_seq(2, 3).unwrap().start, 3, "freed run reused");
    assert_eq!(
        c.release_seq(7),
        0,
        "releasing an unknown sequence is a no-op"
    );
}

/// C8b S1: with one span per sequence, the span list must resolve exactly like the
/// run it describes — and it must follow the run when the run moves, or when it goes
/// away. A miss here is what the resolver turns into a loud error.
#[test]
fn a_single_span_resolves_exactly_like_its_run() {
    let mut c = cache(16);
    c.reserve_seq(1, 4).unwrap(); // [0, 4)
    let slot = c.reserve_seq(0, 4).unwrap(); // [4, 8)
    assert_eq!(c.spans_of(0), &[(0, slot.start, 4)]);
    for pos in 0..4 {
        assert_eq!(c.cell_of(0, pos), Some(slot.start + pos), "pos {pos}");
    }
    assert_eq!(c.cell_of(0, 4), None, "past the run");
    assert_eq!(c.spans_of(9), &[] as &[(usize, usize, usize)], "no run");

    // A compaction moves the run down into the gap sequence 1 left, and the span
    // list has to follow it (this is the maintenance point that would otherwise
    // drift silently).
    c.release_seq(1);
    let plan = c.compaction_plan(None);
    assert!(!plan.is_empty(), "sequence 0 must move down: {plan:?}");
    c.apply_moves(&plan).unwrap();
    let moved = c.seq_slot(0).unwrap().start;
    assert_eq!(moved, 0, "packed to the bottom");
    assert_eq!(c.spans_of(0), &[(0, 0, 4)]);
    assert_eq!(c.cell_of(0, 3), Some(3));

    // Growing republishes it, and releasing drops it.
    assert_eq!(c.set_cap(0, 6).unwrap(), Vec::new());
    assert_eq!(c.spans_of(0), &[(0, 0, 6)]);
    c.release_seq(0);
    assert_eq!(c.cell_of(0, 0), None);
}

/// C8b S1b: the **read** path resolves through the same span list the write path
/// uses, and with one span it must produce exactly the range the old `start +
/// position` arithmetic produced. The two are compared here rather than trusted,
/// because this is the change that would silently shift an attention window.
#[test]
fn a_single_span_attn_window_equals_the_run_arithmetic() {
    let mut c = cache(16);
    // A run that does not start at cell 0, so `cell == position` cannot hide a
    // regression in the resolution.
    c.reserve_seq(7, 3).unwrap(); // [0, 3)
    let a = c.reserve_seq(SEQ_MAIN, 4).unwrap(); // [3, 7)
    let b = c.reserve_seq(1, 2).unwrap(); // [7, 9)
    assert_eq!((a.start, b.start), (3, 7));
    for layer in [0usize, 1] {
        c.note_written(SEQ_MAIN, layer, &[3, 4, 5, 6]);
        c.note_written(1, layer, &[7, 8]);
    }
    for positions in [vec![0], vec![0, 1, 2, 3], vec![3, 3, 0], vec![1, 0]] {
        let seqs = vec![SEQ_MAIN; positions.len()];
        let span = c.attn_span(&seqs, &positions).unwrap();
        let n = positions.len();
        for (t, &rel) in positions.iter().enumerate() {
            assert_eq!(
                (span[t], span[n + t]),
                (a.start as u32, (a.start + rel + 1) as u32),
                "position {rel} of sequence {SEQ_MAIN}"
            );
        }
    }
    // Two sequences still get their own disjoint windows through the span list.
    assert_eq!(
        c.attn_span(&[SEQ_MAIN, 1], &[2, 1]).unwrap(),
        vec![3, 7, 6, 9]
    );
    // A sequence whose span list is gone is a loud error, never a window at 0.
    c.release_seq(SEQ_MAIN);
    let err = c.attn_span(&[SEQ_MAIN], &[0]).unwrap_err();
    assert!(err.contains("holds no cells"), "got: {err}");
}

/// C8b S1b's boundary: a multi-span sequence has no single contiguous window to
/// hand the kernel, so the read path must refuse it instead of guessing a span.
/// The span list is written directly here — no mutation produces several spans
/// yet, that is S2 — so the test pins the refusal before the layout exists.
#[test]
fn a_multi_span_sequence_is_refused_by_the_read_path() {
    let mut c = cache(16);
    c.reserve_seq(SEQ_MAIN, 4).unwrap();
    for layer in [0usize, 1] {
        c.note_written(SEQ_MAIN, layer, &[0, 1, 2, 3]);
    }
    // A shared prefix would look like this: rows 0..2 at cell 0, the rest private
    // at cell 8 — two ranges, no single `[lo, hi)` window.
    let mut slot = c.seq_slot(SEQ_MAIN).unwrap();
    slot.start = 8;
    c.seqs.insert(SEQ_MAIN, slot);
    c.spans.insert(SEQ_MAIN, vec![(0, 0, 2), (2, 8, 2)]);
    assert_eq!(
        c.cell_of(SEQ_MAIN, 3),
        Some(9),
        "the resolver still reads it"
    );
    let err = c.attn_span(&[SEQ_MAIN], &[3]).unwrap_err();
    assert!(err.contains("2 spans"), "got: {err}");
    assert!(err.contains("kv_map"), "got: {err}");
}

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

/// C7b: growing a run moves the runs above it **up**, and their ownership
/// travels with the rows. This is the direction CUDA's row-move kernel could not
/// do before, and the plan — not the caller — decides it.
#[test]
fn growing_a_run_pushes_the_runs_above_it_up() {
    let mut c = cache(16);
    assert_eq!(c.reserve_seq(0, 4).unwrap().start, 0);
    assert_eq!(c.reserve_seq(1, 4).unwrap().start, 4);
    assert_eq!(c.reserve_seq(2, 4).unwrap().start, 8);
    for layer in [0usize, 1] {
        c.note_written(1, layer, &[4, 5, 6]);
        c.note_written(2, layer, &[8, 9]);
    }
    // Growing the *lowest* run pushes both runs above it up by two.
    let plan = c.set_cap(0, 6).unwrap();
    assert_eq!(plan.len(), 2, "both runs above move: {plan:?}");
    assert!(
        plan.iter().all(|m| m.to > m.from),
        "both moves are upward: {plan:?}"
    );
    assert_eq!(c.seq_slot(0).unwrap().cap, 6);
    c.apply_moves(&plan).unwrap();
    assert_eq!(c.seq_slot(1).unwrap().start, 6);
    assert_eq!(c.seq_slot(2).unwrap().start, 10);
    let owner = |cell: usize| c.get(0).unwrap().owner[cell];
    assert_eq!(
        (owner(6), owner(7), owner(8)),
        (1, 1, 1),
        "run 1's three rows"
    );
    assert_eq!((owner(10), owner(11)), (2, 2), "run 2's two rows");
    for cell in [4usize, 5, 12, 13] {
        assert_eq!(owner(cell), FREE, "cell {cell} is vacant after the move");
    }
}

/// C7b: the two refusals that keep a resize from trading data or truth away.
#[test]
fn a_resize_refuses_to_eat_rows_or_overcommit_the_arena() {
    let mut c = cache(8);
    c.reserve_seq(0, 3).unwrap();
    c.reserve_seq(1, 3).unwrap();
    for layer in [0usize, 1] {
        c.note_written(0, layer, &[0, 1]);
    }
    // Shrinking below the rows a sequence has written would lose them.
    let err = c.set_cap(0, 1).unwrap_err();
    assert!(err.contains("cannot shrink"), "got: {err}");
    // Growing past what the arena holds with the other reservations is refused
    // *before* the table changes, so there is nothing to undo.
    let err = c.set_cap(0, 6).unwrap_err();
    assert!(err.contains("reserved elsewhere"), "got: {err}");
    assert_eq!(c.seq_slot(0).unwrap().cap, 3, "the refusal left it alone");
    // What does fit (3 + 5 = 8) is allowed, and it moves the run above up.
    let plan = c.set_cap(0, 5).unwrap();
    assert_eq!(plan.len(), 1, "run 1 moves: {plan:?}");
    assert_eq!((plan[0].from, plan[0].to), (3, 5));
    assert_eq!(c.seq_slot(0).unwrap().cap, 5);
}

/// A window the resolver cannot justify must be loud, not a wrong bound.
#[test]
fn unwritten_rows_and_unknown_sequences_are_errors() {
    // No reservation for the sequence at all.
    let mut c = cache(8);
    let err = c.attn_span(&[1], &[0]).unwrap_err();
    assert!(err.contains("holds no cells"), "got: {err}");

    // A query outside its reserved run, and a row inside the run that was
    // never written (it would attend to zeroes).
    let mut c = cache(8);
    c.reserve_seq(SEQ_MAIN, 2).unwrap();
    let err = c.attn_span(&[SEQ_MAIN], &[5]).unwrap_err();
    assert!(err.contains("outside its reserved run"), "got: {err}");
    let err = c.attn_span(&[SEQ_MAIN], &[1]).unwrap_err();
    assert!(err.contains("not written"), "got: {err}");
    // Once written, the same query resolves.
    for layer in [0usize, 1] {
        c.note_written(SEQ_MAIN, layer, &[0, 1]);
    }
    assert_eq!(c.attn_span(&[SEQ_MAIN], &[1]).unwrap(), vec![0, 2]);
}

/// A shift must reproduce "the same tokens, roped at their new positions".
/// It cannot be bitwise (two composed rotations vs one), so this pins both
/// the equivalence and the size of the difference — C2's tolerance class.
#[test]
fn rope_shift_matches_roping_at_the_new_position() {
    let rope = KvRope {
        freq_base: 10_000.0,
        freq_scale: 1.0,
        n_head_kv: 2,
        hd: 4,
        style: RopeStyle::NonInterleaved,
    };
    // Rope a row at `pos` exactly the way the kernels do.
    fn rope_at(x: &mut [f32], pos: usize, rope: &KvRope) {
        let half = rope.hd / 2;
        for h in 0..rope.n_head_kv {
            let b = h * rope.hd;
            for i in 0..half {
                let f = rope.freq_scale / rope.freq_base.powf((2 * i) as f32 / rope.hd as f32);
                let (sn, cs) = (pos as f32 * f).sin_cos();
                let (i0, i1) = (b + i, b + i + half);
                let (x0, x1) = (x[i0], x[i1]);
                x[i0] = x0 * cs - x1 * sn;
                x[i1] = x0 * sn + x1 * cs;
            }
        }
    }

    let base: Vec<f32> = (0..(rope.n_head_kv * rope.hd))
        .map(|i| (i as f32 + 1.0) * 0.25)
        .collect();
    let pos = 7;
    let delta = 3;

    let mut shifted = base.clone();
    rope_at(&mut shifted, pos, &rope);
    rope_shift_kv(&mut shifted, 1, delta, &rope);

    let mut reference = base.clone();
    rope_at(&mut reference, pos - delta as usize, &rope);

    let worst = shifted
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = reference.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
    eprintln!(
        "[c2] rope-shift tolerance class: max |Δ| = {worst} (relative {})",
        worst / scale
    );
    assert!(
        worst < 1e-5,
        "shift must land on the new-position rope, got |Δ| = {worst}"
    );
    // delta = 0 must be a no-op.
    let mut untouched = base.clone();
    rope_at(&mut untouched, pos, &rope);
    let mut same = untouched.clone();
    rope_shift_kv(&mut same, 1, 0, &rope);
    assert_eq!(same, untouched);
}

/// Reserve `seq` a `cap`-cell run and take ownership of all of it, so the
/// run has a live prefix a compaction must carry.
fn place(c: &mut KvCache, seq: SeqId, cap: usize) -> usize {
    let slot = c.reserve_seq(seq, cap).unwrap();
    c.own_range(seq, slot.start, slot.start + cap);
    slot.start
}

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

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `kvcache.rs` (bucket B of the dead-code census —
// every test caller already lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl KvCache {
    /// The single-sequence case: reserve the whole arena for `seq` if it has no
    /// reservation yet, then take ownership of `0..n_used`.
    ///
    /// **Test-only (#239).** Its last production-looking caller, `GraphAllocator::kv_note_used`,
    /// was E1's C1 remnant and was deleted with E1 (#228): production now records the
    /// written extent per position through [`Self::own_positions`], which is
    /// reservation-aware. Driven by `kvcache::tests::own_prefix_and_release_round_trip`
    /// and its three siblings (`after_shift_renumbers_and_frees_the_tail`,
    /// `after_rm_removes_a_middle_range_and_frees_the_tail`,
    /// `a_single_sequence_resolves_to_the_causal_window`), which use it to set up a
    /// written prefix directly.
    pub fn own_prefix(&mut self, seq: SeqId, n_used: usize) {
        if !self.seqs.contains_key(&seq) {
            let cap = self.n_ctx;
            if let Err(e) = self.reserve_seq(seq, cap) {
                debug_assert!(false, "own_prefix: {e}");
                return;
            }
        }
        self.own_range(seq, 0, n_used);
    }

    /// Record that `rows` were written at the given resolved cells by `seq`.
    ///
    /// Test-only (#239): driven by `kvcache::tests::ownership_follows_the_written_cells`
    /// and the C2/C8b gates in this file (19 tests in all).
    ///
    /// Note: its `#[allow(dead_code)]` comment claimed an "E2 surface"; the E2 batched path
    /// records written extents through `own_positions` / `fill_batch_inputs` instead, so that
    /// surface never arrived. Recorded for [#244](https://github.com/yusiwen/minfer/issues/244).
    pub fn note_written(&mut self, seq: SeqId, layer: usize, cells: &[u32]) {
        if let Some(l) = self.layers.get_mut(&layer) {
            for &c in cells {
                let c = c as usize;
                if c < l.owner.len() {
                    l.owner[c] = seq;
                    l.n_used = l.n_used.max(c + 1);
                }
            }
        }
        // C8b S2: a recorded row is a written position, whatever the cell holds.
        let mut written = self.seqs.get(&seq).map_or(0, |s| s.written);
        for &c in cells {
            if let Some(pos) = self.pos_of_cell(seq, c as usize) {
                written = written.max(pos + 1);
            }
        }
        if let Some(slot) = self.seqs.get_mut(&seq) {
            slot.written = written;
        }
    }

    /// Sliding-window special case of [`KvCache::after_rm`]: drop the oldest
    /// `drop` rows, so every survivor's position decreases by `drop`.
    ///
    /// Test-only (#239): driven by `kvcache::tests::after_shift_renumbers_and_frees_the_tail`
    /// and `kvcache::tests::a_shared_prefix_stops_a_physical_removal`.
    pub fn after_shift(&mut self, drop: usize) -> Result<usize, String> {
        self.after_rm(0, drop)
    }

    /// The cell range a sequence owns in `layer`, as `(start, len)`; `None` when
    /// it owns nothing.
    ///
    /// Ownership is contiguous by construction — a sequence's cells are written
    /// in position order and a removal slides the survivors down (C2) — so one
    /// range describes it completely. A gap is a bug rather than a supported
    /// layout, and this returns `Err` instead of letting attention bound itself
    /// to the wrong window; a per-cell mask is what a hole-creating layout would
    /// need (C3/D1).
    ///
    /// Test-only (#239): driven by `kvcache::tests::two_sequences_resolve_to_disjoint_windows`
    /// and `kvcache::tests::reservations_are_disjoint_first_fit_and_releasable`.
    pub fn seq_range(&self, layer: usize, seq: SeqId) -> Result<Option<(usize, usize)>, String> {
        if !self.layers.contains_key(&layer) {
            return Err(format!("no KV arena for layer {layer}"));
        }
        Ok(self.seqs.get(&seq).map(|s| (s.start, s.cap)))
    }

    /// Rows `seq` wrote into its **own** run — the rows a relocation copies, since
    /// the shared prefix is another sequence's memory (C8b S2).
    ///
    /// Test-only (#239): driven by `kvcache::tests::a_shared_prefix_is_read_in_place_and_written_past`,
    /// `a_store_inside_a_shared_prefix_takes_a_private_row` and
    /// `a_copy_on_write_can_shrink_the_share_twice`.
    pub fn private_written(&self, seq: SeqId) -> usize {
        self.seqs.get(&seq).map_or(0, |s| s.private_written())
    }
}
