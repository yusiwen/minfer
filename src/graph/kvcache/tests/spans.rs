//! Span resolution: the causal window, disjoint sequences and the span/map arithmetic.
//!
//! Split out of `src/graph/kvcache/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
