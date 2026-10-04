//! C1/C2 cell identity, ownership and the physical shift/removal.
//!
//! Split out of `src/graph/kvcache/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
