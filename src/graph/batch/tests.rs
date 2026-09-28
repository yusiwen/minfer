//! `#[cfg(test)] mod tests` for `src/graph/batch.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn single_is_one_group_on_the_main_sequence() {
    let b = Batch::single(&[1, 2, 3], &[0, 1, 2]);
    assert_eq!(b.n_seqs(), 1);
    assert_eq!(b.groups(), vec![(SEQ_MAIN, 0, 3)]);
    assert!(b.check().is_ok());
    // One sequence: `n_out` selects the tail rows, as before E2.
    assert_eq!(b.out_rows(1), vec![2]);
    assert_eq!(b.out_rows(3), vec![0, 1, 2]);
}

#[test]
fn several_sequences_yield_one_logits_row_each() {
    // Two sequences decoding one token each, in slot order.
    let b = Batch::new(vec![10, 20], vec![3, 7], vec![5, 9]);
    assert_eq!(b.n_seqs(), 2);
    assert_eq!(b.groups(), vec![(5, 0, 1), (9, 1, 2)]);
    assert!(b.check().is_ok());
    assert_eq!(b.out_rows(1), vec![0, 1], "n_out is ignored: one row each");
}

#[test]
fn interleaved_sequences_are_refused() {
    let b = Batch::new(vec![1, 2, 3], vec![0, 3, 1], vec![5, 9, 5]);
    let err = b.check().unwrap_err();
    assert!(err.contains("contiguous"), "got: {err}");
}

#[test]
fn shape_mismatches_are_refused() {
    let b = Batch::new(vec![1, 2], vec![0], vec![5, 5]);
    assert!(b.check().unwrap_err().contains("2 tokens"));
    assert!(Batch::new(vec![], vec![], vec![]).check().is_err());
}
