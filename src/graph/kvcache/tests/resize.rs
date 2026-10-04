//! Run growth/shrink and the arena bounds.
//!
//! Split out of `src/graph/kvcache/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
