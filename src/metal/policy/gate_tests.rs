//! `#[cfg(test)] mod gate_tests` for `src/metal/policy.rs` — extracted so a
//! non-test build does not parse it. See the parent module for the docs.
//!
//! Device-free and `MINFER_*`-free: these run anywhere the Metal module compiles,
//! with no GPU and no environment mutation.
use super::*;

/// #431: the matvec/GEMM boundary is one authority, pinned to the **measured**
/// crossover on two axes.
///
/// `nt`: the GEMM's 32-row M tile costs 100.8 ms at `nt = 3` where the matvec
/// family costs 58.3 ms, and 103.1 ms at `nt = 5` where the matvec costs 114.4 ms
/// (7B q4_K_M, `specverify`, `_multi` arm) — so the row boundary is 4 for a family
/// with no `grid.y = nt` band.
/// `od`: at `nt = 4..8` a narrow row (0.5B q4_0, `od` 896-1152) still loses
/// through the GEMM by 1.56x (341 vs 533 t/s), because the 64-wide N tile is then
/// the half that under-fills — so the width boundary stays 2048.
#[test]
fn the_gemm_boundary_is_the_measured_crossover() {
    use BandArm::{MultiOnly, SingleGrid};
    // Decode never takes the 32-row tile, whatever the width or the arm.
    for arm in [MultiOnly, SingleGrid] {
        assert!(
            !gemm_preferred(1, 4096, arm),
            "decode must stay on the matvec kernels"
        );
        assert!(!gemm_preferred(1, 152_064, arm));
    }
    // Rows, `_multi` arm: 3 is still matvec, 4 flips (wide rows only).
    assert!(!gemm_preferred(3, 4096, MultiOnly));
    assert!(gemm_preferred(4, 4096, MultiOnly));
    assert_eq!((GEMM_MIN_NT, GEMM_MIN_OD), (4, 2048));
    // Width: a narrow row stays on the matvec family at every nt, including the
    // 2..8 band #431 is about.
    for arm in [MultiOnly, SingleGrid] {
        for nt in 2..=8 {
            assert!(
                !gemm_preferred(nt, 1152, arm),
                "nt = {nt} over a narrow od must stay on the matvec kernels"
            );
        }
        assert!(!gemm_preferred(2, 2047, arm));
        assert!(gemm_preferred(gemm_min_nt(arm), 2048, arm));
        // A full prefill of a wide target takes the GEMM.
        assert!(gemm_preferred(430, 3584, arm));
    }
}

/// The single-grid arm's floor is two rows higher, and that is the whole point of
/// the parameter: `nt = 4, 5` are wins for Q4_K/Q6_K and losses for everyone else.
///
/// The numbers behind it are in `GEMM_MIN_NT_SINGLE_GRID`'s doc table; this pins
/// the *shape* of the decision so a future edit cannot silently move the boundary
/// without its measurement.
#[test]
fn the_single_grid_arm_keeps_the_band_two_rows_longer() {
    use BandArm::{MultiOnly, SingleGrid};
    assert_eq!(gemm_min_nt(MultiOnly), 4);
    assert_eq!(gemm_min_nt(SingleGrid), 6);
    assert_eq!(GEMM_MIN_NT_SINGLE_GRID, 6);
    for nt in 4..6 {
        assert!(
            !gemm_preferred(nt, 4096, SingleGrid),
            "nt = {nt} is a measured win for the single-token arm and must stay matvec"
        );
        assert!(
            gemm_preferred(nt, 4096, MultiOnly),
            "nt = {nt} stays on the GEMM for the `_multi` arm — that is unmeasured territory"
        );
    }
    // From the single-grid floor up, both arms agree.
    for nt in 6..=64 {
        assert_eq!(
            gemm_preferred(nt, 4096, MultiOnly),
            gemm_preferred(nt, 4096, SingleGrid),
            "nt = {nt}"
        );
    }
}

/// For a fixed output width the boundary is monotone in `nt`: no call takes the
/// matvec family after a smaller one took the GEMM. A non-monotone policy is what
/// the old `nt >= 2 && (od >= 2048 || nt >= 9)` expression produced inside the
/// band, and callers cannot reason about that.
#[test]
fn the_boundary_is_monotone_in_nt() {
    for arm in [BandArm::MultiOnly, BandArm::SingleGrid] {
        for od in [512usize, 1152, 2048, 3584, 152_064] {
            let mut seen_gemm = false;
            for nt in 1..=64 {
                let p = gemm_preferred(nt, od, arm);
                assert!(
                    !(seen_gemm && !p),
                    "{arm:?} od = {od}: nt = {nt} regressed to the matvec family after the GEMM had started"
                );
                seen_gemm |= p;
            }
            // A wide row must reach the GEMM somewhere; a narrow one never does.
            assert_eq!(seen_gemm, od >= GEMM_MIN_OD, "{arm:?} od = {od}");
        }
    }
}
