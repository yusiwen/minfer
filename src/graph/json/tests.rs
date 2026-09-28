//! `#[cfg(test)] mod tests` for `src/graph/json.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// The export/trace preview must offer the decode fusions on CUDA-only runs
/// too — the old Metal-only gate dropped `FusedQKV` from CUDA previews and
/// shifted node ids away from the live events (`viz/README.md`).
#[test]
fn preview_fuse_flags_include_cuda_only_runs() {
    // prefill never fuses, regardless of backends
    assert_eq!(preview_fuse_flags(4, false, false), (false, false));
    assert_eq!(preview_fuse_flags(4, true, true), (false, false));
    // decode + CUDA only: the shipped divergence
    assert_eq!(preview_fuse_flags(1, false, true), (true, true));
    // decode + Metal only, and decode + no GPU
    assert_eq!(preview_fuse_flags(1, true, false), (true, true));
    assert_eq!(preview_fuse_flags(1, false, false), (false, false));
}
