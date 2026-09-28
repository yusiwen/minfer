//! `#[cfg(test)] mod tests` for `src/models/mod.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// D3: the gate's whole matrix, with no device — the same reason E6 made
/// `batch_mode` pure: CI has no GPU, and this decides which graph a GPU run
/// builds.
#[test]
fn ffn_composition_is_opt_in_and_refuses_where_it_cannot_work() {
    // Default: the hand-written node, on every device.
    for d in [Device::Cpu, Device::Metal, Device::Cuda] {
        assert!(!ffn_composition(None, d), "{d:?}");
    }
    // Forced on: allowed where offset views exist (CUDA), refused elsewhere
    // (loudly, in the function) so a Metal graph is never built with a
    // partial window it cannot express.
    assert!(ffn_composition(Some("1"), Device::Cuda));
    assert!(!ffn_composition(Some("1"), Device::Metal));
    assert!(!ffn_composition(Some("1"), Device::Cpu));
    // Forced off, and anything unrecognised, keeps the default.
    for v in ["0", "true", "banana", ""] {
        for d in [Device::Cpu, Device::Metal, Device::Cuda] {
            assert!(!ffn_composition(Some(v), d), "{v:?} {d:?}");
        }
    }
}
