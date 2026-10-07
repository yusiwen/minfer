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

/// C8b S4 / #362: `gathers_attn_map` is the single authority for the set-valued
/// window, and every backend now answers yes — the CPU and CUDA kernels always
/// did, and Metal's sibling `kernel_gqa_attn_map_f32/_f16` closed the gap this
/// ticket is about. Pure, so CI (no GPU) covers it.
#[test]
fn every_device_gathers_the_attn_map() {
    for d in [Device::Cpu, Device::Metal, Device::Cuda] {
        assert!(d.gathers_attn_map(), "{d:?} must gather a kv_map window");
    }
}
