//! `#[cfg(test)] mod route_tests` for `src/metal/policy.rs` — extracted so a
//! non-test build does not parse it. See the parent module for the docs.
use super::*;

/// #310: the whole packed routing matrix. `stage_forced` is an explicit
/// parameter, so the A/B arm is gated with no environment mutation and no
/// Metal device — this is the CI-visible half of the dispatch decision.
#[test]
fn packed_route_covers_the_matrix() {
    // Decode (nt == 1), the flash shapes: native, or staged when forced.
    assert_eq!(
        packed_attn_route(64, 1, false, false),
        PackedRoute::NativeDecode
    );
    assert_eq!(
        packed_attn_route(128, 1, false, false),
        PackedRoute::NativeDecode
    );
    assert_eq!(
        packed_attn_route(64, 1, false, true),
        PackedRoute::StageThenFlash
    );
    assert_eq!(
        packed_attn_route(128, 1, false, true),
        PackedRoute::StageThenFlash
    );
    // A non-flash hd keeps the classic kernel even with the stage forced.
    assert_eq!(packed_attn_route(80, 1, false, false), PackedRoute::Classic);
    assert_eq!(packed_attn_route(80, 1, false, true), PackedRoute::Classic);

    // Causal prefill, the flash shapes: staged prefill; odd/small hd classic.
    assert_eq!(
        packed_attn_route(64, 8, false, false),
        PackedRoute::StageThenPrefill
    );
    assert_eq!(
        packed_attn_route(128, 197, false, false),
        PackedRoute::StageThenPrefill
    );
    assert_eq!(packed_attn_route(80, 8, false, false), PackedRoute::Classic);

    // Explicit window, prefill only: staged window; nt == 1 stays classic
    // (the explicit-span fast family is `nt > 1` only).
    assert_eq!(
        packed_attn_route(64, 8, true, false),
        PackedRoute::StageThenWindow
    );
    assert_eq!(
        packed_attn_route(128, 197, true, false),
        PackedRoute::StageThenWindow
    );
    assert_eq!(packed_attn_route(64, 1, true, false), PackedRoute::Classic);
    assert_eq!(packed_attn_route(80, 8, true, false), PackedRoute::Classic);
}
