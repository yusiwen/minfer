//! `#[cfg(test)] mod tests` for `src/device_tier.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn encodes_llama_keys_from_runtime_cc() {
    // minfer runtime cc (major*100 + minor) -> llama.cpp tier key.
    assert_eq!(llama_key(1201), 1210); // GB10 sm_12.1
    assert_eq!(llama_key(1200), 1200); // sm_12.0
    assert_eq!(llama_key(809), 890); // Ada
    assert_eq!(llama_key(807), 870); // Orin
    assert_eq!(llama_key(806), 860); // Ampere
    assert_eq!(llama_key(705), 750); // Turing
    assert_eq!(llama_key(800), 800); // sm_8.0
    assert_eq!(llama_key(0), 0); // no device
}

#[test]
fn resolves_exact_rows() {
    let s = select(1201);
    assert_eq!(s.tier.name, "DGX Spark GB10");
    assert!(s.mmq_available);
    assert_eq!(s.tier.provenance, Provenance::Measured);

    assert_eq!(select(809).tier.name, "Ada (RTX 4090/4080/4070)");
    assert_eq!(select(807).tier.name, "Jetson Orin (Nano/NX/AGX)");
    assert_eq!(select(806).tier.name, "Ampere (RTX 3090/3080/3070/3060)");

    let t = select(705);
    assert_eq!(t.tier.name, "Turing (RTX 2080/2060)");
    assert!(!t.mmq_available); // ruling #4: conservative gate
}

#[test]
fn resolves_family_inheritance_and_generic() {
    // Hypothetical sm_10.0 (cc 1000 -> key 1000): >= 800 family -> Ampere.
    let s = select(1000);
    assert_eq!(s.tier.name, "Ampere (RTX 3090/3080/3070/3060)");
    assert!(s.mmq_available);
    // Hypothetical sm_12.2 (cc 1202 -> key 1220): >= 1200 -> Blackwell.
    assert_eq!(
        select(1202).tier.name,
        "Blackwell consumer (RTX 5090/5080/5070)"
    );
    // Pascal sm_6.1 (cc 601 -> key 610): no family -> GENERIC, no MMQ.
    let g = select(601);
    assert_eq!(g.tier.name, "GENERIC (unknown device)");
    assert!(!g.mmq_available);
    assert_eq!(g.tier.provenance, Provenance::Generic);
    // Unknown >= 800 architecture: GENERIC row, conservative gate stays on.
    assert!(select(803).mmq_available);
}

#[test]
fn forced_override_takes_llama_keys() {
    let s = select_forced(1210);
    assert_eq!(s.tier.name, "DGX Spark GB10");
    assert_eq!(select_forced(890).tier.name, "Ada (RTX 4090/4080/4070)");
    assert!(!select_forced(750).mmq_available);
}

#[test]
fn per_type_batch_overrides_match_source_tables() {
    let gb10 = select(1201).tier;
    for class in [QClass::K4, QClass::K5, QClass::K6, QClass::Other] {
        assert_eq!(mmvq_batch_limit(gb10, class), 8, "GB10 is 8 everywhere");
    }
    let blackwell = select(1200).tier;
    assert_eq!(mmvq_batch_limit(blackwell, QClass::K4), 5);
    assert_eq!(mmvq_batch_limit(blackwell, QClass::K5), 6);
    assert_eq!(mmvq_batch_limit(blackwell, QClass::K6), 7);
    assert_eq!(mmvq_batch_limit(blackwell, QClass::Other), 8);
    let orin = select(807).tier;
    for class in [QClass::K4, QClass::K5, QClass::K6] {
        assert_eq!(mmvq_batch_limit(orin, class), 1, "Orin k-quants cap at 1");
    }
    assert_eq!(mmvq_batch_limit(orin, QClass::Other), 8);
}

#[test]
fn caps_clamp_to_the_identity_bound() {
    // No tier data may exceed the doc-95 bitwise bound.
    for tier in TIERS {
        for class in [QClass::K4, QClass::K5, QClass::K6, QClass::Other] {
            assert!(mmvq_cap(tier, class) <= IDENTITY_BATCH_BOUND);
            assert!(mmvq_cap(tier, class) >= 1);
        }
    }
    assert_eq!(IDENTITY_BATCH_BOUND, 8);
}

#[test]
fn every_row_cites_a_source() {
    for tier in TIERS {
        assert!(
            !tier.source.is_empty(),
            "{} row lacks provenance",
            tier.name
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `device_tier.rs` (bucket B of the dead-code census —
// the only test callers live in this module's subtree; `mmvq_batch_limit`'s only
// other reader was `mmvq_cap`, which moved with it).
// ────────────────────────────────────────────────────────────────────────────

/// doc-95 bitwise bound for the speculative verify path. The batch limit fed
/// to any dispatch arm must never exceed this — the greedy identity chain
/// (multi-MMVQ bitwise vs nt=1 decode) is proven only up to 8.
///
/// Test-only (#239): driven by `device_tier::tests::caps_clamp_to_the_identity_bound`
/// (and asserted literally at 8).
pub const IDENTITY_BATCH_BOUND: i32 = 8;

/// Batch limit for a quant class on a tier (default or per-type override).
///
/// Test-only (#239): driven by
/// `device_tier::tests::per_type_batch_overrides_match_source_tables`.
pub fn mmvq_batch_limit(tier: &DeviceTier, class: QClass) -> i32 {
    tier.mmvq_batch_by_type
        .iter()
        .find(|(c, _)| *c == class)
        .map(|(_, v)| *v)
        .unwrap_or(tier.mmvq_batch_default)
}

/// Dispatch cap for a quant class: the tier batch limit clamped by the
/// doc-95 identity bound. The bound always wins — no tier data may strip the
/// multi-MMVQ family above its own limit (the speculative verify path and
/// the identity battery depend on it, plan §5.4/R3).
///
/// Activation note (plan §14 R8): the dispatch arms do not consume this cap
/// yet — with the current dispatch structure a limit < 8 has no destination
/// for the vacated nt range (it would fall to the f32 fallbacks, a likely
/// pessimization) and would strip the spec identity family (R3). The cap
/// activates together with a small-nt BT destination (T2 tile candidates) or
/// field evidence; the table data and this function are ready and tested.
///
/// Test-only (#239): driven by
/// `device_tier::tests::caps_clamp_to_the_identity_bound`.
pub fn mmvq_cap(tier: &DeviceTier, class: QClass) -> i32 {
    mmvq_batch_limit(tier, class).min(IDENTITY_BATCH_BOUND)
}
