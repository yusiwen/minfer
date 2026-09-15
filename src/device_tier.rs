//! Device tier table + selector — the Device Adaptation Layer (T-series).
//!
//! Design record: `docs/DEVICE-ADAPTATION-PLAN.md`. This module is pure data
//! and pure functions: no CUDA calls, no unsafe, no I/O — fully testable
//! offline. `cuda.rs` depends on it one-way; never the reverse.
//!
//! Adoption principle (ruled 2026-09-14): knobs measured at parity-or-better
//! vs llama.cpp on a device keep minfer's values (`Provenance::Measured`);
//! devices minfer has never been measured on adopt llama.cpp's
//! community-calibrated tiers (`Provenance::Adopted`); unknown devices fall
//! through to the generic convention (`Provenance::Generic`).
//!
//! Encoding note: minfer's runtime `cc` is `major*100 + minor` (GB10 = 1201),
//! while llama.cpp's tier keys are `major*100 + minor*10` (GB10 = 1210). The
//! table keys use the llama.cpp encoding so adopted rows map 1:1 onto their
//! source constants; [`select`] converts minfer's runtime cc internally.

/// Hardware vendor for a tier key. The key space is vendor-namespaced from
/// day one (llama.cpp offset scheme: AMD `0x1000000`, Moore Threads
/// `0x0100000`) so foreign rows can be added without restructuring; only
/// NVIDIA rows exist today.
#[allow(dead_code)] // the namespace is the design; rows arrive per backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
    Mthreads,
    Apple,
    Unknown,
}

/// Vendor-namespaced capability key. `cap_key` semantics per vendor:
/// Nvidia — llama.cpp-style cc (major*100 + minor*10, e.g. 1210 = GB10);
/// Amd/Mthreads — offset-prefixed gfx key; Apple — chip generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceKey {
    pub vendor: Vendor,
    pub cap_key: i32,
}

/// Where a tier's numbers come from — the data-model encoding of the
/// adoption principle. Drives future Adopted → Measured promotion when
/// field data (RTX 2080 Ti / Jetson Orin Nano, plan §9.1/§9.2) arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// minfer measured on this device; parity-or-better vs llama.cpp verified.
    Measured,
    /// Adopted from llama.cpp community tables; never run on minfer.
    Adopted,
    /// Fallback convention for unknown devices.
    Generic,
}

/// Quant class for per-type batch overrides. Only the K-quant families carry
/// overrides in llama.cpp's tables (their k-quant MMVQ dequant cost is per
/// column, so MMQ wins sooner); every other supported type uses the default.
#[allow(dead_code)] // Other is consumed by callers at cap-activation time
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QClass {
    K4,
    K5,
    K6,
    Other,
}

/// One device tier: the dispatch parameters a cc resolves to.
#[allow(dead_code)] // `source` is provenance documentation + test-checked
pub struct DeviceTier {
    pub key: DeviceKey,
    pub name: &'static str,
    /// Provenance: minfer doc or llama.cpp file:line — every row cites its
    /// source so future syncs and promotions have an anchor.
    pub source: &'static str,
    pub provenance: Provenance,
    /// Batch limit for the quantized decode (MMVQ-family) kernels.
    pub mmvq_batch_default: i32,
    /// Per-quant-class overrides (sparse; K-quant only in the source data).
    pub mmvq_batch_by_type: &'static [(QClass, i32)],
    /// int8 BT (block-tile tensor-core) prefill availability. The GENERIC
    /// row encodes `false` here; [`select`] replaces it with `cc >= 800`
    /// (the conservative minfer gate — plan §9 ruling #4).
    pub mmq_available: bool,
}

/// doc-95 bitwise bound for the speculative verify path. The batch limit fed
/// to any dispatch arm must never exceed this — the greedy identity chain
/// (multi-MMVQ bitwise vs nt=1 decode) is proven only up to 8.
pub const IDENTITY_BATCH_BOUND: i32 = 8;

/// The tier table. Row order: exact matches are found by key scan, so order
/// is free; the GENERIC row must be last (selected as the fallback tail).
pub const TIERS: &[DeviceTier] = &[
    DeviceTier {
        key: DeviceKey {
            vendor: Vendor::Nvidia,
            cap_key: 1210,
        },
        name: "DGX Spark GB10",
        source: "minfer docs 94-104 (measured); agrees with llama.cpp mmvq.cu:349",
        provenance: Provenance::Measured,
        mmvq_batch_default: 8,
        mmvq_batch_by_type: &[],
        mmq_available: true,
    },
    DeviceTier {
        key: DeviceKey {
            vendor: Vendor::Nvidia,
            cap_key: 1200,
        },
        name: "Blackwell consumer (RTX 5090/5080/5070)",
        source: "llama.cpp mmvq.cu:335 (tuned on RTX 5090)",
        provenance: Provenance::Adopted,
        mmvq_batch_default: 8,
        mmvq_batch_by_type: &[(QClass::K4, 5), (QClass::K5, 6), (QClass::K6, 7)],
        mmq_available: true,
    },
    DeviceTier {
        key: DeviceKey {
            vendor: Vendor::Nvidia,
            cap_key: 870,
        },
        name: "Jetson Orin (Nano/NX/AGX)",
        source: "llama.cpp mmvq.cu:368 (tuned for Jetson Orin); field TODO plan §9.2",
        provenance: Provenance::Adopted,
        mmvq_batch_default: 8,
        mmvq_batch_by_type: &[(QClass::K4, 1), (QClass::K5, 1), (QClass::K6, 1)],
        mmq_available: true,
    },
    DeviceTier {
        key: DeviceKey {
            vendor: Vendor::Nvidia,
            cap_key: 890,
        },
        name: "Ada (RTX 4090/4080/4070)",
        source:
            "llama.cpp mmvq.cu:323 (tuned on RTX 4090); overrides touch only unsupported q2_k/q3_k",
        provenance: Provenance::Adopted,
        mmvq_batch_default: 8,
        mmvq_batch_by_type: &[],
        mmq_available: true,
    },
    DeviceTier {
        key: DeviceKey {
            vendor: Vendor::Nvidia,
            cap_key: 860,
        },
        name: "Ampere (RTX 3090/3080/3070/3060)",
        source: "llama.cpp generic — no Ampere batch specialization exists",
        provenance: Provenance::Adopted,
        mmvq_batch_default: 8,
        mmvq_batch_by_type: &[],
        mmq_available: true,
    },
    DeviceTier {
        key: DeviceKey {
            vendor: Vendor::Nvidia,
            cap_key: 750,
        },
        name: "Turing (RTX 2080/2060)",
        source: "batch: generic; mmq: plan §9 ruling #4 (provisional, field TODO §9.1)",
        provenance: Provenance::Adopted,
        mmvq_batch_default: 8,
        mmvq_batch_by_type: &[],
        // Conservative minfer gate: BT uses mma.m16n8k32 (sm_80+); Turing has
        // a different int8 MMA shape. llama.cpp serves Turing MMQ with its
        // own paths; minfer keeps f16 GEMM prefill until the 2080 Ti field
        // run re-adjudicates (plan §9.1).
        mmq_available: false,
    },
    DeviceTier {
        key: DeviceKey {
            vendor: Vendor::Unknown,
            cap_key: -1,
        },
        name: "GENERIC (unknown device)",
        source: "llama.cpp fallback convention; covers Pascal (sm_61) without a dedicated row",
        provenance: Provenance::Generic,
        mmvq_batch_default: 8,
        mmvq_batch_by_type: &[],
        // Placeholder — select() computes the effective gate as cc >= 800
        // for this row (unknown architectures keep the conservative gate).
        mmq_available: false,
    },
];

/// The GENERIC tail row (last entry of [`TIERS`]).
const GENERIC: usize = TIERS.len() - 1;

/// Convert minfer's runtime cc encoding (`major*100 + minor`, GB10 = 1201)
/// into the llama.cpp-style tier key (`major*100 + minor*10`, GB10 = 1210).
/// Minor is always < 10, so the conversion is lossless.
fn llama_key(minfer_cc: i32) -> i32 {
    (minfer_cc / 100) * 100 + (minfer_cc % 100) * 10
}

/// A resolved tier: the static row plus the effective MMQ gate (the GENERIC
/// row's gate is `cc >= 800`, every other row uses its own flag).
pub struct Selected {
    pub tier: &'static DeviceTier,
    pub mmq_available: bool,
}

/// Resolve a tier from minfer's runtime cc. Exact `cap_key` match first
/// (1210 must win over the >= 1200 family rule), then NVIDIA family
/// inheritance (>= 1200 Blackwell, >= 800 Ampere, >= 750 Turing), then the
/// GENERIC fallback.
pub fn select(minfer_cc: i32) -> Selected {
    select_by_key(llama_key(minfer_cc), minfer_cc)
}

/// Resolve a tier by an explicit llama.cpp-style key (the
/// `MINFER_DEVICE_TIER` override path — values match the table/source docs).
/// Forcing the GENERIC key simulates a fully unknown device (MMQ off).
pub fn select_forced(llama_style_key: i32) -> Selected {
    select_by_key(llama_style_key, i32::MIN)
}

fn select_by_key(key: i32, minfer_cc: i32) -> Selected {
    for tier in TIERS {
        if tier.key.vendor == Vendor::Nvidia && tier.key.cap_key == key {
            return Selected {
                tier,
                mmq_available: tier.mmq_available,
            };
        }
    }
    // Family inheritance (llama.cpp-style fallback chain).
    let family = if key >= 1200 {
        Some(&TIERS[1]) // Blackwell consumer
    } else if key >= 800 {
        Some(&TIERS[4]) // Ampere
    } else if key >= 750 {
        Some(&TIERS[5]) // Turing
    } else {
        None
    };
    if let Some(tier) = family {
        return Selected {
            tier,
            mmq_available: tier.mmq_available,
        };
    }
    let tier = &TIERS[GENERIC];
    Selected {
        tier,
        mmq_available: minfer_cc >= 800,
    }
}

/// Batch limit for a quant class on a tier (default or per-type override).
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
#[allow(dead_code)] // activated with the R8 destination decision (plan §14)
pub fn mmvq_cap(tier: &DeviceTier, class: QClass) -> i32 {
    mmvq_batch_limit(tier, class).min(IDENTITY_BATCH_BOUND)
}

#[cfg(test)]
mod tests {
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
}
