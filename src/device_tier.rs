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
///
/// Each never-constructed vendor names itself ([#244] — replacing the
/// container-level allowance #243 could not tighten): the variants are the
/// namespace the design reserves, and a non-NVIDIA row in [`TIERS`] is what
/// would construct one. `Nvidia`/`Unknown` are constructed by the table and the
/// selector, so the enum itself stays checked.
///
/// [#244]: https://github.com/yusiwen/minfer/issues/244
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    /// Reserved for an AMD row keyed by the `0x1000000` gfx offset.
    #[allow(dead_code)]
    Amd,
    /// Reserved for a Moore Threads row keyed by the `0x0100000` offset.
    #[allow(dead_code)]
    Mthreads,
    /// Reserved for an Apple-silicon row keyed by chip generation.
    #[allow(dead_code)]
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
///
/// `Other` (the no-override class) is constructed only by
/// `device_tier::tests` today, so it carries the `not(test)` allowance at the
/// member ([#244]) instead of the enum-level blanket #243 could not tighten; a
/// caller that classifies a non-K-quant weight type would construct it.
///
/// [#244]: https://github.com/yusiwen/minfer/issues/244
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QClass {
    K4,
    K5,
    K6,
    /// See the enum note: constructed only by `device_tier::tests` today; a caller
    /// that classifies a non-K-quant weight type would construct it.
    #[cfg_attr(not(test), allow(dead_code))]
    Other,
}

/// One device tier: the dispatch parameters a cc resolves to.
///
/// The three fields production does not read carry member-level `not(test)`
/// allowances ([#244], replacing the container blanket #243 could not tighten):
/// they are read by `device_tier::tests` and are the provenance/calibration
/// record every row cites. [`select`] reads `key`/`name`/`provenance`/
/// `mmq_available`; `mmvq_batch_default`/`mmvq_batch_by_type` are what the
/// batch-cap activation (plan §14 R8) would read.
///
/// [#244]: https://github.com/yusiwen/minfer/issues/244
pub struct DeviceTier {
    pub key: DeviceKey,
    pub name: &'static str,
    /// Provenance: minfer doc or llama.cpp file:line — every row cites its
    /// source so future syncs and promotions have an anchor. Read by
    /// `device_tier::tests` (a row with no source is a data bug).
    #[cfg_attr(not(test), allow(dead_code))]
    pub source: &'static str,
    pub provenance: Provenance,
    /// Batch limit for the quantized decode (MMVQ-family) kernels. Read by
    /// `device_tier::tests`; the batch-cap activation (plan §14 R8) is what would
    /// read it in production ([#244]).
    #[cfg_attr(not(test), allow(dead_code))]
    pub mmvq_batch_default: i32,
    /// Per-quant-class overrides (sparse; K-quant only in the source data). Read by
    /// `device_tier::tests`; the batch-cap activation (plan §14 R8) is what would read
    /// the per-class override in production ([#244]).
    #[cfg_attr(not(test), allow(dead_code))]
    pub mmvq_batch_by_type: &'static [(QClass, i32)],
    /// int8 BT (block-tile tensor-core) prefill availability. The GENERIC
    /// row encodes `false` here; [`select`] replaces it with `cc >= 800`
    /// (the conservative minfer gate — plan §9 ruling #4).
    pub mmq_available: bool,
}

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
        source: "llama.cpp mmvq.cu:356 (tuned for Jetson Orin); field TODO plan §9.2",
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
            "llama.cpp mmvq.cu:324 (tuned on RTX 4090); overrides touch only unsupported q2_k/q3_k",
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

/// Family-row lookup by key, NOT by position — the [`TIERS`] doc comment
/// promises row order is free, so the inheritance chain must never index the
/// array (reordering rows would otherwise silently misroute the fallback).
fn family_row(cap_key: i32) -> Option<&'static DeviceTier> {
    TIERS
        .iter()
        .find(|t| t.key.vendor == Vendor::Nvidia && t.key.cap_key == cap_key)
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
    // Family inheritance — a minfer extension: llama.cpp matches per-CC by
    // exact equality, unknown revisions inherit the nearest measured family.
    let family = if key >= 1200 {
        family_row(1200) // Blackwell consumer
    } else if key >= 800 {
        family_row(860) // Ampere
    } else if key >= 750 {
        family_row(750) // Turing
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

#[cfg(test)]
mod tests;
