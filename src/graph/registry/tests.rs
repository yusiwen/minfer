//! `#[cfg(test)] mod tests` for `src/graph/registry.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
// The capability free functions the trait forwards to — imported here because
// the parent module no longer imports them (only tests name the op/dtype types
// now that `BackendCaps` carries just `reads_packed_kv`, [#244]).
use crate::graph::ops::{FusedOp, Op};
use crate::graph::DType;

/// The registered set, per configuration. Pinned so a backend that stops
/// registering (or gains an entry it should not have) fails here.
#[cfg(all(target_os = "macos", feature = "cuda"))]
const EXPECTED_REGISTERED: &[&str] = &["cpu", "metal", "cuda"];
#[cfg(all(target_os = "macos", not(feature = "cuda")))]
const EXPECTED_REGISTERED: &[&str] = &["cpu", "metal"];
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const EXPECTED_REGISTERED: &[&str] = &["cpu", "cuda"];
#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
const EXPECTED_REGISTERED: &[&str] = &["cpu"];

/// The assignment order, per configuration — the pre-F4 behaviour
/// (Metal, CUDA, CPU) as **literal** numbers.
///
/// Literals on purpose: deriving them from `PRIORITY_*` would make the value
/// half of this gate vacuous (only a reordering would fail, not a silent
/// re-numbering), and the numbers are what the assignment order *is*.
#[cfg(all(target_os = "macos", feature = "cuda"))]
const EXPECTED_PRIORITY: &[(&str, u16)] = &[("metal", 300), ("cuda", 200), ("cpu", 100)];
#[cfg(all(target_os = "macos", not(feature = "cuda")))]
const EXPECTED_PRIORITY: &[(&str, u16)] = &[("metal", 300), ("cpu", 100)];
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const EXPECTED_PRIORITY: &[(&str, u16)] = &[("cuda", 200), ("cpu", 100)];
#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
const EXPECTED_PRIORITY: &[(&str, u16)] = &[("cpu", 100)];

/// F4 gate 2: the **registered set**, the **priority order** and the priority
/// **numbers** are pinned, per configuration. A future accidental reordering
/// — or a backend that silently stops registering, or a re-numbering that
/// moves the order — fails here instead of quietly changing where every
/// graph's nodes land (the assignment is topology, standing rule 3).
#[test]
fn the_registered_set_and_priority_order_are_pinned() {
    let r = registry();
    let registered: Vec<&str> = r.iter().map(|e| e.name).collect();
    assert_eq!(
        registered, EXPECTED_REGISTERED,
        "the registered backend set moved"
    );
    let priority: Vec<(&str, u16)> = r
        .by_priority()
        .iter()
        .map(|&b| {
            (
                b.name(),
                r.get(b).expect("a priority handle is registered").priority,
            )
        })
        .collect();
    assert_eq!(
        priority, EXPECTED_PRIORITY,
        "the backend priority order moved (assignment would move with it)"
    );

    // The order is a total order: no two entries may tie, or the outcome
    // would depend on how they were registered.
    let mut prios: Vec<u16> = priority.iter().map(|(_, p)| *p).collect();
    let n = prios.len();
    prios.sort_unstable();
    prios.dedup();
    assert_eq!(prios.len(), n, "backend priorities must be distinct");

    // The cpu fallback is last: every device is offered before it.
    assert_eq!(priority.last().expect("cpu is always registered").0, "cpu");

    // …and with no pool enabled (a fresh allocator) the answer is the cpu,
    // deterministically, on every configuration.
    let alloc = GraphAllocator::new();
    assert_eq!(alloc.supports(&Op::Input, DType::F32), Some(Backend::CPU));
}

/// F4 gate 1: the known names resolve to the pinned handles, an unknown name
/// is a distinct loud error, and a compiled-out name is a *different* loud
/// error that names which of the two situations it is.
#[test]
fn names_resolve_and_unknown_names_are_refused() {
    // The id space is a file-format contract (the KV-session backend tag)
    // and it is the pre-F4 declaration order for `Ord`.
    assert_eq!(Backend::CPU.index(), 0);
    assert_eq!(Backend::METAL.index(), 1);
    assert_eq!(Backend::CUDA.index(), 2);
    assert_eq!(Backend::from_index(0), Some(Backend::CPU));
    assert_eq!(Backend::from_index(2), Some(Backend::CUDA));
    assert_eq!(Backend::from_index(3), None);
    assert_eq!(Backend::CPU.name(), "cpu");
    assert_eq!(Backend::METAL.name(), "metal");
    assert_eq!(Backend::CUDA.name(), "cuda");
    assert!(Backend::CPU < Backend::METAL && Backend::METAL < Backend::CUDA);

    // Diagnostics keep the pre-F4 spelling.
    assert_eq!(format!("{:?}", Backend::CPU), "CPU");
    assert_eq!(format!("{:?}", Backend::METAL), "Metal");
    assert_eq!(format!("{:?}", Backend::CUDA), "Cuda");

    // Spellings: trimmed, case-insensitive, canonical in the answer. A
    // device name resolves where the backend is compiled in and is the
    // NotCompiled refusal (never Unknown) where it is not.
    assert_eq!(resolve_name(" cpu "), Ok(Backend::CPU));
    assert_eq!(resolve_name("CPU"), Ok(Backend::CPU));
    match resolve_name("CuDa") {
        Ok(b) => assert_eq!(b, Backend::CUDA),
        Err(e) => assert!(
            matches!(e, NameError::NotCompiled { name: "cuda", .. }),
            "{e:?}"
        ),
    }
    assert_eq!(names(), &["cpu", "metal", "cuda"]);

    // "no such backend name" — always an error, on every configuration, and
    // the message lists the accepted names.
    let err = resolve_name("gpu2").unwrap_err();
    assert!(matches!(err, NameError::Unknown { .. }), "{err:?}");
    assert!(
        err.message()
            .contains("unknown backend 'gpu2'; known backends are: cpu, metal, cuda"),
        "{}",
        err.message()
    );

    // "that backend exists but is not compiled in" — a different message.
    #[cfg(not(target_os = "macos"))]
    {
        let e = resolve_name("metal").unwrap_err();
        assert!(
            matches!(e, NameError::NotCompiled { name: "metal", .. }),
            "{e:?}"
        );
        assert!(
            e.message()
                .contains("backend 'metal' is known but not compiled into this build"),
            "{}",
            e.message()
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let e = resolve_name("cuda").unwrap_err();
        assert!(
            matches!(e, NameError::NotCompiled { name: "cuda", .. }),
            "{e:?}"
        );
        assert!(e.message().contains("--features cuda"), "{}", e.message());
    }
    // …and the compiled-in names resolve on this configuration.
    #[cfg(target_os = "macos")]
    assert_eq!(resolve_name("metal"), Ok(Backend::METAL));
    #[cfg(feature = "cuda")]
    assert_eq!(resolve_name("cuda"), Ok(Backend::CUDA));
    assert_eq!(resolve_name("cpu"), Ok(Backend::CPU));

    // An unregistered handle claims nothing rather than guessing.
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        assert!(!Backend::METAL.is_registered());
        assert!(!Backend::METAL.caps().reads_packed_kv);
        assert!(!Backend::CUDA.caps().reads_packed_kv);
        assert!(Backend::CPU.is_registered());
        assert!(Backend::CPU.caps().reads_packed_kv);
        // "not compiled in" is not "available": the reason is still named.
        assert!(unavailable_reason(Backend::METAL).is_some());
        // …and a name that exists but is not in this binary is *not* the
        // same error as a name that does not exist.
        assert_ne!(
            resolve_name("metal").unwrap_err(),
            NameError::Unknown {
                name: "nope".to_string()
            }
        );
    }
}

/// The fence surface: a name list resolves, unknown spellings are refused
/// from either surface, `cpu` survives every fence, and unset is the pre-F4
/// behaviour.
#[test]
fn the_name_surface_fences_devices_and_keeps_cpu() {
    let cpu_only = BackendFilter::from_names(["cpu"]).expect("cpu is a valid name");
    assert!(cpu_only.allows(Backend::CPU));
    assert!(!cpu_only.allows(Backend::METAL));
    assert!(!cpu_only.allows(Backend::CUDA));
    assert!(!cpu_only.is_unfiltered());

    // cpu is the universal fallback: naming only a device keeps it. Which
    // device names this build has is configuration-dependent, so this picks
    // one it does have (the "neither" configuration's case is the
    // NotCompiled assertion at the end).
    #[cfg(target_os = "macos")]
    let device = Backend::METAL;
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    let device = Backend::CUDA;
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    {
        let device_only =
            BackendFilter::from_names([device.name()]).expect("a compiled-in device resolves");
        assert!(
            device_only.allows(Backend::CPU),
            "the fallback must survive a device-only fence"
        );
        assert!(device_only.allows(device));
        assert!(!device_only.is_unfiltered());
    }

    // Comma-separated and repeated spellings are the same request, and
    // whitespace/case are not part of a name.
    assert_eq!(
        BackendFilter::from_names([" CPU , "]).unwrap(),
        BackendFilter::from_names(["cpu", "cpu"]).unwrap()
    );

    // Unset = every backend (the pre-F4 behaviour).
    assert!(BackendFilter::all().is_unfiltered());
    assert!(BackendFilter::default().allows(Backend::CUDA));
    assert!(request_filter(&[], None).unwrap().is_unfiltered());

    // The environment surface, and the flag winning over it.
    let env_only = request_filter(&[], Some("cpu")).unwrap();
    assert!(!env_only.allows(Backend::CUDA));
    let flag_wins = request_filter(&["cpu".to_string()], Some("does-not-matter")).unwrap();
    assert!(!flag_wins.allows(Backend::CUDA));
    assert!(flag_wins.allows(Backend::CPU));
    // An empty environment value is "unset", not "empty fence".
    assert!(request_filter(&[], Some("")).unwrap().is_unfiltered());

    // A name that is not compiled in is refused as such, so a fence can
    // never quietly become "fewer backends than asked for".
    #[cfg(not(feature = "cuda"))]
    {
        let e = BackendFilter::from_names(["cuda"]).unwrap_err();
        assert!(e.contains("not compiled into this build"), "{e}");
        assert!(!e.contains("unknown backend"), "{e}");
    }

    // An unknown spelling is refused from either surface, with one message.
    let flag_err = BackendFilter::from_names(["nope"]).unwrap_err();
    assert!(flag_err.contains("unknown backend 'nope'"), "{flag_err}");
    assert!(
        flag_err.contains("known backends are: cpu, metal, cuda"),
        "{flag_err}"
    );
    let env_err = request_filter(&[], Some("cpu,nope")).unwrap_err();
    assert!(env_err.contains("unknown backend 'nope'"), "{env_err}");
}

/// The #87 seam: `reads_packed_kv` is one authority, and the two call sites
/// that used to hardcode "CPU only" read it.
///
/// C4 S2b flipped CUDA's answer: the CUDA attention kernels are layout-tagged
/// and read a packed region, so this is now a two-backend yes (CPU + CUDA) with
/// Metal still at G5 ([#44]). A build without the `cuda` feature registers no
/// CUDA entry, so it claims nothing — which is the point of reading a registry
/// field instead of a hardcoded list.
///
/// [#44]: https://github.com/yusiwen/minfer/issues/44
#[test]
fn the_packed_kv_capability_is_the_registrys_answer() {
    assert!(Backend::CPU.caps().reads_packed_kv);
    assert!(!Backend::METAL.caps().reads_packed_kv);
    assert!(reads_packed_kv(Backend::CPU));
    #[cfg(feature = "cuda")]
    {
        assert!(Backend::CUDA.caps().reads_packed_kv);
        assert!(reads_packed_kv(Backend::CUDA));
    }
    #[cfg(not(feature = "cuda"))]
    {
        assert!(!Backend::CUDA.caps().reads_packed_kv);
        assert!(!reads_packed_kv(Backend::CUDA));
    }
    // …and the C4 format gate asks the same field.
    use crate::models::Device;
    for device in [Device::Cpu, Device::Metal, Device::Cuda] {
        assert_eq!(
            KvFormat::Q8_0.supports(device),
            reads_packed_kv(device.backend()),
            "{device:?}"
        );
    }
    // F32/f16 are every backend's business (unchanged).
    for device in [Device::Cpu, Device::Metal, Device::Cuda] {
        assert!(KvFormat::F32.supports(device));
        assert!(KvFormat::F16.supports(device));
    }
}

/// The capability answer is **one authority**: each `Backend` trait method
/// forwards to its backend module's own free function or constant, and the
/// assignment pass reads the trait method. Nothing keeps a second copy, so a
/// graph can never be assigned to a backend whose trait would refuse the op.
///
/// Repointed in [#244]: the test used to compare `Backend::CPU.caps()` (a
/// registry field) against the trait. The registry no longer carries the three
/// `supports_*` fields, so the comparison is now module function vs. trait —
/// which is the authority the field merely mirrored.
///
/// [#244]: https://github.com/yusiwen/minfer/issues/244
#[test]
fn registry_caps_match_the_backend_trait() {
    use super::super::backend_takes;
    use super::super::cpu_backend::{self, CpuBackend};
    use crate::graph::backend::Backend as BackendTrait;
    let cpu = CpuBackend::new();
    for op in [
        Op::Input,
        Op::Add,
        Op::Silu,
        Op::SwiGLU,
        Op::Softmax { dim: 0 },
    ] {
        for dtype in [DType::F32, DType::F16] {
            assert_eq!(
                cpu_backend::supports_op(&op, dtype),
                BackendTrait::supports_op(&cpu, &op, dtype),
                "{op:?} {dtype:?}"
            );
        }
    }
    assert_eq!(
        cpu_backend::SUPPORTS_ATTN_SPAN,
        BackendTrait::supports_attn_span(&cpu)
    );
    assert_eq!(
        cpu_backend::supports_fused(&FusedOp::SwiGLU),
        BackendTrait::supports_fused(&cpu, &FusedOp::SwiGLU)
    );
    // …and the assignment answers what the trait says.
    let alloc = GraphAllocator::new();
    assert_eq!(
        alloc.supports(&Op::Silu, DType::F32).is_some(),
        backend_takes(&cpu, &Op::Silu, DType::F32)
    );
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `registry.rs` (bucket B of the dead-code census — the
// only test caller lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

/// Every accepted name, in id order.
///
/// Test-only (#239): driven by
/// `graph::registry::tests::names_resolve_and_unknown_names_are_refused`.
pub fn names() -> &'static [&'static str; N_BACKENDS] {
    &NAMES
}
