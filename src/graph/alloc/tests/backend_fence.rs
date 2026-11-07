//! The backend fence, the offload plan and the per-engine KV format the allocator stamps.
//!
//! Split out of `src/graph/alloc/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// F4: a fresh allocator inherits the run's backend fence, so the two places
/// the fence is enforced (this allocator's assignment and the graph
/// builders' `Device`) read the same answer. The default — nothing installed
/// — is every backend, i.e. the pre-F4 behaviour.
#[test]
fn a_fresh_allocator_inherits_the_runs_backend_filter() {
    let alloc = GraphAllocator::new();
    assert_eq!(
        alloc.backend_filter(),
        &crate::graph::registry::active_filter()
    );
    // The process-wide default is unfiltered in the test binary (nothing
    // installs a fence here), so the assignment is unchanged for every
    // existing gate.
    assert!(crate::graph::registry::active_filter().is_unfiltered());
    assert_eq!(alloc.supports(&Op::Input, DType::F32), Some(Backend::CPU));
}
/// F4: the fence reaches the **assignment pass**, not just the name parser.
///
/// Needs a usable device, so it is `#[ignore]`d like the other device gates
/// (run it serially: the CUDA device state is a process-wide singleton).
/// Without a fence the device takes an op it supports; with the device
/// fenced off the *same* op must land on the CPU — and the device stays
/// enabled, because the fence is a policy, not a teardown.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_the_backend_fence_moves_assignment_off_a_usable_device() {
    crate::cuda::CudaState::init();
    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("no CUDA device; this gate needs one");
        return;
    }
    assert_eq!(
        alloc.supports(&Op::Silu, DType::F32),
        Some(Backend::CUDA),
        "an unfenced allocator must offer the device"
    );
    alloc.set_backend_filter(crate::graph::registry::BackendFilter::from_names(["cpu"]).unwrap());
    assert_eq!(
        alloc.supports(&Op::Silu, DType::F32),
        Some(Backend::CPU),
        "a fenced allocator must not offer the device"
    );
    assert!(
        alloc.cuda().is_some(),
        "the fence must not tear the device pool down"
    );
}
/// E5: with a plan in force, the device is only offered for a node whose block the plan
/// offloaded — and a node outside any block only under a full plan. Needs a device to be
/// meaningful (without one the answer is CPU either way), so it is gated on the CUDA
/// build and skips when no device is present.
#[test]
#[cfg(feature = "cuda")]
fn the_offload_plan_keeps_late_blocks_off_the_device() {
    use crate::graph::offload::OffloadPlan;
    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("no CUDA device; skipping the E5 placement gate");
        return;
    }
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_cuda());
    // No plan: the pre-E5 behaviour — the device takes every node it can.
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(3)),
        Some(Backend::CUDA)
    );
    alloc.set_offload_plan(Some(OffloadPlan {
        gpu_layers: 2,
        n_layers: 4,
    }));
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(0)),
        Some(Backend::CUDA),
        "an offloaded block may use the device"
    );
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(2)),
        Some(Backend::CPU),
        "a block past the plan stays on the CPU"
    );
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, None),
        Some(Backend::CPU),
        "a partial plan keeps the unblocked tensors (embed/output) on the CPU"
    );
    // A full plan puts the unblocked tensors back on the device.
    alloc.set_offload_plan(Some(OffloadPlan {
        gpu_layers: 4,
        n_layers: 4,
    }));
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, None),
        Some(Backend::CUDA)
    );
}
/// #153: the allocator stamps the engine's resolved format onto its **CUDA**
/// backend — both when the backend already exists (`set_kv_format`) and when it is
/// created afterwards (`enable_cuda`), so the tag is always the engine's and never
/// a process global. Device-gated.
#[test]
#[cfg(feature = "cuda")]
fn set_kv_format_stamps_the_cuda_layout_per_engine() {
    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("no CUDA device; skipping the #153 layout-stamp gate");
        return;
    }
    // (a) the format is known before the backend exists: `enable_cuda` must build
    // it with the stamped layout, not a default.
    let mut alloc = GraphAllocator::new();
    assert_eq!(
        alloc.kv_format(),
        KvFormat::F32,
        "a fresh allocator defaults to f32"
    );
    alloc.set_kv_format(KvFormat::Q8_0);
    assert!(alloc.enable_cuda());
    assert_eq!(
        alloc.cuda().unwrap().kv_layout(),
        crate::cuda::KV_LAYOUT_Q8_0,
        "the backend must be built with the engine's stamped format"
    );
    assert_eq!(alloc.cuda().unwrap().kv_format(), KvFormat::Q8_0);

    // (b) the format changes while the backend exists: `set_kv_format` re-stamps it
    // (an engine's format never moves in production, but the stamp must be total).
    alloc.set_kv_format(KvFormat::F16);
    assert_eq!(
        alloc.cuda().unwrap().kv_layout(),
        crate::cuda::KV_LAYOUT_F16,
        "an existing backend must follow the stamp"
    );
    assert_eq!(alloc.kv_format(), KvFormat::F16);
    assert_eq!(alloc.cuda().unwrap().kv_format(), KvFormat::F16);
}

/// #763 S5: the fence belongs to the **engine**, not the process.
///
/// An allocator built inside a scope inherits that scope's fence, and the
/// process default is back as soon as the scope ends — which is what lets two
/// engines in one process disagree about which backends they may use.
#[test]
fn a_scoped_engine_carries_its_own_fence_and_restores_the_default() {
    use crate::graph::registry::{active_filter, with_filter, BackendFilter};

    let cpu = BackendFilter::from_names(["cpu"]).expect("cpu is a backend name");
    assert!(active_filter().is_unfiltered(), "the test process default");

    let (seen, support) = with_filter(cpu.clone(), || {
        let alloc = GraphAllocator::new();
        (
            alloc.backend_filter().clone(),
            alloc.supports(&Op::Input, DType::F32),
        )
    });
    assert_eq!(seen, cpu, "the allocator inherits the engine's fence");
    assert_eq!(support, Some(Backend::CPU));
    assert!(
        active_filter().is_unfiltered(),
        "leaving the scope must restore the process configuration"
    );
}

/// Nested scopes compose: an inner engine sees its own fence, and the outer
/// engine's fence is back when the inner one ends.
#[test]
fn two_engine_scopes_see_their_own_fence() {
    use crate::graph::registry::{active_filter, with_filter, BackendFilter};

    let cpu = BackendFilter::from_names(["cpu"]).expect("cpu is a backend name");
    let all = BackendFilter::all();
    with_filter(cpu.clone(), || {
        assert_eq!(active_filter(), cpu);
        with_filter(all.clone(), || {
            assert_eq!(active_filter(), all, "the inner engine's fence");
        });
        assert_eq!(active_filter(), cpu, "the outer engine's fence is restored");
    });
    assert!(active_filter().is_unfiltered());
}
