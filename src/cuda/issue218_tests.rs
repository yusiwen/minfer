//! `#[cfg(test)] mod issue218_tests` for `src/cuda.rs` — the prefill-GEMM
//! dynamic-smem opt-in invariant (#218).
//!
//! `gemm_prefill_smem_init` (the **eager** startup sweep) lost its production
//! caller in #188 and was later annotated `allow(dead_code)` instead of being
//! asked about. Plan B keeps the lazy per-launch production path
//! (`gemm_smem_optin`, reached through `launch_gemm_f16`) and makes the
//! invariant explicit:
//!
//! - [`cuda_prefill_smem_optin_is_done_by_production`] — a **real** prefill
//!   forward opts a >48 KiB `gemm_f16_nt_kernel_t` instantiation in, asserted
//!   through the device's own `cudaFuncGetAttributes().maxDynamicSharedSizeBytes`
//!   read-back. Non-vacuous by construction: it runs in a **fresh process** (the
//!   tile config and the attribute are both process-scoped, so this is the only
//!   way to observe `opted_in == 0` before the forward) and asserts that
//!   precondition first;
//! - [`cuda_prefill_smem_optin_refusal_fails_the_prefill`] — the control arm:
//!   `MINFER_TEST_CALL_FAIL=attr:gemm_f16_f16` must make the production
//!   prefill refuse the launch and name the site. Separated from the value arm
//!   by process: the opt-in answer is cached per instantiation, so a second
//!   process is where "production consults the opt-in on this launch" is
//!   unambiguous.
//!
//! The captured-graph half of the invariant
//! (`cuda_prefill_smem_optin_is_never_set_inside_a_capture_window`) lives next
//! to the capture machinery in `graph::cuda_backend::tests`.
use super::*;

/// The >48 KiB instantiation the fresh-process harness selects: with
/// `MINFER_GEMM_K64=1` and the default `MINFER_GEMM_TM=128`, `launch_gemm_f16`
/// instantiates `gemm_f16_nt_kernel_t<128,64,false>` at 57344 B — over the
/// 48 KiB `cudaFuncSetAttribute` default cap, so production must opt it in.
pub(crate) const TM: i32 = 128;
pub(crate) const KS: i32 = 64;
pub(crate) const AF32: i32 = 0;
pub(crate) const KERNEL: &str = "gemm_f16_nt_kernel_t<128,64,false>";

/// A prefill-shaped quantized matmul: `[ID x NT] * W[ID x OD]` with `nt > 1`, so
/// `matmul_f32_ptr_layout` takes the `prefill_gemm_f16` arm (`MINFER_MMQ=0`)
/// and the launcher is `gemm_f16_nt_kernel_t<128,64,false>`.
pub(crate) const NT: usize = 16;
pub(crate) const ID: usize = 32;
pub(crate) const OD: usize = 64;
pub(crate) const WEIGHT: &str = "issue218_bigsmem_w";

/// `od * (id / 32) * 18` bytes of a valid q4_0 weight (small `d`, biased
/// nibbles — the layout `cuda_prefill_gemm_bitparity` documents).
pub(crate) fn weight_bytes() -> Vec<u8> {
    let mut out = vec![0u8; OD * (ID / 32) * 18];
    for (g, blk) in out.chunks_mut(18).enumerate() {
        let d = half::f16::from_f32(0.01 + g as f32 * 1e-5).to_le_bytes();
        blk[0] = d[0];
        blk[1] = d[1];
        for (j, b) in blk[2..].iter_mut().enumerate() {
            *b = ((g * 7 + j * 13) % 16) as u8 | (((g * 5 + j * 3) % 16) as u8) << 4;
        }
    }
    out
}

/// The shared prefill fixture: one quantized matmul, `nt = NT > 1`.
pub(crate) fn big_smem_prefill_graph() -> crate::graph::ComputeGraph {
    use crate::graph::builder::GraphBuilder;
    use crate::graph::DType;
    use crate::tensor::{Tensor, TensorType};
    let mut b = GraphBuilder::new();
    let x = b.input("x", [ID, NT, 1, 1], DType::F32);
    let mut w = Tensor::from_data(
        TensorType::Q4_0,
        &[ID as i64, OD as i64, 1, 1],
        weight_bytes(),
    );
    w.name = WEIGHT.to_string();
    let m = b.matmul(x, &w, None);
    b.output(m);
    b.build()
}

/// The gate's precondition, asserted the same way in both arms.
pub(crate) fn assert_optin_preconditions(what: &str) {
    let need = unsafe { gemm_smem_need(TM, KS, AF32) };
    assert!(
        need > 48 * 1024,
        "[{what}] the gate needs a >48 KiB instantiation, got {need} B"
    );
    assert_eq!(
        unsafe { gemm_smem_opted_in(TM, KS, AF32) },
        0,
        "[{what}] precondition: {KERNEL} must not already be opted in — this gate \
         runs in a fresh process, and this is the first thing that touches it"
    );
    assert_eq!(
        unsafe { gemm_smem_optin_in_capture_count() },
        0,
        "[{what}] precondition: no smem opt-in may have run inside a capture window"
    );
}

// ─── Gate 1: production opts the >48 KiB instantiation in ────────────────

/// A real prefill forward must opt `gemm_f16_nt_kernel_t<128,64,false>` in,
/// asserted through the device's own read-back. Fresh process (see the harness
/// for why `opted_in == 0` cannot be observed in-process).
///
/// Mutation evidence (rule 3): make `gemm_smem_optin` return `true` without
/// calling `minfer_smem_optin` (i.e. never call `cudaFuncSetAttribute`); the
/// post-forward `opted_in == 1` assertion goes red.
#[test]
fn cuda_prefill_smem_optin_is_done_by_production() {
    const FILTER: &str = "cuda_prefill_smem_optin_is_done_by_production";
    match super::test_child::child_phase().as_deref() {
        Some("optin") => optin_child(),
        _ => {
            let child = super::test_child::run_self(FILTER, "optin", &[]);
            child.verdict("production prefill-GEMM smem opt-in");
        }
    }
}

fn optin_child() {
    CudaState::init();
    if CudaState::get().is_none() {
        super::test_child::child_skip("no CUDA device");
    }
    assert_optin_preconditions("optin");
    // A real prefill: 24 arbitrary in-vocabulary tokens (> 1, so the graph is
    // prefill-shaped and every quantized matmul with `id % 32 == 0` runs
    // `prefill_gemm_f16`). Qwen2.5-0.5B's every matmul id is a multiple of 64,
    // so the harness's ks=64 instantiation is the one selected.
    let mut p = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"));
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    if !p.exists() {
        super::test_child::child_skip("Qwen2.5-0.5B q4_0 GGUF is not cached");
    }
    let _guard = CudaState::model_load_guard();
    let gguf = crate::gguf::load_gguf_model(&p).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let q2: &crate::models::qwen2::Qwen2Model = model.as_any().downcast_ref().expect("Qwen2Model");
    let ids: Vec<u32> = (0..24u32).map(|i| 100 + i).collect();
    let positions: Vec<usize> = (0..ids.len()).collect();
    let mut cache = crate::graph::cache::GraphCache::new();
    let logits = crate::models::qwen2::graph::Qwen2Graph::forward_cached(
        q2, &ids, &positions, 1, 4096, &mut cache,
    );
    assert!(!logits.is_empty(), "the real prefill produced no logits");
    assert_eq!(
        unsafe { gemm_smem_opted_in(TM, KS, AF32) },
        1,
        "the real prefill must have made the device report {KERNEL} opted in \
         (maxDynamicSharedSizeBytes >= 57344); it still reports the 48 KiB default, \
         so production never called cudaFuncSetAttribute for it"
    );
    super::test_child::child_ok();
}

// ─── The control arm for gate 1 ────────────────────────────────────────

/// The `MINFER_TEST_CALL_FAIL=attr:gemm_f16_f16` injection drives the site's
/// real `cudaFuncSetAttribute` into failure, and the production prefill must
/// **refuse the launch and name it** rather than run with an un-opted-in
/// >48 KiB dynamic smem. Env-gated behind `MINFER_TEST_ISSUE218=1` (like the
/// #147/#162 deliberate-failure gates) because it makes a real CUDA call fail —
/// a `compute-sanitizer` run must not see it.
///
/// Mutation evidence (rule 3): drop the `if (launched == 0) return Err(..)` arm
/// in `prefill_gemm_f16_inner` (or make `launch_gemm_f16` ignore the opt-in
/// answer); the `expect_err` goes red. The arm is separated from the value arm
/// by **process**: `gemm_smem_optin` caches its answer per instantiation, so
/// this arm's injection is the only thing that can move the first launch.
#[test]
fn cuda_prefill_smem_optin_refusal_fails_the_prefill() {
    if std::env::var("MINFER_TEST_ISSUE218").is_err() {
        eprintln!(
            "skipping: set MINFER_TEST_ISSUE218=1 to run the deliberate-failure arm (it makes \
             a real cudaFuncSetAttribute call fail; a compute-sanitizer run must not set it)"
        );
        return;
    }
    const FILTER: &str = "cuda_prefill_smem_optin_refusal_fails_the_prefill";
    match super::test_child::child_phase().as_deref() {
        Some("refuse") => refuse_child(),
        _ => {
            let child = super::test_child::run_self(
                FILTER,
                "refuse",
                &[("MINFER_TEST_CALL_FAIL", "attr:gemm_f16_f16")],
            );
            if child.verdict("the injected attribute failure").is_none() {
                return;
            }
            // The site's own report, not just the Rust error: the arm proves the
            // *launch site* named the failed call.
            assert!(
                child.stderr.contains("attr:gemm_f16_f16"),
                "the site report must name attr:gemm_f16_f16:\n{}",
                child.stderr
            );
            assert!(
                child.stderr.contains("cudaFuncSetAttribute"),
                "the site report must name the call:\n{}",
                child.stderr
            );
            assert!(
                child.stderr.contains("cudaErrorInvalidValue"),
                "the site report must name the error:\n{}",
                child.stderr
            );
        }
    }
}

fn refuse_child() {
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::scheduler::BackendScheduler;
    CudaState::init();
    let Some(state) = CudaState::get() else {
        super::test_child::child_skip("no CUDA device");
    };
    let _guard = CudaState::model_load_guard();
    state.register_weight(WEIGHT, &weight_bytes());
    let sched = BackendScheduler::new();
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_cuda(), "a CUDA device must answer the probe");
    let mut graph = big_smem_prefill_graph();
    sched.assign_backends(&mut graph, &alloc);
    alloc.alloc_graph(&graph).unwrap();
    let xs: Vec<f32> = (0..ID * NT)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
        .collect();
    alloc.fill_input(&graph, "x", &xs).unwrap();
    let err = sched
        .execute(&graph, &mut alloc)
        .expect_err("the injected attribute failure must refuse the >48 KiB prefill launch");
    assert!(
        err.contains("refused the launch"),
        "the error must say the launch was refused: {err}"
    );
    super::test_child::child_ok();
}
