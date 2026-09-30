//! `#[cfg(test)] mod tests` for `src/graph/cuda_backend.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::alloc::GraphAllocator;
use crate::graph::backend::{Backend as _, KvProvider};
use crate::graph::builder::GraphBuilder;
use crate::graph::cache::GraphCache;
use crate::graph::scheduler::BackendScheduler;
use crate::graph::DType;

/// Init the CUDA singleton; silent-skip the test when no device answers
/// (e.g. CI without a GPU). Run with --nocapture to see skips.
fn device() -> Option<&'static crate::cuda::CudaState> {
    crate::cuda::CudaState::init();
    crate::cuda::CudaState::get()
}

/// Median of `v` (does not reorder the caller's slice).
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// Matched pairs per phase of the S4 map-window A/B, fixed in advance
/// (issue #189).
const PAIRS: usize = 9;

/// The S4 A/B's sign-test threshold: the gate refuses when at least this many
/// of the `PAIRS` matched pairs put the map arm above the bar. For 9 pairs the
/// one-sided binomial tail under the null (a fair coin, i.e. map == bar *
/// span) is `P(X >= 7) = 46/512 = 0.090`. A median of the same 9 pairs flips
/// once 5 are disturbed — a minority cannot decide the sign test.
const SIGN_TEST_REFUSALS: usize = 7;

/// The map-window A/B's verdict statistic (issue #189), shared by both its
/// phases.
///
/// `span` and `map` hold µs/launch for the **same** round index, which is why
/// the callers interleave the rounds: every ratio is a matched pair measured
/// next to each other on one machine state. A pair is a *refusal* when the map
/// arm spent more than `bar` times its matched span arm. The verdict is the
/// refusal **count**, and the gate refuses only at [`SIGN_TEST_REFUSALS`] —
/// the one-sided sign-test threshold for [`PAIRS`] pairs at `alpha = 0.090`.
/// A median flips once half the pairs are disturbed; the sign test needs a
/// two-thirds supermajority, so a load spike that moves a minority (or even a
/// bare majority) of pairs cannot decide it. Returns
/// `(refusals, span_median, map_median, sorted_ratios)` so the gate prints
/// every sample and not just the verdict.
fn sign_test_ratio(span: &[f64], map: &[f64], bar: f64) -> (usize, f64, f64, Vec<f64>) {
    assert_eq!(span.len(), map.len(), "one ratio per matched pair");
    assert_eq!(span.len(), PAIRS, "the pair count is fixed in advance");
    let mut ratios: Vec<f64> = span.iter().zip(map).map(|(s, m)| m / s).collect();
    let refusals = ratios.iter().filter(|r| **r > bar).count();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (refusals, median(span), median(map), ratios)
}

/// The statistic itself, on the recorded failing distribution (issue #189).
///
/// The pre-#189 gate asserted the **median** of these nine ratios and went
/// red at 1.398x on a loaded parallel device run (GB10 sm_121, 2026-09-26).
/// The paired sign test must (a) pass that same distribution — 5 refusals is
/// below its threshold of 7 — while (b) refusing a real regression, where the
/// map arm's work doubles and every matched pair moves. Pure, so it runs in
/// the always-run CUDA unit suite with no device (rule 1 of the gate contract:
/// the gate must assert the value, not just a relation between two paths).
#[test]
fn the_s4_ab_statistic_absorbs_a_loaded_run_and_still_refuses_a_real_regression() {
    // The recorded prefill ratios, verbatim.
    let recorded = [
        0.632f64, 0.697, 0.989, 1.091, 1.398, 1.440, 1.466, 2.198, 6.695,
    ];
    let span = vec![100.0f64; PAIRS];
    let map: Vec<f64> = recorded.iter().map(|r| r * 100.0).collect();
    let (refusals, _, _, ratios) = sign_test_ratio(&span, &map, 1.25);
    assert_eq!(
        refusals, 5,
        "the recorded run has 5 of 9 pairs above 1.25x (ratios {ratios:?})"
    );
    assert!(
        median(&ratios) > 1.25,
        "the old median statistic must be red on the recorded distribution — otherwise \
         the recorded failure could not have happened (median {})",
        median(&ratios)
    );
    assert!(
        refusals < SIGN_TEST_REFUSALS,
        "the sign test must absorb the recorded loaded run ({refusals} < \
         {SIGN_TEST_REFUSALS})"
    );

    // A doubled map cost moves every pair, and the sign test refuses all nine.
    let doubled: Vec<f64> = span.iter().map(|s| s * 2.0).collect();
    let (d_refusals, ..) = sign_test_ratio(&span, &doubled, 1.25);
    assert_eq!(
        d_refusals, PAIRS,
        "a doubled map cost must be refused on every matched pair"
    );
}

#[test]
fn cuda_pool_roundtrip() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut cb = CudaBackend::new().expect("backend after device init");
    let id = cb.alloc_buffer(16);
    cb.write_host(id, &[1.5f32; 16]).unwrap();
    assert_eq!(cb.copy_to_host(id).unwrap(), vec![1.5f32; 16]);
    // shorter than the buffer is fine, longer is rejected
    cb.write_host(id, &[2.0f32; 4]).unwrap();
    assert!(cb.write_host(id, &[2.0f32; 32]).is_err());
    // free-list reuse hands back the same id; pool_gen tracked both times
    cb.free_buffer(id);
    let id2 = cb.alloc_buffer(16);
    assert_eq!(id, id2);
    assert_eq!(cb.pool_gen, 2);
}

/// Issue #188 acceptance, the **probe** — the instrument the mode decision
/// and the fix are both judged by.
///
/// It constructs the recorded race *deterministically*: thread B opens a
/// capture window on its stream, records a device→device copy inside it and
/// signals; thread A then performs a weight registration **while the window
/// is open**; only then does B close the window (end capture + instantiate +
/// launch) and read the copied bytes back. Without the handshake the raw
/// failure rate was 3/6 bare and 2/10 under gdb — too low to distinguish a
/// fix from luck.
///
/// Two env knobs make it the experiment rather than a single post-fix
/// assertion (see `docs/CUDA-BACKEND-DESIGN.md` §"Per-instance streams and
/// capture"):
/// - `MINFER_PROBE_STREAM=context` captures on `CudaState`'s own **blocking**
///   stream — the pre-#188 shared stream — instead of a fresh non-blocking
///   instance stream (the default);
/// - `MINFER_PROBE_LEGACY_MEMCPY=1` issues the registration through the
///   pre-#188 **blocking** `cudaMemcpy` instead of the stream-ordered path.
/// - `MINFER_CUDA_CAPTURE_MODE=0|1|2` selects the capture mode.
///
/// The 2×3 (stream × mode) matrix is run externally, one process per cell:
/// a fault kills the process, so it cannot be looped in-process. On the
/// fixed code the probe passes in every cell; on the pre-#188 code the
/// shared-stream + blocking-copy cells crash or report 901 under global
/// mode.
///
/// The verdict has two independent arms: the capture window must close with
/// `cudaStreamEndCapture` code `0` (never 901
/// `cudaErrorStreamCaptureInvalidated`) and the graph must produce the bytes
/// the window recorded.
#[test]
fn capture_window_on_one_thread_survives_a_weight_registration_on_another() {
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let use_context_stream = std::env::var("MINFER_PROBE_STREAM").as_deref() == Ok("context");
    let legacy_memcpy = std::env::var("MINFER_PROBE_LEGACY_MEMCPY").as_deref() == Ok("1");
    let iterations: usize = std::env::var("MINFER_PROBE_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let stream = if use_context_stream {
        state.stream()
    } else {
        state.create_stream()
    };
    assert!(!stream.is_null(), "the probe needs a capture stream");

    const N: usize = 4096;
    let bytes = N * 4;
    let src = crate::cuda::CudaState::cuda_malloc(bytes);
    let dst = crate::cuda::CudaState::cuda_malloc(bytes);
    assert!(!src.is_null() && !dst.is_null(), "probe buffers allocated");
    let pattern: Vec<f32> = (0..N).map(|i| ((i % 251) as f32) + 0.25).collect();
    let raw = unsafe { std::slice::from_raw_parts(pattern.as_ptr() as *const u8, bytes) };
    // The *wrong* value, so `dst == src` can only come from the graph's own
    // recorded copy — the value arm must not be satisfied by the setup
    // (gate contract rule 1).
    let other: Vec<f32> = (0..N).map(|i| ((i % 97) as f32) - 3.5).collect();
    let other_raw = unsafe { std::slice::from_raw_parts(other.as_ptr() as *const u8, bytes) };
    state.copy_to_device(raw, src);
    state.copy_to_device(other_raw, dst);

    for it in 0..iterations {
        // Seed `dst` with the wrong value again: the graph replays the
        // recorded `copy_device_to_device(src → dst)`, so only a replayed
        // graph makes `dst` equal `src`.
        state.copy_to_device(other_raw, dst);
        let b1 = std::sync::Barrier::new(2);
        let b2 = std::sync::Barrier::new(2);
        let outcome: std::sync::Mutex<Option<(bool, i32, bool)>> = std::sync::Mutex::new(None);
        let name = format!("probe.capturectx.{it}");

        // Raw pointers are not `Send`; carry addresses and rebuild them in
        // each closure.
        let stream_addr = stream as usize;
        let (src_addr, dst_addr) = (src as usize, dst as usize);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let stream = stream_addr as *mut std::ffi::c_void;
                let (src, dst) = (
                    src_addr as *mut std::ffi::c_void,
                    dst_addr as *mut std::ffi::c_void,
                );
                let _bound = crate::cuda::bind_stream(stream);
                assert!(
                    state.graph_begin_capture(),
                    "graph_begin_capture must open on the probe stream"
                );
                // A capturable op so the window is not empty.
                state.copy_device_to_device(src, dst, bytes);
                b1.wait(); // the window is open; let A register
                b2.wait(); // A is done; close the window
                let exec = state.graph_end_capture_to_exec();
                let code = crate::cuda::last_capture_end_code();
                let launched = !exec.is_null() && state.graph_launch_exec(exec);
                if launched {
                    state.sync();
                }
                state.graph_destroy(exec);
                *outcome.lock().unwrap() = Some((!exec.is_null(), code, launched));
            });
            scope.spawn(|| {
                b1.wait();
                if legacy_memcpy {
                    state.register_weight_blocking_legacy(&name, raw);
                } else {
                    state.register_weight(&name, raw);
                }
                b2.wait();
            });
        });

        let (exec_ok, code, launched) = outcome
            .lock()
            .unwrap()
            .expect("the capturing thread recorded its outcome");
        eprintln!(
            "PROBE: stream={} mode={} registration={} iter={it} exec_ok={exec_ok} \
             end_code={code} launched={launched}",
            if use_context_stream {
                "context"
            } else {
                "instance"
            },
            std::env::var("MINFER_CUDA_CAPTURE_MODE").unwrap_or_else(|_| "default".into()),
            if legacy_memcpy { "blocking" } else { "ordered" },
        );
        assert_eq!(
            code,
            0,
            "iteration {it}: the capture window was invalidated (code {code} = {}); the \
             window must survive a registration on another thread",
            crate::cuda::cuda_error_name(code)
        );
        assert!(exec_ok, "iteration {it}: capture produced no exec");
        assert!(
            launched,
            "iteration {it}: the captured graph did not launch"
        );

        // The value arm: the replayed copy must have produced `src`'s bytes.
        let mut got = vec![0u8; bytes];
        state.sync();
        state.copy_from_device_pinned(dst as *const std::ffi::c_void, &mut got);
        assert_eq!(
            got, raw,
            "iteration {it}: the captured graph's output does not match the bytes it recorded"
        );
    }

    crate::cuda::CudaState::cuda_free(src);
    crate::cuda::CudaState::cuda_free(dst);
    if !use_context_stream {
        state.destroy_stream(stream);
    }
}

#[test]
fn cuda_pinned_readback_roundtrip() {
    // 5.6 MB > the 4 MiB initial pinned readback buffer: exercises the
    // grow-on-demand path of copy_from_device_pinned (R3-A2).
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut cb = CudaBackend::new().expect("backend after device init");
    let n = 1_400_000usize; // elements — alloc_buffer takes an element count
    let id = cb.alloc_buffer(n);
    let data: Vec<f32> = (0..n).map(|i| (i % 997) as f32 + 0.5).collect();
    cb.write_host(id, &data).unwrap();
    let got = cb.copy_to_host(id).unwrap();
    assert_eq!(got.len(), n);
    assert_eq!(got, data, "full roundtrip mismatch");
}

#[test]
fn copy_across_cpu_to_cuda_and_back() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut b = crate::graph::builder::GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    alloc.alloc_graph(&g).unwrap();
    let data = [1.0f32, -2.0, 3.0, 4.0];
    alloc.fill_input(&g, "x", &data).unwrap();

    let canon = alloc.node_buffer(x).unwrap();
    // CPU → CUDA staging copy: canonical buffer untouched, cross map holds
    // the device copy (Phase 7c: the old remap-into-node_to_buf semantics
    // broke re-execution of reused graphs — the producing split found its
    // buffer remapped to another backend on the next execute)
    alloc
        .copy_across(1, x, crate::graph::Backend::CUDA)
        .unwrap();
    let cross = alloc
        .cross_buffer(1, x, crate::graph::Backend::CUDA)
        .expect("cross staging buffer");
    assert_eq!(cross.backend, crate::graph::Backend::CUDA);
    assert_eq!(
        alloc.node_buffer(x).unwrap(),
        canon,
        "canonical buffer must not be remapped"
    );
    assert_eq!(alloc.copy_to_cpu(x).unwrap(), data.to_vec());
    // re-copy (same dst) reuses the same staging buffer id
    alloc
        .copy_across(1, x, crate::graph::Backend::CUDA)
        .unwrap();
    assert_eq!(
        alloc
            .cross_buffer(1, x, crate::graph::Backend::CUDA)
            .unwrap()
            .id,
        cross.id
    );
    // a same-backend copy is a no-op and does not create staging for CPU
    alloc.copy_across(1, x, crate::graph::Backend::CPU).unwrap();
    assert!(
        alloc
            .cross_buffer(1, x, crate::graph::Backend::CPU)
            .is_none(),
        "the copy was already on the destination backend"
    );
    assert!(alloc
        .cross_buffer(1, x, crate::graph::Backend::CUDA)
        .is_some());
    // E4 S3: a rebuild keeps the staging buffer — it is keyed by (graph uid, node,
    // backend), so it belongs to this graph's shape and survives a re-map. Re-creating it
    // per rebuild leaked (staging is `alloc_fresh`, which never recycles).
    alloc.alloc_graph(&g).unwrap();
    assert_eq!(
        alloc
            .cross_buffer(1, x, crate::graph::Backend::CUDA)
            .expect("staging survives a re-map")
            .id,
        cross.id,
        "and it is the same buffer, not a fresh one"
    );
    // A different graph (its own uid) gets its own entry: node ids restart per graph.
    assert!(alloc
        .cross_buffer(2, x, crate::graph::Backend::CUDA)
        .is_none());
}

/// F5 ([#58]) device gate: **a real split graph's boundary copies out of the
/// device are asynchronous, the boundary waits once per staged input, and the
/// result is bitwise identical to the synchronous reference.**
///
/// This is the cheap, focused half of the F5 acceptance (the real-model half is
/// `models::qwen2::graph::async_cross_copies_never_block_and_stay_bitwise_identical`):
/// a three-node graph whose middle node is pinned to the device makes the
/// scheduler alternate CPU → CUDA → CPU, so the boundary stages one value *into*
/// the device and one *out of* it. Both modes run the same kernels in the same
/// order — only the transfer differs — so bitwise equality is the honest claim.
///
/// The counters are deterministic, which is why this is the mutation gate: with
/// the boundary's `await_cross` loop removed, `copies == waits` fails **and**
/// the consumer's read of the still-pending staging entry is a loud error.
#[test]
fn a_split_graph_waits_once_per_staged_copy_and_stays_bitwise() {
    use crate::graph::copystats::{self, CrossCopyStats};
    use crate::graph::scheduler::BackendScheduler;

    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let data = [1.0f32, -2.0, 3.5, 4.25];
    // x → silu (device) → add(silu, x) (CPU). Every backend is set explicitly:
    // `split_graph` makes an unassigned node inherit the previous one's backend.
    let build = || {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let o = b.add(s, x);
        b.output(o);
        let mut g = b.build();
        g.nodes[0].backend = Some(crate::graph::Backend::CPU);
        g.nodes[1].backend = Some(crate::graph::Backend::CUDA);
        g.nodes[2].backend = Some(crate::graph::Backend::CPU);
        g
    };
    // The mode override is process-wide, so the whole measurement holds the
    // gate (the parallel harness shares the process).
    let gate = copystats::gate();
    let run = |sync: bool| -> (Vec<f32>, CrossCopyStats, u64, u64) {
        let _mode = copystats::set_sync_for_test(sync);
        let g = build();
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_cuda(), "a CUDA device answered the probe");
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &data).unwrap();
        let before = alloc.cross_stats();
        let readbacks_before = alloc.cuda().unwrap().blocking_readback_count();
        // #185: read the count through the backend, never the process-wide
        // `cuda::stream_sync_count()` — a concurrent device test's syncs would
        // otherwise land inside this delta (the failure the ticket records).
        let syncs_before = alloc.cuda().unwrap().stream_sync_count();

        BackendScheduler::new().execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, 2).unwrap().to_vec();
        let stats = alloc.cross_stats().delta(before);
        let readbacks = alloc.cuda().unwrap().blocking_readback_count() - readbacks_before;
        (
            got,
            stats,
            readbacks,
            alloc.cuda().unwrap().stream_sync_count() - syncs_before,
        )
    };

    let (async_out, a, a_readbacks, a_syncs) = run(false);
    let (sync_out, s, s_readbacks, s_syncs) = run(true);
    drop(gate);

    eprintln!(
        "[f5] split graph: async copies={} waits={} blocking={} async_host={} event_syncs={} \
         reads={} syncs={} | sync copies={} waits={} blocking={} reads={} syncs={}",
        a.copies,
        a.waits,
        a.blocking_host_copies,
        a.async_host_copies,
        a.event_syncs,
        a_readbacks,
        a_syncs,
        s.copies,
        s.waits,
        s.blocking_host_copies,
        s_readbacks,
        s_syncs
    );

    // The graph really crosses the boundary in both directions: one CPU→CUDA
    // copy (node 0 into the device split) and one CUDA→CPU copy (node 1 back
    // out). Node 0 feeding the last split is a same-backend input and must not
    // be counted.
    assert_eq!(a.copies, 2, "one copy in, one copy out");
    assert_eq!(
        a.async_host_copies, 1,
        "the CUDA→CPU copy took the async path"
    );
    assert_eq!(a.event_syncs, 1, "…and owes exactly one event wait");
    assert!(
        a.all_copies_awaited(),
        "one wait per staged input: {} copies, {} waits",
        a.copies,
        a.waits
    );
    assert_eq!(
        a.blocking_host_copies, 0,
        "no blocking device→host copy on the boundary"
    );
    assert_eq!(a_readbacks, 0, "no blocking readback on the boundary");
    assert!(
        a_syncs < s_syncs,
        "the async boundary removes the per-copy stream sync ({a_syncs} vs {s_syncs})"
    );
    assert_eq!(
        s.blocking_host_copies, 1,
        "the synchronous reference blocks on the device→host copy — the 'before' number"
    );
    assert!(s_readbacks >= 1, "the device counter sees the sync path");
    assert_eq!(
        async_out, sync_out,
        "the async staging copies must be bitwise identical to the synchronous reference"
    );
    let silu = |v: f32| v / (1.0 + (-v).exp());
    for (i, v) in data.iter().enumerate() {
        assert!((async_out[i] - (silu(*v) + v)).abs() < 1e-5);
    }
}

#[test]
fn kv_persistent_regions_survive_realloc() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut b = crate::graph::builder::GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], DType::I32);
    let k = b.input("k", [16, 1, 1, 1], DType::F32);
    let v = b.input("v", [16, 1, 1, 1], DType::F32);
    let store = b.kvcache_store(0, k, v, 1024);
    let load = b.kvcache_load(0, 16, 1024, 2);
    b.output(load);
    let mut g = b.build();
    g.nodes[store].backend = Some(crate::graph::Backend::CUDA);
    g.nodes[load].backend = Some(crate::graph::Backend::CUDA);

    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    alloc.alloc_graph(&g).unwrap();
    let pair = alloc.kv_pair(0).unwrap();

    // the store node's buffer IS the K region, on the CUDA pool
    let kbuf = alloc.node_buffer(store).unwrap();
    assert_eq!(kbuf.backend, crate::graph::Backend::CUDA);
    assert_eq!(kbuf.id, pair.0);
    {
        let c = alloc.cuda_mut().unwrap();
        c.write_host(kbuf.id, &[7.5f32; 16]).unwrap();
    }

    // rebuild: liveness buffers recycle, KV regions survive unchanged
    alloc.alloc_graph(&g).unwrap();
    assert_eq!(alloc.kv_pair(0).unwrap(), pair);
    let back = alloc.copy_to_cpu(store).unwrap();
    assert_eq!(&back[..16], &[7.5f32; 16]);
}

// ─── Phase 7b: per-op dispatch parity ───────────────────────

use crate::graph::ops::{AttnMeta, AttnMode, RoPEMeta};
use crate::tensor::{Tensor, TensorType};

/// Fresh backend on an initialized device (None → skip on no-GPU hosts).
fn pool() -> Option<CudaBackend> {
    device()?;
    CudaBackend::new()
}

/// D5-R follow-up (doc 89): row-marginal localization bench for the
/// multi-token matmul kernels. Runs the REAL dispatch path (graph
/// execute_node -> quantize + kernel) at nt = 1..8 over real 14B shapes
/// with a cold-L2 protocol: each nt owns NC independent weight copies
/// (>L2 aggregate) cycled so no copy is revisited within 2 runs — L2 is
/// evicted between uses exactly like in a real forward, and runs per
/// (uid, range) stay below the 3-run capture trigger so timing is never
/// capture/replay. Per-run cost = one synchronized burst / R; the
/// per-row marginal = (t(nt) - t(1)) / (nt - 1), attributable to extra
/// in-kernel row work only (launch count is nt-invariant).
#[test]
fn cuda_row_marginal_bench() {
    if std::env::var("MINFER_BENCH_ROW_MARGINAL").is_err() {
        return;
    }
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();

    fn gen_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    // (label, type, od, id, padded, weight copies). Aggregate weight
    // footprint per case > 126 MB L2, and >= 8x per-copy footprint
    // streams between revisits.
    let cases: Vec<(&str, TensorType, usize, usize, bool, usize)> = vec![
        ("q4k_attn_qo", TensorType::Q4_K, 5120, 5120, false, 40),
        ("q4k_ffn_up", TensorType::Q4_K, 13824, 5120, false, 40),
        ("q6k_ffn_down", TensorType::Q6_K, 5120, 13824, true, 24),
    ];
    let nts = [1usize, 2, 3, 4, 6, 8];

    for (ci, (label, tt, od, id_, padded, nc)) in cases.into_iter().enumerate() {
        let nbe = (id_ + 255) / 256;
        let row_bytes = match tt {
            TensorType::Q4_K => nbe * 144,
            TensorType::Q6_K => nbe * 210,
            other => panic!("unexpected {other:?}"),
        };
        let wb = gen_bytes(od * row_bytes, 0x5EED_C0DE + ci as u64);
        let mut wts = Vec::with_capacity(nc);
        for j in 0..nc {
            let name = format!("w{ci}_{j}");
            let mut wt = Tensor::from_data(tt, &[id_ as i64, od as i64, 1, 1], wb.clone());
            wt.name = name.clone();
            if tt == TensorType::Q6_K && padded {
                cb.state.register_weight_q6k_padded(&name, &wb, od, id_);
            } else {
                cb.state.register_weight(&name, &wb);
            }
            wts.push(wt);
        }

        let mut lines = Vec::new();
        for &nt in nts.iter() {
            // NC graphs (one per weight copy) sharing one x/out buffer.
            let xs: Vec<f32> = (0..id_ * nt)
                .map(|i| ((i * 2654435761 % 2000) as f32 / 1000.0 - 1.0))
                .collect();
            let mut graphs = Vec::with_capacity(nc);
            for wt in &wts {
                let mut b = GraphBuilder::new();
                let x = b.input("x", [id_, nt, 1, 1], DType::F32);
                let m = b.matmul(x, wt, None);
                b.output(m);
                let g = b.build();
                graphs.push(g);
            }
            let xb = cb.alloc_buffer(id_ * nt);
            cb.write_host(xb, &xs).unwrap();
            let ob = cb.alloc_buffer(od * nt);

            let reps = 2 * nc; // < 3 runs per (uid, range): no capture
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                cb.exec_ids(
                    &graphs[r % nc].nodes[graphs[r % nc].outputs[0]],
                    &[xb],
                    ob,
                    None,
                )
                .unwrap();
            }
            cb.synchronize();
            let per_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            lines.push((nt, per_us));
            let _ = cb.copy_to_host(ob).unwrap(); // keep result live
        }
        let t1 = lines[0].1;
        let gb = (od * row_bytes) as f64 / 1e9;
        eprintln!("[bench] {label} ({tt:?} {od}x{id_}, {gb:.1} MB/copy, NC={nc}):");
        for (nt, us) in &lines {
            let bw = gb * 1e3 / (us / 1e3) / 1e3;
            eprintln!(
                "[bench]   nt={nt}: {:9.1} us/run  ({bw:5.0} GB/s w-stream)",
                us
            );
        }
        for w in [2usize, 3, 4] {
            let tn = lines[w - 1].1;
            eprintln!(
                "[bench]   marginal/row (1->{w}): {:6.2} us per matmul per forward",
                (tn - t1) / (w - 1) as f64
            );
        }
        let _ = &cb;
    }
}

fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    assert!(
        worst.0 <= tol,
        "{name}: max diff {} at {} (got {}, want {})",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
}

#[test]
fn cuda_elementwise_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let n = 257usize; // odd size exercises the elementwise tail guard
    let mut b = GraphBuilder::new();
    let a = b.input("a", [n, 1, 1, 1], DType::F32);
    let c = b.input("c", [n, 1, 1, 1], DType::F32);
    let add = b.add(a, c);
    let mul = b.mul(add, c);
    let sw = b.swiglu(mul, c); // gate is the RAW pre-activation
    let silu = b.silu(mul);
    b.output(silu);
    b.output(sw);
    let g = b.build();

    let (x, y) = (cb.alloc_buffer(n), cb.alloc_buffer(n));
    let (t1, t2, t3) = (cb.alloc_buffer(n), cb.alloc_buffer(n), cb.alloc_buffer(n));
    let xs: Vec<f32> = (0..n).map(|i| ((i * 37) % 23) as f32 / 4.0 - 2.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| ((i * 91) % 17) as f32 / 3.0 - 2.0).collect();
    cb.write_host(x, &xs).unwrap();
    cb.write_host(y, &ys).unwrap();

    cb.exec_ids(&g.nodes[add], &[x, y], t1, None).unwrap();
    cb.exec_ids(&g.nodes[mul], &[t1, y], t2, None).unwrap();
    // SwiGLU consumes the RAW mul output, so it must run before the
    // in-place Silu overwrites t2 (alias path, graph rules §5).
    cb.exec_ids(&g.nodes[sw], &[t2, y], t3, None).unwrap();
    cb.exec_ids(&g.nodes[silu], &[t2], t2, None).unwrap();

    // Host reference through the same vec_ops the CPU backend uses.
    let mut r1 = vec![0f32; n];
    crate::vec_ops::vec_add_f32(n, &mut r1, &xs, &ys);
    assert_eq!(cb.copy_to_host(t1).unwrap(), r1, "add must be bit-exact");
    let mut r2 = vec![0f32; n];
    crate::vec_ops::vec_mul_f32(n, &mut r2, &r1, &ys);
    let mut r3 = vec![0f32; n];
    crate::vec_ops::vec_silu_f32(n, &mut r3, &r2);
    let got2 = cb.copy_to_host(t2).unwrap();
    assert_close("mul+silu (in-place)", &got2, &r3, 1e-5);
    let mut r4 = vec![0f32; n];
    crate::vec_ops::vec_mul_f32(n, &mut r4, &r3, &ys);
    assert_close("swiglu", &cb.copy_to_host(t3).unwrap(), &r4, 1e-5);
}

#[test]
fn cuda_norm_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // RmsNorm: d=64, nt=5
    let (d, nt) = (64usize, 5usize);
    let w: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
    let wbytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    cb.state.register_weight("nw", &wbytes);
    let mut wt = Tensor::from_data(TensorType::F32, &[d as i64, 1, 1, 1], wbytes.clone());
    wt.name = "nw".to_string();

    // QkNorm: hd=16, nh=4, nt=3 — rows (t*nh + h) form a contiguous
    // [nt*nh, hd] matrix, so the same rms_norm kernel covers it.
    let (hd, nh, nt2) = (16usize, 4usize, 3usize);
    let qw: Vec<f32> = (0..hd).map(|i| 1.0 / (1.0 + i as f32)).collect();
    let qbytes: Vec<u8> = qw.iter().flat_map(|v| v.to_le_bytes()).collect();
    cb.state.register_weight("qw", &qbytes);
    let mut qwt = Tensor::from_data(TensorType::F32, &[hd as i64, 1, 1, 1], qbytes);
    qwt.name = "qw".to_string();

    let mut b = GraphBuilder::new();
    let x = b.input("x", [d, nt, 1, 1], DType::F32);
    let rn = b.rms_norm(x, Some(&wt), 1e-5);
    let q = b.input("q", [hd * nh, nt2, 1, 1], DType::F32);
    let qn = b.qk_norm(q, Some(&qwt), hd, nh, 1e-5);
    b.output(rn);
    b.output(qn);
    let g = b.build();

    let xb = cb.alloc_buffer(d * nt);
    let xs: Vec<f32> = (0..d * nt)
        .map(|i| ((i * 53) % 31) as f32 / 7.0 - 2.0)
        .collect();
    cb.write_host(xb, &xs).unwrap();
    let ob = cb.alloc_buffer(d * nt);
    cb.exec_ids(&g.nodes[rn], &[xb], ob, None).unwrap();

    let qb = cb.alloc_buffer(hd * nh * nt2);
    let qs: Vec<f32> = (0..hd * nh * nt2)
        .map(|i| ((i * 71) % 29) as f32 / 6.0 - 2.5)
        .collect();
    cb.write_host(qb, &qs).unwrap();
    let qo = cb.alloc_buffer(hd * nh * nt2);
    cb.exec_ids(&g.nodes[qn], &[qb], qo, None).unwrap();

    let mut want = vec![0f32; d * nt];
    for t in 0..nt {
        crate::vec_ops::rms_norm_fused_f32(
            d,
            &mut want[t * d..(t + 1) * d],
            &xs[t * d..(t + 1) * d],
            &w,
            1e-5,
        );
    }
    assert_close("rms_norm", &cb.copy_to_host(ob).unwrap(), &want, 1e-4);

    let mut want2 = vec![0f32; hd * nh * nt2];
    for r in 0..nh * nt2 {
        crate::vec_ops::rms_norm_fused_f32(
            hd,
            &mut want2[r * hd..(r + 1) * hd],
            &qs[r * hd..(r + 1) * hd],
            &qw,
            1e-5,
        );
    }
    assert_close("qk_norm", &cb.copy_to_host(qo).unwrap(), &want2, 1e-4);
}

/// #169, the CUDA half: a norm weight whose registered length is not the
/// f32 the `rms_norm` kernel indexes (`d*4` bytes) must be refused **before**
/// the launch, because the kernel reads `d*4` bytes regardless and an f16
/// norm (2 B/element) would be read past its end.
///
/// The two arms differ only in that property: the same graph, the same
/// `d = 64`, both names registered, both dims valid float4 multiples — so
/// the **f32** arm (the control) can only pass and the **f16** arm can only
/// fail through the size check. Before #169 the registry lookup alone
/// admitted the f16 arm, which is what makes this a value gate rather than
/// a relation: it asserts the refusal's own text names both lengths.
#[test]
fn cuda_norm_weight_size_is_part_of_the_invariant() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (d, eps) = (64usize, 1e-5f32);
    let run = |cb: &mut CudaBackend, name: &str, t: TensorType, bytes: Vec<u8>| {
        cb.state.register_weight(name, &bytes);
        let mut wt = Tensor::from_data(t, &[d as i64, 1, 1, 1], bytes);
        wt.name = name.to_string();
        let mut b = GraphBuilder::new();
        let x = b.input("x", [d, 1, 1, 1], DType::F32);
        let rn = b.rms_norm(x, Some(&wt), eps);
        b.output(rn);
        let g = b.build();
        let xb = cb.alloc_buffer(d);
        cb.write_host(xb, &vec![1.0f32; d]).unwrap();
        let ob = cb.alloc_buffer(d);
        cb.exec_ids(&g.nodes[rn], &[xb], ob, None)
    };

    // Control arm: f32, exactly `d*4` bytes registered — must execute.
    let w: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
    let wbytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    run(&mut cb, "n169_f32", TensorType::F32, wbytes)
        .expect("an f32 norm weight of d*4 bytes must execute");

    // Property arm: the same norm weight at 2 B/element. Registered (so the
    // "not registered" refusal cannot be the one that fires) and d is a
    // valid float4 dim, so only the size check can refuse it.
    let f16bytes: Vec<u8> = vec![0u8; d * 2];
    let err = run(&mut cb, "n169_f16", TensorType::F16, f16bytes)
        .expect_err("an f16 norm weight must be refused before the launch");
    assert!(err.contains("n169_f16"), "{err}");
    assert!(
        err.contains(&format!("{} B", d * 2)),
        "the refusal must name the registered length ({} B): {err}",
        d * 2
    );
    assert!(
        err.contains(&format!("{} B", d * 4)),
        "the refusal must name the length the kernel reads ({} B): {err}",
        d * 4
    );
    assert!(err.contains("f16-norm"), "{err}");
}

#[test]
fn cuda_matmul_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (od, id_, nt) = (32usize, 64usize, 3usize);
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
        .collect();
    let bias: Vec<f32> = (0..od).map(|i| (i % 5) as f32 / 10.0).collect();

    // Q8_0 weight [out][in] row-major, quantized per row
    let wf8: Vec<f32> = (0..od * id_)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let mut w8b = Vec::new();
    for r in 0..od {
        w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
            &wf8[r * id_..(r + 1) * id_],
        ));
    }
    let mut w8 = Tensor::from_data(
        TensorType::Q8_0,
        &[id_ as i64, od as i64, 1, 1],
        w8b.clone(),
    );
    w8.name = "mw8".to_string();
    cb.state.register_weight("mw8", &w8b);
    let biasb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    cb.state.register_weight("mb", &biasb);
    let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], biasb);
    bt.name = "mb".to_string();

    // Q4_0 weight (18 bytes per 32 values: f16 d + 16 nibbles)
    let wf4: Vec<f32> = (0..od * id_)
        .map(|i| ((i * 40503) % 991) as f32 / 496.0 - 1.0)
        .collect();
    let mut w4b = Vec::new();
    for r in 0..od {
        let row = &wf4[r * id_..(r + 1) * id_];
        for bi in 0..id_ / 32 {
            let blk = &row[bi * 32..bi * 32 + 32];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let dsc = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
            w4b.extend_from_slice(&half::f16::from_f32(dsc).to_le_bytes());
            for j in 0..16 {
                let q0 = ((blk[j] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                let q1 = ((blk[j + 16] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                w4b.push(q0 | (q1 << 4));
            }
        }
    }
    let mut w4 = Tensor::from_data(
        TensorType::Q4_0,
        &[id_ as i64, od as i64, 1, 1],
        w4b.clone(),
    );
    w4.name = "mw4".to_string();
    cb.state.register_weight("mw4", &w4b);

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let m8 = b.matmul(x, &w8, Some(&bt));
    let m4 = b.matmul(x, &w4, Some(&bt));
    b.output(m8);
    b.output(m4);
    let g = b.build();

    let xb = cb.alloc_buffer(id_ * nt);
    cb.write_host(xb, &xs).unwrap();
    let (o8, o4) = (cb.alloc_buffer(od * nt), cb.alloc_buffer(od * nt));
    cb.exec_ids(&g.nodes[m8], &[xb], o8, None).unwrap();
    cb.exec_ids(&g.nodes[m4], &[xb], o4, None).unwrap();

    // References: dequantized weight rows × f32 activations + bias
    // (embed_tokens doubles as the row dequantizer for these types).
    let mut dq8 = vec![0f32; od * id_];
    crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8, &mut dq8, id_);
    let mut dq4 = vec![0f32; od * id_];
    crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w4, &mut dq4, id_);
    for (name, o, dq, quantize_acts) in [
        ("q8_0 matmul", o8, &dq8, false),
        // 8c: prefill Q4_0 runs the Q8_0-activation GEMM (the CPU path
        // has always quantized activations too) — mirror that in the
        // reference and keep a tight tolerance.
        ("q4_0 matmul", o4, &dq4, true),
    ] {
        let got = cb.copy_to_host(o).unwrap();
        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            let xrow = &xs[t * id_..(t + 1) * id_];
            let acts: Vec<f32> = if quantize_acts {
                let q8 = crate::quants::quantize_row_q8_0(xrow);
                (0..id_)
                    .map(|i| {
                        let b = i / 32;
                        let d8 = half::f16::from_le_bytes([q8[b * 34], q8[b * 34 + 1]]).to_f32();
                        d8 * q8[b * 34 + 2 + (i % 32)] as i8 as f32
                    })
                    .collect()
            } else {
                xrow.to_vec()
            };
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += dq[r * id_ + i] * acts[i];
                }
                want[t * od + r] = acc + bias[r];
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(name, &got, &want, scale * 2e-2);
    }
}

/// 7e②: K-quant matmul parity (Q4_K + Q6_K). The reference dequantizes
/// each row with an independent in-test implementation of the
/// llama.cpp block layout and dots it with the f32 activations. The
/// original scalar CUDA kernels and the 7e② vectorized ones must both
/// agree with it (coverage gap found in 7e②: q6_K previously had NO
/// parity test, which let a broken vectorized variant pass the suite).
#[test]
fn cuda_kquant_matmul_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // id_ = 512 = 2 super-blocks of 256; od = 8 rows (NR0 = 2 → 4 row
    // pairs across 2 warps per block).
    let (od, id_, nt) = (8usize, 512usize, 3usize);
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
        .collect();

    // get_scale_min_k4 (llama.cpp Q4_K scale packing, reimplemented
    // here independently of the kernel under test).
    fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    // ── Q4_K tensor: 144 bytes per 256-element super-block ──
    // layout: f16 d, f16 dmin, u8 scales[12], nibble bytes qs[128]
    let mut w4b = Vec::new();
    let mut w4dq = vec![0f32; od * id_];
    for r in 0..od {
        for ib in 0..id_ / 256 {
            let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
            let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
            w4b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            w4b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            let mut scb = [0u8; 12];
            for j in 0..12 {
                scb[j] = ((r * 31 + j * 17 + ib * 5) % 63) as u8;
            }
            w4b.extend_from_slice(&scb);
            let mut qs = [0u8; 128];
            for j in 0..128 {
                let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                qs[j] = lo | (hi << 4);
            }
            w4b.extend_from_slice(&qs);
            // reference dequant: LOW nibbles of bytes[32j..32j+31] are
            // elements [64j..64j+31] (scale 2j), HIGH nibbles are
            // elements [64j+32..64j+63] (scale 2j+1);
            // value = d*sc*nibble - dmin*m
            for j in 0..4 {
                let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                for l in 0..32 {
                    let b = qs[j * 32 + l];
                    let base = r * id_ + ib * 256 + j * 64;
                    w4dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                    w4dq[base + 32 + l] = (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                }
            }
        }
    }

    // ── Q6_K tensor: 210 bytes per 256-element super-block ──
    // layout: ql[128], qh[64], i8 scales[16], f16 d
    let mut w6b = Vec::new();
    let mut w6dq = vec![0f32; od * id_];
    for r in 0..od {
        for ib in 0..id_ / 256 {
            let d = 0.027f32 + 0.004 * ((r * 11 + ib * 7) % 6) as f32;
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut sc = [0i8; 16];
            for i in 0..128 {
                ql[i] = ((r * 29 + i * 7 + ib * 3) % 255) as u8;
            }
            for i in 0..64 {
                qh[i] = ((r * 17 + i * 13 + ib * 11) % 255) as u8;
            }
            for i in 0..16 {
                sc[i] = (((r * 5 + i * 3 + ib) % 15) as i8) - 7;
            }
            w6b.extend_from_slice(&ql);
            w6b.extend_from_slice(&qh);
            w6b.extend(sc.iter().map(|&x| x as u8));
            w6b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            // reference dequant (llama.cpp Q6_K layout):
            // value = d * sc[n*8 + l/16 + t*2] * (nibble|2bits<<4 - 32)
            for n in 0..2usize {
                let qlh = &ql[n * 64..n * 64 + 64];
                let qhh = &qh[n * 32..n * 32 + 32];
                for l in 0..32usize {
                    let is = l / 16;
                    let q1 = ((qlh[l] & 0xF) as i32 | (((qhh[l] >> 0) as i32 & 3) << 4)) - 32;
                    let q2 = ((qlh[l + 32] & 0xF) as i32 | (((qhh[l] >> 2) as i32 & 3) << 4)) - 32;
                    let q3 = ((qlh[l] >> 4) as i32 | (((qhh[l] >> 4) as i32 & 3) << 4)) - 32;
                    let q4 = ((qlh[l + 32] >> 4) as i32 | (((qhh[l] >> 6) as i32 & 3) << 4)) - 32;
                    let base = r * id_ + ib * 256 + n * 128;
                    w6dq[base + l] = d * sc[n * 8 + is] as f32 * q1 as f32;
                    w6dq[base + l + 32] = d * sc[n * 8 + is + 2] as f32 * q2 as f32;
                    w6dq[base + l + 64] = d * sc[n * 8 + is + 4] as f32 * q3 as f32;
                    w6dq[base + l + 96] = d * sc[n * 8 + is + 6] as f32 * q4 as f32;
                }
            }
        }
    }

    let mut w4t = Tensor::from_data(
        TensorType::Q4_K,
        &[id_ as i64, od as i64, 1, 1],
        w4b.clone(),
    );
    w4t.name = "mw4k".to_string();
    cb.state.register_weight("mw4k", &w4b);
    let mut w6t = Tensor::from_data(
        TensorType::Q6_K,
        &[id_ as i64, od as i64, 1, 1],
        w6b.clone(),
    );
    w6t.name = "mw6k".to_string();
    cb.state.register_weight("mw6k", &w6b);
    // 7e② padded layout path (register_weight_q6k_padded)
    cb.state.register_weight_q6k_padded("mw6kp", &w6b, od, id_);
    assert!(cb.state.is_weight_padded("mw6kp"));

    let mut w6pt = Tensor::from_data(
        TensorType::Q6_K,
        &[id_ as i64, od as i64, 1, 1],
        w6b.clone(),
    );
    w6pt.name = "mw6kp".to_string();

    // ── F32 weight (7e④): aligned id (512) and odd id (513, scalar path)
    let wfb: Vec<u8> = w4dq.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut wft = Tensor::from_data(TensorType::F32, &[id_ as i64, od as i64, 1, 1], wfb.clone());
    wft.name = "mwf32".to_string();
    cb.state.register_weight("mwf32", &wfb);
    // odd id: 513-wide rows (first 512 = w4dq, element 512 synthetic)
    let (od_o, id_o) = (8usize, 513usize);
    let mut wfo_vals = Vec::with_capacity(od_o * id_o);
    for r in 0..od_o {
        for i in 0..id_o {
            wfo_vals.push(if i < id_ {
                w4dq[r * id_ + i]
            } else {
                (r + 1) as f32 * 0.25
            });
        }
    }
    let wfo: Vec<u8> = wfo_vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut wfot = Tensor::from_data(
        TensorType::F32,
        &[id_o as i64, od_o as i64, 1, 1],
        wfo.clone(),
    );
    wfot.name = "mwf32o".to_string();
    cb.state.register_weight("mwf32o", &wfo);

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let m4 = b.matmul(x, &w4t, None);
    let m6 = b.matmul(x, &w6t, None);
    let m6p = b.matmul(x, &w6pt, None);
    let mf = b.matmul(x, &wft, None);
    b.output(m4);
    b.output(m6);
    b.output(m6p);
    b.output(mf);
    // odd-id graph: x sliced to id_ = 513
    let xo = b.input("xo", [id_o, nt, 1, 1], DType::F32);
    let mfo = b.matmul(xo, &wfot, None);
    b.output(mfo);
    let g = b.build();

    let xb = cb.alloc_buffer(id_ * nt);
    cb.write_host(xb, &xs).unwrap();
    let (o4, o6, o6p) = (
        cb.alloc_buffer(od * nt),
        cb.alloc_buffer(od * nt),
        cb.alloc_buffer(od * nt),
    );
    let of = cb.alloc_buffer(od * nt);
    cb.exec_ids(&g.nodes[m4], &[xb], o4, None).unwrap();
    cb.exec_ids(&g.nodes[m6], &[xb], o6, None).unwrap();
    cb.exec_ids(&g.nodes[m6p], &[xb], o6p, None).unwrap();
    cb.exec_ids(&g.nodes[mf], &[xb], of, None).unwrap();
    let mut xso = xs.clone();
    xso.resize(id_o * nt, 0.25f32); // extend for the odd-id input
    let xob = cb.alloc_buffer(id_o * nt);
    cb.write_host(xob, &xso).unwrap();
    let ofo = cb.alloc_buffer(od_o * nt);
    cb.exec_ids(&g.nodes[mfo], &[xob], ofo, None).unwrap();

    for (name, o, dq) in [
        ("q4_k matmul", o4, &w4dq),
        ("q6_k matmul", o6, &w6dq),
        ("q6_k padded matmul", o6p, &w6dq),
    ] {
        let got = cb.copy_to_host(o).unwrap();
        // Step 82: at nt in [2, 8] the q4_K/q6_K arms dispatch to the
        // multi-token MMVQ kernels, whose activations are the pad40 q8
        // plane (f16 d + 2B pad + 32 i8 per 32-element block, per
        // token) — the reference dots the dequantized q8 values, with
        // the same 1e-2 relative tolerance the mmvq parity tests use
        // for the kernel-side quantization rounding.
        let mut x8 = vec![0u8; nt * (id_ / 32) * 40];
        for t in 0..nt {
            for blk in 0..id_ / 32 {
                let base = t * id_ + blk * 32;
                let mut am = 0f32;
                for j in 0..32 {
                    am = am.max(xs[base + j].abs());
                }
                let dd = am / 127.0;
                let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                let off = (t * (id_ / 32) + blk) * 40;
                x8[off..off + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
                for j in 0..32 {
                    let q = (xs[base + j] * di).round().clamp(-128.0, 127.0) as i8;
                    x8[off + 4 + j] = q as u8;
                }
            }
        }
        let dq8 = |t: usize, i: usize| -> f32 {
            let off = (t * (id_ / 32) + i / 32) * 40;
            half::f16::from_le_bytes([x8[off], x8[off + 1]]).to_f32()
                * (x8[off + 4 + (i % 32)] as i8) as f32
        };
        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += dq[r * id_ + i] * dq8(t, i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(name, &got, &want, scale * 1e-2);
    }

    // F32 matmul (aligned + odd-id scalar path) vs the same reference rows
    {
        let got = cb.copy_to_host(of).unwrap();
        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w4dq[r * id_ + i] * xs[t * id_ + i];
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("f32 matmul", &got, &want, scale * 2e-3);
    }
    {
        let got = cb.copy_to_host(ofo).unwrap();
        let mut want = vec![0f32; od_o * nt];
        for t in 0..nt {
            for r in 0..od_o {
                let mut acc = 0f32;
                for i in 0..id_o {
                    acc += wfo_vals[r * id_o + i] * xso[t * id_o + i];
                }
                want[t * od_o + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("f32 matmul odd id", &got, &want, scale * 2e-3);
    }
}

/// 7e⑤: fused FFN gate+up parity — the concat matmul + in-place offset
/// swiglu must equal silu(gate·x)·(up·x) computed on the host, for a
/// plain-registered q4_K concat and a padded-repacked q6_K concat.
/// The reference dequantizes with the same independent in-test block
/// layouts as `cuda_kquant_matmul_parity`.
#[test]
fn cuda_q4k_decode_mmvq_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // the mmvq scratch (buf_q8_decode) is singleton state — serialize the
    // parallel mmvq parity tests so one test's scratch grow cannot free
    // the buffer another test just enqueued kernels against
    let _guard = crate::cuda::CudaState::model_load_guard();
    // 8e-reversal: decode (nt == 1) Q4_K dispatches to the MMVQ structure
    // kernel (q8 activations + dp4a, one 256-thread block per row) when
    // id >= 2048 && id % 32 == 0. Reference: independent dequant of the
    // same bytes + the SAME q8 activation quantization the kernel applies
    // (round-trip error ~2/127 per element), so the tolerance stays tight.
    let (od, id_, nt) = (5120usize, 3584usize, 1usize); // 7B fused-qkv shape
                                                        // activations with real-model spread (RMSNorm outputs reach ±4, and
                                                        // some 32-blocks are near-zero: exercises the f16 d8 denormal range)
    let mut rng_state = 12345u64;
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|_| {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0; // [-1, 1)
            let mag = if (rng_state >> 60) & 7 == 0 {
                1e-5
            } else {
                3.0
            };
            (u as f32) * mag
        })
        .collect();

    fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    let mut w4b = Vec::new();
    let mut w4dq = vec![0f32; od * id_];
    for r in 0..od {
        for ib in 0..id_ / 256 {
            let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
            let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
            w4b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            w4b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            let mut scb = [0u8; 12];
            for j in 0..12 {
                scb[j] = ((r * 31 + j * 17 + ib * 5) % 63) as u8;
            }
            w4b.extend_from_slice(&scb);
            let mut qs = [0u8; 128];
            for j in 0..128 {
                let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                qs[j] = lo | (hi << 4);
            }
            w4b.extend_from_slice(&qs);
            for j in 0..4 {
                let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                for l in 0..32 {
                    let b = qs[j * 32 + l];
                    let base = r * id_ + ib * 256 + j * 64;
                    w4dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                    w4dq[base + 32 + l] = (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                }
            }
        }
    }

    // mirror quantize_q8_0_pad40: per 32-element block, d = amax/127,
    // payload at byte offset 4 (padded 40B blocks)
    let mut x8 = vec![0u8; (id_ / 32) * 40];
    for b in 0..id_ / 32 {
        let mut am = 0f32;
        for j in 0..32 {
            am = am.max(xs[b * 32 + j].abs());
        }
        let dd = am / 127.0;
        let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
        x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
        for j in 0..32 {
            let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
            x8[b * 40 + 4 + j] = q as u8;
        }
    }
    let dq8 = |i: usize| -> f32 {
        half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
            * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
    };

    let mut w4t = Tensor::from_data(
        TensorType::Q4_K,
        &[id_ as i64, od as i64, 1, 1],
        w4b.clone(),
    );
    w4t.name = "mmw4k".to_string();
    cb.state.register_weight("mmw4k", &w4b);

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let m = b.matmul(x, &w4t, None);
    b.output(m);
    let g = b.build();

    let xb = cb.alloc_buffer(id_ * nt);
    cb.write_host(xb, &xs).unwrap();
    let ob = cb.alloc_buffer(od * nt);
    cb.exec_ids(&g.nodes[m], &[xb], ob, None).unwrap();
    let got = cb.copy_to_host(ob).unwrap();

    let mut want = vec![0f32; od * nt];
    for t in 0..nt {
        for r in 0..od {
            let mut acc = 0f32;
            for i in 0..id_ {
                acc += w4dq[r * id_ + i] * dq8(t * id_ + i);
            }
            want[t * od + r] = acc;
        }
    }
    let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
    // q8 activation quantization noise (<= 2/127 per element, random
    // signs over 2048 elements) lands well under 1e-2 of the row scale.
    assert_close("q4_k decode mmvq", &got, &want, scale * 1e-2);

    // ── real 7B weights (Q4_K tensors of the decode path) ──
    // dumped via the ignored `dump_real_q4k_tensor` helper; skipped when
    // the file is absent so the suite stays hermetic.
    let real_shapes = [
        ("real_blk_0_attn_q_weight.bin", 3584usize, 3584usize),
        ("real_blk_0_attn_k_weight.bin", 3584usize, 512usize),
        ("real_blk_0_ffn_gate_weight.bin", 3584usize, 18944usize),
    ];
    for (file, id2, od2) in real_shapes {
        let Ok(wb) = std::fs::read(format!("/tmp/minfer_phase7/{file}")) else {
            break;
        };
        assert_eq!(wb.len(), od2 * (id2 / 256) * 144);
        let mut w2t = Tensor::from_data(
            TensorType::Q4_K,
            &[id2 as i64, od2 as i64, 1, 1],
            wb.clone(),
        );
        w2t.name = "realq4k".to_string();
        cb.state.register_weight("realq4k", &wb);
        let mut b2 = GraphBuilder::new();
        let x2 = b2.input("x2", [id2, 1, 1, 1], DType::F32);
        let m2 = b2.matmul(x2, &w2t, None);
        b2.output(m2);
        let g2 = b2.build();
        let xb2 = cb.alloc_buffer(id2);
        cb.write_host(xb2, &xs).unwrap();
        let ob2 = cb.alloc_buffer(od2);
        cb.exec_ids(&g2.nodes[m2], &[xb2], ob2, None).unwrap();
        let got2 = cb.copy_to_host(ob2).unwrap();

        // CPU reference over the real bytes: dequant (same map as the
        // parity reference above) + q8-quantized activations
        let mut want2 = vec![0f32; od2];
        for r in 0..od2 {
            let mut acc = 0f32;
            for ib in 0..id2 / 256 {
                let blk = &wb[(r * (id2 / 256) + ib) * 144..];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                let mut scb = [0u8; 12];
                scb.copy_from_slice(&blk[4..16]);
                for j in 0..4 {
                    let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                    let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                    for l in 0..32 {
                        let b8 = blk[16 + j * 32 + l];
                        let base = ib * 256 + j * 64;
                        let v_lo = (b8 & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                        let v_hi = (b8 >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                        acc += v_lo * dq8(base + l);
                        acc += v_hi * dq8(base + 32 + l);
                    }
                }
            }
            want2[r] = acc;
        }
        // activations here only cover id2=3584 — regenerate quantization
        // for the real id (xs has id_=3584 entries from the scaled-up test)
        let scale2 = want2.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        let mut worst = (0f32, 0usize);
        for r in 0..od2 {
            let e = (got2[r] - want2[r]).abs();
            if e > worst.0 {
                worst = (e, r);
            }
        }
        println!(
            "real q4_k [{file}]: max err {:.4} at row {} (got {:.4} want {:.4}, scale {scale2:.3})",
            worst.0, worst.1, got2[worst.1], want2[worst.1]
        );
        assert!(
            worst.0 <= scale2 * 1e-2,
            "real q4_k mmvq err {} > {}",
            worst.0,
            scale2 * 1e-2
        );
    }
}

/// 8e follow-up: decode (nt == 1) Q6_K dispatches to the MMVQ structure
/// kernel (16-element units over q8 activations). The synthetic block
/// covers a partial tail super-block (id = 2176 = 8×256 + 128) and full
/// 6-bit scale/nibble ranges; both strides go through direct
/// `q6_k_decode_mmvq` calls — the synthetic shape (od=64) is far below
/// the measured od*id >= 24M dispatch gate on purpose (small shapes
/// stay on the f32 kernel at runtime), and gate coverage comes from the
/// full-size 7B ffn_down real-weight section below.
#[test]
fn cuda_q6k_decode_mmvq_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    // serialize against the other mmvq parity tests (shared scratch)
    let _guard = crate::cuda::CudaState::model_load_guard();
    // two shapes: 2176 (partial tail super-block → v1 kernels) and 2560
    // (full super-blocks → the R2 v2 weight-streaming kernels)
    for (i, (od, id_, nt)) in [(64usize, 2176usize, 1usize), (64usize, 2560usize, 1usize)]
        .into_iter()
        .enumerate()
    {
        let mut rng_state = 12345u64;
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng_state >> 60) & 7 == 0 {
                    1e-5
                } else {
                    3.0
                };
                (u as f32) * mag
            })
            .collect();

        let nbe = (id_ + 255) / 256;
        let mut w6b = Vec::new();
        let mut w6dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.02f32 + 0.004 * ((r * 7 + ib * 3) % 5) as f32;
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                let mut sc = [0u8; 16];
                for s in 0..16usize {
                    sc[s] = (((r * 11 + s * 5 + ib * 3) % 64) as i32 - 32) as u8;
                }
                // only the real elements of a partial tail super-block
                let n_elem = 256usize.min(id_ - ib * 256);
                for e in 0..n_elem {
                    let s = e / 16;
                    let l = e % 16;
                    let chunk = s / 8;
                    let g = (s / 2) % 4;
                    let is = s % 2;
                    let q6 = ((r * 13 + e * 7 + ib * 3) % 64) as u8;
                    w6dq[r * id_ + ib * 256 + e] = d * (sc[s] as i8 as f32) * (q6 as f32 - 32.0);
                    let nib = q6 & 0xF;
                    let hi2 = (q6 >> 4) & 3;
                    let qpos = chunk * 64 + (g % 2) * 32 + is * 16 + l;
                    if g < 2 {
                        ql[qpos] |= nib;
                    } else {
                        ql[qpos] |= nib << 4;
                    }
                    let hpos = chunk * 32 + is * 16 + l;
                    qh[hpos] |= hi2 << (2 * g);
                }
                // block_q6_K field order: ql[128], qh[64], scales[16], d —
                // d is the LAST field (offset 208), not the first
                w6b.extend_from_slice(&ql);
                w6b.extend_from_slice(&qh);
                w6b.extend_from_slice(&sc);
                w6b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            }
        }
        assert_eq!(w6b.len(), od * nbe * 210);

        // mirror quantize_q8_0_pad40 (padded 40B blocks, payload at offset 4)
        let mut x8 = vec![0u8; (id_ / 32) * 40];
        for b in 0..id_ / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x8[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq8 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
                * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };

        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w6dq[r * id_ + i] * dq8(t * id_ + i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));

        // ── padded 224B registration, direct decode call (gate bypassed) ──
        let mut w6t = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6t.name = format!("mw6kp{i}");
        cb.state
            .register_weight_q6k_padded(&format!("mw6kp{i}"), &w6b, od, id_);
        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        cb.state.q6_k_decode_mmvq(
            cb.state.get_weight_ptr(&format!("mw6kp{i}")).unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(ob).unwrap(),
            od,
            id_,
            nt,
            true,
        );
        let got = cb.copy_to_host(ob).unwrap();
        let mut worst = (0f32, 0usize);
        for i in 0..od * nt {
            let e = (got[i] - want[i]).abs();
            if e > worst.0 {
                worst = (e, i);
            }
        }
        assert!(
            worst.0 <= scale * 1e-2,
            "q6_k mmvq padded err {} > {} at {}",
            worst.0,
            scale * 1e-2,
            worst.1
        );

        // ── raw 210B stride via a direct decode call (gate bypassed) ──
        let mut w6tr = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6tr.name = format!("mw6kr{i}");
        cb.state.register_weight(&format!("mw6kr{i}"), &w6b);
        let obr = cb.alloc_buffer(od * nt);
        cb.state.q6_k_decode_mmvq(
            cb.state.get_weight_ptr(&format!("mw6kr{i}")).unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(obr).unwrap(),
            od,
            id_,
            nt,
            false,
        );
        let gotr = cb.copy_to_host(obr).unwrap();
        for i in 0..od * nt {
            assert!(
                (gotr[i] - want[i]).abs() <= scale * 1e-2,
                "q6_k mmvq raw stride mismatch at {i}"
            );
        }
    }
    // ── real weights (Q6_K tensors of the decode path) ──
    // 7B blk.0.ffn_down (q4_k_m, od=3584 x id=18944 — the shape that
    // passes the od*id >= 24M dispatch gate, so this also covers the
    // graph-dispatch wiring), 7B blk.0.attn_v and 0.5B blk.0.ffn_down
    // (both BELOW the gate — direct calls, since runtime keeps those
    // on the f32 kernel). Dumped via the ignored helpers; skipped when
    // a file is absent so the suite stays hermetic. The padded
    // registration repacks the raw 210B bytes, matching the loader.
    let real_shapes: [(&str, usize, usize, bool); 3] = [
        ("real_blk_0_ffn_down_weight.bin", 18944, 3584, true),
        ("real_blk_0_attn_v_weight.bin", 3584, 512, false),
        ("real05_blk_0_ffn_down_weight.bin", 4864, 896, false),
    ];
    for (file, id2, od2, via_graph) in real_shapes {
        let Ok(wb) = std::fs::read(format!("/tmp/minfer_phase7/{file}")) else {
            println!("real q6_k [{file}]: absent — skipped");
            break;
        };
        assert_eq!(wb.len(), od2 * (id2 / 256) * 210);
        // fresh activations at the real width
        let mut rng2 = 777u64;
        let xs2: Vec<f32> = (0..id2)
            .map(|_| {
                rng2 = rng2
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng2 >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng2 >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                (u as f32) * mag
            })
            .collect();
        let mut x82 = vec![0u8; (id2 / 32) * 40];
        for b in 0..id2 / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs2[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x82[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs2[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x82[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq82 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x82[(i / 32) * 40], x82[(i / 32) * 40 + 1]]).to_f32()
                * (x82[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };
        let mut w2t = Tensor::from_data(
            TensorType::Q6_K,
            &[id2 as i64, od2 as i64, 1, 1],
            wb.clone(),
        );
        w2t.name = "realq6k".to_string();
        cb.state
            .register_weight_q6k_padded("realq6k", &wb, od2, id2);
        let mut b2 = GraphBuilder::new();
        let x2 = b2.input("x2", [id2, 1, 1, 1], DType::F32);
        let m2 = b2.matmul(x2, &w2t, None);
        b2.output(m2);
        let g2 = b2.build();
        let xb2 = cb.alloc_buffer(id2);
        cb.write_host(xb2, &xs2).unwrap();
        let ob2 = cb.alloc_buffer(od2);
        if via_graph {
            // shape above the od*id gate — dispatch selects the MMVQ
            cb.exec_ids(&g2.nodes[m2], &[xb2], ob2, None).unwrap();
        } else {
            // below the gate — runtime dispatch would keep the f32
            // kernel; call the decode path directly for kernel parity
            cb.state.q6_k_decode_mmvq(
                cb.state.get_weight_ptr("realq6k").unwrap(),
                cb.ptr_of(xb2).unwrap(),
                cb.ptr_of(ob2).unwrap(),
                od2,
                id2,
                1,
                true,
            );
        }
        let got2 = cb.copy_to_host(ob2).unwrap();

        let mut want2 = vec![0f32; od2];
        for r in 0..od2 {
            let mut acc = 0f32;
            for ib in 0..id2 / 256 {
                let blk = &wb[(r * (id2 / 256) + ib) * 210..];
                let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
                for e in 0..256 {
                    let s = e / 16;
                    let l = e % 16;
                    let chunk = s / 8;
                    let g = (s / 2) % 4;
                    let is = s % 2;
                    let qlb = blk[chunk * 64 + (g % 2) * 32 + is * 16 + l];
                    let nib = if g < 2 { qlb & 0xF } else { qlb >> 4 };
                    let qhb = blk[128 + chunk * 32 + is * 16 + l];
                    let hi = (qhb >> (2 * g)) & 3;
                    acc += d
                        * (blk[192 + s] as i8 as f32)
                        * ((nib | (hi << 4)) as f32 - 32.0)
                        * dq82(ib * 256 + e);
                }
            }
            want2[r] = acc;
        }
        let scale2 = want2.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        let mut worst = (0f32, 0usize);
        for r in 0..od2 {
            let e = (got2[r] - want2[r]).abs();
            if e > worst.0 {
                worst = (e, r);
            }
        }
        println!(
            "real q6_k [{file}]: max err {:.4} at row {} (got {:.4} want {:.4}, scale {scale2:.3})",
            worst.0, worst.1, got2[worst.1], want2[worst.1]
        );
        assert!(
            worst.0 <= scale2 * 1e-2,
            "real q6_k mmvq err {} > {}",
            worst.0,
            scale2 * 1e-2
        );
    }
}

/// 8e follow-up: decode (nt == 1) Q5_K dispatches to the MMVQ structure
/// kernel (q4_K shape with the q5 high-bit plane folded in). Scale bytes
/// cover the full 0..255 range so the get_scale_min_k4 high-bit splicing
/// (bits 6..7 of bytes 0..3 feeding the upper scales — the 8e lesson that
/// %63 data never reaches those paths) is exercised. No local model
/// carries Q5_K tensors (the "q5_k_m"-branded 0.5B GGUF actually stores
/// Q5_1/Q8_0/Q6_K), so parity rests on this synthetic; real-weight
/// parity lands when a model with Q5_K tensors is used.
#[test]
fn cuda_q5k_decode_mmvq_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    // serialize against the other mmvq parity tests (shared scratch)
    let _guard = crate::cuda::CudaState::model_load_guard();
    // two shapes: 2176 (partial tail super-block → v1 kernels) and 2560
    // (full super-blocks → the R2 v2 weight-streaming kernels)
    for (i, (od, id_, nt)) in [(128usize, 2176usize, 1usize), (128usize, 2560usize, 1usize)]
        .into_iter()
        .enumerate()
    {
        let mut rng_state = 54321u64;
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng_state >> 60) & 7 == 0 {
                    1e-5
                } else {
                    3.0
                };
                (u as f32) * mag
            })
            .collect();

        fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
            if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                (
                    (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                    (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                )
            }
        }

        let nbe = (id_ + 255) / 256;
        let mut w5b = Vec::new();
        let mut w5dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                w5b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                w5b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                let mut scb = [0u8; 12];
                for j in 0..12 {
                    // full byte range — exercises the scale packing hi bits
                    scb[j] = ((r * 131 + j * 29 + ib * 7) % 256) as u8;
                }
                w5b.extend_from_slice(&scb);
                let mut qh = [0u8; 32];
                let mut qs = [0u8; 128];
                let n_elem = 256usize.min(id_ - ib * 256);
                for e in 0..n_elem {
                    let s = e / 32;
                    let l = e % 32;
                    let u = ((r * 13 + e * 7 + ib * 5) % 32) as u8;
                    let (s8, m8) = k4_scale(&scb, s);
                    w5dq[r * id_ + ib * 256 + e] = u as f32 * d * s8 as f32 - dmin * m8 as f32;
                    if s % 2 == 0 {
                        qs[(s >> 1) * 32 + l] |= u & 0xF;
                    } else {
                        qs[(s >> 1) * 32 + l] |= (u & 0xF) << 4;
                    }
                    qh[l] |= ((u >> 4) & 1) << s;
                }
                w5b.extend_from_slice(&qh);
                w5b.extend_from_slice(&qs);
            }
        }
        assert_eq!(w5b.len(), od * nbe * 176);

        let mut x8 = vec![0u8; (id_ / 32) * 40];
        for b in 0..id_ / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x8[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq8 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
                * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };

        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w5dq[r * id_ + i] * dq8(t * id_ + i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));

        let mut w5t = Tensor::from_data(
            TensorType::Q5_K,
            &[id_ as i64, od as i64, 1, 1],
            w5b.clone(),
        );
        w5t.name = format!("mw5k{i}");
        cb.state.register_weight(&format!("mw5k{i}"), &w5b);
        // direct decode call: the synthetic shape (od=128) is far below the
        // measured od*id >= 24M dispatch gate (small shapes stay on the f32
        // kernel at runtime); no local model carries Q5_K tensors, so the
        // gate wiring for q5_K is covered by this kernel parity + the shared
        // dispatch code path with q6_K.
        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        cb.state.q5_k_decode_mmvq(
            cb.state.get_weight_ptr(&format!("mw5k{i}")).unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(ob).unwrap(),
            od,
            id_,
            nt,
        );
        let got = cb.copy_to_host(ob).unwrap();
        for i in 0..od * nt {
            assert!(
                (got[i] - want[i]).abs() <= scale * 1e-2,
                "q5_k mmvq mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}

/// Viz/trace capture staging: async D2H queued behind the producing
/// kernel must survive a later overwrite of the same pool buffer
/// (intra-split pool reuse), drain in enqueue order, refuse oversized
/// buffers, and leave nothing queued after a drain.
#[test]
fn cuda_capture_staging_order_and_fallback() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();

    // 1. stream-order safety: enq BEFORE the buffer is overwritten
    let a = cb.alloc_buffer(8);
    let v1: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
    cb.write_host(a, &v1).unwrap();
    assert!(
        cb.capture_enq(a),
        "enq of a fresh small buffer must succeed"
    );
    // overwrite the SAME buffer on the stream after the enqueued D2H —
    // the staged value must still be v1
    let v2: Vec<f32> = (0..8).map(|i| -(i as f32) - 1.0).collect();
    cb.write_host(a, &v2).unwrap();
    let drained = cb.capture_drain();
    assert_eq!(drained.len(), 1);
    assert_close("staged pre-overwrite value", &drained[0], &v1, 1e-6);
    // the buffer itself holds the overwrite
    assert_close(
        "buffer post-overwrite value",
        &cb.copy_to_host(a).unwrap(),
        &v2,
        1e-6,
    );
    // drained clean: nothing queued, second drain is empty
    assert!(cb.capture_drain().is_empty());

    // 2. multiple buffers drain in enqueue order
    let b0 = cb.alloc_buffer(4);
    let b1 = cb.alloc_buffer(6);
    let w0: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let w1: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    cb.write_host(b0, &w0).unwrap();
    cb.write_host(b1, &w1).unwrap();
    assert!(cb.capture_enq(b0));
    assert!(cb.capture_enq(b1));
    let drained = cb.capture_drain();
    assert_eq!(drained.len(), 2);
    assert_close("order[0]", &drained[0], &w0, 1e-6);
    assert_close("order[1]", &drained[1], &w1, 1e-6);

    // 3. oversized buffer: refused (sync fallback in the scheduler), and
    // the refusal leaves the staging usable
    let big = cb.alloc_buffer(34 << 20); // 136 MB > 128 MB staging ceiling
    let bw: Vec<f32> = (0..34 << 20).map(|i| (i % 977) as f32 * 0.5).collect();
    cb.write_host(big, &bw).unwrap();
    assert!(!cb.capture_enq(big), "oversized buffer must be refused");
    assert_close(
        "fallback readback",
        &cb.copy_to_host(big).unwrap(),
        &bw,
        0.0,
    );
    assert!(cb.capture_enq(b0), "staging usable after a refusal");
    assert_eq!(cb.capture_drain().len(), 1);

    // 4. unknown buffer id: refused
    assert!(!cb.capture_enq(9_999_999));
}

/// Step 82: multi-token matmul dispatch — for every quant type, one
/// nt = 3 batched forward must be BITWISE-equal to three nt = 1
/// forwards over the same weight bytes and the same per-token
/// activations. The Step 82 kernels (multi-token MMVQ for the
/// K-quants, in-block token loops for the legacy f32 kernels and the
/// 8c q8-GEMM) preserve the per-(row, token) op order by
/// construction; this test pins it. Shapes are chosen so the nt = 1
/// and nt = 3 paths share the kernel family:
///   - q4_K id 3584 (3584 % 256 == 0 → v2 family) and id 3904
///     (id % 256 != 0 → v1 family), both above the nt == 1 id >= 2048
///     gate,
///   - q5_K od·id >= 24M so nt == 1 rides MMVQ too (v2: id 3072,
///     v1: id 3104),
///   - q6_K od·id >= 4M (padded 224B registration → v2 family; raw
///     210B → v1 family),
///   - the legacy f32 kernels (q8_0 / q4_0 with id > 8192 so the 8c
///     q8-GEMM gate is out / q4_1 / q5_0 / q5_1 / f32) run the same
///     token-looped kernel at nt == 1 and nt == 3.
/// The 8c q4_0 × q8-GEMM arm (nt > 1, id <= 8192) has no nt == 1
/// sibling, so it is checked against an independent host dequant
/// E1b: the CUDA windowed instantiations must give each sequence its own
/// window, matching `cpu_backend`'s `two_sequences_do_not_cross_attend`.
/// Device-gated — CI has no GPU, so it compiles there and runs where one
/// exists (on dgxspark it passes on GB10/sm_121; the original `hd = 2`
/// fixture could not have, see the E1b record).
#[test]
fn cuda_two_sequences_do_not_cross_attend() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_f16_for_test(false); // f32 KV keeps the store and the attention in one dtype

    // One head, hd = 4, two sequences: sequence 0 owns row 0, sequence 1
    // owns row 2. The values make a leak change the answer — query 1 scores
    // 1.0 against sequence 0's key, so a window starting at 0 would blend
    // V(0) into the result instead of returning V(2).
    //
    // hd must be a multiple of 4: the CUDA attention kernels reject anything
    // else (`attention head dim ... outside the kernel's supported range`),
    // so the original hd = 2 fixture could never execute on a device — the
    // test compiled and skipped everywhere until dgxspark got a working GPU.
    let (nh, nk, hd, nt, n_ctx) = (1usize, 1usize, 4usize, 2usize, 4usize);
    let nkt = nk * hd;
    let mut gb = GraphBuilder::new();
    gb.set_explicit_span(true);
    let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
    let q = gb.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
    let st = gb.kvcache_store(0, k, v, n_ctx);
    let kv = gb.kvcache_load(0, nkt, n_ctx, nk);
    let at = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: hd,
            nkt,
            scale: 1.0,
        },
    );
    gb.output(at);
    let g = gb.build();
    assert!(
        g.nodes.iter().any(|n| matches!(
            n.op,
            crate::graph::ops::Op::Attn {
                explicit_span: true,
                ..
            }
        )),
        "the attention node must declare explicit_span"
    );

    let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    let kreg = cb.alloc_buffer(n_ctx * nkt);
    let vreg = cb.alloc_buffer(n_ctx * nkt);
    let kb = cb.alloc_buffer(nkt * nt);
    let vb = cb.alloc_buffer(nkt * nt);
    let qb = cb.alloc_buffer(nh * hd * nt);
    let pb = cb.alloc_buffer(nt);
    let sb = cb.alloc_buffer(2 * nt);
    let ob = cb.alloc_buffer(nh * hd * nt);
    // token 0 = [1,0,0,0] (sequence 0), token 1 = [1,0,0,0] (sequence 1)
    cb.write_host(qb, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        .unwrap();
    // row 0 = k [1,0,0,0] / v [1,0,0,0]; row 2 = k [0,1,0,0] / v [0,1,0,0]
    cb.write_host(kb, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap();
    cb.write_host(vb, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap();
    cb.write_host(pb, &i32bits(&[0, 2])).unwrap();
    cb.write_host(sb, &i32bits(&[0, 2, 1, 3])).unwrap(); // lo block, hi block

    cb.exec_ids(&g.nodes[st], &[kb, vb, pb], kreg, Some((kreg, vreg)))
        .unwrap();
    cb.exec_ids(&g.nodes[at], &[qb, kreg, pb, sb], ob, Some((kreg, vreg)))
        .unwrap();
    // `copy_to_host`, not the trait's `read_host`: CUDA cannot return a
    // borrowed slice of device memory, so its `read_host` is `None` by
    // design (alloc.rs's `copy_to_cpu` CUDA arm does the same copy).
    let got = cb.copy_to_host(ob).unwrap();
    assert!(
        (got[0] - 1.0).abs() < 1e-4
            && got[1].abs() < 1e-4
            && got[2].abs() < 1e-4
            && got[3].abs() < 1e-4,
        "token 0 must attend to its own row: {got:?}"
    );
    assert!(
        got[4].abs() < 1e-4
            && (got[5] - 1.0).abs() < 1e-4
            && got[6].abs() < 1e-4
            && got[7].abs() < 1e-4,
        "token 1 saw the other sequence: {got:?}"
    );
}

/// E1b's recorded residual gap, closed: the **causal** and the **windowed**
/// instantiation must compute the same numbers over the same rows.
///
/// The equivalence rests on row arithmetic being the only difference between
/// the two kernels (the SASS comparison showed identical instruction counts
/// and opcode histograms). This is the direct check: one query per token, a
/// window that starts at cell 0 (so `positions[t] + 1` and the explicit span
/// describe the *same* rows), the same K/V, executed through both
/// instantiations — the outputs must be bitwise equal.
///
/// Without this, "the windowed path is correct" rested on the SASS identity
/// plus a windowed-only test with `lo = 0`; the E1b record named that as the
/// gap to close on the first device session.
#[test]
fn cuda_causal_and_windowed_agree_on_the_same_rows() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_f16_for_test(false); // f32 KV keeps the store and the attention in one dtype

    // Two tokens at cells 0 and 1: token 0's window is [0, 1), token 1's is
    // [0, 2) — exactly what `positions[t] + 1` gives, so both instantiations
    // must agree. hd = 4 (CUDA requires a nonzero multiple of 4).
    //
    // NOTE (2026-09-19, corrected): this fixture is **degenerate** — one-hot
    // queries and values make the output insensitive to the scores, so it is
    // weak evidence on its own and was never the windowed path's gate. The
    // randomized `cuda_windowed_attention_matches_causal_for_long_windows` is
    // that gate, and it is green (see its note: it first ran red because of a
    // harness bug — one advancing LCG shared by both calls — not the kernel).
    let (nh, nk, hd, nt, n_ctx) = (1usize, 1usize, 4usize, 2usize, 4usize);
    let nkt = nk * hd;
    let meta = crate::graph::ops::AttnMeta {
        layer: 0,
        n_head: nh,
        n_head_kv: nk,
        hd,
        hd_kv: hd,
        nkt,
        scale: 1.0,
    };
    let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    let q = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // token 0 = e0, token 1 = e1
    let k = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let positions = [0u32, 1u32];

    // Build one graph per instantiation; both store into and read the same
    // region, so they see identical K/V.
    let build = |explicit: bool| -> crate::graph::ComputeGraph {
        let mut gb = GraphBuilder::new();
        gb.set_explicit_span(explicit);
        let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let qq = gb.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let kk = gb.input("k", [nkt, nt, 1, 1], DType::F32);
        let vv = gb.input("v", [nkt, nt, 1, 1], DType::F32);
        let _st = gb.kvcache_store(0, kk, vv, n_ctx);
        let kv = gb.kvcache_load(0, nkt, n_ctx, nk);
        let at = gb.attn(qq, kv, pos, crate::graph::ops::AttnMode::Gqa, meta.clone());
        gb.output(at);
        gb.build()
    };
    let causal = build(false);
    let windowed = build(true);
    assert!(
        !causal.nodes.iter().any(|n| matches!(
            n.op,
            crate::graph::ops::Op::Attn {
                explicit_span: true,
                ..
            }
        )),
        "the causal graph must not declare an explicit span"
    );
    assert!(
        windowed.nodes.iter().any(|n| matches!(
            n.op,
            crate::graph::ops::Op::Attn {
                explicit_span: true,
                ..
            }
        )),
        "the windowed graph must declare an explicit span"
    );

    let mut run = |g: &crate::graph::ComputeGraph| -> Vec<f32> {
        let kreg = cb.alloc_buffer(n_ctx * nkt);
        let vreg = cb.alloc_buffer(n_ctx * nkt);
        let kb = cb.alloc_buffer(nkt * nt);
        let vb = cb.alloc_buffer(nkt * nt);
        let qb = cb.alloc_buffer(nh * hd * nt);
        let pb = cb.alloc_buffer(nt);
        let sb = cb.alloc_buffer(2 * nt);
        let ob = cb.alloc_buffer(nh * hd * nt);
        cb.write_host(qb, &q).unwrap();
        cb.write_host(kb, &k).unwrap();
        cb.write_host(vb, &v).unwrap();
        cb.write_host(pb, &i32bits(&positions)).unwrap();
        // token 0: [0, 1); token 1: [0, 2) — the same rows `positions` names.
        cb.write_host(sb, &i32bits(&[0, 0, 1, 2])).unwrap();
        let sti = g
            .nodes
            .iter()
            .position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheStore { .. }))
            .expect("store node");
        let ati = g
            .nodes
            .iter()
            .position(|n| matches!(n.op, crate::graph::ops::Op::Attn { .. }))
            .expect("attn node");
        cb.exec_ids(&g.nodes[sti], &[kb, vb, pb], kreg, Some((kreg, vreg)))
            .unwrap();
        cb.exec_ids(&g.nodes[ati], &[qb, kreg, pb, sb], ob, Some((kreg, vreg)))
            .unwrap();
        cb.copy_to_host(ob).unwrap()
    };
    let a = run(&causal);
    let b = run(&windowed);
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        worst, 0.0,
        "the causal and windowed instantiations disagree over the same rows: \
         causal {a:?} vs windowed {b:?}"
    );
}

/// The 2026-09-19 blocker, isolated: the **windowed** attention
/// instantiation must agree with the causal one over the *same relative
/// rows*, and it does not once the window is longer than a handful of rows.
///
/// Why this shape: the server puts every slot but the first at a non-zero
/// KV offset, so every slot but the first uses the windowed instantiation.
/// Because the kernels' row arithmetic is the only intended difference, the
/// assertion is **bitwise** equality against the causal run over rows `0..n`.
///
/// Note (2026-09-19): this test first ran RED, and the failure was the
/// **harness**, not the kernel — the two `run` calls shared one advancing LCG,
/// so "causal" and "windowed" were compared over *different* q/k/v (the
/// causal-vs-`V(row 0)` check still passed, because a single-key softmax is
/// that identity whatever the data). With the data now seeded per shape inside
/// `run`, every case is bitwise equal; plan §14's "narrowed to the windowed
/// instantiation" entry is retracted on that evidence.
#[test]
fn cuda_windowed_attention_matches_causal_for_long_windows() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_f16_for_test(false);

    // Real-ish shapes: the 7B decodes with hd 128 / 4 KV heads, which is a
    // larger `hd` and `nkv` than the 0.5B's 64 / 2 — the other reason this
    // was model-dependent.
    let n_ctx = 512usize;
    let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };

    // Deterministic pseudo-random data (an LCG, so a failure is reproducible).
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((seed >> 8) as f32 / 8_388_608.0) - 1.0
    };

    let mut run = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                   explicit: bool,
                   start: usize,
                   n: usize,
                   shape: (usize, usize, usize)|
     -> (Vec<f32>, Vec<f32>) {
        let (nh, nk, hd) = shape;
        let nkt = nk * hd;
        // The two calls this test compares (causal / windowed) MUST see the
        // same q/k/v: the outer LCG advances on every call, so using it here
        // compared different data and reported a divergence that was an
        // artefact of the harness. Seed a local LCG from the shape only —
        // never from `start`/`explicit`, which are what the calls differ in.
        let mut lseed = 0x9e37_79b9u32
            ^ (n as u32).wrapping_mul(0x85eb_ca6b)
            ^ (nh as u32).wrapping_mul(0xc2b2_ae35)
            ^ (hd as u32).wrapping_mul(0x27d4_eb2f);
        let mut next = || {
            lseed = lseed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((lseed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let meta = crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: hd,
            nkt,
            scale: 1.0,
        };
        let mut gb = GraphBuilder::new();
        gb.set_explicit_span(explicit);
        let pos = gb.input("positions", [n, 1, 1, 1], DType::I32);
        let qq = gb.input("q", [nh * hd, n, 1, 1], DType::F32);
        let kk = gb.input("k", [nkt, n, 1, 1], DType::F32);
        let vv = gb.input("v", [nkt, n, 1, 1], DType::F32);
        let _st = gb.kvcache_store(0, kk, vv, n_ctx);
        let kv = gb.kvcache_load(0, nkt, n_ctx, nk);
        let at = gb.attn(qq, kv, pos, crate::graph::ops::AttnMode::Gqa, meta);
        gb.output(at);
        let g = gb.build();

        let posv: Vec<usize> = (start..start + n).collect();
        let qv: Vec<f32> = (0..nh * hd * n).map(|_| next()).collect();
        let kvv: Vec<f32> = (0..nkt * n).map(|_| next()).collect();
        let vvv: Vec<f32> = (0..nkt * n).map(|_| next()).collect();
        let span: Vec<u32> = (0..n)
            .flat_map(|t| [(start as u32), (start + t + 1) as u32])
            .collect();
        // The span array is laid out as all `lo`s then all `hi`s.
        let mut span_u32 = vec![0u32; 2 * n];
        for t in 0..n {
            span_u32[t] = start as u32;
            span_u32[n + t] = (start + t + 1) as u32;
        }
        let _ = span;

        let kreg = cb.alloc_buffer(n_ctx * nkt);
        let vreg = cb.alloc_buffer(n_ctx * nkt);
        let kb = cb.alloc_buffer(nkt * n);
        let vb = cb.alloc_buffer(nkt * n);
        let qb = cb.alloc_buffer(nh * hd * n);
        let pb = cb.alloc_buffer(n);
        let sb = cb.alloc_buffer(2 * n);
        let ob = cb.alloc_buffer(nh * hd * n);
        cb.write_host(qb, &qv).unwrap();
        cb.write_host(kb, &kvv).unwrap();
        cb.write_host(vb, &vvv).unwrap();
        cb.write_host(
            pb,
            &i32bits(&posv.iter().map(|&p| p as u32).collect::<Vec<u32>>()),
        )
        .unwrap();
        cb.write_host(sb, &i32bits(&span_u32)).unwrap();
        // Harness self-check: the buffers the launch will read must hold
        // exactly what this test wrote. If they do, the inputs are not the
        // explanation for the divergence and the launch/kernel is.
        let pos_back = cb.copy_to_host(pb).unwrap();
        let want_pos: Vec<f32> = i32bits(&posv.iter().map(|&p| p as u32).collect::<Vec<u32>>());
        assert_eq!(pos_back, want_pos, "positions buffer read back wrong");
        let span_back = cb.copy_to_host(sb).unwrap();
        assert_eq!(span_back, i32bits(&span_u32), "span buffer read back wrong");
        let sti = g
            .nodes
            .iter()
            .position(|nd| matches!(nd.op, crate::graph::ops::Op::KvcacheStore { .. }))
            .unwrap();
        let ati = g
            .nodes
            .iter()
            .position(|nd| matches!(nd.op, crate::graph::ops::Op::Attn { .. }))
            .unwrap();
        cb.exec_ids(&g.nodes[sti], &[kb, vb, pb], kreg, Some((kreg, vreg)))
            .unwrap();
        cb.exec_ids(&g.nodes[ati], &[qb, kreg, pb, sb], ob, Some((kreg, vreg)))
            .unwrap();
        let out = cb.copy_to_host(ob).unwrap();
        // Token t's V block is contiguous (`[t*nkt, (t+1)*nkt)`), so this is
        // the exact expected output of query 0, whose window is a single row.
        (out, vvv[..hd].to_vec())
    };

    // (window length, non-zero start): the lengths sweep the kernel variants
    // the 7B selects; the starts are the server's kind of offset.
    let mut bad: Vec<String> = Vec::new();
    // Both KV dtypes: the production default is the engine's resolved format
    // (`kvformat::auto_device_format`: f16 when `n_layers * n_kv_embd >= 8192`) —
    // so the 7B runs f16 KV while the 0.5B runs f32, and a gate that forces f32
    // cannot see a windowed-f16 fault at all.
    for f16 in [false, true] {
        for (shape, n, start) in [
            ((1usize, 1usize, 4usize), 1usize, 0usize),
            ((1, 1, 4), 2, 0),
            ((1, 1, 4), 2, 64),
            ((1, 1, 128), 1, 0),
            ((1, 1, 128), 2, 0),
            ((2, 2, 64), 1, 0),
            ((4, 4, 128), 1, 0),
            ((4, 4, 128), 2, 0),
            ((4, 4, 128), 2, 64),
            ((4, 4, 128), 16, 64),
            ((4, 4, 128), 34, 64),
        ] {
            cb.set_kv_f16_for_test(f16);
            let (causal, v_row0) = run(&mut cb, false, 0, n, shape);
            let (windowed, _) = run(&mut cb, true, start, n, shape);
            let hd = shape.2;
            let ref_delta = |x: &[f32]| {
                x[..hd]
                    .iter()
                    .zip(&v_row0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            };
            // The f16 KV rounds V, so the exactness check is f32-only; the
            // causal-vs-windowed comparison below is unaffected (bitwise for
            // both dtypes, since both runs see the same stored K/V).
            assert!(
                f16 || ref_delta(&causal) == 0.0,
                "the CAUSAL instantiation does not return V(row 0) for a single-row window (n={n})"
            );
            let worst = causal
                .iter()
                .zip(&windowed)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if worst == 0.0 {
                eprintln!("[window] kv_f16={f16} {shape:?} n={n} start={start}: bitwise equal");
            } else {
                let first = causal
                    .iter()
                    .zip(&windowed)
                    .position(|(a, b)| a != b)
                    .map(|k| (k, causal[k], windowed[k]));
                eprintln!(
                    "[window] kv_f16={f16} {shape:?} n={n} start={start}: DIVERGES max|d|={worst} first={first:?}"
                );
                bad.push(format!(
                    "kv_f16={f16} {shape:?} n={n} start={start} max|d|={worst} first={first:?}"
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "the windowed instantiation diverges from the causal one over the same relative rows \
         in {} case(s):\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// C8b S4: a `kv_map` window must give each query exactly the rows the
/// equivalent `attn_span` gives it — the same cells, named through a run list
/// instead of one range. **Bitwise**, both KV dtypes: the two modes differ in
/// how they resolve a row, not in what they compute over it.
///
/// The shapes sweep the kernel families a map can land in: `nt == 1` (split-K
/// flash decoding — a sharing slot's decode step), `1 < nt <= 16` (the batched
/// split path) and `nt > 16` (prefill: FA when hd = 128, the legacy per-(token,
/// head) kernel otherwise). Both a prefill-shaped batch and a **decode-shaped**
/// one are run: a single token at the end of the window, whose window is several
/// runs. A first draft of this test only ever gave one token at position 0,
/// which touches the first run alone — the multi-run decode case is exactly what
/// the real-model gate caught it missing.
#[test]
fn cuda_map_window_matches_the_span_over_the_same_rows() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 512;
    const KMAX: usize = crate::graph::kvcache::KV_MAP_MAX_SPANS;

    // Reference rows [0, n) of both regions, seeded from (n, shape) alone so
    // every call compared here sees the same bytes.
    let reference = |n: usize, shape: (usize, usize, usize)| -> (Vec<f32>, Vec<f32>) {
        let (nh, nk, hd) = shape;
        let nkt = nk * hd;
        let mut seed = 0x51ed_2701u32
            ^ (n as u32).wrapping_mul(0x9e37_79b9)
            ^ (nh as u32).wrapping_mul(0x85eb_ca6b)
            ^ (nkt as u32).wrapping_mul(0xc2b2_ae35);
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let k: Vec<f32> = (0..n * nkt).map(|_| next()).collect();
        let v: Vec<f32> = (0..n * nkt).map(|_| next()).collect();
        (k, v)
    };

    // One attention call. `runs` are `(cell, len, base)`: `len` cells from
    // `cell` hold the reference rows starting at `base`, which is how a
    // non-contiguous window is compared against the span over the same bytes.
    let mut run = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                   map: bool,
                   pos: &[u32],
                   window: &[u32],
                   shape: (usize, usize, usize),
                   runs: &[(usize, usize, usize)],
                   reference: &(Vec<f32>, Vec<f32>),
                   layout: i32|
     -> Vec<f32> {
        let nt = pos.len();
        let (nh, nk, hd) = shape;
        let nkt = nk * hd;
        // C4 S2b: the packed cell's word width. Every region here is built
        // with the *layout's* own size — `N_CTX * row_elems` — which is also
        // the discipline issue #122 asks for (an under-sized buffer plus a
        // device read past it latches `cudaErrorIllegalAddress` and poisons
        // every later allocation in the process).
        let row_elems = if layout == crate::cuda::KV_LAYOUT_Q8_0 {
            crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt)
        } else {
            nkt
        };
        let meta = crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: hd,
            nkt,
            scale: 1.0,
        };
        let mut gb = GraphBuilder::new();
        gb.set_explicit_span(true);
        gb.set_kv_map(map);
        let p = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let qq = gb.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let _kk = gb.input("k", [nkt, nt, 1, 1], crate::graph::DType::F32);
        let _vv = gb.input("v", [nkt, nt, 1, 1], crate::graph::DType::F32);
        let kv = gb.kvcache_load(0, nkt, N_CTX, nk);
        let at = gb.attn(qq, kv, p, crate::graph::ops::AttnMode::Gqa, meta);
        gb.output(at);
        let g = gb.build();

        // The KV region as the runs describe it. With an f16 cache the kernel
        // reads the *low half* of each 4-byte slot, so a value is written as the
        // half's bit pattern there: clean f16 data rather than the garbage an
        // arbitrary f32 write leaves behind — which would make a wrong-row
        // comparison compare zero to zero. With Q8_0 the row is packed through
        // the same quantizer the store kernel uses (`pack_q8_0_cell`), so the
        // kernel reads exactly the bytes a real store would have written.
        let (rk, rv) = reference;
        let mut region_k = vec![0.0f32; N_CTX * row_elems];
        let mut region_v = vec![0.0f32; N_CTX * row_elems];
        let enc = |x: f32| -> f32 {
            if layout == crate::cuda::KV_LAYOUT_F16 {
                f32::from_bits(half::f16::from_f32(x).to_bits() as u32)
            } else {
                x
            }
        };
        for &(cell, len, base) in runs {
            for i in 0..len {
                let src = (base + i) * nkt;
                let dst = (cell + i) * row_elems;
                if layout == crate::cuda::KV_LAYOUT_Q8_0 {
                    crate::graph::kvformat::pack_q8_0_cell(
                        &mut region_k[dst..dst + row_elems],
                        nkt,
                        &rk[src..src + nkt],
                    );
                    crate::graph::kvformat::pack_q8_0_cell(
                        &mut region_v[dst..dst + row_elems],
                        nkt,
                        &rv[src..src + nkt],
                    );
                } else {
                    for e in 0..nkt {
                        region_k[dst + e] = enc(rk[src + e]);
                        region_v[dst + e] = enc(rv[src + e]);
                    }
                }
            }
        }
        // q is its own data (the comparison is over the same q in both modes).
        let mut seed = 0x1234_abcdu32
            ^ (nt as u32).wrapping_mul(0x27d4_eb2f)
            ^ (nh as u32).wrapping_mul(0x9e37_79b9);
        let qv: Vec<f32> = (0..nh * hd * nt)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((seed >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect();

        let kreg = cb.alloc_buffer(N_CTX * row_elems);
        let vreg = cb.alloc_buffer(N_CTX * row_elems);
        let qb = cb.alloc_buffer(nh * hd * nt);
        let pb = cb.alloc_buffer(nt);
        let wb = cb.alloc_buffer(window.len());
        let ob = cb.alloc_buffer(nh * hd * nt);
        let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
        cb.write_host(qb, &qv).unwrap();
        cb.write_host(kreg, &region_k).unwrap();
        cb.write_host(vreg, &region_v).unwrap();
        cb.write_host(pb, &i32bits(pos)).unwrap();
        cb.write_host(wb, &i32bits(window)).unwrap();
        let ati = g
            .nodes
            .iter()
            .position(|nd| matches!(nd.op, crate::graph::ops::Op::Attn { .. }))
            .unwrap();
        cb.exec_ids(&g.nodes[ati], &[qb, kreg, pb, wb], ob, Some((kreg, vreg)))
            .unwrap();
        cb.copy_to_host(ob).unwrap()
    };

    // A span window for a batch at `pos`: one `[lo, hi)` pair per query.
    let span_batch = |pos: &[u32]| -> Vec<u32> {
        let mut w = vec![0u32; 2 * pos.len()];
        for (i, &p) in pos.iter().enumerate() {
            w[i] = 64;
            w[pos.len() + i] = 64 + p + 1;
        }
        w
    };
    // ...and the runs `attn_map` would emit for it (in position order, each
    // query's own run clipped to its row).
    let map_batch = |runs: &[(usize, usize, usize)], pos: &[u32]| -> Vec<u32> {
        let mut w = Vec::with_capacity(pos.len() * KMAX * 2);
        for &p in pos {
            let mut left = p as usize + 1;
            let mut kk = 0usize;
            for &(cell, len, _) in runs {
                let take = len.min(left);
                if take == 0 {
                    continue;
                }
                assert!(kk < KMAX, "the fixture needs more than {KMAX} runs");
                w.push(cell as u32);
                w.push(take as u32);
                left -= take;
                kk += 1;
            }
            assert_eq!(left, 0, "the runs must cover query {p}'s window");
            while kk < KMAX {
                w.push(0);
                w.push(0);
                kk += 1;
            }
        }
        w
    };

    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    // C4 S2b: all three layouts. Q8_0 rows go through exactly the same
    // modes — the run list is resolved before the load, so the packed accessor
    // cannot change which rows a window names.
    for layout in [
        crate::cuda::KV_LAYOUT_F32,
        crate::cuda::KV_LAYOUT_F16,
        crate::cuda::KV_LAYOUT_Q8_0,
    ] {
        cb.set_kv_layout_for_test(layout);
        let layout_name = match layout {
            crate::cuda::KV_LAYOUT_Q8_0 => "q8_0",
            crate::cuda::KV_LAYOUT_F16 => "f16",
            _ => "f32",
        };
        for (shape, n) in [
            ((1usize, 1usize, 4usize), 1usize),
            ((2, 2, 64), 1),
            ((4, 4, 128), 1),
            ((2, 2, 64), 6),
            ((4, 4, 128), 8),  // 1 < nt <= 16: the batched split path (f32/f16)
            ((4, 4, 128), 34), // nt > 16: FA prefill (hd = 128) for f16, general otherwise
            ((2, 2, 64), 34),  // nt > 16 with f32 KV: the legacy kernel
        ] {
            // A Q8_0 cell is a whole number of 32-element blocks, so `nkt` must
            // be a multiple of 32 — the invariant `ensure_kv`'s `check_width`
            // enforces (`nkt` is `n_head_kv * hd` and every supported arch has
            // `hd % 32 == 0`). The `hd = 4` fixture is a kernel-shape probe that
            // a packed format cannot express at all; it is skipped rather than
            // silently run with a zero-word cell.
            if layout == crate::cuda::KV_LAYOUT_Q8_0 && (shape.1 * shape.2) % 32 != 0 {
                continue;
            }
            let reference = reference(n, shape);
            let mut variants: Vec<(&str, Vec<(usize, usize, usize)>)> = vec![
                ("one run", vec![(64, n, 0)]),
                (
                    "two runs",
                    vec![(64, 4.min(n), 0), (300, n - 4.min(n), 4.min(n))],
                ),
            ];
            if n >= 6 {
                // Three runs, so the walk is exercised past its second entry.
                let (l0, l1) = (n / 3, n / 3);
                variants.push((
                    "three runs",
                    vec![(64, l0, 0), (200, l1, l0), (300, n - l0 - l1, l0 + l1)],
                ));
            }
            for (label, runs) in variants {
                if runs.iter().map(|r| r.1).sum::<usize>() != n {
                    continue; // a window this short cannot be split that far
                }
                // (a) a prefill-shaped batch (one query per window row) and
                // (b) a decode-shaped one (a single token at the window's end,
                //     which is the multi-run case a sharing slot decodes).
                let prefill: Vec<u32> = (0..n as u32).collect();
                let decode = [n as u32 - 1];
                for (which, pos) in [("prefill", &prefill[..]), ("decode", &decode[..])] {
                    let span = run(
                        &mut cb,
                        false,
                        pos,
                        &span_batch(pos),
                        shape,
                        &[(64, n, 0)],
                        &reference,
                        layout,
                    );
                    let map = run(
                        &mut cb,
                        true,
                        pos,
                        &map_batch(&runs, pos),
                        shape,
                        &runs,
                        &reference,
                        layout,
                    );
                    checked += 1;
                    let worst = span
                        .iter()
                        .zip(map.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if worst != 0.0 {
                        bad.push(format!(
                            "kv={layout_name} {shape:?} n={n} {label} {which}: max|d|={worst}"
                        ));
                    }
                }
            }
        }
    }
    assert!(checked >= 30, "the fixture checked only {checked} cases");
    assert!(
        bad.is_empty(),
        "the map window disagrees with the span over the same bytes in {} case(s):\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
    // A window that attended to *no* row would compare equal trivially (both
    // sides zero), so the gate has to see real output too.
    cb.set_kv_f16_for_test(false);
    let reference8 = reference(8, (2, 2, 64));
    let out = run(
        &mut cb,
        false,
        &(0..8u32).collect::<Vec<u32>>(),
        &span_batch(&(0..8u32).collect::<Vec<u32>>()),
        (2, 2, 64),
        &[(64, 8, 0)],
        &reference8,
        crate::cuda::KV_LAYOUT_F32,
    );
    assert!(
        out.iter().any(|&x| x != 0.0 && x.is_finite()),
        "the fixture attends to no row — the comparison above would be vacuous"
    );
    // Every comparison above is **between modes over the same bytes**, so a
    // value-level fault in the layout accessor shifts both sides identically and
    // the bitwise equality still holds. Mutation-checked by hand: deleting the
    // block base from `kv4<KV_LAYOUT_Q8_0>` (reading block 0 for every group)
    // left this test green while `cuda_kv_q8_0_roundtrip_attn` failed — the F6
    // lesson, a gate passing for the wrong reason. So one single-row window is
    // compared against an **independently computed** reference: the dequantized
    // V row the fixture packed. A single-row window's softmax is 1, so the output
    // is exactly `d * q[i]` of the cell — the same expression `kv4<Q8_0>`
    // evaluates — and this arm now fails on a wrong block, scale or quant offset.
    {
        use crate::graph::kvformat::{pack_q8_0_cell, unpack_q8_0_cells, KvFormat};
        let shape = (2usize, 2usize, 64usize);
        let nkt = shape.1 * shape.2;
        let row_elems = KvFormat::Q8_0.row_elems(nkt);
        let reference1 = reference(1, shape);
        let mut cell_v = vec![0f32; row_elems];
        pack_q8_0_cell(&mut cell_v, nkt, &reference1.1);
        let mut want_v = vec![0f32; nkt];
        unpack_q8_0_cells(&cell_v, nkt, 0, 1, &mut want_v);
        cb.set_kv_layout_for_test(crate::cuda::KV_LAYOUT_Q8_0);
        let out = run(
            &mut cb,
            false,
            &[0u32],
            &span_batch(&[0u32]),
            shape,
            &[(64, 1, 0)],
            &reference1,
            crate::cuda::KV_LAYOUT_Q8_0,
        );
        let worst = out[..shape.2]
            .iter()
            .zip(&want_v[..shape.2])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            out[..shape.2].iter().any(|x| x.abs() > 1e-6),
            "the single-row Q8_0 fixture returned all zeros — vacuous"
        );
        assert_eq!(
            worst, 0.0,
            "the Q8_0 single-row window does not return the dequantized V cell row: \
             max |Δ| = {worst}"
        );
    }
    eprintln!("[map] {checked} cases bitwise-equal to the span over the same rows");
}

/// C8b S4: a cell move must stride by a **row's** size, whatever the KV dtype
/// stores. With an f16 cache a row is `nkt` halves = `nkt / 2` f32 elements,
/// while `copy_cells`'s `elems_per_cell` is a count of f32 — the unit the move
/// kernel walks by. Passing `nkt` there moved every row twice as far as it
/// should, so a copy-on-write (or a compaction) on an f16 device wrote the
/// wrong cells; the real-model hd = 128 gate found it, and this pins it.
#[test]
fn cuda_f16_kv_cell_move_strides_by_row_bytes() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 64;
    // A row of `nkt` halves, stored in the first `nkt / 2` f32 slots.
    let nkt = 8usize;
    cb.set_kv_f16_for_test(true);
    let (src_row, dst_row, rows) = (2usize, 10usize, 4usize);
    let slots = N_CTX * (nkt / 2);
    let region = cb.alloc_buffer(slots);
    // Every f32 slot carries its own index in its low 16 bits, so the halves a
    // kernel would read are distinct and the *slot* layout is verifiable.
    let pattern: Vec<f32> = (0..slots)
        .map(|i| f32::from_bits(((i as u32) * 7 + 1) & 0xFFFF))
        .collect();
    cb.write_host(region, &pattern).unwrap();
    let before = pattern.clone();
    let r = BufRef::own(crate::graph::Backend::CUDA, region, slots);
    cb.copy_cells(r, r, dst_row, src_row, rows, nkt).unwrap();
    let after = cb.copy_to_host(region).unwrap();
    assert_eq!(after.len(), slots);
    for r in 0..rows {
        // Slot view of a row: `row * nkt / 2` f32.
        let sd = (src_row + r) * (nkt / 2);
        let dd = (dst_row + r) * (nkt / 2);
        assert_eq!(
            &after[dd..dd + nkt / 2],
            &before[sd..sd + nkt / 2],
            "row {r} did not land at row {} (f16 KV strides in f32 elements)",
            dst_row + r
        );
    }
    // Nothing outside the destination rows may move.
    for (i, (x, y)) in before.iter().zip(&after).enumerate() {
        let moved = (dst_row * nkt / 2..(dst_row + rows) * nkt / 2).contains(&i);
        if !moved {
            assert_eq!(x, y, "cell move touched f32 slot {i} outside its rows");
        }
    }
}

/// C4 S2b: the packed twin of
/// [`Self::cuda_f16_kv_cell_move_strides_by_row_bytes`]. A Q8_0 cell is a whole
/// number of f32 words (`KvFormat::Q8_0.row_elems(nkt)` = ceil(nkt/32*34 / 4)),
/// and that is exactly the unit `copy_cells`'s `elems_per_cell` already carries
/// — the caller (`GraphAllocator::kv_set_cap_with_defrag`, the compaction and
/// the copy-on-write) passes `region.elems / n_ctx`. So the device backend must
/// pass it through **unchanged**: applying the f16 halving here would move
/// `row_elems / 2` words per row, land every moved cell short of its slot and
/// silently corrupt the arena on the first compaction.
///
/// `nkt = 64` is chosen so the two candidate strides cannot be confused:
/// `row_elems` is 17 words, while the pre-C4 `nkt` (and the f16 half of it) are
/// 64 and 32.
#[test]
fn cuda_q8_0_kv_cell_move_strides_by_row_bytes() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 64;
    let nkt = 64usize;
    let row_elems = crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt);
    assert_eq!(row_elems, 17, "the fixture assumes 2 Q8_0 blocks of 34 B");
    cb.set_kv_q8_for_test();
    let (src_row, dst_row, rows) = (2usize, 10usize, 4usize);
    let slots = N_CTX * row_elems;
    let region = cb.alloc_buffer(slots);
    // Distinct word per slot, so a move of the wrong length is visible as a
    // mismatched suffix *and* as a stray write outside the destination rows.
    let pattern: Vec<f32> = (0..slots)
        .map(|i| f32::from_bits(((i as u32) * 2654435761) | 1))
        .collect();
    cb.write_host(region, &pattern).unwrap();
    let before = pattern.clone();
    let r = BufRef::own(crate::graph::Backend::CUDA, region, slots);
    // The host passes the packed cell's word count, exactly as alloc.rs does.
    cb.copy_cells(r, r, dst_row, src_row, rows, row_elems)
        .unwrap();
    let after = cb.copy_to_host(region).unwrap();
    assert_eq!(after.len(), slots);
    for r in 0..rows {
        let sd = (src_row + r) * row_elems;
        let dd = (dst_row + r) * row_elems;
        // Compared as **bits**: the pattern is arbitrary words and some are NaN,
        // for which `==` is never true (the same reason C4's V-verbatim gate
        // compares bits).
        let same = after[dd..dd + row_elems]
            .iter()
            .zip(&before[sd..sd + row_elems])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same,
            "packed row {r} did not land at row {} (a Q8_0 cell is {row_elems} f32 words)",
            dst_row + r
        );
    }
    for (i, (x, y)) in before.iter().zip(&after).enumerate() {
        let moved = (dst_row * row_elems..(dst_row + rows) * row_elems).contains(&i);
        if !moved {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "packed cell move touched f32 slot {i} outside its rows"
            );
        }
    }
}

/// C4 S2b: the Q8_0 store, two ways. (a) The bytes the device writes for a row
/// are **exactly** `kvformat::pack_q8_0_cell`'s — the CPU store's own output —
/// so a CPU/device Q8_0 comparison is a layout check, not a tolerance question.
/// (b) Those bytes survive the pool round trip (`copy_to_host` returns the same
/// words), which is what a physical shift's dequantize → re-rope → requantize
/// reads and writes back.
///
/// Mutation-checked by hand: pointing the kernel at `elem % 32` instead of
/// `(elem >> 5, elem & 31)` (i.e. dropping the block base) fails (a) on the
/// second block; storing the scale at byte 2 and the quants at 0 fails (a) on
/// every element.
#[test]
fn cuda_q8_0_store_matches_the_cpu_quantizer() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 64;
    const NT: usize = 5;
    // hd = 64 (two blocks per row) and hd = 32 (one) so a block-base mistake is
    // exercised, not just an offset one.
    for nkt in [64usize, 32] {
        let row_elems = crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt);
        let row_bytes = crate::graph::kvformat::KvFormat::Q8_0.row_bytes(nkt);
        cb.set_kv_q8_for_test();
        // Rows with a wide dynamic range inside each block, so `amax/127` is not
        // a degenerate 0 or 1 and the quants exercise the round/clamp.
        let rows: Vec<f32> = (0..NT * nkt)
            .map(|i| {
                let b = (i % nkt) / 32;
                let sign = if (i / 7) % 2 == 0 { 1.0 } else { -1.0 };
                sign * (((i * 37 % 101) as f32) / 101.0) * (b as f32 + 1.0) * 2.5
            })
            .collect();
        let pos: Vec<usize> = vec![0, 3, 7, 11, 15];
        let region = cb.alloc_buffer(N_CTX * row_elems);
        let src = cb.alloc_buffer(NT * nkt);
        let posb = cb.alloc_buffer(NT);
        cb.write_host(src, &rows).unwrap();
        cb.write_host(
            posb,
            &pos.iter()
                .map(|&p| f32::from_bits(p as u32))
                .collect::<Vec<f32>>(),
        )
        .unwrap();
        cb.write_host(region, &vec![0f32; N_CTX * row_elems])
            .unwrap();
        cb.state.store_kv_q8_0(
            cb.ptr_of(src).unwrap(),
            cb.ptr_of(region).unwrap(),
            nkt,
            NT,
            row_bytes,
            cb.ptr_of(posb).unwrap(),
        );

        let after = cb.copy_to_host(region).unwrap();
        // (a) byte-for-byte against the CPU quantizer, per real row. Compared as
        // **bits**: a packed word is an f16 scale and int8 quants, so as f32 it
        // is frequently NaN and `==` on it is never true.
        for (t, &p) in pos.iter().enumerate() {
            let row_f32 = &rows[t * nkt..(t + 1) * nkt];
            let expected: Vec<f32> = {
                let mut w = vec![0f32; row_elems];
                crate::graph::kvformat::pack_q8_0_cell(&mut w, nkt, row_f32);
                w
            };
            let same = after[p * row_elems..(p + 1) * row_elems]
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                same,
                "nkt={nkt} row {t} (cell {p}): the device store is not the CPU quantizer's bytes"
            );
            assert_ne!(
                expected.iter().fold(0u32, |m, x| m | x.to_bits()),
                0,
                "the fixture wrote all-zero packed bytes; the comparison would be vacuous"
            );
        }
        // (b) unwritten cells stay zero (the store writes only its rows).
        for cell in 0..N_CTX {
            if pos.contains(&cell) {
                continue;
            }
            assert!(
                after[cell * row_elems..(cell + 1) * row_elems]
                    .iter()
                    .all(|x| x.to_bits() == 0),
                "the store touched cell {cell}, which no position named"
            );
        }
    }
}

/// C4 S2b: the Q8_0 store → attention round trip, the packed twin of
/// [`Self::cuda_kv_f16_roundtrip_attn`]. The device writes real packed cells
/// through `Op::KvcacheStore`, then attention reads them back through the
/// layout-tagged split path — and the output must match the **f32** kernel run
/// over the same cells dequantized on the host: `kv4<Q8_0>` reconstructs
/// exactly `d * q[i]`, which is what `kvformat::unpack_q8_0_cells` puts in the
/// reference region. A wrong block index, scale offset or quant offset moves a
/// value by whole quant steps (the 0.117-magnitude outputs here), so the 1e-6
/// bound is many orders tighter than any layout fault while leaving room for
/// the one real difference between the two kernels: nvcc may contract the
/// packed accessor's `d * q` into the following FMA chain, and the f32 kernel
/// reads that product from memory, so the two can differ by one ulp. Measured:
/// exactly one ULP on one element (2.38e-7 of 0.117). The **byte-exact** claim
/// for the store lives in [`Self::cuda_q8_0_store_matches_the_cpu_quantizer`],
/// where it belongs.
#[test]
fn cuda_kv_q8_0_roundtrip_attn() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    let (nh, nk_h, hd) = (4usize, 2usize, 64usize);
    let nkt = nk_h * hd;
    let row_elems = crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt);
    let (nt, n_ctx) = (3usize, 32usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = vec![1, 4, 9];

    let mut b = GraphBuilder::new();
    b.set_kv_format(crate::graph::kvformat::KvFormat::Q8_0);
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let qr = b.rope(
        q,
        pp,
        RopeStyle::NonInterleaved,
        RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: nh,
            hd,
        },
    );
    let at = b.attn(
        qr,
        load,
        pp,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk_h,
            hd,
            hd_kv: hd,
            nkt,
            scale,
        },
    );
    b.output(at);
    let g = b.build();

    let (xb_q, xb_k, xb_v) = (
        cb.alloc_buffer(nh * hd * nt),
        cb.alloc_buffer(nkt * nt),
        cb.alloc_buffer(nkt * nt),
    );
    let xb_p = cb.alloc_buffer(nt);
    let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
    let (kreg, vreg) = (
        cb.alloc_buffer(n_ctx * row_elems),
        cb.alloc_buffer(n_ctx * row_elems),
    );
    // The f32 reference regions the dequantized cells are written into, read by
    // the same kernel instantiated on KV_LAYOUT_F32.
    let (kf32, vf32) = (cb.alloc_buffer(n_ctx * nkt), cb.alloc_buffer(n_ctx * nkt));

    let qs: Vec<f32> = (0..nh * hd * nt)
        .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
        .collect();
    let ks: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let vs: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    cb.write_host(kreg, &vec![0f32; n_ctx * row_elems]).unwrap();
    cb.write_host(vreg, &vec![0f32; n_ctx * row_elems]).unwrap();

    // Q8_0 arm: store through the kernel, attention over the packed cells.
    cb.set_kv_q8_for_test();
    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
        .unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kreg, xb_p],
        ob_at,
        Some((kreg, vreg)),
    )
    .unwrap();
    let got_q8 = cb.copy_to_host(ob_at).unwrap();

    // Reference arm: dequantize the *same packed bytes* the device wrote on the
    // host, put them in f32-shaped regions, and run the f32 kernel. The two arms
    // share every byte of K/V, so the only variable is the layout arithmetic.
    let packed_k = cb.copy_to_host(kreg).unwrap();
    let packed_v = cb.copy_to_host(vreg).unwrap();
    let mut ref_k = vec![0f32; n_ctx * nkt];
    let mut ref_v = vec![0f32; n_ctx * nkt];
    for &p in &pos {
        crate::graph::kvformat::unpack_q8_0_cells(
            &packed_k,
            nkt,
            p,
            1,
            &mut ref_k[p * nkt..(p + 1) * nkt],
        );
        crate::graph::kvformat::unpack_q8_0_cells(
            &packed_v,
            nkt,
            p,
            1,
            &mut ref_v[p * nkt..(p + 1) * nkt],
        );
    }
    assert!(
        ref_k.iter().any(|x| *x != 0.0),
        "the dequantized reference is all zero; the comparison would be vacuous"
    );
    cb.set_kv_layout_for_test(crate::cuda::KV_LAYOUT_F32);
    cb.write_host(kf32, &ref_k).unwrap();
    cb.write_host(vf32, &ref_v).unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kf32, xb_p],
        ob_at,
        Some((kf32, vf32)),
    )
    .unwrap();
    let want = cb.copy_to_host(ob_at).unwrap();

    let worst = got_q8
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let first = got_q8
        .iter()
        .zip(&want)
        .position(|(a, b)| a != b)
        .map(|i| (i, got_q8[i], want[i]));
    assert!(
        worst <= 1e-6,
        "the packed attention read is not the f32 kernel over the dequantized \
         cells: max |Δ| = {worst}, first {first:?}"
    );
    eprintln!(
        "[c4s2] packed vs dequantized f32 attention: max |Δ| = {worst} ({} of {} \
         elements differ)",
        got_q8.iter().zip(&want).filter(|(a, b)| a != b).count(),
        got_q8.len()
    );

    // …and the arm is real: a zero-output kernel would also compare equal to a
    // zero reference, so require non-trivial output.
    assert!(
        got_q8.iter().any(|x| x.abs() > 1e-6),
        "the packed attention returned all zeros"
    );

    // ── #186: the `__dp4a` decode arm ───────────────────────────────────────
    //
    // The int-dot path lives only in the `nt == 1` split-K kernel, which the
    // `nt = 3` arms above never reach. This arm runs it at `nt = 1` with an
    // explicit span `[0, 2)` over **two** stored cells, so the query's K scores
    // actually reach the output through the softmax: with a single key the score
    // cancels and the arm would be vacuous (the first version of this arm used
    // `positions = [0]`/one cell and a mutated block base still passed — the
    // mutation is what found it).
    //
    // `positions = [0]` keeps the rope the identity permutation, and the query is
    // built **exactly Q8_0-representable** (`amax = 1`, values in `{-1, 0, 1}`):
    // its block scale is then `1/127` and its quants are `±127`, so the int dot's
    // answer equals the f32 kernel's over the dequantized cells and the bound
    // stays tight enough to catch a wrong block base, scale offset or byte pack.
    // The query quantization itself is a numerics change with its own class — it
    // is measured by the real-model gate
    // `a_packed_kv_cache_answers_like_the_f32_one`, not pinned here.
    {
        const NCELL: usize = 2;
        let mut b1 = GraphBuilder::new();
        b1.set_kv_format(crate::graph::kvformat::KvFormat::Q8_0);
        b1.set_explicit_span(true);
        let q1 = b1.input("q", [nh * hd, 1, 1, 1], DType::F32);
        let k1 = b1.input("k", [nkt, NCELL, 1, 1], DType::F32);
        let v1 = b1.input("v", [nkt, NCELL, 1, 1], DType::F32);
        let p1 = b1.input("positions", [1, 1, 1, 1], DType::I32);
        let st1 = b1.kvcache_store(0, k1, v1, n_ctx);
        let ld1 = b1.kvcache_load(0, nkt, n_ctx, nk_h);
        let qr1 = b1.rope(
            q1,
            p1,
            RopeStyle::NonInterleaved,
            RoPEMeta {
                freq_base: 10000.0,
                freq_scale: 1.0,
                n_head: nh,
                hd,
            },
        );
        let at1 = b1.attn(
            qr1,
            ld1,
            p1,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk_h,
                hd,
                hd_kv: hd,
                nkt,
                scale,
            },
        );
        b1.output(at1);
        let g1 = b1.build();

        // The Attn node's 4th input is the builder's span node; one `[lo, hi)`
        // pair per query, here `[0, 2)` over both stored cells.
        let span1: Vec<f32> = [0u32, NCELL as u32]
            .iter()
            .map(|&x| f32::from_bits(x))
            .collect();

        let qv1: Vec<f32> = (0..nh * hd)
            .map(|i| match i % 32 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        assert!(
            qv1.chunks(32).all(|c| c.iter().any(|x| *x != 0.0)),
            "every query block must have a non-zero amax, or its scale is 0"
        );
        let s1_q = cb.alloc_buffer(nh * hd);
        let s1_k = cb.alloc_buffer(nkt * NCELL);
        let s1_v = cb.alloc_buffer(nkt * NCELL);
        let s1_p = cb.alloc_buffer(1);
        let s1_c = cb.alloc_buffer(NCELL);
        let s1_w = cb.alloc_buffer(2);
        let o1_q = cb.alloc_buffer(nh * hd);
        let o1_a = cb.alloc_buffer(nh * hd);
        let rk1 = cb.alloc_buffer(n_ctx * row_elems);
        let rv1 = cb.alloc_buffer(n_ctx * row_elems);
        let fk1 = cb.alloc_buffer(n_ctx * nkt);
        let fv1 = cb.alloc_buffer(n_ctx * nkt);
        let bits1 = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
        cb.write_host(s1_q, &qv1).unwrap();
        cb.write_host(s1_k, &ks[..nkt * NCELL]).unwrap();
        cb.write_host(s1_v, &vs[..nkt * NCELL]).unwrap();
        cb.write_host(s1_p, &bits1(&[0])).unwrap();
        cb.write_host(s1_c, &bits1(&[0, 1])).unwrap();
        cb.write_host(s1_w, &span1).unwrap();
        cb.write_host(rk1, &vec![0f32; n_ctx * row_elems]).unwrap();
        cb.write_host(rv1, &vec![0f32; n_ctx * row_elems]).unwrap();

        cb.set_kv_q8_for_test();
        cb.exec_ids(&g1.nodes[st1], &[s1_k, s1_v, s1_c], rk1, Some((rk1, rv1)))
            .unwrap();
        cb.exec_ids(&g1.nodes[qr1], &[s1_q, s1_p], o1_q, None)
            .unwrap();
        crate::testfail::reset_checked();
        cb.exec_ids(
            &g1.nodes[at1],
            &[o1_q, rk1, s1_p, s1_w],
            o1_a,
            Some((rk1, rv1)),
        )
        .unwrap();
        let got1 = cb.copy_to_host(o1_a).unwrap();
        // Rule 3's observation half: the counter is bumped by the launch
        // chokepoint, so it proves the arm the A/B selects actually ran — and
        // tracks the control (`MINFER_NO_DP4A_Q8_KV=1`) rather than lying about
        // it.
        assert_eq!(
            crate::testfail::checked("cuda_q8_kv_dp4a"),
            u64::from(crate::cuda::q8_kv_dp4a_enabled()),
            "the Q8_0 decode launch's dp4a observation counter must match its control"
        );

        let pk1 = cb.copy_to_host(rk1).unwrap();
        let pv1 = cb.copy_to_host(rv1).unwrap();
        let mut ref_k1 = vec![0f32; n_ctx * nkt];
        let mut ref_v1 = vec![0f32; n_ctx * nkt];
        crate::graph::kvformat::unpack_q8_0_cells(&pk1, nkt, 0, NCELL, &mut ref_k1[..nkt * NCELL]);
        crate::graph::kvformat::unpack_q8_0_cells(&pv1, nkt, 0, NCELL, &mut ref_v1[..nkt * NCELL]);
        cb.set_kv_layout_for_test(crate::cuda::KV_LAYOUT_F32);
        cb.write_host(fk1, &ref_k1).unwrap();
        cb.write_host(fv1, &ref_v1).unwrap();
        cb.exec_ids(
            &g1.nodes[at1],
            &[o1_q, fk1, s1_p, s1_w],
            o1_a,
            Some((fk1, fv1)),
        )
        .unwrap();
        let want1 = cb.copy_to_host(o1_a).unwrap();
        let worst1 = got1
            .iter()
            .zip(&want1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            got1.iter().any(|x| x.abs() > 1e-6),
            "the dp4a decode arm returned all zeros"
        );
        assert!(
            worst1 <= 1e-5,
            "the dp4a decode K dot is not the f32 kernel over the dequantized \
             cells with an exactly-representable query: max |Δ| = {worst1}"
        );
        eprintln!(
            "[c4s2] dp4a decode (nt=1, span over {} cells) vs dequantized f32 \
             attention: max |Δ| = {worst1}, dp4a arm {}",
            NCELL,
            crate::cuda::q8_kv_dp4a_enabled()
        );
    }
}

/// #144 item 1: the **packed fused decode epilogue** must write the unfused
/// chain's bytes. The reference is the CPU quantizer (`quants::
/// quantize_row_q8_0`) run over the roped / bias-added values computed on the
/// host — an implementation the kernel does not share — so a wrong block
/// index, rope pairing, scale offset or quant offset moves a value by whole
/// quant steps.
///
/// Two arms:
/// - `pos = 0` (the rope is the identity permutation: `cs = 1`, `sn = 0`), where
///   K and V are compared **byte for byte** — pairing, block math and the
///   quantizer have no transcendental to hide behind;
/// - `pos = 7`, where the angle is real: q (never quantized) is compared as a
///   value, and K is compared after dequantization at one quant step's class,
///   because the device's `cosf`/`sinf` may differ from the host's in the last
///   ulp and that can flip a quant sitting on a boundary.
#[test]
fn cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer() {
    use crate::graph::kvformat::KvFormat;
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    let (nh, nk_h, hd) = (4usize, 2usize, 64usize);
    let (nqt, nkt) = (nh * hd, nk_h * hd);
    let row_words = KvFormat::Q8_0.row_elems(nkt);
    let row_bytes = KvFormat::Q8_0.row_bytes(nkt);
    let half = hd / 2;
    let (fb, fs) = (10000.0f32, 1.0f32);

    let vals = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * seed + 7) % 17) as f32 / 4.0 - 2.0)
            .collect()
    };
    let (qs, ks, vs) = (vals(37, nqt), vals(41, nkt), vals(57, nkt));
    let (bq, bk, bv) = (vals(13, nqt), vals(19, nkt), vals(23, nkt));

    let (b_q, b_k, b_v) = (
        cb.alloc_buffer(nqt),
        cb.alloc_buffer(nkt),
        cb.alloc_buffer(nkt),
    );
    let (b_bq, b_bk, b_bv) = (
        cb.alloc_buffer(nqt),
        cb.alloc_buffer(nkt),
        cb.alloc_buffer(nkt),
    );
    let b_p = cb.alloc_buffer(1);
    let b_c = cb.alloc_buffer(1);
    // Two packed rows so the two arms cannot alias.
    let (kreg, vreg) = (
        cb.alloc_buffer(2 * row_words),
        cb.alloc_buffer(2 * row_words),
    );
    cb.write_host(b_q, &qs).unwrap();
    cb.write_host(b_k, &ks).unwrap();
    cb.write_host(b_v, &vs).unwrap();
    cb.write_host(b_bq, &bq).unwrap();
    cb.write_host(b_bk, &bk).unwrap();
    cb.write_host(b_bv, &bv).unwrap();
    cb.write_host(kreg, &vec![0f32; 2 * row_words]).unwrap();
    cb.write_host(vreg, &vec![0f32; 2 * row_words]).unwrap();

    // The host's rope + bias, per head (neox pairing d <-> d + hd/2).
    let roped = |src: &[f32], bias: &[f32], heads: usize, pos: usize| -> Vec<f32> {
        let mut out = vec![0f32; heads * hd];
        for h in 0..heads {
            for d in 0..hd {
                let dd = if d < half { d } else { d - half };
                let ja = h * hd + dd;
                let jb = ja + half;
                let x0 = src[ja] + bias[ja];
                let x1 = src[jb] + bias[jb];
                let theta = pos as f32 * fs / fb.powf((2.0 * dd as f32) / hd as f32);
                let (cs, sn) = (theta.cos(), theta.sin());
                out[h * hd + d] = if d < half {
                    x0 * cs - x1 * sn
                } else {
                    x0 * sn + x1 * cs
                };
            }
        }
        out
    };
    // The packed payload the CPU quantizer produces for a flat value row.
    let pack = |v: &[f32]| -> Vec<u8> {
        let nblk = v.len() / 32;
        let mut raw = vec![0u8; nblk * 34];
        for b in 0..nblk {
            let q = crate::quants::quantize_row_q8_0(&v[b * 32..(b + 1) * 32]);
            raw[b * 34..(b + 1) * 34].copy_from_slice(&q);
        }
        raw
    };
    let cell_bytes = |region: &[f32], row: usize| -> Vec<u8> {
        let words = &region[row * row_words..(row + 1) * row_words];
        let mut out = Vec::with_capacity(words.len() * 4);
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    };

    let run = |cb: &mut CudaBackend, pos: usize, row: usize| {
        // #188: the direct `attn_bias_rope_store*` calls below bypass
        // `execute_node`, so bind this backend's stream explicitly.
        let _bound = cb.bind();
        // q is roped IN PLACE, so each arm starts from the original input.
        cb.write_host(b_q, &qs).unwrap();
        cb.write_host(b_p, &[f32::from_bits(pos as u32)]).unwrap();
        cb.write_host(b_c, &[f32::from_bits(row as u32)]).unwrap();
        let (q, k, v) = (
            cb.ptr_of(b_q).unwrap(),
            cb.ptr_of(b_k).unwrap(),
            cb.ptr_of(b_v).unwrap(),
        );
        let (pq, pk, pv) = (
            cb.ptr_of(b_bq).unwrap(),
            cb.ptr_of(b_bk).unwrap(),
            cb.ptr_of(b_bv).unwrap(),
        );
        let pp = cb.ptr_of(b_p).unwrap();
        let cc = cb.ptr_of(b_c).unwrap();
        cb.state.attn_bias_rope_store_q8_0(
            q,
            k,
            v,
            pq,
            pk,
            pv,
            cb.ptr_of(kreg).unwrap(),
            cb.ptr_of(vreg).unwrap(),
            nqt,
            nkt,
            hd,
            fb,
            fs,
            pp,
            cc,
            row_bytes,
        );
        cb.synchronize();
        (
            cb.copy_to_host(b_q).unwrap(),
            cb.copy_to_host(kreg).unwrap(),
            cb.copy_to_host(vreg).unwrap(),
        )
    };

    // ── arm 1: pos = 0 — K and V byte for byte ──────────────────────────
    let (q0, k0, v0) = run(&mut cb, 0, 0);
    let want_k = pack(&roped(&ks, &bk, nk_h, 0));
    let want_v = pack(&(0..nkt).map(|i| vs[i] + bv[i]).collect::<Vec<f32>>());
    assert_eq!(
        &cell_bytes(&k0, 0)[..want_k.len()],
        &want_k[..],
        "the packed K cell is not the CPU quantizer's bytes at pos = 0"
    );
    assert_eq!(
        &cell_bytes(&v0, 0)[..want_v.len()],
        &want_v[..],
        "the packed V cell is not the CPU quantizer's bytes at pos = 0"
    );
    let want_q0 = roped(&qs, &bq, nh, 0);
    let qdelta0 = q0
        .iter()
        .zip(&want_q0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        qdelta0 <= 1e-6,
        "the roped q buffer is not the host's identity-rope result: max |Δ| = {qdelta0}"
    );

    // ── arm 2: pos = 7 — the angle is real ─────────────────────────────
    let (q7, k7, v7) = run(&mut cb, 7, 1);
    let want_q7 = roped(&qs, &bq, nh, 7);
    let qdelta = q7
        .iter()
        .zip(&want_q7)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        qdelta <= 1e-5,
        "the roped q buffer diverges from the host's rope: max |Δ| = {qdelta}"
    );
    assert_eq!(
        &cell_bytes(&v7, 1)[..want_v.len()],
        &want_v[..],
        "V does not depend on the rope angle, so its bytes must still match"
    );
    let mut got_k7 = vec![0f32; nkt];
    crate::graph::kvformat::unpack_q8_0_cells(&k7, nkt, 1, 1, &mut got_k7);
    let kdelta = got_k7
        .iter()
        .zip(&roped(&ks, &bk, nk_h, 7))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        kdelta <= 0.03,
        "the packed K cell is not the host's roped values at one quant step's class: \
         max |Δ| = {kdelta}"
    );
    assert!(
        got_k7.iter().any(|x| x.abs() > 1e-3),
        "the packed K cell is all zeros; the comparison would be vacuous"
    );
}

/// C8b S4 device A/B — what naming a row through the run list costs over the
/// span's `row0 + i`, at the decode shape that pays it on every step.
///
/// **The value arm is the primary signal (gate contract rule 1).** The gate
/// used to be a pure stopwatch: it timed two arms and never looked at what
/// they computed, so two equally-wrong kernels would pass and a loaded box
/// could decide the verdict. Before any timing it now asserts that
/// - a **one-row** map window at a non-zero cell returns exactly that row's
///   V — an absolute value computed on the host, not a relation between the
///   two modes (softmax over one key is exactly 1.0, so the kernel is the
///   identity on V);
/// - a **two-run** map window with a non-zero base cell returns the span's
///   bytes over the same rows, bit for bit — one run is indistinguishable to
///   a resolver that reads `(cell, len)` as `(lo, hi)`, two runs are not;
/// - the map instantiation actually ran (the observation half of rule 3):
///   `testfail::note_checked("cuda_attn_map_window")` moves on a map call and
///   not on a span call.
///
/// **The timing arm is a paired sign test, not a median (issue #189).** Both
/// phases (decode and prefill) interleave matched rounds of the two modes and
/// count how many pairs put map above `1.25x` span. The gate refuses only when
/// [`SIGN_TEST_REFUSALS`] = 7 of [`PAIRS`] = 9 pairs do — the one-sided sign
/// test at `alpha = 46/512 = 0.090`. A median of the same 9 pairs flips at 5,
/// and the recorded parallel device run (GB10 sm_121, 2026-09-26) measured
/// exactly 5 disturbed pairs (`[0.632, 0.697, 0.989, 1.091, 1.398, 1.440,
/// 1.466, 2.198, 6.695]`, median 1.398x) and failed a kernel that was not
/// slower; the sign test passes that run and still fails a doubled map cost
/// (`MINFER_S4_AB_MAP_REPS=2`, the reproducible form of #123's map-work
/// doubling), which moves every pair.
///
/// The bar is unchanged at 1.25x and the timing fixture is the pre-#189 one
/// (one run at cell 0), so the recorded margins stay comparable. Run with
/// `--ignored --nocapture` to see every sample and the refusal count.
#[test]
#[ignore = "timing: needs a CUDA device"]
fn cuda_map_window_costs_no_more_than_the_span_it_replaces() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    // Issue #188: this gate drives `CudaBackend`/`CudaState` **directly** — no
    // `scheduler::execute`. It no longer takes the #185 device-entry guard:
    // the direct `attn_bias_rope_store` calls below bind `cb`'s own stream
    // (see `run`), so they no longer land on the context stream that another
    // test could be capturing. The process-wide `model_load_guard` stays —
    // this gate is a full model-load-shaped device workload.
    let _guard = crate::cuda::CudaState::model_load_guard();
    const KMAX: usize = crate::graph::kvcache::KV_MAP_MAX_SPANS;
    // The 7B decode shape (hd 128, 4 KV heads) at a 2K window: long enough
    // that the split-K body does real work, short enough that the 4-warp
    // hybrid gate stays out of it.
    let (nh, nk, hd, nkv) = (28usize, 4usize, 128usize, 2048usize);
    let nkt = nk * hd;
    let n_ctx = 4096usize;
    let bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    let scale = 1.0 / (hd as f32).sqrt();
    let (reps, warmup) = (100usize, 3usize);
    let (preps, pwarm) = (50usize, 3usize);
    // The prefill fixture's `nt`. `q` is read as `nt` token rows of
    // `nh * hd` floats (`fa_prefill_f16kv`: `q[t * nh * hd + h * hd + d]`,
    // `t < nt`), so it needs its own `nt`-row q buffer. Reusing the decode
    // phase's single-row `qb` here made the kernel read ~7 MB past it: a
    // silent read of whatever device memory followed when those pages were
    // mapped, and a latched `cudaErrorIllegalAddress` (700) when they were
    // not — which then failed every later `cudaMemGetInfo` in the process.
    let nt = 512usize;
    let qb = cb.alloc_buffer(nh * hd);
    let kreg = cb.alloc_buffer(n_ctx * nkt);
    let vreg = cb.alloc_buffer(n_ctx * nkt);
    let ob = cb.alloc_buffer(nh * hd);
    let spb = cb.alloc_buffer(2);
    let mpb = cb.alloc_buffer(KMAX * 2);
    let spb2 = cb.alloc_buffer(2 * nt);
    let mpb2 = cb.alloc_buffer(nt * KMAX * 2);
    let qb2 = cb.alloc_buffer(nh * hd * nt);
    let ob2 = cb.alloc_buffer(nh * hd * nt);
    cb.write_host(qb, &vec![0.01f32; nh * hd]).unwrap();
    let enc = |x: f32| -> f32 { f32::from_bits(half::f16::from_f32(x).to_bits() as u32) };

    // ── the value arms (gate contract rule 1) ────────────────────────────
    //
    // The fixture is harder than the timed one on purpose: a non-zero base
    // cell (so a resolver that ignores a run's `cell` cannot alias row 0) and
    // two ascending runs (so a resolver that reads the map as one `(lo, hi)`
    // pair cannot cover the window). Rows [VBASE, VBASE + nkv) carry distinct
    // K/V, so resolving the wrong cell changes the output instead of aliasing
    // the same constant.
    const VBASE: usize = 512;
    let distinct = |seed: u32, r: usize| -> Vec<f32> {
        (0..nkt)
            .map(|i| {
                (((i as u32)
                    .wrapping_mul(seed)
                    .wrapping_add((r as u32).wrapping_mul(2_654_435_761)))
                    % 101) as f32
                    / 101.0
                    - 0.5
            })
            .collect()
    };
    let mut kfill = vec![0.02f32; n_ctx * nkt];
    let mut vfill = vec![0.03f32; n_ctx * nkt];
    for r in VBASE..VBASE + nkv {
        kfill[r * nkt..(r + 1) * nkt].copy_from_slice(&distinct(31, r));
        vfill[r * nkt..(r + 1) * nkt].copy_from_slice(&distinct(57, r));
    }
    cb.set_kv_f16_for_test(false);
    cb.write_host(kreg, &kfill).unwrap();
    cb.write_host(vreg, &vfill).unwrap();

    // One decode-window call, returning the output buffer.
    let run = |cb: &CudaBackend, mode: crate::cuda::AttnWindow, win: usize| -> Vec<f32> {
        cb.state.gqa_attn_split(
            cb.ptr_of(qb).unwrap(),
            cb.ptr_of(kreg).unwrap(),
            cb.ptr_of(vreg).unwrap(),
            cb.ptr_of(ob).unwrap(),
            cb.ptr_of(win).unwrap(),
            mode.code(),
            nh,
            nk,
            hd,
            scale,
            crate::cuda::KV_LAYOUT_F32,
            nkt * 4,
        );
        cb.copy_to_host(ob).unwrap()
    };

    // (a) the absolute arm: a window of one row returns that row's V exactly.
    cb.write_host(spb, &bits(&[VBASE as u32, 1])).unwrap();
    let mut one_map = vec![0u32; KMAX * 2];
    one_map[0] = VBASE as u32;
    one_map[1] = 1;
    cb.write_host(mpb, &bits(&one_map)).unwrap();
    let got = run(&cb, crate::cuda::AttnWindow::Map, mpb);
    let gqa = nh / nk;
    let want = &vfill[VBASE * nkt..(VBASE + 1) * nkt];
    for h in 0..nh {
        let kw = (h / gqa) * hd;
        for d in 0..hd {
            assert_eq!(
                got[h * hd + d].to_bits(),
                want[kw + d].to_bits(),
                "a one-row map window at cell {VBASE} did not return that row's V \
                 (head {h}, dim {d})"
            );
        }
    }
    assert!(
        got.iter().any(|x| x.abs() > 1e-3),
        "the one-row map window returned all zeros; the comparison is vacuous"
    );

    // (b) the relation arm: a two-run window with a non-zero base equals the
    // span over the same rows, bit for bit.
    const VW: usize = 64;
    const VSPLIT: usize = 17;
    cb.write_host(spb, &bits(&[VBASE as u32, (VBASE + VW) as u32]))
        .unwrap();
    let mut map_win = vec![0u32; KMAX * 2];
    map_win[0] = VBASE as u32;
    map_win[1] = VSPLIT as u32;
    map_win[2] = (VBASE + VSPLIT) as u32;
    map_win[3] = (VW - VSPLIT) as u32;
    cb.write_host(mpb, &bits(&map_win)).unwrap();
    let o_span = run(&cb, crate::cuda::AttnWindow::Span, spb);
    let o_map = run(&cb, crate::cuda::AttnWindow::Map, mpb);
    let first = o_span
        .iter()
        .zip(&o_map)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|i| (i, o_span[i], o_map[i]));
    assert!(
        first.is_none(),
        "a two-run map window over cells [{VBASE}, {}) diverges from the span over the \
         same rows: first {first:?}",
        VBASE + VW
    );
    assert!(
        o_map.iter().any(|x| x.abs() > 1e-3),
        "the two-run map window returned all zeros; the comparison is vacuous"
    );

    // (c) the counted observation (rule 3's observation half): the map
    // instantiation ran, and a span call does not touch the map counter.
    crate::testfail::reset_checked();
    let _ = run(&cb, crate::cuda::AttnWindow::Span, spb);
    assert_eq!(
        crate::testfail::checked("cuda_attn_map_window"),
        0,
        "a span call must not bump the map-window chokepoint"
    );
    let _ = run(&cb, crate::cuda::AttnWindow::Map, mpb);
    assert_eq!(
        crate::testfail::checked("cuda_attn_map_window"),
        1,
        "the map-mode attention launch never reached its chokepoint; the arm under test \
         did not run"
    );

    // (d) the same relation on the *prefill* path the parallel run failed on:
    // an f16 KV region with distinct rows and one two-run map window per query.
    cb.set_kv_f16_for_test(true);
    let mut kfill16 = vec![enc(0.02f32); n_ctx * nkt];
    let mut vfill16 = vec![enc(0.03f32); n_ctx * nkt];
    for r in VBASE..VBASE + nt {
        for (e, x) in distinct(31, r).iter().enumerate() {
            kfill16[r * nkt + e] = enc(*x);
        }
        for (e, x) in distinct(57, r).iter().enumerate() {
            vfill16[r * nkt + e] = enc(*x);
        }
    }
    cb.write_host(kreg, &kfill16).unwrap();
    cb.write_host(vreg, &vfill16).unwrap();
    cb.write_host(qb2, &vec![enc(0.01f32); nh * hd * nt])
        .unwrap();
    let mut span3 = vec![0u32; 2 * nt];
    let mut map3 = vec![0u32; nt * KMAX * 2];
    for t in 0..nt {
        let n = t + 1;
        let a = n / 2;
        span3[t] = VBASE as u32;
        span3[nt + t] = (VBASE + n) as u32;
        let at = t * KMAX * 2;
        map3[at] = VBASE as u32;
        map3[at + 1] = a as u32;
        map3[at + 2] = (VBASE + a) as u32;
        map3[at + 3] = (n - a) as u32;
    }
    cb.write_host(spb2, &bits(&span3)).unwrap();
    cb.write_host(mpb2, &bits(&map3)).unwrap();
    let prefill_run = |cb: &CudaBackend, mode: crate::cuda::AttnWindow, win: usize| -> Vec<f32> {
        cb.state.gqa_attn_kv_prefill(
            cb.ptr_of(qb2).unwrap(),
            cb.ptr_of(kreg).unwrap(),
            cb.ptr_of(vreg).unwrap(),
            cb.ptr_of(ob2).unwrap(),
            cb.ptr_of(win).unwrap(),
            mode.code(),
            crate::cuda::KV_LAYOUT_F16,
            nh,
            nk,
            hd,
            scale,
            nkt * 2,
            nt,
        );
        cb.copy_to_host(ob2).unwrap()
    };
    let p_span_v = prefill_run(&cb, crate::cuda::AttnWindow::Span, spb2);
    let p_map_v = prefill_run(&cb, crate::cuda::AttnWindow::Map, mpb2);
    let p_first = p_span_v
        .iter()
        .zip(&p_map_v)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|i| (i, p_span_v[i], p_map_v[i]));
    assert!(
        p_first.is_none(),
        "a two-run map prefill over cells [{VBASE}, {}) diverges from the span over the \
         same rows: first {p_first:?}",
        VBASE + nt
    );
    assert!(
        p_map_v.iter().any(|x| x.abs() > 1e-3),
        "the two-run map prefill returned all zeros; the comparison is vacuous"
    );
    crate::testfail::reset_checked();
    let _ = prefill_run(&cb, crate::cuda::AttnWindow::Map, mpb2);
    assert_eq!(
        crate::testfail::checked("cuda_attn_map_window"),
        1,
        "the map-mode prefill launch never reached its chokepoint"
    );

    // ── the timing fixture (the pre-#189 one: constant K/V, one run at cell
    // 0), so the recorded margins stay comparable ─────────────────────────
    cb.set_kv_f16_for_test(false);
    cb.write_host(kreg, &vec![0.02f32; n_ctx * nkt]).unwrap();
    cb.write_host(vreg, &vec![0.03f32; n_ctx * nkt]).unwrap();
    cb.write_host(spb, &bits(&[0, nkv as u32])).unwrap();
    let mut map = vec![0u32; KMAX * 2];
    map[0] = 0;
    map[1] = nkv as u32;
    cb.write_host(mpb, &bits(&map)).unwrap();
    // µs/launch for one timed round of `reps` launches, after `warmup`
    // untimed ones. A first launch pays the module load, which is not the
    // measurement (the prewarm list covers the production instantiations, not
    // necessarily these).
    let time = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                mode: crate::cuda::AttnWindow,
                reps: usize,
                warmup: usize|
     -> f64 {
        let win = if mode == crate::cuda::AttnWindow::Map {
            mpb
        } else {
            spb
        };
        let mut call = |cb: &mut crate::graph::cuda_backend::CudaBackend| {
            cb.state.gqa_attn_split(
                cb.ptr_of(qb).unwrap(),
                cb.ptr_of(kreg).unwrap(),
                cb.ptr_of(vreg).unwrap(),
                cb.ptr_of(ob).unwrap(),
                cb.ptr_of(win).unwrap(),
                mode.code(),
                nh,
                nk,
                hd,
                scale,
                crate::cuda::KV_LAYOUT_F32,
                nkt * 4,
            );
        };
        for _ in 0..warmup {
            call(cb);
        }
        cb.state.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            call(cb);
        }
        cb.state.sync();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };

    // The timing statistic (issue #189). The old form asserted the **median of
    // the per-round ratios**, which a loaded harness flips once `rounds / 2`
    // pairs are disturbed: the recorded parallel device run (GB10 sm_121,
    // 2026-09-26) measured 5 of 9 pairs above 1.25x (median 1.398x) and failed
    // a kernel that was not slower. The verdict is now a **paired sign test**:
    // the count of pairs above the bar, fixed in advance at `PAIRS = 9`, and
    // the gate refuses only at `SIGN_TEST_REFUSALS = 7` — the one-sided
    // binomial tail `P(X >= 7 | fair coin) = 46/512 = 0.090`. A minority of
    // disturbed rounds cannot decide it; the recorded run's 5 does not, and
    // doubling the map work (`MINFER_S4_AB_MAP_REPS=2`) moves all 9 and does.
    //
    // The bar is **not** widened: it is the pre-#123 gate's 1.25x, and the
    // margin stays justified by measurement. On an idle GB10 the median ratio
    // is 1.001-1.004 (decode) and 1.087-1.107 (prefill) over 6 runs; with 16
    // CPU spinners plus two concurrent CUDA attention loops it stays
    // 1.001-1.018 and 1.079-1.145, although individual rounds reach 1.4-8.6x.
    const MAX_MAP_OVER_SPAN: f64 = 1.25;
    let mut span_us: Vec<f64> = Vec::with_capacity(PAIRS);
    let mut map_us: Vec<f64> = Vec::with_capacity(PAIRS);
    for _ in 0..PAIRS {
        span_us.push(time(&mut cb, crate::cuda::AttnWindow::Span, reps, warmup));
        map_us.push(time(&mut cb, crate::cuda::AttnWindow::Map, reps, warmup));
    }
    let (d_refusals, d_span, d_map, d_ratios) =
        sign_test_ratio(&span_us, &map_us, MAX_MAP_OVER_SPAN);
    eprintln!(
        "[s4-ab] decode nkv={nkv} nh={nh} nk={nk} hd={hd}: span {d_span:.1} / map {d_map:.1} \
         us/launch ({PAIRS} interleaved matched pairs of {reps}); per-round ratios \
         {d_ratios:?} — {d_refusals}/{PAIRS} above {MAX_MAP_OVER_SPAN}x (sign test refuses at \
         {SIGN_TEST_REFUSALS})"
    );
    assert!(
        d_refusals < SIGN_TEST_REFUSALS,
        "the map window is above {MAX_MAP_OVER_SPAN}x the span in {d_refusals} of {PAIRS} \
         matched pairs (medians {d_map:.1} vs {d_span:.1} us/launch; per-round ratios \
         {d_ratios:?}); the sign test refuses at {SIGN_TEST_REFUSALS}, so this is a systematic \
         map cost, not a load spike"
    );

    // The other half of the A/B: a *prefill* window (each query's whole
    // prefix), where the window is walked tile by tile. Both modes run FA here
    // since S4 taught its staging loop to resolve runs, so this measures the
    // resolution cost on the prefill path too. The buffers were allocated with
    // the value arms above, so this only rewrites the timing fixture.
    cb.set_kv_f16_for_test(true);
    cb.write_host(kreg, &vec![enc(0.02f32); n_ctx * nkt])
        .unwrap();
    cb.write_host(vreg, &vec![enc(0.03f32); n_ctx * nkt])
        .unwrap();
    cb.write_host(qb2, &vec![enc(0.01f32); nh * hd * nt])
        .unwrap();
    let mut span2 = vec![0u32; 2 * nt];
    let mut map2 = vec![0u32; nt * KMAX * 2];
    for t in 0..nt {
        span2[t] = 0;
        span2[nt + t] = (t + 1) as u32;
        map2[t * KMAX * 2] = 0;
        map2[t * KMAX * 2 + 1] = (t + 1) as u32;
    }
    cb.write_host(spb2, &bits(&span2)).unwrap();
    cb.write_host(mpb2, &bits(&map2)).unwrap();
    let time_prefill = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                        mode: crate::cuda::AttnWindow,
                        reps: usize,
                        warmup: usize|
     -> f64 {
        let win = if mode == crate::cuda::AttnWindow::Map {
            mpb2
        } else {
            spb2
        };
        let mut call = |cb: &mut crate::graph::cuda_backend::CudaBackend| {
            cb.state.gqa_attn_kv_prefill(
                cb.ptr_of(qb2).unwrap(),
                cb.ptr_of(kreg).unwrap(),
                cb.ptr_of(vreg).unwrap(),
                cb.ptr_of(ob2).unwrap(),
                cb.ptr_of(win).unwrap(),
                mode.code(),
                crate::cuda::KV_LAYOUT_F16,
                nh,
                nk,
                hd,
                scale,
                nkt * 2,
                nt,
            );
        };
        for _ in 0..warmup {
            call(cb);
        }
        cb.state.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            call(cb);
        }
        cb.state.sync();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    // The same paired sign test as the decode half — this is the assertion
    // that failed on the loaded parallel GB10, and the old form was weaker
    // here than there (a single 20-launch block per mode, no interleaving and
    // no round-to-round statistic at all). 9 matched pairs × 50 launches at
    // ~85 µs/launch is ~40 ms per mode.
    if hd == 128 {
        let mut pspan_us: Vec<f64> = Vec::with_capacity(PAIRS);
        let mut pmap_us: Vec<f64> = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            pspan_us.push(time_prefill(
                &mut cb,
                crate::cuda::AttnWindow::Span,
                preps,
                pwarm,
            ));
            pmap_us.push(time_prefill(
                &mut cb,
                crate::cuda::AttnWindow::Map,
                preps,
                pwarm,
            ));
        }
        let (p_refusals, p_span, p_map, p_ratios) =
            sign_test_ratio(&pspan_us, &pmap_us, MAX_MAP_OVER_SPAN);
        eprintln!(
            "[s4-ab] prefill nt={nt} nkv={nt} hd={hd}: span {p_span:.1} / map {p_map:.1} \
             us/launch ({PAIRS} interleaved matched pairs of {preps}); per-round ratios \
             {p_ratios:?} — {p_refusals}/{PAIRS} above {MAX_MAP_OVER_SPAN}x (sign test refuses \
             at {SIGN_TEST_REFUSALS})"
        );
        assert!(
            p_refusals < SIGN_TEST_REFUSALS,
            "the map prefill is above {MAX_MAP_OVER_SPAN}x the span in {p_refusals} of {PAIRS} \
             matched pairs (medians {p_map:.1} vs {p_span:.1} us/launch; per-round ratios \
             {p_ratios:?}); the sign test refuses at {SIGN_TEST_REFUSALS}"
        );
    }
}

/// reference with the standard q8-activation tolerance instead.
#[test]
fn cuda_verify_attention_nt_invariance() {
    // doc 94: the verify batch's attention must be bitwise-equal to the
    // nt=1 decode path at every position — the greedy identity at the
    // kernel level. Same KV buffer (prefix + the 3 intra-batch rows),
    // same queries; batched (positions [P, P+1, P+2]) vs three decode
    // calls; compare outputs bitwise. Both KV dtype variants, the small
    // parity fixture AND the 14B decode dims (hd=128 -> the decode
    // path's dual-kernel gate is live; prefix 512 keeps rpw < 16 so the
    // incumbent 1-warp body owns both paths).
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    let _guard = crate::cuda::CudaState::model_load_guard();

    let nts = [3usize, 8usize];
    let shapes: [(usize, usize, usize, usize, f32); 2] = [
        (4, 2, 8, 100, 0.3),                    // parity fixture dims
        (40, 8, 128, 512, 0.08838834764831845), // 14B decode dims
    ];
    for (nh, nk, hd, prefix, scale) in shapes {
        for nt in nts {
            let rows = prefix + nt;

            // deterministic inputs
            let mut s: u64 = 0x9E3779B97F4A7C15;
            let mut next = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let q: Vec<f32> = (0..nt * nh * hd)
                .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
                .collect();
            // f16 K/V bit patterns packed two-per-f32
            let kv_f16: Vec<u16> = (0..rows * nk * hd)
                .map(|_| half::f16::from_f32(((next() % 2001) as f32 - 1000.0) / 1000.0).to_bits())
                .collect();
            let kv_f32: Vec<f32> = (0..rows * nk * hd)
                .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
                .collect();

            let pack_u16 = |v: &[u16]| -> Vec<f32> {
                v.chunks(2)
                    .map(|c| {
                        f32::from_bits(
                            (*c.first().unwrap()) as u32 | ((*c.get(1).unwrap_or(&0)) as u32) << 16,
                        )
                    })
                    .collect()
            };

            for f16_kv in [true, false] {
                // f32 KV needs rows*nk*hd f32 slots; f16 needs half — allocate
                // the max and write the right byte count per variant
                let kb = cb.alloc_buffer(rows * nk * hd);
                let vb = cb.alloc_buffer(rows * nk * hd);
                if f16_kv {
                    cb.write_host(kb, &pack_u16(&kv_f16)).unwrap();
                    cb.write_host(vb, &pack_u16(&kv_f16)).unwrap();
                } else {
                    cb.write_host(kb, &kv_f32).unwrap();
                    cb.write_host(vb, &kv_f32).unwrap();
                }
                let qb = cb.alloc_buffer(nt * nh * hd);
                cb.write_host(qb, &q).unwrap();
                let obt = cb.alloc_buffer(nt * nh * hd);
                let oseq = cb.alloc_buffer(nt * nh * hd);
                let post = cb.alloc_buffer(nt);
                let posv: Vec<f32> = (0..nt)
                    .map(|t| f32::from_bits((prefix as i32 + t as i32) as u32))
                    .collect();
                cb.write_host(post, &posv).unwrap();

                // batched verify: one call, positions [P, P+1, P+2]
                cb.state.gqa_attn_split_batched(
                    cb.ptr_of(qb).unwrap(),
                    cb.ptr_of(kb).unwrap(),
                    cb.ptr_of(vb).unwrap(),
                    cb.ptr_of(obt).unwrap(),
                    cb.ptr_of(post).unwrap(),
                    crate::cuda::AttnWindow::Causal.code(), // single-sequence
                    nh,
                    nk,
                    hd,
                    scale,
                    f16_kv,
                    nt,
                );

                // sequential decode: three nt=1 calls at the same positions
                let qrow = cb.alloc_buffer(nh * hd);
                let o1 = cb.alloc_buffer(nh * hd);
                let p1 = cb.alloc_buffer(1);
                for t in 0..nt {
                    cb.write_host(qrow, &q[t * nh * hd..(t + 1) * nh * hd])
                        .unwrap();
                    cb.write_host(p1, &[f32::from_bits((prefix as i32 + t as i32) as u32)])
                        .unwrap();
                    cb.state.gqa_attn_split(
                        cb.ptr_of(qrow).unwrap(),
                        cb.ptr_of(kb).unwrap(),
                        cb.ptr_of(vb).unwrap(),
                        cb.ptr_of(o1).unwrap(),
                        cb.ptr_of(p1).unwrap(),
                        crate::cuda::AttnWindow::Causal.code(), // single-sequence
                        nh,
                        nk,
                        hd,
                        scale,
                        if f16_kv {
                            crate::cuda::KV_LAYOUT_F16
                        } else {
                            crate::cuda::KV_LAYOUT_F32
                        },
                        if f16_kv { nk * hd * 2 } else { nk * hd * 4 },
                    );
                    let got = cb.copy_to_host(o1).unwrap();
                    let mut full = cb.copy_to_host(oseq).unwrap();
                    full[t * nh * hd..(t + 1) * nh * hd].copy_from_slice(&got);
                    cb.write_host(oseq, &full).unwrap();
                }

                let bt = cb.copy_to_host(obt).unwrap();
                let sq = cb.copy_to_host(oseq).unwrap();
                for (i, (a, b)) in bt.iter().zip(sq.iter()).enumerate() {
                    assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "nh={nh} nk={nk} hd={hd} prefix={prefix} kv_f16={f16_kv} elem {i}: batched {a} vs sequential {b}"
                );
                }
            }
        }
    } // nt sweep
}

#[test]
fn cuda_multi_token_matmul_bitwise() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    // doc 95: sweep the verify depths the adaptive controller may pick —
    // the identity needs multi-MMVQ bitwise-equal to single at every nt,
    // not just the historical nt=3 probe.
    for nt in [3usize, 5, 8] {
        fn gen_f32(n: usize, seed: u64) -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                    let mag = if (s >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                    (u as f32) * mag
                })
                .collect()
        }

        fn gen_bytes(n: usize, seed: u64) -> Vec<u8> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 33) as u8
                })
                .collect()
        }

        // (label, type, od, id, q6_padded). Weight-byte lengths per type:
        // K-quants ceil(id/256) blocks per row (144/176/210 B), the rest
        // id/32 blocks per row (18/20/22/24/34 B), f32 raw.
        let cases: Vec<(&str, TensorType, usize, usize, bool)> = vec![
            ("q4k_v2", TensorType::Q4_K, 2048, 3584, false),
            ("q4k_v1", TensorType::Q4_K, 512, 3904, false),
            ("q5k_v2", TensorType::Q5_K, 8192, 3072, false),
            ("q5k_v1", TensorType::Q5_K, 8192, 3104, false),
            ("q6k_padded", TensorType::Q6_K, 2048, 2048, true),
            ("q6k_raw", TensorType::Q6_K, 2048, 2048, false),
            ("q8_0", TensorType::Q8_0, 512, 2048, false),
            ("q4_0_big", TensorType::Q4_0, 512, 9216, false),
            // 14B decode/verify shapes (doc 94): the identity chain needs
            // single-MMVQ (nt=1) == multi-MMVQ (nt 2..8) bitwise at the real
            // Qwen2.5-14B q4_k_m dims, not just the small fixtures.
            ("q4k_14b_attn", TensorType::Q4_K, 5120, 5120, false),
            ("q4k_14b_gu", TensorType::Q4_K, 13824, 5120, false),
            ("q6k_14b_down", TensorType::Q6_K, 5120, 13824, false),
            ("q4_1", TensorType::Q4_1, 512, 2048, false),
            ("q5_0", TensorType::Q5_0, 512, 2048, false),
            ("q5_1", TensorType::Q5_1, 512, 2048, false),
            ("f32", TensorType::F32, 512, 2048, false),
        ];

        for (i, (label, tt, od, id_, padded)) in cases.into_iter().enumerate() {
            let nbe = (id_ + 255) / 256;
            let row_bytes = match tt {
                TensorType::Q4_K => nbe * 144,
                TensorType::Q5_K => nbe * 176,
                TensorType::Q6_K => nbe * 210,
                TensorType::Q8_0 => (id_ / 32) * 34,
                TensorType::Q4_0 => (id_ / 32) * 18,
                TensorType::Q4_1 => (id_ / 32) * 20,
                TensorType::Q5_0 => (id_ / 32) * 22,
                TensorType::Q5_1 => (id_ / 32) * 24,
                TensorType::F32 => id_ * 4,
                other => panic!("unexpected type {other:?}"),
            };
            let wb = gen_bytes(od * row_bytes, 0x5EED_0000 + i as u64);
            let xs = gen_f32(id_ * nt, 0xA11C_0000 + i as u64);

            let wt_name = format!("wbit{i}");
            let mut wt = Tensor::from_data(tt, &[id_ as i64, od as i64, 1, 1], wb.clone());
            wt.name = wt_name.clone();
            if tt == TensorType::Q6_K && padded {
                cb.state.register_weight_q6k_padded(&wt_name, &wb, od, id_);
            } else {
                cb.state.register_weight(&wt_name, &wb);
            }

            // batched: one nt = 3 forward
            let mut b = GraphBuilder::new();
            let x = b.input("x", [id_, nt, 1, 1], DType::F32);
            let m = b.matmul(x, &wt, None);
            b.output(m);
            let g = b.build();
            let xb = cb.alloc_buffer(id_ * nt);
            cb.write_host(xb, &xs).unwrap();
            let ob = cb.alloc_buffer(od * nt);
            cb.exec_ids(&g.nodes[m], &[xb], ob, None).unwrap();
            let got = cb.copy_to_host(ob).unwrap();

            // reference: nt separate nt = 1 forwards over the same weights
            let mut refs: Vec<Vec<f32>> = Vec::with_capacity(nt);
            for t in 0..nt {
                let mut b1 = GraphBuilder::new();
                let x1 = b1.input("x1", [id_, 1, 1, 1], DType::F32);
                let m1 = b1.matmul(x1, &wt, None);
                b1.output(m1);
                let g1 = b1.build();
                let xb1 = cb.alloc_buffer(id_);
                cb.write_host(xb1, &xs[t * id_..(t + 1) * id_]).unwrap();
                let ob1 = cb.alloc_buffer(od);
                cb.exec_ids(&g1.nodes[m1], &[xb1], ob1, None).unwrap();
                refs.push(cb.copy_to_host(ob1).unwrap());
            }

            for t in 0..nt {
                for (r, (a, bref)) in got[t * od..(t + 1) * od]
                    .iter()
                    .zip(refs[t].iter())
                    .enumerate()
                {
                    assert_eq!(
                        a.to_bits(),
                        bref.to_bits(),
                        "{label} token {t} row {r}: batched nt={nt} vs single nt=1 mismatch"
                    );
                }
            }
        }

        // ── 8c q4_0 × q8-GEMM arm (nt > 1, id <= 8192) — tolerance vs the
        // independent host reference (dequant + the same q8 activation
        // quantization the kernel applies), the cuda_kquant_matmul_parity
        // method. The in-block token loop does not change per-token math.
        {
            let (od, id_) = (512usize, 2048usize);
            let wb = gen_bytes(od * (id_ / 32) * 18, 0x5EED_00C0);
            let xs = gen_f32(id_ * nt, 0xA11C_00C0);
            let mut wt =
                Tensor::from_data(TensorType::Q4_0, &[id_ as i64, od as i64, 1, 1], wb.clone());
            wt.name = "w8c".to_string();
            cb.state.register_weight("w8c", &wb);

            let mut b = GraphBuilder::new();
            let x = b.input("x", [id_, nt, 1, 1], DType::F32);
            let m = b.matmul(x, &wt, None);
            b.output(m);
            let g = b.build();
            let xb = cb.alloc_buffer(id_ * nt);
            cb.write_host(xb, &xs).unwrap();
            let ob = cb.alloc_buffer(od * nt);
            cb.exec_ids(&g.nodes[m], &[xb], ob, None).unwrap();
            let got = cb.copy_to_host(ob).unwrap();

            // host reference: q4_0 dequant (val = (nib - 8) * d) dotted with
            // the q8-quantized activations
            let mut x8 = vec![0u8; nt * (id_ / 32) * 40];
            for t in 0..nt {
                for blk in 0..id_ / 32 {
                    let base = t * id_ + blk * 32;
                    let mut am = 0f32;
                    for j in 0..32 {
                        am = am.max(xs[base + j].abs());
                    }
                    let dd = am / 127.0;
                    let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                    let off = (t * (id_ / 32) + blk) * 40;
                    x8[off..off + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
                    for j in 0..32 {
                        let q = (xs[base + j] * di).round().clamp(-128.0, 127.0) as i8;
                        x8[off + 4 + j] = q as u8;
                    }
                }
            }
            let dq8 = |t: usize, i: usize| -> f32 {
                let off = (t * (id_ / 32) + i / 32) * 40;
                half::f16::from_le_bytes([x8[off], x8[off + 1]]).to_f32()
                    * (x8[off + 4 + (i % 32)] as i8) as f32
            };
            // The per-block contraction order mirrors the kernel (d applied
            // per block); compare with the standard q8 tolerance.
            let mut want = vec![0f32; od * nt];
            let mut scale = 1e-9f32;
            for t in 0..nt {
                for r in 0..od {
                    let mut acc = 0f32;
                    for blk in 0..id_ / 32 {
                        let blkb = &wb[(r * (id_ / 32) + blk) * 18..];
                        let d = half::f16::from_le_bytes([blkb[0], blkb[1]]).to_f32();
                        let mut sdot = 0f32;
                        for j in 0..16 {
                            let b0 = blkb[2 + j];
                            sdot += ((b0 & 0x0F) as f32 - 8.0) * dq8(t, blk * 32 + j)
                                + ((b0 >> 4) as f32 - 8.0) * dq8(t, blk * 32 + 16 + j);
                        }
                        acc += d * sdot;
                    }
                    want[t * od + r] = acc;
                    scale = scale.max(acc.abs());
                }
            }
            assert_close("q4_0 8c multi-token", &got, &want, scale * 1e-2);
        }
    } // nt sweep
}

#[test]
fn cuda_fused_ffn_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (nf, id_) = (8usize, 512usize); // concat od = 16, decode nt = 1
    let xs: Vec<f32> = (0..id_)
        .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
        .collect();

    // ── q4_K gate/up weights (144-byte super-blocks, llama layout) ──
    // get_scale_min_k4 (llama.cpp Q4_K scale packing): the second half
    // of the 8 scale/min pairs is spliced across the 12 scale bytes.
    fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }
    fn build_q4k(seed: u64, rows: usize, id: usize, bytes: &mut Vec<u8>, dq: &mut Vec<f32>) {
        for r in 0..rows {
            for ib in 0..id / 256 {
                let d = 0.031f32 + 0.005 * ((seed + (r * 7 + ib * 3) as u64) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((seed + (r * 3 + ib) as u64) % 4) as f32;
                bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                bytes.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                let mut scb = [0u8; 12];
                for j in 0..12 {
                    scb[j] = ((seed as usize + r * 31 + j * 17 + ib * 5) % 63) as u8;
                }
                bytes.extend_from_slice(&scb);
                let mut qs = [0u8; 128];
                for j in 0..128 {
                    let lo = ((seed as usize + r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((seed as usize + r * 5 + j * 11 + ib * 2) % 15) as u8;
                    qs[j] = lo | (hi << 4);
                }
                bytes.extend_from_slice(&qs);
                for j in 0..4 {
                    let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                    let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                    for l in 0..32 {
                        let b = qs[j * 32 + l];
                        let base = r * id + ib * 256 + j * 64;
                        dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                        dq[base + 32 + l] = (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                    }
                }
            }
        }
    }

    // ── q6_K gate/up weights (210-byte super-blocks, llama layout) ──
    fn build_q6k(seed: u64, rows: usize, id: usize, bytes: &mut Vec<u8>, dq: &mut Vec<f32>) {
        for r in 0..rows {
            for ib in 0..id / 256 {
                let d = 0.027f32 + 0.004 * ((seed + (r * 11 + ib * 7) as u64) % 6) as f32;
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                let mut sc = [0i8; 16];
                for i in 0..128 {
                    ql[i] = ((seed as usize + r * 29 + i * 7 + ib * 3) % 255) as u8;
                }
                for i in 0..64 {
                    qh[i] = ((seed as usize + r * 17 + i * 13 + ib * 11) % 255) as u8;
                }
                for i in 0..16 {
                    sc[i] = (((seed as usize + r * 5 + i * 3 + ib) % 15) as i8) - 7;
                }
                bytes.extend_from_slice(&ql);
                bytes.extend_from_slice(&qh);
                bytes.extend(sc.iter().map(|&x| x as u8));
                bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                // reference dequant (llama.cpp Q6_K layout, same as
                // the kquant test): four interleaved 32-element groups
                // per 128-element half.
                for n in 0..2usize {
                    let qlh = &ql[n * 64..n * 64 + 64];
                    let qhh = &qh[n * 32..n * 32 + 32];
                    for l in 0..32usize {
                        let is = l / 16;
                        let q1 = ((qlh[l] & 0xF) as i32 | (((qhh[l] >> 0) as i32 & 3) << 4)) - 32;
                        let q2 =
                            ((qlh[l + 32] & 0xF) as i32 | (((qhh[l] >> 2) as i32 & 3) << 4)) - 32;
                        let q3 = ((qlh[l] >> 4) as i32 | (((qhh[l] >> 4) as i32 & 3) << 4)) - 32;
                        let q4 =
                            ((qlh[l + 32] >> 4) as i32 | (((qhh[l] >> 6) as i32 & 3) << 4)) - 32;
                        let base = r * id + ib * 256 + n * 128;
                        dq[base + l] = d * sc[n * 8 + is] as f32 * q1 as f32;
                        dq[base + l + 32] = d * sc[n * 8 + is + 2] as f32 * q2 as f32;
                        dq[base + l + 64] = d * sc[n * 8 + is + 4] as f32 * q3 as f32;
                        dq[base + l + 96] = d * sc[n * 8 + is + 6] as f32 * q4 as f32;
                    }
                }
            }
        }
    }

    let mut g4b = Vec::new();
    let mut g4dq = vec![0f32; nf * id_];
    build_q4k(1, nf, id_, &mut g4b, &mut g4dq);
    let mut u4b = Vec::new();
    let mut u4dq = vec![0f32; nf * id_];
    build_q4k(101, nf, id_, &mut u4b, &mut u4dq);
    let mut g6b = Vec::new();
    let mut g6dq = vec![0f32; nf * id_];
    build_q6k(7, nf, id_, &mut g6b, &mut g6dq);
    let mut u6b = Vec::new();
    let mut u6dq = vec![0f32; nf * id_];
    build_q6k(207, nf, id_, &mut u6b, &mut u6dq);

    // concat rows: gate rows then up rows (concat_rows semantics)
    let gu4: Vec<u8> = g4b.iter().chain(u4b.iter()).copied().collect();
    let gu6: Vec<u8> = g6b.iter().chain(u6b.iter()).copied().collect();
    cb.state.register_weight("mgu4", &gu4);
    // q6_K concat goes through the padded repack (7e② layout)
    cb.state
        .register_weight_q6k_padded("mgu6", &gu6, 2 * nf, id_);
    assert!(cb.state.is_weight_padded("mgu6"));

    let (xb, ogu4, ogu6) = (
        cb.alloc_buffer(id_),
        cb.alloc_buffer(2 * nf),
        cb.alloc_buffer(2 * nf),
    );
    cb.write_host(xb, &xs).unwrap();

    for (ttype, wname, ogu, gdq, udq) in [
        (crate::tensor::TensorType::Q4_K, "mgu4", ogu4, &g4dq, &u4dq),
        (crate::tensor::TensorType::Q6_K, "mgu6", ogu6, &g6dq, &u6dq),
    ] {
        let mut b = crate::graph::builder::GraphBuilder::new();
        let x = b.input("x", [id_, 1, 1, 1], crate::graph::DType::F32);
        let gu = b.fused_ffn(
            x,
            crate::graph::ops::FusedFfnMeta {
                gu_weight: wname.to_string(),
                weight_ttype: ttype,
                in_dim: id_,
                nf,
            },
        );
        b.output(gu);
        let g = b.build();
        cb.exec_ids(&g.nodes[gu], &[xb], ogu, None).unwrap();

        // host reference: silu(gate·x) × (up·x)
        let got = cb.copy_to_host(ogu).unwrap();
        let mut want = vec![0f32; nf];
        for r in 0..nf {
            let mut ag = 0f32;
            let mut au = 0f32;
            for i in 0..id_ {
                ag += gdq[r * id_ + i] * xs[i];
                au += udq[r * id_ + i] * xs[i];
            }
            let s = ag / (1.0f32 + (-ag).exp());
            want[r] = s * au;
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(
            &format!("{ttype:?} fused ffn"),
            &got[..nf],
            &want,
            scale * 1e-3,
        );
    }
}

/// 7e③: embedding / row-gather parity. Device embed kernels (one per
/// supported weight type, incl. the padded Q6_K layout) must match
/// `kernel::embed_tokens` — the CPU path these nodes used before 7e③ —
/// and the generic f32 gather (G3 tail get_rows) must match a manual
/// row copy.
#[test]
fn cuda_embed_getrows_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (vocab, n_embd, nt) = (6usize, 512usize, 3usize); // 2 super-blocks/row
    let ids: Vec<u32> = vec![0, 5, 2];
    let ids_f32: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();

    // ── build one tensor per supported type (rows = vocab) ──
    // f32
    let wf: Vec<f32> = (0..vocab * n_embd)
        .map(|i| (((i as u64) * 2654435761 % 1009) as f32) / 504.0 - 1.0)
        .collect();
    let wf_bytes: Vec<u8> = wf.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tf = Tensor::from_data(
        TensorType::F32,
        &[n_embd as i64, vocab as i64, 1, 1],
        wf_bytes.clone(),
    );
    tf.name = "ewf32".to_string();

    // q8_0 (34B blocks: f16 d + 32 i8)
    let mut w8 = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 32 {
            let d = 0.02f32 + 0.003 * ((r * 5 + ib) % 7) as f32;
            w8.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for i in 0..32 {
                w8.push((((r * 37 + ib * 17 + i * 3) % 255) as i8 as u8).wrapping_add(0));
            }
        }
    }
    let mut t8 = Tensor::from_data(
        TensorType::Q8_0,
        &[n_embd as i64, vocab as i64, 1, 1],
        w8.clone(),
    );
    t8.name = "ewq8".to_string();

    // q4_0 (18B blocks: f16 d + 16 nibble bytes; elem j = LOW of byte j)
    let mut w40 = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 32 {
            let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
            w40.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for i in 0..16 {
                let lo = ((r * 11 + ib * 7 + i * 3) % 15) as u8;
                let hi = ((r * 7 + ib * 5 + i) % 15) as u8;
                w40.push(lo | (hi << 4));
            }
        }
    }
    let mut t40 = Tensor::from_data(
        TensorType::Q4_0,
        &[n_embd as i64, vocab as i64, 1, 1],
        w40.clone(),
    );
    t40.name = "ewq40".to_string();

    // q5_0 (22B blocks: f16 d + u32 qh + 16 nibble bytes; value =
    // nibble + 16*high_bit - 16) — the tok_embd type of 0.5B q4_k_m GGUFs
    let mut w50 = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 32 {
            let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
            w50.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            let qh: u32 = (((r * 13 + ib * 7) % 5) as u32) << 17
                | (((r * 5 + ib * 3) % 7) as u32) << 3
                | 0b101;
            w50.extend_from_slice(&qh.to_le_bytes());
            for i in 0..16 {
                let lo = ((r * 11 + ib * 7 + i * 3) % 31) as u8;
                let hi = ((r * 7 + ib * 5 + i) % 31) as u8;
                w50.push(lo | (hi << 4));
            }
        }
    }
    let mut t50 = Tensor::from_data(
        TensorType::Q5_0,
        &[n_embd as i64, vocab as i64, 1, 1],
        w50.clone(),
    );
    t50.name = "ewq50".to_string();

    // q4_k (144B super-blocks) — same generator scheme as the matmul test
    let mut w4k = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 256 {
            let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
            let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
            w4k.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            w4k.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            for j in 0..12 {
                w4k.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
            }
            for j in 0..128 {
                let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                w4k.push(lo | (hi << 4));
            }
        }
    }
    let mut t4k = Tensor::from_data(
        TensorType::Q4_K,
        &[n_embd as i64, vocab as i64, 1, 1],
        w4k.clone(),
    );
    t4k.name = "ewq4k".to_string();

    // q6_k (210B raw; also registered padded)
    let mut w6k = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 256 {
            let d = 0.027f32 + 0.004 * ((r * 11 + ib * 7) % 6) as f32;
            for i in 0..128 {
                w6k.push(((r * 29 + i * 7 + ib * 3) % 255) as u8);
            }
            for i in 0..64 {
                w6k.push(((r * 17 + i * 13 + ib * 11) % 255) as u8);
            }
            for i in 0..16 {
                w6k.push(((((r * 5 + i * 3 + ib) % 15) as i8) - 7) as u8);
            }
            w6k.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        }
    }
    let mut t6k = Tensor::from_data(
        TensorType::Q6_K,
        &[n_embd as i64, vocab as i64, 1, 1],
        w6k.clone(),
    );
    t6k.name = "ewq6k".to_string();
    let mut t6kp = Tensor::from_data(
        TensorType::Q6_K,
        &[n_embd as i64, vocab as i64, 1, 1],
        w6k.clone(),
    );
    t6kp.name = "ewq6kp".to_string();

    // ── register + build one graph with an embed node per type ──
    cb.state.register_weight("ewf32", &wf_bytes);
    cb.state.register_weight("ewq8", &w8);
    cb.state.register_weight("ewq40", &w40);
    cb.state.register_weight("ewq50", &w50);
    cb.state.register_weight("ewq4k", &w4k);
    cb.state.register_weight("ewq6k", &w6k);
    cb.state
        .register_weight_q6k_padded("ewq6kp", &w6k, vocab, n_embd);
    assert!(cb.state.is_weight_padded("ewq6kp"));

    let mut b = GraphBuilder::new();
    let ids_in = b.input("ids", [nt, 1, 1, 1], DType::F32);
    let e_f32 = b.embedding(ids_in, &tf);
    let e_q8 = b.embedding(ids_in, &t8);
    let e_q40 = b.embedding(ids_in, &t40);
    let e_q50 = b.embedding(ids_in, &t50);
    let e_q4k = b.embedding(ids_in, &t4k);
    let e_q6k = b.embedding(ids_in, &t6k);
    let e_q6kp = b.embedding(ids_in, &t6kp);
    // ── 7e③ model-shape q4_0 case (0.5B): n_embd=896 (nb=28 blocks),
    // large ids — the exact shape that E2E first exercised ──
    let (mv, me) = (10000usize, 896usize);
    let mids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let mut mw = Vec::new();
    for r in 0..mv {
        for ib in 0..me / 32 {
            let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
            mw.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for i in 0..16 {
                let lo = ((r * 11 + ib * 7 + i * 3) % 15) as u8;
                let hi = ((r * 7 + ib * 5 + i) % 15) as u8;
                mw.push(lo | (hi << 4));
            }
        }
    }
    let mut mt = Tensor::from_data(TensorType::Q4_0, &[me as i64, mv as i64, 1, 1], mw.clone());
    mt.name = "ewq40m".to_string();
    cb.state.register_weight("ewq40m", &mw);
    let mids_f32: Vec<f32> = mids.iter().map(|&i| f32::from_bits(i)).collect();
    let midb = cb.alloc_buffer(mids.len());
    cb.write_host(midb, &mids_f32).unwrap();
    let mout = cb.alloc_buffer(me * mids.len());
    let mut mb = GraphBuilder::new();
    let mi = mb.input("mids", [mids.len(), 1, 1, 1], DType::F32);
    let me_node = mb.embedding(mi, &mt);
    mb.output(me_node);
    let mg = mb.build();
    cb.exec_ids(&mg.nodes[me_node], &[midb], mout, None)
        .unwrap();
    {
        let got = cb.copy_to_host(mout).unwrap();
        let mut want = vec![0f32; me * mids.len()];
        crate::kernel::embed_tokens(&mids, &mt, &mut want, me);
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("embed q4_0 model-shape", &got, &want, scale * 2e-3);
    }

    // generic gather (G3 tail): x[ids[t]] — the source has vocab rows so
    // every id is in range
    let xin = b.input("x", [n_embd, vocab, 1, 1], DType::F32);
    let gr = b.get_rows(xin, ids_in, [n_embd, nt, 1, 1]);
    for n in [e_f32, e_q8, e_q40, e_q50, e_q4k, e_q6k, e_q6kp, gr] {
        b.output(n);
    }
    let g = b.build();

    let idsb = cb.alloc_buffer(nt);
    cb.write_host(idsb, &ids_f32).unwrap();
    let xvals: Vec<f32> = (0..n_embd * vocab)
        .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
        .collect();
    let xb = cb.alloc_buffer(n_embd * vocab);
    cb.write_host(xb, &xvals).unwrap();

    let mut outs = Vec::new();
    for node in [e_f32, e_q8, e_q40, e_q50, e_q4k, e_q6k, e_q6kp] {
        let out = cb.alloc_buffer(n_embd * nt);
        cb.exec_ids(&g.nodes[node], &[idsb], out, None).unwrap();
        outs.push(out);
    }
    let grb = cb.alloc_buffer(n_embd * nt);
    cb.exec_ids(&g.nodes[gr], &[xb, idsb], grb, None).unwrap();

    // ── references ──
    let names = ["f32", "q8_0", "q4_0", "q5_0", "q4_k", "q6_k", "q6_k padded"];
    let tensors = [&tf, &t8, &t40, &t50, &t4k, &t6k, &t6kp];
    for ((name, t), &ob) in names.iter().zip(tensors).zip(outs.iter()) {
        let got = cb.copy_to_host(ob).unwrap();
        let mut want = vec![0f32; n_embd * nt];
        if t.ttype == TensorType::F32 {
            // embed_tokens handles the quantized types; f32 is a row copy
            for (ti, &id) in ids.iter().enumerate() {
                let src = (id as usize) * n_embd;
                want[ti * n_embd..(ti + 1) * n_embd].copy_from_slice(&wf[src..src + n_embd]);
            }
        } else {
            crate::kernel::embed_tokens(&ids, t, &mut want, n_embd);
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(&format!("embed {name}"), &got, &want, scale * 2e-3);
    }
    let got = cb.copy_to_host(grb).unwrap();
    for t in 0..nt {
        let id = ids[t] as usize;
        for i in 0..n_embd {
            let want = xvals[id * n_embd + i];
            assert!(
                (got[t * n_embd + i] - want).abs() <= 1e-6 * (1.0 + want.abs()),
                "gather [{t},{i}]: got {} want {want}",
                got[t * n_embd + i]
            );
        }
    }
}

#[test]
fn cuda_rope_kv_attn_roundtrip() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
    let nkt = nk_h * hd;
    let (nt, n_ctx) = (3usize, 32usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = vec![1, 4, 9]; // sparse, exercises the scatter

    let mut b = GraphBuilder::new();
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let p = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let qr = b.rope(
        q,
        p,
        RopeStyle::NonInterleaved,
        RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: nh,
            hd,
        },
    );
    let at = b.attn(
        qr,
        load,
        p,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk_h,
            hd,
            hd_kv: hd,
            nkt,
            scale,
        },
    );
    b.output(at);
    let g = b.build();

    let (xb_q, xb_k, xb_v) = (
        cb.alloc_buffer(nh * hd * nt),
        cb.alloc_buffer(nkt * nt),
        cb.alloc_buffer(nkt * nt),
    );
    let xb_p = cb.alloc_buffer(nt);
    let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
    let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

    let qs: Vec<f32> = (0..nh * hd * nt)
        .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
        .collect();
    let ks: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let vs: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    let pb: Vec<f32> = pos.iter().map(|&pp| f32::from_bits(pp as u32)).collect();
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    // Zero the KV regions first: rows the store never touches stay
    // uninitialized in a recycled cudaMalloc block, and the reference
    // below treats unwritten rows as zeros (deterministic vs pool state).
    cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
    cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
        .unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kreg, xb_p],
        ob_at,
        Some((kreg, vreg)),
    )
    .unwrap();

    // a) stored K rows are bit-exact at the scattered positions
    let kback = cb.copy_to_host(kreg).unwrap();
    for (t, &pp) in pos.iter().enumerate() {
        assert_eq!(
            &kback[pp * nkt..(pp + 1) * nkt],
            &ks[t * nkt..(t + 1) * nkt],
            "K row {pp}"
        );
    }
    // b) RoPE vs cpu_rope (also covers the non-alias D2D staging path)
    let qgot = cb.copy_to_host(ob_qr).unwrap();
    let mut qref = qs.clone();
    crate::graph::cpu_backend::cpu_rope(
        &mut qref,
        &pos,
        nh,
        hd,
        10000.0,
        1.0,
        RopeStyle::NonInterleaved,
    );
    assert_close("rope", &qgot, &qref, 1e-4);
    // c) GQA attention vs cpu_gqa_attn over the scattered KV regions
    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for (t, &pp) in pos.iter().enumerate() {
        kfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&ks[t * nkt..(t + 1) * nkt]);
        vfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&vs[t * nkt..(t + 1) * nkt]);
    }
    // E1: the CPU reference takes the allowed-cell span; this is the
    // single-sequence causal window the test compares against.
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qref, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    assert_close("gqa_attn", &agot, &aref, 1e-4);
}

// 8b: f16 KV cache — the store rounds K/V to half and the attention
// kernel reads half4. The reference builds its KV from the SAME
// half-rounded values so the comparison isolates the kernel from the
// f16 quantization noise (tolerance stays tight).
/// C3's copy primitive on the device: the rows land where the plan says,
/// *including when source and destination overlap* — the case a bulk
/// device-to-device copy cannot express (CUDA documents overlapping
/// `cudaMemcpyAsync` as undefined).
#[test]
fn cuda_copy_cells_moves_overlapping_rows_in_both_directions() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    // 6 rows x 4 elements; a row's value identifies the row it came from.
    let id = cb.alloc_buffer(24);
    let data: Vec<f32> = (0..24)
        .map(|i| (i / 4) as f32 + (i % 4) as f32 / 10.0)
        .collect();
    cb.write_host(id, &data).unwrap();
    let r = BufRef::own(crate::graph::Backend::CUDA, id, 24);
    // Rows [1, 4) -> rows [0, 3): rows 1 and 2 are both read and overwritten.
    cb.copy_cells(r, r, 0, 1, 3, 4).unwrap();
    let got = cb.copy_to_host(id).unwrap();
    let want: Vec<f32> = (0..24)
        .map(|i| {
            let row = if i / 4 < 3 { i / 4 + 1 } else { i / 4 };
            row as f32 + (i % 4) as f32 / 10.0
        })
        .collect();
    assert_eq!(got, want, "the moved rows must be byte-identical");
    // C7b: the upward direction works too. The kernel walks the rows in the
    // order the overlap requires — descending here — so the result is what a
    // copy through a temporary would give.
    cb.write_host(id, &data).unwrap();
    cb.copy_cells(r, r, 1, 0, 3, 4).unwrap();
    let got = cb.copy_to_host(id).unwrap();
    let want: Vec<f32> = (0..24)
        .map(|i| {
            let row = match i / 4 {
                0 => 0,
                n if n <= 3 => n - 1,
                n => n,
            };
            row as f32 + (i % 4) as f32 / 10.0
        })
        .collect();
    assert_eq!(
        got, want,
        "an upward overlapping move must be byte-identical"
    );
}

#[test]
fn cuda_kv_f16_roundtrip_attn() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    cb.set_kv_f16_for_test(true);
    let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
    let nkt = nk_h * hd;
    let (nt, n_ctx) = (3usize, 32usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = vec![1, 4, 9];

    let mut b = GraphBuilder::new();
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let qr = b.rope(
        q,
        pp,
        RopeStyle::NonInterleaved,
        RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: nh,
            hd,
        },
    );
    let at = b.attn(
        qr,
        load,
        pp,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk_h,
            hd,
            hd_kv: hd,
            nkt,
            scale,
        },
    );
    b.output(at);
    let g = b.build();

    let (xb_q, xb_k, xb_v) = (
        cb.alloc_buffer(nh * hd * nt),
        cb.alloc_buffer(nkt * nt),
        cb.alloc_buffer(nkt * nt),
    );
    let xb_p = cb.alloc_buffer(nt);
    let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
    let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

    let qs: Vec<f32> = (0..nh * hd * nt)
        .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
        .collect();
    let ks: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let vs: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
    // the reference KV: what the f16 store actually persists (f32→f16→f32)
    let to_half =
        |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect() };
    let ks_h = to_half(&ks);
    let vs_h = to_half(&vs);
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    // zero the regions (unwritten rows read as f16 zeros)
    cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
    cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
        .unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kreg, xb_p],
        ob_at,
        Some((kreg, vreg)),
    )
    .unwrap();

    // a) stored K rows equal the half-rounded values at the scatter positions
    let kback_f32 = cb.copy_to_host(kreg).unwrap();
    // reinterpret the region as f16 pairs (store wrote 2 bytes/elem)
    let kbytes: Vec<u8> = kback_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    for (t, &p) in pos.iter().enumerate() {
        for j in 0..nkt {
            let byte_off = (p * nkt + j) * 2;
            let got = half::f16::from_le_bytes([kbytes[byte_off], kbytes[byte_off + 1]]);
            assert!(
                (got.to_f32() - ks_h[t * nkt + j]).abs() < 1e-6,
                "f16 K row {p}[{j}]"
            );
        }
    }
    // b) attention vs cpu_gqa_attn over the half-rounded KV
    let qgot = cb.copy_to_host(ob_qr).unwrap();
    let mut qref = qs.clone();
    crate::graph::cpu_backend::cpu_rope(
        &mut qref,
        &pos,
        nh,
        hd,
        10000.0,
        1.0,
        RopeStyle::NonInterleaved,
    );
    assert_close("rope(f16 kv)", &qgot, &qref, 1e-4);
    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for (t, &p) in pos.iter().enumerate() {
        kfull[p * nkt..(p + 1) * nkt].copy_from_slice(&ks_h[t * nkt..(t + 1) * nkt]);
        vfull[p * nkt..(p + 1) * nkt].copy_from_slice(&vs_h[t * nkt..(t + 1) * nkt]);
    }
    // E1: the CPU reference takes the allowed-cell span; this is the
    // single-sequence causal window the test compares against.
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qref, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    assert_close("gqa_attn(f16 kv)", &agot, &aref, 1e-4);
}

// 8n: prefill attention (nt >= 64, hd == 128) routes through the
// FA-style tiled kernel (wmma QK^T, online softmax, per-thread register
// O accumulator). Reference: cpu_gqa_attn over the f16-rounded KV — the
// kernel reads the same f16 cache; its q and probs carry f16 rounding,
// measured ~1.4e-4 on the standalone harness, so 5e-3 leaves headroom.
#[test]
fn cuda_prefill_fused_b_bitparity() {
    // 8p: the fused dequant-in-GEMM path must be BIT-identical to the
    // legacy dequant-to-f16 two-pass path (same __float2half rounding,
    // same wmma accumulate). All 8 types x {1, 2} super-blocks; the
    // legacy path is reference-validated by cuda_prefill_f16_gemm_parity.
    let _guard = crate::cuda::CudaState::model_load_guard();
    crate::cuda::CudaState::init();
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let state = cb.state;
    let mut seed = 0x9E3779B9u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for (od, id, nt) in [(70usize, 256usize, 33usize), (70usize, 512usize, 70usize)] {
        let nsp = id / 256;
        let xs: Vec<f32> = (0..id * nt)
            .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let mut mk =
            |nbytes: usize| -> Vec<u8> { (0..nbytes).map(|_| (rnd() & 0xFF) as u8).collect() };
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        let (xptr, optr) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out).unwrap());
        let dbytes = |v: f32| half::f16::from_f32(v).to_le_bytes();

        // benign d (and m for the min-carrying types) per block
        let mut wq80 = mk(od * (id / 32) * 34);
        let mut wq40 = mk(od * (id / 32) * 18);
        let mut wq41 = mk(od * (id / 32) * 20);
        let mut wq50 = mk(od * (id / 32) * 22);
        let mut wq51 = mk(od * (id / 32) * 24);
        for g in 0..od * (id / 32) {
            let set = |w: &mut [u8], base: usize, off: usize, v: f32| {
                let db = dbytes(v);
                w[base + off] = db[0];
                w[base + off + 1] = db[1];
            };
            let b32 = g * 34;
            set(&mut wq80, b32, 0, 0.01);
            let b18 = g * 18;
            set(&mut wq40, b18, 0, 0.05);
            let b20 = g * 20;
            set(&mut wq41, b20, 0, 0.05);
            set(&mut wq41, b20, 2, 0.1);
            let b22 = g * 22;
            set(&mut wq50, b22, 0, 0.05);
            let b24 = g * 24;
            set(&mut wq51, b24, 0, 0.05);
            set(&mut wq51, b24, 2, 0.1);
        }
        let mut wq4k = mk(od * nsp * 144);
        let mut wq5k = mk(od * nsp * 176);
        let mut wq6k = mk(od * nsp * 210);
        for r in 0..od {
            for sp in 0..nsp {
                let base4 = (r * nsp + sp) * 144;
                wq4k[base4..base4 + 2].copy_from_slice(&dbytes(0.01));
                wq4k[base4 + 2..base4 + 4].copy_from_slice(&dbytes(0.005));
                let base5 = (r * nsp + sp) * 176;
                wq5k[base5..base5 + 2].copy_from_slice(&dbytes(0.01));
                wq5k[base5 + 2..base5 + 4].copy_from_slice(&dbytes(0.005));
                let base6 = (r * nsp + sp) * 210;
                wq6k[base6 + 208..base6 + 210].copy_from_slice(&dbytes(0.01));
            }
        }

        state.register_weight("bp_w80", &wq80);
        state.register_weight("bp_w40", &wq40);
        state.register_weight("bp_w41", &wq41);
        state.register_weight("bp_w50", &wq50);
        state.register_weight("bp_w51", &wq51);
        state.register_weight("bp_w4k", &wq4k);
        state.register_weight("bp_w5k", &wq5k);
        state.register_weight("bp_w6k_raw", &wq6k);
        state.register_weight_q6k_padded("bp_w6k_pad", &wq6k, od, id);

        let cases: [(TensorType, &str, bool); 9] = [
            (TensorType::Q8_0, "bp_w80", false),
            (TensorType::Q4_0, "bp_w40", false),
            (TensorType::Q4_1, "bp_w41", false),
            (TensorType::Q5_0, "bp_w50", false),
            (TensorType::Q5_1, "bp_w51", false),
            (TensorType::Q4_K, "bp_w4k", false),
            (TensorType::Q5_K, "bp_w5k", false),
            (TensorType::Q6_K, "bp_w6k_raw", false),
            (TensorType::Q6_K, "bp_w6k_pad", true),
        ];
        for (ttype, name, padded) in cases {
            let wptr = state.get_weight_ptr(name).unwrap();
            state
                .prefill_gemm_f16_inner(wptr, ttype, xptr, optr, od, id, nt, padded, true)
                .unwrap();
            cb.synchronize();
            let gotf = cb.copy_to_host(out).unwrap();
            state
                .prefill_gemm_f16_inner(wptr, ttype, xptr, optr, od, id, nt, padded, false)
                .unwrap();
            cb.synchronize();
            let gotl = cb.copy_to_host(out).unwrap();
            assert_eq!(gotf.len(), gotl.len(), "{name} len");
            for (i, (a, b)) in gotf.iter().zip(gotl.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} od={od} id={id} fused vs legacy bit mismatch at [{i}] ({a} vs {b})"
                );
            }
        }
    }
}

// R1 host helpers (module level: the reference fn below can't capture
// the test fn's locals)
fn mmq_f16v(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}
// llama.cpp get_scale_min_k4 (host mirror of the device helper)
fn mmq_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

// R1: the int8 MMQ prefill GEMM must reproduce the CPU q8_0-activation
// dot math (the structure llama.cpp's MMQ implements): int8×int8 dots
// are exact on both sides and the block scales are f16→f32 on both
// sides; only accumulation order differs, so 1e-3 absolute leaves
// orders of magnitude of headroom over f32 rounding while still failing
// loudly on any fragment-layout or unpacking mistake. All 8 types ×
// {odd tile edges, 2 super-blocks}; q6_K in both registered layouts.
#[test]
fn cuda_prefill_mmq_parity() {
    let _guard = crate::cuda::CudaState::model_load_guard();
    crate::cuda::CudaState::init();
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let state = cb.state;
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };

    // reference: CPU q8_0-activation dot math, per 32-block:
    //   out += da · (ds · Σ w_i·q_i + dm · Σ q_i)
    // q6_K carries 16-element sub-scales → two halves per 32-block.
    fn reference(
        ttype: TensorType,
        w: &[u8],
        x: &[f32],
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
    ) -> Vec<f32> {
        let nb = id / 32;
        let _ = padded_q6k; // the host reference always reads raw 210B rows
        let mut out = vec![0f32; nt * od];
        for t in 0..nt {
            let mut da = vec![0f32; nb];
            let mut q = vec![0i32; id];
            let mut sa = vec![0i64; nb];
            for b in 0..nb {
                let blk = &x[t * id + b * 32..t * id + b * 32 + 32];
                let am = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = am / 127.0;
                da[b] = half::f16::from_f32(d).to_f32(); // f16 rounding, as the GPU kernel stores it
                let di = if d != 0.0 { 1.0 / d } else { 0.0 };
                for (i, v) in blk.iter().enumerate() {
                    let qi = (*v * di).round_ties_even();
                    let qi = qi.clamp(-128.0, 127.0) as i32;
                    q[b * 32 + i] = qi;
                    sa[b] += qi as i64;
                }
            }
            for j in 0..od {
                let mut acc = 0f32;
                for b in 0..nb {
                    // (ds, dm, val(i)) per type for element i of block b
                    let mut ds = 0f32;
                    let mut dm = 0f32;
                    let mut dot = 0i64;
                    match ttype {
                        TensorType::Q8_0 => {
                            let blk = &w[(j * nb + b) * 34..][..34];
                            ds = mmq_f16v(blk);
                            for i in 0..32 {
                                dot += (blk[2 + i] as i8 as i64) * q[b * 32 + i] as i64;
                            }
                        }
                        TensorType::Q4_0 => {
                            let blk = &w[(j * nb + b) * 18..][..18];
                            ds = mmq_f16v(blk);
                            for i in 0..32 {
                                let byte = blk[2 + (i & 15)];
                                let nib = if i < 16 { byte & 0xF } else { byte >> 4 };
                                dot += (nib as i64 - 8) * q[b * 32 + i] as i64;
                            }
                        }
                        TensorType::Q4_1 => {
                            let blk = &w[(j * nb + b) * 20..][..20];
                            ds = mmq_f16v(blk);
                            dm = mmq_f16v(&blk[2..]);
                            for i in 0..32 {
                                let byte = blk[4 + (i & 15)];
                                let nib = if i < 16 { byte & 0xF } else { byte >> 4 };
                                dot += nib as i64 * q[b * 32 + i] as i64;
                            }
                        }
                        TensorType::Q5_0 => {
                            let blk = &w[(j * nb + b) * 22..][..22];
                            ds = mmq_f16v(blk);
                            let qh = blk[2] as u32
                                | ((blk[3] as u32) << 8)
                                | ((blk[4] as u32) << 16)
                                | ((blk[5] as u32) << 24);
                            for i in 0..32 {
                                let byte = blk[6 + (i & 15)];
                                let nib = if i < 16 { byte & 0xF } else { byte >> 4 };
                                let v = nib as i64 + 16 * ((qh >> i) & 1) as i64 - 16;
                                dot += v * q[b * 32 + i] as i64;
                            }
                        }
                        TensorType::Q5_1 => {
                            let blk = &w[(j * nb + b) * 24..][..24];
                            ds = mmq_f16v(blk);
                            dm = mmq_f16v(&blk[2..]);
                            let qh = blk[4] as u32
                                | ((blk[5] as u32) << 8)
                                | ((blk[6] as u32) << 16)
                                | ((blk[7] as u32) << 24);
                            for i in 0..32 {
                                let byte = blk[8 + (i & 15)];
                                let nib = if i < 16 { byte & 0xF } else { byte >> 4 };
                                let v = nib as i64 + 16 * ((qh >> i) & 1) as i64;
                                dot += v * q[b * 32 + i] as i64;
                            }
                        }
                        TensorType::Q4_K => {
                            let nsp = nb / 8;
                            let blk = &w[(j * nsp + b / 8) * 144..][..144];
                            let s = b % 8;
                            let (sc, m) = mmq_scale_min_k4(s, &blk[4..]);
                            ds = mmq_f16v(blk) * sc as f32;
                            dm = -(mmq_f16v(&blk[2..]) * m as f32);
                            for i in 0..32 {
                                let byte = blk[16 + (s / 2) * 32 + i];
                                let nib = if s % 2 == 0 { byte & 0xF } else { byte >> 4 };
                                dot += nib as i64 * q[b * 32 + i] as i64;
                            }
                        }
                        TensorType::Q5_K => {
                            let nsp = nb / 8;
                            let blk = &w[(j * nsp + b / 8) * 176..][..176];
                            let s = b % 8;
                            let (sc, m) = mmq_scale_min_k4(s, &blk[4..]);
                            ds = mmq_f16v(blk) * sc as f32;
                            dm = -(mmq_f16v(&blk[2..]) * m as f32);
                            for i in 0..32 {
                                let byte = blk[48 + (s / 2) * 32 + i];
                                let nib = if s % 2 == 0 { byte & 0xF } else { byte >> 4 };
                                let bit = (blk[16 + i] >> s) & 1;
                                dot += (nib as i64 + 16 * bit as i64) * q[b * 32 + i] as i64;
                            }
                        }
                        TensorType::Q6_K => {
                            let nsp = nb / 8;
                            // host bytes are the RAW 210B layout — the
                            // 224B padding only exists on the device
                            // (register_weight_q6k_padded repack)
                            let blk = &w[(j * nsp + b / 8) * 210..][..210];
                            // two 16-element sub-blocks per 32-block
                            for half in 0..2 {
                                let s = (b * 2 + half) % 16;
                                let sc = blk[192 + s] as i8 as f32;
                                let chunk = s / 8;
                                let g = (s / 2) % 4;
                                let is = s % 2;
                                let ql = chunk * 64 + (g % 2) * 32 + is * 16;
                                let qh = 128 + chunk * 32 + is * 16;
                                let mut hdot = 0i64;
                                for r in 0..16 {
                                    let byte = blk[ql + r];
                                    let nib = if g < 2 { byte & 0xF } else { byte >> 4 };
                                    let q2 = (blk[qh + r] >> (2 * g)) & 3;
                                    hdot += ((nib as i64) | ((q2 as i64) << 4) - 32)
                                        * q[b * 32 + half * 16 + r] as i64;
                                }
                                acc += da[b] * mmq_f16v(&blk[208..]) * sc * hdot as f32;
                            }
                            continue;
                        }
                        _ => unreachable!(),
                    }
                    acc += da[b] * (ds * dot as f32 + dm * sa[b] as f32);
                }
                out[t * od + j] = acc;
            }
        }
        out
    }

    // shape sweep: isolate which dimension (k depth / od tiles / token
    // tiles) breaks the kernel if any — small cases passed first
    for (od, id, nt) in [
        (70usize, 256usize, 33usize),
        (70usize, 512usize, 70usize),
        (70usize, 1024usize, 70usize),
        (70usize, 2048usize, 70usize),
        (70usize, 3584usize, 70usize),
        (3584usize, 512usize, 33usize),
        (3584usize, 3584usize, 70usize),
        (128usize, 512usize, 256usize),
    ] {
        let nsp = id / 256;
        let xs: Vec<f32> = (0..id * nt)
            .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let mut mk =
            |nbytes: usize| -> Vec<u8> { (0..nbytes).map(|_| (rnd() & 0xFF) as u8).collect() };
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        let (xptr, optr) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out).unwrap());
        let dbytes = |v: f32| half::f16::from_f32(v).to_le_bytes();

        // benign d (and m for the min-carrying types) per block; payload
        // nibbles/scales stay random bytes (any int8 value is legal)
        let mut wq80 = mk(od * (id / 32) * 34);
        let mut wq40 = mk(od * (id / 32) * 18);
        let mut wq41 = mk(od * (id / 32) * 20);
        let mut wq50 = mk(od * (id / 32) * 22);
        let mut wq51 = mk(od * (id / 32) * 24);
        for g in 0..od * (id / 32) {
            let set = |w: &mut [u8], base: usize, off: usize, v: f32| {
                let db = dbytes(v);
                w[base + off] = db[0];
                w[base + off + 1] = db[1];
            };
            set(&mut wq80, g * 34, 0, 0.01);
            set(&mut wq40, g * 18, 0, 0.05);
            set(&mut wq41, g * 20, 0, 0.05);
            set(&mut wq41, g * 20, 2, 0.1);
            set(&mut wq50, g * 22, 0, 0.05);
            set(&mut wq51, g * 24, 0, 0.05);
            set(&mut wq51, g * 24, 2, 0.1);
        }
        let mut wq4k = mk(od * nsp * 144);
        let mut wq5k = mk(od * nsp * 176);
        let mut wq6k = mk(od * nsp * 210);
        for r in 0..od {
            for sp in 0..nsp {
                let base4 = (r * nsp + sp) * 144;
                wq4k[base4..base4 + 2].copy_from_slice(&dbytes(0.01));
                wq4k[base4 + 2..base4 + 4].copy_from_slice(&dbytes(0.005));
                let base5 = (r * nsp + sp) * 176;
                wq5k[base5..base5 + 2].copy_from_slice(&dbytes(0.01));
                wq5k[base5 + 2..base5 + 4].copy_from_slice(&dbytes(0.005));
                let base6 = (r * nsp + sp) * 210;
                wq6k[base6 + 208..base6 + 210].copy_from_slice(&dbytes(0.01));
            }
        }

        state.register_weight("mmq_w80", &wq80);
        state.register_weight("mmq_w40", &wq40);
        state.register_weight("mmq_w41", &wq41);
        state.register_weight("mmq_w50", &wq50);
        state.register_weight("mmq_w51", &wq51);
        state.register_weight("mmq_w4k", &wq4k);
        state.register_weight("mmq_w5k", &wq5k);
        state.register_weight("mmq_w6k_raw", &wq6k);
        state.register_weight_q6k_padded("mmq_w6k_pad", &wq6k, od, id);

        let cases: [(TensorType, &str, bool); 9] = [
            (TensorType::Q8_0, "mmq_w80", false),
            (TensorType::Q4_0, "mmq_w40", false),
            (TensorType::Q4_1, "mmq_w41", false),
            (TensorType::Q5_0, "mmq_w50", false),
            (TensorType::Q5_1, "mmq_w51", false),
            (TensorType::Q4_K, "mmq_w4k", false),
            (TensorType::Q5_K, "mmq_w5k", false),
            (TensorType::Q6_K, "mmq_w6k_raw", false),
            (TensorType::Q6_K, "mmq_w6k_pad", true),
        ];
        for (ttype, name, padded) in cases {
            if state.cc() < 800 {
                eprintln!("skipping: mma.m16n8k32 s8 needs sm_80+ (cc {})", state.cc());
                return;
            }
            let wbytes: &[u8] = match name {
                "mmq_w80" => &wq80,
                "mmq_w40" => &wq40,
                "mmq_w41" => &wq41,
                "mmq_w50" => &wq50,
                "mmq_w51" => &wq51,
                "mmq_w4k" => &wq4k,
                "mmq_w5k" => &wq5k,
                // both layouts share the intra-block byte layout; the
                // padded variant only widens the row stride
                _ => &wq6k,
            };
            let wptr = state.get_weight_ptr(name).unwrap();
            state
                .prefill_mmq(wptr, ttype, xptr, optr, od, id, nt, padded, 1)
                .unwrap();
            cb.synchronize();
            let got = cb.copy_to_host(out).unwrap();
            let want = reference(ttype, wbytes, &xs, od, id, nt, padded);
            assert_close(name, &got, &want, 1e-3);
        }
    }
}

#[test]
fn cuda_q6k_exp_dense_byte_exact() {
    // r53 gate 1: the pre-expanded dense W_exp plane must be byte-identical
    // to an independent scalar mirror of the device expand_q6_elem over the
    // whole tensor (the r44 readback gate, 0 mismatches) — checked on the
    // HOST expander and on the DEVICE upload (pinned readback).
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for (od, id) in [
        (64usize, 256usize),
        (40usize, 512usize),
        (24usize, 768usize),
    ] {
        let nbe = id / 256;
        let row_len = nbe * 210;
        let raw: Vec<u8> = (0..od * row_len).map(|_| (rnd() & 0xFF) as u8).collect();
        // padded repack (the register_weight_q6k_padded layout)
        let mut padded = vec![0u8; od * nbe * 224];
        for r in 0..od {
            for ib in 0..nbe {
                let src = r * row_len + ib * 210;
                let dst = r * nbe * 224 + ib * 224;
                padded[dst..dst + 210].copy_from_slice(&raw[src..src + 210]);
            }
        }
        // independent scalar mirror, straight from the device formula
        let mut want = vec![0u8; od * id];
        for j in 0..od {
            for sb in 0..nbe {
                let base = (j * nbe + sb) * 224;
                let blk = &padded[base..base + 210];
                let (ql, rest) = blk.split_at(128);
                let qh = &rest[..64];
                for e in 0..256usize {
                    let m = e & 31;
                    let it = e >> 7;
                    let n = e & 127;
                    let ql_idx = it * 64 + (n & 63);
                    let ql_shift = (n >> 6) * 4;
                    let qh_idx = it * 32 + m;
                    let qh_shift = ((n >> 5) & 3) * 2;
                    let v = ((ql[ql_idx] >> ql_shift) & 0x0F)
                        | (((qh[qh_idx] >> qh_shift) & 0x03) << 4);
                    want[j * id + sb * 256 + e] = (v as i32 - 32) as u8;
                }
            }
        }
        // host-side production expander vs the mirror
        let host = crate::cuda::CudaState::expand_q6k_dense(&padded, od, id);
        let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(hmis, 0, "expand_q6k_dense vs mirror ({od}x{id})");
        // device upload path: build + read back + compare
        let name = format!("r53exp{od}x{id}");
        state.register_weight_q6k_padded(&name, &raw, od, id);
        state.register_weight_q6k_exp(&name, &padded, od, id);
        let exp_name = format!("{name}__exp{od}x{id}");
        let p = state.get_weight_ptr(&exp_name).expect("W_exp registered");
        let mut got = vec![0u8; od * id];
        state.copy_from_device_pinned(p, &mut got);
        let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(dmis, 0, "device W_exp vs mirror ({od}x{id})");
    }
}

#[test]
fn cuda_q6k_dsc_dense_byte_exact() {
    // r56 (Session E item 2b) gate 1: the precomputed dsc f32-pair plane
    // must be byte-identical to an independent scalar mirror of the
    // kernel's in-loop dsc computation (d = f16(blk+208); dsc = d *
    // (int8)blk[192 + 2*(c&7) + {0,1}]) — checked on the HOST expander and
    // on the DEVICE upload (pinned readback), over shapes covering several
    // super-blocks per row and od values that exercise the chunk-major
    // [c*od + j] layout.
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut s: u64 = 0xC0FF_EE12_3456_789A;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for (od, id) in [
        (64usize, 256usize),
        (40usize, 512usize),
        (24usize, 768usize),
    ] {
        let nbe = id / 256;
        let nchunk = id / 32;
        let row_len = nbe * 210;
        let raw: Vec<u8> = (0..od * row_len).map(|_| (rnd() & 0xFF) as u8).collect();
        let mut padded = vec![0u8; od * nbe * 224];
        for r in 0..od {
            for ib in 0..nbe {
                let src = r * row_len + ib * 210;
                let dst = r * nbe * 224 + ib * 224;
                padded[dst..dst + 210].copy_from_slice(&raw[src..src + 210]);
            }
        }
        // independent scalar mirror straight from the kernel formula
        let mut want = vec![0u8; nchunk * od * 8];
        for j in 0..od {
            for sb in 0..nbe {
                let base = (j * nbe + sb) * 224;
                let blk = &padded[base..base + 210];
                let d = half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();
                for cc in 0..8usize {
                    let sc0 = blk[192 + 2 * cc] as i8 as f32;
                    let sc1 = blk[192 + 2 * cc + 1] as i8 as f32;
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    want[idx..idx + 4].copy_from_slice(&(d * sc0).to_bits().to_le_bytes());
                    want[idx + 4..idx + 8].copy_from_slice(&(d * sc1).to_bits().to_le_bytes());
                }
            }
        }
        // host-side production expander vs the mirror
        let host = crate::cuda::CudaState::expand_q6k_dsc(&padded, od, id);
        let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(hmis, 0, "expand_q6k_dsc vs mirror ({od}x{id})");
        // device upload path: build + read back + compare
        let name = format!("r56dsc{od}x{id}");
        state.register_weight_q6k_padded(&name, &raw, od, id);
        state.register_weight_q6k_dsc(&name, &padded, od, id);
        let dsc_name = format!("{name}__dsc{od}x{id}");
        let p = state.get_weight_ptr(&dsc_name).expect("W_dsc registered");
        let mut got = vec![0u8; nchunk * od * 8];
        state.copy_from_device_pinned(p, &mut got);
        let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(dmis, 0, "device W_dsc vs mirror ({od}x{id})");
    }
}

#[test]
fn cuda_q4k_dsc_dense_byte_exact() {
    // r59 (Session F item 1) gate 1: the precomputed q4_K dsc f32-pair
    // plane must be byte-identical to an independent scalar mirror of the
    // kernel's in-loop SDS decode (d = f16(blk), dmin = f16(blk+2),
    // (sc, m) = get_scale_min_k4(c&7, blk+4), pair = (d*sc, -(dmin*m))).
    // Checked on the HOST expander and on the DEVICE upload (pinned
    // readback), over shapes covering several super-blocks per row and od
    // values that exercise the chunk-major [c*od + j] layout. Q4_K needs
    // no padding: the raw 144-byte block stride is already 16-B aligned.
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut s: u64 = 0xC0FF_EE12_3456_789B;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for (od, id) in [
        (64usize, 256usize),
        (40usize, 512usize),
        (24usize, 768usize),
    ] {
        let nbe = id / 256;
        let nchunk = id / 32;
        let row_len = nbe * 144;
        let raw: Vec<u8> = (0..od * row_len).map(|_| (rnd() & 0xFF) as u8).collect();
        // independent scalar mirror straight from the kernel formula
        let mut want = vec![0u8; nchunk * od * 8];
        for j in 0..od {
            for sb in 0..nbe {
                let base = (j * nbe + sb) * 144;
                let blk = &raw[base..base + 144];
                let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
                let q = &blk[4..16]; // 12 packed 6-bit scales+mins
                for cc in 0..8usize {
                    let (sc, m) = if cc < 4 {
                        (q[cc] & 63, q[cc + 4] & 63)
                    } else {
                        (
                            (q[cc + 4] & 0xF) | ((q[cc - 4] >> 6) << 4),
                            (q[cc + 4] >> 4) | ((q[cc] >> 6) << 4),
                        )
                    };
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    want[idx..idx + 4].copy_from_slice(&(d * (sc as f32)).to_bits().to_le_bytes());
                    want[idx + 4..idx + 8]
                        .copy_from_slice(&(-(dmin * (m as f32))).to_bits().to_le_bytes());
                }
            }
        }
        // host-side production expander vs the mirror
        let host =
            crate::cuda::CudaState::expand_q4k_dsc(&raw, od, id).expect("q4_K payload length");
        let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(hmis, 0, "expand_q4k_dsc vs mirror ({od}x{id})");
        // device upload path: build + read back + compare
        let name = format!("r59dsc{od}x{id}");
        state.register_weight(&name, &raw);
        state.register_weight_q4k_dsc(&name, &raw, od, id);
        let dsc_name = format!("{name}__q4dsc{od}x{id}");
        let p = state.get_weight_ptr(&dsc_name).expect("W_dsc registered");
        let mut got = vec![0u8; nchunk * od * 8];
        state.copy_from_device_pinned(p, &mut got);
        let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(dmis, 0, "device W_dsc vs mirror ({od}x{id})");
    }
}

/// #165: a payload that is not a q4_K payload registers **nothing** — no
/// `__q4dsc` device weight and no `q4k_dsc` map entry — while the q4_K payload
/// does. The positive control comes first and its registration is asserted by the
/// same registry queries the refusal uses, so a green run really observes the plane
/// (a query that is blind to planes could not see the control either).
#[test]
fn cuda_q4dsc_plane_is_q4k_only() {
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Qwen3-0.6B `ffn_down` geometry — the shape #165 names.
    let (od, id) = (1024usize, 3072usize);
    let q4k_len = id / 256 * 144 * od;
    let planes = |s: &crate::cuda::CudaState| -> (usize, usize) {
        let p = s.q4dsc_planes();
        (p.len(), p.iter().map(|(_, b)| b).sum())
    };
    let (base_n, base_b) = planes(state);

    // positive control: a real q4_K payload DOES register
    let ok_name = format!("f165q4k{od}x{id}");
    let raw = vec![0u8; q4k_len];
    state.register_weight(&ok_name, &raw);
    state.register_weight_q4k_dsc(&ok_name, &raw, od, id);
    let ok_dsc = format!("{ok_name}__q4dsc{od}x{id}");
    assert!(
        state.get_weight_ptr(&ok_dsc).is_some(),
        "positive control: the q4_K payload must register {ok_dsc}"
    );
    let (n_after_ok, b_after_ok) = planes(state);
    assert_eq!(
        (n_after_ok, b_after_ok),
        (base_n + 1, base_b + (id / 32) * od * 8),
        "the registry query must see exactly the control's plane"
    );

    // a q8_0-length payload (34 B / 32 elements) is LONGER than q4_K's: refused
    let q80_name = format!("f165q80{od}x{id}");
    let q80 = vec![0u8; od * (id / 32) * 34];
    state.register_weight(&q80_name, &q80);
    state.register_weight_q4k_dsc(&q80_name, &q80, od, id);
    assert!(
        state
            .get_weight_ptr(&format!("{q80_name}__q4dsc{od}x{id}"))
            .is_none(),
        "a q8_0 payload must not register a __q4dsc plane"
    );

    // a shorter payload (a future smaller-ratio type) is refused too, instead of
    // being read past the tensor
    let short_name = format!("f165short{od}x{id}");
    let short = vec![0u8; q4k_len - 144];
    state.register_weight(&short_name, &short);
    state.register_weight_q4k_dsc(&short_name, &short, od, id);
    assert!(
        state
            .get_weight_ptr(&format!("{short_name}__q4dsc{od}x{id}"))
            .is_none(),
        "a short payload must not register a __q4dsc plane"
    );

    // the two refusals added exactly zero planes
    assert_eq!(
        planes(state),
        (n_after_ok, b_after_ok),
        "only the q4_K control may add a plane"
    );
}

/// #165 acceptance: loading a real model on CUDA registers a `W_dsc` plane for
/// **exactly** the admissible q4_K weights and nothing else. The default cached model
/// is the 0.5B q4_0, whose `ffn_down` `[4864, 896]` passed the old geometry gate: the
/// type gate is the only thing that can refuse it (q4_0's bytes/element equals
/// q4_K's), so before the fix this asserted 24 planes / 26 148 864 B. Point
/// `MINFER_BATCH_TEST_MODEL` at a qwen2 q8_0 GGUF to measure the q8_0 model the
/// ticket names (expected: 0 either way, qwen3's loader never registered the plane).
/// `#[ignore]`: it needs a cached GGUF (the real-model set).
#[test]
#[ignore]
fn cuda_real_model_registers_q4dsc_planes_only_for_q4k() {
    use crate::gguf::GgmlType;
    let Some(state) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let path = match std::env::var("MINFER_BATCH_TEST_MODEL") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            let mut p = std::path::PathBuf::from(std::env::var("HOME").unwrap());
            p.push(
                ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/\
                 qwen2.5-0.5b-instruct-q4_0.gguf",
            );
            p
        }
    };
    if !path.exists() {
        eprintln!("skipping: {} not cached", path.display());
        return;
    }
    // Hold the model-load lock across load + query (a parallel load of another
    // architecture registers same-named tensors and swaps the registry underneath).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    // Expected set, from the GGUF index with the loader's own rule: a 2-D q4_K
    // tensor (ne[0] = in = id, ne[1] = out = od) with `id % 256 == 0` and `od` even.
    let mut want: Vec<String> = Vec::new();
    for part in &gguf.parts {
        for ti in &part.ctx.info {
            if ti.type_ != GgmlType::Q4_K {
                continue;
            }
            let (id, od) = (ti.ne[0] as usize, ti.ne[1] as usize);
            if id == 0 || id % 256 != 0 || od == 0 || od % 2 != 0 {
                continue;
            }
            want.push(format!("{}__q4dsc{od}x{id}", ti.name));
        }
    }
    want.sort();
    let _model = crate::models::load_model(&gguf).expect("load model");
    let planes = state.q4dsc_planes();
    let mut got: Vec<String> = planes.iter().map(|(n, _)| n.clone()).collect();
    got.sort();
    let bytes: usize = planes.iter().map(|(_, b)| b).sum();
    eprintln!(
        "q4dsc planes for {}: {} expected (q4_K), {} registered, {} bytes",
        path.display(),
        want.len(),
        got.len(),
        bytes
    );
    assert_eq!(
        got, want,
        "the W_dsc plane set must be exactly the model's admissible q4_K weights"
    );
}

#[test]
fn cuda_fa_prefill_attention_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    cb.set_kv_f16_for_test(true);
    let (nh, nk_h, hd) = (4usize, 2usize, 128usize);
    let nkt = nk_h * hd;
    let (nt, n_ctx) = (100usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = (0..nt).collect();

    let mut b = GraphBuilder::new();
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let _store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let at = b.attn(
        q,
        load,
        pp,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk_h,
            hd,
            hd_kv: hd,
            nkt,
            scale,
        },
    );
    b.output(at);
    let g = b.build();

    let (xb_q, xb_k, xb_v) = (
        cb.alloc_buffer(nh * hd * nt),
        cb.alloc_buffer(nkt * nt),
        cb.alloc_buffer(nkt * nt),
    );
    let xb_p = cb.alloc_buffer(nt);
    let ob_at = cb.alloc_buffer(nh * hd * nt);
    let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

    let qs: Vec<f32> = (0..nh * hd * nt)
        .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
        .collect();
    let ks: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let vs: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
    let to_half =
        |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect() };
    let ks_h = to_half(&ks);
    let vs_h = to_half(&vs);
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
    cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

    cb.exec_ids(
        &g.nodes[_store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
        .unwrap();

    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for (t, &p) in pos.iter().enumerate() {
        kfull[p * nkt..(p + 1) * nkt].copy_from_slice(&ks_h[t * nkt..(t + 1) * nkt]);
        vfull[p * nkt..(p + 1) * nkt].copy_from_slice(&vs_h[t * nkt..(t + 1) * nkt]);
    }
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qs, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    let mut maxe = 0f32;
    for (a, r) in agot.iter().zip(aref.iter()) {
        maxe = maxe.max((a - r).abs());
    }
    println!("fa prefill attention: max err {maxe:.6}");
    assert_close("fa_prefill_f16kv", &agot, &aref, 5e-3);
}

/// #144 item 3: FA prefill over a **packed** cache. Same shape and reference as
/// [`Self::cuda_fa_prefill_attention_parity`] — `nt = 100 > 16` and `hd = 128`,
/// the only combination that reaches the FA prefill — but the cache is Q8_0, so
/// the staging dequantizes each packed cell into the f16 tile instead of
/// copying halves. The reference is the CPU attention over the **same packed
/// bytes** dequantized on the host, so the only difference is the f16 staging:
/// the class the f16 FA gate already pins.
///
/// The observation arm is what makes this a gate about the *packed FA route*
/// rather than about "some attention kernel": the parity arm alone would pass
/// if the launch silently fell back to the general layout-tagged kernel, so the
/// chokepoint's `testfail::note_checked` counter is asserted to have moved.
#[test]
fn cuda_q8_0_fa_prefill_attention_parity() {
    use crate::graph::kvformat::KvFormat;
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_q8_for_test();
    let (nh, nk_h, hd) = (4usize, 2usize, 128usize);
    let nkt = nk_h * hd;
    let row_words = KvFormat::Q8_0.row_elems(nkt);
    let (nt, n_ctx) = (100usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = (0..nt).collect();

    let mut b = GraphBuilder::new();
    b.set_kv_format(KvFormat::Q8_0);
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let at = b.attn(
        q,
        load,
        pp,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk_h,
            hd,
            hd_kv: hd,
            nkt,
            scale,
        },
    );
    b.output(at);
    let g = b.build();

    let (xb_q, xb_k, xb_v) = (
        cb.alloc_buffer(nh * hd * nt),
        cb.alloc_buffer(nkt * nt),
        cb.alloc_buffer(nkt * nt),
    );
    let xb_p = cb.alloc_buffer(nt);
    let ob_at = cb.alloc_buffer(nh * hd * nt);
    let (kreg, vreg) = (
        cb.alloc_buffer(n_ctx * row_words),
        cb.alloc_buffer(n_ctx * row_words),
    );

    let qs: Vec<f32> = (0..nh * hd * nt)
        .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
        .collect();
    let ks: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let vs: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    cb.write_host(kreg, &vec![0f32; n_ctx * row_words]).unwrap();
    cb.write_host(vreg, &vec![0f32; n_ctx * row_words]).unwrap();

    crate::testfail::reset_checked();
    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
        .unwrap();

    // The reference reads the *same* packed bytes the device wrote (the store is
    // byte-exact against the CPU quantizer, its own gate), dequantized on the
    // host, so the only difference left is the f16 staging.
    let pk = cb.copy_to_host(kreg).unwrap();
    let pv = cb.copy_to_host(vreg).unwrap();
    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for &p in &pos {
        crate::graph::kvformat::unpack_q8_0_cells(
            &pk,
            nkt,
            p,
            1,
            &mut kfull[p * nkt..(p + 1) * nkt],
        );
        crate::graph::kvformat::unpack_q8_0_cells(
            &pv,
            nkt,
            p,
            1,
            &mut vfull[p * nkt..(p + 1) * nkt],
        );
    }
    assert!(
        kfull.iter().any(|x| *x != 0.0),
        "the dequantized reference is all zero; the comparison would be vacuous"
    );
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qs, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    let mut maxe = 0f32;
    for (a, r) in agot.iter().zip(aref.iter()) {
        maxe = maxe.max((a - r).abs());
    }
    println!("q8_0 fa prefill attention: max err {maxe:.6}");
    assert_close("fa_prefill_q8kv", &agot, &aref, 5e-3);

    // Observation arm: the dispatch must have taken the FA launch, not the
    // general layout-tagged fallback.
    assert!(
        crate::testfail::checked("cuda_fa_prefill_q8_0") > 0,
        "the packed prefill did not reach the FA launch (it fell back to the general \
         layout-tagged kernel); this gate would otherwise pass on the old route"
    );
}

// 8c: prefill Q4_0 matmul (nt > 1, id <= 8192) routes through the
// Q8_0-activation GEMM. The reference builds the SAME Q8_0 activation
// blocks and uses dot_q4_0_q8_0 — the kernel's exact math — so the
// tolerance is tight. The nt == 1 call takes the f32-activation path
// (decode); its reference dequantizes the weights.
#[test]
fn cuda_q4_0_prefill_q8_0_gemm_parity() {
    crate::cuda::CudaState::init();
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (od, id, nt) = (32usize, 64usize, 3usize);
    let nb = id / 32;

    // build a Q4_0 weight: d = amax/7, biased nibbles (v + 8)
    let wf: Vec<f32> = (0..od * id)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let mut wq = Vec::with_capacity(od * nb * 18);
    for r in 0..od {
        for b in 0..nb {
            let row = &wf[r * id + b * 32..r * id + (b + 1) * 32];
            let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / 7.0;
            let di = if d != 0.0 { 1.0 / d } else { 0.0 };
            let dbits = half::f16::from_f32(d).to_le_bytes();
            wq.push(dbits[0]);
            wq.push(dbits[1]);
            for j in 0..16 {
                let q0 = (row[j] * di).round().clamp(-8.0, 7.0) as i8 + 8;
                let q1 = (row[j + 16] * di).round().clamp(-8.0, 7.0) as i8 + 8;
                wq.push(((q1 as u8) << 4) | (q0 as u8));
            }
        }
    }
    let state = cb.state;
    state.register_weight("w40", &wq);
    let wptr = state.get_weight_ptr("w40").unwrap();

    let xs: Vec<f32> = (0..id * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    // per-token Q8_0 activation blocks (same layout the kernel reads)
    let q8s: Vec<Vec<u8>> = (0..nt)
        .map(|t| crate::quants::quantize_row_q8_0(&xs[t * id..(t + 1) * id]))
        .collect();

    let xb = cb.alloc_buffer(id * nt);
    let out = cb.alloc_buffer(od * nt);
    cb.write_host(xb, &xs).unwrap();
    let (xptr, optr) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out).unwrap());

    // nt > 1: Q8_0-activation path
    state
        .matmul_f32_ptr(wptr, TensorType::Q4_0, xptr, optr, od, id, nt)
        .unwrap();
    cb.synchronize();
    let got = cb.copy_to_host(out).unwrap();
    for t in 0..nt {
        for r in 0..od {
            let want = crate::quants::dot_q4_0_q8_0(&wq[r * nb * 18..(r + 1) * nb * 18], &q8s[t]);
            assert!(
                (got[t * od + r] - want).abs() < 1e-3,
                "q8_0 path [{t}][{r}] {} vs {want}",
                got[t * od + r]
            );
        }
    }

    // CPU cross-check: my hand dequant vs dot_q4_0_q8_0 (same wq bytes)
    let q8_tok0 = &q8s[0];
    for r in [0usize, 1, 17] {
        let via_dot = crate::quants::dot_q4_0_q8_0(&wq[r * nb * 18..(r + 1) * nb * 18], q8_tok0);
        let mut deq = 0f32;
        for b in 0..nb {
            let blk = &wq[r * nb * 18 + b * 18..r * nb * 18 + (b + 1) * 18];
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let q8b = &q8_tok0[b * 34..(b + 1) * 34];
            let d8 = half::f16::from_le_bytes([q8b[0], q8b[1]]).to_f32();
            let mut si = 0i32;
            for j in 0..16 {
                let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                let v1 = (blk[2 + j] >> 4) as i32 - 8;
                si += v0 * q8b[2 + j] as i8 as i32 + v1 * q8b[2 + j + 16] as i8 as i32;
            }
            deq += si as f32 * d * d8;
        }
        let deq_f32 = {
            // dequant-want against raw f32 x (what the f32 kernel reads);
            // weight block b pairs with x[b*32 .. b*32+32]
            let mut acc = 0f32;
            for b in 0..nb {
                let blk = &wq[r * nb * 18 + b * 18..r * nb * 18 + (b + 1) * 18];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let xb = &xs[b * 32..(b + 1) * 32];
                for j in 0..16 {
                    let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                    let v1 = (blk[2 + j] >> 4) as i32 - 8;
                    acc += d * (v0 as f32 * xb[j] + v1 as f32 * xb[j + 16]);
                }
            }
            acc
        };
        assert!(
            (via_dot - deq).abs() < 1e-2 && (via_dot - deq_f32).abs() < 5e-2,
            "crosscheck r={r}: dot_q8 {via_dot} vs dequant-q8 {deq} vs dequant-f32 {deq_f32}"
        );
    }

    // nt == 1: f32-activation path (decode), reference dequantizes weights
    let out1 = cb.alloc_buffer(od);
    let (x1, o1) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out1).unwrap());
    state
        .matmul_f32_ptr(wptr, TensorType::Q4_0, x1, o1, od, id, 1)
        .unwrap();
    cb.synchronize();
    let got1 = cb.copy_to_host(out1).unwrap();
    for r in 0..od {
        let mut want = 0f32;
        for b in 0..nb {
            let blk = &wq[r * nb * 18 + b * 18..r * nb * 18 + (b + 1) * 18];
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let xrow = &xs[b * 32..(b + 1) * 32];
            for j in 0..16 {
                let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                let v1 = (blk[2 + j] >> 4) as i32 - 8;
                want += d * (v0 as f32 * xrow[j] + v1 as f32 * xrow[j + 16]);
            }
        }
        assert!(
            (got1[r] - want).abs() < 0.05,
            "f32 path [{r}] got {} want {want} diff {}",
            got1[r],
            got1[r] - want
        );
    }
}

/// 8m: the prefill f16 GEMM path (nt >= 16) for every supported quant
/// type — random VALID block bytes with small d/dmin, reference computed
/// in Rust by dequantizing those exact bytes (kernel-vs-reference parity;
/// quantization quality is irrelevant). Tails: od=70, nt=33 (id stays
/// %32==0 like every real tensor). Real 7B Q4_K check at the end, skipped
/// when the dump is absent so the suite stays hermetic.
#[test]
fn cuda_prefill_f16_gemm_parity() {
    fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }
    let _guard = crate::cuda::CudaState::model_load_guard();
    crate::cuda::CudaState::init();
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (od, id, nt) = (70usize, 256usize, 33usize);
    let state = cb.state;

    // seeded pseudo-random source (deterministic across runs)
    let mut seed = 0x2545F491u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let xs: Vec<f32> = (0..id * nt)
        .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let xb = cb.alloc_buffer(id * nt);
    let out = cb.alloc_buffer(od * nt);
    cb.write_host(xb, &xs).unwrap();
    let (xptr, optr) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out).unwrap());

    // build [type → (wq bytes, dequant closure)]
    // d values are small so the f16 scratch never overflows.
    let mut mk = |nbytes: usize| -> Vec<u8> { (0..nbytes).map(|_| (rnd() & 0xFF) as u8).collect() };

    // Q8_0: d=0.01 + int8 q
    let mut wq80 = mk(od * (id / 32) * 34);
    for g in 0..od * (id / 32) {
        let db = half::f16::from_f32(0.01).to_le_bytes();
        wq80[g * 34] = db[0];
        wq80[g * 34 + 1] = db[1];
    }
    // Q4_0: d=0.05 + biased nibbles (kernel does nib - 8)
    let mut wq40 = mk(od * (id / 32) * 18);
    for g in 0..od * (id / 32) {
        let db = half::f16::from_f32(0.05).to_le_bytes();
        wq40[g * 18] = db[0];
        wq40[g * 18 + 1] = db[1];
    }
    // Q4_K: d=0.01, dmin=0.005, raw scales/nibbles
    let nsp = id / 256;
    let mut wq4k = mk(od * nsp * 144);
    for r in 0..od {
        for sp in 0..nsp {
            let blk = &mut wq4k[(r * nsp + sp) * 144..(r * nsp + sp) * 144 + 144];
            let db = half::f16::from_f32(0.01).to_le_bytes();
            blk[0] = db[0];
            blk[1] = db[1];
            let mb_ = half::f16::from_f32(0.005).to_le_bytes();
            blk[2] = mb_[0];
            blk[3] = mb_[1];
        }
    }
    // Q5_K: d=0.01, dmin=0.005 (176B blocks)
    let mut wq5k = mk(od * nsp * 176);
    for r in 0..od {
        for sp in 0..nsp {
            let blk = &mut wq5k[(r * nsp + sp) * 176..(r * nsp + sp) * 176 + 176];
            let db = half::f16::from_f32(0.01).to_le_bytes();
            blk[0] = db[0];
            blk[1] = db[1];
            let mb_ = half::f16::from_f32(0.005).to_le_bytes();
            blk[2] = mb_[0];
            blk[3] = mb_[1];
        }
    }
    // Q6_K: raw 210B blocks, d = 0.01 at offset 208 (LAST field)
    let mut wq6k = mk(od * nsp * 210);
    for r in 0..od {
        for sp in 0..nsp {
            let blk = &mut wq6k[(r * nsp + sp) * 210..(r * nsp + sp) * 210 + 210];
            let db = half::f16::from_f32(0.01).to_le_bytes();
            blk[208] = db[0];
            blk[209] = db[1];
        }
    }

    state.register_weight("gemm_w80", &wq80);
    state.register_weight("gemm_w40", &wq40);
    state.register_weight("gemm_w4k", &wq4k);
    state.register_weight("gemm_w5k", &wq5k);
    state.register_weight_q6k_padded("gemm_w6k", &wq6k, od, id);

    // ── run the GEMM path per type (nt=33 ≥ 16 hits the gate) ──
    let cases: [(TensorType, &str, bool); 5] = [
        (TensorType::Q8_0, "gemm_w80", false),
        (TensorType::Q4_0, "gemm_w40", false),
        (TensorType::Q4_K, "gemm_w4k", false),
        (TensorType::Q5_K, "gemm_w5k", false),
        (TensorType::Q6_K, "gemm_w6k", true),
    ];
    for (ttype, name, padded) in cases {
        let wptr = state.get_weight_ptr(name).unwrap();
        state
            .matmul_f32_ptr_layout(wptr, ttype, xptr, optr, od, id, nt, padded)
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();

        // reference: dequant the same bytes, plain f32 dot with raw xs
        let mut want = vec![0f32; od * nt];
        for r in 0..od {
            for t in 0..nt {
                let mut acc = 0f32;
                match ttype {
                    TensorType::Q8_0 => {
                        for g in 0..id / 32 {
                            let blk = &wq80[(r * (id / 32) + g) * 34..][..34];
                            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                            for i in 0..32 {
                                acc += d * (blk[2 + i] as i8 as f32) * xs[t * id + g * 32 + i];
                            }
                        }
                    }
                    TensorType::Q4_0 => {
                        for g in 0..id / 32 {
                            let blk = &wq40[(r * (id / 32) + g) * 18..][..18];
                            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                            for j in 0..16 {
                                let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                                let v1 = (blk[2 + j] >> 4) as i32 - 8;
                                acc += d
                                    * (v0 as f32 * xs[t * id + g * 32 + j]
                                        + v1 as f32 * xs[t * id + g * 32 + j + 16]);
                            }
                        }
                    }
                    TensorType::Q4_K => {
                        for ib in 0..nsp {
                            let blk = &wq4k[(r * nsp + ib) * 144..][..144];
                            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                            let mut scb = [0u8; 12];
                            scb.copy_from_slice(&blk[4..16]);
                            for j in 0..4 {
                                let (s0, m0) = k4_scale(&scb, 2 * j);
                                let (s1, m1) = k4_scale(&scb, 2 * j + 1);
                                for l in 0..32 {
                                    let b8 = blk[16 + j * 32 + l];
                                    let base = ib * 256 + j * 64;
                                    let v0 = (b8 & 0x0F) as f32 * d * s0 as f32 - dmin * m0 as f32;
                                    let v1 = (b8 >> 4) as f32 * d * s1 as f32 - dmin * m1 as f32;
                                    acc += v0 * xs[t * id + base + l];
                                    acc += v1 * xs[t * id + base + 32 + l];
                                }
                            }
                        }
                    }
                    TensorType::Q5_K => {
                        for ib in 0..nsp {
                            let blk = &wq5k[(r * nsp + ib) * 176..][..176];
                            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                            let mut scb = [0u8; 12];
                            scb.copy_from_slice(&blk[4..16]);
                            for sub in 0..8 {
                                let (scb_s, mb) = k4_scale(&scb, sub);
                                let ci = sub >> 1;
                                let hi = sub & 1;
                                let q4 = &blk[48 + ci * 32..48 + ci * 32 + 32];
                                let qh = &blk[16..48]; // 256 high bits = 32 bytes
                                for l in 0..32 {
                                    let nib = if hi != 0 { q4[l] >> 4 } else { q4[l] & 0x0F };
                                    let wv = nib as f32 + 16.0 * ((qh[l] >> sub) & 1) as f32;
                                    let v = d * scb_s as f32 * wv - dmin * mb as f32;
                                    acc += v * xs[t * id + ib * 256 + sub * 32 + l];
                                }
                            }
                        }
                    }
                    TensorType::Q6_K => {
                        for ib in 0..nsp {
                            let blk = &wq6k[(r * nsp + ib) * 210..][..210];
                            let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
                            for sub in 0..16 {
                                let n = sub / 8;
                                let rem = sub % 8;
                                let tt = rem / 2;
                                let gq = rem % 2;
                                let ql_off = n * 64 + (tt % 2) * 32 + gq * 16;
                                // qh field lives at blk[128..192] (64 bytes,
                                // 2 bits per element); qh_off is relative to it.
                                let qh_off = 128 + n * 32 + gq * 16;
                                let dsc = d * (blk[192 + n * 8 + tt * 2 + gq] as i8 as f32);
                                for rr in 0..16 {
                                    let nib = if tt < 2 {
                                        (blk[ql_off + rr] & 0x0F) as i32
                                    } else {
                                        (blk[ql_off + rr] >> 4) as i32
                                    };
                                    let q2 = ((blk[qh_off + rr] >> (tt * 2)) & 3) as i32;
                                    let v = dsc * (((nib | (q2 << 4)) - 32) as f32);
                                    acc += v * xs
                                        [t * id + ib * 256 + n * 128 + tt * 32 + gq * 16 + rr];
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                want[t * od + r] = acc;
            }
        }

        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        let mut worst = (0f32, 0usize);
        for i in 0..got.len() {
            let e = (got[i] - want[i]).abs();
            if e > worst.0 {
                worst = (e, i);
            }
        }
        println!(
            "prefill f16 gemm [{name:?}]: max err {:.5} at {} (got {:.4} want {:.4}, scale {scale:.3})",
            worst.0,
            worst.1,
            got[worst.1],
            want[worst.1]
        );
        // f16 weight/activation rounding (~2^-11 rel per element) over a
        // 256-length dot: well under 2% of the row scale.
        assert!(
            worst.0 <= scale * 2e-2,
            "prefill gemm {name:?}: err {} > {} at {}",
            worst.0,
            scale * 2e-2,
            worst.1
        );
    }

    // ── real 7B Q4_K weight (attn_q 3584×3584) through the GEMM path ──
    let Ok(wb) = std::fs::read("/tmp/minfer_phase7/real_blk_0_attn_q_weight.bin") else {
        eprintln!("real q4_k dump absent — skipping the real-weight GEMM check");
        return;
    };
    let (rod, rid) = (3584usize, 3584usize);
    assert_eq!(wb.len(), rod * (rid / 256) * 144);
    state.register_weight("gemm_realq4k", &wb);
    let wptr = state.get_weight_ptr("gemm_realq4k").unwrap();
    let rnt = 17usize;
    let rxs: Vec<f32> = (0..rid * rnt)
        .map(|i| ((i * 73) % 17) as f32 / 8.0 - 1.0)
        .collect();
    let rxb = cb.alloc_buffer(rid * rnt);
    let rout = cb.alloc_buffer(rod * rnt);
    cb.write_host(rxb, &rxs).unwrap();
    let (rxp, rop) = (cb.ptr_of(rxb).unwrap(), cb.ptr_of(rout).unwrap());
    state
        .matmul_f32_ptr_layout(wptr, TensorType::Q4_K, rxp, rop, rod, rid, rnt, false)
        .unwrap();
    cb.synchronize();
    let got = cb.copy_to_host(rout).unwrap();
    let mut worst = (0f32, 0usize);
    let mut scale = 1e-9f32;
    for t in 0..rnt {
        for r in 0..rod {
            let mut acc = 0f32;
            for ib in 0..rid / 256 {
                let blk = &wb[(r * (rid / 256) + ib) * 144..][..144];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                let mut scb = [0u8; 12];
                scb.copy_from_slice(&blk[4..16]);
                for j in 0..4 {
                    let (s0, m0) = k4_scale(&scb, 2 * j);
                    let (s1, m1) = k4_scale(&scb, 2 * j + 1);
                    for l in 0..32 {
                        let b8 = blk[16 + j * 32 + l];
                        let base = ib * 256 + j * 64;
                        let v0 = (b8 & 0x0F) as f32 * d * s0 as f32 - dmin * m0 as f32;
                        let v1 = (b8 >> 4) as f32 * d * s1 as f32 - dmin * m1 as f32;
                        acc += v0 * rxs[t * rid + base + l];
                        acc += v1 * rxs[t * rid + base + 32 + l];
                    }
                }
            }
            scale = scale.max(acc.abs());
            let e = (got[t * rod + r] - acc).abs();
            if e > worst.0 {
                worst = (e, t * rod + r);
            }
        }
    }
    println!(
        "real 7B q4_k f16 gemm: max err {:.4} at {} (scale {scale:.3})",
        worst.0, worst.1
    );
    assert!(
        worst.0 <= scale * 2e-2,
        "real q4_k f16 gemm err {}",
        worst.0
    );
}

// 8d: split-K decode attention parity (nt == 1 routes to the split path).
// nkv = 3 exercises EMPTY splits (positions[0] = 2 → splits 3..7 have no
// rows); nkv = 37 exercises a partial last split with SPLITS = 8. Both KV
// layouts checked. Reference: cpu_gqa_attn over the same KV (zero rows +
// one stored row).
#[test]
fn cuda_attn_split_decode_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // D3-4 L1: shape 2 (hd=128) drives the hybrid kernel on the
    // kv_f16=true arm — n_ctx 4200 covers the runtime rpw dispatch
    // boundary (nkv 1920 -> rpw 15 -> 1-warp body; nkv 1921 -> rpw 16 ->
    // 4-warp body), full 32-row windows and chunk boundaries; shape 1
    // (hd=8) keeps covering the plain 1-warp kernel. n_ctx of shape 1 is
    // sized so pos0 can sweep the ATTN_SPLITS=32 chunk boundaries: full
    // splits, a partially-filled split, and trailing idle splits
    // (mx=-INF/S=0 partials) all get exercised (nkv = pos0 + 1).
    for (nh, nk_h, hd, n_ctx, pos0s) in [
        (
            4usize,
            2usize,
            8usize,
            208usize,
            [2usize, 32, 62, 63, 64, 126, 127, 128, 190, 206, 207],
        ),
        (
            4usize,
            2usize,
            128usize,
            4200usize,
            [2usize, 32, 63, 64, 127, 128, 1023, 1919, 1920, 4094, 4095],
        ),
        // D3-6 2a: the 14B GQA geometry (40:8, gqa=5) drives the
        // GQA-batched kernel (grid (ATTN_SPLITS, 8), 160 threads) on the
        // same pos0 sweep — the 1920/1921 boundary picks between the
        // bitwise 1-warp incumbent (nkv 1920) and the batched body
        // (nkv 1921), and 4094/4095 cover full-window chunk tails.
        (
            40usize,
            8usize,
            128usize,
            4200usize,
            [2usize, 32, 63, 64, 127, 128, 1023, 1919, 1920, 4094, 4095],
        ),
        // D3-6 2a: the 7B GQA geometry (28:4, gqa=7 → 224-thread blocks).
        // nkv 2808 (pos0 2807) reproduces the in-situ decode-step shape at
        // the divergence point seen in the 7B greedy gate.
        (
            28usize,
            4usize,
            128usize,
            4200usize,
            [2usize, 32, 63, 64, 127, 128, 1919, 1920, 2807, 4094, 4095],
        ),
    ] {
        let nkt = nk_h * hd;
        let scale = 1.0 / (hd as f32).sqrt();

        for kv_f16 in [false, true] {
            for pos0 in pos0s {
                let nkv = pos0 + 1;
                cb.set_kv_f16_for_test(kv_f16);
                let mut b = GraphBuilder::new();
                let q = b.input("q", [nh * hd, 1, 1, 1], DType::F32);
                let k = b.input("k", [nkt, 1, 1, 1], DType::F32);
                let v = b.input("v", [nkt, 1, 1, 1], DType::F32);
                let pp = b.input("positions", [1, 1, 1, 1], DType::I32);
                let store = b.kvcache_store(0, k, v, n_ctx);
                let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
                let at = b.attn(
                    q,
                    load,
                    pp,
                    AttnMode::Gqa,
                    AttnMeta {
                        layer: 0,
                        n_head: nh,
                        n_head_kv: nk_h,
                        hd,
                        hd_kv: hd,
                        nkt,
                        scale,
                    },
                );
                b.output(at);
                let g = b.build();

                let (xb_q, xb_k, xb_v) = (
                    cb.alloc_buffer(nh * hd),
                    cb.alloc_buffer(nkt),
                    cb.alloc_buffer(nkt),
                );
                let xb_p = cb.alloc_buffer(1);
                let ob_at = cb.alloc_buffer(nh * hd);
                let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

                let qs: Vec<f32> = (0..nh * hd)
                    .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
                    .collect();
                let ks: Vec<f32> = (0..nkt)
                    .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
                    .collect();
                let vs: Vec<f32> = (0..nkt)
                    .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
                    .collect();
                let pb = vec![f32::from_bits(pos0 as u32)];
                let to_half = |x: &[f32]| -> Vec<f32> {
                    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
                };
                let (ks_r, vs_r) = if kv_f16 {
                    (to_half(&ks), to_half(&vs))
                } else {
                    (ks.clone(), vs.clone())
                };
                cb.write_host(xb_q, &qs).unwrap();
                cb.write_host(xb_k, &ks).unwrap();
                cb.write_host(xb_v, &vs).unwrap();
                cb.write_host(xb_p, &pb).unwrap();
                cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
                cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

                cb.exec_ids(
                    &g.nodes[store],
                    &[xb_k, xb_v, xb_p],
                    kreg,
                    Some((kreg, vreg)),
                )
                .unwrap();
                cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
                    .unwrap();

                let mut kfull = vec![0f32; nkt * n_ctx];
                let mut vfull = vec![0f32; nkt * n_ctx];
                kfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&ks_r);
                vfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&vs_r);
                let mut aref = vec![0f32; nh * hd];
                crate::graph::cpu_backend::cpu_gqa_attn(
                    &qs,
                    &kfull,
                    &vfull,
                    &crate::graph::cpu_backend::causal_span(&[pos0]),
                    1,
                    nh,
                    nk_h,
                    hd,
                    hd,
                    nkt,
                    &mut aref,
                    scale,
                )
                .unwrap();
                let agot = cb.copy_to_host(ob_at).unwrap();
                assert_close(
                    &format!("attn_split(f16kv={kv_f16}, nkv={nkv})"),
                    &agot,
                    &aref,
                    1e-4,
                );
            }
        }
    }

    // D3-6 2a: kernel-level gate of the calibrated tolerance package on
    // realistic outlier-scale data (docs/CUDA_OPTIMIZATION.md §2D D3a:
    // residual |q|~50, V outliers ±127 — the h4w body measured 6.5e-5
    // vs CPU on this class, the incumbent 3.8e-5). The GQA-batched body
    // shares the h4w window loop verbatim, so the same ≤1e-4 bound
    // applies; 14B geometry (40:8, gqa=5) inside the batched regime
    // (nkv 1921 / 4096, f16 KV).
    for pos0 in [1920usize, 4095usize] {
        let (nh, nk_h, hd, n_ctx) = (40usize, 8usize, 128usize, 4200usize);
        let nkt = nk_h * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let nkv = pos0 + 1;
        cb.set_kv_f16_for_test(true);
        let mut b = GraphBuilder::new();
        let q = b.input("q", [nh * hd, 1, 1, 1], DType::F32);
        let k = b.input("k", [nkt, 1, 1, 1], DType::F32);
        let v = b.input("v", [nkt, 1, 1, 1], DType::F32);
        let pp = b.input("positions", [1, 1, 1, 1], DType::I32);
        let store = b.kvcache_store(0, k, v, n_ctx);
        let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
        let at = b.attn(
            q,
            load,
            pp,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk_h,
                hd,
                hd_kv: hd,
                nkt,
                scale,
            },
        );
        b.output(at);
        let g = b.build();

        let (xb_q, xb_k, xb_v) = (
            cb.alloc_buffer(nh * hd),
            cb.alloc_buffer(nkt),
            cb.alloc_buffer(nkt),
        );
        let xb_p = cb.alloc_buffer(1);
        let ob_at = cb.alloc_buffer(nh * hd);
        let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

        // Outlier scale: q residual |q|~50-60, V outliers |v|~140 (f16
        // representable); K stays at the tame scale like the D3a probe.
        let qs: Vec<f32> = (0..nh * hd)
            .map(|i| (((i * 37) % 19) as f32 / 5.0 - 1.9) * 30.0)
            .collect();
        let ks: Vec<f32> = (0..nkt)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let vs: Vec<f32> = (0..nkt)
            .map(|i| (((i * 57) % 11) as f32 / 3.0 - 1.8) * 80.0)
            .collect();
        let pb = vec![f32::from_bits(pos0 as u32)];
        let to_half = |x: &[f32]| -> Vec<f32> {
            x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
        };
        let (ks_r, vs_r) = (to_half(&ks), to_half(&vs));
        cb.write_host(xb_q, &qs).unwrap();
        cb.write_host(xb_k, &ks).unwrap();
        cb.write_host(xb_v, &vs).unwrap();
        cb.write_host(xb_p, &pb).unwrap();
        cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
        cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

        cb.exec_ids(
            &g.nodes[store],
            &[xb_k, xb_v, xb_p],
            kreg,
            Some((kreg, vreg)),
        )
        .unwrap();
        cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
            .unwrap();

        let mut kfull = vec![0f32; nkt * n_ctx];
        let mut vfull = vec![0f32; nkt * n_ctx];
        kfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&ks_r);
        vfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&vs_r);
        let mut aref = vec![0f32; nh * hd];
        crate::graph::cpu_backend::cpu_gqa_attn(
            &qs,
            &kfull,
            &vfull,
            &crate::graph::cpu_backend::causal_span(&[pos0]),
            1,
            nh,
            nk_h,
            hd,
            hd,
            nkt,
            &mut aref,
            scale,
        )
        .unwrap();
        let agot = cb.copy_to_host(ob_at).unwrap();
        assert_close(
            &format!("attn_split_gqa_batched_outlier(nkv={nkv})"),
            &agot,
            &aref,
            1e-4,
        );
    }
}

// 8f: Q5_1 / Q5_K f32-activation matmul parity (incl. the Q5_K partial
// tail super-block at id = 896 = 3.5 × 256). The weight blocks are
// quantized in-test against scales unpacked with the REAL
// block::unpack_q4k_scales, so kernel and reference share the exact
// decode math and the tolerance stays tight.
#[test]
fn cuda_q5_matmul_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let nt = 3usize;

    // ── Q5_1: od 8, id 64 (2 blocks / row) ──
    {
        let (od, id) = (8usize, 64usize);
        let nb = id / 32;
        let wf: Vec<f32> = (0..od * id)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let mut wq = Vec::new();
        for r in 0..od {
            for b in 0..nb {
                let row = &wf[r * id + b * 32..r * id + (b + 1) * 32];
                let amax = row.iter().fold(0f32, |m, &v| m.max(v));
                let amin = row.iter().fold(0f32, |m, &v| m.min(v));
                let d = (amax - amin) / 31.0;
                let di = if d != 0.0 { 1.0 / d } else { 0.0 };
                wq.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                wq.extend_from_slice(&half::f16::from_f32(amin).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((row[j] - amin) * di).round().clamp(0.0, 31.0) as u32;
                    let u_hi = ((row[j + 16] - amin) * di).round().clamp(0.0, 31.0) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                wq.extend_from_slice(&qh.to_le_bytes());
                wq.extend_from_slice(&qs);
            }
        }
        let state = cb.state;
        state.register_weight("w51", &wq);
        let wptr = state.get_weight_ptr("w51").unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        state
            .matmul_f32_ptr(
                wptr,
                TensorType::Q5_1,
                cb.ptr_of(xb).unwrap(),
                cb.ptr_of(out).unwrap(),
                od,
                id,
                nt,
            )
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();
        // independent dequant reference
        for t in 0..nt {
            for r in 0..od {
                let mut want = 0f32;
                for b in 0..nb {
                    let blk = &wq[(r * nb + b) * 24..(r * nb + b) * 24 + 24];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let m = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                    let xrow = &xs[t * id + b * 32..t * id + (b + 1) * 32];
                    for j in 0..16 {
                        let u_lo = ((blk[8 + j] & 0xF) as f32) + 16.0 * ((qh >> j) & 1) as f32;
                        let u_hi =
                            ((blk[8 + j] >> 4) as f32) + 16.0 * ((qh >> (j + 16)) & 1) as f32;
                        want += d * (u_lo * xrow[j] + u_hi * xrow[j + 16])
                            + m * (xrow[j] + xrow[j + 16]);
                    }
                }
                assert!(
                    (got[t * od + r] - want).abs() < 5e-3,
                    "q5_1 [{t}][{r}] {} vs {want}",
                    got[t * od + r]
                );
            }
        }
    }

    // ── Q5_0: od 8, id 64 (2 blocks / row) — the tok_embd type of the
    // 0.5B q4_k_m GGUFs; decode f32-activation kernel parity ──
    {
        let (od, id) = (8usize, 64usize);
        let nb = id / 32;
        let mut wq = Vec::new();
        for r in 0..od {
            for b in 0..nb {
                let d = 0.02f32 + 0.003 * ((r * 5 + b) % 7) as f32;
                wq.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((r * 11 + b * 7 + j * 3) % 32) as u32;
                    let u_hi = ((r * 7 + b * 5 + j) % 32) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                wq.extend_from_slice(&qh.to_le_bytes());
                wq.extend_from_slice(&qs);
            }
        }
        let state = cb.state;
        state.register_weight("w50", &wq);
        let wptr = state.get_weight_ptr("w50").unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        state
            .matmul_f32_ptr(
                wptr,
                TensorType::Q5_0,
                cb.ptr_of(xb).unwrap(),
                cb.ptr_of(out).unwrap(),
                od,
                id,
                nt,
            )
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();
        // independent dequant reference
        for t in 0..nt {
            for r in 0..od {
                let mut want = 0f32;
                for b in 0..nb {
                    let blk = &wq[(r * nb + b) * 22..(r * nb + b) * 22 + 22];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                    let xrow = &xs[t * id + b * 32..t * id + (b + 1) * 32];
                    for j in 0..16 {
                        let v_lo =
                            ((blk[6 + j] & 0xF) as f32) + 16.0 * ((qh >> j) & 1) as f32 - 16.0;
                        let v_hi = ((blk[6 + j] >> 4) as f32)
                            + 16.0 * ((qh >> (j + 16)) & 1) as f32
                            - 16.0;
                        want += d * (v_lo * xrow[j] + v_hi * xrow[j + 16]);
                    }
                }
                assert!(
                    (got[t * od + r] - want).abs() < 5e-3,
                    "q5_0 [{t}][{r}] {} vs {want}",
                    got[t * od + r]
                );
            }
        }
    }

    // ── Q5_K: od 8, id 896 (PARTIAL tail super-block: 3.5 × 256) ──
    // Weight values are GENERATED from the decode formula with random
    // per-sub w (0..31) against scales unpacked from random sc bytes —
    // the test targets the kernel's decode/indexing/tail-masking
    // correctness, not a quantizer.
    {
        let (od, id) = (8usize, 896usize);
        let nsp = (id + 255) / 256; // 4 — last one is partial (4 valid subs)
        let mut wf: Vec<f32> = (0..od * id)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let mut wq = vec![0u8; od * nsp * 176];
        for r in 0..od {
            for sp in 0..nsp {
                let blk_off = (r * nsp + sp) * 176;
                let sc: [u8; 12] = core::array::from_fn(|i| ((i * 7 + 3) % 63 + 1) as u8);
                let (scales, mins) = crate::block::unpack_q4k_scales(&sc);
                wq[blk_off..blk_off + 4].copy_from_slice(&{
                    // d = 0.25, dmin = 0.25 (exact in f16)
                    let b = half::f16::from_f32(0.25).to_le_bytes();
                    [b[0], b[1], b[0], b[1]]
                });
                wq[blk_off + 4..blk_off + 16].copy_from_slice(&sc);
                // qh/qs stay zero for invalid tail subs (masked out)
                let valid = ((id - sp * 256).min(256) + 31) / 32;
                for sub in 0..valid {
                    let base = sp * 256 + sub * 32;
                    let row = &wf[r * id + base..r * id + base + 32];
                    // invert the decode: v = d·s8·w − dmin·m8 →
                    // w = (v + dmin·m8) / (d·s8); needs w ∈ 0..31 —
                    // instead regenerate v FROM w so it is exact:
                    for l in 0..32 {
                        let seed = (r * 91 + base + l) % 32;
                        let v = 0.25 * scales[sub] as f32 * seed as f32 - 0.25 * mins[sub] as f32;
                        // overwrite wf so the reference dot uses exact values
                        wf[r * id + base + l] = v;
                        let wv = seed as u8;
                        let ci = sub >> 1;
                        if sub & 1 == 1 {
                            qs_byte(&mut wq[blk_off + 48..], ci, l, wv, true);
                        } else {
                            qs_byte(&mut wq[blk_off + 48..], ci, l, wv, false);
                        }
                        qh_byte(&mut wq[blk_off + 16..], l, sub, wv);
                    }
                }
            }
        }
        let state = cb.state;
        state.register_weight("w5k", &wq);
        let wptr = state.get_weight_ptr("w5k").unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        // Step 82: nt = 3 dispatches Q5_K to the multi-token MMVQ
        // kernel (weights-once; the 24M nt == 1 crossover does not
        // apply in-block) — the reference dots the pad40 q8 activation
        // round-trip, tolerance as in the mmvq parity tests.
        let mut x8 = vec![0u8; nt * (id / 32) * 40];
        for t in 0..nt {
            for blk in 0..id / 32 {
                let base = t * id + blk * 32;
                let mut am = 0f32;
                for j in 0..32 {
                    am = am.max(xs[base + j].abs());
                }
                let dd = am / 127.0;
                let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                let off = (t * (id / 32) + blk) * 40;
                x8[off..off + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
                for j in 0..32 {
                    let q = (xs[base + j] * di).round().clamp(-128.0, 127.0) as i8;
                    x8[off + 4 + j] = q as u8;
                }
            }
        }
        let dq8 = |t: usize, i: usize| -> f32 {
            let off = (t * (id / 32) + i / 32) * 40;
            half::f16::from_le_bytes([x8[off], x8[off + 1]]).to_f32()
                * (x8[off + 4 + (i % 32)] as i8) as f32
        };
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        state
            .matmul_f32_ptr(
                wptr,
                TensorType::Q5_K,
                cb.ptr_of(xb).unwrap(),
                cb.ptr_of(out).unwrap(),
                od,
                id,
                nt,
            )
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();
        // independent dequant reference (mirrors the kernel decode)
        let deq = |r: usize| -> Vec<f32> {
            let mut outv = vec![0f32; id];
            for sp in 0..nsp {
                let blk_off = (r * nsp + sp) * 176;
                let d = half::f16::from_le_bytes([wq[blk_off], wq[blk_off + 1]]).to_f32();
                let dmin = half::f16::from_le_bytes([wq[blk_off + 2], wq[blk_off + 3]]).to_f32();
                let sc: [u8; 12] = wq[blk_off + 4..blk_off + 16].try_into().unwrap();
                let (scales, mins) = crate::block::unpack_q4k_scales(&sc);
                let valid = ((id - sp * 256).min(256) + 31) / 32;
                for sub in 0..valid {
                    let ci = sub >> 1;
                    for l in 0..32 {
                        let qbyte = wq[blk_off + 48 + ci * 32 + l];
                        let nib = if sub & 1 == 1 {
                            qbyte >> 4
                        } else {
                            qbyte & 0xF
                        };
                        let w = nib as f32 + 16.0 * (((wq[blk_off + 16 + l] >> sub) & 1) as f32);
                        outv[sp * 256 + sub * 32 + l] =
                            d * scales[sub] as f32 * w - dmin * mins[sub] as f32;
                    }
                }
            }
            outv
        };
        let mut wants = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let dq = deq(r);
                let mut want = 0f32;
                for i in 0..id {
                    want += dq[i] * dq8(t, i);
                }
                wants[t * od + r] = want;
            }
        }
        let scale = wants.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        for t in 0..nt {
            for r in 0..od {
                assert!(
                    (got[t * od + r] - wants[t * od + r]).abs() < scale * 1e-2,
                    "q5_K [{t}][{r}] {} vs {}",
                    got[t * od + r],
                    wants[t * od + r]
                );
            }
        }
    }
}

// q5_K qs nibble packing: 4 chunks of 32 bytes; chunk ci, byte l:
// low nibble = element l of sub 2ci, high = element l of sub 2ci+1
fn qs_byte(qs: &mut [u8], ci: usize, l: usize, w: u8, hi: bool) {
    if hi {
        qs[ci * 32 + l] |= w << 4;
    } else {
        qs[ci * 32 + l] |= w & 0xF;
    }
}
// q5_K qh layout: byte l, bit sub = the >16 bit of element (sub, l)
fn qh_byte(qh: &mut [u8], l: usize, sub: usize, w: u8) {
    qh[l] |= ((w >> 4) & 1) << sub;
}

/// Q5_0 real-shape isolation: 0.5B q4_k_m was the first model to reach
/// CUDA with Q5_0 weights, and an end-to-end run died with a sticky
/// cudaErrorMisalignedAddress (716). Run every Q5_0 device path at the
/// model's REAL shapes with a sync after each step so the first faulting
/// path is identified exactly (small-shape parity above already proves
/// the math; this test targets shape/alignment coverage).
#[test]
fn cuda_q5_0_realshape_isolation() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let nt = 30usize; // the "Hello" prompt length
    macro_rules! step {
        ($tag:expr, $run:expr) => {{
            // Backend::synchronize() does NOT wait on the stream outside a
            // capture window — use the real state sync so an async fault
            // surfaces HERE, not at the next cudaMalloc.
            eprintln!("[isolation] begin {}", $tag);
            $run;
            cb.state.sync();
            eprintln!("[isolation] end {}", $tag);
        }};
    }
    cb.state.sync(); // baseline: context healthy after CudaBackend::new()
    eprintln!("[isolation] baseline sync done");

    // ── 0. embed bisect: isolate the fault dimension. Parity (6-row
    //    table, ids [0,5,2], nt=3, n_embd=512) is clean; the model-real
    //    (4096-row table, ids [7,1020,2033]) faults. Vary one dimension
    //    at a time: table size, id values. ──
    let n_embd_b = 512usize;
    let nb_b = n_embd_b / 32; // 16
    let build_table = |rows: usize| -> Vec<u8> {
        let mut t = Vec::new();
        for r in 0..rows {
            for ib in 0..nb_b {
                let d = 0.02f32 + 0.003 * ((r * 5 + ib) % 7) as f32;
                t.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((r * 11 + ib * 7 + j * 3) % 32) as u32;
                    let u_hi = ((r * 7 + ib * 5 + j) % 32) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                t.extend_from_slice(&qh.to_le_bytes());
                t.extend_from_slice(&qs);
            }
        }
        t
    };
    let cases: [(&str, usize, &[u32]); 5] = [
        ("a_smalltable_parityids", 6, &[0, 5, 2]),
        ("b_bigtable_parityids", 4096, &[0, 5, 2]),
        ("c_bigtable_bigids", 4096, &[7, 1020, 2033]),
        ("d_smalltable_midids", 16, &[7, 12, 15]),
        ("e_bigtable_row7only", 4096, &[7, 7, 7]),
    ];
    for &(cname, rows, ids) in &cases {
        let tbl = build_table(rows);
        let name = format!("iso_emb_{cname}");
        cb.state.register_weight(&name, &tbl);
        let wptr = cb.state.get_weight_ptr(&name).unwrap();
        let ids_f: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();
        let idb = cb.alloc_buffer(ids.len());
        cb.write_host(idb, &ids_f).unwrap();
        let ob = cb.alloc_buffer(n_embd_b * ids.len());
        step!(format!("embed {cname}"), {
            cb.state
                .embed_rows_on_gpu(
                    TensorType::Q5_0,
                    wptr,
                    cb.ptr_of(idb).unwrap(),
                    cb.ptr_of(ob).unwrap(),
                    n_embd_b,
                    ids.len(),
                    false,
                )
                .unwrap();
        });
    }

    // ── 2-5. prefill + decode matmuls at the model's real matmul shapes
    //    (attn_q 896x896, attn_k 896x128, ffn_gu 896x9728, ffn_down-class
    //    896x4864) through all three prefill paths ──
    let shapes = [
        (896usize, 896usize),
        (896usize, 128usize),
        (4864usize, 896usize),
    ];
    for (si, &(od, id)) in shapes.iter().enumerate() {
        let nb = id / 32;
        let mut wq = Vec::new();
        for r in 0..od {
            for b in 0..nb {
                let d = 0.02f32 + 0.003 * ((r * 5 + b + si) % 7) as f32;
                wq.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((r * 11 + b * 7 + j * 3) % 32) as u32;
                    let u_hi = ((r * 7 + b * 5 + j) % 32) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                wq.extend_from_slice(&qh.to_le_bytes());
                wq.extend_from_slice(&qs);
            }
        }
        let name = format!("iso_w{si}");
        cb.state.register_weight(&name, &wq);
        let wptr = cb.state.get_weight_ptr(&name).unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let xb = cb.alloc_buffer(id * nt);
        cb.write_host(xb, &xs).unwrap();
        let out = cb.alloc_buffer(od * nt);

        // 2. legacy f32-activation kernel (also the decode kernel)
        step!(format!("legacy f32 matmul od={od} id={id} nt={nt}"), {
            cb.state
                .matmul_f32_ptr(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                )
                .unwrap();
        });

        // 3. f16 wmma GEMM path (MINFER_MMQ=0 territory)
        step!(format!("f16 GEMM od={od} id={id} nt={nt}"), {
            cb.state
                .prefill_gemm_f16_inner(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                    false,
                    false,
                )
                .unwrap();
        });

        // 4. MMQ int8 GEMM path (the r60 default)
        step!(format!("MMQ od={od} id={id} nt={nt}"), {
            cb.state
                .prefill_mmq(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                    false,
                    1,
                )
                .unwrap();
        });

        // 5. decode nt==1 through the top dispatch (routing check)
        let x1 = cb.alloc_buffer(id);
        cb.write_host(x1, &xs[..id]).unwrap();
        let o1 = cb.alloc_buffer(od);
        step!(format!("decode dispatch od={od} id={id} nt=1"), {
            cb.state
                .matmul_f32_ptr(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(x1).unwrap(),
                    cb.ptr_of(o1).unwrap(),
                    od,
                    id,
                    1,
                )
                .unwrap();
        });
    }
}

#[test]
fn cuda_scheduler_chain() {
    crate::cuda::CudaState::init();
    let Some(state) = crate::cuda::CudaState::get() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (id_, od, nt) = (64usize, 32usize, 2usize);
    let cw: Vec<f32> = (0..id_).map(|i| 0.8 + (i % 5) as f32 / 10.0).collect();
    let cwb: Vec<u8> = cw.iter().flat_map(|v| v.to_le_bytes()).collect();
    state.register_weight("cw", &cwb);
    let mut cwt = Tensor::from_data(TensorType::F32, &[id_ as i64, 1, 1, 1], cwb);
    cwt.name = "cw".to_string();
    let wf: Vec<f32> = (0..od * id_)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let mut w8b = Vec::new();
    for r in 0..od {
        w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
            &wf[r * id_..(r + 1) * id_],
        ));
    }
    state.register_weight("cw8", &w8b);
    let mut w8t = Tensor::from_data(TensorType::Q8_0, &[id_ as i64, od as i64, 1, 1], w8b);
    w8t.name = "cw8".to_string();
    let bias: Vec<f32> = (0..od).map(|i| (i % 3) as f32 / 7.0).collect();
    let bb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    state.register_weight("cb", &bb);
    let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], bb);
    bt.name = "cb".to_string();

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let n1 = b.rms_norm(x, Some(&cwt), 1e-5);
    let m = b.matmul(n1, &w8t, Some(&bt));
    let s = b.silu(m);
    b.output(s);
    let mut g = b.build();

    // Full pipeline: assign → alloc → fill → execute (no fusion needed).
    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = crate::graph::scheduler::BackendScheduler::new();
    sched.assign_backends(&mut g, &alloc);
    for (i, nd) in g.nodes.iter().enumerate() {
        assert_eq!(
            nd.backend,
            Some(crate::graph::Backend::CUDA),
            "node {i} ({})",
            nd.name
        );
    }
    alloc.alloc_graph(&g).unwrap();
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|i| ((i * 97) % 21) as f32 / 5.0 - 2.0)
        .collect();
    alloc.fill_input(&g, "x", &xs).unwrap();
    sched.execute(&g, &mut alloc).unwrap();

    // Host reference: rms → dequant matmul + bias → silu
    let mut rmsd = vec![0f32; id_ * nt];
    for t in 0..nt {
        crate::vec_ops::rms_norm_fused_f32(
            id_,
            &mut rmsd[t * id_..(t + 1) * id_],
            &xs[t * id_..(t + 1) * id_],
            &cw,
            1e-5,
        );
    }
    let mut dq = vec![0f32; od * id_];
    crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8t, &mut dq, id_);
    let mut mm = vec![0f32; od * nt];
    for t in 0..nt {
        for r in 0..od {
            let mut acc = 0f32;
            for i in 0..id_ {
                acc += dq[r * id_ + i] * rmsd[t * id_ + i];
            }
            mm[t * od + r] = acc + bias[r];
        }
    }
    let mut want = vec![0f32; od * nt];
    crate::vec_ops::vec_silu_f32(od * nt, &mut want, &mm);
    let got = alloc.copy_to_cpu(s).unwrap();
    let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
    assert_close("scheduler chain", &got, &want, scale * 1e-3);
}

// ─── Phase 7d: CUDA Graph capture/replay ─────────────────────

/// x, y → silu(x) + y: a weightless all-CUDA graph exercising the
/// capture/replay bookkeeping without model weights.
fn replay_graph() -> crate::graph::ComputeGraph {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [8, 1, 1, 1], DType::F32);
    let y = b.input("y", [8, 1, 1, 1], DType::F32);
    let s = b.silu(x);
    let o = b.add(s, y);
    b.output(o);
    b.build()
}

fn replay_alloc(graphs_enabled: bool) -> GraphAllocator {
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_cuda(), "cuda device required");
    if !graphs_enabled {
        alloc.cuda_mut().unwrap().set_graphs_enabled_for_test(false);
    }
    alloc
}

fn replay_step(
    sched: &BackendScheduler,
    graph: &crate::graph::ComputeGraph,
    alloc: &mut GraphAllocator,
    seed: f32,
) -> Vec<f32> {
    let xs: Vec<f32> = (0..8).map(|i| seed + i as f32).collect();
    let ys: Vec<f32> = (0..8).map(|i| (seed * 0.5) - i as f32).collect();
    alloc.fill_input(graph, "x", &xs).unwrap();
    alloc.fill_input(graph, "y", &ys).unwrap();
    sched.execute(graph, alloc).unwrap();
    alloc.copy_to_cpu(graph.outputs[0]).unwrap()
}

/// Warmup → capture → replay must be bit-identical to pure direct
/// launches for every step (llama.cpp's core replay guarantee).
#[test]
fn cuda_graph_replay_bit_parity() {
    // 8m: serialize against other tests' stream users — capture on the shared
    // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = BackendScheduler::new();
    let mut g_cap = replay_graph();
    let mut g_ref = replay_graph();
    let mut cap = replay_alloc(true);
    let mut refr = replay_alloc(false);
    sched.assign_backends(&mut g_cap, &cap);
    sched.assign_backends(&mut g_ref, &refr);
    cap.alloc_graph(&g_cap).unwrap();
    refr.alloc_graph(&g_ref).unwrap();

    for step in 0..5u32 {
        let seed = 10.0 + 10.0 * step as f32;
        let got = replay_step(&sched, &g_cap, &mut cap, seed);
        let want = replay_step(&sched, &g_ref, &mut refr, seed);
        assert_eq!(
            got, want,
            "step {step}: replay path diverged from direct launches"
        );
    }
    // steps 1-2 direct, step 3 captured, steps 4-5 replayed
    assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);
}

/// 8g①: a prefill-shaped graph (any matmul with nt > 1) must not open a
/// capture window while prefill capture is OFF — the R3-B default is ON,
/// so this exercises the opt-out (set_prefill_capture_for_test(false)
/// standing in for `MINFER_NO_PREFILL_CAPTURE=1`): no capture even after
/// 3+ executions of the same (uid, range).
#[test]
fn cuda_prefill_shaped_graph_never_captures() {
    // 8m: serialize against other tests' stream users — capture on the shared
    // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut g = {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 8, 1, 1], DType::F32);
        let y = b.input("y", [8, 8, 1, 1], DType::F32);
        let s = b.silu(x);
        let a = b.add(s, y);
        let wb: Vec<u8> = (0..32)
            .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
            .collect();
        let mut w = Tensor::from_data(crate::tensor::TensorType::F32, &[8, 4, 1, 1], wb);
        w.name = "w".to_string();
        let o = b.matmul(a, &w, None);
        b.output(o);
        b.build()
    };
    assert_eq!(g.capture_nt_hint(), Some(8), "prefill-shaped hint");

    let sched = BackendScheduler::new();
    let mut cap = replay_alloc(true);
    cap.cuda_mut().unwrap().set_prefill_capture_for_test(false);
    sched.assign_backends(&mut g, &mut cap);
    cap.cuda_mut().unwrap().state.register_weight(
        "w",
        &(0..32)
            .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    cap.alloc_graph(&g).unwrap();

    for step in 0..4u32 {
        let seed = 3.0 + 7.0 * step as f32;
        let xs: Vec<f32> = (0..64).map(|i| seed + i as f32).collect();
        let ys: Vec<f32> = (0..64).map(|i| seed * 0.25 - i as f32).collect();
        cap.fill_input(&g, "x", &xs).unwrap();
        cap.fill_input(&g, "y", &ys).unwrap();
        sched.execute(&g, &mut cap).unwrap();
    }
    let cb = cap.cuda_mut().unwrap();
    assert_eq!(
        cb.captured_count(),
        0,
        "prefill-shaped graph must never be captured"
    );
    assert!(cb.capturing.is_none());
}

/// 8i-1: MULTI-SPLIT capture. A CUDA op → CPU op → CUDA op graph yields
/// two CUDA splits; each must capture and replay independently with
/// bit-identical results vs pure direct launches (per-split capture is
/// supported but 7d's parity only covered single-split graphs).
#[test]
fn cuda_multisplit_capture_bit_parity() {
    // 8m: serialize against other tests' stream users — capture on the shared
    // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // Softmax has no CUDA kernel (stays on CPU) → forces a split between
    // two CUDA segments.
    let mut b = GraphBuilder::new();
    let x = b.input("x", [8, 1, 1, 1], DType::F32);
    let y = b.input("y", [8, 1, 1, 1], DType::F32);
    let s = b.silu(x);
    let sm = b.softmax(s, 0);
    let o = b.add(sm, y);
    b.output(o);
    let mut g_cap = b.build();

    let mut b = GraphBuilder::new();
    let x = b.input("x", [8, 1, 1, 1], DType::F32);
    let y = b.input("y", [8, 1, 1, 1], DType::F32);
    let s = b.silu(x);
    let sm = b.softmax(s, 0);
    let o = b.add(sm, y);
    b.output(o);
    let mut g_ref = b.build();

    let sched = BackendScheduler::new();
    let mut cap = replay_alloc(true);
    let mut refr = replay_alloc(false);
    sched.assign_backends(&mut g_cap, &cap);
    sched.assign_backends(&mut g_ref, &refr);
    cap.alloc_graph(&g_cap).unwrap();
    refr.alloc_graph(&g_ref).unwrap();

    for step in 0..5u32 {
        let seed = 5.0 + 3.0 * step as f32;
        let got = replay_step(&sched, &g_cap, &mut cap, seed);
        let want = replay_step(&sched, &g_ref, &mut refr, seed);
        assert_eq!(
            got, want,
            "step {step}: multi-split replay diverged from direct launches"
        );
    }
    assert_eq!(
        cap.cuda_mut().unwrap().captured_count(),
        2,
        "both CUDA splits must be captured"
    );
}

/// R3-B: the prefill-capture gate defaults ON (8g②'s opt-in flipped).
/// The OFF path is covered by cuda_prefill_shaped_graph_never_captures.
#[test]
fn cuda_prefill_capture_defaults_on() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let cb = CudaBackend::new().expect("backend after device init");
    assert!(cb.prefill_capture, "prefill capture must default ON (R3-B)");
}

/// 8g②: prefill capture — with the gate ON, a repeated identical-nt
/// prefill-shaped graph captures after the 3-run protocol and replays
/// BIT-IDENTICAL to direct launches, at both pp16 and pp300 (the
/// ~437-node real-prefill scale). R3-B: the gate now defaults ON (the
/// set call below is kept as an explicit statement of intent).
#[test]
fn cuda_prefill_capture_bit_parity_pp16_pp300() {
    // 8m: serialize against other tests' stream users — capture on the shared
    // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let build = |nt: usize| -> crate::graph::ComputeGraph {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, nt, 1, 1], DType::F32);
        let y = b.input("y", [8, nt, 1, 1], DType::F32);
        let s = b.silu(x);
        let a = b.add(s, y);
        let wb: Vec<u8> = (0..32)
            .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
            .collect();
        let mut w = Tensor::from_data(crate::tensor::TensorType::F32, &[8, 4, 1, 1], wb);
        w.name = "w".to_string();
        let o = b.matmul(a, &w, None);
        b.output(o);
        b.build()
    };

    let sched = BackendScheduler::new();
    for nt in [16usize, 300usize] {
        let mut g_cap = build(nt);
        let mut g_ref = build(nt);
        let mut cap = replay_alloc(true);
        let mut refr = replay_alloc(false);
        cap.cuda_mut().unwrap().set_prefill_capture_for_test(true);
        let wb: Vec<u8> = (0..32)
            .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
            .collect();
        cap.cuda_mut().unwrap().state.register_weight("w", &wb);
        refr.cuda_mut().unwrap().state.register_weight("w", &wb);
        sched.assign_backends(&mut g_cap, &cap);
        sched.assign_backends(&mut g_ref, &refr);
        cap.alloc_graph(&g_cap).unwrap();
        refr.alloc_graph(&g_ref).unwrap();

        for step in 0..5u32 {
            let seed = 1.0 + 2.0 * step as f32;
            let xs: Vec<f32> = (0..8 * nt).map(|i| seed + (i % 9) as f32).collect();
            let ys: Vec<f32> = (0..8 * nt).map(|i| seed * 0.5 - (i % 7) as f32).collect();
            cap.fill_input(&g_cap, "x", &xs).unwrap();
            cap.fill_input(&g_cap, "y", &ys).unwrap();
            refr.fill_input(&g_ref, "x", &xs).unwrap();
            refr.fill_input(&g_ref, "y", &ys).unwrap();
            sched.execute(&g_cap, &mut cap).unwrap();
            sched.execute(&g_ref, &mut refr).unwrap();
            let got = cap.copy_to_cpu(g_cap.outputs[0]).unwrap();
            let want = refr.copy_to_cpu(g_ref.outputs[0]).unwrap();
            assert_eq!(
                got, want,
                "pp{nt} step {step}: prefill replay diverged from direct launches"
            );
        }
        assert_eq!(
            cap.cuda_mut().unwrap().captured_count(),
            1,
            "pp{nt}: the prefill split must be captured exactly once"
        );
    }
}

/// #218: the captured-graph half of the prefill-GEMM dynamic-smem invariant.
///
/// A **>48 KiB** prefill-shaped graph must capture on the 3-run protocol and
/// replay **bitwise-identically** to direct launches, and the opt-in that makes
/// the >48 KiB launch legal must be shown to have run **before** the window
/// opened, never inside it (`gemm_smem_optin_in_capture_count() == 0`). That is
/// the value assertion replacing the deleted eager sweep: if the attribute were
/// not already in force when the window opened, the in-window >48 KiB launch
/// would fail.
///
/// Non-vacuity. The fixture's instantiation is `gemm_f16_nt_kernel_t<128,64,false>`
/// at 57344 B (`MINFER_GEMM_K64=1`, forced by the fresh-process harness), the
/// gate asserts `captured_count() == 1` so an uncaptured path cannot satisfy it,
/// and (in the child) it asserts `opted_in == 0` **before** anything launches.
/// The fresh process is required: the tile env is read once per process and the
/// `cudaFuncSetAttribute` answer sticks to the kernel for the process's life.
///
/// Mutation evidence (rule 3).
/// - Make `gemm_smem_optin` answer `true` without calling `cudaFuncSetAttribute`:
///   the first warmup launch of a >48 KiB dynamic smem then fails, the scheduler
///   returns `Err`, and the child panics (and `opted_in` stays 0).
/// - `MINFER_TEST_CAPTURE_WARMUP=1` (the documented test-only seam) opens the
///   window on the **first** run, so the opt-in happens inside it and
///   `gemm_smem_optin_in_capture_count() == 0` goes red — deterministically,
///   whether or not this driver tolerates an in-window attribute call.
///
/// #223 runs the child with `MINFER_NO_GEMM_PREWARM=1` (the lazy-path-alone
/// control): under the default eager pre-warm the attribute is already in force
/// at context creation, so `opted_in == 0` before the first launch — the
/// precondition this gate is built on — would not be observable. The claim stays
/// the lazy path's: the opt-in must happen in the warmup runs, **before** the
/// window opens, never inside it. The pre-warmed configuration's copy of the
/// guarantee is `issue223_tests`.
#[test]
fn cuda_prefill_smem_optin_is_never_set_inside_a_capture_window() {
    const FILTER: &str = "cuda_prefill_smem_optin_is_never_set_inside_a_capture_window";
    match crate::cuda::test_child::child_phase().as_deref() {
        Some("capture") => big_smem_capture_child(),
        _ => {
            let child = crate::cuda::test_child::run_self(
                FILTER,
                "capture",
                &[("MINFER_NO_GEMM_PREWARM", "1")],
            );
            child.verdict("the lazy-path-alone >48 KiB captured prefill");
        }
    }
}

fn big_smem_capture_child() {
    use crate::cuda::issue218_tests as fx;
    if device().is_none() {
        crate::cuda::test_child::child_skip("no CUDA device");
    }
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    // Preconditions, established by running in a fresh process with this as its
    // only test: the >48 KiB instantiation has never launched, so it is not
    // opted in and no opt-in has happened inside a window.
    fx::assert_optin_preconditions("capture");

    let sched = BackendScheduler::new();
    let mut g_cap = fx::big_smem_prefill_graph();
    let mut g_ref = fx::big_smem_prefill_graph();
    let mut cap = replay_alloc(true);
    let mut refr = replay_alloc(false);
    assert_eq!(
        g_cap.capture_nt_hint(),
        Some(fx::NT),
        "the fixture must be prefill-shaped"
    );
    cap.cuda_mut()
        .unwrap()
        .state
        .register_weight(fx::WEIGHT, &fx::weight_bytes());
    sched.assign_backends(&mut g_cap, &cap);
    sched.assign_backends(&mut g_ref, &refr);
    cap.alloc_graph(&g_cap).unwrap();
    refr.alloc_graph(&g_ref).unwrap();

    // Runs 1-2 direct (the warmup, where the opt-in happens), run 3 opens the
    // capture window, runs 4-5 replay. Every step must be bitwise-equal to the
    // direct-launch reference.
    for step in 0..5u32 {
        let seed = 1.0 + 2.0 * step as f32;
        let xs: Vec<f32> = (0..fx::ID * fx::NT)
            .map(|i| seed + (i % 9) as f32)
            .collect();
        cap.fill_input(&g_cap, "x", &xs).unwrap();
        refr.fill_input(&g_ref, "x", &xs).unwrap();
        sched.execute(&g_cap, &mut cap).unwrap();
        sched.execute(&g_ref, &mut refr).unwrap();
        let got = cap.copy_to_cpu(g_cap.outputs[0]).unwrap();
        let want = refr.copy_to_cpu(g_ref.outputs[0]).unwrap();
        assert_eq!(
            got, want,
            "step {step}: the >48 KiB prefill replay diverged from direct launches"
        );
    }
    assert_eq!(
        cap.cuda_mut().unwrap().captured_count(),
        1,
        "the >48 KiB prefill split must be captured exactly once — an uncaptured \
         path must not be able to satisfy this gate"
    );
    assert_eq!(
        unsafe { crate::cuda::gemm_smem_opted_in(fx::TM, fx::KS, fx::AF32) },
        1,
        "the warmup runs must have opted {} in before the window opened",
        fx::KERNEL
    );
    assert_eq!(
        unsafe { crate::cuda::gemm_smem_optin_in_capture_count() },
        0,
        "the smem opt-in must be performed before a capture window opens, never \
         inside one — this is the load-bearing part of the design"
    );
    crate::cuda::test_child::child_ok();
}

/// Phase 8 review: an execute_node error during an open capture window
/// must ABORT the window (the scheduler propagates before the boundary
/// sync, so nothing else would close it). Driven directly here because
/// no supported model can fail a node mid-capture today.
#[test]
fn cuda_capture_abort_on_error() {
    // 8m: serialize against other tests' stream users — capture on the shared
    // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = BackendScheduler::new();
    let mut g_cap = replay_graph();
    let mut g_ref = replay_graph();
    let mut cap = replay_alloc(true);
    let mut refr = replay_alloc(false);
    sched.assign_backends(&mut g_cap, &cap);
    sched.assign_backends(&mut g_ref, &refr);
    cap.alloc_graph(&g_cap).unwrap();
    refr.alloc_graph(&g_ref).unwrap();

    // open the window via the 3-run protocol WITHOUT executing nodes
    let cb = cap.cuda_mut().unwrap();
    for _ in 0..3 {
        cb.graph_replay_step(7, (0, 1), None);
    }
    // Issue #188: the window is this backend's own — `capturing` is the
    // whole exclusion, there is no process-wide stream lock any more.
    assert!(cb.capturing.is_some(), "3rd run must open a capture window");
    assert!(
        crate::cuda::CudaState::stream_is_capturing(cb.device_stream()),
        "the backend's own stream must be the one in a capture window"
    );

    // the error path: abort, not close
    cb.abort_capture("unit test");
    assert!(cb.capturing.is_none(), "window must be closed");
    assert!(
        !crate::cuda::CudaState::stream_is_capturing(cb.device_stream()),
        "the stream must leave the capture window when it is aborted"
    );
    assert_eq!(
        cb.graphs_mode,
        GraphMode::Disabled,
        "graphs disabled after an aborted window"
    );
    assert_eq!(cb.captured_count(), 0, "aborted window must not be cached");
    assert!(
        !cb.graph_replay_step(7, (0, 1), None),
        "no replay after graphs are disabled"
    );

    // direct execution keeps working after the abort
    let got = replay_step(&sched, &g_cap, &mut cap, 99.0);
    let want = replay_step(&sched, &g_ref, &mut refr, 99.0);
    assert_eq!(got, want, "post-abort direct execution diverged");
}

/// A pool generation change after capture must invalidate the stored exec
/// (conservative: pointers may differ) and re-capture on a later run.
#[test]
fn cuda_graph_recaptures_on_pool_gen_change() {
    // 8m: serialize against other tests' stream users — capture on the shared
    // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = BackendScheduler::new();
    let mut g = replay_graph();
    let mut cap = replay_alloc(true);
    let mut refr = replay_alloc(false);
    sched.assign_backends(&mut g, &cap);
    cap.alloc_graph(&g).unwrap();
    refr.alloc_graph(&g).unwrap();

    for step in 0..3u32 {
        let seed = 1.0 + step as f32;
        let got = replay_step(&sched, &g, &mut cap, seed);
        let want = replay_step(&sched, &g, &mut refr, seed);
        assert_eq!(got, want, "warmup step {step}");
    }
    assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);

    // bump pool_gen behind the backend's back (as a new staging alloc
    // would). Invalidation is lazy: the stale exec is dropped at the next
    // graph_replay call, before it could ever be launched.
    let c = cap.cuda_mut().unwrap();
    let _fresh = Backend::alloc_fresh(c, 64);

    // run 4: graph_replay sees the pool_gen change → drops the exec and
    // runs direct (warmup restarts). Parity holds throughout.
    let got = replay_step(&sched, &g, &mut cap, 4.0);
    let want = replay_step(&sched, &g, &mut refr, 4.0);
    assert_eq!(got, want, "post-invalidation step 4");
    assert_eq!(
        cap.cuda_mut().unwrap().captured_count(),
        0,
        "stale exec must be dropped after pool churn"
    );

    // run 5 direct (warmup 2), run 6 re-captures — parity holds
    for step in 5..7u32 {
        let seed = step as f32;
        let got = replay_step(&sched, &g, &mut cap, seed);
        let want = replay_step(&sched, &g, &mut refr, seed);
        assert_eq!(got, want, "post-invalidation step {step}");
    }
    assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);
}

/// #153: a captured exec was instantiated for one KV layout, so a backend whose
/// tag moves must not replay it. `set_kv_layout` drops the execs eagerly, and the
/// `graph_replay_step` lookup refuses a mismatched tag as a second line of
/// defence; either way the run re-warms and re-captures under the new tag.
///
/// The graph here is weightless (no KV store/attention), so the *kernels* do not
/// change with the tag — what this test pins is the **identity**: which execs are
/// held and under which tag. The real-model gate
/// (`two_cuda_engines_with_different_kv_layouts_run_interleaved`) is where the tag
/// changing the kernels' bytes is asserted.
#[test]
fn cuda_graph_recaptures_on_kv_layout_change() {
    // 8m: serialize against other tests' stream users — capture on the shared
    // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = BackendScheduler::new();
    let mut g = replay_graph();
    let mut cap = replay_alloc(true);
    let mut refr = replay_alloc(false);
    sched.assign_backends(&mut g, &cap);
    cap.alloc_graph(&g).unwrap();
    refr.alloc_graph(&g).unwrap();

    // `replay_alloc` builds an f32 backend; warm up and capture under that tag.
    for step in 0..3u32 {
        let seed = 1.0 + step as f32;
        let got = replay_step(&sched, &g, &mut cap, seed);
        let want = replay_step(&sched, &g, &mut refr, seed);
        assert_eq!(got, want, "warmup step {step}");
    }
    assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);
    assert_eq!(
        cap.cuda_mut().unwrap().captured_layouts(),
        vec![crate::cuda::KV_LAYOUT_F32],
        "the exec must be recorded under the tag it was captured for"
    );

    // `set_kv_layout` (what `GraphAllocator::set_kv_format` calls) invalidates a
    // changed tag eagerly — the exec's kernels were recorded for the old one.
    cap.cuda_mut()
        .unwrap()
        .set_kv_layout(crate::cuda::KV_LAYOUT_Q8_0);
    assert_eq!(
        cap.cuda_mut().unwrap().captured_count(),
        0,
        "a layout change must drop every captured exec"
    );
    assert!(
        cap.cuda_mut().unwrap().graph_runs.is_empty(),
        "and restart the warmup protocol"
    );

    // Run 4 is direct (warmup 1); run 5 direct (warmup 2); run 6 re-captures
    // under the new tag. Parity with the direct-launch reference holds throughout.
    for step in 4..7u32 {
        let seed = step as f32;
        let got = replay_step(&sched, &g, &mut cap, seed);
        let want = replay_step(&sched, &g, &mut refr, seed);
        assert_eq!(got, want, "post-layout-change step {step}");
    }
    assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);
    assert_eq!(
        cap.cuda_mut().unwrap().captured_layouts(),
        vec![crate::cuda::KV_LAYOUT_Q8_0],
        "the re-captured exec must carry the new tag"
    );

    // Second line of defence: bypass the eager clear (poke the field) and prove the
    // lookup itself refuses an exec whose recorded tag no longer matches. The next
    // scheduler step calls `graph_replay_step`, which must destroy the mismatched
    // exec and run direct instead of launching it.
    cap.cuda_mut().unwrap().kv_layout = crate::cuda::KV_LAYOUT_F16;
    let got = replay_step(&sched, &g, &mut cap, 98.0);
    let want = replay_step(&sched, &g, &mut refr, 98.0);
    assert_eq!(
        got, want,
        "mismatched-tag step must still produce the right values"
    );
    assert_eq!(
        cap.cuda_mut().unwrap().captured_count(),
        0,
        "an exec captured for q8_0 must be destroyed, not launched, for an f16 backend"
    );

    // And the direct-launch reference still agrees after all of it.
    let got = replay_step(&sched, &g, &mut cap, 99.0);
    let want = replay_step(&sched, &g, &mut refr, 99.0);
    assert_eq!(got, want, "bottom of the layout-change sequence");
}
#[test]
fn cuda_graph_generation_replay_parity_real_model() {
    use crate::models::qwen2::graph::Qwen2Graph;
    use crate::models::qwen2::Qwen2Model;

    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut p = std::path::PathBuf::from(std::env::var("HOME").unwrap());
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    if !p.exists() {
        eprintln!("skipping: qwen2.5-0.5b q4_0 not cached");
        return;
    }
    // Hold the model-load lock from BEFORE the load through the whole
    // comparison: a parallel test loading a different architecture
    // registers same-named tensors of a different size, which would swap
    // the weight registry underneath these loops and corrupt one of them.
    // (The guard is reentrant — load_model takes it again internally.)
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let gguf = crate::gguf::load_gguf_model(&p).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().unwrap();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    let nt = ids.len();
    // Full model context (32k) would size f32 KV regions at ~800 MB per
    // cache — x3 caches here. 4096 comfortably covers a 200-token decode
    // and keeps the parallel suite's device-memory footprint small.
    let n_ctx = 4096;

    fn generate(
        q2: &Qwen2Model,
        ids: &[u32],
        nt: usize,
        n_ctx: usize,
        steps: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut cache = GraphCache::new();
        let positions: Vec<usize> = (0..nt).collect();
        let mut logits = Qwen2Graph::forward_cached(q2, ids, &positions, 1, n_ctx, &mut cache);
        let mut toks = Vec::with_capacity(steps);
        for step in 0..steps {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            toks.push(next);
            logits = Qwen2Graph::forward_cached(q2, &[next], &[nt + step], 1, n_ctx, &mut cache);
        }
        (toks, logits)
    }

    // loop 1: warmup → capture → replay across the steps (200 tokens)
    let (toks1, last1) = generate(q2, &ids, nt, n_ctx, 200);
    // loop 2: everything replays (fresh cache, fresh backend bookkeeping)
    let (toks2, last2) = generate(q2, &ids, nt, n_ctx, 200);
    assert_eq!(toks1, toks2, "replay generation diverged from mixed-mode");
    assert_eq!(last1, last2, "final-step logits diverged bitwise");

    // loop 3: graphs force-disabled — the direct-launch reference. The
    // allocator must get its CUDA backend (and the disabled flag) before
    // the first forward_cached call, which would otherwise create it.
    let mut cache3 = GraphCache::new();
    cache3.alloc().disable_graphs_for_test();
    let positions: Vec<usize> = (0..nt).collect();
    let mut logits3 = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, n_ctx, &mut cache3);
    let mut toks3 = Vec::with_capacity(200);
    for step in 0..200 {
        let next = logits3
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        toks3.push(next);
        logits3 = Qwen2Graph::forward_cached(q2, &[next], &[nt + step], 1, n_ctx, &mut cache3);
    }
    assert_eq!(
        toks1, toks3,
        "graph-captured generation diverged from direct launches"
    );
}

/// Issue #185: the F5 host-stall counter is **per backend**, not the
/// process-wide `cuda::stream_sync_count()`. Two backends in one process each
/// count only their own `CudaState::sync` calls, so a gate can read a delta
/// attributable to its own workload even when other tests are syncing on the
/// shared singleton.
///
/// Mutation evidence (rule 3): make `CudaBackend::stream_sync_count` return
/// `crate::cuda::stream_sync_count()` instead of `self.stream_syncs`, and the
/// second assertion goes red — the process-wide total moves for both.
#[test]
fn stream_sync_counts_are_per_backend_not_process_wide() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut a = GraphAllocator::new();
    let mut b = GraphAllocator::new();
    assert!(a.enable_cuda(), "a CUDA device answers the first probe");
    assert!(b.enable_cuda(), "a CUDA device answers the second probe");
    let a_before = a.cuda().unwrap().stream_sync_count();
    let b_before = b.cuda().unwrap().stream_sync_count();
    // `synchronize()` is what a split boundary calls; it ends in
    // `CudaState::sync` (through the backend's own counter).
    a.cuda_mut().unwrap().synchronize();
    assert_eq!(
        a.cuda().unwrap().stream_sync_count(),
        a_before + 1,
        "the syncing backend must count its own stall"
    );
    assert_eq!(
        b.cuda().unwrap().stream_sync_count(),
        b_before,
        "a sync on one backend must not move another backend's counter"
    );
}
