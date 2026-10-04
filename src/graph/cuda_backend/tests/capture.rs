//! CUDA graph capture and replay, the smem opt-in invariant and the per-backend stream counters.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

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
/// Issue #185: the F5 host-stall counter is **per backend**, not process-wide.
/// Two backends in one process each count only their own `CudaState::sync`
/// calls, so a gate can read a delta attributable to its own workload even when
/// other tests are syncing on the shared singleton. The process-wide counter
/// this test was written against was deleted in [#242]; the per-instance
/// property it pins is the surviving one.
///
/// Mutation evidence (rule 3): make `CudaBackend::state_sync` bump a
/// process-wide `static` and `stream_sync_count()` read that static instead of
/// `self.stream_syncs` — the pre-[#185] shape — and the second assertion goes
/// red, because `a`'s sync then moves `b`'s count too.
///
/// [#242]: https://github.com/yusiwen/minfer/issues/242
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
