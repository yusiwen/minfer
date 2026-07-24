// examples/cuda_graph_diag.rs
// Minimal Rust → FFI → CUDA graph capture diagnostic.
// Tests graph capture from Rust FFI with and without stream reuse.
//
// Usage: cargo run --example cuda_graph_diag

use std::ffi::c_void;

// ─── CUDA Runtime FFI ───────────────────────────────────────────
#[link(name = "cudart")]
extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiate(
        exec: *mut *mut c_void, graph: *mut c_void,
        err_node: *mut c_void, log_buf: *mut c_void, buf_size: usize,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
}

const cudaMemcpyHostToDevice: i32 = 1;
const cudaMemcpyDeviceToHost: i32 = 2;

// ─── Kernel Launch FFI (from cuda_kernels.cu) ───────────────────
extern "C" {
    fn launch_rms_norm_f32(
        x: *const f32, w: *const f32, y: *mut f32,
        d: i32, eps: f32, n: i32, stream: *mut c_void,
    );
    fn launch_add_f32(
        x: *const f32, y: *const f32, z: *mut f32, n: i32, stream: *mut c_void,
    );
    fn launch_quantize_q8_0(
        x: *const f32, y: *mut u8, dim: i32, nt: i32, stream: *mut c_void,
    );
}

// ─── Helpers ────────────────────────────────────────────────────

fn check(msg: &str) {
    let err = unsafe { cudaGetLastError() };
    if err != 0 {
        println!("  [ERR {err:4}] {msg}");
    } else {
        println!("  [OK     ] {msg}");
    }
}

fn alloc_dev_f32(count: usize) -> *mut f32 {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let err = unsafe { cudaMalloc(&mut ptr, count * 4) };
    assert_eq!(err, 0, "cudaMalloc failed");
    unsafe { cudaMemset(ptr, 0, count * 4) };
    ptr as *mut f32
}

// ─── Test: simple kernel capture (like C++ test_01) ─────────────

fn test_simple_kernel(stream: *mut c_void, dummy_a: *mut f32, dummy_b: *mut f32) {
    println!("\n=== Test A: simple rms_norm kernel capture ===");

    for mode in [0i32, 1, 2] {
        let mode_names = ["Global", "ThreadLocal", "Relaxed"];
        println!("  mode={mode} ({})", mode_names[mode as usize]);

        let err = unsafe { cudaStreamBeginCapture(stream, mode) };
        println!("    begin capture: {err}");
        if err != 0 { continue; }

        // Launch rms_norm via FFI (same call path as cuda.rs)
        unsafe {
            launch_rms_norm_f32(
                dummy_a as *const f32, dummy_b as *const f32, dummy_a,
                896, 1e-5, 1, stream,
            );
        }
        check("rms_norm kernel");

        let mut graph: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
        println!("    end capture: {err}  graph={graph:p}");
        if err != 0 || graph.is_null() {
            unsafe { cudaGetLastError(); }
            continue;
        }

        let mut exec: *mut c_void = std::ptr::null_mut();
        let err = unsafe {
            cudaGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)
        };
        println!("    instantiate: {err}");
        if err != 0 {
            unsafe { cudaGraphDestroy(graph); }
            continue;
        }

        let err = unsafe { cudaGraphLaunch(exec, stream) };
        println!("    launch: {err}");
        let err = unsafe { cudaStreamSynchronize(stream) };
        println!("    sync: {err}");

        unsafe {
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        }
    }
}

// ─── Test: simulate prefill → decode flow (key hypothesis H1) ───

fn test_stream_reuse(stream: *mut c_void, dummy_a: *mut f32, dummy_b: *mut f32) {
    println!("\n=== Test B: stream reuse (prefill → decode capture) ===");

    // Simulate prefill: launch some kernels
    println!("  -- simulated prefill --");
    unsafe {
        launch_rms_norm_f32(
            dummy_a as *const f32, dummy_b as *const f32, dummy_a,
            896, 1e-5, 1, stream,
        );
        launch_add_f32(
            dummy_a as *const f32, dummy_b as *const f32, dummy_a,
            896, stream,
        );
    }
    // Sync prefill
    let err = unsafe { cudaStreamSynchronize(stream) };
    let le = unsafe { cudaGetLastError() };
    println!("    after prefill sync: sync_err={err}  last_error={le}");

    // Now try capture on the SAME stream (mimicking decode step)
    println!("  -- simulated decode capture --");
    let err = unsafe { cudaStreamBeginCapture(stream, 2) }; // Relaxed
    println!("    begin capture: {err}");
    if err != 0 {
        let le = unsafe { cudaGetLastError() };
        println!("    cudaGetLastError after begin: {le}");
        return;
    }

    unsafe {
        launch_rms_norm_f32(
            dummy_a as *const f32, dummy_b as *const f32, dummy_a,
            896, 1e-5, 1, stream,
        );
    }
    check("kernel during capture");

    let mut graph: *mut c_void = std::ptr::null_mut();
    let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
    println!("    end capture: {err}  graph={graph:p}");
    if err != 0 || graph.is_null() {
        let le = unsafe { cudaGetLastError() };
        println!("    cudaGetLastError after end: {le}");
        return;
    }

    let mut exec: *mut c_void = std::ptr::null_mut();
    let err = unsafe {
        cudaGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)
    };
    println!("    instantiate: {err}");
    if err != 0 { unsafe { cudaGraphDestroy(graph); }; return; }

    let err = unsafe { cudaGraphLaunch(exec, stream) };
    println!("    launch: {err}");
    let err = unsafe { cudaStreamSynchronize(stream) };
    println!("    sync: {err}");

    unsafe { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
}

// ─── Test: multi-kernel capture (like layer_gpu) via FFI ────────

fn test_multi_kernel(stream: *mut c_void, dummy_a: *mut f32, dummy_b: *mut f32) {
    println!("\n=== Test C: multi-kernel capture (rms_norm + add + quantize) ===");

    let ne = 896;
    let nt = 1;
    let q8_buf = alloc_dev_f32(nt * (ne / 32) * 34 / 4) as *mut u8;

    let err = unsafe { cudaStreamBeginCapture(stream, 2) }; // Relaxed
    println!("    begin capture: {err}");
    if err != 0 { return; }

    // Kernel 1: rms_norm
    unsafe {
        launch_rms_norm_f32(
            dummy_a as *const f32, dummy_b as *const f32, dummy_a,
            ne as i32, 1e-5, nt as i32, stream,
        );
    }
    check("  [1/3] rms_norm");

    // Kernel 2: add
    unsafe {
        launch_add_f32(
            dummy_a as *const f32, dummy_b as *const f32, dummy_b,
            ne as i32, stream,
        );
    }
    check("  [2/3] add");

    // Kernel 3: quantize_q8_0
    unsafe {
        launch_quantize_q8_0(
            dummy_a as *const f32, q8_buf, ne as i32, nt as i32, stream,
        );
    }
    check("  [3/3] quantize_q8_0");

    let mut graph: *mut c_void = std::ptr::null_mut();
    let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
    println!("    end capture: {err}  graph={graph:p}");
    if err != 0 || graph.is_null() {
        let le = unsafe { cudaGetLastError() };
        println!("    cudaGetLastError after end: {le}");
        unsafe { cudaFree(q8_buf as *mut c_void); }
        return;
    }

    let mut exec: *mut c_void = std::ptr::null_mut();
    let err = unsafe {
        cudaGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)
    };
    println!("    instantiate: {err}");
    if err != 0 {
        unsafe { cudaGraphDestroy(graph); }
        return;
    }

    let err = unsafe { cudaGraphLaunch(exec, stream) };
    println!("    launch: {err}");
    let err = unsafe { cudaStreamSynchronize(stream) };
    println!("    sync: {err}");

    unsafe {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaFree(q8_buf as *mut c_void);
    }
}

// ─── Test: cudaMalloc during capture (H3) ───────────────────────

fn test_malloc_during_capture(stream: *mut c_void, dummy_a: *mut f32) {
    println!("\n=== Test D: cudaMalloc during capture ===");

    let err = unsafe { cudaStreamBeginCapture(stream, 2) };
    println!("    begin capture: {err}");
    if err != 0 { return; }

    // Kernel launch
    unsafe {
        launch_rms_norm_f32(
            dummy_a as *const f32, dummy_a as *const f32, dummy_a,
            896, 1e-5, 1, stream,
        );
    }
    check("  kernel launch");

    // cudaMalloc DURING capture
    let mut test_ptr: *mut c_void = std::ptr::null_mut();
    let alloc_err = unsafe { cudaMalloc(&mut test_ptr, 1024) };
    println!("    cudaMalloc during capture: {alloc_err}");

    if !test_ptr.is_null() {
        unsafe { cudaFree(test_ptr); }
    }

    let mut graph: *mut c_void = std::ptr::null_mut();
    let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
    println!("    end capture: {err}");
    if err == 0 && !graph.is_null() {
        unsafe { cudaGraphDestroy(graph); }
    }
    let le = unsafe { cudaGetLastError() };
    println!("    cudaGetLastError after cleanup: {le}");
}

// ─── main ──────────────────────────────────────────────────────

fn main() {
    // ── Init CUDA ──
    let mut count: i32 = 0;
    let err = unsafe { cudaGetDeviceCount(&mut count) };
    if err != 0 || count == 0 {
        eprintln!("No CUDA devices");
        return;
    }

    // Auto-select best GPU (same logic as CudaState::try_new)
    let mut best_dev: i32 = 0;
    let mut best_score: i32 = 0;
    for dev in 0..count {
        let (major, minor) = unsafe {
            let mut mj: i32 = 0;
            let mut mn: i32 = 0;
            cudaDeviceGetAttribute(&mut mj, 75, dev); // CUDA_DEV_ATTR_COMPUTE_MAJOR
            cudaDeviceGetAttribute(&mut mn, 76, dev); // CUDA_DEV_ATTR_COMPUTE_MINOR
            (mj, mn)
        };
        let score = major * 100 + minor;
        println!("Device {dev}: CC {major}.{minor} (score={score})");
        if score > best_score {
            best_score = score;
            best_dev = dev;
        }
    }
    println!("Selected device {best_dev}\n");
    unsafe { cudaSetDevice(best_dev); }

    // ── Create stream ──
    let mut stream: *mut c_void = std::ptr::null_mut();
    let err = unsafe { cudaStreamCreate(&mut stream) };
    assert_eq!(err, 0, "cudaStreamCreate failed");

    // ── Allocate dummy buffers ──
    let ne = 896;
    let dummy_a = alloc_dev_f32(ne);
    let dummy_b = alloc_dev_f32(ne);

    // ── Run tests ──
    test_simple_kernel(stream, dummy_a, dummy_b);
    test_multi_kernel(stream, dummy_a, dummy_b);
    test_stream_reuse(stream, dummy_a, dummy_b);
    test_malloc_during_capture(stream, dummy_a);

    // ── Cleanup ──
    unsafe {
        cudaStreamDestroy(stream);
        cudaFree(dummy_a as *mut c_void);
        cudaFree(dummy_b as *mut c_void);
    }
    println!("\nDone.");
}

// Extra CUDA attribute functions needed for device selection
extern "C" {
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}
