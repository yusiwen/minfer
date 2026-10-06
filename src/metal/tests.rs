//! `#[cfg(test)] mod tests` for `src/metal.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use objc2_foundation::NSString;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::ptr::NonNull;

/// Regression guard: the Metal shader program (metal.metal) is compiled at
/// RUNTIME by `try_new` — `cargo build` does NOT catch shader errors. A
/// duplicate/missing kernel or a Metal compile error makes `MpsState::init`
/// fall back to CPU silently, which looks like a "GPU throttling" slowdown
/// (2026-08-06: the Q5_0 `block_q5_0_dot_y` redefine bug did exactly this).
/// This test compiles every pipeline and fails if MPS is unavailable.
#[test]
fn metal_pipelines_compile() {
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    assert!(
        MpsState::get().is_some(),
        "MPS unavailable — Metal shader compilation failed (check src/metal.metal for \
         duplicate/missing kernel definitions); the model would run on CPU"
    );
}

/// [#53] / [#122]: Metal answers the E4/E5 device-memory question through the same
/// three-way `DeviceMemory` CUDA does, and its value is the device's own
/// `recommendedMaxWorkingSetSize` — not a hardcoded or guessed number.
///
/// The oracle is the API call itself: the gate asserts the reported `free`/`total`
/// are what the device says, so a mutation that returned a literal (or swapped the
/// two) is red.
///
/// [#53]: https://github.com/yusiwen/minfer/issues/53
/// [#122]: https://github.com/yusiwen/minfer/issues/122
#[test]
fn device_memory_reports_the_recommended_working_set() {
    use crate::graph::allocplan::DeviceMemory;
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active on a Mac");
    let expected = mps.inner.device.recommendedMaxWorkingSetSize() as usize;
    match mps.device_memory() {
        DeviceMemory::Reported { free, total } => {
            assert!(expected > 0, "a real Mac reports a non-zero working set");
            assert_eq!(free, expected, "`free` must be the device's own number");
            assert_eq!(
                total, expected,
                "`total` mirrors `free` (no separate total)"
            );
        }
        other => panic!("expected Reported, got {other:?}"),
    }
}

/// The failure half ([#122]): a forced query failure is `QueryFailed`, **not**
/// `Reported { free: 0 }`. The pure `budget_decision` then keeps the backend usable
/// (weights-only accounting, unbounded) with the real reason named once, and E5's
/// `auto` refuses with that reason instead of planning 0 device blocks.
///
/// Mutation: without the `testfail::guard` chokepoint the outcome is `Reported`, so
/// this gate is red whenever the failure channel is dropped.
///
/// [#122]: https://github.com/yusiwen/minfer/issues/122
#[test]
fn a_forced_device_memory_query_failure_is_not_zero() {
    use crate::graph::allocplan::{budget_decision, DeviceMemory};
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active on a Mac");
    let _arm = crate::testfail::InjectionGuard::arm("metal_device_memory");
    let mem = mps.device_memory();
    let name = match &mem {
        DeviceMemory::QueryFailed { name, .. } => name.clone(),
        other => panic!("a forced failure must be QueryFailed, got {other:?}"),
    };
    assert!(
        name.contains("metal_device_memory"),
        "the reason must name the real site: {name}"
    );
    // #122's rule: a failed query is unbounded weights-only accounting, not a 0 budget.
    let decision = budget_decision(None, &mem);
    assert_eq!(decision.budget, Some(usize::MAX));
    let note = decision.note.expect("a failed query must carry its reason");
    assert!(note.contains("metal_device_memory"), "{note}");
    // E5's `auto` refuses with the real reason; an explicit cap still plans.
    let err = crate::graph::offload::weight_budget(&mem, None).unwrap_err();
    assert!(err.contains("metal_device_memory"), "{err}");
    assert_eq!(
        crate::graph::offload::weight_budget(&mem, Some("64")).unwrap(),
        64 << 20
    );
}

/// Batched-cb bandwidth profile of each nt==1 matmul kernel (decode path).
/// Dispatches the SAME matmul N times in one command buffer (per the
/// 2026-08-03 methodology: a single dispatch is dominated by the ~165 µs
/// cb launch+sync floor — batch dozens before trusting a per-matmul time),
/// then reports GB/s of weight reads. Goal (2026-08-06 #1): find whether any
/// specific matmul (output/QKV/O/GU/down) is far below the ~200 GB/s floor.
#[test]
fn matmul_bandwidth_profile() {
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active for the bandwidth profile");
    let dev = &mps.inner.device;

    // (label, ttype, od, id, batches) — Qwen2.5-0.5B Q4_K_M + 7B Q4_K decode dims.
    let cases: &[(&str, TensorType, usize, usize, usize)] = &[
        (
            "QKV  q5_0 (od=1152,id=896)",
            TensorType::Q5_0,
            1152,
            896,
            400,
        ),
        (
            "O    q5_0 (od=896, id=896)",
            TensorType::Q5_0,
            896,
            896,
            400,
        ),
        (
            "GU   q5_0 (od=9728,id=896)",
            TensorType::Q5_0,
            9728,
            896,
            100,
        ),
        (
            "down q6_K (od=896, id=4864)",
            TensorType::Q6_K,
            896,
            4864,
            200,
        ),
        (
            "out  q8_0 (od=151936,id=896)",
            TensorType::Q8_0,
            151936,
            896,
            6,
        ),
        // Q4_0 kernel (the "fast interleaved-ushort" one) at the SAME small
        // dims — isolates whether the low GB/s is the kernel or the small-od structure.
        (
            "QKV  q4_0 (od=1152,id=896)",
            TensorType::Q4_0,
            1152,
            896,
            400,
        ),
        (
            "GU   q4_0 (od=9728,id=896)",
            TensorType::Q4_0,
            9728,
            896,
            100,
        ),
        (
            "out  q4_0 (od=151936,id=896)",
            TensorType::Q4_0,
            151936,
            896,
            6,
        ),
        // Q5_1 shares Q5_0's qh (variable-shift) handling + an m term.
        (
            "QKV  q5_1 (od=1152,id=896)",
            TensorType::Q5_1,
            1152,
            896,
            400,
        ),
        // Qwen2.5-7B Q4_K decode dims (to-do #7 pre-port baseline):
        // attn_q/attn_output (3584/3584), attn_k (3584/512), ffn_gate/up (18944/3584).
        (
            "7B attn_q q4_K (3584/3584)",
            TensorType::Q4_K,
            3584,
            3584,
            400,
        ),
        (
            "7B attn_k q4_K (3584/512)",
            TensorType::Q4_K,
            3584,
            512,
            400,
        ),
        (
            "7B ffn_g/u q4_K (18944/3584)",
            TensorType::Q4_K,
            18944,
            3584,
            100,
        ),
    ];

    println!("\n=== nt==1 matmul bandwidth profile (batched cb, M4 Pro) ===");
    // Warm up the first pipeline (Q4_0) so the first measured case isn't a cold start.
    {
        let wb = dev
            .newBufferWithLength_options((65536) as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let acts = dev
            .newBufferWithLength_options((4096) as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let out = dev
            .newBufferWithLength_options((65536) as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let cb = mps.cmd_buffer();
        for _ in 0..50 {
            cb.matmul_on_gpu_buf(
                &wb,
                0,
                TensorType::Q4_0,
                &acts,
                &acts,
                0,
                &out,
                2048,
                128,
                1,
            );
        }
        cb.submit().expect("warmup");
    }
    for &(label, ttype, od, id, n) in cases {
        let bq = quant_block_q(ttype);
        let bb = quant_block_bytes(ttype);
        let nblocks = (id + bq - 1) / bq;
        let wbytes = nblocks * bb * od;
        let wb = dev
            .newBufferWithLength_options(
                (wbytes as u64) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        // Deterministic fill: d bytes 0x3333 (finite half), data nibbles 3.
        unsafe {
            std::slice::from_raw_parts_mut(wb.contents().as_ptr() as *mut u8, wbytes).fill(0x33);
        }
        let acts = dev
            .newBufferWithLength_options(
                ((id * 4) as u64) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        unsafe {
            let p = acts.contents().as_ptr() as *mut f32;
            for i in 0..id {
                *p.add(i) = 0.5;
            }
        }
        let out = dev
            .newBufferWithLength_options(
                ((od * 4) as u64) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();

        // Warm this kernel with a discard batch (GPU clock/pipeline ramp-up —
        // the first measurement of a kernel is up to ~4x slow otherwise).
        {
            let cb = mps.cmd_buffer();
            for _ in 0..(n / 2).max(16) {
                cb.matmul_on_gpu_buf(&wb, 0, ttype, &acts, &acts, 0, &out, od, id, 1);
            }
            cb.submit().expect("warmup");
        }

        // Measure TWICE; report the second (warm) value — even after the
        // warmup batch the very first timed cb can still be slow.
        let mut warm_gbs = 0.0f64;
        for rep in 0..2 {
            let cb = mps.cmd_buffer();
            for _ in 0..n {
                cb.matmul_on_gpu_buf(&wb, 0, ttype, &acts, &acts, 0, &out, od, id, 1);
            }
            let t0 = std::time::Instant::now();
            cb.submit().expect("submit");
            let dt = t0.elapsed().as_secs_f64();
            warm_gbs = wbytes as f64 * n as f64 / dt / 1e9;
            if rep == 0 {
                println!(
                    "  {label:<26} (cold run {rep}: {:>5.0} GB/s) — warming…",
                    warm_gbs
                );
            }
        }
        println!(
            "  {label:<26} {:>7.1} MB  x{n:>3} = {:>6.0} MB  {:>5.0} GB/s  (warm)",
            wbytes as f64 / 1e6,
            wbytes as f64 * n as f64 / 1e6,
            warm_gbs
        );
    }
    println!("=== end profile ===");
}

/// Batched-cb per-kernel GPU time profile of the NON-MATMUL decode kernels
/// (rms_norm, add, add_bias, swiglu, rope, store_kv, BSR, attn partial +
/// combine). P0 of the "per-kernel GPU distribution" plan (2026-08-10):
/// the final gap report says the ~1.2 ms non-matmul tail is "~340 kernels at
/// ~4x llama" but the per-kernel distribution was UNKNOWN (xctrace CLI can't
/// give per-kernel durations). This batches each kernel dozens-hundreds of
/// times in ONE command buffer (the 2026-08-03 methodology: single-dispatch
/// timing has a ~165 us cb launch+sync floor; batch to amortize), warms each
/// kernel, measures twice, and reports per-kernel GPU time in us.
#[test]
fn non_matmul_bandwidth_profile() {
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active");
    let dev = &mps.inner.device;

    // Qwen2.5-0.5B decode dims.
    let (ne, nqt, nkt, nf) = (896usize, 896usize, 128usize, 4864usize);
    let (nh, nk, hd) = (14usize, 2usize, 64usize);
    let nkv = 430usize; // long-ish context (matches the -n 512 avg)

    // Shared activation buffers (f32).
    let x = dev
        .newBufferWithLength_options(
            ((ne * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let y = dev
        .newBufferWithLength_options(
            ((ne * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let bqkv = dev
        .newBufferWithLength_options(
            (((nqt + 2 * nkt) * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let w = dev
        .newBufferWithLength_options(
            ((ne * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let g = dev
        .newBufferWithLength_options(
            ((nf * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let u = dev
        .newBufferWithLength_options(
            ((nf * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let bq = dev
        .newBufferWithLength_options(
            ((nqt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let bk = dev
        .newBufferWithLength_options(
            ((nkt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let bv = dev
        .newBufferWithLength_options(
            ((nkt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let kv = dev
        .newBufferWithLength_options(
            ((nkv * nkt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let pos = dev
        .newBufferWithLength_options((4) as usize, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let o = dev
        .newBufferWithLength_options(
            ((ne * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    // finite fill (0.5) so no denormal/NaN paths skew timing
    for b in [&x, &y, &bqkv, &w, &g, &u, &bq, &bk, &bv, &kv, &o] {
        unsafe {
            std::slice::from_raw_parts_mut(
                b.contents().as_ptr() as *mut f32,
                (b.length() / 4) as usize,
            )
            .fill(0.5);
        }
    }
    unsafe {
        std::slice::from_raw_parts_mut(pos.contents().as_ptr() as *mut i32, 1)[0] =
            (nkv - 1) as i32;
    }

    // (label, dispatch closure, batches) — each closure dispatches ONE kernel.
    let cases: Vec<(&str, Box<dyn Fn(&MpsCommandBuffer)>, usize)> = vec![
        (
            "rms_norm 32t  (d=896, 1 row)",
            Box::new(|cb| cb.rms_norm(&x, Some(&w), 0, &y, ne, 1, 1e-6, 0, 0)),
            400,
        ),
        (
            "rms_norm 256t (d=896, 1 row)",
            Box::new(|cb| cb.rms_norm_256(&x, Some(&w), 0, &y, ne, 1, 1e-6, 0, 0)),
            400,
        ),
        (
            "add_f32 (n=896, 256t)",
            Box::new(|cb| cb.add_f32(&x, &y, &x, ne)),
            400,
        ),
        (
            "add_bias_f32 (d=896, 64t)",
            Box::new(|cb| cb.add_bias_f32(&x, &w, 0, ne, 1, 0)),
            400,
        ),
        (
            "swiglu_f32 (n=4864, 256t)",
            Box::new(|cb| cb.swiglu_f32(&g, &u, &g, nf)),
            400,
        ),
        (
            "rope_f32 (q: 14h x 64d)",
            Box::new(|cb| cb.rope_f32(&bqkv, nh, hd, 1, 1e6, 1.0, &pos, 0, 0)),
            400,
        ),
        (
            "store_kv (nkt=128, 1t)",
            Box::new(|cb| cb.store_kv(&bk, &kv, nkt, 1, &pos, 0, false)),
            400,
        ),
        (
            "attn_bsr (q+k+v, 256t)",
            Box::new(|cb| {
                cb.attn_bias_rope_store(
                    &bqkv,
                    &bq,
                    0,
                    &bk,
                    0,
                    &bv,
                    0,
                    &kv,
                    &kv,
                    nqt,
                    nkt,
                    hd,
                    1e6,
                    1.0,
                    (nkv - 1) as i32,
                    0,
                    false,
                )
            }),
            400,
        ),
        // split = partial + combine as a PAIR (2 dispatches/layer, decode path)
        (
            "attn split p+c (c=16)",
            Box::new(|cb| {
                cb.gqa_attn_split_f32(&bqkv, &kv, &kv, &o, &pos, nh, nk, hd, 0.125, 1, 16, false)
            }),
            100,
        ),
    ];

    println!("\n=== nt==1 non-matmul GPU profile (batched cb, M4 Pro) ===");
    // warm the whole pipeline once
    {
        let cb = mps.cmd_buffer();
        for _ in 0..50 {
            cb.rms_norm(&x, Some(&w), 0, &y, ne, 1, 1e-6, 0, 0);
        }
        cb.submit().expect("warmup");
    }

    for (i, (label, dispatch, n)) in cases.iter().enumerate() {
        // warm this kernel (pipeline/clock ramp)
        {
            let cb = mps.cmd_buffer();
            for _ in 0..(n / 2).max(16) {
                dispatch(&cb);
            }
            cb.submit().expect("warmup");
        }
        // median of 3 warm runs (the docs' methodology: batched-cb per-kernel
        // numbers vary ~2x run-to-run due to GPU clock — take the median).
        let mut us: Vec<f64> = Vec::new();
        for _ in 0..3 {
            let cb = mps.cmd_buffer();
            for _ in 0..*n {
                dispatch(&cb);
            }
            let t0 = std::time::Instant::now();
            cb.submit().expect("submit");
            let dt = t0.elapsed().as_secs_f64();
            us.push(dt * 1e6 / *n as f64);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = us[1];
        println!(
            "  [{i}] {label:<26} {:>7.2} us/kernel  (median, n={}, [{:.2},{:.2},{:.2}])",
            med, n, us[0], us[1], us[2]
        );
    }

    // Classic single-pass attention (baseline for the split pair):
    let cases2: Vec<(&str, Box<dyn Fn(&MpsCommandBuffer)>, usize)> = vec![(
        "attn classic (nkv=430)",
        Box::new(|cb| cb.gqa_attn_f32(&bqkv, &kv, &kv, &o, &pos, nh, nk, hd, 0.125, 1, false)),
        100,
    )];
    for (i, (label, dispatch, n)) in cases2.iter().enumerate() {
        {
            let cb = mps.cmd_buffer();
            for _ in 0..(n / 2).max(16) {
                dispatch(&cb);
            }
            cb.submit().expect("warmup");
        }
        let mut us: Vec<f64> = Vec::new();
        for _ in 0..3 {
            let cb = mps.cmd_buffer();
            for _ in 0..*n {
                dispatch(&cb);
            }
            let t0 = std::time::Instant::now();
            cb.submit().expect("submit");
            let dt = t0.elapsed().as_secs_f64();
            us.push(dt * 1e6 / *n as f64);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  [{i}] {label:<26} {:>7.2} us/kernel  (median, n={}, [{:.2},{:.2},{:.2}])",
            us[1], n, us[0], us[1], us[2]
        );
    }
    println!("=== end non-matmul profile ===");
}

/// Correctness of the 256-thread multi-simdgroup rms_norm vs a scalar CPU
/// reference (and vs the 32-thread kernel). P1 (2026-08-10): the multi-
/// simdgroup reduction (shmem + 2 barriers) is the riskiest new piece —
/// must be byte-deterministic before it touches the decode path.
#[test]
fn rms_norm_256_correctness() {
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active");
    let dev = &mps.inner.device;
    let d = 896usize;

    let x = dev
        .newBufferWithLength_options(
            ((d * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let w = dev
        .newBufferWithLength_options(
            ((d * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let y32 = dev
        .newBufferWithLength_options(
            ((d * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let y256 = dev
        .newBufferWithLength_options(
            ((d * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    // Deterministic input: x = sin(i), w = cos(i/7) — exercises varied magnitudes.
    unsafe {
        let xp = x.contents().as_ptr() as *mut f32;
        let wp = w.contents().as_ptr() as *mut f32;
        for i in 0..d {
            *xp.add(i) = (i as f32 * 0.37).sin() * 3.0;
            *wp.add(i) = (i as f32 / 7.0).cos() + 1.0;
        }
    }

    for (label, buf, method) in [("32t", &y32, 0), ("256t", &y256, 1)] {
        let cb = mps.cmd_buffer();
        if method == 0 {
            cb.rms_norm(&x, Some(&w), 0, buf, d, 1, 1e-6, 0, 0);
        } else {
            cb.rms_norm_256(&x, Some(&w), 0, buf, d, 1, 1e-6, 0, 0);
        }
        cb.submit().expect("submit");
    }

    // CPU scalar reference: scale = 1/sqrt(mean(x^2)+eps); y = x*scale*w.
    let xs: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let ws: Vec<f32> = (0..d).map(|i| (i as f32 / 7.0).cos() + 1.0).collect();
    let mean: f32 = xs.iter().map(|v| v * v).sum::<f32>() / d as f32;
    let scale = 1.0f32 / (mean + 1e-6f32).sqrt();
    let mut ref_y = vec![0.0f32; d];
    for i in 0..d {
        ref_y[i] = xs[i] * scale * ws[i];
    }

    for (label, buf) in [("32t", &y32), ("256t", &y256)] {
        let mut got = vec![0.0f32; d];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents().as_ptr() as *const f32,
                got.as_mut_ptr(),
                d,
            );
        }
        let mut maxd = 0.0f32;
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        for i in 0..d {
            maxd = maxd.max((got[i] - ref_y[i]).abs());
            dot += got[i] * ref_y[i];
            na += got[i] * got[i];
        }
        let cos = dot / (na.sqrt() * ref_y.iter().map(|v| v * v).sum::<f32>().sqrt());
        println!("  rms_norm {label}: maxdiff={maxd:.3e} cos={cos:.9}");
        assert!(cos > 0.9999, "rms_norm {label} wrong vs CPU (cos={cos})");
        assert!(maxd < 1e-3, "rms_norm {label} maxdiff {maxd} > 1e-3");
    }
    // 32t vs 256t should be bit-close (same math, different reduction order).
    let mut y32v = vec![0.0f32; d];
    let mut y256v = vec![0.0f32; d];
    unsafe {
        std::ptr::copy_nonoverlapping(y32.contents().as_ptr() as *const f32, y32v.as_mut_ptr(), d);
        std::ptr::copy_nonoverlapping(
            y256.contents().as_ptr() as *const f32,
            y256v.as_mut_ptr(),
            d,
        );
    }
    let maxdd: f32 = (0..d)
        .map(|i| (y32v[i] - y256v[i]).abs())
        .fold(0.0, f32::max);
    println!("  rms_norm 32t vs 256t maxdiff={maxdd:.3e}");
    assert!(maxdd < 1e-3, "32t vs 256t diverge (maxdiff {maxdd})");
}

/// Correctness of the 3-pass parallel prefill attention vs a CPU reference
/// using REAL dumped layer-0 activations (q, k, v). P1.
#[test]
fn attn_parallel_realdata_correctness() {
    let _g = crate::metal::metal_test_lock();
    use std::io::Read;
    let dir = std::env::var("MINFER_TEST_DUMP").unwrap_or_else(|_| "/tmp/dp3".into());
    // The dump files are generated by a debug_dump GPU run; skip (not fail)
    // when they are absent, like the other fixture-dependent tests.
    let (bq_path, bk_path, bv_path) = (
        format!("{dir}/minfer_gpu_dump_layer0_bq.f32"),
        format!("{dir}/minfer_gpu_dump_layer0_bk.f32"),
        format!("{dir}/minfer_gpu_dump_layer0_bv.f32"),
    );
    if !std::path::Path::new(&bq_path).exists()
        || !std::path::Path::new(&bk_path).exists()
        || !std::path::Path::new(&bv_path).exists()
    {
        eprintln!("layer-0 dump files not found in {dir}; skipping realdata attention test");
        return;
    }
    let mut bq = Vec::new();
    std::fs::File::open(bq_path)
        .unwrap()
        .read_to_end(&mut bq)
        .unwrap();
    let bq: Vec<f32> = bq
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut bk = Vec::new();
    std::fs::File::open(bk_path)
        .unwrap()
        .read_to_end(&mut bk)
        .unwrap();
    let bk: Vec<f32> = bk
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut bv = Vec::new();
    std::fs::File::open(bv_path)
        .unwrap()
        .read_to_end(&mut bv)
        .unwrap();
    let bv: Vec<f32> = bv
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (nh, nk, hd, nkt, nqt) = (14usize, 2usize, 64usize, 128usize, 896usize);
    let nt = bq.len() / nqt;
    let nkv = bk.len() / nkt;
    let gqa = nh / nk;
    let scale = 1.0 / (hd as f32).sqrt();
    assert_eq!(nt, 35);
    MpsState::init();
    let mps = MpsState::get().expect("MPS");
    let dev = &mps.inner.device;
    let qb = unsafe {
        dev.newBufferWithBytes_length_options(
            NonNull::new(bq.as_ptr() as *const _ as *mut c_void).unwrap(),
            ((bq.len() * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
    };
    let kb = unsafe {
        dev.newBufferWithBytes_length_options(
            NonNull::new(bk.as_ptr() as *const _ as *mut c_void).unwrap(),
            ((bk.len() * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
    };
    let vb = unsafe {
        dev.newBufferWithBytes_length_options(
            NonNull::new(bv.as_ptr() as *const _ as *mut c_void).unwrap(),
            ((bv.len() * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
    };
    let ob = dev
        .newBufferWithLength_options(
            ((nt * nqt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let pb = dev
        .newBufferWithLength_options(
            ((nt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    unsafe {
        for t in 0..nt {
            (pb.contents().as_ptr() as *mut i32).add(t).write(t as i32);
        }
    }
    let cb = mps.cmd_buffer();
    cb.attn_parallel_prefill(
        &qb, &kb, &vb, &ob, &pb, nkv, nkt, nqt, nt, nh, hd, gqa, scale,
    );
    cb.submit().expect("submit");
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(ob.contents().as_ptr() as *const f32, nt * nqt) }
            .to_vec();
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(ob.contents().as_ptr() as *const f32, nt * nqt) }
            .to_vec();
    let nan = got.iter().filter(|v| !v.is_finite()).count();
    println!("  realdata parallel: nan={nan} of {}", nt * nqt);
    assert!(nan == 0, "realdata parallel produced NaN");
    // CPU reference
    let mut ref_out = vec![0.0f32; nt * nqt];
    let mut scrs = vec![0.0f32; nkv];
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qq = t * nqt + h * hd;
            let vl = (t + 1).min(nkv);
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks_ = kv * nkt + hk * hd;
                let s = (0..hd).map(|d| bq[qq + d] * bk[ks_ + d]).sum::<f32>() * scale;
                scrs[kv] = s;
                if s > mx {
                    mx = s;
                }
            }
            for kv in vl..nkv {
                scrs[kv] = f32::NEG_INFINITY;
            }
            let mut sum = 0.0f32;
            for kv in 0..nkv {
                scrs[kv] = if scrs[kv] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (scrs[kv] - mx).exp()
                };
                sum += scrs[kv];
            }
            for kv in 0..nkv {
                scrs[kv] /= sum;
            }
            let oo = t * nqt + h * hd;
            for d in 0..hd {
                ref_out[oo + d] = 0.0;
            }
            for kv in 0..nkv {
                for d in 0..hd {
                    ref_out[oo + d] += scrs[kv] * bv[kv * nkt + hk * hd + d];
                }
            }
        }
    }
    let maxerr = (0..nt * nqt)
        .map(|i| (got[i] - ref_out[i]).abs())
        .fold(0.0f32, f32::max);
    println!("  realdata parallel maxerr vs CPU: {maxerr:.5}");
    assert!(maxerr < 0.1, "realdata wrong (maxerr {maxerr})");
}

/// End-to-end correctness of the 3-pass parallel prefill attention vs a CPU
/// scalar reference (the exact algorithm in forward.rs::gqa_attn). P1.
#[test]
fn attn_parallel_prefill_correctness() {
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active");
    let dev = &mps.inner.device;
    let (nh, nk, hd, nkt, nqt) = (14usize, 2usize, 64usize, 128usize, 896usize);
    let (nt, nkv_real) = (35usize, 35usize);
    let nkv_p = ((nkv_real + 31) / 32) * 32;
    let gqa = nh / nk;
    let scale = 1.0 / (hd as f32).sqrt();

    // deterministic q [nt][nqt], kv [nkv][nkt]
    let q = dev
        .newBufferWithLength_options(
            ((nt * nqt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let k = dev
        .newBufferWithLength_options(
            ((nkv_real * nkt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let v = dev
        .newBufferWithLength_options(
            ((nkv_real * nkt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let out = dev
        .newBufferWithLength_options(
            ((nt * nqt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let pos = dev
        .newBufferWithLength_options(
            ((nt * 4) as u64) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    unsafe {
        let qp = q.contents().as_ptr() as *mut f32;
        for i in 0..(nt * nqt) {
            *qp.add(i) = ((i as f32) * 0.37).sin() * 1.5;
        }
        let kp = k.contents().as_ptr() as *mut f32;
        for i in 0..(nkv_real * nkt) {
            *kp.add(i) = ((i as f32) * 0.11).cos() * 1.2;
        }
        let vp = v.contents().as_ptr() as *mut f32;
        for i in 0..(nkv_real * nkt) {
            *vp.add(i) = ((i as f32) * 0.23).sin() * 0.9;
        }
        let pp = pos.contents().as_ptr() as *mut i32;
        for t in 0..nt {
            *pp.add(t) = t as i32;
        }
    }

    let cb = mps.cmd_buffer();
    cb.attn_parallel_prefill(
        &q, &k, &v, &out, &pos, nkv_real, nkt, nqt, nt, nh, hd, gqa, scale,
    );
    cb.submit().expect("submit");

    // CPU reference (mirror of forward.rs::gqa_attn)
    let qs: Vec<f32> = (0..nt * nqt)
        .map(|i| ((i as f32) * 0.37).sin() * 1.5)
        .collect();
    let ks: Vec<f32> = (0..nkv_real * nkt)
        .map(|i| ((i as f32) * 0.11).cos() * 1.2)
        .collect();
    let vs: Vec<f32> = (0..nkv_real * nkt)
        .map(|i| ((i as f32) * 0.23).sin() * 0.9)
        .collect();
    let mut ref_out = vec![0.0f32; nt * nqt];
    let mut scrs = vec![0.0f32; nkv_real];
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qq = t * nqt + h * hd;
            let vl = (t + 1).min(nkv_real);
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks_ = kv * nkt + hk * hd;
                let s = (0..hd).map(|d| qs[qq + d] * ks[ks_ + d]).sum::<f32>() * scale;
                scrs[kv] = s;
                if s > mx {
                    mx = s;
                }
            }
            for kv in vl..nkv_real {
                scrs[kv] = f32::NEG_INFINITY;
            }
            let mut sum = 0.0f32;
            for kv in 0..nkv_real {
                scrs[kv] = if scrs[kv] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (scrs[kv] - mx).exp()
                };
                sum += scrs[kv];
            }
            for kv in 0..nkv_real {
                scrs[kv] /= sum;
            }
            let oo = t * nqt + h * hd;
            for d in 0..hd {
                ref_out[oo + d] = 0.0;
            }
            for kv in 0..nkv_real {
                let vbase = kv * nkt + hk * hd;
                for d in 0..hd {
                    ref_out[oo + d] += scrs[kv] * vs[vbase + d];
                }
            }
        }
    }

    let mut got = vec![0.0f32; nt * nqt];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out.contents().as_ptr() as *const f32,
            got.as_mut_ptr(),
            nt * nqt,
        );
    }
    let mut maxerr = 0.0f32;
    for i in 0..nt * nqt {
        maxerr = maxerr.max((got[i] - ref_out[i]).abs());
    }
    println!("  attn_parallel_prefill: maxerr vs CPU {maxerr:.5}");
    assert!(
        maxerr < 0.1,
        "matmul attention wrong vs CPU (maxerr {maxerr})"
    );
}

/// Prefill GEMM (nt=430) throughput — P1 prefill-gap investigation (2026-08-11):
/// minfer pp430 ~1860 t/s vs llama-Metal ~6940 t/s (3.7x). GEMM params match
/// (64x32 tile, 4 sg, both legacy-simdgroup on M4). This measures the GEMM
/// kernel's achieved GB/s at the REAL Q4_K prefill dims to see if it's
/// bandwidth-bound or latency/occupancy-bound vs llama.
#[test]
#[ignore = "heavy Metal throughput profile (~450 MB, ~20 s); timing-sensitive under parallel load — run opt-in: cargo test -- --ignored prefill_gemm_throughput_profile"]
fn prefill_gemm_throughput_profile() {
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active");
    let dev = &mps.inner.device;
    // Qwen2.5-0.5B Q4_K_M prefill dims: attn_q=Q5_0 (od=896,id=896),
    // ffn_up=Q5_0 (od=18944,id=896), ffn_down=Q6_K (od=896,id=4864).
    // 7B Q4_K_M prefill GEMMs (od=ne[1], id=ne[0] from `minfer info`):
    //   q4_K: attn_q/attn_output (3584/3584), attn_k (512/3584), ffn_gate/up (18944/3584)
    //   q6_K: attn_v (512/3584), ffn_down (3584/18944), output (152064/3584)
    // Use the 64x32-tile GEMM (nt>=16) which is what prefill uses.
    let cases: &[(&str, TensorType, usize, usize)] = &[
        (
            "attn_q Q5_0  od=896    id=896   nt=430",
            TensorType::Q5_0,
            896,
            896,
        ),
        (
            "attn_q Q4_0  od=896    id=896   nt=430",
            TensorType::Q4_0,
            896,
            896,
        ),
        (
            "ffn_up Q5_0  od=18944  id=896   nt=430",
            TensorType::Q5_0,
            18944,
            896,
        ),
        (
            "ffn_up Q4_0  od=18944  id=896   nt=430",
            TensorType::Q4_0,
            18944,
            896,
        ),
        (
            "down  Q6_K   od=896    id=4864  nt=430",
            TensorType::Q6_K,
            896,
            4864,
        ),
        // 7B prefill GEMMs (2026-08-18, llama test-backend-ops A/B)
        (
            "7B attn_q Q4_K od=3584   id=3584  nt=430",
            TensorType::Q4_K,
            3584,
            3584,
        ),
        (
            "7B attn_k Q4_K od=512    id=3584  nt=430",
            TensorType::Q4_K,
            512,
            3584,
        ),
        (
            "7B ffn_gu Q4_K od=18944  id=3584  nt=430",
            TensorType::Q4_K,
            18944,
            3584,
        ),
        (
            "7B attn_v Q6_K od=512    id=3584  nt=430",
            TensorType::Q6_K,
            512,
            3584,
        ),
        (
            "7B ffn_down Q6_K od=3584 id=18944 nt=430",
            TensorType::Q6_K,
            3584,
            18944,
        ),
        (
            "7B output Q6_K od=152064 id=3584  nt=430",
            TensorType::Q6_K,
            152064,
            3584,
        ),
    ];
    let nt = 430usize;
    println!("\n=== prefill GEMM throughput (nt=430, batched cb) ===");
    for &(label, ttype, od, id) in cases {
        let bq = quant_block_q(ttype);
        let bb = quant_block_bytes(ttype);
        let nblocks = (id + bq - 1) / bq;
        let wbytes = nblocks * bb * od;
        let wb = dev
            .newBufferWithLength_options(
                (wbytes as u64) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        // Fill with valid finite weights: each block's d (first 2 bytes, fp16)
        // = 1.0 (0x00 0x3C LE), remaining bytes 0x33 (finite nibbles). Avoids
        // the denormal-fp16 slow path that skews GEMM timing.
        unsafe {
            std::slice::from_raw_parts_mut(wb.contents().as_ptr() as *mut u8, wbytes).fill(0x33);
        }
        {
            let p = wb.contents().as_ptr() as *mut u8;
            let row = nblocks * bb;
            for r in 0..od {
                for b in 0..nblocks {
                    let off = (r * row + b * bb) as isize;
                    unsafe {
                        *p.offset(off) = 0x00;
                        *p.offset(off + 1) = 0x3C;
                    }
                }
            }
        }
        let acts = dev
            .newBufferWithLength_options(
                ((id * nt * 4) as u64) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        unsafe {
            std::slice::from_raw_parts_mut(acts.contents().as_ptr() as *mut f32, id * nt).fill(0.5);
        }
        let out = dev
            .newBufferWithLength_options(
                ((od * nt * 4) as u64) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        // warm
        {
            let cb = mps.cmd_buffer();
            for _ in 0..20 {
                cb.quant_matmul_f32_on_gpu_buf(&wb, 0, ttype, &acts, 0, &out, od, id, nt);
            }
            cb.submit().expect("warmup");
        }
        let n = 50;
        let mut us: Vec<f64> = Vec::new();
        for _ in 0..3 {
            let cb = mps.cmd_buffer();
            for _ in 0..n {
                cb.quant_matmul_f32_on_gpu_buf(&wb, 0, ttype, &acts, 0, &out, od, id, nt);
            }
            let t0 = std::time::Instant::now();
            cb.submit().expect("submit");
            let dt = t0.elapsed().as_secs_f64();
            us.push(dt * 1e6 / n as f64);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let per_us = us[1]; // median
        let gbs = wbytes as f64 / (per_us * 1e-6) / 1e9;
        let tflops = 2.0 * od as f64 * id as f64 * nt as f64 / (per_us * 1e-6) / 1e12;
        println!(
            "  {label:<32} {:.1} MB {per_us:>8.1} us => {:>5.0} GB/s  {:>5.2} TFLOPS (warm, n=50)",
            wbytes as f64 / 1e6,
            gbs,
            tflops
        );
    }
}
