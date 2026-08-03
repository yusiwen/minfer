// Isolation test for kernel_q4_0_mm_f32: run the GEMM on the same input twice
// and verify determinism + correctness vs a scalar CPU reference.
// macOS only (Metal). Ignored otherwise.
#![cfg(target_os = "macos")]

const Q4B: usize = 18;

// GGUF Q4_0 block layout: byte j low nibble = element j, byte j high nibble = element j+16.
fn dq(blk: &[u8], k: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
    let nib = if k < 16 { blk[2 + k] & 0x0F } else { blk[2 + (k - 16)] >> 4 };
    d * (nib as f32 - 8.0)
}

fn cpu_q4_0_mm(weights: &[u8], acts: &[f32], od: usize, id: usize, nt: usize) -> Vec<f32> {
    let nblk = id / 32;
    let mut out = vec![0.0f32; nt * od];
    for t in 0..nt {
        for o in 0..od {
            let mut acc = 0.0f32;
            for b in 0..nblk {
                let blk = &weights[(o * nblk + b) * Q4B..(o * nblk + b + 1) * Q4B];
                for k in 0..32 {
                    acc += dq(blk, k) * acts[t * id + b * 32 + k];
                }
            }
            out[t * od + o] = acc;
        }
    }
    out
}

#[test]
fn gemm_isolation() {
    let device = metal::Device::system_default().expect("no metal device");
    let src = include_str!("../src/metal.metal");
    let opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(src, &opts)
        .unwrap_or_else(|e| panic!("shader compile: {e}"));
    let f = lib.get_function("kernel_q4_0_mm_f32", None).unwrap();
    let pl = device.new_compute_pipeline_state_with_function(&f).unwrap();

    let (od, id) = (896usize, 896usize); // real Qwen2.5-0.5B attention dims
    let nblk = id / 32;

    // deterministic Q4_0 weights
    let mut weights = vec![0u8; od * nblk * Q4B];
    for (i, chunk) in weights.chunks_mut(Q4B).enumerate() {
        let d = 0.05 + (i % 7) as f32 * 0.01;
        chunk[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for b in 0..16 {
            let lo = ((i * 3 + b * 5) % 15) as u8;
            let hi = ((i * 7 + b * 3) % 15) as u8;
            chunk[2 + b] = lo | (hi << 4);
        }
    }
    // deterministic f32 activations (size for max nt)
    let mut acts = vec![0.0f32; 33 * id];
    for (i, v) in acts.iter_mut().enumerate() {
        *v = ((i as f32) * 0.37).sin() * 0.5;
    }

    let wb = device.new_buffer_with_data(
        weights.as_ptr() as *const _,
        (weights.len()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let run = |cmdq: &metal::CommandQueue, nt: usize| -> (Vec<f32>, Vec<f32>) {
        let xb = device.new_buffer_with_data(
            acts.as_ptr() as *const _,
            (acts.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out = device.new_buffer(
            (nt * od * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let cb = cmdq.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&xb), 0);
        enc.set_buffer(2, Some(&out), 0);
        let mm_p = [od as i32, id as i32, nt as i32];
        enc.set_bytes(3, 12, mm_p.as_ptr() as *const _);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(
            metal::MTLSize { width: ((nt + 31) / 32) as u64, height: ((od + 63) / 64) as u64, depth: 1 },
            metal::MTLSize { width: 32, height: 4, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ptr = out.contents() as *const f32;
        let data = unsafe { std::slice::from_raw_parts(ptr, nt * od) }.to_vec();
        let ref_out = cpu_q4_0_mm(&weights, &acts, od, id, nt);
        (data, ref_out)
    };

    let cmdq = device.new_command_queue();
    for &nt in &[12usize, 30usize, 32usize, 33usize] {
        let (r1, ref_out) = run(&cmdq, nt);
        let (r2, _) = run(&cmdq, nt);
        let maxdiff_gg = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let maxdiff_ref = r1.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let nan1 = r1.iter().filter(|v| v.is_nan()).count();
        // half-precision sa/sb staging -> allow ~2e-3 absolute tolerance
        let ok_det = maxdiff_gg == 0.0;
        let ok_corr = maxdiff_ref < 2.5e-3 && nan1 == 0;
        println!("nt={nt}: deterministic={ok_det} correct={ok_corr} (maxdiff_gg={maxdiff_gg:.1e} maxdiff_ref={maxdiff_ref:.1e} nan={nan1})");
        assert!(ok_det, "nt={nt}: GEMM non-deterministic (maxdiff_gg={maxdiff_gg:.1e})");
        assert!(ok_corr, "nt={nt}: GEMM wrong vs CPU (maxdiff_ref={maxdiff_ref:.1e} nan={nan1})");
    }
}

/// Row-major weight concat along the output (row) dimension must make a fused
/// matmul equivalent to the separate matmuls — the layout assumption behind the
/// fused QKV/FFN gate+up decode path (loader concat_rows + one matmul at od =
/// oq+ok+ov with a shared input read). Pure CPU check, no Metal needed.
#[test]
fn qkv_row_concat_layout() {
    let (id, oq, ok, ov) = (32usize, 64usize, 16usize, 16usize);
    let synth = |od: usize| -> Vec<u8> {
        let nblk = id / 32;
        let mut w = vec![0u8; od * nblk * Q4B];
        for (i, chunk) in w.chunks_mut(Q4B).enumerate() {
            let d = 0.03 + (i % 5) as f32 * 0.01;
            chunk[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for b in 0..16 {
                let lo = ((i * 3 + b * 5) % 15) as u8;
                let hi = ((i * 7 + b * 3) % 15) as u8;
                chunk[2 + b] = lo | (hi << 4);
            }
        }
        w
    };
    let (wq, wk, wv) = (synth(oq), synth(ok), synth(ov));
    let mut concat = Vec::new();
    concat.extend_from_slice(&wq);
    concat.extend_from_slice(&wk);
    concat.extend_from_slice(&wv);
    let acts: Vec<f32> = (0..id).map(|i| ((i as f32) * 0.7).sin() * 0.5).collect();
    let fused = cpu_q4_0_mm(&concat, &acts, oq + ok + ov, id, 1);
    let mut sep = cpu_q4_0_mm(&wq, &acts, oq, id, 1);
    sep.extend(cpu_q4_0_mm(&wk, &acts, ok, id, 1));
    sep.extend(cpu_q4_0_mm(&wv, &acts, ov, id, 1));
    let maxdiff = fused.iter().zip(&sep).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(maxdiff < 1e-6, "row-major QKV concat mismatch vs separate matmuls: {maxdiff}");
}
