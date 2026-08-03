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

/// CPU dequant for Q8_0: d(half) + qs(int8[32]).
fn dq_q8_0(blk: &[u8], k: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
    d * (blk[2 + k] as i8 as f32)
}

/// CPU dequant for Q5_0: d(2) + qh(u32,4) + qs(u8,16), signed (val - 16).
/// elem j (0..15) = lo nibble of qs[j] + qh bit j; elem j+16 = hi nibble + qh bit j+16.
fn dq_q5_0(blk: &[u8], k: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
    let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
    let b = blk[6 + k % 16];
    let nib = if k < 16 { (b & 0x0F) as i32 } else { ((b >> 4) & 0x0F) as i32 };
    let v = nib | (((qh >> k) & 1) as i32) << 4;
    d * ((v - 16) as f32)
}

/// CPU dequant for Q5_1: d(2) + m(2) + qh(4) + qs(16), unsigned + m.
fn dq_q5_1(blk: &[u8], k: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
    let m = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])));
    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
    let b = blk[8 + k % 16];
    let nib = if k < 16 { (b & 0x0F) as i32 } else { ((b >> 4) & 0x0F) as i32 };
    let v = nib | (((qh >> k) & 1) as i32) << 4;
    d * (v as f32) + m
}

/// CPU dequant for Q6_K super-block (d at offset 208): ql(128) + qh(64) + sc(16),
/// element value = d * sc[e/16] * (q - 32).
fn dq_q6_k(blk: &[u8], e: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])));
    let sc = blk[192 + e / 16] as i8 as f32;
    // element e within the 256-super-block, interleaved ql/qh layout
    let g = e / 128;         // 0 or 1 (two 128-element halves)
    let l = e % 128;
    let g_sub = l / 64;      // 0 or 1
    let l_sub = l % 64;      // 0..63
    let ql_off = g * 64 + g_sub * 32 + (l_sub % 32);
    let qh_off = g * 32 + (l_sub % 32);
    let qh_b = blk[128 + qh_off] as i32;
    let (q, bit_shift) = match (g_sub, l_sub / 32) {
        (0, 0) => ((blk[ql_off] & 0x0F) as i32, 0),
        (0, 1) => (((blk[ql_off] >> 4) & 0x0F) as i32, 4),
        (1, 0) => ((blk[ql_off] & 0x0F) as i32, 2),
        _ => (((blk[ql_off] >> 4) & 0x0F) as i32, 6),
    };
    let q_val = q | (((qh_b >> bit_shift) & 3) << 4);
    d * sc * ((q_val - 32) as f32)
}

/// Isolation test for the non-Q4_0 simdgroup GEMMs (Q8_0/Q5_0/Q5_1/Q6_K):
/// deterministic + correct vs a scalar CPU reference at prefill dims (nt>=16).
#[test]
fn non_q4_0_gemm_isolation() {
    let device = metal::Device::system_default().expect("no metal device");
    let src = include_str!("../src/metal.metal");
    let opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(src, &opts)
        .unwrap_or_else(|e| panic!("shader compile: {e}"));

    let od = 896usize;
    let nt = 32usize;
    // acts sized for the largest id (nt*1024); Q6_K uses id=512, others id=896.
    let acts: Vec<f32> = (0..nt * 1024).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();

    // (kernel name, block bytes, block elems, dequant fn)
    let quants: [(&str, usize, usize, fn(&[u8], usize) -> f32); 5] = [
        ("kernel_q4_0_mm_f32", 18, 32, dq),
        ("kernel_q8_0_mm_f32", 34, 32, dq_q8_0),
        ("kernel_q5_0_mm_f32", 22, 32, dq_q5_0),
        ("kernel_q5_1_mm_f32", 24, 32, dq_q5_1),
        ("kernel_q6_k_mm_f32", 210, 256, dq_q6_k),
    ];

    for &(kname, bbytes, belem, dq) in &quants {
        let id = if belem == 256 { 512usize } else { 896usize };
        let nblk = id / belem;
        let mut weights = vec![0u8; od * nblk * bbytes];
        for (i, chunk) in weights.chunks_mut(bbytes).enumerate() {
            let d = 0.05 + (i % 7) as f32 * 0.01;
            chunk[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            if kname == "kernel_q5_1_mm_f32" {
                chunk[2..4].copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
            }
            // fill only the quantized payload (never the d / m / qh header bytes)
            let (nq, qoff) = match kname {
                "kernel_q6_k_mm_f32" => (bbytes - 2, 0usize),
                "kernel_q5_1_mm_f32" => (16, 8usize),
                "kernel_q5_0_mm_f32" => (16, 6usize),
                "kernel_q4_0_mm_f32" => (16, 2usize),
                _ => (32, 2usize),
            };
            let mut seed = i as u64 * 131 + 7;
            for b in 0..nq {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                chunk[qoff + b] = (seed >> 32) as u8;
            }
        }

        // CPU reference
        let mut ref_out = vec![0.0f32; nt * od];
        for t in 0..nt {
            for o in 0..od {
                let mut acc = 0.0f32;
                for b in 0..nblk {
                    let blk = &weights[(o * nblk + b) * bbytes..(o * nblk + b + 1) * bbytes];
                    for k in 0..belem {
                        acc += dq(blk, k) * acts[t * id + k + b * belem];
                    }
                }
                ref_out[t * od + o] = acc;
            }
        }

        // run GEMM kernel
        let f = lib.get_function(kname, None).unwrap();
        let pl = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let wb = device.new_buffer_with_data(weights.as_ptr() as *const _, weights.len() as u64, metal::MTLResourceOptions::StorageModeShared);
        let xb = device.new_buffer_with_data(acts.as_ptr() as *const _, (acts.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let ob = device.new_buffer((nt * od * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let cmdq = device.new_command_queue();
        let run = || -> Vec<f32> {
            let cb = cmdq.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            let p = [od as i32, id as i32, nt as i32];
            enc.set_bytes(3, 12, p.as_ptr() as *const _);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(
                metal::MTLSize { width: ((nt + 31) / 32) as u64, height: ((od + 63) / 64) as u64, depth: 1 },
                metal::MTLSize { width: 32, height: 4, depth: 1 },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let ptr = ob.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, nt * od) }.to_vec()
        };
        let (r1, r2) = (run(), run());
        let maxdiff_gg = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let maxdiff_ref = r1.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_abs = ref_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0);
        let nan1 = r1.iter().filter(|v| v.is_nan()).count();
        // relative error handles the f16 sa/sb staging rounding, which scales with
        // the weight magnitude (Q8_0 int8 data is larger than small nibbles)
        println!("{kname}: deterministic={} relerr={:.2e} nan={nan1}", maxdiff_gg == 0.0, maxdiff_ref / max_abs);
        assert!(maxdiff_gg == 0.0, "{kname}: GEMM non-deterministic");
        assert!(maxdiff_ref / max_abs < 5e-3 && nan1 == 0, "{kname}: GEMM wrong vs CPU (relerr={})", maxdiff_ref / max_abs);
    }
}
