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

/// CPU dequant for Q4_1: d(2) + m(2) + qs(u8,16), unsigned + m.
fn dq_q4_1(blk: &[u8], k: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
    let m = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])));
    let b = blk[4 + k % 16];
    let q = if k < 16 { (b & 0x0F) as i32 } else { ((b >> 4) & 0x0F) as i32 };
    d * (q as f32) + m
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
/// element value = d * sc[e/16] * (q - 32). q indexing matches avx2.rs dequant.
fn dq_q6_k(blk: &[u8], e: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])));
    let sc = blk[192 + e / 16] as i8 as f32;
    let half = e / 128;   // 0 or 1 (two 128-element halves)
    let o = e % 128;
    let ql_idx = half * 64 + if o < 64 { o } else { o - 64 };
    let ql_b = blk[ql_idx] as i32;
    let qh_b = blk[128 + half * 32 + (o % 32)] as i32;
    let (q, shift) = match o {
        0..32 => (ql_b & 0x0F, 0),
        32..64 => (ql_b & 0x0F, 2),
        64..96 => ((ql_b >> 4) & 0x0F, 4),
        _ => ((ql_b >> 4) & 0x0F, 6),
    };
    let q_val = q | (((qh_b >> shift) & 3) << 4);
    d * sc * ((q_val - 32) as f32)
}

/// get_scale_min_k4_just2 (llama port): scale/min pair from the 12-byte scales array.
fn scale_min_k4_just2(j: usize, k: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j + 0 + k] & 63, q[j + 4 + k] & 63)
    } else {
        ((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2), (q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))
    }
}

/// CPU dequant for Q4_K (d+dmin+scales(12)+qs(128)=144B): element e = dl*(qs nibble)-ml.
fn dq_q4_k(blk: &[u8], e: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
    let dmin = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])));
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    let il = e / 16;
    let i = e % 16;
    let is = (il / 4) * 2;
    let q_off = (il / 4) * 32 + 16 * (il & 1);
    let il2 = il & 3;
    let (s, m) = scale_min_k4_just2(is, il2 / 2, scales);
    let dsc = if il2 < 2 { d } else { d / 16.0 };
    let dl = dsc * s as f32;
    let ml = dmin * m as f32;
    let mask = if il2 < 2 { 0x0F } else { 0xF0 };
    dl * ((qs[q_off + i] & mask) as f32) - ml
}

/// CPU dequant for Q5_K (d+dmin+scales(12)+qh(32)+qs(128)=176B): + qh high bit.
fn dq_q5_k(blk: &[u8], e: usize) -> f32 {
    let d = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
    let dmin = f32::from(half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])));
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let qs = &blk[48..176];
    let il = e / 16;
    let i = e % 16;
    let is = (il / 4) * 2;
    let q_off = (il / 4) * 32 + 16 * (il & 1);
    let qh_off = 16 * (il & 1);
    let ul = 1u8 << (il / 2);
    let il2 = il & 3;
    let (s, m) = scale_min_k4_just2(is, il2 / 2, scales);
    let dsc = if il2 < 2 { d } else { d / 16.0 };
    let dl = dsc * s as f32;
    let ml = dmin * m as f32;
    let mask = if il2 < 2 { 0x0F } else { 0xF0 };
    let qh_val = if il2 < 2 { 16.0 } else { 256.0 };
    let hi = if qh[qh_off + i] & ul != 0 { qh_val } else { 0.0 };
    dl * ((qs[q_off + i] & mask) as f32 + hi) - ml
}

/// Isolation test for the non-Q4_0 simdgroup GEMMs (Q8_0/Q5_0/Q5_1/Q6_K/Q4_K/Q5_K):
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
    let quants: [(&str, usize, usize, fn(&[u8], usize) -> f32); 8] = [
        ("kernel_q4_0_mm_f32", 18, 32, dq),
        ("kernel_q4_1_mm_f32", 20, 32, dq_q4_1),
        ("kernel_q8_0_mm_f32", 34, 32, dq_q8_0),
        ("kernel_q5_0_mm_f32", 22, 32, dq_q5_0),
        ("kernel_q5_1_mm_f32", 24, 32, dq_q5_1),
        ("kernel_q4_k_mm_f32", 144, 256, dq_q4_k),
        ("kernel_q5_k_mm_f32", 176, 256, dq_q5_k),
        ("kernel_q6_k_mm_f32", 210, 256, dq_q6_k),
    ];

    for &(kname, bbytes, belem, dq) in &quants {
        let id = if belem == 256 { 512usize } else { 896usize };
        let nblk = id / belem;
        let mut weights = vec![0u8; od * nblk * bbytes];
        for (i, chunk) in weights.chunks_mut(bbytes).enumerate() {
            let d = 0.05 + (i % 7) as f32 * 0.01;
            chunk[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            if kname == "kernel_q5_1_mm_f32" || kname == "kernel_q4_1_mm_f32" {
                chunk[2..4].copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
            }
            // Per-quant fill: header fields get valid small values, payload gets PRNG.
            match kname {
                "kernel_q4_k_mm_f32" | "kernel_q5_k_mm_f32" => {
                    chunk[2..4].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes()); // dmin
                    for b in 4..16 { chunk[b] = 0x11; }  // scales[12]: scale=1, min=1 (6-bit)
                }
                "kernel_q6_k_mm_f32" => {
                    for b in 192..208 { chunk[b] = 0x11; } // int8 scales
                    chunk[208..210].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes()); // d at END
                }
                _ => {}
            }
            let (nq, qoff) = match kname {
                "kernel_q4_k_mm_f32" => (128, 16usize),   // qs only
                "kernel_q5_k_mm_f32" => (160, 16usize),   // qh(32) + qs(128)
                "kernel_q6_k_mm_f32" => (192, 0usize),    // ql(128) + qh(64)
                "kernel_q5_1_mm_f32" => (16, 8usize),
                "kernel_q5_0_mm_f32" => (16, 6usize),
                "kernel_q4_0_mm_f32" => (16, 2usize),
                "kernel_q4_1_mm_f32" => (16, 4usize),
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

/// Isolation test for kernel_get_rows_q4_k (GPU embedding lookup for Q4_K tables):
/// deterministic + bit-exact vs the scalar CPU dq_q4_k reference (same arithmetic
/// order as the kernel's dequant_q4_k_16). Exercises the 7B/1.5B Q4_K embedding
/// path that previously fell back to CPU dequant + upload.
#[test]
fn get_rows_q4_k_isolation() {
    let device = metal::Device::system_default().expect("no metal device");
    let src = include_str!("../src/metal.metal");
    let opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(src, &opts)
        .unwrap_or_else(|e| panic!("shader compile: {e}"));

    let ne = 512usize; // embedding dim (2 Q4_K super-blocks); 256-aligned
    let vocab = 7usize;
    let nt = 5usize;
    let ids: [usize; 5] = [3, 1, 5, 0, 6];

    let bbytes = 144usize;
    let nsuper = ne / 256;
    let mut table = vec![0u8; vocab * nsuper * bbytes];
    for (i, chunk) in table.chunks_mut(bbytes).enumerate() {
        let d = 0.05 + (i % 7) as f32 * 0.01;
        chunk[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        chunk[2..4].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes()); // dmin
        for b in 4..16 { chunk[b] = 0x11; } // scales[12]: scale=1, min=1
        let mut seed = i as u64 * 131 + 7;
        for b in 0..128 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            chunk[16 + b] = (seed >> 32) as u8;
        }
    }

    // CPU reference: row ids[t], element k -> dq_q4_k(super_block, k)
    let mut ref_out = vec![0.0f32; nt * ne];
    for t in 0..nt {
        let row = ids[t];
        for s in 0..nsuper {
            let blk = &table[(row * nsuper + s) * bbytes..(row * nsuper + s + 1) * bbytes];
            for k in 0..256 {
                ref_out[t * ne + s * 256 + k] = dq_q4_k(blk, k);
            }
        }
    }

    let f = lib.get_function("kernel_get_rows_q4_k", None).unwrap();
    let pl = device.new_compute_pipeline_state_with_function(&f).unwrap();
    let wb = device.new_buffer_with_data(table.as_ptr() as *const _, table.len() as u64, metal::MTLResourceOptions::StorageModeShared);
    let ids_i: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    let ib = device.new_buffer_with_data(ids_i.as_ptr() as *const _, (ids_i.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let ob = device.new_buffer((nt * ne * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let cmdq = device.new_command_queue();

    let run = || -> Vec<f32> {
        let cb = cmdq.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&ib), 0);
        enc.set_buffer(2, Some(&ob), 0);
        let ne_i = ne as i32;
        let nt_i = nt as i32;
        enc.set_bytes(3, 4, &ne_i as *const i32 as *const _);
        enc.set_bytes(4, 4, &nt_i as *const i32 as *const _);
        let nsb = (ne / 256) * 16;
        enc.dispatch_thread_groups(
            metal::MTLSize { width: ((nt * nsb + 255) / 256) as u64, height: 1, depth: 1 },
            metal::MTLSize { width: 256, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ptr = ob.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, nt * ne) }.to_vec()
    };

    let (r1, r2) = (run(), run());
    let maxdiff_gg = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let maxdiff_ref = r1.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let nan1 = r1.iter().filter(|v| v.is_nan()).count();
    println!("get_rows_q4_k: deterministic={} maxdiff_vs_ref={:.3e} nan={nan1}", maxdiff_gg == 0.0, maxdiff_ref);
    assert!(maxdiff_gg == 0.0, "get_rows_q4_k non-deterministic");
    assert!(maxdiff_ref == 0.0 && nan1 == 0, "get_rows_q4_k wrong vs CPU (maxdiff={maxdiff_ref})");
}
