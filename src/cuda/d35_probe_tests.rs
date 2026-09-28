//! `#[cfg(test)] mod d35_probe_tests` for `src/cuda.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

fn device() -> Option<&'static CudaState> {
    CudaState::init();
    CudaState::get()
}

fn dev_alloc(bytes: usize) -> *mut std::ffi::c_void {
    let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { cudaMalloc(&mut p, bytes) };
    assert_eq!(err, 0, "cudaMalloc failed");
    p
}

fn h2d_f32(dst: *mut std::ffi::c_void, src: &[f32]) {
    let err = unsafe {
        cudaMemcpy(
            dst,
            src.as_ptr() as *const std::ffi::c_void,
            src.len() * 4,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    };
    assert_eq!(err, 0);
}

fn d2h_f32(src: *mut std::ffi::c_void, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let err = unsafe {
        cudaMemcpy(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            n * 4,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    assert_eq!(err, 0);
    out
}

/// D2H readback of this stream's decode q8 scratch (private-field probe).
/// #188 made the scratch per stream, so the slot is read through the
/// accessor rather than a process-wide `(ptr, size)` mutex.
fn read_q8(st: &CudaState, bytes: usize) -> Vec<u8> {
    let (ptr, size) = st.buf_q8_decode.slot();
    assert!(size >= bytes, "q8 scratch smaller than probe readback");
    let mut out = vec![0u8; bytes];
    let err = unsafe {
        cudaMemcpy(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            ptr.0 as *const std::ffi::c_void,
            bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    assert_eq!(err, 0);
    out
}

/// Deterministic activation with real-model spread (RMSNorm outputs reach
/// ±3 and some 32-blocks are near-zero).
fn gen_acts(n: usize, seed: u64) -> Vec<f32> {
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

/// Valid finite q4_K bytes (od x id), deterministic per (r, ib).
fn gen_q4_k(od: usize, id: usize) -> Vec<u8> {
    let nbe = id / 256;
    let mut b = Vec::with_capacity(od * nbe * 144);
    for r in 0..od {
        for ib in 0..nbe {
            let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
            let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
            b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            for j in 0..12 {
                b.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
            }
            for j in 0..128 {
                let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                b.push(lo | (hi << 4));
            }
        }
    }
    b
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// D3-5 1a bitwise probe: fused-producer q8 epilogues vs the standalone
/// quantize kernel — identical pad40 bytes, identical f32 producer
/// outputs, bit-identical MMVQ outputs through the cache-hit path.
#[test]
fn cuda_embed_rows_q4_1_reference() {
    let Some(st) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = CudaState::model_load_guard();
    let (vocab, d) = (5usize, 64usize);
    let nb = d / 32;
    // synthetic Q4_1 blocks (20B: f16 d, f16 m, 16 nibble bytes)
    let mut w = vec![0u8; vocab * nb * 20];
    for r in 0..vocab {
        for b in 0..nb {
            let off = (r * nb + b) * 20;
            let dbits = 0x3800u16.wrapping_add(((r * 7 + b * 3) as u16) * 64);
            let mbits = 0x3800u16.wrapping_add(((r * 5 + b) as u16) * 32);
            w[off..off + 2].copy_from_slice(&dbits.to_le_bytes());
            w[off + 2..off + 4].copy_from_slice(&mbits.to_le_bytes());
            for j in 0..16usize {
                w[off + 4 + j] = ((j * 17 + r * 13 + b * 29) & 0xFF) as u8;
            }
        }
    }
    st.register_weight("q41_embed_w", &w);
    let wptr = st.get_weight_ptr("q41_embed_w").unwrap();
    // row ids travel as I32-as-f32 bit patterns (graph rule §4)
    let ids: Vec<i32> = vec![0, 3, 4, 1];
    let nt = ids.len();
    let ids_f: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i as u32)).collect();
    let dids = dev_alloc(nt * 4);
    h2d_f32(dids, &ids_f);
    let dout = dev_alloc(nt * d * 4);
    st.embed_rows_on_gpu(TensorType::Q4_1, wptr, dids, dout, d, nt, false)
        .unwrap();
    let got = d2h_f32(dout, nt * d);
    for (t, &row) in ids.iter().enumerate() {
        for b in 0..nb {
            let off = (row as usize * nb + b) * 20;
            let dv = half::f16::from_bits(u16::from_le_bytes([w[off], w[off + 1]])).to_f32();
            let mv = half::f16::from_bits(u16::from_le_bytes([w[off + 2], w[off + 3]])).to_f32();
            for j in 0..16usize {
                let lo = (w[off + 4 + j] & 0x0F) as f32;
                let hi = (w[off + 4 + j] >> 4) as f32;
                // the kernel's d*q+m may compile to a fused multiply-add;
                // accept either rounding (both are one-ulp forms)
                for (e, v) in [(b * 32 + j, lo), (b * 32 + j + 16, hi)] {
                    let sep = dv * v + mv;
                    let fma = dv.mul_add(v, mv);
                    assert!(
                        got[t * d + e] == sep || got[t * d + e] == fma,
                        "row {row} elem {e}: got {} want {sep} (fma {fma})",
                        got[t * d + e]
                    );
                }
            }
        }
    }
}

#[test]
fn cuda_decode_a_quant_fuse_bitwise() {
    let Some(st) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = CudaState::model_load_guard();
    let eps = 1e-5f32;

    // ---- rms probe: d 5120 (14B hidden) -> q4_K matmul (MMVQ gate) ----
    let (d, od) = (5120usize, 256usize);
    let x = gen_acts(d, 0x9E37);
    let w: Vec<f32> = (0..d).map(|i| 0.5 + 0.001 * (i % 17) as f32).collect();
    let dx = dev_alloc(d * 4);
    h2d_f32(dx, &x);
    let dw = dev_alloc(d * 4);
    h2d_f32(dw, &w);
    let dy_a = dev_alloc(d * 4);
    let dy_b = dev_alloc(d * 4);
    let do_a = dev_alloc(od * 4);
    let do_b = dev_alloc(od * 4);
    st.register_weight("d35_probe_w4", &gen_q4_k(od, d));
    let wq = st
        .get_weight_ptr("d35_probe_w4")
        .expect("weight registered");

    // path A (pre-D3-5 shape): plain rms; matmul cache-cleared -> the
    // standalone quantize launch inside the decode matmul
    st.clear_mmq_cache();
    st.rms_norm(dx, Some(dw), dy_a, d, 1, eps);
    st.matmul_f32_ptr_layout(wq, TensorType::Q4_K, dy_a, do_a, od, d, 1, false)
        .unwrap();
    let q8_a = read_q8(st, (d / 32) * 40);

    // path B: fused rms (records the plane) + matmul (cache hit)
    st.clear_mmq_cache();
    st.rms_norm_quant_on_gpu(dx, dw, dy_b, d, 1, eps);
    st.matmul_f32_ptr_layout(wq, TensorType::Q4_K, dy_b, do_b, od, d, 1, false)
        .unwrap();
    let q8_b = read_q8(st, (d / 32) * 40);

    assert_eq!(q8_a, q8_b, "rms epilogue q8 bytes differ from standalone");
    let y_a = d2h_f32(dy_a, d);
    let y_b = d2h_f32(dy_b, d);
    assert!(
        bits_eq(&y_a, &y_b),
        "fused rms f32 output not bit-identical"
    );
    let o_a = d2h_f32(do_a, od);
    let o_b = d2h_f32(do_b, od);
    assert!(bits_eq(&o_a, &o_b), "MMVQ output diverged (rms path)");

    // ---- swiglu probe: n 2048 (down id, q4_K MMVQ gate) ----
    let (nf, odf) = (2048usize, 256usize);
    let gate = gen_acts(nf, 0x1234);
    let up = gen_acts(nf, 0x5678);
    let mut buf = vec![0f32; 2 * nf];
    buf[..nf].copy_from_slice(&gate);
    buf[nf..].copy_from_slice(&up);
    let db_a = dev_alloc(2 * nf * 4);
    h2d_f32(db_a, &buf);
    let db_b = dev_alloc(2 * nf * 4);
    h2d_f32(db_b, &buf);
    let da_a = dev_alloc(odf * 4);
    let da_b = dev_alloc(odf * 4);
    st.register_weight("d35_probe_w4b", &gen_q4_k(odf, nf));
    let wq2 = st
        .get_weight_ptr("d35_probe_w4b")
        .expect("weight registered");

    st.clear_mmq_cache();
    st.swiglu_f32_off_on_gpu(db_a, nf, nf);
    st.matmul_f32_ptr_layout(wq2, TensorType::Q4_K, db_a, da_a, odf, nf, 1, false)
        .unwrap();
    let q8_a2 = read_q8(st, (nf / 32) * 40);

    st.clear_mmq_cache();
    st.swiglu_quant_off_on_gpu(db_b, nf, nf);
    st.matmul_f32_ptr_layout(wq2, TensorType::Q4_K, db_b, da_b, odf, nf, 1, false)
        .unwrap();
    let q8_b2 = read_q8(st, (nf / 32) * 40);

    assert_eq!(
        q8_a2, q8_b2,
        "swiglu epilogue q8 bytes differ from standalone"
    );
    let bu_a = d2h_f32(db_a, 2 * nf);
    let bu_b = d2h_f32(db_b, 2 * nf);
    assert!(
        bits_eq(&bu_a, &bu_b),
        "fused swiglu f32 output not bit-identical"
    );
    let oa_a = d2h_f32(da_a, odf);
    let oa_b = d2h_f32(da_b, odf);
    assert!(bits_eq(&oa_a, &oa_b), "MMVQ output diverged (swiglu path)");
}
