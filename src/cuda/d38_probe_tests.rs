//! `#[cfg(test)] mod d38_probe_tests` for `src/cuda.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::tensor::TensorType;

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

fn h2d(dst: *mut std::ffi::c_void, src: &[u8]) {
    let err = unsafe {
        cudaMemcpy(
            dst,
            src.as_ptr() as *const std::ffi::c_void,
            src.len(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    };
    assert_eq!(err, 0);
}

fn h2d_f32(dst: *mut std::ffi::c_void, src: &[f32]) {
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for v in src {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    h2d(dst, &bytes);
}

fn d2h(src: *mut std::ffi::c_void, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let err = unsafe {
        cudaMemcpy(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    assert_eq!(err, 0);
    out
}

fn d2h_f32(src: *mut std::ffi::c_void, n: usize) -> Vec<f32> {
    let bytes = d2h(src, n * 4);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Deterministic activation with real-model spread (same generator class
/// as the D3-5 probes).
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

/// D3-8 probe A: fused `attn_bias_rope_store` vs the unfused chain
/// (add_bias×3 + rope×2 + store_kv×2) on the SAME concat-matmul output —
/// bitwise on the q/k/v sections AND both KV regions, f32 + f16 KV,
/// 14B + 7B shapes, BOTH pointer forms (concat sections AND three
/// separate buffers — the mixed-quant class-2 wiring).
#[test]
fn cuda_fused_qkv_epilogue_bitwise() {
    let Some(st) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = CudaState::model_load_guard();
    let (freq_base, freq_scale) = (10000.0f32, 1.0f32);
    let pos0: i32 = 1234;

    // device positions buffer in the graph's i32 form (the kernel reads
    // positions[0]; the graph path converts via f32_bits_to_i32)
    let dpos = dev_alloc(4);
    h2d(dpos, &pos0.to_le_bytes());

    // (nh, nk, hd, tag): 14B GQA 40:8, 7B GQA 28:4 — hd 128 both
    for &(nh, nk, hd, tag) in &[(40usize, 8usize, 128usize, "14B"), (28, 4, 128, "7B")] {
        let (nqt, nkt) = (nh * hd, nk * hd);
        let total = nqt + 2 * nkt;
        let ctx_elems = nkt * (pos0 as usize + 2); // room for the store row
        let acts = gen_acts(total, 0xD380 + nh as u64);
        let bq = gen_acts(nqt, 0xB10 + nh as u64);
        let bk = gen_acts(nkt, 0xB20 + nh as u64);
        let bv = gen_acts(nkt, 0xB30 + nh as u64);

        let dbq = dev_alloc(nqt * 4);
        h2d_f32(dbq, &bq);
        let dbk = dev_alloc(nkt * 4);
        h2d_f32(dbk, &bk);
        let dbv = dev_alloc(nkt * 4);
        h2d_f32(dbv, &bv);

        for kv_f16 in [false, true] {
            let kv_bytes = if kv_f16 { 2 } else { 4 };

            // ---- FUSED form 1: concat buffer, section pointers ----
            let d_fused = dev_alloc(total * 4);
            h2d_f32(d_fused, &acts);
            let base = d_fused as *mut u8;
            let (q1, k1, v1) = (
                d_fused,
                unsafe { base.add(nqt * 4) } as *mut std::ffi::c_void,
                unsafe { base.add((nqt + nkt) * 4) } as *mut std::ffi::c_void,
            );
            let dk_f = dev_alloc(ctx_elems * kv_bytes);
            let dv_f = dev_alloc(ctx_elems * kv_bytes);
            st.attn_bias_rope_store(
                q1,
                k1,
                v1,
                dbq,
                dbk,
                dbv,
                dk_f,
                dv_f,
                nqt,
                nkt,
                hd,
                freq_base,
                freq_scale,
                dpos,
                dpos,
                if kv_f16 { KV_LAYOUT_F16 } else { KV_LAYOUT_F32 },
            );

            // ---- FUSED form 2: three separate buffers (class-2 shape) --
            let d_q2 = dev_alloc(nqt * 4);
            h2d_f32(d_q2, &acts[..nqt]);
            let d_k2 = dev_alloc(nkt * 4);
            h2d_f32(d_k2, &acts[nqt..nqt + nkt]);
            let d_v2 = dev_alloc(nkt * 4);
            h2d_f32(d_v2, &acts[nqt + nkt..]);
            let dk_f2 = dev_alloc(ctx_elems * kv_bytes);
            let dv_f2 = dev_alloc(ctx_elems * kv_bytes);
            st.attn_bias_rope_store(
                d_q2,
                d_k2,
                d_v2,
                dbq,
                dbk,
                dbv,
                dk_f2,
                dv_f2,
                nqt,
                nkt,
                hd,
                freq_base,
                freq_scale,
                dpos,
                dpos,
                if kv_f16 { KV_LAYOUT_F16 } else { KV_LAYOUT_F32 },
            );

            // ---- UNFUSED: the 7-launch chain on split sections ----
            let d_q = dev_alloc(nqt * 4);
            h2d_f32(d_q, &acts[..nqt]);
            let d_k = dev_alloc(nkt * 4);
            h2d_f32(d_k, &acts[nqt..nqt + nkt]);
            let d_v = dev_alloc(nkt * 4);
            h2d_f32(d_v, &acts[nqt + nkt..]);
            st.add_bias_f32(d_q, dbq, nqt, 1);
            st.add_bias_f32(d_k, dbk, nkt, 1);
            st.add_bias_f32(d_v, dbv, nkt, 1);
            st.rope_f32(d_q, nh, hd, 1, freq_base, freq_scale, dpos);
            st.rope_f32(d_k, nk, hd, 1, freq_base, freq_scale, dpos);
            let dk_u = dev_alloc(ctx_elems * kv_bytes);
            let dv_u = dev_alloc(ctx_elems * kv_bytes);
            if kv_f16 {
                st.store_kv_f16(d_k, dk_u, nkt, 1, dpos);
                st.store_kv_f16(d_v, dv_u, nkt, 1, dpos);
            } else {
                st.store_kv_f32(d_k, dk_u, nkt, 1, dpos);
                st.store_kv_f32(d_v, dv_u, nkt, 1, dpos);
            }

            // ---- compare: q/k/v sections + both KV rows ----
            let (rq, rk, rv) = (d2h_f32(d_q, nqt), d2h_f32(d_k, nkt), d2h_f32(d_v, nkt));
            let f1 = d2h_f32(d_fused, total);
            let (qf, kf, vf) = (
                &f1[..nqt],
                &f1[nqt..nqt + nkt],
                &f1[nqt + nkt..nqt + 2 * nkt],
            );
            assert!(
                bits_eq(qf, &rq),
                "{tag} f16={kv_f16}: concat q section diverged"
            );
            assert!(
                bits_eq(kf, &rk),
                "{tag} f16={kv_f16}: concat k section diverged"
            );
            assert!(
                bits_eq(vf, &rv),
                "{tag} f16={kv_f16}: concat v section diverged"
            );
            assert!(
                bits_eq(&d2h_f32(d_q2, nqt), &rq)
                    && bits_eq(&d2h_f32(d_k2, nkt), &rk)
                    && bits_eq(&d2h_f32(d_v2, nkt), &rv),
                "{tag} f16={kv_f16}: separate-buffer form diverged"
            );
            let row = (pos0 as usize) * nkt;
            if kv_f16 {
                for (f, u, what) in [
                    (dk_f, dk_u, "K"),
                    (dv_f, dv_u, "V"),
                    (dk_f2, dk_u, "K form2"),
                    (dv_f2, dv_u, "V form2"),
                ] {
                    let fh = d2h(f, ctx_elems * 2);
                    let uh = d2h(u, ctx_elems * 2);
                    assert_eq!(
                        fh[row * 2..(row + nkt) * 2],
                        uh[row * 2..(row + nkt) * 2],
                        "{tag}: f16 {what} region diverged"
                    );
                }
            } else {
                for (f, u, what) in [
                    (dk_f, dk_u, "K"),
                    (dv_f, dv_u, "V"),
                    (dk_f2, dk_u, "K form2"),
                    (dv_f2, dv_u, "V form2"),
                ] {
                    assert!(
                        bits_eq(
                            &d2h_f32(f, ctx_elems)[row..row + nkt],
                            &d2h_f32(u, ctx_elems)[row..row + nkt],
                        ),
                        "{tag}: f32 {what} region diverged"
                    );
                }
            }
            eprintln!("{tag} kv_f16={kv_f16}: epilogue bitwise OK (both pointer forms)");
        }
    }
}

/// D3-8 probe B: the concat matmul (one launch over wq|wk|wv rows) is
/// per-row bit-identical to the three separate decode matmuls — the MMVQ
/// kernels map one block per row and dispatch on (ttype, id, nt) only.
#[test]
fn cuda_fused_qkv_concat_matmul_bitwise() {
    let Some(st) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = CudaState::model_load_guard();

    // (oq, okv, id, tag): 14B wq|wk|wv = 5120|1024|1024 id 5120;
    // 7B = 3584|512|512 id 3584. Both decode-MMVQ (id ≥ 2048, % 32 == 0).
    for &(oq, okv, id, tag) in &[
        (5120usize, 1024usize, 5120usize, "14B"),
        (3584, 512, 3584, "7B"),
    ] {
        let od_total = oq + 2 * okv;
        let row_bytes = (id / 256) * 144; // q4_K
        let concat = gen_q4_k(od_total, id);
        st.register_weight("d38_concat", &concat);
        st.register_weight("d38_wq", &concat[..row_bytes * oq]);
        st.register_weight("d38_wk", &concat[row_bytes * oq..row_bytes * (oq + okv)]);
        st.register_weight(
            "d38_wv",
            &concat[row_bytes * (oq + okv)..row_bytes * od_total],
        );
        let w_cat = st.get_weight_ptr("d38_concat").expect("concat registered");
        let w_q = st.get_weight_ptr("d38_wq").expect("wq registered");
        let w_k = st.get_weight_ptr("d38_wk").expect("wk registered");
        let w_v = st.get_weight_ptr("d38_wv").expect("wv registered");
        assert!(!st.is_weight_padded("d38_concat"));

        let x = gen_acts(id, 0xC0DE + id as u64);
        let dx = dev_alloc(id * 4);
        h2d_f32(dx, &x);
        let d_cat = dev_alloc(od_total * 4);
        // unfused: wq/wk/wv write separate output buffers (the live path)
        let d_q = dev_alloc(oq * 4);
        let d_k = dev_alloc(okv * 4);
        let d_v = dev_alloc(okv * 4);

        // unfused: three separate decode matmuls (share the A-quantize
        // MmqCache exactly like the live wq/wk/wv group)
        st.clear_mmq_cache();
        st.matmul_f32_ptr_layout(w_q, TensorType::Q4_K, dx, d_q, oq, id, 1, false)
            .unwrap();
        st.matmul_f32_ptr_layout(w_k, TensorType::Q4_K, dx, d_k, okv, id, 1, false)
            .unwrap();
        st.matmul_f32_ptr_layout(w_v, TensorType::Q4_K, dx, d_v, okv, id, 1, false)
            .unwrap();

        // fused: one concat matmul (fresh cache window, standalone quantize)
        st.clear_mmq_cache();
        st.matmul_f32_ptr_layout(w_cat, TensorType::Q4_K, dx, d_cat, od_total, id, 1, false)
            .unwrap();

        let cat = d2h_f32(d_cat, od_total);
        let sq = d2h_f32(d_q, oq);
        let sk = d2h_f32(d_k, okv);
        let sv = d2h_f32(d_v, okv);
        assert!(bits_eq(&cat[..oq], &sq), "{tag}: q rows diverged");
        assert!(bits_eq(&cat[oq..oq + okv], &sk), "{tag}: k rows diverged");
        assert!(bits_eq(&cat[oq + okv..], &sv), "{tag}: v rows diverged");
        eprintln!("{tag}: concat matmul bitwise OK ({od_total} rows)");
    }
}

// D4-4 L1: the dense split-plane (dpl) decode path must be bitwise
// identical to the padded path. Registers the same raw q6_K bytes twice
// (dpl built vs MINFER_Q6K_DPL=0) and compares decode outputs bit-exact.
#[test]
fn cuda_q6k_dpl_bitwise() {
    let Some(st) = device() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = CudaState::model_load_guard();
    // (od, id): a v2_pf_dpl shape (id in (8192, 16384]) and a v2_dpl
    // loop shape; both must clear the decode MMVQ gate (nt==1, id%32==0,
    // od*id >= 4M).
    for (od, id) in [(512usize, 8960usize), (4096usize, 1024usize)] {
        let raw = gen_q6_k_raw(od, id);
        std::env::remove_var("MINFER_Q6K_DPL");
        st.register_weight_q6k_padded("d44_dpl_a", &raw, od, id);
        std::env::set_var("MINFER_Q6K_DPL", "0");
        st.register_weight_q6k_padded("d44_dpl_b", &raw, od, id);
        std::env::remove_var("MINFER_Q6K_DPL");
        let wa = st.get_weight_ptr("d44_dpl_a").expect("a registered");
        let wb = st.get_weight_ptr("d44_dpl_b").expect("b registered");
        let x: Vec<f32> = (0..id).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();
        let dx = dev_alloc(id * 4);
        h2d_f32(dx, &x);
        let oa = dev_alloc(od * 4);
        let ob = dev_alloc(od * 4);
        st.matmul_f32_ptr_layout(wa, TensorType::Q6_K, dx, oa, od, id, 1, true)
            .unwrap();
        st.matmul_f32_ptr_layout(wb, TensorType::Q6_K, dx, ob, od, id, 1, true)
            .unwrap();
        let ra = d2h_f32(oa, od);
        let rb = d2h_f32(ob, od);
        assert_eq!(
            ra, rb,
            "dpl-vs-padded decode outputs must be bit-identical (od {od} id {id})"
        );
        eprintln!("d44_dpl: od {od} id {id} bitwise OK");
    }
}

// raw GGUF-layout q6_K bytes: random ql/qh nibbles, small int8 scales,
// d = 1.0 (finite outputs so the bit-exact compare is meaningful).
fn gen_q6_k_raw(od: usize, id: usize) -> Vec<u8> {
    let nbe = id.div_ceil(256);
    let mut v = vec![0u8; od * nbe * 210];
    let mut s: u32 = 0x5DEE_CE6D;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    for r in 0..od {
        for b in 0..nbe {
            let blk = &mut v[(r * nbe + b) * 210..(r * nbe + b + 1) * 210];
            for i in 0..192 {
                blk[i] = (next() & 0x55) as u8;
            }
            for i in 0..16 {
                blk[192 + i] = (next() % 15) as u8;
            }
            blk[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
    }
    v
}
