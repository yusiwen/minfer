//! Prefill GEMM: MMQ, the fused epilogue and the f16 path.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

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
