//! x86_64 correctness gate for the AVX2/FMA and AVX-512/VNNI K-quant dots:
//! each SIMD kernel is compared against its `*_scalar` reference **bitwise**
//! (`f32::to_bits()`), per superblock (so a failure names the offset) and over
//! the whole multi-superblock row. This is the x86_64 counterpart of
//! `neon_correctness`, which uses a tolerance; the x86 kernels are exact
//! because the integer products accumulate in i32 and the per-superblock float
//! arithmetic keeps the scalar kernel's order.
use super::avx2;
use super::avx512;
use super::*;

fn rng_state() -> u64 {
    0x9E3779B97F4A7C15
}
fn next(st: &mut u64) -> u64 {
    *st = st
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *st
}

fn q4k_block(st: &mut u64) -> Vec<u8> {
    let mut b = vec![0u8; Q4KB];
    for v in b.iter_mut() {
        *v = (next(st) >> 8) as u8;
    }
    b[0] = 0x00;
    b[1] = 0x3C; // d = 1
    b[2] = 0x00;
    b[3] = 0x3C; // dmin = 1
    b
}

fn q5k_block(st: &mut u64) -> Vec<u8> {
    let mut b = vec![0u8; 176];
    for v in b.iter_mut() {
        *v = (next(st) >> 8) as u8;
    }
    b[0] = 0x00;
    b[1] = 0x3C; // d = 1
    b[2] = 0x00;
    b[3] = 0x3C; // dmin = 1
    b
}

fn q6k_block(st: &mut u64) -> Vec<u8> {
    let mut b = vec![0u8; Q6KB];
    for v in b.iter_mut() {
        *v = (next(st) >> 8) as u8;
    }
    b[208] = 0x00;
    b[209] = 0x3C; // d = 1
    b
}

/// A real Q8_K row (quantizer output, so the bsums are in the format's
/// valid range) plus the matching q4/q5/q6 weight rows, multi-superblock.
fn build_case(st: &mut u64, n_super: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let dim = 256 * n_super;
    let x: Vec<f32> = (0..dim)
        .map(|_| (((next(st) >> 40) % 1000) as f32) * 1e-3 - 0.5)
        .collect();
    let mut q8k = vec![0u8; n_super * crate::block::Q8KB];
    quantize_row_q8_k_buf(&x, 1, dim, &mut q8k);
    let mut q4 = Vec::new();
    let mut q5 = Vec::new();
    let mut q6 = Vec::new();
    for _ in 0..n_super {
        q4.extend_from_slice(&q4k_block(st));
        q5.extend_from_slice(&q5k_block(st));
        q6.extend_from_slice(&q6k_block(st));
    }
    (q4, q5, q6, q8k)
}

fn assert_bits(what: &str, offset: usize, got: f32, expected: f32) {
    assert_eq!(
        got.to_bits(),
        expected.to_bits(),
        "{what} superblock offset {offset}: got {got:?} ({:#010x}) != scalar {expected:?} ({:#010x})",
        got.to_bits(),
        expected.to_bits()
    );
}

#[test]
fn avx2_q8k_dots_match_scalar_bitwise() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        return;
    }
    let mut st = rng_state();
    for n_super in 1..=4 {
        let (q4, q5, q6, q8k) = build_case(&mut st, n_super);

        // Per superblock: names the offset on mismatch.
        for i in 0..n_super {
            let e4 = dot_q4_k_q8_k_scalar(&q4[i * Q4KB..], &q8k[i * crate::block::Q8KB..]);
            let g4 =
                unsafe { avx2::dot_q4_k_q8_k(&q4[i * Q4KB..], &q8k[i * crate::block::Q8KB..]) };
            assert_bits("q4_K AVX2", i, g4, e4);

            let e5 = dot_q5_k_q8_k_scalar(&q5[i * 176..], &q8k[i * crate::block::Q8KB..]);
            let g5 = unsafe { avx2::dot_q5_k_q8_k(&q5[i * 176..], &q8k[i * crate::block::Q8KB..]) };
            assert_bits("q5_K AVX2", i, g5, e5);

            let e6 = dot_q6_k_q8_k_scalar(&q6[i * Q6KB..], &q8k[i * crate::block::Q8KB..]);
            let g6 =
                unsafe { avx2::dot_q6_k_q8_k(&q6[i * Q6KB..], &q8k[i * crate::block::Q8KB..]) };
            assert_bits("q6_K AVX2", i, g6, e6);
        }

        // Whole row: the cross-superblock accumulation order must match too.
        assert_eq!(
            unsafe { avx2::dot_q4_k_q8_k(&q4, &q8k) }.to_bits(),
            dot_q4_k_q8_k_scalar(&q4, &q8k).to_bits(),
            "q4_K AVX2 full-row mismatch at n_super={n_super}"
        );
        assert_eq!(
            unsafe { avx2::dot_q5_k_q8_k(&q5, &q8k) }.to_bits(),
            dot_q5_k_q8_k_scalar(&q5, &q8k).to_bits(),
            "q5_K AVX2 full-row mismatch at n_super={n_super}"
        );
        assert_eq!(
            unsafe { avx2::dot_q6_k_q8_k(&q6, &q8k) }.to_bits(),
            dot_q6_k_q8_k_scalar(&q6, &q8k).to_bits(),
            "q6_K AVX2 full-row mismatch at n_super={n_super}"
        );
    }
}

#[test]
fn avx512_q8k_dots_match_scalar_bitwise() {
    if !(is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vnni"))
    {
        return;
    }
    let mut st = rng_state();
    for n_super in 1..=4 {
        let (q4, q5, q6, q8k) = build_case(&mut st, n_super);
        for i in 0..n_super {
            let o = i * crate::block::Q8KB;
            let e4 = dot_q4_k_q8_k_scalar(&q4[i * Q4KB..], &q8k[o..]);
            let g4 = unsafe { avx512::dot_q4_k_q8_k(&q4[i * Q4KB..], &q8k[o..]) };
            assert_bits("q4_K AVX-512", i, g4, e4);

            let e5 = dot_q5_k_q8_k_scalar(&q5[i * 176..], &q8k[o..]);
            let g5 = unsafe { avx512::dot_q5_k_q8_k(&q5[i * 176..], &q8k[o..]) };
            assert_bits("q5_K AVX-512", i, g5, e5);

            let e6 = dot_q6_k_q8_k_scalar(&q6[i * Q6KB..], &q8k[o..]);
            let g6 = unsafe { avx512::dot_q6_k_q8_k(&q6[i * Q6KB..], &q8k[o..]) };
            assert_bits("q6_K AVX-512", i, g6, e6);
        }
        assert_eq!(
            unsafe { avx512::dot_q4_k_q8_k(&q4, &q8k) }.to_bits(),
            dot_q4_k_q8_k_scalar(&q4, &q8k).to_bits(),
            "q4_K AVX-512 full-row mismatch at n_super={n_super}"
        );
        assert_eq!(
            unsafe { avx512::dot_q5_k_q8_k(&q5, &q8k) }.to_bits(),
            dot_q5_k_q8_k_scalar(&q5, &q8k).to_bits(),
            "q5_K AVX-512 full-row mismatch at n_super={n_super}"
        );
        assert_eq!(
            unsafe { avx512::dot_q6_k_q8_k(&q6, &q8k) }.to_bits(),
            dot_q6_k_q8_k_scalar(&q6, &q8k).to_bits(),
            "q6_K AVX-512 full-row mismatch at n_super={n_super}"
        );
    }
}

/// The public dispatcher (whichever runtime path this box has) must be
/// bitwise-identical to the scalar reference.
#[test]
fn dispatch_q8k_dots_match_scalar_bitwise() {
    let mut st = rng_state();
    for n_super in 1..=4 {
        let (q4, q5, q6, q8k) = build_case(&mut st, n_super);
        assert_eq!(
            dot_q4_k_q8_k(&q4, &q8k).to_bits(),
            dot_q4_k_q8_k_scalar(&q4, &q8k).to_bits(),
            "q4_K dispatch mismatch at n_super={n_super}"
        );
        assert_eq!(
            dot_q5_k_q8_k(&q5, &q8k).to_bits(),
            dot_q5_k_q8_k_scalar(&q5, &q8k).to_bits(),
            "q5_K dispatch mismatch at n_super={n_super}"
        );
        assert_eq!(
            dot_q6_k_q8_k(&q6, &q8k).to_bits(),
            dot_q6_k_q8_k_scalar(&q6, &q8k).to_bits(),
            "q6_K dispatch mismatch at n_super={n_super}"
        );
    }
}

/// Timing harness (not a correctness gate): `cargo test --release -- --ignored
/// kquant_simd_dot_speedup --nocapture` prints ns/call for the scalar, AVX2 and
/// AVX-512 K-quant dots on a 4096-element row (16 superblocks). Neighbouring 4
/// KB weight rows are the shapes the cached Qwen2.5 models use; the number the
/// ticket records is the SIMD/scalar ratio, not an assertion.
#[test]
#[ignore]
fn kquant_simd_dot_speedup() {
    use std::time::Instant;
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    let has_avx512 = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vnni");
    let mut st = rng_state();
    let n_super = 4096 / 256;
    let (q4, q5, q6, q8k) = build_case(&mut st, n_super);
    let iters = 20000usize;
    let med = |f: &dyn Fn() -> f32| -> f64 {
        let mut times = Vec::new();
        for _ in 0..5 {
            let mut acc = 0.0f32;
            for _ in 0..200 {
                acc += f();
            }
            let t = Instant::now();
            for _ in 0..iters {
                acc += f();
            }
            times.push(t.elapsed().as_secs_f64() * 1e9 / iters as f64);
            std::hint::black_box(acc);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };
    let s4 = med(&|| dot_q4_k_q8_k_scalar(&q4, &q8k));
    let s5 = med(&|| dot_q5_k_q8_k_scalar(&q5, &q8k));
    let s6 = med(&|| dot_q6_k_q8_k_scalar(&q6, &q8k));
    eprintln!("scalar  q4_K {s4:.1} ns  q5_K {s5:.1} ns  q6_K {s6:.1} ns");
    if has_avx2 {
        let a4 = med(&|| unsafe { avx2::dot_q4_k_q8_k(&q4, &q8k) });
        let a5 = med(&|| unsafe { avx2::dot_q5_k_q8_k(&q5, &q8k) });
        let a6 = med(&|| unsafe { avx2::dot_q6_k_q8_k(&q6, &q8k) });
        eprintln!(
            "AVX2    q4_K {a4:.1} ns ({:.2}x)  q5_K {a5:.1} ns ({:.2}x)  q6_K {a6:.1} ns ({:.2}x)",
            s4 / a4,
            s5 / a5,
            s6 / a6
        );
    }
    if has_avx512 {
        let a4 = med(&|| unsafe { avx512::dot_q4_k_q8_k(&q4, &q8k) });
        let a5 = med(&|| unsafe { avx512::dot_q5_k_q8_k(&q5, &q8k) });
        let a6 = med(&|| unsafe { avx512::dot_q6_k_q8_k(&q6, &q8k) });
        eprintln!(
            "AVX-512 q4_K {a4:.1} ns ({:.2}x)  q5_K {a5:.1} ns ({:.2}x)  q6_K {a6:.1} ns ({:.2}x)",
            s4 / a4,
            s5 / a5,
            s6 / a6
        );
    }
}
