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
    // valid d/dmin (finite fp16), scales/mins as u8
    b[0] = 0x00;
    b[1] = 0x3C;
    b[2] = 0x00;
    b[3] = 0x3C;
    b
}
fn q6k_block(st: &mut u64) -> Vec<u8> {
    let mut b = vec![0u8; Q6KB];
    for v in b.iter_mut() {
        *v = (next(st) >> 8) as u8;
    }
    b[208] = 0x00;
    b[209] = 0x3C;
    b
}
fn q8k_block(st: &mut u64) -> Vec<u8> {
    let mut b = vec![0u8; crate::block::Q8KB];
    for v in b.iter_mut() {
        *v = (next(st) >> 8) as u8;
    }
    b[0] = 0x00;
    b[1] = 0x3C;
    b
}

#[test]
fn neon_q8k_dots_match_scalar() {
    // Realistic data: activations come from the q8_K quantizer (bsums
    // bounded to ±2032 — vpaddq_s16 in the kernel wraps on full-range
    // int16, which the scalar i32 path does not; real bsums never reach
    // that range), weights use valid d/dmin/scales/mins with nibble data.
    let mut st = rng_state();
    for _ in 0..20 {
        let dim = 256 * (1 + (next(&mut st) % 4) as usize);
        let x: Vec<f32> = (0..dim)
            .map(|i| (((next(&mut st) >> 40) % 1000) as f32) * 1e-3 - 0.5)
            .collect();
        let mut q8k = vec![0u8; (dim / 256) * crate::block::Q8KB];
        quantize_row_q8_k_buf(&x, 1, dim, &mut q8k);
        // q4_K / q6_K weight blocks
        let mut q4 = Vec::new();
        let mut q6 = Vec::new();
        for _ in 0..dim / 256 {
            let mut b = vec![0u8; Q4KB];
            for v in b.iter_mut() {
                *v = ((next(&mut st) >> 8) % 8) as u8;
            }
            b[0] = 0x00;
            b[1] = 0x3C;
            b[2] = 0x00;
            b[3] = 0x3C; // d = dmin = 1
            b[4] = 8;
            b[5] = 8;
            b[6] = 8;
            b[7] = 8; // scales[0..4] = 8
            q4.extend_from_slice(&b);
            let mut b6 = vec![0u8; Q6KB];
            for v in b6.iter_mut() {
                *v = ((next(&mut st) >> 8) % 8) as u8;
            }
            b6[208] = 0x00;
            b6[209] = 0x3C; // d = 1
            q6.extend_from_slice(&b6);
        }
        let a = dot_q4_k_q8_k(&q4, &q8k);
        let b = dot_q4_k_q8_k_scalar(&q4, &q8k);
        assert!(
            (a - b).abs() <= (a.abs() + b.abs()).max(1e-6) * 1e-5,
            "q4_K q8_K NEON {a} != scalar {b}"
        );
        let a = dot_q6_k_q8_k(&q6, &q8k);
        let b = dot_q6_k_q8_k_scalar(&q6, &q8k);
        assert!(
            (a - b).abs() <= (a.abs() + b.abs()).max(1e-6) * 1e-5,
            "q6_K q8_K NEON {a} != scalar {b}"
        );
    }
}

#[test]
fn neon_q8k_quantize_matches_scalar() {
    let mut st = rng_state();
    for _ in 0..10 {
        let dim = 256 * (1 + (next(&mut st) % 4) as usize);
        let x: Vec<f32> = (0..dim)
            .map(|i| ((next(&mut st) >> 40) as f32) * 1e-3 - 0.5)
            .collect();
        let mut ba = vec![0u8; (dim / 256) * crate::block::Q8KB];
        let mut bb = vec![0u8; (dim / 256) * crate::block::Q8KB];
        quantize_row_q8_k_buf(&x, 1, dim, &mut ba);
        // scalar path
        let n_super = dim / 256;
        let rowb = n_super * crate::block::Q8KB;
        for s in 0..n_super {
            let blk = &x[s * 256..(s + 1) * 256];
            let o = s * crate::block::Q8KB;
            let mut amax = 0.0f32;
            for &v in blk {
                amax = amax.max(v.abs());
            }
            let d = amax / 127.0f32;
            let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
            let db = half::f16::from_f32(d).to_bits().to_le_bytes();
            bb[o] = db[0];
            bb[o + 1] = db[1];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                let q = (blk[j] * id).round_ties_even().clamp(-128.0, 127.0) as i8;
                bb[o + 2 + j] = q as u8;
                bsums[j / 16] += q as i16;
            }
            for j in 0..16 {
                bb[o + 258 + j] = 0;
                bb[o + 274 + 2 * j..o + 274 + 2 * j + 2].copy_from_slice(&bsums[j].to_le_bytes());
            }
        }
        assert_eq!(ba, bb, "q8_K quantize NEON != scalar at dim {dim}");
    }
}
