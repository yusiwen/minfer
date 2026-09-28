//! `#[cfg(test)] mod tests` for `src/quants.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

fn f32_to_fp16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[test]
fn test_unpack_q4k_scales_boundary() {
    // All zeros
    let sc = [0u8; 12];
    let (scales, mins) = block::unpack_q4k_scales(&sc);
    assert_eq!(scales, [0; 8]);
    assert_eq!(mins, [0; 8]);

    // All 63 (max 6-bit value) in low-6-bit slots, high-2-bit slots set to pack 63 into indices 4-7
    let mut sc = [0u8; 12];
    for j in 0..4 {
        sc[j] = 63; // scales[j] low 6 bits = 63
        sc[j + 4] = 63; // mins[j] low 6 bits = 63
    }
    // For indices 4-7: high 2 bits stored in sc[0..3]>>6 and sc[4..7]>>6
    // scales[4..7] = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4)
    // To get 63 = 0x3F: low 4 bits = 0xF, high 2 bits = 0x3
    for j in 4..8 {
        sc[j + 4] = 0xFF; // low 4 bits = 0xF for scales, high 4 bits = 0xF for mins
    }
    for j in 0..4 {
        sc[j] |= 0xC0; // high 2 bits = 0x3 for scales
        sc[j + 4] |= 0xC0; // high 2 bits = 0x3 for mins
    }
    let (scales, mins) = block::unpack_q4k_scales(&sc);
    assert_eq!(scales, [63; 8], "scales={:?}", scales);
    assert_eq!(mins, [63; 8], "mins={:?}", mins);

    // Mixed: known values
    let mut sc = [0u8; 12];
    sc[0] = 10;
    sc[1] = 20;
    sc[2] = 30;
    sc[3] = 40;
    sc[4] = 5;
    sc[5] = 15;
    sc[6] = 25;
    sc[7] = 35;
    // scales[4]=50: low4=0x2, high2=0x3 → sc[8]&0xF=2, sc[0]>>6=3 → sc[0]|=0xC0
    // mins[4]=45: low4=0xD, high2=0x2 → sc[8]>>4=0xD, sc[4]>>6=2 → sc[4]|=0x80
    sc[8] = 0x02 | (0x0D << 4); // scales[4] low=2, mins[4] low=0xD
    sc[0] |= 0xC0; // scales[4] high=3
    sc[4] |= 0x80; // mins[4] high=2
    let (scales, mins) = block::unpack_q4k_scales(&sc);
    assert_eq!(scales[0], 10);
    assert_eq!(scales[1], 20);
    assert_eq!(scales[2], 30);
    assert_eq!(scales[3], 40);
    assert_eq!(scales[4], 50);
    assert_eq!(mins[0], 5);
    assert_eq!(mins[1], 15);
    assert_eq!(mins[2], 25);
    assert_eq!(mins[3], 35);
    assert_eq!(mins[4], 45);
}

#[test]
fn test_q8k_dot_simple() {
    let mut x = vec![0u8; Q8B];
    let mut y = vec![0u8; Q8B];
    let dx = 0.5f32;
    let dy = 2.0f32;
    let dx_bits = f32_to_fp16(dx).to_le_bytes();
    let dy_bits = f32_to_fp16(dy).to_le_bytes();
    x[0] = dx_bits[0];
    x[1] = dx_bits[1];
    y[0] = dy_bits[0];
    y[1] = dy_bits[1];
    for j in 0..32 {
        x[2 + j] = (j as i8) as u8;
        y[2 + j] = (31 - j as i8) as u8;
    }
    let result = dot_q8_0_q8_0(&x, &y);
    let mut ref_sum = 0.0f32;
    for j in 0..32 {
        ref_sum += ((j as i8) as f32 * dx) * (((31 - j) as i8) as f32 * dy);
    }
    eprintln!(
        "test_q8k_dot_simple: result={:e} ref={:e} diff={:e}",
        result,
        ref_sum,
        (result - ref_sum).abs()
    );
    assert!((result - ref_sum).abs() < 0.01);
}

#[test]
fn test_q5_1_dot() {
    use crate::block::fp16_to_f32;
    // Q5_1: d(f16,2) + m(f16,2) + qh(u32,4) + qs(u8,16) = 24B
    // weight = d * unsigned_5bit + m
    let mut q5 = vec![0u8; 24];
    // d = 2.0 (fp16: 0x4000)
    q5[0] = 0x00;
    q5[1] = 0x40;
    // m = 0.5 (fp16: 0x3800)
    q5[2] = 0x00;
    q5[3] = 0x38;
    // qh = 0 (no high bits)
    // qs nibbles: 0,1,2,...,15 for both lo and hi
    for j in 0..16u8 {
        q5[8 + j as usize] = j | (j << 4);
    }

    // Build Q8_0 activation: all 1.0 -> d_q8 = 1.0/127 ≈ 0.007874, quants = 127
    let mut q8 = vec![0u8; 34];
    q8[0] = 0x00;
    q8[1] = 0x20; // fp16 1.0/128? Let's use d_q8=1.0, actually use known values
                  // Actually, let's use d_q8 = 1.0 (fp16 0x3C00) and all quants = 1
    q8[0] = 0x00;
    q8[1] = 0x3C; // d_q8 = 1.0
    for j in 0..32 {
        q8[2 + j] = 1u8;
    } // quants = 1

    let result = dot_q5_1_q8_0(&q5, &q8);

    // Manual: Σ(d_q8 * (d * unsigned_5bit + m) * q8_quant)
    // unsigned_5bit = nibble (0..15), q8_quant = 1
    // result = 1.0 * Σ((2.0 * j + 0.5) * 1) for j in 0..15, counted twice (lo+hi)
    let mut ref_sum = 0.0f32;
    for j in 0..16 {
        ref_sum += 2.0 * j as f32 + 0.5 + 2.0 * j as f32 + 0.5;
    }
    // ref = 32*0.5 + 2*2*Σ(j=0..15) = 16 + 4*120 = 16 + 480 = 496
    eprintln!(
        "test_q5_1_dot: result={:e} ref={:e} diff={:e}",
        result,
        ref_sum,
        (result - ref_sum).abs()
    );
    assert!(
        (result - ref_sum).abs() < 1.0,
        "result={} ref={}",
        result,
        ref_sum
    );
}
