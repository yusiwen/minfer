//! `#[cfg(test)] mod tests` for `src/quantize.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// The reference vector values from llama.cpp's `test-quantize-fns`
/// (`tests/test-quantize-fns.cpp`), which are also a convenient
/// human-checkable block layout: a ramp of 32 values repeated.
fn ramp() -> Vec<f32> {
    (0..32).map(|i| (i as f32 - 16.0) / 8.0).collect()
}

/// The K-quant block sizes and byte sizes, pinned against `ggml-common.h`
/// (`QK_K = 256`; `sizeof(block_q4_K)` = 2+2+12+128, `q5_K` adds `qh[32]`,
/// `q6_K` = 128+64+16+2). `type_size`/`blck_size` come from `gguf.rs`; this
/// is the writer-side half of the same contract.
#[test]
fn k_quant_type_sizes_and_block_sizes_match_ggml_common_h() {
    assert_eq!(QuantTarget::Q4_K.blck_size(), 256);
    assert_eq!(QuantTarget::Q5_K.blck_size(), 256);
    assert_eq!(QuantTarget::Q6_K.blck_size(), 256);
    assert_eq!(GgmlType::Q4_K.blck_size(), 256);
    assert_eq!(GgmlType::Q5_K.blck_size(), 256);
    assert_eq!(GgmlType::Q6_K.blck_size(), 256);
    assert_eq!(GgmlType::Q4_K.type_size(), 144);
    assert_eq!(GgmlType::Q5_K.type_size(), 176);
    assert_eq!(GgmlType::Q6_K.type_size(), 210);
}

#[test]
fn targets_parse_and_refuse_by_name() {
    assert_eq!(QuantTarget::parse("Q4_0").unwrap(), QuantTarget::Q4_0);
    assert_eq!(QuantTarget::parse("f16").unwrap(), QuantTarget::F16);
    // #140: the three K-quants this engine *reads* now have encoders, spelled
    // the way the GGUF type names them (`q4_K`, not `q4_k`/`Q4_K`).
    for t in ["q4_K", "q5_K", "q6_K", "Q6_k"] {
        let parsed = QuantTarget::parse(t).unwrap();
        assert!(parsed.blck_size() == 256, "{t}");
    }
    assert_eq!(QuantTarget::parse("q4_K").unwrap(), QuantTarget::Q4_K);
    assert_eq!(QuantTarget::parse("q5_K").unwrap(), QuantTarget::Q5_K);
    assert_eq!(QuantTarget::parse("q6_K").unwrap(), QuantTarget::Q6_K);
    // Still refused by name: the K-quants with no encoder, every I-quant and
    // the other non-float GGUF types.
    for t in ["q2_K", "q3_K", "q8_K", "iq2_xxs", "tq1_0", "bf16", "q8_1"] {
        let e = QuantTarget::parse(t).unwrap_err();
        assert!(e.contains("no weight encoder"), "{t}: {e}");
        assert!(e.contains(SUPPORTED_TARGETS), "{t}: {e}");
    }
    let e = QuantTarget::parse("banana").unwrap_err();
    assert!(e.contains("unknown quant target"), "{e}");
    assert!(e.contains(SUPPORTED_TARGETS), "{e}");
}

#[test]
fn written_block_bytes_match_type_size_and_blck_size() {
    // 512 elements: 16 legacy 32-blocks, 2 K-quant 256-blocks.
    let x = vec![0.5f32; 512];
    for t in [
        QuantTarget::Q4_0,
        QuantTarget::Q4_1,
        QuantTarget::Q5_0,
        QuantTarget::Q5_1,
        QuantTarget::Q8_0,
        QuantTarget::Q4_K,
        QuantTarget::Q5_K,
        QuantTarget::Q6_K,
    ] {
        let y = quantize_row(t, &x);
        let gt = t.ggml_type();
        assert_eq!(
            y.len(),
            (x.len() / gt.blck_size() as usize) * gt.type_size(),
            "{}",
            t.name()
        );
    }
    // f16 / f32 element casts
    assert_eq!(quantize_row(QuantTarget::F16, &x).len(), x.len() * 2);
    assert_eq!(quantize_row(QuantTarget::F32, &x).len(), x.len() * 4);
}

#[test]
fn q8_0_encoder_matches_the_reference_numbers() {
    // d = amax/127 with amax = 2.0 → 0.015748031..., and the quants are the
    // rounded ratio, so the block round-trips within half a step.
    let x = ramp();
    let y = quantize_row(QuantTarget::Q8_0, &x);
    assert_eq!(y.len(), 34);
    let d = half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
    assert!((d - 2.0 / 127.0).abs() < 1e-6, "{d}");
    // min value -2.0 → -127, max +1.875 → 119
    assert_eq!(y[2] as i8, -127);
    assert_eq!(y[2 + 31] as i8, roundf(1.875 / d as f32) as i8);
    let back = dequantize_block(GgmlType::Q8_0, &y);
    for (a, b) in x.iter().zip(back.iter()) {
        assert!((a - b).abs() <= d as f32 / 2.0 + 1e-6, "{a} vs {b}");
    }
}

#[test]
fn q4_0_encoder_matches_the_reference_layout() {
    // max (largest |v|) is -2.0 → d = 0.25, id = 4. The reference packs
    // element j in the low nibble and j+16 in the high nibble.
    let x = ramp();
    let y = quantize_row(QuantTarget::Q4_0, &x);
    assert_eq!(y.len(), 18);
    let d = half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
    assert_eq!(d, 0.25);
    // element 0 = -2.0 → (-2*4 + 8.5) = 0.5 → 0; element 16 = 0.0 → 8
    assert_eq!(y[2] & 0x0F, 0);
    assert_eq!(y[2] >> 4, 8);
    let back = dequantize_block(GgmlType::Q4_0, &y);
    assert_eq!(back.len(), 32);
    // every value within one quant step (d) of the source
    for (a, b) in x.iter().zip(back.iter()) {
        assert!((a - b).abs() <= d + 1e-6, "{a} vs {b}");
    }
}

#[test]
fn q4_1_q5_0_q5_1_round_trip_within_their_steps() {
    let x = ramp();
    for (t, step) in [
        (QuantTarget::Q4_1, 1.0 / 15.0),
        (QuantTarget::Q5_0, 1.0 / 16.0),
        (QuantTarget::Q5_1, 1.0 / 31.0),
    ] {
        let y = quantize_row(t, &x);
        let back = dequantize_block(t.ggml_type(), &y);
        assert_eq!(back.len(), 32);
        let range = 4.0; // max - min for the ramp
        for (a, b) in x.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() <= range * step + 1e-6,
                "{}: {a} vs {b}",
                t.name()
            );
        }
    }
}

#[test]
fn a_zero_block_has_zero_scale_and_no_nans() {
    let x = vec![0.0f32; 32];
    for t in [
        QuantTarget::Q4_0,
        QuantTarget::Q4_1,
        QuantTarget::Q5_0,
        QuantTarget::Q5_1,
        QuantTarget::Q8_0,
    ] {
        let y = quantize_row(t, &x);
        for f in dequantize_block(t.ggml_type(), &y) {
            assert!(f.is_finite(), "{}", t.name());
        }
    }
}

#[test]
fn decode_refuses_k_quants_and_decodes_the_supported_set() {
    assert!(decode_to_f32(GgmlType::Q4_K, &[0u8; 144], 256).is_none());
    assert!(decode_to_f32(GgmlType::Q6_K, &[0u8; 210], 256).is_none());
    let x = ramp();
    for t in [
        QuantTarget::F16,
        QuantTarget::F32,
        QuantTarget::Q4_0,
        QuantTarget::Q8_0,
    ] {
        let y = quantize_row(t, &x);
        let back = decode_to_f32(t.ggml_type(), &y, 32).unwrap();
        assert_eq!(back.len(), 32, "{}", t.name());
        if t == QuantTarget::F16 || t == QuantTarget::F32 {
            assert_eq!(back, x, "{} must round-trip exactly", t.name());
        }
    }
}

#[test]
fn f16_overflow_saturates_to_infinity_not_a_wrong_finite_value() {
    // f32→f16 is NOT exact: 70000 exceeds the f16 range. The cast yields
    // inf — the honest representation — rather than silently wrapping.
    let y = quantize_row(QuantTarget::F16, &[70000.0f32]);
    let back = half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
    assert!(back.is_infinite(), "{back}");
}

// === #140: the K-quant encoders ===

/// The vector the K-quant encoders are pinned against: eight 32-element
/// ramps, **ascending** in the even 32-groups and **descending** in the odd
/// ones. The alternation matters: a vector whose 32-groups all run the same
/// way gives `L[j] == L[j+32]` and therefore identical low/high nibbles, so
/// a swapped-nibble bug cancels and the pin cannot see it (gate contract
/// rule 2). This one makes the two nibbles differ (`0xf0`, `0xe1`, …).
fn k_ref_vector() -> Vec<f32> {
    (0..256)
        .map(|i| {
            let (g, j) = (i / 32, i % 32);
            if g % 2 == 0 {
                (j as f32 - 16.0) / 8.0
            } else {
                (15.0 - j as f32) / 8.0
            }
        })
        .collect()
}

/// Dequantize a K-quant block with the *documented* on-disk layout
/// (`ggml-common.h` + llama.cpp's `dequantize_row_q{4,5,6}_K`), written here
/// rather than reusing the encoder, so the round-trip check is an
/// independent read-back of the bytes the encoder produced.
fn k_dequantize(t: QuantTarget, b: &[u8]) -> Vec<f32> {
    let f16_at = |o: usize| half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32();
    match t {
        QuantTarget::Q4_K => {
            let (d, dmin) = (f16_at(0), f16_at(2));
            let scales: [u8; 12] = b[4..16].try_into().unwrap();
            let q = &b[16..144];
            let mut out = Vec::with_capacity(256);
            let mut is = 0usize;
            for j in (0..256).step_by(64) {
                let (sc, m) = get_scale_min_k4(is, &scales);
                let (d1, m1) = (d * sc as f32, dmin * m as f32);
                let (sc, m) = get_scale_min_k4(is + 1, &scales);
                let (d2, m2) = (d * sc as f32, dmin * m as f32);
                for l in 0..32 {
                    out.push(d1 * (q[j / 2 + l] & 0xF) as f32 - m1);
                }
                for l in 0..32 {
                    out.push(d2 * (q[j / 2 + l] >> 4) as f32 - m2);
                }
                is += 2;
            }
            out
        }
        QuantTarget::Q5_K => {
            let (d, dmin) = (f16_at(0), f16_at(2));
            let scales: [u8; 12] = b[4..16].try_into().unwrap();
            let qh = &b[16..48];
            let ql = &b[48..176];
            let mut out = Vec::with_capacity(256);
            let (mut is, mut u1, mut u2) = (0usize, 1u8, 2u8);
            for j in (0..256).step_by(64) {
                let (sc, m) = get_scale_min_k4(is, &scales);
                let (d1, m1) = (d * sc as f32, dmin * m as f32);
                let (sc, m) = get_scale_min_k4(is + 1, &scales);
                let (d2, m2) = (d * sc as f32, dmin * m as f32);
                for l in 0..32 {
                    let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                    out.push(d1 * ((ql[j / 2 + l] & 0xF) + hi) as f32 - m1);
                }
                for l in 0..32 {
                    let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                    out.push(d2 * ((ql[j / 2 + l] >> 4) + hi) as f32 - m2);
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            out
        }
        QuantTarget::Q6_K => {
            let d = f16_at(208);
            let sc = &b[192..208];
            let mut out = vec![0.0f32; 256];
            for (g, group) in out.chunks_exact_mut(128).enumerate() {
                let (ql, qh) = (&b[g * 64..g * 64 + 64], &b[128 + g * 32..128 + g * 32 + 32]);
                for l in 0..32 {
                    let is = g * 8 + l / 16;
                    let q = |lo: u8, sh: u32| {
                        ((lo as i32 | (((qh[l] >> sh) & 3) as i32) << 4) - 32) as f32
                    };
                    group[l] = d * sc[is] as i8 as f32 * q(ql[l] & 0xF, 0);
                    group[l + 32] = d * sc[is + 2] as i8 as f32 * q(ql[l + 32] & 0xF, 2);
                    group[l + 64] = d * sc[is + 4] as i8 as f32 * q(ql[l] >> 4, 4);
                    group[l + 96] = d * sc[is + 6] as i8 as f32 * q(ql[l + 32] >> 4, 6);
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// The reference bytes for `k_ref_vector()`, produced by **llama.cpp's own
/// reference quantizers** (`quantize_row_q4_K_ref` / `q5_K_ref` /
/// `q6_K_ref`), compiled from `ggml/src/ggml-quants.c` with the production
/// flags `-O3 -DNDEBUG -std=gnu11 -mcpu=native` and driven directly — not
/// by this module. A single flipped nibble or a dropped FMA fails the
/// comparison.
#[test]
fn k_quant_encoders_match_the_llama_cpp_reference_bytes() {
    let x = k_ref_vector();
    let want: [(QuantTarget, &[u8]); 3] = [
        (
            QuantTarget::Q4_K,
            &[
                0x11, 0x1c, 0xe2, 0x27, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff, 0xf0, 0xf0, 0xe1, 0xe1, 0xd2, 0xd2, 0xc3, 0xc3, 0xb4, 0xb4, 0xa5, 0xa5,
                0x96, 0x96, 0x87, 0x87, 0x78, 0x78, 0x69, 0x69, 0x5a, 0x5a, 0x4b, 0x4b, 0x3c, 0x3c,
                0x2d, 0x2d, 0x1e, 0x1e, 0x0f, 0x0f, 0xf0, 0xf0, 0xe1, 0xe1, 0xd2, 0xd2, 0xc3, 0xc3,
                0xb4, 0xb4, 0xa5, 0xa5, 0x96, 0x96, 0x87, 0x87, 0x78, 0x78, 0x69, 0x69, 0x5a, 0x5a,
                0x4b, 0x4b, 0x3c, 0x3c, 0x2d, 0x2d, 0x1e, 0x1e, 0x0f, 0x0f, 0xf0, 0xf0, 0xe1, 0xe1,
                0xd2, 0xd2, 0xc3, 0xc3, 0xb4, 0xb4, 0xa5, 0xa5, 0x96, 0x96, 0x87, 0x87, 0x78, 0x78,
                0x69, 0x69, 0x5a, 0x5a, 0x4b, 0x4b, 0x3c, 0x3c, 0x2d, 0x2d, 0x1e, 0x1e, 0x0f, 0x0f,
                0xf0, 0xf0, 0xe1, 0xe1, 0xd2, 0xd2, 0xc3, 0xc3, 0xb4, 0xb4, 0xa5, 0xa5, 0x96, 0x96,
                0x87, 0x87, 0x78, 0x78, 0x69, 0x69, 0x5a, 0x5a, 0x4b, 0x4b, 0x3c, 0x3c, 0x2d, 0x2d,
                0x1e, 0x1e, 0x0f, 0x0f,
            ],
        ),
        (
            QuantTarget::Q5_K,
            &[
                0x10, 0x18, 0x10, 0x28, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
                0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87,
                0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5,
                0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f, 0xf0, 0xe1, 0xd2, 0xc3,
                0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f, 0xf0, 0xe1,
                0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f,
                0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d,
                0x1e, 0x0f, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b,
                0x3c, 0x2d, 0x1e, 0x0f, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69,
                0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87,
                0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f,
            ],
        ),
        (
            QuantTarget::Q6_K,
            &[
                0x00, 0x22, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee, 0x00, 0x22, 0x44, 0x66, 0x88, 0xaa,
                0xcc, 0xee, 0x00, 0xee, 0xcc, 0xaa, 0x88, 0x66, 0x44, 0x11, 0xff, 0xdd, 0xbb, 0x99,
                0x77, 0x55, 0x33, 0x11, 0x11, 0x33, 0x55, 0x77, 0x99, 0xbb, 0xdd, 0xff, 0x11, 0x44,
                0x66, 0x88, 0xaa, 0xcc, 0xee, 0x00, 0xee, 0xcc, 0xaa, 0x88, 0x66, 0x44, 0x22, 0x00,
                0xee, 0xcc, 0xaa, 0x88, 0x66, 0x44, 0x22, 0x00, 0x00, 0x22, 0x44, 0x66, 0x88, 0xaa,
                0xcc, 0xee, 0x00, 0x22, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee, 0x00, 0xee, 0xcc, 0xaa,
                0x88, 0x66, 0x44, 0x11, 0xff, 0xdd, 0xbb, 0x99, 0x77, 0x55, 0x33, 0x11, 0x11, 0x33,
                0x55, 0x77, 0x99, 0xbb, 0xdd, 0xff, 0x11, 0x44, 0x66, 0x88, 0xaa, 0xcc, 0xee, 0x00,
                0xee, 0xcc, 0xaa, 0x88, 0x66, 0x44, 0x22, 0x00, 0xee, 0xcc, 0xaa, 0x88, 0x66, 0x44,
                0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55,
                0x55, 0x55, 0x55, 0x99, 0x66, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x99, 0x66, 0x55, 0x55, 0x55, 0x55, 0x55,
                0x55, 0x55, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x7b, 0x7b, 0x80,
                0x80, 0x7b, 0x7b, 0x80, 0x80, 0x7b, 0x7b, 0x80, 0x80, 0x7b, 0x7b, 0x80, 0x00, 0x90,
            ],
        ),
    ];
    for (t, bytes) in want {
        let got = quantize_row(t, &x);
        assert_eq!(got.len(), bytes.len(), "{} length", t.name());
        assert_eq!(
            got,
            bytes,
            "{} bytes differ from the llama.cpp reference",
            t.name()
        );
    }
}

/// A second reference vector at exact half-integer values (`i*0.5 - 128`
/// plus a `.5` every third element), so the search's `iscale` candidates
/// land on ties and a different set of sub-blocks wins. Same provenance as
/// the pin above; the extra distribution is the point (the `k_ref_vector`
/// pin and this one do not fail on the same mutations).
#[test]
fn k_quant_encoders_match_the_reference_at_rounding_boundaries() {
    let x: Vec<f32> = (0..256)
        .map(|i| (i as f32) * 0.5 - 128.0 + if i % 3 == 0 { 0.5 } else { 0.0 })
        .collect();
    let want: [(QuantTarget, &[u8]); 3] = [
        (
            QuantTarget::Q4_K,
            &[
                0x1c, 0x24, 0x0a, 0x40, 0xfe, 0xff, 0xfe, 0xfe, 0xbf, 0x77, 0x2f, 0x27, 0x0f, 0x8e,
                0xfe, 0x8f, 0x00, 0x00, 0x00, 0x11, 0x21, 0x22, 0x23, 0x33, 0x33, 0x44, 0x54, 0x55,
                0x56, 0x66, 0x66, 0x77, 0x77, 0x78, 0x89, 0x99, 0x99, 0x9a, 0xaa, 0xab, 0xbc, 0xcc,
                0xcc, 0xcd, 0xdd, 0xde, 0xef, 0xff, 0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x22, 0x22,
                0x33, 0x43, 0x44, 0x45, 0x55, 0x55, 0x66, 0x76, 0x77, 0x78, 0x88, 0x88, 0x99, 0xa9,
                0xaa, 0xab, 0xbb, 0xbb, 0xcc, 0xdc, 0xdd, 0xde, 0xee, 0xee, 0x01, 0x12, 0x22, 0x22,
                0x23, 0x33, 0x34, 0x45, 0x55, 0x55, 0x56, 0x66, 0x67, 0x78, 0x88, 0x88, 0x99, 0xa9,
                0xaa, 0xab, 0xbb, 0xbb, 0xcc, 0xdc, 0xdc, 0xdd, 0xed, 0xee, 0xff, 0xff, 0xff, 0xff,
                0x00, 0x10, 0x10, 0x20, 0x30, 0x31, 0x32, 0x42, 0x42, 0x53, 0x63, 0x64, 0x65, 0x75,
                0x75, 0x86, 0x96, 0x97, 0x98, 0xa8, 0xa8, 0xb9, 0xc9, 0xca, 0xcb, 0xdb, 0xdb, 0xec,
                0xec, 0xed, 0xfe, 0xfe,
            ],
        ),
        (
            QuantTarget::Q5_K,
            &[
                0x16, 0x20, 0x0c, 0x40, 0xff, 0xff, 0xff, 0xff, 0xbf, 0x77, 0x6f, 0x27, 0x0f, 0x8f,
                0x0f, 0x8f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x20, 0x70, 0xf0, 0xf5, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x10, 0x11, 0x23, 0x43, 0x44, 0x56, 0x76,
                0x77, 0x89, 0xa9, 0xaa, 0xbc, 0xdc, 0xdd, 0xef, 0xff, 0xf0, 0x02, 0x22, 0x23, 0x35,
                0x55, 0x56, 0x68, 0x88, 0x89, 0x9b, 0xbb, 0xbc, 0xce, 0xee, 0x00, 0x00, 0x01, 0x21,
                0x22, 0x34, 0x54, 0x55, 0x67, 0x87, 0x88, 0x9a, 0xba, 0xbb, 0xcd, 0xed, 0xee, 0xf0,
                0x10, 0x11, 0x23, 0x43, 0x44, 0x56, 0x76, 0x77, 0x89, 0xa9, 0xaa, 0xbc, 0xdc, 0xdd,
                0x11, 0x23, 0x43, 0x44, 0x56, 0x76, 0x77, 0x89, 0xa9, 0xaa, 0xbc, 0xdc, 0xdd, 0xef,
                0x0f, 0x00, 0x12, 0x32, 0x33, 0x45, 0x65, 0x66, 0x78, 0x98, 0x99, 0xab, 0xcb, 0xcc,
                0xde, 0xfe, 0xff, 0xff, 0x02, 0x22, 0x23, 0x35, 0x55, 0x56, 0x68, 0x88, 0x89, 0x9b,
                0xbb, 0xbc, 0xce, 0xee, 0xef, 0xf1, 0x11, 0x12, 0x24, 0x44, 0x45, 0x57, 0x77, 0x78,
                0x8a, 0xaa, 0xab, 0xbd, 0xdd, 0xde, 0xef, 0xff,
            ],
        ),
        (
            QuantTarget::Q6_K,
            &[
                0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x21, 0x21, 0x21,
                0x21, 0x22, 0x10, 0x10, 0x10, 0x10, 0x20, 0x21, 0x21, 0x21, 0x21, 0x21, 0x31, 0x31,
                0x31, 0x31, 0x32, 0x32, 0x00, 0x00, 0x00, 0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x21,
                0x22, 0x22, 0x22, 0x22, 0x32, 0x32, 0x00, 0x00, 0x10, 0x10, 0x10, 0x10, 0x11, 0x21,
                0x21, 0x21, 0x21, 0x31, 0x32, 0x32, 0x32, 0x32, 0x00, 0x01, 0x11, 0x21, 0x21, 0x21,
                0x32, 0x32, 0x42, 0x52, 0x53, 0x53, 0x63, 0x64, 0x74, 0x84, 0x00, 0x00, 0x20, 0x21,
                0x21, 0x41, 0x42, 0x42, 0x62, 0x63, 0x63, 0x83, 0x83, 0x83, 0xa4, 0xa4, 0x00, 0x20,
                0x21, 0x31, 0x51, 0x52, 0x62, 0x82, 0x83, 0x93, 0xb3, 0xb4, 0xc4, 0xe4, 0xe5, 0xf5,
                0x21, 0x22, 0x42, 0x82, 0x83, 0xa3, 0xe4, 0xe4, 0x04, 0x45, 0x45, 0x65, 0xa6, 0xa7,
                0xc7, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x80, 0x80, 0x88, 0x8f, 0x98,
                0xa0, 0xa5, 0xb0, 0xb7, 0xbf, 0xc8, 0xd0, 0xd6, 0xe0, 0xe8, 0xf0, 0xf8, 0xed, 0xa7,
            ],
        ),
    ];
    for (t, bytes) in want {
        let got = quantize_row(t, &x);
        assert_eq!(got.len(), bytes.len(), "{} length", t.name());
        assert_eq!(
            got,
            bytes,
            "{} bytes differ from the llama.cpp reference at a rounding boundary",
            t.name()
        );
    }
}

/// The block layout, read back through the documented field order: the f16
/// scale(s), the 6-bit scale/min packing (`get_scale_min_k4`), the
/// `j` / `j+32` nibble split and q5_K's separate 5th-bit plane. The
/// round-trip is bounded by the block's own quantisation step, and the
/// measured worst error is printed.
#[test]
fn k_quant_blocks_round_trip_within_their_step() {
    let x = k_ref_vector();
    // The vector spans 4.0 and q4_K's 4-bit step is ~0.36, so a half step is
    // ~0.18; 0.1 is tighter than a half step and a swapped nibble (error
    // ~4.0) cannot pass. The measured worst error per type is printed.
    let bound = 0.1f32;
    for t in [QuantTarget::Q4_K, QuantTarget::Q5_K, QuantTarget::Q6_K] {
        let y = quantize_row(t, &x);
        let back = k_dequantize(t, &y);
        assert_eq!(back.len(), 256, "{}", t.name());
        let worst = x
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= bound, "{}: worst |Δ| = {worst}", t.name());
        eprintln!("{}: worst |Δ| over one QK_K block = {worst}", t.name());
    }
    // q5_K uses its 5th-bit plane on this vector: the high bits are not all
    // zero, which is what makes the `qh` read-back above meaningful.
    let y = quantize_row(QuantTarget::Q5_K, &x);
    assert!(
        y[16..48].iter().any(|&b| b != 0),
        "q5_K: the vector must exercise the 5th-bit plane"
    );
}

/// `nearest_int_mul` really is the *fused* form, and the fixture proves it
/// exercises the property: for this pair the exact product plus the magic
/// constant rounds to 1517, while rounding the product first and then
/// adding rounds to 1516. Writing `a * b + c` here — which is what the
/// obvious port looks like — is therefore a wrong quant, not a style
/// choice. (The pair was found by search; the reference emits
/// `fmadd a, b, #12582912.0`.)
#[test]
fn nearest_int_mul_fuses_the_magic_constant_into_the_product() {
    let a = f32::from_bits(0xc205_b75f);
    let b = f32::from_bits(0xc235_7575);
    assert_eq!(
        nearest_int(a * b),
        1516,
        "the fixture must distinguish the fused form from the plain one"
    );
    assert_eq!(nearest_int_mul(a, b), 1517);
}

/// A zero super-block is representable exactly: all-zero quants, zero
/// scale(s) and no NaN. q6_K takes the reference's `memset` path; q4_K and
/// q5_K take `make_qkx2_quants`'s `max == min` path plus the `if (!d)
/// continue` re-round, which leaves the sub-block quants at 0.
#[test]
fn a_zero_k_block_is_all_zero_bytes() {
    let x = vec![0.0f32; 256];
    for t in [QuantTarget::Q4_K, QuantTarget::Q5_K, QuantTarget::Q6_K] {
        let y = quantize_row(t, &x);
        assert!(
            y.iter().all(|&b| b == 0),
            "{}: a zero block must be all-zero bytes, got {:?}",
            t.name(),
            &y[..y.len().min(16)]
        );
        for f in k_dequantize(t, &y) {
            assert!(f.is_finite(), "{}", t.name());
        }
    }
}

/// `q4_K`/`q5_K` are the `_M` mixtures in llama.cpp's *CLI* vocabulary, but
/// `minfer quantize --type q4_K` writes one uniform type. The `file_type`
/// metadata number is the ftype llama.cpp records for the same number, and
/// the row-length fallback is llama.cpp's `tensor_type_fallback`.
#[test]
fn k_quant_file_types_and_row_length_fallbacks_match_llama_cpp() {
    assert_eq!(QuantTarget::Q4_K.file_type(), 15); // LLAMA_FTYPE_MOSTLY_Q4_K_M
    assert_eq!(QuantTarget::Q5_K.file_type(), 17); // LLAMA_FTYPE_MOSTLY_Q5_K_M
    assert_eq!(QuantTarget::Q6_K.file_type(), 18); // LLAMA_FTYPE_MOSTLY_Q6_K
    assert_eq!(
        QuantTarget::Q4_K.row_len_fallback(),
        Some(QuantTarget::Q5_0)
    );
    assert_eq!(
        QuantTarget::Q5_K.row_len_fallback(),
        Some(QuantTarget::Q5_1)
    );
    assert_eq!(
        QuantTarget::Q6_K.row_len_fallback(),
        Some(QuantTarget::Q8_0)
    );
    // The legacy targets have no fallback in this module: a 2-D tensor whose
    // row is not a multiple of 32 keeps its source type (llama.cpp demotes
    // it to F16, which is the same file for the f16 sources the gate uses).
    for t in [
        QuantTarget::Q4_0,
        QuantTarget::Q4_1,
        QuantTarget::Q5_0,
        QuantTarget::Q5_1,
        QuantTarget::Q8_0,
        QuantTarget::F16,
        QuantTarget::F32,
    ] {
        assert_eq!(t.row_len_fallback(), None, "{}", t.name());
    }
    assert_eq!(
        SUPPORTED_TARGETS,
        "q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K, q6_K, f16, f32"
    );
}
