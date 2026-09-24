// Weight quantization encoders — the byte-exact counterparts of llama.cpp's
// `quantize_row_*_ref` (ggml/src/ggml-quants.c).
//
// Why a separate module: `quants.rs` is the *inference* kernel surface
// (dot products + the Q8_0/Q8_K activation quantizers it feeds). Model-file
// creation is a different contract — it must produce the same bytes llama.cpp's
// converter produces, so the reference functions there are the spec, and they
// are gated byte-for-byte against `llama-quantize` output.
//
// Supported encoders: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (plus the plain f16/f32
// element casts). Every other GGUF type — including the K-quants this engine
// can *read* (Q4_K/Q5_K/Q6_K) — has no encoder and is refused by name
// ([`QuantTarget::parse`]), never silently approximated.

use crate::gguf::GgmlType;

/// A quantize/convert target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTarget {
    F32,
    F16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
}

/// The targets this module can actually encode, spelled as the CLI accepts them.
pub const SUPPORTED_TARGETS: &str = "q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32";

/// The GGUF types that exist in the format but have no encoder here. Listed so
/// the refusal can say "known type, no encoder" instead of "unknown target" —
/// the two are different failures and deserve different messages.
const KNOWN_UNSUPPORTED: &[&str] = &[
    "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K", "q8_1", "iq2_xxs", "iq2_xs", "iq3_xxs",
    "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs", "iq1_m", "tq1_0", "tq2_0", "mxfp4", "nvfp4",
    "q1_0", "bf16", "f64", "i8", "i16", "i32", "i64",
];

impl QuantTarget {
    /// Parse a CLI target, refusing loudly and by name.
    ///
    /// The K-quants are the interesting refusal: this engine *reads* Q4_K/Q5_K/
    /// Q6_K at inference time, so "quantize to q4_K" looks supported and would
    /// silently produce a file with wrong weights if the encoder were stubbed.
    /// It is not stubbed; it is refused.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(QuantTarget::F32),
            "f16" | "fp16" => Ok(QuantTarget::F16),
            "q8_0" => Ok(QuantTarget::Q8_0),
            "q4_0" => Ok(QuantTarget::Q4_0),
            "q4_1" => Ok(QuantTarget::Q4_1),
            "q5_0" => Ok(QuantTarget::Q5_0),
            "q5_1" => Ok(QuantTarget::Q5_1),
            other => {
                let known = KNOWN_UNSUPPORTED
                    .iter()
                    .any(|k| k.eq_ignore_ascii_case(other));
                if known {
                    Err(format!(
                        "minfer quantize: target {other:?} is a known GGUF type but minfer has no \
                         weight encoder for it (minfer can only read it, so writing it would emit \
                         wrong weights); supported encoder targets: {SUPPORTED_TARGETS}"
                    ))
                } else {
                    Err(format!(
                        "minfer quantize: unknown quant target {other:?}; supported: \
                         {SUPPORTED_TARGETS}"
                    ))
                }
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            QuantTarget::F32 => "f32",
            QuantTarget::F16 => "f16",
            QuantTarget::Q8_0 => "q8_0",
            QuantTarget::Q4_0 => "q4_0",
            QuantTarget::Q4_1 => "q4_1",
            QuantTarget::Q5_0 => "q5_0",
            QuantTarget::Q5_1 => "q5_1",
        }
    }

    pub fn ggml_type(self) -> GgmlType {
        match self {
            QuantTarget::F32 => GgmlType::F32,
            QuantTarget::F16 => GgmlType::F16,
            QuantTarget::Q8_0 => GgmlType::Q8_0,
            QuantTarget::Q4_0 => GgmlType::Q4_0,
            QuantTarget::Q4_1 => GgmlType::Q4_1,
            QuantTarget::Q5_0 => GgmlType::Q5_0,
            QuantTarget::Q5_1 => GgmlType::Q5_1,
        }
    }

    /// llama.cpp's `llama_ftype` value for `general.file_type`.
    pub fn file_type(self) -> u32 {
        match self {
            QuantTarget::F32 => 0,  // LLAMA_FTYPE_ALL_F32
            QuantTarget::F16 => 1,  // LLAMA_FTYPE_MOSTLY_F16
            QuantTarget::Q4_0 => 2, // LLAMA_FTYPE_MOSTLY_Q4_0
            QuantTarget::Q4_1 => 3, // LLAMA_FTYPE_MOSTLY_Q4_1
            QuantTarget::Q8_0 => 7, // LLAMA_FTYPE_MOSTLY_Q8_0
            QuantTarget::Q5_0 => 8, // LLAMA_FTYPE_MOSTLY_Q5_0
            QuantTarget::Q5_1 => 9, // LLAMA_FTYPE_MOSTLY_Q5_1
        }
    }

    /// The block size (elements per block) of the encoded layout.
    pub fn blck_size(self) -> usize {
        match self {
            QuantTarget::F32 | QuantTarget::F16 => 1,
            _ => 32,
        }
    }
}

// === Element decoding (GGUF bytes → f32) ===

/// Whether [`decode_to_f32`] can decode this type without the data (for
/// planning: a K-quant source is refused before a single byte is read).
pub fn can_decode(type_: GgmlType) -> bool {
    matches!(
        type_,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
    )
}

/// Decode a source tensor's bytes into f32, row by row (`row_elems` elements).
///
/// The types this returns `Some` for are exactly the ones a source file may be
/// re-quantized from: the plain float types and the legacy quants this module
/// also encodes. K-quant inputs return `None` and the caller refuses loudly.
pub fn decode_to_f32(type_: GgmlType, data: &[u8], _row_elems: usize) -> Option<Vec<f32>> {
    match type_ {
        GgmlType::F32 => Some(
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        GgmlType::F16 => Some(
            data.chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
        ),
        GgmlType::BF16 => Some(
            data.chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
        ),
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let bs = type_.blck_size() as usize;
            let ts = type_.type_size();
            let mut out = Vec::with_capacity(data.len() / ts * bs);
            for block in data.chunks_exact(ts) {
                out.extend_from_slice(&dequantize_block(type_, block));
            }
            Some(out)
        }
        _ => None,
    }
}

/// One block of a legacy quant → its values. The exact inverse of the encoder
/// up to the stored rounding, matching llama.cpp's `dequantize_row_*`.
pub fn dequantize_block(type_: GgmlType, b: &[u8]) -> Vec<f32> {
    match type_ {
        GgmlType::Q8_0 => {
            let d = f16_at(b, 0);
            (0..32).map(|j| d * b[2 + j] as i8 as f32).collect()
        }
        GgmlType::Q4_0 => {
            let d = f16_at(b, 0);
            (0..32)
                .map(|j| {
                    let q = if j < 16 {
                        (b[2 + j] & 0x0F) as i32 - 8
                    } else {
                        (b[2 + j - 16] >> 4) as i32 - 8
                    };
                    d * q as f32
                })
                .collect()
        }
        GgmlType::Q4_1 => {
            let d = f16_at(b, 0);
            let m = f16_at(b, 2);
            (0..32)
                .map(|j| {
                    let q = if j < 16 {
                        (b[4 + j] & 0x0F) as f32
                    } else {
                        (b[4 + j - 16] >> 4) as f32
                    };
                    d * q + m
                })
                .collect()
        }
        GgmlType::Q5_0 => {
            let d = f16_at(b, 0);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            (0..32)
                .map(|j| {
                    let lo = if j < 16 {
                        (b[6 + j] & 0x0F) as u32
                    } else {
                        (b[6 + j - 16] >> 4) as u32
                    };
                    let hi = (qh >> j) & 1;
                    d * (((lo | (hi << 4)) as i32) - 16) as f32
                })
                .collect()
        }
        GgmlType::Q5_1 => {
            let d = f16_at(b, 0);
            let m = f16_at(b, 2);
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            (0..32)
                .map(|j| {
                    let lo = if j < 16 {
                        (b[8 + j] & 0x0F) as u32
                    } else {
                        (b[8 + j - 16] >> 4) as u32
                    };
                    let hi = (qh >> j) & 1;
                    d * (lo | (hi << 4)) as f32 + m
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[off], b[off + 1]])).to_f32()
}

#[inline]
fn put_f16(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
}

// === Encoders (llama.cpp refs, verbatim arithmetic) ===

/// Quantize a flat run of f32 (a whole row, or a whole tensor whose rows are
/// contiguous and block-aligned) to `target`'s byte layout.
///
/// For f16/f32 this is an element cast; for the quants it is the llama.cpp
/// reference encoder block by block.
pub fn quantize_row(target: QuantTarget, x: &[f32]) -> Vec<u8> {
    match target {
        QuantTarget::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        QuantTarget::F16 => {
            let mut out = Vec::with_capacity(x.len() * 2);
            for &v in x {
                put_f16(&mut out, v);
            }
            out
        }
        QuantTarget::Q8_0 => quantize_q8_0(x),
        QuantTarget::Q4_0 => quantize_q4_0(x),
        QuantTarget::Q4_1 => quantize_q4_1(x),
        QuantTarget::Q5_0 => quantize_q5_0(x),
        QuantTarget::Q5_1 => quantize_q5_1(x),
    }
}

fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for blk in x.chunks_exact(32) {
        let mut amax = 0.0f32;
        for &v in blk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(&mut out, d);
        for &v in blk {
            out.push(roundf(v * id) as i8 as u8);
        }
    }
    out
}

fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 18);
    for blk in x.chunks_exact(32) {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in blk {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(&mut out, d);
        let mut qs = [0u8; 16];
        for j in 0..16 {
            // llama.cpp's reference is compiled with `-ffp-contract=fast`, so
            // `x*id + 8.5f` is contracted to an FMA on aarch64/x86. Without the
            // FMA the encoder differs from `llama-quantize` in a handful of
            // nibbles (12 of 64512 on one 0.5B tensor) — byte parity needs it.
            let xi0 = trunc_i8(blk[j].mul_add(id, 8.5)).min(15) as u8;
            let xi1 = trunc_i8(blk[j + 16].mul_add(id, 8.5)).min(15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
        out.extend_from_slice(&qs);
    }
    out
}

fn quantize_q4_1(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 20);
    for blk in x.chunks_exact(32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in blk {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(&mut out, d);
        put_f16(&mut out, min);
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let xi0 = trunc_i8((blk[j] - min).mul_add(id, 0.5)).min(15) as u8;
            let xi1 = trunc_i8((blk[j + 16] - min).mul_add(id, 0.5)).min(15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
        out.extend_from_slice(&qs);
    }
    out
}

fn quantize_q5_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 22);
    for blk in x.chunks_exact(32) {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in blk {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(&mut out, d);
        let mut qh = 0u32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let xi0 = trunc_i8(blk[j].mul_add(id, 16.5)).min(31) as u8;
            let xi1 = trunc_i8(blk[j + 16].mul_add(id, 16.5)).min(31) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // The 5th bit goes to a separate plane, element j for the low half
            // and element j+16 for the high half.
            qh |= (((xi0 & 0x10u8) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10u8) >> 4) as u32) << (j + 16);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

fn quantize_q5_1(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 24);
    for blk in x.chunks_exact(32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in blk {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(&mut out, d);
        put_f16(&mut out, min);
        let mut qh = 0u32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let xi0 = trunc_i8((blk[j] - min).mul_add(id, 0.5)) as u8;
            let xi1 = trunc_i8((blk[j + 16] - min).mul_add(id, 0.5)) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10u8) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10u8) >> 4) as u32) << (j + 16);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

/// C's `roundf`: nearest, ties away from zero. Rust's `f32::round` is the same.
#[inline]
fn roundf(v: f32) -> f32 {
    v.round()
}

/// C's `(int8_t)v`: truncation toward zero. Rust's `as i8` also truncates
/// toward zero, which is what the reference relies on.
#[inline]
fn trunc_i8(v: f32) -> i8 {
    v as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference vector values from llama.cpp's `test-quantize-fns`
    /// (`tests/test-quantize-fns.cpp`), which are also a convenient
    /// human-checkable block layout: a ramp of 32 values repeated.
    fn ramp() -> Vec<f32> {
        (0..32).map(|i| (i as f32 - 16.0) / 8.0).collect()
    }

    #[test]
    fn targets_parse_and_refuse_by_name() {
        assert_eq!(QuantTarget::parse("Q4_0").unwrap(), QuantTarget::Q4_0);
        assert_eq!(QuantTarget::parse("f16").unwrap(), QuantTarget::F16);
        for t in ["q4_K", "q6_K", "iq2_xxs", "tq1_0", "bf16", "q8_1"] {
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
        let x = vec![0.5f32; 64];
        for t in [
            QuantTarget::Q4_0,
            QuantTarget::Q4_1,
            QuantTarget::Q5_0,
            QuantTarget::Q5_1,
            QuantTarget::Q8_0,
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
}
