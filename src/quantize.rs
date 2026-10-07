// Weight quantization encoders — the byte-exact counterparts of llama.cpp's
// `quantize_row_*_ref` (ggml/src/ggml-quants.c).
//
// Why a separate module: `quants.rs` is the *inference* kernel surface
// (dot products + the Q8_0/Q8_K activation quantizers it feeds). Model-file
// creation is a different contract — it must produce the same bytes llama.cpp's
// converter produces, so the reference functions there are the spec, and they
// are gated byte-for-byte against `llama-quantize` output.
//
// Supported encoders: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the three K-quants this
// engine reads (Q4_K, Q5_K, Q6_K), plus the plain f16/f32 element casts. Every
// other GGUF type has no encoder and is refused by name
// ([`QuantTarget::parse`]), never silently approximated.

use crate::gguf::GgmlType;

/// A quantize/convert target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // Q4_K/Q5_K/Q6_K are ggml's on-disk spellings
pub enum QuantTarget {
    F32,
    F16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4_K,
    Q5_K,
    Q6_K,
}

/// The targets this module can actually encode, spelled as the CLI accepts them.
pub const SUPPORTED_TARGETS: &str = "q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K, q6_K, f16, f32";

/// The GGUF types that exist in the format but have no encoder here. Listed so
/// the refusal can say "known type, no encoder" instead of "unknown target" —
/// the two are different failures and deserve different messages.
const KNOWN_UNSUPPORTED: &[&str] = &[
    "q2_K", "q3_K", "q8_K", "q8_1", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s",
    "iq2_s", "iq4_xs", "iq1_m", "tq1_0", "tq2_0", "mxfp4", "nvfp4", "q1_0", "bf16", "f64", "i8",
    "i16", "i32", "i64",
];

impl QuantTarget {
    /// Parse a CLI target, refusing loudly and by name.
    ///
    /// The I-quants and the K-quants without an encoder are the interesting
    /// refusals: this engine *reads* Q4_K/Q5_K/Q6_K at inference time, so
    /// "quantize to q2_K" looks supported and would silently produce a file
    /// with wrong weights if the encoder were stubbed. It is not stubbed; it is
    /// refused.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(QuantTarget::F32),
            "f16" | "fp16" => Ok(QuantTarget::F16),
            "q8_0" => Ok(QuantTarget::Q8_0),
            "q4_0" => Ok(QuantTarget::Q4_0),
            "q4_1" => Ok(QuantTarget::Q4_1),
            "q5_0" => Ok(QuantTarget::Q5_0),
            "q5_1" => Ok(QuantTarget::Q5_1),
            "q4_k" => Ok(QuantTarget::Q4_K),
            "q5_k" => Ok(QuantTarget::Q5_K),
            "q6_k" => Ok(QuantTarget::Q6_K),
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
            QuantTarget::Q4_K => "q4_K",
            QuantTarget::Q5_K => "q5_K",
            QuantTarget::Q6_K => "q6_K",
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
            QuantTarget::Q4_K => GgmlType::Q4_K,
            QuantTarget::Q5_K => GgmlType::Q5_K,
            QuantTarget::Q6_K => GgmlType::Q6_K,
        }
    }

    /// llama.cpp's `llama_ftype` value for `general.file_type`.
    ///
    /// `q4_K`/`q5_K` are the `_M` mixtures in llama.cpp's CLI vocabulary
    /// (`LLAMA_FTYPE_MOSTLY_Q4_K_M` = 15, `_Q5_K_M` = 17); `q6_K` is 18. The
    /// value describes the ftype, and a file whose tensors are all one K type
    /// is what `--pure` writes under the same number.
    pub fn file_type(self) -> u32 {
        match self {
            QuantTarget::F32 => 0,   // LLAMA_FTYPE_ALL_F32
            QuantTarget::F16 => 1,   // LLAMA_FTYPE_MOSTLY_F16
            QuantTarget::Q4_0 => 2,  // LLAMA_FTYPE_MOSTLY_Q4_0
            QuantTarget::Q4_1 => 3,  // LLAMA_FTYPE_MOSTLY_Q4_1
            QuantTarget::Q8_0 => 7,  // LLAMA_FTYPE_MOSTLY_Q8_0
            QuantTarget::Q5_0 => 8,  // LLAMA_FTYPE_MOSTLY_Q5_0
            QuantTarget::Q5_1 => 9,  // LLAMA_FTYPE_MOSTLY_Q5_1
            QuantTarget::Q4_K => 15, // LLAMA_FTYPE_MOSTLY_Q4_K_M
            QuantTarget::Q5_K => 17, // LLAMA_FTYPE_MOSTLY_Q5_K_M
            QuantTarget::Q6_K => 18, // LLAMA_FTYPE_MOSTLY_Q6_K
        }
    }

    /// The block size (elements per block) of the encoded layout.
    pub fn blck_size(self) -> usize {
        match self {
            QuantTarget::F32 | QuantTarget::F16 => 1,
            QuantTarget::Q4_K | QuantTarget::Q5_K | QuantTarget::Q6_K => K_QUANT_BLOCK,
            _ => 32,
        }
    }

    /// The type llama.cpp's `tensor_type_fallback` demotes this one to when a
    /// 2-D tensor's row length is not a multiple of the target's block size
    /// (`GGML_TYPE_QK_K`-block types demote to a legacy 32-block type).
    pub fn row_len_fallback(self) -> Option<QuantTarget> {
        match self {
            QuantTarget::Q4_K => Some(QuantTarget::Q5_0),
            QuantTarget::Q5_K => Some(QuantTarget::Q5_1),
            QuantTarget::Q6_K => Some(QuantTarget::Q8_0),
            _ => None,
        }
    }
}

// === Element decoding (GGUF bytes → f32) ===

/// `QK_K` — the elements per block of every K-quant (ggml-common.h).
pub const K_QUANT_BLOCK: usize = 256;

/// llama.cpp's `GROUP_MAX_EPS` (ggml-quants.c): a sub-block whose largest
/// magnitude is below this is treated as all-zero and gets a zero scale.
const GROUP_MAX_EPS: f32 = 1e-15;

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

/// Append already-rounded f16 bits (the K-quant reference stores `d`/`dmin` as
/// f16 and then reads them back for the final re-round, so the bits are the
/// value that matters).
#[inline]
fn put_f16_bits(out: &mut Vec<u8>, bits: u16) {
    out.extend_from_slice(&bits.to_le_bytes());
}

// === Encoders (llama.cpp refs, verbatim arithmetic) ===

/// Which `-ffp-contract` semantics an encoder run reproduces.
///
/// This is the one input the byte-parity claim of this module is *conditional*
/// on. llama.cpp's `quantize_row_q4_0_ref` computes `x*id + c`; a compiler that
/// contracts that expression across statements (`-ffp-contract=fast`, GCC's
/// default and what every recorded reference was built with) emits one FMA,
/// while one that does not (Apple clang without the flag, or GCC/clang with
/// `-ffp-contract=off`) emits `fmul` + `fadd` and picks a different quant at an
/// exact rounding boundary. Production always asks for [`FmaContract::Fast`];
/// the F6 byte-parity gate's provenance probe asks for [`FmaContract::Off`] to
/// identify which build produced the `llama-quantize` reference it is comparing
/// against (`docs/GGUF-TOOLING.md` §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmaContract {
    /// `-ffp-contract=fast`: every `a*b + c` is one fused multiply-add.
    Fast,
    /// Every fused operation split back into a multiply and an add — what a
    /// compiler that does not contract across statements emits. **Exact** for
    /// the legacy `q4_0`/`q4_1`/`q5_0`/`q5_1` encoders, whose reference has one
    /// contractable shape per element, and a *model* of the K-quants, whose
    /// reference additionally depends on the compiler's per-expression fusion
    /// and vectorization choices (§4.2.1). Only the F6 parity gate's provenance
    /// probe constructs it (`src/tooling/tests/quantize_bounds.rs`), so it is
    /// scoped to the test build rather than annotated with an `allow`.
    #[cfg(test)]
    Off,
}

/// `a*b + c` under an explicit [`FmaContract`]. The `Fast` arm is the
/// production arithmetic (`f32::mul_add`); the `Off` arm exists only in the
/// test build, so a non-test build has exactly one arm and no choice to make.
#[inline]
fn fma(contract: FmaContract, a: f32, b: f32, c: f32) -> f32 {
    match contract {
        FmaContract::Fast => a.mul_add(b, c),
        #[cfg(test)]
        FmaContract::Off => a * b + c,
    }
}

/// Quantize a flat run of f32 (a whole row, or a whole tensor whose rows are
/// contiguous and block-aligned) to `target`'s byte layout.
///
/// For f16/f32 this is an element cast; for the quants it is the llama.cpp
/// reference encoder block by block. This is the `-ffp-contract=fast`
/// arithmetic — the documented reference build; see [`quantize_row_with`].
pub fn quantize_row(target: QuantTarget, x: &[f32]) -> Vec<u8> {
    quantize_row_with(target, x, FmaContract::Fast)
}

/// [`quantize_row`] with an explicit FMA-contract variant.
///
/// The variant is a no-op for `f16`/`f32` (an element cast) and for `q8_0` (a
/// single multiply and no add, so there is nothing to contract); it is the
/// whole difference for the legacy quants and part of it for the K-quants.
pub fn quantize_row_with(target: QuantTarget, x: &[f32], contract: FmaContract) -> Vec<u8> {
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
        QuantTarget::Q4_0 => quantize_q4_0(x, contract),
        QuantTarget::Q4_1 => quantize_q4_1(x, contract),
        QuantTarget::Q5_0 => quantize_q5_0(x, contract),
        QuantTarget::Q5_1 => quantize_q5_1(x, contract),
        QuantTarget::Q4_K => quantize_q4_k(x, contract),
        QuantTarget::Q5_K => quantize_q5_k(x, contract),
        QuantTarget::Q6_K => quantize_q6_k(x, contract),
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

fn quantize_q4_0(x: &[f32], contract: FmaContract) -> Vec<u8> {
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
            let xi0 = trunc_i8(fma(contract, blk[j], id, 8.5)).min(15) as u8;
            let xi1 = trunc_i8(fma(contract, blk[j + 16], id, 8.5)).min(15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
        out.extend_from_slice(&qs);
    }
    out
}

fn quantize_q4_1(x: &[f32], contract: FmaContract) -> Vec<u8> {
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
            let xi0 = trunc_i8(fma(contract, blk[j] - min, id, 0.5)).min(15) as u8;
            let xi1 = trunc_i8(fma(contract, blk[j + 16] - min, id, 0.5)).min(15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
        out.extend_from_slice(&qs);
    }
    out
}

fn quantize_q5_0(x: &[f32], contract: FmaContract) -> Vec<u8> {
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
            let xi0 = trunc_i8(fma(contract, blk[j], id, 16.5)).min(31) as u8;
            let xi1 = trunc_i8(fma(contract, blk[j + 16], id, 16.5)).min(31) as u8;
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

fn quantize_q5_1(x: &[f32], contract: FmaContract) -> Vec<u8> {
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
            let xi0 = trunc_i8(fma(contract, blk[j] - min, id, 0.5)) as u8;
            let xi1 = trunc_i8(fma(contract, blk[j + 16] - min, id, 0.5)) as u8;
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

// === K-quant encoders (llama.cpp's `quantize_row_q{4,5,6}_K_ref`) ===
//
// The K-quant reference is searching: `make_qkx2_quants` scans `nstep + 1`
// candidate scale/min pairs, and `make_qx_quants` re-derives the least-squares
// scale for 19 candidate `iscale` values, keeping the best by weighted error.
// Every candidate is evaluated in the same `a*b + c` shape, so each one is a
// rounding decision the FMA contraction can move by one quant. The whole
// arithmetic below therefore mirrors the C line for line — including which
// products are fused, which is *not* a free choice: it is what `-ffp-contract=fast`
// does to the C, and a plain `a * b + c` here differs from `llama-quantize` at
// exact rounding boundaries (see `docs/GGUF-TOOLING.md` §4.2).

/// llama.cpp's `nearest_int`: round to nearest, ties to **even**, via the
/// `fval + 1.5·2²³` mantissa trick — not `f32::round`, which rounds ties away
/// from zero. At an exact `.5` the two differ by one, which is one quant.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// `nearest_int(a * b)`: `-ffp-contract=fast` folds the magic constant into the
/// product's FMA. The production object emits `fmadd s0, a, b, #12582912.0`
/// (`fmov w0, #0x4b400000`) and only then reads the mantissa, so the product is
/// **not** rounded before the add. Writing `nearest_int(a * b)` rounds twice and
/// picks a different integer at a boundary.
#[inline]
fn nearest_int_mul(a: f32, b: f32, contract: FmaContract) -> i32 {
    let val = fma(contract, a, b, 12_582_912.0f32);
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// The weight `make_qx_quants` assigns to element `i` when no explicit `qw`
/// was supplied: `rmse_type` 1 → `x²`, 2 → `1`, 3 → `|x|`, else `sqrt(|x|)`.
#[inline]
fn rmse_weight(rmse_type: i32, x: f32) -> f32 {
    match rmse_type {
        1 => x * x,
        2 => 1.0,
        3 => x.abs(),
        _ => x.abs().sqrt(),
    }
}

/// llama.cpp's `make_qx_quants` (ggml-quants.c), verbatim arithmetic.
///
/// Returns the scale and writes `nmax`-offset quants into `l`. Only the
/// `rmse_type == 1, qw == None` configuration is reached from q6_K, but the
/// whole search is ported so the function is the reference's function.
fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
    contract: FmaContract,
) -> f32 {
    let (mut max, mut amax) = (0.0f32, 0.0f32);
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        // all zero
        for v in l[..n].iter_mut() {
            *v = 0;
        }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int_mul(iscale, x[i], contract);
            l[i] = (nmax + li.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut return_early = false;
    let mut rmse_type = rmse_type;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let li = nearest_int_mul(iscale, x[i], contract).clamp(-nmax, nmax - 1);
        l[i] = (li + nmax) as i8;
        let w = qw.map_or_else(|| rmse_weight(rmse_type, x[i]), |q| q[i]);
        sumlx = fma(contract, w * x[i], li as f32, sumlx);
        suml2 = fma(contract, w * li as f32, li as f32, suml2);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9..=9i32 {
        if is == 0 {
            continue;
        }
        // The original spelling is `-0.1f32.mul_add(is, nmax)`, and the method
        // call binds first: the fused expression is `0.1*is + nmax` and only
        // *then* negated (llama.cpp computes `-(nmax + 0.1f*is)`), which is not
        // `fma(-0.1, is, nmax)` — the sign of `nmax` would move.
        iscale = -fma(contract, 0.1f32, is as f32, nmax as f32) / max;
        // The search loop is *vectorized* by GCC (4-wide), and the reduction is
        // a plain `fmul` product per element plus an in-order `fadd` sum — it
        // does NOT contract the `+=` into an FMA, unlike the scalar initial loop
        // above. The two therefore round differently, and the search's verdict
        // depends on it. (Read off `make_qx_quants.constprop.1`.)
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let li = nearest_int_mul(iscale, x[i], contract).clamp(-nmax, nmax - 1);
            let w = qw.map_or_else(|| rmse_weight(rmse_type, x[i]), |q| q[i]);
            let lf = li as f32;
            sumlx += (w * x[i]) * lf;
            suml2 += (w * lf) * lf;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int_mul(iscale, x[i], contract).clamp(-nmax, nmax - 1);
                l[i] = (li + nmax) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// llama.cpp's `make_qkx2_quants` (ggml-quants.c), `use_mad == false`.
///
/// This is the search q4_K and q5_K use: `nstep + 1` candidate `iscale` values,
/// each ranked by the weighted squared error of its best-fit `(scale, min)`.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    contract: FmaContract,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x = fma(contract, w, x[i], sum_x);
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for v in l[..n].iter_mut() {
            *v = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = nearest_int_mul(iscale, x[i] - min, contract);
        l[i] = li.clamp(0, nmax) as u8;
        let diff = fma(contract, scale, l[i] as f32, min) - x[i];
        let diff = diff * diff;
        best_error = fma(contract, weights[i], diff, best_error);
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (fma(contract, rdelta, is as f32, rmin) + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let li = nearest_int_mul(iscale, x[i] - min, contract).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            let lf = li as f32;
            let wl = w * lf;
            sum_l += wl;
            sum_l2 = fma(contract, wl, lf, sum_l2);
            sum_xl = fma(contract, wl, x[i], sum_xl);
        }
        // `a*b - c*d` contracts as `fma(a, b, -(c*d))` — the *left* product
        // is the fused one (the production object computes this with
        // `fmul` + `fnmsub`). All three of these are rounding boundaries the
        // search ranks candidates by, so the contraction is load-bearing.
        let d = fma(contract, sum_w, sum_l2, -(sum_l * sum_l));
        if d > 0.0 {
            let mut this_scale = fma(contract, sum_x, -sum_l, sum_w * sum_xl) / d;
            let mut this_min = fma(contract, sum_l2, sum_x, -(sum_l * sum_xl)) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = fma(contract, this_scale, laux[i] as f32, this_min) - x[i];
                let diff = diff * diff;
                cur_error = fma(contract, weights[i], diff, cur_error);
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// llama.cpp's `get_scale_min_k4`: unpack sub-block `j`'s 6-bit scale and min
/// from the 12-byte packed `scales` field.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// What `quantize_row_q4_K_ref` and `quantize_row_q5_K_ref` share: the eight
/// `make_qkx2_quants` sub-blocks, the 6-bit scale/min packing and the final
/// re-round of every quant against the *stored* (f16-rounded) `d`/`dmin`.
///
/// `nmax` is 15 for q4_K and 31 for q5_K; `rmin`/`nstep` are the reference's
/// per-type search parameters.
fn q4k_q5k_common(
    x: &[f32],
    nmax: i32,
    rmin: f32,
    nstep: i32,
    contract: FmaContract,
) -> ([u8; 12], u16, u16, [u8; 256]) {
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; 8];
    let mut scales = [0.0f32; 8];
    let mut l_all = [0u8; 256];
    let mut laux = [0u8; 32];
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for j in 0..8 {
        let xb = &x[32 * j..32 * j + 32];
        // llama.cpp weights each element by `av_x + |x|` with
        // `av_x = sqrt(sum(x²)/32)`. The sum is contracted to an FMA by
        // `-ffp-contract=fast` (seen as `fmadd s0, s1, s1, s0` in the
        // production object), and the weight moves the search's error metric,
        // so the contraction is part of the encoder.
        let mut sum_x2 = 0.0f32;
        for l in 0..32 {
            sum_x2 = fma(contract, xb[l], xb[l], sum_x2);
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for l in 0..32 {
            weights[l] = av_x + xb[l].abs();
        }
        scales[j] = make_qkx2_quants(
            32,
            nmax,
            xb,
            &weights,
            &mut l_all[32 * j..32 * j + 32],
            &mut mins[j],
            &mut laux,
            rmin,
            0.1,
            nstep,
            contract,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut sc_packed = [0u8; 12];
    for j in 0..8 {
        let ls = (nearest_int_mul(inv_scale, scales[j], contract) as u8).min(63);
        let lm = (nearest_int_mul(inv_min, mins[j], contract) as u8).min(63);
        if j < 4 {
            sc_packed[j] = ls;
            sc_packed[j + 4] = lm;
        } else {
            sc_packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc_packed[j - 4] |= (ls >> 4) << 6;
            sc_packed[j] |= (lm >> 4) << 6;
        }
    }
    let d = half::f16::from_f32(max_scale / 63.0);
    let dmin = half::f16::from_f32(max_min / 63.0);
    let d_f32 = d.to_f32();
    let dmin_f32 = dmin.to_f32();
    for j in 0..8 {
        let (sc, m) = get_scale_min_k4(j, &sc_packed);
        let dq = d_f32 * sc as f32;
        if dq == 0.0 {
            // The reference `continue`s, leaving the `make_qkx2_quants` quants
            // for this sub-block in `L` — reproduced by not touching them.
            continue;
        }
        let dm = dmin_f32 * m as f32;
        for ii in 0..32 {
            let l = nearest_int((x[32 * j + ii] + dm) / dq).clamp(0, nmax);
            l_all[32 * j + ii] = l as u8;
        }
    }
    (sc_packed, d.to_bits(), dmin.to_bits(), l_all)
}

fn quantize_q4_k(x: &[f32], contract: FmaContract) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / K_QUANT_BLOCK * 144);
    for xb in x.chunks_exact(K_QUANT_BLOCK) {
        let (scales, d, dmin, l) = q4k_q5k_common(xb, 15, -1.0, 20, contract);
        put_f16_bits(&mut out, d);
        put_f16_bits(&mut out, dmin);
        out.extend_from_slice(&scales);
        // Element j goes in the low nibble and j+32 in the high one, 32 bytes
        // at a time (QK_K/2 bytes total).
        for j in (0..K_QUANT_BLOCK).step_by(64) {
            for l2 in 0..32 {
                out.push(l[j + l2] | (l[j + l2 + 32] << 4));
            }
        }
    }
    out
}

fn quantize_q5_k(x: &[f32], contract: FmaContract) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / K_QUANT_BLOCK * 176);
    for xb in x.chunks_exact(K_QUANT_BLOCK) {
        let (scales, d, dmin, l) = q4k_q5k_common(xb, 31, -0.5, 15, contract);
        put_f16_bits(&mut out, d);
        put_f16_bits(&mut out, dmin);
        out.extend_from_slice(&scales);
        // The 5th bit is a separate 32-byte plane: two bits per 64-element
        // group, elements n+j in bits {m1}, n+j+32 in bits {m2}.
        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        let (mut m1, mut m2) = (1u8, 2u8);
        let mut qi = 0usize;
        for n in (0..K_QUANT_BLOCK).step_by(64) {
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                qs[qi + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            qi += 32;
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

fn quantize_q6_k(x: &[f32], contract: FmaContract) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / K_QUANT_BLOCK * 210);
    for xb in x.chunks_exact(K_QUANT_BLOCK) {
        // 16 sub-blocks of 16 elements, each with its own `make_qx_quants`
        // scale (rmse_type 1, no importance weights).
        let mut scales = [0.0f32; 16];
        let mut l_all = [0i8; K_QUANT_BLOCK];
        let (mut max_scale, mut max_abs_scale) = (0.0f32, 0.0f32);
        for ib in 0..16 {
            let scale = make_qx_quants(
                16,
                32,
                &xb[16 * ib..16 * ib + 16],
                &mut l_all[16 * ib..16 * ib + 16],
                1,
                None,
                contract,
            );
            scales[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        if max_abs_scale < GROUP_MAX_EPS {
            // An all-zero super-block is written as 210 zero bytes (d = 0).
            out.extend_from_slice(&[0u8; 210]);
            continue;
        }
        let iscale = -128.0 / max_scale;
        let d = half::f16::from_f32(1.0 / iscale);
        let mut sc = [0i8; 16];
        for ib in 0..16 {
            sc[ib] = nearest_int_mul(iscale, scales[ib], contract).min(127) as i8;
        }
        let d_f32 = d.to_f32();
        for j in 0..16 {
            let dq = d_f32 * sc[j] as f32;
            if dq == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let l = nearest_int(xb[16 * j + ii] / dq).clamp(-32, 31);
                l_all[16 * j + ii] = (l + 32) as i8;
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let (mut qli, mut qhi) = (0usize, 0usize);
        for j in (0..K_QUANT_BLOCK).step_by(128) {
            for l in 0..32 {
                let q1 = (l_all[j + l] as u8) & 0xF;
                let q2 = (l_all[j + l + 32] as u8) & 0xF;
                let q3 = (l_all[j + l + 64] as u8) & 0xF;
                let q4 = (l_all[j + l + 96] as u8) & 0xF;
                ql[qli + l] = q1 | (q3 << 4);
                ql[qli + l + 32] = q2 | (q4 << 4);
                qh[qhi + l] = ((l_all[j + l] as u8) >> 4)
                    | (((l_all[j + l + 32] as u8) >> 4) << 2)
                    | (((l_all[j + l + 64] as u8) >> 4) << 4)
                    | (((l_all[j + l + 96] as u8) >> 4) << 6);
            }
            qli += 64;
            qhi += 32;
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&sc.map(|v| v as u8));
        put_f16_bits(&mut out, d.to_bits());
    }
    out
}

/// C's `(int8_t)v`: truncation toward zero. Rust's `as i8` also truncates
/// toward zero, which is what the reference relies on.
#[inline]
fn trunc_i8(v: f32) -> i8 {
    v as i8
}

#[cfg(test)]
mod tests;
