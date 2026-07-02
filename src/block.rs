// Quantized Block Types + fp16 Utilities
// Translated from: llama.cpp/ggml/src/ggml-common.h (block structs)
//   + ggml/src/ggml-impl.h (fp16 conversion)
//   + ggml/include/ggml.h (fp16 type)

// === fp16 type (ggml.h line 370) ===

/// IEEE 754-2008 half-precision float16
pub type Fp16 = u16;

// === fp16 <-> f32 conversion (ggml-impl.h lines 366-430) ===

/// Convert fp16 to f32 using half crate (matches minfer2's f16::from_bits)
#[inline]
pub fn fp16_to_f32(h: Fp16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Convert f32 to fp16 using half crate (matches minfer2's f16::from_f32)
/// Translated from: ggml_compute_fp32_to_fp16 (ggml-impl.h lines 406-430)
#[inline]
pub fn f32_to_fp16(f: f32) -> Fp16 {
    half::f16::from_f32(f).to_bits()
}

// === Block byte-size constants (ggml-common.h) ===

pub const Q4B: usize = 18;   // sizeof(block_q4_0)
pub const Q41B: usize = 20;  // sizeof(block_q4_1)
pub const Q8B: usize = 34;   // sizeof(block_q8_0)
pub const Q4KB: usize = 144; // sizeof(block_q4_k)
pub const Q6KB: usize = 210; // sizeof(block_q6_k)
pub const Q8KB: usize = 34;  // sizeof(block_q8_0), same as Q8B

/// Unpack 8 scales and 8 mins from the 12-byte scales field of a Q4_K block.
/// Matches llama.cpp `get_scale_min_k4`.
#[inline]
pub fn unpack_q4k_scales(sc: &[u8; 12]) -> ([i32; 8], [i32; 8]) {
    let mut scales = [0i32; 8];
    let mut mins = [0i32; 8];
    for j in 0..4 {
        scales[j] = (sc[j] & 0x3F) as i32;
        mins[j]   = (sc[j + 4] & 0x3F) as i32;
    }
    for j in 4..8 {
        scales[j] = ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as i32;
        mins[j]   = ((sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4)) as i32;
    }
    (scales, mins)
}

// === Quantized Block Structures (ggml-common.h) ===

// Q4_0 — 4-bit quantization, 32 elements per block (line 184-189)
// Each value is stored as a 4-bit nibble (signed, offset by 8)
// Scale is fp16
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: Fp16,           // delta (scale)
    pub qs: [u8; 16],      // nibbles / quants (32 × 4-bit = 16 bytes)
}

// Q4_1 — 4-bit quantization with min, 32 elements per block (line 191-202)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: Fp16,           // delta (scale)
    pub m: Fp16,           // min
    pub qs: [u8; 16],      // nibbles / quants
}

// Q5_0 — 5-bit quantization, 32 elements per block (line 219-225)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub d: Fp16,           // delta
    pub qh: [u8; 4],       // 5-th bit of quants (32 bits = 4 bytes)
    pub qs: [u8; 16],      // nibbles / quants (low 4 bits)
}

// Q5_1 — 5-bit quantization with min, 32 elements per block (line 227-239)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub d: Fp16,           // delta
    pub m: Fp16,           // min
    pub qh: [u8; 4],       // 5-th bit of quants
    pub qs: [u8; 16],      // nibbles / quants (low 4 bits)
}

// Q8_0 — 8-bit quantization, 32 elements per block (line 241-246)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: Fp16,           // delta
    pub qs: [i8; 32],      // quants
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self { d: 0, qs: [0i8; 32] }
    }
}

// Q8_1 — 8-bit quantization with sum, 32 elements per block (line 248-259)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub d: Fp16,           // delta
    pub s: Fp16,           // d * sum(qs[i])
    pub qs: [i8; 32],      // quants
}

// Q2_K — 2-bit super-block quantization, 256 elements (line 288-299)
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each, effectively 2.625 bits per weight
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ2_K {
    pub scales: [u8; 16],  // scales and mins, quantized with 4 bits (QK_K/16)
    pub qs: [u8; 64],      // quants (QK_K/4)
    pub d: Fp16,           // super-block scale for quantized scales
    pub dmin: Fp16,        // super-block scale for quantized mins
}

// Q3_K — 3-bit super-block quantization, 256 elements (line 305-311)
// Effectively 3.4375 bits per weight
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ3_K {
    pub hmask: [u8; 32],   // quants - high bit (QK_K/8)
    pub qs: [u8; 64],      // quants - low 2 bits (QK_K/4)
    pub scales: [u8; 12],  // scales, quantized with 6 bits
    pub d: Fp16,           // super-block scale
}

// Q4_K — 4-bit super-block quantization, 256 elements (line 317-328)
// 8 blocks of 32 elements each, effectively 4.5 bits per weight
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_K {
    pub d: Fp16,           // super-block scale for quantized scales
    pub dmin: Fp16,        // super-block scale for quantized mins
    pub scales: [u8; 12],  // scales and mins, quantized with 6 bits (K_SCALE_SIZE)
    pub qs: [u8; 128],     // 4-bit quants (QK_K/2)
}

// Q5_K — 5-bit super-block quantization, 256 elements (line 334-346)
// 8 blocks of 32 elements each, effectively 5.5 bits per weight
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ5_K {
    pub d: Fp16,           // super-block scale for quantized scales
    pub dmin: Fp16,        // super-block scale for quantized mins
    pub scales: [u8; 12],  // scales and mins, quantized with 6 bits (K_SCALE_SIZE)
    pub qh: [u8; 32],      // quants, high bit (QK_K/8)
    pub qs: [u8; 128],     // quants, low 4 bits (QK_K/2)
}

// Q6_K — 6-bit super-block quantization, 256 elements (line 352-358)
// 16 blocks of 16 elements each, effectively 6.5625 bits per weight
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ6_K {
    pub ql: [u8; 128],     // quants, lower 4 bits (QK_K/2)
    pub qh: [u8; 64],      // quants, upper 2 bits (QK_K/4)
    pub scales: [i8; 16],  // scales, quantized with 8 bits (QK_K/16)
    pub d: Fp16,           // super-block scale
}

// Q8_K — 8-bit intermediate quantization, 256 elements (line 361-366)
// Used for intermediate quantization and dot products
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_K {
    pub d: f32,            // delta
    pub qs: [i8; 256],     // quants (QK_K)
    pub bsums: [i16; 16],  // sum of quants in groups of 16 (QK_K/16)
}

// Q1_0 — 1-bit quantization, 128 elements per block (line 177-182)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ1_0 {
    pub d: Fp16,           // delta
    pub qs: [u8; 16],      // bits / quants (128/8 = 16 bytes)
}

// TQ1_0 — Ternary 1-bit, 256 elements (line 266-271)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockTq1_0 {
    pub qs: [u8; 48],      // quants (QK_K - 4 * QK_K / 64) / 5 = 48
    pub qh: [u8; 4],       // QK_K/64 = 256/64 = 4
    pub d: Fp16,
}

// TQ2_0 — Ternary 2-bit, 256 elements (line 274-278)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockTq2_0 {
    pub qs: [u8; 64],      // QK_K/4 = 64
    pub d: Fp16,
}

// IQ2_XXS — "True" 2-bit, 256 elements (line 371-375)
// 2.0625 bpw
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq2Xxs {
    pub d: Fp16,
    pub qs: [u16; 32],     // QK_K/8 * sizeof(uint16_t)
}

// IQ2_XS — 2.3125 bpw (line 378-383)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq2Xs {
    pub d: Fp16,
    pub qs: [u16; 32],     // QK_K/8
    pub scales: [u8; 8],   // QK_K/32 = 8
}

// IQ2_S — 2.5625 bpw (line 386-392)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq2S {
    pub d: Fp16,
    pub qs: [u8; 64],      // QK_K/4
    pub qh: [u8; 8],       // QK_K/32 = 8
    pub scales: [u8; 8],   // QK_K/32 = 8
}

// IQ3_XXS — "True" 3-bit, 256 elements (line 397-401)
// 3.0625 bpw
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub d: Fp16,
    pub qs: [u8; 96],      // 3*QK_K/8 = 96
}

// IQ3_S — 3.4375 bpw (line 405-412)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq3S {
    pub d: Fp16,
    pub qs: [u8; 64],      // QK_K/4
    pub qh: [u8; 8],       // QK_K/32 = 8
    pub signs: [u8; 32],   // QK_K/8 = 32
    pub scales: [u8; 4],   // IQ3S_N_SCALE = QK_K/64 = 4
}

// IQ1_S — 1.5625 bpw (line 415-420)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq1S {
    pub d: Fp16,
    pub qs: [u8; 32],      // QK_K/8 = 32
    pub qh: [u16; 8],      // QK_K/32 = 8
}

// IQ1_M — 1.75 bpw (line 423-428)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq1M {
    pub qs: [u8; 32],      // QK_K/8 = 32
    pub qh: [u8; 16],      // QK_K/16 = 16
    pub scales: [u8; 8],   // QK_K/32 = 8
}

// IQ1_M scale type (line 431-434)
#[derive(Clone, Copy)]
#[repr(C)]
pub union Iq1mScale {
    pub f16: Fp16,
    pub u16: u16,
}

// IQ4_NL — Non-linear 4-bit, 32 elements (line 437-442)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq4Nl {
    pub d: Fp16,
    pub qs: [u8; 16],      // QK4_NL/2 = 16
}

// IQ4_XS — 4-bit with scales, 256 elements (line 444-450)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockIq4Xs {
    pub d: Fp16,
    pub scales_h: u16,
    pub scales_l: [u8; 4], // QK_K/64 = 4
    pub qs: [u8; 128],     // QK_K/2 = 128
}

// MXFP4 — MXFP4 4-bit, 32 elements (line 204-209)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockMxfp4 {
    pub e: u8,             // E8M0
    pub qs: [u8; 16],      // QK_MXFP4/2
}

// NVFP4 — NVFP4 4-bit, 64 elements (line 211-217)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockNvfp4 {
    pub d: [u8; 4],        // UE4M3 scales (64/16 = 4)
    pub qs: [u8; 32],      // packed 4-bit E2M1 values (64/2 = 32)
}

// === Static assertions equivalent ===
// These verify the struct sizes match the C counterparts
// Values from the C static_asserts in ggml-common.h
const _: () = {
    assert!(core::mem::size_of::<BlockQ4_0>() == 2 + 16);
    assert!(core::mem::size_of::<BlockQ4_1>() == 2 + 2 + 16);
    assert!(core::mem::size_of::<BlockQ5_0>() == 2 + 4 + 16);
    assert!(core::mem::size_of::<BlockQ5_1>() == 2 + 2 + 4 + 16);
    assert!(core::mem::size_of::<BlockQ8_0>() == 2 + 32);
    assert!(core::mem::size_of::<BlockQ8_1>() == 2 + 2 + 32);
    assert!(core::mem::size_of::<BlockQ2_K>() == 2 + 2 + 16 + 64);
    assert!(core::mem::size_of::<BlockQ3_K>() == 2 + 32 + 64 + 12);
    assert!(core::mem::size_of::<BlockQ4_K>() == 2 + 2 + 12 + 128);
    assert!(core::mem::size_of::<BlockQ5_K>() == 2 + 2 + 12 + 32 + 128);
    assert!(core::mem::size_of::<BlockQ6_K>() == 2 + 16 + 128 + 64);
    assert!(core::mem::size_of::<BlockQ8_K>() == 4 + 256 + 16 * 2);
    assert!(core::mem::size_of::<BlockQ1_0>() == 2 + 16);
};
