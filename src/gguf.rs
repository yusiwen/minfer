// GGUF Format Parser

use std::fmt;
use std::mem;
use std::ptr;

// === [#767] a parse refusal is a value ===
//
// [#745] S3 gave the model layer a named error; the parser it sits on still reported
// out of band: the deep `Option` chain recorded a note in a thread-local, and only the
// outermost call turned that note into a `GgufLoadError`. A helper's signature therefore
// never said it could refuse, and a stage's specific reason could be overwritten by the
// next stage's aggregate message before anyone read it.
//
// Now every stage returns its reason. `ParseResult`'s error is a complete `GGUF: ...`
// sentence, which is exactly what `GgufLoadError::reason` carries, so there is no side
// channel left to skip.
//
// [#767]: https://github.com/yusiwen/minfer/issues/767
type ParseResult<T> = Result<T, String>;

// === Constants (from gguf.h) ===

pub const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const GGUF_KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

const GGUF_MAX_STRING_LENGTH: u64 = 1024 * 1024 * 1024;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1024 * 1024 * 1024;

// Note: GGML_MAX_DIMS and GGML_MAX_NAME from ggml.h
const GGML_MAX_DIMS: usize = 4;
const GGML_MAX_NAME: usize = 64;

// === GGUF Types (from gguf.h lines 53-68) ===

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    /// `None` for a value the format does not define: the reader turns that into a
    /// refusal naming the byte, never a panic ([#767]).
    fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(GgufType::Uint8),
            1 => Some(GgufType::Int8),
            2 => Some(GgufType::Uint16),
            3 => Some(GgufType::Int16),
            4 => Some(GgufType::Uint32),
            5 => Some(GgufType::Int32),
            6 => Some(GgufType::Float32),
            7 => Some(GgufType::Bool),
            8 => Some(GgufType::String),
            9 => Some(GgufType::Array),
            10 => Some(GgufType::Uint64),
            11 => Some(GgufType::Int64),
            12 => Some(GgufType::Float64),
            _ => None,
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            GgufType::Uint8 => 1,
            GgufType::Int8 => 1,
            GgufType::Uint16 => 2,
            GgufType::Int16 => 2,
            GgufType::Uint32 => 4,
            GgufType::Int32 => 4,
            GgufType::Float32 => 4,
            GgufType::Bool => 1,
            GgufType::String => 0, // undefined
            GgufType::Array => 0,  // undefined
            GgufType::Uint64 => 8,
            GgufType::Int64 => 8,
            GgufType::Float64 => 8,
        }
    }
}

// === GGML types (subset needed for GGUF — from ggml.h lines 389-433) ===

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
/// GGML quant type enum. Variant names follow the GGUF/ggml-common.h naming
/// convention verbatim (Q4_K, IQ2_XXS, …) so discriminator values and on-disk
/// names match upstream — hence the non-camel-case allow.
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // removed
    // Q4_3 = 5, // removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // Q4_0_4_4 = 31, // removed from gguf
    // Q4_0_4_8 = 32,
    // Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    // IQ4_NL_4_4 = 36,
    // IQ4_NL_4_8 = 37,
    // IQ4_NL_8_8 = 38,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
}

pub const GGML_TYPE_COUNT: i32 = 42;

impl GgmlType {
    /// `None` for a value ggml does not define (4 and 5 were removed): the parser
    /// turns it into a refusal naming the type byte, never a panic ([#767]).
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(GgmlType::F32),
            1 => Some(GgmlType::F16),
            2 => Some(GgmlType::Q4_0),
            3 => Some(GgmlType::Q4_1),
            6 => Some(GgmlType::Q5_0),
            7 => Some(GgmlType::Q5_1),
            8 => Some(GgmlType::Q8_0),
            9 => Some(GgmlType::Q8_1),
            10 => Some(GgmlType::Q2_K),
            11 => Some(GgmlType::Q3_K),
            12 => Some(GgmlType::Q4_K),
            13 => Some(GgmlType::Q5_K),
            14 => Some(GgmlType::Q6_K),
            15 => Some(GgmlType::Q8_K),
            16 => Some(GgmlType::IQ2_XXS),
            17 => Some(GgmlType::IQ2_XS),
            18 => Some(GgmlType::IQ3_XXS),
            19 => Some(GgmlType::IQ1_S),
            20 => Some(GgmlType::IQ4_NL),
            21 => Some(GgmlType::IQ3_S),
            22 => Some(GgmlType::IQ2_S),
            23 => Some(GgmlType::IQ4_XS),
            24 => Some(GgmlType::I8),
            25 => Some(GgmlType::I16),
            26 => Some(GgmlType::I32),
            27 => Some(GgmlType::I64),
            28 => Some(GgmlType::F64),
            29 => Some(GgmlType::IQ1_M),
            30 => Some(GgmlType::BF16),
            34 => Some(GgmlType::TQ1_0),
            35 => Some(GgmlType::TQ2_0),
            39 => Some(GgmlType::MXFP4),
            40 => Some(GgmlType::NVFP4),
            41 => Some(GgmlType::Q1_0),
            _ => None,
        }
    }

    /// Returns the number of bytes for one block of this type
    /// Corresponds to: ggml_type_size() — type_traits[].type_size (ggml.c lines 1306-1309)
    /// These are sizeof() values from ggml-common.h block structs
    pub fn type_size(&self) -> usize {
        match self {
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            GgmlType::F64 => 8,
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            // sizeof(block_q1_0) = sizeof(ggml_half) + QK1_0/8 = 2 + 4 = 6
            GgmlType::Q1_0 => 6,
            // sizeof(block_q4_0) = sizeof(ggml_half) + QK4_0/2 = 2 + 16 = 18
            GgmlType::Q4_0 => 18,
            // sizeof(block_q4_1) = sizeof(ggml_half)*2 + QK4_1/2 = 2 + 2 + 16 = 20
            GgmlType::Q4_1 => 20,
            // sizeof(block_q5_0) = sizeof(ggml_half) + QK5_0/2 + QK5_0/8 = 2 + 16 + 4 = 22
            GgmlType::Q5_0 => 22,
            // sizeof(block_q5_1) = sizeof(ggml_half)*2 + QK5_1/2 + QK5_1/8 = 2+2+16+4 = 24
            GgmlType::Q5_1 => 24,
            // sizeof(block_q8_0) = sizeof(ggml_half) + QK8_0 = 2 + 32 = 34
            GgmlType::Q8_0 => 34,
            // sizeof(block_q8_1) = sizeof(ggml_half)*2 + QK8_1 = 2+2+32 = 36
            GgmlType::Q8_1 => 36,
            // QK_K=256, super-block types — sizes from type_traits
            GgmlType::Q2_K => 70,
            GgmlType::Q3_K => 110,
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            GgmlType::Q8_K => 290,
            GgmlType::IQ2_XXS => 42,
            GgmlType::IQ2_XS => 36,
            GgmlType::IQ3_XXS => 58,
            GgmlType::IQ1_S => 36,
            GgmlType::IQ4_NL => 18,
            GgmlType::IQ3_S => 64,
            GgmlType::IQ2_S => 48,
            GgmlType::IQ4_XS => 28,
            GgmlType::IQ1_M => 54,
            GgmlType::BF16 => 2,
            GgmlType::TQ1_0 => 68,
            GgmlType::TQ2_0 => 100,
            GgmlType::MXFP4 => 72,
            GgmlType::NVFP4 => 80,
            // deprecated types 4, 5 have type_size=0 but aren't enum variants
        }
    }

    /// Returns the block size (number of values per block)
    /// Corresponds to: ggml_blck_size() — from type_traits[].blck_size (ggml.c line 1300-1303)
    pub fn blck_size(&self) -> i64 {
        match self {
            GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 => 1,
            GgmlType::F64 | GgmlType::F32 | GgmlType::F16 => 1,
            GgmlType::BF16 => 1,
            GgmlType::TQ1_0 => 256,
            GgmlType::TQ2_0 => 256,
            GgmlType::Q1_0 => 32,
            GgmlType::Q4_0 => 32,
            GgmlType::Q4_1 => 32,
            GgmlType::Q5_0 => 32,
            GgmlType::Q5_1 => 32,
            GgmlType::Q8_0 => 32,
            GgmlType::Q8_1 => 32,
            GgmlType::Q2_K => 256,
            GgmlType::Q3_K => 256,
            GgmlType::Q4_K => 256,
            GgmlType::Q5_K => 256,
            GgmlType::Q6_K => 256,
            GgmlType::Q8_K => 256,
            GgmlType::IQ2_XXS => 256,
            GgmlType::IQ2_XS => 256,
            GgmlType::IQ3_XXS => 256,
            GgmlType::IQ1_S => 256,
            GgmlType::IQ4_NL => 32,
            GgmlType::IQ3_S => 256,
            GgmlType::IQ2_S => 256,
            GgmlType::IQ4_XS => 32,
            GgmlType::IQ1_M => 256,
            GgmlType::MXFP4 => 32,
            GgmlType::NVFP4 => 32,
            // deprecated types 4, 5 have blck_size=0 but aren't enum variants
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::Q4_0 => "q4_0",
            GgmlType::Q4_1 => "q4_1",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q5_1 => "q5_1",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Q8_1 => "q8_1",
            GgmlType::Q2_K => "q2_K",
            GgmlType::Q3_K => "q3_K",
            GgmlType::Q4_K => "q4_K",
            GgmlType::Q5_K => "q5_K",
            GgmlType::Q6_K => "q6_K",
            GgmlType::Q8_K => "q8_K",
            GgmlType::I8 => "i8",
            GgmlType::I16 => "i16",
            GgmlType::I32 => "i32",
            GgmlType::I64 => "i64",
            GgmlType::F64 => "f64",
            GgmlType::IQ2_XXS => "iq2_xxs",
            GgmlType::IQ2_XS => "iq2_xs",
            GgmlType::IQ3_XXS => "iq3_xxs",
            GgmlType::IQ1_S => "iq1_s",
            GgmlType::IQ4_NL => "iq4_nl",
            GgmlType::IQ3_S => "iq3_s",
            GgmlType::IQ2_S => "iq2_s",
            GgmlType::IQ4_XS => "iq4_xs",
            GgmlType::IQ1_M => "iq1_m",
            GgmlType::BF16 => "bf16",
            GgmlType::TQ1_0 => "tq1_0",
            GgmlType::TQ2_0 => "tq2_0",
            GgmlType::MXFP4 => "mxfp4",
            GgmlType::NVFP4 => "nvfp4",
            GgmlType::Q1_0 => "q1_0",
        }
    }
}

// === Helper: GGML_PAD macro (ggml.h line 267) ===

#[inline]
pub fn ggml_pad(x: usize, n: usize) -> usize {
    // ((x) + (n) - 1) & ~((n) - 1)
    // Assumes n is power of 2
    (x + n - 1) & !(n - 1)
}

// === GGUF KV pair (gguf.cpp lines 131-210, struct gguf_kv) ===

#[derive(Clone)]
pub struct GgufKv {
    pub key: String,
    pub is_array: bool,
    pub type_: GgufType,
    // raw binary data for non-string types
    pub data: Vec<u8>,
    // string data
    pub data_string: Vec<String>,
}

impl GgufKv {
    pub fn new_u8(key: String, value: u8) -> Self {
        let mut data = Vec::with_capacity(1);
        data.push(value);
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Uint8,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_i8(key: String, value: i8) -> Self {
        let mut data = Vec::with_capacity(1);
        data.push(value as u8);
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Int8,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_u16(key: String, value: u16) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Uint16,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_i16(key: String, value: i16) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Int16,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_u32(key: String, value: u32) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Uint32,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_i32(key: String, value: i32) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Int32,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_f32(key: String, value: f32) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Float32,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_f64(key: String, value: f64) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Float64,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_bool(key: String, value: bool) -> Self {
        let mut data = Vec::with_capacity(1);
        data.push(if value { 1 } else { 0 });
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Bool,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_u64(key: String, value: u64) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Uint64,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_i64(key: String, value: i64) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Int64,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_string(key: String, value: String) -> Self {
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: vec![value],
        }
    }

    pub fn new_string_array(key: String, value: Vec<String>) -> Self {
        GgufKv {
            key,
            is_array: true,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: value,
        }
    }

    pub fn new_array(key: String, type_: GgufType, raw_data: Vec<u8>) -> Self {
        GgufKv {
            key,
            is_array: true,
            type_,
            data: raw_data,
            data_string: Vec::new(),
        }
    }

    pub fn get_key(&self) -> &str {
        &self.key
    }

    pub fn get_type(&self) -> GgufType {
        self.type_
    }

    pub fn get_ne(&self) -> usize {
        if self.type_ == GgufType::String {
            let ne = self.data_string.len();
            assert!(self.is_array || ne == 1);
            return ne;
        }
        let type_size = self.type_.type_size();
        assert!(self.data.len() % type_size == 0);
        let ne = self.data.len() / type_size;
        assert!(self.is_array || ne == 1);
        ne
    }

    pub fn get_val_u8(&self, i: usize) -> u8 {
        assert!(self.type_ == GgufType::Uint8);
        self.data[i]
    }

    pub fn get_val_i8(&self, i: usize) -> i8 {
        assert!(self.type_ == GgufType::Int8);
        self.data[i] as i8
    }

    pub fn get_val_u16(&self, i: usize) -> u16 {
        assert!(self.type_ == GgufType::Uint16);
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(&self.data[i * 2..(i + 1) * 2]);
        u16::from_le_bytes(bytes)
    }

    pub fn get_val_i16(&self, i: usize) -> i16 {
        assert!(self.type_ == GgufType::Int16);
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(&self.data[i * 2..(i + 1) * 2]);
        i16::from_le_bytes(bytes)
    }

    pub fn get_val_u32(&self, i: usize) -> u32 {
        assert!(self.type_ == GgufType::Uint32);
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.data[i * 4..(i + 1) * 4]);
        u32::from_le_bytes(bytes)
    }

    pub fn get_val_i32(&self, i: usize) -> i32 {
        assert!(self.type_ == GgufType::Int32);
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.data[i * 4..(i + 1) * 4]);
        i32::from_le_bytes(bytes)
    }

    pub fn get_val_f32(&self, i: usize) -> f32 {
        assert!(self.type_ == GgufType::Float32);
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.data[i * 4..(i + 1) * 4]);
        f32::from_le_bytes(bytes)
    }

    pub fn get_val_u64(&self, i: usize) -> u64 {
        assert!(self.type_ == GgufType::Uint64);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(bytes)
    }

    pub fn get_val_i64(&self, i: usize) -> i64 {
        assert!(self.type_ == GgufType::Int64);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[i * 8..(i + 1) * 8]);
        i64::from_le_bytes(bytes)
    }

    pub fn get_val_f64(&self, i: usize) -> f64 {
        assert!(self.type_ == GgufType::Float64);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[i * 8..(i + 1) * 8]);
        f64::from_le_bytes(bytes)
    }

    pub fn get_val_bool(&self, i: usize) -> bool {
        assert!(self.type_ == GgufType::Bool);
        self.data[i] != 0
    }

    pub fn get_val_str(&self, i: usize) -> &str {
        assert!(self.type_ == GgufType::String);
        &self.data_string[i]
    }
}

// === GGUF Tensor Info (gguf.cpp lines 212-215, struct gguf_tensor_info) ===

#[derive(Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub ne: [i64; GGML_MAX_DIMS],   // number of elements per dimension
    pub nb: [usize; GGML_MAX_DIMS], // stride in bytes per dimension
    pub type_: GgmlType,
    pub offset: u64, // offset from start of data section
}

impl GgufTensorInfo {
    /// The tensor's payload size in bytes: whole quant blocks times the block size. This is
    /// the number the loader slices the part with, and (E5 S2) what the `auto` offload fit
    /// measures each block by — from the index, before anything is loaded or registered.
    pub fn nbytes(&self) -> usize {
        let n: i64 = self.ne.iter().product();
        (n / self.type_.blck_size()) as usize * self.type_.type_size()
    }
}

// === GGUF Context (gguf.cpp lines 217-228, struct gguf_context) ===

#[derive(Clone)]
pub struct GgufContext {
    pub version: u32,
    pub kv: Vec<GgufKv>,
    pub info: Vec<GgufTensorInfo>,
    pub alignment: usize,
    pub offset: usize, // offset of data section from beginning of file
    pub size: usize,   // size of data section in bytes
                       // In C++: void * data = nullptr — we'll store a reference to the mmap'd data
                       // This is handled by the caller (loader.rs), not stored here
}

// Raw key/value accessors (from gguf.cpp lines 1004-1193) still reached from
// production (`find_key`/`get_key`/`get_n_kv`/`get_val_u32`) or by the
// `gguf_write` round-trip tests (`#[cfg(test)]`). The rest of the cpp accessor
// surface had no caller in any build and was deleted in [#242].
//
// [#242]: https://github.com/yusiwen/minfer/issues/242
impl GgufContext {
    // === Accessor functions (from gguf.cpp lines 1004-1193) ===

    /// Test-only (#238): driven by `gguf_write::tests::every_metadata_type_round_trips_through_the_parser`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn get_data_offset(&self) -> usize {
        self.offset
    }

    pub fn get_n_kv(&self) -> i64 {
        self.kv.len() as i64
    }

    pub fn find_key(&self, key: &str) -> i64 {
        // return -1 if key not found
        // gguf.cpp lines 1020-1034
        let n_kv = self.get_n_kv();
        for i in 0..n_kv {
            if key == self.get_key(i) {
                return i;
            }
        }
        -1
    }

    pub fn get_key(&self, key_id: i64) -> &str {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        self.kv[key_id as usize].get_key()
    }

    /// Test-only (#238): driven by `gguf_write::tests::every_metadata_type_round_trips_through_the_parser`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn get_arr_type(&self, key_id: i64) -> GgufType {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].is_array);
        self.kv[key_id as usize].get_type()
    }

    /// Test-only (#238): driven by `gguf_write::tests::every_metadata_type_round_trips_through_the_parser`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn get_arr_data(&self, key_id: i64) -> &[u8] {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_type() != GgufType::String);
        &self.kv[key_id as usize].data
    }

    /// Test-only (#238): driven by `gguf_write::tests::every_metadata_type_round_trips_through_the_parser`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn get_arr_str(&self, key_id: i64, i: usize) -> &str {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_type() == GgufType::String);
        &self.kv[key_id as usize].data_string[i]
    }

    /// Test-only (#238): driven by `gguf_write::tests::every_metadata_type_round_trips_through_the_parser`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn get_arr_n(&self, key_id: i64) -> usize {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        self.kv[key_id as usize].get_ne()
    }

    pub fn get_val_u32(&self, key_id: i64) -> u32 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_u32(0)
    }

    /// Test-only (#238): driven by `gguf_write::tests::every_metadata_type_round_trips_through_the_parser`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn get_val_bool(&self, key_id: i64) -> bool {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_bool(0)
    }
}

impl fmt::Debug for GgufContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GgufContext {{ version: {}, n_kv: {}, n_tensors: {}, alignment: {}, offset: {}, size: {} }}",
            self.version, self.kv.len(), self.info.len(), self.alignment, self.offset, self.size)
    }
}

// === Helper: ggml_nelements (ggml.c lines 1259-1263) ===

fn ggml_nelements(ne: &[i64; GGML_MAX_DIMS]) -> i64 {
    ne[0] * ne[1] * ne[2] * ne[3]
}

// === Helper: ggml_nbytes (ggml.c lines 1271-1294) ===

fn ggml_nbytes(info: &GgufTensorInfo) -> usize {
    for i in 0..GGML_MAX_DIMS {
        if info.ne[i] <= 0 {
            return 0;
        }
    }

    let blck_size = info.type_.blck_size();
    if blck_size == 1 {
        let mut nbytes = info.type_.type_size();
        for i in 0..GGML_MAX_DIMS {
            nbytes += (info.ne[i] - 1) as usize * info.nb[i];
        }
        nbytes
    } else {
        let mut nbytes = info.ne[0] as usize * info.nb[0] / blck_size as usize;
        for i in 1..GGML_MAX_DIMS {
            nbytes += (info.ne[i] - 1) as usize * info.nb[i];
        }
        nbytes
    }
}

// === GGUF Reader (gguf.cpp lines 230-419, struct gguf_reader) ===

/// Reader that reads from a byte slice (mmap'd file), analogous to
/// gguf_buffer_reader / gguf_reader in llama.cpp
struct GgufReader<'a> {
    data: &'a [u8],
    offset: u64,
    nbytes_remain: u64,
}

impl<'a> GgufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        GgufReader {
            data,
            offset: 0,
            nbytes_remain: data.len() as u64,
        }
    }

    // read_raw: copy `size` bytes from current offset (gguf.cpp lines 381-412)
    fn read_raw(&mut self, dst: &mut [u8], size: usize) -> usize {
        if size == 0 {
            return 0;
        }

        let mut total_nread: usize = 0;

        while total_nread < size {
            let chunk_size = size - total_nread;
            let read_offset = self.offset as usize + total_nread;
            if read_offset >= self.data.len() {
                break;
            }
            let available = self.data.len() - read_offset;
            let to_copy = chunk_size.min(available);

            dst[total_nread..total_nread + to_copy]
                .copy_from_slice(&self.data[read_offset..read_offset + to_copy]);
            total_nread += to_copy;

            if to_copy != chunk_size {
                // reached_eof
                break;
            }
        }

        self.offset += total_nread as u64;
        assert!(total_nread as u64 <= self.nbytes_remain);
        self.nbytes_remain -= total_nread as u64;

        total_nread
    }

    // read a single value of type T (gguf.cpp lines 266-273)
    fn read_val<T: Copy>(&mut self) -> ParseResult<T> {
        let size = mem::size_of::<T>();
        if size as u64 > self.nbytes_remain {
            return Err(self.eof_reason(size as u64));
        }
        let mut dst = vec![0u8; size];
        let nread = self.read_raw(&mut dst, size);
        if nread != size {
            return Err(self.eof_reason(size as u64));
        }
        // Interpret as little-endian T
        let val = unsafe { ptr::read_unaligned(dst.as_ptr() as *const T) };
        Ok(val)
    }

    /// What a read needed, and where the part ran out. This is the detail the old
    /// thread-local note held and a later stage message then overwrote ([#767]).
    ///
    /// [#767]: https://github.com/yusiwen/minfer/issues/767
    fn eof_reason(&self, needed: u64) -> String {
        format!(
            "GGUF: unexpected end of file: needed {needed} bytes at offset {}, {} remain",
            self.offset, self.nbytes_remain
        )
    }

    // read bool as int8_t (gguf.cpp lines 313-320)
    fn read_bool(&mut self) -> ParseResult<bool> {
        let tmp: i8 = self.read_val()?;
        Ok(tmp != 0)
    }

    // read ggml_type as int32_t (gguf.cpp lines 322-329)
    fn read_ggml_type(&mut self) -> ParseResult<GgmlType> {
        let tmp: i32 = self.read_val()?;
        GgmlType::from_i32(tmp).ok_or_else(|| format!("GGUF: invalid GGML type: {tmp}"))
    }

    // read gguf_type as int32_t (gguf.cpp lines 331-338)
    fn read_gguf_type(&mut self) -> ParseResult<GgufType> {
        let tmp: i32 = self.read_val()?;
        GgufType::from_i32(tmp).ok_or_else(|| format!("GGUF: invalid GGUF type: {tmp}"))
    }

    // read string: first uint64 length, then bytes (gguf.cpp lines 340-355)
    fn read_string(&mut self) -> ParseResult<String> {
        let size: u64 = self.read_val()?;
        if size > GGUF_MAX_STRING_LENGTH {
            return Err(format!(
                "GGUF: string length {size} exceeds maximum {GGUF_MAX_STRING_LENGTH}"
            ));
        }
        if size > self.nbytes_remain {
            return Err(format!(
                "GGUF: string length {size} exceeds remaining file size {} bytes",
                self.nbytes_remain
            ));
        }
        let mut buf = vec![0u8; size as usize];
        let nread = self.read_raw(&mut buf, size as usize);
        if nread != size as usize {
            return Err(self.eof_reason(size));
        }
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    // read a vector of values (gguf.cpp lines 275-311)
    fn read_vec<T: Copy>(&mut self, n: u64) -> ParseResult<Vec<T>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(format!(
                "GGUF: array of {n} elements exceeds the maximum {GGUF_MAX_ARRAY_ELEMENTS}"
            ));
        }
        if n > self.nbytes_remain / mem::size_of::<T>() as u64 {
            return Err(self.eof_reason(n * mem::size_of::<T>() as u64));
        }
        let mut dst = Vec::with_capacity(n as usize);
        for _ in 0..n {
            dst.push(self.read_val()?);
        }
        Ok(dst)
    }

    // read a vector of Strings (gguf.cpp lines 280-285 special case)
    fn read_string_vec(&mut self, n: u64) -> ParseResult<Vec<String>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(format!(
                "GGUF: string array of {n} elements exceeds the maximum {GGUF_MAX_ARRAY_ELEMENTS}"
            ));
        }
        if n.saturating_mul(mem::size_of::<u64>() as u64) > self.nbytes_remain {
            return Err(self.eof_reason(n.saturating_mul(mem::size_of::<u64>() as u64)));
        }
        let mut dst = Vec::with_capacity(n as usize);
        for _ in 0..n {
            dst.push(self.read_string()?);
        }
        Ok(dst)
    }

    fn tell(&self) -> u64 {
        self.offset
    }

    fn seek(&mut self, absolute_offset: u64) -> ParseResult<()> {
        let end_offset = self.data.len() as u64;
        if absolute_offset > end_offset {
            return Err(format!(
                "GGUF: seek to offset {absolute_offset} is past the end of the {end_offset}-byte part"
            ));
        }
        self.offset = absolute_offset;
        self.nbytes_remain = end_offset - absolute_offset;
        Ok(())
    }
}

/// The data section's size: the furthest tensor end (`offset + nbytes`), with
/// every tensor's range checked against `limit` (the part's data-section length).
///
/// The layout is **not** required to be contiguous in index order. Each tensor
/// carries its own offset, writers insert padding between tensors (unsloth's UD
/// quants do) and may even order the data differently from the index, and the
/// section ends at the furthest tensor. Requiring `offset == accumulated` rejected
/// legal files ([#575](https://github.com/yusiwen/minfer/issues/575)); what matters
/// is that every range fits.
fn data_section_size(info: &[GgufTensorInfo], limit: usize) -> Result<usize, String> {
    let mut size = 0usize;
    for ti in info {
        let nbytes = ggml_nbytes(ti) as u64;
        let Some(end) = ti.offset.checked_add(nbytes) else {
            return Err(format!(
                "GGUF: tensor '{}' has offset {} and {} bytes, which overflows",
                ti.name, ti.offset, nbytes
            ));
        };
        if end > limit as u64 {
            return Err(format!(
                "GGUF: tensor '{}' spans bytes {}..{} of the data section, which is only {} bytes \
                 (the file is truncated, or the offsets are corrupt)",
                ti.name, ti.offset, end, limit
            ));
        }
        size = size.max(end as usize);
    }
    Ok(size)
}

// === Main init function (gguf.cpp lines 451-893, gguf_init_from_reader) ===

impl GgufContext {
    /// Initialize from a byte slice (mmap'd file contents)
    /// Corresponds to gguf_init_from_buffer (gguf.cpp lines 966-977)
    ///
    /// `Err` is the refusal's reason, a complete `GGUF: ...` sentence: the same text
    /// [`load_gguf_model`] puts in [`GgufLoadError::reason`] ([#767]).
    ///
    /// [#767]: https://github.com/yusiwen/minfer/issues/767
    pub fn init_from_data(data: &[u8]) -> ParseResult<GgufContext> {
        let mut reader = GgufReader::new(data);
        parse_context(&mut reader)
    }
}

/// One part, in file order: header, metadata, tensor index, data-section size.
///
/// Each stage returns its own refusal ([#767]); nothing is recorded on the side for a
/// later stage to overwrite.
fn parse_context(gr: &mut GgufReader<'_>) -> ParseResult<GgufContext> {
    let (version, n_kv, n_tensors) = read_header(gr)?;
    let mut ctx = GgufContext {
        version,
        kv: read_kv_pairs(gr, n_kv)?,
        info: Vec::new(),
        alignment: GGUF_DEFAULT_ALIGNMENT,
        offset: 0,
        size: 0,
    };
    // `general.alignment` is one of the pairs just read (gguf.cpp lines 609-616).
    ctx.alignment = read_alignment(&ctx)?;
    ctx.info = read_tensor_info(gr, n_tensors)?;

    // Align to the data section (gguf.cpp lines 751-756). With no tensors there is no
    // data section, so the offset stays where the index ended.
    if n_tensors > 0 {
        gr.seek(ggml_pad(gr.tell() as usize, ctx.alignment) as u64)?;
    }
    ctx.offset = gr.tell() as usize;

    // Total data-section size: the end of the furthest tensor (gguf.cpp line 759,
    // `size = max(offset + nbytes)`). The layout is **not** required to be contiguous
    // in index order: each tensor carries its own offset, and writers insert padding
    // between tensors (unsloth's UD quants do) or order the data differently from the
    // index. Requiring `offset == accumulated` rejected legal files (#575); what
    // matters is that every tensor's range fits in the part.
    ctx.size = data_section_size(&ctx.info, gr.data.len().saturating_sub(ctx.offset))?;
    Ok(ctx)
}

/// Magic, version and the two counts (gguf.cpp lines 457-541), with the version checks
/// of gguf.cpp lines 481-541 on the way. Returns `(version, n_kv, n_tensors)`.
fn read_header(gr: &mut GgufReader<'_>) -> ParseResult<(u32, i64, i64)> {
    let magic: Vec<u8> = gr.read_vec::<u8>(4).map_err(|e| header_refusal(&e))?;
    if magic != GGUF_MAGIC {
        let shown: String = magic
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '?' })
            .collect();
        return Err(format!(
            "GGUF: invalid magic characters: '{shown}', expected 'GGUF'"
        ));
    }

    let version = gr.read_val::<u32>().map_err(|e| header_refusal(&e))?;
    if version == 0 {
        return Err(format!("GGUF: bad GGUF version: {version}"));
    }
    // A version whose low half is zero was written by a host with the opposite byte
    // order (gguf.cpp lines 490-500).
    if (version & 0x0000FFFF) == 0 {
        return Err(format!(
            "GGUF: failed to load model: this GGUF file version {version} is extremely large, is there a mismatch between the host and model endianness?"
        ));
    }
    if version == 1 {
        return Err(
            "GGUF: GGUFv1 is no longer supported, please use a more up-to-date version".to_string(),
        );
    }
    if version > GGUF_VERSION {
        return Err(format!(
            "GGUF: this GGUF file is version {version} but this software only supports up to version {GGUF_VERSION}"
        ));
    }

    let n_tensors = gr.read_val::<i64>().map_err(|e| header_refusal(&e))?;
    if n_tensors < 0 || n_tensors > (usize::MAX / mem::size_of::<GgufTensorInfo>()) as i64 {
        return Err(format!(
            "GGUF: number of tensors is {n_tensors} but must be in [0, {}]",
            usize::MAX / mem::size_of::<GgufTensorInfo>()
        ));
    }
    let n_kv = gr.read_val::<i64>().map_err(|e| header_refusal(&e))?;
    if n_kv < 0 || n_kv > (usize::MAX / mem::size_of::<GgufKv>()) as i64 {
        return Err(format!(
            "GGUF: number of key value pairs is {n_kv} but must be in [0, {}]",
            usize::MAX / mem::size_of::<GgufKv>()
        ));
    }
    Ok((version, n_kv, n_tensors))
}

/// Keep the phrase a truncated header has always reported, and append what the reader
/// actually hit. The phrase alone was all a caller got before [#767].
///
/// [#767]: https://github.com/yusiwen/minfer/issues/767
fn header_refusal(detail: &str) -> String {
    let detail = detail.strip_prefix("GGUF: ").unwrap_or(detail);
    format!("GGUF: failed to read header: {detail}")
}

/// `general.alignment`, defaulted, shape-checked and validated (gguf.cpp lines 609-616).
fn read_alignment(ctx: &GgufContext) -> ParseResult<usize> {
    let idx = ctx.find_key(GGUF_KEY_GENERAL_ALIGNMENT);
    let alignment = if idx == -1 {
        GGUF_DEFAULT_ALIGNMENT
    } else {
        // Refuse a shape the accessor would assert on: a parse refusal is a value, not
        // a panic (gguf.cpp's `gguf_get_val_u32` asserts the same thing).
        let kv = &ctx.kv[idx as usize];
        if kv.is_array || kv.type_ != GgufType::Uint32 {
            let found = if kv.is_array {
                "an array".to_string()
            } else {
                format!("a {:?}", kv.type_)
            };
            return Err(format!(
                "GGUF: key '{GGUF_KEY_GENERAL_ALIGNMENT}' must be a single uint32, found {found}"
            ));
        }
        ctx.get_val_u32(idx) as usize
    };
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return Err(format!("GGUF: alignment {alignment} is not a power of 2"));
    }
    Ok(alignment)
}

/// The metadata block (gguf.cpp lines 544-617): one pair per iteration, in file order.
fn read_kv_pairs(gr: &mut GgufReader<'_>, n_kv: i64) -> ParseResult<Vec<GgufKv>> {
    // A capacity hint only, and capped: `n_kv` is file-controlled, so a corrupt count
    // must not ask for a huge allocation before the first read fails.
    let mut kv: Vec<GgufKv> = Vec::with_capacity(n_kv.clamp(0, 4096) as usize);
    for i in 0..n_kv {
        let key = gr.read_string()?;
        for (j, seen) in kv.iter().enumerate() {
            if seen.key == key {
                return Err(format!(
                    "GGUF: duplicate key '{key}' for tensors {j} and {i}"
                ));
            }
        }

        // A pair is `type: u32`, then, for an array, `elem_type: u32` and `n: u64`.
        let mut type_ = gr.read_gguf_type()?;
        let mut is_array = false;
        let mut n: u64 = 1;
        if type_ == GgufType::Array {
            is_array = true;
            type_ = gr.read_gguf_type()?;
            n = gr.read_val::<u64>()?;
        }

        // Read the value(s) for the type (gguf.cpp lines 580-599). A numeric array is
        // stored as little-endian bytes, a string array as strings.
        let pair = match type_ {
            GgufType::Uint8 => {
                if is_array {
                    GgufKv::new_array(key, GgufType::Uint8, gr.read_vec::<u8>(n)?)
                } else {
                    GgufKv::new_u8(key, gr.read_val()?)
                }
            }
            GgufType::Int8 => {
                if is_array {
                    let raw: Vec<u8> = gr.read_vec::<i8>(n)?.iter().map(|&x| x as u8).collect();
                    GgufKv::new_array(key, GgufType::Int8, raw)
                } else {
                    GgufKv::new_i8(key, gr.read_val()?)
                }
            }
            GgufType::Uint16 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<u16>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Uint16, raw)
                } else {
                    GgufKv::new_u16(key, gr.read_val()?)
                }
            }
            GgufType::Int16 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<i16>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Int16, raw)
                } else {
                    GgufKv::new_i16(key, gr.read_val()?)
                }
            }
            GgufType::Uint32 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<u32>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Uint32, raw)
                } else {
                    GgufKv::new_u32(key, gr.read_val()?)
                }
            }
            GgufType::Int32 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<i32>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Int32, raw)
                } else {
                    GgufKv::new_i32(key, gr.read_val()?)
                }
            }
            GgufType::Float32 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<f32>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Float32, raw)
                } else {
                    GgufKv::new_f32(key, gr.read_val()?)
                }
            }
            GgufType::Bool => {
                if is_array {
                    let mut raw = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        raw.push(u8::from(gr.read_bool()?));
                    }
                    GgufKv::new_array(key, GgufType::Bool, raw)
                } else {
                    GgufKv::new_bool(key, gr.read_bool()?)
                }
            }
            GgufType::String => {
                if is_array {
                    GgufKv::new_string_array(key, gr.read_string_vec(n)?)
                } else {
                    GgufKv::new_string(key, gr.read_string()?)
                }
            }
            GgufType::Uint64 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<u64>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Uint64, raw)
                } else {
                    GgufKv::new_u64(key, gr.read_val()?)
                }
            }
            GgufType::Int64 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<i64>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Int64, raw)
                } else {
                    GgufKv::new_i64(key, gr.read_val()?)
                }
            }
            GgufType::Float64 => {
                if is_array {
                    let raw: Vec<u8> = gr
                        .read_vec::<f64>(n)?
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    GgufKv::new_array(key, GgufType::Float64, raw)
                } else {
                    GgufKv::new_f64(key, gr.read_val()?)
                }
            }
            GgufType::Array => {
                return Err(format!("GGUF: key '{key}' has invalid GGUF type Array"));
            }
        };
        kv.push(pair);
    }
    Ok(kv)
}

/// The tensor index (gguf.cpp lines 619-749).
fn read_tensor_info(gr: &mut GgufReader<'_>, n_tensors: i64) -> ParseResult<Vec<GgufTensorInfo>> {
    let mut info: Vec<GgufTensorInfo> = Vec::with_capacity(n_tensors.clamp(0, 4096) as usize);
    for i in 0..n_tensors {
        let name = gr.read_string()?;
        if name.len() >= GGML_MAX_NAME {
            return Err(format!(
                "GGUF: tensor name {i} is too long: {} >= {GGML_MAX_NAME}",
                name.len()
            ));
        }
        for (j, seen) in info.iter().enumerate() {
            if seen.name == name {
                return Err(format!(
                    "GGUF: duplicate tensor name '{name}' for tensors {j} and {i}"
                ));
            }
        }
        let mut t = GgufTensorInfo {
            name,
            ne: [1i64; GGML_MAX_DIMS],
            nb: [0usize; GGML_MAX_DIMS],
            type_: GgmlType::F32,
            offset: 0,
        };

        // Shape (gguf.cpp lines 656-694).
        let n_dims: u32 = gr.read_val()?;
        if n_dims > GGML_MAX_DIMS as u32 {
            return Err(format!(
                "GGUF: tensor '{}' has invalid number of dimensions: {n_dims} > {GGML_MAX_DIMS}",
                t.name
            ));
        }
        for j in 0..GGML_MAX_DIMS {
            if (j as u32) < n_dims {
                t.ne[j] = gr.read_val::<i64>()?;
                if t.ne[j] < 0 {
                    return Err(format!(
                        "GGUF: tensor '{}' dimension {j} has invalid number of elements: {} < 0",
                        t.name, t.ne[j]
                    ));
                }
            }
        }

        // The element count must be representable: `ggml_nelements` multiplies i64s. A
        // zero dimension makes the count zero, and it also makes the division below
        // impossible, so it skips the check instead of dividing by zero.
        let (n0, n1, n2, n3) = (t.ne[0], t.ne[1], t.ne[2], t.ne[3]);
        if n1 != 0
            && n2 != 0
            && n3 != 0
            && ((i64::MAX / n1 <= n0)
                || (i64::MAX / n2 <= n0 * n1)
                || (i64::MAX / n3 <= n0 * n1 * n2))
        {
            return Err(format!(
                "GGUF: total number of elements in tensor '{}' with shape ({n0}, {n1}, {n2}, {n3}) is >= {}",
                t.name,
                i64::MAX
            ));
        }

        // Type (gguf.cpp lines 697-736). `from_i32` refuses an undefined byte, so no
        // separate range check is needed here.
        t.type_ = gr.read_ggml_type()?;
        let type_size = t.type_.type_size();
        let blck_size = t.type_.blck_size();
        if blck_size == 0 || t.ne[0] % blck_size != 0 {
            return Err(format!(
                "GGUF: tensor '{}' of type {} ({}) has {} elements per row, not a multiple of block size ({blck_size})",
                t.name,
                t.type_ as i32,
                t.type_.type_name(),
                t.ne[0]
            ));
        }
        let nelements = ggml_nelements(&t.ne);
        if (nelements / blck_size) as u64 > (usize::MAX / type_size) as u64 {
            return Err(format!(
                "GGUF: tensor '{}' with shape ({}, {}, {}, {}) has a size in bytes > {}",
                t.name,
                t.ne[0],
                t.ne[1],
                t.ne[2],
                t.ne[3],
                usize::MAX
            ));
        }

        // Byte offsets (gguf.cpp lines 728-732).
        t.nb[0] = type_size;
        t.nb[1] = t.nb[0] * (t.ne[0] / blck_size) as usize;
        for j in 2..GGML_MAX_DIMS {
            t.nb[j] = t.nb[j - 1] * t.ne[j - 1] as usize;
        }

        // Offset within the part's data section (gguf.cpp lines 738-739).
        t.offset = gr.read_val::<u64>()?;
        info.push(t);
    }
    Ok(info)
}

// === GgufContext extension methods (moved from old model.rs) ===

impl GgufContext {
    pub fn get_key_val_str(&self, key: &str) -> Option<String> {
        for kv in &self.kv {
            if kv.key == key {
                return self.get_string(kv);
            }
        }
        None
    }

    pub fn get_key_val_i64(&self, key: &str) -> Option<i64> {
        for kv in &self.kv {
            if kv.key == key {
                return self.get_i64(kv);
            }
        }
        None
    }

    pub fn get_key_val_f32(&self, key: &str) -> Option<f32> {
        for kv in &self.kv {
            if kv.key == key {
                return self.get_f32(kv);
            }
        }
        None
    }

    fn get_string(&self, kv: &GgufKv) -> Option<String> {
        if kv.type_ == GgufType::String && !kv.data_string.is_empty() {
            Some(kv.data_string[0].clone())
        } else {
            None
        }
    }

    fn get_i64(&self, kv: &GgufKv) -> Option<i64> {
        match kv.type_ {
            GgufType::Int64 => Some(kv.get_val_i64(0)),
            GgufType::Uint32 => Some(kv.get_val_u32(0) as i64),
            GgufType::Int32 => Some(kv.get_val_i32(0) as i64),
            GgufType::Uint64 => Some(kv.get_val_u64(0) as i64),
            GgufType::Uint16 => Some(kv.get_val_u16(0) as i64),
            GgufType::Int16 => Some(kv.get_val_i16(0) as i64),
            GgufType::Uint8 => Some(kv.get_val_u8(0) as i64),
            GgufType::Int8 => Some(kv.get_val_i8(0) as i64),
            GgufType::Bool => Some(if kv.get_val_bool(0) { 1 } else { 0 }),
            _ => None,
        }
    }

    fn get_f32(&self, kv: &GgufKv) -> Option<f32> {
        match kv.type_ {
            GgufType::Float32 => Some(kv.get_val_f32(0)),
            GgufType::Float64 => Some(kv.get_val_f64(0) as f32),
            _ => None,
        }
    }
}

// === Multi-part (split) GGUF support — aligned with llama.cpp ===
// A split model is `name-0000X-of-0000Y.gguf` (Y parts). Part 0 holds the model
// metadata (architecture, hparams) and the tensors whose data it contains;
// parts 1+ are standalone GGUFs that each list the tensors they hold (offsets
// relative to that part's own data section). This mirrors llama.cpp's
// `llama_model_loader` (each part is parsed separately; a tensor is read from
// its own part's data).

/// Read-only mmap of a GGUF file part (zero-dependency: raw mmap/munmap via
/// the system libc, which Rust links by default). The file pages are shared
/// with the CPU and (via newBufferWithBytesNoCopy) the GPU instead of being
/// copied — llama's `llama_mmap` / `newBufferWithBytesNoCopy` equivalent
/// (ggml-metal-device.m:1668). MAP_PRIVATE (no writes happen), PROT_READ.
pub struct MmapFile {
    ptr: *mut u8,
    len: usize,
    _file: std::fs::File, // keeps the fd alive for the mapping's lifetime
    /// [#745]: flipped by `Drop`. It exists so a gate can assert that dropping the
    /// last handle really unmapped the file — "the mapping is gone" is otherwise
    /// only observable through `/proc/self/maps`, which no macOS test can read.
    alive: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

// [#745]: a read-only file mapping is safe to share between threads — the pages are
// immutable, and `munmap` happens only when the last `Arc<MmapFile>` goes. Without
// this, `Tensor` (which carries a handle) would stop being `Send`, and `ModelDef`
// requires `Send + Sync`.
unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

// Generic POSIX mmap (Linux + macOS; the syscall ABI is identical on both).
#[cfg(unix)]
const PROT_READ: i32 = 0x1;
#[cfg(unix)]
const MAP_PRIVATE: i32 = 0x0002;

#[cfg(unix)]
extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
}

impl MmapFile {
    /// Map `path` read-only. The error is the reason a load failed: an unreadable or
    /// empty file, or a failing `mmap`, as a value the caller renders ([#767]).
    ///
    /// [#767]: https://github.com/yusiwen/minfer/issues/767
    pub fn map(path: &std::path::Path) -> ParseResult<Self> {
        #[cfg(not(unix))]
        {
            let _ = path;
            return Err(
                "GGUF: memory-mapping a file is not supported on this platform".to_string(),
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let file = std::fs::File::open(path)
                .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
            let len = file
                .metadata()
                .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
                .len() as usize;
            if len == 0 {
                return Err(format!("cannot map {}: the file is empty", path.display()));
            }
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            // MAP_FAILED = (void*)-1
            if ptr as isize == -1 {
                return Err(format!(
                    "cannot map {}: mmap failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }
            // (The GPU-side warm-up read happens in metal.rs register_part —
            // the first GPU access to file-backed pages costs ~44 ms of one-time
            // page/TLB setup per process, METAL_OPTIMIZATIONS #39. A CPU-side
            // madvise/touch does NOT fix it — the cost is the GPU's own access.)
            Ok(MmapFile {
                ptr: ptr as *mut u8,
                len,
                _file: file,
                alive: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            })
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// [#745]: a handle that turns false when this mapping is unmapped — the
    /// observable a gate can assert on any OS.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn alive(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.alive.clone()
    }

    /// [#745] the mapping's own file descriptor. A test uses it to prove the fd is
    /// closed by the same drop that unmaps.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn raw_fd(&self) -> i32 {
        use std::os::unix::io::AsRawFd;
        self._file.as_raw_fd()
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            munmap(self.ptr as *mut std::ffi::c_void, self.len);
        }
        self.alive.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The bytes a GGUF part is backed by (#797).
///
/// A path load **maps** the file; a host that already holds the model in memory
/// (`Engine::load_from_bytes`) hands over an **owned** buffer. Both are shared
/// through an `Arc`, so a weight tensor's range keeps its backing alive after the
/// `GgufModel` is dropped — the same contract #745 gave the mapping handle.
#[derive(Clone)]
pub enum PartBytes {
    /// A read-only file mapping, shared with every tensor range in this part.
    Mapped(std::sync::Arc<MmapFile>),
    /// A buffer the host owns and handed over; there is no file and no `munmap`.
    Owned(std::sync::Arc<Vec<u8>>),
}

impl PartBytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            PartBytes::Mapped(map) => map.as_slice(),
            PartBytes::Owned(bytes) => bytes.as_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// The mapping handle, when this part is file-backed. Metal's zero-copy
    /// registration needs it; an owned part is copied once instead
    /// (`newBufferWithBytesNoCopy` requires page-aligned pages whose lifetime the
    /// engine can guarantee, and a `Vec`'s allocation is neither).
    pub fn mapping(&self) -> Option<&std::sync::Arc<MmapFile>> {
        match self {
            PartBytes::Mapped(map) => Some(map),
            PartBytes::Owned(_) => None,
        }
    }

    /// The "still live" flag the #745 gates hold. A mapping flips it on `munmap`;
    /// an owned buffer has no unmap (it is freed by the last `Arc`), so it hands
    /// back a flag that stays `true` — there is nothing for a test to observe.
    pub fn alive(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        match self {
            PartBytes::Mapped(map) => map.alive(),
            PartBytes::Owned(_) => std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }
}

/// One GGUF part: its parsed context + the bytes that hold it.
///
/// [#745] the part **owns** its bytes. Unlike the leaked version, there is no
/// `'static` view of them: [`GgufPart::data`] borrows from `bytes`, so a slice
/// cannot outlive the part. A weight tensor gets a range plus a clone of the
/// handle ([`crate::tensor::TensorData`]) and therefore keeps its own bytes alive
/// after the `GgufModel` is dropped.
pub struct GgufPart {
    pub ctx: GgufContext,
    /// The backing that owns this part's bytes (a mapping or an owned buffer).
    pub bytes: PartBytes,
}

impl GgufPart {
    /// This part's bytes. The borrow is tied to the part (and therefore to the
    /// backing's handle); it is never a `'static` view.
    pub fn data(&self) -> &[u8] {
        self.bytes.as_slice()
    }
}

/// A model loaded from one GGUF file, or from the N parts of a split.
pub struct GgufModel {
    pub parts: Vec<GgufPart>,
    /// [#745] a process-local identity for the file(s) this model was read from.
    ///
    /// The CUDA weight registry is name-keyed; a same-name+same-size registration
    /// is only safe to reuse for the *same* model. This fingerprint (path + inode +
    /// length + mtime, over every split part) is what lets that registry tell a
    /// reload of the same file from a different model with identical shapes. It is
    /// deliberately not a content hash: hashing 4.4 GB at load would cost more than
    /// the reuse saves, and the four metadata fields are enough for a file that is
    /// replaced or rewritten between loads.
    ///
    /// Read by the CUDA registry's model identity (#745); a CPU build has no
    /// CUDA registry, so the field is named there but has no reader.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fingerprint: u64,
}

/// [#745] S3: a load failure as a value, with the file and the parser's reason.
///
/// The response replaces `Option` + a stderr print: a caller (the CLI, the
/// tooling, an embedder) can render it, match on it, or hand it to its own error
/// type, and the reason is no longer only on a stream.
#[derive(Debug, Clone)]
pub struct GgufLoadError {
    pub path: std::path::PathBuf,
    pub reason: String,
}

impl fmt::Display for GgufLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to load '{}': {}",
            self.path.display(),
            self.reason
        )
    }
}

impl std::error::Error for GgufLoadError {}

// The trait requires it; the mapping handle is immutable and `Send + Sync`, so
// this is a compile-time promise rather than a test-only claim.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GgufModel>();
};

fn model_fingerprint(paths: &[&std::path::Path]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in paths {
        p.hash(&mut h);
        if let Ok(md) = std::fs::metadata(p) {
            md.len().hash(&mut h);
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                md.dev().hash(&mut h);
                md.ino().hash(&mut h);
            }
            if let Ok(modified) = md.modified() {
                modified.hash(&mut h);
            }
        }
    }
    h.finish()
}

/// Parse a split filename `name-0000X-of-0000Y.gguf` → (prefix, zero-based idx, count).
/// Returns None when the name isn't a split file.
pub fn split_file_info(name: &str) -> Option<(String, usize, usize)> {
    let stem = name.strip_suffix(".gguf")?;
    let dash = stem.rfind("-of-")?;
    let idx_part = &stem[..dash];
    let count_str = &stem[dash + 4..];
    let idx_dash = idx_part.rfind('-')?;
    let prefix = &idx_part[..idx_dash];
    let idx_num = &idx_part[idx_dash + 1..];
    if idx_num.len() != 5 || count_str.len() != 5 {
        return None;
    }
    let idx: usize = idx_num.parse().ok()?;
    let count: usize = count_str.parse().ok()?;
    if idx == 0 || count == 0 || idx > count {
        return None;
    }
    Some((prefix.to_string(), idx - 1, count))
}

/// Given a part path, resolve the full ordered list of all N part paths.
/// Returns None when the path isn't a valid split file or isn't the first part.
pub fn resolve_splits(path: &std::path::Path) -> Option<Vec<std::path::PathBuf>> {
    let name = path.file_name()?.to_str()?;
    let (prefix, idx, count) = split_file_info(name)?;
    if idx != 0 {
        return None;
    }
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let fname = format!("{}-{:05}-of-{:05}.gguf", prefix, i + 1, count);
        out.push(dir.join(fname));
    }
    Some(out)
}

/// Load a GGUF model from `path` (a single file or the FIRST part of a split),
/// assembling all parts. Aligned with llama.cpp: part 0 must be loaded first.
///
/// Each part file is mmap'd (zero-copy). [#745] the mapping is **owned**, not
/// leaked: it stays alive exactly as long as this model — or any `Tensor` read out
/// of it, or a Metal zero-copy part registration — holds a handle, and the last
/// one unmaps the file and closes its fd. On a CPU or CUDA host, loading and
/// dropping models in one process (a server, the tooling, the test suite)
/// therefore no longer accumulates address space and descriptors. On macOS the
/// zero-copy `MTLBuffer` keeps the pages mapped while the model's registration is
/// alive; the model owns that registration (ADR-0077), so the buffers are released
/// before the mapping when the model — and the `GraphCache` that uses it — drops.
///
/// No `'static` view exists: [`GgufPart::data`] borrows from the part, and a
/// weight tensor carries its own [`crate::tensor::TensorData::Mapped`] range plus
/// the handle.
///
/// A failure is a [`GgufLoadError`]: the path plus the parser's own reason. Since
/// [#767] that reason is a value all the way from the reader — a stage cannot record
/// it on the side, and the outermost call cannot lose it.
///
/// [#767]: https://github.com/yusiwen/minfer/issues/767
pub fn load_gguf_model(path: &std::path::Path) -> Result<GgufModel, GgufLoadError> {
    load_gguf_model_inner(path).map_err(|reason| GgufLoadError {
        path: path.to_path_buf(),
        reason,
    })
}

/// Load a model from bytes the host already holds (#797) — a model received over
/// the network, an embedded asset, a volume the host decrypted itself.
///
/// The buffer is **not** copied: the part owns the `Arc<Vec<u8>>` and every weight
/// tensor is a range of it, exactly as a path load's tensors are ranges of the
/// mapping. A split model is refused by name: the path loader resolves its
/// siblings next to the file, and a bytes load has no directory to look in.
///
/// The refusal carries `<memory>` where a path load carries the file, so a host
/// can tell the two apart in the rendered error.
pub fn load_gguf_model_from_bytes(bytes: Vec<u8>) -> Result<GgufModel, GgufLoadError> {
    const VIRTUAL: &str = "<memory>";
    let bytes = std::sync::Arc::new(bytes);
    let ctx = GgufContext::init_from_data(&bytes).map_err(|reason| GgufLoadError {
        path: std::path::PathBuf::from(VIRTUAL),
        reason,
    })?;
    let split_count = ctx
        .get_key_val_i64("split.count")
        .map(|v| v as usize)
        .unwrap_or(1);
    if split_count > 1 {
        return Err(GgufLoadError {
            path: std::path::PathBuf::from(VIRTUAL),
            reason: format!(
                "GGUF: a split model ({split_count} parts) cannot be loaded from memory — the \
                 path loader resolves the parts beside the file, and a bytes load has no \
                 directory"
            ),
        });
    }
    let fingerprint = bytes_fingerprint(&bytes);
    Ok(GgufModel {
        parts: vec![GgufPart {
            ctx,
            bytes: PartBytes::Owned(bytes),
        }],
        fingerprint,
    })
}

/// A process-local identity for a model held in memory (#797), the counterpart of
/// [`model_fingerprint`]'s path+inode+length+mtime.
///
/// Deliberately not a content hash: hashing several GB at load costs more than the
/// CUDA registry's reuse saves, so this hashes the length plus the first and last
/// 4 KiB — enough to tell a reloaded buffer from a different model, and it is a
/// pure function of the bytes (no path, no filesystem).
pub(crate) fn bytes_fingerprint(data: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.len().hash(&mut h);
    let head = &data[..data.len().min(4096)];
    head.hash(&mut h);
    let tail_start = data.len().saturating_sub(4096);
    data[tail_start..].hash(&mut h);
    h.finish()
}

fn load_gguf_model_inner(path: &std::path::Path) -> ParseResult<GgufModel> {
    let mmap0 = std::sync::Arc::new(MmapFile::map(path)?);
    let ctx0 = GgufContext::init_from_data(mmap0.as_slice())?;
    let split_count = ctx0
        .get_key_val_i64("split.count")
        .map(|v| v as usize)
        .unwrap_or(1);

    if split_count <= 1 {
        return Ok(GgufModel {
            parts: vec![GgufPart {
                ctx: ctx0,
                bytes: PartBytes::Mapped(mmap0),
            }],
            fingerprint: model_fingerprint(&[path]),
        });
    }

    let split_no = ctx0
        .get_key_val_i64("split.no")
        .map(|v| v as usize)
        .unwrap_or(0);
    if split_no != 0 {
        return Err(format!(
            "GGUF: split.no = {split_no} — must load the first split (00001-of-{split_count})"
        ));
    }

    let part_paths = resolve_splits(path).ok_or_else(|| {
        format!(
            "GGUF: cannot resolve the parts of '{}': expected the set's first part, <name>-00001-of-NNNNN.gguf",
            path.display()
        )
    })?;
    let fingerprint =
        model_fingerprint(&part_paths.iter().map(|p| p.as_path()).collect::<Vec<_>>());
    if part_paths.len() != split_count {
        return Err(format!(
            "GGUF: split count mismatch: filename implies {} parts, split.count = {split_count}",
            part_paths.len()
        ));
    }

    let mut parts = Vec::with_capacity(split_count);
    for (i, p) in part_paths.iter().enumerate() {
        let mmap = std::sync::Arc::new(MmapFile::map(p)?);
        let ctx = GgufContext::init_from_data(mmap.as_slice())?;
        let no = ctx
            .get_key_val_i64("split.no")
            .map(|v| v as usize)
            .unwrap_or(0);
        if no != i {
            return Err(format!("GGUF: split {p:?} has split.no={no}, expected {i}"));
        }
        parts.push(GgufPart {
            ctx,
            bytes: PartBytes::Mapped(mmap),
        });
    }
    Ok(GgufModel { parts, fingerprint })
}

#[cfg(test)]
mod tests;
/// Number of stored dimensions (llama.cpp's `ggml_n_dims`): trailing 1s dropped,
/// at least one dimension always present. Lives here rather than in the writer
/// so the model loaders (and a crate consumer) can size a tensor without
/// pulling the GGUF writer in (#763 S2).
pub fn ggml_n_dims(ne: &[i64; 4]) -> u32 {
    let mut n = 1u32;
    for j in 1..4 {
        if ne[j] != 1 {
            n = j as u32 + 1;
        }
    }
    n
}
