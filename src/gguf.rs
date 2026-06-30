// GGUF Format Parser
// Translated from: llama.cpp/ggml/src/gguf.cpp + ggml/include/gguf.h

use std::fmt;
use std::mem;
use std::ptr;

// === Constants (from gguf.h) ===

const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;
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

const GGUF_TYPE_COUNT: i32 = 13;

impl GgufType {
    fn from_i32(v: i32) -> Self {
        match v {
            0 => GgufType::Uint8,
            1 => GgufType::Int8,
            2 => GgufType::Uint16,
            3 => GgufType::Int16,
            4 => GgufType::Uint32,
            5 => GgufType::Int32,
            6 => GgufType::Float32,
            7 => GgufType::Bool,
            8 => GgufType::String,
            9 => GgufType::Array,
            10 => GgufType::Uint64,
            11 => GgufType::Int64,
            12 => GgufType::Float64,
            _ => panic!("invalid GGUF type: {}", v),
        }
    }

    fn type_size(&self) -> usize {
        match self {
            GgufType::Uint8 => 1,
            GgufType::Int8 => 1,
            GgufType::Uint16 => 2,
            GgufType::Int16 => 2,
            GgufType::Uint32 => 4,
            GgufType::Int32 => 4,
            GgufType::Float32 => 4,
            GgufType::Bool => 1,
            GgufType::String => 0,   // undefined
            GgufType::Array => 0,     // undefined
            GgufType::Uint64 => 8,
            GgufType::Int64 => 8,
            GgufType::Float64 => 8,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            GgufType::Uint8 => "u8",
            GgufType::Int8 => "i8",
            GgufType::Uint16 => "u16",
            GgufType::Int16 => "i16",
            GgufType::Uint32 => "u32",
            GgufType::Int32 => "i32",
            GgufType::Float32 => "f32",
            GgufType::Bool => "bool",
            GgufType::String => "str",
            GgufType::Array => "arr",
            GgufType::Uint64 => "u64",
            GgufType::Int64 => "i64",
            GgufType::Float64 => "f64",
        }
    }
}

// === GGML types (subset needed for GGUF — from ggml.h lines 389-433) ===

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
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
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            16 => GgmlType::IQ2_XXS,
            17 => GgmlType::IQ2_XS,
            18 => GgmlType::IQ3_XXS,
            19 => GgmlType::IQ1_S,
            20 => GgmlType::IQ4_NL,
            21 => GgmlType::IQ3_S,
            22 => GgmlType::IQ2_S,
            23 => GgmlType::IQ4_XS,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::IQ1_M,
            30 => GgmlType::BF16,
            34 => GgmlType::TQ1_0,
            35 => GgmlType::TQ2_0,
            39 => GgmlType::MXFP4,
            40 => GgmlType::NVFP4,
            41 => GgmlType::Q1_0,
            _ => panic!("invalid GGML type: {}", v),
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
            // they're not reachable via the GgmlType enum
            _ => panic!("ggml_type_size: unknown type {:?}", self),
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
            // they're not reachable via the GgmlType enum
            _ => panic!("ggml_blck_size: unknown type {:?}", self),
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
            _ => "DEPRECATED",
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
    fn new_u8(key: String, value: u8) -> Self {
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

    fn new_i8(key: String, value: i8) -> Self {
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

    fn new_u16(key: String, value: u16) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Uint16,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_i16(key: String, value: i16) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Int16,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_u32(key: String, value: u32) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Uint32,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_i32(key: String, value: i32) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Int32,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_f32(key: String, value: f32) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Float32,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_f64(key: String, value: f64) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Float64,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_bool(key: String, value: bool) -> Self {
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

    fn new_u64(key: String, value: u64) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Uint64,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_i64(key: String, value: i64) -> Self {
        let data = value.to_le_bytes().to_vec();
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::Int64,
            data,
            data_string: Vec::new(),
        }
    }

    fn new_string(key: String, value: String) -> Self {
        GgufKv {
            key,
            is_array: false,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: vec![value],
        }
    }

    fn new_string_array(key: String, value: Vec<String>) -> Self {
        GgufKv {
            key,
            is_array: true,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: value,
        }
    }

    fn new_array(key: String, type_: GgufType, raw_data: Vec<u8>) -> Self {
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

    fn cast(&mut self, new_type: GgufType) {
        let new_type_size = new_type.type_size();
        assert!(self.data.len() % new_type_size == 0);
        self.type_ = new_type;
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
    pub ne: [i64; GGML_MAX_DIMS],  // number of elements per dimension
    pub nb: [usize; GGML_MAX_DIMS], // stride in bytes per dimension
    pub type_: GgmlType,
    pub offset: u64, // offset from start of data section
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

impl GgufContext {
    // === Accessor functions (from gguf.cpp lines 1004-1193) ===

    pub fn get_version(&self) -> u32 {
        self.version
    }

    pub fn get_alignment(&self) -> usize {
        self.alignment
    }

    pub fn get_data_offset(&self) -> usize {
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

    pub fn get_kv_type(&self, key_id: i64) -> GgufType {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        if self.kv[key_id as usize].is_array {
            GgufType::Array
        } else {
            self.kv[key_id as usize].get_type()
        }
    }

    pub fn get_arr_type(&self, key_id: i64) -> GgufType {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].is_array);
        self.kv[key_id as usize].get_type()
    }

    pub fn get_arr_data(&self, key_id: i64) -> &[u8] {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_type() != GgufType::String);
        &self.kv[key_id as usize].data
    }

    pub fn get_arr_str(&self, key_id: i64, i: usize) -> &str {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_type() == GgufType::String);
        &self.kv[key_id as usize].data_string[i]
    }

    pub fn get_arr_n(&self, key_id: i64) -> usize {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        self.kv[key_id as usize].get_ne()
    }

    pub fn get_val_u8(&self, key_id: i64) -> u8 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_u8(0)
    }

    pub fn get_val_i8(&self, key_id: i64) -> i8 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_i8(0)
    }

    pub fn get_val_u16(&self, key_id: i64) -> u16 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_u16(0)
    }

    pub fn get_val_i16(&self, key_id: i64) -> i16 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_i16(0)
    }

    pub fn get_val_u32(&self, key_id: i64) -> u32 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_u32(0)
    }

    pub fn get_val_i32(&self, key_id: i64) -> i32 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_i32(0)
    }

    pub fn get_val_f32(&self, key_id: i64) -> f32 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_f32(0)
    }

    pub fn get_val_u64(&self, key_id: i64) -> u64 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_u64(0)
    }

    pub fn get_val_i64(&self, key_id: i64) -> i64 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_i64(0)
    }

    pub fn get_val_f64(&self, key_id: i64) -> f64 {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_f64(0)
    }

    pub fn get_val_bool(&self, key_id: i64) -> bool {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_bool(0)
    }

    pub fn get_val_str(&self, key_id: i64) -> &str {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        self.kv[key_id as usize].get_val_str(0)
    }

    pub fn get_val_data(&self, key_id: i64) -> &[u8] {
        assert!(key_id >= 0 && key_id < self.get_n_kv());
        assert!(self.kv[key_id as usize].get_ne() == 1);
        assert!(self.kv[key_id as usize].get_type() != GgufType::String);
        &self.kv[key_id as usize].data
    }

    pub fn get_n_tensors(&self) -> i64 {
        self.info.len() as i64
    }

    pub fn find_tensor(&self, name: &str) -> i64 {
        // return -1 if tensor not found
        // gguf.cpp lines 1159-1173
        let n_tensors = self.get_n_tensors();
        for i in 0..n_tensors {
            if name == self.get_tensor_name(i) {
                return i;
            }
        }
        -1
    }

    pub fn get_tensor_offset(&self, tensor_id: i64) -> usize {
        assert!(tensor_id >= 0 && tensor_id < self.get_n_tensors());
        self.info[tensor_id as usize].offset as usize
    }

    pub fn get_tensor_name(&self, tensor_id: i64) -> &str {
        assert!(tensor_id >= 0 && tensor_id < self.get_n_tensors());
        &self.info[tensor_id as usize].name
    }

    pub fn get_tensor_type(&self, tensor_id: i64) -> GgmlType {
        assert!(tensor_id >= 0 && tensor_id < self.get_n_tensors());
        self.info[tensor_id as usize].type_
    }

    pub fn get_tensor_size(&self, tensor_id: i64) -> usize {
        assert!(tensor_id >= 0 && tensor_id < self.get_n_tensors());
        ggml_nbytes(&self.info[tensor_id as usize])
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
    fn read_val<T: Copy>(&mut self) -> Option<T> {
        let size = mem::size_of::<T>();
        if size as u64 > self.nbytes_remain {
            return None;
        }
        let mut dst = vec![0u8; size];
        let nread = self.read_raw(&mut dst, size);
        if nread != size {
            return None;
        }
        // Interpret as little-endian T
        let val = unsafe { ptr::read_unaligned(dst.as_ptr() as *const T) };
        Some(val)
    }

    // read bool as int8_t (gguf.cpp lines 313-320)
    fn read_bool(&mut self) -> Option<bool> {
        let tmp: i8 = self.read_val()?;
        Some(tmp != 0)
    }

    // read ggml_type as int32_t (gguf.cpp lines 322-329)
    fn read_ggml_type(&mut self) -> Option<GgmlType> {
        let tmp: i32 = self.read_val()?;
        Some(GgmlType::from_i32(tmp))
    }

    // read gguf_type as int32_t (gguf.cpp lines 331-338)
    fn read_gguf_type(&mut self) -> Option<GgufType> {
        let tmp: i32 = self.read_val()?;
        Some(GgufType::from_i32(tmp))
    }

    // read string: first uint64 length, then bytes (gguf.cpp lines 340-355)
    fn read_string(&mut self) -> Option<String> {
        let size: u64 = self.read_val()?;
        if size > GGUF_MAX_STRING_LENGTH {
            eprintln!("GGUF: string length {} exceeds maximum {}", size, GGUF_MAX_STRING_LENGTH);
            return None;
        }
        if size > self.nbytes_remain {
            eprintln!("GGUF: string length {} exceeds remaining file size {} bytes", size, self.nbytes_remain);
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        let nread = self.read_raw(&mut buf, size as usize);
        if nread != size as usize {
            return None;
        }
        let s = String::from_utf8_lossy(&buf).to_string();
        Some(s)
    }

    // read a vector of values (gguf.cpp lines 275-311)
    fn read_vec<T: Copy>(&mut self, n: u64) -> Option<Vec<T>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return None;
        }
        if n as u64 > self.nbytes_remain / mem::size_of::<T>() as u64 {
            return None;
        }
        let mut dst = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let val: T = self.read_val()?;
            dst.push(val);
        }
        Some(dst)
    }

    // read a vector of Strings (gguf.cpp lines 280-285 special case)
    fn read_string_vec(&mut self, n: u64) -> Option<Vec<String>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return None;
        }
        if n as u64 * mem::size_of::<u64>() as u64 > self.nbytes_remain {
            return None;
        }
        let mut dst = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let s = self.read_string()?;
            dst.push(s);
        }
        Some(dst)
    }

    fn tell(&self) -> u64 {
        self.offset
    }

    fn seek(&mut self, absolute_offset: u64) -> bool {
        let end_offset = self.data.len() as u64;
        if absolute_offset > end_offset {
            return false;
        }
        self.offset = absolute_offset;
        self.nbytes_remain = end_offset - absolute_offset;
        true
    }
}

// === Main init function (gguf.cpp lines 451-893, gguf_init_from_reader) ===

impl GgufContext {
    /// Initialize from a byte slice (mmap'd file contents)
    /// Corresponds to gguf_init_from_buffer (gguf.cpp lines 966-977)
    pub fn init_from_data(data: &[u8]) -> Option<GgufContext> {
        let mut reader = GgufReader::new(data);
        GgufContext::init_from_reader(&mut reader)
    }

    fn init_from_reader(gr: &mut GgufReader) -> Option<GgufContext> {
        let mut ctx = GgufContext {
            version: GGUF_VERSION,
            kv: Vec::new(),
            info: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            offset: 0,
            size: 0,
        };

        let mut ok = true;

        // === Read magic (gguf.cpp lines 457-478) ===
        {
            let magic: Vec<u8> = gr.read_vec::<u8>(4)?;
            for i in 0..4 {
                if magic[i] != GGUF_MAGIC[i] {
                    let c0 = if magic[0].is_ascii_graphic() { magic[0] as char } else { '?' };
                    let c1 = if magic[1].is_ascii_graphic() { magic[1] as char } else { '?' };
                    let c2 = if magic[2].is_ascii_graphic() { magic[2] as char } else { '?' };
                    let c3 = if magic[3].is_ascii_graphic() { magic[3] as char } else { '?' };
                    eprintln!("GGUF: invalid magic characters: '{}{}{}{}', expected 'GGUF'", c0, c1, c2, c3);
                    return None;
                }
            }
        }

        // === Read header (gguf.cpp lines 481-541) ===
        let mut n_kv: i64 = 0;
        let mut n_tensors: i64 = 0;

        if ok {
            if let Some(version) = gr.read_val::<u32>() {
                ctx.version = version;
                if ctx.version == 0 {
                    eprintln!("GGUF: bad GGUF version: {}", ctx.version);
                    ok = false;
                }
                // endianness check (gguf.cpp lines 490-500)
                if ok && (ctx.version & 0x0000FFFF) == 0x00000000 {
                    eprintln!("GGUF: failed to load model: this GGUF file version {} is extremely large, is there a mismatch between the host and model endianness?", ctx.version);
                    ok = false;
                }
                if ok && ctx.version == 1 {
                    eprintln!("GGUF: GGUFv1 is no longer supported, please use a more up-to-date version");
                    ok = false;
                }
                if ok && ctx.version > GGUF_VERSION {
                    eprintln!("GGUF: this GGUF file is version {} but this software only supports up to version {}", ctx.version, GGUF_VERSION);
                    ok = false;
                }
            } else {
                ok = false;
            }
        }

        if ok {
            if let Some(v) = gr.read_val::<i64>() {
                n_tensors = v;
                if n_tensors < 0 || n_tensors > (usize::MAX / mem::size_of::<GgufTensorInfo>()) as i64 {
                    eprintln!("GGUF: number of tensors is {} but must be in [0, {}]", n_tensors, usize::MAX / mem::size_of::<GgufTensorInfo>());
                    ok = false;
                }
            } else {
                ok = false;
            }
        }

        if ok {
            if let Some(v) = gr.read_val::<i64>() {
                n_kv = v;
                if n_kv < 0 || n_kv > (usize::MAX / mem::size_of::<GgufKv>()) as i64 {
                    eprintln!("GGUF: number of key value pairs is {} but must be in [0, {}]", n_kv, usize::MAX / mem::size_of::<GgufKv>());
                    ok = false;
                }
            } else {
                ok = false;
            }
        }

        if !ok {
            eprintln!("GGUF: failed to read header");
            return None;
        }

        // === Read KV pairs (gguf.cpp lines 544-617) ===
        {
            for i in 0..n_kv {
                let key: String;
                match gr.read_string() {
                    Some(s) => key = s,
                    None => {
                        eprintln!("GGUF: encountered length_error while reading key {}", i);
                        ok = false;
                        break;
                    }
                }

                // check for duplicate keys
                for j in 0..ctx.kv.len() {
                    if key == ctx.kv[j].key {
                        eprintln!("GGUF: duplicate key '{}' for tensors {} and {}", key, j, i);
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }

                let mut type_: GgufType = GgufType::Uint8;
                let mut is_array: bool = false;
                let mut n: u64 = 1;

                match gr.read_gguf_type() {
                    Some(t) => type_ = t,
                    None => { ok = false; break; }
                }

                if type_ == GgufType::Array {
                    is_array = true;
                    match gr.read_gguf_type() {
                        Some(t) => type_ = t,
                        None => { ok = false; break; }
                    }
                    match gr.read_val::<u64>() {
                        Some(v) => n = v,
                        None => { ok = false; break; }
                    }
                }

                if !ok {
                    break;
                }

                // Read value based on type (gguf.cpp lines 580-599)
                match type_ {
                    GgufType::Uint8 => {
                        if is_array {
                            let values: Vec<u8> = match gr.read_vec::<u8>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().map(|&x| x).collect::<Vec<_>>();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Uint8, raw));
                        } else {
                            let value: u8 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_u8(key, value));
                        }
                    }
                    GgufType::Int8 => {
                        if is_array {
                            let values: Vec<i8> = match gr.read_vec::<i8>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().map(|&x| x as u8).collect::<Vec<_>>();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Int8, raw));
                        } else {
                            let value: i8 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_i8(key, value));
                        }
                    }
                    GgufType::Uint16 => {
                        if is_array {
                            let values: Vec<u16> = match gr.read_vec::<u16>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Uint16, raw));
                        } else {
                            let value: u16 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_u16(key, value));
                        }
                    }
                    GgufType::Int16 => {
                        if is_array {
                            let values: Vec<i16> = match gr.read_vec::<i16>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Int16, raw));
                        } else {
                            let value: i16 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_i16(key, value));
                        }
                    }
                    GgufType::Uint32 => {
                        if is_array {
                            let values: Vec<u32> = match gr.read_vec::<u32>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Uint32, raw));
                        } else {
                            let value: u32 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_u32(key, value));
                        }
                    }
                    GgufType::Int32 => {
                        if is_array {
                            let values: Vec<i32> = match gr.read_vec::<i32>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Int32, raw));
                        } else {
                            let value: i32 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_i32(key, value));
                        }
                    }
                    GgufType::Float32 => {
                        if is_array {
                            let values: Vec<f32> = match gr.read_vec::<f32>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Float32, raw));
                        } else {
                            let value: f32 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_f32(key, value));
                        }
                    }
                    GgufType::Bool => {
                        if is_array {
                            let mut raw = Vec::with_capacity(n as usize);
                            for _ in 0..n {
                                match gr.read_bool() {
                                    Some(v) => raw.push(if v { 1 } else { 0 }),
                                    None => { ok = false; break; }
                                }
                            }
                            if !ok { break; }
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Bool, raw));
                        } else {
                            let value: bool = match gr.read_bool() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_bool(key, value));
                        }
                    }
                    GgufType::String => {
                        if is_array {
                            let values: Vec<String> = match gr.read_string_vec(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_string_array(key, values));
                        } else {
                            let value: String = match gr.read_string() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_string(key, value));
                        }
                    }
                    GgufType::Uint64 => {
                        if is_array {
                            let values: Vec<u64> = match gr.read_vec::<u64>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Uint64, raw));
                        } else {
                            let value: u64 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_u64(key, value));
                        }
                    }
                    GgufType::Int64 => {
                        if is_array {
                            let values: Vec<i64> = match gr.read_vec::<i64>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Int64, raw));
                        } else {
                            let value: i64 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_i64(key, value));
                        }
                    }
                    GgufType::Float64 => {
                        if is_array {
                            let values: Vec<f64> = match gr.read_vec::<f64>(n) {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            let raw = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ctx.kv.push(GgufKv::new_array(key, GgufType::Float64, raw));
                        } else {
                            let value: f64 = match gr.read_val() {
                                Some(v) => v, None => { ok = false; break; }
                            };
                            ctx.kv.push(GgufKv::new_f64(key, value));
                        }
                    }
                    GgufType::Array => {
                        eprintln!("GGUF: key '{}' has invalid GGUF type Array", key);
                        ok = false;
                        break;
                    }
                }
            }

            if !ok {
                eprintln!("GGUF: failed to read key-value pairs");
                return None;
            }
            assert!(ctx.kv.len() as i64 == n_kv);

            // Read alignment from KV (gguf.cpp lines 609-616)
            let alignment_idx = ctx.find_key(GGUF_KEY_GENERAL_ALIGNMENT);
            ctx.alignment = if alignment_idx == -1 {
                GGUF_DEFAULT_ALIGNMENT
            } else {
                ctx.get_val_u32(alignment_idx) as usize
            };

            if ctx.alignment == 0 || (ctx.alignment & (ctx.alignment - 1)) != 0 {
                eprintln!("GGUF: alignment {} is not a power of 2", ctx.alignment);
                return None;
            }
        }

        // === Read tensor info (gguf.cpp lines 619-749) ===
        for i in 0..n_tensors {
            let mut info = GgufTensorInfo {
                name: String::new(),
                ne: [1i64; GGML_MAX_DIMS],
                nb: [0usize; GGML_MAX_DIMS],
                type_: GgmlType::F32,
                offset: 0,
            };

            // tensor name
            {
                let name: String;
                match gr.read_string() {
                    Some(s) => name = s,
                    None => {
                        eprintln!("GGUF: encountered length_error while reading tensor name {}", i);
                        ok = false;
                        break;
                    }
                }
                if name.len() >= GGML_MAX_NAME {
                    eprintln!("GGUF: tensor name {} is too long: {} >= {}", i, name.len(), GGML_MAX_NAME);
                    ok = false;
                    break;
                }
                info.name = name;

                // check for duplicate tensor names
                for j in 0..i {
                    if info.name == ctx.info[j as usize].name {
                        eprintln!("GGUF: duplicate tensor name '{}' for tensors {} and {}", info.name, j, i);
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                break;
            }

            // tensor shape (gguf.cpp lines 656-694)
            {
                let n_dims: u32 = match gr.read_val() {
                    Some(v) => v,
                    None => { ok = false; break; }
                };
                if n_dims > GGML_MAX_DIMS as u32 {
                    eprintln!("GGUF: tensor '{}' has invalid number of dimensions: {} > {}", info.name, n_dims, GGML_MAX_DIMS);
                    ok = false;
                    break;
                }
                for j in 0..GGML_MAX_DIMS {
                    info.ne[j] = 1;
                    if j < n_dims as usize {
                        match gr.read_val::<i64>() {
                            Some(v) => info.ne[j] = v,
                            None => { ok = false; break; }
                        }
                        if info.ne[j] < 0 {
                            eprintln!("GGUF: tensor '{}' dimension {} has invalid number of elements: {} < 0", info.name, j, info.ne[j]);
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    break;
                }

                // check total number of elements is representable
                if ok {
                    let n0 = info.ne[0];
                    let n1 = info.ne[1];
                    let n2 = info.ne[2];
                    let n3 = info.ne[3];
                    if (i64::MAX / n1 <= n0) ||
                       (i64::MAX / n2 <= n0 * n1) ||
                       (i64::MAX / n3 <= n0 * n1 * n2)
                    {
                        eprintln!("GGUF: total number of elements in tensor '{}' with shape ({}, {}, {}, {}) is >= {}",
                            info.name, n0, n1, n2, n3, i64::MAX);
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                break;
            }

            // tensor type (gguf.cpp lines 697-736)
            {
                match gr.read_ggml_type() {
                    Some(t) => info.type_ = t,
                    None => { ok = false; break; }
                }
                let type_val = info.type_ as i32;
                if type_val < 0 || type_val >= GGML_TYPE_COUNT {
                    eprintln!("GGUF: tensor '{}' has invalid ggml type {}. should be in [0, {})", info.name, type_val, GGML_TYPE_COUNT);
                    ok = false;
                    break;
                }
                let type_size = info.type_.type_size();
                let blck_size = info.type_.blck_size();

                // check that row size is divisible by block size
                if blck_size == 0 || info.ne[0] % blck_size != 0 {
                    eprintln!("GGUF: tensor '{}' of type {} ({}) has {} elements per row, not a multiple of block size ({})",
                        info.name, type_val, info.type_.type_name(), info.ne[0], blck_size);
                    ok = false;
                    break;
                }

                // check that size in bytes is representable
                let nelements = ggml_nelements(&info.ne);
                if ok && (nelements / blck_size) as u64 > (usize::MAX / type_size) as u64 {
                    eprintln!("GGUF: tensor '{}' with shape ({}, {}, {}, {}) has a size in bytes > {}",
                        info.name, info.ne[0], info.ne[1], info.ne[2], info.ne[3], usize::MAX);
                    ok = false;
                    break;
                }

                // calculate byte offsets (gguf.cpp lines 728-732)
                info.nb[0] = type_size;
                info.nb[1] = info.nb[0] * (info.ne[0] / blck_size) as usize;
                for j in 2..GGML_MAX_DIMS {
                    info.nb[j] = info.nb[j - 1] * info.ne[j - 1] as usize;
                }
            }
            if !ok {
                break;
            }

            // tensor data offset within buffer (gguf.cpp lines 738-739)
            match gr.read_val::<u64>() {
                Some(v) => info.offset = v,
                None => { ok = false; break; }
            }

            ctx.info.push(info);
        }

        if !ok {
            eprintln!("GGUF: failed to read tensor info");
            return None;
        }
        assert!(ctx.info.len() as i64 == n_tensors);

        // align to data section (gguf.cpp lines 751-756)
        if n_tensors > 0 {
            let aligned_offset = ggml_pad(gr.tell() as usize, ctx.alignment);
            if !gr.seek(aligned_offset as u64) {
                eprintln!("GGUF: failed to seek to beginning of data section");
                return None;
            }
        }

        // store data section offset (gguf.cpp line 759)
        ctx.offset = gr.tell() as usize;

        // compute total data section size (gguf.cpp lines 762-782)
        {
            ctx.size = 0;
            for i in 0..ctx.info.len() {
                let ti = &ctx.info[i];
                if ti.offset != ctx.size as u64 {
                    eprintln!("GGUF: tensor '{}' has offset {}, expected {}", ti.name, ti.offset, ctx.size);
                    eprintln!("GGUF: failed to read tensor data");
                    return None;
                }
                let padded_size = ggml_pad(ggml_nbytes(ti), ctx.alignment);
                if usize::MAX - ctx.size < padded_size {
                    eprintln!("GGUF: tensor '{}' size overflow, cannot accumulate size {} + {}", ti.name, ctx.size, padded_size);
                    return None;
                }
                ctx.size += padded_size;
            }
        }

        Some(ctx)
    }

    /// Debug dump of all KV metadata
    pub fn dump_metadata(&self) {
        println!("GGUF Context:");
        println!("  version: {}", self.version);
        println!("  alignment: {}", self.alignment);
        println!("  data_offset: {}", self.offset);
        println!("  data_size: {}", self.size);
        println!("  KV pairs ({}):", self.kv.len());
        for (i, kv) in self.kv.iter().enumerate() {
            let type_str = if kv.is_array {
                format!("array[{}] of {}", kv.get_ne(), kv.type_.type_name())
            } else {
                kv.type_.type_name().to_string()
            };
            let val_str = match kv.type_ {
                GgufType::Uint8 => format!("{}", kv.get_val_u8(0)),
                GgufType::Int8 => format!("{}", kv.get_val_i8(0)),
                GgufType::Uint16 => format!("{}", kv.get_val_u16(0)),
                GgufType::Int16 => format!("{}", kv.get_val_i16(0)),
                GgufType::Uint32 => format!("{}", kv.get_val_u32(0)),
                GgufType::Int32 => format!("{}", kv.get_val_i32(0)),
                GgufType::Float32 => format!("{}", kv.get_val_f32(0)),
                GgufType::Bool => format!("{}", kv.get_val_bool(0)),
                GgufType::String => format!("\"{}\"", kv.get_val_str(0)),
                GgufType::Uint64 => format!("{}", kv.get_val_u64(0)),
                GgufType::Int64 => format!("{}", kv.get_val_i64(0)),
                GgufType::Float64 => format!("{}", kv.get_val_f64(0)),
                GgufType::Array => {
                    if kv.get_type() == GgufType::String {
                        let strs: Vec<&str> = (0..kv.get_ne()).map(|j| kv.get_val_str(j)).collect();
                        format!("{:?}", strs)
                    } else {
                        format!("[{} elements]", kv.get_ne())
                    }
                }
            };
            if kv.is_array && kv.get_type() != GgufType::String {
                println!("  [{:4}] {}: {} ({} elements)", i, kv.key, type_str, kv.get_ne());
            } else {
                println!("  [{:4}] {}: {} = {}", i, kv.key, type_str, val_str);
            }
        }
        println!("  Tensors ({}):", self.info.len());
        for (i, ti) in self.info.iter().enumerate() {
            print!("  [{:4}] {}: type={} shape=(", i, ti.name, ti.type_.type_name());
            let mut first = true;
            for j in 0..GGML_MAX_DIMS {
                if ti.ne[j] != 1 || j == 0 {
                    if !first { print!(","); }
                    print!("{}", ti.ne[j]);
                    first = false;
                }
            }
            println!(") offset={} size={}", ti.offset, ggml_nbytes(ti));
        }
    }
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
