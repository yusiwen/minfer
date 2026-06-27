// Phase 4: Tensor Structure + Basic Operations
// Translated from: llama.cpp/ggml/include/ggml.h (struct ggml_tensor, lines 667-699)
//   + ggml/src/ggml.c (ggml_nelements, ggml_nbytes, ggml_nrows, lines 1259-1294)
// Strict 1:1 translation — no extra code, no design changes

use crate::block;
use crate::gguf::GgmlType;

/// Tensor type: represents which memory representation is used for the tensor data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q4_K,
    Q6_K,
    Q8_0,
    I8,
    I16,
    I32,
    Raw, // opaque bytes (for GGUF data blob)
}

impl TensorType {
    /// Number of bytes per element (or per block for quantized types)
    pub fn type_size(&self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::F16 => 2,
            TensorType::Q4_0 => 18,  // sizeof(block_q4_0)
            TensorType::Q4_1 => 20,  // sizeof(block_q4_1)
            TensorType::Q4_K => 144, // sizeof(block_q4_K)
            TensorType::Q6_K => 210, // sizeof(block_q6_K)
            TensorType::Q8_0 => 34,  // sizeof(block_q8_0)
            TensorType::I8 => 1,
            TensorType::I16 => 2,
            TensorType::I32 => 4,
            TensorType::Raw => 1,
        }
    }

    /// Number of values per block (1 for non-quantized types)
    pub fn blck_size(&self) -> i64 {
        match self {
            TensorType::F32 => 1,
            TensorType::F16 => 1,
            TensorType::Q4_0 => 32,
            TensorType::Q4_1 => 32,
            TensorType::Q4_K => 256,
            TensorType::Q6_K => 256,
            TensorType::Q8_0 => 32,
            TensorType::I8 => 1,
            TensorType::I16 => 1,
            TensorType::I32 => 1,
            TensorType::Raw => 1,
        }
    }

    /// Convert from GgmlType (from GGUF metadata) to our TensorType
    pub fn from_ggml_type(t: GgmlType) -> Self {
        match t {
            GgmlType::F32 => TensorType::F32,
            GgmlType::F16 => TensorType::F16,
            GgmlType::Q4_0 => TensorType::Q4_0,
            GgmlType::Q4_1 => TensorType::Q4_1,
            GgmlType::Q4_K => TensorType::Q4_K,
            GgmlType::Q6_K => TensorType::Q6_K,
            GgmlType::Q8_0 => TensorType::Q8_0,
            GgmlType::I8 => TensorType::I8,
            GgmlType::I16 => TensorType::I16,
            GgmlType::I32 => TensorType::I32,
            _ => TensorType::Raw,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TensorType::F32 => "f32",
            TensorType::F16 => "f16",
            TensorType::Q4_0 => "q4_0",
            TensorType::Q4_1 => "q4_1",
            TensorType::Q4_K => "q4_K",
            TensorType::Q6_K => "q6_K",
            TensorType::Q8_0 => "q8_0",
            TensorType::I8 => "i8",
            TensorType::I16 => "i16",
            TensorType::I32 => "i32",
            TensorType::Raw => "raw",
        }
    }
}

// === Tensor struct (ggml.h lines 667-699, simplified) ===

/// n-dimensional tensor
/// Fields map to ggml_tensor:
///   type  → ttype
///   ne[]  → shape (number of elements per dimension)
///   nb[]  → strides (stride in bytes per dimension)
///   data  → stored as Vec<u8>
///   name  → name
#[derive(Clone)]
pub struct Tensor {
    pub ttype: TensorType,
    pub shape: [i64; 4],    // ne[0..3]: number of elements per dimension
    pub strides: [usize; 4], // nb[0..3]: stride in bytes per dimension
    pub data: Vec<u8>,
    pub name: String,
}

impl Tensor {
    /// Create a new tensor with given type and shape.
    /// Allocates data buffer with proper alignment.
    pub fn new(ttype: TensorType, shape: &[i64; 4]) -> Self {
        let mut tensor = Tensor {
            ttype,
            shape: *shape,
            strides: [0; 4],
            data: Vec::new(),
            name: String::new(),
        };

        // Compute strides (ggml.h lines 673-676 + gguf.cpp lines 728-732)
        let type_size = ttype.type_size();
        let blck_size = ttype.blck_size();
        tensor.strides[0] = type_size;
        tensor.strides[1] = tensor.strides[0] * (tensor.shape[0] / blck_size) as usize;
        for j in 2..4 {
            tensor.strides[j] = tensor.strides[j - 1] * tensor.shape[j - 1] as usize;
        }

        // Allocate data
        let nbytes = tensor.nbytes();
        tensor.data = vec![0u8; nbytes];

        tensor
    }

    /// Create a tensor from an existing data buffer (zero-copy, takes ownership)
    pub fn from_data(ttype: TensorType, shape: &[i64; 4], data: Vec<u8>) -> Self {
        let mut tensor = Tensor {
            ttype,
            shape: *shape,
            strides: [0; 4],
            data,
            name: String::new(),
        };

        // Compute strides
        let type_size = ttype.type_size();
        let blck_size = ttype.blck_size();
        tensor.strides[0] = type_size;
        tensor.strides[1] = tensor.strides[0] * (tensor.shape[0] / blck_size) as usize;
        for j in 2..4 {
            tensor.strides[j] = tensor.strides[j - 1] * tensor.shape[j - 1] as usize;
        }

        tensor
    }

    /// Create a tensor using the strides computed by the GGUF parser
    /// (when loading from a GGUF file, the strides are already computed)
    pub fn from_data_with_strides(
        ttype: TensorType,
        shape: &[i64; 4],
        strides: &[usize; 4],
        data: Vec<u8>,
    ) -> Self {
        Tensor {
            ttype,
            shape: *shape,
            strides: *strides,
            data,
            name: String::new(),
        }
    }

    /// Set tensor name
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    // === Shape queries (ggml.c lines 1259-1269) ===

    /// Total number of elements (ggml.c line 1259-1263)
    pub fn nelements(&self) -> i64 {
        // ggml_nelements: ne[0]*ne[1]*ne[2]*ne[3]
        self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3]
    }

    /// Number of rows (ggml.c lines 1265-1269)
    pub fn nrows(&self) -> i64 {
        // ggml_nrows: ne[1]*ne[2]*ne[3]
        self.shape[1] * self.shape[2] * self.shape[3]
    }

    /// Number of elements in the first dimension (columns per row)
    pub fn ncols(&self) -> i64 {
        self.shape[0]
    }

    /// Total size in bytes (ggml.c lines 1271-1294)
    pub fn nbytes(&self) -> usize {
        for i in 0..4 {
            if self.shape[i] <= 0 {
                return 0;
            }
        }

        let blck_size = self.ttype.blck_size();
        if blck_size == 1 {
            let mut nbytes = self.ttype.type_size();
            for i in 0..4 {
                nbytes += (self.shape[i] - 1) as usize * self.strides[i];
            }
            nbytes
        } else {
            let mut nbytes = self.shape[0] as usize * self.strides[0] / blck_size as usize;
            for i in 1..4 {
                nbytes += (self.shape[i] - 1) as usize * self.strides[i];
            }
            nbytes
        }
    }

    // === Data access ===

    /// Get the raw data pointer (slice into data Vec)
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get the data as f32 slice (panics if type is not F32)
    pub fn data_f32(&self) -> &[f32] {
        assert!(self.ttype == TensorType::F32);
        let n = self.data.len() / 4;
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, n) }
    }

    pub fn data_f32_mut(&mut self) -> &mut [f32] {
        assert!(self.ttype == TensorType::F32);
        let n = self.data.len() / 4;
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, n) }
    }

    /// Get the quantized data as raw bytes (Q4_0)
    pub fn data_q4_0(&self) -> &[u8] {
        assert!(self.ttype == TensorType::Q4_0, "expected Q4_0 tensor");
        &self.data
    }

    pub fn data_q4_0_mut(&mut self) -> &mut [u8] {
        assert!(self.ttype == TensorType::Q4_0, "expected Q4_0 tensor");
        &mut self.data
    }

    /// Get the quantized data as raw bytes (Q4_1)
    pub fn data_q4_1(&self) -> &[u8] {
        assert!(self.ttype == TensorType::Q4_1, "expected Q4_1 tensor");
        &self.data
    }

    pub fn data_q4_1_mut(&mut self) -> &mut [u8] {
        assert!(self.ttype == TensorType::Q4_1, "expected Q4_1 tensor");
        &mut self.data
    }

    /// Get the quantized data as raw bytes (Q8_0)
    pub fn data_q8_0(&self) -> &[u8] {
        assert!(self.ttype == TensorType::Q8_0, "expected Q8_0 tensor");
        &self.data
    }

    pub fn data_q8_0_mut(&mut self) -> &mut [u8] {
        assert!(self.ttype == TensorType::Q8_0, "expected Q8_0 tensor");
        &mut self.data
    }

    /// Get the quantized data as raw bytes (Q4_K)
    pub fn data_q4_k(&self) -> &[u8] {
        assert!(self.ttype == TensorType::Q4_K, "expected Q4_K tensor");
        &self.data
    }

    pub fn data_q4_k_mut(&mut self) -> &mut [u8] {
        assert!(self.ttype == TensorType::Q4_K, "expected Q4_K tensor");
        &mut self.data
    }

    /// Get the quantized data as raw bytes (Q6_K)
    pub fn data_q6_k(&self) -> &[u8] {
        assert!(self.ttype == TensorType::Q6_K, "expected Q6_K tensor");
        &self.data
    }

    pub fn data_q6_k_mut(&mut self) -> &mut [u8] {
        assert!(self.ttype == TensorType::Q6_K, "expected Q6_K tensor");
        &mut self.data
    }

    /// Access a single f32 value at linear index (F32 only)
    pub fn get_f32(&self, i: usize) -> f32 {
        self.data_f32()[i]
    }

    pub fn set_f32(&mut self, i: usize, val: f32) {
        self.data_f32_mut()[i] = val;
    }

    /// Copy data from another tensor (for views/slices)
    pub fn copy_from(&mut self, src: &Tensor) {
        assert!(self.nbytes() == src.nbytes());
        self.data.copy_from_slice(&src.data);
    }

    // === Reshape ===

    /// Change shape without changing data layout (ne[0]*ne[1]*ne[2]*ne[3] must match)
    /// Corresponds to ggml_reshape in spirit — just changes ne, recalculates nb
    pub fn reshape(&mut self, new_shape: &[i64; 4]) {
        let old_ne = self.nelements();
        let new_ne = new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3];
        assert!(old_ne == new_ne, "reshape: element count mismatch ({} vs {})", old_ne, new_ne);

        self.shape = *new_shape;

        // Recalculate strides
        let type_size = self.ttype.type_size();
        let blck_size = self.ttype.blck_size();
        self.strides[0] = type_size;
        self.strides[1] = self.strides[0] * (self.shape[0] / blck_size) as usize;
        for j in 2..4 {
            self.strides[j] = self.strides[j - 1] * self.shape[j - 1] as usize;
        }
    }

    /// Display
    pub fn summary(&self) -> String {
        let mut s = format!("Tensor<{}> shape=(", self.ttype.name());
        let mut first = true;
        for d in 0..4 {
            if self.shape[d] != 1 || d == 0 {
                if !first { s.push_str(","); }
                s.push_str(&self.shape[d].to_string());
                first = false;
            }
        }
        s.push_str(&format!(") nbytes={}", self.nbytes()));
        if !self.name.is_empty() {
            s.push_str(&format!(" \"{}\"", self.name));
        }
        s
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}
