// Tensor Structure + Basic Operations

use crate::gguf::GgmlType;

/// Tensor type: represents which memory representation is used for the tensor data
// Variant names follow GGUF conventions (Q4_K, Q5_K, Q6_K) — allow non-camel-case.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4_K,
    Q5_K,
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
            TensorType::BF16 => 2,
            TensorType::Q4_0 => 18,  // sizeof(block_q4_0)
            TensorType::Q4_1 => 20,  // sizeof(block_q4_1)
            TensorType::Q5_0 => 22,  // sizeof(block_q5_0)
            TensorType::Q5_1 => 24,  // sizeof(block_q5_1)
            TensorType::Q4_K => 144, // sizeof(block_q4_K)
            TensorType::Q5_K => 176, // sizeof(block_q5_K)
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
            TensorType::BF16 => 1,
            TensorType::Q4_0 => 32,
            TensorType::Q4_1 => 32,
            TensorType::Q5_0 => 32,
            TensorType::Q5_1 => 32,
            TensorType::Q4_K => 256,
            TensorType::Q5_K => 256,
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
            GgmlType::BF16 => TensorType::BF16,
            GgmlType::Q4_0 => TensorType::Q4_0,
            GgmlType::Q4_1 => TensorType::Q4_1,
            GgmlType::Q5_0 => TensorType::Q5_0,
            GgmlType::Q5_1 => TensorType::Q5_1,
            GgmlType::Q4_K => TensorType::Q4_K,
            GgmlType::Q5_K => TensorType::Q5_K,
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
            TensorType::BF16 => "bf16",
            TensorType::Q4_0 => "q4_0",
            TensorType::Q4_1 => "q4_1",
            TensorType::Q5_0 => "q5_0",
            TensorType::Q5_1 => "q5_1",
            TensorType::Q4_K => "q4_K",
            TensorType::Q5_K => "q5_K",
            TensorType::Q6_K => "q6_K",
            TensorType::Q8_0 => "q8_0",
            TensorType::I8 => "i8",
            TensorType::I16 => "i16",
            TensorType::I32 => "i32",
            TensorType::Raw => "raw",
        }
    }
}

// === Tensor data: owned or a mapping range (#745) ===

/// The bytes of a tensor — and, when they are a range of a GGUF part, the
/// mapping handle that keeps that range alive.
///
/// [#745] A weight tensor is not a `'static` slice of a leaked mapping. The
/// mapping is owned by the model, and a tensor is free to outlive the
/// `GgufModel` it was read from, so the bytes carry their owner: `Mapped` holds
/// the `Arc<MmapFile>` plus an `(off, len)` range, and the last clone of either
/// unmaps the file.
///
/// `Deref<Target = [u8]>` is deliberate: every read site keeps the same syntax
/// (`len`, indexing, `as_ptr`, `to_vec`, iteration), so the typed view costs
/// the constructors, not the consumers.
#[derive(Clone)]
pub struct TensorData {
    inner: TensorDataInner,
}

#[derive(Clone)]
enum TensorDataInner {
    /// A scratch/owned buffer. Test-only: production tensors are ranges of a
    /// GGUF part; `Tensor::new`/`Tensor::from_data` build the scratch case.
    #[cfg(test)]
    Owned(Vec<u8>),
    /// A range of a read-only file mapping, with the handle that keeps it alive.
    Mapped {
        map: std::sync::Arc<crate::gguf::MmapFile>,
        off: usize,
        len: usize,
    },
}

impl TensorData {
    #[cfg(test)]
    pub(crate) fn owned(bytes: Vec<u8>) -> Self {
        Self {
            inner: TensorDataInner::Owned(bytes),
        }
    }

    /// A range of `map`. The range is checked here, once, so `deref` needs no
    /// `unsafe` and a malformed range fails at construction instead of inside a
    /// kernel read.
    pub(crate) fn mapped(
        map: std::sync::Arc<crate::gguf::MmapFile>,
        off: usize,
        len: usize,
    ) -> Self {
        let end = off
            .checked_add(len)
            .unwrap_or_else(|| panic!("tensor range {off}+{len} overflows usize"));
        assert!(
            end <= map.as_slice().len(),
            "tensor range {off}..{end} is past the mapping's {} bytes",
            map.as_slice().len()
        );
        Self {
            inner: TensorDataInner::Mapped { map, off, len },
        }
    }
}

impl std::ops::Deref for TensorData {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.inner {
            #[cfg(test)]
            TensorDataInner::Owned(bytes) => bytes,
            TensorDataInner::Mapped { map, off, len } => &map.as_slice()[*off..*off + *len],
        }
    }
}

impl AsRef<[u8]> for TensorData {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl std::fmt::Debug for TensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            #[cfg(test)]
            TensorDataInner::Owned(bytes) => write!(f, "TensorData::Owned({} bytes)", bytes.len()),
            TensorDataInner::Mapped { off, len, .. } => {
                write!(f, "TensorData::Mapped({off}..{})", off + len)
            }
        }
    }
}

// === Tensor struct (ggml.h lines 667-699, simplified) ===

/// n-dimensional tensor
/// Fields map to ggml_tensor:
///   type  → ttype
///   ne[]  → shape (number of elements per dimension)
///   nb[]  → strides (stride in bytes per dimension)
///   data  → `TensorData`: an owned scratch buffer, or a range of the mmap'd
///           GGUF part (weight tensors — zero-copy load, [#745])
///   name  → name
#[derive(Clone)]
pub struct Tensor {
    pub ttype: TensorType,
    pub shape: [i64; 4],     // ne[0..3]: number of elements per dimension
    pub strides: [usize; 4], // nb[0..3]: stride in bytes per dimension
    pub data: TensorData,
    pub name: String,
}

// ggml-parity Tensor API: the graph path exercises this subset (borrowed
// weights, nbytes, data, …). The rest of the ggml_tensor interface had no
// caller in any build and was deleted in [#242].
//
// [#242]: https://github.com/yusiwen/minfer/issues/242
impl Tensor {
    /// Create a new tensor with given type and shape.
    /// Allocates data buffer with proper alignment.
    /// Test-only (#238): driven by `graph::builder::tests::builder_creates_topo_sorted_graph` (its `f32_tensor` helper); `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn new(ttype: TensorType, shape: &[i64; 4]) -> Self {
        let mut tensor = Tensor {
            ttype,
            shape: *shape,
            strides: [0; 4],
            data: TensorData::owned(Vec::new()),
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
        tensor.data = TensorData::owned(vec![0u8; nbytes]);

        tensor
    }

    /// Create a tensor from an existing data buffer (takes ownership)
    /// Test-only (#238): driven by `graph::op_matrix::matrix_cases_match_their_reference` (test-only module) and `graph::alloc::tests::a_rebuild_does_not_touch_the_device_pool`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn from_data(ttype: TensorType, shape: &[i64; 4], data: Vec<u8>) -> Self {
        let mut tensor = Tensor {
            ttype,
            shape: *shape,
            strides: [0; 4],
            data: TensorData::owned(data),
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

    /// Create a weight tensor as a range of an mmap'd GGUF part (zero-copy load —
    /// the file pages are shared with the CPU/GPU instead of being copied per
    /// tensor).
    ///
    /// [#745] `map` is the part's mapping and `(off, len)` is the tensor's payload
    /// inside it. The tensor carries the handle, so it keeps its own bytes alive
    /// even after the `GgufModel` is dropped; the range is validated here.
    pub fn from_mapped(
        ttype: TensorType,
        shape: &[i64; 4],
        strides: &[usize; 4],
        map: std::sync::Arc<crate::gguf::MmapFile>,
        off: usize,
        len: usize,
    ) -> Self {
        Tensor {
            ttype,
            shape: *shape,
            strides: *strides,
            data: TensorData::mapped(map, off, len),
            name: String::new(),
        }
    }

    /// Set tensor name
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    // === Shape queries (ggml.c lines 1259-1269) ===

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
        self.data.as_ref()
    }

    /// Get the data as f32 slice (panics if type is not F32)
    pub fn data_f32(&self) -> &[f32] {
        assert!(
            self.ttype == TensorType::F32,
            "data_f32 on '{}' of type {:?}",
            self.name,
            self.ttype
        );
        let n = self.data.len() / 4;
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, n) }
    }

    /// Display
    pub fn summary(&self) -> String {
        let mut s = format!("Tensor<{}> shape=(", self.ttype.name());
        let mut first = true;
        for d in 0..4 {
            if self.shape[d] != 1 || d == 0 {
                if !first {
                    s.push_str(",");
                }
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
