// HuggingFace checkpoint → GGUF conversion (F6, issue #49).
//
// Scope, stated up front and enforced by loud refusals:
//   * architecture: `Qwen2ForCausalLM` (`model_type: "qwen2"`) only — the
//     architecture this engine actually runs. Everything else is refused by
//     name, never half-converted.
//   * tensor set: the 13 Qwen2 names + `model.norm` / `model.embed_tokens` /
//     `lm_head`; an unknown tensor name is a refusal, so a future HF revision
//     cannot silently drop weights.
//   * dtypes: bf16, f16, f32 (safetensors). Anything else is refused.
//   * output dtype: f16 (default) or f32. bf16 output is out of scope.
//
// No ML framework: safetensors is a length-prefixed JSON header + raw tensor
// bytes, and tokenizer.json/config.json are plain JSON, so `serde_json` is the
// only dependency.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::gguf::GgmlType;
use crate::gguf_write::{self, TensorSpec};

/// The output element type a `convert` run produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutType {
    F16,
    F32,
}

impl OutType {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Ok(OutType::F16),
            "f32" | "fp32" => Ok(OutType::F32),
            "bf16" => Err(
                "minfer convert: --outtype bf16 is not supported; supported output types: f16, \
                 f32 (a bf16 writer is a follow-up — f32 preserves every bf16 value exactly)"
                    .to_string(),
            ),
            other => Err(format!(
                "minfer convert: unknown --outtype {other:?}; supported: f16, f32"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            OutType::F16 => "f16",
            OutType::F32 => "f32",
        }
    }

    pub fn ggml_type(self) -> GgmlType {
        match self {
            OutType::F16 => GgmlType::F16,
            OutType::F32 => GgmlType::F32,
        }
    }

    /// llama.cpp's `general.file_type` (MOSTLY_F16 / ALL_F32).
    pub fn file_type(self) -> u32 {
        match self {
            OutType::F16 => 1,
            OutType::F32 => 0,
        }
    }
}

/// A source dtype we can decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfDtype {
    Bf16,
    F16,
    F32,
}

impl HfDtype {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(HfDtype::Bf16),
            "F16" => Some(HfDtype::F16),
            "F32" => Some(HfDtype::F32),
            _ => None,
        }
    }

    fn elem_size(self) -> usize {
        match self {
            HfDtype::Bf16 | HfDtype::F16 => 2,
            HfDtype::F32 => 4,
        }
    }
}

/// One tensor discovered in the checkpoint, with where its bytes live.
#[derive(Debug, Clone)]
pub struct HfTensor {
    /// HuggingFace name, e.g. `model.layers.0.self_attn.q_proj.weight`.
    pub hf_name: String,
    /// GGUF name, e.g. `blk.0.attn_q.weight`.
    pub gguf_name: String,
    /// HF shape, `[out, in]` row-major (already reversed for GGUF by the caller).
    pub shape: Vec<i64>,
    pub dtype: HfDtype,
    /// Index into [`HfCheckpoint::files`].
    file: usize,
    /// Byte range inside that file's data section.
    start: usize,
    end: usize,
}

/// A parsed safetensors file: the JSON header plus the data-section base offset.
pub struct SafeFile {
    pub path: PathBuf,
    pub data_offset: u64,
    pub header: Vec<u8>,
    pub data_len: u64,
}

impl SafeFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)
            .map_err(|e| format!("{}: reading header length: {e}", path.display()))?;
        let hlen = u64::from_le_bytes(len8);
        if hlen == 0 || hlen > 512 * 1024 * 1024 {
            return Err(format!(
                "{}: implausible safetensors header length {hlen}",
                path.display()
            ));
        }
        let mut header = vec![0u8; hlen as usize];
        f.read_exact(&mut header)
            .map_err(|e| format!("{}: reading header: {e}", path.display()))?;
        let file_len = f
            .metadata()
            .map_err(|e| format!("{}: metadata: {e}", path.display()))?
            .len();
        let data_offset = 8 + hlen;
        if data_offset > file_len {
            return Err(format!(
                "{}: header ends at {data_offset} but the file is {file_len} bytes",
                path.display()
            ));
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_offset,
            header,
            data_len: file_len - data_offset,
        })
    }

    /// Read `end-start` bytes of the data section.
    pub fn read_range(&self, start: usize, end: usize) -> Result<Vec<u8>, String> {
        if end < start || end as u64 > self.data_len {
            return Err(format!(
                "{}: tensor range {start}..{end} is outside the {}-byte data section",
                self.path.display(),
                self.data_len
            ));
        }
        let mut buf = vec![0u8; end - start];
        let f = File::open(&self.path).map_err(|e| format!("open {}: {e}", self.path.display()))?;
        f.read_exact_at(&mut buf, self.data_offset + start as u64)
            .map_err(|e| format!("{}: reading tensor data: {e}", self.path.display()))?;
        Ok(buf)
    }
}

/// The checkpoint's tensor list plus the files it spans.
pub struct HfCheckpoint {
    pub files: Vec<SafeFile>,
    pub tensors: Vec<HfTensor>,
    /// Tensor order as it will be written (canonical, not header order).
    pub order: Vec<usize>,
}

impl HfCheckpoint {
    /// Read `model.safetensors` or `model.safetensors.index.json` from `dir`.
    pub fn open(dir: &Path, n_layer: i64) -> Result<Self, String> {
        let single = dir.join("model.safetensors");
        let index = dir.join("model.safetensors.index.json");
        let shards: Vec<PathBuf> = if single.is_file() {
            vec![single.clone()]
        } else if index.is_file() {
            let text = std::fs::read_to_string(&index)
                .map_err(|e| format!("read {}: {e}", index.display()))?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| format!("{}: invalid JSON: {e}", index.display()))?;
            let map = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| format!("{}: no weight_map object", index.display()))?;
            let mut files: Vec<PathBuf> = map
                .values()
                .filter_map(|v| v.as_str())
                .map(|s| dir.join(s))
                .collect();
            files.sort();
            files.dedup();
            files
        } else {
            return Err(format!(
                "minfer convert: no 'model.safetensors' or 'model.safetensors.index.json' in {}",
                dir.display()
            ));
        };

        let mut safe_files = Vec::with_capacity(shards.len());
        for s in &shards {
            safe_files.push(SafeFile::open(s)?);
        }

        // Parse every shard's header. BTreeMap iteration sorts tensor names,
        // which is deterministic; the canonical write order is re-imposed below.
        let mut found: BTreeMap<String, HfTensor> = BTreeMap::new();
        for (fi, sf) in safe_files.iter().enumerate() {
            let v: Value = serde_json::from_slice(&sf.header)
                .map_err(|e| format!("{}: invalid safetensors header: {e}", sf.path.display()))?;
            let obj = v.as_object().ok_or_else(|| {
                format!("{}: safetensors header is not an object", sf.path.display())
            })?;
            for (name, meta) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype_s = meta.get("dtype").and_then(|d| d.as_str()).ok_or_else(|| {
                    format!("{}: tensor '{name}' has no dtype", sf.path.display())
                })?;
                let dtype = HfDtype::parse(dtype_s).ok_or_else(|| {
                    format!(
                        "minfer convert: tensor '{name}' has safetensors dtype '{dtype_s}'; \
                         minfer converts bf16, f16 and f32 only"
                    )
                })?;
                let shape: Vec<i64> = meta
                    .get("shape")
                    .and_then(|s| s.as_array())
                    .ok_or_else(|| format!("{}: tensor '{name}' has no shape", sf.path.display()))?
                    .iter()
                    .map(|x| x.as_i64().unwrap_or(-1))
                    .collect();
                let offs: Vec<usize> = meta
                    .get("data_offsets")
                    .and_then(|s| s.as_array())
                    .ok_or_else(|| {
                        format!("{}: tensor '{name}' has no data_offsets", sf.path.display())
                    })?
                    .iter()
                    .map(|x| x.as_u64().unwrap_or(u64::MAX) as usize)
                    .collect();
                if offs.len() != 2 || offs[1] < offs[0] {
                    return Err(format!(
                        "{}: tensor '{name}' has invalid data_offsets {offs:?}",
                        sf.path.display()
                    ));
                }
                let expect: usize =
                    shape.iter().product::<i64>().max(0) as usize * dtype.elem_size();
                if offs[1] - offs[0] != expect {
                    return Err(format!(
                        "{}: tensor '{name}' shape {shape:?} {dtype_s} needs {expect} bytes but \
                         data_offsets span {}",
                        sf.path.display(),
                        offs[1] - offs[0]
                    ));
                }
                let gguf_name =
                    map_tensor_name(name, n_layer).ok_or_else(|| unknown_tensor_error(name))?;
                found.insert(
                    name.clone(),
                    HfTensor {
                        hf_name: name.clone(),
                        gguf_name,
                        shape,
                        dtype,
                        file: fi,
                        start: offs[0],
                        end: offs[1],
                    },
                );
            }
        }

        // Canonical write order: embedding, then each block's attention -> ffn
        // tensors, then the final norm and the (optional) output projection.
        // The reader resolves by name, so this is for reproducibility only.
        let mut tensors: Vec<HfTensor> = found.into_values().collect();
        let order = canonical_order(&tensors);
        tensors.shrink_to_fit();
        Ok(HfCheckpoint {
            files: safe_files,
            tensors,
            order,
        })
    }

    pub fn tensor(&self, i: usize) -> &HfTensor {
        &self.tensors[i]
    }

    /// Read tensor `i`'s bytes from its shard.
    pub fn read(&self, i: usize) -> Result<Vec<u8>, String> {
        let t = &self.tensors[i];
        self.files[t.file].read_range(t.start, t.end)
    }
}

fn unknown_tensor_error(name: &str) -> String {
    format!(
        "minfer convert: HuggingFace tensor '{name}' has no GGUF mapping in minfer's Qwen2 \
         converter; supported tensors are model.embed_tokens, model.norm, lm_head, and \
         model.layers.N.{{input_layernorm,post_attention_layernorm,self_attn.{{q,k,v,o}}_proj,\
         mlp.{{gate,up,down}}_proj}} — minfer refuses to drop a weight it does not recognise"
    )
}

/// HF tensor name → GGUF tensor name, or `None` when unknown.
///
/// `rotary_emb.inv_freq` is deliberately mapped to nothing: llama.cpp's
/// converter drops it too (the engine recomputes RoPE), and it is named here so
/// it is an intentional omission rather than an unknown-tensor refusal.
pub fn map_tensor_name(name: &str, n_layer: i64) -> Option<String> {
    if name.ends_with(".rotary_emb.inv_freq") || name.ends_with("rotary_emb.inv_freq") {
        return None;
    }
    let mapped = match name {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "model.norm.weight" => "output_norm.weight".to_string(),
        "lm_head.weight" => "output.weight".to_string(),
        _ => {
            let rest = name.strip_prefix("model.layers.")?;
            let (idx_s, tail) = rest.split_once('.')?;
            let i: i64 = idx_s.parse().ok()?;
            if i < 0 || i >= n_layer {
                return None;
            }
            let suffix = match tail {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                _ => return None,
            };
            format!("blk.{i}.{suffix}")
        }
    };
    Some(mapped)
}

/// Canonical GGUF write order (indices into `tensors`).
fn canonical_order(tensors: &[HfTensor]) -> Vec<usize> {
    let rank = |n: &str| -> (i64, u8, String) {
        let gguf = n;
        let block = if let Some(rest) = gguf.strip_prefix("blk.") {
            rest.split('.')
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(-1)
        } else {
            -1
        };
        let kind = if block < 0 {
            0u8 // non-block tensors first (token_embd), then trailing ones sorted below
        } else {
            let tail = gguf.splitn(3, '.').nth(2).unwrap_or("");
            match tail {
                "attn_norm.weight" => 0,
                "attn_q.weight" => 1,
                "attn_q.bias" => 2,
                "attn_k.weight" => 3,
                "attn_k.bias" => 4,
                "attn_v.weight" => 5,
                "attn_v.bias" => 6,
                "attn_output.weight" => 7,
                "ffn_norm.weight" => 8,
                "ffn_gate.weight" => 9,
                "ffn_up.weight" => 10,
                "ffn_down.weight" => 11,
                _ => 12,
            }
        };
        (block, kind, gguf.to_string())
    };
    // Non-block tensors: token_embd first, output_norm/output last.
    let non_block_rank = |n: &str| -> u8 {
        match n {
            "token_embd.weight" => 0,
            "output_norm.weight" => 1,
            "output.weight" => 2,
            "output.bias" => 3,
            _ => 4,
        }
    };
    let mut idx: Vec<usize> = (0..tensors.len()).collect();
    idx.sort_by_key(|&i| {
        let n = tensors[i].gguf_name.as_str();
        if n.starts_with("blk.") {
            // blocks after the embedding, before the trailing tensors
            let (b, k, s) = rank(n);
            (1i64, b, k as i64, 0u8, s)
        } else {
            let r = non_block_rank(n);
            // token_embd before blocks, output_norm/output after
            let phase = if r == 0 { 0i64 } else { 2i64 };
            (phase, 0, 0, r, n.to_string())
        }
    });
    idx
}

/// Convert one tensor's bytes from `dtype` to the GGUF type `out`. bf16→f16 is
/// exact in the mantissa but can overflow f16's range; f32→f16 is lossy
/// (nearest-even); f16→f32 and bf16→f32 are exact.
pub fn convert_bytes_ggml(dtype: HfDtype, out: GgmlType, src: &[u8]) -> Vec<u8> {
    let out = match out {
        GgmlType::F32 => OutType::F32,
        _ => OutType::F16,
    };
    convert_bytes(dtype, out, src)
}

/// Convert one tensor's bytes from `dtype` to `out`. See
/// [`convert_bytes_ggml`] for the exactness of each pair.
pub fn convert_bytes(dtype: HfDtype, out: OutType, src: &[u8]) -> Vec<u8> {
    match (dtype, out) {
        (HfDtype::F16, OutType::F16) => src.to_vec(),
        (HfDtype::F32, OutType::F32) => src.to_vec(),
        (HfDtype::Bf16, OutType::F32) | (HfDtype::F16, OutType::F32) => {
            let n = src.len() / dtype.elem_size();
            let mut o = Vec::with_capacity(n * 4);
            for c in src.chunks_exact(dtype.elem_size()) {
                let v = match dtype {
                    HfDtype::Bf16 => {
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
                    }
                    HfDtype::F16 => half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32(),
                    HfDtype::F32 => unreachable!(),
                };
                o.extend_from_slice(&v.to_le_bytes());
            }
            o
        }
        (HfDtype::Bf16, OutType::F16) | (HfDtype::F32, OutType::F16) => {
            let n = src.len() / dtype.elem_size();
            let mut o = Vec::with_capacity(n * 2);
            for c in src.chunks_exact(dtype.elem_size()) {
                let v = match dtype {
                    HfDtype::Bf16 => {
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
                    }
                    HfDtype::F32 => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    HfDtype::F16 => unreachable!(),
                };
                o.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
            o
        }
    }
}

// === config.json ===

pub struct Config {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub n_layer: i64,
    pub hidden_size: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub intermediate: i64,
    pub max_seq_len: i64,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: i64,
    pub eos: Option<u32>,
    pub bos: Option<u32>,
    pub name: Option<String>,
    _tie: bool,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| format!("{}: invalid JSON: {e}", p.display()))?;
        let get_i = |k: &str| v.get(k).and_then(|x| x.as_i64());
        let get_f = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let architectures: Vec<String> = v
            .get("architectures")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        let arch_ok = architectures.iter().any(|a| a == "Qwen2ForCausalLM");
        if !arch_ok || model_type != "qwen2" {
            return Err(format!(
                "minfer convert: unsupported architecture (model_type = {model_type:?}, \
                 architectures = {architectures:?}); minfer's converter supports \
                 'Qwen2ForCausalLM' (model_type \"qwen2\") only"
            ));
        }
        let n_layer = get_i("num_hidden_layers").ok_or("config.json: num_hidden_layers")?;
        let hidden_size = get_i("hidden_size").ok_or("config.json: hidden_size")?;
        let n_head = get_i("num_attention_heads").ok_or("config.json: num_attention_heads")?;
        let n_head_kv = get_i("num_key_value_heads").unwrap_or(n_head);
        let intermediate = get_i("intermediate_size").ok_or("config.json: intermediate_size")?;
        Ok(Self {
            model_type,
            architectures,
            n_layer,
            hidden_size,
            n_head,
            n_head_kv,
            intermediate,
            max_seq_len: get_i("max_position_embeddings").unwrap_or(32768),
            rms_eps: get_f("rms_norm_eps").unwrap_or(1e-6),
            rope_theta: get_f("rope_theta").unwrap_or(10000.0),
            vocab_size: get_i("vocab_size").unwrap_or(0),
            eos: v
                .get("eos_token_id")
                .and_then(|x| x.as_u64())
                .map(|x| x as u32),
            bos: v
                .get("bos_token_id")
                .and_then(|x| x.as_u64())
                .map(|x| x as u32),
            name: v
                .get("_name_or_path")
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from),
            _tie: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }
}

// === tokenizer.json / tokenizer_config.json ===

pub struct TokenizerData {
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
    pub chat_template: String,
    pub add_bos: bool,
    pub pad: Option<u32>,
}

const TOKEN_UNUSED: i32 = 5;

impl TokenizerData {
    pub fn load(dir: &Path, vocab_size: i64) -> Result<Self, String> {
        if vocab_size <= 0 {
            return Err("minfer convert: config.json has no usable vocab_size".to_string());
        }
        let vs = vocab_size as usize;
        let p = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| format!("{}: invalid JSON: {e}", p.display()))?;
        let model = v
            .get("model")
            .ok_or_else(|| format!("{}: no 'model' object", p.display()))?;
        let kind = model.get("type").and_then(|x| x.as_str()).unwrap_or("");
        if kind != "BPE" {
            return Err(format!(
                "minfer convert: tokenizer.json model.type = {kind:?}; minfer's converter \
                 supports byte-level BPE (\"BPE\") only"
            ));
        }
        let base = model
            .get("vocab")
            .and_then(|x| x.as_object())
            .ok_or_else(|| format!("{}: model.vocab is not an object", p.display()))?;

        let mut tokens: Vec<Option<String>> = vec![None; vs];
        for (tok, id) in base {
            let id = id.as_u64().ok_or_else(|| {
                format!("{}: vocab entry '{tok}' has a non-integer id", p.display())
            })? as usize;
            if id >= vs {
                return Err(format!(
                    "minfer convert: vocab entry '{tok}' has id {id} >= vocab_size {vs}"
                ));
            }
            tokens[id] = Some(tok.clone());
        }

        // added_tokens carry their own id and the special flag that decides
        // CONTROL (3) vs USER_DEFINED (4). llamba.cpp additionally treats a
        // token shaped like `<|...|>` as control; that is reproduced here.
        let mut token_types = vec![1i32; vs];
        let added = v
            .get("added_tokens")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        for a in &added {
            let id = a.get("id").and_then(|x| x.as_u64()).unwrap_or(u64::MAX) as usize;
            let content = a.get("content").and_then(|x| x.as_str()).unwrap_or("");
            if id >= vs {
                return Err(format!(
                    "minfer convert: added token '{content}' has id {id} >= vocab_size {vs}"
                ));
            }
            let special = a.get("special").and_then(|x| x.as_bool()).unwrap_or(false);
            tokens[id] = Some(content.to_string());
            token_types[id] = if special || looks_special(content) {
                3
            } else {
                4
            };
        }
        // Whatever is still unset is a reserved slot, exactly as llama.cpp
        // writes it, so a tool that walks the vocabulary sees every id.
        for (id, t) in tokens.iter_mut().enumerate() {
            if t.is_none() {
                *t = Some(format!("[PAD{id}]"));
                token_types[id] = TOKEN_UNUSED;
            }
        }
        let tokens: Vec<String> = tokens.into_iter().map(|t| t.unwrap()).collect();

        let merges_v = model
            .get("merges")
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("{}: model.merges is not an array", p.display()))?;
        let mut merges = Vec::with_capacity(merges_v.len());
        for m in merges_v {
            // Newer tokenizer.json: "a b". Older: ["a", "b"].
            if let Some(s) = m.as_str() {
                merges.push(s.to_string());
            } else if let Some(pair) = m.as_array() {
                let a = pair.first().and_then(|x| x.as_str()).unwrap_or("");
                let b = pair.get(1).and_then(|x| x.as_str()).unwrap_or("");
                merges.push(format!("{a} {b}"));
            } else {
                return Err(format!("{}: malformed merge entry {m}", p.display()));
            }
        }
        if merges.is_empty() {
            return Err(format!(
                "{}: model.merges is empty; the converted GGUF would not load (minfer's strict \
                 tokenizer refuses an empty merge table)",
                p.display()
            ));
        }

        let tc_path = dir.join("tokenizer_config.json");
        let tc: Value = std::fs::read_to_string(&tc_path)
            .map_err(|e| format!("read {}: {e}", tc_path.display()))
            .and_then(|t| {
                serde_json::from_str(&t)
                    .map_err(|e| format!("{}: invalid JSON: {e}", tc_path.display()))
            })?;
        let chat_template = tc
            .get("chat_template")
            .and_then(|x| x.as_str())
            .ok_or_else(|| {
                format!(
                    "minfer convert: {} has no 'chat_template'; minfer's strict loader renders the \
                     model's own template, so a converted GGUF without one is not accepted",
                    tc_path.display()
                )
            })?
            .to_string();
        let add_bos = tc
            .get("add_bos_token")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let pad = tc
            .get("pad_token")
            .and_then(|x| x.as_str())
            .and_then(|s| tokens.iter().position(|t| t == s))
            .map(|i| i as u32);
        Ok(Self {
            tokens,
            token_types,
            merges,
            chat_template,
            add_bos,
            pad,
        })
    }
}

fn looks_special(token: &str) -> bool {
    (token.starts_with("<|") && token.ends_with("|>"))
        || matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
}

/// llama.cpp's `general.name` spelling for a checkpoint directory: separators
/// become spaces and each word is capitalised (`hf-src` → `Hf Src`).
fn title_case(s: &str) -> String {
    s.split(['-', '_'])
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// === Metadata assembly + the write driver ===

/// Build the full metadata list a converted file carries.
///
/// The keys marked "strict" are the ones `Tokenizer::load` and
/// `hparams_from_gguf` read; a file missing any of them is refused at load.
pub fn build_metadata(
    dir: &Path,
    cfg: &Config,
    tok: &TokenizerData,
    out: OutType,
) -> Vec<crate::gguf::GgufKv> {
    use crate::gguf::{GgufKv, GgufType};
    let mut kv = Vec::new();
    kv.push(GgufKv::new_string(
        "general.architecture".into(),
        "qwen2".into(),
    ));
    kv.push(GgufKv::new_string("general.type".into(), "model".into()));
    let name = cfg
        .name
        .clone()
        .or_else(|| dir.file_name().and_then(|n| n.to_str()).map(title_case))
        .unwrap_or_else(|| "model".to_string());
    kv.push(GgufKv::new_string("general.name".into(), name));
    // Architecture hyperparameters (strict: hparams_from_gguf reads these).
    kv.push(GgufKv::new_u32(
        "qwen2.block_count".into(),
        cfg.n_layer as u32,
    ));
    kv.push(GgufKv::new_u32(
        "qwen2.context_length".into(),
        cfg.max_seq_len as u32,
    ));
    kv.push(GgufKv::new_u32(
        "qwen2.embedding_length".into(),
        cfg.hidden_size as u32,
    ));
    kv.push(GgufKv::new_u32(
        "qwen2.feed_forward_length".into(),
        cfg.intermediate as u32,
    ));
    kv.push(GgufKv::new_u32(
        "qwen2.attention.head_count".into(),
        cfg.n_head as u32,
    ));
    kv.push(GgufKv::new_u32(
        "qwen2.attention.head_count_kv".into(),
        cfg.n_head_kv as u32,
    ));
    kv.push(GgufKv::new_f32(
        "qwen2.rope.freq_base".into(),
        cfg.rope_theta,
    ));
    kv.push(GgufKv::new_f32(
        "qwen2.attention.layer_norm_rms_epsilon".into(),
        cfg.rms_eps,
    ));
    kv.push(GgufKv::new_u32("general.file_type".into(), out.file_type()));
    kv.push(GgufKv::new_u32("general.quantization_version".into(), 2));

    // Optional sampling defaults from generation_config.json (cosmetic; the
    // engine's sampler defaults are its own, but a reader like llama.cpp's
    // model card shows these).
    if let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) {
        if let Ok(g) = serde_json::from_str::<Value>(&text) {
            if let Some(v) = g.get("top_k").and_then(|x| x.as_i64()) {
                kv.push(GgufKv::new_i32("general.sampling.top_k".into(), v as i32));
            }
            if let Some(v) = g.get("top_p").and_then(|x| x.as_f64()) {
                kv.push(GgufKv::new_f32("general.sampling.top_p".into(), v as f32));
            }
            if let Some(v) = g.get("temperature").and_then(|x| x.as_f64()) {
                kv.push(GgufKv::new_f32("general.sampling.temp".into(), v as f32));
            }
            if let Some(v) = g.get("repetition_penalty").and_then(|x| x.as_f64()) {
                kv.push(GgufKv::new_f32(
                    "general.sampling.penalty_repeat".into(),
                    v as f32,
                ));
            }
        }
    }

    // Tokenizer (strict: Tokenizer::load reads all of these).
    kv.push(GgufKv::new_string(
        "tokenizer.ggml.model".into(),
        "gpt2".into(),
    ));
    kv.push(GgufKv::new_string(
        "tokenizer.ggml.pre".into(),
        "qwen2".into(),
    ));
    kv.push(GgufKv::new_string_array(
        "tokenizer.ggml.tokens".into(),
        tok.tokens.clone(),
    ));
    kv.push(GgufKv::new_array(
        "tokenizer.ggml.token_type".into(),
        GgufType::Int32,
        tok.token_types
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    ));
    kv.push(GgufKv::new_string_array(
        "tokenizer.ggml.merges".into(),
        tok.merges.clone(),
    ));
    if let Some(p) = tok.pad {
        kv.push(GgufKv::new_u32("tokenizer.ggml.padding_token_id".into(), p));
    }
    if let Some(e) = cfg.eos {
        kv.push(GgufKv::new_u32("tokenizer.ggml.eos_token_id".into(), e));
    }
    if let Some(b) = cfg.bos {
        kv.push(GgufKv::new_u32("tokenizer.ggml.bos_token_id".into(), b));
    }
    kv.push(GgufKv::new_bool(
        "tokenizer.ggml.add_bos_token".into(),
        tok.add_bos,
    ));
    kv.push(GgufKv::new_string(
        "tokenizer.chat_template".into(),
        tok.chat_template.clone(),
    ));
    kv
}

/// A fully-planned conversion: metadata + per-tensor output specs.
pub struct Conversion {
    pub kv: Vec<crate::gguf::GgufKv>,
    pub specs: Vec<TensorSpec>,
    pub ckpt: HfCheckpoint,
    pub out: OutType,
    /// The output GGUF type per tensor. For `f16` output this is F32 for 1-D
    /// tensors (norms and biases) — llama.cpp's "except 1d tensors" rule, and
    /// what the engine's f32 norm/bias path consumes. For `f32` output every
    /// tensor is F32.
    pub targets: Vec<GgmlType>,
}

impl Conversion {
    /// Load `dir`, validate everything, and plan the output. No bytes are written.
    pub fn plan(dir: &Path, out: OutType) -> Result<Self, String> {
        let cfg = Config::load(dir)?;
        let ckpt = HfCheckpoint::open(dir, cfg.n_layer)?;
        // Every converted tensor must have a spec; an empty checkpoint is a refusal.
        if ckpt.tensors.is_empty() {
            return Err(format!(
                "minfer convert: {} contains no tensors",
                dir.join("model.safetensors").display()
            ));
        }
        let tok = TokenizerData::load(dir, cfg.vocab_size)?;
        let kv = build_metadata(dir, &cfg, &tok, out);
        let mut specs = Vec::with_capacity(ckpt.tensors.len());
        let mut targets = Vec::with_capacity(ckpt.tensors.len());
        for t in &ckpt.tensors {
            // GGUF ne is the HF shape reversed (ne[0] = row length = last HF dim).
            let mut ne = [1i64; 4];
            for (j, d) in t.shape.iter().rev().enumerate().take(4) {
                ne[j] = *d;
            }
            let tgt = if out == OutType::F16 && t.shape.len() <= 1 {
                GgmlType::F32
            } else {
                out.ggml_type()
            };
            targets.push(tgt);
            specs.push(TensorSpec::new(t.gguf_name.clone(), ne, tgt));
        }
        Ok(Conversion {
            kv,
            specs,
            ckpt,
            out,
            targets,
        })
    }

    /// Write the conversion as a single file.
    pub fn write_single(&self, path: &Path) -> Result<(), String> {
        let ckpt = &self.ckpt;
        let targets = &self.targets;
        gguf_write::write_single(path, &self.kv, self.specs.clone(), 32, |i, w| {
            let src = ckpt.read(i).map_err(io::Error::other)?;
            let bytes = convert_bytes_ggml(ckpt.tensor(i).dtype, targets[i], &src);
            w.write_all(&bytes)
        })
    }

    /// Write the conversion under `dir` split by data size.
    pub fn write_split(
        &self,
        dir: &Path,
        stem: &str,
        max_part_bytes: u64,
    ) -> Result<Vec<PathBuf>, String> {
        let ckpt = &self.ckpt;
        let targets = &self.targets;
        gguf_write::write_split(
            dir,
            stem,
            &self.kv,
            self.specs.clone(),
            32,
            max_part_bytes,
            |i, w| {
                let src = ckpt.read(i).map_err(io::Error::other)?;
                let bytes = convert_bytes_ggml(ckpt.tensor(i).dtype, targets[i], &src);
                w.write_all(&bytes)
            },
        )
    }
}

// === quantization of an existing GGUF ===

/// Where one source tensor's bytes live, resolved against the loaded model's
/// mmap'd parts (`GgufModel` owns the mappings; the plan only records offsets).
struct SourceTensor {
    part: usize,
    offset: usize,
    src_type: GgmlType,
    src_nbytes: usize,
    /// Row length, for the block-loop split (rows are contiguous and
    /// block-aligned, so a flat decode/encode is equivalent).
    row_elems: usize,
    /// True when the tensor is copied verbatim instead of re-encoded.
    preserved: bool,
    /// Set when the tensor is intentionally encoded at a type other than the
    /// requested target (the tied-embedding policy).
    retarget: Option<crate::quantize::QuantTarget>,
}

/// A planned quantize/convert run.
pub struct QuantizePlan {
    pub kv: Vec<crate::gguf::GgufKv>,
    pub specs: Vec<TensorSpec>,
    sources: Vec<SourceTensor>,
    target: crate::quantize::QuantTarget,
    /// Human-readable report of the tensors left in their source type.
    pub preserved: Vec<String>,
    /// Tensors deliberately encoded at another type (tied embedding → q8_0).
    pub retargeted: Vec<String>,
}

impl QuantizePlan {
    /// Plan re-encoding every tensor of `model` to `target`.
    ///
    /// llama.cpp's "except 1d tensors" rule, one statement per target:
    ///
    /// * **quant** — a 1-D tensor (norm, bias) and a tensor whose row length is
    ///   not a multiple of the target block size are **copied verbatim**, i.e.
    ///   keep their source type;
    /// * **f16** — a 1-D tensor keeps its source type too
    ///   (`tensor_allows_quantization` is false below 2 dims, and both
    ///   `minfer convert --outtype f16` and `llama-quantize … F16` write those
    ///   tensors as **f32**). Converting a 1-D norm to f16 would produce a file
    ///   this engine cannot run: the CPU RMSNorm reads the weight through
    ///   `Tensor::data_f32` (which asserts `F32`) and neither `mat_mul_f16` nor
    ///   the f16 embedding decode has an f16-norm sibling (issue #169);
    /// * **f32** — every tensor, 1-D included, becomes f32.
    ///
    /// The tensors left in their source type are reported through `preserved`,
    /// never silent.
    ///
    /// A source tensor of a type this engine has no dequantizer for (the
    /// K-quants and I-quants) is a loud refusal: re-quantizing it would emit
    /// wrong weights.
    pub fn plan(
        model: &crate::gguf::GgufModel,
        target: crate::quantize::QuantTarget,
    ) -> Result<Self, String> {
        use crate::quantize::QuantTarget as QT;
        let bs = target.blck_size();
        // llama.cpp's tied-embedding policy: when `output.weight` is absent the
        // output projection *is* the token embedding, and for a sub-8-bit
        // (legacy) ftype llama.cpp quantizes that shared tensor at Q8_0 rather
        // than at the requested type. Reproduced here so `minfer quantize
        // --type q4_0` is byte-identical to `llama-quantize … q4_0` on a tied
        // model (Qwen2.5/Qwen3 are tied).
        let tied = !model
            .parts
            .iter()
            .any(|p| p.ctx.info.iter().any(|ti| ti.name == "output.weight"));
        let mut retargeted: Vec<String> = Vec::new();
        let sub_8bit = matches!(target, QT::Q4_0 | QT::Q4_1 | QT::Q5_0 | QT::Q5_1);
        let mut specs = Vec::new();
        let mut sources = Vec::new();
        let mut preserved = Vec::new();
        for (pi, part) in model.parts.iter().enumerate() {
            for ti in &part.ctx.info {
                let decodable = crate::quantize::can_decode(ti.type_);
                if !decodable {
                    return Err(format!(
                        "minfer quantize: tensor '{}' has type {}, which minfer cannot decode (no \
                         dequantizer for it); re-quantizing a K-quant/I-quant source is \
                         unsupported — start from an f16 or f32 GGUF",
                        ti.name,
                        ti.type_.type_name()
                    ));
                }
                let row_ok = ti.ne[0] % bs as i64 == 0;
                // The 1-D rule per target (see the `plan` doc comment): a 1-D
                // tensor is preserved for the quant targets and for f16, and
                // f32 converts everything.
                let keep = match target {
                    QT::F32 => false,
                    QT::F16 => ti.ne[1] <= 1,
                    _ => ti.ne[1] <= 1 || !row_ok,
                };
                if keep {
                    preserved_push(&mut preserved, ti);
                }
                let tied_embed = tied && sub_8bit && ti.name == "token_embd.weight";
                if tied_embed {
                    retargeted.push(format!("{} -> q8_0 (tied embedding)", ti.name));
                }
                let tgt = if keep {
                    ti.type_
                } else if tied_embed {
                    crate::gguf::GgmlType::Q8_0
                } else {
                    target.ggml_type()
                };
                let spec = TensorSpec::new(ti.name.clone(), ti.ne, tgt);
                sources.push(SourceTensor {
                    part: pi,
                    offset: part.ctx.offset + ti.offset as usize,
                    src_type: ti.type_,
                    src_nbytes: ti.nbytes(),
                    row_elems: ti.ne[0] as usize,
                    preserved: keep,
                    retarget: tied_embed.then_some(crate::quantize::QuantTarget::Q8_0),
                });
                specs.push(spec);
            }
        }
        let mut kv = model.parts[0].ctx.kv.clone();
        gguf_write::kv_upsert(
            &mut kv,
            crate::gguf::GgufKv::new_u32("general.file_type".into(), target.file_type()),
        );
        gguf_write::kv_upsert(
            &mut kv,
            crate::gguf::GgufKv::new_u32("general.quantization_version".into(), 2),
        );
        Ok(Self {
            kv,
            specs,
            sources,
            target,
            preserved,
            retargeted,
        })
    }

    fn encode_into(
        &self,
        model: &crate::gguf::GgufModel,
        i: usize,
        w: &mut dyn Write,
    ) -> io::Result<()> {
        let s = &self.sources[i];
        let src = &model.parts[s.part].data[s.offset..s.offset + s.src_nbytes];
        if s.preserved {
            return w.write_all(src);
        }
        let f32s =
            crate::quantize::decode_to_f32(s.src_type, src, s.row_elems).ok_or_else(|| {
                io::Error::other(format!(
                    "tensor {} of type {} cannot be decoded",
                    self.specs[i].name,
                    s.src_type.type_name()
                ))
            })?;
        let bytes = crate::quantize::quantize_row(s.retarget.unwrap_or(self.target), &f32s);
        w.write_all(&bytes)
    }

    pub fn write_single(&self, model: &crate::gguf::GgufModel, path: &Path) -> Result<(), String> {
        gguf_write::write_single(path, &self.kv, self.specs.clone(), 32, |i, w| {
            self.encode_into(model, i, w)
        })
    }

    pub fn write_split(
        &self,
        model: &crate::gguf::GgufModel,
        dir: &Path,
        stem: &str,
        max_part_bytes: u64,
    ) -> Result<Vec<PathBuf>, String> {
        gguf_write::write_split(
            dir,
            stem,
            &self.kv,
            self.specs.clone(),
            32,
            max_part_bytes,
            |i, w| self.encode_into(model, i, w),
        )
    }
}

fn preserved_push(list: &mut Vec<String>, ti: &crate::gguf::GgufTensorInfo) {
    list.push(format!("{} ({})", ti.name, ti.type_.type_name()));
}

fn _is_float_type(t: GgmlType) -> bool {
    matches!(t, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_name_mapping_covers_qwen2_and_refuses_the_rest() {
        assert_eq!(
            map_tensor_name("model.embed_tokens.weight", 24).unwrap(),
            "token_embd.weight"
        );
        assert_eq!(
            map_tensor_name("model.norm.weight", 24).unwrap(),
            "output_norm.weight"
        );
        assert_eq!(
            map_tensor_name("lm_head.weight", 24).unwrap(),
            "output.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.5.self_attn.q_proj.weight", 24).unwrap(),
            "blk.5.attn_q.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.5.mlp.down_proj.weight", 24).unwrap(),
            "blk.5.ffn_down.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.5.post_attention_layernorm.weight", 24).unwrap(),
            "blk.5.ffn_norm.weight"
        );
        // RoPE tables are intentionally dropped, matching llama.cpp.
        assert!(map_tensor_name("model.layers.0.self_attn.rotary_emb.inv_freq", 24).is_none());
        // Unknown names still refuse (the caller turns None into a loud error).
        assert!(map_tensor_name("model.layers.0.mlp.gate_up_proj.weight", 24).is_none());
        assert!(map_tensor_name("model.layers.99.self_attn.q_proj.weight", 24).is_none());
    }

    #[test]
    fn outtype_parse_refuses_bf16_by_name() {
        assert_eq!(OutType::parse("F16").unwrap(), OutType::F16);
        assert_eq!(OutType::parse("f32").unwrap(), OutType::F32);
        let e = OutType::parse("bf16").unwrap_err();
        assert!(e.contains("bf16"), "{e}");
        assert!(OutType::parse("q4_0").is_err());
    }

    #[test]
    fn bf16_to_f16_is_exact_in_the_mantissa_and_saturates_on_overflow() {
        // bf16 1.0 = 0x3F80
        let one = 0x3F80u16.to_le_bytes();
        let y = convert_bytes(HfDtype::Bf16, OutType::F16, &one);
        assert_eq!(
            half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32(),
            1.0
        );
        // bf16 0x7F80 is +inf → f16 +inf
        let inf = 0x7F80u16.to_le_bytes();
        let y = convert_bytes(HfDtype::Bf16, OutType::F16, &inf);
        assert!(half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).is_infinite());
        // bf16 2.0 → f16 2.0 exactly
        let two = 0x4000u16.to_le_bytes();
        let y = convert_bytes(HfDtype::Bf16, OutType::F16, &two);
        assert_eq!(
            half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32(),
            2.0
        );
    }

    #[test]
    fn f32_to_f16_rounds_and_f16_to_f32_is_exact() {
        let v = 1.0009765625f32; // exactly representable in f16
        let y = convert_bytes(HfDtype::F32, OutType::F16, &v.to_le_bytes());
        let back = convert_bytes(HfDtype::F16, OutType::F32, &y);
        assert_eq!(f32::from_le_bytes([back[0], back[1], back[2], back[3]]), v);
        // A value needing rounding: nearest-even to f16.
        let v2 = 1.00048828125f32; // halfway; rounds to even (1.0)
        let y2 = convert_bytes(HfDtype::F32, OutType::F16, &v2.to_le_bytes());
        let back2 = convert_bytes(HfDtype::F16, OutType::F32, &y2);
        assert_eq!(
            f32::from_le_bytes([back2[0], back2[1], back2[2], back2[3]]),
            1.0
        );
    }

    #[test]
    fn looks_special_matches_llama_cpp() {
        assert!(looks_special("<|fim_prefix|>"));
        assert!(looks_special("<|im_start|>"));
        assert!(!looks_special("<tool_call>"));
        assert!(!looks_special("hello"));
        assert!(looks_special("<pad>"));
    }
}
