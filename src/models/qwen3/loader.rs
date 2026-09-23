// Qwen3 GGUF tensor loader (dense architecture).
//
// Mirrors qwen2/loader.rs with two Qwen3 deltas (docs/QWEN3-SUPPORT-PLAN.md §2):
//   1. `n_embd_head` comes from `qwen3.attention.key_length` (128 for the 0.6B)
//      — NOT `n_embd / n_head` (64). Using the naive derivation silently
//      corrupts attention (wrong projection widths, rope dims, scale, KV stride).
//   2. per-layer `attn_q_norm` / `attn_k_norm` weights (f32 [n_embd_head]).
// Note: the QKV concat weight (`blk.{i}.attn_qkv`) is intentionally NOT
// registered here — Qwen3 has no attention biases and the fused decode kernel
// (`attn_bias_rope_store`) cannot express the per-head norm, so decode uses the
// unfused 3-matmul path (fused QKV is a follow-up, Phase E of the plan).

use crate::gguf::GgufContext;
use crate::graph::offload;
use crate::tensor::{Tensor, TensorType};
use crate::vec_ops::RopeStyle;

use super::tensor_names as tn;

/// Qwen3 hyperparameters — read from GGUF metadata.
#[derive(Debug, Clone)]
pub struct HParams {
    pub n_embd: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub n_layer: i64,
    pub n_ff: i64,
    pub n_vocab: i64,
    pub max_seq_len: i64,
    pub f_norm_rms_eps: f32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub eos_token_id: u32,
    pub im_end_token_id: Option<u32>,
    pub rope_style: RopeStyle,
    /// Per-head Q/K dim — DECOUPLED from `n_embd / n_head` in Qwen3; read from
    /// `qwen3.attention.key_length` (128 for the 0.6B, vs n_embd/n_head = 64).
    /// `n_rot = n_embd_head` and the QK scale is `1/sqrt(n_embd_head)`.
    pub n_embd_head: i64,
    /// Actual KV embedding dimension (ne[1] of K weight) = n_head_kv * n_embd_head.
    pub n_kv_embd: i64,
}

impl HParams {
    pub fn n_embd_head(&self) -> i64 {
        self.n_embd_head
    }

    pub fn attention_scale(&self) -> f32 {
        1.0 / (self.n_embd_head as f32).sqrt()
    }
}

/// Per-layer weights for Qwen3 (dense).
#[derive(Clone)]
pub struct LayerWeights {
    pub attn_norm: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub wo: Option<Tensor>,
    /// Per-head Q RMSNorm weight (Qwen3): [n_embd_head], applied to Q before RoPE.
    pub q_norm: Option<Tensor>,
    /// Per-head K RMSNorm weight (Qwen3): [n_embd_head], applied to K before RoPE.
    pub k_norm: Option<Tensor>,
    pub ffn_norm: Option<Tensor>,
    pub ffn_gate: Option<Tensor>,
    pub ffn_up: Option<Tensor>,
    pub ffn_down: Option<Tensor>,
}

impl LayerWeights {
    pub fn new() -> Self {
        Self {
            attn_norm: None,
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            q_norm: None,
            k_norm: None,
            ffn_norm: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down: None,
        }
    }
}

// ============================================================
// HParams extraction from GGUF
// ============================================================

fn get_i64(ctx: &GgufContext, key: &str) -> Option<i64> {
    ctx.get_key_val_i64(key)
}
fn get_f32(ctx: &GgufContext, key: &str) -> Option<f32> {
    ctx.get_key_val_f32(key)
}
fn get_u32(ctx: &GgufContext, key: &str) -> Option<u32> {
    for kv in &ctx.kv {
        if kv.key == key {
            return match kv.type_ {
                crate::gguf::GgufType::Uint32 => Some(kv.get_val_u32(0)),
                crate::gguf::GgufType::Int32 => Some(kv.get_val_i32(0) as u32),
                crate::gguf::GgufType::Uint64 => Some(kv.get_val_u64(0) as u32),
                crate::gguf::GgufType::Int64 => Some(kv.get_val_i64(0) as u32),
                _ => None,
            };
        }
    }
    None
}

pub fn hparams_from_gguf(ctx: &GgufContext) -> Option<HParams> {
    let n_vocab = {
        let mut found = 0i64;
        for kv in &ctx.kv {
            if kv.key == "tokenizer.ggml.tokens" && kv.is_array {
                found = kv.get_ne() as i64;
                break;
            }
        }
        found
    };
    if n_vocab == 0 {
        eprintln!("Warning: could not determine vocabulary size from GGUF");
    }

    let n_embd = get_i64(ctx, "qwen3.embedding_length")
        .or_else(|| get_i64(ctx, "llama.embedding_length"))?;
    let n_head = get_i64(ctx, "qwen3.attention.head_count")
        .or_else(|| get_i64(ctx, "llama.attention.head_count"))?;
    let n_head_kv = get_i64(ctx, "qwen3.attention.head_count_kv")
        .or_else(|| get_i64(ctx, "llama.attention.head_count_kv"))
        .unwrap_or(n_head);
    let n_layer =
        get_i64(ctx, "qwen3.block_count").or_else(|| get_i64(ctx, "llama.block_count"))?;
    let n_ff = get_i64(ctx, "qwen3.feed_forward_length")
        .or_else(|| get_i64(ctx, "llama.feed_forward_length"))?;

    // Qwen3 head dim is decoupled from n_embd/n_head (llama.cpp reads
    // `attention.key_length` / `value_length` the same way).
    let n_embd_head = get_i64(ctx, "qwen3.attention.key_length")
        .or_else(|| get_i64(ctx, "llama.attention.key_length"))
        .unwrap_or(n_embd / n_head);
    let _ = get_i64(ctx, "qwen3.attention.value_length"); // same value for dense Qwen3

    let eos = get_u32(ctx, "tokenizer.ggml.eos_token_id").unwrap_or(0);
    let im_end = find_token_id(ctx, "<|im_end|>").or(Some(eos));

    Some(HParams {
        n_embd,
        n_head,
        n_head_kv,
        n_layer,
        n_ff,
        n_vocab,
        max_seq_len: get_i64(ctx, "qwen3.context_length")
            .or_else(|| get_i64(ctx, "llama.context_length"))
            .unwrap_or(32768),
        f_norm_rms_eps: get_f32(ctx, "qwen3.attention.layer_norm_rms_epsilon")
            .or_else(|| get_f32(ctx, "llama.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6),
        rope_freq_base: get_f32(ctx, "qwen3.rope.freq_base")
            .or_else(|| get_f32(ctx, "llama.rope.freq_base"))
            .unwrap_or(10000.0),
        rope_freq_scale: get_f32(ctx, "qwen3.rope.frequency_scale")
            .or_else(|| get_f32(ctx, "llama.rope.frequency_scale"))
            .unwrap_or(1.0),
        eos_token_id: eos,
        im_end_token_id: im_end,
        rope_style: RopeStyle::NonInterleaved,
        n_embd_head,
        n_kv_embd: n_head_kv * n_embd_head, // default, updated from K weight shape below
    })
}

fn find_token_id(ctx: &GgufContext, target: &str) -> Option<u32> {
    for kv in &ctx.kv {
        if kv.key == "tokenizer.ggml.tokens" && kv.is_array {
            for i in 0..kv.get_ne() {
                if kv.get_val_str(i) == target {
                    return Some(i as u32);
                }
            }
        }
    }
    None
}

// ============================================================
// Tensor loading
// ============================================================

fn load_tensor(
    ctx: &GgufContext,
    raw: &'static [u8],
    ti: &crate::gguf::GgufTensorInfo,
    ns: &str,
    plan: crate::graph::offload::OffloadPlan,
    device_bytes: &std::cell::Cell<usize>,
) -> Tensor {
    let ttype = TensorType::from_ggml_type(ti.type_);
    // Registry names are namespaced for non-primary models (see Qwen3Model::ns).
    let reg_name = if ns.is_empty() {
        ti.name.clone()
    } else {
        format!("{ns}{}", ti.name)
    };
    let mut shape = [1i64; 4];
    for j in 0..4 {
        shape[j] = ti.ne[j];
    }
    let off = ctx.offset + ti.offset as usize;
    // Use GGML type for byte-size calculation — always correct regardless of TensorType mapping
    // (E5 S2: `GgufTensorInfo::nbytes` is the same arithmetic, shared with the offload fit).
    let ts = ti.type_.type_size();
    let bs = ti.type_.blck_size() as usize;
    let nbytes = ti.nbytes();
    // Borrow the tensor bytes straight from the mmap'd part file (zero-copy —
    // the file pages are shared with the CPU and GPU instead of a per-tensor copy).
    let src = &raw[off..off + nbytes];

    let mut strides = [0usize; 4];
    strides[0] = ts;
    strides[1] = strides[0] * (shape[0] / bs as i64) as usize;
    for j in 2..4 {
        strides[j] = strides[j - 1] * shape[j - 1] as usize;
    }

    let mut tensor = Tensor::from_data_borrowed_with_strides(ttype, &shape, &strides, src);
    tensor.set_name(&reg_name);

    // E5: register only when the offload plan puts this tensor's block on the device (see
    // qwen2's loader twin for the rule and the reason).
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    let on_device = plan.allows_weight(&reg_name);
    #[cfg(target_os = "macos")]
    if on_device {
        if let Some(mps) = crate::metal::MpsState::get() {
            if matches!(
                ttype,
                TensorType::Q4_0
                    | TensorType::Q4_1
                    | TensorType::Q4_K
                    | TensorType::Q5_0
                    | TensorType::Q5_1
                    | TensorType::Q5_K
                    | TensorType::Q6_K
                    | TensorType::Q8_0
            ) {
                mps.register_weight(&reg_name, tensor.data());
                device_bytes.set(device_bytes.get() + tensor.data().len());
            } else if ttype == TensorType::F32 {
                mps.register_weight(&reg_name, tensor.data());
                device_bytes.set(device_bytes.get() + tensor.data().len());
            }
        }
    }
    #[cfg(feature = "cuda")]
    if on_device {
        if let Some(cuda) = crate::cuda::CudaState::get() {
            if matches!(
                ttype,
                TensorType::Q4_0
                    | TensorType::Q4_1
                    | TensorType::Q4_K
                    | TensorType::Q5_0
                    | TensorType::Q5_1
                    | TensorType::Q6_K
                    | TensorType::Q8_0
            ) {
                if ttype != TensorType::Q4_K && ttype != TensorType::Q6_K {
                    // r60: non-NB-BT-consumable quantized weight — mode-2
                    // skip-write producers unsound (see qwen2 loader twin).
                    // Global flag, global mix: any registered weight counts,
                    // namespaced or not.
                    cuda.clear_mmq_nb_bt_only();
                }
                if ttype == TensorType::Q6_K {
                    // 7e②: register Q6_K in the padded 224-byte block layout so
                    // the matmul kernel can use aligned uint4 weight loads
                    // (the raw 210-byte stride forces 1-byte-per-instruction
                    // reads and caps 7B decode near ~38 GB/s).
                    // NOTE: under the NAMESPACED draft load the registry key
                    // must be the namespaced reg_name, not the raw GGUF tensor
                    // name - `ti.name` here silently REPLACED the target's
                    // registry entry for the same tensor name (e.g.
                    // 'token_embd.weight'), which failed the target's
                    // has_weight_of_size at the next build_graph and dropped
                    // BOTH graphs to CPU (doc 102).
                    cuda.register_weight_q6k_padded(
                        &reg_name,
                        tensor.data(),
                        tensor.shape[1] as usize,
                        tensor.shape[0] as usize,
                    );
                } else {
                    cuda.register_weight(&reg_name, tensor.data());
                    // doc 104: q8_0 also registers the p32 split planes for the
                    // decode MMVQ (raw registration stays; method self-gates).
                    if ttype == TensorType::Q8_0 {
                        cuda.register_weight_q80_p32(
                            &reg_name,
                            tensor.data(),
                            tensor.shape[1] as usize,
                            tensor.shape[0] as usize,
                        );
                    }
                }
            } else if ttype == TensorType::F32 {
                cuda.register_weight(&reg_name, tensor.data());
                // r60: a 2-D F32 weight is an f32 MATMUL weight (norms/biases
                // are 1-D) — mode-2 skip-write producers unsound upstream.
                if tensor.shape.len() == 2 {
                    cuda.clear_mmq_nb_bt_only();
                }
            }
            device_bytes.set(device_bytes.get() + tensor.data().len());
        }
    }

    // A CPU-only build compiles both registration blocks out; the parameters stay part of
    // the signature (the filter is the contract) and the plan still reaches the model.
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    let _ = (plan, device_bytes);

    tensor
}

/// E5 S2: the weight bytes each transformer block will register, measured from the **GGUF
/// index** (type and shape) before anything is loaded — the `auto` fit has to decide the plan
/// before the first registration, because the registration filter *is* the plan.
///
/// The fused concat copies (`blk.{i}.attn_qkv` / `blk.{i}.ffn_gu`) and the extra device planes
/// (padded Q6_K, the q8_0 p32 split, the q4_K dsc pair) are *not* in this number: they are
/// built while loading. That is what the fit's reserve is for, and an underestimate still ends
/// as a loud E4 refusal at the first forward, never as a silent overcommit.
fn block_weight_bytes(model: &crate::gguf::GgufModel, n_layer: usize) -> Vec<usize> {
    let mut bytes = vec![0usize; n_layer];
    for part in &model.parts {
        for ti in &part.ctx.info {
            if let Some(i) = crate::graph::offload::block_of(&ti.name) {
                if i < n_layer {
                    bytes[i] += ti.nbytes();
                }
            }
        }
    }
    bytes
}

// ============================================================
// Architecture loader
// ============================================================

pub fn load(
    model: &crate::gguf::GgufModel,
    ns: &str,
    offload: crate::graph::offload::OffloadRequest,
) -> Option<super::Qwen3Model> {
    #[cfg(feature = "cuda")]
    // Same rationale as qwen2/loader.rs: block until CUDA init completes so
    // per-tensor registration is all-or-nothing (no partial gate flips).
    crate::cuda::CudaState::init();
    // Serialize weight registration against other threads' loads (same-named
    // tensors across architectures) and against graph tests that hold this
    // lock across several forwards.
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let ctx = &model.parts[0].ctx;
    let mut hparams = hparams_from_gguf(ctx)?;

    // E5: resolve the offload plan before the first weight is registered (see qwen2's twin).
    let n_layer = hparams.n_layer as usize;
    let env_request = std::env::var("MINFER_GPU_LAYERS").ok();
    // The CLI's explicit request wins; `Default` reads the environment (E5 S1).
    let request = match offload {
        offload::OffloadRequest::Default => {
            match offload::OffloadRequest::parse(env_request.as_deref()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("minfer: {e}");
                    return None;
                }
            }
        }
        explicit => explicit,
    };
    let (plan, offload_source) = match request {
        // E5 S2: `auto` (see qwen2's twin for the full rationale).
        offload::OffloadRequest::Auto => {
            let per_block = block_weight_bytes(model, n_layer);
            let free = crate::models::device_free_bytes();
            let cap = std::env::var("MINFER_GPU_MEM").ok();
            let budget = match offload::weight_budget(free, cap.as_deref()) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("minfer: {e}");
                    return None;
                }
            };
            let reserve = budget / 4;
            let k = offload::fit_blocks(budget, &per_block, reserve);
            (
                offload::OffloadPlan {
                    gpu_layers: k,
                    n_layers: n_layer,
                },
                offload::auto_source(k, n_layer, budget, reserve, free, cap.as_deref()),
            )
        }
        other => {
            let source = other.source(env_request.as_deref());
            let plan = match other.plan(
                env_request.as_deref(),
                n_layer,
                crate::models::device_available(),
            ) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("minfer: {e}");
                    return None;
                }
            };
            (plan, source)
        }
    };
    let device_bytes = std::cell::Cell::new(0usize);

    // Merged tensor index across all split parts (llama.cpp weights_map): each
    // tensor lives in the part that lists it, read from that part's own data.
    let mut tensor_map =
        std::collections::HashMap::<String, (usize, &crate::gguf::GgufTensorInfo)>::new();
    for (pi, part) in model.parts.iter().enumerate() {
        for ti in &part.ctx.info {
            tensor_map.insert(ti.name.clone(), (pi, ti));
        }
    }

    // Qwen3 KV dim = n_head_kv * n_embd_head = 8*128 = 1024 (override from the
    // K weight's actual output dim). Resolve BEFORE the KV cache type pick
    // (set_kv_cache_type auto-selects f16 for the 7B class from n_layers * n_kv_embd).
    if let Some((_, ti)) = tensor_map.get(&tn::attn_k(0)) {
        hparams.n_kv_embd = ti.ne[1];
        // sanity: kv dim must equal n_head_kv * n_embd_head (catches a wrong
        // key_length fallback before it silently corrupts attention)
        assert_eq!(
            hparams.n_kv_embd,
            hparams.n_head_kv * hparams.n_embd_head,
            "Qwen3 KV dim {} != n_head_kv {} * n_embd_head {}",
            hparams.n_kv_embd,
            hparams.n_head_kv,
            hparams.n_embd_head,
        );
    }
    #[cfg(target_os = "macos")]
    crate::metal::set_kv_cache_type(hparams.n_layer as usize, hparams.n_kv_embd as usize);
    // 8b: CUDA side shares the same policy and MINFER_CACHE_TYPE override.
    #[cfg(feature = "cuda")]
    crate::cuda::set_kv_cache_type(hparams.n_layer as usize, hparams.n_kv_embd as usize);

    // Zero-copy weight registration: tell the Metal backend about each mmap'd
    // part (page-aligned base) BEFORE any weight is registered, so weights are
    // wrapped as (buffer, offset) into the part buffer instead of being copied.
    #[cfg(target_os = "macos")]
    if let Some(mps) = crate::metal::MpsState::get() {
        for part in &model.parts {
            mps.register_part(part.data);
        }
    }

    let load_one = |n: &str| -> Option<Tensor> {
        tensor_map.get(n).map(|(pi, ti)| {
            let part = &model.parts[*pi];
            load_tensor(&part.ctx, &part.data, ti, ns, plan, &device_bytes)
        })
    };
    let load_ti = |(pi, ti): &(usize, &crate::gguf::GgufTensorInfo)| -> Tensor {
        let part = &model.parts[*pi];
        load_tensor(&part.ctx, &part.data, ti, ns, plan, &device_bytes)
    };

    // Token embedding
    let tok_embd = load_one(tn::TOKEN_EMBD)?;

    // Output norm
    let output_norm = load_one(tn::OUTPUT_NORM);

    // Output weight (with weight tying fallback)
    let output = load_one(tn::OUTPUT).unwrap_or_else(|| tok_embd.clone());

    // Output bias (optional; Qwen3 has none)
    let output_b = load_one(tn::OUTPUT_BIAS);

    // Per-layer weights
    let n_layer = hparams.n_layer as usize;
    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let mut layer = crate::models::qwen3::loader::LayerWeights::new();

        if let Some(ti) = tensor_map.get(&tn::attn_norm(i)) {
            layer.attn_norm = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_q(i)) {
            layer.wq = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k(i)) {
            layer.wk = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_v(i)) {
            layer.wv = Some(load_ti(ti));
        }

        // Fused QKV projection (nt==1 decode, Qwen3): concatenate Wq/Wk/Wv along
        // the output (row) dim into one GPU buffer so a single matmul produces
        // q+k+v, then the per-head Q/K RMSNorm + no-bias rope+store are applied
        // in one fused pass (Op::FusedQkvNorm). Only registered when the three
        // weights share a matmul (same quant type + same input dim); decode
        // falls back to three separate matmuls otherwise. Requires the per-head
        // Q/K norm weights to be present (Qwen3 always has them).
        #[cfg(target_os = "macos")]
        if plan.on_device(i) {
            if let Some(mps) = crate::metal::MpsState::get() {
                if let (Some(wq), Some(wk), Some(wv)) = (&layer.wq, &layer.wk, &layer.wv) {
                    if let Some(data) = crate::metal::concat_rows(&[wq, wk, wv]) {
                        mps.register_weight(&format!("{ns}blk.{i}.attn_qkv"), &data);
                    }
                }
            }
        }
        if let Some(ti) = tensor_map.get(&tn::attn_out(i)) {
            layer.wo = Some(load_ti(ti));
        }
        // Qwen3 per-head Q/K norms (f32 [n_embd_head]) — required for Qwen3;
        // loaded by name (the `minfer info` listing truncates tensors and hides
        // them, but they ARE in the GGUF).
        if let Some(ti) = tensor_map.get(&tn::attn_q_norm(i)) {
            layer.q_norm = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k_norm(i)) {
            layer.k_norm = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_norm(i)) {
            layer.ffn_norm = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_gate(i)) {
            layer.ffn_gate = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_up(i)) {
            layer.ffn_up = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_down(i)) {
            layer.ffn_down = Some(load_ti(ti));
        }

        // Fused FFN gate+up (nt==1 decode): one matmul produces both gate and
        // up from a concatenated weight (Qwen3 reuses the qwen2 fused-FFN path;
        // the FFN has no Qwen3-specific differences).
        // Gated like the build-side fusion decision (nf ≤ 16384): the 7B's
        // nf = 18944 concat would otherwise hold ~2.0 GiB of weights no node
        // ever reads (Phase 8 review). Memory-footprint-only change.
        #[cfg(target_os = "macos")]
        if plan.on_device(i) {
            if let Some(mps) = crate::metal::MpsState::get() {
                if let (Some(fg), Some(fu)) = (&layer.ffn_gate, &layer.ffn_up) {
                    let fuse = fg.shape[1] <= 16384
                        && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1");
                    if fuse {
                        if let Some(data) = crate::metal::concat_rows(&[fg, fu]) {
                            mps.register_weight(&format!("{ns}blk.{i}.ffn_gu"), &data);
                        }
                    }
                }
            }
        }
        // 7e⑤: Fused FFN gate+up (nt==1 decode) — CUDA registration. The
        // concat rows go through register_weight (block-quant types) or the
        // padded repack (Q6_K: 210→224-byte slots for aligned uint4 loads),
        // matching what matmul_f32_ptr_layout dispatches on.
        // Gated like the build-side fusion decision (nf ≤ 16384; the 7B's
        // nf = 18944 concat would otherwise waste ~2 GiB of VRAM on weights
        // no graph node ever reads — 7e⑤ review finding).
        #[cfg(feature = "cuda")]
        if plan.on_device(i) {
            if let Some(cuda) = crate::cuda::CudaState::get() {
                if let (Some(fg), Some(fu)) = (&layer.ffn_gate, &layer.ffn_up) {
                    let nf = fg.shape[1] as usize;
                    let fuse_ffn = nf <= 16384
                        && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1");
                    if fuse_ffn {
                        if let Some(data) = crate::cuda::concat_rows(&[fg, fu]) {
                            if fg.ttype == crate::tensor::TensorType::Q6_K {
                                cuda.register_weight_q6k_padded(
                                    &format!("{ns}blk.{i}.ffn_gu"),
                                    &data,
                                    (fg.shape[1] + fu.shape[1]) as usize,
                                    fg.shape[0] as usize,
                                );
                            } else {
                                cuda.register_weight(&format!("{ns}blk.{i}.ffn_gu"), &data);
                            }
                        }
                    }
                }
            }
        }
        layers.push(layer);
    }

    eprintln!("Loaded: {} layers", n_layer);

    let mut model = super::Qwen3Model {
        hparams,
        tok_embd: Some(tok_embd),
        output_norm,
        output: Some(output),
        output_b,
        layers,
        ns: ns.to_string(),
        offload: crate::models::OffloadState {
            plan,
            device_bytes: device_bytes.get(),
            source: offload_source,
        },
    };

    // E5: verify the plan against what actually got registered (see qwen2's loader twin).
    if !model.offload.plan.is_cpu_only()
        && super::graph::Qwen3Graph::device(&model) == crate::models::Device::Cpu
    {
        eprintln!(
"minfer: {} of {} blocks were asked onto the device ({}), but their weights are not usable there — running on CPU",
            model.offload.plan.gpu_layers, model.offload.plan.n_layers, model.offload.source
        );
        model.offload = crate::models::OffloadState::cpu_only(n_layer);
    }

    // 8p: warm the persistent per-weight f16 dequant cache at load.
    // Only for models big enough to amortize the +2 B/element resident
    // copy: small fixtures keep the exact pre-8p memory footprint.
    #[cfg(feature = "cuda")]
    if let Some(cuda) = crate::cuda::CudaState::get() {
        let warm: Vec<(&Option<crate::tensor::Tensor>, String)> = model
            .layers
            .iter()
            .enumerate()
            .flat_map(|(i, l)| {
                [
                    (&l.wq, tn::attn_q(i)),
                    (&l.wk, tn::attn_k(i)),
                    (&l.wv, tn::attn_v(i)),
                    (&l.wo, tn::attn_out(i)),
                    (&l.ffn_gate, tn::ffn_gate(i)),
                    (&l.ffn_up, tn::ffn_up(i)),
                    (&l.ffn_down, tn::ffn_down(i)),
                ]
            })
            .collect();
        let warm_bytes: usize = warm
            .iter()
            .filter_map(|(t, _)| t.as_ref())
            .map(|t| t.data.len())
            .sum();
        // R1: with the int8 MMQ prefill GEMM active the f16 cache would be
        // dead weight (MMQ streams raw quantized bytes) — skip the warm pass.
        let warmable = warm_bytes >= crate::cuda::W16_ENABLE_BYTES && !cuda.mmq_active();
        if warmable {
            cuda.enable_w16_cache();
        }
        for (t, name) in &warm {
            if warmable {
                if let Some(t) = t {
                    cuda.warm_w16(name, t);
                }
            }
        }
        if warmable {
            if let Some(out) = &model.output {
                cuda.warm_w16(tn::OUTPUT, out);
            }
        }
    }

    Some(model)
}
