// Phase 8: Model Loader — load tensors from GGUF into Model struct
// Translated from: llama.cpp/src/llama-model-loader.cpp (simplified)
//   + models/qwen2.cpp (load_arch_tensors)

use crate::gguf::GgufContext;
use crate::model::{self, Model, tensor_names as tn};
use crate::tensor::{Tensor, TensorType};

/// Load all tensors from GGUF data into a Model
pub fn load_model(data: &[u8]) -> Option<Model> {
    let ctx = GgufContext::init_from_data(data)?;
    println!("GGUF: {} KV, {} tensors", ctx.kv.len(), ctx.info.len());

    let hparams = model::hparams_from_gguf(&ctx)?;
    println!("Qwen2-{}B: n_embd={}, n_head={}, n_head_kv={}, n_ff={}, n_vocab={}",
        hparams.n_layer, hparams.n_embd, hparams.n_head, hparams.n_head_kv, hparams.n_ff, hparams.n_vocab);
    println!("  rope_base={}, rms_eps={}", hparams.rope_freq_base, hparams.f_norm_rms_eps);

    let mut tensor_map = std::collections::HashMap::<String, &crate::gguf::GgufTensorInfo>::new();
    for ti in &ctx.info {
        tensor_map.insert(ti.name.clone(), ti);
    }

    let mut model = Model::new(hparams);

    // Helper: load a named tensor from GGUF
    let load_one = |n: &str| -> Option<Tensor> {
        tensor_map.get(n).map(|ti| {
            load_gguf_tensor(&ctx, data, ti, n)
        })
    };

    if let Some(t) = load_one(tn::TOKEN_EMBD) { model.set_tok_embd(t); }
    if let Some(t) = load_one(tn::OUTPUT_NORM) { model.set_output_norm(t); }

    if let Some(t) = load_one(tn::OUTPUT) {
        model.set_output(t);
    } else {
        println!("  Weight tying: output.weight ← token_embd.weight");
        if let Some(ref tok) = model.tok_embd { model.set_output(tok.clone()); }
    }

    if let Some(t) = load_one(tn::OUTPUT_BIAS) { model.set_output_b(t); }

    let n_layer = model.n_layer();
    for i in 0..n_layer {
        let layer = model.layer_mut(i);
        if let Some(ti) = tensor_map.get(&tn::attn_norm(i)) {
            layer.attn_norm = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_norm(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_q(i)) {
            layer.wq = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_q(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_q_bias(i)) {
            layer.bq = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_q_bias(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k(i)) {
            layer.wk = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_k(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k_bias(i)) {
            layer.bk = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_k_bias(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_v(i)) {
            layer.wv = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_v(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_v_bias(i)) {
            layer.bv = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_v_bias(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_out(i)) {
            layer.wo = Some(load_gguf_tensor(&ctx, data, ti, &tn::attn_out(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_norm(i)) {
            layer.ffn_norm = Some(load_gguf_tensor(&ctx, data, ti, &tn::ffn_norm(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_gate(i)) {
            layer.ffn_gate = Some(load_gguf_tensor(&ctx, data, ti, &tn::ffn_gate(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_up(i)) {
            layer.ffn_up = Some(load_gguf_tensor(&ctx, data, ti, &tn::ffn_up(i)));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_down(i)) {
            layer.ffn_down = Some(load_gguf_tensor(&ctx, data, ti, &tn::ffn_down(i)));
        }
    }

    println!("Loaded: {} layers", n_layer);
    Some(model)
}

fn load_gguf_tensor(ctx: &GgufContext, raw: &[u8],
    ti: &crate::gguf::GgufTensorInfo, name: &str) -> Tensor
{
    let ttype = TensorType::from_ggml_type(ti.type_);
    let mut shape = [1i64; 4];
    for j in 0..4 { shape[j] = ti.ne[j]; }

    let off = ctx.offset + ti.offset as usize;
    let ts = ttype.type_size();
    let bs = ttype.blck_size() as usize;
    let n = (shape[0] * shape[1] * shape[2] * shape[3]) as usize;
    let nbytes = (n / bs) * ts;
    let src = &raw[off..off + nbytes];

    let mut data = Vec::with_capacity(src.len());
    data.extend_from_slice(src);

    let mut strides = [0usize; 4];
    strides[0] = ts;
    strides[1] = strides[0] * (shape[0] / bs as i64) as usize;
    for j in 2..4 {
        strides[j] = strides[j - 1] * shape[j - 1] as usize;
    }

    let mut tensor = Tensor::from_data_with_strides(ttype, &shape, &strides, data);
    tensor.set_name(name);
    tensor
}
