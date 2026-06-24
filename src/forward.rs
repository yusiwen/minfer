// Phase 7: Qwen2 Forward Pass + KV Cache
// Translated from: llama.cpp/src/models/qwen2.cpp (build_arch_graph, lines 49-154)
//   + ggml/src/ggml-cpu/ops.cpp (RMSNorm, Softmax, RoPE sections)
// Strict 1:1 translation — no extra code, no design changes

use crate::block;
use crate::model::{HParams, LayerWeights, Model};
use crate::tensor::Tensor;
use crate::vec_ops;

/// Maximum context length supported
const MAX_CTX: usize = 65536;

// ============================================================
// KV Cache (simplified — stores float vectors per layer)
// ============================================================

/// Per-layer KV cache entry
#[derive(Clone)]
pub struct KVCacheLayer {
    /// Stored K values: [max_seq_len, n_embd_kv_total]
    /// where n_embd_kv_total = n_head_kv * n_embd_head
    pub k: Vec<f32>,
    /// Stored V values: [max_seq_len, n_embd_kv_total]
    pub v: Vec<f32>,
    /// How many positions are currently cached
    pub size: usize,
    /// Capacity
    pub max_size: usize,
    /// Dimension per position
    pub dim: usize,
}

impl KVCacheLayer {
    pub fn new(max_size: usize, dim: usize) -> Self {
        Self {
            k: vec![0.0f32; max_size * dim],
            v: vec![0.0f32; max_size * dim],
            size: 0,
            max_size,
            dim,
        }
    }

    /// Store K and V at given position
    pub fn store(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let dim = self.dim;
        let offset = pos * dim;
        self.k[offset..offset + dim].copy_from_slice(k);
        self.v[offset..offset + dim].copy_from_slice(v);
        if pos + 1 > self.size {
            self.size = pos + 1;
        }
    }

    /// Get cached K for all positions up to size
    pub fn get_k(&self) -> &[f32] {
        &self.k[..self.size * self.dim]
    }

    /// Get cached V for all positions up to size
    pub fn get_v(&self) -> &[f32] {
        &self.v[..self.size * self.dim]
    }

    pub fn clear(&mut self) {
        self.size = 0;
    }
}

/// KV cache for all layers
#[derive(Clone)]
pub struct KVCache {
    pub layers: Vec<KVCacheLayer>,
}

impl KVCache {
    pub fn new(hp: &HParams) -> Self {
        let n_embd_head = hp.n_embd_head() as usize;
        let n_head_kv = hp.n_head_kv as usize;
        let dim = n_head_kv * n_embd_head;
        let max_size = hp.max_seq_len as usize;
        Self {
            layers: (0..hp.n_layer as usize)
                .map(|_| KVCacheLayer::new(max_size, dim))
                .collect(),
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

// ============================================================
// Token Embedding (Qwen2 — weight tying with output)
// ============================================================

/// Embed token IDs into f32 vectors
/// tok_embd: Tensor with shape [n_embd, n_vocab], Q4_0 or Q6_K or F32 type
pub fn embed_tokens(
    token_ids: &[u32],
    tok_embd: &Tensor,
    hp: &HParams,
) -> Vec<f32> {
    let n_embd = hp.n_embd as usize;
    let n_tokens = token_ids.len();
    let mut output = vec![0.0f32; n_tokens * n_embd];

    match tok_embd.ttype {
        crate::tensor::TensorType::F32 => {
            let data = tok_embd.data_f32();
            for (t, &token_id) in token_ids.iter().enumerate() {
                let src_offset = token_id as usize * n_embd;
                let dst_offset = t * n_embd;
                for j in 0..n_embd {
                    // GGUF stores layout as [n_embd_k, n_embd] = [n_embd, n_vocab]
                    output[dst_offset + j] = data[src_offset + j];
                }
            }
        }
        _ => {
            // For quantized embeddings: dequantize each block based on type
            // token_embd layout: [n_embd, n_vocab]
            let n_vocab = hp.n_vocab as usize;
            let blk_size = 32usize;
            let n_blocks_per_row = (n_embd + blk_size - 1) / blk_size;

            // Determine block byte size from type
            let blk_bytes = tok_embd.ttype.type_size();
            let is_q8_0 = tok_embd.ttype == crate::tensor::TensorType::Q8_0;

            for (t, &token_id) in token_ids.iter().enumerate() {
                let token_idx = token_id as usize;
                let dst_offset = t * n_embd;

                for b in 0..n_blocks_per_row {
                    let block_idx = token_idx * n_blocks_per_row + b;
                    let block_data = &tok_embd.data;
                    let off = block_idx * blk_bytes;
                    let d = u16::from_le_bytes([
                        block_data[off],
                        block_data[off + 1],
                    ]);
                    let d_f32 = block::fp16_to_f32(d);
                    let max_v = core::cmp::min(blk_size, n_embd - b * blk_size) as usize;

                    if is_q8_0 {
                        // Q8_0: values are direct i8 quants (no nibble unpacking)
                        for j in 0..max_v {
                            let qs = block_data[off + 2 + j] as i8;
                            output[dst_offset + b * blk_size + j] = qs as f32 * d_f32;
                        }
                    } else {
                        // Q4_0 / Q4_1 / others: nibble-based
                        for j in 0..max_v {
                            let nibble = if j % 2 == 0 {
                                block_data[off + 2 + j / 2] & 0x0F
                            } else {
                                block_data[off + 2 + j / 2] >> 4
                            };
                            let val = (nibble as i8 - 8) as f32 * d_f32;
                            output[dst_offset + b * blk_size + j] = val;
                        }
                    }
                }
            }
        }
    }

    output
}

// ============================================================
// GQA Attention
// ============================================================

/// Compute grouped-query attention for a single position (decode step)
/// q, k, v: per-head vectors after projection + RoPE
/// k_cache, v_cache: full cached sequence
/// n_head, n_head_kv, n_embd_head: model parameters
/// Returns: attention output [n_head * n_embd_head]
pub fn gqa_attention_decode(
    q: &[f32],          // [n_head * n_embd_head]
    k_cache: &[f32],    // [seq_len * n_head_kv * n_embd_head]
    v_cache: &[f32],    // [seq_len * n_head_kv * n_embd_head]
    n_head: usize,
    n_head_kv: usize,
    n_embd_head: usize,
    n_kv: usize,
) -> Vec<f32> {
    let n_groups = n_head / n_head_kv; // GQA groups
    let n_total = n_head * n_embd_head;
    let mut scores = vec![0.0f32; n_head * n_kv];
    let scale = 1.0 / (n_embd_head as f32).sqrt();

    // Compute attention scores: S[h][t] = Q[h] · K_cache[group(h)][t] * scale
    for h in 0..n_head {
        let g = h / n_groups; // which KV head group
        let q_offset = h * n_embd_head;
        for t in 0..n_kv {
            let k_offset = t * n_head_kv * n_embd_head + g * n_embd_head;
            let mut score = 0.0f32;
            for d in 0..n_embd_head {
                score += q[q_offset + d] * k_cache[k_offset + d];
            }
            scores[h * n_kv + t] = score * scale;
        }
    }

    // Softmax per head
    for h in 0..n_head {
        let offset = h * n_kv;
        // Find max
        let mut max_val = scores[offset];
        for t in 1..n_kv {
            if scores[offset + t] > max_val {
                max_val = scores[offset + t];
            }
        }
        // exp and sum
        let mut sum = 0.0f64;
        for t in 0..n_kv {
            let val = (scores[offset + t] - max_val).exp();
            scores[offset + t] = val;
            sum += val as f64;
        }
        let inv_sum = (1.0 / sum) as f32;
        for t in 0..n_kv {
            scores[offset + t] *= inv_sum;
        }
    }

    // Weighted sum of V: out[h] = Σ_t S[h][t] * V_cache[group(h)][t]
    let mut output = vec![0.0f32; n_total];
    for h in 0..n_head {
        let g = h / n_groups;
        let out_offset = h * n_embd_head;
        for d in 0..n_embd_head {
            let mut val = 0.0f32;
            for t in 0..n_kv {
                let v_offset = t * n_head_kv * n_embd_head + g * n_embd_head + d;
                val += scores[h * n_kv + t] * v_cache[v_offset];
            }
            output[out_offset + d] = val;
        }
    }

    output
}

// ============================================================
// QKV Projection for one position
// ============================================================

/// Project input through QKV weight matrices (Q4_0 × f32 using AVX2)
/// Returns (q, k, v) as flat f32 vectors
fn project_qkv(
    input: &[f32],      // [n_embd]
    layer: &LayerWeights,
    hp: &HParams,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_embd = hp.n_embd as usize;
    let n_head = hp.n_head as usize;
    let n_head_kv = hp.n_head_kv as usize;
    let n_embd_head = hp.n_embd_head() as usize;

    let n_q_total = n_head * n_embd_head;
    let n_kv_total = n_head_kv * n_embd_head;

    let mut q = vec![0.0f32; n_q_total];
    let mut k = vec![0.0f32; n_kv_total];
    let mut v = vec![0.0f32; n_kv_total];

    // Q projection: [n_embd] × [n_q_total, n_embd] → [n_q_total]
    project_row(input, &mut q, layer.wq.as_ref().unwrap(), n_embd, n_q_total);
    // Add Q bias (llama.cpp: create_tensor_qkv → layer.wq_b)
    if let Some(ref bq) = layer.bq {
        let bq_data = bq.data_f32();
        for i in 0..n_q_total.min(bq_data.len()) {
            q[i] += bq_data[i];
        }
    }
    // K projection: [n_embd] × [n_kv_total, n_embd] → [n_kv_total]
    project_row(input, &mut k, layer.wk.as_ref().unwrap(), n_embd, n_kv_total);
    // Add K bias (llama.cpp: layer.wk_b)
    if let Some(ref bk) = layer.bk {
        let bk_data = bk.data_f32();
        for i in 0..n_kv_total.min(bk_data.len()) {
            k[i] += bk_data[i];
        }
    }
    // V projection: [n_embd] × [n_kv_total, n_embd] → [n_kv_total]
    project_row(input, &mut v, layer.wv.as_ref().unwrap(), n_embd, n_kv_total);
    // Add V bias (llama.cpp: layer.wv_b)
    if let Some(ref bv) = layer.bv {
        let bv_data = bv.data_f32();
        for i in 0..n_kv_total.min(bv_data.len()) {
            v[i] += bv_data[i];
        }
    }

    (q, k, v)
}

/// Project input row through weight matrix (Q4_0/Q8_0 weight, f32 activation)
/// weight shape: [output_dim, input_dim] (each output dimension is a row)
/// Uses Q4_0×Q8_0 AVX2 dot product with runtime f32→Q8_0 quantization
fn project_row(
    input: &[f32],
    output: &mut [f32],
    weight: &Tensor,
    input_dim: usize,
    output_dim: usize,
) {
    match weight.ttype {
        crate::tensor::TensorType::F32 => {
            let w = weight.data_f32();
            for o in 0..output_dim {
                let mut sum = 0.0f32;
                let w_row = &w[o * input_dim..(o + 1) * input_dim];
                for i in 0..input_dim {
                    sum += input[i] * w_row[i];
                }
                output[o] = sum;
            }
        }
        crate::tensor::TensorType::Q4_0 => {
            let qk = 32; // QK4_0
            let nb = input_dim / qk;
            debug_assert!(input_dim % qk == 0,
                "project_row: input_dim {} not multiple of 32", input_dim);

            let blocks = weight.data_q4_0();

            // Step 1: quantize f32 input → Q8_0 blocks (quantize_row_q8_0, ggml-quants.c:238)
            let q8 = crate::avx2::quantize_row_q8_0(input);

            // Step 2: compute each output element via Q4_0×Q8_0 dot product
            for o in 0..output_dim {
                let row_blocks = &blocks[o * nb..(o + 1) * nb];
                output[o] = crate::avx2::vec_dot_q4_0_q8_0(input_dim as i32, row_blocks, &q8);
            }
        }
        crate::tensor::TensorType::Q8_0 => {
            // Q8_0 weight × f32 input: quantize input to Q8_0 → Q8_0×Q8_0 dot (quants.c lines 1170-1236)
            let qk = 32;
            let nb = input_dim / qk;
            debug_assert!(input_dim % qk == 0);
            let blocks = weight.data_q8_0();

            // Step 1: quantize f32 input → Q8_0 blocks
            let q8 = crate::avx2::quantize_row_q8_0(input);

            // Step 2: compute each output element via Q8_0×Q8_0 dot product
            for o in 0..output_dim {
                let row_blocks = &blocks[o * nb..(o + 1) * nb];
                output[o] = crate::avx2::vec_dot_q8_0_q8_0(input_dim as i32, row_blocks, &q8);
            }
        }
        _ => {
            panic!("Unsupported weight type: {:?}", weight.ttype);
        }
    }
}

// ============================================================
// SwiGLU FFN for one position
// ============================================================

/// Compute SwiGLU FFN for a single position
/// SwiGLU(x) = (SiLU(x·Wgate) ⊙ (x·Wup)) · Wdown
fn ffn_swiglu(
    input: &[f32],
    layer: &LayerWeights,
    hp: &HParams,
) -> Vec<f32> {
    let n_embd = hp.n_embd as usize;
    let n_ff = hp.n_ff as usize;

    // gate = input × Wgate  [n_ff]
    let mut gate = vec![0.0f32; n_ff];
    project_row(input, &mut gate, layer.ffn_gate.as_ref().unwrap(), n_embd, n_ff);

    // up = input × Wup  [n_ff]
    let mut up = vec![0.0f32; n_ff];
    project_row(input, &mut up, layer.ffn_up.as_ref().unwrap(), n_embd, n_ff);

    // Apply SiLU activation on a copy (can't borrow gate as mut and immut)
    let gate_input = gate.clone();
    vec_ops::vec_silu_f32(n_ff, &mut gate, &gate_input);

    // gate ⊙ up (element-wise multiply)
    for i in 0..n_ff {
        gate[i] *= up[i];
    }

    // down = (gate ⊙ up) × Wdown  [n_embd]
    let mut output = vec![0.0f32; n_embd];
    project_row(&gate, &mut output, layer.ffn_down.as_ref().unwrap(), n_ff, n_embd);

    output
}

// ============================================================
// Single Position Forward (decode step)
// ============================================================

/// Run the Qwen2 transformer for a single token position (decode)
/// input: [n_embd] — embedded token vector
/// Returns: (logits [n_vocab], hidden_state [n_embd])
pub fn forward_decode(
    input: &[f32],
    pos: usize,
    model: &Model,
    cache: &mut KVCache,
) -> (Vec<f32>, Vec<f32>) {
    let hp = &model.hparams;
    let n_embd = hp.n_embd as usize;
    let n_head = hp.n_head as usize;
    let n_head_kv = hp.n_head_kv as usize;
    let n_embd_head = hp.n_embd_head() as usize;
    let n_vocab = hp.n_vocab as usize;
    let eps = hp.f_norm_rms_eps;

    let mut hidden = input.to_vec();

    for il in 0..model.n_layer() {
        let layer = model.layer(il);
        let cache_layer = &mut cache.layers[il];

        // === Attention path ===

        // 1. RMSNorm (attention)
        let mut normed = vec![0.0f32; n_embd];
        let norm_weight = layer.attn_norm.as_ref().unwrap();
        if norm_weight.ttype == crate::tensor::TensorType::F32 {
            vec_ops::rms_norm_f32(n_embd, &mut normed, &hidden, eps);
            // Apply norm weight (element-wise multiply)
            let w = norm_weight.data_f32();
            for i in 0..n_embd {
                normed[i] *= w[i];
            }
        }

        // 2. QKV Projection
        let (mut q, k, v) = project_qkv(&normed, layer, hp);

        // 3. Apply RoPE to Q and K (per head)
        let rope_base = hp.rope_freq_base;
        let rope_scale = hp.rope_freq_scale;

        // RoPE on each Q head (clone to avoid aliasing)
        let q_input = q.clone();
        for h in 0..n_head {
            let offset = h * n_embd_head;
            vec_ops::rope_f32(
                n_embd_head,
                &mut q[offset..offset + n_embd_head],
                &q_input[offset..offset + n_embd_head],
                pos as i32,
                rope_base,
                rope_scale,
            );
        }
        // RoPE on each KV head (clone to avoid aliasing)
        let k_input = k.clone();
        let mut k_rope = k.clone();
        for h in 0..n_head_kv {
            let offset = h * n_embd_head;
            vec_ops::rope_f32(
                n_embd_head,
                &mut k_rope[offset..offset + n_embd_head],
                &k_input[offset..offset + n_embd_head],
                pos as i32,
                rope_base,
                rope_scale,
            );
        }

        // 4. Store KV in cache
        cache_layer.store(pos, &k_rope, &v);

        // 5. GQA Attention
        let n_kv = cache_layer.size;
        let attn_out = gqa_attention_decode(
            &q,
            cache_layer.get_k(),
            cache_layer.get_v(),
            n_head,
            n_head_kv,
            n_embd_head,
            n_kv,
        );

        // 6. Output projection
        let mut attn_proj = vec![0.0f32; n_embd];
        project_row(&attn_out, &mut attn_proj, layer.wo.as_ref().unwrap(), n_embd, n_embd);

        // 7. Residual connection
        for i in 0..n_embd {
            attn_proj[i] += hidden[i];
        }

        // === FFN path ===

        // 8. RMSNorm (FFN)
        let mut ffn_normed = vec![0.0f32; n_embd];
        let ffn_norm_weight = layer.ffn_norm.as_ref().unwrap();
        if ffn_norm_weight.ttype == crate::tensor::TensorType::F32 {
            vec_ops::rms_norm_f32(n_embd, &mut ffn_normed, &attn_proj, eps);
            let w = ffn_norm_weight.data_f32();
            for i in 0..n_embd {
                ffn_normed[i] *= w[i];
            }
        }

        // 9. SwiGLU FFN
        let ffn_out = ffn_swiglu(&ffn_normed, layer, hp);

        // 10. Residual connection
        for i in 0..n_embd {
            hidden[i] = ffn_out[i] + attn_proj[i];
        }
    }

    // === Final RMSNorm ===
    let mut final_normed = vec![0.0f32; n_embd];
    let output_norm = model.output_norm.as_ref().unwrap();
    if output_norm.ttype == crate::tensor::TensorType::F32 {
        vec_ops::rms_norm_f32(n_embd, &mut final_normed, &hidden, eps);
        let w = output_norm.data_f32();
        for i in 0..n_embd {
            final_normed[i] *= w[i];
        }
    }

    // === LM Head (output projection) ===
    let mut logits = vec![0.0f32; n_vocab];
    if let Some(output) = &model.output {
        project_row(&final_normed, &mut logits, output, n_embd, n_vocab);
    }

    (logits, final_normed)
}
