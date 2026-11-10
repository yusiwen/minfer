// Qwen2 model definition + ModelDef impl

pub mod graph;
pub mod loader;

use crate::models::{ModelDef, SpecialTokens};
use crate::tensor::Tensor;

pub use loader::{HParams, LayerWeights};

/// Qwen2 model with all weights loaded.
#[derive(Clone)]
pub struct Qwen2Model {
    pub hparams: HParams,
    pub tok_embd: Option<Tensor>,
    pub output_norm: Option<Tensor>,
    pub output: Option<Tensor>,
    pub output_b: Option<Tensor>,
    pub layers: Vec<LayerWeights>,
    /// GPU weight-registry namespace. The registry is process-global and
    /// name-keyed ("single-model-per-process" held until D5-R speculative
    /// decoding loaded a second, draft model whose GGUF tensor names collide
    /// with the target's — the collision silently failed the target's
    /// all-or-nothing CUDA check and dropped it to CPU). The first model
    /// loads with `""` (all names unchanged); every later model gets a
    /// prefix ("draft.") applied to its tensor names at load, so
    /// registration, graph build, and the participation checks all resolve
    /// namespaced keys consistently.
    pub ns: String,
    /// E5: the layer offload plan in force (the effective one) plus what the device
    /// registration measured. The graph builders and the assignment pass read this
    /// **field** (`model.offload.plan`) directly — it is the one plan surface since
    /// [#244] deleted the `ModelDef::offload()` method. `offload_report()` prints the line.
    ///
    /// [#244]: https://github.com/yusiwen/minfer/issues/244
    pub offload: crate::models::OffloadState,
    /// C4 per-engine (issue #99): the KV storage format this engine resolved from
    /// `MINFER_CACHE_TYPE` at load. `F32` until `ModelDef::set_kv_format` stamps it;
    /// the graph builder and the allocator read it, so no test or second model can
    /// change it out from under this engine.
    pub kv_format: crate::graph::kvformat::KvFormat,
    /// [#758] the model's CUDA registration. The loader creates it before the
    /// first weight registration; dropping the last clone releases the model's
    /// device allocations (or retires them until the last captured graph that
    /// references them is dropped — ADR-0078). `None` in a build without the
    /// `cuda` feature or when the device is unavailable.
    #[cfg(feature = "cuda")]
    pub cuda: Option<std::sync::Arc<crate::cuda::CudaModelRegistration>>,
    /// [#745] the model's Metal registration. The loader creates it before the
    /// first part/weight registration; dropping the last clone releases the
    /// device buffers and then the mapping handles. `None` off macOS or when MPS
    /// is unavailable.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub metal: Option<std::sync::Arc<crate::metal::ModelRegistration>>,
}

impl Qwen2Model {
    /// Inherent accessor. Most callers use the trait's `n_layer()` (the graph
    /// tests go through `Box<dyn ModelDef>`), but a concrete `&Qwen2Model`
    /// caller exists — `forward_batch`'s `MINFER_GRAPH_DUMP` branch in
    /// `models/qwen2/graph.rs` — so this is live, not a dead helper.
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }
}

impl ModelDef for Qwen2Model {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn kv_format(&self) -> crate::graph::kvformat::KvFormat {
        self.kv_format
    }

    fn set_kv_format(&mut self, format: crate::graph::kvformat::KvFormat) {
        self.kv_format = format;
    }

    /// Where this model's forwards run (E6) — delegated to the graph's single
    /// authority, so the server's batching default and `CParams.gpu` agree.
    fn device(&self) -> crate::models::Device {
        // [#758] a live CUDA model owns a registration; without one it has no
        // release path for the weight buffers it registered, so it must not claim
        // the device (the CUDA twin of the Metal gate below).
        #[cfg(feature = "cuda")]
        if crate::cuda::CudaState::get().is_some() && self.cuda.is_none() {
            return crate::models::Device::Cpu;
        }
        // [#745] a live MPS model owns a registration; without one it cannot
        // be the Metal path (the registration is what keeps the pages mapped).
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if crate::metal::MpsState::get().is_some() && self.metal.is_none() {
            return crate::models::Device::Cpu;
        }
        graph::Qwen2Graph::device(self)
    }

    /// E5: the startup line naming where each block landed, or `None` on the CPU-only path.
    fn offload_report(&self) -> Option<String> {
        self.offload.report(self.device())
    }

    fn build_graph(
        &self,
        params: &crate::graph::params::GraphParams,
    ) -> crate::graph::ComputeGraph {
        graph::Qwen2Graph::build(self, params)
    }

    fn forward_graph_cached(
        &self,
        tokens: &[u32],
        positions: &[usize],
        n_out: usize,
        n_ctx: usize,
        cache: &mut crate::graph::cache::GraphCache,
    ) -> Vec<f32> {
        graph::Qwen2Graph::forward_cached(self, tokens, positions, n_out, n_ctx, cache)
    }

    fn forward_batch(
        &self,
        batch: &crate::graph::batch::Batch,
        n_out: usize,
        n_ctx: usize,
        cache: &mut crate::graph::cache::GraphCache,
    ) -> Vec<f32> {
        graph::Qwen2Graph::forward_batch(self, batch, n_out, n_ctx, cache)
    }

    fn special_tokens(&self) -> SpecialTokens {
        let eos = self.hparams.eos_token_id;
        let im_end = self.hparams.im_end_token_id;
        SpecialTokens {
            eos,
            im_end,
            eot: self.hparams.eot_token_id,
            eom: self.hparams.eom_token_id,
        }
    }

    fn n_layer(&self) -> usize {
        self.hparams.n_layer as usize
    }
    fn n_head_kv(&self) -> usize {
        self.hparams.n_head_kv as usize
    }
    fn n_embd_head(&self) -> usize {
        self.hparams.n_embd_head() as usize
    }
    fn n_kv_embd(&self) -> usize {
        self.hparams.n_kv_embd as usize
    }
    fn n_vocab(&self) -> usize {
        self.hparams.n_vocab as usize
    }
    fn max_seq_len(&self) -> usize {
        self.hparams.max_seq_len as usize
    }
    fn rope_style(&self) -> crate::vec_ops::RopeStyle {
        self.hparams.rope_style
    }
    fn rope_params(&self) -> crate::vec_ops::RopeParams {
        crate::vec_ops::RopeParams {
            freq_base: self.hparams.rope_freq_base,
            freq_scale: self.hparams.rope_freq_scale,
            freqs: self.hparams.rope_freqs,
        }
    }
}

// ============================================================
// GGUF tensor name mapping
// ============================================================

pub mod tensor_names {
    pub const TOKEN_EMBD: &str = "token_embd.weight";

    /// #724: the `llama3` per-pair frequency factors llama.cpp's converter writes
    /// (one tensor per block; the supported files repeat one table for all blocks).
    pub fn rope_freqs(i: usize) -> String {
        format!("blk.{i}.rope_freqs.weight")
    }
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    pub const OUTPUT: &str = "output.weight";
    pub const OUTPUT_BIAS: &str = "output.bias";

    pub fn attn_norm(i: usize) -> String {
        format!("blk.{}.attn_norm.weight", i)
    }
    pub fn attn_q(i: usize) -> String {
        format!("blk.{}.attn_q.weight", i)
    }
    pub fn attn_q_bias(i: usize) -> String {
        format!("blk.{}.attn_q.bias", i)
    }
    pub fn attn_k(i: usize) -> String {
        format!("blk.{}.attn_k.weight", i)
    }
    pub fn attn_k_bias(i: usize) -> String {
        format!("blk.{}.attn_k.bias", i)
    }
    pub fn attn_v(i: usize) -> String {
        format!("blk.{}.attn_v.weight", i)
    }
    pub fn attn_v_bias(i: usize) -> String {
        format!("blk.{}.attn_v.bias", i)
    }
    pub fn attn_out(i: usize) -> String {
        format!("blk.{}.attn_output.weight", i)
    }
    pub fn ffn_norm(i: usize) -> String {
        format!("blk.{}.ffn_norm.weight", i)
    }
    pub fn ffn_gate(i: usize) -> String {
        format!("blk.{}.ffn_gate.weight", i)
    }
    pub fn ffn_up(i: usize) -> String {
        format!("blk.{}.ffn_up.weight", i)
    }
    pub fn ffn_down(i: usize) -> String {
        format!("blk.{}.ffn_down.weight", i)
    }
}
