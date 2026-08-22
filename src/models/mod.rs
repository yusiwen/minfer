// Model architecture trait + factory dispatch
// Reference: minfer2/src/models/mod.rs

pub mod qwen2;

use crate::cache::KVCache;
use crate::gguf::{GgufContext, GgufModel};
use crate::graph::cache::GraphCache;
use crate::vec_ops::RopeStyle;

/// Architecture-agnostic model interface.
pub trait ModelDef {
    fn forward(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache, n_out: usize) -> Vec<f32>;

    /// Downcast helper for the graph path's weight registration.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Build the declarative compute graph for one forward step (Phase 5).
    /// Topology is a deterministic function of `params` (reuse invariant).
    fn build_graph(&self, _params: &crate::graph::params::GraphParams) -> crate::graph::ComputeGraph {
        unimplemented!("build_graph not implemented for this architecture")
    }

    /// Graph-based forward (Phase 6); defaults to the imperative path.
    fn forward_graph(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache, n_out: usize) -> Vec<f32> {
        self.forward(tokens, positions, kv, n_out)
    }

    /// Graph-based forward with a caller-provided cache and explicit context
    /// size (server / multi-slot path, OPENAI-CHAT-API-PLAN.md Phase 0).
    ///
    /// `cache` owns the persistent KV regions and survives graph rebuilds;
    /// `n_ctx` sizes those regions. Callers must guarantee
    /// `positions[i] < n_ctx` for all `i`.
    fn forward_graph_cached(
        &self,
        tokens: &[u32],
        positions: &[usize],
        n_out: usize,
        n_ctx: usize,
        cache: &mut GraphCache,
    ) -> Vec<f32> {
        let _ = (tokens, positions, n_out, n_ctx, cache);
        unimplemented!("forward_graph_cached not implemented for this architecture")
    }

    fn format_chat(&self, messages: &[(String, String)]) -> String;
    fn special_tokens(&self) -> SpecialTokens;
    fn n_layer(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn n_embd_head(&self) -> usize;
    fn n_kv_embd(&self) -> usize;
    fn n_vocab(&self) -> usize;
    fn rope_style(&self) -> RopeStyle;
}

/// Token IDs used by the sampler to stop generation.
pub struct SpecialTokens {
    pub eos: u32,
    pub im_end: Option<u32>,
}

/// Load a model from GGUF (single file or multi-part split), dispatching on
/// `general.architecture` from part 0.
pub fn load_model(model: &GgufModel) -> Option<Box<dyn ModelDef>> {
    let ctx = &model.parts[0].ctx;
    let arch = ctx.get_key_val_str("general.architecture")?;
    match arch.as_str() {
        "qwen2" => {
            let m = qwen2::loader::load(model)?;
            Some(Box::new(m))
        }
        other => {
            eprintln!("Unsupported architecture: '{}'", other);
            None
        }
    }
}
