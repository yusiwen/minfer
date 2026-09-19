// Model architecture trait + factory dispatch
// Reference: minfer2/src/models/mod.rs

pub mod qwen2;
pub mod qwen3;

use crate::cache::KVCache;
use crate::gguf::GgufModel;
use crate::graph::cache::GraphCache;
use crate::vec_ops::RopeStyle;

/// Which backend a model's forwards actually run on (E6).
///
/// The variants exist unconditionally so callers need no `cfg`: an unavailable
/// backend is simply never returned. This is the *fact* the server's batching
/// default keys off (continuous batching is a measured win on CUDA and a
/// measured loss on CPU), and it is deliberately the same gate the graph builder
/// uses — see `graph::Qwen2Graph::device`, the single authority for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

impl Device {
    pub fn is_gpu(self) -> bool {
        !matches!(self, Device::Cpu)
    }

    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
            Device::Cuda => "cuda",
        }
    }
}

/// Architecture-agnostic model interface.
///
/// `Send + Sync` so a model can be shared across threads (the HTTP server's
/// worker thread owns the only inference path; `Arc<dyn ModelDef>` is used by
/// the OpenAI-compatible server, OPENAI-CHAT-API-PLAN.md).
// NOTE: several required methods (as_any, forward_graph, format_chat,
// n_head_kv, n_embd_head, n_kv_embd, n_vocab, rope_style) are only reached
// through `Box<dyn ModelDef>` calls or planned paths; rustc's dead-code lint
// flags them anyway, so the trait is allow'd as the model API surface.
#[allow(dead_code)]
pub trait ModelDef: Send + Sync {
    /// Single-shot forward. `n_ctx` sizes the graph's persistent KV regions
    /// (the graph path; the legacy `kv` arg is ignored there). Callers must
    /// guarantee `positions[i] < n_ctx` for every position.
    fn forward(
        &self,
        tokens: &[u32],
        positions: &[usize],
        kv: &mut KVCache,
        n_out: usize,
        n_ctx: usize,
    ) -> Vec<f32>;

    /// Downcast helper for the graph path's weight registration.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Build the declarative compute graph for one forward step (Phase 5).
    /// Topology is a deterministic function of `params` (reuse invariant).
    fn build_graph(
        &self,
        _params: &crate::graph::params::GraphParams,
    ) -> crate::graph::ComputeGraph {
        unimplemented!("build_graph not implemented for this architecture")
    }

    /// Graph-based forward (Phase 6); defaults to the imperative path.
    fn forward_graph(
        &self,
        tokens: &[u32],
        positions: &[usize],
        kv: &mut KVCache,
        n_out: usize,
        n_ctx: usize,
    ) -> Vec<f32> {
        self.forward(tokens, positions, kv, n_out, n_ctx)
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

    /// One forward over a batch of sequences (Phase E / E2). Returns one logits
    /// row per sequence in batch order (`n_tokens * n_vocab` for a decode batch
    /// of `n_tokens` sequences, one token each).
    ///
    /// The default refuses: an architecture that has not been taught batching
    /// must not silently serve a batch as if it were one sequence.
    fn forward_batch(
        &self,
        _batch: &crate::graph::batch::Batch,
        _n_out: usize,
        _n_ctx: usize,
        _cache: &mut crate::graph::cache::GraphCache,
    ) -> Vec<f32> {
        unimplemented!("forward_batch not implemented for this architecture")
    }

    /// Where this model's forwards run (E6). The default is CPU, which is the
    /// truth for any implementation that does not override it.
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn format_chat(&self, messages: &[(String, String)]) -> String;
    fn special_tokens(&self) -> SpecialTokens;
    fn n_layer(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn n_embd_head(&self) -> usize;
    fn n_kv_embd(&self) -> usize;
    fn n_vocab(&self) -> usize;
    fn rope_style(&self) -> RopeStyle;

    /// RoPE base and frequency scale (`(freq_base, freq_scale)`). The context
    /// shift re-ropes stored K rows after a position change, which needs both
    /// (Phase C / C2).
    fn rope_params(&self) -> (f32, f32);
}

/// Token IDs used by the sampler to stop generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecialTokens {
    pub eos: u32,
    pub im_end: Option<u32>,
}

/// Load a model from GGUF (single file or multi-part split), dispatching on
/// `general.architecture` from part 0.
pub fn load_model(model: &GgufModel) -> Option<Box<dyn ModelDef>> {
    load_model_ns(model, "")
}

/// Load with a GPU weight-registry namespace. The first (primary) model of a
/// process uses `""` — every name resolves exactly as before. A second model
/// (the D5-R spec-decode draft) must pass a distinct prefix (e.g. "draft."):
/// the registry is process-global and name-keyed, and without the prefix the
/// second load collides with the first, fails its all-or-nothing CUDA check,
/// and silently drops the primary model to CPU.
pub fn load_model_ns(model: &GgufModel, ns: &str) -> Option<Box<dyn ModelDef>> {
    let ctx = &model.parts[0].ctx;
    let arch = ctx.get_key_val_str("general.architecture")?;
    match arch.as_str() {
        "qwen2" => {
            let m = qwen2::loader::load(model, ns)?;
            Some(Box::new(m))
        }
        "qwen3" => {
            let m = qwen3::loader::load(model, ns)?;
            Some(Box::new(m))
        }
        other => {
            eprintln!("Unsupported architecture: '{}'", other);
            None
        }
    }
}
