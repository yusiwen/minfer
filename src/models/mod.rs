// Model architecture trait + factory dispatch
// Reference: minfer2/src/models/mod.rs

pub mod qwen2;
pub mod qwen3;

use crate::cache::KVCache;
use crate::gguf::GgufModel;
use crate::graph::cache::GraphCache;
use crate::graph::offload::OffloadPlan;
use crate::vec_ops::RopeStyle;

/// E5: what the loader decided about layer offload, and what it measured while registering
/// weights. Stored on the model, so `device()`, the graph builder and the startup report all
/// read the same decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffloadState {
    /// The plan in force. After a load it is the **effective** plan: a request the device
    /// could not honour has already been cut back to CPU-only (with a printed reason).
    pub plan: OffloadPlan,
    /// Device bytes the offloaded weights occupy — summed while registering, so the report
    /// states what was actually placed rather than what the plan asked for.
    pub device_bytes: usize,
    /// Where the request came from ("--gpu-layers 4", "MINFER_GPU_LAYERS=4", "default"),
    /// so a surprising placement is traceable to the thing that asked for it.
    pub source: String,
}

impl OffloadState {
    /// The pre-E5 state: no block on the device, nothing measured.
    pub fn cpu_only(n_layers: usize) -> Self {
        Self {
            plan: OffloadPlan::all_on_cpu(n_layers),
            device_bytes: 0,
            source: "default".to_string(),
        }
    }

    /// E5's startup line, or `None` when there is nothing to report: the CPU-only path is
    /// silent **unless the request asked for it** (`--gpu-layers 0` deserves an answer; the
    /// pre-E5 default CPU run printed nothing and still should not).
    pub fn report(&self, device: Device) -> Option<String> {
        if self.plan.is_cpu_only() && self.source == "default" {
            return None;
        }
        Some(crate::graph::offload::report(
            self.plan,
            device_name(device),
            self.device_bytes,
            &self.source,
        ))
    }
}

/// The device's name as the offload report spells it.
pub fn device_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Cuda => "cuda",
    }
}

/// E5 S2: the device's own memory answer, with the query's outcome explicit (issue #122).
/// The `auto` offload fit consumes it, with the same quarter held back that E4's
/// feasibility gate uses.
///
/// This used to return `Option<usize>` with `None` meaning "no device" — but a **failed**
/// CUDA query came back as `Some(0)`, and `auto` then read it as "0 bytes free" and
/// silently planned 0 device blocks. The three-way [`DeviceMemory`] keeps the two apart:
/// the fit refuses a failed query with the real error and still fits nothing on a platform
/// that reports no free-bytes number at all.
pub fn device_memory() -> crate::graph::allocplan::DeviceMemory {
    #[cfg(feature = "cuda")]
    if let Some(cuda) = crate::cuda::CudaState::get() {
        return cuda.device_memory();
    }
    // Metal reports no free-bytes number through the current wrapper, so `auto` on macOS falls
    // back to an explicit `MINFER_GPU_MEM` (or fits nothing) — the device still participates,
    // it just cannot be planned against. Documented in rule 14.
    crate::graph::allocplan::DeviceMemory::NoDevice
}

/// E5: whether **any** device is present, decided before any weight is registered — the
/// offload plan is resolved at the start of a load, so "unset request" can mean "every block
/// the device can hold" without waiting for the registration checks that follow.
pub fn device_available() -> bool {
    #[cfg(target_os = "macos")]
    if crate::graph::metal_backend::metal_available() {
        return true;
    }
    #[cfg(feature = "cuda")]
    if crate::cuda::CudaState::get().is_some() {
        return true;
    }
    false
}

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

    /// Whether this device's attention kernel can gather a `kv_map` window — a
    /// query's allowed cells as a list of `(cell, len)` runs (C8b S4).
    ///
    /// This is the **single** authority for it, shared by the models' graph
    /// builder (which asks for the map input) and the server's admission (which
    /// shares a prefix in place only where a kernel can read one): if the two
    /// ever disagreed, the graph would be built with a one-range window and the
    /// share would make attention resolve a *set* of cells — the S2 refusal
    /// catches that loudly, but the point is that they cannot disagree.
    pub fn gathers_attn_map(self) -> bool {
        matches!(self, Device::Cpu | Device::Cuda)
    }

    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
            Device::Cuda => "cuda",
        }
    }

    /// F4: the backend registry handle this device names — the single bridge
    /// between the two id spaces, so a per-backend capability can be asked of the
    /// registry from a `Device` (this is what `KvFormat::supports` does).
    /// `Device` itself stays the coarse "the device participates" fact the
    /// server's batching default reads.
    pub fn backend(self) -> crate::graph::Backend {
        match self {
            Device::Cpu => crate::graph::Backend::CPU,
            Device::Metal => crate::graph::Backend::METAL,
            Device::Cuda => crate::graph::Backend::CUDA,
        }
    }
}

/// D3: should the FFN decode fusion be built as a **composition** (concat
/// `MatMul` + gate/up windows + in-place `SwiGLU`) rather than as the hand-written
/// `Op::FusedFFN` node?
///
/// The default is the **node**: it is the device-specific fast path (D2 measured
/// the composition 0.3-0.8% slower on CUDA for the models that actually fuse),
/// and Metal needs it until G5, since a partial window at a non-zero offset is
/// refused there. The composition stays reachable — `MINFER_FFN_COMPOSITION=1` —
/// because it is the *proof* that the fused node is expressible without special
/// cases, and the reference an A/B is run against.
///
/// Pure on purpose (CI has no GPU), and loud about the one combination that
/// cannot work: forcing the composition on a backend without offset views falls
/// back to the node with a warning rather than building a graph that would fail
/// at allocation.
pub fn ffn_composition(requested: Option<&str>, device: Device) -> bool {
    let forced = match requested {
        None => false,
        Some("1") => true,
        Some("0") => false,
        Some(other) => {
            eprintln!("[model] MINFER_FFN_COMPOSITION={other:?} is neither \"0\" nor \"1\"; using the default");
            false
        }
    };
    if forced && !matches!(device, Device::Cuda) {
        eprintln!(
            "[model] MINFER_FFN_COMPOSITION=1 ignored on {}: the composition needs a partial window at a \
             non-zero offset, which that backend does not have (D1/G5)",
            device.name()
        );
        return false;
    }
    forced
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

    /// C4 per-engine (issue #99): the KV storage format this engine resolved from
    /// `MINFER_CACHE_TYPE` at load (`graph::kvformat::resolve`, device-aware). It is a
    /// property of the loaded model, **not** a process global, so two engines with
    /// different formats coexist in one process without sizing each other's regions.
    /// The graph builder reads it through `CParams::kv_format` and the allocator
    /// through `GraphAllocator::set_kv_format`.
    fn kv_format(&self) -> crate::graph::kvformat::KvFormat;

    /// C4 per-engine: stamp the resolved format on the engine. Called once by the
    /// loader after `device()` is known. Required, not defaulted: an architecture that
    /// forgot it would silently build f32-width nodes for a packed run.
    fn set_kv_format(&mut self, format: crate::graph::kvformat::KvFormat);

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
    ///
    /// E5: with a partial offload plan this is "the device participates" — the *per-block*
    /// answer is `offload().on_device(block)`, and the graph's assignment pass reads that.
    fn device(&self) -> Device {
        Device::Cpu
    }

    /// E5: the layer offload plan in force (the **effective** one: a request the device
    /// could not honour has already been reduced to what it can). The default is "no block
    /// on the device", which is the pre-E5 behaviour for an implementation that does not
    /// override it.
    fn offload(&self) -> crate::graph::offload::OffloadPlan {
        crate::graph::offload::OffloadPlan::all_on_cpu(self.n_layer())
    }

    /// E5's startup report line ("which blocks landed where"), or `None` when there is
    /// nothing to report (the CPU-only path). The arch loaders implement it with the device
    /// memory their offloaded weights actually occupy.
    fn offload_report(&self) -> Option<String> {
        None
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
    load_model_with(model, ns, crate::graph::offload::OffloadRequest::Default)
}

/// Load with an explicit **E5 offload request** — the CLI's `--gpu-layers`. `Default` reads
/// `MINFER_GPU_LAYERS`, which is the interface the server and the tests use, so a caller that
/// has already parsed a number does not have to go through the environment.
pub fn load_model_with(
    model: &GgufModel,
    ns: &str,
    offload: crate::graph::offload::OffloadRequest,
) -> Option<Box<dyn ModelDef>> {
    // C4: `MINFER_CACHE_TYPE` is the interface the CLI and the server use. Resolved
    // once per load by `load_model_configured`, which is also the explicit-format
    // entry point a test uses instead of mutating the environment (issue #99).
    let cache_type = std::env::var("MINFER_CACHE_TYPE").ok();
    load_model_configured(model, ns, offload, cache_type.as_deref())
}

/// Load with an explicit **C4 KV cache type** (`MINFER_CACHE_TYPE`'s spelling, or
/// `None` for the unset default). This is what makes the format per engine instead
/// of per process: the answer is resolved against the loaded model's device and
/// stored on that model, and a caller can ask for two engines with different
/// formats in one process without touching a global (the C4 gate does exactly
/// that). `load_model_with` is the environment-backed wrapper.
pub fn load_model_configured(
    model: &GgufModel,
    ns: &str,
    offload: crate::graph::offload::OffloadRequest,
    cache_type: Option<&str>,
) -> Option<Box<dyn ModelDef>> {
    let ctx = &model.parts[0].ctx;
    let arch = ctx.get_key_val_str("general.architecture")?;
    let mut loaded: Box<dyn ModelDef> = match arch.as_str() {
        "qwen2" => Box::new(qwen2::loader::load(model, ns, offload)?),
        "qwen3" => Box::new(qwen3::loader::load(model, ns, offload)?),
        other => {
            eprintln!("Unsupported architecture: '{}'", other);
            return None;
        }
    };
    // C4: the KV storage format is resolved once the device is known (weights are
    // registered by the arch loader, so `device()` is the same all-or-nothing answer
    // every forward will use), and then **stored on the engine**. This is the loud
    // gate an unsupported `MINFER_CACHE_TYPE` fails on — an unknown spelling, or a
    // packed format the device has no kernel for, ends the load here instead of
    // quietly running f32.
    match crate::graph::kvformat::resolve(loaded.device(), cache_type) {
        Ok(format) => {
            loaded.set_kv_format(format);
            // C4 S2b: the *device* layout must be the format this load resolved.
            // The arch loaders already pushed `MINFER_CACHE_TYPE` through
            // `cuda::set_kv_cache_type`, but the resolver is the one authority, and
            // a packed region addressed as f32 rows is silent corruption — so a
            // `q8_0` resolution restates the layout here. `f32`/`f16` keep
            // `set_kv_cache_type`'s own auto policy (the region shape is the same
            // for both, so the pre-C4 split stands).
            //
            // This is the process-wide device half that #99 deliberately left in
            // place: the CUDA kernels read `cuda::KV_LAYOUT`, so the layout is still
            // a per-load process policy, and a device run keeps its serial
            // discipline. Making it per-graph is the filed follow-up.
            #[cfg(feature = "cuda")]
            if format == crate::graph::kvformat::KvFormat::Q8_0 {
                crate::cuda::set_kv_cache_layout(crate::cuda::KV_LAYOUT_Q8_0);
            }
        }
        Err(e) => {
            eprintln!("minfer: {e}");
            return None;
        }
    }
    // E5's third acceptance: say which blocks landed where (and how much device memory they
    // took) at startup — a placement nobody can see is a placement nobody can debug. Printed
    // after the KV policy resolved, so a load that fails there prints nothing.
    if let Some(line) = loaded.offload_report() {
        eprintln!("minfer: {line}");
    }
    Some(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D3: the gate's whole matrix, with no device — the same reason E6 made
    /// `batch_mode` pure: CI has no GPU, and this decides which graph a GPU run
    /// builds.
    #[test]
    fn ffn_composition_is_opt_in_and_refuses_where_it_cannot_work() {
        // Default: the hand-written node, on every device.
        for d in [Device::Cpu, Device::Metal, Device::Cuda] {
            assert!(!ffn_composition(None, d), "{d:?}");
        }
        // Forced on: allowed where offset views exist (CUDA), refused elsewhere
        // (loudly, in the function) so a Metal graph is never built with a
        // partial window it cannot express.
        assert!(ffn_composition(Some("1"), Device::Cuda));
        assert!(!ffn_composition(Some("1"), Device::Metal));
        assert!(!ffn_composition(Some("1"), Device::Cpu));
        // Forced off, and anything unrecognised, keeps the default.
        for v in ["0", "true", "banana", ""] {
            for d in [Device::Cpu, Device::Metal, Device::Cuda] {
                assert!(!ffn_composition(Some(v), d), "{v:?} {d:?}");
            }
        }
    }
}
