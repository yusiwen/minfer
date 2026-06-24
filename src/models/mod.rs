// Model architecture trait + factory dispatch
// Reference: minfer2/src/models/mod.rs

pub mod qwen2;

use crate::cache::KVCache;
use crate::gguf::GgufContext;

/// Architecture-agnostic model interface.
pub trait ModelDef {
    fn forward(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache) -> Vec<f32>;
    fn format_chat(&self, messages: &[(String, String)]) -> String;
    fn special_tokens(&self) -> SpecialTokens;
    fn n_layer(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn n_embd_head(&self) -> usize;
    fn n_vocab(&self) -> usize;
}

/// Token IDs used by the sampler to stop generation.
pub struct SpecialTokens {
    pub eos: u32,
    pub im_end: Option<u32>,
}

/// Load a model from GGUF data, dispatching on `general.architecture`.
pub fn load_model(ctx: &GgufContext, raw: &[u8]) -> Option<Box<dyn ModelDef>> {
    let arch = ctx.get_key_val_str("general.architecture")?;
    match arch.as_str() {
        "qwen2" => {
            let model = qwen2::loader::load(ctx, raw)?;
            Some(Box::new(model))
        }
        other => {
            eprintln!("Unsupported architecture: '{}'", other);
            None
        }
    }
}
