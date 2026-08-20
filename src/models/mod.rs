// Model architecture trait + factory dispatch
// Reference: minfer2/src/models/mod.rs

pub mod qwen2;

use crate::cache::KVCache;
use crate::gguf::{GgufContext, GgufModel};
use crate::vec_ops::RopeStyle;

/// Architecture-agnostic model interface.
pub trait ModelDef {
    fn forward(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache, n_out: usize) -> Vec<f32>;
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
