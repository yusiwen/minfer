//! The engine as a library (#763 S6a): load a GGUF, run one chat turn, print it.
//!
//! ```text
//! cargo run --release --example embed -- <model.gguf> ["prompt"]
//! ```

use minfer::conversation::TurnParams;
use minfer::engine::{Engine, SessionConfig};
use minfer::sampler::SamplerConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: embed <model.gguf> [prompt]")?;
    let prompt = args.next().unwrap_or_else(|| "hello".to_string());

    let engine = Engine::load(std::path::Path::new(&path))?;
    if engine.chatml_notice() {
        eprintln!("[embed] no chat template: using the ChatML fallback");
    }
    let mut session = engine.session(SessionConfig::default());
    let params = TurnParams {
        n_predict: 64,
        sampler: SamplerConfig::default(),
        stop_strings: Vec::new(),
    };
    let mut out = Vec::new();
    session.chat(&prompt, &params, &mut |delta| out.extend_from_slice(delta))?;
    print!("{}", String::from_utf8_lossy(&out));
    Ok(())
}
