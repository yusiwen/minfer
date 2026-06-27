// Phase 9: End-to-End Inference Engine
// Translated from: llama.cpp/src/llama-context.cpp (llama_decode, context init)
//   + llama.cpp/examples/embedding/embedding.cpp (main flow pattern)
//   + llama.cpp/common/common.cpp (sampling, tokenization helpers)

mod gguf;
mod block;
mod avx2;
mod tensor;
mod vec_ops;
mod sampler;
mod tokenizer;
mod template;
mod cache;
mod models;
mod download;

use std::time::Instant;

struct GenParams {
    n_predict: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    n_ctx: usize,
}

impl Default for GenParams {
    fn default() -> Self {
        Self { n_predict: 512, temp: 0.0, top_k: 40, top_p: 0.9, seed: 42, n_ctx: 4096 }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} <model> [prompt]              — run inference", args[0]);
        eprintln!("  {} download hf <repo> [file]      — download from Hugging Face", args[0]);
        eprintln!("  {} download ollama <model>[:tag]   — pull from Ollama", args[0]);
        eprintln!("  {} list                           — list cached models", args[0]);
        std::process::exit(1);
    }

    // === Subcommands ===
    match args[1].as_str() {
        "download" => {
            if args.len() < 4 {
                eprintln!("Usage: {} download hf <repo> [file] | ollama <model>[:tag]", args[0]);
                std::process::exit(1);
            }
            let source = &args[2];
            let target = &args[3];
            match source.as_str() {
                "hf" => {
                    let uri = if args.len() > 4 {
                        format!("hf:{}:{}", target, args[4])
                    } else {
                        format!("hf:{}", target)
                    };
                    match download::resolve(&uri) {
                        Ok(p) => println!("Model downloaded: {}", p.display()),
                        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
                    }
                }
                "ollama" => {
                    let uri = format!("ollama:{}", target);
                    match download::resolve(&uri) {
                        Ok(p) => println!("Model ready: {}", p.display()),
                        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
                    }
                }
                _ => {
                    eprintln!("Unknown download source '{}'. Use 'hf' or 'ollama'.", source);
                    std::process::exit(1);
                }
            }
            return;
        }
        "list" => {
            match download::list_local() {
                Ok(()) => {}
                Err(e) => eprintln!("Error: {}", e),
            }
            return;
        }
        _ => {}  // fall through to model inference
    }

    let model_path = &args[1];

    // Auto-download if URI starts with hf: or ollama:
    let model_path = if model_path.starts_with("hf:") || model_path.starts_with("ollama:") {
        match download::resolve(model_path) {
            Ok(p) => {
                eprintln!("Model ready: {}", p.display());
                p.to_string_lossy().to_string()
            }
            Err(e) => {
                eprintln!("Download error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        model_path.clone()
    };
    let prompt = if args.len() > 2 { args[2..].join(" ") } else {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap_or(0);
        input.trim().to_string()
    };
    let params = GenParams::default();

    // === Load GGUF ===
    println!("Loading model: {} ...", model_path);
    let data = {
        let mut f = std::fs::File::open(model_path).expect("open model");
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut f, &mut buf).expect("read model");
        buf
    };
    println!("File: {} bytes ({:.1} MB)", data.len(), data.len() as f64 / 1_048_576.0);

    let ctx = gguf::GgufContext::init_from_data(&data).expect("parse GGUF");
    println!("GGUF: {} KV, {} tensors", ctx.kv.len(), ctx.info.len());

    // === Load model (dispatches on general.architecture) ===
    let model = models::load_model(&ctx, &data).expect("load model");
    println!("Model loaded.");

    // === KV Cache ===
    let n_embd_head = model.n_embd_head();
    let n_head_kv = model.n_head_kv();
    let n_layer = model.n_layer();
    let n_vocab = model.n_vocab();
    let mut kv_cache = cache::KVCache::new(n_layer, n_head_kv, n_embd_head, params.n_ctx);

    // === Tokenizer ===
    let tokenizer = tokenizer::Tokenizer::load(&ctx);
    println!("Vocabulary: {} tokens", tokenizer.vocab_size());

    // === Chat template (need tokenizer for bos_token text) ===
    let processed = if let Some(tmpl) = get_chat_template(&data) {
        let bos_text = tokenizer.id_to_token.get(tokenizer.bos_token as usize)
            .map(|s| s.as_str())
            .unwrap_or("");
        template::render_template(&tmpl, &prompt, true, bos_text)
    } else {
        prompt.clone()
    };
    let input_ids = tokenizer.encode(&processed);
    if input_ids.is_empty() { eprintln!("tokenize failed"); std::process::exit(1); }
    println!("Prompt: {} tokens", input_ids.len());

    // === Prefill ===
    let infer_start = Instant::now();
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    let logits = model.forward(&input_ids, &positions, &mut kv_cache);
    let last_logits: Vec<f32> = logits[(input_ids.len() - 1) * n_vocab..].to_vec();

    let prefill_time = infer_start.elapsed();
    println!("Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        input_ids.len(), prefill_time.as_secs_f64(),
        input_ids.len() as f64 / prefill_time.as_secs_f64());

    // === Generate ===
    let mut logits = last_logits;
    let mut generated: Vec<u32> = Vec::new();
    let special = model.special_tokens();
    let mut current_pos = input_ids.len();

    while generated.len() < params.n_predict {
        let sampled = if params.temp < 1e-6 {
            sampler::sample_greedy(&logits)
        } else {
            sampler::apply_top_k(&mut logits, params.top_k);
            sampler::apply_top_p(&mut logits, params.top_p);
            sampler::sample_temperature(&mut logits, params.temp)
        };

        if is_stop_token(sampled.token_id, &special) { break; }
        generated.push(sampled.token_id);
        print!("{}", tokenizer.decode(&[sampled.token_id]));
        std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());

        let logits_all = model.forward(&[sampled.token_id], &[current_pos], &mut kv_cache);
        logits = logits_all[..n_vocab].to_vec();
        current_pos += 1;
    }

    println!();
    let total_time = infer_start.elapsed();
    println!("\n---");
    println!("Generated: {} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(), total_time.as_secs_f64(),
        (input_ids.len() + generated.len()) as f64 / total_time.as_secs_f64());
}

fn is_stop_token(id: u32, special: &models::SpecialTokens) -> bool {
    id == special.eos || Some(id) == special.im_end || id == 0 || id == 2
}

fn get_chat_template(data: &[u8]) -> Option<String> {
    if let Some(ctx) = gguf::GgufContext::init_from_data(data) {
        for kv in &ctx.kv {
            if kv.key == "tokenizer.chat_template" {
                return Some(kv.get_val_str(0).to_string());
            }
        }
    }
    None
}
