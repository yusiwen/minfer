// End-to-End Inference Engine
// Translated from: llama.cpp/src/llama-context.cpp (llama_decode, context init)
//   + llama.cpp/examples/embedding/embedding.cpp (main flow pattern)
//   + llama.cpp/common/common.cpp (sampling, tokenization helpers)

mod gguf;
mod block;
mod avx2;
mod kernel;
mod tensor;
mod vec_ops;
mod sampler;
mod tokenizer;
mod template;
mod cache;
mod models;
#[cfg(target_os = "macos")]
mod metal;
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
    let raw_args: Vec<String> = std::env::args().collect();
    let meta_flag = raw_args.iter().any(|a| a == "--meta");
    let args: Vec<String> = raw_args.into_iter().filter(|a| a != "--meta").collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} <model> [prompt]              — run inference", args[0]);
        eprintln!("  {} --meta <model> [prompt]       — run with GGUF metadata dump", args[0]);
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
    if meta_flag {
        dump_gguf_metadata(&ctx);
    } else {
        println!("GGUF: {} KV, {} tensors", ctx.kv.len(), ctx.info.len());
    }

    // === MPS GPU backend ===
    #[cfg(target_os = "macos")]
    metal::MpsState::init();

    // === Load model (dispatches on general.architecture) ===
    let model = models::load_model(&ctx, &data).expect("load model");
    if meta_flag {
        dump_key_tensors(&ctx);
    } else {
        println!("Model loaded.");
    }

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
    id == special.eos || Some(id) == special.im_end
}

fn dump_array<T: std::fmt::Debug>(key: &str, label: &str, items: &[T]) {
    const SHOW_PREFIX: usize = 5;
    const SHOW_SUFFIX: usize = 3;
    if items.len() <= SHOW_PREFIX + SHOW_SUFFIX {
        eprintln!("  {} (arr:{}) = {:?}", key, label, items);
    } else {
        eprint!("  {} (arr:{}) = [", key, label);
        for i in 0..SHOW_PREFIX {
            if i > 0 { eprint!(", "); }
            eprint!("{:?}", items[i]);
        }
        eprint!(", ..., ");
        for i in items.len() - SHOW_SUFFIX..items.len() {
            if i > items.len() - SHOW_SUFFIX { eprint!(", "); }
            eprint!("{:?}", items[i]);
        }
        eprintln!("]");
    }
}

fn dump_gguf_metadata(ctx: &gguf::GgufContext) {
    use gguf::GgufType;
    eprintln!("\n=== GGUF Metadata ===");
    for kv in &ctx.kv {
        let key = kv.get_key();
        if kv.is_array {
            let ne = kv.get_ne();
            match kv.get_type() {
                GgufType::String => {
                    let items: Vec<&str> = (0..ne).map(|i| kv.get_val_str(i)).collect();
                    dump_array(key, "str", &items);
                }
                GgufType::Int32 => {
                    let items: Vec<i32> = (0..ne).map(|i| kv.get_val_i32(i)).collect();
                    dump_array(key, "i32", &items);
                }
                GgufType::Uint32 => {
                    let items: Vec<u32> = (0..ne).map(|i| kv.get_val_u32(i)).collect();
                    dump_array(key, "u32", &items);
                }
                GgufType::Float32 => {
                    let items: Vec<f32> = (0..ne).map(|i| kv.get_val_f32(i)).collect();
                    dump_array(key, "f32", &items);
                }
                GgufType::Int64 => {
                    let items: Vec<i64> = (0..ne).map(|i| kv.get_val_i64(i)).collect();
                    dump_array(key, "i64", &items);
                }
                GgufType::Uint64 => {
                    let items: Vec<u64> = (0..ne).map(|i| kv.get_val_u64(i)).collect();
                    dump_array(key, "u64", &items);
                }
                GgufType::Float64 => {
                    let items: Vec<f64> = (0..ne).map(|i| kv.get_val_f64(i)).collect();
                    dump_array(key, "f64", &items);
                }
                t => eprintln!("  {} (arr:{:?}) = <{} elements>", key, t, ne),
            }
        } else {
            match kv.get_type() {
                GgufType::String => eprintln!("  {} = \"{}\"", key, kv.get_val_str(0)),
                GgufType::Bool => eprintln!("  {} = {}", key, kv.get_val_bool(0)),
                GgufType::Int32 => eprintln!("  {} = {}", key, kv.get_val_i32(0)),
                GgufType::Uint32 => eprintln!("  {} = {}", key, kv.get_val_u32(0)),
                GgufType::Int64 => eprintln!("  {} = {}", key, kv.get_val_i64(0)),
                GgufType::Uint64 => eprintln!("  {} = {}", key, kv.get_val_u64(0)),
                GgufType::Float32 => eprintln!("  {} = {}", key, kv.get_val_f32(0)),
                GgufType::Float64 => eprintln!("  {} = {}", key, kv.get_val_f64(0)),
                t => eprintln!("  {} ({:?})", key, t),
            }
        }
    }
    eprintln!("=== Metadata End ===");
}

fn dump_key_tensors(ctx: &gguf::GgufContext) {
    let key_names = [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q.bias",
        "blk.0.attn_k.weight",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.weight",
        "blk.0.attn_v.bias",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_norm.weight",
    ];
    eprintln!("--- Key Tensors ---");
    for name in &key_names {
        if let Some(ti) = ctx.info.iter().find(|t| t.name == *name) {
            let dims: Vec<String> = {
                let mut d: Vec<String> = ti.ne.iter().filter(|&&v| v > 0).map(|v| v.to_string()).collect();
                if d.is_empty() { d.push("1".into()); }
                d
            };
            eprintln!("  {:<50} {}  [{}]", ti.name, ti.type_.type_name(), dims.join(","));
        }
    }
    eprintln!("--------");
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
