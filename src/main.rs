// End-to-End Inference Engine

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
#[cfg(feature = "cuda")]
mod cuda;
mod download;
mod dump;

use std::time::Instant;
use rand::SeedableRng;

struct GenParams {
    n_predict: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: u64,
    n_ctx: usize,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            n_predict: 512,
            temp: 0.8,          // llama.cpp default (sampling, not greedy)
            top_k: 40,
            top_p: 0.95,        // llama.cpp default
            repeat_penalty: 1.1, // 1.0 = disabled; mild penalty reduces repetition
            seed: 42,
            n_ctx: 4096,
        }
    }
}

fn print_usage(prog: &str) {
    eprintln!("minfer — a minimal local LLM inference engine");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  {prog} <model> [prompt] [OPTIONS]");
    eprintln!("  {prog} info <model>");
    eprintln!("  {prog} download hf <repo> [quant]");
    eprintln!("  {prog} download ollama <model>[:tag]");
    eprintln!("  {prog} download <hf|ollama>:<name>[:variant]");
    eprintln!("  {prog} list");
    eprintln!();
    eprintln!("MODEL — <model> may be any of:");
    eprintln!("  · a local file path     /abs/model.gguf   ./model.gguf   ~/model.gguf");
    eprintln!("  · a Hugging Face repo   hf:<repo>[:<quant>]              (auto-download)");
    eprintln!("                         <quant> e.g. Q4_0, q4_k_m (case-insensitive;");
    eprintln!("                         single file or split auto-detected)");
    eprintln!("  · an Ollama model       ollama:<model>[:tag]            (pull)");
    eprintln!("  · a cached model name   <filename>   resolved from ~/.cache/minfer/models");
    eprintln!("                         (see `{prog} list`)");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --meta               print GGUF metadata and key tensors");
    eprintln!("  --no-template        use the raw prompt without the chat template");
    eprintln!("  --temp <T>           sampling temperature (default 0.8; 0 = greedy)");
    eprintln!("  --greedy             greedy decoding (--temp 0)");
    eprintln!("  --top-k <K>          top-K sampling (default 40)");
    eprintln!("  --top-p <P>          top-P nucleus sampling (default 0.95)");
    eprintln!("  --repeat-penalty <N> penalize repeated tokens (default 1.1; 1.0 = off)");
    eprintln!("  -n, --n-predict <N>  max tokens to generate (default 512)");
    eprintln!("  --seed <N>           RNG seed for sampling (default 42)");
    eprintln!("  -h, --help           show this help");
}

fn main() {
    let raw_args: Vec<String> = std::env::args().collect();
    let prog = raw_args[0].clone();

    // Parse flags + positional args. Sampling flags map to GenParams.
    let mut params = GenParams::default();
    let mut meta_flag = false;
    let mut no_template = false;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 1;
    let mut parse_err: Option<String> = None;
    while i < raw_args.len() {
        let a = raw_args[i].as_str();
        let mut next_val = |name: &str| -> Option<String> {
            if i + 1 < raw_args.len() {
                Some(raw_args[i + 1].clone())
            } else {
                parse_err = Some(format!("missing value for {name}"));
                None
            }
        };
        match a {
            "-h" | "--help" => {
                print_usage(&prog);
                std::process::exit(0);
            }
            "--meta" => { meta_flag = true; i += 1; }
            "--no-template" => { no_template = true; i += 1; }
            "--greedy" => { params.temp = 0.0; i += 1; }
            "--temp" | "-t" => {
                if let Some(v) = next_val(a) {
                    params.temp = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --temp '{v}'")); 0.0 });
                }
                i += 2;
            }
            "--top-k" => {
                if let Some(v) = next_val(a) {
                    params.top_k = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --top-k '{v}'")); 0 });
                }
                i += 2;
            }
            "--top-p" => {
                if let Some(v) = next_val(a) {
                    params.top_p = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --top-p '{v}'")); 0.0 });
                }
                i += 2;
            }
            "--repeat-penalty" => {
                if let Some(v) = next_val(a) {
                    params.repeat_penalty = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --repeat-penalty '{v}'")); 1.0 });
                }
                i += 2;
            }
            "-n" | "--n-predict" => {
                if let Some(v) = next_val(a) {
                    params.n_predict = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid -n '{v}'")); 0 });
                }
                i += 2;
            }
            "--seed" => {
                if let Some(v) = next_val(a) {
                    params.seed = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --seed '{v}'")); 0 });
                }
                i += 2;
            }
            _ => {
                if a.starts_with('-') && a.len() > 1 {
                    // Unknown option — reject instead of treating as model path.
                    print_usage(&prog);
                    eprintln!("Error: unknown option '{a}'");
                    std::process::exit(1);
                }
                positional.push(raw_args[i].clone());
                i += 1;
            }
        }
    }
    if let Some(e) = parse_err {
        eprintln!("Error: {e}");
        print_usage(&prog);
        std::process::exit(1);
    }

    if positional.is_empty() {
        print_usage(&prog);
        std::process::exit(1);
    }

    // === Subcommands ===
    match positional[0].as_str() {
        "download" => {
            if positional.len() < 2 {
                eprintln!("Usage: {prog} download hf <repo> [quant] | ollama <model>[:tag]");
                eprintln!("       {prog} download hf:<repo>[:quant] | ollama:<model>[:tag]");
                std::process::exit(1);
            }
            let arg = &positional[1];
            let uri = if arg == "hf" || arg == "ollama" {
                // space form: download hf <repo> [quant]  /  download ollama <model>[:tag]
                if positional.len() < 3 {
                    eprintln!("Usage: {prog} download {} <repo-or-model> [variant]", arg);
                    std::process::exit(1);
                }
                format!("{}:{}", arg, positional[2..].join(":"))
            } else if arg.starts_with("hf:") || arg.starts_with("ollama:") {
                // URI form: download hf:<repo>[:quant]  /  download ollama:<model>[:tag]
                if positional.len() != 2 {
                    eprintln!("Usage: {prog} download <hf|ollama>:<name>[:variant]");
                    std::process::exit(1);
                }
                arg.clone()
            } else {
                eprintln!("Unknown download source '{}'. Use 'hf' or 'ollama'.", arg);
                std::process::exit(1);
            };
            match download::resolve(&uri) {
                Ok(p) => println!("Model downloaded: {}", p.display()),
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
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
        "info" => {
            if positional.len() < 2 {
                eprintln!("Usage: {prog} info <model>");
                std::process::exit(1);
            }
            // Resolve paths, hf:/ollama: URIs, and cached model names.
            let model_path = match download::resolve(&positional[1]) {
                Ok(p) => { eprintln!("Model ready: {}", p.display()); p.to_string_lossy().to_string() }
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let gguf_model = gguf::load_gguf_model(std::path::Path::new(&model_path)).expect("parse GGUF");
            let ctx = &gguf_model.parts[0].ctx;
            dump_gguf_metadata(ctx);
            dump_key_tensors(ctx);
            return;
        }
        _ => {}  // fall through to model inference
    }

    let model_path = &positional[0];

    // Resolve paths, hf:/ollama: URIs, and cached model names.
    let is_uri = model_path.starts_with("hf:")
        || model_path.starts_with("ollama:")
        || (!model_path.starts_with('/') && !model_path.starts_with('.') && !model_path.starts_with('~'));
    let model_path = match download::resolve(model_path) {
        Ok(p) => {
            if is_uri {
                eprintln!("Model ready: {}", p.display());
            }
            p.to_string_lossy().to_string()
        }
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    let prompt = if positional.len() > 1 { positional[1..].join(" ") } else {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap_or(0);
        input.trim().to_string()
    };

    // === Load GGUF (single file or multi-part split) ===
    println!("Loading model: {} ...", model_path);
    let gguf_model = gguf::load_gguf_model(std::path::Path::new(&model_path)).expect("parse GGUF");
    let n_parts = gguf_model.parts.len();
    let total_bytes: usize = gguf_model.parts.iter().map(|p| p.data.len()).sum();
    println!("File: {} bytes ({:.1} MB) in {n_parts} part(s)", total_bytes, total_bytes as f64 / 1_048_576.0);

    let ctx = &gguf_model.parts[0].ctx;
    if meta_flag {
        dump_gguf_metadata(ctx);
    } else {
        println!("GGUF: {} KV, {} tensors", ctx.kv.len(), ctx.info.len());
    }

    // === GPU backends ===
    #[cfg(target_os = "macos")]
    metal::MpsState::init();
    #[cfg(feature = "cuda")]
    cuda::CudaState::init();

    // === Load model (dispatches on general.architecture) ===
    let model = models::load_model(&gguf_model).expect("load model");
    if meta_flag {
        dump_key_tensors(ctx);
    } else {
        println!("Model loaded.");
    }

    // === KV Cache ===
    let n_kv_embd = model.n_kv_embd();
    let n_layer = model.n_layer();
    let n_vocab = model.n_vocab();
    let mut kv_cache = cache::KVCache::new(n_layer, n_kv_embd, params.n_ctx);

    // Pre-allocate GPU KV cache (avoids O(n²) incremental growth during generation)
    #[cfg(feature = "cuda")]
    if let Some(cuda) = cuda::CudaState::get() {
        cuda.init_kv_cache(n_layer, params.n_ctx, n_head_kv * n_embd_head);
    }

    // === Tokenizer ===
    let tokenizer = tokenizer::Tokenizer::load(&ctx);
    println!("Vocabulary: {} tokens", tokenizer.vocab_size());

    // === Chat template (need tokenizer for bos_token text) ===
    let processed = if no_template {
        prompt.clone()
    } else if let Some(tmpl) = get_chat_template(&gguf_model.parts[0].data) {
        let bos_text = tokenizer.id_to_token.get(tokenizer.bos_token as usize)
            .map(|s| s.as_str())
            .unwrap_or("");
        template::render_template(&tmpl, &prompt, true, bos_text)
    } else {
        prompt.clone()
    };
    #[cfg(feature = "debug_dump")]
    crate::dump::maybe_dump_text("minfer_dump_prompt", &processed);
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

    // Seeded RNG for reproducible sampling; recent-token window for the
    // repetition penalty (llama.cpp repeat_last_n default = 64).
    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    const REPEAT_LAST_N: usize = 64;
    let mut prev_tokens: Vec<u32> = input_ids.iter().copied().rev().take(REPEAT_LAST_N).collect();
    prev_tokens.reverse();

    while generated.len() < params.n_predict {
        let sampled = sampler::sample(
            &mut logits, params.temp, params.top_k, params.top_p,
            params.repeat_penalty, &prev_tokens, &mut rng,
        );

        if is_stop_token(sampled.token_id, &special) { break; }
        generated.push(sampled.token_id);
        prev_tokens.push(sampled.token_id);
        if prev_tokens.len() > REPEAT_LAST_N {
            prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
        }
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
