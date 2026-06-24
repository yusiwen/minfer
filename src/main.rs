// Phase 9: End-to-End Inference Engine
// Translated from: llama.cpp/src/llama-context.cpp (llama_decode, context init)
//   + llama.cpp/examples/embedding/embedding.cpp (main flow pattern)
//   + llama.cpp/common/common.cpp (sampling, tokenization helpers)
// Strict 1:1 translation of the inference flow — no extra code, no design changes

mod gguf;
mod block;
mod avx2;
mod tensor;
mod vec_ops;
mod model;
mod forward;
mod sampler;
mod tokenizer;
mod loader;
mod template;

use std::time::Instant;

/// Default generation parameters matching llama.cpp defaults
struct GenParams {
    n_predict: usize,         // max tokens to generate (-1 = infinite)
    temp: f32,                // temperature (0 = greedy)
    top_k: usize,             // top-K sampling (0 = disabled)
    top_p: f32,               // top-P / nucleus sampling (1.0 = disabled)
    seed: u64,                // random seed
    n_ctx: usize,             // context size
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            n_predict: 512,
            temp: 0.0,
            top_k: 40,
            top_p: 0.9,
            seed: 42,
            n_ctx: 4096,
        }
    }
}

fn main() {
    // === Command-line argument parsing (translated from common_params in common/common.cpp) ===
    // llama.cpp: examples/common/common.h / common_params
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [prompt]", args[0]);
        eprintln!("  If prompt is omitted, reads from stdin");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap_or(0);
        input.trim().to_string()
    };

    let params = GenParams::default();

    // === Model loading (translated from llama_load_model_from_file in llama.cpp) ===
    // llama.cpp: src/llama-model-loader.cpp / llama_model_loader
    println!("Loading model: {} ...", model_path);
    let load_start = Instant::now();

    let data = {
        use std::io::Read;
        let mut file = std::fs::File::open(model_path).expect("Failed to open model file");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).expect("Failed to read model file");
        buf
    };

    println!("File: {} bytes ({:.1} MB)", data.len(), data.len() as f64 / 1_048_576.0);

    let model = loader::load_model(&data).expect("Failed to load model from GGUF");
    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}s", load_time.as_secs_f64());

    // === KV Cache creation (translated from llama_context init in llama.cpp) ===
    // llama.cpp: src/llama-context.cpp / llama_new_context_with_model
    // Allocates KV cache for up to n_ctx positions per layer
    let cache = forward::KVCache::new(&model.hparams);
    let n_embd = model.hparams.n_embd as usize;
    let n_vocab = model.hparams.n_vocab as usize;

    // === Tokenizer loading (translated from llama_tokenize in llama.cpp) ===
    // llama.cpp: src/llama-vocab.cpp / llama_tokenize
    // Load tokenizer from GGUF metadata (tokens, scores, types, merges)
    let ctx = gguf::GgufContext::init_from_data(&data)
        .expect("Failed to parse GGUF for tokenizer");
    let tokenizer = tokenizer::Tokenizer::load(&ctx);
    println!("Vocabulary: {} tokens", tokenizer.vocab_size());

    // === Tokenization (translated from common_batch_add / llama_tokenize in llama.cpp) ===
    // llama.cpp: src/llama-vocab.cpp / llama_tokenize
    // Encode the prompt into token IDs
    let processed_prompt = if let Some(template) = get_chat_template(&data) {
        template::render_template(&template, &prompt, true)
    } else {
        prompt.clone()
    };
    let input_ids = tokenizer.encode(&processed_prompt);
    if input_ids.is_empty() {
        eprintln!("Error: failed to tokenize prompt");
        std::process::exit(1);
    }
    println!("Prompt: {} tokens", input_ids.len());

    // === Prefill (translated from llama_decode for initial batch in llama.cpp) ===
    // llama.cpp: src/llama-context.cpp / llama_decode
    // Process all input tokens sequentially, populating KV cache
    let infer_start = Instant::now();
    let mut cache = cache;

    for (pos, &token_id) in input_ids.iter().enumerate() {
        // === Embedding lookup (translated from build_inp_embd in qwen2.cpp) ===
        // qwen2.cpp line 62: inpL = build_inp_embd(model.tok_embd);
        let embedded = forward::embed_tokens(
            &[token_id],
            model.tok_embd.as_ref().unwrap(),
            &model.hparams,
        );

        // === Forward pass (translated from llama_decode / build_arch_graph in qwen2.cpp) ===
        // qwen2.cpp lines 71-134: transformer layer loop
        let (_logits, _hidden) = forward::forward_decode(
            &embedded,
            pos,
            &model,
            &mut cache,
        );
        // During prefill, we process logits but don't sample until the last token
    }

    // === Final logits from prefill (last token) ===
    let last_pos = input_ids.len() - 1;
    let last_embedded = forward::embed_tokens(
        &[input_ids[last_pos]],
        model.tok_embd.as_ref().unwrap(),
        &model.hparams,
    );
    let (mut logits, _last_hidden) = forward::forward_decode(
        &last_embedded,
        last_pos,
        &model,
        &mut cache,
    );

    let prefill_time = infer_start.elapsed();
    println!("Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        input_ids.len(), prefill_time.as_secs_f64(),
        input_ids.len() as f64 / prefill_time.as_secs_f64());

    // === Generate loop (translated from main inference loop in llama.cpp) ===
    // llama.cpp: examples/main/main.cpp (the main generation loop)
    //   or examples/embedding/embedding.cpp (simpler batch pattern)
    let mut generated: Vec<u32> = Vec::new();
    let mut total_generated = 0usize;

    while total_generated < params.n_predict {
        let pos = input_ids.len() + total_generated;

        // === Sampling (translated from llama_sample_* in common/sampling.cpp) ===
        // llama.cpp: common/sampling.cpp / llama_sample_token
        let sampled = if params.temp < 1e-6 {
            sampler::sample_greedy(&logits)
        } else {
            // Temperature sampling with top-k and top-p (llama.cpp defaults)
            sampler::apply_top_k(&mut logits, params.top_k);
            sampler::apply_top_p(&mut logits, params.top_p);
            sampler::sample_temperature(&mut logits, params.temp)
        };

        if is_stop_token(sampled.token_id, &tokenizer.id_to_token) {
            break;
        }
        generated.push(sampled.token_id);
        print!("{}", tokenizer.decode(&[sampled.token_id]));
        std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());

        // === Decode next token (translated from llama_decode in llama.cpp) ===
        let embedded = forward::embed_tokens(
            &[sampled.token_id],
            model.tok_embd.as_ref().unwrap(),
            &model.hparams,
        );
        (logits, _) = forward::forward_decode(&embedded, pos, &model, &mut cache);
        total_generated += 1;
    }

    println!();
    let total_time = infer_start.elapsed();
    let total_tokens = input_ids.len() + total_generated;
    println!("\n---");
    println!("Generated: {} tokens in {:.2}s ({:.1} tok/s)",
        total_generated, total_time.as_secs_f64(),
        total_tokens as f64 / total_time.as_secs_f64());
}

/// Check if a token ID is a stop token (EOS)
fn is_stop_token(id: u32, tokens: &[String]) -> bool {
    // Common EOS tokens in Qwen2: <|endoftext|>, <|im_end|>
    if id as usize >= tokens.len() { return false; }
    let t = &tokens[id as usize];
    t == "<|endoftext|>" || t == "<|im_end|>" || t == "</s>" || t == "<eos>"
        || id == 0 || id == 2
}

/// Extract chat template from GGUF metadata
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
