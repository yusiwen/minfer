// End-to-End Inference Engine

mod gguf;
mod block;
mod avx2;
mod kernel;
mod tensor;
mod vec_ops;
mod graph;
mod sampler;
mod tokenizer;
mod template;
mod conversation;
mod cache;
mod models;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(feature = "cuda")]
mod cuda;
mod download;
mod dump;
mod server;

use std::time::Instant;
use rand::SeedableRng;

/// Conversation-mode color behavior (CLI-CONVERSATION-PLAN.md §5.6).
#[derive(Clone, Copy, PartialEq)]
enum ColorMode { On, Off, Auto }

impl ColorMode {
    fn enabled(self) -> bool {
        use std::io::IsTerminal;
        match self {
            ColorMode::On => true,
            ColorMode::Off => false,
            ColorMode::Auto => std::io::stdout().is_terminal() && std::io::stderr().is_terminal(),
        }
    }
}

struct GenParams {
    n_predict: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    seed: u64,
    n_ctx: usize,
    stop_strings: Vec<String>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            n_predict: 512,
            temp: 0.8,          // llama.cpp default (sampling, not greedy)
            top_k: 40,
            top_p: 0.95,        // llama.cpp default
            repeat_penalty: 1.1, // 1.0 = disabled; mild penalty reduces repetition
            frequency_penalty: 0.0, // llama.cpp default (0.0 = disabled)
            presence_penalty: 0.0,  // llama.cpp default (0.0 = disabled)
            seed: 42,
            n_ctx: 4096,
            stop_strings: Vec::new(),
        }
    }
}

fn print_usage(prog: &str) {
    eprintln!("minfer — a minimal local LLM inference engine");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  {prog} <model> [prompt] [OPTIONS]");
    eprintln!("  {prog} <model> --server [--port N] [--n-ctx N] [--n-slots N]");
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
    eprintln!("  --cnv, --conversation   multi-turn conversation mode (interactive REPL)");
    eprintln!("  -st, --single-turn   conversation: run one turn, then exit");
    eprintln!("  --system <STR>       conversation: system prompt");
    eprintln!("  -mli, --multiline-input   conversation: submit input on an empty line");
    eprintln!("  --color [on|off|auto]     conversation: color output (default auto = tty)");
    eprintln!("  --session <FILE>          conversation: save/load conversation history (JSON)");
    eprintln!("  --temp <T>           sampling temperature (default 0.8; 0 = greedy)");
    eprintln!("  --greedy             greedy decoding (--temp 0)");
    eprintln!("  --top-k <K>          top-K sampling (default 40)");
    eprintln!("  --top-p <P>          top-P nucleus sampling (default 0.95)");
    eprintln!("  --repeat-penalty <N> penalize repeated tokens (default 1.1; 1.0 = off)");
    eprintln!("  --frequency-penalty <N> penalize by token count in the window (default 0.0)");
    eprintln!("  --presence-penalty <N>  penalize any token present in the window (default 0.0)");
    eprintln!("  --stop <STR>         stop generation at this string (repeatable)");
    eprintln!("  -n, --n-predict <N>  max tokens to generate (default 512)");
    eprintln!("  --seed <N>           RNG seed for sampling (default 42)");
    eprintln!("  --server             run as an OpenAI-compatible HTTP server");
    eprintln!("  --port <N>           server port (default 8080)");
    eprintln!("  --n-ctx <N>          server total context (default 4096; divided among slots)");
    eprintln!("  --n-slots <N>        server slot count (default 1)");
    eprintln!("  -h, --help           show this help");
    eprintln!("  -V, --version        print version and exit");
}

fn main() {
    let raw_args: Vec<String> = std::env::args().collect();
    let prog = raw_args[0].clone();

    // Parse flags + positional args. Sampling flags map to GenParams.
    let mut params = GenParams::default();
    let mut meta_flag = false;
    let mut no_template = false;
    // Multi-turn conversation mode (CLI-CONVERSATION-PLAN.md, Phase 2).
    let mut conv_mode = false;
    let mut single_turn = false;
    let mut system_prompt: Option<String> = None;
    let mut multiline_input = false;
    let mut color_mode = ColorMode::Auto;
    let mut session_file: Option<String> = None;
    // `--graph` is accepted for compatibility; the graph path is now the
    // default forward (Phase 6).
    let _graph_mode = false;
    let mut dump_graph: Option<String> = None;
    // OpenAI-compatible HTTP server mode (OPENAI-CHAT-API-PLAN.md).
    let mut server_mode = false;
    let mut server_port: u16 = 8080;
    let mut server_n_ctx: usize = params.n_ctx;
    let mut server_n_slots: usize = 1;
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
            "-V" | "--version" => {
                // Value injected at build time (build.rs): "v<tag>(<short sha>)"
                // for release builds, "v<Cargo pkg version>" otherwise.
                println!("minfer {}", env!("MINFER_VERSION"));
                std::process::exit(0);
            }
            "--server" => { server_mode = true; i += 1; }
            "--port" => {
                if let Some(v) = next_val(a) {
                    server_port = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --port '{v}'")); 8080 });
                }
                i += 2;
            }
            "--n-ctx" => {
                if let Some(v) = next_val(a) {
                    server_n_ctx = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --n-ctx '{v}'")); 4096 });
                    params.n_ctx = server_n_ctx;
                }
                i += 2;
            }
            "--n-slots" => {
                if let Some(v) = next_val(a) {
                    server_n_slots = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --n-slots '{v}'")); 1 });
                }
                i += 2;
            }
            "--meta" => { meta_flag = true; i += 1; }
            "--no-template" => { no_template = true; i += 1; }
            "--cnv" | "--conversation" => { conv_mode = true; i += 1; }
            "-st" | "--single-turn" => { single_turn = true; i += 1; }
            "--system" => {
                if let Some(v) = next_val(a) {
                    system_prompt = Some(v);
                }
                i += 2;
            }
            "-mli" | "--multiline-input" => { multiline_input = true; i += 1; }
            "--session" => {
                if let Some(v) = next_val(a) {
                    session_file = Some(v);
                }
                i += 2;
            }
            "--color" => {
                if let Some(v) = next_val(a) {
                    color_mode = match v.as_str() {
                        "on" => ColorMode::On,
                        "off" => ColorMode::Off,
                        "auto" => ColorMode::Auto,
                        _ => { parse_err = Some(format!("invalid --color '{v}' (on|off|auto)")); ColorMode::Auto }
                    };
                }
                i += 2;
            }
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
            "--frequency-penalty" => {
                if let Some(v) = next_val(a) {
                    params.frequency_penalty = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --frequency-penalty '{v}'")); 0.0 });
                }
                i += 2;
            }
            "--presence-penalty" => {
                if let Some(v) = next_val(a) {
                    params.presence_penalty = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid --presence-penalty '{v}'")); 0.0 });
                }
                i += 2;
            }
            "--stop" => {
                if let Some(v) = next_val(a) {
                    params.stop_strings.push(v);
                }
                i += 2;
            }
            "-n" | "--n-predict" => {
                if let Some(v) = next_val(a) {
                    params.n_predict = v.parse().unwrap_or_else(|_| { parse_err = Some(format!("invalid -n '{v}'")); 0 });
                }
                i += 2;
            }
            "--graph" => {
                i += 1; // accepted for compat; graph path is the default
            }
            "--dump-graph" => {
                if let Some(v) = next_val("--dump-graph") {
                    dump_graph = Some(v);
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

    // Conversation mode needs the chat template (ChatML fallback included);
    // --no-template would leave the history unformatted.
    if conv_mode && no_template {
        eprintln!("Error: --cnv conflicts with --no-template (conversation needs the chat template)");
        print_usage(&prog);
        std::process::exit(1);
    }

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
    // Conversation mode: the positional prompt is the FIRST user turn; stdin
    // is read interactively by the loop (never consume it here).
    let first_prompt = if conv_mode && positional.len() > 1 {
        Some(positional[1..].join(" "))
    } else {
        None
    };
    let prompt = if positional.len() > 1 {
        positional[1..].join(" ")
    } else if conv_mode {
        String::new()
    } else {
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
    let mut kv_cache = cache::KVCache::new(n_layer, n_kv_embd, params.n_ctx);

    // Pre-allocate GPU KV cache (avoids O(n²) incremental growth during generation)
    #[cfg(feature = "cuda")]
    if let Some(cuda) = cuda::CudaState::get() {
        cuda.init_kv_cache(n_layer, params.n_ctx, n_head_kv * n_embd_head);
    }

    // === Tokenizer ===
    let tokenizer = tokenizer::Tokenizer::load(&ctx);
    println!("Vocabulary: {} tokens", tokenizer.vocab_size());

    // === OpenAI-compatible HTTP server mode ===
    if server_mode {
        if server_n_slots == 0 {
            eprintln!("Error: --n-slots must be >= 1");
            std::process::exit(1);
        }
        if server_n_ctx == 0 {
            eprintln!("Error: --n-ctx must be >= 1");
            std::process::exit(1);
        }
        eprintln!("Starting OpenAI-compatible server (model={}, n_ctx={}, n_slots={})",
            model_path, server_n_ctx, server_n_slots);
        server::run(model, tokenizer, &gguf_model, server_port, server_n_ctx, server_n_slots);
        return;
    }

    // === Multi-turn conversation mode (--cnv) ===
    if conv_mode {
        let code = run_conversation(
            model,
            &tokenizer,
            &gguf_model.parts[0].data,
            &params,
            system_prompt,
            single_turn,
            multiline_input,
            color_mode,
            session_file,
            first_prompt,
        );
        std::process::exit(code);
    }

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
    // forward() computes logits for only the LAST n_out tokens (n_out=1 here:
    // single sequence, only the final token is sampled). llama.cpp does the same
    // via ggml_get_rows(inp_out_ids) at the last layer, shrinking the lm_head
    // to n_outputs rows — saves the full-nt output GEMM + logits download.
    let logits = model.forward(&input_ids, &positions, &mut kv_cache, 1);
    let last_logits: Vec<f32> = logits;

    let prefill_time = infer_start.elapsed();
    println!("Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        input_ids.len(), prefill_time.as_secs_f64(),
        input_ids.len() as f64 / prefill_time.as_secs_f64());

    // === Graph DOT export (--dump-graph <path>) ===
    if let Some(path) = &dump_graph {
        use crate::graph::params::{CParams, GraphParams, GraphType};
        let gparams = GraphParams {
            n_tokens: input_ids.len(),
            n_seqs: 1,
            n_out: 1,
            gtype: if input_ids.len() == 1 { GraphType::Decode } else { GraphType::Prefill },
            cparams: CParams { n_ctx: params.n_ctx, n_batch: input_ids.len(), flash_attn: false, gpu: false, fuse_qkv: false },
            weights_version: 1,
        };
        let g = model.build_graph(&gparams);
        let mut f = std::fs::File::create(path).expect("create dot file");
        g.dump_dot(&mut f).expect("write dot");
        println!("Graph DOT exported to {path} ({} nodes)", g.n_nodes());
        std::process::exit(0);
    }

    // === Generate ===
    let mut logits = last_logits;
    let gen_start = Instant::now();   // pure-decode start (llama "Generation" caliber)
    let mut generated: Vec<u32> = Vec::new();
    let special = model.special_tokens();
    let mut current_pos = input_ids.len();

    // Seeded RNG for reproducible sampling; recent-token window for the
    // penalties (llama.cpp repeat_last_n default = 64), seeded with the prompt
    // tail so the first generated tokens are penalized too.
    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    const REPEAT_LAST_N: usize = 64;
    let mut prev_tokens = sampler::recent_window(&input_ids, REPEAT_LAST_N);

    // Stop strings (byte-level, llama.cpp antiprompt): `full` accumulates ALL
    // generated bytes and `emitted` tracks what was already written, so a stop
    // string split across tokens is caught even if its earlier tokens were
    // already flushed (they stay in the terminal, like llama.cpp). Output is
    // byte-wise (never lossy), so split multi-byte chars pass through intact.
    let stop_bytes: Vec<Vec<u8>> = params.stop_strings.iter().map(|s| s.as_bytes().to_vec()).collect();
    let stop_refs: Vec<&[u8]> = stop_bytes.iter().map(|v| v.as_slice()).collect();
    let mut full: Vec<u8> = Vec::new();
    let mut emitted: usize = 0;

    // MINFER_TIMING=1: decompose the per-token wall-clock into CPU sampling vs
    // forward() (CPU encode + GPU exec + logits download). Debug tool only.
    let timing = std::env::var("MINFER_TIMING").map_or(false, |v| v == "1");
    let (mut t_samp, mut t_fwd, mut n_tok) = (0.0f64, 0.0f64, 0usize);
    // t0/t1 are (re)assigned at the top of each loop iteration before use.
    let (mut t0, mut t1);

    while generated.len() < params.n_predict {
        t0 = std::time::Instant::now();
        let sampled = sampler::sample_with_penalties(
            &mut logits, params.temp, params.top_k, params.top_p,
            params.repeat_penalty, params.frequency_penalty, params.presence_penalty,
            &prev_tokens, &mut rng,
        );
        if timing { t_samp += t0.elapsed().as_secs_f64(); }

        if is_stop_token(sampled.token_id, &special) { break; }
        generated.push(sampled.token_id);
        prev_tokens.push(sampled.token_id);
        if prev_tokens.len() > REPEAT_LAST_N {
            prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
        }

        // Stop-string detection on the FULL byte stream before emitting.
        full.extend_from_slice(&tokenizer.decode_bytes(&[sampled.token_id]));
        if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
            full.truncate(cut);
            if cut > emitted {
                std::io::Write::write_all(&mut std::io::stdout(), &full[emitted..]).unwrap_or(());
                std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
                emitted = full.len();
            }
            break;
        }
        if emitted < full.len() {
            std::io::Write::write_all(&mut std::io::stdout(), &full[emitted..]).unwrap_or(());
            std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
            emitted = full.len();
        }

        // forward() returns n_out*nv logits (n_out=1 for single-token decode,
        // exactly n_vocab), so move the Vec in place instead of copying 607 KB/token.
        t1 = std::time::Instant::now();
        logits = model.forward(&[sampled.token_id], &[current_pos], &mut kv_cache, 1);
        if timing { t_fwd += t1.elapsed().as_secs_f64(); n_tok += 1; }
        current_pos += 1;
    }
    // Final flush of any bytes not yet written.
    if emitted < full.len() {
        std::io::Write::write_all(&mut std::io::stdout(), &full[emitted..]).unwrap_or(());
        std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
    }

    if timing {
        eprintln!("\n[MINFER_TIMING] over {n_tok} tokens: sample {:>6.2} ms/tok ({:>4.1}%), forward {:>6.2} ms/tok ({:>4.1}%)",
            t_samp / n_tok as f64 * 1e3, 100.0 * t_samp / (t_samp + t_fwd),
            t_fwd / n_tok as f64 * 1e3, 100.0 * t_fwd / (t_samp + t_fwd));
    }

    println!();
    let gen_time = gen_start.elapsed();
    let total_time = infer_start.elapsed();
    println!("\n---");
    // Pure-decode rate (generated tokens / decode time) — matches llama.cpp's
    // "Generation:" caliber. The "Total:" line below keeps the previous blended
    // caliber (prompt+generated / prefill+decode) for comparison.
    println!("Generated: {} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(), gen_time.as_secs_f64(),
        generated.len() as f64 / gen_time.as_secs_f64());
    println!("Total:     {} tokens in {:.2}s ({:.1} tok/s)",
        input_ids.len() + generated.len(), total_time.as_secs_f64(),
        (input_ids.len() + generated.len()) as f64 / total_time.as_secs_f64());
}

fn is_stop_token(id: u32, special: &models::SpecialTokens) -> bool {
    id == special.eos || Some(id) == special.im_end
}

// === Multi-turn conversation mode (CLI-CONVERSATION-PLAN.md Phase 2) ===

/// 读一行用户输入。多行模式（-mli）下累积直到空行提交；EOF 时提交已输入内容或返回 None。
fn read_user_input(multiline: bool) -> Option<String> {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut buf = String::new();
    loop {
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).ok()? == 0 {
            // EOF（Ctrl+D）
            return if buf.is_empty() { None } else { Some(buf) };
        }
        let line = line.trim_end_matches(['\n', '\r']).to_string();
        if !multiline {
            return Some(line);
        }
        if line.is_empty() {
            return Some(buf); // 空行提交
        }
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(&line);
    }
}

/// 交互式多轮对话循环（--cnv）。
/// 追加式 KV + 增量模板渲染：`Conversation` 状态机 + `GraphEngine` 推理引擎。
fn run_conversation(
    model: Box<dyn models::ModelDef>,
    tokenizer: &tokenizer::Tokenizer,
    gguf_data: &[u8],
    params: &GenParams,
    system_prompt: Option<String>,
    single_turn: bool,
    multiline: bool,
    color_mode: ColorMode,
    session_file: Option<String>,
    first_prompt: Option<String>,
) -> i32 {
    use std::io::Write;
    use crate::conversation::Engine as _; // reset_cache / forward trait methods

    let template = get_chat_template(gguf_data);
    let special = model.special_tokens();
    let bos_text = tokenizer
        .id_to_token
        .get(tokenizer.bos_token as usize)
        .cloned()
        .unwrap_or_default();
    let eog = {
        let mut v = vec![special.eos];
        if let Some(im) = special.im_end {
            v.push(im);
        }
        v
    };
    let eot = special.im_end.unwrap_or(special.eos);

    let spec = conversation::ConversationSpec {
        template,
        bos_text,
        eog,
        eot,
        seed: params.seed,
        n_ctx: params.n_ctx,
        system_prompt,
    };
    let mut conv = conversation::Conversation::new(spec);
    let mut engine = conversation::GraphEngine::new(&*model, params.n_ctx);

    // --session 加载：历史 JSON → 全量重灌 KV（§5.8）。
    if let Some(path) = &session_file {
        if let Ok(text) = std::fs::read_to_string(path) {
            match conversation::Conversation::messages_from_json(&text) {
                Some(msgs) => {
                    conv.load_history(msgs, &*tokenizer, &mut engine);
                    eprintln!("[session] loaded {} message(s) from {path}", conv.messages.len());
                }
                None => eprintln!("[session] ignoring unreadable session file {path}"),
            }
        }
    }

    // --session 保存（退出时）。
    let save_session = |conv: &conversation::Conversation| {
        if let Some(path) = &session_file {
            let json = conv.messages_to_json();
            if std::fs::write(path, json).is_ok() {
                eprintln!("[session] saved {} message(s) to {path}", conv.messages.len());
            } else {
                eprintln!("[session] failed to save session to {path}");
            }
        }
    };
    let tp = conversation::TurnParams {
        n_predict: params.n_predict,
        temp: params.temp,
        top_k: params.top_k,
        top_p: params.top_p,
        repeat_penalty: params.repeat_penalty,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        stop_strings: params.stop_strings.clone(),
    };
    let color_on = color_mode.enabled();
    let mut turn = 0usize;

    let mut pending = first_prompt;
    loop {
        if pending.is_none() {
            // 提示符 + flush（管道化测试的前置条件，见文档 §8.3）。
            if color_on {
                print!("\x1b[1;36m");
            }
            print!("> ");
            if color_on {
                print!("\x1b[0m");
            }
            std::io::stdout().flush().unwrap_or(());
            let Some(input) = read_user_input(multiline) else { break };
            let input = input.trim().to_string();
            if input.is_empty() {
                continue;
            }
            pending = Some(input);
        }
        let input = pending.take().unwrap();

        // Slash 命令。
        if input.starts_with('/') {
            let cmd = input.split_whitespace().next().unwrap_or("");
            match cmd {
                "/exit" | "/quit" => break,
                "/help" => {
                    println!("\ncommands:");
                    println!("  /exit, /quit   exit (Ctrl+D / EOF also exits)");
                    println!("  /clear         clear the chat history");
                    println!("  /regen         regenerate the last assistant reply");
                    println!("  /help          this help");
                    continue;
                }
                "/clear" => {
                    conv.clear();
                    engine.reset_cache();
                    println!("\n[history cleared]");
                    continue;
                }
                "/regen" => {
                    if color_on {
                        print!("\x1b[32m");
                        std::io::stdout().flush().unwrap_or(());
                    }
                    let t0 = Instant::now();
                    let r = conv.regen_turn(&*tokenizer, &tp, &mut engine, &mut |b| {
                        let _ = std::io::stdout().write_all(b);
                        let _ = std::io::stdout().flush();
                    });
                    if color_on {
                        print!("\x1b[0m");
                        std::io::stdout().flush().unwrap_or(());
                    }
                    match r {
                        Ok(out) => {
                            turn += 1;
                            println!();
                            eprintln!("[turn {turn}] regen: prefill {} tokens, generated {} tokens in {:.2}s",
                                out.prefill_tokens, out.tokens_generated, t0.elapsed().as_secs_f64());
                        }
                        Err(e) => eprintln!("\n[regen failed: {e}]"),
                    }
                    if single_turn {
                        break;
                    }
                    continue;
                }
                _ => {
                    eprintln!("unknown command '{cmd}' — try /help");
                    continue;
                }
            }
        }

        // 普通用户回合。
        if color_on {
            print!("\x1b[32m");
            std::io::stdout().flush().unwrap_or(());
        }
        let t0 = Instant::now();
        let result = conv.user_turn(&input, &*tokenizer, &tp, &mut engine, &mut |b| {
            let _ = std::io::stdout().write_all(b);
            let _ = std::io::stdout().flush();
        });
        if color_on {
            print!("\x1b[0m");
            std::io::stdout().flush().unwrap_or(());
        }
        match result {
            Ok(out) => {
                turn += 1;
                println!();
                eprintln!("[turn {turn}] prefill {} tokens, generated {} tokens in {:.2}s",
                    out.prefill_tokens, out.tokens_generated, t0.elapsed().as_secs_f64());
                if out.hit_n_predict {
                    eprintln!("[turn {turn}] stopped by n_predict / context limit");
                }
                if out.dropped_turns > 0 {
                    eprintln!("[turn {turn}] <<context full: dropped oldest {n} turn(s)>>",
                        n = out.dropped_turns);
                }
            }
            Err(e) => eprintln!("\n[error] {e}"),
        }
        if single_turn {
            break;
        }
    }
    save_session(&conv);
    eprintln!("\n---\nconversation ended after {turn} turn(s)");
    0
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
