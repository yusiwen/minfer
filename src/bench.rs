//! `minfer bench` — llama-bench-style performance benchmark (focused subset).
//!
//! Runs up to two tests over one loaded model (llama-bench's defaults):
//! - `pp<P>`: prefill-only — ingest P prompt tokens in a single forward,
//!   generate nothing. Measures prompt-processing throughput.
//! - `tg<T>`: prefill P context tokens (untimed setup), then time the
//!   autoregressive decode of exactly T tokens.
//!
//! Timing reuses the engine's own single-shot semantics (main.rs): prefill
//! t/s = prompt tokens / prefill wall time; decode t/s = generated tokens /
//! generation wall time (sampling included), i.e. the same caliber as the
//! "Prefill: ... (tok/s)" / "Generated: ... (tok/s)" lines.
//!
//! Per-rep fresh context without a reload: every rep re-runs the forward from
//! position 0 into the same [`GraphCache`] (no model reload, no KV zeroing).
//! This is sound because **KV positions are data, not structure** (graph rules
//! §1): attention visibility is bounded by the query token's own position
//! (`vl = pos[t]+1` in every backend), so a rep only ever reads the KV slots
//! its own stores wrote this rep — stale slots are never read. Reps are
//! independent by construction, and the graph is reused by params after the
//! warmup rep (no rebuild inside the timed region; the decode-graph build the
//! first decode step pays mirrors main.rs's single-shot timing).
//!
//! Sampling is greedy (llama-bench's deterministic caliber): the token stream
//! is identical across reps, so reps measure the same work.

use crate::graph::cache::GraphCache;
use crate::models::ModelDef;
use std::time::Instant;

pub fn print_usage(prog: &str) {
    eprintln!("minfer bench — llama-bench-style performance test");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  {prog} bench [OPTIONS] <model>");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -p <N>       prompt tokens for the pp test (default 512; 0 = skip)");
    eprintln!("  -n <N>       tokens to generate for the tg test (default 128; 0 = skip)");
    eprintln!("  -r <N>       measured reps per test, plus 1 untimed warmup rep each (default 3;");
    eprintln!("               reports mean ± stddev over the reps)");
    eprintln!("  -o <FMT>     output format: md | csv | json (default md)");
    eprintln!("  --n-ctx <N>  KV cache size for the run (default pp+tg+16; clamped to the");
    eprintln!("               model's context length, never below what the tests need)");
    eprintln!("  -t, --threads <N>  CPU worker threads (GPU backends are unaffected)");
    eprintln!("  -h, --help   show this help");
    eprintln!();
    eprintln!("The model may be a path, a hf:/ollama: URI, or a cached name (see `list`).");
    eprintln!("Tests: pp<P> = prefill-only (ingest P tokens); tg<T> = prefill + decode T");
    eprintln!("tokens. Sampling is greedy; every rep starts from an empty KV context.");
}

/// One measured test (a row of the result table).
struct BenchRow {
    test: String,      // "pp512" / "tg128"
    mean: f64,         // tok/s
    stdev: f64,        // tok/s (sample stddev over the reps)
    samples: Vec<f64>, // tok/s per rep
}

pub fn run(prog: &str, args: &[String]) -> i32 {
    // === Parse (hand-rolled, bench-local flags) ===
    let mut p_prompt: usize = 512;
    let mut n_gen: usize = 128;
    let mut reps: usize = 3;
    let mut out_fmt = "md".to_string();
    let mut n_ctx_user: Option<usize> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut parse_err: Option<String> = None;
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        let mut next = |name: &str| -> Option<String> {
            if i + 1 < args.len() {
                Some(args[i + 1].clone())
            } else {
                parse_err = Some(format!("missing value for {name}"));
                None
            }
        };
        match a {
            "-h" | "--help" => {
                print_usage(prog);
                std::process::exit(0);
            }
            "-p" => {
                if let Some(v) = next("-p") {
                    p_prompt = v.parse().unwrap_or_else(|_| {
                        parse_err = Some(format!("invalid -p '{v}'"));
                        0
                    });
                }
                i += 2;
            }
            "-n" => {
                if let Some(v) = next("-n") {
                    n_gen = v.parse().unwrap_or_else(|_| {
                        parse_err = Some(format!("invalid -n '{v}'"));
                        0
                    });
                }
                i += 2;
            }
            "-r" => {
                if let Some(v) = next("-r") {
                    reps = v.parse().unwrap_or_else(|_| {
                        parse_err = Some(format!("invalid -r '{v}'"));
                        0
                    });
                }
                i += 2;
            }
            "-o" => {
                if let Some(v) = next("-o") {
                    out_fmt = v;
                }
                i += 2;
            }
            "--n-ctx" => {
                if let Some(v) = next("--n-ctx") {
                    n_ctx_user = Some(v.parse().unwrap_or_else(|_| {
                        parse_err = Some(format!("invalid --n-ctx '{v}'"));
                        0
                    }));
                }
                i += 2;
            }
            "-t" | "--threads" => {
                if let Some(v) = next("-t") {
                    let n = v.parse().unwrap_or_else(|_| {
                        parse_err = Some(format!("invalid --threads '{v}'"));
                        0
                    });
                    crate::kernel::set_cpu_threads(n);
                }
                i += 2;
            }
            _ => {
                if a.starts_with('-') && a.len() > 1 {
                    parse_err = Some(format!("unknown bench option '{a}'"));
                    i += 1;
                } else {
                    positional.push(a.to_string());
                    i += 1;
                }
            }
        }
    }
    if let Some(e) = parse_err {
        eprintln!("Error: {e}");
        print_usage(prog);
        return 1;
    }
    if positional.is_empty() {
        eprintln!("Error: bench needs a model");
        print_usage(prog);
        return 1;
    }
    if positional.len() > 1 {
        eprintln!(
            "Error: bench takes exactly one model (got {})",
            positional.len()
        );
        return 1;
    }
    if reps == 0 {
        eprintln!("Error: -r must be >= 1");
        return 1;
    }
    if p_prompt == 0 && n_gen == 0 {
        eprintln!("Error: -p 0 and -n 0 — nothing to bench");
        return 1;
    }
    let out_fmt = match out_fmt.as_str() {
        "md" => Format::Md,
        "csv" => Format::Csv,
        "json" => Format::Json,
        other => {
            eprintln!("Error: invalid -o '{other}' (md|csv|json)");
            return 1;
        }
    };

    // === Resolve the model (paths, hf:/ollama: URIs, cached names) ===
    let model_arg = positional[0].clone();
    let is_uri = model_arg.starts_with("hf:")
        || model_arg.starts_with("ollama:")
        || (!model_arg.starts_with('/')
            && !model_arg.starts_with('.')
            && !model_arg.starts_with('~'));
    let model_path = match crate::download::resolve(&model_arg) {
        Ok(p) => {
            if is_uri {
                eprintln!("Model ready: {}", p.display());
            }
            p.to_string_lossy().to_string()
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            return 1;
        }
    };

    // === Load GGUF (single file or multi-part split) ===
    eprintln!("Loading model: {} ...", model_path);
    let Some(gguf_model) = crate::gguf::load_gguf_model(std::path::Path::new(&model_path)) else {
        eprintln!("Error: failed to parse GGUF: {model_path}");
        return 1;
    };
    // Size (GiB) comes from the GGUF itself, summed over all split parts.
    let total_bytes: u64 = gguf_model.parts.iter().map(|p| p.data.len() as u64).sum();
    let ctx = &gguf_model.parts[0].ctx;
    // Parameter count: sum of tensor element counts (llama-bench's n_params).
    // Split GGUFs list disjoint tensor subsets per part (parts[0].info holds
    // only part 0's tensors), so sum across every part.
    let n_params: u64 = gguf_model
        .parts
        .iter()
        .flat_map(|p| p.ctx.info.iter())
        .map(|t| t.ne.iter().product::<i64>() as u64)
        .sum();

    // === GPU backends + model ===
    #[cfg(target_os = "macos")]
    crate::metal::MpsState::init();
    #[cfg(feature = "cuda")]
    crate::cuda::CudaState::init();
    let model = match crate::models::load_model(&gguf_model) {
        Some(m) => m,
        None => {
            eprintln!("Error: failed to load model");
            return 1;
        }
    };
    let tokenizer = crate::tokenizer::Tokenizer::load(ctx);
    eprintln!("Model loaded.");

    // Backend label from the engine's own availability report (same checks
    // main.rs's graph-export block uses: device presence, no weights gate —
    // an all-CPU-registered model would run slow, not wrong).
    let backend = detect_backend();
    let threads = crate::kernel::cpu_threads();

    // === KV sizing: pp+tg+16 unless --n-ctx, clamped to the model context ===
    // The internal sizing must NOT clamp pp away (P+T+16 always ≥ the positions
    // the tests use); the model's own context length is the only hard bound —
    // beyond it the test genuinely does not fit and we error out instead of
    // silently corrupting KV (forward_cached asserts positions < n_ctx).
    let model_ctx_len: Option<usize> = {
        let arch = ctx
            .get_key_val_str("general.architecture")
            .unwrap_or_default();
        ctx.get_key_val_i64(&format!("{arch}.context_length"))
            .or_else(|| ctx.get_key_val_i64("llama.context_length"))
            .map(|v| v as usize)
    };
    // Positions the tests need: pp uses P (0..P), tg uses P+T (prefill 0..P,
    // decode P..P+T); with P=0 the tg seed token occupies position 0.
    let need = if p_prompt > 0 {
        p_prompt + n_gen
    } else {
        n_gen + 1
    };
    let mut n_ctx = n_ctx_user.unwrap_or(p_prompt + n_gen + 16).max(need);
    if let Some(cl) = model_ctx_len {
        n_ctx = n_ctx.min(cl);
    }
    if need > n_ctx {
        eprintln!(
            "Error: bench needs {need} KV positions (pp{p_prompt} + tg{n_gen}) but n_ctx is {n_ctx}{}",
            model_ctx_len
                .map(|cl| format!(" (model context length {cl})"))
                .unwrap_or_default()
        );
        return 1;
    }

    // === Synthetic prompt (token ids don't affect the compute; deterministic
    // spread over the vocab instead of a pathological all-same-token run) ===
    let n_vocab = (model.n_vocab() as u64).max(1);
    let prompt_tokens: Vec<u32> = (0..p_prompt)
        .map(|i| (((i as u64) * 7919 + 13) % n_vocab) as u32)
        .collect();
    // Seed token for the P=0 tg test (decode needs one starting forward).
    let seed_tok = tokenizer.bos_token.min(n_vocab as u32 - 1);

    let mut cache = GraphCache::new();

    // === Run the test matrix ===
    let mut rows: Vec<BenchRow> = Vec::new();
    if p_prompt > 0 {
        eprintln!("bench: pp{p_prompt} (prefill-only), {reps} reps + 1 warmup, n_ctx={n_ctx} ...");
        let positions: Vec<usize> = (0..p_prompt).collect();
        // Warmup rep (untimed): builds/first-executes the prefill graph (CUDA
        // Graph capture, buffer pools) so measured reps reuse it by params.
        let _ = model.forward_graph_cached(&prompt_tokens, &positions, 1, n_ctx, &mut cache);
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            // Fresh context per rep: positions from 0 overwrite every slot this
            // rep reads (see module docs); no reload, no KV zeroing.
            let _ = model.forward_graph_cached(&prompt_tokens, &positions, 1, n_ctx, &mut cache);
            samples.push(p_prompt as f64 / t0.elapsed().as_secs_f64());
        }
        rows.push(BenchRow {
            test: format!("pp{p_prompt}"),
            mean: mean(&samples),
            stdev: stdev(&samples),
            samples,
        });
    }
    if n_gen > 0 {
        eprintln!(
            "bench: tg{n_gen} (prefill {p_prompt} + decode {n_gen}), {reps} reps + 1 warmup, n_ctx={n_ctx} ..."
        );
        // Warmup rep (untimed), same fresh-context shape as a measured rep.
        let _ = decode_once(
            &*model,
            &mut cache,
            &prompt_tokens,
            n_ctx,
            p_prompt,
            n_gen,
            seed_tok,
        );
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            samples.push(decode_once(
                &*model,
                &mut cache,
                &prompt_tokens,
                n_ctx,
                p_prompt,
                n_gen,
                seed_tok,
            ));
        }
        rows.push(BenchRow {
            test: format!("tg{n_gen}"),
            mean: mean(&samples),
            stdev: stdev(&samples),
            samples,
        });
    }

    let model_name = std::path::Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.clone());

    match out_fmt {
        Format::Md => print_md(&model_name, total_bytes, n_params, backend, threads, &rows),
        Format::Csv => print_csv(&model_name, total_bytes, n_params, backend, threads, &rows),
        Format::Json => print_json(&model_name, total_bytes, n_params, backend, threads, &rows),
    }
    0
}

/// One tg rep: untimed context prefill from position 0, then the timed decode
/// of exactly `n_gen` tokens (sampling included — main.rs's "Generated" caliber).
/// Returns tok/s. Greedy sampling → identical token stream across reps.
fn decode_once(
    model: &dyn ModelDef,
    cache: &mut GraphCache,
    prompt_tokens: &[u32],
    n_ctx: usize,
    p_prompt: usize,
    n_gen: usize,
    seed_tok: u32,
) -> f64 {
    // Fresh context per rep: the prefill re-writes KV slots 0..P and attention
    // never reads beyond a query token's own position (module docs), so no KV
    // zeroing or reload is needed between reps.
    let mut logits = if p_prompt > 0 {
        let positions: Vec<usize> = (0..p_prompt).collect();
        model.forward_graph_cached(prompt_tokens, &positions, 1, n_ctx, cache)
    } else {
        // No context: one seed token starts the decode chain (position 0).
        model.forward_graph_cached(&[seed_tok], &[0], 1, n_ctx, cache)
    };
    let start = Instant::now();
    let mut pos = if p_prompt > 0 { p_prompt } else { 1 };
    for _ in 0..n_gen {
        let sampled = crate::sampler::sample_greedy(&logits);
        logits = model.forward_graph_cached(&[sampled.token_id], &[pos], 1, n_ctx, cache);
        pos += 1;
    }
    n_gen as f64 / start.elapsed().as_secs_f64()
}

enum Format {
    Md,
    Csv,
    Json,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample stddev (n-1 divisor), the llama-bench formula
/// (tools/llama-bench/llama-bench.cpp `stdev<T>`).
fn stdev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let n = v.len() as f64;
    let m = mean(v);
    let sq_sum: f64 = v.iter().map(|x| x * x).sum();
    (sq_sum / (n - 1.0) - m * m * n / (n - 1.0)).max(0.0).sqrt()
}

fn print_md(
    model_name: &str,
    total_bytes: u64,
    n_params: u64,
    backend: &str,
    threads: usize,
    rows: &[BenchRow],
) {
    let headers = [
        "model", "size", "params", "backend", "threads", "test", "t/s",
    ];
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|r| {
            vec![
                model_name.to_string(),
                format!("{:.2} GiB", total_bytes as f64 / 1024.0 / 1024.0 / 1024.0),
                format!("{:.2} B", n_params as f64 / 1e9),
                backend.to_string(),
                threads.to_string(),
                r.test.clone(),
                format!("{:.2} ± {:.2}", r.mean, r.stdev),
            ]
        })
        .collect();
    let mut widths = headers.map(|h| h.len());
    for row in &cells {
        for (w, c) in widths.iter_mut().zip(row) {
            *w = (*w).max(c.len());
        }
    }
    let line: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            if i == 0 {
                format!("{:<width$}", h, width = widths[0])
            } else {
                format!("{:>width$}", h, width = widths[i])
            }
        })
        .collect();
    println!("| {} |", line.join(" | "));
    let dashes: Vec<String> = widths
        .iter()
        .enumerate()
        .map(|(i, w)| {
            if i == 0 {
                "-".repeat(w + 2)
            } else {
                format!("{}:", "-".repeat(w + 1))
            }
        })
        .collect();
    println!("|{}|", dashes.join("|"));
    for row in &cells {
        let line: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i == 0 {
                    format!("{:<width$}", c, width = widths[0])
                } else {
                    format!("{:>width$}", c, width = widths[i])
                }
            })
            .collect();
        println!("| {} |", line.join(" | "));
    }
}

fn print_csv(
    model_name: &str,
    total_bytes: u64,
    n_params: u64,
    backend: &str,
    threads: usize,
    rows: &[BenchRow],
) {
    println!(
        "\"model\",\"size\",\"params\",\"backend\",\"threads\",\"test\",\"ts_mean\",\"ts_stddev\""
    );
    for r in rows {
        println!(
            "\"{model_name}\",\"{total_bytes}\",\"{n_params}\",\"{backend}\",\"{threads}\",\"{}\",\"{:.2}\",\"{:.2}\"",
            r.test, r.mean, r.stdev
        );
    }
}

fn print_json(
    model_name: &str,
    total_bytes: u64,
    n_params: u64,
    backend: &str,
    threads: usize,
    rows: &[BenchRow],
) {
    let r2 = |x: f64| (x * 100.0).round() / 100.0;
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "model": model_name,
                "size": total_bytes,
                "params": n_params,
                "backend": backend,
                "threads": threads,
                "test": r.test,
                "ts_mean": r2(r.mean),
                "ts_stddev": r2(r.stdev),
                "samples_ts": r.samples.iter().map(|s| r2(*s)).collect::<Vec<f64>>(),
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".to_string())
    );
}

/// Backend label from the engine's own availability report (device presence +
/// opt-out envs, mirroring main.rs's graph-export block). CUDA > Metal > CPU.
fn detect_backend() -> &'static str {
    #[cfg(feature = "cuda")]
    if crate::cuda::CudaState::get().is_some() {
        return "CUDA";
    }
    #[cfg(target_os = "macos")]
    if crate::graph::metal_backend::metal_available()
        && !std::env::var("MINFER_DISABLE_MPS").map_or(false, |v| v == "1")
    {
        return "Metal";
    }
    "CPU"
}
