//! `minfer specverify` — D5-1a gate instrument: measure the target model's
//! batched verify-step cost `C_T(nt)` at a fixed deep KV depth.
//!
//! One speculative round verifies `d+1` drafted tokens in a single target
//! forward at `nt = d+1` with `n_out = d+1` (every row needs logits). The
//! D5-0 gate (step doc 80) hangs on ONE number: the per-step amortization
//!
//!   amortization(nt) = nt · C_T(1) / C_T(nt)   — required ≥ 2.5× at nt=3 (d=2)
//!
//! This subcommand measures that ratio directly, with no engine changes:
//! the generic `forward_graph_cached(tokens, positions, n_out)` primitive
//! already accepts `nt > 1, n_out = nt`, and at `nt > 1` the CUDA backend
//! dispatches the batched path (BT-MMQ GEMM + full attention) — exactly the
//! verify regime. The decode-only fast paths (MMVQ, FusedQKV/FusedFFN,
//! split-K flash-decoding) are `nt == 1`-gated and correctly stay off.
//!
//! Protocol (campaign methodology: fixed-depth reps, medians, drift check):
//! 1. prefill `P` prompt tokens (untimed) — fills KV slots `0..P` so every
//!    later attention read hits an initialized slot;
//! 2. for each `nt` in {1, 3, 5}: 3 untimed warmup steps (absorbing the
//!    one-time graph rebuild on the `nt` switch — `GraphParams.n_tokens`
//!    enters the reuse identity), then `r` timed steps at positions
//!    `[P-nt..P)`, `n_out = nt`. Fixed depth, fixed shapes: every rep reads
//!    the same KV span (slots are overwritten in place — sound because KV
//!    positions are data, not structure; see `bench.rs` module docs);
//! 3. pass 1 runs nt in [1, 3, 5], pass 2 in reverse [5, 3, 1]; the
//!    amortization is computed from pass 1, pass 2 is the drift check.
//!
//! Output is JSON (default) or a markdown table on stdout; progress goes to
//! stderr. Exit code is 0 whenever the measurement completes — the PASS/FAIL
//! verdict is data, not an error.

use crate::graph::cache::GraphCache;
use crate::models::ModelDef;
use std::time::Instant;

pub fn print_usage(prog: &str) {
    eprintln!("minfer specverify — D5-1a verify-step cost micro-bench (C_T(nt) amortization gate)");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  {prog} specverify [OPTIONS] <model>");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -p <N>     KV depth: prefill P tokens, then measure every nt at that");
    eprintln!("             depth (default 512)");
    eprintln!("  -r <N>     timed reps per nt phase, after 3 untimed warmups (default 40;");
    eprintln!("             reports median / p10 / p90 in ms)");
    eprintln!("  -o <FMT>   output format: json | md (default json)");
    eprintln!("  -t, --threads <N>  CPU worker threads (GPU backends are unaffected)");
    eprintln!("  -h, --help show this help");
    eprintln!();
    eprintln!("Phases: nt = 1 (baseline C_T(1)), 3 (d=2 verify), 5 (d=4 verify).");
    eprintln!(
        "Gate (D5-0): amortization = nt·C_T(1)/C_T(3) (doc 81 §2) must be ≥ 2.5× to proceed."
    );
    eprintln!("The model may be a path or a cached name (see `list`).");
}

struct Phase {
    nt: usize,
    n_out: usize,
    median_ms: f64,
    p10_ms: f64,
    p90_ms: f64,
    samples_ms: Vec<f64>,
}

/// Timed phase: warmups + reps of one batched forward at fixed depth.
fn measure_phase(
    model: &dyn ModelDef,
    cache: &mut GraphCache,
    n_ctx: usize,
    depth: usize,
    nt: usize,
    warmups: usize,
    reps: usize,
    n_out: usize,
) -> Phase {
    let n_vocab = (model.n_vocab() as u64).max(1);
    let tokens: Vec<u32> = (0..nt)
        .map(|i| (((i as u64) * 7919 + 13) % n_vocab) as u32)
        .collect();
    let positions: Vec<usize> = (depth - nt..depth).collect();
    for _ in 0..warmups {
        let _ = model.forward_graph_cached(&tokens, &positions, n_out, n_ctx, cache);
    }
    let mut samples_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = model.forward_graph_cached(&tokens, &positions, n_out, n_ctx, cache);
        samples_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = samples_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| sorted[((q * (sorted.len() - 1) as f64).round()) as usize];
    Phase {
        nt,
        n_out,
        median_ms: sorted[sorted.len() / 2],
        p10_ms: pick(0.10),
        p90_ms: pick(0.90),
        samples_ms,
    }
}

pub fn run(prog: &str, args: &[String]) -> i32 {
    let mut depth: usize = 512;
    let mut reps: usize = 40;
    let mut out_fmt = "json".to_string();
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
                    depth = v.parse().unwrap_or_else(|_| {
                        parse_err = Some(format!("invalid -p '{v}'"));
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
                    parse_err = Some(format!("unknown specverify option '{a}'"));
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
    if positional.len() != 1 {
        eprintln!("Error: specverify takes exactly one model");
        print_usage(prog);
        return 1;
    }
    if depth < 32 {
        eprintln!("Error: -p must be ≥ 32 (the fixed-depth protocol needs a deep KV)");
        return 1;
    }
    if reps < 5 {
        eprintln!("Error: -r must be ≥ 5 (median needs samples)");
        return 1;
    }
    let json = match out_fmt.as_str() {
        "json" => true,
        "md" => false,
        other => {
            eprintln!("Error: invalid -o '{other}' (json|md)");
            return 1;
        }
    };

    // === Resolve + load (same path/URI resolution as bench) ===
    let model_arg = positional[0].clone();
    let model_path = match crate::download::resolve(&model_arg) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(e) => {
            eprintln!("Error: {}", e);
            return 1;
        }
    };
    eprintln!("Loading model: {model_path} ...");
    let Some(gguf_model) = crate::gguf::load_gguf_model(std::path::Path::new(&model_path)) else {
        eprintln!("Error: failed to parse GGUF: {model_path}");
        return 1;
    };
    let ctx = &gguf_model.parts[0].ctx;
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
    eprintln!("Model loaded.");

    // Backend label (same availability checks bench uses).
    #[cfg(feature = "cuda")]
    let cuda_on = crate::cuda::CudaState::get().is_some();
    #[cfg(not(feature = "cuda"))]
    let cuda_on = false;
    let backend = if cuda_on { "cuda" } else { "cpu" };

    // KV sizing: depth + margin, clamped to the model's context length.
    let model_ctx_len: Option<usize> = {
        let arch = ctx
            .get_key_val_str("general.architecture")
            .unwrap_or_default();
        ctx.get_key_val_i64(&format!("{arch}.context_length"))
            .or_else(|| ctx.get_key_val_i64("llama.context_length"))
            .map(|v| v as usize)
    };
    let mut n_ctx = depth + 64;
    if let Some(cl) = model_ctx_len {
        n_ctx = n_ctx.min(cl);
    }
    if depth + 8 > n_ctx {
        eprintln!("Error: -p {depth} does not fit the model context length");
        return 1;
    }

    // === Prefill: fill KV slots 0..depth (untimed setup) ===
    let n_vocab = (model.n_vocab() as u64).max(1);
    let prompt_tokens: Vec<u32> = (0..depth)
        .map(|k| (((k as u64) * 7919 + 13) % n_vocab) as u32)
        .collect();
    let prefill_positions: Vec<usize> = (0..depth).collect();
    let mut cache = GraphCache::new();
    eprintln!("specverify: prefill {depth} tokens (KV fill, untimed) ...");
    let _ = model.forward_graph_cached(&prompt_tokens, &prefill_positions, 1, n_ctx, &mut cache);

    // === Two passes over the nt list (pass 2 reversed: drift check) ===
    // MINFER_SPECVERIFY_NTS overrides the phase list (e.g. "1,3,5,16" to probe
    // the tile-regime pricing curve). MINFER_SPECVERIFY_NOUT=1 forces the
    // prefill-style n_out=1 (tail-row reduction) at every nt — the
    // discriminator between "nt>1 GEMM dispatch cost" and "n_out=nt output
    // path cost".
    let nts: Vec<usize> = std::env::var("MINFER_SPECVERIFY_NTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .filter(|&x| x >= 1 && x <= depth)
                .collect()
        })
        .unwrap_or_else(|| vec![1, 3, 5]);
    let force_nout1 = std::env::var("MINFER_SPECVERIFY_NOUT").map_or(false, |v| v == "1");
    let warmups = 3;
    let measure_at = |model: &dyn ModelDef, cache: &mut GraphCache, nt: usize| {
        let n_out = if force_nout1 { 1 } else { nt };
        measure_phase(model, cache, n_ctx, depth, nt, warmups, reps, n_out)
    };
    let mut pass1: Vec<Phase> = Vec::new();
    for &nt in &nts {
        eprintln!(
            "specverify: pass 1 nt={nt} n_out={} ({warmups} warmup + {reps} timed) ...",
            if force_nout1 { 1 } else { nt }
        );
        pass1.push(measure_at(&*model, &mut cache, nt));
    }
    let mut pass2: Vec<Phase> = Vec::new();
    for &nt in nts.iter().rev() {
        eprintln!("specverify: pass 2 (drift check) nt={nt} ...");
        pass2.push(measure_at(&*model, &mut cache, nt));
    }

    let get = |phases: &[Phase], nt: usize| {
        phases
            .iter()
            .find(|p| p.nt == nt)
            .map(|p| p.median_ms)
            .unwrap_or(f64::NAN)
    };
    let c1 = get(&pass1, 1);
    let c3 = get(&pass1, 3);
    let c5 = get(&pass1, 5);
    // Doc 80/81 definition: amortization(nt) = nt * C_T(1) / C_T(nt) — the
    // serial-time equivalent of one batched verify step. (The C_T(1)/C_T(nt)
    // ratio shipped here originally missed the nt factor and understated the
    // metric by exactly nt; docs 81/82 computed their tables from the raw
    // medians with the correct formula.)
    let amort3 = 3.0 * c1 / c3;
    let amort5 = 5.0 * c1 / c5;
    let gate_pass = amort3 >= 2.5;
    let model_name = std::path::Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.clone());
    eprintln!(
        "specverify: C_T(1)={c1:.2} ms  C_T(3)={c3:.2} ms  C_T(5)={c5:.2} ms  \
         amortization nt=3 {amort3:.2}×  nt=5 {amort5:.2}×  →  gate {}: \
         required ≥ 2.5× at nt=3",
        if gate_pass { "PASS" } else { "FAIL" }
    );

    if json {
        println!("{{");
        println!(
            "  \"tool\": \"specverify\", \"date\": \"{}\",",
            chrono_today()
        );
        println!("  \"model\": \"{model_name}\", \"backend\": \"{backend}\",");
        println!("  \"kv_depth\": {depth}, \"reps\": {reps}, \"warmups\": {warmups},");
        println!("  \"pass1\": [");
        for (k, p) in pass1.iter().enumerate() {
            let comma = if k + 1 < pass1.len() { "," } else { "" };
            println!(
                "    {{\"nt\": {}, \"n_out\": {}, \"median_ms\": {:.3}, \"p10_ms\": {:.3}, \"p90_ms\": {:.3}, \"samples_ms\": [{}]}}{}",
                p.nt,
                p.n_out,
                p.median_ms,
                p.p10_ms,
                p.p90_ms,
                fmt_samples(&p.samples_ms),
                comma
            );
        }
        println!("  ],");
        println!("  \"pass2_drift_check\": [");
        for (k, p) in pass2.iter().enumerate() {
            let comma = if k + 1 < pass2.len() { "," } else { "" };
            println!(
                "    {{\"nt\": {}, \"n_out\": {}, \"median_ms\": {:.3}, \"p10_ms\": {:.3}, \"p90_ms\": {:.3}}}{}",
                p.nt, p.n_out, p.median_ms, p.p10_ms, p.p90_ms, comma
            );
        }
        println!("  ],");
        println!("  \"C_T1_ms\": {c1:.3}, \"C_T3_ms\": {c3:.3}, \"C_T5_ms\": {c5:.3},");
        println!("  \"amortization_nt3\": {amort3:.3}, \"amortization_nt5\": {amort5:.3},");
        println!("  \"gate\": \"amortization >= 2.5 at nt=3\", \"gate_pass\": {gate_pass}");
        println!("}}");
    } else {
        println!("## specverify — {model_name} ({backend}), KV depth {depth}, {reps} reps");
        println!();
        println!("| nt | median ms | p10 | p90 |");
        println!("|---|---|---|---|");
        for p in &pass1 {
            println!(
                "| {} | {:.2} | {:.2} | {:.2} |",
                p.nt, p.median_ms, p.p10_ms, p.p90_ms
            );
        }
        println!();
        println!(
            "amortization: nt=3 {:.2}×, nt=5 {:.2}× — gate {}: required ≥ 2.5× at nt=3",
            amort3,
            amort5,
            if gate_pass { "PASS" } else { "FAIL" }
        );
    }
    0
}

fn fmt_samples(samples: &[f64]) -> String {
    samples
        .iter()
        .map(|s| format!("{s:.3}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Local date (YYYY-MM-DD) without pulling a date crate in.
fn chrono_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}
