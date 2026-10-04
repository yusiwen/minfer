//! Continuous batching: the batched-vs-serial verdict and its timing helpers.
//!
//! Split out of `src/server/batch/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// Whether a CUDA device participates in this process (`load_model` above
/// initialises the state when the model is loaded).
fn cuda_device_active() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::CudaState::get().is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
/// The E2 correctness half, shared by the batched and staggered arms: every
/// row of `got` must match its serial reference.
///
/// Byte-equality is a **CPU** property: both sides drive the same engine with
/// the same slot reservations (`submit_on` pins each request to the slot it
/// would occupy), so on CPU the arithmetic is identical. On a device it cannot
/// be — the batched step is `nt = 4` and the serial step `nt = 1`, and CUDA's
/// kernels tile by `nt` (measured drift 0.22–0.37 on logits; the plan records
/// it as a named tolerance class), so a greedy continuation may legitimately
/// diverge after a few tokens. Device runs therefore assert the *structural*
/// property that a wrong window would break immediately — the two
/// continuations must start identically — and report how far they track; the
/// window assignment itself is pinned bitwise on device by
/// `batch_order_does_not_change_a_sequences_logits` (same shape, same layout)
/// and `cuda_two_sequences_do_not_cross_attend`.
fn assert_replies_match(what: &str, got: &[Reply], serial: &[Reply]) {
    assert_eq!(got.len(), serial.len(), "{what}: row count differs");
    for (i, (b, s)) in got.iter().zip(serial).enumerate() {
        assert_eq!(
            b.reason, s.reason,
            "{what} request {i}: finish reason differs"
        );
        assert!(!b.text.is_empty(), "{what} request {i}: generated nothing");
        assert!(
            !s.text.is_empty(),
            "serial request {i}: generated nothing (the reference is empty)"
        );
        if cuda_device_active() {
            let common = b
                .text
                .bytes()
                .zip(s.text.bytes())
                .take_while(|(x, y)| x == y)
                .count();
            assert!(
                common > 0,
                "{what} request {i}: {b:?} and serial {s:?} diverge at the first byte on a \
                 device, which numerics cannot explain"
            );
            eprintln!(
                "[e2] {what} request {i}: {common} leading byte(s) shared on device; {:?} ({}) \
                 vs serial {:?} ({})",
                b.text, b.tokens, s.text, s.tokens
            );
        } else {
            assert_eq!(
                b.text, s.text,
                "{what} request {i}: {:?} ({}) vs serial {:?} ({})",
                b.text, b.tokens, s.text, s.tokens
            );
            assert_eq!(
                b.tokens, s.tokens,
                "{what} request {i}: token count differs"
            );
        }
    }
}
/// The location estimate #154's timing verdict uses (the upper median for an
/// even count, matching `cuda_backend::tests::attn_window::median`).
fn median(v: &[f64]) -> f64 {
    assert!(!v.is_empty(), "median of an empty sample");
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}
/// E2's acceptance, measured at the engine: four requests served as one decode
/// batch must generate exactly what four serial requests generate, and the
/// **median** of the per-round `serial/batched` ratios over interleaved rounds
/// must exceed 1.0 (#154).
///
/// Ignored by default like the other real-model tests:
///   cargo test --release --bin minfer -- --ignored server_batch --nocapture
#[test]
#[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
fn server_batch_matches_serial_and_is_faster() {
    let Some(path) = cached_model() else {
        eprintln!("0.5B q4_0 not cached; skipping the batching measurement");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let texts = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Italy is",
        "The capital of Spain is",
    ];
    // Configurable so a bisect can put the *server's* exact configuration on
    // this path: the field bug of 2026-09-19 (plan §14) reproduces through the
    // server with the 7B and two slots, but not here with the 0.5B and four —
    // and these three knobs are the differences that are left.
    let templated = std::env::var("MINFER_BATCH_TEST_TEMPLATED").is_ok();
    let prompts: Vec<Vec<u32>> = texts
        .iter()
        .map(|p| {
            if templated {
                // Exactly what the server does (`server::mod`): the GGUF's
                // chat template, rendered with a generation prompt, then
                // tokenized. The legacy trait `format_chat` fallback (deleted
                // in #242) was a *different* path and produced a 13-token
                // prompt where the server's is 34 — which is why the first
                // bisect compared unequal inputs.
                let tpl = crate::server::chat_template_from_gguf(&gguf.parts[0].data);
                let msgs = vec![("user".to_string(), Some(p.to_string()))];
                tok.encode(
                    &crate::template::render_messages_opt(
                        tpl.as_deref(),
                        &msgs,
                        true,
                        &tok.bos_text(),
                    )
                    .expect("chat template renders"),
                )
            } else {
                tok.encode(p)
            }
        })
        .collect();
    let n_ctx: usize = std::env::var("MINFER_BATCH_TEST_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let n_slots: usize = std::env::var("MINFER_BATCH_TEST_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let n_req = n_slots.min(prompts.len());
    let prompts: Vec<Vec<u32>> = prompts.into_iter().take(n_req).collect();
    let max_tokens = 16;
    // ---- #154: the throughput verdict is a median over interleaved rounds ----
    //
    // The pre-#154 form measured the two whole workloads **once, sequentially**
    // (batched, then serial) and asserted the wall-clock relation
    // `t_serial > t_batch`. That is not a property of the code: under the
    // parallel `--ignored` harness the first-measured phase absorbs the
    // start-up wave and the ratio can invert with nothing to absorb it.
    // Measured on dgxspark (CPU build) at `e1ac17f`: parallel **21.20s batched
    // vs 9.95s serial (0.47x)** — the only failure of the 29-gate set (28
    // passed / 1 failed) — while the same binary serially was **0.72s vs
    // 1.07s (1.50x)**.
    //
    // The rounds are now **interleaved** (batched, serial, batched, serial,
    // …), so each ratio is a matched pair measured next to each other on the
    // same machine state, and the assertion is on the **median of the
    // per-round `serial/batched` ratios** — the location estimate that
    // tolerates up to `rounds / 2` rounds a passing load spike disturbed.
    // Every sample and the median are printed so a loaded box's verdict is
    // auditable. This is the shape #123 gave
    // `cuda_map_window_costs_no_more_than_the_span_it_replaces`.
    //
    // Cost, and the round count. The correctness comparison below stays a
    // **full-length** pair (`max_tokens`); the timed rounds are a separate,
    // shorter workload so the gate's total wall clock stays bounded. On an
    // idle box the full-length pair is ~1.8s and the timing pair is
    // proportionally less; on the loaded parallel harness a full-length pair
    // reached ~31s. Seven rounds is the smallest odd count whose median
    // tolerates three disturbed rounds; `timing_tokens` halves the timed
    // workload's per-round cost without touching the correctness comparison.
    // Both knobs are env-overridable so a bisect can trade cost for samples.
    let rounds: usize = std::env::var("MINFER_BATCH_TEST_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let timing_tokens: i64 = std::env::var("MINFER_BATCH_TEST_TIMING_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    eprintln!(
        "[e2] config: {n_slots} slot(s), n_ctx {n_ctx}, {n_req} request(s), templated={templated}, \
         prompt len {}, correctness max_tokens {max_tokens}, {rounds} interleaved timing round(s) \
         at max_tokens {timing_tokens}",
        prompts[0].len()
    );

    // ---- correctness: the full-length pair, byte-comparable on CPU ----
    let (batched, t_batch) =
        run_batched(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens, false);
    let (serial, t_serial) = run_serial(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens);
    assert_replies_match("batched", &batched, &serial);
    // ---- the server's pattern: staggered admission (mixed step widths) ----
    let (staggered, t_stag) =
        run_batched(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens, true);
    assert_replies_match("staggered", &staggered, &serial);
    eprintln!(
        "[e2] staggered {:.2}s vs simultaneous {:.2}s for {n_slots} slots",
        t_stag, t_batch
    );
    let total: usize = serial.iter().map(|r| r.tokens).sum();
    assert!(total > 0, "the workload generated nothing");
    eprintln!(
        "[e2] {n_slots} slots: batched {t_batch:.2}s vs serial {t_serial:.2}s for {total} tokens \
         (per-request {:?} batched / {:?} serial)",
        batched.iter().map(|r| r.tokens).collect::<Vec<_>>(),
        serial.iter().map(|r| r.tokens).collect::<Vec<_>>()
    );
    eprintln!(
        "[e2] full-length single pair: {:.1} tok/s batched vs {:.1} tok/s serial = {:.2}x \
         (audit only, not the verdict)",
        total as f64 / t_batch,
        total as f64 / t_serial,
        t_serial / t_batch
    );

    // ---- throughput: interleaved matched rounds, verdict on the median ----
    let mut t_batches: Vec<f64> = Vec::with_capacity(rounds);
    let mut t_serials: Vec<f64> = Vec::with_capacity(rounds);
    let mut ratios: Vec<f64> = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let (_, tb) = run_batched(
            &*model,
            &tok,
            &prompts,
            n_slots,
            n_ctx,
            timing_tokens,
            false,
        );
        let (_, ts) = run_serial(&*model, &tok, &prompts, n_slots, n_ctx, timing_tokens);
        t_batches.push(tb);
        t_serials.push(ts);
        ratios.push(ts / tb);
    }
    let med_batch = median(&t_batches);
    let med_serial = median(&t_serials);
    let med = median(&ratios);
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "[e2] timing: {rounds} interleaved rounds at max_tokens {timing_tokens} — batched \
         {t_batches:?} s, serial {t_serials:?} s; per-round serial/batched ratios (round order) \
         {ratios:?}, sorted {sorted:?} — median {med:.3}x (medians {med_serial:.2}s serial vs \
         {med_batch:.2}s batched)"
    );
    // The threshold is the pre-#154 gate's "must not be slower": 1.0x, not a
    // named margin. What changed is the statistic (a median of matched pairs
    // instead of one sequential pair). On an idle CPU box the median is
    // ~1.5x, so the gate still trips on a mutation that halves the batched
    // arm's win (see the plan record's mutation check).
    assert!(
        med > 1.0,
        "batching must not be slower: median serial/batched {med:.3}x over {rounds} interleaved \
         rounds (per-round ratios {sorted:?}; medians {med_serial:.2}s serial vs {med_batch:.2}s \
         batched)"
    );
}
