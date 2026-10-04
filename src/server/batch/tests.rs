//! `#[cfg(test)] mod tests` for `src/server/batch.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::models::ModelDef;
use std::time::Instant;

mod batching;
mod http;
mod kv_sharing;
mod metrics;
mod prefill;
mod slots;
mod stall;
fn cached_model() -> Option<std::path::PathBuf> {
    // `MINFER_BATCH_TEST_MODEL` points the measurement at another cached
    // model (the 7B is where batching should pay: decode is weight-bandwidth
    // bound there, while the 0.5B's decode is kernel-compute bound).
    if let Ok(custom) = std::env::var("MINFER_BATCH_TEST_MODEL") {
        let p = std::path::PathBuf::from(custom);
        return p.exists().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    p.exists().then_some(p)
}
/// Drive `n` requests through the engine, returning each one's generated
/// tokens and the wall time. Requests are submitted together (the engine
/// admits what fits) and stepped until they all finish, which is what
/// continuous batching does.
#[derive(Debug, PartialEq)]
struct Reply {
    text: String,
    tokens: usize,
    reason: String,
}
fn sampling_params(max_tokens: i64) -> SamplingParams {
    SamplingParams {
        // Greedy, identical on both sides of the comparison.
        temp: 0.0,
        top_k: 1,
        top_p: 1.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        typical_p: 1.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.5,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 64,
        dry_breakers: Vec::new(),
        mirostat: crate::sampler::MirostatMode::Off,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        mirostat_m: 100,
        logit_bias: Vec::new(),
        grammar_source: None,
        grammar: None,
        seed: 7,
        stop_strings: Vec::new(),
        max_tokens,
    }
}
fn run_batched(
    model: &dyn ModelDef,
    tok: &Tokenizer,
    prompts: &[Vec<u32>],
    n_slots: usize,
    n_ctx: usize,
    max_tokens: i64,
    stagger: bool,
) -> (Vec<Reply>, f64) {
    let mut engine = BatchEngine::new(model, n_slots, n_ctx).expect("engine");
    let mut out: Vec<Option<Reply>> = (0..prompts.len()).map(|_| None).collect();
    let mut text: Vec<String> = vec![String::new(); prompts.len()];
    let mut pending: Vec<(usize, mpsc::Receiver<StreamEvent>)> = Vec::new();
    let mut queue: Vec<(usize, Job)> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
        queue.push((
            i,
            Job {
                input_ids: p.clone(),
                params: sampling_params(max_tokens),
                tx,
            },
        ));
        pending.push((i, rx));
    }
    let t0 = Instant::now();
    // #160: the whole drive is bounded by work, not by the clock. `sum of the
    // prompts` plus every request's own token cap is the most the engine can
    // legitimately consume; the budget's 4x margin absorbs the admission and
    // prefill steps, which do not show up in `work_units`.
    let total_prompt: usize = prompts.iter().map(|p| p.len()).sum();
    let budget = step_budget(total_prompt, prompts.len() * step_cap(max_tokens, n_ctx));
    let mut bound = WorkBound::new(&engine, budget, "the batched run");
    // Admit, then step until every request has finished.
    while !queue.is_empty() || engine.busy() {
        // `stagger` reproduces the **server's** admission pattern: requests
        // arrive while others decode, so `serve_loop` admits one per step and
        // the step sequence mixes widths (a 1-sequence step, then wider
        // ones). Without it, everything is admitted before the first tick and
        // every step has the same width — which is what this helper always
        // did, and why the blocker below hid from it.
        let mut admit = if stagger { 1 } else { engine.idle_slots() };
        while !queue.is_empty() && engine.idle_slots() > 0 && admit > 0 {
            let (i, job) = queue.remove(0);
            engine
                .submit(model, tok, job)
                .unwrap_or_else(|e| panic!("submit request {i}: {}", e.message));
            admit -= 1;
        }
        engine.tick(model, tok).expect("tick");
        bound.step(&engine);
        // Drain events; a finished request is reported by `Finish`, and its
        // text is the concatenation of the `Text` events.
        for (i, rx) in pending.iter_mut() {
            loop {
                match rx.try_recv() {
                    Ok(StreamEvent::Text(t)) => text[*i].push_str(&t),
                    Ok(StreamEvent::Finish { reason, tokens }) => {
                        out[*i] = Some(Reply {
                            text: std::mem::take(&mut text[*i]),
                            tokens,
                            reason,
                        });
                    }
                    Ok(StreamEvent::Err(e)) => panic!("request {i}: {}", e.message),
                    Err(_) => break,
                }
            }
        }
    }
    (
        out.into_iter()
            .map(|r| r.expect("every request finished"))
            .collect(),
        t0.elapsed().as_secs_f64(),
    )
}
/// The serial baseline: the same four requests, each served alone, **on the
/// slot it would occupy in the batched run** — a request's window offset is
/// part of its determinism (see [`BatchEngine::submit_on`]), so comparing
/// across offsets would measure arithmetic, not batching.
fn run_serial(
    model: &dyn ModelDef,
    tok: &Tokenizer,
    prompts: &[Vec<u32>],
    n_slots: usize,
    n_ctx: usize,
    max_tokens: i64,
) -> (Vec<Reply>, f64) {
    // One engine, so the baseline pays the same one-time graph builds as the
    // batched run; each request is placed on the slot it would occupy.
    let t0 = Instant::now();
    let mut out = Vec::new();
    let mut engine = BatchEngine::new(model, n_slots, n_ctx).expect("engine");
    for (i, p) in prompts.iter().enumerate() {
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
        engine
            .submit_on(
                model,
                tok,
                i,
                Job {
                    input_ids: p.clone(),
                    params: sampling_params(max_tokens),
                    tx,
                },
            )
            .unwrap_or_else(|e| panic!("submit request {i}: {}", e.message));
        // #160: this request's own drive is bounded by work (progress and the
        // step budget), exactly like the batched arm.
        let mut bound = WorkBound::new(
            &engine,
            step_budget(p.len(), step_cap(max_tokens, n_ctx)),
            "the serial baseline",
        );
        let mut text = String::new();
        let mut done = None;
        while done.is_none() {
            engine.tick(model, tok).expect("tick");
            bound.step(&engine);
            loop {
                match rx.try_recv() {
                    Ok(StreamEvent::Text(t)) => text.push_str(&t),
                    Ok(StreamEvent::Finish { reason, tokens }) => {
                        done = Some(Reply {
                            text: std::mem::take(&mut text),
                            tokens,
                            reason,
                        });
                    }
                    Ok(StreamEvent::Err(e)) => panic!("request {i}: {}", e.message),
                    Err(_) => break,
                }
            }
        }
        out.push(done.expect("finished"));
    }
    (out, t0.elapsed().as_secs_f64())
}
/// The step budget for a real-model stepper loop (#158, shared by #160).
///
/// A legitimate run does one decode forward and one sample per answer token,
/// plus one prefill forward per chunk (with `MINFER_N_BATCH` low, one per
/// prompt token), so `MARGIN * (prompt + max_tokens + SLACK)` is far above
/// anything the engine can legitimately need. `MARGIN = 4` and `SLACK = 8` are
/// deliberately loose: this bound exists to catch an engine that *cannot*
/// terminate, never one that is merely slow, and a false negative hangs the
/// suite while a false positive is the flaky gate this ticket removes.
///
/// Measured on dgxspark (0.5B q4_0, GB10 host CPU, 2026-09-25): the warm
/// request (8-token prompt, `max_tokens = 4`) takes **4 steps** against a
/// budget of **80**, and the long one (120-token prompt, `max_tokens = 64`)
/// takes **64** against **768** — margins of 20x and 12x. The gates print both
/// counts on every run, so a workload that outgrows the budget says so.
const STEP_BUDGET_MARGIN: usize = 4;
const STEP_BUDGET_SLACK: usize = 8;
fn step_budget(prompt_tokens: usize, max_tokens: usize) -> usize {
    STEP_BUDGET_MARGIN * (prompt_tokens + max_tokens + STEP_BUDGET_SLACK)
}
/// #160: the answer tokens a request may legitimately produce, for
/// [`step_budget`]. An unbounded request (`max_tokens < 0`) ends on its
/// context bound, which is the cap the engine itself enforces.
fn step_cap(max_tokens: i64, n_ctx: usize) -> usize {
    if max_tokens < 0 {
        n_ctx
    } else {
        max_tokens as usize
    }
}
/// The per-step bound every stepper loop in this module shares (#158, made
/// the common shape by #160).
///
/// A `tick` that leaves the engine busy must have advanced
/// [`BatchEngine::work_units`]: if no forward ran, then some slot's `advance`
/// returned `Continue`, and `Continue` commits exactly one token (every other
/// `advance` outcome ends the run and takes it). A drive can violate that in
/// exactly two ways, and this type names the arm that catches each:
///
/// - the counter freezes while the engine stays busy — a **wedge**; [`step`]
///   panics on the step that wedged it (this is the arm `MINFER_TEST_TICK=wedge`
///   drives), or
/// - the counter keeps moving along a path that cannot terminate — only the
///   **step budget** catches that (the arm `MINFER_TEST_TICK=spin` drives).
///
/// Neither bound is a wall-clock number, so a loaded box runs the same steps
/// more slowly and still passes.
///
/// [`step`]: WorkBound::step
struct WorkBound<'a> {
    what: &'a str,
    budget: usize,
    steps: usize,
    work: u64,
}
impl<'a> WorkBound<'a> {
    fn new(engine: &BatchEngine, budget: usize, what: &'a str) -> Self {
        Self {
            what,
            budget,
            steps: 0,
            work: engine.work_units(),
        }
    }

    /// Record one step and assert it was honest. Call it immediately after
    /// every `tick` whose `busy()` state the drive is about to re-check.
    fn step(&mut self, engine: &BatchEngine) {
        self.steps += 1;
        let now = engine.work_units();
        assert!(
            !engine.busy() || now > self.work,
            "{}: the engine is wedged — step {} left it busy without advancing \
             the work counter (still {})",
            self.what,
            self.steps,
            self.work
        );
        self.work = now;
        assert!(
            self.steps <= self.budget,
            "{}: {} steps exceeded the {}-step budget — the run is not \
             progressing toward completion",
            self.what,
            self.steps,
            self.budget
        );
    }
}
// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `batch.rs` (bucket B of the dead-code census — every
// test caller already lives in this module's subtree). The `cb.submit()` matches
// in `metal/tests.rs` are the Metal `CommandBuffer`, a different item.
// ────────────────────────────────────────────────────────────────────────────

impl BatchEngine {
    /// Slots without a request (their reservation and KV stay).
    ///
    /// Test-only (#239): driven by the `run_batched` helper and by
    /// `server::batch::tests::stall::{a_failed_decode_forward_answers_a_single_run_and_releases_its_slot,
    /// the_stall_answers_every_live_run_exactly_once}`.
    pub fn idle_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.run.is_none()).count()
    }

    /// Admit a single request: [`BatchEngine::admit`] with one job.
    ///
    /// Test-only (#239): driven by `server::batch::tests::slots::a_slot_snapshot_resumes_the_context_without_re_prefilling`
    /// and the chunked-prefill gates.
    pub fn submit(
        &mut self,
        model: &dyn ModelDef,
        tokenizer: &Tokenizer,
        job: Job,
    ) -> Result<usize, ApiError> {
        self.admit(model, tokenizer, vec![job])
            .into_iter()
            .next()
            .expect("one job in, one answer out")
    }

    /// E3: `(largest nt any prefill forward carried, prefill forwards run)`. The
    /// activation-memory bound is `max_nt`, so the gate asserts on this rather than
    /// on a claim about buffers.
    ///
    /// Test-only (#239): driven by
    /// `server::batch::tests::prefill::{a_chunked_prefill_answers_like_an_unchunked_one,
    /// a_long_prefill_keeps_another_slot_decoding}`.
    pub fn prefill_stats(&self) -> (usize, usize) {
        (self.prefill_max_nt, self.prefill_forwards)
    }

    /// B2/C5 S2: prompt tokens this engine has fed to prefills (see `prefill_fed`).
    ///
    /// Test-only (#239): driven by
    /// `server::batch::tests::slots::a_slot_snapshot_resumes_the_context_without_re_prefilling`.
    pub fn prefill_fed(&self) -> usize {
        self.prefill_fed
    }

    /// E3: decode steps run between the chunks of a prefill (0 with chunking off).
    ///
    /// Test-only (#239): driven by
    /// `server::batch::tests::prefill::a_long_prefill_keeps_another_slot_decoding`.
    pub fn interleaved_ticks(&self) -> u64 {
        self.interleaved_ticks
    }
}
