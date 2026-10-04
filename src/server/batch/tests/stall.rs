//! Stall and failure handling: the injected failing forward and the answer each run gets exactly once.
//!
//! Split out of `src/server/batch/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// #151's deterministic injection: a model whose **batch forward fails**.
///
/// `forward_batch` is the only forward the batched engine calls, and it cannot
/// return an error — a real failure is a panic that `guarded_forward_batch`
/// turns into `ApiError::server` (every refusal inside
/// `Qwen2Graph::forward_batch`, e.g. the `position exceeds n_ctx` invariant, is
/// exactly that shape). This double reproduces it: it counts attempts so the
/// gate can see a retry, and panics with a fixed message, so the failure
/// reaches `tick` through the **real** guard and the real error path. No model
/// on disk is involved, so the gate runs in CI.
struct FailingForward {
    attempts: std::sync::atomic::AtomicUsize,
}
impl FailingForward {
    fn new() -> Self {
        Self {
            attempts: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Batch forwards attempted. 1 after the failure and still 1 once the runs
    /// have been answered; an unbroken retry makes it grow.
    fn attempts(&self) -> usize {
        self.attempts.load(std::sync::atomic::Ordering::SeqCst)
    }
}
impl ModelDef for FailingForward {
    fn forward(
        &self,
        _tokens: &[u32],
        _positions: &[usize],
        _n_out: usize,
        _n_ctx: usize,
    ) -> Vec<f32> {
        unreachable!("the batched engine calls forward_batch, never forward")
    }

    fn forward_batch(
        &self,
        _batch: &crate::graph::batch::Batch,
        _n_out: usize,
        _n_ctx: usize,
        _cache: &mut GraphCache,
    ) -> Vec<f32> {
        self.attempts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        panic!("deterministic decode-forward failure (#151's gate injection)");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn kv_format(&self) -> crate::graph::kvformat::KvFormat {
        crate::graph::kvformat::KvFormat::F32
    }

    fn set_kv_format(&mut self, _format: crate::graph::kvformat::KvFormat) {}

    fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens {
            eos: 0,
            im_end: None,
        }
    }

    fn n_layer(&self) -> usize {
        1
    }
    fn n_head_kv(&self) -> usize {
        1
    }
    fn n_embd_head(&self) -> usize {
        1
    }
    fn n_kv_embd(&self) -> usize {
        1
    }
    fn n_vocab(&self) -> usize {
        4
    }
    fn rope_style(&self) -> crate::vec_ops::RopeStyle {
        crate::vec_ops::RopeStyle::NonInterleaved
    }
    fn rope_params(&self) -> (f32, f32) {
        (10_000.0, 1.0)
    }
}
/// Put a live request on `idx` whose next token is already committed and
/// waiting for a decode forward — the state `tick`'s batch builder looks for.
/// `cached_tokens` is pre-filled so the gate can see the failure clear it.
fn install_pending_run(
    engine: &mut BatchEngine,
    idx: usize,
    tx: mpsc::Sender<StreamEvent>,
    tok: u32,
) {
    let params = sampling_params(8);
    let cfg = params.sampler_config();
    let mirostat = crate::sampler::MirostatState::new(params.mirostat_tau);
    let rng = StdRng::seed_from_u64(params.seed);
    engine.slots[idx].cached_tokens = vec![1, 2, 3];
    engine.slots[idx].run = Some(Run {
        tx,
        params,
        cfg,
        mirostat,
        grammar_state: None,
        rng,
        prev_tokens: Vec::new(),
        stop_bytes: Vec::new(),
        full: Vec::new(),
        emitted: 0,
        completion_tokens: 0,
        current_pos: 3,
        last_logits: vec![0.0; 4],
        needs_forward: Some(tok),
        live_on: false,
    });
}
/// Drain a run's channel and report `(errors, finishes, text chunks, closed)`.
/// `closed` is true when the sender is gone — the observable that the slot was
/// released and nothing further will arrive.
fn read_run_channel(rx: &mut mpsc::Receiver<StreamEvent>) -> (Vec<ApiError>, usize, usize, bool) {
    let (mut errs, mut finishes, mut texts) = (Vec::new(), 0usize, 0usize);
    loop {
        match rx.try_recv() {
            Ok(StreamEvent::Err(e)) => errs.push(e),
            Ok(StreamEvent::Finish { .. }) => finishes += 1,
            Ok(StreamEvent::Text(_)) => texts += 1,
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => return (errs, finishes, texts, true),
        }
    }
    (errs, finishes, texts, false)
}
/// #151: a failed decode forward answers the **one** run whose row it carried,
/// instead of leaving it stuck.
///
/// The defect is an infinite retry, so the gate bounds itself instead of
/// hanging: it drives `tick` a second time and asserts the forward was
/// attempted **once**. With the pre-fix early `return`, the run keeps
/// `needs_forward`, so the second `tick` attempts the same forward again
/// (attempts == 2) and the client never hears.
#[test]
fn a_failed_decode_forward_answers_a_single_run_and_releases_its_slot() {
    let model = FailingForward::new();
    let mut engine = BatchEngine::new(&model, 1, 32).expect("engine");
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(8);
    install_pending_run(&mut engine, 0, tx, 7);
    assert_eq!(engine.running_slots(), 1);

    let err = engine
        .tick(&model, &Tokenizer::empty())
        .expect_err("the forward failed, so the step must report it");
    assert_eq!(
        err.status, 500,
        "a step failure is a server error: {}",
        err.message
    );
    assert_eq!(err.error_type, "server_error");

    let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
    assert_eq!(errs.len(), 1, "exactly one error, not zero and not a retry");
    assert_eq!(finishes, 0, "a failed run must not be finished");
    assert_eq!(texts, 0, "a failed run must not emit text");
    assert_eq!(
        errs[0].status, 500,
        "the run's own error: {}",
        errs[0].message
    );
    assert_eq!(errs[0].error_type, "server_error");
    assert!(
        !errs[0].message.is_empty(),
        "the client must be told why, not just that"
    );
    assert!(closed, "the run's sender is dropped: the slot is released");
    assert!(
        engine.slots[0].run.is_none(),
        "the slot must be free after the failure"
    );
    assert!(
        engine.slots[0].cached_tokens.is_empty(),
        "a half-written forward's prefix must not be reused"
    );
    assert_eq!(engine.idle_slots(), 1);
    assert!(!engine.busy(), "the failed request is no longer in flight");

    let attempts_after_failure = model.attempts();
    assert_eq!(attempts_after_failure, 1, "one forward for the one run");
    engine
        .tick(&model, &Tokenizer::empty())
        .expect("nothing is pending, so the next step is a no-op");
    assert_eq!(
        model.attempts(),
        attempts_after_failure,
        "the loop must not retry the failed forward"
    );
}
/// #151: a failed decode forward answers **every** run whose row was in the
/// batch, and leaves a run whose row was not.
///
/// Four slots: three with a pending token (one batch of three rows) and one
/// live run already sampled (`needs_forward = None`). The failure must answer
/// the three and touch nothing on the fourth — the membership rule, and what
/// keeps the fix from blaming a run the forward never carried.
#[test]
fn a_failed_decode_forward_answers_every_row_in_the_batch() {
    let model = FailingForward::new();
    let mut engine = BatchEngine::new(&model, 4, 64).expect("engine");
    let mut rxs = Vec::new();
    for (slot, tok) in [(0usize, 11u32), (1, 12), (2, 13)] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(8);
        install_pending_run(&mut engine, slot, tx, tok);
        rxs.push(rx);
    }
    // Slot 3 is live but has no committed-but-unwritten token: its row is not
    // in the batch, so this batch's failure is not its error.
    let (quiet_tx, mut quiet_rx) = mpsc::channel::<StreamEvent>(8);
    install_pending_run(&mut engine, 3, quiet_tx, 14);
    engine.slots[3]
        .run
        .as_mut()
        .expect("run installed")
        .needs_forward = None;
    assert_eq!(engine.running_slots(), 4);

    let err = engine
        .tick(&model, &Tokenizer::empty())
        .expect_err("the three-row batch forward failed");
    assert_eq!(err.status, 500);

    for (slot, rx) in rxs.iter_mut().enumerate() {
        let (errs, finishes, texts, closed) = read_run_channel(rx);
        assert_eq!(errs.len(), 1, "slot {slot}: exactly one error");
        assert_eq!(errs[0].status, 500, "slot {slot}: {}", errs[0].message);
        assert_eq!(errs[0].error_type, "server_error", "slot {slot}");
        assert_eq!(finishes, 0, "slot {slot}: no Finish");
        assert_eq!(texts, 0, "slot {slot}: no text");
        assert!(closed, "slot {slot}: released");
        assert!(engine.slots[slot].run.is_none(), "slot {slot}: free");
        assert!(engine.slots[slot].cached_tokens.is_empty());
    }
    // The run outside the batch survives it: still live, still its own sender,
    // still no event and its prefix untouched.
    assert!(
        engine.slots[3].run.is_some(),
        "a run outside the failed batch must be left alone"
    );
    assert_eq!(engine.slots[3].cached_tokens, vec![1, 2, 3]);
    let (q_errs, q_finishes, q_texts, q_closed) = read_run_channel(&mut quiet_rx);
    assert_eq!((q_errs.len(), q_finishes, q_texts), (0, 0, 0));
    assert!(!q_closed, "slot 3's run is still connected");

    let attempts_after_failure = model.attempts();
    assert_eq!(
        attempts_after_failure, 1,
        "one forward for the three-row batch"
    );
    // Drop the untouched run so the no-op step below cannot try to sample it
    // (this gate's tokenizer has no vocabulary); the retry assertion is about
    // the forward, which is what would have repeated.
    engine.slots[3].run = None;
    engine
        .tick(&model, &Tokenizer::empty())
        .expect("nothing is pending, so the next step is a no-op");
    assert_eq!(
        model.attempts(),
        attempts_after_failure,
        "the loop must not retry the failed forward"
    );
    assert!(!engine.busy());
}
/// #196: the stall path answers **every** live run exactly once.
///
/// `serve_loop`'s no-progress bound knows the engine is wedged but not which
/// batch wedged it, so [`BatchEngine::fail_all`] is its `fail_batch` without a
/// row list. Three live runs and one idle slot: each live run gets exactly one
/// `500 the worker stalled` (never zero — #121/#151's dropped-or-never-answered
/// defect — and never a second), is taken so the slot is free and no retry can
/// repeat the wedged batch, and has its cached prefix cleared. The idle slot is
/// untouched. No model is involved, so this half runs in CI.
#[test]
fn the_stall_answers_every_live_run_exactly_once() {
    let model = FailingForward::new();
    let mut engine = BatchEngine::new(&model, 4, 64).expect("engine");
    let mut rxs = Vec::new();
    for slot in [0usize, 1, 3] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(8);
        install_pending_run(&mut engine, slot, tx, 20 + slot as u32);
        rxs.push((slot, rx));
    }
    assert_eq!(engine.running_slots(), 3);

    let e = ApiError::server(WORKER_STALLED_MESSAGE);
    let answered = engine.fail_all(&e);
    assert_eq!(answered, 3, "every live run is answered");
    assert!(!engine.busy(), "no run is left occupying a slot");
    assert_eq!(engine.idle_slots(), 4);

    for (slot, mut rx) in rxs {
        let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
        assert_eq!(errs.len(), 1, "slot {slot}: exactly one terminal error");
        assert_eq!(
            (finishes, texts),
            (0, 0),
            "slot {slot}: the error is the whole answer"
        );
        assert_eq!(errs[0].status, 500, "slot {slot}: {}", errs[0].message);
        assert_eq!(errs[0].error_type, "server_error", "slot {slot}");
        assert_eq!(errs[0].message, WORKER_STALLED_MESSAGE, "slot {slot}");
        assert!(
            closed,
            "slot {slot}: the sender is dropped, so the slot is released"
        );
        assert!(engine.slots[slot].run.is_none(), "slot {slot}: free");
        assert!(
            engine.slots[slot].cached_tokens.is_empty(),
            "slot {slot}: a wedged engine's prefix must not be reused"
        );
    }
    assert!(engine.slots[2].run.is_none(), "the idle slot was untouched");
    assert_eq!(
        engine.fail_all(&e),
        0,
        "an already-idle engine has nothing to answer"
    );
}
/// #196: the stall path answers every **queued but unadmitted** job too.
///
/// Ending `serve_loop` drops a queued [`Job`] — and with it the only
/// `Sender<StreamEvent>` its handler listens on, which the handler reads as a
/// *completed* empty answer (#121's silent drop). [`reject_queued`] answers the
/// worker's own deque and whatever is still in the channel with the same
/// terminal error, exactly once each, before the loop ends. CI: no model.
#[test]
fn the_stall_answers_every_queued_job_exactly_once() {
    let (job_tx, mut job_rx) = mpsc::channel::<Job>(8);
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut rxs = Vec::new();
    for i in 0..3 {
        let (tx, rx) = mpsc::channel::<StreamEvent>(8);
        rxs.push(rx);
        let job = Job {
            input_ids: vec![1, 2, 3],
            params: sampling_params(2),
            tx,
        };
        if i == 0 {
            pending.push_back(job); // already in the worker's deque
        } else {
            job_tx.blocking_send(job).expect("queue job"); // still in the channel
        }
    }

    let e = ApiError::server(WORKER_STALLED_MESSAGE);
    let answered = reject_queued(&mut pending, &mut job_rx, &e);
    assert_eq!(answered, 3, "every queued job is answered");
    assert!(pending.is_empty(), "the worker's deque is drained");

    for (i, mut rx) in rxs.into_iter().enumerate() {
        let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
        assert_eq!(errs.len(), 1, "job {i}: exactly one terminal error");
        assert_eq!(
            (finishes, texts),
            (0, 0),
            "job {i}: never a Finish and never text"
        );
        assert_eq!(errs[0].status, 500, "job {i}: {}", errs[0].message);
        assert_eq!(errs[0].error_type, "server_error", "job {i}");
        assert_eq!(errs[0].message, WORKER_STALLED_MESSAGE, "job {i}");
        assert!(closed, "job {i}: the sender is dropped after the answer");
    }
}
/// #171 deliverable D: the seam replaces #151's bespoke mock, on the real path.
///
/// The gate above needs [`FailingForward`], a test-local `ModelDef`, because a
/// failed batch forward has no other deterministic trigger. The
/// failure-injection seam makes that mock one environment variable: this gate
/// loads the **real** cached 0.5B, arms `MINFER_TEST_CALL_FAIL=forward_batch`
/// for the scope, and asserts the #151 contract still holds — exactly one
/// `500` to the run's own sender, the slot released, no retry — plus that the
/// chokepoint's observation counter proves the real forward entry was reached.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF). It is run serially
/// by `scripts/real_model_gates.sh`; the transcript lives in the #171 record
/// because CI cannot run it.
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn the_seam_fails_the_batch_forward_without_a_bespoke_mock() {
    let Some(path) = cached_model() else {
        eprintln!("[#171] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    // Arm the seam for exactly this scope; the guard restores the process
    // value on drop, so a panicking gate cannot leave the switch on.
    let _seam = crate::testfail::InjectionGuard::arm("forward_batch");
    crate::testfail::reset_checked();

    let mut engine = BatchEngine::new(&*model, 1, 64).expect("engine");
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(8);
    // One committed token waiting for its decode forward — the state `tick`'s
    // batch builder looks for, so no prefill is involved.
    install_pending_run(&mut engine, 0, tx, 7);
    assert_eq!(engine.running_slots(), 1);

    let err = engine
        .tick(&*model, &tok)
        .expect_err("the injected forward failure must be reported");
    assert_eq!(
        err.status, 500,
        "a step failure is a server error: {}",
        err.message
    );
    assert_eq!(err.error_type, "server_error");

    let (errs, finishes, texts, closed) = read_run_channel(&mut rx);
    assert_eq!(errs.len(), 1, "exactly one error, and no retry");
    assert_eq!(errs[0].status, 500, "{}", errs[0].message);
    assert_eq!((finishes, texts), (0, 0), "a failed run emits neither");
    assert!(closed, "the run's sender is dropped: the slot is released");
    assert!(engine.slots[0].run.is_none(), "the slot must be free");
    assert!(
        engine.slots[0].cached_tokens.is_empty(),
        "a half-written forward's prefix must not be reused"
    );
    assert!(!engine.busy());
    assert_eq!(
        crate::testfail::checked("forward_batch"),
        1,
        "the real batch-forward entry was reached exactly once before the seam fired"
    );

    // Disarm and prove the path is clean again: the next step is a no-op and
    // the observation counter does not move.
    drop(_seam);
    crate::testfail::reset_checked();
    engine
        .tick(&*model, &tok)
        .expect("nothing is pending, so the next step is a no-op");
    assert_eq!(crate::testfail::checked("forward_batch"), 0);
    eprintln!(
        "[#171] forward_batch seam: one 500, slot released, no retry, \
         {} observed forward entries",
        1
    );
}
