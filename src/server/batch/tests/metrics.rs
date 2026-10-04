//! Published metrics: queue/running depth and the counters a served request moves.
//!
//! Split out of `src/server/batch/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// F8 (#51): `serve_loop` publishes the queue and running depth as it serves.
///
/// Two rounds, both driving the **real** serve loop with a real model:
///
/// 1. Two jobs on two slots: the feeder thread plays the HTTP handler's two
///    increments (`requests_total` on accept, `in_flight`), sends the jobs,
///    and watches the registry while the loop admits, steps and finishes
///    them. The gate asserts the running count was observed non-zero, that
///    every accepted job **left** the queue (`accepted - queue_depth == n`),
///    and that everything settles at zero.
///
///    `queue_depth` is deliberately **not** peak-asserted: `serve_loop` calls
///    `admit` on every iteration whether or not it is busy, so a job is
///    placed or rejected within one loop pass and its non-zero window is
///    shorter than a sampler's interval on dgxspark. Its arithmetic and its
///    saturation are gated purely in `server::metrics`, and the loop's *drain*
///    is gated here through `jobs_admitted_total`.
/// 2. Two jobs on **one** slot, both queued before the loop starts: the
///    "no idle slot" path is taken deterministically and its counter
///    (`jobs_dropped_total`) must fire — while the queue arithmetic still
///    balances, because a rejected job has also *left* the queue.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF).
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn serve_loop_publishes_the_queue_and_running_depth() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let Some(path) = cached_model() else {
        eprintln!("[f8] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let metric_source = |path: &str| -> Vec<u32> { tok.encode(path) };

    // --- round 1: two jobs, two slots, observed while running ---------------
    // 1024 rows over 2 slots = 512 a slot, far more than the 134 a 6-token
    // prompt asking for 128 tokens wants, so no admission has to repartition
    // the arena and both requests are served concurrently.
    let mut engine = BatchEngine::new(&*model, 2, 1024).expect("engine");
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let n = 2usize;
    // 128 tokens so the running window is seconds wide — far wider than the
    // sampler's 1 ms interval: "watched it run" must not be a race.
    let prompts: Vec<Vec<u32>> = (0..n)
        .map(|i| metric_source(&format!("Say the number {}.", i + 1)))
        .collect();
    let feeder_metrics = metrics.clone();
    let feeder = std::thread::spawn(move || {
        let mut rxs = Vec::new();
        for p in prompts {
            let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
            rxs.push(rx);
            job_tx
                .blocking_send(Job {
                    input_ids: p,
                    params: sampling_params(128),
                    tx,
                })
                .expect("send job");
            // The HTTP handler's side of the contract: accepted + in flight.
            feeder_metrics.requests_total.fetch_add(1, Ordering::SeqCst);
            feeder_metrics.in_flight.fetch_add(1, Ordering::SeqCst);
        }
        let mut peak_running = 0u64;
        // #160: `FEEDER_POLL_BACKSTOP` is **only a backstop**, not the gate's
        // failure signal. The verdict below is `peak_running > 0` plus the
        // queue arithmetic. Since #196 a wedge in `tick` no longer hangs the
        // join: `serve_loop` ends itself on its counted no-progress bound, so
        // the feeder also leaves as soon as it sees `worker_stalled_total`
        // move — the assertion that follows needs no deadline to run, and the
        // mutation run fails in seconds. What the terminator bounds is a
        // worker that never settles and never trips the bound; a healthy run
        // breaks out on `drained` within milliseconds.
        let deadline = Instant::now() + FEEDER_POLL_BACKSTOP;
        loop {
            let s = feeder_metrics.snapshot();
            peak_running = peak_running.max(s.running);
            let drained = s.running == 0 && s.queue_depth == 0 && s.requests_total >= n as u64;
            // Only stop once the run has actually been *seen* running: with
            // 256-token answers the window is seconds wide, so this is an
            // assertion about the worker, not about sampler luck.
            if (peak_running > 0 && drained)
                || s.worker_stalled_total > 0
                || Instant::now() > deadline
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // Drain the responses; the worker's `blocking_send` must never wedge.
        // #196: a healthy `serve_loop` never trips its no-progress bound, so
        // every response here is `Text`/`Finish`; the errors are printed so a
        // mutation run shows the terminal answer each client actually got.
        let mut errors: Vec<String> = Vec::new();
        for mut rx in rxs {
            while let Some(ev) = rx.blocking_recv() {
                if let StreamEvent::Err(e) = ev {
                    errors.push(format!("{} {} ({})", e.status, e.message, e.error_type));
                }
            }
        }
        if !errors.is_empty() {
            eprintln!(
                "[f8] serve_loop answered {} error(s): {}",
                errors.len(),
                errors.join("; ")
            );
        }
        peak_running
    });

    super::serve_loop(&*model, &tok, job_rx, &mut engine, &metrics);
    let peak_running = feeder.join().expect("feeder");

    let s = metrics.snapshot();
    assert_eq!(
        s.worker_stalled_total, 0,
        "the worker tripped its counted no-progress bound: a healthy serve_loop \
         always advances work on a step that leaves the engine busy (#196)"
    );
    assert_eq!(
        s.requests_total - s.queue_depth,
        n as u64,
        "every accepted job left the queue"
    );
    assert_eq!(s.jobs_dropped_total, 0, "two slots served two jobs");
    assert_eq!(s.running, 0, "nothing is running once the loop drains");
    assert_eq!(s.worker_pending, 0);
    assert!(
        peak_running > 0,
        "the running count was never observed non-zero"
    );
    // The KV reading is live, not a startup snapshot.
    assert_eq!(s.kv.layers, model.n_layer() as u64);
    assert!(s.kv.region_bytes > 0);

    // --- round 2: two jobs, one slot, both queued up front ------------------
    let mut one = BatchEngine::new(&*model, 1, 512).expect("engine");
    let one_metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (tx1, rx1) = mpsc::channel::<Job>(64);
    for p in [metric_source("Alpha"), metric_source("Beta")] {
        let (tx, _rx) = mpsc::channel::<StreamEvent>(1024);
        tx1.blocking_send(Job {
            input_ids: p,
            params: sampling_params(2),
            tx,
        })
        .expect("queue job");
        one_metrics.requests_total.fetch_add(1, Ordering::SeqCst);
    }
    drop(tx1); // no more senders: the loop drains and exits
    super::serve_loop(&*model, &tok, rx1, &mut one, &one_metrics);
    let s1 = one_metrics.snapshot();
    assert_eq!(
        s1.requests_total - s1.queue_depth,
        2,
        "a rejected job still left the queue"
    );
    assert_eq!(
        s1.jobs_dropped_total, 1,
        "the second job cannot fit one slot and must be counted as dropped"
    );
    assert_eq!(
        s1.worker_stalled_total, 0,
        "the worker tripped its counted no-progress bound on a run that must \
         simply serve one request and reject the other (#196)"
    );
    assert_eq!(s1.running, 0);
    assert_eq!(s1.worker_pending, 0);

    eprintln!(
        "[f8] serve_loop: peak running {peak_running}, layers {} region {} B owned {} \
         cells; one slot -> {} dropped",
        s.kv.layers, s.kv.region_bytes, s.kv.owned_cells, s1.jobs_dropped_total
    );
}
/// #160: the `serve_loop_publishes_the_queue_and_running_depth` feeder's poll
/// terminator, named so the number is justified where it is used.
///
/// It is **not** a gate's failure signal: the verdict is the downstream
/// `peak_running > 0` and the queue arithmetic. Since #196 the worker also
/// stops itself, so this is no longer the only way out of the join: the feeder
/// leaves as soon as `worker_stalled_total` moves (the mutation run fails in
/// seconds), and it leaves on the healthy `drained` condition within
/// milliseconds. It is also not load-sensitive: `peak_running` is observed
/// within milliseconds of the first admission on every run. The value is
/// generous only because it is the last resort for a worker that drains but
/// whose published metrics never settle.
const FEEDER_POLL_BACKSTOP: std::time::Duration = std::time::Duration::from_secs(120);
/// #158: drive `engine` to idle with a bound on *work*, not wall-clock seconds.
///
/// Replaces the absolute deadlines this gate used (`Instant::now() +
/// Duration::from_secs(120/180)`), which a slow box could exceed while a
/// healthy engine ran on. Every `tick` that leaves the engine busy must
/// advance [`BatchEngine::work_units`] — a wedged engine (one whose `tick`
/// returns without forwarding or committing) trips that assertion on the
/// stalling step — and the step budget backstops an engine that keeps "moving"
/// along a path that cannot terminate. Neither bound is a wall-clock number, so
/// the verdict is a property of the code. Returns the number of steps it took.
fn drive_by_work(
    engine: &mut BatchEngine,
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    rx: &mut mpsc::Receiver<StreamEvent>,
    budget: usize,
    what: &str,
) -> usize {
    let mut bound = WorkBound::new(engine, budget, what);
    while engine.busy() {
        engine.tick(model, tokenizer).expect("tick");
        // The response channel is bounded, so every step's frames are drained:
        // a full channel would block the worker's `blocking_send` and stall the
        // step before the work assertion could see it.
        while rx.try_recv().is_ok() {}
        bound.step(engine);
    }
    bound.steps
}
/// F8 (#51): the published occupancy and running counts are a **live**
/// reading, not a startup snapshot — they move as requests are served.
///
/// Real-model gate (`#[ignore]`, like the rest of this module's measurement
/// tests: CI has no cached GGUF). Run it with
/// `cargo test --release --bin minfer -- --ignored --test-threads=1`.
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn published_metrics_move_as_requests_are_served() {
    let Some(path) = cached_model() else {
        eprintln!("[f8] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let n_slots = 2usize;
    let n_ctx = 128usize;
    let mut engine = BatchEngine::new(&*model, n_slots, n_ctx).expect("engine");
    let metrics = crate::server::metrics::ServerMetrics::new();

    // Before any forward: sequences are reserved, but no KV region exists yet
    // (the arena is sized by the first `alloc_graph`), so the layers gauge is
    // genuinely 0 — the reading tracks the arena, it does not assume one.
    engine.publish_metrics(&metrics);
    let cold = metrics.snapshot();
    assert_eq!(cold.kv.layers, 0, "no graph built yet, so no KV region");
    assert_eq!(cold.kv.sequences, n_slots as u64);
    assert_eq!(cold.running, 0);

    // Serve one request a step at a time and watch the numbers move.
    let prompt: Vec<u32> = (1..=8).collect();
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
    engine
        .submit_on(
            &*model,
            &tok,
            0,
            Job {
                input_ids: prompt.clone(),
                params: sampling_params(4),
                tx,
            },
        )
        .expect("submit");
    engine.publish_metrics(&metrics);
    let warm = metrics.snapshot();
    assert_eq!(
        warm.kv.layers,
        model.n_layer() as u64,
        "the first forward allocates every layer's KV region"
    );
    assert_eq!(warm.kv.rows, n_ctx as u64);
    assert!(warm.kv.region_bytes > 0, "a live arena has bytes");
    assert_eq!(warm.running, 1, "one slot holds a live request");
    // The admission prefilled the prompt, so those cells are already owned —
    // the gauge tracks the store, it does not wait for a decode step.
    assert!(
        warm.kv.owned_cells >= prompt.len() as u64,
        "the prefill wrote the prompt's {} cells (got {})",
        prompt.len(),
        warm.kv.owned_cells
    );

    // Step until the request finishes; `running` must fall back to 0 and the
    // owned-cell gauge must have grown. The bound is on work, not seconds
    // (#158): a loaded box runs the same steps more slowly and still passes,
    // while a wedged engine stops moving the counter and trips the assertion.
    let steps = drive_by_work(
        &mut engine,
        &*model,
        &tok,
        &mut rx,
        step_budget(prompt.len(), 4),
        "the warm request",
    );
    eprintln!("[f8] warm request: {steps} step(s)");
    assert!(!engine.busy(), "the warm request completed");
    engine.publish_metrics(&metrics);
    let done = metrics.snapshot();
    assert_eq!(done.running, 0, "no live request is running");
    assert!(
        done.kv.owned_cells >= warm.kv.owned_cells,
        "the run wrote at least the prompt's cells ({} -> {})",
        warm.kv.owned_cells,
        done.kv.owned_cells
    );
    assert_eq!(done.kv.layers, warm.kv.layers);

    // The snapshot is exactly what `/metrics` renders, so the endpoint sees
    // the live numbers too.
    let text = metrics.render();
    assert!(text.contains(&format!("minfer_kv_layers {}\n", model.n_layer())));
    assert!(text.contains(&format!("minfer_kv_rows {n_ctx}\n")));
    assert!(text.contains("minfer_requests_running 0\n"));

    // --- growth and release: the same gauges across an arena repartition -----
    // Issue #51's acceptance asks the metrics to be correct while slots are
    // "admitted, grown and released". A small arena forces a growth: the long
    // prompt needs more cells than its slot's share, so the engine releases an
    // **idle** slot's run and grows the asking one — both of which are visible
    // in `sequences` and `reserved_cells`.
    let mut growing = BatchEngine::new(&*model, 2, 256).expect("engine");
    let gm = crate::server::metrics::ServerMetrics::new();
    growing.publish_metrics(&gm);
    let g0 = gm.snapshot();
    assert_eq!(g0.kv.sequences, 2, "two slots hold a reservation up front");
    assert_eq!(g0.kv.layers, 0, "no arena before the first forward");
    assert_eq!(g0.kv.reserved_cells, 256, "128 cells a slot");
    assert_eq!(g0.running, 0);

    let long: Vec<u32> = (1..=120).collect();
    let (ltx, mut lrx) = mpsc::channel::<StreamEvent>(4096);
    growing
        .submit_on(
            &*model,
            &tok,
            0,
            Job {
                input_ids: long.clone(),
                params: sampling_params(64),
                tx: ltx,
            },
        )
        .expect("long submit");
    growing.publish_metrics(&gm);
    let g1 = gm.snapshot();
    assert_eq!(g1.running, 1, "the grown request is running");
    assert_eq!(g1.kv.layers, model.n_layer() as u64);
    assert!(
        g1.kv.sequences < g0.kv.sequences,
        "growing past its share releases an idle slot's run ({} -> {})",
        g0.kv.sequences,
        g1.kv.sequences
    );
    assert!(
        g1.kv.owned_cells >= 120,
        "the long prefill wrote its prompt ({} cells)",
        g1.kv.owned_cells
    );

    // Finish it, then ask for work again: the released slot re-reserves, so
    // the gauge comes back — and `running` is 0 at both ends. Same work bound
    // as the warm request, sized for this run's prompt and token budget.
    let steps = drive_by_work(
        &mut growing,
        &*model,
        &tok,
        &mut lrx,
        step_budget(long.len(), 64),
        "the long request",
    );
    eprintln!("[f8] long request: {steps} step(s)");
    assert!(!growing.busy(), "the long request completed");
    growing.publish_metrics(&gm);
    let g2 = gm.snapshot();
    assert_eq!(g2.running, 0);
    assert!(g2.kv.owned_cells >= 120);
    assert!(
        g2.kv.sequences >= g1.kv.sequences,
        "a released slot's reservation is gone until it is used again"
    );

    let short: Vec<u32> = (1..=8).collect();
    let (stx, _srx) = mpsc::channel::<StreamEvent>(1024);
    growing
        .submit_on(
            &*model,
            &tok,
            1,
            Job {
                input_ids: short,
                params: sampling_params(2),
                tx: stx,
            },
        )
        .expect("short submit");
    growing.publish_metrics(&gm);
    let g3 = gm.snapshot();
    assert_eq!(g3.running, 1);
    assert!(
        g3.kv.sequences > g2.kv.sequences,
        "the second slot's reservation is back ({} -> {})",
        g2.kv.sequences,
        g3.kv.sequences
    );

    eprintln!(
        "[f8] metrics: layers {} rows {} region {} B owned {} cells idle_slots {} \
         reserved_classes {}",
        done.kv.layers,
        done.kv.rows,
        done.kv.region_bytes,
        done.kv.owned_cells,
        done.kv.idle_slots,
        done.kv.reserved_classes
    );
    eprintln!(
        "[f8] growth/release: sequences {} -> {} -> {} -> {}, reserved_cells {} -> {}, \
         owned {} -> {}",
        g0.kv.sequences,
        g1.kv.sequences,
        g2.kv.sequences,
        g3.kv.sequences,
        g0.kv.reserved_cells,
        g1.kv.reserved_cells,
        g1.kv.owned_cells,
        g2.kv.owned_cells
    );
}
