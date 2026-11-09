//! #731's supervision state machine: every death is classified, counted and
//! named, and a worker that ended with its resources intact is restarted with a
//! bounded backoff.

use super::*;
use std::sync::atomic::AtomicUsize;

fn metrics() -> Arc<ServerMetrics> {
    Arc::new(ServerMetrics::default())
}

/// One worker incarnation's leftovers; the toy model loads fast enough to build
/// one per test.
fn resources(exit: WorkerExit) -> WorkerResources {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/toy/toy-qwen2-f32.gguf"
    ));
    let gguf = minfer::gguf::load_gguf_model(path).expect("toy gguf");
    let model = minfer::models::load_model(&gguf).expect("toy model");
    WorkerResources {
        model,
        slots: Vec::new(),
        job_rx: tokio::sync::mpsc::channel(1).1,
        exit,
    }
}

/// The policy is a pure decision: only a loop that returned its resources
/// (`Stalled`) below the cap restarts.
#[test]
fn the_restart_policy_is_bounded_and_excludes_panics() {
    assert_eq!(
        [1u32, 2, 3, 4, 5].map(restart_backoff_ms),
        [100, 200, 400, 800, 1_600]
    );
    assert_eq!(restart_backoff_ms(9), 2_000, "the backoff is capped");
    assert!(can_restart(WorkerExit::Stalled, 0, 2));
    assert!(can_restart(WorkerExit::Stalled, 1, 2));
    assert!(!can_restart(WorkerExit::Stalled, 2, 2), "the cap stops it");
    assert!(
        !can_restart(WorkerExit::Panic, 0, 2),
        "a panic unwound past the return, so the resources are gone"
    );
    assert!(
        !can_restart(WorkerExit::Shutdown, 0, 2),
        "a drain is not a crash"
    );
}

/// A panicking worker is `Panic`, not "shutdown", and it disables rather than
/// restart (there is nothing left to restart from).
#[test]
fn a_panicking_worker_is_marked_panic_and_disabled() {
    let status = Arc::new(WorkerStatus::new());
    let m = metrics();
    let handle = std::thread::spawn(|| -> WorkerResources { panic!("worker boom") });
    run_supervisor(
        handle,
        Box::new(|_| unreachable!("a panic must not restart")),
        status.clone(),
        m.clone(),
        None,
        RestartPolicy::default(),
    );
    assert!(status.is_stopped(), "the status must publish the death");
    assert!(
        status.refusal_message().contains("unexpectedly"),
        "the refusal must name a panic, got {:?}",
        status.refusal_message()
    );
    assert_eq!(m.worker_stops_total.load(Ordering::Relaxed), 1);
    assert_eq!(m.worker_restarts_total.load(Ordering::Relaxed), 0);
}

/// The loop's own `Stalled` exit is restarted once; the restarted worker's
/// `Shutdown` then ends the supervision.
#[test]
fn a_stalled_worker_is_restarted_once_and_then_drains() {
    let status = Arc::new(WorkerStatus::new());
    let m = metrics();
    let spawned = Arc::new(AtomicUsize::new(0));
    let counter = spawned.clone();
    let first = std::thread::spawn(|| resources(WorkerExit::Stalled));
    let respawn: WorkerSpawner = Box::new(move |res| {
        counter.fetch_add(1, Ordering::SeqCst);
        let mut res = res;
        res.exit = WorkerExit::Shutdown;
        std::thread::spawn(move || res)
    });
    run_supervisor(
        first,
        respawn,
        status.clone(),
        m.clone(),
        None,
        RestartPolicy {
            max_restarts: 2,
            backoff: |_| 0,
        },
    );
    assert_eq!(spawned.load(Ordering::SeqCst), 1, "one restart expected");
    assert_eq!(m.worker_restarts_total.load(Ordering::Relaxed), 1);
    assert_eq!(m.worker_stops_total.load(Ordering::Relaxed), 2);
    assert!(
        status.refusal_message().contains("shutting down"),
        "the final exit is the drain, got {:?}",
        status.refusal_message()
    );
}

/// A worker that keeps ending `Stalled` is restarted until the cap, then
/// disabled — never looped forever.
#[test]
fn the_cap_disables_after_the_last_restart() {
    let status = Arc::new(WorkerStatus::new());
    let m = metrics();
    let first = std::thread::spawn(|| resources(WorkerExit::Stalled));
    let respawn: WorkerSpawner =
        Box::new(|_| std::thread::spawn(|| resources(WorkerExit::Stalled)));
    run_supervisor(
        first,
        respawn,
        status.clone(),
        m.clone(),
        None,
        RestartPolicy {
            max_restarts: 1,
            backoff: |_| 0,
        },
    );
    assert_eq!(m.worker_restarts_total.load(Ordering::Relaxed), 1);
    assert_eq!(m.worker_stops_total.load(Ordering::Relaxed), 2);
    assert!(
        status.refusal_message().contains("making progress"),
        "the last state is the stall, got {:?}",
        status.refusal_message()
    );
}

/// A running worker refuses nothing.
#[test]
fn a_running_worker_has_no_refusal() {
    let status = WorkerStatus::new();
    assert!(!status.is_stopped());
    assert_eq!(status.refusal_message(), "");
}
