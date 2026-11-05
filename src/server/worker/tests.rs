//! #731's supervision state machine: a dead worker is classified, published and
//! counted, and the handler's refusal names it.

use super::*;

fn metrics() -> Arc<ServerMetrics> {
    Arc::new(ServerMetrics::default())
}

/// A panicking worker is `Panic`, not "shutdown": the thread's `Err` is the only
/// evidence the loop never returned.
#[test]
fn a_panicking_worker_is_marked_panic_and_counted() {
    let status = Arc::new(WorkerStatus::new());
    let m = metrics();
    let handle = std::thread::spawn(|| -> WorkerResources { panic!("worker boom") });
    supervise_worker(handle, status.clone(), m.clone());
    assert!(status.is_stopped(), "the status must publish the death");
    assert!(
        status.refusal_message().contains("unexpectedly"),
        "the refusal must name a panic, got {:?}",
        status.refusal_message()
    );
    assert_eq!(
        m.worker_stops_total.load(Ordering::Relaxed),
        1,
        "/metrics must count the transition"
    );
}

/// The loop's own return value carries its reason; `Stalled` must be published as
/// such, not flattened to a shutdown.
#[test]
fn a_stopped_loop_publishes_its_own_reason() {
    let status = Arc::new(WorkerStatus::new());
    let m = metrics();
    let resources = WorkerResources {
        model: crate::models::load_model(
            &crate::gguf::load_gguf_model(std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/toy/toy-qwen2-f32.gguf"
            )))
            .expect("toy gguf"),
        )
        .expect("toy model"),
        slots: Vec::new(),
        job_rx: tokio::sync::mpsc::channel(1).1,
        exit: WorkerExit::Stalled,
    };
    let handle = std::thread::spawn(move || resources);
    supervise_worker(handle, status.clone(), m.clone());
    assert!(status.is_stopped());
    assert!(
        status.refusal_message().contains("making progress"),
        "the refusal must name the stall, got {:?}",
        status.refusal_message()
    );
    assert_eq!(m.worker_stops_total.load(Ordering::Relaxed), 1);
}

/// A running worker refuses nothing.
#[test]
fn a_running_worker_has_no_refusal() {
    let status = WorkerStatus::new();
    assert!(!status.is_stopped());
    assert_eq!(status.refusal_message(), "");
}
