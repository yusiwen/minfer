//! [#731]: worker supervision — a dead worker must be visible, never silent.
//!
//! `serve::run` spawns one worker thread. A panic inside a job is contained by
//! `run_job_isolated`, but a worker that dies any other way (a panic outside the
//! isolated call, an abort, the #196 no-progress bound) used to leave its
//! `job_rx` dropped with no signal: every later request got a bare
//! `503 "server shutting down"` from a send error.
//!
//! This module is the supervision point, and it is deliberately one point so the
//! two slices of the ticket stay separate:
//!
//! - **B (this landing)** — *disable*: the exit is classified, logged once,
//!   counted in `/metrics`, and published as a [`WorkerStatus`]; every later
//!   request is refused with a message that names the state.
//! - **A (follow-up)** — *restart*: the supervisor already receives the
//!   [`WorkerResources`] the loop hands back, so A respawns from them (with a
//!   bounded backoff) instead of dropping them, and keeps B's refusal as the
//!   window during a backoff or after the restart cap.
//!
//! [#731]: https://github.com/yusiwen/minfer/issues/731

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use super::chat::Job;
use super::metrics::ServerMetrics;
use super::slot::Slot;

/// Why the worker stopped. The loop returns `Shutdown` / `Stalled`; a panic is
/// observed through `JoinHandle::join`'s `Err` and never reaches the loop's
/// return value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerExit {
    /// All senders dropped — the server is draining.
    Shutdown,
    /// [#196]'s counted no-progress bound tripped.
    Stalled,
    /// The thread unwound (a panic outside `run_job_isolated`).
    Panic,
}

impl WorkerExit {
    pub fn reason(self) -> &'static str {
        match self {
            WorkerExit::Shutdown => "shutdown",
            WorkerExit::Stalled => "stalled",
            WorkerExit::Panic => "panic",
        }
    }

    fn code(self) -> u8 {
        match self {
            WorkerExit::Shutdown => STOPPED_SHUTDOWN,
            WorkerExit::Stalled => STOPPED_STALLED,
            WorkerExit::Panic => STOPPED_PANIC,
        }
    }
}

const RUNNING: u8 = 0;
const STOPPED_SHUTDOWN: u8 = 1;
const STOPPED_STALLED: u8 = 2;
const STOPPED_PANIC: u8 = 3;

/// Everything the worker owns. Handed back when its loop ends so the restart
/// slice (#731 A) can respawn with the same model and queue instead of
/// reloading; B drops it after recording the exit.
pub struct WorkerResources {
    /// Read by B: the stop event names the device the dead worker ran on, and the
    /// restart slice (A) respawns the loop with the same model.
    pub model: Box<dyn crate::models::ModelDef>,
    /// Read by B: the stop event reports how many slots the worker held.
    pub slots: Vec<Slot>,
    /// Read by B: the jobs still queued when the worker died are answered with the
    /// named refusal here (they would otherwise wait for a loop that is gone).
    /// A drains it after a restart instead.
    pub job_rx: tokio::sync::mpsc::Receiver<Job>,
    pub exit: WorkerExit,
}

/// The worker's liveness, read by every request handler and written only by the
/// supervisor. One atomic byte: a request either proceeds or is refused with the
/// reason the worker stopped.
pub struct WorkerStatus {
    code: AtomicU8,
}

impl Default for WorkerStatus {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerStatus {
    pub fn new() -> Self {
        Self {
            code: AtomicU8::new(RUNNING),
        }
    }

    /// Record the transition. One call per worker lifetime in B; A also resets it
    /// to `RUNNING` before a respawn.
    pub fn mark(&self, exit: WorkerExit) {
        self.code.store(exit.code(), Ordering::SeqCst);
    }

    pub fn is_stopped(&self) -> bool {
        self.code.load(Ordering::SeqCst) != RUNNING
    }

    /// The named refusal: never a bare "shutting down" from a closed channel.
    pub fn refusal_message(&self) -> &'static str {
        match self.code.load(Ordering::SeqCst) {
            RUNNING => "",
            STOPPED_SHUTDOWN => "the inference worker is down: the server is shutting down",
            STOPPED_STALLED => {
                "the inference worker is down: it stopped making progress (see the server log)"
            }
            _ => "the inference worker is down: it stopped unexpectedly (see the server log)",
        }
    }
}

/// Watch the worker's thread and record how it ended. Runs on its own thread so
/// `serve::run`'s shutdown wait is unchanged.
pub fn supervise_worker(
    handle: std::thread::JoinHandle<WorkerResources>,
    status: Arc<WorkerStatus>,
    metrics: Arc<ServerMetrics>,
) {
    // `join` distinguishes a panic (`Err`) from the loop's own return value; the
    // resources are kept for the restart slice, which is why they are not dropped
    // before the exit is classified.
    let (exit, resources) = match handle.join() {
        Ok(res) => (res.exit, Some(res)),
        Err(_) => (WorkerExit::Panic, None),
    };
    status.mark(exit);
    metrics.worker_stops_total.fetch_add(1, Ordering::Relaxed);
    let level = if exit == WorkerExit::Shutdown {
        super::log::Level::Info
    } else {
        super::log::Level::Error
    };
    let (slots, device) = resources
        .as_ref()
        .map_or(("0".to_string(), "none".to_string()), |r| {
            (
                r.slots.len().to_string(),
                r.model.device().name().to_string(),
            )
        });
    super::log::event(
        level,
        "worker.stopped",
        &[
            ("reason", exit.reason()),
            ("slots", &slots),
            ("device", &device),
        ],
    );
    // The queue outlives the loop: answer what is still waiting with the same
    // named refusal a new request gets, and count it the way a queued refusal is
    // counted (left the queue, could not be placed), so `queue_depth` stays
    // `requests_total - jobs_admitted_total`.
    if let Some(mut resources) = resources {
        let reason = status.refusal_message();
        let mut answered = 0u64;
        while let Ok(job) = resources.job_rx.try_recv() {
            let _ = job.tx.blocking_send(super::chat::StreamEvent::Err(
                super::types::ApiError::unavailable(reason),
            ));
            answered += 1;
        }
        if answered > 0 {
            metrics
                .jobs_admitted_total
                .fetch_add(answered, Ordering::Relaxed);
            metrics
                .jobs_dropped_total
                .fetch_add(answered, Ordering::Relaxed);
            super::log::warn(format_args!(
                "[server] worker down: answered {answered} queued job(s) with 503"
            ));
        }
        // B: disable. A replaces this drop with a bounded-backoff respawn from the
        // same resources (the tokenizer is cloned in `serve::run` for that).
        drop(resources);
    }
}

#[cfg(test)]
mod tests;
