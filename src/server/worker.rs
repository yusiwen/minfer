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

    /// A respawn puts the status back before its first request can arrive.
    pub fn mark_running(&self) {
        self.code.store(RUNNING, Ordering::SeqCst);
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

/// #731 A: how many times a worker that ended with its resources intact is
/// restarted before the server disables instead.
pub const MAX_WORKER_RESTARTS: u32 = 5;

/// The backoff before restart `attempt` (1-based): 100, 200, 400, 800, 1600 ms,
/// capped at 2 s. Pure, so the policy is unit-tested (#731 A).
pub fn restart_backoff_ms(attempt: u32) -> u64 {
    const BASE_MS: u64 = 100;
    const CAP_MS: u64 = 2_000;
    BASE_MS
        .saturating_mul(1u64 << attempt.saturating_sub(1).min(5))
        .min(CAP_MS)
}

/// A stopped worker is restartable only when the loop **returned** its resources
/// (the #196 bound) and only below the cap. A panic unwound past the return, so
/// the model and the queue are gone — that exit disables, it cannot restart.
pub fn can_restart(exit: WorkerExit, restarts: u32, max_restarts: u32) -> bool {
    exit == WorkerExit::Stalled && restarts < max_restarts
}

/// The restart policy, injectable so a gate can run a zero-delay, two-attempt
/// version of the production sequence.
#[derive(Clone, Copy)]
pub struct RestartPolicy {
    pub max_restarts: u32,
    pub backoff: fn(u32) -> u64,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: MAX_WORKER_RESTARTS,
            backoff: restart_backoff_ms,
        }
    }
}

/// The supervisor's view of a worker incarnation.
pub type WorkerHandle = std::thread::JoinHandle<WorkerResources>;
/// Respawn a worker from the resources the previous one handed back.
pub type WorkerSpawner = Box<dyn FnMut(WorkerResources) -> WorkerHandle + Send>;

/// Watch the worker and, when it can be restarted, restart it with a bounded
/// backoff — otherwise publish the stop and disable (#731 A).
///
/// One `worker.stopped` event per death, one `worker.restarted` per restart, and
/// one `worker.disabled` when the server gives up (a panic, or the cap). The
/// queue is answered with the named refusal on every death, so a queued request
/// never outlives the loop that would have run it.
pub fn run_supervisor(
    first: WorkerHandle,
    mut respawn: WorkerSpawner,
    status: Arc<WorkerStatus>,
    metrics: Arc<ServerMetrics>,
    done: Option<std::sync::mpsc::Sender<()>>,
    policy: RestartPolicy,
) {
    let mut handle = first;
    let mut restarts: u32 = 0;
    loop {
        // `join` distinguishes a panic (`Err`) from the loop's own return value.
        let (exit, mut resources) = match handle.join() {
            Ok(res) => (res.exit, Some(res)),
            Err(_) => (WorkerExit::Panic, None),
        };
        let level = if exit == WorkerExit::Shutdown {
            super::log::Level::Info
        } else {
            super::log::Level::Error
        };
        status.mark(exit);
        metrics.worker_stops_total.fetch_add(1, Ordering::Relaxed);
        let (slots, device) =
            resources
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
        // named refusal a new request gets, and count it the way a queued refusal
        // is counted (left the queue, could not be placed).
        if let Some(r) = resources.as_mut() {
            let reason = status.refusal_message();
            let mut answered = 0u64;
            while let Ok(job) = r.job_rx.try_recv() {
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
                    "[server] worker stopped: answered {answered} queued job(s) with 503"
                ));
            }
        }
        if exit == WorkerExit::Shutdown {
            break;
        }
        if !can_restart(exit, restarts, policy.max_restarts) {
            super::log::event(
                super::log::Level::Error,
                "worker.disabled",
                &[
                    ("reason", exit.reason()),
                    ("restarts", &restarts.to_string()),
                ],
            );
            break;
        }
        let Some(resources) = resources else {
            // Unreachable: `can_restart` refuses `Panic`.
            break;
        };
        restarts += 1;
        let delay = (policy.backoff)(restarts);
        metrics
            .worker_restarts_total
            .fetch_add(1, Ordering::Relaxed);
        super::log::event(
            super::log::Level::Warn,
            "worker.restarted",
            &[
                ("attempt", &restarts.to_string()),
                ("after_ms", &delay.to_string()),
            ],
        );
        std::thread::sleep(std::time::Duration::from_millis(delay));
        status.mark_running();
        handle = respawn(resources);
    }
    // The server's bounded shutdown wait (F8) is for the *final* stop, not a
    // restart.
    if let Some(done) = done {
        let _ = done.send(());
    }
}

#[cfg(test)]
mod tests;
