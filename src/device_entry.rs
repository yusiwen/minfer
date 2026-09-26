//! Issue #185: the **one thread at a time** contract of the process-wide CUDA
//! device layer, as a checked invariant instead of a comment.
//!
//! `CudaState` is a process-wide singleton ([#64]). Its stream, its **capture
//! window** (`cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)`), its
//! device buffer pool, its MMQ/positions memos and its captured-graph execs are
//! shared by every thread in the process. `cudaStreamCaptureModeGlobal` in
//! particular means *another* thread's driver call is not capture-safe: it either
//! invalidates the capture (`cudaErrorStreamCaptureInvalidated`, 901 — the
//! failures [#185] observed) or, as the gdb backtrace of the SIGSEGV shows,
//! faults inside the driver (`cuMemcpyHtoD_v2`) while `register_weight` copies a
//! weight with a plain blocking `cudaMemcpy` from a thread that is not the one
//! holding the capture.
//!
//! Thus: **two threads inside the device path at once is undefined behaviour**,
//! not a performance question. The supported configuration is one thread at a
//! time — `scripts/cuda_test.sh`, i.e. `--test-threads=1`, for the device suite.
//!
//! This module is the guard that says so. [`enter`] takes a process-wide,
//! re-entrant-per-thread token; a second *thread* is refused with the reason and
//! the remedy, before any driver call it would have raced. The device path calls
//! it at the two chokepoints the crash evidence names: the scheduler's
//! [`execute`](crate::graph::scheduler::BackendScheduler::execute) (which owns the
//! capture window) and `CudaState::register_weight`'s caller
//! (`models::weight_reg::register_cuda_weight`).
//!
//! **It is a chokepoint, not a structural exclusion.** A caller that reaches
//! `CudaBackend::execute_node` / `synchronize` / `graph_replay_step`, or
//! `CudaState::register_weight` directly instead of through those two entry
//! points, still runs unguarded — as does `Drop for CudaBackend`'s frees (which
//! the existing `stream_guard` serializes unless the backend is mid-capture).
//! Making the whole device path structurally exclusive — per-instance streams and
//! capture contexts, or a lock every device call takes — is
//! [#188](https://github.com/yusiwen/minfer/issues/188). This guard is what makes
//! the *observed* crash a loud refusal instead of a segfault.
//!
//! The module is deliberately **pure and feature-independent** so the CPU CI job
//! executes its tests even though a hosted runner has no GPU — the same reason
//! `models::weight_reg::cuda_weight_reg` keeps its decision pure (see that
//! module's docs). The `cuda`-gated callers are the only production users.
//!
//! [#64]: https://github.com/yusiwen/minfer/issues/64
//! [#185]: https://github.com/yusiwen/minfer/issues/185

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Mutex;
use std::thread::ThreadId;

/// The thread currently inside the device path and what it is doing. `None`
/// means the path is free.
static DEVICE_ENTRY: Mutex<Option<DeviceHolder>> = Mutex::new(None);

thread_local! {
    /// Re-entrancy depth on the owning thread. Only the owner ever sees a
    /// non-zero value, so a foreign thread's `enter` always falls through to the
    /// process-wide slot (and is refused if the owner holds it).
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

#[derive(Clone, Copy)]
struct DeviceHolder {
    thread: ThreadId,
    what: &'static str,
}

/// Exclusive, per-thread-re-entrant ownership of the process-wide CUDA device
/// path. Dropping it (in the other direction) releases the path; a panicking
/// holder releases it on unwind, and a poisoned mutex is recovered rather than
/// wedging every later device user (the slot is a single `Option`, so there is no
/// inconsistent state to recover from).
///
/// `!Send` on purpose: the thread-local depth is what makes re-entrancy work, so
/// a token must be dropped on the thread that took it.
pub struct DeviceEntry {
    _not_send: PhantomData<*const ()>,
}

/// Enter the process-wide CUDA device path.
///
/// `what` names the operation for the refusal message (a `&'static str`, so the
/// message needs no allocation on the hot path).
///
/// Returns `Err` — **without touching the driver** — when another thread is
/// already inside the device path. The same thread may re-enter (nested
/// `execute` → `register_weight`, or a `DeviceEntry` held across several calls);
/// the outermost drop releases it.
pub fn enter(what: &'static str) -> Result<DeviceEntry, String> {
    if DEPTH.with(|d| d.get()) > 0 {
        DEPTH.with(|d| d.set(d.get() + 1));
        return Ok(DeviceEntry {
            _not_send: PhantomData,
        });
    }
    let me = std::thread::current().id();
    let mut slot = DEVICE_ENTRY.lock().unwrap_or_else(|e| e.into_inner());
    match *slot {
        Some(holder) if holder.thread != me => Err(format!(
            "minfer: refusing to enter the CUDA device path ({what}): another thread is already \
             inside it ({other}). `CudaState` is a process-wide singleton — one stream, one \
             capture window (`cudaStreamCaptureModeGlobal`), one device pool, one captured-graph \
             cache — so two threads inside the device path is undefined behaviour: the parallel \
             `#[ignore]`d device suite has segfaulted inside libcuda at `cuMemcpyHtoD_v2` during \
             weight registration against another thread's open capture window (issue #185). Run \
             the device suite serially: `scripts/cuda_test.sh`, or `cargo test --release \
             --features cuda -- --test-threads=1`.",
            other = holder.what
        )),
        _ => {
            *slot = Some(DeviceHolder { thread: me, what });
            DEPTH.with(|d| d.set(1));
            Ok(DeviceEntry {
                _not_send: PhantomData,
            })
        }
    }
}

impl Drop for DeviceEntry {
    fn drop(&mut self) {
        let depth = DEPTH.with(|d| d.get());
        if depth > 1 {
            DEPTH.with(|d| d.set(depth - 1));
            return;
        }
        DEPTH.with(|d| d.set(0));
        let mut slot = DEVICE_ENTRY.lock().unwrap_or_else(|e| e.into_inner());
        *slot = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #185 acceptance, the refusal half: a **second thread** is refused, the
    /// message names the holder and the issue, and the entry is released again
    /// when the holder drops.
    ///
    /// Mutation evidence (rule 3 of the gate contract): deleting the
    /// `Some(holder) if holder.thread != me` arm — i.e. letting every thread in —
    /// makes this test fail at `must be refused`.
    ///
    /// Both halves live in **one** test on purpose: the guard is a process global,
    /// so two tests running in parallel (libtest's default) would refuse each
    /// other — the very property being asserted. One test keeps the module's own
    /// two halves sequential.
    #[test]
    fn the_device_path_is_exclusive_across_threads_and_re_entrant_on_one() {
        // ── the refusal half ────────────────────────────────────────────────
        let held = enter("the first thread's forward").expect("the device path starts free");

        let (tx, rx) = std::sync::mpsc::channel::<Option<String>>();
        let other = std::thread::spawn(move || {
            // `.err()` drops the token on the second thread either way, so a bug
            // that let it in would not leak the entry past this test.
            let reason = enter("a second thread's forward")
                .err()
                .map(|e| e.to_string());
            tx.send(reason).expect("the observer is alive");
        });
        let reason = rx
            .recv()
            .expect("the observer answered")
            .expect("a second thread must be refused");
        other.join().expect("the observer thread");

        assert!(
            reason.contains("a second thread's forward"),
            "the refusal must name what was refused: {reason}"
        );
        assert!(
            reason.contains("the first thread's forward"),
            "the refusal must name the operation already inside: {reason}"
        );
        assert!(
            reason.contains("process-wide singleton"),
            "the refusal must state the mechanism: {reason}"
        );
        assert!(
            reason.contains("#185"),
            "the refusal must point at the evidence: {reason}"
        );
        assert!(
            reason.contains("--test-threads=1"),
            "the refusal must name the remedy: {reason}"
        );

        // ── the re-entrancy half ────────────────────────────────────────────
        // The same thread nests freely (an `execute` that registers a weight in
        // the same thread must not refuse itself) …
        let inner = enter("a nested same-thread entry").expect("re-entrant on the owner");
        drop(inner);
        // … and dropping the inner entry must not release the outer one.
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        std::thread::spawn(move || {
            tx.send(enter("foreign while the outer entry is held").is_err())
                .expect("alive");
        })
        .join()
        .expect("thread");
        assert!(
            rx.recv().expect("answered"),
            "dropping the inner entry must not release the outer one"
        );

        // ── release ─────────────────────────────────────────────────────────
        drop(held);
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        std::thread::spawn(move || {
            tx.send(enter("after the release").is_ok())
                .expect("the observer is alive");
        })
        .join()
        .expect("the observer thread");
        assert!(
            rx.recv().expect("the observer answered"),
            "a released device path must be enterable again"
        );
    }
}
