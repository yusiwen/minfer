//! Issue #185's **one thread at a time** contract, **narrowed by issue #188** to
//! the one device path that still shares the context stream.
//!
//! ## What #185 saw, and what #188 changed
//!
//! `CudaState` is a process-wide singleton ([#64]): one device, one name-keyed
//! weight registry. Before #188 it also meant one **stream** and one
//! **capture window**, opened with `cudaStreamBeginCapture(stream,
//! cudaStreamCaptureModeGlobal)`. Under Global semantics another thread's
//! capture-unsafe driver call belongs to that window: it either invalidates the
//! capture (`cudaErrorStreamCaptureInvalidated`, 901) or faults inside the
//! driver. The recorded SIGSEGV was `register_weight`'s blocking `cudaMemcpy`
//! (`cuMemcpyHtoD_v2`, i.e. the legacy *null* stream) landing while another
//! thread sat in `graph_end_capture_to_exec → cudaGraphInstantiate` ([#188]).
//!
//! #188 made the **stream** (and the capture window, and the activation
//! scratches) travel with the backend instance: every `CudaBackend` creates its
//! own `cudaStreamNonBlocking` stream (`CudaState::create_stream`), binds it for
//! the duration of each device operation (`crate::cuda::bind_stream`), and opens
//! capture with `cudaStreamCaptureModeThreadLocal`. Weight registration no longer
//! uses a blocking `cudaMemcpy` at all — it queues the H2D copy on the context
//! stream and waits on that stream.
//!
//! ## The narrowed guard
//!
//! What remains process-wide is the **context stream** an *unbound* caller gets:
//! a caller that reaches `CudaState`'s launch/copy helpers without a
//! `CudaBackend` to bind. In production exactly one such path survives —
//! [`CudaState::layer_gpu`](crate::cuda::CudaState::layer_gpu), the legacy
//! per-layer path (the graph path and `main` do not drive it) — and it drives
//! the context-keyed `buf_*` activation scratches. Two threads inside *that*
//! path would overwrite one another's scratch.
//!
//! So this module is now the exclusion for **the legacy unbound path only**:
//!
//! - **Covered** (takes [`enter`]): `CudaState::layer_gpu`.
//! - **No longer covered** (the #188 fix removed the call): a device graph
//!   execution (`BackendScheduler::execute`) and `register_cuda_weight`. Both are
//!   per-instance or stream-ordered now, and the concurrent device gate
//!   (`graph::cuda_backend::tests::
//!   two_cuda_engines_forward_concurrently_and_stay_bitwise_identical`) asserts
//!   the positive property — two threads run at once and stay bitwise correct —
//!   that replaces this module's refusal.
//! - **Never covered, still unbound**: direct `CudaState` scratch calls
//!   (`layer_gpu`'s siblings `upload_hidden`/`download_logits`/…), all
//!   `#[allow(dead_code)]` legacy surface. They are single-threaded by
//!   construction (dead code); naming them here is the honest scope.
//!
//! The module stays **pure and feature-independent** so the CPU CI job executes
//! its test even though a hosted runner has no GPU — the same reason
//! `models::weight_reg::cuda_weight_reg` keeps its decision pure. The
//! `cuda`-gated callers are the only production users.
//!
//! [#64]: https://github.com/yusiwen/minfer/issues/64
//! [#185]: https://github.com/yusiwen/minfer/issues/185
//! [#188]: https://github.com/yusiwen/minfer/issues/188

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Mutex;
use std::thread::ThreadId;

/// The thread currently inside the legacy unbound device path and what it is
/// doing. `None` means the path is free.
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

/// Exclusive, per-thread-re-entrant ownership of the legacy unbound device path.
/// Dropping it releases the path; a panicking holder releases it on unwind, and a
/// poisoned mutex is recovered rather than wedging every later device user (the
/// slot is a single `Option`, so there is no inconsistent state to recover from).
///
/// `!Send` on purpose: the thread-local depth is what makes re-entrancy work, so
/// a token must be dropped on the thread that took it.
pub struct DeviceEntry {
    _not_send: PhantomData<*const ()>,
}

/// Enter the legacy unbound device path.
///
/// `what` names the operation for the refusal message (a `&'static str`, so the
/// message needs no allocation on the hot path).
///
/// Returns `Err` — **without touching the driver** — when another thread is
/// already inside it. The same thread may re-enter; the outermost drop releases
/// it.
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
            "minfer: refusing to enter the legacy unbound CUDA device path ({what}): another \
             thread is already inside it ({other}). This path reaches `CudaState`'s launch/copy \
             helpers without a `CudaBackend` to bind a stream, so it shares the context stream's \
             `buf_*` activation scratches with every other unbound caller. The graph path and \
             weight registration are per-instance/stream-ordered since issue #188 and do not take \
             this guard; only the legacy `layer_gpu` path does. Run one unbound caller at a time.",
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

    /// The narrowed guard's acceptance: a **second thread** is refused, the
    /// message names the holder, the mechanism (the unbound context stream's
    /// scratches) and the issue that narrowed the scope, and the entry is
    /// released again when the holder drops.
    ///
    /// Mutation evidence (rule 3 of the gate contract): deleting the
    /// `Some(holder) if holder.thread != me` arm — i.e. letting every thread in —
    /// makes this test fail at `must be refused`.
    ///
    /// Both halves live in **one** test on purpose: the guard is a process global,
    /// so two tests running in parallel (libtest's default) would refuse each
    /// other — the very property being asserted.
    #[test]
    fn the_legacy_unbound_path_is_exclusive_across_threads_and_re_entrant_on_one() {
        // ── the refusal half ────────────────────────────────────────────────
        let held = enter("the first thread's layer pass").expect("the path starts free");

        let (tx, rx) = std::sync::mpsc::channel::<Option<String>>();
        let other = std::thread::spawn(move || {
            // `.err()` drops the token on the second thread either way, so a bug
            // that let it in would not leak the entry past this test.
            let reason = enter("a second thread's layer pass")
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
            reason.contains("a second thread's layer pass"),
            "the refusal must name what was refused: {reason}"
        );
        assert!(
            reason.contains("the first thread's layer pass"),
            "the refusal must name the operation already inside: {reason}"
        );
        // #188 narrowed the scope: the message must name the unbound context
        // stream as the mechanism, not "CudaState is a process-wide singleton"
        // (that reading is gone with per-instance streams).
        assert!(
            reason.contains("unbound"),
            "the refusal must state the narrowed mechanism: {reason}"
        );
        assert!(
            reason.contains("layer_gpu"),
            "the refusal must name the only path that still needs it: {reason}"
        );
        assert!(
            reason.contains("#188"),
            "the refusal must point at the narrowing issue: {reason}"
        );

        // ── the re-entrancy half ────────────────────────────────────────────
        // The same thread nests freely …
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
