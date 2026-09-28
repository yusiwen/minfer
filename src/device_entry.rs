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
mod tests;
