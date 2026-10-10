# 0031. Capture runs in thread-local mode, because Global lets a foreign thread's call join the window

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #188, #185

## Context

A CUDA graph capture window is global state in the driver's eyes: while it is open, a capture-unsafe
driver call made by **any** thread can belong to it. In the pre-#188 engine that class produced the
recorded SIGSEGV ([#185](https://github.com/yusiwen/minfer/issues/185)) — one thread at `cuMemcpyHtoD_v2`
inside `register_weight` while another was at `cuGraphInstantiateWithFlags` inside
`graph_end_capture_to_exec` — and, on the legacy null-stream configuration, a hard deadlock: a blocking
`cudaMemcpy` synchronizes with every blocking stream, including the one holding a capture window that
cannot complete until the host closes it.

`docs/CUDA-BACKEND-DESIGN.md` §2.4 states the choice and the probe that measured it
(`capture_window_on_one_thread_survives_a_weight_registration_on_another`, GB10 sm_121, 5 process runs per
cell, 90 s watchdog, 2026-09-27).

## Decision

**Every capture window is opened with `cudaStreamCaptureModeThreadLocal`** (`graph_begin_capture`), so
invalidation is scoped to the capturing thread and a weight registration on another thread is benign.
`MINFER_CUDA_CAPTURE_MODE=0|1|2` (relaxed / global / thread-local) overrides the mode; it exists **for the
probe's measurement, not for production**.

The mode is one half of the fix and the stream discipline is the other: since #188 the stream, the capture
window and the activation scratches are **per `CudaBackend` instance**, and weight registration is
stream-ordered on the context stream. Two engines can therefore forward — and capture — concurrently, and
the positive gate is
`models::qwen2::graph::tests::cuda_kv::two_cuda_engines_forward_concurrently_and_stay_bitwise_identical`
(two threads, two distinct streams, bitwise equal to their serial references).

## Alternatives considered

- **Global (`cudaStreamCaptureModeGlobal`).** Rejected, and it is the mode this decision exists to remove:
  it lets a *foreign* thread's driver call join the capture, which is the #185 SIGSEGV class. It does pass
  the probe's instance-stream cell (5/5), which is why the choice is not merely about that cell.
- **Relaxed (`cudaStreamCaptureModeRelaxed`).** Ruled out by direct measurement: with the structural fix
  in place it still returns `cudaErrorStreamCaptureInvalidated` (901) — the exact code the acceptance
  forbids — in **5/5** runs, and `cudaMalloc` inside the window also fails
  (`CUDA: failed to allocate 16384 bytes`).
- **The pre-#188 configuration (context/blocking stream + blocking `cudaMemcpy`).** Rejected on
  measurement, and this is the reading that decides the shape of the fix: that cell **hangs 5/5 in all
  three modes**, so the mode is not what fixes it — a stream-ordered copy on a non-blocking per-instance
  stream is.
- **Keep the mode configurable as a production knob.** Rejected by the document's own wording: the env
  override is documented as the probe's instrument, so a production reader is not invited to tune it.

## Consequences

- A capture window cannot be invalidated by another thread's registration, which is what made concurrent
  engines possible; the concurrent bitwise gate is the positive half of the evidence.
- The override is a measurement lever, which means the mode is *judgeable* rather than assumed — the same
  pattern as the launch-return injection lever (#162, ADR-0030).
- Cost accepted: thread-local keeps the **capturing thread's own** capture-unsafe calls fatal, so a
  mistake inside the capturing thread is still an error rather than a warning. That is preferred to
  relaxed, which trades that strictness for a 901 it cannot avoid.
- **Recorded tension:** §2.4's summary paragraph reads as though thread-local and relaxed both make the
  *historical shared-stream* setup safe, while its own table shows that cell hanging 5/5 in **all three**
  modes. The table is the measurement; this ADR follows it, and the summary's parenthetical is about the
  instance-stream cells.

## References

- `docs/CUDA-BACKEND-DESIGN.md` §2.4 — the probe, the table and the three readings.
- [#188](https://github.com/yusiwen/minfer/issues/188) — the per-instance streams;
  [#185](https://github.com/yusiwen/minfer/issues/185) — the SIGSEGV class;
  [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
