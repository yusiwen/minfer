# 0008. GPU safety: bounded waits, no early return past a barrier, runtime device limits

- Status: Accepted
- Date: 2026-08-02

## Context

On 2026-08-02 an M4 Pro GPU (AGXG16X) hardware-hung at ~15:37:25: WindowServer's compositing
thread froze in `mtl_submit → IOGPU → AGXG16X` for 40 s with zero progress, every Metal client
blocked behind it (including new minfer instances stuck in `MTLCreateSystemDefaultDevice`), and the
machine required a forced shutdown. **minfer was the only active GPU workload at the time.**
The exact faulting kernel was never identified — the snapshots show only the post-freeze state —
but the review found three structural amplifiers and several latent landmines for other models
(`docs/GPU_SAFETY.md` §1).

The lesson was structural rather than about one kernel: the code could *turn a GPU fault into an
unbounded host block*, and it made assumptions about device limits that it could not justify.

## Decision

Four hard rules, each traceable to a defect found in that review, plus one standing rule for
device numbers.

1. **A wait is bounded and its status is checked.** `submit()` waits 10 s on its semaphore, not
   `DISPATCH_TIME_FOREVER`, and checks `MTLCommandBufferStatus` — a non-`Completed` buffer is an
   error, and a timeout reports "GPU hang" rather than continuing silently. `submit()` returns
   `Result<(), String>`, and callers handle it.
2. **No early return past a `threadgroup_barrier`.** Invalid heads must still reach every barrier
   (the fix runs the full loop with a dummy head index and skips only the output write via a
   `valid_head` flag), because a simdgroup that exits early while others wait is a permanent GPU
   deadlock.
3. **Device limits are queried at runtime, never guessed.** Threadgroup memory, maximum threads per
   threadgroup and alignment limits come from `MTLDevice` properties. A guessed "32 KB
   threadgroup-memory limit" was wrong; the queried value is the only acceptable input. A hardcoded
   number is legitimate only where a *kernel declaration* fixes it (a `float acc[256]` array size),
   and then the dispatched check uses the queried limit. On CUDA the same rule reads free memory from
   `cudaMemGetInfo`, kernel limits from `cudaFuncGetAttributes` and the shared-memory opt-in limit
   from `cudaDevAttrMaxSharedMemoryPerBlockOptin` — "no hardcoded SM/arch assumptions".
4. **A guard failure is loud.** An invariant violation reports the actual values; in the graph path
   it is an `Err` from `execute_node`, never a silent CPU fallback (ADR-0009).

## Alternatives considered

Every one of these is the pre-incident code, and each was rejected by the incident itself:

- **Wait forever and trust the driver** (`DISPATCH_TIME_FOREVER`, no status check). Rejected: a
  single GPU fault blocked minfer forever, and because Metal clients share the GPU it could stall
  WindowServer into a whole-machine freeze. That is what happened.
- **Return early before a barrier when a head is out of range** (`if (h >= nh) return;`). Rejected:
  with `nh % nk != 0` it deadlocks the GPU permanently. This is the one defect that is a *hang by
  construction*, not by fault.
- **Guess the device limits** (the hardcoded 32 KB threadgroup-memory figure). Rejected because it
  was simply wrong for the M4 Pro; the code now queries `max_threadgroup_memory_length()`.
- **Query a fixed device instead of the selected one.** Found on 2026-09-14, six weeks later:
  `cuda_shared_per_sm()` queried device 0 regardless of which device the backend had selected, which
  would mis-validate the shared-memory budget on a multi-GPU host. It is the same mistake one level
  up — a device-specific number assumed rather than asked for — and it was fixed on 2026-09-15 by
  making the gates query-driven (the T2 work).
- **Fall back to the CPU when a guard fails.** This was the legacy semantics: the whole-layer
  `layer_gpu` / `output_norm_gpu` entry points returned `false` to mean "do it on the CPU". It is
  rejected because a silent fallback turns a kernel-invariant violation into a performance change
  while the run continues — the failure disappears from the record (ADR-0009 carries the general
  rule; these entry points were deleted with the imperative forward in Phase 6).

## Consequences

- A hang becomes a bounded, diagnosable error: on error or timeout, the last 16 dispatch op labels
  are printed so the faulting kernel *family* can be identified. Trace recording is env-gated, so it
  costs nothing when off.
- The fallback semantics changed for the whole graph path: "the guard failed" is an error, not a
  slower path.
- The remaining route to whole-machine harm is narrow but not zero: `docs/GPU_SAFETY.md` keeps a
  *recurrence playbook* rather than pretending the risk is retired.
- The rules acquired two addenda within the fortnight, both from new kernel shapes rather than from
  a new incident: float4 and split-attention guards (2026-08-03) and the flash-attention family
  (2026-08-14). The rule is standing, not a one-time fix.
- On CUDA the same discipline extends to launch errors: a latched error is never attributed to
  whichever kernel just ran, and every `<<<>>>` reads its own error (#145/#147/#162). The pre-#188
  context stream with a blocking copy hangs 5/5 in all three capture modes, which is why the stream,
  the capture window and the activation scratches became per-`CudaBackend`-instance.
- Honest gap, recorded here because it is a property of the evidence: the full diagnostic report
  lives **outside** the repository (`~/macbook-gpu-hang-report-2026-08-02.md`), so the in-repo
  record is the summary in `docs/GPU_SAFETY.md`, and the faulting kernel was never identified.
- CI has no Metal device, so these rules are held by review and by Mac runs, not by a CI gate.

## References

- `docs/GPU_SAFETY.md` §1 (the incident), §2.1–§2.4 (the fixes, dated), §4 (the runtime-query rule),
  §5 (recurrence playbook).
- `docs/BACKEND-REGISTRY-DESIGN.md` §11.5 — the same bounded-and-loud discipline at the cross-backend
  staging boundary.
- `docs/ARCHITECTURE-EXECUTION-PLAN.md` §1 rule 4 — the rule restated as a plan constraint.

## See also

- ADR-0009 — the general rule that a failure is an error, never a fallback.
