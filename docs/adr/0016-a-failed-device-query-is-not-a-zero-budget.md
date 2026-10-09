# 0016. A failed device-memory query is not a zero budget

- Status: Accepted
- Date: 2026-09-24
- Issues: #122

## Context

E4's feasibility gate (2026-09-22) accounts memory before allocating it, and it needs the device's
free bytes. The query function **discarded the `cudaMemGetInfo` return code**:

```rust
unsafe { cudaMemGetInfo(&mut free, &mut total) }; // return code dropped
```

On failure `free` stayed 0, so the budget became `Some(0)` and every later device allocation was
refused with a message that named the wrong cause:

> out of Cuda memory: 553 MiB of weights + 0 MiB of pooled buffers + 0 MiB for this activation
> exceeds the 0 byte budget (0 MiB)

Instrumentation gave the real error — `rc=700 cudaErrorIllegalAddress`, "an illegal memory access was
encountered" — and that error is **sticky**: once a kernel in the context performs an illegal access,
every later CUDA call in the process, `cudaMemGetInfo` included, returns 700 until the process ends.
So one stray fault disabled the engine for the rest of the run, and the refusal blamed a budget that
had never been measured.

## Decision

**The query's outcome becomes a type, and only a *measured* zero refuses.**

- `allocplan::DeviceMemory { Reported { free, total }, QueryFailed { code, name }, NoDevice }`, with
  the pure `allocplan::budget_decision(explicit, &DeviceMemory) -> BudgetDecision { budget, note }`.
- A **reported** reading keeps `free / 4 * 3` byte for byte, and a genuine `free == 0` **still
  refuses** — the gate keeps its teeth.
- A **failed** query falls back to weights-only accounting, returns an unbounded sentinel, and prints
  a note naming the real error (`cudaErrorIllegalAddress (700)`) **once per process**.
- The registry read has the same shape: `weights_from_lock` recovers a *poisoned* lock (the registry
  is append-only) and says so rather than reporting zero.
- E5's `auto` **refuses** with the real CUDA error instead of fitting against a phantom budget, and
  points at `MINFER_GPU_MEM`.

## Alternatives considered

- **Treat any failure as a zero budget** — the defect itself, and the alternative this decision
  rejects. The record is precise about why: *"the refusal was correct given a 0 budget; the defect was
  that a failed query was indistinguishable from 'the device is full', and the message blamed the
  budget instead of naming the cause."* Mutation-checked: making `budget_decision` return `Some(0)` on
  `QueryFailed` fails the gate (**0 passed; 3 failed**).
- **Refuse to run on a failed query.** Explicitly rejected: the fallback *"lets the backend's own
  allocation be the authority — it reports the real error if the context is genuinely unusable —
  instead of turning a broken accounting query into a total outage."* A sticky error is not evidence
  that the device is out of memory.
- **Fail open in the accounting too** (the twin defect): `weights_bytes()` summed the registry through
  `.map(…).unwrap_or(0)`, so a **poisoned lock silently reported 0 weights** — the opposite error,
  equally invisible. Rejected and mutation-checked the same way.
- **A measured zero still refusing** is the alternative the decision **kept**: the gate
  `a_measured_zero_free_read_is_still_a_zero_budget` is the control that stops "failed ⇒ unbounded"
  from becoming "everything is unbounded".

## Consequences

- A sticky CUDA error no longer disables the engine for the rest of the process; the failure shows up
  where it belongs — at the allocation that actually fails.
- The metrics surface **never publishes a number that was not measured**: `MemoryReport::budget_is_bounded()`,
  and the snapshot omits the `minfer_memory_budget_bytes` / `headroom_bytes` gauges when the budget is
  the sentinel. A monitoring system cannot chart a fabricated budget.
- Measured effect on the `#[ignore]`d serial CUDA set: **5 passed / 14 failed with 26×**
  "exceeds the 0 byte budget" became **20 passed / 2 failed with 0×** that message.
- Accepted risk, recorded honestly: Metal was **not exercised** at the time (no Mac), and
  `DeviceMemory::NoDevice` is its arm — `auto` fits nothing there without `MINFER_GPU_MEM`. The
  unbounded sentinel is deliberately silent only for `NoDevice`.

## References

- `docs/MEMORY-POLICY-DESIGN.md` §1 — the current contract for memory accounting.
- `ARCHITECTURE-EXECUTION-PLAN.md` — "E4 record, S4 (#122, 2026-09-24) — a failed device query is not a
  zero budget", against "E4 record, S1 (2026-09-22) — account first, allocate second".
- `src/graph/allocplan.rs` — `DeviceMemory`, `budget_decision`; `src/graph/alloc.rs` — the gate and
  the report.
