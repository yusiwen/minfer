# 0006. The KV storage format is a per-engine gate, not a process-wide global

- Status: Accepted
- Date: 2026-09-25
- Issues: #99, #42, #153

## Context

C4 introduced a quantized KV cache (`q8_0` cells) alongside `f32` and `f16`. The first cut kept
the choice in **one process-wide setting** that the C4 packed-cache gate set mid-run. That broke
the suite in a way that looked like a flaky test rather than a design fault: the `#[ignore]`d
real-model set, run in parallel, turned red — *on master too* — because a gate had switched the
process-wide format under other tests
([#99](https://github.com/yusiwen/minfer/issues/99); recorded in `ARCHITECTURE-EXECUTION-PLAN.md`
§C4). A process-wide value is also wrong on its face: one process can load two engines, and they
need not agree.

## Decision

The format is resolved **once per load, per engine**, and it is a *gate*: `MINFER_CACHE_TYPE` is
parsed strictly into `KvFormat { f32, f16, q8_0 }` (`src/graph/kvformat.rs`, the single
authority). There is deliberately **no** process-wide `kvformat` global.

The resolved answer is stored on the loaded engine (`ModelDef::kv_format` / `set_kv_format`), and
travels to the two places that need it:

- the graph, through `CParams::kv_format` — part of the reuse identity, because it changes each KV
  node's stored width (`KvcacheMeta::row_elems`);
- the kernels, through `GraphAllocator::set_kv_format`.

An unknown value fails the load on **every** device. A `q8_0` region additionally requires the
backend to answer `BackendCaps::reads_packed_kv`; `ensure_kv` refuses a packed width where that
capability is absent.

## Alternatives considered

- **Keep the process-wide global.** Rejected on measured evidence. With the global in place, master
  `6d53649` ran **5 passed / 7 failed** in parallel against **12 passed / 0 failed** serially, and
  every failure named the same symptom: `KV region for layer 0 was allocated with 57600 elements but
  15300 are requested` — the two numbers differ by exactly the Q8_0 packing ratio (3.76×). The
  record re-measured it at `a756419` as **19 / 9** parallel against **28 / 0** serial, and a
  deliberate mutation that re-introduced the global gave **20 / 8**. A `Drop` guard could not close
  it, because the hazard is a *concurrent* graph build; the parallel red set was also what made the
  E4 S2 gate run ambiguous.
- **Infer the format from the file or the device, silently.** Rejected by the rule that governs the
  rest of the engine: the storage format is a gate, never a guess. CUDA used to read anything that
  was not `f16` as `f32` — exactly the silent fallback the ticket forbids — and issue #42's
  acceptance states it: unsupported backends refuse the setting instead of silently falling back.
- **Make the format a build-time constant.** Rejected because the right format depends on the model
  size and the device (`kvformat::auto_device_format`: `f16` for the 7B class, `f32` for small
  models, never packed on the CPU), so it cannot be fixed at compile time.
- **A serial-only entry point for the gate set** (issue #99's other option) landed as a
  *complement* — `scripts/real_model_gates.sh` runs the device set serially — but the record names
  the per-engine change "the real one, not the guard".

## Consequences

- The format became reproducible: same `MINFER_CACHE_TYPE`, same engine, same width — independent
  of what else the process is doing.
- On CUDA the layout tag is part of the **captured-graph identity**: `graph_replay_step` refuses
  an exec whose recorded `kv_layout` moved, exactly as it does for a `pool_gen` change.
- A backend's inability to read a packed region is now a *capability answer* in the registry
  (`reads_packed_kv`), not a special case in the attention code — which is what let Metal enable
  it later without touching the seam (ADR-0005).
- Cost accepted: a KV session file must carry its element type in the header, and a foreign or
  truncated one is refused rather than interpreted (`docs/KV-CACHE-DESIGN.md` §4).

## References

- `docs/KV-CACHE-DESIGN.md` §3 — the current contract for the format gate.
- `ARCHITECTURE-EXECUTION-PLAN.md` — the **#99 record** (the parallel-vs-serial failure, the
  3.76× packing-ratio diagnosis, the re-measurement and the re-introduction mutation), and §C4 for
  the surrounding C4 context.
- Commits: `a8f997a` (2026-09-25, "fix(kv): make the KV storage format per engine, not a process
  global (#99)") and the record `0dcf5c8`.
