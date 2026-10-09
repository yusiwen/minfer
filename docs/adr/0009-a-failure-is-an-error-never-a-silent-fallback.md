# 0009. A failure is an error, never a silent fallback

- Status: Accepted
- Date: 2026-08-21

## Context

The pre-graph engine chose a backend per layer *at run time*: `layer_gpu()` was decided inside the
forward loop, and a support limitation could therefore change the execution path mid-run without
anyone deciding it should. The guards of that era made the same move at a smaller scale — the
whole-layer `layer_gpu` / `output_norm_gpu` entry points returned `false` to mean "do this on the
CPU" — so a kernel whose assumptions did not hold became a *performance* change and the run
continued (`docs/COMPUTE-GRAPH-DESIGN.md` §1.1, `docs/GPU_SAFETY.md` §2.3).

That is the failure mode this decision removes: not "the CPU differs from the GPU" (the two paths
*are* allowed to differ by design — Core Convention 1), but "the engine quietly took a different
path than the one that was built".

The dates are worth stating precisely, because the rule has two: the wording enters the record on
2026-08-17, and it becomes structural on 2026-08-21 when the graph path replaces the imperative
forward and the graph's error contract becomes the engine's only semantics
(`e54070c`; the same day as the Phase-1 IR and the plan). This ADR is dated at the second, because
that is when the alternative stopped existing in the code.

## Decision

**Backend assignment is decided at build time, and a node a backend cannot execute is an error.**

- `BackendScheduler::assign_backends` resolves each node to a backend once, when the graph is
  built; `Backend::supports_op`/`supports_for` is asked there, not during execution.
- `Backend::execute_node` returns `Result<(), String>`. On a kernel-invariant violation it returns
  `Err` carrying the *actual values* (the offending dimension, the registered byte length, the shape
  it expected), not a boolean and not a fallback. The design doc states the same thing in the
  negative: "There is no 'delete the node and fall back to CPU' path."
- A backend that cannot run a node never runs it on another backend. The run stops and says why.

The rule is stated as the graph's §1.4 invariant "**Errors are errors**" and restated as the plan's
standing rule 2, "**No silent fallback**".

## Alternatives considered

- **Return `false` and fall back to the CPU** — the legacy semantics, and the alternative this ADR
  rejects. It hides a kernel-invariant violation behind a slope change: the run continues, and the
  only trace is a timing difference. It also makes the *contract* unverifiable, because the same
  code path legitimately produces different numbers on the two backends (ADR-0013), so a silent
  fallback is indistinguishable from a normal schedule.
- **Choose the backend per node at run time, best-available.** Rejected: the path then depends on
  the environment and on which capability happened to be compiled in, which makes a decode step
  unreproducible — and it is what made reuse impossible before the graph (ADR-0001).
- **Clamp, skip, or partially execute a node whose guard fails.** Rejected: a partially computed
  tensor is not a smaller answer, it is a wrong one, and the failure would surface later at a place
  that cannot name the cause.

Two points a reader should not have to infer:

- **Choosing CPU is not a fallback.** A device's all-or-nothing load gate may log the failing tensor
  and select the CPU backend **at load/assignment time**. That is a build-time decision made with the
  weights in hand, and it is the sanctioned way to end up on the CPU. What this ADR forbids is the
  same move *mid-run*, where the graph has already been assigned and something has gone wrong since.
- **No single bug is attributed to a silent fallback in the record.** The concrete defect class the
  guards were built for is the attention kernel's `hd == hd_kv` assumption, which produced
  out-of-bounds reads and was guarded on 2026-08-03 via `gpu_abort`
  (`docs/GPU_SAFETY.md` §3, finding "H1"). This ADR claims the *rule*, not a specific incident.

## Consequences

- The assignment becomes part of the graph's identity, not a runtime detail: `CParams.gpu`,
  `CParams.gpu_layers` and `CParams.kv_format` are in `GraphParams`, and a change to any of them
  forces a rebuild (ADR-0002). That is the price of determinism and the reason reuse is sound.
- A refusal is testable. The suite asserts the refusal itself, not a fallback: a packed KV width on
  a backend without `reads_packed_kv`, an unknown cache type on every device, an offload spelling
  that is not a block count, a session file whose header does not describe the run.
- Cost accepted: there is no graceful degradation. A device that can run *most* of a graph either
  gets the whole thing or the run fails with a named reason; "make it work by doing the rest on the
  CPU" is a deliberate non-feature, and the layer-offload plan
  (`docs/MEMORY-POLICY-DESIGN.md` §2) is the sanctioned way to ask for a mixed run.
- Every module that can be compiled out must still answer for itself: `Registry::build` is per
  `#[cfg]`, and a named backend that cannot run is *refused by name*, never dropped.

## References

- `docs/COMPUTE-GRAPH-DESIGN.md` §1.4 ("Errors are errors") and §7.4 (the error contract).
- `docs/GPU_SAFETY.md` §2.3 — the legacy `return false` semantics this replaced.
- `docs/ARCHITECTURE-EXECUTION-PLAN.md` §1 standing rule 2.
- `docs/BACKEND-REGISTRY-DESIGN.md` — the capability answers a build-time decision reads, and the
  "a named backend that cannot run is refused, never dropped" rule.
- Commits: `e54070c` (2026-08-21, the graph path replaces the imperative forward — the date this ADR
  carries), `c6edd20` (2026-08-17, the first recorded "silent CPU fallback" wording) and `a613b96`
  (2026-08-21, "never a silent"); the invariant is written down in `4aacc5e` (2026-09-14).

## See also

- ADR-0008 — the same rule applied to the GPU guards that followed the 2026-08-02 hang.
