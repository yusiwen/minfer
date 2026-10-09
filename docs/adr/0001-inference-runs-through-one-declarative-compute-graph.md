# 0001. Inference runs through one declarative compute graph

- Status: Accepted
- Date: 2026-08-21
- Issues: #57

## Context

Before the graph, a forward pass was computed imperatively: a hand-written `forward()` walked the
layers, dispatched each operation to a "whole layer" GPU fast path or a CPU fallback, and
allocated scratch buffers per step. Four costs followed, and the design doc states them: no
reuse across decode steps, no explicit place for fusion, a backend choice taken at run time
(`layer_gpu()` decided *inside* the loop, so a support limitation could silently change the
execution path mid-run), and roughly 620 lines of hand-written forward code per architecture
(`docs/COMPUTE-GRAPH-DESIGN.md` §1.1).

## Decision

Inference is `build ComputeGraph → assign backends → fuse → allocate → execute`. One graph is
built per `GraphParams` and reused across decode steps. The IR is the executable program; a
device supplies only an executor (`Backend::execute_node`) plus capability answers in its own
module, and per-node assignment is fixed at **build time**.

The device seam was later formalised so that **only the executor layer (L4) is polymorphic**:
`src/graph/backend.rs` + `src/graph/registry.rs` are the one place a device is selected, the
registry offers backends in a pinned priority order (Metal 300, CUDA 200, CPU 100), and backend
ids are compile-time-fixed and **appended, never renumbered** (`cpu = 0`, `metal = 1`,
`cuda = 2`) because an id is part of the KV-session file format.

## Alternatives considered

- **Keep the imperative per-layer `forward()`.** Rejected and deleted: the graph path became the
  default and `forward.rs` was removed (`e54070c`, 2026-08-21, "the imperative forward is fully
  replaced by the graph path").
- **Keep the whole-layer `layer_gpu()` fast path** — which the plan had proposed keeping.
  **Explicitly not kept**: per-op execution through the backends is the only inference path, and
  the legacy `cuda.rs::layer_gpu` survives only as dead code
  (`docs/COMPUTE-GRAPH-DESIGN.md`, `docs/METAL-BACKEND-DESIGN.md` §2 non-goals). This is the
  alternative closest to the decision, and it lost because a whole-layer path reintroduces exactly
  the runtime backend choice the graph exists to remove.
- **A hardcoded backend chain plus a compile-time enum.** Rejected because adding a backend forced
  every consumer to be edited, and it buried the ordering decision inside the allocator. The
  registry replaced a "hardcoded Metal-then-CUDA-then-CPU `if let` chain" with
  `registry().by_priority()` + `entry.caps` (`docs/BACKEND-REGISTRY-DESIGN.md`).
- **A dynamic `register(Box<dyn Backend>)` plugin registry** (the roadmap's original
  recommendation). Not the shape that landed, and the doc records the boundary: "Not a
  plugin/`dlopen` system: the set of backends is fixed at compile time", with `fn`-pointer entries.
  Rejected because the backend set is a build-time property of the crate, and `dlopen` would add a
  failure mode with no user.

## Consequences

- Backend assignment, fusion and buffer liveness became *build-time* properties, so "which device
  ran this node" is a decision that can be dumped, exported to DOT/JSON and asserted on.
- The graph is deterministic in `GraphParams`, which is what makes reuse possible — and what makes
  `GraphParams` part of the reuse identity permanently (ADR-0002).
- A node a backend cannot execute returns `Err`; there is never a silent CPU fallback
  (`docs/GPU_SAFETY.md`).
- Costs knowingly accepted: `NodeMeta` is a concrete enum, so a new metadata kind edits the enum;
  the IR is single-output, so a multi-part result needs `Op::View` aliases and `BatchMatMul`
  fusion stays deferred; liveness must follow *build* order (using `topo_order()` produced a
  regression with logits off by 21.79); and one per-backend branch deliberately remains in the
  trace-capture path.

## References

- `docs/COMPUTE-GRAPH-DESIGN.md` — the design and the current contract.
- `docs/BACKEND-REGISTRY-DESIGN.md` §1 — the single device seam, the pinned priority order and the
  fixed id space.
- Commits: `e06d735` (plan/corrected analysis), `a163a07` (Phase-1 IR) and `e54070c` (the single
  path), all 2026-08-21; the registry contract `a8506fd` (design) and its implementation `cdf41b2`,
  both 2026-09-24.
