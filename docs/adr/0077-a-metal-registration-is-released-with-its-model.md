# 0077. A Metal registration is released with its model, and the cache belongs to the engine

- Status: Accepted
- Date: 2026-11-06
- Issues: #745

## Context

[#753](https://github.com/yusiwen/minfer/pull/753) made a tensor's bytes a
`TensorData::Mapped` range with the mapping handle inside it, and
[#751](https://github.com/yusiwen/minfer/pull/751) had already made the Metal
zero-copy part registration hold that handle. The consequence recorded then was
deliberate: `MpsState` keeps its part/weight entries for the process lifetime, so
on macOS a loaded model's pages stayed mapped until MPS was torn down. `MpsState`
is a `OnceLock` singleton and has no unload, so a model drop could never release
the pages there — the one device path the host-side ownership change did not reach.

The device side is not free to release at arbitrary times. A zero-copy
`MTLBuffer` aliases the mapped pages, so the buffer must be gone before the
mapping is unmapped. A `GraphCache` holds `Tensor` clones and its backend
dispatches by weight name; it may use a registered buffer after the model value is
dropped unless the engine owns the lifetime.

## Decision

**1. A registration is model-scoped.** The loader starts a
`ModelRegistration` before the first `register_part`/`register_weight`; the
registry tags every entry with its id. The model owns the `Arc<ModelRegistration>`,
so a clone of the model is a handle of its own.

**2. The last handle releases buffers before mappings.** Dropping the token removes
the weight entries first (their `MTLBuffer`s), then the part entries (their buffers
and their `Arc<MmapFile>` handles). No `MTLBuffer` can outlive the pages it aliases;
the mapping is unmapped only when the last `Arc` goes.

**3. The cache belongs to the engine.** A `GraphCache`/backend may reference a
registered buffer by name; it must be dropped with (or before) the model, never
used after. `device()` also refuses the Metal path when MPS is live but the model
owns no registration: a model without the handle cannot keep its pages mapped.

**4. MPS itself stays process-scoped.** The device, queue, pipelines and scratch
buffers are not torn down; only the model's parts and weights are. Releasing the
model's device buffers is not a device teardown.

## Alternatives considered

- **Keep the process-lifetime hold.** It was the recorded state after #751, but it
  is the blocker this decision removes: a macOS process that loads a model and
  drops it keeps the mapping and its `MTLBuffer` forever.
- **Release at MPS teardown.** There is no teardown (`OnceLock<Option<MpsState>>`),
  and it would not help a process that swaps models without exiting.
- **Release on model drop without the cache contract.** The buffers would be freed
  while a live `GraphCache` still dispatches by name — a GPU read of unmapped
  pages, or a stale buffer. The contract is explicit instead.
- **Refcount the cache into the registration.** The correct automatic shape, but it
  would thread a registration handle through every cache/backend; the engine-level
  contract is the smaller first step and is what the current callers (worker,
  CLI, spec engine) already do.

## Consequences

- A macOS model drop releases its part mappings and device buffers, provided the
  engine's cache is dropped too; the `a_dropped_model_releases_its_metal_registration`
  gate asserts the mapping's `alive` flag after model + cache drop.
- The registration adds one `AtomicU64` per load and a tag per registry entry; the
  zero-copy path, the buffer offsets and the kernels are unchanged.
- A cache used after its model is gone is a contract violation. The failure is a
  namespaced lookup miss, not a silent read of another model's weights.
- MPS pipelines/scratch remain for the process; an unload does not reclaim them.
- The CUDA device-memory path is separate: its registry now keys by model identity
  (ADR-0076's follow-up, #755) but still does not free replaced device buffers.

## References

- `docs/METAL-BACKEND-DESIGN.md` (zero-copy parts and the backend lifetime),
  `docs/METAL_OPTIMIZATIONS.md` row 36 (the original no-copy decision).
- `docs/inference_e2e_walkthrough/02-gguf-load.md` §2.5/§3.4 (the host-side
  ownership contract that this completes on the device).
- Issue [#745](https://github.com/yusiwen/minfer/issues/745); PRs
  [#751](https://github.com/yusiwen/minfer/pull/751) and
  [#753](https://github.com/yusiwen/minfer/pull/753).
