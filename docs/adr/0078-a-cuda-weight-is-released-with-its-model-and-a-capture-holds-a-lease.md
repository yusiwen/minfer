# 0078. A CUDA weight is released with its model, and a live captured graph holds a lease

- Status: Accepted
- Date: 2026-11-07
- Issues: #758

## Context

#745/#755 gave the name-keyed CUDA weight registry a model identity: a same-name
and same-size registration from a *different* model (different file fingerprint)
is now a fresh device copy instead of a silent reuse of the first model's bytes.
The old `cudaMalloc` buffer was deliberately not freed, because a live captured
CUDA graph resolves its weight pointers at capture time and replaying it over a
freed buffer would read whatever the driver put there next.

That left a normal reload leaking. Load a model, drop it, load another with the
same tensor names and sizes: the first model's device weights stay allocated for
the life of the process. `weights_bytes()` summed only the current name map, so
the replaced buffers were not charged to the E4 budget while `cudaMemGetInfo`
reported their memory as unavailable. A reloading server could therefore lose
device capacity per swap, pass its own feasibility checks, and eventually fail
the budget or the auto-offload fit with numbers that did not explain themselves.

`CudaState` is a process-wide singleton with no teardown, and a `GraphCache` (and
its `CudaBackend`) may legitimately outlive a model value — it dispatches by
weight name and holds captured graph execs. The release point has to be the
model's lifetime, and the "may a captured graph still read this?" question has to
be answered by the captures themselves.

## Decision

**1. A CUDA weight belongs to a model registration.** The loader calls
`begin_model(fingerprint)` before the first weight registration and stores the
returned `Arc<CudaModelRegistration>` on the model (the Metal `ModelRegistration`
shape, ADR-0077). The registration id is separate from the model fingerprint:
two loads of one file share the fingerprint — so the same-name+size device copy is
still reused — but are separate owners, and each is releasable on its own.

**2. Ownership is a ledger, and the name map is only a lookup.** Every
model-scoped device allocation is recorded in `model_allocs` with its pointer,
size and owner list: the main `register_weight` buffers, the q6_K/q4_K/q8_0
companion planes, the q6_K split-plane siblings, and the lazily-built prefill f16
cache. Same-fingerprint reuse adds the current registration to the existing
allocation's owners instead of allocating. A replace by a different model or size
starts a new allocation; the old one keeps its owner.

**3. Release frees when unowned, and defers while captured.** Dropping the last
`Arc<CudaModelRegistration>` removes the id from every owner list. An allocation
whose list becomes empty is freed (`cudaFree`, after synchronizing the context
stream) when no captured graph is live. The names that still point at it are
removed from the registry — a newer model's replaced names are untouched because
the pointer check is what selects them.

**4. A live captured graph holds a `CaptureLease`.** `CapturedGraph` owns the
lease. With at least one lease live, an emptied allocation is marked `retired`,
not freed: the release reports `ReleaseOutcome::Retained { bytes, captures }` and
prints the reason. The last lease's `Drop` reaps every retired allocation and
reports `Freed { bytes }`. This is the named refusal the ticket asks for — the
memory is deferred, never skipped, and never a dangling pointer. Every path that
drops captured execs (`CudaBackend` drop, `set_kv_layout`, the graph-mode toggles,
an exec replacement) drops leases naturally, so no consumer needs a new call.

**5. `weights_bytes()` sums the ledger, retired bytes included.** The E4 gate's
weights term and `cudaMemGetInfo` then describe the same memory: a replaced buffer
stays charged while it is resident, and stops being charged when it is actually
freed. The poisoned-lock recovery stays fail-closed (report the entries, never 0).

## Alternatives considered

- **Free immediately on replace.** The #755 comment is the reason it was not done:
  a captured exec's kernels hold raw pointers, and replaying over freed memory is
  undefined behaviour, not a slow path.
- **Only free when `graph_execs` is empty at release time.** `CudaState` is
  process-global and the cache is per engine; a model drop cannot see another
  engine's captured execs. The registry would have to ask every backend, which is
  the lease in a worse shape.
- **A capture-epoch check before each replay.** It detects a stale exec but cannot
  make the buffer safe to free: the epoch is read at replay, after the free would
  already have happened.
- **Reference-count the cache into the registration.** The correct automatic
  shape, but it threads a registration handle through every pool, buffer and
  backend. The lease puts the count where the references actually are.
- **Keep the process-lifetime hold and document it.** That is the leak this
  decision removes; the device has no unload the way `MpsState` does not, so no
  later teardown would ever reclaim it.

## Consequences

- A dropped model with no live capture returns its weight bytes immediately; a
  dropped model whose cache is still alive retains them and frees them when that
  cache goes. A cache used after its model is gone still works only while it holds
  a lease, and it must be dropped for the memory to return.
- `weights_bytes()` counts retired allocations, so the E4 budget is conservative
  in exactly the window where the device is still holding the bytes.
- The f16 prefill cache is now model-scoped as well; a model drop frees it with
  the weight it caches rather than leaving it for the process lifetime.
- The `a_cuda_reload_frees_the_first_models_device_weights` gate asserts the free
  bytes return to the pre-reload baseline within 128 MiB after the reload, and
  `a_cuda_release_is_retained_while_a_capture_is_live` asserts the named
  `Retained` outcome and the deferred reap.
- CPU behaviour is unchanged: the registration, ledger and lease live entirely
  under `feature = "cuda"`, and the models' `device()` still answers from the
  graph's single authority.
- A registration is not a substitute for the cache contract: the model must still
  not be used after it is dropped, and a `GraphCache` must be dropped with (or
  before) the model once its last forward is done.

## References

- `docs/CUDA-BACKEND-DESIGN.md` ("Weight lifetime and the ownership ledger").
- `docs/MEMORY-POLICY-DESIGN.md` §1 (the E4 weights + pooled + allocation check).
- ADR-0077 (`docs/adr/0077-a-metal-registration-is-released-with-its-model.md`) —
  the model-registration shape this mirrors on the CUDA device path.
- Issue [#758](https://github.com/yusiwen/minfer/issues/758); the identity fix it
  builds on is [#755](https://github.com/yusiwen/minfer/issues/755) (found while
  landing [#745](https://github.com/yusiwen/minfer/issues/745)).
