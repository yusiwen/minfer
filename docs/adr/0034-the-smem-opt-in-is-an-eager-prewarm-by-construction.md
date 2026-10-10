# 0034. The smem opt-in is an eager pre-warm by construction, with the lazy path as defence in depth

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #223, #218

## Context

A `gemm_f16_nt_kernel_t` instantiation whose dynamic shared memory exceeds the 48 KiB default must be
opted in with `cudaFuncSetAttribute(.., cudaFuncAttributeMaxDynamicSharedMemorySize, N)`: a launch over an
un-opted-in dynamic smem is rejected with `cudaErrorInvalidValue` and cannot succeed. The historical claim
is that the call is **illegal inside a capture window** and poisons the context (error 700) — but nothing
in the repository establishes that for the *adopted* capture mode, so the design cannot treat legality as
given.

The placement history is what makes this a decision rather than a preference. #188 deleted the init's
`CudaState::try_new` call site with no mention in its commit message; two later "dead code hygiene"
commits annotated the orphan `#[cfg_attr(not(test), allow(dead_code))]` instead of asking why a
production-looking init had no production caller; #218 removed the function and its `checked`/`skipped`
introspection, leaving the invariant **tested but not enforced by production**. `docs/CUDA-BACKEND-DESIGN.md`
§2.4 states the resulting design.

## Decision

**The opt-in is done once per process at context creation, and the lazy per-launch path stays as defence
in depth.**

1. **Eager pre-warm.** `CudaState::try_new` calls `gemm_prefill_smem_prewarm_one(tm, ks, af32)` once per
   process for every launchable combination (the `MINFER_GEMM_OPTIN_SET` X-macro in
   `src/cuda/kernels/common.cuh`, shared with the fatbin lookup and the test seam). **The placement is the
   argument**: `try_new` runs under `CUDA.get_or_init`, before the state is published, before any
   `CudaBackend` exists, and therefore before the per-instance stream `graph_begin_capture` needs — so "the
   attribute is set outside any capture window" holds **by construction**, not by inference from a warmup
   count or the capture mode. Once per process, not once per backend. `MINFER_NO_GEMM_PREWARM=1` skips it.
2. **The lazy per-launch opt-in stays.** `gemm_smem_optin` (called from the launcher's `GEMM_ONE`) runs on
   an instantiation's **first** launch and caches the answer in a function-local `static` per
   instantiation (a test injection is never cached); the launcher **does not launch** when the answer is
   false.
3. **One cache, one `cudaFuncSetAttribute` site.** The pre-warm drives the **same**
   `gemm_smem_optin<TM,KS,AF32>` the launcher reads, so a cache-keying regression cannot hide behind the
   pre-warm — it would leave the pre-warmed instantiations un-opted-in, which the gates read back from the
   device.
4. **An over-limit request is skipped without calling the attribute**, with the reason named: the call
   could only return `cudaErrorInvalidValue` (which `compute-sanitizer` counts) and the instantiation
   cannot launch on that device at all. On GB10/sm_121 (limit 101376 B) that is exactly one combination,
   `gemm_f16_nt_kernel_t<256,64,true>` at 122880 B.
5. **The MMQ launchers take the same treatment.** The terminal ones return 0 → `Err` at the Rust caller;
   the fallback ones keep their documented `0 = clean fallback` contract.

**Why the invariant holds without the error-700 claim:** the design does not rely on the call being legal
in a window, it relies on the call **never happening in one** — by construction, plus three mechanisms
that remain callable and are called out at their sites (a change to any of them is a design change, not a
tuning knob):

1. **The 3-run capture warmup** (`capture_warmup`, default 3): capture opens only from the third run of a
   `(uid, range)` key, so an instantiation's first launch — the one that calls the attribute — always runs
   uncaptured.
2. **`cudaStreamCaptureModeThreadLocal`** (ADR-0031): the window belongs to the capturing thread, so a
   foreign thread's driver call cannot join it.
3. **The per-instantiation cache**, keyed on the **`(tm, ks, af32)` template parameters**, not on a
   deduced `K`: the pre-#218 `template <typename K>` gave the whole family one `static`, and the #218
   coverage gate found `<64,64,true>` answering for `<128,64,false>`, whose attribute had never been set.

## Alternatives considered

- **The lazy path alone** — the state #188–#218 left, with the invariant tested but not enforced. Rejected:
  the gates cannot see a pre-warm that skips one `(tm, ks, af32)`, because the lazy path simply opts it in
  on first launch, so the coverage and counter arms stay green. That is why #223 added a fresh-process
  read-back immediately after `CudaState::init()`, and why one process runs the negation with
  `MINFER_NO_GEMM_PREWARM=1` so the read-back is not always `1`.
- **Call `cudaFuncSetAttribute` on every launch** instead of caching. Rejected: the cache is precisely what
  makes the >48 KiB launch *inside* a window never call the attribute.
- **Call it for an over-limit request and let it fail.** Rejected: a guaranteed
  `cudaErrorInvalidValue` that `compute-sanitizer` counts, for an instantiation that cannot launch on that
  device anyway — a noisy call that can only fail.
- **Rely on the error-700 claim** — i.e. accept the call inside a window and guard it by mode. Rejected
  because the repository does not establish the claim for the adopted mode; making the call never happen is
  a stronger position than arguing about whether it may.

## Consequences

- The guarantee is **by construction**, not by inference from a counter or the capture mode, and the
  operator's signal is the **first-launch site report**: a refused or skipped instantiation prints once at
  the launch site; a fully admitted pre-warm is silent. The removed `checked`/`skipped` counters did not
  come back.
- `MINFER_NO_GEMM_PREWARM=1` is the documented control: it provides the same-binary A/B, and the #218 arms
  run their fresh-process children with it so their claims stay the lazy path's and the pre-warm cannot
  make them vacuous.
- **Load-bearing coupling.** The pre-warm's cost is the fatbin's **one-time module load**, which the
  process pays before the first kernel from it can run either way — so it is ~free in a real run *only
  while* the pre-warm triggers the same load the first kernel would. It is one module finalization, not
  twelve (looping over a single instantiation costs the same as looping over all of them). The measured
  figures are the record's, in §2.4's cost block.
- Cost accepted: the first invocation pays that module load at init rather than at the first launch.

## References

- `docs/CUDA-BACKEND-DESIGN.md` §2.4 — the design, the three mechanisms, the #218/#223 gates and the cost
  block.
- [ADR-0031](0031-capture-runs-in-thread-local-mode.md) — mechanism 2, the capture mode.
- [#145](https://github.com/yusiwen/minfer/issues/145), [#147](https://github.com/yusiwen/minfer/issues/147),
  [#218](https://github.com/yusiwen/minfer/issues/218), [#223](https://github.com/yusiwen/minfer/issues/223)
  — the ticket chain; [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
