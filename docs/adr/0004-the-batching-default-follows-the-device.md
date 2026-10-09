# 0004. The batching default follows the device

- Status: Accepted
- Date: 2026-09-19

## Context

Continuous batching (E2) landed with the mechanism in place but the default **off**, opt-in via
`MINFER_BATCH=1`. Its acceptance was a throughput claim, and the claim came out different on the
two boxes the project measures on. Measured on **dgxspark (DGX Spark GB10, aarch64 Linux, CUDA
13.0, sm_121)**:

- 7B Q4_K_M, prefix reuse on: serial 32 tokens / 16.50 s vs concurrent 32 tokens / 18.42 s —
  **0.49×**.
- 0.5B: **0.88×**; engine-level batched forward **1.45×** on the 0.5B and **1.00×** on the 7B.
- On the GB10: serial 1.314 s vs batched 0.686 s — **1.9×**; 15 four-wide steps at 6.5–6.7 ms/token
  against ~19 ms/token serial (~2.9× per token).

So the sign of the effect is a property of the hardware, not of the implementation. Two measured
causes explain the CPU side: the CPU `nt > 1` decode path is not more efficient per token, and
concurrency forfeits cross-request prefix reuse. A default is a statement about the user's machine,
and the machines disagree.

E2 and E6 predate the tracker, so the plan's records are the source.

## Decision

The batching default is **device-aware**, decided by one pure function:
`chat::batch_mode(requested, model.device())`. `MINFER_BATCH` unset means batched iff the model's
forwards run on a device (CUDA, and Metal since 2026-10-06); `=1` forces on, `=0` forces serial, and
an invalid value warns and falls back. Startup prints `[server] batching: on|off (device …)`.
`ModelDef::device()` (`Device::{Cpu, Metal, Cuda}`) is the single authority for "the device
participates", shared with the builder's `CParams.gpu`, so batching and graph topology cannot
disagree.

## Alternatives considered

- **Keep batching opt-in (`MINFER_BATCH=1`).** Rejected once the GPU number was understood: a CUDA
  server was leaving a ~2× win on the table behind an environment variable almost nobody would set.
- **Default batching on for every device.** Rejected by E2's CPU measurement: 0.49× is a 2×
  regression for the common case, in exchange for a win the CPU user never sees. (The earlier
  reaction to that measurement had been to keep the default with the measured-better serial path —
  "exactly as A6 was reverted on measurement" — which was correct while no GPU was available.)
- **Revert the CUDA default to opt-in while the f16 windowed-prefill fault was being diagnosed.**
  Proposed and **rejected because it would not have worked**: the fault lives in the *per-slot
  prefill*, which non-batched serving also performs, so the change would have moved the failure
  without removing it.
- **A single global default with a documented warning.** Rejected because the engine already knows
  which device it loaded onto; deriving it is a strictly better interface, and the derivation is
  testable in CI while a warning is not.

## Consequences

- The default is *derived*, so it cannot disagree with where the weights actually are.
- Making batching the GPU default promoted a latent correctness blocker to default behaviour: the
  f16 windowed FA prefill mask fault, found and fixed 2026-09-19. E2's recorded "GPU acceptance
  1.9×" is corrected in the same place — its *correctness* had never been checked per request.
- The CPU stays serial by default, so the original CPU throughput claim remains **unmet**; a CPU
  `nt > 1` decode kernel (the F1 family) is the only route to it. E2 closed on a measured negative
  result, recorded so nobody re-opens it.
- A measurement that is a property of one box became a *rule conditioned on the device* rather than
  a number in a doc — the shape later reused for the memory budget's `budget_decision` and for the
  layer-offload fit.

## References

- `docs/COMPUTE-GRAPH-DESIGN.md` §1.3 non-goals — restates the 0.49×/1.9× figures and the rule.
- `ARCHITECTURE-EXECUTION-PLAN.md` §7 — the **E2 step 3** record (the CPU 0.49× measurement and
  its two causes) and the **E2 step 6** record (the GPU 1.9× re-measurement), plus the E6 record
  ("a device-aware default", 2026-09-19) and the §0 status row. Note that §B3 is a *different*
  record — the B2 cross-request prefix-reuse measurement (≈11× TTFT), not the batching numbers.
- Commit `3993377` (2026-09-19, "feat(server): make the batching default follow the device (E6)").
