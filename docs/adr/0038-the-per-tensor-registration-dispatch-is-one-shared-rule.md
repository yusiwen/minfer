# 0038. The per-tensor registration dispatch is one shared rule, not a copy per loader

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #167, #141, #165

## Context

Putting a model on the device means registering every tensor the E5 offload plan places there, and each
type needs its own treatment: the quantized-type set, the F16 raw branch, the F32 branch (1-D norms/biases
versus 2-D matmul weights), the Q6_K padded repack, the q8_0 p32 plane, the q4_K `W_dsc` plane (under the
two-gate admission ADR-0035 records), and the `clear_mmq_nb_bt_only` rule. Both loaders — `models/qwen2` and
`models/qwen3` — need all of it.

Two copies of that block is two places to forget a type, and the record shows that is not hypothetical:
`docs/CUDA-BACKEND-DESIGN.md` §2.3 states the decision and the drift it ended.

## Decision

**One shared rule owns the per-tensor registration dispatch.** `src/models/weight_reg.rs`'s
`register_cuda_weight` carries the whole contract, and both loaders call it for every tensor the E5 plan
puts on the device:

- the quantized-type `matches!` set, the F16 raw branch, the F32 (1-D norms/biases vs 2-D matmul weights)
  branch, the Q6_K padded repack, the q8_0 p32 plane, the q4_K `W_dsc` plane under
  `q4k_dsc_plane_admitted`, and the `clear_mmq_nb_bt_only` rule;
- **the decision function is pure** (`cuda_weight_reg`): no `CudaState`, no environment — the r59 dispatch
  gates are passed in — so CI's CPU job **runs** its tests, exactly like `src/q4k_dsc.rs`;
- **the graph-side type gate stays per architecture** and must list the same types
  (`Qwen3Graph::weights_on_cuda` gained F16 in #167).

## Alternatives considered

**The one recorded alternative is the arrangement this replaced: a copy of the block in each loader.**
Rejected by what actually happened, not by an argument — the copies **drifted twice**:

- the qwen3 copy had **neither** the f16 branch (**#141**) **nor** the q4_K dsc call (r59/**#165**), so
  **an f16 Qwen3 fell to the CPU** and **a q4_K Qwen3 kept the in-kernel scalar dsc decode**.

**Purity is stated as a property with its consequence, not weighed against an alternative.** The record
says the decision function has no `CudaState` and no environment *so that* CI's CPU job runs its tests; it
does not describe a considered impure variant that lost. A reader should not infer one.

## Consequences

- A newly supported type is admitted **in one place**, and the drift class above cannot recur in the
  loaders — which is the whole point, since the symptom was silent (a CPU fallback and a slower decode
  path, not a failure).
- The per-architecture graph-side gate remains a manual obligation: the shared rule does not know what a
  given model's graph will claim, so `Qwen3Graph::weights_on_cuda` (which gained F16 in #167) has to list
  the same types by hand.
- Testability is a consequence, not an accident: because `cuda_weight_reg` is pure, the registration
  contract is covered by the **CPU** CI job rather than only by a GPU run — the same property
  `src/q4k_dsc.rs` was given for the same reason.

## References

- `docs/CUDA-BACKEND-DESIGN.md` §2.3 — the shared rule, the purity and the drift.
- [ADR-0035](0035-the-q4k-dsc-plane-admission-is-two-gates.md) — the `q4k_dsc_plane_admitted` gate this
  rule calls into; [ADR-0036](0036-bf16-gets-its-own-device-kernels.md) — the bf16 registration that goes
  through it.
- [#167](https://github.com/yusiwen/minfer/issues/167), [#141](https://github.com/yusiwen/minfer/issues/141),
  [#165](https://github.com/yusiwen/minfer/issues/165) — the tickets;
  [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
