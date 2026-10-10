# 0037. A Metal weight dtype a kernel cannot consume is refused, never run as a wrong kernel

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #317, #329

## Context

Registration is what makes the device claim true. Before
[#317](https://github.com/yusiwen/minfer/issues/317) an f32 2-D weight had **no arm** on Metal and fell
through to the catch-all `_` arm — the Q4_0 kernel: `kernel_q4_0_f32_matmul` reads the f32 bytes as Q4_0
blocks (the first two bytes of `1.0f32` are `0x0000`, an f16 scale of 0) and **writes zeros**. A registered
weight type whose op then ran a kernel that could not consume it is the failure this guard exists to
prevent: the device claim was true and the computation was silently wrong.

The gap was invisible for a second reason. The reporting test,
`graph::op_matrix::matrix_cases_match_their_reference`, ran its Metal column only when some earlier test in
the process had already initialized `MpsState` — so the column's coverage depended on test order.
`docs/METAL-BACKEND-DESIGN.md` §4.4 states the resulting rule.

## Decision

**A weight dtype no kernel consumes is refused, and the catch-all dispatch arm is a guard rather than a
fallback.**

1. **Register only what a kernel consumes.** The loader's Metal branch is
   `matches!(ttype, F32 | F16 | BF16)`, one raw registration per type with its own arm. Before #164 f16 was
   refused here for exactly this reason: registering a weight type a kernel cannot consume "would make the
   device claim true while the op silently ran the wrong (or no) kernel".
2. **At dispatch, refuse with `Err` that names the evidence.** Since
   [#329](https://github.com/yusiwen/minfer/issues/329) the catch-all `_` arm returns `Err` naming the
   **node**, the **observed dtype** and the **kernel that would have run** (`pl_q4_0_f32` / `_multi`), so
   the next unregistered dtype aborts instead of repeating #317's silent zero.
3. **The refusal is driven through the production dispatch**, by
   `graph::metal_backend::tests::metal_matmul_refuses_an_unkerneled_weight_dtype`, whose control arm builds
   the same graph with an `F32` weight and asserts it still computes — so the dtype is the only difference.
4. **The test-order dependence was fixed with it**: the op-matrix case's Metal arm now calls
   `MpsState::init()` explicitly, exactly as its CUDA arm calls `CudaState::init()`.

## Alternatives considered

- **Keep the catch-all `_` arm as a fallback** (i.e. run the Q4_0 kernel for a dtype it cannot read).
  Rejected, and this is the bug the decision replaces: it reads the right bytes as the wrong format and
  returns zeros, with no error and no signal.
- **Register the dtype and let the op fall back to another backend.** Rejected by the same principle
  (ADR-0009): a fallback that is not an error is the silent path this rule forbids, and it would leave the
  model's *participation* claim true while the computation moved.
- **Leave the silence and detect it in tests.** Rejected: the reporting test itself was order-dependent, so
  the class of defect was invisible to it — which is why the fix includes making that column
  unconditional rather than relying on it.
- **The CUDA twin is not an alternative but a different mechanism**, and it is worth naming so the two are
  not confused: CUDA refuses at **eligibility** time (`supports_op` answers per op/dtype, and §4.3's table
  gives `Err("cuda: op … has no kernel …")`), so it never had a dispatch-time catch-all to remove. Metal's
  guard exists because its catch-all *was* the fallback.

## Consequences

- The next unregistered dtype **aborts** rather than computing zeros, and the message names the node, the
  dtype and the kernel that would have run — the three facts needed to diagnose it.
- The op-matrix Metal column no longer depends on another test having initialized `MpsState`, so the
  coverage that reported this class is now unconditional.
- Scope accepted and documented: **no model in the gate set carries a 2-D f32 weight**, so the f32 path is
  covered by the synthetic `metal_matmul_f32_matches_cpu` gate and the op-matrix case only.
- The rule is symmetric with the f16 paragraph it sits beside: a dtype is either registered with a kernel
  that consumes it, or refused — there is no third state.

## References

- `docs/METAL-BACKEND-DESIGN.md` §4.4 — the f16 and f32 paragraphs, the catch-all guard and its gate.
- `docs/CUDA-BACKEND-DESIGN.md` §4.3 — the eligibility twin (`supports_op`, and the `Err` for an op with no
  kernel), for the contrast the ADR draws.
- [ADR-0009](0009-a-failure-is-an-error-never-a-silent-fallback.md) — a failure is an error, never a silent
  fallback; [#317](https://github.com/yusiwen/minfer/issues/317),
  [#329](https://github.com/yusiwen/minfer/issues/329), [#164](https://github.com/yusiwen/minfer/issues/164)
  — the tickets; [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
