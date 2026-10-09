# 0012. Device is the first axis, the layer the second — and no premature `common`

- Status: Accepted
- Date: 2026-10-04
- Issues: #261

## Context

Four backend files had grown into the largest in the crate: `src/cuda.rs` at **6,578 lines** (a
single 4,546-line `impl CudaState`, 126 functions), `src/cuda_kernels.cu` at **10,215 lines** (95
kernels plus 80 host launchers in one nvcc translation unit), `src/metal.rs` at 2,472 and
`src/metal.metal` at 5,151. The CPU side had a comparable trio (`kernel.rs`, `quants.rs`,
`vec_ops.rs`). Each file mixed layers — device/runtime, launch/dispatch, kernel source — so the
split raised a layout question with only two real answers: group by *layer*, or group by *device*.

The question had a cost attached, which is why it was decided before the files moved: **126
documents** named those four paths and 35 of them carried **401 line anchors**, so the order of the
splits and the landing of an anchor checker were part of the decision rather than an afterthought.

There was also a precedent to reason against: llama.cpp keeps `ggml-backend.cpp` /
`ggml-backend-impl.h` / `ggml-backend-reg.cpp` **flat, beside** the per-device directories, rather
than in a top-level layer tree.

## Decision

Three conventions, landed with the plan (`docs/SOURCE-LAYOUT-PLAN.md`):

1. **Device is the first axis, the layer is the second axis inside each device.** The per-device
   modules stay where they are, and each is split into its own inner axis. There is **no** top-level
   `L1/`, `L2/`, `L3/` directory tree: layers only become directories *inside* a device
   (`src/cuda/kernels/`, `src/metal/kernels/`, …). Each backend is free to choose a different second
   axis, and does: **CUDA by kernel family, Metal by layer, CPU by ISA**.
2. **The cross-device interface stays flat and singular.** `src/graph/backend.rs` (`Backend`,
   `KvProvider`) plus `src/graph/registry.rs` remain the one device seam. **No new trait is
   introduced.**
3. **A `common` module must earn its existence**: interface eligibility requires **at least two real
   implementations** *and* **at least two callers** using it with the same semantics.

The splits themselves then followed this convention: `#262` (`src/cuda/`), `#263`
(`src/cuda/kernels/`), `#264` (the CPU trio into `src/quants/`, `src/kernel/`, `src/vec_ops/`),
`#265` (`src/metal/`, `src/metal/kernels/`).

## Alternatives considered

- **A layer-first tree** — top-level `L1/`, `L2/`, `L3/`. Explicitly rejected, and the plan states
  the reason as a fact about the code rather than a preference about directories: **L2 is not a layer
  that can be moved away from L1**, because the CUDA launchers are inherent methods of `CudaState`
  and the Metal ones are inherent methods of `MpsCommandBuffer`. Splitting L1 from L2 by directory
  would therefore be a **type refactor, not a file move** — and the whole point of this work was that
  the moves be pure. The plan says it in the negative: "There is **no** top-level `L1/`, `L2/`,
  `L3/` directory tree: the layers only become directories *inside* a device."
- **Introduce a trait for the cross-device interface** (one abstraction per layer, or a
  device-agnostic trait per capability). Rejected: "**No new trait is introduced by this plan.**"
  The seam is already singular, and a second polymorphic layer would be a second place for the
  device decision to be made — the thing ADR-0001 exists to prevent.
- **Create the shared `common` module up front**, with `allocplan::DeviceMemory` as its first
  inhabitant. Rejected by the eligibility rule, and recorded as the rule's *worked example*: with
  only `CudaState::device_memory()` implemented, creating the abstraction "would be the
  single-real-implementation case this rule forbids, so it is deliberately **not** created here."
  The plan then re-analysed it after Metal's second implementation landed (2026-10-05) and the
  answer was *still* "no new module" — the type and the policy were already device-agnostic, and the
  device answer fits behind the existing resolver. This is the alternative that most repays being
  written down: the abstraction was refused **twice**, for two different reasons.

## Consequences

- A backend's four layers are co-located and findable: working on CUDA means reading `src/cuda/`,
  not four top-level directories.
- The seam stays singular, so "which device runs this node" has exactly one place to be decided
  (ADR-0001) and one place to be asked (ADR-0011's registry).
- The moves were **pure**: "0 field-visibility edits, 0 `pub(super)`", no behaviour change, no public
  renames, no new abstraction and no size ratchet, with test counts identical across the split.
- The documentation debt was real and was paid inside the same campaign, in a deliberate order: the
  anchor checker (#266) landed **before** the second split step, 401 anchors across 35 documents were
  re-pointed, and the historical records whose anchors cite a pre-split revision are frozen behind a
  mapping table rather than rewritten.
- Because the layer axis is *not* visible at the top level, the convention has to be documented and
  audited rather than inferred from the tree: the layout rules live in `AGENTS.md` ("Layout") with
  their rationale here, and `scripts/check_source_layout.py` enforces the mechanically checkable half
  (a test module is a file; every `src/**.rs` is named by a `mod` declaration).
- The decision and its implementation are a day apart for one backend: the umbrella and `#262`–`#264`
  are 2026-10-03/04, while `#265` (Metal) landed 2026-10-05 on a Mac. This ADR is dated at the
  decision.
- **CPU is the documented exception**: `quants.rs` / `vec_ops.rs` are the crate's shared numeric
  kernel library rather than a device-private layer, split by ISA instead of into a `kernels/`
  directory.
- Cost accepted: a contributor's instinct ("put all the kernels in `kernels/`") is wrong here, and
  the rule that corrects it lives in a document rather than in the directory names. A CUDA `.cu` also
  continues to hold kernels *and* their host launchers.

## References

- `docs/SOURCE-LAYOUT-PLAN.md` — the conventions (§"the plan"), the step table, and the `common`
  eligibility rule with its step-5 pre/post analysis.
- `docs/ARCHITECTURE.md` — the interface-eligibility convention restated for contributors.
- `AGENTS.md` ("Layout") — the tree and a one-line pointer, with the rationale here.
- Commits: `9174644` (2026-10-04, "docs: land the source layout plan and its conventions (#261)");
  the splits in `#262`–`#265`.

## See also

- ADR-0001 — why only L4 is polymorphic, which is the premise of convention 1.
