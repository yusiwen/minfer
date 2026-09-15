# 106 — Query-Driven Formula Gates (T2: ksplit × SM count, BT smem feasibility, plane VRAM budgets)

**Campaign**: T-series · Device Adaptation Layer — design record
[`docs/DEVICE-ADAPTATION-PLAN.md`](../DEVICE-ADAPTATION-PLAN.md) (T2 phase).
**Status**: LANDED. **Scope guard**: formula parameterization + feasibility
gates only; GB10 numbers identical by construction.

## What landed

### 1. Auto-ksplit parameterized by SM count (plan §6.1)

The two duplicated doc-92 auto-ksplit blocks (q6_K NB path, q4_K NB path) are
extracted into one `CudaState::auto_ksplit(nt, nbt_y, nktile)`:

- M-starve gate unchanged: `nt <= 64` (a single M tile row) with `nktile > 1`.
- Resident-block target: `max(256, 2 × SM)` — on GB10 (48 SMs → 96 < 256)
  this is **exactly the calibrated 256**, so GB10 behavior is bit-identical;
  larger SM counts scale the target proportionally.
- `MINFER_MMQ_KSPLIT_TARGET` still overrides everything (explicit human
  choice beats the formula, precedence rule plan §4).

### 2. BT dynamic-smem feasibility + the R2 device-0 fix (plan §6.2)

- The BT tile config's smem demand is now a **single-source** formula:
  `mmq_dynamic_smem_bytes()` in cuda_kernels.cu, consumed by both
  `launch_mmq_nt` and a new `cuda_mmq_smem_bytes()` extern (the formula stays
  owned by the kernel it serves; no Rust mirror to drift).
- **R2 fixed**: `cuda_shared_per_sm` / `cuda_shared_per_block_optin` queried
  device 0 unconditionally; they now query the CURRENT device
  (`cudaSetDevice(best_device)` runs at init, so this is the selected GPU —
  correct on multi-GPU hosts).
- Init folds the feasibility check into the tier gate:
  `mmq_smem > per-block-optin → tier_mmq = false` (loud log) — prefill
  degrades to the f16 GEMM path instead of failing launches on 100 KB-class
  devices. GB10 passes (identical to the previous unconditional behavior;
  the demand is ~92 KB vs GB10's optin ceiling).
- `cuda_shared_per_sm` stays C-side only: per-block optin ≤ per-SM on every
  arch, so the per-block check subsumes it.

### 3. Plane VRAM budget gate (plan §6.3)

`CudaState::plane_budget_ok(extra_bytes)` — `free > extra + extra/4` via
`cudaMemGetInfo` — applied to every OPTIONAL weight plane before its device
upload: q8_0 p32 pair (+100% of the raw weight), q6_K W_exp dense plane,
q6_K W_dsc, q4_K W_dsc. Deliberately silent: registration is best-effort by
design; every consumer already falls back to its raw path on a map miss.
GB10's 128 GB always passes (zero change); on 8 GB unified memory
(Orin Nano, plan §9.2) the planes self-disable and the raw paths serve.

### 4. Not in this phase (per the plan's own marking)

- **Tile candidate search** (plan §6.4, "optional/last, separable"): small-nt
  BT instantiations are the R8 destination for tier batch caps < 8. Deferred
  until the Orin Nano field A/B decides whether the caps are worth acting on
  — building them first would be speculative kernel work.
- **Batch-cap activation in the decode arms**: follows the same decision.
  Table data + `mmvq_cap` are ready and tested (doc 105).
- **T3 auto-calibration**: per the plan, "later, independent" — its value
  materializes only on devices minfer has not measured; on GB10 it would
  only reproduce the table.

## Verification

- Full build warning-clean (nvcc targets unchanged: sm_75…sm_121 + PTX
  compute_121).
- GB10 zero-change arguments, per item: ksplit target `max(256, 96) = 256`;
  smem check `~92 KB ≤ GB10 optin` passes; plane budgets pass with >100 GB
  free. All three are identity-preserving gates — no accumulation order or
  kernel choice changes on this device.
- Full suite + forced-tier soaks + identity battery: see the T-series
  closure entry in `CUDA_OPTIMIZATION.md`.

## Files

- `src/cuda.rs` (auto_ksplit, smem feasibility, plane_budget_ok + 4 gate
  sites, FFI decls), `src/cuda_kernels.cu` (R2 fix, smem single-source +
  extern).
