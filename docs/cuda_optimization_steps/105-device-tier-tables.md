# 105 — Device Tier Tables (T1: tier table + selector + MMQ gate)

**Campaign**: T-series · Device Adaptation Layer — design record
[`docs/DEVICE-ADAPTATION-PLAN.md`](../DEVICE-ADAPTATION-PLAN.md) (T1 phase).
**Status**: LANDED. **Scope guard**: this phase changes dispatch *data and
gates only* — no kernels touched, GB10 behavior provably unchanged.

## Goal

Every dispatch constant the GB10 campaign measured (batch bound nt ≤ 8,
size floors, shape crossovers, 8c gate, ksplit formula) is an unconditional
constant — correct here, a guess elsewhere. T1 replaces the *kernel-family*
gates with a cc-keyed tier table so foreign devices resolve sensible values,
while keeping every GB10-measured number as the 1210 row.

## What landed

### `src/device_tier.rs` (new, ~340 lines, pure data + pure fns)

- `Vendor` / `DeviceKey` — vendor-namespaced key space from day one (llama.cpp
  offset scheme reserved for AMD/Moore Threads; only NVIDIA rows exist).
- `Provenance` — `Measured` (1210: minfer docs 94–104) / `Adopted` (1200, 890,
  870, 860, 750: llama.cpp tables, MIT-attributed per row) / `Generic`.
- `DeviceTier` — key, name, source citation, provenance,
  `mmvq_batch_default` + sparse `mmvq_batch_by_type` (K-quant overrides),
  `mmq_available`.
- `TIERS` — the seven reviewed rows (plan §5.2): 1210 GB10 (all 8),
  1200 Blackwell (q4_K→5, q5_K→6, q6_K→7), 870 Orin (K-quants→1),
  890 Ada (no overrides for supported types), 860 Ampere, 750 Turing
  (mmq **false** — ruling #4), −1 GENERIC (mmq = cc ≥ 800 via the selector).
- `select(minfer_cc)` — exact key → family inheritance (≥1200 Blackwell,
  ≥800 Ampere, ≥750 Turing) → GENERIC. `select_forced(key)` — the
  `MINFER_DEVICE_TIER` override path (soak testing).
- `mmvq_cap(tier, class)` — tier limit clamped by
  `IDENTITY_BATCH_BOUND = 8` (the doc-95 bitwise bound always wins).

### Encoding discovery (fixed en passant)

minfer's runtime cc is `major*100 + minor` — **GB10 = 1201, not 1210**. The
old doc comments ("1210 = sm_121") were wrong. The table keys use the
llama.cpp encoding (`major*100 + minor*10`, GB10 = 1210) so adopted rows map
1:1 onto their source constants; `device_tier::llama_key(1201) = 1210`
converts. Both stale comments in cuda.rs corrected.

### `cuda.rs` integration

- CudaState gains `tier` / `tier_mmq` / `sm_count` (queried at init,
  previously print-only). Tier resolved **once** at init — dispatch reads
  plain fields, never scans.
- `mmq_active()` = `tier_mmq && mmq_enabled()` — the bare `cc >= 800`
  replaced by the resolved gate. On GB10: identical verdict.
- Banner line: `CUDA: device tier <name> (<provenance>, mmq <bool>)`;
  `MINFER_DEVICE_TIER=<key>` forces a row (logged as FORCED).
- **Batch caps intentionally not wired into the decode arms yet** (code
  comment at the dispatch head): a limit < 8 has no destination for the
  vacated nt range — BT starts at nt ≥ 9, and the `MINFER_SMALL_M_GEMM`
  experiment measured ~1.7× over multi-MMVQ at small nt (doc 91) — and would
  strip the spec identity family (plan §14 R3/R8). Activation waits for a
  small-nt BT destination (T2 tile candidates) or field A/B data; the table
  data and `mmvq_cap` are ready and tested.

## Verification

- `cargo test --bin minfer device_tier` — **7/7** offline unit tests:
  encoding conversion, exact rows, family inheritance + GENERIC, forced
  override, per-type override values vs the source tables, identity-bound
  clamp, provenance citations.
- Full build warning-clean; GB10 zero-change argument: the 1210 row resolves
  `mmq_available = true` and batch 8 = the previous constants.

## Files

- `src/device_tier.rs` (new), `src/cuda.rs` (fields, init, gate, comments),
  `src/main.rs` (module decl).
