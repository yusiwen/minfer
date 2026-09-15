# Device Adaptation Layer — Design Plan (T-series)

**Status**: 📐 proposed (not implemented). This document consolidates the design
workspan from the post-doc-104 review sessions: minfer's device-parameter gating
audited against llama.cpp's, a GB10 tier cross-check, and a three-phase plan to
make minfer's dispatch adapt to devices it has never been measured on.

Naming: campaign **T-series (Tiering)**, deliverable module **设备适配层 /
Device Adaptation Layer**, code `src/device_tier.rs`. Doc numbering for the
implementation records starts at `docs/cuda_optimization_steps/105-*` (T1).

## 1. Motivation

Every dispatch constant that the CUDA optimization campaign measured — the
MMVQ batch bound (nt ≤ 8), the small-shape floors (id ≥ 2048), the q5_K/q6_K
shape crossovers (od·id ≥ 24M / 4M), the 8c int8-GEMM gate (id ≤ 8192), the
auto-ksplit M-starve formula (nt ≤ 64) — is a **GB10 measurement written as an
unconditional constant**. On this device they are load-bearing and verified;
on any other GPU they are guesses. llama.cpp solves the same problem with a
four-layer, community-calibrated tier system. This plan gives minfer the same
capability while preserving its measured assets and its bitwise-identity
discipline.

## 2. Goals and non-goals

**Goals**

1. Runtime device-tier selection (cc-keyed table, exact-match → family
   inheritance → generic fallback) driving the kernel-family gates.
2. Tier data for foreign devices adopted from llama.cpp's community-calibrated
   tables (MIT, with attribution); GB10 tier stays minfer-measured.
3. Query-driven formula gates: SM-count-parameterized ksplit, shared-memory
   feasibility checks, free-VRAM budgeting for weight planes.
4. (Later) load-time auto-calibration restricted to bitwise-safe knobs.
5. Backend-neutral key design so AMD (ROCm/HIP) and Apple Metal tiers can be
   added without restructuring.

**Non-goals**

- cuBLAS fallback integration (separately scoped; see §13).
- New kernels for foreign vendors (the tier layer selects among kernels that
  exist; it does not create them).
- Cross-device bitwise output equality (already not a project guarantee; each
  path is compared against its own reference).

## 3. Current state

### 3.1 minfer (no dedicated module — inline gates)

| Location | Responsibility today |
|---|---|
| `src/cuda.rs` init (`:1463–1539`) | queries cc / SM count / free mem / name; only cc is consumed by dispatch |
| `src/cuda.rs` dispatch arms (`~:2700–2870`) | inline shape gates (nt / id / od·id), all GB10-measured constants |
| `src/cuda.rs:2928` `mmq_active()` | cc ≥ 800 feature gate |
| `src/cuda.rs` plane registration | load-time p32 / q4k_dsc / q6k decisions, no VRAM budget check |
| `src/cuda_kernels.cu:5452` | `MMQ_BI/BJ/WS/KD` tile macros, single config for all archs |
| `src/cuda_kernels.cu:7608` | `cuda_shared_per_sm` / `_block_optin` externs — **written, never consumed** |

### 3.2 llama.cpp (reference design, four layers)

| Layer | Where | What |
|---|---|---|
| Capability detection | `common.cuh` | `GGML_CUDA_CC_*` constants incl. vendor offsets (`AMD 0x1000000`, `MTHREADS 0x0100000`), feature predicates (`turing_mma_available` …), `ggml_cuda_info()` per-device table |
| Kernel-family selection | `should_use_mmvq` (mmvq.cu:318), `should_use_mmq` (mmq.cu:259) | per-cc-tier batch thresholds with tuning provenance comments ("tuned on RTX 4090 / RTX 5090 / DGX Spark GB10 / Jetson Orin"), smpbo ≥ 48 KB feasibility, `GGML_CUDA_FORCE_*` escapes |
| Tile configuration | `mmq-config-{ampere,blackwell,cdna,pascal-*,rdna*}.cuh` | per-(type, J-bucket) `constexpr` CASE tables; inheritance fallback (blackwell → ampere) |
| Launch configuration | `calc_nwarps` + `MMVQ_PARAMETERS_*` tables (mmvq.cu) | device-tiered nwarps incl. `MMVQ_PARAMETERS_GB10`; plus the MMQ J-search (mmq.cuh:1472): iterate J = 8..128, filter rows by **runtime-queried smpbo**, minimize tile count |

Key llama.cpp mechanisms minfer lacks: per-device tier data, the smem-filtered
J/tile search, and runtime-consumed device queries.

## 4. Design overview

```
src/
├── device_tier.rs   ★ T1 — tier table + selector (pure data + pure fns, no CUDA,
│                            no unsafe, offline-testable; cuda.rs depends on it,
│                            never the reverse)
├── device_calib.rs  ☆ T3 — load-time auto-calibration harness + frozen CalibTable
├── cuda.rs          T1/T2 consumer — resolves tier once at init; dispatch arms
│                          read tier values; new smem fields; plane VRAM budgets
├── cuda_kernels.cu  T2 minor — expose `cuda_mmq_smem_bytes(tile_id)` extern;
│                            (optional, last) tile multi-instantiation
└── models/*/loader.rs   unchanged (plane budget gates live in cuda.rs methods)

Dependency rule:  loader.rs → cuda.rs → device_tier.rs  (one-way)
                  cuda.rs ↔ cuda_kernels.cu  (extern boundary: queries/timing)
                  device_calib.rs → writes CudaState.CalibTable; dispatch reads
```

Precedence for every gated decision (highest wins):

```
env escapes (MINFER_NO_*, explicit human) >
MINFER_DEVICE_TIER=<cc> (forced tier — testing) >
T3 calibration (frozen per process) >
tier table > GENERIC fallback
```

## 5. T1 — tier table & selector

### 5.1 Data model (vendor-namespaced from day one)

```rust
pub enum Vendor { Nvidia, Amd, Mthreads, Apple, Unknown }

pub struct DeviceKey { pub vendor: Vendor, pub cap_key: u32 }
// Nvidia: cc (1210); Amd: llama.cpp offset scheme; Apple: chip generation.

/// Where a tier's numbers come from — the data-model encoding of the
/// adoption principle in §5.2 ("measured on a device where we matched or
/// beat llama.cpp → ours; untested devices → theirs").
pub enum Provenance {
    /// minfer measured on this device; parity-or-better vs llama.cpp verified.
    Measured,
    /// Adopted from llama.cpp community tables; never run on minfer.
    Adopted,
    /// Fallback convention for unknown devices.
    Generic,
}

pub struct DeviceTier {
    pub key: DeviceKey,
    pub name: &'static str,
    pub source: &'static str,          // provenance: minfer doc or llama.cpp file:line
    pub provenance: Provenance,        // drives future Adopted → Measured promotion
    pub mmvq_batch_default: i32,       // batch limit for quantized decode kernels
    pub mmvq_batch_by_type: &'static [(QuantClass, i32)],  // per-type overrides (sparse)
    pub mmq_available: bool,           // int8 BT path availability
}
pub const TIERS: &[DeviceTier] = &[ ... ];
pub fn select(cc: i32) -> &'static DeviceTier;   // exact → family → GENERIC
pub const IDENTITY_BATCH_BOUND: i32 = 8;         // doc-95 bitwise bound, always a min-cap
pub fn mmvq_batch_limit(cc: i32, class: QuantClass) -> i32;
```

Deliberate YAGNI: no payload polymorphism / trait hierarchy. Fields stay
CUDA-semantic until a second backend actually populates rows.

### 5.2 Tier table contents

**Adoption principle** (ruled 2026-09-14): on devices where minfer has been
measured at parity or better vs llama.cpp, minfer's own values win; on devices
minfer has never been measured on, llama.cpp's community-calibrated values are
adopted. The principle applies **per knob, not just per device**: knobs that
exist on both sides (batch thresholds) get adopted; knobs tied to kernel
internals that differ structurally (launch configs — their nwarps tables
cannot drive minfer kernels) keep minfer values and lean on the
correctness-first property (every kernel choice is correct everywhere; only
performance varies). Note the principle never has to adjudicate on GB10: the
§8 cross-check showed the directly comparable values already agree there — its
real job is protecting minfer-only knobs (p32 planes, 8c branch, shape floors)
that llama.cpp cannot express.

| key | name | mmvq_batch | mmq | provenance | source |
|---|---|---|---|---|---|
| 1210 | DGX Spark GB10 | 8 (all supported types) | yes | **Measured** | minfer docs 94–104; agrees with llama.cpp mmvq.cu:349 |
| 1200 | Blackwell consumer (RTX 5090/5080/5070) | default 8; **q4_K → 5, q5_K → 6, q6_K → 7** | yes | Adopted | llama.cpp mmvq.cu:335 (tuned on RTX 5090) |
| 89 | Ada (RTX 4090/4080/4070) | 8 (their overrides touch only unsupported q2_k/q3_k) | yes | Adopted | llama.cpp mmvq.cu:323 (tuned on RTX 4090) |
| 86 | Ampere (RTX 3090/3080/3070/3060) | 8 | yes | Adopted | llama.cpp generic — no Ampere batch specialization exists |
| 75 | Turing (RTX 2080/2060) | 8 | **no** | Adopted / provisional | batch: generic; mmq: divergence ruling #4 (§9, field TODO §9.1) |
| −1 | GENERIC (unknown, incl. Pascal GTX 10-series) | 8 | cc ≥ 800 | Generic | fallback convention; Pascal resolves here with batch 8 + no MMQ, so no dedicated row is needed |

Scope notes from the source audit: llama.cpp's NVIDIA batch specializations in
`should_use_mmvq` are **exactly Ada and Blackwell** (plus DGX Spark, which we
own, and Jetson Orin, ruled out — OQ3); Ampere/Turing have no batch
specializations and fall through to the generic ≤ 8. Q2_K / Q3_K / IQ* rows
are dropped (unsupported types), which is why the Ada tier carries no
overrides for minfer's types. License note: llama.cpp is MIT; every adopted
row carries an attribution comment.

### 5.3 Selection algorithm

Exact `cap_key` match first (1210 must win over the ≥ 1200 family rule), then
NVIDIA family inheritance (≥ 1200 → 1200 tier; ≥ 800 → 86; ≥ 750 → 75; else
GENERIC). The tier is resolved **once at init** (cheap; also honors
`MINFER_DEVICE_TIER=<cc>` forcing for tests) and stored in `CudaState` — the
per-dispatch cost is a struct-field read, never a scan.

### 5.4 Dispatch integration

- Regular decode: `nt ≤ min(tier.mmvq_batch_limit(class), IDENTITY_BATCH_BOUND)`.
- **Spec verify path is exempt from the tier cap** and pinned to
  `IDENTITY_BATCH_BOUND` alone (see review finding R3 — the bitwise
  transparency requirement of speculative decoding must not be tightened by a
  foreign tier's performance threshold).
- `mmq_available` replaces the bare `cc ≥ 800` check.
- On cc = 1210 the resolved values equal today's constants: **zero behavior
  change on GB10**.

### 5.5 Gates owned by the tier vs global (T1 boundary)

Tier-owned in T1: batch limit, mmq availability. **Global (unchanged) in T1**:
id ≥ 2048 floors, od·id ≥ 24M/4M crossovers, the 8c id ≤ 8192 branch, ksplit
formula, p32 plane gates. These are minfer-private shape-dimension constants
with no llama.cpp counterpart (llama.cpp delegates the fallback side to
cuBLAS; minfer's fallbacks are its own kernels — the crossovers exist only
here). They become tier-owned only when measured on another device (T3) or
when a formula basis exists (T2 for ksplit).

## 6. T2 — query-driven formula gates

1. **ksplit parameterization**: the M-starve test (today `nt ≤ 64`) becomes
   `device_tier::ksplit_target(sm_count, nt)`; `sm_count` is already queried
   at init (currently print-only).
2. **BT smem feasibility**: init stores `smem_per_sm` / `smem_per_block`
   (wiring the dormant `cuda_shared_per_sm` externs); dispatch validates the
   tile config's dynamic-smem demand (exposed via a new
   `cuda_mmq_smem_bytes(tile_id)` extern — the formula stays in the .cu,
   owned by the kernel) and degrades (deeper ksplit → f16 GEMM) instead of
   failing launches on 100 KB-class devices.
   *Fix on the way*: `cuda_shared_per_sm()` hardcodes device 0 — must query
   the selected device (review finding R2).
3. **Plane VRAM budget**: `register_weight_q80_p32` / q4k_dsc entry checks
   `plane_budget_ok(free_mem, need)` — the +94 % / +3.3 GB trades that are
   free on 128 GB GB10 must self-disable on smaller cards.
4. *(Optional, last, separable commit)* **tile candidate search**, modeled on
   llama.cpp's J-search: candidates {64×64, 128×64, 128×128} in a table,
   filtered by queried smem, minimizing tile count. GB10 must resolve to the
   current 64×64 (assert in test); foreign cards get a legal, reasonable
   choice without human calibration.

## 7. T3 — load-time auto-calibration

Runs inside the existing `prewarm` window, strictly before any graph capture;
if prewarm is skipped, the tier table stands (calibration is best-effort).

**Hard rule (review finding R5): auto-selection may only choose among
variants proven bitwise-identical** (e.g., thread-count variants whose
reduction differences are exact-zero contributions). Anything that changes
accumulation order — tile shapes, ksplit depth — is tolerance-class and must
remain pinned by tables/formulas, never by timing; otherwise cross-run greedy
reproducibility on the same device breaks. Non-bitwise knobs get one-time,
version-boundary changes with a full gate rerun instead.

Calibration results freeze into a per-process `CalibTable` (`OnceLock` in
`CudaState`); dispatch order: calibration override → tier table → GENERIC.

## 8. GB10 tier cross-check (minfer vs llama.cpp, cc = 1210)

| # | Decision point | llama.cpp (1210) | minfer (current) | Verdict |
|---|---|---|---|---|
| 1 | decode batch bound | MMVQ nt ≤ 8 (mmvq.cu:349; only q2_k ≤ 6 — unsupported type) | MMVQ family nt 1–8 | ✅ agree (different rationale: perf crossover vs doc-95 identity bound) |
| 2 | MMVQ nt=1 threads | GB10 table: generic 4 warps ×2 = 256 threads (mmvq.cu:539) | 256 threads | ✅ agree |
| 3 | MMVQ nt 2–8 threads | generic 4 warps (128) | multi 256 | ⚠️ not comparable (kernel structures differ: per-token weight reads vs doc-104 hoist) |
| 4 | MMQ availability | sm_75+ (`turing_mma_available`) | cc ≥ 800 | ❌ divergence at sm_75 |
| 5 | MMQ tile geometry | Ampere-inherited I=128, runtime J-search, stream_k | fixed 64×64 + ksplit formula | ❌ different — both self-calibrated, no right answer |
| 6 | q8_0 decode kernel | raw 34B MMVQ | **p32 split planes** (doc 104, faster kernel) | ✅ minfer-only advantage |
| 7 | small-shape f32 tier | none (MMVQ any shape) | id ≥ 2048 floor + od·id crossovers | structural (minfer identity asset) |
| 8 | q4_0 8c branch (id ≤ 8192) | none | present (+38–44 % measured) | minfer-only |
| 9 | smem feasibility query | smpbo ≥ 48 KB gate + J-search filter (live) | externs written, never consumed | gap → T2 |

**Verdict**: the two directly comparable values (batch bound, nt=1 threads)
agree exactly on this machine; the only true data disagreement is #4.

## 9. Divergence rulings

- **#4 sm_75 MMQ**: keep minfer's conservative `cc ≥ 800` for now — the
  ruling is **provisional**, pending field measurement on an RTX 2080 Ti
  (§9.1). The 75 tier row encodes `mmq_available = false` with a comment
  citing this ruling.
- #3/#5/#6/#7/#8: keep current values — each has minfer measurements behind it
  or is a competitive advantage with no llama.cpp counterpart.

### 9.1 Field-measurement TODO — RTX 2080 Ti (sm_75)

A 2080 Ti is available to the project owner. This is the first foreign-device
measurement opportunity and doubles as the T1 field-validation run.

**What runs on it today (zero new code needed — sm_75 is already a build
target, build.rs:534):** MMVQ family (dp4a works on Turing), f16 GEMM prefill
(`gemm_f16_nt` compiles for sm_75), f32 fallbacks. The BT kernel family is
**compile-time excluded** on sm_75 (`#if __CUDA_ARCH__ >= 800` guards in
cuda_kernels.cu) on top of the runtime `cc ≥ 800` gate — so a Turing run
exercises everything except BT.

**Phase A — T1 validation + #4 data (no code changes):**

1. Build (`--features cuda`) and run the full suite + identity battery on the
   2080 Ti; confirm the 75 tier resolves (batch 8, mmq off) and everything
   passes — correctness must be device-independent.
2. tg128 A/B vs llama-bench (same window discipline as the GB10 campaign).
   Expected shape: decode competitive (MMVQ vs their MMVQ), prefill behind
   (our f16 GEMM vs their Turing int8 MMQ). The size of the prefill gap is
   the decision input.

**Phase B — outcome branches:**

- If the prefill gap is small or the owner deems it acceptable: close #4 as
  "conservative by choice", promote the 75 row's mmq ruling from provisional
  to final, and flip its provenance toward Measured.
- If the gap is large: two candidate remedies, to be scoped then:
  (a) port BT to Turing MMA layouts (Turing has int8 MMA but different
  instruction shapes than Ampere — a real kernel campaign, llama.cpp ships
  separate Turing paths for exactly this reason), or
  (b) the cheaper cuBLAS prefill fallback for cc < 800 (per the cuBLAS
  analysis, Turing is precisely where a cuBLAS fallback is strongest relative
  to hand-written code).

Either way the measurement promotes the 75 row from Adopted to Measured for
the knobs it covers — the first exercise of the provenance workflow.

## 10. Cross-vendor extension

- **AMD (ROCm/HIP)**: keys via the llama.cpp offset scheme; tier rows copied
  from their CDNA/RDNA tables. The layer generalizes for free — but the rows
  are inert until minfer has AMD kernels (the layer is the bookshelf, not the
  books).
- **Apple Metal**: minfer's second backend already exists; M-series rows are
  the realistic first non-CUDA consumers. `IDENTITY_BATCH_BOUND` and formula
  constants are CUDA-kernel properties and must be re-established per backend.
- **Moore Threads etc.**: same namespacing pattern (llama.cpp ships
  `MTHREADS 0x0100000`).

## 11. Acceptance gates (per phase)

1. GB10 zero behavior change: suite 189/0/3, identity battery 4/4, tg128
   baseline ± 0.1 tok/s.
2. Offline unit tests in `device_tier.rs`: tier resolution for
   1210 / 1200 / 89 / unknown cc; per-type override lookup.
3. **Forced-tier soak**: full suite with `MINFER_DEVICE_TIER=89` on GB10 must
   pass — gates may only ever choose among kernels that are correct
   everywhere, so a foreign tier must degrade performance, never correctness.
4. Every divergence and adopted row carries a provenance comment; disagreements
   with llama.cpp concentrated in the table header for future syncing.

## 12. Implementation order

T1 (half day) → T2 (one day; item 4 optional/last) → T3 (later, independent).
Each phase = one doc (`105-…`, `106-…`, `107-…`) + the full gate set.

## 13. Explicitly out of scope (pointer)

**cuBLAS fallback integration** — replacing the hand-written f32/f16-GEMM
fallback tier with `dequant_f16 + cuBLAS` (chain-tail position, llama.cpp
style) was analyzed separately: feasible at roughly neutral net LOC, buys
all-type coverage and foreign-arch shape handling, but changes the numeric
class of fallback shapes (breaks 0.6B greedy parity vs llama.cpp) and adds
capture-window workspace discipline. Keep as a follow-up campaign; the tier
layer here is a prerequisite consumer of its decisions.

## 14. Review findings incorporated (second pass)

- **R1 — per-type thresholds**: llama.cpp's tiers vary *by quant type*
  (blackwell: q4_k 5 / q5_k 6 / q6_k 7), not one number → `mmvq_batch_by_type`
  override list added to the data model; a single `mmvq_batch` field would
  have silently mis-ported the 1200 tier.
- **R2 — device-0 bug**: `cuda_shared_per_sm()` queries device 0 regardless of
  the selected device — must fix when wiring (would mis-validate smem on
  multi-GPU hosts).
- **R3 — spec path vs tier cap**: a foreign tier with a batch limit < 8 (e.g.,
  llama.cpp's Orin tier caps k-quants at 1) would starve the speculative
  verify path of the multi-MMVQ kernels its bitwise guarantee needs → spec
  verify is exempt from the tier cap (§5.4).
- **R4 — precedence order**: env escapes vs forced tier vs calibration vs
  table was unspecified → fixed order in §4.
- **R5 — calibration scope**: timing-based auto-selection must be restricted
  to bitwise-identical variants; tile/ksplit knobs change accumulation order
  and would break same-device cross-run reproducibility (§7 hard rule).
- **R6 — tier resolution caching**: per-dispatch `select()` scans avoided;
  tier resolved once at init into `CudaState`.
- **R7 — gate ownership boundary**: explicit list of which gates stay global
  in T1 (§5.5) — shape-dimension crossovers have no llama.cpp counterpart and
  cannot be "adopted" from their tables.

## 15. Open questions

All three resolved by ruling (2026-09-14), see §5.2 / §9:

- **OQ1 — resolved**: adopt consumer-GPU tiers only. The audited scope is
  smaller than expected: llama.cpp's NVIDIA batch specializations are exactly
  Ada and Blackwell-consumer, so the only non-trivial adoption is the 1200
  tier's q4_K/q5_K/q6_K overrides (tuned on RTX 5090). Ampere/Turing carry
  generic values. The adopted-vs-measured distinction is carried by the
  `Provenance` field.
- **OQ2 — resolved by the adoption principle**: the tile candidate search is
  untested territory for minfer → follow llama.cpp's proven approach (J-search
  + smem filter); GB10 pinned to 64×64 by test assertion; the one-time
  tolerance-class change on foreign devices is accepted at the T2 boundary.
- **OQ3 — resolved**: Jetson Orin rows are dropped per the consumer-only
  scope; the GENERIC fallback covers unknown devices more honestly than an
  untested copied row.
