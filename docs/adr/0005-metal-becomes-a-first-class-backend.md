# 0005. Metal becomes a first-class backend

- Status: Accepted
- Date: 2026-09-20
- Issues: #44, #54, #164, #208, #310
- Supersedes: ADR-0003

## Context

ADR-0003 deferred Metal for a round, on two grounds: no macOS machine was available to compile or
run it, and the KV arena's shape was still moving. The reversal was a decision in its own right,
taken on 2026-09-20, and the plan's own text states it: "Metal is a first-class target — it is the
default backend on macOS and a plain `cargo build --release` builds it — so this phase is
**scheduled, not deferred**" (`2c3c395`). The drift ADR-0003 knowingly accepted then became the
work: Metal had to be brought to the same arena semantics rather than approximately near them.

This ADR is dated at that decision, not at the completion of its work; the round closed 2026-10-06
and two capabilities landed after it (below).

## Decision

Metal is a **first-class backend**: it implements the same arena semantics as CPU and CUDA, and its
capability answers are the same kind of registry answer as any other backend's. Landed on a Mac
between 2026-10-05 and 2026-10-06:

- `#44` part (a) — the KV cell store, removal and shift, and explicit attention spans
  (`kernel_gqa_attn_window_f32/_f16`, `supports_attn_span()` true), PR #313.
- `#44` part (b) — the write/move side: `copy_cells` via `MTLBlitCommandEncoder`,
  `copy_kv_to_cpu`, the per-engine `kv_format`, and Metal joining the device-aware batching default;
  the process-wide `KV_F16` / `kv_cache_is_f16` / `set_kv_cache_type` were **deleted**, PR #316.
- `#53` (2026-10-05) — `MpsState::device_memory()` answers
  `MTLDevice.recommendedMaxWorkingSetSize`, so `--gpu-layers auto` and the E4 budget work on macOS.
- `#164` f16 weights — matmul + embedding kernels, PR #319; `#208` bf16 — PR #323.
- `#54` (G7) — the gap and parity re-measurement, PR #325.

### Later addenda (after the round closed)

- **2026-10-07, #362 / PR #364** — the set-valued `kv_map` gather, which took the macOS real-model
  set from 43/1 to **44/0**.
- **2026-10-09, #310, commit `0efd4db`** — a packed `q8_0` KV cell is read on Metal (mechanisms A
  and B), so `BackendCaps::reads_packed_kv` is now true for CPU, CUDA **and** Metal.

## Alternatives considered

- **Keep Metal as a second-class or experimental path.** Rejected: two sets of arena semantics is
  precisely the drift ADR-0003 accepted *temporarily*, and the round showed the cost. A
  second-class backend also cannot answer the registry's capability questions honestly.
- **Port Metal before the arena stabilised.** Already rejected in ADR-0003, and the reasoning held:
  porting against a moving arena would have meant porting twice.
- **Declare parity by re-running the existing CPU expectations.** Rejected: the Metal path is
  *allowed* to produce different numbers from the CPU by design (the CPU quantizes activations to
  `Q8_0`, the device reads `f32`), so each path is compared against its own reference and the
  real-model gate compares the greedy continuation rather than logits.
- **G4's own recorded alternative, kept as precedent:** for the mixed-layer dispatch asymmetry the
  decision was to **keep the refusal**, not to close it — closing it "would save 6 dispatches per
  mixed layer (10 → 4), 84/token on Qwen2.5-7B-Q4_K_M … with no numerical difference and a sub-1%
  time ceiling". A known asymmetry is recorded rather than papered over.

## Consequences

- `ModelDef::device()` answers `Metal` for f16/bf16 weights and both supported architectures, so
  batching (ADR-0004) and graph topology follow automatically — no consumer changes.
- Per-engine `kv_format` (ADR-0006) stopped being a CUDA-only concern.
- The evidence for Metal claims lives on a Mac, not in CI: `build-macos` only type-checks, so the
  macOS suite and the real-model set are run-and-recorded, and `docs/METAL-BACKEND-DESIGN.md` §7.4
  carries the baseline a later Mac run diffs against.
- Costs knowingly accepted, and they are the interesting part of this record: the **macOS test
  target was silently uncompilable for eleven days** (`cargo build` does not compile `#[cfg(test)]`,
  #303); "the macOS suite was **red for the whole Metal round**" — 21 failures at round start,
  twelve of them one production bug (#305); the windowed `attn_span` kernel is a correctness path
  whose throughput is still unmeasured (#315); and a packed read is still *slower* than f16 on CUDA
  (1.23×/1.24× at hd 64, 1.01×/1.04× at hd 128), so enabling it is a capability, not a win.

## References

- `docs/METAL-BACKEND-DESIGN.md` §7.4 — the current Metal record and its baseline.
- `docs/KV-CACHE-DESIGN.md` §1–§3 — the arena contract Metal now shares.
- `ARCHITECTURE-EXECUTION-PLAN.md` — Phase G (7/7 complete) and the `#44` / `#54` / `#53` records.
- Commits: `2c3c395` (2026-09-20, the scheduling decision this ADR is dated at); `0efd4db`
  (2026-10-09, `READS_PACKED_KV` enabled).
