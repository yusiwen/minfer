# 101 · Steady-clock measurement method — warmup budgets landed, headline table re-measured with tight error bars

**Status**: ✅ landed (2026-09-15). doc 100's method rule is now executable:
`specverify` and `bench` warm the GPU on a **time budget** (default 2000 ms,
`MINFER_SPECVERIFY_WARMUP_MS` / `MINFER_BENCH_WARMUP_MS`) instead of 3/1
un-ramped iterations, and the headline table below was re-measured in one hot
session. Same-config repetition tightened from the doc-100 ±7–12% drift band
to **±0.1–0.5 tok/s**.

## 1. The method

The SM idles at 208 MHz (max 3003); seconds-long runs never finish the ramp,
so where a run lands on the curve dominated every sequential comparison (doc
100). Two changes:

- `specverify`: each nt phase warms until the budget is spent (floor 3
  iterations for allocator/graph stability); the JSON reports the budget.
- `bench`: the pp/tg loops warm until the same kind of budget is spent
  (keeping the existing "fresh-context shape" rule).

For end-to-end runs (`minfer --greedy ...`) the recipe is a **throwaway warm
run** (`-n 16`) before each measured run; all rows below were measured that
way, back to back, in one session.

## 2. Headline table (Qwen2.5-14B q4_k_m, 0.5B q4_k_m draft, greedy, −n 200)

| cell | sequential | static d=2 | static d=8 | adaptive (production) |
|---|---|---|---|---|
| prose | 25.3 ± 0.1 | **36.6 ± 0.1** | 25.9 ± 0.1 (collapse) | 35.9 ± 0.2 |
| code | 25.3 ± 0.1 | 41.6 ± 0.1 | 40.7 | **44.8 ± 0.1** |

- Speedup vs sequential: prose **1.42×**, code **1.77×**.
- The doc-95 structure reproduces exactly: on prose, d=8 collapses to
  sequential (the acceptance chain dies by depth 2 while the round cost
  roughly doubles — nt=9 also crosses the multi-MMVQ → BT dispatch boundary);
  on code, adaptive beats the best static (+7.7% over d=2) and adaptive is
  the only configuration that is best-or-near-best on both cells.
- doc 95's recorded absolutes (code adaptive 48.4) sat in a faster machine
  state than today's 44.8 — that gap is the doc-100 drift band, not a
  regression; the relative ordering is identical.

## 3. Steady-state micro benches

| metric | value (steady) | earlier sessions (band) |
|---|---|---|
| pp512 | **2083.0 ± 7.2 tok/s** | 1881–2104 (drift) |
| C_T(1) | 38.91 ms | 39.06–39.91 |
| C_T(3) | 48.23 ms | 48.03–48.55 |
| C_T(5) | 56.08 ms | 55.75–56.30 |
| C_T(9) | ≈ 73 ms (e2e-derived, doc 92) | identity-safe floor |

The C_T ladder was always ramp-insensitive (long phases self-warm); pp512
needed the budget — its ±25 tok/s spread collapsed to ±7.

## 4. Rule

Every kernel- or e2e-level claim on this box: steady-state warmup +
interleaved A/B when comparing builds. Sequential before/after runs are
invalid instruments (doc 100 §4, now executable via the warmup budgets).
