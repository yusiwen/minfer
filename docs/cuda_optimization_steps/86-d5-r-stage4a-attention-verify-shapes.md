# 86 · D5-R stage 4a — attention for the verify shapes (LANDED)

> **Result**: one gate change — `fa_prefill` now serves nt ≥ 2 (was nt ≥ 64): C_T(3) **56.90 → 48.78 ms (−8.1)**, C_T(9) **101.3 → 86.44 ms (−14.9)**, C_T(1) untouched (38.98; nt=1 keeps the split-KV path). End-to-end 14B+0.5B d=2: **35.5 / 42.4 tok/s = 1.39× / 1.66×** (was 1.33×/1.59×) — code passes llama's same-window ratio (1.64×). Suite 179 green.
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. Background — where things stood

The doc 85 ledger priced the verify marginal and put the batched-attention
hole first: verify shapes (nt = d+1, i.e. 2..=9) ran the legacy
per-(token,head) kernel at 9.8 ms (nt=3) where the nt=1 split-KV path does
the same KV read in 0.8 ms — a shape range that had no caller before
speculative decoding, so the kernel choice there was never revisited.

## 2. Principle — why the FA prefill kernel covers verify rows

`fa_prefill_f16kv` computes rows of a block against the KV prefix with
per-row causal masking driven by the positions array — "positions are
data", no topology dependence. The verify block has exactly that structure:
rows at pos..pos+d, each attending to KV[..pos+j] (stale draft slots beyond
the row's position are masked off). What differs from prefill is only the
block length, so the kernel that wins at nt=512 should also win at nt=3 —
the open question was purely empirical (the gate had never been exercised
below 64), and the doc 85 projection (~1.0–1.5 ms vs 9.8) predicted the
prize.

## 3. Implementation

One condition in `src/cuda.rs` (`gqa_attn_f16kv` dispatch):
`nt >= 64` → `nt >= 2`, with the comment recording why. The existing
guards stay: `hd == 128` (the 0.5B q4_0 draft has hd=64 and keeps its
old path), `MINFER_NO_FA_PREFILL=1` kill switch, and the `rc != 0`
fallback to the legacy kernel. nt == 1 keeps the split-KV decode path —
the d=0 fallback and the entire non-spec decode path are untouched.

Numerics: the verify stream's kernel assignment changes, so its greedy
chain flaps move (G1c class, expected); the d=0 fallback and spec-off
paths are kernel-identical to before. Acceptance at 14B d=2 measured
50.8% vs 51.6% — same class, prompt-noise level.

## 4. Results

specverify (14B q4_k_m, KV 512, medians):

| nt | before | after | Δ |
|---|---|---|---|
| 1 | 39.34 | 38.98 | unchanged (split path) |
| 3 | 56.90 | **48.78** | **−8.12** (ledger predicted ~8.3–9.0) |
| 9 | 101.3 | **86.44** | **−14.9** (ledger predicted ~14–19) |

End-to-end (doc-84 protocol, 14B+0.5B d=2, greedy n=128):

| prompt | before (doc 84) | after | speedup vs serial 25.5 |
|---|---|---|---|
| prose | 34.0 (1.33×) | **35.5** | **1.39×** |
| code | 40.6 (1.59×) | **42.4** | **1.66×** — above llama's same-window 1.64× |

llama same-window absolutes: 38.8 prose / 49.0 code → minfer now at 92% /
87% of llama's absolute speculative speed (was 88% / 83%).

The doc 85 recovery model (attention −8.5, matmul −4, capture −2 → 1.64×)
is on track: item (1) alone delivered most of the attention term; the
remaining items are the multi-MMVQ nt 9–16 extension (prize ~14 ms at
nt=9, irrelevant at d=2) and graph capture (~2 ms/round).

## 5. What was learned

- The ledger's projection validated within ~5% end-to-end — the per-kernel
  ledger is a reliable spending guide.
- A "never exercised shape range" hid a 12× kernel-choice mistake behind a
  gate written for a different caller; kill switches
  (`MINFER_NO_FA_PREFILL`) and rc-fallbacks made the fix a one-liner
  instead of a new kernel.
- G1 gates behaved as designed: the flap point moved (byte 552 → 497 on
  the 14B d=0-vs-off check), acceptance stayed in class, d=0/spec-off
  paths untouched.

## 6. Next steps

Stage 4b: multi-MMVQ nt 9–16 extension or a small-M GEMM tile (doc 82's
acc-register cap to re-derive) — targets d=8 (C_T(9) 86.4 → ~72 projected
with matmul at the multi-MMVQ slope; round ≈ 72+23+overhead → ~1.5×).
Stage 4c: CUDA-graph capture for the fixed verify shapes (~2 ms/round).
Then stage ⑤ re-runs the doc-84 dual-engine battery with d=8 included.
