# 96 · nt=9 Verify Profile — Phase 0 (stop-gate measured)

**Status**: ✅ Phase 0 done (2026-09-15). **Stop-gate verdict: PROCEED** — the staging-stall share
is 40–59%, well above the pre-registered 30% threshold. Phase 1 (tile/staging redesign) has a
mechanism-backed target; re-priced separately below.

## 1. Setup

Pre-registered Phase 0 (user-approved): profile the nt=9 verify forward, split the per-kernel
timeline (BT-MMQ / ksplit reduce / attention / quantize), measure the staging-vs-mma stall share
inside the BT kernel, and apply the stop-gate: **staging stall share < 30% → re-price or stop** —
doc 92's tile-geometry redesign premise dies if staging serialization is not the dominant residual.

Captures (GB10, Qwen2.5-14B q4_K_M, code prompt):

- `nsys profile --trace cuda minfer --greedy -n 64 --spec-draft <0.5B q4_K_M> --spec-draft-n 8`
  — the production d=8 mixture (draft nt=1 + verify nt=9 + repairs).
- `ncu --section SpeedOfLight --section WarpStateStats` on the BT kernels (12 instances).

## 2. nsys kernel timeline (d=8 spec run, 64 tokens)

| kernel | time % | instances | avg / med µs |
|---|---|---|---|
| `mmq_raw_nb_bt_kernel` (Q4_K BT, nt≥9) | **64.8%** | 873 | 187.6 / 142.4 |
| `mmq_raw_nb_bt_q6k_kernel` (Q6_K BT) | **18.7%** | 157 | 301.7 / 56.3 (max 4.83 ms = lm_head) |
| `mmq_nt_kernel<3>` (draft prefill) | 2.6% | 130 | 49.8 |
| `quantize_q8_0_pad40_t` | 1.7% | 434 | 10.2 |
| `mmq_ksplit_reduce_kernel` | 1.3% | 1 030 | 4.2 |
| `gqa_attn_split_partial_bt` + combine + fa_prefill | ~1.0% | 216 | — |
| everything else (norms, rope, store_kv, adds) | ~7% | — | — |

The verify path at nt=9 is **~84% BT-MMQ GEMM**; attention is ~1%, the ksplit reduce ~1.3%. There
is no second lever of size: whatever the doc-92 residual is, it lives inside the two BT kernels.

## 3. ncu inside the BT kernels — the stall share

SpeedOfLight (12 instances across Q4_K (grids (1,40,7), (1,8,20), (1,108,3)) and Q6_K variants):

- Memory throughput **14.7–26.0%**, Compute (SM) throughput **10.8–23.6%** — neither ceiling is
  approached; the kernels are latency-bound at ~1/5 of both roofs.
- Warp Cycles Per Issued Instruction **15.5–24.0**.

WarpStateStats: the dominant stall reason in every instance is the **shared-memory scoreboard
dependency (Stall Short Scoreboard): 39.9–58.7% of the total average warp-stall cycles**
(8.4–10.4 of 15.5–24.0 cycles). This is doc 92's "per-tile staging serialization" measured
directly: warps wait on smem loads for the staged quantized-weight/activation tiles.

**Stop-gate: 40–59% ≥ 30% → PROCEED.** The staging share is not marginal; it is the mechanism.

## 4. Phase 1 re-pricing (from this data, not guesswork)

- The fix class that preserves the k-accumulation order — **double-buffered (cp.async) staging**
  so smem load latency is overlapped with mma work — is **bitwise-preserving by construction**
  (arithmetic unchanged, only arrival timing). That keeps the BT path run-to-run deterministic and
  leaves the doc-95 identity facts untouched. Larger k-tiles / ldmatrix raw-nibble B would change
  accumulation grouping → tolerance-class → only usable where identity is not claimed.
- Ceiling if staging exposure is mostly hidden: the kernels run at ~20% of roofs; a 1.5–2× kernel
  win is plausible → C_T(9) 73 → ~40–50 ms → the d=8 verify path approaches the llama-MMQ class.
  Realistic wall impact on the spec cells requires the d=8 door anyway (acceptance-bound on prose;
  code is the only cell that could use it — and doc 95's adaptive d≈3.5 already beats static d=8
  while identity-safe, so the payoff is **narrow: it re-opens d=8 only if the BT path also becomes
  bitwise vs MMVQ, which the larger-tile variant is not**). Phase 1's honest EV: a prefill/multi-
  token lever and a d=8 speed win for identity-relaxed use — not a spec-throughput lever.
- Recommendation: Phase 1 as a **separately-priced campaign** (as planned), prioritized BELOW
  战役 97 (conversation/server integration) — the user-facing integration is worth more than a
  narrow d=8 speed win.

## 5. Tooling notes

- `ncu` requires GPU performance-counter permission: **runs under `sudo` on this box**
  (`ERR_NVGPUCTRPERM` otherwise) — doc 90's "counter permissions" open lead is resolved for
  interactive profiling.
- `ncu --kernel-name` matches the *full* demangled name; use
  `--kernel-name-base demangled --kernel-name "regex:.*<substr>.*"`.
- The specverify instrument spans nt=1/3/5 only; nt=9 timing comes from end-to-end d=8 runs
  (C_T(9) = 73.0 ms, doc 92/94). Extending specverify's nt ladder is cheap if Phase 1 lands.
