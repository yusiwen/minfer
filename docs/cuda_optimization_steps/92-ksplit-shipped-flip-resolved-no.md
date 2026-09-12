# 92 · D5-R follow-up 3 — K-split shipped behind the gate; the ≤55 flip condition resolved to "no"

> **Result**: the doc-91 §4 K-split is implemented for both BT GEMM kernels (q4_K and q6_K): grid.z slots own contiguous k-tile ranges, write fp32 partials `[ksplit][nt][od]`, and a deterministic two-pass reduce sums them in fixed z order — run-to-run bit-stable, capture-replay parity preserved. Fully gated (`MINFER_SMALL_M_GEMM=1`, default off; the default prefill passes `ksplit=1` and is unchanged). Measured convergence: C_T(9) 86.4 → 74.7 (q4_K split) → **72.9 ms** (+q6_K split); C_T(3) 85.4 → 70.9; C_T(5) 71.8; pp512 = 2005 tok/s (no prefill regression); suite 180 green; e2e acceptance with the gate on is bit-for-bit the doc-88 baseline (50.8% prose / 73.1% code). The pre-registered flip condition — C_T(9) ≤ 55, which would have made the mma path the production route for nt 2..16 — was **not met** (72.9): multi-MMVQ stays the production verify path, and the remaining ~18 ms is a kernel-redesign item, not a tuning item.
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. What shipped

Both BT kernels now take `(Cpart, ksplit)`. With `ksplit > 1` each z-slot accumulates its chunk range in ascending kt order (the same per-slot order the unsplit kernel uses) into `Cpart[((z·nt)+i)·od+j]`, and the shared `mmq_ksplit_reduce_kernel` sums the planes in fixed ascending z order — deterministic, so capture and replay follow the same association (atomicAdd would not have). The launcher refines the requested ksplit so every slot owns a non-empty tile range, launches the reduce on the same stream, and Rust grows the partials buffer on demand. With a K-split active the q4_K kernel drops its double-buffered staging (single 42 KB buffer buys SM residency that the grid's new block-level parallelism makes more valuable); q6_K keeps its r39 double-buffer.

Two bugs were caught and fixed on the way, both worth recording:
- **Regime-inconsistent prologue**: when the launcher allocated a single buffer set (dbuf off) but the slot's first tile parity was odd, the prologue bound `base + BUF` past the allocation — an illegal-address (error 700) that surfaced as NaN timings. The prologue and loop now bind buffers consistently per regime, and the single-buffer loop skips restaging the prologue's tile (`kt > kt_lo`, not `kt > 0`).
- **Silent NaN outputs look like fast runs**: a crashed kernel makes specverify print ~2.5 ms medians with `NaN` aggregates — a 30× "speedup" that is actually an early-exit. Any specverify number should be sanity-checked against its JSON's `gate_pass`/error stream before believing it.

## 2. The measured boundary

| configuration | C_T(3) | C_T(9) |
|---|---:|---:|
| unsplit BT (doc-91 state) | 85.4 | 86.4 |
| + q4_K K-split (target=256) | 72.9 | 74.7 |
| + q6_K K-split (shipped) | **70.9** | **72.9** |
| multi-MMVQ (production nt 2..8) | 48.9 | — |

The ksplit target sweep (128…768) peaks at 256 and degrades beyond — more slots add partials traffic and reduce work faster than they add tile parallelism. The q6_K gain (−1.8 ms) is small because its r39 double-buffered staging had already hidden most of its latency; the q4_K gain (−11.7 ms) recovered the starvation loss but the path is still ~1.7× its weight-stream floor (~40 ms forward for q4_K+q6_K weights at nt=1 rates).

The residual is **per-tile staging serialization**: every k-tile pays a bulk stage → barrier → mma → barrier sequence, and total tile-instances are independent of ksplit (the split only spreads them over more blocks). Closing the remaining gap means fewer, larger tiles (stage 2–3 super-blocks per k-tile to amortize the barriers), a llama.cpp-class ldmatrix B path on raw nibbles, or both — a kernel redesign, not a flag.

## 3. Decision record

- **Dispatch flip: NO.** The pre-registered condition (C_T(9) ≤ 55) failed; multi-MMVQ remains the production nt 2..8 path and the strict bitwise net is untouched. The K-split ships as infrastructure behind `MINFER_SMALL_M_GEMM=1`.
- **Deferred proposal (measured, not flipped)**: auto-ksplit for the *default* nt 9..64 prefill path would take C_T(9) 86.4 → 72.9 today (−13.5 ms) with deterministic capture-replay, since capture and replay would both use the same ksplit. Not enabled unilaterally because it shifts default-prefill numerics (fp32 association) outside any pre-registered condition; it is a one-line `ksplit_req` change when wanted.
- **The campaign's small-M ledger**: doc 89 priced the menu at 1.7 ms/row (R-rows) + mma tiles + hygiene; doc 90 falsified the R-rows pricing and re-confirmed mma as the only lever of size; docs 91–92 built the mma path's missing parallelism and measured its floor at ~1.7× — the verify marginal that remains (~4 ms/row → now ~3.3 ms/row-equivalent in mma terms) is structural for this kernel family. The next real step is the tile-geometry redesign above, or accepting the multi-MMVQ production path and moving to the campaign's open non-kernel leads (nt-invariant accumulation, conversation/server modes).

## 4. Verification

- Suite: 180 passed / 0 failed (default paths; the strict bitwise net intact).
- K-split path: specverify runs at nt 3/5/9 with no parity errors; e2e speculative decode with the gate on produces acceptance identical to baseline (50.8% / 73.1%).
- Prefill: pp512 = 2005 tok/s (≥ the 1977 doc-91 state and the ~1830 historical baseline).
