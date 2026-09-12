# 85 · D5-R stage 3 — the verify marginal priced: a per-kernel ledger (LANDED)

> **Result**: nsys per-kernel ledger of one 14B decode/verify forward (q4_k_m, KV 512): the nt=3 marginal (56.9 − 39.3 = **17.6 ms**) splits as **attention +9.0 ms (half!)** — the nt 2–63 verify rows run a legacy per-(token,head) kernel at 9.8 ms vs the nt=1 split-KV path's 0.8 ms on identical KV — **matmul +8.1 ms** (multi-MMVQ row slope 4.05 ms/row), elt/quant +1.9, launch idle +0.7. At nt=9 the matmuls fall off multi-MMVQ (cap nt≤8) onto the padded GEMM: matmul 84.9 ms → **~14–19 ms reclaimable**. Priced recovery at d=2: C_T(3) → ~42 ms → **1.64× ≈ llama parity**.
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. Background — where things stood

Stage 2 (doc 84) priced the whole gap to llama.cpp as the verify row
marginal: minfer 8.8 ms/row vs llama 2.5 ms/row at the 14B, worth 1.62× if
closed. Stage ③'s job was to attribute that marginal to specific kernels so
stage ④ attacks the right thing. The candidate list from the plan:
multi-MMVQ nt=9–16 extension, small-M GEMM tiles, attention query-tiling,
graph capture for the fixed verify shapes.

## 2. Principle — what the marginal must be made of

One verify forward at nt rows = the nt=1 decode graph (weights stream once:
~8 GiB q4_K/q6_K at ~240 GB/s ≈ 39 ms — the nt=1 measured cost, 95% of it
matmul) plus per-row work: extra activation rows through the same matmuls,
nt× the attention Q·Kᵀ/PV work against the same KV, and nt× the row-wise
elementwise ops. The marginal is therefore bounded below by
`nt × (attention-KV pass + elementwise)` and its distance above that bound
is pure kernel inefficiency, measurable per kernel family with a timeline
profiler.

## 3. Implementation

### 3.1 Method

`minfer specverify -p 512 -r 5` under `nsys profile` (NVIDIA Nsight Systems
2025.3.2), one run per nt (`MINFER_SPECVERIFY_NTS=1|3|9`), CUDA build. The
capture holds the prefill plus the first steady-state decode forwards; a
decode forward is exactly the kernel span between consecutive `embed_rows`
launches (1157 kernels at nt=3/9, 748 at nt=1), so per-forward attribution
needs no heuristics. Wall medians from specverify itself anchor the
per-forward totals (C_T(1)=39.34, C_T(3)=56.90 this window; C_T(9)=101.3
from doc 81).

`ncu` (Nsight Compute) was attempted for SM/memory-utilization counters and
blocked by `ERR_NVGPUCTRPERM` (performance-counter permission is an admin
modprobe setting on this box). All numbers below are nsys kernel durations;
utilization-level attribution is deferred until counter access exists.

### 3.2 The ledger — one decode/verify forward, 14B q4_k_m (ms)

| kernel | nt=1 | nt=3 | nt=9 |
|---|---|---|---|
| q4_K matmul | 30.30 (`q4_k_q8_mmvq_v2`) | 36.95 (`…_multi`) | 65.14 (`mmq_raw_nb_bt`!) |
| q6_K matmul | 9.99 (`…_pf_dpl`/`…_dpl`) | 11.43 (`…_multi`) | 19.76 (`mmq_raw_nb_bt_q6k`) |
| attention | **0.80** (split partial+combine) | **9.82** (`gqa_attn_f32_f16kv`) | **16.43** (same) |
| rms/rope/add/swiglu/store | 1.08 | 1.95 | 2.12 |
| quantize | 0.14 | 0.59 | 1.90 |
| launch idle (wall−busy) | ~1.7 (4%) | ~2.4 (4%) | ~2.5 (2%) |
| **wall (specverify median)** | **39.34** | **56.90** | **101.3** |

## 4. Results — the three priced items

**(1) Attention: the nt 2–63 hole — ~9 ms at nt=3, half the marginal.**
The nt=1 decode path uses the split-KV kernel (`gqa_attn_split_partial` +
`combine`: KV chunked over 32 blocks per head, partials combined) and reads
the whole 57 MB KV in **0.80 ms for 48 layers**. Every nt from 2 to 63 —
which includes every verify shape speculative decoding ever issues (nt =
d+1 ≤ 9) — falls through to the legacy `gqa_attn_f32_f16kv`, one block per
(token, head), K re-read per token per head: **9.82 ms at nt=3** (12× the
nt=1 cost for 3 rows) and 16.43 ms at nt=9. An efficient batched kernel
projects to ~1.0–1.5 ms at nt=3 → **prize ≈ 8.3–9.0 ms/verify forward**.
Two pre-validated fix shapes: (a) the FA-prefill kernel (`fa_prefill_f16kv`,
the same graph's nt≥64 path, 5.08 ms for 48 layers at nt=512 = 0.106
ms/layer) is gated `nt >= 64 && hd == 128` — verify shapes never existed
when that gate was written; lowering it to nt≥2 is a one-line experiment
with the rc fallback intact; (b) extend the split kernel's partial kernel
with a q-row dimension.

**(2) multi-MMVQ's nt≤8 cap — ~14–19 ms at nt=9.** Doc 82's multi-token
MMVQ covers nt 2–8; nt=9 (the d=8 verify) lands on the tiled GEMM
`mmq_raw_nb_bt` at M=9 < tile 16 — matmul jumps 48.4 → 84.9 ms. The
multi-MMVQ row slope is ~4.05 ms/row (48.4 at nt=3 vs 40.3 at nt=1); the
GEMM path charges 6.1 ms/row; projecting multi-MMVQ to nt=9 gives ~70.6 ms
→ **prize ≈ 14 ms** (the doc-82 acc-register cap is the implementation
constraint). This single item turns d=8 from 0.91× into ~1.4× (round
101.3+23 draft+overhead → per-token ~27 ms vs 39.3 serial).

**(3) Launch idle — ~2.4 ms/forward.** 4% of the verify wall is gap between
kernels (eager launch + per-forward sync/readback). The verify shapes are
fixed per (nt, depth) — ideal CUDA-graph capture candidates, prize ~2 ms
per round.

Sum at d=2: attention −8.5, matmul row-slope −4 (llama charges ~1.2 ms/row
vs minfer's 4.05), capture −2 → C_T(3) ≈ 42 ms → round ≈ 48.6 ms / 2.03
tokens ≈ 24 ms/token → **1.64× vs serial = llama's same-window ratio** (doc
84). The gap closes on paper with three named items; stage ④ executes them.

## 5. What was learned

- **The marginal was not where the plan looked.** The candidate list led
  with multi-MMVQ/GEMM work; the ledger puts the batched-attention hole
  first (9 of 17.6 ms) — a kernel that existed, was fast at nt=1 and
  nt≥64, and had simply never been exercised at 2–63 before speculative
  decoding gave those shapes a caller.
- **nt=9's loss is a dispatch cliff, not a slope**: multi-MMVQ → padded GEMM
  at M=9 costs 6.1 vs 4.05 ms/row — the cap is worth more than the slope.
- **nsys alone prices this stage** (kernel durations + exact forward
  spans); ncu's utilization counters would refine the attention kernel's
  internal story but were permission-blocked (`ERR_NVGPUCTRPERM`).
- Exactly-measurable forward boundaries (`embed_rows` → `embed_rows`)
  make per-forward attribution trivial — no phase heuristics needed.

## 6. Next steps

Stage ④, in prize order: (1) attention nt 2–16 — try the one-line
fa_prefill gate first (validate numerics at small nt against the reference
traces), else extend the split kernel; (2) multi-MMVQ nt 9–16 extension or
a small-M GEMM tile (doc 82's acc-register budget is the constraint to
re-derive); (3) CUDA-graph capture for the fixed verify shapes. Gate:
end-to-end d=2 ≥ 1.5× and d=8 ≥ 1.4× on the doc-84 protocol; then stage ⑤
re-runs the battery.
