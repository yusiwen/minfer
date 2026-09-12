# 91 · D5-R follow-up 2 — the mma path was already there; the small-M floor is block starvation

> **Result**: three findings. (1) The doc-85 "2× over the weight-stream floor" mystery at nt≥9 is **not a missing-mma problem** — `mmq_raw_nb_bt_kernel` has been an `mma.m16n8k32` (s8) tensor-core kernel all along; the floor defect is **block starvation**: its grid is `(ceil(nt/64), od/128)` and at nt≤64 the M axis contributes one block row, leaving only ~40 resident blocks for 128+ SMs (measured M-flat: C_T-forward 85.4 ms at nt=3 ≈ 86.4 ms at nt=9, vs the ~30 ms q4_K weight floor). (2) Double-buffering the staging (the q6_K r39 pattern) is worth only ~2–4 ms at small M and, applied unconditionally, costs prefill ~10% through SM-residency loss — shipped **conditional on ntb == 1**, where it is free. (3) The designed fix that can actually move the number is a **K-split** (grid.z + deterministic two-pass reduce), which is the next round's work; it is what would revive d=8.
> **Status**: suite 180 green; prefill pp512 = 1977 tok/s (≥ the ~1830 baseline); mma-at-nt=3 = 83.0 ms behind the multi-MMVQ gate (`MINFER_SMALL_M_GEMM=1`, default off). **Date**: 2026-09-12.

## 1. What this round set out to test

Doc 90 concluded the nt=3 chain residual is per-row arithmetic throughput, removable only by small-M tensor-core tiles, and priced the mma path as a from-scratch kernel project. Before writing one, the obvious question: what is the existing nt≥9 GEMM (`mmq_raw_nb_bt_kernel`) actually made of?

## 2. Finding 1 — the mma path already exists; the floor defect is grid geometry

Reading the kernel family answered it: `mma.h` is included, `mmq_mma_k32` issues `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, and the BT kernel is a tiled mma GEMM with a bulk LDG→STS staging macro (the r18→r59 lineage). The doc-85 ledger's 65.1 ms for the q4_K GEMM portion is this mma kernel running at ~65 GB/s effective — 3–4× under the multi-MMVQ nt=1 kernels' ~230 GB/s.

The geometry explains it. The grid is `(ceil(nt / MMQ_NBI), od / MMQ_NBJ)` with `MMQ_NBI = 64` tokens per A-tile and `MMQ_NBJ = 128` output columns per block. At nt ≤ 64 the M axis contributes exactly one block row, so the whole matmul runs in `od / 128` blocks: 40 blocks for the 14B attention/FFN-down shapes, 108 for ffn_up — while GB10 exposes 128+ SMs. There are not enough blocks to hide the per-K-tile staging latency, and the B stream stalls. At prefill (nt=512 → ntb=8) the same kernel has 320+ blocks and reaches the bandwidth floor — which is why the defect was invisible until the small-M regime was measured directly.

Measured M-flatness (forward pass with all matmuls routed to the BT path via the new `MINFER_SMALL_M_GEMM=1` gate, nt 2..8):

| nt | C_T forward (BT path) | multi-MMVQ reference |
|---:|---:|---:|
| 3 | 85.4 ms | 48.9 ms |
| 9 | 86.4 ms | (GEMM is the default here) |

The GEMM is flat in M — the 16-row mma-tile padding costs nothing measurable — and simultaneously 1.7× worse than the dp4a multi kernels at nt=3. The prize stands but the lever moved: fix the block starvation and the same kernel beats multi at every nt.

## 3. Finding 2 — double-buffering: small win at small M, prefill landmine

The single-buffer main loop stages tile kt (`RAW_STAGE_NB_BT(kt)`: 16 KB A + 16 KB B bulk LDG→STS, plus the DSC `cp.async` scale stream), barriers, computes, barriers — every K-tile pays full staging latency serially. Mirroring the q6_K r39 pattern, the loop now prefetches tile kt+1 into a second buffer set while tile kt computes (`cp.async.wait_group 1` lets the next tile's scale stream stay in flight). The restructure moves data only — fragments, operand values and accumulation order are unchanged, so results are bitwise identical to the single-buffer kernel; the strict bitwise suite confirms it.

The footprint, however, doubles to 84 KB per block, dropping residency from ~4 to 2 blocks/SM — harmless when only 40 blocks exist (small M) and costly when 320 do (prefill): unconditional double-buffering measured pp512 at 1637 tok/s, −10% versus the ~1830 baseline. Shipped therefore **conditional**: the kernel takes `dbuf = nt <= MMQ_NBI` and the launcher sizes shared memory to match; prefill keeps the original 42 KB single-buffer sequence byte-for-byte.

| variant | pp512 | mma C_T(3) | suite |
|---|---:|---:|---|
| original (single-buffer) | ~1830 | 85.4 | 180 ✓ |
| double-buffer, unconditional | 1637 (−10%) | 81.6 | (dispatch-gate failures, see §5) |
| double-buffer, ntb==1 only (shipped) | 1977 | 83.0 | 180 ✓ |

The pp512 number at 1977 sits above the historical ~1830; treat the +8% as clock variance until re-measured, the headline is "no regression".

## 4. Finding 3 — the fix that matters is a K-split (next round)

Pipelining hides per-tile latency inside a block; starvation means there are too few blocks to hide it **between** blocks. The designed follow-up: split the K dimension across `grid.z` — each z-slot computes a chunk range and writes fp32 partials `[ksplit][nt][od]`, then a small deterministic reduce kernel sums the slots in fixed order (run-to-run bit-stable, preserving the capture-replay parity contract that `atomicAdd` would break). Launcher picks `ksplit = ceil(256 / (od / MMQ_NBJ))` (≥ 256 blocks ≈ 2/SM), partial traffic at nt=9 is ~1 MB per matmul. Expected outcomes if the weight stream then reaches the floor: mma at nt=3 ≈ multi (≈ 45–50 ms forward) and — the bigger prize — **C_T(9) ≈ 86 → ~50 ms**, which would make d=8 speculative decoding competitive (code-class ~50 tok/s vs llama d=2's 48.4). Dispatch consequence if it lands: nt 2..16 all route to the BT path, multi-MMVQ retires to the nt=1 bitwise kernel, and the GEMM cliff at nt≥9 disappears.

## 5. Process notes

- `mmq_gate_on(name)` is default-ON (`true` unless the variable equals `"0"`) — the experiment gate needed an explicit `== "1"` opt-in instead. Two identical-looking runs (85.4 / 85.4) were both the mma path before this was noticed; the multi reference requires `MINFER_SMALL_M_GEMM=0`.
- With the gate default-on, three suite tests failed for the *right* reason — the bitwise test's batched nt=3 case legitimately routed to the mma path and is not bitwise against the nt=1 single kernels. Default-off restored 180 green without touching the assertions.
- nvcc reports pre-existing `sm_70`/`sm_72` fatal-arch noise in the build log that the build script tolerates; the C++ error surface is the only real signal.

## 6. Next steps

1. **K-split** (§4 design) behind the same `MINFER_SMALL_M_GEMM` gate; validate: suite, capture-replay parity, C_T(3)/C_T(9), pp512, then flip the dispatch default for nt 2..16 and re-run the doc-88 same-window battery + e2e acceptance.
2. If the K-split lands the floor, retire the multi-MMVQ nt 2..8 dispatch (keeping the nt=1 single kernels and their bitwise net) and re-price d=4/d=8 with the doc-88 acceptance model.
3. The cold-L2 bench (`MINFER_BENCH_ROW_MARGINAL=1`) gains BT-path cases when the K-split lands, as the regression instrument for the new dispatch boundary.
