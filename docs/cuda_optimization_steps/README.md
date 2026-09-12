# CUDA Optimization Step Documents — Master Index

This directory is the expansion layer of [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md):
it writes **every step** of the twelve sessions, the two Phase-8 batch records
(78–79), and ~60 optimization levers from
2026-08-29 → 2026-09-09 as a standalone, readable document — background, the GPU
principle (arguing with arithmetic), real code before/after, verification methods,
results, and lessons.

**Reading guidance**: only care about the current state → read
[`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) §1 (current status) and the
[doc 77 methodology](77-verification-methodology.md) at the end of this directory;
want to understand why a mechanism is the way it is → find that step in the table
below; want to follow the whole campaign → read in number order — the story is
continuous.

**Status legend**: 🟢 LANDED · 🔵 MEAS-ONLY · 🔴 REVERTED (with the veto mechanism) · ⚪ CLOSED (no code, or analytical conclusions)

The writing contract is in [STYLE.md](STYLE.md).

---

## Part I · Foundations (Era A: Phase 7/8, 2026-08-29 → 08-30)

| # | doc | topic | status |
|---|---|---|---|
| 01 | [phase7-cuda-backend](01-phase7-cuda-backend.md) | CUDA backend: raw-FFI device layer + graph backend (the 30.7 tok/s starting point) | 🟢 |
| 02 | [wmma-f16-prefill-gemm-8m](02-wmma-f16-prefill-gemm-8m.md) | wmma f16 tiled GEMM: 30.7→1204 tok/s (39×) | 🟢 |
| 03 | [fa-tiled-prefill-attention-8n](03-fa-tiled-prefill-attention-8n.md) | FA-style tiled prefill attention: 176→8.5 ms/layer (20×) | 🟢 |
| 04 | [decode-start-stall-8o](04-decode-start-stall-8o.md) | decode start stall: killing the 635 ms heavyweight clone (724→35 ms first step) | 🟢 |
| 05 | [persistent-f16-cache-8p](05-persistent-f16-cache-8p.md) | resident f16 weight cache + dequant folded into the GEMM (→~1400 tok/s) | 🟢 |
| 06 | [decode-mmvq-8e](06-decode-mmvq-8e.md) | decode MMVQ (dp4a × q8_0): +37% on q4_K | 🟢 |
| 78 | [phase8-correctness-batch](78-phase8-correctness-batch.md) | the 8a review batch (11 fixes) + the F32-matmul latent bug + 8h①/8i tests | 🟢 |
| 79 | [phase8-coverage-batch](79-phase8-coverage-batch.md) | KV f16, shaped Q8_0 GEMM, split-K attention, Q5_K/Q5_1/Q5_0 kernels, the 8l llama baseline | 🟢/🔵 |

## Part II · The R-and-P5 sessions (Era B: R/P5, 2026-08-31 → 09-01)

| # | doc | topic | status |
|---|---|---|---|
| 07 | [r3-small-model-overhead](07-r3-small-model-overhead.md) | small-model per-token overhead: prefill single-split | 🟢 |
| 08 | [r1-int8-mmq-prefill-gemm](08-r1-int8-mmq-prefill-gemm.md) | int8 MMQ prefill GEMM (opt-in): the parity-first strategy | 🟢 |
| 09 | [r2-mmvq-weight-streaming](09-r2-mmvq-weight-streaming.md) | MMVQ weight-streaming rework: tg128 +6.9% | 🟢 |
| 10 | [r4-split-attention-dim-parallel](10-r4-split-attention-dim-parallel.md) | split-attention dim-parallel rewrite: the bulk of the @2K gap | 🟢 |
| 11 | [p5-gemm-tiles-fa-rewrite](11-p5-gemm-tiles-fa-rewrite.md) | P5: TM=128 big tiles + FA rewrite (2.37×→1.43×) | 🟢 |

## Part III · The q4_K campaign (Era C: r5–r37, 2026-08-31 → 09-05)

| # | doc | topic | status |
|---|---|---|---|
| 12 | [r5-r6-rerank-rewrite-spec](12-r5-r6-rerank-rewrite-spec.md) | re-ranking + the structural rewrite spec | ⚪ |
| 13 | [r7-r8-raw-byte-kernel](13-r7-r8-raw-byte-kernel.md) | raw-byte kernel + wide tile + FA probe | 🟢 |
| 14 | [r9-llama-mmq-reference](14-r9-llama-mmq-reference.md) | llama.cpp MMQ reference decode; the shape axis closed | ⚪ |
| 15 | [r10-r11-inner-loop-port](15-r10-r11-inner-loop-port.md) | reference inner-loop decomposition port; ILP verification | 🟢 |
| 16 | [r12-warp-tile-ldmatrix](16-r12-warp-tile-ldmatrix.md) | 16-chain warp tile + ldmatrix | 🟢 |
| 17 | [staging-shape-family](17-staging-shape-family.md) | the x-tile / j-tile / cp.async-db staging shape family | ⚪ |
| 18 | [r13-counter-forensics](18-r13-counter-forensics.md) | counter forensics vs llama.cpp | ⚪ |
| 19 | [r14-b-fragments-ldmatrix](19-r14-b-fragments-ldmatrix.md) | ldmatrix B fragments + widened scale reads | 🟢 |
| 20 | [r15-f32-acc-mma-rank1](20-r15-f32-acc-mma-rank1.md) | f32-accumulate mma probe + rank-1 term2 rescale | 🟢 |
| 21 | [r16-narrow-kernel-rank1-fold](21-r16-narrow-kernel-rank1-fold.md) | the narrow kernel gains the rank-1 fold | 🟢 |
| 22 | [r17-wide-warp-remap](22-r17-wide-warp-remap.md) | wide warp remap 32od×64tok | 🔴 |
| 23 | [r18-load-time-b-preexpansion](23-r18-load-time-b-preexpansion.md) | load-time B pre-expansion (W_exp's debut) | 🔴 |
| 24 | [r19-weight-l2-residency](24-r19-weight-l2-residency.md) | weight L2 residency | 🔴 |
| 25 | [r20-split-phase-a-staging](25-r20-split-phase-a-staging.md) | split-phase A staging | 🟢 |
| 26 | [r21-coalesced-block-linear-a](26-r21-coalesced-block-linear-a.md) | block-linear coalesced A staging | 🔴 |
| 27 | [r22-qa8-xor-swizzle](27-r22-qa8-xor-swizzle.md) | qa8 XOR swizzle (the d/ssum fold reverted separately) | 🟢 |
| 28 | [r23-f16-wall-decomposition](28-r23-f16-wall-decomposition.md) | f16-path wall decomposition + FA_TKV lift | 🔴 |
| 29 | [r24-scheduling-ladder](29-r24-scheduling-ladder.md) | the scheduling-structure ladder (the +1.5% bar calibrated) | 🔴 |
| 30 | [r25-sass-opcode-census](30-r25-sass-opcode-census.md) | the SASS opcode census; the unroll wall's inertia | ⚪ |
| 31 | [r28-nb-kernel-2blocks](31-r28-nb-kernel-2blocks.md) | the Direction-A raw-nibble NB kernel, 2 blocks/SM | 🟢 |
| 32 | [r29-nb-kd-loop-unroll](32-r29-nb-kd-loop-unroll.md) | the NB kd-loop unroll | 🟢 |
| 33 | [r30-swar-unpack](33-r30-swar-unpack.md) | SWAR unpack: the compiler already did it | ⚪ |
| 34 | [r31-qmajor-sda-repack](34-r31-qmajor-sda-repack.md) | the q-major sda scale-read rearrangement | 🟢 |
| 35 | [r32-finite-lever-sweep](35-r32-finite-lever-sweep.md) | the finite-lever sweep: both regions bounded | ⚪ |
| 36 | [r33-hybrid-inner-loop](36-r33-hybrid-inner-loop.md) | the hybrid inner-loop port: falsified by SASS identity | 🔴 |
| 37 | [r34-quantize-transpose-prepass](37-r34-quantize-transpose-prepass.md) | the quantize-transpose prepass (+9.72% of layout-transform locality) | 🟢 |
| 38 | [r35-scale-predecode](38-r35-scale-predecode.md) | scale pre-decode | 🔴 |
| 39 | [r36-a-frag-wavefront](39-r36-a-frag-wavefront.md) | A-fragment wavefront economics: H1 falsified | ⚪ |
| 40 | [r37-post-parity-attribution](40-r37-post-parity-attribution.md) | post-parity whole-prefill attribution | ⚪ |

## Part IV · q6_K-FA and the promotion (Era D: r38–r60, 2026-09-05 → 09-06)

| # | doc | topic | status |
|---|---|---|---|
| 41 | [r38-q6k-bt-rawbyte-mma](41-r38-q6k-bt-rawbyte-mma.md) | the q6_K BT-style raw-byte mma kernel (+2.9%) | 🟢 |
| 42 | [r39-q6k-kdr2-double-buffer](42-r39-q6k-kdr2-double-buffer.md) | q6_K KDR=2 double buffer (+13.3%) | 🟢 |
| 43 | [r40-third-resident-block](43-r40-third-resident-block.md) | the `launch_bounds(256,3)` third resident block (+13.0%) | 🟢 |
| 44 | [r41-q6k-bexpand-uint4](44-r41-q6k-bexpand-uint4.md) | q6_K B-expand uint4 widen (+30.7%) | 🟢 |
| 45 | [r42-stage-wide-dsc-read](45-r42-stage-wide-dsc-read.md) | stage-wide dsc scale reads | 🔴 |
| 46 | [r43-pc-sampling-attribution](46-r43-pc-sampling-attribution.md) | PC-sampling attribution; the pre-expand-B parity FAIL | ⚪ |
| 47 | [r44-wexp-stride-mismatch](47-r44-wexp-stride-mismatch.md) | the W_exp stride-mismatch root cause; the fix wall-neutral | 🔴 |
| 48 | [r45-cpasync-q6k-a-staging](48-r45-cpasync-q6k-a-staging.md) | cp.async q6_K A-side staging | 🔴 |
| 49 | [r46-fap1-fa-audit](49-r46-fap1-fa-audit.md) | FAP1: the FA audit + occupancy/conflict levers (−11% kernel but wall-neutral) | 🔴 |
| 50 | [r47-converged-wall-decomposition](50-r47-converged-wall-decomposition.md) | the converged-regime wall decomposition | ⚪ |
| 51 | [r48-fap2-register-softmax](51-r48-fap2-register-softmax.md) | FAP2: register-resident softmax (FA 2.43×, prefill +5.6%) | 🟢 |
| 52 | [r49-a-quantize-shared-dedup](52-r49-a-quantize-shared-dedup.md) | the A-quantize prepass's shared-A dedup | 🟢 |
| 53 | [r50-fa-tkv-16](53-r50-fa-tkv-16.md) | FA_TKV 32→16 | 🔴 |
| 54 | [r51-producer-fused-a-quantize](54-r51-producer-fused-a-quantize.md) | producer-fused A quantize (mode 1) | 🟢 |
| 55 | [r52-skip-write-mode2](55-r52-skip-write-mode2.md) | skip-write mode 2 (`MINFER_MMQ_A_FUSE=2`) | 🟢 |
| 56 | [r53-q6k-wexp-cpasync-bundle](56-r53-q6k-wexp-cpasync-bundle.md) | the q6_K bundle: W_exp + cp.async B staging (prefill 1.05×) | 🟢 |
| 57 | [r54-q6k-exp-optout](57-r54-q6k-exp-optout.md) | the `MINFER_MMQ_Q6K_EXP` opt-out (−5.04% for 1.52 GB back) | 🟢 |
| 58 | [r55-swiglu-roofline-prefill-graph](58-r55-swiglu-roofline-prefill-graph.md) | the swiglu roofline + prefill CUDA-Graph (both recorded on the books and skipped) | ⚪ |
| 59 | [r56-q6k-a-cpasync-wdsc](59-r56-q6k-a-cpasync-wdsc.md) | the q6_K A-side bundle: A cp.async + the W_dsc plane | 🟢 |
| 60 | [r57-fa-kv-staging-db](60-r57-fa-kv-staging-db.md) | FA KV staging double buffer | 🔴 |
| 61 | [r58-q4k-bt-cpasync-transplant](61-r58-q4k-bt-cpasync-transplant.md) | the q4_K BT spec + cp.async-db2 transplant (−12.6%, the pipeline value formula) | 🔴 |
| 62 | [r59-q4k-wdsc-plane](62-r59-q4k-wdsc-plane.md) | the q4_K W_dsc plane + riders | 🟢 |
| 63 | [r59b-clean-remeasure](63-r59b-clean-remeasure.md) | the clean re-measurement + the baseline-pollution correction | ⚪ |
| 64 | [r60-promotion-default-on](64-r60-promotion-default-on.md) | the promotion: the verified gate set flipped default-on (the 1.080× path) | 🟢 |

## Part V · The decode campaign (§2D: D1–D4-4, 2026-09-07 → 09-09)

| # | doc | topic | status |
|---|---|---|---|
| 65 | [d1-decode-attribution](65-d1-decode-attribution.md) | D1 attribution: the split-attention staging depth is the only kernel that grows with KV | ⚪ |
| 66 | [d2-kv-register-staging](66-d2-kv-register-staging.md) | D2: explicit K+V register staging (+2.0% @1641) + two negative results | 🟢 |
| 67 | [d3-14b-attribution-bitwise-mmvq](67-d3-14b-attribution-bitwise-mmvq.md) | D3-1 14B attribution + the D3b bitwise MMVQ trio (1a/1b/1c) | 🟢/🔴 |
| 68 | [d3a-fattn-rewrite-rpw](68-d3a-fattn-rewrite-rpw.md) | D3a: the 4-warp fattn rewrite reverted (rpw pathology) + the tolerance gate package calibrated | 🔴 |
| 69 | [d3-4-hybrid-rpw-dispatch](69-d3-4-hybrid-rpw-dispatch.md) | D3-4: the hybrid rpw dual-kernel dispatch landed; the L2 prefetch reverted | 🟢 |
| 70 | [d3-5-fused-producer-a-quantize](70-d3-5-fused-producer-a-quantize.md) | D3-5: fused-producer decode A quantize (quantize launches −78%) | 🟢 |
| 71 | [d3-6-gqa-batching-reverted](71-d3-6-gqa-batching-reverted.md) | D3-6: GQA batching — all gates green, still reverted; the 5× L2 re-read was not the residual | 🔴 |
| 72 | [d3-7-attnv-mmvq-rms](72-d3-7-attnv-mmvq-rms.md) | D3-7: attn_v MMVQ routing + the rms wide-block/positions memo | 🟢 |
| 73 | [d3-8-fusedqkv-port](73-d3-8-fusedqkv-port.md) | D3-8: FusedQKV ported to CUDA (both layer classes covered, short KV breaks through) | 🟢 |
| 74 | [d4-2-b0-correctness-fix](74-d4-2-b0-correctness-fix.md) | D4-2: the B0 latent correctness fix (7B dropped 13.5% of down-proj) + all bitwise axes closed | 🟢 |
| 75 | [d4-3-attention-attempt2-artifact](75-d4-3-attention-attempt2-artifact.md) | D4-3: attention attempt 2 NO-GO + the llama-bench artifact correction | 🔴 |
| 76 | [d4-4-dpl-q6k-final](76-d4-4-dpl-q6k-final.md) | D4-4: dpl dense split-plane q6_K (+5.5/+4.3%, +7.7/+8.1%); PDL/fused-FFN closed | 🟢 |
| 80 | [d5-0-cost-model](80-d5-0-cost-model.md) | D5-0: the speculative-decoding gate — measured costs + acceptance p≈0.68–0.70; conditional go at d=2, gate = nt=3 verify amortization ≥ 2.5× | 📏 |
| 81 | [d5-1a-verify-gate-measured](81-d5-1a-verify-gate-measured.md) | D5-1a: the gate measured end-to-end (`specverify` instrument) — C_T(3)=106 ms, per-token amortization 0.52× vs ≥2.5× required; nt=2–8 batched path costs a flat ~35 ms/token (no regime anywhere), real tile step only at M≥16 → **D5 CLOSED** by the pre-registered stop rule; external check: llama-cli `-md` (same pair) lands at 0.99–1.00× | 🔴 |
| 82 | [small-m-multi-token-mmvq](82-small-m-multi-token-mmvq.md) | small-M dispatch fix: multi-token MMVQ + token-looped legacy kernels — 7B nt=3 105.9→29.4 ms (3.60×), nt=8 5.75×, marginal 34.4→4.3 ms/token; bitwise batched-vs-serial on all 8 quants; pre-registered 2.5× bar missed at 1.87× (cost-model error recorded); D5 verdict unchanged | 🟢 |

## Part V-B · D5-R — speculative decoding reopened and closed (2026-09-12)

The doc 81 §4.3 errata voided the original closure's external anchor, doc 82
made the verify amortization real (2.14× at nt=3), and the plan was rewritten
([`SPECULATIVE-DECODING-PLAN.md`](../SPECULATIVE-DECODING-PLAN.md)) with the
old plan kept as an appendix. Six records:

| # | doc | what | verdict |
|---|---|---|---|
| 83 | [d5-r-stage1-spec-loop](83-d5-r-stage1-spec-loop.md) | the greedy d=2 loop (`--spec-draft`): second GraphCache, lazy accept loop (unit-tested), namespaced weight registries + `nb_bt_only` global-mix fix — the three single-model assumptions a second model breaks; 14B d=2 = 1.34×/1.58× | 🟢 |
| 84 | [d5-r-stage2-dual-engine-battery](84-d5-r-stage2-dual-engine-battery.md) | same-window dual-engine protocol (3 reps × prose/code × 4 cells): minfer 1.33×/1.59× vs llama 1.64×/2.08× — the whole gap = verify row marginal (8.8 vs 2.5 ms/row) | 🟢 |
| 85 | [d5-r-stage3-verify-marginal-ledger](85-d5-r-stage3-verify-marginal-ledger.md) | nsys per-kernel ledger: nt=3 marginal 17.6 ms = attention nt 2–63 hole 9.0 (legacy per-(token,head) kernel vs the 0.8 ms split path) + matmul 8.1 + elt 1.9 + idle 0.7; nt=9 = dispatch cliff onto padded GEMM | 📏 |
| 86 | [d5-r-stage4a-attention-verify-shapes](86-d5-r-stage4a-attention-verify-shapes.md) | one gate: fa_prefill nt≥64 → nt≥2 — C_T(3) 56.9→48.8, C_T(9) 101.3→86.4; e2e 1.42×/1.68× (code ≥ llama's same-window 1.64×); ledger projection validated ~5% | 🟢 |
| 87 | [d5-r-stage4b-multi-mmvq-nt16-closed](87-d5-r-stage4b-multi-mmvq-nt16-closed.md) | multi-MMVQ nt 9–16: groups-of-8 = parity (weights re-streamed per group), acc[16] = register spill (111 ms) → doc-82 GEMM boundary stands; d=8 retired (0.71× prose projected, 0.63× measured in doc 88) | 🔴 |
| 88 | [d5-r-stage5-final-battery](88-d5-r-stage5-final-battery.md) | final battery: minfer d=2 **1.42×/1.68×** (35.7/42.5 tok/s) = 95%/88% of llama's absolute speed; capture prize verified already banked (R3-B); D5-R closes | 🟢 |

## Part VI · Methodology

| # | doc | topic |
|---|---|---|
| 77 | [verification-methodology](77-verification-methodology.md) | the verification system in full: the gate chain, the GB10 tool protocol, the master library of transferable rules |

---

## State at the campaign's close (2026-09-12, after D5-R)

| | tg128 | @long KV | device memory |
|---|---:|---:|---:|
| Qwen2.5-7B Q4_K_M | **1.074×** vs llama.cpp | **1.052×** @1.6K | ~10.4 GB |
| Qwen2.5-14B Q4_K_M | **1.018×** | 0.950× @3.3K | ~14.1 GB |

Prefill: 7B pp3314 ~3581 tok/s (**1.080×**); 14B pp3254 ~1830 tok/s (**1.12×**).
Speculative decoding (D5-R, closed): 14B+0.5B q4_0 d=2 = **1.42×/1.68×**
(prose/code, 35.7/42.5 tok/s) = 95%/88% of llama.cpp's absolute speculative
speed in the same window; d=8 measured 0.63× (retired). Open leads: ncu on
the small-M MMQ gap (counter permissions), nt-invariant accumulation
(prose acceptance + exact greedy identity).
