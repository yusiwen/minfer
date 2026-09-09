# CUDA Inference Path — Optimization History and Current State

> **STATUS (2026-09-06, post-r60): history-organized reference.** This document
> was restructured from a part-based roadmap into a history-ordered record:
> §0 is the master history table (the outline — every landed, reverted, or
> measured lever with its commit and perf delta), §1 is the current state,
> §2 is one chapter per table row, and §3 holds the appendices (env-gate
> reference, methodology, and the pre-Phase-7 legacy history). Single-sourced
> implementation records: `docs/CUDA-BACKEND-PLAN.md` (Phase 7a–7e) and
> `docs/CUDA-FOLLOWUP-PLAN.md` (Phase 8, 8a–8p); the per-round MMQ redesign
> records are mirrored in `docs/LLAMA-CPP-MMQ-ANALYSIS.md` §11.
> Default env = the verified 1.080×-vs-llama path; `MINFER_MMQ=0` = the
> legacy f16 escape (§1, Appendix A).

## §0 Master history table — the complete optimization record

One row per optimization step: every lever that landed, every lever that was
reverted, and the measurement-only rounds that directed the campaign.
Chapters in §2 follow this table row by row.

**Reading conventions.**

- **Perf column**: whole-prefill tok/s for the 7B q4_k_m model on DGX Spark
  GB10 unless noted. The absolute anchor moves with the prompt length used by
  each session (2630/2659 → 3325/3314/3354 tokens ≈ "pp2K/3314-eq") and with
  machine state (co-tenant load); every row's numbers are interleaved
  same-binary A/B medians **within one session window** — cross-session
  absolutes are not comparable (the r59b lesson). "—" = not measured at whole
  prefill (kernel-level or decode-level metric instead).
- **vs-llama column**: whole-prefill vs llama-bench at the 3325-eq (later
  pp3314) anchor — 3324.42 tok/s (clean machine) and 3323.29 (r59b window).
  The 8m–8p/P5 rows' early multipliers are vs the 3401 @2K llama-bench
  figure instead. Before r37 the campaign ran on the opt-in MMQ path, so no
  whole-prefill vs-llama was recorded at the 3325-eq anchor for those rows
  ("—"); the default f16 path sat at 1.43× (2340–2370 vs 3401) from P5
  until the q6_K/FA lines moved it.
- **Commit(s)**: code commit first, record/docs commit second where both
  exist. Twelve hashes quoted in the older session records are pre-amend
  duplicates that no longer resolve (e.g. the r59 record's `feb37de`); this
  table cites their reachable twins (same subject — see the r59 chapter for
  the one case worth naming).
- **Cross-reference**: the r28→r59 MMQ-redesign rounds are mirrored, round
  for round, in `docs/LLAMA-CPP-MMQ-ANALYSIS.md` §11.8–§11.37 (11.8 = r28
  Phase-2 outcome, 11.26 = r47, 11.34 = r55, 11.36 = r58, 11.37 = r59).
- **r26–r27 do not appear in the record** — the P6 round numbering skips from
  r25 to r28 (the intermediate commits are the MMQ-analysis §11 design/doc
  work, kept in `docs/LLAMA-CPP-MMQ-ANALYSIS.md`).

| Step | Lever | Commit(s) | Perf before→after (anchor) | Δ | vs-llama | Status | One-line lesson |
|---|---|---|---|---|---|---|---|
| Phase 7 | CUDA backend: raw FFI device layer + graph backend (7a–7e) | `0dc2a54` | baseline 7B @2K 30.7 tok/s | — | ~110× | LANDED | resident weights + per-op dispatch ended the Part-IV ping-pong |
| 8m/8m② | tiled wmma f16 prefill GEMM + cp.async tile staging | `ba3f317`, `cdc6599` | 30.7 → 294 → 1204 tok/s @2K | 39× | ~2.8× | LANDED | one tensor-core GEMM over all 8 types replaces per-token weight re-streaming |
| 8n | FA-style tiled prefill attention | `cb66fca` | 176 → 8.5 ms/layer @2K | 20× | — | LANDED | online softmax + register O accumulator; 256 B P stride avoids a score-clobber race |
| 8o | decode-start CPU stalls killed | `65b686c` | first decode step 724 → 35 ms | 20× | — | LANDED | `Cow::Owned` clone + eager concat probe cost ~1.6 s per graph rebuild |
| 8p | persistent f16 weight cache + fused dequant-in-GEMM | `2992f57` (+`b9e7a91` docs) | 7B @2K prefill → ~1400–1500 | ~4.7× vs 8m | ~2.3× | LANDED | dequant once per weight at load (≥2 GB gate); exposed a latent Q5_0 misaligned load |
| 8e/8e② | decode MMVQ (dp4a, per-type kernels, shape gate) | `b7b8e73`, `1298cb2`, `1d28235` | 7B decode +37% (q4_K), then q6_K/q5_K | +37% | — | LANDED | integer dp4a dots + llama.cpp `MMVQ_PARAMETERS_GB10` launch table |
| R3-A1 | single-split prefill (tail_ids input at graph head) | `029a9a4` | 4 splits → 1 per prefill forward | — | — | LANDED | a mid-graph declared input forced 2 extra full-stream syncs per forward |
| R3-A2 | pinned D2H logits readback, no redundant clone | `a213c89` | parity-to-slightly-ahead under load | — | — | LANDED | pageable readback paid a driver-internal pinned bounce |
| R3-B | prefill capture defaults ON (3-run protocol) | `761e236` | repeated identical-nt prefills capture automatically | — | — | LANDED | one-shot CLI prefill never reaches 3 runs and pays nothing |
| R1 | int8 MMQ prefill GEMM, opt-in `MINFER_MMQ=1` | `40e97c9` | 155 (co-tenant) / 412 quiet vs f16 630–880 / 1460; 441 in the r7–r8 window | — | — | LANDED (opt-in; superseded by raw line) | parity-clean but ~2.9 TMAC/s vs llama ~24 — the 8× gap was unprofiled |
| R2 | MMVQ weight-streaming rework (per-thread sub-pairs, uint4) | `6df3245` | tg128 42.2 → 45.1; @2K 36.7 → 38.8 | +6.9% / +5.7% | 1.05× / 1.16× | LANDED | per-sub-block nibble re-reads doubled load instructions; L1 hid the bytes, not the issue stream |
| R4 | decode split-attention dim-parallel rewrite | `70f57db` | @2K 39.2 → 43.2–45.1; tg128 45.1 → 47.5 | +10–15% | ~0–4% / ahead | LANDED | LOCAL-memory `float4 oc[32]` accumulator = ~80 MB/layer of local traffic |
| P5·0 | FA prefill P·V on tensor cores | `86ca78c` | 10.06 → 4.24 ms/layer; @2K prefill +15% | +15% | 2.37× | LANDED | wmma for P·V halves the FA pass |
| P5·1 | elementwise vectorization (store_kv, convert) | `d713e6e` | 1435 → 1493 | +4% | 2.28× | LANDED | 1-elem kernels left 15/16 of every transaction unused |
| P5·2 | 128-wide GEMM tiles (TM=128) | `725e307` | 1493 → 2267 | +30% | 1.50× | LANDED | halves B-panel L2 re-reads and barriers per FLOP; `fb[1]` offsets +16 elements, not rows |
| P5·3 | FA softmax on all 8 warps + padded smem rows | `fc07c04` | 2267 → 2365–2371 | +4–5% | 1.44× | LANDED | 256 B rows ≡ 0 mod 32 banks = 8-way ldmatrix conflicts; +8-half stride fixes |
| P5·4 | GEMM k-step KS=64 | `1365c82` | 1464 vs 2345 | −38% | — | REVERTED | 56 KB footprint halves resident blocks — depth vs occupancy inverted |
| P5·neg | TM=256 GEMM tile | `a189837` | −3% | −3% | — | REVERTED | wider tile, same wall |
| P5·neg | in-kernel f32→f16 A staging (AF32 mirror) | `b254c22` (WIP `a3b0dcd`, `69c3933`, `aa40ed3`) | −8% end-to-end; parity hole open | −8% | — | REVERTED | the convert pass is cheaper off the hot path (r23 later quantified it at 6%) |
| r5–r6 | MMQ re-rank + structural rewrite spec | `1e0673f`, `491eb5c` | KD=4 re-negative (427 vs 438); 4-warp 32×32 tile 399 + parity hole; dequant pass is at LOAD, not in the wall | — | — | MEAS-ONLY + REVERTED | depth (KD=8) beats occupancy for the word-staging kernel; spec: raw-byte smem, dequant at mma time |
| r7–r8 | raw-byte MMQ kernel + 128-token wide tile; FA KV L2-prefetch probe | `d440d16`, `d9d626a`, `a41eac0`, `ef9d5b4`, `87bade0` | raw 472 vs 441 (+7%), quantize 129→74 ms; FA probe null (2319–2345 vs 2345) | +7% | — | LANDED (raw) + REVERTED (probe) | wide KD=8 first measured 2124 = phantom (silent smem-cap failure) — guard the cap; GQA already keeps KV L2-hot |
| r9 | llama.cpp MMQ reference decoded; shape axis closed | `84831d2` | narrow cp.async KD=8 481 = local optimum (6-shape matrix) | — | — | MEAS-ONLY | ~0.018 inst/MAC/thread vs our 0.133 — the ratio, not tile shape, is their speed |
| r10 | reference inner-loop decomposition ported | `025a69f` | 462–468 vs 470 (flat, ~6.4 TMAC/s) | ~0 | — | REVERTED (edits lost to a post-checkout hook; measurements valid) | same decomposition as llama still 5× slower — residual is ILP depth/ldsm/tile |
| r11 | ILP-chain reading verified (tile ne = I·J/32) | `83e5580`, `6f29e65` | — | — | — | MEAS-ONLY | the I·J/64 ne was the AMD MFMA branch; NVIDIA needs 4 C regs per m16n8k32 |
| r12 | 16-chain warp tile + ldmatrix | `774a116` | wide-16 KD=4 1020–1058 vs narrow 441–481 (~2.3×); KD=8 973–995 | 2.3× | — | LANDED | accumulator depth wall passed; 4-warp 32×32 and 48B A-padding both negative |
| x-tile | 256-token wide-MMQ block | `f061cb8` | ~942 vs 1035–1058; kernel +27% | −9% | — | REVERTED | A re-reads (327 MB) already ≈2× B re-reads (152 MB) — trading B for A is strictly negative |
| j-tile | 128tok × 256od A-reuse (outer jh loop) | `4993804` | ~1067 (+2.6%, bar 1150 missed) | +2.6% | — | REVERTED | L2-byte savings do not convert to time at 1 block/SM — latency-bound, not byte-bound |
| cp.async-db | cp.async double-buffered raw staging | `784786d` | ~1034 (neutral; baseline band 1035–1050) | ~0 | — | REVERTED | kernel is L2-throughput-bound, not MLP-starved; re-timing identical bytes wins nothing |
| r13 | ncu counter forensics vs llama.cpp + FULL/MINIMAL staging fixes | `5ca037d` | FULL 865–871 (−16%); MINIMAL +0.3% (noise) | — | — | MEAS-ONLY | the 3× gap is the per-MAC warp-instruction stream (10.14 vs 6.06 M/GMAC), not bytes |
| r14 | B-fragments via one ldmatrix.x4 + widened scale loads | `c64cd99` | 1225 @KD=4 / 1273 @KD=8; kernel 3.632 → 2.378 ms | +18.5% / +23–30% | — | LANDED | slot-major 48B qb8 tiling is conflict-free; fewer/wider smem ops, zero staging-ALU growth |
| r15 | f32-accum s8 mma probe (dead) + rank-1 term2 rescale | `b999e9a` | 1295 @KD=8; inst 9.45 → 8.69 M/GMAC | +1.9% | — | LANDED | f32-accumulate integer mma does not exist (ptxas probe); merged-chunk rescale is mathematically invalid |
| r16 | narrow kernel gets the rank-1 fold | `151fa97` | 480 vs 447–473 | +1.7% (noise-band) | — | LANDED | 3-site port of r15; narrow is not the perf path |
| r17 | wide warp remap 32od × 64tok | `9d09a81` | inst −5.8%; wall +0.9/+1.0% (noise) | ~0 | — | REVERTED | pure per-MAC instruction cuts pay ~0 wall while SM% sits at ~30 — stall-bound |
| r18 | load-time B pre-expansion (bulk-copy staging) | `0a26b35` | KD=8 +0.9% (noise); KD=4 −19% median | ~0 | — | REVERTED | +5.8 GB for a −4.5% kernel-time that does not reach the wall; EB/SB machinery preserved for future L2 experiments |
| r19 | weight L2 residency (`__ldg`, persisting window) | `072dd9a` | `__ldg` +2% (noise); L2WIN −50% | ~0 / −50% | — | REVERTED | weight tiles already re-read from L2; a persisting carveout starves C stores/activations/KV |
| r20 | split-phase A staging | `5ac8917` | 1230.4 → 1317.7 @KD=4; 1275.8 → 1319.9 @KD=8 | +7.1% / +3.5% | — | LANDED | the gap carrier is long_scoreboard (97% of named-stall excess) in the LDG→STS chains; freed stalls re-saturate on lg_throttle |
| r21 | coalesced block-linear A staging | `3c009cc` | sectors −28.6%, lg_throttle −90%, wall −2.3%/−0.7% | −2% | — | REVERTED | stall mass is conserved: sectors/queue are not the binder, warp-instruction count is |
| r22 | qa8 XOR swizzle (+ d/ssum fold negative) | `5b40058` | KD=8 1329.6 vs 1311.8 (+1.4%, 3/3); op_ld 16.86 M → 0 | +1.4% | — | LANDED (fold REVERTED) | precompute all 8 A-frag offsets once — per-ldmatrix address ALU eats freed wavefronts; d/ssum loads are L1 hits |
| r23 | f16-path full-graph wall decomposition + FA_TKV=32 occupancy lift | `e8c348d` | lift: occ 16.7 → 32.68%, kernel −6.7%, wall −0.3% | −0.3% | 1.43× (f16 path) | MEAS-ONLY + REVERTED | FA's 2.5×/layer gap is structural (llama keeps 128-wide KV tiles); GEMM is 74% of the f16 wall |
| r24 | scheduling ladder: tile-order swizzle + persistent blocks | `d90b3e9` | swizzle −2.3/−4.7%; persistent −3.3% | negative | — | REVERTED | default B-hot x-fastest order is best; no wave-quantization tail exists to remove |
| r25 | SASS opcode-class census + kd-unroll attempt | `8658f1b` | inst −9.7% (surplus halved to +46.6k/tile); wall +0.37/+0.49% | ~0 | — | MEAS-ONLY (census is the deliverable) + REVERTED | 100% of the +25% inst surplus is support instructions (int ALU 77%) — but it is wall-inert at 1 block/SM |
| r28 | Direction-A raw-nibble NB kernel (2 blocks/SM) | `0957a08` (design `2f783a3`) | 1375.2 → 1410.4 @KD=8; 45,056 B smem, 123 regs | +2.56% | — | LANDED | occupancy was the binding resource; unsigned-nibble + rank-1 rescale is the B-frag contract |
| r29 | NB kd-loop unroll | `bfe6bba` | 1387.9 → 1426.8; int ALU −25%, inst −6.5% | +2.80% | — | LANDED | r25's wall-inert int-ALU cut becomes real at 2 blocks/SM — occupancy unlocks instruction cuts |
| r30 | SWAR word-granular B-nibble unpack | `0071b31` | +0.54% (noise); SASS byte-identical | ~0 | — | REVERTED | r29's unroll already induced the exact CSE — check SASS before writing the lever |
| r31 | q-major sda scale-read repack | `851a896` (+`76d495a` docs) | 1424.10 → 1439.40; LDS.64 32→0, LDS.128 16→32 | +1.07% | — | LANDED (sub-bar) | naive q-major repack was 2-way conflicted — region-split layout is conflict-free |
| r32 | finite-lever sweep (staging hoist, epilogue widening) | `153d28c` | epilogue +0.46% (structurally capped ~0.4%) | ~0 | — | REVERTED | staging addressing already hoisted by ptxas; run-once epilogue cannot clear a bar |
| r33 | hybrid inner-loop port (llama j0/n order) | `697ef04` | median −0.25%; SASS byte-identical | ~0 | — | REVERTED (hypothesis falsified) | ptxas already schedules the 8-mma + rescale loop optimally — source reorders are SASS no-ops |
| r34 | quantize-transpose prepass (A-side layout transform out of the kernel) | `ba977bf` | 1364.2 → 1496.8 @3354 tok; prepass 0.908×; 103 regs | +9.72% | — | LANDED | the residual was layout-transformation locality (llama's `quantize_mmq_q8_1` design), not instruction composition |
| r35 | sda d/ssum scale pre-decode | `6112db3` | −0.46%; SHF 64→0 but wall flat | ~0 | — | REVERTED | decode ALU hides in the IMMA shadow — removing int/fp ops that fill idle slots frees nothing |
| r36 | A-frag wavefront economics (H1/H2 endpoint) | `f44fc44` | 1.76× shared wavefronts/IMMA but 1.85× wavefronts/s at equal IMMA rate | — | — | MEAS-ONLY (H1 refuted) | the MIO pipe is not scarce; llama's edge is A-fragment reuse (0.125 vs 0.5 LDSM/IMMA), a tiling property |
| r37 | post-parity whole-prefill attribution | `ea234f1` | 1521 tok/s; q6_K GEMM 1094.7 ms = 51.2% of wall at 6.38×/GMAC | — | 2.15× | MEAS-ONLY | MMQ made q4_K fast and left q6_K on a slower-than-f16 path — the next lever is a different kernel |
| r38 | q6_K BT-style raw-byte mma kernel (KSPLIT=2, KDR=4) | `75aabb9` | 1518.4 → 1561.9; q6_K 368.9 → 221.8 µs/GMAC (1.66×) | +2.87% | 2.13× | LANDED | q6_K is 16 sub-blocks of 16 (not 8×32) → two m16n8k16 with separate dsc; KDR=8 regressed to 1097.8 |
| r39 | q6_K KDR=2 double-buffer (A+B pipelined) | `f2b9e54` | 1568.7 → 1777.5; attn_v kernel −19.7% | +13.3% | 1.87× | LANDED | doubling every plane at KDR=4 = the 1-block/SM trap; KDR=2 hits the same 29,696 B with real overlap |
| r40 | 3rd resident block via `__launch_bounds__(256,3)` | `65ecef7` | 1784.0 → 2015.6; kernel −23% | +13.0% | 1.65× | LANDED | the 0-spill gate is disproven-immaterial: 80 regs + 4 B spill beats 87 regs at 2 blocks |
| r41 | q6_K B-expand widened to uint4 groups | `b891e1b` (+`aa82e8f` docs) | 1979.9 → 2605.2; kernel 1.70 → 0.654 ms (−61.5%) | +30.7% | 1.27× | LANDED | 32 per-byte ql/qh LDGs per thread-kt = the L1TEX scoreboard (85.5% → 33.6%) |
| r42 | q6_K stage-wide dsc scale read | `a1421e6` | −0.19%; L1TEX traffic down, stall share unchanged (33.6%) | ~0 | 1.27× | REVERTED | cutting dsc bytes does not cut dsc latency — the stall is at the I2F consumer |
| r43 | PC-sampling attribution + pre-expand-B (parity FAIL) | `b7fa305` | attribution: B-expand recomb 45% + A-sts 28% + dsc I2F 26%; W_exp byte-correct but diff 448 | — | 1.27× | MEAS-ONLY + REVERTED | attribute stalls to the consuming instruction; byte-correct data at wrong offsets = stride mismatch next door |
| r44 | W_exp stride mismatch root-caused; dense-index fix | `6d02017` | parity green; kernel −10.9% but wall −0.42% | ~0 | 1.27× | REVERTED | dense W_exp indexed with the padded raw-W stride = the paradox; removing recomb only transforms the latency |
| r45 | cp.async the q6_K A-side staging | `9825ffd` | kernel −10.2%, longsb −18%, wall −0.34% | ~0 | 1.27× | REVERTED | after r41 the q6_K GEMM is no longer the bottleneck — a faster kernel that is not the wall does not reach it |
| r46 (FAP1) | FA audit + FA_TKV 64→32 + S/P row padding | `a186f51` | FA kernel 5.16 → 4.58 ms (−11%); wall +0.27% | ~0 | 1.27× | REVERTED | FA was already wmma + online-softmax — occupancy-starved and S/P-conflicted; but not wall-critical yet |
| r47 | converged-regime wall decomposition (r37 table stale) | `11e3640` | 2585–2623 tok/s; q6_K 1094.7 → 196.4 ms; FA = #1 residual 5.72× (125.8 ms, 10.2%) | — | 1.27× | MEAS-ONLY | q4_K 1.06× and q6_K 1.13× both at parity — recommend FAP2 register-resident softmax (2× → −4.9% wall) |
| r48 (FAP2) | register-resident softmax in `fa_prefill_f16kv` | `d38744d` (+`7e2ee62` docs) | 2603.5 → 2749.9; FA kernel 5.16 → 2.12 ms (2.43×) | +5.6% | 1.21× | LANDED | softmax on the QK^T accumulator fragments; P built in-register as the P·V A-operand; K col_major vs V row_major is the trap |
| r49 | A-quantize prepass shared-A dedup (window memoization) | `87a75a3` | 2734.1 → 2797.5; prepass 193 → 110 launches, 118.4 → 83.9 ms | +2.32% | 1.18× | LANDED | q/k/v and gate/up share one A — the prepass was re-quantizing it per GEMM; cache keyed on (src ptr, nt, id), cleared at any non-MatMul node |
| r50 | FA_TKV 32→16 occupancy trial | `9128468` | −0.5% / −0.01% (3 blocks/SM reached); greedy identity lost | ~0 | 1.18× | REVERTED | occupancy gain cancelled by doubled per-tile sync/softmax overhead — FA_TKV reduction is a dead lever that also breaks byte-identity |
| r51 | producer-fused A-quantize, mode 1 (rms/swiglu emit pad40_t) | `cf1ed4b` (+`bf0c986` docs) | 2803.4 → 2856.4; prepass 110 → 28 launches, 83.0 → 10.1 ms | +1.89% | 1.16× | LANDED | the quantize input is L2-hot in the producer; register-resident swiglu quantize was rejected (uncoalesced f32 stores) |
| r52 | fused-producer phase 2: skip-write mode (`MINFER_MMQ_A_FUSE=2`) | `910d967` (+`fb659f7` docs) | 2855.7 → 3011.3; fused producers 151.9 → 86.5 ms | +5.45% | 1.09× | LANDED | f32 output is provably dead under the window enumeration; dead-write backstop turns violations into loud errors |
| r53 | q6_K bundle: pre-expanded-B dense W_exp + cp.async B staging | `83fee77` (+`4907d9f` docs) | 3024.7 → 3176.9; ffn_down kernel −20.5%; +1.52 GB device | +5.03% | 1.05× | LANDED | r44 (removes WORK) + r45 (removes WAIT) are individually wall-neutral and compose — the basket thesis |
| r54 | `MINFER_MMQ_Q6K_EXP` opt-out of the W_exp plane | `3252e96` (+`b860b7e` docs) | default 3181 (unchanged) / EXP=0 3020.7; 7636 vs 6182 MiB | −5.04% for 1.52 GB | 1.04× | LANDED (gate) | memory-for-speed knob with a three-way liveness label (exp=off vs fallback!) |
| r55 | fused-swiglu roofline audit + one-shot prefill CUDA-Graph decision | `83c3c67` | swiglu at 242 GB/s = 89% roofline (cap +0.74%); capture ≤ +0.1% + capture-illegal malloc | — | 1.05× | MEAS-ONLY (both documented skips; campaign CONVERGED) | bound the roofline before coding — no implementation of this kernel can clear the bar |
| r56 | q6_K A-side bundle: A cp.async + W_dsc f32 plane | `4cf7c74` (+`29084de` docs) | 3138.6 → 3212.5; ffn_down −5.9%, attn_v −4.2% kernel; +363 MB | +2.35% | 1.035× | LANDED | post-r53 the A-side wait and dsc consumer became the wall — r45's mechanism finally lands in a bundle |
| r57 | FA KV staging double-buffer (FA_TQ=48) | `c3268cc` | greedy-32 identity lost at token 19 (×2 attempts) | — | 1.035× | REVERTED | FA_TQ is a tile size too — the r50 rounding caveat applies to any FA tile change |
| r58 | q4_K BT structural spec + cp.async-db2 transplant | `093ae41` (code reverted) | 2819.3 vs 3227.6 (−12.6%); spec: gate/up −37% potential, ceil-waves 2.6% | −12.6% | 1.03× | REVERTED | a mechanism whose COST depends on the granularity of what it replaces is amortization-bound (the r45 mirror) |
| r59 | q4_K W_dsc f32-pair plane + pre-warm/pre-grow riders | `36a481f` (+`15c04ba` docs) | recorded +26.2% (2843.2 → 3588.8, co-tenant) — superseded by r59b; bt kernel busy −30.9%; +1456 MB | +11.1% (true) | ~ahead | LANDED (Δ corrected) | gate/up (−37%) and q/o (−18%) carried it, not ffn_down — the r58 "staging scales with kt" reading confounded decode cost with A-plane DRAM traffic |
| r59b | clean-machine re-measure + baseline-poisoning correction | `074ca94` | definitive 3590.8 (HEAD) vs 3232.0 (fresh baseline rebuild) = 1.080× llama-bench 3323.29 @pp3314 | +11.1% | 1.080× (ahead) | MEAS-ONLY (correction) | the r59 "baseline" binary was the stale r58-delta build (−12.5%) — anchor every A/B baseline behaviorally in the same window |
| r60 | PROMOTION: the verified MMQ gate set flips DEFAULT-ON | `57edcf6` (+`7029ee4` docs) | default ≈3578–3599 (~3581); `MINFER_MMQ=0` legacy f16 ~2226–2353-class; planes +3.27 GB | — | 1.080× | LANDED | promotion = default-on with "0" opt-outs (r54 pattern); the bisect caught the mode-2 multiturn break → NB-BT-only guard |
| D1 | decode @1641 KV attribution: `gqa_attn_split_partial` is 100% of the KV-scaling wall (34.1 µs/launch, 76.5% long_scoreboard); ATTN_SPLITS sweep = dead end | measurement-only (`/tmp/d1/`) | tg128 49.3 (KV~1) / 47.2 (@1641) vs llama 49.41 (tg128) | — | 0.956× (tg128) | MEASURED | probe-verified: staging-depth changes bitwise-safe (ndiff=0); ATTN_SPLITS changes reorder the float sum |
| D2 | explicit K+V register staging in `gqa_attn_split_partial` (4-row window staged before the softmax chain); cp.async smem pipe + pair-lookahead measured worse | D2 commit (this row) | decode @1641 **47.2 → 48.2** (+2.0%); kernel 34.1 → 19.4 µs/launch; tg128 49.4 flat | — | **0.975×** (tg@1641) | LANDED | bitwise-identical end-to-end (greedy-32/256 byte-identical); probe −42%, nsys −43%, wall +2.0% agree |
| D3b-1b | down-q6K pipelined MMVQ `q6_k_q8_mmvq_v2_pf` (npair>256: both serial units' weight+q8 loads issue up front) | `f1825b5` | 7B decode tg128 48.03→**49.47** (+3.0% SEP), @1641 46.66→**47.94** (+2.7% SEP); 14B tg128 22.80→22.90 (+0.44%), @3254 21.01→21.06 (+0.24% SEP) | **+2.7–3.0%** (7B decode) | 0.970× (7B tg@1641, this window) | **LANDED** | bitwise-identical (114/114 dump memcmp, greedy-256 byte-identical, suite 169/0/3); for npair>256 the second serial unit's exposed load latency WAS the 198.9-vs-220 GB/s gap — **D4-2 CORRECTION: the 7B numbers are void (the dispatch dropped units 512..591 on npair-592 rows; the "gain" was mostly the missing work — see the D4-2 chapter); 14B numbers stand** |
| D3b-1a | attn_v-q6K off the padded-f32 kernel: (a) MMVQ routing via a lowered `od*id>=24M` gate — NOT bitwise (MMVQ quantizes activations to q8, different accumulation semantics); (b) NSG 2→1 row→warp re-map — bitwise-green but kernel 36.4→39.9 µs (2× warps = 2× y re-read L2 traffic) | reverted (both routes) | 14B tg128 −0.74%, @3254 −0.33% | — | — | REVERTED | the padded kernel is not warp-starved; y re-read traffic scales 1:1 with warp count — rows-per-warp is the only bitwise-free knob and 2 is already the sweet spot |
| D3b-1c | output-head dynamic block size (npair 160 → 160-thread blocks, warp-count-bounded `mmvq_block_reduce`) | reverted (patch `/tmp/d3/patch_1c.py`) | 7B @1641 +0.26% (SEP); 14B tg128 +0.04%, @3254 +0.09% | — | — | REVERTED | GB10's 1536-thread/SM limit: 9 blocks×160 live threads ≈ 6×256 allocated (960 live) — the idle-thread win does not exist at 14B shapes |
| D3b-2 | short-KV combine skip (single-split path for nkv ≤ threshold) | not implemented | — | — | — | ANALYSIS-NEGATIVE | single-split ≠ 32-split partial+combine bitwise for ANY nkv>1 (the merge reorders the float sum — D1's split-count evidence: ndiff 3.6e-3 of outputs, max\|Δ\|~3e-9); the split grid is frozen by CUDA-graph replay capture; the bitwise-safe residual (combine early-out of empty splits — exact +0.0 terms) is ≤ ~15 µs/step, below every bar |
| D3a | 4-warp fattn-vec-style split-attention rewrite (`gqa_attn_split_partial_h4w`, hd=128: 128 threads, K/V streamed, Q in registers, 8-lane subgroups, 32-row windows/warp, smem LSE merge; grid unchanged, replay-safe) | reverted (patch `/tmp/d3/d3a_kernel_patch.diff`, findings `/tmp/d3/D3A_FINDINGS.md`) | kernel 14B @3254 73.4 → 68.4 µs (−6.9%) but 7B @1641 21.1 → 34.7 µs (+64%); wall 14B @3254 −0.95% (noise), 7B @1641 −3.4% (real), tg128 noise-level | — | — | REVERTED | rows-per-warp pathology: rpw = ceil(ceil(nkv/32)/4) = 26/13/1 at @3254/@1641/tg128 — the 32-row window idles 59–75% of lanes below rpw≈16 and per-block fixed costs amortize over rpw; kernel −6.9% at the best shape is only ~+0.5% wall (attention = 7.4% of the step), under the +1.5% bar and the ±2% A/B noise; numerics fully green (probe ≤1.3e-7 vs CPU, argmax hard-gated, greedy 0/10 diverged) — the session's durable output is the tolerance-gate calibration: end-to-end max|Δlogits| is 0.38/0.39 (14B/7B) for ANY accumulation-order change, so the D3-1 ≤1e-3 logits gate is unsatisfiable; argmax+greedy+A/B are the operative gate set |
| D3-4 L1 | hybrid rpw dispatch: dual-kernel self-gating split attention (f16 KV, hd==128) — 4-warp h4w kernel when rpw = ceil(ceil(nkv/32)/4) ≥ 16 (nkv ≥ 1921), incumbent 32-thread kernel below; BOTH launch per layer with static grids, each re-reads `positions[0]` per replay, exactly one is live per nkv (nkv-uniform branch → replay-safe) | `22336b2` | 14B @3254 split 72.1 → 62.1 µs (−13.9%) + 1.5 µs dud launch; wall 14B @3254 21.20 → **21.33** (+0.61%, SEP), tg128 22.96 → 22.94, 7B tg128 50.28 → 50.20, @1641 48.78 → 48.68 (guards hold) | **+0.61%** (14B @3254) | 0.944× / 0.877× (14B tg128/@3254 vs llama 24.31/24.32) | **LANDED** | 1-warp path bitwise (7B @1845 dump: all gated files identical; `node{3,5,8}_prefill` diffs = pre-calibrated slot aliasing); h4w tolerance class (max\|Δlogits\| 0.309, argmax identical margin 0.716, 1 greedy flip at the regime entry = 1/256 < 2%, temp-0.8 controls identical); suite 169/0/3 incl. the hd=128/n_ctx-4200 parity shape sweeping the rpw 15/16 boundary; an in-kernel 1-warp-fallback form was REJECTED pre-commit: inside 128-thread blocks the 1-warp body caps at 12 working warps/SM (1536/128) = **+78% kernel at 7B @1641** (35.4 vs 19.8 µs nsys) — geometry, not math |
| D3-4 L2 | window-level K/V prefetch pipelining in the h4w body (K-pass software pipeline +2 uint4, 4-deep V-bulk ring +8 uint4; issue-point-only → bitwise vs h4w by construction) | reverted (patch `/tmp/d3/patch_l2.py`) | 14B @3254 h4w 62.1 → 66.5 µs (**+7%**) | −7% kernel | — | REVERTED | the kernel is bytes+tail-bound at 79% of the 48.9 µs floor, not chain-bound enough: funding the pipeline buffers needs `__launch_bounds__` minBlocks 8→4 (64→128 regs) → occupancy 32→16 warps/SM and 3.33→4.44 waves — the occupancy/wave-tail tax outweighs the shorter load chains; the D2 4-row-scale lesson (issue-point moves are free) does NOT transplant to window scale under a 64-reg budget |
| D3-4 findings | pre-existing behaviors calibrated this session: (a) at long prompts (≥2.8K tokens) `MINFER_GRAPH_DUMP` PREFILL-phase files (all `kv*_prefill`, `logits_prefill`, prefill nodes) are non-deterministic pre-vs-pre (wholesale, garbage-magnitude — aliased dump reads); decode-phase dumps stay deterministic; (b) CLI prompts longer than the n_ctx default 4096 leave zero generation headroom (`position N exceeds n_ctx N` panic, graph.rs:367); bench unaffected | measurement-only | — | — | — | RECORDED | dump gates at long prompts must anchor pre-vs-pre at the EXACT shape and gate only decode-phase files; long-prompt CLI greedy needs `prompt + n ≤ 4096` until n_ctx sizing is fixed |
| D3-5 1a | fused-producer decode A-quantize: `rms_norm_quant_pad40` (rms body + the standalone per-block quantize body, both verbatim) and `swiglu_quant_pad40` write the pad40 q8 plane beside their f32 output; decode matmuls consult `decode_quantize_native` (MmqCache, native form) and skip the standalone quantize launch on a hit; FusedFFN joins the cache-clear preserve set; `MINFER_NO_DECODE_A_FUSE=1` opt-out | `3230b2b` | nsys 14B @3254 (NO_CUDA_GRAPH): standalone `quantize_q8_0_pad40` 4448 → **964** launches (−78%; the ~50/step remainder = the attn_o class), total kernels 29402 → 25918, sub-2µs 15171 → 11723; wall 14B tg128 23.05 → **23.35** (+1.30% SEP), @3254 21.36 → **21.68** (+1.50% SEP), 7B tg128 49.86 → **50.69** (+1.66% SEP), @1641 48.47 → **49.16** (+1.43% SEP) | **+1.30%** (14B tg128) | 0.961× / 0.891× (14B tg128/@3254 vs llama 24.31/24.32); 7B **1.026×** / 0.995× | **LANDED** | q8 bytes bit-identical by construction (max is exact for any association; rintf/clamp elementwise — the epilogue IS the standalone body) and probe-verified bitwise (rms+swiglu q8 buffers, f32 producer outputs, MMVQ outputs through the cache-hit path); suite 170/0/3; 14B −n 4 dump gate: logits both phases + all KV byte-identical, the 3 node-dump diffs are same-binary pool-slot aliasing reproduced pre-vs-pre AND post-vs-post; greedy −n 256 token streams byte-identical both models; fused epilogues add ~0.2–0.25 µs/launch (swiglu 2.06 → 2.31 µs), priced into the wall |
| D3-5 1b | output-head od-split / 512-thread re-map (lm_head q6_K od 152064, id 5120, npair 160) | not implemented | — | — | — | ANALYSIS-NEGATIVE | all three candidate mechanisms are measured or computed dead at this shape: (a) idle-thread removal (96 of 256 idle at npair 160) = D3b-1c, measured neutral; (b) rows-in-flight: 288 resident rows either way (6×256-thread blocks/SM vs 3×512 dual-row), and D3b-1c's 9×160 = 432-row form was ALSO neutral — occupancy is not the limiter; (c) block-scheduling rate: the head sustains 47.6 blocks/µs while ffn_gu demonstrates 76/µs — not the limiter. A dual-row 512-thread form is bitwise-capable (per-row half-block reduce with the same 8-warp tree) but carries no mechanism → not built per the "measured mechanism, don't guess" rule |
| D3-5 1c | ffn_down-q6K 512-thread single-unit variant (id 13824 → npair 432) | not implemented | — | — | — | ANALYSIS-NEGATIVE |
| D3-6 2a | GQA q-head batching in the decode split attention (grid (ATTN_SPLITS, n_kv_heads), 32\*gqa threads, warp w = q head hk\*gqa+w, full split stripe per warp, shared `h4w_warp_windows` window pass, per-warp 8/16-butterfly epilogue; same rpw ≥ 16 dispatch slot, static grids → replay-safe) | reverted (patch `/tmp/d3/patch_2a.py`) | nsys 14B @3254: h4w 63.76 → batched 64.79 µs (+1.6%); ncu `lts__t_sectors`: 2,121,671 (4.63× analytic 1×) → 472,583 (**1.03×**) — traffic ÷4.5 with time flat | −1.6% kernel | — | **REVERTED** | mechanism-nailed: the 5× L2 re-read is fully HIDDEN under the latency roofline in the live regime (D1's latency-bound attribution stands; D3-4's L2-composition re-attribution revised) — ncu's −28% appears only serialized/cold; ALL gates were green first: parity ≤1e-4 at gqa 5/7 incl. exact nkv 2808 + outlier calibration (shapes kept as permanent h4w coverage), dump argmax HARD gate (margins 2.187/0.557), greedy byte-identical with repeat-penalty 1.0 both models, default-penalty flips = 1/256 sampler knife-edges (14B step-63 raw top-2 probgap 0.0167; 7B step-8 penalized rank-6 winner), temp-0.8 controls identical, suite 170/0/3; new gate rule: attribute greedy flips to sampler vs kernel via the penalty-free stream + per-step logits_top trace |
| D3-7 2b | attn_v-q6K decode MMVQ routing: Q6_K decode dispatch gate lowered `od*id >= 24M` → `>= 4M` (the only affected shape in the supported set is the 14B attn_v, od 1024 × id 5120 = 5.24M, 11 layers; GGUF census; 7B attn_v od 512 × id 3584 = 1.8M stays padded-f32) | this commit | nsys 14B @3254: attn_v kernel 33.16 → 24.32 µs (−26.7%, ~177 GB/s — short of the 220 class, as the 8e small-shape crossover data warned for od≈1024 but still −26.7%); ×11 layers ≈ 97 µs/step; quantize launch count UNCHANGED (attn_v joins attn_o's MmqCache hit — same src buffer + id) | **+0.42%** (14B @3254, 8-pair median) | see D3-7 | **LANDED** | Tolerance-gated per the D3a package: logits_decode max\|Δ\| 0.254 (calibrated 0.39-class), logits_prefill byte-identical, argmax HARD gate green (margin 1.915), kv1+ decode-side f16-boundary drift (kv0 bitwise — earlier onset than D3-6's sub-ULP class, expected for input-quantization noise); penalty-free (rp=1.0) greedy streams byte-identical both models (the D3-6 clean kernel gate); default-penalty flips = 1 knife-edge event/256 steps (5/5 seeds, coherent text, no degeneracy); temp-0.8 controls: 7B identical, 14B reorders (sampled reordering expected at 0.22-logit drift); wall: @3254 21.605 → 21.72 (+0.42%, 7/7 clean pairs positive, sign-test p≈0.008; strict SEP missed by 0.05% — min-new-excl-outlier 21.66 vs max-base 21.67 — medians carried per the D3-5 outlier precedent); tg128 clean-window +0.26% (sub-bar; the extension window was co-tenant-contaminated post-side) |
| D3-7 2c | rms/elementwise-launch consolidation, two bitwise sub-levers: (i) `rms_norm_quant_pad40` wide-block geometry (launch 32 → 128 threads; the reduction keeps lanes 0..31 exactly — same element→lane map, serial per-lane chains, `warp_reduce_sum` tree; scale broadcasts via smem; write/quantize loops are element/per-32-block independent so their wider mapping cannot move a bit; reduce loop `#pragma unroll 8` deepens load pipelining), (ii) `positions_i32` one-execution-window memo (every Rope/KvcacheStore/Attn node re-converted the same positions buffer: 240 launches/step at 14B; key (buf id, pool_gen), cleared in `synchronize` next to the MmqCache clear; capture-safe: only the first consumer's conversion is recorded and replay re-executes it) | this commit | nsys 14B @3254 decode census: rms_norm_quant_pad40 9.43 → **5.66 µs** (−40%), 94.6–96/step; f32_bits_to_i32 239.6 → **1.2** launches/step; wall-effective ≈ −0.62 ms/step (rms −0.348 + bits −0.275) | **+1.76%** (14B @3254 cumulative with 2b, SEP) | see D3-7 | **LANDED** | Bitwise end-to-end: dump gate (both sides under `MINFER_NO_KQ_MMVQ=1`) 109/114 files byte-identical, the 5 diffs = the documented slot-aliasing class; logits both phases + all KV byte-identical; 7B greedy streams byte-identical 5/5 seeds + temp-0.8 control; suite green. GATE GOTCHA recorded: `MINFER_NO_KQ_MMVQ=1` also reverts the Q5_K decode arm (pre-existing), so a 2b-off control must set it on BOTH sides — a one-sided control shows a fake 0.22-logit drift from the Q5_K f32-activation fallback |
| D3-8 | FusedQKV decode fusion ported to CUDA (Stage-3 Tier A; the G4 Metal fusion): (1) `attn_bias_rope_store_f32` kernel + `launch_attn_bias_rope_store` — one launch replaces the per-layer add_bias×3 + rope×2 + store_kv×2 chain, pointer-form (serves concat sections q=base/k=base+nqt/v=base+2nkt AND three separate buffers), positions read device-side (`positions[0]`) so the launch is capture-safe, math verbatim `add_bias_f32`+`rope_f32`+`store_kv_f32/f16`; (2) class 1 (wq\|wk\|wv same quant type): `Op::FusedQKV` — one concat matmul (`blk.{i}.attn_qkv` loader-registered wq\|wk\|wv rows) + the fused epilogue (MMVQ is per-row, dispatch on (ttype,id,nt) only → concat bitwise-equal to 3 separate matmuls, probe-proven); (3) class 2 (mixed quant, e.g. Q6_K attn_v among Q4_K q/k — 24/48 layers at 14B, 14/28 at 7B): new `Op::QkvBiasRopeStore` — the three SEPARATE matmuls (bias-free) + one epilogue launch (CUDA-only; Metal keeps the unfused chain for these layers), builder wires attention to the epilogue node so q's matmul buffer has exactly one consumer and the §5 in-place alias applies; gated by `nt==1 && gpu && fuse_qkv` (part of the reuse identity — `MINFER_NO_FUSE_QKV=1` reverts both classes for A/B) | this commit | nsys 14B @3254: total launches −4968/trace (−22.5%); per decode step **−310** (add_bias −144, rope −96, store_kv −96, fused +48, mmvq −48 (24 concat layers 3→1), quantize +24) ≈ the D3-7 §2-listed 0.45 ms/step qkv-chain item | **+1.63%** (14B @3254; tg128 **+3.11%**; 7B +1.23%/+1.05%; isolation A/B post-vs-post NO_FUSE_QKV: +2.28%/+3.15%/+1.00%/+1.03% — all 3/3 pairs clean-separated) | see D3-8 | **LANDED** | Bitwise: probe tests (epilogue vs the 7-launch chain bitwise on q/k/v sections + KV rows, f32+f16 KV, both pointer forms, 14B+7B shapes; concat matmul vs 3 separate bitwise) + dump gate logits_prefill/decode + ALL kv\*.f32 byte-identical both models (98/98 + 58/58; the informational node\* dumps are a documented instrument limitation — recycled pool slots, binary-layout-dependent) + greedy −n 256 **byte-identical 5/5 seeds × both models** + rp=1.0 + `MINFER_NO_FUSE_QKV=1` control + temp-0.8 controls (the only diffs are the perf-banner tok/s numbers); prefill DOT graph byte-identical (prefill untouched); suite 172/0/3 (D3-7's 170 + 2 probes); prefill graph topology unchanged (nt>1 gate) so prefill perf untouched (pp3254 1833 t/s pre-vs-post) |
 NOT bitwise vs the landed v2_pf: in the 256-thread form thread t accumulates fma(u_t) then += fma(u_{t+256}) into ONE float acc before the block reduce; at 512 threads those units live in different threads and their sum happens in the reduce tree (16-warp cross-warp serial order) — a different float sum. A bitwise emulation (smem pair-exchange so thread t still sums u_t+u_{t+256} first) adds a barrier for zero resident-parallelism gain (5120 rows = 18 waves either way), and the exposure mechanism 1c targets was already fixed by v2_pf's up-front load issue (D3b-1b) |

| D4-2 | Decode-GEMM Tier B session (design `/tmp/d4/D4_DESIGN.md`): (B0) **correctness** — v2_pf dispatch bounded to npair ≤ 512 (see the D4-2 chapter; D3b-1b's 7B gains were mostly dropped units); (A) llama L2-prefetch port closed PRE-BUILD: prefetch distance 2·bpi requires bpr > 32 blocks (QI4_K=16/VDR=2, QI6_K=8/VDR=1 → bpi = 4·nwarps) — at 14B only ffn_down (bpr 54) qualifies, and our kernels map one 64-elem unit per thread over 256 threads → exactly ONE K-loop iteration at every decode shape (npair 80/216/160; v2_pf's 432 are unrolled u0/u1): there is no "2 iterations ahead" to prefetch, and where llama's prefetch does fire its ffn_down-q4K runs 224.1 GB/s vs our 228.6; (B1) `__launch_bounds__(256,6)` on v2_pf: 48→40 regs + 40 B stack spill, probe tg128 +0.25% / @3254 **−0.75%** → killed; (B1c) v2-loop at npair 432 via `MINFER_Q6K_PF=0`: v2_pf wins/ties (the 5-block × 2-unit-MLP form beats 6-block × 1-unit) → default kept, env kept as opt-out; (B2) 160-thread v2 right-size (= D3b-1c repeat, re-measured with per-kernel isolation): bitwise 98/98 but nsys lm_head **+1.51%**, attn_v **+3.04%** → killed | `b31084c` (B0) + docs commit | per-kernel nsys deltas above; walls ≈ 0 as expected for a fix-only tree | correctness fix; perf-neutral | see D4-2 | Dump gates 107/107 (14B pre-vs-post) + 98/98 (B2 bitwise check); greedy byte-identical 14B pre-vs-post; 7B fixed-vs-v1 first-step logits at the v1-vs-v2 rounding class (max\|Δ\| 0.254, argmax same) vs 4.72-4.79 pre-bug; suite 173/0/3 |

| D4-3 | Attention-structure attempt 2 (probe `/tmp/d4/probe_attn2.cu`, 690-row sweep, 0 skips): llama-fattn-geometry split-attention kernel `vec_attn` over pb/T/R/minb/STG (load-scheduling axis); **NO-GO per the pre-registered bar** — best 41.07 µs kernel-total @14B (bar ≤~32; 1.64× vs current 67.4) and 16.22 @7B@1641 (bar ≤~10.35; 1.53×) → projected wall +1.6–1.8% < the +2% bar → no integration. **Headline: the D4-1 llama attention target (14.02 µs/layer @14B/@3254) is a llama-bench artifact** — ncu: the bench decode fattn-vec (grid (1,2,40)) loads a constant 5,427,200 B ≈ one 128-row KV iteration per block = 256 of 3255 rows covered, byte-identical at KV 1024/2474/3255, while llama-cli's decode (grid (1,7,40)) loads 53.2/142.7 MB scaling with context (mid-prompt recall A/B confirms). Honest llama full-context decode attention ≈ 2.0–2.1 TB/s ≈ 1.7 ms/step @14B — minfer's 3.32 ms is ~1.9× off, not 4.9×; the honest @3254 gap is ~10% wall (~3.5% attention) | docs commit | sweep table in the D4-3 chapter | line closed (measurement-corrected) | see D4-3 | probe gate 0.05 abs w/ adversarial outliers, CPU ref in double; SASS-level LDG counts + recall A/B + reductio (9.6 TB/s impossible) all consistent |

**Footnotes.**

1. **r59 correction (visible in-table).** The r59 record originally reported
   **+26.2%** interleaved under a co-tenant and attributed −12% to a
   "co-tenant tax". r59b proved the r59 baseline binary was itself the stale
   r58-delta build (~−12.5% deficient), making the true clean delta **+11.1%**
   (3232.0 → 3590.8). The +26.2% and the co-tenant-tax claim are superseded;
   the corrected value is what the table's Δ column carries.
2. **Measurement contexts.** Rows r12–r25 interleaved on a box that drifted
   session to session (−9% to +38% vs neighbors) — only intra-session deltas
   are meaningful. r59's interleaved series ran with a 46 GB sglang co-tenant
   (later shown irrelevant: idle residency taxes nothing, r59b §1). r55's
   baseline sanity was 3144.4–3151.4 under a live co-tenant vs the quiet-box
   3181.
3. **Hash substitutions.** The older records quote HEAD/revert anchors that
   were later amended away (e.g. r22's "HEAD 9819410", r18's "HEAD 1e0dded",
   r55's "HEAD dd6d842", r58's "HEAD 2105b08", r59's "commit feb37de"). Each
   has a reachable twin with an identical subject; the table cites the twins.
   The P5 record's session range start `b8568cd` does not resolve to any
   commit — the P5 code commits are `86ca78c`, `d713e6e`, `725e307`,
   `fc07c04`, `1365c82`. llama.cpp-side hashes (`ca3d5a3e1` bench build) are
   upstream identifiers, not minfer commits.
4. **Campaign arc**: R1 MMQ 441 tok/s (first parity-clean MMQ measurement,
   r7–r8 window) → 3590.8 tok/s (r59b definitive) = **8.1×**.

## §1 Current state (post-D3-8, 2026-09-08)

### 1.1 Performance summary (DGX Spark GB10, 7B q4_k_m unless noted)

llama.cpp reference: llama-bench @ `ca3d5a3e1` (upstream build), 8 threads,
`-ngl 99`.

| Config (all default env unless noted) | 7B q4_k_m whole-prefill | Memory (peak, per-PID) | Note |
|---|---|---|---|
| **Default (promoted MMQ gate set, mode 2)** | **~3581** (r59b definitive median; r60 A/B window 3578.0–3598.7) | **9484 MiB** | = the verified 1.080× path |
| `MINFER_MMQ_Q6K_EXP=0 MINFER_MMQ_Q4K_DSC=0` (planes off) | −5%-class (r54: −5.04%; r59-class) | 6217 MiB (planes cost **+3.27 GB**) | fine-grained opt-out |
| `MINFER_MMQ=0` (legacy f16 w16-cache escape) | ~2353 clean-class (~2226 in the r60 window) | **~20.5 GB** | the escape is ~11 GB HEAVIER, not lighter |
| llama-bench pp3314 (r59b window) | 3323.29 ± 3.08 | — | minfer 3590.8 / 3323.29 = **1.080×** |

Decode (nt==1) is untouched by the MMQ campaign (r60 evidence: no
`MINFER_MMQ*` read on the nt==1 path; decode `-n 16 --greedy` byte-identical;
tg128 45.2 = 45.2 in the r60 window). Recorded decode state (R4-era session,
llama.cpp 47.1 / 44.9): **tg128 47.5–47.6** (at/above parity), **@2K
43.2–45.1** (~0–4% gap). Small models (pre-MMQ-campaign numbers, Part-I
record): 0.6B q8_0 prefill @2K 4792 (llama 23909), decode tg128 ~195 (290);
0.5B q4_0 prefill ~3020 (30550), decode ~257 (453).

**D2 update (2026-09-07):** decode @1641 KV 47.2 → **48.2** (+2.0%, tg128
unchanged at 49.4) via explicit K+V staging in `gqa_attn_split_partial` —
see §2D.

**D3b update (2026-09-07):** decode down-q6K MMVQ pipelined for npair>256
(`q6_k_q8_mmvq_v2_pf`, bitwise-identical): 7B tg128 → **49.47** / @1641 →
**47.94** (same-window interleaved A/B vs 48.03/46.66 pre; the @1641 gap to
llama 49.41 is −3.0%); 14B (48L) decode tg128 → **22.90** / @3254 → **21.06**
window anchors vs llama 24.31/24.32. The remaining 14B short-KV gap is
elementwise/launch chain + the two reverted MMVQ straggler routes (§2D D3b);
attention rewrite = D3a (tolerance-gated, separate session).
**D4-2 CORRECTION (2026-09-09):** the 7B half of this row is void — the
v2_pf dispatch dropped units 512..591 on npair-592 rows (7B ffn_down), so
the 7B "+3.0 %/+2.7 %" was mostly the missing 13.5 % of down-q6K work, not
pipelining. The pipelining gain itself is the 14B-sized +0.2-0.4 % class
(14B numbers in this row stand; npair 432 ≤ 512 was always correct). 7B
decode re-anchors lower on the fixed engine — see the D4-2 chapter.

**D4-2 update (2026-09-09):** a silent 7B decode correctness bug (D3b-1b's
v2_pf dispatch, units dropped on npair > 512 rows) found and fixed
(`b31084c`); the llama L2-prefetch port closed pre-build (mechanism inert at
our decode shapes); the bitwise q6_K occupancy axes (reg-shave, pipeline
depth, block geometry) all measured closed — see the D4-2 chapter. 7B
guards are re-anchored on the fixed engine: **tg128 ≥ 48.3 / @1641 ≥ 47.5**
(the old 49.0/47.9 were set on the dropping kernel and are superseded —
correct-over-fast). Same-window vs-llama on the fixed engine: 7B tg128
1.035× / @1641 1.017× (llama 47.65/47.69); 14B tg128 parity 0.995×, @3254
0.928×, pp3254 1.11-1.14×.

**D4-3 CORRECTION (2026-09-09): the 14B @3254 "0.928×" gap is mostly a
llama-bench artifact.** ncu proves the bench decode fattn-vec covers only
256 of 3255 KV rows (constant 5.43 MB of loads at any context) while
llama-cli's decode covers everything (53–143 MB, scaling with ctx;
mid-prompt recall A/B green). Honest-llama @3254 ≈ 23.3–23.5 tok/s, so
minfer 21.06 is ~10% behind, of which attention is ~3.5% (the D4-1
"~2.53 ms attention prize" recalibrates to ~1.6 ms). The attention-structure
rewrite attempt itself (vec_attn sweep) measured NO-GO per the
pre-registered bar (best 41.07 µs @14B / 16.22 @7B, bars ≤32/≤10.35) — see
the D4-3 chapter. Do not quote llama-bench long-ctx tg rates as attention
targets without an ncu byte-count or llama-cli recall cross-check.

**D3a update (2026-09-07):** the 4-warp fattn-vec-style split-attention
rewrite was REVERTED — numerics fully green but the rows-per-warp
pathology makes it +64% kernel-slower at 7B @1641 and only −6.9%
(~+0.5% wall, sub-bar/sub-noise) at 14B @3254 (§2D). Session outputs
that stand: the tolerance-gate calibration (end-to-end logits drift
0.38/0.39 is the inherent class of ANY accumulation-order change — the
≤1e-3 logits gate is unsatisfiable; argmax + greedy-divergence + A/B are
the operative gates) and the 14B @3254 attention bytes-floor gap
(73.4 µs/layer vs 48.9 µs floor) with window-prefetch pipelining as the
next lever.

**D3-4 update (2026-09-07):** L1 hybrid rpw dispatch **LANDED** (`22336b2`):
the f16-KV hd==128 decode split attention now runs BOTH the D3a 4-warp
fattn-vec-style kernel (rpw ≥ 16, nkv ≥ 1921 — measured 72.1 → 62.1 µs at
14B @3254, −13.9%) and the incumbent 32-thread kernel (rpw < 16, bitwise),
each self-gating per replay from `positions[0]` — 14B @3254 wall +0.61% SEP,
7B guards hold (48.68 @1641 / 50.20 tg128 this window). L2 (window-level K/V
prefetch) measured +7% kernel (occupancy/wave-tail tax) and was REVERTED.
Window (interleaved 3× medians): 14B tg128 22.94 / @3254 21.33 vs llama
24.31/24.32 (0.944×/0.877×); 7B tg128 50.20 / @1641 48.68 vs 49.41
(1.016×/0.985×). Distance-to-parity and next levers: §2D D3-4.

**D3-5 update (2026-09-08):** fused-producer decode A-quantize
**LANDED** (`3230b2b`): the decode rms_norm/swiglu producers write the pad40
q8 plane (bit-identical bytes) and the following decode matmul group skips its
standalone quantize launch — standalone quantize launches −78% (nsys), sub-2µs
launches −3448/step-census, and the win shows at all lengths: 14B tg128 23.05
→ 23.35 (+1.30% SEP) / @3254 21.36 → 21.68 (+1.50% SEP); 7B tg128 50.69
(**1.026× vs llama — ahead**) / @1641 49.16 (0.995×, parity). Window vs the
pre-D3-5 anchors: 14B tg128 +1.65% (session bar ≥ +1.5%), @3254 +2.95%. 14B
decode geometry levers (head od-split, ffn_down 512-thread) resolved
analysis-negative with the math recorded (§2D D3-5); Stage 2 = GQA q-head
batching (attention re-read) + rms-launch consolidation.

**D3-7 update (2026-09-08):** the two remaining Stage-2 levers both LANDED.
2b attn_v-q6K → MMVQ routing (Q6_K decode gate 24M → 4M; 14B attn_v 33.16 →
24.32 µs × 11 layers) and 2c rms/elementwise consolidation (wide-block rms
9.43 → 5.66 µs bitwise; positions i32 conversion 240 → 1 launch/step
bitwise). Cumulative wall (interleaved SEP): 14B tg128 23.31 → **23.73**
(+1.80%), @3254 21.57 → **21.95** (+1.76%); 7B tg128 **50.47** / @1641
**48.99** (+1.04/+1.07%). 14B @3254 distance-to-parity: −10.8% → **−9.7%**
(21.95 vs llama 24.32). Stage-3 front: matmul aggregate (~2.9 ms) + launch
structure (unfused decode qkv chain ~0.45 ms/step — FusedQKV not active on
CUDA — plus the small-kernel tail) + exposed-latency attention (≤0.63 ms,
mechanism uncertain) + mechanism-less stragglers (§2D D3-7).

**D3-8 update (2026-09-08):** the G4 FusedQKV decode fusion is now ACTIVE on
CUDA, both layer classes (Stage-3 Tier A). One `attn_bias_rope_store` kernel
(pointer-form: concat sections OR three separate buffers) replaces the
per-layer bias×3 + rope×2 + store×2 chain; class 1 (same-quant wq|wk|wv —
24/48 layers at 14B, 14/28 at 7B) also collapses 3 matmuls into one concat
matmul; class 2 (mixed quant) keeps its 3 matmuls and fuses only the
epilogue (`Op::QkvBiasRopeStore`, CUDA-only). Decode launch census −310/step
(−22.5% trace total). Wall (interleaved, every pair clean): 14B tg128 23.81
→ **24.55** (+3.11%), @3254 22.04 → **22.40** (+1.63%); 7B tg128 **50.98**
(+1.05%), @1641 **49.51** (+1.23%); isolation (post vs post
`MINFER_NO_FUSE_QKV=1`): +3.15/+2.28/+1.03/+1.00%. Bitwise throughout:
probe tests + dump gate (logits + ALL KV byte-identical both models) +
greedy 5/5 seeds × 2 models byte-identical + NO_FUSE control + rp=1.0 +
sampled controls; suite 172/0/3. 14B tg128 crosses parity (24.55 vs llama
24.31, 1.010×); 14B @3254 distance-to-parity −9.7% → **−7.9%** (§2D D3-8).

**D3-6 update (2026-09-08):** Stage 2's GQA q-head batching was built and
REVERTED with the mechanism nailed: eliminating the 5× K/V L2 re-read
(2.12M → 473K sectors = the 1× analytic minimum, ncu-verified) leaves live
kernel time flat (63.8 → 64.8 µs) — the re-read was already hidden under the
latency roofline, so the 14B attention residual is exposed-latency, NOT
traffic; no bytes-side attention lever remains (§2D D3-6). Every correctness
gate was green (incl. byte-identical penalty-free greedy streams on both
models — the new sampler-vs-kernel attribution gate). attn_v-q6K MMVQ
routing (2b, ≈ +0.6% projected) deferred; rms-launch consolidation (~1.0 ms)
is the largest remaining known lever.

### 1.2 Wall decomposition (converged regime, r55/r58/r59-era records)

Production nsys at nt=3314, kernel busy ≈ 984 ms (r58 census, pre-r59):

| Slice | Share | vs-llama | State |
|---|---:|---|---|
| q4_K BT GEMM (`mmq_raw_nb_bt`) | 63.2% (622.4 ms; r59: −30.9% → 526.8 ms) | 1.06× | **closed** absent a llama.cpp-style q8_1 GEMM-prologue rewrite |
| q6_K BT GEMM (`mmq_raw_nb_bt_q6k`) | 15.4% (125.5–148.8 ms) | ~1.1× kernel-side | **closed** (r53/r56: both staging planes precomputed, all stagings async) |
| fused A-producers (swiglu+quant, rms+quant) | 8.8% (swiglu 64.0 ms) | swiglu at **89% of the 273 GB/s DRAM roofline** | swiglu closed (cap +0.74%); rms at 56% roofline = last incremental lead (ideal +0.98%) |
| FA prefill attention | 5.3% (51.8 ms) | **2.43×** taken in r48 (5.16 → 2.12 ms) | closed for tile levers (r46/r50/r57) |
| elementwise/rope/kv/store | ~6% | — | bandwidth-bound |
| standalone wo quantize + mmvq tail | ~2% | — | tail section runs the MMVQ/native path |
| host/launch gaps | ~1% | — | one-time stalls 0.6% + recurring gaps ~0.1% + tail malloc 0.1% |

**Campaign verdict (r55, re-validated by r59/r60): no identified lever
≥ +1.5% remains within the current architecture; the next meaningful step is
the step-function q8_1 GEMM-prologue fusion, not incremental optimization.**

### 1.3 What the engine does now (dispatch shape)

All in `src/cuda_kernels.cu` + `src/cuda.rs`, dispatched by
`src/graph/cuda_backend.rs`:

- **Weights resident at load**: every matmul weight uploaded once and
  registered by name; q6_K optionally 224-byte-padded (7e②). Registration
  additionally builds the q6_K `W_exp` (1.52 GB) and q4_K/q6_K `W_dsc`
  (363.2 MB + 1456 MB) planes when their gates are on (Appendix A).
- **Prefill (nt ≥ 16)**: the promoted MMQ path — `quantize_q8_0_pad40_t`
  pre-transposed A planes (r34), fused producers (r51/r52), NB-BT raw-byte
  int8-tensor-core GEMMs for q4_K (`mmq_raw_nb_bt_kernel`, 2 blocks/SM) and
  q6_K (`mmq_raw_nb_bt_q6k_kernel`, KSPLIT=2, 3 blocks/SM, cp.async
  A/B/dsc staging), FA-style tiled prefill attention (8n, FAP2 register
  softmax). Fallbacks compile in and are byte-identical (EXP=false,
  DSC=false, mode-1 producers, generic `mmq_nt`/f16 arms).
- **Decode (nt == 1)**: per-type MMVQ over q8_0 activations (8e/8e② + R2
  v2), fused bias+rope+store, whole-step CUDA-graph capture/replay (7d);
  pinned D2H logits readback (R3-A2). Repeated identical-nt prefills capture
  after the 3-run protocol (R3-B); one-shot prefills never capture (r55:
  measured no-win + capture-illegal mid-window malloc).
- Memory etiquette (shared box): no raw allocation probes; check `free -g`
  before suite runs (the suite transiently reserves up to ~100 GB of the
  overcommitted pool); benches stay at single-process 7B scale while sglang
  serves.

### 1.4 Remaining roadmap (post-campaign)

- **Not planned** (revisit with a concrete need): cuBLAS/cublasLt (closed as
  8k — 8m's wmma GEMM covered the f16 path), VMM pool, multi-GPU, node
  reordering, Windows, IQ/Q2/Q3 quants.
- Open leads, all sub-bar or step-function: q8_1 GEMM-prologue fusion (the
  step change); rms_nw roofline (+0.5–1%); wave re-tile for small-od classes
  (+0.3–0.8%, needs ≤85 regs); fused ffn_gu concat (needs the G5 nf≤16384
  gate re-measured); FA deep-opt only with numerics-order-preserving
  structure (r50/r57 caveat).

## §2 Chapters — one per history-table row

Chapters follow §0 row order. **Method** = what was built/changed and the
mechanism; **Result** = measured numbers + gates; **Lesson** = the
transferable rule. REVERTED rows record why the lever failed and the bound
it established.

### Era A — Phase 7/8 foundations (2026-08-30)

#### Phase 7 — CUDA backend (row 1)

**Method.** Raw-CUDA-FFI device layer (`cuda.rs`, `cuda_kernels.cu`) behind
the compute graph: per-op dispatch, resident weight registry, CUDA Graph
capture/replay, pinned staging — Phases 7a–7e, recorded in
`docs/CUDA-BACKEND-PLAN.md`.
**Result.** First working CUDA path; 7B @2K prefill 30.7 tok/s (per-op CPU↔GPU
ping-pong still dominated — the Part-IV diagnosis, fixed structurally here).
**Lesson.** Resident weights + graph-backend dispatch is the precondition for
everything after; per-op H2D/D2H on the hot path is fatal at this scale.

#### 8m/8m② — tiled wmma f16 prefill GEMM (row 2)

**Method.** One 64×64 f16 tensor-core GEMM over all 8 quant types replaces
per-token weight re-streaming (`ba3f317`); cp.async tile staging added on top
(8m②, `cdc6599`).
**Result.** 7B @2K: 30.7 → 294 → 1204 tok/s (39× total); 31 → 35 TFLOPS with
staging.
**Lesson.** A single tiled wmma GEMM amortizes weight traffic across the whole
tile — the f16 baseline everything else competes with.

#### 8n — FA-style tiled prefill attention (row 3)

**Method.** Tiled prefill attention with online softmax and a register O
accumulator; f16 probs at a 256 B stride to avoid a cross-thread
score-clobbering race found by a standalone harness.
**Result.** 176 → 8.5 ms/layer at 7B @2K (20×).
**Lesson.** A standalone harness catches cross-thread race classes the graph
parity tests cannot.

#### 8o — decode-start CPU stalls (row 4)

**Method.** Killed a ~635 ms full weight re-clone per graph rebuild
(`Tensor: Cow::Owned` makes `clone()` deep-copy) and a ~920 ms eager concat
probe in the decode-graph build.
**Result.** First decode step 724 → 35 ms; all decode rates unchanged.
**Lesson.** `Cow::Owned` turns `clone()` into a deep copy — audit clones on
startup paths.

#### 8p — persistent f16 weight cache + fused dequant-in-GEMM (row 5)

**Method.** Dequant once per weight at load instead of every call (288
ms/call on 7B); enabled by the loader only when quantized matmul weights
total ≥ 2 GB (`W16_ENABLE_BYTES`). Fused `gemm_qb_nt_kernel` (dequant-in-GEMM,
all 8 types) added as the memory-lean alternative (`MINFER_FUSED_B=1`;
slower on large nt).
**Result.** 7B @2K prefill → ~1400–1500 tok/s. The bitparity test exposed a
latent `cudaErrorMisalignedAddress` in `dequant_q5_0_f16` (u32 load at blk+2
on 22-byte blocks).
**Lesson.** Load-time materialization beats per-call dequant; a bitparity
test pays for itself immediately.

#### 8e/8e② — decode MMVQ (row 6)

**Method.** Integer `__dp4a` dots over q8_0 activations per llama.cpp's MMVQ
design (`MMVQ_PARAMETERS_GB10` launch table: 8 warps, one output row per
block); q6_K/q5_K kernels + the od·id ≥ 24M shape gate.
**Result.** 7B decode +37% (q4_K); q6_K/q5_K follow-ups; dispatch gated by
shape so small tensors keep the safe path.
**Lesson.** Port the launch-table parameters, not just the math — the config
is part of the design (the Part-IV "stream-K" misreading is corrected here).

### Era B — R and P5 sessions (2026-08-31 → 09-01)

#### R3 — small-model per-token overhead (rows 7–9)

**Method.** 0.5B decode was ~4.0 ms/token with ~1.6 ms GPU floor — ~2.4 ms
CPU/sync overhead, found via `MINFER_GRAPH_TRACE` + DOT dump: (A1) the G3
tail-row-reduction input `tail_ids` was declared MID-graph, splitting every
prefill into 4 splits (2 extra full-stream syncs + host round trips); moved
to the graph head (conditional on `n_out < nt`). (A2) per-step logits readback
used a blocking `cudaMemcpy` into a PAGEABLE Vec (driver-internal pinned
bounce) + a redundant clone; replaced with grow-on-demand pinned staging,
no clone (`MINFER_NO_PINNED_READBACK=1` reverts). (B) prefill capture
defaults ON (8g②'s validated opt-in becomes automatic; 3-run protocol
bounds the cost) (`MINFER_NO_PREFILL_CAPTURE=1` opts out).
**Result.** Prefill is one CUDA split (decode already was); greedy bit-identical;
bench recorded under ~96% co-tenant util — pinned path parity-to-slightly-ahead
interleaved.
**Lesson.** Input declaration position is graph topology: a mid-graph input
splits execution. Trace before assuming where overhead lives.

#### R1 — int8 MMQ prefill GEMM, opt-in (row 10)

**Method.** Not a verbatim llama.cpp port: custom kernel on the 64×64×256
tile implementing llama's MMQ *math*. Activations quantize to q8_0 once per
launch (pad40 + per-block int sum at offset 36); weights stay RAW (no f16
dequant, 2–4× less weight traffic). Tiled `mma.m16n8k32.row.col.s32.s8.s32`
(sm_80+; sm_75 keeps f16), one 32-k chunk per step; int C rescaled per
(token, row, k-block): `sum += da·ds·acc + da·dm·sa`. K-quant nibbles
UNSIGNED; q6_K = k32 chunks with dual m16n8k16 + separate accumulators.
B staging: 2 threads/row, 8 chunks per double-buffered ~94 KB dynamic-smem
tile (sm_121: 1 block/SM).
**Result.** Parity: `cuda_prefill_mmq_parity` (all 8 types × 8-shape sweep,
max diff < 1e-3); greedy 7B ≡ f16 path token-for-token. Perf: 155 tok/s
under sglang ~96% util, 412 quiet, vs f16 630–880 / 1460; fixed overhead
~0.6 ms; ~2.9 TMAC/s vs llama ~24. KD=8 beat KD=4 under load; per-chunk
staging exposed full load latency (2.5 TMAC/s); ncu blocked on the device
(ERR_NVGPUCTRPERM) — the ≈8× gap unprofiled. Committed behind `MINFER_MMQ=1`
(default stays f16 — a 3.5× regression at this state).
**Lesson.** Land parity-clean opt-ins even when slow: the raw-line campaign
(r7+) starts from this scaffold. Re-measure the same code in a quiet window
before judging it (441, not 155).

#### R2 — MMVQ weight-streaming rework (row 11)

**Method.** The 8e kernels ran at ~60% of llama's effective streaming rate
(147 vs ~197 GB/s): each 32 B nibble chunk was read per SUB-BLOCK (sibling
sub re-reads the same bytes; 2× load instructions), and q6_K fetched ql/qh
as eight 2-byte loads. v2 (default when `id % 256 == 0`, q6_K needs the
padded 224 B stride; `MINFER_MMVQ_V1=1` forces v1): q4_K/q5_K one thread per
sub-PAIR (chunk loads once, uint4×2, serves both subs); q6_K one thread per
is-pair with uint4 loads; q5_K's qh plane is bit-indexed (no per-chunk
offset).
**Result.** tg128 42.2 → 45.1 (+6.9%), @2K 36.7 → 38.8 (+5.7%); llama at
47.1 / 44.9. A q6_K nibble-group bug and a q5_K qh-offset bug slipped past
the first suite run because the original id=2176 shape dispatched to v1 —
caught by the engine-level greedy check after adding an id=2560 shape.
Greedy v1 ≡ v2 token-for-token.
**Lesson.** Test shapes must exercise every dispatch arm (the gate condition,
not just the kernel); instruction count, not bytes, carried this gap.

#### R4 — decode split-attention dim-parallel rewrite (row 12)

**Method.** The 8d flash-decoding pass was the whole @2K decode gap: nsys:
`gqa_attn_split_partial` ~150 µs/layer (28 × 150 µs of the
~25.8 ms 7B @2K step) at 28 GB/s effective on a 4.3 MB K+V read — the
runtime-indexed `float4 oc[32]` accumulator lived in LOCAL memory (~80
MB/layer re-read/re-written per rescale), lanes walked rows with 4-byte loads
(64 scattered sectors/row, 12.5% sector efficiency), 224 single-warp blocks.
Raising split count made it monotonically worse (148/172/419/609 µs for
8/16/32/64). Rewrite: each lane owns 4 fixed dims (ONE float4 in registers,
zero spill; `hd % 4 == 0 && hd <= 128` dispatch), coalesced row accesses,
warp-reduction dots, rows in batches of 4 to overlap the serial softmax
chain; `ATTN_SPLITS` 8 → 32 (fixed grid stays capture-safe; idle splits
write mx=-INF/S=0 partials).
**Result.** Kernel 148 → 79 µs/layer; @2K 39.2 → 43.2–45.1 (gap 14% → ~4%);
tg128 45.1 → 47.5–47.6. SPLITS=64 no better (44.7 vs 45.1); parity sweep
extended to the SPLITS=32 chunk boundaries.
**Lesson.** Register-resident accumulators + coalesced row walks; more
split-parallelism thrashes L1 via local memory — measure before raising it.

#### P5 — prefill gap: GEMM tiles + flash-attn rewrite (rows 13–19)

Goal: close the 7B @2K prefill gap (1435 vs llama.cpp 3401, 2.37×). Each
step suite-verified with interleaved same-binary A/B (full-output diff, never
prompt-echo grep). The record's session-range start `b8568cd` no longer
resolves; the code commits are below.

- **P5·0 — FA P·V on tensor cores** (`86ca78c`): 10.06 → 4.24 ms/layer, +15%
  whole-prefill.
- **P5·1 — elementwise vectorization** (`d713e6e`; the record labels it "8p",
  a label collision with Phase-8's 8p f16 cache): store_kv_f16 1→4 dims/lane,
  convert_f32_f16 1→8 elems/lane. 1435 → 1493 (+4%).
- **P5·2 — 128-wide GEMM tiles** (`725e307`, "8q"): `gemm_f16_nt_kernel_t<TM>`
  with TM=128 halves B-panel L2 re-reads and barriers per FLOP (455 → 302 ms
  kernel time); TM=64 kept via `MINFER_GEMM_TM=64`. First version corrupted
  the GEMM: `fb[1]` must offset +16 ELEMENTS (k-half) not +16 rows — caught
  by `cuda_prefill_f16_gemm_parity`. 1493 → 2267 (+30%).
- **P5·3 — FA softmax on all 8 warps + padded smem rows** (`fc07c04`, "8r"+
  "8s"): warp-per-row online softmax (8 warps × 8 rows, shuffle reductions,
  -INF seeds) replaced 64-deep serial chains on 2 of 8 warps (+3%); with
  hd=128 every smem row was 256 B ≡ 0 mod 32 banks → 8-way conflicts, fixed
  by +8-half row stride (272 B). Two bugs en route: (a) double-buffered
  working set exceeded the 99 KB/block smem cap → `cudaFuncSetAttribute`
  failed → SILENT fallback to the legacy kernel (313 tok/s) — the fallback
  now prints a warning, padded layout ships single-buffered (69 KB); (b)
  dropping the double buffer left cp.async without
  `commit_group`/`wait_group 0` — a bare `__syncthreads` does NOT order async
  copies (0.28 parity diff). 2267 → 2365–2371. fa_prefill_f16kv: 4.25 → 1.92
  ms/layer (llama 0.79).
- **P5·4 — k-step KS=64 negative** (`1365c82`): halves barriers per FLOP but
  56 KB footprint halves resident blocks — 1464 vs 2345 tok/s (−38%).
  `MINFER_GEMM_K64=1` re-tries; KS=32 stays default.
- **P5·neg — TM=256** (`a189837`): −3%; thread-count parameterized on the way.
- **P5·neg — in-kernel f32→f16 A staging / AF32 mirror** (`b254c22`; WIP
  chain `a3b0dcd`, `69c3933`, `aa40ed3`): −8% end-to-end; the standalone
  mirror was proven correct but end-to-end integration never reached parity —
  abandoned. (r23 later quantified the convert pass at 6% of the f16 wall.)

**P5 net: 1435 → 2340–2370 tok/s; gap 2.37× → 1.43×.** Post-P5 nsys budgets
(per 2K prefill): GEMM ~597 ms (~46 TFLOPS eff; llama 455 ms), FA ~54 ms,
convert ~56 ms, swiglu ~51 ms.

**Lessons.** (1) Silent fallbacks are a correctness hazard — guard resource
caps loudly (the r8 phantom and r58's fake-OOM are the same class). (2)
`__syncthreads` does not order cp.async — use commit/wait groups. (3) Tile
widening pays until smem halves residency: depth-vs-occupancy is the
recurring tradeoff of this whole campaign (r38/r39/r40 re-learn it on MMQ).

### Era C — P6 q4_K MMQ line (2026-08-31 → 09-05, r5–r37)

Goal as set at r6: MMQ GEMM 6.1 TMAC/s → ≥24 (f16 parity) or ~30 (llama
parity); at parity the MMQ path (quantize ~130 + GEMM ~600) deletes the 56 ms
convert → ~2670 tok/s; at llama parity → ~3250 tok/s. The campaign closes at
per-IMMA parity (r36) and 1.06× wall (r47), with the whole-prefill arc
carried by the q6_K/FA/prepass lines of Era D.

#### r5–r6 — re-rank + the structural rewrite spec (row 20)

**Method.** KD=4 retested 427 vs 438 tok/s (occupancy 1→2 blocks/SM does not
pay for the word-staging kernel; staging-depth amortization dominates). The
bigger 4-warp 32×32 per-warp tile measured 399 tok/s AND broke `mmq_w80`
parity (zero cells — coverage hole in hand-mapped fragments) — reverted. Wall
-split correction: the 352 ms dequant pass runs at LOAD (w16 warm), not in
the prefill wall — a vectorized q4_K dequant was a null delta and reverted.
The 890 ms f16 prefill wall = GEMM ~600 + convert 56 + fa 54 + swiglu 51
(~257 GB/s, bandwidth peak) + add 19 + host gaps — GEMM ~85% is the only
material lever. The r6 spec (executed by the rest of Era C): smem holds RAW
bytes only (A pad40 40 B/chunk, B 144 B superblock), staging = pure cp.async
16 B chunks, dequant IN REGISTERS at mma time, warp tile 32×16 × 8 warps,
land as `mmq_raw_nt_kernel<TYPE>` gated `MINFER_MMQ_RAW=1`, q4_K first.
**Result.** Measurement-only + two reverts; the spec became the campaign's
contract.
**Lesson.** Re-read profiles against the current code before ranking levers
(the dequant pass was never in the wall); write the execution spec with
parity gates before touching kernels.

#### r7–r8 — raw-byte kernel + wide tile + FA probe (row 21)

**Method.** `mmq_raw_nt_kernel` (MINFER_MMQ_RAW=1, q4_K): raw-byte staging,
dequant at mma time; quantize pass rebuilt (tree-reduced amax + packed
stores, `d9d626a`). Wide tile `mmq_raw_wide_nt_kernel` (128-token block, 8
warps × 32×32, `a41eac0`). FA Step-4 probe: L2-prefetch of the next KV tile
before the single-buffered stage (`87bade0`).
**Result.** Raw kernel parity green, 7B token-identical: 472 vs 441 (+7%),
kernel 20.2 vs 23.0 ms, quantize 129 → 74 ms. Wide tile: the first 2124 tok/s
was a PHANTOM — KD=8 needs 135 KB dynamic smem, over the ~99 KB opt-in cap;
attr-set AND launch both failed silently and the GEMM wrote nothing. The
launcher now guards the cap and refuses (Rust falls back to narrow); KD=4
(86 KB) is the only feasible wide depth. Constraint re-rank (all parity
clean): narrow KD=8 472 > wide KD=4 428 ≈ narrow KD=4 427 > R1 441 — wide's
halved B traffic bought nothing (L2 absorbs B re-reads); the shared
bottleneck is per-chunk inner-loop overhead. FA probe: null (2319–2345 vs
2345) — GQA sharing (7 q-heads/kv-head) already keeps KV tiles L2-hot;
reverted, fa_prefill_f16kv stays 1.86 ms/layer, ~6% of the wall.
**Result (MMQ status at r8).** Best 472 is 4.9× off the f16 GEMM (2318);
default f16 path untouched at 2320–2370.
**Lesson.** A fast wall time with a silently-failed launch is a phantom —
guard smem caps and verify the kernel actually ran; B-traffic reduction is
dead on L2-resident re-reads.

#### r9 — the llama.cpp MMQ reference, decoded; shape axis closed (row 22)

**Method.** Read the actual reference (mmq-config-ampere.cuh for Q4_K): 256
threads, occupancy 1, SRAM tile I=128 × J≤128, ITER_K=256, SYNCHRONOUS
staging, float sum accumulators, 16 mma.m16n8k32 per warp per chunk.
Instruction model: ~0.018 inst/MAC/thread vs our 0.133. Sync-staging applied
to the wide kernel (single buffer, 54 KB, 2 blocks/SM) as a test.
**Result.** Sync-wide parity green, 462–466 vs narrow 481. Full shape matrix
@2K (all parity-clean): narrow cp.async KD=8 **481** (local optimum) > sync
wide 464 > compact 410 > wide cp.async KD=4 428 > narrow KD=4 427 > R1
word-stage 441. Shape axis exhausted with evidence. Remaining lever: llama's
pre-arranged mma-fragment B layout, producible at weight-load time.
**Lesson.** Copy the instruction model, not the shape — and measure the whole
shape axis before optimizing inside one point.

#### r10–r11 — reference inner-loop decomposition ported; ILP verified (rows 23–24)

**Method/Result.** r10 ported the universal q8_1×q8_1 decomposition (nibble
unpack + dmin fold once per (row, chunk) in staging; compute applies
preloaded scales; per-k signed int8 B): parity green, 462–468 vs narrow 470 —
FLAT at ~6.4 TMAC/s vs llama ~30 with the SAME decomposition. Residual
localized to (1) ILP depth (llama: 16 independent mma chains + 128
accumulator regs; ours 8 chains/~64 regs), (2) ldmatrix A staging (1 LDSM vs
8 LDS.32 per fragment), (3) tile 128×128. The session's edits were lost to a
post-checkout hook revert after measurement (measurements valid). r11
verified the tile `ne` reading: NVIDIA (Turing+) is ne = I·J/32 = 4 C regs
per m16n8k32 (the earlier I·J/64 was the AMD MFMA branch) — the 16-chain
reading stands.
**Lesson.** When the same math is 5× slower, the delta is NOT the math:
instruction-level structure (ILP depth, load width) is the suspect class.

#### r12 — 16-chain warp tile + ldmatrix LANDED (row 25)

**Method.** Block 128 tok × 128 od; each warp owns a private 16-od-row slice
× the full 128-token tile = 8 A-frags × 2 B-frags = 16 independent mma chains
per chunk with `sum[64]` + `clow[8][2][4]` live; A fragments via
`ldmatrix.m8n8.x4`; KDR=4 restage-skip when consecutive k-tiles share a
super-block; B format and two-term rescale untouched. A-row 48B padding tried
and reverted (flat).
**Result.** Parity green, suite 169/0, greedy identical. Wide-16 KD=4
1020–1058 vs narrow 441–481 same-session interleaved = ~2.3×; KD=8 973–995;
vs the pre-rewrite wide (~719) = 1.44×. MMQ ~2.3 TMAC/s-class, still ~2.2×
below the f16 default (2284 same session). Best config: MINFER_MMQ=1
MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_WIDE=1 MINFER_MMQ_RAW_KD=4.
**Lesson.** Express the ILP the hardware needs — the compiler cannot create
chains the source does not declare. (This is the largest single P6 structural
landing before r34.)

#### x-tile / j-tile / cp.async-db — the staging-shape family, closed (rows 26–28)

- **256-token x-tile** (`f061cb8`): block 256 tok × 64 od. Parity green,
  REGRESSED: ~942 vs 1035–1058 (kernel +27%, same L2/SM utilization for 27%
  longer). Root cause: halving the od tile doubles y-blocks; A re-staging
  (4,480 B/token) scales with od/WBJ while weight bytes per visit are 2,016
  B/row — A re-reads (327 MB) were already ~2× B (152 MB), so −72 MB B for
  +327 MB A is strictly negative. The 128×256 alternative is impossible:
  512 threads cannot be resident at REG 156–166.
- **j-tile A-reuse (128tok × 256od)** (`4993804`): outer jh loop, A staged
  once per k-tile, reused by both 128-od halves; `sum[2][64]`; B re-expanded
  per (kt, jh). Parity green first build; ~1067 (+2.6%, bar 1150 missed); A
  re-reads halved (−163 MB) but the kernel is latency-bound at 1 block/SM
  (2.00 active warps/sched, issue 0.22) — the added second B-stage + barrier
  round gives back what the A saving removes.
- **cp.async double-buffered raw staging** (`784786d`): two-level staging,
  raw transfers via cp.async one k-tile ahead, expansions smem→smem. Parity
  green; ~1034 (neutral); L2 SOL rose 75.7 → 78.2% — the prefetch WORKS but
  the kernel is L2-THROUGHPUT-bound, not MLP-starved.
**Lesson (the family).** With x-tile, j-tile, and latency-shape all measured
flat, staging order is closed: remaining levers are per-MAC L2-byte
reduction or SM-side instruction efficiency — not traffic shape.

#### r13 — counter forensics vs llama.cpp (row 29)

**Method.** First working ncu session (metrics behind
`sudo -n env LD_LIBRARY_PATH=... ncu`; GB20B has NO `dram__*` metrics — use
`lts__t_sectors_aperture_device`; int8 mma counts under the imma subpipe, not
the hmma counter, which reads 0 for both). Matched q-proj GEMMs per GMAC.
**Result.** Per-GMAC: warp instructions 10.14 M vs llama 6.06 M (1.67×);
IMMA identical (2.0–2.05 G = 2×MAC — mma work at parity); LDS conflicts
940.3 K+499.0 K vs 6.5+0; L2 read sectors 22.0 vs 13.7 MB (only 1.6×);
duration tracks instruction count ~1:1 across all data points (~0.10–0.15
warp-inst/ns on both engines). Two implemented fixes: FULL variant (merged
A staging, per-superblock scale staging, 272 B qb8, float2 C stores) — L1
requests −55% but instructions +33%, wall −16%; MINIMAL (272 B stride +
float2 C stores) — +0.3% noise. UNEXPLAINED then: ~66 MB/GMAC of L2 write
sectors with no source-level writer (later understood as counter artifact /
denominator effects).
**Result.** Neither L2 bytes, nor store efficiency, nor bank conflicts bind
the kernel — all three fixed simultaneously with zero wall effect.
**Lesson.** Per-MAC instruction count is the first-order predictor for this
kernel class on GB10 — but see r25: only while issue is stall-bound.

#### r14 — B-fragments via ldmatrix + widened scale loads LANDED (row 30)

**Method.** Three changes to the wide kernel only: (1) B-fragments via ONE
`ldmatrix.m8n8.x4` per (warp, chunk) replacing 4 LDS.32 — qb8 re-tiled
SLOT-MAJOR `[8 sg][128 od-row][48B]` (48B stride is 16B-aligned and gives 8
distinct bank phases; the ldmatrix distribution equals the mma B-operand
layout, verified standalone); (2) od-col scales packed float2 (d | dmin·m) —
8 LDS.32 → 2 LDS.128 per minitile; (3) sda_q uint2-tiling — 16 LDS.32 → 8
LDS.64. First cut's uint4 tiling overflowed KDR·1024 B and corrupted qb8
(found by bisect after fixing a missing j0w term and a word-offset slip).
**Result.** Wide KD=4 1225 (+18.5% vs 1036), KD=8 1273 (+23–30%); narrow
control noise. ncu: warp-inst −6.8%, shared-load inst −64%, LDS conflicts
−47%, duration 3.632 → 2.378 ms (−34.5%) — most of the win is issue
efficiency, confirming r13's longsb reading. Parity green both depths; suite
166/0/3; greedy-32 identity. Gap to llama: 70.4 vs 41.1 µs/GMAC.
**Lesson.** Fewer/wider smem ops with zero staging-ALU growth is the lever
class that pays while stall-bound; relayout-only changes are free
correctness-wise if validated standalone first.

#### r15 — f32-accumulate mma probe + rank-1 term2 rescale LANDED (row 31)

**Method.** r14's named next lever (`mma...f32.s8.s8.f32`) is IMPOSSIBLE:
ptxas rejects it on sm_80–121 under PTX 8.8/9.0 while the s32 control
assembles — integer mma has s32-only accumulators, and no f16/tf32/fp8
operand carries q8·scale bit-exactly (~18 significand bits needed vs 11).
The fallback (accumulate 2 chunks in int before one I2F) is mathematically
INVALID regardless of the 2^21 bound: the rescale coefficients differ per
32-k chunk, so a merged int dot loses the split. What landed: the dmv
correction term is RANK-1 in (token, od-col) — `dma = da·(float)sa` replaces
64 FMUL/chunk with 16 FMUL + plain FFMAs.
**Result.** KD=8 1295.0 (+1.9%, 3/3 consistent); KD=4 noise; narrow control
stable. ncu: 9.45 → 8.69 M/GMAC (−8.1%), duration −4.9%; SpeedOfLight
Compute (SM) 31.5% — stall-bound, so further ALU cuts pay sub-linearly.
Suite 166/0/3, greedy-32 identity. Rescale floor now ~346 ops/chunk; the I2F
stream is ISA-irreducible.
**Lesson.** Probe ISA limits with a raw-PTX build before designing around
them; exploit rank-1 structure in rescale terms.

#### r16 — narrow kernel gets the rank-1 fold (row 32)

**Method/Result.** r15's one-line follow-up ported to `mmq_raw_nt_kernel`
(3 sites): 8 I2F + 16 FMUL → 4 I2F + 12 FMUL per chunk. Patched 480.2/480.9
vs baseline 447.0/472.6 (+1.7%/+1.8%, within noise of r15's control).
Parity green at KD=8 and KD=4; suite 166/0/3; no ncu (narrow is not the perf
path).
**Lesson.** Keep sibling kernels instruction-compatible when a fold lands —
or drop the sibling.

#### r17 — wide warp remap 32od × 64tok REVERTED (row 33)

**Method.** Pure index remap to llama.cpp's warp shape (each warp owns
32 od-rows × 64 tokens; B-frags via two ldmatrix.x4; sum[64] unchanged).
Parity green first build.
**Result.** KD=4 1250 vs 1239 (+0.9%), KD=8 1301 vs 1288 (+1.0%) — noise-band,
far under the bar. ncu: warp-inst −5.8% (LDSM 72 → 48/chunk) but duration
−0.3%; SM% 31.5 → 29.75; SM Active Cycles +9.2% — issue efficiency dropped,
offsetting the leaner stream.
**Lesson.** Third independent confirmation: pure per-MAC instruction cuts pay
~0 wall while SM% sits at ~30 — the binding lever is latency/stall structure.

#### r18 — load-time B pre-expansion REVERTED (row 34)

**Method.** Materialize the qb8 expansion once at weight registration: EB
per-k int8 plane (one k-tile = one contiguous 32 KB range) + SB per-(chunk,
row) float2 scales; staging becomes 2048 16B LDG/STS per superblock, zero
dequant ALU; +5.8 GB device on 7B.
**Result.** Parity green first build. KD=8 +0.9% wall (noise; kernel duration
−4.5%); KD=4 −19% median with wild variance (bulk copy reads 256 B/row vs
144 B — 2× staging bytes). Instructions −0.3% (the expansion ALU was ~0.1%
of the stream). Box −9% vs the r15/r17 session — only intra-session deltas
count. Bar ≥1350 decisively missed; reverted; EB/SB machinery preserved as
prerequisite for any future L2-persistence experiment.
**Lesson.** Staging-ALU cuts extend the r17 rule (op cuts pay ~0 at SM% 30);
byte-bloat can regress at shallow staging depth.

#### r19 — weight L2 residency REVERTED (row 35)

**Method.** (1) `__ldg` on weight-side loads; (2) `MINFER_MMQ_L2WIN=1`:
cudaAccessPolicyWindow on the raw weight range (hitRatio 1.0, Persisting),
carveout at max.
**Result.** `__ldg` neutral (+2% KD=4 median; L2 was not the constraint —
SOL 37–46%). The persisting window: −50%, 6/6 consistent — 12.8–34 MB/weight
of persisting lines starve C stores, activations, KV; every re-mark churns
the carveout. Untested cheaper variants (hitRatio ~0.25, small weights only,
per-layer reset) out of budget.
**Lesson.** The mechanism works; the direction is wrong at hitRatio 1.0 —
persisting windows need selectivity, and "read-only hint" is a no-op when
the loads were already non-coherent.

#### r20 — split-phase A staging LANDED (row 36)

**Method.** Session question: why is llama at issue 0.42/sched vs our 0.26 at
identical occupancy? Matched nt=512 stall sets on both q-proj kernels +
PC-sampling. Findings: barrier density already identical (4 barriers/256 k
both sides — the "their staging is less synchronous" hypothesis false at
source level); IMMA exactly at parity (1,605,632 both); the gap carrier is
long_scoreboard (6.22 vs 1.15 per-issue-active; 97% of the named-stall
excess) sampled inside the A-staging LDG→STS chains (top site 16.7% of ALL
stalls, one ~600-cycle latency per 4-deep batch; A fetch 8-sector-scattered,
62.5% sector efficiency). Fix: issue ALL A-staging global loads into
register arrays first (`av[KDR*4]`, d/sv), then store — identical addresses,
traffic, instruction count; only the dependency schedule changes.
**Result.** KD=4 1230.4 → 1317.7 (+7.1%), KD=8 1275.8 → 1319.9 (+3.5%), 6/6
reproducible (landed despite missing the ≥1350 bar). ncu: longsb 6.22 → 2.92
(−53%), duration −12.2%, warp-inst −16%; freed stalls moved to lg_throttle
0.33 → 2.38 — staging latency and request count are serially co-binding.
Not the cause (all measured): barrier structure, launch shape (stream-k
wash; fixup +34 µs), occupancy, tensor work, memory bytes.
**Lesson.** Split-phase scheduling converts latency-bound staging into
queue-bound — one fix moves the bound rather than removing it (r21 proves
the second half).

#### r21 — coalesced block-linear A staging REVERTED (row 37)

**Method.** r20's "remaining gap (1)": warp-strided uint4 loads over
contiguous per-token regions (d/ssum folded during store; split-phase kept).
Parity took three fixes (per-lane slice count; idx from `lane`; global token
must include i0 — only the nt=256 sweep shape caught it); a byte-exact CPU
simulator confirmed identical smem images.
**Result.** Sectors 79.35 → 56.67 M (−28.6%), lg_throttle 1.42 → 0.14
(−90%), mio/math throttle gone — but `wait` +110%, short_scoreboard +126%,
warp-inst +11.5%, duration +10.9%, wall −2.3%/−0.7%.
**Lesson.** Stall mass is conserved: the freed issue slots re-saturate on the
next bound. Sector efficiency and request count are NOT the binder;
warp-instruction count is (until r25 revises it to issue/occupancy).

#### r22 — qa8 XOR swizzle LANDED; d/ssum fold REVERTED (row 38)

**Method.** Attribution step: the 16.86 M op_ld conflicts ARE the A-side qa8
ldmatrix 2-way (4.74 M LDSM × 4 phases × 1 extra wavefront per 2-way phase);
op_st (6.5 M) is qb8/sds stores, left alone. Lever 1: XOR-swizzled qa8 —
`gr(row,h) = ((row&3)*2+h) ^ ((row>>2)&7)` on both staging stores and
ldmatrix loads; verified standalone (524,288 → 0 conflicts); final form
precomputes all 8 A-frag offsets ONCE (`G[g]`, lane-invariant) so each
ldmatrix address is ONE IADD and registers DROP below baseline. Lever 2
(REVERTED): fold d/ssum into the A pass-through — both enumeration variants
−19–20% wall: the "scattered" d/ssum loads are L1 HITS (same lines the qs
pass fetched), the flat enumeration pays mod-9 ALU + a branchy store that
breaks r20's deep LDG batch, and ptxas hit 255 regs + 112 B spill.
**Result.** Lever 1: KD=8 1329.6 vs 1311.8 (+1.4%, 3/3 positive); KD=4
−0.5% (noise band); op_ld 16,859,136 → 0. Kept (default-path positive,
mechanism complete); the combined ≥1350 bar not reached — split outcome
documented.
**Lesson.** Fix address generation ONCE per thread (hoisted offset tables),
not per load; request-count "savings" that add ALU lose.

#### r23 — f16-path wall decomposition + FA_TKV lift REVERTED (row 39)

**Method.** nsys on the default f16 path (2659-token prefill), per-launch
bucketing + ncu SOL; co-tenant outlier cleaning; quiet-window 2285 tok/s.
Graph discovery: the prefill is 27 full layers + an nt=1 TAIL — the q6_K
lm_head (1.45 TMAC = 9.6% of model MACs) is already tail-priced; no
[nt,vocab] logits GEMM exists in the wall.
**Result.** Op classes: gate+up GEMM 37.8% (mem SOL 85%), down 27.5% (runs
20% slower per-MAC than gate/up — unexplained asymmetry), q+o 7.5%, k/v
1.4%, FA 7.2% (occupancy 16.7%), convert f32→f16 6.0% (llama pays ~zero —
in-kernel quantize), swiglu 5.6% (DRAM peak), add/rms/rope ~5%, host gaps
0.8%. q6_K GEMMs show NO per-MAC penalty on the f16 kernel (the w16 cache
erases the type). llama split: GEMM 597 ms (MMQ 39.5 µs/GMAC) vs our 59.5
µs/GMAC + 73 ms convert; the prefill gap IS the GEMM byte-width gap — raw
q4_K B-stream (4.5 bit/w) at 85% SOL beats f16 (16 bit/w) by ~1.5×/MAC,
exactly the campaign's premise (updated: MMQ at llama GEMM parity → ~2900
tok/s). FA_TKV 64→32 lift: occupancy 16.7 → 32.68%, kernel −6.7%, wall
−0.3% — the 2× k-loop fixed costs ate the latency-hiding gain; a REAL bug
caught en route (TKV=32 lanes 16–31 must be masked by `c0/c1 < FA_TKV`, not
just `kt+c0 < kv_end`). FA's 2.5×/layer gap is structural.
**Lesson.** Byte-width arithmetic validates (or kills) a campaign premise
before the kernels exist; FA tile reshrinks must re-mask lanes (r46/r50
inherit this).

#### r24 — scheduling-structure ladder REVERTED (row 40)

**Method.** The last untried structural family on the wide kernel. Box
re-measured FASTER than the documented band (KD=8 ~1385–1391 vs ~1330) — the
bar reinterpreted as relative ≥ +1.5%. Rung 1: tile-order swizzle via
`MINFER_MMQ_RAW_SCHED` (a bijection on the grid — bit-identical outputs).
Rung 2: persistent blocks (num_sms blocks walking a strided tile list;
register-identical per ptxas). Rung 3 (full stream-k) not attempted — would
reorder fp accumulation and fail the greedy-identity gate.
**Result.** Swizzle: default x-fastest/B-hot 1370.0 best; A-hot transposed
−2.3%, G-grouped −2.4%. Persistent: non-persistent u-loop −3.3%, persistent
occ=1/occ=2 ≈ same — no wave-quantization tail exists (the GPU pipelines
launches, not lockstep waves).
**Lesson.** The default order was already optimal (B-hot: the small weight
panel stays resident while the unique A streams); persistent grids buy
nothing where there is no quantization tail.

#### r25 — SASS opcode-class census; unroll wall-inert (row 41)

**Method.** Per-opcode-class ncu metrics on matched layer-0 q-proj GEMMs
(thread-granularity `sass_thread_inst_executed_op_*`, /32 validated against
warp counts). Totals reconcile: ours 445,544 warp-inst/tile vs llama
355,758 — surplus +25.2%.
**Result.** The surplus is 100% support instructions: integer ALU +69.5k/tile
(77%), fp32 rescale FMUL +15.2k, conversions +14.3k; IMMA and FFMA
IDENTICAL (114,688 FFMA/tile both) — compute is exactly MAC-bound on both
sides. Memory movement is NOT the surplus (ours is LOWER on shared loads;
LDSM 4.5× theirs — the 128-token warp tile runs 9 LDSM/chunk vs their
plain-LDS A). The unroll fix cut integer ALU −38%, total inst −9.7%
(surplus halved) — wall +0.37/+0.49%, below the +1.5% bar. Reverted; census
is the deliverable of record. Paradigm verdict: the kernel is
issue/occupancy-bound (98 KB smem → 1 block/SM → ~2 warps/sched → latency
unhidden), not instruction-count-bound.
**Lesson.** Attribute the instruction stream before cutting it — then check
whether the class is on the critical path at all (r29 shows the same cut
pays once occupancy is fixed).

#### r28 — Direction-A raw-nibble NB kernel, 2 blocks/SM LANDED (row 42)

**Method.** The one occupancy lever r13–r25 never touched: shrink B smem to
the raw-packed 2-nibbles/byte qs plane and accept a small in-loop B-expansion
cost → 45,056 B → 2 blocks/SM. New parallel kernel `mmq_raw_nb_kernel`
(KD=8-native, 64 tok × 128 od, 8 warps × 16 od-rows, `sum[32]`, 123 regs/0
spill), gated `MINFER_MMQ_RAW_NB=1` AND kd==8 (clean fallback otherwise;
wide stays the default raw path, byte-identical). The #1 risk — the
B-fragment nibble layout — was derived from the verified wide kernel's
ldmatrix path and byte-equated standalone (0 mismatches, all 8 sgs × 32
lanes × 4 regs) BEFORE integration: unsigned 0..15 nibble + fp32 two-term
rescale per chunk, never the `(nib − m)` fold.
**Result.** Parity 1.5e-5..9.9e-5 (pure f32 rounding — a layout bug would be
~1e0); greedy-32 byte-identical. Perf: 1410.4 vs 1375.2 median (+2.56%, 5/5
positive; earlier batch +2.43% 3/3). ncu: warps_active 16.17 = ~4.04
warps/sched (2 blocks/SM confirmed); longsb 2.92 → 2.03; issue 25 → 37.36%.
**Lesson.** r25's verdict inverted: occupancy was the binding resource, and
it is buyable by shrinking smem. Validate layout maps standalone before
integration.

#### r29 — NB kd-loop unroll LANDED (row 43)

**Method.** r25's census re-run on the NB kernel: integer ALU 2.85× llama
(FFMA at parity); stalls cluster on the shared path (mio 18.89% + longsb
16.80%). Candidate ladder: (a) PRMT extraction REFUTED by a standalone
sm_120 micro-test (still needs shift+mask); (b) software-pipelined B-raw
load — NEUTRAL and bloated regs (123 → 177), reverted; (c) LDSM A-frags
already in place. LANDED: `#pragma unroll` on the NB kd loop — each chunk's
select/base/bounds becomes a compile-time constant.
**Result.** 1387.9 → 1426.8 (+2.80%, 5/5 positive; consecutive pair +1.88%);
123 regs/0 spill held (occupancy preserved); parity 1/0; greedy-32
byte-identical; total inst −6.5%, int ALU −25%; suite 166/0/3. (A ~1206
outlier in the un-warmed harness was a GPU power artifact — gone after
warmup.)
**Lesson.** The same instruction cut that was wall-inert at 1 block/SM (r25)
moves the wall +2.80% at 2 blocks/SM — occupancy unlocks instruction cuts,
not the reverse.

#### r30 — SWAR unpack: the compiler already did it (row 44)

**Method.** Replace the in-loop nibble unpack with word-granular SWAR;
standalone byte-equivalence validated first (0 mismatches). Then SASS-first:
`cuobjdump -sass` shows the r29 unroll already made ptxas CSE each raw word
(16 LDS.32 for both chunks, shared SHF+LOP3).
**Result.** Measured anyway (faithful b_hi-carry): +1.09% un-warmed / +0.54%
warmed — noise; SASS deltas +2 SHF/+3 LOP3 = behaviorally identical machine
code. Reverted cmp-verified.
**Lesson.** Read the SASS before writing the lever — a source-level version
of what the compiler already emits can only add overhead (the r30 pattern,
repeated in r32/r33).

#### r31 — q-major sda scale-read repack LANDED (row 45)

**Method.** SASS confirmed ptxas had NOT coalesced the sda reads (4 LDS.64
per chunk). Repack to one-uint32-per-token with a group-region split; a lane
reads its whole per-chunk set as TWO LDS.128 at 16 B warp stride
(bank-conflict-free). The naive q-major 32 B-per-lane first attempt was
2-way conflicted (caught by conflict analysis, +0.57%) — fixed to the
region-split layout.
**Result.** 1424.10 → 1439.40 (+1.07% median of 45 samples; range +0.49 to
+2.38) — below the +1.5% bar, landed as real-but-sub-bar: mechanism
ncu-confirmed (longsb 24.61 → 21.46%, mio −0.92 pp), 109 regs/0 spill, smem
45,056 → 43,008 B. Suite 166/0/3 (one flaky 164/2 re-ran green).
**Lesson.** Sub-bar but mechanism-confirmed positives are keepable when they
minimize a stall class without regression — record the bar decision
explicitly.

#### r32 — finite-lever sweep: both regions bounded (row 46)

**Method.** SASS regional census of the r34-era NB kernel (staging ~21%,
epilogue ~0.4%): the staging's kt-independent addressing is already hoisted
by ptxas (A-token base in the prolog; per-kt term in a uniform register) —
the staging lever is dead at source level. The epilogue (32 scalar STG.E,
ptxas does not vectorize) got a float2 interior + scalar tail.
**Result.** SASS: 32 STG.E → 16 STG.E.64 + 24 — but static int ALU INCREASED
(dual-path guard work); +0.46% wall = noise, and structurally capped (~0.4%
of instructions). Reverted cmp-verified.
**Lesson.** Run-once regions cannot clear any bar; widening stores at the
cost of a branch can cost more than it saves.

#### r33 — hybrid inner-loop port: SASS-identical, falsified (row 47)

**Method.** Hypothesis: "the remaining 1.15×/GMAC gap is pure SASS codegen
from loop organization" — port llama's j0-outer/n-inner nesting (weight
B-fragment reused across all 4 token-minitiles; k01 degenerate here so every
numeric point unchanged; fragment maps validated 0 mismatches).
**Result.** Parity 1/0; greedy byte-identical; perf median −0.25% — the SASS
is BYTE-IDENTICAL to r31 (64 IMMA same order, 109 regs, same LDS counts):
ptxas already reschedules the mma/rescale optimally regardless of source
order. Census: int ALU 1.66 e-3/MAC (llama 0.751) — did NOT converge.
Scope caveat: the deeper A=weight transposing port was NOT reached — that
would change instruction composition (r34's subject).
**Lesson.** Loop-organization hypotheses are testable by SASS identity
before perf; a byte-identical SASS is a falsified hypothesis by definition.

#### r34 — quantize-transpose prepass LANDED (row 48)

**Method.** Reframe the residual as layout-transformation locality: llama
quantizes activations PRE-TRANSPOSED into the exact mma-consumed layout, so
its A staging is a near-bulk copy; minfer re-staged the A tile ~28× per
buffer (once per od-tile column), each re-stage paying the swizzle/repack
index math (r32: staging = 21% of kernel instructions). New
`quantize_q8_0_pad40_t` (bit-identical values, transposed layout, zero-filled
pad tokens) + `mmq_raw_nb_bt_kernel` whose A staging is a bulk LDG→STS
(uint4 copies, no per-element index math); router gated
`MINFER_MMQ_A_TRANSPOSE=1`; NB kernel SASS byte-identical between binaries
(A/B integrity).
**Result.** Byte-exactness validator: 0 mismatches across 9 shapes (incl.
non-64-multiple nt). 103 regs/0 spill (6 fewer — staging index math gone);
smem unchanged → 2 blocks/SM. Prepass 0.405 vs 0.446 ms (0.908× — NOT
bigger). Perf: 1364.2 → 1496.8 (+9.72%, every pair positive, no overlap) —
the largest single-mechanism P6 gain since r28. ncu census unobtainable this
session (platform injection failure — mechanism confirmed by regs/prepass/
wall instead). Suite 166/0/3; greedy identity.
**Lesson.** Hoist layout transformation out of the inner loop entirely (the
r6 spec's principle, applied to the A side); "ncu unavailable" does not
block a landing when three independent mechanisms corroborate.

#### r35 — scale pre-decode REVERTED (row 49)

**Method.** Regional census of the BT kernel (compute-kd 66% of instructions;
largest cuttable block = sda decode: 64 SHF sign-extends + 64 HADD2 + 64
I2FP). Pre-decode in the prepass: d as f32 + ssum as i32 (8 B/token/chunk vs
4), smem 43,008 → 45,056 B (still 2 blocks/SM).
**Result.** SASS: SHF 64 → 0, HADD2 64 → 10, LDS.128 32 → 48, net −130
instructions. Perf −0.46% (noise). The decode instructions were scheduled in
the IMMA shadow (filling idle FP/INT slots under the 64/kt tensor mma) —
removing them frees no critical resource; the added LDS offsets it.
**Lesson.** Instructions that hide in tensor-core shadow are free — cut only
what competes with the mma pipe.

#### r36 — A-frag wavefront economics: H1 refuted (row 50)

**Method.** Test the last named candidate (replace A-frag LDSM with plain
LDS) under its own metric — shared wavefronts per IMMA (ncu injection
unblocked via the sudo env prefix). Measured: LDSM wavefronts 4× llama
(2.000 vs 0.500/IMMA), total shared wavefronts 1.76×.
**Result.** But minfer performs 1.85× the wavefronts/second (39.0 vs 21.1
G/s) while landing at equal-or-better tensor throughput (6.33 vs 6.02
G-IMMA/s, issue 0.457 vs 0.365) — if MIO were scarce, both kernels would cap
at the same wavefronts/s. The causal claim fails. Why the fix cannot help:
LDSM.x4 moves 512 B = 4 wavefronts; a conflict-free plain LDS of the same
payload moves the same 4 — swapping access method is wavefront-neutral unless
the geometry changes. llama's real edge is A-fragment REUSE (0.125
LDSM/IMMA vs our 0.5 — it iterates od inside the kernel), a tiling
restructure, not an LDS swap. No code change (HEAD stays).
**Lesson.** Throughput accounting (work/sec vs work/IMMA) distinguishes
"more of a resource" from "bound by that resource"; H1-class swap levers die
on byte-equality.

#### r37 — post-parity whole-prefill attribution (row 51)

**Method.** Full-graph nsys + matched-nt ncu on the BT-enabled path
(3325-token prefill; GPU busy 2139.6 ms, wall 2190 ms = 1521 tok/s; co-tenant
idle @0%).
**Result.** The residual is a DIFFERENT kernel: bt covers q4_K only; q6_K
(attn_v + ffn_down) still runs the generic `mmq_nt<7,2>` dequant-staging
path — 1094.7 ms = 51.2% of the wall at 368.9 µs/GMAC vs llama 57.8 (6.38×);
the q6_K ffn_down GEMM alone (1063.5 ms) is larger than the entire q4_K BT
GEMM (600 ms). q4_K BT: 60.0 TFLOPs (vs-llama 1.15× at matched nt; 1.43×
per-IMMA at nt≈511 = short-nt prologue dilution). Quantize prepass 86.8 ms
(1.25× llama, partly 2× per-shared-A redundancy). FA 124.7 ms (5.7×).
Whole-prefill vs llama @3325-eq: 2.15×. Priority queue: (1) q6_K on a raw
byte-width kernel (up to ~2720 tok/s); (2) q4_K short-nt dilution; (3) FA
structure.
**Lesson.** Optimizing one weight-type line exposes the next one — attribute
the WHOLE wall after every convergence before picking the next lever.
**Campaign state at r37: 1521 tok/s, 2.15× vs-llama.**

### Era D — q6_K, FA, prepass, and promotion (2026-09-05 → 09-06, r38–r60)

#### r38 — q6_K BT-style raw-byte mma kernel LANDED (row 52)

**Method.** r37's #1 lever. A layout correction came first: the working model
"q6_K = 8 sub-blocks of 32" is WRONG — `block_q6_K` is `ql[128] + qh[64] +
sc[16] + d[2]` = **16 sub-blocks of 16 elements**, so a 32-k chunk spans two
differently-scaled sub-blocks → KSPLIT=2 (two m16n8k16, one per 16-sub, each
with its own dsc; single-term rescale `sum += da·dsc`, the dmin term drops —
confirmed against the CPU reference; accumulation as TWO separate `+=`, the
fused form rounded 1.2e-3, just over the 1e-3 bar). New
`mmq_raw_nb_bt_q6k_kernel` (gated `MINFER_MMQ_Q6K_NB=1`) reuses the r34 BT
shell (A side identical); B staging expands each super-block to centered
int8 so the ql+qh recomb leaves the hot loop (element→super-block map
validated standalone, 0 mismatches).
**Result.** The occupancy lever was decisive: the full-super-block KDR=8
variant (59,392 B, 1 block/SM) REGRESSED the prefill to 1097.8 tok/s; KDR=4
(29,696 B → 2 blocks/SM) recovered and beat baseline: 1518.4 → 1561.9
(+2.87%, 3/3). Matched-nt q6_K: 368.9 → 221.8 µs/GMAC (1.66×) — below the
≥2× bar, landed as strictly positive (the baseline path was terrible). 85
regs/0 spill; parity 1/0 (raw 210B + padded 224B); greedy-32 byte-identical;
attn_v still latency-bound (compute 16.7%).
**Lesson.** Verify the quant layout against the CPU reference before
designing the mma structure; depth-vs-occupancy (r5's lesson) re-appears on
every new kernel.

#### r39 — q6_K KDR=2 double-buffer LANDED (row 53)

**Method.** r38 left the kernel latency-bound with the B expansion serialized
behind a per-kt single-buffer barrier. Pipeline it: stage kt+1's expansion
into a second buffer while kt computes (the `mmq_nt<7,2>` scheme). The smem
arithmetic comes out for free: doubling every plane at KDR=4 = 59,392 B = the
1-block/SM trap; KDR=2 gives 29,696 B — the r38 footprint at 2 blocks/SM —
while pipelining BOTH A (bulk uint4) and B (ALU expansion). A double-B-only
KDR=4 variant was rejected on correctness (A single-buffered → clobber).
**Result.** 87 regs/0 spill; parity 1/0; greedy byte-identical; 1568.7 →
1777.5 (+13.3%); attn_v kernel −19.7% (2,549,248 → 2,046,848 ns), compute
21.5%; suite 166/0/3.
**Lesson.** Pipeline beats occupancy when occupancy is already bought: the
gain is pure overlap (r20's split-phase lesson at the staging level), and the
smem budget must price BOTH planes.

#### r40 — 3rd resident block via `__launch_bounds__(256,3)` LANDED (row 54)

**Method.** Register arithmetic: 3 blocks needs ≤80 regs/ptxas granularity;
smem already permits 3. The one-line hint force-fits 87 → 80 with exactly one
4 B spill (10 manual trim variants measured, none beat the 4 B floor — ptxas
needs 81 live registers at this revision).
**Result.** The "spills are a loss on a latency-bound kernel" premise is
EMPIRICALLY FALSIFIED: 3 blocks/SM confirmed (18.12 warps/SM, 37.74%), 1784.0
→ 2015.6 (+13.0%, 3/3 + an independent pair); kernel −23%, compute 21.5 →
29.06%, No-Eligible 74.5 → 70.1%; parity/greedy green; suite 166/0/3.
**Lesson.** The 0-spill gate is a heuristic, not a law — a +50% resident-warp
occupancy win dominates a one-register spill. Measure the spill's cost
before paying register pressure to avoid it.

#### r41 — q6_K B-expand uint4 widen LANDED (row 55)

**Method.** Warp-state attribution: CPIStall = 85.5% long_scoreboard (13.7 of
16.0 cy/inst) — the B-expand issues 32 per-byte `LDG.E.U8` (ql/qh) per
thread per kt and consumes them immediately in the recomb→STS, exposing full
L1TEX latency serially in front of compute (SASS confirms staging sits atop
the loop). The padded 224-byte stride is 16-aligned, so each 16-element
group loads in ONE uint4 each (ql + qh) — a ~16× cut in B-expand load
instructions; the recomb (nibble + 2-bit field − 32) moves to registers via
a closed form validated element-for-element (512,000 elements, 0 mismatches).
Raw 210-B (test-only) keeps the scalar path via a `(bstride & 15) == 0` gate.
**Result.** 80 regs/4 B spill held (3-block budget intact). 1979.9 → 2605.2
(+30.7%); kernel 1.70 → 0.654 ms (−61.5%); L1TEX scoreboard 85.5% → 33.6%
(13.7 → 3.6 cy); compute 27 → 37.8%; parity/greedy green; suite 166/0/3.
**Lesson.** The largest single q6_K lever was load WIDTH, not scheduling:
byte-granular global loads in a staging loop are the classic L1TEX
scoreboard generator.

#### r42 — stage-wide dsc scale read REVERTED (row 56)

**Method.** r41 named the residual as the dsc `d·sc` reads. Widen them: 2
wide loads per (row, window) instead of 6 narrow; same multiply order →
bit-identical dsc; gated on the 16-aligned padded stride.
**Result.** L1TEX throughput 33.64 → 26.45%, kernel −1.8% — but
long_scoreboard UNCHANGED (3.6 → 3.8 cy, 33.6% both) and wall −0.19%
(noise). The r41 premise is FALSIFIED for the dsc path.
**Lesson.** Cutting bytes does not cut latency: the stall is at the
CONSUMER (the I2F.S8 waiting on the dsc round-trip) — profile to the
dependent op, not the load.

#### r43 — PC-sampling attribution; pre-expand-B parity FAIL (row 57)

**Method.** ncu warp-stall sampling (`pcsamp_warps_issue_stalled_long_scoreboard`,
`--page source`) distributes stalls to the CONSUMING instruction. Task 2: hoist
the ql+qh recomb to registration as a `W_exp` centered-int8 plane; kernel B
staging becomes a bulk copy.
**Result.** Attribution: B-expand recomb (LOP3/SHF) 45% + A-staging STS.128
28% + dsc I2F.S8 consumer 26% = the 33.6% share. W_exp is byte-CORRECT
(0/17,920 readback) but parity FAILS (diff 448 @ index 554, identical across
three kernel variants — including in-kernel expand of W passing while
in-kernel copy of W_exp fails). Could not isolate within budget → REVERTED
(cmp-match HEAD).
**Lesson.** Byte-correct data at the wrong address expression is worse than
obviously-wrong data — the readback validates CONTENT, not OFFSETS (r44
resolves it).

#### r44 — W_exp stride mismatch root-caused; fix wall-neutral (row 58)

**Method/Result.** Root cause (one line): the r43 fix indexed the DENSE
`W_exp` (row stride `id`, super-block stride 256) with the PADDED raw-W
expression (row stride `nsb·bstride`, super-block stride `bstride`) — r43's
three variants split exactly as that predicts. The correct dense index is
parity-green (80 regs, 20/28 B spill; 3-block budget holds) but the WALL is
neutral: whole-prefill 2595.9 → 2584.9 (−0.42%). Kernel elapsed cycles −10.9%
(the recomb ALU and its 45% of samples are gone) but the stall only
TRANSFORMS: long_scoreboard share rises to 57.1% (denominator effect) —
removing the recomb converts load→ALU-recomb into load→STS copy, the same
L1TEX exposure.
**Lesson.** After r41, the q6_K GEMM is no longer the wall bottleneck —
kernel-level wins in a converged line do not reach the wall. Same physical
latency hides under different instruction mixes.

#### r45 — cp.async the q6_K A-side staging REVERTED (row 59)

**Method.** r44's named physical lever: the A-side bulk LDG→STS becomes
explicit-PTX `cp.async.cg.shared.global` (`__pipeline_memcpy_async` fell
back to LDG+STS — the compiler could not prove 16B alignment on a generic
pointer; SASS gotcha: CP.ASYNC is emitted as `LDGSTS.E.BYPASS.128`, grep
LDGSTS). Group-count pipeline: `wait1` in-loop, `wait0` on the last tile, +
one visibility `__syncthreads`; the end-of-loop WAR barrier kept.
**Result.** REG 80/LOCAL 0 (the r41 4 B spill dropped to 0), LDGSTS in SASS,
parity/greedy green — kernel −10.2% duration, longsb −18%, compute 37.4 →
42.1% … wall −0.34% (noise). REVERTED; q6_K line CONVERGED (the kernel is no
longer the binding constraint; further q6_K-kernel tuning cannot move the
wall).
**Lesson.** A mechanism can be correct, confirmed, and worthless — wall
value depends on what ELSE is on the critical path. "Not dead, WAITING"
(r53/r56 prove the second half).

#### r46 (FAP1) — FA audit + occupancy/conflict lever REVERTED (row 60)

**Method.** Audit: `fa_prefill_f16kv` is ALREADY wmma m16n16k16 + online
softmax (not scalar) — but occupancy-starved: 69.38 KB smem → 1 block/SM
(16.64% occ) + a bank-conflicted S/P smem round-trip (row stride ≡ 0 mod 32
banks). Lever: FA_TKV 64→32 (~43.8 KB → 2 blocks/SM) + fixed a latent
launcher smem over-allocation (`3*FA_TQ` was only right when
FA_TQ==FA_TKV) + S/P row padding (+8 f32).
**Result.** Kernel 5.16 → 4.58 ms (−11%), 2 blocks/SM, SM busy 40% — whole-
prefill +0.27% (below the +1.5% bar): FA is not the wall-critical path in
the converged-GEMM regime (the 124.7 ms slice −16 ms is noise-level).
Reverted; FAP2 named: register-resident softmax eliminating the S/P
round-trip entirely.
**Lesson.** Positive-mechanism/negative-wall is a real outcome class —
record the mechanism and re-rank the target; also re-verify with a fresh
nsys whether a slice is genuinely wall-relevant before investing (r47 does).

#### r47 — converged-regime wall decomposition (row 61)

**Method.** The r37 table is stale exactly as predicted: re-decompose at the
current best gate set (3325-token prefill; GPU busy 1239.2 ms; no code
change).
**Result.** q6_K GEMM 1094.7 → 196.4 ms (51.2 → 15.8%); wall 2190 → 1274 ms
(1521 → 2610 tok/s); vs-llama 2.15× → 1.27×. q4_K 598.1 ms = 33.2 µs/GMAC
(1.06× — parity); q6_K 65.0 µs/GMAC (1.13× — converged); FA 125.8 ms = 5.72×
= the #1 structural residual and genuinely wall-relevant (r46's "smaller/
overlapped" caveat does NOT hold); quantize prepass grew +31.6 ms (the q6_K
BT port routes it through the same prepass — the hidden tax; net q6_K wall
win still +866.7 ms). Matched-nt: q4_K dilution (1.17× at nt≈511 vs 1.06× at
prefill nt) is a short-nt-only effect — shelved. Recommendation: FAP2
(estimated 2× → −63 ms = −4.9% wall); A-quantize shared-A dedup next.
**Lesson.** Wall decompositions have shelf lives — re-run after every
convergence; the hidden taxes of a landing (prepass growth) must be netted
against its wins.

#### r48 (FAP2) — register-resident softmax LANDED (row 62)

**Method.** S and P no longer go through shared memory: 4-warp (128 thr)
full-row warp-tile; QK^T wmma writes S to accumulator FRAGMENTS; online
softmax runs on the fragments (`__shfl_xor` over the 4-lane row group); P is
built in-place as the P@V `matrix_a` fragment (f32 accumulator and f16
row_major share the same lane→(row,col) map — unit-validated); P·V runs wmma
with V as row_major B while QK^T keeps K as col_major B (the two operands
have OPPOSITE major-ness — using row_major for both corrupts both; the FA
parity test's max err went 1e-4 → 0.54 until K was re-set). Removes 1 of 4
`__syncthreads`/tile, the S/P conflicts, the 36% MIO stall, and Sf/Pf smem
(69.38 → 34.82 KB; 1 → 2 blocks/SM). Launcher: `(FA_TQ + 2*FA_TKV)*(hd+8)*2`;
threads 256 → 128 (the staging stride had to be parameterized — otherwise
half the K/V tile went unstaged).
**Result.** FA kernel 5.16 → 2.12 ms (2.43×; top stall now global K/V
loads); whole-prefill 2603.5 → 2749.9 (+5.6%, the r47-predicted −5% wall);
all parity tests green; greedy-32 byte-identical; suite 166/0/3; FA ~10.2 →
~4.7% of wall.
**Lesson.** The first campaign-level removal of a structural >2× residual:
eliminating a smem round-trip beats optimizing around it. wmma operand
major-ness differs per stage — validate lane maps standalone before
integration.

#### r49 — A-quantize prepass shared-A dedup LANDED (row 63)

**Method.** q/k/v consume the SAME `normed` A and gate/up the same
`normed2`, but the MMQ path re-ran the prepass per matmul (193 launches,
~2× redundant). A per-execution cache on `CudaState` keyed (src device ptr,
nt, id) → quantized buffers, valid only while the same src arrives on
CONSECUTIVE MatMul nodes; cleared by any non-MatMul node and at split
boundaries; dedicated scratch OUTSIDE the graph allocator pool; byte-identical
by construction (quantize is a pure function of (x, nt, id)). HARD RULE: the
dedup lives entirely in the CUDA layer — `graph.rs` untouched.
**Result.** Prepass 193 → 110 launches (118.4 → 83.9 ms, −34.5 ms); GEMM
launches unchanged at 193 (the dedup removes only redundant prepasses);
2734.1 → 2797.5 (+2.32%); parity ×3 green (the parity harness clears the
cache between cases — the HIT path is validated by greedy-32 identity
instead); suite 166/0/3.
**Lesson.** Redundant work on shared inputs is a scheduling-window property —
memoize against the node stream with conservative invalidation, and validate
the hit path through an end-to-end identity check when the unit harness
cannot reach it.

#### r50 — FA_TKV 32→16 REVERTED (row 64)

**Method.** Single `#define FA_TKV 32→16` (everything else symbolic); 3
blocks/SM derived from the r48 device config + the new 32 KB request. One
latent non-symbolic constant surfaced: the tail-tile O writeout reuses smem
as a 64×128 f32 buffer (32 KB) — the launcher must take
`max((FA_TQ+2*FA_TKV)*(hd+8)*2, FA_TQ*hd*4)`.
**Result.** Correct (parity green, suite 166/0/3) but wall-neutral (−0.5% /
−0.01%) — the occupancy gain is cancelled by doubled per-tile sync/softmax
overhead (the r46 lesson). Greedy-32 NOT byte-identical (one argmax boundary
flips on ULP accumulation-order noise) — INHERENT to any FA_TKV reduction.
**Lesson.** Two bounds in one round: FA tile-size reduction is a dead lever
(r46 + r50), AND the strict greedy-byte-identity gate is only satisfiable
for FA changes that preserve accumulation order — a NEW caveat for the FA
campaign (r57 hits it again).

#### r51 — producer-fused A-quantize, mode 1 LANDED (row 65)

**Method.** Every rms_norm/swiglu output in the prefill graph is EXCLUSIVELY
a GEMM input — fuse the quantize INTO the producers: `rms_norm_quant_f32_t`
(rms lane mapping unchanged → f32 bit-identical; one `__syncthreads`; then
the `quantize_q8_0_pad40_t` body VERBATIM re-reading y through L1/L2, grid
covers the 64-padded count so tails zero-fill identically) and
`swiglu_quant_f32_t` (block per token row; coalesced float4 silu·mul; the
register-resident variant was REJECTED by design — uncoalesced f32 stores =
8× sector amplification on 254 MB). Host: same scratch sizing, plane
registered in the r49 cache keyed on the f32 output pointer; gate
`MINFER_MMQ_A_FUSE=1` ANDed with the full gate set + rows ≥ 16 + dim % 256
== 0; OOM falls back to the unfused pair.
**Result.** Standalone validator: BYTE-EXACT on 11 shapes with POISONED plane
buffers (proves tail zero-fill). Prepass 110 → 28 launches (only wo remains
— its producer is the FA kernel, deliberately untouched in v1), 83.0 →
10.1 ms; fused swiglu −25%/launch, fused rms a wash at d=3584. Perf
2803.4 → 2856.4 (+1.89%, distributions fully separated). Greedy byte-identical;
suite 166/0/3.
**Lesson.** The r34 lesson generalized to its end: consume producer output
while it is L2-hot. The win concentrates where the producer is large (swiglu
→ down), not uniform.

#### r52 — skip-write mode 2 (`MINFER_MMQ_A_FUSE=2`) LANDED (row 66)

**Method.** The fused kernels still pay the f32 write + L1/L2 re-read
(swiglu: 251 MB). Mode 2 computes the pad40_t plane REGISTER-RESIDENTLY and
never writes the f32 output (`*_nw` kernels; cache keyed on the unwritten
pointer exactly as r51). Window-safety proof per node: full gate set (the
consumers then always reach transposed-plane paths) ∧ rows ≥ 16 ∧ dim % 256
== 0 ∧ `!no_prefill_gemm()` ∧ no debug/trace reader (MINFER_GRAPH_DUMP /
MINFER_DUMP_DIR / MINFER_TRACE / viz all degrade mode 2 → mode 1);
topology-verified (rms/swiglu outputs' only consumers are immediately-
consecutive plain MatMuls; residual adds read the PRE-norm buffer; G3 tail
runs n_out=1 rows; FusedQkv/FusedFFN are decode-only; RmsNorm/SwiGLU are not
in-place). Backstop: `MmqCache::dead_write` + a `q8 == 0` guard — any future
window violation errors loudly instead of reading garbage.
**Result.** Two bugs caught by gates before landing: (a) a transcription
error (chunk index `k*8 + lane/8`) wrote past the plane → IllegalAddress
surfacing as cascading fake "OOM" — caught by greedy-32 (NOT by parity, whose
fixtures never exercise rms-nw); (b) the first swiglu-nw mapping made gate/up
loads lane-strided (fused swiglu 110.5 → 114.0 ms, whole-prefill only
+1.39%) — discarded for the coalesced-round form. Landed: fused rms 41.5 →
22.4 ms, fused swiglu 110.5 → 64.1 ms (−42%), fused producers 151.9 →
86.5 ms; prefill window −6.1%; 2855.7 → 3011.3 (+5.45%, distributions fully
separated); parity ×3, greedy identical across A_FUSE=1/2; suite 166/0/3.
(The standalone validator SIGBUSed in this env — kernel-level identity
covered by greedy-32.)
**Lesson.** Skip-write is only as safe as its consumer enumeration — prove
the window per node, degrade under readers, and backstop with loud refusals.
Greedy identity caught a corruption parity could not see.

#### r53 — q6_K bundle: W_exp + cp.async B staging LANDED (row 67)

**Method.** r44 (pre-expanded dense W_exp — removes WORK) and r45 (cp.async
staging — removes WAIT) were each parity-green and individually WALL-NEUTRAL.
Bundle: B staging becomes a pure cp.async bulk copy from the pre-expanded
plane — no recomb ALU, no register round-trip, no ql/qh reads, latency
handed to the async unit. Kernel templated `<KDR, EXP>` (EXP=false compiles
the cp.async branch away — byte-identical r41 fallback); dense index
`W_exp + j*id + sb*256 + cbase*32 + cc*16` (the r44 one-line root cause
respected); `gemm_cp16` explicit PTX with the src-size qualifier zero-filling
rows beyond od; r45's group-count pipeline + visibility barrier. Host:
`expand_q6k_dense` + `register_weight_q6k_exp` under the existing
`MINFER_MMQ_Q6K_NB=1` gate (+ `id % 256 == 0`); map miss → fallback with a
once-per-process eprintln. Memory measured exactly: W_exp =
od × id bytes per padded q6_K tensor — 14 attn_v + 14 ffn_down + output.weight
= 1,521,237,632 B ≈ 1.52 GB (the task's "+15 MB" estimate treated one
ffn_down's delta as the whole cost; this also explains the 27-launch q6_K
count).
**Result.** W_exp byte-exactness cargo test (independent scalar mirror vs
host expander AND device plane read back): 0 mismatches. `<2,true>` 80
regs/24 B stack (3-block budget holds); SASS LDGSTS ×12 (the explicit-PTX
trap avoided). Parity ×3, greedy identical (178-char streams). Perf 3024.7 →
3176.9 (+5.03%, distributions separated, base max < new min); ffn_down kernel
−20.5% (16.06 → 12.76 ms — r44's −10.9% + r45's −10.2% compose almost
additively); attn_v −15.9%. **Liveness catch**: the first integrated build
passed parity AND greedy with ZERO fast-path launches (the map insert keyed
the entry by the exp buffer's own pointer) — the extended
`MINFER_MMQ_RAW_NB_DEBUG` label counted 27 W_exp-cp.async after the one-line
key fix. Suite 167/0/3 (+1 new gate test).
**Lesson.** The basket thesis: mechanisms that overlap in traffic but not in
mechanism compose. And the round's namesake rule: **a fallback-correct
optimization needs an "is the fast path actually live" check — parity/greedy
cannot see a silently-never-taken fast path.**

#### r54 — `MINFER_MMQ_Q6K_EXP` opt-out LANDED (row 68)

**Method.** Independent switch (user-approved): unset/"1" = r53 behavior;
"0" = registration early-returns before the sibling build → dispatch
map-miss → the `<KDR, EXP=false>` r41 instantiation (byte-identical,
compiled in since r53). ANDed with `MINFER_MMQ_Q6K_NB=1`; default-on
(`!= Ok("0")`). The B-path label becomes three-way so an INTENTIONAL fallback
(`exp=off`) is distinguishable from an ACCIDENTAL one (`fallback!`).
**Result.** Parity ×3 under BOTH modes; greedy identical exp1 vs exp0 AND vs
the r53 record; liveness census 27×/0 both ways; perf exp1 3181.0 (unchanged)
vs exp0 3020.7 = **−5.04% for 1.52 GB back**; memory measured per-PID via
`nvidia-smi --query-compute-apps` (GB10's aggregate is `[N/A]`): 7636 vs
6182 MiB = Δ1454 MiB ≈ the census. Suite 167/0/3 (one 165/2 under a 46 GB
co-tenant — both pass `--exact` in isolation; co-tenant flake, unrelated).
**Lesson.** Memory-for-speed knobs need three things: a compiled-in
byte-identical fallback, a liveness label that distinguishes intentional from
accidental fallback, and measured (not assumed) memory deltas.

#### r55 — swiglu roofline + prefill CUDA-Graph: both documented skips (row 69)

**Method/Result.** Two low-risk levers closed BY MEASUREMENT before any code
(tree unchanged; baseline cmp-verified). (1) The swiglu proposal's traffic
estimate was 2× low: ncu proves the reads are f32 (502.2 MB = exactly
2 × nt × dim × 4 B, sectors/request = 16 → already maximally vectorized);
minimal DRAM traffic 573.1 MB in 2.367 ms = **242 GB/s ≈ 89% of the 273 GB/s
spec roofline** (94.6% occupancy; stall mix a latency-limited pure stream).
Decisive bound: 573.1 MB / 273 GB/s = 2.099 ms ideal → max saving 7.2 ms =
**+0.74% whole-prefill — below the bar even for a perfect kernel**. (2)
Prefill capture is already live for the repeat case (8g②/R3-B); the one-shot
case: idle 7.84 ms = two one-time ~3 ms host stalls (minfer's own host code,
not inside CUDA APIs) + a 0.78 ms mid-window `cudaMalloc` (capture-ILLEGAL)
+ recurring gaps ~0.1% under nsys instrumentation. Projected win ≤ 0.1% —
documented skip. Residual leads recorded: rms_nw at 153 GB/s = 56% roofline
(ideal +0.98%); the host stalls deserve a root-cause pass; tail pre-grow
+0.1%.
**Campaign convergence statement:** whole-prefill 2.15× (r37) → **3181 vs
3324.4 = 1.05×** (r53/r54); wall = q4_K 63.2% (closed absent the q8_1
prologue), q6_K 15.4% (closed), fused producers 8.8% (swiglu closed, rms
last lead), FA 5.3% (2.43× taken), host ~1%. **No identified lever ≥ +1.5%
remains; the next step is the step-function q8_1 GEMM-prologue fusion.
Campaign verdict: CONVERGED** (at this gate set — r56/r59 below re-open it).
**Lesson.** Roofline-bound-before-coding: a byte-count-derived bound (GB10
has no `dram__*` counters) killed both levers for the cost of one ncu run;
one-time host stalls and capture legality are quantified, not guessed.

#### r56 — q6_K A-side bundle: A cp.async + W_dsc plane LANDED (row 70)

**Method.** The §11.32 verdict named the two remaining q6_K residuals: "A-side
staging STS (~28%) + the dsc I2F consumer (~26%) — a W_dsc f32 plane would be
the symmetric next bundle member, and an A-side cp.async redo could now
compose". Both landed: (a) A-side qa8/sda copies become explicit-PTX
`gemm_cp16` in the SAME per-kt commit group as the r53 B copy (commit moved
to RAW_STAGE end; waits unconditional); (b) `W_dsc` path — chunk-major
`float2(d·sc[2c], d·sc[2c+1])` plane streamed as one contiguous 16 B
cp.async per pair (od even is a registration gate), null → r41 scalar path.
Host: `q6k_dsc` map (the r53 pattern incl. geometry-encoded sibling name +
failure eprintln); `expand_q6k_dsc` bit-identical by construction (exact
f16→f32, exact i8→f32, ONE f32 multiply, no FMA contraction either side);
registered under Q6K_NB + Q6K_EXP ≠ "0" + id%256==0 (+ od%2==0) so EXP=0
still returns ALL plane memory. Memory: W_dsc = od × id / 4 B = **363.2 MB**.
**Result.** `cuda_q6k_dsc_dense_byte_exact` 0 mismatches (3 shapes);
`<2,true>` AND `<2,false>` both 80 regs / 0 stack (r53's 24 B stack gone)
with LDGSTS in BOTH instantiations; parity ×3; greedy identical (453-char
streams); liveness 27× `A=cp.async DSC=f32-plane, B=W_exp-cp.async`, 0
fallback. ncu: ffn_down 12.76 → 12.01 ms (−5.9%), attn_v −4.2%. Perf
3138.6 → 3212.5 (+2.35%, distributions separated excl. one co-tenant
outlier). Suite 167/1 — the 1 (`cuda_conversation_multiturn_reuse`) fails
IDENTICALLY on clean HEAD (bisect-verified via `git stash`): pre-existing,
environment-sensitive, not this change.
**Lesson.** r45's mechanism finally reaches the wall exactly as the bundle
thesis predicted — with B a pure copy, the A-side wait and dsc consumer
became the wall, and removing both moved it. The sudo-env-strip gotcha is
worth its own line: plain `sudo -n` strips gate env and silently profiles
the LEGACY path (ncu's "Available Kernels" list on a zero-match filter is
the tell).

#### r57 — FA KV staging double-buffer REVERTED (row 71)

**Method.** Session-E basket item 1: smem arithmetic picked FA_TQ=48 +
double-buffered K/V (47,872 B → 2 blocks/SM; FA_TKV stays 32 so per-row
accumulation order was EXPECTED unchanged). Prologue-staged tile 0, per-tile
issue of kt+FA_TKV into buf^1 with wait_group 1; warp 3 staging-only.
**Result.** Attempt 1 diverged at greedy-32 token 19; the fix found a REAL
bug (the tail-tile stage→global O copy was inside the warp guard) — attempt 2
STILL diverged: the residual is the r50-class inherent rounding shift —
FA_TQ is a tile size too; the premise "the r50 caveat does not apply" was
falsified. REVERTED per the two-attempt stop rule (rebuild md5 differs —
rebuilds are not bit-deterministic — but the greedy stream is identical and
perf sanity 3222.4 ≈ the landed median). Items 3/4/5 (rms_nw roofline, host
stalls, tail pre-grow) not reached — out of budget.
**Lesson.** The r50 caveat generalizes: ANY FA tile-size change (TKV or TQ)
breaks byte-identity. The double-buffer mechanism stays viable only for a
future FA change that accepts non-identical output.

#### r58 — q4_K BT spec + cp.async-db2 transplant REVERTED (row 72)

**Method.** Baseline re-measured first (3219.6 median, consistent with the
landed 3212.5). Phase 1a — the structural diff measured: kernel busy 984.2
ms, q4_K bt 622.4 ms (63.2%) across 166 launches split into four classes —
gate/up 7.51 G-IMMA/s, q/o 7.46, ffn_down 5.94 (21% below our own steady
state), k/v 5.05 (27.8% ceil-wave loss); matched-nt ncu: ours 7.5 vs llama
~6–6.5 G/s on the big classes — the deficit is NOT in the mma loop. Where
the 1.06× lives: ffn_down per-kt staging exposure, ceil-wave quantization
(26.1 ms ≈ 2.6% of whole-prefill), launch gaps closed. Top delta: transplant
the r39+r53+r56 staging pipeline onto the q4_K bt kernel as KDR=2
double-buffer + cp.async A/sds + sb-parity cp.async B window (46,080 B,
122 regs, LDGSTS in SASS).
**Result.** Three bugs caught by gates en route (a `uint32_t*` vs byte
pointer stepping 4× the sda plane; the same word/byte confusion on the
compute side — the r44-class stride mismatch, caught by greedy-32; the B
window must rotate at SUPER-BLOCK parity, not kt parity — buffer 1 never
written, the nt=13 dump looked clean by stale-node luck). After the fixes:
greedy-32 IDENTICAL, parity ×3 green — perf 2819.3 vs 3227.6 = **−12.6%** →
REVERTED. Why the q6_K winner loses here: its cost model inverts — q6_K
replaced an EXPENSIVE staging (r41 ql+qh recomb + I2F) and KDR=2 amortized
it; the q4_K bt staging is already cheap pure copies, so KDR=2 buys 4× the
barrier density, a 1-deep lookahead that cannot cover ~600–900 cyc latency,
and removes no register/ALU work.
**Lesson.** The r45 lesson's mirror image: **a mechanism whose COST depends
on the granularity of what it replaces is not free, it is AMORTIZATION-BOUND.**
Phase-2 spec ranked: (1) q4_K W_dsc plane (the r56 scaffold on the other
63%), (2) wave re-tile (+0.3–0.8%, byte-identical), (3) fused ffn_gu concat,
(4) riders.

#### r59 — q4_K W_dsc plane + riders LANDED (Δ corrected by r59b) (row 73)

**Method.** r58's item 1 implemented exactly: `mmq_raw_nb_bt_kernel` templated
`<KDR, DSC>`; DSC=true replaces the per-(chunk, od-row) rank-1 decode
(`get_scale_min_k4` + 2 h2f + 2 multiplies × 256 per kt per block — branchy,
warp-divergent) with a contiguous 16 B `gemm_cp16` stream from the chunk-
major plane `float2(d·sc, −dmin·m)` (u8 6-bit scales — NOT i8 as in q6_K;
one IEEE f32 multiply each → mma-side rescale bit-identical). Plus the r57
riders: `minfer_prewarm_kernels()` (cudaFuncGetAttributes pass), pinned
readback pre-grow, MmqCache scratch pre-sized for a nominal 4096-token
prefill. Host: `q4k_dsc` map (`{name}__q4dsc{od}x{id}` siblings),
`expand_q4k_dsc` (exact half→f32, no FMA contraction), registration gated
RAW_NB + A_TRANSPOSE + `MINFER_MMQ_Q4K_DSC != "0"` + id%256==0 + od%2==0.
Memory measured: **+1456 MB** (the r58 spec's ~1.07 GB estimate used the
wrong q4_K byte mass — 5.8 GB real, not 4.29 GB).
**Result (as recorded, co-tenant window).** Interleaved 5× ×2 series: base
2836.3/2843.2 → new 3574.7/3588.8 = **+26.1/+26.2%**; a −12% "co-tenant tax"
was inferred from the base being below the 3219.6 record. ncu matched-nt:
ffn_down-q4_K −34.6% (4.30 → 6.58 G-IMMA/s), gate/up −35%, regs 124 → 105.
nsys census: q4_K bt busy 762.59 → 526.82 ms = **−30.9%** — but ffn_down
only −5.2%: **the r58 premise was half wrong** — the win came from gate/up
(−37%) and q/o (−18%) where decode ALU/I2F share was large; ffn_down's
deficit was L2-reuse-shaped (61.6 MB qa8 plane per launch), not
decode-shaped. Item 3 (od re-tile) SKIPPED with evidence (needs ≤85 regs;
DSC=true has 105 → ~+0.2%, below the co-tenant noise floor). Gates: byte-
exactness 0 mismatches; parity ×3 + Q4K_DSC=0; greedy identical BOTH paths;
liveness 166×/0; suite 169/0/3.
**CORRECTION (r59b, binding):** the r59 baseline binary was itself the stale
r58-delta build (−12.5% deficient — it matched the r58 A/B delta almost
exactly); the true clean delta is **+11.1%** and the co-tenant-tax claim is
superseded. The recorded `feb37de` code commit is a now-unreachable
pre-amend duplicate; the reachable commit is `36a481f` (same subject).
**Lesson.** Two structural misreadings corrected by measurement: (1) per-kt
decode cost and per-launch A-plane DRAM traffic produce the same symptom
(a slow class) with opposite fixes; (2) a baseline binary is a measurement
instrument — if it was not rebuilt from the code you think it was, every
delta it produces is fiction (r59b formalizes the rule).

#### r59b — clean re-measure + baseline-poisoning correction (row 74)

**Method/Result.** Window validation first: idle co-tenants (0% util × 15
samples) with two independent anchors matching their records — llama-bench
3323.29 ± 3.08 vs the clean-machine 3324.42 (0.03% apart) and the
3219.6-record binary re-measured 3217.0 → idle residency taxes nothing.
Binary-drift test (the smoking gun): same baseline code, `/tmp/minfer_pre_r58`
vs `/tmp/minfer_pre_r59` = 3217.0 vs 2824.7 (−12.2%); a fresh worktree
rebuild of the same commit = 3232.0 (healthy) — the r59 session had
snapshotted the r58-delta binary as its baseline. Definitive clean numbers:
fresh HEAD rebuild 3590.8 median (3553.8–3591.2) vs fresh baseline 3232.0 =
**+11.1%** (vs the r58-era clean record: +11.5%); all 10 HEAD runs
3553.8–3591.5, combined median 3580.7 → headline **~3581 tok/s**. vs-llama
same window: 3590.8 / 3323.29 = **1.080×** (minfer ahead). Memory two-mode
census: +1454 MB reproduced (57189 vs 55735 MB with the constant co-tenant
set subtracted).
**Lesson (the campaign's protocol rule): any A/B baseline must be
behaviorally anchored in the same window** — re-measure a known-record
binary, or `git worktree`-rebuild the baseline commit — before trusting
deltas. Corollary: idle co-tenancy is clean-equivalent; never infer a
"co-tenant tax" without an anchor.

#### r60 — PROMOTION: the verified gate set flips DEFAULT-ON (row 75)

**Method.** The six promoted gates invert to the r54 opt-out pattern —
absent/any non-"0" = ON (the verified best), explicit "0" = the pre-r60
behavior: `MINFER_MMQ`, `MINFER_MMQ_RAW`, `MINFER_MMQ_RAW_NB`,
`MINFER_MMQ_A_TRANSPOSE`, `MINFER_MMQ_Q6K_NB`, `MINFER_MMQ_A_FUSE` (absent =
mode 2; "1"/"2" keep r51/r52 semantics; "0"/unrecognized = off);
single-sourced in `CudaState::mmq_gate_on(name)`. Already-default-on
`MINFER_MMQ_Q6K_EXP` / `MINFER_MMQ_Q4K_DSC` unchanged. Every dispatch guard
UNCHANGED and now protecting the default: MMQ entry `nt >= 16 && id % 32 ==
0 && !no_prefill_gemm`; NB-BT `(id / 32) % 8 == 0`; plane registration
`id % 256 == 0` (+ `od % 2 == 0` dsc); fused producers `rows >= 16 && dim %
256 == 0`; mode-2 degradation under debug/trace readers.
**Evidence packed in.** Decode untouched: no `MINFER_MMQ*` read on the nt==1
path (grep + the `rows/n >= 16` guards ahead of every `mmq_a_fuse_mode`
call); decode `-n 16 --greedy` byte-identical; tg128 45.2 = 45.2. Reuse
identity untouched: graph topology reads zero MMQ env (`GraphParams`/`CParams`
env-free; `supports_op`/`supports_fused` env-free); gates choose KERNELS at
dispatch and PLANES at registration — CUDA Graph capture embeds the
per-process-constant selection. **The promotion blocker the bisect caught
(fixed, not reverted):** gate 7 first came back 168/1/3 — the r52 mode-2
dead-write guard fires on ANY quant mix that is not all-NB-BT-consumable
(the 0.5b q4_0 fixture): a fused producer skips the f32 write a generic
`mmq_nt` consumer then legitimately re-quantizes. Stash bisect: clean HEAD
default-env PASSES, clean HEAD gated-env FAILS identically, r60 build FAILS
identically — a faithful reproduction of the verified config's wart, not a
promotion regression. **Fix: mode-2 producers conditioned on a
registration-time `nb_bt_only` flag** — cleared when a non-q4_K/q6_K
quantized weight or a 2-D F32 weight registers (the latter a
silent-corruption exposure the refusal path never had); mixed-quant models
degrade mode 2 → mode 1 (plane AND f32 both written, correct everywhere).
**Result.** Prefill A/B: pre-fix base 3592.1 vs new 3578.0 (−0.39%,
overlapping) — no missed gate; post-fix re-check +0.44% the other way
(noise; mode 2 confirmed active on 7B). Parity ×3 default-env 9/9; greedy-32
default vs snapshot+gated byte-identical (453 B); opt-out `MINFER_MMQ=0` →
zero mmq dispatch labels, f16 spot ~2226 this window (documented ~2353
clean-class); 0.5b q4_k_m smoke byte-identical (post-fix it runs mode-1
producers and STILL matches — the plane is a pure function of A). Suite
169/0/3 (one transient SIGSEGV run did not reproduce; overcommitted-pool
hazard class). Memory: default-on 9484 MiB; planes-off 6217 MiB (**+3.27
GB**); `MINFER_MMQ=0` ~20.5 GB — the escape hatch is ~11 GB HEAVIER than the
promoted default.
**Doc rule going forward: default = the verified 1.080× path; `MINFER_MMQ=0`
= the legacy f16 path.**
**Lesson.** Promotion is a measurement problem, not a flag flip: verify
decode-neutrality, reuse-neutrality, per-gate liveness, mixed-quant
degradation, AND memory in both directions — then fix the wart the default
now exposes (loudly, via the existing guard) instead of reverting the
promotion.

## §2D — Decode campaign (Phase P7-D, D-series): split-attention staging depth (2026-09-07)

Decode is the one phase left with a measurable gap to llama.cpp: tg128 49.3
vs 49.41 (parity) but **@1641 KV 47.2 vs 49.41-class (−4.5%)** on the 7B
q4_k_m / GB10. Phase D opened with a measurement-only attribution round (D1,
artifacts under `/tmp/d1/D1_FINDINGS.md` — ephemeral, key numbers inlined
here) followed by the D2 landing.

### D1 — attribution (measurement-only, no repo change)

- `Op::Attn` nt==1 → `gqa_attn_split_partial<KV>`: grid (ATTN_SPLITS=32 ×
  28 heads) = 896 single-warp blocks, serial online-softmax over
  ceil(nkv/32)=52 rows/split in 4-row batches. **The ONLY decode kernel that
  scales with KV**: 1.98 → 34.1 µs/launch (1.64 → 1641 KV) = +0.90 ms/step =
  100% of the measured 0.91 ms/step wall delta. Everything else flat.
- ncu (standalone probe, minfer's flags): 76.5% long_scoreboard stall, all
  pipes ≤ 12%, 40 regs, 0.78 waves → the kernel rides the memory-LATENCY
  roofline (in-situ 34.1 µs ≈ 12 µs of bytes @273 GB/s + ~22 µs exposed
  latency), not the byte roofline. Root cause: each row's V load is issued
  inside the dependency chain, and the 4-row batch window gives only 4 rows
  of load-level parallelism.
- **ATTN_SPLITS sweep = measured dead end**: 32/64/128 flat partial time,
  combine cost 2–3×, and any split-count change reorders the float sum
  (ndiff ≈ 3.6e-3 of outputs differ, max|Δ|~3e-9 — r57-class, not
  byte-identity). Conversely **staging-depth changes (same split ranges, row
  order, per-row ops) are bit-identical**: probe-verified ndiff=0.
- llama.cpp decode uses the same flash-decoding skeleton but consumes
  256-row × 128-dim windows with 8-warp blocks — latency hidden by per-block
  load depth instead of per-row chaining.

### D2 — explicit K+V register staging LANDED (+2.0% @1641)

**Change** (`gqa_attn_split_partial<KV>`, both KV=__half and KV=float): stage
BOTH the K and the V rows of each 4-row window into registers before the
first online-softmax step, instead of K-only + V loaded inline per row. The
old comment claimed the compiler hoists the inline V loads above the chain —
measured false (that is the whole win): the V loads sat behind the
shfl/expf chain, exposing ~a full memory latency per row. With explicit
staging all 16 row loads (4 rows × K+V × 2×4B) issue back-to-back — SASS
confirms: 16 `LDG.E.CONSTANT` clustered at 0x7a0–0x9f0, first
`SHFL.BFLY`/`MUFU.EX2` at 0xde0. Same rows, same order, same per-row op
sequence → **bitwise-identical by construction**, verified three ways.

**Results** (7B q4_k_m, GB10, interleaved same-window A/B vs pre-change
binary):

| Evidence | baseline | D2 | Δ |
|---|---|---|---|
| probe, cold-DRAM 28-layer rotation, nkv=1641 | 34.6 µs | 19.9 µs | **−42%** |
| nsys in-situ per-launch µs @1641 | 34.1 | 19.4 | **−43%** |
| decode `-n 128` @1641 (3× interleaved medians) | 47.2 tok/s | **48.2 tok/s** | **+2.0%** |
| decode tg128 (KV~1) | 49.4 | 49.4 | flat |
| combine kernel | 96.0 µs/step | 95.7 µs/step | flat |
| ptxas | 40 regs (f16) | 52–58 regs (f16), 72 (f32), STACK/LOCAL 0 | occupancy unchanged (24 blocks/SM cap-bound) |

vs llama.cpp: tg@1641 47.2/49.41 = 0.956× → **48.2/49.41 = 0.975×** (gap
−4.5% → −2.4%); tg128 already at parity.

**Bitwise gates** (all green): probe partial-buffer memcmp ndiff=0, 45/45
checks at nkv ∈ {1, 29, 52, 512, 1641} across every candidate variant;
greedy-32 and greedy-256 token streams on the 1641-token prompt byte-identical
vs the pre-change binary; parity trio (`cuda_prefill_mmq_parity`,
`cuda_prefill_capture_bit_parity_pp16_pp300`, `cuda_fa_prefill_attention_parity`)
+ `cuda_attn_split_decode_parity` + q4k/q6k decode MMVQ parity all ok; full
suite **169/0/3**.

### D2 negative results (do not retry blindly)

All bitwise-safe (after fixing a probe wait-group bug — see below), all
MEASURED WORSE than the landed form in the realistic cold-DRAM mode:

| Variant | cold-DRAM µs | vs landed 19.9 |
|---|---|---|
| register staging NR=8 | 22.7 | worse |
| cp.async smem K-only pipe NR=4 S=2 / S=3 | 23.4 / 22.4 | worse |
| cp.async smem K-only pipe NR=8 S=2 | 30.9 | much worse |
| cp.async smem K+V pipe NR=4 S=2 | 21.0 | worse |
| pair-unrolled register lookahead (95 regs) | 22.1 | worse |

Read: at 8 B/lane/row (f16) the LDGSTS granularity is too small and the LDS
round-trip too costly to beat direct LDG→register consumption; deeper
register windows (NR=8) hit the in-flight-load limit instead of hiding more
latency. cp.async remains the right tool where staging tiles are ≥16 B/lane
(the MMQ kernels) — not here.

**Probe-bug lesson (r59b-class):** the first probe run reported several pipe
variants DIVERGENT (and the rest "OK" by luck). Root cause was in the PROBE,
not the concept: `cp.async.wait_group <STAGES-1>` only forces the oldest
group complete when exactly STAGES groups are outstanding; tail iterations
must `wait_group 0`. Any future cp.async ring over a runtime trip count
needs the same branchy wait.

**Residual decode levers** (for a future D3): the serial chain itself is now
the floor (19.4 µs ≈ 12 µs bytes + ~7 µs chain); crossing below it needs the
llama-vec-style multi-warp 256-row cooperative rewrite, which replaces the
block/work mapping and is NOT byte-identity-able (tolerance gate required).
`f32_bits_to_i32` (~0.5%/step) and the short-KV combine idle reads
(~90 µs/step) remain untouched micro-levers.

### D3 — 14B decode attribution (D3-1, measurement-only) + D3b bitwise MMVQ levers (2026-09-07)

**D3-1** (full report `/tmp/d3/D3_FINDINGS.md`, artifacts retained): 14B
q4_k_m (48L, hidden 5120, 40:8 GQA) decode-step census at tg128 and nkv=3254.
The short-KV wall gap vs llama.cpp (43.84 vs 41.14 ms/step) is **not
attention** (0.75% of the step): ~half is three MMVQ stragglers running
below the 220–225 GB/s class their siblings hit — attn_v-q6K on the
padded-f32 kernel (**134.8 GB/s**, dispatched because the `od*id >= 24M`
MMVQ gate excludes its 5.2M-element shape), ffn_down-q6K (**198.9**),
output head (**200.1**) — together ≈ +1.2 ms/step; the other half is the
elementwise/launch chain (97 rms + 265 quantize + ~700 sub-2 µs launches).
Attention itself scales +3.33 ms/step to 3.3K (the ONLY KV-scaling kernel,
67% of the DRAM-bytes floor, long_scoreboard-bound) and is D3a's
tolerance-gated target, not this session's.

**D3b session** (bitwise-gated; bar: ≥ +0.4% on the lever's target or
revert; one landed):

- **D3b-1b — down-q6K pipelined MMVQ, LANDED (`f1825b5`).** For npair > 256
  (id > 8192: ffn_down id 13824 → npair 432) `q6_k_q8_mmvq_v2` gives threads
  0..npair−257 a SECOND serial unit whose weight loads sat exposed on the
  critical path. `q6_k_q8_mmvq_v2_pf` issues both units' weight+q8 loads
  back-to-back before either accumulates. Bitwise-identical by construction:
  same thread→unit map (u = tid, tid+256), same per-unit dp4a tree, the
  per-unit accumulation statement is textually identical (same FMA
  contraction shape), ascending-u order, same block reduce — only load
  scheduling moves (the D2 staging-depth precedent). Gates: 114/114
  `MINFER_GRAPH_DUMP` files byte-identical vs the pre-change binary (logits
  prefill+decode, kv0, all 48 layer KV, node dumps), greedy −n 256 stream
  byte-identical, suite 169/0/3 (+ FA trio, split-decode parity,
  replay-bit-parity by name). Wall (interleaved 3× medians): **7B tg128
  48.03 → 49.47 (+3.0%, min-new > max-base), 7B @1641 46.66 → 47.94
  (+2.7%, SEP)** — down-q6K is ~25% of the 7B per-step weight stream, so
  the 198.9 → ~220 GB/s projection lands almost exactly; 14B tg128
  22.80 → 22.90 (+0.44%), @3254 21.01 → 21.06 (+0.24%, SEP).
  **D4-2 CORRECTION (2026-09-09): the 7B legs are void** — v2_pf's dispatch
  had no npair upper bound, so on the 7B ffn_down (npair 592) it dropped
  units 512..591: most of the "gain" was 13.5 % of the down-q6K work never
  performed, not pipelining (the bitwise gates compared v2_pf-to-v2_pf
  binaries and could not see it). The pipelining gain itself is the
  14B-sized +0.2-0.4 % class. See the D4-2 chapter for the fix and the
  re-anchored 7B numbers.
- **D3b-1a — attn_v-q6K off the padded-f32 kernel, REVERTED (both routes).**
  (a) The D3-1-suggested MMVQ routing (lower the `od*id >= 24M` gate for
  id 5120) is **not bitwise-able at all**: the MMVQ path quantizes
  activations to q8 (dp4a) while the padded kernel consumes f32 — different
  accumulation semantics, so 0-byte gate impossible; would need the D3a
  tolerance session. (b) The bitwise-safe re-map (kernel+launcher NSG 2→1,
  rows still 2/warp, 2× the warps) passed every bitwise gate but measured
  kernel 36.4 → 39.9 µs/launch (nsys, same window) and wall −0.74% tg128 /
  −0.33% @3254: doubling warps doubles the y re-read traffic (every warp
  streams the full 20 KB activation row for its 2 rows), and the padded
  kernel at this shape is not warp-starved. Lesson: within byte-identity the
  padded kernel's only free knob is rows-per-warp, and 2 is the sweet spot;
  134.8 GB/s here is an L2/latency composition, not a parallelism deficit.
- **D3b-1c — output-head dynamic block size, REVERTED.** npair = 160 at
  id 5120 means 96 of 256 threads idle per block; launched warp-round-up
  blocks (160 threads, 5 warps) with `mmvq_block_reduce` bounded by the
  actual warp count (idle threads only ever contributed exact +0.0 terms —
  bitwise-safe). Bitwise-green, but 14B wall +0.04%/+0.09%: GB10's SM limit
  is 1536 threads, so 6×256-thread blocks already allocate 1536 (960 live)
  vs 9×160 = 1440 live — the win mechanism does not exist; 7B @1641
  (+0.26%, SEP; npair 112 → 144 idle) was real but sub-bar.
- **D3b-2 — short-KV combine skip: NOT implemented (analysis-negative).**
  A single-split path is bitwise-equal to the landed 32-split
  partial+combine ONLY when one split is live, which under
  `chunk = ceil(nkv/32)` means nkv = 1; for any real decode step the combine
  merges ~nkv/chunk live partials with `exp(mx_sp − gmx)` rescaling, and
  that merge reorders the float sum relative to the serial online-softmax
  chain (D1 measured split-count reordering: ndiff ≈ 3.6e-3 of outputs,
  max|Δ| ~3e-9 — r50/r57 class, not byte-identity). Additionally the
  split grid is frozen by CUDA-graph replay capture (the kernel reads
  `positions` at runtime, so a capture-time dispatch branch would lock the
  capture-step's shape for the whole session). The bitwise-safe residual —
  combine early-out of empty splits (their contributions are exact +0.0) —
  is worth ≤ ~15 µs/step at tg128, below every bar. The real ~166 µs/step
  prize needs the D3a tolerance-gated session.
- **Dump-gate artifact worth recording:** the `node{3,5,8}_prefill.f32`
  graph-dump files read ALIASED pool slots whose node→slot identity can
  differ per binary even when every real value is bit-identical (contents
  byte-equal modulo slot swap in the D3b-1a run; downstream tensors —
  logits both phases, all KV, decode nodes — identical in every run). Gate
  bitwise checks on logits/kv/decode-node files; treat same-size prefill
  node-slot diffs as aliasing, not numerics.

**Window anchors (interleaved pre-binary medians, this session):** 14B
tg128 22.80–23.11 (D3-1 window: 22.81), @3254 21.01–21.11 (21.25); 7B tg128
48.03 (guard 49.4 — co-tenant state; post-1b 49.47 re-reaches it), @1641
46.66 (guard 48.2). pp512 guard untouched (prefill path unchanged;
2055 tok/s spot-checked).

### D3a — 4-warp fattn-vec-style split-attention rewrite, REVERTED (negative result) + tolerance-gate calibration (2026-09-07)

**Method** (full record `/tmp/d3/D3A_FINDINGS.md`, code diff preserved at
`/tmp/d3/d3a_kernel_patch.diff`, experimental binary `/tmp/d3/minfer_post_d3a`):
`gqa_attn_split_partial_h4w` re-expressed the f16-KV decode split attention
(hd==128 dispatch; hd≠128 keeps the 1-warp kernel) with the llama.cpp
`fattn-vec` structure source-verified @ca3d5a3e1 — 128 threads (4 warps), K/V
streamed from global (no smem K/V), Q in registers (16 dims/thread), 8-lane
subgroups (one row each), 32-row windows per warp with ONE online-softmax
rescale per window, probs staged to a 128-float smem row, 4-warp LSE-merge
epilogue through `vkq_s`/`mx_sh` smem staging. Grid stayed
`dim3(ATTN_SPLITS, n_head)` (nkv-independent, replay-safe). `__launch_bounds__(128, 8)`
→ REG 64 / STACK 0 / SMEM 8960 B = 32 warps/SM vs 24.

**Why it failed — the rows-per-warp pathology.** `rpw = ceil(ceil(nkv/32)/4)`
is 26 at 14B @3254, 13 at 7B @1641, **1 at tg128**. The 32-row window maps
live rows onto 32 lane-slots, so rpw=13 idles 59% of the slots (subgroups 2–3
fully idle) and rpw=1 leaves 3 of 4 warps exited with the live warp using 8 of
32 lanes; per-block fixed costs (8 KB of Q loads vs the 1-warp kernel's
512 B, the epilogue sync/staging, partial writes) amortize over rpw rows.
The incumbent is serial-dense — every lane busy on every row, 0.58–0.83
waves all-resident. nsys (NO_CUDA_GRAPH, −n 8 mean): 14B @3254 split kernel
73.4 → 68.4 µs (**−6.9%**, 71.5% of the 48.9 µs bytes floor vs 67%) but 7B
@1641 21.1 → **34.7 µs (+64%)**; wall (interleaved 3× medians): 14B @3254
21.04 → 20.84 (−0.95%, inside ±2% noise — a −6.9% kernel is only ~+0.5% of a
step where attention is 7.4%), 7B @1641 47.53 → 45.91 (**−3.4%**, tight
clusters, matches the kernel), tg128 noise-level both models. Every bar
failed (14B @3254 ≥ +1.5%, 7B @1641 ≥ 47.9). **Reverted per the r44
precedent** (parity-green, wall-sub-bar). Rescue path for lever 2: keep the
4-warp path for dense chunks and fall back to the D2-staged 1-warp body
inside the same kernel when rpw < ~16 — the branch is nkv-uniform, so
replay-safe.

**The durable result — tolerance-gate calibration.** The kernel was
numerically fully green, which made the session the first exercise of the
D3-1 tolerance gate — and it failed against the PROPOSED gate, not the code:

- Kernel level: h4w vs CPU ≤ **1.3e-7** (nkv 3..4096 incl. the exact in-situ
  14B shape and all chunk boundaries); vs the incumbent on identical inputs
  ≤ 8.9e-8; on realistic outlier-scale data (residual |q|~50, V outliers
  ±127) new-vs-CPU 6.5e-5 vs old-vs-CPU 3.8e-5 — the same error class.
- In-situ (NO_CUDA_GRAPH + MINFER_TRACE per-node attn outputs, 14B): prefill
  attention bitwise; decode per-layer delta L0 2.4e-7, L1–L11 **bitwise 0.0**
  (the f16-KV store is a noise gate — sub-ULP reorder noise is quantized away
  at each layer's K/V write), L12+ 1e-3..5e-1 (noise crosses f16 rounding
  boundaries and amplifies through outlier-dim cancellations).
- End-to-end logits: max|Δ| **0.376 (14B, 48L) vs 0.389 (7B, 28L)** — the
  same magnitude at both depths ⇒ depth-independent class, not a bug.

**D3-1's proposed gate "end-to-end max|Δlogits| ≤ 1e-3" is unsatisfiable for
ANY accumulation-order-changing rewrite on 28–48-layer models** — a
kernel-level-1e-7 change produces O(0.4) end-to-end drift (the r50/r57
lesson generalized to decode). The operative tolerance-gate set, all
demonstrated green here on the experimental kernel: (1) kernel-level parity
vs CPU ≤ ~1e-4 on realistic data; (2) **argmax identical at every dumped
step with top-2 margin > 0.1 — HARD** (margins 4.95/10.97); (3) greedy
−n 256 × 5 seeds × both models **0/10 diverged a single token** + temp 0.8
seed 7 sampled controls identical; (4) suite 169/0/3 + FA trio +
split-decode parity (extended with an hd=128/n_ctx 4200 shape driving the
new kernel — reverted with the code); (5) interleaved A/B bars.

**Dump-gate gotchas (extend D3b's aliasing note).** Decode node dumps
`node{2,3,5,8,11}_decode` read ALIASED pool slots (node11 = kv_load's dump
is 20 KB, not the 16.7 MB KV region) — node-diff noise is slot aliasing.
And KV-region diffs between `-n 1` and `-n 2` dumps are a **pre-existing
step-dependent row rewrite** present pre-vs-pre (rows 17–20 of layer-10 K
change between decode steps 1→2 in the incumbent binary too): gate on
logits + final-step KV with the SAME -n on both sides.

**shfl_sync deadlock (Appendix-B lesson).** `__shfl_xor_sync(0xFFFFFFFF, v,
off, 8)` inside `if (row < wend)` deadlocks when lanes disagree on row
validity (the mask names all 32 lanes; some never arrive). Fix: compute the
contribution conditionally (0.0f default), reduce unconditionally for ALL
lanes, mask after (`s = -INFINITY`). Presents as a GPU-spin hang in
`cargo test` and a hang in the standalone probe.

**Follow-ups.** Lever 2 (multi-warp decode attention) must now solve the rpw
pathology (subgroup-dense row mapping or the runtime small-rpw fallback).
Lever 3 (attn_v-q6K MMVQ routing) is NOT bitwise-able and needs exactly the
calibrated gate set above. 14B @3254 attention remains the open KV-scaling
gap: 73.4 µs/layer vs the 48.9 µs floor; the residual ~20 µs needs
window-level K/V prefetch pipelining (D2's staging trick at window
granularity), unattempted.

### D3-4 — L1 hybrid rpw dispatch LANDED + L2 window-prefetch pipelining REVERTED + long-prompt dump calibration (2026-09-07)

Base binary `/tmp/d3/minfer_pre_d34` (sha1 d25f6d84, HEAD 9d5c24f+docs
a05af20 window). Anchors this window (interleaved 3× medians, pre): 14B
tg128 23.81→22.96-class (drifts), @3254 21.90→21.20; 7B tg128 50.90→50.28,
@1641 49.32→48.78 (co-tenant sglang resident; ±2% window drift — every
judgment is same-window interleaved A/B).

**L1 — dual-kernel self-gating rpw dispatch, LANDED (`22336b2`).** D3a's
rescue path, with the geometry lesson applied. `rpw = ceil(chunk/4)`,
`chunk = ceil(nkv/ATTN_SPLITS)`: rpw ≥ 16 (nkv ≥ 1921) → the D3a 4-warp
fattn-vec-style kernel (`gqa_attn_split_partial_hybrid`, the D3a body
verbatim, probe-verified ≤1.3e-7 vs CPU in D3a); rpw < 16 → the incumbent
D2-staged 32-thread kernel (bitwise, via the shared `attn_split_1w_body`
device function). The first cut put BOTH bodies inside one 128-thread
kernel — bitwise-green (7B @1845-token dump identical) but the 1-warp body
in a 128-thread block runs at 12 working warps/SM (1536/128) vs the
incumbent's 24-32, and 7B @1641 measured **35.4 vs 19.8 µs (+78%, nsys)** —
block GEOMETRY, not math. Landed form: launch both kernels (static grids →
CUDA-graph capture/replay unaffected); each reads `positions[0]` and
exactly one is live per nkv (the branch is nkv-uniform across the grid).
Cost: one dud launch/layer (~1.3-1.5 µs, measured).

**L1 results.** nsys (NO_CUDA_GRAPH, bench -p 3254/1641 -n 8, mean of last
384): 14B @3254 split 72.1 → 62.11 µs (−13.9%; D3-1's window: 73.4 → 68.4
for the pre-hybrid form) + 1.5 µs dud; 7B @1641 incumbent 20.7 µs + 1.3 µs
dud (min identical to pre → body intact). Wall (interleaved 3× medians):
14B @3254 21.20 → 21.33 (**+0.61%**, SEP: min-new 21.25 > max-base 21.23);
14B tg128 22.96 → 22.94 (−0.09%); 7B tg128 50.28 → 50.20 (−0.16%); 7B
@1641 48.78 → 48.68 (−0.20%; one 44.40 co-tenant outlier in pre).
Guards: 7B ≥ 49.0 / ≥ 47.9 and 14B tg128 ≥ 22.7 all hold. Gates: 7B
@1845-token dump bitwise on every gated file (logits prefill+decode, all
KV, decode nodes — 71/71; the 3 `node{3,5,8}_prefill` diffs are the
documented slot-aliasing trio, reproduced pre-vs-pre); 7B @2800 h4w
tolerance class — max|Δlogits| 0.309 (the calibrated 0.39-class), argmax
identical at margin 0.716 (HARD gate), upper-layer KV f16-noise pattern
(kv0-7 bitwise, kv8-27 decode-side drift); greedy −n 256 × 5 seeds × both
models on h4w-regime prompts: exactly one divergence each at the regime
entry (1/256 = 0.4% < 2%), coherent continuation (no repetition
degeneracy), temp 0.8 seed-7 sampled controls identical; suite 169/0/3
with the parity test extended by an hd=128/n_ctx-4200 shape whose pos0
sweep crosses the rpw 15/16 dispatch boundary (nkv 1920/1921).

**L2 — window-level K/V prefetch pipelining in the h4w body, REVERTED.**
The brief's main-lever hypothesis: 62.1 µs is 79% of the 48.9 µs bytes
floor and the K phase serializes 8 load→reduce passes per 32-row window
(the V phase 7 load→FMA passes), so issue-point-only pipelining (D2's
bitwise class) should close toward the floor. Built (`patch_l2.py`): K-pass
software pipeline (+2 uint4/thread) + 4-deep V-bulk ring (+8 uint4),
`__launch_bounds__` minBlocks 8→4 to fund the registers (64→128 cap).
Measured: h4w 62.1 → **66.46 µs (+7%)** — the occupancy halving (32→16
warps/SM) and wave growth (1280 blocks: 3.33 → 4.44 waves) cost more than
the shorter chains recover. Read: at 131 KB/SM of loads already in flight
(≈30× the latency-BW product), the kernel is bytes+tail-bound; the
remaining 21% over the floor is L2 5×-re-read composition + wave tail, not
per-warp chain depth. The D2 lesson holds only where registers are free —
at a 64-reg budget any pipeline funding trades occupancy 1:2. Bitwise gate
was never reached (no point — regressed before gating).

**Distance to parity (post-D3-4, 14B @3254, this window).** minfer 21.33
t/s = 46.88 ms/step vs llama 24.32 = 41.12 ms → **−5.76 ms needed (−12.3%)**.
Known-lever inventory: attention residual (62.1 − 48.9) × 48 = 0.63 ms;
the three D3-1 MMVQ stragglers (attn_v-q6K 0.28 + ffn_down-q6K 0.49 +
output-head 0.44) = 1.21 ms; D3c elementwise fusion ≈ 1.0 ms (projected
+2.1% wall, not implemented this session). Sum ≈ 2.84 ms = 49% of the gap
→ ~22.6 t/s (0.93×) if ALL landed. The remaining ~2.9 ms is the matmul
aggregate (D3-1's wall-effective 194.9 vs llama 207.6 GB/s, which its
implied per-kernel BWs exceed) + launch-structure slack — decode-GEMM
levers beyond the D3 list. 7B: tg128 1.016× (ahead), @1641 0.985× — the
7B decode campaign is effectively closed.

**Follow-ups.** (1) L3 attn_v-q6K → MMVQ q8-activation routing — NOT
bitwise-able, needs exactly the calibrated gate set (argmax hard gate +
greedy + suite + A/B); D3-1 estimate +0.28 ms/step ≈ +0.6% 14B wall;
skipped here on time. (2) D3c elementwise fusion (rms+quantize, quantize
into MMVQ prologue) ≈ +2.1% at all lengths — the largest single remaining
KNOWN lever. (3) The 14B attention residual needs a bytes-side lever (GQA
q-head batching to cut the 5× L1/L2 re-read), not more pipelining.
(4) Long-prompt CLI n_ctx headroom fix (pre-existing).

### D3-5 — Stage 1 of the decode-alignment program: fused-producer decode A-quantize LANDED (1a), head/ffn_down geometry levers analysis-negative (1b/1c) (2026-09-08)

Base binary `/tmp/d3/minfer_pre_d35` (sha1 ab09c1de, HEAD 6788e98 window).
Anchors this window (interleaved 3× medians, pre): 14B tg128 22.97 (A/B
interleaved pre side 23.05), @3254 21.06 (one 15.87 co-tenant outlier rep;
interleaved pre 21.36); 7B tg128 49.90 (interleaved 49.86), @1641 48.71
(interleaved 48.47). sglang co-tenant resident throughout (idle-co-tenancy
clean-equivalent; per-rep outliers landed in both sides and medians carried).

**1a — fused-producer decode A-quantize, LANDED (`3230b2b`).** D3-1's census
put ~0.46 ms/step in 265 standalone `quantize_q8_0_pad40` launches (one in
front of EVERY decode MMVQ matmul, each ~1.7 µs, each re-quantizing a row its
sibling matmuls also re-quantize — attn_norm's output was quantized 3× per
layer for q/k/v) plus part of the ~700 sub-2µs launches. The inline-per-block
form (each MMVQ row-block re-quantizes the row itself) was rejected on paper:
it multiplies the A-side L2 traffic od×14 KB (ffn_gu alone: +18.6 GB/step) —
the r49 shared-A lesson. The landed form is the r51 prefill winner transplanted
to decode: the PRODUCER writes the q8 plane. `rms_norm_quant_pad40` (rms body
verbatim + the standalone kernel's per-32-block body verbatim, re-reading the
row the block just wrote — L1-hot after a barrier) and `swiglu_quant_pad40`
(swiglu body verbatim, 8 threads per 256-thread block quantize the block's own
8 output blocks — the same per-32-block thread count as the standalone kernel)
record their plane in the r49 MmqCache (native pad40 form, keyed on the
producer's f32 output pointer); `decode_quantize_native` consults the cache in
the three decode MMVQ entries and skips the standalone launch on a hit. The
FusedFFN node joins the cache-clear preserve set (its input IS the ffn_norm rms
output and the internal gu matmul is that src's first consumer — same
consecutive-consumer window as MatMul→MatMul). Attn_o keeps a standalone
quantize (its producer is the attention node, not touched this session). The
q8 bytes are bit-identical by construction — max is exact for ANY association,
the rintf/clamp pass is elementwise — and the whole GEMV byte-consumption is
unchanged. `MINFER_NO_DECODE_A_FUSE=1` restores the exact pre-D3-5 path.

**1a gates.** Probe (new `cuda_decode_a_quant_fuse_bitwise` test): fused vs
standalone q8 buffers memcmp-equal at the 14B hidden size and the ffn_down
shape, f32 producer outputs bit-identical, MMVQ matmul outputs bit-identical
through the cache-hit path. Suite **170/0/3**. Dump gate (14B, short prompt,
−n 4): `logits_{prefill,decode}` + all `kv*` byte-identical; 3 same-size
node-dump diffs (`node{2,3,11}_decode`) are the documented pool-slot aliasing
class, reproduced pre-vs-pre AND post-vs-post (the same-binary diff sets cover
every pre-vs-post diff — no numeric delta). Greedy −n 256 token streams
byte-identical vs pre on both models (14B @2799-token prompt, 7B @1847). One
operational note: `prompt_3k3` (5451 tokens) panics in BOTH binaries at the
n_ctx 4096 default (graph.rs:367 — the D3-4 follow-up #4 headroom bug), so
long-prompt greedy gates must use ≤2.8K-token prompts until n_ctx sizing is
fixed. nsys (14B @3254, NO_CUDA_GRAPH, bench −n 8): standalone quantize
launches 4448 → 964 (−78%), total kernels 29402 → 25918, sub-2µs launches
15171 → 11723 (−3448 = exactly the quantize delta); swiglu 2.06 → 2.31 µs,
fused rms ~+0.2 µs (epilogue cost, priced into the wall).

**1a results** (interleaved 3× medians, pre vs post, all SEP): 14B tg128
23.05 → **23.35** (+1.30%), @3254 21.36 → **21.68** (+1.50%); 7B tg128 49.86
→ **50.69** (+1.66%), @1641 48.47 → **49.16** (+1.43%). The win lands at ALL
lengths on both models — the launch/round-trip tax was length-independent, as
projected. vs the pre-D3-5 anchors: 14B tg128 +1.65% (session bar ≥ +1.5%),
@3254 +2.95%. vs llama.cpp: 14B 0.961× (tg128) / 0.891× (@3254); 7B **1.026×**
(tg128, ahead) / 0.995× (@1641, parity) — the 7B decode campaign is closed.

**1b — output-head od-split / 512-thread re-map: ANALYSIS-NEGATIVE.** The
brief's occupancy math, done first: npair = 160 q6_K units/row → 96 of 256
threads idle (the D3b-1c knob, measured neutral); 6×256-thread blocks/SM =
288 resident rows, and D3b-1c's 9×160 = 432-row form was ALSO neutral —
rows-in-flight is not the limiter; the head sustains 47.6 blocks/µs while
ffn_gu demonstrates 76/µs — block scheduling is not the limiter. The dual-row
512-thread form is bitwise-capable (each 256-thread half reduces its row with
the identical 8-warp tree) but moves none of the measured-neutral quantities.
Per the brief's own rule (pick the variant with a MEASURED mechanism), not
built. The head's 200.1 vs 220-225 GB/s class remains unexplained by any
geometry knob tried so far — same conclusion class as D3b-1a's padded-kernel
verdict (latency/L2 composition, not a parallelism deficit).

**1c — ffn_down-q6K 512-thread single-unit: ANALYSIS-NEGATIVE.** Not bitwise
vs the landed `q6_k_q8_mmvq_v2_pf`: in the 256-thread form thread t sums
fma(u_t) + fma(u_{t+256}) into one acc BEFORE the block reduce; at 512 threads
those units sit in different threads and their sum moves into the 16-warp
cross-warp reduce order — a different float sum (the r50/r57 class). A bitwise
pair-exchange emulation adds a barrier for zero resident-parallelism gain
(5120 rows = 18 waves either way), and the exposure path 1c targets was
already fixed by v2_pf's up-front dual-unit load issue.

**Distance to parity (post-D3-5-1a, 14B @3254).** minfer 21.68 t/s = 46.12
ms/step vs llama 24.32 = 41.12 ms → **−5.00 ms needed (−10.8%)**. Known-lever
inventory: attention residual (62.1 − 48.9) × 48 = 0.63 ms (GQA q-head
batching to cut the 5× L1/L2 re-read — Stage 2); attn_v-q6K padded-f32
straggler ≈ 0.28 ms (tolerance-gated MMVQ routing, D3-4 follow-up); output
head ≈ 0.44 ms (mechanism-less per 1b above); rms/elementwise chain ≈ 1.0 ms
(97 rms launches — the fused-epilogue rms is now slightly heavier, so the
next elementwise lever is rms-launch consolidation, D3-1's follow-up class).
Sum ≈ 2.35 ms = 47% of the remaining gap. The 1a win itself removed
~0.56–0.69 ms/step (−78% of the quantize launches + their gaps).

**Follow-ups.** (1) attn_o keeps a standalone quantize (48/step, ~0.08 ms) —
fusing it into the attention combine epilogue touches the D3-4 hybrid kernel
pair; low value, high regression risk, skipped. (2) Stage 2: GQA q-head
batching in the decode split attention (the 5× q re-read) is the largest
remaining single lever (0.63 ms). (3) rms-launch consolidation (97 rms →
fewer, wider launches) is the remaining elementwise mass (1.15 ms class).
(4) Long-prompt CLI n_ctx headroom fix (pre-existing, now also blocks
long-prompt greedy gates).

### D3-6 — Stage 2 of the decode-alignment program: GQA q-head batching in the decode split attention built + gate-green + REVERTED with the mechanism nailed — the 5× L2 re-read is NOT the residual (2026-09-08)

Base binary `/tmp/d3/minfer_pre_d36` (sha1 92485d3c, HEAD 6085928 window);
sglang co-tenant resident throughout (same-window interleaved methodology).

**2a — GQA q-head batching, REVERTED (mechanism-negative, not gate-failing).**
The D3-4 follow-up held that the attention residual (62.1 vs the 48.9 µs
DRAM floor at 14B @3254) is the K/V 5×-re-read composition. The lever was
built as designed: `gqa_attn_split_partial_gqa` puts ALL gqa q-heads of one
kv head in ONE block — grid `(ATTN_SPLITS, n_kv_heads)`, `32*gqa` threads
(warp w = q head `hk*gqa+w`, full split stripe per warp), the per-warp
32-row-window pass extracted verbatim from `attn_split_h4w_body` into a
shared `h4w_warp_windows` device function, per-warp epilogue (warp-local LSE
state is final; the 4 row-class oc copies fold by an 8/16 butterfly), static
nkv-independent grid/block (replay-safe), same `rpw >= H4W_MIN_RPW` dispatch
slot (gqa 2..8; gqa 1 and > 8 keep the 4-warp hybrid; rpw < 16 incumbent
bitwise). Patch preserved at `/tmp/d3/patch_2a.py`.

**Every gate in the calibrated package passed** — the kernel was correct;
the WALL mechanism was not there:

- Kernel-level vs CPU: the split-decode parity test extended with the 14B
  (40:8, gqa=5) and 7B (28:4, gqa=7) geometries across nkv 3..4096 (incl.
  the exact in-situ 2808 divergence-point shape and the rpw 15/16 boundary)
  plus the D3a outlier calibration (|q|~57, V ±144) at ≤1e-4 — all green
  (these shapes now permanently cover the LIVE h4w kernel too).
- Dump gate (both models, 2.8K-token prompt = full batched regime, −n 4):
  `logits_prefill`/`kv0_prefill`/early-layer KV bitwise (14B kv0–15, 7B
  kv0–4); `logits_decode` max|Δ| 0.265 (14B) / 0.259 (7B) — the calibrated
  0.39-class; argmax identical (HARD gate) at margins 2.187 / 0.557;
  upper-layer KV decode-side drift = the documented f16-boundary class; the
  node{2,3,5,8,11}_decode + node{3,5,8}_prefill diffs reproduced pre-vs-pre
  (slot aliasing).
- Greedy (the sampler-vs-kernel attribution below is the new part):
  with `--repeat-penalty 1.0` (raw argmax) the −n 256 streams are
  **byte-identical pre-vs-post on BOTH models** — the kernel never flips a
  raw argmax in 512 steps. With the default repeat penalty each run shows
  exactly ONE knife-edge flip (1/256 = 0.4% < 2%): 14B at step 63 where the
  raw top-2 sat at probgap 0.0167 (logit gap 0.037) and swapped order; 7B at
  step 8 where the penalized winner was raw-rank 6+ (outside the traced
  top-5). Post-flip text coherent, no repetition degeneracy; temp-0.8
  seed-7 sampled controls byte-identical (incumbent regime both sides).
- Suite 170/0/3 incl. FA trio + the extended split-decode parity.

**Measurement (why it reverted).** nsys (NO_CUDA_GRAPH, bench −n 8, mean of
last 384): h4w 63.76 → batched 64.79 µs/layer (**+1.6%** — not the projected
≤54). ncu on the same launches (2-launch capture, serialized):

| kernel | `lts__t_sectors.sum` | vs 1× analytic (458K) | nsys (live) | ncu (serialized) |
|---|---|---|---|---|
| h4w hybrid (5× re-read) | 2,121,671 | 4.63× | 63.76 µs | 150.8 µs |
| GQA-batched (1× read) | 472,583 | 1.03× | 64.79 µs | 108.7 µs |

The batched form eliminates the L2 re-read EXACTLY as designed (1.03× the
analytic one-pass minimum — L1 catches the sibling warps' same-address
loads), yet live time is flat: **the 5× L2 re-read was already fully hidden
under the latency roofline in the live regime** (L2 has bandwidth surplus;
the kernel is latency/dependency-chain + wave-structure bound — D1's
original attribution, which D3-4's "L2 composition" note had over-corrected).
ncu's serialized replay shows the −28% only because cold caches make traffic
dominant there — the opposite regime. Consequence for Stage 3: the attention
residual (62.1 − 48.9 = 13.2 µs/layer ≈ 0.63 ms/step) is NOT addressable by
any bytes-side traffic lever; the next attention lever would have to cut
exposed latency (cross-window prefetch at retained occupancy — D3-4 L2's
register-pipeline form lost 1:2 to occupancy; smem-tile cooperative staging
is the untried remainder, historically disfavored at this granularity by
D2's cp.async negatives).

**New durable gate methodology (sampler-vs-kernel attribution).** A greedy
divergence under the default repeat penalty is NOT automatically kernel
drift: the penalized argmax is a knife-edge among up to 64 penalized
candidates, and flips there cascade (the next step embeds a different token
— visible as `embed`-node input change in `MINFER_TRACE`, with steps 0..k−1
top-5s matching to drift class). The clean kernel-numerics stream is greedy
with `--repeat-penalty 1.0`; byte-identity there is the strong gate. The
per-step `logits_top` trace (`cmp_trace.py`) supplies the raw top-2 probgap
at any flip step.

**2b — attn_v-q6K → MMVQ q8-activation routing: NOT attempted** (time). The
lever stands as the D3-4/D3-1 follow-up: attn_v (od 1024 × id 5120 = 5.24M
elements < the 24M MMVQ gate) runs the padded-f32 kernel at 134.8 GB/s vs
the 220 class; routing = lowering the Q6_K decode gate for this shape
(mostly a dispatch change; numerics delta confined to f32→q8 rounding of the
attention output, tolerance-gated per the D3a package). D3-1 estimate
+0.28 ms/step ≈ +0.6% 14B wall. `MINFER_NO_DECODE_A_FUSE=1` semantics
unaffected (attn_v's producer keeps its standalone quantize either way).

**Window anchors** (unchanged from D3-5, the pre_d36 window): 14B tg128
23.35, @3254 21.68; 7B tg128 50.69, @1641 49.16. Guards: 7B tg128 ≥ 49.0 /
@1641 ≥ 47.9; 14B tg128 ≥ 22.7.

**Distance to parity (post-D3-6, 14B @3254, unchanged walls).** minfer
21.68 t/s = 46.12 ms/step vs llama 24.32 = 41.12 ms → −5.00 ms (−10.8%).
Stage-2 outcome: the attention residual's 0.63 ms is re-attributed from
"L2 re-read traffic" to exposed-latency (bytes-side levers dead per the ncu
sector evidence above). Remaining KNOWN inventory: rms-launch consolidation
≈ 1.0 ms; attn_v-q6K MMVQ routing ≈ 0.28 ms (2b, deferred); output head
0.44 ms (mechanism-less per D3-5 1b); attention latency levers ≈ ≤0.63 ms
mechanism now uncertain. Sum ≈ 1.7–2.35 ms = 34–47% of the gap; the rest is
the matmul aggregate + launch-structure slack — Stage 3 decision input.

### D3-7 — Stage-2 final session: attn_v-q6K MMVQ routing LANDED (2b) + rms/elementwise-launch consolidation LANDED (2c) (2026-09-08)

Base binary `/tmp/d3/minfer_pre_d37` (sha1 b29f2ae6, HEAD 92b0712). Window
anchors (pre_d37, 3× medians): 14B tg128 23.36, @3254 21.65; 7B tg128 49.91,
@1641 48.31 (7B co-tenant-drifted vs the D3-5 window; same-window interleaved
A/B throughout). sglang co-tenant resident.

**2b — attn_v-q6K → MMVQ routing, LANDED.** The Q6_K decode dispatch gate
`od*id >= 24M` lowered to `>= 4M`. GGUF census (`/tmp/d3/census_q6k.py`):
the only q6_K shape in (4M, 24M) across the supported model set is the 14B
attn_v (od 1024 × id 5120 = 5.24M, **11 of 48 layers** — the D3-1 +0.28 ms
projection implicitly assumed more); 7B attn_v is od 512 × id 3584 = 1.8M
and stays on the padded-f32 kernel, so 7B is untouched (streams identical).
The D3-5 MmqCache consult serves the path as-is: attn_v's producer is the
attention node (no fused quantize), but attn_v and attn_o consume the SAME
buffer with the same id, so attn_v's `decode_quantize_native` records and
attn_o HITS — standalone quantize count per step unchanged (964, verified in
the trace). Dispatch inside `q6_k_decode_mmvq`: id 5120 → npair 160 → the
plain v2 kernel, 1024 blocks × 256 threads.

Measurement: kernel 33.16 → 24.32 µs (nsys, same-window, −26.7% = ~177 GB/s
weight-stream — short of the 220-225 sibling class, consistent with the 8e
small-shape crossover data, but a solid −8.8 µs × 11 layers ≈ 97 µs/step).
Wall (interleaved A/B): @3254 8 pairs 21.605 → 21.72 (**+0.42%**, 7/7 clean
pairs positive, sign-test p≈0.008; strict SEP missed by 0.05% after
excluding one co-tenant rep — medians carried per the D3-5 precedent);
tg128 clean-window +0.26% (sub-bar; the tg128 extension window was
co-tenant-contaminated post-side and discarded). Landed on the @3254
evidence + measured mechanism; the session bar (+0.3%) is met on the median
standard.

Gates (D3a package): dump gate max|Δlogits_decode| 0.254 (calibrated
0.39-class), logits_prefill byte-identical, argmax HARD gate green at
margin 1.915; kv0 bitwise, kv1+ decode-side f16-boundary drift (earlier
onset than D3-6's sub-ULP reorder class — expected: input-quantization
noise ~4e-3 relative vs 1e-7 reordering); **penalty-free (rp=1.0) greedy
−n 256 streams byte-identical both models** (the D3-6 clean kernel gate);
default-penalty flips = ONE knife-edge event per 256 steps, 5/5 seeds,
coherent continuation ("near the" → "as the" mid-repetition — the penalized
argmax sits among repeated-token candidates); 14B temp-0.8 control
reorders (sampled top-p reordering is expected at a 0.22-logit drift; 7B's
is byte-identical).

**2c — rms/elementwise-launch consolidation, LANDED (two bitwise
sub-levers).** Census first (post-D3-5 14B @3254 decode step, decode-class
launches): rms_norm_quant_pad40 9.43 µs × 94.6/step = 0.892 ms;
f32_bits_to_i32 1.15 µs × 239.6/step = 0.276 ms; add_bias 0.181; store_kv
0.179; combine 0.171; rope 0.139; add 0.139; swiglu 0.110; standalone
quantize 0.084; dud split launch 0.070 — ≈ 2.2 ms/step of sub-6µs kernels.
The census also shows the 14B decode chain runs UNFUSED qkv (rope ×2/layer,
store ×2/layer, bias ×3/layer — FusedQKV is not active on CUDA; recorded as
a Stage-3 front item, ~0.45 ms/step).

- **(i) rms wide block**: `rms_norm_quant_pad40` launched at 128 threads
  (was one 32-thread warp per row). At hidden 5120 the 32-thread form is
  latency-bound: 40 serial float4 loads/lane with no other warp on the SM.
  The reduction is bitwise-preserved — lanes 0..31 keep the exact
  element→lane mapping, per-lane serial chains and `warp_reduce_sum` tree;
  scale broadcasts through smem; the write/quantize loops are
  element/per-32-block independent, so widening their mapping cannot move a
  bit; `#pragma unroll 8` on the reduce loop deepens load pipelining
  without touching the accumulation order. Kernel 9.43 → **5.66 µs** (−40%;
  the residue is launch ≈1.3 µs + the epilogue's q8 write + the w row
  stream). −0.35 ms/step.
- **(ii) positions_i32 memo**: every Rope (×2/layer), KvcacheStore (×2/layer)
  and Attn (×1/layer) node re-converted the same positions f32→i32 buffer —
  240 launches/step of pure redundant conversion. One-execution-window memo
  keyed (buf id, pool_gen), cleared in `synchronize` next to the MmqCache
  clear; capture-safe (the first consumer's conversion is the only one
  recorded; replay re-executes it every step). 239.6 → **1.2** launches/step;
  −0.28 ms/step.

Gates: bitwise by construction, proven end-to-end — dump gate with BOTH
sides under `MINFER_NO_KQ_MMVQ=1` (isolating 2b): 109/114 files
byte-identical, the 5 diffs = the documented pool-slot aliasing class;
logits both phases + all KV byte-identical; 7B greedy 5/5 seeds +
temp-0.8 control byte-identical. **Gate gotcha recorded**: the first
2c-only control ran `MINFER_NO_KQ_MMVQ=1` on the post side ONLY and showed
a fake 0.22-logit drift (prefill+decode, depth-independent) — the env var
also reverts the Q5_K decode arm (pre-existing 8e behavior), so the
"2b-off" control must set it on BOTH sides.

**2c results** (interleaved 3× medians, cumulative 2b+2c vs pre_d37, all
SEP strict): 14B tg128 23.31 → **23.73** (+1.80%), @3254 21.57 → **21.95**
(+1.76%); 7B tg128 49.95 → **50.47** (+1.04%), @1641 48.47 → **48.99**
(+1.07%). Guards all hold (14B tg128 ≥ 22.7; 7B ≥ 49.0 / ≥ 47.9). vs the
window anchors: 14B tg128 +1.59%, @3254 +1.39%. The 2c individual
contribution ≈ +1.4% (cumulative minus 2b's kernel-projected +0.21-0.3%),
matching the census projection (−0.62 ms/step). Suite green (170/0/3
standard, full log `/tmp/d3/suite_2b2c_full.log`).

**Distance to parity (post-D3-7, 14B @3254, window A/B numbers).** minfer
21.95 t/s = 45.56 ms/step vs llama 24.32 = 41.12 ms → **−4.44 ms needed
(−9.7%)** (vs −5.00 ms / −10.8% post-D3-6). Stage-3 front summary — what
remains, in order of size:

1. **Matmul aggregate** (~2.9 ms, the largest block): D3-1's
   wall-effective 194.9 GB/s vs llama's implied ~207.6 — a decode-GEMM
   program (q8_1-prologue fusion / llama-class MMVQ+GEMM rework) is the
   only lever class that reaches it.
2. **Launch structure** (~0.7-1.0 ms residual): the unfused decode qkv
   chain (rope ×2 + store ×2 + bias ×3 per layer ≈ 0.45 ms/step — FusedQKV
   is not active on CUDA and would need a CUDA `attn_bias_rope_store`
   kernel), the add_bias/add/store_kv/swiglu small-kernel tail, the D3-4
   dud split launch (~0.07 ms).
3. **Exposed-latency attention** (≤0.63 ms, mechanism uncertain): the
   bytes-side levers are dead (D3-6); smem-tile cooperative staging is the
   untried remainder, historically disfavored at this granularity (D2's
   cp.async negatives).
4. **Mechanism-less stragglers**: output head 200.1 GB/s (0.44 ms,
   D3-5 1b found no geometry knob), ffn_down-q6K 198.9 (0.49 ms).

7B decode remains closed (tg128 1.021× vs llama, @1641 0.992× — ahead/at
parity in this window).


### D3-8 — Stage 3 Tier A: the G4 FusedQKV decode fusion ported to CUDA, both layer classes LANDED (2026-09-08)

Base binary `/tmp/d3/minfer_pre_d38` (sha1 ccb7df8f, HEAD 81e5d6e). Window
anchors (pre_d38, 3× medians, this session): 14B tg128 23.81, @3254 22.04;
7B tg128 50.45, @1641 48.91 — the machine ran ~1.8% faster than the D3-7
window (co-tenant state), so all comparisons below are same-window
interleaved A/B plus a same-binary isolation A/B.

**The D3-7 finding**: the CUDA decode chain ran UNFUSED qkv — rope ×2 +
store ×2 + bias ×3 per layer ≈ 0.45 ms/step at 14B (D3-7 §2 census) —
because the CUDA backend did not advertise `Op::FusedQKV`. Metal has had the
G4 fusion since the graph era with bit-identical logits. This session ports
it and lands BOTH layer classes:

**Kernel** — `attn_bias_rope_store_f32` (+ `launch_attn_bias_rope_store`):
one launch computes bias+rope on q and k (neox pairs, math verbatim
`rope_f32`), bias on v (verbatim `add_bias_f32`), and stores k/v into the
persistent regions (verbatim `store_kv_f32`/`store_kv_f16`, `__float2half`
= RN matches the scalar tail). Pointer-form: q/k/v are section bases, so the
SAME kernel serves the concat layout (q=base, k=base+nqt, v=base+2·nkt) and
three separate buffers. `pos = positions[0]` is read device-side
(capture-safe; the graph's `positions_i32` memo supplies the buffer).

**Class 1 (concat, wq|wk|wv same quant type)** — the existing
`Op::FusedQKV` path: one concat matmul over the loader-registered
`blk.{i}.attn_qkv` rows + the fused epilogue. Bitwise argument, probe-proven
(`d38_probe_tests`): the decode MMVQ kernels map one 256-thread block per
row and dispatch on (ttype, id, nt) only, so a concat matmul's rows are
bit-identical to the three separate matmuls (14B od 7168 id 5120, 7B od 4608
id 3584, Q4_K); the epilogue is verbatim-reused math. Coverage: 24/48 layers
at 14B, 14/28 at 7B (the rest are mixed-quant).

**Class 2 (mixed quant, e.g. Q6_K attn_v among Q4_K q/k)** — new
`Op::QkvBiasRopeStore` + `QkvBiasRopeStoreMeta`: the three SEPARATE matmuls
(bias-free) followed by ONE epilogue launch on the three buffers. CUDA-only
(emission is `cfg(feature="cuda")` + CudaState-present gated; Metal keeps
the unfused chain for these layers, bitwise-neutral vs pre-D3-8). The
builder wires attention to the epilogue node, so q's matmul buffer has
exactly one consumer and the §5 in-place alias rule applies unchanged (the
allocator arm extends the Silu/RoPE family + `ensure_kv`; the backend falls
back to a D2D copy if the alias ever doesn't hold, mirroring the RoPE arm).
Scheduler resolves `kv_pair(layer)`; cpu_backend lists the op in its loud-Err
set. Decode coverage: 48/48 layers at 14B (24 concat + 24 epilogue), 28/28
at 7B — `MINFER_GRAPH_TRACE=1` matches the D3-7 GGUF census exactly.

**Mechanism (nsys, 14B @3254, NO_CUDA_GRAPH, same 16-decode-step trace)**:
total kernel launches 22098 → 17130 (**−22.5%**); per decode step −310:
add_bias −144, rope −96, store_kv −96, `attn_bias_rope_store` +48, q4_K
MMVQ −48 (the 24 concat layers go 3 launches → 1), standalone quantize +24
(the concat matmul's shared-A quantize; the D3-5 memo window doesn't cover
the new consumer pair). Net ≈ −0.9 ms/step wall (measured below).

**Gates.** (i) Probe tests (both green, both pointer forms, f32+f16 KV, 14B
+ 7B shapes): fused epilogue vs the unfused 7-launch chain bitwise on the
q/k/v sections AND the KV rows; concat matmul vs three separate matmuls
bitwise. (ii) Dump gate (`MINFER_GRAPH_DUMP`, 14B prompt_short + 7B
prompt_1k7, −n 4): `logits_{prefill,decode}` + ALL `kv{layer}_*.f32`
byte-identical pre-vs-post — 98/98 files at 14B, 58/58 at 7B. The
informational `node{N}_*` dumps are a documented instrument limitation, NOT
a gate: they read whole recycled pool slots whose dump-time content is
binary-layout-dependent (node3 differs within one binary; node5/8 differ
pre-vs-post even under `MINFER_NO_FUSE_QKV=1` where graphs AND loader
behavior are identical; prefill DOT graph is byte-identical, proving the
prefill topology unchanged). (iii) Greedy battery −n 256: 5/5 seeds × both
models byte-identical, rp=1.0 identical, `MINFER_NO_FUSE_QKV=1` control
identical, temp-0.8 sampled controls identical — **the only pre/post diffs
in the raw outputs are the perf-banner tok/s numbers** (a gate-script
gotcha: strip timing lines before comparing; the first "divergence at byte
~400" in every run was `1789.6` vs `1788.7 tok/s`). (iv) Suite 172/0/3
(D3-7's 170 + the 2 probes). Prefill perf untouched (pp3254 1833 t/s
pre-vs-post; the fusion is nt==1-gated at build time).

**Results** (interleaved 3-pair medians, every pair clean-separated):

| config | pre_d38 | post | Δ | isolation (post vs post NO_FUSE_QKV) |
|---|---|---|---|---|
| 14B tg128 | 23.81 | **24.55** | **+3.11%** | +3.15% (24.53 vs 23.78) |
| 14B @3254 | 22.04 | **22.40** | **+1.63%** | +2.28% (22.46 vs 21.96) |
| 7B tg128 | 50.45 | **50.98** | **+1.05%** | +1.03% (50.92 vs 50.40) |
| 7B @1641 | 48.91 | **49.51** | **+1.23%** | +1.00% (49.44 vs 48.95) |

Guards hold with margin (14B tg128 ≥ 22.7, @3254 ≥ 21.5; 7B ≥ 49.0 / ≥
47.9). The @3254 gain exceeds the +0.8% full-fusion bar on the isolation
standard (+2.28%) — the class-2 epilogue (the "tail lever") landed in the
same commit, so no separate tail-commit measurement window was needed.

**Distance to parity (post-D3-8, 14B @3254).** minfer 22.40 t/s = 44.64
ms/step vs llama 24.32 = 41.12 ms → **−3.52 ms (−7.9%)** (vs −4.44 ms /
−9.7% post-D3-7). **14B tg128 crosses parity in this window: 24.55 vs llama
24.31 (1.010×)**; 7B stays ahead (tg128 1.032×, @1641 1.002×). Remaining
14B @3254 inventory: matmul aggregate ~2.9 ms (a decode-GEMM program),
exposed-latency attention ≤0.63 ms (mechanism uncertain), output head 0.44
ms, ffn_down-q6K 0.49 ms, launch-structure residue (add/add/swiglu tail,
dud split ~0.07 ms) — the qkv-chain item is CLOSED.

### D4-2 — Decode-GEMM Tier B: a silent 7B correctness bug found and fixed (B0), the bitwise occupancy/prefetch axes closed with mechanisms (2026-09-09)

Session against the D4-1 census (`/tmp/d4/D4_DESIGN.md`: minfer's decode GEMV
aggregate is already 0.85 ms AHEAD of llama; the gap is attention structure
2.53 ms + q6_K occupancy + tail). Budget ~3 h, r44/r46 discipline. Outcomes:
one critical correctness fix landed (`b31084c`), four levers closed with
measured mechanisms and zero perf regressions, the @3254 gap decomposition
re-confirmed for the D4-3 brief.

**B0 — the headline: 7B ffn_down-q6K decode was silently wrong since
D3b-1b.** `q6_k_q8_mmvq_v2_pf` processes exactly two units per thread
(u = tid, tid+256 → covers npair ≤ 512), but its dispatch gate was only
`id > 8192` with no upper bound. Qwen2.5-7B ffn_down (10 q6_K layers,
id 18944, npair 592) therefore dropped units 512..591 — 13.5 % of every
down-projection row, every decode step, for the whole D3b-1b→D4-2 period.
Why every gate missed it: (a) the argmax at the gated prompt survived (the
first-decode-step top-1 was unchanged; logit corruption max|Δ| 4.79 shows up
only in the dump), (b) every pre-vs-post gate compared v2_pf-to-v2_pf
binaries — the bug was on BOTH sides. The fix bounds the pipelined route to
`8192 < id ≤ 16384`; taller rows take the v2 loop form (identical per-unit
arithmetic, ascending-u order — D3b-1b's bitwise invariant). Gates: 14B
(npair 432, unaffected) pre-vs-post `MINFER_GRAPH_DUMP` 107/107 gate files
byte-identical (7 node{N} diffs = the documented pool-slot instrument class)
+ greedy byte-identical; 7B first-decode-step (−n 1) logits post-fix vs the
full-coverage v1 reference differ only at the v1-vs-v2 rounding class
(max|Δ| 0.254 — the same class 14B's own v2-vs-v1 shows, 0.231; argmax
identical) while pre-vs-v1 differs by 4.72 (the bug); 7B greedy now follows
the v1 reference stream; suite 173/0/3.

**Why the bug was invisible in the KV/logit dumps, and the instrument
lessons.** Comparing full-run dumps across binaries whose generated tokens
diverge is invalid twice over: (1) after the first divergent token the two
runs feed different contexts — later-step logits/KV differences are cascade
noise (measured up to 1e18 "deltas" that are NOT compute corruption); (2)
the persistent KV regions' unwritten tail slots contain pool garbage whose
layout is binary-dependent. The clean comparable point is the FIRST decode
step (`-n 1`: prompt-only context, identical inputs). Also recorded: the
v1-vs-v2 MMVQ rounding class (max|Δlogit| ≈ 0.23-0.25, argmax-preserving at
the gated prompt) is pre-existing (R2-era) — v1 is not a bitwise reference
for v2, only a semantic one.

**Lever A — llama GB10 L2 weight prefetch: closed PRE-BUILD.** llama's
prefetch distance is 2 loop iterations = 2·blocks_per_iter with
blocks_per_iter = vdr·nwarps·32/qi; from vecdotq.cuh: VDR_Q4_K_Q8_1_MMVQ=2 /
QI4_K=16 and VDR_Q6_K_Q8_1_MMVQ=1 / QI6_K=8 → 8 threads per 256-elem block
and bpi = 4·nwarps = 16 (GB10 ncols_dst=1 nwarps 4). The prefetch fires only
where bpr > 2·bpi = 32 blocks: at 14B that is ONLY ffn_down (bpr 54); every
id-5120 shape (gu, qkv, q/o, lm_head, attn_v; bpr 20) never prefetches. Our
kernels map one 64-element unit per thread over 256 threads (npair =
4·bpr: 80 gu/qkv/qo, 216 down-q4K, 160 lm_head/attn_v; v2_pf's 432 units are
unrolled as u0/u1) — exactly ONE K-loop iteration at every 14B/7B decode
shape, so a faithful port is runtime-dead code. And where llama's prefetch
does fire, its ffn_down-q4K rate (224.1 GB/s) is BELOW our unprefetched
228.6 — the mechanism buys llama nothing net. Both families sit at 75-84 %
of GB10's DRAM peak: a bandwidth-saturated regime where L2 prefetch adds no
bandwidth. No build; no commit (D4-1 §3 precedent).

**Lever B — the bitwise occupancy axes, all measured closed.** Fresh ncu
(2025.3.1, sudo recipe, 14B @3254): v2_pf holds 48 regs/thread →
Block-Limit-Registers 5 blocks/SM vs Block-Limit-Warps 6 (the at-class q4_K
v2 and v2 both sit at 40/39 regs and 6 blocks). Candidates:

| variant | change | result |
|---|---|---|
| B1 | `__launch_bounds__(256,6)` on v2_pf (48→40 regs, funds the 6th block) | SASS: REG 40 + **STACK 40** (10 spilled words); probe: tg128 +0.25 %, @3254 **−0.75 %** (22.54→22.37 medians) → **KILLED** — the spill costs more than the 6th block pays |
| B1c | route npair-432 to the v2 loop (`MINFER_Q6K_PF=0`): 6 blocks × 1-unit-MLP vs 5 blocks × 2-unit-MLP | v2_pf wins/ties (tg128 24.08 vs 24.02/24.08; @3254 medians 22.28 vs 22.16, one noisy pair each) → **default kept**; env kept as the documented opt-out |
| B2 | v2 right-size to 160 threads (npair 160; zero idle warps, 9 resident blocks × 5 active warps vs 6 × 5) — a **repeat of D3b-1c**, re-measured with per-kernel isolation | bitwise 98/98 gate files, but nsys: lm_head 3243.7 → 3292.7 µs (**+1.51 %**), attn_v 24.41 → 25.15 µs (**+3.04 %**), v2_pf/q4_K rows unchanged (±0.6 % control) → **KILLED** |

B2's prior art was in the §0 table all along (D3b-1c: "9 blocks×160 live
threads ≈ 6×256 allocated — the idle-thread win does not exist"; D3-5 1b:
"occupancy is not the limiter") — it should have been checked BEFORE
building; the re-measurement at least upgrades "neutral" to "measured worse
with a clean per-kernel control". Process lesson: grep the master table for
prior art on the exact mechanism before writing a line of kernel code.

Per the brief's stop rule: the q6_K occupancy gap cannot be closed bitwise
on this kernel family; the limiter record stands (v2_pf is register-funded
for its dual-unit pipeline and register-capped one block below the warp cap;
shaving spills, un-funding the pipeline loses more, and block geometry is
already at its measured optimum). Tolerance-gated redesigns remain out of
scope this session.

**Lever C — PDL: documented as follow-up, not built** (time + CUDA-graph
capture interplay risk; llama launches every decode kernel with
`cudaGridDependencySynchronize` + programmatic-stream-serialization — the
graphs-banked ~0.3 ms residue is the target, and a graph-capture-safe PDL
integration wants its own session).

**Measurements (interleaved 3× medians, pre = `b31084c` binary, post =
final tree):**

| config | pre (B0) | post (final) | Δ | note |
|---|---:|---:|---:|---|
| 14B tg128 | 24.13 | 24.01 | ≈ 0 (noise; pair-3 straddle) | fix-only tree behaves identically, as designed |
| 14B @3254 | 22.12 | 22.42 | ≈ 0 (noise) | guards hold (≥ 24.0 / ≥ 21.9) |
| 7B tg128 | 49.21 | 49.29 | +0.2 % | guard re-anchored below |
| 7B @1641 | 48.41 | 48.50 | +0.2 % | guards hold |

**7B fix re-anchor (pre-D4-2 buggy binary vs the fixed B0 binary, 3×
interleaved medians):** tg128 **50.67 → 49.30 (−2.7 %)**, @1641
**49.84 → 48.49 (−2.7 %)** — both shapes exactly −2.7 %, the honest cost of
decoding correctly (13.5 % of the down-q6K work the bug skipped; down-q6K ≈
20 % of the 7B per-step stream). The old 7B guards (49.0 / 47.9) were set
on the dropping kernel and are superseded: **7B guards tg128 ≥ 48.3 /
@1641 ≥ 47.5** on the fixed engine. 14B buggy-vs-fixed crosscheck: medians
straddle inside window noise (bitwise-identical binary paths — the fix does
not touch npair ≤ 512 dispatch).

**Same-window vs-llama table (llama-bench `ca3d5a3e1`, 2026-09-09 window,
matched -n 128 anchors):**

| config | minfer (post-D4-2) | llama | ratio |
|---|---:|---:|---:|
| 14B tg128 (KV≈0) | 24.01 | 24.14 ± 0.02 | **0.995× (parity)** |
| 14B @3254 | 22.42 | 24.14 (tg128 @ -p 3254) | **0.928×** |
| 14B pp3254 | ~1816-1859 | 1634 ± 112 | **1.11-1.14×** |
| 7B tg128 (KV≈0) | 49.30 | 47.65 ± 0.09 | **1.035×** |
| 7B @1641 | 48.49 | 47.69 ± 0.01 | **1.017×** |

Note: D4-1's 0.921× headline compared minfer tg64-exclusive against llama
tg8 in the sglang-co-tenant window; at matched tg128 anchors the 14B
short-KV decode is at parity, and the 14B long-KV gap (0.928×) is the
attention-structure item below. The 7B ratios are on the CORRECT engine
(pre-fix numbers were flattered by the dropped units).

**Updated @3254 gap decomposition (for the D4-3 brief).** The census
corrections this session: (1) the q6_K "205 GB/s" census rate counted
210-byte blocks while the padded stride streams 224 — the DRAM-true q6_K
rates are ≈ 218.9 (ffn_down), 221.6 (lm_head), 189.5 (attn_v) vs the q4_K
class 228.6, so the true remaining q6_K-vs-class deficit is ≈ 0.42 ms/step,
not 1.1-1.2; (2) the bitwise occupancy axes are closed (above), so that
0.42 ms needs a tolerance-gated redesign (or the attention-structure lever)
to reach; (3) **the attention-structure ~2.53 ms item is untouched and
remains the D4-3 prize** *(superseded by the D4-3 chapter below: ~1.6 ms
honest — the 2.53 ms was anchored on a llama-bench artifact)*; (4) 7B
decode numbers before `b31084c` are not
comparable to anything after it.

### D4-3 — attention-structure attempt 2 (llama fattn-vec geometry rewrite): NO-GO per the pre-registered bar, and the D4-1 llama target is a llama-bench artifact — the honest llama decode attention is ~2× off, not 4.9× (2026-09-09)

Session against the D4-1 census (brief: the traced llama attention
"6.94 µs split + 7.20 µs combine = 14.02 µs/layer" at 14B/@3254 implies
~9.6 TB/s effective L2 read vs our kernels' 1.0–1.4 TB/s; prize ~2.3–2.5
ms/step). Pre-registered GO bar: a llama-structured kernel (pb splits ×
windows × subgroup lanes, Q-in-regs, direct-global K/V, shfl-broadcast
probs — no smem in the hot loop) must show **≥2× kernel-total vs the
current dispatch kernel at BOTH 14B/@3254 (≤~32 µs) and 7B/@1641
(≤~10.35 µs)**; probe-first with hard go/no-go, tolerance-gated
integration only on GO.

**Probe** (`/tmp/d4/probe_attn2.cu`, nvcc -O3 -arch=sm_121a): verbatim
incumbent bodies (`attn_split_1w_body`, `gqa_attn_split_partial_hybrid`,
`gqa_attn_split_combine`) as baselines + a `vec_attn<T,R,MINB,STG>`
llama-geometry kernel swept over pb∈{2,4,8,16,32}, threads∈{32,64,128},
window R∈{32,64,128}, minb∈{1,4,8}, and a load-scheduling axis
STG∈{0,1,2} (0 = chunked staging CS=8 rows; 1 = whole-window K+V
staging; 2 = STG1 + next-window K prefetch). Timing batch-of-32 min-of-8
with 2 rotating L2-hot KV copies; CPU reference in double; correctness
gate 0.05 abs with adversarial outliers (q ±57 every 911th, v ±138 every
1543rd). **690 sweep rows, 0 correctness skips** (`/tmp/d4/d43_sweep5.csv`).
Two probe bugs found and fixed during bring-up (subgroup row-map missing
the warp stripe offset; a refactor dropped the chunked V-pass's initial
staging) — both were correctness-FAILs caught by the gate, not silent
numbers.

**THE DECISIVE FINDING — the D4-1 llama target is an artifact.** ncu on
the traced binary's own decode `flash_attn_ext_vec<128,1,F16,F16,false>`
at 14B:

| capture | grid | SASS global-load bytes | LDG/warp | KV rows covered |
|---|---|---|---|---|
| **llama-bench** decode, KV≈1024 / 2474 / 3255 | (1,**2**,40) | **5,427,200 — byte-identical at all three** | 44 ≈ one 128-row KV iteration (K 16 + V 16 + Q 4 + mask 8) | **2 splits × 128 = 256 of 3255 (7.9%)** |
| **llama-cli** decode, KV≈2477 (`-c 2610`) | (1,**7**,40) | **53.2 MB** (≈ the 50.7 MB full-coverage arithmetic) | — | full |
| **llama-cli** decode, KV≈4869 (`-c 0`) | (1,**7**,40) | **142.7 MB**, scales with ctx | — | full |

The bench-path kernel's load traffic is **context-independent** (68
KB/block whether KV is 1k or 3.3k), i.e. its 2-split grid runs ~one
128-row KV iteration per block and covers only the first 256 rows. The
CLI path picks 7 splits and full coverage. End-to-end A/B: a mid-prompt
secret (KESTREL-5150 at ~row 1200 of 2470) is recalled correctly by
llama-cli at both `-c 0` and bench-like `-c 2610`; the bench's tg output
is never validated by llama-bench. Reductio on the D4-1 reading: 48
layers × 66.8 MB in 0.333 ms/step = **9.6 TB/s L2** — beyond the fabric;
the traced kernel's own 5.43 MB × 48 = 260 MB at 0.333 ms = 0.78 TB/s —
exactly the latency-bound class everything else lives in. (Why the bench
host picks pb=2 where ca3d5a3e1's occupancy loop would pick 7 is not
pinned — the observed facts above are unambiguous and reproducible; the
binary is `b10665-ca3d5a3e1` per its own banner.)

**Recalibration.** llama's honest full-context decode fattn-vec at 14B
runs at ≈ **2.0–2.1 TB/s effective** (53.2 MB/60 µs-ncu ≈ 25 µs live at
KV 2477; 142.7/171.5 ≈ 70 µs live at 4869), i.e. ~33–36 µs/layer at
KV 3254 including combine ≈ **1.7 ms/step, not 0.69**. minfer's current
3.32 ms/step is **~1.9× off, not 4.9×**, and the honest llama-vs-minfer
@3254 wall gap shrinks from 13.4% to ~10%, of which attention is ~3.5%.
The D4-1 "attention is the 14B decode gap" story was mostly the artifact.

**Sweep result (all correctness-gated; split+combine totals):**

| shape | best vec geometry | split | combine | total | probe baselines (1w/h4w) | in-situ dispatch today | GO bar |
|---|---|---|---|---|---|---|---|
| 14B @3254 | pb=4 T=128 R=64 STG=1 | 38.89 | 2.17 | **41.07** | 49.74 / 55.26 | 67.4 (63.8+3.6) | ≤~32 → **FAIL (1.64×)** |
| 7B @1641 | pb=16 T=32 R=32 STG=1 | 12.17 | 4.04 | **16.22** | 20.45 / 32.78 | 24.8 (20.7+4.1) | ≤~10.35 → **FAIL (1.53×)** |
| 7B @512 | pb=4 T=128 R=32 STG=1 | 8.04 | 2.12 | **10.16** | 11.61 / 24.59 | ~11.8 | — |

The geometry plateau is flat (pb 2–8 within 1.5 µs at 14B) and the STG
axis is inside noise at the plateau — once loads are batched per window,
more scheduling depth buys nothing at these block counts. The probe's
transient headline (interleaved load→dot→shfl per row ≈ 1.0 TB/s,
100 µs @14B) vs staged (1.7 TB/s) is the transferable mechanism: **SASS
load batching is the whole game at ≤2 blocks/SM** (llama's SASS batches
~21 LDGs ahead; ptxas does NOT do it from a natural loop shape — it took
explicit staging loops).

**GO/NO-GO arithmetic:** best kernel-total 41.07 vs bar ≤32 @14B (1.64×
vs current dispatch), 16.22 vs ≤10.35 @7B (1.53×) — **NO-GO on the
pre-registered bar**. In-situ projection via the probe→in-situ factor
measured on the h4w baseline (63.8/51.2 = 1.25): 14B ≈ 48.5+3.6 ≈ 52 µs
→ +0.72–0.82 ms/step ≈ **+1.6–1.8% wall < the +2% integration bar**; 7B
≈ +0.7%. Both bars fail → **no integration, line closed.** The honest
remaining attention headroom (~1.6 ms/step at 14B) would need ~2.1 TB/s
— llama's own rate — i.e. roughly the pb=7 (280-block) geometry plus the
staged-load recipe beyond what the sweep plateau shows; that is a fresh
session with a re-anchored bar, not a D4-3 continuation.

**Verdict: D-series attention line closed as measurement-corrected.**
D4-1's census stands (the traced kernels DID run 48×6.94+48×7.20 µs and
the books close); what was wrong was reading those kernels as
full-context attention. Actionable carry-forward: (1) never take
llama-bench long-ctx tg rates as attention targets — validate with
llama-cli recall or ncu byte counts; (2) the honest 14B decode gap to
llama is ~10% wall with ~3.5% attention; (3) any future attention
session starts from the (1,7,40)-class geometry + explicit load staging
and a ≤25–32 µs @14B bar.

## §3 Appendices

### Appendix A — Env-gate reference (post-r60 semantics)

**Promoted gates (r60): absent / any non-`"0"` value = ON (the verified best
path); explicit `"0"` = opt-out (the pre-r60 default behavior).** All reads
single-sourced in `CudaState::mmq_gate_on`.

| Gate | Default | `"0"` effect | Introduced |
|---|---|---|---|
| `MINFER_MMQ` | on | whole MMQ path off → legacy f16 w16-cache prefill (~20.5 GB, ~2353-class) | R1 (opt-in), r60 (default-on) |
| `MINFER_MMQ_RAW` | on | raw-byte kernels off → generic `mmq_nt` arms | r7–r8, r60 |
| `MINFER_MMQ_RAW_NB` | on | NB kernels off → wide raw kernel | r28, r60 |
| `MINFER_MMQ_A_TRANSPOSE` | on | pre-transposed prepass off → native quantize + NB kernel | r34, r60 |
| `MINFER_MMQ_Q6K_NB` | on | q6_K BT kernel off → generic q6_K path | r38, r60 |
| `MINFER_MMQ_A_FUSE` | absent = mode 2 (skip-write) | off (mode 1 = "1": fused producers write plane AND f32; "2" = skip-write override) | r51/r52, r60 |
| `MINFER_MMQ_Q6K_EXP` | on | q6_K W_exp plane not built (−1.52 GB, ~−5% prefill) → EXP=false r41 path | r54 |
| `MINFER_MMQ_Q4K_DSC` | on | q4_K/q6_K W_dsc planes not built (−1.46 GB) → in-kernel decode | r59 |

**Overrides / debug (opt-in "1"):** `MINFER_MMQ_RAW_KD` (default 8),
`MINFER_MMQ_RAW_WIDE` (wide 128×128 kernel), `MINFER_MMQ_RAW_NB_DEBUG`
(prints the dispatch label — `B=W_exp-cp.async` / `DSC=f32-plane` /
`exp=off` / `fallback!` — the liveness instrument from r53/r54).

**Legacy f16-path A/B gates (unchanged):** `MINFER_NO_PREFILL_GEMM`,
`MINFER_NO_W16CACHE`, `MINFER_NO_FA_PREFILL`, `MINFER_NO_CUDA_GRAPH`,
`MINFER_NO_PREFILL_CAPTURE` (`MINFER_CAPTURE_PREFILL=1` accepted, redundant),
`MINFER_NO_PINNED_READBACK`, `MINFER_MMVQ_V1`, `MINFER_NO_KQ_MMVQ`,
`MINFER_GEMM_TM` (64), `MINFER_GEMM_K64`, `MINFER_FUSED_B`.

**Dispatch guards (unchanged by r60, now protect the default path):** MMQ
entry `nt >= 16 && id % 32 == 0 && !no_prefill_gemm`; NB-BT
`(id / 32) % 8 == 0`; plane registration `id % 256 == 0` (+ `od % 2 == 0`
for dsc); fused producers `rows >= 16 && dim % 256 == 0`; mode-2 auto-degrade
under MINFER_GRAPH_DUMP / MINFER_DUMP_DIR / MINFER_TRACE / viz capture; the
r60 `nb_bt_only` flag degrades mode 2 → mode 1 on mixed-quant models.

### Appendix B — Verification methodology and transferable lessons

**The gate chain (every landed round).**

1. **Parity ×3**, separate invocations under the candidate gate set:
   `cuda_prefill_mmq` (1/0 — the 8-type × 8-shape host-reference sweep),
   `cuda_prefill` (7/0), `cuda_fa_prefill_attention_parity` (1/0). Tolerance
   1e-3; a nibble-layout bug presents as ~1e0, f32 rounding as ~1e-5.
2. **Greedy-32 token identity**: `-n 32 --greedy --seed 42` on prompt2k vs
   the pre-change binary — byte-identical streams. Catches corruption parity
   fixtures miss (r52's rms-nw OOB, r58's buffer-1 smem). FA exception:
   tile-size changes inherently shift accumulation order (r50/r57).
3. **Interleaved A/B medians**: 3×/5× same-slot pairs, warmup, alternating
   order; distributions must separate (min-new > max-base) for a headline;
   the +1.5% whole-prefill bar (relative to the re-measured baseline, r24).
4. **Suite**: 166 → 169 passed / 0 failed / 3 ignored (grew with gate-1
   byte-exactness tests); co-tenant flakes re-run `--exact` in isolation.
5. **ncu/nsys protocols**: ncu behind `sudo -n env LD_LIBRARY_PATH=...`
   (plain sudo strips gate env and silently profiles the legacy path —
   r56); GB10/GB20B has NO `dram__*`/`launch__grid_size`/shared-sector
   metrics — use `lts__t_sectors_aperture_device` and byte-count-derived
   rooflines (r55); ncu serializes replay — nsys is authoritative for
   walls; PC-sampling (`--page source`) attributes stalls to the CONSUMING
   instruction (r20/r43); SASS via `cuobjdump -sass` before writing levers
   (r30/r32) — CP.ASYNC is emitted as `LDGSTS.E.BYPASS.128` (grep LDGSTS,
   r45); ptxas `-Xptxas -v` for regs/spill/occupancy budgets.

**Transferable lessons (the campaign's durable rules).**

- **Baseline-anchoring (r59b)**: every A/B baseline must be behaviorally
  anchored in the same window (re-measure a known-record binary or
  worktree-rebuild the baseline commit); idle co-tenancy is
  clean-equivalent; never infer a co-tenant tax without an anchor.
- **Liveness check (r53/r54)**: a fallback-correct optimization needs an
  "is the fast path actually live" counter/label — parity and greedy cannot
  see a silently-never-taken fast path; distinguish intentional fallback
  (`exp=off`) from accidental (`fallback!`).
- **Tile-size vs greedy-identity (r50/r57)**: on FA (and any
  accumulation-order-sensitive kernel), strict byte-identity is only
  satisfiable for changes that preserve accumulation order — every tile-size
  change breaks it by ULP regrouping, even when parity stays green.
- **Pipeline-value formula (r58, the r45 mirror)**: a staging mechanism's
  wall value = what it removes MINUS what its granularity costs. It replaced
  expensive work (q6_K r39/r53/r56) → +13/+5/+2.35%; it replaced cheap
  copies (q4_K r58) → −12.6%. "Not dead, WAITING" applies only when the
  mechanism's cost model is preserved.
- **Roofline-bound-before-coding (r55)**: derive the byte-traffic bound
  first (sector counters × 32 B when `dram__*` is absent); if the perfect
  kernel cannot clear the bar, skip the implementation.
- **Issue/occupancy, not instructions (r13→r25→r28/r29)**: instruction-count
  surplus is real but wall-inert at 1 block/SM; buy occupancy first, then
  the same cuts pay (+2.6/+2.8%). At 3 blocks/SM a 4 B spill is immaterial
  (r40).
- **The compiler already did it (r30/r32/r33)**: read the SASS before
  writing a lever; source reorders that ptxas already schedules are
  SASS-identical no-ops.
- **Stall mass is conserved (r20/r21)**: fixing one binder moves the stall
  to the next (latency → lg_throttle → wait); stage levers in that order.
- **Layout-transformation locality (r34)**: hoist layout transformation into
  a prepass (or the producer) instead of adapting per tile-consumer —
  +9.72% for moving the transform OUT of the kernel.
- **Mechanisms compose across traffic, not mechanism (r53/r56)**: two
  individually-wall-neutral levers (one removes WORK, one removes WAIT)
  compose almost additively when the first de-bottlenecks the line.
- **Attribute to the consumer (r42/r43)**: stall counters name the resource,
  PC-sampling names the instruction that WAITS on it — cut latency where it
  is exposed, not where bytes move.
- **Wall decompositions expire (r37→r47)**: re-attribute the whole wall
  after every line converges; net hidden taxes (the q6_K prepass +31.6 ms)
  against wins.
- **Phantom results (r8, P5·3, r52a)**: silent fallbacks / attr failures /
  OOM-masks produce fast wrong timings and fake errors — guard caps loudly,
  verify launches happened.
- **Memory etiquette (shared box, 2026-08-31)**: kernel OOM under pool
  exhaustion kills the OTHER workload (it happened once); no raw allocation
  probes; check `free -g` before suite runs; single-process 7B-scale benches
  while sglang serves.

### Appendix C — Part IV legacy: the pre-Phase-7 roadmap (2026-08-29) and where it ended

Everything below described the deleted imperative path (`layer_gpu`,
`forward.rs`) on an RTX 4080 Laptop with Qwen2-0.5B Q4_0: prefill 40, decode
20 tok/s (CPU 18/15). Kept for the record; outcomes annotated.

**Root cause as diagnosed then: per-op CPU↔GPU ping pong.**

```
CPU path → quantize f32→Q8_0 (CPU) → cudaMemcpy H2D → CUDA kernel → sync → cudaMemcpy D2H → CPU path
```

~6 PCIe round trips × 24 layers ≈ 144 DMA operations per decode step, 2–7 ms
of pure overhead. (Correct for that path; Phase 7's resident-weight graph
backend eliminated it structurally.)

**Original P0–P5 and actual outcomes:**

| Item | Claim then | Outcome |
|---|---|---|
| P0 full-layer GPU offload | add Q4_1/Q8_0 kernels, kill 144 DMAs, 3–4× | Absorbed by Phase 7 graph backend (weights resident, per-op dispatch, split syncs) |
| P1 GPU-side activation quantize | GPU q8_0 kernel unused, 1.2× | Landed as 8c with a measure-first gate (`nt>1 && id≤8192`); the q8_0 path LOSES 63% at 7B ffn_down (weight-bound) — a blind wire would have regressed |
| P2 fused GQA on GPU | wire `gqa_attn_f32`, 1.5× | Landed in Phase 7a/7e (`gqa_attn_f32_f16kv`); prefill attention replaced by FA tiling (8n: 20×); decode attention is 0.12 ms/token at 2K — no longer material |
| P3 cuBLAS for output projection | `cublasSgemm` "leverages tensor cores", 2× | Closed as 8k (not planned). Two errors: `cublasSgemm` is FP32 SGEMM — tensor cores require cublasGemmEx with f16/int8; and the need disappeared once 8m's custom wmma GEMM covered large matmuls |
| P4 tiled quantized matmul | llama.cpp MMQ "shared-memory tiling with Stream-K decomposition", 1.5× | First judged negative, then REVERSED same-day (8e): the real design is integer `__dp4a` dots over q8_0 activations with a per-CC launch table — ported in-tree as the decode MMVQ win; "Stream-K" was never part of llama.cpp's MMQ. The prefill int8 version became R1 |
| P5 CUDA graph for launch overhead | capture decode, 1.2× | Landed as Phase 7d decode capture/replay (one ~57 µs graph launch per token) + opt-in prefill capture (8g②, default since R3-B). The original "2,000+ launches (…× ~14 heads)" miscounted — heads don't multiply launches; the true figure is ~95 nodes/layer × 28 layers ≈ 2.7K, same order |

**Implementation order as drawn then:**

```
P0 (layer_gpu) ─→ P1 (GPU quantize) ─→ P2 (GPU attention)
                                      ↘
                                       P3 (cuBLAS) ─→ P4 (tiled MMQ) ─→ P5 (CUDA Graph)
```

All six landed in some form by 2026-08-31 — none via its original mechanism
except P0's idea. Measured budgets recorded at the Part-III era: f16 wmma
GEMM ~35 TFLOPS (llama.cpp int8 MMQ ≈ 52 equivalent); MMVQ weight streaming
130–147 GB/s effective vs the 252.7 GB/s read-only probe (93% of the
273 GB/s theoretical); llama.cpp ~197 GB/s on the same decode shape.
