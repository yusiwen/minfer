# CUDA Inference Path — Optimization History and Current State

> **STATUS (2026-09-06, post-r60): history-organized reference.** This document
> was restructured from a part-based roadmap into a history-ordered record:
> §0 is the master history table (the outline — every landed, reverted, or
> measured lever with its commit and perf delta), §1 is the current state,
> §2 is one chapter per table row, and §3 holds the appendices (env-gate
> reference, methodology, and the pre-Phase-7 legacy history). Single-sourced
> implementation records: `docs/CUDA-BACKEND-PLAN.md` (Phase 7a–7e) and the
> per-step documents of §2 (Phase 8: docs 01–11 plus the supplementary records
> 78–79 — the former `CUDA-FOLLOWUP-PLAN.md` was consolidated into them and
> retired on 2026-09-10); the per-round MMQ redesign records are mirrored in
> `docs/LLAMA-CPP-MMQ-ANALYSIS.md` §11.
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
| 8b | KV f16 on CUDA (`store_kv_f16` + f16-KV attention mirror) | `f7b0036` | 7B @2K decode +~11% | +11% | — | LANDED | auto-f16 at n_layers×n_kv_embd ≥ 8192 (`MINFER_CACHE_TYPE` override); f32 accumulation |
| 8c | prefill Q8_0-activation GEMM, shape-gated | `69a27c5` | 0.5B @3.6K 1005 → 1246 tok/s | +24% | — | LANDED | the shape gate is load-bearing: −63% at 7B ffn_down (weight-bound shapes stream slower) |
| 8d | split-K flash-decoding decode attention | `a5af60f` | 7B @2K decode 10.1 → 13.7 tok/s | +36% | — | LANDED | 28 warps → an 8-way KV-split scan; superseded by R4's dim-parallel rewrite |
| 8f | Q5_K + Q5_1 f32-activation kernels | `b959ec9` | 0.5B q5_k_m admitted to CUDA (was CPU wholesale) | — | — | LANDED | the all-or-nothing gate needs a kernel for EVERY matmul weight type |
| 8e/8e② | decode MMVQ (dp4a, per-type kernels, shape gate) | `b7b8e73`, `1298cb2`, `1d28235` | 7B decode +37% (q4_K), then q6_K/q5_K | +37% | — | LANDED | integer dp4a dots + llama.cpp `MMVQ_PARAMETERS_GB10` launch table |
| 8l | llama.cpp parity benchmark (the decode + prefill gap sheet) | `acca28f` | decode 1.15–1.76×, prefill 18–110× | — | — | MEAS-ONLY | found the Q5_K registration gap (51.6 → 246.3 tok/s, 4.8×); the sheet ranked 8m–8p / R1 / the r-campaign |
| 8q | Q5_0 CUDA enablement + u16 qh loads | `9f419f9` | 0.5B q4_k_m 148.7 → ~1200 prefill / 56.9 → ~306 decode | — | — | LANDED | a 22-byte block's qh word is not 4-byte aligned — two u16 loads; the CPU fallback eliminated |
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
| D3a | 4-warp fattn-vec-style split-attention rewrite (`gqa_attn_split_partial_h4w`, hd=128: 128 threads, K/V streamed, Q in registers, 8-lane subgroups, 32-row windows/warp, smem LSE merge; grid unchanged, replay-safe) | reverted (patch `/tmp/d3/d3a_kernel_patch.diff`, findings `/tmp/d3/D3A_FINDINGS.md`) | kernel 14B @3254 73.4 → 68.4 µs (−6.9%) but 7B @1641 21.1 → 34.7 µs (+64%); wall 14B @3254 −0.95% (noise), 7B @1641 −3.4% (real), tg128 noise-level | — | — | REVERTED | rows-per-warp pathology: rpw = ceil(ceil(nkv/32)/4) = 26/13/1 at @3254/@1641/tg128 — the 32-row window idles 59–75% of lanes below rpw≈16 and per-block fixed costs amortize over rpw; kernel −6.9% at the best shape is only ~+0.5% wall (attention = 7.4% of the step), under the +1.5% bar and the ±2% A/B noise; numerics fully green (probe ≤1.3e-7 vs CPU, argmax hard-gated, greedy 0/10 diverged) — the session's durable output is the tolerance-gate calibration: end-to-end max\|Δlogits\| is 0.38/0.39 (14B/7B) for ANY accumulation-order change, so the D3-1 ≤1e-3 logits gate is unsatisfiable; argmax+greedy+A/B are the operative gate set |
| D3-4 L1 | hybrid rpw dispatch: dual-kernel self-gating split attention (f16 KV, hd==128) — 4-warp h4w kernel when rpw = ceil(ceil(nkv/32)/4) ≥ 16 (nkv ≥ 1921), incumbent 32-thread kernel below; BOTH launch per layer with static grids, each re-reads `positions[0]` per replay, exactly one is live per nkv (nkv-uniform branch → replay-safe) | `22336b2` | 14B @3254 split 72.1 → 62.1 µs (−13.9%) + 1.5 µs dud launch; wall 14B @3254 21.20 → **21.33** (+0.61%, SEP), tg128 22.96 → 22.94, 7B tg128 50.28 → 50.20, @1641 48.78 → 48.68 (guards hold) | **+0.61%** (14B @3254) | 0.944× / 0.877× (14B tg128/@3254 vs llama 24.31/24.32) | **LANDED** | 1-warp path bitwise (7B @1845 dump: all gated files identical; `node{3,5,8}_prefill` diffs = pre-calibrated slot aliasing); h4w tolerance class (max\|Δlogits\| 0.309, argmax identical margin 0.716, 1 greedy flip at the regime entry = 1/256 < 2%, temp-0.8 controls identical); suite 169/0/3 incl. the hd=128/n_ctx-4200 parity shape sweeping the rpw 15/16 boundary; an in-kernel 1-warp-fallback form was REJECTED pre-commit: inside 128-thread blocks the 1-warp body caps at 12 working warps/SM (1536/128) = **+78% kernel at 7B @1641** (35.4 vs 19.8 µs nsys) — geometry, not math |
| D3-4 L2 | window-level K/V prefetch pipelining in the h4w body (K-pass software pipeline +2 uint4, 4-deep V-bulk ring +8 uint4; issue-point-only → bitwise vs h4w by construction) | reverted (patch `/tmp/d3/patch_l2.py`) | 14B @3254 h4w 62.1 → 66.5 µs (**+7%**) | −7% kernel | — | REVERTED | the kernel is bytes+tail-bound at 79% of the 48.9 µs floor, not chain-bound enough: funding the pipeline buffers needs `__launch_bounds__` minBlocks 8→4 (64→128 regs) → occupancy 32→16 warps/SM and 3.33→4.44 waves — the occupancy/wave-tail tax outweighs the shorter load chains; the D2 4-row-scale lesson (issue-point moves are free) does NOT transplant to window scale under a 64-reg budget |
| D3-4 findings | pre-existing behaviors calibrated this session: (a) at long prompts (≥2.8K tokens) `MINFER_GRAPH_DUMP` PREFILL-phase files (all `kv*_prefill`, `logits_prefill`, prefill nodes) are non-deterministic pre-vs-pre (wholesale, garbage-magnitude — aliased dump reads); decode-phase dumps stay deterministic; (b) CLI prompts longer than the n_ctx default 4096 leave zero generation headroom (`position N exceeds n_ctx N` panic, graph.rs:367); bench unaffected | measurement-only | — | — | — | RECORDED | dump gates at long prompts must anchor pre-vs-pre at the EXACT shape and gate only decode-phase files; long-prompt CLI greedy needs `prompt + n ≤ 4096` until n_ctx sizing is fixed |
| D3-5 1a | fused-producer decode A-quantize: `rms_norm_quant_pad40` (rms body + the standalone per-block quantize body, both verbatim) and `swiglu_quant_pad40` write the pad40 q8 plane beside their f32 output; decode matmuls consult `decode_quantize_native` (MmqCache, native form) and skip the standalone quantize launch on a hit; FusedFFN joins the cache-clear preserve set; `MINFER_NO_DECODE_A_FUSE=1` opt-out | `3230b2b` | nsys 14B @3254 (NO_CUDA_GRAPH): standalone `quantize_q8_0_pad40` 4448 → **964** launches (−78%; the ~50/step remainder = the attn_o class), total kernels 29402 → 25918, sub-2µs 15171 → 11723; wall 14B tg128 23.05 → **23.35** (+1.30% SEP), @3254 21.36 → **21.68** (+1.50% SEP), 7B tg128 49.86 → **50.69** (+1.66% SEP), @1641 48.47 → **49.16** (+1.43% SEP) | **+1.30%** (14B tg128) | 0.961× / 0.891× (14B tg128/@3254 vs llama 24.31/24.32); 7B **1.026×** / 0.995× | **LANDED** | q8 bytes bit-identical by construction (max is exact for any association; rintf/clamp elementwise — the epilogue IS the standalone body) and probe-verified bitwise (rms+swiglu q8 buffers, f32 producer outputs, MMVQ outputs through the cache-hit path); suite 170/0/3; 14B −n 4 dump gate: logits both phases + all KV byte-identical, the 3 node-dump diffs are same-binary pool-slot aliasing reproduced pre-vs-pre AND post-vs-post; greedy −n 256 token streams byte-identical both models; fused epilogues add ~0.2–0.25 µs/launch (swiglu 2.06 → 2.31 µs), priced into the wall |
| D3-5 1b | output-head od-split / 512-thread re-map (lm_head q6_K od 152064, id 5120, npair 160) | not implemented | — | — | — | ANALYSIS-NEGATIVE | all three candidate mechanisms are measured or computed dead at this shape: (a) idle-thread removal (96 of 256 idle at npair 160) = D3b-1c, measured neutral; (b) rows-in-flight: 288 resident rows either way (6×256-thread blocks/SM vs 3×512 dual-row), and D3b-1c's 9×160 = 432-row form was ALSO neutral — occupancy is not the limiter; (c) block-scheduling rate: the head sustains 47.6 blocks/µs while ffn_gu demonstrates 76/µs — not the limiter. A dual-row 512-thread form is bitwise-capable (per-row half-block reduce with the same 8-warp tree) but carries no mechanism → not built per the "measured mechanism, don't guess" rule |
| D3-5 1c | ffn_down-q6K 512-thread single-unit variant (id 13824 → npair 432) | not implemented | — | — | — | ANALYSIS-NEGATIVE | NOT bitwise vs the landed v2_pf: in the 256-thread form thread t accumulates fma(u_t) then += fma(u_{t+256}) into ONE float acc before the block reduce; at 512 threads those units live in different threads and their sum happens in the reduce tree (16-warp cross-warp serial order) — a different float sum. A bitwise emulation (smem pair-exchange so thread t still sums u_t+u_{t+256} first) adds a barrier for zero resident-parallelism gain (5120 rows = 18 waves either way), and the exposure mechanism 1c targets was already fixed by v2_pf's up-front load issue (D3b-1b) |
| D3-6 2a | GQA q-head batching in the decode split attention (grid (ATTN_SPLITS, n_kv_heads), 32\*gqa threads, warp w = q head hk\*gqa+w, full split stripe per warp, shared `h4w_warp_windows` window pass, per-warp 8/16-butterfly epilogue; same rpw ≥ 16 dispatch slot, static grids → replay-safe) | reverted (patch `/tmp/d3/patch_2a.py`) | nsys 14B @3254: h4w 63.76 → batched 64.79 µs (+1.6%); ncu `lts__t_sectors`: 2,121,671 (4.63× analytic 1×) → 472,583 (**1.03×**) — traffic ÷4.5 with time flat | −1.6% kernel | — | **REVERTED** | mechanism-nailed: the 5× L2 re-read is fully HIDDEN under the latency roofline in the live regime (D1's latency-bound attribution stands; D3-4's L2-composition re-attribution revised) — ncu's −28% appears only serialized/cold; ALL gates were green first: parity ≤1e-4 at gqa 5/7 incl. exact nkv 2808 + outlier calibration (shapes kept as permanent h4w coverage), dump argmax HARD gate (margins 2.187/0.557), greedy byte-identical with repeat-penalty 1.0 both models, default-penalty flips = 1/256 sampler knife-edges (14B step-63 raw top-2 probgap 0.0167; 7B step-8 penalized rank-6 winner), temp-0.8 controls identical, suite 170/0/3; new gate rule: attribute greedy flips to sampler vs kernel via the penalty-free stream + per-step logits_top trace |
| D3-7 2b | attn_v-q6K decode MMVQ routing: Q6_K decode dispatch gate lowered `od*id >= 24M` → `>= 4M` (the only affected shape in the supported set is the 14B attn_v, od 1024 × id 5120 = 5.24M, 11 layers; GGUF census; 7B attn_v od 512 × id 3584 = 1.8M stays padded-f32) | this commit | nsys 14B @3254: attn_v kernel 33.16 → 24.32 µs (−26.7%, ~177 GB/s — short of the 220 class, as the 8e small-shape crossover data warned for od≈1024 but still −26.7%); ×11 layers ≈ 97 µs/step; quantize launch count UNCHANGED (attn_v joins attn_o's MmqCache hit — same src buffer + id) | **+0.42%** (14B @3254, 8-pair median) | see D3-7 | **LANDED** | Tolerance-gated per the D3a package: logits_decode max\|Δ\| 0.254 (calibrated 0.39-class), logits_prefill byte-identical, argmax HARD gate green (margin 1.915), kv1+ decode-side f16-boundary drift (kv0 bitwise — earlier onset than D3-6's sub-ULP class, expected for input-quantization noise); penalty-free (rp=1.0) greedy streams byte-identical both models (the D3-6 clean kernel gate); default-penalty flips = 1 knife-edge event/256 steps (5/5 seeds, coherent text, no degeneracy); temp-0.8 controls: 7B identical, 14B reorders (sampled reordering expected at 0.22-logit drift); wall: @3254 21.605 → 21.72 (+0.42%, 7/7 clean pairs positive, sign-test p≈0.008; strict SEP missed by 0.05% — min-new-excl-outlier 21.66 vs max-base 21.67 — medians carried per the D3-5 outlier precedent); tg128 clean-window +0.26% (sub-bar; the extension window was co-tenant-contaminated post-side) |
| D3-7 2c | rms/elementwise-launch consolidation, two bitwise sub-levers: (i) `rms_norm_quant_pad40` wide-block geometry (launch 32 → 128 threads; the reduction keeps lanes 0..31 exactly — same element→lane map, serial per-lane chains, `warp_reduce_sum` tree; scale broadcasts via smem; write/quantize loops are element/per-32-block independent so their wider mapping cannot move a bit; reduce loop `#pragma unroll 8` deepens load pipelining), (ii) `positions_i32` one-execution-window memo (every Rope/KvcacheStore/Attn node re-converted the same positions buffer: 240 launches/step at 14B; key (buf id, pool_gen), cleared in `synchronize` next to the MmqCache clear; capture-safe: only the first consumer's conversion is recorded and replay re-executes it) | this commit | nsys 14B @3254 decode census: rms_norm_quant_pad40 9.43 → **5.66 µs** (−40%), 94.6–96/step; f32_bits_to_i32 239.6 → **1.2** launches/step; wall-effective ≈ −0.62 ms/step (rms −0.348 + bits −0.275) | **+1.76%** (14B @3254 cumulative with 2b, SEP) | see D3-7 | **LANDED** | Bitwise end-to-end: dump gate (both sides under `MINFER_NO_KQ_MMVQ=1`) 109/114 files byte-identical, the 5 diffs = the documented slot-aliasing class; logits both phases + all KV byte-identical; 7B greedy streams byte-identical 5/5 seeds + temp-0.8 control; suite green. GATE GOTCHA recorded: `MINFER_NO_KQ_MMVQ=1` also reverts the Q5_K decode arm (pre-existing), so a 2b-off control must set it on BOTH sides — a one-sided control shows a fake 0.22-logit drift from the Q5_K f32-activation fallback |
| D3-8 | FusedQKV decode fusion ported to CUDA (Stage-3 Tier A; the G4 Metal fusion): (1) `attn_bias_rope_store_f32` kernel + `launch_attn_bias_rope_store` — one launch replaces the per-layer add_bias×3 + rope×2 + store_kv×2 chain, pointer-form (serves concat sections q=base/k=base+nqt/v=base+2nkt AND three separate buffers), positions read device-side (`positions[0]`) so the launch is capture-safe, math verbatim `add_bias_f32`+`rope_f32`+`store_kv_f32/f16`; (2) class 1 (wq\|wk\|wv same quant type): `Op::FusedQKV` — one concat matmul (`blk.{i}.attn_qkv` loader-registered wq\|wk\|wv rows) + the fused epilogue (MMVQ is per-row, dispatch on (ttype,id,nt) only → concat bitwise-equal to 3 separate matmuls, probe-proven); (3) class 2 (mixed quant, e.g. Q6_K attn_v among Q4_K q/k — 24/48 layers at 14B, 14/28 at 7B): new `Op::QkvBiasRopeStore` — the three SEPARATE matmuls (bias-free) + one epilogue launch (CUDA-only; Metal keeps the unfused chain for these layers), builder wires attention to the epilogue node so q's matmul buffer has exactly one consumer and the §5 in-place alias applies; gated by `nt==1 && gpu && fuse_qkv` (part of the reuse identity — `MINFER_NO_FUSE_QKV=1` reverts both classes for A/B) | this commit | nsys 14B @3254: total launches −4968/trace (−22.5%); per decode step **−310** (add_bias −144, rope −96, store_kv −96, fused +48, mmvq −48 (24 concat layers 3→1), quantize +24) ≈ the D3-7 §2-listed 0.45 ms/step qkv-chain item | **+1.63%** (14B @3254; tg128 **+3.11%**; 7B +1.23%/+1.05%; isolation A/B post-vs-post NO_FUSE_QKV: +2.28%/+3.15%/+1.00%/+1.03% — all 3/3 pairs clean-separated) | see D3-8 | **LANDED** | Bitwise: probe tests (epilogue vs the 7-launch chain bitwise on q/k/v sections + KV rows, f32+f16 KV, both pointer forms, 14B+7B shapes; concat matmul vs 3 separate bitwise) + dump gate logits_prefill/decode + ALL kv\*.f32 byte-identical both models (98/98 + 58/58; the informational node\* dumps are a documented instrument limitation — recycled pool slots, binary-layout-dependent) + greedy −n 256 **byte-identical 5/5 seeds × both models** + rp=1.0 + `MINFER_NO_FUSE_QKV=1` control + temp-0.8 controls (the only diffs are the perf-banner tok/s numbers); prefill DOT graph byte-identical (prefill untouched); suite 172/0/3 (D3-7's 170 + 2 probes); prefill graph topology unchanged (nt>1 gate) so prefill perf untouched (pp3254 1833 t/s pre-vs-post) |
| D4-2 | Decode-GEMM Tier B session (design `/tmp/d4/D4_DESIGN.md`): (B0) **correctness** — v2_pf dispatch bounded to npair ≤ 512 (see the D4-2 chapter; D3b-1b's 7B gains were mostly dropped units); (A) llama L2-prefetch port closed PRE-BUILD: prefetch distance 2·bpi requires bpr > 32 blocks (QI4_K=16/VDR=2, QI6_K=8/VDR=1 → bpi = 4·nwarps) — at 14B only ffn_down (bpr 54) qualifies, and our kernels map one 64-elem unit per thread over 256 threads → exactly ONE K-loop iteration at every decode shape (npair 80/216/160; v2_pf's 432 are unrolled u0/u1): there is no "2 iterations ahead" to prefetch, and where llama's prefetch does fire its ffn_down-q4K runs 224.1 GB/s vs our 228.6; (B1) `__launch_bounds__(256,6)` on v2_pf: 48→40 regs + 40 B stack spill, probe tg128 +0.25% / @3254 **−0.75%** → killed; (B1c) v2-loop at npair 432 via `MINFER_Q6K_PF=0`: v2_pf wins/ties (the 5-block × 2-unit-MLP form beats 6-block × 1-unit) → default kept, env kept as opt-out; (B2) 160-thread v2 right-size (= D3b-1c repeat, re-measured with per-kernel isolation): bitwise 98/98 but nsys lm_head **+1.51%**, attn_v **+3.04%** → killed | `b31084c` (B0) + docs commit | per-kernel nsys deltas above; walls ≈ 0 as expected for a fix-only tree | correctness fix; perf-neutral | — | see D4-2 | Dump gates 107/107 (14B pre-vs-post) + 98/98 (B2 bitwise check); greedy byte-identical 14B pre-vs-post; 7B fixed-vs-v1 first-step logits at the v1-vs-v2 rounding class (max\|Δ\| 0.254, argmax same) vs 4.72-4.79 pre-bug; suite 173/0/3 |
| D4-3 | Attention-structure attempt 2 (probe `/tmp/d4/probe_attn2.cu`, 690-row sweep, 0 skips): llama-fattn-geometry split-attention kernel `vec_attn` over pb/T/R/minb/STG (load-scheduling axis); **NO-GO per the pre-registered bar** — best 41.07 µs kernel-total @14B (bar ≤~32; 1.64× vs current 67.4) and 16.22 @7B@1641 (bar ≤~10.35; 1.53×) → projected wall +1.6–1.8% < the +2% bar → no integration. **Headline: the D4-1 llama attention target (14.02 µs/layer @14B/@3254) is a llama-bench artifact** — ncu: the bench decode fattn-vec (grid (1,2,40)) loads a constant 5,427,200 B ≈ one 128-row KV iteration per block = 256 of 3255 rows covered, byte-identical at KV 1024/2474/3255, while llama-cli's decode (grid (1,7,40)) loads 53.2/142.7 MB scaling with context (mid-prompt recall A/B confirms). Honest llama full-context decode attention ≈ 2.0–2.1 TB/s ≈ 1.7 ms/step @14B — minfer's 3.32 ms is ~1.9× off, not 4.9×; the honest @3254 gap is ~10% wall (~3.5% attention) | docs commit | sweep table in the D4-3 chapter | line closed (measurement-corrected) | — | see D4-3 | probe gate 0.05 abs w/ adversarial outliers, CPU ref in double; SASS-level LDG counts + recall A/B + reductio (9.6 TB/s impossible) all consistent |
| D4-4 | Final decode-kernel session (three levers, `/tmp/d4/probe_l1_dpl.cu` + `/tmp/d4/probe_l3_fuse.cu`): **(L1) dense split-plane (dpl) q6_K decode MMVQ LANDED** — the padded 256-elem/224B row layout streams 14 dead bytes per super-block (215/256 useful = 84%); dpl repacks to `[ql: nbe×128][qh: nbe×64][sc: nbe×16][d: nbe×2]` = 210B content/row at row stride `(nbe·210+15)&~15` (16B-aligned uint4, zero pad sectors), same per-unit values + accumulation order → bitwise; sibling plane (+2.0 GB 14B, +0.9 GB 7B) under `MINFER_Q6K_DPL` ("0" opt-out), only `id % 256 == 0` shapes, padded plane retained (prefill MMQ block_stride 224 + W_exp/W_dsc derivation + dequant/embed fallback). Probe: ffn_down 176.5→212.9 GB/s content (−17.1%), lm_head 208.3→250.0 (−16.7%); the group-split (gs) probe variant (−7.9/−10.7%, tolerance) dominated → dropped. **(L2) PDL on the decode chain NO-GO in-situ**: standalone probe green (graph capture+instantiate with `cudaLaunchAttributeProgrammaticStreamSerialization` OK on driver 580.173.02, 200 replays stable, PDL-graph vs plain-eager bitwise; but a compute-bound chain probe ran +2.8% slower — co-residency tax warning), full integration (PSS launch attribute + `cudaGridDependencySynchronize()` on 13 decode-chain kernels, `MINFER_PDL` gate) passed all bitwise gates, then the same-binary env-flip isolation A/B read 14B tg128 **−2.6%/−1.8%**, 7B ≈ 0%, @3254 within noise → below the +0.3% bar → reverted; mechanism: PSS early-launch co-residency taxes the compute-tail kernels (attention h4w, lm_head) more than the ~2 µs/launch graph-gap pool it recovers. **(L3) fused gate+up+SwiGLU+q8 (Form B) NO-GO**: 32-row-block fused q4_K kernel (grid nf/32 = 432 blocks, 64 serial per-row dots each, in-kernel silu + `quantize_pad40_block`) measured **+28.2%** vs the incumbent gu-matmul + swiglu_quant pair (412.1 vs 321.4 µs at 27648×5120; 193.2 vs 247.8 GB/s content) — grid = 1.5 waves at 6 blocks/SM (wave quantization) + exposed per-row latency; Form A (fused gu+swiglu f32, separate quantize) saves ~2–5 µs/step by arithmetic = sub-bar, not probed | code commit + docs commit | wall deltas this row (3× interleaved same-window A/B, medians of 3) | 14B tg128 23.28→**24.57** (+5.53%) / @3254 22.00→**22.94** (+4.27%); 7B tg128 47.55→**51.20** (+7.68%) / @1641 46.44→**50.18** (+8.05%) — **+5.53%/+4.27%** (14B) | vs-llama: 14B tg128 **1.018×**, @3254 **0.950×**; 7B tg128 **1.074×**, @1641 **1.052×** (llama 24.14/24.14/47.65/47.69) | **LANDED (L1)**; L2/L3 closed with mechanism | L1: dpl kernels bitwise vs padded (probe + unit test both forms: od 512/id 8960 pf-form, od 4096/id 1024 loop-form); 14B −n 1 dumps 107 identical + 7 node{N} diffs (= the documented D4-2 pool-slot aliasing class), 7B 72 + 2; greedy rp=1.0 byte-identical both models; suite 174/0/3 (incl. the new dpl bitwise test; one earlier full-suite run flaked 2 pool/parity tests under the sglang co-tenant window — both pass in isolation and the rerun is green) |
| D5-0 | Speculative-decoding cost model (gate, no engine change): measured per-token costs (7B q4_k_m CUDA **54.3** tok/s; 0.5B q4_0 CUDA **342.2** / CPU **73.3**; q4_k_m 365.0; q5_k_m 321.8) + real greedy acceptance via llama.cpp `speculative-simple` (0.5B-on-7B: aggregate 58.8/42.6/25.0% at d=2/4/8 → conditional **p ≈ 0.68–0.70**, stable across prose/code and draft quant) → break-even requires p\* = 0.73/0.81/0.90 at d=2/4/8 → **conditional go at d=2 only**, gate = minfer measured nt=3 verify-batch amortization **≥ 2.5×** (D4-1 anchor 2.7× at nt=4; interpolation 2.28× vs tile-step 2.7× disagree — D5-1 re-ordered primitive-first to measure); projected 1.04–1.05× at the anchor, ceiling ~1.2×; CPU-draft cross-device dead (1.35×) | docs commit | battery: 3× interleaved `-p 0 -n 128` medians; acceptance n=256 greedy, prose+code | — | — | MEAS-ONLY (gate open) | measure the gate variable with someone else's binary; the cross-device fallback died by measurement not argument; interpolation is not measurement — nt=3 lands either side of the 2.5× line and only `C_T(3)` arbitrates |
| D5-1a | The gate measured end-to-end — **FAILED, D5 closed**: new `minfer specverify` instrument (`src/spec_verify.rs`; fixed-depth protocol, warmups absorb the graph rebuild, two-pass drift check) drives the generic `forward_graph_cached` at nt>1/n_out=nt; 7B q4_k_m CUDA @KV512: C_T(1)=18.33 ms, **C_T(3)=106.0 ms → per-token amortization 0.52×** vs ≥2.5× (needed C_T(3) ≤ 22.1 ms). Full curve: nt=2–8 costs a flat **~35 ms/token** (batched path re-streams weights per row — 1.9× worse per token than the nt=1 MMVQ path; zero amortization anywhere), the real tile-regime step sits at **M≥16** (nt=16 = 56.7 ms total, 3.1× the weight floor; nt=64 = 64.9 ms) — unreachable for verify (nt=d+1 ≤ 9), and even padding nt=3→16 caps at 0.97×. Five probes eliminated alternatives (graph-launch asymmetry, KV depth, FA kernel, n_out path, drift). The D4-1 2.7×@nt=4 anchor was a kernel micro-bench that never existed at graph level | this commit + docs addendum | medians of 15–40 reps × 2 passes, ±2% spread; probes + llama-cli battery in doc 81 | gate FAIL 4.8× (0.52× vs 2.5×); external check: llama-cli `-md` same pair lands 0.99–1.00× (doc 81 §4.1) — **VOID, see doc 81 §4.3 errata: the battery never engaged the draft (missing `--spec-type`); corrected = 1.53–2.43×** | — | see D5-1a | **LANDED** (instrument) · **D5 CLOSED** per the pre-registered stop rule | kernel micro-benchmarks are not engine facts — anchors must be measured at the level they are consumed; a batched path that re-streams weights per row is worse than no batching; the strategy itself has no headroom on GB10 (llama.cpp also lands at 1.00×) — the dispatch fix matters only for future multi-token features |
| 82 | small-M dispatch fix: multi-token MMVQ (K-quants, nt 2–8, in-block token loop) + token-looped legacy kernels (grid.y=nt removed) + GEMM gate 16→9 | this commit | 7B batched decode nt=3 105.9 → 29.4 ms (**3.60×**), nt=8 279.3 → 48.6 ms (**5.75×**); marginal 34.4 → 4.3 ms/token; nt=1 paths bitwise-unchanged (tg128/pp512 clean) | — | — | LANDED | the pre-registered 2.5× amortization bar was mis-derived (marginal ≈ nt×(attention+compute), not ε) — measured 1.87×; weight traffic is nt-independent now, D5 stays closed (C_T(3)=29.4 > 22.1); small models gain ~1.0× only (per-layer weights already L2-buffered) |
| D5-R ① | speculative loop LANDED (reopen per doc 81 §4.3 + doc 82): `src/spec.rs` SpecEngine — d×nt==1 draft forwards + one nt=d+1 verify through `forward_graph_cached`, lazy accept loop (unit-tested), full-accept draft-KV repair, CLI `--spec-draft/--spec-draft-n`; two-model process fixes: namespaced GPU weight registries (`load_model_ns`), `nb_bt_only` global-mix semantics (q4_0 draft degrades mode-2→mode-1) | this commit + doc 83 | 14B+0.5B q4_0 d=2 greedy n=128: **34.0/40.6 tok/s = 1.34×/1.58×** vs serial 25.5/25.5; 7B 1.18×; suite 179 green | G1 re-scoped by measurement: batched-verify vs decode kernels differ ~0.01–0.05 logits → near-tie flaps only (first-flap margin 0.043); d=0 fallback == non-spec to one exact-tie flap | — | LANDED | "single-model-per-process" was load-bearing in three places (registry, dispatch flags, prewarm); all-or-nothing CUDA failure is silent CPU — watch tok/s, not errors; greedy equivalence across kernel paths is a numerics statement, not a logic statement |
| D5-R ② | same-window dual-engine battery: 3 interleaved reps × prose/code × 4 cells (minfer off/d2 × llama base/d2), doc 81 §4.3 protocol | this commit + doc 84 | minfer **1.33×/1.59×** (34.0/40.6) vs llama **1.64×/2.08×** (38.8/49.0) in one window; minfer = 88%/83% of llama's absolute spec speed; gate ≥1.2× PASS | — | — | LANDED | same-window interleaving beats rep count (llama's spec cell moved 13% between windows, base <1%); the whole gap to llama is the verify row marginal (minfer 8.8 vs llama 2.5 ms/row) — closing it prices at 1.62×; acceptance prose 51.6% (near-tie dilution) / code 73.1% |

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

## §1 Current state (post-D4-4, 2026-09-09)

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
`MINFER_MMQ*` read on the nt==1 path; decode `-n 16 --greedy` byte-identical).
Final decode-campaign state (D1→D4-4, 2026-09, all measured on the B0-fixed
engine — the D4-2 correction found the v2_pf dispatch dropping units on
npair-592 rows and re-anchored every 7B claim):

| model | tg128 | long-ctx | vs-llama (same-window) |
|---|---|---|---|
| 7B q4_k_m | **51.20** | @1641 **50.18** | **1.074× / 1.052×** (ahead) |
| 14B (48L) | **24.57** | @3254 **22.94** | **1.018×** tg128 / **0.950×** @3254 |

Small models (pre-MMQ-campaign numbers, Part-I record): 0.6B q8_0 prefill @2K
4792 (llama 23909), decode tg128 ~195 (290); 0.5B q4_0 prefill ~3020 (30550),
decode ~257 (453).

The per-session narratives (D2/D3-4/D3-5/D3-7/D3-8 updates), the D4-2/D4-3
correction chain, and the standing measurement rules live in the §0 rows
D1–D4-4 and step docs 65–76. Two rules worth surfacing: never quote
llama-bench long-ctx tg rates as attention targets without an ncu byte-count
or llama-cli recall cross-check (D4-3: the bench decode fattn-vec covers only
~8% of KV rows; honest llama decode attention ≈ 1.7 ms/step, minfer ~1.9×
off), and an end-to-end max\|Δlogits\| ≈ 0.38/0.39 is the inherent class of
ANY accumulation-order change — argmax + greedy-divergence + A/B are the
operative gates (D3a calibration).

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

- **D5 speculative decoding — CLOSED 2026-09-10, REOPENED as D5-R 2026-09-12
  (docs 81 §4.3, 82, 83, 84)**: the original closure's bar was mis-derived
  and its llama-cli anchor was a measurement artifact (`--spec-type`
  silently defaults to none); doc 82's small-M dispatch fix made the verify
  amortization 2.14×. D5-R stage ① landed the loop (1.34×/1.58× at 14B d=2),
  stage ② priced the gap to llama (verify row marginal 8.8 vs 2.5 ms/row →
  1.62× recoverable). Current plan:
  [`SPECULATIVE-DECODING-PLAN.md`](./SPECULATIVE-DECODING-PLAN.md) (the old
  plan is an appendix there).
- **Not planned** (revisit with a concrete need): cuBLAS/cublasLt (closed as
  8k — 8m's wmma GEMM covered the f16 path), VMM pool, multi-GPU, node
  reordering, Windows, IQ/Q2/Q3 quants.
- Open leads, all sub-bar or step-function: q8_1 GEMM-prologue fusion (the
  step change); rms_nw roofline (+0.5–1%); wave re-tile for small-od classes
  (+0.3–0.8%, needs ≤85 regs); fused ffn_gu concat (needs the G5 nf≤16384
  gate re-measured); FA deep-opt only with numerics-order-preserving
  structure (r50/r57 caveat).
- Open Phase-8 ledger items (inherited from the retired `CUDA-FOLLOWUP-PLAN.md`;
  records in step docs 78/79): **8e② follow-up** — llama.cpp's
  shape-dependent `halve_iters` idle-tail rule, not started; **8h②** —
  self-hosted CUDA CI runner, DEFERRED (needs standing runner infrastructure);
  **8h③** — the Phase-7 `/tmp/minfer_phase7/` ledger cleanup, awaiting user
  decision.
- Closed Phase-8 ledger item: **8a①** macOS Metal regression run (fuse_ffn
  decoupling + the `MINFER_NO_FUSE_FFN` A/B on 0.5B + 7B) — was BLOCKED on
  hardware; **DONE 2026-09-10** on an Apple M4 Pro, all three checks green
  (pre/post greedy byte-identity, fused-vs-unfused byte-identity, 0.5B decode
  graph still emitting 24 × `fused_ffn` on Metal). Record: doc 78 §3.2.

## §2 Step documents — one doc per history row

The full per-step chapters (process narrative, principle explanations, real code
excerpts, verification gates, lessons — previously inlined here) now live in
**[`docs/cuda_optimization_steps/`](./cuda_optimization_steps/README.md)** as 79
standalone documents (01–76, the Phase-8 supplementary records 78–79, and the
verification-methodology capstone 77). The §0 master table above remains the
one-row-per-step index; the tables below link each row to its step document.
Appendix B points at the cross-cutting methodology.

### Part I · Era A — Phase 7/8 foundations (rows 1–6 + 78–79)

| # | doc |
|---|---|
| 01 | [Phase 7 — CUDA backend: raw FFI device layer + graph backend 7a–7e (LANDED)](./cuda_optimization_steps/01-phase7-cuda-backend.md) |
| 02 | [8m/8m② — Tiled wmma f16 prefill GEMM (LANDED)](./cuda_optimization_steps/02-wmma-f16-prefill-gemm-8m.md) |
| 03 | [8n — FA-style tiled prefill attention (LANDED)](./cuda_optimization_steps/03-fa-tiled-prefill-attention-8n.md) |
| 04 | [8o — Killing the CPU stall at decode start (LANDED)](./cuda_optimization_steps/04-decode-start-stall-8o.md) |
| 05 | [8p — Persistent f16 weight cache + fused dequant-in-GEMM (LANDED)](./cuda_optimization_steps/05-persistent-f16-cache-8p.md) |
| 06 | [8e/8e② — decode MMVQ: dp4a integer dot products + the llama.cpp launch table (LANDED)](./cuda_optimization_steps/06-decode-mmvq-8e.md) |
| 78 | [Phase-8 correctness & engineering-debt batch — 8a/8h①/8i (LANDED; 8a① closed 2026-09-10)](./cuda_optimization_steps/78-phase8-correctness-batch.md) |
| 79 | [Phase-8 coverage & first-measurement batch — 8b/8c/8d/8f/8l/8q (LANDED)](./cuda_optimization_steps/79-phase8-coverage-batch.md) |

### Part II · Era B — R and P5 sessions (rows 7–13)

| # | doc |
|---|---|
| 07 | [R3 — Small-model per-token overhead: single-split prefill (LANDED)](./cuda_optimization_steps/07-r3-small-model-overhead.md) |
| 08 | [R1 — int8 MMQ prefill GEMM (opt-in): the parity-first strategy (LANDED)](./cuda_optimization_steps/08-r1-int8-mmq-prefill-gemm.md) |
| 09 | [R2 — MMVQ weight-streaming rework (LANDED)](./cuda_optimization_steps/09-r2-mmvq-weight-streaming.md) |
| 10 | [R4 — decode split-attention dim-parallel rewrite (LANDED)](./cuda_optimization_steps/10-r4-split-attention-dim-parallel.md) |
| 11 | [P5 — prefill gap session: TM=128 big tiles + FA rewrite (LANDED, with three REVERTED probes)](./cuda_optimization_steps/11-p5-gemm-tiles-fa-rewrite.md) |

### Part III · Era C — the P6 q4_K MMQ line, r5–r37 (rows 14–51)

| # | doc |
|---|---|
| 12 | [r5–r6 re-ranking + structural rewrite spec (MEAS-ONLY + REVERTED)](./cuda_optimization_steps/12-r5-r6-rerank-rewrite-spec.md) |
| 13 | [r7–r8 raw-byte MMQ kernel + wide tile + FA probe (LANDED (raw) + REVERTED (probe))](./cuda_optimization_steps/13-r7-r8-raw-byte-kernel.md) |
| 14 | [r9 llama.cpp MMQ reference decode; shape axis closed (MEAS-ONLY)](./cuda_optimization_steps/14-r9-llama-mmq-reference.md) |
| 15 | [r10–r11 — reference inner-loop decomposition port; ILP reading verification (REVERTED / MEAS-ONLY)](./cuda_optimization_steps/15-r10-r11-inner-loop-port.md) |
| 16 | [r12 — 16-chain warp tile + ldmatrix (LANDED)](./cuda_optimization_steps/16-r12-warp-tile-ldmatrix.md) |
| 17 | [x-tile / j-tile / cp.async-db — the staging-shape family, closed (REVERTED / closed)](./cuda_optimization_steps/17-staging-shape-family.md) |
| 18 | [r13 — counter forensics against llama.cpp (MEAS-ONLY, closed)](./cuda_optimization_steps/18-r13-counter-forensics.md) |
| 19 | [r14 — B fragments via ldmatrix + widened scale reads (LANDED)](./cuda_optimization_steps/19-r14-b-fragments-ldmatrix.md) |
| 20 | [r15 — f32-accumulate mma probe (dead end) + rank-1 term2 rescale (LANDED)](./cuda_optimization_steps/20-r15-f32-acc-mma-rank1.md) |
| 21 | [r16 — Narrow kernel gets the rank-1 fold (LANDED)](./cuda_optimization_steps/21-r16-narrow-kernel-rank1-fold.md) |
| 22 | [r17 — Wide warp remap 32od × 64tok (REVERTED)](./cuda_optimization_steps/22-r17-wide-warp-remap.md) |
| 23 | [r18: Load-time B pre-expansion — staging becomes a bulk copy (W_exp's debut, REVERTED)](./cuda_optimization_steps/23-r18-load-time-b-preexpansion.md) |
| 24 | [r19: Weight L2 residency — `__ldg` imperceptible, persisting window catastrophic (REVERTED)](./cuda_optimization_steps/24-r19-weight-l2-residency.md) |
| 25 | [r20: Split-phase A staging — attribute first, shoot second: the first hit (LANDED)](./cuda_optimization_steps/25-r20-split-phase-a-staging.md) |
| 26 | [r21 — Coalesced block-linear A staging: stall-mass conservation (REVERTED)](./cuda_optimization_steps/26-r21-coalesced-block-linear-a.md) |
| 27 | [r22 — qa8 XOR swizzle (LANDED); d/ssum fold reverted separately](./cuda_optimization_steps/27-r22-qa8-xor-swizzle.md) |
| 28 | [r23 — f16-path whole-graph wall decomposition + FA_TKV occupancy raise (MEAS-ONLY + REVERTED)](./cuda_optimization_steps/28-r23-f16-wall-decomposition.md) |
| 29 | [r24 — The scheduling-structure ladder (REVERTED; the +1.5% whole-prefill landing bar calibrated here)](./cuda_optimization_steps/29-r24-scheduling-ladder.md) |
| 30 | [r25 — SASS opcode census; the unroll is wall-inert (MEAS-ONLY + REVERTED)](./cuda_optimization_steps/30-r25-sass-opcode-census.md) |
| 31 | [r28 — Direction-A raw-nibble NB kernel, 2 blocks/SM (LANDED)](./cuda_optimization_steps/31-r28-nb-kernel-2blocks.md) |
| 32 | [r29 — NB kd-loop unroll: 2 blocks/SM lets integer-ALU pruning move the wall clock for the first time (LANDED)](./cuda_optimization_steps/32-r29-nb-kd-loop-unroll.md) |
| 33 | [r30 — SWAR unpack: the compiler already did it (REVERTED)](./cuda_optimization_steps/33-r30-swar-unpack.md) |
| 34 | [r31 — q-major sda scale-read repack: a sub-bar positive gain caught by conflict analysis (LANDED)](./cuda_optimization_steps/34-r31-qmajor-sda-repack.md) |
| 35 | [r32 — The finite lever sweep: two regions fenced off (REVERTED)](./cuda_optimization_steps/35-r32-finite-lever-sweep.md) |
| 36 | [r33 — Hybrid inner-loop port: SASS fully identical, hypothesis falsified (REVERTED)](./cuda_optimization_steps/36-r33-hybrid-inner-loop.md) |
| 37 | [r34 — The quantize-transpose prepass: layout-transform locality (LANDED, +9.72%)](./cuda_optimization_steps/37-r34-quantize-transpose-prepass.md) |
| 38 | [r35 — sda scale predecode: a total SASS win, a wall-clock tie (REVERTED)](./cuda_optimization_steps/38-r35-scale-predecode.md) |
| 39 | [r36 — A-frag wavefront economics: H1 falsified (MEAS-ONLY, no code change)](./cuda_optimization_steps/39-r36-a-frag-wavefront.md) |
| 40 | [r37 — Post-parity whole-prefill attribution: the wall clock re-decomposed (MEAS-ONLY, no code change)](./cuda_optimization_steps/40-r37-post-parity-attribution.md) |

### Part IV · Era D — q6_K, FA, prepass, promotion, r38–r60 (rows 52–75)

| # | doc |
|---|---|
| 41 | [r38 — q6_K BT-style raw-byte mma kernel (LANDED)](./cuda_optimization_steps/41-r38-q6k-bt-rawbyte-mma.md) |
| 42 | [r39 — q6_K KDR=2 double-buffer (LANDED)](./cuda_optimization_steps/42-r39-q6k-kdr2-double-buffer.md) |
| 43 | [r40 — `__launch_bounds__(256,3)` third resident block (LANDED)](./cuda_optimization_steps/43-r40-third-resident-block.md) |
| 44 | [r41 — q6_K B-expand widened to uint4 group loads (LANDED)](./cuda_optimization_steps/44-r41-q6k-bexpand-uint4.md) |
| 45 | [r42 — q6_K stage-wide dsc scale reads (REVERTED)](./cuda_optimization_steps/45-r42-stage-wide-dsc-read.md) |
| 46 | [r43 — PC-sampling attribution + pre-expand-B parity FAIL (MEAS-ONLY + REVERTED)](./cuda_optimization_steps/46-r43-pc-sampling-attribution.md) |
| 47 | [r44 — W_exp stride mismatch root cause: fix goes parity all-green but wall-neutral (REVERTED)](./cuda_optimization_steps/47-r44-wexp-stride-mismatch.md) |
| 48 | [r45 — cp.async for the q6_K A-side staging: mechanism confirmed, wall-neutral (REVERTED)](./cuda_optimization_steps/48-r45-cpasync-q6k-a-staging.md) |
| 49 | [r46 (FAP1) — FA audit + occupancy/bank-conflict levers: kernel −11% but wall-neutral (REVERTED)](./cuda_optimization_steps/49-r46-fap1-fa-audit.md) |
| 50 | [r47 — converged-era whole-wall re-decomposition (MEAS-ONLY)](./cuda_optimization_steps/50-r47-converged-wall-decomposition.md) |
| 51 | [r48 (FAP2) — register-resident softmax: deleting the S/P smem round trip outright (LANDED)](./cuda_optimization_steps/51-r48-fap2-register-softmax.md) |
| 52 | [r49 — A-quantize prepass shared-A dedup: consecutive-window memoization (LANDED)](./cuda_optimization_steps/52-r49-a-quantize-shared-dedup.md) |
| 53 | [r50 — FA_TKV 32→16 occupancy experiment (REVERTED)](./cuda_optimization_steps/53-r50-fa-tkv-16.md) |
| 54 | [r51 — producer-fused A-quantize mode 1 (LANDED)](./cuda_optimization_steps/54-r51-producer-fused-a-quantize.md) |
| 55 | [r52 — skip-write mode 2: skipping the f32 intermediate write-out (LANDED)](./cuda_optimization_steps/55-r52-skip-write-mode2.md) |
| 56 | [r53 — q6_K bundle: W_exp pre-expansion plane + cp.async B staging (LANDED)](./cuda_optimization_steps/56-r53-q6k-wexp-cpasync-bundle.md) |
| 57 | [r54 — `MINFER_MMQ_Q6K_EXP`: an exit valve for the 1.52 GB W_exp plane (LANDED)](./cuda_optimization_steps/57-r54-q6k-exp-optout.md) |
| 58 | [r55 — swiglu roofline audit + one-shot prefill CUDA-Graph: both closed on the record (CLOSED)](./cuda_optimization_steps/58-r55-swiglu-roofline-prefill-graph.md) |
| 59 | [r56 — q6_K A-side bundle: A cp.async + W_dsc f32 plane (LANDED)](./cuda_optimization_steps/59-r56-q6k-a-cpasync-wdsc.md) |
| 60 | [r57 — FA KV staging double buffering (REVERTED)](./cuda_optimization_steps/60-r57-fa-kv-staging-db.md) |
| 61 | [r58 — q4_K BT spec + cp.async-db2 transplant (REVERTED)](./cuda_optimization_steps/61-r58-q4k-bt-cpasync-transplant.md) |
| 62 | [r59 — q4_K W_dsc plane + riders (LANDED, Δ corrected by r59b)](./cuda_optimization_steps/62-r59-q4k-wdsc-plane.md) |
| 63 | [r59b — clean re-measurement + baseline-contamination correction (measurement round)](./cuda_optimization_steps/63-r59b-clean-remeasure.md) |
| 64 | [r60 — the coronation: flipping the verified gate set to default-on (PROMOTION, LANDED)](./cuda_optimization_steps/64-r60-promotion-default-on.md) |

### Part V · The decode campaign, D1→D4-4 (rows 76–87)

| # | doc |
|---|---|
| 65 | [D1 decode attribution: split-attention staging depth is the only wall that grows with KV (measurement round, CLOSED)](./cuda_optimization_steps/65-d1-decode-attribution.md) |
| 66 | [D2: explicit K+V register staging (LANDED, +2.0% @1641) and the cp.async negative result](./cuda_optimization_steps/66-d2-kv-register-staging.md) |
| 67 | [D3: 14B decode attribution (D3-1) + the D3b bitwise MMVQ triple (1b LANDED; 1a/1c REVERTED)](./cuda_optimization_steps/67-d3-14b-attribution-bitwise-mmvq.md) |
| 68 | [D3a — the 4-warp fattn-vec-style split-attention rewrite (REVERTED) + tolerance-gate calibration](./cuda_optimization_steps/68-d3a-fattn-rewrite-rpw.md) |
| 69 | [D3-4 — L1 hybrid rpw dispatch (LANDED) + L2 window-prefetch pipelining (REVERTED) + long-prompt dump calibration](./cuda_optimization_steps/69-d3-4-hybrid-rpw-dispatch.md) |
| 70 | [D3-5 — decode-alignment plan Stage 1: fused-producer decode A-quantize (LANDED) + negative analysis of the output-head/ffn_down geometry levers (1b/1c)](./cuda_optimization_steps/70-d3-5-fused-producer-a-quantize.md) |
| 71 | [D3-6 — GQA q-head batched attention: all gates green, still reverted — the 5× L2 re-read was not the residual (REVERTED)](./cuda_optimization_steps/71-d3-6-gqa-batching-reverted.md) |
| 72 | [D3-7 — attn_v-q6K MMVQ routing (2b) + rms wide-block / positions memo (2c): the Stage-2 closing ledger (LANDED ×2)](./cuda_optimization_steps/72-d3-7-attnv-mmvq-rms.md) |
| 73 | [D3-8 — G4 FusedQKV ported to CUDA: both layer classes covered, 14B short-KV breaks through parity (LANDED)](./cuda_optimization_steps/73-d3-8-fusedqkv-port.md) |
| 74 | [D4-2 — B0 latent correctness fix + all bitwise occupancy/prefetch axes closed (LANDED)](./cuda_optimization_steps/74-d4-2-b0-correctness-fix.md) |
| 75 | [D4-3 — attention structure rewrite attempt 2 NO-GO + D4-1's llama target was a llama-bench artifact (CLOSED)](./cuda_optimization_steps/75-d4-3-attention-attempt2-artifact.md) |
| 76 | [D4-4 — the endgame kernel session: dpl dense split-plane q6_K decode MMVQ lands (bitwise); PDL and fused-FFN closed with mechanism (LANDED)](./cuda_optimization_steps/76-d4-4-dpl-q6k-final.md) |
| 80 | [D5-0 — speculative-decoding cost model: measured baseline, acceptance, and the d=2 gate (MEAS-ONLY)](./cuda_optimization_steps/80-d5-0-cost-model.md) |
| 81 | [D5-1a — the verify-batch gate measured end-to-end: no amortization at any nt, D5 closed (LANDED · gate FAIL)](./cuda_optimization_steps/81-d5-1a-verify-gate-measured.md) |
| 82 | [small-M dispatch fix — multi-token MMVQ + token-looped legacy kernels: the batching invariant restored, D5 verdict unchanged (LANDED)](./cuda_optimization_steps/82-small-m-multi-token-mmvq.md) |
| 83 | [D5-R stage 1 — speculative decode loop: two-model process fixes, accept-loop unit tests, greedy-identity investigation (LANDED)](./cuda_optimization_steps/83-d5-r-stage1-spec-loop.md) |
| 84 | [D5-R stage 2 — same-window dual-engine battery vs llama.cpp: 1.33×/1.59× vs 1.64×/2.08×, gap = verify row marginal (LANDED)](./cuda_optimization_steps/84-d5-r-stage2-dual-engine-battery.md) |
| D5-R ③ | verify marginal priced with an nsys per-kernel ledger (specverify per-nt runs, exact forward spans) | this commit + doc 85 | nt=3 marginal 17.6 ms = attention +9.0 (nt 2–63 legacy per-(token,head) kernel; nt=1 split path does the same KV in 0.8 ms) + matmul +8.1 (multi-MMVQ row slope 4.05 ms/row) + elt +1.9 + idle +0.7; nt=9 falls off multi-MMVQ onto padded GEMM (48.4→84.9 ms matmul) | ncu counters permission-blocked (ERR_NVGPUCTRPERM); durations from nsys suffice for pricing | — | LANDED | the marginal was not where the plan looked — the batched-attention nt 2–63 hole (a shape range no caller ever exercised before spec decode) is half the prize; d=8's loss is a dispatch cliff, not a slope; priced recovery at d=2 → C_T(3) ≈ 42 ms → 1.64× ≈ llama parity |
| D5-R ④a | attention for the verify shapes: fa_prefill gate nt≥64 → nt≥2 (one line; hd==128, kill switch, rc fallback kept; nt=1 keeps split-KV) | this commit + doc 86 | C_T(3) 56.90 → **48.78**, C_T(9) 101.3 → **86.44**, C_T(1) unchanged; end-to-end 14B d=2 **1.39×/1.66×** (code ≥ llama's same-window 1.64×); suite 179 green | ledger projection validated within ~5% | — | LANDED | a never-exercised shape range hid a 12× kernel-choice mistake behind a gate written for a different caller; kill switches made the fix a one-liner |
| D5-R ④b | multi-MMVQ nt 9–16: token-groups-of-8 (parity 85.0 vs 84.9 — group re-streams weights from DRAM) and acc[16] single-pass (111 ms, register spill) both measured; doc-82 GEMM boundary at nt ≥ 9 stands, kernels keep the group structure (bitwise at nt ≤ 8) | this commit + doc 87 | C_T(9) stays 86.4; d=8 re-scoped: 0.71× prose / ~1.25× code — not viable at the current verify curve; MMQ's own 65 vs ~30 ms small-M gap filed as the open kernel question | bitwise suite caught a variable-shadowing corruption (acceptance 50.8→38.4%) before it shipped | — | CLOSED (measured negative) | extrapolating a per-row slope across a structural boundary (accumulator lifetime) is how plans go wrong; the bitwise multi-token test is the cheapest insurance in the campaign |
| D5-R ⑤ | final dual-engine battery + capture A/B: the doc-85 capture prize was already banked by R3-B prefill capture (48.08 captured vs 50.15 eager) | this commit + doc 88 | 14B d=2 **1.42×/1.68×** (35.7/42.5 tok/s) vs llama 1.65×/2.10× same-window — 95%/88% of llama's absolute spec speed; d=8 e2e 0.63× confirms the doc-87 retirement; acceptance unchanged (50.8%/73.1%) | — | — | LANDED · D5-R CLOSED | A/B the controlling flag before scheduling work — one ledger prize was banked two campaigns ago; the gap to llama is now two named bounded causes (multi-kernel row slope, near-tie acceptance dilution), both deeper-kernel work |
| D5-R ⑤+ | row-marginal localization (no ncu): cold-L2 real-kernel bench + chain nsys + ablation | this commit + doc 89 | marginal 4.5 ms/row = q4_K 2.4 + q6_K 1.0 + norm/quant 0.7 + attn 0.05; ~half the matmul term is the one-block-per-row activation re-read (L2 ~3.4 TB/s), rest = per-row ALU on 69%-idle lanes + chain overhead | bench test `cuda_row_marginal_bench` (MINFER_BENCH_ROW_MARGINAL=1), ablation reverted after DCE | — | LOCALIZED | "needs ncu" was too pessimistic; the dominant term is a design property (block-per-row × re-read × 31% lanes) — fix menu: R-rows-per-block (~1.7 ms/row), small-M mma tiles, chain hygiene |
| D5-R ⑤++ | doc-89 menu item 1 (R-rows-per-block) implemented → measured → reverted | this commit + doc 90 | C_T(3) neutral (47.5–48.1 vs 48.04; acceptance stable 51.6%/73.1%); nt≥6 flatten real (attn nt=8 127→94 µs) but no production shape uses it; q6_K R=8 scatter regression | cold-L2 bench retained (MINFER_BENCH_ROW_MARGINAL=1); kernels byte-identical to doc-88 state | — | CLOSED (measured) | isolated ablations price terms the chain gives away free — act rows are L2-hot from producers; block-parallel latency hiding dominates utilization at small nt; the nt=3 residual is per-row arithmetic → small-M mma is the only lever of size left |
| D5-R ⑤+++ | mma path audit: BT kernel already tensor-core; small-M floor = block starvation (40 blocks < SMs); conditional double-buffer shipped | doc 91 | M-flat measured (85.4@3 ≈ 86.4@9 vs multi 48.9); dbuf unconditional cost prefill −10% (1637) → shipped ntb==1-only; pp512 1977; suite 180 ✓; bitwise preserved (data-movement-only) | MINFER_SMALL_M_GEMM=1 gate (default off) routes nt 2..8 to BT | K-split (grid.z + deterministic 2-pass reduce) designed — expected C_T(9) 86→~50, revives d=8 | OPEN |
| D5-R ⑤++++ | K-split shipped for both BT kernels (grid.z + deterministic reduce), gated; flip condition C_T(9)≤55 NOT met (72.9) | doc 92 | C_T(9) 86.4→72.9, C_T(3) 85.4→70.9 (gate on); pp512 2005; suite 180 ✓; acceptance identical to baseline (50.8/73.1) | residual = per-tile staging serialization (~1.7× floor) — tile-geometry redesign, not tuning | deferred proposal: auto-ksplit default nt 9..64 (−13.5 ms at nt=9, deterministic) — user decision | CLOSED (flip = no) |
| D5-R ⑤+++++ | auto-ksplit enabled on the DEFAULT path (user decision, doc 92 §3b) | ba0b96d | default C_T(9) 86.4→73.0; C_T(1/3/5) unchanged; pp512 1999; suite 180 ✓; acceptance 51.6/73.1 | d=8 still loses at p=0.731 (needs ≈0.755 on code) | value: short-prompt prefill + acceptance option value | DONE |
| D5-R follow-up | draft-quant sweep (mixed knob: q4_k_m code +3.0%/prose −2%) + greedy identity test: spec output ≠ sequential — flips originate in batched verify attention/softmax, nt-invariance campaign proposed | doc 93 | draft standalone already ~1.34× floor (no speed headroom); identity diverges prose ~tok 8, code line 19 | payoff = exact transparency guarantee + acceptance de-jitter | NEXT CAMPAIGN CANDIDATE |
| D5-R ⑥++++++ | greedy identity ACHIEVED: spec output byte-identical to sequential (4-prompt battery) — batched verify attention (bitwise position-invariant, nkv<1921) + spec penalty window capped to repeat_last_n; cost ≈1% C_T(3); d=8 door CROSSES with q4_k_m draft (code 76.5% ≥ 0.755: 46.6 tok/s vs d=2 43.6, 96% of llama; prose stays d=2) | doc 94 | identity battery 4/4 + suite 181 green + C_T/pp512 gate passed | spec decoding now exactly transparent to sequential greedy | LANDED |
| D5-R ⑦ | adaptive draft depth (`--spec-draft-adaptive`): per-round d from beta-smoothed per-depth acceptance + min-of-4 verify/draft costs, 10% switch hysteresis, unobserved depths inherit the depth-1 rate (emergent exploration); **identity boundary discovered and pinned — verify nt ≤ 8 (single+multi MMVQ) bitwise vs decode, nt=9 (BT-MMQ) lm_head is tolerance-class while its KV stays bitwise (48-layer dump proof)** → adaptive cap d=7 | doc 95 | all four throughput gates pass (prose q4_0 36.0 ≥ 34.1, code q4_0 46.5 ≥ 41.7, prose q4km 35.6 ≥ 33.8, code q4km 48.4 ≥ 44.3); identity battery 4/4 byte-identical at −n 200; suite 185/0 | adaptive ≥ static everywhere, +3.9..5.9% over the best static on code — the controller's dmean ≈ 3.5 is an operating point the static sweep (silently broken by a CLI bug, all runs were d=8) had never measured | LANDED |
| D5-R ⑦P0 | nt=9 verify profiled (Phase 0, pre-registered stop-gate): nsys — BT-MMQ kernels = 84% of GPU time (attention ~1%, ksplit reduce 1.3%, quantize 1.7%); ncu (sudo; ERR_NVGPUCTRPERM workaround) — both BT kernels at ~15-24% SM / ~15-26% memory throughput, **smem scoreboard stall = 40–59% of warp cycles** → the <30% stop-gate says PROCEED; Phase 1 = cp.async double-buffered staging (bitwise-preserving), EV narrow (prefill lever / identity-relaxed d=8 — adaptive d≈3.5 already beats d=8 identity-safely) → re-priced separately below 战役 97 | doc 96 | gate applied on measured share; no engine change this phase | doc 90's counter-permission lead resolved (sudo) | ✅ P0 |
| D5-R ⑦✦ | speculative decoding wired into ALL frontends: `--cnv --spec-draft` (Engine trait hooks + SpecAwareEngine + a spec sibling decode loop mirroring the plain one token-for-token) and `serve --spec-draft` (per-slot draft engines, seed carry across rounds, mid-batch stop/EOG termination); **pre-existing server bug fixed in passing** — slot GraphCache reuse across different-prompt requests leaked stale KV rows into the new attention window (the plain path was contaminated too; identical back-to-back requests hid it) → per-request slot-cache + draft reset | doc 97 | conversation identity 4/4 + multi-turn 2/2 byte-identical; server identity 3 rounds × 2 reqs × (static-4, adaptive) byte-identical vs plain, finish_reason + streaming verified; suite 187/0 | spec is now uniformly available; conversation/server parity with the gen loop | LANDED |
| D5-R ⑦P1 | 96 Phase 1 lever (cp.async double-buffered staging for the q4_K BT kernel) implemented in full — A/B planes + one-group-per-tile (r56 pattern) + the dbuf regime extended to the K-split path — and **measured NULL on GB10**: q4_K BT 188.5/146.1 µs vs baseline 187.6/142.4, dbuf on/off within noise of each other, C_T ladder and pp512 unchanged; bitwise-preservation gate passed (pre == post == dbuf-off, 4/4 prompts). The 40–59% Short-Scoreboard stall is **compute-side smem dependency (ldmatrix→mma operand chains), not tile staging** — doc 92's staging attribution corrected. Remaining lever class reorders fp32 accumulation → tolerance-class → excluded on the identity-claimed path → **patch reverted, campaign closed**; C_T(9) ≈ 73 ms stands as the identity-safe floor | doc 98 | null result measured on kernel µs + C_T + pp512; suite 187/0 after revert | the staging-hypothesis line is spent; reopen only with an identity-relaxed verify knob or new hardware | ⚫ NULL / REVERTED |

### Methodology

- [77 · Verification methodology — how numbers are taken, gated, and kept
  honest](./cuda_optimization_steps/77-verification-methodology.md)

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
| `MINFER_Q6K_DPL` | on | dense split-plane q6_K decode planes not built (−2.0 GB 14B / −0.9 GB 7B) → padded-224B MMVQ path (bitwise) | D4-4 |

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

### Appendix B — Verification methodology (summary)

The full capstone — the five-gate chain (① parity ×3 via
`cuda_prefill_mmq` / `cuda_prefill` / `cuda_fa_prefill_attention_parity`, ②
greedy-32 token identity, ③ interleaved A/B medians with the +1.5%
whole-prefill bar, ④ the device suite, ⑤ the ncu/nsys/SASS protocol incl. the
GB10 metric gaps and the sudo-LD_LIBRARY_PATH gotcha) and the campaign's
transferable lessons (baseline anchoring, liveness labels, tile-size vs
greedy-identity, the pipeline-value formula, roofline-before-coding,
occupancy-before-instructions, SASS-first, stall-mass conservation,
layout-transformation locality, mechanism composition, consumer attribution,
expiring wall decompositions, phantom results, shared-box memory etiquette) —
lives in
[`cuda_optimization_steps/77-verification-methodology.md`](./cuda_optimization_steps/77-verification-methodology.md).
Read it before running any A/B on this engine.

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
