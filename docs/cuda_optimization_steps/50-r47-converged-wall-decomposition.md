# 50 · r47 — converged-era whole-wall re-decomposition (MEAS-ONLY)

> **Result**: r37's attribution table expired exactly as forecast. Re-measured on the current best gate set after the whole r38–r45 q6_K package landed: q6_K GEMM **1094.7 → 196.4 ms** (51.2% → 15.8% of the wall), the whole wall 2190 → 1274 ms (1521 → 2610 tok/s, same-window interleaved band 2585–2623), vs-llama 2.15× → **1.27×**. q4_K at 33.2 µs/GMAC (**1.06×**) and q6_K at 65.0 µs/GMAC (**1.13×**) both reached parity — the GEMMs are no longer the wall; **FA at 125.8 ms = 5.72× became the #1 structural residual** (10.2% of GPU busy), falsifying r46's "smaller/overlapped" assumption; the quantize prepass grew **+31.6 ms** (the hidden tax of the q6_K BT port). Recommended next lever: FAP2 register-resident softmax (estimated at 2× → −63 ms = **−4.9% of the wall**, clearing the +1.5% bar), followed by the A-quantize shared-A dedup.
> **Commit**: `11e3640` (docs-only record commit, no code change). **Date**: 2026-09-05.

## 1. Background — where things stood

The r38–r41 q6_K campaign was a string of textbook landings: the BT-style raw-byte kernel (+2.87%), KDR=2 double buffering (+13.3%), the third resident block (+13.0%), the B-expand uint4 widening (+30.7%) — four steps pulled q6_K from 368.9 µs/GMAC down to the 65 µs magnitude and pushed whole-prefill from 1521 all the way to 2605 tok/s. But the next three levers hit the wall in succession: r42 (dsc widening −0.19%), r44 (the W_exp stride fix went parity green but the wall −0.42%), r45 (cp.async A-side staging, wall −0.34%) — all REVERTED. r45's conclusion was blunt: **the q6_K line has converged — the kernel is no longer the binding constraint, and tuning the q6_K kernel further cannot move the wall**.

r46 (FAP1) supplied another instance of the same lesson. The audit found the FA kernel (`fa_prefill_f16kv`) already wmma + online softmax, with the bottleneck being occupancy (69.38 KB smem → 1 block/SM) and the S/P smem round trip's bank conflict; after the fix the kernel went 5.16 → 4.58 ms (−11%), fully mechanism-positive, yet **the wall moved only +0.27%** — below the +1.5% bar, REVERTED. "Mechanism-positive / wall-negative" became a real outcome class: whether a kernel-level 2× improvement opportunity is worth pursuing depends on its share of the wall and on whether it is really on the critical path.

This is exactly where r37's rule applies: **"attribute the WHOLE wall after every convergence"**. r37's attribution table drove the entire r38–r41 priority queue (q6_K raw kernel first), and that table was drawn in a world of 2139.6 ms GPU busy: q6_K 51.2%, FA 5.7×, prepass 86.8 ms. Eight landing rounds later, both uses of that table had expired — the denominator (the whole wall) shrank 40%, and every slice's share drifted. Without redrawing the table, the next lever choice would rest on stale data.

Two candidate directions sat on the table, neither backed by current wall-level evidence: one is the FAP2 r46 named (register-resident softmax, deleting the S/P smem round trip outright), the other the quantize prepass redundancy r37 had already flagged ("1.25× llama, partly 2× per-shared-A redundancy"). r47 is a pure measurement round: **without changing a line of code, slice the whole wall open under the current best gate set and rank the two candidates with the freshest numbers.**

"Current best gate set" has concrete content at this moment: the P5-era TM=128 wide-tile GEMM, r20's split-phase A staging, r22's qa8 XOR swizzle, the r28–r32 NB kernel family, r34's quantize-transpose prepass, and the r38–r41 q6_K BT package (KSPLIT=2 raw-byte kernel, KDR=2 double buffer, 3 resident blocks, uint4 B-expand), all on by default; the REVERTEDs of r42/r44/r45 and r46's REVERTED mean the FA and q6_K kernels sit at their respective convergence points. In other words, this table portrays "the engine after every proven lever has been pulled" — precisely why it deserves the word "converged" — and precisely because of that, its differences from r37's table can be attributed directly to the four rounds r38–r41, with no in-flight changes mixed in.

One more methodological detail in the r46→r47 handoff worth recording: r46's grounds for vetoing the FA lever were "small slice share + mechanism in doubt", but that judgment used r37-era shares. The question r47 must answer is therefore sharp — **after the GEMM collapsed from 51.2% to parity, is FA's 125.8 ms still "small"?** The answer is no (10.2% busy, 5.72× ratio), which is exactly the value of re-measuring: a share judgment's shelf life does not outlast one convergence.

## 2. Principle — the GPU mechanism

**The attribution's dimensions.** One decomposition = run nsys over the complete prefill forward and group GPU busy time by kernel role: each weight type's GEMM (q4_K / q6_K / f16…), attention (FA), the quantize prepass, and the elementwise/copy long tail. Normalization within a role uses **GMAC**: a matmul class's slice time ÷ that class's total GMAC count gives µs/GMAC. The point of GMAC normalization is stripping shape — q6_K's attn_v and ffn_down differ by an order of magnitude in size, but µs/GMAC is directly comparable and directly matches llama.cpp's reference implementation's same-class numbers; the ratio of the two is **that line's parity multiple** (the llama reference is provided by the bench build `ca3d5a3e1`).

Worked example (r37's q6_K row): for the 3325-token prefill, the q6_K matmuls (attn_v + ffn_down, 28 layers) have a fixed total GMAC count, and 1094.7 ms divided into it = **368.9 µs/GMAC**; llama at the same shape is 57.8 µs/GMAC, a ratio of 6.38×. r47 re-measures: 196.4 ms ÷ the same GMAC total = **65.0 µs/GMAC**, a ratio of 1.13×. Note the numerator shrank 5.6× between the rounds while the denominator (the GMAC total) varies only with token count and both rounds share the anchor — so the ratio's improvement comes entirely from the numerator. That is what normalization buys: **separating "the wall got smaller" from "the kernel got better"**. Likewise q4_K: 598.1 ms ÷ its GMAC total = 33.2 µs/GMAC, 1.06× vs llama.

**The denominator effect: shares squeeze each other by nature.** r44 supplies a miniature sample of this principle: inside the q6_K kernel, after fixing the recomb the kernel duration fell −10.9%, but long_scoreboard's **share** rose from 33.6% to 57.1% — the numerator fell, the denominator fell with it, and the remaining stall's share rose instead. The wall-level decomposition behaves the same: after the q6_K slice collapsed, FA's and prepass's shares rise passively even if their absolute values are unchanged (124.7/2190 = 5.7% → 125.8/1239.2 = 10.2%, FA's absolute duration differing by only 1.1 ms). When reading a decomposition table you must read absolute durations and shares together, or you will misread "the denominator effect" as "the residual worsening".

**Why decompositions expire.** The wall is a denominator; a landing changes the numerators and the denominator. At r37, q6_K held 51.2%; four q6_K rounds pressed it to 196.4 ms, but the whole wall also shrank from 2190 to 1274 ms — if nothing outside q6_K had changed, its share would have fallen from 51.2% to 196.4/1274 = 15.4%; the actual 15.8% says the other slices barely moved. **Ranking is relative**: once r37's top lever was done, the second tier (FA 5.7×, prepass redundancy) automatically moved to the front of the window, but their absolute shares must be re-measured — r46 had already proven FA's 124.7 ms slice "not worth it" in the then-GEMM-dominated world; now that the GEMM has collapsed to parity, the same 125.8 ms slice is the biggest non-parity residual.

**Why 5.72× is called "structural".** This multiple is not r47's discovery — r23 named it structural back in the f16-path decomposition era ("FA's 2.5×/layer gap is structural: llama keeps 128-wide KV tiles"): llama's attention tile geometry holds the KV panel at a completely different width, so the per-layer attention gap on the two sides is decided by tiling choice, not instruction efficiency. r37 measured 5.7×, r47 measured 5.72× — across all the GEMM landings of r38–r45 this number did not move an inch, which is exactly what validates the word "structural": **nothing on the GEMM side reaches it**. It also predicts two things: per-instruction micro-tuning will not converge it (r46's padding route was tried and failed), and only a geometry-class rewrite (r48's whole-row warp tile) has a chance.

**Hidden taxes must be booked.** A landing's net contribution to the wall = its direct gain − the indirect cost it introduces. r38–r41's direct gain sits on the q6_K slice: 1094.7 − 196.4 = 898.3 ms. But r47 finds the quantize prepass rose from 86.8 to 118.4 ms (**+31.6 ms**): the q6_K BT port also connected attn_v's and ffn_down's A side to r34's `quantize_q8_0_pad40_t` prepass, and the quantization work that used to run on the generic `mmq_nt<7,2>` path was "moved" into the prepass. Net wall gain = 898.3 − 31.6 = **866.7 ms** — the direction unchanged, but the per-lever accounting is only honest with the tax included.

**Converting shares into budget.** Once you have the shares, estimating a candidate lever's ceiling is one line of arithmetic. FAP2's claim is that deleting the S/P smem round trip makes the FA kernel ~2×: 125.8 ms → ~63 ms, saving 63 ms; 63 / 1274 = **4.9%** of the wall, more than three times the +1.5% bar. The prepass dedup's claim is that q/k/v and gate/up each share one A (see doc 52 for detail), estimated at the ~40–60 ms order from r37's "~2× redundancy" reading — both candidates clear the bar, and the ranking depends on which mechanism is more certain.

| Candidate lever | Mechanism claim | Budget arithmetic | vs the bar |
|---|---|---|---|
| FAP2 register-resident softmax | delete the S/P smem round trip → FA kernel ~2× | 125.8/2 ≈ 63 ms = 63/1274 ≈ 4.9% of wall | threefold headroom, **#1** |
| A-quantize shared-A dedup | same A not re-quantized → prepass slims down | redundancy ~43% of launches × 0.6 ms each ≈ 40–60 ms | clears the bar, **#2** (simpler mechanism, lower risk) |
| q4_K short-nt dilution | matched-nt 1.17× reading | actually 1.06× at prefill nt — nothing to harvest | **shelved** |

The most informative line of the budget arithmetic is the third: **the biggest slice, at a 48.3% share, has a budget of zero** — because its ratio is 1.06×. The decomposition table's whole value is putting "share" and "ratio" side by side; missing either one misranks the queue.

## 3. Implementation

### 3.1 Measurement design: zero code changes, same window, same tools

r47 is a deliberate "no-code" round: the repo tree sits at the post-r45/r46-convergence state (r46's change already reverted), and the only artifact is the docs commit `11e3640`. The measurement protocol follows r37's template:

- **Load**: the 3325-token prefill (the same anchor as r37), which keeps µs/GMAC directly comparable with r37;
- **Tools**: full-graph nsys for per-kernel durations and the GPU busy total; matched-nt ncu for each weight type's per-IMMA/µs-GMAC ratio;
- **Gate set**: every landed lever (the q6_K BT package, TM=128, the MMQ paths, etc.) on by default, with no A/B variable — this table measures "what the engine looks like now", not "the increment of some change";
- **Same-window interleave**: whole-prefill rates sampled interleaved within one session window, reporting the median and the band (2585–2623), honoring the master table's reading convention — cross-session absolute values are not comparable.

The evidence actually collected (item-for-item isomorphic to r37, keeping the two tables mutually readable):

- the nsys full-graph per-kernel duration table → bucketed and summed by (weight-type GEMM / FA / prepass / elementwise), giving each slice's absolute milliseconds;
- ncu matched-nt (the bench shape at nt≈511) → each weight type's per-IMMA / µs-GMAC ratio, normalized against the llama reference;
- whole-prefill interleaved A/B (same binary, same window) → the 2610 tok/s median and the 2585–2623 band;
- no A/B variable, no env flips — this round's only "control" is r37's old table itself.

### 3.2 r37 vs r47: the two tables read side by side

| Slice | r37 (2026-09-05, morning) | r47 (same day, after r38–r45) | Change |
|---|---|---|---|
| GPU busy | 2139.6 ms | 1239.2 ms | −900.4 ms |
| Whole wall | 2190 ms (1521 tok/s) | 1274 ms (2610 tok/s) | −916 ms |
| vs-llama | 2.15× | **1.27×** | −0.88× |
| q6_K GEMM | 1094.7 ms = 51.2% (368.9 µs/GMAC, 6.38×) | **196.4 ms = 15.8%** (65.0 µs/GMAC, 1.13×) | −898.3 ms, parity reached |
| q4_K GEMM (BT) | ~600 ms (60.0 TFLOPs, matched-nt 1.15×) | 598.1 ms = 48.3% (33.2 µs/GMAC, **1.06×**) | flat, now parity-class |
| FA attention | 124.7 ms (5.7×) | **125.8 ms (5.72×) = 10.2% busy** | flat, promoted to #1 structural residual |
| quantize prepass | 86.8 ms (1.25× llama) | **118.4 ms = 9.6% busy** (+31.6 ms) | the hidden tax of the q6_K BT port |
| Rest (elementwise/rope/store_kv/lm_head/copy long tail) | ~230 ms | ~200 ms | the four named slices total 1038.7 ms; the remainder is the long tail |

Three points from reading them together:

1. **q6_K's win is real, but must be netted**: 898.3 ms of direct gain minus the +31.6 ms prepass tax is a net +866.7 ms, the overwhelming majority of the GPU busy reduction (900.4 ms) — the q6_K line really was the only big mover of these eight rounds.
2. **The biggest slice is already parity-class**: q4_K's 598.1 ms is the largest single slice at 48.3%, but 1.06× means llama could not be much faster either — **a big share ≠ a lever**. Residual ranking must read share and ratio together: FA is the largest product of the two (10.2% × 5.72×).
3. **The prepass went from background noise to the third-biggest slice**: 9.6% busy, with a clear mechanistic redundancy (the same A quantized repeatedly) — r49's foreshadowing.

The decomposition's self-consistency can be written directly as an equation (STYLE's "if a diagram doesn't fit, write the equation" usage):

```text
GPU busy 1239.2 ms
  = q4_K GEMM   598.1  (48.3%  — 33.2 µs/GMAC = 1.06×, parity class, no lever)
  + q6_K GEMM   196.4  (15.8%  — 65.0 µs/GMAC = 1.13×, converged this round)
  + FA attn     125.8  (10.2%  — 5.72×,        #1 structural residual)
  + prepass     118.4  ( 9.6%  — ~43% launch redundancy, #2)
  + long tail   ~200.5 (16.1%  — rms/swiglu/rope/store_kv/lm_head/copies)
wall 1274 ms = busy 1239.2 + launch gap/D2H ~35 ms   →  3325 tok ÷ 1.274 s ≈ 2610 tok/s
```

The q4_K row deserves a pause: 1.06× does not mean "we are only 6% behind"; it means "the llama reference is simply this fast on this class of matmul" — jointly determined by the physical limits of IMMA throughput, smem bandwidth, and DRAM traffic. At this point q4_K's 598.1 ms stops being an "optimization target" and becomes "terrain": any proposal to shave time off it must first explain how it would beat the reference by more than 6%. This is also why after r47 the campaign never touched the q4_K kernel body again (r58's failed transplant was a structure port, not kernel tuning).

The four named slices total 1038.7 ms, **83.8%** of the 1239.2 ms GPU busy; the remaining ~200 ms is the elementwise and copy long tail (rms/swiglu/rope/store_kv, embedding, lm_head, and the GPU-side work ahead of the D2H readback). Later rounds proved this long tail worth slicing too: r51/r52 measured its fused-producers slice at 151.9 ms and cut it to 86.5 ms with producer fusion (−5.45% of the wall), and r55 first used a roofline to prove swiglu already sits at 89% of the bandwidth roof (242 GB/s) — **the long tail is not miscellaneous; it is an unranked pool of candidate levers**; r47 leaving it as a watch item was the right restraint.

### 3.3 Pitfalls: two attribution traps

**Pit one: hidden taxes not booked.** Looking only at the q6_K slice (−898.3 ms) and not the prepass (+31.6 ms) overstates the q6_K campaign's books by 3.5%. The prepass's growth mechanism is subtle: it is not that some commit "got slower", but that r38's BT port changed **which work takes the prepass road** — attn_v/ffn_down's A quantization migrated from the generic kernel path to the shared prepass. This class of "work relocation" is exposed only by the difference between two decompositions.

**Pit two: taking the matched-nt dilution reading at face value.** matched-nt ncu measures q4_K per-IMMA at 1.17× on the nt≈511 bench shape (even 1.43× in the r37 era) — far worse than the 1.06× at prefill nt. r47 explicitly rules this a **short-nt-specific prologue dilution effect** (prologue/staging overhead amortized over a small nt), not a real optimizable gap, and shelves it. The lesson is isomorphic to r46's: **kernel-level ratios must be read at the right shape**, or you schedule phantom levers.

**Pit three: reading cross-session absolute values as increments.** 1521 (r37's window) and 2610 (r47's window) are not two numbers from the same machine in the same state — co-tenant load and clock policy both move absolute anchors (the master table reading convention was later reinforced painfully in r59b: "anchor every A/B baseline behaviorally in the same window"). r47's numbers are all same-window interleaved medians; cross-round comparison runs only on **ratios and shares**, never on absolute tok/s differences. This discipline cost nothing this round and was worth a +26.2% → +11.1% correction in r59.

## 4. Verification

This is a measurement round; what is "verified" is the decomposition's own credibility:

- **Same-window interleaved median**: whole-prefill reported with its same-window band of 2585–2623 tok/s, avoiding single-sample machine-state noise (the master table reading convention + a forerunner of r59b's lesson);
- **The busy-vs-wall reconciliation**: GPU busy 1239.2 ms vs wall 1274 ms, the ~35 ms difference being launch gap and D2H readback — a reasonable magnitude, showing the slice summation missed no big item;
- **Slice-sum self-consistency**: the four named slices 1038.7 ms + long tail ~200 ms ≈ busy 1239.2 ms, with each slice's duration from nsys's per-kernel sums cross-checked against ncu's per-kernel duration × launch count (r37 protocol's two independent sources interlocking);
- **Anchor regression**: vs-llama uses the same-window llama-bench 3325-eq anchor (the ~3400 tok/s family), 2610/3400 ≈ 1.27×, subtractable in the same coordinate system as r37's 2.15×;
- **r48's retrospective check**: r47 predicted FAP2 2× → −63 ms; r48 measured FA kernel 5.16 → 2.12 ms (2.43×) and whole-prefill 2603.5 → 2749.9 (+5.6%, ~68 ms of wall saved) — the predicted −63 ms and the measured saving agree within noise. The decomposition's ranking was cashed in by the subsequent landing.
- **r49's retrospective check**: three mutually corroborating levels of independent evidence for the prepass redundancy (r37's 1.25×-llama annotation → r47's 9.6% busy → r49's 193→110 census and +2.32%) — r47's "mechanistic redundancy" judgment lands precisely in the launch census.

## 5. Results

**The measured wall (3325-token prefill, current best gate set, no code changes)**:

- GPU busy 1239.2 ms, whole wall 1274 ms, **2610 tok/s** (interleaved band 2585–2623), vs-llama **1.27×**;
- q6_K GEMM 1094.7 → 196.4 ms (51.2% → 15.8%), 65.0 µs/GMAC = **1.13× (parity-class)**;
- q4_K GEMM 598.1 ms = 48.3% busy, 33.2 µs/GMAC = **1.06× (parity-class)**;
- **FA 125.8 ms = 10.2% busy, 5.72× — the #1 structural residual**; r46's "FA smaller/overlapped" assumption does not hold (the slice surfaced intact after the GEMM's collapse);
- quantize prepass 118.4 ms = 9.6% busy (+31.6 ms since r37, the hidden tax of the BT port).

**The produced priority queue** (r47's direct deliverable):

1. **FAP2 register-resident softmax**: estimated 2× → −63 ms = −4.9% of the wall, threefold headroom over the bar — r48 subsequently cashed it at +5.6% (2603.5 → 2749.9);
2. **A-quantize shared-A dedup**: q/k/v share the same `normed`, gate/up share the same `normed2`, and the prepass re-runs per matmul — r49 cashed it at +2.32% (2734.1 → 2797.5);
3. q4_K matched-nt dilution: a short-nt-only effect, **shelved** (later restarted in another form in r58's q4_K BT structure spec).

The prediction-vs-cash-in comparison (this table is the measurement round's "correctness verification" — a wrong ranking would have wasted every later round):

| r47 prediction | Cashing round | Predicted | Measured |
|---|---|---|---|
| FAP2 2× → −4.9% wall | r48 | −63 ms | +5.6% (~68 ms of wall saved) ✓ |
| prepass shared-A dedup | r49 | the ~40–60 ms order | +2.32% (−34.5 ms prepass, ~28 ms of wall cashed) ✓ |
| q4_K dilution shelved | r58 (restarted) | — | the q4_K BT port −12.6% REVERTED — the "shelved" judgment held in r58's window too |

Campaign coordinates: 1521 (r37) → 2610 (r47) → 2750 (r48) → 2798 (r49) → 2856/3011/3177 (the prepass-and-q6_K bundle line of r51/r52/r53) → 3590.8 (r59b's final figure); vs-llama 2.15× → 1.27× → 1.21× → 1.18× → 1.05× → **1.080×**. r47 sits at the inflection of this curve: it confirmed both GEMM lines at parity and switched the campaign from "fix the slowest kernel" to "harvest the remaining structure by wall share" — every landed gain of the following six rounds (r48/r49/r51/r52/r53/r56) finds its slice in this decomposition table.

Equally worth recording is what r47 deliberately **did not** do: it did not subdivide the ~200 ms long tail (which elementwise kernels hold what, how far swiglu sits from the bandwidth roof), merely flagging it as a watch item. Subdividing the tail was r51/r52's (the 151.9 ms pre-producer-fusion attribution) and r55's (the swiglu 242 GB/s = 89% roofline capping audit) work. One decomposition needs to answer only "what is next"; slicing every slice to the leaves is the next round's job — **the decomposition's granularity follows the decision, not completeness**.

## 6. Lessons

1. **Wall decompositions have a shelf life**: after every convergence (a line reaching parity, or a kernel leaving the critical path) the attribution must be re-run, or the priority queue rests on stale shares — r37's table served r38–r41 precisely, r47's table served r48–r49 precisely, and no table spans two convergence periods.
2. **Net the landings**: direct gain minus hidden tax (q6_K's −898.3 ms paired with prepass's +31.6 ms) is the campaign-level accounting; "work relocation" taxes surface only in the differential between before/after decompositions.
3. **A big share ≠ a lever**: rank residuals by share and parity ratio together; the biggest already-parity slice (q4_K at 48.3%) has no room to move; mechanism-positive/wall-negative (r46) and mechanism-positive/share-not-yet (FA at r37's time) are both real outcome classes.
4. **Read kernel-level ratios at the right shape**: the matched-nt dilution (1.17× @nt≈511 vs 1.06× @prefill nt) is a prologue-amortization shape artifact; not taking it seriously saves an entire wrong optimization line.

---
← 49 · [Index](./README.md) · [51 →](./51-r48-fap2-register-softmax.md)
