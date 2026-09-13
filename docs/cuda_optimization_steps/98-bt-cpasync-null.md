# 98 · BT cp.async staging (96 Phase 1) — implemented, measured, NULL; campaign closed

**Status**: ⚫ measured null (2026-09-15). The pre-registered Phase-1 lever
(cp.async double-buffered staging for the q4_K BT kernel) was implemented in
full and produced **zero kernel-level change** on GB10: q4_K BT avg/med µs and
the whole C_T ladder are within noise of the baseline, in both dbuf regimes.
The patch was **reverted** (doc-90 discipline: implemented → measured →
reverted when the gates fail). No engine change landed. The verdict closes the
bitwise-preserving lever class for the BT kernels and, with it, the D5-R
performance line at its identity-safe ceiling.

## 1. What was implemented (the full lever, not a slice)

The q6_K BT kernel already runs the target design (r39 double-buffer + r45/r53
cp.async A/B staging + r56 one-group-per-tile); Phase 1 was to bring the q4_K
kernel (64.8% of BT time) to the same standard:

- A plane (qa8/sda bulk LDG→STS) → `gemm_cp16` cp.async 16-B copies;
- B plane (qb_raw super-block copy) → cp.async with src-size zero-fill for
  rows beyond `od` / super-blocks beyond the weights (byte-identical to the
  old zero + store);
- commit placement per r56: ONE group per tile covering A + B + DSC, waited at
  the consumption point (`wait_group 1` while the next tile flies, `wait_group
  0` on the last tile) — unconditional now, since A/B are async with DSC=false
  too;
- the double-buffer regime extended to the K-split path (the C_T(9)
  configuration, previously excluded by doc 91's `ksplit == 1` gate): the
  launcher sizes the smem (86 KB) and passes `dbuf_flag`; `MINFER_NO_BT_DBUF=1`
  forces the old single-buffer regime as an A/B switch;
- bitwise preservation argued by construction and verified by gate (§3).

## 2. Debug note: the stride bug the sanity gates caught

The first build produced garbage from the *plain* path (prefill nt ≥ 9 runs
the BT kernel) in both dbuf regimes. Bisect (A-only / B-only / both reverted)
pinned it in three rebuilds: the `sda_q` copy used `sda_q + off*16` on a
`uint32_t*` — 4× the intended byte stride (the old code cast to `uint4*`
*before* indexing). Correct form: `sda_q + off*4` (uint32 elements) == byte
`off*16`. The lesson generalizes doc 90's "silent NaN looks like a fast run":
**a pointer-type change in staging arithmetic is invisible to the compiler and
to perf counters — only output sanity or identity gates catch it.** The
identity battery and a one-line prompt check are part of the harness for this
reason.

## 3. Measurements (all with the corrected patch)

**Bitwise-preservation gate — PASS.** d=8 spec outputs at −n 200 on the
4-prompt battery are byte-identical across pre-patch / post-patch dbuf-on /
post-patch dbuf-off (4/4 × 3 builds). The patch never changed an output; the
d=8-vs-sequential tolerance-class differences (doc 95) are pre-existing and
unchanged.

**Kernel level (nsys, d=8 run, 64 tokens, 873 q4_K instances) — NULL.**

| build | q4_K avg µs | q4_K med µs | q6_K avg µs |
|---|---|---|---|
| doc 96 baseline | 187.6 | 142.4 | 301.7 |
| patch, dbuf ON | 188.5 | 146.1 | 306.8 |
| patch, dbuf OFF | 188.4 | 146.5 | 305.1 |

**C_T ladder (specverify −p 512 −r 40)**: 39.22 / 48.55 / 56.30 ms at
nt = 1/3/5 — baseline within noise (doc 92: C_T(3) 48.1, C_T(5) 55.8); these
points run multi-MMVQ and were never in scope. **pp512 = 2104 tok/s** (doc-92
baseline 2005) — no prefill regression from the cp.async restaging.

## 4. Verdict and mechanism read

The lever is refuted at the mechanism level, not just the outcome level: if
the 40–59% Short-Scoreboard stalls measured in doc 96 Phase 0 were
*staging-side* (the LDG→STS tile load), handing the tile loads to the async
unit and overlapping them with compute would have shown up here. It did not —
in either regime, and the dbuf on/off pair differs by less than noise, which
also says the extra 43 KB (and the 4→2 blocks/SM residency cost doc 91 feared)
is irrelevant because there is nothing to gain.

The stalls therefore live in the **compute-side shared-memory dependency
chain**: the `ldmatrix` A-fragment fetches, the per-(kd, nh) `qb_raw`/`sds`/
`sda` LDS reads, and the mma accumulation sequence — i.e., doc 92's "per-tile
staging serialization" was a *misattribution by proximity*: the stall is inside
the tile *consumption*, not the tile *arrival*. The remaining lever class —
larger k-tiles per ldmatrix, restructured operand schedules, deeper mma
pipelining — reorders the fp32 accumulation → tolerance-class by construction
(doc 96 §4 predicted exactly this trade). On the identity-claimed path that
class is excluded, and the d=8 door stays closed on performance grounds:
**C_T(9) ≈ 73 ms stands as the identity-safe floor, multi-MMVQ remains the
production verify path, and doc 95's adaptive d (dmean ≈ 3.5) remains the best
identity-safe speculative configuration.**

**Campaign closed.** Re-opening conditions, for the record: (a) the project
ships an identity-relaxed ("fast, transcript-unstable") verify knob — then the
kernel-redesign item from doc 92 (tolerance-class) becomes priceable; or (b) a
future GPU changes the smem/mma latency ratio enough to re-ask the question.

## 5. Disposition

- Kernel patch reverted; `MINFER_NO_BT_DBUF` did **not** ship; suite 187/0 on
  the reverted baseline; prefill sanity re-verified.
- Measurement artifacts: `/tmp/d96p1/` (nsys reps + identity battery outputs).

## 6. Verification recipe

```bash
# kernel-level A/B (null expected)
nsys profile --trace cuda minfer --greedy -n 64 --spec-draft <0.5B> --spec-draft-n 8 <14B> "<code prompt>"
nsys stats --report cuda_gpu_kern_sum <rep>   # mmq_raw_nb_bt_kernel avg/med

# bitwise-preservation across builds
cmp <(pre-patch d=8 output) <(post-patch d=8 output)   # byte-equal
```
