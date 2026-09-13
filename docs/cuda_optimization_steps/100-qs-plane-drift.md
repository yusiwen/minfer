# 100 · qs-plane experiment: the weight-repack lever measured NULL — and the benchmark drift that explains half the campaign's residuals

**Status**: ⚫ measured null, reverted (2026-09-15). The last priced lever
(32-B-aligned weight reads, doc 99 §4's ~7–10%) was implemented in full and
measured **timing-NULL under a controlled interleaved A/B**. The experiment
surfaced something more important: this box's short kernel benchmarks carry a
**±7–12% clock-ramp noise band** (SM idle 208 MHz → max 3003 MHz; runs never
reach steady state), which is large enough to have contaminated several
sequential comparisons across docs 92–99. Only interleaved same-state A/B
runs are valid instruments here.

## 1. The experiment (doc 99 §4's lever, implemented)

The BT kernels read q4_K qs windows from the GGUF 144-B super-block stride:
odd super-blocks start at +16 mod 32, so every other 128-B window spans 5
32-B sectors instead of 4 (ncu: 25–27% excessive global sectors). Since
d/dmin/scales already live in the W_dsc plane (r59), the B staging needs only
the qs bytes — so we built a **qs-only plane** at registration (host-side
gather, `out[j*nsb*128 + sb*128] = raw[j*nsb*144 + sb*144 + 16]`, pure data
movement), gated by `MINFER_BT_QS_PLANE=1`, passed to the BT kernel as a new
`w_qs` arg with the staging macro selecting the aligned source
(`j*nsb*128 + sb*128`, no +16 tail). +0.89× the raw q4_K bytes (~7 GB on the
14B); VRAM headroom makes that affordable.

## 2. The measurement trap this experiment sprang

- Sequential comparison #1: qs-plane build 161.5/120.0 µs vs the doc-99
  baseline 184.2/142.5 → looked like **−12.4%/−15.8%**. pp512 also drifted
  (2064 → 1890 tok/s) across the same period.
- **Interleaved A/B** (same minutes, same machine state, 2 rounds, -n 64 d=8
  run): q4_K BT totals 141.97/141.31 ms (off) vs 141.88/141.60 ms (on) —
  **NULL**. e2e -n 200 totals: 141.7 vs 142.0 ms — NULL.
- ncu instance duration: 38.0 µs (plane) vs 42.5 µs (no plane) — also
  confounded by the same drift (the two runs sat in different machine states).
- Root cause: `nvidia-smi` shows SM at **208 MHz** when idle; the benchmark
  runs are seconds long and never stabilize clocks, so where the ramp lands
  differs per run. The measured "states" differ by ~12% — larger than every
  lever this campaign has priced since doc 92.

## 3. What this recontextualizes

- The qs plane: **NULL** — reverted (doc-90 discipline). Even with 25–27%
  fewer wasted sectors, a latency-bound kernel at ~27% memory SOL does not
  convert traffic savings into time. The repack family (in-place 160-B stride
  included) inherits this null: closed.
- doc 99's swizzle (−1.8% avg, −4.3% instance): the timing delta sits inside
  the drift band; what stands is the **mechanism evidence** (the 40%
  excessive-shared-wavefront warning eliminated) and the bitwise gate. It
  stays as a verified micro-fix, now with honest error bars.
- Historical sequential numbers (doc 92's 86.4 → 74.7 → 72.9 convergence,
  doc 96's profile ratios, doc 98/99 kernel deltas) all carry the same band.
  The k-split's gain (+11.7 ms) is large enough to survive it; the rest
  should be read with ±7–12% error bars.

## 4. Method rule going forward

Every kernel-level claim on this box must come from an **interleaved A/B**
(build A, run A, build B, run B, repeat ×2) or a long warmup to steady
clocks. Sequential before/after runs are only good for order-of-magnitude
signals. This rule is the campaign's real deliverable from this phase — the
campaign's remaining residuals (~5 ms of C_T(9)) are inside the band that
this method can resolve, which is the strongest evidence yet that the
performance line is closed.

## 5. Disposition

- Reverted: the qs plane (`cuda_kernels.cu` kernel arg + staging branch,
  `cuda.rs` extern/map/expand/register, loader registration gate). Tree =
  doc 99 state (`ab75b48`).
- Not landed: anything this phase. `MINFER_BT_QS_PLANE` did not ship.
- Artifacts: `/tmp/d96p1/` (nsys `ab_{1,2}_{,on}`, `e2e_{,on}`, `qs_d8`,
  `pcs3`; the drift evidence lives in those reports).

## 6. Verification recipe

```bash
# interleaved A/B (the only valid instrument):
for round in 1 2; do for cfg in OFF ON; do
  nsys profile --trace cuda minfer <same workload>   # alternate builds
  nsys stats --report cuda_gpu_kern_sum <rep>        # compare totals, not avgs
done; done
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader   # expect ramping
```
