# 104 · q8_0 decode reaches the DRAM ceiling — llama.cpp parity at tg, spec +14% more

**Status**: ✅ landed (2026-09-14). Follow-up to doc 103: after the MMVQ port,
7B Q8_0 decode sat at 0.94× of llama.cpp (28.79 vs 30.68 tok/s) while q4_0
was at 1.045×. Profiling located the difference inside the kernel, and one
load-pattern change closed it to **statistical parity** (31.89–32.09 vs
32.05–32.08, same-window interleaved) plus **+14.3% more on the speculative
path** (69.1 → 79.0 tok/s e2e).

## 1. Where the 7% was (nsys + microbench evidence)

- Graph-node nsys on a 7B Q8_0 decode: **97% of GPU time is the two q8_0
  MMVQ kernels** (49.8% nt=1 + 46.9% multi); attention ~1%. The gap had to
  be kernel-internal.
- Cold-L2 rotating microbench (6 weight sets, working set ≫ L2): the doc-103
  raw-34B kernel streams 228–250 GB/s across the 7B shapes while the q4_0
  kernel (same structure) does 233–259 — and llama.cpp's own MMVQ hits 253
  GB/s on the one unambiguous shape (lm_head, 1 token/launch), i.e. the
  measured doc-99 probe ceiling.
- Root cause: the 34-byte block stride forces **16 two-byte weight loads per
  32-element unit** (the payload is only 2B-aligned). At a 34-byte lane
  stride each load instruction scatters across ~34 32B sectors, so q8_0 pays
  ~2× the L1TEX wavefront work per weight byte that q4_0 (8 loads/16 B)
  pays. The padded-36B variant was measured and REJECTED first: +6% traffic
  cancels the alignment win (233–250 GB/s, no better).

## 2. The lever: p32 split planes (new code only)

Repack each q8_0 tensor at registration (host, like the q6_K padded
precedent) into two planes:

- **payload plane** — 32 B/block, 16B-aligned → **two `uint4` loads per
  unit** instead of 16 scattered u16s;
- **d plane** — dense 2 B f16/block (one u16 load).

Total traffic is unchanged (34 B/block); the raw registration stays
untouched for the f32 fallback and every existing consumer. Arithmetic is
identical — same int8 values, same dp4a order, same reduction — and the
harness verifies **byte-equal outputs** against the raw kernel on all five
7B shapes. Dispatch prefers the planes when registered
(`MINFER_NO_Q80_P32=1` reverts both the build and the dispatch). The multi
variant additionally hoists the weight words out of the token loop (doc
103's multi re-read them per token from L1) — bitwise per token.

Memory cost: the planes add ~+94% of the q8_0 weight bytes (7B Q8_0: 7.5 →
~14.6 GB device; GB10's 128 GB is unaffected). The method self-gates:
id ≥ 2048, id % 32 == 0.

## 3. Measured (interleaved steady-state A/B, ±0.02–0.05 tok/s)

| path | 7B Q8_0 tg128 | note |
|---|---|---|
| f32 kernel (pre-doc-103) | 28.47 | |
| raw MMVQ (doc 103) | 29.89 / 29.94 | |
| **p32 (this doc)** | **31.92 / 32.02 / 31.89 / 32.09** | **+7.0% over raw, +12.3% over f32** |
| llama.cpp, same window | 32.08 / 32.05 | **0.99–1.00× — parity** |
| k/v shape (od 512) kernel | 6.14 → 4.14 µs | −33% |

Speculative (7B Q8_0 + 0.5B Q4_K_M draft, adaptive, −n 200 e2e):
**69.1 → 79.0 tok/s (+14.3%)** = **2.47× sequential** (doc 103: 2.36×).

## 4. Verification

- Suite **187 / 0 / 3** (one transient cross-binary CUDA-context flake on a
  full `cargo test` — clean on rerun, the doc-101 era pattern).
- Identity battery on Qwen3-0.6B Q8_0: **4/4 byte-identical** (spec ==
  sequential; exercises the p32 multi-vs-nt1 bitwise consistency).
- Harness bitwise gate: p32 outputs byte-equal to the raw kernel on every
  shape (0 diffs).
- Microbench also measured and **rejected** a q4_0 p32 variant (no
  consistent gain — q4_0's 8-loads/16B pattern is already efficient), and a
  2-block-per-thread ILP variant (no gain, correctness caveat at nb > 128).

## 5. Where this leaves the legacy-quant line

| quant | decode nt 1–8 | vs llama.cpp (same file) |
|---|---|---|
| Q4_0 | MMVQ + multi (doc 103) | **1.045× — faster** |
| **Q8_0** | **MMVQ + multi + p32 planes (this doc)** | **~1.00× — parity** (0.87× → 0.94× → 1.00×) |

Both legacy quants now sit at or above llama.cpp's decode, and both engines
are bounded by the same DRAM ceiling (~253–266 GB/s streaming on this box) —
further decode gains for ANY quant require raising the streaming ceiling
itself, not better kernels. The prefill gap (0.21–0.24×) remains the open
front for these types.
