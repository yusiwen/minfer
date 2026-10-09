# 0020. A quantized file is byte-identical to `llama-quantize`, or it is wrong

- Status: Accepted
- Date: 2026-09-27
- Issues: #140, #203, #205

## Context

The engine could **read** the K-quants (`q4_K`, `q5_K`, `q6_K`) before it could write them, and
`minfer quantize --type q4_K` refused by name rather than emit unverified bytes: *"minfer has no
weight encoder for it (minfer can only read it, so writing it would emit wrong weights)"*. The reason
for that caution was measured: the first `q4_0` encoder differed from `llama-quantize` in **12 of
64512 bytes** on one tensor — every difference a single nibble off by one, caused by fused
multiply-add contraction.

A quantized file is not a lossy artifact you can be "close" on. It is a weight file: a near-miss is a
different model.

## Decision

**Every quantized target is byte-identical, per tensor, to `llama-quantize`'s output**, and the
K-quant reference is explicitly `llama-quantize --pure`.

- The legacy targets (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`) are **290/290 tensors byte-identical**;
  the K-quants reached the same **290/290** in #140.
- **`--pure` is the reference because `minfer` writes one uniform type.** `q4_K`/`q5_K` are the CLI
  aliases for `LLAMA_FTYPE_MOSTLY_Q4_K_M`/`_Q5_K_M` in `llama-quantize`, but `minfer quantize --type
  q4_K` puts *every* encodable 2-D tensor at `q4_K` — which is what `--pure` writes, and the mixture
  planner (per-layer Q6_K bumps, the OUTPUT/tied-embedding branch, `use_more_bits`) is **not**
  implemented.
- Rows that cannot take a K-quant are demoted the way llama.cpp does it: `row_len_fallback` maps
  `Q4_K → Q5_0`, `Q5_K → Q5_1`, `Q6_K → Q8_0`, and to `f16` when even a 32-element block does not
  divide the row (`K_QUANT_BLOCK = 256`). On the 0.5B, hidden 896 = 3.5 × 256, so **145 of 290
  tensors take the demotion** and only the 24 `ffn_down` tensors reach the K encoder.
- The gate's **inputs** are recorded, not just its result: `docs/f6-fixtures.json` carries per
  artifact the path, bytes, sha256, the exact producer command, the producer's identity (minfer
  commit, or llama.cpp commit + compiler + effective `-ffp-contract` + cflags), date and an absolute
  box label; `scripts/check_f6_fixtures.py` audits that manifest in CI and carries tamper cases, and
  every F6 gate verifies the fixture it resolves.

## Alternatives considered

- **Reference without `--pure`.** Rejected because that file is a *mixture*: the mixture planner is
  not implemented ([#203](https://github.com/yusiwen/minfer/issues/203), still open), so a
  `q4_K` file from `llama-quantize` without `--pure` is not what this command produces.
- **Accept a tolerance — "a near-miss".** Rejected in the ticket's own words: *"A K-quant encoder that
  is not byte-identical to llama.cpp's is a wrong file, not a near-miss."* The FMA case is the worked
  example: switching to `f32::mul_add` reproduced the reference and the difference went to **zero**.
- **Trust the fixture cache.** Rejected by [#205](https://github.com/yusiwen/minfer/issues/205): a
  stale `~/.cache/minfer/f6-src/*.gguf` would be compared against silently and the gate would stay
  green — hence the recorded provenance and the per-gate fixture verification.
- Not an alternative but a recorded residual: for **legacy** targets `minfer` keeps the source type
  where llama.cpp demotes it (`q4_0` → `F16`), so an f32 source with such a row would differ
  ([#204](https://github.com/yusiwen/minfer/issues/204), open).

## Consequences

- `minfer quantize` output is interchangeable with llama.cpp's for the supported types, and the claim
  is machine-checkable rather than asserted.
- The claim is **build-dependent**, and the record says so: it is conditional on the compiler, the
  effective `-ffp-contract`, and the llama.cpp revision. Measured 2026-10-07, the Mac's reference
  disagrees with minfer in **168 of 290 `q4_0` tensors** — every difference a single data nibble, zero
  scale bytes (12 of 64512 bytes on the tensor examined). The gate therefore detects the build
  (`quantize::FmaContract`, `quantize_row_with(…, Off)`), reports five named outcomes, and **skips
  loudly** for a recorded foreign build instead of reporting an encoder defect.
- The K-quant half is additionally compiler-sensitive
  ([#349](https://github.com/yusiwen/minfer/issues/349)): the encoders were ported from disassembly,
  and one of the four findings was that *a scalar loop and its vectorized twin can disagree* —
  matching only the scalar form left 137 of 424 random 16-element groups differing.
- `sha2` is a **dev**-dependency, so the production graph is unchanged by the fixture discipline.

## References

- `docs/GGUF-TOOLING.md` — the writer contract, the `--pure` reference, the demotion rule, the
  tolerances and the fixture manifest.
- `docs/f6-fixtures.json` + `scripts/check_f6_fixtures.py` — the recorded inputs and the auditor.
- `src/quantize.rs` — `row_len_fallback`, `FmaContract`; `src/tooling/` — the `quantize` command.
- Commits: `34b0dd8` (2026-09-27, the K-quant encoders, #140); the legacy byte-identity landed with
  F6 on 2026-09-24.
