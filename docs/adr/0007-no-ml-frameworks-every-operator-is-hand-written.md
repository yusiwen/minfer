# 0007. No ML frameworks: every operator is hand-written

- Status: Accepted
- Date: 2026-06-24

## Context

The engine was started from scratch rather than as a wrapper. The premise is stated
positively in three places and has not changed since: "A minimal local LLM inference engine built
from scratch in Rust" (`README.md`), "minfer is a pure-Rust LLM inference engine with no ML
framework dependency" (`README.md` §Architecture), and "A pure-Rust LLM inference engine written
from scratch, inspired by [llama.cpp]" (`docs/ARCHITECTURE.md`). The decision is dated at the
initial commit (`95fd5d1`, 2026-06-24), because that is when the premise became the repository's.

## Decision

**No ML framework is a dependency, and no operator is delegated to one.** Everything in the
inference path is this repository's own code:

- quantized block types are `repr(C)` and laid out after `ggml-common.h` (`src/block.rs`);
- the dot kernels are hand-written SIMD per architecture — AVX2/FMA and AVX-512/VNNI on x86_64,
  NEON with `SDOT` inline asm on aarch64 — each with a scalar fallback (`src/quants/`,
  `src/vec_ops/`);
- activations are quantized on the fly (`Q8_0`, `Q8_K`) and the CPU path is
  `dot_q*(weight, q8_activation)`;
- the GGUF v3 parser and writer are the project's own (`src/gguf.rs`, `src/gguf_write.rs`).

The dependency set is correspondingly small and has been from the first commit (then `rand`,
`regex`, `half`): `rand`, `regex`, `half`, `serde` + `serde_json`, `minijinja`, and for the server
`axum`/`tokio`/`tower-http`/`uuid`. `README.md` states it as "**No ML framework** — pure Rust,
minimal runtime deps, all kernels handwritten", and `docs/ARCHITECTURE.md` as "attention,
RMSNorm, RoPE, SiLU, softmax are all handwritten. Only 5 external crates". Nothing in that list
computes a tensor.

The correlated decision is that **llama.cpp is a reference oracle, not a dependency**: it is used
for byte-identity comparisons (`minfer quantize` against `llama-quantize --pure`), for the
compute-graph study in `docs/LLAMA-COMPUTE-GRAPH.md`, and as a semantic reference for the KV cache
and sampling. It is never linked.

## Alternatives considered

**No rejected alternative is recorded for this one, and that is a fact about the record rather than
an omission.** The tree states the rule and never names a framework — `tract`, `ort`, `candle`,
`burn` — or a ggml binding as considered-and-rejected. The choice is the project's premise, not the
winner of a comparison. The two alternatives a reader would expect, and why each is inconsistent
with that premise:

- **Depend on an ML framework** (a tensor library such as `candle`, `burn`, `tch`/`ort`). The
  inference path — graph, scheduler, allocator, kernels, KV arena — *is* this project's subject; a
  framework would hide exactly the layer the repository exists to expose, and the axes it measures
  (backend assignment, liveness, quantization layout) are the framework's business rather than
  this code's.
- **Link ggml / llama.cpp directly.** Rejected in substance by the same premise, and it would
  remove the ability to use llama.cpp as an *independent* oracle: a byte-identity check against
  code you have linked is not a check.

Both are inferences from the stated premise; neither appears in the tree as a considered option,
and this ADR does not claim otherwise.

## Consequences

- Every kernel is hand-written, and every architecture is hand-optimized: the SIMD work is real
  maintenance (the aarch64 path uses inline asm; `MINFER_NO_NEON` / `MINFER_NO_AVX2` /
  `MINFER_NO_AVX512` force the lower path so the fallbacks stay exercised).
- The file formats are this repository's contracts and must be maintained as such: a quantized
  block is `repr(C)` and pinned to `ggml-common.h`, so a layout change is a format change.
- Correctness rests on external oracles rather than a framework's own tests: llama.cpp's output,
  GGUF fixtures recorded with provenance (`docs/f6-fixtures.json`, `scripts/check_f6_fixtures.py`),
  and each backend compared against its own reference.
- Cost accepted: no framework means no framework's kernels, and a supported model family costs a
  hand-written graph (`models/qwen2/`, `models/qwen3/`) rather than a config file. One piece of that
  cost is still open at the time of writing: CPU K-quant **weight repacking** (the F1 increment) is
  unimplemented.

## References

- `README.md` (the premise, the dependency statement) and `docs/ARCHITECTURE.md`.
- `docs/CPU_OPTIMIZATIONS.md` — what hand-written SIMD costs and buys.
- `docs/GGUF-TOOLING.md` — the format contract, with the `llama-quantize` byte-identity bar.
- `docs/LLAMA-COMPUTE-GRAPH.md` — llama.cpp used as a reference, not a dependency.
- Commit `95fd5d1` (2026-06-24, "feat: initial commit of Rust LLM inference engine").
