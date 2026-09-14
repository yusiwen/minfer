# CUDA tutorial for minfer contributors — writing contract (all authors must follow)

This directory teaches CUDA from zero to "can read and optimize minfer's CUDA
backend". It is a **taught progression**, not a reference: each chapter builds
on the previous ones and ends by handing the reader to the next. Intended
reader: an engineer who can read Rust, **has read `docs/inference_e2e_walkthrough/`
09–13** (especially 10 — CPU matmul kernels, 11 — attention/vec ops/KV), and
has **never written CUDA**.

## Relationship to the other CUDA docs (do not duplicate, link instead)

| Doc | Role | This tutorial's policy |
|---|---|---|
| `docs/CUDA-TECH-PRIMER.md` | technique reference (every term/technique) | one-line mention + link; never re-explain at reference depth |
| `docs/CUDA-BACKEND-DESIGN.md` | backend design + phase record | link for design history; this tutorial teaches the code as it is |
| `docs/CUDA_OPTIMIZATION.md` + `docs/cuda_optimization_steps/` | campaign history (70+ step records) | Part-4 techniques each link to the step(s) that used them |
| `docs/inference_e2e_walkthrough/` 09–15 | inference pipeline (incl. CUDA backend chapter) | assumed background for CPU counterparts; doc 15 is the "what" — this is the "how CUDA works" |
| `docs/GLOSSARY.md` | campaign glossary | deep-term fallback link |
| `docs/BUILD.md`, `docs/GPU_SAFETY.md`, `docs/SUPPORT-MATRIX.md` | build details / safety rules / quant matrix | linked, not copied |

## Audience and voice (same rules as the walkthrough)

- Patient mentor voice, plain English, short sentences, full paragraphs.
- **Define every CUDA term on first use** in one or two plain sentences: host,
  device, kernel, thread, warp, block, grid, SM, occupancy, shared memory,
  coalescing, stream, event, CUDA Graph, MMQ, tiling, divergent branch…
  Assume zero CUDA knowledge. Rust knowledge is assumed.
- Mechanism before jargon; arithmetic beats adjectives (show byte/element/op
  counts when claiming fast/cheap).
- Prose in English; code, function names, env vars, paths as-is.

## Fixed per-chapter structure (order must not change)

```markdown
# NN · <title>

> **Part**: <which part of the tutorial ladder>. **Prereq**: chapters assumed.
> **Code**: files this chapter reads (`src/...` — verified lines).

## 1. Background — where this sits
What the reader knows by now, what this chapter adds, why it comes here.

## 2. Principle — the concepts
Mechanisms first, from data shapes and arithmetic. Where minfer's real code is
too advanced to start from, teach a **toy example** here first (see Toy rules).

## 3. In minfer's code
Real excerpts (10–40 lines, current tree, `file:line` verified) read line by
line. For kernel chapters: kernel source → line-by-line → CPU counterpart
(walkthrough doc 10/11) → why the GPU version looks the way it does.

## 4. Performance intuition
Cost model in numbers: bytes moved, threads launched, occupancy math, what to
expect and what would make it slow.

## 5. Try it / Observe
Compile-and-run commands for toys; env vars and tools to see it live
(`MINFER_NO_CUDA_GRAPH=1`, ncu/nsys one-liners…).

## 6. Cross-references
Neighboring chapters + linked reference docs, one line each.

← NN-1 · Index · NN+1 →
```

## Toy example rules

- A toy is a **complete, compilable** `.cu` file in a fenced block, ≤80 lines,
  followed by the exact shell command to compile and run it (`nvcc toy.cu -o
  toy && ./toy`) and the **actual observed output** (the authoring machine is
  the GB10 itself — compile and run every toy before landing it).
- Toolchain: nvcc 13.0 at `/usr/local/cuda/bin/nvcc` (NOT on every shell's
  PATH — use the full path or `export PATH=/usr/local/cuda/bin:$PATH`);
  `ncu`/`nsys` for profiling; GPU: NVIDIA GB10 (aarch64, driver 580.x). Mark
  the CUDA version the toy was verified with.
- Toy code style: no error-checking shortcuts that hide the failure modes
  chapter 2.2 teaches; show the checks.
- Keep the total at three (Toy #1 vector add, Toy #2 elementwise/SiLU,
  Toy #3 streams); they are indexed in `07-where-next.md`'s appendix.

## Hard rules

0. **Code-extraction forensics protocol**: enumerate `src/cuda_kernels.cu`
   with `grep '__global__'` first, locate the target kernel, **read a bounded
   line range**, excerpt 10–40 lines, and verify every `file:line` you cite by
   actually reading it. Line numbers move; the function name is the stable
   address, the line number is a convenience — re-verify at writing time.
1. **Write only your assigned `docs/cuda_tutorial/NN-*.md`.** Do not touch
   source files, other docs, `SUMMARY.md`, or `book.toml`. Do not commit (the
   coordinator commits).
2. **Write the doc to disk as soon as its forensics is done** — do not stage
   the whole chapter in memory first.
3. Facts and numbers come from the code or existing docs; measured numbers
   cite their doc (e.g. a step record). Do not invent numbers.
4. File naming: `NN-english-dash-slug.md`; NN is assigned by the task.
5. **Length: content-first, target 500–900 lines** unless the brief says
   otherwise; every paragraph must teach something.
6. Beginner definitions are mandatory, not decorative: a reader who has never
   written a kernel must finish `01` able to predict what a kernel launch
   does; must finish `04` able to explain why decode is memory-bound.
7. End every doc with the nav line; the index link is `./README.md`.
8. At most one small diagram (ascii or mermaid) per chapter where it clearly
   helps; mermaid must render under mdbook-mermaid (plain ` ```mermaid `).

## Series facts every author can rely on

- Kernels live in **`src/cuda_kernels.cu`** (compiled by `build.rs`, ccbin
  pinned; see `docs/BUILD.md`). Host/device layer: **`src/cuda.rs`** (device
  management, FFI bindings, RAII device buffers). Backend: **`src/graph/cuda_backend.rs`**
  (`Backend` trait impl, per-node dispatch, buffer pool, CUDA Graph capture/replay).
- Decode path: GEMV-style matvec + fused QKV/attention kernel + **CUDA Graph
  replay** (`MINFER_NO_CUDA_GRAPH=1` reverts to per-kernel launches).
- Prefill path: **int8 MMQ GEMM** (weights stay quantized, activations
  quantized on the fly) + `fa_prefill_f16kv` flash-attention-style kernel.
- GPU activations are f32 (prefill MMQ internally int8); quantized weights are
  **dequantized to f16 on device** at load (`dequant_*_f16` kernels); KV is f16.
- Supported quants: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K (no
  Q2_K/Q3_K/I-quants) — `docs/SUPPORT-MATRIX.md`.
- Hard safety rules (`docs/GPU_SAFETY.md`): `submit()` waits bounded and
  checks status; device limits queried at runtime; kernel-invariant violations
  return `Err` from `execute_node` — never a silent CPU fallback.
- Verified kernel inventory includes (name → purpose):
  `add_f32`/`add_bias_f32` (elementwise), `dequant_q*_f16` (weight load),
  `embed_rows_q*` (embedding gather), `f32_f32_matmul_scalar`/`f32_f32_matmul_vec`
  (decode matvec), `gemm_f16_nt_kernel_t` (prefill GEMM), `fa_prefill_f16kv`
  (prefill attention), `attn_bias_rope_store_f32` (fused decode attention tail),
  `convert_f32_f16_kernel`, `f32_bits_to_i32`, `gather_rows_f32` — re-enumerate
  with grep before citing; more exist (rope/rmsnorm/silu/MMQ families).
