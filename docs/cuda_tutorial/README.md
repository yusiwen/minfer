# CUDA Tutorial for minfer contributors — from zero to optimization

This tutorial takes a minfer contributor who has **never written CUDA** to the
point where they can read every kernel in `src/cuda_kernels.cu`, modify the
CUDA backend safely, and follow the optimization records in
`docs/cuda_optimization_steps/` — including reproducing a historical step's
measurement. It teaches by **reading minfer's real code**: each kernel chapter
walks an actual `__global__` function from this repository, line by line, with
the CPU counterpart from the
[inference walkthrough](../inference_e2e_walkthrough/README.md) as the mirror.

> **Who this is for.** You can read Rust (minfer is pure Rust), you have read
> walkthrough chapters [09](../inference_e2e_walkthrough/09-prefill-forward-path.md)–
> [13](../inference_e2e_walkthrough/13-decode-loop-graph-reuse.md) — especially
> [10 · CPU matmul kernels](../inference_e2e_walkthrough/10-cpu-matmul-kernels.md)
> and [11 · attention & KV](../inference_e2e_walkthrough/11-attention-vecops-kv.md) —
> and you have never written a CUDA kernel. Every CUDA concept is defined where
> it first appears.

> **The machine.** Everything in this tutorial was written and verified on the
> target platform itself: an **NVIDIA GB10 (DGX Spark, aarch64)** with
> **CUDA 13.0** (`/usr/local/cuda/bin/nvcc`). Toy programs were compiled and
> run for real; their outputs are quoted as observed. To build minfer with the
> CUDA backend: `cargo build --release --features cuda` (details and pitfalls:
> [`docs/BUILD.md`](../BUILD.md)).

## What you will be able to do afterwards

- Predict what a kernel launch does — threads, blocks, memory traffic — before running it.
- Read any kernel in `src/cuda_kernels.cu` and any dispatch site in `src/graph/cuda_backend.rs`.
- Explain why minfer's decode path is GEMV + CUDA Graph replay while prefill is int8 MMQ GEMM.
- Profile with `ncu`/`nsys`, read the numbers, and **prove** an optimization instead of trusting it (the campaign's gate chain).

## The ladder — read in order

| Chapter | Part | What it adds |
|---|---|---|
| [01 · What kind of machine is a GPU](01-gpu-mental-model.md) | 1 — mental model | Host/device, SIMT (thread/warp/block/grid), memory hierarchy, why LLM inference fits — plus your first compiled-and-run kernel (Toy #1) |
| [02 · The minimal CUDA you actually need](02-minimal-cuda.md) | 2 — language surface | Kernel syntax & indexing, error checking & sync semantics, device memory & minfer's Rust/FFI wrapper, streams, the build system (Toys #2–#3) |
| [03 · Reading minfer's kernels I](03-kernels-elementwise.md) | 3a — first real kernels | Elementwise (`add_f32`), quantized-weight dequant (`dequant_q4_0_f16`), embedding gather (`embed_rows_q4_0`) |
| [04 · Reading minfer's kernels II](04-kernels-matmul.md) | 3b — the matmul ladder | Scalar GEMV → vectorized GEMV → tiled GEMM (`gemm_f16_nt_kernel_t`) → int8 MMQ pointer |
| [05 · Reading minfer's kernels III](05-kernels-attention-host.md) | 3c — attention + host | Flash-attention-style prefill (`fa_prefill_f16kv`), fused decode tail (`attn_bias_rope_store_f32`), KV in device memory, the Rust `Backend` layer, CUDA Graph capture/replay |
| [06 · Optimization methods](06-optimization-methods.md) | 4 — from reading to changing | Profiling with ncu/nsys, the technique catalog (each anchored to the step that used it), the verification gate chain, three exercises |
| [07 · Where to go next](07-where-next.md) | 5 — the map | Reading order for the reference docs, nvcc/ncu cheat sheet, pitfall list, toy index |

```mermaid
flowchart LR
    A["01 mental model<br/>(Toy #1)"] --> B["02 minimal CUDA<br/>(Toys #2–#3)"]
    B --> C["03 kernels I<br/>elementwise · dequant · embed"]
    C --> D["04 kernels II<br/>GEMV → tiled GEMM → MMQ"]
    D --> E["05 kernels III<br/>attention · host side · graphs"]
    E --> F["06 optimization<br/>profile · change · verify"]
    F --> G["07 the map<br/>reference docs · appendix"]
```

## How this relates to the other CUDA docs (no duplication, only links)

| Document | Role | This tutorial's policy |
|---|---|---|
| [`docs/CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) | technique reference — every term and technique, explained at depth | one-line mention + link; never re-explained here |
| [`docs/CUDA-BACKEND-DESIGN.md`](../CUDA-BACKEND-DESIGN.md) | backend design & implementation phases | linked for design history; this tutorial teaches the code as it stands |
| [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) + [`docs/cuda_optimization_steps/`](../cuda_optimization_steps/77-verification-methodology.md) | the optimization campaign: live status + 70+ step records | chapter 06 anchors each technique to the step that used it |
| [walkthrough 14 · Metal](../inference_e2e_walkthrough/14-metal-backend.md) / [15 · CUDA](../inference_e2e_walkthrough/15-cuda-backend.md) | backend chapters of the inference series | chapter 15 is the "what" of the backend — this series is the "how CUDA works" underneath it |
| [`docs/GPU_SAFETY.md`](../GPU_SAFETY.md) | hard safety rules for GPU code | cited where the rules come from; read it before touching backends |
| [`docs/GLOSSARY.md`](../GLOSSARY.md) | campaign glossary | fallback for any term this series defines too briefly |

## Conventions

- Prose in English; code, env vars and paths as-is. Writing contract:
  [STYLE.md](STYLE.md) (same voice rules as the inference walkthrough).
- Line numbers were verified against the tree at the time each chapter was
  written; the function/kernel name is the stable address, the line number a
  convenience. Toy outputs are real runs on the GB10 with CUDA 13.0.
- Toys are standalone `.cu` files embedded in the chapters; compile commands
  are given verbatim and none of them are part of minfer's build.

Start reading: [01 · What kind of machine is a GPU](01-gpu-mental-model.md).
