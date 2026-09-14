# 07 · Where to go next — the map, the cheat sheet, the pitfalls

> **Part**: Part 5 — leaving the ladder. **Prereq**: chapters 01–06.
> **Code**: none — pointers and reference tables.

## 1. You are here

You can now read every kernel in `src/cuda_kernels.cu`, you know how minfer's
decode (GEMV + fused tail + CUDA Graph replay) and prefill (MMQ GEMM +
`fa_prefill_f16kv`) paths are assembled, and you know the verification
discipline that keeps kernel changes honest. This chapter hands you to the
deep references — the tutorial deliberately stopped where they begin.

## 2. The reading map

| Order | Document | What it gives you now | Read how |
|---|---|---|---|
| 1 | [`docs/CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) | every technique the campaign uses, at reference depth (platform §1, software stack §2, memory §5, kernel inventory §6, sync §7, graphs §8, numerics §9, profiling §10, env gates §11) | full read once; it will feel like revisiting old friends — you met every section's topic in chapters 01–06 |
| 2 | [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) | the campaign's live status: what landed, what was rejected, current best numbers | read the current status block; keep it open while browsing steps |
| 3 | [`docs/cuda_optimization_steps/`](../cuda_optimization_steps/77-verification-methodology.md) | 70+ step records — each is a worked example of chapter 06's method: hypothesis → change → gate chain → verdict | start with [77 · verification methodology](../cuda_optimization_steps/77-verification-methodology.md), then follow the era indexes (Part I foundations → Part IV Era D); skim failures — they teach the defect classes |
| 4 | [`docs/CUDA-BACKEND-DESIGN.md`](../CUDA-BACKEND-DESIGN.md) | why the backend is shaped the way it is (phases, risks, llama.cpp reference map) | one pass; you now recognize every phase's code |
| 5 | [`docs/GLOSSARY.md`](../GLOSSARY.md) | every term/formula, classified | reference — jump in when a term resurfaces |
| 6 | walkthrough [14 · Metal](../inference_e2e_walkthrough/14-metal-backend.md) | the same concepts on a different backend API | optional; a good transfer test: map threadgroup↔block, MSL↔CUDA C yourself |

## 3. Appendix A — command cheat sheet

```bash
# Toolchain (GB10, CUDA 13.0; nvcc is not on every shell's PATH)
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version                       # 13.0
nvidia-smi                           # NVIDIA GB10, driver 580.x

# Compile a toy from any chapter (example: Toy #1)
nvcc toy1_vec_add.cu -o toy1 && ./toy1

# Build minfer with the CUDA backend (see docs/BUILD.md for details/pitfalls)
cargo build --release --features cuda          # + CUDA backend
cargo build --release --features cuda,cuda_static   # static cudart

# Run + observe
./target/release/minfer model.gguf "hello"     # graph path on GPU
MINFER_NO_CUDA_GRAPH=1 ./target/release/minfer model.gguf "hello"   # A/B: per-kernel launches
MINFER_NO_FUSE_QKV=1 MINFER_NO_FUSE_FFN=1 ./target/release/minfer model.gguf "hello"  # A/B: unfused
MINFER_TRACE=/tmp/t.json ./target/release/minfer model.gguf "hello"  # per-node trace (viz/README.md)

# Profile a kernel (see ch 06 §1 for how to read the output)
ncu --set basic ./target/release/minfer model.gguf "hi"
nsys profile ./target/release/minfer model.gguf "hi"
```

## 4. Appendix B — pitfalls the campaign already paid for

Each of these is a real defect class from minfer's history or the CUDA
programming model; the walkthrough/GPU_SAFETY links go to the full stories.

- **Index out of bounds at the tail** — every `i = blockIdx.x*blockDim.x +
  threadIdx.x` kernel needs the `if (i < n)` guard; ceil-div grids overshoot.
- **`__syncthreads()` inside a divergent branch** — threads that skip the
  branch never arrive; the barrier hangs or corrupts. Barriers live on
  straight-line block-wide control flow only.
- **Forgetting the error check** — kernel failures are asynchronous; without
  `cudaGetLastError()` right after launch (and a sync before readback) a
  broken kernel silently produces garbage. This is why `docs/GPU_SAFETY.md`
  mandates checked, bounded submission.
- **Host-copying a GPU-pending buffer** — reading device memory the GPU is
  still writing gives stale/corrupt data; minfer's rule: never host-copy a
  GPU-pending buffer (the Phase-3 KV-corruption bug; `docs/GPU_SAFETY.md`).
- **Unbounded sync waits** — "wait forever" turns a wedged kernel into a wedged
  process; minfer's `synchronize` waits bounded and checks status.
- **Silent CPU fallback** — a guard failure must `Err`, never quietly run the
  op on CPU: benchmark numbers from a fallback are phantom gains
  (`docs/GPU_SAFETY.md`; backend assignment is a build-time decision).
- **Trusting an unisolated speedup** — thermal/tenancy drift and one-off runs
  invent gains; the gate chain (ch 06 §3) exists because "it felt faster" is
  not evidence.

## 5. Appendix C — toy index

| Toy | Chapter | Teaches | Verified with |
|---|---|---|---|
| #1 `toy1_vec_add.cu` | [01](01-gpu-mental-model.md) | first kernel: launch, indexing, error checks, timing | nvcc 13.0 / GB10 |
| #2 SiLU elementwise | [02](02-minimal-cuda.md) | elementwise kernel + CPU reference compare | nvcc 13.0 / GB10 |
| #3 stream timing | [02](02-minimal-cuda.md) | default vs named stream, cudaEvent timing | nvcc 13.0 / GB10 |

## 6. Cross-references

- [06 · Optimization methods](06-optimization-methods.md) — the chapter before this one: profiling and the gate chain.
- [README](./README.md) — the ladder overview and the doc-relationship table.
- [`docs/CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) — your next full read.

← [06 · Optimization methods](06-optimization-methods.md) · Index
