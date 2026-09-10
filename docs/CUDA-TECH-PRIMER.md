# CUDA & GPU Technology Primer — every technique minfer uses, explained

This is the reference for the CUDA and NVIDIA-GPU technologies the minfer
inference engine actually uses — what each one is, how minfer uses it, where
it lives in the code, and what the optimization campaign (step docs
`cuda_optimization_steps/01–80`) learned about it. It is a teaching document
first: every concept is explained before it is used. Deep-dive narratives stay
in the step docs; this file is the map.

## 0. How to read this document — the three layers (and which layer each term lives in)

The campaign's vocabulary mixes three different technical domains. When a
term confuses you, first ask which layer it belongs to:

| Layer | Domain | Terms that live here | Decided by |
|---|---|---|---|
| **Algorithm** | LLM inference algorithms (the speculative-decoding literature: Leviathan/Chen 2023; llama.cpp's `draft-simple`) | `d` (draft length, e.g. "d=2"), acceptance rate `p`, `E[a] = Σp^i`, break-even | probability & decision math — hardware-independent |
| **Performance model** | computer-architecture performance analysis (the roofline model, arithmetic intensity) | "amortization" (spreading fixed costs over more work), memory-bound vs compute-bound, bytes-per-token | measured bytes/FLOPs against hardware ceilings |
| **Micro-architecture (kernel implementation)** | GPU GEMM kernel engineering (tiling, CUTLASS-style vocabulary) | tile shape, "tile-regime", wave quantization, occupancy, `__launch_bounds__`, coalescing | the kernel's tiling configuration vs the shape it is launched with |

A concrete chain from the D5-0 gate (doc 80) showing all three at once:

```
d=2 (algorithm layer: draft 2 tokens per round)
  → the verify forward is a batched nt=3 decode step (a SHAPE)
    → which tile-regime does M=3 land in? (micro-architecture layer)
      → 2.28x (interpolation) or 2.7x (same-tile-as-M=4)? = the amortization
        (performance-model layer: fixed weight-traffic cost spread over rows)
        → ≥ 2.5x needed for break-even at measured p≈0.68
          → back up: is the algorithm parameter d=2 worth it?
```

No single layer could answer the gate question — the algorithm parameter's
fate was decided by a micro-architectural regime measurement. That
cross-layer interaction is exactly what the campaign's step docs record, and
what this primer maps.

> **Full taxonomy:** a corpus-wide audit extended these three layers to
> **seven** (adding Platform & tooling, Engine architecture, Methodology) and
> classified every term and formula in the campaign docs — see
> [GLOSSARY.md](./GLOSSARY.md).

Code surfaces referenced throughout:

| File | Lines | Role |
|---|---|---|
| `src/cuda.rs` | ~5.4k | Device layer: hand-written CUDA bindings, context/state, weight upload & registration, buffer concat, KV type management |
| `src/cuda_kernels.cu` | ~7.3k | Every CUDA kernel (~65 `__global__` functions) + their `launch_*` C-shim wrappers, compiled by nvcc into `libcuda_kernels.a` |
| `src/graph/cuda_backend.rs` | ~6.1k | The `Backend` trait implementation: buffer pools, per-node dispatch, CUDA Graph capture/replay, synchronization |
| `build.rs` | — | nvcc orchestration: compiler discovery, host-compiler pinning, arch detection, SASS/PTX emission, cudart linking |

---

## 1. The platform — what "GB10 / DGX Spark" means for this code

minfer's CUDA campaign ran on an **NVIDIA GB10** (DGX Spark): a Grace-Blackwell
superchip where a 20-core Arm CPU (the same machine runs minfer's NEON+SDOT
CPU backend) and a Blackwell-class GPU share one coherent memory pool
(~128 GB). Practical consequences that show up everywhere in the record:

- **The GPU reports a Blackwell-class compute capability** — the build covers
  `sm_70 … sm_121` (see §3); the campaign machine (driver 580.173.02) uses one
  of the newest targets in that list.
- **Unified memory does NOT mean minfer should use `cudaMallocManaged`** — it
  doesn't. All device memory is plain `cudaMalloc` pools (§5); "resident
  weights" (§6.1) is the deliberate design: upload once, keep on the GPU,
  never page across the link.
- **CPU and GPU measurements interleave on one machine** — which is why every
  campaign number is a same-window interleaved A/B median (co-tenant load once
  moved a kernel's measurement by 2.7×; step doc 15).

## 2. The software stack — how Rust reaches the GPU

minfer uses **no CUDA Rust ecosystem crates** (no bindgen, no cuda-rust, no
cudarc). The stack is four layers, each thin:

```
src/graph/scheduler.rs          Rust: picks the backend per node, splits the graph
src/graph/cuda_backend.rs       Rust: buffer pools, dispatch, capture/replay, sync
        │  extern "C" — hand-declared symbols (cuda.rs + backend.rs)
        ▼
libcuda_kernels.a               nvcc-compiled from src/cuda_kernels.cu:
  launch_q6_k_q8_mmvq(...)        every kernel + a C `launch_*` shim per kernel
        │                         (the shim contains the <<<grid,block>>> syntax,
        ▼                          which is CUDA-C++ only)
libcudart.so / libcudart_static.a   the CUDA runtime API (cudaMalloc, cudaGraph…)
        ▼
libcuda.so.1                    the kernel-mode driver (preloaded via dlopen,
                                 see §2.1)
```

**Why a C shim for launches**: Rust has no `<<<>>>` launch syntax; rather than
generate it, each kernel gets a ~20-line C wrapper (`launch_<name>(...)`) that
takes plain pointers and ints, computes the launch configuration, and calls
the kernel. Rust declares these as `extern "C"` blocks (cuda.rs) — the entire
binding layer is hand-written and auditable (114 `launch_*` declarations).

**Why dlopen libcuda first** (cuda.rs): `libcudart` itself `dlopen`s the
driver library `libcuda.so.1`. Preloading it from the well-known locations
(`RTLD_NOW | RTLD_GLOBAL`) guarantees cudart reuses the already-loaded driver
instead of failing on a stub-dependency resolution — a robustness detail that
matters on non-standard installs.

**Static vs shared runtime**: by default minfer links the shared
`libcudart.so`; the `cuda_static` cargo feature links `libcudart_static.a`
instead (plus pthread/dl) so the binary carries no cudart dependency — at the
cost of a larger binary. `build.rs` wires both link modes.

## 3. nvcc and the build system (build.rs)

`nvcc` is NVIDIA's compiler driver: it compiles `.cu` files by splitting them
into host C++ (compiled by a host compiler) and device code (compiled by
NVVM/PTXAS into GPU machine code). minfer's build must orchestrate it because
plain `cargo build` **never touches nvcc** — the CUDA compile only runs when
the `cuda` feature is requested (an early footgun: any build with an nvcc on
PATH used to attempt the compile; build.rs now gates strictly on the feature).

Key mechanics, all in `build.rs`:

- **Compiler discovery**: `find_nvcc()` / `find_cuda_home()` locate the
  toolkit; `CUDA_HOME` style overrides apply.
- **Host-compiler pinning (`-ccbin`)**: nvcc inherits the first `cc`/`g++` on
  PATH as its host compiler and hard-fails when that compiler is newer than
  the toolkit supports. build.rs only pins `-ccbin` when the default is
  actually rejected — otherwise the environment's choice is left untouched
  (documented in `docs/BUILD.md`).
- **Architecture detection (`detect_archs`)**: GPU machine code is
  architecture-specific. build.rs probes nvcc with a dummy `.cu` compiled for
  each candidate `sm_70, 72, 75, 80, 86, 89, 90, 100, 103, 110, 120, 121` and
  keeps the ones the toolkit accepts.
- **SASS + backward-JIT PTX**: for each detected arch it emits
  `arch=compute_<a>,code=sm_<a>` — i.e. **SASS** (the arch-specific machine
  code, fastest launch, no JIT) — plus a `compute_70/72` **PTX** (NVIDIA's
  virtual-ISA assembly) so older GPUs JIT-forward to a working image even
  without exact SASS.
- **Output**: one static library `libcuda_kernels.a` per build, exposing the
  `launch_*` shims; the Rust linker is pointed at it and at the cudart
  link path.

Terminology used in the campaign: **SASS** = the actual per-architecture
instruction stream (read with `cuobjdump -sass` / `nvdisasm`; step doc 30's
"opcode census" counted LDG/IMAD/HMMA instructions there). **PTX** = the
virtual assembly JIT'd at load time. **sm_XX / compute_XX** = the
architecture generations.

## 4. The CUDA programming model — the vocabulary every kernel section uses

> Every term in this section also appears, classified by layer, in the
> corpus-wide [GLOSSARY.md](./GLOSSARY.md) (L4/L5).

- **Host / device**: the CPU side (Rust) and the GPU side (.cu). They have
  separate address spaces; all data crossing the boundary goes through
  explicit copies (`cudaMemcpy`) or pinned-memory staging (§5.4).
- **Kernel (`__global__`)**: a function executed by many GPU threads at once.
  minfer has ~65 of them (the inventory in §6).
- **Grid / block / warp / thread**: a launch specifies a **grid** of **blocks**
  (each block scheduled onto one SM — streaming multiprocessor, the GPU's
  compute unit); a block contains up to 1024 **threads**; threads execute in
  32-thread **warps** (the SIMD unit — every thread in a warp executes the
  same instruction). minfer's kernels mostly use 256-thread blocks
  (`__launch_bounds__(256)`, §4.1) or 128 (rms pad40).
- **`__launch_bounds__(max_threads, min_blocks)`**: a compiler hint capping
  register usage so a block/thread combination is guaranteed launchable —
  D4-2 tested `__launch_bounds__(256,6)` explicitly (48→40 regs, but +40 B
  stack spill → measured −0.75% → reverted). Lesson: launch bounds trade
  registers for spill; measure, don't assume.
- **Streams**: an ordered queue of GPU work. minfer creates one stream
  (`cudaStreamCreate`) per backend; all launches and copies are stream-ordered
  so `synchronize()` semantics stay simple (§7).
- **Occupancy & waves**: an SM can host a limited number of concurrent blocks
  (bounded by registers/shared memory/threads). **Occupancy** = how much of
  that capacity is used. The GPU runs blocks in **waves** — total blocks ÷
  (blocks resident per SM × SM count). Wave quantization is a recurring
  campaign villain: a grid of 1.14 waves runs as 2 waves (the second nearly
  empty) — e.g. D4-4's fused-FFN probe grid of 1.5 waves explained its +28.2%
  loss, and M=1 decode GEMMs at **0.14 waves** are the reason batched decode
  (spec-decode's verify, doc 80) amortizes so well.
- **Occupancy ≠ performance**: r24's scheduling-structure ladder showed grid
  rearrangements that "improved" occupancy yet measured flat — the +1.5%
  whole-prefill landing bar was calibrated in that round.

## 5. Memory — hierarchy, movement, and the rules the campaign burned in

### 5.1 The hierarchy

| Level | Scope | Latency class | minfer use |
|---|---|---|---|
| Registers | per-thread | ~0 cycles | accumulator tiles (`float acc[8]`), the ≤85-regs ceiling that gated r59's wave re-tile |
| Shared memory | per-block, user-managed | ~30 cycles | staging tiles for wmma GEMM (8m), swizzle space (r22), smem over-cap silently killed r8's first wide tile (attr-set + launch both failed — "suspiciously fast" readings must be checked against resource caps) |
| L1 / L2 cache | per-SM / chip | ~200 / ~400 cycles | weight L2 residency tested in r19 (see below) |
| DRAM (unified pool) | device | ~600+ cycles | weights, KV, activations; **byte bandwidth through this level is the decode wall** (D1–D3 attribution) |

### 5.2 Coalescing and vectorized access

A warp's 32 threads should touch **contiguous memory** so the hardware folds
their accesses into the minimum number of 128-byte transactions
(**coalescing**). minfer's kernels load weights as `uint4` (16-byte vector
loads, the widest LDG) along unit-aligned rows; the dpl repack (D4-4) exists
precisely to make this true end-to-end — the stock Q6_K row layout wasted 14
of every 224 bytes on padding (84% useful), and the dense split-plane repack
(`[ql][qh][sc][d]` planes, 210 B content, 16 B-aligned stride) turned padding
bytes into bandwidth: −17.1% content traffic on ffn_down, +5.5% 14B tg128.

### 5.3 Cache-control intrinsics (and their negative results)

- **`__ldg`** (read-only, non-coherent load path): r19 tested marking the
  entire weight stream `__ldg` — imperceptible; then a **persisting L2
  window** (`cudaAccessPolicyWindow` pinning weights in L2) — catastrophic.
  Lesson recorded in doc 24: L2 residency management is the wrong lever when
  the working set is streamed once per token anyway.
- **`cp.async`** (asynchronous shared-memory copy, Ampere+): the 8m prefill
  GEMM stages weight/activation tiles into shared memory with cp.async so the
  copy pipeline overlaps the mma pipeline; the r21 "coalesced block-linear A
  staging" variant (removing the XOR swizzle in favor of linear layout)
  measured as a stall-mass wash → reverted (the swizzle stays).

### 5.4 Pinned host memory (page-locked staging)

Device↔host copies through pageable memory bounce via an internal driver
staging buffer and serialize. minfer instead allocates **pinned** host buffers
(`cudaHostAlloc`) — one grow-on-demand staging buffer (R3-A2) capped at 128 MB
per split — and copies results into it device→host directly. `MINFER_NO_PINNED_READBACK=1` reverts to plain `cudaMemcpy`. The capture-readback path
(§8) reads logits from this pinned buffer immediately after each captured
node's launch.

### 5.5 Buffer pools (no cudaMalloc in the hot path)

`cudaMalloc` is expensive (driver round-trip). `CudaBackend` owns **buffer
pools**: allocation hands out pool slots (bumped `pool_gen` tracks
generations), the graph allocator's liveness analysis (AGENTS §"Compute
Graph" rules) reuses slots, and input buffers are never freed mid-graph. All
pools are plain device memory — no unified/managed allocations anywhere.

## 6. The kernel inventory — every `__global__` family, what it does and why

### 6.1 Weight-resident model (Phase 7 / 8p)

At load, every weight tensor is uploaded once and **stays on the GPU** for the
process lifetime (`cuda.rs` registers them; execution gates on
all-weights-registered). The 8p round added a persistent **f16 weight cache**
(+ fused dequant-in-GEMM) for shapes where dequant-on-the-fly beats quantized
GEMM. This is the foundation of everything else: decode has zero host↔device
weight traffic.

### 6.2 The matmul families (the campaign's main battlefield)

| Kernel family | Compute primitive | Used for | Step docs |
|---|---|---|---|
| `q*_f32_matmul` (q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k) | CUDA cores, per-thread dot over dequantized weights | pre-Phase-7 / fallback paths, GPU reads f32 activations | — |
| `gemm_f16_nt_kernel_t` (+ `gemm_qb_nt`) | **Tensor cores via `nvcuda::wmma`** (f16 in, f32 accumulate) | prefill GEMM on the f16 weight cache (8m) — 30.7→1204 tok/s (39×) | 02, 05 |
| `mmq_nt_kernel`, `mmq_raw_nt/wide/nb/bt(_q6k)` | integer dp4a/imad over **int8-quantized activations** × quantized weights | prefill MMQ (R1 parity-first, opt-in → default), the r5–r59 line — 8.1× arc. **"BT"** names the raw-byte BT-style kernel variant of this family (introduced r38 as the q6_K BT-style kernel; `mmq_raw_nb_bt_kernel`) — the form whose batched (nt>1) executions the D series measured; "BT-MMQ 2.7× at nt=4" (doc 80's gate anchor) refers to this family running the multi-token batch | 08, 13–31, 33–40, 67, 76 |
| `q*_q8_mmvq`, `v2`, `v2_pf`, `_dpl` | **dp4a** (4-way int8 dot) per weight row | decode matrix-vector (one token): q4_K/q5_K/q6_K; v2 = D3b bitwise re-tile, pf = D4-2 padded-form dispatch, dpl = D4-4 dense split-plane | 06, 67, 74, 76 |
| `dequant_*_f16` | per-block dequant to f16 | feeds the f16 GEMM path | 05 |
| `embed_rows_*`, `gather_rows_f32` | row-gather | embedding lookup on GPU | — |

**The two activation regimes** (a core convention, AGENTS §Core Conventions):
CPU quantizes activations to Q8_0 on the fly (Q8_K for K-quant weights) and
runs integer math; GPU prefill MMQ also uses int8 activations, but the GPU's
f32/wmma paths read f32 activations directly. CPU-vs-GPU logits therefore
**differ by design** — each path is verified against its own reference
(§9).

**Quantized weight formats** (GGUF v3, `src/block.rs` = the `repr(C)`
ggml-common layout): super-block schemes where a group of 256 values carries
scales/mins at coarser granularity (Q4_K: 8 sub-blocks of 32 + a super-scale
per 256; Q6_K: 16-byte scales + 2-bit high bits packed with the 6-bit lows).
The GPU kernels unpack these layouts directly from the byte stream — no
pre-dequantization — which is why the dpl layout experiment (a different
**byte arrangement** of the same values) could be bitwise-safe: same values,
same accumulation order, only the memory map changed.

### 6.3 Attention kernels

- **`fa_prefill_f16kv`** — the 8n **FlashAttention-style tiled prefill
  attention**: online softmax (rescaling the running max/sum instead of
  materializing the score matrix), the O accumulator held in registers, 256 B
  P-stride to avoid a score-clobber race. 20× over the naive path.
- **`gqa_attn_f32`** (h4w = 4-warp variant) and **`gqa_attn_f32_f16kv`** —
  decode attention with **GQA** (grouped-query: N query heads share KV heads;
  the q-head batching experiment D3-6 lost to the 5× L2 re-read it created).
- **`gqa_attn_split_partial` + `gqa_attn_split_combine`** — **flash-decoding
  split-K** (8d, later R4's dim-parallel rewrite): split the KV sequence
  across blocks, each computes partial (num, den, max) triples, a combine
  kernel folds them. The D1 attribution found `split_partial` was 100% of the
  KV-growth wall (memory-LATENCY-bound, 76.5% long_scoreboard) — the number
  that authorized D2/D3.

### 6.4 Element-wise and fused epilogue kernels

`rms_norm_*` (with `_quant` fused output-quantize variants, `_nw` no-wide,
`_pad40`), `silu_f32`, `swiglu_*` (gate×up + SiLU), `rope_f32`, `add_bias_f32`,
`add_f32`, `mul_f32`, `store_kv_f32/f16`, `quantize_q8_0` (`_pad40`, `_t`
transposed), `f32_bits_to_i32` (positions as bits — AGENTS rule 4),
`convert_f32_f16`. The **fused epilogues** are the D3-5/D3-8 story:
`attn_bias_rope_store_f32` replaced a 7-launch chain (bias×3 + rope×2 +
store×2) with one launch (−310 launches/step), and producer-fused A-quantize
(`MINFER_MMQ_A_FUSE`) folds activation quantization into the producer kernels
(quantize launches −78%).

**Why fusion matters on this GPU**: decode chains are launch-overhead-bound
(2 µs/graph-gap scale); every kernel boundary costs a launch + memory
round-trip through L2. The fusion ledger lives in step docs 70, 73, 76.

## 7. Synchronization discipline (GPU Safety, `docs/GPU_SAFETY.md`)

The hard rules that bound every kernel addition:

- **`submit()` waits bounded + checks status** — never an unbounded block on
  the GPU; failures abort with actual values, never silently.
- **No early return past a `threadgroup_barrier()`** (Metal phrasing; the CUDA
  analogue is `__syncthreads()`) — divergent barrier exits deadlock or corrupt.
- **Kernel-invariant violations return `Err` from `execute_node`** — never a
  silent CPU fallback mid-graph; backend assignment is decided at build time
  (`supports_op`), and the graph records GPU participation in `CParams.gpu`.
- **Never host-copy a GPU-pending buffer** — the Phase-3 KV-corruption bug
  class; copies cross-backend happen only at scheduler split boundaries.
- `synchronize()` (backend.rs) is the one choke point: stream-ordered work is
  waited with a bounded loop and the status is checked.

## 8. CUDA Graphs — capture once, replay many (Phase 7d)

Decode = the same ~13-kernel chain re-launched hundreds of times; per-launch
CPU overhead (~2–7 µs) is pure tax. **CUDA Graphs** fix this: a sequence of
launches is recorded (**captured**) into a graph object, then **replayed**
with one call — driver-side launch overhead collapses.

minfer's machinery (`cuda_backend.rs`):

- **Capture eligibility**: a (graph uid, split node-range) is captured after a
  **3rd consecutive direct-launch warmup** (llama.cpp's warmup-twice heuristic
  adapted); captures are valid only for the `pool_gen` they were captured at —
  any pool re-allocation invalidates them (the buffers the graph's pointers
  reference must be the same memory).
- **Prefill capture** is default-on now (`MINFER_NO_PREFILL_CAPTURE=1` opts
  out; replay at real-prefill scale validated it).
- **`MINFER_NO_CUDA_GRAPH=1`** force-disables (the A/B control used by every
  graph-adjacent step doc).
- **Why replay is safe in minfer's design**: AGENTS rule 1 — "KV positions
  are data, not structure". The graph's kernel arguments (pointers, dims) are
  identical every step; only buffer *contents* (positions, tokens) change, and
  those are written into the same device buffers before replay. This is also
  exactly the property the spec-decode verify batch needs (doc 80: fixed d ⇒
  one captured verify graph, positions refilled host-side per round).
- **Validation**: `MINFER_GRAPH_DUMP` replays capture real-data dumps; the
  standard gate is replay-dump byte-identity vs eager launches.

## 9. Numerics — precision, accumulation, and the verification gates

- **f32 accumulation everywhere in tensor-core paths**: wmma accumulators are
  f32 (`mma_sync` with f32 D/C operands); r15's probe of f16 accumulation was
  a dead end (numerics + throughput both lost).
- **Tolerance gate vs bitwise**: the campaign's two-tier standard. New math
  gets a **tolerance gate** (CPU reference in `double`, 0.05 abs with
  adversarial outliers, per step doc 68's calibration) plus
  **bitwise-identity** tests whenever a rewrite claims "same math"
  (byte-identical dumps, greedy outputs byte-identical across seeds). The
  rounding classes are understood, not hand-waved: D4-2's B0 bug (7B dropped
  13.5% of down-proj rows) showed up first as max|Δ| 0.254 first-step logits
  at a documented rounding class.
- **The dump instruments** (`--features debug_dump`, `MINFER_DUMP_DIR`,
  `MINFER_GRAPH_DUMP`, `MINFER_TRACE`): per-node real-data dumps that make
  "bitwise vs eager" and "KV rows identical" checkable claims. Pool-slot
  recycling makes some informational dumps binary-layout-dependent — a
  documented instrument limitation, not a bug.

## 10. Profiling & forensics — how the campaign knows what it knows

- **nsys** (Nsight Systems): timeline traces → **launch counts** per decode
  step (the D3-8 −310 launches/step claim) and wall decomposition.
- **ncu** (Nsight Compute): per-kernel counters → the D1 verdict
  ("memory-LATENCY-bound, 76.5% long_scoreboard" = threads stalled waiting on
  DRAM loads) and the D4-3 llama-bench artifact bust (a decode kernel loading
  a constant 5.4 MB regardless of context = not real work).
- **SASS census** (step doc 30): counting actual machine instructions to
  explain instruction-mix changes.
- **Counter forensics** (doc 18): reconciling achieved bytes/FLOPs against
  hardware ceilings — a claim survives only if the counters, the SASS, and a
  reductio all agree.

## 11. The env-gate inventory (A/B controls)

Every lever is env-gated so the same binary runs both sides of an A/B
(never a rebuild between A and B — the campaign's method):

| Gate | Meaning |
|---|---|
| `MINFER_DISABLE_CUDA=1` | force CPU (backend selection off) |
| `MINFER_NO_CUDA_GRAPH=1` | disable capture/replay (eager launches) |
| `MINFER_NO_PREFILL_CAPTURE=1` | disable prefill-graph capture |
| `MINFER_NO_PINNED_READBACK=1` | revert to plain cudaMemcpy readback |
| `MINFER_NO_FUSE_QKV` / `MINFER_NO_FUSE_FFN` | revert the decode fusions |
| `MINFER_MMQ_A_FUSE` | producer-fused activation quantize |
| `MINFER_Q6K_DPL=0` | opt out of the dpl split-plane layout |
| `MINFER_Q6K_PF=0` | revert v2_pf padded-form dispatch |
| `MINFER_PDL` | (closed NO-GO, D4-4) programmatic dependent launch |
| `MINFER_NO_MPS=1` | Metal: force CPU (macOS paths) |
| `MINFER_NO_NEON=1` | CPU: force scalar |
| `MINFER_GRAPH_DUMP` / `MINFER_TRACE` / `MINFER_DUMP_DIR` | verification instruments (§9) |

**PDL** (programmatic dependent launch — letting the next kernel launch before
the previous finishes via `cudaLaunchAttributeProgrammaticStreamSerialization`
+ `cudaGridDependencySynchronize()`): the one CUDA feature the campaign
integrated fully and then reverted — in-situ A/B read −2.6%/−1.8% on 14B
because early-launch co-residency taxes compute-tail kernels more than the
graph-gap it recovers (doc 76). A standing warning against co-residency
tricks on this workload.

## 12. Where to go next

- Build specifics (toolkit install, ccbin, cudart modes): `docs/BUILD.md`
- GPU safety rules: `docs/GPU_SAFETY.md`
- The campaign's full history, one doc per step: `docs/cuda_optimization_steps/README.md`
- The compute graph these kernels serve: `docs/GRAPH-REFACTOR-PLAN.md`
- The spec-decode consumer of the nt>1 regime: `docs/SPECULATIVE-DECODING-PLAN.md`
