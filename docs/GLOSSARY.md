# Glossary — every term and formula in the CUDA campaign docs, by layer

This is the consolidated vocabulary of the CUDA campaign corpus
(`CUDA_OPTIMIZATION.md`, `CUDA-TECH-PRIMER.md`, the 80 step docs in
`cuda_optimization_steps/`, the four analysis/plan docs, and the build/safety
references). Every entry is corpus-verified: it appears in at least one of
those documents.

## 0. The layer taxonomy

The primer's §0 originally introduced **three layers** (Algorithm /
Performance model / Micro-architecture) to explain the D5-0 gate chain. The
full-corpus audit showed that roughly half of the vocabulary does not fit any
of those three, so the taxonomy is extended to **seven layers**:

| Layer | Domain | Typical questions it answers |
|---|---|---|
| **L1 Algorithm** | LLM inference algorithms & their math | What does the model compute? What is the optimal `d`? |
| **L2 Numerics & data formats** | quant formats, byte layouts, rounding, tolerances | How are numbers represented and how wrong may they be? |
| **L3 Performance model** | roofline, bandwidth, amortization, break-even | How fast *should* it go, and why isn't it? |
| **L4 Micro-architecture** | tiling, tensor cores, SASS, shared memory, scheduling | How does the kernel actually use the machine? |
| **L5 Platform & tooling** | CUDA API/runtime, nvcc/PTX/SASS toolchain, profilers, env gates | What do we drive the hardware with, and how do we look inside? |
| **L6 Engine architecture** | minfer's compute graph, backends, allocator, fusion, safety rules | How is the engine itself structured? |
| **L7 Methodology** | measurement discipline, parity gates, forensics, doc conventions | How do we know a number is real? |

The D5-0 gate chain still illustrates why layers matter — the same terms now
map to L1/L4/L3 explicitly:

```
d=2 (L1) → verify = batched nt=3 decode step (shape) → which tile-regime (L4)
  → amortization ≥ 2.5x? (L3) → break-even at p≈0.68 (L3) → go/no-go on d=2 (L1)
```

### Source-doc tags used below

| Tag | Document |
|---|---|
| PRIMER | `CUDA-TECH-PRIMER.md` |
| HUB | `CUDA_OPTIMIZATION.md` (campaign hub + live status) |
| STEPS | `cuda_optimization_steps/NN-*.md` (numbered step doc) |
| MMQ | `LLAMA-CPP-MMQ-ANALYSIS.md` |
| SPEC | `LLAMA-CPP-SPECULATIVE-ANALYSIS.md` |
| D5 | `SPECULATIVE-DECODING-PLAN.md` |
| GRAPH | `GRAPH-REFACTOR-PLAN.md` |
| BACKEND | `CUDA-BACKEND-PLAN.md` |
| BUILD | `BUILD.md` |
| SAFETY | `GPU_SAFETY.md` |
| METAL | `METAL_OPTIMIZATIONS.md` (its §5.6 is a doc-local symbol glossary) |

---

## L1 — Algorithm: LLM inference algorithms & math

| Term / Formula | One-line meaning | Source |
|---|---|---|
| prefill | One forward pass over the whole prompt, batched, compute-bound. | PRIMER, STEPS 20+ |
| decode | One token per step (nt=1), memory-bound: streams all weights every token. | PRIMER, STEPS |
| logits | The `[nt][vocab]` output of the final matmul; input to sampling. | GRAPH, STEPS |
| greedy decoding | Always take argmax of logits — deterministic, used for parity gates. | HUB, STEPS |
| top-k / top-p / temperature / repeat-penalty | The sampler chain in `sampler.rs`, defaults matching llama.cpp (0.8/0.95/1.1). | GRAPH |
| BPE tokenizer | Byte-pair-encoding tokenizer parsed from GGUF metadata; special-token match matters (DeepSeek-R1 distill). | GRAPH |
| RoPE | Rotary position embedding applied to Q/K in-place, fused into decode QKV. | PRIMER, GRAPH |
| RMSNorm | Root-mean-square layer norm; fused with residual add in decode kernels. | PRIMER, GRAPH |
| SiLU / SwiGLU | Activation and gated FFN; SwiGLU is fused (gate+up concat + silu-mul). | PRIMER, GRAPH |
| softmax | Attention normalization; FA prefill uses tiled online softmax, FAP2 keeps it register-resident. | STEPS 46-48 |
| GQA | Grouped-query attention: fewer KV heads than Q heads; decode attention must replicate K/V. | STEPS 26 |
| KV cache | Persistent per-layer K/V tensors grown token by token; the "positions as data" rule keeps topology fixed. | GRAPH, PRIMER |
| `n_past` | Number of cached tokens; must never appear in graph topology, only as data. | GRAPH |
| draft model | Small model (0.5B) proposing `d` tokens the big model verifies. | D5, SPEC |
| draft length `d` | Tokens drafted per speculative round (d=2/4/8 measured). | D5 |
| acceptance rate `p` | Probability the target accepts a drafted token; measured p≈0.68–0.70 (llama `speculative-simple`, greedy). | D5, STEPS 80 |
| `E[a] = Σ_{i=1..d} p^i` | Expected tokens accepted per round under independence. | D5, PRIMER |
| verify pass | One batched forward (nt = d+1) checking all drafted tokens at target cost C_T(n). | D5 |
| speculative speedup rule | Win only if `E[a]·C_T(1) > C_T(d+1) + d·C_D` — the whole D5 plan reduces to this inequality. | D5, STEPS 80 |
| MTP (multi-token prediction) | Draft source using MTP heads (DeepSeek-V3 / Qwen3-Next GGUFs); unavailable to minfer's dense models. | SPEC, D5 |
| EAGLE-3 / DFlash / DSpark / n-gram self-speculators | The other draft mechanisms in llama.cpp's speculative zoo, outside `draft-simple`'s scope. | SPEC |
| MoE + MLA | Architecture prerequisite for MTP drafts — its own future campaign; minfer targets dense. | SPEC, D5 |

## L2 — Numerics & data formats

| Term / Formula | One-line meaning | Source |
|---|---|---|
| GGUF v3 | Single-file model format; multi-part files merge into one tensor index, entry = part 0. | GRAPH, BACKEND |
| Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 | Legacy block quants: fixed block of 32 values + fp scale(s); q8_0 also used for activations. | PRIMER, BACKEND |
| Q4_K / Q5_K / Q6_K | K-quants: super-block of 256 split into 8 (or 16 for q6_K) sub-blocks with packed multi-bit scales. | PRIMER, MMQ |
| super-block / sub-block | q4_K: 8×32 with 6-bit scales packed 4-per-32-bit word; q6_K: 16×16 with 8-bit scales + ql/qh nibble halves. | MMQ, PRIMER |
| `d`, `dmin`, `min` | Per-block fp16 scale, and for K-quant the per-super-block scale of scales / offset. | PRIMER |
| dsc | Per-super-block scale descriptor staging (q4_K DSC path; `MINFER_MMQ_Q4K_DSC`). | STEPS 38-39 |
| ql / qh | Low/high nibble halves of packed 4-bit weights (q6_K: ql+qh interleaved by bit plane). | MMQ |
| SWAR unpack | SIMD-within-register nibble→int8 expansion via bit masks instead of per-byte ops (r30). | STEPS 30 |
| dequantize / dequant | Converting packed blocks to arithmetic values inside the kernel; raw kernels avoid materializing f16. | MMQ, STEPS |
| raw-byte / raw-nibble kernels | Operate directly on packed bytes ("BT") or nibbles ("NB") without a dequant f16 round-trip. | MMQ, STEPS 28-38 |
| round-trip (quant↔dequant) | The correctness check pattern: quantize, dequantize, compare against fp reference. | STEPS |
| tolerance gate | Numerical acceptance: abs err ≤ 0.05 vs a CPU reference computed in f64. | HUB, STEPS 78 |
| bitwise identity | Stronger gate: refactor must produce byte-identical outputs (fused vs unfused is bit-identical by design). | GRAPH, STEPS 78 |
| f16 storage vs f32 accumulate | Weights/activations may be stored f16 (`__half`) but mma/dp4a accumulate in f32 or int32. | PRIMER, STEPS |
| int8 prefill activations | CUDA prefill quantizes activations to int8 for IMMA GEMM; decode MMVQ reads f32. | PRIMER, BACKEND |
| Q8_0 activation quant (CPU) | CPU quantizes activations on the fly; GPU reads f32 — logits differ by design, compare per-path. | GRAPH |
| f32-accumulate mma | r15 experiment: accumulate tensor-core results in fp32 registers instead of int32. | STEPS 15 |
| `__expf` scale path | q6_K dsc rebuild uses `__expf`; parity-guarded (`W_exp` debug, r44/r54 gates). | STEPS 44, 54 |
| W16 cache | CUDA-side cache of weights converted to f16 for some paths (`MINFER_NO_W16CACHE` to disable). | STEPS 34+ |
| split-k (dpl) | "dpl" = split-plane B layout used by the final q6_K BT kernel (doc 76). | STEPS 76 |
| MMVQ `uint4` sub-pairs | Vectorized 16-byte loads split per-thread sub-pairs in the MMVQ weight-streaming rework. | STEPS 12+ |
| `QI8_1` | llama.cpp MMQ tiling constant: int8-activation tile width per 32-k chunk (= QK8_1/(4·QR8_1) = 8). | MMQ |

### Tensor-layout symbols (from METAL §5.6)

| Symbol | One-line meaning | Source |
|---|---|---|
| `n_embd` / `n_head` / `nk` | Model hidden size, query-head count, KV-head count; `gqa = n_head/nk`. | METAL |
| `hd` / `hd_kv` | Attention head dim and KV head dim (may differ under GQA). | METAL |
| `nt` / `nkv` / `nkt` | Tokens in the batch (decode nt==1), KV positions used, KV capacity. | METAL |
| `od` / `id` / `nf` | Matmul output/input dims (weight rows/cols) and FFN intermediate dim. | METAL |
| `positions` | Per-token KV position array; `nkv = positions[t] + 1`. | METAL |
| `ne00..ne33` | ggml tensor dims: `ne0x` = dim0 of the x-th src, `ne1x` = dim1, etc. | METAL |
| `nb10..nb33` | ggml byte strides per dim for src1 (`nb10` elem stride, `nb11` row/token stride). | METAL |
| `ns10` / `ns20` | Element counts per head/row/token (`nb11/nb10`, `nb21/nb20`) — flash-KV inner-loop stride. | METAL |
| `nwg` / `nsg` | Workgroups and simdgroups per threadgroup (Metal launch geometry). | METAL |

## L3 — Performance model

| Term / Formula | One-line meaning | Source |
|---|---|---|
| memory-bound / compute-bound | Limited by bytes moved vs FLOPs issued; GB10 decode is memory-bound, prefill compute-bound. | PRIMER |
| roofline model | Performance ceiling = min(peak FLOPs, AI × peak bandwidth); AI = FLOPs per byte. | PRIMER |
| arithmetic intensity (AI) | FLOPs per byte of traffic; decode GEMM at nt=1 is ~1 MAC/weight-byte → bandwidth-bound. | PRIMER |
| GB/s, TB/s | Effective bandwidth; GB10 unified LPDDR5x ~273 GB/s shared CPU+GPU. | PRIMER, STEPS |
| tok/s | Decode throughput in tokens per second (headline metric: 7B q4_k_m CUDA 54.3). | HUB, STEPS 80 |
| MAC / GMAC | Multiply-accumulate; GMAC = 10⁹ MACs; TMAC/s = 10¹² MACs per second (kernel-level throughput). | STEPS 65-73 |
| M/GMAC | SASS instructions issued per GMAC — the instruction-stream efficiency metric (llama 6.06 vs ours 10.14). | STEPS 65-72 |
| amortization | Spreading fixed weight traffic over more rows: nt=4 batched decode gives BT-MMQ 2.7×. | PRIMER, STEPS 80 |
| `C_T(n)` | Cost of a target verify at batch n; C_T(1)=18.42 ms for 7B q4_k_m CUDA. | STEPS 80 |
| `C_D` | Draft-model cost per token (0.5B: 2.92 ms CUDA → CPU-draft dead at 1.35×). | STEPS 80 |
| break-even `p*` | Minimum acceptance rate for speculative win: 0.73 / 0.81 / 0.90 at d=2/4/8. | STEPS 80 |
| D5-0 gate | Condition to proceed: measured nt=3 verify amortization ≥ 2.5×. | STEPS 80 |
| batched-decode regime | nt = tokens per decode step; nt=4 is the campaign's anchor amortization point. | PRIMER, STEPS |
| KV traffic share | Per-token bytes = weights (dominant) + KV read/write + logits; quantized KV shrinks the KV share. | PRIMER |
| 3× gap attribution | Method of splitting the llama.cpp-vs-minfer wall-clock gap into per-kernel shares before optimizing. | STEPS 12, 47, 65 |
| wavefront count | Shared-memory work serialized per wavefront — 1.76× wavefronts/IMMA at equal IMMA rate meant inefficiency, not scarcity. | STEPS 36 |
| bytes-per-token | Decomposition of decode memory traffic; the roofline input for every decode optimization. | PRIMER |

## L4 — Micro-architecture (kernel implementation)

| Term / Formula | One-line meaning | Source |
|---|---|---|
| SM (streaming multiprocessor) | The GPU core unit; GB10 has 6144 CUDA cores across SMs; occupancy counts blocks/SM. | PRIMER |
| blocks/SM | Resident blocks per SM; NB kernel uses 2, q6_K BT uses 3 (r40 probe). | STEPS 28, 40 |
| occupancy | Ratio of resident warps to maximum; raised by lowering registers/smem per block. | STEPS 12, 40 |
| register pressure | Too many registers per thread kills occupancy; measured via ptxas `-v` spill output. | STEPS 12-73 |
| `__launch_bounds__` | Compiler directive capping registers/threads to hit a target occupancy. | PRIMER, STEPS |
| warp | 32 threads executing in lockstep; divergence inside a warp serializes paths. | PRIMER |
| tile / tile shape | The M×N×K block a kernel iterates over; TM=128/256 x-tile widening experiments (r13, r23). | PRIMER, MMQ |
| tile-regime | Which pre-tuned launch/tile configuration an (M,N,K) shape lands in — the D5-0 pivot concept. | PRIMER, STEPS 80 |
| wave quantization | Partial last wave of blocks leaves SMs idle; small GEMMs must size grids to avoid it. | PRIMER, STEPS 12 |
| mma.sync m16n8k16 | Tensor-core int8 matrix-multiply-accumulate instruction; the BT GEMM inner op. | PRIMER, MMQ |
| wmma | Legacy warp-level matrix API; used for FA prefill P·V on tensor cores. | STEPS 20 |
| dp4a | 4-way int8 dot-product instruction; the MMVQ decode path's inner op. | PRIMER, MMQ |
| IMMA / tensor pipe | The int8 tensor-core hardware pipe; `smsp__inst_executed_pipe_tensor_subpipe_imma` counts it. | STEPS 36, 65 |
| LDSM / `ldmatrix` | Loads an 8×8 f16 fragment into registers laid out for mma; A-fragment reuse ratio 0.125 vs 0.5 was the llama edge. | STEPS 36, 65 |
| A-fragment / B-fragment | The mma operand fragments each warp holds; reuse rate decides LDSM traffic. | STEPS 36, 65 |
| LDGSTS / `cp.async` | Async global→shared copy bypassing registers; the BT kernels stage A/B/dsc with it. | PRIMER, STEPS 45 |
| cp.async-db2 | cp.async with 2-stage double buffering (`MINFER_MMQ_RAW_*` sched gates). | STEPS 45-54 |
| LDG / STS / LDS | Global load, shared store, shared load — the synchronous counterpart trio. | STEPS, PRIMER |
| SASS instruction names (IMAD, I2F, F2I, FMUL, FFMA, LOP3, SHF, PRMT, LEA, SEL, CS2R, IADD3, FADD, BRA) | The assembly opcodes read in SASS forensics to count real work per loop. | STEPS 14-73 |
| shared memory / dynamic smem | On-chip scratchpad; sized via `cudaFuncAttributeMaxDynamicSharedMemorySize`. | PRIMER, BACKEND |
| bank conflicts | Simultaneous LDS hits to the same bank serialize; r22 removed them via layout. | STEPS 22 |
| swizzle | XOR-based shared-memory address permutation to avoid bank conflicts. | PRIMER, STEPS 22 |
| scoreboard stall | Warp waiting on a memory dependency tracked by L1TEX scoreboard (r41 attack). | STEPS 41 |
| MIO pipe | Memory-IO instruction queue; shown not scarce (r36) — A-fragment reuse was. | STEPS 36 |
| coalescing | Warp-wide global accesses touching contiguous lines; r21 coalesced A staging. | STEPS 21 |
| L2 window / access policy window | Pinning a buffer's residency in L2 via `cudaAccessPolicyWindow` (`MINFER_MMQ_L2WIN`). | STEPS 23 |
| KSPLIT | Splitting the K reduction across blocks with atomic/partial adds (q6_K KSPLIT=2). | STEPS 39 |
| KS (k-step) | K elements processed per inner iteration (KS=64 GEMM). | STEPS 12 |
| KD / KDR | K-depth unroll factor and K-depth register pipeline depth (q6_K KDR=4, =8 regressed). | STEPS 39-40 |
| double buffering | Overlapping stage(n+1) loads with compute(n); the NB/BT staging pattern. | STEPS 9, 45 |
| software pipelining | Restructuring the loop so load/compute phases of different iterations overlap. | STEPS 24 |
| unroll (kd-loop) | Compiler/pragma loop unrolling to expose ILP; r28 "NB kd-loop unroll". | STEPS 28 |
| epilogue | The post-mma tail: scaling, output store; r32 cut its cost. | STEPS 32 |
| B pre-format / quantize-transpose prepass | Repacking B (weights) offline into kernel-friendly layout (r34). | STEPS 34 |
| `MMVQ_PARAMETERS_GB10` | llama.cpp launch-config constant table for GB10 MMVQ, adopted by minfer. | STEPS 12 |
| block reduce | Warp/block-wide reduction for logits accumulation (`mmvq_block_reduce`). | STEPS 12 |
| PDL / programmatic dependent launch | Overlapping dependent kernel launch tails: `cudaGridDependencySynchronize` + programmatic stream serialization attribute. | STEPS 74, PRIMER |
| griddepcontrol | The SASS/PTX-level instruction pair behind PDL. | STEPS 74 |
| wave (n) | One full pass of all resident blocks; kernel iteration wave counting for sched analysis. | STEPS 36 |
| elect.sync | Warp election intrinsic seen in SASS forensics. | STEPS 65 |
| `MMQ_TILE_NE_K` / `MMQ_TILE_Y_K` | llama.cpp MMQ shared-memory tile pitches (B tile 32+4 ints; y-tile row stride 36 ints = 144 B, the +4 avoids bank conflicts). | MMQ |

## L5 — Platform & tooling

| Term / Formula | One-line meaning | Source |
|---|---|---|
| GB10 / DGX Spark | The target machine: Grace 20-core ARM + Blackwell GPU, unified LPDDR5x, sm_121. | PRIMER, BUILD |
| sm_XX / compute_XX | GPU arch targets; build emits SASS for sm_70…sm_121 probes + PTX compute_70/72 for backward JIT. | BUILD, PRIMER |
| nvcc / ptxas | CUDA compiler and its SASS backend; ptxas `-v` gives register/spill counts. | BUILD, STEPS |
| PTX | Virtual ISA JIT-compiled at load; the backward-compatibility artifact. | BUILD, PRIMER |
| SASS | Real GPU assembly; forensic disassembly via `cuobjdump`/`nvdisasm`. | STEPS 14+ |
| `-ccbin` pinning | Forcing nvcc's host compiler when the default is rejected (`MINFER_CUDA_CCBIN`). | BUILD |
| detect_archs | build.rs probing sm_70…sm_121 by compiling a dummy `.cu` per arch. | BUILD |
| `libcuda_kernels.a` | Static kernel archive nvcc produces, linked into the Rust binary. | BUILD |
| cuda_static feature | Link cudart statically so no libcudart.so is needed at runtime. | BUILD |
| CUDA driver vs runtime API | `libcuda` low-level vs `libcudart` convenience layer; minfer binds runtime via hand-written externs. | PRIMER, BACKEND |
| CUDA Graphs | Captured kernel sequence replayed with one launch (`cudaStreamBeginCapture`/`cudaGraphLaunch`/`cudaGraphInstantiate`); `MINFER_NO_CUDA_GRAPH` to disable. | BACKEND, STEPS 19 |
| graph capture (prefill) | Pre-capturing the prefill segment (`MINFER_CAPTURE_PREFILL`). | STEPS 57+ |
| pinned memory | Page-locked host memory (`cudaHostAlloc`) for fast H2D/D2H; readback path has a kill switch. | BACKEND, STEPS |
| `cudaMemcpyAsync` / streams | Async copies on streams (`cudaStreamCreate`); decode uses graph launch, prefill streams. | BACKEND |
| cudaMallocManaged / unified memory | Memory visible to both CPU and GPU (used once; avoided on GB10 due to bandwidth sharing). | PRIMER |
| `cudaFuncSetAttribute` | Runtime call to raise per-kernel dynamic smem limits. | BACKEND |
| error guards (`cudaGetLastError`) | Every launch checks the error; `cudaErrorMisalignedAddress` was a real campaign bug. | STEPS 13, SAFETY |
| ncu (Nsight Compute) | Kernel profiler: `sm__warps_active`, `lts__t_sectors`, `smsp__inst_executed` metric families. | STEPS 12+ |
| nsys (Nsight Systems) | Timeline profiler for wall decomposition and graph-launch analysis. | STEPS 57+ |
| locked clocks | `nvidia-smi -lgc` fixes GPU clocks so medians are comparable across runs. | STEPS 77 |
| `MINFER_*` env gates | ~40 kill-switch env vars (`MINFER_MMQ_RAW`, `MINFER_MMQ_A_TRANSPOSE`, `MINFER_PDL`, `MINFER_FUSED_B`, …) toggling one experiment at a time. | HUB, STEPS |
| cuobjdump / nvdisasm | Tools producing the SASS listings used in forensics. | STEPS 25 |
| MINFER_TRACE / MINFER_GRAPH_DUMP | Per-node real-data trace and graph dumps for viz tooling. | GRAPH |

## L6 — Engine architecture

| Term / Formula | One-line meaning | Source |
|---|---|---|
| ComputeGraph / CNode | The declarative graph and its nodes; inference = build → assign → fuse → allocate → execute. | GRAPH |
| Op enum / NodeMeta | Typed operations and per-node metadata in `graph/ops.rs`. | GRAPH |
| GraphBuilder | Deterministic graph construction; identical `GraphParams` ⇒ identical topology. | GRAPH |
| GraphParams / params-only reuse | Reuse check compares parameters only (`GraphCache::try_reuse`), never data. | GRAPH |
| CParams.gpu | Participation flag recording whether the run used the GPU backend. | GRAPH, BACKEND |
| Backend trait | `supports_op`/`supports_fused`, buffer pool, `execute_node`, host IO, `synchronize` — the CUDA backend is the worked example. | GRAPH, BACKEND |
| GraphAllocator / liveness | Single buffer owner; allocates by liveness in build order; persistent KV regions survive rebuilds. | GRAPH |
| kv_pair / persistent KV regions | Each layer owns two allocator regions (K/V) that outlive a decode step. | GRAPH |
| scheduler (assign → split → execute) | Assigns backends, splits the graph at backend boundaries, executes splits serially. | GRAPH |
| split boundary | Cross-backend sync/copy point; one Metal command buffer per split (CUDA: one stream). | GRAPH |
| FusionPass | Build-time fusion of QKV (bias+rope+store) and FFN (swiglu); fused vs unfused bit-identical; gated by env. | GRAPH |
| positions-as-data | Rule 1: topology never depends on `n_past` — the precondition for decode reuse and CUDA graphs. | GRAPH |
| fill_input_i32 | Integer inputs stored via `f32::from_bits` so the f32-typed input buffer carries token ids. | GRAPH |
| in-place aliasing rule | Silu/RoPE alias their input (sole consumer + same backend); never host-copy a GPU-pending buffer. | GRAPH, SAFETY |
| ModelDef trait | Per-architecture `forward`/`build_graph`/`forward_graph`; models live in `models/<name>/`. | GRAPH |
| weight layout convention | Metadata `[in, out]`, memory row-major `[out][in]`, activations token-major `[nt][d]`. | GRAPH |
| guard failure = abort | Kernel-invariant violations return `Err` from `execute_node` — never silent CPU fallback; guards print actual values. | SAFETY |
| `submit()` bounded wait | GPU submission waits bounded and checks status — never blocks forever. | SAFETY |
| no early return past barrier | Metal/CUDA rule: no exit path may skip a `threadgroup_barrier`/`__syncthreads`. | SAFETY |
| prefill capture | Backend feature storing the captured prefill graph for replay. | BACKEND |
| IR (intermediate representation) | The graph as an op-level IR; fusion makes fused ops first-class IR citizens. | GRAPH |
| NodeId / DType | Node handle and tensor data-type enum carried by every CNode. | GRAPH |
| `GetRows` | Row-selection op: embedding lookup, and the n_out tail-row optimization (G3). | GRAPH |
| `BatchMatMul` | Batched matmul op (shared activation quantization, Q4_0); composable with fusion. | GRAPH |
| FusedOp / `supports_fused` | Pattern-matched fusion ops (SwiGLU / FusedBiasRope) dispatched by backend capability. | GRAPH |
| `n_out` tail-row optimization | After the final `wo`, run FFN/norm/lm_head only on the tail n_out rows (llama `inp_out_ids` style); `GraphParams.n_out` joins the reuse decision. | GRAPH |
| DOT / JSON export | `graph/dot.rs` and `graph/json.rs` render the graph for viz. | GRAPH |

## L7 — Methodology

| Term / Formula | One-line meaning | Source |
|---|---|---|
| interleaved same-window A/B | Alternating minfer/llama.cpp runs inside one time window so thermal/clock drift cancels. | STEPS 77, HUB |
| median of N | Take the median of repeated runs; means are corrupted by outliers on shared hardware. | STEPS 77 |
| pre-registered bar | The success threshold is written into the step doc *before* measuring. | STEPS 77 |
| parity gate | Greedy output must match llama.cpp (or CPU f64 ref within 0.05) before any perf number counts. | HUB, STEPS 77-78 |
| correctness batch | Docs 78/79: batch verification sweeps over all kernels/quant types. | STEPS 78-79 |
| MEAS-ONLY | Step status: measurement without landing code (e.g. doc 80). | HUB |
| LANDED / REVERTED | Step outcome statuses in the hub tables. | HUB |
| r-numbers (r1…r76) | Experiment numbering across the MMQ/GEMM campaign; each step doc records one. | HUB, MMQ |
| Era A/B/C/D | Campaign phases: baseline (A), MMVQ (B), MMQ GEMM (C), decode/GEMM (D). | HUB |
| Direction-A/B, Session A–F | Named experiment tracks within a phase (e.g. raw-nibble vs BT; quantize-fusion sessions). | STEPS 28-56 |
| SASS forensics | Explaining a perf delta by diffing disassembly (r25 opcode diff) instead of guessing. | STEPS 25, 65 |
| wall decomposition | Splitting end-to-end time per kernel/phase before optimizing anything. | STEPS 12, 47, 65 |
| counter-guided iteration | Next experiment chosen by the ncu counter that bounds the kernel (occupancy → wavefronts → IMMA rate). | STEPS 36-73 |
| one-variable-at-a-time | Each experiment toggles exactly one gate/env var; everything else frozen. | STEPS 77 |
| llama.cpp as reference | `$HOME/git/reading/llama.cpp` is the ground truth for both parity and technique adoption. | MMQ, SPEC |
| doc-per-step convention | Every step writes one numbered record with a fixed six-section structure (STYLE.md). | STEPS STYLE |
| measurement artifacts | Raw bench JSONs kept under `/tmp` per step and reported, never committed. | STEPS 77-80 |
| G1 / G2 / G3 | Graph-refactor Phase-9 sub-experiment labels (attention dispatch, `rms_norm_256`, n_out tail-row). | GRAPH |
