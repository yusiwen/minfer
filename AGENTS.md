# minfer — AI Agent Context

minfer is a pure Rust LLM inference engine written from scratch (~4400 LOC), inspired by llama.cpp with 0 ML framework dependencies.
Supports **Qwen2/Qwen2.5 architecture**, **CPU + Apple MPS (Metal) GPU** inference, and **GGUF v3 format**.
Inference runs through a **declarative compute graph** (builder → scheduler → per-backend kernels), modeled on llama.cpp's `ggml_cgraph` + backend scheduler — see `docs/GRAPH-REFACTOR-PLAN.md` for the design and implementation record.

This file is the always-loaded project index. Deep dives, performance analysis, and historical fixes live in `docs/` (index at the bottom) — don't re-litigate them here.

## Code Search (ccc)

minfer is a llama.cpp-inspired engine; both codebases are indexed with [ccc](https://cocoindex.io/cocoindex-code/). When the question is "where / how is X implemented", prefer ccc semantic search over whole-repo grep/rg.

- **minfer's own code** -> use the `ccc` MCP `search` tool, or CLI: `ccc search "<query>"` from this directory (index: `.cocoindex_code/`, 1255 chunks).
- **llama.cpp reference source** (`$HOME/git/reading/llama.cpp`, C++ upstream: ggml, quantization kernels, sampler, tokenizer, llama_server, etc.) -> use the `ccc-llamacpp` MCP `search` tool, or CLI: `cd $HOME/git/reading/llama.cpp && ccc search "<query>"` (index: 47326 chunks).
- **Structural search**: `ccc grep '<pattern>'` -- e.g. `ccc grep 'fn \NAME(\(A*\))' src/` for Rust functions, or `ccc grep '\NAME(\(A*\))'` inside llama.cpp for call sites.
- **Filters**: `--lang rust|cpp|...`, `--path 'src/**'` to narrow results.
- Both MCP servers auto-refresh their indexes (`refresh_index=true` default); stale results -> pass `refresh_index=true` explicitly or run `ccc index` / `ccc search --refresh`.

## Architecture at a Glance

```
src/
├── main.rs          # CLI + inference loop (prefill → autoregressive generation)
├── graph/           # ★ declarative compute graph — the inference core (see below)
├── gguf.rs          # GGUF v3 parser (~1650 lines, largest file)
├── block.rs         # 20+ quantized block types (repr(C), matching ggml-common.h)
├── avx2.rs          # AVX2 dot product kernels + f32→Q8_0 quantization
├── kernel.rs        # Quantized matmul dispatch + CPU scalar fallbacks (+ embed_tokens row getter)
├── vec_ops.rs       # SIMD vector ops (RMSNorm, RoPE, Softmax, SiLU)
├── tensor.rs        # 4D Tensor (shape/strides/data)
├── cache.rs         # Legacy KV Cache type (graph path owns KV in the allocator)
├── dump.rs          # Debug dump module (gated by `--features debug_dump`)
├── tokenizer.rs     # BPE tokenizer (self-contained, loaded from GGUF metadata)
├── sampler.rs       # Repeat-penalty / Top-K / Top-P / Temperature (seeded) sampling
├── template.rs      # ChatML / Llama3 / Mistral template rendering (minijinja)
├── conversation.rs  # Multi-turn conversation session (append-only KV + Engine abstraction)
├── server/          # OpenAI-compatible HTTP server (axum; types/slot/chat)
├── download/mod.rs  # HuggingFace + Ollama auto-download + cached-name resolution
├── metal.rs         # MPS kernels + per-op GPU methods (graph backend entry points)
├── metal.metal      # Metal GPU shaders (Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0 kernels)
├── cuda.rs          # CUDA backend (feature-gated; graph integration pending — Phase 7)
└── models/
    ├── mod.rs       # ModelDef trait + factory dispatch
    └── qwen2/
        ├── mod.rs   # Qwen2Model + ModelDef implementation
        ├── graph.rs # ★ build_graph + graph forward (Qwen2Graph)
        └── loader.rs # GGUF weight loading + GPU registration
```

### src/graph/ — the compute graph core

| File | Role |
|------|------|
| `mod.rs` | `ComputeGraph` (topo-validated node list + inputs/outputs), `CNode`, `DType`, `Backend`, `BufRef` |
| `ops.rs` | `Op` enum (full payload `PartialEq`), `NodeMeta`, `AttnMode`, `FusedOp` |
| `builder.rs` | `GraphBuilder` — declarative construction (embedding/matmul/rope/attn/kvcache/…) |
| `alloc.rs` | Per-backend liveness allocator + persistent per-layer KV regions + `KvProvider` |
| `backend.rs` | `Backend` trait (pool alloc/execute/host access/synchronize) + `KvProvider` |
| `cpu_backend.rs` | CPU execution (wraps kernel.rs + vec_ops.rs) |
| `metal_backend.rs` | Metal execution (per-op MPS kernels; `cfg(target_os = "macos")`) |
| `scheduler.rs` | assign → split → execute (+ cross-backend copies at split boundaries) |
| `fusion.rs` | Pattern-matching fusion (SwiGLU/BiasRope), gated by backend `supports_fused` |
| `cache.rs` | `GraphCache` — params-only deterministic graph reuse |
| `params.rs` | `GraphParams`/`CParams`/`GraphType` — the reuse identity |
| `dot.rs` | Graphviz DOT export (`--dump-graph`) |

## Build & Run

```bash
cargo build --release
cargo build --release --features debug_dump    # + per-layer debug dumps

./target/release/minfer <model.gguf> "hello"                      # run (compute-graph forward)
./target/release/minfer --graph <model> "hello"                   # accepted for compat (graph path is default)
./target/release/minfer --dump-graph graph.dot <model> "hello"    # export the prefill graph as DOT
./target/release/minfer --no-template <model> "prompt"            # raw prompt (skip chat template)
./target/release/minfer info <model>                              # list tensor names/types/shapes
MINFER_DISABLE_MPS=1 ./target/release/minfer <model> "hello"      # force CPU
MINFER_DUMP_DIR=/tmp/dump ./target/release/minfer <model> "hello" # debug dump (debug_dump build)
MINFER_GRAPH_DUMP=/tmp/d ./target/release/minfer --graph <model> "hello"  # dump graph logits/KV (any build)
```

Split (multi-part) GGUF is supported: entry is part 0; `load_gguf_model` parses every part and builds a merged tensor index; download resume is size-checked (curl `-C -`).

## Debug Dump

`MINFER_DUMP_DIR` + `--features debug_dump` writes per-layer hidden states (embed out, per-layer attn/FFN stages, final logits; `_gen0` suffix = first generation step). Full file list: `docs/debug-dump.md`. The graph path additionally supports `MINFER_GRAPH_DUMP=<dir>` (any build): writes `logits_<prefill|decode>.f32`, `kv0_<prefill|decode>.f32` and layer-0 intermediate nodes — useful for GPU-vs-CPU graph comparison.

## Quantization Support

Working: **Q4_0, Q4_1, Q8_0, Q4_K, Q6_K, Q5_0, Q5_1, Q5_K** (CPU + Metal GPU, see matrix below).
Not supported (CLI): Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL, etc.

## GPU Support Matrix

| Quant | MPS (Metal) | CPU |
|-------|-------------|-----|
| Q4_0, Q4_1 | ✓ (Q8_0-activation path + f32 path) | ✓ |
| Q4_K, Q6_K | ✓ (f32 path) | ✓ |
| Q8_0 | ✓ (f32 path) | ✓ |
| Q5_0 | ✓ (f32 path) | ✓ |
| Q5_1 | ✓ (f32 path) | ✓ |
| Q5_K | ✓ (f32 path, `kernel_q5_k_f32_matmul` + `_multi`) | ✓ |
| F32 | ✓ (RMSNorm, biases, etc.) | ✓ |

Prefill GEMM: Q4_0 simdgroup GEMM for nt ≥ 16 (`MINFER_GEMM=0` forces f32 multi); other quants use the f32 multi kernel. Perf/progress tracking (single source): `docs/METAL_OPTIMIZATIONS.md` §0.

## Model Support Matrix

| Model | CPU | MPS GPU (graph path) | Notes |
|-------|-----|---------|-------|
| Q4_0 (qwen2.5-0.5b-instruct-q4_0) | ✓ | ✓ (~200+ tok/s) | All weights Q4_0; graph-vs-old-forward logits bit-identical (max diff 0.0) |
| Q4_K_M (qwen2.5-0.5b-instruct-q4_k_m) | ✓ | ✓ | Q5_0/Q8_0/Q4_K/Q6_K mixed |
| Q5_K_M (qwen2.5-0.5b-instruct-q5_k_m) | ✓ | ✓ | Q5_1/Q8_0/Q5_K/Q6_K, full GPU |
| Q4_K_M (qwen2.5-7b-instruct-q4_k_m) | ✓ | ✓ (~42 tok/s) | hd=128, split GGUF, graph path verified end-to-end |

## GPU Safety

Read `docs/GPU_SAFETY.md` before touching Metal code. Hard rules (apply to the per-op kernels in `metal.rs` and to `MetalBackend`): `submit()` waits bounded (10 s) + checks status, reports the dispatch trace (`MINFER_TRACE=1`) and exits — never blocks forever; no early return past a `threadgroup_barrier` in a kernel (GPU deadlock/freeze); device limits (threadgroup memory/threads) queried at runtime, never hardcoded; all guard failures `gpu_abort` with actual values. In the graph architecture, **kernel-invariant violations return `Err` from `execute_node` and must NOT be treated as a silent CPU fallback** — backend assignment at build time decides which ops run where; only genuine support limitations (e.g. Raw weights) select the CPU backend.

## Compute Graph Architecture (core rules)

Inference = **build a `ComputeGraph` (pure, side-effect free) → assign backends → fuse → allocate → execute**. The graph is built once per distinct `GraphParams` and reused (decode steps reuse the same graph). The model's `forward()` routes through this (Qwen2's imperative forward was deleted in Phase 6).

1. **KV positions are data, not structure.** `Op::KvcacheStore/Load` carry only the layer index; write positions come from the `positions` input node. Graph topology **never** depends on `n_past` — this is the precondition for decode-time reuse (llama.cpp `allow_reuse` same invariant).
2. **Each layer owns TWO persistent KV regions (K and V)**, resolved by `kv_pair(layer)` (a `KvProvider` the allocator implements). The store node's output buffer is the K region; backends write the V sibling via `kv_pair`. Regions survive graph rebuilds (the allocator lives inside `GraphCache`).
3. **Reuse is params-only.** `GraphParams` (n_tokens/n_seqs/gtype/cparams/weights_version) deterministically determines the topology; `n_past` is absent, and `CParams.gpu` records backend participation (a backend config change forces a rebuild). `GraphCache::try_reuse` compares params only; debug builds assert structural consistency.
4. **Weight layout is llama.cpp/GGUF convention.** Tensor metadata `[in, out]` (ne[0] fastest) with memory `[out][in]` row-major → matmul `od = shape[1]`, `id = shape[0]`. Activations: shape metadata `[d, nt, 1, 1]`, memory token-major `[nt][d]`. I32 inputs (token ids/positions) are stored as `f32::from_bits` bit patterns (exact for |v| < 2^24); use `fill_input_i32`.
5. **In-place ops (`Silu`, `RoPE`) alias their input buffer** — the allocator maps their output to the input's `BufRef` (only when the input's sole consumer is this op AND it is on the same backend). **Never host-copy a GPU-pending buffer**: a host `copy_in` of a producer that has been encoded but not submitted reads stale data (the Phase-3 KV-corruption bug). Cross-backend in-place inputs get a fresh buffer (the producer completed before the split boundary, so the copy is safe there).
6. **Execution follows build order** (the builder appends sources before consumers, so node order is a valid topological order — ggml executes `nodes[0..n]` the same way). This guarantees a KV store executes before the attention that reads it. Nodes with no allocated buffer (dead, e.g. fusion orphans) are skipped. **The allocator's liveness uses the same build order** (`topo_order()` may reorder srcless nodes like `kv_load` ahead and would let reuse clobber a still-alive input — G3 regression); **input buffers are never freed** (all inputs are host-filled before execution, so two inputs sharing a buffer would clobber each other at fill time).
7. **decode (nt==1) QKV fusion (G4, `Op::FusedQKV`)** replaces the 3 matmul + 3 bias + 2 rope + 2 store chain with one concat matmul (`blk.{i}.attn_qkv`, loader-registered `wq|wk|wv` rows) + one `attn_bias_rope_store` pass; attention reads q from concat offset 0 (nt==1 ⇒ no stride issue). Gated by `nt==1 && gpu && fuse_qkv` (part of the reuse identity — `MINFER_NO_FUSE_QKV=1` forces a rebuild for A/B). **`attn()` output shape comes from `AttnMeta` (n_head*hd), not the q input's shape** — the fused q is a larger concat buffer. Fused vs unfused decode logits are bit-identical (0.0, verified 0.5B + 7B). **FFN gate+up fusion (G5, `Op::FusedFFN`)** does the same for the FFN: one concat matmul (`blk.{i}.ffn_gu`) + one in-place `swiglu_f32_off` (same kernel as the fused `Op::SwiGLU` — bit-identical), down reads rows 0..nf. Gated `nf ≤ 16384` (7B Q4_K concat matmul measured slower); `MINFER_NO_FUSE_FFN=1` reverts. Test rule: when comparing fused vs unfused graphs, the unfused path MUST run the FusionPass (silu+mul → SwiGLU) — otherwise the two-kernel silu+mul differs from the one-kernel swiglu by ~1e-6 float noise.
8. **Backends own their buffer pools; the allocator is the single owner.** The scheduler orchestrates assign → fuse → alloc → execute and performs cross-backend copies (`copy_across`, a host round trip through shared memory) at split boundaries after `sync_backend`. Metal batching: one `MpsCommandBuffer` per split, submitted at `synchronize()`.
9. **CPU activations are Q8_0-quantized on the fly** (Q4_0×Q8_0 etc.); **Metal reads f32 activations directly** — so graph-CPU vs graph-Metal logits differ by the activation-quantization path (~1e1 on logits; each path is internally correct). Compare Metal against manual Q8_0×f32 references or layer-gpu-style math, not against the Q8_0-activation CPU path.

## Core Conventions

1. Activations quantized to Q8_0 on-the-fly — all CPU matmuls use Q8_0 quantized activations (Q5_0 → `dot_q5_0_q8_0()`, Q4_K → `dot_q4_k_q8_0()`). Metal reads f32 activations directly for all weight types (Q4_0 included since P1), matching llama.cpp's Metal backend.
2. AVX2 dispatch: `is_x86_feature_detected!("avx2")` + scalar fallback (ARM Mac always scalar).
3. No ML frameworks — Attention, RMSNorm, RoPE, SiLU, Softmax all handwritten.
4. Tensor data uses raw `&[u8]` — avx2.rs dot products operate on byte slices, not structs.
5. GGUF padding: `ggml_pad()`: `(x + n - 1) & !(n - 1)`.
6. Cross-backend execution: splits sync and copy at boundaries; per-op backend assignment is decided at graph build time by `supports_op` (weights registered on the backend decide feasibility); guard failures abort (see GPU Safety) — never silent mid-execution fallback.

## Adding a New Architecture

1. Create `models/<name>/` with `mod.rs`, `graph.rs`, `loader.rs`
2. Add dispatch branch in `models/mod.rs::load_model()`
3. Define `HParams` (including `n_kv_embd`) and `LayerWeights` in `loader.rs`
4. Implement `build_graph(&self, params: &GraphParams) -> ComputeGraph` in `graph.rs` — deterministic in params (reuse invariant). Use `GraphBuilder` (embedding/rms_norm/matmul/rope/kvcache_store/load/attn/swiglu/add), mirroring llama.cpp's `llm_graph_context` builder methods.
5. Implement `ModelDef` in `mod.rs`: `forward` (Qwen2 routes it through the graph path), `build_graph`, `forward_graph`, `as_any` (downcast for weight registration). `weights_version` lives in `GraphParams` (bump on LoRA/weight changes to break reuse).
6. Add template format support in `template.rs` if needed

## Adding a New Backend (e.g. CUDA — Phase 7)

1. Implement the `Backend` trait (`src/graph/backend.rs`) for the new device:
   - `supports_op(op, dtype)` / `supports_fused(fused)` — capability gates for assignment and the fusion pass
   - `alloc_buffer`/`free_buffer` — the backend's own buffer pool (ids are resolved inside `execute_node`)
   - `execute_node(node, in_bufs, out_buf, kv_pair)` — per-op dispatch; resolve weights by name from the backend's registry (`metal.rs` pattern: `weight_buf(name) -> (buffer, offset)`); honor the **in-place alias** rule (see Compute Graph rules §5) and the GGUF weight layout (§4)
   - `read_host`/`write_host` — host access for input filling and cross-backend copies (shared-memory or staged transfers)
   - `synchronize` — flush async work (split boundaries)
2. Register it in `GraphAllocator` (follow `enable_metal`/`metal_mut`): add an `Option<CudaBackend>`, a `supports()` priority (GPU before CPU), and a `sync_backend`/`copy_across` arm.
3. Wire the model layer: register weights on the new backend at load (`loader.rs`), and gate graph execution on "all weights registered" (mirror `Qwen2Graph::weights_on_gpu`). Record backend participation in `CParams.gpu` so a backend toggle forces a graph rebuild.
4. CUDA specifics (recorded in the plan §9/§17): wrap the existing `cuda.rs` (do NOT stub it — it already has `layer_gpu` + CUDA Graph capture), keep CUDA Graph caching keyed on the graph `uid`, and map `supports_op` from the existing capability matrix. Requires nvcc to build (`cargo build --features cuda`).

## Sampling

`sampler.rs`: repetition penalty → top-k → top-p → temperature, seeded `StdRng`. Defaults match llama.cpp: `temp=0.8`, `top_p=0.95`, `repeat_penalty=1.1`. CLI: `--temp`, `--greedy` (temp=0), `--top-k`, `--top-p`, `--repeat-penalty`, `-n/--n-predict`, `--seed`. Repetition penalty applies to the last 64 tokens (llama `repeat_last_n`): positive logits ÷ penalty, negative ×; alone fixes the 0.5B greedy repetition loops.

## Dependencies

Core inference deps are minimal: `rand` (sampling), `regex` (BPE pre-tokenization), `half` (fp16),
`serde+serde_json` (download API + `--session` history), `minijinja` (template rendering).
The OpenAI-compatible server adds `axum`/`tokio`/`tokio-stream`/`tower-http`/`uuid`/`futures-util`;
macOS adds `metal`/`block`.

## Documentation Locations

All non-root documentation lives in **`docs/`** (the project root only keeps `AGENTS.md` and `README.md`).

| Topic | Location |
|---|---|
| Overall architecture design (module map, pipeline, backend layering, quant layout, adding an arch) | `docs/ARCHITECTURE.md` |
| **Compute graph design + rewrite plan + implementation record (per-phase commits)** | `docs/GRAPH-REFACTOR-PLAN.md` |
| llama.cpp compute-graph design analysis (ggml_cgraph / scheduler / reuse) | `docs/LLAMA-COMPUTE-GRAPH.md` |
| Metal backend optimization plans/gap analysis (primary tracking doc) | `docs/METAL_OPTIMIZATIONS.md` |
| objc 0.2 vs objc2 ecosystem — why block is vendored, nix devShell xcrun fix, migration path | `docs/METAL_OBJC-ECOSYSTEM.md` |
| GPU safety conventions + audit | `docs/GPU_SAFETY.md` |
| CPU backend optimizations | `docs/CPU_OPTIMIZATIONS.md` |
| CUDA backend (draft) / problems | `docs/CUDA_OPTIMIZATION.md`, `docs/CUDA_PROBLEMS.md` |
| Parameter audit vs llama.cpp | `docs/PARAMETER_AUDIT.md` |
| KV cache indexing bug #6 | `docs/BUG-6-KV-CACHE-INDEXING.md` |
| Debugging plans / summaries | `docs/DEBUGGING-PLAN.md`, `docs/DEBUGGING-SUMMARY.md` |
| Qwen2.5-1.5B bugs / debugging notes | `docs/QWEN2.5-1.5B-BUGS.md`, `docs/QWEN2.5-DEBUGGING-NOTES.md` |
| Debug dump format reference | `docs/debug-dump.md` |
| Metal inference / multi-token kernel analyses | `docs/metal-inference-analysis.md`, `docs/multi-token-kernel-analysis.md` |
| **OpenAI-compatible Chat API plan (Plan B: multi-slot + serial)** | `docs/OPENAI-CHAT-API-PLAN.md` |
| **CLI multi-turn conversation plan (append-only KV + incremental template diff)** | `docs/CLI-CONVERSATION-PLAN.md` |
