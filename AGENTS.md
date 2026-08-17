# minfer — AI Agent Context

minfer is a pure Rust LLM inference engine written from scratch (~4400 LOC), inspired by llama.cpp with 0 ML framework dependencies.
Supports **Qwen2/Qwen2.5 architecture**, **CPU + Apple MPS (Metal) GPU** inference, and **GGUF v3 format**.

This file is the always-loaded project index. Deep dives, performance analysis, and historical fixes live in `docs/` (index at the bottom) — don't re-litigate them here.

## Architecture at a Glance

```
src/
├── main.rs          # CLI + inference loop (prefill → autoregressive generation)
├── gguf.rs          # GGUF v3 parser (~1650 lines, largest file)
├── block.rs         # 20+ quantized block types (repr(C), matching ggml-common.h)
├── avx2.rs          # AVX2 dot product kernels + f32→Q8_0 quantization
├── kernel.rs        # Quantized matmul dispatch + CPU scalar fallbacks
├── vec_ops.rs       # SIMD vector ops (RMSNorm, RoPE, Softmax, SiLU)
├── tensor.rs        # 4D Tensor (shape/strides/data)
├── cache.rs         # KV Cache
├── dump.rs          # Debug dump module (gated by `--features debug_dump`)
├── tokenizer.rs     # BPE tokenizer (self-contained, loaded from GGUF metadata)
├── sampler.rs       # Repeat-penalty / Top-K / Top-P / Temperature (seeded) sampling
├── template.rs      # ChatML / Llama3 / Mistral template rendering (minijinja)
├── download/mod.rs  # HuggingFace + Ollama auto-download + cached-name resolution
├── metal.rs         # Apple MPS (Metal) GPU backend + dispatch
├── metal.metal      # Metal GPU shaders (Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0 kernels)
└── models/
    ├── mod.rs       # ModelDef trait + factory dispatch
    └── qwen2/
        ├── mod.rs   # Qwen2Model + ModelDef implementation
        ├── forward.rs  # Forward pass (quantized inference, CPU + GPU fallback)
        └── loader.rs   # GGUF weight loading + MPS/CUDA GPU registration
```

## Build & Run

```bash
cargo build --release
cargo build --release --features debug_dump    # + per-layer debug dumps

./target/release/minfer <model.gguf> "hello"                      # run
./target/release/minfer --no-template <model> "prompt"            # raw prompt (skip chat template)
./target/release/minfer info <model>                              # list tensor names/types/shapes
MINFER_DISABLE_MPS=1 ./target/release/minfer <model> "hello"      # force CPU
MINFER_DUMP_DIR=/tmp/dump ./target/release/minfer <model> "hello" # debug dump (debug_dump build)

# Auto-download (quant auto-matches single or split files, case-insensitive)
./target/release/minfer download hf Qwen/Qwen2.5-0.5B-Instruct-GGUF Q4_0
./target/release/minfer download ollama qwen2.5:0.5b
```

Split (multi-part) GGUF is supported: entry is part 0; `load_gguf_model` parses every part and builds a merged tensor index; download resume is size-checked (curl `-C -`).

## Debug Dump

`MINFER_DUMP_DIR` + `--features debug_dump` writes per-layer hidden states (embed out, per-layer attn/FFN stages, final logits; `_gen0` suffix = first generation step). Full file list: `docs/debug-dump.md`.

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

| Model | CPU | MPS GPU | Notes |
|-------|-----|---------|-------|
| Q4_0 (qwen2.5-0.5b-instruct-q4_0) | ✓ | ✓ (361 tok/s) | All weights Q4_0 |
| Q4_K_M (qwen2.5-0.5b-instruct-q4_k_m) | ✓ | ✓ (226 tok/s) | Q5_0/Q8_0/Q4_K/Q6_K mixed |
| Q5_K_M (qwen2.5-0.5b-instruct-q5_k_m) | ✓ | ✓ (~250 tok/s) | Q5_1/Q8_0/Q5_K/Q6_K, full GPU |

## GPU Safety

Read `docs/GPU_SAFETY.md` before touching Metal code. Hard rules: `submit()` waits bounded (10 s) + checks status, reports the dispatch trace (`MINFER_TRACE=1`) and exits — never blocks forever; no early return past a `threadgroup_barrier` in a kernel (GPU deadlock/freeze); device limits (threadgroup memory/threads) queried at runtime, never hardcoded; all guard failures `gpu_abort` with actual values — no silent CPU fallback.

## Design Notes

- KV head dimension is independent: `n_kv_embd` read from K weight's ne[1] (Qwen2.5-0.5B: n_embd=896, n_head=14, hd=64, n_kv_embd=128); `gqa_attn` takes separate `hd_kv`/`nkt` for correct strides.
- Historical bugs (GQA simd_max divergence, Q5_K formula/qh indexing, Q4_K nibble layout, KV cache #6, IQ4_NL, …) are all FIXED and documented in `docs/` (`BUG-6-*`, `QWEN2.5-*-BUGS`, `DEBUGGING-*`, `multi-token-kernel-analysis.md`) — don't re-diagnose them.

## Core Conventions

1. Activations quantized to Q8_0 on-the-fly — all CPU matmuls use Q8_0 quantized activations (Q5_0 → `dot_q5_0_q8_0()`, Q4_K → `dot_q4_k_q8_0()`). Metal reads f32 activations directly for all weight types (Q4_0 included since P1), matching llama.cpp's Metal backend.
2. AVX2 dispatch: `is_x86_feature_detected!("avx2")` + scalar fallback (ARM Mac always scalar).
3. No ML frameworks — Attention, RMSNorm, RoPE, SiLU, Softmax all handwritten.
4. Tensor data uses raw `&[u8]` — avx2.rs dot products operate on byte slices, not structs.
5. GGUF padding: `ggml_pad()`: `(x + n - 1) & !(n - 1)`.
6. Metal per-layer fallback: when `layer_gpu` fails (e.g. Raw weights), submit partial GPU work, download hidden state, continue on CPU; only genuine support limitations fall back.

## Adding a New Architecture

1. Create `models/<name>/` with `mod.rs`, `forward.rs`, `loader.rs`
2. Add dispatch branch in `models/mod.rs::load_model()`
3. Define `HParams` (including `n_kv_embd`) and `LayerWeights` in `loader.rs`
4. Implement forward pass in `forward.rs`
5. Implement `ModelDef` trait in `mod.rs`
6. Add template format support in `template.rs` if needed

## Sampling

`sampler.rs`: repetition penalty → top-k → top-p → temperature, seeded `StdRng`. Defaults match llama.cpp: `temp=0.8`, `top_p=0.95`, `repeat_penalty=1.1`. CLI: `--temp`, `--greedy` (temp=0), `--top-k`, `--top-p`, `--repeat-penalty`, `-n/--n-predict`, `--seed`. Repetition penalty applies to the last 64 tokens (llama `repeat_last_n`): positive logits ÷ penalty, negative ×; alone fixes the 0.5B greedy repetition loops.

## Dependencies

Only 5 external crates: `rand` (sampling), `regex` (BPE pre-tokenization), `half` (fp16), `serde+serde_json` (download API), `minijinja` (template rendering).

## Documentation Locations

All non-root documentation lives in **`docs/`** (the project root only keeps `AGENTS.md` and `README.md`).

| Topic | Location |
|---|---|
| Metal backend optimization plans/gap analysis (primary tracking doc) | `docs/METAL_OPTIMIZATIONS.md` |
| GPU safety conventions + audit | `docs/GPU_SAFETY.md` |
| CPU backend optimizations | `docs/CPU_OPTIMIZATIONS.md` |
| CUDA backend (draft) / problems | `docs/CUDA_OPTIMIZATION.md`, `docs/CUDA_PROBLEMS.md` |
| Parameter audit vs llama.cpp | `docs/PARAMETER_AUDIT.md` |
| KV cache indexing bug #6 | `docs/BUG-6-KV-CACHE-INDEXING.md` |
| Debugging plans / summaries | `docs/DEBUGGING-PLAN.md`, `docs/DEBUGGING-SUMMARY.md` |
| Qwen2.5-1.5B bugs / debugging notes | `docs/QWEN2.5-1.5B-BUGS.md`, `docs/QWEN2.5-DEBUGGING-NOTES.md` |
| Debug dump format reference | `docs/debug-dump.md` |
| Metal inference / multi-token kernel analyses | `docs/metal-inference-analysis.md`, `docs/multi-token-kernel-analysis.md` |

## Code Search (ccc)

minfer is a llama.cpp-inspired engine; both codebases are indexed with [ccc](https://cocoindex.io/cocoindex-code/). When the question is "where / how is X implemented", prefer ccc semantic search over whole-repo grep/rg.

- **minfer's own code** -> use the `ccc` MCP `search` tool, or CLI: `ccc search "<query>"` from this directory (index: `.cocoindex_code/`, 1255 chunks).
- **llama.cpp reference source** (`$HOME/git/reading/llama.cpp`, C++ upstream: ggml, quantization kernels, sampler, tokenizer, llama_server, etc.) -> use the `ccc-llamacpp` MCP `search` tool, or CLI: `cd $HOME/git/reading/llama.cpp && ccc search "<query>"` (index: 47326 chunks).
- **Structural search**: `ccc grep '<pattern>'` -- e.g. `ccc grep 'fn \NAME(\(A*\))' src/` for Rust functions, or `ccc grep '\NAME(\(A*\))'` inside llama.cpp for call sites.
- **Filters**: `--lang rust|cpp|...`, `--path 'src/**'` to narrow results.
- Both MCP servers auto-refresh their indexes (`refresh_index=true` default); stale results -> pass `refresh_index=true` explicitly or run `ccc index` / `ccc search --refresh`.
