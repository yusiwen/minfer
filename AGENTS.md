# minfer — AI Agent Context

## Project Overview

minfer is a pure Rust LLM inference engine written from scratch (~4400 LOC), inspired by llama.cpp with 0 ML framework dependencies.  
Supports **Qwen2/Qwen2.5 architecture**, **CPU + Apple MPS (Metal) GPU** inference, and **GGUF v3 format**.

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

# Debug dumps (per-layer hidden states)
cargo build --release --features debug_dump

# Run Q4_0 (GPU on Apple Silicon, CPU on other platforms)
./target/release/minfer ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf "hello"

# Run Q4_K_M (GPU on Apple Silicon when available)
./target/release/minfer ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf "hello"

# Skip chat template (raw prompt)
./target/release/minfer --no-template <model> "hello"

# Debug dump
MINFER_DUMP_DIR=/tmp/dump target/release/minfer --features debug_dump <model> "hello"

# Force CPU (disable MPS)
MINFER_DISABLE_MPS=1 target/release/minfer <model> "hello"

# Info (list tensor names/types/shapes)
./target/release/minfer info <model>

# Auto-download — quant auto-matches single file or split (case-insensitive)
./target/release/minfer download hf Qwen/Qwen2.5-0.5B-Instruct-GGUF Q4_0        # single file
./target/release/minfer download hf Qwen/Qwen2.5-7B-Instruct-GGUF q4_k_m        # → both -00001/-00002-of-00002 parts
./target/release/minfer download hf:Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M        # URI form (case-insensitive)
./target/release/minfer download ollama qwen2.5:0.5b                            # ollama (tag = variant)
./target/release/minfer download ollama:qwen2.5:0.5b                            # ollama URI form
```

> **Multi-part (split) GGUF support** (2026-08-03, aligned with llama.cpp): a
> split model is `name-0000X-of-0000Y.gguf` (Y parts). Part 0 holds the model
> metadata (architecture/hparams) + its own tensors; each part is a standalone
> GGUF that lists the tensors it holds (offsets relative to its own data
> section). `GgufContext`-level `GgufModel`/`load_gguf_model` (gguf.rs) parse
> every part and validate `split.no == 0` (must load the first split); the qwen2
> loader builds a **merged tensor index** across parts and reads each tensor from
> its own part's data. The model entry is part 0 (`minfer
> qwen2.5-7b-instruct-q4_k_m 'hi'` — cached-name resolution dedupes split parts
> to part 0). `get_i64` handles Uint16/Int16/Uint8/Int8/Bool (llama's
> `split.count`/`split.no` are Uint16). Download resume is size-checked
> (expected size from the HF API, or a HEAD Content-Length fallback for repos
> that omit `size`; a partial part is resumed via curl `-C -`, never skipped).
> Quant matching is case-insensitive and auto-detects single vs split.
> Verified: 7B Q4_K_M loads as 2 parts (4466 MB) with correct GPU output.

## Debug Dump Files (`MINFER_DUMP_DIR`)

| File | Shape | Description |
|------|-------|-------------|
| `minfer_dump_embed_out.f32` | (nt, ne) | Token embedding output |
| `minfer_dump_layer{N}_out.f32` | (nt, ne) | Hidden state after layer N |
| `minfer_dump_layer{N}_attn_out.f32` | (nt, ne) | Hidden after attention + residual |
| `minfer_dump_layer0_bn.f32` | (nt, ne) | RMSNorm (attn_norm) output (layer 0 only) |
| `minfer_dump_layer0_bq.f32` | (nt, nqt) | Q projection output, pre-RoPE (layer 0 only) |
| `minfer_dump_layer0_bq_rope.f32` | (nt, nqt) | Q projection after RoPE (layer 0 only) |
| `minfer_dump_layer0_ba.f32` | (nt, ne) | Attention output before O_proj (layer 0 only) |
| `minfer_dump_layer0_bg.f32` | (nt, nf) | FFN gate output before SiLU (layer 0 only) |
| `minfer_dump_layer0_swiglu.f32` | (nt, nf) | SwiGLU output (layer 0 only) |
| `minfer_dump_layer0_fd.f32` | (nt, ne) | FFN down projection output (layer 0 only) |
| `minfer_dump_last_norm.f32` | (nt, ne) | Final RMSNorm (output_norm) output |
| `minfer_dump_logits.f32` | (nt, nv) | Final logits |
| `minfer_dump_prompt.txt` | — | Rendered prompt text |

Gen0 suffix (`_gen0.f32`) = first autoregressive generation step (single token).

## Quantization Support

### Working (verified)
- **Q4_0** — standard 4-bit, all weights, CPU + GPU → **works correctly**
- **Q4_1** — 4-bit with min
- **Q8_0** — 8-bit
- **Q4_K** — 4-bit K-quant super-block (nibble layout fixed)
- **Q6_K** — 6-bit K-quant
- **Q5_0** — 5-bit, CPU path works correctly, Q5_0×Q8_0 dot product implemented
- **Q5_K** — 5-bit K-quant, nibble + qh layout fixed, unsigned formula verified (Q5_K_M works CPU + Metal GPU)
- **Q5_1** — 5-bit with min (Q5_K_M model), CPU + Metal GPU verified (j+16 qh indexing matches llama-ARM)

### Not supported (CLI)
- Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL, etc.

## GPU Support Matrix

| Quant | MPS (Metal) | CPU |
|-------|-------------|-----|
| Q4_0, Q4_1 | ✓ (Q8_0-activation path + f32 path) | ✓ |
| Q4_K, Q6_K | ✓ (f32 path) | ✓ |
| Q8_0 | ✓ (f32 path) | ✓ |
| **Q5_0** | ✓ (f32 path, qh unaligned-read bug fixed) | ✓ |
| **Q5_1** | ✓ (F32 path) | ✓ |
| **Q5_K** | ✓ (f32 path, `kernel_q5_k_f32_matmul` + `_multi`) | ✓ |
| F32 | ✓ (RMSNorm, biases, etc.) | ✓ |

> **Prefill GEMM is Q4_0-only** (2026-08-01): `kernel_q4_0_mm_f32` uses the
> simdgroup GEMM for nt ≥ 16; all other quants (Q4_K, Q5_0, Q5_1, Q8_0, Q6_K,
> Q5_K) use the scalar f32 multi kernel. This is why Q4_K_M/Q5_K_M prefill is
> ~240 t/s vs Q4_0's ~554 t/s and llama.cpp's ~1750 t/s. See
> METAL_OPTIMIZATIONS.md for the full gap analysis and the P1 (non-Q4_0 GEMM)
> plan (the decode-dispatch plan was replaced by the shipped fused-matmuls +
> split-attention work).

> **P1 SHIPPED: non-Q4_0 simdgroup GEMMs** (2026-08-03): `kernel_q8_0_mm_f32`,
> `kernel_q5_0_mm_f32`, `kernel_q5_1_mm_f32` (32-elem blocks — drop-in copies of
> the Q4_0 GEMM with a per-quant `dequant_*_16` into the f16 sa tile), and the
> 256-elem super-block GEMMs `kernel_q4_k_mm_f32`, `kernel_q5_k_mm_f32`,
> `kernel_q6_k_mm_f32` (dequant il = (loop_k%256)/16 + il0 spans the 2 il-halves
> per 32-elem K step). All dispatched for nt ≥ 16 (MINFER_GEMM=0 disables) with
> the 8 KB threadgroup-memory guard. Faithful llama.cpp transcriptions (llama
> block_q6_K stores d at the END; block_q5_K = qh BEFORE qs — both match
> minfer). Verified: `tests/gemm_isolation.rs` `non_q4_0_gemm_isolation`
> (deterministic + relative error <5e-3 vs CPU refs for all 6 + Q4_0; the Q6_K
> CPU ref initially only passed trivially with d=0 — fixed to match avx2.rs
> indexing), end-to-end A/B byte-identical vs MINFER_GEMM=0, all models correct.
> Q4_K_M prefill ~300 → **~650-1000 t/s**, Q5_K_M ~330 → **~610-680 t/s**;
> **Qwen2.5-1.5B Q4_K_M prefill 48 → 442 t/s (~9×)** (its weights are Q4_K/Q6_K,
> so it exercises the Q4_K GEMM end-to-end). Decode unchanged (GEMM is
> prefill-only).

> **KV cache type** (2026-08-01): GPU KV cache defaults to **F32**;
> `MINFER_CACHE_TYPE=f16` switches to an F16 cache (2 bytes/elem, llama.cpp's
> default, with `kernel_store_kv_f16` + `kernel_gqa_attn_f16`). Measured F16 is
> ~3% slower than F32 on the 0.5B model (decode is dispatch-latency-bound, not
> KV-bandwidth-bound), so F16 is opt-in for larger models / longer contexts
> where attention bandwidth matters.

> **Decode bottleneck is GPU execution, not encode** (2026-08-03, A1 dead-end):
> minfer dispatches ~484 kernels/forward, but measuring the A1 parallel-encoding
> prototype showed the **encode is only ~1 ms/step** (4 threads × 6 layers,
> `MINFER_TIMING`), NOT the ~24µs/kernel × 484 ≈ 11.6 ms the 2026-08-01 analysis
> claimed. The measured decode step is ~7 ms/token and does not scale with the
> encode split: parallel encoding into **4 command buffers REGRESSED** decode
> (120-token gen: 1.67/1.08/1.43 s, nondeterministic, vs serial 0.93 s stable)
> because each extra command buffer adds GPU launch overhead and the encode is
> already hidden behind the GPU. **A1 was reverted** (2026-08-03). The decode
> cost is dominated by GPU execution of the 24 small per-layer kernels
> (attention reads a growing KV), not by CPU-side encoding. Next target should
> be GPU-side: kernel fusion / fewer dispatches (e.g. fuse QKV proj into one
> kernel, larger threadgroups per dispatch), not parallel command buffers.

> **Fused QKV + FFN gate/up matmuls for decode** (2026-08-03, ~5% decode):
> Wq/Wk/Wv and ffn_gate/ffn_up are **row-major-concatenated at load**
> (`concat_rows` → `blk.{i}.attn_qkv` / `blk.{i}.ffn_gu`) when the weights share
> a quant type + input dim. For **nt==1 decode only**, layer_gpu runs ONE matmul
> per group (od = nqt+2·nkt and 2·nf) into a `buf_bqkv`/`buf_bgu`; the rope/
> store/swiglu read the q/k/v and gate/up sections via the metal crate's
> `set_buffer` byte offsets (per-token sections are only contiguous when nt==1,
> so prefill keeps 3 separate matmuls incl. the Q4_0 GEMM). GPU-safety guards:
> the concat buffer byte length must exactly equal rows·row_bytes (else
> `gpu_abort`), bqkv/bgu buffers must be large enough, and the path is gated on
> `nt==1` + equal weight types. Verified: fused output is **byte-identical** to
> the separate path (`MINFER_NO_FUSE_QKV=1` A/B diff), and `tests/gemm_isolation.rs`
> `qkv_row_concat_layout` locks in the row-major layout. Median 200-token decode:
> **1.63 → 1.55 s (~5 %)** at a clean GPU state (the 2026-08-03 "~24 %" figure was
> inflated by a GPU-state artifact during heavy crash/abort testing).

> **Further dispatch fusions are dead-ends** (2026-08-03): `store_kv_both`
> (K+V in one kernel) and `residual_rms_norm` (residual add + FFN RMSNorm in one
> kernel) were implemented, verified correct in isolation
> (`store_kv_both_isolation` / `residual_rms_norm_isolation`), and REVERTED —
> they save 2 dispatches/layer but measured **no gain** (1.79 vs 1.74 s). Decode
> for nt==1 is **weight-read-bound** (~7 ms/token ≈ 4× the ~250 MB model's
> memory floor at 200 GB/s), NOT dispatch-launch-bound; reducing tiny-kernel
> launches doesn't move it. The remaining lever is a more efficient nt==1
> matmul kernel (vectorized block reads / better weight reuse), or attention
> for very long contexts — not more fusions.

> **nt==1 matmul rewrite is ALSO a dead-end — kernels already hit memory
> bandwidth** (2026-08-03): batched command-buffer benchmarking (many matmuls in
> ONE cb, removing the per-cb launch+sync overhead that confounds single-dispatch
> isolation timings) shows the current `kernel_q4_0_f32_matmul` reaches
> **182–240 GB/s** (near the M4 Pro floor): fused GU 0.027 ms, vocab output
> 0.32 ms. A custom full-block matvec kernel was ~equal (GU +15 %, vocab −30 %)
> and the prefill GEMM is 3× SLOWER at nt==1 (its 32-token tile is wasted), so
> none were integrated. The ~7 ms/token decode is therefore NOT the matmul
> kernels — it is the growing-KV attention + ~120 small kernels/layer + the
> per-token command-buffer submit/sync. Next lever: profile the attention and
> small-kernel GPU time in the real decode (batched methodology), not more
> matmul rewrites. Also: single-dispatch isolation timing is UNRELIABLE for
> small kernels (a ~165 µs cb launch+sync floor dominates; always batch ≥ dozens
> of dispatches per command buffer before trusting a per-matmul time).

> **Decode profile: attention is the #1 bottleneck (~48 % and grows with KV)**
> (2026-08-03): subtractive measurement on the REAL decode (temporary
> `MINFER_SKIP_ATTN` / `MINFER_SKIP_MATMULS` / `MINFER_SKIP_SMALL` env gates
> that skip kernel groups for nt==1 only — gated on nt==1 so prefill fills the
> buffers with finite values first, else the downstream exp/sqrt hit the slow
> denormal path and skew the timing) gives, per token at avg KV≈140: **attention
> ~4.2 ms (48 %)**, matmuls ~2.75 ms (32 %), small kernels ~0.95 ms (11 %), base
> encode+sync ~0.75 ms (9 %). Attention scales with KV: 2.0 ms/token at KV≈70 →
> 5.2 ms/token at KV≈240. Root cause: for nt==1 the `kernel_gqa_attn_f32` grid is
> only (1, nk)=2 threadgroups (each 32×gqa=224 threads) that loop the KV tiles
> SEQUENTIALLY with threadgroup barriers — massive GPU underutilization, KV-read
> latency-bound. Next target: parallelize the attention KV loop across more
> threadgroups for nt==1 (split/flash attention with a combine pass), the classic
> flash-attention KV-parallel structure. GPU-safety: this is the riskiest rewrite
> yet (online-softmax + threadgroup barriers + cross-TG combine); must be built
> incrementally with isolation tests (deterministic vs a scalar reference at
> several nkv, incl. partial tiles) before touching the decode path.

> **KV-parallel split attention SHIPPED (~32 % decode)** (2026-08-03):
> `kernel_gqa_attn_partial_f32` + `kernel_gqa_attn_combine_f32`. Pass 1 grids
> (nt, nk, n_chunks) — each TG computes an online-softmax PARTIAL (mx, S, acc)
> for its KV chunk [c·cs, min(nkv,(c+1)·cs)) with the SAME tile/barrier/valid-head
> structure as the classic kernel (each TG loops only its chunk; empty chunks
> produce mx=-INF/S=0/acc=0). Pass 2 (grid (nt, nh)) merges the partials via the
> standard max/exp/l-sum (a pure elementwise kernel — no shared memory, no
> barriers; guarded m==-INF→zeros). Used for nt==1 decode; n_chunks default
> adaptive `clamp((max_pos+1+31)/32, 1, 16)` (`MINFER_ATTN_CHUNKS` overrides).
> **2026-08-10**: the original `/16..32` formula over-parallelized at long
> context (decode chunks=32 → ~5.0-5.7 ms/token at KV≈430; batched isolation
> 0.108 ms/layer at nkv=2510 vs 0.081@8/16, and nkv=4000 is best at chunks=16:
> 0.089 vs 0.127@8 / 0.112@32). The `/32..16` formula also matches short context
> (KV=35-128 → 2-4 chunks, flat) and is byte-identical. Also (2026-08-10)
> `sync_kv_to_cpu` removed from the two full-GPU-success paths in `forward.rs`
> (kept only in the `gpu_failed` branch): the CPU KVCache is only read by the CPU
> fallback loop and GPU-layer failure is deterministic (weight types), so the
> per-token O(nkv)/token GPU→CPU KV copy was pure drain — the "KV-growth" decode
> cost llama does not pay. Interleaved -n 512 greedy (gpu submit-wait): OLD
> 4.65-4.76 → NEW 4.50-4.55 ms/token (~0.2-0.25 ms, byte-identical f32+f16).
> GPU-safety: built first in
> `tests/gqa_attn_isolation.rs::gqa_attn_split_isolation` (deterministic + cos 1.0
> vs the scalar reference at nkv=1/30/33/65/100/240, n_chunks=1/2/4/8, empty
> chunks, nt=1/2), then A/B-verified byte-identical to the classic path
> (`MINFER_NO_SPLIT_ATTN=1`) before integration. Median 200-token decode:
> 1.56 → **1.06 s (~32 %)**.

> **F16 split attention SHIPPED** (2026-08-03): `kernel_gqa_attn_partial_f16`
> reads the half K/V cache and converts to f32 float4 when staging the
> threadgroup tiles; the combine pass is shared (partials are f32). The split is
> now used for BOTH cache types at nt==1 (`MINFER_CACHE_TYPE=f16` no longer falls
> back to the classic single-pass kernel). Verified in
> `gqa_attn_split_isolation` (f32 AND f16 variants, cos>0.999) + end-to-end A/B
> vs the f16 classic path. f16 200-token decode: 1.60 → **0.95 s** (f32 split
> 0.89 s) — closes the f16-vs-f32 gap for larger models / long context.

> **Attention float4 + adaptive chunks + KV geometric growth** (2026-08-03,
> ~15 % more decode; long-context decode 10.6 → ~8 ms/token at KV≈2510):
> (1) `kernel_gqa_attn_partial_f32` acc is now `float4 acc4[64]` (vectorized
> dot/V-accum/corr-scale + float4 tile loads); the scalar dynamic-indexed
> `float acc[256]` landed in per-thread local memory (a per-thread serial DRAM
> bottleneck that parallelism didn't fix). Caught a divergent-simd_sum write bug
> (per-lane d loop) — the reduction MUST be a uniform loop (all lanes step the
> same d together). (2) `n_chunks` is now adaptive: `clamp((max_pos+1+15)/16,
> 1, 32)` (more chunks for longer KV) — **updated 2026-08-10 to
> `clamp((max_pos+1+31)/32, 1, 16)`** (the 32-chunk cap over-parallelized long
> contexts; see the split-attention note above). (3) **`kv_ensure_layer` grows KV buffers
> GEOMETRICALLY (×2)** — it previously grew by exactly `(max_pos+1)*nkt*4`,
> reallocating a new MTLBuffer + copying the whole old KV on EVERY decode token
> (the CPU encode cost was O(n²) in context: 0.5 ms at KV≈140 → 4.2 ms at
> KV≈2510; now ~0.13 ms). ⚠️ During that change a typo cloned `kvec[il]` into
> `old_v` (V cache polluted with K data → Q4_K_M garbage) — NOT caught by the
> split-vs-classic A/B because both share the same corrupted KV; caught by
> checking the output against a known-good reference. Verified: isolation 5
> passed (nkv up to 4097), all models correct, short 200-token decode
> 1.06 → **0.88 s**, long-context (KV≈2510) decode ~8 ms/token.

> **Decode micro-opt P6/P7** (2026-08-03, ~10%): the element-wise kernels
> (`add_f32`/`mul_f32`/`silu_f32`/`swiglu_f32`/`add_bias_f32`) were vectorized to
> process 4 elements/thread (float4) with a scalar tail, and `kernel_rope_f32`
> parallelized to one thread per (dim, head, token) — the Q rope was previously
> 14 threads doing 32 dims serially (recomputes pow/cos/sin per dim but the
> parallelism hides transcendental latency). 200-token decode ~0.88 → **~0.80 s**
> at a clean GPU state (measurements are bimodal: fast ~0.80 s vs throttled
> ~1.15 s — GPU-state artifact, not code). All models + isolation 6 passed.

> **Decode bottleneck is the CPU sampler, not the GPU** (2026-08-06, llama.cpp
> A/B): same Qwen2.5-0.5B Q4_K_M model — llama.cpp `Generation:` **247.2 tok/s**
> vs minfer default-sampling ~80-93 tok/s. Isolating GPU decode with `--greedy`
> gives ~180-208 tok/s (**5.0-5.8 ms/token**) — the default sampler adds
> **~7.6 ms/token**. Root cause: `sampler.rs apply_top_k` does a full-vocab
> (151,936) sort + 607 KB copy and `apply_top_p` a full softmax + sort of
> 151,936 `(usize,f32)` (2.4 MB/token), serialized in the decode loop; llama.cpp's
> candidate-list chain (top_k `std::partial_sort` O(n·log k) → top_p over the ≤k
> survivors, verified in src/llama-sampler.cpp @ 88b47a755) is near-free.
> **FIX SHIPPED 2026-08-06**: top_k → `select_nth_unstable_by` (O(n), on a copy
> to preserve index→token mapping); top_p → softmax+sort only the ≤k survivors;
> temp skips masked logits; main.rs moves the logits Vec instead of a 607 KB
> copy. Default decode ~12.6-14.8 → **~5.5-6.5 ms/token** (~2×: 512 tokens
> 6.9 → ~3.5 s; the ~150-200 tok/s figure was the OLD blended caliber — since
> 2026-08-06 "Generated:" is pure decode like llama's "Generation:"); fixed-seed
> output **byte-identical** (7 sampler tests pass). Full measurements in
> METAL_OPTIMIZATIONS.md §3.2.

> **Decode gap: matmuls at ~130 GB/s (structural for nt==1), NOT launch
> overhead, NOT dequant-compute-bound** (2026-08-06, final model): greedy decode
> 5.4 ms/token = matmuls **~3.0 ms (~130 GB/s)** + attention 0.7 ms (flat —
> split attention works) + small element-wise ~0.5 ms + base infra ~0.7 ms. The
> real matmul sweep is **~392 MB/token** (Q5_0 173 + Q8_0 146 + Q6_K 43 + Q4_K
> 29 MB, parsed from GGUF) → ~1.96 ms floor; matmuls run at 63 % of bandwidth.
> **Phase 1 SHIPPED**: fused bias+RoPE+KV-store (7→1) saved 144 kernels but only
> ~0.27 ms/token (~5 %, byte-identical f32+f16) → tiny kernels cost ~2-3 µs
> each; the "launch-overhead / aggressive fusion" model was wrong. **Q5_0
> vectorized dot SHIPPED**: the Q5_0 matmul was the last scalar kernel; using
> the existing `block_q5_0_dot_y` gained only ~0.08 ms (~2.6 % matmul) → matmuls
> are NOT dequant-compute-bound; the ~1 ms to the bandwidth floor is structural
> (nt==1 small-grid launch/latency + occupancy), and dequant vectorization has
> diminishing returns. Remaining low-value opts (Q6_K/Q4_K vectorization, encode
> opt, small fusions) → realistic ceiling ~4.8-5.0 ms/token vs llama ~4.05.
> **llama per-op comparison (2026-08-06 #3)**: this llama version has no per-op
> timing (ggml_perf removed); graph = 822 nodes (~490-530 dispatches, comparable
> to minfer's ~436), Flash Attention fused, and the matmul kernels are line-for-
> line llama translations — so llama's advantage is NOT faster matmuls. **Fair
> A/B + decomposition (2026-08-06 #4)**: same-session interleaved → llama 3.51 vs
> minfer 5.16 ms/token (1.47×, ~68 %); `MINFER_TIMING=1` splits minfer's wall-clock
> into CPU-encode 0.13 ms + **GPU 4.3-4.6 ms** + download 0.08 ms + sampling 0.43 ms —
> the gap is 100 % GPU execution (per-dispatch 10.3 µs vs llama 6.2 µs), and the
> CPU-side "pack setBytes / parallel encode" ideas are ~worthless (encode is only
> 0.13 ms). No hidden matmul kernel lever exists. **#5 (2026-08-06)**: "structural"
> is an INFERENCE, not a proven architecture gap — VERIFIED: matmul kernel source
> is line-for-line identical, dispatch count comparable (~436 vs ~490-530),
> per-dispatch GPU time differs. **#6 (2026-08-06) — hypotheses CLOSED**:
> (1) dispatch params DISPROVEN — llama's N_R0/N_SG (Q5_0=4/2, Q8_0=2/4,
> Q6_K=2/2, Q4_K=2/2 + the Q8_0 special grid) match minfer exactly; (2) attention
> NOT the main lever — llama `-fa on` vs off = 3.64 vs 3.88 ms (~0.25 ms);
> (3) multi-cb NOT a lever — llama's multi-cb hides CPU encode (minfer encode is
> 0.13 ms), `MINFER_SPLIT_CB=N` re-test regresses linearly (0.67 → 0.93/1.23/1.62 s).
> Conclusion: the ~1 ms GPU gap is the per-kernel execution of ~436 serial kernels
> in one cb (small ops f32 vs llama f16 + MPS serialization) — genuine structural
> difference, not micro-optimizable. **#7 (2026-08-06) architecture plan**: matmul
> source+params match llama ⇒ gap is non-matmul kernels. Step 0 (done): clean
> per-category GPU decomposition (`MINFER_TIMING`+DecodeSkips, greedy) = matmul
> **2.99 ms (72 %)** + attention 0.54 + small 0.52 = 4.18 ms GPU/token — matmuls
> identical to llama, so the lever is attention fusion + small-op efficiency.
> xctrace limitation: the Metal System Trace CLI export doesn't give per-kernel
> durations (execution-points underestimate; shader profiler not captured) —
> aggregate GPU-work comparable (24.1 vs 24.0 ms), per-kernel needs the Xcode GUI.
> **Step 1 result (2026-08-06)**: naive 1-kernel/layer attention (classic
> `kernel_gqa_attn_f32`, `MINFER_NO_SPLIT_ATTN=1`) is SLOWER than split — 4.80 vs
> 4.15 ms GPU/token — confirming the split design is right; llama's flash is fast
> because of simdgroup_matrix, not "one kernel". Only a faithful
> `kernel_flash_attn_ext_impl` port (~600 lines, function constants) could give a
> fast 1-kernel attention, at multi-day/high-risk cost for ~0.3 ms. Net: no
> low-risk path remains to close the GPU gap in this architecture.
> **Phase A gate (2026-08-06) — flash port STOPPED**: f16 KV attention 0.54 vs
> 0.52 ms (f16 is the core of llama's flash advantage but minfer 0.5B attention
> isn't KV-read-bound), chunk tuning already optimal (adaptive 0.51 ms), attention
> scales sub-linearly with KV. minfer's split attention is at its design limit
> (~0.5 ms, 13 % of the 4.2 ms GPU); the llama flash port is a dead-end for this
> model. **Final gap report (2026-08-06, precise)**: minfer GPU 4.18 ms ≈
> matmul ~3.0 ms (identical source+params, ~130 GB/s) + **non-matmul ~1.2 ms**
> (the `no_matmul` config isolates it cleanly). llama's non-matmul ≈ 0.3 ms →
> **the structural gap (~0.9-1.0 ms) is 100 % in the ~340 non-matmul kernels**
> (attention + small f32 elementwise + single-cb serialization), at ~4× llama's
> efficiency — NOT the matmuls. Plus a KV-growth component (~0.5 ms/token at
> -n 512: minfer's AVERAGE decode grows 5.05 → 6.7 ms, llama stays ~flat).
> llama GPU is inferred (no per-op timing); no single fixable component without
> an architecture-level rewrite — minfer's Metal decode is at this
> architecture's practical limit.
> Profiling gates kept
> as `MINFER_SKIP_ATTN/MATMULS/SMALL=1` (decode-only), centralized in
> `metal.rs::DecodeSkips` (OnceLock-cached env read like MINFER_TRACE; each
> dispatch gated in its exact original position — the FFN down-matmul must stay
> AFTER swiglu). Full detail in METAL_OPTIMIZATIONS.md §2.3/§4.
> **minfer vs llama.cpp: full Metal inference path comparison (2026-08-10)**: the
> comparison section in METAL_OPTIMIZATIONS.md (§1.3/§1.4) gives both sides'
> kernel sequences (minfer 20/layer Q4_K_M / 12/layer Q4_0 vs llama 17/layer
> flash), per-category timings (matmul ~3.0 ms zero gap; non-matmul minfer
> ~1.2 ms vs llama ~0.3 ms = 4×), and a kernel inventory table. llama per-op
> timings are inferred (that version has no ggml_perf per-op timing); minfer
> numbers are measured via MINFER_TIMING + skip-gate subtraction.
> **Same-model, same-parameter A/B (2026-08-11)**: decode is now 72-88 % of
> llama (Q4_K_M 218 vs 293-299 t/s, Q4_0 279 vs 314-339 t/s pure GPU; default
> sampling 197 vs 247 t/s). Prefill remains 2.8-3.6× (llama 6909 vs minfer 2466
> t/s at pp430) — now mostly matmuls + small kernels at the architecture limit,
> not attention (see METAL_OPTIMIZATIONS.md §1.1).

> **Per-kernel non-matmul profile + 256-thread RMSNorm (2026-08-10, P0/P1)**:
> `metal.rs::tests::non_matmul_bandwidth_profile` (batched-cb per-kernel timing,
> median of 3) showed the 32-thread `kernel_rms_norm_f32` costs ~13.8 µs/dispatch
> vs ~2.2 µs for the 256-thread elementwise kernels (add/swiglu/rope) doing the
> same traffic — a single simdgroup cannot hide DRAM latency for one 896-element
> row (the Phase 4 "32 threads already saturate" claim was a different-kernel
> comparison). New `kernel_rms_norm_f32_256` (llama transcription: 256-thread
> threadgroup, per-simdgroup partial sums reduced through a threadgroup buffer
> with 2 barriers) is **bit-identical** (maxdiff 0.0, `rms_norm_256_correctness`
> test) and ~3.7× faster isolated (3.7 vs 13.8 µs); integrated into the decode
> path (all 3 rms_norm sites) gated by `MINFER_NO_RMS_256=1` for A/B. Interleaved
> decode: gpu submit-wait ~4.25-4.32 → ~4.40-4.49 ms/token (~0.1-0.2 ms, ~3-4 %),
> wall-clock 203-207 → 206-212 t/s. Byte-identical f32+f16, Q4_K_M+Q4_0. The same
> profile shows the **attention split pair (partial+combine) is the dominant
> non-matmul kernel** (~44 µs/layer at nkv=430, and the classic single-pass is
> ~352 µs — 8× worse), confirming the split design and that flash-port remains
> the only attention lever.
> **7B decode "28-31 t/s" was a blended-caliber artifact, NOT a regression**
> (2026-08-11, P0): a suspicious 7B slowdown (12 vs 45 t/s) bisected cleanly back
> to... nothing — the old binaries label the BLENDED rate as `Generated:` (pre-
> `dc66d0d` caliber change), and 7B hits EOS after ~9 tokens so short generations
> make the difference huge. Same-caliber A/B (parent dfd7866 vs current a7f21e4):
> steady-state gpu submit-wait 56.2 vs 54.4 ms/token, wall 10.8-14.1 vs
> 14.2-14.5 t/s — current is marginally FASTER. The real 7B pure-decode rate is
> ~14-15 t/s (~55 ms/token GPU); docs' historical "28-31 t/s" was blended.
> Lesson: when A/B'ing binaries across the `dc66d0d` caliber boundary, compare
> **MINFER_TIMING steady-state gpu submit-wait** (tok 5+), never the `Generated:`
> line of pre-caliber-fix builds.

> **Parallel prefill attention (P1, 2026-08-11) — pp430 ~32 % faster**:
> with llama-Metal rebuilt (GGML_METAL=ON), the real prefill gap was localized:
> minfer pp430 ~1812 t/s vs llama ~6939 (3.8×), and the classic `kernel_gqa_attn_f32`
> (grid (nt,nk), sequential KV loop with ~24K barriers at nt=430) was ~100 ms
> (48 % of prefill, ~25× llama's attention). Replaced with a 3-pass parallel
> attention for nt>1 (all barrier-free): `kernel_attn_scores` (one 256-thread TG
> per (t,h) row) → `kernel_softmax_attn` (masked softmax over kv per row) →
> `kernel_attn_output` (softmax·V). GQA via per-head hk=h/gqa indexing (the
> broadcast-GEMM idea was abandoned — a 2D GEMM can't produce the per-head
> 3D scores tensor). ⚠️ threadgroup-memory bug: the 256-thread softmax's
> `shmem[tiisg]` init writes 32 floats but only 8 were allocated (8·4=32 B) —
> OOB into adjacent threadgroup memory corrupted the max/sum reductions → NaN
> rows; fixed by allocating 32·4=128 B (same latent bug fixed in rms_norm_256).
> Verified: `attn_parallel_prefill_correctness` (maxerr 0.0 vs CPU),
> `attn_parallel_realdata_correctness` (real layer-0 activations, maxerr 0.0),
> end-to-end byte-identical (f32+f16, greedy+sampled), 34 bin + 6 isolation tests.
> Measured (prefill GPU): pp430 classic 212 → **144 ms (~32 %)**, attention
> 100 → **30 ms**; pp30 44 → 40 ms (~9 %). Gated by `MINFER_NO_MATMUL_ATTN=1`
> (default = parallel). Remaining pp430 gap to llama (62 ms) is matmuls + small
> kernels (at their documented architecture limit), not attention.
> **7B verified + prefill GEMM ceiling (2026-08-11 follow-up)**: parallel prefill
> attn + rms_256 are correct at 7B (byte-identical, 34 bin + 6 isolation pass);
> 7B pp230 prefill 944 → **832 ms (~12 %)**, parallel attention 169 → **57 ms
> (~3×)**. `prefill_gemm_throughput_profile` measures minfer's prefill GEMM
> ceiling at ~**5.4 TFLOPs/s** (Q5_0 dequant-bound, grid-shape variance 3.5-5.4)
> vs llama's ~7 TFLOPs/s effective — a ~30 % GEMM execution-efficiency gap, the
> same "structural" class as the decode finding. llama per-op timing is NOT
> obtainable via CLI (`GGML_METAL_CAPTURE_COMPUTE` needs Xcode; per-kernel shader
> intervals are not recorded by the Metal System Trace on this setup — see
> METAL_OPTIMIZATIONS.md §4.1, which replaces the old "Xcode-GUI-only" claim with
> the correct xctrace path + Performance Limiters workflow). See §3.4/§4.
> **Optimization-plan status (2026-08-12, trace DONE 2026-08-13, q6_K FIXED)**: the
> 2026-08-06 "accept the architecture floor" verdict is **REVOKED** — the goal is
> to match llama.cpp performance. METAL_OPTIMIZATIONS.md §0 progress table is the
> single tracking source; §4 is the only action path: (1) GPU trace DONE —
> `scripts/export_trace.sh` + Performance Limiters counter set gives BOTH
> per-kernel durations (metal-shader-profiler-intervals) and limiter profile;
> the trace per-kernel table was later SUPERSEDED by a clean isolation A/B
> (llama test-backend-ops perf vs minfer matmul_bandwidth_profile) — the early
> "1.6-3.9×" numbers were trace-semantics artifacts (fused-vs-separate, mixed
> od); limiter: prefill = under-occupied (no HW limit), decode =
> cache/memory-bound on BOTH sides. (2) decode matmul per-call: the real gap
> was **q6_K ffn_down** (72 vs 217 GB/s, stride-64 loop ~30% TG utilization) —
> **FIXED 2026-08-13** (ported llama's stride-2/float4 layout): q6_K 209 GB/s,
> decode 4.27→3.72 ms/tok (~13%), byte-identical, tests green. q5_0/q8_0 at
> parity; q4_K only if a model uses it, (3) flash-attention port (attention
> ~7-10× isolation-confirmed 2026-08-13: minfer split 42.8 µs/layer vs llama
> flash ~4-6 µs/layer at nkv=430; structural cause = simd_shuffle_down vs
> threadgroup barriers; port decision pending KV-layout pre-check — see METAL
> §4.2.2; llama build env has a pre-existing ObjC/SDK issue recorded there),
> (4) prefill GEMM execution efficiency toward ~7 TFLOPs/s (grid-shape
> probe first), (5) 7B same-model A/B + per-step regression check. Dropped:
> 2D-simdgroup GEMM (llama disables tensor GEMM on M4 Pro per PARAMETER_AUDIT A)
> + bf16 staging (llama reads f32 activations per Core convention #1).

> **Performance-verification methodology** (2026-08-06, after the Q5_0
> shader-compile-bug was misread as a GPU throttle): (0) **tok/s caliber**: since
> 2026-08-06 minfer's `Generated:` is **pure decode** (`generated / gen_time`,
> aligned with llama.cpp's `Generation:`); `Total:` keeps the blended rate
> (`prompt+gen / total_time`). Historical `Generated:` numbers (pre-fix) were
> blended. (1) `metal.rs::tests::metal_pipelines_compile`
> compiles every Metal pipeline at `cargo test` — `cargo build` does NOT compile
> shaders, so a duplicate/missing kernel only fails at model load (MPS falls back
> to CPU silently; the ~7 tok/s CPU time reads like "GPU throttling"). (2)
> `scripts/bench.sh <args>` runs minfer and refuses (non-zero exit) unless `MPS:
> GPU acceleration enabled` is in the output; `scripts/bench.sh --health <model>`
> additionally requires prefill ≥ 200 tok/s (healthy ~500+, CPU fallback ~7).
> Benchmark decode only after a passing `--health`. Beware: sustained GPU
> benchmarking thermally throttles the M4 Pro (extreme case: everything ~1.3 s
> regardless of config) — interleave configs, take min/median, use few runs.
> Also: **batched-cb per-kernel timing has a cold-start/GPU-clock-ramp artifact**
> (the first timed kernel measured ~4× slow; same kernel later: 23 → 107 GB/s) —
> always warm each kernel and measure twice, and treat isolated kernel GB/s as
> relative, never absolute (`metal.rs::tests::matmul_bandwidth_profile` does
> this).

> **GPU nondeterminism was a state artifact** (2026-08-03): during heavy
> crash/abort testing (Metal encoder asserts, timeout kills) the GPU entered a
> bad state producing intermittent wrong logits (different greedy outputs per
> run, and a transient 8 tok/s prefill). After a clean restart everything is
> deterministic (6-10/6-10 identical) and prefill recovers to ~500 tok/s. Not a
> code bug — the Metal runtime/GPU needs a clean process after abort crashes.

> **Metal cb/encoder autorelease fix** (2026-08-03, from the A1 prototype): the
> `metal` crate returns **autoreleased** ObjC objects from `commandBuffer` /
> `newComputeCommandEncoder` (not `new`). On a background thread, the thread's
> autorelease pool drains at thread exit → the encoder is released without
> `endEncoding` → Metal asserts ("Command encoder released without endEncoding")
> and the process aborts. `cmd_buffer()` now explicitly `retain`s both objects
> and `MpsCommandBuffer::drop` releases them, so a command buffer created on
> any thread survives the autorelease-pool drain. Harmless for the serial path;
> required for any future multi-command-buffer/threaded encoding.

> **GPU-safety measures** (2026-08-02, after an M4 Pro GPU hang froze the
> machine): (1) `MpsCommandBuffer::submit()` waits **bounded 10 s** and checks
> `MTLCommandBufferStatus`; a GPU fault/hang reports the dispatch trace
> (`MINFER_TRACE=1`, last 16 op labels) and exits instead of blocking forever.
> (2) `kernel_gqa_attn_f32/f16` never return early for `h >= nh` before the
> threadgroup barrier (would deadlock the GPU when `nh % nk != 0`). (3)
> `layer_gpu`/`output_norm_gpu` assert `nh % nk == 0`, `hd ≤ 256`, `id % 32 == 0`
> and fall back to CPU otherwise. Full details + audit status in
> **[`GPU_SAFETY.md`](GPU_SAFETY.md)**.

## GPU Safety Conventions

1. **Device thresholds MUST be queried at runtime, never guessed** (2026-08-02,
   after the GPU-hang review): device-specific limits such as
   `max_threadgroup_memory_length()` and `max_threads_per_threadgroup()` are
   queried via the `metal` crate's `MTLDevice` properties and cached at MpsState
   init. Do NOT hardcode guessed limits (e.g. a remembered "32 KB threadgroup
   memory") — verify via runtime query. See `GPU_SAFETY.md` §4.
2. **All GPU safety guards error-exit** — never silently fall back to
   CPU. A guard that detects an unsafe/unsupported configuration (dimension
   misalignment, head-dispatch mismatch, threadgroup memory/threads exceeding the
   queried device limit, kernel-array overflow) prints a clear message with the
   actual values via `gpu_abort` and exits, so the user knows the GPU path can't
   run the model instead of silently running slow on CPU. Only genuine support
   limitations (Raw weights, unregistered weights, mixed-quant attention/FFN)
   fall back to CPU.
3. **Never return before a `threadgroup_barrier`** in a kernel: a per-simdgroup
   early return can leave other simdgroups waiting on the barrier → GPU
   permanent deadlock (machine freeze). Skip computation with a mask/flag
   instead, keeping all simdgroups alive through every barrier.
4. **Fixed-size kernel arrays must be guarded by a dimension check that
   error-exits** (e.g. `float acc[256]` needs `hd ≤ 256`).
5. **All GPU kernel dimension assumptions are asserted in `layer_gpu` /
   `output_norm_gpu`** (`nh % nk == 0`, `id % 32 == 0`, quant-alignment) — on
   violation, error-exit rather than risk a GPU fault or silently run on CPU.

## Model Support Matrix

| Model | CPU | MPS GPU | Notes |
|-------|-----|---------|-------|
| Q4_0 (qwen2.5-0.5b-instruct-q4_0) | ✓ | ✓ (361 tok/s) | All weights Q4_0 |
| Q4_K_M (qwen2.5-0.5b-instruct-q4_k_m) | ✓ (3.2s) | ✓ (226 tok/s) | Q5_0/Q8_0/Q4_K/Q6_K mixed |
| Q5_K_M (qwen2.5-0.5b-instruct-q5_k_m) | ✓ | ✓ (~250 tok/s) | Q5_1/Q8_0/Q5_K/Q6_K, formula + qh indexing FIXED, Q5_K Metal kernel added — full GPU |

## Known Issues

### GQA Attention `simd_max` divergence (fixed 2026-08-01)
`kernel_gqa_attn_f32` used `for (int j = tiisg; j < tile_sz; j += 32)` — when a
KV tile has `tile_sz < 32` (i.e. `nkv % 32 != 0`, which is nearly every prefill
token and every decode step), the lanes with `j >= tile_sz` **exit the loop
early**, so `simd_max(dot)` runs across divergent lanes and includes stale
register values from the exited lanes. The online-softmax running max is then
corrupted, producing wrong attention outputs for tokens whose KV count spans a
partial second tile (e.g. token 36 of a 37-token prefill had cos ≈ 0.97 vs CPU;
the whole GPU decode then diverged → repetition loops).

Symptom: GPU output loops (repeats a sentence forever) while the CPU path
(which masks invalid KV positions) generates coherent text. Q/K/V/cache and the
matmul kernels were all verified correct — the error was isolated to attention
via per-stage GPU/CPU dump comparisons.

**Fix**: all 32 lanes execute the same iteration count (`for j0 in 0..tile_sz
step 32`), with a `valid = j < tile_sz` mask; invalid lanes use `dot = -INFINITY`
(so `simd_max` ignores them) and contribute `e = 0`. Verified with real model
data: prefill last-token logits cos vs CPU went 0.83 → 0.999, gen0 0.96 → 0.999,
and the looping prompt now generates coherent text matching llama.cpp.
Regression test: `tests/gqa_attn_isolation.rs` (multiple nkv incl. partial tiles).

### Q4_K / Q5_K Nibble Layout (fixed)
Q4_K and Q5_K store `qs[128]` with cross-subblock nibble packing. Each byte's lo nibble belongs to subblock 2k and hi nibble to subblock 2k+1, aligned by element index. minfer originally assumed Q4_0-style layout (lo=sub_elem[j], hi=sub_elem[j+16]).

**Files affected and fixed:**
- `avx2.rs`: `dot_q4_k_q8_0_scalar`, `dot_q5_k_q8_0_scalar` — deinterleave before dot product
- `kernel.rs`: `cpu_q5_k_matmul_f32` — deinterleave before dequant
- `forward.rs`: Q4_K embed, Q5_K embed — deinterleave before dequant
- `metal.metal`: `kernel_q4_k_f32_matmul`, `kernel_q4_k_f32_matmul_multi` — deinterleave before dot product

### Q5_0 Metal GPU Kernel (fixed)
The Q5_0 Metal shader kernel is implemented in `metal.metal` (`block_q5_0_dot_y`, `kernel_q5_0_f32_matmul`, `kernel_q5_0_f32_matmul_multi`, `kernel_q5_0_debug`).

- **Root cause found & fixed**: `*(uint32_t *)(block + 2)` reads qh at a 2-byte aligned address — undefined behavior on ARM/Metal. Fixed by reading 4 individual bytes and combining. Verified via `kernel_q5_0_debug` (single-block dequant matches CPU bit-for-bit).
- **Verified working**: CPU vs GPU per-layer comparison shows all 24 layers with cos ≥ 0.998. Output matches CPU ("Hello! How can I assist you today?").

### Q5_K formula + qh indexing bugs — FIXED (2026-07-31)
**Q5_K_M garbled output had TWO root causes in minfer's Q5_K:**

**Bug 1 (formula)**: minfer used the Q5_0-style **signed** formula `dl*(u-16)-ml`. llama.cpp's Q5_K (both CPU `dequantize_row_q5_K` and Metal `dequantize_q5_K`, unchanged since Oct 2023) uses **unsigned** `dl*u - ml`.

**Bug 2 (qh high-bit indexing)**: minfer read the 5th bit as `qh[sub*4 + pos/8] >> (pos%8)`. The correct layout (from `quantize_row_q5_K_impl`, llama-quants.c:1829-1844) is **`qh[pos]` bit `sub`** — qh byte index = element position within the 32-elem subblock, bit index = subblock number (0-7).

**Evidence** (real `blk.2.ffn_down.weight` dequant):
- unsigned formula: mean≈0, symmetric ✓ vs signed-16: mean=-0.008 ✗
- correct qh indexing on llama's own swiglu: ffn_down cos=**0.99999902** vs llama; minfer's wrong indexing: 0.994

**Fixed in 3 files × 2 bugs**:
- `avx2.rs:dot_q5_k_q8_0_scalar` — formula + `(qh[k] >> s) & 1`
- `forward.rs:embed_tokens` Q5_K branch — formula + `(qh[j] >> sub) & 1`
- `kernel.rs:cpu_q5_k_matmul_f32` — formula + `(qh[j] >> sub) & 1`, `(qh[j+16] >> sub) & 1`

**MPS GPU kernel** (added for full GPU support): `metal.metal` `kernel_q5_k_f32_matmul` + `_multi` (176 B/256-elem superblock, `get_scale_min_k4` scales, `qh[p] bit s` high bits, unsigned formula). Dispatch in `metal.rs` + MPS registration in `loader.rs`. Must be placed AFTER `get_scale_min_k4` in metal.metal (Metal requires declaration before use).

**Why earlier verification missed it**: `scripts/verify_q5k_gate_up.py` reimplemented minfer's OWN (wrong) formula AND qh indexing, so "cos=1.0" only proved Python↔Rust self-consistency, not correctness vs llama.

**Result**: Q5_K_M per-layer now matches llama (blk.2 cos=0.99999894, layers 4-21 = 0.99999971, no collapse at 22-23). CPU and MPS both produce correct output.

### Q5_K_M — VERIFIED WORKING (2026-07-31)
- CPU: "Hello! How can I assist you today?" ✓
- **MPS GPU: full GPU support** ✓ — Q5_K Metal kernel (`kernel_q5_k_f32_matmul` + `_multi`) added, ~250 tok/s (prefill 282, gen 247). Per-layer GPU vs CPU cos ≥ 0.9999998 (layers 2-20), logits argmax matches llama.
- Q4_0 / Q4_K_M: no regression ✓
- llama.cpp dispatch for reference: Q5_1→Q8_1 activations (ARM `s` = fp16(d_q8·Σq8)), Q4_K/Q5_K/Q6_K→Q8_K, Q5_0/Q8_0→Q8_0. **These are the CPU backend** `vec_dot_type`s. The **Metal backend reads f32 activations directly for ALL types including Q4_0** (no Q8_0 quantization). Since P1 (2026-08-01) minfer's Metal Q4_0 also uses f32 activations, matching llama Metal. Q8_0 activations in minfer's CPU path match llama's CPU to ~1e-6.

### KV Head Dimension (n_kv_embd)
Qwen2.5 models may use separate KV head dimensions (e.g., Qwen2.5-0.5B: n_embd=896, n_head=14 → hd=64, n_kv_embd=128). The KV cache and attention now use `n_kv_embd` from `HParams` (read from K weight's ne[1]) instead of computing `n_head_kv * n_embd_head()`. The attention function `gqa_attn` accepts separate `hd_kv` and `nkt` parameters for correct stride calculation.

### Q4_0 Prefill GEMM (`kernel_q4_0_mm_f32`) — SHIPPED 2026-08-01
Faithful port of llama.cpp's `kernel_mul_mm_q4_0_f32` (legacy simdgroup path):
64×32 tile, 4 simdgroups × 32 threads, Q4_0 dequant staged into `sa` (transposed
A), f32 activations staged into `sb` via scalar stores (equivalent to llama's
float2x4), `simdgroup_half8x8` inputs → `simdgroup_float8x8` accumulators.
- **Q4_0 block layout**: GGUF byte j = {element j (lo), element j+16 (hi)} —
  NOT byte j/2 = {2j, 2j+1}. llama's `dequantize_q4_0` reads `qs` as uint16.
- **B-staging**: write to raw `(tiitg/NL1)/8` and `(tiitg/NL1)%8` sb positions
  (unclamped, fills OOB rows with the clamped row's data); read activations
  from the clamped `lr1` row.
- **Store**: `transpose=false` is the ROW-major store. Direct:
  `C + 8*(i/4)*od + 8*(i%4)`; bc_out (partial tiles): `temp_str + 8*(i%4) +
  8*NR0*(i/4)` with `temp_str = shmem + 32*(sgitg&1) + (16*(sgitg>>1))*NR0`
  (float, reuses sa/sb → smem 8192 B for bc_out, 6144 for full tiles).
- **Barrier**: the FIRST multiply-loop barrier must be `mem_threadgroup`
  (makes other simdgroups' sa/sb writes visible to `simdgroup_load`); the two
  later ones are `mem_none`. Using `mem_none` everywhere caused a
  non-deterministic race at od=4864 (gate/up) with nt%32 != 0.
- **Dispatch**: GEMM for nt ≥ 16 (f32 multi wins below that due to lower fixed
  overhead); `MINFER_GEMM=0` forces f32 multi. ~11% faster at 30 tokens,
  ~34% at 70 tokens vs the f32 multi kernel.
- **Isolation test**: `tests/gemm_isolation.rs` (macOS-only, needs GPU) checks
  determinism + correctness vs a scalar CPU reference at nt = 12/30/32/33.

### Q4_K AVX2 dot product — RESOLVED 2026-08-06 (was a stale test reference, not a bug)
`cargo test --bin minfer` previously showed 5 failing `test_q4k_dot_*` tests
(diff 39–167). **Root cause: the TESTS' reference implementations used the OLD
16-bytes-per-subblock Q4_K layout** (`reference_dot`, `independent_dot_q4k`),
not the real llama layout. The `dot_q4_k_q8_0` implementation itself was correct
(4-chunk deinterleave — chunk c covers subblocks 2c/2c+1, byte l → sub 2c elem l
lo / sub 2c+1 elem l hi, verified line-for-line vs llama `dequantize_row_q4_K`),
and there is NO AVX2 Q4_K path (scalar only). Both test references were fixed to
the correct layout → **all 29 bin tests pass (0 failures)**. x86 CPU users were
never affected. Full detail in the 2026-08-06 fix.

## Core Conventions

1. **Activations quantized to Q8_0 on-the-fly** — all CPU matmuls use `Q8_0` quantized activations. Q5_0 weights use `dot_q5_0_q8_0()`; Q4_K uses `dot_q4_k_q8_0()`. The **Metal backend reads f32 activations directly for all weight types** (Q4_0 included since P1), matching llama.cpp's Metal backend.
2. **AVX2 dispatch pattern**: all kernels use `is_x86_feature_detected!("avx2")` runtime detection + scalar fallback (ARM Mac always uses scalar).
3. **No ML frameworks** — Attention, RMSNorm, RoPE, SiLU, Softmax all handwritten loops.
4. **Tensor data uses raw `&[u8]` interface** — avx2.rs dot products operate on byte slices, not structs.
5. **GGUF padding rule**: `ggml_pad()`: `(x + n - 1) & !(n - 1)`.
6. **Metal GPU per-layer fallback**: when `layer_gpu` fails (e.g., unsupported weight type like `Raw`), the engine submits the partial GPU work, downloads the hidden state, and continues remaining layers on CPU. Q5_K is now GPU-supported.

## Adding a New Architecture

1. Create `models/<name>/` with `mod.rs`, `forward.rs`, `loader.rs`
2. Add dispatch branch in `models/mod.rs::load_model()`
3. Define `HParams` (including `n_kv_embd`) and `LayerWeights` in `loader.rs`
4. Implement forward pass in `forward.rs`
5. Implement `ModelDef` trait in `mod.rs`
6. Add template format support in `template.rs` if needed

## Sampling (2026-08-01)

`sampler.rs` implements a llama.cpp-style pipeline: repetition penalty →
top-k → top-p → temperature, with a seeded `StdRng` for reproducibility.
Defaults align with llama.cpp: `temp=0.8`, `top_p=0.95`, `repeat_penalty=1.1`
(1.0 = off). CLI flags override: `--temp`, `--greedy` (temp=0), `--top-k`,
`--top-p`, `--repeat-penalty`, `-n/--n-predict`, `--seed`.

The repetition penalty (`apply_repetition_penalty`) applies to the last 64
tokens (llama `repeat_last_n` default): positive logits divided by the penalty,
negative multiplied. It alone breaks the 0.5B model's greedy repetition loops
(e.g. "Tell me about Transformer architecture." previously hit the 512-token
cap repeating; with `repeat_penalty=1.1` it stops at EOS naturally even with
`--greedy`).

## Dependencies

Only 5 external crates: `rand` (sampling), `regex` (BPE pre-tokenization), `half` (fp16), `serde+serde_json` (download API), `minijinja` (template rendering).
