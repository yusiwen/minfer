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

# Auto-download
./target/release/minfer download hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_0.gguf
```

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
> / P3 (decode dispatch) plan.

> **KV cache type** (2026-08-01): GPU KV cache defaults to **F32**;
> `MINFER_CACHE_TYPE=f16` switches to an F16 cache (2 bytes/elem, llama.cpp's
> default, with `kernel_store_kv_f16` + `kernel_gqa_attn_f16`). Measured F16 is
> ~3% slower than F32 on the 0.5B model (decode is dispatch-latency-bound, not
> KV-bandwidth-bound), so F16 is opt-in for larger models / longer contexts
> where attention bandwidth matters.

> **Decode bottleneck is per-dispatch encode cost** (2026-08-01): minfer
> dispatches ~484 kernels/forward (layer_gpu, 20/layer × 24 + 3 + embed) — nearly
> identical to llama.cpp's ~490–530 actual Metal kernels (822 ggml graph nodes
> minus ~300 no-op views). The ~3× decode gap is ~24µs/kernel encode overhead
> (single command buffer, serial `set_buffer`+`set_bytes` per kernel) vs llama's
> ~7µs (multi-command-buffer parallel encoding, packed-params `setBytes`). See
> METAL_OPTIMIZATIONS.md "Decode Gap §0a". Next target: P3 (per-dispatch encode
> cost + parallel command buffers).

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
2. **All GPU safety guards error-exit (报错退出)** — never silently fall back to
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
