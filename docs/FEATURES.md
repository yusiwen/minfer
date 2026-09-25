# Features

This page expands the feature list from the [README](https://github.com/yusiwen/minfer) with the full detail. Performance numbers refer to Qwen2.5-7B-Instruct q4_K_m prefill on GB10 (sm_121) unless noted.

## Inference Core

### Declarative compute graph

Inference builds a `ComputeGraph` (pure IR) then assigns backends, fuses ops, allocates and executes via a scheduler — inspired by llama.cpp's `ggml_cgraph` + backend scheduler. Graph reuse is params-only (decode steps skip reconstruction), backend assignment is per-op, and the whole design is documented in [COMPUTE-GRAPH-DESIGN.md](./COMPUTE-GRAPH-DESIGN.md).

### Interactive graph visualization (`viz/`)

A zero-dependency browser page for the compute graph. `minfer viz <model>` serves the page, live SSE inference, per-node tensor stats/heatmaps and logits top-5 in one process; `--dump-graph-json` / `MINFER_TRACE` export graphs and real traces. See the [viz README](https://github.com/yusiwen/minfer/tree/master/viz) for the full user guide.

### GGUF loader

Parses GGUF v3 files (metadata + quantized tensors) with split multi-part support; weights are **mmap'd and shared zero-copy with the GPU**.

### Self-contained BPE tokenizer

Loaded directly from GGUF metadata — no external dependency on tiktoken. `tokenizer.ggml.pre` selects the pre-tokenization rule (`qwen2`, alias `deepseek-r1-qwen`; `qwen35`), implemented as hand-written splitters because the `regex` crate has no lookahead; an unknown or missing rule, an empty merge table, a non-`gpt2` model or an incomplete byte vocabulary refuses the load. Special tokens (GGUF type 3/4 table plus `<|im_start|>`/EOS fallbacks) match as single IDs before BPE, so special-token templates (DeepSeek-R1's `<｜User｜>`/`<think>`, etc.) tokenize exactly like llama.cpp, and an unmatched piece falls back byte by byte instead of silently emitting id 0. Token ids are gated byte-for-byte against transformers / llama.cpp over `tests/fixtures/tokenizer/`.

## Backends

### CPU — AVX2 / NEON+SDOT

All 8 quantized dot products as SIMD kernels (AVX2 on x86, NEON+SDOT via inline asm on Apple Silicon), plus a persistent row-parallel thread pool (`-t/--threads`). Qwen3-4B CPU decode runs ~52–58 tok/s on M4 Pro (vs 1.1 before the pool).

### GPU — Metal (Apple Silicon)

Flash attention (single fused kernel for decode + prefill), simdgroup GEMM prefill for every quant type, SIMD-parallel RMSNorm, float4-vectorized kernels, a build-time precompiled `.metallib` (no per-run shader compile), and auto-selected f16 KV cache for 7B-class models. Tracked in [METAL_OPTIMIZATIONS.md](./METAL_OPTIMIZATIONS.md).

### GPU — CUDA (NVIDIA, feature-gated `--features cuda`)

The performance headline of the project. The int8 tensor-core MMQ path is **default-on** in CUDA builds (opt-out per gate with `"0"`; `MINFER_MMQ=0` restores the legacy f16 path):

- **Default prefill: ~3581 tok/s** (7B q4_K_m @3314-token prompt) = **1.080× llama.cpp** (llama-bench 3323.3 same shape) — from 441 tok/s when the path first landed, an 8.1× campaign documented step-by-step in [CUDA_OPTIMIZATION.md](./CUDA_OPTIMIZATION.md) (75-step history table).
- Raw-nibble int8 `mma.m16n8k32` GEMMs for q4_K and q6_K with producer-fused activation quantization (rms-norm/swiglu emit the transposed q8 plane directly, skipping intermediate writes), registration-time weight-expansion planes (W_exp / W_dsc) staged by `cp.async`, and flash attention with register-resident softmax (2.43× kernel).
- CUDA Graph capture/replay for repeated identical-length prefills; decode uses the MMVQ weight-streaming path.
- Memory/speed knobs: the weight-expansion planes cost ~3.3 GB device for ~+6% prefill; `MINFER_MMQ_Q6K_EXP=0` / `MINFER_MMQ_Q4K_DSC=0` return the memory.
- Device adaptation (doc 105): the CUDA banner reports the resolved device tier — `CUDA: device tier <name> (<provenance>, mmq <bool>)` — from a cc-keyed table (GB10 measured; consumer GPUs adopted from llama.cpp; unknown → GENERIC). Dispatch gates (MMQ prefill availability, future batch caps) read the tier; on foreign devices the smem/VRAM feasibility checks self-degrade to slower-but-correct paths. `MINFER_DEVICE_TIER=<key>` forces a row for soak testing.

## Model Support

### Qwen2 / Qwen3 architectures

GQA attention, SwiGLU FFN, RoPE (Neox style), RMSNorm. Qwen3 adds the decoupled head dim and per-head Q/K RMSNorm (`attn_q_norm`/`attn_k_norm`, `Op::QkNorm`). Supported models include Qwen2.5 0.5B/7B, Qwen3 0.6B/4B, and DeepSeek-R1-Distill-Qwen-1.5B — see the matrix in [AGENTS.md](https://github.com/yusiwen/minfer/blob/master/AGENTS.md).

## User-Facing

### Model download

Auto-download from the Hugging Face Hub or the Ollama registry, with resume and cache-name resolution.

### Multi-turn conversation CLI (`--cnv`)

Append-only KV + incremental chat-template rendering: each turn only prefills the new message delta while the whole conversation accumulates in the KV cache. In-session commands (`/clear`, `/regen`, …), automatic overflow handling, `--session` persistence. On overflow the dropped turn's KV rows are removed in place and the tail is re-based/re-roped (Phase C / C2), so the turn prefills its own delta instead of the retained history (measured 185 → 14 tokens per overflowing turn on the 0.5B probe); `MINFER_NO_CONTEXT_SHIFT=1` restores the exact drop-and-re-render path. Plan: [CLI-CONVERSATION-PLAN.md](./CLI-CONVERSATION-PLAN.md).

### OpenAI-compatible HTTP server (`serve`)

`/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/health`, and `/metrics` — a Prometheus text snapshot of request counts, queue depth, live KV/arena occupancy and (under `MINFER_OP_TIMING`) per-op seconds; multi-slot with queued serial execution (`MINFER_BATCH=0`); a request that finds every slot busy on the **batched** path is refused with `503` (`minfer_jobs_dropped_total` +1, an SSE error frame when streaming) rather than queued — queueing is [#150](https://github.com/yusiwen/minfer/issues/150); a decode step whose forward fails answers every run in that batch with `500 server_error`, clears their cached prefix and frees the slots ([#151](https://github.com/yusiwen/minfer/issues/151)), so a deterministic failure cannot spin the worker on the same forward; SIGINT/SIGTERM drain bounded by `MINFER_DRAIN_MS`. Plan: [OPENAI-CHAT-API-PLAN.md](./OPENAI-CHAT-API-PLAN.md), F8 record: [ARCHITECTURE-EXECUTION-PLAN.md](./ARCHITECTURE-EXECUTION-PLAN.md).

### Chat templates (F7)

The model's own `tokenizer.chat_template` is rendered by minijinja plus a
Python-`str`-method hook, so the published Qwen2.5/Qwen3 templates (including
Qwen3's `<think>`-block split and re-emission) render as published. Reference
renderings for every supported model live in `tests/fixtures/chat/`
(transformers 5.17.0, generated from each model's `tokenizer_config.json`), and a
rendered prompt must equal them byte for byte. A template the engine cannot
render is a **loud refusal naming the construct and the template line** — checked
at load, so the CLI exits and the server refuses to start; the generic ChatML
renderer applies only to a GGUF with no template at all. Design + accepted and
refused construct sets: [CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md](./CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md).

### Constrained decoding — grammar and JSON Schema (F2)

A GBNF-style grammar or a JSON Schema is compiled **once per request** into a pushdown automaton
that masks the logits inside the one sampler pipeline, so decoding cannot leave the accepted
language.

- GBNF subset: rules, string literals, character classes with negation, `.`, grouping, alternation,
  `*`/`+`/`?`, repetition ranges `{m}`/`{m,}`/`{m,n}`, and `#` comments.
- JSON Schema subset: `type` (string or array), `enum`, `const`, object
  `properties`/`required`/`additionalProperties`, array `items`/`prefixItems`/`minItems`/`maxItems`,
  strings, integer bounds (inclusive and exclusive), numbers, booleans, null, `anyOf`/`oneOf`, and
  `$defs` + local `$ref` (recursive schemas work).
- Anything outside the subset is a **loud refusal** naming the construct (CLI startup error or HTTP
  `400`) — never a silent guess. The catalogues are in
  [GRAMMAR-DESIGN.md](./GRAMMAR-DESIGN.md).
- Token advancement is byte-level correct: a token whose piece is one byte of a multi-byte
  character is handled, a token that a rule only partially accepts is rejected with its longest
  accepted prefix named, end-of-generation is legal only at a complete state, and "no token is
  allowed" stops with a printed reason instead of emitting an arbitrary token.
- Surfaces: CLI `--grammar`/`--grammar-str`/`--json-schema`/`--json-schema-str`; the server's
  `response_format` (`json_object` / `json_schema`) and a `grammar` extension field. The mask is
  cached per automaton state and computed with a DFA-style transition memo (measured 5.4 ms per
  new state on a 151,936-token vocabulary).

### Performance benchmark (`bench`)

`minfer bench <model>` runs llama-bench-style prefill (`pp<P>`) / decode (`tg<T>`) throughput tests on the active backend — mean ± stddev over reps after an untimed warmup, each rep from an empty KV context without a model reload — reported as a markdown/CSV/JSON table.

### GGUF tooling — convert / quantize / split (F6)

`minfer convert <hf-dir> out.gguf` produces a GGUF v3 from a HuggingFace Qwen2
checkpoint with the metadata the strict tokenizer/template loader requires
(`tokenizer.ggml.model`/`pre`, the 256 byte tokens, merges, special ids and
`tokenizer.chat_template`); `minfer quantize in.gguf out.gguf --type …`
re-encodes weights to q4_0/q4_1/q5_0/q5_1/q8_0 (byte-identical to
`llama-quantize` on the same source) or f16/f32; `minfer split in.gguf dir
--max-size N` writes `split.no`/`split.count` parts the loader merges back into
one tensor index. Unsupported architectures, tensor names, dtypes and quant
targets are refused by name. A converted f16 GGUF loads and runs (f16 weights
are CPU-only). See [`GGUF-TOOLING.md`](./GGUF-TOOLING.md).

## Philosophy

**No external ML framework** — pure Rust; runtime deps are minimal (`rand`, `regex`, `half`, `serde`, `serde_json`, `minijinja`; `axum`/`tokio` only for the HTTP server). Attention, RMSNorm, RoPE, SiLU, Softmax and every quantized dot product are handwritten.
