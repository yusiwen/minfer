# Architecture Roadmap — Which Model Families to Support Next

Status: **planning** (no code written yet). Recorded 2026-08 after adding
DeepSeek-R1-Distill-Qwen support; serves as the decision reference for the
next architecture work.

## Current Coverage

minfer supports the Qwen2/Qwen2.5 graph (`general.architecture = "qwen2"`,
including DeepSeek-R1-Distill-Qwen) and the Qwen3 dense graph
(`"qwen3"`, 0.6B–32B). All dense models share one graph family; the only
attention-level deltas are Qwen3's decoupled head dim and per-head Q/K
RMSNorm (`Op::QkNorm`). Backend kernels (8 quantized CPU dot products,
Metal flash attention) are architecture-agnostic and reused as-is.

## Selection Criteria

Ranked by (a) ecosystem weight — HF download trends as of 2026 put Qwen,
DeepSeek, Llama, GLM, Gemma in the first tier — and (b) **code reuse from
the existing Qwen2/Qwen3 graphs**, which dominates the cost estimate.

## Tier 1: Nearly Isomorphic to Qwen2 (pure parameter mapping, ≤ 1 day)

| Architecture | Delta vs Qwen2 | Effort |
|---|---|---|
| **Llama 3.1/3.2/3.3/3.4** | no attention bias, RoPE variant (`llama3` vs `qwen2`), SwiGLU gate/up order | smallest |
| **Mistral 7B** | same as Llama 3 (no bias) | smallest |
| **InternLM2** | ~none | smallest |
| **Phi-3/Phi-4** | qkv bias, RoPE variant, no norm | small |
| **GLM-4-9B** | dense, minor attention details | small |
| **Gemma 2** | GeGLU, shared QKV layer, **alternating SWA** (needs a sliding-window attention kernel) | medium |

## Tier 2: New Operators Needed, Bounded Increment (recommended path)

### 1. Qwen3-MoE (30B-A3B / 32B / 235B-A22B) — most natural next step

- The Qwen3 **dense** graph already contains all attention logic (QkNorm,
  decoupled head dim) — fully reused.
- Only three additions: the router (`ffn_gate_inp` linear + top-k softmax),
  3D expert weight indexing (`[n_embd, n_ff_exp, n_expert]` layout), and a
  `moe_ffn` operator.
- Watch out for `expert_weights_scale` (1.0 for 30B-A3B, 0.5 for 235B-A22B).
- 30B-A3B activates only 3B params; Q4_K_M is ~17–18 GB — **runs on M4 Pro**.
- Reference: llama.cpp `build_moe_ffn` (src/models/qwen3moe.cpp).

### 2. DeepSeek-V3/R1 (MLA + MoE)

- MLA is a KV-cache revolution: per token it stores only the compressed
  latent (`kv_lora_rank` 512 + RoPE 64), **not** `n_kv_heads × hd` — a
  >10× KV footprint reduction at long context.
- Needs the `wq_a / wq_b / wkv_a_mqa / wkv_b` weight chain, a new KV cache
  shape, and a matching attention kernel.
- V3 Q4_K_M ~20 GB — marginal on M4 Pro; R1's inference popularity makes it
  high value.
- Reference: llama.cpp src/models/deepseek2.cpp.

### 3. Qwen3-Next (hybrid SWA + MoE) — after the above

- Adds a sliding-window mask on top of the Qwen3 graph; llama.cpp's
  `build_attn` takes a `swa` parameter directly, and the Metal flash
  attention kernel needs mask support.

## Tier 3: Large Architectural Deltas (new kernels, high cost)

- **Gemma 3 / Qwen3-VL**: multimodal (vision encoder) — minfer has no
  multimodal framework; highest cost.
- **Llama 4 Scout**: MoE + interleaved attention; open weights but special
  training.
- **RWKV / Mamba / Jamba**: SSM recurrence, entirely different kernels,
  unrelated to the transformer line.

## Suggested Order

```
1. Qwen3-MoE (30B-A3B)   ← reuses the Qwen3 dense graph; +3 components; runnable for verification
2. Llama 3.x dense       ← widest ecosystem coverage; smallest effort
3. DeepSeek-V3 (MLA)     ← KV revolution + R1 popularity; medium-large
4. GLM-4-9B              ← domestic ecosystem; easy
5. Qwen3-Next / V3.2     ← SWA mask
```

## Why the Graph Architecture Keeps This Cheap

The declarative compute graph (build_graph + loader weight registration +
shared matmul / RMSNorm / attention kernels) means a new architecture's
increment lives almost entirely in the model-logic layer: add
`models/<name>/{mod,loader,graph}.rs` and dispatch in `models/mod.rs`.
The backend kernels (CPU NEON dots for all 8 quants, Metal flash attention)
are shared and untouched. See `docs/ARCHITECTURE.md` for the module design.
