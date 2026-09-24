# Model Support Roadmap — Which Model Families to Support Next

Status: **planning** (no code written for the architectures proposed below).
Recorded 2026-08 after adding DeepSeek-R1-Distill-Qwen support; serves as the
decision reference for the next model-family work.

> **Companion document.** `docs/ARCHITECTURE-ROADMAP.md` covers the *system*
> layers (IR, scheduler, allocator, KV cache, batching, backend abstraction).
> The `Prerequisites` column below references its backlog item numbers
> (`§3 item N`); several Tier 2/3 entries are gated on that work.

## Current Coverage

minfer supports the Qwen2/Qwen2.5 graph (`general.architecture = "qwen2"`,
including DeepSeek-R1-Distill-Qwen) and the Qwen3 dense graph
(`"qwen3"`, 0.6B–32B). All dense models share one graph family; the only
attention-level deltas are Qwen3's decoupled head dim and per-head Q/K
RMSNorm (`Op::QkNorm`). Backend kernels are architecture-agnostic and reused
as-is — but only 2 of the 8 quantized CPU dot products have AVX2 kernels
(`docs/ARCHITECTURE-ROADMAP.md` §2.7), so "reused as-is" is not the same as
"equally fast everywhere".

## Selection Criteria

Ranked by (a) ecosystem weight — HF download trends as of 2026 put Qwen,
DeepSeek, Llama, GLM, Gemma in the first tier — and (b) how much of the
existing Qwen2/Qwen3 graph the family reuses.

Criterion (b) only dominates the cost estimate for **parameter-isomorphic**
families. For anything needing new IR structure, the reuse is small and the
cost is dominated by the touch sites listed below — see the next section
before reading the effort column in the tier tables.

## Cost model: what a port actually costs

### Parameter-isomorphic (the cheap case)

The bulk of the work is model logic: add `models/<name>/{mod,loader,graph}.rs`
and dispatch in `models/mod.rs::load_model()`. The shared matmul / RMSNorm /
attention kernels and the backend scheduling are untouched. The Tier 1 effort
estimates below assume exactly this.

Four items are **not** covered by that estimate and apply to every port:

1. **Decode-fusion weight registration.** The decode path fuses QKV
   (`blk.{i}.attn_qkv`) and gate+up (`blk.{i}.ffn_gu`) only when the loader has
   registered the concatenated tensors — once for Metal and once for CUDA
   (`models/qwen2/loader.rs:425`, `:488`; `models/qwen3/loader.rs:426`, `:468`)
   — plus the matching `qkv_concat_available` / `gu_concat_available`
   predicates (`models/qwen2/graph.rs:282-320`). Skip them and the port is
   still **correct**, but decode fusion silently disables: no error, just a
   slower decode path.
2. **RoPE style.** `rope_style` is hard-coded to `NonInterleaved` in both
   loaders (`models/qwen2/loader.rs:154`, `models/qwen3/loader.rs:172`).
   `RopeStyle::Interleaved` exists and the CPU implements it
   (`graph/cpu_backend.rs:491-492`), but CUDA refuses it
   (`graph/cuda_backend.rs:1324`) and no loader ever selects it. A family
   needing interleaved RoPE (Llama 1/2, Mistral 7B v0.1) needs a loader change
   **and** a CUDA kernel, or must be converted to the non-interleaved
   convention.
3. **RoPE scaling.** Only `*.rope.freq_base` and `*.rope.frequency_scale` are
   read (`models/qwen2/loader.rs:146-149`). The long-context scaling types
   (`rope.scaling.type = llama3`, i.e. the low/high-frequency factors in
   Llama 3.1+) are not implemented, so a Llama 3.x port is correct at short
   context and wrong past the original training window until this lands.
4. **Chat template.** minijinja 2.21 exposes no `str` methods natively; since
   F7 ([#50](https://github.com/yusiwen/minfer/issues/50)) minfer installs an
   unknown-method hook implementing the Python `str` methods with CPython
   semantics, so a port whose template uses them renders as published. A
   template using a construct outside the implemented set is refused loudly at
   load (`docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`), never silently replaced
   by a generic prompt.

### New structure (the expensive case)

Anything that needs a new operator shape pays a five-site change
(`docs/ARCHITECTURE-ROADMAP.md` §2.1): the builder constructor, the
allocator's special case, the scheduler's `kv_pair` resolution, every
backend's `supports_op` + `execute_node`, and both models' `build_graph`. The
existing decode fusions (`FusedQKV`, `QkvBiasRopeStore`, `FusedFFN`,
`FusedQkvNorm`) are four worked examples of this tax.

The counterpart fix is `ARCHITECTURE-ROADMAP.md` §3 item 7 (strided views with
allocator-known aliasing, plus multi-output nodes): with it, the new-structure
cases below can be expressed as compositions and derived by the fusion pass
instead of being hand-written per backend.

## Tier 1: Nearly Isomorphic to Qwen2 (parameter mapping + the prerequisites below)

| Architecture | Delta vs Qwen2 | Prerequisites | Effort |
|---|---|---|---|
| **Llama 3.1/3.2/3.3/3.4** | no attention bias (optional in the IR), RoPE variant, SwiGLU gate/up order | **RoPE scaling** (cost model #3) — short context works without it, long context does not; concat-weight registration (#1) | smallest + rope work |
| **Mistral 7B** | same as Llama 3 (no bias) | **Interleaved RoPE** for v0.1 (cost model #2): loader + CUDA kernel; or convert to the non-interleaved convention | small–medium |
| **InternLM2** | ~none | concat-weight registration (#1) only | smallest |
| **Phi-3/Phi-4** | qkv bias, RoPE variant, no norm | qkv bias is already supported (`models/qwen2/loader.rs:403`); verify the RoPE variant | small |
| **GLM-4-9B** | dense, minor attention details | none known beyond #1 | small |
| **Gemma 2** | GeGLU, shared QKV layer, **alternating SWA** | **`Gelu` op** (absent from `Op`, `graph/ops.rs` — GeGLU needs it) and a **sliding-window mask** in the attention kernels (adjacent to `ARCHITECTURE-ROADMAP.md` §3 item 2, but a distinct parameter — a window bound rather than a cell set); GeGLU itself is a five-site change unless `§3 item 7` lands first | medium |

## Tier 2: New Operators Needed (see prerequisites before starting)

### 1. Qwen3-MoE (30B-A3B / 32B / 235B-A22B) — highest structural value

- The Qwen3 **dense** graph already contains all attention logic (QkNorm,
  decoupled head dim) — fully reused.
- Three additions: the router (`ffn_gate_inp` linear + top-k softmax), 3-D
  expert weight indexing (`[n_embd, n_ff_exp, n_expert]` layout), and a
  `moe_ffn` operator.
- **Prerequisite: `ARCHITECTURE-ROADMAP.md` §3 item 7.** The IR has no
  `mul_mat_id` equivalent and no multi-output node, so 3-D expert indexing is
  not expressible today; `moe_ffn` would have to be a bespoke op at the
  five-site cost. With item 7 landed, the three additions above are
  compositions.
- Watch out for `expert_weights_scale` (1.0 for 30B-A3B, 0.5 for 235B-A22B).
- 30B-A3B activates only 3B params; Q4_K_M is ~17–18 GB — **runs on M4 Pro**,
  which makes it the natural first structural target once item 7 lands.
- Reference: llama.cpp `build_moe_ffn` (src/models/qwen3moe.cpp).

### 2. DeepSeek-V3/R1 (MLA + MoE)

- MLA is a KV-cache revolution: per token it stores only the compressed
  latent (`kv_lora_rank` 512 + RoPE 64), **not** `n_kv_heads × hd` — a
  >10× KV footprint reduction at long context.
- Needs the `wq_a / wq_b / wkv_a_mqa / wkv_b` weight chain, a new KV cache
  shape, and a matching attention kernel.
- **Prerequisites: `§3 items 1, 2, 7`** — the KV layout change is exactly the
  case the current fixed per-layer K/V regions cannot express
  (`ARCHITECTURE-ROADMAP.md` §2.4), and the latent split needs views.
- V3 Q4_K_M ~20 GB — marginal on M4 Pro; R1's inference popularity makes it
  high value.

### 3. Qwen3-Next (hybrid SWA + MoE) — after the above

- Adds a sliding-window mask on top of the Qwen3 graph; the flash-attention
  kernels need mask support (the same prerequisite as Gemma 2, so the two
  share one piece of work).
- **Prerequisites:** the SWA mask (see Tier 1, Gemma 2), plus MoE
  (`§3 item 17`) and `§3 item 7`.

## Tier 3: Large Architectural Deltas (new kernels, high cost)

- **Gemma 3 / Qwen3-VL**: multimodal (vision encoder) — minfer has no
  multimodal framework; highest cost.
- **Llama 4 Scout**: MoE + interleaved attention; open weights but special
  training.
- **RWKV / Mamba / Jamba**: SSM recurrence, entirely different kernels, and a
  KV/memory model that is neither the current per-layer regions nor the
  cell store proposed in `ARCHITECTURE-ROADMAP.md` §2.4.

## Suggested Order

Two tracks, because the gates are the whole point: the unblocked ports can
start today, and the structural ones should not be scheduled before their
prerequisites.

```
Unblocked today (Tier 1):
  1. Llama 3.x dense          ← widest ecosystem coverage; add RoPE scaling
  2. GLM-4-9B                 ← easy; no known prerequisite
  3. InternLM2 · Phi-3/4      ← easy; same wave as 1–2

Gated on ARCHITECTURE-ROADMAP work:
  4. Qwen3-MoE (30B-A3B)      ← §3 item 7 (IR views / multi-output)
  5. DeepSeek-V3 / R1 (MLA)   ← §3 items 1 + 2 + 7
  6. Gemma 2 · Qwen3-Next     ← sliding-window mask (+ §3 item 17 for Qwen3-Next)
```

The single highest-leverage system-layer item for this document is
`ARCHITECTURE-ROADMAP.md` §3 item 7: it unlocks Qwen3-MoE, is a prerequisite
for MLA, and removes the five-site tax from every future structural port. The
second is the sliding-window mask, which Gemma 2 and Qwen3-Next share.
