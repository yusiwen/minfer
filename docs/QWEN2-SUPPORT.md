# Qwen2 / Qwen2.5 Architecture Support

Status page for minfer's support of the **Qwen2 architecture** (Qwen2 and
Qwen2.5 dense models). Qwen2 is the engine's oldest and most-verified model
family — the reference implementation for "adding an architecture" in
`docs/ARCHITECTURE.md` §5, and the model every optimization campaign was
measured on. The Qwen3 counterpart is `docs/QWEN3-SUPPORT-PLAN.md`.

## 1. What the architecture is

A Qwen2 decoder layer (as built by `src/models/qwen2/graph.rs`, walkthrough
diagram in `docs/ARCHITECTURE.md` §4.6):

```text
h ─ RMSNorm ─ WQ/WK/WV matmuls + bias ─ RoPE(Q, K) ─ KV store
  ─ GQA attention (Q·Kᵀ·scale → softmax → ·V) ─ WO matmul ─ + residual
  ─ RMSNorm ─ gate/up matmuls ─ SiLU(gate)·up ─ down matmul ─ + residual
```

Distinguishing properties a loader/graph must get right:

| Property | Qwen2 value | Where in minfer |
|---|---|---|
| Normalization | RMSNorm, **pre-norm**, `f_norm_rms_eps` from GGUF (`qwen2.attention.layer_norm_rms_epsilon`, `loader.rs:143`) | `HParams.f_norm_rms_eps` (`loader.rs:19`) |
| FFN activation | SwiGLU (`silu(gate) · up`), no bias | fused `SwiGLU` node |
| Attention | GQA — `n_head` query heads share `n_head_kv` K/V heads | `attn_heads` GQA mapping (walkthrough 11 §3.2.4) |
| QKV biases | **present** on WQ/WK/WV (none on WO, none in FFN) | `QKVBiasRopeStore` decode fusion needs them |
| Output-head bias | optional `output.bias` tensor | `Qwen2Model.output_b: Option<Tensor>` (`mod.rs:19`, consumed at `graph.rs:267-271`) |
| RoPE | **NonInterleaved** pairs `(i, i+hd/2)` — GGUF's NEOX convention | hardcoded `RopeStyle::NonInterleaved` (`loader.rs:154`) |
| Attention scale | `1/√n_embd_head` | `HParams::attention_scale()` (`loader.rs:36-38`) |
| KV head dim | may differ from query head dim (`n_kv_embd` decoupled) | `HParams.n_kv_embd` — Qwen2.5-0.5B: `kv_dim=128` vs `n_embd=896` (`loader.rs:25-28`) |

## 2. Where the implementation lives

```
src/models/qwen2/
├── mod.rs     # Qwen2Model + ModelDef impl (forward / build_graph / forward_graph)
├── loader.rs  # GGUF tensor loader + HParams (standard qwen2.* metadata keys)
└── graph.rs   # the compute-graph builder — deterministic in GraphParams
```

- Dispatch: `models/mod.rs::load_model()` selects Qwen2 by
  `general.architecture == "qwen2"` (Qwen3 has its own module).
- `HParams` fields (`loader.rs:11-29`): `n_embd`, `n_head`, `n_head_kv`,
  `n_layer`, `n_ff`, `n_vocab`, `max_seq_len`, `f_norm_rms_eps`,
  `rope_freq_base` / `rope_freq_scale` (read from `qwen2.rope.freq_base`,
  falling back to `llama.rope.*` keys — `loader.rs:146-150`), `eos_token_id`,
  `im_end_token_id`, `rope_style`, `n_kv_embd`.
- Per-layer weights (`LayerWeights`): `attn_norm`, `wq/wk/wv` (+ biases),
  `wo`, `ffn_norm`, `ffn_gate/ffn_up/ffn_down` — all as the GGUF lays them
  out; weights are never dequantized as a whole (walkthrough 03/10).

## 3. Graph-shape specifics (Qwen2-flavored)

- **G3 tail-row optimization** — when `n_out < nt`, the graph reduces the
  hidden states to the last `n_out` rows **before the final layer's FFN**, so
  the tail FFN, final RMSNorm and `lm_head` all run on `n_out` rows only
  (`graph.rs:61-64`, `214-222`). With tied embeddings
  (`Qwen2.5-0.5B`: `output` is `token_embd`), the output matmul is a
  `n_vocab × n_embd` Q4_0/Q8_0 matmul — the single most expensive decode
  op, which is why the tail cut matters (walkthrough 05 §3, 10 §3.2).
- **Decode fusions** — `Op::FusedQKV` (concat WQ/WK/WV matmul + 3 biases +
  2 RoPEs + KV store in one node) exists because Qwen2 has QKV biases;
  `Op::FusedFFN` (gate+up concat + SwiGLU). Both are env-revertable
  (`MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1`) and bit-identical when
  fused; backend support is per-`supports_fused` (see `docs/BACKENDS.md` §3).
- **RoPE style is model-level**: NonInterleaved is the Qwen2 family
  convention. Mixing it with the Llama-style Interleaved layout produces
  plausible-but-wrong scores with nothing crashing — it was debugging suspect
  #1 during Qwen2.5 bring-up (`docs/DEBUGGING-PLAN.md` H1).

## 4. Verified models

Per `docs/SUPPORT-MATRIX.md` and the AGENTS verified list (CPU + graph-GPU
backends; greedy output matches llama.cpp where noted):

| Model | Quants verified | Notes |
|---|---|---|
| **Qwen2.5-0.5B** | Q4_0, Q4_K_M, Q5_K_M | 24 layers · `n_embd=896` · 14 query heads / 2 KV heads (GQA 7:1) · `hd=64`, `n_kv_embd=128` · tied embeddings · KV cost 24 KB/token across all layers (walkthrough 11 §2.2) |
| **Qwen2.5-7B** | Q4_K_M | the 7B decode-bandwidth reference (28 GB→4.4 GB quantization argument, walkthrough 10 §2.1) |
| **Qwen2.5-1.5B** | historical | had a dedicated debugging era — `docs/QWEN2.5-1.5B-BUGS.md`, `docs/QWEN2.5-DEBUGGING-NOTES.md` (predates the graph refactor; the kernel fixes it drove — Metal attention hd=128 overflow, Q4_K interleaved scale/min layout, Q6_K embedding-scale indexing — are folded into `metal.metal`/`quants.rs` and regression-tested) |
| DeepSeek-R1-Distill-Qwen-1.5B | works | needs the tokenizer special-token match (same Qwen2 architecture) |

Any GGUF with `general.architecture = "qwen2"` that stays inside the supported
quant set (`docs/SUPPORT-MATRIX.md`) should load; unverified sizes are
untested rather than known-broken.

## 5. Tokenizer and chat

- Byte-level BPE, self-contained from GGUF metadata (`src/tokenizer.rs`).
- Stop tokens: `SpecialTokens { eos, im_end }` (`src/models/mod.rs:88-91`) —
  Qwen2's `<|im_end|>` id is read from GGUF (`HParams.im_end_token_id`) and
  stops generation alongside EOS.
- Chat template: the GGUF's Jinja template is rendered with minijinja
  (`src/template.rs`); on missing/invalid templates minfer falls back to
  multi-message **ChatML** (`template.rs:10-14`, `:119-130`) — which is
  Qwen2.5's native format, so the fallback is semantically safe here. System
  prompt default: `"You are a helpful assistant."`.

## 6. Context, KV sizing, and backend notes

- `max_seq_len` comes from GGUF (`qwen2.context_length`); the **KV cache is
  sized by `--n-ctx`**, not by the model's training context — 0.5B costs
  24 KB of KV per position (all 24 layers), so `--n-ctx 4096` ≈ 96 MB of
  persistent regions (walkthrough 07).
- KV positions are data: the graph is identical for prefill, decode, and
  multi-turn continuation; `positions[i] < n_ctx` is enforced with a hard
  `Err` in the KV-store arm (walkthrough 11 §3.2.3).
- Quant support per backend: `docs/SUPPORT-MATRIX.md` (CPU activations are
  Q8_0, Q8_K for K-quant weights; GPU reads f32; CUDA prefill int8 MMQ).
- CPU-vs-GPU logits differ **by design**; each path is compared against its
  own llama.cpp reference.

## 7. Troubleshooting checklist

When a Qwen2-family model produces incoherent output, check in this order
(every item here is a real bug class from the Qwen2.5 bring-up):

1. **RoPE style** — NonInterleaved must be in effect (§3); wrong style keeps
   magnitudes plausible.
2. **Quant block layouts** — Q4_K scales/mins are *interleaved* within the
   12-byte field (`docs/QWEN2.5-DEBUGGING-NOTES.md` Bug 4); Q6_K embedding
   scale indexing was its own bug (Bug 3/5). Both are covered by
   `cargo test quants::` parity tests now.
3. **Tokenizer special tokens** — chat-mode garbage on distilled models is
   usually an EOS/im_end mismatch, not a kernel bug.
4. **Head-dimension limits on GPU** — hd=128 models need the widened Metal
   attention registers (the `oc[32]` fix, Bug 1); guard failures abort with
   values per `docs/GPU_SAFETY.md`.
5. **`--n-ctx`** — a context larger than the KV allocation is rejected by the
   store arm with `Err`, not truncated silently.

## 8. Adding a Qwen2-like architecture

Qwen2 is the worked example for new architectures: mirror
`src/models/qwen2/{mod,loader,graph}.rs` with your `HParams` + `LayerWeights`,
wire tensor names, set the correct `RopeStyle` and `attention_scale`, register
in `models/mod.rs::load_model()`, and keep `build_graph` deterministic in
`GraphParams` (the reuse invariant). Details: `docs/ARCHITECTURE.md` §5,
`docs/GRAPH-REFACTOR-PLAN.md`.
