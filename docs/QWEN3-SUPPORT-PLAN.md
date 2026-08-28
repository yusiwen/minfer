# Qwen3 Support Plan (dense architecture)

> **STATUS: IMPLEMENTED (2026-08-23).** Dense Qwen3 is fully supported on CPU
> and Metal GPU. See **§6 Implementation Record** below; the rest of this
> document remains the design record the implementation followed.

Status: **Qwen3 is NOT supported by minfer today.** `load_model()` (`src/models/mod.rs`)
only dispatches on `"qwen2"`; a `general.architecture = "qwen3"` GGUF fails with
`Unsupported architecture: 'qwen3'` and `main.rs` panics at "load model".

This document analyzes the Qwen3 dense architecture against minfer's existing
Qwen2 support (which already implements all the primitives Qwen3 needs) and
gives a phased implementation plan. Reference: llama.cpp's upstream Qwen3
implementation.

Reference model: `Qwen3-0.6B-Instruct-Q8_0.gguf` (Qwen/Qwen3-0.6B-GGUF, Q8_0,
file_type 7, quantization_version 2).

---

## 1. What the model is

| Key | Value | minfer impact |
|---|---|---|
| architecture | `qwen3` (dense — no MoE, no sliding window) | new dispatch branch |
| block_count | 28 | — |
| embedding_length | 1024 | `n_embd` |
| head_count / head_count_kv | 16 / 8 (GQA) | `n_head` / `n_head_kv` |
| **key_length / value_length** | **128 / 128** | **`n_embd_head` — decoupled from `n_embd/n_head = 64`** |
| feed_forward_length | 3072 | `n_ff` (SwiGLU) |
| rope.freq_base | 1 000 000 | `freq_base` (NeoX / NonInterleaved RoPE) |
| layer_norm_rms_epsilon | 1e-6 | `f_norm_rms_eps` |
| context_length | 40960 | `max_seq_len` / `n_ctx` |
| tokenizer | gpt2 model, `pre = "qwen2"` | minfer's BPE tokenizer works unchanged |
| eos / bos / pad | 151645 (`<|im_end|>`) / 151643 / 151643, `add_bos = false` | `SpecialTokens` fine |
| chat_template | Qwen3 ChatML + `<think>` tags | **does NOT render via minijinja** — uses Python str-method syntax (`.split()/.lstrip()`) that minijinja 2.21.0 lacks → falls back to ChatML (see §5 gotcha) |
| vocab | 151936 (÷32 ✓ for Q8_0 blocks) | — |
| weights | all Q8_0 (q/k/v/wo/gate/up/down) + f32 norms | Q8_0 fully supported (CPU + Metal) |

Tensor inventory per layer (from the GGUF):

```
blk.{i}.attn_norm.weight    f32   [1024]
blk.{i}.attn_q.weight       q8_0  [1024, 2048]      # 16 × 128
blk.{i}.attn_k.weight       q8_0  [1024, 1024]      #  8 × 128
blk.{i}.attn_v.weight       q8_0  [1024, 1024]
blk.{i}.attn_output.weight  q8_0  [2048, 1024]
blk.{i}.attn_q_norm.weight  f32   [128]   # NEW vs Qwen2 — per-head Q RMSNorm
blk.{i}.attn_k_norm.weight  f32   [128]   # NEW vs Qwen2 — per-head K RMSNorm
blk.{i}.ffn_norm.weight     f32   [1024]
blk.{i}.ffn_gate/up/down    q8_0  [1024,3072] / [1024,3072] / [3072,1024]
```

> **Gotcha:** both the `minfer info` listing and the original metadata dump
> truncate the tensor list, so `attn_q_norm` / `attn_k_norm` do not appear in
> them. They **are** in the file (56 occurrences, verified with
> `strings model.gguf | grep attn_q_norm`). Do not "fix" the file or skip the
> tensors — the loader must load them or the model produces garbage.

---

## 2. Architecture deltas vs Qwen2.5 (the entire difference)

Everything else — RMSNorm pre-norm, SwiGLU FFN with `ffn_norm`, no biases,
GQA attention, NeoX (non-interleaved) RoPE, tied lm_head — is byte-identical
in structure to Qwen2.5 and already implemented in minfer. Only two things
differ:

### 2.1 Decoupled head dimension (`n_embd_head = 128`, not `n_embd/n_head = 64`)

Qwen3 decouples `head_dim` from `hidden_size / n_head` (HF config `head_dim`).
Consequences:

- `attn_q` projects to `n_head × 128 = 2048`, `attn_k`/`attn_v` to
  `n_head_kv × 128 = 1024`, `attn_output` maps `2048 → 1024`.
- RoPE rotates the **full** 128-dim head (llama.cpp: `GGML_ASSERT(n_embd_head == n_rot)`),
  `freq_base = 1e6`.
- QK scale is `1/sqrt(128)`.
- KV cache row is 1024 floats wide (`n_kv_embd = 1024`, `hd_kv = 128`).

llama.cpp derives this generically in `llm_load_hparams`
(`src/llama-model.cpp:1219-1226`):

```cpp
hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);   // "qwen3.attention.key_length" = 128
hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false); // "qwen3.attention.value_length" = 128
hparams.n_rot_full = hparams.n_embd_head_k_full;                              // no "qwen3.rope.dimension_count" → n_rot = 128
```

minfer's Qwen2 `HParams::n_embd_head()` computes `n_embd / n_head` and the
loader only overrides `n_kv_embd` from the K weight — for Qwen3 we must also
carry an explicit `n_embd_head` read from `qwen3.attention.key_length` (and
`value_length`, same value here).

### 2.2 Per-head Q/K RMSNorm (`attn_q_norm`, `attn_k_norm`)

New for Qwen3: after the Q and K projections (no biases), each head's `hd`
elements are RMS-normalized and scaled by a per-head weight of length `hd`,
**before** RoPE:

```
Qcur = Wq @ x                        # [nt, 2048]
Qcur = reshape [128, 16, nt] → rms_norm(eps=1e-6) per (head, token) → × attn_q_norm[128]
Qcur = rope(Qcur, n_rot=128)         # NeoX
```

llama.cpp reference (`src/models/qwen3.cpp` graph):
- `load_arch_tensors` creates `attn_q_norm` / `attn_k_norm` with shape
  `{n_embd_head_k}` (`qwen3.cpp:33-34`).
- graph: `build_qkv(...)` reshapes Q/K/V to 3D `[n_embd_head, n_head, n_tokens]`
  (`src/llama-graph.cpp:build_qkv`), then
  `build_norm(Qcur, attn_q_norm, NULL, LLM_NORM_RMS, il)` — i.e.
  `ggml_rms_norm` over `ne[0] = 128` per (head, token), then multiply by the
  weight — then `ggml_rope_ext` with `n_rot = 128`
  (`qwen3.cpp:78-100`).

In minfer's flat token-major activation layout this is a **contiguous
`[nt·n_head][hd]` matrix**, so it is exactly the existing RMSNorm kernel with
`d = hd`, `n = nt·n_head`, weight length `hd` — no new math, just a new op that
encodes the row grouping (`n_heads` instead of 1).

The rest of `qwen3.cpp` (FFN, output) is a standard
`rms_norm → swiglu(gate,up) → down → add` + `output_norm → lm_head`,
identical to Qwen2.

---

## 3. Implementation plan

Phased, correctness first; each phase leaves the tree compiling and the model
run-able.

### Phase A — `models/qwen3` scaffolding (mirror `models/qwen2`)

1. `src/models/qwen3/mod.rs`
   - `Qwen3Model { hparams, tok_embd, output_norm, output, output_b, layers }`
     and `impl ModelDef` — copy `qwen2/mod.rs` (forward routes through the
     graph; `format_chat` ChatML fallback; `special_tokens` from hparams;
     accessors use explicit `n_embd_head`).
   - `tensor_names`: copy + add
     `attn_q_norm(i) = "blk.{i}.attn_q_norm.weight"`,
     `attn_k_norm(i) = "blk.{i}.attn_k_norm.weight"`.
2. `src/models/qwen3/loader.rs`
   - `HParams`: same fields as Qwen2 **plus** explicit `n_embd_head: i64`
     (from `qwen3.attention.key_length`, fallback `n_embd/n_head`);
     `n_embd_head()` returns the stored value; `attention_scale() =
     1/sqrt(n_embd_head)`.
   - Read keys with the `qwen3.` prefix (`qwen3.embedding_length`,
     `qwen3.attention.head_count[_kv]`, `qwen3.block_count`,
     `qwen3.feed_forward_length`, `qwen3.context_length`,
     `qwen3.attention.layer_norm_rms_epsilon`, `qwen3.rope.freq_base`,
     `qwen3.rope.frequency_scale` → default 1.0).
   - `LayerWeights`: add `q_norm: Option<Tensor>`, `k_norm: Option<Tensor>`.
   - Load `blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight` (f32, `[128]`);
     register on Metal/CUDA like other weights (f32 registration already
     exists in `load_tensor`).
   - Resolve `n_kv_embd` from `blk.0.attn_k.weight` `ne[1]` (= 1024) and call
     `set_kv_cache_type(n_layer, 1024)` **after** that (28×1024 = 28672 ≥ 8192
     → f16 KV auto-pick; using the naive default 8×64 = 512 would still give
     f16, but be correct).
   - Keep the QKV concat (`blk.{i}.attn_qkv`) and FFN concat
     (`blk.{i}.ffn_gu`) GPU registration from qwen2 — FFN fusion is reused
     as-is; QKV fusion is gated off in Phase C.
3. `src/models/mod.rs` — add `pub mod qwen3;` and dispatch
   `"qwen3" => qwen3::loader::load(model)`.

### Phase B — new graph op `Op::QkNorm { hd, nh }`

Per-head RMSNorm is not expressible with the existing `RmsNorm` (it
normalizes the whole row). Add a dedicated op; the kernels themselves are the
existing norm kernels with a different row grouping:

1. `src/graph/ops.rs` — add variant
   `QkNorm { hd: usize, nh: usize }` (eps comes from `RmsNorm`-style meta;
   reuse `NodeMeta::Norm { weight_name, bias_name }`); add to `NodeMeta`
   mapping (out_shape = in_shape).
2. `src/graph/builder.rs` — `pub fn qk_norm(&mut self, x, weight_name,
   hd, nh, eps) -> NodeId`.
3. `src/graph/cpu_backend.rs` — execute: the input buffer is `[nt · nh · hd]`
   floats and the norm rows are **contiguous** (`t·(nh·hd) + h·hd`), so run the
   existing loop with `d = hd`, `n = nt·nh`, using
   `vec_ops::rms_norm_fused_f32(hd, dst, row, w.data_f32(), eps)` (weight
   length `hd`). Add a `supports_op` arm (CPU: always true).
4. `src/graph/metal_backend.rs` — `QkNorm` arm → `cb.rms_norm_256(x, w, w_off,
   y, hd, nt·nh, eps, 0)` (fall back to `rms_norm` if the 256-thread kernel is
   disabled); `supports_op` arm. The MPS kernel takes `(d, n)` — no shader
   change needed. Add to `supports_op` gates so backend assignment works.
5. Fusion pass (`src/graph/fusion.rs`): `QkNorm` is not a fusion target —
   ensure the matcher leaves it untouched (default no-op is fine).

Math to match (llama.cpp): `y = x · rsqrt(mean(x²) + eps) · w` with
`eps = f_norm_rms_eps = 1e-6`.

### Phase C — graph wiring (`src/models/qwen3/graph.rs`)

Copy `qwen2/graph.rs` and change only the attention spine:

```rust
let hd  = hp.n_embd_head as usize;            // 128 (NOT n_embd/n_head = 64)
let nkt = hp.n_kv_embd as usize;              // 1024
let hd_kv = nkt / nk;                         // 128
let attn_scale = hp.attention_scale();        // 1/sqrt(128)

// per layer, unfused path (decode AND prefill):
let q = b.matmul(normed, l.wq, None);
let k = b.matmul(normed, l.wk, None);
let v = b.matmul(normed, l.wv, None);
let q = b.qk_norm(q, "blk.{i}.attn_q_norm.weight", hd, nh, eps);
let k = b.qk_norm(k, "blk.{i}.attn_k_norm.weight", hd, nk, eps);
let q = b.rope(q, inp_pos, RopeStyle::NonInterleaved, RoPEMeta { freq_base: 1e6, freq_scale: 1.0, n_head: nh, hd });
let k = b.rope(k, inp_pos, ..., RoPEMeta { n_head: nk, hd });
b.kvcache_store(i, k, v, inp_pos, n_ctx);
let kv = b.kvcache_load(i, nkt, n_ctx, nk);
let attn = b.attn(q, kv, inp_pos, Gqa, AttnMeta { n_head: nh, n_head_kv: nk, hd, hd_kv, nkt, scale });
```

- **Do NOT use the fused decode QKV path (`FusedQKV`) for Qwen3 initially**:
  the `attn_bias_rope_store` Metal kernel does bias+rope+store in one pass and
  cannot express the per-head norm between projection and rope. Gate
  `fuse_qkv = false` for Qwen3 (the env toggle `MINFER_NO_FUSE_QKV` already
  exists for A/B; make it the default for this arch until Phase E). The
  unfused 3-matmul path is correct on both backends.
- **FFN unchanged**: gate/up/down + `ffn_norm`; the fused FFN decode path
  (`blk.{i}.ffn_gu` concat, `nf = 3072 ≤ 16384`) is reused as-is.
- G3 tail-row reduction (`get_rows` on the last layer + logits), KV-in-allocator
  persistent regions, params-only reuse, `weights_on_gpu`,
  `register_graph_weights` (add `q_norm`/`k_norm` to the lists) — all carried
  over unchanged.

### Phase D — verification

1. `cargo build --release`; `minfer info <model>` parses; add a strings-level
   check that `attn_q_norm`/`attn_k_norm` load (loader logs layer count).
2. **Self-consistency**: prefill + decode produce a sensible greedy
   continuation; prefill logits are deterministic across runs.
3. **CPU vs Metal**: with `MINFER_GRAPH_DUMP=/tmp/d`, compare prefill/decode
   logits — greedy tokens must agree, logits may differ ~1e1 (CPU quantizes
   activations to Q8_0, Metal reads f32 — known path difference, see
   AGENTS.md §9).
4. **Cross-check vs llama.cpp** (the authoritative reference for this arch):
   run the same Q8_0 GGUF in `llama-cli` (or `llama-server`) with `temp 0`,
   same prompt, and compare greedy token sequences — they must match; compare
   raw logits (`--logits-all` / server `logprobs`) to the minfer **Metal**
   graph path (f32 activations on both sides): expect ~1e-2 max abs diff
   (reduction-order noise), which also validates the per-head norm math.
5. **Tokenizer/template smoke**: `minfer <model> "hello"` without
   `--no-template` — the GGUF chat template renders via minijinja (it already
   handles `tools`/`enable_thinking` as undefined → falsy).
6. Add hermetic tests in `qwen3/graph.rs` modeled on the qwen2 tests
   (`graph_logits_match_forward_real_model`, `forward_cached_isolates_kv_between_caches`)
   using the cached Qwen3 GGUF; update the model matrix in `AGENTS.md` /
   `README.md`.

### Phase E — follow-ups (explicitly out of scope for the initial port)

- **Fused QKV decode with qk-norm**: extend the Metal `attn_bias_rope_store`
  kernel (or add `attn_qk_norm_rope_store`) so nt==1 decode gets the single
  concat matmul + one-pass norm+rope+store back for Qwen3. Until then Qwen3
  decode pays 3 matmuls (same as Qwen2 prefill path).
- **Thinking-tag output**: Qwen3 (instruct) emits `<think>…</think>` blocks;
  optionally strip/collect them like llama.cpp's `--reasoning-format` — a
  CLI/formatting concern, not an engine change.
- **Other Qwen3 family members** (different arch IDs in llama.cpp): MoE
  (`LLM_ARCH_QWEN3MOE` — expert tensors, router), hybrid SWA/MLA
  (`LLM_ARCH_QWEN3NEXT` — sliding-window + per-layer dense flags), VL
  (`QWEN3VL*`), reranker/embedding (pooling + `cls_out`), Qwen3.5
  (`LLM_ARCH_QWEN35*`). Each is a separate project; the dense port above
  already covers Qwen3-0.6B/1.7B/4B/8B/14B/32B.

---

## 4. File touch list

| File | Change |
|---|---|
| `src/models/mod.rs` | dispatch `"qwen3"` |
| `src/models/qwen3/mod.rs` | new — `Qwen3Model` + `ModelDef` + `tensor_names` |
| `src/models/qwen3/loader.rs` | new — HParams (explicit `n_embd_head`), LayerWeights + q/k_norm, tensor load + GPU registration |
| `src/models/qwen3/graph.rs` | new — `Qwen3Graph::build/forward/forward_cached` (per-head norm spine, fuse_qkv off) |
| `src/graph/ops.rs` | new `Op::QkNorm { hd, nh }` |
| `src/graph/builder.rs` | new `qk_norm()` builder method |
| `src/graph/cpu_backend.rs` | `QkNorm` exec arm (d=hd, n=nt·nh) + `supports_op` |
| `src/graph/metal_backend.rs` | `QkNorm` exec arm (rms_norm_256) + `supports_op` |
| `docs/` (this plan), `AGENTS.md`, `README.md` | model matrix / docs |

---

## 5. Risks & gotchas

1. **Head dim trap**: if `n_embd_head` falls back to `n_embd/n_head = 64`, the
   Q/K projections are misinterpreted, RoPE rotates the wrong dims, the
   attention scale is wrong, and the KV row stride mismatches — silent garbage.
   Read `qwen3.attention.key_length` and assert `n_kv_embd == n_head_kv ·
   key_length` (1024 = 8 × 128) and `ne[1](attn_q) == n_head · key_length`.
2. **Truncated tensor listing**: `minfer info` and naive metadata dumps hide
   `attn_q_norm`/`attn_k_norm`; the loader must load them by name regardless.
3. **Fused decode QKV cannot express qk-norm** — keep `fuse_qkv` off for Qwen3
   or decode logits will silently skip the norm (bitwise different from llama).
4. **RoPE**: `freq_base = 1e6`, `freq_scale = 1.0`, `n_rot = hd = 128`,
   NeoX/NonInterleaved — minfer's `cpu_rope`/Metal `rope_f32` already rotate
   the full head dim given `hd`, so only the `hd` value changes.
5. **KV f16 auto-pick** uses `n_layer × n_kv_embd`; pass the real 1024.
6. **ChatML template** — expected to render, but it does NOT: the Qwen3
   `chat_template` uses Python string-method syntax that minijinja 2.21.0
   cannot run, so it always falls back to ChatML. The fallback emits
   `<|im_start|>` correctly (Qwen3-compatible), so plain chat works and the
   model still emits `<think>` blocks (thinking mode). But the template's
   think-block extraction, tool-call formatting and `enable_thinking` handling
   are lost — see gotcha #9 below.
7. **No biases anywhere** in Qwen3 — `bq/bk/bv/output_b` stay `None`; the
   loader's optional-bias paths already handle that.
8. **Context 40960**: `n_ctx` defaults from `qwen3.context_length`; KV region
   sizing is `n_kv_embd × n_ctx` = 1024 × 40960 ≈ 40 MB/layer → ~1.1 GB f16 —
   fine, but keep the f16 KV path (auto-selected).
9. **Qwen3 chat_template is NOT rendered by minijinja** (2026-08-27) — the
   engine log shows
   `chat template rendering failed (unknown method: string has no method named split), falling back to ChatML`.
   Root cause: the template (`tokenizer.chat_template` from the GGUF) is written
   with **Python string-method syntax** (`message.content.split('</think>')`,
   `.lstrip('\n')`, `.rstrip('\n')` at template lines 35-36). minijinja 2.21.0
   strings are Rust strings and expose **no** `str` methods; string operations
   must be Jinja **filters** (`|split`, `|replace`). It also lacks the
   `lstrip`/`rstrip`/`strip`/`contains` filters. So the template always fails at
   line 35 and `template.rs::render_messages` falls back to `fallback_chatml_messages`
   (`src/template.rs:50-56`). Consequence: the template's think-block extraction
   (`<think>…</think>` split into `reasoning_content` vs `content`), tool-call
   formatting (`<tool_call>`/`<tool_response>`), multi-step-tool collapse, and
   `enable_thinking` are LOST; the fallback keeps `<think>` inline and feeds it
   back verbatim on the next turn. Plain chat still works (the ChatML fallback
   uses the Qwen3-compatible `<|im_start|>` markers). Fix direction and the
   engine-level limitation are recorded in
   `docs/OPENAI-CHAT-API-PLAN.md` §Chat Template Handling.

---

## 6. Implementation Record (2026-08-23)

### 6.1 What was implemented

- `src/models/qwen3/` (mod.rs, loader.rs, graph.rs) — dense Qwen3 mirroring
  qwen2 with the two deltas: explicit `n_embd_head` from
  `qwen3.attention.key_length` (= 128) and the per-head Q/K RMSNorm.
- `src/models/mod.rs` — dispatch `"qwen3"`.
- New graph op **`Op::QkNorm { hd, nh, eps }`** (`src/graph/ops.rs`) +
  `GraphBuilder::qk_norm()` + CPU and Metal execution arms. The flat
  token-major buffer is a contiguous `[nt·nh][hd]` matrix, so both backends
  reuse the existing RMSNorm kernels with `d = hd`, `n = nt·nh` (Metal:
  `rms_norm_256`; no shader change).
- Decode fused-QKV (`Op::FusedQKV`) is NOT used for Qwen3 (per §3 Phase C):
  the `attn_bias_rope_store` kernel cannot express the per-head norm. Fused
  FFN (`Op::FusedFFN`) IS reused unchanged.
- `MINFER_GRAPH_DUMP` for qwen3 writes per-position decode logits
  (`logits_decode_{pos}.f32`) + layer-0 intermediate nodes, so a full
  generation can be inspected step by step (used heavily during verification).

### 6.2 Verification (all on Qwen3-0.6B-Q8_0)

1. **Prefill math vs llama.cpp**: for 6 raw prompts, the argmax token at the
   last position is identical and the log-prob agrees within 0.01–0.06
   (minfer CPU Q8_0-activation path vs llama.cpp CPU).
2. **Greedy generation vs llama.cpp**: with matching sampler config
   (`temp 0`, `repeat_penalty 1.0`) the raw-prompt continuation is
   **byte-identical for 60 tokens** ("The capital of France is" →
   " Paris. The capital of France is also the capital of the Republic of
   France. …"). The default `repeat_penalty = 1.1` (both engines) changes the
   greedy path at ambiguous positions — a sampler-config difference, not an
   engine bug (verified: identical config ⇒ identical output).
3. **Templated (chat) mode**: `<think>`-mode generations match for the first
   ~8 tokens, then flip at genuinely ambiguous positions where the top-2
   candidates are within float noise (verified via full top-10 log-prob
   comparisons at the diverging step — distributions agree, order flips).
4. **In-crate tests** (`src/models/qwen3/graph.rs`): CPU self-consistency
   (two independent caches bit-identical), KV isolation between caches, Metal
   greedy == llama.cpp-verified reference token sequence, Metal prefill
   bitwise determinism.

### 6.3 Bug found & fixed: Metal Q8_0 multi-token matmul race

Verification exposed a **pre-existing Metal nondeterminism bug**:
`kernel_q8_0_f32_matmul_multi` (`src/metal.metal`) loops over tokens and
re-zeroes its shared-memory accumulator (`sh0[tiisg] = 0`) at the top of each
iteration, but had **no trailing `threadgroup_barrier`** — a fast thread's
re-zero could overtake a slow thread still reducing `sh0[tiisg]`, producing
wrong (often 0) output elements. Symptoms: greedy text differed between
identical runs, and K-cache rows had 0-vs-nonzero value pairs at even indices
(the `NR0 = 2` row-pair output). Exposed only by Qwen3 because it is the first
model whose prefill Q/K/V matmuls take the `pl_q8_0_f32_multi` path
(`od = 1024 < 2048`, `nt < 9` → no GEMM); qwen2 models use other kernels
(Q4_0/Q4_K paths are per-simgroup, no shmem).

Fix: `threadgroup_barrier(mem_flags::mem_threadgroup);` at the end of the
t-loop body. Audited all other `_multi` kernels (lines 115/249/466/528/1627/
1791/1979/2148): they use per-simgroup rows + `simd_sum` only, no cross-
simgroup shmem inside a loop — only the Q8_0 kernel needed the barrier.
Regression test: `metal_prefill_determinism` (bitwise determinism of the
layer-0 K path across two Metal executions).

### 6.4 Known follow-ups (unchanged from §3 Phase E)

- **Fused QKV decode with qk-norm** — **DONE 2026-08-27** (`Op::FusedQkvNorm` + the
  no-bias `kernel_attn_rope_store`). We added a new fused decode op that
  concatenates Wq/Wk/Wv into one matmul and applies the per-head Q/K RMSNorm +
  no-bias RoPE + KV store in place on the concat buffer (a single op replacing
  3 matmul + 2 qk_norm + 2 rope + 2 store). The Qwen2 `attn_bias_rope_store`
  path (biases, no per-head norm) is untouched.
- Qwen3 MoE / hybrid-SWA / VL / reranker variants (`LLM_ARCH_QWEN3MOE`,
  `QWEN3NEXT`, `QWEN3VL*`) — separate architectures, out of scope here.
- Optional `<think>`-block stripping at the CLI layer — note: since the Qwen3
  template isn't rendered (gotcha #9), the `<think>` block stays inline and is
  fed back verbatim; fixing the template render (see `docs/OPENAI-CHAT-API-PLAN.md`
  §Chat Template Handling) is the real fix, CLI-side stripping is a stopgap.
