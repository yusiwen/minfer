# minfer Debug Dump Mechanism

> Feature: `--features debug_dump`
> Controlled by: `MINFER_DUMP_DIR` environment variable

## Overview

The `debug_dump` feature writes raw f32 binary files of hidden states and logits at specific points during inference. Combined with the `scripts/compare_layers.py` tool (which compares against llama.cpp reference dumps), this enables precise layer-by-layer numerical debugging.

**Performance**: When not enabled (`cargo build --release`), all dump code is eliminated at compile time — zero instructions, zero branches. When enabled, overhead is one `OnceLock` env var read + file write per dump point.

---

## Dump Points

```
│  forward pass
│
├─ ⓪ minfer_dump_prompt.txt ─────────────────────────────── text dump, NOT a hidden state
│     file: minfer_dump_prompt.txt
│     dump: rendered chat template prompt (for validation against llama prompt)
│
├─ ① embed_out ─────────────────────────────────────────── token_embd (Q5_0 dequant) → hidden
│     file: minfer_dump_embed_out.f32
│     shape: [nt * ne]   (nt = num tokens, ne = hidden dim)
│     dump: embedding lookup output, before any transformer layers
│
└─  for each layer N (0..23):
      │
      ├─ QKV matmul: bn × WQ/WK/WV → bq, bk, bv           [Q5_0 dequant × 3]
      ├─ add_bias(bq) / add_bias(bk) / add_bias(bv)
      ├─ RoPE(bq, bk)
      ├─ store KV cache
      ├─ GQA attention (f32)
      ├─ WO matmul: ba × WO → bn                            [Q5_0 dequant]
      ├─ residual: hidden += bn
      │
      ├─ ② layer{N}_attn_out ──────────────────────────── attention output
      │     file: minfer_dump_layer{N}_attn_out.f32
      │     shape: [nt * ne]
      │     dump: hidden state after attention branch, before FFN
      │
      ├─ RMSNorm(hidden, ffn_norm) → ffn_in
      ├─ gate/up matmul: ffn_in × Wg/Wu → bg, bf           [Q5_0 dequant × 2]
      ├─ SwiGLU: silu(bg) * bf → bg
      ├─ down matmul: bg × Wd → bn                          [Q5_0 / Q4_K / Q6_K dequant]
      ├─ residual: hidden += bn
      │
      └─ ③ layer{N}_out ───────────────────────────────── FFN output
            file: minfer_dump_layer{N}_out.f32
            shape: [nt * ne]
            dump: hidden state after full layer (attention + FFN + residuals)

   after all layers:

   ├─ RMSNorm(hidden, output_norm) → bn
   ├─ LM head: bn × output.weight → logits                 [Q8_0 matmul]
   ├─ add output_bias
   │
   └─ ④ logits ────────────────────────────────────────── final logits
         file: minfer_dump_logits.f32
         shape: [nt * n_vocab]
         dump: raw logits before sampling
```

---

## minfer Side

### Feature flag

```toml
# Cargo.toml
[features]
debug_dump = []
```

### Core module: `src/dump.rs`

```rust
// For float arrays (hidden states, logits)
pub fn maybe_dump(name: &str, data: &[f32])

// For text (prompts)
pub fn maybe_dump_text(name: &str, text: &str)
```

- Both controlled by `MINFER_DUMP_DIR` env var
- If not set → no-op (returns immediately)
- If set → `maybe_dump` writes raw f32 bytes to `{MINFER_DUMP_DIR}/{name}.f32`
- If set → `maybe_dump_text` writes UTF-8 string to `{MINFER_DUMP_DIR}/{name}.txt`
- Uses `OnceLock` to cache the env var check (read once, reused)

### Build & run

```bash
# Normal build — zero overhead, dump code eliminated at compile time
cargo build --release

# Debug build — dump enabled
cargo build --release --features debug_dump

# Run with CPU path + dump
MINFER_DISABLE_MPS=1 MINFER_DUMP_DIR=/tmp \
  cargo run --release --features debug_dump -- <model> "Hello"
```

---

## Python Side

### `scripts/dump_llama_ref.py`

Generates llama.cpp reference hidden states for comparison.

```bash
# Bare text (no chat template wrapping)
uv run python -m scripts.dump_llama_ref \
  --model <path-to-gguf> \
  --prompt "Hello" \
  --output ./llama_ref

# With chat template — reads tokenizer.chat_template from GGUF,
# renders with Jinja2, producing the same prompt minfer would use
uv run python -m scripts.dump_llama_ref \
  --model <path-to-gguf> \
  --prompt "Hello" --chat \
  --output ./llama_ref
```

**Output** (per layer):
| File | Shape | Content |
|------|-------|---------|
| `layer{N}_hidden_states.npy` | [hidden_dim] | Last-token hidden state after N layers |
| `logits_prefill.npy` | [vocab_size] | Final logits |
| `token_ids.npy` | [seq_len] | Input token IDs |
| `prompt.txt` | — | Rendered prompt text (for validation against minfer's `minfer_dump_prompt.txt`) |

**How it works**: Uses the "truncated model" technique — creates a fake GGUF with `block_count=N` (metadata only, zero-copy weights), runs `llama.eval()`, and extracts the embedding output via `llama_get_embeddings()`. Since a decoder-only transformer is feed-forward, the first N layers produce identical outputs to a full model.

When `--chat` is specified, the script reads `tokenizer.chat_template` from GGUF metadata, renders it with Jinja2 using `messages=[{"role":"user", "content": prompt}]` and `add_generation_prompt=True`, producing a prompt identical to minfer's chat template rendering.

### `scripts/compare_layers.py`

Compares minfer dumps against llama.cpp reference.

```bash
uv run python -m scripts.compare_layers \
  --llama-dir ./llama_ref \
  --minfer-dir /tmp \
  --hidden-dim 896
```

**Layer mapping**: llama layer N (N=1..24) = minfer layer N-1 (0..23). Both dump the hidden state AFTER the corresponding layer's computation.

**Prompt validation**: Before layer comparison, the script reads `prompt.txt` (from llama reference) and `minfer_dump_prompt.txt` (from minfer dump). If both exist and differ, the script immediately aborts with `PROMPT MISMATCH — comparison aborted` and `sys.exit(1)`.

**Output**: Per-layer:

| Column | Meaning |
|--------|---------|
| `minfer RMS` | RMS of minfer's hidden state |
| `llama RMS` | RMS of llama.cpp's hidden state |
| `ratio` | RMS ratio (should be ~1.000) |
| `cos` | Cosine similarity (≥0.999 = match) |

Plus logits comparison with top token and cosine.

---

## Full Workflow

```bash
# Step 1: Generate llama.cpp reference (one-time, ~20 min for 24-layer model)
#         --chat ensures the SAME prompt as minfer (rendered from tokenizer.chat_template)
uv run python -m scripts.dump_llama_ref \
  --model ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "Hello" --chat --output ./llama_ref

# Step 2: Run minfer with debug dump (CPU path)
#         MINFER_DUMP_DIR enables all dump points including prompt
MINFER_DISABLE_MPS=1 MINFER_DUMP_DIR=/tmp \
  cargo run --release --features debug_dump \
  -- <model> "Hello"

# Step 3: Compare
#         Script validates prompt files match before comparing layers
uv run python -m scripts.compare_layers \
  --llama-dir ./llama_ref --minfer-dir /tmp --hidden-dim 896
```

---

## Diagnostic Logic

Given `compare_layers.py` output showing first divergence at some layer, use the sequence below to isolate the bug:

| Divergence pattern | Llama ref | Minfer dump | Diagnosis |
|---|---|---|---|
| Prompt mismatch | `prompt.txt` | `minfer_dump_prompt.txt` | **Template rendering or tokenizer differs** between llama and minfer |
| `embed_out` already diverged | `layer1_hidden` | `embed_out` | **Q5_0 embedding dequant is wrong** |
| `layerN_out` diverged, `layerN_attn_out` ok | `layer{N}_hidden` vs `layer{N-1}_hidden` | `attn_out` vs `layer_out` | **FFN matmul is wrong** (gate/up/down Q5_0 dequant) |
| `layerN_attn_out` already diverged | `layer{N-1}_hidden` | `layer{N-1}_out` vs `layer{N}_attn_out` | **Attention matmul is wrong** (Q/K/V/WO Q5_0 dequant) |
| 24 layers all match, logits diverge | `layer24_hidden` | `layer23_out` | **output.weight (Q8_0) matmul is wrong** |
| All cosine ≥ 0.999, logits match | — | — | **Bug is elsewhere**: sampler, or model architecture hparams mismatch |

### Layer numbering

| llama.cpp dump | minfer dump | Computation |
|----------------|-------------|-------------|
| `layer1_hidden_states.npy` | `minfer_dump_layer0_out.f32` | Hidden after layer 0 (attention + FFN) |
| `layerN_hidden_states.npy` | `minfer_dump_layer{N-1}_out.f32` | Hidden after layer N-1 |
| `logits_prefill.npy` | `minfer_dump_logits.f32` | Final logits |

---

## Output Files (minfer)

| File | Dump point | Shape | Bytes (Qwen2.5-0.5B, prompt="Hello"≈30 tokens) |
|------|-----------|-------|------|
| `minfer_dump_prompt.txt` | ⓪ prompt | text | ~200 B |
| `minfer_dump_embed_out.f32` | ① embedding | 30 × 896 = 26,880 | ~105 KB |
| `minfer_dump_layer0_attn_out.f32` | ② attention | 26,880 | ~105 KB |
| `minfer_dump_layer0_out.f32` | ③ FFN | 26,880 | ~105 KB |
| ... | | | |
| `minfer_dump_layer23_attn_out.f32` | ② attention | 26,880 | ~105 KB |
| `minfer_dump_layer23_out.f32` | ③ FFN | 26,880 | ~105 KB |
| `minfer_dump_logits.f32` | ④ logits | 30 × 151,936 = 4,558,080 | ~17.4 MB |

Total: 24 × 2 × 105 KB + 17.4 MB ≈ **22 MB** for a 24-layer model.
