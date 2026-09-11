# 11 · Attention, vec ops, and the KV cache

> **Stage**: CPU matmul kernels (doc 10) → **this stage: every non-matmul op of the layer** → the sampler (doc 12).
> **Code**: `src/vec_ops.rs` (`rms_norm_f32`, `vec_soft_max_f32`, `vec_silu_f32`, `mat_mul_f32`), `src/graph/cpu_backend.rs` (`cpu_rope`:469, the `KvcacheStore`/`KvcacheLoad`/`Attn` arms, `cpu_gqa_attn`:503, `attn_heads`:572), `src/graph/builder.rs:323` (the `attn` node contract).

## 1. Background — where this stage sits

Doc 05 built the layer as a graph; docs 08–10 executed its matmuls. What is left is everything a transformer does *between* and *around* those matmuls — and one op that is unlike any other in the network:

- **RMSNorm** — rescale each token's vector so its root-mean-square is 1, then apply learned per-dimension gains. Runs twice per layer, plus a final one before the output matmul.
- **RoPE** — rotate the query and key vectors by an angle that depends on the token's *position*. This is how a transformer without recurrence knows word order.
- **Attention** — the op the whole architecture orbits: every token *looks at* other tokens and mixes their information. It is the only op whose inputs reach across positions, which is why the engine needs a KV cache, a causal mask, and positions-as-data (docs 05/07/09 set those up; this doc shows the mechanics).
- **SiLU and the elementwise glue** — add (residuals), mul, scale, softmax: short vector passes that doc 06's fusion pass already minimized.

For a beginner, the mental model of one decoder layer is:

```text
h ──► RMSNorm ──► [W_q W_k W_v matmuls] ──► RoPE(q, k) ──► store K,V into the cache
                                                                │
              ┌─────────────────────────────────────────────────┘
              ▼
   attention: each query token reads the cached K/V of its allowed prefix
              ──► [W_o matmul] ──► (+ residual) ──► RMSNorm ──► FFN ──► (+ residual) ──► h'
```

This document walks the four mechanisms in order: RMSNorm (§3.2.1), RoPE (§3.2.2), the KV store/load pair (§3.2.3), and attention (§3.2.4), each with the actual code and a worked numeric example. Everything here is the **CPU** implementation; the GPU backends run the same math with different execution models (docs 14/15).

One modern-model wrinkle to know up front: Qwen3 makes the query head dimension (`hd`) and the key/value head dimension (`hd_kv`) **decoupled** — they can differ per model (doc 03's loader). Every function in this doc carries both (`hd` and `hd_kv` parameters) and `cpu_gqa_attn` refuses (`Err`) any configuration where `hd < hd_kv`, because a query head shorter than its key cannot be dotted against it. Qwen2/Qwen2.5 models simply have `hd == hd_kv`.

## 2. Principle — attention from zero

### 2.0 The residual stream: why the norm sits *before* each block

One piece of architecture context explains half the ops in this doc. A decoder layer does not compute `h' = ffn(attn(h))` — it computes:

```text
h = h + attn_out( rms_norm(h) )      // attention sub-block, pre-norm
h = h + ffn_out( rms_norm(h) )      // FFN sub-block, pre-norm
```

This is called **pre-norm** (the norm comes before the attention/FFN instead of after), and the running `h` that both sub-blocks read and *add onto* is called the **residual stream**: a highway where each layer's contribution is an *update*, not a replacement. Two consequences worth knowing:

- If any sub-block is useless for a given token, the layer can output it with (near-)zero weight and the token's information passes through untouched — this is why very deep models train stably at all.
- The two `+` operations are `Op::Add` nodes (doc 05), the plainest vector op in the engine — and they are why doc 06's fusion pass cares so much about elementwise buffer reuse: in the 0.5B graph the residual adds are among the most frequent nodes.

### 2.1 What attention computes

A transformer represents each token as a vector (the *hidden state*, dimension `d = n_embd`). Attention lets token *t* build a new representation out of the other tokens' vectors. Mechanically, each token's vector is projected three ways — by three of doc 10's matmuls:

- a **query** vector `q` — "what am I looking for?" (`W_q`),
- a **key** vector `k` — "what do I contain?" (`W_k`), and
- a **value** vector `v` — "what do I hand over if someone attends to me?" (`W_v`).

The projections are split into **heads** — `n_head` independent attention channels of dimension `hd` each (`n_embd = n_head × hd`; Qwen2.5-0.5B: 14 × 64 = 896). Heads attend in parallel and are concatenated before the output projection `W_o`.

Then, for query token `t` in head `h`:

1. **Score** every earlier-or-equal token `s` (`s ≤ t`): `score[t][s] = q_t · k_s × scale`, with `scale = 1/√hd`. The dot product measures similarity between what `t` seeks and what `s` offers; the scale keeps the numbers in a range where softmax behaves (a raw dot over `hd` elements grows like `hd`, and softmax over huge numbers degenerates).
2. **Mask** the future: `s > t` is forbidden (a token must not see tokens that come after it — that is the *causal* property that makes one pass over the prompt meaningful). Implemented not by a mask matrix but by *length*: `s` only ranges over `[0, pos[t]]` (§3.2.4).
3. **Softmax** the scores into weights that sum to 1.
4. **Output** the weighted sum of the values: `out_t = Σ_s weight[t][s] · v_s`.

The matmuls are doc 10's job. This doc is steps 1–4 plus the ops that condition the inputs (RMSNorm, RoPE).

### 2.2 Why the KV cache exists

Consider decode, step 100 (doc 13): one query token needs scores against the keys of tokens 0..99. The keys and values of tokens 0..98 were *already computed* on earlier steps — recomputing them would mean re-running `W_k`/`W_v` over the entire history every step, making step 100 cost as much as prefill of 100 tokens. Instead, the engine **stores every token's K and V rows as they are produced** and attention reads them back.

That is the `kvcache_store` / `kvcache_load` node pair from doc 05, backed by doc 07's persistent per-layer regions: each layer owns two buffers (K and V) of `n_kv_embd × n_ctx` floats, and the store node writes this step's rows *at the positions the `positions` input carries*. "KV positions are data" — the write offset comes from an input, not from any internal counter — is what lets the same graph serve prefill (write 100 positions), decode (write 1 position), and multi-turn continuation (write positions 100..120 — doc 13's conversation path).

The cost arithmetic, concretely for Qwen2.5-0.5B (24 layers, `n_kv_embd = 128`): one position costs `128 floats × 4 B × 2 (K and V) × 24 layers = 24 KB` — the number doc 09 quoted. For Qwen3-4B (36 layers, `n_kv_embd = 1024`): `1024 × 4 × 2 × 36 = 288 KB` per position, i.e. **33.6 MB per layer at `n_ctx = 4096`** (doc 07's number). GQA (§2.4) is what keeps that from being much larger.

### 2.3 RoPE: position without position embeddings

Dot products are permutation-invariant: shuffle the input tokens and every dot product stays the same. A transformer needs some way to know that "dog bites man" differs from "man bites dog". Older models added a fixed position vector to each token's embedding; modern LLM-family models use **rotary position embeddings (RoPE)**: rotate each `q` and `k` vector by an angle proportional to its position, in consecutive 2-dimensional subspaces.

The property that makes it work: for a query at position `m` and a key at position `n` rotated in the same 2-D subspace, their dot product depends only on `m − n` — the *relative* distance — because rotating both vectors by their own angles leaves a net rotation of `m − n` between them. Attention scores then encode "how far apart are these tokens", which generalizes to positions never seen in training far better than absolute embeddings.

The angles: in subspace `i` (of `hd/2`), the angular speed is `freq_i = 1 / base^(2i/hd)` — low subspaces rotate fast (fine, local position distinctions), high subspaces rotate slowly (coarse, long-range distinctions). `base` is `freq_base` from the model's hyperparameters (e.g. 1,000,000 for Qwen2.5 — doc 03's `HParams`). This is also why one long-context trick is simply scaling those frequencies (`freq_scale` in the RoPE meta — NTK/RoPE-scaling territory).

### 2.4 GQA: many query heads share few key/value heads

Attention runs in parallel **heads**: `n_head` independent query/attention channels, each with its own smaller dimension `hd`. Classic multi-head attention gives every query head its own K/V heads, so the KV cache costs `n_head × hd` per token per region.

**Grouped-query attention (GQA)** shares: `n_head_kv` K/V heads serve all `n_head` query heads, with query head `h` reading kv head `hk = h / (n_head / n_head_kv)`. Qwen2.5-0.5B: 14 query heads, 2 kv heads → each kv head serves 7 query heads → the KV cache is **7× smaller** than classic MHA with negligible quality loss (the models are trained that way). The code and the cache arithmetic both inherit this: the K/V regions are `n_kv_embd = n_head_kv × hd_kv` wide (128 floats for 0.5B), and doc 07's per-position byte cost follows from it.

The attention scale is defined per model next to the head geometry it depends on:

```rust
// src/models/qwen2/loader.rs:36-38 (Qwen3 has its own at loader.rs:48)
pub fn attention_scale(&self) -> f32 {
    1.0 / (self.n_embd_head() as f32).sqrt()
}
```

and flows into the graph as `attn_scale` (`models/qwen2/graph.rs:76`), landing in every `AttnMeta` (`graph.rs:200-207`). Doc 05 covered the meta's plumbing; here it is the `c.scale` of §3.2.4.

### 2.5 What changes on the GPU (preview)

Docs 14/15 run the *same* formulas with different execution models; three deltas to keep in mind so nothing here surprises you later:

- **Storage**: Metal/CUDA may keep the K/V regions in `f16` (`MINFER_CACHE_TYPE=f16`), halving the bandwidth of §2.2's arithmetic; the CPU path stays `f32` (§3.3).
- **Shape**: the GPU attention kernels tile the (query, key) matrix and apply the softmax *online* — max and sum accumulate block-by-block instead of one full pass — the "flash attention" trick; the CPU path computes full rows because everything already fits in cache.
- **Parallelism axis**: CPU splits by head (§3.3); CUDA additionally splits the KV dimension across blocks and reduces (`split-KV`), because a GPU has thousands of threads and only 14–40 heads to give them.

The invariants survive all three deltas: positions are data, windows come from `pos[t]+1`, and store-before-attention still holds.

## 3. Implementation

### 3.1 Data in / data out

| Op | Reads | Writes |
|---|---|---|
| `RmsNorm` | `[nt][d]` f32 + `d` gains | `[nt][d]` f32 (aliases its input — doc 07) |
| `RoPE` | q or k `[nt][n_head×hd]`, positions `[nt]` (I32-as-f32) | same buffer, rotated in place (aliases — doc 07) |
| `KvcacheStore` | k `[nt][nkt]`, v `[nt][nkt]`, positions | the layer's persistent K/V regions at those positions |
| `KvcacheLoad` | nothing (a *view* of the K region) | nothing |
| `Attn` | q `[nt][n_head·hd]`, the K/V regions, positions | `[nt][n_head·hd]` f32 |
| `Softmax` / `SiLU` / `Add` / `Mul` | elementwise f32 | elementwise f32 |

`nkt` is the K/V row width (`n_kv_embd`); positions ride as I32-in-f32 bit patterns (doc 07's `fill_input_i32`), and every arm decodes them with `.to_bits() as usize`.

### 3.2 Key code

#### 3.2.1 RMSNorm — `vec_ops.rs:519`

The formula: `y = x / sqrt(mean(x²) + eps) × weight`. The scalar fallback shows every piece:

```rust
// src/vec_ops.rs:519-545 (scalar path; AVX2 path at :547)
pub fn rms_norm_f32(n: usize, y: &mut [f32], x: &[f32], eps: f32) {
    let mut sum_sq = 0.0f64;
    for i in 0..n {
        sum_sq += (x[i] as f64) * (x[i] as f64);       // ① sum of squares, f64
    }
    let mean = (sum_sq / n as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();             // ② 1/rms(+eps)
    if y.as_ptr() != x.as_ptr() {
        vec_cpy_f32(n, y, x);                          // ③ skip the copy when aliased
    }
    vec_scale_f32(n, y, scale);                        // ④ normalize
    ...                                                // ⑤ multiply by gains (weight)
}
```

Design notes a beginner should keep:

- **No mean subtraction.** Classic LayerNorm subtracts the mean before normalizing; RMSNorm skips it (`mean of squares` directly). One less pass over the vector, and empirically it works as well — llama.cpp's models all use it, and minfer matches them op-for-op. `eps` comes from the model hyperparameters (`f_norm_rms_eps`, doc 03) and just keeps `sqrt` away from zero for an all-but-zero vector.
- **The accumulation is f64** (①) — 896-wide sums in f32 would lose real precision; the AVX2 path keeps the same f64 accumulator semantics so both paths agree.
- **Line ③ is doc 07's aliasing made visible**: the fusion/allocator pass maps RMSNorm's output onto its input's buffer where legal, and the copy self-suppresses by pointer comparison. `rms_norm_fused_f32` (`vec_ops.rs:589`) is the variant that also multiplies the gains in one pass.

Worked example: `x = [1, 2, 3, 4]`, `eps ≈ 0`. `mean(x²) = (1+4+9+16)/4 = 7.5`; `scale = 1/√7.5 ≈ 0.365`; normalized `x ≈ [0.365, 0.730, 1.095, 1.461]`, then element-wise multiplied by the learned gains.

#### 3.2.2 RoPE — the node arm (`cpu_backend.rs:343`) and `cpu_rope` (`:469`)

The `RoPE` node arm shows the two conventions this series keeps meeting — positions decoded from f32 bits, and in-place execution:

```rust
// src/graph/cpu_backend.rs:343-356 (abridged)
Op::RoPE { style } => {
    let meta = match &node.meta { NodeMeta::Rope(m) => m, ... };
    let nh = meta.n_head;
    let hd = meta.hd;
    let nt = node.out_shape[1];
    // positions are I32 bit patterns in ins[1]
    let pos: Vec<usize> = (0..nt).map(|t| ins[1][t].to_bits() as usize).collect();
    out.copy_from_slice(ins[0]);                 // free when aliased (doc 07)
    cpu_rope(out, &pos, nh, hd, meta.freq_base, meta.freq_scale, *style);
    Ok(())
}
```

`cpu_rope` then does the rotation of §2.3:

```rust
// src/graph/cpu_backend.rs:469-501 (core loop, abridged)
pub(crate) fn cpu_rope(x: &mut [f32], pos: &[usize], nh: usize, hd: usize,
                       freq_base: f32, freq_scale: f32, style: RopeStyle) {
    let half = hd / 2;
    let mut freqs = [0.0f32; 128];
    for i in 0..half {
        freqs[i] = freq_scale / freq_base.powf((2 * i) as f32 / hd as f32);  // ① angular speeds
    }
    for t in 0..pos.len() {
        let p = pos[t] as f32;
        for h in 0..nh {
            let b = t * nh * hd + h * hd;
            for i in 0..half {
                let th = p * freqs[i];                       // ② this subspace's angle
                let (sn, cs) = th.sin_cos();
                let (i0, i1) = match style {
                    RopeStyle::NonInterleaved => (b + i, b + i + half),  // ③ Qwen2 layout
                    RopeStyle::Interleaved   => (b + 2*i, b + 2*i + 1),  //    Llama layout
                };
                let (x0, x1) = (x[i0], x[i1]);
                x[i0] = x0 * cs - x1 * sn;                   // ④ 2-D rotation
                x[i1] = x0 * sn + x1 * cs;
            }
        }
    }
}
```

- ①: the angular speeds — subspace 0 spins fastest, the last subspace slowest (§2.3). The table holds `hd/2` entries; the fixed `[128; ...]` bound covers every supported head dim.
- ②: the angle is `position × speed` — position enters *only* here, from the `pos` input.
- ③: the two **layout styles** are a pure memory convention — which two slots form a rotating pair. Qwen2/Qwen3 put the pairs at `(i, i+half)` (all "first halves" contiguous — GGUF's NEOX convention); Llama interleaves `(2i, 2i+1)`. Same rotation math, different addresses; `ModelDef::rope_style()` (doc 03) picks per model, and getting it wrong silently scrambles position information (it is one of the first things the Qwen2.5 bring-up docs checked — see `docs/DEBUGGING-PLAN.md` H1).
- ④: the 2-D rotation, straight from §2.3.

Worked example: `hd = 4` (`half = 2`), `base = 10⁴` → `freqs = [1.0, 10⁻¹]`. Token at `pos = 2`, pair 0: angle `2×1.0 = 2 rad` → `(x0,x1) ← (x0·cos2 − x1·sin2, x0·sin2 + x1·cos2)`. Pair 1 rotates 10× slower — exactly the "fine vs coarse" structure of §2.3.

#### 3.2.3 The KV store / load pair — `cpu_backend.rs:143` and `:381`

The store arm is where "KV positions are data" becomes memory writes:

```rust
// src/graph/cpu_backend.rs:143-176 (core, abridged)
if let Op::KvcacheStore { layer } = &node.op {
    let (k_id, v_id) = kv_pair.ok_or_else(...)?;          // doc 07's persistent regions
    if k_id != out_buf { return Err("KV store out buffer must be the K region".into()); }
    let nkt = node.out_shape[0];                           // K/V row width (n_kv_embd)
    let n_ctx = node.out_shape[1];
    let nt = self.buffers[in_bufs[0]].len() / nkt;
    let pos: Vec<usize> = self.buffers[in_bufs[2]].iter()
                              .map(|b| b.to_bits() as usize).collect();  // I32-as-f32
    let (k_dst, v_dst): (&mut [f32], &mut [f32]) = /* split_at_mut over the two regions */;
    for t in 0..nt {
        let p = pos[t];
        if p >= n_ctx {
            return Err(format!("KV store position {p} >= n_ctx {n_ctx}"));  // ① hard error
        }
        let ks = p * nkt;                                  // ② THE write offset
        k_dst[ks..ks + nkt].copy_from_slice(&k_src[t * nkt..(t + 1) * nkt]);
        v_dst[ks..ks + nkt].copy_from_slice(&v_src[t * nkt..(t + 1) * nkt]);
    }
}
```

Two things to internalize:

- ② is the entire KV cache write: a `memcpy` of one K row and one V row to `position × row_width` inside the layer's persistent regions. No scaling, no math — the K/V rows arriving on `in_bufs[0..1]` were already computed by the `W_k`/`W_v` matmuls and rotated by RoPE.
- ① is the doc 08 error contract in miniature: a position out of range is a **topology/contract violation**, so it returns `Err` and aborts the run — it never silently clamps and continues, because a clamped write would corrupt a *different* token's cache row and you would see it a thousand tokens later as subtly wrong text.
- The **load** arm is one line (`cpu_backend.rs:381`): `Op::KvcacheLoad { .. } => Ok(())`. The load node's output buffer *is* the K region (doc 07 mapped it directly), so "loading" is a bookkeeping view — no data moves. The scheduler's build-order guarantee (doc 08) is what makes the view safe: this layer's store always executes before this layer's attention.

#### 3.2.4 GQA attention — the `Attn` arm (`cpu_backend.rs:388`) and `attn_heads` (`:572`)

The node arm resolves the regions and computes one derived quantity — the *current* KV length:

```rust
// src/graph/cpu_backend.rs:388-428 (core, abridged)
Op::Attn { .. } => {
    let meta = ...;                                        // AttnMeta: n_head, hd, nkt, scale, layer
    let (k_id, v_id) = kv_pair.ok_or_else(...)?;           // this layer's K and V regions
    let k_slice /*, v_slice */ = /* the two persistent regions, borrowed */;
    let n_ctx = k_slice.len() / nkt;
    let nkv = (0..nt).map(|t| ins[2][t].to_bits() as usize + 1).max()   // ① max position + 1
                     .unwrap_or(0).min(n_ctx);
    let pos: Vec<usize> = (0..nt).map(|t| ins[2][t].to_bits() as usize).collect();
    cpu_gqa_attn(ins[0], k_slice, v_slice, &pos, nt, nkv,
                 meta.n_head, meta.n_head_kv, meta.hd, meta.hd_kv, nkt, out, meta.scale)?;
}
```

① is the causal mask in data form: the attention window is `max(positions) + 1` — during prefill of 100 tokens that is 100; during decode at position 100 it is 101. There is no mask tensor anywhere; the window *is* the mask, derived from the same positions input the store used. (builder.rs:323-325 documents the contract: `vl = pos[t]+1`.)

`cpu_gqa_attn` (`:503`) validates the head geometry (`hd < hd_kv` → `Err` — the Qwen3 decoupled-dims guard from §1), then farms head ranges to the same thread pool as the matmuls (`kernel::par_for`, doc 10), each worker with a private scores buffer. Heads never reduce against each other → bit-identical at any thread count, the same invariant as doc 10's row ownership. Single-threaded fallback when `--threads 1` or `n_head < 2`.

The per-head worker does the math of §2.1:

```rust
// src/graph/cpu_backend.rs:572-625 (core loop, abridged; runs per head range via par_for)
let gqa = c.nh / c.hk;                                     // e.g. 14/2 = 7
for h in h0..h1 {                                          // this worker's query heads
    let hk = h / gqa;                                      // ① my shared kv head
    for t in 0..c.nt {
        let vl = (*c.pos.add(t) + 1).min(c.nkv);           // ② causal window for token t
        for kv in 0..vl {
            let s = vec_dot_f32(c.hd_kv, &q_row, &k_row(kv)) * c.scale;   // ③ score
            scrs[kv] = s; if s > mx { mx = s; }            //    (max tracked on the fly)
        }
        for kv in vl..c.nkv { scrs[kv] = f32::NEG_INFINITY; }             // ④ mask = −∞
        let sm = vec_soft_max_inplace_f32(c.nkv, &mut scrs, mx);          // ⑤ softmax
        vec_scale_f32(c.nkv, &mut scrs, (1.0 / sm) as f32);
        out_row.fill(0.0);
        for kv in 0..c.nkv {                                              // ⑥ weighted V sum
            vec_muladd_f32(c.hd_kv, out_row, &v_row(kv), scrs[kv]);
        }
    }
}
```

Walk it against §2.1:

- ①: GQA's only appearance in the code — one integer division mapping query head → kv head (0.5B: heads 0–6 read kv head 0, heads 7–13 read kv head 1).
- ②③: scores are `q·k × scale` over the causal window `vl = pos[t]+1`.
- ④: tokens beyond the window get `-INF` scores rather than a shorter loop — see ⑥ for why that is safe *and* cheap.
- ⑤: softmax with max subtraction (§3.2.5). Weights now sum to 1.
- ⑥: the output is the weighted sum of V rows. The loop runs over **all** `nkv` rows, including masked ones — their softmax weight is exactly `exp(−∞ − mx) = 0.0`, and `0.0 × v = 0.0` contributes nothing. Unwritten region bytes would be zeros even if read, so there is no uninitialized-memory hazard either.

Worked example (one query head, `hd_kv = 4`, GQA 1:1 for simplicity). Query at `pos = 1` (the second token); cache holds K rows `k0 = [1,0,0,0]`, `k1 = [0,1,0,0]`, V rows `v0 = [1,2,0,0]`, `v1 = [3,4,0,0]`; `q = [1,1,0,0]`, `scale = 1/√4 = 0.5`; suppose the KV region is sized for 3 (`nkv = 3`) so there is a *masked* third slot:

```text
vl = pos+1 = 2                          (token 1 may see tokens 0 and 1)
score0 = q·k0 × 0.5 = 1 × 0.5 = 0.5
score1 = q·k1 × 0.5 = 1 × 0.5 = 0.5
score2 = −∞                             (④ masked: beyond the causal window)
softmax([0.5, 0.5, −∞]) = [0.5, 0.5, 0.0]
out = 0.5·v0 + 0.5·v1 + 0.0·v2 = [2.0, 3.0, 0.0, 0.0]
```

A query aligned equally with both cached keys blends both values. Change the query to `[1,0,0,0]` and score0 wins (0.5 vs 0.0 after scale → softmax ≈ [0.73, 0.27, 0.0]) and the output tilts toward `v0` — that *tilting* is, mechanically, all "attention" is.

**The same example one step later (decode).** Suppose step 3 now emits token at `pos = 2` with `q2 = [0,1,1,0]`; the store arm appends `k2 = [1,1,0,0]`, `v2 = [5,0,0,0]` at offset `2 × nkt` (② in §3.2.3); the attention arm re-derives `nkv = max(2)+1 = 3`. Nothing about the graph changed — the *positions input* grew by one element, and the window grew with it:

```text
token 0 sees:  vl = 1  →  [w0]                     (still blind to 1, 2)
token 1 sees:  vl = 2  →  [w0, w1]
token 2 sees:  vl = 3  →  [w0, w1, w2]
score2·k0 = 0×0.5 = 0.0 · k1 = 1×0.5 = 0.5 · k2 = 1×0.5 = 0.5
softmax([0.0, 0.5, 0.5]) ≈ [0.27, 0.37, 0.37]
out2 ≈ 0.27·v0 + 0.37·v1 + 0.37·v2
```

That is decode in miniature: one new K/V row per step, windows growing monotonically, and every past token's cached rows read again without recomputation — the entire reason §2.2's cache exists. (Doc 13 shows the loop that drives it and doc 07 the regions that hold it.)

#### 3.2.5 Softmax and SiLU — `vec_ops.rs:208`, `:158`

```rust
// src/vec_ops.rs:208-226 (scalar path; caller supplies the max)
pub fn vec_soft_max_f32(n: usize, y: &mut [f32], x: &[f32], max: f32) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..n {
        let val = (x[i] - max).exp();      // ① subtract max BEFORE exp
        y[i] = val;
        sum += val as f64;                 // ② f64 sum, returned for normalization
    }
    sum
}
```

① is the numerical-stability trick the whole series keeps meeting: `softmax` is invariant under subtracting any constant, but `e^1000` overflows `f32` while `e^(1000−1000) = 1` does not. So every caller finds the max first — the standalone `Softmax` node arm (`cpu_backend.rs:357-368`) shows the full ceremony (scan for max → copy → softmax → caller divides by the returned sum), while attention's hot path uses the in-place variant (`vec_soft_max_inplace_f32`, `:276`) since its scores buffer is scratch anyway. The `f64` sum ② then normalizes without precision loss.

SiLU is one formula — `silu(x) = x / (1 + e^(−x))` (`vec_silu_f32`, `:158`, scalar shown):

```rust
for i in 0..n { y[i] = x[i] / (1.0 + (-x[i]).exp()); }
```

It is the FFN's activation (doc 05's `silu(gate) × up`); doc 06 fused it into `SwiGLU`, whose CPU execution (two passes over `vec_silu_f32` then `vec_mul_f32`) you saw in doc 06 §3. The `add`/`mul`/`scale`/`muladd` helpers are the same pattern — a short SIMD-able loop each — and `attn_heads` ⑥ uses `vec_muladd_f32` for the weighted-V accumulation. The f32 weights path (`mat_mul_f32`, `vec_ops.rs:673`) closes the loop back to doc 10: norm biases and F32 tensors skip quantization entirely and use this plain dot-product matmul.

### 3.3 Design choices (why this shape and not another)

**Why is attention the op everything else serves?** Every matmul is per-token: token *t*'s matmul output depends only on token *t*'s input row. Attention is the only op whose output depends on **other tokens** — that is where the model's ability to relate words to each other lives, and it is the only reason the engine needs positions, a causal window, and a cross-call cache. Remove attention and the remaining stack is just per-token transforms that a single forward could do in any order.

**Why length-as-mask instead of a mask matrix?** llama.cpp-style engines could build an `[nt, nt]` additive mask; minfer derives the window from positions (`nkv = max(pos)+1`, `vl = pos[t]+1`). That is cheaper (no mask buffer, no per-score mask add), it composes with multi-turn continuation for free (a resumed sequence's positions just continue — doc 13), and it keeps the "positions are data" invariant doing double duty. The −∞ tail (④) is the one concession, and it exists so the softmax normalize step can treat scores as one fixed-length buffer.

**Why store K/V at input-carried positions instead of an internal `n_past` counter?** A counter inside the store would make the graph's *behavior* depend on hidden execution state — the second decode step would write somewhere else than the first, with identical topology. With positions as data, the same rebuilt graph is correct for prefill, for decode, and for resuming a saved conversation (doc 13's `conversation.rs` just continues the position sequence); doc 07's reuse invariant stays intact.

**Why compute scores in f32 (`vec_dot_f32`) instead of the int8 trick?** Doc 10's Q8_0 machinery quantizes *matmul activations*; attention scores are computed once per (query, key) pair, and quantizing q/k per score would cost more than it saves — the dot is over `hd_kv` (≤128) elements, not `n_embd`. The K/V *storage* is where bandwidth matters (hence the GPU f16 cache, below), not the score math on CPU.

**Why is the KV cache f32 on CPU while GPUs offer f16?** The CPU path never re-quantizes attention inputs; f16 K/V on CPU would add a conversion per score for negligible bandwidth win at these sizes. The GPU backends, where bandwidth per token is the budget, do offer the f16 cache (`MINFER_CACHE_TYPE=f16`, docs 03/14). Same invariant — regions are persistent and position-addressed — different storage per backend.

**Why parallelize over heads instead of over tokens?** Heads are perfectly independent (① is the only cross-head coupling, and it is read-only), so the split has zero communication; splitting over tokens would share each head's `scrs` buffer across workers. It also composes with decode: at `nt=1` the token axis is empty, but 14 heads still parallelize the dot products.

**Why refuse `hd < hd_kv` instead of supporting it?** A query head shorter than its key cannot be dotted against it without a padding convention; no supported model needs it (Qwen3's decoupling goes the *other* way where relevant), and a silent pad would hide model-definition mistakes. Fail loudly (doc 08's error contract).

### 3.4 Pitfalls & invariants

- **`positions[i] < n_ctx` is a caller obligation** — the store arm enforces it with `Err` (① in §3.2.3); doc 09 showed the CLI's double clamp making that hold. Silent clamping would be a data-corruption bug.
- **Store must execute before the attention that reads the region** — guaranteed only by build-order execution (doc 08's invariant 5); a "smart" reordering that hoisted attention above store would read stale rows. This is why `kvcache_store`/`kvcache_load` are explicit nodes rather than hidden side effects.
- **RoPE styles are model-level, not per-call** — mixing `NonInterleaved` and `Interleaved` on one model silently produces wrong scores with plausible magnitudes; it was suspect #1 in the Qwen2.5 bring-up (`docs/DEBUGGING-PLAN.md` H1) precisely because nothing crashes.
- **The attention window derives from positions, never from a counter** — if you find yourself reaching for `n_past` inside a kernel, you are breaking the reuse invariant (docs 05/07).
- **Softmax needs the max first** — skipping the subtraction works on toy inputs and overflows on real logits (score magnitudes grow with context); the AVX2 path preserves the same subtract-then-exp order.
- **Masked V rows are multiplied by exactly 0.0** — any change to the −∞ masking that produced NaN (`−∞ × 0` patterns) would poison the output; `exp(−∞)` → `0.0` is the behavior the tail loop relies on.
- **`bsums`-style exactness discipline applies here too** — the Q8_K `bsums` lesson of doc 10 (sums computed from the *saturated* integers) has its mirror in attention: `1/sm` normalization happens once, outside the V accumulation, so every worker's output matches the single-threaded order bit for bit.

## 4. Observe & verify

- `cargo test graph::` — the CPU attention round-trip tests (KV store → GQA attention against hand-computed references) and the decode/prefill KV-persistence tests live with the graph suite; `attn_parallel_realdata_correctness` checks multi-threaded attention against a real-dump reference.
- `cargo test vec_ops::` — per-op parity: every AVX2/NEON path vs the scalar reference you read above.
- `MINFER_TRACE=<path> ./target/release/minfer <model> "hi"` — per-node traces (doc 08 §4); the attention nodes' output stats land in the viz pipeline view (viz/README.md).
- `MINFER_DUMP_DIR` (debug-dump builds, `--features debug_dump`) — per-layer hidden-state dumps; comparing layer-by-layer against llama.cpp dumps is how the RoPE-style and attention-scale issues in `docs/DEBUGGING-*.md` were cornered.
- The KV arithmetic (24 KB/token for 0.5B, 288 KB/token for Qwen3-4B — §2.2) is directly observable: run a long context and watch RSS grow by the KV region totals doc 07 computed.

## 5. Cross-references

- [05 — Graph build](05-graph-builder-ir.md) §3 — where the `RmsNorm`/`RoPE`/`Kvcache*`/`Attn` nodes and their metas come from.
- [07 — Allocator](07-allocator-liveness-kv.md) §3.2 — the persistent K/V regions and the aliasing this doc's in-place ops rely on.
- [08 — Scheduler](08-scheduler-execute.md) — the build-order guarantee that orders store before attention.
- [10 — CPU matmul](10-cpu-matmul-kernels.md) — the `W_q/W_k/W_v/W_o` projections around these ops; the shared thread pool; the Q8_K `bsums` exactness lesson.
- [12 — Sampler](12-sampler.md) — the next stage: what happens to the logits the last layer produces.
- [13 — Decode loop](13-decode-loop-graph-reuse.md) — the cache in action: one position written per step, `nkv` growing by one.
- [14](14-metal-backend.md)/[15](15-cuda-backend.md) — the same attention on GPU (flash/split variants on Metal; split-KV on CUDA).
- `docs/GRAPH-REFACTOR-PLAN.md` §17 deviations 4/5 — the KV layout and `pos`-input decisions this doc's mechanics implement.

← [10 — CPU matmul: quantized weights × Q8_0 activations](10-cpu-matmul-kernels.md) · [Index](./README.md) · [12 — The sampler: from logits to a token](12-sampler.md) →
