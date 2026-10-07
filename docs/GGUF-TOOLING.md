# GGUF tooling — convert, quantize, split (F6, [#49](https://github.com/yusiwen/minfer/issues/49))

minfer reads GGUF (a *reader* in `src/gguf.rs`); F6 adds the **writer** and the
three subcommands that use it. This document is the writer's contract, the
subcommands' exact shapes, the supported/refused sets, and the reference used to
verify each claim. It is the design and the implementation record in one file.

- Reader: `src/gguf.rs` — `GgufContext::init_from_data`, `load_gguf_model`,
  `split_file_info`, `resolve_splits`.
- Writer: `src/gguf_write.rs` — `GgufWriter`, `TensorSpec`, `encode_kv`,
  `write_single`, `write_split`, `split_assignment`.
- Encoders: `src/quantize.rs` — `QuantTarget`, `quantize_row`, `decode_to_f32`.
- Converter: `src/convert.rs` — safetensors + `config.json` + `tokenizer.json`
  parsing, `Conversion`, `QuantizePlan`.
- Subcommands: `src/tooling.rs` — `run_convert`, `run_quantize`, `run_split`.

---

## 1. The writer contract

The parser **is** the contract: `gguf_write` emits exactly the bytes
`GgufContext::init_from_data` reads back, and the unit tests in
`src/gguf_write.rs` assert that (write → parse → compare). The layout is
llama.cpp's `gguf_write_to_file` layout.

### 1.1 Header

```text
magic      "GGUF"              (4 bytes)
version    u32 = 3
n_tensors  i64                 (little-endian)
n_kv       i64
```

### 1.2 Metadata (KV) encoding — every `GgufType` the reader supports

A key is a `u64` length followed by its UTF-8 bytes. Then:

| `GgufType` | tag | payload |
|---|---|---|
| `Uint8`, `Int8`, `Bool` | 0, 1, 7 | 1 byte |
| `Uint16`, `Int16` | 2, 3 | 2 bytes LE |
| `Uint32`, `Int32`, `Float32` | 4, 5, 6 | 4 bytes LE |
| `Uint64`, `Int64`, `Float64` | 10, 11, 12 | 8 bytes LE |
| `String` | 8 | `u64` length + UTF-8 bytes |
| `Array` | 9 | element `u32` tag, `u64` count, then `count` element payloads |

An array of strings is `count` length-prefixed strings; an array of a numeric
type is the raw little-endian values (the reader's `GgufKv.data` already holds
that form, so the writer writes it verbatim). This covers all 13 `GgufType`
variants; there is no metadata type the writer cannot emit.

`general.alignment` is read back by the parser (`u32`, powers of two). The
writer does **not** invent it: the caller passes the alignment (32 for a fresh
conversion, the source's own `ctx.alignment` for a rewrite/split), and a source
file's `general.alignment` key is copied through by the rewrite. A single-file
rewrite with a default-aligned source is therefore metadata-equivalent to it,
key for key (asserted by `f6_rewriting_a_gguf_is_bitwise_and_metadata_equivalent`).

### 1.3 Tensor index

For each tensor, in write order:

```text
name      string (must be < 64 bytes)
n_dims    u32   (trailing 1s dropped, at least 1)
ne[0..n_dims]  i64 each
type      u32   (GgmlType)
offset    u64   (relative to the start of the data section)
```

`n_dims` is `ggml_n_dims(ne)`: the highest index whose dimension is not 1, so
`[896, 1, 1, 1]` is written with `n_dims = 1`. The parser reconstructs the same
`ne` (trailing 1s) and derives `nb` from `ne` and the type, so the writer never
writes strides.

### 1.4 Alignment

After the tensor index the writer pads to `alignment` (the parser seeks from its
current position to `ggml_pad(tell, alignment)`), and pads **after every tensor's
payload** to the same alignment. The parser's data-section size is

```text
size = Σ ggml_pad(nbytes(t_i), alignment)          (in tensor order)
```

and it *requires* `info[i].offset == Σ_{j<i} ggml_pad(nbytes(t_j), alignment)` —
so the writer computes exactly those offsets and lays the data down in the same
order. `nbytes` is `(Π ne) / blck_size × type_size`, the same arithmetic as
`GgufTensorInfo::nbytes`.

### 1.5 Per-tensor byte layout for each emitted type

The writer copies or encodes the bytes that the graph's kernels already consume;
it does not add row padding. A tensor of type `T` with row length `ne[0]`:

| Type | bytes per block | elements/block | row bytes | layout |
|---|---|---|---|---|
| `F32` | 4 | 1 | `ne[0] × 4` | LE f32 |
| `F16` | 2 | 1 | `ne[0] × 2` | LE half bits |
| `BF16` | 2 | 1 | `ne[0] × 2` | LE bf16 bits (`f32`'s top 16 bits) |
| `Q4_0` | 18 | 32 | `ne[0]/32 × 18` | `d: f16`, `qs[16]` (element `j` low nibble, `j+16` high) |
| `Q4_1` | 20 | 32 | `ne[0]/32 × 20` | `d: f16`, `m: f16`, `qs[16]` |
| `Q5_0` | 22 | 32 | `ne[0]/32 × 22` | `d: f16`, `qh: u32` (5th bits, element `j` bit `j`), `qs[16]` |
| `Q5_1` | 24 | 32 | `ne[0]/32 × 24` | `d: f16`, `m: f16`, `qh: u32`, `qs[16]` |
| `Q8_0` | 34 | 32 | `ne[0]/32 × 34` | `d: f16`, `qs[32]: i8` |
| `Q4_K` | 144 | 256 | `ne[0]/256 × 144` | `d: f16`, `dmin: f16`, `scales[12]` (8×6-bit scale + 8×6-bit min, `get_scale_min_k4`), `qs[128]` (element `j` low nibble, `j+32` high, 64 elements per 32-byte group) |
| `Q5_K` | 176 | 256 | `ne[0]/256 × 176` | as `Q4_K`, plus `qh[32]`: the 5th bit of element `n+j` is bit `m1` of `qh[j]` and of `n+j+32` is `m2`, `m1`/`m2` shifting left by 2 per 64-element group |
| `Q6_K` | 210 | 256 | `ne[0]/256 × 210` | `ql[128]` (16 sub-blocks of 16, `L[j+l]&0xF` / `L[j+l+64]&0xF` low, `L[j+l+32]`/`L[j+l+96]` high), `qh[64]` (the 2 high bits of each of the four 32-element quarters, shifted 0/2/4/6), `scales[16]: i8`, `d: f16` |
| `Q2_K`/`Q3_K`/`Q8_K` and every I-quant | — | — | — | copied verbatim (rewrite/split only; **no encoder**) |

The writer validates that `ne[0] % blck_size == 0`, that every dimension is
≥ 1, that names are unique and < 64 bytes, and that the provider hands it
exactly `nbytes` — a short or long payload is a loud error, never a file whose
later tensors are shifted.

### 1.6 The multi-part convention

The reader's convention, which the writer must satisfy:

- Count in metadata: **`split.count`** (total parts).
- Index in metadata: **`split.no`** — 0-based. (llama.cpp's `LLM_KV_SPLIT_NO`
  spelling; *not* `split.index`.)
- `split.tensors.count` (total tensors across parts) is written too;
  llama.cpp emits it, the reader ignores it.
- File names: `{prefix}-NNNNN-of-MMMMM.gguf`, **1-based**, five digits;
  `resolve_splits` builds all `M` names from the entry point's name.

`write_split`:

1. Assigns tensors to parts greedily by padded data size (`split_assignment`),
   **never splitting a tensor**. A tensor larger than `--max-size` is refused
   with its own size, because the request cannot be honoured.
2. Gives every part the **full metadata** plus its own `split.no`, the shared
   `split.count`, and `split.tensors.count` — this is what makes each part parse
   standalone, which is what the reader does before merging.
3. Preserves the global tensor order: part 0's tensors first, then part 1's, …
   The merged index (each part's `info` concatenated in part order) therefore
   equals the single-file index exactly.
4. When the assignment is a single part, writes a plain `{stem}.gguf` with **no
   `split.*` keys at all** — one part is not a split, and the loader then sees an
   ordinary single file.

The reader's failure modes are `#[test]`-covered on a synthetic 2-part split:
a missing part, a part whose `split.no` does not match its position, and a
filename count that disagrees with `split.count` all fail the load.

---

## 2. The subcommands

```text
minfer convert  <hf-model-dir> <out.gguf> [--outtype f16|bf16|f32] [--split-max-size N]
minfer quantize <in.gguf> <out.gguf> --type <target> [--split-max-size N]
minfer split    <in.gguf> <out-dir> --max-size N [--stem NAME]
```

All three are dispatched before the global option parser (like `bench` /
`specverify`), so their flags do not collide with the inference options, and
none of them initializes a GPU backend.

`N` accepts plain bytes or a binary suffix (`1K`, `512M`, `2G`). For
`convert`/`quantize` with `--split-max-size`, the second positional is the
single-file output path and the parts are written beside it using its stem; for
`split`, the second positional is the output directory. Exit code is 0 on
success and 1 on any refusal, with the reason on stderr.

### 2.1 `convert` — HuggingFace → GGUF

Inputs in `<hf-model-dir>`: `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors` (or a
`model.safetensors.index.json` shard map), optional `generation_config.json`.

Tensor mapping (HF → GGUF):

| HuggingFace | GGUF |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` (absent for a tied model) |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` |
| `model.layers.N.self_attn.q_proj.{weight,bias}` | `blk.N.attn_q.{weight,bias}` |
| `…k_proj` / `…v_proj` / `…o_proj` | `blk.N.attn_k` / `attn_v` / `attn_output` |
| `model.layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` |
| `model.layers.N.mlp.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}.weight` |
| `…self_attn.rotary_emb.inv_freq` | *dropped* (llama.cpp drops it too; RoPE is recomputed) |

Any other tensor name is a refusal. Shapes are reversed: HF `[out, in]` becomes
GGUF `ne = [in, out, 1, 1]`, and the data stays row-major `[out][in]`.

Metadata written (marked **strict** = read by `Tokenizer::load` /
`hparams_from_gguf`; a file missing one is refused at load):

- `general.architecture = "qwen2"`, `general.type = "model"`, `general.name`
- **strict** `qwen2.{block_count,context_length,embedding_length,feed_forward_length}`
- **strict** `qwen2.attention.{head_count,head_count_kv,layer_norm_rms_epsilon}`
- **strict** `qwen2.rope.freq_base`
- `general.file_type` (1 = f16, 32 = bf16, 0 = f32; llama.cpp's `llama_ftype`
  numbers), `general.quantization_version = 2`
- `general.sampling.{top_k,top_p,temp,penalty_repeat}` from `generation_config.json`
- **strict** `tokenizer.ggml.model = "gpt2"`, **strict** `tokenizer.ggml.pre = "qwen2"`
- **strict** `tokenizer.ggml.tokens` (all `vocab_size` ids), `tokenizer.ggml.token_type`,
  `tokenizer.ggml.merges`, `tokenizer.ggml.{eos,bos,padding}_token_id`,
  `tokenizer.ggml.add_bos_token`
- **strict** `tokenizer.chat_template` (required; absent is a refusal)

`tokenizer.ggml.token_type` follows llama.cpp's `get_vocab_base`: type 1
(NORMAL) for the base vocabulary, type 3 (CONTROL) for an added token that is
flagged special **or** shaped `<|…|>`, type 4 (USER_DEFINED) for the other added
tokens, and type 5 (UNUSED) for ids in `[|vocab|, vocab_size)` with the
`[PAD{id}]` placeholder. `tokenizer.ggml.scores` is **not** written (llama.cpp
does not either; the strict loader defaults missing scores to 0).

`--outtype f16` writes 2-D tensors as f16 and **1-D tensors (norms, biases) as
f32** — llama.cpp's "except 1d tensors" rule, and what the engine's f32
norm/bias path consumes. `--outtype f32` writes everything as f32.
`--outtype bf16` ([#142](https://github.com/yusiwen/minfer/issues/142)) is the
same shape with a bf16 2-D payload: f32 → bf16 is **round-to-nearest-even**
(`ggml_compute_fp32_to_bf16`, including the quiet-NaN rule), and 1-D stays f32 —
`conversion/base.py`'s `n_dims <= 1` rule, which byte parity against
`llama-quantize --pure … BF16` confirms (§4.1.1).

### 2.2 `quantize` — re-encode a single-file GGUF

Targets with an implemented and byte-verified encoder:
**`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_K`, `q5_K`, `q6_K`**, plus the
`f16` / `f32` element casts.

- For a **quant** target only 2-D float tensors are quantized. 1-D tensors (norms,
  biases) and any tensor whose row length is not a multiple of the target block
  size keep their source type, and the CLI prints the list — llama.cpp's rule, not
  a silent choice.
- **The K-quant targets write one uniform type — they are not llama.cpp's
  `_M` mixtures.** `q4_K`/`q5_K` are the *CLI aliases* for `LLAMA_FTYPE_MOSTLY_Q4_K_M`
  / `_Q5_K_M` in `llama-quantize` (and their `general.file_type` numbers here are
  llama.cpp's 15/17/18), but `minfer quantize --type q4_K` puts **every**
  encodable 2-D tensor at `q4_K`. That is what `llama-quantize --pure` writes, and
  it is the reference the byte-parity gate uses; the mixture planner (per-layer
  Q6_K bumps, the OUTPUT/tied-embedding branch, `use_more_bits`) is **not**
  implemented — [#203](https://github.com/yusiwen/minfer/issues/203). A file from
  `llama-quantize … q4_K` without `--pure` is therefore *not* what this command
  produces.
- **A K-quant row must be a multiple of 256.** `QK_K` is 256, and a 2-D tensor
  whose `ne[0]` is not a multiple of it cannot be encoded at the requested type;
  minfer **demotes** it exactly the way llama.cpp's `tensor_type_fallback` does:

  | requested | demoted to |
  |---|---|
  | `q4_K` | `q5_0` |
  | `q5_K` | `q5_1` |
  | `q6_K` | `q8_0` |

  and to `f16` when even a 32-element block does not divide the row. The CLI
  prints the demoted list (`quantize: N tensor(s) demoted from q4_K (row length
  not a multiple of 256; llama.cpp's tensor_type_fallback): …`). This is not
  cosmetic: the 0.5B's hidden size is 896 = 3.5 × 256, so **145 of its 290
  tensors take the demotion** and only the 24 `ffn_down` tensors (`ne[0] = 4864 =
  19 × 256`) reach the `q4_K`/`q5_K`/`q6_K` encoder. §4.2 gates a second source
  (hidden 1024) where every 2-D tensor does.
  For the **legacy** targets (block size 32) the plan is unchanged: a 2-D row
  that is not a multiple of 32 keeps its source type, exactly as before #140.
- The `f16` cast follows the same "except 1d tensors" rule: 2-D tensors become
  **f16**, 1-D tensors (norms, biases) keep their source type — **f32** in every
  file `minfer convert --outtype f16` or `llama-quantize … F16` writes, and the
  only type the engine's norm/bias path reads (an f16 norm is a file the engine
  cannot run). The CLI prints the preserved list exactly as for a quant target.
  `minfer quantize --type f16` now produces a runnable file this way
  ([#169](https://github.com/yusiwen/minfer/issues/169)).
- The `f32` cast is the one target that converts **every** tensor, 1-D included,
  to f32.
- On a **tied** model (no `output.weight`), a sub-8-bit legacy target quantizes the
  shared `token_embd.weight` at **q8_0** — llama.cpp's tied-embedding policy.
  The CLI prints this too. (A K-quant target skips the mixture, so its tied
  embedding goes to the requested type and then to that type's fallback.)
- A multi-part source is refused (quantize one file, then split).
- K-quant/I-quant sources are refused: there is no dequantizer, so re-encoding
  them would produce wrong weights.

### 2.3 `split` — single file → parts

Verbatim tensor copy (no re-encoding) under `{out-dir}`, using the source's own
alignment and metadata. A multi-part input, an already-split-looking filename,
and a model with no tensors are refused.

---

## 3. Supported / refused sets and the exact refusal texts

```text
minfer convert: unsupported architecture (model_type = "llama",
  architectures = ["LlamaForCausalLM"]); minfer's converter supports
  'Qwen2ForCausalLM' (model_type "qwen2") only

minfer convert: HuggingFace tensor 'model.layers.0.mlp.gate_up_proj.weight' has
  no GGUF mapping in minfer's Qwen2 converter; supported tensors are
  model.embed_tokens, model.norm, lm_head, and model.layers.N.{input_layernorm,
  post_attention_layernorm,self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj} —
  minfer refuses to drop a weight it does not recognise

minfer convert: tensor 'x' has safetensors dtype 'I8'; minfer converts bf16,
  f16 and f32 only

minfer convert: unknown --outtype "q4_0"; supported: f16, bf16, f32

minfer convert: <dir>/tokenizer_config.json has no 'chat_template'; minfer's
  strict loader renders the model's own template, so a converted GGUF without
  one is not accepted

minfer quantize: --type is required (no silent default: a wrong target would
  write wrong weights); supported: q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K,
  q6_K, f16, f32

minfer quantize: target "q2_K" is a known GGUF type but minfer has no weight
  encoder for it (minfer can only read it, so writing it would emit wrong
  weights); supported encoder targets: q4_0, q4_1, q5_0, q5_1, q8_0, q4_K,
  q5_K, q6_K, f16, f32

minfer quantize: unknown quant target "banana"; supported: q4_0, q4_1, q5_0,
  q5_1, q8_0, q4_K, q5_K, q6_K, f16, f32

minfer quantize: tensor 'blk.0.ffn_down.weight' has type q4_K, which minfer
  cannot decode (no dequantizer for it); re-quantizing a K-quant/I-quant source
  is unsupported — start from an f16 or f32 GGUF

minfer quantize: <path> is a multi-part split (N parts); quantize a
  single-file GGUF, then split the result

GGUF split: tensor 'output.weight' is 144643072 bytes, larger than the
  67108864-byte part cap — a tensor cannot be split across parts; raise
  --split-max-size (at least 144643072 bytes) or use a smaller-capable format

minfer split: <path> is already a multi-part split (N parts); point at a
  single-file GGUF

minfer split: --max-size is required (no silent default part size)

size mismatch for <path>: expected 1048576 bytes, got 1572864 bytes (the server
  may have ignored the resume Range, or the download was truncated); removed the
  partial file
```

### What each conversion step is, exactly

| Step | Exactness |
|---|---|
| f16 → f16 (copy), f32 → f32 (copy) | bit-exact |
| f16 → f32, bf16 → f32 | bit-exact (exponent and mantissa are preserved; bf16 is `f32`'s top 16 bits) |
| bf16 → f16 | **exact in the mantissa** (bf16's 8 mantissa bits fit f16's 10) but **not in the exponent range**: a bf16 value outside f16's range becomes ±inf, and one below f16's smallest normal (2^-14) is rounded onto f16's subnormal grid (below 2^-24 it flushes to zero). No saturation is applied, so the loss is visible rather than a silently clamped weight. |
| f32 → f16 | **not exact** — round-to-nearest-even, and the same exponent-range rule |
| f32 → bf16 (#142) | **not exact** — round-to-nearest-even on f32's top 16 bits (`ggml_compute_fp32_to_bf16`, NaN forced quiet). bf16 keeps f32's exponent range, so there is no overflow, only mantissa rounding |
| bf16 → bf16 (#142) | bit-exact: the identity, since bf16 *is* f32's top half. Routed through the RNE encoder rather than copied, so a NaN payload is quieted exactly as the reference does. |
| f16 → bf16 (#142) | **not exact** — f16's 10-bit mantissa is rounded to bf16's 7 stored bits; no overflow, because bf16's exponent range covers f16's |
| rewrite / split (any type) | bit-exact: the payload is copied |
| f16/f32 → q4_0/q4_1/q5_0/q5_1/q8_0 | **lossy by construction** (that is the point); the encoder is byte-identical to llama.cpp's reference |
| q4_0…q8_0 → f32 | the exact stored values (a dequantize, not a re-quantize) |

---

## 4. Verification: references and tolerances

### 4.1 The reference the HF conversion was checked against

Two independent references, both on the **same** `Qwen/Qwen2.5-0.5B-Instruct`
checkpoint (bf16 safetensors downloaded to `/tmp/f6-work/hf-src`):

1. **llama.cpp's converter** (`convert_hf_to_gguf.py --outtype f16` with
   torch 2.14.0 / transformers 5.17.0) → `ref-f16.gguf`.
   **All 290 tensor payloads are byte-identical** to minfer's output
   (`sha256` per tensor, compared by name). Metadata is equivalent for every
   value the loader reads; the only differences are key order and that llama.cpp
   also writes the cosmetic `general.size_label`.
2. **llama.cpp itself**, via `llama-cli`/the formula below, for the logits.
   Running the two *different files* through minfer gives **bitwise-identical**
   logits and an identical greedy continuation, which is the strong claim: the
   files carry the same weights and minfer computes the same thing from them.

Because the source is bf16 and the output f16, "bit-exact" here is the
**bf16 → f16** row above: exact in the mantissa. It is *not* a claim that no
information was lost. The mantissa half is true — bf16 has 7 stored mantissa
bits (8 with the implicit one) and f16 has 10, so no bf16 mantissa is rounded by
f16 — but the **exponent range is not**: f16's smallest normal is 2^-14 and its
smallest subnormal is 2^-24, so a bf16 value below 2^-14 lands on f16's subnormal
grid (and is rounded there), and one below 2^-24 flushes to zero. On this
checkpoint that is **123 024 values in the 169 2-D tensors** (measured
2026-09-27, `f142_bf16_output_runs_within_the_stated_bound`); every one of them
is an f16 subnormal. This was found while checking #142's premise that "every
bf16 value is exactly representable in f16" — it is not, and that is why the
bf16-vs-f16 file comparison is a stated bound, not a bitwise claim (§4.3).

### 4.1.1 The bf16 writer (#142)

`--outtype bf16` is accepted and writes **2-D BF16 / 1-D F32**. The reference is
a pure cast by llama.cpp, but from the **f32** conversion rather than the f16
one:

```bash
minfer convert ~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct \
  ~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f32.gguf --outtype f32
llama-quantize --pure ~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f32.gguf \
  ~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-bf16-from-f32.gguf BF16
MINFER_F142_LLAMACPP_BF16=~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-bf16-from-f32.gguf \
  cargo test --release --bin minfer f142_bf16_conversion_is_byte_identical_to_the_reference -- --ignored
```

`minfer convert --outtype f32` is exact for a bf16 source, so the cast is the
f32→bf16 RNE projection of the checkpoint's own values. Result (measured
2026-09-27, aarch64): **290/290 tensor payloads byte-identical** — 169 BF16
2-D tensors, 121 F32 1-D tensors. The 1-D rule agrees because
`conversion/base.py` sets `data_qtype = F32` for `n_dims <= 1` on every file
type, and `llama-quantize`'s `tensor_allows_quantization` returns the source
type for a 1-D tensor.

**Why not the f16 source.** #142's Source B (`llama-quantize --pure <f16>.gguf …
BF16`) was tried first and is **not** byte-identical here: casting the f16 file
cannot recover the 123 024 subnormal values that the f16 conversion already
rounded, so 126 575 bytes across all 169 2-D tensors differ (measured
2026-09-27, per-tensor byte diff). That is the reference's input, not minfer's
writer — which is exactly what the f32-source reference isolates. The writer has
no torch/transformers converter to check against on dgxspark
(`convert_hf_to_gguf.py --outtype bf16` needs torch); the f32-source cast is the
strongest available reference, and the missing direct converter reference is
[#209](https://github.com/yusiwen/minfer/issues/209).

### 4.2 The reference the encoders were checked against

`llama-quantize ref-f16.gguf ref-<type>.gguf <type>` on the same f16 source.
For **each** of q4_0, q4_1, q5_0, q5_1, q8_0, all **290/290 tensors are
byte-identical — against the recorded `dgxspark` build: `gcc 13.3.0`,
`-ffp-contract=fast`, llama.cpp revision unrecorded (`docs/f6-fixtures.json`
records it per content, §4.2.2)** (measured 2026-09-27,
`dgxspark (aarch64, GB10 sm_121)`,
`MINFER_F6_F16_GGUF=… cargo test --release --bin minfer
f6_quantize_encoder_is_byte_identical_to_llamacpp -- --ignored`; the `env_path`
convention and the per-type command are below). **The claim is conditional on all
three of those** — the compiler, the effective `-ffp-contract`, and the llama.cpp
revision — because a `llama-quantize` binary *is* a (compiler, flags, revision)
triple and the reference is that binary's output.
That qualifier is not decoration, and this section records it because the number
is otherwise a statement about **one build** of `llama-quantize`: measured
2026-10-07, **the same llama.cpp source built with the one flag changed does not
match**, and a differently-built reference made the same gate red on a platform
this project ships on ([#334](https://github.com/yusiwen/minfer/issues/334)). The
K-quant half of the claim is compiler-sensitive as well as flag-sensitive, which
§4.2.1 records ([#349](https://github.com/yusiwen/minfer/issues/349)).

**Which build, and how it was identified.** The `dgxspark` producer is a GCC
13.3.0 `-O3 -DNDEBUG` build with no explicit `-ffp-contract` (so GCC's default
`fast`). It is identified by reproduction rather than by a recorded version
string — nothing recorded the binary's own source revision, which is part of what
[#205](https://github.com/yusiwen/minfer/issues/205) still owes. Measured
2026-10-07 on `dgxspark`, one variable at a time, from `~/git/reading/llama.cpp`
HEAD `050dde50c` and the cached f16 source:

| `llama-quantize` build | `ref/qwen2.5-0.5b-q4_0.gguf` sha256 |
|---|---|
| the cached `build/bin/llama-quantize` (GCC 13.3.0, `-O3 -DNDEBUG`) | `04634958ae0289b8c557c4d50491d225b23728326fd6d5bbfabedd299a89b3b9` |
| fresh build of HEAD `050dde50c`, same flags | `04634958…` — **identical**, so the flag is the only variable |
| that same build **+ `-ffp-contract=off`** | `ea94611ef461e5a65c1af0b6c7189d731735a31e310b54b48ad9b94fa2521930` |

The Mac's reference (`macbook (macOS 27.0.1, Apple M4 Pro)`, 2026-10-07; the
binary built 2026-09-01 from `458681e1d`, `CMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG`,
no `-ffp-contract=fast`) is sha256 `51c2b000…` and **disagrees with minfer in 168
of 290 q4_0 tensors** — every difference a single data nibble, zero scale bytes
(`blk.0.attn_k.weight`: exactly **12 of 64512** bytes).

**Why.** llama.cpp computes `x*id + c` in `quantize_row_q4_0_ref`, and whether
the compiler contracts that across statements is a *rounding decision*: under
`-ffp-contract=fast` the expression is one FMA, otherwise `fmul` + `fadd`, and
the two pick a different quant at an exact rounding boundary. The Mac's
`quantize_row_q4_0_ref` disassembles to `fmul.4s` + `fadd.4s` with no `fmla`.
GCC contracts by default and Apple clang does not without the flag, so **both
sides are right about their own build**; minfer's encoder uses `f32::mul_add`
unconditionally, i.e. it reproduces the contracting one. This is the mirror of
the experiment this section used to record alone — minfer *without* `mul_add`
against the `dgxspark` reference differs in **12 of 64512** bytes on the same
tensor, the same count — which is what makes "the reference's build" the whole
content of the claim. Q8_0 uses a single multiply and matched without it.

**What the gate does about it.** `f6_quantize_encoder_is_byte_identical_to_llamacpp`
prints the reference's path, size and the recorded build it established it matched
(`build -ffp-contract=fast, recorded reference gcc 13.3.0 …` on a pass). On a
mismatch it does **not** report an encoder defect first: it re-encodes the whole
source with the uncontracted arithmetic — `quantize::FmaContract::Off` through
`quantize_row_with`, which is **exact** for the legacy quants and a *model* of the
K-quants' per-expression fusion (§4.2.1) — and then it asks the manifest which
recorded content the file is, because re-encoding alone cannot tell a different
compiler from a corrupted file: both match neither model. The three byte
comparisons — `Fast` against the reference, the uncontracted model against it, and
the file's digest against `docs/f6-fixtures.json` (§4.2.2) — plus the entry the
digest matched give five named outcomes, and the classifier that decides them is
pure and unit-tested (`the_f6_parity_verdict_names_the_recorded_build`):

| outcome | when | what the gate does |
|---|---|---|
| **reproduces** | the `Fast` model matches | pass, and the pass line names the recorded compiler and flag |
| **flag mismatch** | only the uncontracted model matches | fails with the build named: "the reference is NOT the `-ffp-contract=fast` build this encoder reproduces … rebuild `llama-quantize` with `-ffp-contract=fast`" (#334) |
| **not the recorded content** | the digest matches no recorded content for the path | the resolver of §4.2.2 has already refused the file by name and digest; a path outside the cache is not a fixture at all |
| **a recorded foreign build** | the digest *is* recorded, but the entry is not the path's `authoritative_reference`, and neither model matches | **skips loudly** (a passing run with a `[f6 parity] SKIP` line) naming the build the file is, the authoritative build the claim was measured against, the reason (§4.2.1's compiler sensitivity), and this box's own `cc --version` |
| **the authoritative build, not reproduced** | the digest *is* the `authoritative_reference` and neither model matches | fails: the encoder no longer reproduces the reference the claim is asserted against — a defect, not a compiler difference |

The `#[ignore]` reason names the same prerequisite. The identity is a whole-file,
per-tensor byte comparison in every outcome. The fixture it resolves is
additionally verified against the manifest of §4.2.2 first, so *which file* is
being compared against is checked too — a stale or replaced reference in the
cache refuses the run instead of silently becoming the reference.

The pure encoder gate (`quantize::tests`) additionally pins the block layout
against hand-checked reference numbers: the scale, the `j` / `j+16` nibble
packing, the 5th-bit plane, `type_size`/`blck_size`, and the zero-block case.

#### 4.2.1 The K-quants (#140)

The K-quant reference functions (`quantize_row_q4_K_ref` / `q5_K_ref` /
`q6_K_ref`) are **search** quantizers: `make_qkx2_quants` scans 21 (q4_K) or 16
(q5_K) candidate scale/min pairs and `make_qx_quants` re-derives the
least-squares scale for 19 candidate `iscale` values. Every candidate is
evaluated in an `a*b + c` shape, so the FMA contraction is a *rounding
decision*, and one ULP picks a different quant. Matching llama.cpp took four
distinct findings, each read off the disassembly of the **production object**
(`objdump -d build/ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o`), then
confirmed by byte parity:

1. **`sum_x2 += x*x` is an FMA.** The q4_K/q5_K per-element weight is
   `av_x + |x|` with `av_x = sqrt(sum(x²)/32)`, and the sum is contracted
   (`fmadd s0, s1, s1, s0`). The weight feeds the search's error metric.
2. **`nearest_int(a*b)` folds the magic constant into the product's FMA.** The
   reference's round-to-nearest-even trick becomes `fmadd a, b, #12582912.0`
   (`fmov w0, #0x4b400000`) and only then masks the mantissa — the product is
   **not** rounded before the add. `nearest_int(a * b)` in Rust rounds twice and
   picks a different integer at a boundary; the port needs `nearest_int_mul`.
3. **`a*b - c*d` contracts per expression, not per shape.** In
   `make_qkx2_quants` the discriminant `D = sum_w*sum_l2 - sum_l*sum_l` fuses its
   *left* product (`fmul` + `fnmsub`), while `this_scale = sum_w*sum_xl -
   sum_x*sum_l` fuses its *right* one (`fmul` + `fmsub`). Both are a different
   ULP from the plain expression; the port writes each explicitly.
4. **A scalar loop and its vectorized twin can disagree.** In
   `make_qx_quants` the initial accumulation loop is scalar and uses `fmadd`,
   while the 19-candidate search loop is 4-wide vectorized and computes plain
   `fmul` products with an in-order `fadd` reduction — no FMA at all. Matching
   only the scalar form left **137 of 424 random 16-element groups** differing
   from the reference; matching the vectorized form as well made all 424 equal.

**Reference procedure (uniform encoders).** `llama-quantize`'s `q4_K`/`q5_K` are
CLI aliases for the `Q4_K_M`/`Q5_K_M` **mixtures**; the uniform encoder this
project implements is what `--pure` selects. The gate's reference must therefore
be:

```bash
llama-quantize --pure <f16>.gguf <ref>-q4_K.gguf q4_K    # likewise q5_K, q6_K
llama-quantize          <f16>.gguf <ref>-q4_0.gguf q4_0   # legacy: no mixture to disable
```

**Sources and results (measured 2026-09-27, aarch64).** The f16 sources live in
the persistent cache `~/.cache/minfer/f6-src/` (never `/tmp`):

| source | hidden | 2-D tensors that reach the K encoder |
|---|---|---|
| `qwen2.5-0.5b-instruct-f16.gguf` (from `Qwen/Qwen2.5-0.5B-Instruct`, bf16 safetensors → `minfer convert --outtype f16`) | 896 | **24** of 144 — 145 rows are not a multiple of 256 and take `tensor_type_fallback`, 121 tensors are 1-D copies |
| `qwen3-0.6b-f16.gguf` (`minfer quantize <Qwen3-0.6B-Q8_0.gguf> … --type f16`) | 1024 | **197** of 197 — no demotion, every 2-D tensor |

`MINFER_F6_QUANT_TYPE=q4_K|q5_K|q6_K` with `MINFER_F6_F16_GGUF` /
`MINFER_F6_LLAMACPP_QUANT` set to the pair above: **290/290 tensors byte-identical
on the 0.5B** (`encoded-as q5_0: 145, f32: 121, q4_K: 24`) and **310/310 on the
0.6B** (`encoded-as f32: 113, q4_K: 197`), for each of the three types — against
the same recorded `dgxspark` build as §4.2: **`gcc 13.3.0`,
`-ffp-contract=fast`**. The gate prints that breakdown, so "byte-identical" always
carries how many tensors the new encoder actually saw; since
[#349](https://github.com/yusiwen/minfer/issues/349) it names the recorded
compiler too. Legacy regression on the same 0.5B source:
q4_0/q4_1/q5_0/q5_1/q8_0 all still 290/290 against the reference of §4.2 (the
K-quant rows above are against that same build, and their per-expression fusion is
*not* reproducible by the `Off` variant — that variant is the gate's provenance
probe, not a second encoder).

**The K-quant claim is compiler-sensitive, and the record says which compiler.**
Measured 2026-10-07 on `macbook (macOS 27.0.1, Apple M4 Pro)`: after rebuilding
`llama-quantize` from `c479922ac` with
`-DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -ffp-contract=fast"` (the flag confirmed by
disassembly — `fmul` + `fadd` → `fmadd`/`fmla`, Apple clang 21.0.0), the five
legacy quads pass **290/290** and **every K-quant fails**: `tensor
blk.0.ffn_down.weight payload differs (2285 of 2451456 bytes) … build unmatched`
for `q4_K`, `97 of 2996224` for `q5_K`, `269 of 3575040` for `q6_K` — the
reference matches neither minfer's `FmaContract::Fast` nor its uncontracted `Off`
model. `git log 050dde50c..c479922ac -- ggml/src/ggml-quants.c` touches only the
q3_K/i-quant encoders, so the residual is **Apple-clang-vs-GCC codegen of the
search quantizers above**, not `-ffp-contract`.

**The decision this forces, recorded.** The byte-parity claim is asserted against
**the content `docs/f6-fixtures.json` marks `authoritative_reference`** (§4.2.2) —
for every `ref/…` path exactly one, the `dgxspark` `gcc 13.3.0`
`-ffp-contract=fast` build. A reference whose digest *is* recorded but is not that
entry is a different compiler's build of the same source: the gate names the file's
build, the authoritative one and the reason and **skips loudly**, because a
permanently red gate on a supported platform is the failure mode the project has
already filed once (`docs/ARCHITECTURE-EXECUTION-PLAN.md` §14 row 6). A mismatch
against the authoritative entry, or against a file no entry records, still
**fails**: the skip is never available to the claim's own reference, and a
corrupted or unrecorded file cannot be excused as a compiler difference.

Reproduce the sources and references:

```bash
mkdir -p ~/.cache/minfer/f6-src/ref ~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct
# 1. the HF checkpoint (5 files, ~988 MB)
for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
  curl -sL -o ~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct/$f \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/$f
done
# 2. the two common f16 sources
minfer convert ~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct \
  ~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f16.gguf --outtype f16
minfer quantize ~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
  ~/.cache/minfer/f6-src/qwen3-0.6b-f16.gguf --type f16
# 3. the llama-quantize references
for t in q4_0 q4_1 q5_0 q5_1 q8_0; do
  llama-quantize ~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f16.gguf \
    ~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-$t.gguf $t
done
for m in qwen2.5-0.5b qwen3-0.6b; do for t in q4_K q5_K q6_K; do
  llama-quantize --pure ~/.cache/minfer/f6-src/$([ $m = qwen2.5-0.5b ] && echo qwen2.5-0.5b-instruct || echo qwen3-0.6b)-f16.gguf \
    ~/.cache/minfer/f6-src/ref/$m-$t.gguf $t
done; done
```

#### 4.2.2 The fixture manifest (#205)

The f16 sources and the `llama-quantize` references are inputs the gates do not
produce, so until #205 nothing noticed when the cache held *something else*: a
stale or replaced reference was compared against silently and the gate stayed
green. `docs/f6-fixtures.json` is the record — one entry per artifact **content
identity**, carrying `path`, `bytes`, `sha256` (or a `sha256_prefix` where the
2026-10-07 table truncated it), the exact `producer` command, the producer's
identity (`minfer_commit` for a minfer-produced file; `llamacpp_commit` +
`compiler` + `ffp_contract` + `cflags` for a llama-produced one), `date` and an
**absolute** box label (gate contract rule 5). Two entries for one path are a
*recorded divergence*, and each of them carries a `divergence_notes` entry
explaining it: the f16 source and its bf16 cast differ by **minfer producer
version** (`dgxspark`, 2026-09-27, commit unrecorded — a Mac regeneration at
master `ab34a72` produced other bytes, and `bf16-from-f32`, whose input *is*
identical on both boxes, is identical too), while the `ref/qwen2.5-0.5b-*`
references differ by the reference **build** (§4.2: the two boxes'
`llama-quantize`s — `dgxspark` `gcc 13.3.0` and the Mac's Apple clang 21.0.0).

Exactly one of a `ref/…` path's entries carries `authoritative_reference: true`
([#349](https://github.com/yusiwen/minfer/issues/349)): **the content the
byte-parity claim is asserted against**, which for every path today is the
`dgxspark` `gcc 13.3.0` `-ffp-contract=fast` build. The mark is what lets the
parity gate's verdict say *which* recorded build it is looking at, and therefore
whether a mismatch against both encoder models may be excused as a different
compiler (§4.2.1) or must be reported as an encoder defect. `--check` enforces it
as **S6**: a path with a `llama-quantize` record has exactly one marked entry, the
mark is on a `llama-quantize` entry built with `-ffp-contract=fast`, and it
appears nowhere else.

```bash
python3 scripts/check_f6_fixtures.py --check     # CI: the manifest's shape + the tree cross-check
python3 scripts/check_f6_fixtures.py --verify    # + every cached file's bytes and sha256
python3 scripts/check_f6_fixtures.py --selftest  # the checker's own cases, including a tampered copy
python3 scripts/check_f6_fixtures.py --file /tmp/copy.gguf   # one file
```

`--verify` names the file, every recorded digest and the actual one when they
disagree, and reports a file no entry names; `--check` additionally requires that
every `~/.cache/minfer/f6-src/…` fixture the source tree spells is an entry, that
every `.gguf` a recorded `producer` command names is too, and (**S7**) that the
command runs the program its `producer_kind` names and writes the entry's own
path. The other half runs inside the gates: the shared fixture resolver verifies a
path it hands back (`env_path` → `src/tooling/tests/f6_fixtures.rs`), so the
documented `cargo test … --ignored` invocation refuses a tampered cache by name and
digest rather than comparing against it. `MINFER_F6_CACHE` relocates the cache root
for an experiment — which is how the refusal is demonstrated without touching
`~/.cache/minfer/f6-src/` — and a *deliberate* reference belongs outside the
cache root, where a path the manifest does not name is not a fixture at all.

The **record itself** is relocatable too, with `MINFER_F6_MANIFEST`
([#354](https://github.com/yusiwen/minfer/issues/354)): the manifest-side twin of
`MINFER_F6_CACHE`, honoured by both halves — the gate's reader
(`src/tooling/tests/f6_fixtures.rs`) and the checker (`--manifest` still wins over
it) — so a test can point the gate at a `/tmp` record instead of editing a tracked
file:

```bash
MINFER_F6_MANIFEST=/tmp/f6-fixtures.json cargo test --release --bin minfer -- \
    the_manifest_override_drives_the_whole_resolver
MINFER_F6_MANIFEST=/tmp/f6-fixtures.json python3 scripts/check_f6_fixtures.py --check
```

Both halves honour it **loudly**: a value that is set but empty, or a record that
is missing, unreadable or malformed, refuses by name and never falls back to
`docs/f6-fixtures.json` — the empty case is a usage error (exit 2) in the checker.
The checker's `--regenerate` (§4.2.3) **refuses** the variable outright, because
re-recording into a manifest chosen by the environment is the accident the
override must not create; `--manifest <path>` is that mode's auditable spelling.
The override is what makes the parity gate's fifth verdict testable: with a
fabricated record that names a reference's content as a non-authoritative build,
`ParityVerdict::RecordedForeignBuild` — the loud `[f6 parity] SKIP` — is exercised
end-to-end from a unit test and from a real gate run, instead of by editing the
tracked manifest under a `cp` backup ([#349](https://github.com/yusiwen/minfer/issues/349)).

**What it does not cover.** The manifest checks content, not truth: nothing else
recorded those bytes, so a wrong digest in the record is accepted. Every entry
carries a full `sha256` since [PR
#350](https://github.com/yusiwen/minfer/pull/350) re-captured the eleven Mac
references (the 2026-10-07 table had kept 8 hex characters), so
`--strict-digests` passes with no `WEAK` line; the reader still accepts a
`sha256_prefix` entry weakly, for a digest that has to be re-captured again. The
`hf/` checkpoint's revision and `f164/minfer-f16.gguf`'s producer are recorded
(they were the other two gaps that PR closed), but a `llama-quantize` reference
is still re-recorded **by hand**: `--regenerate` (§4.2.3) refuses one on purpose.

#### 4.2.3 Regenerating from the manifest (#345)

`--verify` checks the cached bytes against the record; it does not *produce* them.
`--regenerate` closes that gap: it re-runs the recorded `producer` command for the
selected entries and re-records the content identity from what the run wrote.

```bash
python3 scripts/check_f6_fixtures.py --regenerate --only qwen2.5-0.5b-instruct-f32.gguf
python3 scripts/check_f6_fixtures.py --regenerate --box 'dgxspark (aarch64, GB10 sm_121)'
python3 scripts/check_f6_fixtures.py --regenerate --dry-run     # classify, run nothing (CI)
```

`--only` takes a manifest `path` or `PATH@BOX` (the form that selects one of a
recorded divergence's contents); `--box` selects every entry recorded against that
box label. The mode is a **verification gate first**:

- **It runs the producer the record names, or refuses.** The command's program is
  resolved *before* anything runs — `minfer` on `PATH` and then
  `./target/release/minfer`, `curl` for an `hf-download` entry, and the entry's own
  `llamacpp_binary` for a `llama-quantize` one. There is **no fallback between
  producers**: a missing `llama-quantize` is a refusal naming the path it looked
  for, never a reason to run `minfer` instead.
- **Regenerating a `llama-quantize` reference is out of scope.** The byte-parity
  claim of §4.2 is about one compiler's build, so the mode refuses such an entry —
  naming the `llamacpp_binary` it records and whether that binary exists here — and
  says to re-run it on the box that records it. Even a box that *has* the build
  takes the refusal: a half-implemented compiler-identity check would be worse than
  not running it.
- **The run must reproduce the record.** The content the producer wrote is hashed
  and compared with the entry's full `sha256`. A match re-records `bytes` and the
  `date` of the reproducing run (`sha256` is equal by construction) and prints
  every change; a second run the same day writes nothing, so the mode is
  idempotent. A digest that matches **no** recorded content for the path is a
  **finding** (exit 1), not an update — the record's producer no longer reproduces
  the record — and the message names both digests, the bytes and the commit that
  ran.
- **The producer identity is checked before the run.** A `minfer` entry whose
  `minfer_commit` differs from what runs here (`MINFER_F6_COMMIT`, else the tree's
  `HEAD`) is **refused** (exit 3) naming both commits: that run is not the producer
  the entry names, so its bytes cannot be recorded against it. An `unrecorded`
  identity may run — there is no claim to contradict — but a differing digest is
  then a finding like any other, never a "producer version" excuse.
- **It writes only `sha256`/`bytes`/`date`.** `authoritative_reference`, the
  producer command and every identity field are left exactly as they were, so the
  mode can neither invent provenance nor move the content the byte-parity claim is
  asserted against. `--check` (`S1`–`S7`) re-runs on the result before it is kept.
- **A rejected run cannot destroy a fixture.** The existing cache file is renamed
  to `<path>.regen-before` before the producer starts (a rename, not a second
  copy); a finding, a refusal after a run, or a producer failure restores it and
  keeps the rejected content at `<path>.regen-rejected`; a verified success removes
  the pre-run copy, which is byte-identical to the new file.
- **`--dry-run`** classifies every selected entry — `RUN` with the `~`-expanded
  command, or `REFUSED` with the reason — and writes nothing. CI runs it, so a
  manifest whose command has the wrong program, the wrong output path or no
  command at all fails there (S7).

Exit codes: `0` clean, `1` a finding or a producer failure, `3` a refused target,
`2` a selection that matches nothing. A refusal is the honest per-box answer to
"this producer cannot run here", not a silent skip; `--strict-runnable` turns it
into a failure for a box that is expected to hold every producer.

**Measured (`dgxspark (aarch64, GB10 sm_121)`, 2026-10-07).**
`--regenerate --only qwen2.5-0.5b-instruct-f32.gguf` re-ran `minfer convert … --outtype
f32` and the file came back **byte-identical** — `6894f9ea3eb79e29…`, 1 982 078 784 B,
the recorded digest and size — so the only manifest change was the `date`
(`2026-09-27` → `2026-10-07`); running it again the same day wrote nothing at all.
The eleven `ref/…` entries refuse by name: the Mac's `build-fpc` binary is not on
this box, and the `dgxspark` one is refused as out of scope even though it is.

### 4.3 Tolerances

| Comparison | Tolerance | Measured |
|---|---|---|
| rewrite vs source logits | **bitwise** (`assert_eq!`) | equal |
| minfer-converted vs llama.cpp-converted logits (same engine) | **bitwise** | equal |
| split vs unsplit logits | **bitwise** | equal |
| `f16 → q8_0` logits, same context | max \|Δ\| ≤ 1.0 **and** greedy text identical | max \|Δ\| = **0.481**, mean 0.082, max \|logit\| = 18.43 (2.6% of the largest logit); greedy continuation identical |
| `f16 → q4_K` logits, same context, `minfer quantize` output (0.5B) | max \|Δ\| ≤ 0.30 × max \|logit\| **and** the first greedy token identical | max \|Δ\| = **2.72**, mean 0.44, max \|logit\| = 18.43 (**14.8%**); greedy `[12095, 13, 1084, 374]` identical to the f16 source |
| `f16 → q5_K` logits, same context (0.5B; the file is mostly Q5_1 after the fallback) | same | max \|Δ\| = **4.10**, mean 0.83 (**22.3%**); greedy `[12095, 13, 12095, 374]` — 3 of 4 tokens; the f16 source's `1084` flips to `12095` at token 2. The file is byte-identical to `llama-quantize --pure … q5_K`'s, so the flip is the quantisation, not minfer |
| `f16 → q6_K` logits, same context (0.5B) | same | max \|Δ\| = **0.996**, mean 0.144 (**5.4%**); greedy identical |
| `f16 → q4_K/q5_K/q6_K` logits (Qwen3-0.6B, hidden 1024 — every 2-D tensor K-encoded) | same | max \|Δ\| = **3.71 / 2.39 / 1.19**, mean 0.69 / 0.43 / 0.21, max \|logit\| = 19.99 (**18.6% / 11.9% / 5.9%**); greedy `[12095, 13, 576, 6722]` identical for all three |
| an f16 file's CUDA logits vs the same file's CPU logits (#141, 34-token prompt, ctx 512, Qwen2.5-0.5B-Instruct f16) | max \|Δ\| ≤ **0.01** and max relative ≤ **1e-3**, greedy continuation identical | max \|Δ\| = **7.34e-5**, mean 1.26e-5, max \|logit\| = 18.43 (**4.0e-6** relative); greedy `[12095, 13, 1084, 374]` on both |
| that f16 file under llama.cpp (same prompt, `--temp 0`) | — | `Paris.`, the same greedy continuation minfer produces on CPU and CUDA |
| a **bf16** file's CPU logits vs the f16 file's, same context (0.5B, bf16 source) | max \|Δ\| ≤ **1e-4** and max relative ≤ **1e-5** **and** greedy continuation identical (the bitwise expectation of #142's text is refuted by measurement, §4.1) | max \|Δ\| = **2.29e-5**, mean 3.19e-6, max \|logit\| = 18.43 (**1.24e-6** relative); 147 357 of 151 936 logits differ in the last bits, and the whole difference is attributed to the **123 024 f16-subnormal weight values** the f16 file rounds (the bf16 file carries the checkpoint's exact values); greedy `[12095, 13, 1084, 374]` on both |

The K-quant run gate (`f6_k_quant_output_runs_within_the_stated_bound`) states
its bound as **max \|Δlogit\| ≤ 0.30 × max \|logit\| and the first greedy token
identical** before measuring, prints all six measurements, and is run twice (0.5B
and 0.6B). The 0.30 bound was set after the first, too-optimistic pass of 2.0
absolute (taken from the q8_0 row) came back at 2.72 on the 0.5B; the measured
worst is 22.3%, so the bound has headroom of under 1.4×, not an order of
magnitude. The full greedy continuation is *reported*, not asserted, for the
reason the q5_K row gives.

---

## 5. The download size gate

`download::http_download` used to ignore the expected size it was handed:
`curl -C -` could exit 0 with a file of the wrong length, which the "already
cached" check (a size comparison) would then never see, because it was never
populated. F6 adds `download::check_downloaded_size(path, expected)`:

- `expected == None` → the length is reported but not judged (the remote size
  could not be determined; guessing would reject valid files);
- `expected == Some(n)` and the file is `n` bytes → accepted;
- otherwise → an error naming both sizes, and the partial file is **removed**
  so the next attempt cannot resume onto it.

The gate is unit-tested at the pure level and end-to-end against a local
`TcpListener` HTTP server, with two modes: a correct `206 Partial Content`
resume (accepted, bytes equal), and a server that answers with a correct-looking
`206` header for the requested range but ships the whole object (curl appends,
exits 0, the file is larger than expected → rejected and removed). No external
network is used.

---

## 6. Known gaps (follow-ups)

- **No K-quant *mixture* planner.** The three encoders landed in
  [#140](https://github.com/yusiwen/minfer/issues/140) and `minfer quantize
  --type q4_K` writes a uniform file (`llama-quantize --pure`). llama.cpp's
  `Q4_K_M`/`Q5_K_M` per-tensor policy — the `OUTPUT`/tied-embedding branch, the
  `use_more_bits` Q6_K bumps for `attn_v` and `ffn_down`, the fused-QKV rule — is
  not implemented, so `llama-quantize … q4_K` (no `--pure`) is not reproduced.
  Tracked as [#203](https://github.com/yusiwen/minfer/issues/203).
- **The legacy targets keep `tensor_type_fallback`'s first step only by
  accident.** For a 2-D tensor whose row length is not a multiple of the target's
  block size, minfer keeps the source type; llama.cpp demotes it (`q4_0` →
  `F16`) and, for the K targets, to a smaller-block type (implemented here in
  §2.2). For the f16 sources every gate uses the two agree, because the source
  type already *is* F16. An f32 source with such a row would differ. Filed as
  [#204](https://github.com/yusiwen/minfer/issues/204).
- **The byte-parity chain is a recipe, not a script.** The f16 sources and the
  `llama-quantize` references live in `~/.cache/minfer/f6-src/` and are
  regenerated by the recipe in §4.2.1; nothing checks that they are the ones the
  record describes, and a stale cache entry would silently be compared against.
  [#205](https://github.com/yusiwen/minfer/issues/205) closed that: the record
  is `docs/f6-fixtures.json` (§4.2.2), the checker is
  `scripts/check_f6_fixtures.py`, and the gates verify the fixture they resolve.
  [#345](https://github.com/yusiwen/minfer/issues/345) then made the recipe
  runnable: `--regenerate` (§4.2.3) re-runs a recorded producer and re-records the
  content it produces, idempotently, and refuses a target it cannot verify. What
  remains by hand: a `llama-quantize` reference (deliberately out of scope, §4.2.3)
  and any digest whose run is at a `minfer` commit the entry does not name — which
  is also why the cross-box f16 divergence below is *explained* (a producer-version
  difference) but not *settled* (one side's minfer commit is unrecorded).
- **`qwen2.5-0.5b-instruct-f16.gguf` is not byte-identical across boxes.**
  `dgxspark`'s copy is `aef12ad44a60d2dd…` (2026-09-27) and the Mac's is
  `a26884ee1286c1d3…` (2026-10-07, master `ab34a72`), while their f32
  conversions (`6894f9ea3eb79e29…`) and the bf16-from-f32 reference
  (`688109f4c9a4a8ca…`) **are** identical — so the cause is a *producer version*
  difference, not a platform one, and it is only readable at all if the record
  names the minfer commit that produced each fixture. That is [#205]'s field, not
  this one's; the two boxes' full tables are on
  [#333](https://github.com/yusiwen/minfer/issues/333) and
  [#205](https://github.com/yusiwen/minfer/issues/205).
- **f16 weights run on CPU, CUDA and Metal for both architectures (Qwen2/Qwen2.5
  and Qwen3).**
  `Op::MatMul`/`Op::GetRows` dispatch f16 on the CPU (one weight row at a time,
  never an f32 copy of the weights) and the dot is **vectorized** — AVX2 `F16C`
  / aarch64 baseline NEON `FCVTL`, an f64 scalar oracle, `MINFER_NO_NEON=1`
  forcing scalar, and the multi-token prefill decoding each row once and
  threading the row loop through the shared CPU pool: measured on the 0.5B at a
  34-token prefill **3.2 → 217 tok/s** (10.71s → 0.16s; 25.0 tok/s with the
  vectorized dot alone). CUDA registers the raw 2 B/element weights and converts
  in-register (`f16_f32_matmul_vec` / `_scalar`, `embed_rows_f16`), so the f16
  file keeps its memory advantage (0.5B: 948 MiB of device weights vs ~1.9 GiB
  dequantized); an f16 prefill does not enter the int8 MMQ GEMM (which streams
  quantized bytes) and runs the f32-activation kernel instead. **Metal gained the
  same pair in [#164](https://github.com/yusiwen/minfer/issues/164)**
  (`kernel_f16_f32_matmul` + `kernel_get_rows_f16`, `src/metal/kernels/f16.metal`;
  both loaders' Metal arm registers F32/F16/BF16), so an f16 GGUF no longer falls
  to the CPU there — and its prefill runs the same f32-activation kernel, not a
  simdgroup GEMM. Completed by
  [#141](https://github.com/yusiwen/minfer/issues/141) (qwen2) and
  [#167](https://github.com/yusiwen/minfer/issues/167) (qwen3's loader *and* its
  graph type gate, plus the one shared registration rule in
  `models::weight_reg`); the F6 half was
  [#49](https://github.com/yusiwen/minfer/issues/49). The **file** contract is 2-D
  f16 and 1-D f32, and `minfer quantize --type f16` now honours it (§2.2,
  [#169](https://github.com/yusiwen/minfer/issues/169)); the CUDA norm path also
  refuses a non-f32 norm weight (a registered length other than `d*4` bytes)
  instead of reading `d*4` bytes out of a `d*2` buffer
  ([#169](https://github.com/yusiwen/minfer/issues/169)).
- **bf16 weights run on CPU, CUDA and Metal.** [#142](https://github.com/yusiwen/minfer/issues/142)
  added `--outtype bf16` and the CPU weight path (`Op::MatMul` decodes one bf16
  row at a time via `vec_ops::mat_mul_bf16`, `Op::GetRows` decodes bf16
  embedding rows, 1-D stays f32); [#208](https://github.com/yusiwen/minfer/issues/208)
  then registered the type and added the kernels on both devices
  (`bf16_f32_matmul_vec`/`_scalar` + `embed_rows_bf16` on CUDA,
  `kernel_bf16_f32_matmul` + `kernel_get_rows_bf16` on Metal), each device half
  with its own exactness gate and real-model gate. bf16 is not an MMQ format, so
  a bf16 prefill runs the f32-activation kernel on both, and neither device
  registers the `attn_qkv`/`ffn_gu` concat copies (`cuda::concat_rows` has no
  2 B/element arm), so bf16 runs unfused.
- **The bf16 converter reference is a cast, not the converter.** §4.1.1 checks
  minfer's bf16 output against `llama-quantize --pure <f32>.gguf … BF16`, because
  `convert_hf_to_gguf.py --outtype bf16` needs torch (absent here). The cast is a
  valid per-tensor byte reference but shares minfer's RNE rule by construction;
  the direct converter check is [#209](https://github.com/yusiwen/minfer/issues/209).
- **`general.size_label` is not written** (cosmetic; llama.cpp derives it from
  the parameter count). Every other key llama.cpp writes for this architecture is
  written, with the same value.
- **Tied-embedding policy follows llama.cpp for the legacy quants only.**
  For an *untied* model, llama.cpp promotes `output.weight` to Q6_K under a
  sub-8-bit target; minfer keeps it at the requested target (Q6_K has no encoder
  yet). Documented, not silently different-by-accident.
- **`minfer convert` holds one tensor in memory at a time** (it streams from
  the safetensors file into the writer), so its footprint is the largest single
  tensor, not the model. `quantize` and `split` stream through the mmap. The
  output file itself is a full copy: a 0.5B f16 conversion is ~948 MiB.

---

## 7. Using a converted model

```bash
# 1. HF -> f16 GGUF
minfer convert /path/to/Qwen2.5-0.5B-Instruct /tmp/model-f16.gguf --outtype f16
minfer /tmp/model-f16.gguf "The capital of France is" -n 8 --greedy

# 2. quantize it (and the engine runs the quantized weight kernels)
minfer quantize /tmp/model-f16.gguf /tmp/model-q4_0.gguf --type q4_0
minfer /tmp/model-q4_0.gguf "The capital of France is" -n 8 --greedy

# 3. split a single file for transport
minfer split /tmp/model-q4_0.gguf /tmp/parts --max-size 200M
minfer /tmp/parts/model-q4_0-00001-of-00003.gguf "The capital of France is"

# round-trip (rewrite with the writer), bit-exact
minfer split /tmp/model-q4_0.gguf /tmp/rewrite --max-size 4G   # one part -> a plain file
```
