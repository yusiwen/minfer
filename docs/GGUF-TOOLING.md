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
| `Q4_0` | 18 | 32 | `ne[0]/32 × 18` | `d: f16`, `qs[16]` (element `j` low nibble, `j+16` high) |
| `Q4_1` | 20 | 32 | `ne[0]/32 × 20` | `d: f16`, `m: f16`, `qs[16]` |
| `Q5_0` | 22 | 32 | `ne[0]/32 × 22` | `d: f16`, `qh: u32` (5th bits, element `j` bit `j`), `qs[16]` |
| `Q5_1` | 24 | 32 | `ne[0]/32 × 24` | `d: f16`, `m: f16`, `qh: u32`, `qs[16]` |
| `Q8_0` | 34 | 32 | `ne[0]/32 × 34` | `d: f16`, `qs[32]: i8` |
| `Q4_K`/`Q5_K`/`Q6_K` and every other type | — | — | — | copied verbatim (rewrite/split only; **no encoder**) |

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
minfer convert  <hf-model-dir> <out.gguf> [--outtype f16|f32] [--split-max-size N]
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
- `general.file_type` (1 = f16, 0 = f32), `general.quantization_version = 2`
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

### 2.2 `quantize` — re-encode a single-file GGUF

Targets with an implemented and byte-verified encoder:
**`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`**, plus the `f16` / `f32` element casts.

- For a **quant** target only 2-D float tensors are quantized. 1-D tensors (norms,
  biases) and any tensor whose row length is not a multiple of the target block
  size keep their source type, and the CLI prints the list — llama.cpp's rule, not
  a silent choice.
- The `f16` cast follows the same "except 1d tensors" rule: 2-D tensors become
  **f16**, 1-D tensors (norms, biases) keep their source type — **f32** in every
  file `minfer convert --outtype f16` or `llama-quantize … F16` writes, and the
  only type the engine's norm/bias path reads (an f16 norm is a file the engine
  cannot run). The CLI prints the preserved list exactly as for a quant target.
  `minfer quantize --type f16` now produces a runnable file this way
  ([#169](https://github.com/yusiwen/minfer/issues/169)).
- The `f32` cast is the one target that converts **every** tensor, 1-D included,
  to f32.
- On a **tied** model (no `output.weight`), a sub-8-bit target quantizes the
  shared `token_embd.weight` at **q8_0** — llama.cpp's tied-embedding policy.
  The CLI prints this too.
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

minfer convert: --outtype bf16 is not supported; supported output types: f16,
  f32 (a bf16 writer is a follow-up — f32 preserves every bf16 value exactly)

minfer convert: <dir>/tokenizer_config.json has no 'chat_template'; minfer's
  strict loader renders the model's own template, so a converted GGUF without
  one is not accepted

minfer quantize: --type is required (no silent default: a wrong target would
  write wrong weights); supported: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32

minfer quantize: target "q4_K" is a known GGUF type but minfer has no weight
  encoder for it (minfer can only read it, so writing it would emit wrong
  weights); supported encoder targets: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32

minfer quantize: unknown quant target "banana"; supported: q4_0, q4_1, q5_0,
  q5_1, q8_0, f16, f32

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
| bf16 → f16 | **exact in the mantissa** (bf16 has 8 mantissa bits, f16 has 10) but can **overflow**: a bf16 value beyond f16's ±65504 becomes ±inf. No saturation is applied, so the overflow is visible rather than a silently clamped weight. |
| f32 → f16 | **not exact** — round-to-nearest-even, and the same overflow rule |
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
information was lost — bf16 has 8 mantissa bits and f16 has 10, so every bf16
value is representable, and the exactness held on this checkpoint.

### 4.2 The reference the encoders were checked against

`llama-quantize ref-f16.gguf ref-<type>.gguf <type>` on the same f16 source.
For **each** of q4_0, q4_1, q5_0, q5_1, q8_0, all **290/290 tensors are
byte-identical**.

One subtlety is worth recording because it is not obvious from the reference
source: llama.cpp computes `x*id + c` and is compiled with
`-ffp-contract=fast`, so the CPU reference contracts that expression to an FMA.
Without `f32::mul_add` the q4_0 encoder differed from `llama-quantize` in
**12 of 64512 bytes** on one tensor — each a single nibble off by one, at an
exact rounding boundary. With the FMA the difference is zero. Q8_0 uses a single
multiply and matched without it.

The pure encoder gate (`quantize::tests`) additionally pins the block layout
against hand-checked reference numbers: the scale, the `j` / `j+16` nibble
packing, the 5th-bit plane, `type_size`/`blck_size`, and the zero-block case.

### 4.3 Tolerances

| Comparison | Tolerance | Measured |
|---|---|---|
| rewrite vs source logits | **bitwise** (`assert_eq!`) | equal |
| minfer-converted vs llama.cpp-converted logits (same engine) | **bitwise** | equal |
| split vs unsplit logits | **bitwise** | equal |
| `f16 → q8_0` logits, same context | max \|Δ\| ≤ 1.0 **and** greedy text identical | max \|Δ\| = **0.481**, mean 0.082, max \|logit\| = 18.43 (2.6% of the largest logit); greedy continuation identical |
| an f16 file's CUDA logits vs the same file's CPU logits (#141, 34-token prompt, ctx 512, Qwen2.5-0.5B-Instruct f16) | max \|Δ\| ≤ **0.01** and max relative ≤ **1e-3**, greedy continuation identical | max \|Δ\| = **7.34e-5**, mean 1.26e-5, max \|logit\| = 18.43 (**4.0e-6** relative); greedy `[12095, 13, 1084, 374]` on both |
| that f16 file under llama.cpp (same prompt, `--temp 0`) | — | `Paris.`, the same greedy continuation minfer produces on CPU and CUDA |

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

- **No K-quant encoders.** `q4_K`/`q5_K`/`q6_K` (readable by the engine) and
  every I-quant are refused by name for `quantize`. [#140](https://github.com/yusiwen/minfer/issues/140)
- **f16 weights run on CPU and CUDA for both architectures (Qwen2/Qwen2.5 and
  Qwen3); Metal refuses them.**
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
  quantized bytes) and runs the f32-activation kernel instead. **Metal has no
  f16 weight kernel yet**, so an f16 GGUF there falls to the CPU through the
  loader's all-or-nothing registration check — loudly, because registering a
  weight type no kernel can consume would be a silent wrong path
  ([#164](https://github.com/yusiwen/minfer/issues/164)). Completed by
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
- **No `--outtype bf16`.** f32 preserves every bf16 value exactly, so nothing is
  lost today, but a bf16 writer (and a bf16 weight path) is [#142](https://github.com/yusiwen/minfer/issues/142).
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
