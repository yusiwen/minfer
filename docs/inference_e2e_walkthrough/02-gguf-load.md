# 02 · GGUF load — from file bytes to a model handle

> **Stage**: model path resolved ([01](01-cli-args-model-resolution.md)) →
> **GGUF load: parse header + metadata, mmap the data blob** →
> model dispatch + weight registration ([03](03-model-dispatch-weights.md)).
> **Code**: `src/gguf.rs` — `load_gguf_model` (line 1998), `GgufContext::init_from_reader`
> (line 1021), `MmapFile` (line 1866), `ggml_pad` (line 328); call site
> `src/main.rs` line 579. Block layouts: `src/block.rs`.

## 1. Background — where this stage sits

Doc 01 ended with a plain string: a filesystem path to a `.gguf` file (resolved
from a local path, a `hf:`/`ollama:` URI, or a cached model name). This doc is
where that path stops being a name and starts being data. When `main.rs` runs
`gguf::load_gguf_model(...)` (line 579), the engine has exactly one asset: a
file on disk, typically hundreds of megabytes to several gigabytes. When the
call returns, the engine holds a `GgufModel` handle: the model's hyperparameters
and tokenizer sit in a parsed key-value table, every weight tensor is catalogued
by name/type/shape, and the raw weight bytes are reachable in memory without
having copied a single one of them.

First, the vocabulary, because everything else builds on these five words:

- **GGUF** (GPT-Generated Unified Format) is llama.cpp's single-file model
  container. One file holds *everything* needed to run the model: the weight
  tensors, the hyperparameters (layer count, head count, …), the tokenizer's
  vocabulary, and the chat template. Version 3 is the current revision; minfer
  reads exactly that (`GGUF_VERSION = 3`, `gguf.rs` line 10).
- A **tensor** is an n-dimensional array of numbers. A transformer's weights are
  a few hundred of them: embeddings, per-layer projection matrices, norm
  vectors. In this engine a tensor is just *named bytes* — the GGUF file stores
  each one under a name like `blk.0.attn_q.weight` with a shape and a
  quantization type.
- **Quantization** is compressing those numbers: instead of storing each weight
  as a 4-byte `f32`, store it in fewer bits (4.5 bits per weight for Q4_0) by
  sharing one scale factor across a small *block* of values. The details are
  §2.4 and doc 10; for this doc you only need the consequence: **a tensor's byte
  size is not "element count × 4"** and the parser must know each type's block
  arithmetic to slice tensors correctly.
- **Metadata KV** (key-value) pairs are the file's self-description: strings,
  numbers, and arrays such as `general.architecture = "qwen2"` or
  `tokenizer.ggml.tokens = [151,936 strings]`. Think of them as a JSON document
  embedded at the front of a binary file.
- **mmap** (memory mapping) asks the operating system to make a file's bytes
  appear at an address in the process's address space. Nothing is read up front:
  the first touch of any page triggers an OS fault that pulls that 4 KiB page
  from disk. The process reads file bytes through an ordinary slice, and the
  OS page cache (shared memory of disk pages, reused by all processes) does the
  caching once, for everyone.

Why does an engine written from scratch need its own GGUF parser? Because there
are no ML-framework dependencies to lean on — minfer's whole `gguf.rs` is a
faithful Rust port of llama.cpp's `gguf.cpp`, down to the `gguf.cpp` line
numbers in the comments. The parser is deliberately *dumb but strict*: it reads
the container faithfully, checks every length and offset, and hands clean data
to the next stage. It does not interpret the model — deciding that
`general.architecture = "qwen2"` means "build a Qwen2 transformer" is doc 03's
job.

What would break without this stage? Everything downstream. Doc 03 needs the
metadata to pick an architecture and the tensor table to find
`blk.{i}.attn_q.weight` by name. Doc 04 needs the tokenizer arrays. Docs 07–08
need tensor *strides* (byte distances between rows) to be ggml-compatible so
kernels index memory correctly. And doc 14's zero-copy Metal path needs the
weight bytes to live in an mmap region — which is why this doc, not doc 03, is
where the mmap is created. The load stage is also the last line of defense
against a corrupt file: a truncated download or a misaligned write is caught
here, with a message, instead of producing garbage logits at step 200.

## 2. Principle — how it works and why

### 2.1 The file is four regions, read in three

A GGUF v3 file is a strictly ordered sequence of regions:

```text
byte 0 ┌──────────────────────────────┐
       │ magic "GGUF" (4 B)           │  ┐
       │ version  u32 (4 B)           │  │
       │ n_tensors i64 (8 B)          │  ├─ header: 24 bytes, always
       │ n_kv      i64 (8 B)          │  ┘
       ├──────────────────────────────┤
       │ metadata KV pairs (n_kv of)  │   key string, type tag, value
       │  · scalars: u32/i32/f32/…    │   arrays: elem type + count + data
       │  · strings: u64 len + bytes  │   (the tokenizer vocab lives here:
       │  · arrays of the above       │    151,936 strings ≈ megabytes)
       ├──────────────────────────────┤
       │ tensor info table (n_tensors)│   name, n_dims, shape[nd], ggml
       │                              │   type, offset (from data start)
       ├──────────────────────────────┤
       │ (padding to `general.alignment`, default 32 B)
       ├──────────────────────────────┤
       │ tensor data blob             │   quantized blocks, back to back,
       │  tensor 0 | tensor 1 | …     │   each padded to the alignment
       └──────────────────────────────┘   (offsets are absolute within it)
```

The parser consumes the first three regions with a cursor (`GgufReader`) and
*stops*. The fourth region — 99% of the file — is never read by the parser at
all. We measured this on a real file (Qwen2.5-0.5B-Instruct Q4_0, 428,730,208
bytes): the header is 24 bytes, the 26 KV pairs end at byte 5,931,189 (the
tokenizer vocabulary and BPE merges are the bulk of it), the 291-entry tensor
table ends at byte 5,947,741, three alignment pad bytes follow, and the data
section starts at byte 5,947,744. That is **1.39% of the file parsed and 98.61%
left untouched** — untouched not skipped-and-copied, but never read, because the
data region becomes an mmap that the OS pages in lazily, on demand, forever
after.

This is the first design decision worth internalizing: *loading* a model must
not mean *reading* it. `Loading model: … → Model loaded.` prints in tens of
milliseconds precisely because the only O(file-size) work is the kernel setting
up the mapping (a few syscall-rounds), not transferring gigabytes.

### 2.2 Alignment: why every region is padded, and to what

The header, KV block, and tensor table are all variable-length — strings and
arrays make their ends land on arbitrary byte offsets. But the data section
must start at an address that is friendly to consumers: SIMD loads want 16–64
byte alignment, GPU buffer offsets want more. GGUF solves this with one rule:
**the data section starts at the next multiple of `general.alignment` (default
32)**, and **every tensor's data is padded up to a multiple of that alignment
too**, so tensor N starts right where tensor N−1 ended.

The padding operation is ggml's `GGML_PAD` macro, ported verbatim (`gguf.rs`
lines 327–332):

```rust
#[inline]
pub fn ggml_pad(x: usize, n: usize) -> usize {
    // ((x) + (n) - 1) & ~((n) - 1)
    // Assumes n is power of 2
    (x + n - 1) & !(n - 1)
}
```

One line of bit tricks: add `n−1` to force a carry past the next boundary, then
mask off the low bits so only multiples of `n` survive. For `n = 32` the mask
is `!31` = `…11100000`. Worked example: the tensor table of the 0.5B file ends
at byte 5,947,741; `(5,947,741 + 31) & !31` = `5,947,744`, so 3 pad bytes are
inserted. Because the trick *requires* a power of two, the parser rejects any
`general.alignment` that isn't one (line 1488) rather than silently rounding
wrong.

Why does the parser care at all — couldn't each consumer align itself? Because
the offsets are *stored in the file*: each tensor's `offset` field is relative
to the (aligned) start of the data section, and tensor bytes must be found at
exactly that offset. Three things go wrong if padding is ignored or miscomputed:

1. **Wrong slices.** Offsets would drift by the pad bytes; every tensor after
   the first misalignment would be cut from shifted bytes — a garbled model
   that produces fluent nonsense or crashes in a kernel far from the cause.
   The parser defends itself: it re-derives each tensor's expected offset as a
   running sum of padded sizes and rejects the file if the stored offset
   disagrees (lines 1692–1713).
2. **Slow or faulting vector loads.** The CPU kernels read quantized blocks as
   16/32-byte SIMD chunks; a base pointer off by 3 bytes splits every chunk
   across cache lines (x86 tolerates this slowly; some aarch64 NEON loads
   fault on misalignment). Doc 10's kernels assume ggml's alignment exactly.
3. **GPU mapping breaks.** Metal's zero-copy path wraps the mmap in a GPU
   buffer, and `newBufferWithBytesNoCopy` requires a *page-aligned* base
   address (16,384 bytes on Apple Silicon). minfer satisfies this by mapping
   the *whole file* from offset 0: the mapping base — hence
   `part.data.as_ptr()` — is page-aligned by construction, and every tensor is
   then addressed as an offset into that one buffer (§3.2 l). Per-tensor
   32-byte alignment keeps those offsets sane for SIMD loads; the page
   alignment comes from the mapping itself.

The alignment is "per-region" in one more sense: it is applied twice, once
before the data section (after the table) and once after *each tensor* inside
it. Both applications use the same `ggml_pad`, and the running-sum check verifies
both. A writer that forgot interior padding would fail the parse rather than
poison memory.

### 2.3 Quantized sizing: 32 values in 18 bytes

Quantization is the reason this 630M-parameter model (its own
`general.size_label` metadata) fits in 409 MiB of Q4_0/Q8_0 bytes instead of
the ~2.3 GiB its f32 form would take. The scheme (fully dissected in doc 10)
is *block quantization*: group
values into blocks, store one shared `f16` scale per block, and store each
value as a small integer relative to that scale. Two block shapes exist:

- **32-value blocks** ("Q4_0 family"): `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`.
- **256-value super-blocks** ("K-quants"): `Q4_K`, `Q5_K`, `Q6_K`, … — 256
  values per super-block, subdivided into 8–16 sub-blocks that each get their
  own quantized scale.

The canonical example is Q4_0. Its 32 values are stored in 18 bytes — an `f16`
(2-byte) scale plus 16 bytes holding 32 4-bit nibbles:

```text
BlockQ4_0 (18 bytes)                      value ≈ d × (q − 8)
┌─────────────┬───────────────────────────────┐
│ d : f16 (2B)│ qs : 16 bytes = 32 × 4 bits   │
└─────────────┴───────────────────────────────┘
  18 × 8 / 32 = 4.5 bits per weight
```

The layouts are `repr(C)` structs in `block.rs` mirroring llama.cpp's
`ggml-common.h` — `BlockQ4_0 { d: Fp16, qs: [u8; 16] }` (lines 51–56) — with
compile-time asserts that the Rust sizes equal the C sizes (lines 190–204). The
type table in `gguf.rs` is what the parser actually consults (`type_size()`,
lines 200–246). Every supported type, with its block arithmetic:

| Type | values/block (`blck_size`) | bytes/block (`type_size`) | bits/weight | anatomy |
|---|---:|---:|---:|---|
| F32 | 1 | 4 | 32.0 | raw |
| F16 | 1 | 2 | 16.0 | raw |
| Q4_0 | 32 | 18 | 4.5 | f16 d + 16 B nibbles |
| Q4_1 | 32 | 20 | 5.0 | f16 d + f16 m + 16 B nibbles |
| Q5_0 | 32 | 22 | 5.5 | f16 d + 4 B high bits + 16 B nibbles |
| Q5_1 | 32 | 24 | 6.0 | f16 d + f16 m + 4 B high + 16 B nibbles |
| Q8_0 | 32 | 34 | 8.5 | f16 d + 32 i8 |
| Q4_K | 256 | 144 | 4.5 | f16 d + f16 dmin + 12 B scales + 128 B nibbles |
| Q5_K | 256 | 176 | 5.5 | f16 d + f16 dmin + 12 B scales + 32 B high + 128 B nibbles |
| Q6_K | 256 | 210 | 6.5625 | 128 B low + 64 B high + 16 i8 scales + f16 d |

(source: `gguf.rs` lines 200–283; struct comments in `block.rs` lines 16–27,
135–167. Q8_K at 290 bytes is the *activation* format — it appears in files
rarely and is used at runtime on CPU; doc 10 covers it.)

Note the two `f16` fields in a Q4_K block are the *super-block* scale and min;
the 12-byte `scales` field packs eight 6-bit sub-block scales *and* eight
6-bit mins (the `unpack_q4k_scales` bit-shuffling in `block.rs` lines 32–44
decodes it). That two-level structure is the whole trick of K-quants: fine
scales capture local variance, the super-scale keeps them honest.

The byte-size formula follows directly from the table: **blocks per tensor ×
bytes per block**. `ggml_nbytes` (`gguf.rs` lines 828–849) implements it via
the strides (next section); in its simplest one-dimensional reading it is
`n_elements / blck_size × type_size`. Real numbers for the 0.5B model's largest
tensor, `token_embd.weight` (shape `[896, 151936]` = 136,134,656 elements, on
disk as Q4_0):

| Layout | bytes | MiB |
|---|---:|---:|
| F32 | 544,538,624 | 519.3 |
| Q8_0 | 144,643,072 | 137.9 |
| **Q4_0 (as on disk)** | **76,575,744** | **73.0** |
| Q4_K | 76,575,744 | 73.0 |
| Q6_K | 111,672,960 | 106.5 |

Q4_0 and Q4_K land on the same 4.5 bits/weight — the K-quant's extra scale
structure buys *quality* at equal size, which is why Q4_K_M variants are the
popular download. And note the file itself uses different types for different
tensors: in this very file `output.weight` is Q8_0 (137.9 MiB — the output
projection is quality-sensitive) while `token_embd.weight` is Q4_0 (73.0 MiB),
and the norm vectors are F32 (896 × 4 = 3,584 bytes each). The parser never
assumes a uniform type; it reads each tensor's type tag and computes sizes
per tensor.

### 2.4 Strides: the shape's byte-geometry, computed once in the parser

A **stride** is the number of bytes you skip to move one step along a
dimension. GGML (and therefore GGUF) stores shapes as `ne[4]` (elements per
dimension, `ne[0]` fastest-varying — for weight matrices `ne[0]` is the input
dim, `ne[1]` the output dim) and strides as `nb[4]` (bytes per step). Strides
are what make a kernel able to walk a tensor without knowing what a
"Q4_K super-block" is: a row is `nb[1]` bytes away from the previous row,
period.

The parser computes strides the instant it learns shape + type (`gguf.rs`
lines 1650–1655):

```rust
// calculate byte offsets (gguf.cpp lines 728-732)
info.nb[0] = type_size;
info.nb[1] = info.nb[0] * (info.ne[0] / blck_size) as usize;
for j in 2..GGML_MAX_DIMS {
    info.nb[j] = info.nb[j - 1] * info.ne[j - 1] as usize;
}
```

Read it as three claims:

- `nb[0] = type_size` — the atom of dimension 0 is one *block*, not one
  element. For F32 that's 4 bytes per step; for Q4_0 that's 18 bytes per step
  (one block of 32 values).
- `nb[1] = nb[0] × ne[0]/blck_size` — one row contains `ne[0]` values, which is
  `ne[0]/blck_size` blocks, so a row is that many block-sizes long. For a
  Q4_0 row of `ne[0] = 896`: `896/32 = 28` blocks × 18 B = 504 bytes per row.
- Every higher stride multiplies: `nb[2] = nb[1] × ne[1]` (one full plane),
  `nb[3] = nb[2] × ne[2]`.

`ggml_nbytes` then totals it: dimension 0 contributes `ne[0]/blck_size`
blocks' worth (`ne[0] × nb[0] / blck_size`), each higher dimension contributes
`(ne[i] − 1) × nb[i]` (the last step doesn't add bytes). The division-aware
form is exactly what keeps the 4.5-bit types honest — you cannot compute this
size without the block size, which is why the parser demands
`ne[0] % blck_size == 0` (line 1627) and fails otherwise (a Q4_0 tensor with
895 elements per row is unrepresentable, not merely odd).

Two facts make this stage's stride work load-bearing. First, the *same*
formula is reimplemented at tensor-creation time (`tensor.rs` lines 142–149,
`loader.rs` lines 192–197), so the GGUF-parsed strides and the in-memory
`Tensor.strides` agree by construction — kernels can be written against one
geometry. Second, the geometry is ggml's, byte for byte, which is what allows
minfer to consume llama.cpp-quantized files *and* to be compared against
llama.cpp numerically (docs record the greedy-output matches).

### 2.5 mmap: borrowing the file instead of owning a copy

The obvious loader reads the file into a `Vec<u8>` (`read()`-into-heap) and
hands out slices of that vector. minfer instead calls `mmap` and hands out
slices of the *mapping*. The difference is not academic:

1. **Load latency.** `read()` copies every byte before "Model loaded." can
   print; mmap copies no bytes at all. The kernel just installs a page table
   entry: microseconds, independent of file size.
2. **RAM footprint vs file size.** With `read()`, RSS (resident memory, the
   pages actually in RAM) immediately includes the whole 4 GB. With mmap, RSS
   grows only as pages are touched — the embedding table and the layers you
   actually use — and the OS can evict cold pages under pressure because it
   knows they're backed by a file it can re-read.
3. **Page cache sharing.** The OS keeps one copy of the file's pages in its
   page cache. mmap'd readers attach to it: a second minfer process on the same
   model adds no new copies, and `download`'s freshly written file is already
   warm.
4. **Zero-copy GPU mapping** (the payoff that dominates later docs). Because
   the weight bytes live in one stable, page-aligned mapping for the life of
   the process, Metal can wrap that *same* memory in a `MTLBuffer` with
   `newBufferWithBytesNoCopy` — the GPU reads the model file through the page
   cache with no upload step at all (doc 14). That is only possible because
   the data was never copied into a heap allocation; the mapping *is* the
   storage.

The cost of mmap is honesty about lifetime: the mapping must outlive every
slice borrowed from it. minfer's answer is the simplest correct one — the
`MmapFile` is deliberately *leaked* (`Box::leak`, line 1999) so its backing
memory lives until process exit, and every tensor slice is `&'static [u8]`.
For a process whose whole purpose is to run one model, that is not a leak in
any meaningful sense; it is a pool that is freed by `exit`. (The alternatives
— `Arc<MmapFile>` reference counting, or self-referential structs — buy
nothing here and cost real complexity; §3.3.)

### 2.6 Multi-part files: many GGUFs, one model

Models above a few GB are published as *splits* — `name-00001-of-00002.gguf`,
`…-00002-of-00002.gguf` — because of file-size limits on hosting platforms.
Each part is a *complete, valid GGUF file*: own header, own (small) metadata,
own tensor table, own data blob. The metadata in part 0 carries
`split.no = 0`, `split.count = 2`, `split.tensors.count = 339` (verified in a
real 7B Q4_K_M part 0). Loading = mmap and parse each part in order, then
*concatenate the tensor catalogs*: the merged index maps every tensor name to
"the part that lists it", and a lookup slices from that part's own mapping.
Doc 03's loader builds exactly that map (`loader.rs` lines 329–343); this doc's
`load_gguf_model` produces the `parts: Vec<GgufPart>` it iterates.

The merge rule mirrors llama.cpp: **entry = part 0**. Part 0's metadata is *the*
model metadata (architecture, tokenizer, template — the 7B part 0 has 29 KV
pairs where a non-split file of the same family has 26, the extras being the
`split.*` trio), and part 0 must be the file you point minfer at (`split.no != 0`
is rejected, line 2020). Tensors are distributed round-robin-ish by the
quantizer; minfer never assumes where, it just looks the name up in whichever
part claims it.

### 2.7 What the parse hands to doc 03

The product of this stage is one struct, and its shape is the contract with
everything after:

```text
GgufModel
└── parts: Vec<GgufPart>            one per file; [0] is the entry
    ├── ctx: GgufContext            parsed regions (header + KV + tensor table)
    │   ├── version, alignment      format facts
    │   ├── kv: Vec<GgufKv>         metadata: keys → typed values/arrays
    │   └── info: Vec<GgufTensorInfo>  name, ne[4], nb[4], type, offset
    └── data: &'static [u8]         the mmap'd file bytes (whole file)
```

Who reads what next:

| Metadata key(s) | Consumer | Doc |
|---|---|---|
| `general.architecture` | `models/mod.rs::load_model` (line 97) → `"qwen2"` / `"qwen3"` dispatch | 03 |
| `qwen2.block_count`, `qwen2.embedding_length`, `qwen2.attention.head_count(_kv)`, `qwen2.feed_forward_length`, `qwen2.context_length`, `qwen2.attention.layer_norm_rms_epsilon`, `qwen2.rope.freq_base`, `qwen2.rope.frequency_scale` (+ `llama.*` aliases) | `models/qwen2/loader.rs::hparams_from_gguf` (lines 118–150) → `HParams` | 03 |
| tensor table (`info`) | loader's merged tensor map → `Tensor`s → GPU registration | 03 |
| `tokenizer.ggml.tokens/scores/token_type/merges/bos_token_id/eos_token_id` | `tokenizer.rs::Tokenizer::load` (lines 82–149) | 04 |
| `tokenizer.chat_template` | `main.rs::get_chat_template` (lines 1495–1504) | 04 |
| `split.no`, `split.count` | `load_gguf_model` itself (lines 2002–2025) | 02 |
| `general.alignment` | `init_from_reader` (lines 1481–1486) | 02 |

(`general.quantization_version` is present in files — value 2 on disk — but
minfer doesn't consult it; the ggml type tag per tensor is the operative fact.)

## 3. Implementation

### 3.1 Data in / data out

**In:** a `&Path` — one `.gguf` file, which is either a whole model or part
00001 of a split. Nothing else: no config, no sidecar files.

**Out:** `Option<GgufModel>` (`None` ⇒ error already printed, `main.rs` exits).
The two structs (`gguf.rs` lines 602–623, 1945–1954):

```rust
pub struct GgufTensorInfo {
    pub name: String,
    pub ne: [i64; GGML_MAX_DIMS],   // number of elements per dimension
    pub nb: [usize; GGML_MAX_DIMS], // stride in bytes per dimension
    pub type_: GgmlType,
    pub offset: u64, // offset from start of data section
}

pub struct GgufContext {
    pub version: u32,
    pub kv: Vec<GgufKv>,
    pub info: Vec<GgufTensorInfo>,
    pub alignment: usize,
    pub offset: usize, // offset of data section from beginning of file
    pub size: usize,   // size of data section in bytes
}

pub struct GgufPart {
    pub ctx: GgufContext,
    /// `'static` slice of the (leaked, process-lifetime) mmap of the part file.
    pub data: &'static [u8],
}

pub struct GgufModel {
    pub parts: Vec<GgufPart>,
}
```

A `GgufKv` (line 337) is `{ key, is_array, type_: GgufType, data: Vec<u8>,
data_string: Vec<String> }` — scalars live in `data` as little-endian raw
bytes, strings in `data_string`, so typed getters decode on demand.

The tensor *bytes* themselves are not yet wrapped as tensors — that is doc 03.
What this stage guarantees: for any `ti` in `ctx.info`, the bytes
`data[ctx.offset + ti.offset .. + ggml_nbytes(ti)]` are the tensor's full
contents, contiguous, and the address `data.as_ptr()` is page-aligned (it is
mmap's return value).

### 3.2 Key code

**(a) The format constants** (`gguf.rs` lines 9–19). Everything downstream of
this point is these numbers plus the type table:

```rust
const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const GGUF_KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

const GGUF_MAX_STRING_LENGTH: u64 = 1024 * 1024 * 1024;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1024 * 1024 * 1024;

// Note: GGML_MAX_DIMS and GGML_MAX_NAME from ggml.h
const GGML_MAX_DIMS: usize = 4;
const GGML_MAX_NAME: usize = 64;
```

The two `MAX_` caps are not decoration: every length read from the file is
checked against them *before* allocating (`read_string`, `read_vec`), so a
corrupt header cannot make the parser `malloc` 2⁶⁴ bytes and die. The parser
is, among other things, a hostile-input parser — model files come from the
internet.

**(b) The type table's arithmetic half** (`gguf.rs` lines 209–229). The
comments are the spec — each entry is `sizeof(struct)` spelled out:

```rust
// sizeof(block_q4_0) = sizeof(ggml_half) + QK4_0/2 = 2 + 16 = 18
GgmlType::Q4_0 => 18,
// sizeof(block_q4_1) = sizeof(ggml_half)*2 + QK4_1/2 = 2 + 2 + 16 = 20
GgmlType::Q4_1 => 20,
// sizeof(block_q5_0) = sizeof(ggml_half) + QK5_0/2 + QK5_0/8 = 2 + 16 + 4 = 22
GgmlType::Q5_0 => 22,
// sizeof(block_q8_0) = sizeof(ggml_half) + QK8_0 = 2 + 32 = 34
GgmlType::Q8_0 => 34,
// QK_K=256, super-block types — sizes from type_traits
GgmlType::Q4_K => 144,
GgmlType::Q5_K => 176,
GgmlType::Q6_K => 210,
GgmlType::Q8_K => 290,
```

The sibling method `blck_size()` (lines 250–283) returns values-per-block (32
for the first family, 256 for the K family, 1 for F32/F16). `type_size` and
`blck_size` together are the complete sizing algebra — everything in §2.3/§2.4
derives from them. (The full enum spans all 42 ggml type discriminants, lines
109–152, including the unsupported IQ/quaternion families — their entries
exist so the parser can *recognize and reject* files that need them.)

**(c) The reader's primitives.** All typed reads go through one function,
`read_val::<T>` (lines 905–918): bounds-check against `nbytes_remain`, copy
`size_of::<T>()` bytes, then `ptr::read_unaligned` them into a `T` — unaligned
because the *cursor* has no alignment guarantee (strings of odd length precede
most scalars). GGUF is little-endian; x86-64 and aarch64 are little-endian
hosts, so the byte copy *is* the decode, and the version field's own check
(below) is the early tripwire for byte-swapped files. Strings are `u64`
length + bytes (`read_string`, lines 938–962): the length is checked against
the 1 GiB cap *and* the remaining file size before any allocation, so a
corrupt header cannot make the parser malloc itself to death.

**(d) Header + version guards** (`gguf.rs` lines 1071–1096). After the 4-byte
magic comparison (lines 1033–1065, which prints the four offending characters
it found), the version is screened:

```rust
if let Some(version) = gr.read_val::<u32>() {
    ctx.version = version;
    if ctx.version == 0 {
        eprintln!("GGUF: bad GGUF version: {}", ctx.version);
        ok = false;
    }
    // endianness check (gguf.cpp lines 490-500)
    if ok && (ctx.version & 0x0000FFFF) == 0x00000000 {
        eprintln!("GGUF: failed to load model: this GGUF file version {} is extremely large, is there a mismatch between the host and model endianness?", ctx.version);
        ok = false;
    }
    if ok && ctx.version == 1 {
        eprintln!(
            "GGUF: GGUFv1 is no longer supported, please use a more up-to-date version"
        );
        ok = false;
    }
    if ok && ctx.version > GGUF_VERSION {
        eprintln!("GGUF: this GGUF file is version {} but this software only supports up to version {}", ctx.version, GGUF_VERSION);
        ok = false;
    }
}
```

The endianness heuristic is clever: if the file were written big-endian, the
u32 version 3 (`0x00000003` little-endian) would read back as `0x03000000`,
whose low 16 bits are zero — an "impossible" version number, reported with the
endianness hint instead of a mystery failure downstream.

**(e) The KV loop's type dispatch** (`gguf.rs` lines 1162–1190). Each pair is
key-string → type tag → (if array) element type + count → value:

```rust
let mut type_: GgufType;
let mut is_array: bool = false;
let mut n: u64 = 1;

match gr.read_gguf_type() {
    Some(t) => type_ = t,
    None => { ok = false; break; }
}

if type_ == GgufType::Array {
    is_array = true;
    match gr.read_gguf_type() {
        Some(t) => type_ = t,   // element type of the array
        None => { ok = false; break; }
    }
    match gr.read_val::<u64>() {
        Some(v) => n = v,       // element count
        None => { ok = false; break; }
    }
}
```

The 13 type tags (`GgufType`, lines 25–39) are u8…f64 plus string and array;
the `match type_` block that follows (lines 1197–1471) reads each, arrays
element-wise. Duplicate keys are rejected (lines 1151–1157) — llama.cpp relies
on unique keys and so does every consumer.

**(f) Per-tensor type checks and stride computation** (`gguf.rs` lines
1623–1655). This is where §2.3 and §2.4 become code, per tensor:

```rust
let type_size = info.type_.type_size();
let blck_size = info.type_.blck_size();

// check that row size is divisible by block size
if blck_size == 0 || info.ne[0] % blck_size != 0 {
    eprintln!("GGUF: tensor '{}' of type {} ({}) has {} elements per row, not a multiple of block size ({})",
        info.name, type_val, info.type_.type_name(), info.ne[0], blck_size);
    ok = false;
    break;
}

// check that size in bytes is representable
let nelements = ggml_nelements(&info.ne);
if ok && (nelements / blck_size) as u64 > (usize::MAX / type_size) as u64 {
    eprintln!(
        "GGUF: tensor '{}' with shape ({}, {}, {}, {}) has a size in bytes > {}",
        info.name, info.ne[0], info.ne[1], info.ne[2], info.ne[3], usize::MAX
    );
    ok = false;
    break;
}

// calculate byte offsets (gguf.cpp lines 728-732)
info.nb[0] = type_size;
info.nb[1] = info.nb[0] * (info.ne[0] / blck_size) as usize;
for j in 2..GGML_MAX_DIMS {
    info.nb[j] = info.nb[j - 1] * info.ne[j - 1] as usize;
}
```

Before this: the shape itself is validated (`n_dims ≤ 4`, lines 1548–1562;
negative dims rejected; a product-of-dims overflow check, lines 1585–1598).
After it: the tensor's data-section-relative offset is read as u64 (lines
1661–1668) — not computed, *read*, because the writer chose the layout.

**(g) Alignment, contiguity, total size** (`gguf.rs` lines 1679–1714). The
parse's final act is to pin down the data section and prove the tensor table
consistent with it:

```rust
// align to data section (gguf.cpp lines 751-756)
if n_tensors > 0 {
    let aligned_offset = ggml_pad(gr.tell() as usize, ctx.alignment);
    if !gr.seek(aligned_offset as u64) {
        eprintln!("GGUF: failed to seek to beginning of data section");
        return None;
    }
}

// store data section offset (gguf.cpp line 759)
ctx.offset = gr.tell() as usize;

// compute total data section size (gguf.cpp lines 762-782)
{
    ctx.size = 0;
    for i in 0..ctx.info.len() {
        let ti = &ctx.info[i];
        if ti.offset != ctx.size as u64 {
            eprintln!(
                "GGUF: tensor '{}' has offset {}, expected {}",
                ti.name, ti.offset, ctx.size
            );
            eprintln!("GGUF: failed to read tensor data");
            return None;
        }
        let padded_size = ggml_pad(ggml_nbytes(ti), ctx.alignment);
        if usize::MAX - ctx.size < padded_size {
            eprintln!(
                "GGUF: tensor '{}' size overflow, cannot accumulate size {} + {}",
                ti.name, ctx.size, padded_size
            );
            return None;
        }
        ctx.size += padded_size;
    }
}
```

The running-sum check is the quiet hero: it recomputes where each tensor
*must* start (previous end, padded) and compares with the file's claim. Our
real-file walk shows it passing exactly: `output.weight` (Q8_0) is 151,936 ×
896 / 32 blocks × 34 bytes = 144,643,072 bytes, sits at offset 0, and indeed
`token_embd.weight` (Q4_0) follows at offset 144,643,072;
`blk.0.attn_norm.weight` follows the 73 MiB embedding at
221,218,816 = 144,643,072 + 76,575,744. No drift, no gaps: quantized
arithmetic and padding compose perfectly.

**(h) The mmap** (`gguf.rs` lines 1861–1890 and 1892–1927). No `memmap2`
crate — raw libc, declared once:

```rust
/// Read-only mmap of a GGUF file part (zero-dependency: raw mmap/munmap via
/// the system libc, which Rust links by default). The file pages are shared
/// with the CPU and (via newBufferWithBytesNoCopy) the GPU instead of being
/// copied — llama's `llama_mmap` / `newBufferWithBytesNoCopy` equivalent
/// (ggml-metal-device.m:1668). MAP_PRIVATE (no writes happen), PROT_READ.
pub struct MmapFile {
    ptr: *mut u8,
    len: usize,
    #[allow(dead_code)]
    _file: std::fs::File, // keeps the fd alive for the mapping's lifetime
}

// Generic POSIX mmap (Linux + macOS; the syscall ABI is identical on both).
#[cfg(unix)]
const PROT_READ: i32 = 0x1;
#[cfg(unix)]
const MAP_PRIVATE: i32 = 0x0002;

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32,
            flags: i32, fd: i32, offset: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
}
```

and the map call:

```rust
pub fn map(path: &std::path::Path) -> Option<Self> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).ok()?;
        let len = file.metadata().ok()?.len() as usize;
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ,
                                MAP_PRIVATE, file.as_raw_fd(), 0) };
        // MAP_FAILED = (void*)-1
        if ptr as isize == -1 {
            return None;
        }
        // (The GPU-side warm-up read happens in metal.rs register_part —
        // the first GPU access to file-backed pages costs ~44 ms of one-time
        // page/TLB setup per process, METAL_OPTIMIZATIONS #39. A CPU-side
        // madvise/touch does NOT fix it — the cost is the GPU's own access.)
        Some(MmapFile { ptr: ptr as *mut u8, len, _file: file })
    }
}
```

Three details deserve their sentence each. `_file` is kept (though unused) so
the file descriptor cannot close — on some systems closing the fd is allowed
while mapped, but holding it makes the lifetime story trivially airtight.
`MAP_FAILED` is `(void*)-1`, not NULL, hence the `as isize == -1` check rather
than a null check. And the comment records a measured fact from
`docs/METAL_OPTIMIZATIONS.md` (#39): the *first* GPU touch of file-backed
pages costs ~44 ms of page/TLB setup, which Metal's `register_part` pays once
at load, on purpose, outside the timed region. `as_slice()`
(lines 1930–1932) turns the pointer into `&[u8]`; `Drop` (lines 1935–1942)
calls `munmap` — though with the leak below, it effectively never runs.

**(i) The entry point: parts, leak, merge** (`gguf.rs` lines 1998–2014):

```rust
pub fn load_gguf_model(path: &std::path::Path) -> Option<GgufModel> {
    let mmap0 = Box::leak(Box::new(MmapFile::map(path)?));
    let data0: &'static [u8] = mmap0.as_slice();
    let ctx0 = GgufContext::init_from_data(data0)?;
    let split_count = ctx0
        .get_key_val_i64("split.count")
        .map(|v| v as usize)
        .unwrap_or(1);

    if split_count <= 1 {
        return Some(GgufModel {
            parts: vec![GgufPart {
                ctx: ctx0,
                data: data0,
            }],
        });
    }
```

`Box::leak` is the `'static` trick from §2.5: `MmapFile::map` returns an owned
value; leaking it converts ownership into a process-lifetime guarantee, which
is exactly the borrow lifetime every downstream tensor slice needs. Then
either the fast path (no split: parse and return) or the split path — checks
first (`split.no` must be 0, lines 2016–2025; filename-derived part list must
match `split.count`, lines 2027–2034), then parse each part in order:

```rust
let mut parts = Vec::with_capacity(split_count);
for (i, p) in part_paths.iter().enumerate() {
    let mmap = Box::leak(Box::new(MmapFile::map(p)?));
    let data: &'static [u8] = mmap.as_slice();
    let ctx = GgufContext::init_from_data(data)?;
    let no = ctx
        .get_key_val_i64("split.no")
        .map(|v| v as usize)
        .unwrap_or(0);
    if no != i {
        eprintln!("GGUF: split {p:?} has split.no={no}, expected {i}");
        return None;
    }
    parts.push(GgufPart { ctx, data });
}
Some(GgufModel { parts })
```

Note what is *not* here: no merging of KV maps, no re-allocation of tensor
data, no concatenation of bytes. Each part keeps its own `ctx` and its own
mmap slice; the "one tensor index" is built lazily by doc 03's loader from the
parts (`tensor_map` over all `part.ctx.info`, loader.rs lines 331–337). The
parse's job is to make that trivial, not to do it.

**(j) Split filename parsing** (`gguf.rs` lines 1958–1974) — `split_file_info`
recognizes the `name-0000X-of-0000Y.gguf` pattern (exactly 5 digits, 1 ≤ X ≤
Y) and `resolve_splits` (lines 1979–1992) rebuilds the full ordered path list
from part 00001, rejecting non-first parts:

```rust
pub fn split_file_info(name: &str) -> Option<(String, usize, usize)> {
    let stem = name.strip_suffix(".gguf")?;
    let dash = stem.rfind("-of-")?;
    let idx_part = &stem[..dash];
    let count_str = &stem[dash + 4..];
    let idx_dash = idx_part.rfind('-')?;
    let prefix = &idx_part[..idx_dash];
    let idx_num = &idx_part[idx_dash + 1..];
    if idx_num.len() != 5 || count_str.len() != 5 {
        return None;
    }
    let idx: usize = idx_num.parse().ok()?;
    let count: usize = count_str.parse().ok()?;
    if idx == 0 || count == 0 || idx > count {
        return None;
    }
    Some((prefix.to_string(), idx - 1, count))
}
```

Both functions have unit tests right below (lines 2058–2096) covering the
2-part, 3-part, non-split, and invalid-index cases.

**(k) The call site's accounting** (`main.rs` lines 622–635). This is the
"File: … bytes … in N part(s)" line you see at startup, and the part-0
convention made explicit:

```rust
let n_parts = gguf_model.parts.len();
let total_bytes: usize = gguf_model.parts.iter().map(|p| p.data.len()).sum();
println!(
    "File: {} bytes ({:.1} MB) in {n_parts} part(s)",
    total_bytes,
    total_bytes as f64 / 1_048_576.0
);

let ctx = &gguf_model.parts[0].ctx;
if meta_flag {
    dump_gguf_metadata(ctx);
} else {
    println!("GGUF: {} KV, {} tensors", ctx.kv.len(), ctx.info.len());
}
```

For the 7B split on disk here: 3,993,201,344 + 689,872,288 = 4,683,073,632
bytes → `File: 4683073632 bytes (4466.1 MB) in 2 part(s)`. Note the sizes
summed are the *mmap lengths* — the whole files, data blob included — not the
data sections; this line is honest about disk footprint, while `ctx.size`
(from §3.2 g) is the tensor-data footprint.

**(l) The handoff, one step further** (`models/qwen2/loader.rs` lines
182–199, excerpted; doc 03 owns the full story). When the next stage wants a
weight, it computes the global offset and slices the mmap:

```rust
let off = ctx.offset + ti.offset as usize;
// Use GGML type for byte-size calculation — always correct regardless of TensorType mapping
let ts = ti.type_.type_size();
let bs = ti.type_.blck_size() as usize;
let n = (shape[0] * shape[1] * shape[2] * shape[3]) as usize;
let nbytes = (n / bs) * ts;
// Borrow the tensor bytes straight from the mmap'd part file (zero-copy —
// the file pages are shared with the CPU and GPU instead of a per-tensor copy).
let src = &raw[off..off + nbytes];

let mut strides = [0usize; 4];
strides[0] = ts;
strides[1] = strides[0] * (shape[0] / bs as i64) as usize;
for j in 2..4 {
    strides[j] = strides[j - 1] * shape[j - 1] as usize;
}

let mut tensor = Tensor::from_data_borrowed_with_strides(ttype, &shape, &strides, src);
```

The stride recomputation (loader lines 192–197) is deliberately identical to
the parser's (§3.2 f) — same formula, same result, and a cross-check: if the
two ever disagreed, tensors would be sliced with one geometry and walked with
another. Inside `Tensor` (`tensor.rs` lines 201–214), the slice becomes
`Cow::Borrowed`:

```rust
/// Create a weight tensor as a Borrowed slice of the mmap'd GGUF file
/// (zero-copy load — the file pages are shared with the CPU/GPU instead of
/// being copied per tensor). The slice must be 'static: the gguf loader
/// leaks the Mmap for the process lifetime.
pub fn from_data_borrowed_with_strides(
    ttype: TensorType,
    shape: &[i64; 4],
    strides: &[usize; 4],
    data: &'static [u8],
) -> Self {
    Tensor {
        ttype, shape: *shape, strides: *strides,
        data: std::borrow::Cow::Borrowed(data),
        name: String::new(),
    }
}
```

**Cow** (clone-on-write) is Rust's "borrowed until someone needs to modify
it" enum: scratch/activation tensors are `Cow::Owned(Vec<u8>)`, weights are
`Cow::Borrowed(&'static [u8])` — same type, two ownership stories, and a
`clone()` of a weight tensor copies *nothing* (the `t.clone()` at model call
sites is free for borrowed data; `graph/cpu_backend.rs` line 37 notes this).

And the final link in the zero-copy chain, doc 14's anchor
(`metal.rs` lines 2330–2344):

```rust
let page = 16384; // macOS page size on Apple Silicon
let base = data.as_ptr() as usize;
debug_assert!(base % page == 0, "mmap'd GGUF part not page-aligned");
let buf = unsafe {
    self.inner
        .device
        .newBufferWithBytesNoCopy_length_options_deallocator(
            NonNull::new(data.as_ptr() as *const std::ffi::c_void as *mut c_void)
                .unwrap(),
            (data.len() as u64) as usize,
            MTLResourceOptions::StorageModeShared,
            None,
        )
        .unwrap()
};
```

The GPU buffer *is* the file's pages — `data` here is exactly the
`&'static [u8]` this doc's mmap produced, registered per part before any
weight is wrapped as `(buffer, offset)` (loader lines 322–327). mmap's
page-aligned return value is what makes `newBufferWithBytesNoCopy` legal; a
heap `Vec` could never qualify.

### 3.3 Design choices (why this shape and not another)

**mmap vs `read()`-into-heap.** §2.5 gave the four wins (latency, RSS
tolerance, page-cache sharing, GPU mapping). The honest costs: a mapping is
address-space (irrelevant on 64-bit), pages can fault mid-inference if the
file is truncated underneath you (llama.cpp documents this failure mode —
minfer's download layer size-checks resumes for the same reason, per
`docs/ARCHITECTURE.md` §9), and lifetime needs the leak discipline below. On
net, for multi-GB read-only files on machines that also want GPU access to
the same bytes, there is no contest — llama.cpp made the identical call
(`llama_mmap`), and minfer is explicitly in that lineage.

**Leak-for-`'static` vs `Arc<MmapFile>`.** The tensors that borrow the mapping
sit inside a `ModelDef` behind `Box<dyn ModelDef>`, get cloned, and get
registered by name in two or three registries (CPU tensors, Metal buffers,
CUDA device copies). Threading an `Arc` through all of them (or a
self-referential struct crate) would infect every signature with a lifetime
story that never varies in practice: the model lives for the whole process.
`Box::leak` states that invariant once, at the only place that could violate
it. The `Drop` impl still exists and is correct — it runs if a non-leaked
mapping (e.g. a future tool) is dropped — but the load path never triggers it.

**Raw bytes now, decode never-at-load.** The tempting alternative is
"dequantize everything to f32 at load" — one clean uniform representation,
simple kernels. Why minfer doesn't: (1) it would quadruple memory (73.0 MiB →
519.3 MiB for one 0.5B-model tensor, from §2.3's table) and add a full
file-size pass to load time; (2) the GPU backends *want the quantized bytes* —
Metal kernels dequantize in registers and CUDA's MMQ path (int8 matrix-multiply
quantized, doc 15) multiplies in quantized space directly, so decoding at load
would *force* an f32 path that is both slower and numerically a different
model; (3) the CPU kernels' speed comes precisely from SIMT/SIMD-friendly
block layout (doc 10), not from pre-decoded rows. The engine's actual decode
budget is spent where it pays: on activations, per matmul, at runtime
("Activations stay f32 … CPU quantizes to Q8_0 on the fly",
`docs/ARCHITECTURE.md` §1.4).

**Parser strictness as a feature.** Every length is capped, every offset is
cross-checked, duplicates are rejected, alignment must be a power of two,
version > 3 fails. A more permissive parser would "work" until the first
corrupt file — and then mis-slice weights and hand the failure to a matmul
kernel hundreds of milliseconds later, ten layers deep in a forward pass. The
error messages (with the tensor name, the expected and found offsets) turn a
hex-editor session into a one-line diagnosis. The cost is a few branches per
metadata element; metadata is 1.39% of the file.

**Parse-then-borrow, not parse-and-copy.** `GgufContext` deliberately does
*not* hold tensor data (the `gguf.cpp`-parity comment says so, lines 621–622:
"data … handled by the caller"). The parse produces *coordinates*; the mmap is
the *territory*. That separation is what lets the same `GgufModel` serve the
CPU path (slice → `Cow::Borrowed`), the Metal path (`newBufferWithBytesNoCopy`
on the same pointer), and the CUDA path (device copies made once at
registration, from the same slices) without the parser knowing any backend
exists.

**Why strides are computed at parse time, not in the kernel.** GGUF stores
`ne[]` (shape) but *derives* `nb[]` (strides) — they are not on disk. Computing them at parse
time (rather than at kernel time) means every consumer agrees on geometry
before any kernel runs, and the divisibility check (`ne[0] % blck_size == 0`)
fails at load with the tensor's name, instead of as a mis-decoded value inside
a dot product. ggml computes identical strides in `ggml_new_tensor`; minfer
mirrors it in two places (parser + tensor factory) on purpose — the
redundancy is an assertion.

### 3.4 Pitfalls & invariants

- **Alignment must be a power of two** — `ggml_pad`'s masking is only correct
  then, and the parser enforces it (lines 1488–1491) instead of trusting the
  file. A `general.alignment` of 48 would otherwise round *down* for some
  offsets and corrupt every subsequent slice.
- **Tensors are contiguous, and the parse proves it.** Each stored offset must
  equal the running padded sum (lines 1694–1703). A file with gaps (or with
  padding computed at a different width) is rejected here — not at first
  inference.
- **The mapping must outlive every slice.** The `Box::leak` at lines 1999 and
  2038 is load-bearing: drop or unmap early and every weight tensor in the
  process is a use-after-free. Corollary: model loading is one-way — there is
  no unload-and-load-another within a process (the server keeps one model per
  slot, per `docs/ARCHITECTURE.md` §2).
- **`ne[0]` must be a multiple of the block size** (line 1627). This is why a
  model whose vocabulary isn't divisible by 32 cannot be exported as Q8_0
  without padding — `docs/QWEN3-SUPPORT-PLAN.md` records Qwen3's vocab
  151,936 passing exactly this check ("151936 (÷32 ✓ for Q8_0 blocks)").
- **Version guards are endianness-aware**: `version & 0xFFFF == 0` catches
  byte-swapped files before any misread scalar can do damage (lines
  1079–1082); v1 is refused outright, v4+ refused with a "file is newer than
  software" message (lines 1083–1092).
- **The data blob is never parsed, only addressed.** If any future code "just
  reads one tensor" during load by scanning bytes instead of seeking to
  `ctx.offset + ti.offset`, it breaks the lazy-paging model (and on a cold
  cache, load time). The invariant: the parser reads bytes [0, data_start) and
  nothing beyond.
- **Part 0 is the entry, always.** Loading `…-00002-of-00002.gguf` fails by
  design (`resolve_splits` rejects non-first parts, line 1982; `split.no`
  re-checked per part, lines 2041–2048), and the filename pattern must agree
  with `split.count` (lines 2028–2034). Doc 01's resolver can hand over any
  part of a split from a cache listing — this stage is where the wrong one is
  caught.
- **Name and key uniqueness**: duplicate metadata keys (lines 1151–1157) and
  duplicate tensor names (lines 1531–1540) abort the parse — consumers do
  linear scans or hash maps keyed by name, and "first match wins" would make
  file ordering semantic. It isn't.

## 4. Observe & verify

- **`minfer info <model>`** (`main.rs` line 492) runs this exact stage and
  dumps its output instead of continuing: you get the full metadata KV dump
  (`dump_gguf_metadata`, main.rs lines 1396–1449 — every key with its type and
  value, arrays itemized) followed by `dump_key_tensors`' name/type/shape
  table. On the 0.5B Q4_0 file: `n_kv=26`, `n_tensors=291`,
  `general.architecture = "qwen2"`, `qwen2.block_count = 24`,
  `output.weight` q8_0 `[896,151936]`, `token_embd.weight` q4_0 `[896,151936]`.
- **The normal startup lines** are this stage's stdout: `Loading model: …`,
  `File: N bytes (X MB) in P part(s)` (mmap lengths summed, lines 622–628),
  `GGUF: 26 KV, 291 tensors` (line 634), then `Model loaded.` after doc 03.
  On a split you'll see `in 2 part(s)`; the number printed is the sum of the
  parts.
- **`GgufContext::dump_metadata`** (`gguf.rs` lines 1721–1788) is a
  self-contained debug printer of the same data (kept `#[allow(dead_code)]`
  for tooling/tests).
- **Unit tests**: `cargo test gguf` runs the split-pattern tests
  (`split_file_info_parses_pattern` at line 2059,
  `resolve_splits_builds_all_parts` at line 2078) — the filename grammar and
  part ordering, no model file needed.
- **The failure side is observable too**: point minfer at a non-GGUF file and
  you get `GGUF: invalid magic characters: '…', expected 'GGUF'` then
  `Error: failed to parse GGUF: …` (main.rs lines 614–619); at a directory you
  get a candidate list (lines 582–610); at part 2 of a split, the `split.no`
  error.
- **What you can't see directly** (by design): the mmap. `top`/Activity
  Monitor show RSS climbing *during* the first prefill as tensor pages fault
  in, not during `Loading model` — that is §2.5's point made visible. The one
  mmap-related timing you may notice was paid deliberately: Metal's part
  warm-up (~44 ms page/TLB setup, `METAL_OPTIMIZATIONS.md` #39) happens at
  registration, outside the `Total:` timing.
- **`MINFER_TRACE` / `--dump-graph` / debug_dump** are graph-stage tools; they
  say nothing about this stage. The graph's weight *names* (visible in traces)
  are this stage's tensor names passed through untouched.

## 5. Cross-references

- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §6 — the quantization/tensor-layout
  summary this doc expands (ggml_pad, block sizes, split merging); §3 places
  the stage in the pipeline.
- [01 — CLI args and model resolution](01-cli-args-model-resolution.md) — where
  the path came from (download/cache resolution, size-checked resume — the
  reason truncation-induced mmap faults are rare).
- [03 — Model dispatch and weights](03-model-dispatch-weights.md) — the direct
  consumer: `general.architecture` dispatch, `HParams` from `qwen2.*` keys,
  the merged tensor map, `Tensor` creation, GPU registration.
- [04 — Tokenizer and template](04-tokenizer-template.md) — the other consumer
  of the KV map (`tokenizer.ggml.*`, `tokenizer.chat_template`).
- [10 — CPU matmul kernels](10-cpu-matmul-kernels.md) — the block layouts
  catalogued here, used in anger (dot products on 18-byte Q4_0 blocks).
- [14 — Metal backend](14-metal-backend.md) — the payoff of mmap:
  `newBufferWithBytesNoCopy`, `(buffer, offset)` weight wrapping, part warm-up.
- [15 — CUDA backend](15-cuda-backend.md) — the third consumer of the same
  slices (registration-time device copies; Q6_K's padded-224 variant shows
  why per-type layout knowledge matters end to end).
- [`docs/GLOSSARY.md`](../GLOSSARY.md) — backstop definitions for every term
  used here.
- llama.cpp provenance: `gguf.rs` comments cite `gguf.cpp`/`ggml.c` line
  numbers throughout (e.g. lines 334, 600, 851) — the parser is a readable
  diff against upstream.

← [01 — CLI args and model resolution](01-cli-args-model-resolution.md) · [Index](./README.md) · [03 — Model dispatch and weights](03-model-dispatch-weights.md) →
