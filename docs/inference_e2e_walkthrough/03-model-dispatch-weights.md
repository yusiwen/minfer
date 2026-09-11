# 03 · Model dispatch and weights

> **Stage**: GGUF parsed (02) → **this stage: the file becomes a runnable
> model, with weights wired into every backend** → tokenizer + template (04).
> **Code**: `models/mod.rs::load_model` (dispatch), `models/qwen2/loader.rs::load`
> and `models/qwen3/loader.rs::load` (the two implementations), `main.rs:637-657`
> (GPU init + legacy KV cache), `tensor.rs` (`Cow<'static,[u8]>` weight bytes),
> `graph/alloc.rs::register_weight`, `metal.rs::register_weight` /
> `cuda.rs::register_weight` (per-backend registries),
> `models/qwen2/graph.rs::weights_on_gpu` (the GPU participation gate).

---

## 1. Background — where this stage sits

Doc 02 left us holding a `GgufModel`. It is a *container*, not yet a *model*:
a parsed metadata table (key/value pairs), a tensor index (name, quant type,
shape, offset for every tensor), and the raw data blob of the file memory-mapped
into the process. Nothing in that container knows what a transformer is. If you
asked it "how many layers does this model have?", it can only answer "there is
a metadata key somewhere that might say". The bytes are all there, but they
have no meaning yet.

This stage gives them meaning. Two things happen, and they happen in this
order. First, the engine initializes its GPU backends — Metal on macOS,
CUDA when the binary was built with `--features cuda` and an NVIDIA GPU is
present. Second, the engine looks at one metadata string,
`general.architecture`, and dispatches the whole file to the model
implementation that matches it: `"qwen2"` goes to the Qwen2/Qwen2.5 loader,
`"qwen3"` to the Qwen3 loader. The loader then reads the *hyperparameters*
(the dimension numbers of the architecture — layer count, head counts,
embedding width; these are the settings that were chosen before training and
are never learned) and gathers every *weight* tensor by name. A **weight** is
one block of the model's learned numbers: the big matrices that projections
multiply by, the small gain vectors of the normalization layers, and the
optional bias vectors. "Learned" means fixed by training; inference never
changes them.

The output of this stage is a single object behind the `ModelDef` trait —
for example `Qwen2Model` — holding hyperparameters plus every weight tensor,
with its bytes still sitting in the mmap'd file. Alongside it, each available
backend has been told about the weights it cares about. Those registrations
are what make the next stages work: doc 05 will build a compute graph whose
nodes *reference weights by name*, and docs 07–08 will allocate buffers and
execute ops that fetch their weight through the registries this stage fills.

Why is the ordering "GPU init first, then dispatch" and not the other way
round? Because loading is not a read-only operation. The loaders register
weight tensors into the GPU registries *while* they walk the tensor list, so
the registry must already exist. Get this wrong in one specific way — CUDA
initialized lazily in the middle of a load — and you get a half-registered
model whose backend gate flips between the first and the second half of the
tensor list. The loader defends against exactly that (we will see the guard
in §3.2).

What would break without this stage? Almost everything downstream, and in
ways that are loud rather than subtle. The graph builder needs the
hyperparameters to know how many layers to emit and what shapes the ops have.
The allocator needs `n_kv_embd` (the per-layer key/value width) to size the
persistent KV regions — a KV cache stores, per layer, the key and value
vectors of every token generated so far (doc 11 covers the mechanism). The
samplers need the end-of-sequence token ids, which are metadata this stage
parses. And the backends need the name→weight registries: a matmul whose
weight is not registered fails at execute time with `weight '...' not
registered`. In short, this stage turns a parsed file into a machine the rest
of the pipeline can drive. One thing to keep in mind from here on: **nothing
is dequantized at load** — the weights stay exactly the bytes the file
shipped, 4-bit nibbles and block scales and all. §2.3 explains why that is a
feature, not laziness.

## 2. Principle — how it works and why

### 2.1 The cast of characters

The stage has five players. It is worth naming them once, because the rest of
the doc is just their handshakes.

1. **`GgufModel`** (doc 02) — metadata key/values, the tensor index, and
   `&'static [u8]` slices into the mmap'd file parts.
2. **`load_model`** (`models/mod.rs`) — reads `general.architecture` and picks
   the implementation. This is the *dispatch*.
3. **The per-architecture loader** (`models/qwen2/loader.rs`,
   `models/qwen3/loader.rs`) — parses hyperparameters, gathers tensors by
   name, registers them with the backends, and assembles the model struct.
4. **`ModelDef`** (`models/mod.rs`) — the architecture-agnostic interface the
   rest of the engine talks to. Downstream code never says "Qwen2"; it says
   "whatever model is loaded, give me `n_layer()`, build me a graph, format
   this chat".
5. **The backend registries** — `CpuBackend` (a name→tensor map),
   `MpsState` (a name→(Metal buffer, offset) map), `CudaState` (a name→(device
   pointer, size) map). "Registry" here just means a hash map from a weight's
   GGUF name to whatever the backend needs in order to use it.

```
 GgufModel ──"general.architecture"──► load_model ──► qwen2::loader::load
                                                          │
                           ┌──────────────────────────────┴───────────────┐
                           ▼                                              ▼
                HParams (dims, token ids)               Tensor per weight (bytes
                           │                            borrowed from the mmap)
                           ▼                                              │
                 Qwen2Model : ModelDef ────t.clone()────► GraphAllocator  │
                           │                     (CPU: name → Tensor)    │
                           │                                              │
       GPU init happens BEFORE load:                                      │
         MpsState::init() → name → (MTLBuffer, offset) ───────────────────┤
         CudaState::init   → name → (device ptr, size) ───────────────────┘
                                          (graph nodes reference weights
                                           by name only, not by pointer)
```

### 2.2 Dispatch on a metadata string

GGUF writes the architecture family into a top-level metadata key. minfer
reads it once and matches it against the implementations it ships:

- `"qwen2"` → Qwen2 / Qwen2.5 (the 0.5B, 1.5B, 7B… checkpoints all report
  `qwen2`; Qwen2.5 differs from Qwen2 only in training, not in tensor layout).
- `"qwen3"` → Qwen3 dense (same overall wiring plus two twists we cover in
  §3.2: a decoupled head dimension and per-head Q/K norms).

The alternative — "try each loader until one succeeds" — is strictly worse,
and the reasons are worth spelling out because they shape the whole design.

**Determinism.** A string match is a total function: one key, one answer. A
trial-parse loop depends on what each loader happens to tolerate, and both of
minfer's loaders deliberately accept `llama.*` metadata keys as a fallback
(some fine-tunes re-label their metadata). Two tolerant loaders plus a
try-loop is a recipe for loading a Qwen3 file as "some kind of qwen2" — a
parse success that is semantically wrong and would corrupt attention shapes.

**One clear error.** When the string does not match anything, the user gets
`Unsupported architecture: 'llama'` — the actual offending string — and the
run stops before any GPU work happens. A try-loop's failure mode is instead
"N loaders each printed a different complaint", or worse, a quiet success.

**Load is side-effectful.** This is the mechanical clincher. Loading
*registers weights into GPU registries*, and on CUDA a registration is a
one-way upload into device memory. A "try and reject" loader would leave
half a model uploaded with no clean way to un-register (see §3.4: CUDA
deliberately never frees stale weight buffers). Dispatching first, then
loading exactly once, keeps the side effects all-or-nothing too.

### 2.3 Weights stay raw bytes

The single most important data decision of this stage: a loaded weight tensor
is a *view*, not a copy. The `Tensor` type stores its payload as
`std::borrow::Cow<'static, [u8]>` — a Rust "clone-on-write" enum that here is
always the `Borrowed` variant, i.e. a plain `(pointer, length)` slice pointing
into the mmap'd GGUF file. The `'static` lifetime works because doc 02's
loader `Box::leak`s each mmap for the process lifetime (`gguf.rs:1996-2000`).

Three consequences follow, and each one answers a "why not the obvious
alternative":

**Why not dequantize to f32 at load?** Because every consumer wants the bytes
*as they are*. The CPU matmul kernels (doc 10) consume packed 4-bit nibbles
directly — they dequantize a block on the fly inside the dot product, one
32-value block at a time, and never materialize the full f32 tensor. The GPU
kernels do the same in shaders (Metal) or stream raw quantized bytes (CUDA's
int8 MMQ path). Meanwhile the memory math is brutal for the alternative: a
Q4_0 block is 18 bytes for 32 values (2-byte fp16 scale + 16 bytes of
nibbles, `block.rs:53-56`), which is ≈ 0.56 bytes per value. A
0.5-billion-parameter checkpoint is then ≈ 0.28 GB of file bytes; dequantized
to f32 it would be 2 GB — 7× more, copied at load time, for zero benefit.

**Why does cloning a `Tensor` not copy bytes?** Because `Cow::Borrowed`
clones as pointer + length. That is what makes it affordable for the graph
path to re-register all weights on every graph (re)build (`t.clone()` at
`qwen2/graph.rs:348`) — the clone duplicates a small struct and a name
string, not gigabytes. The one caveat — registries still guard against
re-registration when a caller hands them *owned* bytes — has a measured war
story attached, told with excerpt 10.

**Why do GPU backends get their own representation?** Because "the weight"
means something different per backend: on CPU it *is* the mmap bytes; on
Metal it is a byte range inside a shared-memory buffer; on CUDA it is a
pointer into device memory. The registry abstracts exactly that, and §2.4
walks each one.

### 2.4 What "the weight is on the GPU" means, per backend

The phrase "weights on GPU" hides three quite different mechanisms. Getting
them straight explains everything the loader does.

**CPU — registration is free.** `CpuBackend` keeps
`HashMap<String, Tensor>`. Registering inserts the tensor struct. The bytes
were already in the process (they are the mmap pages, faulted in on first
touch), so the "registration" moves no data at all. On CPU, "the weight is
registered" means only "the name resolves to a byte range".

**Metal — the GPU reads the same physical pages.** On Apple Silicon, CPU and
GPU share one physical memory ("unified memory"). Metal exposes buffers that
both sides can address (`StorageModeShared`). The trick is that minfer does
not copy each weight into such a buffer. Before any weight is registered, the
loader hands each mmap'd file part to `MpsState::register_part`, which wraps
the *whole mmap* in one Metal buffer via `newBufferWithBytesNoCopy` — "no
copy" is the API's name and its contract. Each individual weight is then
registered as `(buffer, byte offset)` into that one buffer. The GPU reads the
file's pages directly; there is no GPU-side allocation and no memcpy, ever.
The one cost is a first-touch one: the very first GPU access to file-backed
pages pays ~44 ms of page/TLB setup, which the loader deliberately triggers
once at load time, outside the timed inference window (`metal.rs:2350-2355`).

**CUDA — one upload, resident forever.** NVIDIA GPUs have *discrete* memory
(device memory, VRAM) that the CPU cannot address; bytes must be copied
across the PCIe bus ("H2D", host-to-device). `CudaState::register_weight`
does `cudaMalloc` for the tensor's size, one `cudaMemcpy` H2D, and stores
`(device pointer, size)` under the name. That copy happens exactly once, at
load. From then on the weight is *resident*: every decode step reads it from
device memory at GPU bandwidth instead of re-uploading. This is the Phase 7
thesis in one sentence — a graph backend is only fast if the weights are
already addressable on the executing device, so registration is a load-time
job, not a per-step one. For scale: the 7B Q4_K_M model is ~4.4 GB of
weights; the alternative (per-step host staging) is exactly what the old
imperative path did for activations, and the CUDA campaign measured such
host round-trips at "~6 PCIe round trips × 24 layers ≈ 144 DMA operations
per decode step, 2–7 ms" (`docs/CUDA_OPTIMIZATION.md`). Resident weights
delete that entire class of cost.

### 2.5 The gate: GPU participation is all-or-nothing

Per §2.4, a backend can only execute an op if the op's weight lives where the
op runs. The graph's backend assignment *is* per-op (doc 06), but the weights
constrain it globally, so before building anything the graph path asks: "is
every weight this model will use registered — and kernel-supported — on this
backend?" Two functions do this:

- `Qwen2Graph::weights_on_gpu` (Metal): every weight name must be present in
  `MpsState`'s registry (`models/qwen2/graph.rs:630-671`).
- `Qwen2Graph::weights_on_cuda` (CUDA): every weight must be registered *and*
  of a type a kernel exists for — e.g. the embedding gather supports every
  registered type except Q4_1 (`models/qwen2/graph.rs:689-750`).

The result — `metal_on || cuda_on` — is stored in `CParams.gpu`, which is
part of the *reuse identity*: the fingerprint that decides whether a cached
graph can be reused (doc 13). Flip any env toggle or unplug the eGPU and the
next forward rebuilds the graph rather than executing stale assignments.

Why all-or-nothing rather than "put what fits on the GPU, layer by layer"?
The old imperative engine *had* a per-layer fallback, and it is preserved in
`docs/ARCHITECTURE.md` Appendix A.3 as a cautionary diagram: the moment one
layer failed its GPU check, the hidden state had to cross back to host
memory, the KV cache had to be synced to CPU, and the rest of the layers ran
on CPU — per token. In the graph path that cost is even sharper: KV regions
live on the backend that executes attention (by construction), so one
CPU-resident layer would force the whole layer's KV traffic across the bus
every step. The one-buffer-at-a-split-boundary design (doc 08) exists
precisely so cross-backend traffic happens a handful of times per forward,
not per op. All-or-nothing is how the design keeps that promise: either the
backend can host everything the graph reads, or it does not participate at
all.

### 2.6 Two architectures, one interface

Qwen2 and Qwen3 differ in exactly two load-time-visible ways, and both exist
to keep the *graph builder* simple.

First, the head dimension. In Qwen2, the per-head size is derivable:
`n_embd_head = n_embd / n_head`. Qwen3 broke that identity — the 0.6B model
has `n_embd / n_head = 64` but its keys are 128-wide — so the loader reads
`qwen3.attention.key_length` explicitly and asserts the K weight's actual
output width agrees (`qwen3/loader.rs:311-326`). Trust the bytes, not the
derived formula.

Second, per-head Q/K RMSNorm: Qwen3 normalizes each head's query and key
vectors before RoPE, with small learned gain vectors (`q_norm`, `k_norm`).
The loader stores them; the graph builder has a dedicated `qk_norm` op
(`graph/builder.rs:99-119`) that consumes them. The loader's job is
recognizing that these tensors exist and must not be lost — the `minfer
info` listing truncates names, but they are in the file. Everything else —
the loader shape, the registration calls, the fused-QKV and fused-FFN concat
weights — is deliberately mirrored between the two loaders, so a new
architecture is a copy-and-edit job (§5).

## 3. Implementation

### 3.1 Data in / data out

**In:** the `GgufModel` from doc 02. Concretely, per part: `ctx.kv` (metadata
key/values), `ctx.info` (tensor index; each entry has `name`, `type_`, `ne[4]`
shape, and `offset` — offset from the start of the part's data section), and
`part.data: &'static [u8]` (the mmap'd bytes). A weight's file position is
`ctx.offset + ti.offset`, where `ctx.offset` is where the data section starts
in the file (`gguf.rs:603-619`).

**Out:** four things.

1. `Box<dyn ModelDef>` — the polymorphic model object (`Qwen2Model` /
   `Qwen3Model`): `HParams` + `tok_embd` + `output_norm` + `output` +
   optional `output_b` + one `LayerWeights` per layer.
2. Populated backend registries: CPU always; Metal's `(buffer, offset)`
   entries when MPS initialized; CUDA's device copies when a device exists.
3. The KV element-type decision (f16 vs f32), set once from the model's
   dimensions before any forward runs.
4. The legacy `KVCache` in `main.rs` — allocated, then ignored by the graph
   path (§3.2, excerpt 1).

**Shapes to internalize now** (they recur in every later doc): a GGUF weight
matrix is stored with shape metadata `[in, out]` (ne[0] = input dim,
fastest-varying) while memory is row-major `[out][in]` — so `wq` of a 0.5B
model is 896×896 and `wk` is 128×896 *as bytes* even though its logical
projection is 896 → 128. The embedding table `token_embd.weight` is
`[n_vocab, n_embd]`: one row per vocabulary entry, each row the vector that
token id looks up to. Norm weights (`attn_norm`, `ffn_norm`, `output_norm`)
are 1-D f32 gain vectors of length `n_embd` (a **gain** is just a learned
per-feature multiplier applied after normalizing); biases (`bq`, `bk`, `bv`,
`attn_output` has none, `output.bias` optional) are 1-D f32 too. Quantized
matmul weights are one of Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 (32-value blocks) or
Q4_K/Q5_K/Q6_K (256-value super-blocks) — quantization stores values in
fewer bits, grouped into blocks that share a scale factor.

### 3.2 Key code

Excerpt 1 — the startup order in `main.rs`: GPU init, dispatch, legacy KV
cache. (`src/main.rs:637-657`)

```rust
    // === GPU backends ===
    #[cfg(target_os = "macos")]
    metal::MpsState::init();
    #[cfg(feature = "cuda")]
    cuda::CudaState::init_with_gpu(gpu);
    // On CPU/Metal builds `--gpu` is a no-op: it is parsed but unused.
    #[cfg(not(feature = "cuda"))]
    let _ = gpu;

    // === Load model (dispatches on general.architecture) ===
    let model = models::load_model(&gguf_model).expect("load model");
    ...
    // === KV Cache ===
    let n_kv_embd = model.n_kv_embd();
    let n_layer = model.n_layer();
    let mut kv_cache = cache::KVCache::new(n_layer, n_kv_embd, params.n_ctx);
```

Annotations: `MpsState::init()` is a `OnceLock` singleton init — inside, it
honors `MINFER_DISABLE_MPS` by returning `None`, so "disabled" and "no
device" are the same state downstream (`metal.rs:2031-2035`). CUDA likewise
honors `MINFER_DISABLE_CUDA` and takes the `--gpu N` index here. The
`load_model` call is where this entire doc's work happens — note `.expect`:
an unsupported architecture is fatal, by design (§2.2). And the final three
lines allocate the legacy KV cache *before* the tokenizer is even loaded —
its only remaining job is to satisfy the `ModelDef::forward` signature's
`&mut KVCache` parameter, which the graph path ignores. Nothing reads it; it
is kept until that trait signature is refactored away (`cache.rs:1-7`). It
is not free, though: at 4096 context, 128-wide KV and 24 layers it is
2 × 24 × 4096 × 128 × 4 B ≈ 100 MB of zeroed memory — a good illustration of
why vestigial plumbing should eventually die.

Excerpt 2 — the dispatch itself. (`src/models/mod.rs:95-112`)

```rust
pub fn load_model(model: &GgufModel) -> Option<Box<dyn ModelDef>> {
    let ctx = &model.parts[0].ctx;
    let arch = ctx.get_key_val_str("general.architecture")?;
    match arch.as_str() {
        "qwen2" => {
            let m = qwen2::loader::load(model)?;
            Some(Box::new(m))
        }
        "qwen3" => {
            let m = qwen3::loader::load(model)?;
            Some(Box::new(m))
        }
        other => {
            eprintln!("Unsupported architecture: '{}'", other);
            None
        }
    }
}
```

Annotations: part 0 is authoritative for metadata even in a multi-part split
(doc 02 merged the tensor index across parts; metadata comes from the first).
`?` on the string lookup means a file *without* the key is "no model", not a
panic — the caller reports it. The error branch prints the offending string,
which is what makes a mistyped or future architecture diagnosable in one
glance.

Excerpt 3 — the interface everything downstream codes against.
(`src/models/mod.rs:22-45`, `:77-85` — abridged)

```rust
pub trait ModelDef: Send + Sync {
    fn forward(&self, tokens: &[u32], positions: &[usize],
               kv: &mut KVCache, n_out: usize, n_ctx: usize) -> Vec<f32>;
    /// Downcast helper for the graph path's weight registration.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Build the declarative compute graph for one forward step (Phase 5).
    /// Topology is a deterministic function of `params` (reuse invariant).
    fn build_graph(&self, _params: &GraphParams) -> ComputeGraph { ... }
    /// Graph-based forward with a caller-provided cache and explicit context
    /// size (server / multi-slot path).
    fn forward_graph_cached(&self, tokens: &[u32], positions: &[usize],
                            n_out: usize, n_ctx: usize, cache: &mut GraphCache)
                            -> Vec<f32> { ... }
    fn format_chat(&self, messages: &[(String, String)]) -> String;
    fn special_tokens(&self) -> SpecialTokens;
    fn n_layer(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn n_embd_head(&self) -> usize;
    fn n_kv_embd(&self) -> usize;
    fn n_vocab(&self) -> usize;
    fn rope_style(&self) -> RopeStyle;
}
```

This trait looks wide for an interface with two implementations, and that is
the point: each method exists because a *downstream stage* needs it and must
not know which architecture it is talking to.

| Method | Who consumes it, and for what |
|---|---|
| `forward`, `forward_graph_cached` | the CLI loop (doc 09) and the server's per-slot path — both just "run a forward"; the default `forward_graph` routes to the graph |
| `build_graph` | doc 05: the pure-IR graph builder; topology is a function of `GraphParams` only |
| `n_layer` | loader-loop sizing here, the graph builder's per-layer loop (doc 05), the legacy KV cache above |
| `n_head_kv`, `n_embd_head`, `n_kv_embd` | GQA (grouped-query attention: fewer K/V heads than query heads) head mapping and strides (doc 11), and the KV region width `n_kv_embd × n_ctx` (doc 07) |
| `n_vocab` | logits width — the sampler's input size (doc 12) |
| `special_tokens` | the sampler's stop condition: `main.rs` fetches `eos`/`im_end` ids once and checks every sampled token against them (`main.rs:839,900,1020`) |
| `format_chat` | doc 04's ChatML fallback when the GGUF has no renderable template |
| `rope_style` | doc 11: RoPE (rotary positional encoding) has two layout styles — Qwen's non-interleaved vs Llama's interleaved — and the vec-op must be told which |
| `as_any` | lets graph code downcast to the concrete model when it needs specifics |
| `Send + Sync` | the HTTP server shares the model across threads (`Arc<dyn ModelDef>`) |

Excerpt 4 — hyperparameter parsing with the dual metadata prefix.
(`src/models/qwen2/loader.rs:117-156`, abridged)

```rust
    // Try qwen2 prefix first, fall back to llama/generic
    let n_embd = get_i64(ctx, "qwen2.embedding_length")
        .or_else(|| get_i64(ctx, "llama.embedding_length"))?;
    let n_head = get_i64(ctx, "qwen2.attention.head_count")
        .or_else(|| get_i64(ctx, "llama.attention.head_count"))?;
    let n_head_kv = get_i64(ctx, "qwen2.attention.head_count_kv")
        .or_else(|| get_i64(ctx, "llama.attention.head_count_kv"))
        .unwrap_or(n_head);                       // no GQA ⇒ KV heads = Q heads
    let n_layer =
        get_i64(ctx, "qwen2.block_count").or_else(|| get_i64(ctx, "llama.block_count"))?;
    ...
        f_norm_rms_eps: get_f32(ctx, "qwen2.attention.layer_norm_rms_epsilon")
            .or_else(|| get_f32(ctx, "llama.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6),
        rope_freq_base: /* ... llama.* fallback ... */ .unwrap_or(10000.0),
        rope_style: RopeStyle::NonInterleaved,
        n_kv_embd: n_head_kv * (n_embd / n_head), // default, updated from K weight below
```

Annotations: every dimension is a fallback chain — the architecture's own
prefix first, then the LLaMA-family prefix that several fine-tunes use. The
`.unwrap_or` defaults are also data: `n_head_kv` defaulting to `n_head`
means "no grouped-query attention"; `rms_eps = 1e-6` and `freq_base =
10000.0` are the values llama.cpp would use. `n_vocab` is not read from a
metadata key at all but counted from the `tokenizer.ggml.tokens` array — the
tokenizer data is the ground truth (it arrives next stage). And `n_kv_embd`
starts as the naive product, purely so the struct is initialized; the loader
immediately overwrites it from the K weight's real shape.

Excerpt 5 — a weight tensor is born as a borrowed slice, then handed to the
GPU registries. (`src/models/qwen2/loader.rs:176-220`, abridged; the CUDA
branch at :221-285 is discussed in the annotations)

```rust
fn load_tensor(ctx: &GgufContext, raw: &'static [u8], ti: &GgufTensorInfo) -> Tensor {
    let ttype = TensorType::from_ggml_type(ti.type_);
    ...
    let off = ctx.offset + ti.offset as usize;
    let ts = ti.type_.type_size();      // bytes per block
    let bs = ti.type_.blck_size() as usize; // values per block
    let n = (shape[0] * shape[1] * shape[2] * shape[3]) as usize;
    let nbytes = (n / bs) * ts;
    // Borrow the tensor bytes straight from the mmap'd part file (zero-copy —
    // the file pages are shared with the CPU and GPU instead of a per-tensor copy).
    let src = &raw[off..off + nbytes];
    ...
    let mut tensor = Tensor::from_data_borrowed_with_strides(ttype, &shape, &strides, src);

    // Register weight tensors with GPU backends.
    #[cfg(target_os = "macos")]
    if let Some(mps) = crate::metal::MpsState::get() {
        if matches!(ttype,
            TensorType::Q4_0 | TensorType::Q4_1 | TensorType::Q4_K
          | TensorType::Q5_0 | TensorType::Q5_1 | TensorType::Q5_K
          | TensorType::Q6_K | TensorType::Q8_0)
        {
            mps.register_weight(&ti.name, tensor.data());
        } else if ttype == TensorType::F32 {
            mps.register_weight(&ti.name, tensor.data());
        }
    }
    // #[cfg(feature = "cuda")] branch: same shape, more work — see below.
    tensor
}
```

Annotations: byte size comes from the GGML type's block size, not from a
`TensorType` guess — the comment notes this is "always correct regardless of
TensorType mapping". `src` is a sub-slice of the mmap: constructing the
tensor did one range check and zero copies, and the registration is woven
into the same walk rather than done as a second pass over the model. On
Metal, essentially everything quantized plus f32 gets registered — the
shader kernels handle the block formats natively. The CUDA branch is pickier
and does more work at registration: Q6_K weights are repacked into padded
224-byte slots (raw blocks are 210 bytes, which forces byte-granular GPU
loads — padding restores 16-byte-aligned vector loads), a f32-pair plane may
be precomputed for the Q4_K kernel, and unsupported types clear a fast
matmul-mode flag because a mode that assumed certain weight layouts would
otherwise read garbage. The point for this doc: **registration is where
per-backend representation is decided** — bytes for CPU, (buffer, offset)
for Metal, device allocation (+optional repack) for CUDA.

Excerpt 6 — two load-time side decisions: the KV element type, and the true
KV width. (`src/models/qwen2/loader.rs:310-317` and `:495-498`; Qwen3's
guarded version at `src/models/qwen3/loader.rs:311-331`)

```rust
    // KV cache element type (GPU path): auto-select f16 for the 7B class (KV
    // bandwidth-bound decode) unless MINFER_CACHE_TYPE overrides. Must run
    // before the first forward (kv_cache_is_f16 reads the OnceLock).
    #[cfg(target_os = "macos")]
    crate::metal::set_kv_cache_type(hparams.n_layer as usize, hparams.n_kv_embd as usize);
    // 8b: CUDA side shares the same policy and MINFER_CACHE_TYPE override.
    #[cfg(feature = "cuda")]
    crate::cuda::set_kv_cache_type(hparams.n_layer as usize, hparams.n_kv_embd as usize);
    ... // (later, after the per-layer weights are loaded:)

    // Override n_kv_embd from layer 0 K weight's actual output dimension
    if let Some((_, ti)) = tensor_map.get(&tn::attn_k(0)) {
        hparams.n_kv_embd = ti.ne[1];
    }
```

```rust
    // qwen3/loader.rs — same override, resolved BEFORE the KV type pick, plus
    // an assert that is only sound for Qwen3 (Qwen2's `n_kv_embd` may
    // legitimately differ from `n_head_kv × n_embd_head`, so it cannot assert):
    if let Some((_, ti)) = tensor_map.get(&tn::attn_k(0)) {
        hparams.n_kv_embd = ti.ne[1];
        // sanity: kv dim must equal n_head_kv * n_embd_head (catches a wrong
        // key_length fallback before it silently corrupts attention)
        assert_eq!(
            hparams.n_kv_embd, hparams.n_head_kv * hparams.n_embd_head, ...)
    }
```

Annotations: the K projection is a real matrix sitting in the file — on the
0.5B it is `[896 → 128]` — so its output width *is* the KV width, whatever
the head-count metadata might imply; the loader trusts it over any derived
value. The Qwen3 loader reads it *before* the KV type pick (its comment says
why: the f16 auto-select multiplies `n_layers × n_kv_embd`), and its assert
turns a wrong `key_length` fallback into a load-time crash instead of
silently corrupting attention. The policy `set_kv_cache_type` implements
(`metal.rs:132-153`): if `MINFER_CACHE_TYPE` says `f16`/`f32`, obey;
otherwise auto-select — f16 (half precision: 2 bytes per value instead of 4)
when `n_layers × n_kv_embd ≥ 8192`, i.e. models big enough that decode is
KV-bandwidth-bound (measured −1 ms/token on the 7B at 2K context), f32 for
small models where f16 measured ~3% *slower*.

Excerpt 7 — the merged tensor index and name lookup.
(`src/models/qwen2/loader.rs:329-343`)

```rust
    // Merged tensor index across all split parts (llama.cpp weights_map): each
    // tensor lives in the part that lists it, read from that part's own data.
    let mut tensor_map =
        std::collections::HashMap::<String, (usize, &GgufTensorInfo)>::new();
    for (pi, part) in model.parts.iter().enumerate() {
        for ti in &part.ctx.info {
            tensor_map.insert(ti.name.clone(), (pi, ti));
        }
    }
    let load_one = |n: &str| -> Option<Tensor> {
        tensor_map.get(n).map(|(pi, ti)| {
            let part = &model.parts[*pi];
            load_tensor(&part.ctx, &part.data, ti)
        })
    };
```

The loader then reads weights by *canonical name*: `token_embd.weight`,
`output_norm.weight`, `output.weight`, `blk.{i}.attn_norm.weight`,
`blk.{i}.attn_q.weight`, … — the names come from a small `tensor_names`
module (`models/qwen2/mod.rs:124-166`), so a naming convention change is a
one-file edit. Two notable lookups: `output` falls back to the embedding
table when absent (`load_one(tn::OUTPUT).unwrap_or_else(|| tok_embd.clone())`
— *weight tying*: small models reuse the embedding table as the final
projection instead of shipping a second matrix), and every per-layer tensor
is `Option` because Qwen3 has no biases while Qwen2.5-7B does.

Excerpt 8 — Metal's zero-copy registry. (`src/metal.rs`, annotated
condensation of `register_part` :2320-2372 and `register_weight` :2374-2424)

```rust
    pub fn register_part(&self, data: &'static [u8]) {
        let page = 16384; // macOS page size on Apple Silicon
        let base = data.as_ptr() as usize;
        debug_assert!(base % page == 0, "mmap'd GGUF part not page-aligned");
        let buf = unsafe {
            self.inner.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr, data.len(), MTLResourceOptions::StorageModeShared, None)
                .unwrap()
        };
        self.inner.mmap_parts.lock().unwrap().push((base, data.len(), buf.clone()));
        // GPU-side warm-up (#39): the FIRST GPU access to file-backed pages
        // costs ~44 ms of one-time page/TLB setup → do it here, at load.
    }

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        let force_copy = std::env::var("MINFER_WEIGHT_COPY").map_or(false, |v| v == "1");
        // Zero-copy path: the weight is a slice of a registered mmap'd part
        // → (part buffer, offset). The GPU reads the mapped file pages
        // directly — no CPU→GPU memcpy, no GPU-side allocation.
        let entry = if !force_copy {
            parts.iter().find(|(base, len, _)| ptr >= *base && ptr + data.len() <= base + len)
                .map(|(base, _, buf)| (buf.clone(), (ptr - base) as u64))
        } else { None };
        let (buf, off) = match entry {
            Some(e) => e,
            None => { /* copy into a fresh shared buffer (offset 0) */ }
        };
        self.inner.weights.lock().unwrap().insert(name.to_string(), (buf, off));
    }
```

Annotations: the loader calls `register_part` for every mmap'd part *before*
registering any weight (`qwen2/loader.rs:319-327`) — the ordering is load-
bearing, because `register_weight` locates its zero-copy entry by finding the
part that contains the pointer. `StorageModeShared` on Apple Silicon means
one physical allocation both CPU and GPU address; "zero-copy" is literal.
The copy fallback exists for the two cases where bytes are *not* file pages:
the fused `attn_qkv`/`ffn_gu` concat weights (built in RAM at load,
`qwen2/loader.rs:391-426,446-491`) and `MINFER_WEIGHT_COPY=1`, an A/B switch
that makes the cost of the zero-copy path measurable.

Excerpt 9 — CUDA's upload-once registry. (`src/cuda.rs:1507-1562`, abridged)

```rust
    pub fn register_weight(&self, name: &str, data: &[u8]) {
        if data.is_empty() { return; }
        {
            let w = self.weights.lock().unwrap();
            if let Some((_, size)) = w.get(name) {
                if *size == data.len() {
                    // Device weights are immutable: same name + size ⇒ the
                    // same GGUF tensor ... Reuse the existing device copy
                    // instead of leaking one buffer per load.
                    return;
                }
                // Different size ...: replace the entry. The stale buffer is
                // deliberately NOT freed — a live captured graph may still
                // reference it; ...
            }
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
        ...
        let err = unsafe {
            cudaMemcpy(ptr, data.as_ptr() as *const c_void, data.len(),
                       CUDA_MEMCPY_HOST_TO_DEVICE)
        };
        ...
        self.weights.lock().unwrap().insert(name.to_string(), (CudaPtr(ptr), data.len()));
    }
```

Annotations: three details repay attention. (1) The dedup check makes
re-registration a no-op — graph rebuilds and unit tests that reload a model
must not each leak another full-weight-set upload (~4.4 GB on the 7B).
(2) The refusal to free stale
buffers is deliberate, not sloppy: a captured CUDA Graph (doc 15) holds raw
device pointers; freeing under it would be use-after-free. (3) A plain
registration clears any stale "padded" flag for that name, so a second
model reusing a tensor name with a non-Q6_K type cannot be dispatched
through the padded-224 kernel on a raw-210 buffer (a Phase 8 review
finding).

Excerpt 10 — CPU registration: the cheapest one. (`src/graph/cpu_backend.rs:34-47`)

```rust
    /// Register a weight tensor by name (Phase 6 wires this from the model).
    pub fn register_weight(&mut self, name: &str, t: Tensor) {
        // Skip re-registration of an already-known weight: Tensor carries its
        // bytes as Cow::Owned, so the `t.clone()` at the model call sites
        // deep-copies the full weight set (~4.4 GB on 7B) on EVERY graph
        // (re)build — measured as a ~635 ms pure-CPU stall at the
        // prefill→decode graph switch (no CUDA calls, no kernels). Model
        // weights are immutable after load (weights_version guards any future
        // change), so a same-name registration always carries the same data.
        if self.weights.contains_key(name) {
            return;
        }
        self.weights.insert(name.to_string(), t);
    }
```

The comment is the whole lesson: with borrowed bytes, even the *unguarded*
insert is cheap; the guard exists because one call path produced *owned*
clones. The graph allocator simply forwards to it
(`graph/alloc.rs:134-137`: `self.cpu.register_weight(name, t)`), which is
why the allocator's registration costs nothing on CPU.

Excerpt 11 — the graph's own registration pass, at first build.
(`src/models/qwen2/graph.rs:338-372`, abridged)

```rust
    /// Register every weight the graph references on the allocator's backend.
    pub(crate) fn register_graph_weights(model: &Qwen2Model, alloc: &mut GraphAllocator) {
        for t in [&model.tok_embd, &model.output_norm, &model.output, &model.output_b] {
            if let Some(t) = t {
                let name = t.name.clone();
                alloc.register_weight(&name, t.clone());
            }
        }
        for l in &model.layers {
            for t in [&l.attn_norm, &l.wq, &l.bq, &l.wk, &l.bk, &l.wv, &l.bv,
                      &l.wo, &l.ffn_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down] {
                if let Some(t) = t {
                    let name = t.name.clone();
                    alloc.register_weight(&name, t.clone());
                }
            }
        }
    }
```

Annotations: this runs inside `forward_cached` on graph build only (guarded
by the reuse check), and the `t.clone()` is the cheap borrowed-clone of
§2.3. Twelve entries per layer is the Qwen2 inventory: 7 matmul weights
(wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down), 2 norm gains, and 3–4 biases.
The list is deliberately spelled out — not derived by reflection — so the
compiler catches field renames in *both* the registration and the gate
(excerpt 12), which must enumerate the same weights.

Excerpt 12 — the Metal participation gate. (`src/models/qwen2/graph.rs:628-671`,
names list abridged)

```rust
    /// Every weight the graph reads must be GPU-registered for the Metal path.
    #[cfg(target_os = "macos")]
    fn weights_on_gpu(model: &Qwen2Model) -> bool {
        let names: Vec<String> = {
            let mut v = Vec::new();
            for t in [&model.tok_embd, &model.output_norm, &model.output, &model.output_b] {
                if let Some(t) = t { v.push(t.name.clone()); }
            }
            for l in &model.layers {
                for t in [&l.attn_norm, &l.wq, /* ... all 12 per layer ... */ &l.ffn_down] {
                    if let Some(t) = t { v.push(t.name.clone()); }
                }
            }
            v
        };
        let Some(mps) = crate::metal::MpsState::get() else { return false; };
        names.iter().all(|n| mps.has_weight(n))
    }
```

And where the verdict lands (`src/models/qwen2/graph.rs:425-451,462-470`):

```rust
        #[cfg(target_os = "macos")]
        let metal_on =
            crate::graph::metal_backend::metal_available() && Self::weights_on_gpu(model);
        ...
        #[cfg(feature = "cuda")]
        let cuda_on = crate::cuda::CudaState::get().is_some() && Self::weights_on_cuda(model);
        ...
        let params = GraphParams {
            n_tokens: nt, n_seqs: 1, n_out,
            gtype: if nt == 1 { GraphType::Decode } else { GraphType::Prefill },
            cparams: CParams {
                n_ctx, n_batch: nt, flash_attn: false,
                gpu: metal_on || cuda_on,          // ← participation recorded
                fuse_qkv: nt == 1 && (metal_on || cuda_on) && !env("MINFER_NO_FUSE_QKV"),
                fuse_ffn: nt == 1 && (metal_on || cuda_on) && !env("MINFER_NO_FUSE_FFN"),
            },
            weights_version: 1,
        };
        if !cache.try_reuse(&params) { /* build → register → assign → fuse → alloc */ }
```

Annotations: `gpu` is a single flag in the reuse identity, so toggling the
environment forces a rebuild (doc 13). The CUDA gate (`weights_on_cuda`,
:689-750) is stricter than Metal's: each matmul weight must be registered
*and* match a kernel type (`has_weight_of_size` compares the byte length so
padded Q6_K registrations still match by raw size), and the embedding is
checked separately because its gather kernel supports one fewer type
(Q4_1). On failure it names the first offending tensor instead of returning
a bare `false` — the difference between a debugging session and a support
ticket.

### 3.3 Design choices (why this shape and not another)

**Dispatch on a string, once, before any side effect.** Covered in §2.2;
the one-line summary: deterministic, one honest error message, and compatible
with the fact that loading mutates GPU state.

**A wide trait instead of a narrow one.** The tempting alternative is a
minimal trait (`forward` + a getter or two) with the rest downcast via
`as_any`. That pushes every consumer into arch-specific code. The chosen
shape inverts it: the trait declares everything the *pipeline* needs
(dims for graph shapes, `build_graph` for doc 05, `special_tokens` for doc
12, `rope_style` for doc 11, `n_vocab` for the sampler width), each
architecture implements them once, and downstream code stays
architecture-blind. The cost is some `#[allow(dead_code)]` ceremony on
methods only reached through `Box<dyn ModelDef>` — noted in the trait's own
comment (`models/mod.rs:17-21`) — which is a fair price for the type safety.

**The IR references weights by name, not by pointer.** Graph nodes carry
`NodeMeta::MatMul { weight_name, weight_ttype, in_dim, out_dim }`
(`graph/builder.rs:127-145`); backends resolve the name through their
registry at execute time. The alternatives: embedding raw byte pointers in
the IR (couples the pure graph to mmap lifetimes and makes the CUDA
representation impossible), or embedding `Tensor`s (makes graph comparison —
the reuse identity — expensive). Names are cheap, comparable, and each
backend maps them to its own representation. The same choice is what makes
fusion possible: the fused `attn_qkv` weight is just *another name*, so a
fused node differs from three unfused ones only in metadata.

**Zero-copy on Metal, upload-once on CUDA, free on CPU.** Three honest
answers to "where can this hardware read bytes from?" — not three
implementations of one idea. What they share is the *invariant*: after load,
no backend ever moves weight bytes again during inference.

**No dequantization at load.** §2.3's arithmetic: 0.56 B/value vs 4 B/value,
and both CPU and GPU kernels are built to consume the packed forms directly.
The exceptions prove the rule — every repack that *does* happen (CUDA's
padded Q6_K slots, the Q4_K descriptor plane, the optional f16 dequant
cache, the fused concat weights) exists because a specific kernel measured
faster on a different layout, is gated on model size or env flag, and is
documented with its byte math at the registration site.

**All-or-nothing GPU participation, recorded in the reuse identity.** §2.5.
The alternative (per-layer fallback) is the old engine's design, and its
cost — 144 DMA operations per decode step in the worst case — is on record
in `docs/CUDA_OPTIMIZATION.md`.

**The legacy `KVCache` lives on, ignored.** Deleting it means changing the
`ModelDef::forward` signature and every test that constructs a `KVCache`;
the graph path's real KV lives in the allocator's persistent per-layer
regions (docs 07–08). Keeping a vestigial 100 MB allocation is the cheaper
mess until that signature refactor happens — and it is annotated as such at
both ends (`cache.rs:1-7`, `qwen2/graph.rs:388` where `_kv` is underscored
and ignored).

### 3.4 Pitfalls & invariants

- **Registration order on Metal**: `register_part` for every mmap part
  *before* any `register_weight`. The zero-copy lookup finds weights by
  pointer containment in a registered part; weights registered first would
  silently take the copy path. Page alignment of the mmap base is a
  `debug_assert`, not a hope (`metal.rs:2332`).
- **CUDA init must complete before the first registration.** The loaders
  call `CudaState::init()` up front and hold a model-load guard for the whole
  load (`qwen2/loader.rs:295-306`). The recorded failure mode: lazy init
  mid-load flips the backend gate between tensors, producing a graph that
  mixes CPU/CUDA assignment against persistent KV regions that were sized
  for one of them.
- **Weights are immutable after load.** Both CPU and CUDA registries skip or
  dedup same-name re-registration; `weights_version` in `GraphParams` is the
  escape hatch if that ever changes. The bug behind the CPU guard cost a
  measured 635 ms per prefill→decode switch.
- **CUDA device buffers are never freed on replace.** A captured graph may
  reference them; the leak is bounded by distinct (architecture, tensor)
  shapes ever loaded (`cuda.rs:1521-1526`).
- **The gate and the registration must enumerate the same weights.** Loader
  registers; `weights_on_gpu`/`weights_on_cuda` check the same field list
  spelled out twice. That duplication is intentional — a new weight field
  fails the compile in both places until acknowledged.
- **Loader registers ⊋ gate accepts (on CUDA).** Some types are registered
  for the legacy path but have no graph kernel; the gate's type check is
  what keeps those on CPU. The embedding's Q4_1 exclusion is the standing
  example (`qwen2/graph.rs:679-687`).
- **The KV element-type decision is write-once.** `set_kv_cache_type`
  initializes a `OnceLock`; it must run before the first forward, which is
  why the loaders do it mid-load — and why the Qwen3 loader resolves
  `n_kv_embd` *before* calling it.
- **`positions[i] < n_ctx` is a caller obligation.** The KV regions are
  sized `n_kv_embd × n_ctx` once; `forward_cached` asserts it loudly
  (`qwen2/graph.rs:415-420`) rather than corrupting a region.

## 4. Observe & verify

- `./target/release/minfer info <model>` — dumps the tensor table (names,
  quant types, shapes) and metadata KV, so you can see exactly the names the
  loader will look up and the `general.architecture` value dispatch matches.
- Startup log, the stage's own narration: `File: … bytes in N part(s)` (doc
  02), then `MPS: GPU acceleration enabled` or `MPS: disabled by
  MINFER_DISABLE_MPS` / `CUDA: GPU acceleration enabled`, then
  `Loaded: N layers`, `Model loaded.`, and `Vocabulary: N tokens` (the
  `n_vocab` this stage counted).
- `MINFER_DISABLE_MPS=1` — forces CPU on macOS; the log flips to
  `MPS: disabled by MINFER_DISABLE_MPS` and the graph's backend colors (next
  bullet) go all-CPU. `MINFER_DISABLE_CUDA=1` is its CUDA twin.
- `MINFER_WEIGHT_COPY=1` / `MINFER_CACHE_TYPE=f16|f32` — the first makes
  Metal copy each weight into a fresh buffer instead of wrapping the mmap
  pages (A/B the zero-copy path); the second pins the KV element type
  instead of the size-based auto-select.
- `--dump-graph out.dot` / `--dump-graph-json` (or `MINFER_TRACE=/tmp/t.json`)
  — exports the built graph with real backend assignment; nodes whose
  matmuls reference registered weights show their assigned backend, which is
  the visible outcome of this stage's gate. `minfer viz` renders the same in
  a browser.
- `MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` — skips building the
  fused concat weights at load too, so their memory cost disappears from
  your process footprint; a way to feel the difference between "raw GGUF
  bytes" and "registration-time derived copies".
- Tests: `cargo test` covers the weight registry round-trip (cpu_backend
  tests register and look up by name), and the Metal/CUDA graph suites run
  the same tiny graphs on GPU and CPU asserting bit-identical output — the
  end-to-end proof that registration made weights reachable on each backend.

## 5. Cross-references

- `docs/ARCHITECTURE.md` §2 (module map), §3 (pipeline position of this
  stage), §5 (backend layering + selection rules), §8 (the
  add-a-new-architecture checklist that mirrors this doc).
- `docs/GRAPH-REFACTOR-PLAN.md` — Phase 5/6 record: how the imperative
  forward became `build_graph` + registries, and why nodes carry names.
- `docs/CUDA-BACKEND-PLAN.md` — Phase 7 design and §2 inventory of the CUDA
  weight registry; the resident-weights thesis this doc leans on.
- `docs/CUDA_OPTIMIZATION.md` (+ `docs/cuda_optimization_steps/`) — the
  measured cost of per-step host round-trips that resident weights delete;
  Q6_K padded registration details (7e②).
- `docs/METAL_OPTIMIZATIONS.md` — the mmap-part zero-copy design and the
  #39 first-touch warm-up; KV f16 auto-select measurements (§0/§2.5).
- `docs/QWEN3-SUPPORT-PLAN.md` §2 — the decoupled head dim and per-head
  Q/K norm rationale.
- `docs/PERF-QWEN3-4B-VS-LLAMACPP.md` — why `n_ctx` (not the model's max
  context) sizes the KV regions.
- Neighbors: 02 (what the `GgufModel` container is), 04 (tokenizer +
  template — the next consumers of metadata), 05 (the graph that finally
  reads these weights), 07 (the allocator that owns registration and KV
  regions), 14/15 (the Metal and CUDA backends whose registries were filled
  here).

← [02 — GGUF load](02-gguf-load.md) · [Index](./README.md) · [04 — Tokenizer and chat template](04-tokenizer-template.md) →
