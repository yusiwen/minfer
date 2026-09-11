# 14 · The Metal backend

> **Stage**: decode loop running (docs 09–13) → **this stage: the same graph, executed on the Apple GPU instead of the CPU kernels of docs 10/11** → the CUDA backend (doc 15).
> **Code**: `src/graph/metal_backend.rs` (`MetalBackend`, the `Backend` trait implementation — `supports_op` at :265, `execute_node` at :316, `synchronize` at :1023), `src/metal.rs` (`MpsState` device layer, `MpsCommandBuffer`, `submit` at :1935), `src/metal.metal` (the Metal Shading Language kernels).

## 1. Background — where this stage sits

Docs 05 through 08 built a declarative compute graph — one node per math op of the transformer — assigned every node to a backend, and allocated buffers by liveness. Docs 09 through 11 then followed the *CPU* execution of that graph: quantized matmuls with AVX2/NEON dot kernels (doc 10), RMSNorm, RoPE, GQA attention over the KV cache (doc 11). Doc 13 showed the decode loop re-running that graph one token at a time. This document is the parallel universe: **the identical graph, node for node, executed on an Apple GPU through Metal**. Nothing about the graph changes — the builder emits the same nodes, the scheduler walks the same splits — only the thing that *runs* each node changes. Doc 14 and doc 15 (NVIDIA) are the two chapters of "the same math, a different executor".

Two words this document leans on constantly, defined before anything else:

- A **backend** is an *executor*: a component that owns a pool of buffers and knows how to run every operation of the graph on one piece of hardware. In minfer a backend is whatever implements the `Backend` trait from `src/graph/backend.rs` — the CPU backend of doc 08, the Metal backend of this doc, the CUDA backend of doc 15.
- A **kernel** is one *program that runs on the accelerator*: for Metal, a function written in Metal Shading Language (MSL — Apple's C-derived GPU language, the code in `src/metal.metal`) that thousands of GPU threads execute in parallel. "The backend dispatches ops to kernels": `MetalBackend::execute_node` is the dispatch, `kernel_rms_norm_f32` is a kernel.

So the division of labor is: `src/graph/metal_backend.rs` (~2080 lines) is the *graph-facing* half — it takes one `CNode` at a time and decides which kernel to encode. `src/metal.rs` (~3940 lines) is the *device-facing* half — it owns the Metal device object, the compiled kernels, the weight registry, and the command buffers. `src/metal.metal` (~5150 lines) is the *hardware-facing* half — the actual shader programs. One name to un-confuse early: the code calls its device layer `MpsState` ("MPS" = Metal Performance Shaders, Apple's own math library). That is a historical name — minfer uses **none of Apple's MPS library**; every kernel is handwritten, following llama.cpp's Metal kernels, because the project has zero ML-framework dependencies by design.

Why does a GPU backend exist at all, and why does macOS get one first? Apple Silicon is a **unified-memory** machine: the CPU and the GPU physically share one pool of RAM and one memory controller. A traditional discrete GPU needs every weight and every activation explicitly copied over PCIe (doc 15's pinned-staging story); on an M-series Mac the GPU can be handed a *pointer* to memory the CPU already filled. That removes the classic tax of GPU inference — the copy — and leaves the GPU's real advantage: thousands of arithmetic lanes that can chew through a quantized matmul while the CPU's handful of cores (doc 10's thread pool) stream the same bytes far more slowly. The measured stakes, from `docs/METAL_OPTIMIZATIONS.md` §0.1: Qwen2.5-0.5B decodes at ~299–331 tokens/s on Metal versus ~5.9 tokens/s on the CPU of the same machine; the 7B model runs at ~19.3 ms/token — llama.cpp parity.

What would break without this stage? Nothing crashes — the engine would simply always take the CPU path. But the *design* must answer a harder question: how do you plug a second executor into a graph that was built for the CPU without introducing hangs, silent wrong answers, or GPU faults that freeze the whole machine? That last failure mode is real: `docs/GPU_SAFETY.md` opens with an incident where a GPU hang froze an entire M4 Pro (WindowServer included, forced shutdown). So the Metal backend is built around four contracts, all verified in this doc: ops are only assigned to it when it can run them (`supports_op`), every weight it needs is registered before the graph builds (the all-weights-registered gate), its async GPU work is flushed exactly at split boundaries (`synchronize`), and every wait on the GPU is bounded (`submit`).

The walk: §2 covers the principles — the trait contract, unified memory, the f32-activations decision, the one-command-buffer-per-split rule, the dispatch matrix, and when Metal actually wins. §3 walks the code: the trait implementation, the device layer, zero-copy weight registration, the gate, the bounded submit, and one full shader. §4 shows how to watch it run; §5 points onward.

## 2. Principle — how it works and why

### 2.1 The contract: what a backend must provide

The `Backend` trait (`src/graph/backend.rs`) is the seam between the graph machinery (docs 05–08, which know nothing about GPUs) and the hardware executors. Seven methods, each with one job:

```rust
// src/graph/backend.rs:21-42 (contract core; read_host/write_host/synchronize at :59-70)
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    /// Op support by (op, dtype). `supports_fused` gates the fusion pass
    /// (Phase 4) so fused IR nodes are only produced when a kernel exists.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool;
    fn supports_fused(&self, fused: &FusedOp) -> bool;

    /// Buffer pool: allocate / release a buffer of `size` f32 elements.
    fn alloc_buffer(&mut self, size: usize) -> usize;
    fn free_buffer(&mut self, id: usize);

    /// Allocate a buffer that bypasses the recycle free list. Split-boundary
    /// staging needs this: at execute time the free list holds ids whose
    /// physical contents are still referenced by node_to_buf and get
    /// read/written later in the same execute — recycling one would clobber
    /// in-flight data. Fresh buffers enter the normal free list on
    /// free_buffer (at graph rebuild), where liveness recycling is safe.
    fn alloc_fresh(&mut self, size: usize) -> usize;
```

Read the method list as a small operating system for one accelerator: **capability queries** (`supports_op`, `supports_fused` — "can you run this op at all?") are asked at *build* time, before any execution; **memory management** (`alloc_buffer`/`free_buffer`/`alloc_fresh`) is asked by the allocator of doc 07, once per graph build; **execution** (`execute_node`) is asked by the scheduler of doc 08, once per node per forward; and **data movement** (`read_host`/`write_host` — host↔buffer access for filling inputs and reading logits) plus **`synchronize`** (flush async work) are asked at split boundaries. The CPU backend implements all of these trivially — its buffers are `Vec<f32>`s, `execute_node` just calls the function, `synchronize` is a no-op. The Metal backend implements the same signatures with an asynchronous accelerator behind them, which is where every interesting design decision in this chapter comes from.

Two contract details deserve a second look because the Metal backend leans on them hard. First, `execute_node` receives *buffer ids* into the backend's own pool, not raw pointers, and takes a `kv_pair` argument — the `(k_id, v_id)` ids of the layer's two persistent KV regions — because KV ops (store, attention) need to touch regions the allocator keeps alive across the whole session. Second, the trait doc for `execute_node` warns that the output buffer **may alias an input buffer** (liveness reuse, doc 07) — the backend must handle in-place execution. On the CPU that is automatic (same memory); on Metal it becomes a visible piece of code you will meet in §3.2 (`copy_in`).

### 2.2 Unified memory changes the copy math

On a discrete GPU the data story is: allocate device memory, copy weights in once, copy activations back and forth every step over PCIe. On Apple Silicon, *every buffer can be one allocation both sides can see*. minfer leans on this in three places:

1. **Weights are never copied at all.** The GGUF file is memory-mapped (doc 02); at load time minfer wraps the mapped bytes in a Metal buffer with `newBufferWithBytesNoCopy` (§3.2.7) and registers each weight as `(buffer, byte-offset)` into that one wrapper. The GPU reads the model directly out of the file page cache. A 7B model's ~4.4 GB of weights (doc 10's number) occupy the same physical pages for the CPU and the GPU.
2. **The activation pool is shared-memory.** `new_f32_buffer` allocates with `MTLResourceOptions::StorageModeShared` — host and GPU address the same bytes. Filling an input is a plain `memcpy` into the buffer; reading logits is a plain read out of it. `read_host`/`write_host` are therefore views, not transfers.
3. **Cross-backend copies are host round trips.** When the scheduler must move a value from a Metal split to a CPU split (doc 08), `copy_across` does `read_host` → `write_host` — two `memcpy`s, no DMA engine, no staging queue. `src/graph/alloc.rs:564` documents exactly this: *"host round trip through read_host/write_host (shared-memory GPU buffers make this a plain memcpy both ways)"*.

The consequence for the reader of docs 14 vs 15: this chapter has almost no *data movement* code, because on this hardware there is almost no data movement. The CUDA chapter is, in large part, the story of managing exactly the copies that unified memory makes unnecessary.

### 2.3 Why f32 activations on the GPU when the CPU quantizes to Q8_0

Doc 10 established the CPU's inner-loop trick: activations are quantized to Q8_0 (per 32 values, one `f16` scale + 32 int8 values) right before each matmul, so the inner loop becomes an int8×int8 dot product that the CPU has *dedicated silicon* for (`vpmaddubsw` on x86, `SDOT` on ARM). The Metal path does none of that: its shaders read the activation buffers as plain `f32`. Three reasons, in decreasing order of importance:

- **There is no int8 hardware to exploit.** MSL exposes no fast int8×int8 matrix instruction to shaders comparable to `SDOT` — Apple's integer muscle lives in the Neural Engine, which Metal Shading Language cannot reach. An int8 activation path on Metal would add a quantize kernel before and a dequantize inside the shader, and the inner loop would still be float math converted from ints. llama.cpp's Metal backend makes the same choice (minfer's `METAL_OPTIMIZATIONS.md` #4 records adopting it: *"Q4_0 → f32 activations (aligned with llama Metal, removed Q8_0 quantize) — decode +5-10 %"* — removing the quantize step made it *faster*).
- **Decode bandwidth is bound by weights, not activations.** During decode (doc 13) one token's activation row is read once per matmul, while the entire weight matrix is streamed per matmul. Arithmetic for one layer of Qwen2.5-0.5B (`attn_q`, an `[896, 896]` Q4_0 weight): weights = 28 blocks × 18 B × 896 rows ≈ **451 KB**; activations = 1 token × 896 × 4 B ≈ **3.5 KB** — the weights are 99.2 % of the traffic. Quantizing activations to 1 byte would shrink the 3.5 KB to 0.9 KB: a ~0.6 % saving on a stream that is memory-bound. The weights stay 4-bit (§3.2: the same GGUF bytes as the CPU reads); only the activation side differs.
- **The two paths are allowed to disagree.** minfer's rule (AGENTS.md, core rule 9): *"CPU quantizes activations to Q8_0, GPU reads f32 — CPU-vs-GPU logits differ by design; compare each path against its own reference."* The CPU path is verified bit-for-bit against llama.cpp (doc 10); the Metal path is verified against its own f32 reference within a small tolerance (§4's tests use `max diff < 1e-3`). Greedy token *outputs* still match in practice across backends, but the guarantee lives per path, and any test that quietly compared CPU logits to Metal logits would be testing the wrong invariant.

### 2.4 One command buffer per split — the encode/submit rhythm

A **command buffer** is Metal's unit of work submission: you *encode* into it a list of kernel launches (a compute command encoder wraps them), then *commit* it to the GPU, which executes the launches in order. Encoding is cheap CPU-side bookkeeping; the expensive part is the commit — the driver round trip. That asymmetry shapes minfer's execution model:

- `execute_node` never talks to the GPU. It **encodes** one kernel launch (or a few) into the split's command buffer and returns immediately. A whole decode forward — ~50 nodes for one layer loop — costs the CPU only the encode time.
- The command buffer lives for exactly one **split** (doc 08: a maximal run of consecutive nodes assigned to the same backend). At the split's end the scheduler calls `synchronize`, which **submits** — commits the buffer and waits for the GPU to finish.

Why per *split* and not per op, or per whole graph? Per op would pay the commit round trip dozens of times per forward for no benefit — nothing between two ops in one split needs the GPU's results. Per whole graph would be wrong the moment the graph has a CPU node: the CPU node downstream needs its input *now*, so the GPU work feeding it must be flushed at the boundary — and the split boundaries are precisely where the scheduler already stops and syncs (`scheduler.rs:177-188`, shown in §3.2). On a fully-GPU-eligible model the graph is one Metal split, so one submit per forward; a mixed graph pays one submit per Metal split. The `metal_backend.rs` module header states the rule: *"One `MpsCommandBuffer` is kept per split and submitted by `synchronize()` (called at split boundaries), so ops within a split share a single GPU submission — the plan's §15 'split shares one command buffer' rule."*

One more subtlety hides inside the encoder: within a split, kernels run back-to-back on the GPU, and **Metal does not guarantee that one kernel's writes are visible to the next kernel without an explicit barrier**. Every dispatch helper in `metal.rs` therefore ends with `enc.memoryBarrierWithScope(MTLBarrierScope::Buffers)` (`metal.rs:346-348`). This is not paranoia — `METAL_OPTIMIZATIONS.md` #28 records the bug it fixed: on 1.5B/7B prefill, the RMSNorm output buffer `bn` was reused as the input of the next op, and *"a kernel that reads a buffer written by a preceding dispatch can race with that dispatch's last threadgroups, intermittently corrupting the tail rows"* — last-2-token garbage, roughly 10-30 % wrong tokens on the first token, fixed by inserting the barrier (and a `threadgroup_barrier` in the GEMM kernels), after which 24/24 dump comparisons became deterministic.

### 2.5 The dispatch matrix: which op runs which kernel

`supports_op` (`metal_backend.rs:265-283`) is the capability table, and it is nearly a yes for everything the Qwen2/Qwen3 builders emit: elementwise ops (`Add`/`Mul`/`Silu`/`SwiGLU`), norms (`RmsNorm`, Qwen3's per-head `QkNorm`), `MatMul` in every supported quant type, `GetRows` (embedding gather and the tail-row gather), `RoPE`, `KvcacheStore`/`KvcacheLoad`, `Attn`, and the decode fusions `FusedQKV`/`FusedFFN`/`FusedQkvNorm`. It refuses `Scale`, `Softmax`, `FusedBiasRope`, and `BatchMatMul` — none of which the model builders emit in the graph path (attention is one fused node that does its own softmax; the attention scale rides in `AttnMeta`; the fusion pass only rewrites `RoPE∘Add` where the backend claims it) — and it refuses `QkvBiasRopeStore`, a CUDA-only fusion (doc 15). Refusal here is not an error: it simply makes `assign_backends` (doc 06) hand such a node to the CPU, creating a split boundary.

Within the ops Metal *does* run, the dispatch is a decision tree of kernel *families* — this is where `docs/METAL_OPTIMIZATIONS.md`'s campaign lives, so here is the map (the doc has the measurements):

- **Matmul, three tiers** (`quant_matmul_f32_on_gpu_buf`, `metal.rs:495`): a simdgroup **GEMM** kernel (64×32 output tile per threadgroup, 8 KB of threadgroup scratch) when prefill-shaped (`nt ≥ 2` and big enough), a **`_multi`** kernel (one threadgroup per 2 output rows, handles all `nt` at once) when multi-token but small, and a **single-token** kernel for decode (`nt == 1`). Every supported quant type (Q4_0 through Q6_K) has all three tiers where it matters; the q4_K decode kernel was ported to llama.cpp's layout in #27 and took 7B decode from ~51 to ~19.3 ms/token.
- **Attention, five variants** (the `Op::Attn` arm, §3.2.5): decode (`nt == 1`) picks the **flash** kernel (llama's `kernel_flash_attn_ext` port, online softmax, `hd` 64 or 128) or the two-pass **split** KV-parallel kernel, else the **classic** kernel; prefill picks the **prefill-flash** port, else the **3-pass parallel** implementation (scores → masked softmax → output, fully barrier-free), else classic. The CPU (doc 11) computes full attention rows because everything fits in cache; the GPU variants exist because a 32-lane simdgroup cannot hold a row and the fix is to tile and accumulate online — the "flash attention" trick doc 11 §2.5 previewed.
- **RMSNorm, two widths**: the 32-thread single-simdgroup kernel and the 256-thread multi-simdgroup one (default, ~2× faster per dispatch — §3.2.8 walks the shader).
- **KV cache, two element widths**: `MINFER_CACHE_TYPE=f16` stores K/V as half floats (auto-on for the 7B class, `metal.rs:147-153`), halving attention memory bandwidth (#13 measured f16 decode 1.60 → 0.95 s on a long-context case). The *region allocation* stays f32-shaped — the IR dtype is F32 — and the f16 kernels use the first half of the bytes as `half` storage; the win is bandwidth, not footprint.
- **Decode fusions** (docs 05/06 introduced them): `FusedQKV` is one concat matmul (`blk.{i}.attn_qkv`, built at load with `concat_rows`) + one `attn_bias_rope_store` kernel replacing 3 matmuls + 3 biases + 2 RoPEs + 2 stores (10 dispatches → 2); `FusedFFN` is one concat matmul (`blk.{i}.ffn_gu`) + an in-place swiglu (4 → 2), gated `nf ≤ 16384` because on the 7B the concat matmul measured slower than two separate ones; `FusedQkvNorm` (Qwen3) adds the per-head Q/K RMSNorms between the matmul and the rope+store.

### 2.6 When Metal wins — and by how much

The physics from doc 10 carries over unchanged: decode is **weight streaming** (one token ⇒ nothing to amortize; every matmul reads its whole weight matrix), prefill is **compute-bound** (many tokens share each weight byte). Metal wins both races on an M-series machine, for different reasons:

- **Decode on unified memory**: the ~4.4 GB of 7B weights stream from the same LPDDR the CPU would use, but thousands of GPU lanes keep the memory controller saturated in a way 8–12 CPU cores cannot. `METAL_OPTIMIZATIONS.md` §0.1: 7B Q4_K_M decode ~49 t/s on the graph path (steady ~19.3 ms/token after #27 — 4.4 GB / 19.3 ms ≈ **228 GB/s** of effective weight streaming), Qwen3-4B ~75.9 t/s, 0.5B ~300 t/s — versus the CPU's ~5.9 t/s on the 0.5B (doc 10's table put the CPU's 7B ceiling at ~18 tok/s from bandwidth arithmetic; the measured CPU is even lower in practice).
- **Prefill on the GPU**: doc 10's quantized int8 kernels are genuinely fast, but a GEMM-shaped workload is exactly what simdgroup matrix hardware is for; the graph path reached ~3900–4000 t/s prefill on the 0.5B (+~55 % over minfer's own pre-graph path, llama-parity on Qwen3-4B per `docs/PERF-QWEN3-4B-VS-LLAMACPP.md`), with the remaining gap on 7B prefill ~−10 % (`METAL_OPTIMIZATIONS.md` §0.1 reading).

When does Metal *not* win? The doc records the counterexamples as guardrails: the 7B FFN fusion is disabled (`nf ≤ 16384` gate) because the concat matmul measured *slower* on the 7B's decode scalar kernel; the 0.5B f16 KV cache measured ~3 % *slower* (dispatch-latency-bound, so auto-f16 is reserved for the 7B class); and the very first GPU touch of file-backed pages costs ~44 ms of one-time page/TLB setup — which is why `register_part` runs a warmup read at load (§3.2.7). "GPU faster" is a per-op, per-shape, measured claim here, never an assumption.

### 2.7 GPU participation is a build-time decision — priority and the gate

Three rules decide whether any node runs on Metal, all resolved **before the graph exists**:

1. **Priority order Metal → CUDA → CPU.** `GraphAllocator::supports` (`alloc.rs:140-157`) asks backends in that fixed order and returns the first yes. If Metal is enabled, it wins every op it supports; CUDA (when compiled in) gets the leftovers it can run; CPU mops up the rest. Doc 15's backend joins the same queue.
2. **Metal is enabled only if the whole model made it to the GPU.** The gate (`models/qwen2/graph.rs:630`, §3.2.6) requires *every* weight the graph reads — embeddings, all 28+ layer weights and biases, norms, lm_head — to be registered in the Metal registry (`has_weight`). One missing tensor ⇒ `metal_on = false` ⇒ the entire graph runs on CPU. This is the **all-weights-registered gate**: all-or-nothing participation, never a partial run where some layers are on the GPU and some on the CPU (which would be correct-but-mysterious; a split boundary per layer would also hammer the sync path).
3. **Participation is part of graph identity.** The gate's outcome is recorded in `CParams.gpu` (`graph/params.rs:27`), and `GraphParams` is the *only* thing graph reuse compares (doc 13). So "was the GPU on" is baked into the cached graph: a change (MPS unavailable, weights not registered) changes `CParams.gpu`, fails `try_reuse`, and forces a rebuild with the new assignment — rather than silently reusing a graph whose backend assignments no longer hold.

And the failure posture, which the repo treats as a hard rule: **kernel-invariant violations return `Err` from `execute_node` — never a silent CPU fallback.** If a weight is somehow not registered when a matmul executes, the arm returns `Err("weight '...' not on GPU")` and the run aborts with that message; if an op reaches the Metal arm that `supports_op` rejected, the arm returns `Err` too (`metal_backend.rs:986-994`). The reasoning (`GPU_SAFETY.md` §2.3): a silent fallback *masks* a broken invariant — the model would keep producing text while the user has no idea half the graph silently ran somewhere else, or that a shape assumption was violated. Loud failure at the offending node is debuggable; quiet degradation is not. (Frontier guards that sit *below* the graph layer — dispatch-time checks like `gpu_abort` for device-limit overruns — print the actual values and exit, for the same reason.)

## 3. Implementation

### 3.1 Data in / data out

The data story at this stage is the doc 07 allocator's, seen from the GPU side. Everything the backend touches is an id into its own pool of shared-memory `MTLBuffer`s:

| Item | Shape / layout | Where it comes from, where it goes |
|---|---|---|
| Pool buffer | `size` f32 elements = `size × 4` bytes, `StorageModeShared` | `alloc_buffer`/`alloc_fresh` → `MpsState::new_f32_buffer`; host fills inputs via `write_host`, reads outputs via `read_host` |
| Weights | raw GGUF quantized bytes, `[out][in]` row-major, registered by name as `(MetalBuffer, u64 offset)` | `register_weight` at load (zero-copy into the mmap'd part wrapper); looked up per node via `state.weight_buf(name)` |
| Activations `x` | `[nt][id]` f32, token-major — **never quantized** on the GPU path | previous node's output buffer; the matmul shader reads them directly |
| Positions | `[nt]` u32 values stored as `f32::from_bits` bit patterns (doc 07's `fill_input_i32`) | input buffer; kernels decode with `float_to_int` semantics, host helpers read them as `u32` |
| KV regions | per layer two persistent buffers `[n_kv_embd, n_ctx]` (f32 elements; f16 mode uses the first half of the bytes) | `alloc_persistent` at first touch (doc 07); written by `KvcacheStore`/fused kernels at `positions[t]`, read by attention |
| Logits | `[n_out][n_vocab]` f32, token-major | the graph's output buffer; `copy_to_cpu` at the end of the forward (doc 09's extraction) |

Two entries deserve a beginner's double-take. **Positions as f32 bit patterns**: the graph's IR has one dtype (`DType::F32`) for buffers, so integer token ids and positions are smuggled through as raw bit patterns and converted back with `.to_bits()` wherever the backend needs the integer (`positions_max` at `metal_backend.rs:203-208` reads them as `u32` directly, which is the same bytes). **The f16 KV region**: the graph *shapes* the region as `[n_kv_embd, n_ctx]` F32 no matter what, so the allocation is unchanged; when `kv_cache_is_f16()` is true the *kernels* (`kernel_store_kv_f16`, `kernel_gqa_attn_f16`, the f16 flash variants) treat that memory as `half*` — 2 bytes per element in the first half of the allocation. The saving is bandwidth during attention (half the bytes per read), which is what the decode loop is bound by.

And one negative entry, because it is the headline difference from doc 10: **there is no Q8_0 activation scratch anywhere on the GPU path.** The CPU quantizes activations per matmul call (doc 10 §3.2); the Metal path's activations stay f32 from embedding lookup to logits.

### 3.2 Key code

#### 3.2.1 The backend object: pool, command buffer, and the `'static` trick

`MetalBackend` (`metal_backend.rs:56`) is a thin shell over the device singleton plus its own pool:

```rust
// src/graph/metal_backend.rs:56-77
pub struct MetalBackend {
    state: &'static MpsState,
    /// f32-element pool: id → shared MTLBuffer (size * 4 bytes)
    pool: Vec<crate::metal::MetalBuffer>,
    free: Vec<usize>,
    /// P2/P3 capture staging: host-readable buffers written by per-split blits.
    /// Only allocated while a trace/live capture is armed.
    staging: Vec<crate::metal::MetalBuffer>,
    free_staging: Vec<usize>,
    /// Pending command buffer for the current split. Stored as a leaked box
    /// pointer (null = none) because MpsCommandBuffer is !Send/!Sync; all
    /// access happens sequentially through &self/&mut self methods on the
    /// scheduler thread, so the raw pointer is contained.
    cb_ptr: *mut crate::metal::MpsCommandBuffer<'static>,
}

// Safety: every field is either owned (pool/free), a 'static reference
// (MpsState is a Sync singleton), or the command-buffer pointer which is only
// dereferenced inside &self/&mut self methods (sequential, single-threaded).
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}
```

Three fields carry the design. `state` is a `&'static MpsState` — the device layer is a process-wide singleton (`static MPS: OnceLock<Option<MpsState>>`, `metal.rs:21`), created once by `MpsState::init()` from `main.rs:639` before the model loads; every backend instance just borrows it. `pool`/`free` implement the `Backend` buffer-pool contract: `pool[id]` is a shared `MTLBuffer`, and `free` is the recycle list the allocator's liveness analysis drives (doc 07). `cb_ptr` is the oddest: the in-flight command buffer is stored as a raw pointer to a leaked `Box`, because the objc2 command-buffer type is `!Send`/`!Sync` and Rust's borrow checker would otherwise forbid the pattern *create the buffer on first op → keep appending on later ops → submit at the split end* while also letting `cb()` hand out a `&'static mut` that does not borrow `self` (so encode methods can still touch the pool). The comment block is the safety argument: all access is sequential through `&mut self` on the single scheduler thread. This is also why `unsafe impl Send/Sync` appears — Metal objects are thread-safe in reality, but objc2 cannot prove it, so the type asserts it with the reasoning written out (`metal.rs:197-201` says the same for `MpsState`).

The command buffer is created lazily on the first op of a split:

```rust
// src/graph/metal_backend.rs:146-157
    /// The current split's command buffer (created on first op of a split).
    /// The box is leaked, so the returned reference is 'static and does not
    /// borrow `self` — callers can freely touch the pool afterwards.
    fn cb(&mut self) -> &'static mut crate::metal::MpsCommandBuffer<'static> {
        if self.cb_ptr.is_null() {
            let cb = Box::new(self.state.cmd_buffer());
            self.cb_ptr = Box::into_raw(cb);
        }
        // SAFETY: cb_ptr is null or points to a live box created here; all
        // callers hold &mut self, so no concurrent mutation.
        unsafe { &mut *self.cb_ptr }
    }
```

`MpsState::cmd_buffer()` (`metal.rs:2428-2445`) asks the device's command queue for a fresh `MTLCommandBuffer` and immediately opens a compute command encoder on it — so from the first `execute_node` of the split, every dispatch lands in one encoder. The `Drop` impl (`metal_backend.rs:245-258`) flushes any never-submitted buffer so a dropped backend cannot leak an open encoder.

#### 3.2.2 Capability table: `supports_op` and `supports_fused`

```rust
// src/graph/metal_backend.rs:265-290 (abridged to the shape; full match in tree)
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        match op {
            Op::Input => true,
            Op::Add | Op::Mul | Op::Silu | Op::RmsNorm { .. } | Op::QkNorm { .. } | Op::SwiGLU => {
                dtype == DType::F32
            }
            Op::MatMul { .. } => {
                matches!(dtype, DType::F32) // activations are f32; weight type in meta
            }
            Op::GetRows | Op::RoPE { .. } | Op::Attn { .. } => dtype == DType::F32,
            Op::KvcacheStore { .. } | Op::KvcacheLoad { .. } => dtype == DType::F32,
            Op::FusedQKV { .. } | Op::FusedQkvNorm { .. } | Op::FusedFFN => dtype == DType::F32,
            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => true,
            Op::Scale(_) | Op::Softmax { .. } | Op::FusedBiasRope | Op::BatchMatMul => false,
            // Mixed-quant decode QKV epilogue (D3-8 class 2) is CUDA-only; on
            // Metal the graph builder never emits it (qkv_epilogue_ok = false
            // without `--features cuda`), so it is never assigned here.
            Op::QkvBiasRopeStore { .. } => false,
        }
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        // swiglu_f32 and attn_bias_rope_store kernels exist (the latter is the
        // fused decode QKV store path, nt==1 only)
        matches!(fused, FusedOp::SwiGLU | FusedOp::QKVBiasRopeStore)
    }
```

Two things to notice. The `dtype` checks are almost tautological — the graph's node outputs are `DType::F32` everywhere — but the *weight* type is not part of this signature; it lives in `NodeMeta::MatMul.weight_ttype` and is checked at dispatch time against the shader families that exist (§3.2.4). And the two `false` arms are informative negatives: `QkvBiasRopeStore` is refused *with a comment explaining that the builder never emits it on macOS* — the support table and the builder's emission rules are kept in lockstep by that comment, and doc 15's backend claims the same op because CUDA *can* fuse it. `supports_fused` similarly gates the fusion pass (doc 06): `FusedBiasRope` is not claimed, so on a Metal-assigned node chain the fusion pass leaves it unfused while `SwiGLU` (whose kernel exists) gets rewritten.

#### 3.2.3 The buffer pool: recycle vs fresh

```rust
// src/graph/metal_backend.rs:292-314
    fn alloc_buffer(&mut self, size: usize) -> usize {
        if let Some(idx) = self
            .free
            .iter()
            .position(|&id| self.pool[id].length() as usize == size * 4)
        {
            return self.free.swap_remove(idx);
        }
        self.pool.push(self.state.new_f32_buffer(size));
        self.pool.len() - 1
    }

    fn free_buffer(&mut self, id: usize) {
        if !self.free.contains(&id) {
            self.free.push(id);
        }
    }

    fn alloc_fresh(&mut self, size: usize) -> usize {
        // never recycled from the free list (see Backend::alloc_fresh)
        self.pool.push(self.state.new_f32_buffer(size));
        self.pool.len() - 1
    }
```

The recycle logic is an exact-size free-list search: reuse any dead buffer whose byte length matches `size × 4`, else allocate a new shared buffer. The reason `alloc_fresh` exists is written in the trait (§2.1): split-boundary staging buffers must not be recycled mid-execute, because the free list at that moment holds ids whose contents are still in flight (encoded kernels will write them later this same execute). Fresh buffers join the normal free list only when the whole graph is freed at rebuild. Note what is *not* here: no reference counting, no GPU-side allocator — the allocator of doc 07 does the liveness math and calls these methods; the backend just owns the bytes. Underneath, `new_f32_buffer` (`metal.rs:2297-2313`) is one call: `device.newBufferWithLength_options(size * 4, StorageModeShared)` — the unified-memory allocation that makes `read_host`/`write_host` plain views.

#### 3.2.4 `execute_node`: the dispatch arms

`execute_node` (`metal_backend.rs:316-996`) is a 680-line `match &node.op`, and every arm follows the same rhythm: resolve metadata → look up weights by name → **encode** one or two kernel launches into `cb` → `Ok(())`. Nothing waits; the submit happens at the split boundary. The `MatMul` arm is the cleanest example:

```rust
// src/graph/metal_backend.rs:468-497
            Op::MatMul { .. } => {
                let meta = match &node.meta {
                    NodeMeta::MatMul(m) => m,
                    other => return Err(format!("matmul node missing MatMulMeta: {other:?}")),
                };

                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.weight_name)
                    .ok_or_else(|| format!("weight '{}' not on GPU", meta.weight_name))?;
                let nt = node.out_shape[1];
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0]),
                    0,
                    self.buf(out_buf),
                    meta.out_dim,
                    meta.in_dim,
                    nt,
                );
                if let Some(bname) = &meta.bias_name {
                    let (bb, b_off) = self
                        .state
                        .weight_buf(bname)
                        .ok_or_else(|| format!("bias '{}' not on GPU", bname))?;
                    cb.add_bias_f32(self.buf(out_buf), &bb, b_off, meta.out_dim, nt, 0);
                }
                Ok(())
            }
```

Every line here teaches the backend's shape. The weight arrives as `(buffer, byte offset)` from the registry — the offset is the zero-copy trick of §3.2.7, letting one buffer hold the whole mmap'd model with per-tensor offsets. The failure mode is the gate leaking: if a weight somehow is not registered, the arm returns `Err("weight '...' not on GPU")` — the run stops with the tensor's name, it does not fall back (§2.7). And the bias is a *second encode* into the same command buffer: matmul then `add_bias_f32`, two kernels back-to-back with the encoder's memory barrier between them, both paid only when the split submits.

The kernel side, `quant_matmul_f32_on_gpu_buf` (`metal.rs:495-970`), is the three-tier dispatch of §2.5. Its head shows the tier decision and a GPU-safety guard:

```rust
// src/metal.rs:511-541 (Q8_0 arm of the dispatch; guard + GEMM tier selection)
        if matches!(
            ttype,
            TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K
        ) && id % 256 != 0
        {
            gpu_abort(&format!(
                "matmul input dim id={id} is not 256-aligned for {ttype:?} (K-quant kernels use K/256 super-block floor)"
            ));
        }
        match ttype {
            TensorType::Q8_0 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q8_0_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q8_0_f32_multi
                        } else {
                            &self.state.pl_q8_0_f32
                        }),
                    );
```

The guard is audit finding **M1** from `GPU_SAFETY.md` §3: the K-quant shaders compute super-block counts as `K/256` (integer floor), so a non-256-aligned `id` would *silently drop the tail elements* — wrong numbers, no fault. Rather than risk it, `gpu_abort` (`metal.rs:64-69`) prints the actual misaligned value plus the hint `"(force CPU with MINFER_DISABLE_MPS=1)"` and exits. The tier rule `nt >= 2 && (od >= 2048 || nt >= 9)` sends prefill-shaped work to the simdgroup GEMM (a 64×32 output tile per threadgroup amortizes the weight load across 2048 MACs) and everything else to the `_multi`/single kernels. Each per-type arm repeats this skeleton — buffer 0 = weights (with offset), buffer 1 = activations, buffer 2 = output, bytes 3 = `[od, id, nt]` params — with per-kernel threadgroup shapes tuned in the optimization campaign (#27's q4_K port is the headline: `METAL_OPTIMIZATIONS.md` records 7B `attn_q` going 70 → 265 GB/s effective).

The **attention arm** (`metal_backend.rs:591-731`) is the largest, and its opening is the safety story:

```rust
// src/graph/metal_backend.rs:596-616 (guards; dispatch decision follows)
                // GPU safety (H1): kernel_gqa_attn strides KV by nk*hd
                if meta.nkt != meta.n_head_kv * meta.hd {
                    return Err(format!(
                        "Metal attention: nkt={} != n_head_kv*hd={} (kernel_gqa_attn strides KV by nk*hd)",
                        meta.nkt, meta.n_head_kv * meta.hd
                    ));
                }
                if meta.hd != meta.hd_kv {
                    return Err(format!(
                        "Metal attention: hd={} != hd_kv={} (kernel_gqa_attn uses query head dim)",
                        meta.hd, meta.hd_kv
                    ));
                }
                let (k_id, v_id) = kv_pair
                    .ok_or_else(|| format!("KV regions for layer {} not allocated", meta.layer))?;
                let nt = node.out_shape[1];
                // G1: dispatch the fast attention kernels (flash / split /
                // parallel) exactly like the legacy layer_gpu path. The fast
                // paths are gated to the isolation-tested shapes (hd 64/128);
                // anything else falls back to the classic kernel.
```

These are audit findings **H1** (`GPU_SAFETY.md` §3): the attention kernels stride the KV cache by `nk*hd`, so a model where the K/V row width `nkt` was built from a *different* head dim (`hd_kv ≠ hd`, possible on other Qwen2.5 family models) would read **out of bounds** — a GPU fault, the class of bug that froze the machine once. Both checks are `Err`s with the actual numbers, per the invariant-violation rule. Then the dispatch decision tree of §2.5: decode (`nt == 1`) tries `flash_attn_enabled(hd)` → `gqa_attn_flash` with a chunk count from `attention_chunks` (one chunk per 32 KV rows, capped at 16, `MINFER_ATTN_CHUNKS` override — `metal_backend.rs:191-198`); else the split two-pass kernel for `hd` 64/128 (unless `MINFER_NO_SPLIT_ATTN=1`); else classic. Prefill tries `attn_flash_prefill`, then the 3-pass `attn_parallel_prefill` (unless `MINFER_NO_MATMUL_ATTN=1`), then classic. Note the vocabulary: "falls back to the classic kernel" here means *another GPU kernel in the same family*, not the CPU — the backend never hands a node back to the CPU mid-flight.

#### 3.2.5 The fused decode ops: fewer dispatches per token

The decode-fusion arms show why the graph's fused nodes exist. `Op::FusedQKV` (`metal_backend.rs:789-861`) is two encodes for what unfused would be ten:

```rust
// src/graph/metal_backend.rs:789-812 (head of the FusedQKV arm)
            Op::FusedQKV { layer } => {
                let meta = match &node.meta {
                    NodeMeta::FusedQkv(m) => m,
                    other => return Err(format!("fused_qkv node missing FusedQkvMeta: {other:?}")),
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.qkv_weight)
                    .ok_or_else(|| format!("qkv weight '{}' not on GPU", meta.qkv_weight))?;
                let nt = node.out_shape[1];
                debug_assert!(nt == 1, "FusedQKV is decode (nt==1) only, got nt={nt}");
                let od_total = meta.nqt + 2 * meta.nkt;
                // 1) concat matmul: x × [wq|wk|wv] → q|k|v concat buffer
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0]),
                    0,
                    self.buf(out_buf),
                    od_total,
                    meta.in_dim,
                    nt,
                );
```

The weight `blk.{i}.attn_qkv` was built at load time by `concat_rows` (`metal.rs:100-130`) — a byte-level row-major concatenation of the Q4_0-or-whatever blocks of `wq|wk|wv`, possible only when all three share a type and input dim. One matmul over the concat produces the `q|k|v` packed buffer; then step 2 (not shown) encodes `attn_bias_rope_store` — one kernel that adds the three biases, applies RoPE to q and k, and scatters k and v into the layer's KV regions at the current position. `FusedFFN` (`:738-787`) mirrors it for gate/up + an *in-place* swiglu (the shader reads gate rows `0..nf` and up rows `nf..2*nf` of the same buffer and writes back into the gate rows — the aliasing the `Backend` contract warned about). `FusedQkvNorm` (`:863-985`, Qwen3) inserts two in-place per-head RMSNorms between the concat matmul and a no-bias rope+store, reusing the `rms_norm_256` kernel with byte offsets into the concat buffer. Each fused arm ends the same way as the unfused ones: encode, `Ok(())`, no waiting — the win is fewer dispatches per decode step (10 → 2), which `METAL_OPTIMIZATIONS.md` §0.1 credits for part of the 0.5B decode gain.

#### 3.2.6 The device layer: one device, ~70 compiled pipelines, one queue

`MpsState::try_new` (`metal.rs:2031-2252`) is where the GPU is actually acquired, and its opening decides *participation*:

```rust
// src/metal.rs:2031-2042
    pub fn try_new() -> Option<Self> {
        if std::env::var("MINFER_DISABLE_MPS").is_ok() {
            eprintln!("MPS: disabled by MINFER_DISABLE_MPS");
            return None;
        }
        // dummy for non-macOS — never called due to cfg
        #[cfg(not(target_os = "macos"))]
        return None;

        #[cfg(target_os = "macos")]
        {
            let device = MTLCreateSystemDefaultDevice()?;
```

`MINFER_DISABLE_MPS=1` returns `None` before anything GPU-ish happens — this is the documented force-CPU switch (`AGENTS.md` build table). Then: grab the default Metal device; optionally start an Xcode GPU capture (`MINFER_METAL_CAPTURE=1`, `:2045-2052`); load the shader library — the build precompiles `metal.metal` into a `metallib` at build time (llama.cpp-style, `:2054-2063`), falling back to a ~0.3-1 s *runtime* source compile when the toolchain was missing, with `MINFER_METALLIB_FILE` as a runtime override for A/B-ing compiler flags. Then comes the part that looks like boilerplate and is actually the capability table made real: ~70 `get_pl("kernel_...")` calls (one `MetalComputePipelineState` per shader entry point — 3 matmul tiers × 7 quant types, 9 `get_rows` variants, 2 rms_norms, elementwise ops, 5 attention families × f32/f16, store_kv × 2, the fused store kernels, warmup) — and any *missing kernel name* makes `try_new` return `None` here, so a shader typo degrades to CPU at startup, loudly printed ("`MPS: no function '...'`"), rather than faulting later. The regression test `metal_pipelines_compile` (`metal.rs:2484`) exists because of exactly that failure mode: *"a duplicate/missing kernel or a Metal compile error makes `MpsState::init` fall back to CPU silently, which looks like a 'GPU throttling' slowdown"* — the Q5_0 incident of 2026-08-06.

Finally the state is built with the device, the **runtime-queried limits**, and the single command queue:

```rust
// src/metal.rs:2163-2169 (fields that matter; the full struct init spans :2166-2240)
            let dummy_buf = device
                .newBufferWithLength_options((1) as usize, MTLResourceOptions::StorageModeShared)
                .unwrap();
            let m = MpsStateInner {
                device: device.clone(),
                max_threadgroup_memory: device.maxThreadgroupMemoryLength() as u64,
                queue: device.newCommandQueue().unwrap(),
                pl_q4_0_f32,
                pl_q4_0_f32_multi,
                pl_q4_0_mm_f32,
                ...
```

`max_threadgroup_memory` is the `GPU_SAFETY.md` §4 rule in code: *"device-specific thresholds … MUST be queried at runtime — never hardcoded."* Every dispatch that uses threadgroup scratch compares its need against this cached value first (the GEMM's 8 KB check at `metal.rs:445-450` is the worked example — §3.3). One `queue` serializes all of the engine's GPU work; `try_new` ends by printing the line every minfer-on-Mac user knows: `MPS: using Metal on Apple M4 Pro (unified: yes)` (`:2241-2249`).

#### 3.2.7 Zero-copy weights: wrapping the mmap in a Metal buffer

The registration chain starts in the loader (doc 03). `load_tensor` (`models/qwen2/loader.rs:202-220`) hands every weight tensor's raw bytes to the Metal registry as it is parsed:

```rust
// src/models/qwen2/loader.rs:202-220
    // Register weight tensors with GPU backends.
    #[cfg(target_os = "macos")]
    if let Some(mps) = crate::metal::MpsState::get() {
        if matches!(
            ttype,
            TensorType::Q4_0
                | TensorType::Q4_1
                | TensorType::Q4_K
                | TensorType::Q5_0
                | TensorType::Q5_1
                | TensorType::Q5_K
                | TensorType::Q6_K
                | TensorType::Q8_0
        ) {
            mps.register_weight(&ti.name, tensor.data());
        } else if ttype == TensorType::F32 {
            mps.register_weight(&ti.name, tensor.data());
        }
    }
```

Registration happens *per tensor at parse time*, before any graph exists — which is what makes the §3.2.8 gate meaningful later. Before the tensors, though, the loader registers the **parts** — the mmap'd GGUF blobs themselves (`loader.rs:319-327`, one `mps.register_part(part.data)` per file part, multi-part included):

```rust
// src/metal.rs:2320-2344 (register_part core; warmup dispatch follows)
    pub fn register_part(&self, data: &'static [u8]) {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = data;
        }
        #[cfg(target_os = "macos")]
        {
            if data.is_empty() {
                return;
            }
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

`newBufferWithBytesNoCopy` is the whole zero-copy story in one call: Metal wraps an *existing* virtual-address range (the mmap of doc 02) as a GPU-visible buffer, with no copy — the hardware requirement is a **page-aligned base** (16 KB on Apple Silicon), which `mmap` guarantees and the `debug_assert` pins. The part is stored as `(base_ptr, len, buffer)`; then `register_weight` (`metal.rs:2374-2425`) resolves each weight by **pointer-range containment**: find the part whose range contains the weight's `data.as_ptr()`, and record `(part_buffer, ptr - base)` as the weight's `(buffer, offset)` — the offset that `execute_node` passes to `setBuffer_offset_atIndex` (§3.2.4). The fallback path (`MINFER_WEIGHT_COPY=1`, or a weight that somehow falls outside every part) copies into a fresh per-weight buffer at offset 0, so correctness never depends on the zero-copy path landing.

`register_part` does one more thing worth understanding — the warmup. Right after wrapping the part it encodes a trivial `kernel_warmup_read` over the whole buffer and submits it. The comment (`metal.rs:2350-2355`) records the measurement: *"the FIRST GPU access to file-backed (mmap) pages costs ~44 ms of one-time page/TLB setup. Doing a dummy full-buffer read HERE (at model load, outside the CLI's Total timing) moves that cost out of the first prefill."* Without it, the first prompt of every session pays a hidden ~44 ms + the GPU's own page-in; with it, load time absorbs the cost and benchmark numbers are equally warm (llama.cpp does the same thing — `ggml-metal-device.m`).

The same loader file also pre-builds the **fusion weights**: when `wq/wk/wv` share a quant type, `concat_rows` (`metal.rs:100-130`) byte-concatenates them and registers the result as `blk.{i}.attn_qkv` (`loader.rs:394-395`); likewise `ffn_gu` (`:457-458`). This is why the decode fusions of §3.2.5 can look up a single weight — the concat exists in the registry before the graph builder ever runs, but the *graph* only uses it when `GraphParams` says fuse (the params-only reuse rule of doc 13 keeps fused and unfused builds distinct).

#### 3.2.8 The all-weights-registered gate and `CParams.gpu`

At forward time, before the graph is built or reused, the model asks a yes/no question (`models/qwen2/graph.rs:421-433`):

```rust
// src/models/qwen2/graph.rs:421-433
        // GPU availability is part of the reuse identity (backend assignment
        // lives in the built graph, not in the params' other fields). Uses
        // attributes rather than `cfg!()` so the `metal_backend` path is not
        // resolved on non-macOS builds (the module does not exist there).
        #[cfg(target_os = "macos")]
        let metal_on =
            crate::graph::metal_backend::metal_available() && Self::weights_on_gpu(model);
        #[cfg(not(target_os = "macos"))]
        let metal_on = false;
        // CUDA participation (Phase 7): requires a usable device AND every
        // matmul weight registered on the CUDA registry in a kernel-supported
        // type (all-or-nothing; 7e③ moved the embedding gather on device, so
        // tok_embd is gated like every other weight).
        #[cfg(feature = "cuda")]
        let cuda_on = crate::cuda::CudaState::get().is_some() && Self::weights_on_cuda(model);
        #[cfg(not(feature = "cuda"))]
        let cuda_on = false;
```

`metal_available()` (`metal_backend.rs:1030-1032`) is just `MpsState::get().is_some()` — device present, not disabled. `weights_on_gpu` (`:628-677`) is the gate itself: it enumerates *every* tensor name the graph will read (embedding, output norm, lm_head, output bias, then per layer the norm, `wq/bq/wk/bk/wv/bv/wo`, the FFN norm, `ffn_gate/ffn_up/ffn_down`) and requires all of them in the Metal registry:

```rust
// src/models/qwen2/graph.rs:665-671 (tail of weights_on_gpu)
        #[cfg(target_os = "macos")]
        {
            let Some(mps) = crate::metal::MpsState::get() else {
                return false;
            };
            names.iter().all(|n| mps.has_weight(n))
        }
```

One `false` ⇒ `metal_on = false` ⇒ the whole model on CPU. Why all-or-nothing instead of per-layer participation? Three reasons: a mixed graph would put a split boundary at every layer (§2.4's sync-per-split cost, multiplied); the layer loop's intermediate buffers would need cross-backend copies of GB-scale activations; and — the decisive one — the reuse identity would become fragile, since `CParams.gpu` could no longer describe "the GPU runs this graph". The gate's name list and `register_graph_weights`' list (`:339-372`, the CPU-side registration) are maintained as mirror images, so "the CPU has it" and "the GPU has it" stay in sync by construction.

The result flows into `CParams` (`:447-468`): `gpu: metal_on || cuda_on`, plus `fuse_qkv`/`fuse_ffn` gated on `nt == 1 && (metal_on || cuda_on)` — on the GPU the fusions are always shape-eligible at decode, on CPU they are not built (the CPU prefers the batched-quantization shape of doc 10). From there, if `try_reuse` fails, the rebuild path enables the backend and assigns:

```rust
// src/models/qwen2/graph.rs:472-486 (rebuild: register → enable → assign)
        if !cache.try_reuse(&params) {
            let mut graph = Self::build(model, &params);
            let sched = BackendScheduler::new();
            {
                let alloc = cache.alloc();
                Self::register_graph_weights(model, alloc);
                #[cfg(target_os = "macos")]
                if metal_on {
                    alloc.enable_metal();
                }
                #[cfg(feature = "cuda")]
                if cuda_on {
                    alloc.enable_cuda();
                }
                sched.assign_backends(&mut graph, alloc);
```

`enable_metal` (`alloc.rs:68-73`) constructs the `MetalBackend` (which succeeds only if MPS is live), and from this moment `assign_backends`'s `alloc.supports(...)` (§2.7's priority order) hands nodes to Metal. When the gate passed, that is *every* node of the Qwen2/Qwen3 graph — the resulting graph is one all-Metal split, and `CParams.gpu=true` means the cached graph keeps that assignment until the params change.

#### 3.2.9 One command buffer per split, flushed by the scheduler

Now assemble §2.4's rhythm from both sides. The scheduler's `execute` (`scheduler.rs:123-354`) walks splits; at every backend change it flushes and copies:

```rust
// src/graph/scheduler.rs:176-189
        for split in &splits {
            if let Some(pb) = prev_backend {
                if pb != split.backend {
                    // 1. flush the previous backend's async work
                    alloc.sync_backend(pb);
                    // 1b. staged Metal/CUDA captures are valid now — read back
                    flush_metal_captures(graph, alloc, &mut staged, trace_on, live_on);
                    flush_cuda_captures(graph, alloc, &mut cuda_caps, trace_on, live_on);
                    // 2. copy this split's inputs across backends
                    for &inp in &split.inputs {
                        alloc.copy_across(inp, split.backend)?;
                    }
                }
            }
```

Step 1 is the command buffer's submit: `sync_backend` (`alloc.rs:542-562`) routes to `MetalBackend::synchronize`, which is one line — `self.submit_pending()` (`metal_backend.rs:1023-1025`). Step 2 is the cross-backend copy of §2.2 (host round trip through the shared buffers, into a fresh staging buffer so the producer's own buffer is untouched for the graph's re-executability). On a fully-Metal graph there is one split, so this `if` never fires mid-graph — but the *final* sync after the loop (`:348-352`) always does, which is where the decode forward's single submit lands. If any node's buffer turned out to live on a different backend than the split executing it, the scheduler returns a hard `Err` ("assignment/alloc mismatch", `:234-241`) — the same no-silent-fallback posture, one level up.

The submit itself (`metal_backend.rs:160-186`) takes the leaked box back, calls `cb.submit()`, and — under `MINFER_OP_PROFILE=1` — accumulates the GPU wait time that §4's profile table prints. The `Drop` impl does the same flush best-effort (`let _ = cb.submit()`) so a backend dropped mid-split cannot leak an unterminated encoder.

#### 3.2.10 `submit()`: the bounded wait

`MpsCommandBuffer::submit` (`metal.rs:1935-1978`) is the GPU-safety centerpiece — the fix for the incident that motivated `docs/GPU_SAFETY.md`:

```rust
// src/metal.rs:1940-1977 (core; encoder-end at :1936-1938)
        // dispatch_semaphore_t is already a reference-counted opaque pointer.
        let sem = unsafe { dispatch_semaphore_create(0) };
        let sem_val = sem as usize;

        let blk = RcBlock::new(
            move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| unsafe {
                dispatch_semaphore_signal(sem_val as *mut c_void);
            },
        );
        unsafe {
            self.cmd_buf.addCompletedHandler(RcBlock::into_raw(blk));
        }
        self.cmd_buf.commit();

        // Bounded wait (10 s). If the GPU hangs (hardware fault), the completion
        // handler never fires and we bail out instead of blocking forever.
        let timeout = unsafe { dispatch_time(0, 10_000_000_000i64) }; // 10 s from now
        let rc = unsafe { dispatch_semaphore_wait(sem, timeout) };
        unsafe {
            dispatch_release(sem);
        }

        if rc == 0 {
            // Command buffer finished (possibly with an error status).
            match self.cmd_buf.status() {
                MTLCommandBufferStatus::Completed => Ok(()),
                st => Err(format!(
                    "Metal command buffer status={st:?}. recent dispatches: {}",
                    self.recent_trace()
                )),
            }
        } else {
            // Timed out: the GPU did not complete the work.
            Err(format!(
                "Metal command buffer timed out after 10s (GPU hang). recent dispatches: {}",
                self.recent_trace()
            ))
        }
    }
```

Walk it as three defenses. **(1) A completion handler on a semaphore**: `addCompletedHandler` registers an Objective-C block that fires when the GPU finishes (or fails) the command buffer; the host sleeps on a `dispatch_semaphore` with a **10-second deadline** — `dispatch_semaphore_wait` returns non-zero on timeout instead of blocking forever. Before this hardening (`GPU_SAFETY.md` §2.1) the code waited `DISPATCH_TIME_FOREVER` and never checked status, so *"a single GPU fault would block minfer forever (and, since Metal clients share the GPU, could stall WindowServer → whole-machine freeze)"*. **(2) A status check**: even when the semaphore fires, the buffer's `MTLCommandBufferStatus` is verified — `Completed` means success; `Error` means a kernel faulted. **(3) A diagnosis trail**: both failure paths append `recent_trace()` — the ring of the last 16 dispatch labels recorded by `trace_op` when `MINFER_TRACE=1` (`metal.rs:317-327`) — so the error message names the kernel family that was last encoded. All three exist because the alternative (an unkillable hang, or an error with no clue) is disproportionately expensive on shared-GPU macOS. Note the return type: `Result<(), String>`, which `submit_pending` turns into `.expect(...)` — a submit failure *panics the run* rather than pretending it succeeded, the same fail-loud rule as `execute_node`'s `Err`s.

#### 3.2.11 Host read/write: views, plus the readback that feeds the sampler

```rust
// src/graph/metal_backend.rs:998-1021
    fn read_host(&self, id: usize) -> Option<&[f32]> {
        let buf = self.pool.get(id)?;
        let len = (buf.length() as usize) / 4;
        Some(unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len) })
    }

    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
        let buf = self.pool.get(id).ok_or_else(|| format!("no buffer {id}"))?;
        let len = (buf.length() as usize) / 4;
        if len != data.len() {
            return Err(format!(
                "buffer {id}: expected {len} elements, got {}",
                data.len()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().as_ptr() as *mut f32,
                data.len(),
            );
        }
        Ok(())
    }
```

This is §2.2's promise made concrete: `read_host` reinterprets the Metal buffer's `contents()` pointer as a Rust `&[f32]` — no copy, no sync — and `write_host` is `memcpy` with a length check. The two call patterns that matter: **inputs** are written here *before* the split's nodes encode (the scheduler skips `Op::Input` nodes precisely because "data pre-filled by the allocator", `scheduler.rs:225-227`), and **outputs** are read after the split's sync — the logits path ends with `alloc.copy_to_cpu(graph.outputs[0])` (`models/qwen2/graph.rs:618`), which lands here once the final submit's completion handler has fired. Reading a buffer whose producing kernels are still *encoded but not submitted* would be the classic race; the one-command-buffer-per-split discipline plus the bounded submit is what makes these plain views safe.

There is one more readback path, used only by the trace/viz tooling (`MINFER_TRACE`, the viz server): `capture_split` (`metal_backend.rs:83-97`) encodes a *blit* pass — `encode_captures` (`metal.rs:363-390`), a GPU→GPU copy into per-split staging buffers appended *after* all the split's kernels — so the capture reads this step's data without forcing a per-node flush. The scheduler queues `(node, staging)` pairs during the split (`scheduler.rs:306-344`) and drains them in `flush_metal_captures` right after the boundary sync (`:394-417`). It is a nice illustration of the submit model: even debug tooling has to work *with* the async pipeline, by scheduling its reads into the same command buffer.

#### 3.2.12 One shader, walked: `kernel_rms_norm_f32`

Every graph node ends in something like this — a Metal Shading Language function in `src/metal.metal`. RMSNorm (doc 11 §3.2.1) is the best first shader: short, and it shows every GPU-programming concept this backend uses:

```metal
// src/metal.metal:2528-2567 — one threadgroup per row
kernel void kernel_rms_norm_f32(
    device const float * x       [[buffer(0)]],
    device const float * w       [[buffer(1)]],
    device       float * y       [[buffer(2)]],
    constant    int    & d       [[buffer(3)]],
    constant    float  & eps     [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]]
) {
    int row = tgpig.x;
    int d4 = d / 4;

    device const float4 * x4 = (device const float4 *)(x + row * d);

    float ss = 0.0f;
    for (int i = tpitg.x; i < d4; i += 32) {
        ss += dot(x4[i], x4[i]);
    }
    int rem = d - d4 * 4;
    if (tpitg.x == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        for (int i = 0; i < rem; i++) ss += x_tail[i] * x_tail[i];
    }
    ss = simd_sum(ss);

    float scale = 1.0f / sqrt(ss / (float)d + eps);

    device float4 * y4 = (device float4 *)(y + row * d);
    device const float4 * w4 = (device const float4 *)w;
    for (int i = tpitg.x; i < d4; i += 32) {
        y4[i] = x4[i] * scale * w4[i];
    }
    if (tpitg.x == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        device       float * y_tail = y + row * d + d4 * 4;
        device const float * w_tail = w + d4 * 4;
        for (int i = 0; i < rem; i++) y_tail[i] = x_tail[i] * scale * w_tail[i];
    }
}
```

Beginner's decoder ring, line by line:

- **`kernel void`** marks an entry point the CPU can launch (the `get_pl("kernel_rms_norm_f32")` of §3.2.6 compiles it into a pipeline). The `[[buffer(N)]]` attributes are the *argument slots* — they match the `setBuffer_offset_atIndex(_, _, N)` and `setBytes_length_atIndex(_, _, N)` calls you saw in `metal.rs`: buffer 0/1/2 are the device pointers for `x`, the norm gains `w`, and the output `y`; buffers 3/4 are small by-value scalars (`d`, `eps`) passed via `setBytes`. This is the whole CPU↔shader calling convention: pointers into shared memory plus a few ints.
- **The built-in coordinates** replace the CPU's loop indices. A GPU launch is a grid of **threadgroups** (here: one threadgroup per row — `rms_norm` dispatches `dispatch_2d(n, 1, 32, 1)` with `n` = row count, `metal.rs:1090`), each holding 32 threads (one **simdgroup** — the hardware unit of 32 lanes that execute in lockstep). `tgpig.x` is "which row am I", `tpitg.x` is "which of my 32 lanes am I". Compare doc 10's `mm_rows`: there the loop variable was a row owned by a worker thread; here it is a row owned by 32 lanes.
- **The strided loop** `for (i = tpitg.x; i < d4; i += 32)` splits the row's `d/4` float4-vectors across the 32 lanes — lane 0 takes vectors 0, 32, 64…, lane 1 takes 1, 33, …. Each `dot(x4[i], x4[i])` is a 4-wide multiply-add per lane: with `d = 896` that is 224 vectors, so 7 iterations per lane. The `float4` cast is free vectorization — the compiler emits 128-bit loads against the row-major layout the doc 07 allocator laid down.
- **`simd_sum(ss)`** is the cross-lane reduction: one instruction that adds the 32 lanes' partial sums and broadcasts the total. On the CPU this was `hsum_float_8`'s shuffle dance (doc 10 §3.2); here the *hardware* does it, in lockstep, with no explicit synchronization — this is the "same math, different execution model" of doc 11 §2.5 in one line.
- **The tail loop** (`rem = d - d4*4`, `if (tpitg.x == 0)`) handles `d` not divisible by 4 in scalar — lane 0 alone sweeps the last `rem` elements. A beginner-relevant detail: *which* lanes do work is decided by lane index, not by data, so all 32 lanes still reach the `simd_sum` together — the barrier-safety rule of §3.4 in miniature.
- **The second pass** reuses the same strided pattern to write `y = x · scale · w`. Note the two passes over `x` (sum of squares, then normalize): on the CPU, doc 11 could do the same because the row fits in cache; a flash-style *online* variant exists for the cases where it does not — here the row is small enough that two passes through L1/L2 are cheaper than saving state across the reduction.

The 256-thread sibling `kernel_rms_norm_f32_256` (`metal.metal:2579-2632`, dispatched by default — `rms_norm_256_enabled()`, ~2× faster per `METAL_OPTIMIZATIONS.md` #16) shows the *other* reduction tool, and the repo's most important GPU-safety rule in its natural habitat:

```metal
// src/metal.metal:2608-2617 — two-barrier handshake across 8 simdgroups
    ss = simd_sum(ss);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ss = shmem[tiisg];
    ss = simd_sum(ss);
```

With 256 threads (8 simdgroups), one `simd_sum` is no longer enough: each simdgroup reduces its own 32 lanes, then lane 0 of each simdgroup publishes its subtotal into **threadgroup shared memory** (`shmem`, declared `[[threadgroup(0)]]`, 8 floats). The two `threadgroup_barrier`s are the handshake: the first guarantees `shmem` is zeroed/ready, the second guarantees *every* simdgroup's store landed before anyone reads. A `threadgroup_barrier` is a lane-count-wide rendezvous — **every** thread must arrive; if any lane took an early `return`, the rest would wait forever and the GPU would deadlock (the machine-freeze class from `GPU_SAFETY.md` §2.2). That is why the rule is "no early return past a `threadgroup_barrier`": guards must be expressed as *predication* (lanes run the code with harmless values, then skip their stores) rather than exits — exactly how this kernel and the attention kernels do it.

### 3.3 Design choices (why this shape and not another)

**Why handwritten kernels instead of Apple's MPS/MPSGraph?** Because minfer has zero ML-framework dependencies and needs llama.cpp-parity numerics. Apple's MPS is a closed library: fixed quant formats, no visibility into summation order, and no way to guarantee the byte-identical greedy outputs the verification gates rely on. Writing kernels in MSL (mostly as transcriptions of llama.cpp's, which `metal.metal` says openly — "faithful llama.cpp transcription") keeps the GGUF block layouts as the single contract between disk, CPU, and GPU, the same way doc 10's CPU kernels do. The cost is real — every quant needs Metal kernels for its tiers, and the optimization campaign is hand-measured (`METAL_OPTIMIZATIONS.md`) — but the control is what makes the correctness claims checkable.

**Why the objc2 ecosystem — how does Rust call Metal without bindgen?** Metal is an Objective-C API; Rust reaches it by sending Objective-C messages. minfer's device layer speaks that protocol through **objc2-metal** (plus `block2` for the completion-handler block and `objc2-foundation` for strings), migrated from the legacy `metal`/`objc` 0.2 crates on 2026-08-25 (`docs/METAL_OBJC-ECOSYSTEM.md` records the whole story). The practical differences you can see in this doc's code: `Retained<ProtocolObject<dyn MTLBuffer>>` is a *typed, ref-counted* handle (RAII — drop releases, versus the old manual `msg_send!` retain/release); method calls are real Rust signatures checked at compile time (the old `msg_send!` with a typo'd selector failed at *runtime*); and `RcBlock` (§3.2.10's completion handler) wraps the Objective-C block safely. The old ecosystem was frozen at `objc` 0.2.7 (2019) and carried a future-rustc breakage that minfer had to vendor-patch; the migration removed the vendored patch, the `metal` crate, and that whole risk class. What did *not* change: the architecture — `MpsState` as the singleton, `MpsCommandBuffer` as the encoder wrapper — survived the migration mostly untouched, which is the best argument that the layering was right.

**Why name-keyed weight registration?** `register_weight(name, bytes)` stores `(buffer, offset)` under the tensor's GGUF name, and `execute_node` looks weights up from `NodeMeta`'s *name strings* (`meta.weight_name`), never from a `Tensor`. The alternative — handing the backend a reference to the tensor — would couple the backend's lifetime to the model object and break on graph rebuilds (doc 13: the graph is rebuilt while the model, and its registered weights, stay put). With names, registration is a one-time load event and every rebuild re-resolves by name; the all-weights-registered gate is then literally a `contains_key` sweep over the same map (§3.2.8).

**Why is Metal first in the priority queue?** `supports` (`alloc.rs:140-157`) checks Metal, then CUDA, then CPU — so on a Mac with both compiled in, Metal wins every op it can run and CUDA gets only what Metal refuses (today: nothing on the standard graph, since `supports_op` covers it all; CUDA's unique claims are its own fusions like `QkvBiasRopeStore`). The order is a policy statement: on Apple Silicon, Metal is the native, zero-copy path — CUDA there runs through a translation layer with real copy costs. On a datacenter box the CUDA backend's `supports_op` is broader (int8 MMQ prefill, doc 15), so "first yes wins" naturally routes to whichever backend is *both* present and most capable per op.

**Why a fresh allocation for cross-split staging (`alloc_fresh`) instead of reusing the free list?** The trait comment (§2.1) is the record of a real bug class: at execute time, free-list ids still back `node_to_buf` entries that later nodes in the *same* execute will read or write; recycling one for a staging copy would clobber in-flight data. The staging buffer is born outside the recycle economy and joins it only at graph-rebuild time, when liveness recycling is safe again (doc 07's allocator owns that transition).

**Why is the f16 KV cache auto-selected by model size?** `set_kv_cache_type` (`metal.rs:147-153`) turns f16 on when `n_layers × n_kv_embd ≥ 8192` — the 7B class — and keeps f32 for the 0.5B class. The asymmetry is measured, not aesthetic: on the 7B, attention streams the whole KV per decode step, so halving those bytes is a direct win (−~1 ms/token at 2K ctx; #13 measured 1.60 → 0.95 s on a long-context case); on the 0.5B the decode is dispatch-latency-bound, the f16 conversion kernels cost more than the bandwidth saves, and f16 measured ~3 % *slower* (the comment cites the decided-not record). `MINFER_CACHE_TYPE=f16|f32` overrides either way.

**Why does `execute_node` return `Err` while some dispatch guards `gpu_abort`?** Two layers, two severities. `Err` is for *graph-level* invariant violations — wrong meta, missing weight, unsupported op — where the scheduler can propagate a clean message naming the node and abort the run. `gpu_abort` is for *dispatch-level* hazards — a shape that would make a shader index out of bounds or overrun threadgroup memory — where the safest thing is to print the actual offending numbers and exit before the GPU ever sees the launch (`GPU_SAFETY.md` §2.3/§4). Both exist because the third option, silently continuing on another path, hides the bug while the model keeps talking.

### 3.4 Pitfalls & invariants

- **Never host-copy a GPU-pending buffer.** Shared memory makes every pool byte look readable at all times — but between `execute_node` and the split's submit, a buffer's *future* contents are still in flight. The Phase-3 KV-corruption bug (recorded in AGENTS.md core rule 5) came from exactly this: an in-place op's input was host-copied while the GPU had pending writes to it. The invariants that make the current code safe: inputs are host-written before their split encodes; outputs are read only after `synchronize`; and in-place GPU ops snapshot their input inside the backend when aliasing is not already guaranteed:

```rust
// src/graph/metal_backend.rs:210-222
    fn copy_in(&self, dst: usize, src: usize) {
        // in-place-ish ops (silu/rope) may alias; snapshot to dst first
        let src_buf = self.buf(src);
        let dst_buf = self.buf(dst);
        let n = (src_buf.length().min(dst_buf.length()) / 4) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_buf.contents().as_ptr() as *const f32,
                dst_buf.contents().as_ptr() as *mut f32,
                n,
            );
        }
    }
```

  The `Silu` and `RoPE` arms call this when `in_bufs[0] != out_buf` (e.g. `metal_backend.rs:334-341`) — the allocator promised the alias is safe *for the same backend* (doc 07's sole-consumer rule), and this memcpy makes it byte-identical to executing in place, host-side or GPU-side.

- **One memory barrier between every pair of dispatches.** Metal orders kernels but does not make writes visible to the next kernel for free; the `barrier()` at the end of `dispatch_1d/2d/3d` (`metal.rs:346-348, 405, 421`) is mandatory glue. The 2026-08-19 incident (`METAL_OPTIMIZATIONS.md` #28, `GPU_SAFETY.md` §3 post-audit finding): without it, the reused `bn` buffer raced between RMSNorm's writes and the next op's reads, producing first-token nondeterminism on 1.5B/7B (~10-30 % wrong tokens) that vanished and reappeared with system load. Corollary recorded there: *when a threadgroup-memory buffer is reused for a different purpose at a different loop stage, a `threadgroup_barrier` must separate the last read from the first write* — the GEMM `temp_str` fix.

- **No early return past a `threadgroup_barrier`** (§3.2.12's ending). The original sin: `kernel_gqa_attn_f32` had `if (h >= nh) return;` before a barrier; when `nh % nk != 0`, some simdgroups exited while others waited at the barrier — *"GPU permanent deadlock = machine freeze"* (`GPU_SAFETY.md` §2.2). The fix pattern — invalid lanes run the full loop on a dummy index and skip only the final store via a `valid_head` flag — is now the review rule for every new kernel, and the flash kernels' discipline (`GPU_SAFETY.md` §4b: mask computed inline per lane, `break`-only control flow on lane-independent conditions, shuffles instead of barrier-protected shared arrays) is the same rule applied to a harder kernel.

- **Device limits are queried, then compared — never guessed.** The worked example is the GEMM's threadgroup-scratch check:

```rust
// src/metal.rs:445-450
        if 8192 > self.state.max_threadgroup_memory {
            gpu_abort(&format!(
                "GEMM needs 8192 B threadgroup memory, device max is {} B",
                self.state.max_threadgroup_memory
            ));
        }
```

  `max_threadgroup_memory` was captured from the device at init (§3.2.6); the guard compares the kernel's real need (4 KB + 2 KB + reused 8 KB staging, per the comment at `metal.rs:430-432`) and aborts with *both* numbers. The rule exists because the first draft hardcoded a guessed 32 KB for the M4 Pro (`GPU_SAFETY.md` §4) — a number that would be silently wrong on the next chip.

- **Quant-shape assumptions get refused, not absorbed.** The K-quant `id % 256` guard (§3.2.4, audit M1) exists because the failure mode is *wrong numbers, no crash* — the worst kind. The same audit table accepts genuinely low-risk gaps with reasoning (L1: matmul pointers computed past the buffer for out-of-range rows but reads guarded; L2: `store_kv` trusts the host to keep positions < capacity, which doc 07's region sizing guarantees). Every accepted risk is written down with its mitigation — the audit doc is the invariant ledger.

- **The registry and the graph must agree on shapes.** `FusedQkvNorm`'s per-head norms assume the concat layout (`q` at byte 0, `k` at `nqt*4` — `metal_backend.rs:906-911`); `attn_bias_rope_store` assumes the same packing; the f16 KV kernels assume the region's first half is theirs (§3.1). These cross-file layout contracts are exactly where doc 10's lesson applies: they are tested with real-scale isolation tests (`metal_attn_kv_real_scale`, `metal_store_real_dims`, … — the `#[cfg(test)]` module at `metal_backend.rs:1035`), not just small shapes, because the transposed-output bug of doc 10 hid from `nt == 1` tests.

- **A shader typo must fail at startup, not at token 500.** §3.2.6's pipeline table makes every kernel name a startup-checked fact; the `metal_pipelines_compile` test pins it in CI. The Q5_0 incident (2026-08-06) — a duplicate symbol that silently downgraded the process to CPU and "looked like GPU throttling" — is the reason this is treated as an invariant rather than an annoyance.

## 4. Observe & verify

- **Startup line**: a Metal-enabled run prints `MPS: using Metal on <device> (unified: yes)` then `MPS: GPU acceleration enabled` (`metal.rs:2241-2249, 2262`). Its absence — or `MPS: not available, using CPU fallback` — is the first thing to check when numbers look CPU-shaped; `MINFER_DISABLE_MPS=1` produces the explicit `MPS: disabled by MINFER_DISABLE_MPS`.
- **`MINFER_GRAPH_TRACE=1`** prints the split table (`scheduler.rs:127-144`): one line per split (`split 0: Metal nodes 0-440`) plus an op×backend census — the quickest way to *see* the all-Metal split a passing gate produces, or the CPU stragglers when the gate failed.
- **`MINFER_OP_PROFILE=1`** turns on the backend's built-in profiler (`metal_backend.rs:26-54, 224-242`): after the first submit it prints a host-encode-per-op table (top 20 by time), then one line per submit with the GPU wait — prefill shows one big submit, decode shows one line per token. This is the tool that measured the 32-vs-256-thread RMSNorm dispatch cost (#16).
- **`MINFER_TRACE=<dir>`** records per-node real-data traces (doc 08's staged Metal capture path — blits at split end, read after sync); the same env var arms `submit()`'s dispatch-label ring, so a Metal error/timeout message names the last 16 kernels encoded.
- **A/B levers for every §2.5 decision**: `MINFER_NO_FLASH=1` (decode flash → split), `MINFER_NO_PREFILL_FLASH=1`, `MINFER_NO_MATMUL_ATTN=1` (parallel prefill → classic), `MINFER_NO_SPLIT_ATTN=1`, `MINFER_NO_RMS_256=1`, `MINFER_GEMM=0` (GEMM tier off), `MINFER_CACHE_TYPE=f16|f32` (KV width), `MINFER_ATTN_CHUNKS=N`, `MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` (fusion off — changes `CParams`, forces a rebuild). Each is a one-env-var kernel-family A/B, the same levers the optimization campaign used.
- **`MINFER_METAL_CAPTURE=1`** starts an Xcode GPU capture at device init (`metal.rs:2045-2052`) — open the .gpu capture in Xcode to see every encoded dispatch of a run; `MINFER_METALLIB_FILE=<path>` swaps the precompiled shader library at runtime; `MINFER_WEIGHT_COPY=1` forces the copied-weight path to isolate zero-copy registration bugs.
- **Tests** (macOS, `cargo test`): `metal_pipelines_compile` (`metal.rs:2484`) fails if any kernel is missing or the library does not compile — the anti-silent-CPU-fallback guard; the `metal_backend.rs` test module carries per-op correctness gates against host-computed references, e.g. `metal_matmul_q8_matches_cpu` builds a graph, runs it through the real scheduler with every node forced to Metal, and asserts max diff < 1e-3 against a manual Q8×f32 reference (`:1326-1333`), plus cross-backend copy, split alternation, KV attention, decode-step, and real-scale (d=896) variants; the greedy end-to-end gates of doc 12 close the loop (byte-identical greedy output across the optimization campaign is the standard the records claim).
- **The honest caveat**: on a CPU-only build these tests print `MPS unavailable; skipping` and pass — Metal coverage exists only where Metal does. A no-op GPU test suite is one of the failure modes `docs/GPU_SAFETY.md`'s recurrence playbook is written for.

## 5. Cross-references

- [06 — Assign and fusion](06-assign-fusion.md) — how `supports_op`/`supports_fused` are consumed at build time; the priority walk of §2.7 starts there.
- [07 — Allocator, liveness, and KV regions](07-allocator-liveness-kv.md) — who owns the buffers this backend pools, the aliasing rule `copy_in` implements, and the persistent KV regions the KV ops write.
- [08 — Scheduler and execute](08-scheduler-execute.md) §3.2 — the split loop, cross-backend copies, and the "one Metal command buffer per split" rule this doc implements from the backend side.
- [10 — CPU matmul kernels](10-cpu-matmul-kernels.md) — the execution model this doc replaces: Q8_0 activations vs f32 (§2.3), thread-pool rows vs simdgroup lanes, and the bandwidth physics both share.
- [11 — Attention, vec ops, and the KV cache](11-attention-vecops-kv.md) §2.5 — the GPU preview this doc expands (flash vs full-row softmax, f16 KV); §3.2's CPU RMSNorm is this doc's shader in scalar form.
- [13 — The decode loop and graph reuse](13-decode-loop-graph-reuse.md) — why `CParams.gpu` is part of the reuse identity and when the gated graph gets rebuilt.
- [15 — The CUDA backend](15-cuda-backend.md) — the same Backend trait where every unified-memory shortcut of §2.2 becomes an explicit copy, plus int8 MMQ and CUDA Graph replay.
- `docs/GPU_SAFETY.md` — the hard rules and the incident report behind §3.2.10, §3.2.12, and §3.4; read before touching Metal/CUDA code.
- `docs/METAL_OPTIMIZATIONS.md` — the kernel campaign behind §2.5/§2.6 (§0.1's table maps every graph op to its kernel, with measurements).
- `docs/METAL_OBJC-ECOSYSTEM.md` — why objc2, what the 2026-08-25 migration changed, and the nix/Xcode toolchain gotchas (§3.2.6's metallib story).
- `docs/metal-inference-analysis.md`, `docs/PERF-QWEN3-4B-VS-LLAMACPP.md` — deeper per-kernel analyses and the Qwen3-4B llama-parity A/B.

← [13 — The decode loop and graph reuse](13-decode-loop-graph-reuse.md) · [Index](./README.md) · [15 — The CUDA backend](15-cuda-backend.md) →
