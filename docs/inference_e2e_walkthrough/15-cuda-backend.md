# 15 · The CUDA backend

> **Stage**: the Metal contrast (doc 14) → **this stage: the same compute graph on NVIDIA GPUs** → end of the series ([Index](./README.md)).
> **Code**: `src/graph/cuda_backend.rs` (`CudaBackend`, `execute_node_inner`, `graph_replay_step`), `src/cuda.rs` (`CudaState` singleton, `register_weight`, `matmul_f32_ptr_layout`, `prefill_mmq`, `gqa_attn_split`), `src/cuda_kernels.cu` (the CUDA C++ kernels, ~7,700 lines), `build.rs` (the nvcc build chain).

## 1. Background — where this stage sits

Docs 08 and 13 left the inference loop in a particular shape: the scheduler
walks each backend-uniform **split** of the compute graph and calls
`execute_node` for every node, and the decode loop repeats that walk once per
generated token, reusing the cached graph. Doc 10 showed what a `MatMul` node
becomes on the CPU; doc 14 showed the *same nodes* executing on Apple's GPU
via Metal. This document completes the trilogy: the same graph, the same
allocation rules, the same safety contract — executed on an NVIDIA GPU.

The CUDA backend is the third backend of the engine and the only one that is
**opt-in at build time**. A plain `cargo build --release` never touches the
CUDA toolchain at all; adding `--features cuda` makes `build.rs` locate
`nvcc`, compile `src/cuda_kernels.cu` into a static library, and link it in
(`docs/BUILD.md` records the full recipe, including which GPU architectures
get machine code baked in). That opt-in flag is why the file
`src/graph/cuda_backend.rs` — the subject of most of this document — is
wrapped in `#[cfg(feature = "cuda")]` and simply does not exist in other
builds.

Three source files cooperate, and keeping their roles separate makes the rest
of the document easy to follow:

- **`src/cuda.rs` (~5,600 lines of Rust)** — the *device layer*. A
  process-wide singleton, `CudaState`, owns the CUDA context pieces: the
  device, one command stream, the weight registry (name → device pointer),
  the KV-cache regions, pinned-host staging pools, and one host-side wrapper
  function per kernel. It talks to the CUDA runtime through hand-written
  `extern "C"` declarations (there is no `-lcuda` crate and no bindgen — the
  FFI surface is explicit and auditable).
- **`src/graph/cuda_backend.rs` (~6,400 lines, mostly tests)** — the *graph
  backend*. It implements the same `Backend` trait as the CPU and Metal
  backends (doc 08): `supports_op`, a device buffer pool, `execute_node`,
  host read/write, `synchronize`. Its job is translation, not math: turn a
  `CNode` into one or two kernel launches on the shared stream, and enforce
  the kernel invariants loudly (`Err`) when they are violated.
- **`src/cuda_kernels.cu`** — the *kernels themselves*, CUDA C++ compiled by
  nvcc. Every function the backend calls is a `launch_*` wrapper in
  `cuda.rs` that eventually reaches a `__global__` kernel here.

Everything upstream of this doc is unchanged by the backend swap: the graph
was built by the model code (doc 05), backends were assigned at build time by
`supports_op` priority Metal → CUDA → CPU (doc 06), buffers were placed by
the liveness allocator with two persistent KV regions per layer (doc 07). The
CUDA backend executes whatever it was assigned; it never rewrites the graph
and never falls back to the CPU mid-run.

What this document adds beyond the Metal story is three CUDA-specific
mechanisms, each answering a question the other backends never had to ask:

1. **Prefill runs an int8 tensor-core GEMM ("MMQ")** — quantized weight bytes
   feed NVIDIA's `mma` integer matrix instruction directly, llama.cpp-style,
   instead of dequantizing first (§2.2–§2.3, §3.2.5–§3.2.6).
2. **Decode attention is "split-KV"** — the KV dimension is cut into 32
   stripes processed by 32× more warps than heads alone would occupy, then
   reduced (§2.4, §3.2.7).
3. **Decode steps are recorded as CUDA Graphs and replayed** — one launch per
   split instead of hundreds of kernel launches (§2.5, §3.2.8).

The implementation arc itself is worth knowing up front, because the code
carries its history in comments: the backend was built in five planned
sub-phases (7a skeleton → 7b per-op parity → 7c model wiring → 7d graph
replay → 7e polish, all recorded in `docs/CUDA-BACKEND-PLAN.md`), and then a
long measurement-driven optimization campaign (Phase 8 and the r-series,
recorded in `docs/CUDA_OPTIMIZATION.md` plus one document per step in
`docs/cuda_optimization_steps/`) turned a working backend that ran 7B prefill
at 30.7 tok/s into the current default path at ~3,581 tok/s — 1.080×
llama.cpp on the same hardware and model. §2.3 and §3.3 cite that record
where a design decision needs its measurement.

## 2. Principle — how it works and why

### 2.1 Five GPU words you need before the code makes sense

NVIDIA's execution model can be compressed into five terms, and every kernel
excerpt in §3 uses them:

- A **thread** is the unit of work — it computes, say, four elements of one
  dot product. A kernel launch creates a **grid** of threads organized into
  **blocks**; the hardware schedules whole blocks onto **SMs** (Streaming
  Multiprocessors, the GPU's ~100-odd cores).
- Threads within a block can share a small **shared memory** scratchpad and
  synchronize with barriers. Threads are grouped 32 at a time into a **warp**,
  which executes one instruction across its lanes in lockstep — a warp
  reduction (`__shfl_xor_sync`) sums 32 lanes in 5 steps, which is why the
  attention kernel in §3.2.7 maps one warp to one KV stripe.
- **Occupancy** is how many warps an SM can keep resident at once. Latency
  (a few hundred cycles for a VRAM read) hides only behind *other* warps'
  work, so "more resident warps" is the default medicine — until shared
  memory or registers per block cap it. Several optimization records in this
  doc are occupancy stories (r40's third resident block, D3-4's dual-kernel
  dispatch).
- **Coalescing**: adjacent threads should read adjacent addresses so one
  memory transaction serves the whole warp. The Q6_K weight repack (§3.2.2)
  exists purely to make this possible.
- The **stream** is the GPU's in-order work queue. Everything the backend
  enqueues — copies and kernels — runs in issue order on one stream, which is
  precisely what makes the pinned-staging async fills (§3.2.9) race-free and
  what CUDA Graph capture records.

One more word: a **tensor core** is a piece of silicon inside each SM that
executes small matrix-multiply-accumulate instructions (the `mma` family) at
many times the rate of ordinary arithmetic. The f16 flavor (`wmma`) is the
floor for minfer's CUDA kernels — sm_70 (Volta) is the minimum supported
architecture because of it (`docs/BUILD.md`) — and the int8 flavor
(`mma.m16n8k32`, sm_80+) is what prefill uses (§2.3).

### 2.2 Two forwards, two physics: why prefill and decode run different matmul kernels

The single most important fact about GPU inference is that **prefill and
decode are bound by different resources**, and the backend dispatches on that
difference. Recall the two phases (docs 09 and 13):

- **Prefill** pushes all prompt tokens through the graph at once. Each matmul
  is now a true GEMM (General Matrix-Multiply): `nt` token rows stream
  through the same weight matrix, so the weights are read **once** while the
  arithmetic grows with `nt`. With enough tokens, the GPU runs out of math to
  do before it runs out of bytes to fetch — *compute-bound*.
- **Decode** produces one token per forward. Every matmul must still read its
  **entire weight matrix** to produce that one token's outputs — there is
  nothing to amortize over. *Weight-streaming-bound*: the speed of light is
  VRAM bandwidth, not FLOPs.

The engine's own numbers make both limits visible. The 7B Q4_K_M model's
weights total ~4.4 GB; measured decode is ~51.2 tok/s (`README.md`
performance table), which is ~225 GB/s of sustained weight streaming —
exactly the "200–225 GB/s class" the dispatch comments in `cuda.rs` report
for the decode kernels. That also shows why quantization matters just as much
on a GPU as on a CPU (doc 10 §2.1): the same model in f32 weights would move
4× the bytes per token and stream 4× slower, and would not fit next to the
KV cache on most GPUs anyway (14B Q4_K_M already occupies ~14.1 GB of device
memory). Decode reads **every weight byte per token from VRAM** — "weight
streaming" — so bits-per-weight converts 1:1 into tokens-per-second.

Prefill is the opposite trade. Once `nt` is large enough, the weight bytes
are a one-time cost and the tensor cores want maximum math throughput. That
is why `matmul_f32_ptr_layout` — the single dispatch every CUDA matmul flows
through — opens with a token-count gate:

```rust
// src/cuda.rs:2556-2575 (dispatch gate; abridged comment)
if nt >= 9
    && id % 32 == 0
    && !Self::no_prefill_gemm()
    && matches!(ttype, TensorType::Q4_0 | TensorType::Q4_1
        | TensorType::Q5_0 | TensorType::Q5_1 | TensorType::Q8_0
        | TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K)
{
    if self.mmq_active() {
        return self.prefill_mmq(wptr, ttype, x, out, od, id, nt, padded_q6k);
    }
    return self.prefill_gemm_f16(wptr, ttype, x, out, od, id, nt, padded_q6k);
}
```

`nt >= 9` (Step 82 lowered it from 16) sends prefill-shaped batches to one of
two tiled GEMMs; the `nt == 1` decode shapes fall through to per-type
kernels that keep the f32 activations and optimize for weight streaming.
There is even a middle band: `nt` 2–8 runs *multi-token MMVQ* — MMVQ with an
in-block token loop — because a batch that small still cannot fill GEMM
tiles, but re-running the whole per-token path per token wastes weight reads
(Step 82, `docs/CUDA_OPTIMIZATION.md` §0 row 82).

So the answer to "why does prefill use int8 MMQ while decode uses a different
path?" is not taste — it is which resource is scarce in each phase:

| Phase | Shapes | Scarce resource | Kernel family | Why |
|---|---|---|---|---|
| Prefill | `nt ≥ 9` | math throughput | int8 MMQ tensor-core GEMM (or f16 `wmma` GEMM) | one tiled GEMM; tensor cores do the MACs; weights stream once |
| Small batch | `nt` 2–8 | weight bytes, but too few rows for tiles | multi-token MMVQ (dp4a, token loop in-block) | weights-once like a GEMM, launch-lean like decode |
| Decode | `nt == 1` | VRAM bandwidth | per-type MMVQ (dp4a, one row per block) + f32-activation kernels | maximize bytes/s; integer dots are free alongside the stream |

### 2.3 MMQ, explained from zero: int8 tensor cores eat quantized weights

**MMQ** (llama.cpp's name for "matrix-matrix quantized") is the trick the CPU
doc 10 introduced in scalar form — keep weights quantized, quantize
activations to int8 on the fly, do an integer dot — lifted onto NVIDIA's
int8 tensor-core instruction. The pieces:

- The **weights are never dequantized**. Their raw block bytes
  (`block_q4_K`'s nibbles, sub-scales, mins...) are staged into shared memory
  as-is and decoded in registers right where the `mma` instruction needs
  them.
- The **activations are quantized once per matmul call** into padded int8
  blocks — `nt × (id/32) × 40` bytes (each 32-value block takes 40 bytes: 32
  int8 values plus scale/sum fields the kernels consume). A small prepass
  kernel does this before the GEMM.
- The `mma.m16n8k32.s8` instruction multiplies an 16×32 int8 tile by a 32×8
  int8 tile and accumulates into int32 — exact integer math, exactly like the
  AVX2 `vpmaddubsw` chain of doc 10, but 8× wider and issued by tensor-core
  silicon.
- The **float scales fold in afterwards**: each accumulated integer tile gets
  multiplied by the matching weight/activation block scales once, outside the
  integer loop, before being added to the f32 output accumulator.

This is llama.cpp's structure, transplanted: `docs/LLAMA-CPP-MMQ-ANALYSIS.md`
dissects the reference kernel instruction-by-instruction, and minfer's
kernel grew against it round by round. The campaign record
(`docs/CUDA_OPTIMIZATION.md` §0) shows the arc — the first parity-clean
int8 MMQ measured 441 tok/s of 7B prefill; the promoted default path measures
~3,581 tok/s, **8.1×** — with each lever measured in isolation:

| Campaign step (docs/cuda_optimization_steps/) | Lever | Effect |
|---|---|---|
| R1 (step 08) | first int8 MMQ GEMM, opt-in | parity-clean but 8× off llama — the gap was unprofiled |
| r34 (step 37) | quantize+**transpose** activations in the prepass (llama's `quantize_mmq_q8_1` design) so the GEMM's A-staging is a bulk copy | +9.72% whole prefill |
| r41 (step ~22 of r-series) | q6_K "B-expand" plane widened to `uint4` group loads — 32 per-byte loads per thread-k had become the stall | q6_K kernel −61.5% time |
| r52 | RMSNorm/SwiGLU producers *fuse* the activation quantize (mode 2 skips the f32 output write entirely) | +5.45% |
| r59/r60 | q4_K scale-pair plane + **promotion: the verified gate set flips default-on** | +11.1%; final 1.080× vs llama.cpp |

Two details of that table deserve a beginner's pause. First, the **planes**
(W_exp, W_dsc): for q4_K/q6_K the kernels can either decode the packed
sub-scales inside the hot loop or read precomputed f32 scale pairs staged at
load time — trading ~3 GB of extra VRAM (the default path peaks at ~9.5 GB
for 7B, vs ~20.5 GB for the legacy f16 escape, which is *heavier*, not
lighter — `CUDA_OPTIMIZATION.md` §1.1) for removing a stall from the inner
loop. Second, **promotion** (r60) is the campaign's exit ritual: a lever
proves itself behind an opt-in env gate, the A/B record accumulates, and only
then does the default flip — with `"0"` opt-outs kept so every step stays
A/B-able forever.

And why does *decode* not want MMQ? At `nt == 1` the GEMM degenerates to a
matrix-vector product: there is one output row, so the 16-row M-tile of
`mma.m16n8k32` is 15/16 wasted, and the binding resource is weight bytes
anyway. The decode path instead quantizes the single activation row to the
same padded int8 format and runs **MMVQ** ("matrix-vector quantized"): one
256-thread block per weight row, integer `dp4a` (4-way int8 dot) over the
row's blocks, one launch table per type. Measured: +74–77% on 7B shapes for
q4_K (8e②), and the K-quant arms carry measured shape gates for when the
integer path loses to the simpler f32-activation kernel (§3.2.5 shows the
exact gates).

### 2.4 Split-KV decode attention: thousands of threads, few heads

Attention on the CPU (doc 11) parallelizes over heads. During prefill that is
fine — the grid also spans tokens. But decode has **one** query token, so a
per-head kernel for Qwen2.5-7B occupies 28 warps... on a GPU that fits
hundreds. The rest of the machine idles while each warp serially walks the
whole KV history.

**Split-KV** (the "split-K" family, flash-decoding style) adds a second
parallel axis: cut the KV sequence dimension into `ATTN_SPLITS = 32` stripes
and launch one warp per *(stripe, head)* pair. Each warp computes a *partial*
attention over its stripe only, and a second small kernel merges the 32
partials. Two pieces of math make the merge correct, and both are worth
internalizing because they appear verbatim in the kernel excerpt of §3.2.7:

1. **Online softmax** (the "flash attention" trick, already previewed in doc
   11 §2.5). Instead of a first pass to find the row max, each new score
   `s = q·k·scale` updates a running max `m` and rescales everything seen so
   far: with `corr = exp(m_old − m_new)`, the running sum `S ← S·corr +
   exp(s − m_new)` and the running output accumulator `oc ← oc·corr +
   exp(s − m_new)·v`. Nothing overflows, and one pass suffices.
2. **Split merging.** A stripe's partial is exactly `(m_sp, S_sp, oc_sp)`.
   The global max is `gmx = max_sp m_sp`; each partial's weight is
   `w_sp = exp(m_sp − gmx)`; then `S = Σ w_sp·S_sp` and `out = Σ w_sp·oc_sp /
   S`. The rescaling makes the 32 partial sums combine into precisely the
   softmax the single-warp version would have computed — up to float
   addition order, which is a *semantic* difference here: merging 32 partials
   reorders the float sum, so split-KV output is not bitwise-identical to a
   one-stripe scan (the D1 record measured the drift at ~1e-9 magnitude). That
   is why `ATTN_SPLITS` is frozen at 32 rather than tuned per context length:
   the split grid is baked into captured CUDA Graphs (§2.5), and changing it
   would change numerics, not just speed.

The payoff was immediate (8d: 7B decode 10.1 → 13.7 tok/s, +36%), and the
campaign kept refining the body: staging K and V rows into registers before
the serial softmax chain (D2: kernel 34.1 → 19.4 µs/launch), a dim-parallel
rewrite (R4: +10–15%), and a dual-kernel dispatch that swaps in a 4-warp body
only when the per-warp row count is high enough to amortize it (D3-4: the
same structure ran *slower* at short context — rows-per-warp pathology — so
both kernels launch and each self-gates on the device-side context length,
keeping CUDA-Graph replay valid).

### 2.5 CUDA Graph capture/replay: one launch per split

A decode step executes ~150–300 nodes — hundreds of kernel launches on the
stream. Each launch carries a few microseconds of host-side submission
overhead, and at decode's ~50 tok/s pace the overhead is a measurable tax
(7d measured +18% decode on 0.5B from removing it).

A **CUDA Graph** is a recorded, replayable bundle of stream work. Capture the
sequence of launches once, and every later execution of the identical
sequence becomes a single `cudaGraphLaunch`. That is a *huge* if — "identical"
means identical kernel parameters, which includes **identical device
pointers**. The whole capture machinery in `cuda_backend.rs` exists to keep
that promise:

- The graph's node buffers live in the backend's device pool and are
  **reused across decode steps** (doc 07's allocator never frees while the
  graph is alive), so every kernel keeps the same address arguments step
  after step.
- Anything address-affecting bumps a `pool_gen` counter, and captured graphs
  keyed to an older generation are destroyed and re-captured (§3.2.8 shows
  the invalidation branch).
- **No host decisions inside the window**: the causal attention bound is
  read from the device-side positions buffer inside the kernel (the `Attn`
  arm's comment calls this out as a *precondition* for replay), and input
  data is H2D-copied into the stable staging addresses *before* the split, so
  replay reads fresh data through fixed pointers.
- Capture uses llama.cpp's warmup protocol: executions 1 and 2 of a split run
  direct launches (warming up per-kernel state), the 3rd opens a capture
  window, and every execution after that replays. One-shot graphs — a CLI
  prefill that runs once — never reach 3 runs and never pay capture cost.
  Repeated identical-length prefills (the server's slot scenario) do capture,
  by default since R3-B (`MINFER_NO_PREFILL_CAPTURE=1` opts out).

The replay hook lives in the scheduler (doc 08's split loop), not in the
backend alone: before executing a CUDA split the scheduler asks
`graph_replay(uid, node_range, nt_hint)`; a `true` answer means the captured
graph covers the whole node loop and the loop is skipped. Tracing/viz capture
forcibly disables replay — per-node host readbacks inside a capture window
are exactly the "host decisions inside the window" that break it.

### 2.6 What the GPU *doesn't* change

It is worth stating the invariants explicitly, because the series has spent
ten documents building them and the CUDA backend preserves every one:

- **KV positions are data, not structure** — the attention kernel derives its
  causal bound from the device positions buffer at run time.
- **Weights are the GGUF bytes** — registered to the device once at load;
  execution never host-copies a weight (llama.cpp's "ops follow their
  weights" rule, `CUDA-BACKEND-PLAN.md` §3).
- **Backend assignment is a build-time decision** — `supports_op` + the
  all-weights gate decide placement before the first forward; a mid-run
  invariant violation is an `Err`, never a silent CPU detour (§3.4).
- **CPU-vs-GPU logits differ by design** — the GPU path uses f32 activations
  (int8 only inside the MMQ GEMM), so parity gates compare each path against
  its own reference plus greedy-token equality, never cross-path bitwise.

One thing *does* change on the GPU: the KV cache element type. The device KV
regions may be **f16** — halving attention's read bandwidth — auto-selected
for models where KV streaming dominates decode (`n_layers × n_kv_embd ≥
8192`, so 7B and up; `MINFER_CACHE_TYPE` overrides), mirroring the Metal
policy (doc 14). Store and attention come in matching f32/f16 kernel pairs;
doc 11's invariants (positions are data, windows are `pos[t]+1`,
store-before-attention) are untouched.

## 3. Implementation

### 3.1 Data in / data out

The backend's whole world is device pointers. Every buffer the allocator
hands it (doc 07) is an id into its device pool; every weight is a name
resolved to a device address; the KV regions are pool slots that simply never
rejoin the free list.

| Item | Layout | Where it lives |
|---|---|---|
| Node buffers (activations) | `[nt][d]` f32, token-major — 4 bytes/elem device buffers | `CudaBackend.pool: Vec<CudaBuf>` (`cudaMalloc`'d; free-list recycled) |
| Weights | raw quantized bytes exactly as the GGUF has them (`[out][in]` row-major, block layouts of doc 02); Q6_K repacked to 224-byte block slots; f32 norms/biases as-is | `CudaState.weights: HashMap<String, (CudaPtr, usize)>` — registered once at load |
| MMQ activation scratch | `[nt][id/32]` × 40 B padded int8 blocks (+ transposed variant for the raw kernels) | `CudaState` grow-on-demand scratch buffers (`buf_q8_prefill`, `buf_q8_decode`) |
| MMQ precomputed planes | q6_K expanded-B / scale planes, q4_K f32 scale-pair planes (~3 GB total on 7B) | keyed by weight device pointer in `CudaState` maps |
| KV regions | per layer, `[n_ctx][nkt]` f32 **or f16** (auto-selected, §2.6) | the pool's persistent regions (`kv_pair(layer)`, doc 07) |
| Positions | `[nt]` I32 — arriving as `f32::from_bits` bit patterns (doc 07), decoded *on device* to raw int32 by a tiny kernel | `pos_scratch`, memoized per execution window |
| Host→device input fills | Rust `&[f32]` → pinned staging slot → `cudaMemcpyAsync` H2D | 8 × 2 MiB `cudaHostAlloc` ring (`staging`) |
| Device→host readback | logits (and viz/trace node dumps) D2H through a pinned readback buffer | `readback` / `CaptureStaging` (128 MB ceiling) |

Two asymmetries against the CPU backend are worth noticing. First, the CPU
matmul quantizes activations per call into a `Vec<u8>` (doc 10); the CUDA
backend keeps them in **persistent device scratch** that grows on demand and
is memoized per execution window — because a `cudaMalloc` in the hot path
would sync the device and because decode re-quantizes the *same* producer
output every step. Second, `read_host` returns `None` unconditionally: a
staged D2H transfer cannot hand back a borrowed slice, so host reads go
through `copy_to_host` (the allocator's `copy_to_cpu` CUDA arm) instead.

### 3.2 Key code

#### 3.2.1 The build chain: what `--features cuda` (and `cuda_static`) actually do

`build.rs` only runs the CUDA section when the cargo feature is set — plain
builds never touch nvcc. With the feature requested, CUDA is *required*: a
missing toolkit is a hard error rather than a silent CPU-only binary, because
`src/cuda.rs` declares `launch_*` symbols that only the compiled kernel
archive can satisfy.

```rust
// build.rs:181-240 (abridged)
let nvcc = match find_nvcc() {
    Some(n) => n,
    None => panic!(
        "CUDA feature requested but nvcc was not found — install the CUDA \
         toolkit or point CUDA_HOME at its root (e.g. /usr/local/cuda)"
    ),
};
...
// nvcc inherits the first cc/g++ on PATH as its host compiler and
// hard-fails when that is newer than the toolkit supports. Keep nvcc's
// own default whenever it works ...; only pin -ccbin when the default is
// rejected.
let (ccbin, ccbin_label) = match std::env::var("MINFER_CUDA_CCBIN") {
    Ok(v) if !v.is_empty() => (Some(v), format!("MINFER_CUDA_CCBIN={v}")),
    _ => match detect_host_compiler(&nvcc, &out_dir, &include_flag) {
        Some(HostCompiler::Default) => (None, "nvcc default".to_string()),
        Some(HostCompiler::Pinned(c)) => {
            println!("cargo:warning=CUDA: nvcc default host compiler rejected, \
                      pinning -ccbin {c}");
            (Some(c.clone()), format!("-ccbin {c}"))
        }
        None => panic!(...),
    },
};
let archs = detect_archs(&nvcc, &out_dir, &include_flag, ccbin.as_deref());
...
let cudart_static = std::env::var_os("CARGO_FEATURE_CUDA_STATIC").is_some();
```

Read the three decisions:

- **Host compiler pinning (`-ccbin`)**: nvcc compiles *host* C++ too, and it
  rejects host compilers newer than the toolkit supports (a nix devShell
  putting GCC 15 first breaks a CUDA 13 that accepts ≤ GCC 13). The build
  probes nvcc's default first and only pins an older GCC when that probe
  fails; `MINFER_CUDA_CCBIN` forces one explicitly.
- **GPU arch coverage**: `detect_archs` tries `sm_70…sm_121` and keeps what
  the toolkit accepts — SASS (native machine code) per supported
  architecture plus PTX for the highest, so **one binary covers older and
  newer GPUs** (older ones JIT the PTX forward). The floor is sm_70/Volta
  because the f16 `wmma` kernels require tensor cores.
- **`cuda_static`**: the plain feature links cudart *shared* (the binary
  needs `libcudart.so.N` and a CUDA toolkit runtime at run time, with an
  rpath baked in). `--features cuda,cuda_static` links `libcudart_static.a`
  instead: the binary has **no `libcudart.so` dependency** — it needs only
  the NVIDIA driver `libcuda.so.1` (never a link-time dependency;
  `dlopen`'d at run time, with a `preload_driver` helper that resolves it
  from well-known paths because nix shells bypass `/etc/ld.so.cache`) plus
  libstdc++. That is the deployment story: build on a toolkit machine, copy
  the binary to a driver-only machine.

#### 3.2.2 Device init and the weight registry

`CudaState::try_new` probes devices, honors `--gpu N` (falling back to
auto-selecting the highest compute capability), creates the one shared
stream, queries the device properties at run time — SM count, compute
capability, free/total memory, name — and prints the banner you see at
startup (`CUDA: using ... (SM 12.1, ... MB, ... SMs)`). Two details are
load-bearing beyond the boilerplate: the compute capability is stored as an
integer (`major*100 + minor`) and later gates the int8 MMQ path
(`mmq_active` requires `cc >= 800`, i.e. sm_80+, because `mma.m16n8k32`
exists only from Ampere on); and `gemm_prefill_smem_init()` runs *eagerly* —
opting the prefill GEMM into >48 KB dynamic shared memory is illegal inside
a stream-capture window, so it must happen before any capture can open.

Weights register through `register_weight` — a `cudaMalloc` plus one
blocking H2D `cudaMemcpy` of the raw GGUF bytes:

```rust
// src/cuda.rs:1528-1561 (core of register_weight)
let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
if err != 0 || ptr.is_null() {
    eprintln!("CUDA: failed to allocate {} bytes for '{}'", data.len(), name);
    return;
}
let err = unsafe {
    cudaMemcpy(
        ptr,
        data.as_ptr() as *const std::ffi::c_void,
        data.len(),
        CUDA_MEMCPY_HOST_TO_DEVICE,
    )
};
if err != 0 {
    eprintln!("CUDA: failed to copy '{}' to device", name);
    unsafe { cudaFree(ptr) };
    return;
}
// a plain (unpadded) registration must clear any stale padded flag
// for the same name ...
self.padded_weights.lock().unwrap().remove(name);
self.weights.lock().unwrap()
    .insert(name.to_string(), (CudaPtr(ptr), data.len()));
```

The comment lines this excerpt elides contain two ownership rules that
matter: a re-registration with the **same name and size reuses the existing
device copy** (unit tests reload the same file; weights are immutable, so
there is nothing to refresh), while a *different* size replaces the entry and
**deliberately leaks the stale buffer** — a live captured CUDA Graph may
still reference the old address, and the leak is bounded by the number of
distinct tensor shapes ever loaded. Device memory ownership is serious
enough that this one rule got its own bullet in `docs/GPU_SAFETY.md`.

Special layouts register through special entries. The important one is Q6_K:
the raw 210-byte block stride forces 1-byte-per-instruction weight loads that
capped 7B decode near ~38 GB/s, so the loader repacks each block into a
**224-byte slot** (a multiple of 16) and the kernels stream with aligned
`uint4` loads:

```rust
// src/models/qwen2/loader.rs:234-244
if ttype == TensorType::Q6_K {
    // 7e②: register Q6_K in the padded 224-byte block layout so
    // the matmul kernel can use aligned uint4 weight loads
    // (the raw 210-byte stride forces 1-byte-per-instruction
    // reads and caps 7B decode near ~38 GB/s).
    cuda.register_weight_q6k_padded(
        &ti.name,
        tensor.data(),
        tensor.shape[1] as usize,
        tensor.shape[0] as usize,
    );
}
```

That one repack was a 3.1× decode win (8.4 → 26.4 tok/s on 7B — the 7e②
record), and it made `has_weight_of_size` match padded entries by their
*original raw* length so the participation gate (below) stays honest.

Also at load, the loaders flip global policy switches that must be settled
before the first forward: `set_kv_cache_type` auto-selects the f16 KV policy
from model dims, and a non-q4_K/q6_K quantized weight (or any 2-D f32 matmul
weight) clears the `nb_bt_only` flag, which degrades the skip-write fused
producers of §3.2.6 to a safe mode — the flag exists so a mixed-quant model
can never feed a dead buffer to a GEMM that reads f32 activations.

#### 3.2.3 Build-time assignment: `supports_op` and the all-weights gate

The CUDA backend claims almost the whole per-layer chain — everything except
the ops it has no kernel for (a standalone `Softmax` node; the `Scale` node)
and RoPE in a layout it does not implement:

```rust
// src/graph/cuda_backend.rs:1259-1300 (abridged to the decision arms)
fn supports_op(&self, op: &Op, dtype: DType) -> bool {
    if dtype != DType::F32 {
        return false;
    }
    match op {
        Op::Input
        | Op::Add | Op::Mul | Op::Silu | Op::SwiGLU
        | Op::RmsNorm { .. } | Op::QkNorm { .. }
        | Op::MatMul { .. } | Op::Attn { .. }
        | Op::KvcacheStore { .. } | Op::KvcacheLoad { .. }
        | Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. }
        | Op::GetRows                     // 7e③: embed + tail gather on device
        | Op::FusedQKV { .. }             // D3-8: decode QKV fusion (G4 port)
        | Op::QkvBiasRopeStore { .. }     // D3-8: mixed-quant QKV epilogue
        | Op::FusedFFN => true,           // 7e⑤: decode FFN fusion
        // MatMul ttype gating happens at the model level (weights must all
        // be registered on CUDA — same all-or-nothing rule as Metal).
        Op::RoPE { style } => matches!(style, RopeStyle::NonInterleaved),
        _ => false,
    }
}
```

The allocator's `supports()` walks priority Metal → CUDA → CPU, so on a
CUDA build every claimed node lands in a single CUDA split per graph (the
7e③ record: embedding and the G3 tail gather moved on device precisely to
eliminate the last cross-backend copies — a prefill/decode graph is now *one*
CUDA split, no host round trips at all).

But per-op support is not sufficient: a graph with *one* weight missing from
the device registry would interleave CPU and CUDA execution against
persistent KV regions. So the model wiring applies an **all-or-nothing
gate** before the graph is even built:

```rust
// src/models/qwen2/graph.rs:430-451 (abridged)
// CUDA participation (Phase 7): requires a usable device AND every
// matmul weight registered on the CUDA registry in a kernel-supported
// type (all-or-nothing; 7e③ moved the embedding gather on device, so
// tok_embd is gated like every other weight).
#[cfg(feature = "cuda")]
let cuda_on = crate::cuda::CudaState::get().is_some() && Self::weights_on_cuda(model);
...
cparams: CParams {
    n_ctx,
    n_batch: nt,
    flash_attn: false,
    gpu: metal_on || cuda_on,
    ...
},
```

`weights_on_cuda` (`qwen2/graph.rs:689`) walks every weight the graph reads
— embedding, output head, every layer's norms/biases/matmul weights — and
requires each to be (a) registered and (b) a type with a matching kernel.
The type check is per *role*: matmul weights admit all eight quant types
plus f32, while the embedding gather lacks a Q4_1 kernel, so a Q4_1
`tok_embd` keeps the whole model on CPU. When the gate fails it names the
first offending tensor in the log rather than emitting a generic complaint.
The boolean result is recorded in `CParams.gpu`, which makes GPU
participation part of the **reuse identity** (doc 13): flipping CUDA on or
off deterministically changes the built graph, so a stale reuse can never mix
assignments.

#### 3.2.4 `execute_node`: one match, many kernels

The scheduler calls `execute_node(node, in_bufs, out_buf, kv_pair)` per
node. The CUDA implementation is a single match over the op enum; each arm
resolves device pointers (failing loudly if any is missing), checks the
kernel's structural invariants, and launches. The wrapper adds one
capture-window duty:

```rust
// src/graph/cuda_backend.rs:1354-1375
fn execute_node(
    &mut self,
    node: &CNode,
    in_bufs: &[usize],
    out_buf: usize,
    kv_pair: Option<(usize, usize)>,
) -> Result<(), String> {
    match self.execute_node_inner(node, in_bufs, out_buf, kv_pair) {
        Ok(()) => Ok(()),
        Err(e) => {
            // A node error during an open capture window dooms the window:
            // the scheduler propagates before the boundary sync, so nothing
            // would close it — later input fills would be RECORDED into the
            // window and the eventual close would cache a multi-step graph
            // (double KV commit on every replay). Abort the window loudly.
            if self.capturing.is_some() {
                self.abort_capture(&e);
            }
            Err(e)
        }
    }
}
```

A tour of the arms, in execution order per layer, tells you what actually
runs on the GPU:

- `Input` / `KvcacheLoad` — **no kernel**. Inputs were H2D-filled by the
  allocator before the split (§3.2.9); a KV load is a *view* of the
  persistent region (`out_buf` *is* the region).
- `View`/`Reshape`/`Permute` — a device-to-device copy (layout-only nodes
  keep the CPU backend's identity-copy semantics).
- `GetRows` — the embedding gather (dequantize-on-gather, type-dispatched on
  device, ids read from the I32-as-f32 buffer) or the generic f32 tail
  gather for the `n_out` reduction (doc 05).
- `RmsNorm`/`QkNorm` — the float4 RMSNorm kernel; **decode producers fuse an
  int8 quantize epilogue** (the `n == 1` branch at the bottom of the arm)
  so the following MMVQ group skips its standalone quantize launch.
- `MatMul` — the decision tree of §3.2.5.
- `RoPE` — neox-style rotate in place (a D2D copy first if the allocator did
  not alias input and output); positions decoded from the f32-bits buffer by
  `positions_i32`.
- `KvcacheStore` — scatter k/v rows into the persistent regions at the
  positions; f32 or f16 destination per the KV policy.
- `Attn` — the split-KV decode kernel (`nt == 1`) or the prefill attention
  kernel, §3.2.7.
- `FusedQKV` / `QkvBiasRopeStore` / `FusedFFN` — the decode fusions (doc 06):
  one concat matmul + one fused bias/rope/store (or offset-swiglu) launch,
  decode-only (`nt != 1` returns `Err` — the kernels are not shaped for
  prefill), replacing 3 matmuls + bias×3 + rope×2 + store×2 launches with
  two.
- The catch-all arm (line 1156) returns `Err("cuda: op ... has no kernel
  (stays on the CPU backend per supports_op)")` — a deferral that can only
  fire if `supports_op` and the dispatcher drift apart, which is exactly the
  loud-abort behavior the safety contract wants.

#### 3.2.5 The matmul decision tree

Both prefill and decode funnels converge on
`CudaState::matmul_f32_ptr_layout(weight_ptr, ttype, x, out, od, id, nt,
padded_q6k)`. Its opening gate you saw in §2.2 (`nt >= 9` → MMQ or f16
GEMM). What falls through is the decode-side dispatch, and reading one arm
teaches you the shape of all of them:

```rust
// src/cuda.rs:2628-2644 (Q4_K arm; abridged comment)
TensorType::Q4_K => {
    // 8e-reversal: decode (nt == 1) runs the MMVQ structure
    // (dp4a over q8 activations, one row per 256-thread block) —
    // +74–77% at 7B shapes (bench8e2); id >= 2048 gate (below
    // that it is launch-latency noise), id % 32 == 0 for the
    // sub-block tail granularity. Prefill keeps the f32 kernel.
    if nt == 1 && id >= 2048 && id % 32 == 0 {
        self.q4_k_decode_mmvq(wptr, x, out, od, id, nt);
        Ok(())
    } else if nt >= 2 && nt <= 8 && id % 32 == 0 {
        // Step 82: multi-token MMVQ (in-block token loop).
        self.q4_k_decode_mmvq_multi(wptr, x, out, od, id, nt);
        Ok(())
    } else {
        launch!(launch_q4_k_f32_matmul)
    }
}
```

Three things to take from this arm:

1. **The shape gates are measured, not guessed.** Below `id = 2048` the
   kernel-count win is lost in launch latency; the q5_K/q6_K arms carry an
   even sharper measured crossover (`od*id ≥ 24M` for q5_K, lowered to 4M for
   q6_K's attn_v class) — under the crossover the simpler coalesced
   f32-activation kernel wins because the MMVQ byte loads are uncoalesced and
   1–2 units per thread expose their latency (the arm comments cite the
   on-device micro-bench numbers).
2. **The activation quantize is shared machinery.** `q4_k_decode_mmvq` calls
   `decode_quantize_native`, which consults the `MmqCache` first — when the
   preceding fused RMSNorm/SwiGLU already wrote the padded int8 plane for
   this exact source buffer, the standalone quantize launch is skipped
   entirely (D3-5 1a: standalone quantize launches dropped 78% at 14B).
3. **Every arm ends in a launch or an `Ok(())`** — the dispatch never
   returns "unsupported" for a type `supports_op` admitted; the
   `weights_on_cuda` gate guaranteed the type exists here.

The full tree, in one table:

| Condition | Path |
|---|---|
| `nt ≥ 9`, quant type, MMQ active (sm_80+, `MINFER_MMQ` on) | `prefill_mmq` — int8 tensor-core GEMM (§3.2.6) |
| `nt ≥ 9`, quant type, MMQ off | `prefill_gemm_f16` — dequant weights to an f16 scratch, f16 `wmma` GEMM (the pre-campaign path; also what `MINFER_MMQ=0` escapes to) |
| `nt == 1`, per-type shape gate passes | MMVQ dp4a kernel (q4_K `id ≥ 2048`; q5_K `od·id ≥ 24M`; q6_K `od·id ≥ 4M`) |
| `nt` 2–8 | multi-token MMVQ (weights-once token loop) |
| everything else | per-type f32-activation kernels (the original Phase-7 family; every supported type has one) |

#### 3.2.6 Inside `prefill_mmq`: the int8 GEMM and its fallback ladder

`prefill_mmq` maps the quant type to a small integer id, pre-checks the
activation scratch for OOM (so a failure surfaces *before* any launch),
picks the Q6_K block stride (224 padded vs 210 raw), and then walks a
**ladder of increasingly general kernels** — each specialized variant tries
first and each failure falls through cleanly:

```rust
// src/cuda.rs:3356-3413 (the q4_K raw-byte branch, abridged)
// P6: raw-byte staging variant (q4_K, whole super-blocks only).
// Same quantized activations; the GEMM stages RAW weight bytes via
// cp.async and dequants in registers (docs/CUDA_OPTIMIZATION.md).
if type_id == 5 && Self::mmq_gate_on("MINFER_MMQ_RAW") && (id / 32) % 8 == 0 {
    ...
    // P6 r34: relocate the A-side layout transform out of the mma
    // kernel into a quantize-transpose prepass (llama.cpp's design).
    ...
    let (qa8g, sdag) = self.mmq_quantize_transposed(
        x as *const f32, id as i32, nt as i32, nchunk, ntb, stream,
    );
    // r59: the q4_K W_dsc f32-pair plane (null on miss ->
    // the DSC=false in-kernel scalar decode instantiation).
    let w_dsc = self.q4k_dsc.lock().unwrap()
        .get(&(wptr as usize)).map(|cp| cp.0)
        .unwrap_or(std::ptr::null_mut());
    nb_ok = qa8g != 0 && sdag != 0
        && launch_mmq_raw_nb_bt_nt(
            type_id, wptr as *const u8, w_dsc as *const u8,
            qa8g as *const u8, sdag as *const u8,
            out as *mut f32, nt as i32, od as i32, id as i32,
            nchunk, stream, kd,
        ) == 1;
    ...
}
```

The vocabulary, translated: **A** is the activation matrix, **B** the
quantized weight matrix; "raw NB-BT" is the fastest kernel family (raw weight
bytes staged by `cp.async` — the hardware copy engine — into shared memory,
nibbles unpacked in registers, `mma` per k-chunk); the **transposed prepass**
(`mmq_quantize_transposed`) emits the int8 activations already laid out the
way the GEMM's staging loop wants them, removing the layout transform from
the hot kernel (+9.72% whole prefill); and the **W_dsc plane** is the
load-time precomputed f32 scale-pair table that replaces in-kernel sub-scale
decoding (+11.1%). Each plane lookup is `null` on miss, and the launcher
returns `0` when its preconditions (shared-memory caps, geometry gates)
fail — which is how the ladder degrades to the generic `launch_mmq_nt`
fallback at the bottom:

```rust
// src/cuda.rs:3504-3521 (generic tail)
unsafe {
    let q8 = self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
    if q8 == 0 {
        return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
    }
    launch_mmq_nt(
        type_id,
        wptr as *const u8,
        q8 as *const u8,
        out as *mut f32,
        nt as i32,
        od as i32,
        id as i32,
        block_stride,
        stream,
    );
}
Ok(())
```

Note the one path that is *not* a clean fallback: when the A-quantize helper
returns 0 because the r52 skip-write guard refused to re-quantize a dead
buffer, `prefill_mmq` returns `Err` (§3.4). Perf fallbacks are safe —
correctness fallbacks are not, and the code keeps the distinction visible.

#### 3.2.7 Split-KV attention, the kernel

The host side (`gqa_attn_split`, `cuda.rs:4169`) computes the partial-row
stride `pstr = (4 + hd + 3) & !3` (running max, running sum, then the
`hd`-wide output accumulator, rounded to a 16-byte boundary for the `float4`
writes), grows the partials scratch *once* (a fixed `[32][nh][pstr]` slab —
`nh`/`hd` are graph constants, so it never grows inside a capture window),
and launches two kernels on the stream: the partial kernel and the combine.

The partial kernel's grid is `(ATTN_SPLITS, n_head)` — 32 stripes × heads —
with one warp (32 threads) per block. Each lane owns 4 consecutive head
dimensions (a 128-dim head fills all 32 lanes × 4 = 128 dims exactly; the
dispatch rejects `hd > 128` or `hd % 4 != 0` before launch). The core loop
is the online softmax of §2.4:

```rust
// src/cuda_kernels.cu:2878-2920 (attn_split_1w_body core loop)
for (int base = lo; base < hi; base += 4) {
    int nr = min(4, hi - base); // warp-uniform
    // D2: stage BOTH K and V for the whole 4-row window before the first
    // softmax step. All 8 row loads then issue back-to-back and their
    // latency overlaps the serial chain; ...
    float4 k4[4], v4[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        k4[j] = (live && j < nr)
            ? kv_ld4<KV>(k + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        v4[j] = (live && j < nr)
            ? kv_ld4<KV>(v + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j >= nr) break; // warp-uniform: all lanes exit together
        // Full-row dot: this lane's 4-dim partial, then a warp reduction
        // so every lane holds the row's complete dot (uniform softmax).
        float d = q4.x * k4[j].x + q4.y * k4[j].y
                + q4.z * k4[j].z + q4.w * k4[j].w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFF, d, off);
        float s = d * scale;
        float nmx = fmaxf(mx, s);
        float corr = expf(mx - nmx);
        float e = expf(s - nmx);
        S = S * corr + e;
        mx = nmx;
        if (live) {
            float4 vv = v4[j];
            oc.x = oc.x * corr + e * vv.x;
            oc.y = oc.y * corr + e * vv.y;
            oc.z = oc.z * corr + e * vv.z;
            oc.w = oc.w * corr + e * vv.w;
        }
    }
}
```

Walk it as a beginner: each warp iteration covers 4 KV rows. The warp first
stages 4 K rows and 4 V rows into registers (memory latency overlaps the
following serial math — the D2 record measured −42% on cold DRAM from
exactly this scheduling). Then per row: every lane multiplies its 4 query
dims by its 4 key dims, `__shfl_xor_sync` butterflies the 32 partials into a
full row dot (5 shuffle rounds), the online-softmax rescale updates `mx`,
`S`, and the 4-dim output accumulator `oc`. Each lane owns *distinct output
dims*, so there is no cross-lane reduction at the end — lane `i` simply
writes its 4 accumulator dims into the stripe's partial. Stripes beyond the
KV end write an empty partial (`mx = -INF`, `S = 0`), which the combine
weights to zero.

The combine kernel is 20 lines and re-derives the whole answer:

```rust
// src/cuda_kernels.cu:2959-2978
__global__ void gqa_attn_split_combine(
    const float* __restrict__ partial,
    float* __restrict__ o,
    int nh, int hd, int pstr
) {
    int h = blockIdx.y;
    int i = threadIdx.x; // hd threads
    if (i >= hd) return;
    float gmx = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        gmx = fmaxf(gmx, partial[((size_t)sp * nh + h) * pstr]);
    float S = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        const float* p = partial + ((size_t)sp * nh + h) * pstr;
        float w = expf(p[0] - gmx);
        S += p[1] * w;
        acc += p[4 + i] * w;
    }
    o[h * hd + i] = (S > 0.0f) ? acc / S : 0.0f;
}
```

One thread per output dim; 32 reads of the partials slab; the §2.4 merge
math verbatim. (For f16-KV, hd==128 models at long context, a second
4-warp body exists and *both* kernels launch with static grids, each
self-gating on the device-side context length — the D3-4 record's answer to
a dispatch that must stay replay-safe.)

Two design consequences to keep: the grid is **static** — no kernel reads
`n_past` on the host to size the launch, so capturing this launch in a CUDA
Graph and replaying it at any later context length is safe (the kernel
re-derives everything from `positions[0]` per replay); and the partials
scratch is **size-stable**, so its address never churns under a captured
graph.

#### 3.2.8 CUDA Graph capture/replay, the state machine

`graph_replay_step` is called by the scheduler before each CUDA split
executes (`scheduler.rs:201`) and returns whether replay covered the whole
node loop. Three states live in the backend: `graph_execs` (the captured,
instantiated graphs keyed by `(uid, range)`), `graph_runs` (the warmup
counter per key), and `capturing` (an open capture window, closed by
`synchronize`):

```rust
// src/graph/cuda_backend.rs:199-250 (abridged)
let key = (uid, range);
if let Some(pos) = self.graph_execs
    .iter().position(|g| g.uid == uid && g.range == range)
{
    if self.graph_execs[pos].pool_gen != self.pool_gen {
        // pool churned since capture — pointers may differ, re-capture
        let g = self.graph_execs.remove(pos);
        self.graph_runs.remove(&key);
        self.state.graph_destroy(g.exec);
    } else {
        let exec = self.graph_execs[pos].exec;
        // a plain stream launch — serialized like any other stream op
        let _sg = self.stream_guard();
        if self.state.graph_launch_exec(exec) {
            return true;
        }
        eprintln!("CUDA: graph replay launch failed; graphs disabled for this session");
        self.graphs_mode = GraphMode::Disabled;
        return false;
    }
}
let runs = self.graph_runs.entry(key).or_insert(0);
*runs += 1;
// 8g①: capture decode-shaped graphs by default ... Prefill-shaped graphs
// (nt > 1) capture only with the prefill_capture gate: 8g② made
// that a deliberate opt-in after an audit caught unvalidated
// capture; R3-B (2026-08-31) flips the default ON — the 3-run
// protocol bounds the cost ...
if *runs >= 3
    && self.capturing.is_none()
    && nt_hint.map_or(true, |nt| nt == 1 || self.prefill_capture)
{
    // Hold the process-wide stream lock across the capture window:
    // any other backend's stream work would otherwise be recorded
    // into this graph (capture is per-stream, not per-thread).
    let guard = self.state.stream_lock().lock().unwrap();
    if self.state.graph_begin_capture() {
        self.capturing = Some(key);
        self.stream_guard = Some(guard);
    } else {
        drop(guard);
        eprintln!("CUDA: stream capture unavailable; graphs disabled for this session");
        self.graphs_mode = GraphMode::Disabled;
    }
}
false
```

Every line answers a "what could go wrong":

- **Pool churn** (`pool_gen` mismatch): any buffer allocation can move
  pointers; the exec keyed to the old generation is destroyed and the next
  executions re-warm and re-capture. This is the pointer-stability promise
  of §2.5 enforced mechanically.
- **Replay failure**: logs, disables graphs for the session, returns
  `false` — the split then executes by direct launch, still correct.
- **The 3-run protocol**: executions 1–2 direct-launch (llama.cpp's warmup
  ×2), the 3rd opens the window. `nt_hint` (the first MatMul node's token
  count, `graph/mod.rs:127`) distinguishes decode-shaped from
  prefill-shaped graphs so the prefill capture gate applies only where it
  was validated.
- **The process-wide stream lock**: stream capture is per-stream; while one
  backend holds an open window, *every other* backend's stream work (the
  CPU-side fills of a mixed split boundary, another model slot's launches)
  must block instead of being recorded into the graph. The capturing
  backend holds the mutex across its window and its own enqueues skip
  re-locking.

Closing the window happens in `synchronize` → `close_capture_or_sync`
(`cuda_backend.rs:268`): end capture, **instantiate**, launch once so the
step still produces output, cache the exec, and only then release the stream
lock. If instantiate/launch fails, the recorded launches never executed —
this step's outputs are undefined, and the code says so out loud (graphs
disabled for the session; rerun with `MINFER_NO_CUDA_GRAPH=1`), instead of
pretending the step succeeded.

One subtlety ties the whole section together: input staging. Replay reads
whatever is *in* the recorded input addresses at launch time, so the
allocator's `fill_input` H2D copies happen **before** `graph_replay` is
called for the split — fresh token data lands at the same addresses every
step, which is the invariant that makes one captured graph serve every
decode step.

#### 3.2.9 The buffer pool, pinned staging, and the sync contract

The pool mirrors Metal's (doc 14): a `Vec<CudaBuf>` plus a byte-length-matched
free list. Two details are CUDA-specific:

```rust
// src/graph/cuda_backend.rs:1307-1327 (alloc_buffer)
fn alloc_buffer(&mut self, size: usize) -> usize {
    let _sg = self.stream_guard(); // cudaMalloc syncs the device
    let bytes = size * 4;
    if let Some(pos) = self.free.iter()
        .position(|&id| self.pool[id].bytes == bytes)
    {
        let id = self.free.remove(pos);
        self.pool_gen += 1;
        return id;
    }
    // On OOM, cuda_malloc logs and returns null; the null buffer fails
    // cleanly (Err) at execute time via ptr_of — do NOT panic here: the
    // backend may be holding the process-wide stream lock, and panicking
    // under a mutex poisons it for every other user.
    let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
    self.pool.push(CudaBuf { ptr, bytes });
    self.pool_gen += 1;
    self.pool.len() - 1
}
```

Both the free-list reuse and the fresh `cudaMalloc` bump `pool_gen` — the
replay-invalidations trigger of §3.2.8. `free_buffer` *recycles* (never
`cudaFree`s) so persistent KV regions survive rebuilds, and teardown is
`Drop`'s job. And the OOM comment is worth a second read: allocating under
the stream lock means panicking would poison the mutex for every other
backend user, so OOM is carried as a null pointer and converted to `Err` at
first use.

Host transfers avoid the pageable-memory penalty with **pinned staging**.
Input fills (`write_host`) copy into a slot of a lazily-allocated ring of
8 × 2 MiB `cudaHostAlloc` buffers and enqueue `cudaMemcpyAsync` — returning
before the copy lands, which is race-free *because* everything consumer-side
runs later on the same stream (GPU_SAFETY rule 5). Logits readback goes
through a pinned readback buffer for the same reason in reverse: a blocking
`cudaMemcpy` into pageable memory bounces through a driver-internal pinned
buffer (R3-A2). Viz/trace node dumps queue async D2H copies into a 128 MB
`CaptureStaging` arena and drain with one sync at the split boundary —
replacing what would otherwise be a per-node full-stream sync.

`synchronize` (called by the scheduler at split boundaries) closes the
capture window if one is open — otherwise it plain-syncs — and also clears
the two execution-window memos (the MMQ A-quantize cache and the positions
i32 conversion), because at a boundary the pool reuses buffer ids for
*different* data and a stale memo would alias old content onto a new node.
The sync itself is bounded and checked, per GPU_SAFETY:

```rust
// src/cuda.rs:2200-2209
pub fn sync(&self) {
    let err = unsafe { cudaGetLastError() };
    if err != 0 {
        eprintln!("CUDA kernel launch error: {}", err);
    }
    let err = unsafe { cudaStreamSynchronize(self.stream()) };
    if err != 0 {
        eprintln!("CUDA stream sync error: {}", err);
    }
}
```

Launch errors are checked here, at sync points, rather than after every
launch (rule 4) — the stream serializes everything, so one checked sync
after a batch of launches observes all of their errors.

### 3.3 Design choices (why this shape and not another)

**Why two layers (a `CudaState` singleton wrapped by a `CudaBackend`)?**
`cuda.rs` predates the graph (it began as a direct-inference device layer)
and `CUDA-BACKEND-PLAN.md` §2.3 made the call explicit: *wrap, do not
rewrite*. The singleton owns everything device-global (the stream, the
weight registry, the KV regions, staging pools) and is shared by tests and
the legacy surface; the backend owns everything graph-shaped (the buffer
pool, capture state machine, per-node dispatch). The benefit shows at the
seams: the model loader talks only to `CudaState` (register weights), the
scheduler talks only to the `Backend` trait, and neither sees the other.

**Why hand-written kernels instead of cuBLAS?** cuBLAS has no quantized-weight
GEMM entry point that consumes llama.cpp block layouts, so the quantized
matmuls — the whole point of the engine — would need manual dequantization
into f16/f32 scratch anyway (the pre-8m path did exactly that, at 30.7
tok/s). The campaign's answer was to write the quantized GEMMs directly
against the `mma` instruction, and the plan doc lists "cuBLAS paths" under
deliberately-skipped llama.cpp machinery. The payoff is that weights are
*never* materialized in f32 — the 8p f16 cache, the one exception, costs
+8.6 GB on 7B and was itself made obsolete by MMQ (the MMQ gate skips the
warm pass, `cuda.rs:2762-2765`).

**Why is capture keyed on `(uid, range, pool_gen)` instead of llama.cpp's
node-props snapshot?** llama.cpp memcmps per-node properties and can update a
captured exec in place; minfer's topology is already deterministic in
`GraphParams` (doc 13), so the only thing that can invalidate a capture is a
device-address change — exactly what `pool_gen` tracks. The simpler key
buys a much smaller state machine, at the cost of a full re-capture (warmup
×2 again) whenever allocation churns; decode steps don't churn, so the
common case pays nothing.

**Why capture on the 3rd run, and why is prefill capture on by default?**
Warmup exists because per-kernel state (dynamic shared-memory opt-ins,
module loading) must be settled before capture — capturing a first-call
launch would record a slow path forever. The 3-run protocol also makes
capture *self-funding*: one-shot graphs never pay for it, and repeated
identical-length prefills (the server slot scenario, where the same `nt`
prefills over and over) amortize capture across many replays. The 8g②
default-off interlude is instructive: an audit found unvalidated prefill
capture, the default flipped off until the bit-parity harness (pp16/pp300)
proved replay at real prefill scale, and R3-B flipped it back on — the
promote-then-default pattern the optimization campaign uses everywhere.

**Why f16 KV on the GPU (and f32 on CPU)?** Decode attention streams the KV
history per token; at 7B that is 28 layers × 512 kv-dims × 2 (K and V) ×
2 bytes (f16) × context — halving it is an ~11% whole-decode win (8b). The
CPU path stays f32 because its attention is compute/latency-carried, not
KV-bandwidth-bound, and f32 keeps the reference path simplest. The policy
boundary is measured, not aesthetic: `n_layers × n_kv_embd ≥ 8192`.

**Why a 224-byte padded Q6_K layout (§3.2.2) and precomputed scale planes
instead of faster kernels?** Both are memory-layout answers to a
memory-bound problem: the padded stride makes weight loads coalesced, the
planes (W_exp/W_dsc, ~3 GB) move scale decoding out of the inner loop.
Both carry opt-outs (`MINFER_MMQ_Q6K_EXP=0` saves 1.52 GB for −5%; r54's
record quantified the exact trade), because VRAM is the one resource you
cannot grow at run time.

**Why does the norm arm *require* a weight when Metal's degrades?** Metal's
rms_norm silently runs weightless if the gains are not registered on the
backend — a debuggable-but-wrong result. CUDA's `norm_weight` returns
`Err("cuda: weight '...' not registered")` instead
(`cuda_backend.rs:1220-1246`, comment citing `docs/GPU_SAFETY.md`). The
difference is deliberate: the all-or-nothing participation gate makes a
missing CUDA weight *unreachable* through normal wiring, so reaching that
error means an invariant is already broken — and broken invariants abort.

**Why is the backend feature-gated at all, when Metal is unconditional?**
Metal ships with macOS; CUDA needs a toolkit at build time and a driver at
run time that not every machine has. The opt-in flag keeps plain builds
nvcc-free (`build.rs` never probes for it), keeps CI green on GPU-less
machines, and pairs with `cuda_static` for the toolkit-free deployment
story of §3.2.1.

### 3.4 Pitfalls & invariants

- **Never sync inside a capture window.** A `cudaStreamSynchronize` inside
  the window corrupts the capture — the 7e② record's "faster but wrong"
  incident was exactly this: a temporary debug sync inside the matmul
  dispatch produced garbage *only when graphs were enabled*. Rule 2 of
  `docs/GPU_SAFETY.md` exists because of it; the general lesson (a perf win
  that changes load counts and breaks only one configuration is a bug) is
  recorded alongside.
- **Invariant violations return `Err` — with the values.** The `Attn` arm
  rejects `nkt != n_head_kv*hd`, mismatched query/KV head dims, `hd > 128`,
  `hd % 4 != 0`, and non-divisible GQA head counts, each message carrying
  the actual numbers; the MatMul arm rejects non-multiple-of-32 quant dims
  and buffer-size mismatches; FusedQKV/FusedFFN reject `nt != 1`; RoPE
  rejects any non-neox style; a transposed-B matmul is refused outright.
  The catch-all arm turns any un-kernelled op into an error naming the op.
  None of these fall back to CPU — placement was decided at build time, so
  a guard failure means the run aborts with the blocking node's name.
- **The dead-write refusal (r52's skip-write mode).** In default mode 2, a
  fused RMSNorm/SwiGLU producer writes *only* the int8 quantize plane and
  skips the f32 output. If any GEMM then tries to quantize that f32 source
  again, `mmq_quantize_native` refuses (the `dead_write` flag in the
  `MmqCache`) and `prefill_mmq` returns `Err("... mode-2 dead-write A
  refused ...")` — a silent garbage read was the alternative, and the
  record says the guard "turns any window violation into a loud error
  instead of reading the unwritten buffer". The `nb_bt_only` flag
  (§3.2.2) degrades mode 2 to mode 1 *at load time* when the model's quant
  mix makes skip-write unsound at all.
- **Positions and ids cross the host boundary as bit patterns.** The
  allocator fills I32 inputs as `f32::from_bits` (doc 07); the GPU never
  converts them on the host — `positions_i32` runs a tiny device kernel
  (`bits_to_i32`) once per execution window (memoized), and the embed
  kernel reads token ids with `__float_as_int` (the 7e③ record warns that
  `__float2int_rn` would read the denormal float and always yield 0).
- **Memo lifecycles are execution-window-bounded.** The MmqCache, the
  positions memo, and the capture staging all key on or clear at
  `synchronize` — pool buffer ids are recycled across executions, so a memo
  that leaked past a boundary would alias *different data* under the same
  id. The `positions_i32` scratch growth even bumps `pool_gen`, because the
  freed scratch pointer may be embedded in captured graphs.
- **Device memory is not host memory.** On GB10, plain `memcpy` of a device
  pointer SIGSEGVs (GPU_SAFETY rule 3); every D2H goes through
  `copy_to_host`/pinned staging, and `read_host` returns `None` so no
  caller can even attempt a borrowed device read.
- **Test with the fusion passes and both graph modes.** The decode fusions
  (FusedQKV/FusedFFN) are part of the reuse identity; the unfused path must
  run the same FusionPass to be comparable (graph rule 7), and graph replay
  A/B (`MINFER_NO_CUDA_GRAPH=1`) is the standard harness for anything that
  touches stream state — the 7d verification matrix runs replay-vs-direct
  bit-parity as its first gate.

## 4. Observe & verify

- **Startup banner** — `CUDA: using <device> (SM 12.1, ... MB, ... SMs)` +
  `CUDA: GPU acceleration enabled` (or `not available, using CPU fallback`)
  tells you in one line which device was picked and whether the graph will
  run on it at all. `--gpu N` pins the device; `MINFER_DISABLE_CUDA=1`
  forces CPU.
- **The dispatch labels** — `MINFER_MMQ_RAW_NB_DEBUG=1` prints *which* MMQ
  kernel variant is active per launch class, including *why* a fast path was
  skipped (`fallback!` vs the deliberate `exp=off` — the r53 lesson that a
  fallback-correct fast path needs a visible label, since parity tests
  cannot see which path ran).
- **Escape hatches, one per mechanism** (all default to the promoted path;
  each restores the pre-optimization behavior so A/Bs stay reproducible):
  `MINFER_MMQ=0` (f16 prefill GEMM), `MINFER_NO_PREFILL_GEMM=1` (legacy
  per-type kernels), `MINFER_NO_KQ_MMVQ=1` (K-quant decode back to f32
  kernels), `MINFER_NO_DECODE_A_FUSE=1` (standalone decode quantize),
  `MINFER_NO_W16CACHE=1` (per-call f16 scratch),
  `MINFER_MMQ_Q6K_EXP=0`/`MINFER_MMQ_Q4K_DSC=0` (drop the precomputed
  planes), `MINFER_NO_FUSE_QKV=1`/`MINFER_NO_FUSE_FFN=1` (decode fusions),
  `MINFER_CACHE_TYPE=f32|f16` (KV element type),
  `MINFER_NO_PINNED_READBACK=1` (pageable readback).
- **CUDA Graph state** — `MINFER_NO_CUDA_GRAPH=1` forces direct launches
  (the A/B lever for replay); `MINFER_NO_PREFILL_CAPTURE=1` disables
  prefill capture only. Note that `MINFER_TRACE`/viz capture and
  `MINFER_GRAPH_DUMP` node dumps *silently* disable replay while active
  (per-node host readbacks are illegal in a capture window) — and mode-2
  fused producers degrade to mode 1 when any dump/trace reader is on, so
  instrumented runs are not perf runs.
- **Device-side debugging** — `MINFER_CUDA_DEBUG` turns on per-node labeled
  syncs (`debug_sync`) that report launch/sync errors with a layer tag;
  each sync costs a full stream flush, so it is a debugging tool, not a
  profiling one.
- **Kernel selection in one command** — `MINFER_NO_CUDA_GRAPH=1
  MINFER_TIMING=1 ./target/release/minfer <model> "hi"` gives clean
  per-forward timing (doc 09's caliber), and `bench [-p N] [-n N]` measures
  prefill/decode tok/s the way the campaign's tables do. `nsys`/`ncu`
  profiles are what the optimization records cite per kernel.
- **Tests** (device-gated: they skip when no GPU answers, so CI without
  CUDA stays green) — `cargo test --features cuda cuda_` covers the whole
  backend: per-kernel parity (`cuda_elementwise_parity`,
  `cuda_matmul_parity`, `cuda_kquant_matmul_parity`, the per-type MMVQ
  parity tests), attention round trips (`cuda_attn_split_decode_parity`,
  f16-KV variants), the fusions (`cuda_fused_qkv_epilogue_bitwise`,
  `cuda_fused_ffn_parity`), embedding/gather
  (`cuda_embed_getrows_parity`), MMQ prefill bit-parity
  (`cuda_prefill_mmq_parity`, `cuda_q6k_dsc_dense_byte_exact`), and the
  graph machinery (`cuda_graph_replay_bit_parity`,
  `cuda_graph_recaptures_on_pool_gen_change`,
  `cuda_prefill_capture_bit_parity_pp16_pp300`,
  `cuda_capture_abort_on_error`). End-to-end: greedy output must equal the
  CPU path's greedy output at temp 0 — the Phase-7 acceptance gate for
  every supported model.

## 5. Cross-references

- [08 — The scheduler](08-scheduler-execute.md) — the split walk that calls
  `execute_node` and the replay hook this doc's §3.2.8 plugs into.
- [10 — CPU matmul kernels](10-cpu-matmul-kernels.md) — the scalar/AVX2
  origin of the quantize-and-integer-dot scheme MMQ lifts onto tensor cores;
  §2's block-layout vocabulary is reused here without redefinition.
- [11 — Attention, vec ops, and the KV cache](11-attention-vecops-kv.md) —
  the attention math the split-KV kernel parallelizes (§2.5 previews the GPU
  deltas this doc implements).
- [13 — The decode loop](13-decode-loop-graph-reuse.md) — why decode reuses
  the graph and its buffers, the precondition for CUDA Graph replay.
- [14 — The Metal backend](14-metal-backend.md) — the sibling backend:
  same `Backend` trait, same pool/free-list shape, same f16-KV policy; Metal
  dispatches one command buffer per split where CUDA captures one graph per
  split, and Metal's norm arm degrades silently where CUDA's returns `Err`.
- `docs/CUDA-BACKEND-PLAN.md` — the backend's design + implementation
  record: §3's llama.cpp reference map, §4 the design, §5 phases 7a–7e with
  per-item A/B numbers (7e②'s 3.1× decode, 7d's +18% replay).
- `docs/CUDA_OPTIMIZATION.md` + `docs/cuda_optimization_steps/` — the
  optimization campaign: §0's master history table (Phase 8 rows 8b–8q, the
  R-series, P5, the r28–r60 MMQ rounds, the D-series decode sessions), one
  standalone record per step; §1.1 the current perf/memory state.
- `docs/CUDA-TECH-PRIMER.md` — the technology primer behind §2.1's five
  words: warps, occupancy, `cp.async`, `mma`, CUDA Graphs, each with its
  campaign context.
- `docs/LLAMA-CPP-MMQ-ANALYSIS.md` — the reference MMQ kernel dissected
  (dispatch, tiling, numerics, SASS census) plus minfer's round-by-round
  contrast; the source for §2.3's claims.
- `docs/BUILD.md` — the build reference this doc's §3.2.1 summarizes:
  nvcc/host-compiler detection, `MINFER_CUDA_CCBIN`, arch coverage
  (sm_70…sm_121, CUDA 12.8-vs-13 Volta note), `cuda_static` linking.
- `docs/SUPPORT-MATRIX.md` — the per-quant support table incl. the CUDA
  notes (which path runs at which `nt`, the Q5_K `id % 32` gate, the F32
  caveats).
- `docs/GPU_SAFETY.md` §"CUDA" — the seven hard rules §3.4 paraphrases
  (capture windows vs syncs, launch-error policy, weight-registry
  ownership, runtime device limits).
- `docs/GLOSSARY.md` — the campaign glossary if a term (NB-BT, KDR, W_dsc,
  rpw...) from the records above needs its formula.

← [14 — The Metal backend](14-metal-backend.md) · [Index](./README.md) →
