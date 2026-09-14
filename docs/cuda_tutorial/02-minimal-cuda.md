# 02 · The minimal CUDA you actually need

> **Part**: Part 2 — the minimal language/API surface. **Prereq**: [01 · What kind of machine is a GPU](01-gpu-mental-model.md) — you know what a thread, warp (a group of 32 threads that execute in lockstep), block and grid are, and what a kernel launch does conceptually.
> **Code**: `src/cuda.rs`, `src/cuda_kernels.cu`, `build.rs` — every `file:line` was verified against the current tree at writing time (the function name is the stable address, the line number a convenience).

## 1. Background — where this sits

Chapter 01 gave you the execution model in the abstract: a kernel is a
function that runs once per thread, threads come in blocks, blocks come in a
grid, and the GPU schedules blocks onto its SMs (Streaming Multiprocessors —
the GPU's processor units, each running many warps concurrently). This chapter
turns that model into the concrete API surface minfer actually uses.

The surface is small. CUDA (NVIDIA's C/C++ extension plus its runtime library)
exposes hundreds of API calls, but minfer's whole device layer — all 6,225
lines of `src/cuda.rs` — is built on roughly twenty of them, one language
feature (`__global__` functions) and one launch syntax (`<<<...>>>`). The five
things you cannot read minfer's CUDA code without: kernel syntax and indexing
(§2.1); error checking and sync semantics (§2.2); device memory and the Rust
FFI layer (§2.3); streams and asynchronous execution (§2.4); and the build
pipeline (§2.5).

Scope note: the deep technique material — coalescing (when consecutive
threads access consecutive addresses so the hardware merges their loads into
one wide transaction), occupancy, tiling, the MMQ (Matrix-Multiply
Quantization) prefill path — is deliberately deferred to
`docs/CUDA-TECH-PRIMER.md` and Part 4; this chapter teaches just enough of
the language and runtime for those to read like prose.

Throughout we anchor on one real model: **Qwen2.5-0.5B, hidden size 896, FFN
(Feed-Forward Network) intermediate width 4864**. The walkthrough prints these
shapes from the real graph (`docs/inference_e2e_walkthrough/05-graph-builder-ir.md:184-188`
— `ffn_gate [896 -> 4864]`, `ffn_down [4864 -> 896]`) and from the GGUF
(`docs/inference_e2e_walkthrough/02-gguf-load.md:232` — `token_embd.weight`
`[896, 151936]`). Both numbers are small enough to do the arithmetic in your
head, and they are the exact shapes the elementwise kernels in
`src/cuda_kernels.cu` process on every decode step.

## 2. Principle — the concepts

### 2.1 Kernel syntax and thread indexing

**The three qualifiers.** CUDA adds three function qualifiers to C++; they
answer one question — *who can call this function, from where?*

- `__global__` — a **kernel**: a function callable **from the host** (host =
  the CPU and its memory, as opposed to the device = the GPU and its memory)
  that executes **on the device**, once per thread. Always returns `void`.
- `__device__` — callable **only from device code**. An ordinary inline
  function on the GPU: it creates no threads, it runs inside the calling
  thread.
- `__host__` — an ordinary CPU function (the default). Written explicitly only
  for functions that compile in *both* worlds — `__host__ __device__`.

`src/cuda_kernels.cu` uses exactly this split: `__global__` for the 89 kernels
(the count from `grep -c '__global__' src/cuda_kernels.cu`), plain
`__device__` helpers for shared math, and ordinary C++ for the host-side
launcher functions below.

**The launch syntax and the index formula.** A kernel launch is a function
call preceded by a `<<<...>>>` configuration clause, and inside the kernel the
single most important line in CUDA programming tells each thread *which
element it owns*:

```cuda
silu<<<grid, block, shared_mem_bytes, stream>>>(dx, dy, n);
//    └──┬──┘ └─┬─┘ └─────┬─────┘ └──┬──┘
//     blocks   threads   optional   optional
//     (dim3)   per block shared mem  stream

int i = blockIdx.x * blockDim.x + threadIdx.x;
//       └────┬───┘   └───┬───┘   └────┬────┘
//       which block  threads per   my lane
//       (grid coord) block        (inside the block)
```

`grid` and `block` are `dim3` values — three-dimensional sizes (x, y, z); a
plain integer fills x and y/z default to 1, so `block = 256` means "256
threads in a 1-D line". With `blockDim.x = 256`, block 0 owns `i ∈ [0, 256)`,
block 1 owns `[256, 512)`, and so on. Consecutive threads get consecutive `i`
— the property that makes memory access coalesced, and why 1-D elementwise
kernels are the fastest thing a GPU does. The last two launch arguments
default to 0 and the default stream (§2.4).

**Toy #2 — a SiLU kernel, with a CPU reference.** SiLU (Sigmoid Linear Unit,
the activation in every Qwen FFN: `silu(x) = x / (1 + exp(-x))`) is minfer's
most-launched elementwise op, so it is the right first kernel. (Toy #1,
vector add, lives in chapter 01; the toy index is appendix C of
[07](07-where-next.md).) The error checks below are not decoration — §2.2
explains the two *different* places an error can surface:

```cuda
// toy2_silu.cu — a SiLU elementwise kernel + CPU reference comparison.
// Toy #2 of the minfer CUDA tutorial (chapter 02). Verified with CUDA 13.0
// on GB10 (aarch64, sm_121): nvcc -arch=sm_121 -O2 toy2_silu.cu -o toy2 && ./toy2
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return 1; } } while (0)

__global__ void silu(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global 1-D index
    if (i >= n) return;                    // guard: grid is ceil-div, so the
    y[i] = x[i] / (1.0f + expf(-x[i]));   // last block is partly idle
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 4864;   // Qwen2.5-0.5B FFN width
    float *hx = new float[n], *hy = new float[n], *ref = new float[n];
    for (int i = 0; i < n; i++) hx[i] = -8.0f + 16.0f * (float)i / (n - 1);

    float *dx = nullptr, *dy = nullptr;
    CK(cudaMalloc(&dx, n * sizeof(float)));
    CK(cudaMalloc(&dy, n * sizeof(float)));
    CK(cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, grid = (n + block - 1) / block;   // ceil-div
    printf("n=%d block=%d grid=%d -> %d threads launched, %d idle\n",
           n, block, grid, grid * block, grid * block - n);

    silu<<<grid, block>>>(dx, dy, n);
    CK(cudaGetLastError());          // launch errors surface HERE (async!)
    CK(cudaDeviceSynchronize());     // run errors surface HERE
    CK(cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) ref[i] = hx[i] / (1.0f + expf(-hx[i]));
    float maxerr = 0.0f; int worst = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(hy[i] - ref[i]);
        if (e > maxerr) { maxerr = e; worst = i; }
    }
    printf("x=%.6f gpu=%.6f cpu=%.6f | maxerr=%.3e %s\n",
           hx[worst], hy[worst], ref[worst], maxerr, maxerr < 1e-6f ? "PASS" : "FAIL");
    CK(cudaFree(dx)); CK(cudaFree(dy));
    delete[] hx; delete[] hy; delete[] ref;
    return 0;
}
```

Compile and run on the GB10 (CUDA 13.0 at `/usr/local/cuda/bin/nvcc`, aarch64;
`-arch=sm_121` targets its compute capability 12.1 — §2.5). The toy takes an
optional size, so both grid regimes are observable:

```console
$ /usr/local/cuda/bin/nvcc -arch=sm_121 -O2 toy2_silu.cu -o toy2 && ./toy2
n=4864 block=256 grid=19 -> 4864 threads launched, 0 idle
x=2.679828 gpu=2.507852 cpu=2.507852 | maxerr=4.768e-07 PASS
$ ./toy2 896
n=896 block=256 grid=4 -> 1024 threads launched, 128 idle
x=2.815642 gpu=2.656601 cpu=2.656602 | maxerr=2.384e-07 PASS
```

Three lessons from the real output. First, **grid sizing is ceil-div, and the
guard is what makes it safe**: `(n + block - 1) / block` is the standard
integer ceiling division (every launcher in minfer uses it verbatim), and it
over-issues threads whenever `n` is not a multiple of 256 — for the hidden
size 896, `896/256 = 3.5`, so 4 blocks × 256 = 1024 threads start and **128
must do nothing**, which is exactly what `if (i >= n) return;` is for.
Without the guard those threads read and write past the buffer end, and
out-of-bounds writes on a GPU corrupt *other allocations' bytes silently* or
surface as an async error later (§2.2). Second, **the CPU reference is the
whole quality model**: `maxerr ≈ 5e-07` is one rounding difference on `expf`
— the GPU's `expf` and glibc's `expf` are different implementations of the
same function — and minfer's test gates use the same shape, comparing against
the CPU reference and reserving *bit-identical* for comparisons against the
same code path (§3). Third, note the ratio: the kernel is 5 lines; launch,
copy back, reference loop and checks dwarf it.

**The 2-D formula — minfer's real `add_bias_f32`.** A bias add works on
token-major activations `[rows][d]`: row `t` is a token, column `i` a
hidden-dimension index, and every row adds the *same* `b[i]`. minfer's kernel
(`src/cuda_kernels.cu:2391-2399`) maps the 2-D shape directly onto the grid:

```cuda
// src/cuda_kernels.cu:2391-2399
__global__ void add_bias_f32(
    float* __restrict__ y,
    const float* __restrict__ b,
    int d
) {
    int t = blockIdx.x, i = threadIdx.x + blockIdx.y * blockDim.x;
    if (i >= d) return;
    y[t * d + i] += b[i];
}
```

Line by line: `t = blockIdx.x` is the token — the launcher makes the grid's x
dimension the row count, one block per token. `i =
threadIdx.x + blockIdx.y * blockDim.x` is the column within the row — the 1-D
formula with the *block* index taken from the **y** dimension. The lesson:
the x/y/z grid dimensions are free coordinates; map whichever data axis is
longest onto x (the one consecutive threads walk — the coalescing-friendly
one) and use y/z for the slower axes. `if (i >= d) return;` is the toy's
guard, now on the column axis; `d` needs no guard because the grid's x
dimension *is* the row count. `__restrict__` is a promise that this pointer
is the only way the function accesses that memory — an optimization hint
that frees the compiler from proving non-aliasing (two pointers referring to
the same memory, which would forbid reordering loads and stores). The bias
vector itself is 896 × 4 B = 3.5 KB, re-read by every token's block — small
enough to stay hot in L2 cache across blocks.

The launcher builds that 2-D grid (`src/cuda_kernels.cu:3872-3878`): 64
threads in x per block, `dim3 grid(n, (d + 63) / 64, 1)` — the same ceil-div
over the column axis, 896 columns → 14 blocks in y. For one decode token
(`n = 1`) the grid is 1 × 14 blocks of 64 threads — 896 threads for 896
elements, one thread per element. Note the kernel takes no row count: the
grid *encodes* it, which makes the launcher responsible for the launch
geometry matching the kernel's expectation. That is a contract, and breaking
it is exactly the bug minfer documents on the Rust wrapper
(`src/cuda.rs:4201-4203`): *"`rows` is the ROW COUNT (token count) — the
kernel grid maps one block row per token, so passing the total element count
writes out of bounds."* Passing `rows * d` would launch `rows*d` block-rows —
a grid that reads and writes far past the buffer. On CUDA this does not
segfault the host; it either corrupts unrelated device memory or surfaces at
the *next* synchronization point (§2.2). The backend repeats the warning at
its only call site (`src/graph/cuda_backend.rs:987-990`, which passes `nt`,
the token count) — leave this kind of comment on every launch geometry you
write.

**minfer's real elementwise family, side by side with the toy.** The kernel
inventory (`grep -n '__global__' src/cuda_kernels.cu`) lists 89 kernels; the
elementwise ones are the plainest and are all structured like your toy
(global 1-D index, guard, one memory access per thread):

| Kernel (`src/cuda_kernels.cu`) | Lines | Body | Used for |
|---|---|---|---|
| `add_f32` | 2403-2412 | `z[i] = x[i] + y[i]` | residual add (`Op::Add`) |
| `mul_f32` | 2416-2425 | `z[i] = x[i] * y[i]` | elementwise multiply (`Op::Mul`) |
| `silu_f32` | 2429-2434 | `y[i] = v/(1+expf(-v))` in-place | FFN activation (`Op::Silu`) |
| `swiglu_f32` | 2471-2481 | `dst = silu(gate) * up` | unfused SwiGLU (`Op::SwiGLU`) |
| `swiglu_f32_off` | 2441-2446 | in-place, `up` at offset `off` | fused-FFN path (decode) |
| `add_bias_f32` | 2391-2399 | `y[t*d+i] += b[i]`, 2-D | attention/FFN output bias |
| `f32_bits_to_i32` | 2489-2497 | bit-reinterpret f32 → int32 | graph I32-input convention |

`add_f32` (`src/cuda_kernels.cu:2403-2412`) is your toy's skeleton exactly —
index formula, guard, `z[tid] = x[tid] + y[tid]` — with two idiom changes:
the parameter is named `tid` (thread id) and every pointer is
`const ... __restrict__`. `silu_f32` (`src/cuda_kernels.cu:2429-2434`) is the
toy's kernel verbatim except it is *in-place* — one buffer, read and written
through the same pointer — legal because each element is touched by exactly
one thread. In-place-ness is a graph-level decision in minfer (the alias rule:
a node may alias its input only when it is the sole consumer and runs on the
same backend — §3 shows the D2D (device-to-device) copy the backend stages
when the allocator did *not* alias it). Both launch through their launchers'
ceil-div grid of 256-thread blocks (`launch_add_f32`
`src/cuda_kernels.cu:3880-3887`, `launch_silu_f32` `:3898-3903`), and every
launcher in the file ends with `, stream)`: the entry-point family is uniform
so the Rust side can target any stream uniformly. Why that matters is §2.4.

### 2.2 Error checking and synchronization semantics

The toy's last checks before trusting its output were two *different* calls —
`CK(cudaGetLastError())` right after the launch, then
`CK(cudaDeviceSynchronize())` before the D2H copy — and the difference is the
most confusing thing about CUDA error handling.
**A kernel launch is asynchronous**: `silu<<<...>>>` does not run your kernel
— it *enqueues* it into a queue (a stream, §2.4) and returns immediately,
typically in a few microseconds. The GPU pulls work off the queue when it gets
to it; the host keeps running straight past the launch. So:

- Errors detectable *at enqueue time* — invalid launch configuration (0
  threads per block), a kernel not compiled into the binary, too many threads
  per block for this GPU — come back from the launch itself (and from
  `cudaGetLastError()` right after it). Cheap, synchronous checks.
- Errors that happen *while the kernel executes* — out-of-bounds write,
  misaligned address, illegal instruction — can only be discovered by the GPU
  at run time, after the host has moved on; they sit in a per-thread error
  slot until a later CUDA call picks them up — conventionally a
  synchronization point: `cudaDeviceSynchronize()` (wait until **every**
  stream has finished all its work) or `cudaStreamSynchronize(s)` (wait until
  stream `s` is drained).

That is the whole "a kernel's error surfaces later" phenomenon: the queue
means the host-side call that would report the error may run milliseconds
before the GPU even starts the faulty kernel. If you never synchronize, you
may never see the error — the process can exit with silently corrupted
results. One wrinkle: an execution error is *sticky*. Once the context (the
per-process GPU state that all your allocations and streams live in) records
a fault, every subsequent CUDA call returns the same error until the context
is re-created — so a misdiagnosed "error at line 4000" is usually the first
*sync after* the real culprit. Bisect by syncing after each launch suspicion,
which is what minfer's debug sync (below) is for.

**How minfer wraps this.** minfer checks at *sync points*, not per launch,
and the wrapper is 10 lines (`src/cuda.rs:2303-2313`):

```rust
// src/cuda.rs:2303-2313
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

`cudaGetLastError()` picks up anything sticky from launches since the last
check (launch-time errors); `cudaStreamSynchronize(self.stream())` blocks
until the one shared stream has drained — which both (a) makes the host wait
so results are trustworthy and (b) transfers any *execution* error into the
return value, checked and printed. Note what it does **not** do: no panic, no
abort, no `Result`. `CudaState::sync` is a *drain point*; the safety contract
lives one level up, in the backend, where invariant violations become `Err`.
There is also a per-node debug variant, `debug_sync(il, label)`
(`src/cuda.rs:2318`, gated behind `MINFER_CUDA_DEBUG=1`), printing the layer
index and label with any launch or sync error — the bisection tool.

Why "at sync points, not per launch"? The graph path launches ~100+ kernels
per decode step; a `cudaGetLastError()` after each is cheap, but a *sync*
after each would serialize the pipeline. The compromise — one drain point per
split, plus sticky-error pickup — is `docs/GPU_SAFETY.md` rule 4
(`docs/GPU_SAFETY.md:208`): *"Launch errors are checked at sync points, not
per launch: `CudaState::sync()` polls `cudaGetLastError` +
`cudaStreamSynchronize` and reports both."*

**The GPU safety contract.** `docs/GPU_SAFETY.md` (CUDA section,
`docs/GPU_SAFETY.md:201-211`) translates the project-wide hard rules to the
CUDA backend; read the whole doc before touching this code. In four lines:
kernel-invariant violations **return `Err` from `execute_node`** — never a
silent CPU fallback; a guard failure aborts with the node's name (`:205`).
**No sync inside an active capture window** — a stray `cudaStreamSynchronize`
while CUDA records a graph corrupts the capture (the 7e② "faster but wrong"
incident; §2.4 builds on this, `:206`). **Device memory is not host-readable
via plain memcpy on GB10** — dereferencing a device pointer from the host
segfaults; all D2H (device-to-host) traffic goes through `cudaMemcpy`
(`:207`). Launch errors checked at sync points; same-stream ordering is the
correctness contract for async fills (`:208-209`). Rule 1's Rust shape is on
the elementwise dispatch arm: the backend checks the kernel's preconditions
*before* launching and returns a formatted `Err` naming the node
(`src/graph/cuda_backend.rs:478-489` — `Op::Add` rejects input-size
mismatches with `Err(format!("cuda: {}: add input size mismatch", node.name))`
before calling `add_f32`). That is the boundary between the two error worlds:
*host-visible* checks are ordinary Rust `Result`s raised before launch;
*device-side* faults (an OOB inside a kernel) are the sticky CUDA errors
picked up at the next `sync()` — a well-designed CUDA codebase is explicit
about which world each check belongs to.

### 2.3 Device memory management

**The three calls.** Device memory (the GPU's own DRAM — gigabytes of it,
*not* addressable by normal CPU pointers) is managed by three runtime calls:
`cudaMalloc(void** ptr, size_t bytes)` allocates on the device and returns a
*device pointer* (valid as a kernel argument, not valid to dereference from
the host — the rule just above); `cudaFree(void* ptr)` frees it; and
`cudaMemcpy(dst, src, count, kind)` copies bytes, where `kind` makes one
function serve all three directions: H2D (host→device upload),
`CUDA_MEMCPY_HOST_TO_DEVICE = 1`; D2H (download, `= 2`); D2D (device-internal,
`= 3`) — the kind tells the driver which address spaces the pointers live in,
and minfer defines the constants by hand (`src/cuda.rs:74-77`). A blocking
host↔device `cudaMemcpy` is *synchronous*: it does not return until the
bytes have moved. For decode-speed code that is a problem (waiting for the
GPU), which is why the async variants (`cudaMemcpyAsync` on a stream, plus
**pinned memory**) exist — but the synchronous calls are the right default
for one-shot work like weight loading, and that is how minfer uses them.

**Pinned vs pageable, in one paragraph.** Normal host memory ("pageable") can
be swapped or moved by the OS, so the DMA (Direct Memory Access — the copy
engine that transfers bytes without CPU involvement) hardware cannot be handed
a stable physical address; the driver first copies your pageable buffer into
an internal *pinned* staging area (memory locked against paging, allocated
with `cudaHostAlloc`), then DMAs from there. Fine for one-shot uploads; but
for repeated async transfers you allocate your own pinned buffers and copy
into them yourself — then `cudaMemcpyAsync` can be *enqueued* on a stream and
return before the bytes move, overlapping the transfer with compute. The
cost: pinned memory is a scarce OS resource (over-allocating it degrades the
whole machine, not just your process). minfer keeps small, ring-shaped pinned
pools only where asynchrony pays — the 8-slot × 2 MB staging ring for input
fills (`write_input_async`, `src/cuda.rs:2146-2214`), a grow-on-demand D2H
readback buffer for per-step logits (`PinnedBuf`, `src/cuda.rs:954`), and the
capture staging pool (`CaptureStaging`, `src/cuda.rs:974`) — with a
synchronous pageable fallback whenever pinned allocation fails.
`docs/GPU_SAFETY.md:209` states the rule that makes this safe: *same-stream
ordering* — the async fill is safe because every consumer kernel is enqueued
later on the *same* stream.

**How the Rust side wraps the C API (FFI).** minfer links the CUDA runtime
directly — no `cuda` crate, no bindgen. It *declares* the C functions it
needs in an `extern "C"` block (FFI, Foreign Function Interface — the
mechanism by which Rust calls C-ABI functions; `extern "C"` selects the C
calling convention so Rust and C agree on how arguments are passed) at the
top of `src/cuda.rs:30-72`:

```rust
// src/cuda.rs:30-53 (excerpt — the full block runs to line 72, incl.
// cudaMemcpyAsync, cudaStreamSynchronize, device queries, CUDA Graph APIs)
extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flag: std::ffi::c_int) -> *mut std::ffi::c_void;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaGetLastError() -> i32;
    // cudaMemcpy(dst, src, count, kind), cudaHostAlloc, cudaStreamCreate, ...
}
```

How to read such a block — the skill of *reading* an FFI layer:

- **Types are the contract.** CUDA's `cudaError_t cudaMalloc(void** devPtr,
  size_t size)` becomes `fn cudaMalloc(ptr: *mut *mut c_void, size: usize) ->
  i32`. The C enum `cudaError_t` is an `i32` in Rust; the double pointer is
  Rust's `*mut *mut c_void` — the callee writes the allocated address through
  it. `c_void` everywhere: neither side cares what lives behind the pointer.
- **Out-parameters instead of return values.** C APIs that "return" data do
  it by writing through a pointer argument (`cudaMalloc`,
  `cudaGetDeviceCount`, `cudaStreamCreate`); the wrapper functions in the
  rest of the file perform exactly the translation into Rust-style `Result`s.
- **Raw pointers are not `Send`/`Sync`.** Rust assumes a raw pointer may alias
  anything, so types containing them are not thread-safe by default. minfer's
  `CudaPtr` newtype (`src/cuda.rs:17-21`) is the deliberate, documented
  exception — `unsafe impl Send/Sync` is an assertion *you* must defend:
  sound here because CUDA device allocations are process-global resources
  usable from any thread, and every access is funneled through `Mutex`es.
- **Where `unsafe` lives.** Every call into these functions is `unsafe` (the
  compiler cannot verify C's rules). The design rule to imitate: keep
  `unsafe` *thin* — the `extern` block and the immediate calls inside small,
  named wrapper functions, so the safe surface (`CudaState`'s methods) never
  leaks raw pointers or error codes.

**RAII and the ownership models.** RAII (Resource Acquisition Is
Initialization — the idiom where acquiring a resource in a constructor and
releasing it in a destructor ties the resource's lifetime to an object's, so
drops and panics cannot leak it) is how minfer keeps the *host-side* CUDA
resources leak-proof: every pinned allocation has a `Drop` impl calling
`cudaFreeHost` (`PinnedPool` `src/cuda.rs:936-947`, `PinnedBuf` `:958-961`,
`CaptureStaging` `:1064-1071`). Device memory is the interesting
half-exception. A *weight* buffer is not RAII-managed at all: weights live
for the whole process, keyed by name in the registry
(`weights: Mutex<HashMap<String, (CudaPtr, usize)>>`, `src/cuda.rs:1150`), and
the registry's replace rule deliberately leaks the stale buffer because a
live captured CUDA Graph may still reference the old pointer
(`src/cuda.rs:1624-1628`) — leaking *by decision, with a bound and a comment*
is legitimate, because the alternative (freeing memory a captured graph still
points at) is a use-after-free the GPU hits mid-replay. Scratch buffers, in
contrast, are pooled: the backend frees every pool buffer in its `Drop`
(`src/graph/cuda_backend.rs:358-380`), and `CudaState` offers the raw pair
`cuda_malloc`/`cuda_free` (`src/cuda.rs:2107-2124`) for pool use.

**One complete ownership path, walked.** The simplest non-toy path in the
file: *register a weight → use it → (never) free it*.

1. **Alloc.** `register_weight(name, data)` (`src/cuda.rs:1610-1665`) first
   checks the registry — same name *and* same byte size ⇒ the device copy
   exists, reuse it and return (`:1614-1623`). Otherwise it calls `cudaMalloc`
   for `data.len()` bytes (`:1631-1640`); on OOM (out of memory) it prints
   the *byte count and tensor name* and returns without registering — the
   loader's all-weights-registered gate then refuses to enable the GPU
   backend loudly (a `docs/GPU_SAFETY.md` invariant). Nothing silently
   half-works.
2. **Copy (H2D).** Still inside `register_weight`, a blocking
   `cudaMemcpyHostToDevice` uploads the GGUF (the on-disk model format minfer
   parses) bytes verbatim — quantized weights are uploaded *raw* and
   dequantized on the device by the `dequant_*_f16` kernels at first use (a
   series fact; chapter 03 reads those kernels). On copy failure the freshly
   allocated buffer is freed *before* returning (`:1641-1655`) — the manual
   version of RAII: the error path releases what the success path acquired.
3. **Register.** The pointer is wrapped and inserted
   (`:1661-1664`):
   `self.weights.lock().unwrap().insert(name.to_string(), (CudaPtr(ptr), data.len()))`.
   From here the *only* way to reach the buffer is `get_weight_ptr(name)`
   (`src/cuda.rs:2043`) — the registry is the single owner, and dispatchers
   resolve by name at execution time.
4. **Use.** A decode step later, dispatch resolves the name to a device
   pointer and passes it to a launcher. The Rust side of `add_f32`
   (`src/cuda.rs:4182-4199`) is a 17-line translation unit from safe Rust to
   the C launcher: fetch `self.stream()`, then one `unsafe` call
   `launch_add_f32(x as *const f32, y as *const f32, z as *mut f32, n as i32,
   stream)`. The `extern` declaration sits at `src/cuda.rs:227-233` — these
   launcher symbols are provided by `libcuda_kernels.a`, the archive
   `build.rs` produces from `src/cuda_kernels.cu` (§2.5). The `usize → i32`
   narrowing and the `c_void → *const f32` casts are the FFI layer's whole
   job — the *kernel* wants `const float*` and `int`, and this is where the
   host types are made to match. Note what is *absent*: no error check —
   launches are enqueued asynchronously and checked at the next `sync()`
   (§2.2).
5. **Drop.** For weights: never — process-lifetime by design (the registry
   owns them until exit; the OS reclaims the context). For pool scratch:
   `CudaBackend::drop` frees every pool buffer with `cudaFree`, which
   implicitly synchronizes the device — that is why the `Drop` first takes
   the stream lock (`src/graph/cuda_backend.rs:358-367`). For the
   grow-on-demand scratch slots there is a middle pattern, `get_or_grow`
   (`src/cuda.rs:2081-2104`): if the slot's allocation is too small, *free
   the old buffer then allocate the new one*, only under the slot's own
   `Mutex` so two graph executions cannot grow the same slot concurrently.

That is the complete path: alloc → upload → registry (single owner) →
name-resolve at dispatch → lifetime-by-design. Every other part of
`src/cuda.rs` — the KV regions, the f16 cache, the MMQ scratch planes — is a
variation on this skeleton.

### 2.4 Streams and asynchronous execution

**What a stream is.** A `cudaStream_t` is a **queue of GPU work** — kernel
launches, memory copies, event records — that the GPU executes *in enqueue
order*. Every asynchronous operation you have met takes a stream argument:
`kernel<<<grid, block, 0, stream>>>`, `cudaMemcpyAsync(..., stream)`. The
stream is what makes CUDA asynchronous at all: launching into a stream
returns immediately, and the GPU drains the queue at its own pace. The
semantics you build everything on: **within a stream, total order** — item N
starts only after item N-1 finishes; that is the correctness backbone, and
minfer's "same-stream ordering" rule (`docs/GPU_SAFETY.md:209`) is just this
sentence applied (an async H2D fill is safe because the kernel that reads the
buffer is enqueued *later on the same stream*). **Across streams, no order
and potential parallelism** — work in stream A and stream B may run
concurrently (if resources allow). And the **default stream** (stream `0`,
used whenever you omit the argument) is special: the *legacy* default stream
synchronizes with all other (blocking) streams — a serialization belt, not
just "stream number zero".

**CUDA events.** A `cudaEvent_t` is a marker you record *into* a stream; it
completes when every item before it in that stream has finished. Synchronize
on one event to wait for a *part* of a pipeline;
`cudaEventElapsedTime` between two events measures GPU time between the
markers (on the device's own clock, immune to host scheduling noise) — how
every number in Part 4 was measured.

**Toy #3 — serialization vs overlap, measured.** Do two kernels *actually*
run in parallel on two non-default streams, and does the default stream really
serialize them? The toy launches the same busy-spin kernel twice — first both
into the default stream, then one each into two streams — timing each phase
with events (Toy #3 of the series; verified with CUDA 13.0 on GB10, sm_121):

```cuda
// toy3_streams.cu — Toy #3 of the minfer CUDA tutorial (chapter 02).
// Two identical "busy" kernels: launched back-to-back on the default stream
// they serialize; launched on two non-default streams they overlap.
// Timed with cudaEvents. Verified with CUDA 13.0 on GB10 (sm_121):
//   nvcc -arch=sm_121 -O2 toy3_streams.cu -o toy3 && ./toy3
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return 1; } } while (0)

// Busy kernel: one small block (fits beside other work), spins ~`cycles`
// clocks, then writes one float so the work is not optimized away.
__global__ void busy(float* out, long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) { }
    if (threadIdx.x == 0) out[blockIdx.x] = (float)(clock64() - start);
}

int main() {
    float* d = nullptr;
    CK(cudaMalloc(&d, 64 * sizeof(float)));
    cudaEvent_t beg, end;
    CK(cudaEventCreate(&beg));
    CK(cudaEventCreate(&end));
    cudaStream_t s1, s2;
    CK(cudaStreamCreate(&s1));
    CK(cudaStreamCreate(&s2));
    const long long cycles = 200000000LL;   // tune so one kernel ~ tens of ms
    float ms1 = 0.0f, ms2 = 0.0f;

    // Phase 1: both kernels on the DEFAULT stream (0) -> serialized.
    CK(cudaEventRecord(beg));
    busy<<<1, 32>>>(d, cycles);
    busy<<<1, 32>>>(d, cycles);
    CK(cudaEventRecord(end));
    CK(cudaEventSynchronize(end));
    CK(cudaEventElapsedTime(&ms1, beg, end));

    // Phase 2: one kernel per non-default stream -> overlap.
    CK(cudaEventRecord(beg));
    busy<<<1, 32, 0, s1>>>(d, cycles);
    busy<<<1, 32, 0, s2>>>(d, cycles);
    CK(cudaEventRecord(end));
    CK(cudaEventSynchronize(end));
    CK(cudaEventElapsedTime(&ms2, beg, end));

    printf("default stream : %.2f ms (kernel2 waits for kernel1)\n", ms1);
    printf("two streams    : %.2f ms (kernels run concurrently)\n", ms2);
    printf("overlap speedup: %.2fx\n", ms1 / ms2);
    CK(cudaStreamDestroy(s1)); CK(cudaStreamDestroy(s2));
    CK(cudaEventDestroy(beg)); CK(cudaEventDestroy(end));
    CK(cudaFree(d));
    return 0;
}
```

Observed on the GB10:

```console
$ /usr/local/cuda/bin/nvcc -arch=sm_121 -O2 toy3_streams.cu -o toy3 && ./toy3
default stream : 159.26 ms (kernel2 waits for kernel1)
two streams    : 79.58 ms (kernels run concurrently)
overlap speedup: 2.00x
```

Read the arithmetic, not just the speedup: one busy kernel is ~80 ms, so the
serialized pair costs ~160 ms and the overlapped pair ~80 ms — exactly 2.00×,
because the two one-block kernels fit on the GPU simultaneously. The same
experiment with big grids can show *less* overlap (two kernels that each fill
every SM cannot physically run side by side — streams give the GPU
*permission* to overlap, not extra hardware). And notice the timing pattern:
`cudaEventRecord(end)` is *enqueued*, the real wait is
`cudaEventSynchronize(end)` — the event version of the §2.2 sync story.

**minfer's actual stream usage.** The surprise: for all that machinery, minfer
runs **one** non-default stream, created once at device init
(`src/cuda.rs:1482-1487`) and fetched by every wrapper via `stream()`
(`src/cuda.rs:2074-2076`, a `Mutex<CudaPtr>` deref). Why one stream, when
streams exist for overlap? First, the workload is a dependency chain — a
decode step is a strict sequence (norm → matmul → rope → attention → … →
lm_head) with nothing to overlap *within* it. Second, the real per-step
overhead was launches, not gaps between kernels — ~100+ launches per decode
step, each a few µs of host time — so minfer's answer was not streams but
**CUDA Graph capture/replay**: record the whole step's launches once, then
replay the graph as a single launch. Capture is *per-stream* (only work
enqueued on the capturing stream is recorded), which is why the capture
window takes the process-wide `stream_lock` (`src/cuda.rs:2299-2301`): any
other backend's stream work must block rather than be recorded into the graph
(`src/graph/cuda_backend.rs:237-240`). The state machine lives in
`graph_replay_step` (`src/graph/cuda_backend.rs:168-251`): executions 1–2 of
a split run as plain launches (warmup — llama.cpp's protocol), the 3rd opens
the capture window, `synchronize()` closes it (instantiate + launch once +
cache, `src/graph/cuda_backend.rs:268-279`), and every later execution is a
single `graph_launch_exec` (`src/graph/cuda_backend.rs:212-214`). The
backend's `synchronize` (`src/graph/cuda_backend.rs:1428-1440`) is the
split-boundary drain point from §2.2: take the stream guard, clear the
per-execution memos, then `close_capture_or_sync`. Replay is gated by
`MINFER_NO_CUDA_GRAPH=1` (falls back to per-kernel launches — §5 makes the
launch stream visible in `nsys`). The mental model: **streams are the
substrate** (one stream, strict order, checked at drain points), and **CUDA
Graphs are the optimization on top** (same stream semantics, N launches
folded into 1). Chapter 06 covers the graph machinery; here you only need
the stream vocabulary to read it.

### 2.5 Build and toolchain

**Compiling toys by hand.** nvcc is NVIDIA's compiler driver — it runs the
host C++ compiler on the host parts and its own `cicc`/`ptxas` on the device
parts, then packs both into one object file. On this machine it is **not** on
the default PATH; call it with the full path (this is CUDA **13.0**):

```console
$ /usr/local/cuda/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:57:39_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
```

The one flag you must think about is the GPU architecture: `-arch=sm_NN`
generates SASS (the GPU's native machine code — "Streamed ASSembler") for
compute capability NN, where the digits come from the device itself. On the
GB10 (aarch64): `nvidia-smi --query-gpu=compute_cap` reports **12.1**, driver
580.173.02 — so the flag used for every toy in this tutorial is
**`-arch=sm_121`**, which CUDA 13.0 accepts (verify with
`nvcc --list-gpu-arch`). `-arch=native` (ask the installed driver) also works
on this machine and compiles both toys identically — either is fine; the toys
pin `sm_121` so the command is reproducible on paper. A binary built for a
*newer* arch than your GPU will not load (no SASS match); one built for an
*older* arch only runs if the binary embedded PTX (the intermediate assembly
the driver can JIT — Just-In-Time compile — for a newer GPU): exactly the
compat layer `build.rs` sets up below.

**How the repo does it.** `cargo build --release --features cuda` never calls
nvcc from your shell; `build.rs` (the Cargo build script that runs before
compilation) does the whole pipeline:

1. **Opt-in, and required once opted in.** The CUDA section returns early
   unless `CARGO_FEATURE_CUDA` is set (`build.rs:177-179`) — plain builds
   never touch nvcc. But once the feature is requested, CUDA is *required*:
   `src/cuda.rs` declares the `launch_*` symbols that only the kernels
   archive provides, so every failure below `panic!`s with an actionable
   message (`build.rs:181-187`).
2. **Find nvcc and the toolkit root.** `find_nvcc()` (`build.rs:378-401`)
   probes `CUDA_HOME`/`CUDA_PATH`, then `which nvcc`, resolving to an
   absolute path either way; `find_cuda_home()` (`build.rs:403-422`) derives
   the root for `-I{home}/include`.
3. **Pin the host compiler (ccbin).** nvcc uses the first `cc`/`g++` on PATH
   as its host compiler and *hard-fails* when that GCC is newer than the
   toolkit supports (CUDA 13 rejects GCC 15 — the error surfaces confusingly
   inside `<cmath>`). `detect_host_compiler()` (`build.rs:480-519`) probes
   nvcc's default first, then `g++-15 … g++-11, g++, clang++`; the winner is
   passed as `-ccbin` (`build.rs:201-220`, `:255-258`), and
   `MINFER_CUDA_CCBIN` overrides the probe.
4. **Probe the architectures.** `detect_archs()` (`build.rs:532-559`)
   compiles a one-line dummy kernel for every candidate from `sm_70` to
   `sm_121` (`build.rs:533-535`) and keeps the ones this nvcc accepts —
   candidates newer than the toolkit simply fail their probe and are skipped,
   so one list works on every CUDA version. (The floor is sm_70, not Pascal:
   the prefill GEMM uses WMMA tensor-core intrinsics that require Volta+,
   `build.rs:525-531`.)
5. **Compile once, embed many targets.** The single `nvcc` invocation
   (`build.rs:242-286`) carries `-O3 -fPIC` plus one
   `-gencode arch=compute_NN,code=sm_NN` per detected arch — SASS for every
   GPU class — *plus* two kinds of PTX (`build.rs:263-278`): a backward
   `compute_70/72` (so an older card like a V100 can JIT forward) and a
   forward `compute_{highest}` (so a GPU newer than the newest SASS can JIT).
   The result is the portable fat binary: every probed arch as native SASS,
   plus PTX from Volta up that any future GPU can JIT.

6. **Archive and link into the Rust binary.** The object is packed into
   `libcuda_kernels.a` with `ar rcs` (`build.rs:288-295`), and two cargo
   directives hand it to the linker (`build.rs:297-298`):
   `cargo:rustc-link-search=native={out_dir}` +
   `cargo:rustc-link-lib=static=cuda_kernels`. The `launch_*` symbols the
   `extern "C"` block of §2.3 declares are resolved against this archive —
   the whole seam between the two languages.
7. **cudart: static or shared.** The CUDA runtime is linked according to the
   `cuda_static` feature (`build.rs:327-360`): `cuda_static` links
   `libcudart_static.a` (plus `dl`/`pthread`) so the binary needs only the
   NVIDIA driver at run time; the default links `libcudart.so` and bakes an
   rpath — *only* when the toolkit dir is not a system dir, to avoid
   shadowing a nix-provided glibc (`build.rs:340-359`). The lib directory is
   probed rather than assumed (`build.rs:433-446`), because distro packages
   install cudart into the multiarch dir. What is *never* linked is the
   driver library `libcuda.so.1` — it is dlopen'd lazily at run time
   (`preload_driver`, `src/cuda.rs:1412`), the same lazy-loading trick as the
   `dlopen` declaration at the top of the `extern` block.

For the knobs this section skipped (cross-compiling, `MINFER_CUDA_CCBIN` in
full, distro-vs-toolkit layout quirks), `docs/BUILD.md` is the reference — 73
lines, worth reading once before your first `--features cuda` build.

## 3. In minfer's code — one op, end to end

Now read one real operation through *all* layers at once: `Op::Silu` on a
decode step of Qwen2.5-0.5B (`n = 4864`, one token). This is the path every
CUDA node takes; the layered picture first (the one diagram of this chapter):

```text
 scheduler.rs  : execute_node(node, backend)          (pure Rust, safe)
     │
     ▼
 cuda_backend.rs:506-513   Op::Silu arm               (guards → Err or launch)
     │   in_bufs[0] != out_buf?  → copy_d2d (D2D stage)
     ▼
 cuda.rs:4242-4247         CudaState::silu_f32        (thin unsafe wrapper)
     │   self.stream() = the one shared cudaStream_t
     ▼
 cuda.rs:241               extern "C" launch_silu_f32 (FFI declaration)
     ▼
 cuda_kernels.cu:3898-3903 launch_silu_f32            (grid = ceil-div, <<<>>>)
     ▼
 cuda_kernels.cu:2429-2434 __global__ silu_f32         (index → guard → math)
     ▼
 [ GPU: 19 blocks × 256 threads, enqueued on the stream, drains async ]
     …
 cuda.rs:2303-2313         CudaState::sync()          (sticky errors + drain,
                                                       at the split boundary)
```

The scheduler calls `execute_node` on whichever backend was assigned at build
time (assignment is decided *before* execution — the
never-silently-fallback rule from `docs/GPU_SAFETY.md`). The CUDA backend's
`Op::Silu` arm (`src/graph/cuda_backend.rs:504-513`) is seven lines:

```rust
// src/graph/cuda_backend.rs:504-513
// In-place op (alias rule, graph rules §5): stage via D2D copy when
// the allocator did not alias the input, then run on the output.
Op::Silu => {
    if in_bufs[0] != out_buf {
        self.copy_d2d(in_bufs[0], out_buf)?;
    }
    self.state
        .silu_f32(self.ptr_of(out_buf)?, self.elems(out_buf));
    Ok(())
}
```

Why the D2D copy? `silu_f32` is an *in-place* kernel — it reads and writes
the same buffer. The compute graph permits a node to alias its input only
under the alias rule (sole consumer + same backend); when the liveness
allocator did *not* alias the two buffers (they are distinct pool slots), the
backend stages a device-to-device copy first so the in-place kernel can never
write a buffer some other node still needs. `copy_d2d`
(`src/graph/cuda_backend.rs:1193-1203`) resolves both pool slots to device
pointers, refuses on a byte-size mismatch (`Err` — the §2.2 contract), and
enqueues `cudaMemcpyDeviceToDevice` on the shared stream. Two GPU operations
(copy + kernel) for the price of one node, both asynchronous, both ordered by
the stream.

Then the descent from §2.3 step 4: `CudaState::silu_f32`
(`src/cuda.rs:4242-4247`) fetches the shared stream and calls the extern
launcher (declared at `src/cuda.rs:241`); the C launcher computes
`grid = (4864 + 255) / 256 = 19` and enqueues
`silu_f32<<<19, 256, 0, stream>>>`; the kernel gives each of the 4,864
threads exactly one element — 19 blocks × 256 threads, zero idle (the
4864 = 19 × 256 exact fit from Toy #2). The launch returns in a few
microseconds while the kernel may not even have started. Finally the CPU
counterpart — this tutorial's pattern is kernel → CPU → why the GPU version
looks the way it does (chapter 03 does this line by line for the whole
elementwise family). The same op on CPU is `vec_silu_f32`
(`src/vec_ops.rs:155-173`): signature
`pub fn vec_silu_f32(n: usize, y: &mut [f32], x: &[f32])`, an x86_64
AVX2+FMA arm detected at run time (`src/vec_ops.rs:161-167`, 8 lanes per
step), and the scalar fallback `y[i] = x[i] / (1.0 + (-x[i]).exp())` — the
same formula as the kernel, one explicit loop index instead of 4,864
materialized threads.

Compare the two and the design falls out:

- **Same formula, same one-element-per-iteration shape** — the GPU kernel is
  the scalar loop turned inside out: the CPU *iterates* `i`, the GPU
  *materializes* `i` as a thread. Everything the CUDA version adds (index
  formula, guard, launcher, stream) is machinery for that inversion. The CPU
  side branches per machine (AVX2/FMA detected at run time), the GPU side per
  arch *at build time* (the §2.5 gencode list); both keep a scalar fallback,
  and neither hides a failed dispatch (the GPU arm's fallback is an `Err`,
  per the safety contract) — the backend's test gates compare against this
  exact CPU function (`src/graph/cuda_backend.rs:1773-1782` computes
  `vec_*_f32` and asserts the kernel output matches: the toy's
  CPU-reference pattern, scaled to the whole op set).
- **In-place is a graph decision, not a kernel decision.** The CPU signature
  takes `y` and `x` separately (out-of-place); the GPU kernel is in-place —
  on the device an extra buffer costs a full allocation and an extra stream
  of DRAM traffic, while on the CPU the write is nearly free. Same math,
  different economics — the source of most "why does the GPU version look
  different" moments you will have in Part 4.

## 4. Performance intuition

Numbers, not adjectives. All shapes are Qwen2.5-0.5B decode (one token,
`nt = 1`) — the case this chapter's kernels actually run.

**What an elementwise kernel costs.** `silu_f32` at `n = 4864` reads
4864 × 4 B = 19.5 KB and writes 19.5 KB — **~38.9 KB of DRAM traffic per
layer per step**, ~0.9 MB across the 24 transformer layers (Qwen2.5-0.5B is
24 layers — `docs/inference_e2e_walkthrough/05-graph-builder-ir.md:161`).
Compare the attention output projection (`attn_q`, an `[896, 896]` Q4_0
weight ≈ 451 KB *per layer* — walkthrough doc 14,
`docs/inference_e2e_walkthrough/14-metal-backend.md:71`) and elementwise ops
are noise in the byte budget. Their cost is therefore not bandwidth but
**latency**: launch overhead (microseconds per launch, host-side) plus the
kernel's start-to-finish time. This is why the elementwise family is where
fusion lives: decode never launches `silu_f32` standalone if it can help it —
the fused-FFN path runs `swiglu_f32_off` (`src/cuda_kernels.cu:2441-2446`,
silu+mul in one pass over the concatenated gate|up buffer) and the prefill
path fuses the q8 quantization epilogue into the same kernel
(`swiglu_quant_pad40`, `src/cuda_kernels.cu:2455-2469` — its comment block is
worth reading for the "no early return — every thread reaches the barrier"
discipline).

**Thread-count arithmetic.** 4,864 threads is a *tiny* grid. The GB10's SM
count is queried at runtime (`src/cuda.rs:1499` reads
`CUDA_DEV_ATTR_MULTIPROC_COUNT`); with a few dozen SMs each holding up to
~2048 resident threads, one 4,864-thread kernel cannot come close to filling
the machine — most blocks run, finish, and leave SMs idle. That is fine for a
2-µs, latency-bound kernel, and it is the quantitative reason decode kernels
in minfer are judged on *bytes moved per second* rather than occupancy (how
full the SMs are): at this grid size there is nothing to fill *with*.
Contrast prefill (`nt = 512`): the same op is 512 × 4864 ≈ 2.5M elements →
9,728 blocks of 256 threads — now the grid spans every SM many times over
and bandwidth becomes the limit. The grid-size formula does not change; the
regime does.

**Launch overhead is the decode tax, and CUDA Graphs are the subtraction.**
The D4-4 record in `docs/CUDA_OPTIMIZATION.md:153` measures what graph replay
recovers at "~2 µs/launch" of graph gap; multiply by the ~100+ launches of a
decode step (the same record's census) and you get hundreds of microseconds —
a real fraction of a small-model decode step. That, not kernel bandwidth, is
why §2.4's capture/replay machinery exists, and why `MINFER_NO_CUDA_GRAPH=1`
is the first knob to flip when you want to *see* the launch stream (§5).

**What would make an elementwise kernel slow** — a checklist that applies to
any kernel you write. **Uncoalesced access**: `z[tid] = x[tid] + y[tid]` with
consecutive `tid` gives each warp (32 threads) one contiguous 128-B
transaction per array; a strided index (`tid * 16`) puts each thread's 4 B in
a different transaction — up to 32× more memory transactions for the same
bytes (Part 3 dissects this; for now: keep `tid` consecutive). **A guard
that kills the warp shape**: `if (i >= n) return;` only costs when it divides
*within* a warp (at most the last 31 threads of the grid), while a guard that
diverges every warp (`if (x[i] < 0) return;`) forces the remaining lanes to
idle through the skipped code. **An idle grid**: 1 block for a 4,864-element
op would run 4,864 iterations *inside* one block — serial latency, no SM
parallelism; the ceil-div grid is the standard answer. **Unnecessary sync**:
every `cudaDeviceSynchronize()` between launches drains the whole device —
minfer's per-split `sync()` (§2.2) is the entire synchronization budget of a
step; add none.

Toy #3's 2.00× came from two kernels that *fit* side by side; when two
kernels each saturate DRAM bandwidth (as two big prefill GEMMs would), two
streams buy zero. minfer's single stream is therefore the correct design for
a dependency-chained workload, with CUDA Graphs attacking the *actual*
overhead (launches) — keep both tools in mind and let the measurement pick.

## 5. Try it / Observe

**The toys.** Both are complete listings in this chapter (§2.1 and §2.4) —
save them and run:

```console
$ export PATH=/usr/local/cuda/bin:$PATH        # nvcc is not on PATH by default
$ nvcc -arch=sm_121 -O2 toy2_silu.cu -o toy2 && ./toy2 && ./toy2 896
$ nvcc -arch=sm_121 -O2 toy3_streams.cu -o toy3 && ./toy3
```

(expected outputs are pasted under each toy; `-arch=native` works too on this
GB10 / CUDA 13.0 machine). Two experiments beyond the pasted runs:

- In toy2, delete the `if (i >= n) return;` guard and run `./toy2 896` —
  nothing visibly breaks (the extra 128 threads write past the end of the
  3.5 KB allocation into unallocated context memory), which is exactly why
  the §2.2 sticky-error + sync discipline exists. Run
  `compute-sanitizer --tool memcheck ./toy2 896` to see the OOB
  (out-of-bounds) access reported.
- In toy3, replace both `<<<1, 32>>>` launches with `<<<64, 256>>>` and
  rerun — the kernels now saturate the GPU and the "overlap" speedup
  shrinks toward 1.0× (stream parallelism bounded by resources).

**Watch minfer's real kernels.**

```console
$ cargo build --release --features cuda
$ MINFER_CUDA_DEBUG=1 ./target/release/minfer <model.gguf> "hello"
#   per-node lines from debug_sync (src/cuda.rs:2318): "l{il}: {label} OK",
#   or the launch/sync error with the node's layer + label when something breaks
$ MINFER_NO_CUDA_GRAPH=1 ./target/release/minfer <model.gguf> "hello"
#   decode without graph replay — every kernel is a separate launch; compare
#   tokens/s against the default (replay) run to feel §4's launch tax
```

**See the stream timeline.** One `nsys` command gives you the §2.4 picture —
kernels queued back-to-back on one stream, and (with graphs on) the replay as
a single launch entry; `ncu` then drills into one kernel (all profiling tools
live under `/usr/local/cuda/bin`):

```console
$ nsys profile -o /tmp/dec --force-overwrite=true \
    ./target/release/minfer <model.gguf> "hello" -n 16
$ nsys stats --report cuda_gpu_kern_sum /tmp/dec.nsys-rep     # which kernel, how long
$ nsys stats --report cuda_gpu_trace /tmp/dec.nsys-rep | head -40   # launch timeline
$ ncu --kernel-name regex:silu --launch-count 3 --set basic \
    ./target/release/minfer <model.gguf> "hello" -n 8
```

## 6. Cross-references

- **[01 · What kind of machine is a GPU](01-gpu-mental-model.md)** — the
  execution model you built on: threads, warps, blocks, grids, Toy #1.
- **[03 · Reading minfer's kernels I](03-kernels-elementwise.md)** — next:
  the elementwise/dequant/embedding kernels read line by line, with their
  CPU counterparts.
- [`docs/CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) — the technique
  reference; §2 covered its "runtime basics" layer, the rest goes deeper.
- [`docs/CUDA-BACKEND-DESIGN.md`](../CUDA-BACKEND-DESIGN.md) — the backend
  design record (phases 7a–7d) this chapter's code implements.
- [`docs/BUILD.md`](../BUILD.md) — the full build reference behind §2.5.
- [`docs/GPU_SAFETY.md`](../GPU_SAFETY.md) — the hard rules quoted in §2.2;
  read in full before touching Metal/CUDA code.
- [`docs/inference_e2e_walkthrough/`](../inference_e2e_walkthrough/) 09–15 —
  the pipeline this CUDA layer serves (10: CPU matmul kernels, 11:
  attention/vec ops/KV, 15: the CUDA backend chapter — the "what" to this
  "how").
- [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) — where §4's
  launch-tax numbers come from (per-step records link from each row).
- [`docs/SUPPORT-MATRIX.md`](../SUPPORT-MATRIX.md) — which quants reach the
  GPU and how; the matrix behind the `dequant_*_f16` / MMQ kernel families.
- [`docs/GLOSSARY.md`](../GLOSSARY.md) — the fallback for any term this
  chapter defined too briefly.

← [01 · What kind of machine is a GPU](01-gpu-mental-model.md) · [Index](./README.md) · [03 · Reading minfer's kernels I →](03-kernels-elementwise.md)
