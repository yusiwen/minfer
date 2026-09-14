# 01 · What kind of machine is a GPU

> **Part**: Part 1 — the mental model (chapters 01–02: concepts first, then the minimal CUDA toolkit).
> **Prereq**: [docs/inference_e2e_walkthrough/09–13](../inference_e2e_walkthrough/) — you know minfer's CPU pipeline: prefill (doc 09), the quantized matmul kernels and the thread pool (doc 10), attention and vec ops (doc 11), the decode loop (doc 13). Zero CUDA knowledge is assumed.
> **Code**: none — concepts + one toy. minfer's kernel source reading starts in ch. 03; the platform summary below defers to [docs/CUDA-TECH-PRIMER.md](../CUDA-TECH-PRIMER.md).

## 1. Background — where this sits

Through walkthrough docs 09–13 you followed one full inference run on the CPU.
Prefill pushed the whole prompt through the compute graph in a single forward
pass; decode then looped one token at a time; every matmul was quantized weight
rows multiplied against Q8_0 (8-bit, block-scaled) activations; and a small
persistent thread pool split each matmul's output rows across the CPU's cores.
Doc 15 then showed that the same graph can execute on a CUDA backend — but it
treated the GPU as a black box that receives tensors and returns logits. This
tutorial ladder opens that box.

First, why the box exists at all. Doc 10's decode arithmetic is the whole
motivation: a 7B Q4_K_M model keeps ~4.4 GB of quantized weights, and a decode
step must stream **all of them from RAM for every single token**, because with
one token row there is nothing to amortize the weight reads over. On a CPU-class
memory system moving ~60–100 GB/s, that alone caps decode at roughly 15–25
tokens/s — no amount of clever SIMD changes the ceiling, because the bottleneck
is bytes moved, not math done (doc 10 §2). A GPU's answer is not smarter
arithmetic. Its answer is **orders of magnitude more parallel work in flight**,
so the memory system is kept saturated instead of idling between cache misses —
the same argument doc 14 makes for Metal on Apple silicon: thousands of GPU
lanes keep the memory controller saturated in a way a score of CPU cores
cannot. minfer grew a CUDA backend because on the target machine that
parallelism is what lets the 4.4 GB weight stream run at 75–84% of the memory
system's peak rate instead of trickling through a handful of cores.

But a GPU is a genuinely different *kind* of machine, with its own execution
model, its own memory hierarchy, and its own ways to lose performance. So the
ladder is taught in steps:

| Chapter | What it adds |
|---|---|
| **01 (this one)** | The mental model: host vs device, SIMT execution, the memory hierarchy, and one runnable toy (vector add). |
| **02** | The minimal CUDA you actually need: kernel syntax in full, the runtime API surface, and the error-handling discipline. |
| **03–05** | Reading minfer's real kernels in `src/cuda_kernels.cu`, easiest first: elementwise → decode GEMV/matvec → prefill GEMM and attention. |
| **06** | Optimization: the techniques the CUDA campaign actually measured, each linked to the step record that used it. |

Three runnable toys carry the hands-on thread (vector add here; an
elementwise/SiLU toy and a streams toy later; they are indexed in
`07-where-next.md`'s appendix). After this chapter you should be able to do
four things: define every term in the header of any CUDA kernel launch; predict
what a kernel launch *does* before you run it; do the bytes-vs-bandwidth
arithmetic that decides whether a kernel is memory-bound or compute-bound; and
say *why* minfer's decode path is shaped the way it is. You will also have
compiled and run a CUDA program on this GB10.

## 2. Principle — the concepts

### 2.1 Host and device — two machines, two address spaces

A CUDA program runs on two computers at once, and CUDA has words for both. The
**host** is the CPU side — for minfer, the Rust process. The **device** is the
GPU side — the `.cu` code compiled by nvcc. They are separate machines with
separate **address spaces**: a pointer on the host means nothing on the device,
and a pointer the device returns means nothing to the host. Every byte that
crosses the boundary crosses it *explicitly*, through a copy call
(`cudaMemcpy`) or through memory that is deliberately shared. This is the first
mental shift from CPU programming, where a `&[f32]` is just a `&[f32]` no matter
which function reads it.

The consequence that shapes everything else in minfer: **the data transfer is
the cost**. Copying bytes between host and device is no faster than any other
memory traffic — often slower — so a design that ships weights to the GPU per
forward pass pays a transfer tax on every token. minfer's design (TECH-PRIMER
§1, §6.1) is the direct answer: upload every weight tensor **once** at load
time, leave it resident on the device for the process lifetime, and never copy
it again. After that, decode moves essentially zero bytes across the
host↔device boundary — positions and token ids ride over as a few dozen bytes
per step, written into device buffers. The Phase-7 step record measured what
the opposite extreme costs: a transfer-dominated prefill shape ran at
30.7 tok/s where the weight-resident design later reached 1204 tok/s on the
same model (TECH-PRIMER §6.2, step docs 01–02).

The machine this runs on — and that this tutorial's toys run on — is an
**NVIDIA GB10** (DGX Spark). In a few lines, from
[docs/CUDA-TECH-PRIMER.md §1](../CUDA-TECH-PRIMER.md) (read it for the depth):
it is a Grace-Blackwell "superchip" where a 20-core Arm CPU — the same machine
that runs minfer's NEON+SDOT CPU backend — and a Blackwell-class GPU share one
coherent memory pool (~128 GB, LPDDR5x). The GPU reports compute capability
12.1 (`sm_121` in nvcc's naming; [docs/GLOSSARY.md](../GLOSSARY.md)), the
driver here is 580.173.02. One warning the record has already paid for:
"unified memory" does **not** mean the program can treat CPU and GPU pointers
as one — minfer deliberately uses plain `cudaMalloc` device pools and
weight-resident uploads, never `cudaMallocManaged`. Unified *physics*, still
two *address spaces*.

For the concrete machine, these are the numbers I queried on the GB10 with the
11-line probe in §5 (you should run it too; they are the launch-planning
constants for every later chapter):

| Property (queried via `cudaGetDeviceProperties`) | GB10 value |
|---|---|
| Name / compute capability | NVIDIA GB10 / 12.1 (`sm_121`) |
| **SMs** (streaming multiprocessors — §2.2) | **48** |
| Warp size | 32 threads |
| Max resident threads per SM / per block | 1536 / 1024 |
| Shared memory per block | 48 KiB |
| L2 cache | 24 MiB |
| Reported global memory | 121.6 GiB |

### 2.2 SIMT — thread, warp, block, grid, SM

Now the execution model. A **kernel** is an ordinary-looking function (marked
`__global__` in the code) with one twist: it does not run once — it runs once
*per thread*, on many thousands of threads at once. You write "the work of one
element", and the hardware clones it across the whole dataset. A **thread** is
one of those clones: it has its own program counter, its own registers, and a
built-in idea of *which* element it is responsible for. The launch call —
`kernel<<<grid, block>>>(args)` — does not call the function; it *schedules*
the function to be executed by a specific number of threads, and returns
immediately.

The launch parameters name two levels of that thread army. A **block** is a
group of threads (up to 1024) that is scheduled onto exactly one **SM** — a
*streaming multiprocessor*, the GPU's physical compute unit, with its own
registers, schedulers, and cache. A block never splits across SMs and never
migrates; threads inside a block can cooperate (share a scratchpad, wait on
barriers — chapters 03–04). The **grid** is the collection of all blocks one
launch creates. So the full picture of `vec_add<<<GRID, BLOCK>>>` is: GRID
blocks × BLOCK threads, every block housed by some SM, all executing the same
few lines of code on different data.

Inside the SM, threads do not actually proceed one at a time. They are
organized into **warps** of exactly 32 threads, and the warp is the unit the
hardware actually steps: all 32 threads of a warp fetch and execute the *same
instruction* at the same time, each on its own data. That design is called
**SIMT** — single instruction, multiple threads — and it is why GPUs are cheap
per thread: one instruction decoder and scheduler serves 32 lanes. It is also
why `if` statements have a hidden price: if some threads of a warp take the
`if` branch and others the `else`, the warp executes *both* branches serially,
masking off half its lanes each time (**divergent branch** — a term chapter 03
will make concrete).

A classroom analogy, carried as far as it holds: the kernel is a worksheet
with one exercise ("add `a[i]` and `b[i]`, write `c[i]`"). A **thread** is one
student doing one exercise. A **warp** is a row of 32 desks that must move
together — the teacher (the SM) reads each instruction once and the whole row
executes it in unison. A **block** is a classroom: up to 1024 students who
share a blackboard (shared memory) and can coordinate among themselves; the
whole class is assigned to one physical room (the SM) for its entire stay. The
**grid** is the school day: every classroom launched at once. Where the analogy
lies: classrooms are virtual. One SM hosts *several* blocks simultaneously (up
to its resource limits), and when there are more blocks than fit, they run in
batches — see occupancy below.

Two quantities you will meet in every minfer performance discussion fall out of
this picture. **Occupancy** is how much of an SM's hosting capacity — thread
slots, registers, shared memory — a kernel actually uses; an SM at full
occupancy has enough resident warps to switch to a different warp the moment
the current one stalls (e.g. waiting on a memory load), which is how GPUs hide
latency. **Waves** is the grid-level version: the GPU runs blocks in batches of
"(blocks resident per SM) × (SM count)"; total blocks divided by that is the
wave count, and a kernel whose grid is 1.5 waves wastes half of the second,
nearly-empty wave (**wave quantization** — a recurring villain in the campaign
record: a fused-kernel probe at 1.5 waves measured +28.2% slower, TECH-PRIMER
§4). The GB10 has 48 SMs accepting up to 1536 threads each = 73,728 resident
threads; with minfer's usual 256-thread blocks that is 6 blocks per SM, 288
blocks resident at once. The toy in §3 launches 262,144 blocks — about 910
waves. minfer's decode GEMMs at batch size 1 launch *fewer* blocks than one
wave holds — 0.14 waves in the record — which is exactly why they amortize so
well when batched (TECH-PRIMER §4).

**Contrast with the CPU pool you already know** (walkthrough doc 10). minfer's
CPU backend parallelizes a matmul with a hand-built persistent pool
(`src/kernel.rs:289` `get_pool`): workers are OS threads spawned once because
spawning measured ~170 µs (`src/kernel.rs:40`) — against a per-token budget of
a few milliseconds; they spin on an atomic generation counter to wake in
microseconds; the work unit is a *chunk of output rows*
(`chunk(parts, idx, total)`, `src/kernel.rs:255`); and correctness rests on
"each row belongs to exactly one worker", so the result is bit-identical at any
thread count. Around 20 workers exist on this machine — one per core, because
on the CPU, parallelism is expensive and scarce.

The GPU inverts nearly every line of that design:

| | CPU pool (doc 10) | GPU grid (this chapter) |
|---|---|---|
| Unit of parallelism | OS thread | hardware thread (in warps) |
| How many | ~cores (20 on GB10) | thousands resident; millions launched |
| Cost of one more | real (stack, wake latency) | ~zero until SM slots fill |
| Work unit | a chunk of output *rows* | one output element (or a small tile, ch. 04) |
| Who schedules | you (atomics, chunking, spin) | the hardware (block scheduler) |
| Your job | split rows fairly, wake workers | give every thread a unique index; arrange memory (§2.3); launch enough blocks |
| Parallelism cost floor | thread spawn ~170 µs | kernel launch ~2–7 µs (TECH-PRIMER §8) |

So when "one core becomes 48 SMs × 1536 thread slots", what changes
conceptually is not "more threads of the same kind". It is that the *unit of
scheduling* moves out of your hands: you stop managing workers and start
*describing* an army of independent index-holders, then spend your effort on
the two things the hardware will not do for you — mapping indices to memory
locations efficiently (coalescing, §2.3) and launching enough blocks to keep
all SMs fed (occupancy and waves).

### 2.3 The memory hierarchy — where the bytes actually are

The second mental shift is about memory. The GPU's memory system is a stack of
levels, each smaller and faster than the one below, and a kernel's performance
is largely *which level its data comes from*. From the thread's point of view:

- **Registers** — private per thread, where your local variables live. Fastest
  by far. (Per-thread, ~0 cycles — TECH-PRIMER §5.1.)
- **Shared memory** — a small scratchpad (48 KiB per block here) *you* manage
  explicitly: a block stages data from the big pool into it so all its threads
  can reuse the bytes at low cost. You opt in with `__shared__` declarations;
  chapter 04 exercises it.
- **L1 / L2 caches** — hardware-managed, shared across blocks on an SM (L1) and
  across the chip (L2; 24 MiB here).
- **Device memory** — the big pool (121.6 GiB reported), called **global
  memory** in CUDA because it is visible to every thread of every block via
  plain pointers. This is where `cudaMalloc` puts tensors, where weights and
  KV cache live.

The magnitude table, using the latency classes TECH-PRIMER §5.1 records for
this platform plus the two bandwidth figures this chapter can defend:

| Level | Scope | Who manages it | Typical latency | Bandwidth character |
|---|---|---|---|---|
| Registers | one thread | the compiler | ~0 cycles | effectively free per operand |
| Shared memory | one block | **you** (`__shared__`) | ~30 cycles (tens) | very high, but tiny capacity |
| L1 / L2 | per-SM / chip | hardware | ~200 / ~400 cycles | high; 24 MiB total here |
| Device (global) memory | all threads | `cudaMalloc` | ~600+ cycles (hundreds) | **the ceiling that matters: ~273 GB/s spec** ([docs/GLOSSARY.md](../GLOSSARY.md): GB10's unified LPDDR5x, shared CPU+GPU); **~225–229 GB/s measured** by this chapter's toy (§3) |

**Memory bandwidth** is the term for that last number: how many bytes per
second the memory system can stream, regardless of how much compute sits
around it. It deserves care with vocabulary, because GPUs differ wildly here.
High-end datacenter GPUs stream from **HBM** — High Bandwidth Memory, DRAM
stacks mounted beside the die — at multiple TB/s. The GB10 instead shares one
LPDDR5x pool between CPU and GPU, spec'd at ~273 GB/s (GLOSSARY). That is the
roofline every later chapter measures against: minfer's decode kernels sustain
75–84% of it (step doc 74), the campaign's best-pure-stream kernel hit 89%
(r55, step doc 58), and this chapter's trivial toy will land at ~83%. Bandwidth
*is* the decode wall, so every GPU decision in minfer is ultimately a decision
about bytes.

One paragraph preview of the most important byte-saving idea. The 32 threads of
a warp execute the same load instruction; if their 32 addresses are *adjacent*
(say 32 consecutive `float`s), the hardware folds them into the minimum number
of wide memory transactions — 128-byte sectors. That is **coalescing**. If the
addresses are scattered instead, the same instruction can generate many
separate transactions and burn multiples of the bandwidth for the same data.
TECH-PRIMER §5.2 records the real stakes: repacking one weight layout so its
bytes sat contiguously (the dpl repack) cut content traffic by 17.1% on one
kernel — same math, same values, purely a memory-map change. Coalescing gets
its full treatment with exercises in chapters 03–04; for now, remember the
rule of thumb: *adjacent threads should read adjacent memory*.

### 2.4 Why LLM inference loves (and hates) GPUs

Put the two previous sections together and you get the performance model that
organizes everything else. A kernel's time is bounded by the larger of two
costs: the bytes it must move through the memory system (bytes ÷ bandwidth) or
the arithmetic it must do (operations ÷ peak throughput). The ratio of
operations to bytes — **arithmetic intensity** — decides which bound you sit
against: low intensity means memory-bound (the memory is the bottleneck), high
intensity means compute-bound. LLM inference has one of each phase, which is
why minfer's two paths look so different.

**Prefill** (walkthrough doc 09) runs the whole prompt at once: `nt` token rows
go through every matmul together, so each matmul is a true **GEMM** (General
Matrix-Multiply — matrix × matrix). Every weight byte is reused `nt` times
inside the arithmetic, so the more tokens you batch, the more compute each
byte buys: intensity grows with `nt`, and large-shape prefill is
*throughput-bound* — it wants maximum math, which is what tensor cores (the
SM's matrix-multiply hardware) provide. minfer's prefill path is an int8
**MMQ** GEMM — matrix-matrix quantized: weights stay quantized, activations
are quantized on the fly to 8-bit so the integer multiply hardware multiplies
the throughput (walkthrough doc 09 for the pipeline, TECH-PRIMER §6.2 for the
kernel families) — plus `fa_prefill_f16kv`, a FlashAttention-style tiled
attention kernel. The record's headline of what "the GPU is good at this"
means: swapping the transfer-dominated prefill for weight-resident tensor-core
GEMMs took 7B prefill from 30.7 to 1204 tok/s (39×, TECH-PRIMER §6.2).

**Decode** (walkthrough doc 13) is the opposite regime. One token per forward:
~250 matmuls per token (doc 10 §3), and every one of them degenerates to a
**GEMV** (matrix-×-vector: one activation row against the whole weight
matrix). Each weight byte now travels from device memory to be used in exactly
one multiply-add — the measured class is ~0.03 FLOP per byte (step doc 74) —
so decode is *always memory-bound*: its time is essentially "weight bytes ÷
bandwidth", full stop. That is why minfer's decode path is a GEMV-style matvec
family (`q*_q8_mmvq`, dp4a integer dots per weight row) whose only goal is to
stream bytes at the highest rate the memory system allows (TECH-PRIMER §6.2),
and why the campaign's decode attribution speaks in GB/s, not tok/s (step doc
67). It is also why decode needs a second trick: the step is a chain of ~13
small kernels, each costing ~2–7 µs of CPU-side launch overhead — negligible
for one kernel, but the chain re-runs for every token, and hundreds of launches
per step (the prefill chain runs ~380) make launch overhead a first-class cost.
**CUDA Graphs** (TECH-PRIMER §8) record the whole launch sequence once
(capture) and re-issue it with one call (replay), collapsing that overhead;
`MINFER_NO_CUDA_GRAPH=1` reverts to per-kernel launches and is the standard A/B
control in the step records.

And note what *both* paths never do after load: talk to the host. Weights
resident (§2.1), activations and KV living in device pools — the transfer cost
of inference was paid once, at model load. The rest of this tutorial lives
inside the device.

## 3. Toy #1 — your first kernel: vector add

Everything above, in 58 lines. `c[i] = a[i] + b[i]` for 67 million elements:
the "hello world" of CUDA, and — not coincidentally — the same *shape* as
minfer's elementwise kernels (`add_f32`, `add_bias_f32`: TECH-PRIMER §6.4).
The N is chosen large on purpose: 3 × 256 MiB = 768 MiB of traffic, far beyond
the 24 MiB L2, so the timing measures real device-memory streaming and not
cache or launch effects.

```cuda
// toy1_vec_add.cu — Toy #1 (CUDA tutorial ch. 01): c[i] = a[i] + b[i], on the GPU.
// Verified with CUDA 13.0 (/usr/local/cuda/bin/nvcc) on NVIDIA GB10, -arch=sm_121.
#include <cstdio>
#include <cstdlib>
#include <cmath>

// A "kernel": one function body, executed by MANY GPU threads at once.
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) c[i] = a[i] + b[i];                  // guard: the last block may overrun n
}

int main() {
    const int    N     = 1 << 26;                    // 67,108,864 elements (256 MiB per array)
    const size_t bytes = N * sizeof(float);          // 3 arrays => ~768 MiB of DRAM traffic below
    const int    BLOCK = 256;                        // threads per block (minfer's usual choice)
    const int    GRID  = (N + BLOCK - 1) / BLOCK;    // ceil-div: enough blocks to cover all N

    // host = CPU memory; device = GPU memory. Two separate address spaces.
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { ha[i] = i * 0.001f; hb[i] = 1.0f; }

    float *da, *db, *dc;                             // device-side pointers
    cudaMalloc(&da, bytes);                          // allocate GPU memory for a
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);  // copy a: CPU -> GPU
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);  // copy b: CPU -> GPU

    // Two events on the default stream bracket the kernel; they read GPU time.
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    vec_add<<<GRID, BLOCK>>>(da, db, dc, N);         // the launch: GRID blocks x BLOCK threads
    cudaError_t err = cudaGetLastError();            // launch errors surface HERE, asynchronously
    if (err != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(err)); return 1; }
    cudaEventRecord(t1);
    cudaDeviceSynchronize();                         // block the CPU until the GPU is truly done
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);               // measured time between the two events

    cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);  // read the result back: GPU -> CPU

    for (int i = 0; i < N; i++) {                    // verify against a plain CPU loop
        if (fabsf(hc[i] - (ha[i] + hb[i])) > 1e-5f) {
            printf("MISMATCH at %d: %f vs %f\n", i, hc[i], ha[i] + hb[i]);
            return 1;
        }
    }
    double gbs = 3.0 * bytes / (ms * 1e-3) / 1e9;    // read a + read b + write c
    printf("OK  n=%d grid=%d block=%d  kernel=%.3f ms  ~%.0f GB/s effective\n",
           N, GRID, BLOCK, ms, gbs);

    cudaFree(da); cudaFree(db); cudaFree(dc);        // free GPU memory
    free(ha); free(hb); free(hc);
    return 0;
}
```

Read it in five groups.

**The kernel (lines 8–11).** `__global__` (pronounced "global") marks a
function compiled *for the device*, launchable from the host — that is CUDA's
word for "kernel". The body is the work of *one* thread: read `a[i]`, read
`b[i]`, write `c[i]`. There is no loop over N — the loop is replaced by the
launch. The interesting line is the index formula
`i = blockIdx.x * blockDim.x + threadIdx.x`. Every thread carries three
built-in variables identifying it: `threadIdx.x` (my position within my block),
`blockIdx.x` (my block's position in the grid), and `blockDim.x` (threads per
block, the same everywhere). The product-plus-sum turns the two-level
(block, thread) address into one global element index. Note the types: these
are 32-bit integers, so `i` overflows past 2³¹−1 — a real bug class for large
tensors that chapter 02 returns to. The `if (i < n)` guard exists because GRID
is rounded *up*: with N = 100 and BLOCK = 256, one block of 256 threads is
launched and threads 100–255 must do nothing. Omitting the guard writes past
the buffer — on the GPU that silently corrupts memory or faults later, which
is why the guard is never optional even when the current N divides evenly.

**The mapping, drawn.** A miniature of the same launch with n = 8 and
block = 4 (grid = ⌈8/4⌉ = 2) — every element of `c` is claimed by exactly one
thread, the GPU analogue of doc 10's "each row belongs to exactly one worker":

```text
launch vec_add<<<grid=2, block=4>>>(a, b, c, n=8)

        block 0                           block 1
   (blockIdx.x = 0)                  (blockIdx.x = 1)
   threadIdx:  0   1   2   3         0   1   2   3
   global i:   0   1   2   3         4   5   6   7
               |   |   |   |         |   |   |   |
   c:        c[0] c[1] c[2] c[3]   c[4] c[5] c[6] c[7]

   i = blockIdx.x * blockDim.x + threadIdx.x        (blockDim.x = 4)
   block 1, thread 2  ->  i = 1*4 + 2 = 6  ->  c[6] = a[6] + b[6]
```

**Host and device memory (lines 19–28).** `malloc` gives host pointers `ha`,
`hb`, `hc`; `cudaMalloc` gives device pointers `da`, `db`, `dc` — same C
syntax, different world, and the two pointer kinds are not interchangeable
(§2.1). `cudaMemcpy(dst, src, bytes, kind)` is the explicit bridge; the `kind`
enum (`cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`) says which way.
These two copies are the toy's real "load time": 512 MiB cross the boundary
before any math happens. In minfer that cost is paid once per weight at load
and never again; here we pay it per run because the program is short.

**Timing and launch (lines 30–41).** A **stream** is an ordered queue of GPU
work — launches and copies land in it and execute in order; the code so far
used the implicit *default stream*. A **cudaEvent** is a marker you place in a
stream; when the GPU reaches it, it timestamps it. `cudaEventRecord(t0)` …
launch … `cudaEventRecord(t1)` therefore measures the kernel's time *on the
GPU's own clock* — unlike a stopwatch around the launch call, which would
measure the (asynchronous!) enqueue and not the work. The launch line
`vec_add<<<GRID, BLOCK>>>(...)` is the only non-C++ syntax in the file: it
says "run this kernel with GRID blocks of BLOCK threads". And note the check
right after it: `cudaGetLastError()`. Kernel launches are asynchronous —
the line *returns immediately*, and an invalid launch (too many threads per
block, no device, bad config) does not raise an error at the call site the way
a Rust `Result` would; the error is parked until you ask. Not asking is how
"silently wrong" GPU programs are born; minfer's whole backend checks errors
at every step (`docs/GPU_SAFETY.md` culture). `cudaDeviceSynchronize()` then
blocks the host until *all* queued device work is done — mandatory before
reading results back, because the `cudaMemcpy` readback is stream-ordered but
the host must not race ahead of the GPU. (Chapter 02 replaces the blunt
device-wide sync with the precise stream-ordered tools.)

**Verify, report, free (lines 43–57).** The result is read back and compared
against a plain CPU loop — the same "compare each path against its own
reference" discipline minfer's step records use (TECH-PRIMER §9), in miniature:
the GPU's answer for this kernel is *bitwise* identical to the CPU's (same f32
adds, same order — no reduction is involved), so a tolerance is generosity,
not necessity, here. The printout does the byte accounting for §4: vector add
reads `a` and `b` and writes `c`, so it moves `3 × N × 4` bytes of device
traffic; dividing by the measured seconds gives an *effective* bandwidth.
Finally every `cudaMalloc` is paired with a `cudaFree` — in minfer this pairing
is wrapped in RAII device-buffer types (`src/cuda.rs`) so it cannot be
forgotten.

**Compile, run, and the actual observed output** (CUDA 13.0, V13.0.88,
`/usr/local/cuda/bin/nvcc`; NVIDIA GB10, driver 580.173.02):

```console
$ /usr/local/cuda/bin/nvcc -O2 -arch=sm_121 toy1_vec_add.cu -o toy1_vec_add && ./toy1_vec_add
OK  n=67108864 grid=262144 block=256  kernel=3.587 ms  ~224 GB/s effective
```

Five runs measured 3.514–3.587 ms (~224–229 GB/s effective).
`-arch=sm_121` tells nvcc to emit machine code for this GPU's compute
capability 12.1 (TECH-PRIMER §3 explains the SASS/PTX machinery minfer's
`build.rs` automates). Look at the numbers and connect them to §2.2: 262,144
blocks × 256 threads = 67,108,864 thread instances, ~910 waves over 48 SMs,
each wave alive for ~4 µs — and the whole 768 MiB streaming job is over in
3.5 ms.

## 4. Performance intuition — bytes over bandwidth

Now do to this toy what the campaign does to every kernel: bound it from
below, measure it, and explain the gap.

**Bytes.** Vector add reads two `float` arrays and writes one: `3 × N × 4` =
3 × 67,108,864 × 4 = 805,306,368 bytes ≈ 805 MB (768 MiB). No kernel that must
touch every element can beat the time it takes the memory system to move those
bytes — that is the roofline idea of §2.4, in its simplest case.

**Lower bound.** At the documented GB10 roofline of ~273 GB/s (GLOSSARY;
calibrated by the campaign's r55 audit, step doc 09), the floor is
805.3 MB ÷ 273 GB/s ≈ **2.95 ms**.

**Measured.** The toy measured **3.51–3.59 ms**, i.e. an effective
**225–229 GB/s = 82–84% of the roofline**. The remaining ~15% is the ordinary
tax of real streaming: DRAM latency not fully hidden, L2/write-back effects,
clock behavior — not a bug in our kernel. Two reference points say this is
exactly the right neighborhood: minfer's decode matvec kernels sustain 75–84%
of this same peak (step doc 74), and the campaign's best pure stream — fused
swiglu, audited at 89% (r55, step doc 58) — only ~6 points better. A one-evening
toy lands in the class that the campaign needed months to tune toward, because
vector add has no quantization to unpack, no gather, no reuse — *it is already
the pure byte stream that decode is fighting to become*. (Doc 15's number for
minfer's decode weight streaming, the "200–225 GB/s class", is the same
magnitude you just measured.)

**Arithmetic intensity, seen rather than computed.** Vector add does 2 FLOPs
per 12 bytes moved ≈ 0.17 FLOP/byte. Decode GEMV sits at ~0.03 (step doc 74).
Both are deep on the memory-bound side — and you can *see* it in the numbers
without knowing the GPU's peak FLOPs: the kernel's time tracked the byte count
(805 MB → 3.5 ms at ~83% of bandwidth), not any arithmetic budget. Whenever a
kernel's time is predictable from its bytes, it is memory-bound; that one test
is most of chapter 04's toolkit.

**What breaks the simple model — the small end.** Re-run the same toy smaller
and the model collapses honestly: at N = 2²⁰ (12 MiB of traffic), the same
binary measured 0.127–0.155 ms — bytes shrank 64× but time only ~25×, so the
effective bandwidth collapsed to ~80–100 GB/s; at N = 2¹² it measured
0.057–0.105 ms on an idle GPU, where fixed path costs and clocks ramping from
idle dominate and the "bandwidth" printed is meaningless (~1 GB/s). Below some
size, time stops tracking bytes: launch latency, memory *latency* (not
bandwidth — too few blocks in flight to hide it), and measurement hygiene take
over. This is not a corner case — it is minfer's decode regime (grids at 0.14
waves; the D1 attribution's verdict of memory-*latency*-bound with 76.5% of
stalls on long scoreboard waits, TECH-PRIMER §6.3), and it is why the campaign
insists on same-window interleaved A/B medians (TECH-PRIMER §1): an idle
integrated GPU measures nothing the way a busy one does.

**Checklist of ways this kernel could be slow or wrong** (each will get its
chapter): non-adjacent per-thread addresses → no coalescing, multiple
transactions per warp (§2.3); a grid smaller than one wave → SMs idle; the
`int i` index overflowing past 2³¹ elements; reading `c` back before the
synchronize → stale bytes; and skipping the `cudaGetLastError` check → an
invalid launch fails silently and the CPU verification "passes" against
uninitialized memory — the failure mode chapter 02's error-handling discipline
exists to prevent.

## 5. Try it

Save the §3 block as `toy1_vec_add.cu` in any scratch directory (the filename
the toy index in `07-where-next.md` uses), then:

```console
$ export PATH=/usr/local/cuda/bin:$PATH     # nvcc is not on every shell's PATH
$ nvcc -O2 -arch=sm_121 toy1_vec_add.cu -o toy1_vec_add && ./toy1_vec_add
OK  n=67108864 grid=262144 block=256  kernel=3.587 ms  ~224 GB/s effective
```

(Without the `export`, use the full path: `/usr/local/cuda/bin/nvcc …`.
Verified with CUDA 13.0, V13.0.88, on the GB10, driver 580.173.02.)

Experiments, each one a one-line edit away:

- **N = 1 << 20** — watch effective bandwidth collapse (~80–100 GB/s): the
  fixed-cost/latency regime of §4, decode's home turf.
- **BLOCK = 32** (one warp per block) vs **BLOCK = 1024** — the time should
  barely move for this trivial kernel; note for later that minfer chose 256
  (`__launch_bounds__(256)`, TECH-PRIMER §4.1) for reasons that matter in
  *real* kernels (registers, occupancy tuning), not for vector add.
- **N = 1000003** (odd) with the `if (i < n)` guard deleted — ~1,500 threads
  now write ~768 bytes past the end of `hc`: heap corruption that may not
  fault, and may not even trip the CPU check. Never trust "it divided evenly
  last time"; the guard is free insurance.
- **See the machine, not just the timing:** `nsys profile ./toy1_vec_add` opens
  the Nsight Systems timeline; you should see the two H2D copies, the kernel
  (~3.5 ms), and the D2H copy as separate stream-ordered items — §2.1 and the
  event story, drawn.

And the device-properties probe used in §2.1 — the 11 lines that print this
chapter's platform table (48 SMs, warp 32, 1536 threads/SM, 48 KiB shared,
24 MiB L2):

```cuda
// devprobe.cu — query the device constants every later chapter assumes.
#include <cstdio>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("name=%s cc=%d.%d SMs=%d warpSize=%d maxThreadsPerSM=%d maxThreadsPerBlock=%d\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.warpSize,
           p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock);
    printf("sharedMemPerBlock=%zu KiB L2=%d MiB globalMem=%.1f GiB\n",
           p.sharedMemPerBlock / 1024, p.l2CacheSize >> 20, p.totalGlobalMem / 1073741824.0);
    return 0;
}
```

```console
$ nvcc -O2 -arch=sm_121 devprobe.cu -o devprobe && ./devprobe
name=NVIDIA GB10 cc=12.1 SMs=48 warpSize=32 maxThreadsPerSM=1536 maxThreadsPerBlock=1024
sharedMemPerBlock=48 KiB L2=24 MiB globalMem=121.6 GiB
```

## 6. Cross-references

- **[02 · The minimal CUDA you actually need](02-minimal-cuda.md)** (next):
  kernel syntax in full, the runtime API surface, and the error-handling
  discipline this chapter only gestured at with `cudaGetLastError`.
- **[docs/CUDA-TECH-PRIMER.md §1](../CUDA-TECH-PRIMER.md)** — the platform in
  depth (GB10, unified memory, weight-resident design); **§4** — the
  programming-model vocabulary used throughout (grid/block/warp, occupancy,
  waves, `__launch_bounds__`); **§5** — the memory hierarchy and coalescing at
  reference depth.
- **[docs/inference_e2e_walkthrough/10](../inference_e2e_walkthrough/10-cpu-matmul-kernels.md)**
  — the CPU counterpart of this chapter's execution model: the persistent
  thread pool, row-chunked work, and the byte arithmetic that makes decode
  bandwidth-bound on the CPU too.
- **[docs/inference_e2e_walkthrough/13](../inference_e2e_walkthrough/13-decode-loop-graph-reuse.md)**
  — the decode loop whose kernel chain (GEMV + CUDA Graph replay) chapters
  03–05 read line by line.
- **[docs/GLOSSARY.md](../GLOSSARY.md)** — the campaign glossary, for any term
  this chapter defines only in passing (sm_121, LPDDR5x, roofline).

← [Index](./README.md) · [02 · The minimal CUDA you actually need](02-minimal-cuda.md) →
