# 70 · D3-5 — decode-alignment plan Stage 1: fused-producer decode A-quantize (LANDED) + negative analysis of the output-head/ffn_down geometry levers (1b/1c)

> **Result**: **1a landed** (`3230b2b`) — `rms_norm_quant_pad40` and
> `swiglu_quant_pad40` write a pad40 q8 plane alongside the f32 output; decode
> MMVQ consults via `decode_quantize_native` (MmqCache, keyed by the
> producer's f32 output pointer) and on a hit skips the standalone quantize:
> 14B @3254 standalone quantize launches **4448 → 964 (−78%)**, wall clock
> 14B tg128 **+1.30%**, @3254 **+1.50%**, 7B tg128 **+1.66%**, @1641
> **+1.43%** (all SEP; the q8 bytes are bitwise by construction). **1b/1c
> analysis-negative** — neither the output-head od-split/512-thread nor the
> ffn_down 512-thread single-unit geometry variant has a measurable
> mechanism; per the "measure the mechanism before writing code" rule, not
> built.
> **Commit**: `3230b2b` (1b/1c no repo change). **Date**: 2026-09-08.

## 1. Background — where things stood

After D3-4 landed the hybrid rpw dispatch (doc 69), 14B @3254's distance
account read 46.88 vs 41.12 ms/step (−12.3%), and the largest class in the
known lever list was neither GEMM nor attention but **launch structure**:
D3-1's census counted ~700 sub-2 µs launches per decode step, of which ~265
are standalone `quantize_q8_0_pad40` — **one hangs in front of every decode
MMVQ matmul**, ~1.7 µs each, ~0.46 ms/step total. More embarrassing, these
quantizes repeat work: one row of attn_norm output is re-quantized **3 times
per layer** before the q/k/v matmuls. r49 had already prescribed for the
same disease on the prefill side (the A-quantize prepass's shared-A window
memoization), and r51/r52 advanced it to "the producer writes the q8 plane
directly" (mode 1/2) — prefill's quantize launches fell from 193 to 28. The
decode side was still the old world.

This doc is Stage 1 (1a) of the decode-alignment plan, plus the paper
settlement of two geometry levers listed in the brief (1b/1c). Anchors
(same-window interleaved 3× medians, pre side): 14B tg128 22.97 (interleaved
pre 23.05), @3254 21.06 (one 15.87 co-tenant outlier rep; interleaved pre
21.36); 7B tg128 49.90 (interleaved 49.86), @1641 48.71 (interleaved 48.47).
The sglang co-tenant was resident throughout (idle co-residency is
equivalent to a clean machine; per-rep outliers land on both sides and the
median carries).

Where things stall without this step: launch structure is a **length-independent** fixed tax — tg128 and @3254 fire the same number of quantize launches per step. It presses on every shape's wall clock and aligns directly with CUDA-graph's "one launch ≈ 2 µs pool" account: 78% fewer quantize launches ≈ reclaiming nearly 350 launch slots per step.

## 2. Principle — the GPU mechanism: the three-layer account of moving quantization back to the producer

### 2.1 The launch account

One decode step fires ~265 quantize launches × ~1.7 µs ≈ 0.46 ms/step. These launches waste twice: (a) the launch itself plus the f32 row's global round-trip (writing the q8 plane means first reading the f32 row back); (b) the same row is repeatedly quantized by adjacent matmuls (attn_norm output 3×/layer). Both follow from the layout choice of "hanging quantization on the consumer side".

### 2.2 Why not inline-per-block (the A-side amplification)

The most obvious form is to have each MMVQ row-block quantize its own A
rows. Dead on paper: every row-block would re-read the whole A row (f32),
multiplying **A-side L2 traffic by od × 14 KB** — one ffn_gu layer is on the
order of +18.6 GB/step. This is exactly r49's shared-A lesson: **the
quantization input is L2-hot to the producer, while the consumer is a
weight-streaming machine**; making the consumer go back and re-read A swaps
the cheapest read for the most expensive one. So the only correct landing
spot is the producer.

### 2.3 Constructive bitwise and the MmqCache window

The q8 plane's bit-for-bit invariance is argued constructively, requiring no
belief in numerical coincidence:

- **max is exact for any associativity** — `amax`'s `fmaxf` chain is the
  same number however associated;
- **rintf/clamp are per-element** — every byte of the quantized payload
  depends only on its own f32 value.

So as long as the epilogue reuses the standalone kernel's per-32-block body
(`quantize_pad40_block` verbatim), the q8 bytes are bitwise. After the
producer writes the f32 output it barriers, and the epilogue re-reads the
row it just wrote — **L1-hot**, cheaper than the standalone reading f32 from
global.

Cache correctness follows r49's MmqCache window rules: the plane is a pure
function of (src, nt, id); any non-(MatMul\|FusedFFN) node clears entries;
`synchronize` clears at execution boundaries. The hit path must also
**re-verify the physical pointer** (one `get_or_grow` between record and
consult may relocate the plane — an entry whose q8 moved falls back to the
standalone launch, re-quantizing from the live f32 src). FusedFFN joins the
cache-clear preserve set: its input is ffn_norm's rms output and its
internal gu matmul is that src's first consumer — the same "contiguous
consumer window" as MatMul→MatMul.
attn_o keeps the standalone quantize: its producer is the attention node,
untouched this session (regression-risk/benefit ratio unfavorable).

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **A shared epilogue body**: the `quantize_pad40_block` device function is
  simultaneously the standalone kernel's and both fused producers' body —
  "the epilogue IS the standalone body" is the code-level form of the
  bitwise claim.
- **The native (pad40, non-transposed) form**: decode MMVQ consumes the
  pad40 native plane (only prefill's MMQ wants the transposed pad40_t), so
  the producer records a native entry (`transposed=false`).
- **The opt-out ring**: `MINFER_NO_DECODE_A_FUSE=1` skips both consult and
  record, restoring a path bit-identical to pre-D3-5 — the A/B gate.
- **Caller-side gates**: the rms arm `n == 1 && d % 32 == 0`; the swiglu arm
  `n % 32 == 0` (whole 32-blocks; supported models' fused-FFN intermediate
  is always a multiple of 32).

### 3.2 Key code

**The shared per-32-block body** (current tree `src/cuda_kernels.cu`, the
standalone and fused epilogues share one body):

```cuda
// 40B layout: 2B f16 d, 2B pad, 32B int8 payload (offset 4),
// 4B i32 sum of the quantized values (offset 36 — the pad40 slack).
__device__ __forceinline__ void quantize_pad40_block(
    const float* __restrict__ src, uint8_t* __restrict__ dst
) {
    float4 sv[8];
    #pragma unroll
    for (int v = 0; v < 8; v++)
        sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
    float am = 0.0f;
    #pragma unroll
    for (int v = 0; v < 8; v++)
        am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                             fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
    float d = am / 127.0f;
    float di = (d != 0.0f) ? 1.0f / d : 0.0f;
    *reinterpret_cast<__half*>(dst) = __float2half(d);
    ...
    for (int v = 0; v < 8; v++) {          // rintf + clamp per element → bitwise
        const float* e = &sv[v].x;
        uint32_t p = 0;
        for (int j = 0; j < 4; j++) {
            int q = int(rintf(e[j] * di));
            q = max(-128, min(127, q));
            p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);
            s += q;
        }
        packed[v] = p;
    }
    ...
}
```

**The fused rms epilogue** (the rms body verbatim first, then a whole-block
barrier + re-reading the row it just wrote — L1-hot):

```cuda
    // epilogue: whole block arrives (all threads share `row`), then each
    // thread quantizes blocks tid, tid+blockDim.x, ... of its own row.
    __syncthreads();
    int nb = d / 32;
    const float* src = y + (size_t)row * d;
    uint8_t* dst = q8 + (size_t)row * nb * Q8PB;
    for (int b = tid; b < nb; b += blockDim.x)
        quantize_pad40_block(src + (size_t)b * 32, dst + (size_t)b * Q8PB);
```

**The fused swiglu** (before: the 6-line `swiglu_f32_off` body; after: the
body preserved verbatim + barrier + 8 threads quantizing the block's 8
output blocks):

```cuda
// Block bx wrote output elements [bx*256, bx*256+256) = quant blocks
// bx*8 .. bx*8+7, so 8 threads per block re-read them (L1-hot) and quantize;
// across the grid this is the same thread-count as the standalone kernel
// (one thread per 32-block). REQUIRES the 256-thread launch geometry.
__global__ void swiglu_quant_pad40(float* __restrict__ buf,
                                   uint8_t* __restrict__ q8, int n, int off) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {                                  // the body verbatim (guarded, no early return)
        float g = buf[tid];
        buf[tid] = (g / (1.0f + expf(-g))) * buf[off + tid];
    }
    __syncthreads();
    int b = blockIdx.x * 8 + (int)threadIdx.x;
    if (threadIdx.x < 8 && b < (n >> 5))
        quantize_pad40_block(buf + (size_t)b * 32, q8 + (size_t)b * Q8PB);
}
```

**The consumer-side consult (`decode_quantize_native`)**:

```rust
fn decode_quantize_native(&self, x: *const f32, id: usize, nt: usize) -> *mut u8 {
    let need = nt * (id / 32) * 40;
    let key = (x as usize, nt, id);
    if !Self::no_decode_a_fuse() {
        let cache = self.mmq_cache.lock().unwrap();
        if cache.active && !cache.transposed && !cache.dead_write
            && cache.key == key {
            let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as usize;
            if q8 == cache.q8 {                    // re-verify the physical pointer
                return q8 as *mut u8;              // hit: skip the standalone launch
            }
        }
    }
    let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as *mut u8;
    unsafe { launch_quantize_q8_0_pad40(x, q8, id as i32, nt as i32, stream); }
    if !Self::no_decode_a_fuse() {
        self.record_mmq_cache_native(key.0, nt, id, q8 as usize);
    }
    q8
}
```

All three decode MMVQ entries (q4_K/q6_K/q5_K classes) begin with this one
consult; on miss the behavior is exactly pre-D3-5 (launch + record). The
producer-side hook sits in the backend's rms/swiglu arms:

```rust
// D3-5 1a: decode (n==1) producers fuse the pad40 q8 epilogue — bit-identical
// f32 y and q8 bytes, one launch fewer per producer, and the following decode
// matmul group skips its standalone quantize.
if n == 1 && d % 32 == 0 && !CudaState::no_decode_a_fuse() {
    self.state.rms_norm_quant_on_gpu(x, wptr, out, d, n, eps);
    return Ok(());
}
```

### 3.3 Pitfalls

- **Barrier completeness under "no early return"**: the swiglu body of
  `swiglu_quant_pad40` was originally the `if (tid >= n) return;`
  early-return form — the fused version must become guarded (every thread
  reaches the barrier), otherwise threads in tail blocks return early and
  the `__syncthreads()` deadlocks (an inherent clause of the GPU safety
  rules).
- **Launch-geometry coupling**: `swiglu_quant_pad40`'s epilogue depends on
  the 256-thread-block invariant "block bx wrote [bx*256, bx*256+256)",
  stated as REQUIRES in the kernel comment — geometry and body must not
  evolve separately.
- **A new member of the cache window**: without FusedFFN in the preserve
  set, ffn_norm's rms plane would be cleared when the FusedFFN node executes
  and the gu matmul's consult would miss forever (functionally correct,
  optimization inert) — it must be treated like MatMul.

## 4. Verification

- **bitwise probe (defends against "the epilogue is not the standalone
  body")**: new test `cuda_decode_a_quant_fuse_bitwise`: fused vs standalone
  q8 buffers memcmp-equal at 14B hidden size and the ffn_down shape; f32
  producer outputs bit-identical; MMVQ matmul outputs bit-identical via the
  cache-hit path.
- **suite (defends against the regression surface)**: 170/0/3.
- **dump gate (defends against end-to-end numerical drift)**: 14B short
  prompt −n 4: `logits_{prefill,decode}` + all `kv*` byte-identical; the 3
  same-size node-dump diffs (`node{2,3,11}_decode`) are the calibrated
  pool-slot-aliasing class — reproducible both pre-vs-pre **and**
  post-vs-post (the same binary's diff set covers all of pre-vs-post's
  differences, i.e. no numerical delta).
- **greedy byte-for-byte (defends against sampling flips)**: the −n 256
  token stream byte-identical vs pre on both models (14B @2799-token prompt,
  7B @1847).
- **Operational note**: `prompt_3k3` (5451 tokens) panics at the n_ctx 4096
  default on **both** binaries (the headroom bug D3-4 calibrated) — until
  n_ctx sizing is fixed, long-prompt greedy gates must use ≤2.8K-token
  prompts.
- **nsys (defends against "the launch account is miscounted")**: 14B @3254
  (NO_CUDA_GRAPH, bench −n 8).

## 5. Results

### 5.1 1a (LANDED, `3230b2b`)

nsys launch account (14B @3254):

| Metric | pre | post | Δ |
|---|---|---|---|
| standalone `quantize_q8_0_pad40` launches | 4448 | **964** | **−78%** (the remaining ~50/step = the attn_o class) |
| total kernels | 29402 | 25918 | −3484 |
| sub-2µs launches | 15171 | 11723 | −3448 (= the quantize delta) |
| swiglu kernel | 2.06 µs | 2.31 µs | epilogue +0.25 µs, counted into the wall clock |
| fused rms | — | ~+0.2 µs | same |

Wall clock (same-window interleaved 3× medians, all SEP):

| Shape | pre | post | Δ |
|---|---|---|---|
| 14B tg128 | 23.05 | **23.35** | **+1.30%** |
| 14B @3254 | 21.36 | **21.68** | **+1.50%** |
| 7B tg128 | 49.86 | **50.69** | **+1.66%** |
| 7B @1641 | 48.47 | **49.16** | **+1.43%** |

Wins at **all lengths, both models** — the launch/round-trip tax is
length-independent, matching the prediction. Against the pre-D3-5 anchors:
14B tg128 +1.65% (session bar ≥ +1.5% cleared), @3254 +2.95%. Against
llama.cpp: 14B 0.961× (tg128) / 0.891× (@3254); 7B **1.026×** (tg128,
ahead) / 0.995× (@1641, parity) — **the 7B decode campaign closes**. 1a
itself reclaimed ~0.56–0.69 ms/step (the quantize launches' −78% plus their
launch gaps).

The distance account (post-1a, 14B @3254): minfer 21.68 t/s = 46.12
ms/step vs llama 24.32 = 41.12 ms → still 5.00 ms short (−10.8%). List:
attention residual 0.63 ms (Stage 2's GQA q-head batching); the attn_v-q6K
straggler 0.28 ms; the output head 0.44 ms (1b judged mechanism-free); the
rms/elementwise chain 1.0 ms (97 rms launches — the fused epilogue made rms
slightly heavier; the next elementwise lever is rms launch consolidation).
Together ≈ 2.35 ms = 47% of the remaining gap.

### 5.2 1b: output-head od-split / 512-thread rework (ANALYSIS-NEGATIVE, not implemented)

The brief's occupancy account, computed first: lm_head q6_K's npair=160 → 96
of 256 threads idle (the D3b-1c knob, already measured neutral); 6×256-thread
blocks/SM = 288 resident rows, while D3b-1c's 9×160 = 432-row form is
**also** neutral — rows-in-flight is not the limiter; the head sustains
47.6 blocks/µs while ffn_gu demonstrates 76/µs — block dispatch rate is not
the limiter either. The two-row 512-thread form could be bitwise (each
256-thread half-block reduces its own row with the same 8-warp tree), but it
moves none of the quantities **already measured neutral**. Per the brief's
own rule (pick the variant with a MEASURED mechanism), not built. The head's
200.1 vs the 220–225 GB/s class remains unexplained by any geometry knob —
the same class as D3b-1a's conclusion on the padded kernel (a latency/L2
composition, not a parallelism deficit).

### 5.3 1c: ffn_down-q6K 512-thread single-unit (ANALYSIS-NEGATIVE, not implemented)

Not bitwise against the landed `q6_k_q8_mmvq_v2_pf`: in the 256-thread form
thread t adds `fma(u_t) + fma(u_{t+256})` into **the same** float
accumulator before the block reduction; in the 512-thread form those two
units live in different threads and their sum moves into the 16-warp
cross-warp reduction order — a different float sum (the r50/r57 class). A
bitwise-emulating form (an smem pair-exchange so thread t still adds
u_t+u_{t+256} first) needs an extra barrier and buys zero
resident-parallelism gain (5120 rows = 18 waves in both forms); and the
exposed path 1c meant to treat is already fixed by v2_pf's dual-unit early
issue.

### 5.4 Follow-ups (the list this session left)

(1) attn_o keeps the standalone quantize (48/step, ~0.08 ms) — fusing it
into the attention combine epilogue would touch D3-4's hybrid kernel pair;
low value, high regression risk, skipped.
(2) Stage 2: GQA q-head batching (0.63 ms) is the largest remaining single
item.
(3) rms launch consolidation (97 rms → fewer, wider; the 1.15 ms class) is
the remaining elementwise mass.
(4) The long-prompt n_ctx headroom fix (pre-existing; still blocking
long-prompt greedy gates).

## 6. Lessons

1. **The correct landing spot for quantization launches is the producer**:
   the quantization input is L1/L2-hot to the producer and a global
   out-of-cache read to the consumer; inline-per-block makes the most
   expensive reader re-read A (the od×14 KB amplification), producer-fusion
   lets the cheapest reader write it in passing.
2. **Constructive bitwise precedes numerical verification**: max is exact
   for any associativity and rintf/clamp are per-element — write the
   epilogue as the standalone body's verbatim reuse and bitwise goes from
   "measured" to "proven"; the probe is just a recheck.
3. **A fused kernel's body fidelity includes control flow**: the original
   body's `if (...) return;` early return deadlocks under the new barrier
   semantics and must become a guard; "body verbatim" must preserve control
   flow too.
4. **Analysis-negative output is also numbers**: 1b's three candidate
   mechanisms (idle threads, resident rows, dispatch rate) each got a
   measured-neutral value attached, closing a branch more cheaply than
   building another kernel.

---
← 69 · [Index](./README.md) · 71 →
