# 72 · D3-7 — attn_v-q6K MMVQ routing (2b) + rms wide-block / positions memo (2c): the Stage-2 closing ledger (LANDED ×2)

> **Result**: **2b** — the Q6_K decode dispatch gate `od*id >= 24M` lowered to `>= 4M`, taking 14B attn_v (od 1024 × id 5120 = 5.24M, 11/48 layers) off the padded-f32 kernel: 33.16 → 24.32 µs (**−26.7%**, ~177 GB/s), ×11 layers ≈ 97 µs/step; attn_v and attn_o share an MmqCache hit, so the standalone quantize count is unchanged. **2c** — rms wide-block 32→128 threads (9.43 → 5.66 µs, −40%, −0.35 ms/step) + a positions_i32 per-execution-window memo (239.6 → **1.2** launches/step, −0.28 ms/step), both bitwise. Combined wall clock (SEP): 14B @3254 21.57 → **21.95 (+1.76%)**, tg128 +1.80%; 7B +1.04%/+1.07%; distance to parity −10.8% → **−9.7%**.
> **Commit**: `1088bc8` (2b, `src/cuda.rs` only, +11−1) + `b3b6077` (2c, `cuda_kernels.cu` + `graph/cuda_backend.rs`). **Date**: 2026-09-08.

## 1. Background — where things stood

The D3-6 session left this closing window two legacies. The first is negative: the attention residual was judged to be exposed latency and every bytes-side lever ruled dead (doc 71) — the largest single item on the Stage-2 list (0.63 ms) moved from "winning-able" to "bookkeeping". The second is deferred: 2b (attn_v-q6K → MMVQ routing) went untouched last session for lack of time, but its mechanism estimate had long been sitting in D3-1's census — attn_v runs the padded-f32 kernel at only **134.8 GB/s**, while its q6_K MMVQ siblings (ffn_up/down, attn_q/k/o) hold the 200–225 GB/s class; D3-1 estimated +0.28 ms/step.

So D3-7's (base `/tmp/d3/minfer_pre_d37`, sha1 b29f2ae6, HEAD 92b0712) positioning was very clear: **settle the two "small and certain" items on the list**, clearing a clean ledger for Stage 3's big move (the decode-GEMM program). Window anchors (pre_d37, 3× interleaved medians): 14B tg128 23.36 / @3254 21.65; 7B tg128 49.91 / @1641 48.31 (7B shows co-tenant drift relative to the D3-5 window, so everything is same-window interleaved A/B; sglang co-tenant present).

2c's working basis came directly from the decode launch census taken after D3-5 (14B @3254, decode-class launches):

| kernel | µs each | launches/step | ms/step |
|---|---|---|---|
| `rms_norm_quant_pad40` | 9.43 | 94.6 | **0.892** |
| `f32_bits_to_i32` | 1.15 | 239.6 | **0.276** |
| `add_bias` | — | — | 0.181 |
| `store_kv` | — | — | 0.179 |
| attention combine | — | — | 0.171 |
| `rope` | — | — | 0.139 |
| `add` | — | — | 0.139 |
| `swiglu` | — | — | 0.110 |
| standalone quantize | — | — | 0.084 |
| dud split launch | — | — | 0.070 |

Together ≈ **a 2.2 ms/step ocean of sub-6µs kernels**. The census also exposed one thing: the 14B decode qkv chain is **not fused** (per layer rope ×2 + store ×2 + bias ×3; FusedQKV is inactive on CUDA) — recorded as a front-row Stage-3 item ≈ 0.45 ms/step, which is doc 73's entry point. This session settles only the top two rows of the table (0.892 + 0.276); the rest is left to their respective larger levers.

## 2. Principle — the GPU mechanism

**2b — why the crossover gate wrongly killed attn_v.** The Q6_K/Q5_K decode dispatch gate from the 8e era is `od*id >= 24M`, based on the on-device micro-bench of the time (padded f32 vs mmvq): small shapes lose under MMVQ — od 512 → 4.5× slower, od 896 → 3.0×, 2048×4864 → 1.66× — because when each thread gets only 1–2 16-element units to amortize, the uncoalesced cost of the q5/q6 byte loads has nowhere to go; large shapes win under dp4a — 7B ffn_down 3584×18944 → 1.5× faster, lm_head 152064×3584 → 1.4×. But that crossover line was fitted through a handful of od points. Decode MMVQ's structure is **one 256-thread block per row** (the row's work split across npair 16-element units), so the longer the id, the more work per block and the flatter the amortization of the byte-load cost. attn_v is a **medium-od, long-id** shape (od 1024, id 5120 → npair 160 units/row → 1024 blocks × 256 threads) — exactly the class between the fitted points that the gate constant mis-killed. The measured −26.7% (~177 GB/s) confirms: the direction is right, but it still falls short of the 220-225 sibling class — consistent with the 8e data's small-shape warning at od≈1024, just not losing to padded-f32.

**Why the standalone quantize count does not grow.** D3-5 1a's MmqCache (producers record their q8 plane; later matmuls consult the cache, keyed by the producer's f32 output pointer) holds automatically under 2b's routing: attn_v's producer is the attention node (there is no fused quantize to speak of), but **attn_v and attn_o consume the same attention output buffer and the same id** — attn_v's `decode_quantize_native` arrives first (record), attn_o arrives later (hit). So 2b merely "moves one quantize that had to happen anyway earlier and shares it"; the standalone quantize launch count stays at 964 (trace-verified). The net gain is pure kernel-time difference: 33.16 → 24.32 µs × 11 layers.

**2c(i) — why the 32-thread rms is latency-bound.** `rms_norm_quant_pad40` in its original form is one 32-thread warp per row. hidden 5120 → d4 = 1280 float4s, i.e. **40 serial float4 loads per lane**, and no other warp on the SM to fill the load latency — pure latency-bound. Widening to 128 threads: 10 loads per lane + 4 concurrent warps, giving the load pipeline material to fill the holes. **Is bitwise preserved?** Yes, and the argument is structural: the reduction is still carried by lanes 0..31, the `i += WARP` stride keeps the original form's element→lane mapping and per-lane serial accumulation chain, and the `warp_reduce_sum` tree is untouched; scale is broadcast through smem to the whole block; the y write and the quantize epilogue are per-element / per-32-block independent, so the wider thread mapping **cannot** move any output bit. `#pragma unroll 8` only deepens the load pipeline. The remaining 5.66 µs is launch (~1.3 µs) + the epilogue's q8 write + the w-row stream — this item is already near the structural floor for a kernel of this class.

**2c(ii) — where the 240 redundant conversions come from.** Each graph layer has rope ×2, KvcacheStore ×2, Attn ×1 — 5 node classes consuming the same `positions` I32 input (f32::from_bits bit patterns, filled by `fill_input_i32`), and the CUDA backend calls `positions_i32(id)` at every consuming node to convert it to native int32 on device (the rope/store/attention kernels read `const int* positions`) — 48 layers × 5 = **240 of the same conversion per step**, 1.15 µs each, pure redundancy. The memo's semantic window is clean: within one graph execution, the same input buffer's bit patterns do not change (the host fills them before execution), so "one conversion per buffer per execution window" is mathematically identical. The key `(buf id, pool_gen)` handles free-list reuse: when the same id is returned and taken out again, `pool_gen` has changed and the cache invalidates naturally.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **2b changes the gate constant; no special-case routing.** The precondition was the GGUF census (`census_q6k.py`) first: sweep the q6_K weight shapes of all supported models and confirm that **within the (4M, 24M) interval only 14B attn_v exists** (7B attn_v is od 512 × id 3584 = 1.8M and stays on padded-f32, so the 7B stream is untouched byte for byte). With the census as backstop, `24_000_000 → 4_000_000` is the minimal change surface — no new dispatch branch, no new kernel, no new env var (the `MINFER_NO_KQ_MMVQ=1` semantics are kept as-is).
- **2c: census first, then act — and hit only the top two items.** In the sub-6µs ocean, add_bias/rope/store_kv are on the list too, but they fall in the blast radius of qkv-chain fusion (D3-8) — this session settles only rms (0.892 ms) and bits_to_i32 (0.276 ms), avoiding two mechanisms muddying one measurement window.
- **Both sub-levers must be bitwise.** 2b already introduced the first decode numeric freedom this campaign has tolerated (f32→q8 activation rounding, tolerance-gated per the D3a package); if 2c also changed numbers, the gate attribution would tangle. A bitwise 2c makes "measuring 2b in isolation" possible — wrap 2b on both sides with `MINFER_NO_KQ_MMVQ=1`, and every remaining diff belongs to 2c, whose diff must be zero.

### 3.2 Key code

**Excerpt A · the Q6_K dispatch gate (current tree `src/cuda.rs`, the landing spot of `1088bc8`)** — the before/after is one gate-constant line: `od * id >= 24_000_000` → `>= 4_000_000` (that commit's entire change is this plus comments, +11−1):

```rust
// src/cuda.rs — TensorType::Q6_K decode arm (current tree)
// Shape gate: see the Q5_K arm comment (measured od*id
// crossover ~24M elements; below it the padded f32 kernel's
// coalesced loop wins, above it MMVQ's dp4a wins).
// D3-7 2b: gate lowered 24M -> 4M for the attn_v class —
// the 14B attn_v (od 1024 x id 5120 = 5.24M, 11 layers)
// sat on the padded-f32 kernel at 134.8 GB/s (D3-1 census)
// while its q6_K MMVQ siblings sustain the 200-225 GB/s
// class; attn_v/attn_o share the attention-output buffer
// and id, so the D3-5 MmqCache dedupes their standalone
// quantize to one launch. GGUF census: no other q6_K shape
// falls in (4M, 24M) (7B attn_v 1.8M stays padded-f32).
// Tolerance-gated (f32->q8 activation rounding): D3a
// package; MINFER_NO_KQ_MMVQ=1 keeps the padded kernel.
if nt == 1 && id % 32 == 0 && od * id >= 4_000_000 && !Self::no_kq_mmvq() {
    self.q6_k_decode_mmvq(wptr, x, out, od, id, nt, padded_q6k);
    Ok(())
} else if padded_q6k {
    launch!(launch_q6_k_f32_matmul_padded)
} else {
    launch!(launch_q6_k_f32_matmul)
}
```

Once dispatched into `q6_k_decode_mmvq`, the shapes split naturally: id 5120 → npair 160 → the plain v2 kernel, 1024 blocks × 256 threads — no new kernel form was added for attn_v.

**Excerpt A2 · the MmqCache consult entry (current tree `src/cuda.rs`, head of `q6_k_decode_mmvq`)** — the "attn_v quantize moved earlier and shared" that 2b depends on happens at the entry of every decode MMVQ; in the current tree this spot also carries the post-D4-4 dpl fast path (later than this step, see doc 76), so the excerpt takes only the D3-5-era consult semantics:

```rust
// src/cuda.rs (current tree)
pub fn q6_k_decode_mmvq(&self, wptr, x, out, od, id, nt, blk_stride_padded) {
    // D3-5 1a: consult the MmqCache first — the fused decode producers
    // (rms_norm_quant_on_gpu / swiglu_quant_off_on_gpu) recorded their
    // pad40 plane, so the matmul group following the producer skips the
    // standalone quantize launch entirely (MINFER_NO_DECODE_A_FUSE=1
    // restores the unconditional standalone launch).
    let q8 = self.decode_quantize_native(x as *const f32, id, nt);
    /* ... the D4-4 dpl fast path and padded/v2 dispatch (later than this step) ... */
}
```

attn_v's `decode_quantize_native` records here (its producer is the attention node, so the cache misses → quantize on the spot and record), and attn_o's identical call then hits — 2b's "quantize count unchanged" is this one line of code serving two consumers.

**Excerpt B · the Q5_K arm — the code evidence for 2c's gate trap (current tree `src/cuda.rs`)**: `no_kq_mmvq()` does **not** control Q6_K only — the Q5_K decode arm reads it too (8e-era behavior). This is the mechanical root of "a one-sided env control fabricates drift":

```rust
// src/cuda.rs — TensorType::Q5_K decode arm (current tree; landed in 8e)
// Shape gate measured on-device (dbg micro-bench, padded f32
// vs mmvq): od*id < ~24M elements loses (od 512 → 4.5x
// slower, 896 → 3.0x, 2048x4864 → 1.66x) ...
// MINFER_NO_KQ_MMVQ=1 forces f32.
if nt == 1 && od * id >= 24_000_000 && !Self::no_kq_mmvq() {
    self.q5_k_decode_mmvq(wptr, x, out, od, id, nt);
    Ok(())
} else {
    launch!(launch_q5_k_f32_matmul)
}
```

**Excerpt C · rms wide-block (current tree `src/cuda_kernels.cu`, the kernel side of `b3b6077`)** — the reduction is lane-for-lane equivalent; only the block is widened from 32 to 128 (launcher: `rms_norm_quant_pad40<<<n, 128, 0, stream>>>`):

```cuda
// src/cuda_kernels.cu (current tree)
__global__ void __launch_bounds__(128) rms_norm_quant_pad40(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, uint8_t* __restrict__ q8,
    int d, float eps, int n
) {
    int row = blockIdx.x;
    if (row >= n) return;
    int tid = threadIdx.x;
    int d4 = d / 4;
    // D3-7 2c: wide-block geometry (launch picks 32 or 128 threads). The
    // reduction is bitwise-preserved: lanes 0..31 keep the exact 32-thread
    // form's element->lane mapping, serial per-lane accumulation order and
    // warp_reduce_sum tree; the unroll only deepens load pipelining. scale
    // reaches the whole block through shared memory. The write and quantize
    // loops are per-element / per-32-block independent, so their wider
    // thread mapping cannot change any output bit.
    __shared__ float s_scale;
    float scale;
    if (tid < WARP) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row * d);
        float ss = 0.0f;
        #pragma unroll 8
        for (int i = tid; i < d4; i += WARP) {   // serial chain in the original form's order
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = warp_reduce_sum(ss);
        scale = rsqrtf(ss / (float)d + eps);
        if (tid == 0) s_scale = scale;
    }
    __syncthreads();
    scale = s_scale;                              // broadcast to the 96 non-reducing threads
    /* write y: for (i = tid; i < d4; i += blockDim.x) — 128-way independent float4 writes
       quantize epilogue: for (b = tid; b < nb; b += blockDim.x)
                        quantize_pad40_block(src + b*32, dst + b*Q8PB); */
}
```

The Rust-side entry `rms_norm_quant_on_gpu` (`src/cuda.rs`) calls `record_mmq_cache_native(y, n, d, q8)` right after the launch — D3-5's MmqCache record point, and the foundation of 2b's "attn_v quantize moved earlier and shared" chain:

```rust
// src/cuda.rs (current tree)
pub fn rms_norm_quant_on_gpu(&self, x, w, y, d, n, eps) {
    let q8 = Self::get_or_grow(&self.buf_q8_decode, n * (d / 32) * 40);
    unsafe { launch_rms_norm_quant_pad40(x, w, y, q8, d as i32, eps, n as i32, stream); }
    self.record_mmq_cache_native(y as usize, n, d, q8 as usize);
}
```

**Excerpt D · the positions_i32 memo (current tree `src/graph/cuda_backend.rs`)** — key `(id, pool_gen)`, one conversion per execution window:

```rust
// src/graph/cuda_backend.rs (current tree)
fn positions_i32(&mut self, id: usize) -> Result<*mut std::ffi::c_void, String> {
    // D3-7 2c: one conversion per execution window per input buffer.
    // Capture mode: the first consumer's launch is recorded at capture
    // time and replay re-executes it every step (memo hits are never
    // recorded). Non-capture mode: synchronize() clears the memo at the
    // execution boundary, so each step re-converts exactly once.
    if self.pos_memo == Some((id, self.pool_gen)) {
        return Ok(self.pos_scratch);
    }
    let src = self.ptr_of(id)?;
    let bytes = self.pool[id].bytes;
    /* scratch growth path: after a fresh cuda_malloc, pool_gen += 1,
       which invalidates and re-captures any captured graph exec that may
       still embed the old scratch pointer */
    self.state.bits_to_i32(src, self.pos_scratch, bytes / 4);
    self.pos_memo = Some((id, self.pool_gen));
    Ok(self.pos_scratch)
}
```

Capture safety is a key design clause, with semantics for both modes: during CUDA Graph capture only the **first** consuming node's conversion launch gets recorded into the graph — memo hits never emit, and replay re-executes that one conversion each step; in non-capture mode `synchronize()` clears the memo at the execution boundary (`pos_memo = None`, right next to r49's MmqCache clearing point), guaranteeing the boundary of "one execution window". Sharing the clearing point also means the MmqCache and the pos memo always invalidate in lockstep — the half-state "cache alive, conversion not run" cannot occur.

**Excerpt D2 · the kernel the memo eliminated, in full (current tree `src/cuda_kernels.cu`)** — `f32_bits_to_i32` is 8 lines in total; the 0.276 ms/step buys its emission and scheduling × 240:

```cuda
// src/cuda_kernels.cu (current tree)
__global__ void f32_bits_to_i32(
    const float* __restrict__ src,
    int* __restrict__ dst,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = __float_as_int(src[tid]);
}
```

A kernel of this class spends its wall clock almost entirely on launch tax and the memory round trip — which is why the "once per execution window" memo converts 239.6 → 1.2 directly into time.

### 3.3 Pitfalls

- **The one-sided `MINFER_NO_KQ_MMVQ=1` control is a fake-drift factory.** The first 2c-only control set this env on the post side only, and the dump gate immediately showed a 0.22-logit "drift" (prefill+decode appearing together, depth-independent — not a propagating per-layer numeric difference but the whole path swapped). Mechanism: the env **also reverts the Q5_K decode arm** (excerpt B, pre-existing 8e behavior), so one side runs Q5_K through MMVQ and the other through f32 activations — the fake drift has nothing to do with 2b. The rule was therefore finalized: **control envs must be set on both sides**. From then on the 2c dump gate was double-sidedly wrapped throughout.
- **Co-tenant pollution cost a window.** 2b's tg128 extended measurement window was polluted on the post side by co-tenant activity; the whole window was discarded rather than forced into shape, and the landing evidence rests on the clean @3254 8-pair window (21.605 → 21.72, 7/7 pairs all positive, sign-test p≈0.008). The strict SEP differed by 0.05% (excluding one co-tenant rep: min-new-excl-outlier 21.66 vs max-base 21.67); per the D3-5 precedent it was carried on the median basis.
- **MmqCache depends on consumption order.** attn_v's producer is the attention node (not a fusable producer like rms/swiglu), so the dedup holds only if attn_v's quantize executes before attn_o — a same-buffer + same-id ordered consumption pair. The graph build order guarantees this order; if attn_o arrived first, the record/hit roles would swap with the conclusion unchanged; if the two ids differed, 2b would push the standalone quantize count above 964 — the trace-verified 964-unchanged is part of the gate.
- **The census counts are decode-class averages.** The 94.6/239.6 in the table are non-integers — they are the means over a 16-step trace (diluted by boundary effects like graph rebuilds), not per-step constants. Read the census mean-to-mean; do not check it against single-step integers.

## 4. Verification

- **2b (the D3a tolerance package, defending against f32→q8 activation rounding drift)**:
  - dump gate: `logits_decode` max|Δ| 0.254 (inside the calibrated 0.39-class), `logits_prefill` byte-identical; **argmax HARD gate green** (margin 1.915); kv0 bitwise, kv1+ showing decode-side f16-boundary drift — **appearing earlier than D3-6's sub-ULP reorder class, as expected**: input-quantization noise is ~4e-3 relative vs the reorder's 1e-7, so of course the former hits the f16 boundary first. The per-layer reading: layer-0's KV is written by prefill activations (this step does not touch prefill numerics) so it is bitwise; from attn_v rounding onward the decode-step hidden states carry a ~4e-3-class relative difference that amplifies with depth — the kv1+ drift gradient is exactly the shape of this propagation chain. The 0.39-class headroom from the D3a calibration package (h4w measured 6.5e-5 and incumbent 3.8e-5 on the outlier class with residual |q|~50 and V ±127) exists precisely for this class of input-quantization noise.
  - the greedy gate uses D3-6's newly established attribution method: **rp=1.0 (penalty waived) greedy −n 256 byte-identical on both models** (the clean kernel-numerics gate); the default-penalty stream = exactly one knife-edge event per 256 steps (5/5 seeds, coherent after the flip — in the mid-run repetition of "near the" → "as the", the penalized argmax was sitting among the repetition candidates anyway).
  - temp-0.8 control: 7B byte-identical, 14B reordered — top-p reordering under a 0.22-logit drift is expected and not a gate failure.
- **2c (bitwise end-to-end, defending against "bitwise by construction" becoming a slogan)**: dump gate with `MINFER_NO_KQ_MMVQ=1` set on both sides (after isolating 2b, 2c's diff must be zero): 109/114 files byte-identical, the 5 diffs = the documented pool-slot aliasing class; logits in both phases + all KV byte-identical; 7B greedy 5/5 seeds + temp-0.8 control byte-identical.
- **Suite 170/0/3** (the combined 2b+2c window; the full log was in `/tmp/d3/suite_2b2c_full.log` at the time and has since vanished with /tmp).

## 5. Results

**2b (nsys, 14B @3254, same window)**: attn_v kernel 33.16 → **24.32 µs** (−26.7%, ~177 GB/s weight stream — short of the 220-225 sibling class, consistent with the 8e small-shape crossover data's warning at od≈1024); ×11 layers ≈ **97 µs/step**; standalone quantize launch count unchanged (964). Wall clock: @3254 21.605 → 21.72 (**+0.42%**, 8-pair median, 7/7 clean pairs positive, sign-test p≈0.008); tg128 clean window +0.26% (below the +0.3% session landing bar; the extended window was discarded as co-tenant-polluted) — landed on the @3254 evidence + the measured mechanism.

**2c (nsys, reconciled against the same-window census)**: `rms_norm_quant_pad40` 9.43 → **5.66 µs** (−40%; 94.6–96 launches/step); `f32_bits_to_i32` 239.6 → **1.2** launches/step; wall-clock effect ≈ **−0.62 ms/step** (rms −0.348 + bits −0.275).

**Cumulative (2b+2c vs pre_d37, interleaved 3× medians, SEP strict)**:

| config | pre_d37 | post | Δ |
|---|---|---|---|
| 14B tg128 | 23.31 | **23.73** | **+1.80%** |
| 14B @3254 | 21.57 | **21.95** | **+1.76%** |
| 7B tg128 | 49.95 | **50.47** | +1.04% |
| 7B @1641 | 48.47 | **48.99** | +1.07% |

All guards hold (14B tg128 ≥ 22.7; 7B ≥ 49.0 / ≥ 47.9). Against the window anchors: 14B tg128 +1.59%, @3254 +1.39%. 2c's standalone contribution ≈ **+1.4%** (cumulative minus 2b's kernel projection of +0.21–0.3%), matching the census projection — where the census aimed and how much it hit, both ends reconcile.

**Distance to parity (post-D3-7, 14B @3254)**: 21.95 t/s = 45.56 ms/step vs llama 41.12 ms → **−4.44 ms (−9.7%)** (after D3-6 it was −5.00 ms / −10.8%). The Stage-3 front-row list (ordered by size):

1. **Matmul aggregation (~2.9 ms, the largest block)**: D3-1's wall-effective 194.9 GB/s vs llama's implied ~207.6 — the decode-GEMM program (q8_1-prologue fusion / llama-class MMVQ+GEMM rework) is the only lever class that reaches it.
2. **Launch-structure residual (~0.7–1.0 ms)**: led by the unfused qkv chain ≈ 0.45 ms/step (rope ×2 + store ×2 + bias ×3 per layer — a CUDA `attn_bias_rope_store` is needed), followed by the add_bias/add/store_kv/swiglu small-kernel tail and the dud split launch (~0.07 ms).
3. **Exposed-latency attention (≤0.63 ms, mechanism in doubt)**: the bytes side is ruled dead (D3-6); smem-tile cooperative staging is the untried remainder, but D2's cp.async negative results make this granularity historically disfavored.
4. **Mechanism-missing tail**: output head 200.1 GB/s (0.44 ms, no geometry knob at D3-5 1b), ffn_down-q6K 198.9 GB/s (0.49 ms).

7B decode stays closed out (tg128 1.021× vs llama, @1641 0.992× — ahead/even). The rest of the sub-6µs ocean (add_bias/store_kv/rope/add/swiglu ≈ 0.75 ms/step) was untouched by this step — it belongs to the qkv-chain fusion (doc 73) and larger levers; this census table is kept as Stage 3's comparison baseline.

## 6. Lessons

1. **census-first, census-close.** 2c's targets (0.892 + 0.276 ms) and its acceptance (239.6 → 1.2, 9.43 → 5.66) come from the same launch×µs table — the data decides where to strike, and the same table adjudicates whether the strike landed, with no second basis of measurement introduced.
2. **Read the side-effect surface of a control env in the code before designing the control.** `MINFER_NO_KQ_MMVQ=1` is nominally a Q6_K gate but actually also flips the Q5_K arm — a one-sided setting manufactured a 0.22-logit fake drift. Rule: **switch-type controls are set on both sides**, and prefer isolating variables with the double-sided diff's "must be zero".
3. **A crossover gate constant is not a universal constant.** `24M` was fitted through a handful of od points and wrongly killed the "medium-od, long-id" attn_v. Before changing a gate: run the shape census first (a full-model sweep found only this one shape in (4M,24M)), then let measurement adjudicate — analytic boundary + full sweep + small landing step; missing any one of the three invites a crash.
4. **Settle steps that add tolerance freedom separately from bitwise steps.** 2b (tolerance-gated) and 2c (bitwise) landed in the same window, but the gates were read under strict isolation: with NO_KQ_MMVQ wrapping 2b on both sides, 2c must show zero diff — clean attribution is a low-cost gate to maintain.

---
← 71 · [Index](./README.md) · 73 →
