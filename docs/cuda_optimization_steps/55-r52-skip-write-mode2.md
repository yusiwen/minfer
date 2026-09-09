# 55 · r52 — skip-write mode 2: skipping the f32 intermediate write-out (LANDED)

> **Result**: `MINFER_MMQ_A_FUSE=2` — the fused producers become **register-resident**, computing the pad40_t plane and **never writing the f32 output at all** (the `*_nw` kernels). fused rms 41.5 → 22.4 ms, fused swiglu 110.5 → 64.1 ms (−42%), fused producers total 151.9 → 86.5 ms, prefill window −6.1%; whole prefill **2855.7 → 3011.3 (+5.45%)**, vs-llama 1.16× → 1.09×. Before landing, the greedy gate caught an OOB transcription error that parity could not see.
> **Commit**: `910d967` (code, +534/−55) + `fb659f7` (record). **Date**: 2026-09-06.

## 1. Background — where things stood

r51 folded quantization into the producers, leaving the prepass at 28 launches and 10.1 ms, whole prefill +1.89%. But r51's post-landing wall decomposition lit up the next target brightly: **fused swiglu at 110.5 ms, 10% of the wall** — it reads gate and up at 251.9 MB each (7B's ffn intermediate dim is 18944), writes the f32 dst of 251.9 MB, and then phase 2 **re-reads those 251.9 MB** from L1/L2 to quantize. r51's record left an explicit hook: a register-resident quantize in phase 2 "could squeeze out another ~0.5–1 ms per launch."

One step further is the logical endgame: if quantization happens in registers, **does anyone still read the f32 output?** In the prefill graph, the only consumers of the rms/swiglu outputs are the immediately following MatMul groups, and after r51 those consumers read the **plane** (via r49's MmqCache) — the f32 buffer itself is no longer touched by anyone. The f32 write + L1/L2 re-read in mode 1 is a complete dead code path: written out, read back, and then — quantized once — never touched again. Mode 2 deletes that path wholesale: the producer computes the f32 values directly from its inputs (expressions verbatim-unchanged), completes quantization in registers, and writes only the plane.

The correctness conditions here are an order of magnitude harsher than mode 1's: **"nobody reads the f32 output" is not a kernel property, it is a graph-topology property**. Any missed reader (debug dump, trace, the legacy GEMM path, a non-adjacent consumer) would read a buffer that was never written — garbage. So the bulk of r52's work is not in the kernels (the kernels are actually simpler) but in the **window safety proof** and the **loud-failure mechanism for violations**.

## 2. Principle — the GPU mechanism

### 2.1 The traffic account that gets saved

Per-layer swiglu (7B, `nf=18944`, 3325 tokens):

| Path | DRAM/L1L2 traffic |
|---|---|
| mode 1 (r51) | read gate 251.9 + read up 251.9 + **write dst 251.9** + phase 2 **re-read 251.9** (hot in L1/L2) + write plane ~71 MB |
| mode 2 (r52) | read gate 251.9 + read up 251.9 + write plane ~71 MB |

That saves ~503.8 MB per layer of f32 write+read round trip (28 layers ≈ 14.1 GB/prefill of SM-level traffic; the re-read half mostly never hit DRAM anyway, but L1/L2 bandwidth and the store/load instructions are real money). rms is analogous: mode 1 pays a 47.7 y-write + 47.7 re-read, mode 2 is left with only the 47.7 x-read + plane write. Measured: swiglu 110.5 → 64.1 ms (−42%), rms 41.5 → 22.4 ms.

Note that what is saved is **traffic, not memory**: the f32 output buffer is still allocated by the graph allocator (it is still the MmqCache key and still the resolution target of the node's output buffer) — it is just never written.

### 2.2 Why the register-resident lane remapping is byte-safe

In mode 2 the quantization is no longer one thread serially walking 32 values; instead 8 lanes each compute 4 values, and an `__shfl_xor` tree regroups the amax and ssum. Regrouping usually means float reordering — r50 just taught us that is the source of ULPs — but here the **two quantities being reduced happen to both be exact**:

- `amax = max(|v0|…|v3|)` then the shfl tree: `fmaxf` is **exact** under any associative/commutative order (no NaN in this value domain);
- `ssum = Σq` (int) then the shfl tree: integer addition is **exact**.

And the f32 values feeding the quantization themselves: `v = xv * scale * wv` (rms) and `silu(g)*u` (swiglu) are **verbatim elementwise copies**, and `scale`'s reduction lane mapping is the same as mode 1's → identical bits. The element-level `rintf`/clamp are unchanged. So the plane is **byte-identical** to mode 1 / the standalone prepass — r50's lesson applied in reverse: regrouping only bites on inexact operations (f32 sum chains); max and int-sum can be reordered freely. This is also the theoretical basis for greedy-32 being byte-identical across `A_FUSE=1/2`.

### 2.3 The skip-write window conditions (why not writing is dared)

The f32 output buffer becomes memory **promised unread**. The promise is justified node by node, each condition paired with a concrete "what happens otherwise":

| # | Window condition | Guarantee mechanism | If violated |
|---|---|---|---|
| 1 | the only consumers of the rms/swiglu outputs are the immediately following consecutive plain MatMul groups | graph build order = execution order (builder topology); MmqCache's consecutive-window rule clears the cache at any non-MatMul node | a later node reads the f32 → reads an unwritten buffer |
| 2 | the residual adds the **pre-norm** buffer (x), not y | qwen2 topology: `attn_out + x_in`, `ffn_out + x_mid` — the add's inputs are the norm's **input** side | if the deleted y were referenced by a residual → garbage residual |
| 3 | the graph-tail G3 segment runs only `n_out=1` rows | that segment falls outside the fusion gate (rows ≥ 16) | — |
| 4 | `FusedQkv`/`FusedFFN` are decode-only | gated on `nt==1`; neither op appears in prefill graphs | — |
| 5 | `RmsNorm`/`SwiGLU` are not in-place ops | they do not appear in §5's aliasing rule, so the output buffer is not any other node's input | "skip-write" under aliasing would starve other readers |

Runtime degradation conditions stack on top of the static proof: any debug/trace reader present (`MINFER_GRAPH_DUMP` layer-0 node dump / `MINFER_DUMP_DIR` / `MINFER_TRACE` / viz live capture) or the legacy prefill GEMM enabled (its kernels read the f32 A directly under `MINFER_NO_PREFILL_GEMM=1`) → mode 2 **degrades to mode 1** (still taking r51's gains, just not skipping the write).

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**A three-tier degradation ladder**: mode 2 → (OOM / window readers / degradation conditions) → mode 1 → (OOM) → the unfused pair. Every tier is correct, and every tier either writes or doesn't write the f32; the caller falls back tier by tier. This determines that mode 2's failure mode is always "a performance loss," never "a wrong result."

**A loud backstop, not a quiet assumption.** The window proof is an argument, and arguments go stale — the moment the graph structure changes (say, some op starts reading the rms output), the proof silently expires. So r52 added a `dead_write` flag to MmqCache: mode 2 records its plane with `dead_write=true`; any path that would **re-quantize** that buffer (the transposed or native prepass) **refuses to execute** when it hits a dead_write entry, returning (0,0)/0, and the caller errors out the whole A path — a loud error replaces reading garbage.

**The key is unchanged.** The plane is still keyed on the (unwritten) f32 output pointer, exactly as in r51 — the consumer side `prefill_mmq` changes by zero lines, and the mode-1/mode-2 difference is sealed inside `mmq_a_fuse_mode()`'s return value and the two new kernels.

**The swiglu-nw mapping choice** is covered in §3.3 pitfall (b): coalesced rounds won over naive thread-per-chunk — this one was **measured**, not designed.

### 3.2 Key code

The mode-2 rms kernel (current tree `src/cuda_kernels.cu` lines 1037–1131). Phase 1 shares mode 1's lane mapping and reduction order (`scale` bit-identical), with **no y store**; phase 2 has each warp sweep its own row, with 8-lane-group exchanges of amax/ssum; padded tail rows take a dedicated zero-fill arm:

```cuda
__global__ void rms_norm_quant_nw_f32_t(
    const float* __restrict__ x, const float* __restrict__ w,
    uint8_t* __restrict__ yqs, uint8_t* __restrict__ ysda,
    int d, float eps, int n, int nchunk, int ntb
) {
    ...
    // Phase 1: identical to rms_norm_quant_f32_t — same lane mapping and
    // accumulation order over x, so `scale` is bit-identical. No y store.
    ...
        ss = warp_reduce_sum(ss);
        scale = rsqrtf(ss / (float)d + eps);
    ...
    if (row >= n) {
        // Padded-tail row: zero-fill this row's plane slots exactly like the
        // standalone prepass (deterministic plane regardless of scratch
        // reuse). grid = ntb*(64/RPB) covers every padded row.
        for (int b = lane; b < nchunk; b += WARP) { ... = 0; ... }
        return;
    }
    // Phase 2: quantize THIS warp's row (warp-uniform row => the shfl_xor
    // reductions below never see divergence). Chunk c = 4k + lane/8 covers
    // float4s 32k+lane; d % 256 == 0 makes d4 % 32 == 0 (exact loop).
    for (int k = 0; k < d4 / 32; k++) {
        float4 xv = x4[k * 32 + lane];
        float4 wv = w4[k * 32 + lane];
        // y expression verbatim from rms_norm_quant_f32_t phase 1 (bit-identical
        // f32 values — the mode-1 kernel quantizes these after a memory
        // round-trip, which is exact for f32).
        float v0 = xv.x * scale * wv.x;  ...
        float am = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        // 8-lane group reduce (lanes [g8*8, g8*8+8) own one 32-float chunk).
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));   // max: exact, safe to reorder
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        ...
        int ssum = (q0 + q1) + (q2 + q3);
        ssum += __shfl_xor_sync(0xffffffffu, ssum, 4);          // int-sum: exact
        ...
        const int b = k * 4 + (lane >> 3);   // chunk index (the pre-landing OOB was on this line)
```

The mode-2 swiglu kernel (current tree lines 1144–1210) — the **coalesced-rounds scheme**: lane `l` reads gate/up's float4 `f = round stride + threadIdx.x` (one 512 B access per warp, i.e. r51's phase-1 pattern), silu·mul completes in registers, and one 8-lane group is one chunk; `dim % 256 == 0` guarantees every 8-lane group maps onto **whole** chunks:

```cuda
    // Mode-2 swiglu: no dst write, single phase. Per token row, the block sweeps
    // the row's float4s in COALESCED rounds (lane l loads float4 rd*256+l of
    // gate/up — 512 B per warp access, the r51 phase-1 pattern), computes
    // silu*mul in registers, and quantizes via the rms-nw lane mapping ...
    // (A naive per-thread chunk mapping — one thread quantizing a whole 128-B
    // chunk — makes the gate/up loads lane-strided and measured SLOWER than
    // the mode-1 write+re-read.)
    for (int f = threadIdx.x; f < ...; f += blockDim.x) {
        const bool active = t < nt && f < n4;
        float v0 = 0.0f, ...;
        if (active) {
            float4 gv = g4[f]; float4 uv = u4[f];
            v0 = (gv.x / (1.0f + expf(-gv.x))) * uv.x;   // silu*mul verbatim
            ...
        }
        ... 8-lane shfl amax/ssum, same mapping as rms-nw ...
        if (f < n4) {
            // t >= nt rows land here with all-zero v/am/ssum — exactly the
            // standalone prepass's deterministic padded-tail zero-fill.
            ... write qs/swizzled + write d|ssum when u==0 ...
```

Host side: the mode-2 wrapper records with `dead_write = true` (current tree `src/cuda.rs` lines 3094/3129):

```rust
// the f32 output y serves only as the MmqCache key (the consumer matmul's
// A pointer); the buffer is promised dead
self.record_mmq_cache_transposed(y as usize, n, d, qa8 as usize, sda as usize, true);
```

The backstop: a re-quantize path that hits a dead_write entry refuses (current tree `src/cuda.rs` lines 2797–2811; the native path at lines 2855–2865 is isomorphic):

```rust
// r52: a mode-2 (skip-write) fused producer left the f32 src UNWRITTEN
// and promised its consumers would hit this cache. Reaching the launch
// path with a dead-write entry for the SAME buffer means the window
// guarantee broke ... re-quantizing would read the dead buffer's garbage.
// Refuse: the callers treat (0, 0) as a failed path and prefill_mmq
// errors out loudly instead of silently producing wrong results.
if cache.active && cache.dead_write && cache.key.0 == x as usize {
    eprintln!("minfer/cuda: MMQ A-quantize refused: mode-2 dead-write A ...");
    return (0, 0);
}
```

The scheduling end of the degradation ladder (current tree `src/graph/cuda_backend.rs` lines 539–566):

```rust
match self.state.mmq_a_fuse_mode() {
    2 => {
        if self.state.swiglu_quant_nw(...).is_ok() { return Ok(()); }  // mode 2
        if self.state.swiglu_quant(...).is_ok() { return Ok(()); }     // → mode 1
    }                                                                   // → unfused pair
    1 => { /* swiglu_quant, fall back to the unfused pair on failure */ }
```

Mode 2's enabling conditions (current tree `src/cuda.rs` `mmq_a_fuse_mode`; the reader-detection branch — the function in the current tree also carries r60's default-on modification; at r52 the semantics were that an explicit `MINFER_MMQ_A_FUSE=2` enables it):

```rust
2 if mode2_possible
    && std::env::var_os("MINFER_GRAPH_DUMP").is_none()
    && std::env::var_os("MINFER_DUMP_DIR").is_none()
    && !crate::trace::enabled()
    && !crate::live::enabled() =>
{
    2
}
// Mode 2 requested but a window-reader/fallback condition is
// active: keep the r51 fused semantics (write the f32 output).
2 => 1,
```

### 3.3 Pitfalls

1. **One transcription error, two wrong faces**. Before landing, rms-nw wrote the chunk index as `k*8 + lane/8` — the correct form is `k*4 + (lane>>3)`: each round k covers 32 float4s = **4** chunks (8 lanes/chunk × 4 float4s/lane), `lane>>3 ∈ {0,1,2,3}`, hence `b = k*4 + lane>>3`. Written as `k*8 + lane/8`, adjacent k rounds' b values overlap and the last round runs off the plane → **IllegalAddress**; and CUDA's async error only surfaces at the **next API call**, where it masqueraded as a string of fake "OOM"s. What caught it was greedy-32 (fork → investigate immediately), with **parity all green** — the parity fixtures never run the rms-nw path at all. Two lessons: when transcribing an index, first count how many chunks each round covers; and error masquerading lies — the first response to an error is to check the most recent kernel write boundary, not to trust the error's name.
2. **Register-resident ≠ faster; it depends on the store pattern**. The first swiglu-nw used the naive thread-per-chunk mapping (one thread quantizes a whole 128 B chunk): it saved the re-read, but the gate/up loads became lane-strided — the warp access shattered into non-coalesced segments. Measured, the fused swiglu went 110.5 → **114.0 ms** (slower than the re-read), and whole prefill only +1.39%. Only after switching to coalesced rounds did it reach 64.1 ms. r51's reason for rejecting register residency (non-coalesced f32 store = 8× sector amplification) re-established itself in a different spot under the nw shape: **warp-level 512 B coalesced access is the floor — whoever breaks it dies**.
3. **When the verifier breaks in an environment, swap the gate honestly**. r51's poisoned-plane standalone verifier (`valid_r51.cu`) **SIGBUS**es in this environment and cannot run. r52's byte-identity is instead covered by greedy-32 being byte-identical across `A_FUSE=1/2` (the two modes must produce exactly the same generation stream — this gate is sensitive to any kernel-level bit difference). The record states the downgrade explicitly, leaving no illusion of "verified."
4. **A promise needs a penalty clause**. The window proof covers every reader that exists today, but proofs don't stop the future; the `dead_write` refusal mechanism turns "the proof expired" from silently reading garbage into an error at startup — the maintainability of skip-write-class optimizations is entirely staked on this layer.

## 4. Verification

- **greedy-32 byte-identical across `A_FUSE=1/2`**: this doc's workhorse gate — both a proxy proof of kernel-level bit identity (§3.3 pitfall 3) and the catcher of the OOB transcription error. Guards against: the nw mapping changing bits, OOB, window violations.
- **parity ×3**: end-to-end numeric correctness — **but it has zero coverage of rms-nw** (the fixtures never go through that path), honestly noted in the record; this doc is the best object lesson in "which gate sees what."
- **suite 166/0/3**: the regression surface.
- **A/B interleaved measurement** (same window, same binary): +5.45% with distributions fully separated.
- **window safety proof** (the 5 conditions of §2.3, checked node by node): an argument, not a test — its staleness is backstopped by the `dead_write` mechanism.

## 5. Results

- **Kernel level** (whole-prefill totals): fused rms **41.5 → 22.4 ms**; fused swiglu **110.5 → 64.1 ms (−42%)**; fused producers total **151.9 → 86.5 ms**; prefill window **−6.1%**.
- **Wall clock** (7B @3325 tok): 2855.7 → **3011.3 (+5.45%)**, distributions fully separated; vs-llama **1.16× → 1.09×**.
- **Correctness**: parity ×3; greedy-32 byte-identical under both `A_FUSE=1` and `=2`; suite 166/0/3.
- **The mode ladder** (the quick reference for all subsequent A-FUSE behavior):

  | `MINFER_MMQ_A_FUSE` | behavior | f32 write? |
  |---|---|---|
  | `2` (window-safe) | `_nw` kernels: register-resident quantize | no (dead_write=true) |
  | `2` (readers/degradation conditions present) | r51's `_t` kernels | yes |
  | `1` | r51's `_t` kernels | yes |
  | `0`/other | unfused pair (producer + standalone quantize) | yes |
- **The signpost for the next round**: with the fused producers slimmed down, the wall's headliner returns to the q6_K GEMM — r45 (cp.async A staging) and r44 (W_exp), two mechanisms each measured "wall-neutral" alone, were waiting to be retried as a package, which is r53's basket thesis.

## 6. Lessons

1. **Skip-write safety = the completeness of the consumer enumeration**: prove the window node by node, degrade automatically when readers are present, and backstop proof expiry with a loud refusal mechanism — all three, or none.
2. **The greedy byte-identity gate sees corruption parity cannot**: the fixtures' path coverage defines parity's blind spots; the bit-identity gate is the only end-to-end gate sensitive to "any kernel-level bit difference."
3. **Async CUDA errors change faces**: IllegalAddress can masquerade as OOM — the first response to an error should be checking the most recent kernel write boundary, not trusting the error's name.
4. **Coalesced access beats register cleverness**: any remapping that costs a warp's 512 B coalesced shape should be costed at worst-case sector amplification before a line is written — r51's design veto and r52's measured retreat are the same physics.

← 54-r51-producer-fused-a-quantize · [Index](./README.md) · 56-r53-q6k-wexp-cpasync-bundle →
