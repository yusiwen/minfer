# 52 · r49 — A-quantize prepass shared-A dedup: consecutive-window memoization (LANDED)

> **Result**: the q/k/v attention GEMMs consume the **same** `normed` activation and gate/up consume the same `normed2`, but the MMQ path's A-quantize prepass previously **re-ran once per matmul** (193 launches on a 3325-tok prefill). `CudaState` gains a consecutive-window cache keyed on `(src device pointer, nt, id)`: on a hit the `quantize_q8_0_pad40_t` launch is skipped entirely. prepass **193 → 110 launches** (118.4 → **83.9 ms**, 9.6% → 7.4% of GPU busy), GEMM launches constant at 193 (only the redundant prepass is deleted, not the GEMMs); whole-prefill **2734.1 → 2797.5 tok/s (+2.32%)**, vs-llama 1.21× → 1.18×. Hits are byte-identical by construction (quantize is a pure function of `(x, nt, id)`); parity ×3 green, greedy-32 byte-identical, suite 166/0/3.
> **Commit**: `87a75a3` (`src/cuda.rs` + `src/graph/cuda_backend.rs`; **`graph.rs` untouched** — a hard rule). **Date**: 2026-09-06.

## 1. Background — where things stood

After r48 cashed in the first item of r47's priority queue (FAP2) at +5.6%, the second item in the queue was the prepass. This thread had been hanging in the campaign for three rounds:

- **r34** introduced the quantize-transpose prepass itself (`quantize_q8_0_pad40_t`, see doc 37): moving the A-side layout transform out of the kernel, +9.72% — pure profit at the time;
- **r37**'s attribution annotated it: "Quantize prepass 86.8 ms (**1.25× llama**, partly **2× per-shared-A redundancy**)" — part of the 25% over llama was exactly repeated quantization;
- **r47**'s re-decomposition pushed it to the front: the prepass had grown to **118.4 ms = 9.6% of GPU busy** (a +31.6 ms "hidden tax": the q6_K BT port wired the A sides of attn_v/ffn_down onto the same prepass), the third-largest slice of the wall, with a clear redundancy mechanism.

The redundancy's source is a **graph-topology fact**, not a kernel defect. Qwen2's 7 prefill matmuls per layer consume 4 distinct A inputs:

| matmul | A source | shared? |
|---|---|---|
| q, k, v | `normed` (attn rms output) | **3 GEMMs share 1 copy** |
| gate, up | `normed2` (ffn rms output) | **2 GEMMs share 1 copy** |
| attn_o | attention output | exclusive |
| down | swiglu output | exclusive |

The builder unrolls by dataflow, so `normed` is referenced once by each of the three matmul nodes; the MMQ path unconditionally re-ran the prepass before every matmul — of the 7 quantizations per layer, **3 were recomputations** (the last two of q/k/v + the last one of gate/up). Across 28 layers that is ~84 redundant launches, all of them the smallest, fastest kernels, and their launch overhead + repeated reads of x genuinely occupied 9.6% of GPU busy.

What one prepass launch actually does (r34's legacy, see doc 37): read the f32 activations `x` (`nt × id` floats), compute the amax per 32-element block, `rintf`-scale into q8_0, zero-pad to pad40, and produce two planes — the native form `q8` (`nt × (id/32) × 40` B, read block by block by the fallback kernels) and the transposed form `qa8 [ntb][nchunk][2048]` + `sda [ntb][nchunk][256]` (`ntb = ceil(nt/64)`, `nchunk = id/32`, used by the BT/NB kernels for bulk staging). The cost is proportional to `nt × id` bytes plus a launch's fixed overhead; on the 3325-token, d=3584 shape one such launch amortizes to ~0.6 ms (118.4 ms / 193). The price of doing it three times is not compute — it is pure repeated data movement and launch queuing.

One easily underestimated property of this post-r48 residual: it **modifies no math** — the quantized bytes are identical, and the only issue is that "the same bytes were computed three times." All of the risk in such a change lives in **cache freshness**: when may it hit, when must it be invalidated.

## 2. Principle — the GPU mechanism

**Purity is the linchpin.** `quantize_q8_0_pad40_t(x, qa8, sda, id, nt, ...)`'s output is a pure function of `(x, nt, id)` — per-block amax, `rintf` scaling, pad40 zero fill, no cross-call state whatsoever. So "a second request with the same `(src pointer, nt, id)`" and "re-running it" are indistinguishable at the byte level, and a hit is equivalent. This collapses the correctness question into one: **does an identical key guarantee identical input?**

**The semantic gap of the pointer key.** The key is `(src device pointer, nt, id)` rather than a buffer id or node id because the state layer only sees raw pointers. But the graph allocator (liveness allocator) **reuses pool buffer ids between nodes**: `normed`'s buffer may be reallocated as some other node's output after q is done with it, and later a fresh `normed` may land on **the same device address**. Same pointer ≠ same data — the key alone cannot distinguish "a second consumption of the same A" from "new data landed in an old address."

**Conservative invalidation rules close the gap.** r49's approach is to do no write tracking at all, and instead tighten the window with two rules:

1. **Valid only across "consecutive MatMul nodes"**: any non-MatMul node clears the cache when it executes. The graph executes in build order (topological order), so matmuls sharing an A are naturally adjacent in the builder's output; within the window no buffer can be rewritten (no other node runs). A late buffer-id reuse necessarily happens at some non-MatMul node — and that node has already cleared the cache.
2. **Clear at split boundaries / per execution**: cleared at `synchronize`. The next graph execution hands the same pool buffer ids to different data, and the `(src, nt, id)` key reproduces verbatim — residue across executions is the most dangerous kind of false hit.

Neither exception chases "one more hit"; both only chase "impossible to get wrong." The cost is real: non-adjacent same-A consumers (which do not exist in this campaign's graphs) are collateral-damaged into misses — the loss runs in the conservative direction, costing performance, not correctness.

**Three rejected alternatives in the design space**, recorded here so nobody "optimizes" back into them:

- **A cache persisting across executions** (keeping the planes by `(ptr, nt, id)` into the next execution): steps directly on the split-boundary problem — the same pool buffer id holds different data next execution, the key reproduces verbatim, and the false hit is a silent error. Infeasible unless content fingerprints are introduced (hash verification means reading the planes back, which costs more than recomputing);
- **Write-tracking invalidation** (hooking every allocator write, invalidating a cached pointer when written): requires the allocator to expose write events and welds cache semantics into a module that has nothing to do with it (alloc.rs is shared by all backends) — high complexity, hard to prove, all for "a slightly larger window";
- **Adding buffer id to the key**: the state layer cannot get the id (it sees raw device pointers), and threading the id down means changing the matmul dispatch signature — which violates the "graph.rs / dispatch layer untouched" hard rule.

Two lines of backend code bought all the correctness the first two alternatives were chasing — that is the engineering meaning of the "scheduling-window property" judgment.

**Why this is a "scheduling-window" problem.** The redundancy is not a property of the kernel (`quantize_q8_0_pad40_t` is beyond reproach) but a property of the **node stream** — the same input got scheduled three times inside a window. So the cache must take the form of "a window memo attached to the node stream," not "a constant cache attached to weights/model": it starts from zero on every execution, grows with the node stream, and is invalidated when the window closes.

**Why the scratch must live outside the pool.** The cache's output planes qa8/sda live in `CudaState`'s dedicated scratch (`buf_qa8_t`/`buf_sda_t` transposed, `buf_q8_prefill` native) — **outside the graph allocator's pool**. Inside the pool, the allocator could hand this "cache" to some node as its output, and the next layer writing it would clobber the cache contents; out-of-pool scratch guarantees no node output ever aliases the cached planes.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **The cache lives in `CudaState` (backend layer), `graph.rs` untouched** — this round's hard rule. Window semantics are the execution layer's scheduling knowledge; the graph-construction layer (shared by all backends) should not know whether some backend memoizes. `graph.rs` untouched also means zero risk to the CPU/Metal paths.
- **Key `(src ptr, nt, id)`, with a physical-pointer check after a key hit**: `get_or_grow` reallocs on a larger miss, so a cached plane's old address may be stale — the hit condition requires not just key equality but that the cached record's `qa8/sda` pointers match the actual pointers after growth. `nt` and `id` must be in the key: the same A pointer quantizes differently under different graph shapes (different nt) or different GEMM dimensions — the pure function's argument is the triple, not the bare pointer.
- **Both quantization forms go into the cache**: transposed pad40_t (`qa8` `[ntb][nchunk][2048]` + `sda` `[ntb][nchunk][256]`, for the BT/NB kernels) and native pad40 (`nt × id/32 × 40` B, for the fallback kernels), distinguished by the `transposed` flag — the two forms' plane contents differ, and the cache must not mix them.
- **No new env gate**: the dedup rides the `MINFER_MMQ` path, behavior is byte-level identical to not caching, and there is no semantic switch worth A/B-ing.
- **OOM front-loading**: the native plane's buffer is `get_or_grow`-ed once before the GEMMs, so an allocation failure errors at the matmul entry instead of landing on the q4_K fallback kernel's final launch as a null dereference.

### 3.2 Key code

**The cache itself** (`src/cuda.rs`, current tree; the `dead_write` field is r52's skip-write guard, not present in r49's original). Four invariants written into the struct's doc comment first — the code is just their implementation:

> 1. Valid only across **consecutive** prefill-MMQ MatMul nodes (any other node clears it);
> 2. Cleared at split boundaries / per execution (no leakage across executions);
> 3. The cached planes live in dedicated out-of-pool scratch (never alias node outputs);
> 4. The quantize output is a pure function of `(src, nt, id)` (a hit ≡ a recompute, byte-identical).

```rust
/// r49: consecutive-window memoization of the MMQ A-quantize prepass.
/// Correctness relies on two rules (both conservative, no write-tracking):
///   * It is valid ONLY across CONSECUTIVE prefill-MMQ MatMul nodes. Any other
///     node kind clears it (see `CudaBackend::execute_node_inner`), so a late
///     buffer-id reuse by the liveness allocator can never alias the cached A.
///   * It is cleared at split boundaries (`CudaBackend::synchronize`) so a
///     cache from a previous graph execution never leaks stale data into a
///     later one (the same pool buffer id holds different data each step).
/// The buffers are the state-level buf_qa8_t/buf_sda_t (transposed) and
/// buf_q8_prefill (native) scratch — dedicated allocations OUTSIDE the
/// graph allocator pool ... The quantize output is a pure function of
/// (src, nt, id), so a hit is byte-identical to a recompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MmqCache {
    active: bool,                     // whether an entry is recorded
    key: (usize, usize, usize),       // (src device pointer, nt, id)
    transposed: bool,                 // true: qa8_t/sda_t planes; false: native q8
    dead_write: bool,                 // r52: mode-2 skip-write entry (later round)
    qa8: usize, sda: usize, q8: usize, // physical addresses of the cached planes (validated on hit)
}
```

**The complete hit/miss path** (`mmq_quantize_transposed`, hit validation + miss recompute + record; r52's mid-section dead-write guard omitted):

```rust
let need_qa8 = (ntb as usize) * (id as usize / 32) * 2048;
let need_sda = (ntb as usize) * (id as usize / 32) * 256;
let key = (x as usize, nt as usize, id as usize);
let mut cache = self.mmq_cache.lock().unwrap();
if cache.active && cache.key == key && cache.transposed {
    // get_or_grow may have reallocated on a larger miss: validate the
    // physical pointers so a grown buffer is never reused stale.
    let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8) as usize;
    let sda = Self::get_or_grow(&self.buf_sda_t, need_sda) as usize;
    if qa8 == cache.qa8 && sda == cache.sda {
        return (qa8, sda);            // HIT: zero launches, the pointers ARE the last quantize result
    }
}
... (r52's dead-write rejection path)
let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8);
let sda = Self::get_or_grow(&self.buf_sda_t, need_sda);
launch_quantize_q8_0_pad40_t(
    x,
    qa8 as *mut u8,
    sda as *mut u8,
    id,
    nt,
    nchunk,
    ntb,
    stream,
);                                    // MISS: recompute verbatim
cache.active = true;
cache.key = key;
cache.transposed = true;
cache.qa8 = qa8 as usize;
cache.sda = sda as usize;
cache.q8 = 0;                         // r52 also clears the dead_write flag here
```

**The symmetric implementation for the native (non-transposed) form** — the pad40 plane (`buf_q8_prefill`) used by the fallback NB/wide/narrow kernels, same window rules, with the `transposed=false` gate keeping the two forms from colliding:

```rust
let need = (nt as usize) * (id as usize / 32) * 40;   // native pad40 plane size
let key = (x as usize, id as usize, nt as usize);
let mut cache = self.mmq_cache.lock().unwrap();
if cache.active && cache.key == key && !cache.transposed {
    let q8 = Self::get_or_grow(&self.buf_q8_prefill, need) as usize;
    if q8 == cache.q8 {
        return q8;                    // HIT
    }
}
... (r52's dead-write rejection path)
let q8 = Self::get_or_grow(&self.buf_q8_prefill, need);
launch_quantize_q8_0_pad40(x, q8 as *mut u8, id, nt, stream);   // MISS
cache.active = true;
cache.key = key;
cache.transposed = false;
cache.q8 = q8 as usize;
```

The consumer side is untouched — the matmul dispatch just feeds the returned `(qa8, sda)` pointers to the GEMM launcher (q6_K BT path, `src/cuda.rs`):

```rust
// r49: A-quantize prepass via the consecutive-window cache —
// a same-A (q/k/v, gate/up) matmul reuses qa8g/sdag without a
// fresh quantize launch.
let (qa8g, sdag) = self.mmq_quantize_transposed(
    x as *const f32, id as i32, nt as i32, nchunk, ntb, stream,
);
```

**The two anchors of window invalidation** (all of `87a75a3`'s changes to `src/graph/cuda_backend.rs`, 13 lines total):

```rust
// execute_node_inner: any non-MatMul node clears it (conservative — no write tracking)
if !matches!(&node.op, Op::MatMul { .. }) {
    self.state.clear_mmq_cache();
}
```

```rust
// synchronize: the cache never leaks across graph executions (the same pool
// buffer id holds different data on the next execution)
self.state.clear_mmq_cache();
```

(Later rounds added `Op::FusedFFN` to the preserve set — its input plane was just recorded by the fused rms epilogue and its internal gu matmul is the first consumer, so the window semantics match MatMul→MatMul; that was D3-5's business — r49's rules are exactly the two above.)

**OOM front-loading** (the matmul entry first guarantees the native plane is allocatable):

```rust
// r49: the native pad40 buffer is sized upfront so an OOM surfaces here
// (before any GEMM launch) instead of as a null deref in the q4_K
// fallback's final `launch_mmq_raw_nt`.
if Self::get_or_grow(&self.buf_q8_prefill, nt * (id / 32) * 40).is_null() {
    return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
}
```

### 3.3 Pitfalls

1. **Parity cannot reach the HIT path**. The parity harness clears the cache between cases, so all three parity tests exercise only the miss path — the most correctness-critical part of caching (hit equivalence) is exactly what the unit gates cannot reach. r49's solution was to hand the HIT path to **greedy-32 byte-identical**: in a 32-step greedy generation every layer's q/k/v and gate/up go through the hit path, and any false hit shows up in the byte stream. Lesson: **pair a "cannot-be-wrong" construction with an end-to-end identity gate that can actually reach it**.
2. **Hit validation must check the physical pointers, not just the key**. `get_or_grow`'s realloc semantics mean that with an identical key the cached plane may have been moved/grown — a key hit + matching physical pointers is a real hit. Skipping this layer reads a stale address in a "small prefill first, then a large prefill" session.
3. **Launch counts and milliseconds are two different ledgers**. The 83 deleted launches were all narrow width-3584 planes (the duplicate copies of normed/normed2); the 110 kept ones include 28 wide width-18944 planes (swiglu output) — launches dropped to 57%, duration only to 71% (118.4 → 83.9 ms). Estimating the time win from the launch share would overestimate by nearly half (predicted −50.9 ms vs actual −34.5 ms): the deleted launches happened to be the smallest batch.
4. **Do not force-fit the theoretical account to single digits**. 4 distinct A copies per layer × 28 layers = 112 expected surviving launches; 110 measured — the 2-launch gap comes from graph-level tails (the lm_head input after the final norm, etc.) and a few layers' shape differences. Argue the mechanism with **ratios** (0.571 predicted vs 0.570 measured; time-weighted 0.734 vs 0.709 measured), not by reconciling launch by launch — failing to match single digits is not evidence of a bug, it is the limit of model granularity.

## 4. Verification

| Gate | Numbers | What it defends against |
|---|---|---|
| launch census (nsys) | prepass 193 → 110, **GEMM constant at 193** | the mechanism gate — proves only the redundant prepass was deleted and the GEMM side is untouched (if the GEMM count had changed too, real work was deleted by mistake) |
| `cuda_prefill_mmq_parity` | 1/0 green | MMQ numeric regression (exercises the **miss** path — the parity framework clears the cache between cases), catching a broken recompute path |
| `cuda_prefill` | 7/0 green | whole-graph prefill numerics, catching any downstream drift introduced by the cache (also miss-only) |
| `cuda_fa_prefill_attention_parity` | 1/0 green | FA and MMQ coexisting in one graph, catching this round's changes spilling onto r48's freshly landed attention |
| greedy-32 byte-identical | byte-for-byte | **the HIT path's end-to-end identity gate** — every layer's q/k/v, gate/up hits many times; any false hit or staleness hole shows up in the byte stream (the path parity cannot reach, see pitfall 1 in 3.3) |
| suite | 166/0/3 | full regression, including the CPU path — structural corroboration that `graph.rs` was untouched |
| A/B interleaved ×3 median | 2734.1 → 2797.5, distributions fully separated | +2.32% clears the +1.5% bar, catching machine-state noise being read as a gain |

## 5. Results

| Metric | before | after | Δ |
|---|---|---|---|
| prepass launches (3325-tok prefill) | 193 | **110** | −83 (−43%) |
| prepass duration | 118.4 ms (9.6% busy) | **83.9 ms** (7.4%) | −34.5 ms |
| GEMM launches | 193 | 193 | unchanged |
| whole-prefill (same-window interleaved ×3 median) | 2734.1 tok/s | **2797.5 tok/s** | **+2.32%** |
| vs-llama (3325-eq anchor) | 1.21× | 1.18× | −0.03× |

**The mechanism's two-ratio cross-check** (7 matmuls / 4 distinct A copies per layer):

- launch ratio: theoretical 4/7 = 0.571 → 193 × 0.571 ≈ 110.3, measured **110**;
- duration ratio: linearly weighted by plane width, the surviving share = (3×3584 + 18944)/(6×3584 + 18944) = 0.734 → 118.4 × 0.734 ≈ 86.9 ms, measured **83.9 ms** (−3%, made up by the omitted fixed launch overhead).

Both independent ratios lock onto the measured values — what was deleted really is the single thing "3 duplicate A copies per layer," with no other effect mixed in. The wall clock realized ~28 ms (the wall change corresponding to +2.32%), slightly less than the 34.5 ms of kernel-time savings — the typical discount when launch-level savings cash into wall time (the same pattern recurs in r51: prepass 83.0 → 10.1 ms bought wall +1.89%).

Follow-on coordinates: the prepass line ran to its end in r51/r52 — producer fusion cut launches from 110 to 28 (prepass 83.0 → 10.1 ms), and skip-write mode then waived the f32 output writes entirely (fused producers 151.9 → 86.5 ms); r49's MmqCache and its window rules became the foundation of both rounds verbatim (the fused producers record their quantized planes straight into the same cache). This cache later grew a second job as well:

| Later round | Increment on MmqCache | Numbers |
|---|---|---|
| r51 (doc 54) | fused rms/swiglu **records directly into** the cache (`record_mmq_cache_transposed`), leaving the prepass only wo | prepass 110 → 28 launches, 83.0 → 10.1 ms; +1.89% |
| r52 (doc 55) | mode-2 skip-write: the f32 output write is waived, and the `dead_write` guard turns window violations into loud errors | fused producers 151.9 → 86.5 ms; +5.45% |
| D3-5 1a | the decode-side isomorph (`record_mmq_cache_native`, MmqCache consult skipping the standalone quantize) | standalone quantize 4448 → 964 launches (−78%) |

A small change that began as "don't quantize the same A three times" eventually grew into the backbone of the A-quantize supply line on both the prefill and decode sides — the conservative window semantics (key + two invalidation rules) did not change by a single word from r49 through the D series; that is the compound interest of "invariants first."

## 6. Lessons

1. **Redundant work on shared inputs is a scheduling-window property**: memoize on the node stream — the key (pointer+shape) buys the fast path, and conservative window invalidation (any foreign node clears + execution-boundary clears) buys correctness; no write tracking, and prefer a missed hit over a wrong one.
2. **Swap the gate when the unit harness cannot reach the path**: cache-hit equivalence cannot enter the parity framework that clears caches, so end-to-end greedy byte-for-byte identity is the only verification that reaches it — assign gates by "which gate can exercise which path," not by piling gates on.
3. **The backend layer's scheduling knowledge stays out of the graph-construction layer**: zero changes to `graph.rs` made the dedup structurally risk-free for the CPU/Metal paths; window semantics are the executor's private property.
4. **Launch count and duration are two different profit ledgers**: what gets deleted tends to be the smallest launches (shared narrow planes), and what stays tends to be the wide planes — weight estimates by bytes, not by launch counts.

---
← [51 · r48 FAP2 register-resident softmax](./51-r48-fap2-register-softmax.md) · [Index](./README.md) · [53 →](./53-r50-fa-tkv-16.md)
