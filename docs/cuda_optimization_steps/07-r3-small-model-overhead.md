# 07 · R3 — Small-model per-token overhead: single-split prefill (LANDED)

> **Result**: of 0.5B decode's ~4.0 ms/token, ~2.4 ms of non-GPU overhead was located and dismantled item by item — prefill collapsed from 4 splits into a single CUDA split (A1), the D2H logits readback moved to a self-held pinned buffer and the per-step clone was cut (A2), prefill capture flipped default-on (B); greedy output bit-identical throughout. Bench recorded under ~96% co-tenant load: the pinned path parity to slightly ahead.
> **Commit**: `029a9a4` (A1) · `a213c89` (A2) · `761e236` (B). **Date**: 2026-08-31.

## 1. Background — where things stood

The state at the close of Era A (Phase 7/8): 7B @2K prefill reached ~1204 via 8m's wmma GEMM and
was pushed to ~1400–1500 tok/s by 8p's resident f16 weight cache; on the decode side, 8e's MMVQ
(dp4a × q8_0) gave q4_K +37%, and 8o cut decode-start's 635 ms heavy clone to 35 ms. The
big-model numbers were moving, but the 0.5B small model exposed another wall: **the GPU accounts
for only a minor share of each token's wall time**. The `MINFER_GRAPH_TRACE` + DOT dump profile
shows 0.5B decode at ~4.0 ms/token, of which the GPU floor is only ~1.6 ms — the remaining ~2.4
ms is CPU/synchronization overhead. 1000/4.0 ≈ 250 tok/s, matching Part-I's recorded 0.5B decode
~257 tok/s (llama 453) exactly: no matter how much faster the kernels get, this overhead caps
them in place.

Small models are a magnifying glass for this wall, for a direct reason: GPU kernel time scales
with model size, while the per-step fixed costs — split-boundary syncs, host↔device round trips,
the driver bounce of pageable readbacks, redundant clones, the launch structure — do not scale.
The smaller the model, the larger the fixed costs' share; on 0.5B it is 60% of the wall. The
same fixed costs also surface on large models with short prompts (small nt, thin GPU work), so
this is not a "small-model-only" corner case — it is the graph-execution architecture's bill.

Why fix it now: first, the graph architecture's core claims (declarative build → single-pass
execute, CUDA Graph capture/replay) must hold at every scale, and a 4-split prefill on 0.5B
plainly violates the "single-pass" promise; second, R3-B (prefill capture default-on)
presupposes that prefill is one continuous split — with a host fill point in the middle of the
split, capture can only wrap the body segment. Without this step, all later "whole-graph replay"
dividends (launch-overhead elimination in server scenarios) have no footing.

The trace gave three targets, ordered "structural → constant → policy":

- **A1**: the G3 tail-row reduction's input `tail_ids` was declared **mid-graph** (right before
  its consumers in the last layer), slicing every prefill forward into 4 splits (inputs | body
  | tail_ids | tail) — each forward pays 2 extra full-stream syncs + host round-trip copies;
- **A2**: every decode step's logits readback goes through a blocking `cudaMemcpy` into a
  **PAGEABLE** Vec (the driver first bounces into its own pinned bounce buffer), and
  `forward_graph` then `.to_vec()`-clones logits that are already exactly n_out*nv;
- **B**: 8g② had made prefill capture a deliberate opt-in (default off), so the server's
  repeated same-shape prefills got no benefit.

## 2. Principle — the GPU mechanism

**Why a mid-graph input = a split.** The scheduler executes nodes in build order (topological
order); when it meets an input node that needs host filling, it must first let all queued async
work on the stream complete (full-stream sync), copy the data from host into that input buffer,
and only then continue encoding later nodes — this "stop, sync, fill, continue" point is a
CPU/CUDA boundary. When inputs are declared at the graph head, filling happens before execution
begins, none at all; declared between two CUDA ops, every such input adds one more cut to the
forward. The pre-R3 prefill graph was:

```
[inputs | body | tail_ids | tail section]   ← 4 splits
          ↑ 2 extra full-stream syncs + host round-trip copies per forward
```

The crux: **node order is not semantics** — `tail_ids`'s consumers merely reference the handle;
declaring it at the graph head and consuming it at the tail leaves the dataflow completely
unchanged while removing the mid-execution fill point from scheduling. That is what "input
declaration position is graph topology" means: declaration position decides split boundaries
even when semantics are equivalent.

The cost arithmetic: 0.5B's GPU floor is ~1.6 ms/step, and each of the two full-stream syncs
drains the entire pipeline — the CPU wakes up, does bookkeeping, fills data, re-encodes; that
chain is ms-scale on a machine with a busy co-tenant. **The extra syncs are the same order of
magnitude as the entire GPU step** — the largest single source of the 2.4 ms overhead on small
models (on the prefill side).

**Why a pageable D2H readback is slow.** For a blocking `cudaMemcpy` into pageable (plain
malloc) memory, the driver has no stable bus address available, so it must first use its
internal **pinned bounce buffer** as intermediary: device → driver-internal pinned buffer → copy
into the caller's pageable destination — two DMA hops plus a possible staging allocation each
time. 7e⑥ already paid this tuition on the H2D direction (`write_input_async`) and switched to a
self-held pinned ring; **the D2H direction never got the same treatment** — the logits readback
is precisely the D2H that runs every decode step. After switching to a `cudaHostAlloc` self-held
buffer, the destination address is DMA-reachable, one hop direct, and reading out afterwards is
just an ordinary CPU memory copy. The clone is cut in passing: the graph output is already
exactly n_out*nv (reduced by G3, or n_out==nt), so the 608 KB (151936 vocab × f32) `.to_vec()`
per step is pure waste.

**Why capture has a 3-run protocol.** CUDA Graph's benefit model: a one-time cost (capture +
instantiate, ms-scale) buys the per-launch CPU cost down to zero (one `cudaGraphLaunch` for the
whole graph). A 437-node prefill graph's per-launch CPU cost is substantial, but for a CLI
prefill that **runs only once**, capture is a pure loss. The 3-run protocol: capture only when
the same (uid, nt) graph appears a 3rd time — one-shot calls never reach 3 and pay nothing;
repeated same-shape prefills like a server slot start netting a profit after the 3rd. A1's
single split is the precondition: a capture window must contain no host fill point, otherwise it
can only wrap the body segment (exactly the crippled form before 8g②).

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**A1 — move the input to the head rather than eliminating the input.** `tail_ids` carries G3's
tail-row reduction (`get_rows` the n_out rows before the last layer's FFN so ffn/lm_head compute
only the output rows — llama's `ggml_get_rows(cur/inpSA, inp_out_ids)` semantics); the data
dependency itself must stay. The chosen form is `.then(|| b.input(...))` under the `params.n_out
< nt` condition: the condition is identical to before, and when n_out == nt the input node
simply does not exist — the graph's determinism is decided by params (part of the reuse
identity), and the declaration-position change does not touch that invariant. The decode graph
never had `tail_ids` (nt==1 does not trigger the tail-row reduction), so the decode graph
changed not at all.

**A2 — a blocking memcpy into self-held pinned, not a switch to async.** The readback point's
semantics are "I want the result now": the caller has already `sync()`ed the stream. The gain
comes from **skipping the driver's bounce**, not from overlap — so the shape is the simplest
blocking memcpy + pinned destination, not an async chain with events/callbacks. The buffer is
grow-on-demand: the first read allocates `dst.len().max(4 MiB)` (4 MiB of headroom prevents
small size jitter from churning alloc/free), and later only growth triggers reallocation; if
`cudaHostAlloc` fails, silently fall back to the pageable path and warn once.
`MINFER_NO_PINNED_READBACK=1` keeps the A/B switch. The clone cut is defensive: `forward_graph`
returns the readback buffer directly only when its length already equals n_out*nv — "always
true", but a shape check is kept rather than betting on structure.

**B — flip the default rather than add a new mechanism.** 8g②'s verification assets (the
pp16/pp300 bit-parity harness) already existed; all R3-B did was flip the switch's default:
`MINFER_CAPTURE_PREFILL=1` is redundant but still accepted, `MINFER_NO_PREFILL_CAPTURE=1`
restores the old default. No new protocol, no change to the 3-run threshold.

### 3.2 Key code

**A1: the input declaration moves from mid-graph to the graph head**
(`src/models/qwen2/graph.rs`, commit `029a9a4`). Before — the input declared right beside its
consumers, inserted before the last layer's FFN:

```rust
let wo = b.matmul(attn_out, l.wo.as_ref().unwrap(), None);
let is_last = il == model.layers.len() - 1;
if is_last && params.n_out < nt {
    // G3: reduce to the tail n_out rows BEFORE the last layer's FFN
    // (llama `ggml_get_rows(cur/inpSA, inp_out_ids)` at
    // qwen2.cpp:106-108) — ffn_norm, gate/up/down, swiglu, both
    // residuals and lm_head all run on n_out rows only.
    let tail_ids = b.input(                 // ← the input node appears mid-graph:
        "tail_ids",                         //    every CUDA op before it
        [params.n_out, 1, 1, 1],            //    gets cut by this boundary
        crate::graph::DType::I32,
    );
    let cur_tail = b.get_rows(wo, tail_ids, [ne, params.n_out, 1, 1]);
    let res_tail = b.get_rows(residual, tail_ids, [ne, params.n_out, 1, 1]);
    h = b.add(res_tail, cur_tail);
```

After — the same input declared at the graph head beside `token_ids`/`positions`, condition
unchanged:

```rust
let inp_ids = b.input("token_ids", [nt, 1, 1, 1], crate::graph::DType::I32);
let inp_pos = b.input("positions", [nt, 1, 1, 1], crate::graph::DType::I32);
// G3 tail-row reduction input, declared at the graph HEAD (not beside
// its consumers at the last layer): an input node mid-graph splits the
// forward into extra CPU/CUDA boundaries (2 full-stream syncs + host
// round-trip copies per step on the split path). R3-A1.
// Node order is not semantics — the consumers below just reference the handle.
let tail_ids = (params.n_out < nt).then(|| {          // condition verbatim-identical to before
    b.input(
        "tail_ids",
        [params.n_out, 1, 1, 1],
        crate::graph::DType::I32,
    )
});
```

The consumption site changes by one line — unwrap the handle from the Option; the reduction
logic untouched:

```rust
if is_last && params.n_out < nt {
    let tail_ids = tail_ids.expect("tail_ids input declared when n_out < nt");
    let cur_tail = b.get_rows(wo, tail_ids, [ne, params.n_out, 1, 1]);
    let res_tail = b.get_rows(residual, tail_ids, [ne, params.n_out, 1, 1]);
    h = b.add(res_tail, cur_tail);
```

**A2: the pinned D2H readback** (`src/cuda.rs`, commit `a213c89`). Self-held buffer +
grow-on-demand + failure fallback; the core path:

```rust
pub fn copy_from_device_pinned(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
    static FALLBACK_WARNED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if std::env::var("MINFER_NO_PINNED_READBACK").as_deref() == Ok("1") {
        self.copy_from_device(src, dst);            // A/B switch: the old pageable path
        return;
    }
    // headroom so small size changes don't churn the allocation
    let need = dst.len().max(4 * 1024 * 1024);      // ≥4 MiB, guards against small-size jitter
    let mut guard = self.readback.lock().unwrap();
    if guard.as_ref().map_or(true, |b| b.bytes < need) {
        if let Some(old) = guard.take() {
            drop(old); // cudaFreeHost
        }
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaHostAlloc(&mut p, need, 0) };
        if err != 0 {                               // alloc failed → pageable fallback,
            drop(guard);                            // warn only once
            if !FALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "CUDA: pinned readback alloc failed (err {err}); pageable D2H fallback"
                );
            }
            self.copy_from_device(src, dst);
            return;
        }
        *guard = Some(PinnedBuf { ptr: p as *mut u8, bytes: need });
    }
    let buf = guard.as_mut().unwrap();
    unsafe {
        cudaMemcpy(                                 // destination is pinned: one DMA hop,
            buf.ptr as *mut std::ffi::c_void,       // no driver-internal bounce
            src,
            dst.len(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );
        std::ptr::copy_nonoverlapping(buf.ptr, dst.as_mut_ptr(), dst.len());
    }                                               // reading out is just an ordinary CPU memcpy
}
```

The backend-side call site (`copy_to_host` in `src/graph/cuda_backend.rs`); the clone cut pairs
with it:

```rust
// R3-A2: read through the pinned staging buffer (pageable-memcpy
// bounce removed); MINFER_NO_PINNED_READBACK=1 reverts.
self.state.copy_from_device_pinned(b.ptr, dst);
```

**B: the capture default flip** (`src/graph/cuda_backend.rs`, commit `761e236`):

```rust
-        let prefill_capture = std::env::var("MINFER_CAPTURE_PREFILL").as_deref() == Ok("1");
+        let prefill_capture = std::env::var("MINFER_NO_PREFILL_CAPTURE").as_deref() != Ok("1");
```

The 3-run protocol's gate itself is unchanged; only `prefill_capture`'s initial value changes:

```rust
if *runs >= 3
    && self.capturing.is_none()
    && nt_hint.map_or(true, |nt| nt == 1 || self.prefill_capture)
```

### 3.3 Pitfalls

- **Input nodes serve readability and pay in scheduling.** G3 declared `tail_ids` beside its
  consumers at the time — it reads nicely in source, but an input node's declaration position
  is scheduling metadata. This class of pitfall has no compile-time signal whatsoever: the
  graph still topo-validates, execution is still correct — just slow.
- **Env vars are process-global, and the suite runs tests in parallel.** R3-B changed 8g①'s
  "prefill never captures" negative test to drive the opt-out through
  `set_prefill_capture_for_test(false)` — had the test kept setting the env var, it would
  cross-contaminate other configurations inside the parallel test processes.
- **`cudaHostAlloc` can fail.** The pinned pool is not an infinite resource; the failure path
  must silently fall back to pageable and warn only once (the `FALLBACK_WARNED` static bit),
  otherwise every decode step sprays a stderr line.
- **The measurement window was occupied by a co-tenant.** This session's bench ran throughout
  under ~96% sglang utilization — absolute numbers are incomparable; only interleaved A/B on
  the same binary, gate on/gate off, is meaningful. This was an early rehearsal of r59b's
  later "absolute values across windows are incomparable" lesson.

## 4. Verification

- **The split trace** (A1): the prefill forward's split count fell from 4 to 2 and CPU/CUDA
  boundaries from 2 to 1 (the commit's own words "prefill split trace 4 -> 2", "2 -> 1
  boundaries"); the master table condenses it to "4 splits → 1 per prefill forward". What this
  gate proves is **the structural claim itself** — the split boundaries really disappeared.
- **Greedy bit-identity** (verified separately for A1/A2/B, 0.5B q4_0, 48 tokens): moving the
  input, changing the readback path, and flipping the capture default must not change a single
  bit of output. Defends against "structural refactoring casually changing the math".
- **`cuda_pinned_readback_roundtrip`** (A2): a 5.6 MB round trip, deliberately larger than the 4
  MiB initial buffer — forcing out the grow-on-demand path. Defends the buffer growth logic
  and copy-out correctness.
- **The pp16/pp300 bit-parity harness** (B, 8g②'s legacy): capture/replay vs direct launch
  compared bit-for-bit, covering pp16 and pp300 (~437 nodes, real prefill scale). Defends
  against "capture semantics missing some node class".
- **The 8g① negative test redirected** (B): under the opt-out the prefill graph never captures
  even after 3+ runs; a new default-on test proves the flipped default from the other side.
- **Suite**: 161 (A1) → 162 (A2) → 163 (B) all green, run bounded at 8 threads (sglang was
  serving on the shared box). Defends against cross-module regressions.
- **Interleaved A/B** (A2): same binary, `MINFER_NO_PINNED_READBACK` toggled on/off alternately,
  0.5B decode, ~96% co-tenant — the pinned path parity to slightly ahead. Defends against
  "taking a single sample as a conclusion in a contended environment".

## 5. Results

Three structural settlements, all LANDED:

- **A1**: the prefill forward merged from 4 splits into a single CUDA split (decode already was
  one); each forward saves 2 full-stream syncs + host round-trip copies. It is the direct
  precondition of R3-B's whole-graph capture.
- **A2**: every decode step's logits readback (608 KB at 0.5B/7B-class vocab) skips the
  driver-internal pinned bounce, and the same-size per-step clone is cut. The master table's
  numeric verdict: **parity-to-slightly-ahead under load** — this session had no quiet window,
  and the honest record is "not worse under contention, structural waste deterministically
  eliminated".
- **B**: repeated same-shape prefills (server/slot scenarios) automatically capture/replay from
  the 3rd occurrence; one-shot CLI prefills never reach 3 and pay nothing. (r55 later measured
  whole-prefill capture on 7B at ≤ +0.1% with a capture-illegal malloc mid-window — prefill
  capture stays default-on, but its value scenario is repeated prefill, consistent with the
  judgment that followed.)

The small models' absolute level (Part-I records, pre-MMQ-campaign): 0.5B q4_0 decode ~257 tok/s
(llama 453), prefill ~3020 (llama 30550). R3 located and partially removed the self-inflicted
fixed overhead; the remaining small-model gap is structural in launch (the number of kernels per
step and the launch chain) — territory of the later decode campaign (the D series), outside this
step's scope.

## 6. Lessons

1. **Input declaration position is graph topology**: an input declared mid-graph cuts execution
   into multiple segments — wherever the consumers are, inputs belong at the graph head; node
   order is not semantics, but it is scheduling.
2. **Trace first, then assume where the overhead is**: of 4.0 ms/token, 2.4 ms is not GPU work —
   without the trace and DOT profiles, none of the three targets would have been found.
3. **D2H and H2D are a symmetric tax**: the pageable readback's driver bounce is the same money
   as the H2D side; and after reading back, do not clone a buffer that is already exact.
4. **Land structural-by-necessity changes even when the window shows no big win**: A1 alone has
   no pretty tok/s number, but it is the switch for B's whole-graph capture.

---
← [06 · decode MMVQ (8e)](./06-decode-mmvq-8e.md) · [Index](./README.md) · [08 →](./08-r1-int8-mmq-prefill-gemm.md)
