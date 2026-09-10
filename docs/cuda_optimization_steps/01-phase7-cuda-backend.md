# 01 · Phase 7 — CUDA backend: raw FFI device layer + graph backend 7a–7e (LANDED)

> **Result**: the first working NVIDIA GPU inference path; 7B q4_k_m @2K prefill **30.7 tok/s** (the Era-A baseline anchor — ~110× behind llama-bench at the time, but from here on every optimization step had ground to stand on).
> **Commit**: `0dc2a54` (raw-CUDA-FFI device layer; graph backend 7a–7e landed phase by phase per `docs/CUDA-BACKEND-PLAN.md`, with the 7e series closing out on 2026-08-29). **Date**: 2026-07-15 (device-layer commit) / 2026-08-30 (Era A record).

## 1. Background — where things stood

Before this step, minfer's GPU story existed only on macOS: the Metal backend (`metal.rs` + `metal.metal`) ran on the same compute-graph architecture (Phase 6 had already deleted Qwen2's imperative forward, so all inference went through build graph → assign backend → allocate → execute). On x86-64 Linux there was no GPU path at all — every token came out of the CPU's AVX2 kernels.

There was also an earlier failure on record (kept in `CUDA_OPTIMIZATION.md` Appendix C, the "Part IV" history): a CUDA attempt made without the compute graph, shaped so that every op call moved weights/activations back and forth between CPU and GPU. That experiment's outcome was distilled into the Part-IV diagnosis: per-op H2D/D2H round trips are fatal at the 7B scale — the GPU idles between ops waiting for transfers, the transfer cost eats the entire speedup, and in the end the whole path was abandoned, leaving only a problem list.

The goal of Phase 7 was therefore explicit: not "port a few kernels to CUDA", but **make CUDA the compute-graph architecture's third backend** — peer to CPU and Metal, behind the same `Backend` trait and the same scheduler. Two preconditions make that possible, and they are also the thesis of this document:

1. **Resident weights**: every matmul weight is uploaded to device memory once at model load and registered by name in a device-side registry; no forward ever touches host memory for weights again.
2. **Graph-backend dispatch**: the entire prefill/decode graph executes as one CUDA split; activations cross PCIe/unified memory only at split boundaries.

Without this step, everything that follows (8m's wmma GEMM, 8n's FA attention, R1's MMQ, the whole 39×/8.1× campaign) has no footing: 7B on pure CPU is a single-digit token rate, and levers like wmma, cp.async and tensor cores only mean something in an architecture where weights are already resident and dispatch has already eliminated the round trips.

The hardware target is the DGX Spark (GB10, sm_121, Blackwell family, unified memory architecture). That is also why every number in the chapters that follow comes from the GB10 — it is the fixed battlefield of this campaign.

## 2. Principle — the GPU mechanism

**Why per-op round trips are fatal.** Let the byte counts speak. The 7B q4_k_m weights total about 4.4 GB. In the Part-IV shape, every matmul op re-streams the weights on every forward (from the host, or from a one-shot device-side staging buffer). A prefill of 2048 tokens × 28 layers × 7 large matmuls per layer, with non-resident weights, means the device either pulls the data across PCIe over and over or shuffles it around on-chip over and over — the GB10 DRAM roofline is 273 GB/s (a number later measured precisely in the r55 roofline audit), and 4.4 GB × a few tens of re-transmissions puts transfer time in the tens of seconds. The measured 30.7 tok/s means a 2048-token prefill takes **66.7 seconds** — the true price of the "transfer-dominated" shape, and the reason it was abandoned.

**What residency + graph dispatch remove.** With weights resident, a forward's cross-device traffic shrinks to activations only: 2048 tokens × 3584 dims × f32 ≈ 29 MB per operator-level buffer, and most buffers never leave VRAM within their device-side lifetime (the allocator's liveness reuse).

The dispatch-side accounting splits in two. **Prefill**: the graph dispatches dozens of nodes as a single CUDA split, and cross-device copies happen only at split boundaries (ideally zero — after 7e③ both prefill and decode are a single split). **Decode (nt==1)**: a few hundred kernels per step, each Rust→C launch costing ~2–5 µs of host time, which accumulates to milliseconds per step — a non-negligible share of a 20+ tok/s target. That is why 7d introduced **CUDA Graph capture/replay**: every launch inside one execution window is natively recorded into a `cudaGraphExec_t`, and each subsequent step replays the whole step with a single `cudaGraphLaunch`. This is the CUDA counterpart of the Metal side's command-buffer batch submission.

**Host→device filling has to be async too.** Decode writes small inputs (token ids, positions, …) into device buffers every step; a `cudaMemcpy` from pageable memory introduces driver-internal sync points. 7e⑥ turned input filling into **pinned-staged async**: host data first lands in a pinned-memory ring staging area (`STAGING_SLOTS` slots), then goes to the device via `cudaMemcpyAsync` — same-stream ordering makes this naturally lock-free against the kernels that consume it, and a sync is needed only when the ring runs out of slots. `write_host` (excerpt 3 in §3.2) is that layer.

**Why raw FFI.** This project's hard line is zero ML-framework dependencies; the CUDA runtime API actually used here amounts to about a dozen functions (`cudaMalloc`/`cudaMemcpyAsync`/`cudaStreamCreate`/`cudaStreamBeginCapture`/`cudaGraphLaunch`/`cudaGetDeviceProperties`…), which can be declared directly with `extern "C"` — no bindgen, no third-party crate. `build.rs` compiles `src/cuda_kernels.cu` with nvcc into a static library linked into the binary; when nvcc is absent the whole feature degrades cleanly to "does not compile", never touching CUDA at all. The initial kernel set was 12 kernels (q4_0 matmul, rms_norm, rope, silu, swiglu, gqa_attn, …), Q4_0-only, with every other quantization type transparently falling back to CPU — the coverage was deliberately narrow: first make "a whole layer runs on the GPU" true.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **The `CudaState` singleton mirrors `MpsState`**: the division of responsibilities — weight registry, buffer pool, KV management — directly reuses the shape already proven by the Metal backend; in the graph architecture the two backends are "the same thing on different devices".
- **Phased landing (7a–7e), each phase independently testable**, with different acceptance gates:

  | Phase | Content | Acceptance |
  |---|---|---|
  | 7a | Skeleton: `CudaBackend` struct/buffer pool/trait impl; `alloc.rs` wiring (`enable_cuda` + `supports()`); `execute_node` handles only `Input` | Pool alloc/write/read round trip, `copy_across` both directions, KV persistent regions survive re-runs |
  | 7b | Full per-op dispatch + error checking | Per-op parity (bit-identical / tolerance, two tiers) + whole-layer chain + whole-graph logits + greedy equality |
  | 7c | Model wiring (qwen2/qwen3 cuda gate, `weights_on_cuda`, FusionPass) | GB10 E2E greedy == CPU for three models; negative paths fall back cleanly to CPU |
  | 7d | CUDA Graph capture/replay (the dispatch tax from §2) | Replay bit-identical, recapture triggers, 200-token KV growth, injected-failure fallback |
  | 7e | Polish + independently landable perf items (①–⑥) | Each item A/B'd on its own |

  A bad phase never contaminates the ones before it; this is also how the "wrap, do NOT stub" decision materialized — the `cuda.rs` device layer was kept as-is and the graph backend merely wraps it.
- At the initial commit `0dc2a54` the code still hung off `kernel.rs` dispatch + a `forward.rs` whole-layer GPU path; the 7a–7e graph backend later replaced that temporary path.
- **The error model follows the GPU Safety conventions**: kernel-invariant violations always return `Err` from `execute_node` and abort via the scheduler — never a silent fallback to CPU. CUDA's failure model is friendlier than Metal's (it does not freeze the whole machine): a fault surfaces on the *next* API call, so every launch is followed by a `cudaGetLastError` check.
- At the initial commit `0dc2a54` the code still hung off `kernel.rs` dispatch + the `forward.rs` whole-layer GPU path (Q4_0-only, other quants transparently falling back to CPU); the 7a–7e graph backend subsequently replaced that temporary path, while the device layer itself (`cuda.rs`/`cuda_kernels.cu`) was preserved in full — which is where the plan's "wrap, do NOT stub" decision came from.

### 3.2 Key code

What the raw FFI layer looks like (excerpted from `src/cuda.rs` as introduced by `0dc2a54`; the hand-written `repr(C)` `cudaDeviceProp` exists because the CUDA headers cannot be depended on):

```rust
// ─── FFI declarations for CUDA runtime API ────────────────────
extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                  count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                       count: usize, kind: i32, stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamCreate(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> i32;
    // …kernel launch wrappers (launch_q4_0_q8_0_matmul and 11 more — 12 total) in another extern "C" block
}
```

The weight registry (current tree `src/cuda.rs:1446`) — note that the "resident" semantics land here as **idempotent registration**:

```rust
pub fn register_weight(&self, name: &str, data: &[u8]) {
    if data.is_empty() { return; }
    {
        let w = self.weights.lock().unwrap();
        if let Some((_, size)) = w.get(name) {
            if *size == data.len() {
                // Device weights are immutable: same name + size ⇒ the
                // same GGUF tensor. Reuse the existing device copy instead
                // of leaking one buffer per load.
                return;
            }
            // Different size: replace the entry. The stale buffer is
            // deliberately NOT freed — a live captured graph may still
            // reference it; the leak is bounded by distinct (arch, tensor).
        }
    }
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
    // …cudaMemcpy H2D upload, stored into the registry by name
}
```

**The capture/replay entry layer** (7d, `src/cuda.rs:2297`; on begin failure it clears the error and returns false so the caller falls back to direct launches instead of continuing on a poisoned stream):

```rust
    pub fn graph_begin_capture(&self) -> bool {
        let stream = self.stream();
        let err = unsafe { cudaStreamBeginCapture(stream, 1) };
        if err != 0 {
            unsafe { cudaGetLastError(); }   // clear the error — never leave the stream poisoned
            false
        } else {
            true
        }
    }
    // graph_end_capture: cudaStreamEndCapture → cudaGraphInstantiate →
    // stored into decode_graph_exec; each later step replays with one cudaGraphLaunch.
```

Two passages from the `Backend` trait implementation that best show the architectural constraints (current tree `src/graph/cuda_backend.rs:1354`): when `execute_node` fails inside an open capture window, the window must be **aborted loudly** (otherwise later input fills get recorded into a multi-step mega-graph and every replay double-commits KV); and `write_host` uses 7e⑥'s pinned-staged async fill:

```rust
fn execute_node(&mut self, node: &CNode, in_bufs: &[usize], out_buf: usize,
                kv_pair: Option<(usize, usize)>) -> Result<(), String> {
    match self.execute_node_inner(node, in_bufs, out_buf, kv_pair) {
        Ok(()) => Ok(()),
        Err(e) => {
            // A node error during an open capture window dooms the window:
            // nothing would close it — later input fills would be RECORDED
            // into the window and the eventual close would cache a
            // multi-step graph (double KV commit on every replay).
            if self.capturing.is_some() {
                self.abort_capture(&e);
            }
            Err(e)
        }
    }
}

fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
    // …size guard…
    // 7e⑥: pinned-staged async fill (same-stream ordering makes this
    // race-free with the kernels that read the input).
    let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
    self.state.write_input_async(src, dst);
    Ok(())
}
```

### 3.3 Pitfalls

The parallel test suite was the touchstone of this phase; it flushed out several traps a single-threaded path would never meet:

- **Capture is per-stream, not per-thread.** While one backend holds an open capture window, any other thread's enqueue onto the shared stream (fills, copies, launches) gets recorded into that graph. The fix is a process-level `stream_lock`: the capturer holds the lock for the entire window, and every other stream touchpoint takes it per operation.
- **Weight-registry leak.** The parallel suite loads the same GGUF repeatedly; the early implementation `cudaMalloc`ed a fresh copy of the weights on every load — 100+ OOM aborts. The fix is the "same name + size ⇒ reuse" rule above; on a size mismatch the entry is replaced but the stale buffer is deliberately NOT freed (a live captured graph may still reference it).
- **`ModelLoadGuard`**: loading another architecture in parallel flips the CUDA gate between two forwards, producing a mixed state of "CPU-allocated persistent KV regions + a CUDA-executed split". The loader holds a re-entrant lock until registration completes.
- **No legacy `cudaMemcpy` inside a capture window**: `copy_device_to_device` was switched to `cudaMemcpyAsync`. A mine that a graph-free engine would never step on, yet it sat one step away.
- **The 7e② lesson**: a sync added temporarily in the matmul dispatch silently corrupts the capture window (garbage output whenever the graph is on) — the rule was hardened into: never sync inside `execute_node`. The same round produced a "faster but wrong" red flag: a v-selector that read only ¼ of the y values was actually faster — proof that any change to load counts must be suspected of breaking correctness first.
- **The 7e③ quantization-convention trap**: minfer's Q4_0 stores `round(v/d) + 8`, so the embed kernel must subtract the 8 back. A version without the `-8` passed the entire suite (the coverage happened to include no q4_0 model) and produced garbage only on q4_0 models — once a shared kernel changes, E2E must be re-run per quantization type.

## 4. Verification

The gates, set per phase — each defends against one class of regression:

- **7a**: buffer-pool alloc/write/read round trip, `copy_across` CPU↔CUDA in both directions, KV persistent regions surviving a re-run of `alloc_graph` — defends the lowest-level addressing/lifetime errors.
- **7b**: per-op parity — elementwise/rms ops **bit-identical** against `vec_ops`; each quant type's matmul within tolerance against the CPU Q8_0-activation reference and bit-identical against itself; RoPE / KV store-load round trip (including n_past growth) / GQA attention matching the CPU implementation; whole-layer chain tests + 0.5B Q4_0 whole-graph CUDA-vs-CPU logits + **greedy text identical word for word** — defends against "the GPU computed a different answer".
- **7c**: GB10 E2E over three models (0.5B Q4_0 / 0.6B Q8_0 / 7B Q4_K_M), greedy == CPU greedy; prefill→decode→multi-turn session checks KV growth and graph reuse; negative paths (`MINFER_DISABLE_CUDA=1`, no device, quant outside the gate) → clean CPU fallback — defends against wiring and gating mistakes.
- **7d**: replay decode logits **bit-identical** against direct launches; a `pool_gen` change triggers recapture; a 200-token generation run (KV growth + the nk fix); `MINFER_NO_CUDA_GRAPH=1` A/B; injected capture failure → session fallback still correct — defends against the state leaks specific to the capture/replay layer.

## 5. Results

- **The first working CUDA path**: 7B q4_k_m @2K prefill **30.7 tok/s**. Against llama-bench's contemporaneous ~3401 tok/s @2K that is ~110× behind — a gap deliberately kept as an honest baseline: it measures exactly the "structure in place, kernels not yet optimized" starting point.
- The 7e series close-out (each item A/B'd independently): 7e① judged the graph-vs-forward 0.449/0.525 diff a **path-identity artifact** (after Phase 6, `forward` also goes through the graph, so the "reference" was itself the CUDA graph — cross-backend f32 reduction-order noise, not a defect); 7e② vectorized q4_K/q6_K decode kernels **8.4 → 26.4 tok/s (3.1×)**; 7e③ moved embed/GetRows onto the device, making prefill/decode a **single CUDA split (zero cross-backend copies)**; 7e④ F32×F32 matmul and 7e⑤ FusedFFN completed the dispatch table; 7e⑥ pinned-staged async input filling (§2).
- The structural deliverables (what matters most to every later chapter): resident weights, by-name registration, a single split, CUDA-graph replay — on this skeleton, 8m swapped in a new GEMM kernel and turned 30.7 into 294 and then 1204.

## 6. Lessons

1. **Resident weights + graph-backend dispatch are the precondition for every later optimization**; per-op H2D/D2H on the hot path is fatal at the 7B scale (the Part-IV lesson — this step killed it at the architecture level).
2. **A capture window is a process-level mutually exclusive resource**: the per-stream semantics plus "never sync inside `execute_node`" must be written down as hard rules.
3. **The weight registry must be idempotent** (same name+size reuses, replacement does not free) — only a parallel test suite can flush out this class of trap.
4. Raw FFI is enough: a dozen runtime functions + kernel wrappers, no bindgen and no third-party crate — the dependency red line and engineering cost can both be had.

---
← [Index](./README.md) · [02 →](./02-wmma-f16-prefill-gemm-8m.md)
