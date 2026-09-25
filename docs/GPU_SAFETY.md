# GPU Safety (Metal + CUDA Backends)

> **Scope**: preventing GPU faults/hangs from freezing the machine, and the
> review discipline that produced these guards. The Metal sections are the
> original incident-driven rules; the CUDA section at the end translates them
> for the graph CUDA backend. See
> [`METAL_OPTIMIZATIONS.md`](METAL_OPTIMIZATIONS.md) for performance work and
> [`AGENTS.md`](../AGENTS.md) for the project overview.

---

## 1. Incident: M4 Pro GPU hang (2026-08-02)

A full diagnostic report lives at `~/macbook-gpu-hang-report-2026-08-02.md`.
Summary:

- The GPU (AGXG16X = M4 Pro) hardware hung at ~15:37:25. WindowServer's
  compositing thread froze in `mtl_submit → IOGPU → AGXG16X` (40 s zero
  progress), then every Metal client (incl. new minfer instances stuck in
  `MTLCreateSystemDefaultDevice`) blocked behind WindowServer. The machine
  required a forced shutdown.
- minfer was the only active GPU workload at the time (llama-cli / Ollama were
  idle). The exact faulting kernel was **not** identified (snapshots only show
  the post-freeze state).
- No deterministic OOB/deadlock was found for Qwen2-0.5B, but the review found
  three structural amplifiers and several latent landmines for other models.

## 2. Safety fixes applied (2026-08-02)

### 2.1 `submit()` hardening — no more infinite block

`MpsCommandBuffer::submit()` previously waited `DISPATCH_TIME_FOREVER` on a
semaphore and never checked `MTLCommandBufferStatus`. A single GPU fault would
block minfer forever (and, since Metal clients share the GPU, could stall
WindowServer → whole-machine freeze).

Now (`src/metal.rs`):
- Bounded wait: `dispatch_semaphore_wait(sem, dispatch_time(NOW, 10s))`.
- On completion, checks `MTLCommandBufferStatus` — non-`Completed` reports an
  error instead of silently continuing.
- On timeout, reports "GPU hang".
- `MINFER_TRACE=1` records the last 16 dispatch op labels
  (`rms_norm`/`matmul`/`gqa_attn`/`store_kv`/`swiglu`/`add`/`bias`/`rope`/
  `embed`); an error/timeout prints the trace so the faulting kernel family can
  be identified. Trace recording is env-gated (zero overhead when off).
- `submit()` now returns `Result<(), String>`; all callers print + exit (or fall
  back to CPU for `embed_tokens_gpu`).

### 2.2 `gqa_attn` barrier deadlock — never return before a barrier

`kernel_gqa_attn_f32/f16` used `if (h >= nh) return;` **before** the
`threadgroup_barrier`. When `nh % nk != 0`, some simdgroups exit early while
others wait on the barrier → GPU permanent deadlock = machine freeze.

Fix (`src/metal.metal`, both kernels): no early return. Invalid heads
(`h0 >= nh`) run the full loop with a dummy head index (`h = 0`, keeps pointers
in-bounds) so **all** simdgroups reach every barrier, then **skip the output
write** via a `valid_head` flag.

### 2.3 Runtime guards — fall back instead of risking a fault

The legacy whole-layer `layer_gpu` / `output_norm_gpu` entry points (deleted
with the imperative forward, Phase 6) returned `false` (CPU fallback) when the
kernels' assumptions did not hold; the graph path reports the same invariants
as `Err` from `MetalBackend::execute_node` (never a silent CPU fallback). The
checked assumptions:

- `nh % nk != 0` — attention barrier participation (see 2.2).
- `hd > 256` — the `float acc[256]` private array would overflow. **Note**: the
  threadgroup memory limit is device-specific and must be queried, not assumed —
  see §4. This hardcoded `256` is the array size, which is a fixed kernel
  declaration; the threadgroup-smem check belongs in the dispatch (§4).
- `ne/nqt/nkt/nf % 32 != 0` — quantized-matmul block alignment.
- `ne % 32 != 0` in the output matmul.

## 3. Audit findings (2026-08-02) — status

Review of all 30 Metal kernels for the same failure classes (barrier deadlock,
OOB, fixed arrays, dimension assumptions, infinite waits).

| ID | Finding | Risk | Status |
|----|---------|------|--------|
| H1 | Attention assumes `hd == hd_kv` — kernel uses `stride_kv = nk*hd` but the KV cache row is `nkt = nk*hd_kv`; OOB reads if they differ (other Qwen2.5 models may have `hd_kv != hd`). | High (fault) | **GUARDED 2026-08-03** — `layer_gpu` aborts (`nkt != nk*hd` → `gpu_abort`) instead of risking misaligned KV reads. Qwen2.5 0.5B/1.5B have `hd == hd_kv`, so this never fires on supported models |
| H2 | Attention threadgroup smem `2*32*hd*4` may exceed the device limit for large `hd`. | High (dispatch failure) | **GUARDED 2026-08-02** — `layer_gpu` queries `device.max_threadgroup_memory_length()` at init (cached) and `gpu_abort`s when `2*32*hd*4` exceeds it (see §4) |
| M1 | Q4_K/Q5_K/Q6_K kernels assume `K % 256 == 0` (`nbe = K/256` floor); non-aligned K (e.g. 896) → missed elements (wrong). | Medium | **GUARDED 2026-08-03** — `quant_matmul_f32_on_gpu_buf` aborts when `id % 256 != 0` for Q4_K/Q5_K/Q6_K (GEMM and scalar paths both use K/256) |
| M2 | `kernel_get_rows_q4_0` (embed) computes `(token_id*nb+b)*Q4B` with no `token_id < vocab` check. Sampler guarantees valid ids (low risk), but no defense. | Medium | **GUARDED 2026-08-03** — `embed_tokens_gpu` aborts if any token_id >= vocab (host-side) |
| L1 | Matmul kernels compute `ax` pointers past the buffer for OOB rows; reads guarded by `if (r0+N < p[0])` — pointer arithmetic only, no fault. | Low | Accept |
| L2 | `store_kv` has no in-kernel position bound; host `kv_ensure_layer` keeps positions < capacity. | Low | Accept |
| H3 (CUDA, C4 S2b) | A packed Q8_0 `kv4<LAYOUT>` load assumes the 4-element group sits inside one 32-element block: it forms `elem/32` and `elem%32`. A `hd` that is not a multiple of 32, or a KV head base that is not `hd`-aligned, would read a neighbouring block's scale/quants — wrong values (and, at the row end, past the cell). | High (silent wrong values) | **GUARDED** — `KvFormat::Q8_0.check_width` refuses a packed width that is not a non-zero multiple of 32 in `ensure_kv`, `hd % 32 == 0` holds for every supported architecture, and the head base is always `hd`-aligned. `cuda_kv_q8_0_roundtrip_attn` and the Q8_0 arm of `cuda_map_window_matches_the_span_over_the_same_rows` (which compares a single-row window against the dequantized cell) fail loudly on a wrong block base; `compute-sanitizer --tool memcheck` over both reports no memory error |
| H4 (CUDA, C4 S2b) | A KV cell is addressed by **bytes** (`kv_row(base, cell, row_bytes)`), and `row_bytes` comes from the *backend's* layout, while the region is sized from the *builder's* `KvFormat` in `ensure_kv`. Two policies that disagree would stride a packed region as f32 rows (the pre-S2b `bool` did exactly that for anything not `f16`). | High (silent corruption) | **GUARDED** — one tag (`KV_LAYOUT_F32/F16/Q8_0` = the `KvFormat` discriminants) reaches the kernels, `set_kv_cache_type` maps `q8_0` to `KV_LAYOUT_Q8_0` rather than to `false`, `models::load_model_ns` re-states the layout from the resolved format, and the `KvFormat::supports` / `reads_packed_kv` gate refuses the load when the backend cannot read it |

Verified clean: simple elementwise kernels (add/mul/silu/bias/swiglu/rope all
have `tid < n` guards), Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 matmuls (K reads bounded),
GEMM smem/bc_out (within 8192 B), Q5_K qh/qs reads (within the 176 B block).

**Post-audit finding (2026-08-19, fixed) — cross-kernel and within-kernel write
visibility (not a deadlock, but a correctness race)**:
- **No `memoryBarrier` between dispatches in the single prefill compute encoder**:
  dispatches are ordered but write-visibility across them is NOT guaranteed by
  Metal. `bn` reused as RMSNorm output / QKV input / WO output / ffn_down output
  raced → last-2-token garbage. Fixed with `memoryBarrierWithScope` after every
  `dispatch_*` (`src/metal.rs`), matching llama.cpp. Rule for future code: any
  buffer written by one dispatch and read by the next in the same encoder needs
  the barrier; do not rely on "it's serialized".
- **GEMM partial-tile `temp_str` overlaps `sa`/`sb`**: after the K-loop, a fast
  simdgroup could overwrite `sa`/`sb` while a slow simdgroup still read them —
  `threadgroup_barrier` is required BEFORE the `temp_str` stores (all 8 mm
  kernels). Audit rule: when a threadgroup-memory buffer is REUSED for a different
  purpose at a different loop stage, there must be a `threadgroup_barrier`
  between the last read and the first write of the reuse.

## 4. Device metrics: query at runtime, never guess (2026-08-02 rule)

**Rule**: device-specific thresholds (threadgroup memory, max threads per
threadgroup, alignment limits, etc.) **MUST be queried at runtime** via the
`metal` crate's `MTLDevice` properties — **never hardcoded from a guessed or
remembered value**.

Background: the initial H2 estimate used a guessed "32 KB threadgroup-memory
limit" for the M4 Pro. The correct approach (what llama.cpp does,
`ggml-metal-device.m:851` + `ggml-metal-ops.cpp:2367`) is:

```rust
// metal.rs — query the real limit and guard against it:
let shmem = 2 * 32 * hd * 4; // Bc * hd * 2 * sizeof(f32)
let max = self.inner.device.max_threadgroup_memory_length(); // real device value
if shmem > max {
    // CPU fallback
    return false;
}
```

The `metal` crate exposes `DeviceRef::max_threadgroup_memory_length()` and
`max_threads_per_threadgroup` for exactly this purpose. Guards that hardcode a
magic number should be re-examined: prefer a query, and document the queried
value when a hard limit (like a kernel's fixed array size) genuinely exists.

## 4a. Split-attention and float4 kernel guards (2026-08-03)

- **`kernel_gqa_attn_partial_f32`** (KV-parallel split, pass 1) preserves the
  classic kernel's barrier discipline: every simdgroup reaches every
  `threadgroup_barrier` (empty KV chunks produce `mx=-INF/S=0/acc=0` — they skip
  the tile loop *together*, so no barrier divergence). The **final acc reduction
  MUST be a uniform `d` loop** (all 32 lanes step the same `d` together) — a
  per-lane loop makes `simd_sum` reduce mismatched acc components (divergent
  reduction bug caught by the isolation test).
- **`kernel_gqa_attn_combine_f32`** (pass 2) is pure elementwise: no shared
  memory, no barriers. Guards: `t<nt`, `h<nh`, and `m==-INFINITY → write zeros`
  (avoids `exp(-INF - -INF) = NaN`).
- New `layer_gpu` guard: `hd % 4 == 0` (gpu_abort) — the float4 vectorized acc
  requires it. Existing `hd <= 256` guard covers the `acc4[64]` array (64
  float4s = 256 floats).
- **Shared-mutable-state change lesson (KV growth)**: a typo that cloned the K
  buffer into `old_v` during KV-cache growth polluted the V cache (Q4_K_M
  garbage). The split-vs-classic A/B did **NOT** catch it — both paths share the
  same corrupted KV. **Any change to shared mutable GPU state (KV cache, buffer
  growth) must be checked against a known-good reference output, not just an
  A/B of two code paths over the same state.**

## 4b. Flash-attention kernels (`kernel_flash_attn_ext_f32/_f16`, 2026-08-14)

The llama-port decode attention kernel (NSG=1, one 32-lane simdgroup per
(t, h, chunk) threadgroup). Deadlock/race discipline:

- **Mask is computed inline per lane** (`(ic+NE*tx+ty < nkv) ? 0 : -MINF_MAXHALF`),
  never via a shared `sm[]` array. llama's `sm[tiisg]` write → `sm[NE*tx+ty]`
  read is a cross-lane threadgroup access with NO barrier (works only by NSG=1
  lockstep) — a race this kernel removes on purpose.
- **All control flow is `break`-only**: `if (ic >= nkv) break` depends on
  lane-independent values, so all 32 lanes exit together. No `continue`, no
  per-lane early returns. Every lane reaches both `threadgroup_barrier`s per
  chunk. Out-of-range KV reads are clamped to `nkv-1` (in-bounds, value masked
  to ~0 via `exp(-MINF_MAXHALF)`).
- **Shuffle reductions are intra-simdgroup** (`simd_shuffle_down(8,4,2,1)` +
  `simd_shuffle(·, NL*ty)` broadcast) — no threadgroup barrier inside the reduce.
  The route-to-lane-0/16 pattern keeps the DK4=16-lane reduction pure within
  each NE group.
- **Fixed-shape guard on the host**: `flash_attn_enabled(hd)` gates dispatch on
  `hd == 64` (DK/DV are hardcoded); `layer_gpu` falls back to the split path
  for any other hd (a support limitation, not a silent safety degradation).
- **shared `ss[]` handoffs** (QK^T → softmax → PV) are the only cross-lane
  threadgroup accesses and are protected by the two `threadgroup_barrier`s per
  chunk.
- Partials {M, S, O[hd]} reuse the combine kernel's format; the combine's
  `m==-INFINITY → zeros` guard also covers empty flash chunks (iwg with
  `iwg*C >= nkv`, which break on the first iteration and write an empty
  partial).

## 5. Recurrence playbook

1. On a GPU fault/hang, `submit()` now reports the dispatch trace
   (`MINFER_TRACE=1` for labels). Reproduce with **one** app (kill Ollama /
   llama-cli / screen-capture tools), bounded `-n` generations, no long loops.
2. Bisect with `MINFER_GEMM=0`, `MINFER_CACHE_TYPE=f32`, and `git stash` of the
   metal changes.
3. If the machine freezes again: SSH in (enable Remote Login in System Settings)
   and run `sudo spindump -n minfer` + `sudo spindump`; check
   `/Library/Logs/DiagnosticReports/` for new spin/ips artifacts.
4. Only after 2+ recurrences in 1–2 weeks: run Apple Diagnostics (hold D at
   boot) and file a GPU hang report via Feedback Assistant.


## CUDA (Phase 7, aarch64 GB10)

Hard rules (mirror the Metal section; enforced in `graph/cuda_backend.rs` + `cuda.rs`):

1. **Kernel-invariant violations return `Err` from `execute_node`** — never a silent CPU fallback. Backend assignment at build time (`supports_op` + registered-weight gates) decides placement; a mid-execution guard failure aborts the run with the blocking node's name.
2. **No sync inside an active capture window.** The 7d CUDA Graph capture wraps decode splits; any `cudaStreamSynchronize` (e.g. a debug print that reads device data) inside the window corrupts the capture — the 7e② "faster but wrong" incident was exactly this (a temp wrapper's internal sync produced garbage only with graphs ON). Debug reads must go through `close_capture_or_sync` paths.
3. **Device memory is not host-readable via plain memcpy on GB10** — host probes that `copy_from_slice` a device pointer SIGSEGV (`__memcpy_sve`). All D2H goes through `cudaMemcpy` staging (`copy_to_host`).
4. **A latched error is never blamed on the kernel that just ran.** `CudaState::sync()` polls `cudaGetLastError` + `cudaStreamSynchronize`. The first reports whatever an **earlier** call on the thread latched — it is not evidence about the kernel — so its message names the observer and the `cudaGetErrorName` symbol and says explicitly that it is not attributed to a kernel, and the error is **counted** (`latched_api_error_count`) and cleared, never dropped. A persistent latched error is a *call-site* bug: find the call that discarded its return value and fix it there, do not add a louder sync. Issue [#145](https://github.com/yusiwen/minfer/issues/145): a rejected `cudaFuncSetAttribute` in the eager prefill-GEMM smem opt-in and a `cudaGraphDestroy` called on a `cudaGraphExec_t` both latched `cudaErrorInvalidValue`, and `minfer bench` printed it as "CUDA kernel launch error: 1". Its follow-on [#147](https://github.com/yusiwen/minfer/issues/147) closed the remaining sites of the same class (below) and added the `MINFER_TEST_CALL_FAIL` / `MINFER_TEST_ISSUE147` gates that prove each one refuses loudly.
5. **A return value that gates a later launch or allocation is read where the call is made.**
   - `cudaFuncSetAttribute` (the dynamic-smem opt-in) returns `cudaErrorInvalidValue` for a request above `cudaDevAttrMaxSharedMemoryPerBlockOptin` — a request already known to exceed the **queried** device limit must be **skipped with the reason printed**, never called just to observe the error (`compute-sanitizer` counts every such call, and the launch cannot succeed anyway). Any other failure must be named once (function, attribute, requested bytes, device limit, `cudaGetErrorName`), cleared at the call site, and the **following launch refused** — a launch over an un-opted-in dynamic smem cannot succeed, so it must not be issued into a checked error. Every opt-in in `cuda_kernels.cu` goes through `minfer_smem_optin`; every launch through `minfer_launch_ok`.
   - A `<<<>>>` has no return value, so its error *is* read with an immediately-following `cudaGetLastError` — that is a launch check, not attribution by position, provided **nothing else runs in between**: `minfer_launch_prelude` clears (and reports) a latch that predates the launch first, so the post-launch read can only be the launch's.
   - A `cudaGraph_t` (from `cudaStreamEndCapture`) is destroyed with `cudaGraphDestroy`, a `cudaGraphExec_t` (from `cudaGraphInstantiate`) with `cudaGraphExecDestroy` — mixing them returns `cudaErrorInvalidValue` **and leaks the handle**. Both returns are read; a failed graph destroy leaks the graph but leaves the exec valid, so it is named and cleared, and the exec is still returned.
   - **The test knobs**: `MINFER_TEST_CALL_FAIL=<comma-separated site tokens|all>` makes a hardened site perform its real call with a value that fails (an over-limit attribute / dynamic-smem request, or `cudaGraphDestroy` on the exec), and `MINFER_TEST_ISSUE147=1` enables the gates that drive it. Both are unset in every default, bench and `compute-sanitizer` run; the gates name the site, the instantiation, the requested value and the error, and assert the latch is gone.
6. **Same-stream ordering is the correctness contract for async fills**: the 7e⑥ pinned-staging `write_input_async` queues `cudaMemcpyAsync` on the stream and returns before the copy lands — safe because every consumer kernel runs later on the same stream. The staging ring syncs once when it wraps (>8 fills without a sync); never hand a slot back before that.
7. **The same contract, host-side, for async readbacks (F5)**: an `cudaMemcpyAsync` D2H into a pinned slab is stream-ordered but the *host* is not — the bytes are undefined until the event recorded after it has been waited on (`cudaEventSynchronize`). So (a) a pinned slab may only be reused or freed **after** its event wait, (b) nothing on the host may read the destination before it, and (c) the failure of the wait is a loud `Err` (never a fallback), because the alternative is reading bytes the DMA may not have written. `CudaBackend::take_cross` is the only place that wait happens, and `GraphAllocator::cross_input` refuses a staging entry whose wait has not been issued.
8. **Weight registry ownership**: `register_weight` is name-keyed (same name+size reuses the device copy; different size replaces and deliberately leaks the stale buffer — bounded, a live captured graph may still reference it). Padded Q6_K repacks register through `register_weight_q6k_padded` and dispatch on `is_weight_padded`.
9. **Device limits are queried at runtime** (SM count, compute capability, `cudaMemGetInfo`, and the per-function `cudaFuncGetAttributes` / per-device `cudaDevAttrMaxSharedMemoryPerBlockOptin` the smem opt-in reads); no hardcoded SM/arch assumptions beyond the compiled targets list.
10. **A registration-time expansion validates its payload against the layout it is about to index.** A plane built by decoding a weight's raw bytes (the q6_K `W_exp`/`_dsc`, the q4_K `_dsc`, the q8_0 p32) carries a host-side row arithmetic — block count × block size — that the raw tensor only guarantees if its type matches. The gate is therefore **two checks, and both are load-bearing**: the *type* the plane is for, and the *exact* payload length that type's ratio implies (`src/q4k_dsc.rs`, called by the loader and re-checked inside `register_weight_q4k_dsc`). Issue [#165](https://github.com/yusiwen/minfer/issues/165): the q4_K `W_dsc` gate sat in the `else` of the Q6_K branch, so it ran for every non-Q6_K type in the loader's `matches!`; a q4_0 weight has exactly q4_K's bytes/element ratio, so only the type check can refuse it, and a q8_0 weight (longer rows) was misread into a plane no kernel reads — the size check is what a future *smaller*-ratio type (a 2-bit K-quant) hits, and it must refuse rather than read past the tensor. Exact equality, never `>=`: a lower bound admits the q8_0 length. The check cannot tell a q4_K payload from another type's bytes of the same length — that is the type check's job, and the pair is what is tested.

