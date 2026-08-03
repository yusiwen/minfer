# GPU Safety (Metal Backend)

> **Scope**: preventing GPU faults/hangs from freezing the machine, and the
> review discipline that produced these guards. See
> [`METAL_OPTIMIZATIONS.md`](METAL_OPTIMIZATIONS.md) for performance work and
> [`AGENTS.md`](AGENTS.md) for the project overview.

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

### 2.3 Runtime guards — fall back to CPU instead of risking a fault

`layer_gpu` / `output_norm_gpu` now return `false` (CPU fallback) when the
kernels' assumptions do not hold:

- `nh % nk != 0` — attention barrier participation (see 2.2).
- `hd > 256` — the `float acc[256]` private array would overflow. **Note**: the
  threadgroup memory limit is device-specific and must be queried, not assumed —
  see §4. This hardcoded `256` is the array size, which is a fixed kernel
  declaration; the threadgroup-smem check belongs in the dispatch (§4).
- `ne/nqt/nkt/nf % 32 != 0` — quantized-matmul block alignment.
- `ne % 32 != 0` in `output_norm_gpu`.

## 3. Audit findings (2026-08-02) — status

Review of all 30 Metal kernels for the same failure classes (barrier deadlock,
OOB, fixed arrays, dimension assumptions, infinite waits).

| ID | Finding | Risk | Status |
|----|---------|------|--------|
| H1 | Attention assumes `hd == hd_kv` — kernel uses `stride_kv = nk*hd` but the KV cache row is `nkt = nk*hd_kv`; OOB reads if they differ (other Qwen2.5 models may have `hd_kv != hd`). | High (fault) | **OPEN** — add `nkt != nk*hd` guard in `layer_gpu` → CPU fallback |
| H2 | Attention threadgroup smem `2*32*hd*4` may exceed the device limit for large `hd`. | High (dispatch failure) | **OPEN** — must be a runtime query, see §4 |
| M1 | Q4_K/Q5_K/Q6_K kernels assume `K % 256 == 0` (`nbe = K/256` floor); guard only checks `% 32`. Non-aligned K (e.g. 896) → missed elements (wrong) or OOB if rows unpadded. | Medium | **OPEN** — add per-weight-type `id % 256 == 0` guard |
| M2 | `kernel_get_rows_q4_0` (embed) computes `(token_id*nb+b)*Q4B` with no `token_id < vocab` check. Sampler guarantees valid ids (low risk), but no defense. | Medium | **OPEN** — host-side token_id range check → CPU fallback |
| L1 | Matmul kernels compute `ax` pointers past the buffer for OOB rows; reads guarded by `if (r0+N < p[0])` — pointer arithmetic only, no fault. | Low | Accept |
| L2 | `store_kv` has no in-kernel position bound; host `kv_ensure_layer` keeps positions < capacity. | Low | Accept |

Verified clean: simple elementwise kernels (add/mul/silu/bias/swiglu/rope all
have `tid < n` guards), Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 matmuls (K reads bounded),
GEMM smem/bc_out (within 8192 B), Q5_K qh/qs reads (within the 176 B block).

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
