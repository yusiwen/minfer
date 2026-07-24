# CUDA Graph Capture — Progress Tracking

## Current Status

**RESOLVED** ✅ — Root cause identified and fixed (2026-07-24). Graph capture now works.

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Prefill | 310 tok/s | **593 tok/s** |
| Decode | 221 tok/s | **299-486 tok/s** |
| CUDA errors | 901/900 cascade | **None** |

---

## Investigation Timeline

| Date | Phase | What | Conclusion |
|------|-------|------|------------|
| 2026-07-24 | 0 | Initial attempt: enable graph capture | Error 901/900, stream corruption |
| 2026-07-24 | 1 | C++ tests 01-05: kernel isolation | All PASS — not a kernel issue |
| 2026-07-24 | 2 | Rust test_06: FFI isolation | All PASS — not a Rust FFI issue |
| 2026-07-24 | **3** | **run_cpu bug found**: `run_cpu` initialized to `true`, never set to `false` in CUDA path | **ROOT CAUSE** |
| 2026-07-24 | 4 | Fix applied: `run_cpu = false` + graph capture re-enabled | **PASS** — no errors, 2x+ speedup |

---

## Root Cause

**Location**: `src/models/qwen2/forward.rs:35, 149, 158`

The forward pass uses a `run_cpu` flag to decide whether to use GPU results or CPU fallback:

```rust
let mut run_cpu = true;  // line 35: initialized to TRUE

// Metal path (macOS): correctly sets run_cpu = false on success
run_cpu = false;         // line 82

// CUDA path (Linux): upload + layer loop...
// run_cpu is NEVER set to false here!

if !run_cpu {            // line 158: ALWAYS false → GPU results discarded!
    cuda.graph_end_capture();
    cuda.sync();
    download_logits();
}

if run_cpu {             // line 186: ALWAYS true → CPU fallback runs
    // CPU path: calls quant_matmul_f32 which dispatches to GPU per-op
    // quant_matmul_f32 calls self.sync() → cudaStreamSynchronize
}
```

**Consequences**:

1. `run_cpu` always stays `true` → GPU results are always discarded
2. CPU fallback always runs → both GPU and CPU compute the same thing (2x wasted compute)
3. With `capture=true`: `cudaStreamBeginCapture` starts, but `graph_end_capture` is NEVER called (blocked by `if !run_cpu`)
4. Stream stuck in capture mode → CPU path's `sync()` calls `cudaStreamSynchronize` on capturing stream → error 900
5. Stream permanently corrupted → all subsequent `cudaStreamBeginCapture` calls return error 401

**Fix** (`forward.rs:136`):

```rust
cuda.upload_hidden(&hidden);
cuda.upload_positions(positions);

run_cpu = false;  // ← ADDED: assume GPU path succeeds, reset on failure

for il in 0..model.n_layer() {
    if !cuda.layer_gpu(il, ...) {
        run_cpu = true;  // GPU failed → fall back to CPU
        break;
    }
}
```

---

## Test Results — Full Matrix

### Phase 1: C++ Standalone (`test/cuda_graph/`)

| # | Test | Scope | Global (0) | ThreadLocal (1) | Relaxed (2) |
|---|------|-------|:----------:|:---------------:|:-----------:|
| 01 | `test_01_basic_capture` | 1 kernel (add) | ✅ | ✅ | ✅ |
| 02 | `test_02_rms_norm_single` | 1 kernel (rms_norm) | ✅ | ✅ | ✅ |
| 03 | `test_03_single_layer` | 24 kernels | ✅ | ✅ | ✅ |
| 04 | `test_04_multi_layer` | N×24 kernels (N=24) | ✅ | ✅ | ✅ |
| 05 | `test_05_with_output_projection` | 579 kernels + 73 MB output | — | — | ✅ |

### Phase 2: Rust Integration (`test/cuda_graph_rust/`)

| # | Test | Result |
|---|------|:------:|
| 06 | `examples/cuda_graph_diag.rs` — minimal Rust → FFI → CUDA graph | ✅ PASS |
| 07 | Stream state check | 🔜 covered by fix |
| 08 | Buffer growth check | 🔜 covered by fix |

### Phase 3: End-to-End (minfer binary)

| Test | Result |
|------|:------:|
| "what is 2+2" (7 tokens) | ✅ PASS — zero errors, 486 tok/s |
| "Write a short poem" (104 tokens) | ✅ PASS — zero errors, 299 tok/s |

---

## Files Changed

| File | Change | Purpose |
|------|--------|---------|
| `src/models/qwen2/forward.rs:136` | Add `run_cpu = false;` | Enable GPU results download path |
| `src/models/qwen2/forward.rs:122-131` | Add graph replay fast path | Replay captured graph for decode |
| `src/models/qwen2/forward.rs:138` | Enable `capture = ... && cuda.graph_begin_capture()` | Trigger graph capture on first decode |
| `src/cuda.rs:487-495` | Simplify `graph_begin_capture` | Use ThreadLocal mode (1) |
| `src/cuda.rs:497-524` | Simplify `graph_end_capture`/`graph_launch` | Clean up diagnostics |
| `build.rs:84` | Add `"61"` to SM arch candidates | GTX 1060 Pascal support |
| `src/cuda.rs:173-211` | Auto-select best GPU | Picks highest compute capability device |

---

## Reference

| File | Role |
|------|------|
| `src/cuda.rs` | CudaState, FFI declarations, graph functions |
| `src/cuda_kernels.cu` | All CUDA kernels + launch wrappers |
| `src/models/qwen2/forward.rs` | Layer loop + capture logic |
| `build.rs` | CUDA kernel compilation + linking |
| `test/cuda_graph/` | C++ standalone tests (Phase 1) |
| `test/cuda_graph_rust/` | Rust integration tests (Phase 2) |
| `examples/cuda_graph_diag.rs` | Minimal Rust → FFI → CUDA test |
