# Metal Runtime Parameter Alignment Audit (minfer ↔ llama.cpp)

> **Principle**: any parameter that llama.cpp decides at compile time (`#define`)
> or via configuration, minfer must NOT hardcode — it must be runtime-selected
> (device auto-detection) or configurable (env-var). Compile-time constants are
> only allowed when no runtime mechanism makes sense, and must note which llama
> configuration they correspond to.
>
> Created 2026-08-06. Reference llama.cpp commit `88b47a755`.

## Audit Table

| # | Parameter | minfer (hardcoded location) | llama.cpp mechanism (location) | Status | Action |
|---|---|---|---|---|---|
| **A** | **GEMM tile (prefill)** | `constexpr NR0=64, NR1=32` in the 5 `_mm_f32` kernels (metal.metal:768/892/1006/1120/1234); dispatch grid `(nt/32, od/64)` (metal.rs:290) | Compile `#define` `N_MM_BLOCK_X/Y`, `SZ_SIMDGROUP` → `NRA=64, NRB=128` (ggml-metal-impl.h:8-15); `#ifdef GGML_METAL_HAS_TENSOR` selects the `mpp::tensor_ops::matmul2d` kernel vs the legacy simdgroup kernel; runtime `has_tensor` (`MTLGPUFamilyMetal4_GGML`) picks `nr0/nr1/smem` 64/128/4096 vs 64/32/6144 (ggml-metal-device.cpp:748-759) | ✅ **MOOT on M4 Pro** (2026-08-06) | **CLOSED**: llama DISABLES the tensor GEMM for pre-M5 devices (ggml-metal-device.m:713-725 — "M4 no significant difference", M2 "5% slower"; enabled only on M5/M6/A19/A20 or `GGML_METAL_TENSOR_ENABLE=1`). minfer's 64×32 simdgroup is what llama uses on the M4 Pro. The prefill gap is NOT the GEMM — it is attention (see "Prefill Gap" in METAL_OPTIMIZATIONS.md) |
| **B** | **mul_vec tile (decode)** | Per-kernel `const short NR0/NSG`: Q5_0=4/2, Q8_0=2/4, Q6_K=2/2, Q4_K=2/2 (metal.metal:183…1707) | Per-quant `N_R0_*/N_SG_*` header constants (ggml-metal-impl.h:24-74), picked via `get_pipeline_mul_mv` (ggml-metal-device.cpp:766+) | ✅ values match llama exactly (verified 2026-08-06) | Audit whether the values should be centralized / device-dependent; no change needed now |
| **C** | **Attention chunk size** | `split_chunks = clamp(nkv/16, 1, 32)` (metal.rs:1468-1470), overridable via `MINFER_ATTN_CHUNKS`; Bc=32 tile (metal.metal:2587) | `OP_FLASH_ATTN_EXT_NCPSG/NQPSG` compile constants; `nwg=32`, `nsg` runtime-computed (ggml-metal-ops.cpp:2726-2975) | ⚠️ default hardcoded but env-overridable | Confirm the env mechanism suffices; document the design difference (minfer adaptive chunks vs llama fixed C) |
| **D** | **KV cache type** | `MINFER_CACHE_TYPE` env (default f32) (metal.rs:78-80) | Compile-time (f16 default, baked into the context) | ✅ minfer is more flexible | No change |
| **E** | **Dispatch threadgroup/grid** | Hardcoded TG (32,4)/(64,1)/(128,1) in dispatch calls (metal.rs:290-627) | Computed per-op from pipeline `nr0/nr1/nsg` (ggml-metal-ops.cpp) | ⚠️ matmuls verified aligned (2026-08-06 #6); others unaudited | Audit each for shape/device dependence |
| **F** | **Elementwise/small kernels** | float4, fixed thread counts | similar | ✅ | Low priority |

## Key Parameter Mapping (A — the GEMM tile)

```
llama compile:  ggml-metal-impl.h  SZ_SIMDGROUP=16, N_MM_NK=2, N_MM_BLOCK_X=4,
                N_MM_BLOCK_Y=2, N_MM_SIMD_GROUP_X=2, N_MM_SIMD_GROUP_Y=2
                → NRA = 16*2*2 = 64, NRB = 16*4*2 = 128   (// TODO: become function constants)
llama compile:  #ifdef GGML_METAL_HAS_TENSOR → <metal_tensor> + <MetalPerformancePrimitives/...>
                → kernel_mul_mm uses mpp::tensor_ops::matmul2d (128×64)
llama runtime:  has_tensor = [dev supportsFamily:MTLGPUFamilyMetal4_GGML]
                → nr0/nr1/smem: 64/128/4096 (M4)   or   64/32/6144 (fallback)
minfer today:   constexpr NR0=64, NR1=32 (the fallback path, hardcoded)
```

## TODOs

1. [x] **A**: GEMM tile — **CLOSED 2026-08-06**: llama disables the mpp/tensor GEMM on M4 Pro (pre-M5); minfer's 64×32 simdgroup IS llama's M4 path. No port needed.
2. [ ] **B**: centralize/audit mul_vec NR0/NSG (confirm vs llama `N_R0_*/N_SG_*`).
3. [ ] **C**: confirm attention chunk default mechanism (env override suffices).
4. [ ] **E**: audit dispatch grid/threadgroup params for shape/device dependence.
5. [ ] Audit method: full grep of `constexpr`/dispatch constants in `metal.metal`/`metal.rs` vs llama `ggml-metal-impl.h` / `-device.cpp` / `-ops.cpp`.
6. [x] **Prefill attention** — **FIXED 2026-08-11**: the 2026-08-06 "low value" verdict was based on an OLD llama baseline (the gap was measured ~1.2-1.3x back then; with llama-Metal rebuilt it is 3.8x at pp430). The classic `kernel_gqa_attn_f32` (sequential KV-loop, grid (nt,nk), barrier per 32-row tile) was measured at ~100ms/430tok (48% of prefill, ~25x llama's attention). A 3-pass parallel attention (`kernel_attn_scores`/`kernel_softmax_attn`/`kernel_attn_output`, one 256-thread TG per (t,h) row, all barrier-free) cut it to ~30ms: pp430 212→144ms (~32%), pp30 44→40ms. GQA via per-head hk=h/gqa (the broadcast-GEMM idea abandoned — a 2D GEMM can't produce the per-head 3D scores tensor). See METAL_OPTIMIZATIONS.md §3.4. (llama flash half8x8 with simdgroup matrices would close the residual ~2.3x but is still a ~600-line port for ~30ms of a one-time prefill — lower priority now that the attention kernel itself is fixed.) `MINFER_SKIP_ATTN` applies during prefill for profiling; `MINFER_NO_MATMUL_ATTN=1` restores the classic kernel for A/B.
7. [ ] **Plan reconciliation (2026-08-12)** — the 2026-08-06 "accept the architecture floor" verdict is **REVOKED**; the goal is to match llama.cpp performance. Tracking + action path: METAL_OPTIMIZATIONS.md §0 (single progress table) + §4 (the only action path: Xcode GUI per-kernel trace → flash-attention port for the decode non-matmul 4× gap → prefill GEMM execution efficiency toward ~7 TFLOPs/s). Open audit items B/C/E (#2-5 above) stay open but are secondary to §4; grid-shape probe (`prefill_gemm_throughput_profile`, 3.5-5.4 TFLOPs/s by nt) is the first prefill step. 2D-simdgroup GEMM + bf16 staging explicitly dropped (see METAL_OPTIMIZATIONS.md §4.3).
