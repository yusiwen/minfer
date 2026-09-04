# llama.cpp MMQ — quantized-weight int8 tensor-core matmul: structure, numerics, SASS census, and minfer contrast

This document consolidates every established fact about llama.cpp's MMQ ("matrix-matrix
quantized") GEMM from minfer's CUDA optimization campaign (P6 r6–r25, `docs/CUDA_OPTIMIZATION.md`),
cross-checked line-by-line against the actual source. It is the analysis side of the "why is
llama.cpp MMQ fast" question; the empirical records it consolidates are the r20 stall table and the
r25 SASS opcode-class census, both reproduced verbatim below.

**Sources.** llama.cpp at `ca3d5a3e1` (matches the `mul_mat_q<12,128,0>` that was profiled):
`ggml/src/ggml-cuda/{mmq.cuh, mmq-vec-dot.cuh, mmq-load-tiles.cuh, mma.cuh,
mmq-config-ampere.cuh, mmq.cu, quantize.cu}`, plus `ggml/src/ggml-common.h` and
`ggml/src/ggml-quants.c`. minfer at the r25 HEAD: `src/cuda_kernels.cu`
(`mmq_raw_wide_nt_kernel<KDR>` + `mmq_stage_b`). Where the campaign document and the source
disagree, the discrepancy is listed in §Corrections rather than silently repeated. (This write-up
was re-verified against the source line-by-line by the post-r25 audit; the audit's corrections are
the resolved items in §Corrections.)

Reading convention: `x` = src0 = the **weight** matrix (only quantized tensors reach MMQ; it is
kept RAW in smem); `y` = src1 = the **activation** tokens (quantized to q8_1 on device); `dst` is
the fp32 output. In `mul_mat_q_process_tile` the weight tile is `tile_x` and the activation tile is
`tile_y` (mmq.cuh:889-891).

---

## 1. Scope & dispatch

MMQ is the quantized-weight path for `ggml_mul_mat` on CUDA: it runs the integer tensor-core MMA
(`mma.m16n8k32.s32.s8.s8.s32`) directly over the raw quantized weight bytes and a q8_1-quantized
copy of the activations, instead of dequantizing weights to f16 and running an f16 wmma GEMM.
There is no f32/f16 weight materialization anywhere in the hot loop (unlike minfer's default 8p
f16 wmma path).

**Which file implements it.** `ggml-cuda/mmq.cuh` (kernel + launch), `mmq-vec-dot.cuh`
(per-warp dot/accumulate), `mmq-load-tiles.cuh` (weight staging), `mma.cuh` (tile/mma wrapper),
`mmq-config-*.cuh` (per-arch config tables), and `mmq.cu` (host dispatch). The `mul_mat_q`
kernel family is the entry point (mmq.cuh:952-1237).

**When it dispatches.** `ggml_cuda_should_use_mmq` (mmq.cu:259-386) decides. For Q4_K the type is
in the supported switch (mmq.cu:277). The decisive rule on NVIDIA is **Turing+**: `if
(turing_mma_available(cc)) return true;` (mmq.cu:312-314) — i.e. on GB10 (Blackwell, sm_121;
`turing_mma_available` = NVIDIA && highest-compiled-arch ≥ Turing, common.cuh:348-350) MMQ is
chosen **unconditionally** for every supported quantized type, for any batch size. So the
"prefill threshold" framing is an AMD-only idea: the `ne11 < MMQ_DP4A_MAX_BATCH_SIZE (=64)` gate
(mmq.cu:327, constant at mmq.cuh:8) sits **inside the NVIDIA-only branch** (`if
(GGML_CUDA_CC_IS_NVIDIA(cc))`, mmq.cu:326-328) and applies only when `turing_mma_available` is
false; the AMD/RDNA gates are **separate** (mmq.cu:330-385, incl. the `ne11 <= 128/256` RDNA branch
at mmq.cu:337-345). On GB10 the only extra requirement is ≥48 KiB per-block smem (mmq.cu:303-310).

**Ubatch shape.** The "nt-512" the campaign profiled is not a dispatch threshold; it is the ubatch
(ubatch size) that llama.cpp splits the prefill into. `-p 2600` becomes 512-token ubatches
(llama-bench: `-p 2600 → 512-token ubs`), so each `mul_mat_q` sees src1→ne1 = 512 tokens → 4
`J=128` token tiles (`ntx = 512/128 = 4`). The campaign profiled `mul_mat_q<12,128,0>` at exactly
this shape.

**The q4_K instantiation.** The profiled kernel is `mul_mat_q<GGML_TYPE_Q4_K, 128, false>`
(= `<12,128,0>`: type 12 = Q4_K, J=128, fallback=0). Its config, from
`ggml_cuda_mmq_get_config_ampere` (mmq-config-ampere.cuh:172):
`CASE(GGML_TYPE_Q4_K, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false)`
→ **nthreads=256, occupancy=1, I=128 (od-rows), J=128 (tokens), sram_layout=Q8_1,
K_vram=MMQ_ITER_K=256, stream_k=true, fallback=false.** Template meaning per `ggml_cuda_mmq_config`
(mmq.cuh:165-178): I = SRAM tile width in src0→ne1 / dst→ne0 (the od-dim), J = SRAM tile width in
src1→ne1 / dst→ne1 (the token dim), sram_layout = the weight-tile byte layout, K_vram = logical
K per inner loop (= MMQ_ITER_K = 256, mmq.cuh:9). `fallback` toggles out-of-bounds guards in the
od direction (selected at mmq.cuh:1560-1566 by `nrows_x % 128 == 0`).

**MoE batch edge cases.** `ggml_cuda_mul_mat_q` has a second (MoE, `ids != nullptr`) arm
(mmq.cu:179-256) that only differs in how the activation rows are gathered and scattered: it builds
an `ids_src1`/`ids_dst` inverse map + `expert_bounds` (mmq.cu:187-189) via `ggml_cuda_launch_mm_ids_helper`
(mmq.cu:200-201), and for the gate/up broadcast case (`dedup_bcast = ne11 == 1 && n_expert_used > 1`,
mmq.cu:193) it quantizes each token once and scatters to its compact rows through the `ids_src1`
map (`quantize_scatter_mmq_q8_1_cuda`, mmq.cu:233-234). Both arms pad the activation's inner dim to a
row-multiple: `ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)` (mmq.cu:120), which sizes the
`src1_q8_1` buffer (mmq.cu:136-138, 205-207); the `s12`/`s13` strides and the `ntx`/`nty` grid are
computed from `ne10_padded`, not `ne10`.

## 2. Block / warp / tile geometry

| Param | Value | Source |
|---|---|---|
| threads/block | 256 (8 warps) | config nthreads=256; mmq-config-ampere.cuh:172 |
| warp size | 32 | mmq.cuh:969 |
| block od tile (I) | 128 | mmq-config-ampere.cuh:172 |
| block token tile (J) | 128 | mmq-config-ampere.cuh:172 |
| rows_per_warp | 32 (J=128 → `J>=48 && J%16==0` → 32) | mmq.cuh:180-186 |
| ntx (x-minitiles/warp) | `rows_per_warp / tile_C::I` = 32/16 = **2** | mmq-vec-dot.cuh:377 |
| accumulator regs | `sum[J*I/(nwarps*warp_size)]` = 128·128/256 = **sum[64]** | mmq.cuh:903 |
| mma per vec_dot (32-k chunk) | j0:8 × k01:4 (step `QI8_1=8`) × ntx:2 = **64 mma.m16n8k32** (16 per k01 sub-iter) | mmq-vec-dot.cuh:414-440 |

Warp-to-tile mapping (mmq-vec-dot.cuh:389-390): the 8 warps split into 4 od-groups by
`i0 = (threadIdx.y/ntx)*rows_per_warp = (ty/2)*32`, and within each od-group the two warps
de-interleave the token stream via `y += (ty%ntx)*(tile_C::J*MMQ_TILE_Y_K)` (mmq-vec-dot.cuh:379).
Each od-group covers `rows_per_warp=32` od-rows (`ntx=2` × `tile_C::I=16` minitiles) and the full
`J=128` token tile, but **each warp covers only 64 token columns**: a warp does
`J/(ntx·tile_C::J) = 128/(2·8) = 8` j0-steps of `tile_C::J=8` token columns, and the pair jointly
covers all 128. Each j0-step × k01-step × n issues one `tile_C::ne = 16·8/32 = 4`-register C fragment
per minitile (mma.cuh:227) — so the 64 `mma` instances per 32-k chunk (per `vec_dot` call) fill the
64 `sum` registers, each accumulated 4× (once per k01 sub-iteration).

**tile_C / fragment mapping.** The mma wrapper is `mma(D, A, B)` with `tile<16,8,int>` D (C), A,
and `tile<8,8,int>` B (mmq-vec-dot.cuh:370-372), which expands to
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (mma.cuh:946). Because each `int` holds four
int8 (K=32 packed into K/4=8 words), `tile<16,8,int>::ne = 16·8/32 = 4` registers per thread
(mma.cuh:227). The CAMPAIGN'S "I·J/32 = 4 regs per m16n8k32 C" reading is confirmed (`tile<I,J,T,DATA_LAYOUT_I_MAJOR>`:
`ne = I*J/32` on the NVIDIA Turing+ branch, mma.cuh:226-227); the `I·J/64 = 2` figure belongs to the
AMD MFMA branch (`#if defined(AMD_MFMA_AVAILABLE)`, mma.cuh:107-108) and was corrected in r11.

**Lane-level C-fragment map (NVIDIA `tile<16,8,int>`, `DATA_LAYOUT_I_MAJOR`).** For the C (sum)
fragment each lane holds `ne = 4` values mapped by `get_i(l) = (l/2)*8 + threadIdx.x/4` and
`get_j(l) = (threadIdx.x%4)*2 + (l%2)` (mma.cuh:245,262 — note the audit's pointer to
mmq-vec-dot.cuh is off by file; the helpers live in mma.cuh). So a lane owns a 2×2 block of the
16×8 C tile at rows `{threadIdx.x/4, threadIdx.x/4+8}` and columns `{(threadIdx.x%4)*2,
(threadIdx.x%4)*2+1}`: `l=0,1` sit on row `tid/4` (cols `(tid%4)*2` and `+1`), `l=2,3` on row
`tid/4+8`. Because `tile_C::J=8`, these per-lane columns are the token columns the mma writes, and
the `get_j` map is what ties the C fragment's columns to the j0-stepped token tile.

## 3. Activation pipeline

The activation (src1, f32) is quantized to `block_q8_1_mmq` **once per GEMM launch, before the
kernel**, not per block. In `ggml_cuda_mul_mat_q` (mmq.cu:85-256) a vmem pool buffer
`src1_q8_1` is allocated (mmq.cu:138) and filled by a **separate quantize kernel** —
`quantize_mmq_q8_1` (quantize.cu:458) launched via `quantize_mmq_q8_1_cuda` (quantize.cu:575) inside
`ggml_cuda_mul_mat_q` (mmq.cu:156-157 non-MoE, :236-237 MoE; `quantize_scatter_mmq_q8_1_cuda` for the
MoE `dedup_bcast` arm, :233-234) — then `tile_y` is bulk-loaded from it (mmq.cuh:909-939). So llama
does **not** fuse the activation quantize into the MMQ kernel; it runs a per-GEMM q8_1 quantize pass
as a separate kernel (once per launch, not per block in the hot loop). The minfer asymmetry stands —
llama quantizes the activations **once per GEMM call** while minfer's default path pays its own
r23 "convert f32→f16" 73 ms per-launch tax — but the cost is of the same order: **minfer's r23
default-path accounting should credit llama with a quantize-pass cost similar to its own convert tax.**
(The q8_1 quantize is cheaper than minfer's f32→f16 convert; the point is that "llama pays zero"
is not correct.)

`block_q8_1_mmq` (mmq.cuh:27-46) is a 128-element block (QK8_1_MMQ = 4·QK8_1 = 128): a leading
16-byte union of scales (`d4[4]`, `ds4[4]`, or `d2s6[8]`) plus `int8_t qs[128]`; `sizeof == 144 B`
(mmq.cuh:56-57). The layout comment (mmq.cuh:28-36) states the y data is grouped into 128-value
blocks, transposed, and **each block padded with 16 bytes, the pad reused to store the block scale
and partial sum** — this is the "d/ssum in pad bytes" claim. For Q4_K/Q5_K the DS4 layout is used
(mmq.cuh:82-84): `half2 ds4[4]` carries one 16-bit scale + one 16-bit partial sum per 32 values
(d0,s0,d1,s1,…).

Inside the kernel the activation tile `tile_y` has row stride `MMQ_TILE_Y_K = 36 ints = 144 B`
(mmq.cuh:119; `MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1 = 32 + 4`, since `QI8_1 = QK8_1/(4·QR8_1) = 32/4 = 8`,
ggml-common.h:124,258). The row is `[scale || qs]`: the q8_1 scale union — `half2 ds4[4]` in the DS4
layout used for Q4_K/Q5_K, i.e. 4 ints — is read at `(half2*)y` and the 32-int qs plane at `y+4`
(mmq-vec-dot.cuh:383-384). So the smem row stride **equals** the global `block_q8_1_mmq` size (144 B,
mmq.cuh:56-57); the two were only "different" earlier because the scale word was miscounted as 1 int
instead of the DS4 layout's 4 ints.

## 4. Weight staging & smem layout

`load_tiles_q4_K` (mmq-load-tiles.cuh:703-812) stages the raw q4_K weight into `tile_x` in the MMA
data layout. Per 256-k super-block per row the weight is 144 B raw (block_q4_K = d[2] + dmin[2] +
scales[12] + qs[128]). In smem the weight stays **raw nibbles, one per byte (0..15), not expanded
to signed int8**:

```
x_qs[i*sram_stride + 16*(txi/8) + txi%8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;   // mmq-load-tiles.cuh:736
x_qs[i*sram_stride + 16*(txi/8) + txi%8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;   // mmq-load-tiles.cuh:737
```

The `0x0F0F0F0F` mask isolates each 4-bit nibble into its own byte; **no `__vsubss4` centering (no
dmin subtraction at this point)** — the dmin is instead folded into the scale: `x_dm[i*sram_stride
+ 4*ksc + l] = (bxi->dm * make_half2(1.0f,-1.0f)) * make_half2(sc8[l], m8[l])`
(mmq-load-tiles.cuh:772-777), so the half2 holds `(d·sc, −dmin·m)` per 32-value sub-block.

The smem row stride is `sram_stride = ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1)
= 2·MMQ_TILE_NE_K + 2·MMQ_TILE_NE_K/QI8_1 + 4 = 64 + 8 + 4 = **76 ints (304 B)**` (mmq.cuh:137;
`2·MMQ_TILE_NE_K/QI8_1 = (2·32)/8 = 8`, not 2 — the earlier "70 ints (280 B)" undercounted the
second term). `K%8 == 4` is statically enforced (mmq.cuh:153-159). The "+4" is the 16-byte pad that makes
consecutive rows rotate bank phases; the nibble plane occupies the leading
`2·MMQ_TILE_NE_K = 64` ints and the half2 scale plane follows at offset 64 (mmq-vec-dot.cuh:381-382).

**Barrier structure.** Per `MMQ_ITER_K = 256` k-iteration, `mul_mat_q_process_tile`
(mmq.cuh:907-940) does `load_tiles(weight) + stage y-half-1 → barrier → vec_dot(k00=0) → barrier →
stage y-half-2 → barrier → vec_dot(k00=32) → barrier` = **4 barriers per 256 k = 2 per 128 k**,
identical to minfer at `MMQ_KD=8` (256-k) — this is what the r20 "barrier density at parity"
finding verified (structurally established from the source, r20).

**Synchronous staging, no cp.async.** The weight and activation tiles are staged with plain
`global→(register)→smem` stores (mmq.cuh:907-939), ordered by `__syncthreads`. There is **no
cp.async / TMA pipeline** in the classic (non-Blackwell-fp4) MMQ path; the latency hiding comes
solely from having enough resident warps, not from prefetch depth.

**Q6_K specialization — different scale layout from q4_K.** Q6_K uses `GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K`
with its own load (`ggml_cuda_mmq_load_tiles_q6_K`, mmq-load-tiles.cuh:938) and vec_dot
(`ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma`, mmq-vec-dot.cuh:1018), and no raw-nibble dmin fold. Its smem
row stride is `2·MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K + MMQ_TILE_NE_K/8 + 7 = 64 + 1 + 4 + 7 = 76 ints`
(mmq.cuh:143; `QI6_K = QK_K/(4·QR6_K) = 256/8 = 32`, ggml-common.h:139). The q6_K nibbles are **fully
centered at load**: `x_qs[...] = __vsubss4(ql | qh, 0x20202020)` (mmq-load-tiles.cuh:982-983) subtracts
32 from each byte, producing signed int8 in [−32,31] — no dmin term later (unlike q4_K, which keeps
raw 0..15 nibbles and removes dmin at accumulate). The scale layout is split rather than a half2:
one `float d` per row (`x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K] = bxi->d`, mmq-load-tiles.cuh:1003) plus
an `int8` scale per 16-value sub-block (`x_sc`, mmq-load-tiles.cuh:1021; unpacked per byte at
mmq-vec-dot.cuh:1115-1121). The rescale is therefore two-stage: `tmp = (C0·scA0 + C1·scA1)·dB`
accumulated per j0-step, then `sum += tmp·dA` (mmq-vec-dot.cuh:1161,1170) — i.e. the signed int8 scale
is folded with the float `d` at the end, a materially different flow from q4_K's `sum += dmA·dsB·C`.

## 5. Launch schedule

The host launch is `launch_mul_mat_q` (mmq.cuh:1393-1473). It reads `nsm` (mmq.cuh:1397) and
decides between an xy-tiled grid and the stream-k grid:

```
const int ntiles_dst = ntx*nty*ntzw;                            // mmq.cuh:1439
const int tiles_nwaves = (ntiles_dst + nsm - 1)/nsm;            // mmq.cuh:1440
const int tiles_efficiency_percent = 100*ntiles_dst/(nsm*tiles_nwaves);  // mmq.cuh:1441
block_nums_stream_k = (NVIDIA && efficiency >= 90) ? ntiles_dst : nsm;    // mmq.cuh:1442
```

**ntx/nty derivation.** The grid dims are host-derived from the tile sizes, not hardcoded:
`nty = (nrows_x + I - 1)/I` and `ntx = (ncols_max + J - 1)/J` (mmq.cuh:1410-1411); the kernel
recomputes `nty = (nrows_x + I - 1)/I` from the passed `nrows_x` (mmq.cuh:974) and receives `ntx`
as a fast-divisor param (`ntx_fd`, mmq.cuh:1421). This parameterization is what drives the
stream-k `ntiles_dst = ntx·nty·ntzw` (mmq.cuh:1439) and the fixup `ntiles_dst % blocks != 0`
decision below — so `ntx`/`nty` are dictated by `ncols_max`/`nrows_x` (the ubatch shape), not by a
dispatch threshold.

For the nt-512 q-proj (`ntx=4, nty=28`): `ntiles_dst = 112`, `tiles_nwaves = ceil(112/48) = 3`,
`efficiency = 100·112/144 = 77.8% < 90%` → `block_nums_stream_k = nsm = 48` → **grid (48,1,1)
persistent blocks**, plus the fixup kernel because 112 % 48 ≠ 0 (`fixup_needed` mmq.cuh:1446).

**Why the fixup exists.** Stream-k splits the K-range (`blocks_per_ne00` super-block count) across
the 48 blocks (mmq.cuh:1066-1074), so a block may work on the **tail of one output tile and the head
of another**. Two blocks can non-deterministically contribute to the same output tile, and because
the accumulation is fp, the order of partial-sum adds is not reproducible. `mul_mat_q_stream_k_fixup`
(mmq.cuh:1239-1375) runs as a second launch (grid (48,4,1), mmq.cuh:1454), reads the
`tmp_last_tile` partials written by the `fixup=true` last iterator (mmq.cuh:1232-1236) and combines
them into `dst`. This is the numeric cost of stream-k: the campaign measured the fixup at **+34 μs**
(r20).

**Write-back epilogue.** The accumulator is already fp32 end-to-end: the C fragment is an int32
mma result, but it is converted at the rescale `sum += dmA.x·dsB.x·C.x + dmA.y·dsB.y`
(mmq-vec-dot.cuh:434-437) and `sum[]` is `float` (mmq.cuh:903). So `write_back` is a **plain
per-value global store** — `dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l]`
(`ggml_cuda_mmq_write_back_mma`, mmq.cuh:473-525, esp. :519; no NVFP4/y_scale rescale on the
Q4_K path). For stream-k the `fixup=true` arm writes **contiguous per-block** partials instead:
`write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J)` (mmq.cuh:943), and the
separate `mul_mat_q_stream_k_fixup` second launch runs **only** when `ntiles_dst % block_nums_stream_k.x
!= 0` (`fixup_needed`, mmq.cuh:1446; allocated tmp_fixup :1450-1451, fixup launch :1464-1469).

**Occupancy math.** `nbytes_shared = nbs_ids + nbs_x + GGML_PAD(nbs_y, nthreads·sizeof(int))`
(mmq.cuh:1386-1391): `nbs_ids = J·4 = 512`, `nbs_x = I·sram_stride·4 = 128·76·4 = 38,912`,
`nbs_y = J·144 = 18,432`. At the profiled I=J=128 shape `GGML_PAD(18,432, 1024) = 18,432`
(18,432 = 18×1024, so **no pad step** — the pad only applies for J not a multiple of 64, e.g. J=80)
→ **57,856 B ≈ 56.5 KB** per block. With ~99 KB shared/SM on GB10 that
is 1 block/SM, and it is additionally **pinned to 1** by `__launch_bounds__(nthreads, 1)`
(mmq.cuh:953). 8 warps over 4 schedulers → **2.00 active warps/sched** — matching the r20 capture.
The occupancy lever that raw-nibble smem buys is real for **smaller J tiles** (e.g. J=64 →
`nbs_y = 9,216`, total 48,384 B ≈ 47.3 KB → 2 blocks/SM); at the profiled I=J=128 shape llama is
1 block/SM exactly like minfer.

## 6. Compute loop & numerics

The q4_K compute loop is the **universal `ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma`** (dispatched at
mmq.cuh:770), with the weight dequantize done once in `load_tiles_q4_K`. Core (mmq-vec-dot.cuh:369-442):

- **Loop granularity (NVIDIA path).** `vec_dot` is called once per 32-k chunk (per `k00`), and each
  call issues **64 `mma`**: `j0` = {0,16,…,112} (8 steps, stride `ntx·tile_C::J = 16`) × `k01` =
  {0,8,16,24} (4 steps, step `QI8_1=8`) × `n` = {0,1} (2) (mmq-vec-dot.cuh:414-440). A-frags = `tile_A
  A[ntx][MMQ_TILE_NE_K/QI8_1]` = A[2][4] = **8** A-frags (mmq-vec-dot.cuh:386, loaded once before the
  j0 loop), B-frags = **32** `tile_B` (one per j0×k01, :420, reused across the 2 n-iterations). Because
  the `sum` index does not depend on `k01`, each `sum` slot is accumulated **4×** per vec_dot (once
  per k01 sub-iteration). (The campaign's "16 mma per 32-k chunk" is the per-k01 count, 8 j0 × 2 n.)
- A-fragments `tile_A[ntx]` loaded by `load_ldmatrix(A[n], x_qs + (i0+n·tile_A::I)·sram_stride + k0,
  sram_stride)` (mmq-vec-dot.cuh:397) — ldmatrix.m8n8.x4 over the raw-nibble rows.
- B-fragments `tile_B` loaded by `load_generic(B, y_qs + j0·MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K)`
  (mmq-vec-dot.cuh:420) — plain LDS (the source comment: "**faster than load_ldmatrix**").
- `mma(C, A[n][k01/QI8_1], B)` → `mma.m16n8k32.row.col.s32.s8.s8.s32`, **int32 accumulate**
  (mma.cuh:946; int-only C/D — the f32-accumulate spelling is rejected by ptxas, r15).
- **Per-chunk rescale.** For each C value (mmq-vec-dot.cuh:434-437):
  `sum[i] += dmA.x·dsB.x·C.x + dmA.y·dsB.y`, where `dmA = __half22float2(x_dm[...])` and
  `dsB = __half22float2(y_dm[...])` (mmq-vec-dot.cuh:426,408). `dmA.x` = weight scale
  `d·sc`, `dmA.y` = `−dmin·m`, `dsB.x` = activation scale `d`, `dsB.y` = activation partial sum
  `ssum`. So the dmin correction is a **rank-1 fold in (token, od-col)** applied at accumulate time,
  exactly the term the campaign's r15 `dma = da·(float)sa` fold mirrors.

**`get_scale_min_k4` semantics** (ggml-quants.c:880-887): decodes the packed 12-byte `scales`
array of a q4_K/q5_K super-block into per-32-value `(d_scale, dmin)` pairs. For `j < 4`:
`d = scales[j] & 63, m = scales[j+4] & 63`; else `d = (scales[j+4]&0xF) | ((scales[j-4]>>6)<<4)`,
`m = (scales[j+4]>>4) | ((scales[j]>>6)<<4)` — i.e. six 6-bit and six 4-bit-composed codes. The
kernel does not call it directly; the mma path's `load_tiles_q4_K` applies the equivalent unpack
via `unpack_scales_q45_K` (mmq-load-tiles.cuh:766-767). The semantic is: each 256-value super-block
has 8 sub-blocks, each with its own `(scale, dmin)`; a value `v ∈ [0,15]` dequantizes to
`d·s·v − dmin·m`.

**Where "dequant-at-use" happens.** The weight nibbles are raw in smem (0..15); they are **not**
signed-centered and not expanded at staging. The mma consumes them as the int8 operand (so the
accumulator holds `Σ nibble·act`, with values in the unsigned range), and the centering (dmin) offset
is removed by the `dmA.y·dsB.y` term in the fp rescale. So llama's "dequant" is split: nibble
isolation at load (0x0F mask), dmin removal at accumulate. In contrast minfer byte-expands the
nibbles to per-k int8 during staging (`qb8`) and folds dmin into a `float2 (d, dmin·m)` scale.

**Rescale precision.** At the CUDA level the rescale is **fp32** (the `float2` from `__half22float2`
multiplied and accumulated in fp32, mmq-vec-dot.cuh:434-437). See §Corrections for the tension
between this source-level reading and the r25 census attribution.

## 7. Measured profile (r20 + r25, matched nt-512 q-proj)

Both captures are on the **same matched layer-0 q-proj GEMM** (nt=512, id=od=3584).
`mul_mat_q<12,128,0>` runs grid (48,1,1) + fixup (grid 48,4,1). minfer `mmq_raw_wide_nt_kernel<..>`
grid (4,28) = 112 short blocks, 2.33 waves.

**r20 stall table** (per-issue-active warp ratio; llama = launch 0 of 8, identical at launch 6):

| metric | minfer `<4>` | minfer `<8>` | llama.cpp `<12,128>` |
|---|---:|---:|---:|
| duration q-proj (μs) | 632.4 | 609.6 | 263.6 |
| issue /cyc/sched | 0.16–0.26 | 0.20 | **0.42** |
| eligible /cyc | 0.22–0.28 | 0.28 | 0.64 |
| warps active /cyc | 2.00 | 2.00 | 2.00 |
| warp inst / tile (k) | 499 | 488 | 356 |
| IMMA mma-inst | 1,605,632 | 1,605,632 | 1,605,632 |
| long_scoreboard | **6.22** | 5.76 | **1.15** |
| wait | 0.63 | 0.57 | 0.58 |
| barrier | 0.26 | 0.17 | 0.19 |
| short_scoreboard | 0.34 | 0.53 | 0.12 |
| not_selected | 0.41 | 0.40 | 0.50 |
| math_pipe_throttle | 0.36 | 0.34 | 0.47 |
| mio_throttle | 0.11 | 0.28 | 0.25 |
| lg_throttle | 0.33 | 0.60 | 0.09 |
| dispatch_stall | 0.29 | 0.28 | 0.31 |
| LDS bank conflicts | 3,211,264 | 3,211,264 | **42** |

r20 readings that survive: **issue efficiency (0.42 vs 0.26) at equal occupancy (2.00 warps/sched)
is the carrier; the stall gap is long_scoreboard (6.22 vs 1.15, 97% of the named excess —
minfer warps spend ~86% of resident cycles on global-load latency vs llama ~24%); the r13 stall
table was double-distorted (pre-r14 kernel at nt-2630 vs llama nt-512); IMMA is EXACTLY at parity
(1,605,632 = mma.m16n8k32 count for the same GEMM).** Launch shape (stream-k vs 3-wave), occupancy,
tensor work and memory bytes are ruled out; the fixup costs +34 μs.

**r25 SASS opcode-class census** (per 128×128 output tile, fully reduced over K=id; `pred_on` /
32 → warp, validated to ~1.4% of `smsp__inst_executed.sum`):

| SASS class (ncu opcode metric) | ours/tile | theirs/tile | delta/tile | ratio | verdict |
|---|---:|---:|---:|---:|---|
| integer ALU (IADD3/IMAD/LEA/SHF/SEL/ISETP/LOP3) | **113,552** | 44,087 | **+69,465** | 2.58× | ← 77% of surplus |
| FP32 FMUL (rescale) | 72,592 | 57,344 | +15,248 | 1.27× | dequant-rescale |
| conversion (I2FP/F2I) | 72,600 | 58,254 | +14,346 | 1.25× | int-mma→fp32 |
| misc (NOP/CS2R) | 13,336 | 5,851 | +7,485 | 2.28× | loop/init |
| control-flow (BRA/isync) | 3,584 | 1,160 | +2,424 | 3.09× | loop control |
| uniform datapath (UR) | 1,808 | 55 | +1,753 | 33× | uniform regs |
| FP32 FFMA (rescale/accum) | 114,688 | 114,688 | +0 | 1.00× | **identical** |
| bit (LOP3/PRMT/SHF) | 8 | 456 | −448 | 0.02× | (theirs more) |
| fp16 HADD2/HFMA path | 15,232 | 36,400 | −21,168 | 0.42× | (theirs more) |
| memory (LDG/STS/LDS/LDSM) | 31,816 | 36,836 | −5,020 | 0.86× | (theirs more) |

Totals reconcile: ours `smsp__inst_executed.sum` = 49,900,928 → **445,544 warp-inst/tile**; theirs
39,844,864 → **355,758**; **surplus +89,786/tile (25.2%)**. Dedicated warp-family memory metrics
(per tile): global_ld ours 6,944 / theirs 6,384; LDSM ours 8,064 / theirs 1,792 (+6,272, 4.5×);
shared_ld ours 8,960 / theirs 19,346 (2.2× more); shared_st ours 5,376 / theirs 7,730.

**r20/r23 duration.** The matched nt-512 q-proj GEMM is **263.6 μs (llama) + ~34 μs fixup**; the
per-tile wall is 113 μs (llama) vs 211 μs (minfer). The r23 cross-model figure for the full prefill
GEMM class is **llama MMQ 39.5 μs/GMAC** (incl. fixup) vs minfer's default f16 path 59.5 μs/GMAC +
73 ms convert tax. LDS bank conflicts: llama ~42 vs minfer 3.2M (r20) / 16.86M op_ld + 6.47M op_st
(r22, before the XOR swizzle zeroed op_ld).

## 8. Why it is fast — the design's logic

The campaign's cross-cutting conclusion (r13, r15, r20, r22, r25) is that llama's MMQ wins on two
coupled axes, and the census (r25) proves the winning mechanism is **not** the mma or the memory
path (both at parity/over-parity) but the *support instruction stream* and the *occupancy that the
smem budget buys*:

1. **Raw-nibble weight plane halves the smem budget.** `mmq-load-tiles.cuh` keeps the weight as 4-bit nibbles
   (one per byte, `0x0F` mask) in the 70-int stride; the expanded-byte form (minfer's `qb8` int8
   per-k) is ~2× the bytes. At a smaller J this is the difference between 1 and 2 blocks/SM, i.e.
   between ~2 and ~4 warps/scheduler latency coverage. The tradeoff it accepts: the mma consumes
   raw 0..15 nibbles (accumulator carries the unsigned dot) and the dmin centering is deferred to a
   rank-1 fp rescale term.
2. **q8_1 pre-quantization of activations, once per GEMM** — the Q8_1 SRAM layout carries the
   per-32 scale **and** the partial sum in the 16-B pad word, so the activation operand needs no
   in-loop dequant; `dsB.y` (partial sum) is available for free.
3. **fp16 scale path keeps the rescale op count low** (per r25 attribution): the dequant-rescale
   runs through 16-bit half2 scale values rather than a full fp32 I2F → FP32 FMUL chain, so the
   per-chunk `sum += dmA·dsB·C + dmA.y·dsB.y` is few instructions. minfer's fp32 rescale pays more
   FMUL + I2FP (72,592 + 72,600 vs 57,344 + 58,254).
4. **Tight index math / no redundant work per MAC.** The 32-od-row × 64-token warp shape (the two
   warps of an od-group jointly cover the full 128 tokens; see §2) halves the A-fragment loads per
   MAC vs a 16-row warp (r13: "their warp covers 2× the od-rows"), and the B-fragment is loaded once
   per (warp, j0-step) via plain LDS. This is the warp shape minfer's r17 remap tested — measured
   NEUTRAL for minfer at 1 block/SM (see §10 Direction C).
5. **The mma work itself is at hard parity** — IMMA and FFMA are byte-identical between the
   engines (r20: 1,605,632; r25: FFMA 114,688/tile both, IMMA 2×MAC both). Compute is never the gap.

**The tradeoffs it accepts:** (a) the nibble is expanded/dequantized at use inside the
accumulate (a per-32-k rescale term rather than a fully pre-centering staging); (b) the stream-k
schedule requires a **separate fixup pass** (+34 μs) to reorder fp partial sums; (c) the tile is
**coupled to the ubatch shape** — `J=128` is chosen to minimize `ntx` for the 512-token ubatch, and
`mul_mat_q_switch_J` picks the largest `J` that fits `ntiles_x` (mmq.cuh:1484-1500).

## 9. Contrast with minfer's MMQ

The campaign's kernel is `mmq_raw_wide_nt_kernel<KDR>` (cuda_kernels.cu:4451-4770):
block 128 od × 128 tokens, warp = 16 od-rows × full 128-token tile, 8 A-frags + 2 B-frags = 16
chains, `sum[64]`. Its staging layout (cuda_kernels.cu:4459-4482):

| operand | minfer wide (r22/r25 HEAD) | llama MMQ |
|---|---|---|
| activation | `qa8` 32-B chunks, **XOR-swizzled** granules, d/ssum in `sda_q` | Q8_1 SRAM, scale+partial in the 16-B pad |
| weight | `qb8` **expanded per-k int8** (nibbles 0..15, one per byte), slot-major 48-B stride | **raw nibbles** (0x0F mask), 70-int stride |
| scale | `sds` **float2** (d, dmin·m) — **fp32 rescale** | half2 (d·sc, −dmin·m) — fp16-op code path |
| staging | **single-buffer synchronous**, split-phase (r20) | synchronous, 4 barriers/256-k |

**The design differences and their measured consequences:**

- **Expanded B (qb8) + fp32 rescale.** minfer's smem budget at KD=8 is **98,304 B → 1 block/SM**,
  vs llama's 56.5 KB (1 block/SM at I=J=128, but with the headroom to reach 2 at smaller J). The r25
  census attributes minfer's +89,786 surplus to **integer ALU +69,465 (77%)** (address/predicate
  math for the 8-A-frag LDSM + staging bounds) and **fp32 dequant-rescale +29,594** (FMUL+I2FP).
- **Measured efficiency.** minfer issue **0.25–0.26** vs llama **0.42** (r13/r20/r25; r15 noted a
  sub-1:1 instruction→duration tracking at stall-bound SM% ≈ 31);
  llama eligible 0.64 vs minfer 0.28. minfer's long_scoreboard 6.22 (pre-r20) → 2.92 (post-split-phase),
  but the freed stalls re-saturated on lg_throttle 0.33→2.38 (r20). The remaining occupancy-bound
  **residual ≈ 1.4×** (minfer 211 μs/tile vs llama 113 μs/tile + 34 μs fixup).

**The quadruple-confirmed paradigm verdict (r20, r21–r22, r24, r25).** Four independent lever
families all land on the same wall: (r20) the gap carrier is long-scoreboard/latency, not barriers or
bytes; (r21/r22) the staging is **instruction/issue-bound** — coalesced staging and swizzle change
sector/LDSM-conflict counts but move wall >0, and any "saving" that adds ALU loses; (r24) tile-order
swizzle and persistent blocks are regressive — the scheduling structure family is closed; (r25)
a **−38% integer-ALU cut** (kd `#pragma unroll`), which halves the instruction surplus, moves the
wall by **<0.5%** — the kernel is **issue/occupancy-bound, not instruction-count-bound**. The
binding resource is "more resident warps per SM", i.e. 2 blocks/SM, i.e. the smem budget; llama's
86% (r23) memory-throughput GEMM illustrates the other half — the raw-byte B-stream (4.5 bit/weight)
at high SOL is what beats an f16 16-bit stream by ~1.5×/MAC.

## 10. Implications for a redesign

The campaign's lever map (each family measured-closed in r13–r25) leaves exactly three untried, and
all three are on the *occupancy* and *instruction-width* axes, not the memory/scheduling axes:

- **Direction A — raw-nibble B in smem to halve the smem budget → 2 blocks/SM (the occupancy lever
  no measured family touched).** Move the weight nibble expansion out of the staging byte-expansion
  (`qb8`) and keep the raw 4-bit plane (0x0F mask) so the B tile costs ~half the bytes. This is the
  one lever that directly attacks `1 block/SM → 2.00 warps/sched` (the r25 "occupancy/latency-hiding"
  conclusion) without adding instructions. It requires the mma to consume unsigned nibbles with a
  rank-1 dmin fold (arithmetic identical to llama's; parity-safe, no add-reordering).
- **Direction B — fp16 half2 rescale (numerics gated).** Shift the dequant-rescale from fp32
  (FMUL+I2FP, minfer's +29.6k/tile) to the half2 path, cutting the largest remaining fp surplus.
  Gated on numerics because the parity gate is 1e-3 and minfer's fp32 rescale is what the campaign
  kept for exactness; the r15 "f32-accumulate mma does not exist" result also caps how far this can go.
- **Direction C — revisit the 2×-od-row warp shape after A.** The warp is currently 16 od-rows;
  llama's 32-od-row shape (2 minitiles) halves the A-fragment load work per MAC. This was measured
  NEUTRAL in r17 at 1 block/SM (instructions −5.8%, wall ~0), so it is only meaningful once A (2
  blocks/SM) has changed the latency regime.

**Risks / gates to respect with any of these:** (1) **parity** — the q4_K dmin fold and the
`0x0F` mask must reproduce `get_scale_min_k4` semantics bit-identically under the 1e-3 tolerance;
(2) **token identity** — any fp add-reordering (e.g. a stream-k k-split) breaks the greedy
token-identity gate, which is the harder gate than the 1e-3 numeric one (r24 rung-3 decision);
(3) **smem-cap guard** — the launcher must re-derive smem and refuse KD=8 if it exceeds the device
cap (the silent-attr-failure regression of the r7-era wide tile); (4) the **op_st conflict mass**
(6.47M, r22) remains the store-side residual and would need re-tiling if a future shape change
touches the B staging stores.

---

## Corrections & source-level nuances

The following are places where the campaign document's claims differ from, or are not fully
reconciled with, the source at `ca3d5a3e1`. The empirical core (r20 stall table, r25 census) is
unchanged; these are precision notes.

The numbered items below fold in the post-r25 source audit. Items marked **(resolved)** supersede the
corresponding claim; the r13/r25 historical notes are kept but flagged where the audit corrects them.

1. **"65-int padded stride" ≠ source.** `CUDA_OPTIMIZATION.md` r13 describes llama's weight plane as
   "raw-nibble (65-int padded stride, bank-rotating)". The source's `sram_stride` for the Q4_K
   (Q8_1) layout is **76 ints (304 B)** — `2·32 + 2·32/8 + 4` (mmq.cuh:137), `K%8==4` enforced
   (mmq.cuh:153-159). The "16 B pad" half matches (the `+4` ints), but the "65-int" figure was not
   reproduced; treat the verified stride as 76. *(resolved — the audit confirmed the stride but
   carried the earlier "70 ints" slip; the `2·32/QI8_1` term is 8, so the stride is 76.)*
2. **"fp16 dequant-rescale" vs source-level fp32.** `CUDA_OPTIMIZATION.md` r25 attributes llama's
   throughput advantage to "llama dequantizes to **fp16** (HADD2/HFMA path)". At the CUDA level the
   `q8_1×q8_1_mma` rescale is **fp32** — `sum += dmA.x·dsB.x·C.x + dmA.y·dsB.y` with `dmA`/`dsB`
   as `float2` produced by `__half22float2` (mmq-vec-dot.cuh:426,434-437) — the half2 is the
   *storage* of the scale, and the arithmetic is fp32. The census's high llama fp16 opcode count
   (36,400) is therefore empirical but its origin is not fully resolved from the source (the
   half2 scale construction in `load_tiles_q4_K`, mmq-load-tiles.cuh:772-777, is the most likely
   contributor); the campaign's "fp16 rescale" wording is an interpretation, not a source-verified
   arithmetic description. The *direction* of the binary advantage for llama (fewer fp32 FMUL/I2FP,
   more fp16 ALU) is real and is what the census records. *(still interpreted, not source-verified.)*
3. **`MMQ_TILE_Y_K` = 36 ints = 144 B (resolved).** The doc and `CUDA_OPTIMIZATION.md` quoted **33 ints
   / 132 B** (`MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1 = 32+1`). The Q8_1 DS4 scale word is **4 ints**
   (`half2 ds4[4]`), not 1, so `MMQ_TILE_Y_K = 32 + 32/QI8_1 = 32 + 4 = 36` ints = **144 B**
   (mmq.cuh:119; `QI8_1 = QK8_1/(4·QR8_1) = 32/4 = 8`, ggml-common.h:124,258). The smem row stride
   therefore **equals** the global `block_q8_1_mmq` size (144 B, mmq.cuh:56-57); the earlier "global 144 B
   ≠ smem-row 132 B" distinction (old item 3) is superseded — the two were only different because the
   scale word was miscounted as 1 int.
4. **"2 blocks/SM → 4 warps/scheduler" as an explanation of the profiled result.** The profiled
   `mul_mat_q<12,128,0>` runs at **1 block/SM** (56.5 KB > 99/2, plus `__launch_bounds__(…, 1)`),
   exactly like minfer. The 2-blocks/SM benefit is the design's occupancy *lever* that materializes
   at smaller J tiles; it should not be read as the measured occupancy of the nt-512 q-proj capture.
   (Not verified at any J in this campaign.)
5. **"x_qs" naming.** `CUDA_OPTIMIZATION.md` and this doc use `x` for the weight and `y`/`tile_y`
   for the q8_1 activation (per mul_mat ordering src0=weight, src1=activation). In `block_q8_1_mmq`
   the int8 data field is `qs`; in-kernel it is read at `y+4` (mmq-vec-dot.cuh:383). Any
   reference to "x_qs" as the *activation* data in the campaign should be read as the weight's
   nibble plane (`x_qs` inside `load_tiles_*`), not the activation; the activation qs is `y_qs`.
6. **Compute-loop granularity (resolved).** Per `vec_dot` call (one 32-k `k00` chunk) there are **64
   `mma`**: `j0` 8 steps × `k01` 4 steps (step `QI8_1=8`) × `n` 2 (mmq-vec-dot.cuh:414-440). The
   campaign's "16 mma per 32-k chunk" is the per-`k01` count (8 `j0` × 2 `n`). A-frags = `A[2][4]`
   = 8, B-frags = 32, and each `sum` slot is accumulated 4× per vec_dot.
7. **"llama pays zero convert tax" ≠ source (resolved).** The activations are quantized by a **separate
   kernel** — `quantize_mmq_q8_1` (quantize.cu:458), launched by `quantize_mmq_q8_1_cuda`
   (quantize.cu:575) — inside `ggml_cuda_mul_mat_q` (mmq.cu:156-157 non-MoE, :236-237 MoE). It is not
   fused in-kernel. minfer's r23 default-path accounting should credit llama with a quantize-pass
   cost of **similar order to minfer's convert tax** (the q8_1 quantize is cheaper than f32→f16, but
   "llama pays zero" is wrong).
8. **nbs_y pad (resolved).** `nbytes_shared = nbs_ids + nbs_x + GGML_PAD(nbs_y, nthreads·sizeof(int))`
   (mmq.cuh:1386-1391). At J=128, `GGML_PAD(18,432, 1024) = 18,432` (18,432 = 18×1024), i.e. **no pad
   step** for the profiled shape. The total at this shape is `512 + 128·76·4 + 18,432 = 57,856 B`
   (≈56.5 KB), not the earlier 54,784 B (53.5 KB) — the audit kept the total unchanged while fixing
   the pad, but the `nbs_x = I·sram_stride·4` term also rises to 38,912 because `sram_stride` is 76.
   The pad only applies for J not a multiple of 64.
9. **`tile<16,8,int>::ne` cite (resolved).** `ne = I·J/32 = 4` is on the NVIDIA Turing+ branch at
   **mma.cuh:227**; mma.cuh:108 is the AMD MFMA `I·J/64` branch.
10. **Warp token coverage (resolved).** Each warp covers **64 token columns**, not the full 128:
    `y += (ty%ntx)*(tile_C::J*MMQ_TILE_Y_K)` (mmq-vec-dot.cuh:379) de-interleaves the od-group pair,
    whose two warps jointly cover 128. This is the warp shape minfer's r17 remap tested (measured
    NEUTRAL for minfer).
11. **Dispatch edge (resolved).** The `ne11 < MMQ_DP4A_MAX_BATCH_SIZE` gate (mmq.cu:327) sits **inside
    the NVIDIA-only branch** (`GGML_CUDA_CC_IS_NVIDIA`, mmq.cu:326-328); the AMD/RDNA gates are
    separate (mmq.cu:330-385).
12. **Audit citation notes.** The GAP A lane-map helpers `get_i`/`get_j` for `tile<16,8,int>` live in
    **mma.cuh:245,262** (the audit's mmq-vec-dot.cuh:245,262 pointer is off by file — the helpers are
    in mma.cuh, which mmq-vec-dot.cuh includes). The GAP E host `ntx` is computed at **mmq.cuh:1410-1411**
    (not :136-158), and the kernel recomputes `nty` at mmq.cuh:974.
