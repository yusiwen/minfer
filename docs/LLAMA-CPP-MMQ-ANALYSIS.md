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


## 11. minfer redesign design — direction A (raw-nibble B smem, 2 blocks/SM)

This is the **Phase-1 design** (document only, `src/` untouched) for minfer's next CUDA MMQ GEMM kernel:
the **raw-nibble-smem variant targeting 2 blocks/SM**. It is the one occupancy lever that no
measured family in the r13–r25 campaign touched, per §9–§10. Everything below is derived from the
r-numbered facts in `docs/CUDA_OPTIMIZATION.md` and the llama.cpp source at `ca3d5a3e1`; a Phase-2
implementer can build the kernel from this section alone without re-deriving the geometry.

**Headline hypothesis (the bet).** The r25 census verdict is exact: IMMA and FFMA are at **hard
parity** (FFMA 114,688/tile both; IMMA 2×MAC both), the surplus is 100% support instructions
(integer/address ALU +69,465/tile = 77% of the +89,786 surplus), and a **−38% integer-ALU cut moved
wall by <0.5%** (r25). The kernel is therefore **issue/occupancy-bound, not instruction-count-bound**;
the binding resource is more resident warps per SM. Direction A buys exactly that — **2 blocks/SM →
~4 active warps/scheduler (from 2.00)** — by shrinking the weight B smem to the raw-packed 4-bit form
and accepting a small in-loop B-expansion instruction cost. The bet: the occupancy gain hides latency
better than the added instructions hurt. Per §10 this is the one untried lever that directly attacks
`1 block/SM → 2.00 warps/sched` without adding instructions to the A/balance path.

**The one hard constraint.** 2 blocks/SM on GB10 (48 SMs; shared/SM ≈ 99 KB usable, opt-in per-block
≈ 99 KB, §2) requires **per-block dynamic smem ≤ ~49.5 KB**. The current wide kernel at KD=8 is
**98,304 B** (cuda_kernels.cu:4793) → hard-pinned 1 block/SM (2.00 warps/sched, r20). The B weight
tile is the dominant term and the only one that shrinks by switching representation.

### 11.1 Tile geometry

Block = **256 threads (8 warps)**, one block per (token-tile × od-tile). Smem byte formulas (KDR =
chunks/super-block = 256/32 = 8 at KD=8; region map cuda_kernels.cu:4459-4482):

| region | per block | element | notes |
|---|---|---|---|
| QA8 (activation) | `KDR·T·32` | chunk q8 planes | r22 XOR swizzle, r20 split-phase staging |
| SDA (act d/ssum) | `KDR·T·8` | uint2 (d f16 \| ssum i16) | token pair (t,t+8) per LDS.64 |
| QB EXP (weight) | `8·O·48` | 1 byte/nibble, 48B slot | expanded-qb8 (current, cuda_kernels.cu:4573) |
| QB RAW (weight) | `O·128` qs plane (or `O·144` full super-block) | 2 nibbles/byte | raw-packed GGUF qs |
| SDS (weight scale) | `KDR·O·8` | float2 (d·sc, −dmin·m) | r15 rank-1 rescale |

Candidate geometry table (bytes; KD=8, KDR=8; T=tokens, O=od-rows):

| geom | QA8 | SDA | QB_exp | SDS | **exp total** | QB_raw | **raw total** | blocks/SM (exp / raw) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **64×128** | 16,384 | 4,096 | 49,152 | 8,192 | **77,824** | 16,384 | **45,056** | 1 / **2** |
| 128×64 | 32,768 | 8,192 | 24,576 | 4,096 | **69,632** | 8,192 | **53,248** | 1 / 1 |
| 64×64 | 16,384 | 4,096 | 24,576 | 4,096 | **49,152** | 8,192 | **32,768** | 2 / 2–3 |

(`exp` = expanded-qb8 with 48B slot; `raw` = 2-nibbles/byte qs plane. `blocks/SM` = per-block ≤
49.5 KB ⇒ 2. `QB_raw` holds the qs plane only; the header scales are staged into SDS separately.)

**Justification (A-reuse vs B-reuse, r23 + r24).** The measured re-read amplification at 128×128 is
**A ≈ 327 MB vs B ≈ 152 MB ⇒ A dominates 2.1:1** (CUDA_OPTIMIZATION.md:314-319). A-re-reads scale
with the od-tile count (`od/O`); B-re-reads with the token-tile count (`nt/T`). The **dominant**
stream is A (activation), so the geometry must keep **O large** and take any shrink on **T**:

- **64×128** — `od/O` unchanged ⇒ A re-reads stay at base (327 MB); `nt/T` doubles ⇒ B re-reads →
  304 MB. Total 631 MB. **B-reuse is sacrificed (the smaller, reusable stream — r24 "B is the
  reusable operand in dispatch windows"), A-reuse (the 2:1 dominant stream) is preserved.**
- **64×64** (the only expanded-qb8 2-block shape) — `od/O`=2× ⇒ A re-reads → 654 MB, B → 304 MB,
  total 958 MB. Doubles the **dominant** stream: strictly worse than 64×128.
- **128×64** — 53,248 B raw → 1 block (fails the goal).

**Chosen primary geometry: 64 tokens × 128 od, KDR=8 (KD=8), raw-packed QB → 45,056 B → 2
blocks/SM.** It is the only shape that both (a) reaches 2 blocks/SM and (b) keeps the od tile at 128
so the dominant A-re-read stream is untouched.

### 11.2 Warp shape & register budget

Keep the per-warp structure (each warp owns a private 16-od-row slice and reads the full T-token
tile, cuda_kernels.cu:4484-4490) now over T=64: 8 warps × 16 od-rows = 128 od. mma.m16n8k32 maps
**m = token, n = od** (epilogue `C[i·od + j]`, cuda_kernels.cu:4764-4767). Using the audit C-lane map
`get_i(l) = (l/2)*8 + tid/4`, `get_j(l) = (tid%4)*2 + (l%2)` (mma.cuh:245,262) and
`tile<16,8,int>::ne = I·J/32 = 4` (mma.cuh:226-227):

| quantity | current 128×128 | new 64×128 |
|---|---|---|
| token-groups per warp (m-steps, T/16) | 8 | **4** |
| od-groups per warp (n-steps, 16/8) | 2 | 2 |
| mma per 32-k chunk | 16 | **8** |
| sum[] size (groups × 4 C regs) | `sum[64]` | **`sum[32]`** |
| A-frag ldmatrix per chunk | 8 | **4** |
| B-frag per chunk | 1 ldmatrix.x4 | 1 raw-expand |
| chains per thread | 16 | **8** |

**Register estimate.** Dropping `sum[64]→sum[32]` (−32), halving the A-frag array (`a[8][4]→a[4][4]`,
−16) and the temp C array (`clow[8][2][4]→clow[4][2][4]`, −32) is partially offset by the raw-B
in-loop expansion temps. Estimated **~110–130 regs** (vs current 141–149, r22) — well under the
**255-spill cliff** (the r22 Lever-2 255-reg + 112B-spill is the failure mode to avoid). The halved
A-frag count also lowers the per-chunk shared-load count, partly offsetting the B-expansion ALU (§11.3).

### 11.3 Staging plan & the B-representation decision

The single unmeasured decision in Direction A is how the B weight enters smem:

**(Option 1) expanded-qb8 + ldmatrix (current).** Stage the raw weight, nibble-isolate at staging
(`0x0F0F0F0F`, 1 byte/nibble), 48B slot-major, load B-frags with ONE `ldmatrix.x4`
(cuda_kernels.cu:4569-4608, 4685-4698). **B smem = 8·O·48 = 49,152 B @ O=128 ⇒ 64×128 totals
77,824 B ⇒ 1 block/SM.** It does not reach 2 blocks at any geometry keeping O=128. **Rejected.**

**(Option 2) raw-nibbles + in-loop expansion (chosen).** Stage the **raw GGUF qs plane, 2 nibbles/byte**
(mask at USE, not at stage — the inverse payload of llama's `x_qs[...]=(qs0>>0)&0x0F0F0F0F`,
mmq-load-tiles.cuh:736-737), `O×128 B` (@ O=128: **16,384 B**), as a **bulk copy** (no staging ALU —
mirroring r18's bulk-copy staging into a raw region). B-frags are then assembled in the compute loop:
`LDS` the packed bytes + PRMT/SHF/LOP3 to spread each 2-nibble byte into two int8. **B smem =
16,384 B ⇒ 64×128 totals 45,056 B ⇒ 2 blocks/SM.** The header scales are staged separately into SDS
(`float2`, `KDR·O·8`), keeping the per-chunk rescale byte-identical to the current kernel.

**The tradeoff, quantified from the r25 census.** Option 2 raises the in-loop instruction count
because the memory-side fact is already inverted: the census had **ours LDSM 8,064/tile vs theirs
1,792** and **ours shared_ld 8,960 vs theirs 19,346** — minfer already uses the *leaner* (ldmatrix)
B path. Option 2 moves B back to plain-LDS + unpack: per (warp, chunk) the B-fragment costs ~1 LDS
(was 1 ldmatrix, ~saved 0) + **~30–60 ALU** to unpack 8 rows × 32 nibbles → int8. Over the per-tile
stream that is ~**+5–10% warp-inst** on the integer-ALU class (already the dominant surplus at
113,552/tile). The counterweight: the A-side drops 8 → 4 ldmatrix per chunk (−4 LDSM/chunk), so the
**net instruction move is ~+3–6% total** — an order of magnitude smaller than the r25-introspection
baseline. Since r25 showed a **−38% integer cut moves wall <0.5%**, a **+few-%** instruction change is
expected to be ~wall-inert **provided** the occupancy lever fires — which is exactly the bet under test.

**Kept from minfer / adopted from llama.** Keep **split-phase A staging (r20)** and the **qa8 XOR
swizzle (r22)**; keep the **rank-1 two-term rescale (r15/r16)** and the **f16-scale-with-fp32-rescale**
(per §Corrections item 2 the fp16-vs-fp32 attribution is unresolved at the source level, and the fp32
rescale is what the campaign kept for exactness). Do **NOT** copy llama's sram layout (sram_stride=76
ints, mmq.cuh:137) — it is 1-byte-per-nibble and is not the halving. What is adopted from llama is only
the *concept* of a raw nibble B plane, but **packed 2/byte** (the `block_q4_K.qs[128]` plane,
ggml-common.h), which is what actually halves the smem. The dispatch guard `(id/32)%8 == 0` (cuda.rs:2030)
applies unchanged (same pad40 q8 quantize, cuda.rs:2038).

### 11.4 Numerics

Raw-nibble semantics are **exactly** the r13-era two-term rescale (the parity-safe form, r15/r16-verified):

- The mma consumes the **unsigned 0..15 nibble** as the int8 B operand (upper nibble zero ⇒ positive
  int8), so the int accumulator holds `C_int = Σ_k nib(k)·act(k)`, nib ∈ **[0,15]**.
- At accumulate (per chunk, fp32): `sum += da·dsv·C_int + dma·dmv` with `da = act d`,
  `dsv = d·sc`, `dma = da·ssum`, `dmv = −dmin·m` (cuda_kernels.cu:4742-4752). This is the exact
  `d·s·nib − dmin·m` dequant form. **There is no `(nib − m)` centering anywhere** — that fold is
  proven wrong for q4_K because the dmin offset is per-sub-block-scaled (`−dmin·m`, not a fixed
  subtraction), which is the "82.896 diff mode" lesson. mma is `.s32.s8.s8.s32` (cuda_kernels.cu:4061);
  the f32-accumulate spelling does not exist (r15).
- **fp32 write-back epilogue** (adopt llama's): `sum[]` is already fp32 at the rescale, so the
  epilogue is a plain per-value `C[i·od + j] = sum[...]` global store (cuda_kernels.cu:4758-4768) — no
  fp16 anywhere in the mma→store path (llama's Q4_K `write_back` is likewise a plain fp32 store,
  mmq.cuh:519).

### 11.5 Risk table

| # | risk | consequence | guard / signal |
|---|---|---|---|
| 1 | **Nibble-layout mistake** at in-loop unpack (wrong nibble=k, sign-extend the high 4 bits, double dmin) | the r13-era **82.896 max-diff** parity mode | `cuda_prefill_mmq` parity arm (§11.6) must run BEFORE the first perf run; a garbage-magnitude diff (like r14's uint4-tiling corrupting qb8) = layout bug; ~1e-6 diff = legit fp rounding. |
| 2 | **Token identity** — any fp add reordering | greedy token identity diverges | the design does NOT reorder (same per-chunk mma + same two-term fp fold order); still gate on greedy-32 (r24 rung-3 convention). |
| 3 | **Smem-cap overrun** (the r7-era silent-attr-failure regression, phantom 2124) | launcher quietly fallback/corrupts | launcher re-derives smem and **return 0** (→ narrow fallback, cuda.rs:2059-2071) if over cap; `cudaFuncSetAttribute` result checked (cuda_kernels.cu:4787-4790, 4795-4798). KD=8 @ 45,056 B safe; KD=16 or O=256 would not be. |
| 4 | **Register spill at KD=8** (in-loop B-expand temps + sum[32]) | ptxas → 255 regs + local spill (the r22 Lever-2 failure) | `-Xptxas -v` gate: expect ~110–130 regs, 0 spill; `REG > 160` → risk. |
| 5 | **Occupancy gained but wall flat** (ncu ~4 warps/sched, duration unchanged) | falsifies the occupancy hypothesis | this is the designed kill criterion (§11.8), not a bug — it closes the line. |
| 6 | **A-side re-staging for the smaller T** | more A per od-tile | A re-reads unchanged (od/O held at 128); only B re-reads grow (the designed sacrifice). |

### 11.6 Phase-2 gate plan (in order)

1. **Build + correctness.** `cargo build --features cuda`; new env gate **`MINFER_MMQ_RAW_NB=1`**
   selects the raw-nibble variant as a **parallel kernel** — it never replaces
   `mmq_raw_wide_nt_kernel` in this phase; the existing wide kernel remains the default raw path.
2. **Parity.** `MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_WIDE=1 MINFER_MMQ_RAW_KD=4` (KD defaults
   to 8; the KD=8 default path is likewise gated). Gate: ≤1e-3 max-diff parity vs the host reference.
3. **Greedy token identity.** greedy-32 output identical to the default path (r24 rung-3).
4. **Perf (interleaved 3× medians, relative bar).** `≥ +1.5%` over the **re-measured** baseline (r24
   convention; current baseline KD=8 ~1385, KD=4 ~1366 tok/s). The absolute ≥1350 bar is superseded by
   the relative bar because the baseline already clears 1350.
5. **ncu occupancy + stall re-check.** `sm__warps_active`/sched should read **~4** at 2 blocks/SM;
   `long_scoreboard` should fall from ~2.92 (post-r20) toward llama's ~1.15; duration/GMAC (r13 target
   ≤107.7 μs/GMAC; llama parity ≈41 μs/GMAC — but the gate is the relative wall bar).
6. **Suite.** 166/0/3.

### 11.7 Kill criteria (variant abandoned)

- **Parity unresolvable after 3 attempts** (a nibble/dmin layout bug surviving 3 fixes), **or**
- **Occupancy achieved (ncu reads ~4 warps/sched) but wall < baseline** — which **falsifies the
  occupancy hypothesis** and closes Direction A (it would show that, like r23's FA TKV=32
  2-blocks/SM, the 2× k-loop fixed costs / added in-loop ALU consume the latency-hiding gain), **or**
- **Register spill at KD=8** that cannot be recovered without dropping to a smaller O.

### 11.8 Phase-2 outcome — LANDED (2026-09-04)

Direction A was implemented as the parallel `mmq_raw_nb_kernel` (env-gated
`MINFER_MMQ_RAW_NB=1`; KD=8-native, 64×128, 8 warps × 16 od-rows, `sum[32]`,
smem 45,056 B = QA8 16,384 + SDA 4,096 + QB(raw) 16,384 + SDS 8,192). The
existing wide kernel is unchanged and remains the default raw path; NB activates
only under `MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_NB=1` + `kd==8`, and
the launcher returns 0 (clean fallback → wide/narrow) on any smem/reg cap
failure or KD!=8.

All six Phase-2 gates green — **the occupancy hypothesis is CONFIRMED, not
falsified** (line-closing positive outcome):

1. Build clean; KD=8 instantiation **123 regs, 0 spill** (§11.5 risk #4 clear).
2. **Parity** (gate 2): NB-active `cuda_prefill_mmq` max diff ~1e-5 (f32
   rounding); the KD=4 arm is inapplicable (raw qs plane = full 256-k
   super-block, so KD=8-native) and clean-falls through to the wide kernel. The
   standalone B-unpack byte-equated to the wide kernel's ldmatrix B-fragment
   (0 mismatches), de-risking §11.5 risk #1 before integration.
3. **Greedy-32 identity** (gate 3): byte-identical completion vs the default f16
   path.
4. **Perf** (gate 4): NB median **1410.4** vs re-measured wide baseline
   **1375.2** = **+2.56%** (5/5 positive), ≥ +1.5% bar.
5. **Occupancy** (gate 5): `sm__warps_active.avg.per_cycle_active` **16.17** =
   **~4.04 warps/sched → 2 blocks/SM**; `long_scoreboard` 2.03 (from post-r20
   2.92, toward llama's 1.15); issue_active 37.36%.
6. **Suite** (gate 6): 166/0/3.

Verdict: the 2× resident-warps gain hid latency better than the in-loop
B-expansion ALU cost — exactly the §11 bet. Both kernels kept. Recorded in
docs/CUDA_OPTIMIZATION.md P6 r28.

### 11.9 Follow-up (r29, 2026-09-04): the kd-unroll — at 2 blocks/SM the
integer-ALU surplus now moves the wall (+2.80%)

The r25 census proved an instruction-cut lever (−38% integer ALU) was wall-
inert at **1 block/SM**; that drove §11's occupancy bet, which r28 confirmed
(2 blocks/SM, +2.56%). r29 re-ran the opcode-class census on the NB kernel
directly and found the **same dominating class — integer ALU, 2.85× llama
per-MAC** (2.142 vs 0.751 e-3/MAC) — but now the kernel sits at 2 blocks/SM.
The two candidate micro-levers were measured and rejected: **(a) PRMT
nibble extraction** (a sm_120 micro-test showed PRMT still needs shift+mask
and cannot beat SHF+LOP3; the kernel already emits 0 PRMT) and **(b)
software-pipelining the B-raw load** (neutral, reg-bloat; reverted). The
landed fix is the r25 lever applied at the new occupancy: **`#pragma unroll`
on the NB kd loop** (one line), folding each chunk's `is_hi` select, smem
base, and bounds into compile-time constants. Result: integer ALU −25%,
total inst −6.5%, warps_active 3.94 (2 blocks/SM preserved, 123 regs, 0
spill), parity/greedy/suite all green, **+2.80%** (interleaved 5-pair,
baseline 1387.9 → 1426.8). This closes the loop on §10's "occupancy is the
lever": once occupancy is had, the *instruction* surplus becomes visible
again — the two levers were not independent, just sequenced. Recorded in
docs/CUDA_OPTIMIZATION.md P6 r29.

### 11.10 Follow-up (r30, 2026-09-04): the SWAR word-granular unpack — a no-op
(the r29 unroll already realized it), REVERTED

Task 2 (`mmq_raw_nb_kernel`): replace the per-chunk raw-nibble unpack with a
word-granular SWAR (read one 32-bit word = 8 nibbles, produce lo/hi in ~3 ops,
"4x fewer LDS"), re-deriving the lane/word→fragment-byte map. A standalone scan
(`/tmp/minfer_nb/b_swar_validate.cu`) confirmed the word-granular variant is
**byte-identical** to the verified ldmatrix B-fragment (8 sgs × 32 lanes × 4
regs, 0 mismatches) — the map is safe.

The gate-1 pass was necessary but insufficient: disassembling the *r29* kernel
(SM121, `cuobjdump -sass`) shows the B-raw path is already **16 × `LDS.32`** =
read-once, because the r29 kd-unroll let ptxas CSE each raw word across the
kd-pair (lo chunk `LOP3 v&M`, hi chunk `SHF.R.U32.HI + LOP3`). That is precisely
the SWAR win (half the naive loads, minimal SHF/LOP3/word) — already in the
binary. A source-level SWAR (an explicit `b_hi`-carry: even kd reads + drops lo/
hi, odd kd reuses) produced essentially identical SASS (**+2 SHF, +3 LOP3**,
same 16/32/16/32 LDS/LDS.64/LDS.128/LDSM and 64 IMMA), 113 regs/0 spill,
parity + greedy-32 green, and **+0.54%** median (warm, alternating-order
4-pair, round-4 regression) — **below the +1.5% bar**.

**Outcome: neutral → reverted (cmp-verified = HEAD).** The §11 direction-A
raw-nibble kernel is therefore already at the compiler's floor for the B-unpack
instruction stream; the remaining integer-ALU surplus (1.598 e-3/MAC) and the
shared-memory-path stalls (`mio_throttle` 15.4%, `long_scoreboard` 19.0%) come
from the A-frag LDSM, the sda/sds scale reads, the staging index math, and the
epilogue — not the byte-vs-word unpack. Recorded: docs/CUDA_OPTIMIZATION.md P6 r30.

### 11.11 Follow-up (r31, 2026-09-04): the sda scale-read repack — a real but
sub-bar win (LANDED)

§11.10 left the sda/sds scale reads as an untouched surplus class. r31 restaged
only the **sda** side (the sds path is already at its max-width floor) and the
compute-side consumption together. SASS-first confirmed ptxas had **not** merged
the 4 per-kd sda `LDS.64` (each kd emits them at 0x40 stride — headroom existed,
unlike the r30 B-unpack which the compiler had already CSE'd).

The repack makes sda one-uint32-per-token with a *group-region split*
(`kd*64 + rg*32 + q*4 + gsel*2 + half`, rg = g/2, gsel = g&1) so a lane's four
token-groups' `(d|ssum)` are read as **TWO `LDS.128`** (s0 = groups 0,1, s1 =
groups 2,3) at **16 B warp stride → bank-conflict-free**. A first q-major
`[q][g0..g3]` attempt (32 B per lane at 32 B stride) was 2-way bank-conflicted
and reached only +0.57%; the region-split killed the conflict. Same values, same
per-chunk application points; the r15 two-term rank-1 fold and r22 qa8 swizzle
are untouched.

**Result:** SASS `LDS.64` 32 → 0, `LDS.128` 16 → 32 (scale path 48 → 32
conflict-free LDS/kt); sm_121 **109 regs, 0 spill** (was 111); smem 45,056 →
43,008 B (2 blocks/SM kept). Parity (NB-active) 1/0, greedy-32 byte-identical,
suite 166/0/3. ncu: `long_scoreboard` 24.61% → 21.46%, `mio_throttle` 16.53% →
15.61%. Wall **+1.07%** median across 45 samples — above noise but **below the
+1.5% bar**. Landed as a positive but sub-bar improvement (mechanism-confirmed,
register-neutral). See docs/CUDA_OPTIMIZATION.md P6 r31.

### 11.12 Follow-up (r32, 2026-09-04): the finite-lever sweep — the remaining
integer-ALU surplus is bounded at the compiler floor (both levers REVERTED)

§11.10/§11.11 left the integer-ALU surplus (1.598 e-3/MAC, ~2.1× llama)
attributed to the A-frag LDSM (irreducible), the staging index math, and the
epilogue. r32 classified the SASS per region (full kernel = 2,617 instructions:
prolog+stage0 500, per-kt in-loop staging 455, per-kt compute-kd 1,515,
run-once epilogue 115) and tested the two cuttable candidates.

**Staging — ptxas already hoisted the kt-independent addressing.** The A-token
base `(i0+r)*nb32` is materialized once in the prolog, and the A-swizzle STS
targets are byte-identical between the prolog and the in-loop copy
(`STS [R57+UR11+0x400..]`, the per-kt term carried only in uniform UR11); the
av loads group into 3 base regs + immediate offsets (`4 + kd*40`). The
remainder is per-kt addressing and the intrinsic sds scale-decode — not a
hoistable index chain, so a source-level staging change is a no-op (the r30
"compiler already did it" pattern, confirmed by SASS not source intent).

**Epilogue — run-once (0.4% ceiling), store-widening stops at float2.** The
write-back is 32 scalar `STG.E` (ptxas does not vectorize). A (g,nh) yields two
float2 pairs at rows iA/iA+8, cols (j,j+1) → 8B-aligned `STG.64` for the
interior tile, scalar tail for the od/nt boundary block. Implemented: SASS
32 `STG.E` → 16 `STG.E.64` + 24 `STG.E`, but integer ALU *rose* statically
(IMAD 55→74, LEA.HI.X 8→24) from the dual-path branch; ptxas 111 regs / 0
spill. Parity 1/0, greedy-32 byte-identical. Perf **+0.46%** (baseline 1441.5
→ 1448.15, interleaved 4-pair) — below noise, and capped anyway (run-once).

**Outcome: both reverted (cmp-verified = HEAD).** The NB kernel's remaining
integer-ALU surplus is at the compiler floor: A-frag LDSM (irreducible) +
intrinsic sda/sds decode + the fp rescale (FFMA/FMUL/I2FP at parity/deficit).
No source-level integer-ALU cut remains. Recorded: docs/CUDA_OPTIMIZATION.md P6 r32.

### 11.13 Follow-up (r33, 2026-09-04): the hybrid inner-loop port — a SASS
no-op; the "loop organization is the residual" hypothesis is FALSIFIED (REVERTED)

r32 closed every source-level *instruction-count* lever. §11.13 is the decisive
test of the *remaining* framing — that the 1.15×/GMAC residual is **pure SASS
codegen from loop organization**: port llama.cpp's `ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma`
inner-loop *shape* into `mmq_raw_nb_kernel` while keeping minfer's shell, and
see whether the emitted machine code converges toward llama and the wall turns.

**The port.** Keep the kernel signature, the 64×128 geometry, KD=8, all smem
layouts (qa8 / sda q-major repack / qb-raw / sds), the r20 split-phase staging,
the r22 XOR swizzle, the launcher + `MINFER_MMQ_RAW_NB` gate, the fp32 two-term
rank-1 rescale semantics, and the fp32 write-back. Replace only the compute-loop
*ordering* with llama's `j0`-outer / `n`-inner structure (their vec_dot is
`for j0 { for k01 { load B; for n { mma; rescale } } }`; mmq-vec-dot.cuh:414-440).
Adapted geometry: our warp is **16 od-rows × 64 tokens** (vs their 32 od-rows ×
64 tokens → `rows_per_warp/tile_C::I = 2` M-minitiles at `ntx=2`), so with the
same m16n8k32/tile<16,8,int> lane map ours fold to **j0 = 2 od-groups × n = 4
token-minitiles = 8 mma per 32-k chunk** (same 8 mma as before, just re-ordered
so a weight B-fragment is reused across the 4 token-minitiles — llama's
B-reuse-across-n). Our **k01 is degenerate** (one m16n8k32 covers the whole 32-k
chunk; `nchunk = id/32`), so the per-32-k-chunk rescale boundary is untouched
and **no fp accumulation order changes** — parity is preserved by construction.

**Result.** Every gate that depends on the *emitted code* came back unchanged:
ptxas `mmq_raw_nb_kernel<8>` **109 regs, 0 spill** (r31-identical), smem
43,008 B, 2 blocks/SM; parity 1/0; greedy-32 byte-identical; SASS **byte-identical
to r31** (same 64 IMMA ordering, same LDS/LDSM counts); perf interleaved 4-pair
**neutral** (−0.73/−0.59/+1.30/−0.10%, median −0.25%); census **integer ALU
1.66 e-3/MAC** (did NOT move toward llama's 0.751). Reverted (cmp-verified = HEAD).

**Verdict — the residual is NOT loop-organization codegen.** ptxas already
schedules the 8 mma + rescale identically whether the source loop is
g-outer/nh-inner (r28–r32) or llama's j0-outer/n-inner (r33) — the compiler
flattens the unrolled loop and re-schedules the tensor-core + FP32 stream the
same way. The 1.15×/GMAC surplus is therefore **intrinsic instruction
composition** (A-frag LDSM consuming + intrinsic sda/sds scale decode +
staging index math — the region r32 had already attributed) and **nvcc/ptxas
scheduling of the whole kernel**, not the source-level ordering of the mma
loop. A source-level reorder cannot close it. Scope caveat recorded in
docs/CUDA_OPTIMIZATION.md P6 r33: the deeper llama-faithful port that also
*transposes* to A=weight (eliminating the A-frag LDSM, B=activation via plain
LDS) would change instruction *composition* rather than loop organization, and
was not reached within budget — it is the remaining avenue if the "residual"
line is to be pursued further. The pure loop-organization hypothesis is falsified
and this line closes at that finding.

### 11.14 Follow-up (r34, 2026-09-04): the quantize-transpose prepass — the
"composition" residual WAS the A-side layout-transformation locality (LANDED)

§11.13's scope caveat named the deeper port that changes instruction
*composition* rather than loop organization: eliminate the A-frag LDSM staging
altogether. r34 tests a narrower, llama-faithful slice of that — **relocate the
A-side layout transform OUT of the mma kernel into a quantize-transpose
prepass** (exactly llama's `quantize_mmq_q8_1` design of §3), while keeping the
mma kernel's A-frag reads (the A-frag LDSM itself is untouched; what changes is
how the smem qa8/sda_q tiles are *produced*).

**The asymmetry r34 removes.** minfer's NB kernel re-stages the activation tile
ad hoc: for each (64-token block, od-tile-column) it reads the native token-major
40B chunks and writes the smem qa8 region through the r22 XOR swizzle plus the
r31 q-major sda repack — per-element index math in the hot loop. And because the
grid is `(nt/64, od/128)`, a given 64-token A tile is re-staged once per od-tile
column (~28× per activation buffer). llama avoids both by having the quantize
prepass emit the activations already pre-transposed into the mma kernel's layout.

**The port (gated `MINFER_MMQ_A_TRANSPOSE=1`, paired with `MINFER_MMQ_RAW_NB=1`).**
`quantize_q8_0_pad40_t` emits the qs plane swizzled per-64-token-block
(`[ntb][nchunk][2048]`) and the packed d|ssum (`[ntb][nchunk][256]`), both
matching byte-for-byte the smem content the NB kernel's old staging produced
(quantized values bit-identical to `quantize_q8_0_pad40`, only reordered). The
new `mmq_raw_nb_bt_kernel` then stages by **bulk LDG→STS** (`uint4` copies of the
`KDR*NBI*32` qa8 + `KDR*NBI*4` sda bytes) with no per-element index math; the B
weight staging, SDS fold, mma loop, and fp32 write-back are all unchanged. The
plain NB kernel is untouched (SASS byte-identical between pre/post binaries).

**Result — hypothesis CONFIRMED.** Gates: byte-exact validator 0/9 shapes;
ptxas bt kernel **103 regs / 0 spill** (NB 109); `cuda_prefill_mmq` parity 1/0;
greedy-32 token identity identical to the f16 default; prepass cost **0.405 ms
vs native 0.446 ms (0.908×)** — the transpose does not bloat the prepass; perf
(7B q4_k_m @3354-token prefill, interleaved 4-pair) baseline **1364.2 → 1496.8
tok/s = +9.72%**; suite 166/0/3. The census (op_integer/MAC) could not be
captured this session — ncu fails to inject into `mmq_raw_nb_kernel` and
`mmq_raw_nb_bt_kernel` alike ("Unknown Error on device 0", a platform/tooling
limit), but the mechanism is corroborated by the register drop (109→103, the
staging index math leaves the kernel) and the wall. Recorded:
docs/CUDA_OPTIMIZATION.md P6 r34. The A-frag LDSM consumption and fp rescale
remain intrinsic (as r32 concluded), but the staging-bound share of the residual
is now closed; the "composition" class is no longer monolithic.

### 11.15 Follow-up (r35, 2026-09-04): the scale pre-decode — a SASS-verified
no-op on the wall; the compute loop is NOT ALU-bound (REVERTED)

§11.14 closed the staging-bound share of the residual. The r35 candidate was the
other named "composition" piece: **pre-decode the A-side sda d|ssum in the
prepass** so the mma kernel's compute loop reads LDS.128-ready f32/i32 instead of
re-deriving d (h2f) and ssum (sign-extend) per (block, kt). `quantize_q8_0_pad40_t`
was extended to emit per-token d as f32 (exact `__half2float` of the stored f16)
and ssum as i32 (exact), 8 B/token/chunk, and the BT kernel's smem scale plane
became [d f32][ssum i32] per chunk (43,008 → 45,056 B, 2 blocks/SM preserved).

**What it proved.** The SASS did what was asked — `SHF.R.S32.HI` 64→**0**,
`HADD2.F32` 64→**10**, net **-130** instructions (2304 → 2174) — but the sda reads
**doubled** (`LDS.128` 32→**48**) because two f32/i32 planes replace one packed
u32. ptxas 113 regs/0 spill (r34 103; <128 keeps 2 blocks/SM). Parity 1/0 and
greedy-32 byte-identical to r34 (numerically exact). **Perf NEUTRAL**: interleaved
5-round median **1493.2 → 1486.3 = -0.46%**, within noise; **no +1.5%**.

**Interpretation (the falsification).** Removing 128 int/fp ALU from the compute
loop moved the wall 0.0% while adding 16 LDS.128. The compute-loop wall is not
ALU-bound — the decode instructions were scheduled in the IMMA shadow (they use
spare FP/INT pipe slots under the 64/kt tensor-core mma), so they were never the
critical resource. The residual is intrinsic **composition** (A-frag LDSM
consuming + fp rescale), exactly as r32/r33 concluded; the "decode class" is now
closed as *not* a wall lever. Recorded: docs/CUDA_OPTIMIZATION.md P6 r35.

### 11.16 Follow-up (r36, 2026-09-05): the A-frag wavefront economics — the MIO
wavefront count is NOT the scarce resource (H1 REFUTED, H2 endpoint)

r35 left one named candidate untested on its own metric: replace the A-frag LDSM
supply with plain-LDS (llama's `load_generic` style) on the theory that the MIO
*wavefront* count, not the instruction count, is what gates an IMMA-bound loop.
The r34 census blockage is first cleared: `ncu` injects into
`mmq_raw_nb_bt_kernel` under the `sudo -n env LD_LIBRARY_PATH=...` prefix
(without it the driver returns ERR_NVGPUCTRPERM / "Unknown Error on device 0" —
the exact failure r34/r35 hit). Profile = qwen2.5-7b q4_k_m nt=3325 prefill (bt)
vs llama `mul_mat_q` at nt=512 (llama-bench -p 512), launch 1 each.

**wavefronts per IMMA** (`smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_*`
over `smsp__inst_executed_pipe_tensor_subpipe_imma`):

| per-IMMA | minfer bt | llama | ratio |
|---|---:|---:|---:|
| LDSM wf | 2.000 | 0.500 | 4.00× |
| LDS wf | 3.500 | 2.163 | 1.62× |
| ST wf | 0.656 | 0.844 | 0.78× |
| **total wf** | **6.156** | **3.507** | **1.76×** |
| LDSM inst | 0.500 | 0.125 | 4.00× |
| LDS inst | 0.750 | 1.349 | 0.56× |

The asymmetry is real — minfer moves 6.156 vs 3.507 shared wavefronts per IMMA
and exactly 4× the LDSM wavefronts — but the throughput side falsifies the
"scarce resource" claim: minfer does **1.85× the shared wavefronts/s** (39.0 vs
21.1 G/s) while landing at the same per-IMMA tensor rate (**6.33 vs 6.02
G-IMMA/s**, 1.05×). A per-SM fixed shared-memory pipe would cap both kernels at
the same wavefronts/s if MIO were the limiter; minfer crushes past llama's 21
G/s, so the bt kernel's MIO pipe has ~1.85× headroom. The loop is tensor/IMMA
bound (r35 again); the extra LDSM wavefronts are hidden in the tensor shadow.

**Why the LDSM→plain-LDS fix is wavefront-neutral.** LDSM.m8n8.x4 = 512 B = 4.0
wavefronts (20,873,216/5,218,304); a conflict-free LDS of the same payload = 512
B = 4 wavefronts. Same bytes → same wavefronts. H1 compared an LDSM.x4 (4 tiles,
4 wf) to a single-tile plain LDS (1 wf) — apples to oranges; llama's plain-LDS
average 1.60 wf/inst (also not single-wavefront). The actual cause of llama's
lower wavefronts/IMMA is **A-fragment reuse**: llama loads 8 A-frags once per
32-k chunk and reuses them across 64 mma (0.125 LDSM/IMMA), minfer loads 4 per
chunk and reuses each across 2 mma (0.500 LDSM/IMMA) — a 4× reuse gap from warp
tiling (llama iterates od in-kernel, minfer's warp owns one 16-od strip). That is
a loop/tiling restructure, not an access-method swap.

**Verdict — H1 REFUTED, H2 endpoint.** minfer runs 1.85× the shared wavefronts/s
and still matches llama's per-IMMA tensor throughput, so MIO is not gating. Its
issue_active/cycle/sched (0.457) is *above* llama's (0.365), so H2's issue-active
deficit is also absent. **The bt mma kernel is at per-IMMA parity with llama**;
the remaining prefill gap lives outside a structurally addressable mma lever
(per-tile prologue/wave amortization, quantize prepass, fixup). No A-side
implementation is warranted. Recorded: docs/CUDA_OPTIMIZATION.md P6 r36.

### 11.17 Follow-up (r37, 2026-09-05): the post-parity gap is the q6_K path, and
bt's wall parity only holds at prefill-scale nt (measurement-only)

§11.16 closed the bt kernel's per-IMMA economics, but it measured the bt kernel in
isolation. A full-graph nsys on the BT path (all four gates, qwen2.5-7b q4_k_m,
3325-token prefill) shows the bt kernel only covers the **q4_K** weights; the
**q6_K** weights (attn_v, ffn_down) run through the generic
`mmq_nt_kernel<(int)7,(int)2,(bool)0>`. Per-launch bucketing (GPU busy 2139.6 ms):
**GEMM q6_K = 1094.7 ms (51.2%)**, GEMM q4_K bt = 600.0 ms (28.0%), FA 5.8%,
quantize-prepass 4.1%, swiglu 3.9%, rest ~7%. The single largest slice is the q6_K
**ffn_down** GEMM — 13 launches @ ~80 ms = **1063.5 ms (48.6% of the prefill wall)**.

**Matched-nucleus wall/GMAC (this is the r37 headline number).** Both kernels at
nt≈512 (minfer 511-tok prompt, llama-bench `-p 512 -n 0 -r 1 -t 8`), nsys GEMM wall
normalized to GMAC (identical weights → identical GMAC on both sides):

| GEMM class | minfer wall/GMAC | llama wall/GMAC | ratio |
|---|---:|---:|---:|
| q4_K (bt vs `mul_mat_q<12>`) | 36.2 µs/GMAC | 31.4 µs/GMAC | 1.15× |
| **q6_K (mmq_nt<7> vs `mul_mat_q<14>`)** | **368.9 µs/GMAC** | 57.8 µs/GMAC | **6.38×** |
| total | 84.0 µs/GMAC | 35.2 µs/GMAC | 2.38× |

ncu single-launch on the matched q GEMM (grid 8×28, both 1,605,632 IMMA): minfer
379.9 µs / 4.23 G-IMMA/s vs llama 265.7 µs / 6.04 G-IMMA/s — **1.43× wall**. The
§11.16 "1.05× per-IMMA" compared minfer@3325 to llama@512, i.e. at *different* nt. At
matched nt, bt's wall per-IMMA is 1.43× slower — the r34/r36 "tile prologue / wave
amortization" dilution, hidden at prefill-scale nt but exposed at short nt.

**Conclusion.** The bt mma kernel is at per-IMMA parity **and** near wall parity
(1.15× /AGGREGATE) with llama's mul_mat_q for q4_K. The whole-prefill gap is instead
concentrated in the **q6_K GEMM path (6.38× wall/GMAC, 11.5× per-MAC vs the q4_K bt
kernel)** — a structural deficit in how minfer's q6_K is dequantized (fp16-B staging via
`mmq_nt<7>`), not a per-IMMA rate problem. Bringing the q6_K ffn_down to the q4_K bt
rate (63.6 TFLOPs) is the single largest lever (~−970 ms wall, ~2720 tok/s). No code
change. Recorded: docs/CUDA_OPTIMIZATION.md P6 r37.

### 11.18 Follow-up (r38, 2026-09-05): the q6_K BT port — layout correction and the
occupancy-dominant result (LANDED, +2.87% whole-prefill, 1.66× matched-nt; < 2× bar)

r37's priority #1 was executed: the q6_K GEMMs (attn_v + ffn_down) now run on a BT-style
raw-byte kernel `mmq_raw_nb_bt_q6k_kernel` (gated `MINFER_MMQ_Q6K_NB=1`) instead of the
generic `mmq_nt_kernel<7,2,0>`.

**The §11 "32-element sub-block" draw was corrected en route.** q6_K's GGUF layout is
**16 sub-blocks of 16 elements** (sc[16]@192, d@208; block.rs / quants.rs / ggml all agree), not
8×32. Because a 32-k chunk holds two 16-subs with independent scales, a single mma.m16n8k32
rescale is impossible: the kernel must use **KSPLIT=2** (two m16n8k16 with per-sub `dsc0`/`dsc1`,
single-term, no `dmin·m`). This mirrors llama's own q6_K vec_dot (mmq-vec-dot.cuh:1138-1161 splits
the k loop into two mma with separate `scA`). It is a real per-MAC penalty (2× the IMMA
instructions vs q4_K), but it is not what made the first build slow.

**The occupancy lever dominated, exactly as r28 predicted.** The B-side runs into the same
smem-vs-2-blocks tension as §11.3. Two variants measured:
- **KDR=8** (full 256-elem super-block B, 32,768 B; total 59,392 B) → **1 block/SM**, and
  whole-prefill **regressed** (1532.6 → 1097.8 tok/s); ncu 16.7% occupancy, latency-bound.
- **KDR=4** (half super-block B, 16,384 B; total **29,696 B → 2 blocks/SM**) → whole-prefill
  **1518.4 → 1561.9 tok/s (+2.87%)**, matched-nt q6_K GEMM **221.8 vs 368.9 µs/GMAC (1.66×)**.

So a `q4_K`-style raw-B kernel for q6_K behaves exactly like the §11.5-r28 tradeoff: shrink the
B tile to buy the second resident block, and the latency-bound loop wins. The value stays
latency-bound even at 2 blocks (ncu compute 16.7%, memory 10.3%), so the lever is now occupancy
/ pipelining, not the instruction count.

**Result.** Parity-clean (validator 0-mismatch; `cuda_prefill_mmq` 1/0 for raw + padded q6_K),
greedy-32 byte-identical, suite 166/0/3, ptxas 85 regs / 0 spill. **Below the 2× bar** (1.66× at
matched-nt, +2.87% whole-prefill) but strictly positive — landed per the stop condition, with the
~3.8×-to-5.7×-llama shortfall documented. The residual is the latency (single-buffer staging
barrier per kt) plus the intrinsic KSPLIT=2; the named next lever is a **double-buffered**
(pipelined) staging at KDR=2 that keeps 2 blocks/SM (29,696 B) while overlapping kt+1 staging with
kt compute — the mechanism the outgoing `mmq_nt<7>` used to stay near-parity at long nt.
Recorded: docs/CUDA_OPTIMIZATION.md P6 r38.

### 11.19 Follow-up (r39, 2026-09-05): the staging pipeline — LANDED (+13.3% whole-prefill, −19.7% attn_v kernel time)

r38's named next lever was executed: `mmq_raw_nb_bt_q6k_kernel` now stages kt+1's global→smem
expansion into a **second buffer** while kt computes (the `mmq_nt<7,2>` pipeline). Two structural
points worth recording:

**The correct KDR for a full double-buffer is 2, not 4.** Doubling *all four* per-kt planes
(qa8 + sda_q + qb_exp + sds) at KDR=4 is 59,392 B → 1 block/SM → the r38 KDR=8 occupancy
regression. A "double-B-only" KDR=4 variant (~46 KB, 2 blocks) is **not valid**: A stays
single-buffered, so staging kt+1's A during kt's kd-loop clobbers kt's A mid-read. KDR=2 makes the
double-buffer **exactly the r38 footprint (29,696 B → 2 blocks/SM)** because 2×2×each-plane
equals the single-buffer KDR=4 total; nktile doubles but the barrier count is unchanged (57) and
per-kt work is identical.

**Result.** Parity-clean (validator 0/4096 + 0/64, `cuda_prefill_mmq_parity` 1/0), greedy-32
byte-identical, suite 166/0/3, ptxas 87 regs / 0 spill. Whole-prefill **1568.7 → 1777.5 tok/s
(+13.3%, 3/3 interleaved positive)**; matched-nt attn_v kernel **2,046,848 ns vs r38's 2,549,248
(−19.7%)** with compute throughput 16.7% → 21.52%. This is the **largest single q6_K lever closed**
(verified the r38 "the lever is occupancy OR pipelining, not instruction count" call — the gain
is pure staging/compute overlap at constant occupancy). The kernel remains **latency-bound** (74.5%
No-Eligible, SOL OPT) at 2 blocks/SM, so the q6_K-vs-llama **~3.8× gap stays**; the open levers are
a 3rd resident block (regs 87→≤80) or KSPLIT=2's inherent 2× mma.k16. Recorded:
docs/CUDA_OPTIMIZATION.md P6 r39.
