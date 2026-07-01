# Qwen2.5-1.5B Compatibility Bug Analysis

## Background

minfer was developed and tuned using **Qwen2-0.5B** (uniform Q4_0 quantization).
Switching to **Qwen2.5-1.5B-Instruct** (mixed Q4_K/Q6_K quantization) produced:

- Completely garbled output (random characters, mixed Chinese/English)
- GPU inference not effective (prefill only 14 tok/s)

### Key Parameter Comparison

| Parameter | Qwen2-0.5B (verified) | Qwen2.5-1.5B (broken) |
|-----------|----------------------|----------------------|
| `n_embd` | 1024 | **1536** |
| `n_head` | 16 | **12** |
| `n_head_kv` | 2 | 2 |
| `hd` (head dimension) | **64** | **128** |
| `n_ff` | 5632 | 8960 |
| `n_layer` | 24 | 28 |
| Quantization | Q4_0 (uniform) | **Q4_K + Q6_K (mixed)** |
| `token_embd` type | Q4_0 | **Q6_K** |
| `output` weight | Q4_0 | **Q6_K (weight-tied, cloned from embedding)** |

---

## Bug 1 (Fatal): Attention Kernel Head Dimension Array Overflow

### Location

`src/metal.metal` — `kernel_gqa_attn_f32`, line 733

### Problem

```metal
float4 oc[16];  // Hardcoded: supports max hd = 16 × 4 = 64
```

All loops in the kernel are bounded by `hd4 = hd / 4`:

```metal
int hd4 = hd / 4;                          // Qwen2.5-1.5B: hd4 = 128/4 = 32

// All these loops iterate 32 times when hd=128, but oc has only 16 elements
for (int i = 0; i < hd4; i++) oc[i] = (float4)0.0f;              // OOB write oc[16..31]
for (int i = 0; i < hd4; i++) d += dot(q4[i], k4[i]);            // OOB read (q4 is a pointer, no overflow)
for (int i = 0; i < hd4; i++) oc[i] *= corr;                     // OOB write
for (int i = 0; i < hd4; i++) oc[i] += e0 * v4[i];               // OOB write
for (int i = 0; i < hd4; i++) oc[i] = simd_sum(oc[i]);           // OOB read
for (int i = 0; i < hd4; i++) o4[i] = oc[i] * inv;              // OOB read
```

When `hd = 128`, `hd4 = 32`, but `oc` has only 16 elements. **oc[16] through oc[31] are all out-of-bounds writes**, corrupting other stack-local variables (`mx`, `S`, etc.), producing completely random results.

### Root Cause

This kernel was tuned for Qwen2-0.5B (hd=64, hd4=16), where `oc[16]` was exactly sufficient. No generalization for head dimension was implemented.

### Impact

- GPU attention computation produces completely random floating-point values
- All 28 layers operate on garbage data
- Final output is unpredictable

### Fix

Enlarge `oc` to support hd=128 (hd4=32):

```metal
float4 oc[32];  // Supports max hd = 128
```

All loops remain unchanged. For hd=64 models, only the first 16 elements are used — no performance cost.

Note: `q4`, `k4`, `v4` are all `device` pointers, not stack-allocated. Their accesses at `i >= 16` correctly read from device memory — no overflow there. The only problem is the `oc` array.

### Files Changed

| File | Change |
|------|--------|
| `src/metal.metal` | `float4 oc[16]` → `float4 oc[32]` |

---

## Bug 2: `output_norm_gpu` Does Not Support Q4_K/Q6_K

### Location

`src/metal.rs` — `output_norm_gpu`, lines 874-876

### Problem

```rust
if output.ttype != TensorType::Q4_0
    && output.ttype != TensorType::Q4_1
    && output.ttype != TensorType::Q8_0
{
    return false;  // Q4_K, Q6_K rejected
}
```

### Trigger Condition

Qwen2.5-1.5B uses **weight tying**: the GGUF file has no separate `output.weight` tensor, so the loader clones `tok_embd` as the output weight:

```rust
// loader.rs:215
let output = load_one(tn::OUTPUT).unwrap_or_else(|| tok_embd.clone());
```

`token_embd.weight` is Q6_K → `model.output` is also Q6_K → `output_norm_gpu` returns false.

### Current Fallback Flow

```
forward.rs:62-88
├── GPU processes 28 layers → succeeds (but Bug 1 corrupts results)
├── output_norm_gpu → returns false
├── cb.submit() → GPU work completes (28 layers executed)
├── download_hidden(&mut hidden) → downloads buf_hidden
└── run_cpu = false → CPU layer loop skipped
```

Then CPU executes the final output:

```rust
// forward.rs:128-136
rms_norm(&hidden, ...);                              // RMSNorm on GPU result
quant_matmul_f32(output, &bn, &mut logits, ...);     // output projection (CPU scalar path)
```

**Key insight**: `buf_hidden` is the residual stream, updated in-place by `layer_gpu` via `add_f32(&hidden, &bn, &hidden)`. `download_hidden` does download the post-layer residual stream. If Bug 1 did not exist, this flow would be correct — CPU doing the final rms_norm + output projection is valid.

But because of Bug 1, `hidden` is already garbage data.

### Performance Impact

Even after Bug 1 is fixed, `output_norm_gpu` not supporting Q4_K/Q6_K means:
- GPU cannot complete inference end-to-end
- Final rms_norm + output projection runs on CPU
- CPU's `quant_matmul_f32` for Q6_K uses **scalar** dot products (no AVX2), and the vocab_size=151936 matrix multiplication is extremely slow

### Fix

Extend `output_norm_gpu` type check to support Q4_K and Q6_K. The underlying Metal kernels (`kernel_q4_k_f32_matmul`, `kernel_q6_k_f32_matmul`) already exist — only the dispatch gate needs updating:

```rust
// metal.rs:output_norm_gpu
// Updated type check
if output.ttype != TensorType::Q4_0
    && output.ttype != TensorType::Q4_1
    && output.ttype != TensorType::Q8_0
    && output.ttype != TensorType::Q4_K    // NEW
    && output.ttype != TensorType::Q6_K    // NEW
{
    return false;
}

// Matmul dispatch (existing else branch already calls quant_matmul_f32_on_gpu, no change needed)
if output.ttype == TensorType::Q4_0 {
    cb.quantize_q8_0(&bn, &q8_bn, ne, nt);
    cb.quant_matmul_q8(output, &q8_bn, &logits, nv, ne, nt);
} else {
    // f32 path — already supports Q4_1, Q8_0, Q4_K, Q6_K
    cb.quant_matmul_f32_on_gpu(output, &bn, &logits, nv, ne, nt);
}
```

### Files Changed

| File | Change |
|------|--------|
| `src/metal.rs` | Add Q4_K/Q6_K to `output_norm_gpu` type check |

---

## Bug 3: Q6_K Embedding Lookup Scale Index Error

### Location

`src/models/qwen2/forward.rs` — `embed_tokens`, `TensorType::Q6_K` branch, lines 203-224

### Q6_K Block Structure

A Q6_K super-block contains 256 elements, 210 bytes total:

```
Offset  Size    Content
0       128     ql[128]   — Low 4-bit quantization values
128     64      qh[64]    — High 2-bit quantization values
192     16      scales[16] — i8 scale factors
208     2       d (fp16)  — Global scale factor
```

The 256 elements are organized as 2 halves × 128 elements. Each half has 8 scales (16 total), each scale covering 16 elements.

### Correct Scale Mapping

Reference: `dot_q6_k_q8_0_scalar` in `avx2.rs` implements dequantization **correctly**.

The dequantized `a[256]` is linearly arranged, with scale `sc[g]` applied to `a[g*16 .. g*16+15]`:

| Element Range | Scale Index |
|---------------|-------------|
| a[0..15] | sc[0] |
| a[16..31] | sc[1] |
| a[32..47] | sc[2] |
| a[48..63] | sc[3] |
| a[64..79] | sc[4] |
| a[80..95] | sc[5] |
| a[96..111] | sc[6] |
| a[112..127] | sc[7] |
| a[128..143] | sc[8] |
| a[144..159] | sc[9] |
| a[160..175] | sc[10] |
| a[176..191] | sc[11] |
| a[192..207] | sc[12] |
| a[208..223] | sc[13] |
| a[224..239] | sc[14] |
| a[240..255] | sc[15] |

### The Bug in Current Code

```rust
let mut sc_pos = off + 192;
for _ in 0..2 {
    for l in 0..32 {
        let g = l / 16;  // 0 (l<16) or 1 (l>=16)
        out[out_pos + l + 0]  = d * sc[sc_pos + g * 8 + 0] * (...);
        out[out_pos + l + 32] = d * sc[sc_pos + g * 8 + 2] * (...);
        out[out_pos + l + 64] = d * sc[sc_pos + g * 8 + 4] * (...);
        out[out_pos + l + 96] = d * sc[sc_pos + g * 8 + 6] * (...);
    }
    out_pos += 128; ql_pos += 64; qh_pos += 32; sc_pos += 8;
}
```

Expanded analysis of scale access per half:

**First half** (`_ = 0`, `sc_pos = off + 192`):

| Loop | g | Scale byte offset accessed | Actual scale index | Correct scale index | Correct? |
|------|---|---------------------------|-------------------|--------------------|---------|
| l=0..15, +0 | 0 | sc_pos + 0 | sc[0] | sc[0] | ✓ |
| l=0..15, +32 | 0 | sc_pos + 2 | sc[2] | sc[2] | ✓ |
| l=0..15, +64 | 0 | sc_pos + 4 | sc[4] | sc[4] | ✓ |
| l=0..15, +96 | 0 | sc_pos + 6 | sc[6] | sc[6] | ✓ |
| l=16..31, +0 | 1 | sc_pos + **8** | sc[**8**] | sc[**1**] | ✗ |
| l=16..31, +32 | 1 | sc_pos + **10** | sc[**10**] | sc[**3**] | ✗ |
| l=16..31, +64 | 1 | sc_pos + **12** | sc[**12**] | sc[**5**] | ✗ |
| l=16..31, +96 | 1 | sc_pos + **14** | sc[**14**] | sc[**7**] | ✗ |

**Second half** (`_ = 1`, `sc_pos = off + 200`):

| Loop | g | Scale byte offset accessed | Actual scale index | Correct scale index | Correct? |
|------|---|---------------------------|-------------------|--------------------|---------|
| l=0..15, +0 | 0 | sc_pos + 0 | sc[8] | sc[8] | ✓ |
| l=0..15, +32 | 0 | sc_pos + 2 | sc[10] | sc[10] | ✓ |
| l=0..15, +64 | 0 | sc_pos + 4 | sc[12] | sc[12] | ✓ |
| l=0..15, +96 | 0 | sc_pos + 6 | sc[14] | sc[14] | ✓ |
| l=16..31, +0 | 1 | sc_pos + **8** | sc[**16**] | sc[**9**] | ✗ **OUT OF BOUNDS** |
| l=16..31, +32 | 1 | sc_pos + **10** | sc[**18**] | sc[**11**] | ✗ **OUT OF BOUNDS** |
| l=16..31, +64 | 1 | sc_pos + **12** | sc[**20**] | sc[**13**] | ✗ **OUT OF BOUNDS** |
| l=16..31, +96 | 1 | sc_pos + **14** | sc[**22**] | sc[**15**] | ✗ **OUT OF BOUNDS** |

### Root Cause

The stride `g * 8` is wrong. `g` only takes values 0 and 1, so `g * 8` causes the second half's `l >= 16` portion to read far beyond the 16-element scale array.

The correct stride should be 1 (i.e., `g * 1` or simply `g`):
- First half: g=0 → sc[0,2,4,6], g=1 → sc[1,3,5,7]
- Second half: g=0 → sc[8,10,12,14], g=1 → sc[9,11,13,15]

### Impact

- First half `l >= 16` (64 of 128 elements): wrong scale values applied
- Second half `l >= 16` (64 of 128 elements): **out-of-bounds read**, results unpredictable
- 128 of 256 elements affected total (50%)
- Embedding lookup produces incorrect values, degrading output quality

Note: `dot_q6_k_q8_0_scalar` in `avx2.rs` (used for CPU-path matmul) is **not affected by this bug** — its dequantization and scale application are separated. The dequantize loop does not use scales; scales are applied correctly in the subsequent dot product loop as `sc[g]` (g=0..15 sequential).

### Fix

Change `g * 8` to `g`:

```rust
// Before
out[out_pos + l + 0]  = d * (t.data[sc_pos + g * 8 + 0] as i8 as f32) * (...);
out[out_pos + l + 32] = d * (t.data[sc_pos + g * 8 + 2] as i8 as f32) * (...);
out[out_pos + l + 64] = d * (t.data[sc_pos + g * 8 + 4] as i8 as f32) * (...);
out[out_pos + l + 96] = d * (t.data[sc_pos + g * 8 + 6] as i8 as f32) * (...);

// After
out[out_pos + l + 0]  = d * (t.data[sc_pos + g + 0] as i8 as f32) * (...);
out[out_pos + l + 32] = d * (t.data[sc_pos + g + 2] as i8 as f32) * (...);
out[out_pos + l + 64] = d * (t.data[sc_pos + g + 4] as i8 as f32) * (...);
out[out_pos + l + 96] = d * (t.data[sc_pos + g + 6] as i8 as f32) * (...);
```

### Files Changed

| File | Change |
|------|--------|
| `src/models/qwen2/forward.rs` | `embed_tokens` Q6_K branch: `g * 8` → `g` (4 occurrences) |

---

## Bug 4（致命）：Q4_K Metal kernel scale/min 解码格式错误

### 位置

`src/metal.metal` — `kernel_q4_k_f32_matmul`，第 303-310 行

### Q4_K Block 的 scale/min 布局

Q4_K super-block 共 144 字节，包含 256 个元素：

```
偏移    大小    内容
0       2       d (fp16)   — 全局 scale
2       2       dmin (fp16) — 全局 min scale
4       12      scales[12] — 8 个 scale + 8 个 min，各 6-bit
16      128     qs[128]    — 4-bit 量化值（低 4 位 + 高 4 位）
```

12 字节 `scales` 区域的正确布局（**分离式**）：

```
字节 0-5:  8 个 scale（各 6-bit，共 48 bit）
字节 6-11: 8 个 min  （各 6-bit，共 48 bit）
```

每 3 字节编码 4 个 6-bit 值：

```
对于字节组 (a, b, c):
  val0 = a[5:0]
  val1 = a[7:6] | (b[3:0] << 2)
  val2 = b[7:4] | (c[1:0] << 4)
  val3 = c[7:2]
```

### 问题

原代码使用**交错式**索引提取 scale 和 min：

```metal
int s3h = s * 3 >> 1;        // 字节偏移 = s * 1.5
int s3m = s * 3 & 1;         // 奇偶
int sh  = s3m << 2;           // 位移
float dsc0 = bd0 * float((sc0[s3h]     >> sh) & 0x3F);   // scale
float dmn0 = bm0 * float((sc0[3 + s3h] >> sh) & 0x3F);   // min
```

这假设 12 字节是 (scale, min) 交错对，每对 3 字节。但实际 GGUF 格式是 **scale 和 min 分离存储**的。

### 每个子块的 scale/min 访问分析

| 子块 s | Metal 读取 scale 来源 | 正确来源 | Metal 读取 min 来源 | 正确来源 | 正确？ |
|--------|---------------------|---------|-------------------|---------|-------|
| 0 | sc[0] shift 0 → byte0[5:0] | byte0[5:0] | sc[3] shift 0 → byte3[5:0] | byte6[5:0] | scale ✓, min ✗ |
| 1 | sc[1] shift 4 → 错误位 | byte0[7:6]\|byte1[3:0]<<2 | sc[4] shift 4 → 错误位 | byte6[7:6]\|byte7[3:0]<<2 | ✗ |
| 2 | sc[2] shift 0 → 错误位 | byte1[7:4]\|byte2[1:0]<<4 | sc[5] shift 0 → 错误位 | byte7[7:4]\|byte8[1:0]<<4 | ✗ |
| 3 | sc[3] shift 4 → 错误位 | byte2[7:2] | sc[6] shift 4 → 错误位 | byte8[7:2] | ✗ |
| 4 | sc[6] shift 0 → 错误位 | byte3[5:0] | sc[9] shift 0 → 错误位 | byte9[5:0] | ✗ |
| 5 | sc[7] shift 4 → 错误位 | byte3[7:6]\|byte4[3:0]<<2 | sc[10] shift 4 → 错误位 | byte9[7:6]\|byte10[3:0]<<2 | ✗ |
| 6 | sc[8] shift 0 → **qs 数据区** | byte4[5:0] | sc[11] shift 0 → **qs 数据区** | byte10[5:0] | ✗ 越界 |
| 7 | sc[9] shift 4 → **qs 数据区** | byte4[7:2] | sc[12] → **qs 数据区** | byte10[7:2] | ✗ 越界 |

**只有子块 0 的 scale 正确**。子块 1-7（87.5% 的元素）使用了完全错误的 scale 和 min 值。子块 6-7 的 min 读取甚至越界到了量化数据区（qs）。

### 根因

Metal kernel 的 `s * 3 >> 1` 公式假设 scale 和 min 交错排列，但 GGUF Q4_K 格式将 8 个 scale 和 8 个 min **分别**打包在前 6 字节和后 6 字节中。CPU 路径（`avx2.rs` 的 `memcpy + KMASK` 解包）正确实现了此格式，但 Metal kernel 没有。

### 影响

- 所有 GPU 上的 Q4_K 矩阵乘法产生错误结果
- Qwen2.5-1.5B 的大部分权重是 Q4_K → 所有 FFN 和注意力投影计算全部错误
- 输出完全混乱

### 修复方案

将 scale/min 提取改为正确的分离式 6-bit 解包：

```metal
// 前 6 字节：8 个 scale
uchar s0 = sc[0] & 0x3F;
uchar s1 = ((sc[0] >> 6) & 3) | ((sc[1] & 0xF) << 2);
uchar s2 = ((sc[1] >> 4) & 3) | ((sc[2] & 3) << 4);
uchar s3 = (sc[2] >> 2) & 0x3F;
uchar s4 = sc[3] & 0x3F;
uchar s5 = ((sc[3] >> 6) & 3) | ((sc[4] & 0xF) << 2);
uchar s6 = ((sc[4] >> 4) & 3) | ((sc[5] & 3) << 4);
uchar s7 = (sc[5] >> 2) & 0x3F;
// 后 6 字节：8 个 min（同样方式解包）
uchar m0 = sc[6] & 0x3F;
// ... 同理 m1-m7
```

### 涉及文件

| 文件 | 修改 |
|------|------|
| `src/metal.metal` | `kernel_q4_k_f32_matmul`：替换 scale/min 提取逻辑 |

---

## Fix Priority

| Priority | Bug | Impact | Change Size |
|----------|-----|--------|-------------|
| **P0** | Bug 1: Attention kernel hd overflow | Completely garbled output (fatal) | 1 line |
| **P0** | Bug 3: Q6_K embed scale index | 50% of embedding values wrong | 4× `g*8` → `g` |
| **P0** | Bug 4: Q4_K Metal kernel scale/min format | 87.5% of Q4_K sub-blocks use wrong scale/min | ~40 lines |
| **P0** | Bug 5: Q6_K embedding dequantization loop structure | Complex nested loops with incorrect scale indexing | Complete rewrite |
| **P1** | Bug 2: output_norm_gpu type restriction | GPU end-to-end inference fails, CPU fallback extremely slow | ~5 lines |

Recommended fix order: Bug 1 → Bug 3 → Bug 4 → Bug 5 → Bug 2. After fixing Bugs 1, 3, 4, and 5, the model should produce correct text. After fixing Bug 2, GPU can complete inference end-to-end with a major performance improvement.

---

## Bug 5（致命）：Q6_K Embedding Dequantization Loop Structure Error

### Location

`src/models/qwen2/forward.rs` — `embed_tokens`, `TensorType::Q6_K` branch, lines 217-237

### Problem

The original Q6_K embedding dequantization used a complex nested loop structure that attempted to process elements in an interleaved pattern while accessing scales with stride-2 indexing:

```rust
for _ in 0..2 {
    for l in 0..32 {
        let g = l / 16;  // g is 0 or 1
        let ql0 = t.data[ql_pos + l] as i32; 
        let ql1 = t.data[ql_pos + l + 32] as i32; 
        let qh = t.data[qh_pos + l] as i32;
        out[out_pos + l + 0]  = d * (t.data[sc_pos + g + 0] as i8 as f32) * (...);
        out[out_pos + l + 32] = d * (t.data[sc_pos + g + 2] as i8 as f32) * (...);
        out[out_pos + l + 64] = d * (t.data[sc_pos + g + 4] as i8 as f32) * (...);
        out[out_pos + l + 96] = d * (t.data[sc_pos + g + 6] as i8 as f32) * (...);
    }
    out_pos += 128; ql_pos += 64; qh_pos += 32; sc_pos += 8;
}
```

This code has multiple issues:

1. **Scale access pattern**: Uses `sc_pos + g + 0`, `sc_pos + g + 2`, etc., which accesses scales at positions 0,2,4,6 then 1,3,5,7 instead of sequential 0,1,2,3...
2. **Complex interleaving**: The nested loop tries to match llama.cpp's internal dequantization array layout but applies it incorrectly to direct output writing
3. **Position tracking**: Manual offset updates (`out_pos += 128`, etc.) make it hard to verify correctness

### Root Cause

The code appears to be attempting to replicate llama.cpp's `dequantize_row_q6_K` internal logic, which dequantizes into a temporary array `a[256]` using an interleaved pattern. However, when writing directly to output, this pattern doesn't translate correctly without proper index mapping.

Reference implementation from `avx2.rs` shows the correct approach:
- Dequantize all 256 values sequentially
- Access scales sequentially: `sc[g]` where g goes from 0 to 15
- Each scale covers 16 consecutive elements

### Fix

Rewrite the dequantization loop to match llama.cpp's pattern more directly:

```rust
// Dequantize 256 values following llama.cpp's dequantize_row_q6_K pattern
for ei in 0..256 {
    let lo = (t.data[off + (ei >> 1)] >> ((ei & 1) * 4)) as i32 & 0xF;
    let hi = (t.data[off + 128 + (ei >> 2)] >> ((ei & 3) * 2)) as i32 & 0x3;
    let q = (lo | (hi << 4)) - 32;
    let sc_idx = ei >> 4; // ei / 16, gives 0..15
    out[base_out + ei] = d * (t.data[off + 192 + sc_idx] as i8 as f32) * q as f32;
}
```

Key improvements:
- Single loop over all 256 elements
- Sequential element index `ei` makes bit manipulation clear
- Scale index `sc_idx = ei >> 4` gives correct sequential access (0..15)
- Direct correspondence with llama.cpp's indexing pattern

### Files Changed

| File | Change |
|------|--------|
| `src/models/qwen2/forward.rs` | Complete rewrite of Q6_K embedding dequantization loop |

### Verification Status

After applying this fix along with Bug 4 (Q4_K scale/min format), testing shows:
- Model generates varied output instead of single repeated token
- Output still contains repetitive patterns suggesting additional issues may exist
- Both CPU and GPU paths produce different outputs, indicating potential remaining discrepancies
- Further investigation needed to identify root cause of continued incorrect behavior
