# Bug 6 (Critical): GPU Attention KV Cache Indexing Error

## Location

`src/metal.metal` — `kernel_gqa_attn_f32`, lines 735-736

## Problem

The GPU attention kernel computes the wrong memory address when reading from the KV cache.

### Current (Buggy) Code

```metal
int stride_kv = nk * hd;

device const float * khead = k + hk * hd;          // ❌ WRONG: base is [hk][0][0]
...
device const float4 * k4 = (device const float4 *)(khead + kv0 * stride_kv);  // Adds [kv0][0][0]
```

This computes: `k[hk * hd + kv0 * nk * hd]` which accesses elements in order `[hk][kv0][dim]`.

### Expected KV Cache Layout

From `cache.rs` line 26 and `store_multi` line 50:
```rust
let offset = pos * dim;  // dim = nk * hd
self.k[offset..offset + dim].copy_from_slice(k);
```

KV cache layout is **`[nkv][nk][hd]`** (positions first, then heads, then dimensions).

So element `[pos, head, dim_idx]` should be at:
```
offset = pos * (nk * hd) + head * hd + dim_idx
```

### Correct Code

```metal
int stride_kv = nk * hd;

// For each query position t and head h:
int hk = h / gqa;

// Access K/V at position kv0 for head hk
device const float * kbase = k + kv0 * stride_kv + hk * hd;  // ✅ CORRECT: [kv0][hk][0]
device const float4 * k4 = (device const float4 *)kbase;

// Similarly for V:
device const float * vbase = v + kv0 * stride_kv + hk * hd;  // ✅ CORRECT: [kv0][hk][0]
device const float4 * v4 = (device const float4 *)vbase;
```

## Root Cause

The buggy code assumes KV cache is laid out as `[nk][nkv][hd]` (heads first), but it's actually `[nkv][nk][hd]` (positions first). This is a classic row-major vs column-major confusion.

When `hk = 0`, the bug doesn't manifest because `hk * hd = 0`. But for any non-zero head index, the base pointer is completely wrong.

For Qwen2.5-1.5B with `nk=2, hd=128`:
- Head 0: `k + 0 * 128 + kv0 * 256` → accesses `k[kv0 * 256]` (WRONG, should be `k[kv0 * 256]`)
- Head 1: `k + 1 * 128 + kv0 * 256` → accesses `k[128 + kv0 * 256]` (WRONG, should be `k[kv0 * 256 + 128]`)

The second case shows the error: we're accessing position `kv0`'s data at an offset of 128 floats from the start, but we should be accessing position `kv0` at offset `kv0 * 256 + 128`.

## Impact

- **All GPU attention computations are wrong** for any model with multiple KV heads
- Each head attends to the wrong positions in the KV cache
- Produces completely garbled output (explains the random characters)
- CPU path works correctly (different indexing logic)
- Explains why CPU ≠ GPU output

## Fix

In `src/metal.metal`, `kernel_gqa_attn_f32`, replace lines 735-736:

**Before:**
```metal
device const float * khead = k + hk * hd;
device const float * vhead = v + hk * hd;
device       float * ohead = o + t * ne_q + h * hd;

int hd4 = hd / 4;
device const float4 * q4 = (device const float4 *)qhead;

const int NE = 2;
const int C = 32 * NE;

float mx = -INFINITY;
float S = 0.0f;
float4 oc[32];
for (int i = 0; i < hd4; i++) oc[i] = (float4)0.0f;

for (int batch = 0; batch < nkv; batch += C) {
    float s0 = -INFINITY, s1 = -INFINITY;
    int kv0 = batch + tiisg * NE;
    int kv1 = kv0 + 1;

    if (kv0 < nkv) {
        device const float4 * k4 = (device const float4 *)(khead + kv0 * stride_kv);  // ❌ BUG
        float d = 0.0f;
        for (int i = 0; i < hd4; i++) d += dot(q4[i], k4[i]);
        s0 = d * scale;
    }
    if (kv1 < nkv) {
        device const float4 * k4 = (device const float4 *)(khead + kv1 * stride_kv);  // ❌ BUG
        float d = 0.0f;
        for (int i = 0; i < hd4; i++) d += dot(q4[i], k4[i]);
        s1 = d * scale;
    }
    
    // ... rest of softmax ...
    
    if (kv0 < nkv) {
        device const float4 * v4 = (device const float4 *)(vhead + kv0 * stride_kv);  // ❌ BUG
        for (int i = 0; i < hd4; i++) oc[i] += e0 * v4[i];
    }
    if (kv1 < nkv) {
        device const float4 * v4 = (device const float4 *)(vhead + kv1 * stride_kv);  // ❌ BUG
        for (int i = 0; i < hd4; i++) oc[i] += e1 * v4[i];
    }
}
```

**After:**
```metal
device const float * qhead = q + t * ne_q + h * hd;
device       float * ohead = o + t * ne_q + h * hd;

int hd4 = hd / 4;
device const float4 * q4 = (device const float4 *)qhead;

const int NE = 2;
const int C = 32 * NE;

float mx = -INFINITY;
float S = 0.0f;
float4 oc[32];
for (int i = 0; i < hd4; i++) oc[i] = (float4)0.0f;

for (int batch = 0; batch < nkv; batch += C) {
    float s0 = -INFINITY, s1 = -INFINITY;
    int kv0 = batch + tiisg * NE;
    int kv1 = kv0 + 1;

    if (kv0 < nkv) {
        device const float4 * k4 = (device const float4 *)(k + kv0 * stride_kv + hk * hd);  // ✅ FIXED
        float d = 0.0f;
        for (int i = 0; i < hd4; i++) d += dot(q4[i], k4[i]);
        s0 = d * scale;
    }
    if (kv1 < nkv) {
        device const float4 * k4 = (device const float4 *)(k + kv1 * stride_kv + hk * hd);  // ✅ FIXED
        float d = 0.0f;
        for (int i = 0; i < hd4; i++) d += dot(q4[i], k4[i]);
        s1 = d * scale;
    }
    
    // ... rest of softmax ...
    
    if (kv0 < nkv) {
        device const float4 * v4 = (device const float4 *)(v + kv0 * stride_kv + hk * hd);  // ✅ FIXED
        for (int i = 0; i < hd4; i++) oc[i] += e0 * v4[i];
    }
    if (kv1 < nkv) {
        device const float4 * v4 = (device const float4 *)(v + kv1 * stride_kv + hk * hd);  // ✅ FIXED
        for (int i = 0; i < hd4; i++) oc[i] += e1 * v4[i];
    }
}
```

Key changes:
1. Removed `khead` and `vhead` variables (they were computing wrong base addresses)
2. Changed all 4 accesses to use direct calculation: `k + kv * stride_kv + hk * hd`
3. Same fix for both K and V reads

## Files Changed

| File | Change |
|------|--------|
| `src/metal.metal` | Fix 4 KV cache accesses in `kernel_gqa_attn_f32` |

## Testing

After applying this fix:

1. **Rebuild:**
   ```bash
   cargo build --release
   ```

2. **Test inference:**
   ```bash
   ./target/release/minfer ~/.cache/minfer/models/hf/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_0.gguf "Paris"
   ```

3. **Expected result:**
   - Output should be coherent French-related text
   - CPU and GPU should produce identical outputs (within floating-point tolerance)
   - No more "WSTR" pattern on CPU or random characters on GPU

4. **Verify against llama.cpp:**
   ```bash
   cd /home/yusiwen/git/ai/llama.cpp
   ./build/bin/llama-cli -m ~/.cache/minfer/models/hf/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_0.gguf -p "Paris" -n 10
   ```

## Why This Was Hard to Find

1. **Only affects multi-head models**: Qwen2-0.5B has `nk=2`, so the bug exists but might not have been obvious during initial testing
2. **Silent corruption**: Wrong memory access doesn't crash, just produces garbage
3. **Different from CPU path**: The CPU implementation uses correct indexing, so comparison would reveal the discrepancy
4. **Symptoms look like quantization bugs**: Garbled output could be mistaken for dequantization errors

## Priority

**P0 (Critical)** - This is likely THE root cause of all Qwen2.5-1.5B issues. Fix this first before investigating anything else.
