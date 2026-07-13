# Qwen2.5-1.5B Debugging Plan

## ✅ RESOLVED - Root Cause Found and Fixed

**Bug 6 Identified**: GPU attention kernel had KV cache indexing error in `kernel_gqa_attn_f32`.

### The Fix
Changed KV cache reads from:
```metal
device const float * khead = k + hk * hd;
device const float4 * k4 = (device const float4 *)(khead + kv0 * stride_kv);
```

To:
```metal
device const float4 * k4 = (device const float4 *)(k + kv0 * stride_kv + hk * hd);
```

Applied to all 4 accesses (K/V × kv0/kv1).

### Test Results
✅ **Before**: "体质lastname的那个体质..." (garbled)
✅ **After**: "Paris is the capital of France." (correct!)

See `DEBUGGING-SUMMARY.md` for full details.

## Hypothesis Priority

### H1: RoPE Frequency Base Mismatch (Most Likely)

Qwen2.5 uses `rope_freq_base = 1000000` (vs 10000 for older models). If this isn't being read correctly from GGUF or applied incorrectly, it would cause:
- Completely wrong attention patterns
- Different behavior on CPU vs GPU if one path has a bug
- Repetitive/garbled output

**Check:**
```rust
// In loader.rs, verify GGUF reads:
println!("rope_freq_base: {}", hparams.rope_freq_base); // Should be 1000000 for Qwen2.5

// In forward.rs apply_rope, verify freq calculation:
let freq = freq_scale / freq_base.powf(2.0 * i as f32 / hd as f32);
// For Qwen2.5: freq = 1.0 / 1000000^(2i/128)
```

**Test:** Add debug print of first few freqs for layer 0, compare against llama.cpp.

---

### H2: Attention Bias Handling Bug

Qwen2 has attention biases (bq, bk, bv) unlike Llama. If these are:
- Not being loaded from GGUF
- Applied at wrong time
- Wrong shape/stride

This would cause systematic errors in attention computation.

**Check:**
```rust
// In forward.rs, after loading bq/bk/bv:
println!("bq shape: {:?}", bq.shape); // Should be [n_embd]
println!("bk shape: {:?}", bk.shape); // Should be [n_embd_kv]
println!("bv shape: {:?}", bv.shape); // Should be [n_embd_kv]

// After bias add:
println!("bq[0..5]: {:?}", &bq.data[..5]);
```

**Compare:** Run same prompt with llama.cpp, dump attention input/output for first token.

---

### H3: KV Cache Indexing Error

If KV cache is indexed wrong (wrong stride, wrong position offset), attention would attend to garbage positions.

**Check:**
```rust
// In gqa_attn function:
println!("nkv: {}, nk: {}, pos: {}", nkv, nk, pos);
// Verify k_cache layout: [n_layer][nkv][nk][hd]
// Access pattern should be: k_cache[layer][pos][head][dim]
```

**Test:** Print KV cache contents for first 2 tokens, verify they match expected values.

---

### H4: SwiGLU Activation Formula Error

If SwiGLU is computed wrong (`silu(gate) * up`), FFN outputs would be garbage.

**Check:**
```rust
// In vec_ops.rs silu implementation:
fn vec_silu_f32(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = *val / (1.0 + (-*val).exp()); // Should be: x / (1 + e^-x)
    }
}

// In forward.rs after SwiGLU:
println!("bg[0..5] after silu*up: {:?}", &bg.data[..5]);
```

---

### H5: Output Projection Quantization Mismatch

The final `output.weight` projection might have quantization issues:
- Wrong type detection (Q6_K vs Q4_K)
- Incorrect fallback to CPU scalar path
- Byte order in fp16 conversion

**Check:**
```rust
// In metal.rs output_norm_gpu:
println!("output weight type: {:?}", output.ttype); // Should be Q6_K for Qwen2.5-1.5B
println!("output norm weight loaded: {}", output_norm.is_some());

// In forward.rs final matmul:
println!("logits[0..5]: {:?}", &logits.data[..5]);
```

---

### H6: GGUF Tensor Loading Issue

Tensors might be loaded with wrong strides or padding.

**Check:**
```rust
// In loader.rs after loading each tensor:
println!("{}: shape={:?}, type={:?}, size={}", 
         name, tensor.shape, tensor.ttype, tensor.data.len());
```

---

## Step-by-Step Debugging Plan

### Phase 1: Validate Hyperparameters (15 min)

1. **Print model config:**
   ```bash
   ./target/release/minfer <model> "test" --verbose
   ```
   Expected for Qwen2.5-1.5B:
   - n_embd=1536, n_head=12, n_head_kv=2, hd=128
   - n_layer=28, n_ff=8960
   - rope_freq_base=1000000

2. **Compare with GGUF metadata:**
   ```bash
   gguf-dump <model.gguf> | grep -E "(n_embd|n_head|rope)"
   ```

---

### Phase 2: Instrument Forward Pass (30 min)

Add debug prints in `src/models/qwen2/forward.rs`:

```rust
// After embedding (line ~33):
eprintln!("EMBEDDING[0][0..5]: {:?}", &hidden[..5]);

// After RMSNorm (line ~92):
eprintln!("NORM[0][0..5]: {:?}", &bn[..5]);

// After RoPE (line ~102):
eprintln!("ROPE[0][0..5]: {:?}", &bq[..5]);

// After attention (line ~110):
eprintln!("ATTN[0][0..5]: {:?}", &ba[..5]);

// After FFN (line ~138):
eprintln!("FFN_OUT[0][0..5]: {:?}", &bn[..5]);

// Final logits (line ~145):
eprintln!("LOGITS[0..10]: {:?}", &logits[..10]);
```

Run same prompt on CPU and GPU, compare outputs at each stage.

---

### Phase 3: Compare Against llama.cpp (45 min)

1. **Install llama.cpp:**
   ```bash
   cd /home/yusiwen/git/ai/llama.cpp
   make -j
   ```

2. **Run inference with debug output:**
   ```bash
   ./build/bin/llama-cli -m <model.gguf> -p "Paris" -n 5 --verbose-prompt
   ```

3. **Extract intermediate values:**
   - Embedding vector for token "Paris"
   - Attention scores for first token
   - Final logits distribution

4. **Compare with minfer output:**
   - Find first divergence point
   - That's where the bug is

---

### Phase 4: Unit Test Dequantization (30 min)

Create test that loads known Q4_K/Q6_K block and verifies dequantization:

```rust
#[test]
fn test_q4k_dequant_matches_llamacpp() {
    // Hard-code a known Q4_K block from the model
    let q4_block = vec![...]; // First 144 bytes of token_embd.weight
    
    // Dequantize using minfer
    let mut out_minfer = vec![0.0f32; 256];
    embed_tokens_q4k(&q4_block, &mut out_minfer, &[0], 256);
    
    // Compare with llama.cpp reference (pre-computed)
    let expected = vec![...]; // From llama.cpp dequantize_row_q4_K
    
    for i in 0..256 {
        assert!((out_minfer[i] - expected[i]).abs() < 1e-5, 
                "Mismatch at {}: got {} expected {}", i, out_minfer[i], expected[i]);
    }
}
```

---

### Phase 5: Check Byte Order (15 min)

Verify all fp16→f32 conversions handle endianness:

```rust
// In block.rs fp16_to_f32:
pub fn fp16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h.to_le()).to_f32() // Ensure little-endian
}
```

Test with known fp16 value:
```rust
assert_eq!(fp16_to_f32(0x3C00), 1.0); // 0x3C00 = 1.0 in fp16
assert_eq!(fp16_to_f32(0x4000), 2.0); // 0x4000 = 2.0 in fp16
```

---

## Quick Wins to Try First

1. **Verify RoPE freq base** (5 min):
   ```bash
   strings <model.gguf> | grep -i "rope"
   ```

2. **Check if output weight is loaded** (5 min):
   ```bash
   gguf-dump <model.gguf> | grep "output.weight"
   # If missing, confirms weight tying (tok_embd cloned)
   ```

3. **Run with MINFER_DISABLE_MPS=1** to isolate CPU path, then compare first 10 logits with GPU.

4. **Add single debug print** after embedding to see if token lookup works:
   ```rust
   eprintln!("Token {} embedding sum: {}", token_id, hidden.iter().sum::<f32>());
   ```

---

## Expected Root Causes (Ranked)

1. **RoPE frequency base not set correctly** (40% probability)
   - Would explain garbled attention
   - Easy fix: ensure GGUF reader sets `rope_freq_base = 1000000`

2. **Attention bias not applied** (25% probability)
   - Qwen2 has biases, Llama doesn't
   - Fix: verify bq/bk/bv are added after matmul

3. **KV cache position indexing off-by-one** (15% probability)
   - Would cause attention to wrong positions
   - Fix: check `pos` variable in forward loop

4. **Output projection uses wrong quantization** (10% probability)
   - Q6_K treated as Q4_K or vice versa
   - Fix: verify type detection in output_norm_gpu

5. **Something else entirely** (10% probability)
   - Could be tokenizer, sampling, or template rendering

---

## Success Criteria

Model is working when:
- Prompt "Paris" produces coherent French-related text
- CPU and GPU produce identical outputs (within floating-point tolerance)
- Generation speed matches expectations (~45 tok/s GPU, ~5 tok/s CPU)
- No repeated patterns or gibberish

---

## Next Action

Start with **Phase 1** (validate hyperparameters) and **Phase 2** (instrument forward pass). These will quickly reveal if the issue is RoPE, attention biases, or something else.

If RoPE freq base is wrong, fix it and re-test. If hyperparameters are correct, the instrumentation will show exactly where CPU and GPU diverge.
