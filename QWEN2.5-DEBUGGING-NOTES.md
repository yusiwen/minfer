# Qwen2.5-1.5B Debugging Notes - Current Status

## Summary

After fixing multiple documented bugs (Bug 1, 3, 4, 5), Qwen2.5-1.5B still produces incorrect output. The model generates varied text instead of a single repeated token (improvement), but the output contains repetitive patterns and is incoherent.

## Fixes Applied

### Bug 1: Metal Attention Kernel HD Overflow ✓
- File: `src/metal.metal`
- Change: `float4 oc[16]` → `float4 oc[32]`
- Impact: Supports hd=128 for Qwen2.5-1.5B

### Bug 3: Q6_K Embedding Scale Index ✓
- File: `src/models/qwen2/forward.rs`
- Issue: Original code used `g * 8` stride causing out-of-bounds access
- Fix: Complete rewrite with sequential scale access (sc_idx = ei >> 4)

### Bug 4: Q4_K Scale/Min Interleaved Format ✓
- Files: `src/metal.metal`, `src/avx2.rs`, `src/models/qwen2/forward.rs`
- Issue: Assumed separate format (bytes 0-5=scales, 6-11=mins)
- Fix: Changed to interleaved format matching llama.cpp
  - Bytes 0-2: scales[0-3]
  - Bytes 3-5: mins[0-3]
  - Bytes 6-8: scales[4-7]
  - Bytes 9-11: mins[4-7]

### Bug 5: Q6_K Dequantization Loop Structure ✓
- File: `src/models/qwen2/forward.rs`
- Issue: Complex nested loops with unclear indexing
- Fix: Simple sequential loop matching llama.cpp pattern

## Current Behavior

### GPU Path (Metal on M4 Pro)
```
Prompt: "Paris"
Output: "体质lastname的那个体质lastname的那个体质touches体质..."
- Mixed Chinese/English
- Some coherent words (lastname, touches, Trees, Name)
- Repetitive patterns
- Speed: ~45 tok/s generation, ~70 tok/s prefill
```

### CPU Path (MINFER_DISABLE_MPS=1)
```
Prompt: "Paris"
Output: "OrWSTRWSTRWSTRWSTRWSTRWSTRWSTRWSTR..."
- Starts with "Or" then repeats "WSTR"
- Completely incoherent
- Speed: ~2 tok/s generation, ~2 tok/s prefill (VERY SLOW)
```

## Key Observations

1. **CPU and GPU produce DIFFERENT outputs** - This is critical evidence that there's still a discrepancy between the two paths

2. **"WSTR" pattern on CPU** - This looks like UTF-16 wide string encoding or byte-order issue. Could indicate:
   - Incorrect byte interpretation (reading u16 as bytes or vice versa)
   - Memory layout mismatch
   - Quantization parameter error

3. **GPU output has variety** - Suggests the computation is partially working but producing wrong values

4. **Both paths use same dequantization fixes** - So the difference must be elsewhere

## Hypotheses for Remaining Issues

### H1: Activation Quantization Mismatch
The CPU path quantizes f32 activations to Q8_0 before matmul (`cpu_quant_matmul_f32`). Maybe this quantization is incorrect or uses wrong parameters.

### H2: Tensor Data Loading Issue
Maybe the GGUF tensor data isn't being loaded correctly, or there's padding/alignment issue we haven't accounted for.

### H3: RoPE or Positional Encoding Bug
Qwen2.5 uses RoPE frequency base of 1,000,000 (vs 10,000 for older models). Maybe there's an issue with how we apply RoPE.

### H4: Attention Mechanism Bug
Even though we fixed the hd overflow, there might be other issues in the attention computation (GQA, softmax, etc.).

### H5: FFN Layer Bug
The SwiGLU activation or FFN computation might have issues.

### H6: Output Logits Computation
The final RMSNorm + output projection might have bugs.

## Recommended Next Steps

### Priority 1: Validate Dequantization Correctness
Create a unit test that:
1. Loads a known Q6_K block from the GGUF file
2. Dequantizes it using our code
3. Compares against manually computed expected values
4. Do the same for Q4_K blocks

### Priority 2: Compare with llama.cpp Directly
1. Install llama.cpp
2. Run the same prompt with the same model file
3. Compare intermediate outputs (embeddings, layer outputs, logits)
4. Identify where the divergence occurs

### Priority 3: Add Instrumentation
Add debug output to print:
- First few embedding vectors after dequantization
- Attention scores for first token
- FFN outputs for first layer
- Final logits distribution

### Priority 4: Check Byte Order
Verify that all fp16→f32 conversions handle endianness correctly, especially for:
- Scale factors
- Delta values
- RoPE frequencies

### Priority 5: Test Simpler Model
Find or create a minimal Q4_K/Q6_K model to test with fewer layers and smaller dimensions.

## Code Review Checklist

- [ ] Verify Q4_K dequantization matches llama.cpp byte-for-byte
- [ ] Verify Q6_K dequantization matches llama.cpp byte-for-byte
- [ ] Check fp16_to_f32 conversion implementation
- [ ] Verify RoPE frequency calculation
- [ ] Check GQA key/value head mapping
- [ ] Verify softmax implementation
- [ ] Check SwiGLU activation formula
- [ ] Verify RMSNorm computation
- [ ] Check KV cache indexing
- [ ] Verify position ID handling

## Files Modified

- `src/metal.metal` - Q4_K kernel, attention kernel
- `src/avx2.rs` - Q4_K dot product, Q6_K dot product, test helpers
- `src/models/qwen2/forward.rs` - Q4_K & Q6_K embedding dequantization, GPU detection
- `src/metal.rs` - Minor cleanup
- `QWEN2.5-1.5B-BUGS.md` - Documentation

## Conclusion

We've made significant progress fixing documented bugs, but the model still doesn't work correctly. The different outputs from CPU and GPU paths suggest there are additional issues we haven't identified. Systematic comparison with llama.cpp and careful validation of each computation step will be needed to fully resolve this.
