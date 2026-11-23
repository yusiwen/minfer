// Metal shaders for minfer — Q4_0 matmul + element-wise ops.

#include <metal_stdlib>
using namespace metal;

constant int Q4B = 18;
constant int Q41B = 20;
constant int Q5B = 22;

// Shared kernel launch parameters
constant short NW_Q = 32;
constant short NQ_Q = 16;
constant short QK   = 32;

// ─── #724: llama3 rope scaling ─────────────────────────────────────────────
//
// A `llama3`-scaled GGUF carries one frequency factor per rotated pair
// (`blk.N.rope_freqs.weight`, or the table synthesised from
// `*.rope.scaling.*`). Entry `d` divides pair `d`'s inverse frequency, exactly
// as llama.cpp's own rope does (`theta_base / freq_factor`). `n == 0` means "no
// scaling": the kernel then evaluates the original expression unchanged, which
// is what keeps an unscaled model bit-identical.
#define ROPE_FREQS_MAX 128

struct RopeFreqs {
    int n;
    float v[ROPE_FREQS_MAX];
};

inline float rope_freq(constant RopeFreqs & f, int d, int n_dims, float freq_base, float freq_scale) {
    float freq = freq_scale / pow(freq_base, (2.0 * d) / n_dims);
    if (f.n > 0 && d < f.n) freq /= f.v[d];
    return freq;
}

// ─── #844: ONE rope-pair evaluation order, shared by every arm ──────────────
//
// The fused epilogue (`kernel_attn_bias_rope_store` / `kernel_attn_rope_store`,
// which rotate the q/k sections in registers) and the unfused reference chain
// (`kernel_add_bias_f32` ×3, then `kernel_rope_f32` ×2) must write the **same
// f32 words**. Written as `x0 * cs - x1 * sn` the compiler is free to contract
// the subtraction into an FMA in one kernel and not the other (measured: 103 of
// 1152 words differed by 2^-24, i.e. one rounding, in the fused arm). Naming the
// two roundings explicitly with `fma()` fixes the order at both sites, so the
// two chains cannot drift apart by a compiler decision — every rope output is
// `round(x1*sn)` then one fused multiply-add for r0, `round(x1*cs)` then one
// fused multiply-add for r1. `fma` is [MSL's](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
// single-rounding library call, not a hint.
//
// [#844]: https://github.com/yusiwen/minfer/issues/844
struct RopePair {
    float r0;
    float r1;
};

inline RopePair rope_rotate(float x0, float x1, float cs, float sn) {
    RopePair r;
    r.r0 = fma(x0, cs, -(x1 * sn));
    r.r1 = fma(x0, sn,   x1 * cs);
    return r;
}
