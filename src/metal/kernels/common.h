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
