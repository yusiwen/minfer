//! RoPE style vocabulary (the family the graph's RoPE op names).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RopeStyle {
    NonInterleaved = 0, // Qwen2: pairs [i, i+half_dim]
    /// LLaMA/Mistral: pairs [2*i, 2*i+1] — kept for llama.cpp parity.
    ///
    /// Constructed by `graph::op_matrix` (its `RoPE` cases) only, so the
    /// allowance names `not(test)`; what would construct it in production is an
    /// architecture whose `rope_style` is interleaved (a LLaMA-family loader).
    /// [#244]'s verdict: keep the vocabulary, report the membership question.
    ///
    /// [#244]: https://github.com/yusiwen/minfer/issues/244
    #[cfg_attr(not(test), allow(dead_code))]
    Interleaved = 1, // LLaMA/Mistral: pairs [2*i, 2*i+1]
}
