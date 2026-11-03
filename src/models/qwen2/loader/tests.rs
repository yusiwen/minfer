//! The loader's pure decisions. `#723`'s rope layout is the one that was wrong for
//! a whole architecture, so it is pinned here rather than only in a real-model run.

use super::*;

#[test]
fn the_rope_layout_follows_the_architecture_not_the_family() {
    // llama.cpp: LLM_ARCH_LLAMA rotates **consecutive** head values; Qwen2/Qwen3
    // rotate the two halves (NeoX).
    assert_eq!(rope_style_for(Some("llama")), RopeStyle::Interleaved);
    for arch in [Some("qwen2"), Some("qwen3"), None] {
        assert_eq!(rope_style_for(arch), RopeStyle::NonInterleaved, "{arch:?}");
    }
}
