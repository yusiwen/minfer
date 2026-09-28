//! `#[cfg(test)] mod kv_dtype_tests` for `src/cuda.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::{format_of, layout_of, KV_LAYOUT_F16, KV_LAYOUT_F32, KV_LAYOUT_Q8_0};
use crate::graph::kvformat::KvFormat;

/// The three layouts are a total, one-to-one mapping — the tag a `CudaBackend`
/// holds and the format its engine resolved cannot disagree.
#[test]
fn the_layout_tag_is_the_format_discriminant() {
    assert_eq!(layout_of(KvFormat::F32), KV_LAYOUT_F32);
    assert_eq!(layout_of(KvFormat::F16), KV_LAYOUT_F16);
    assert_eq!(layout_of(KvFormat::Q8_0), KV_LAYOUT_Q8_0);
    for f in [KvFormat::F32, KvFormat::F16, KvFormat::Q8_0] {
        assert_eq!(format_of(layout_of(f)), f, "{f:?} round trip");
    }
}

/// C4 S2b: the packed layout is a third value, not `false`. The pre-S2b bool
/// mapped anything that was not exactly `f16` to f32, so a `q8_0` region would
/// have been handed to the f32 kernels.
#[test]
fn the_packed_layout_is_not_the_f32_one() {
    assert_eq!(format_of(KV_LAYOUT_Q8_0), KvFormat::Q8_0);
    assert_ne!(format_of(KV_LAYOUT_Q8_0), KvFormat::F32);
    assert_ne!(format_of(KV_LAYOUT_Q8_0), KvFormat::F16);
}
