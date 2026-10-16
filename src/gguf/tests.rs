//! `#[cfg(test)] mod tests` for `src/gguf.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn split_file_info_parses_pattern() {
    let (prefix, idx, count) =
        split_file_info("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf").unwrap();
    assert_eq!(prefix, "qwen2.5-7b-instruct-q4_k_m");
    assert_eq!(idx, 0);
    assert_eq!(count, 2);

    let (_, idx2, count2) =
        split_file_info("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf").unwrap();
    assert_eq!(idx2, 1);
    assert_eq!(count2, 2);

    // not a split file
    assert!(split_file_info("qwen2.5-0.5b-instruct-q4_k_m.gguf").is_none());
    // invalid idx > count
    assert!(split_file_info("foo-00099-of-00002.gguf").is_none());
}

#[test]
fn resolve_splits_builds_all_parts() {
    let parts = resolve_splits(std::path::Path::new("/m/foo-00001-of-00003.gguf")).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(
        parts[0].file_name().unwrap().to_str().unwrap(),
        "foo-00001-of-00003.gguf"
    );
    assert_eq!(
        parts[1].file_name().unwrap().to_str().unwrap(),
        "foo-00002-of-00003.gguf"
    );
    assert_eq!(
        parts[2].file_name().unwrap().to_str().unwrap(),
        "foo-00003-of-00003.gguf"
    );
    // non-first part rejected
    assert!(resolve_splits(std::path::Path::new("/m/foo-00002-of-00003.gguf")).is_none());
}

/// A tensor info with a shape the size helper can compute (`ne[0]` elements of
/// `type_`), for the recorded-offset-table tests below.
fn info_at(name: &str, offset: u64, ne0: i64, type_: GgmlType) -> GgufTensorInfo {
    // Strides exactly as the parser fills them (gguf.cpp lines 728-732): the
    // size helper reads them, not the raw shape.
    let blck = type_.blck_size() as usize;
    let mut nb = [0usize; GGML_MAX_DIMS];
    nb[0] = type_.type_size();
    nb[1] = nb[0] * (ne0 as usize / blck);
    for j in 2..GGML_MAX_DIMS {
        nb[j] = nb[j - 1];
    }
    GgufTensorInfo {
        name: name.to_string(),
        ne: [ne0, 1, 1, 1],
        nb,
        type_,
        offset,
    }
}

#[test]
fn a_data_section_with_padding_between_tensors_is_accepted() {
    // #575: the offsets are the file's, not an accumulated total. These are the
    // first three tensors of `rwkv-6-world-1.6b-Q2_K.gguf`'s recorded layout,
    // where a 7 340 032-byte gap follows the embedding — a legal file the loader
    // used to reject.
    let info = vec![
        info_at("token_embd.weight", 0, 2048, GgmlType::Q2_K),
        info_at("blk.0.attn_norm.weight", 44_040_192, 2048, GgmlType::F32),
    ];
    let size = data_section_size(&info, 100_000_000).expect("a padded layout is accepted");
    // The section ends at the *furthest* tensor end, not at a running total.
    assert_eq!(size, 44_040_192 + 2048 * 4);
}

#[test]
fn a_tensor_that_spans_past_the_part_is_refused_by_name() {
    // The bounds check replaced the contiguity check: a truncated part (or a
    // corrupt offset) is still a refusal, and the message names the tensor.
    let info = vec![info_at(
        "blk.0.attn_norm.weight",
        44_040_192,
        2048,
        GgmlType::F32,
    )];
    let err = data_section_size(&info, 44_040_192).unwrap_err();
    assert!(err.contains("attn_norm.weight"), "{err}");
    assert!(err.contains("spans bytes"), "{err}");
}
