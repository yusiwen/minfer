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
