//! The CLI's size parser and the split-output stem rule.
//!
//! Split out of `src/tooling/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn parse_size_accepts_bytes_and_binary_suffixes() {
    assert_eq!(parse_size("1024").unwrap(), 1024);
    assert_eq!(parse_size("1K").unwrap(), 1024);
    assert_eq!(parse_size("2m").unwrap(), 2 * 1024 * 1024);
    assert_eq!(parse_size("1G").unwrap(), 1024 * 1024 * 1024);
    for bad in ["", "abc", "1.5M", "-4", "0", "0K"] {
        assert!(parse_size(bad).is_err(), "{bad:?}");
    }
}
#[test]
fn split_targets_strips_the_gguf_suffix() {
    let (dir, stem) = split_targets(Path::new("/tmp/f6-work/model.gguf"));
    assert_eq!(dir, Path::new("/tmp/f6-work"));
    assert_eq!(stem, "model");
}
