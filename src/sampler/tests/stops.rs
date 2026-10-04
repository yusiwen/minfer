//! Stop-string matching, including a multibyte piece split across tokens.
//!
//! Split out of `src/sampler/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

// === Phase 1: stop strings (byte-wise suffix matching) ===

#[test]
fn test_stop_suffix_basic() {
    let buf = b"hello world";
    assert_eq!(match_stop_suffix(buf, &[b"world"]), Some(6));
    assert_eq!(match_stop_suffix(buf, &[b"hello"]), None, "not a suffix");
    assert_eq!(match_stop_suffix(buf, &[b"d"]), Some(10));
    assert_eq!(match_stop_suffix(buf, &[b"x"]), None);
    assert_eq!(match_stop_suffix(buf, &[]), None);
}
#[test]
fn test_stop_suffix_empty_and_too_long_ignored() {
    let buf = b"abc";
    assert_eq!(match_stop_suffix(buf, &[b"", b"abc", b"abcd"]), Some(0));
    assert_eq!(match_stop_suffix(buf, &[b"", b"zzz"]), None);
}
#[test]
fn test_stop_suffix_longest_wins() {
    // both "ab" and "b" are suffixes of "xab"; earliest start (longest) wins
    let buf = b"xab";
    assert_eq!(match_stop_suffix(buf, &[b"b", b"ab"]), Some(1));
    assert_eq!(match_stop_suffix(buf, &[b"ab", b"b"]), Some(1));
}
#[test]
fn test_stop_suffix_multibyte_split_across_tokens() {
    // U+4E2D = E4 B8 AD; first two bytes arrive in one token, last byte next
    let partial = [0xE4u8, 0xB8];
    assert_eq!(
        match_stop_suffix(&partial, &[&[0xE4, 0xB8, 0xAD]]),
        None,
        "stop longer than buf"
    );
    let complete = [0xE4u8, 0xB8, 0xAD, 0xE4, 0xB8, 0xAD];
    assert_eq!(
        match_stop_suffix(&complete, &[&[0xE4, 0xB8, 0xAD]]),
        Some(3)
    );
}
