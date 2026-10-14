//! `#[cfg(test)] mod arg_parse_tests` for `src/main.rs`.
//!
//! The CLI parsers whose *output* is the contract: `main()`'s argv loop itself
//! is only reachable by running the binary, so each parser is tested as the
//! pure function the loop calls.

use super::parse_dry_breakers;

/// #118: `--dry-sequence-breakers` (plural) stays the **token-id** spelling.
///
/// The numeric form still resolves; a non-numeric entry is still the loud parse
/// error it always was — and its message now points at the flag that does take a
/// breaker string, rather than at a closed issue.
#[test]
fn dry_breakers_flag_takes_token_ids_only() {
    assert!(parse_dry_breakers("").unwrap().is_empty());
    assert!(parse_dry_breakers("  ").unwrap().is_empty());
    assert_eq!(
        parse_dry_breakers("198;13,2").unwrap(),
        vec![vec![198u32], vec![13, 2]]
    );
    // "198" is a *number* under this flag, so it is the token id 198.
    assert_eq!(parse_dry_breakers("198").unwrap(), vec![vec![198u32]]);

    let err = parse_dry_breakers("\\n").unwrap_err();
    assert!(err.contains("token ids only"), "{err}");
    assert!(err.contains("--dry-sequence-breaker"), "{err}");
    assert!(err.contains("#118"), "{err}");
    assert!(
        !err.contains("#48"),
        "the closed issue must not be cited: {err}"
    );
}
