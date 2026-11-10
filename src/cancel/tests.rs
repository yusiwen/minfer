//! `#[cfg(test)] mod tests` for `src/cancel.rs`.
use super::*;

/// #796: a clone shares the flag a generator checks, which is the only
/// cross-thread invariant the type carries. Both loops' *behaviour* is asserted
/// where the loops are (`src/conversation/tests/cancel.rs`, `tests/library_api.rs`).
#[test]
fn a_cancel_token_shares_its_flag_with_its_clones() {
    let token = CancelToken::new();
    let clone = token.clone();
    assert!(!clone.is_cancelled(), "a fresh token is not cancelled");
    token.cancel();
    assert!(clone.is_cancelled(), "clones share one flag");
    assert!(format!("{token:?}").contains("cancelled: true"));
}
