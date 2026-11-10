//! The cancellation handle both generation paths share (#796, #800).
//!
//! The single-shot loop (`crate::engine::generate`) and the conversation turn
//! loop (`crate::conversation`) stop at the same boundary — between decoded
//! tokens, and between speculative rounds — and both report the stop through a
//! named `cancelled` field rather than an error. One type, one home: the token
//! is an L1 primitive (the conversation layer needs it too), so it lives here
//! rather than in the facade module that first used it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cancellation handle a host holds across threads (#796).
///
/// The loops check it once per decoded token — and once per speculative round,
/// so a draft round cannot outrun it — and stop at that token boundary. It
/// cannot interrupt a forward pass, which is a single call into the model; a
/// cancel that arrives mid-pass is honoured at the next token.
///
/// Cheap to clone: every clone shares one flag, so a signal handler or a UI
/// thread can hold its own. There is no reset: a cancelled token stays
/// cancelled, so a host that wants to run again makes a new one.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the running generation to stop. Idempotent.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Whether [`cancel`](Self::cancel) has been called, by any clone.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
mod tests;
