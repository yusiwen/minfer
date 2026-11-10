//! The **advanced** tier (ADR-0079): the L1 primitives a host may drive directly.
//!
//! The facade ([`crate::engine`]) is the stable surface; this module is the one
//! the HTTP server and the tooling sit on — `ModelDef`, the graph cache and its
//! params, the batch, the KV format gate, the tokenizer, the sampler pipeline,
//! the grammar and the speculative engine.
//!
//! **Not semver-stable before 1.0.** The items are public because the server is
//! a first-class consumer and must not re-implement the engine to reach them;
//! they may change when the server needs a different shape. A host that only
//! wants to load a model and answer a prompt uses [`crate::engine`], not this.

pub use crate::cancel::CancelToken;
pub use crate::conversation::{
    Conversation, ConversationSpec, Engine as ConversationEngine, TurnOutcome, TurnParams,
};
pub use crate::grammar::Grammar;
pub use crate::graph::batch::Batch;
pub use crate::graph::cache::GraphCache;
pub use crate::graph::kvformat::KvFormat;
pub use crate::graph::offload::OffloadRequest;
pub use crate::graph::params::GraphParams;
pub use crate::graph::registry::{Backend, BackendFilter};
pub use crate::models::{ModelDef, SpecialTokens};
pub use crate::sampler::{MirostatMode, SamplerConfig};
pub use crate::spec::{SpecConfig, SpecEngine, SpecSampler};
pub use crate::tokenizer::Tokenizer;

// ── L0: observation ────────────────────────────────────────────────────
// The visualizer and the metrics endpoint read these; the engine emits into
// them from the scheduler. `live` needs tokio, so it is the `observe` feature.
pub use crate::graph::json;
#[cfg(feature = "observe")]
pub use crate::live;
pub use crate::optiming;
pub use crate::trace;

#[cfg(test)]
mod tests;
