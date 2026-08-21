//! Graph construction/execution parameters (Phase 4).
//!
//! These are the ONLY inputs to graph reuse: the graph topology is a
//! deterministic function of `GraphParams` (llama.cpp `allow_reuse` invariant),
//! so `GraphCache::try_reuse` compares params only — never the node sequence.
//! `n_past` (KV position) is deliberately absent: it is execution data.

use std::sync::atomic::{AtomicU64, Ordering};

/// Decode (n_tokens=1, incremental) vs prefill (n_tokens>1) graph type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    Decode,
    Prefill,
}

/// Runtime parameters that affect graph construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CParams {
    pub n_ctx: usize,
    pub n_batch: usize,
    pub flash_attn: bool,
}

impl Default for CParams {
    fn default() -> Self {
        Self { n_ctx: 4096, n_batch: 128, flash_attn: false }
    }
}

/// Reuse-relevant graph parameters (see module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphParams {
    pub n_tokens: usize,
    pub n_seqs: usize,
    pub gtype: GraphType,
    pub cparams: CParams,
    /// Bumped by the model whenever weights change (LoRA switch, reload).
    pub weights_version: u64,
}

/// Global weight-version counter (Phase 6 wires the model to bump it).
pub fn next_weights_version() -> u64 {
    static V: AtomicU64 = AtomicU64::new(1);
    V.fetch_add(1, Ordering::Relaxed)
}
