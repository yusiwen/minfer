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
///
/// `gpu` records whether the GPU backend participates — the backend assignment
/// is part of the built graph, so a change (e.g. MPS init between runs) must
/// force a rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CParams {
    pub n_ctx: usize,
    pub n_batch: usize,
    pub flash_attn: bool,
    pub gpu: bool,
    /// G4 decode QKV fusion enabled (part of the topology: toggling
    /// `MINFER_NO_FUSE_QKV` must force a rebuild).
    pub fuse_qkv: bool,
}

impl Default for CParams {
    fn default() -> Self {
        Self { n_ctx: 4096, n_batch: 128, flash_attn: false, gpu: false, fuse_qkv: false }
    }
}

/// Reuse-relevant graph parameters (see module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphParams {
    pub n_tokens: usize,
    pub n_seqs: usize,
    /// Number of output (tail) rows: the last layer's FFN + lm_head run on the
    /// last `n_out` rows only (llama `inp_out_ids`). Part of the topology —
    /// a change forces a rebuild.
    pub n_out: usize,
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
