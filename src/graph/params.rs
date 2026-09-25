//! Graph construction/execution parameters (Phase 4).
//!
//! These are the ONLY inputs to graph reuse: the graph topology is a
//! deterministic function of `GraphParams` (llama.cpp `allow_reuse` invariant),
//! so `GraphCache::try_reuse` compares params only — never the node sequence.
//! `n_past` (KV position) is deliberately absent: it is execution data.
//!
//! So is the **number of sequences** a batch covers (E2): `CParams.explicit_span`
//! carries the only topology decision a multi-sequence batch can force, the
//! per-query sequence ids and allowed cell spans are inputs. Carrying the count
//! as well bought nothing and cost a rebuild whenever the batching changed shape
//! with the topology unchanged (see the `sequence_count_is_data_not_topology`
//! test in `models/qwen2/graph.rs`).

use std::sync::atomic::{AtomicU64, Ordering};

use super::kvformat::KvFormat;

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
    pub flash_attn: bool,
    pub gpu: bool,
    /// G4 decode QKV fusion enabled (part of the topology: toggling
    /// `MINFER_NO_FUSE_QKV` must force a rebuild).
    pub fuse_qkv: bool,
    /// G5 decode FFN gate+up fusion enabled (part of the topology: toggling
    /// `MINFER_NO_FUSE_FFN` must force a rebuild). Decoupled from `fuse_qkv`
    /// so A/B-ing one fusion does not flip the other.
    pub fuse_ffn: bool,
    /// E2: attention must read the explicit span because `positions` alone
    /// cannot bound it — more than one sequence in the batch, or a window that
    /// does not start at cell 0. It selects a *kernel instantiation*, so it is
    /// topology (like the fusion flags), bounded to two graphs per shape; the
    /// model derives it from the KV reservations, never from `n_past`, and the
    /// classic single-sequence path (one sequence starting at 0) keeps the
    /// causal instantiation.
    pub explicit_span: bool,
    /// C8b S2: this graph's attention window is a **list of cell runs** (a shared
    /// prefix plus a private run, the `kv_map` input) rather than one `[lo, hi)`
    /// range. It selects a kernel instantiation whose window input has a different
    /// layout, so it is topology like `explicit_span`; only a backend that can
    /// gather a map may take the node (`Backend::supports_kv_map` — CPU until C8b
    /// S4).
    pub kv_map: bool,
    /// E5: how many leading transformer blocks run on the device (`usize::MAX` =
    /// every block, the pre-E5 behaviour). The *assignment* is part of the built
    /// graph, so a different offload plan must rebuild: this is topology like
    /// `gpu`, and `GraphCache::try_reuse` compares it with the rest of `CParams`.
    pub gpu_layers: usize,
    /// C4 per-engine (issue #99): the storage format of this graph's persistent KV
    /// regions. It fixes each KV node's width (`KvcacheMeta::row_elems`), so a graph
    /// built for one format must not be reused for another — it is topology like
    /// the fusion flags. It is **not** read from a process global: the loaded model
    /// resolves `MINFER_CACHE_TYPE` once and stamps it here, so two engines with
    /// different formats in one process cannot size each other's regions.
    pub kv_format: KvFormat,
}

impl Default for CParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            flash_attn: false,
            gpu: false,
            fuse_qkv: false,
            fuse_ffn: false,
            explicit_span: false,
            kv_map: false,
            gpu_layers: usize::MAX,
            kv_format: KvFormat::F32,
        }
    }
}

/// Reuse-relevant graph parameters (see module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphParams {
    pub n_tokens: usize,
    /// Number of output (tail) rows: the last layer's FFN + lm_head run on the
    /// last `n_out` rows only (llama `inp_out_ids`). Part of the topology —
    /// a change forces a rebuild.
    pub n_out: usize,
    pub gtype: GraphType,
    pub cparams: CParams,
    /// Bumped by the model whenever weights change (LoRA switch, reload).
    pub weights_version: u64,
}

/// Global weight-version counter (Phase 6 wires the model to bump it; the
/// counter exists so a future LoRA/reload path can break graph reuse).
#[allow(dead_code)]
pub fn next_weights_version() -> u64 {
    static V: AtomicU64 = AtomicU64::new(1);
    V.fetch_add(1, Ordering::Relaxed)
}
