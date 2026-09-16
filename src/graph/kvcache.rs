//! Sequence-addressable KV cell store (Phase C / ticket C1).
//!
//! The graph path used to model the KV cache as "two persistent regions per
//! layer, indexed by position" — which works only while a position uniquely
//! names a row and nothing is ever dropped. Phase C needs the general form:
//! cells that a sequence can own, release and (C2) shift.
//!
//! C1 is the behaviour-preserving half. It introduces the structure —
//! `owner[cell]` plus the host-side `position -> cell` resolver — while the
//! resolver is still the identity, so every existing path produces the same
//! bytes. `is_identity` is the gate C2 flips: while it is true the raw
//! `positions` input is the cell index and backends need no change; once it is
//! false the resolved array must reach the kernel, and a backend that has not
//! been ported must return `Err` rather than index the wrong row.
//!
//! Design record: `docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 (C1).

use std::collections::BTreeMap;

use super::BufRef;

/// Sequence identifier. C1 has exactly one implicit sequence; C2/E add more.
pub type SeqId = u32;

/// The one sequence the graph path builds today.
pub const SEQ_MAIN: SeqId = 0;

/// A cell no sequence owns (dropped, or never written).
pub const FREE: SeqId = u32::MAX;

/// One layer's KV arena: the two persistent regions plus per-cell ownership.
#[derive(Debug, Clone)]
pub struct KvLayer {
    pub k: BufRef,
    pub v: BufRef,
    /// Element count of one region (`n_kv_embd * n_ctx`) — the value
    /// `ensure_kv` compares on a rebuild so a changed `n_ctx` fails loudly.
    pub elems: usize,
    /// Rows the arena can hold (`n_ctx`).
    pub n_ctx: usize,
    /// `owner[cell]` — one entry per cell, `FREE` when unowned.
    #[allow(dead_code)] // read by C2's resolver and the tests; C1 only writes it
    pub owner: Vec<SeqId>,
    /// Highest written row + 1 (the "used" prefix). Positions at or beyond it
    /// have never been written; attention above it would read zeroes.
    #[allow(dead_code)] // C2's eviction policy reads it
    pub n_used: usize,
}

/// The KV cell store for one allocator (one graph cache, one session).
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    layers: BTreeMap<usize, KvLayer>,
    /// False once C2 has introduced a hole or a window, i.e. once `cell` is no
    /// longer equal to `pos` for every row in use.
    identity: bool,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            layers: BTreeMap::new(),
            identity: true,
        }
    }

    pub fn get(&self, layer: usize) -> Option<&KvLayer> {
        self.layers.get(&layer)
    }

    pub fn insert(&mut self, layer: usize, k: BufRef, v: BufRef, elems: usize, n_ctx: usize) {
        self.layers.insert(
            layer,
            KvLayer {
                k,
                v,
                elems,
                n_ctx,
                owner: vec![FREE; n_ctx],
                n_used: 0,
            },
        );
    }

    pub fn contains(&self, layer: usize) -> bool {
        self.layers.contains_key(&layer)
    }

    /// Layers in layer order (diagnostics / tests).
    #[allow(dead_code)] // C2 surface
    pub fn iter(&self) -> impl Iterator<Item = (usize, &KvLayer)> {
        self.layers.iter().map(|(&l, r)| (l, r))
    }

    /// True while `cell == pos` for every live row, i.e. while a backend may
    /// keep indexing the arena with the raw positions input.
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Host-side `position -> cell` resolution for `layer`.
    ///
    /// C1: the identity, with a bounds check — a position at or beyond `n_ctx`
    /// would index past the arena. C2 replaces exactly this function's body
    /// (consulting `owner` and the layer's window) without touching its
    /// callers.
    pub fn cells_for(&self, layer: usize, positions: &[usize]) -> Result<Vec<u32>, String> {
        let l = self
            .layers
            .get(&layer)
            .ok_or_else(|| format!("no KV arena for layer {layer}"))?;
        let mut cells = Vec::with_capacity(positions.len());
        for &p in positions {
            // In C1 the cell is the position, so the check is also the arena
            // bound; C2 keeps this the arena bound and checks ownership instead.
            if p >= l.n_ctx {
                return Err(format!(
                    "KV layer {layer}: position {p} >= n_ctx {} (arena overflow)",
                    l.n_ctx
                ));
            }
            cells.push(p as u32);
        }
        Ok(cells)
    }

    /// Mark `cell` owned by `seq` in every layer (or release it with `FREE`).
    /// C1 uses it from `own_range`; C2's removal/reuse is where it earns its
    /// keep.
    #[allow(dead_code)] // C2 surface
    pub fn set_owner(&mut self, layer: usize, cell: usize, seq: SeqId) -> Result<(), String> {
        let l = self
            .layers
            .get_mut(&layer)
            .ok_or_else(|| format!("no KV arena for layer {layer}"))?;
        if cell >= l.owner.len() {
            return Err(format!(
                "KV layer {layer}: cell {cell} out of range ({} cells)",
                l.owner.len()
            ));
        }
        l.owner[cell] = seq;
        Ok(())
    }

    /// Take ownership of `0..n_used` for `seq` in every layer, marking the rows
    /// written so far. C1 calls this after a prefill flushes the rows it wrote.
    #[allow(dead_code)] // C2 surface
    pub fn own_prefix(&mut self, seq: SeqId, n_used: usize) {
        for l in self.layers.values_mut() {
            let upto = n_used.min(l.owner.len());
            for cell in 0..upto {
                l.owner[cell] = seq;
            }
            l.n_used = l.n_used.max(upto);
        }
    }

    /// Record that `rows` were written at the given resolved cells.
    #[allow(dead_code)] // C2 surface
    pub fn note_written(&mut self, layer: usize, cells: &[u32]) {
        if let Some(l) = self.layers.get_mut(&layer) {
            for &c in cells {
                let c = c as usize;
                if c < l.owner.len() {
                    l.owner[c] = SEQ_MAIN;
                    l.n_used = l.n_used.max(c + 1);
                }
            }
        }
    }

    /// Drop the identity fast path. C2 calls this when it introduces a hole or
    /// a window; after that, backends that only understand raw positions must
    /// refuse the node (standing rule 2) instead of indexing the wrong row.
    #[allow(dead_code)] // C2 surface
    pub fn clear_identity(&mut self) {
        self.identity = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Backend;

    fn buf(id: usize) -> BufRef {
        BufRef {
            backend: Backend::CPU,
            id,
        }
    }

    fn cache(n_ctx: usize) -> KvCache {
        let mut c = KvCache::new();
        c.insert(0, buf(1), buf(2), n_ctx * 4, n_ctx);
        c.insert(1, buf(3), buf(4), n_ctx * 4, n_ctx);
        c
    }

    #[test]
    fn starts_identity_and_resolves_positions_verbatim() {
        let c = cache(8);
        assert!(c.is_identity(), "C1 is the identity by construction");
        assert_eq!(c.cells_for(0, &[0, 1, 7]).unwrap(), vec![0, 1, 7]);
        // Every layer resolves to the same cells (one implicit sequence).
        assert_eq!(c.cells_for(1, &[3]).unwrap(), vec![3]);
    }

    #[test]
    fn a_position_past_the_arena_is_an_error() {
        let c = cache(8);
        let err = c.cells_for(0, &[8]).unwrap_err();
        assert!(err.contains(">= n_ctx 8"), "got: {err}");
        let err = c.cells_for(9, &[0]).unwrap_err();
        assert!(err.contains("no KV arena"), "got: {err}");
    }

    #[test]
    fn ownership_follows_the_written_cells() {
        let c0 = cache(4);
        // Nothing written yet: every cell is free.
        assert!(c0.get(0).unwrap().owner.iter().all(|&o| o == FREE));
        assert_eq!(c0.get(0).unwrap().n_used, 0);

        let mut c = cache(4);
        let cells = c.cells_for(0, &[0, 1]).unwrap();
        c.note_written(0, &cells);
        assert_eq!(
            c.get(0).unwrap().owner,
            vec![SEQ_MAIN, SEQ_MAIN, FREE, FREE]
        );
        assert_eq!(c.get(0).unwrap().n_used, 2, "used prefix tracks the writes");
        // note_written is per layer: layer 1 is untouched.
        assert!(c.get(1).unwrap().owner.iter().all(|&o| o == FREE));
    }

    #[test]
    fn own_prefix_and_release_round_trip() {
        let mut c = cache(4);
        c.own_prefix(SEQ_MAIN, 3);
        assert_eq!(
            c.get(0).unwrap().owner,
            vec![SEQ_MAIN; 3]
                .into_iter()
                .chain([FREE])
                .collect::<Vec<_>>()
        );
        c.set_owner(0, 1, FREE).unwrap();
        assert_eq!(c.get(0).unwrap().owner[1], FREE);
        assert!(c.set_owner(0, 9, SEQ_MAIN).is_err(), "cell out of range");
    }

    #[test]
    fn clearing_identity_is_the_backend_gate() {
        let mut c = cache(4);
        assert!(c.is_identity());
        c.clear_identity();
        assert!(
            !c.is_identity(),
            "once false, positions are no longer cells"
        );
    }
}
