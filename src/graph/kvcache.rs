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
use crate::vec_ops::RopeStyle;

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

    /// Bookkeeping for a physical removal of rows `[start, start + len)`: the
    /// caller has already moved the rows after them down by `len` in every
    /// arena (see the allocator's `kv_rm`), so the store only renumbers.
    ///
    /// The mapping stays the **identity** — a physical removal leaves `cell ==
    /// pos` for every surviving row (rows `[0, start)` do not even move) —
    /// which is why C2 needs no Backend-trait hand-off and no GPU kernel: the
    /// removal is a host-side memmove plus a re-rope, both done through the
    /// existing `copy_kv_to_cpu`/`write_host` pair. Returns the new `n_used`.
    ///
    /// Removing everything written (`start == 0`, `len == n_used`) is allowed and
    /// empties the arena; removing past the end is an error, not a silent
    /// truncation.
    pub fn after_rm(&mut self, start: usize, len: usize) -> Result<usize, String> {
        let mut new_used = 0usize;
        for (layer, l) in self.layers.iter_mut() {
            if start + len > l.n_used {
                return Err(format!(
                    "KV layer {layer}: cannot remove {len} rows at {start} of {} written rows",
                    l.n_used
                ));
            }
            // Rows [start + len, n_used) slide down; the freed tail becomes FREE.
            l.owner.copy_within(start + len..l.n_used, start);
            for cell in (l.n_used - len)..l.n_used {
                l.owner[cell] = FREE;
            }
            l.n_used -= len;
            new_used = l.n_used;
        }
        Ok(new_used)
    }

    /// Sliding-window special case of [`KvCache::after_rm`]: drop the oldest
    /// `drop` rows, so every survivor's position decreases by `drop`.
    pub fn after_shift(&mut self, drop: usize) -> Result<usize, String> {
        self.after_rm(0, drop)
    }

    /// The cell range a sequence owns in `layer`, as `(start, len)`; `None` when
    /// it owns nothing.
    ///
    /// Ownership is contiguous by construction — a sequence's cells are written
    /// in position order and a removal slides the survivors down (C2) — so one
    /// range describes it completely. A gap is a bug rather than a supported
    /// layout, and this returns `Err` instead of letting attention bound itself
    /// to the wrong window; a per-cell mask is what a hole-creating layout would
    /// need (C3/D1).
    pub fn seq_range(&self, layer: usize, seq: SeqId) -> Result<Option<(usize, usize)>, String> {
        let l = self
            .layers
            .get(&layer)
            .ok_or_else(|| format!("no KV arena for layer {layer}"))?;
        let mut start: Option<usize> = None;
        let mut end = 0usize;
        for (cell, &owner) in l.owner.iter().enumerate() {
            if owner != seq {
                continue;
            }
            match start {
                None => start = Some(cell),
                Some(_) if cell != end => {
                    return Err(format!(
                        "KV layer {layer}: sequence {seq} owns cell {cell} after {end} — \
                         non-contiguous ownership, which a range cannot describe"
                    ))
                }
                Some(_) => {}
            }
            end = cell + 1;
        }
        Ok(start.map(|s| (s, end - s)))
    }

    /// Resolve each query token's allowed cell range into the `attn_span` input
    /// layout: `lo` of token `t` at `span[t]`, `hi` at `span[n + t]`.
    ///
    /// `seq_ids[t]` names the sequence query `t` belongs to; `positions[t]` is
    /// its write position, i.e. a cell index. The start comes from ownership and
    /// **not** from the position (that is the point of E1); the position only
    /// truncates the end, because a token may not attend to cells written after
    /// it.
    ///
    /// Every layer owns the same cells — all mutations go through this store —
    /// so the first layer resolves and a debug assertion checks the rest agree.
    pub fn attn_span(&self, seq_ids: &[u32], positions: &[usize]) -> Result<Vec<u32>, String> {
        let n = seq_ids.len();
        if positions.len() != n {
            return Err(format!(
                "attn_span: {} sequence ids but {} positions",
                n,
                positions.len()
            ));
        }
        let layer = *self
            .layers
            .keys()
            .next()
            .ok_or("attn_span: no KV arena allocated")?;
        let mut span = vec![0u32; 2 * n];
        for t in 0..n {
            let seq = seq_ids[t];
            let (start, len) = self.seq_range(layer, seq)?.ok_or_else(|| {
                format!("attn_span: query {t} belongs to sequence {seq}, which owns no KV cells")
            })?;
            let hi = (start + len).min(positions[t] + 1);
            if hi <= start {
                return Err(format!(
                    "attn_span: query {t} (sequence {seq}, position {}) has an empty window \
                     [{start}, {hi})",
                    positions[t]
                ));
            }
            span[t] = start as u32;
            span[n + t] = hi as u32;
        }
        let first = self.layers.values().next().map(|l| l.n_used);
        debug_assert!(
            self.layers.values().all(|l| Some(l.n_used) == first),
            "KV layers disagree about how much is written"
        );
        Ok(span)
    }
}

/// RoPE parameters a KV shift needs to re-rope the stored K rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvRope {
    pub freq_base: f32,
    pub freq_scale: f32,
    /// KV heads in one row (`n_head_kv`).
    pub n_head_kv: usize,
    /// Per-head width.
    pub hd: usize,
    pub style: RopeStyle,
}

/// Re-rope `k` in place so that every row that was written at position `p`
/// becomes consistent with position `p - delta`.
///
/// RoPE applies the pair rotation `theta_i = p * freq_i`, so changing the
/// position by `-delta` is a rotation by `-delta * freq_i` — the same angle for
/// every row, which is why a whole window can be shifted in one pass.
///
/// **Not bitwise with a fresh prefill.** Re-roping an already-roped vector
/// composes two rotations instead of applying one, so the result differs in the
/// last ulp or two; the measured size of that difference is what C2 records as
/// its named tolerance class (the alternative is re-prefilling the window).
///
/// `k` is the whole arena (`n_ctx * n_head_kv * hd` floats, row-major with the
/// heads contiguous inside a row); only the first `rows` rows are touched.
/// `delta` is signed so the rotation can be undone (`-delta` restores the rows
/// to within the same tolerance class), which is what the tests check.
pub fn rope_shift_kv(k: &mut [f32], rows: usize, delta: isize, rope: &KvRope) {
    let hd = rope.hd;
    let half = hd / 2;
    if half == 0 || delta == 0 || rows == 0 {
        return;
    }
    let row_elems = rope.n_head_kv * hd;
    debug_assert!(k.len() >= rows * row_elems, "K arena shorter than `rows`");
    // The shift is a rotation by `-delta * freq_i`, the same for every row.
    let angles: Vec<f32> = (0..half)
        .map(|i| {
            let freq = rope.freq_scale / rope.freq_base.powf((2 * i) as f32 / hd as f32);
            -(delta as f32) * freq
        })
        .collect();
    for r in 0..rows {
        for h in 0..rope.n_head_kv {
            let b = r * row_elems + h * hd;
            for (i, &th) in angles.iter().enumerate() {
                let (sn, cs) = th.sin_cos();
                let (i0, i1) = match rope.style {
                    RopeStyle::NonInterleaved => (b + i, b + i + half),
                    RopeStyle::Interleaved => (b + 2 * i, b + 2 * i + 1),
                };
                let (x0, x1) = (k[i0], k[i1]);
                k[i0] = x0 * cs - x1 * sn;
                k[i1] = x0 * sn + x1 * cs;
            }
        }
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

    #[test]
    fn after_shift_renumbers_and_frees_the_tail() {
        let mut c = cache(4);
        c.own_prefix(SEQ_MAIN, 4);
        assert_eq!(c.after_shift(1).unwrap(), 3, "new n_used");
        for (_, l) in c.iter() {
            assert_eq!(l.n_used, 3);
            assert_eq!(l.owner, vec![SEQ_MAIN, SEQ_MAIN, SEQ_MAIN, FREE]);
        }
        // Removing past the end is an error, not a silent truncation.
        let err = c.after_shift(4).unwrap_err();
        assert!(err.contains("cannot remove 4 rows at 0 of 3"), "got: {err}");
        // A physical shift keeps the identity mapping: no backend hand-off.
        assert!(c.is_identity());
    }

    #[test]
    fn after_rm_removes_a_middle_range_and_frees_the_tail() {
        let mut c = cache(6);
        c.own_prefix(SEQ_MAIN, 6);
        // Drop cells 2..4: [0,1] stay, [4,5] slide to [2,3], the tail frees.
        assert_eq!(c.after_rm(2, 2).unwrap(), 4, "new n_used");
        for (_, l) in c.iter() {
            assert_eq!(l.n_used, 4);
            assert_eq!(
                l.owner,
                vec![SEQ_MAIN, SEQ_MAIN, SEQ_MAIN, SEQ_MAIN, FREE, FREE]
            );
        }
        // Removing every written row empties the arena.
        assert_eq!(c.after_rm(0, 4).unwrap(), 0);
        for (_, l) in c.iter() {
            assert_eq!(l.n_used, 0);
            assert!(l.owner.iter().all(|&o| o == FREE));
        }
        assert!(c.is_identity(), "a physical removal keeps cell == pos");
    }

    /// E1: a single sequence's window is the one positions used to derive —
    /// `[0, pos + 1)` — which is why the kernel change stays bitwise.
    #[test]
    fn a_single_sequence_resolves_to_the_causal_window() {
        let mut c = cache(8);
        c.own_prefix(SEQ_MAIN, 4);
        c.note_written(0, &[0, 1, 2, 3]);
        // Query at position 2 may see cells 0..3; the last query sees 0..4.
        assert_eq!(
            c.attn_span(&[SEQ_MAIN, SEQ_MAIN], &[2, 3]).unwrap(),
            vec![0, 0, 3, 4],
            "lo row then hi row"
        );
    }

    /// E1's acceptance at the store level: two sequences in one arena get two
    /// disjoint windows, so no query can see the other's cells.
    #[test]
    fn two_sequences_resolve_to_disjoint_windows() {
        let mut c = cache(8);
        // Sequence 1 owns cells 0..3, sequence 2 owns cells 3..6. Ownership is
        // written through the same method for every layer, as the allocator's
        // `own_prefix`/`after_rm` do.
        c.own_prefix(SEQ_MAIN, 3);
        for cell in 3..6 {
            for layer in [0usize, 1] {
                c.set_owner(layer, cell, 1).unwrap();
            }
        }
        assert_eq!(c.seq_range(0, SEQ_MAIN).unwrap(), Some((0, 3)));
        assert_eq!(c.seq_range(0, 1).unwrap(), Some((3, 3)));
        assert_eq!(c.seq_range(1, 1).unwrap(), Some((3, 3)), "layers agree");
        // A token of each sequence, each at its own last position.
        let span = c.attn_span(&[SEQ_MAIN, 1], &[2, 5]).unwrap();
        assert_eq!(&span[..2], &[0, 3], "starts are each sequence's own");
        assert_eq!(&span[2..], &[3, 6], "ends are exclusive and causal");
        // The windows do not overlap: sequence 2 never sees cells 0..3.
        assert!(
            span[2] >= span[1],
            "sequence 2's window starts where sequence 1's ends"
        );
    }

    /// A layout the resolver cannot describe must be loud, not a wrong bound.
    #[test]
    fn ownership_gaps_and_empty_windows_are_errors() {
        let mut c = cache(8);
        c.set_owner(0, 0, SEQ_MAIN).unwrap();
        c.set_owner(0, 2, SEQ_MAIN).unwrap(); // cell 1 belongs to nobody
        let err = c.seq_range(0, SEQ_MAIN).unwrap_err();
        assert!(err.contains("non-contiguous"), "got: {err}");
        let err = c.attn_span(&[SEQ_MAIN], &[2]).unwrap_err();
        assert!(err.contains("non-contiguous"), "got: {err}");

        // A query whose sequence owns nothing has no window at all.
        let mut c = cache(8);
        c.set_owner(0, 0, SEQ_MAIN).unwrap();
        let err = c.attn_span(&[1], &[0]).unwrap_err();
        assert!(err.contains("owns no KV cells"), "got: {err}");

        // A query at a position before its sequence's cells is empty too.
        let mut c = cache(8);
        c.note_written(0, &[3]);
        let err = c.attn_span(&[SEQ_MAIN], &[2]).unwrap_err();
        assert!(err.contains("empty window"), "got: {err}");
    }

    /// A shift must reproduce "the same tokens, roped at their new positions".
    /// It cannot be bitwise (two composed rotations vs one), so this pins both
    /// the equivalence and the size of the difference — C2's tolerance class.
    #[test]
    fn rope_shift_matches_roping_at_the_new_position() {
        let rope = KvRope {
            freq_base: 10_000.0,
            freq_scale: 1.0,
            n_head_kv: 2,
            hd: 4,
            style: RopeStyle::NonInterleaved,
        };
        // Rope a row at `pos` exactly the way the kernels do.
        fn rope_at(x: &mut [f32], pos: usize, rope: &KvRope) {
            let half = rope.hd / 2;
            for h in 0..rope.n_head_kv {
                let b = h * rope.hd;
                for i in 0..half {
                    let f = rope.freq_scale / rope.freq_base.powf((2 * i) as f32 / rope.hd as f32);
                    let (sn, cs) = (pos as f32 * f).sin_cos();
                    let (i0, i1) = (b + i, b + i + half);
                    let (x0, x1) = (x[i0], x[i1]);
                    x[i0] = x0 * cs - x1 * sn;
                    x[i1] = x0 * sn + x1 * cs;
                }
            }
        }

        let base: Vec<f32> = (0..(rope.n_head_kv * rope.hd))
            .map(|i| (i as f32 + 1.0) * 0.25)
            .collect();
        let pos = 7;
        let delta = 3;

        let mut shifted = base.clone();
        rope_at(&mut shifted, pos, &rope);
        rope_shift_kv(&mut shifted, 1, delta, &rope);

        let mut reference = base.clone();
        rope_at(&mut reference, pos - delta as usize, &rope);

        let worst = shifted
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = reference.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
        eprintln!(
            "[c2] rope-shift tolerance class: max |Δ| = {worst} (relative {})",
            worst / scale
        );
        assert!(
            worst < 1e-5,
            "shift must land on the new-position rope, got |Δ| = {worst}"
        );
        // delta = 0 must be a no-op.
        let mut untouched = base.clone();
        rope_at(&mut untouched, pos, &rope);
        let mut same = untouched.clone();
        rope_shift_kv(&mut same, 1, 0, &rope);
        assert_eq!(same, untouched);
    }
}
