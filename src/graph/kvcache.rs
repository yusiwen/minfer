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

/// C8b S2: how many `(cell, len)` runs one query's `kv_map` entry carries.
///
/// Two suffice for what the store builds today — a shared prefix plus a private run
/// — and the padding is what keeps the input a fixed shape (an input's size is
/// topology, so it cannot depend on the batch). A window that needs more is refused
/// loudly by [`KvCache::attn_map`] rather than truncated.
pub const KV_MAP_MAX_SPANS: usize = 4;

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

/// A prefix a sequence reads from **another** sequence's rows (C8b S2).
///
/// Sequence `s` with `shared.rows = r` reads positions `[0, r)` at the donor's
/// cells `[cell, cell + r)` and writes its own positions from `r` on into its
/// private run. The donor is not told: its rows stay its own, and occupancy keeps
/// them taken for as long as any sharer's span list names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SharedPrefix {
    /// First cell of the prefix (the donor's first written cell).
    pub cell: usize,
    /// Rows the prefix covers; 0 means the sequence shares nothing.
    pub rows: usize,
}

/// A sequence's reserved cell run: cells `[start, start + cap)` hold positions
/// `[shared.rows, shared.rows + cap)` (Phase E / E2, C8b S2).
///
/// A reservation is **not** ownership: it says which cells the sequence *may*
/// write, so a query can never attend to rows its sequence has not written yet
/// (`attn_span` still caps the window at the query's own position). Ownership
/// (C1) marks the rows actually written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqSlot {
    pub start: usize,
    pub cap: usize,
    /// C8b S2: rows read in place from another sequence (positions `[0, rows)`).
    pub shared: SharedPrefix,
    /// Positions written so far — the shared prefix plus the rows this sequence
    /// wrote itself. Explicit rather than derived from `owner`, because a shared
    /// row carries the **donor's** stamp.
    pub written: usize,
}

impl SeqSlot {
    /// A private run: cells `[start, start + cap)`, positions from 0, nothing written.
    pub fn new(start: usize, cap: usize) -> Self {
        Self {
            start,
            cap,
            shared: SharedPrefix::default(),
            written: 0,
        }
    }

    /// Rows this sequence wrote into its **own** run (`0..=cap`) — the rows a
    /// relocation has to copy, since the shared prefix lives elsewhere.
    pub fn private_written(&self) -> usize {
        self.written.saturating_sub(self.shared.rows)
    }
}

/// The KV cell store for one allocator (one graph cache, one session).
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    layers: BTreeMap<usize, KvLayer>,
    /// False once C2 has introduced a hole or a window, i.e. once `cell` is no
    /// longer equal to `pos` for every row in use.
    identity: bool,
    /// Reserved runs, one per live sequence. Several sequences share one arena
    /// through these (E2); one implicit sequence takes the whole of it.
    seqs: BTreeMap<SeqId, SeqSlot>,
    /// C8b S1/S2: each sequence's address space as spans
    /// `(position base, first cell, length)`.
    ///
    /// Resolution goes through this list, not through `slot.start` directly, because a
    /// sequence's rows can be *several* spans (a shared prefix plus a private tail).
    /// A sequence that shares nothing has exactly one span covering its whole
    /// reservation, so it must reproduce `start + position` exactly — and a missed
    /// maintenance point is loud rather than silent: `cell_of` returns `None` and the
    /// resolver refuses the query. Occupancy (`reserve_seq`, `free_runs`) is derived
    /// from this list too, which is what keeps a released donor's shared rows taken.
    spans: BTreeMap<SeqId, Vec<(usize, usize, usize)>>,
    /// Arena capacity in rows (`n_ctx`), from the first `insert`.
    n_ctx: usize,
    /// C3: how many compactions ran, and how many cell rows they copied
    /// (summed over layers) — the counters the ticket's acceptance records.
    defrags: u64,
    cells_moved: u64,
    /// C8b S3: how many copy-on-write events ran, and how many of a sequence's own
    /// rows they moved (summed over layers). Like `defrags`, this exists to make
    /// the mechanism observable in a gate that has to *see* it happen.
    cows: u64,
    cow_cells: u64,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            layers: BTreeMap::new(),
            identity: true,
            seqs: BTreeMap::new(),
            spans: BTreeMap::new(),
            n_ctx: 0,
            defrags: 0,
            cells_moved: 0,
            cows: 0,
            cow_cells: 0,
        }
    }

    /// Arena capacity in rows (`n_ctx`), 0 before the first `insert`.
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Declare the arena's row capacity up front (E2). The first `alloc_graph`
    /// sets the same number from `CParams.n_ctx`; declaring it earlier lets a
    /// caller reserve sequences before the first forward, which batching needs.
    pub fn set_n_ctx(&mut self, n_ctx: usize) {
        self.n_ctx = n_ctx;
    }

    pub fn get(&self, layer: usize) -> Option<&KvLayer> {
        self.layers.get(&layer)
    }

    pub fn insert(&mut self, layer: usize, k: BufRef, v: BufRef, elems: usize, n_ctx: usize) {
        self.n_ctx = self.n_ctx.max(n_ctx);
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

    /// Reserve a contiguous run of `cap` cells for `seq` (E2).
    ///
    /// First-fit over cells no live sequence covers, so several sequences share
    /// one arena; an existing reservation is returned unchanged when `cap` fits
    /// in it, and `Err` (never a move) when the arena cannot hold it — moving a
    /// sequence is C3's cell copy, which needs Phase D's views.
    pub fn reserve_seq(&mut self, seq: SeqId, cap: usize) -> Result<SeqSlot, String> {
        if cap == 0 {
            return Err(format!("reserve_seq: sequence {seq} asked for 0 cells"));
        }
        if let Some(slot) = self.seqs.get(&seq) {
            if slot.cap >= cap {
                return Ok(*slot);
            }
            return Err(format!(
                "reserve_seq: sequence {seq} already holds {} cells, {cap} requested (no growth)",
                slot.cap
            ));
        }
        if self.n_ctx == 0 {
            return Err("reserve_seq: no KV arena allocated".to_string());
        }
        // Cells a live sequence covers, whether or not they are written yet.
        let taken = self.occupied();
        let mut run = 0usize;
        for start in 0..self.n_ctx {
            if taken[start] {
                run = 0;
                continue;
            }
            run += 1;
            if run == cap {
                let slot = SeqSlot::new(start + 1 - cap, cap);
                self.seqs.insert(seq, slot);
                self.refresh_spans(seq);
                return Ok(slot);
            }
        }
        Err(format!(
            "reserve_seq: no free run of {cap} cells for sequence {seq} ({} of {} cells reserved)",
            self.seqs.values().map(|s| s.cap).sum::<usize>(),
            self.n_ctx
        ))
    }

    /// Release everything `seq` reserved and owned; returns the freed capacity.
    pub fn release_seq(&mut self, seq: SeqId) -> usize {
        let Some(slot) = self.seqs.remove(&seq) else {
            return 0;
        };
        self.refresh_spans(seq);
        for l in self.layers.values_mut() {
            for cell in slot.start..(slot.start + slot.cap).min(l.owner.len()) {
                if l.owner[cell] == seq {
                    l.owner[cell] = FREE;
                }
            }
        }
        slot.cap
    }

    /// Set `seq`'s reservation to `cap` cells and return the compaction plan that
    /// reconciles the arena (C7b).
    ///
    /// Growing is refused when every reservation cannot fit at once — capacity has
    /// to come from somewhere, which is the caller's decision (the server releases
    /// idle runs first) — and shrinking is refused below the rows already written,
    /// so space can never be traded for data. The run table is updated **before**
    /// planning, which is what lets the planner see the new size; the rows travel
    /// with the plan (the allocator copies them before [`Self::apply_moves`]
    /// renumbers), in whichever direction the plan needs.
    pub fn set_cap(&mut self, seq: SeqId, cap: usize) -> Result<Vec<KvMove>, String> {
        let Some(slot) = self.seqs.get(&seq).copied() else {
            return Err(format!("set_cap: sequence {seq} holds no run"));
        };
        if cap == slot.cap {
            return Ok(Vec::new());
        }
        if cap == 0 {
            self.release_seq(seq);
            return Ok(self.compaction_plan(None));
        }
        if cap < slot.cap {
            // Only the rows in its *own* run bound the shrink: a shared prefix lives
            // in cells this reservation never covered (C8b S2).
            let written = slot.private_written();
            if cap < written {
                return Err(format!(
                    "set_cap: sequence {seq} has written {written} rows and cannot shrink to {cap}"
                ));
            }
        } else {
            let others: usize = self
                .seqs
                .iter()
                .filter(|(&s, _)| s != seq)
                .map(|(_, s)| s.cap)
                .sum();
            if others + cap > self.n_ctx {
                return Err(format!(
                    "set_cap: sequence {seq} asked for {cap} cells, {others} are reserved \
                     elsewhere and the arena holds {}",
                    self.n_ctx
                ));
            }
        }
        self.seqs.get_mut(&seq).expect("checked above").cap = cap;
        self.refresh_spans(seq);
        Ok(self.compaction_plan(None))
    }

    /// Refresh `seq`'s span list from its slot. The slot is authoritative and the
    /// span list derived; every path that changes a run's `start`/`cap`/`shared`
    /// (or drops the run) calls this, so the two cannot drift silently.
    ///
    /// C8b S2: a sharing sequence gets two entries — the shared prefix at positions
    /// `[0, rows)` (the donor's cells) and its private run covering positions
    /// `[rows, rows + cap)`. A sequence that shares nothing gets one entry, which is
    /// the shape S1 kept bitwise.
    fn refresh_spans(&mut self, seq: SeqId) {
        let entry = match self.seqs.get(&seq) {
            Some(slot) => {
                let mut v = Vec::with_capacity(2);
                if slot.shared.rows > 0 {
                    v.push((0, slot.shared.cell, slot.shared.rows));
                }
                if slot.cap > 0 {
                    v.push((slot.shared.rows, slot.start, slot.cap));
                }
                Some(v)
            }
            None => None,
        };
        match entry {
            Some(v) if !v.is_empty() => {
                self.spans.insert(seq, v);
            }
            _ => {
                self.spans.remove(&seq);
            }
        }
    }

    /// The cells covered by **any** live sequence's address space — a reservation
    /// or a shared prefix (C8b S2).
    ///
    /// Derived from the span lists and not from `seqs`, because a shared prefix lives
    /// inside the donor's run: releasing the donor removes its own spans, and the
    /// cells have to stay taken for the sharers that still read them. With one span
    /// per sequence this is exactly the set `reserve_seq` used before.
    fn occupied(&self) -> Vec<bool> {
        let mut taken = vec![false; self.n_ctx];
        for spans in self.spans.values() {
            for &(_, cell, len) in spans {
                for c in cell..(cell + len).min(self.n_ctx) {
                    taken[c] = true;
                }
            }
        }
        taken
    }

    /// The position `cell` holds for `seq`, inverting the span list (`None` when no
    /// span of that sequence covers the cell).
    fn pos_of_cell(&self, seq: SeqId, cell: usize) -> Option<usize> {
        self.spans
            .get(&seq)?
            .iter()
            .find_map(|&(base, c, len)| (cell >= c && cell < c + len).then_some(base + (cell - c)))
    }

    /// C8b S1: the cell `pos` resolves to for `seq`, through the span list. `None` means
    /// the position is outside every span the sequence has — for a correctly maintained
    /// list that cannot happen for a position inside the run, which is why the resolver
    /// treats it as an error rather than falling back.
    pub fn cell_of(&self, seq: SeqId, pos: usize) -> Option<usize> {
        self.spans.get(&seq)?.iter().find_map(|&(base, cell, len)| {
            (pos >= base && pos < base + len).then_some(cell + (pos - base))
        })
    }

    /// The span list of `seq` (diagnostics and tests).
    pub fn spans_of(&self, seq: SeqId) -> &[(usize, usize, usize)] {
        self.spans.get(&seq).map_or(&[], |v| v.as_slice())
    }

    /// The run `seq` reserved, or `None` when it holds none.
    pub fn seq_slot(&self, seq: SeqId) -> Option<SeqSlot> {
        self.seqs.get(&seq).copied()
    }

    /// Take ownership of cells `[from, to)` for `seq` in every layer — the rows
    /// this forward wrote. Replaces C1's `own_prefix`, which hard-coded sequence
    /// 0 and would clobber a second sequence's cells.
    ///
    /// C8b S2: a written cell is also a written *position*, so this raises
    /// `written`. The cells have to lie inside the sequence's span list (the
    /// reservation is what bounds a write); a cell outside it is a caller bug,
    /// which the debug assertion reports rather than folding into a wrong count.
    pub fn own_range(&mut self, seq: SeqId, from: usize, to: usize) {
        for l in self.layers.values_mut() {
            let upto = to.min(l.owner.len());
            for cell in from.min(upto)..upto {
                l.owner[cell] = seq;
            }
            l.n_used = l.n_used.max(upto);
        }
        let mut written = self.seqs.get(&seq).map_or(0, |s| s.written);
        for cell in from..to {
            match self.pos_of_cell(seq, cell) {
                Some(pos) => written = written.max(pos + 1),
                None => debug_assert!(
                    false,
                    "own_range: cell {cell} is outside sequence {seq}'s spans {:?}",
                    self.spans_of(seq)
                ),
            }
        }
        if let Some(slot) = self.seqs.get_mut(&seq) {
            slot.written = written;
        }
    }

    /// Mark the rows this forward wrote, given the **positions** it wrote them at.
    ///
    /// C8b S2: a sharing sequence's positions do not map to `start + pos`, so the
    /// caller cannot hand over a cell range derived from a position count the way
    /// `own_range(seq, start, start + rows)` assumed.
    pub fn own_positions(&mut self, seq: SeqId, positions: &[usize]) -> Result<(), String> {
        for &pos in positions {
            let cell = self.cell_of(seq, pos).ok_or_else(|| {
                format!(
                    "own_positions: sequence {seq} wrote position {pos}, which its span list \
                     {:?} does not cover",
                    self.spans_of(seq)
                )
            })?;
            self.own_range(seq, cell, cell + 1);
        }
        Ok(())
    }

    /// The single-sequence case: reserve the whole arena for `seq` if it has no
    /// reservation yet, then take ownership of `0..n_used`.
    pub fn own_prefix(&mut self, seq: SeqId, n_used: usize) {
        if !self.seqs.contains_key(&seq) {
            let cap = self.n_ctx;
            if let Err(e) = self.reserve_seq(seq, cap) {
                debug_assert!(false, "own_prefix: {e}");
                return;
            }
        }
        self.own_range(seq, 0, n_used);
    }

    /// Record that `rows` were written at the given resolved cells by `seq`.
    #[allow(dead_code)] // E2 surface (per-token writes)
    pub fn note_written(&mut self, seq: SeqId, layer: usize, cells: &[u32]) {
        if let Some(l) = self.layers.get_mut(&layer) {
            for &c in cells {
                let c = c as usize;
                if c < l.owner.len() {
                    l.owner[c] = seq;
                    l.n_used = l.n_used.max(c + 1);
                }
            }
        }
        // C8b S2: a recorded row is a written position, whatever the cell holds.
        let mut written = self.seqs.get(&seq).map_or(0, |s| s.written);
        for &c in cells {
            if let Some(pos) = self.pos_of_cell(seq, c as usize) {
                written = written.max(pos + 1);
            }
        }
        if let Some(slot) = self.seqs.get_mut(&seq) {
            slot.written = written;
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
        // A physical removal moves cells, which would invalidate the other
        // sequences' reservations. C2 is the single-sequence sliding window:
        // refuse the rest loudly rather than shift someone else's rows.
        if self.seqs.keys().any(|&s| s != SEQ_MAIN) {
            return Err(
                "after_rm: a context shift is single-sequence; release the other sequences first"
                    .to_string(),
            );
        }
        // C8b S2: a shared prefix is another sequence's memory, so a shift cannot
        // move it (and a sharer's positions would no longer line up with its
        // prefix). Refuse rather than rewrite half a window; sharing is a server
        // path, a shift is the CLI conversation's.
        if let Some(slot) = self.seqs.get(&SEQ_MAIN) {
            if slot.shared.rows > 0 {
                return Err(format!(
                    "after_rm: sequence {SEQ_MAIN} shares a {}-row prefix, which a physical \
                     removal would have to move",
                    slot.shared.rows
                ));
            }
        }
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
        if !self.layers.contains_key(&layer) {
            return Err(format!("no KV arena for layer {layer}"));
        }
        Ok(self.seqs.get(&seq).map(|s| (s.start, s.cap)))
    }

    /// Resolve each query token's allowed cell range into the `attn_span` input
    /// layout: `lo` of token `t` at `span[t]`, `hi` at `span[n + t]`.
    ///
    /// `seq_ids[t]` names the sequence query `t` belongs to; `positions[t]` is its
    /// **sequence-relative** index (C6), resolved to a cell through the sequence's
    /// span list (C8b S1), not by `position` arithmetic. The window start comes
    /// from ownership and **not** from the position (that is the point of E1); the
    /// position only truncates the end, because a token may not attend to cells
    /// written after it.
    ///
    /// The window is one contiguous `[lo, hi)` range, which is what the input
    /// layout can carry. S1b therefore resolves only the single-span case: a
    /// sequence laid out in several spans is refused loudly here rather than
    /// collapsed to the span that happens to hold the query, because a set-valued
    /// window needs the C8b S2 `kv_map` input.
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
        if self.layers.is_empty() {
            return Err("attn_span: no KV arena allocated".to_string());
        }
        let mut span = vec![0u32; 2 * n];
        for t in 0..n {
            let seq = seq_ids[t];
            let (first_cell, len) = match self.spans_of(seq) {
                [] => {
                    return Err(format!(
                        "attn_span: query {t} belongs to sequence {seq}, which holds no cells"
                    ))
                }
                [one] => (one.1, one.2),
                many => {
                    return Err(format!(
                        "attn_span: query {t} belongs to sequence {seq}, which is laid out in \
                         {} spans; a window that is not one contiguous range needs the C8b S2 \
                         `kv_map` input",
                        many.len()
                    ))
                }
            };
            // C6: `positions[t]` is the token's index *within its sequence*, and the
            // cell that row lives in is whatever the span list says it is. While a
            // run starts at cell 0 the two coincide, which is why the single-sequence
            // path is unchanged.
            let rel = positions[t];
            let cell = self.cell_of(seq, rel).ok_or_else(|| {
                format!(
                    "attn_span: query {t} (sequence {seq}, position {rel}) is outside its \
                     reserved run [{first_cell}, {})",
                    first_cell + len
                )
            })?;
            // The query's own row must have been written by (or for) this
            // sequence: a window that included rows nobody wrote would attend to
            // zeroes. With a shared prefix the row carries the donor's stamp, so
            // the written count — not `owner[]` — is what answers this.
            if rel >= self.written_rows(seq) {
                return Err(format!(
                    "attn_span: query {t} (sequence {seq}, position {rel}) resolves to cell \
                     {cell}, which was not written by sequence {seq}"
                ));
            }
            let hi = (first_cell + len).min(cell + 1);
            span[t] = first_cell as u32;
            span[n + t] = hi as u32;
        }
        let first = self.layers.values().next().map(|l| l.n_used);
        debug_assert!(
            self.layers.values().all(|l| Some(l.n_used) == first),
            "KV layers disagree about how much is written"
        );
        Ok(span)
    }

    /// Resolve each query's allowed cells into the **`kv_map`** input layout (C8b
    /// S2): `KV_MAP_MAX_SPANS` `(cell, len)` runs per query, zero-padded.
    ///
    /// `attn_span` names one contiguous range, which is exactly what a sequence
    /// reading part of its prefix in place does not have. This is the same
    /// resolution kept as a list: every span of the sequence that starts at or
    /// before the query's position, the last one clipped to the query's own row (a
    /// token may not attend to rows written after it). A window that needs more runs
    /// than the input carries is an error, never a truncated window.
    pub fn attn_map(&self, seq_ids: &[u32], positions: &[usize]) -> Result<Vec<u32>, String> {
        let n = seq_ids.len();
        if positions.len() != n {
            return Err(format!(
                "attn_map: {} sequence ids but {} positions",
                n,
                positions.len()
            ));
        }
        if self.layers.is_empty() {
            return Err("attn_map: no KV arena allocated".to_string());
        }
        let mut out = vec![0u32; n * KV_MAP_MAX_SPANS * 2];
        for t in 0..n {
            let seq = seq_ids[t];
            let rel = positions[t];
            let spans = self.spans_of(seq);
            if spans.is_empty() {
                return Err(format!(
                    "attn_map: query {t} belongs to sequence {seq}, which holds no cells"
                ));
            }
            if rel >= self.written_rows(seq) {
                return Err(format!(
                    "attn_map: query {t} (sequence {seq}, position {rel}) has not been written"
                ));
            }
            let mut k = 0usize;
            for &(base, cell, len) in spans {
                if base > rel {
                    break;
                }
                let take = len.min(rel + 1 - base);
                if take == 0 {
                    continue;
                }
                if k == KV_MAP_MAX_SPANS {
                    return Err(format!(
                        "attn_map: query {t} (sequence {seq}, position {rel}) needs more than \
                         {KV_MAP_MAX_SPANS} cell runs (spans {spans:?})"
                    ));
                }
                let at = (t * KV_MAP_MAX_SPANS + k) * 2;
                out[at] = cell as u32;
                out[at + 1] = take as u32;
                k += 1;
            }
            if k == 0 {
                return Err(format!(
                    "attn_map: query {t} (sequence {seq}, position {rel}) resolves to no cells \
                     (spans {spans:?})"
                ));
            }
        }
        Ok(out)
    }

    /// The maximal **free** cell runs, in cell order: free means "not covered by
    /// any live reservation" — the same notion `reserve_seq` uses, so a written
    /// row inside a released run counts as free even though its bytes are still
    /// there.
    pub fn free_runs(&self) -> Vec<(usize, usize)> {
        let n = self.n_ctx;
        let taken = self.occupied();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut open: Option<usize> = None;
        for (c, &t) in taken.iter().enumerate() {
            match (t, open) {
                (false, None) => open = Some(c),
                (true, Some(start)) => {
                    runs.push((start, c - start));
                    open = None;
                }
                _ => {}
            }
        }
        if let Some(start) = open {
            runs.push((start, n - start));
        }
        runs
    }

    /// Positions sequence `seq` has written — its shared prefix plus the rows it
    /// wrote itself (C8b S2: explicit, because a shared row carries the donor's
    /// ownership stamp and a scan of `owner[]` would stop at the first of them).
    ///
    /// This is the number of rows a reader may see and the bound `attn_span` uses.
    pub fn written_rows(&self, seq: SeqId) -> usize {
        self.seqs.get(&seq).map_or(0, |s| s.written)
    }

    /// Rows `seq` wrote into its **own** run — the rows a relocation copies, since
    /// the shared prefix is another sequence's memory (C8b S2).
    pub fn private_written(&self, seq: SeqId) -> usize {
        self.seqs.get(&seq).map_or(0, |s| s.private_written())
    }

    /// C8b S2: let `dst` read the first `rows` positions of `src` **in place**.
    ///
    /// The rows are not copied: `dst`'s span list gains an entry naming the donor's
    /// cells, so both sequences read one copy — which is the whole point, and the
    /// reason the read path needs a span map (`attn_map`) instead of one range.
    ///
    /// Refused loudly rather than approximated when the sharing cannot be expressed:
    /// an empty share, a donor that has not written that far, a destination that
    /// already shares or has written rows of its own (its positions would be
    /// displaced), and a donor prefix that is not **one contiguous cell range** —
    /// that last case is a donor which itself shares a prefix plus private rows, and
    /// it needs a multi-entry map the caller has to fall back from.
    pub fn share_prefix(&mut self, src: SeqId, dst: SeqId, rows: usize) -> Result<(), String> {
        if rows == 0 {
            return Err("share_prefix: asked to share 0 rows".to_string());
        }
        if src == dst {
            return Err(format!(
                "share_prefix: sequence {src} cannot share with itself"
            ));
        }
        let donor = self
            .seqs
            .get(&src)
            .copied()
            .ok_or_else(|| format!("share_prefix: source sequence {src} holds no run"))?;
        if donor.written < rows {
            return Err(format!(
                "share_prefix: sequence {src} has written {} rows, {rows} requested",
                donor.written
            ));
        }
        let dst_slot = self
            .seqs
            .get(&dst)
            .copied()
            .ok_or_else(|| format!("share_prefix: destination sequence {dst} holds no run"))?;
        if dst_slot.shared.rows > 0 {
            return Err(format!(
                "share_prefix: sequence {dst} already shares {} rows",
                dst_slot.shared.rows
            ));
        }
        if dst_slot.written > 0 {
            return Err(format!(
                "share_prefix: sequence {dst} has written {} rows of its own",
                dst_slot.written
            ));
        }
        let cell = self
            .cell_of(src, 0)
            .ok_or_else(|| format!("share_prefix: sequence {src} has no first cell"))?;
        // Contiguity: `rows` positions of the donor must be `rows` adjacent cells,
        // or the prefix would need more than one span entry.
        match self.cell_of(src, rows - 1) {
            Some(last) if last == cell + rows - 1 => {}
            _ => {
                return Err(format!(
                    "share_prefix: sequence {src} does not hold positions [0, {rows}) as one \
                     contiguous range (spans {:?})",
                    self.spans_of(src)
                ))
            }
        }
        let slot = self.seqs.get_mut(&dst).expect("checked above");
        slot.shared = SharedPrefix { cell, rows };
        slot.written = rows;
        self.refresh_spans(dst);
        Ok(())
    }

    /// C8b S3: the copy-on-write a store at position `t` needs, or `None` when `t`
    /// already resolves into the sequence's own run.
    ///
    /// A sequence that reads its prefix in place must never **write** through it: the
    /// cells belong to the donor, and a store there would corrupt every sharer. The
    /// sequence therefore gives the share up from `t` on — the first position this
    /// forward writes privately — and this returns the move that keeps the rows it
    /// already wrote at their positions while their run's base drops.
    ///
    /// Requirements, checked here rather than assumed: the run must have room for
    /// `shared.rows - t` more positions (`written` of them, plus the new ones, all
    /// inside `cap`). A sequence that cannot take the position privately is **no
    /// longer shareable** — the caller gets `Err`, and the alternative (writing into
    /// the donor's block) is exactly what this ticket forbids.
    pub fn private_row_for(&self, seq: SeqId, t: usize) -> Result<Option<KvShift>, String> {
        let Some(slot) = self.seqs.get(&seq).copied() else {
            // No run, no share: the classic single-sequence path.
            return Ok(None);
        };
        if slot.shared.rows == 0 || t >= slot.shared.rows {
            return Ok(None);
        }
        let d = slot.shared.rows - t;
        let rows = slot.private_written();
        if rows + d > slot.cap {
            return Err(format!(
                "private_row_for: sequence {seq} cannot take position {t} privately: its {rows} \
                 written row(s) would move {d} cell(s) into a run of {} cells (shared prefix of \
                 {})",
                slot.cap, slot.shared.rows
            ));
        }
        Ok(Some(KvShift {
            seq,
            base: t,
            from: slot.start,
            to: slot.start + d,
            rows,
        }))
    }

    /// Apply the **bookkeeping** half of a copy-on-write: the caller has already
    /// moved the rows with `Backend::copy_cells`, so this shifts the owner table by
    /// the same range and drops the share to `shift.base`.
    ///
    /// `written` is deliberately untouched: the sequence's readable positions are
    /// unchanged (`[0, written)`), only *where* `[base, written)` lives inside the
    /// run changes — which is why `private_written` grows by the shift's `d`.
    pub fn apply_private_row(&mut self, shift: &KvShift) -> Result<(), String> {
        let Some(slot) = self.seqs.get(&shift.seq).copied() else {
            return Err(format!(
                "apply_private_row: sequence {} holds no run",
                shift.seq
            ));
        };
        if slot.start != shift.from
            || slot.shared.rows <= shift.base
            || slot.private_written() != shift.rows
        {
            return Err(format!(
                "apply_private_row: sequence {} is not in the state the shift was planned from \
                 (run at {}, {} shared row(s), {} private row(s); shift {shift:?})",
                shift.seq,
                slot.start,
                slot.shared.rows,
                slot.private_written()
            ));
        }
        // Validate against every layer before mutating any of them: a partial
        // owner shift would leave the table describing rows that are not there.
        for l in self.layers.values() {
            if shift.to + shift.rows > l.owner.len() || shift.from + shift.rows > l.owner.len() {
                return Err(format!(
                    "apply_private_row: {:?} does not fit a {}-cell arena",
                    shift,
                    l.owner.len()
                ));
            }
        }
        let mut rows_moved = 0usize;
        for l in self.layers.values_mut() {
            rows_moved += shift.rows;
            if shift.rows > 0 {
                l.owner
                    .copy_within(shift.from..shift.from + shift.rows, shift.to);
            }
            // The cells the move vacated no longer hold a written row: they are the
            // new base's unwritten rows (this forward is about to write them).
            for cell in shift.from..shift.to {
                if l.owner[cell] == shift.seq {
                    l.owner[cell] = FREE;
                }
            }
            l.n_used = l
                .owner
                .iter()
                .rposition(|&o| o != FREE)
                .map_or(0, |i| i + 1);
        }
        {
            let slot = self.seqs.get_mut(&shift.seq).expect("checked above");
            slot.shared.rows = shift.base;
        }
        self.refresh_spans(shift.seq);
        self.cows += 1;
        self.cow_cells += rows_moved as u64;
        Ok(())
    }

    /// Fragmentation and utilisation counters (C3's acceptance surface; the
    /// same numbers F8 exports).
    pub fn arena_stats(&self) -> KvArenaStats {
        let reserved_cells: usize = self.seqs.values().map(|s| s.cap.min(self.n_ctx)).sum();
        let runs = self.free_runs();
        KvArenaStats {
            n_ctx: self.n_ctx,
            reserved_cells,
            // C8b S2: rows read in place from another sequence. They are one copy of
            // the arena's bytes shared by two sequences, so `reserved_cells` (the
            // private runs) does not count them and `owned_cells` (below) does — once.
            shared_cells: self.seqs.values().map(|s| s.shared.rows).sum(),
            owned_cells: self
                .layers
                .values()
                .next()
                .map_or(0, |l| l.owner.iter().filter(|&&o| o != FREE).count()),
            free_cells: self.n_ctx.saturating_sub(reserved_cells),
            free_runs: runs.len(),
            largest_free_run: runs.iter().map(|&(_, len)| len).max().unwrap_or(0),
            sequences: self.seqs.len(),
            defrags: self.defrags,
            cells_moved: self.cells_moved,
            cows: self.cows,
            cow_cells: self.cow_cells,
        }
    }

    /// The moves that compact the live runs **downward**, in application order.
    ///
    /// Runs are packed from cell 0 in ascending `start` order, each keeping its
    /// `cap`, so free space coalesces at the top of the arena. `need`: `None`
    /// compacts fully; `Some(n)` keeps only the shortest prefix of the plan whose
    /// last move leaves a free gap of at least `n` cells, which is what makes
    /// `reserve_seq(n)` succeed — moving runs that buy nothing is exactly what a
    /// defragmentation should not do.
    ///
    /// Pure: no mutation and no backend call, so the policy is unit-tested
    /// without a device. The data copy is the allocator's job
    /// (`Backend::copy_cells`), driven by the returned `rows`.
    pub fn compaction_plan(&self, need: Option<usize>) -> Vec<KvMove> {
        let mut runs: Vec<(SeqId, SeqSlot)> = self
            .seqs
            .iter()
            .filter(|(_, s)| s.cap > 0)
            .map(|(&seq, &slot)| (seq, slot))
            .collect();
        runs.sort_by_key(|&(seq, slot)| (slot.start, seq));
        let mut cursor = 0usize;
        let mut moves = Vec::new();
        for (i, &(seq, slot)) in runs.iter().enumerate() {
            if slot.start != cursor {
                moves.push(KvMove {
                    seq,
                    from: slot.start,
                    to: cursor,
                    rows: slot.private_written(),
                });
            }
            cursor += slot.cap;
            if let Some(n) = need {
                // Contiguous free space this prefix of the plan opens: from the
                // packed cursor up to the next run that stays where it is.
                let gap_end = runs.get(i + 1).map_or(self.n_ctx, |&(_, next)| next.start);
                if gap_end.saturating_sub(cursor) >= n {
                    break;
                }
            }
        }
        moves
    }

    /// Apply the **bookkeeping** half of a plan: move each run's ownership in
    /// every layer and renumber the run table. Returns the rows moved per layer.
    ///
    /// The caller has already copied (or is about to copy) the data with
    /// `Backend::copy_cells`; this function never touches a buffer, so it stays
    /// backend-agnostic and provable on CPU. It also leaves `identity` alone: a
    /// compaction renumbers *cells* while every caller's position follows its
    /// run's new `start` (the report), so the position-to-cell relationship the
    /// flag describes is unchanged.
    ///
    /// Refuses any plan whose destinations overlap, or that would run past the
    /// arena — a plan violation is a bug, and the alternative (overwriting a live
    /// sequence's rows) corrupts a session silently. Moves in **both** directions
    /// are accepted (C7b: growing a run pushes the runs above it up).
    pub fn apply_moves(&mut self, moves: &[KvMove]) -> Result<usize, String> {
        if moves.is_empty() {
            return Ok(0);
        }
        // The caller copied the rows in this order (`order_moves`), so the table has
        // to follow it: shifting owner stamps in the plan's raw order would move a
        // sequence's stamps through cells another sequence has already overwritten.
        let mut ordered = moves.to_vec();
        order_moves(&mut ordered);
        let moves: &[KvMove] = &ordered;
        // Validate against the post-state layout before mutating anything.
        let mut planned: Vec<(usize, usize)> = Vec::with_capacity(self.seqs.len());
        for (&seq, slot) in self.seqs.iter() {
            let to = match moves.iter().find(|m| m.seq == seq) {
                Some(m) => {
                    if slot.start != m.from {
                        return Err(format!(
                            "apply_moves: sequence {seq} starts at {}, the plan says {}",
                            slot.start, m.from
                        ));
                    }
                    // Both directions are legal (C7b): the backend copies rows so
                    // an overlapping move never clobbers a row still to be read, and
                    // the non-overlap check below is what makes a plan valid,
                    // whichever way each run travels.
                    if m.rows > slot.cap {
                        return Err(format!(
                            "apply_moves: {m:?} copies {} rows but the run holds {}",
                            m.rows, slot.cap
                        ));
                    }
                    m.to
                }
                None => slot.start,
            };
            planned.push((to, slot.cap));
        }
        planned.sort_unstable();
        for w in planned.windows(2) {
            if w[0].0 + w[0].1 > w[1].0 {
                return Err(format!(
                    "apply_moves: runs would overlap after the plan ({}..{} then {})",
                    w[0].0,
                    w[0].0 + w[0].1,
                    w[1].0
                ));
            }
        }
        if let Some(&(start, cap)) = planned.last() {
            if start + cap > self.n_ctx {
                return Err(format!(
                    "apply_moves: the plan ends at {} but the arena holds {} cells",
                    start + cap,
                    self.n_ctx
                ));
            }
        }

        let layers = self.layers.len();
        let mut rows_moved = 0usize;
        for m in moves {
            rows_moved += m.rows;
            for layer in self.layers.values_mut() {
                let rows = m.rows.min(layer.owner.len().saturating_sub(m.from));
                if rows > 0 {
                    // `copy_within` is a memmove, so the owner table follows the same
                    // direction the backend moved the rows.
                    layer.owner.copy_within(m.from..m.from + rows, m.to);
                    // Free the cells the move vacated: every cell in the span the move
                    // covered that is outside the destination and still carries *this*
                    // sequence's stamp. The stamp guard means a plan can never clear
                    // another sequence's ownership, whichever way the run travelled.
                    let lo = m.to.min(m.from);
                    let hi = (m.to.max(m.from) + rows).min(layer.owner.len());
                    for cell in lo..hi {
                        if !(m.to..m.to + rows).contains(&cell) && layer.owner[cell] == m.seq {
                            layer.owner[cell] = FREE;
                        }
                    }
                }
                layer.n_used = layer
                    .owner
                    .iter()
                    .rposition(|&o| o != FREE)
                    .map_or(0, |i| i + 1);
            }
            if let Some(slot) = self.seqs.get_mut(&m.seq) {
                slot.start = m.to;
            }
            self.refresh_spans(m.seq);
            // C8b S2: another sequence may read these rows in place (a shared
            // prefix), so its pointer has to follow them. Only a sharer's prefix is
            // affected — every other span either belongs to the moved run itself or
            // is some sequence's own reservation, which its own move (or nothing)
            // handles. The plan copies `m.rows` contiguous rows and a share is always
            // a prefix of a run's written rows, so an overlapping share is either
            // entirely inside the copied range (renumber) or entirely outside it; a
            // straddling one would have to split, which no plan produces — refuse it
            // rather than point half a window at stale rows.
            if m.rows > 0 {
                let (mlo, mhi) = (m.from, m.from + m.rows);
                let mut sharers: Vec<SeqId> = Vec::new();
                for (&other, slot) in self.seqs.iter() {
                    if other == m.seq || slot.shared.rows == 0 {
                        continue;
                    }
                    let (lo, hi) = (slot.shared.cell, slot.shared.cell + slot.shared.rows);
                    if lo >= mhi || hi <= mlo {
                        continue;
                    }
                    if lo < mlo || hi > mhi {
                        return Err(format!(
                            "apply_moves: sequence {other}'s {}-row shared prefix ({lo}, {hi}) \
                             straddles the {}-row range moved from {mlo} to {} — it would have \
                             to split",
                            slot.shared.rows, m.rows, m.to
                        ));
                    }
                    sharers.push(other);
                }
                for other in sharers {
                    {
                        let slot = self.seqs.get_mut(&other).expect("checked above");
                        slot.shared.cell = m.to + (slot.shared.cell - m.from);
                    }
                    self.refresh_spans(other);
                }
            }
        }
        self.defrags += 1;
        self.cells_moved += (rows_moved * layers) as u64;
        Ok(rows_moved)
    }
}

/// Order relocations so that an overlapping move never lands on a row that has not
/// been copied yet: runs moving **up** go top-down, runs moving **down** bottom-up,
/// and every upward move precedes the downward ones — a downward destination is
/// always below its own source and below every run above it, so nothing is in its
/// way once the upward moves are done.
///
/// Both the data copy (the allocator) and the bookkeeping ([`KvCache::apply_moves`])
/// use this order, because the owner table is a mirror of where the rows live: they
/// have to move together, in the same sequence.
pub fn order_moves(moves: &mut [KvMove]) {
    moves.sort_by(|a, b| match (a.to > a.from, b.to > b.from) {
        (true, true) => b.from.cmp(&a.from),
        (false, false) => a.from.cmp(&b.from),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
    });
}

/// C8b S3: the move a copy-on-write performs inside one sequence's own run.
///
/// A sequence that reads its prefix in place (C8b S2) owns the positions
/// `[shared.rows, shared.rows + cap)` of its run. A store at an earlier position `t`
/// would land in the **donor's** cells, so the sequence gives the share up from `t`
/// on: the run's position base drops to `t` and the rows it already wrote shift up
/// by `shared.rows - t` to keep their positions. `rows` is that move's length in
/// cells — 0 when the sequence has not written privately yet, in which case the
/// rebase stands alone. The data copy is the allocator's job
/// ([`KvMove`]'s arrangement), which is why this plan is pure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShift {
    pub seq: SeqId,
    /// Position the run's first cell holds after the shift (the new `shared.rows`).
    pub base: usize,
    /// Cell the sequence's written rows move from (the run's `start`).
    pub from: usize,
    /// ... and to: `from + (old shared rows - base)`.
    pub to: usize,
    /// Rows the move covers (the sequence's own written rows).
    pub rows: usize,
}

/// One run relocation produced by [`KvCache::compaction_plan`] (C3).
///
/// `from`/`to` are cell indices; `rows` is the sequence's **written** length, so
/// `rows <= cap` and `rows == 0` means the run moves without a data copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMove {
    pub seq: SeqId,
    pub from: usize,
    pub to: usize,
    pub rows: usize,
}

/// Fragmentation and utilisation counters (C3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvArenaStats {
    pub n_ctx: usize,
    pub reserved_cells: usize,
    /// C8b S2: rows a sequence reads in place from another sequence's run. Two
    /// sharers of one prefix report the same cells here and `owned_cells` counts
    /// them once, so the saving is visible without double-counting the arena.
    pub shared_cells: usize,
    pub owned_cells: usize,
    pub free_cells: usize,
    /// Maximal free runs — the "node count" a compaction reduces.
    pub free_runs: usize,
    pub largest_free_run: usize,
    pub sequences: usize,
    pub defrags: u64,
    pub cells_moved: u64,
    /// C8b S3: copy-on-write events, and the sequence rows they moved (summed over
    /// layers) — a shared prefix costs nothing until a store lands inside it.
    pub cows: u64,
    pub cow_cells: u64,
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
        BufRef::own(Backend::CPU, id, usize::MAX)
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
        c.note_written(SEQ_MAIN, 0, &cells);
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
        c.note_written(SEQ_MAIN, 0, &[0, 1, 2, 3]);
        // Query at position 2 may see cells 0..3; the last query sees 0..4.
        assert_eq!(
            c.attn_span(&[SEQ_MAIN, SEQ_MAIN], &[2, 3]).unwrap(),
            vec![0, 0, 3, 4],
            "lo row then hi row"
        );
    }

    /// E1's acceptance at the store level, on E2's reservations: two sequences
    /// in one arena get two disjoint windows, so no query can see the other's
    /// cells.
    #[test]
    fn two_sequences_resolve_to_disjoint_windows() {
        let mut c = cache(8);
        // Sequence 0 reserves cells 0..3, sequence 1 the next free run of 3.
        let a = c.reserve_seq(SEQ_MAIN, 3).unwrap();
        let b = c.reserve_seq(1, 3).unwrap();
        assert_eq!((a.start, a.cap), (0, 3));
        assert_eq!((b.start, b.cap), (3, 3), "first-fit after the first run");
        // Each sequence owns the rows it wrote (E1's window cap is the
        // reservation; the written-row check keeps unwritten rows out).
        for layer in [0usize, 1] {
            c.note_written(SEQ_MAIN, layer, &[0, 1, 2]);
            c.note_written(1, layer, &[3, 4, 5]);
        }
        assert_eq!(c.seq_range(0, SEQ_MAIN).unwrap(), Some((0, 3)));
        assert_eq!(c.seq_range(0, 1).unwrap(), Some((3, 3)));
        assert_eq!(c.seq_range(1, 1).unwrap(), Some((3, 3)), "layers agree");
        // A token of each sequence, each at its own last position.
        // C6: positions are sequence-relative; cell = start + position.
        let span = c.attn_span(&[SEQ_MAIN, 1], &[2, 2]).unwrap();
        assert_eq!(&span[..2], &[0, 3], "starts are each sequence's own");
        assert_eq!(&span[2..], &[3, 6], "ends are exclusive and causal");
        assert!(
            span[2] >= span[1],
            "sequence 1's window starts where sequence 0's ends"
        );
    }

    /// E2's reservations: disjoint, first-fit, released, and loud when the arena
    /// cannot hold a sequence (no silent overlap, no moving).
    #[test]
    fn reservations_are_disjoint_first_fit_and_releasable() {
        let mut c = cache(8);
        assert_eq!(c.reserve_seq(0, 3).unwrap().start, 0);
        assert_eq!(c.reserve_seq(1, 3).unwrap().start, 3);
        // An existing reservation is returned unchanged when it fits.
        assert_eq!(c.reserve_seq(0, 2).unwrap().start, 0);
        // …and growth is refused, not silently relocated.
        let err = c.reserve_seq(0, 4).unwrap_err();
        assert!(err.contains("no growth"), "got: {err}");
        // The arena is full: 2 free cells cannot hold a third sequence of 3.
        let err = c.reserve_seq(2, 3).unwrap_err();
        assert!(err.contains("no free run of 3"), "got: {err}");
        // Releasing frees exactly its cells, and the next sequence reuses them.
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[3, 4, 5]);
        }
        assert_eq!(c.release_seq(1), 3);
        assert_eq!(c.seq_range(0, 1).unwrap(), None);
        assert!(
            c.get(0).unwrap().owner[3..6].iter().all(|&o| o == FREE),
            "released cells lose their owner"
        );
        assert_eq!(c.reserve_seq(2, 3).unwrap().start, 3, "freed run reused");
        assert_eq!(
            c.release_seq(7),
            0,
            "releasing an unknown sequence is a no-op"
        );
    }

    /// C8b S1: with one span per sequence, the span list must resolve exactly like the
    /// run it describes — and it must follow the run when the run moves, or when it goes
    /// away. A miss here is what the resolver turns into a loud error.
    #[test]
    fn a_single_span_resolves_exactly_like_its_run() {
        let mut c = cache(16);
        c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let slot = c.reserve_seq(0, 4).unwrap(); // [4, 8)
        assert_eq!(c.spans_of(0), &[(0, slot.start, 4)]);
        for pos in 0..4 {
            assert_eq!(c.cell_of(0, pos), Some(slot.start + pos), "pos {pos}");
        }
        assert_eq!(c.cell_of(0, 4), None, "past the run");
        assert_eq!(c.spans_of(9), &[] as &[(usize, usize, usize)], "no run");

        // A compaction moves the run down into the gap sequence 1 left, and the span
        // list has to follow it (this is the maintenance point that would otherwise
        // drift silently).
        c.release_seq(1);
        let plan = c.compaction_plan(None);
        assert!(!plan.is_empty(), "sequence 0 must move down: {plan:?}");
        c.apply_moves(&plan).unwrap();
        let moved = c.seq_slot(0).unwrap().start;
        assert_eq!(moved, 0, "packed to the bottom");
        assert_eq!(c.spans_of(0), &[(0, 0, 4)]);
        assert_eq!(c.cell_of(0, 3), Some(3));

        // Growing republishes it, and releasing drops it.
        assert_eq!(c.set_cap(0, 6).unwrap(), Vec::new());
        assert_eq!(c.spans_of(0), &[(0, 0, 6)]);
        c.release_seq(0);
        assert_eq!(c.cell_of(0, 0), None);
    }

    /// C8b S1b: the **read** path resolves through the same span list the write path
    /// uses, and with one span it must produce exactly the range the old `start +
    /// position` arithmetic produced. The two are compared here rather than trusted,
    /// because this is the change that would silently shift an attention window.
    #[test]
    fn a_single_span_attn_window_equals_the_run_arithmetic() {
        let mut c = cache(16);
        // A run that does not start at cell 0, so `cell == position` cannot hide a
        // regression in the resolution.
        c.reserve_seq(7, 3).unwrap(); // [0, 3)
        let a = c.reserve_seq(SEQ_MAIN, 4).unwrap(); // [3, 7)
        let b = c.reserve_seq(1, 2).unwrap(); // [7, 9)
        assert_eq!((a.start, b.start), (3, 7));
        for layer in [0usize, 1] {
            c.note_written(SEQ_MAIN, layer, &[3, 4, 5, 6]);
            c.note_written(1, layer, &[7, 8]);
        }
        for positions in [vec![0], vec![0, 1, 2, 3], vec![3, 3, 0], vec![1, 0]] {
            let seqs = vec![SEQ_MAIN; positions.len()];
            let span = c.attn_span(&seqs, &positions).unwrap();
            let n = positions.len();
            for (t, &rel) in positions.iter().enumerate() {
                assert_eq!(
                    (span[t], span[n + t]),
                    (a.start as u32, (a.start + rel + 1) as u32),
                    "position {rel} of sequence {SEQ_MAIN}"
                );
            }
        }
        // Two sequences still get their own disjoint windows through the span list.
        assert_eq!(
            c.attn_span(&[SEQ_MAIN, 1], &[2, 1]).unwrap(),
            vec![3, 7, 6, 9]
        );
        // A sequence whose span list is gone is a loud error, never a window at 0.
        c.release_seq(SEQ_MAIN);
        let err = c.attn_span(&[SEQ_MAIN], &[0]).unwrap_err();
        assert!(err.contains("holds no cells"), "got: {err}");
    }

    /// C8b S1b's boundary: a multi-span sequence has no single contiguous window to
    /// hand the kernel, so the read path must refuse it instead of guessing a span.
    /// The span list is written directly here — no mutation produces several spans
    /// yet, that is S2 — so the test pins the refusal before the layout exists.
    #[test]
    fn a_multi_span_sequence_is_refused_by_the_read_path() {
        let mut c = cache(16);
        c.reserve_seq(SEQ_MAIN, 4).unwrap();
        for layer in [0usize, 1] {
            c.note_written(SEQ_MAIN, layer, &[0, 1, 2, 3]);
        }
        // A shared prefix would look like this: rows 0..2 at cell 0, the rest private
        // at cell 8 — two ranges, no single `[lo, hi)` window.
        let mut slot = c.seq_slot(SEQ_MAIN).unwrap();
        slot.start = 8;
        c.seqs.insert(SEQ_MAIN, slot);
        c.spans.insert(SEQ_MAIN, vec![(0, 0, 2), (2, 8, 2)]);
        assert_eq!(
            c.cell_of(SEQ_MAIN, 3),
            Some(9),
            "the resolver still reads it"
        );
        let err = c.attn_span(&[SEQ_MAIN], &[3]).unwrap_err();
        assert!(err.contains("2 spans"), "got: {err}");
        assert!(err.contains("kv_map"), "got: {err}");
    }

    /// C8b S2: sharing a prefix means the destination reads the donor's cells — one
    /// copy of the bytes, two sequences — and keeps writing its own positions into
    /// its own run. The two halves of the address space have to come back out of the
    /// span list as one entry each.
    #[test]
    fn a_shared_prefix_is_read_in_place_and_written_past() {
        let mut c = cache(16);
        let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        assert_eq!(c.share_prefix(1, 2, 4), Ok(()));
        assert_eq!(
            c.spans_of(2),
            &[(0, donor.start, 4), (4, dst.start, 2)],
            "shared prefix then the private tail"
        );
        assert_eq!(c.cell_of(2, 0), Some(donor.start), "read in place");
        assert_eq!(c.cell_of(2, 3), Some(donor.start + 3));
        assert_eq!(c.cell_of(2, 4), Some(dst.start), "then its own run");
        assert_eq!(c.written_rows(2), 4, "the shared rows count as written");
        assert_eq!(c.private_written(2), 0, "but none of them are its own");
        // The sharer's own storage is not in its run's cells.
        assert_eq!(c.get(0).unwrap().owner[dst.start], FREE);

        c.own_positions(2, &[4, 5]).unwrap();
        assert_eq!(c.written_rows(2), 6);
        assert_eq!(c.private_written(2), 2);
        assert_eq!(c.get(0).unwrap().owner[dst.start], 2, "its own rows");
        assert_eq!(
            c.get(0).unwrap().owner[donor.start],
            1,
            "a shared row keeps the donor's stamp"
        );
        let stats = c.arena_stats();
        assert_eq!(stats.shared_cells, 4, "the saving is visible");
        assert_eq!(stats.reserved_cells, 6, "only the two private runs");
        // The donor is untouched by any of it and still reads its own rows.
        assert_eq!(c.spans_of(1), &[(0, donor.start, 4)]);
        assert_eq!(c.written_rows(1), 4);
    }

    /// C8b S3: a store inside a shared prefix takes a **private row** — the share
    /// shrinks to the first position the forward writes, and the rows the sequence
    /// already owns shift up inside its own run so they keep their positions. The
    /// donor's cells are never named again by that position.
    #[test]
    fn a_store_inside_a_shared_prefix_takes_a_private_row() {
        let mut c = cache(16);
        let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let dst = c.reserve_seq(2, 4).unwrap(); // [4, 8)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        c.share_prefix(1, 2, 4).unwrap();
        c.own_positions(2, &[4, 5]).unwrap(); // two rows of its own at [4, 6)
        assert_eq!(c.spans_of(2), &[(0, donor.start, 4), (4, dst.start, 4)]);
        assert_eq!(c.private_written(2), 2);

        // Position 3 and everything after it is written privately; 4 is already
        // private, and an unknown sequence has nothing to copy.
        assert_eq!(c.private_row_for(2, 4), Ok(None), "already private");
        assert_eq!(c.private_row_for(7, 0), Ok(None), "no run, no share");
        let shift = c.private_row_for(2, 3).unwrap().expect("a copy is needed");
        assert_eq!(
            shift,
            KvShift {
                seq: 2,
                base: 3,
                from: dst.start,
                to: dst.start + 1,
                rows: 2,
            }
        );
        c.apply_private_row(&shift).unwrap();

        // The address space is the remaining share plus the whole run, whose base
        // dropped to the position the store starts at.
        assert_eq!(c.spans_of(2), &[(0, donor.start, 3), (3, dst.start, 4)]);
        for p in 0..3 {
            assert_eq!(c.cell_of(2, p), Some(donor.start + p), "still shared");
        }
        assert_eq!(c.cell_of(2, 3), Some(dst.start), "private from here on");
        assert_eq!(c.cell_of(2, 4), Some(dst.start + 1), "the row it moved up");
        assert_eq!(c.cell_of(2, 5), Some(dst.start + 2));
        assert_eq!(c.written_rows(2), 6, "no readable position was lost");
        assert_eq!(c.private_written(2), 3, "and the run holds three of them");
        // The owner table followed the move: [4, 6) -> [5, 7), and the vacated cell
        // is the new base's unwritten row.
        let owner = &c.get(0).unwrap().owner;
        assert_eq!(owner[dst.start], FREE, "vacated");
        assert_eq!(owner[dst.start + 1], 2);
        assert_eq!(owner[dst.start + 2], 2);
        assert_eq!(owner[donor.start], 1, "the donor keeps its own row");
        let stats = c.arena_stats();
        assert_eq!((stats.cows, stats.cow_cells), (1, 4), "2 rows x 2 layers");
        assert_eq!(stats.shared_cells, 3, "the saving shrank with the share");
        // A query at the copy-on-written row reads the private cells and the
        // remaining share — and the donor is untouched.
        let map = c.attn_map(&[2], &[4]).unwrap();
        assert_eq!(&map[..4], &[donor.start as u32, 3, dst.start as u32, 2]);
        assert_eq!(c.written_rows(1), 4);
        assert_eq!(c.spans_of(1), &[(0, donor.start, 4)]);
    }

    /// A sequence that diverges earlier than it did last time copies again: the share
    /// only ever shrinks and the run only ever extends its own written range, so the
    /// address space stays at most two spans however often this happens.
    #[test]
    fn a_copy_on_write_can_shrink_the_share_twice() {
        let mut c = cache(16);
        let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let dst = c.reserve_seq(2, 6).unwrap(); // [4, 10)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        c.share_prefix(1, 2, 4).unwrap();
        c.own_positions(2, &[4, 5]).unwrap();
        // First the store reaches position 3, then 1, then 0.
        let steps = [
            (3usize, 3usize, 4usize, 5usize, 2usize),
            (1, 1, 4, 6, 3),
            (0, 0, 4, 5, 5),
        ];
        for (t, base, from, to, rows) in steps {
            let shift = c.private_row_for(2, t).unwrap().expect("a copy is needed");
            assert_eq!(
                (shift.base, shift.from, shift.to, shift.rows),
                (base, from, to, rows)
            );
            c.apply_private_row(&shift).unwrap();
        }
        assert_eq!(c.spans_of(2), &[(0, dst.start, 6)], "the share is gone");
        assert_eq!(c.cell_of(2, 0), Some(dst.start));
        assert_eq!(c.cell_of(2, 4), Some(dst.start + 4));
        assert_eq!(c.cell_of(2, 5), Some(dst.start + 5));
        assert_eq!(c.written_rows(2), 6);
        assert_eq!(c.private_written(2), 6);
        assert_eq!(
            c.spans_of(1),
            &[(0, donor.start, 4)],
            "the donor is untouched"
        );
        assert_eq!(c.arena_stats().cows, 3);
    }

    /// The other end of the same rule: a store at position 0 gives the share up
    /// entirely, and the sequence's whole run becomes its own.
    #[test]
    fn a_copy_on_write_to_the_first_position_drops_the_share() {
        let mut c = cache(16);
        let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let dst = c.reserve_seq(2, 6).unwrap(); // [4, 10)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        c.share_prefix(1, 2, 4).unwrap();
        c.own_positions(2, &[4, 5]).unwrap();
        let shift = c.private_row_for(2, 0).unwrap().expect("a copy is needed");
        assert_eq!((shift.base, shift.from, shift.to, shift.rows), (0, 4, 8, 2));
        c.apply_private_row(&shift).unwrap();
        assert_eq!(
            c.spans_of(2),
            &[(0, dst.start, 6)],
            "one run, no share left"
        );
        assert_eq!(c.cell_of(2, 0), Some(dst.start));
        assert_eq!(c.cell_of(2, 4), Some(dst.start + 4), "its own row, moved");
        assert_eq!(c.seq_slot(2).unwrap().shared.rows, 0);
        assert_eq!(c.written_rows(2), 6);
    }

    /// A sequence whose run has no room for the extra rows cannot take the store
    /// privately: that is a loud refusal (gate 3), never a write into the donor's
    /// cells.
    #[test]
    fn a_copy_on_write_refuses_a_run_without_room() {
        let mut c = cache(16);
        c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        c.share_prefix(1, 2, 4).unwrap();
        c.own_positions(2, &[4, 5]).unwrap(); // the run is exactly full
        let err = c.private_row_for(2, 0).unwrap_err();
        assert!(
            err.contains("cannot take position 0 privately"),
            "got: {err}"
        );
        assert!(err.contains("run of 2 cells"), "got: {err}");
        let err = c.private_row_for(2, 2).unwrap_err();
        assert!(
            err.contains("cannot take position 2 privately"),
            "got: {err}"
        );
        assert_eq!(c.spans_of(2).len(), 2, "the refusals changed nothing");
        assert_eq!(c.written_rows(2), 6);
        // A plan from a state the store is no longer in is refused rather than
        // applied to the wrong rows.
        let stale = KvShift {
            seq: 2,
            base: 3,
            from: dst.start + 1,
            to: dst.start + 2,
            rows: 1,
        };
        let err = c.apply_private_row(&stale).unwrap_err();
        assert!(err.contains("not in the state"), "got: {err}");
        let gone = KvShift {
            seq: 9,
            base: 0,
            from: 0,
            to: 1,
            rows: 1,
        };
        assert!(c
            .apply_private_row(&gone)
            .unwrap_err()
            .contains("holds no run"));
        let stats = c.arena_stats();
        assert_eq!((stats.cows, stats.cow_cells), (0, 0), "nothing ran");
    }

    /// A released donor must not hand its rows to somebody else while a sharer
    /// still reads them (C8b S2 gate 3): occupancy comes from the span lists, so
    /// the cells stay taken even though no slot reserves them any more.
    #[test]
    fn releasing_the_donor_keeps_the_shared_rows_taken() {
        let mut c = cache(16);
        let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        c.share_prefix(1, 2, 4).unwrap();
        assert_eq!(c.release_seq(1), 4, "the donor's run is what it freed");
        assert_eq!(c.cell_of(2, 0), Some(donor.start), "still reading the rows");
        assert_eq!(c.written_rows(2), 4);
        let held = donor.start..donor.start + 4;
        assert!(
            c.free_runs()
                .iter()
                .all(|&(s, l)| s + l <= held.start || s >= held.end),
            "the shared rows are not free: {:?}",
            c.free_runs()
        );
        // A third sequence cannot be placed on top of them.
        let err = c.reserve_seq(3, 16).unwrap_err();
        assert!(err.contains("no free run"), "got: {err}");
        // Once the sharer drops them they are free again.
        c.release_seq(2);
        assert_eq!(c.free_runs(), vec![(0, 16)]);
    }

    /// A compaction moves the donor's rows and **every sharer's pointer has to
    /// follow** — the requirement the plan calls out for a shared block.
    #[test]
    fn a_shared_prefix_follows_the_donors_rows_when_they_move() {
        let mut c = cache(16);
        c.reserve_seq(0, 4).unwrap(); // [0, 4) — released below, opening the gap
        let donor = c.reserve_seq(1, 4).unwrap(); // [4, 8)
        let dst = c.reserve_seq(2, 2).unwrap(); // [8, 10)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[4, 5, 6, 7]);
        }
        c.share_prefix(1, 2, 4).unwrap();
        assert_eq!(c.spans_of(2), &[(0, 4, 4), (4, dst.start, 2)]);
        c.release_seq(0);
        let plan = c.compaction_plan(None);
        assert_eq!(plan.len(), 2, "both runs pack down: {plan:?}");
        c.apply_moves(&plan).unwrap();
        let moved_donor = c.seq_slot(1).unwrap();
        let moved_dst = c.seq_slot(2).unwrap();
        assert_eq!(moved_donor.start, 0, "the donor packed to the bottom");
        assert_eq!(moved_dst.start, 4);
        assert_eq!(
            moved_dst.shared.cell, 0,
            "the sharer's prefix followed the donor"
        );
        assert_eq!(c.cell_of(2, 0), Some(0));
        assert_eq!(c.cell_of(2, 4), Some(4), "and its own run moved with it");
        assert_eq!(c.spans_of(2), &[(0, 0, 4), (4, 4, 2)]);
    }

    /// What sharing cannot express is refused, never approximated: an empty share,
    /// a donor that has not written that far, itself, a destination that already
    /// shares or has written rows, and a donor prefix that is not one cell range.
    #[test]
    fn sharing_refuses_what_it_cannot_express() {
        let mut c = cache(16);
        let err = c.share_prefix(1, 2, 4).unwrap_err();
        assert!(err.contains("holds no run"), "got: {err}");
        c.reserve_seq(1, 4).unwrap(); // [0, 4)
        c.reserve_seq(2, 2).unwrap(); // [4, 6)
        c.reserve_seq(3, 2).unwrap(); // [6, 8)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        assert!(c.share_prefix(1, 2, 0).unwrap_err().contains("0 rows"));
        assert!(c
            .share_prefix(1, 2, 5)
            .unwrap_err()
            .contains("has written 4"));
        assert!(c
            .share_prefix(1, 1, 2)
            .unwrap_err()
            .contains("cannot share with itself"));
        c.share_prefix(1, 2, 4).unwrap();
        assert!(c
            .share_prefix(1, 2, 2)
            .unwrap_err()
            .contains("already shares"));
        // A destination that has written rows of its own would have them displaced.
        let own = c.reserve_seq(4, 2).unwrap(); // [8, 10)
        c.own_positions(4, &[0]).unwrap();
        let err = c.share_prefix(1, 4, 2).unwrap_err();
        assert!(err.contains("written 1 rows of its own"), "got: {err}");
        assert_eq!(c.cell_of(4, 0), Some(own.start), "and it still resolves");
        // Sequence 3 shares 4 rows and then writes two of its own at [6, 8):
        // positions [0, 6) are two cell ranges, so a 5-row share from it must
        // refuse instead of pointing the prefix across the gap.
        c.share_prefix(1, 3, 4).unwrap();
        c.own_positions(3, &[4, 5]).unwrap();
        assert_eq!(c.spans_of(3).len(), 2);
        let dst = c.reserve_seq(5, 2).unwrap(); // [10, 12)
        let err = c.share_prefix(3, 5, 5).unwrap_err();
        assert!(err.contains("one"), "got: {err}");
        assert!(err.contains("contiguous"), "got: {err}");
        assert_eq!(
            c.cell_of(5, 0),
            Some(dst.start),
            "the refusal changed nothing"
        );
    }

    /// C8b S2: the map is `attn_span` kept as a list — one run when a sequence shares
    /// nothing, two when it reads a prefix in place, and a refusal (never a truncated
    /// window) when the input cannot carry the runs.
    #[test]
    fn the_map_lists_a_querys_runs_and_refuses_to_truncate() {
        let mut c = cache(16);
        let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
        let dst = c.reserve_seq(2, 2).unwrap(); // [4, 6)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        c.share_prefix(1, 2, 4).unwrap();
        c.own_positions(2, &[4, 5]).unwrap();
        let map = c.attn_map(&[2, 2, 2], &[2, 4, 5]).unwrap();
        let k = KV_MAP_MAX_SPANS * 2;
        assert_eq!(
            &map[0..k],
            &[donor.start as u32, 3, 0, 0, 0, 0, 0, 0],
            "only the shared prefix is in reach at position 2"
        );
        assert_eq!(
            &map[k..2 * k],
            &[donor.start as u32, 4, dst.start as u32, 1, 0, 0, 0, 0],
            "position 4 crosses into the private run"
        );
        assert_eq!(
            &map[2 * k..3 * k],
            &[donor.start as u32, 4, dst.start as u32, 2, 0, 0, 0, 0]
        );
        // With one span the map and the span agree, which is what keeps the
        // single-span path bitwise.
        let span = c.attn_span(&[1], &[3]).unwrap();
        let single = c.attn_map(&[1], &[3]).unwrap();
        assert_eq!(&single[0..2], &[span[0], span[1] - span[0]]);
        assert_eq!(&single[2..], &[0u32; 6], "unused slots are zero-length");
        assert!(c
            .attn_map(&[2], &[6])
            .unwrap_err()
            .contains("has not been written"));
        assert!(c
            .attn_map(&[9], &[0])
            .unwrap_err()
            .contains("holds no cells"));
        // More runs than the input carries: written directly, because no store
        // mutation produces five spans yet.
        c.spans.insert(
            1,
            vec![(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1)],
        );
        c.note_written(1, 0, &[4]); // the fifth span's row
        let err = c.attn_map(&[1], &[4]).unwrap_err();
        assert!(err.contains("more than"), "got: {err}");
    }

    /// A physical removal (C2) renumbers cells across the whole arena, so it cannot
    /// run while a prefix is shared — the rows it would move belong to someone else.
    #[test]
    fn a_shared_prefix_stops_a_physical_removal() {
        let mut c = cache(16);
        let donor = c.reserve_seq(1, 4).unwrap(); // [0, 4)
        c.reserve_seq(SEQ_MAIN, 4).unwrap(); // [4, 8)
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[0, 1, 2, 3]);
        }
        c.share_prefix(1, SEQ_MAIN, 4).unwrap();
        c.release_seq(1);
        assert_eq!(c.cell_of(SEQ_MAIN, 0), Some(donor.start));
        let err = c.after_shift(1).unwrap_err();
        assert!(err.contains("shares a 4-row prefix"), "got: {err}");
    }

    /// C7b: growing a run moves the runs above it **up**, and their ownership
    /// travels with the rows. This is the direction CUDA's row-move kernel could not
    /// do before, and the plan — not the caller — decides it.
    #[test]
    fn growing_a_run_pushes_the_runs_above_it_up() {
        let mut c = cache(16);
        assert_eq!(c.reserve_seq(0, 4).unwrap().start, 0);
        assert_eq!(c.reserve_seq(1, 4).unwrap().start, 4);
        assert_eq!(c.reserve_seq(2, 4).unwrap().start, 8);
        for layer in [0usize, 1] {
            c.note_written(1, layer, &[4, 5, 6]);
            c.note_written(2, layer, &[8, 9]);
        }
        // Growing the *lowest* run pushes both runs above it up by two.
        let plan = c.set_cap(0, 6).unwrap();
        assert_eq!(plan.len(), 2, "both runs above move: {plan:?}");
        assert!(
            plan.iter().all(|m| m.to > m.from),
            "both moves are upward: {plan:?}"
        );
        assert_eq!(c.seq_slot(0).unwrap().cap, 6);
        c.apply_moves(&plan).unwrap();
        assert_eq!(c.seq_slot(1).unwrap().start, 6);
        assert_eq!(c.seq_slot(2).unwrap().start, 10);
        let owner = |cell: usize| c.get(0).unwrap().owner[cell];
        assert_eq!(
            (owner(6), owner(7), owner(8)),
            (1, 1, 1),
            "run 1's three rows"
        );
        assert_eq!((owner(10), owner(11)), (2, 2), "run 2's two rows");
        for cell in [4usize, 5, 12, 13] {
            assert_eq!(owner(cell), FREE, "cell {cell} is vacant after the move");
        }
    }

    /// C7b: the two refusals that keep a resize from trading data or truth away.
    #[test]
    fn a_resize_refuses_to_eat_rows_or_overcommit_the_arena() {
        let mut c = cache(8);
        c.reserve_seq(0, 3).unwrap();
        c.reserve_seq(1, 3).unwrap();
        for layer in [0usize, 1] {
            c.note_written(0, layer, &[0, 1]);
        }
        // Shrinking below the rows a sequence has written would lose them.
        let err = c.set_cap(0, 1).unwrap_err();
        assert!(err.contains("cannot shrink"), "got: {err}");
        // Growing past what the arena holds with the other reservations is refused
        // *before* the table changes, so there is nothing to undo.
        let err = c.set_cap(0, 6).unwrap_err();
        assert!(err.contains("reserved elsewhere"), "got: {err}");
        assert_eq!(c.seq_slot(0).unwrap().cap, 3, "the refusal left it alone");
        // What does fit (3 + 5 = 8) is allowed, and it moves the run above up.
        let plan = c.set_cap(0, 5).unwrap();
        assert_eq!(plan.len(), 1, "run 1 moves: {plan:?}");
        assert_eq!((plan[0].from, plan[0].to), (3, 5));
        assert_eq!(c.seq_slot(0).unwrap().cap, 5);
    }

    /// A window the resolver cannot justify must be loud, not a wrong bound.
    #[test]
    fn unwritten_rows_and_unknown_sequences_are_errors() {
        // No reservation for the sequence at all.
        let mut c = cache(8);
        let err = c.attn_span(&[1], &[0]).unwrap_err();
        assert!(err.contains("holds no cells"), "got: {err}");

        // A query outside its reserved run, and a row inside the run that was
        // never written (it would attend to zeroes).
        let mut c = cache(8);
        c.reserve_seq(SEQ_MAIN, 2).unwrap();
        let err = c.attn_span(&[SEQ_MAIN], &[5]).unwrap_err();
        assert!(err.contains("outside its reserved run"), "got: {err}");
        let err = c.attn_span(&[SEQ_MAIN], &[1]).unwrap_err();
        assert!(err.contains("not written"), "got: {err}");
        // Once written, the same query resolves.
        for layer in [0usize, 1] {
            c.note_written(SEQ_MAIN, layer, &[0, 1]);
        }
        assert_eq!(c.attn_span(&[SEQ_MAIN], &[1]).unwrap(), vec![0, 2]);
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

    /// Reserve `seq` a `cap`-cell run and take ownership of all of it, so the
    /// run has a live prefix a compaction must carry.
    fn place(c: &mut KvCache, seq: SeqId, cap: usize) -> usize {
        let slot = c.reserve_seq(seq, cap).unwrap();
        c.own_range(seq, slot.start, slot.start + cap);
        slot.start
    }

    #[test]
    fn a_fresh_arena_is_one_free_run() {
        let c = cache(16);
        let st = c.arena_stats();
        assert_eq!(
            (st.free_cells, st.free_runs, st.largest_free_run),
            (16, 1, 16)
        );
        assert_eq!((st.reserved_cells, st.owned_cells, st.sequences), (0, 0, 0));
    }

    #[test]
    fn fragmentation_refuses_a_run_the_arena_could_hold() {
        let mut c = cache(16);
        // A[0,4) B[4,4) C[8,4) D[12,4), all written.
        for seq in 1u32..=4 {
            place(&mut c, seq, 4);
        }
        // Free A and C: two 4-cell runs, 8 free cells, none of them 8 long.
        c.release_seq(1);
        c.release_seq(3);
        let st = c.arena_stats();
        assert_eq!(
            (st.free_cells, st.free_runs, st.largest_free_run),
            (8, 2, 4)
        );
        let err = c.reserve_seq(5, 8).unwrap_err();
        assert!(err.contains("no free run of 8 cells"), "{err}");
        // This is the failure C3 exists for: capacity is not the constraint.
        assert!(st.free_cells >= 8);
    }

    #[test]
    fn the_plan_stops_as_soon_as_it_opens_the_run_it_needs() {
        let mut c = cache(16);
        for seq in 1u32..=4 {
            place(&mut c, seq, 4);
        }
        c.release_seq(1);
        c.release_seq(3);
        // Moving B down to 0 opens [4, 12): enough for 8 cells, so D stays put.
        assert_eq!(
            c.compaction_plan(Some(8)),
            vec![KvMove {
                seq: 2,
                from: 4,
                to: 0,
                rows: 4
            }]
        );
        // Without `need`, the whole live set is packed.
        assert_eq!(
            c.compaction_plan(None),
            vec![
                KvMove {
                    seq: 2,
                    from: 4,
                    to: 0,
                    rows: 4
                },
                KvMove {
                    seq: 4,
                    from: 12,
                    to: 4,
                    rows: 4
                },
            ]
        );
    }

    #[test]
    fn applying_a_plan_opens_the_run_and_keeps_every_sequences_rows() {
        let mut c = cache(16);
        for seq in 1u32..=4 {
            place(&mut c, seq, 4);
        }
        c.release_seq(1);
        c.release_seq(3);
        let before = c.arena_stats();
        // The full plan packs both live runs, so the free tail is the whole top.
        let plan = c.compaction_plan(None);
        assert_eq!(c.apply_moves(&plan).unwrap(), 8, "4 written rows per run");
        let after = c.arena_stats();
        assert_eq!(after.largest_free_run, 8, "{after:?}");
        assert_eq!(after.free_runs, 1);
        assert_eq!(
            (after.defrags, after.cells_moved),
            (1, 16),
            "8 rows x 2 layers"
        );
        assert!(before.largest_free_run < 8);
        // The reservation that first-fit refused now fits, in the opened tail.
        assert_eq!(c.reserve_seq(5, 8).unwrap().start, 8);
        // Ownership travelled with each sequence; the vacated cells are FREE.
        let owner = &c.get(0).unwrap().owner;
        assert_eq!(owner[0..4], [2, 2, 2, 2]);
        assert_eq!(owner[4..8], [4, 4, 4, 4]);
        assert!(owner[8..16].iter().all(|&o| o == FREE));
        assert_eq!(c.seq_slot(4).unwrap().start, 4);
        assert_eq!(
            c.get(0).unwrap().n_used,
            8,
            "n_used must follow the rows down, not stay at the old top"
        );
        // The data the caller will copy is the sequence's live prefix.
        assert_eq!(c.written_rows(4), 4);
    }

    #[test]
    fn an_unwritten_run_moves_its_reservation_without_copying_rows() {
        let mut c = cache(16);
        c.reserve_seq(1, 4).unwrap();
        c.reserve_seq(2, 4).unwrap();
        c.release_seq(1);
        let plan = c.compaction_plan(None);
        assert_eq!(
            plan,
            vec![KvMove {
                seq: 2,
                from: 4,
                to: 0,
                rows: 0
            }]
        );
        assert_eq!(c.apply_moves(&plan).unwrap(), 0);
        assert_eq!(c.seq_slot(2).unwrap().start, 0);
        assert_eq!(c.arena_stats().cells_moved, 0);
        assert_eq!(c.get(0).unwrap().n_used, 0);
    }

    #[test]
    fn a_plan_that_would_overlap_a_live_run_is_refused() {
        let mut c = cache(16);
        place(&mut c, 1, 4); // [0, 4)
        place(&mut c, 2, 4); // [4, 8)
        let bad = vec![KvMove {
            seq: 2,
            from: 4,
            to: 2,
            rows: 4,
        }];
        let err = c.apply_moves(&bad).unwrap_err();
        assert!(err.contains("would overlap"), "{err}");
        // Nothing moved: the run table and the counters are untouched.
        assert_eq!(c.seq_slot(2).unwrap().start, 4);
        assert_eq!(c.arena_stats().defrags, 0);
        // A stale `from` and an upward move are refused too.
        let stale = vec![KvMove {
            seq: 2,
            from: 0,
            to: 0,
            rows: 0,
        }];
        assert!(c.apply_moves(&stale).unwrap_err().contains("the plan says"));
        // C7b: an upward move is legal now — what a plan may never do is land two
        // runs on the same cells, in either direction.
        let up_overlap = vec![
            KvMove {
                seq: 1,
                from: 0,
                to: 8,
                rows: 0,
            },
            KvMove {
                seq: 2,
                from: 4,
                to: 8,
                rows: 0,
            },
        ];
        assert!(c.apply_moves(&up_overlap).unwrap_err().contains("overlap"));
        // Rows past the run's cap are a bug, not a bigger copy.
        let too_many = vec![KvMove {
            seq: 2,
            from: 4,
            to: 0,
            rows: 5,
        }];
        assert!(c.apply_moves(&too_many).unwrap_err().contains("rows"));
    }
}
