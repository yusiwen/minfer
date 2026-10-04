//! `#[cfg(test)] mod tests` for `src/graph/kvcache.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::Backend;

mod cells;
mod defrag;
mod resize;
mod sharing;
mod spans;
fn buf(id: usize) -> BufRef {
    BufRef::own(Backend::CPU, id, usize::MAX)
}
fn cache(n_ctx: usize) -> KvCache {
    let mut c = KvCache::new();
    c.insert(0, buf(1), buf(2), 4, 4, n_ctx, false);
    c.insert(1, buf(3), buf(4), 4, 4, n_ctx, false);
    c
}
/// Reserve `seq` a `cap`-cell run and take ownership of all of it, so the
/// run has a live prefix a compaction must carry.
fn place(c: &mut KvCache, seq: SeqId, cap: usize) -> usize {
    let slot = c.reserve_seq(seq, cap).unwrap();
    c.own_range(seq, slot.start, slot.start + cap);
    slot.start
}
// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `kvcache.rs` (bucket B of the dead-code census —
// every test caller already lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl KvCache {
    /// The single-sequence case: reserve the whole arena for `seq` if it has no
    /// reservation yet, then take ownership of `0..n_used`.
    ///
    /// **Test-only (#239).** Its last production-looking caller, `GraphAllocator::kv_note_used`,
    /// was E1's C1 remnant and was deleted with E1 (#228): production now records the
    /// written extent per position through [`Self::own_positions`], which is
    /// reservation-aware. Driven by `kvcache::tests::own_prefix_and_release_round_trip`
    /// and its three siblings (`after_shift_renumbers_and_frees_the_tail`,
    /// `after_rm_removes_a_middle_range_and_frees_the_tail`,
    /// `a_single_sequence_resolves_to_the_causal_window`), which use it to set up a
    /// written prefix directly.
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
    ///
    /// Test-only (#239): driven by `kvcache::tests::ownership_follows_the_written_cells`
    /// and the C2/C8b gates in this file (19 tests in all).
    ///
    /// Note: its `#[allow(dead_code)]` comment claimed an "E2 surface"; the E2 batched path
    /// records written extents through `own_positions` / `fill_batch_inputs` instead, so that
    /// surface never arrived. Recorded for [#244](https://github.com/yusiwen/minfer/issues/244).
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

    /// Sliding-window special case of [`KvCache::after_rm`]: drop the oldest
    /// `drop` rows, so every survivor's position decreases by `drop`.
    ///
    /// Test-only (#239): driven by `kvcache::tests::after_shift_renumbers_and_frees_the_tail`
    /// and `kvcache::tests::a_shared_prefix_stops_a_physical_removal`.
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
    ///
    /// Test-only (#239): driven by `kvcache::tests::two_sequences_resolve_to_disjoint_windows`
    /// and `kvcache::tests::reservations_are_disjoint_first_fit_and_releasable`.
    pub fn seq_range(&self, layer: usize, seq: SeqId) -> Result<Option<(usize, usize)>, String> {
        if !self.layers.contains_key(&layer) {
            return Err(format!("no KV arena for layer {layer}"));
        }
        Ok(self.seqs.get(&seq).map(|s| (s.start, s.cap)))
    }

    /// Rows `seq` wrote into its **own** run — the rows a relocation copies, since
    /// the shared prefix is another sequence's memory (C8b S2).
    ///
    /// Test-only (#239): driven by `kvcache::tests::a_shared_prefix_is_read_in_place_and_written_past`,
    /// `a_store_inside_a_shared_prefix_takes_a_private_row` and
    /// `a_copy_on_write_can_shrink_the_share_twice`.
    pub fn private_written(&self, seq: SeqId) -> usize {
        self.seqs.get(&seq).map_or(0, |s| s.private_written())
    }
}
