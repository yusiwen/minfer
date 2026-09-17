//! Batch composition (Phase E / E2).
//!
//! A `Batch` is one forward's worth of *data*: every query token with the
//! sequence it belongs to and the KV position it writes. Nothing here is part
//! of the graph's identity — `GraphParams` carries `n_tokens` and the
//! *topology* flags, never the sequence count (E2 deleted it; see
//! `params.rs`'s module docs) — so the params-only reuse rule is untouched: two
//! batches of the same shape share one graph and refill the inputs, whether they
//! carry one sequence or several.
//!
//! **Contiguity.** A sequence's tokens must be contiguous in the batch. The CPU
//! attention path could tolerate interleaving (E1's span is per token), but CUDA
//! stages a query *tile* against one KV window (`fa_prefill_f16kv`, E1b), so an
//! interleaved batch would be wrong there and right on CPU — the kind of silent
//! divergence this project refuses. `Batch::check` therefore rejects it, and
//! E2's composition (one decode token per sequence, in slot order) satisfies it
//! by construction.

use super::kvcache::{SeqId, SEQ_MAIN};

/// One forward: `tokens[t]` at `positions[t]`, belonging to `seq_ids[t]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub seq_ids: Vec<SeqId>,
}

impl Batch {
    /// A single-sequence batch (the classic forward path).
    pub fn single(tokens: &[u32], positions: &[usize]) -> Self {
        Self {
            tokens: tokens.to_vec(),
            positions: positions.to_vec(),
            seq_ids: vec![SEQ_MAIN; tokens.len()],
        }
    }

    /// A batch from explicit parts, validated by [`Batch::check`].
    pub fn new(tokens: Vec<u32>, positions: Vec<usize>, seq_ids: Vec<SeqId>) -> Self {
        Self {
            tokens,
            positions,
            seq_ids,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Distinct sequences, in first-appearance order.
    pub fn seqs(&self) -> Vec<SeqId> {
        let mut out: Vec<SeqId> = Vec::new();
        for &s in &self.seq_ids {
            if !out.contains(&s) {
                out.push(s);
            }
        }
        out
    }

    pub fn n_seqs(&self) -> usize {
        self.seqs().len()
    }

    /// Contiguous `(seq, from, to)` runs, in batch order.
    pub fn groups(&self) -> Vec<(SeqId, usize, usize)> {
        let mut out: Vec<(SeqId, usize, usize)> = Vec::new();
        for (t, &s) in self.seq_ids.iter().enumerate() {
            match out.last_mut() {
                Some((last, _, to)) if *last == s => *to = t + 1,
                _ => out.push((s, t, t + 1)),
            }
        }
        out
    }

    /// The rows whose logits a caller wants.
    ///
    /// One sequence: the last `n_out` rows (llama's `inp_out_ids`). Several:
    /// the last row of each sequence, one logits row per sequence in batch
    /// order — a decode step wants every sequence's next-token distribution.
    pub fn out_rows(&self, n_out: usize) -> Vec<u32> {
        if self.n_seqs() <= 1 {
            let to = self.len();
            let from = to.saturating_sub(n_out);
            return (from..to).map(|x| x as u32).collect();
        }
        self.groups()
            .iter()
            .map(|&(_, _, to)| (to - 1) as u32)
            .collect()
    }

    /// Reject a batch the backends cannot execute identically.
    pub fn check(&self) -> Result<(), String> {
        let nt = self.tokens.len();
        if self.positions.len() != nt || self.seq_ids.len() != nt {
            return Err(format!(
                "batch: {} tokens, {} positions, {} sequence ids",
                nt,
                self.positions.len(),
                self.seq_ids.len()
            ));
        }
        if nt == 0 {
            return Err("batch: empty".to_string());
        }
        // A sequence's tokens must be contiguous: see the module docs.
        let groups = self.groups();
        if groups.len() != self.n_seqs() {
            return Err(format!(
                "batch: sequence {:?} appears in more than one run (tokens of a sequence must be \
                 contiguous — CUDA's attention tiles one window per sequence)",
                groups.iter().map(|g| g.0).collect::<Vec<_>>()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_is_one_group_on_the_main_sequence() {
        let b = Batch::single(&[1, 2, 3], &[0, 1, 2]);
        assert_eq!(b.n_seqs(), 1);
        assert_eq!(b.groups(), vec![(SEQ_MAIN, 0, 3)]);
        assert!(b.check().is_ok());
        // One sequence: `n_out` selects the tail rows, as before E2.
        assert_eq!(b.out_rows(1), vec![2]);
        assert_eq!(b.out_rows(3), vec![0, 1, 2]);
    }

    #[test]
    fn several_sequences_yield_one_logits_row_each() {
        // Two sequences decoding one token each, in slot order.
        let b = Batch::new(vec![10, 20], vec![3, 7], vec![5, 9]);
        assert_eq!(b.n_seqs(), 2);
        assert_eq!(b.groups(), vec![(5, 0, 1), (9, 1, 2)]);
        assert!(b.check().is_ok());
        assert_eq!(b.out_rows(1), vec![0, 1], "n_out is ignored: one row each");
    }

    #[test]
    fn interleaved_sequences_are_refused() {
        let b = Batch::new(vec![1, 2, 3], vec![0, 3, 1], vec![5, 9, 5]);
        let err = b.check().unwrap_err();
        assert!(err.contains("contiguous"), "got: {err}");
    }

    #[test]
    fn shape_mismatches_are_refused() {
        let b = Batch::new(vec![1, 2], vec![0], vec![5, 5]);
        assert!(b.check().unwrap_err().contains("2 tokens"));
        assert!(Batch::new(vec![], vec![], vec![]).check().is_err());
    }
}
