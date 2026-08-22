//! Slot management: per-slot `GraphCache` (KV regions) + context budget.
//! OPENAI-CHAT-API-PLAN.md §Slot Management — each slot owns its own graph +
//! allocator + persistent KV regions; `n_ctx` is divided equally among slots.

use crate::graph::cache::GraphCache;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum SlotState {
    Idle,
    Processing,
}

/// One inference slot. Only ever touched by the serial worker thread, so no
/// locking is needed; the queue provides concurrency between requests.
pub struct Slot {
    pub id: usize,
    pub state: SlotState,
    pub cache: GraphCache,
    pub n_ctx_slot: usize,
}

/// Build `n_slots` slots dividing `n_ctx_total` equally
/// (n_ctx_slot = n_ctx_total / n_slots; llama.cpp `--parallel` semantics).
pub fn new_slots(n_slots: usize, n_ctx_total: usize) -> Vec<Slot> {
    let per = n_ctx_total / n_slots.max(1);
    (0..n_slots)
        .map(|id| Slot { id, state: SlotState::Idle, cache: GraphCache::new(), n_ctx_slot: per })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divides_context_equally() {
        let slots = new_slots(4, 4096);
        assert_eq!(slots.len(), 4);
        for s in &slots {
            assert_eq!(s.n_ctx_slot, 1024);
            assert_eq!(s.state, SlotState::Idle);
        }
    }

    #[test]
    fn uneven_division_floors() {
        let slots = new_slots(3, 100);
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].n_ctx_slot, 33);
    }

    #[test]
    fn zero_slots_yields_empty_pool() {
        assert!(new_slots(0, 4096).is_empty());
    }

    #[test]
    fn each_slot_has_own_graph_cache() {
        let mut slots = new_slots(2, 4096);
        let a = &mut slots[0].cache as *mut GraphCache;
        let b = &mut slots[1].cache as *mut GraphCache;
        assert_ne!(a, b, "caches must be distinct objects (isolated KV)");
    }
}
