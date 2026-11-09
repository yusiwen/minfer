//! `#[cfg(test)] mod tests` for `src/server/slot.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
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
