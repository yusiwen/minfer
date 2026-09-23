//! Size classes, the allocation plan and memory accounting (Phase E / ticket E4).
//!
//! The allocator used to do one thing at a time: walk the graph in build order and ask a
//! backend's pool for a buffer of exactly the size the node needs. Three consequences
//! (roadmap §2.3): a workload touching *k* activation shapes ends up with *k* sets of live
//! buffers, because an exact match never shares; nothing can answer "will this fit?"
//! before the device is touched (CUDA surfaced it as a null pointer at execute time); and
//! peak memory was not a number anyone could read.
//!
//! This module is the **reserve** half — a *pure* plan over `(size, first use, last use)`
//! intervals that assigns each activation a size class, recycles a class buffer once its
//! interval ends, and reports what the pool will have to hold. The **assign** half is the
//! allocator's pool loop, which now asks for the class size; the plan is checked against a
//! per-backend budget *before* that loop runs, so an over-budget graph is refused with its
//! numbers instead of failing later.
//!
//! The class is a pure function of the size ([`class_size`]), which is what lets the plan
//! and the pool agree without a lookup table between them: the plan's aggregate numbers
//! (reserved bytes, live peak, how many buffers get recycled) are simulated here, and the
//! pool simply rounds every request the same way.

/// Smallest class, in f32 elements (1 KiB).
pub const CLASS_MIN: usize = 256;
/// Granularity above [`CLASS_MIN`], in f32 elements (16 KiB): a class never wastes more
/// than one step, and two shapes that differ by less than a step share a buffer.
pub const CLASS_GRAIN: usize = 4096;

/// The size class `elems` is rounded up to.
///
/// Below the grain the ladder is powers of two (the many small activations of a decode
/// step are cheap to share and expensive to round coarsely); above it, multiples of the
/// grain, so a 1.8 M-element prefill buffer wastes at most 16 KiB instead of up to 2x.
pub fn class_size(elems: usize) -> usize {
    let elems = elems.max(1);
    if elems <= CLASS_MIN {
        return CLASS_MIN;
    }
    if elems <= CLASS_GRAIN {
        return elems.next_power_of_two();
    }
    elems.div_ceil(CLASS_GRAIN) * CLASS_GRAIN
}

/// Bytes a class occupies (`4` bytes per f32 element).
pub fn class_bytes(elems: usize) -> usize {
    elems * std::mem::size_of::<f32>()
}

/// The plan for one backend's activations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AllocPlan {
    /// Per interval, in the order given: the class it will be handed (elements).
    pub classes: Vec<usize>,
    /// Bytes the pool must be able to hold once every interval has been placed — the
    /// number the feasibility gate compares against the budget. The pool is a high-water
    /// mark (it never returns memory), so this is the sum of every distinct slot, not the
    /// transient peak.
    pub reserved_bytes: usize,
    /// Peak bytes *live at one step* (what a shrinkable allocator would need).
    pub live_peak_bytes: usize,
    /// Distinct buffers the plan placed.
    pub buffers: usize,
    /// Intervals that reused a recycled buffer instead of a new one — the observable
    /// that size classes are actually sharing.
    pub reused: usize,
}

impl AllocPlan {
    /// Plan `intervals`, each `(size_elems, first_use, last_use)` with `first <= last`.
    ///
    /// Intervals are consumed in order of `first_use` (ties keep the input order, so the
    /// plan is deterministic); a buffer of class `c` freed at step `f` may be handed to
    /// the next interval of that class whose `first_use > f`. That is exactly the rule
    /// the pool's free list implements, which is why the plan's numbers are the pool's.
    pub fn plan(intervals: &[(usize, usize, usize)]) -> AllocPlan {
        let mut order: Vec<usize> = (0..intervals.len()).collect();
        order.sort_by_key(|&i| (intervals[i].1, i));
        // class -> the steps at which its placed buffers become free (unsorted; the sets
        // are tiny compared with the graph).
        let mut free: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        let mut classes = vec![0usize; intervals.len()];
        let mut reserved = 0usize;
        let mut live = 0usize;
        let mut peak = 0usize;
        let mut buffers = 0usize;
        let mut reused = 0usize;
        for i in order {
            let (size, first, last) = intervals[i];
            let class = class_size(size);
            let slot = free.get_mut(&class).and_then(|f| {
                f.iter()
                    .position(|&free_at| free_at < first)
                    .map(|pos| f.swap_remove(pos))
            });
            match slot {
                Some(free_at) => {
                    reused += 1;
                    let _ = free_at;
                }
                None => {
                    buffers += 1;
                    reserved += class_bytes(class);
                    live += class_bytes(class);
                    peak = peak.max(live);
                }
            }
            free.entry(class).or_default().push(last);
            classes[i] = class;
        }
        // The transient peak needs the free-at steps, not just the count: recompute over
        // the same placement in step order.
        peak = peak.max(live_peak(intervals, &classes));
        AllocPlan {
            classes,
            reserved_bytes: reserved,
            live_peak_bytes: peak,
            buffers,
            reused,
        }
    }

    /// Class sizes, in interval order.
    pub fn class_of(&self, interval: usize) -> usize {
        self.classes[interval]
    }
}

/// The live-bytes peak of a placement: walk the steps, adding each interval's class when
/// its lifetime starts and removing it when it ends.
fn live_peak(intervals: &[(usize, usize, usize)], classes: &[usize]) -> usize {
    if intervals.is_empty() {
        return 0;
    }
    let last_step = intervals.iter().map(|i| i.2).max().unwrap_or(0);
    let mut live = 0usize;
    let mut peak = 0usize;
    for step in 0..=last_step {
        for (i, &(_, first, last)) in intervals.iter().enumerate() {
            if first == step {
                live += class_bytes(classes[i]);
            }
            if last == step {
                live = live.saturating_sub(class_bytes(classes[i]));
            }
        }
        peak = peak.max(live);
    }
    peak
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ladder_rounds_up_without_wasting_a_step() {
        // Below the grain: powers of two, with a 1 KiB floor.
        assert_eq!(class_size(0), CLASS_MIN);
        assert_eq!(class_size(1), CLASS_MIN);
        assert_eq!(class_size(CLASS_MIN), CLASS_MIN);
        assert_eq!(class_size(CLASS_MIN + 1), 512);
        assert_eq!(class_size(1000), 1024);
        assert_eq!(class_size(CLASS_GRAIN), CLASS_GRAIN);
        // Above it: multiples of the grain, so the waste is under one step.
        assert_eq!(class_size(CLASS_GRAIN + 1), CLASS_GRAIN * 2);
        assert_eq!(
            class_size(14336),
            16384,
            "896 x 16 is one step over 3 grains"
        );
        for elems in [1usize, 257, 4095, 4097, 200_000, 1_835_008, 4_194_305] {
            let c = class_size(elems);
            assert!(c >= elems, "{c} < {elems}");
            assert!(
                c - elems < CLASS_GRAIN,
                "{elems} -> {c} wastes {} elements",
                c - elems
            );
        }
    }

    #[test]
    fn two_shapes_in_one_class_share_a_buffer() {
        // 896 x 16 = 14336 and 896 x 17 = 15232 round to the same class, so the second
        // interval reuses the first one's buffer once it is free (the "no silent growth"
        // acceptance: the pool does not keep both).
        let a = 896 * 16;
        let b = 896 * 17;
        assert_eq!(class_size(a), class_size(b));
        let plan = AllocPlan::plan(&[(a, 0, 1), (b, 2, 3)]);
        assert_eq!(plan.buffers, 1, "{plan:?}");
        assert_eq!(plan.reused, 1);
        assert_eq!(plan.reserved_bytes, class_bytes(class_size(a)));
        // Overlapping lifetimes cannot share, so both are reserved.
        let plan = AllocPlan::plan(&[(a, 0, 5), (b, 2, 3)]);
        assert_eq!(plan.buffers, 2, "{plan:?}");
        assert_eq!(plan.reused, 0);
        assert_eq!(plan.reserved_bytes, 2 * class_bytes(class_size(a)));
    }

    #[test]
    fn the_plan_is_deterministic_and_order_independent() {
        let a = 4096;
        let b = 8192;
        let one = AllocPlan::plan(&[(a, 0, 2), (b, 1, 3), (a, 4, 5)]);
        let two = AllocPlan::plan(&[(a, 0, 2), (b, 1, 3), (a, 4, 5)]);
        assert_eq!(one, two);
        // Intervals arriving out of step order are still placed in step order, so the
        // plan does not depend on how the caller enumerated them.
        let shuffled = AllocPlan::plan(&[(a, 4, 5), (b, 1, 3), (a, 0, 2)]);
        assert_eq!(shuffled.buffers, one.buffers);
        assert_eq!(shuffled.reused, one.reused);
        assert_eq!(shuffled.reserved_bytes, one.reserved_bytes);
    }

    #[test]
    fn an_empty_plan_is_free() {
        let plan = AllocPlan::plan(&[]);
        assert_eq!(plan.buffers, 0);
        assert_eq!(plan.reserved_bytes, 0);
        assert_eq!(plan.live_peak_bytes, 0);
    }

    #[test]
    fn the_live_peak_is_not_the_reserved_total() {
        // Three intervals of the same class, one at a time: the pool holds one buffer,
        // and the live peak is that one buffer too.
        let plan = AllocPlan::plan(&[(4096, 0, 0), (4096, 1, 1), (4096, 2, 2)]);
        assert_eq!(plan.buffers, 1);
        assert_eq!(plan.live_peak_bytes, class_bytes(4096));
        assert_eq!(plan.reserved_bytes, class_bytes(4096));
        assert_eq!(plan.reused, 2);
        // Two different classes alive at once: the peak is their sum, and it is the same
        // as the reserved total (nothing to recycle yet).
        let plan = AllocPlan::plan(&[(4096, 0, 1), (8192, 0, 1)]);
        assert_eq!(plan.live_peak_bytes, plan.reserved_bytes);
    }
}
