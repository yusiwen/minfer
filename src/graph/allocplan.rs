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

// ─── The device's own memory answer, made explicit (E4 / issue #122) ─────────────────

/// What a backend's device answered when asked how much memory it has free.
///
/// The distinction is the point. "The device reported N bytes free" and "the query
/// failed" are different facts, and collapsing them into `free = 0` turns a broken
/// query into a zero-byte budget: the E4 feasibility gate then refuses every later
/// allocation and its message blames the budget, hiding the real CUDA error behind a
/// number that was never measured. The outcome is a type so that collapse cannot
/// happen again by accident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceMemory {
    /// The backend answered: `free` of `total` bytes.
    ///
    /// Only the CUDA query constructs these two variants today (`cuda.rs`) and
    /// the planner's own tests construct them. `allow` rather than `#[cfg]`
    /// because removing a variant changes the enum's shape and would force the
    /// same cfg onto every `match` arm here and in `offload.rs`.
    #[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
    Reported { free: usize, total: usize },
    /// The query itself failed. `code` is the backend's error code and `name` its
    /// symbolic name (for CUDA, `cudaGetErrorName`, e.g. `cudaErrorIllegalAddress`).
    #[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
    QueryFailed { code: i32, name: String },
    /// There is no device state to ask (CPU, or a device whose state could not be
    /// created).
    NoDevice,
}

impl DeviceMemory {
    /// The reported free bytes, or `None` when no measurement exists.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn free_bytes(&self) -> Option<usize> {
        match self {
            DeviceMemory::Reported { free, .. } => Some(*free),
            DeviceMemory::QueryFailed { .. } | DeviceMemory::NoDevice => None,
        }
    }

    /// A one-line reason for a failed query, for a diagnostic that must name the real
    /// cause. `None` when the query succeeded or there was no device to query.
    pub fn failure_note(&self, what: &str) -> Option<String> {
        match self {
            DeviceMemory::QueryFailed { code, name } => {
                Some(format!("{what} failed with {name} (code {code})"))
            }
            DeviceMemory::Reported { .. } | DeviceMemory::NoDevice => None,
        }
    }
}

/// The E4 activation budget decision for one backend.
///
/// `budget` is `None` for "unbounded" (CPU/Metal, or no device state) and `Some(n)`
/// for a number the gate may compare against. `note` carries the **reason** when the
/// number is not a measurement, so the caller can print why instead of letting a
/// fallback pass for one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetDecision {
    pub budget: Option<usize>,
    pub note: Option<String>,
}

/// Resolve a backend's activation budget from an explicit override and the device's
/// own memory answer. Pure, so the whole mapping is unit-tested with no device.
///
/// The rules, in order:
/// - an explicit `set_memory_budget` wins and is a measurement the caller chose; no note;
/// - a **reported** free read keeps the pre-existing default, three quarters of it
///   (unchanged on the happy path, including a genuine `free == 0`);
/// - a **failed** query is not a measurement: it falls back to weights-only accounting
///   (an unbounded activation budget) and carries a note naming the real error. Refusing
///   every allocation instead would turn a broken accounting query into a total outage
///   of a device that may be perfectly usable — and the backend's own allocation is the
///   authority on whether memory exists;
/// - no device state is unbounded and silent (a CUDA graph without device state is a
///   configuration error the assignment pass catches).
pub fn budget_decision(explicit: Option<usize>, mem: &DeviceMemory) -> BudgetDecision {
    if let Some(b) = explicit {
        return BudgetDecision {
            budget: Some(b),
            note: None,
        };
    }
    match mem {
        DeviceMemory::Reported { free, .. } => BudgetDecision {
            budget: Some(free / 4 * 3),
            note: None,
        },
        DeviceMemory::QueryFailed { code, name } => BudgetDecision {
            budget: Some(usize::MAX),
            note: Some(format!(
                "the device free-memory query failed with {name} (code {code}); the E4 \
                 activation gate has no measured budget for this backend, so it charges \
                 weights only. The backend's own allocation is now the authority and will \
                 report the real error if the context is unusable"
            )),
        },
        DeviceMemory::NoDevice => BudgetDecision {
            budget: None,
            note: None,
        },
    }
}

/// The plan for one backend's activations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(not(test), allow(dead_code))]
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
    #[cfg_attr(not(test), allow(dead_code))]
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
#[cfg_attr(not(test), allow(dead_code))]
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
mod tests;
