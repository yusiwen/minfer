// device-memory accounting (E4) (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

impl CudaState {
    /// Bytes of device-resident weights this state holds (E4: the feasibility gate
    /// charges the budget for them, so "weights + activations" is one comparison).
    ///
    /// [#758] Sums the **ownership ledger**, not the name map. The name map holds
    /// only the current buffer per name; a replaced buffer stays resident until its
    /// model registration releases it (and, while a captured graph is live, until
    /// the last capture is dropped). Counting the ledger is what keeps this term
    /// equal to what `cudaMemGetInfo` reports as unavailable, so a reloading server
    /// cannot under-charge its budget by the replaced buffers.
    ///
    /// A poisoned lock is recovered rather than answered with `0`: the ledger is
    /// append-only, so the map behind a poison is still valid, and reporting 0
    /// would silently *under*-charge the budget by every resident weight (issue
    /// #122's fail-open twin).
    pub fn weights_bytes(&self) -> usize {
        crate::graph::alloc::weights_from_lock(
            self.model_allocs.lock(),
            "CUDA weight ledger",
            |ledger| ledger.values().map(|alloc| alloc.size).sum(),
        )
    }

    /// Device memory free/total in bytes, queried now (E4's default budget uses `free`).
    ///
    /// The `cudaMemGetInfo` return code is **not** discarded (issue #122). A pre-#122
    /// failure left `free` at 0 and was indistinguishable from "the device is full";
    /// the E4 feasibility gate then refused every later device allocation with
    /// "exceeds the 0 byte budget (0 MiB)" while the real cause — a sticky CUDA error
    /// such as `cudaErrorIllegalAddress` (700) — was thrown away.
    pub fn device_memory(&self) -> crate::graph::allocplan::DeviceMemory {
        let (mut free, mut total) = (0usize, 0usize);
        let rc = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if rc != 0 {
            return crate::graph::allocplan::DeviceMemory::QueryFailed {
                code: rc,
                name: cuda_error_name(rc).to_string(),
            };
        }
        crate::graph::allocplan::DeviceMemory::Reported { free, total }
    }
}
