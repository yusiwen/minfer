// device-memory accounting (E4) (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

impl CudaState {
    /// Bytes of device-resident weights this state holds (E4: the feasibility gate
    /// charges the budget for them, so "weights + activations" is one comparison).
    /// Sums the registry; the padded-plane companions are accounted by their own
    /// registers and are deliberately not double-counted here.
    ///
    /// A poisoned lock is recovered rather than answered with `0`: this registry is
    /// append-only, so the map behind a poison is still valid, and reporting 0 would
    /// silently *under*-charge the budget by every resident weight (issue #122's
    /// fail-open twin).
    pub fn weights_bytes(&self) -> usize {
        crate::graph::alloc::weights_from_lock(self.weights.lock(), "CUDA weight registry", |w| {
            w.values().map(|(_, size)| *size).sum()
        })
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
