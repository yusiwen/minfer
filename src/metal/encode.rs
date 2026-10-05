// Metal backend L2 submission (pre-split `src/metal.rs`).
//
// Moved verbatim by the #265 layout split: ending the compute pass, encoding the
// P2/P3 capture blits, and `submit`'s bounded wait + status check.

use super::*;
use block2::RcBlock;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBufferStatus, MTLCommandEncoder};
use std::ffi::c_void;

#[cfg(target_os = "macos")]
extern "C" {
    fn dispatch_semaphore_create(value: isize) -> *mut std::ffi::c_void;
    fn dispatch_semaphore_signal(sem: *mut std::ffi::c_void) -> isize;
    fn dispatch_semaphore_wait(sem: *mut std::ffi::c_void, timeout: u64) -> isize;
    fn dispatch_time(when: u64, delta: i64) -> u64;
    fn dispatch_release(obj: *mut std::ffi::c_void);
}

#[cfg(target_os = "macos")]
impl MpsCommandBuffer<'_> {
    /// End the compute pass. Call before encoding blits from the same command
    /// buffer (Metal allows only one active encoder at a time); `submit` then
    /// skips its own `end_encoding`.
    pub fn end_compute(&mut self) {
        if self.enc_open {
            self.enc.endEncoding();
            self.enc_open = false;
        }
    }

    /// Encode GPU→host staging copies (P2/P3 live/trace capture) into this
    /// command buffer, AFTER all kernels of the split. The destinations' data
    /// is valid once the command buffer is submitted (`synchronize`).
    pub fn encode_captures(
        &mut self,
        pairs: &[(usize, usize)],
        buffers: &[MetalBuffer],
        staging: &[MetalBuffer],
    ) -> Result<(), String> {
        if pairs.is_empty() {
            return Ok(());
        }
        self.end_compute();
        let blit_ref = self.cmd_buf.blitCommandEncoder().expect("blit encoder");
        for &(src, dst) in pairs {
            let src_buf = buffers
                .get(src)
                .ok_or_else(|| format!("capture: no buffer {src}"))?;
            let dst_buf = staging
                .get(dst)
                .ok_or_else(|| format!("capture: no staging {dst}"))?;
            let len = src_buf.length();
            unsafe {
                blit_ref.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &**src_buf, 0, &**dst_buf, 0, len,
                );
            }
        }
        blit_ref.endEncoding();
        Ok(())
    }

    pub fn submit(self) -> Result<(), String> {
        if self.enc_open {
            self.enc.endEncoding();
        }

        // dispatch_semaphore_t is already a reference-counted opaque pointer.
        let sem = unsafe { dispatch_semaphore_create(0) };
        let sem_val = sem as usize;

        let blk = RcBlock::new(
            move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| unsafe {
                dispatch_semaphore_signal(sem_val as *mut c_void);
            },
        );
        unsafe {
            self.cmd_buf.addCompletedHandler(RcBlock::into_raw(blk));
        }
        self.cmd_buf.commit();

        // Bounded wait (10 s). If the GPU hangs (hardware fault), the completion
        // handler never fires and we bail out instead of blocking forever.
        let timeout = unsafe { dispatch_time(0, 10_000_000_000i64) }; // 10 s from now
        let rc = unsafe { dispatch_semaphore_wait(sem, timeout) };
        unsafe {
            dispatch_release(sem);
        }

        if rc == 0 {
            // Command buffer finished (possibly with an error status).
            match self.cmd_buf.status() {
                MTLCommandBufferStatus::Completed => Ok(()),
                st => Err(format!(
                    "Metal command buffer status={st:?}. recent dispatches: {}",
                    self.recent_trace()
                )),
            }
        } else {
            // Timed out: the GPU did not complete the work.
            Err(format!(
                "Metal command buffer timed out after 10s (GPU hang). recent dispatches: {}",
                self.recent_trace()
            ))
        }
    }

    /// Join the recent dispatch trace into a printable string.
    fn recent_trace(&self) -> String {
        let t = self.state.dispatch_trace.lock().unwrap();
        t.iter().cloned().collect::<Vec<_>>().join(" -> ")
    }
}
