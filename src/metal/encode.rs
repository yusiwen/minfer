// Metal backend L2 submission (pre-split `src/metal.rs`).
//
// Moved verbatim by the #265 layout split: ending the compute pass, encoding the
// P2/P3 capture blits, and `submit`'s bounded wait + status check.

use super::*;
use block2::RcBlock;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBufferStatus, MTLCommandEncoder, MTLEvent};
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

    /// C3 (issue #44 part (b)): encode the overlap-safe row move
    /// [`Backend::copy_cells`](crate::graph::backend::Backend::copy_cells) needs —
    /// one `MTLBlitCommandEncoder` copy per row, in the order the overlap
    /// requires (ascending when the run slides down, descending when it slides
    /// up), into **this** command buffer, so the move rides the split's single
    /// submission.
    ///
    /// Apple documents a same-buffer copy whose source and destination ranges
    /// overlap as undefined, so a bulk blit is not an option; a per-row blit
    /// satisfies the `size <= distance` rule for every `src_row != dst_row` (a
    /// same-row move is a no-op and returns early). This mirrors CUDA's
    /// `kv_move_rows` (`cuda/kernels/kv_store.cu`), which branches on
    /// `dst_row <= src_row` the same way.
    ///
    /// `row_bytes` is one cell's storage stride (`elems_per_cell * 4`, already
    /// halved for an f16 region by the caller), and `src_base`/`dst_base` are the
    /// two views' byte offsets (`BufRef::offset * 4`).
    pub fn encode_move_rows(
        &mut self,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        src_base: usize,
        dst_base: usize,
        src_row: usize,
        dst_row: usize,
        rows: usize,
        row_bytes: usize,
    ) -> Result<(), String> {
        if rows == 0 || src_row == dst_row {
            return Ok(());
        }
        let src_end = src_base
            .checked_add(
                (src_row + rows)
                    .checked_mul(row_bytes)
                    .ok_or("row move overflow")?,
            )
            .ok_or("row move source offset overflow")?;
        if src_end > src.length() {
            return Err(format!(
                "Metal copy_cells: source rows {src_row}..{} at byte {src_base} run past the \
                 {} -byte buffer",
                src_row + rows,
                src.length()
            ));
        }
        let dst_end = dst_base
            .checked_add(
                (dst_row + rows)
                    .checked_mul(row_bytes)
                    .ok_or("row move overflow")?,
            )
            .ok_or("row move destination offset overflow")?;
        if dst_end > dst.length() {
            return Err(format!(
                "Metal copy_cells: destination rows {dst_row}..{} at byte {dst_base} run past the \
                 {} -byte buffer",
                dst_row + rows,
                dst.length()
            ));
        }
        // Close the compute pass first (Metal allows one active encoder).
        self.end_compute();
        let blit = self
            .cmd_buf
            .blitCommandEncoder()
            .ok_or("MTLCommandBuffer.blitCommandEncoder returned nil")?;
        let down = dst_row <= src_row;
        for k in 0..rows {
            // Moving down: copy the lowest row first. Moving up: the highest
            // first. Either way a row is never overwritten before it is read.
            let r = if down { k } else { rows - 1 - k };
            let so = src_base + (src_row + r) * row_bytes;
            let d_off = dst_base + (dst_row + r) * row_bytes;
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src, so, dst, d_off, row_bytes,
                );
            }
        }
        blit.endEncoding();
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

    /// A retained handle to the underlying `MTLCommandBuffer`, so a caller that
    /// hands this buffer to async work can read its real `status()` / `error()`
    /// later. Cheap: it is an Objective-C retain.
    pub fn command_buffer(&self) -> MetalCommandBuffer {
        self.cmd_buf.clone()
    }

    /// F5 (#137): **phase A** of a Metal cross-backend staging copy — encode a
    /// `MTLBlitCommandEncoder` copy of the source window into the staging buffer
    /// and record a signal on the shared event, **into this command buffer**.
    ///
    /// The blit goes into the **split's own** command buffer (the one
    /// `MetalBackend::cb` already opened for the producing split), which is the
    /// repo's "one Metal command buffer per split" rule; it is not a second
    /// submission. That is also a correctness requirement, not just tidiness:
    /// the source is a `StorageModeShared` pool buffer the split's kernels may
    /// still be writing, and a separate command buffer that overlapped them
    /// changed the kernel results (the M4 Pro measurement is in
    /// the F5 S3 record in `docs/ARCHITECTURE-EXECUTION-PLAN.md`). Encoding the
    /// blit after `end_compute` puts it behind those kernels in the *same*
    /// submission, so the copy sees exactly the split's final bytes.
    ///
    /// `bytes` is the node's **logical** window (`BufRef::len`), not the pool
    /// buffer's class-rounded length: the destination staging buffer is allocated
    /// exactly, so copying the padding would overflow it (and would copy another
    /// node's recycled bytes). `src_offset` is the view offset (`BufRef::offset`),
    /// in bytes.
    ///
    /// `signal` is `false` only under the `MINFER_TEST_CALL_FAIL=metal_cross_copy`
    /// injection, which suppresses the signal so that phase B's **bounded** wait
    /// takes its real timeout branch (see `MetalBackend::cross_take`).
    ///
    /// The caller commits (or, when this is a split's buffer, lets `submit`
    /// commit) and retains the buffer if it wants to read its status.
    pub fn encode_blit_signal(
        &mut self,
        src: &MetalBuffer,
        src_offset: usize,
        dst: &MetalBuffer,
        bytes: usize,
        event: &MetalSharedEvent,
        value: u64,
        signal: bool,
    ) -> Result<(), String> {
        if bytes == 0 {
            return Err("Metal cross-backend staging blit of 0 bytes".to_string());
        }
        if src_offset + bytes > src.length() {
            return Err(format!(
                "Metal cross-backend staging blit runs past the source buffer: \
                 offset {src_offset} + {bytes} bytes > {} bytes",
                src.length()
            ));
        }
        if bytes > dst.length() {
            return Err(format!(
                "Metal cross-backend staging blit runs past the staging buffer: \
                 {bytes} bytes > {} bytes",
                dst.length()
            ));
        }
        // Close the compute pass first (Metal allows one active encoder). If a
        // capture blit pass already closed it, this is a no-op.
        self.end_compute();
        let blit = self
            .cmd_buf
            .blitCommandEncoder()
            .ok_or("MTLCommandBuffer.blitCommandEncoder returned nil")?;
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                src, src_offset, dst, 0, bytes,
            );
        }
        blit.endEncoding();
        if signal {
            // `MTLSharedEvent` refines `MTLEvent`; `ProtocolObject::from_ref` is
            // the objc2 upcast to the super-protocol the command buffer asks for.
            let as_event = ProtocolObject::<dyn MTLEvent>::from_ref(&**event);
            self.cmd_buf.encodeSignalEvent_value(as_event, value);
        }
        Ok(())
    }
}
