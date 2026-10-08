//! KV cache storage format and its `MINFER_CACHE_TYPE` gate (Phase C / ticket C4).
//!
//! Three formats can reach a KV region:
//!
//! - **F32** — the CPU default and the reference every tolerance is stated against;
//! - **F16** — the GPU bandwidth policy ([`auto_device_format`], auto-selected
//!   for the 7B class). It keeps the region
//!   *f32-shaped* and stores halves in the first half of those words, so
//!   [`KvFormat::row_elems`] is the same as F32's: the win is bandwidth, not footprint;
//! - **Q8_0** — the first **packed** format. A cell occupies
//!   `ceil(n_kv_embd / 32 * 34)` bytes instead of `4 * n_kv_embd`, so it is the one
//!   that reduces the cache's footprint.
//!
//! Packed rows are padded up to a whole number of f32 words on purpose: the pool is
//! word-addressed, and the C3/C8b machinery (`copy_cells`, the copy-on-write shift, the
//! compaction) moves one cell at a time and must keep moving it verbatim.
//!
//! **How a packed region is read (C4 S2).** The CPU attention kernel reads the blocks
//! directly: the K score is a `dot_q8_0_q8_0` against the Q8_0-quantized query row, and
//! V accumulates out of the cell block by block ([`accumulate_q8_0_row`]) — S1's
//! dequantize-into-a-scratch pass is gone (`MINFER_NO_FUSED_Q8_KV` restores it for the
//! A/B standing rule 3 asks for). A physical shift (`kv_rm`/`kv_shift`) moves the
//! survivors verbatim and then maps each one through
//! [`map_q8_0_cells`]: dequantize → re-rope → requantize.
//!
//! **Where a format is supported is a loud decision, never a fallback.** An unknown
//! `MINFER_CACHE_TYPE` value is refused on every device (CUDA used to map anything that
//! was not `f16` to f32), and a format the device has no kernel for is refused at load.
//!
//! **The format is per engine, not a process global (issues #99, #153).** `models::load_model`
//! resolves `MINFER_CACHE_TYPE` once, against the device the loaded model will actually
//! use, and stores the answer on the model; `CParams::kv_format` carries it into the
//! graph builder (which stamps each KV node's width) and into the allocator (whose CPU
//! **and CUDA** backends speak it). There is deliberately **no** `kvformat::set_kv_format`
//! / `kv_format` global any more: when there was one, the C4 packed-cache gate flipped it
//! for its measurement runs and every other test building a graph in the same process
//! sized its regions for the wrong format — the parallel-red gate set this ticket fixed.
//!
//! The device half landed in [#99]'s follow-up [#153]: the CUDA kernels' `KV_LAYOUT_*`
//! tag is no longer a `cuda::KV_LAYOUT` process global. The engine's resolved format
//! reaches its `CudaBackend` through the same `GraphAllocator::set_kv_format` stamp the
//! CPU kernels get, the graph builder's `CParams::kv_format` still sizes the regions, and
//! the captured-graph identity records the layout, so an exec instantiated for one layout
//! never replays for another. Metal joined them in #44 part (b): `MetalBackend` carries its own
//! `kv_format` and its kernels take it as an explicit `f16` argument — there is no device-static
//! left.
//!
//! Design record: `docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 (C4) and its #99 / #153
//! records.
//!
//! [#99]: https://github.com/yusiwen/minfer/issues/99
//! [#153]: https://github.com/yusiwen/minfer/issues/153

use crate::models::Device;

/// Elements per Q8_0 block (`block::BlockQ8_0`: one f16 scale plus 32 int8 quants).
pub const Q8_0_BLOCK: usize = 32;
/// Bytes per Q8_0 block.
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Bytes per f32 pool word.
pub const WORD_BYTES: usize = 4;

/// KV cache storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum KvFormat {
    /// One f32 per element (CPU default, the tolerance reference).
    #[default]
    F32 = 0,
    /// One f16 per element, stored in the first half of the f32-shaped region.
    F16 = 1,
    /// Packed Q8_0 blocks, one cell rounded up to a whole number of f32 words.
    Q8_0 = 2,
}

impl KvFormat {
    /// The `MINFER_CACHE_TYPE` spelling.
    pub fn name(self) -> &'static str {
        match self {
            KvFormat::F32 => "f32",
            KvFormat::F16 => "f16",
            KvFormat::Q8_0 => "q8_0",
        }
    }

    /// Whether a cell is stored packed (fewer bytes than one f32 per element).
    pub fn is_packed(self) -> bool {
        matches!(self, KvFormat::Q8_0)
    }

    /// f32 words one cell occupies in the allocator's pool.
    ///
    /// F32 and F16 occupy one word per element; Q8_0 occupies
    /// `ceil(blocks * 34 / 4)` words — rounded up so every cell starts on a word
    /// boundary, which is what lets a cell move be a plain `copy_within`.
    pub fn row_elems(self, nkt: usize) -> usize {
        match self {
            KvFormat::F32 | KvFormat::F16 => nkt,
            KvFormat::Q8_0 => {
                let bytes = (nkt / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
                bytes.div_ceil(WORD_BYTES)
            }
        }
    }

    /// Bytes one cell occupies.
    ///
    /// CUDA's packed-KV path and this module's own tests are its callers.
    #[cfg(any(feature = "cuda", test))]
    pub fn row_bytes(self, nkt: usize) -> usize {
        self.row_elems(nkt) * WORD_BYTES
    }

    /// Bytes one cell's *packed payload* uses (before word padding).
    pub fn payload_bytes(self, nkt: usize) -> usize {
        match self {
            KvFormat::F32 => nkt * WORD_BYTES,
            KvFormat::F16 => nkt * 2,
            KvFormat::Q8_0 => (nkt / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES,
        }
    }

    /// Refuse a row width this format cannot express: Q8_0 quantizes in whole
    /// 32-element blocks, so a row shorter than one block or not a multiple of it
    /// has no layout — silently truncating would mis-address every following cell.
    pub fn check_width(self, nkt: usize) -> Result<(), String> {
        if self.is_packed() && (nkt == 0 || nkt % Q8_0_BLOCK != 0) {
            return Err(format!(
                "KV cache type {} needs n_kv_embd to be a non-zero multiple of {Q8_0_BLOCK} \
                 (got {nkt}): a Q8_0 cell is a whole number of 32-element blocks",
                self.name()
            ));
        }
        Ok(())
    }

    /// Whether this device has kernels that read a region in this format.
    ///
    /// F4: the packed answer is the backend registry's
    /// `BackendCaps::reads_packed_kv` — the same field `GraphAllocator::ensure_kv`
    /// reads, so the format gate and the region-sizing gate cannot disagree. The
    /// CPU's attention kernel dots the stored K blocks against the quantized query
    /// and accumulates V out of the cell; CUDA's kernels are layout-tagged
    /// (`KV_LAYOUT_F32/F16/Q8_0` plus the byte-addressed `kv4<LAYOUT>` load) since
    /// C4 S2b, so it reads a packed region too; and since [#310] Metal's packed
    /// store plus its `kernel_gqa_attn_q8_0` / window / map kernels read one as
    /// well.
    ///
    /// [#87]: https://github.com/yusiwen/minfer/issues/87
    /// [#44]: https://github.com/yusiwen/minfer/issues/44
    /// [#310]: https://github.com/yusiwen/minfer/issues/310
    pub fn supports(self, device: Device) -> bool {
        match self {
            KvFormat::Q8_0 => super::registry::reads_packed_kv(device.backend()),
            KvFormat::F32 | KvFormat::F16 => true,
        }
    }
}

/// The KV element count above which an unset `MINFER_CACHE_TYPE` auto-selects F16 on a
/// device (the "7B class": KV-bandwidth-bound decode). The CPU keeps F32 regardless.
pub const AUTO_F16_MIN_KV_ELEMS: usize = 8192;

/// The GPU's own auto policy for an **unset** `MINFER_CACHE_TYPE`: f16 for the 7B class
/// (`n_layers × n_kv_embd >= AUTO_F16_MIN_KV_ELEMS` — KV-bandwidth-bound decode),
/// f32 for small models (0.5B measured f16 ~3% *slower* — dispatch-latency-bound).
///
/// This used to live only in `cuda::set_kv_cache_type` / `metal::set_kv_cache_type`,
/// which wrote a process-wide tag; #153 folded it into the engine's resolved format so
/// the value that sizes the region and the value the kernels read cannot differ, and so
/// two engines with different dims can pick different layouts in one process. The CPU is
/// never auto-f16: `resolve` maps F16 back to F32 there.
pub fn auto_device_format(device: Device, n_layers: usize, n_kv_embd: usize) -> KvFormat {
    if device != Device::Cpu && n_layers * n_kv_embd >= AUTO_F16_MIN_KV_ELEMS {
        KvFormat::F16
    } else {
        KvFormat::F32
    }
}

/// `MINFER_CACHE_TYPE` → the format to use on `device`, for a model with the given dims.
///
/// Pure, so the whole matrix is covered by CI (which has no GPU). The rules:
///
/// - unset/empty → the device's own auto policy ([`auto_device_format`]: f16 for the 7B
///   class on a device, f32 on the CPU and for small models);
/// - `f32`, `f16`, `q8_0` → that format;
/// - `f16` on CPU → F32: the CPU has no f16 KV kernel and
///   `docs/BACKENDS.md` documents "CPU: f32 regions", so an env var set for a GPU run
///   must not break a CPU one;
/// - anything else → **refused on every device** (a typo must not silently run f32);
/// - a format the device has no kernel for → **refused** (the CPU, CUDA and Metal
///   kernels all read a packed region since C4 S2a / S2b and [#310]).
pub fn resolve(
    device: Device,
    cache_type: Option<&str>,
    n_layers: usize,
    n_kv_embd: usize,
) -> Result<KvFormat, String> {
    let format = match cache_type {
        None | Some("") => auto_device_format(device, n_layers, n_kv_embd),
        Some("f32") => KvFormat::F32,
        Some("f16") => KvFormat::F16,
        Some("q8_0") => KvFormat::Q8_0,
        Some(other) => {
            return Err(format!(
                "MINFER_CACHE_TYPE={other} is not a KV cache type (f32, f16, q8_0); refusing \
                 rather than silently running with f32"
            ));
        }
    };
    let format = if format == KvFormat::F16 && device == Device::Cpu {
        KvFormat::F32
    } else {
        format
    };
    if !format.supports(device) {
        return Err(format!(
            "MINFER_CACHE_TYPE={} is not supported on {} yet: the {} attention kernel has no \
             packed Q8_0 read (the CPU's, CUDA's and Metal's do); refusing rather than silently \
             falling back to f32",
            format.name(),
            device.name(),
            device.name()
        ));
    }
    Ok(format)
}

// The process-wide `KV_FORMAT` static, `set_kv_format` and `kv_format` used to live
// here. They were removed by the #99 per-engine change: the loaded model now owns its
// format (`ModelDef::kv_format`), `CParams::kv_format` carries it into the builder,
// and `GraphAllocator::set_kv_format` gives the CPU kernels the same answer. A process
// global let one test's packed measurement size another test's regions.

// ---- packed-row accessors --------------------------------------------------
//
// A packed region is still a `&[f32]` in the allocator's pool, so these convert
// between its words and Q8_0 bytes through the words' bit patterns. No `unsafe`:
// the padding rule (`row_elems * 4 >= payload bytes`) is what makes the copy exact.

/// Copy whole words out of a pool slice as bytes (`out.len()` must be `src.len()*4`).
fn words_to_bytes(src: &[f32], out: &mut [u8]) {
    debug_assert_eq!(out.len(), src.len() * WORD_BYTES);
    for (i, w) in src.iter().enumerate() {
        out[i * WORD_BYTES..(i + 1) * WORD_BYTES].copy_from_slice(&w.to_bits().to_le_bytes());
    }
}

/// Copy bytes into pool words, zero-filling the tail of the last word.
fn bytes_to_words(src: &[u8], out: &mut [f32]) {
    debug_assert!(out.len() * WORD_BYTES >= src.len());
    out.fill(0.0);
    for (i, w) in out.iter_mut().enumerate() {
        let mut b = [0u8; WORD_BYTES];
        let lo = i * WORD_BYTES;
        if lo >= src.len() {
            break;
        }
        let n = WORD_BYTES.min(src.len() - lo);
        b[..n].copy_from_slice(&src[lo..lo + n]);
        *w = f32::from_bits(u32::from_le_bytes(b));
    }
}

/// Quantize one f32 cell row (`src.len() == nkt`) into `dst`, one packed cell.
/// Test-only (#238): driven by `graph::kvformat::tests::a_packed_cell_round_trips_within_the_q8_0_block_error` and `graph::cpu_backend::tests::the_fused_q8_read_matches_the_dequantizing_reference`; `#[cfg(test)]` keeps it out of production builds.
#[cfg(test)]
pub(crate) fn pack_q8_0_cell(dst: &mut [f32], nkt: usize, src: &[f32]) {
    let mut raw = vec![0u8; KvFormat::Q8_0.payload_bytes(nkt)];
    pack_q8_0_cell_into(dst, nkt, src, &mut raw);
}

/// [`pack_q8_0_cell`] with a caller-owned scratch, so a store of `nt` cells allocates
/// once instead of once per row (a 2048-token prefill packs ~98k cells across 24
/// layers — the per-row `vec!` was measurable in the packed prefill path).
pub fn pack_q8_0_cell_into(dst: &mut [f32], nkt: usize, src: &[f32], scratch: &mut [u8]) {
    debug_assert_eq!(dst.len(), KvFormat::Q8_0.row_elems(nkt));
    let payload = KvFormat::Q8_0.payload_bytes(nkt);
    debug_assert!(scratch.len() >= payload);
    crate::quants::quantize_row_q8_0_into(src, &mut scratch[..payload]);
    bytes_to_words(&scratch[..payload], dst);
}

/// Dequantize `rows` packed cells of `region`, starting at cell `first`, into `out`
/// (`rows * nkt` f32 values).
pub fn unpack_q8_0_cells(region: &[f32], nkt: usize, first: usize, rows: usize, out: &mut [f32]) {
    let row_elems = KvFormat::Q8_0.row_elems(nkt);
    let payload = KvFormat::Q8_0.payload_bytes(nkt);
    debug_assert!(out.len() >= rows * nkt);
    let mut raw = vec![0u8; row_elems * WORD_BYTES];
    for r in 0..rows {
        let cell = first + r;
        words_to_bytes(&region[cell * row_elems..(cell + 1) * row_elems], &mut raw);
        crate::quants::dequantize_row_q8_0(&raw[..payload], &mut out[r * nkt..(r + 1) * nkt]);
    }
}

// ---- the fused read path (C4 S2) -------------------------------------------
//
// S1 dequantized a query window into a reusable f32 scratch and then ran the
// unchanged f32 attention kernel over it. S2 reads the packed bytes directly:
// the K side is a `dot_q8_0_q8_0` against the Q8_0-quantized query row, and the V
// side accumulates block by block as it reads. Both drop the scratch's write +
// read pass, which is the point of a packed cache on a bandwidth-bound device.

/// One packed region's bytes: a `&[f32]` reborrowed as `&[u8]`.
///
/// The pool stores a packed cell in f32 words, so the bytes are the words'
/// little-endian bit patterns. `f32` is 4-byte aligned (≥ `u8`'s 1), so the
/// reborrow is always aligned, and the result borrows `region`.
pub fn region_bytes(region: &[f32]) -> &[u8] {
    // SAFETY: `[f32]` is 4-byte aligned, every byte of it is initialized, and the
    // returned slice shares `region`'s lifetime.
    unsafe { std::slice::from_raw_parts(region.as_ptr() as *const u8, region.len() * WORD_BYTES) }
}

/// Byte offset of element `first_elem` inside cell `cell` of a packed region.
///
/// `first_elem` must be block-aligned — the fused read passes a KV head's base,
/// which is why it requires `hd % 32 == 0` (the store's `check_width` already
/// requires `nkt % 32 == 0`, and every supported architecture has `hd` a multiple
/// of 64).
#[inline]
pub fn q8_0_cell_offset(nkt: usize, cell: usize, first_elem: usize) -> usize {
    debug_assert_eq!(first_elem % Q8_0_BLOCK, 0);
    cell * KvFormat::Q8_0.row_elems(nkt) * WORD_BYTES + (first_elem / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES
}

/// `out[i] += weight * dequant(cell)[first_elem + i]` over packed Q8_0 blocks.
///
/// The V side of the fused read: one pass over the cell's bytes, no f32 scratch.
/// `out.len()` must be a non-zero multiple of [`Q8_0_BLOCK`].
pub fn accumulate_q8_0_row(
    region: &[f32],
    nkt: usize,
    cell: usize,
    first_elem: usize,
    weight: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(first_elem % Q8_0_BLOCK, 0);
    debug_assert!(out.len() % Q8_0_BLOCK == 0);
    let bytes = region_bytes(region);
    let base = q8_0_cell_offset(nkt, cell, first_elem);
    for (b, chunk) in out.chunks_exact_mut(Q8_0_BLOCK).enumerate() {
        let blk = &bytes[base + b * Q8_0_BLOCK_BYTES..base + (b + 1) * Q8_0_BLOCK_BYTES];
        let d = crate::block::fp16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) * weight;
        for (i, o) in chunk.iter_mut().enumerate() {
            *o += d * blk[2 + i] as i8 as f32;
        }
    }
}

/// Map one packed cell through `f`: dequantize → `f` → requantize, in place.
///
/// This is what makes a physical shift (`KvCache::kv_rm` / `kv_shift`, C2)
/// expressible on a packed region: the surviving cells move **verbatim** (they are
/// whole words), and then their K row is dequantized into `row`, re-roped in f32 by
/// `f`, and quantized back. `row` is caller-owned (`nkt` f32) so a shift over the
/// whole arena allocates once, not once per row.
///
/// The cost is one extra quantize per surviving row, on an operation that already
/// touches every row once per overflow.
pub fn map_q8_0_cells(
    region: &mut [f32],
    nkt: usize,
    first: usize,
    rows: usize,
    row: &mut [f32],
    mut f: impl FnMut(&mut [f32]),
) {
    let row_elems = KvFormat::Q8_0.row_elems(nkt);
    let payload = KvFormat::Q8_0.payload_bytes(nkt);
    debug_assert!(row.len() >= nkt);
    let mut raw = vec![0u8; row_elems * WORD_BYTES];
    for r in 0..rows {
        let cell = first + r;
        let words = &mut region[cell * row_elems..(cell + 1) * row_elems];
        words_to_bytes(words, &mut raw);
        crate::quants::dequantize_row_q8_0(&raw[..payload], &mut row[..nkt]);
        f(&mut row[..nkt]);
        crate::quants::quantize_row_q8_0_into(&row[..nkt], &mut raw[..payload]);
        bytes_to_words(&raw[..payload], words);
    }
}

#[cfg(test)]
mod tests;
