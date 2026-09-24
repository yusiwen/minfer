//! KV cache storage format and its `MINFER_CACHE_TYPE` gate (Phase C / ticket C4).
//!
//! Three formats can reach a KV region:
//!
//! - **F32** — the CPU default and the reference every tolerance is stated against;
//! - **F16** — the GPU bandwidth policy (`cuda::set_kv_cache_type` /
//!   `metal::set_kv_cache_type`, auto-selected for the 7B class). It keeps the region
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
//! Design record: `docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 (C4).

use std::sync::atomic::{AtomicU8, Ordering};

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
    /// Decode the atomic's value (see [`kv_format`]).
    pub fn from_code(code: u8) -> KvFormat {
        match code {
            1 => KvFormat::F16,
            2 => KvFormat::Q8_0,
            _ => KvFormat::F32,
        }
    }

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
    /// and accumulates V out of the cell; CUDA and Metal address f32/f16 rows, so
    /// they refuse it until their kernels land (issue [#87]).
    ///
    /// [#87]: https://github.com/yusiwen/minfer/issues/87
    pub fn supports(self, device: Device) -> bool {
        match self {
            KvFormat::Q8_0 => super::registry::reads_packed_kv(device.backend()),
            KvFormat::F32 | KvFormat::F16 => true,
        }
    }
}

/// `MINFER_CACHE_TYPE` → the format to use on `device`.
///
/// Pure, so the whole matrix is covered by CI (which has no GPU). The rules:
///
/// - unset/empty → F32 (the GPU's own auto policy — f16 for the 7B class — is applied
///   separately by `set_kv_cache_type`, and it does not change the region's shape);
/// - `f32`, `f16`, `q8_0` → that format;
/// - `f16` on CPU → F32: the CPU has no f16 KV kernel and
///   `docs/BACKENDS.md` documents "CPU: f32 regions", so an env var set for a GPU run
///   must not break a CPU one;
/// - anything else → **refused on every device** (a typo must not silently run f32);
/// - a format the device has no kernel for → **refused** (`q8_0` off CPU).
pub fn resolve(device: Device, cache_type: Option<&str>) -> Result<KvFormat, String> {
    let format = match cache_type {
        None | Some("") => KvFormat::F32,
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
            "MINFER_CACHE_TYPE={} is not supported on {} yet: the {} attention kernel does not \
             read a packed Q8_0 region (the CPU one does); refusing rather than silently \
             falling back to f32",
            format.name(),
            device.name(),
            device.name()
        ));
    }
    Ok(format)
}

/// The process-wide format the graph builder and the CPU backend read.
///
/// Like `cuda::kv_cache_is_f16`, this is a **per-load policy**: a process that loads a
/// second model must be able to change it, so it deliberately overwrites.
static KV_FORMAT: AtomicU8 = AtomicU8::new(KvFormat::F32 as u8);

/// Set the process-wide KV format (called once per model load, before the first
/// forward — see `models::qwen2::loader`).
pub fn set_kv_format(format: KvFormat) {
    KV_FORMAT.store(format as u8, Ordering::Relaxed);
}

/// The process-wide KV format (F32 until a load decides otherwise).
pub fn kv_format() -> KvFormat {
    KvFormat::from_code(KV_FORMAT.load(Ordering::Relaxed))
}

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
pub fn pack_q8_0_cell(dst: &mut [f32], nkt: usize, src: &[f32]) {
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
mod tests {
    use super::*;

    #[test]
    fn a_cell_is_a_whole_number_of_words() {
        // 128 elements -> 4 blocks -> 136 payload bytes -> 34 words (136 = 4*34).
        assert_eq!(KvFormat::Q8_0.row_elems(128), 34);
        // 64 -> 2 blocks -> 68 bytes -> 17 words.
        assert_eq!(KvFormat::Q8_0.row_elems(64), 17);
        // 96 -> 3 blocks -> 102 bytes -> 26 words (104 bytes, 2 bytes of padding).
        assert_eq!(KvFormat::Q8_0.row_elems(96), 26);
        assert_eq!(KvFormat::Q8_0.row_bytes(96), 104);
        for nkt in [32usize, 64, 96, 128, 256, 1024] {
            let words = KvFormat::Q8_0.row_elems(nkt);
            assert!(words * WORD_BYTES >= KvFormat::Q8_0.payload_bytes(nkt));
            assert!(words * WORD_BYTES < KvFormat::Q8_0.payload_bytes(nkt) + WORD_BYTES);
        }
        // F32 and F16 keep one word per element: the region stays f32-shaped.
        for f in [KvFormat::F32, KvFormat::F16] {
            assert_eq!(f.row_elems(128), 128);
            assert!(!f.is_packed());
        }
    }

    #[test]
    fn a_packed_region_is_at_least_three_times_smaller_than_f32() {
        // 1024 elements/cell, one f32 word each vs 32 blocks * 34 bytes = 1088 B.
        let f32_bytes = KvFormat::F32.row_bytes(1024);
        let q8_bytes = KvFormat::Q8_0.row_bytes(1024);
        assert_eq!(f32_bytes, 4096);
        assert_eq!(q8_bytes, 1088);
        assert!(
            q8_bytes * 3 <= f32_bytes,
            "Q8_0 must be at least 3x smaller than f32 ({q8_bytes} vs {f32_bytes})"
        );
    }

    #[test]
    fn a_width_the_format_cannot_express_is_refused() {
        assert!(KvFormat::Q8_0.check_width(96).is_ok());
        let err = KvFormat::Q8_0.check_width(80).unwrap_err();
        assert!(err.contains("multiple of 32"), "{err}");
        assert!(KvFormat::Q8_0.check_width(0).is_err());
        // F32/F16 have no such constraint.
        assert!(KvFormat::F32.check_width(80).is_ok());
        assert!(KvFormat::F16.check_width(1).is_ok());
    }

    #[test]
    fn the_cache_type_gate_is_strict_and_device_aware() {
        // Unset: f32 everywhere (the GPU's own auto policy is applied separately).
        assert_eq!(resolve(Device::Cpu, None).unwrap(), KvFormat::F32);
        assert_eq!(resolve(Device::Cuda, None).unwrap(), KvFormat::F32);
        assert_eq!(resolve(Device::Metal, None).unwrap(), KvFormat::F32);
        assert_eq!(resolve(Device::Cpu, Some("")).unwrap(), KvFormat::F32);
        // The three spellings.
        assert_eq!(resolve(Device::Cpu, Some("f32")).unwrap(), KvFormat::F32);
        assert_eq!(resolve(Device::Cpu, Some("q8_0")).unwrap(), KvFormat::Q8_0);
        assert_eq!(resolve(Device::Cuda, Some("f16")).unwrap(), KvFormat::F16);
        assert_eq!(resolve(Device::Metal, Some("f16")).unwrap(), KvFormat::F16);
        // f16 on CPU is the documented "CPU stays f32", not an error: the env var is
        // usually set for the GPU run a process may also do.
        assert_eq!(resolve(Device::Cpu, Some("f16")).unwrap(), KvFormat::F32);
        // A packed format off a CPU is refused loudly, never silently mapped.
        for dev in [Device::Cuda, Device::Metal] {
            let err = resolve(dev, Some("q8_0")).unwrap_err();
            assert!(err.contains("q8_0"), "{err}");
            assert!(err.contains(dev.name()), "{err}");
        }
        // A typo is refused on every device (CUDA used to read it as f32).
        for dev in [Device::Cpu, Device::Cuda, Device::Metal] {
            let err = resolve(dev, Some("q8")).unwrap_err();
            assert!(err.contains("not a KV cache type"), "{err}");
        }
        assert!(resolve(Device::Cpu, Some("Q8_0")).is_err(), "case matters");
    }

    #[test]
    fn the_process_wide_format_can_be_redecided() {
        let before = kv_format();
        set_kv_format(KvFormat::Q8_0);
        assert_eq!(kv_format(), KvFormat::Q8_0);
        set_kv_format(before);
        assert_eq!(kv_format(), before);
    }

    #[test]
    fn a_packed_cell_round_trips_within_the_q8_0_block_error() {
        let nkt = 128usize;
        let src: Vec<f32> = (0..nkt)
            .map(|i| ((i as f32) * 0.37).sin() * 2.5 + (i % 7) as f32 * 0.11)
            .collect();
        let mut cell = vec![0.0f32; KvFormat::Q8_0.row_elems(nkt)];
        pack_q8_0_cell(&mut cell, nkt, &src);
        let mut back = vec![0.0f32; nkt];
        unpack_q8_0_cells(&cell, nkt, 0, 1, &mut back);
        // Q8_0 keeps a per-block scale, so the error is half a step of that block's
        // own range (`|x - d*q| <= d/2`) in exact arithmetic, plus the stored scale's
        // own rounding amplified by the quantized magnitude: the scale is an f16
        // (half an ulp is `2^-12 * d`), and `|q| <= 127`, so the second term is at
        // most `127 * 2^-12 * d ~ 0.031 d`. 0.55 d covers both and still fails loudly
        // on a wrong layout, which is off by whole steps.
        for i in 0..nkt {
            let d = block_step(&src[(i / Q8_0_BLOCK) * Q8_0_BLOCK..]);
            let bound = d * 0.55 + 1e-5;
            assert!(
                (src[i] - back[i]).abs() <= bound,
                "element {i}: {} vs {} (bound {bound})",
                src[i],
                back[i]
            );
        }
    }

    /// The step (`d`) Q8_0 gives the block `block` starts.
    fn block_step(block: &[f32]) -> f32 {
        let am = block[..Q8_0_BLOCK]
            .iter()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        am / 127.0
    }
}
