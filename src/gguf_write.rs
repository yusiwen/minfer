// GGUF v3 writer — the byte-level counterpart of the parser in `gguf.rs`.
//
// The parser (`GgufContext::init_from_data`) is the *contract*: this module
// emits exactly the bytes that parser reads back, and the unit tests here
// assert that contract directly (write → parse → compare). It is deliberately
// the same layout llama.cpp's `gguf_write_to_file` produces:
//
//   magic "GGUF" | version u32 | n_tensors i64 | n_kv i64
//   n_kv × KV pairs  (key string, type u32, [array elem type u32, count u64], payload)
//   n_tensors × tensor info (name string, n_dims u32, ne[0..n_dims] i64, type u32, offset u64)
//   zero padding to `general.alignment` (default 32)
//   tensor data, concatenated in tensor-info order, each padded to the alignment
//
// `offset` is relative to the start of the data section and the parser
// *requires* `info[i].offset == sum(ggml_pad(nbytes, alignment) for j < i)`
// (`gguf.rs`, the "compute total data section size" block). The writer
// computes exactly that.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::gguf::{
    ggml_pad, GgmlType, GgufKv, GgufType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION,
};

// === Metadata keys the writer itself owns (llama.cpp's LLM_KV_* spellings) ===

/// Per-part index, 0-based. The reader (`load_gguf_model`) reads `split.no`.
pub const KEY_SPLIT_NO: &str = "split.no";
/// Total number of parts. The reader reads `split.count` to decide "is this split?".
pub const KEY_SPLIT_COUNT: &str = "split.count";
/// Total tensor count across all parts (llama.cpp writes this too; the reader
/// does not consume it, but a tool that reads the file should see it).
pub const KEY_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";
/// The alignment key, read back into `GgufContext::alignment`.
pub const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

/// One tensor to write, described by the GGUF index fields the parser reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    /// GGUF element counts per dimension; `ne[0]` is the row length.
    pub ne: [i64; 4],
    pub type_: GgmlType,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, ne: [i64; 4], type_: GgmlType) -> Self {
        Self {
            name: name.into(),
            ne,
            type_,
        }
    }

    /// The payload size in bytes: whole quant blocks times the block size.
    /// Identical arithmetic to `GgufTensorInfo::nbytes` and `ggml_nbytes` for a
    /// contiguous tensor, so the writer and the parser cannot disagree.
    pub fn nbytes(&self) -> usize {
        let n: i64 = self.ne.iter().product();
        (n / self.type_.blck_size()) as usize * self.type_.type_size()
    }

    /// The parsed index form (offset filled in by the writer).
    pub fn to_info(&self, offset: u64) -> crate::gguf::GgufTensorInfo {
        let mut nb = [0usize; 4];
        let ts = self.type_.type_size();
        let bs = self.type_.blck_size() as usize;
        nb[0] = ts;
        nb[1] = nb[0] * (self.ne[0] as usize / bs);
        for j in 2..4 {
            nb[j] = nb[j - 1] * self.ne[j - 1] as usize;
        }
        crate::gguf::GgufTensorInfo {
            name: self.name.clone(),
            ne: self.ne,
            nb,
            type_: self.type_,
            offset,
        }
    }
}

/// Number of dimensions the GGUF index stores: trailing 1s are dropped, but at
/// least one dimension is always present (llama.cpp's `ggml_n_dims`).
pub fn ggml_n_dims(ne: &[i64; 4]) -> u32 {
    let mut n = 1u32;
    for j in 1..4 {
        if ne[j] != 1 {
            n = j as u32 + 1;
        }
    }
    n
}

// === KV encoding ===

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Encode one KV pair exactly as `GgufContext::init_from_reader` decodes it:
/// key string, type u32 (Array for an array), then — for an array — the element
/// type u32 and the element count u64, then the payload.
pub fn encode_kv(out: &mut Vec<u8>, kv: &GgufKv) {
    write_string(out, &kv.key);
    if kv.is_array {
        out.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        out.extend_from_slice(&(kv.type_ as u32).to_le_bytes());
        out.extend_from_slice(&(kv.get_ne() as u64).to_le_bytes());
        if kv.type_ == GgufType::String {
            for s in &kv.data_string {
                write_string(out, s);
            }
        } else {
            out.extend_from_slice(&kv.data);
        }
    } else {
        out.extend_from_slice(&(kv.type_ as u32).to_le_bytes());
        if kv.type_ == GgufType::String {
            write_string(out, &kv.data_string[0]);
        } else {
            out.extend_from_slice(&kv.data);
        }
    }
}

/// Replace an existing key's value in place (keeping its position, which is
/// what makes a rewritten file's metadata *equivalent* rather than reshuffled)
/// or append a new one.
pub fn kv_upsert(kv: &mut Vec<GgufKv>, new: GgufKv) {
    match kv.iter_mut().find(|k| k.key == new.key) {
        Some(slot) => *slot = new,
        None => kv.push(new),
    }
}

/// Drop a key if present. Used to clear `split.*` before writing a single file.
pub fn kv_remove(kv: &mut Vec<GgufKv>, key: &str) {
    kv.retain(|k| k.key != key);
}

// === The writer ===

/// Incremental GGUF v3 writer.
///
/// The header needs every tensor's size up front (the index carries offsets),
/// so the constructor takes the full spec list and writes a valid header; the
/// caller then hands the payloads in spec order, one `write_tensor` each. The
/// padding is written by the writer, so a caller never sees it.
pub struct GgufWriter {
    out: BufWriter<File>,
    specs: Vec<TensorSpec>,
    alignment: usize,
    next: usize,
}

impl GgufWriter {
    /// Create `path` and write magic + header + metadata + tensor index + the
    /// alignment padding that precedes the data section.
    pub fn create(
        path: &Path,
        kv: &[GgufKv],
        specs: Vec<TensorSpec>,
        alignment: usize,
    ) -> io::Result<Self> {
        validate_specs(&specs)?;
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("GGUF: alignment {alignment} is not a power of 2"),
            ));
        }
        let mut header: Vec<u8> = Vec::with_capacity(1 << 16);
        header.extend_from_slice(&GGUF_MAGIC);
        header.extend_from_slice(&(GGUF_VERSION as u32).to_le_bytes());
        header.extend_from_slice(&(specs.len() as i64).to_le_bytes());
        header.extend_from_slice(&(kv.len() as i64).to_le_bytes());
        for pair in kv {
            encode_kv(&mut header, pair);
        }

        // Offsets must be known before the index is written, so compute them first.
        let mut offsets = Vec::with_capacity(specs.len());
        let mut cursor: u64 = 0;
        for s in &specs {
            offsets.push(cursor);
            cursor += ggml_pad(s.nbytes(), alignment) as u64;
        }

        for (spec, off) in specs.iter().zip(offsets.iter()) {
            write_string(&mut header, &spec.name);
            let n_dims = ggml_n_dims(&spec.ne);
            header.extend_from_slice(&n_dims.to_le_bytes());
            for j in 0..n_dims as usize {
                header.extend_from_slice(&spec.ne[j].to_le_bytes());
            }
            header.extend_from_slice(&(spec.type_ as u32).to_le_bytes());
            header.extend_from_slice(&off.to_le_bytes());
        }

        // The parser seeks from `tell()` (right after the index) to the aligned
        // data offset, so the header must be padded, not the file position.
        let aligned = ggml_pad(header.len(), alignment);
        header.resize(aligned, 0);

        let file = File::create(path)?;
        let mut out = BufWriter::new(file);
        out.write_all(&header)?;
        Ok(Self {
            out,
            specs,
            alignment,
            next: 0,
        })
    }

    /// The specs this writer was created with (in write order).
    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }

    /// Flush and close. Fails when any declared tensor was never written.
    pub fn finish(mut self) -> io::Result<()> {
        if self.next != self.specs.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "GGUF writer: {} of {} tensors were written",
                    self.next,
                    self.specs.len()
                ),
            ));
        }
        self.out.flush()
    }
}

fn validate_specs(specs: &[TensorSpec]) -> io::Result<()> {
    for (i, s) in specs.iter().enumerate() {
        if s.name.len() >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "GGUF: tensor name '{}' is {} bytes, the maximum is 63",
                    s.name,
                    s.name.len()
                ),
            ));
        }
        let bs = s.type_.blck_size();
        if bs == 0 || s.ne[0] % bs != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "GGUF: tensor '{}' of type {} has {} elements per row, not a multiple of the \
                     block size ({})",
                    s.name,
                    s.type_.type_name(),
                    s.ne[0],
                    bs
                ),
            ));
        }
        for j in 1..4 {
            if s.ne[j] < 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "GGUF: tensor '{}' dimension {j} is {} (must be >= 1)",
                        s.name, s.ne[j]
                    ),
                ));
            }
        }
        for prev in &specs[..i] {
            if prev.name == s.name {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("GGUF: duplicate tensor name '{}'", s.name),
                ));
            }
        }
    }
    Ok(())
}

// === Whole-file helpers ===

/// Write one single-file GGUF. `data(i, w)` writes tensor `i`'s payload.
pub fn write_single<F>(
    path: &Path,
    kv: &[GgufKv],
    specs: Vec<TensorSpec>,
    alignment: usize,
    mut data: F,
) -> Result<(), String>
where
    F: FnMut(usize, &mut dyn Write) -> io::Result<()>,
{
    let mut kv = kv.to_vec();
    // A single file must not claim to be a split; leaving stale split keys in
    // would make `load_gguf_model` demand parts that do not exist.
    kv_remove(&mut kv, KEY_SPLIT_NO);
    kv_remove(&mut kv, KEY_SPLIT_COUNT);
    kv_remove(&mut kv, KEY_SPLIT_TENSORS_COUNT);
    let mut w = GgufWriter::create(path, &kv, specs, alignment).map_err(|e| e.to_string())?;
    for i in 0..w.specs().len() {
        let spec = w.specs()[i].clone();
        w.write_tensor_data(i, &spec, &mut data)?;
    }
    w.finish().map_err(|e| e.to_string())
}

impl GgufWriter {
    fn write_tensor_data<F>(
        &mut self,
        i: usize,
        spec: &TensorSpec,
        data: &mut F,
    ) -> Result<(), String>
    where
        F: FnMut(usize, &mut dyn Write) -> io::Result<()>,
    {
        // Stream straight into the file through a fixed-size buffer so a large
        // tensor never has to be materialized a second time.
        let mut staging = TensorStaging {
            out: &mut self.out,
            remaining: spec.nbytes(),
            overflow: 0,
        };
        data(i, &mut staging).map_err(|e| format!("GGUF: writing tensor '{}': {e}", spec.name))?;
        if staging.remaining != 0 || staging.overflow != 0 {
            return Err(format!(
                "GGUF: writing tensor '{}': provider wrote {} bytes, expected {}",
                spec.name,
                spec.nbytes() + staging.overflow - staging.remaining,
                spec.nbytes()
            ));
        }
        self.next += 1;
        let pad = ggml_pad(spec.nbytes(), self.alignment) - spec.nbytes();
        if pad > 0 {
            self.out
                .write_all(&vec![0u8; pad])
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// A `Write` view that refuses more than the declared payload (so a provider
/// bug cannot silently shift every later tensor's data).
struct TensorStaging<'a> {
    out: &'a mut BufWriter<File>,
    remaining: usize,
    overflow: usize,
}

impl Write for TensorStaging<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let take = buf.len().min(self.remaining);
        if take > 0 {
            self.out.write_all(&buf[..take])?;
            self.remaining -= take;
        }
        if take < buf.len() {
            self.overflow += buf.len() - take;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "GGUF tensor payload exceeds the declared byte size",
            ));
        }
        Ok(take)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

/// Greedily assign tensors to parts by their padded data size. A tensor is
/// never split across parts; a tensor larger than the cap is a loud refusal
/// (the request cannot be honoured), never a silently oversized part.
pub fn split_assignment(
    specs: &[TensorSpec],
    alignment: usize,
    max_part_bytes: u64,
) -> Result<Vec<Vec<usize>>, String> {
    if specs.is_empty() {
        return Err("GGUF split: cannot split a file with no tensors".to_string());
    }
    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_bytes: u64 = 0;
    for (i, s) in specs.iter().enumerate() {
        let padded = ggml_pad(s.nbytes(), alignment) as u64;
        if padded > max_part_bytes {
            return Err(format!(
                "GGUF split: tensor '{}' is {padded} bytes, larger than the {max_part_bytes}-byte \
                 part cap — a tensor cannot be split across parts; raise --split-max-size (at \
                 least {padded} bytes) or use a smaller-capable format",
                s.name
            ));
        }
        if !cur.is_empty() && cur_bytes + padded > max_part_bytes {
            parts.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
        cur.push(i);
        cur_bytes += padded;
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    Ok(parts)
}

/// Write `specs` as one or more parts under `dir`, named
/// `{stem}-NNNNN-of-MMMMM.gguf` when there is more than one.
///
/// Every part carries the full metadata plus `split.no` / `split.count` /
/// `split.tensors.count`, which is what the existing reader requires: part 0
/// is the entry point (its `split.count` drives `resolve_splits`), and each
/// part must parse standalone.
pub fn write_split<F>(
    dir: &Path,
    stem: &str,
    kv: &[GgufKv],
    specs: Vec<TensorSpec>,
    alignment: usize,
    max_part_bytes: u64,
    mut data: F,
) -> Result<Vec<PathBuf>, String>
where
    F: FnMut(usize, &mut dyn Write) -> io::Result<()>,
{
    if max_part_bytes == 0 {
        return Err("GGUF split: --split-max-size must be > 0".to_string());
    }
    let assignment = split_assignment(&specs, alignment, max_part_bytes)?;
    // One part is not a split: write a plain single file with no `split.*`
    // metadata, so a rewrite with a generous cap is metadata-equivalent to its
    // source and the loader sees an ordinary file.
    if assignment.len() == 1 {
        let path = dir.join(format!("{stem}.gguf"));
        write_single(&path, kv, specs, alignment, data)?;
        return Ok(vec![path]);
    }
    let total = assignment.len();
    let mut base_kv = kv.to_vec();
    kv_remove(&mut base_kv, KEY_SPLIT_NO);
    kv_remove(&mut base_kv, KEY_SPLIT_COUNT);
    kv_remove(&mut base_kv, KEY_SPLIT_TENSORS_COUNT);

    let mut written = Vec::with_capacity(total);
    for (p, ids) in assignment.iter().enumerate() {
        let mut part_kv = base_kv.clone();
        kv_upsert(
            &mut part_kv,
            GgufKv::new_u32(KEY_SPLIT_NO.to_string(), p as u32),
        );
        kv_upsert(
            &mut part_kv,
            GgufKv::new_u32(KEY_SPLIT_COUNT.to_string(), total as u32),
        );
        kv_upsert(
            &mut part_kv,
            GgufKv::new_u32(KEY_SPLIT_TENSORS_COUNT.to_string(), specs.len() as u32),
        );

        let path = if total == 1 {
            dir.join(format!("{stem}.gguf"))
        } else {
            dir.join(format!("{stem}-{:05}-of-{:05}.gguf", p + 1, total))
        };
        let part_specs: Vec<TensorSpec> = ids.iter().map(|&i| specs[i].clone()).collect();
        let mut w = GgufWriter::create(&path, &part_kv, part_specs, alignment)
            .map_err(|e| e.to_string())?;
        for (local, &global) in ids.iter().enumerate() {
            let spec = w.specs()[local].clone();
            w.write_tensor_data(global, &spec, &mut data)?;
        }
        w.finish().map_err(|e| e.to_string())?;
        written.push(path);
    }
    Ok(written)
}

/// The default alignment a converted file uses (llama.cpp's `GGUF_DEFAULT_ALIGNMENT`).
pub fn default_alignment() -> usize {
    GGUF_DEFAULT_ALIGNMENT
}

#[cfg(test)]
mod tests;
