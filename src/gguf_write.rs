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
mod tests {
    use super::*;
    use crate::gguf::GgufContext;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("minfer-f6-writer-{name}"));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn sample_kv() -> Vec<GgufKv> {
        vec![
            GgufKv::new_string("general.architecture".into(), "qwen2".into()),
            GgufKv::new_u32("qwen2.block_count".into(), 3),
            GgufKv::new_i32("signed".into(), -7),
            GgufKv::new_f32("eps".into(), 1e-6),
            GgufKv::new_bool("flag".into(), true),
            GgufKv::new_u64("big".into(), 1 << 40),
            GgufKv::new_u16("small".into(), 7),
            GgufKv::new_string_array("tokens".into(), vec!["a".into(), "b".into()]),
            GgufKv::new_array(
                "scores".into(),
                GgufType::Float32,
                [1.0f32, 2.0, 3.0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            GgufKv::new_array(
                "types".into(),
                GgufType::Int32,
                (-2i32..2).flat_map(|v| v.to_le_bytes()).collect(),
            ),
        ]
    }

    #[test]
    fn every_metadata_type_round_trips_through_the_parser() {
        let path = tmp("kv.gguf");
        let specs = vec![TensorSpec::new("w", [4, 1, 1, 1], GgmlType::F32)];
        write_single(&path, &sample_kv(), specs, 32, |_, w| {
            w.write_all(&[0u8; 16])
        })
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let ctx = GgufContext::init_from_data(&bytes).unwrap();
        assert_eq!(
            ctx.get_key_val_str("general.architecture").unwrap(),
            "qwen2"
        );
        assert_eq!(ctx.get_key_val_i64("qwen2.block_count").unwrap(), 3);
        assert_eq!(ctx.get_key_val_i64("signed").unwrap(), -7);
        assert!((ctx.get_key_val_f32("eps").unwrap() - 1e-6).abs() < 1e-12);
        assert_eq!(ctx.get_key_val_i64("big").unwrap(), 1 << 40);
        assert_eq!(ctx.get_key_val_i64("small").unwrap(), 7);
        let k = ctx.find_key("tokens");
        assert_eq!(ctx.get_arr_n(k), 2);
        assert_eq!(ctx.get_arr_str(k, 1), "b");
        let s = ctx.find_key("scores");
        assert_eq!(ctx.get_arr_type(s), GgufType::Float32);
        assert_eq!(ctx.get_arr_data(s).len(), 12);
        let t = ctx.find_key("types");
        assert_eq!(ctx.get_arr_type(t), GgufType::Int32);
        assert_eq!(ctx.get_arr_n(t), 4);
        assert!(ctx.get_val_bool(ctx.find_key("flag")));
        // data section starts aligned and holds exactly the one tensor
        assert_eq!(ctx.get_data_offset() % 32, 0);
        assert_eq!(ctx.size, 32);
        assert_eq!(ctx.info[0].offset, 0);
        // The file must actually *contain* the padding the index declares.
        assert_eq!(bytes.len(), ctx.get_data_offset() + ctx.size);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tensor_index_offsets_match_the_parser_requirement() {
        let path = tmp("offsets.gguf");
        let specs = vec![
            TensorSpec::new("a", [32, 2, 1, 1], GgmlType::F32), // 256 B
            TensorSpec::new("b", [64, 1, 1, 1], GgmlType::Q4_0), // 36 B
            TensorSpec::new("c", [32, 1, 1, 1], GgmlType::F32), // 128 B
        ];
        let data = |i: usize, w: &mut dyn Write| w.write_all(&vec![i as u8; specs_len(i)]);
        write_single(&path, &sample_kv(), specs, 32, data).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let ctx = GgufContext::init_from_data(&bytes).unwrap();
        assert_eq!(ctx.info[0].offset, 0);
        assert_eq!(ctx.info[1].offset, 256);
        assert_eq!(ctx.info[2].offset, 320); // 256 + pad(36)=64
        assert_eq!(ctx.size, 320 + 128);
        // And the file must be exactly as long as that layout says. Without
        // this, a writer that declares the right offsets but writes a shorter
        // inter-tensor pad still parses (the parser never reads past the last
        // tensor), so the padding rule would be untested.
        assert_eq!(bytes.len(), ctx.get_data_offset() + ctx.size);
        let _ = std::fs::remove_file(&path);
    }

    fn specs_len(i: usize) -> usize {
        [256usize, 36, 128][i]
    }

    #[test]
    fn multiple_dimensions_and_trailing_ones_round_trip() {
        let path = tmp("dims.gguf");
        let specs = vec![
            TensorSpec::new("one_d", [8, 1, 1, 1], GgmlType::F32),
            TensorSpec::new("two_d", [4, 3, 1, 1], GgmlType::F32),
        ];
        write_single(&path, &sample_kv(), specs, 32, |i, w| {
            w.write_all(&vec![0u8; if i == 0 { 32 } else { 48 }])
        })
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let ctx = GgufContext::init_from_data(&bytes).unwrap();
        assert_eq!(ctx.info[0].ne, [8, 1, 1, 1]);
        assert_eq!(ctx.info[1].ne, [4, 3, 1, 1]);
        assert_eq!(ctx.info[1].nb[0], 4);
        assert_eq!(ctx.info[1].nb[1], 16);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn split_writes_parts_the_reader_merges_back() {
        let dir = std::env::temp_dir().join("minfer-f6-writer-split");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut kv = sample_kv();
        kv.push(GgufKv::new_string(
            "tokenizer.chat_template".into(),
            "{{ x }}".into(),
        ));
        // Three 256-byte tensors with a 300-byte cap → one part per tensor.
        let specs: Vec<TensorSpec> = (0..3)
            .map(|i| TensorSpec::new(format!("t{i}"), [64, 1, 1, 1], GgmlType::F32))
            .collect();
        let parts = write_split(&dir, "sample", &kv, specs, 32, 300, |i, w| {
            w.write_all(&vec![(i + 1) as u8; 256])
        })
        .unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(
            parts[0].file_name().unwrap().to_str().unwrap(),
            "sample-00001-of-00003.gguf"
        );
        let info = crate::gguf::split_file_info(parts[0].file_name().unwrap().to_str().unwrap())
            .expect("split filename parses");
        assert_eq!(info.1, 0);
        assert_eq!(info.2, 3);
        let model = crate::gguf::load_gguf_model(&parts[0]).expect("merged load");
        assert_eq!(model.parts.len(), 3);
        let mut names: Vec<String> = Vec::new();
        for (i, part) in model.parts.iter().enumerate() {
            assert_eq!(
                part.ctx.get_key_val_i64(KEY_SPLIT_NO).map(|v| v as usize),
                Some(i)
            );
            assert_eq!(part.ctx.get_key_val_i64(KEY_SPLIT_TENSORS_COUNT), Some(3));
            assert_eq!(
                part.ctx.get_key_val_str("general.architecture").unwrap(),
                "qwen2"
            );
            // Each part is a complete file whose length matches its own index.
            assert_eq!(part.data.len(), part.ctx.get_data_offset() + part.ctx.size);
            for ti in &part.ctx.info {
                names.push(ti.name.clone());
                let base = part.ctx.get_data_offset();
                assert_eq!(
                    part.data[base + ti.offset as usize],
                    (i + 1) as u8,
                    "part {i} tensor {} payload",
                    ti.name
                );
            }
        }
        assert_eq!(names, vec!["t0", "t1", "t2"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_part_is_a_plain_single_file_without_split_metadata() {
        let dir = std::env::temp_dir().join("minfer-f6-writer-onepart");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let specs = vec![TensorSpec::new("t0", [16, 1, 1, 1], GgmlType::F32)];
        let paths = write_split(&dir, "solo", &sample_kv(), specs, 32, u64::MAX, |_, w| {
            w.write_all(&[1u8; 64])
        })
        .unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].file_name().unwrap().to_str().unwrap(), "solo.gguf");
        let bytes = std::fs::read(&paths[0]).unwrap();
        let ctx = GgufContext::init_from_data(&bytes).unwrap();
        assert_eq!(ctx.find_key(KEY_SPLIT_COUNT), -1);
        assert_eq!(ctx.find_key(KEY_SPLIT_NO), -1);
        // and the metadata is the source's, key for key
        assert_eq!(ctx.kv.len(), sample_kv().len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tensor_larger_than_the_cap_is_refused() {
        let specs = vec![TensorSpec::new("big", [1024, 1, 1, 1], GgmlType::F32)];
        let err = split_assignment(&specs, 32, 100).unwrap_err();
        assert!(err.contains("larger than the 100-byte part cap"), "{err}");
        assert!(err.contains("'big'"), "{err}");
    }

    /// A valid 2-part split in a fresh directory, plus its entry path.
    fn two_part_fixture(name: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("minfer-f6-writer-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let specs: Vec<TensorSpec> = (0..2)
            .map(|i| TensorSpec::new(format!("t{i}"), [64, 1, 1, 1], GgmlType::F32))
            .collect();
        let parts = write_split(&dir, "m", &sample_kv(), specs, 32, 300, |i, w| {
            w.write_all(&vec![(i + 1) as u8; 256])
        })
        .unwrap();
        assert_eq!(parts.len(), 2);
        (dir, parts[0].clone())
    }

    #[test]
    fn a_missing_part_is_refused() {
        let (dir, entry) = two_part_fixture("missing");
        let missing = dir.join("m-00002-of-00002.gguf");
        std::fs::remove_file(&missing).unwrap();
        assert!(
            crate::gguf::load_gguf_model(&entry).is_none(),
            "a split with a missing part must not load"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_part_with_the_wrong_split_index_is_refused() {
        let (dir, entry) = two_part_fixture("wrongindex");
        // Part 1 is really part 0 (split.no = 0) under the second filename.
        let src = dir.join("m-00001-of-00002.gguf");
        let dst = dir.join("m-00002-of-00002.gguf");
        std::fs::copy(&src, &dst).unwrap();
        assert!(
            crate::gguf::load_gguf_model(&entry).is_none(),
            "split.no must match the part's position"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_filename_count_that_disagrees_with_split_count_is_refused() {
        let (dir, entry) = two_part_fixture("badcount");
        // The filename claims 3 parts while split.count (and the directory) say 2.
        let renamed = dir.join("m-00001-of-00003.gguf");
        std::fs::rename(&entry, &renamed).unwrap();
        assert!(
            crate::gguf::load_gguf_model(&renamed).is_none(),
            "a count mismatch between filename and split.count must not load"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_refuses_a_payload_of_the_wrong_size() {
        let path = tmp("badsize.gguf");
        let specs = vec![TensorSpec::new("w", [8, 1, 1, 1], GgmlType::F32)];
        let err = write_single(&path, &sample_kv(), specs, 32, |_, w| {
            w.write_all(&[0u8; 8])
        })
        .unwrap_err();
        assert!(err.contains("provider wrote 8 bytes, expected 32"), "{err}");
        let _ = std::fs::remove_file(&path);
    }
}
