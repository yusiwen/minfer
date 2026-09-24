//! KV session save and restore (Phase C / ticket C5).
//!
//! A session's KV rows and run table live only in memory (`KvCache`), so a restart
//! has to re-prefill everything. This module is the **container**: a versioned,
//! checksummed file holding one header, one K/V blob per layer, and the arena's
//! bookkeeping ([`KvSessionState`]).
//!
//! Design choices worth naming:
//!
//! - **Everything a reader needs to validate comes first.** The header carries the
//!   format, the backend and the shape, so a file from another model, another
//!   `n_ctx`, another KV format or another device is rejected before a single byte
//!   is applied.
//! - **Validate before apply.** [`verify`] walks the whole file (header, payload
//!   lengths, bookkeeping, checksum, and that nothing follows it) without touching
//!   an arena; the allocator only then runs the applying pass. A truncated or
//!   corrupted file therefore cannot leave a half-restored cache behind.
//! - **Truncation is caught by construction**, not by a checksum alone: every read
//!   is exact against a length the header fixes, so a short file fails with
//!   "truncated" wherever it stops, and `read_exact` cannot silently stop early.
//!
//! The bytes travel as f32 *words* (the pool's unit), exactly as the backend holds
//! them — a packed Q8_0 region is words too, which is why the element type is
//! recorded and checked rather than inferred. The header's **flags word encodes it**
//! (`FLAG_PACKED` for Q8_0, `FLAG_F16` for f16, `0` for f32 — `#130`), and the
//! reader refuses an unknown bit and the mutually exclusive `packed | f16`
//! combination loudly, so a container is never applied under a layout it does not
//! describe.
//!
//! Design record: `docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 (C5).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::kvcache::{KvSessionState, SeqId, SeqSlot, SharedPrefix};
use super::kvformat::KvFormat;
use super::Backend;

/// File magic, first bytes of every session file.
pub const MAGIC: [u8; 8] = *b"MINFERKV";
/// Container version. A reader refuses anything else, loudly.
///
/// - **1** — the header, one K/V blob per layer and the KV bookkeeping.
/// - **2** — adds the **host-state blob** (C5 S2): an opaque, length-prefixed section
///   after the bookkeeping, carried and checksummed exactly like the rest of the file.
///   The KV rows belong to a host state — the CLI's conversation, a server slot — and a
///   restore that brought the rows back without it would be a different session, so the
///   container carries both or neither. A version-1 file is refused loudly and the
///   caller falls back to re-seeding, which is what the version byte is for.
///
/// The version was deliberately **not** bumped when the element type gained the
/// `FLAG_F16` bit (`#130`): that is an additive flag, exactly as `FLAG_PACKED` was,
/// and an older v2 build refuses the unknown bit rather than guessing — so no file
/// on disk changes meaning. A version bump is for a *layout* change; a new refusal
/// on a bit nobody wrote before is not one.
pub const VERSION: u32 = 2;
/// Flags bit 0: the regions are packed Q8_0 cells (C4).
const FLAG_PACKED: u32 = 1 << 0;
/// Flags bit 1: the regions are f16 cells (C4's GPU bandwidth policy).
///
/// Kept as a **new bit** rather than a format field, so the flag word's byte
/// layout — and therefore every file already on disk — is untouched: a Q8_0 file
/// still reads bit 0 exactly as before, and an f32 file is still `flags == 0`.
/// A pre-#130 build reading an f16 file does not know this bit, so it refuses it
/// loudly at the unknown-flags check instead of decoding the region as f32 — the
/// same property [`FLAG_PACKED`] had when it landed, and the reason no version
/// bump is needed.
const FLAG_F16: u32 = 1 << 1;
/// Every bit this reader understands. A file with any other bit set was written
/// by a newer build, and its layout is not ours to guess.
const KNOWN_FLAGS: u32 = FLAG_PACKED | FLAG_F16;

/// The flag word a header's element type encodes to. The exact inverse of
/// [`format_of_flags`], so the writer and the reader cannot drift: the match is
/// exhaustive over [`KvFormat`], so a fourth format is a compile error here
/// rather than a silently unencoded one.
fn flags_of(format: KvFormat) -> u32 {
    match format {
        KvFormat::Q8_0 => FLAG_PACKED,
        KvFormat::F16 => FLAG_F16,
        KvFormat::F32 => 0,
    }
}

/// The element type a flag word names. The caller must already have refused
/// unknown bits and the mutually exclusive `FLAG_PACKED | FLAG_F16`, so the
/// remaining combinations are exactly the three formats.
fn format_of_flags(flags: u32) -> KvFormat {
    if flags & FLAG_PACKED != 0 {
        KvFormat::Q8_0
    } else if flags & FLAG_F16 != 0 {
        KvFormat::F16
    } else {
        KvFormat::F32
    }
}

/// What a reader must check before it applies anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSessionHeader {
    pub version: u32,
    pub format: KvFormat,
    pub backend: Backend,
    pub n_layer: usize,
    pub n_ctx: usize,
    /// Logical row width (`n_kv_embd`) — what the store's K/V input means.
    pub n_embd: usize,
    /// Stored f32 words per cell (C4: a packed Q8_0 cell is narrower per element).
    pub row_elems: usize,
}

impl KvSessionHeader {
    /// Bytes one layer's K and V regions occupy together.
    pub fn layer_words(&self) -> usize {
        self.n_ctx * self.row_elems
    }
}

/// What the caller knows about the arena a file must describe (C5): the backend
/// the layers will live on, the context length, and the model's logical KV row
/// width. A header that disagrees is refused before anything is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSessionExpect {
    pub backend: Backend,
    pub n_ctx: usize,
    pub n_embd: usize,
}

/// The backend a model's device uses (a session is one arena, so the two must agree).
///
/// F4: `Device::backend()` is the single bridge between the device and the
/// registry id spaces; this delegates rather than spelling the mapping twice.
pub fn backend_of(device: crate::models::Device) -> Backend {
    device.backend()
}

/// What a KV session must match to be loadable by `model` at `n_ctx` (C5): the device
/// the rows will live on, the context length, and the model's logical KV row width.
///
/// One function, so the CLI's `--session` companion, the server's slot snapshot and the
/// gates cannot disagree about what "this file describes this run" means — the three of
/// them used to spell it out separately.
pub fn expect_for(model: &dyn crate::models::ModelDef, n_ctx: usize) -> KvSessionExpect {
    KvSessionExpect {
        backend: backend_of(model.device()),
        n_ctx,
        n_embd: model.n_kv_embd(),
    }
}

/// What a save wrote or a load read (the caller logs it; the gates assert it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSessionReport {
    pub layers: usize,
    pub cells: usize,
    pub bytes: u64,
    /// Highest written position — where the resumed session continues.
    pub written: usize,
}

// ---- FNV-1a, the container's integrity check ---------------------------------
//
// Not cryptographic: it catches a truncated or bit-flipped file, which is what
// "rejected loudly" means here. A hash that needs a dependency for a few hundred
// MB/s of startup I/O is not worth it.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn hash_bytes(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// The on-disk backend tag: the registry id, so the file format is the handle's
/// id space and nothing else (F4). `backend_of_tag` is its inverse.
fn tag_of(backend: Backend) -> u32 {
    backend.index() as u32
}

fn backend_of_tag(tag: u32) -> Result<Backend, String> {
    Backend::from_index(tag as usize).ok_or_else(|| {
        format!("KV session: unknown backend tag {tag} (this file was written by a newer build?)")
    })
}

// ---- writing ----------------------------------------------------------------

struct Sink<W: Write> {
    w: W,
    hash: u64,
    written: u64,
}

impl<W: Write> Sink<W> {
    fn new(w: W) -> Self {
        Self {
            w,
            hash: FNV_OFFSET,
            written: 0,
        }
    }

    fn bytes(&mut self, b: &[u8]) -> Result<(), String> {
        self.w
            .write_all(b)
            .map_err(|e| format!("KV session write: {e}"))?;
        hash_bytes(&mut self.hash, b);
        self.written += b.len() as u64;
        Ok(())
    }

    fn u32(&mut self, v: u32) -> Result<(), String> {
        self.bytes(&v.to_le_bytes())
    }

    fn u64(&mut self, v: u64) -> Result<(), String> {
        self.bytes(&v.to_le_bytes())
    }

    fn words(&mut self, w: &[f32]) -> Result<(), String> {
        // Chunked so a multi-hundred-MB region does not need a second copy, and
        // hashed word by word in the same order the reader will hash it.
        const CHUNK: usize = 1 << 14;
        let mut buf = [0u8; CHUNK * 4];
        for chunk in w.chunks(CHUNK) {
            for (i, &x) in chunk.iter().enumerate() {
                buf[i * 4..(i + 1) * 4].copy_from_slice(&x.to_bits().to_le_bytes());
            }
            self.bytes(&buf[..chunk.len() * 4])?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<u64, String> {
        let sum = self.hash;
        self.w
            .write_all(&sum.to_le_bytes())
            .map_err(|e| format!("KV session write: {e}"))?;
        self.w
            .flush()
            .map_err(|e| format!("KV session flush: {e}"))?;
        Ok(self.written + 8)
    }
}

/// Streaming writer: create it, hand it one layer at a time, then finish with the
/// arena's bookkeeping. Nothing is buffered whole in memory.
pub struct KvSessionWriter {
    sink: Sink<BufWriter<File>>,
    header: KvSessionHeader,
    layers: usize,
    /// The caller's host state (C5 S2), opaque to this container: written after the
    /// bookkeeping, inside the checksum. Empty when the caller has none.
    host: Vec<u8>,
}

impl KvSessionWriter {
    pub fn create(path: &Path, header: &KvSessionHeader) -> Result<Self, String> {
        let file = File::create(path)
            .map_err(|e| format!("KV session: cannot create {}: {e}", path.display()))?;
        let mut sink = Sink::new(BufWriter::new(file));
        sink.bytes(&MAGIC)?;
        sink.u32(header.version)?;
        sink.u32(flags_of(header.format))?;
        sink.u32(tag_of(header.backend))?;
        sink.u32(header.n_layer as u32)?;
        sink.u32(header.n_ctx as u32)?;
        sink.u32(header.n_embd as u32)?;
        sink.u32(header.row_elems as u32)?;
        Ok(Self {
            sink,
            header: header.clone(),
            layers: 0,
            host: Vec::new(),
        })
    }

    /// Attach the host state that the KV rows belong to (C5 S2). Opaque here: the
    /// caller decides the encoding, the container only carries it, length-prefixed and
    /// covered by the checksum.
    pub fn set_host(&mut self, host: &[u8]) {
        self.host = host.to_vec();
    }

    /// One layer's two regions, in pool words.
    pub fn layer(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        let want = self.header.layer_words();
        if k.len() != want || v.len() != want {
            return Err(format!(
                "KV session: layer {layer} has {} K / {} V words, expected {want} each \
                 (n_ctx {} x row_elems {})",
                k.len(),
                v.len(),
                self.header.n_ctx,
                self.header.row_elems
            ));
        }
        if self.layers >= self.header.n_layer {
            return Err(format!(
                "KV session: more than {} layers were written",
                self.header.n_layer
            ));
        }
        self.sink.u32(layer as u32)?;
        self.sink.u32(k.len() as u32)?;
        self.sink.u32(v.len() as u32)?;
        self.sink.words(k)?;
        self.sink.words(v)?;
        self.layers += 1;
        Ok(())
    }

    /// Write the bookkeeping and the checksum. Refuses a short file: a container
    /// that claims `n_layer` and holds fewer is corrupt by construction.
    pub fn finish(self, state: &KvSessionState) -> Result<KvSessionReport, String> {
        if self.layers != self.header.n_layer {
            return Err(format!(
                "KV session: the header declares {} layers but {} were written",
                self.header.n_layer, self.layers
            ));
        }
        if state.n_ctx != self.header.n_ctx {
            return Err(format!(
                "KV session: the bookkeeping describes {}-cell runs, the header {}",
                state.n_ctx, self.header.n_ctx
            ));
        }
        let layers = self.layers;
        let cells = state.n_ctx;
        let written = state.written();
        let mut sink = self.sink;
        write_state(&mut sink, state)?;
        // C5 S2: the host blob last, still inside the checksum.
        sink.u32(self.host.len() as u32)?;
        sink.bytes(&self.host)?;
        let bytes = sink.finish()?;
        Ok(KvSessionReport {
            layers,
            cells,
            bytes,
            written,
        })
    }
}

fn write_state(sink: &mut Sink<impl Write>, st: &KvSessionState) -> Result<(), String> {
    sink.u32(st.identity as u32)?;
    sink.u32(st.n_ctx as u32)?;
    sink.u32(st.seqs.len() as u32)?;
    for (seq, slot) in &st.seqs {
        sink.u32(*seq)?;
        sink.u32(slot.start as u32)?;
        sink.u32(slot.cap as u32)?;
        sink.u32(slot.shared.cell as u32)?;
        sink.u32(slot.shared.rows as u32)?;
        sink.u32(slot.written as u32)?;
    }
    sink.u32(st.spans.len() as u32)?;
    for (seq, spans) in &st.spans {
        sink.u32(*seq)?;
        sink.u32(spans.len() as u32)?;
        for &(base, cell, len) in spans {
            sink.u32(base as u32)?;
            sink.u32(cell as u32)?;
            sink.u32(len as u32)?;
        }
    }
    sink.u32(st.layers.len() as u32)?;
    for ls in &st.layers {
        sink.u32(ls.layer as u32)?;
        sink.u32(ls.n_used as u32)?;
        sink.u32(ls.owner.len() as u32)?;
        for &o in &ls.owner {
            sink.u32(o)?;
        }
    }
    sink.u64(st.defrags)?;
    sink.u64(st.cells_moved)?;
    sink.u64(st.cows)?;
    sink.u64(st.cow_cells)?;
    Ok(())
}

// ---- reading ----------------------------------------------------------------

struct Source<R: Read> {
    r: R,
    hash: u64,
    read: u64,
}

impl<R: Read> Source<R> {
    fn new(r: R) -> Self {
        Self {
            r,
            hash: FNV_OFFSET,
            read: 0,
        }
    }

    fn bytes(&mut self, n: usize, what: &str) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; n];
        self.r.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                format!(
                    "KV session: truncated — the file ends inside {what} (wanted {n} more bytes)"
                )
            } else {
                format!("KV session read: {e}")
            }
        })?;
        hash_bytes(&mut self.hash, &buf);
        self.read += n as u64;
        Ok(buf)
    }

    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let b = self.bytes(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self, what: &str) -> Result<u64, String> {
        let b = self.bytes(8, what)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(&b);
        Ok(u64::from_le_bytes(a))
    }

    fn words(&mut self, n: usize, what: &str) -> Result<Vec<f32>, String> {
        const CHUNK: usize = 1 << 14;
        let mut out = Vec::with_capacity(n);
        let mut left = n;
        while left > 0 {
            let take = left.min(CHUNK);
            let b = self.bytes(take * 4, what)?;
            for w in b.chunks_exact(4) {
                out.push(f32::from_bits(u32::from_le_bytes([w[0], w[1], w[2], w[3]])));
            }
            left -= take;
        }
        Ok(out)
    }

    /// Read the trailing checksum and compare, then require end-of-file.
    fn finish(mut self, what: &str) -> Result<u64, String> {
        let want = self.hash;
        let mut sum = [0u8; 8];
        self.r.read_exact(&mut sum).map_err(|_| {
            format!("KV session: truncated — the file ends before the checksum of {what}")
        })?;
        let got = u64::from_le_bytes(sum);
        if got != want {
            return Err(format!(
                "KV session: checksum mismatch ({got:#018x} in the file, {want:#018x} computed) \
                 — the file is corrupt"
            ));
        }
        let mut extra = [0u8; 1];
        match self.r.read(&mut extra) {
            Ok(0) => Ok(self.read + 8),
            Ok(_) => Err("KV session: trailing bytes after the checksum".into()),
            Err(e) => Err(format!("KV session read: {e}")),
        }
    }
}

/// Streaming reader: `open` validates the header, `next_layer` yields one layer at
/// a time, `finish` returns the bookkeeping after checking the checksum.
pub struct KvSessionReader {
    src: Source<BufReader<File>>,
    header: KvSessionHeader,
    layers: usize,
}

impl KvSessionReader {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("KV session: cannot open {}: {e}", path.display()))?;
        let mut src = Source::new(BufReader::new(file));
        let magic = src.bytes(8, "the magic")?;
        if magic != MAGIC {
            return Err(format!(
                "KV session: {} does not start with the {}-byte magic {:?} (it is not a minfer \
                 KV session file)",
                path.display(),
                MAGIC.len(),
                String::from_utf8_lossy(&MAGIC)
            ));
        }
        let version = src.u32("the version")?;
        if version != VERSION {
            return Err(format!(
                "KV session: version {version} cannot be read by this build (version {VERSION}); \
                 refusing rather than guessing the layout"
            ));
        }
        let flags = src.u32("the flags")?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(format!(
                "KV session: unknown flags {flags:#x} (this file was written by a newer build \
                 that knows element-type bits this one does not)"
            ));
        }
        // The element-type bits are mutually exclusive: "packed Q8_0" and "f16"
        // describe two different cell layouts, so a header claiming both is
        // corrupt. Refuse it rather than resolving it by preferring one bit —
        // the preference would decide, silently, which layout a file with the
        // other one's payload is applied as.
        if flags & FLAG_PACKED != 0 && flags & FLAG_F16 != 0 {
            return Err(format!(
                "KV session: the header claims both packed Q8_0 and f16 cells (flags {flags:#x}); \
                 the two element types are mutually exclusive, so the file is corrupt"
            ));
        }
        let backend = backend_of_tag(src.u32("the backend tag")?)?;
        let n_layer = src.u32("the layer count")? as usize;
        let n_ctx = src.u32("n_ctx")? as usize;
        let n_embd = src.u32("n_kv_embd")? as usize;
        let row_elems = src.u32("the cell width")? as usize;
        let format = format_of_flags(flags);
        if n_layer == 0 || n_ctx == 0 || n_embd == 0 || row_elems == 0 {
            return Err(format!(
                "KV session: the header declares {n_layer} layers of {n_ctx} cells x {row_elems} \
                 words ({n_embd} elements/cell) — a zero-sized arena is not a session"
            ));
        }
        // The packed flag and the cell width must agree: a file claiming packed
        // cells with an f32-shaped width (or the reverse) would be applied as the
        // wrong layout, which is exactly the silent corruption the format check is
        // for.
        let expect_row = format.row_elems(n_embd);
        if row_elems != expect_row {
            return Err(format!(
                "KV session: the header says {} with {row_elems} words per {n_embd}-element cell, \
                 but that format packs a cell into {expect_row}",
                format.name()
            ));
        }
        Ok(Self {
            src,
            header: KvSessionHeader {
                version,
                format,
                backend,
                n_layer,
                n_ctx,
                n_embd,
                row_elems,
            },
            layers: 0,
        })
    }

    pub fn header(&self) -> &KvSessionHeader {
        &self.header
    }

    /// The next layer's `(index, K words, V words)`, or `None` once all layers the
    /// header declared have been read.
    pub fn next_layer(&mut self) -> Result<Option<(usize, Vec<f32>, Vec<f32>)>, String> {
        if self.layers == self.header.n_layer {
            return Ok(None);
        }
        let what = format!("layer {}", self.layers);
        let layer = self.src.u32(&format!("{what}: its index"))? as usize;
        let kw = self.src.u32(&format!("{what}: its K length"))? as usize;
        let vw = self.src.u32(&format!("{what}: its V length"))? as usize;
        let want = self.header.layer_words();
        if kw != want || vw != want {
            return Err(format!(
                "KV session: layer {layer} declares {kw} K / {vw} V words, expected {want} \
                 ({} x {})",
                self.header.n_ctx, self.header.row_elems
            ));
        }
        let k = self.src.words(kw, &format!("{what}: its K rows"))?;
        let v = self.src.words(vw, &format!("{what}: its V rows"))?;
        self.layers += 1;
        Ok(Some((layer, k, v)))
    }

    /// Read the bookkeeping, the host state and the checksum. Everything before this is
    /// validated against the header, so what arrives here is structurally sound.
    pub fn finish(mut self) -> Result<KvSessionBody, String> {
        if self.layers != self.header.n_layer {
            return Err(format!(
                "KV session: the header declares {} layers, the file holds {}",
                self.header.n_layer, self.layers
            ));
        }
        let state = read_state(&mut self.src, &self.header)?;
        // C5 S2: the host blob, then the checksum that covers everything.
        let host_len = self.src.u32("the host-state length")? as usize;
        let host = self.src.bytes(host_len, "the host state")?;
        let layers = self.layers;
        let cells = self.header.n_ctx;
        let written = state.written();
        let bytes = self.src.finish("the bookkeeping")?;
        Ok(KvSessionBody {
            state,
            host,
            report: KvSessionReport {
                layers,
                cells,
                bytes,
                written,
            },
        })
    }
}

/// What a reader hands back: the KV bookkeeping, the caller's host state (C5 S2) and
/// the report. One struct because the three are read together and must stay together —
/// applying the rows without the host state is what this increment exists to prevent.
#[derive(Debug, Clone, PartialEq)]
pub struct KvSessionBody {
    pub state: KvSessionState,
    /// Opaque to this container; whatever [`KvSessionWriter::set_host`] was given.
    pub host: Vec<u8>,
    pub report: KvSessionReport,
}

fn read_state(
    src: &mut Source<impl Read>,
    header: &KvSessionHeader,
) -> Result<KvSessionState, String> {
    let identity = src.u32("the identity flag")? != 0;
    let n_ctx = src.u32("the bookkeeping's n_ctx")? as usize;
    if n_ctx != header.n_ctx {
        return Err(format!(
            "KV session: the bookkeeping describes {n_ctx}-cell runs, the header {}",
            header.n_ctx
        ));
    }
    let n_seq = src.u32("the sequence count")? as usize;
    let mut seqs = Vec::with_capacity(n_seq);
    for i in 0..n_seq {
        let what = format!("sequence {i}");
        let seq = src.u32(&format!("{what}: its id"))? as SeqId;
        let start = src.u32(&format!("{what}: its start"))? as usize;
        let cap = src.u32(&format!("{what}: its capacity"))? as usize;
        let cell = src.u32(&format!("{what}: its shared prefix cell"))? as usize;
        let rows = src.u32(&format!("{what}: its shared prefix rows"))? as usize;
        let written = src.u32(&format!("{what}: its written extent"))? as usize;
        seqs.push((
            seq,
            SeqSlot {
                start,
                cap,
                shared: SharedPrefix { cell, rows },
                written,
            },
        ));
    }
    let n_spans = src.u32("the span-list count")? as usize;
    let mut spans = Vec::with_capacity(n_spans);
    for i in 0..n_spans {
        let what = format!("span list {i}");
        let seq = src.u32(&format!("{what}: its sequence"))? as SeqId;
        let count = src.u32(&format!("{what}: its length"))? as usize;
        let mut v = Vec::with_capacity(count);
        for j in 0..count {
            let base = src.u32(&format!("{what} entry {j}: position base"))? as usize;
            let cell = src.u32(&format!("{what} entry {j}: first cell"))? as usize;
            let len = src.u32(&format!("{what} entry {j}: length"))? as usize;
            v.push((base, cell, len));
        }
        spans.push((seq, v));
    }
    let n_layers = src.u32("the bookkeeping's layer count")? as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let what = format!("bookkeeping layer {i}");
        let layer = src.u32(&format!("{what}: its index"))? as usize;
        let n_used = src.u32(&format!("{what}: its written extent"))? as usize;
        let owners = src.u32(&format!("{what}: its owner-table length"))? as usize;
        if owners != header.n_ctx {
            return Err(format!(
                "KV session: layer {layer} carries a {owners}-entry owner table for a {}-cell \
                 arena",
                header.n_ctx
            ));
        }
        let mut owner = Vec::with_capacity(owners);
        for j in 0..owners {
            owner.push(src.u32(&format!("{what} owner {j}"))?);
        }
        layers.push(super::kvcache::KvLayerSession {
            layer,
            n_used,
            owner,
        });
    }
    let defrags = src.u64("the defrag counter")?;
    let cells_moved = src.u64("the cells-moved counter")?;
    let cows = src.u64("the copy-on-write counter")?;
    let cow_cells = src.u64("the copied-cells counter")?;
    Ok(KvSessionState {
        identity,
        n_ctx,
        seqs,
        spans,
        layers,
        defrags,
        cells_moved,
        cows,
        cow_cells,
    })
}

/// Walk a session file end to end **without applying anything** — the pass that
/// makes a failed load a no-op. Returns the header it validated.
pub fn verify(path: &Path) -> Result<KvSessionHeader, String> {
    let mut r = KvSessionReader::open(path)?;
    let header = r.header().clone();
    while let Some((_, k, v)) = r.next_layer()? {
        // Dropped: the point of this pass is the lengths, the state and the
        // checksum, in file order.
        let _ = (k, v);
    }
    let _body = r.finish()?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::kvcache::{FREE, SEQ_MAIN};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "minfer-kvsession-{}-{name}.bin",
            std::process::id()
        ));
        p
    }

    fn header(n_layer: usize, n_ctx: usize, n_embd: usize, format: KvFormat) -> KvSessionHeader {
        KvSessionHeader {
            version: VERSION,
            format,
            backend: Backend::CPU,
            n_layer,
            n_ctx,
            n_embd,
            row_elems: format.row_elems(n_embd),
        }
    }

    fn state(n_ctx: usize) -> KvSessionState {
        KvSessionState {
            identity: true,
            n_ctx,
            seqs: vec![(
                SEQ_MAIN,
                SeqSlot {
                    start: 0,
                    cap: n_ctx,
                    shared: SharedPrefix::default(),
                    written: 3,
                },
            )],
            spans: vec![(SEQ_MAIN, vec![(0, 0, 3)])],
            layers: vec![
                crate::graph::kvcache::KvLayerSession {
                    layer: 0,
                    n_used: 3,
                    owner: {
                        let mut o = vec![FREE; n_ctx];
                        o[0] = SEQ_MAIN;
                        o[1] = SEQ_MAIN;
                        o[2] = SEQ_MAIN;
                        o
                    },
                },
                crate::graph::kvcache::KvLayerSession {
                    layer: 1,
                    n_used: 3,
                    owner: vec![SEQ_MAIN; n_ctx],
                },
            ],
            defrags: 1,
            cells_moved: 7,
            cows: 2,
            cow_cells: 5,
        }
    }

    fn write_session(path: &Path, h: &KvSessionHeader, st: &KvSessionState) -> KvSessionReport {
        write_session_with_host(path, h, st, &[])
    }

    fn write_session_with_host(
        path: &Path,
        h: &KvSessionHeader,
        st: &KvSessionState,
        host: &[u8],
    ) -> KvSessionReport {
        let mut w = KvSessionWriter::create(path, h).expect("create");
        w.set_host(host);
        for layer in 0..h.n_layer {
            let k: Vec<f32> = (0..h.layer_words())
                .map(|i| (i as f32) * 0.5 + layer as f32)
                .collect();
            let v: Vec<f32> = (0..h.layer_words())
                .map(|i| 1.0 / (i as f32 + 1.0))
                .collect();
            w.layer(layer, &k, &v).expect("layer");
        }
        w.finish(st).expect("finish")
    }

    #[test]
    fn a_session_round_trips_including_the_run_table() {
        let path = temp_path("roundtrip");
        let h = header(2, 16, 32, KvFormat::F32);
        let st = state(16);
        let report = write_session(&path, &h, &st);
        assert_eq!(report.layers, 2);
        assert_eq!(report.cells, 16);
        assert_eq!(report.written, 3);
        assert_eq!(report.bytes, std::fs::metadata(&path).unwrap().len());

        let mut r = KvSessionReader::open(&path).expect("open");
        assert_eq!(r.header(), &h);
        let mut seen = Vec::new();
        while let Some((layer, k, v)) = r.next_layer().expect("layer") {
            assert_eq!(k.len(), h.layer_words());
            assert_eq!(v.len(), h.layer_words());
            assert_eq!(k[1], 0.5 + layer as f32);
            seen.push(layer);
        }
        assert_eq!(seen, vec![0, 1]);
        let body = r.finish().expect("finish");
        assert_eq!(body.state, st);
        assert_eq!(body.report, report);
        assert!(body.host.is_empty(), "no host state was written");
        std::fs::remove_file(&path).ok();
    }

    /// C5 S2: the host state rides inside the container — length-prefixed, after the
    /// bookkeeping and **inside the checksum**, so a flipped byte in it is refused
    /// exactly like one in the KV rows.
    #[test]
    fn the_host_state_round_trips_and_is_covered_by_the_checksum() {
        let path = temp_path("host");
        let h = header(1, 8, 32, KvFormat::F32);
        let host = br#"{"messages":[["user","hi"]],"current_pos":3}"#.to_vec();
        let written = write_session_with_host(&path, &h, &state(8), &host);
        assert!(written.bytes > host.len() as u64, "{written:?}");

        let mut r = KvSessionReader::open(&path).expect("open");
        while r.next_layer().expect("layer").is_some() {}
        let body = r.finish().expect("finish");
        assert_eq!(
            body.host, host,
            "the host blob must round-trip byte for byte"
        );
        assert_eq!(body.state, state(8));

        // Flip a byte inside the host blob: the checksum covers it.
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.len() - host.len() - 5; // inside the blob, before the trailing words
        bytes[at] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        let err = verify(&path).unwrap_err();
        assert!(err.contains("checksum"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    /// C5 S2 keeps version 1 readable-as-a-refusal: a v1 file has no host section, and
    /// applying its rows without the host state is exactly what the bump prevents.
    #[test]
    fn a_version_1_file_is_refused_so_the_caller_can_re_seed() {
        let path = temp_path("v1");
        let h = header(1, 8, 32, KvFormat::F32);
        write_session(&path, &h, &state(8));
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = verify(&path).unwrap_err();
        assert!(
            err.contains("version 1") && err.contains(&format!("version {VERSION}")),
            "{err}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_packed_session_records_its_cell_width() {
        let path = temp_path("packed");
        let h = header(1, 8, 128, KvFormat::Q8_0);
        assert_eq!(h.row_elems, 34);
        let st = KvSessionState {
            layers: vec![crate::graph::kvcache::KvLayerSession {
                layer: 0,
                n_used: 1,
                owner: vec![SEQ_MAIN; 8],
            }],
            ..state(8)
        };
        write_session(&path, &h, &st);
        let r = KvSessionReader::open(&path).expect("open");
        assert_eq!(r.header().format, KvFormat::Q8_0);
        assert!(r.header().format.is_packed());
        std::fs::remove_file(&path).ok();
    }

    /// #130: an f16 arena round-trips at the container level, no device needed —
    /// the container carries f32 words and the element type is a flag. Before the
    /// `FLAG_F16` bit an f16 header was written as `flags == 0` and read back as
    /// f32, so `kv_load` refused the file its own writer had just produced.
    #[test]
    fn an_f16_session_round_trips() {
        let path = temp_path("f16-roundtrip");
        let h = header(2, 16, 128, KvFormat::F16);
        let st = state(16);
        let report = write_session(&path, &h, &st);

        let mut r = KvSessionReader::open(&path).expect("open");
        assert_eq!(
            r.header(),
            &h,
            "the whole header, element type included, survives"
        );
        assert_eq!(r.header().format, KvFormat::F16);
        let mut seen = Vec::new();
        while let Some((layer, k, v)) = r.next_layer().expect("layer") {
            assert_eq!(k.len(), h.layer_words());
            assert_eq!(v.len(), h.layer_words());
            seen.push(layer);
        }
        assert_eq!(seen, vec![0, 1]);
        let body = r.finish().expect("finish");
        assert_eq!(body.state, st);
        assert_eq!(body.report, report);
        // The full-file pass the allocator runs before applying anything agrees.
        assert_eq!(verify(&path).unwrap().format, KvFormat::F16);
        std::fs::remove_file(&path).ok();
    }

    /// #130: the flags word **is** the element type, and every format round trips
    /// through it — the encode/decode symmetry the fix rests on. If `flags_of` stops
    /// writing a format's bit (`F16` → `0`, say) the reader decodes the wrong type
    /// and this fails on the on-disk word, not only on the decoded header.
    #[test]
    fn the_flag_word_encodes_every_element_type_and_round_trips() {
        for (format, want_flags) in [
            (KvFormat::F32, 0u32),
            (KvFormat::F16, FLAG_F16),
            (KvFormat::Q8_0, FLAG_PACKED),
        ] {
            let path = temp_path(&format!("flags-{}", format.name()));
            let h = header(1, 8, 32, format);
            write_session(&path, &h, &state(8));
            // The on-disk flags word is exactly that format's encoding.
            let bytes = std::fs::read(&path).unwrap();
            let on_disk = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
            assert_eq!(on_disk, want_flags, "{} flags word", format.name());
            // And the reader decodes it back symmetrically.
            let r = KvSessionReader::open(&path).expect("open");
            assert_eq!(r.header().format, format);
            assert_eq!(r.header().format, format_of_flags(on_disk));
            std::fs::remove_file(&path).ok();
        }
    }

    /// #130: the two element-type bits are mutually exclusive. "packed Q8_0" and
    /// "f16" are different cell layouts, so a header claiming both is corrupt: the
    /// reader must refuse it, not prefer one bit and apply the payload under a
    /// layout the file does not describe. The message is asserted so that the
    /// *width* check (which would also catch this combination) cannot stand in for
    /// the rule under test.
    #[test]
    fn a_header_claiming_both_packed_and_f16_is_refused() {
        let path = temp_path("both-flags");
        let h = header(1, 8, 32, KvFormat::F32);
        write_session(&path, &h, &state(8));
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[12..16].copy_from_slice(&(FLAG_PACKED | FLAG_F16).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = KvSessionReader::open(&path).err().expect("must be refused");
        assert!(err.contains("mutually exclusive"), "{err}");
        assert!(err.contains("corrupt"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    /// #130 keeps `FLAG_PACKED`'s property: an unknown flag bit is refused, never
    /// ignored. This is also the compatibility story for a *newer* writer — a
    /// pre-#130 build reading an f16 file sees `FLAG_F16` as unknown and takes
    /// exactly this path instead of decoding the region as f32.
    #[test]
    fn an_unknown_flag_bit_is_refused() {
        let path = temp_path("unknown-flag");
        let h = header(1, 8, 32, KvFormat::F16);
        write_session(&path, &h, &state(8));
        let mut bytes = std::fs::read(&path).unwrap();
        // A bit no version has assigned yet, on top of a valid f16 file.
        bytes[12..16].copy_from_slice(&(FLAG_F16 | (1 << 7)).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = KvSessionReader::open(&path).err().expect("must be refused");
        assert!(err.contains("unknown flags"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    /// #130 compatibility: every session written before the f16 bit existed is
    /// `flags == 0`, i.e. f32, and must keep loading **as f32** — a file that merely
    /// predates the bit is not a newer file and must not be refused. (A literal
    /// version-1 file is refused by the version check, `a_version_1_file_is_refused_…`.)
    #[test]
    fn a_legacy_f32_file_without_the_f16_bit_still_loads_as_f32() {
        let path = temp_path("legacy-f32");
        let h = header(1, 8, 32, KvFormat::F32);
        write_session(&path, &h, &state(8));
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            0,
            "a pre-#130 f32 file's flags word is zero"
        );
        let r = KvSessionReader::open(&path).expect("a legacy f32 file still loads");
        assert_eq!(r.header().format, KvFormat::F32);
        assert_eq!(verify(&path).unwrap().format, KvFormat::F32);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_truncated_file_is_refused_and_says_so() {
        let path = temp_path("truncated");
        let h = header(2, 16, 32, KvFormat::F32);
        write_session(&path, &h, &state(16));
        let full = std::fs::read(&path).unwrap();
        // Cut in the middle of the payload: the reader must fail on a short read,
        // not stop early with a half-restored arena.
        std::fs::write(&path, &full[..full.len() / 2]).unwrap();
        let err = verify(&path).unwrap_err();
        assert!(err.contains("truncated"), "{err}");
        assert!(err.contains("the file ends"), "{err}");
        // The streaming walk fails in the same place — the header itself fits, so
        // this is the payload read refusing, which is why `kv_load` runs `verify`
        // before it applies anything.
        let mut r = KvSessionReader::open(&path).expect("the header survives a cut in the payload");
        let err = loop {
            match r.next_layer() {
                Ok(Some(_)) => continue,
                Ok(None) => panic!("a truncated payload must not read to the end"),
                Err(e) => break e,
            }
        };
        assert!(err.contains("truncated"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_version_mismatch_is_refused_loudly() {
        let path = temp_path("version");
        let h = header(1, 8, 32, KvFormat::F32);
        write_session(&path, &h, &state(8));
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = KvSessionReader::open(&path).err().expect("must be refused");
        assert!(err.contains("version"), "{err}");
        assert!(err.contains(&format!("version {}", VERSION)), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_flipped_byte_is_caught_by_the_checksum() {
        let path = temp_path("checksum");
        let h = header(1, 8, 32, KvFormat::F32);
        write_session(&path, &h, &state(8));
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.len() / 2; // inside the payload
        bytes[at] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        let err = verify(&path).unwrap_err();
        assert!(err.contains("checksum mismatch"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn trailing_bytes_after_the_checksum_are_refused() {
        let path = temp_path("trailing");
        let h = header(1, 8, 32, KvFormat::F32);
        write_session(&path, &h, &state(8));
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.push(0);
        std::fs::write(&path, &bytes).unwrap();
        let err = verify(&path).unwrap_err();
        assert!(err.contains("trailing bytes"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_file_that_is_not_a_session_is_refused() {
        let path = temp_path("magic");
        std::fs::write(
            &path,
            b"not a session at all, but long enough to read the magic",
        )
        .unwrap();
        let err = KvSessionReader::open(&path).err().expect("must be refused");
        assert!(err.contains("magic"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_header_whose_flag_and_width_disagree_is_refused() {
        let path = temp_path("flagwidth");
        let h = header(1, 8, 32, KvFormat::F32);
        write_session(&path, &h, &state(8));
        let mut bytes = std::fs::read(&path).unwrap();
        // Claim packed cells while keeping the f32-shaped width: the applying pass
        // would then dequantize f32 rows.
        let flags = FLAG_PACKED.to_le_bytes();
        bytes[12..16].copy_from_slice(&flags);
        std::fs::write(&path, &bytes).unwrap();
        let err = KvSessionReader::open(&path).err().expect("must be refused");
        assert!(err.contains("packs a cell into"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_writer_that_is_short_a_layer_refuses_to_finish() {
        let path = temp_path("short");
        let h = header(2, 8, 32, KvFormat::F32);
        let mut w = KvSessionWriter::create(&path, &h).expect("create");
        w.layer(0, &vec![0.0; h.layer_words()], &vec![0.0; h.layer_words()])
            .expect("layer 0");
        let err = w.finish(&state(8)).unwrap_err();
        assert!(err.contains("2 layers but 1 were written"), "{err}");
        std::fs::remove_file(&path).ok();
    }
}
