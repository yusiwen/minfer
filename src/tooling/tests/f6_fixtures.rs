//! The F6 fixture manifest, verified at the point of use (issue #205).
//!
//! The F6 byte-parity gates compare `minfer quantize` output against files under
//! `~/.cache/minfer/f6-src/` — an f16 source and a `llama-quantize` reference per
//! target. Both are inputs the gates do not produce, so a stale or replaced
//! reference used to be compared against silently and the gate stayed green.
//!
//! `docs/f6-fixtures.json` is the record (path, bytes, sha256 or a recorded
//! prefix, the exact producer command, the producer's identity, date and an
//! absolute box label). `scripts/check_f6_fixtures.py` audits the manifest's
//! structure and the whole cache; this module is the half that matters inside a
//! gate run: [`check`] refuses a resolved fixture whose **content** is not what
//! the record describes, so the documented `cargo test … --ignored` invocation
//! cannot compare against a tampered cache even though CI (which has no cache)
//! never sees those bytes.
//!
//! Three outcomes, and the rule for each:
//!
//! * a path that is not inside the fixture cache, or a directory, is left alone —
//!   `/tmp` scratch and the model cache are not fixtures, and the `hf/…`
//!   directory is an input to a conversion, not a compared artifact;
//! * a fixture inside the cache is looked up by its cache-relative path: a
//!   64-hex `sha256` must match exactly, a recorded `sha256_prefix` (a digest the
//!   2026-10-07 table truncated) matches weakly and says so, and a file no entry
//!   names is refused by name;
//! * the comparison is against **every** recorded digest for that path, so the
//!   two-box divergences the manifest exists to explain read as "one of these",
//!   and the refusal prints all of them next to the actual digest.
//!
//! The cache root is the manifest's `cache_root`, overridable with
//! `MINFER_F6_CACHE` — that override is how the refusal is demonstrated without
//! touching the real cache (copy the root's layout into `/tmp`, tamper one file,
//! point the variable at it). A deliberate experiment belongs *outside* the cache
//! root: a file the manifest does not name is only tolerated when it is not a
//! cache fixture at all. The **record** is overridable too, with
//! `MINFER_F6_MANIFEST` (issue #354): the manifest-side twin of that cache
//! override, so the whole resolver — `check` → `reference_record` → the verdict
//! classifier — can be pointed at a fabricated record in `/tmp` instead of
//! editing a tracked file. Both overrides are honoured *loudly*: an empty,
//! missing, unreadable or malformed manifest refuses by name and never falls back
//! to the checked-in record, because a gate that quietly tests a different record
//! than the one it was pointed at is worse than one that cannot be pointed
//! anywhere.
//!
//! **Which build the file is** (issue #349). A `ref/…` path holds two recorded
//! contents because the two boxes' `llama-quantize` builds differ, and the
//! byte-parity claim was measured against one of them — the entry marked
//! `authoritative_reference` in the manifest (the dgxspark `gcc 13.3.0`
//! `-ffp-contract=fast` build). [`reference_record`] answers that question for a
//! file [`check`] has already vouched for, and [`ParityVerdict::classify`] turns
//! the two byte comparisons plus that record into the gate's verdict. The three
//! outcomes the issue asks for are distinguishable from those three inputs
//! alone, and the fourth and fifth arms keep the gate honest: a mismatch against
//! the *authoritative* entry is a defect, and a mismatch against a file the
//! record does not name (a `/tmp` experiment, a replaced file) cannot be blamed
//! on a compiler.

use std::cell::Cell;
use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The checked-in record. `CARGO_MANIFEST_DIR` keeps it correct from a worktree.
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/f6-fixtures.json");

/// The record's own relocation (issue #354), the manifest-side twin of
/// [`CACHE_ENV`]: the path this reader loads instead of [`MANIFEST`], so a test
/// or an experiment can point the gate at a fabricated record without editing a
/// tracked file. Honoured **loudly** — a value that is set but empty, or a path
/// that is missing, unreadable or malformed, refuses by name and never falls back
/// to [`MANIFEST`]: a gate that quietly reads a different record than the one it
/// was pointed at is worse than one that cannot be pointed anywhere.
const MANIFEST_ENV: &str = "MINFER_F6_MANIFEST";

/// The fixture cache root's relocation (issue #205).
const CACHE_ENV: &str = "MINFER_F6_CACHE";

/// One artifact content identity. Only the fields this module judges are read;
/// `scripts/check_f6_fixtures.py` is the authority on the rest of the shape.
#[derive(Debug, Deserialize)]
struct Entry {
    path: String,
    #[serde(default)]
    bytes: Option<u64>,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    sha256_prefix: Option<String>,
    #[serde(rename = "box")]
    box_label: String,
    date: String,
    /// `llama-quantize` for a reference build; `minfer`, `hf-download` otherwise.
    #[serde(default)]
    producer_kind: Option<String>,
    #[serde(default)]
    compiler: Option<String>,
    #[serde(default)]
    ffp_contract: Option<String>,
    #[serde(default)]
    llamacpp_commit: Option<String>,
    /// The entry the byte-parity claim is asserted against (exactly one per path
    /// that has a `llama-quantize` record; `scripts/check_f6_fixtures.py` S6).
    #[serde(default)]
    authoritative_reference: Option<bool>,
}

impl Entry {
    /// The build this entry describes, for a `llama-quantize` producer.
    fn build(&self) -> Option<RecordedBuild> {
        if self.producer_kind.as_deref() != Some("llama-quantize") {
            return None;
        }
        Some(RecordedBuild {
            compiler: self.compiler.clone().unwrap_or_else(|| "unrecorded".into()),
            ffp_contract: self
                .ffp_contract
                .clone()
                .unwrap_or_else(|| "unrecorded".into()),
            llamacpp_commit: self
                .llamacpp_commit
                .clone()
                .unwrap_or_else(|| "unrecorded".into()),
            box_label: self.box_label.clone(),
            date: self.date.clone(),
        })
    }
}

/// The build identity a `llama-quantize` entry records — what the byte-parity
/// claim is conditional on (`docs/GGUF-TOOLING.md` §4.2: compiler + effective
/// `-ffp-contract` + llama.cpp revision).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedBuild {
    pub compiler: String,
    pub ffp_contract: String,
    pub llamacpp_commit: String,
    pub box_label: String,
    pub date: String,
}

impl RecordedBuild {
    /// One phrase a human can act on, for a gate's verdict.
    pub fn describe(&self) -> String {
        format!(
            "{} with `-ffp-contract={}` from llama.cpp {} ({}, {})",
            self.compiler, self.ffp_contract, self.llamacpp_commit, self.box_label, self.date
        )
    }
}

/// What the record says about one resolved reference file (issue #349).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReferenceRecord {
    /// The file's digest matched a recorded content for its path.
    pub content_recorded: bool,
    /// The build that content was produced by (`None` for a non-`llama-quantize`
    /// entry, or when the path is not a recorded fixture).
    pub matched: Option<RecordedBuild>,
    /// That content is the entry the byte-parity claim is asserted against.
    pub matched_authoritative: bool,
    /// The build the claim *is* asserted against for this path, if recorded.
    pub authoritative: Option<RecordedBuild>,
}

/// The F6 byte-parity gate's verdict, decided from the three byte comparisons —
/// `Fast` against the reference, the uncontracted model against the reference,
/// and the reference's digest against the record — plus which entry it matched
/// (issue #349).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityVerdict {
    /// minfer's `Fast` model reproduces the reference byte-for-byte.
    Reproduces,
    /// Only the uncontracted model matches: the reference is a build without
    /// cross-statement FMA contraction.
    FlagMismatch,
    /// Neither model matches, but the digest is a recorded content that is *not*
    /// the authoritative reference: the file is honest and it is a different
    /// compiler's build of the same source. Not an encoder defect.
    RecordedForeignBuild,
    /// Neither model matches and the digest *is* the authoritative reference: the
    /// encoder no longer reproduces the reference the claim is asserted against.
    AuthoritativeNotReproduced,
    /// Neither model matches and the digest is no recorded content at all: a
    /// corrupted, replaced or deliberately perturbed file looks exactly like
    /// this, so nothing may be blamed on a compiler.
    Unattributable,
}

impl ParityVerdict {
    /// The classifier itself: pure, total, and testable without a cache or a
    /// reference binary (the unit test at the bottom of this module).
    pub fn classify(fast_matches: bool, off_matches: bool, record: &ReferenceRecord) -> Self {
        if fast_matches {
            return Self::Reproduces;
        }
        if off_matches {
            return Self::FlagMismatch;
        }
        if !record.content_recorded {
            return Self::Unattributable;
        }
        if record.matched_authoritative {
            Self::AuthoritativeNotReproduced
        } else {
            Self::RecordedForeignBuild
        }
    }
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema: u32,
    cache_root: String,
    entries: Vec<Entry>,
}

/// Serialises the `MINFER_F6_MANIFEST` window against every reader (issue #354).
///
/// The environment is process-wide, so a test that points the override at a
/// fabricated record has a window in which *any* concurrent reader would resolve
/// the wrong file — and, on the malformed arm, panic. Every reader therefore
/// resolves through a guard, and the swapping test holds the same guard for its
/// whole body. It is reentrant on its own thread because that test necessarily
/// calls the entry points that take it; the repo's preference for an explicit
/// argument over a mutated environment (#185) cannot reach here, because the
/// environment variable *is* the feature.
static OVERRIDE_LOCK: Mutex<()> = Mutex::new(());
thread_local! {
    static OVERRIDE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

struct OverrideGuard {
    _held: Option<MutexGuard<'static, ()>>,
}

impl OverrideGuard {
    fn acquire() -> Self {
        let outermost = OVERRIDE_DEPTH.with(|d| {
            let was = d.get();
            d.set(was + 1);
            was == 0
        });
        Self {
            _held: if outermost {
                Some(OVERRIDE_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
            } else {
                None
            },
        }
    }
}

impl Drop for OverrideGuard {
    fn drop(&mut self) {
        OVERRIDE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

fn resolve_lock<R>(f: impl FnOnce() -> R) -> R {
    let _guard = OverrideGuard::acquire();
    f()
}

/// The record path this process reads: `MINFER_F6_MANIFEST` when it is set, else
/// the checked-in [`MANIFEST`]. A value that is set but empty is refused rather
/// than read as "unset" — the operator who pointed the gate somewhere must never
/// get the checked-in record back without being told.
fn manifest_path() -> Result<PathBuf, String> {
    match std::env::var_os(MANIFEST_ENV) {
        None => Ok(PathBuf::from(MANIFEST)),
        Some(v) if v.is_empty() => Err(format!(
            "{MANIFEST_ENV} is set but empty. An empty override is refused, not read as \
             \"unset\": falling back to the checked-in {MANIFEST} while the operator believed \
             the gate was pointed elsewhere is exactly the silent wrong-file failure this \
             override exists to prevent (issue #354). Unset it, or name a path."
        )),
        Some(v) => Ok(PathBuf::from(v)),
    }
}

/// Parse one manifest, refusing by name and by reason.
///
/// No failure falls back to the checked-in record: the gate compares against the
/// record it was pointed at, or it does not run (issue #354). The message always
/// names the file and, when it came from the override, the variable that named it.
fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let what = if path == Path::new(MANIFEST) {
        format!("the checked-in F6 fixture manifest {MANIFEST}")
    } else {
        format!(
            "the F6 fixture manifest {MANIFEST_ENV} names ({})",
            path.display()
        )
    };
    let text = std::fs::read_to_string(path).map_err(|e| {
        format!("{what} is unreadable: {e} — refusing rather than reading {MANIFEST} instead")
    })?;
    let m: Manifest = serde_json::from_str(&text).map_err(|e| {
        format!("{what} does not parse: {e} — refusing rather than reading {MANIFEST} instead")
    })?;
    if m.schema != 1 {
        return Err(format!(
            "{what} has schema {} — this reader knows 1",
            m.schema
        ));
    }
    Ok(m)
}

/// The record at `path`, parsed once per distinct path.
///
/// The cache is keyed on the path rather than being a single `OnceLock` because
/// the override must be *swappable*: a `OnceLock` freezes whichever path the
/// first caller resolved, which is what made the recorded-foreign-build verdict
/// untestable in-process (issue #354). A run that never sets the variable holds
/// exactly one entry, so the ordinary path is unchanged.
fn manifest_at(path: &Path) -> &'static Manifest {
    static MANIFEST_CACHE: Mutex<Vec<(PathBuf, &'static Manifest)>> = Mutex::new(Vec::new());
    let mut cache = MANIFEST_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, m)) = cache.iter().find(|(p, _)| p.as_path() == path) {
        return *m;
    }
    let m: &'static Manifest = Box::leak(Box::new(
        load_manifest(path).unwrap_or_else(|e| panic!("{e}")),
    ));
    cache.push((path.to_path_buf(), m));
    m
}

/// The record the resolver is using right now: the path the override names (else
/// the checked-in one), its parsed entries and the cache root they are judged
/// against — resolved **together** under the override lock, so a concurrent swap
/// cannot pair one path with another's entries or root (issue #354).
struct Record {
    manifest_path: PathBuf,
    manifest: &'static Manifest,
    cache_root: PathBuf,
}

fn current_record() -> Record {
    resolve_lock(|| {
        let manifest_path = manifest_path().unwrap_or_else(|e| panic!("{e}"));
        let manifest = manifest_at(&manifest_path);
        let cache_root = cache_root(manifest);
        Record {
            manifest_path,
            manifest,
            cache_root,
        }
    })
}

/// The fixture cache root: `MINFER_F6_CACHE` when set, else the manifest's own.
fn cache_root(m: &Manifest) -> PathBuf {
    let raw = std::env::var(CACHE_ENV).unwrap_or_else(|_| m.cache_root.clone());
    expand_home(&raw)
}

fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(p)
}

/// `path` relative to `root`, or `None` when it is not inside it. Canonicalising
/// both ends is what makes a symlinked cache and a relative override work.
fn cache_relative(path: &Path, root: &Path) -> Option<String> {
    let p = path.canonicalize().ok()?;
    let r = root.canonicalize().ok()?;
    let rel = p.strip_prefix(r).ok()?;
    Some(rel.to_string_lossy().replace('\\', "/"))
}

/// The entries that describe `path`; empty when no entry has its cache path.
fn entries_for<'a>(m: &'a Manifest, rel: Option<&str>) -> Vec<&'a Entry> {
    match rel {
        Some(rel) => m.entries.iter().filter(|e| e.path == rel).collect(),
        None => Vec::new(),
    }
}

/// The recorded digests for one path, for a message a human has to act on.
fn recorded(entries: &[&Entry]) -> String {
    entries
        .iter()
        .map(|e| match (&e.sha256, &e.sha256_prefix) {
            (Some(h), _) => format!("{h} ({} {})", e.box_label, e.date),
            (None, Some(p)) => format!("{p}… truncated ({}, {})", e.box_label, e.date),
            (None, None) => "no digest".to_string(),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Which recorded content a resolved reference is, for the parity gate's verdict
/// classifier (issue #349).
///
/// The caller has already run [`check`] on the path, so a file the record does
/// not describe has been refused with its digests; this only answers *which*
/// content the file is and which one the claim is asserted against. A path
/// outside the cache, or one whose digest matches nothing, yields the default
/// (nothing recorded) — the arms the caller turns into "unattributable" and the
/// resolver turns into a refusal respectively.
pub fn reference_record(path: &Path) -> ReferenceRecord {
    let r = current_record();
    reference_record_in(path, r.manifest, &r.cache_root)
}

fn reference_record_in(path: &Path, m: &Manifest, root: &Path) -> ReferenceRecord {
    let Some(rel) = cache_relative(path, root) else {
        return ReferenceRecord::default();
    };
    let entries = entries_for(m, Some(&rel));
    if entries.is_empty() {
        return ReferenceRecord::default();
    }
    let Ok(actual) = sha256_file(path) else {
        return ReferenceRecord::default();
    };
    let matched = entries
        .iter()
        .find(|e| e.sha256.as_deref() == Some(actual.as_str()))
        .or_else(|| {
            entries.iter().find(|e| {
                e.sha256_prefix
                    .as_deref()
                    .is_some_and(|p| actual.starts_with(p))
            })
        });
    let Some(matched) = matched else {
        return ReferenceRecord::default();
    };
    ReferenceRecord {
        content_recorded: true,
        matched: matched.build(),
        matched_authoritative: matched.authoritative_reference == Some(true),
        authoritative: entries
            .iter()
            .find(|e| e.authoritative_reference == Some(true))
            .and_then(|e| e.build()),
    }
}

/// Verify one resolved fixture, returning a weak note when only a truncated
/// digest could be matched.
///
/// `Ok(None)` — not a fixture this module judges (outside the cache, or a
/// directory). `Err(message)` — the caller must refuse; the message names the
/// file, every recorded digest, the actual one and the record it was judged
/// against.
pub fn verify(path: &Path) -> Result<Option<String>, String> {
    let r = current_record();
    verify_in(path, r.manifest, &r.cache_root, &r.manifest_path)
}

fn verify_in(
    path: &Path,
    m: &Manifest,
    root: &Path,
    manifest_path: &Path,
) -> Result<Option<String>, String> {
    if path.is_dir() {
        return Ok(None);
    }
    let rel = cache_relative(path, root);
    let entries = entries_for(m, rel.as_deref());
    if entries.is_empty() {
        if rel.is_none() {
            return Ok(None);
        }
        return Err(format!(
            "F6 fixture cache: {} is not in {} — nothing records what produced it, so a gate \
             comparing against it would certify the wrong bytes (issue #205). Move it out of {} \
             or record it (path, bytes, sha256, producer, date, box).",
            path.display(),
            manifest_path.display(),
            root.display()
        ));
    }
    let actual = sha256_file(path)
        .map_err(|e| format!("F6 fixture cache: {} is unreadable: {e}", path.display()))?;
    let size = std::fs::metadata(path).map(|md| md.len()).unwrap_or(0);
    for e in &entries {
        if e.sha256.as_deref() == Some(actual.as_str()) {
            if let Some(b) = e.bytes {
                if b != size {
                    return Err(format!(
                        "F6 fixture cache: {} has sha256 {actual} but is {size} bytes; the entry \
                         for {} says {b}",
                        path.display(),
                        e.path
                    ));
                }
            }
            return Ok(None);
        }
    }
    let sized: Vec<u64> = entries.iter().filter_map(|e| e.bytes).collect();
    if !sized.is_empty() && !sized.contains(&size) {
        return Err(format!(
            "F6 fixture cache: {} is {size} bytes; the record has {sized:?} for {} (recorded: {}; \
             actual: {actual}) — issue #205",
            path.display(),
            entries[0].path,
            recorded(&entries)
        ));
    }
    for e in &entries {
        if let Some(p) = e.sha256_prefix.as_deref() {
            if actual.starts_with(p) {
                return Ok(Some(format!(
                    "{} matches only the recorded prefix {p}… ({} {}) — the 2026-10-07 table \
                     truncated this digest, so the content is checked to 32 bits; re-capture the \
                     full sha256 on that box (issue #342)",
                    path.display(),
                    e.box_label,
                    e.date
                )));
            }
        }
    }
    Err(format!(
        "F6 fixture cache: {} is not the content {} records (actual sha256 {actual}; recorded: \
         {}). A stale or replaced fixture would otherwise be compared against silently (issue \
         #205): re-run the recipe in docs/GGUF-TOOLING.md §4.2.1, or keep the experiment outside \
         the cache root.",
        path.display(),
        manifest_path.display(),
        recorded(&entries)
    ))
}

/// [`verify`] as the fixture resolver uses it: panic on a refusal, report a weak
/// match once. A test's failure is the point — the alternative is a silently
/// wrong comparison.
pub fn check(path: &Path) {
    let r = current_record();
    check_in(path, r.manifest, &r.cache_root, &r.manifest_path);
}

fn check_in(path: &Path, m: &Manifest, root: &Path, manifest_path: &Path) {
    match verify_in(path, m, root, manifest_path) {
        Ok(None) => {}
        Ok(Some(note)) => eprintln!("[f6 fixtures] WEAK {note}"),
        Err(msg) => panic!("{msg}"),
    }
}

/// The manifest is readable and every entry carries what this reader uses, so a
/// schema drift between the checked-in record and this reader fails in CI rather
/// than at the first fixture lookup. The full shape audit is
/// `scripts/check_f6_fixtures.py` (CI `check-docs`).
#[test]
fn f6_fixture_manifest_parses_and_every_entry_has_an_identity() {
    // The **checked-in** record, never the override: this audit is about the
    // tracked file, and `scripts/check_f6_fixtures.py` audits the same one.
    let m = manifest_at(Path::new(MANIFEST));
    assert!(!m.entries.is_empty());
    for e in &m.entries {
        assert!(!e.path.is_empty() && !e.path.starts_with('/'), "{e:?}");
        assert!(
            e.sha256.is_some() != e.sha256_prefix.is_some(),
            "{}: exactly one digest form",
            e.path
        );
        if let Some(h) = &e.sha256 {
            assert_eq!(h.len(), 64, "{}: a sha256 is 64 hex chars", e.path);
            assert!(h
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            assert!(e.bytes.is_some(), "{}: a full digest needs bytes", e.path);
        }
        if let Some(p) = &e.sha256_prefix {
            assert_eq!(p.len(), 8, "{}: a prefix is 8 hex chars", e.path);
        }
        assert!(
            e.box_label.contains('('),
            "{}: box label {:?}",
            e.path,
            e.box_label
        );
        assert!(
            e.date.len() == 10 && e.date.as_bytes()[4] == b'-',
            "{}: date {:?}",
            e.path,
            e.date
        );
        // #349: a record that names a llama-quantize producer must carry the
        // build identity the claim is conditional on, and a file whose content
        // the claim is asserted against is never silently unmarked.
        if e.producer_kind.as_deref() == Some("llama-quantize") {
            assert!(
                e.compiler.is_some() && e.ffp_contract.is_some(),
                "{}: a llama-quantize entry needs compiler + ffp_contract",
                e.path
            );
        }
        if let Some(true) = e.authoritative_reference {
            assert_eq!(
                e.producer_kind.as_deref(),
                Some("llama-quantize"),
                "{}: only a llama-quantize entry can be the authoritative reference",
                e.path
            );
            assert_eq!(
                e.ffp_contract.as_deref(),
                Some("fast"),
                "{}: the authoritative reference is the contracting build",
                e.path
            );
        }
    }
    // Every path a llama-quantize entry describes has exactly one authoritative
    // reference (the check `scripts/check_f6_fixtures.py` S6 also makes).
    let mut paths: Vec<&str> = m
        .entries
        .iter()
        .filter(|e| e.producer_kind.as_deref() == Some("llama-quantize"))
        .map(|e| e.path.as_str())
        .collect();
    paths.sort_unstable();
    paths.dedup();
    for p in paths {
        let marks = m
            .entries
            .iter()
            .filter(|e| e.path == p && e.authoritative_reference == Some(true))
            .count();
        assert_eq!(marks, 1, "{p}: exactly one authoritative_reference entry");
    }
}

/// A path outside the cache root is not a fixture: `/tmp` scratch, the model
/// cache and a directory all keep the old behaviour. This is the arm that keeps
/// #334's deliberate `/tmp` reference experiments working.
#[test]
fn f6_fixture_verification_leaves_paths_outside_the_cache_alone() {
    // Through the public entry point, so this also covers the root resolution.
    assert!(matches!(
        verify(Path::new("/tmp/no-such-fixture.gguf")),
        Ok(None)
    ));
    assert!(matches!(verify(Path::new("/")), Ok(None)));
    assert!(matches!(verify(&std::env::temp_dir()), Ok(None)));
}

/// A tampered fixture inside the cache is refused, and the refusal names the file
/// and the digests a human needs. The manifest is fabricated with a 4-byte
/// fixture — a real one is ~1 GB and cannot be rewritten to its recorded size in
/// a unit test — so the arm under test is the *digest*, not the size. The real
/// cache is never written to.
#[test]
fn f6_fixture_a_tampered_cached_reference_is_refused_by_name_and_digest() {
    let m = Manifest {
        schema: 1,
        cache_root: "~/.cache/minfer/f6-src".to_string(),
        entries: vec![Entry {
            path: "ref/tiny.gguf".to_string(),
            bytes: Some(4),
            // sha256("tiny"), computed independently of this module.
            sha256: Some("8950abfda7b727630760dd35bcf5c3daa7631aff223a90f7728c0d2521dde10c".into()),
            sha256_prefix: None,
            box_label: "dgxspark (aarch64, GB10 sm_121)".to_string(),
            date: "2026-10-07".to_string(),
            producer_kind: Some("llama-quantize".to_string()),
            compiler: Some("gcc 13.3.0".to_string()),
            ffp_contract: Some("fast".to_string()),
            llamacpp_commit: Some("deadbeef".to_string()),
            authoritative_reference: Some(true),
        }],
    };
    let root = std::env::temp_dir().join(format!("f6-fixture-refusal-{}", std::process::id()));
    let target = root.join("ref/tiny.gguf");
    // The fabricated record's own label, for the refusal messages.
    let record = Path::new("fabricated-f6-fixtures.json");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    // The recorded file verifies...
    std::fs::write(&target, b"tiny").unwrap();
    assert!(matches!(verify_in(&target, &m, &root, record), Ok(None)));
    // ... and one flipped byte is refused by name, actual digest and record.
    std::fs::write(&target, b"tinv").unwrap();
    // The resolver the gates actually reach (`env_path` -> `check`) must refuse
    // too, not just the inner function: a `check` that returned quietly would
    // leave every F6 gate comparing against a tampered file.
    let panicked = std::panic::catch_unwind(|| {
        check_in(&target, &m, &root, record);
    })
    .is_err();
    assert!(
        panicked,
        "the fixture resolver must panic on a tampered file"
    );
    let err = verify_in(&target, &m, &root, record)
        .expect_err("a tampered cache fixture must be refused");
    assert!(err.contains(&target.display().to_string()), "{err}");
    assert!(err.contains("actual sha256"), "{err}");
    assert!(err.contains("issue #205"), "{err}");
    assert!(err.contains(&m.entries[0].box_label), "{err}");
    assert!(err.contains(&m.entries[0].sha256.clone().unwrap()), "{err}");
    // A file the manifest does not name, but which sits inside the cache root,
    // is refused too: an unrecorded reference is the same hazard as a stale one.
    let stray = root.join("ref/unrecorded.gguf");
    std::fs::write(&stray, b"tiny").unwrap();
    let err =
        verify_in(&stray, &m, &root, record).expect_err("an unrecorded cache file must be refused");
    assert!(err.contains("is not in"), "{err}");
    std::fs::remove_dir_all(&root).ok();
}

/// The verdict classifier of issue #349 decides from the three byte comparisons
/// — `Fast` against the reference, the uncontracted model against it, and the
/// file's digest against the record — and its two load-bearing arms must not be
/// confused: a recorded *foreign* build is not an encoder defect, and a file no
/// entry describes is neither.
#[test]
fn the_f6_parity_verdict_names_the_recorded_build() {
    let foreign = RecordedBuild {
        compiler: "Apple clang 21.0.0 (Xcode 27.0)".to_string(),
        ffp_contract: "fast".to_string(),
        llamacpp_commit: "c479922ac".to_string(),
        box_label: "macbook (macOS 27.0.1, Apple M4 Pro)".to_string(),
        date: "2026-10-07".to_string(),
    };
    let authoritative = RecordedBuild {
        compiler: "gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)".to_string(),
        ffp_contract: "fast".to_string(),
        llamacpp_commit: "unrecorded".to_string(),
        box_label: "dgxspark (aarch64, GB10 sm_121)".to_string(),
        date: "2026-09-27".to_string(),
    };
    // The pure classifier: no cache, no reference binary, no encoding.
    assert_eq!(
        ParityVerdict::classify(true, true, &ReferenceRecord::default()),
        ParityVerdict::Reproduces
    );
    // Only the uncontracted model matches: the reference build's flag (PR #340).
    assert_eq!(
        ParityVerdict::classify(false, true, &ReferenceRecord::default()),
        ParityVerdict::FlagMismatch
    );
    let recorded_foreign = ReferenceRecord {
        content_recorded: true,
        matched: Some(foreign.clone()),
        matched_authoritative: false,
        authoritative: Some(authoritative.clone()),
    };
    assert_eq!(
        ParityVerdict::classify(false, false, &recorded_foreign),
        ParityVerdict::RecordedForeignBuild
    );
    let recorded_authoritative = ReferenceRecord {
        content_recorded: true,
        matched: Some(authoritative.clone()),
        matched_authoritative: true,
        authoritative: Some(authoritative.clone()),
    };
    assert_eq!(
        ParityVerdict::classify(false, false, &recorded_authoritative),
        ParityVerdict::AuthoritativeNotReproduced
    );
    // A /tmp copy the record does not name: the manifest cannot attribute the
    // mismatch, so it must NOT be reported as a different compiler.
    assert_eq!(
        ParityVerdict::classify(false, false, &ReferenceRecord::default()),
        ParityVerdict::Unattributable
    );
    assert!(foreign.describe().contains("Apple clang 21.0.0"));
    assert!(foreign.describe().contains("-ffp-contract=fast"));

    // The digest half of the classifier: a two-content record tells the two
    // builds apart, and the authoritative mark says which one the claim is
    // asserted against. The fixture is 4 bytes — a real one is ~350 MB — so the
    // arm under test is the digest lookup, and the real cache is untouched.
    let m = Manifest {
        schema: 1,
        cache_root: "~/.cache/minfer/f6-src".to_string(),
        entries: vec![
            Entry {
                path: "ref/tiny.gguf".to_string(),
                bytes: Some(4),
                // sha256("tiny") / sha256("tinv"), computed independently.
                sha256: Some(
                    "8950abfda7b727630760dd35bcf5c3daa7631aff223a90f7728c0d2521dde10c".to_string(),
                ),
                sha256_prefix: None,
                box_label: authoritative.box_label.clone(),
                date: authoritative.date.clone(),
                producer_kind: Some("llama-quantize".to_string()),
                compiler: Some(authoritative.compiler.clone()),
                ffp_contract: Some("fast".to_string()),
                llamacpp_commit: Some("unrecorded".to_string()),
                authoritative_reference: Some(true),
            },
            Entry {
                path: "ref/tiny.gguf".to_string(),
                bytes: Some(4),
                sha256: Some(
                    "c53fe36c10e45164b7c80362114a9230cc8421eeeddb8fb9547063ad4b2065fd".to_string(),
                ),
                sha256_prefix: None,
                box_label: foreign.box_label.clone(),
                date: foreign.date.clone(),
                producer_kind: Some("llama-quantize".to_string()),
                compiler: Some(foreign.compiler.clone()),
                ffp_contract: Some("fast".to_string()),
                llamacpp_commit: Some("c479922ac".to_string()),
                authoritative_reference: Some(false),
            },
        ],
    };
    let root = std::env::temp_dir().join(format!("f6-reference-record-{}", std::process::id()));
    let target = root.join("ref/tiny.gguf");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    // The authoritative content: matched, and marked authoritative.
    std::fs::write(&target, b"tiny").unwrap();
    let r = reference_record_in(&target, &m, &root);
    assert!(r.content_recorded, "{r:?}");
    assert_eq!(r.matched.as_ref(), Some(&authoritative), "{r:?}");
    assert!(r.matched_authoritative, "{r:?}");
    // The other recorded content: still recorded, but a *foreign* build — the
    // arm the gate turns into a loud skip rather than an encoder defect.
    std::fs::write(&target, b"tinv").unwrap();
    let r = reference_record_in(&target, &m, &root);
    assert!(r.content_recorded, "{r:?}");
    assert_eq!(r.matched.as_ref(), Some(&foreign), "{r:?}");
    assert!(!r.matched_authoritative, "{r:?}");
    assert_eq!(r.authoritative.as_ref(), Some(&authoritative), "{r:?}");
    // A file the record does not describe: nothing is attributed to it.
    let stray = root.join("ref/stray.gguf");
    std::fs::write(&stray, b"tiny").unwrap();
    let r = reference_record_in(&stray, &m, &root);
    assert!(!r.content_recorded && r.matched.is_none(), "{r:?}");
    // And a path outside the cache root is not a fixture at all.
    let outside = std::env::temp_dir().join("f6-reference-record-outside.gguf");
    std::fs::write(&outside, b"tiny").unwrap();
    assert_eq!(
        reference_record_in(&outside, &m, &root),
        ReferenceRecord::default()
    );
    std::fs::remove_dir_all(&root).ok();
    let _ = std::fs::remove_file(&outside);
}

/// Set `key` to `value` for the duration of `f`, restoring the previous value on
/// every exit path — including a panic. The caller holds an [`OverrideGuard`], so
/// no other reader can observe the half-swapped environment.
fn with_env_var<R>(key: &str, value: &OsStr, f: impl FnOnce() -> R) -> R {
    struct Restore {
        key: String,
        previous: Option<OsString>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }
    let _restore = Restore {
        key: key.to_string(),
        previous: std::env::var_os(key),
    };
    std::env::set_var(key, value);
    f()
}

/// Issue #354: the whole resolver — `check` → `reference_record` → `classify`,
/// the path an F6 gate reaches through `env_path` — runs against a fabricated
/// record named by `MINFER_F6_MANIFEST`. The arm this buys back is
/// `ParityVerdict::RecordedForeignBuild`: a reference whose digest *is* recorded
/// but is not the path's `authoritative_reference`, which #349 could only
/// exercise by editing the tracked manifest under a `cp` backup.
///
/// The record is the same two-content shape #349's classifier test builds and the
/// fixture is 4 bytes (a real one is ~350 MB), so the arms under test are the
/// override, the path resolution and the digest lookup — not the file size. The
/// real cache is never read or written.
#[test]
fn the_manifest_override_drives_the_whole_resolver() {
    let _override = OverrideGuard::acquire();
    let tmp = std::env::temp_dir().join(format!("f6-manifest-override-{}", std::process::id()));
    let root = tmp.join("cache");
    let manifest = tmp.join("f6-fixtures.json");
    // sha256("tiny") — the authoritative content, a `gcc` `-ffp-contract=fast`
    // build; sha256("tinv") — the recorded *foreign* content, an Apple clang build
    // of the same source. Both computed independently of this module.
    const AUTHORITATIVE_SHA: &str =
        "8950abfda7b727630760dd35bcf5c3daa7631aff223a90f7728c0d2521dde10c";
    const FOREIGN_SHA: &str = "c53fe36c10e45164b7c80362114a9230cc8421eeeddb8fb9547063ad4b2065fd";
    let doc = serde_json::json!({
        "schema": 1,
        "cache_root": root.display().to_string(),
        "entries": [
            {
                "path": "ref/tiny.gguf",
                "bytes": 4,
                "sha256": AUTHORITATIVE_SHA,
                "box": "dgxspark (aarch64, GB10 sm_121)",
                "date": "2026-10-07",
                "producer_kind": "llama-quantize",
                "compiler": "gcc 13.3.0",
                "ffp_contract": "fast",
                "llamacpp_commit": "unrecorded",
                "authoritative_reference": true,
            },
            {
                "path": "ref/tiny.gguf",
                "bytes": 4,
                "sha256": FOREIGN_SHA,
                "box": "macbook (macOS 27.0.1, Apple M4 Pro)",
                "date": "2026-10-07",
                "producer_kind": "llama-quantize",
                "compiler": "Apple clang 21.0.0 (Xcode 27.0)",
                "ffp_contract": "fast",
                "llamacpp_commit": "c479922ac",
                "authoritative_reference": false,
            },
        ],
    });
    std::fs::create_dir_all(root.join("ref")).unwrap();
    std::fs::write(&manifest, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    let target = root.join("ref/tiny.gguf");

    with_env_var(MANIFEST_ENV, manifest.as_os_str(), || {
        with_env_var(CACHE_ENV, root.as_os_str(), || {
            // 1. The authoritative content: recorded, and marked as such. A
            //    mismatch against it is an encoder defect, never excused.
            std::fs::write(&target, b"tiny").unwrap();
            let verified = verify(&target);
            assert!(
                matches!(verified, Ok(None)),
                "the authoritative content must verify against the override: {verified:?}"
            );
            let r = reference_record(&target);
            assert!(r.content_recorded && r.matched_authoritative, "{r:?}");
            assert_eq!(r.matched.as_ref().unwrap().compiler, "gcc 13.3.0");
            assert_eq!(
                ParityVerdict::classify(false, false, &r),
                ParityVerdict::AuthoritativeNotReproduced
            );

            // 2. The recorded *foreign* build — the verdict #349 could only fake
            //    by editing the tracked manifest. It is recorded, so it is honest;
            //    it is not authoritative, so it is not a defect: the gate turns
            //    this into a loud skip and exits 0.
            std::fs::write(&target, b"tinv").unwrap();
            let verified = verify(&target);
            assert!(
                matches!(verified, Ok(None)),
                "a recorded content verifies whatever build it is: {verified:?}"
            );
            let r = reference_record(&target);
            assert!(r.content_recorded, "{r:?}");
            assert!(!r.matched_authoritative, "{r:?}");
            assert_eq!(
                r.matched.as_ref().unwrap().compiler,
                "Apple clang 21.0.0 (Xcode 27.0)"
            );
            assert_eq!(r.authoritative.as_ref().unwrap().compiler, "gcc 13.3.0");
            assert_eq!(
                ParityVerdict::classify(false, false, &r),
                ParityVerdict::RecordedForeignBuild
            );
            // Both builds are in the record, which is what the skip line names.
            assert!(r
                .matched
                .as_ref()
                .unwrap()
                .describe()
                .contains("Apple clang"));
            assert!(r
                .authoritative
                .as_ref()
                .unwrap()
                .describe()
                .contains("gcc 13.3.0"));

            // 3. A file inside the cache root the record does not name: refused,
            //    and nothing is attributed to it.
            let stray = root.join("ref/stray.gguf");
            std::fs::write(&stray, b"tiny").unwrap();
            let err = verify(&stray).expect_err("an unrecorded cache file is refused");
            assert!(err.contains(&stray.display().to_string()), "{err}");
            assert!(err.contains(&manifest.display().to_string()), "{err}");
            assert!(err.contains("is not in"), "{err}");
            assert_eq!(
                ParityVerdict::classify(false, false, &reference_record(&stray)),
                ParityVerdict::Unattributable
            );

            // 4. A tampered copy of a recorded content: refused by name, digest
            //    and the record it was judged against.
            std::fs::write(&target, b"tanz").unwrap();
            let err = verify(&target).expect_err("a tampered cache fixture is refused");
            assert!(err.contains(&target.display().to_string()), "{err}");
            assert!(err.contains("actual sha256"), "{err}");
            assert!(err.contains(&manifest.display().to_string()), "{err}");
            assert_eq!(
                ParityVerdict::classify(false, false, &reference_record(&target)),
                ParityVerdict::Unattributable
            );
        });
    });

    // The default is unaffected: with the override gone the resolver reads the
    // checked-in record again. The parse cache is keyed on the path, so this is a
    // fresh read of the tracked file and not a stale hit on the fabricated one.
    let r = current_record();
    assert_eq!(r.manifest_path, PathBuf::from(MANIFEST));
    assert_eq!(r.manifest.schema, 1);
    assert!(!r.manifest.entries.is_empty());
    std::fs::remove_dir_all(&tmp).ok();
}

/// An override is honoured **loudly** (issue #354): a value that is set but
/// empty, a path that does not exist, and a file that is not JSON each refuse by
/// name — the reader never falls back to the checked-in record, because a gate
/// quietly comparing against a different manifest than the operator pointed it
/// at is worse than one that cannot be pointed anywhere.
#[test]
fn a_broken_manifest_override_refuses_by_name_instead_of_falling_back() {
    let _override = OverrideGuard::acquire();
    let tmp = std::env::temp_dir().join(format!("f6-manifest-broken-{}", std::process::id()));
    let missing = tmp.join("missing.json");
    let malformed = tmp.join("malformed.json");
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(&malformed, b"{ not json").unwrap();

    // An empty value is refused before any file is touched: it is not "unset".
    with_env_var(MANIFEST_ENV, OsStr::new(""), || {
        let err = manifest_path().expect_err("an empty override is refused");
        assert!(err.contains(MANIFEST_ENV), "{err}");
        assert!(err.contains("empty"), "{err}");
        assert!(err.contains(MANIFEST), "{err}");
    });

    // A path that does not exist, and a file that does not parse: both name the
    // override, the path and the problem.
    let err = match load_manifest(&missing) {
        Ok(m) => panic!("a missing override fell back: {} entries", m.entries.len()),
        Err(e) => e,
    };
    assert!(err.contains(MANIFEST_ENV), "{err}");
    assert!(err.contains(&missing.display().to_string()), "{err}");
    assert!(err.contains("unreadable"), "{err}");
    let err = match load_manifest(&malformed) {
        Ok(m) => panic!(
            "a malformed override fell back: {} entries",
            m.entries.len()
        ),
        Err(e) => e,
    };
    assert!(err.contains(MANIFEST_ENV), "{err}");
    assert!(err.contains(&malformed.display().to_string()), "{err}");
    assert!(err.contains("does not parse"), "{err}");

    // Through the resolver the gates reach: no silent fall back — the run refuses
    // instead of comparing against the checked-in record.
    with_env_var(MANIFEST_ENV, missing.as_os_str(), || {
        let refused = std::panic::catch_unwind(|| {
            let _ = verify(Path::new("/tmp/no-such-fixture.gguf"));
        })
        .is_err();
        assert!(refused, "a missing override must refuse, not fall back");
        let refused = std::panic::catch_unwind(|| {
            let _ = reference_record(Path::new("/tmp/no-such-fixture.gguf"));
        })
        .is_err();
        assert!(refused, "a missing override must refuse, not fall back");
        assert_eq!(
            manifest_path().unwrap(),
            missing,
            "the override, not the checked-in record, is what the resolver reads"
        );
    });

    // ... and the checked-in record is reachable again once the override is gone.
    let r = current_record();
    assert_eq!(r.manifest_path, PathBuf::from(MANIFEST));
    std::fs::remove_dir_all(&tmp).ok();
}
