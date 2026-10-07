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
//! cache fixture at all.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The checked-in record. `CARGO_MANIFEST_DIR` keeps it correct from a worktree.
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/f6-fixtures.json");

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
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema: u32,
    cache_root: String,
    entries: Vec<Entry>,
}

fn manifest() -> &'static Manifest {
    static MANIFEST_CACHE: OnceLock<Manifest> = OnceLock::new();
    MANIFEST_CACHE.get_or_init(|| {
        let text = std::fs::read_to_string(MANIFEST)
            .unwrap_or_else(|e| panic!("the F6 fixture manifest {MANIFEST} is unreadable: {e}"));
        let m: Manifest = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("the F6 fixture manifest {MANIFEST} does not parse: {e}"));
        assert_eq!(
            m.schema, 1,
            "the F6 fixture manifest's schema is {} — this reader knows 1",
            m.schema
        );
        m
    })
}

/// The fixture cache root: `MINFER_F6_CACHE` when set, else the manifest's own.
fn cache_root(m: &Manifest) -> PathBuf {
    let raw = std::env::var("MINFER_F6_CACHE").unwrap_or_else(|_| m.cache_root.clone());
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

/// Verify one resolved fixture, returning a weak note when only a truncated
/// digest could be matched.
///
/// `Ok(None)` — not a fixture this module judges (outside the cache, or a
/// directory). `Err(message)` — the caller must refuse; the message names the
/// file, every recorded digest and the actual one.
pub fn verify(path: &Path) -> Result<Option<String>, String> {
    verify_in(path, manifest(), &cache_root(manifest()))
}

fn verify_in(path: &Path, m: &Manifest, root: &Path) -> Result<Option<String>, String> {
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
            "F6 fixture cache: {} is not in {MANIFEST} — nothing records what produced it, so a \
             gate comparing against it would certify the wrong bytes (issue #205). Move it out of \
             {} or record it (path, bytes, sha256, producer, date, box).",
            path.display(),
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
        "F6 fixture cache: {} is not the content {MANIFEST} records (actual sha256 {actual}; \
         recorded: {}). A stale or replaced fixture would otherwise be compared against silently \
         (issue #205): re-run the recipe in docs/GGUF-TOOLING.md §4.2.1, or keep the experiment \
         outside the cache root.",
        path.display(),
        recorded(&entries)
    ))
}

/// [`verify`] as the fixture resolver uses it: panic on a refusal, report a weak
/// match once. A test's failure is the point — the alternative is a silently
/// wrong comparison.
pub fn check(path: &Path) {
    check_in(path, manifest(), &cache_root(manifest()));
}

fn check_in(path: &Path, m: &Manifest, root: &Path) {
    match verify_in(path, m, root) {
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
    let m = manifest();
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
        }],
    };
    let root = std::env::temp_dir().join(format!("f6-fixture-refusal-{}", std::process::id()));
    let target = root.join("ref/tiny.gguf");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    // The recorded file verifies...
    std::fs::write(&target, b"tiny").unwrap();
    assert!(matches!(verify_in(&target, &m, &root), Ok(None)));
    // ... and one flipped byte is refused by name, actual digest and record.
    std::fs::write(&target, b"tinv").unwrap();
    // The resolver the gates actually reach (`env_path` -> `check`) must refuse
    // too, not just the inner function: a `check` that returned quietly would
    // leave every F6 gate comparing against a tampered file.
    let panicked = std::panic::catch_unwind(|| {
        check_in(&target, &m, &root);
    })
    .is_err();
    assert!(
        panicked,
        "the fixture resolver must panic on a tampered file"
    );
    let err = verify_in(&target, &m, &root).expect_err("a tampered cache fixture must be refused");
    assert!(err.contains(&target.display().to_string()), "{err}");
    assert!(err.contains("actual sha256"), "{err}");
    assert!(err.contains("issue #205"), "{err}");
    assert!(err.contains(&m.entries[0].box_label), "{err}");
    assert!(err.contains(&m.entries[0].sha256.clone().unwrap()), "{err}");
    // A file the manifest does not name, but which sits inside the cache root,
    // is refused too: an unrecorded reference is the same hazard as a stale one.
    let stray = root.join("ref/unrecorded.gguf");
    std::fs::write(&stray, b"tiny").unwrap();
    let err = verify_in(&stray, &m, &root).expect_err("an unrecorded cache file must be refused");
    assert!(err.contains("is not in"), "{err}");
    std::fs::remove_dir_all(&root).ok();
}
