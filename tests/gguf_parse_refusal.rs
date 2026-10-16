//! A parse failure is a refusal, not a panic ([#580]).
//!
//! Fast tier: no model is required. The loader prints the specific reason and the
//! CLI exits non-zero without a panic backtrace — the panic was the whole point of
//! the issue, so the assertion that matters is the *absence* of `panicked`.
//!
//! [#580]: https://github.com/yusiwen/minfer/issues/580

use std::io::Write;
use std::process::Command;

/// Run `minfer info <path>` and return `(exit_code, stderr)`.
fn info(path: &std::path::Path) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_minfer"))
        .arg("info")
        .arg(path)
        .output()
        .expect("spawn minfer");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn a_non_gguf_file_is_refused_by_name_not_by_panic() {
    let dir = std::env::temp_dir().join("minfer-gguf-refusal");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("not-a-gguf.bin");
    let mut f = std::fs::File::create(&path).expect("write scratch file");
    f.write_all(b"this is not a GGUF file at all\n").unwrap();
    drop(f);

    let (code, stderr) = info(&path);
    assert_ne!(code, 0, "a non-GGUF file must exit non-zero");
    assert!(
        !stderr.contains("panicked"),
        "a parse failure must not panic; stderr was:\n{stderr}"
    );
    assert!(
        stderr.contains("GGUF:"),
        "the loader's specific reason must reach stderr; stderr was:\n{stderr}"
    );
    assert!(
        stderr.contains("not a GGUF file minfer can parse"),
        "the refusal must name the failure; stderr was:\n{stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_truncated_gguf_header_is_refused_by_name_not_by_panic() {
    let dir = std::env::temp_dir().join("minfer-gguf-refusal");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("truncated.gguf");
    // The magic and version, then nothing: the loader runs out of bytes while
    // reading the counts.
    let mut f = std::fs::File::create(&path).expect("write scratch file");
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    drop(f);

    let (code, stderr) = info(&path);
    assert_ne!(code, 0, "a truncated GGUF must exit non-zero");
    assert!(
        !stderr.contains("panicked"),
        "a truncated GGUF must not panic; stderr was:\n{stderr}"
    );
    assert!(
        stderr.contains("GGUF:") || stderr.contains("failed to read"),
        "the loader's reason must reach stderr; stderr was:\n{stderr}"
    );
    let _ = std::fs::remove_file(&path);
}
