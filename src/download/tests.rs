//! `#[cfg(test)] mod tests` for `src/download/mod.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::match_model;

fn single(name: &str) -> Vec<String> {
    vec![name.to_string()]
}
fn split(prefix: &str, count: usize) -> Vec<String> {
    (1..=count)
        .map(|i| format!("{}-{:05}-of-{:05}.gguf", prefix, i, count))
        .collect()
}

#[test]
fn match_single_file_quant() {
    let files = single("qwen2.5-0.5b-instruct-q4_0.gguf");
    assert_eq!(match_model(&files, Some("q4_0")).unwrap(), files);
}

#[test]
fn match_single_file_quant_case_insensitive() {
    let files = single("qwen2.5-0.5b-instruct-q4_0.gguf");
    assert_eq!(match_model(&files, Some("Q4_0")).unwrap(), files);
    assert_eq!(
        match_model(&files, Some("Q4_K_M"))
            .unwrap_err()
            .contains("not found"),
        true
    );
}

#[test]
fn match_split_quant() {
    let files = split("qwen2.5-7b-instruct-q4_k_m", 2);
    let got = match_model(&files, Some("q4_k_m")).unwrap();
    assert_eq!(got, files);
    assert_eq!(match_model(&files, Some("Q4_K_M")).unwrap(), files);
    assert_eq!(
        match_model(&files, Some("qwen2.5-7b-instruct-q4_k_m")).unwrap(),
        files
    );
}

#[test]
fn match_exact_filename_expands_split() {
    let files = split("foo", 3);
    let part0 = &files[0];
    let got = match_model(&files, Some(part0)).unwrap();
    assert_eq!(got, files); // whole group
}

#[test]
fn match_mixed_repo_ambiguous() {
    let mut files = Vec::new();
    files.extend(single("m1-q4_k_m.gguf"));
    files.extend(split("m2-q4_k_m", 2));
    files.extend(split("m2-q5_k_m", 2));
    // two different base names share the same quant tail → ambiguous
    let err = match_model(&files, Some("q4_k_m")).unwrap_err();
    assert!(err.contains("ambiguous"), "err: {err}");
    assert!(
        err.contains("m1-q4_k_m") && err.contains("m2-q4_k_m"),
        "err: {err}"
    );
    // unique
    assert_eq!(
        match_model(&files, Some("q5_k_m")).unwrap(),
        split("m2-q5_k_m", 2)
    );
    // not found
    assert!(match_model(&files, Some("q4_0"))
        .unwrap_err()
        .contains("not found"));
}

#[test]
fn match_no_requested() {
    // one model group (split) → all parts
    let files = split("m-q4_k_m", 2);
    assert_eq!(match_model(&files, None).unwrap(), files);
    // multiple models → error
    let mut files2 = Vec::new();
    files2.extend(single("m1-q4_0.gguf"));
    files2.extend(single("m2-q4_k_m.gguf"));
    assert!(match_model(&files2, None)
        .unwrap_err()
        .contains("Multiple models"));
}

// === F6 (#49): the download size gate ===

fn tmp(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("minfer-f6-dl-{name}"));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn a_wrong_size_file_is_refused_and_a_right_size_file_is_accepted() {
    use super::check_downloaded_size;
    let p = tmp("size");
    std::fs::write(&p, vec![7u8; 100]).unwrap();
    // exact match: accepted, length returned
    assert_eq!(check_downloaded_size(&p, Some(100)).unwrap(), 100);
    // too large (the classic `curl -C -` on a server that ignores Range):
    // the whole body is appended to the partial file.
    let e = check_downloaded_size(&p, Some(40)).unwrap_err();
    assert!(e.contains("expected 40 bytes, got 100 bytes"), "{e}");
    // too small (a truncated body)
    let e = check_downloaded_size(&p, Some(400)).unwrap_err();
    assert!(e.contains("expected 400 bytes, got 100 bytes"), "{e}");
    // unknown remote size: length reported, not judged
    assert_eq!(check_downloaded_size(&p, None).unwrap(), 100);
    let _ = std::fs::remove_file(&p);
}

/// End-to-end against a local HTTP server. `correct = true` honours `Range`
/// with a proper `206 Partial Content`; `correct = false` is the
/// mishandled-Range server: it answers `206` but sends the *whole* body
/// (as if the range had started at 0), so `curl -C -` appends and the
/// result is larger than the object. No external network is touched.
fn serve_once(body: Vec<u8>, correct: bool) -> (String, std::thread::JoinHandle<()>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{}/model.gguf", addr);
    let h = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 4096];
        let n = sock.read(&mut buf).unwrap_or(0);
        let req = String::from_utf8_lossy(&buf[..n]).to_string();
        let range = req
            .lines()
            .find_map(|l| l.strip_prefix("Range: bytes="))
            .and_then(|v| v.trim().split('-').next())
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let (head, part) = if correct {
            let start = range.min(body.len());
            (
                format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                     Content-Range: bytes {}-{}/{}\r\nAccept-Ranges: bytes\r\n\r\n",
                    body.len() - start,
                    start,
                    body.len() - 1,
                    body.len()
                ),
                body[start..].to_vec(),
            )
        } else {
            // Answers with a *correct-looking* 206 header for the requested
            // range, but ships the whole object as the body. curl is happy
            // (Content-Length is honoured, exit 0) and appends, so the file
            // ends up larger than the object.
            let start = range.min(body.len());
            (
                format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                     Content-Range: bytes {}-{}/{}\r\nAccept-Ranges: bytes\r\n\r\n",
                    body.len(),
                    start,
                    body.len() - 1,
                    body.len()
                ),
                body.clone(),
            )
        };
        let _ = sock.write_all(head.as_bytes());
        let _ = sock.write_all(&part);
    });
    (url, h)
}

#[test]
fn http_download_resumes_a_partial_file_and_size_checks_it() {
    use super::http_download;
    let body: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
    let p = tmp("resume");
    // A partial file: the server honours Range, so the result is complete.
    let half = body.len() / 2;
    std::fs::write(&p, &body[..half]).unwrap();
    let (url, h) = serve_once(body.clone(), true);
    http_download(&url, &p, Some(body.len() as u64)).expect("resumable download");
    h.join().unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), body);

    // A server that mishandles Range: `curl -C -` exits 0 with a file that
    // is larger than the object. The size gate must reject it and remove it.
    let p2 = tmp("badrange");
    std::fs::write(&p2, &body[..half]).unwrap();
    let (url2, h2) = serve_once(body.clone(), false);
    let err = http_download(&url2, &p2, Some(body.len() as u64)).unwrap_err();
    h2.join().unwrap();
    assert!(err.contains("size mismatch"), "{err}");
    assert!(err.contains("removed the partial file"), "{err}");
    assert!(!p2.exists(), "a rejected download must not stay cached");
}
