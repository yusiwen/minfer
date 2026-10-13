//! `#[cfg(test)] mod tests` for `src/server/viz.rs` — the static half of the viz
//! server. It had no tests before #103: `check-viz` runs the page's JS gates and
//! nothing read the routes, so the `no-store` promise the asset-freshness fix
//! leans on was one comment. See the parent module for the routes.

use super::*;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// An `AppState` with no model, and an empty `/viz/graph`: the static routes never
/// touch either. `Tokenizer::empty` is the same stand-in `server::tests` uses, and
/// the job receiver must stay alive or a later `job_tx.send` would fail for a
/// different reason than the one under test.
fn state() -> (Arc<VizState>, mpsc::Receiver<Job>) {
    let (job_tx, job_rx) = mpsc::channel::<Job>(4);
    let app = Arc::new(AppState {
        job_tx,
        metrics: Arc::new(crate::server::metrics::ServerMetrics::new()),
        model_name: "test-model".to_string(),
        n_ctx: 64,
        n_ctx_slot: 64,
        tokenizer: Arc::new(Tokenizer::empty()),
        special: crate::models::SpecialTokens {
            eos: 0,
            im_end: None,
        },
        spec_enabled: false,
        chat_template: None,
        created: 0,
    });
    (
        Arc::new(VizState {
            app,
            graphs: Arc::new(json!({})),
        }),
        job_rx,
    )
}

/// Bind an ephemeral port and serve `app` on it (the shape `server::tests` uses;
/// its helpers are private to that module).
async fn serve(app: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    addr
}

/// One real HTTP/1.1 exchange with a caller-supplied request line and headers, read
/// to EOF (`Connection: close` makes the server close, so `read_to_end` terminates).
/// Returns `(head, body)`.
async fn request(addr: SocketAddr, extra_headers: &str) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream.write_all(extra_headers.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("response headers");
    (head.to_string(), body.to_string())
}

async fn get(addr: SocketAddr, path: &str) -> (String, String) {
    request(
        addr,
        &format!("GET {path} HTTP/1.1\r\nHost: minfer\r\nConnection: close\r\n\r\n"),
    )
    .await
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

/// The premise of #103, as a test: the engine's copy of every page and asset is
/// `no-store`, so a *rebuilt* binary cannot be viewed through the browser's cache
/// entry from the previous one.
#[tokio::test]
async fn every_embedded_page_and_asset_forbids_caching() {
    let cases: [(&str, Response); 7] = [
        ("index.html", page().await),
        ("style.css", style_asset().await),
        ("app.js", app_asset().await),
        ("e2e.html", e2e_page().await),
        ("e2e.css", e2e_style_asset().await),
        ("e2e-model.js", e2e_model_asset().await),
        ("e2e.js", e2e_app_asset().await),
    ];
    for (name, response) in cases {
        let headers = response.headers();
        assert_eq!(
            headers.get(axum::http::header::CACHE_CONTROL).unwrap(),
            "no-store",
            "{name} must forbid caching"
        );
        assert!(
            headers
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with(if name.ends_with(".html") {
                    "text/html"
                } else if name.ends_with(".css") {
                    "text/css"
                } else {
                    "application/javascript"
                }),
            "{name} content type"
        );
    }
}

/// #103's replacement for the hand-bumped `?v=N`: the two embedded pages compose
/// their asset URLs at load time. What this pins is the half a Rust test can see —
/// no literal survives in the bytes the engine serves (`?v=`), and the loader is
/// there. The URLs it composes are checked by `scripts/check_viz_assets.mjs`.
#[test]
fn the_embedded_pages_use_the_loader_and_carry_no_version_literal() {
    for (name, html) in [("index.html", INDEX_HTML), ("e2e.html", E2E_HTML)] {
        assert!(
            html.contains("document.write"),
            "{name} must compose its asset URLs at load time"
        );
        assert!(
            !html.contains(".css?v=") && !html.contains(".js?v="),
            "{name} still carries a hand-bumped version parameter on an asset URL"
        );
        for asset in ["style.css", "app.js", "e2e.css", "e2e-model.js", "e2e.js"] {
            assert!(
                !html.contains(&format!("\"{asset}\"")),
                "{name} references {asset} with a literal attribute instead of the loader"
            );
        }
    }
}

/// The loader always writes `?t=<ms>`, and the engine's router must serve it: if
/// the query were part of the path, every asset the stamped page asks for would
/// 404 and the page would be blank. The body is the whole file, unchanged.
#[tokio::test]
async fn the_router_serves_the_asset_urls_the_loader_composes() {
    let (state, _jobs) = state();
    let addr = serve(viz_router(state)).await;

    let (head, body) = get(addr, "/style.css?t=1750000000000").await;
    assert!(head.starts_with("HTTP/1.1 200 OK"), "status line: {head}");
    assert_eq!(header(&head, "cache-control"), Some("no-store"));
    assert_eq!(body, STYLE_CSS);

    let (head, body) = get(addr, "/e2e.js?t=1").await;
    assert!(head.starts_with("HTTP/1.1 200 OK"), "status line: {head}");
    assert_eq!(body, E2E_JS);
}

/// A throwaway `viz/` directory with the named files under `samples/`, removed when
/// it drops (`server::viz::sample_response` takes the directory as an argument, so no
/// test has to set the process-wide `MINFER_VIZ_DIR`).
struct SampleDir(std::path::PathBuf);

impl SampleDir {
    fn new(tag: &str, files: &[(&str, &str)]) -> Self {
        let root = std::env::temp_dir().join(format!("minfer-viz-{tag}-{}", std::process::id()));
        let samples = root.join("samples");
        std::fs::create_dir_all(&samples).unwrap();
        for (name, body) in files {
            std::fs::write(samples.join(name), body).unwrap();
        }
        SampleDir(root)
    }
}

impl Drop for SampleDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The sample contract: JSON, the file on disk, a validator, and `no-cache` — a
/// client may keep the copy but must confirm it before using it (#546, #548). The
/// pages ask for the same on every fetch; this is the engine's half, for a client
/// that does not (`curl`, a deep link opened directly).
#[tokio::test]
async fn a_sample_is_served_as_json_that_must_be_revalidated() {
    let dir = SampleDir::new("sample-ok", &[("one.json", r#"{"samples":[]}"#)]);
    let response = super::sample_response(&dir.0, "one.json", None);
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .unwrap(),
        "application/json"
    );
    assert_eq!(
        response
            .headers()
            .get(axum::http::header::CACHE_CONTROL)
            .unwrap(),
        "no-cache"
    );
    assert!(
        response
            .headers()
            .get(axum::http::header::ETAG)
            .is_some_and(|tag| tag.to_str().unwrap().starts_with('"')),
        "a revalidating client needs a validator to send back"
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&body[..], br#"{"samples":[]}"#);
}

/// #548: the validator makes the revalidation cheap, and a rewrite moves it. The
/// three arms are one test on purpose — a `304` is only correct next to the `200`
/// that a changed file produces (gate contract rule 2).
#[tokio::test]
async fn a_sample_that_matches_its_etag_is_not_sent_again() {
    let file = "one.json";
    let dir = SampleDir::new("sample-etag", &[(file, r#"{"samples":["a"]}"#)]);
    let sample = dir.0.join("samples").join(file);

    let first = super::sample_response(&dir.0, file, None);
    assert_eq!(first.status(), StatusCode::OK);
    let tag = first
        .headers()
        .get(axum::http::header::ETAG)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let unchanged = super::sample_response(&dir.0, file, Some(&tag));
    assert_eq!(unchanged.status(), StatusCode::NOT_MODIFIED);
    assert_eq!(
        unchanged
            .headers()
            .get(axum::http::header::ETAG)
            .unwrap()
            .to_str()
            .unwrap(),
        tag,
        "a 304 repeats the validator its 200 would have sent"
    );
    let body = axum::body::to_bytes(unchanged.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(body.is_empty(), "a 304 carries no body");

    // The control arm: a rewrite that changes the length moves the tag, so the very
    // same conditional request gets the new bytes instead of a 304.
    std::fs::write(&sample, r#"{"samples":["a","b"]}"#).unwrap();
    let changed = super::sample_response(&dir.0, file, Some(&tag));
    assert_eq!(changed.status(), StatusCode::OK);
    assert_ne!(
        changed
            .headers()
            .get(axum::http::header::ETAG)
            .unwrap()
            .to_str()
            .unwrap(),
        tag
    );
    let body = axum::body::to_bytes(changed.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&body[..], br#"{"samples":["a","b"]}"#);
}

/// The `If-None-Match` forms a browser, a proxy or `curl` can send, and the values
/// that must never produce a `304`.
#[test]
fn an_if_none_match_value_is_matched_weakly_and_by_list() {
    let tag = "\"12-345\"";
    assert!(super::etag_matches(Some(tag), tag));
    assert!(super::etag_matches(Some("*"), tag));
    assert!(super::etag_matches(Some("W/\"12-345\""), tag));
    assert!(super::etag_matches(Some("\"other\", \"12-345\""), tag));
    assert!(super::etag_matches(Some(" \"12-345\" "), tag));
    assert!(!super::etag_matches(None, tag));
    assert!(!super::etag_matches(Some("\"other\""), tag));
    assert!(!super::etag_matches(Some(""), tag));
    assert!(!super::etag_matches(Some("garbage"), tag));
    assert!(!super::etag_matches(Some("\"12-345\""), "\"12-346\""));
}

/// The engine half over a real socket: the handler reads `If-None-Match` (the
/// `HeaderMap` extraction the direct calls above cannot show) and answers `304` with
/// no body. It reads the checkout's own `viz/samples/manifest.json`, so a missing
/// sample file fails here too.
#[tokio::test]
async fn a_conditional_sample_request_is_answered_with_304_over_http() {
    let (state, _jobs) = state();
    let addr = serve(viz_router(state)).await;

    let (head, body) = get(addr, "/samples/manifest.json").await;
    assert!(head.starts_with("HTTP/1.1 200 OK"), "status line: {head}");
    assert_eq!(header(&head, "cache-control"), Some("no-cache"));
    let etag = header(&head, "etag")
        .expect("an ETag on the 200")
        .to_string();
    assert!(!body.is_empty(), "the first request carries the sample");

    let (head, body) = request(
        addr,
        &format!(
            "GET /samples/manifest.json HTTP/1.1\r\nHost: minfer\r\n\
             If-None-Match: {etag}\r\nConnection: close\r\n\r\n"
        ),
    )
    .await;
    assert!(head.starts_with("HTTP/1.1 304"), "status line: {head}");
    assert_eq!(header(&head, "etag"), Some(etag.as_str()));
    assert!(body.is_empty(), "a 304 carries no body: {body:?}");
}

/// `GET /samples/:name` reads a file under a directory named by `MINFER_VIZ_DIR`, so
/// the name is a path a stranger supplies. The handler refuses a separator or a `..`
/// outright, which is the property — not axum's decoding, so the guard is exercised
/// directly here.
#[test]
fn a_sample_name_cannot_escape_the_samples_directory() {
    let dir = SampleDir::new("sample-traversal", &[("one.json", "{}")]);
    for name in ["..", "../Cargo.toml", "samples/../../Cargo.toml", "a/b"] {
        let status = super::sample_response(&dir.0, name, None).status();
        assert_eq!(status, StatusCode::BAD_REQUEST, "name {name:?}");
    }
    let status = super::sample_response(&dir.0, "does-not-exist.json", None).status();
    assert_eq!(status, StatusCode::NOT_FOUND);
}
