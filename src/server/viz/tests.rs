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

/// One real HTTP/1.1 exchange, read to EOF (`Connection: close` makes the server
/// close, so `read_to_end` terminates). Returns `(head, body)`.
async fn get(addr: SocketAddr, path: &str) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: minfer\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("response headers");
    (head.to_string(), body.to_string())
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

/// `GET /samples/:name` reads a file under a directory named by `MINFER_VIZ_DIR`,
/// so the name is a path a stranger supplies. The handler refuses a separator or a
/// `..` outright, which is the property — not axum's decoding, so the guard is
/// exercised directly here.
#[tokio::test]
async fn a_sample_name_cannot_escape_the_samples_directory() {
    for name in ["..", "../Cargo.toml", "samples/../../Cargo.toml", "a/b"] {
        let status = super::sample_asset(Path(name.to_string())).await.status();
        assert_eq!(status, StatusCode::BAD_REQUEST, "name {name:?}");
    }
    let status = super::sample_asset(Path("does-not-exist.json".to_string()))
        .await
        .status();
    assert_eq!(status, StatusCode::NOT_FOUND);
}
