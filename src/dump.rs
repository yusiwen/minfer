// Layer output dump module — enabled via `--features debug_dump`.
// When enabled, text artifacts are written to the directory specified by
// MINFER_DUMP_DIR (the per-layer f32 dumps were removed with the imperative
// forward path — the graph path dumps via MINFER_GRAPH_DUMP instead).
//
// Usage:
//   MINFER_DUMP_DIR=/tmp cargo run --release --features debug_dump -- <model> "Hello"
//
// Output files:
//   {MINFER_DUMP_DIR}/minfer_dump_prompt.txt         — rendered prompt text

#[cfg(feature = "debug_dump")]
static DUMP_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

#[cfg(feature = "debug_dump")]
fn dump_dir() -> Option<&'static String> {
    DUMP_DIR
        .get_or_init(|| std::env::var("MINFER_DUMP_DIR").ok())
        .as_ref()
}

#[cfg(feature = "debug_dump")]
pub fn maybe_dump_text(name: &str, text: &str) {
    if let Some(root) = dump_dir() {
        let path = format!("{}/{}.txt", root, name);
        std::fs::write(&path, text).ok();
    }
}
