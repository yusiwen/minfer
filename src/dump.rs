// Layer output dump module — enabled via `--features debug_dump`.
// When enabled, each layer's hidden state and the final logits are written
// as raw f32 binary files to the directory specified by MINFER_DUMP_DIR.
//
// Usage:
//   MINFER_DUMP_DIR=/tmp cargo run --release --features debug_dump -- <model> "Hello"
//
// Output files (per layer):
//   {MINFER_DUMP_DIR}/minfer_dump_layer{N}_out.f32  — hidden state after layer N
//   {MINFER_DUMP_DIR}/minfer_dump_logits.f32        — final logits

#[cfg(feature = "debug_dump")]
static DUMP_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

#[cfg(feature = "debug_dump")]
pub fn maybe_dump(name: &str, data: &[f32]) {
    let dir = DUMP_DIR.get_or_init(|| std::env::var("MINFER_DUMP_DIR").ok());
    if let Some(root) = dir {
        let path = format!("{}/{}.f32", root, name);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        std::fs::write(&path, bytes).ok();
    }
}

#[cfg(not(feature = "debug_dump"))]
#[inline(always)]
pub fn maybe_dump(_name: &str, _data: &[f32]) {}
