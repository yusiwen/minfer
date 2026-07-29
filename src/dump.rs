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
pub fn maybe_dump(name: &str, data: &[f32]) {
    if let Some(root) = dump_dir() {
        let path = format!("{}/{}.f32", root, name);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        std::fs::write(&path, bytes).ok();
    }
}

/// Dump during prefill (nt>1), or dump the FIRST generation call to {name}_gen0.
/// Each unique name fires at most once for generation.
#[cfg(feature = "debug_dump")]
pub fn maybe_dump_prefill_or_gen0(name: &str, data: &[f32], nt: usize) {
    if nt > 1 {
        maybe_dump(name, data);
    } else {
        use std::sync::Mutex;
        use std::collections::HashSet;
        static DUMPED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
        let mut set = DUMPED.lock().unwrap();
        let set_ref = set.get_or_insert_with(HashSet::new);
        if set_ref.insert(name.to_string()) {
            let gen_name = format!("{}_gen0", name);
            maybe_dump(&gen_name, data);
        }
    }
}

/// Same as maybe_dump_prefill_or_gen0 but with an additional `active` guard
/// (e.g. `il == 0` for layer-specific dumps).
#[cfg(feature = "debug_dump")]
pub fn maybe_dump_prefill_or_gen0_if(name: &str, data: &[f32], nt: usize, active: bool) {
    if active {
        maybe_dump_prefill_or_gen0(name, data, nt);
    }
}

#[cfg(feature = "debug_dump")]
pub fn maybe_dump_text(name: &str, text: &str) {
    if let Some(root) = dump_dir() {
        let path = format!("{}/{}.txt", root, name);
        std::fs::write(&path, text).ok();
    }
}

#[cfg(not(feature = "debug_dump"))]
#[inline(always)]
pub fn maybe_dump(_name: &str, _data: &[f32]) {}

#[cfg(not(feature = "debug_dump"))]
#[inline(always)]
pub fn maybe_dump_prefill_or_gen0(_name: &str, _data: &[f32], _nt: usize) {}

#[cfg(not(feature = "debug_dump"))]
#[inline(always)]
pub fn maybe_dump_prefill_or_gen0_if(_name: &str, _data: &[f32], _nt: usize, _active: bool) {}

#[cfg(not(feature = "debug_dump"))]
#[inline(always)]
pub fn maybe_dump_text(_name: &str, _text: &str) {}
