//! `#[cfg(test)] mod tests` for `src/gguf.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn split_file_info_parses_pattern() {
    let (prefix, idx, count) =
        split_file_info("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf").unwrap();
    assert_eq!(prefix, "qwen2.5-7b-instruct-q4_k_m");
    assert_eq!(idx, 0);
    assert_eq!(count, 2);

    let (_, idx2, count2) =
        split_file_info("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf").unwrap();
    assert_eq!(idx2, 1);
    assert_eq!(count2, 2);

    // not a split file
    assert!(split_file_info("qwen2.5-0.5b-instruct-q4_k_m.gguf").is_none());
    // invalid idx > count
    assert!(split_file_info("foo-00099-of-00002.gguf").is_none());
}

#[test]
fn resolve_splits_builds_all_parts() {
    let parts = resolve_splits(std::path::Path::new("/m/foo-00001-of-00003.gguf")).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(
        parts[0].file_name().unwrap().to_str().unwrap(),
        "foo-00001-of-00003.gguf"
    );
    assert_eq!(
        parts[1].file_name().unwrap().to_str().unwrap(),
        "foo-00002-of-00003.gguf"
    );
    assert_eq!(
        parts[2].file_name().unwrap().to_str().unwrap(),
        "foo-00003-of-00003.gguf"
    );
    // non-first part rejected
    assert!(resolve_splits(std::path::Path::new("/m/foo-00002-of-00003.gguf")).is_none());
}

/// A tensor info with a shape the size helper can compute (`ne[0]` elements of
/// `type_`), for the recorded-offset-table tests below.
fn info_at(name: &str, offset: u64, ne0: i64, type_: GgmlType) -> GgufTensorInfo {
    // Strides exactly as the parser fills them (gguf.cpp lines 728-732): the
    // size helper reads them, not the raw shape.
    let blck = type_.blck_size() as usize;
    let mut nb = [0usize; GGML_MAX_DIMS];
    nb[0] = type_.type_size();
    nb[1] = nb[0] * (ne0 as usize / blck);
    for j in 2..GGML_MAX_DIMS {
        nb[j] = nb[j - 1];
    }
    GgufTensorInfo {
        name: name.to_string(),
        ne: [ne0, 1, 1, 1],
        nb,
        type_,
        offset,
    }
}

#[test]
fn a_data_section_with_padding_between_tensors_is_accepted() {
    // #575: the offsets are the file's, not an accumulated total. These are the
    // first three tensors of `rwkv-6-world-1.6b-Q2_K.gguf`'s recorded layout,
    // where a 7 340 032-byte gap follows the embedding — a legal file the loader
    // used to reject.
    let info = vec![
        info_at("token_embd.weight", 0, 2048, GgmlType::Q2_K),
        info_at("blk.0.attn_norm.weight", 44_040_192, 2048, GgmlType::F32),
    ];
    let size = data_section_size(&info, 100_000_000).expect("a padded layout is accepted");
    // The section ends at the *furthest* tensor end, not at a running total.
    assert_eq!(size, 44_040_192 + 2048 * 4);
}

#[test]
fn a_tensor_that_spans_past_the_part_is_refused_by_name() {
    // The bounds check replaced the contiguity check: a truncated part (or a
    // corrupt offset) is still a refusal, and the message names the tensor.
    let info = vec![info_at(
        "blk.0.attn_norm.weight",
        44_040_192,
        2048,
        GgmlType::F32,
    )];
    let err = data_section_size(&info, 44_040_192).unwrap_err();
    assert!(err.contains("attn_norm.weight"), "{err}");
    assert!(err.contains("spans bytes"), "{err}");
}

// === #745: the model owns its mapping ===

/// Where `/proc/self/fd/N` points (Linux only); `None` when the descriptor is
/// closed. The target, not the number, is what a test can assert on: this suite
/// runs many tests in one process and descriptors are reused.
#[cfg(target_os = "linux")]
fn fd_target(fd: i32) -> Option<std::path::PathBuf> {
    std::fs::read_link(format!("/proc/self/fd/{fd}")).ok()
}

fn toy_gguf() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/toy-qwen2-f32.gguf")
}

/// [#745] Dropping the model unmaps the file and closes its descriptor.
///
/// Before this, `load_gguf_model` leaked the `MmapFile` with `Box::leak` so weight
/// tensors could hold `&'static` slices, which meant `MmapFile::drop` was
/// unreachable: every load in one process kept a mapping and an fd forever. The two
/// observables asserted here are the ones a drop can actually change — the mapping's
/// own `alive` flag (portable) and its fd's presence in `/proc/self/fd` (Linux).
#[test]
fn a_dropped_model_releases_its_mapping_and_fd() {
    let path = toy_gguf();
    // A macOS test process may have initialised MPS, whose zero-copy part
    // registration legitimately owns the pages (that is what makes the GPU read
    // them after the model is gone). Assert the CPU case; say so when it is not
    // the case rather than passing silently.
    #[cfg(target_os = "macos")]
    if crate::metal::MpsState::get().is_some() {
        eprintln!(
            "745: skipped — MPS is live in this process and its part registration \
             holds the mapping on purpose"
        );
        return;
    }
    let (alive, fd) = {
        let model = load_gguf_model(&path).expect("the toy GGUF loads");
        assert_eq!(model.parts.len(), 1, "the toy fixture is a single part");
        let mmap = model.parts[0].mmap.clone();
        let alive = mmap.alive();
        let fd = mmap.raw_fd();
        assert!(
            alive.load(std::sync::atomic::Ordering::SeqCst),
            "a live model must have a live mapping"
        );
        #[cfg(target_os = "linux")]
        assert_eq!(
            fd_target(fd).as_deref(),
            Some(path.as_path()),
            "the mapping's fd {fd} must be this file while the model is alive"
        );
        (alive, fd)
    };
    assert!(
        !alive.load(std::sync::atomic::Ordering::SeqCst),
        "the mapping outlived the model that owned it — the leak is back"
    );
    // The fd number may already belong to another test's file (the suite runs in
    // one process), so the assertion is about the *target*: our GGUF must not be
    // open on that descriptor any more.
    #[cfg(target_os = "linux")]
    assert_ne!(
        fd_target(fd).as_deref(),
        Some(path.as_path()),
        "fd {fd} still refers to the model file after the model was dropped"
    );
}

/// [#745] A model that outlives its `GgufModel` keeps its own bytes alive, and
/// dropping the model releases the mapping.
///
/// This is the property the per-tensor handle exists for: the loader returns
/// `Box<dyn ModelDef>` whose weight tensors point into the mapping, and callers are
/// free to drop the `GgufModel` first.
#[test]
fn a_loaded_model_keeps_its_mapping_until_it_is_dropped() {
    #[cfg(target_os = "macos")]
    if crate::metal::MpsState::get().is_some() {
        eprintln!("745: skipped — MPS's part registration holds the mapping on purpose");
        return;
    }
    let path = toy_gguf();
    let (model, alive) = {
        let gguf = load_gguf_model(&path).expect("the toy GGUF loads");
        let alive = gguf.parts[0].mmap.alive();
        let model = crate::models::load_model(&gguf).expect("the toy model loads");
        // The gguf is dropped here; the model's weight tensors must hold the pages.
        (model, alive)
    };
    assert!(
        alive.load(std::sync::atomic::Ordering::SeqCst),
        "the model's tensors must keep the mapping alive after the GgufModel is dropped"
    );
    // A *clone* of one weight is a handle in its own right — the property the
    // per-tensor owner exists for — and it keeps the bytes readable after the model
    // is gone. Then the last handle must unmap.
    let weight = model
        .as_any()
        .downcast_ref::<crate::models::qwen2::Qwen2Model>()
        .and_then(|m| m.layers[0].attn_norm.clone())
        .expect("the toy fixture is a Qwen2 model with an attn_norm weight");
    let live_bytes = weight.data.len();
    drop(model);
    assert!(live_bytes > 0, "the cloned weight must carry bytes");
    assert!(
        alive.load(std::sync::atomic::Ordering::SeqCst),
        "a clone of a weight must keep the mapping alive on its own"
    );
    // Reading through the clone is the real check: the pages are unmapped only
    // after this last handle goes.
    let sum: u64 = weight.data.iter().take(64).map(|b| u64::from(*b)).sum();
    assert!(sum > 0, "the clone's bytes must still be readable");
    drop(weight);
    assert!(
        !alive.load(std::sync::atomic::Ordering::SeqCst),
        "the last tensor handle must unmap the file"
    );
}

/// [#745] A reload is a load → drop → load, and it leaves nothing behind.
///
/// This is the acceptance shape the ticket asks for — a *count*, not "looks fine":
/// two loads release the first mapping and its fd before the second takes one. It
/// also pins the zero-copy win the leak was paying for: every weight tensor must
/// still point *inside* its part's mapping, so an `Arc` thread did not quietly turn
/// the borrowed view into a per-tensor copy.
#[test]
fn a_reload_releases_the_first_mapping_and_keeps_the_view_borrowed() {
    #[cfg(target_os = "macos")]
    if crate::metal::MpsState::get().is_some() {
        eprintln!("745: skipped — MPS's part registration holds each mapping on purpose");
        return;
    }
    let path = toy_gguf();
    let mut first = None;
    for round in 0..2 {
        let (model, alive, fd) = {
            let gguf = load_gguf_model(&path).expect("the toy GGUF loads");
            let part = &gguf.parts[0];
            let mmap = part.mmap.clone();
            let (alive, fd) = (mmap.alive(), mmap.raw_fd());
            let model = crate::models::load_model(&gguf).expect("the toy model loads");
            // Zero-copy: every weight tensor's slice is inside the mapping, not a copy.
            let base = mmap.as_slice().as_ptr() as usize;
            let end = base + mmap.as_slice().len();
            let qwen = model
                .as_any()
                .downcast_ref::<crate::models::qwen2::Qwen2Model>()
                .expect("the toy fixture is a Qwen2 model");
            let mut checked = 0usize;
            for layer in &qwen.layers {
                for w in [
                    &layer.attn_norm,
                    &layer.wq,
                    &layer.wk,
                    &layer.wv,
                    &layer.wo,
                    &layer.ffn_norm,
                    &layer.ffn_gate,
                    &layer.ffn_up,
                    &layer.ffn_down,
                ]
                .into_iter()
                .flatten()
                {
                    let p = w.data().as_ptr() as usize;
                    assert!(
                        p >= base && p + w.data().len() <= end,
                        "round {round}: weight {} is not a view of the part's mapping",
                        w.name
                    );
                    checked += 1;
                }
            }
            assert!(checked > 0, "the toy model must expose weights to check");
            (model, alive, fd)
        };
        // Drop first, then assert: the handle the model holds is the thing under test.
        drop(model);
        assert!(
            !alive.load(std::sync::atomic::Ordering::SeqCst),
            "round {round}: the mapping survived the model drop"
        );
        if round == 0 {
            first = Some(fd);
        }
    }
    // On Linux the first round's descriptor must not still be the model file.
    #[cfg(target_os = "linux")]
    if let Some(fd) = first {
        assert_ne!(
            fd_target(fd).as_deref(),
            Some(path.as_path()),
            "the first load's fd {fd} is still open after a full reload cycle"
        );
    }
}
