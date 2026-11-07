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

/// [#745] A reload after a **forward** releases the engine's mapping, and the
/// second model is a different model.
///
/// The earlier gate loaded the same file twice without building a graph; that
/// proved the loader's handle, not the engine's. `forward_graph_cached` puts every
/// weight into the `GraphCache`'s CPU backend as a `Tensor` clone, so the drop
/// order that matters is model + cache. The second half changes one weight byte in
/// a copy of the toy and asserts the output moves — the identity a name-keyed
/// device registry can silently break.
#[test]
fn a_reload_after_a_forward_releases_the_mapping_and_loads_a_different_model() {
    #[cfg(target_os = "macos")]
    if crate::metal::MpsState::get().is_some() {
        eprintln!("745: skipped — MPS's part registration holds each mapping on purpose");
        return;
    }
    let path = toy_gguf();
    let gguf_a = load_gguf_model(&path).expect("the toy GGUF loads");
    let mmap_a = gguf_a.parts[0].mmap.clone();
    let alive_a = mmap_a.alive();
    let fd = mmap_a.raw_fd();
    // A weight's byte offset in the *file* (the toy is a single part).
    let weight_off = {
        let ctx = &gguf_a.parts[0].ctx;
        let ti = ctx
            .info
            .iter()
            .find(|t| t.name == "blk.0.attn_norm.weight")
            .expect("the toy has blk.0.attn_norm.weight");
        ctx.offset + ti.offset as usize
    };
    // Force the CPU path: this gate is about the host mapping and the graph cache,
    // not about a device's copied weights (that is the CUDA identity follow-up).
    let model_a = crate::models::load_model_with(
        &gguf_a,
        "",
        crate::graph::offload::OffloadRequest::Layers(0),
    )
    .expect("the toy model loads");
    let mut cache_a = crate::graph::cache::GraphCache::new();
    let logits_a = model_a.forward_graph_cached(&[1], &[0], 1, 16, &mut cache_a);
    assert!(!logits_a.is_empty(), "the forward step must produce logits");
    // Zero-copy: every weight tensor's slice is inside the part's mapping.
    let base = mmap_a.as_slice().as_ptr() as usize;
    let end = base + mmap_a.as_slice().len();
    let qwen = model_a
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
                "weight {} is not a view of the part's mapping",
                w.name
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "the toy model must expose weights to check");
    // The test's own handle must go too, or it (not the engine) keeps the mapping.
    drop(mmap_a);
    drop(model_a);
    drop(cache_a);
    drop(gguf_a);
    assert!(
        !alive_a.load(std::sync::atomic::Ordering::SeqCst),
        "the forward's graph cache kept the mapping alive after model + cache drop"
    );
    #[cfg(target_os = "linux")]
    assert_ne!(
        fd_target(fd).as_deref(),
        Some(path.as_path()),
        "the first load's fd {fd} is still open after model + cache drop"
    );

    // A *different* model with the same shapes: copy the toy, flip one weight byte,
    // and check the output actually moves.
    let temp = std::env::temp_dir().join(format!("minfer-745-reload-{}.gguf", std::process::id()));
    std::fs::copy(&path, &temp).expect("copy the toy fixture");
    {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&temp)
            .expect("open the copied fixture");
        f.seek(SeekFrom::Start(weight_off as u64))
            .expect("seek to the weight");
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).expect("read the weight byte");
        f.seek(SeekFrom::Start(weight_off as u64))
            .expect("seek back to the weight");
        f.write_all(&[byte[0] ^ 0xFF])
            .expect("change the weight byte");
    }
    let gguf_b = load_gguf_model(&temp).expect("the modified toy GGUF loads");
    let alive_b = gguf_b.parts[0].mmap.alive();
    let model_b = crate::models::load_model_with(
        &gguf_b,
        "",
        crate::graph::offload::OffloadRequest::Layers(0),
    )
    .expect("the modified toy model loads");
    let mut cache_b = crate::graph::cache::GraphCache::new();
    let logits_b = model_b.forward_graph_cached(&[1], &[0], 1, 16, &mut cache_b);
    assert_ne!(
        logits_a, logits_b,
        "the changed weight must change the output; a name+size-keyed registry that \
         reused the first model's weights would fail this"
    );
    drop(model_b);
    drop(cache_b);
    drop(gguf_b);
    assert!(
        !alive_b.load(std::sync::atomic::Ordering::SeqCst),
        "the second model's mapping survived model + cache drop"
    );
    let _ = std::fs::remove_file(&temp);
}

/// [#745] A CUDA reload must register the second model's weights.
///
/// The CUDA weight registry is name-keyed and used to reuse a same-name+size
/// device copy for *any* model. Two checkpoints of one architecture have the
/// same tensor names and sizes but different bytes, so that reuse silently served
/// the first model's weights. The model fingerprint in [`GgufModel`] is what lets
/// `register_weight` tell "reload of the same file" (reuse is correct) from "a
/// different model" (copy).
#[cfg(any(feature = "cuda", target_os = "macos"))]
fn toy_with_changed_weight(label: &str) -> std::path::PathBuf {
    let path = toy_gguf();
    let temp = std::env::temp_dir().join(format!("minfer-745-{label}-{}.gguf", std::process::id()));
    std::fs::copy(&path, &temp).expect("copy the toy fixture");
    let off = {
        let gguf = load_gguf_model(&path).expect("the toy GGUF loads");
        let ctx = &gguf.parts[0].ctx;
        let ti = ctx
            .info
            .iter()
            .find(|t| t.name == "blk.0.attn_norm.weight")
            .expect("the toy has blk.0.attn_norm.weight");
        ctx.offset + ti.offset as usize
    };
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&temp)
        .expect("open the copied fixture");
    f.seek(SeekFrom::Start(off as u64))
        .expect("seek to the weight");
    let mut byte = [0u8; 1];
    f.read_exact(&mut byte).expect("read the weight byte");
    f.seek(SeekFrom::Start(off as u64))
        .expect("seek back to the weight");
    f.write_all(&[byte[0] ^ 0xFF])
        .expect("change the weight byte");
    temp
}

#[cfg(feature = "cuda")]
#[test]
fn a_cuda_reload_registers_the_second_models_weights() {
    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let path = toy_gguf();
    let temp = toy_with_changed_weight("cuda-reload");
    let load = |p: &std::path::Path| {
        let gguf = load_gguf_model(p).expect("the toy GGUF loads");
        let model = crate::models::load_model_with(
            &gguf,
            "",
            crate::graph::offload::OffloadRequest::Layers(1),
        )
        .expect("the toy model loads");
        assert_eq!(
            model.device(),
            crate::models::Device::Cuda,
            "this gate needs the CUDA path"
        );
        (gguf, model)
    };
    let (gguf_a, model_a) = load(&path);
    let mut cache_a = crate::graph::cache::GraphCache::new();
    let logits_a = model_a.forward_graph_cached(&[1], &[0], 1, 16, &mut cache_a);
    assert!(!logits_a.is_empty(), "the forward step must produce logits");
    drop(model_a);
    drop(cache_a);
    drop(gguf_a);

    let (gguf_b, model_b) = load(&temp);
    let mut cache_b = crate::graph::cache::GraphCache::new();
    let logits_b = model_b.forward_graph_cached(&[1], &[0], 1, 16, &mut cache_b);
    assert_ne!(
        logits_a, logits_b,
        "the second model reused the first model's device weights; the CUDA registry must \
         key same-name+size reuse by the model, not just the shape"
    );
    drop(model_b);
    drop(cache_b);
    drop(gguf_b);
    let _ = std::fs::remove_file(&temp);
}

/// [#752] A Metal reload: load, forward, drop, then load a *different* model.
///
/// The zero-copy `MTLBuffer` aliases the mapping, so the part registration must
/// keep the pages alive while the buffer exists; `ModelRegistration` drops the
/// weight buffers first, then the part buffers, then the `Arc<MmapFile>` handles
/// (ADR-0077). The graph cache holds `Tensor` clones, so the engine's release is
/// model + cache, not the model alone. This gate is the two-model shape #752 asks
/// for: the first model's mapping and fd must be gone before the second model
/// runs, and the second model must not reuse the first model's registration.
#[cfg(target_os = "macos")]
fn macos_open_fds_to(path: &std::path::Path) -> usize {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("/usr/sbin/lsof")
        .args(["-p", &pid, "-Fn"])
        .output()
        .expect("run lsof");
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let target = target.to_string_lossy();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with('n') && l[1..].ends_with(target.as_ref()))
        .count()
}

#[cfg(target_os = "macos")]
#[test]
fn a_metal_reload_releases_the_first_mapping_and_runs_a_different_model() {
    crate::metal::MpsState::init();
    if crate::metal::MpsState::get().is_none() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let toy = toy_gguf();
    let a_path =
        std::env::temp_dir().join(format!("minfer-752-metal-a-{}.gguf", std::process::id()));
    let b_path = toy_with_changed_weight("metal-reload-b");
    std::fs::copy(&toy, &a_path).expect("copy the toy fixture for model A");

    let load = |p: &std::path::Path| {
        let gguf = load_gguf_model(p).expect("the toy GGUF loads");
        let alive = gguf.parts[0].mmap.alive();
        let model = crate::models::load_model_with(
            &gguf,
            "",
            crate::graph::offload::OffloadRequest::Layers(1),
        )
        .expect("the toy model loads");
        assert_eq!(
            model.device(),
            crate::models::Device::Metal,
            "this gate needs the Metal path"
        );
        (gguf, model, alive)
    };

    let (gguf_a, model_a, alive_a) = load(&a_path);
    assert_eq!(
        macos_open_fds_to(&a_path),
        1,
        "the live model must hold exactly one fd for its file"
    );
    let mut cache_a = crate::graph::cache::GraphCache::new();
    let logits_a = model_a.forward_graph_cached(&[1], &[0], 1, 16, &mut cache_a);
    assert!(!logits_a.is_empty(), "the forward step must produce logits");
    // Zero-copy: every weight is a range of the part mapping, not a copy.
    let mmap = gguf_a.parts[0].mmap.clone();
    let base = mmap.as_slice().as_ptr() as usize;
    let end = base + mmap.as_slice().len();
    let qwen = model_a
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
                "weight {} is not a view of the part's mapping",
                w.name
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "the toy model must expose weights to check");
    drop(mmap);
    drop(model_a);
    drop(cache_a);
    drop(gguf_a);
    assert!(
        !alive_a.load(std::sync::atomic::Ordering::SeqCst),
        "the first model's Metal registration kept its mapping alive"
    );
    assert_eq!(
        macos_open_fds_to(&a_path),
        0,
        "the first model's fd must be closed after model + cache drop"
    );

    // A different model with the same tensor names/sizes: the registration id
    // and the fingerprint keep it from reusing model A's buffers.
    let (gguf_b, model_b, alive_b) = load(&b_path);
    let mut cache_b = crate::graph::cache::GraphCache::new();
    let logits_b = model_b.forward_graph_cached(&[1], &[0], 1, 16, &mut cache_b);
    assert_ne!(
        logits_a, logits_b,
        "the second model produced the first model's logits; the Metal registration was reused"
    );
    drop(model_b);
    drop(cache_b);
    drop(gguf_b);
    assert!(
        !alive_b.load(std::sync::atomic::Ordering::SeqCst),
        "the second model's Metal registration kept its mapping alive"
    );
    assert_eq!(
        macos_open_fds_to(&b_path),
        0,
        "the second model's fd must be closed after model + cache drop"
    );
    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);
}
