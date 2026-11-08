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

/// [#758] Copy a real GGUF and flip one byte inside its first tensor's payload.
///
/// The copy keeps every tensor name and shape but has a different fingerprint
/// (path + inode + length + mtime, #745), so it stands for a second checkpoint of
/// the same architecture — the "same shapes, different weights" reload arm.
#[cfg(feature = "cuda")]
fn cuda_copy_with_flipped_weight(src: &std::path::Path, label: &str) -> std::path::PathBuf {
    let gguf = load_gguf_model(src).expect("the source GGUF loads");
    let (off, len) = {
        let ctx = &gguf.parts[0].ctx;
        let ti = ctx
            .info
            .iter()
            .find(|t| t.nbytes() >= 128)
            .expect("the model has a tensor payload");
        (ctx.offset + ti.offset as usize, ti.nbytes())
    };
    drop(gguf);
    let dst = std::env::temp_dir().join(format!("minfer-{label}-{}.gguf", std::process::id()));
    std::fs::copy(src, &dst).expect("copy the source GGUF");
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .expect("open the copied GGUF");
    // Halfway through the tensor, then 16 bytes in: a K-quant block's first bytes
    // are the f16 d/dmin scale pair, so this lands in the quantized values instead
    // of turning a scale into an infinity/NaN.
    let pos = off + len / 2 + 16;
    f.seek(SeekFrom::Start(pos as u64))
        .expect("seek to the weight payload");
    let mut byte = [0u8; 1];
    f.read_exact(&mut byte).expect("read the payload byte");
    f.seek(SeekFrom::Start(pos as u64))
        .expect("seek back to the payload byte");
    f.write_all(&[byte[0] ^ 0xFF])
        .expect("flip the payload byte");
    dst
}

/// [#758] Acceptance #1, device half: after a reload, free device memory returns
/// to the pre-reload baseline and the two models produce different outputs.
///
/// Load A, run four forwards through one `GraphCache` (the third captures the
/// decode split), record the free bytes, drop model + cache, and assert
/// `cudaMemGetInfo` reports the baseline again within 128 MiB. Then load B — a copy
/// of A with one weight byte flipped — forward it, and assert the logits differ
/// (#755's identity property, kept green here). `weights_bytes()` must also return
/// to its baseline: it now sums the ownership ledger, so a retained buffer would
/// keep being charged, and it must be **exact** (the free-bytes bound is the
/// looser one).
///
/// A warm-up cycle with B runs first, and the baseline is recorded after it. The
/// CUDA context's one-time growth (kernel module pages, the state-level prefill
/// scratches, graph-exec runtime bookkeeping) is not per-model weight memory;
/// charging it to the tolerance would make the bound meaningless. The measured
/// residual after a reload on GB10 is 50–59 MiB of that non-weight state, so the
/// tolerance is 128 MiB — still far below the 727 MiB of device weights a leaked
/// model holds, which is what this gate must catch. The measured reload is still
/// A → B: the two paths have different fingerprints, so same-name+size reuse
/// cannot hide a retained buffer.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "device memory: set MINFER_CUDA_RELOAD_TEST_MODEL to the cached 0.5B GGUF"]
fn a_cuda_reload_frees_the_first_models_device_weights() {
    let path = match std::env::var("MINFER_CUDA_RELOAD_TEST_MODEL") {
        Ok(p) if !p.trim().is_empty() => std::path::PathBuf::from(p),
        _ => {
            eprintln!("skipping: MINFER_CUDA_RELOAD_TEST_MODEL unset");
            return;
        }
    };
    crate::cuda::CudaState::init();
    let Some(state) = crate::cuda::CudaState::get() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const TOL: usize = 128 << 20;
    let free = |state: &crate::cuda::CudaState| match state.device_memory() {
        crate::graph::allocplan::DeviceMemory::Reported { free, total } => (free, total),
        other => panic!("cudaMemGetInfo must answer; got {other:?}"),
    };
    let b_path = cuda_copy_with_flipped_weight(&path, "758-cuda-reload-b");
    let load = |p: &std::path::Path| {
        let gguf = load_gguf_model(p).expect("the real GGUF loads");
        let model = crate::models::load_model_with(
            &gguf,
            "",
            crate::graph::offload::OffloadRequest::Layers(usize::MAX),
        )
        .expect("the real model loads");
        assert_eq!(
            model.device(),
            crate::models::Device::Cuda,
            "this gate needs the CUDA path"
        );
        (gguf, model)
    };
    let forward = |model: &dyn crate::models::ModelDef,
                   cache: &mut crate::graph::cache::GraphCache| {
        // Four decode steps: the third captures the split, the fourth replays it.
        let mut logits = Vec::new();
        for step in 0..4usize {
            logits = model.forward_graph_cached(&[1], &[step], 1, 32, cache);
        }
        logits
    };

    // Warm-up: one full load/forward/drop cycle with B, so the baseline below
    // includes everything the process grows once for CUDA. B is a different file
    // from the measured A, so this cannot pre-populate A's names.
    {
        let (gguf, model) = load(&b_path);
        let mut cache = crate::graph::cache::GraphCache::new();
        let _ = forward(&*model, &mut cache);
        drop(model);
        drop(cache);
        drop(gguf);
        match state.last_release() {
            Some(crate::cuda::ReleaseOutcome::Freed { bytes }) => assert!(
                bytes > 0,
                "the warm-up model's registration must have owned device buffers"
            ),
            other => panic!("the warm-up drop did not free its weights: {other:?}"),
        }
    }
    let (baseline_free, total) = free(state);
    let baseline_weights = state.weights_bytes();

    let (gguf_a, model_a) = load(&path);
    let mut cache_a = crate::graph::cache::GraphCache::new();
    let logits_a = forward(&*model_a, &mut cache_a);
    assert!(!logits_a.is_empty(), "the forward step must produce logits");
    let (live_free, _) = free(state);
    assert!(
        live_free + TOL < baseline_free,
        "model A's weights must occupy device memory while it is loaded: baseline {baseline_free} \
         B, live {live_free} B"
    );
    let live_weights = state.weights_bytes();
    assert!(
        live_weights > baseline_weights,
        "model A's device weights must be charged to the ledger: baseline {baseline_weights} B, \
         live {live_weights} B"
    );

    drop(model_a);
    drop(cache_a);
    drop(gguf_a);
    let (after_free, _) = free(state);
    let after_weights = state.weights_bytes();
    assert_eq!(
        after_weights, baseline_weights,
        "the dropped model's device weights are still charged: baseline {baseline_weights} B, \
         after {after_weights} B"
    );
    assert!(
        after_free + TOL >= baseline_free,
        "a replaced weight kept its device buffer (#758): free bytes after dropping model + \
         cache are {after_free}, below the pre-reload baseline {baseline_free} by more than the \
         {TOL} B tolerance (total {total})"
    );

    let (gguf_b, model_b) = load(&b_path);
    let mut cache_b = crate::graph::cache::GraphCache::new();
    let logits_b = forward(&*model_b, &mut cache_b);
    assert_ne!(
        logits_a, logits_b,
        "the second model produced the first model's logits; a name+size-keyed registry that \
         served A's buffers would fail this (#755)"
    );
    drop(model_b);
    drop(cache_b);
    drop(gguf_b);
    let _ = std::fs::remove_file(&b_path);
    eprintln!(
        "cuda_reload_free: total={total} baseline_free={baseline_free} live_free={live_free} \
         after_free={after_free} tolerance={TOL} baseline_weights={baseline_weights} \
         live_weights={live_weights} after_weights={after_weights}"
    );
}

/// [#758] Acceptance #2, the negative arm: a captured graph that still references
/// a weight blocks the free, by name.
///
/// Load A, forward four times (the third captures the decode split), then drop the
/// model but **keep** the cache. The registration release must report
/// `ReleaseOutcome::Retained` and keep charging the bytes, because a live exec's
/// recorded kernels hold A's raw weight pointers. Dropping the cache drops the last
/// `CaptureLease`, which reaps the retired allocations — the freeing is deferred,
/// never skipped, and never a silent dangling pointer.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "device memory: set MINFER_CUDA_RELOAD_TEST_MODEL to the cached 0.5B GGUF"]
fn a_cuda_release_is_retained_while_a_capture_is_live() {
    let path = match std::env::var("MINFER_CUDA_RELOAD_TEST_MODEL") {
        Ok(p) if !p.trim().is_empty() => std::path::PathBuf::from(p),
        _ => {
            eprintln!("skipping: MINFER_CUDA_RELOAD_TEST_MODEL unset");
            return;
        }
    };
    crate::cuda::CudaState::init();
    let Some(state) = crate::cuda::CudaState::get() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const TOL: usize = 128 << 20;
    let free = |state: &crate::cuda::CudaState| match state.device_memory() {
        crate::graph::allocplan::DeviceMemory::Reported { free, total } => (free, total),
        other => panic!("cudaMemGetInfo must answer; got {other:?}"),
    };
    let load = || {
        let gguf = load_gguf_model(&path).expect("the real GGUF loads");
        let model = crate::models::load_model_with(
            &gguf,
            "",
            crate::graph::offload::OffloadRequest::Layers(usize::MAX),
        )
        .expect("the real model loads");
        assert_eq!(
            model.device(),
            crate::models::Device::Cuda,
            "this gate needs the CUDA path"
        );
        (gguf, model)
    };
    let forward = |model: &dyn crate::models::ModelDef,
                   cache: &mut crate::graph::cache::GraphCache| {
        let mut logits = Vec::new();
        for step in 0..4usize {
            logits = model.forward_graph_cached(&[1], &[step], 1, 32, cache);
        }
        logits
    };

    // Warm-up: one load/forward/drop, so the context's one-time growth is in the
    // baseline and the reload below registers A's names from scratch.
    {
        let (gguf, model) = load();
        let mut cache = crate::graph::cache::GraphCache::new();
        let _ = forward(&*model, &mut cache);
        drop(model);
        drop(cache);
        drop(gguf);
    }
    let (baseline_free, _) = free(state);
    let baseline_weights = state.weights_bytes();

    let (gguf, model) = load();
    let mut cache = crate::graph::cache::GraphCache::new();
    let logits = forward(&*model, &mut cache);
    assert!(!logits.is_empty(), "the forward step must produce logits");
    assert!(
        state.live_captures() > 0,
        "four forwards through one cache must leave a captured graph holding a lease"
    );
    // Drop the model, keep the cache: the captured graph still references A.
    drop(model);
    drop(gguf);
    let retained = match state.last_release() {
        Some(crate::cuda::ReleaseOutcome::Retained { bytes, captures }) => {
            assert!(bytes > 0, "the retained release must name its bytes");
            assert!(captures > 0, "the retained release must name its captures");
            bytes
        }
        other => panic!(
            "a live captured graph must make the release retain, not free (or skip): {other:?}"
        ),
    };
    let weights_while_retained = state.weights_bytes();
    assert!(
        weights_while_retained > baseline_weights,
        "retained weights must stay charged to the E4 ledger: baseline {baseline_weights} B, \
         retained {weights_while_retained} B"
    );
    let (free_while_retained, _) = free(state);
    assert!(
        free_while_retained + TOL < baseline_free,
        "retaining must actually keep the bytes resident: baseline {baseline_free} B, retained \
         {free_while_retained} B"
    );

    // Dropping the cache drops the captured graphs and their leases; the last one
    // reaps the retired allocations.
    drop(cache);
    match state.last_release() {
        Some(crate::cuda::ReleaseOutcome::Freed { bytes }) => assert!(
            bytes >= retained,
            "the reaped bytes ({bytes}) must cover the retained weights ({retained})"
        ),
        other => panic!("dropping the last capture must reap the retained weights: {other:?}"),
    }
    let weights_after = state.weights_bytes();
    assert_eq!(
        weights_after, baseline_weights,
        "the reaped weights are still charged: baseline {baseline_weights} B, after {weights_after} B"
    );
    let (free_after, _) = free(state);
    assert!(
        free_after + TOL >= baseline_free,
        "free bytes after the capture lease is dropped are {free_after}, below the baseline \
         {baseline_free} by more than {TOL} B — the reaping did not return the weight buffers"
    );
    eprintln!(
        "cuda_retain_lease: baseline_free={baseline_free} retained_free={free_while_retained} \
         reaped_free={free_after} tolerance={TOL} baseline_weights={baseline_weights} \
         retained_bytes={retained} weights_while_retained={weights_while_retained} \
         weights_after={weights_after}"
    );
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
    // `-d '^mem'` excludes lsof's memory-mapped-file rows: the mapping is not an
    // open descriptor, the `File` that owns the fd is. Without it the same file
    // is counted twice (fd + mem) and the baseline assertion is off by one.
    let out = std::process::Command::new("/usr/sbin/lsof")
        .args(["-p", &pid, "-a", "-d", "^mem", "-Fn"])
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
    assert!(
        macos_open_fds_to(&a_path) >= 1,
        "the live model must hold an fd for its file"
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

/// [#745] Acceptance #1, RSS half: after a real-model reload the host's resident
/// set and the model file's descriptor count must return to the baseline.
///
/// The two models are different on purpose (A and B are independent cached
/// GGUFs), and `Layers(0)` keeps every forward on the CPU, so this measures the
/// host mapping lifetime rather than a device copy. The tolerance covers
/// allocator caches the process may retain; 128 MiB is far below either model.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
#[ignore = "real models: set MINFER_RSS_TEST_MODEL_A and MINFER_RSS_TEST_MODEL_B"]
fn a_real_model_reload_returns_rss_and_fds_to_baseline() {
    let a = match std::env::var("MINFER_RSS_TEST_MODEL_A") {
        Ok(p) if !p.trim().is_empty() => std::path::PathBuf::from(p),
        _ => {
            eprintln!("skipping: MINFER_RSS_TEST_MODEL_A unset");
            return;
        }
    };
    let b = match std::env::var("MINFER_RSS_TEST_MODEL_B") {
        Ok(p) if !p.trim().is_empty() => std::path::PathBuf::from(p),
        _ => {
            eprintln!("skipping: MINFER_RSS_TEST_MODEL_B unset");
            return;
        }
    };
    assert_ne!(a, b, "the reload gate needs two different models");
    const TOL_KIB: u64 = 128 * 1024;

    let baseline = rss_kib();
    assert_eq!(
        open_fds_to(&a),
        0,
        "model A must not be open before the test"
    );
    assert_eq!(
        open_fds_to(&b),
        0,
        "model B must not be open before the test"
    );

    let load = |p: &std::path::Path| {
        let gguf = load_gguf_model(p).expect("the real GGUF loads");
        let model = crate::models::load_model_with(
            &gguf,
            "",
            crate::graph::offload::OffloadRequest::Layers(0),
        )
        .expect("the real model loads");
        assert_eq!(
            model.device(),
            crate::models::Device::Cpu,
            "the RSS gate keeps the forward on the CPU"
        );
        let mut cache = crate::graph::cache::GraphCache::new();
        let logits = model.forward_graph_cached(&[1], &[0], 1, 32, &mut cache);
        assert!(!logits.is_empty(), "the forward step must produce logits");
        (gguf, model, cache)
    };

    let (gguf_a, model_a, cache_a) = load(&a);
    let a_live = rss_kib();
    assert!(
        a_live > baseline,
        "lifting model A must raise RSS (baseline {baseline} KiB, live {a_live} KiB)"
    );
    drop(cache_a);
    drop(model_a);
    drop(gguf_a);
    trim_allocator();
    let a_after = rss_kib();
    assert_eq!(
        open_fds_to(&a),
        0,
        "model A's file descriptor stayed open after model + cache drop"
    );
    assert!(
        a_after <= baseline + TOL_KIB,
        "model A's mapping stayed resident: baseline {baseline} KiB, after drop {a_after} KiB \
         (tolerance {TOL_KIB} KiB)"
    );

    let (gguf_b, model_b, cache_b) = load(&b);
    let b_live = rss_kib();
    assert!(
        b_live > baseline,
        "lifting model B must raise RSS (baseline {baseline} KiB, live {b_live} KiB)"
    );
    drop(cache_b);
    drop(model_b);
    drop(gguf_b);
    trim_allocator();
    let b_after = rss_kib();
    assert_eq!(
        open_fds_to(&b),
        0,
        "model B's file descriptor stayed open after model + cache drop"
    );
    assert!(
        b_after <= baseline + TOL_KIB,
        "model B's mapping stayed resident: baseline {baseline} KiB, after drop {b_after} KiB \
         (tolerance {TOL_KIB} KiB)"
    );
    eprintln!(
        "rss_reload: baseline={baseline} KiB a_live={a_live} KiB a_after={a_after} KiB \
         b_live={b_live} KiB b_after={b_after} KiB; fds A/B after drop = 0/0"
    );
}

/// Resident set in KiB, on the host that owns the mapping.
#[cfg(target_os = "linux")]
fn rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let line = status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .expect("/proc/self/status has VmRSS");
    line.split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .expect("VmRSS is a number of KiB")
}

#[cfg(target_os = "macos")]
fn rss_kib() -> u64 {
    let out = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("run ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("ps rss is a number of KiB")
}

/// Count the process's open descriptors whose target is `path`.
#[cfg(target_os = "linux")]
fn open_fds_to(path: &std::path::Path) -> usize {
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    std::fs::read_dir("/proc/self/fd")
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| std::fs::read_link(e.path()).ok())
                .filter(|p| p == &target)
                .count()
        })
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn open_fds_to(path: &std::path::Path) -> usize {
    macos_open_fds_to(path)
}

/// glibc retains freed arenas; ask it to return them so the RSS number is the
/// mapping's lifetime, not the allocator's cache. No-op elsewhere.
#[cfg(target_os = "linux")]
extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}

#[cfg(target_os = "linux")]
fn trim_allocator() {
    // SAFETY: glibc's `malloc_trim` only walks its own arenas; the argument is
    // the padding to keep, and 0 asks for the maximum release.
    unsafe {
        malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
fn trim_allocator() {}

// === #767: a parse refusal is a value ===

/// A minimal GGUF header — magic, version, tensor count, KV count — and nothing else.
fn gguf_header(version: u32, n_tensors: i64, n_kv: i64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC);
    b.extend_from_slice(&version.to_le_bytes());
    b.extend_from_slice(&n_tensors.to_le_bytes());
    b.extend_from_slice(&n_kv.to_le_bytes());
    b
}

/// Append one scalar `uint8` metadata pair: key string, type, value.
fn kv_u8(b: &mut Vec<u8>, key: &str, value: u8) {
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key.as_bytes());
    b.extend_from_slice(&(GgufType::Uint8 as i32).to_le_bytes());
    b.push(value);
}

/// Append one `uint32` array metadata pair: key string, `Array`, element type, count, values.
fn kv_array_u32(b: &mut Vec<u8>, key: &str, values: &[u32]) {
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key.as_bytes());
    b.extend_from_slice(&(GgufType::Array as i32).to_le_bytes());
    b.extend_from_slice(&(GgufType::Uint32 as i32).to_le_bytes());
    b.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        b.extend_from_slice(&v.to_le_bytes());
    }
}

/// Append one tensor-index entry: name, dims, ggml type, data-section offset.
fn tensor_entry(b: &mut Vec<u8>, name: &str, dims: &[i64], type_: i32, offset: u64) {
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&type_.to_le_bytes());
    b.extend_from_slice(&offset.to_le_bytes());
}

#[test]
fn every_truncation_of_a_gguf_header_is_a_named_refusal() {
    // The header is 24 bytes. Every shorter prefix must be a refusal that says which
    // stage it hit — no silent reason, no panic, no thread-local note ([#767]).
    let full = gguf_header(GGUF_VERSION, 0, 0);
    assert_eq!(full.len(), 24);
    for len in 0..full.len() {
        let err = GgufContext::init_from_data(&full[..len]).unwrap_err();
        assert!(!err.is_empty(), "a {len}-byte prefix must carry a reason");
        assert!(
            err.starts_with("GGUF: failed to read header: "),
            "a {len}-byte prefix must name the stage that refused: {err}"
        );
    }
    GgufContext::init_from_data(&full).expect("the 24-byte header parses");
}

#[test]
fn a_bad_magic_is_refused_with_the_magic_it_found() {
    let err = GgufContext::init_from_data(b"this is not a GGUF file at all\n").unwrap_err();
    assert_eq!(
        err,
        "GGUF: invalid magic characters: 'this', expected 'GGUF'"
    );
}

#[test]
fn an_unsupported_version_names_the_version() {
    // Each of these used to be overwritten by a later aggregate note, so a caller could
    // not tell a v1 file from a truncated one ([#767]).
    let cases = [
        (1u32, "GGUF: GGUFv1 is no longer supported"),
        (GGUF_VERSION + 1, "this GGUF file is version 4"),
        (
            0x0001_0000,
            "a mismatch between the host and model endianness",
        ),
    ];
    for (version, expected) in cases {
        let err = GgufContext::init_from_data(&gguf_header(version, 0, 0)).unwrap_err();
        assert!(err.contains(expected), "version {version}: {err}");
    }
    // Version 0 takes its own branch even though its low half is zero too.
    let err = GgufContext::init_from_data(&gguf_header(0, 0, 0)).unwrap_err();
    assert_eq!(err, "GGUF: bad GGUF version: 0");
}

#[test]
fn a_truncated_metadata_block_reports_what_it_needed() {
    let err = GgufContext::init_from_data(&gguf_header(GGUF_VERSION, 0, 1)).unwrap_err();
    assert_eq!(
        err,
        "GGUF: unexpected end of file: needed 8 bytes at offset 24, 0 remain"
    );
}

#[test]
fn a_duplicate_metadata_key_is_refused_by_name() {
    let mut b = gguf_header(GGUF_VERSION, 0, 2);
    kv_u8(&mut b, "general.name", 1);
    kv_u8(&mut b, "general.name", 2);
    let err = GgufContext::init_from_data(&b).unwrap_err();
    assert_eq!(
        err,
        "GGUF: duplicate key 'general.name' for tensors 0 and 1"
    );
}

#[test]
fn an_array_shaped_alignment_is_refused_not_asserted() {
    // `get_val_u32` asserts a scalar shape; the parser refuses instead of panicking.
    let mut b = gguf_header(GGUF_VERSION, 0, 1);
    kv_array_u32(&mut b, "general.alignment", &[32]);
    let err = GgufContext::init_from_data(&b).unwrap_err();
    assert!(err.contains("general.alignment"), "{err}");
    assert!(err.contains("must be a single uint32"), "{err}");
}

#[test]
fn a_tensor_with_an_unknown_type_is_refused_not_a_panic() {
    let mut b = gguf_header(GGUF_VERSION, 1, 0);
    tensor_entry(&mut b, "blk.0.attn_norm.weight", &[4], 99, 0);
    let err = GgufContext::init_from_data(&b).unwrap_err();
    assert_eq!(err, "GGUF: invalid GGML type: 99");
}

#[test]
fn a_missing_file_is_a_load_error_that_names_the_path() {
    let path = std::env::temp_dir().join("minfer-767-does-not-exist.gguf");
    let _ = std::fs::remove_file(&path);
    let err = match load_gguf_model(&path) {
        Ok(_) => panic!("a missing file must not load"),
        Err(e) => e,
    };
    assert!(err.reason.contains("cannot open"), "{}", err.reason);
    assert!(
        err.reason.contains("minfer-767-does-not-exist.gguf"),
        "{}",
        err.reason
    );
    assert_eq!(err.path, path);
}
