//! Layer offload (E5) and the asynchronous cross-backend copies (F5).
//!
//! Split out of `src/models/qwen2/graph/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// E5's acceptance on the real model: a **chosen** number of blocks on the device and the
/// rest on the CPU runs end to end, the scheduler splits at every block boundary (its
/// cross-backend copies are what make the mixed graph executable at all), the report says
/// which blocks landed where, and the greedy tokens match the all-CPU run of the same
/// prompt.
///
/// CPU and device logits differ by design (rule 9: the CPU quantizes activations to Q8_0,
/// the device reads f32), so this compares **greedy tokens**, not logits. The two loads
/// use different registry namespaces (`cpuref.`) so the name-keyed device registry cannot
/// make the CPU-only model look registered.
///
/// Ignored because it needs the cached 0.5B and a CUDA device; run it alone:
///
/// ```text
/// cargo test --release a_partial_offload_runs_the_rest_on_the_cpu -- --ignored --test-threads=1
/// ```
#[test]
#[ignore = "requires the cached 0.5B model and a CUDA device"]
fn a_partial_offload_runs_the_rest_on_the_cpu() {
    use crate::graph::cache::GraphCache;
    use crate::graph::offload::OffloadRequest;
    use crate::graph::scheduler::BackendScheduler;
    use crate::graph::Backend;
    use crate::models::{Device, ModelDef};

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the E5 offload gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let n_layers = crate::models::qwen2::loader::hparams_from_gguf(&gguf.parts[0].ctx)
        .expect("hparams")
        .n_layer as usize;
    let k = 4.min(n_layers);

    // The all-CPU reference, under its own namespace (see the doc comment).
    let reference = crate::models::qwen2::loader::load(&gguf, "cpuref.", OffloadRequest::Layers(0))
        .expect("load the CPU reference");
    assert_eq!(reference.device(), Device::Cpu);
    assert_eq!(reference.offload.plan.gpu_layers, 0);

    // The mixed model: `k` blocks on the device, the rest on the CPU.
    let mixed = crate::models::qwen2::loader::load(&gguf, "", OffloadRequest::Layers(k))
        .expect("load the mixed model");
    if mixed.device() != Device::Cuda {
        eprintln!(
            "no CUDA participation (device {:?}); skipping",
            mixed.device()
        );
        return;
    }
    let plan = mixed.offload.plan;
    assert_eq!((plan.gpu_layers, plan.cpu_layers()), (k, n_layers - k));
    let report = mixed
        .offload_report()
        .expect("a mixed plan must report where its blocks landed");
    assert!(
        report.contains(&format!("{k} of {n_layers} blocks")),
        "{report}"
    );
    assert!(report.contains("on cpu"), "{report}");
    eprintln!("[e5] {report}");

    // It registered the **offloaded blocks'** tensors and nothing else: at least the sum
    // of those blocks' weights (more, because the fused `attn_qkv`/`ffn_gu` concat copies
    // are registered too) and less than the whole model (a partial plan leaves
    // `token_embd`/`output` and the remaining blocks on the CPU).
    let block_bytes = |m: &Qwen2Model, upto: usize| -> usize {
        m.layers
            .iter()
            .take(upto)
            .flat_map(|l| {
                [
                    &l.attn_norm,
                    &l.wq,
                    &l.bq,
                    &l.wk,
                    &l.bk,
                    &l.wv,
                    &l.bv,
                    &l.wo,
                    &l.ffn_norm,
                    &l.ffn_gate,
                    &l.ffn_up,
                    &l.ffn_down,
                ]
            })
            .filter_map(|t| t.as_ref())
            .map(|t| t.data().len())
            .sum()
    };
    assert_eq!(
        reference.offload.device_bytes, 0,
        "the CPU reference registers nothing"
    );
    let offloaded = block_bytes(&mixed, k);
    let all_blocks = block_bytes(&mixed, n_layers);
    assert!(
        mixed.offload.device_bytes >= offloaded,
        "the offloaded blocks' weights must be on the device: {} < {offloaded}",
        mixed.offload.device_bytes
    );
    assert!(
        mixed.offload.device_bytes < all_blocks,
        "a partial plan must leave the remaining blocks off the device: {} >= {all_blocks}",
        mixed.offload.device_bytes
    );

    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let n_ctx = 256;
    let steps = 4;

    // Run both, greedily, and compare the token sequences. The plan is passed in
    // because the closure takes `&dyn ModelDef` and since #244 the plan surface is
    // the concrete `model.offload.plan` field, not a trait method.
    let run = |model: &dyn ModelDef, plan: crate::graph::offload::OffloadPlan| -> Vec<u32> {
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        let mut l =
            model.forward_graph_cached(&ids, &(0..n).collect::<Vec<_>>(), 1, n_ctx, &mut cache);
        let mut next = argmax(&l);
        let mut toks = vec![next];
        for s in 0..steps {
            l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
            next = argmax(&l);
            toks.push(next);
        }
        // The first forward built the graph; inspect its assignment while the cache is
        // still alive (this is the only place the built graph is reachable).
        let (graph, _alloc) = cache.current().expect("a built graph");
        let n_cuda = graph
            .nodes
            .iter()
            .filter(|nd| nd.backend == Some(Backend::CUDA))
            .count();
        let n_cpu = graph
            .nodes
            .iter()
            .filter(|nd| nd.backend == Some(Backend::CPU))
            .count();
        let splits = BackendScheduler::new().split_graph(graph);
        let cuda_splits: Vec<_> = splits
            .iter()
            .filter(|s| s.backend == Backend::CUDA)
            .collect();
        let cpu_splits: Vec<_> = splits
            .iter()
            .filter(|s| s.backend == Backend::CPU)
            .collect();
        eprintln!(
            "[e5] graph: {n_cuda} nodes on cuda, {n_cpu} on cpu; {} splits ({} cuda, {} cpu)",
            splits.len(),
            cuda_splits.len(),
            cpu_splits.len()
        );
        if plan.is_mixed() {
            assert!(
                n_cuda > 0 && n_cpu > 0,
                "a mixed plan must place nodes on both"
            );
            // The split count is not exactly one per block (a block's device nodes are
            // contiguous with its neighbours' when the intervening nodes are on the
            // device too), so the claim is the alternation itself plus the copies below.
            assert!(
                cuda_splits.len() >= 2 && cpu_splits.len() >= 2,
                "a mixed plan must alternate: {} device / {} CPU splits for {k} blocks",
                cuda_splits.len(),
                cpu_splits.len()
            );
            // The placement contract, on a real model graph: no node of a non-offloaded
            // block (and no node outside any block) is on the device.
            for nd in &graph.nodes {
                let on_cpu_side = match nd.layer {
                    Some(l) => l >= k,
                    None => true,
                };
                if on_cpu_side {
                    assert_eq!(
                        nd.backend,
                        Some(Backend::CPU),
                        "block {:?} node '{}' must stay on the CPU under a partial plan",
                        nd.layer,
                        nd.name
                    );
                }
            }
            assert!(
                cuda_splits.iter().any(|s| !s.inputs.is_empty()),
                "a device split after a CPU block must take a cross-backend copy in"
            );
            assert!(
                cpu_splits.iter().any(|s| !s.inputs.is_empty()),
                "the CPU block after a device block must take a cross-backend copy in"
            );
        }
        toks
    };

    let want = run(&reference, reference.offload.plan);
    let got = run(&mixed, mixed.offload.plan);
    assert_eq!(
        got, want,
        "a partial offload must produce the same greedy tokens as the all-CPU run"
    );
    eprintln!("[e5] {k}/{n_layers} blocks on cuda, {steps} greedy steps match the CPU run");
}
/// F5 ([#58]) + #138 ([#138]) acceptance on the real model: a split graph's
/// cross-backend staging copies are **asynchronous**, every staged input owes
/// exactly one event wait — issued since #138 at the consumer's first use rather
/// than at the split boundary — the boundary close no longer blocks the host, and
/// the results are **bitwise identical** to the pre-F5 synchronous reference.
///
/// The test runs the *same* mixed model (the E5 offload plan, so the graph
/// really does alternate CPU and CUDA splits) twice — once with the async
/// substrate, once with `MINFER_SYNC_COPIES`'s synchronous host round trip
/// forced for the duration — and compares the logits of every step. Both modes
/// run the identical kernels in the identical order; only the *transfer* of a
/// cross-boundary value differs, so equality is the honest claim (this is not
/// the rule-9 CPU-vs-GPU comparison, which legitimately differs).
///
/// What it asserts, and why each half matters:
///
/// - **the missing-wait gate** — `copies == waits` (one phase-B wait per
///   phase-A copy) and `waits > 0`. The counts are deterministic; removing the
///   consumer path's resolver (a bare `cross_input` in the node loop) makes them
///   unequal *and* leaves the staging entry pending, which
///   `GraphAllocator::cross_input` turns into a loud error the moment the
///   consumer reads it. The mutation was recorded.
/// - **the deferral (#138)** — `deferred_waits >= async_host_copies`: every
///   device→host copy is read by the host consumer, so its wait is issued at that
///   read (the boundary issues none). The entries nothing read are drained at the
///   end of the execution and show up as the `waits - deferred_waits` difference.
/// - **the no-blocking-copy gate** — `blocking_host_copies == 0` in async mode
///   and `== copies` in sync mode, plus the device-level
///   `CudaBackend::blocking_readback_count()` (which counts actual blocking
///   `cudaMemcpy` D2H calls) not moving at all in async mode. Those are the
///   measured before/after numbers.
/// - **the host stalls** — the backend's own `stream_sync_count()` around the
///   run. The pre-F5 path synced the whole stream once *per staged input
///   inside* `copy_to_host`; the async path issues none of those, and since #138
///   the boundary close issues none either (the copies are stream-ordered behind
///   the producer, so that sync was redundant). This is the latency-shaped
///   evidence, and it is a hard count, not a timing. It is read **through the
///   backend** ([#185]): a process-wide counter let a concurrent device test's
///   stalls land between the two snapshots (run A read 4160 async vs 728 sync —
///   the harness's own load, not the async path). That process-wide counter was
///   deleted in [#242].
///
/// [#138]: https://github.com/yusiwen/minfer/issues/138
/// [#185]: https://github.com/yusiwen/minfer/issues/185
/// [#242]: https://github.com/yusiwen/minfer/issues/242
///
/// Ignored because it needs the cached 0.5B and a CUDA device. Run alone:
///
/// ```text
/// cargo test --release --features cuda async_cross_copies_never_block -- --ignored --test-threads=1
/// ```
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires the cached 0.5B model and a CUDA device"]
fn async_cross_copies_never_block_and_stay_bitwise_identical() {
    use crate::graph::cache::GraphCache;
    use crate::graph::copystats::{self, CrossCopyStats};
    use crate::graph::offload::OffloadRequest;
    use crate::models::{Device, ModelDef};

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the F5 async-copy gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let n_layers = crate::models::qwen2::loader::hparams_from_gguf(&gguf.parts[0].ctx)
        .expect("hparams")
        .n_layer as usize;
    // A prefix of blocks on the device: the graph then alternates CPU → CUDA →
    // CPU, so both copy directions are exercised.
    let k = 4.min(n_layers);
    let model = crate::models::qwen2::loader::load(&gguf, "", OffloadRequest::Layers(k))
        .expect("load the mixed model");
    if model.device() != Device::Cuda {
        eprintln!(
            "no CUDA participation (device {:?}); skipping",
            model.device()
        );
        return;
    }

    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let n_ctx = 256;
    let steps = 6;

    // One mode = one fresh cache (so the graph is built and then reused across
    // the decode steps, which is the steady state the boundary lives in) and
    // one logits vector per step.
    // The mode override is process-wide; hold the gate for the whole
    // comparison so no other test flips it mid-measurement.
    let gate = copystats::gate();
    let run = |sync: bool| -> (Vec<Vec<f32>>, CrossCopyStats, u64, u64) {
        let _mode = copystats::set_sync_for_test(sync);
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        // Bring the device pool up before the first forward, so the
        // device-level readback counter below has an instance to read. The
        // graph builder enables it anyway (idempotently).
        assert!(
            cache.alloc().enable_cuda(),
            "a CUDA device is participating"
        );
        let before = cache.alloc().cross_stats();
        let readbacks_before = cache
            .alloc()
            .cuda()
            .expect("the CUDA pool is enabled")
            .blocking_readback_count();
        // #185: read the stall count through **this** backend, not a
        // process-wide total: a concurrent device test's syncs landed inside
        // this delta and made the async arm look worse than the synchronous one
        // (4160 vs 728 in the ticket's run A).
        let syncs_before = cache
            .alloc()
            .cuda()
            .expect("the CUDA pool is enabled")
            .stream_sync_count();

        let mut l =
            model.forward_graph_cached(&ids, &(0..n).collect::<Vec<_>>(), 1, n_ctx, &mut cache);
        let mut out = vec![l.clone()];
        let mut next = argmax(&l);
        for s in 0..steps {
            l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
            out.push(l.clone());
            next = argmax(&l);
        }

        let stats = cache.alloc().cross_stats().delta(before);
        let readbacks = cache
            .alloc()
            .cuda()
            .expect("the CUDA pool is enabled")
            .blocking_readback_count()
            - readbacks_before;
        let syncs = cache
            .alloc()
            .cuda()
            .expect("the CUDA pool is enabled")
            .stream_sync_count()
            - syncs_before;
        (out, stats, readbacks, syncs)
    };

    let (async_logits, a, a_readbacks, a_syncs) = run(false);
    let (sync_logits, s, s_readbacks, s_syncs) = run(true);
    drop(gate);

    eprintln!(
        "[f5] async: copies={} waits={} deferred_waits={} blocking_host_copies={} \
         async_host_copies={} event_syncs={} stream_waits={} blocking_readbacks={} stream_syncs={}",
        a.copies,
        a.waits,
        a.deferred_waits,
        a.blocking_host_copies,
        a.async_host_copies,
        a.event_syncs,
        a.stream_waits,
        a_readbacks,
        a_syncs
    );
    eprintln!(
        "[f5] sync : copies={} waits={} blocking_host_copies={} async_host_copies={} \
         blocking_readbacks={} stream_syncs={}",
        s.copies, s.waits, s.blocking_host_copies, s.async_host_copies, s_readbacks, s_syncs
    );

    // The workload really does cross a backend boundary (otherwise every
    // assertion below would be vacuously true).
    assert!(
        a.copies > 0,
        "the mixed offload graph must stage at least one cross-backend copy (got {})",
        a.copies
    );

    // ── the missing-wait gate ────────────────────────────────────────────
    assert!(
        a.all_copies_awaited(),
        "every staged input must be waited on: {} copies, {} waits",
        a.copies,
        a.waits
    );
    assert_eq!(
        s.copies, s.waits,
        "the synchronous reference issues the same one-wait-per-copy contract"
    );
    // ── the deferred-wait half (#138) ────────────────────────────────────
    // Every device→host copy is read by the host consumer, so its wait is
    // issued at that read. The CPU→device entries are a mixed bag: the device
    // split's node loop resolves them while it runs, and a split that replayed a
    // captured graph has none — those are drained at the end of the execution and
    // are the reason `deferred_waits` is a floor here, not an equality.
    assert!(
        a.deferred_waits >= a.async_host_copies,
        "the host consumer's reads must be what issues the device→host waits: \
         {} deferred vs {} async device→host copies",
        a.deferred_waits,
        a.async_host_copies
    );

    // ── the no-blocking-copy-on-the-hot-path gate ────────────────────────
    assert_eq!(
        a.blocking_host_copies, 0,
        "the async boundary must issue no blocking device→host copy (got {})",
        a.blocking_host_copies
    );
    assert!(
        a.async_host_copies > 0,
        "the device→host direction must have taken the async path"
    );
    assert_eq!(
        a.event_syncs, a.async_host_copies,
        "each async device→host copy owes exactly one event wait"
    );
    assert_eq!(
        a_readbacks, 0,
        "no blocking device readback (`copy_to_host`) may happen on the hot path (got {a_readbacks})"
    );
    // Only the device→host copies block, so the synchronous reference's count
    // is positive but never more than its total copies (CPU→device copies go
    // through the destination pool's own stream-ordered fill, which never
    // blocked the host either).
    assert!(
        s.blocking_host_copies > 0,
        "the synchronous reference must block on its device→host copies — this is the \
         'before' number (got 0 of {} copies)",
        s.copies
    );
    assert!(s.blocking_host_copies <= s.copies);
    assert_eq!(s.async_host_copies, 0);
    assert!(
        s_readbacks > 0,
        "the device-level counter must see the synchronous path's blocking readbacks"
    );

    // ── host stalls removed ──────────────────────────────────────────────
    // The pre-F5 path synced the whole stream once per staged input inside
    // `copy_to_host`; F5 removed those, and #138 removed the boundary's own sync
    // too (the copies are stream-ordered behind the producer). What is left is
    // what a backend's own submission needs.
    assert!(
        a_syncs < s_syncs,
        "the async path must issue strictly fewer full stream syncs ({a_syncs} vs {s_syncs})"
    );
    assert!(
        s_syncs >= s.blocking_host_copies,
        "the synchronous reference pays one stream sync per blocking copy: {s_syncs} syncs \
         for {} blocking copies",
        s.blocking_host_copies
    );
    // ── bitwise equality ─────────────────────────────────────────────────
    assert_eq!(async_logits.len(), sync_logits.len());
    let mut worst = 0.0f32;
    for (i, (x, y)) in async_logits.iter().zip(&sync_logits).enumerate() {
        let d = max_delta(x, y);
        worst = worst.max(d);
        assert_eq!(
            d, 0.0,
            "step {i}: the async staging copies must be bitwise identical to the \
             synchronous reference (max |Δlogit| = {d})"
        );
    }
    eprintln!(
        "[f5] {steps} decode steps + 1 prefill over {k}/{n_layers} device blocks: \
         max |Δlogit| = {worst}; stream syncs {} -> {} (-{}), blocking D2H copies {} -> {}",
        a_syncs,
        s_syncs,
        s_syncs - a_syncs,
        s.blocking_host_copies,
        a.blocking_host_copies
    );
}
/// F5 ([#58]) + #138 ([#138]) acceptance on the real model, **Metal source** —
/// the port of [#137] ([#137]): a split graph's cross-backend staging copies out
/// of Metal are asynchronous (a `MTLBlitCommandEncoder` copy into a shared
/// staging buffer plus an `MTLSharedEvent` signal), every staged input owes
/// exactly one event wait — issued since #138 at the consumer's first use rather
/// than at the split boundary — and the results are **bitwise identical** to the
/// synchronous reference over a repeated loop.
///
/// This is the same comparison the CUDA gate
/// (`async_cross_copies_never_block_and_stay_bitwise_identical`) makes, with the
/// two Metal-specific differences the port forces:
///
/// - the **device-level evidence** is `MetalBackend::sync_readback_count()`:
///   Metal's `MTLBuffer`s are `StorageModeShared` and there is no blocking
///   `cudaMemcpy` API to count, so the counter counts the host read of a pool
///   buffer (`Backend::read_host`), which is exactly what the synchronous
///   boundary path does. The async path publishes its own staging bytes and
///   never moves it.
/// - there is no CUDA-style `stream_sync_count` to compare: Metal has no
///   separate stream-sync counter, so the "no host stall" claim rests on the
///   readback counter and on `blocking_host_copies == 0`.
///
/// Ignored because it needs the cached 0.5B and a Metal device. Run alone:
///
/// ```text
/// cargo test --release --bin minfer async_cross_copies_never_block_and_stay_bitwise_identical_on_metal -- --ignored --test-threads=1
/// ```
///
/// [#137]: https://github.com/yusiwen/minfer/issues/137
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires the cached 0.5B model and a Metal device"]
fn async_cross_copies_never_block_and_stay_bitwise_identical_on_metal() {
    use crate::graph::cache::GraphCache;
    use crate::graph::copystats::{self, CrossCopyStats};
    use crate::graph::offload::OffloadRequest;
    use crate::models::{Device, ModelDef};

    let _g = crate::metal::metal_test_lock();
    // The Metal device must be up **before** the load, or the loader decides the
    // weights are not usable there and answers `Device::Cpu`.
    crate::metal::MpsState::init();
    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the Metal F5 async-copy gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let n_layers = crate::models::qwen2::loader::hparams_from_gguf(&gguf.parts[0].ctx)
        .expect("hparams")
        .n_layer as usize;
    // A prefix of blocks on the device: the graph then alternates CPU → Metal →
    // CPU, so both copy directions are exercised. The `f5metal.` registry
    // namespace keeps this load's Metal weights out of the way of the other
    // real-model gates that run concurrently in the parallel ignored-set harness
    // (`MpsState`'s weight table is process-global and name-keyed, so two engines
    // loaded without a namespace overwrite each other).
    let k = 4.min(n_layers);
    let model = crate::models::qwen2::loader::load(&gguf, "f5metal.", OffloadRequest::Layers(k))
        .expect("load the mixed model");
    if model.device() != Device::Metal {
        eprintln!(
            "no Metal participation (device {:?}); skipping",
            model.device()
        );
        return;
    }

    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let n_ctx = 256;
    let steps = 6;
    // The mode override is process-wide; hold the gate for the whole comparison
    // so no other test flips it mid-measurement.
    let gate = copystats::gate();
    let run = |sync: bool| -> (Vec<Vec<f32>>, CrossCopyStats, u64) {
        let _mode = copystats::set_sync_for_test(sync);
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        assert!(
            cache.alloc().enable_metal(),
            "a Metal device is participating"
        );
        let before = cache.alloc().cross_stats();
        let readbacks_before = cache
            .alloc()
            .metal()
            .expect("the Metal pool is enabled")
            .sync_readback_count();

        let mut l =
            model.forward_graph_cached(&ids, &(0..n).collect::<Vec<_>>(), 1, n_ctx, &mut cache);
        let mut out = vec![l.clone()];
        let mut next = argmax(&l);
        for s in 0..steps {
            l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
            out.push(l.clone());
            next = argmax(&l);
        }

        let stats = cache.alloc().cross_stats().delta(before);
        let readbacks = cache
            .alloc()
            .metal()
            .expect("the Metal pool is enabled")
            .sync_readback_count()
            - readbacks_before;
        (out, stats, readbacks)
    };

    let (async_logits, a, a_readbacks) = run(false);
    let (sync_logits, s, s_readbacks) = run(true);
    drop(gate);

    eprintln!(
        "[f5-metal] async: copies={} waits={} deferred_waits={} blocking_host_copies={} \
         async_host_copies={} event_syncs={} stream_waits={} sync_readbacks={}",
        a.copies,
        a.waits,
        a.deferred_waits,
        a.blocking_host_copies,
        a.async_host_copies,
        a.event_syncs,
        a.stream_waits,
        a_readbacks
    );
    eprintln!(
        "[f5-metal] sync : copies={} waits={} blocking_host_copies={} async_host_copies={} \
         sync_readbacks={}",
        s.copies, s.waits, s.blocking_host_copies, s.async_host_copies, s_readbacks
    );

    // The workload really does cross a backend boundary.
    assert!(
        a.copies > 0,
        "the mixed offload graph must stage at least one cross-backend copy (got {})",
        a.copies
    );

    // ── the missing-wait gate ────────────────────────────────────────────
    assert!(
        a.all_copies_awaited(),
        "every staged input must be waited on: {} copies, {} waits",
        a.copies,
        a.waits
    );
    assert_eq!(
        s.copies, s.waits,
        "the synchronous reference issues the same one-wait-per-copy contract"
    );
    // ── the deferred-wait half (#138) ────────────────────────────────────
    assert!(
        a.deferred_waits >= a.async_host_copies,
        "the host consumer's reads must be what issues the device→host waits: \
         {} deferred vs {} async device→host copies",
        a.deferred_waits,
        a.async_host_copies
    );

    // ── the no-blocking-copy-on-the-hot-path gate ────────────────────────
    assert_eq!(
        a.blocking_host_copies, 0,
        "the async boundary must issue no blocking device→host copy (got {})",
        a.blocking_host_copies
    );
    assert!(
        a.async_host_copies > 0,
        "the Metal→host direction must have taken the async path"
    );
    assert_eq!(
        a.event_syncs, a.async_host_copies,
        "each async device→host copy owes exactly one event wait"
    );
    assert_eq!(
        a_readbacks, 0,
        "the async path must not read a Metal pool buffer back to the host (got {a_readbacks})"
    );
    assert!(
        s.blocking_host_copies > 0,
        "the synchronous reference must block on its Metal→host copies — this is the \
         'before' number (got 0 of {} copies)",
        s.copies
    );
    assert!(s.blocking_host_copies <= s.copies);
    assert_eq!(s.async_host_copies, 0);
    assert!(
        s_readbacks > 0,
        "the device-level counter must see the synchronous path's readbacks"
    );

    // ── bitwise equality over the repeated loop ──────────────────────────
    assert_eq!(async_logits.len(), sync_logits.len());
    let mut worst = 0.0f32;
    for (i, (x, y)) in async_logits.iter().zip(&sync_logits).enumerate() {
        let d = max_delta(x, y);
        worst = worst.max(d);
        assert_eq!(
            d, 0.0,
            "step {i}: the async Metal staging copies must be bitwise identical to the \
             synchronous reference (max |Δlogit| = {d})"
        );
    }
    eprintln!(
        "[f5-metal] {steps} decode steps + 1 prefill over {k}/{n_layers} Metal blocks: \
         max |Δlogit| = {worst}; blocking host copies {} -> {}, device readbacks {} -> {}",
        s.blocking_host_copies, a.blocking_host_copies, s_readbacks, a_readbacks
    );
}
/// E5 S2's acceptance on the real model: **`auto` picks the block count from the budget**.
///
/// The gate pins a small budget with `OffloadRequest::AutoWithBudget(64)` so the fit is a
/// strict prefix (`0 < k < n_layer`) instead of the trivial "everything fits" a 128 GB
/// device gives, checks the startup line names the fit and its numbers, and requires the
/// same greedy tokens as the all-CPU run (the S1 gate's comparison). A budget that covers
/// the model must select every block — the pre-E5 behaviour, so an `auto` default cannot
/// silently under-offload.
///
/// Issue #185: the budget is an **explicit argument**, never `MINFER_GPU_MEM`. The
/// environment is process-wide, so the earlier form (`set_var` for the capped arm,
/// `remove_var` for the uncapped one) changed what a concurrently loading test computed —
/// one of the four failures that made the harness look flaky. The explicit-argument
/// convention is the repo's (`load_model_configured`'s cache type, #99/#153), and the
/// second arm uses the device's measured free bytes for the same reason.
///
/// Ignored because it needs the cached 0.5B and a CUDA device.
#[test]
#[ignore = "requires the cached 0.5B model and a CUDA device"]
fn an_auto_offload_plan_fits_the_budget() {
    use crate::graph::cache::GraphCache;
    use crate::graph::offload::OffloadRequest;
    use crate::models::{Device, ModelDef};

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the E5 S2 auto gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let n_layers = crate::models::qwen2::loader::hparams_from_gguf(&gguf.parts[0].ctx)
        .expect("hparams")
        .n_layer as usize;
    // A budget far below the model (24 blocks of ~16 MiB) but far above one block.
    // Explicit argument, not `MINFER_GPU_MEM` (#185).
    const CAP_MIB: usize = 64;

    let cpu_ref =
        crate::models::qwen2::loader::load(&gguf, "auto-cpuref.", OffloadRequest::Layers(0))
            .expect("load the CPU reference");
    let mixed =
        crate::models::qwen2::loader::load(&gguf, "", OffloadRequest::AutoWithBudget(CAP_MIB))
            .expect("load the auto model");
    if mixed.device() != Device::Cuda {
        eprintln!(
            "no CUDA participation (device {:?}); skipping",
            mixed.device()
        );
        return;
    }
    let plan = mixed.offload.plan;
    assert!(
        plan.gpu_layers > 0 && plan.gpu_layers < n_layers,
        "a 64 MiB budget must be a strict prefix, got {plan:?}"
    );
    let report = mixed.offload_report().expect("auto must report its fit");
    assert!(
        report.contains("auto:")
            && report.contains("explicit budget 64 MiB")
            && !report.contains("MINFER_GPU_MEM="),
        "{report}"
    );
    eprintln!("[e5-s2] {report}");

    // It runs, and its greedy tokens match the all-CPU run (device logits differ by design).
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let n_ctx = 256;
    let drive = |model: &dyn ModelDef| -> Vec<u32> {
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        let mut l =
            model.forward_graph_cached(&ids, &(0..n).collect::<Vec<_>>(), 1, n_ctx, &mut cache);
        let mut next = argmax(&l);
        let mut toks = vec![next];
        for s in 0..4 {
            l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
            next = argmax(&l);
            toks.push(next);
        }
        toks
    };
    assert_eq!(
        drive(&mixed),
        drive(&cpu_ref),
        "an auto fit must produce the same greedy tokens as the all-CPU run"
    );

    // A budget that covers the model must offload every block — the pre-E5 `auto`
    // behaviour. It is expressed as the device's **measured free bytes** rather than by
    // *unsetting* `MINFER_GPU_MEM`: the explicit argument neither reads nor mutates the
    // process-global (#185).
    let free_mib = match crate::models::device_memory() {
        crate::graph::allocplan::DeviceMemory::Reported { free, .. } => free / (1024 * 1024),
        other => panic!("`auto` needs a device that reports free bytes, got {other:?}"),
    };
    assert!(free_mib > 0, "the device must report free bytes");
    let full = crate::models::qwen2::loader::load(
        &gguf,
        "auto-full.",
        OffloadRequest::AutoWithBudget(free_mib),
    )
    .expect("load the auto (whole-device-budget) model");
    assert_eq!(
        full.offload.plan.gpu_layers,
        n_layers,
        "a budget covering the device's free bytes must offload every block: {}",
        full.offload_report().unwrap_or_default()
    );
}
