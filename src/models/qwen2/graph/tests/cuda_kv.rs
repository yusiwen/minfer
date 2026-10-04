//! CUDA-only gates: the packed cache, two engines with different KV layouts, and a session resumed from disk.
//!
//! Split out of `src/models/qwen2/graph/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// C4's acceptance on the real model: the packed regions are measurably smaller,
/// and the Q8_0 cache stays inside a **named logit tolerance** of the f32 one.
/// The class is not bitwise, and not greedy equality either: the store rounds
/// every K/V cell, so an argmax can legitimately flip a few tokens in (the CLI
/// does exactly that on a chat-templated prompt — 5 tokens, then EOS on the
/// 0.5B), which is why the continuation agreement is *reported* and the
/// perturbation's size is asserted.
///
/// **C4 S2 adds a second term and a second arm.** The fused read quantizes the
/// query row too (the K score is a `Q8_0 × Q8_0` dot), so the tail of the logit
/// vector moves a little more than S1's scratch path did: measured 3.029 against
/// S1's 2.505 on this gate (the decoding-relevant value at the reference's argmax
/// moved 0.604 → 0.646). `MINFER_NO_FUSED_Q8_KV=1` runs the S1 path through the
/// same gate, which is where that pair of numbers comes from. The second arm is
/// the physical shift: `kv_rm` moves the survivors verbatim and re-ropes each
/// survivor's K through dequantize → requantize, and the shifted continuation must
/// stay in the same class as the unshifted one.
///
/// What pins the format itself is one level down: `cpu_backend`'s
/// `a_packed_kv_region_answers_like_the_f32_one_and_is_smaller` asserts that a
/// stored cell is **bitwise** the Q8_0 quantizate of the row it was given, and
/// `a_packed_physical_shift_moves_v_verbatim_and_requantizes_k` does the same for
/// the shift's two halves (V verbatim, K the quantizate of the re-roped row).
///
/// **Per-engine since #99; per-engine on the device too since #153.** This gate
/// used to flip the process-wide KV format for its measurement runs, which sized
/// every other test's KV regions for the wrong format when the harness ran the
/// ignored set in parallel. It now loads **two engines per arm** — one resolved
/// for `f32`, one for `q8_0` — and flips nothing, so the same gate is the proof
/// that two formats coexist in one process. #153 removed the last process-wide
/// tag (`cuda::KV_LAYOUT`): each engine's `kv_format` reaches its own CUDA
/// backend through `GraphAllocator::set_kv_format`, so the device arm's two
/// engines address their regions in the layout each named without any mutation.
///
/// Ignored because it needs the cached 0.5B model; it is **device-aware**: the
/// CPU arm asks for `--gpu-layers 0` (coverage on every build), and on a CUDA
/// build a second arm asserts `device() == Cuda` so a silent CPU fallback fails
/// loudly instead of reporting a CPU number as a device one. `MINFER_C4_MODEL`
/// points it at another cached model:
///
/// ```text
/// cargo test --release a_packed_kv_cache_answers_like_the_f32_one -- --ignored --test-threads=1
/// MINFER_C4_MODEL=~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
///   cargo test --release a_packed_kv_cache_answers_like_the_f32_one -- --ignored --test-threads=1
/// ```
#[test]
#[ignore = "requires the cached 0.5B model"]
fn a_packed_kv_cache_answers_like_the_f32_one() {
    use crate::graph::cache::GraphCache;
    use crate::graph::kvformat::KvFormat;
    use crate::graph::offload::OffloadRequest;
    use crate::models::{Device, ModelDef};

    let Some(path) = std::env::var_os("MINFER_C4_MODEL")
        .map(std::path::PathBuf::from)
        .or_else(cached_model_path)
    else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the C4 packed-cache gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");

    // #153: a KV layout is **one** policy per engine now. The engine's
    // `kv_format` sizes the region (packed or f32-shaped) **and** picks the CUDA
    // kernel that reads it — `GraphAllocator::set_kv_format` stamps both the CPU
    // backend and (when it exists) the CUDA backend's `KV_LAYOUT_*` tag from
    // `model.kv_format`, and `enable_cuda` builds a fresh backend with the same
    // stamp. There is deliberately no device-layout mutation left in this gate:
    // an arm's measurement is always the layout the engine it loaded named.

    // The arms. Before C4 S2b this gate had to ask for `Layers(0)` because the
    // device could not read a packed region at all (#123's device-aware
    // rework); now the device is a first-class arm and the CPU one stays as
    // coverage — on a CPU-only build it is the whole gate, and on a CUDA build
    // it still exercises the CPU kernel. Each arm **asserts the backend it
    // measured**, so a silent fallback (the loader drops to CPU when the
    // offloaded weights do not register) fails loudly instead of reporting a
    // CPU number as a device one.
    //
    // Per-engine (#99): each arm carries an f32 engine and a q8_0 engine, loaded
    // through the explicit cache-type entry point `load_model_configured` — no
    // environment mutation, no process global. The two engines share neither a
    // model nor a `GraphCache`, exactly like the old two `run()` calls did.
    let load = |ns: &str, cache_type: &str, offload: OffloadRequest| -> Box<dyn ModelDef> {
        crate::models::load_model_configured(&gguf, ns, offload, Some(cache_type))
            .expect("load a C4 gate engine")
    };
    let mut arms: Vec<(&str, Box<dyn ModelDef>, Box<dyn ModelDef>)> = Vec::new();
    arms.push((
        "cpu",
        load("cpu.f32.", "f32", OffloadRequest::Layers(0)),
        load("cpu.q8.", "q8_0", OffloadRequest::Layers(0)),
    ));
    #[cfg(feature = "cuda")]
    if crate::cuda::CudaState::get().is_some() {
        arms.push((
            "cuda",
            load("cuda.f32.", "f32", OffloadRequest::Default),
            load("cuda.q8.", "q8_0", OffloadRequest::Default),
        ));
    }

    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let n_ctx = 256;
    let steps = 8;
    let ids = tok.encode("The capital of France is");
    let n = ids.len();

    for (label, ref_model, q8_model) in &arms {
        let models: [(&str, &Box<dyn ModelDef>, KvFormat); 2] = [
            ("f32", ref_model, KvFormat::F32),
            ("q8_0", q8_model, KvFormat::Q8_0),
        ];
        for (fmt_label, model, format) in models {
            // Self-check: the engine carries the format this run names, so a
            // loader that silently resolved the other one fails here instead of
            // reporting a number from the wrong graph.
            assert_eq!(
                model.kv_format(),
                format,
                "[{label}] the {fmt_label} engine must resolve MINFER_CACHE_TYPE={fmt_label}"
            );
            let device = model.device();
            if *label == "cpu" {
                assert_eq!(
                    device,
                    Device::Cpu,
                    "the CPU arm must run on the CPU backend"
                );
            } else {
                assert_eq!(
                    device,
                    Device::Cuda,
                    "the device arm must actually run on the device — a silent CPU fallback \
                     would leave the packed CUDA kernels untested by the very gate that \
                     exists to exercise them"
                );
            }
        }

        let run = |model: &dyn ModelDef| -> (Vec<Vec<f32>>, Vec<u32>, usize) {
            let mut cache = GraphCache::new();
            cache.alloc().kv_set_capacity(n_ctx);
            let positions: Vec<usize> = (0..n).collect();
            let mut l = model.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut cache);
            // #153: the CUDA backend the forward built addresses its regions in
            // the layout this engine resolved — the per-engine tag's value arm.
            // (The `kv_region_bytes` comparison below is the region arm.)
            #[cfg(feature = "cuda")]
            if model.device() == Device::Cuda {
                let cb = cache
                    .alloc()
                    .cuda()
                    .expect("a CUDA engine must have enabled its CUDA backend");
                assert_eq!(
                    cb.kv_layout(),
                    crate::cuda::layout_of(model.kv_format()),
                    "[{}] the CUDA backend tag must be the engine's resolved format",
                    model.kv_format().name()
                );
            }
            let mut logits = vec![l.clone()];
            let mut next = argmax(&l);
            let mut toks = vec![next];
            for s in 0..steps {
                l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
                logits.push(l.clone());
                next = argmax(&l);
                toks.push(next);
            }
            (logits, toks, cache.alloc().kv_region_bytes())
        };

        let (l_ref, t_ref, b_ref) = run(ref_model.as_ref());
        let (l_q8, t_q8, b_q8) = run(q8_model.as_ref());
        let ratio = b_ref as f64 / b_q8 as f64;
        let worst = l_ref
            .iter()
            .zip(&l_q8)
            .map(|(a, b)| max_delta(a, b))
            .fold(0.0f32, f32::max);
        let step_deltas: Vec<f32> = l_ref
            .iter()
            .zip(&l_q8)
            .map(|(a, b)| max_delta(a, b))
            .collect();
        let spread = l_ref
            .iter()
            .flatten()
            .fold(f32::NEG_INFINITY, |m, x| m.max(*x))
            - l_ref.iter().flatten().fold(f32::INFINITY, |m, x| m.min(*x));
        let at_argmax: Vec<f32> = l_ref
            .iter()
            .zip(&l_q8)
            .map(|(a, b)| {
                let i = argmax(a) as usize;
                (a[i] - b[i]).abs()
            })
            .collect();
        let matched = t_ref.iter().zip(&t_q8).take_while(|(a, b)| a == b).count();
        eprintln!(
            "[c4] {label}: KV regions: f32 {b_ref} B vs q8_0 {b_q8} B ({ratio:.2}x smaller); \
             max |Δlogit| = {worst} of a {spread} spread; per step {step_deltas:?}"
        );
        eprintln!("[c4] {label}: |Δ| at the reference argmax per step: {at_argmax:?}");
        eprintln!(
            "[c4] {label}: greedy continuation: {matched}/{} steps agree (reported, not \
             asserted — see the note on this test)",
            t_ref.len()
        );
        assert!(
            b_q8 * 3 <= b_ref,
            "[{label}] packed regions must be at least 3x smaller: {b_q8} vs {b_ref}"
        );
        // Named class, measured on the 0.5B Q4_0 and on Qwen3-0.6B Q8_0 before being
        // fixed here. Q8_0 rounds every K/V cell, so this is never bitwise. The two
        // numbers say different things on purpose:
        //  * at the reference's argmax the logits move by <= 1.0 (measured 0.60 on the
        //    0.5B, 0.55 on Qwen3) — the decoding-relevant error;
        //  * over the whole 152k-way logit vector the tail moves by <= 3.0 (measured
        //    2.50, <= 8% of the 37.8 spread) — a gross-error detector: a wrong row
        //    width or byte order is off by orders of magnitude, not 8%.
        assert!(
            at_argmax.iter().fold(0.0f32, |m, d| m.max(*d)) <= 1.0,
            "[{label}] packed vs f32 at the argmax: {at_argmax:?}"
        );
        // The tail bound is a gross-error detector (a wrong row width or byte order is
        // off by orders of magnitude, not by 11% of the spread). S2's query
        // quantization is the delta from S1's 2.505 to this run's 3.029, both <= 8% of
        // the spread; `MINFER_NO_FUSED_Q8_KV=1` reproduces the smaller one. The device
        // arm is the same class: a packed cell's bytes are the same bytes, so the
        // rounding is the same rounding.
        assert!(
            worst <= 4.0,
            "[{label}] packed vs f32 logits: max |Δ| = {worst} of a {spread} spread"
        );

        // C4 S2's second acceptance: a **physical shift** under Q8_0 keeps the
        // continuation. `kv_rm` moves the surviving cells verbatim (a cell is a whole
        // number of words) and re-ropes each survivor's K through
        // dequantize → rope → requantize; that requantization is the new term, on top
        // of the one the plain run above already carries. On the device arm this is
        // where the packed region's bytes round-trip through the CUDA pool
        // (`copy_kv_to_cpu` → `map_q8_0_cells` → `write_pool`) and move through
        // `kv_move_rows` one whole word per cell.
        //
        // The shift drops the oldest `drop` rows of a longer prefill and continues,
        // which is the conversation's overflow case (C2). Both formats run the *same*
        // shift, so the comparison isolates the packed re-rope — it is not a comparison
        // against a fresh prefill (which C2 records as its own tolerance class).
        let shift_text = tok.encode(
            "The capital of France is Paris and the capital of Japan is Tokyo and the \
             capital of Italy is Rome",
        );
        let shift_n = shift_text.len();
        let drop = 4usize;
        assert!(
            shift_n > drop + steps,
            "the shifted fixture must keep a context"
        );
        let run_shifted = |model: &dyn ModelDef| -> (Vec<Vec<f32>>, Vec<u32>, usize) {
            let mut cache = GraphCache::new();
            cache.alloc().kv_set_capacity(n_ctx);
            let positions: Vec<usize> = (0..shift_n).collect();
            let mut l = model.forward_graph_cached(&shift_text, &positions, 1, n_ctx, &mut cache);
            let (freq_base, freq_scale) = model.rope_params();
            let rope = crate::graph::kvcache::KvRope {
                freq_base,
                freq_scale,
                n_head_kv: model.n_head_kv(),
                hd: model.n_embd_head(),
                style: model.rope_style(),
            };
            let left = cache
                .alloc()
                .kv_rm(0, drop, &rope)
                .expect("packed-aware physical shift");
            // The survivors now address positions 0..left, so the next token continues
            // at `left` — the shift's whole point.
            let mut next = argmax(&l);
            let mut toks = Vec::new();
            let mut logs = Vec::new();
            for s in 0..steps {
                l = model.forward_graph_cached(&[next], &[left + s], 1, n_ctx, &mut cache);
                logs.push(l.clone());
                next = argmax(&l);
                toks.push(next);
            }
            (logs, toks, left)
        };
        let (logs_shift_f32, t_shift_ref, left_ref) = run_shifted(ref_model.as_ref());
        let (logs_shift_q8, t_shift_q8, left_q8) = run_shifted(q8_model.as_ref());
        assert_eq!(
            left_ref, left_q8,
            "[{label}] the same shift must leave the same rows"
        );
        assert_eq!(left_ref, shift_n - drop);
        // Compare the **first** decode step, i.e. the step whose input is the shifted
        // context itself. Later steps are a different matter: a token that flips makes
        // the two runs generate different sequences, so their logits diverge by nature
        // (the run above reports the greedy agreement for exactly that reason).
        let (l_first_f32, l_first_q8) = (&logs_shift_f32[0], &logs_shift_q8[0]);
        let shift_delta = max_delta(l_first_f32, l_first_q8);
        let spread_shift = l_first_f32.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x))
            - l_first_f32.iter().fold(f32::INFINITY, |m, x| m.min(*x));
        let shift_at_argmax = {
            let i = argmax(l_first_f32) as usize;
            (l_first_f32[i] - l_first_q8[i]).abs()
        };
        let shift_matched = t_shift_ref
            .iter()
            .zip(&t_shift_q8)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "[c4s2] {label}: the first decode step after a {drop}-row shift of {shift_n}: max \
             |Δlogit| = {shift_delta} of a {spread_shift} spread, at the argmax \
             {shift_at_argmax}; greedy continuation agrees on {shift_matched}/{} steps \
             (reported, not asserted)",
            t_shift_ref.len()
        );
        // The same *kind* of measurement as the unshifted comparison, one step after a
        // shift — with a slightly wider bound, because the shifted state is fragile in a
        // way the unshifted one is not: `kv_rm` composes C2's re-rope with C4's
        // re-quantization, so the surviving K rows are
        // `quantize(rope(dequantize(quantize(row))))`. Measured here: max |Δlogit| 2.466
        // and 0.064 at the reference's argmax on the fused path, 3.165 / 1.074 on S1's
        // (`MINFER_NO_FUSED_Q8_KV=1`). Both legs must stay green — the knob is an A/B,
        // not an alternate expectation — and a misread cell is off by the whole spread.
        assert!(
            shift_at_argmax <= 2.0,
            "[{label}] packed shift vs f32 shift at the argmax: |Δ| = {shift_at_argmax}"
        );
        assert!(
            shift_delta <= 8.0,
            "[{label}] packed shift vs f32 shift: max |Δ| = {shift_delta} of a \
             {spread_shift} spread"
        );
    }
}
/// #153's acceptance gate: **two CUDA engines with different KV layouts in one
/// process**, driven interleaved through one `CudaState`.
///
/// Before #153 the CUDA kernels read a `static KV_LAYOUT` the loader had set, so
/// two engines in one process could only ever run the last-loaded layout, and
/// the device discipline was serial *for that reason*. Now each engine's resolved
/// `KvFormat` reaches its own `CudaBackend` (`GraphAllocator::set_kv_format` →
/// `cuda::layout_of`), and the captured-graph identity records the tag.
///
/// The arms (gate contract):
/// - **the tag arm** — each cache's `CudaBackend::kv_layout()` is the layout its
///   engine named, and the two are different. This is the value arm that a shared
///   tag fails outright.
/// - **the region arm** — the packed engine's regions are at least 3x smaller, an
///   absolute value computed from the arena, not from the other path.
/// - **the attention arm** — the packed engine's logits stay inside the C4 class
///   of the f32 engine's (at the argmax <= 1.0, tail <= 4.0). A packed region
///   addressed as f32 rows is off by the whole logit spread, so this is the arm
///   that catches a wrong kernel.
/// - **the isolation arm** — each engine's interleaved logits are **bitwise
///   equal** to the same engine run alone in a fresh cache, so neither engine's
///   presence moved the other's attention.
///
/// Interleaved rather than threaded on purpose: `CudaState` is a process-wide
/// singleton and the capture path takes a process-wide stream lock, so two OS
/// threads would serialize on that lock anyway. "Interleaved" is the form the
/// ticket allows and the one that exercises the per-engine tag.
///
/// Ignored because it needs the cached 0.5B and a CUDA device; run it alone:
///
/// ```text
/// cargo test --release --features cuda \
///   two_cuda_engines_with_different_kv_layouts_run_interleaved -- --ignored --test-threads=1
/// ```
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires the cached 0.5B model and a CUDA device"]
fn two_cuda_engines_with_different_kv_layouts_run_interleaved() {
    use crate::graph::cache::GraphCache;
    use crate::graph::kvformat::KvFormat;
    use crate::graph::offload::OffloadRequest;
    use crate::models::{Device, ModelDef};

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the #153 two-engine gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let n_ctx = 256usize;
    let steps = 8usize;
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let positions: Vec<usize> = (0..n).collect();

    // Two engines, loaded **before either runs** and under distinct registry
    // namespaces (the CUDA weight registry is process-global and name-keyed).
    let f32_engine = crate::models::load_model_configured(
        &gguf,
        "pair.f32.",
        OffloadRequest::Default,
        Some("f32"),
    )
    .expect("load the f32 engine");
    let q8_engine = crate::models::load_model_configured(
        &gguf,
        "pair.q8.",
        OffloadRequest::Default,
        Some("q8_0"),
    )
    .expect("load the q8_0 engine");
    for (model, want, name) in [
        (&f32_engine, KvFormat::F32, "f32"),
        (&q8_engine, KvFormat::Q8_0, "q8_0"),
    ] {
        assert_eq!(
            model.kv_format(),
            want,
            "the {name} engine's resolved format"
        );
        assert_eq!(
            model.device(),
            Device::Cuda,
            "the {name} engine must run on the device — a silent CPU fallback would leave \
             the per-engine tag untested by the gate that exists to test it"
        );
    }

    // Interleaved prefill: f32 first, then q8_0, on two live caches.
    let mut caches: [GraphCache; 2] = [GraphCache::new(), GraphCache::new()];
    for c in caches.iter_mut() {
        c.alloc().kv_set_capacity(n_ctx);
    }
    let mut inter_f32: Vec<Vec<f32>> = Vec::new();
    let mut inter_q8: Vec<Vec<f32>> = Vec::new();
    let mut l = f32_engine.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut caches[0]);
    inter_f32.push(l.clone());
    let mut next_f32 = argmax(&l);
    let mut l = q8_engine.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut caches[1]);
    inter_q8.push(l.clone());
    let mut next_q8 = argmax(&l);

    // Tag arm: the two live backends hold the layouts their engines named.
    let tag_f32 = caches[0]
        .alloc()
        .cuda()
        .expect("the f32 engine must have a CUDA backend")
        .kv_layout();
    let tag_q8 = caches[1]
        .alloc()
        .cuda()
        .expect("the q8_0 engine must have a CUDA backend")
        .kv_layout();
    assert_eq!(tag_f32, crate::cuda::KV_LAYOUT_F32, "f32 backend tag");
    assert_eq!(tag_q8, crate::cuda::KV_LAYOUT_Q8_0, "q8_0 backend tag");
    assert_ne!(
        tag_f32, tag_q8,
        "the two engines must hold different tags in one process"
    );

    // Interleaved decode: every step of one engine sits between two steps of the
    // other, so both are live across the whole run.
    for s in 0..steps {
        l = f32_engine.forward_graph_cached(&[next_f32], &[n + s], 1, n_ctx, &mut caches[0]);
        inter_f32.push(l.clone());
        next_f32 = argmax(&l);
        l = q8_engine.forward_graph_cached(&[next_q8], &[n + s], 1, n_ctx, &mut caches[1]);
        inter_q8.push(l.clone());
        next_q8 = argmax(&l);
    }

    // Region arm: the packed regions are an absolute, independently computed size.
    let b_f32 = caches[0].alloc().kv_region_bytes();
    let b_q8 = caches[1].alloc().kv_region_bytes();
    assert!(
        b_q8 * 3 <= b_f32,
        "the packed engine's regions must be at least 3x smaller: {b_q8} vs {b_f32}"
    );

    // Attention arm: the packed engine's answer, in the C4 tolerance class.
    let worst = inter_f32
        .iter()
        .zip(&inter_q8)
        .map(|(a, b)| max_delta(a, b))
        .fold(0.0f32, f32::max);
    let at_argmax = inter_f32
        .iter()
        .zip(&inter_q8)
        .map(|(a, b)| {
            let i = argmax(a) as usize;
            (a[i] - b[i]).abs()
        })
        .fold(0.0f32, f32::max);
    let spread = inter_f32
        .iter()
        .flatten()
        .fold(f32::NEG_INFINITY, |m, x| m.max(*x))
        - inter_f32
            .iter()
            .flatten()
            .fold(f32::INFINITY, |m, x| m.min(*x));
    assert!(
        at_argmax <= 1.0,
        "interleaved packed vs f32 at the argmax: |Δ| = {at_argmax}"
    );
    assert!(
        worst <= 4.0,
        "interleaved packed vs f32: max |Δ| = {worst} of a {spread} spread"
    );

    // Isolation arm: each engine alone in a fresh cache must be bitwise identical
    // to its interleaved run — the other engine's live backend changed nothing.
    let solo = |model: &dyn ModelDef| -> Vec<Vec<f32>> {
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        let mut l = model.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut cache);
        let mut out = vec![l.clone()];
        let mut next = argmax(&l);
        for s in 0..steps {
            l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
            out.push(l.clone());
            next = argmax(&l);
        }
        out
    };
    let solo_f32 = solo(f32_engine.as_ref());
    let solo_q8 = solo(q8_engine.as_ref());
    let drift_f32 = inter_f32
        .iter()
        .zip(&solo_f32)
        .map(|(a, b)| max_delta(a, b))
        .fold(0.0f32, f32::max);
    let drift_q8 = inter_q8
        .iter()
        .zip(&solo_q8)
        .map(|(a, b)| max_delta(a, b))
        .fold(0.0f32, f32::max);
    assert_eq!(
        drift_f32, 0.0,
        "the f32 engine's interleaved logits must be bitwise its solo logits"
    );
    assert_eq!(
        drift_q8, 0.0,
        "the q8_0 engine's interleaved logits must be bitwise its solo logits"
    );

    eprintln!(
        "[153] two live CUDA engines, interleaved {steps} decode steps: tags f32={tag_f32} \
         q8_0={tag_q8}; regions f32 {b_f32} B vs q8_0 {b_q8} B ({:.2}x smaller); \
         interleaved packed-vs-f32 max |Δlogit| = {worst} of a {spread} spread, at the \
         argmax {at_argmax}; interleaved-vs-solo drift {drift_f32} / {drift_q8}",
        b_f32 as f64 / b_q8 as f64
    );
}
/// Issue #188 acceptance, the **concurrency** half: two engines' forwards on
/// two OS threads at the same time, each bitwise equal to its own serial
/// reference.
///
/// This is the configuration [#185](https://github.com/yusiwen/minfer/issues/185)
/// used to refuse (`device_entry::enter`, until
/// [#240](https://github.com/yusiwen/minfer/issues/240)/[#241](https://github.com/yusiwen/minfer/issues/241)
/// deleted that guard together with the dead legacy `CudaState::layer_gpu` path it
/// protected) and the one the pre-#188 `CudaState` singleton cannot express: one
/// stream means a stream's work is serial, and two capture windows cannot be open
/// on it.
/// The fix gives every `CudaBackend` its own non-blocking stream and a
/// thread-local capture mode, so the two engines really do overlap — the
/// `streams` arm asserts the two live backends hold **different** device
/// stream pointers, and the threads rendezvous on a barrier after both caches
/// exist so the overlap is forced rather than hoped for.
///
/// The comparison is exact (`max |Δ| == 0`) against a serial run of the same
/// engine in a fresh cache, per gate contract rule 1: the concurrent run is
/// the arm under test and the serial run is an independent reference. It
/// asserts **values**, not timings, so the S4 map-window co-tenant timing gate
/// (issue #189) plays no part in the verdict.
///
/// Ignored because it needs the cached 0.5B and a CUDA device; run it alone:
///
/// ```text
/// cargo test --release --features cuda \
///   two_cuda_engines_forward_concurrently_and_stay_bitwise_identical -- --ignored --test-threads=1
/// ```
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires the cached 0.5B model and a CUDA device"]
fn two_cuda_engines_forward_concurrently_and_stay_bitwise_identical() {
    use crate::graph::cache::GraphCache;
    use crate::graph::offload::OffloadRequest;
    use crate::models::{Device, ModelDef};
    use std::sync::Barrier;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the #188 concurrency gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let n_ctx = 256usize;
    let steps = 8usize;
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let positions: Vec<usize> = (0..n).collect();

    // Two engines, loaded before either runs, under distinct registry
    // namespaces (the CUDA weight registry is process-global and name-keyed).
    // f32 and q8_0 so the two backends also hold different KV layout tags
    // (#153), which keeps that reasoning inside this gate too.
    let f32_engine = crate::models::load_model_configured(
        &gguf,
        "conc.f32.",
        OffloadRequest::Default,
        Some("f32"),
    )
    .expect("load the f32 engine");
    let q8_engine = crate::models::load_model_configured(
        &gguf,
        "conc.q8.",
        OffloadRequest::Default,
        Some("q8_0"),
    )
    .expect("load the q8_0 engine");
    for (model, name) in [(&f32_engine, "f32"), (&q8_engine, "q8_0")] {
        assert_eq!(
            model.device(),
            Device::Cuda,
            "the {name} engine must run on the device — a silent CPU fallback would leave the \
             concurrency gate measuring the CPU"
        );
    }

    // One forward + `steps` decodes in a fresh cache; returns the logits.
    let run = |model: &dyn ModelDef| -> Vec<Vec<f32>> {
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        let mut l = model.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut cache);
        let mut out = vec![l.clone()];
        let mut next = argmax(&l);
        for s in 0..steps {
            l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
            out.push(l.clone());
            next = argmax(&l);
        }
        out
    };

    // Serial reference, each engine alone.
    let ref_f32 = run(f32_engine.as_ref());
    let ref_q8 = run(q8_engine.as_ref());

    // Concurrent: two threads, both caches alive across a barrier, so the
    // forwards genuinely overlap rather than interleave by luck.
    let (conc_f32, conc_q8, s_f32, s_q8) = {
        let barrier = Barrier::new(2);
        let streams = std::sync::Mutex::new(Vec::<usize>::new());
        let f32_out = std::sync::Mutex::new(Vec::<Vec<f32>>::new());
        let q8_out = std::sync::Mutex::new(Vec::<Vec<f32>>::new());
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut cache = GraphCache::new();
                cache.alloc().kv_set_capacity(n_ctx);
                let mut l = f32_engine.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut cache);
                let mut out = vec![l.clone()];
                let mut next = argmax(&l);
                streams
                    .lock()
                    .unwrap()
                    .push(cache.alloc().cuda().unwrap().device_stream() as usize);
                barrier.wait(); // both backends exist and are live
                for s in 0..steps {
                    l = f32_engine.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
                    out.push(l.clone());
                    next = argmax(&l);
                }
                *f32_out.lock().unwrap() = out;
            });
            scope.spawn(|| {
                let mut cache = GraphCache::new();
                cache.alloc().kv_set_capacity(n_ctx);
                let mut l = q8_engine.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut cache);
                let mut out = vec![l.clone()];
                let mut next = argmax(&l);
                streams
                    .lock()
                    .unwrap()
                    .push(cache.alloc().cuda().unwrap().device_stream() as usize);
                barrier.wait(); // both backends exist and are live
                for s in 0..steps {
                    l = q8_engine.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
                    out.push(l.clone());
                    next = argmax(&l);
                }
                *q8_out.lock().unwrap() = out;
            });
        });
        let s = streams.into_inner().unwrap();
        assert_eq!(s.len(), 2, "one stream recorded per thread");
        assert_ne!(
            s[0], s[1],
            "the two engines must hold two different device streams; a shared stream would \
             serialize them and make two capture windows impossible"
        );
        (
            f32_out.into_inner().unwrap(),
            q8_out.into_inner().unwrap(),
            s[0],
            s[1],
        )
    };

    // Rule 1: value arm against the independent serial reference.
    let drift_f32 = conc_f32
        .iter()
        .zip(&ref_f32)
        .map(|(a, b)| max_delta(a, b))
        .fold(0.0f32, f32::max);
    let drift_q8 = conc_q8
        .iter()
        .zip(&ref_q8)
        .map(|(a, b)| max_delta(a, b))
        .fold(0.0f32, f32::max);
    assert_eq!(
        drift_f32, 0.0,
        "the f32 engine's concurrent logits must be bitwise its serial logits"
    );
    assert_eq!(
        drift_q8, 0.0,
        "the q8_0 engine's concurrent logits must be bitwise its serial logits"
    );

    // The captured-graph identity (#153) must still hold per instance: the
    // captured execs cannot have crossed engines, so the two resolved formats
    // must still differ.
    assert_ne!(
        f32_engine.kv_format(),
        q8_engine.kv_format(),
        "the concurrent engines must still resolve different KV formats"
    );

    eprintln!(
        "[188] two CUDA engines forwarding on two threads ({steps} decode steps each): \
         streams {s_f32:#x} vs {s_q8:#x}; concurrent-vs-serial drift {drift_f32} / {drift_q8}"
    );
}
/// C5's acceptance on the real model: a session resumed from disk continues
/// **bitwise** like the one that stayed in memory. The restored rows *are* the
/// bytes the in-memory run wrote, so this is an equality claim, not a tolerance
/// one — and it fails on anything that loses a row, a run, or a written extent.
///
/// Ignored because it writes a session file; run it alone:
///
/// ```text
/// cargo test --release a_session_resumed_from_disk_continues_bitwise -- --ignored --test-threads=1
/// ```
#[test]
#[ignore = "requires the cached 0.5B model and writes a session file"]
fn a_session_resumed_from_disk_continues_bitwise() {
    use crate::graph::cache::GraphCache;
    use crate::graph::kvsession::KvSessionExpect;
    use crate::graph::Backend;
    use crate::models::{Device, ModelDef};

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the C5 session gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let backend = match model.device() {
        Device::Cpu => Backend::CPU,
        Device::Metal => Backend::METAL,
        Device::Cuda => Backend::CUDA,
    };
    let n_ctx = 256;
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let positions: Vec<usize> = (0..n).collect();

    // A: the session that never leaves memory.
    let mut a = GraphCache::new();
    a.alloc().kv_set_capacity(n_ctx);
    let l_a = model.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut a);

    // B: the same prefill, then saved and forgotten.
    let mut b = GraphCache::new();
    b.alloc().kv_set_capacity(n_ctx);
    let l_b = model.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut b);
    assert_eq!(
        max_delta(&l_a, &l_b),
        0.0,
        "the fixture must be deterministic before anything is saved"
    );
    let file = std::env::temp_dir().join(format!(
        "minfer-c5-session-{}-{}.bin",
        std::process::id(),
        n
    ));
    let report = b.alloc().kv_save(&file).expect("kv_save");
    assert_eq!(report.written, n, "the prefill wrote positions 0..{n}");
    // C4 S2b / C5: the container records the KV element type, and this gate is
    // also the packed-session answer — run it with `MINFER_CACHE_TYPE=q8_0` and
    // the file a packed session wrote is resumed as packed, on whatever backend
    // the model landed on. (f16 is the type the container cannot encode:
    // issue #130.) The assertion makes the packed run self-checking instead of
    // passing vacuously on an f32 fallback.
    eprintln!(
        "[c5] live KV format: {} — the container records it (f16 is not encodable yet, #130)",
        model.kv_format().name()
    );
    if std::env::var("MINFER_CACHE_TYPE").as_deref() == Ok("q8_0") {
        assert_eq!(
            model.kv_format(),
            crate::graph::kvformat::KvFormat::Q8_0,
            "MINFER_CACHE_TYPE=q8_0 must resolve to a packed KV session"
        );
    }
    eprintln!(
        "[c5] saved {} layers / {} cells / {} written / {} bytes to {}",
        report.layers,
        report.cells,
        report.written,
        report.bytes,
        file.display()
    );
    drop(b);

    // C: a **fresh** cache, restored from the file alone.
    let mut c = GraphCache::new();
    c.alloc().kv_set_capacity(n_ctx);
    let expect = KvSessionExpect {
        backend,
        n_ctx,
        n_embd: model.n_head_kv() * model.n_embd_head(),
    };
    let loaded = c.alloc().kv_load(&file, &expect).expect("kv_load");
    assert_eq!(loaded, report);

    // Continue both greedily: A and C must agree to the last bit.
    let mut next = argmax(&l_a);
    let mut pos = n;
    let mut worst = 0.0f32;
    for step in 0..8 {
        let la = model.forward_graph_cached(&[next], &[pos], 1, n_ctx, &mut a);
        let lc = model.forward_graph_cached(&[next], &[pos], 1, n_ctx, &mut c);
        let d = max_delta(&la, &lc);
        worst = worst.max(d);
        assert_eq!(
            d, 0.0,
            "step {step}: a session resumed from disk must be bitwise identical \
             (max |Δ| = {d})"
        );
        next = argmax(&la);
        pos += 1;
    }
    eprintln!("[c5] 8 greedy steps after the restore: max |Δlogit| = {worst}");
    std::fs::remove_file(&file).ok();
}
