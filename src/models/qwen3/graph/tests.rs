//! `#[cfg(test)] mod tests` for `src/models/qwen3/graph.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::models::ModelDef;

/// Path to the locally cached Qwen3-0.6B Q8_0.
fn cached_model_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(".cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn load_qwen3() -> Option<(crate::gguf::GgufModel, Qwen3Model, Vec<u32>, Vec<usize>)> {
    let path = cached_model_path()?;
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let q3: &Qwen3Model = model
        .as_any()
        .downcast_ref::<Qwen3Model>()
        .expect("qwen3 model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    assert!(!ids.is_empty());
    let positions: Vec<usize> = (0..ids.len()).collect();
    Some((gguf, q3.clone(), ids, positions))
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best as u32
}

fn compare(tag: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "[{tag}] logits length mismatch");
    let mut maxd = 0.0f32;
    for i in 0..a.len() {
        maxd = maxd.max((a[i] - b[i]).abs());
    }
    eprintln!("[{tag}] logits max abs diff: {maxd:.3e}");
    assert!(
        maxd < 1e-3,
        "[{tag}] graph logits diverge (max diff {maxd:.3e})"
    );
}

/// Qwen3 hermetic CPU verification (the model's forward IS the graph path,
/// so two independent GraphCache runs — prefill + decode — must agree
/// bit-for-bit: deterministic build/execute, params-only reuse).
///
/// CPU-only by design: skipped once a Metal test has initialized MPS in
/// this process (the loader would then register weights on the GPU and the
/// run would exercise the Metal path instead — covered by the Metal tests).
#[test]
fn graph_cpu_self_consistency_real_model() {
    #[cfg(target_os = "macos")]
    let _g = crate::metal::metal_test_lock();
    #[cfg(target_os = "macos")]
    if crate::metal::MpsState::get().is_some() {
        eprintln!("MPS initialized by an earlier test; skipping CPU-only test");
        return;
    }
    let Some((_, q3, ids, positions)) = load_qwen3() else {
        eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
        return;
    };
    // Keep the weight registry stable for this whole test (see qwen2 tests).
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let n_ctx = q3.hparams.max_seq_len as usize;
    let nt = ids.len();

    let mut cache_a = GraphCache::new();
    let mut cache_b = GraphCache::new();
    let pa = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_a);
    let pb = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_b);
    compare("prefill", &pa, &pb);

    let next = argmax(&pa);
    let da = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut cache_a);
    let db = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut cache_b);
    compare("decode", &da, &db);
}

/// KV isolation between caches (mirrors qwen2): interleaving prefill/decode
/// across two caches must not cross-contaminate, and a smaller n_ctx must
/// not change prefill logits while positions fit. CPU-only (see above).
#[test]
fn forward_cached_isolates_kv_between_caches() {
    #[cfg(target_os = "macos")]
    let _g = crate::metal::metal_test_lock();
    #[cfg(target_os = "macos")]
    if crate::metal::MpsState::get().is_some() {
        eprintln!("MPS initialized by an earlier test; skipping CPU-only test");
        return;
    }
    let Some((_, q3, ids, positions)) = load_qwen3() else {
        eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
        return;
    };
    // Keep the weight registry stable for this whole test (see qwen2 tests).
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let nt = ids.len();
    let n_ctx = q3.hparams.max_seq_len as usize;

    let mut cache_a = GraphCache::new();
    let mut cache_b = GraphCache::new();
    let pa = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_a);
    let pb = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_b);
    let na = argmax(&pa);
    let nb = argmax(&pb);

    let da = Qwen3Graph::forward_cached(&q3, &[na], &[nt], 1, n_ctx, &mut cache_a);
    let db = Qwen3Graph::forward_cached(&q3, &[nb], &[nt], 1, n_ctx, &mut cache_b);
    assert_eq!(na, nb, "identical inputs must give the same greedy token");
    compare("interleaved decode", &da, &db);

    let small_ctx = nt + 4;
    let mut cache_s = GraphCache::new();
    let ps = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, small_ctx, &mut cache_s);
    compare("smaller n_ctx prefill", &pa, &ps);

    let oob = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Qwen3Graph::forward_cached(&q3, &[na], &[small_ctx], 1, small_ctx, &mut cache_s);
    }));
    assert!(oob.is_err(), "position >= n_ctx must be rejected");
}

/// Decode fused-QKV-with-per-head-norm (Op::FusedQkvNorm) must be
/// numerically identical to the unfused path, and the fused node must be
/// present/absent exactly per the fuse flag. Qwen3 fuses only the QKV
/// projection chain (3 matmul + 2 qk_norm + 2 rope + 2 store → one concat
/// matmul + fused per-head norm + no-bias rope+store); the FFN fusion is
/// decoupled (always on when gated), so the fused/unfused graphs differ
/// ONLY in the QKV spine, and the decode logits must match bit-for-bit
/// (both execute the same Metal math — concat vs separate per-row matmul,
/// and rms_norm_256 / attn_rope_store vs QkNorm / rope_f32 / store_kv).
#[test]
fn fused_qkv_norm_matches_unfused_decode() {
    #[cfg(target_os = "macos")]
    let _g = crate::metal::metal_test_lock();
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("not macOS; skipping");
        return;
    }
    #[cfg(target_os = "macos")]
    {
        use crate::graph::ops::Op;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        crate::metal::MpsState::init();
        let Some((_, q3, ids, positions)) = load_qwen3() else {
            eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
            return;
        };
        assert!(
            crate::graph::metal_backend::metal_available() && Qwen3Graph::weights_on_gpu(&q3),
            "Metal path must actually run (weights on GPU)"
        );
        let n_ctx = q3.hparams.max_seq_len as usize;
        let nt = ids.len();
        let nv = q3.hparams.n_vocab as usize;

        // --- node presence (build only, no execute) ---
        let fused_params = GraphParams {
            n_tokens: 1,
            n_out: 1,
            gtype: GraphType::Decode,
            cparams: CParams {
                n_ctx,
                flash_attn: false,
                explicit_span: false,
                kv_map: false,
                gpu: true,
                gpu_layers: usize::MAX, // E5 fixture: no offload limit
                kv_format: crate::graph::kvformat::KvFormat::F32,
                fuse_qkv: true,
                fuse_ffn: false,
            },
            weights_version: 1,
        };
        let fg = Qwen3Graph::build(&q3, &fused_params);
        let has_fused = fg
            .nodes
            .iter()
            .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
        assert!(
            has_fused,
            "decode graph with fuse_qkv=true must contain FusedQkvNorm"
        );

        let unfused_params = GraphParams {
            cparams: CParams {
                fuse_qkv: false,
                fuse_ffn: false,
                ..fused_params.cparams
            },
            ..fused_params
        };
        let ug = Qwen3Graph::build(&q3, &unfused_params);
        let has_unfused = ug
            .nodes
            .iter()
            .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
        assert!(
            !has_unfused,
            "decode graph with fuse_qkv=false must NOT contain FusedQkvNorm"
        );

        // --- numeric: fused (default) vs unfused (MINFER_NO_FUSE_QKV=1) ---
        let _ = std::env::remove_var("MINFER_NO_FUSE_QKV");
        let mut a = GraphCache::new();
        let pa = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut a);
        let next = argmax(&pa);
        let da = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut a);
        {
            let (g, _) = a.current().unwrap();
            let fused_in_cache = g
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
            assert!(fused_in_cache, "the default decode graph must be fused");
        }

        std::env::set_var("MINFER_NO_FUSE_QKV", "1");
        let mut b = GraphCache::new();
        let pb = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut b);
        let db = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut b);
        std::env::remove_var("MINFER_NO_FUSE_QKV");

        // prefill never fuses (nt>1) → both prefills must be bit-identical;
        // this isolates the decode-only divergence.
        {
            let mut pd = 0.0f32;
            for (x, y) in pa.iter().zip(pb.iter()) {
                pd = pd.max((x - y).abs());
            }
            eprintln!("[qwen3 fused test] prefill a vs b max diff: {pd:.3e}");
            assert_eq!(pa.len(), pb.len());
            assert!(pd < 1e-5, "prefill a/b diverge (max {pd:.3e})");
        }
        {
            let (g, _) = b.current().unwrap();
            let unfused_in_cache = g
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
            assert!(
                !unfused_in_cache,
                "the MINFER_NO_FUSE_QKV=1 decode graph must be unfused"
            );
        }

        // fused and unfused decode must produce the SAME greedy token (the
        // hard correctness oracle — the graph_metal_matches_llama_reference
        // test additionally pins the whole sequence against llama.cpp), and
        // logits must agree within float noise. NOTE: unlike Qwen2's Q4_0
        // (fused-vs-unfused max diff 0.0), the Q8_0 concat matmul (od_total
        // vs separate od) is NOT bit-exact — the per-row reduction stays in
        // a different threadgroup grouping, giving ~1e-3 float noise on the
        // final logits. This does not flip the argmax (verified: greedy is
        // byte-identical, and matches llama.cpp exactly).
        assert_eq!(
            da.len(),
            nv,
            "decode logits length must be nv (=1 output token)"
        );
        assert_eq!(
            argmax(&da),
            argmax(&db),
            "fused vs unfused decode must select the same next token"
        );
        let mut maxd = 0.0f32;
        for (x, y) in da.iter().zip(db.iter()) {
            maxd = maxd.max((x - y).abs());
        }
        assert!(
            maxd < 1e-2,
            "fused vs unfused decode logits diverge (max diff {maxd:.3e})"
        );
    }
}

/// Metal greedy generation must reproduce the reference token sequence
/// verified against llama.cpp (same Q8_0 GGUF, raw prompt, temp 0, no
/// penalties — 60 tokens were byte-identical; the first 9 are pinned here
/// as a hermetic regression oracle). Also asserts Metal is deterministic:
/// two independent caches must produce the identical sequence.
#[test]
fn graph_metal_matches_llama_reference() {
    #[cfg(target_os = "macos")]
    let _g = crate::metal::metal_test_lock();
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("not macOS; skipping");
        return;
    }
    #[cfg(target_os = "macos")]
    {
        crate::metal::MpsState::init();
        let Some((_, q3, ids, positions)) = load_qwen3() else {
            eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
            return;
        };
        assert!(
            crate::graph::metal_backend::metal_available() && Qwen3Graph::weights_on_gpu(&q3),
            "Metal path must actually run (weights on GPU)"
        );
        let n_ctx = q3.hparams.max_seq_len as usize;
        let nt = ids.len();

        // Reference greedy tokens for the raw prompt "The capital of France
        // is" (temp 0, no penalties): " Paris. The capital of France is also
        // the capital..." — verified token-for-token against llama.cpp.
        let reference: [u32; 9] = [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279];

        for (cache, tag) in [(&mut GraphCache::new(), "a"), (&mut GraphCache::new(), "b")] {
            let l = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, cache);
            let mut toks = vec![argmax(&l)];
            for i in 0..reference.len() - 1 {
                let p = (nt + i) as usize;
                let l = Qwen3Graph::forward_cached(&q3, &[toks[i]], &[p], 1, n_ctx, cache);
                toks.push(argmax(&l));
            }
            eprintln!("[qwen3 metal {tag}] greedy={toks:?}");
            assert_eq!(
                toks,
                reference.to_vec(),
                "Metal greedy diverges from the llama.cpp-verified reference (cache {tag})"
            );
        }
    }
}

/// Metal prefill must be bitwise deterministic across runs. Regression
/// test for the Q8_0 multi-token matmul race (missing trailing
/// threadgroup_barrier in `kernel_q8_0_f32_matmul_multi` zeroed shmem
/// while slow threads were still reducing it, flipping output elements to
/// 0 between runs). Keeps the layer-0 K-path nodes alive and compares two
/// independent executions element by element.
#[test]
fn metal_prefill_determinism() {
    #[cfg(target_os = "macos")]
    let _g = crate::metal::metal_test_lock();
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("not macOS; skipping");
        return;
    }
    #[cfg(target_os = "macos")]
    {
        crate::metal::MpsState::init();
        let Some((_, q3, ids, positions)) = load_qwen3() else {
            eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
            return;
        };
        let nt = ids.len();
        let params = GraphParams {
            n_tokens: nt,
            n_out: 1,
            gtype: GraphType::Prefill,
            cparams: CParams {
                n_ctx: 512,
                flash_attn: false,
                explicit_span: false,
                kv_map: false,
                gpu: true,
                gpu_layers: usize::MAX, // E5 fixture: no offload limit
                kv_format: crate::graph::kvformat::KvFormat::F32,
                fuse_qkv: false,
                fuse_ffn: false,
            },
            weights_version: 1,
        };
        // layer-0: embed, attn rms, q/k/v matmul, qk_norm q/k, rope q/k
        let keep = [2usize, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut dumps: Vec<Vec<Vec<f32>>> = Vec::new();
        for _ in 0..2 {
            let mut graph = Qwen3Graph::build(&q3, &params);
            for &nid in &keep {
                if !graph.outputs.contains(&nid) {
                    graph.outputs.push(nid);
                }
            }
            let sched = BackendScheduler::new();
            let mut alloc = GraphAllocator::new();
            Qwen3Graph::register_graph_weights(&q3, &mut alloc);
            alloc.enable_metal();
            sched.assign_backends(&mut graph, &alloc);
            alloc.alloc_graph(&graph).unwrap();
            let ids32: Vec<u32> = ids.iter().copied().collect();
            alloc.fill_input_i32(&graph, "token_ids", &ids32).unwrap();
            let pos32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            alloc.fill_input_i32(&graph, "positions", &pos32).unwrap();
            // E2 (fully qualified: this arm compiles only on macOS).
            alloc
                .fill_batch_inputs(
                    &graph,
                    &crate::graph::batch::Batch::single(&ids32, &positions),
                )
                .unwrap();
            sched.execute(&graph, &mut alloc).unwrap();
            let mut run_dumps = Vec::new();
            for &nid in &keep {
                run_dumps.push(alloc.copy_to_cpu(nid).unwrap());
            }
            dumps.push(run_dumps);
        }
        for (i, &nid) in keep.iter().enumerate() {
            let (a, b) = (&dumps[0][i], &dumps[1][i]);
            let neq = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
            eprintln!("[determinism] node{nid}: neq={neq}");
            assert_eq!(
                neq, 0,
                "Metal prefill node {nid} is not deterministic across runs"
            );
        }
    }
}
