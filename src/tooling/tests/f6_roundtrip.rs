//! F6 round trips against the llama.cpp reference: rewrite, HF conversion and split.
//!
//! Split out of `src/tooling/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// G1: rewriting a cached GGUF through the writer is bit-exact.
///
/// Metadata is compared key-for-key (same order, same encoded value bytes),
/// every tensor payload is byte-identical, and logits after a 4-token greedy
/// continuation are `assert_eq!`-identical (a *bitwise* claim).
#[test]
#[ignore = "requires a cached 0.5B GGUF and writes ~0.5 GB under /tmp/f6-work"]
fn f6_rewriting_a_gguf_is_bitwise_and_metadata_equivalent() {
    let src = env_path(
        "MINFER_F6_ROUNDTRIP_MODEL",
        &cached_qwen05().to_string_lossy(),
    );
    let Some(src) = src else { return };
    let dir = work_dir("roundtrip");
    let out = dir.join("roundtrip.gguf");
    let _ = std::fs::remove_file(&out);

    let model = crate::gguf::load_gguf_model(&src).expect("source GGUF");
    let paths = write_verbatim(&model, &dir, "roundtrip", u64::MAX).expect("rewrite");
    assert_eq!(paths, vec![out.clone()]);

    let a = &model.parts[0].ctx;
    let rb = std::fs::read(&out).expect("read rewrite");
    let b = crate::gguf::GgufContext::init_from_data(&rb).expect("rewrite parses");
    assert_eq!(a.kv.len(), b.kv.len(), "KV count");
    for (ka, kb) in a.kv.iter().zip(b.kv.iter()) {
        assert_eq!(ka.key, kb.key, "KV key order");
        assert_eq!(ka.is_array, kb.is_array, "KV {} array flag", ka.key);
        assert_eq!(ka.type_, kb.type_, "KV {} type", ka.key);
        assert_eq!(ka.data, kb.data, "KV {} value bytes", ka.key);
        assert_eq!(ka.data_string, kb.data_string, "KV {} strings", ka.key);
    }
    assert_eq!(a.info.len(), b.info.len(), "tensor count");
    for (ta, tb) in a.info.iter().zip(b.info.iter()) {
        assert_eq!(ta.name, tb.name, "tensor order");
        assert_eq!(ta.ne, tb.ne, "tensor {} shape", ta.name);
        assert_eq!(ta.type_, tb.type_, "tensor {} type", ta.name);
        assert_eq!(ta.nbytes(), tb.nbytes(), "tensor {} bytes", ta.name);
        let sa = &model.parts[0].data
            [a.offset + ta.offset as usize..a.offset + ta.offset as usize + ta.nbytes()];
        let sb = &rb[b.offset + tb.offset as usize..b.offset + tb.offset as usize + tb.nbytes()];
        assert_eq!(sa, sb, "tensor {} payload", ta.name);
    }

    let (la, ga) = logits_greedy(&src, PROMPT, 4, 512);
    let (lb, gb) = logits_greedy(&out, PROMPT, 4, 512);
    assert_eq!(ga, gb, "greedy continuation diverged");
    assert_eq!(la, lb, "round-trip logits must be bitwise identical");
    eprintln!(
        "f6 round-trip: {} tensors bit-identical, {} logits bit-equal, greedy {:?}",
        a.info.len(),
        la.len(),
        ga
    );
}
/// G2: the HF converter's output is byte-identical (per tensor) to
/// llama.cpp's converter on the same checkpoint, the engine's strict
/// tokenizer/template gates accept it, and logits from the two files are
/// bitwise identical under minfer.
#[test]
#[ignore = "requires the HF checkpoint and a llama.cpp-converted reference under /tmp/f6-work"]
fn f6_hf_conversion_matches_the_llamacpp_reference() {
    let Some(hf_dir) = env_path("MINFER_F6_HF_DIR", "/tmp/f6-work/hf-src") else {
        return;
    };
    let Some(ref_gguf) = env_path("MINFER_F6_LLAMACPP_GGUF", "/tmp/f6-work/ref-f16.gguf") else {
        return;
    };
    let out = work_dir("hf").join("minfer-f16.gguf");
    let _ = std::fs::remove_file(&out);

    let conv = crate::convert::Conversion::plan(&hf_dir, OutType::F16).expect("plan");
    conv.write_single(&out).expect("write converted GGUF");

    // Strict loader gates: tokenizer (model=gpt2, pre=qwen2, 256 byte
    // tokens, non-empty merges) and the chat template render.
    let gguf = crate::gguf::load_gguf_model(&out).expect("converted file parses");
    let ctx = &gguf.parts[0].ctx;
    assert_eq!(ctx.get_key_val_str("tokenizer.ggml.model").unwrap(), "gpt2");
    assert_eq!(ctx.get_key_val_str("tokenizer.ggml.pre").unwrap(), "qwen2");
    let tok = crate::tokenizer::Tokenizer::load(ctx).expect("strict tokenizer accepts the file");
    let tmpl = ctx
        .get_key_val_str("tokenizer.chat_template")
        .expect("chat template present");
    let rendered =
        crate::template::render_template(&tmpl, "hello", true, "").expect("chat template renders");
    assert!(rendered.contains("<|im_start|>user"), "{rendered:?}");
    assert!(
        rendered.ends_with("<|im_start|>assistant\n"),
        "{rendered:?}"
    );
    let ids = tok.encode(PROMPT);
    assert!(!ids.is_empty());

    // Weight equality against llama.cpp's converter.
    assert_tensor_payloads_equal(&out, &ref_gguf);

    // Same weights in two files → bitwise-identical logits under minfer.
    let (lm, gm) = logits_greedy(&out, PROMPT, 4, 512);
    let (lr, gr) = logits_greedy(&ref_gguf, PROMPT, 4, 512);
    assert_eq!(
        gm, gr,
        "greedy continuation differs from the llama.cpp file"
    );
    assert_eq!(lm, lr, "logits differ from the llama.cpp file (bitwise)");
    eprintln!(
        "f6 hf: {} tensors equal to llama.cpp, logits bit-equal, greedy {:?}",
        conv.specs.len(),
        gm
    );
}
/// G3: the split file's merged index is exactly the single-file index and
/// the logits are bitwise identical after a re-load.
#[test]
#[ignore = "requires a cached 0.5B GGUF and writes ~0.5 GB under /tmp/f6-work"]
fn f6_split_merged_index_and_logits_match_the_single_file() {
    let Some(src) = env_path("MINFER_F6_SPLIT_MODEL", &cached_qwen05().to_string_lossy()) else {
        return;
    };
    let dir = work_dir("split");
    for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let _ = std::fs::remove_file(e.path());
    }
    let model = crate::gguf::load_gguf_model(&src).expect("source GGUF");
    // The cached q4_k_m 0.5B's largest tensor is `output.weight` at ~138 MiB
    // (Q6_K), so the cap has to clear it; 160 MiB still forces 3 parts on a
    // ~450 MiB data section.
    let parts = write_verbatim(&model, &dir, "m", 160 * 1024 * 1024).expect("split");
    assert!(parts.len() > 1, "the cap must produce a real split");

    let single = &model.parts[0].ctx;
    let merged = crate::gguf::load_gguf_model(&parts[0]).expect("merged split load");
    assert_eq!(merged.parts.len(), parts.len());
    assert_eq!(
        crate::gguf::split_file_info(parts[0].file_name().unwrap().to_str().unwrap())
            .map(|(_, idx, count)| (idx, count)),
        Some((0, parts.len()))
    );
    // Exactly the single-file index, in the same order.
    let mut flat: Vec<(String, [i64; 4], crate::gguf::GgmlType, usize)> = Vec::new();
    for (i, part) in merged.parts.iter().enumerate() {
        assert_eq!(
            part.ctx.get_key_val_i64("split.no").map(|v| v as usize),
            Some(i)
        );
        assert_eq!(
            part.ctx.get_key_val_i64("split.count").map(|v| v as usize),
            Some(parts.len())
        );
        for ti in &part.ctx.info {
            flat.push((ti.name.clone(), ti.ne, ti.type_, ti.nbytes()));
        }
    }
    let expected: Vec<_> = single
        .info
        .iter()
        .map(|ti| (ti.name.clone(), ti.ne, ti.type_, ti.nbytes()))
        .collect();
    assert_eq!(
        flat, expected,
        "merged index must equal the single-file index"
    );

    let (l1, g1) = logits_greedy(&src, PROMPT, 4, 512);
    let (l2, g2) = logits_greedy(&parts[0], PROMPT, 4, 512);
    assert_eq!(g1, g2, "greedy continuation differs after re-load");
    assert_eq!(l1, l2, "split logits must be bitwise identical");
    eprintln!(
        "f6 split: {} parts, {} tensors, logits bit-equal, greedy {:?}",
        parts.len(),
        flat.len(),
        g1
    );
}
