//! The f16 writer: 1-D tensors stay f32, 2-D tensors become f16.
//!
//! Split out of `src/tooling/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

// === #169: the 1-D rule per quantize target, read back from the output ===

/// A miniature stand-in for the files the converters produce: 2-D matmul
/// weights **f16**, 1-D norms/biases **f32** (`minfer convert --outtype f16`
/// and `llama-quantize … F16` both write that shape).
fn miniature_f16_source_specs() -> Vec<gguf_write::TensorSpec> {
    vec![
        gguf_write::TensorSpec::new("token_embd.weight", [64, 4, 1, 1], GgmlType::F16),
        gguf_write::TensorSpec::new("blk.0.attn_norm.weight", [64, 1, 1, 1], GgmlType::F32),
        gguf_write::TensorSpec::new("blk.0.attn_q.weight", [64, 2, 1, 1], GgmlType::F16),
        gguf_write::TensorSpec::new("blk.0.attn_q.bias", [64, 1, 1, 1], GgmlType::F32),
        gguf_write::TensorSpec::new("blk.0.ffn_down.weight", [64, 3, 1, 1], GgmlType::F16),
        gguf_write::TensorSpec::new("output_norm.weight", [64, 1, 1, 1], GgmlType::F32),
        // no `output.weight`: the tied model shape, but no sub-8-bit target
        // here, so the tied-embedding retarget must not fire.
    ]
}
/// `name -> (type, payload)` for one file, resolved through the parser's
/// own index and the mapped data section.
fn tensor_of<'a>(
    model: &'a crate::gguf::GgufModel,
    name: &str,
) -> (&'a crate::gguf::GgufTensorInfo, &'a [u8]) {
    let part = &model.parts[0];
    let ti = part
        .ctx
        .info
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tensor {name} missing"));
    let base = part.ctx.offset + ti.offset as usize;
    (ti, &part.data[base..base + ti.nbytes()])
}
/// #169: every target's 1-D rule, asserted on the **types read back from
/// the written file** (not on the plan), with the payload of a preserved
/// 1-D tensor compared byte-for-byte against the source.
///
/// The arms differ in the property under test, so neither can carry the
/// other: the **f16** arm must come back 1-D f32 / 2-D f16 (the bug wrote
/// 1-D f16), and the **f32** control must come back all f32 (it fails if
/// the target is ignored and the source's f16 2-D weights leak through).
/// The source payloads are non-zero and distinct, so "preserved verbatim"
/// is a check with content, and the expected type per tensor is computed
/// from the source spec's rank — never from `plan.preserved`.
#[test]
fn quantize_f16_keeps_1d_f32_and_encodes_2d_f16() {
    let dir = work_dir("quantize-1d");
    let src = dir.join("src.gguf");
    let _ = std::fs::remove_file(&src);
    let specs = miniature_f16_source_specs();
    let kv = vec![
        crate::gguf::GgufKv::new_string("general.architecture".into(), "qwen2".into()),
        crate::gguf::GgufKv::new_u32("general.file_type".into(), 1),
    ];
    gguf_write::write_single(&src, &kv, specs.clone(), 32, |i, w| {
        w.write_all(&vec![0x11 + i as u8; specs[i].nbytes()])
    })
    .expect("write the miniature source");
    let model = crate::gguf::load_gguf_model(&src).expect("source parses");

    for (target, tag) in [
        (QuantTarget::F16, "f16"),
        (QuantTarget::F32, "f32"),
        (QuantTarget::Q8_0, "q8_0"),
    ] {
        let out = dir.join(format!("out-{tag}.gguf"));
        let _ = std::fs::remove_file(&out);
        let plan = QuantizePlan::plan(&model, target).expect("plan");
        plan.write_single(&model, &out).expect("write");
        let got = crate::gguf::load_gguf_model(&out).expect("output parses");

        // The values first: the type of every tensor **as the written file
        // declares it**, against a want computed from the source spec's rank
        // (never from `plan`). A preserved 1-D tensor must also carry the
        // source's own bytes.
        for s in &specs {
            let (ti, payload) = tensor_of(&got, &s.name);
            let want = if target == QuantTarget::F32 {
                GgmlType::F32
            } else if s.ne[1] <= 1 {
                // The engine's norm/bias path reads f32 only, for every
                // non-f32 target (#169).
                GgmlType::F32
            } else {
                target.ggml_type()
            };
            assert_eq!(
                ti.type_,
                want,
                "{target:?}: tensor {} came back {}, expected {} (rank {})",
                s.name,
                ti.type_.type_name(),
                want.type_name(),
                if s.ne[1] <= 1 { "1-D" } else { "2-D" }
            );
            if target != QuantTarget::F32 && s.ne[1] <= 1 {
                let (_, src_payload) = tensor_of(&model, &s.name);
                assert_eq!(
                    payload, src_payload,
                    "{target:?}: preserved 1-D tensor {} must be the source bytes",
                    s.name
                );
            }
        }

        // Then the report: the 1-D tensors that are *not* re-encoded must
        // be named; f32 re-encodes every tensor, so its list is empty.
        let one_d: Vec<&str> = specs
            .iter()
            .filter(|s| s.ne[1] <= 1)
            .map(|s| s.name.as_str())
            .collect();
        if target == QuantTarget::F32 {
            assert!(
                plan.preserved.is_empty(),
                "f32 converts every tensor, so nothing may be preserved: {:?}",
                plan.preserved
            );
        } else {
            assert_eq!(
                plan.preserved.len(),
                one_d.len(),
                "{target:?}: the preserved list must be exactly the 1-D tensors: {:?}",
                plan.preserved
            );
            for n in &one_d {
                assert!(
                    plan.preserved.iter().any(|p| p.starts_with(n)),
                    "{target:?}: 1-D tensor {n} is missing from {:?}",
                    plan.preserved
                );
            }
        }
    }
}
