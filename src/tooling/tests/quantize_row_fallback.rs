//! #204: a 2-D row that is not a multiple of the target's block size is
//! **demoted**, never copied — the rule llama.cpp's `tensor_type_fallback`
//! applies to every quant target, the legacy 32-block ones included.
//!
//! The fixtures are written here rather than taken from the F6 cache, because
//! the property needs a row length no supported architecture has (every Qwen2 /
//! Qwen3 2-D row in `~/.cache/minfer/f6-src/` is a multiple of 32, so no gate
//! could see the divergence), and because the rule is the *planner's*: a
//! miniature GGUF pins it in CI, where the byte-parity arm cannot run.
//!
//! Split out of `src/tooling/tests.rs` (issue #267): the fixtures live in the
//! parent module and are reached through `use super::*;`.

use super::*;

use crate::quantize::QuantTarget;

/// One file that exercises every arm of the rule at once, on a tied model (no
/// `output.weight`):
///
/// * `token_embd.weight` [60,4] — the shared embedding; the `OUTPUT`/tied branch
///   asks for q8_0 (a 32-block cannot divide 60) and q8_0 cannot encode the row
///   either, so it demotes to F16 and is **not** reported as the retarget it did
///   not get;
/// * `blk.0.attn_q.weight` [64,2] — a multiple of 32 but not of 256: the
///   target's own type for a legacy target, and the 32-block fallback for a
///   K-quant;
/// * `blk.0.ffn_down.weight` [60,3] — not a multiple of 32 either: F16 for
///   *every* quant target, legacy and K-quant alike;
/// * `blk.0.ffn_up.weight` [256,2] — reaches the K-quant encoder itself;
/// * two 1-D norms, the only tensors any target preserves.
fn unaligned_specs() -> Vec<gguf_write::TensorSpec> {
    vec![
        gguf_write::TensorSpec::new("token_embd.weight", [60, 4, 1, 1], GgmlType::F32),
        gguf_write::TensorSpec::new("blk.0.attn_q.weight", [64, 2, 1, 1], GgmlType::F32),
        gguf_write::TensorSpec::new("blk.0.ffn_down.weight", [60, 3, 1, 1], GgmlType::F32),
        gguf_write::TensorSpec::new("blk.0.ffn_up.weight", [256, 2, 1, 1], GgmlType::F32),
        gguf_write::TensorSpec::new("blk.0.attn_norm.weight", [64, 1, 1, 1], GgmlType::F32),
        gguf_write::TensorSpec::new("output_norm.weight", [64, 1, 1, 1], GgmlType::F32),
    ]
}

/// Write one miniature source into the test work dir and return its path.
fn write_source(dir: &Path, tag: &str, specs: &[gguf_write::TensorSpec]) -> PathBuf {
    let path = dir.join(format!("src-{tag}.gguf"));
    let _ = std::fs::remove_file(&path);
    let kv = vec![
        crate::gguf::GgufKv::new_string("general.architecture".into(), "qwen2".into()),
        crate::gguf::GgufKv::new_u32("general.file_type".into(), 1),
    ];
    gguf_write::write_single(&path, &kv, specs.to_vec(), 32, |i, w| {
        // A distinct non-zero payload per tensor: an encoder that wrote the
        // wrong tensor's bytes at the right type would still be caught.
        w.write_all(&vec![0x20 + i as u8; specs[i].nbytes()])
    })
    .expect("write the miniature source");
    path
}

/// The type a written file declares for `name`.
fn type_of(model: &crate::gguf::GgufModel, name: &str) -> GgmlType {
    model.parts[0]
        .ctx
        .info
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tensor {name} missing"))
        .type_
}

/// #204's rule, read back from the written **file** (not from the plan), for a
/// legacy target and a K-quant: an unrepresentable 2-D row is demoted on every
/// arm, and the only tensors left in their source type are the 1-D ones.
///
/// Mutation evidence: with `row_len_fallback()` returning `None` for the legacy
/// targets (the pre-#204 code), the f32 source's 60-wide rows come back **F32**
/// and `preserved` grows to four entries — the assertion on the *file's* type
/// is what fails, not the plan's own report.
#[test]
fn a_non_block_aligned_2d_row_is_demoted_for_every_quant_target() {
    let dir = work_dir("quantize-row-fallback");
    let src = write_source(&dir, "unaligned", &unaligned_specs());
    let model = crate::gguf::load_gguf_model(&src).expect("source parses");

    // (target, the type every tensor must come back as, the exact demotion
    // report). The demotion report is asserted verbatim: its per-tensor text
    // names the block size the row missed, and that text used to say 256 for a
    // 32-block target.
    let cases = [
        (
            QuantTarget::Q4_0,
            [
                GgmlType::F16,  // token_embd.weight  — q8_0 asked, row % 32 != 0
                GgmlType::Q4_0, // blk.0.attn_q.weight — 64 % 32 == 0
                GgmlType::F16,  // blk.0.ffn_down.weight — row % 32 != 0
                GgmlType::Q4_0, // blk.0.ffn_up.weight — 256 % 32 == 0
                GgmlType::F32,  // blk.0.attn_norm.weight — 1-D
                GgmlType::F32,  // output_norm.weight — 1-D
            ],
            vec![
                "token_embd.weight -> f16 (asked q8_0, row % 32 != 0)".to_string(),
                "blk.0.ffn_down.weight -> f16 (asked q4_0, row % 32 != 0)".to_string(),
            ],
        ),
        (
            QuantTarget::Q6_K,
            [
                GgmlType::F16,  // token_embd.weight — Q6_K -> q8_0 -> f16
                GgmlType::Q8_0, // blk.0.attn_q.weight — Q6_K -> q8_0 (64 % 32 == 0)
                GgmlType::F16,  // blk.0.ffn_down.weight — no block divides 60
                GgmlType::Q6_K, // blk.0.ffn_up.weight — 256 % 256 == 0
                GgmlType::F32,  // blk.0.attn_norm.weight — 1-D
                GgmlType::F32,  // output_norm.weight — 1-D
            ],
            vec![
                "token_embd.weight -> f16 (asked q6_K, no encoder block size divides the row)"
                    .to_string(),
                "blk.0.attn_q.weight -> q8_0 (asked q6_K, row % 256 != 0)".to_string(),
                "blk.0.ffn_down.weight -> f16 (asked q6_K, no encoder block size divides the row)"
                    .to_string(),
            ],
        ),
    ];

    for (target, want, want_demoted) in cases {
        let out = dir.join(format!("out-{}.gguf", target.name()));
        let _ = std::fs::remove_file(&out);
        let plan = QuantizePlan::plan(&model, target).expect("plan");
        plan.write_single(&model, &out).expect("write");
        let got = crate::gguf::load_gguf_model(&out).expect("output parses");

        for (spec, want) in unaligned_specs().iter().zip(want) {
            assert_eq!(
                type_of(&got, &spec.name),
                want,
                "{} at {}: a 2-D row that is not a multiple of the block must be demoted, \
                 and a demotion is never a copy of the source type",
                spec.name,
                target.name()
            );
        }
        assert_eq!(plan.demoted, want_demoted, "{}", target.name());
        // Only the two 1-D norms (the rule is asserted on the file above).
        assert_eq!(
            plan.preserved,
            vec![
                "blk.0.attn_norm.weight (f32)".to_string(),
                "output_norm.weight (f32)".to_string()
            ],
            "{}: only 1-D tensors may be preserved",
            target.name()
        );
        // The tied embedding was asked for a type its row cannot take and was
        // then demoted, so the run must not claim the `OUTPUT`-branch retarget
        // happened (it did not).
        assert!(
            plan.retargeted.is_empty(),
            "{}: a demoted tied embedding is not an OUTPUT-branch retarget: {:?}",
            target.name(),
            plan.retargeted
        );
    }
}

/// The `OUTPUT`/tied-embedding branch is a row-length question, and it is the
/// ftype-independent half of #203: on a tied model the shared embedding is asked
/// for `q6_K` when the requested type's block divides the row (`q4_0` on the
/// 0.6B's 1024 — llama.cpp's answer, which the pre-#203 rule missed), for `q8_0`
/// when it does not while a 32-block still does (the 0.5B's 896, with no
/// demotion at all), and for `f16` when not even a 32-block fits.
///
/// Before #204 the unaligned case was *preserved* (so the file held f32) while
/// the CLI still printed it under "tied embedding quantized at q8_0" — the
/// report and the bytes disagreed. #203 changed the *asked* type from a blanket
/// `q8_0` to llama.cpp's branch, and the demotion report says which type was
/// asked, so the two can no longer drift apart.
#[test]
fn the_tied_embedding_takes_llama_cpps_output_branch() {
    let dir = work_dir("quantize-row-tied");
    let kv = vec![
        crate::gguf::GgufKv::new_string("general.architecture".into(), "qwen2".into()),
        crate::gguf::GgufKv::new_u32("general.file_type".into(), 1),
    ];
    let q4_0 = crate::quantize::Target::from(QuantTarget::Q4_0);
    let q4_k_m = crate::quantize::Target::parse("q4_K_M").unwrap();

    for (tag, row, target, want_type, want_retargeted, want_embed_demoted) in [
        // The 0.6B's hidden size: 1024 % 32 == 0 and 1024 % 256 == 0, so the
        // branch's q6_K survives.
        ("aligned-256", 1024i64, q4_0, GgmlType::Q6_K, 1usize, false),
        // A row the 32-block type fits but a 256-block does not: the branch
        // answers q6_K, which the row rule demotes to q8_0.
        ("row-not-256", 64, q4_0, GgmlType::Q8_0, 0, true),
        // The 0.5B's hidden size and a mixture: the branch answers q8_0 itself,
        // so the embedding is not demoted — the measured 13th q8_0 of the
        // reference file (the other 12 are demoted attn_v).
        ("the-0.5b", 896, q4_k_m, GgmlType::Q8_0, 1, false),
        // No block divides the row: f16, reported as a demotion.
        ("unaligned", 60, q4_0, GgmlType::F16, 0, true),
    ] {
        let specs = vec![
            gguf_write::TensorSpec::new("token_embd.weight", [row, 4, 1, 1], GgmlType::F32),
            gguf_write::TensorSpec::new("blk.0.attn_q.weight", [64, 2, 1, 1], GgmlType::F32),
        ];
        let src = dir.join(format!("src-tied-{tag}.gguf"));
        let _ = std::fs::remove_file(&src);
        gguf_write::write_single(&src, &kv, specs.clone(), 32, |i, w| {
            w.write_all(&vec![0x30 + i as u8; specs[i].nbytes()])
        })
        .expect("write");
        let model = crate::gguf::load_gguf_model(&src).expect("source parses");
        let out = dir.join(format!("out-tied-{tag}.gguf"));
        let _ = std::fs::remove_file(&out);
        let plan = QuantizePlan::plan(&model, target).expect("plan");
        plan.write_single(&model, &out).expect("write");
        let got = crate::gguf::load_gguf_model(&out).expect("output parses");

        assert_eq!(
            type_of(&got, "token_embd.weight"),
            want_type,
            "{tag} at {}",
            target.name()
        );
        assert_eq!(
            plan.retargeted.len(),
            want_retargeted,
            "{tag}: {:?}",
            plan.retargeted
        );
        assert_eq!(
            plan.demoted
                .iter()
                .any(|d| d.starts_with("token_embd.weight")),
            want_embed_demoted,
            "{tag}: the embedding's demotion must be reported exactly when the row rule did it: {:?}",
            plan.demoted
        );
    }
}

/// A target whose block size is 1 has nothing to demote: `f16` still writes
/// every 2-D row as f16 and keeps 1-D f32, and `f32` converts everything —
/// neither reports a demotion, and neither copies a 2-D source type.
#[test]
fn f16_and_f32_targets_never_demote() {
    let dir = work_dir("quantize-row-casts");
    let src = write_source(&dir, "unaligned-cast", &unaligned_specs());
    let model = crate::gguf::load_gguf_model(&src).expect("source parses");

    for (target, want_2d, want_1d, want_preserved) in [
        (QuantTarget::F16, GgmlType::F16, GgmlType::F32, 2usize),
        (QuantTarget::F32, GgmlType::F32, GgmlType::F32, 0usize),
    ] {
        let out = dir.join(format!("out-cast-{}.gguf", target.name()));
        let _ = std::fs::remove_file(&out);
        let plan = QuantizePlan::plan(&model, target).expect("plan");
        plan.write_single(&model, &out).expect("write");
        let got = crate::gguf::load_gguf_model(&out).expect("output parses");

        for spec in unaligned_specs() {
            let want = if spec.ne[1] <= 1 { want_1d } else { want_2d };
            assert_eq!(
                type_of(&got, &spec.name),
                want,
                "{} at {}",
                spec.name,
                target.name()
            );
        }
        assert!(
            plan.demoted.is_empty(),
            "{}: {:?}",
            target.name(),
            plan.demoted
        );
        assert_eq!(plan.preserved.len(), want_preserved, "{}", target.name());
    }
}
