//! The bf16 writer and its measured tolerance against the f16 source.
//!
//! Split out of `src/tooling/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// #142 attribution: walk a bf16 GGUF and an f16 GGUF of the same model and
/// count (values compared, values that differ, differing values whose f16
/// side is a subnormal word). 1-D tensors are f32 on both sides and must be
/// bit-equal; a difference there is counted as a difference with a *non*-
/// subnormal origin, so the caller's `n_diff == n_sub` assertion catches it.
fn bf16_vs_f16_weight_value_diffs(a_path: &Path, b_path: &Path) -> (usize, usize, usize) {
    let ma = crate::gguf::load_gguf_model(a_path).expect("a parses");
    let mb = crate::gguf::load_gguf_model(b_path).expect("b parses");
    let mut index = std::collections::HashMap::new();
    for (pi, part) in mb.parts.iter().enumerate() {
        for ti in &part.ctx.info {
            index.insert(
                ti.name.clone(),
                (
                    pi,
                    part.ctx.offset + ti.offset as usize,
                    ti.nbytes(),
                    ti.type_,
                ),
            );
        }
    }
    let (mut n_vals, mut n_diff, mut n_sub) = (0usize, 0usize, 0usize);
    for (pi, part) in ma.parts.iter().enumerate() {
        for ti in &part.ctx.info {
            let (pb, ob, nb, tb_type) = *index
                .get(&ti.name)
                .unwrap_or_else(|| panic!("tensor '{}' only in the bf16 file", ti.name));
            assert_eq!(nb, ti.nbytes(), "tensor {} size", ti.name);
            let a = &ma.parts[pi].data[part.ctx.offset + ti.offset as usize..][..ti.nbytes()];
            let b = &mb.parts[pb].data[ob..ob + nb];
            match (ti.type_, tb_type) {
                (crate::gguf::GgmlType::BF16, crate::gguf::GgmlType::F16) => {
                    for i in 0..nb / 2 {
                        let ba = u16::from_le_bytes([a[2 * i], a[2 * i + 1]]);
                        let bb = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
                        n_vals += 1;
                        if crate::block::bf16_to_f32(ba).to_bits()
                            != crate::block::fp16_to_f32(bb).to_bits()
                        {
                            n_diff += 1;
                            // f16 exponent field zero == subnormal (or zero).
                            if bb & 0x7c00 == 0 {
                                n_sub += 1;
                            }
                        }
                    }
                }
                (crate::gguf::GgmlType::F32, crate::gguf::GgmlType::F32) => {
                    for i in 0..nb / 4 {
                        n_vals += 1;
                        if a[4 * i..4 * i + 4] != b[4 * i..4 * i + 4] {
                            n_diff += 1;
                        }
                    }
                }
                (t1, t2) => panic!("tensor {} type pair {t1:?}/{t2:?}", ti.name),
            }
        }
    }
    (n_vals, n_diff, n_sub)
}
// === #142: bf16 output + the CPU bf16 weight path ===

/// #142: the bf16 writer is byte-identical, per tensor, to a bf16 reference
/// produced by llama.cpp, and the 1-D rule on both sides is "norms/biases
/// stay f32".
///
/// **Reference provenance.** There is no torch/transformers on dgxspark, so
/// this uses Source B of #142's plan rather than
/// `convert_hf_to_gguf.py --outtype bf16` — but the cast's *input* is the
/// **f32** conversion, not the f16 one:
///
/// ```bash
/// minfer convert <hf-dir> qwen2.5-0.5b-instruct-f32.gguf --outtype f32
/// llama-quantize --pure qwen2.5-0.5b-instruct-f32.gguf \
///   ~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-bf16-from-f32.gguf BF16
/// ```
///
/// `minfer convert --outtype f32` is **exact** for a bf16 source (bf16 is
/// f32's top 16 bits), so the cast is a pure f32→bf16 RNE projection of the
/// checkpoint's own values — exactly what minfer's HF→bf16 writer does. That
/// is what makes the per-tensor byte equality meaningful.
///
/// **Why not the f16 source (#142 §3's Source B as literally written).**
/// bf16 → f16 is *not* exact for every bf16 value: f16's exponent range is
/// far smaller, so a bf16 value below f16's smallest normal (2^-14) lands in
/// f16's subnormal grid and is rounded, and one below 2^-24 flushes to zero.
/// On the Qwen2.5-0.5B checkpoint that is 126 575 bytes across 169 of the
/// 290 tensors (measured 2026-10-01, aarch64: `minfer convert --outtype bf16`
/// vs `llama-quantize --pure <f16>.gguf … BF16`, per-tensor byte diff), and
/// every one of them is an f16 subnormal. Casting that f16 file to bf16
/// therefore *cannot* reproduce the checkpoint's bf16 bits, so it is not a
/// valid oracle for those bytes; the f32 source removes the confound. The
/// f16-vs-bf16 difference is measured and attributed in
/// `f142_bf16_output_runs_within_the_stated_bound`.
#[test]
#[ignore = "requires the cached HF checkpoint and the llama-quantize BF16 reference"]
fn f142_bf16_conversion_is_byte_identical_to_the_reference() {
    let Some(hf_dir) = env_path(
        "MINFER_F142_HF_DIR",
        "~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct",
    ) else {
        return;
    };
    let Some(ref_bf16) = env_path(
        "MINFER_F142_LLAMACPP_BF16",
        "~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-bf16-from-f32.gguf",
    ) else {
        return;
    };
    let out = work_dir("f142").join("f142-bf16-ref.gguf");
    let _ = std::fs::remove_file(&out);
    let conv = crate::convert::Conversion::plan(&hf_dir, OutType::Bf16).expect("plan");
    conv.write_single(&out).expect("write the bf16 GGUF");

    // The file contract, asserted on the plan: 2-D bf16, 1-D f32.
    let (mut n_2d, mut n_1d) = (0usize, 0usize);
    for (i, t) in conv.ckpt.tensors.iter().enumerate() {
        if t.shape.len() <= 1 {
            n_1d += 1;
            assert_eq!(
                conv.targets[i],
                crate::gguf::GgmlType::F32,
                "1-D tensor {} must stay f32",
                t.gguf_name
            );
        } else {
            n_2d += 1;
            assert_eq!(
                conv.targets[i],
                crate::gguf::GgmlType::BF16,
                "2-D tensor {} must be bf16",
                t.gguf_name
            );
        }
    }
    let n = assert_tensor_payloads_equal(&out, &ref_bf16);
    assert_eq!(n, conv.specs.len(), "tensor count");
    eprintln!(
        "f142 bf16 parity: {n}/{} tensors byte-identical to {} (2-D bf16 {n_2d}, \
         1-D f32 {n_1d})",
        conv.specs.len(),
        ref_bf16.display(),
    );
}
/// #142's real-model gate: a bf16-converted 0.5B loads, runs on the CPU
/// bf16 weight path, and its greedy continuation matches the f16 conversion
/// of the same checkpoint.
///
/// **The bound, named before measuring.** The issue's premise was that bf16
/// is "looser" than f16 because it has 8 mantissa bits vs f16's 10, so the
/// file comparison would need a wide tolerance. The measurements refute the
/// *first* half but not the second: because the source is bf16, the bf16
/// file carries the checkpoint's exact values and the f16 file differs from
/// it only where f16 cannot represent a bf16 value — its **subnormal**
/// range. So the logits are *not* bitwise (the task text's expectation),
/// but the difference is tiny and fully attributable. Stated bound:
/// **max |Δlogit| ≤ 1e-4 and max relative ≤ 1e-5, with the greedy
/// continuation identical**; measured values are printed first, and the
/// per-value attribution below is asserted, not hand-waved.
///
/// Attribution: every weight value at which the two files differ must sit on
/// an **f16 subnormal** word (exponent field 0). That is the f16 file's own
/// rounding, not a bf16 decode fault — and `f142_bf16_conversion_is_byte_
/// identical_to_the_reference` proves the bf16 writer against an exact
/// (f32-source) reference, so the two files' only difference is the one
/// measured here.
#[test]
#[ignore = "requires the cached HF checkpoint and the f16 source (writes ~1 GB per file)"]
fn f142_bf16_output_runs_within_the_stated_bound() {
    let Some(hf_dir) = env_path(
        "MINFER_F142_HF_DIR",
        "~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct",
    ) else {
        return;
    };
    let Some(f16_path) = env_path(
        "MINFER_F6_F16_GGUF",
        "~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f16.gguf",
    ) else {
        return;
    };
    let bf16_path = work_dir("f142").join("f142-bf16-run.gguf");
    let _ = std::fs::remove_file(&bf16_path);
    crate::convert::Conversion::plan(&hf_dir, OutType::Bf16)
        .expect("plan")
        .write_single(&bf16_path)
        .expect("write the bf16 GGUF");

    // The weight-level attribution, before the (cheap) logit comparison.
    let (n_vals, n_diff, n_sub) = bf16_vs_f16_weight_value_diffs(&bf16_path, &f16_path);
    assert!(n_vals > 0, "no weights compared");
    assert_eq!(
        n_diff, n_sub,
        "of {n_diff} weight values that differ between the bf16 and f16 files, only \
         {n_sub} are f16 subnormals — the rest are not the f16 conversion's rounding"
    );

    // Both files, the same CPU forward. `logits_greedy` loads with
    // `Layers(0)`, so no device is involved and the claim is about weights.
    let (l16, g16) = logits_greedy(&f16_path, PROMPT, 4, 512);
    let (lbf, gbf) = logits_greedy(&bf16_path, PROMPT, 4, 512);
    assert_eq!(l16.len(), lbf.len(), "logit vector length");
    let max_abs = l16
        .iter()
        .zip(lbf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = l16
        .iter()
        .zip(lbf.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / l16.len() as f32;
    let n_bitdiff = l16
        .iter()
        .zip(lbf.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_logit = l16.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "f142 bf16 run: bf16 vs f16 max |Δlogit| = {max_abs} (mean {mean_abs}), \
         {n_bitdiff}/{} logits differ in bits, max |logit| = {max_logit} \
         ({:.3e} relative); weight values: {n_diff}/{n_vals} differ, all {n_sub} on f16 \
         subnormals; greedy bf16 {gbf:?} vs f16 {g16:?}",
        l16.len(),
        max_abs / max_logit.max(1.0),
    );
    assert_eq!(
        gbf, g16,
        "the greedy continuation differs from the f16 file"
    );
    assert!(max_abs <= 1e-4, "max |Δlogit| = {max_abs} (bound 1e-4)");
    assert!(
        max_abs / max_logit.max(1.0) <= 1e-5,
        "max relative Δlogit = {} (bound 1e-5)",
        max_abs / max_logit
    );
}
