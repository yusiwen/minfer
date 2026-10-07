//! The quantize end-to-end bounds and the byte-identical encoder.
//!
//! Split out of `src/tooling/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// G4: quantizing the converted f16 model to q8_0 produces a file that
/// loads, runs, and stays within a stated bound of the f16 source.
#[test]
#[ignore = "requires the converted f16 GGUF under /tmp/f6-work"]
fn f6_quantize_end_to_end_stays_within_the_stated_bound() {
    let Some(src) = env_path("MINFER_F6_F16_GGUF", "/tmp/f6-work/minfer-f16.gguf") else {
        return;
    };
    let dir = work_dir("quant");
    let out = dir.join("q8_0.gguf");
    let _ = std::fs::remove_file(&out);
    let src_gguf = crate::gguf::load_gguf_model(&src).expect("f16 GGUF");
    let target = QuantTarget::Q8_0;
    let plan = QuantizePlan::plan(&src_gguf, target).expect("plan quantize");
    plan.write_single(&src_gguf, &out).expect("write q8_0");
    // The output must carry the target type on the quantized weights and
    // keep the 1-D tensors as f32.
    let q = crate::gguf::load_gguf_model(&out).expect("q8_0 parses");
    for ti in &q.parts[0].ctx.info {
        let want = if ti.ne[1] <= 1 {
            crate::gguf::GgmlType::F32
        } else {
            crate::gguf::GgmlType::Q8_0
        };
        assert_eq!(ti.type_, want, "tensor {} type", ti.name);
    }
    // A target the engine reads but cannot encode is refused by name.
    let err = QuantTarget::parse("q4_K").unwrap_err();
    assert!(err.contains("no weight encoder"), "{err}");

    let (lf, gf) = logits_greedy(&src, PROMPT, 4, 512);

    // #169: the f16 arm of the same 1-D rule, on the real source. 1-D is
    // f32, 2-D is f16, every 1-D tensor is reported preserved — and because
    // every source value is already f16-representable, the f16→f16 cast
    // must be exact, so the source's own logits are an independently
    // computed expected value (bitwise, an `assert_eq!`).
    let out16 = dir.join("f16.gguf");
    let _ = std::fs::remove_file(&out16);
    let plan16 = QuantizePlan::plan(&src_gguf, QuantTarget::F16).expect("plan f16");
    plan16.write_single(&src_gguf, &out16).expect("write f16");
    let f = crate::gguf::load_gguf_model(&out16).expect("f16 parses");
    let mut n_1d = 0usize;
    for ti in &f.parts[0].ctx.info {
        let want = if ti.ne[1] <= 1 {
            n_1d += 1;
            crate::gguf::GgmlType::F32
        } else {
            crate::gguf::GgmlType::F16
        };
        assert_eq!(ti.type_, want, "f16 tensor {} type", ti.name);
    }
    assert!(n_1d > 0, "the f16 output has no 1-D tensor");
    assert_eq!(
        plan16.preserved.len(),
        n_1d,
        "every 1-D tensor must be reported preserved for f16: {:?}",
        plan16.preserved
    );
    let (l16, g16) = logits_greedy(&out16, PROMPT, 4, 512);
    assert_eq!(gf, g16, "the f16 cast changed the greedy continuation");
    assert_eq!(
        lf, l16,
        "an f16→f16 cast must be bitwise: every source value is representable"
    );

    let (lq, gq) = logits_greedy(&out, PROMPT, 4, 512);
    assert_eq!(gf, gq, "q8_0 greedy continuation differs from f16");
    let max_abs = lf
        .iter()
        .zip(lq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = lf
        .iter()
        .zip(lq.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / lf.len() as f32;
    let max_logit = lf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    // Q8_0 stores each weight with a per-32-block f16 scale; the empirical
    // bound on the 0.5B (same 4-token greedy context for both files) is
    // reported and asserted with headroom. The *decision* is what the
    // engine is judged on, and that is exact: the greedy continuation is
    // identical, so every argmax through the run agrees.
    assert!(max_abs <= 1.0, "max |Δlogit| = {max_abs}");
    assert!(
        max_abs / max_logit.max(1.0) <= 0.05,
        "max relative Δlogit = {}",
        max_abs / max_logit
    );
    eprintln!(
        "f6 quantize: q8_0 vs f16 max |Δlogit| = {max_abs} (mean {mean_abs}), \
         max |logit| = {max_logit}, greedy {:?}",
        gf
    );
}
/// Greedy continuation through whichever architecture the file declares —
/// the K-quant run-check gate runs on the same source it quantized, and the
/// 0.6B arm is Qwen3.
fn logits_greedy_any(
    path: &Path,
    prompt: &str,
    steps: usize,
    n_ctx: usize,
) -> (Vec<f32>, Vec<u32>) {
    let gguf = crate::gguf::load_gguf_model(path).expect("parse GGUF");
    let arch = gguf.parts[0]
        .ctx
        .get_key_val_str("general.architecture")
        .unwrap_or_default();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("strict tokenizer load");
    let model =
        crate::models::load_model_with(&gguf, "", crate::graph::offload::OffloadRequest::Layers(0))
            .expect("load model");
    let ids = tok.encode(prompt);
    if arch == "qwen3" {
        let q = model
            .as_any()
            .downcast_ref::<crate::models::qwen3::Qwen3Model>()
            .expect("Qwen3 model");
        logits_greedy_on_qwen3(q, &ids, steps, n_ctx)
    } else {
        let q = model
            .as_any()
            .downcast_ref::<crate::models::qwen2::Qwen2Model>()
            .expect("Qwen2 model");
        logits_greedy_on(q, &ids, steps, n_ctx)
    }
}
/// G4c (#140): a K-quant file this module wrote **runs**.
///
/// Byte parity alone is not the acceptance: the file has to load and
/// generate. The bounds are stated before measuring, in the §4.3 style:
/// **max |Δlogit| ≤ 0.30 × max |logit|** against the f16 source at the same
/// 4-token greedy context, and the **first greedy token identical** (the
/// highest-confidence decision). Everything is printed *before* the
/// assertions so a red run still reports its measurements. The full greedy
/// continuation is reported and deliberately **not** asserted: a 4-5-bit
/// quant can flip a later argmax, and the file is byte-identical to
/// `llama-quantize --pure`'s, so such a flip is a property of the
/// quantisation, not of minfer. On the 0.5B the q5_K arm flips token 2
/// (`1084` → `12095`); the q4_K/q6_K arms do not.
///
/// Run it on `~/.cache/minfer/f6-src/qwen3-0.6b-f16.gguf` (hidden 1024) to
/// exercise the engine's K-quant decode path on *every* 2-D tensor, rather
/// than on the 0.5B's 24 tensors that are not demoted.
#[test]
#[ignore = "requires an f16 GGUF and writes ~0.5 GB under /tmp/f6-work"]
fn f6_k_quant_output_runs_within_the_stated_bound() {
    let Some(src) = env_path(
        "MINFER_F6_F16_GGUF",
        "~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f16.gguf",
    ) else {
        return;
    };
    let dir = work_dir("quant-k-run");
    let src_gguf = crate::gguf::load_gguf_model(&src).expect("f16 source");
    let (lf, gf) = logits_greedy_any(&src, PROMPT, 4, 512);
    let max_logit = lf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut results = Vec::new();
    for target in [QuantTarget::Q4_K, QuantTarget::Q5_K, QuantTarget::Q6_K] {
        let out = dir.join(format!("{}.gguf", target.name()));
        let _ = std::fs::remove_file(&out);
        let plan = QuantizePlan::plan(&src_gguf, target).expect("plan");
        plan.write_single(&src_gguf, &out).expect("write");
        let (lq, gq) = logits_greedy_any(&out, PROMPT, 4, 512);
        assert!(
            lq.iter().all(|v| v.is_finite()),
            "{}: non-finite logits",
            target.name()
        );
        let max_abs = lf
            .iter()
            .zip(lq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = lf
            .iter()
            .zip(lq.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / lf.len() as f32;
        results.push((target, max_abs, mean_abs, gq));
    }
    for (t, max_abs, mean_abs, gq) in &results {
        eprintln!(
            "f6 k-quant run: {} on {} max |Δlogit| = {max_abs} (mean {mean_abs}), \
             max |logit| = {max_logit} ({:.3} relative), greedy {gq:?} \
             (f16 source greedy {gf:?}, {} of 4 tokens match)",
            t.name(),
            src.display(),
            max_abs / max_logit.max(1.0),
            gq.iter().zip(gf.iter()).filter(|(a, b)| a == b).count(),
        );
    }
    for (t, max_abs, _, gq) in &results {
        assert_eq!(
            gq.first(),
            gf.first(),
            "{}: the first greedy token differs from the f16 source",
            t.name()
        );
        assert!(
            max_abs / max_logit.max(1.0) <= 0.30,
            "{}: relative max |Δlogit| = {} (bound 0.30)",
            t.name(),
            max_abs / max_logit
        );
    }
}
/// G4b: every weight encoder is byte-identical to llama.cpp's.
///
/// `MINFER_F6_F16_GGUF` is the common f16 source and
/// `MINFER_F6_LLAMACPP_QUANT` is the file
/// `llama-quantize --pure <src> <out> <type>` produced for the target named
/// by `MINFER_F6_QUANT_TYPE` (default q4_0; `q4_K`, `q5_K` and `q6_K` all
/// pass — `--pure` disables llama.cpp's k-quant *mixture* so every
/// 256-aligned 2-D tensor goes through the one reference encoder this gate
/// is about; a bare `q4_K` is the `Q4_K_M` mixture, see
/// `docs/GGUF-TOOLING.md` §4.2). A single-nibble difference is a wrong
/// file, so this is per-tensor byte equality (the same comparison used for
/// the HF conversion). The gate prints how many tensors each encoder
/// actually encoded — on the 0.5B (hidden 896) only 24 per K type, because
/// 145 rows are not a multiple of 256 and take llama.cpp's
/// `tensor_type_fallback`; the 0.6B (hidden 1024) is the arm where every
/// 2-D tensor goes through the K encoder.
///
/// **The claim is conditional on the reference's build, so the gate names
/// it.** llvm's `quantize_row_q4_0_ref` computes `x*id + c`, and whether a
/// compiler contracts that across statements is a rounding decision: a
/// `-ffp-contract=fast` build (GCC's default; the dgxspark reference) emits one
/// FMA and matches minfer, a build without it (Apple clang by default, or
/// `-ffp-contract=off`) does not, and the two references disagree in the data
/// nibbles at exact rounding boundaries — measured 2026-10-07, 168 of 290 q4_0
/// tensors on the Mac against **290/290** on dgxspark, `docs/GGUF-TOOLING.md`
/// §4.2. So on a mismatch this gate does **not** report an encoder defect
/// first: it re-encodes the whole source with the uncontracted
/// (`FmaContract::Off`) arithmetic — exact for the legacy quants, a model of
/// the K-quants' per-expression fusion (§4.2.1) — and, if the reference matches
/// *that*, refuses loudly naming the build it compared against. A reference
/// that matches neither variant still fails as a payload difference, which is
/// what an encoder defect looks like. The pass line names the build too.
///
/// Defaults live in the **persistent** cache `~/.cache/minfer/f6-src/`
/// (f16 sources and llama-quantize references regenerated by the recipe in
/// `docs/GGUF-TOOLING.md` §4.2) — never `/tmp`. A missing env var (or a
/// default path that does not exist) prints "<var> not found at <path>;
/// skipping this F6 gate" through `env_path` and returns without
/// asserting. It prints that line, so a skipped run is visible in the
/// real-model set's output rather than silently green; the set is run by
/// `scripts/real_model_gates.sh` and its count is recorded in `AGENTS.md`.
#[test]
#[ignore = "requires an f16 GGUF plus a llama-quantize reference of the same target, and the             reference must be built with -ffp-contract=fast (GCC's default; §4.2)"]
fn f6_quantize_encoder_is_byte_identical_to_llamacpp() {
    let Some(src) = env_path(
        "MINFER_F6_F16_GGUF",
        "~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f16.gguf",
    ) else {
        return;
    };
    let Some(reff) = env_path(
        "MINFER_F6_LLAMACPP_QUANT",
        "~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-q4_0.gguf",
    ) else {
        return;
    };
    let tname = std::env::var("MINFER_F6_QUANT_TYPE").unwrap_or_else(|_| "q4_0".to_string());
    let target = QuantTarget::parse(&tname).expect("supported target");
    let dir = work_dir("quant-parity");
    let out = dir.join(format!("{tname}.gguf"));
    let _ = std::fs::remove_file(&out);
    let src_gguf = crate::gguf::load_gguf_model(&src).expect("f16 source");
    let plan = QuantizePlan::plan(&src_gguf, target).expect("plan");
    plan.write_single(&src_gguf, &out).expect("write");
    // Per-*result* counts, not a bare "n/n": how many tensors the requested
    // encoder actually saw, and how many took a demotion or a copy.
    let mut by_type: Vec<(crate::gguf::GgmlType, usize)> = Vec::new();
    for spec in &plan.specs {
        match by_type.iter_mut().find(|(t, _)| *t == spec.type_) {
            Some((_, c)) => *c += 1,
            None => by_type.push((spec.type_, 1)),
        }
    }
    let breakdown = by_type
        .iter()
        .map(|(t, c)| format!("{}: {c}", t.type_name()))
        .collect::<Vec<_>>()
        .join(", ");
    // The provenance line: which reference file, how big, and which build of
    // `llama-quantize` this run established it to be (see the doc comment).
    let ref_bytes = std::fs::metadata(&reff).map(|m| m.len()).unwrap_or(0);
    let provenance = format!("reference {} ({ref_bytes} bytes)", reff.display());
    match tensor_payloads_equal(&out, &reff) {
        Ok(n) => {
            eprintln!(
                "f6 quantize parity: {tname} == llama-quantize on {n} tensors \
                 (encoded-as {breakdown}) [source {}, {} preserved as 1-D; {provenance}, \
                 build -ffp-contract=fast]",
                src.display(),
                plan.preserved.len(),
            );
        }
        Err(primary) => {
            // Not the documented build's bytes. Ask the one other question this
            // claim is conditional on — was the reference built by a compiler
            // that does *not* contract `x*id + c`? — before reporting the
            // encoder as wrong.
            let alt = dir.join(format!("{tname}-uncontracted.gguf"));
            let _ = std::fs::remove_file(&alt);
            plan.write_single_with_contract(&src_gguf, &alt, crate::quantize::FmaContract::Off)
                .expect("write the uncontracted variant");
            match tensor_payloads_equal(&alt, &reff) {
                Ok(n) => {
                    // The K-quant `Off` variant is a model of the reference's
                    // arithmetic, not the disassembly `Fast` is built from
                    // (§4.2.1), so say which half of the claim this is.
                    let caveat = if matches!(
                        target,
                        QuantTarget::Q4_K | QuantTarget::Q5_K | QuantTarget::Q6_K
                    ) {
                        " For a K-quant target that variant is minfer's *model* of the \
                         uncontracted arithmetic, not a disassembly of the builder's object \
                         (§4.2.1), but the match is whole-file and per-tensor exact."
                    } else {
                        ""
                    };
                    panic!(
                        "f6 quantize parity: {tname}: the reference is NOT the \
                         `-ffp-contract=fast` build this encoder reproduces. All {n} tensors \
                         match the uncontracted `x*id + c` -> `fmul` + `fadd` variant instead — \
                         what a `llama-quantize` built without cross-statement FMA contraction \
                         produces (Apple clang without `-ffp-contract=fast`, or any compiler \
                         with `-ffp-contract=off`). The two builds disagree only in the data \
                         nibbles at exact rounding boundaries, so this is a reference-build \
                         mismatch, not an encoder defect.{caveat} Documented reference build: \
                         `docs/GGUF-TOOLING.md` §4.2 (a GCC/clang `-ffp-contract=fast` build; \
                         dgxspark's GCC 13.3.0 build reproduces 290/290). Rebuild \
                         `llama-quantize` with `-ffp-contract=fast`, or point \
                         MINFER_F6_LLAMACPP_QUANT at a reference from such a build. \
                         {provenance}; uncontracted variant written to {} ({primary})",
                        alt.display(),
                    );
                }
                Err(_) => panic!("{primary} ({provenance}, build unmatched)"),
            }
        }
    }
}
