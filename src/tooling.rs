// F6 (issue #49): the `convert` / `quantize` / `split` subcommands.
//
// Each one parses its own arguments and is refused loudly on anything it
// cannot honour. Nothing here falls back silently: an unknown target, an
// unsupported architecture, a tensor without a mapping, an unsafe split cap —
// all are named errors with a non-zero exit.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::convert::{Conversion, OutType, QuantizePlan};
use crate::gguf_write;
use crate::quantize::QuantTarget;

/// Parse a byte size with an optional binary suffix: `1073741824`, `1G`, `512M`, `300k`.
pub fn parse_size(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if t.is_empty() {
        return Err("empty size".to_string());
    }
    let (num, mult) = match t.as_bytes()[t.len() - 1].to_ascii_lowercase() {
        b'k' => (&t[..t.len() - 1], 1024u64),
        b'm' => (&t[..t.len() - 1], 1024 * 1024),
        b'g' => (&t[..t.len() - 1], 1024 * 1024 * 1024),
        _ => (t, 1),
    };
    let n: u64 = num
        .trim()
        .parse()
        .map_err(|_| format!("invalid size {s:?} (expected bytes, or a K/M/G suffix)"))?;
    n.checked_mul(mult)
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("size {s:?} is not a positive number of bytes"))
}

fn usage_convert(prog: &str) -> String {
    format!(
        "Usage: {prog} convert <hf-model-dir> <out.gguf> [--outtype f16|f32] [--split-max-size BYTES]\n\
         \n\
         Converts a HuggingFace Qwen2 checkpoint (config.json + tokenizer.json +\n\
         tokenizer_config.json + model.safetensors) into GGUF v3. Supported\n\
         architectures: Qwen2ForCausalLM only; output types: f16 (default), f32."
    )
}

fn usage_quantize(prog: &str) -> String {
    format!(
        "Usage: {prog} quantize <in.gguf> <out.gguf> --type <target> [--split-max-size BYTES]\n\
         \n\
         Re-encodes a single-file GGUF's 2-D float weights to <target>. One-dimensional\n\
         tensors (norms, biases) keep their source type. Supported targets: {}.",
        crate::quantize::SUPPORTED_TARGETS
    )
}

fn usage_split(prog: &str) -> String {
    format!(
        "Usage: {prog} split <in.gguf> <out-dir> --max-size BYTES [--stem NAME]\n\
         \n\
         Writes a single-file GGUF as `{{stem}}-NNNNN-of-MMMMM.gguf` parts under\n\
         <out-dir>, each no larger than --max-size bytes of tensor data. A tensor is\n\
         never split across parts. When everything fits in one part the output is\n\
         `{{stem}}.gguf` (a single file, no split metadata)."
    )
}

/// Split an output `.gguf` path into (directory, stem-without-.gguf).
fn split_targets(out: &Path) -> (PathBuf, String) {
    let dir = out.parent().unwrap_or(Path::new(".")).to_path_buf();
    let stem = out
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .strip_suffix(".gguf")
        .unwrap_or_else(|| out.file_name().and_then(|n| n.to_str()).unwrap_or("model"))
        .to_string();
    (dir, stem)
}

/// `minfer convert …`
pub fn run_convert(prog: &str, args: &[String]) -> i32 {
    let mut positional: Vec<String> = Vec::new();
    let mut outtype = OutType::F16;
    let mut split_max: Option<u64> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", usage_convert(prog));
                return 0;
            }
            "--outtype" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("Error: --outtype needs a value");
                    return 1;
                };
                match OutType::parse(v) {
                    Ok(t) => outtype = t,
                    Err(e) => {
                        eprintln!("Error: {e}");
                        return 1;
                    }
                }
                i += 2;
            }
            "--split-max-size" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("Error: --split-max-size needs a value");
                    return 1;
                };
                match parse_size(v) {
                    Ok(n) => split_max = Some(n),
                    Err(e) => {
                        eprintln!("Error: {e}");
                        return 1;
                    }
                }
                i += 2;
            }
            a if a.starts_with('-') => {
                eprintln!("Error: unknown option '{a}' for convert");
                eprintln!("{}", usage_convert(prog));
                return 1;
            }
            a => {
                positional.push(a.to_string());
                i += 1;
            }
        }
    }
    if positional.len() != 2 {
        eprintln!("{}", usage_convert(prog));
        return 1;
    }
    let src = Path::new(&positional[0]);
    let out = Path::new(&positional[1]);
    if !src.is_dir() {
        eprintln!(
            "Error: minfer convert: {} is not a directory (expected a HuggingFace checkpoint \
             directory)",
            src.display()
        );
        return 1;
    }
    if out.is_dir() {
        eprintln!(
            "Error: minfer convert: output {} is a directory; give a .gguf file path",
            out.display()
        );
        return 1;
    }

    let conv = match Conversion::plan(src, outtype) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    eprintln!(
        "convert: {} tensors, source {} -> {}",
        conv.specs.len(),
        src.display(),
        outtype.name()
    );
    let result = match split_max {
        None => conv.write_single(out).map(|_| vec![out.to_path_buf()]),
        Some(cap) => {
            let (dir, stem) = split_targets(out);
            conv.write_split(&dir, &stem, cap)
        }
    };
    match result {
        Ok(paths) => {
            for p in &paths {
                match std::fs::metadata(p) {
                    Ok(m) => eprintln!("wrote {} ({} bytes)", p.display(), m.len()),
                    Err(_) => eprintln!("wrote {}", p.display()),
                }
            }
            0
        }
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

/// `minfer quantize …`
pub fn run_quantize(prog: &str, args: &[String]) -> i32 {
    let mut positional: Vec<String> = Vec::new();
    let mut target: Option<QuantTarget> = None;
    let mut split_max: Option<u64> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", usage_quantize(prog));
                return 0;
            }
            "--type" | "-t" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("Error: --type needs a value");
                    return 1;
                };
                match QuantTarget::parse(v) {
                    Ok(t) => target = Some(t),
                    Err(e) => {
                        eprintln!("Error: {e}");
                        return 1;
                    }
                }
                i += 2;
            }
            "--split-max-size" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("Error: --split-max-size needs a value");
                    return 1;
                };
                match parse_size(v) {
                    Ok(n) => split_max = Some(n),
                    Err(e) => {
                        eprintln!("Error: {e}");
                        return 1;
                    }
                }
                i += 2;
            }
            a if a.starts_with('-') => {
                eprintln!("Error: unknown option '{a}' for quantize");
                eprintln!("{}", usage_quantize(prog));
                return 1;
            }
            a => {
                positional.push(a.to_string());
                i += 1;
            }
        }
    }
    let Some(target) = target else {
        eprintln!(
            "Error: minfer quantize: --type is required (no silent default: a wrong target would \
             write wrong weights); supported: {}",
            crate::quantize::SUPPORTED_TARGETS
        );
        eprintln!("{}", usage_quantize(prog));
        return 1;
    };
    if positional.len() != 2 {
        eprintln!("{}", usage_quantize(prog));
        return 1;
    }
    let in_path = Path::new(&positional[0]);
    let out = Path::new(&positional[1]);
    if !in_path.is_file() {
        eprintln!(
            "Error: minfer quantize: {} is not a file",
            in_path.display()
        );
        return 1;
    }
    let model = match crate::gguf::load_gguf_model(in_path) {
        Some(m) => m,
        None => {
            eprintln!(
                "Error: minfer quantize: failed to read GGUF from {}",
                in_path.display()
            );
            return 1;
        }
    };
    if model.parts.len() != 1 {
        eprintln!(
            "Error: minfer quantize: {} is a multi-part split ({} parts); quantize a single-file \
             GGUF, then split the result",
            in_path.display(),
            model.parts.len()
        );
        return 1;
    }
    let plan = match QuantizePlan::plan(&model, target) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    if !plan.preserved.is_empty() {
        // The reason differs per target: a quant target also preserves a 2-D
        // tensor whose row length is not block-aligned, while f16 preserves
        // 1-D tensors only (its block size is 1, so the row-length clause would
        // be meaningless). #169.
        let why = if target == QuantTarget::F16 {
            "1-D".to_string()
        } else {
            format!("1-D or row length not a multiple of {}", target.blck_size())
        };
        eprintln!(
            "quantize: {} tensor(s) keep their source type ({why}): {}",
            plan.preserved.len(),
            plan.preserved.join(", ")
        );
    }
    if !plan.retargeted.is_empty() {
        eprintln!(
            "quantize: tied embedding quantized at q8_0 (llama.cpp's sub-8-bit policy): {}",
            plan.retargeted.join(", ")
        );
    }
    let result = match split_max {
        None => plan
            .write_single(&model, out)
            .map(|_| vec![out.to_path_buf()]),
        Some(cap) => {
            let (dir, stem) = split_targets(out);
            plan.write_split(&model, &dir, &stem, cap)
        }
    };
    match result {
        Ok(paths) => {
            for p in &paths {
                match std::fs::metadata(p) {
                    Ok(m) => eprintln!("wrote {} ({} bytes)", p.display(), m.len()),
                    Err(_) => eprintln!("wrote {}", p.display()),
                }
            }
            0
        }
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

/// Copy a single-file GGUF through the writer with its tensor bytes verbatim
/// (no re-encoding) — the `split` operation, and the F6 round-trip gate.
///
/// A `max_part_bytes` larger than the whole data section produces one file
/// named `{stem}.gguf`; a smaller cap produces
/// `{stem}-NNNNN-of-MMMMM.gguf` parts. A multi-part input is refused (the
/// split convention writes parts, it does not re-merge them).
pub fn write_verbatim(
    model: &crate::gguf::GgufModel,
    out_dir: &Path,
    stem: &str,
    max_part_bytes: u64,
) -> Result<Vec<PathBuf>, String> {
    if model.parts.len() != 1 {
        return Err(format!(
            "minfer: refusing to split a file that is already a multi-part split ({} parts)",
            model.parts.len()
        ));
    }
    let part = &model.parts[0];
    let ctx = &part.ctx;
    if ctx.info.is_empty() {
        return Err("minfer: the GGUF has no tensors".to_string());
    }
    let specs: Vec<gguf_write::TensorSpec> = ctx
        .info
        .iter()
        .map(|ti| gguf_write::TensorSpec::new(ti.name.clone(), ti.ne, ti.type_))
        .collect();
    let data = |i: usize, w: &mut dyn Write| -> std::io::Result<()> {
        let ti = &ctx.info[i];
        let off = ctx.offset + ti.offset as usize;
        let n = ti.nbytes();
        w.write_all(&part.data[off..off + n])
    };
    gguf_write::write_split(
        out_dir,
        stem,
        &ctx.kv,
        specs,
        ctx.alignment,
        max_part_bytes,
        data,
    )
}

/// `minfer split …`
pub fn run_split(prog: &str, args: &[String]) -> i32 {
    let mut positional: Vec<String> = Vec::new();
    let mut max_size: Option<u64> = None;
    let mut stem_opt: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", usage_split(prog));
                return 0;
            }
            "--max-size" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("Error: --max-size needs a value");
                    return 1;
                };
                match parse_size(v) {
                    Ok(n) => max_size = Some(n),
                    Err(e) => {
                        eprintln!("Error: {e}");
                        return 1;
                    }
                }
                i += 2;
            }
            "--stem" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("Error: --stem needs a value");
                    return 1;
                };
                stem_opt = Some(v.clone());
                i += 2;
            }
            a if a.starts_with('-') => {
                eprintln!("Error: unknown option '{a}' for split");
                eprintln!("{}", usage_split(prog));
                return 1;
            }
            a => {
                positional.push(a.to_string());
                i += 1;
            }
        }
    }
    let Some(cap) = max_size else {
        eprintln!("Error: minfer split: --max-size is required (no silent default part size)");
        eprintln!("{}", usage_split(prog));
        return 1;
    };
    if positional.len() != 2 {
        eprintln!("{}", usage_split(prog));
        return 1;
    }
    let in_path = Path::new(&positional[0]);
    let out_dir = Path::new(&positional[1]);
    if !in_path.is_file() {
        eprintln!("Error: minfer split: {} is not a file", in_path.display());
        return 1;
    }
    let model = match crate::gguf::load_gguf_model(in_path) {
        Some(m) => m,
        None => {
            eprintln!(
                "Error: minfer split: failed to read GGUF from {}",
                in_path.display()
            );
            return 1;
        }
    };
    if model.parts.len() != 1 {
        eprintln!(
            "Error: minfer split: {} is already a multi-part split ({} parts); point at a \
             single-file GGUF",
            in_path.display(),
            model.parts.len()
        );
        return 1;
    }
    if model.parts[0].ctx.info.is_empty() {
        eprintln!(
            "Error: minfer split: {} has no tensors; nothing to split",
            in_path.display()
        );
        return 1;
    }
    if let Some(info) =
        crate::gguf::split_file_info(in_path.file_name().and_then(|n| n.to_str()).unwrap_or(""))
    {
        eprintln!(
            "Error: minfer split: {} already looks like split part {} of {} (prefix {:?}); point \
             at a single-file GGUF",
            in_path.display(),
            info.1 + 1,
            info.2,
            info.0
        );
        return 1;
    }
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!(
            "Error: minfer split: cannot create {}: {e}",
            out_dir.display()
        );
        return 1;
    }
    let stem = stem_opt.unwrap_or_else(|| {
        in_path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("model")
            .to_string()
    });
    match write_verbatim(&model, out_dir, &stem, cap) {
        Ok(paths) => {
            for p in &paths {
                match std::fs::metadata(p) {
                    Ok(m) => eprintln!("wrote {} ({} bytes)", p.display(), m.len()),
                    Err(_) => eprintln!("wrote {}", p.display()),
                }
            }
            0
        }
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;

    #[test]
    fn parse_size_accepts_bytes_and_binary_suffixes() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("1K").unwrap(), 1024);
        assert_eq!(parse_size("2m").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_size("1G").unwrap(), 1024 * 1024 * 1024);
        for bad in ["", "abc", "1.5M", "-4", "0", "0K"] {
            assert!(parse_size(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn split_targets_strips_the_gguf_suffix() {
        let (dir, stem) = split_targets(Path::new("/tmp/f6-work/model.gguf"));
        assert_eq!(dir, Path::new("/tmp/f6-work"));
        assert_eq!(stem, "model");
    }

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

    // === F6 real-model gates (#[ignore]: CI has no checkpoint/model) ===

    /// The prompt every F6 logit gate uses. Short enough to prefill quickly on
    /// the 0.5B, long enough that a position/weight error moves a logit.
    const PROMPT: &str = "The capital of France is";

    fn env_path(key: &str, default: &str) -> Option<PathBuf> {
        let p = std::env::var_os(key).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(default.replace('~', &std::env::var("HOME").unwrap()))
        });
        if p.exists() {
            Some(p)
        } else {
            eprintln!("{key} not found at {}; skipping this F6 gate", p.display());
            None
        }
    }

    fn work_dir(name: &str) -> PathBuf {
        let d = PathBuf::from("/tmp/f6-work").join(name);
        std::fs::create_dir_all(&d).expect("create work dir");
        d
    }

    fn cached_qwen05() -> PathBuf {
        let home = std::env::var("HOME").unwrap();
        PathBuf::from(home).join(
            ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/\
             qwen2.5-0.5b-instruct-q4_k_m.gguf",
        )
    }

    /// Greedy continuation `steps` tokens past `prompt`, returning the
    /// final-step logits (whole vocabulary) and the sampled token ids.
    ///
    /// Forced onto the CPU plan (`OffloadRequest::Layers(0)`) so the bitwise
    /// comparisons are device-independent and no device registration is done —
    /// CI has no GPU, and the logit claims here are about *weights*, not backends.
    fn logits_greedy(
        path: &Path,
        prompt: &str,
        steps: usize,
        n_ctx: usize,
    ) -> (Vec<f32>, Vec<u32>) {
        let gguf = crate::gguf::load_gguf_model(path).expect("parse GGUF");
        let tok =
            crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("strict tokenizer load");
        let model = crate::models::load_model_with(
            &gguf,
            "",
            crate::graph::offload::OffloadRequest::Layers(0),
        )
        .expect("load model");
        let q = model
            .as_any()
            .downcast_ref::<crate::models::qwen2::Qwen2Model>()
            .expect("Qwen2 model");
        logits_greedy_on(q, &tok.encode(prompt), steps, n_ctx)
    }

    /// The run itself, on an already-loaded model. Shared by the CPU gates
    /// (which load with `Layers(0)`) and #141's device arm (which loads with a
    /// full device plan): the numeric comparison must be the same forward code
    /// on both sides, differing only in the backend the scheduler assigns.
    fn logits_greedy_on(
        q: &crate::models::qwen2::Qwen2Model,
        ids: &[u32],
        steps: usize,
        n_ctx: usize,
    ) -> (Vec<f32>, Vec<u32>) {
        let nt = ids.len();
        let mut cache = crate::graph::cache::GraphCache::new();
        let mut positions: Vec<usize> = (0..nt).collect();
        let mut logits = crate::models::qwen2::graph::Qwen2Graph::forward_cached(
            q, ids, &positions, 1, n_ctx, &mut cache,
        );
        let mut toks = Vec::with_capacity(steps);
        for step in 0..steps {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            toks.push(next);
            positions = vec![nt + step];
            logits = crate::models::qwen2::graph::Qwen2Graph::forward_cached(
                q,
                &[next],
                &positions,
                1,
                n_ctx,
                &mut cache,
            );
        }
        let _ = positions;
        (logits, toks)
    }

    /// Compare every tensor payload of two GGUFs by name (offset-independent).
    fn assert_tensor_payloads_equal(a_path: &Path, b_path: &Path) {
        let ma = crate::gguf::load_gguf_model(a_path).expect("a parses");
        let mb = crate::gguf::load_gguf_model(b_path).expect("b parses");
        let mut index = std::collections::HashMap::new();
        for (pi, part) in ma.parts.iter().enumerate() {
            for ti in &part.ctx.info {
                let off = part.ctx.offset + ti.offset as usize;
                index.insert(ti.name.clone(), (pi, off, ti.nbytes()));
            }
        }
        let mut n = 0;
        for (pi, part) in mb.parts.iter().enumerate() {
            for ti in &part.ctx.info {
                let off = part.ctx.offset + ti.offset as usize;
                let (pa, oa, na) = index
                    .get(&ti.name)
                    .unwrap_or_else(|| panic!("tensor '{}' only in the second file", ti.name));
                assert_eq!(*na, ti.nbytes(), "tensor {} size", ti.name);
                assert_eq!(
                    &ma.parts[*pa].data[*oa..*oa + *na],
                    &mb.parts[pi].data[off..off + ti.nbytes()],
                    "tensor {} payload differs",
                    ti.name
                );
                n += 1;
            }
        }
        assert_eq!(n, index.len(), "tensor set differs");
        assert!(n > 0);
    }

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
            let sb =
                &rb[b.offset + tb.offset as usize..b.offset + tb.offset as usize + tb.nbytes()];
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
        let Some(ref_gguf) = env_path("MINFER_F6_LLAMACPP_GGUF", "/tmp/f6-work/ref-f16.gguf")
        else {
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
        let tok =
            crate::tokenizer::Tokenizer::load(ctx).expect("strict tokenizer accepts the file");
        let tmpl = ctx
            .get_key_val_str("tokenizer.chat_template")
            .expect("chat template present");
        let rendered = crate::template::render_template(&tmpl, "hello", true, "")
            .expect("chat template renders");
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

    /// G2d (#141): the f16 weights execute on the CUDA **device**, and the
    /// device logits agree with the same file's CPU logits within the stated
    /// backend tolerance.
    ///
    /// Placement is the load-bearing claim, so it is asserted directly rather
    /// than inferred from a timing: the loader's offload report must say every
    /// block is on the device, and every `F16` matmul / embedding node the
    /// scheduler assigns must be `Backend::CUDA`. Break the loader's f16
    /// registration (drop `TensorType::F16` from the CUDA branch) and
    /// `Qwen2Model::device()` answers `Cpu`: this gate then fails on its first
    /// assertion instead of quietly measuring the CPU.
    ///
    /// The model comes from `minfer convert --outtype f16` on the HF
    /// checkpoint, i.e. the F6 acceptance's own file, not a quantize-produced
    /// one — see `MINFER_F141_F16_GGUF`.
    ///
    /// Tolerance: CPU and CUDA reduce in different orders and run different
    /// exp/softmax kernels (graph rule §9), so the two paths are not bit-equal
    /// by design. What must hold is the *decision*: the greedy continuation is
    /// identical. The numeric bound is printed and asserted with headroom.
    #[test]
    #[ignore = "requires a converted f16 GGUF and a CUDA device (#141)"]
    fn f141_f16_weights_run_on_the_cuda_device() {
        #[cfg(not(feature = "cuda"))]
        eprintln!("not a CUDA build; #141's device gate is a no-op here");
        #[cfg(feature = "cuda")]
        {
            use crate::graph::offload::OffloadRequest;
            use crate::graph::Backend;
            use crate::models::qwen2::graph::Qwen2Graph;
            use crate::models::Device;
            use crate::tensor::TensorType;

            // Bring CUDA up the way a load would (the loader calls this too);
            // `CudaState::get()` is None until something initializes it.
            crate::cuda::CudaState::init();
            if crate::cuda::CudaState::get().is_none() {
                eprintln!("no CUDA device; skipping the #141 device gate");
                return;
            }
            let Some(path) = env_path("MINFER_F141_F16_GGUF", "/tmp/f141-work/minfer-f16.gguf")
            else {
                return;
            };
            let gguf = crate::gguf::load_gguf_model(&path).expect("f16 GGUF parses");
            let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx)
                .expect("strict tokenizer accepts the converted file");
            let ids = tok.encode(PROMPT);
            assert!(!ids.is_empty());

            // Fully typed weights only: the converted file's 2-D tensors are f16
            // and its 1-D norms/biases f32. A silent fallback would mean the
            // device plan is not honest, so state it.
            let mut n_f16 = 0usize;
            let mut n_other = 0usize;
            for ti in &gguf.parts[0].ctx.info {
                if ti.ne[1] > 1 {
                    if ti.type_ == crate::gguf::GgmlType::F16 {
                        n_f16 += 1;
                    } else {
                        n_other += 1;
                    }
                }
            }
            assert!(n_f16 > 0, "the f16 file has no 2-D f16 weight");
            assert_eq!(n_other, 0, "every 2-D weight must be f16 in this file");

            // --- device arm: an explicit full plan (no env involvement) --------
            //
            // Both arms load under a **namespace**. The CUDA weight registry is
            // process-global and name-keyed, and `register_weight` reuses a
            // same-name+same-size device copy; the other real-model gates in this
            // set load the cached q4_k_m 0.5B under the default `ns=""`, which
            // shares 121 f32 norm/bias names *and* their byte sizes with this
            // file — but not their values (the cached file's f32 tensors are a
            // different checkpoint revision: `blk.0.attn_norm.weight[0]` is
            // -0.046875 here, matching the HF bf16 safetensors, and -0.082947
            // there). Without the namespace the device arm silently computes with
            // another file's norms and the greedy continuation diverges — the
            // process-global hazard #64 documents, and why the loader has `ns`.
            let dev =
                crate::models::load_model_with(&gguf, "f141:", OffloadRequest::Layers(usize::MAX))
                    .expect("device load");
            let q = dev
                .as_any()
                .downcast_ref::<crate::models::qwen2::Qwen2Model>()
                .expect("Qwen2 model");
            assert_eq!(
                Qwen2Graph::device(q),
                Device::Cuda,
                "an f16 GGUF did not make the model a CUDA model — its weights are not \
                 registered/consumable on the device"
            );
            let report = dev.offload_report().expect("offload report");
            assert!(
                report.contains("on cuda") && !report.contains("cpu only"),
                "offload report does not put the blocks on the device: {report}"
            );
            assert_eq!(
                crate::models::ModelDef::offload(q).gpu_layers,
                crate::models::ModelDef::offload(q).n_layers,
                "not every block is on the device: {report}"
            );

            // Every f16 node the scheduler assigns must be CUDA. This is the
            // assertion that a *registered* weight with no kernel would break:
            // `supports_op` would route the node to the CPU and the count below
            // would drop (or the backend assert would fire).
            let params = crate::graph::params::GraphParams {
                n_tokens: ids.len(),
                n_out: 1,
                gtype: crate::graph::params::GraphType::Prefill,
                cparams: crate::graph::params::CParams {
                    n_ctx: 512,
                    flash_attn: false,
                    explicit_span: false,
                    kv_map: false,
                    gpu: true,
                    gpu_layers: usize::MAX,
                    kv_format: crate::graph::kvformat::KvFormat::F32,
                    fuse_qkv: true,
                    fuse_ffn: true,
                },
                weights_version: 0,
            };
            let mut graph =
                <crate::models::qwen2::Qwen2Model as crate::models::ModelDef>::build_graph(
                    q, &params,
                );
            let mut alloc = crate::graph::alloc::GraphAllocator::new();
            assert!(alloc.enable_cuda(), "the CUDA allocator did not come up");
            crate::graph::scheduler::BackendScheduler::new().assign_backends(&mut graph, &alloc);
            let mut f16_matmul = 0usize;
            let mut f16_embed = 0usize;
            for nd in &graph.nodes {
                let wt = match &nd.meta {
                    crate::graph::ops::NodeMeta::MatMul(m) => Some(m.weight_ttype),
                    crate::graph::ops::NodeMeta::Embed(m) => Some(m.weight_ttype),
                    _ => None,
                };
                if wt != Some(TensorType::F16) {
                    continue;
                }
                match nd.op {
                    crate::graph::ops::Op::MatMul { .. } => f16_matmul += 1,
                    crate::graph::ops::Op::GetRows => f16_embed += 1,
                    _ => {}
                }
                assert_eq!(
                    nd.backend,
                    Some(Backend::CUDA),
                    "f16 node '{}' ({:?}) was not assigned the device",
                    nd.name,
                    nd.op
                );
            }
            // 24 blocks × (q,k,v,o,gate,up,down) + lm_head; measured 169 matmul
            // + 1 embed f16 nodes on the 0.5B. The floor guards against a
            // vacuous pass on an empty node set.
            assert!(
                f16_matmul >= 24 * 7,
                "only {f16_matmul} f16 matmul nodes in the graph"
            );
            assert!(f16_embed >= 1, "the f16 embedding node is missing");

            // --- CPU arm: the same file, the same forward, Layers(0) ----------
            let cpu = crate::models::load_model_with(&gguf, "f141:", OffloadRequest::Layers(0))
                .expect("cpu load");
            let qc = cpu
                .as_any()
                .downcast_ref::<crate::models::qwen2::Qwen2Model>()
                .expect("Qwen2 model");
            assert_eq!(
                Qwen2Graph::device(qc),
                Device::Cpu,
                "the CPU arm must not be a device model"
            );

            let (ld, gd) = logits_greedy_on(q, &ids, 4, 512);
            let (lc, gc) = logits_greedy_on(qc, &ids, 4, 512);
            assert_eq!(gd, gc, "greedy continuation differs between device and CPU");
            assert_eq!(ld.len(), lc.len());
            let max_abs = ld
                .iter()
                .zip(lc.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean_abs = ld
                .iter()
                .zip(lc.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / ld.len() as f32;
            let max_logit = ld.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "f141 f16 device: {n_f16} f16 tensors, {f16_matmul} f16 matmul + {f16_embed} \
                 embed nodes on CUDA, max |Δlogit| = {max_abs} (mean {mean_abs}), max |logit| = \
                 {max_logit}, greedy {:?}",
                gd
            );
            // The stated backend tolerance. Both paths compute f32 activations
            // against f16 weights (an f16 weight has no integer form, so the CPU
            // does *not* quantize its activations as it does for the quantized
            // types), so the difference is accumulation order plus the attention
            // exp/softmax kernel — not weight precision. Measured on the 0.5B at
            // ctx 512: 7.34e-5 absolute, 4.0e-6 relative to the largest logit.
            // The bound keeps ~130x headroom over that, which is still far below
            // any weight-level fault (a wrong f16 row moves logits by O(1)).
            assert!(max_abs <= 0.01, "max |Δlogit| = {max_abs}");
            assert!(
                max_abs / max_logit.max(1.0) <= 1e-3,
                "max relative Δlogit = {}",
                max_abs / max_logit
            );
        }
    }

    /// #167: the Qwen3 twin of the #141 gate. An f16 **Qwen3** GGUF runs on the
    /// CUDA device — the loader registers the f16 weights raw and
    /// `Qwen3Graph::weights_on_cuda` admits the type — and its device logits agree
    /// with the same file's CPU logits within a stated bound (max |Δlogit| ≤ 0.05
    /// and ≤ 1e-3 relative; the gate prints both measured values).
    ///
    /// Why a separate gate: `f141_f16_weights_run_on_the_cuda_device` is
    /// qwen2-specific (it downcasts to `Qwen2Model`), and the qwen3 loader was
    /// exactly the copy that lacked the f16 arm. Placement is asserted directly,
    /// not inferred from a timing: every `F16` matmul / embedding node the
    /// scheduler assigns must be `Backend::CUDA`, so a silent CPU fallback fails on
    /// the first such node instead of reporting a CPU number as a device one.
    ///
    /// The model is `llama-quantize --allow-requantize … F16` on the cached
    /// `Qwen3-0.6B-Q8_0.gguf` (a Q8_0 file is not a K-quant, so re-quantizing is
    /// allowed). **Not** `minfer quantize --type f16`: that path converts 1-D norms to
    /// f16 too, while the engine's f16 contract (and llama.cpp's rule) keeps them f32 —
    /// the assertion below pins the contract, and the tooling deviation is filed
    /// separately. See `MINFER_F167_F16_GGUF`.
    #[test]
    #[ignore = "requires a converted f16 Qwen3 GGUF and a CUDA device (#167)"]
    fn f167_f16_qwen3_weights_run_on_the_cuda_device() {
        #[cfg(not(feature = "cuda"))]
        eprintln!("not a CUDA build; #167's f16 Qwen3 gate is a no-op here");
        #[cfg(feature = "cuda")]
        {
            use crate::graph::offload::OffloadRequest;
            use crate::graph::Backend;
            use crate::models::qwen3::graph::Qwen3Graph;
            use crate::models::Device;
            use crate::tensor::TensorType;

            crate::cuda::CudaState::init();
            if crate::cuda::CudaState::get().is_none() {
                eprintln!("no CUDA device; skipping the #167 f16 Qwen3 gate");
                return;
            }
            let Some(path) = env_path("MINFER_F167_F16_GGUF", "/tmp/f167-work/qwen3-f16.gguf")
            else {
                return;
            };
            let gguf = crate::gguf::load_gguf_model(&path).expect("f16 GGUF parses");
            // The gate must not pass by loading some other architecture's file.
            assert_eq!(
                gguf.parts[0]
                    .ctx
                    .get_key_val_str("general.architecture")
                    .as_deref(),
                Some("qwen3"),
                "#167's f16 gate requires a Qwen3 GGUF"
            );
            let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx)
                .expect("strict tokenizer accepts the converted file");
            let ids = tok.encode(PROMPT);
            assert!(!ids.is_empty());

            let mut n_f16 = 0usize;
            let mut n_other = 0usize;
            let mut n_1d_f32 = 0usize;
            let mut n_1d_other = 0usize;
            for ti in &gguf.parts[0].ctx.info {
                if ti.ne[1] > 1 {
                    if ti.type_ == crate::gguf::GgmlType::F16 {
                        n_f16 += 1;
                    } else {
                        n_other += 1;
                    }
                } else if ti.type_ == crate::gguf::GgmlType::F32 {
                    n_1d_f32 += 1;
                } else {
                    n_1d_other += 1;
                }
            }
            assert!(n_f16 > 0, "the f16 file has no 2-D f16 weight");
            assert_eq!(n_other, 0, "every 2-D weight must be f16 in this file");
            // The engine's f16 contract: 1-D norms/biases stay f32 (the converter's and
            // llama.cpp's rule; `mat_mul_f16`/`embed_rows_f16` have no f16-norm sibling,
            // and the CPU/device RMSNorm reads f32 weights). Asserting it here is what
            // stops the gate from accepting an f16 file the engine cannot actually run.
            assert!(n_1d_f32 > 0, "the f16 file has no 1-D f32 norm/bias");
            assert_eq!(
                n_1d_other, 0,
                "every 1-D norm/bias must stay f32 in an f16 file"
            );

            // A namespace, for the same process-global-registry reason as f141: the
            // other real-model gates register same-named 0.5B/0.6B tensors under ns="".
            let dev = crate::models::load_model_with(
                &gguf,
                "f167f16:",
                OffloadRequest::Layers(usize::MAX),
            )
            .expect("device load");
            let q = dev
                .as_any()
                .downcast_ref::<crate::models::qwen3::Qwen3Model>()
                .expect("Qwen3 model");
            assert_eq!(
                Qwen3Graph::device(q),
                Device::Cuda,
                "an f16 Qwen3 GGUF did not make the model a CUDA model — the loader's f16 \
                 registration or the graph's f16 type gate is missing"
            );
            let report = dev.offload_report().expect("offload report");
            assert!(
                report.contains("on cuda") && !report.contains("cpu only"),
                "offload report does not put the blocks on the device: {report}"
            );
            let n_layers = crate::models::ModelDef::offload(q).n_layers;
            assert_eq!(
                crate::models::ModelDef::offload(q).gpu_layers,
                n_layers,
                "not every block is on the device: {report}"
            );

            let params = crate::graph::params::GraphParams {
                n_tokens: ids.len(),
                n_out: 1,
                gtype: crate::graph::params::GraphType::Prefill,
                cparams: crate::graph::params::CParams {
                    n_ctx: 512,
                    flash_attn: false,
                    explicit_span: false,
                    kv_map: false,
                    gpu: true,
                    gpu_layers: usize::MAX,
                    kv_format: crate::graph::kvformat::KvFormat::F32,
                    fuse_qkv: true,
                    fuse_ffn: true,
                },
                weights_version: 0,
            };
            let mut graph =
                <crate::models::qwen3::Qwen3Model as crate::models::ModelDef>::build_graph(
                    q, &params,
                );
            let mut alloc = crate::graph::alloc::GraphAllocator::new();
            assert!(alloc.enable_cuda(), "the CUDA allocator did not come up");
            crate::graph::scheduler::BackendScheduler::new().assign_backends(&mut graph, &alloc);
            let mut f16_matmul = 0usize;
            let mut f16_embed = 0usize;
            for nd in &graph.nodes {
                let wt = match &nd.meta {
                    crate::graph::ops::NodeMeta::MatMul(m) => Some(m.weight_ttype),
                    crate::graph::ops::NodeMeta::Embed(m) => Some(m.weight_ttype),
                    _ => None,
                };
                if wt != Some(TensorType::F16) {
                    continue;
                }
                match nd.op {
                    crate::graph::ops::Op::MatMul { .. } => f16_matmul += 1,
                    crate::graph::ops::Op::GetRows => f16_embed += 1,
                    _ => {}
                }
                assert_eq!(
                    nd.backend,
                    Some(Backend::CUDA),
                    "f16 node '{}' ({:?}) was not assigned the device",
                    nd.name,
                    nd.op
                );
            }
            // 7 f16 matmuls per block plus lm_head; the floor guards against a
            // vacuous pass on an empty node set (the count is model-derived).
            assert!(
                f16_matmul >= n_layers * 7,
                "only {f16_matmul} f16 matmul nodes in the graph for {n_layers} layers"
            );
            assert!(f16_embed >= 1, "the f16 embedding node is missing");

            let cpu = crate::models::load_model_with(&gguf, "f167f16:", OffloadRequest::Layers(0))
                .expect("cpu load");
            let qc = cpu
                .as_any()
                .downcast_ref::<crate::models::qwen3::Qwen3Model>()
                .expect("Qwen3 model");
            assert_eq!(
                Qwen3Graph::device(qc),
                Device::Cpu,
                "the CPU arm must not be a device model"
            );

            let (ld, gd) = logits_greedy_on_qwen3(q, &ids, 4, 512);
            let (lc, gc) = logits_greedy_on_qwen3(qc, &ids, 4, 512);
            assert_eq!(gd, gc, "greedy continuation differs between device and CPU");
            assert_eq!(ld.len(), lc.len());
            let max_abs = ld
                .iter()
                .zip(lc.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean_abs = ld
                .iter()
                .zip(lc.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / ld.len() as f32;
            let max_logit = ld.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "f167 f16 qwen3 device: {n_f16} f16 tensors, {f16_matmul} f16 matmul + \
                 {f16_embed} embed nodes on CUDA, max |Δlogit| = {max_abs} (mean {mean_abs}), \
                 max |logit| = {max_logit}, greedy {:?}",
                gd
            );
            // Same comparison class as #141: CPU and CUDA reduce in different orders
            // and run different exp/softmax kernels (graph rule §9), so the paths are
            // not bit-equal by design. Qwen3's spread is ~100× the 0.5B f16 gate's
            // (measured: max |Δlogit| 1.7e-2 / relative 5e-4 here against f141's
            // 7.34e-5 / 4.0e-6) because Qwen3 runs **four** norms per layer
            // (attn_norm + per-head q_norm/k_norm + ffn_norm), and the CPU's rms_norm
            // (8-lane AVX2 FMA plus an f64 tail, then `1/sqrt`) and the device's
            // (warp-shuffle f32 reduction, then `rsqrtf`) differ in both reduction
            // order and reciprocal-sqrt form — an order-of-accumulation difference that
            // compounds through 28 layers, not a weight fault. The bounds are stated
            // here and printed above; a wrong f16 row moves logits by O(1), which the
            // greedy continuation would also catch.
            assert!(max_abs <= 0.05, "max |Δlogit| = {max_abs}");
            assert!(
                max_abs / max_logit.max(1.0) <= 1e-3,
                "max relative Δlogit = {}",
                max_abs / max_logit
            );
        }
    }

    /// #167 acceptance: the **qwen3** loader registers a `W_dsc` plane for exactly its
    /// admissible q4_K weights — the set the GGUF index predicts, by name and count, not
    /// a non-zero count — and the NB-BT kernel's own lookup (the raw weight pointer in
    /// `CudaState::q4k_dsc`) finds each one.
    ///
    /// The expected set is restated here **from the GGUF index**, independently of
    /// `models::weight_reg`/`q4k_dsc`: a gate that derived `want` from the predicate under
    /// test would stay green while both sides were wrong.
    ///
    /// The negative arm loads the cached Qwen3-Q8_0 file under its own namespace and
    /// requires zero planes, so an "always register" mutation cannot pass.
    #[test]
    #[ignore = "requires a q4_K Qwen3 GGUF and a CUDA device (#167)"]
    fn f167_qwen3_q4k_registers_the_dsc_plane_exactly() {
        #[cfg(not(feature = "cuda"))]
        eprintln!("not a CUDA build; #167's q4_K Qwen3 gate is a no-op here");
        #[cfg(feature = "cuda")]
        {
            use crate::gguf::GgmlType;
            // `get()` is None until something initializes the process-wide state, and a
            // gate that skipped here would pass while measuring nothing (this gate is
            // runnable on its own, not only after the f16 arm warmed the device).
            crate::cuda::CudaState::init();
            let Some(state) = crate::cuda::CudaState::get() else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            let Some(path) = env_path("MINFER_F167_Q4K_GGUF", "/tmp/f167-work/qwen3-q4k.gguf")
            else {
                return;
            };
            let ns = "f167q4k:";
            let neg_ns = "f167neg:";
            let _model_load_guard = crate::cuda::CudaState::model_load_guard();
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse q4_K GGUF");
            assert_eq!(
                gguf.parts[0]
                    .ctx
                    .get_key_val_str("general.architecture")
                    .as_deref(),
                Some("qwen3"),
                "#167's q4_K gate requires a Qwen3 GGUF"
            );

            // The expected plane set, from the index: q4_K, a whole number of 256-element
            // super-blocks per row, an even row count (the r59 row-pair staging), and a
            // payload of exactly od*(id/256)*144 bytes. Types are counted so the gate can
            // state that the file really exercises the type gate.
            let mut want: Vec<String> = Vec::new();
            let mut n_q4k = 0usize;
            let mut n_other_quant = 0usize;
            for part in &gguf.parts {
                for ti in &part.ctx.info {
                    if ti.type_ == GgmlType::Q4_K {
                        n_q4k += 1;
                    } else if ti.type_ != GgmlType::F32 && ti.ne[1] > 1 {
                        n_other_quant += 1;
                    }
                    if ti.type_ != GgmlType::Q4_K {
                        continue;
                    }
                    let (id, od) = (ti.ne[0] as usize, ti.ne[1] as usize);
                    if id == 0 || id % 256 != 0 || od == 0 || od % 2 != 0 {
                        continue;
                    }
                    if ti.nbytes() != od * (id / 256) * 144 {
                        continue;
                    }
                    want.push(format!("{ns}{}__q4dsc{od}x{id}", ti.name));
                }
            }
            want.sort();
            assert!(
                !want.is_empty(),
                "the file has no admissible q4_K weight to gate on ({n_q4k} q4_K tensors)"
            );
            assert!(
                n_other_quant > 0,
                "the file has no non-q4_K quantized 2-D weight, so the type gate is untested"
            );

            let model = crate::models::load_model_with(
                &gguf,
                ns,
                crate::graph::offload::OffloadRequest::Layers(usize::MAX),
            )
            .expect("load q4_K model");
            assert!(
                matches!(
                    crate::models::ModelDef::device(model.as_ref()),
                    crate::models::Device::Cuda
                ),
                "the q4_K Qwen3 model did not come up on the device"
            );
            drop(model);

            let mut got: Vec<String> = state
                .q4dsc_planes()
                .into_iter()
                .map(|(n, _)| n)
                .filter(|n| n.starts_with(ns))
                .collect();
            got.sort();
            let bytes: usize = state
                .q4dsc_planes()
                .into_iter()
                .filter(|(n, _)| n.starts_with(ns))
                .map(|(_, b)| b)
                .sum();
            eprintln!(
                "f167 q4_K qwen3 planes: {} expected ({} q4_K + {} other quantized 2-D), \
                 {} registered, {bytes} bytes in {}",
                want.len(),
                n_q4k,
                n_other_quant,
                got.len(),
                path.display()
            );
            assert_eq!(
                got, want,
                "the W_dsc plane set must be exactly the model's admissible q4_K weights"
            );
            // The NB-BT kernel's lookup is keyed on the raw weight pointer, not on the
            // plane's name: assert the map the kernel reads, per weight.
            let names: Vec<String> = want
                .iter()
                .map(|n| {
                    n.trim_start_matches(ns)
                        .split("__q4dsc")
                        .next()
                        .unwrap()
                        .to_string()
                })
                .collect();
            for raw in &names {
                let key = format!("{ns}{raw}");
                assert!(
                    state.q4dsc_plane_for(&key).is_some(),
                    "the NB-BT kernel cannot find the plane for {key}"
                );
            }
            // Every plane is non-null and distinct (a name-only set could be aliases).
            let mut ptrs: Vec<*mut std::ffi::c_void> = Vec::new();
            for raw in &names {
                let p = state.q4dsc_plane_for(&format!("{ns}{raw}")).unwrap();
                assert!(!p.is_null(), "null plane for {raw}");
                ptrs.push(p);
            }
            ptrs.sort();
            ptrs.dedup();
            assert_eq!(
                ptrs.len(),
                names.len(),
                "two weights share one plane buffer"
            );

            // Negative arms, each under its own namespace: a q8_0 Qwen3 (whose payload
            // is *longer* than q4_K's, so the payload gate is the only refuser) and a
            // **q4_0** Qwen3 (`minfer quantize --type q4_0` of the same Q8_0 source).
            // q4_0's bytes/element ratio equals q4_K's exactly (18/32 == 144/256), so a
            // q4_0 payload passes the payload contract — the *type* gate is the only
            // thing that can refuse it, and this arm is what makes a type-gate bypass
            // show up in a real-model gate rather than only in the pure test.
            let mut neg_arm = |label: &str, key: &str, default: &str, arm_ns: &str| {
                if let Some(neg) = env_path(key, default) {
                    let ngguf = crate::gguf::load_gguf_model(&neg).expect("parse negative GGUF");
                    assert_eq!(
                        ngguf.parts[0]
                            .ctx
                            .get_key_val_str("general.architecture")
                            .as_deref(),
                        Some("qwen3"),
                        "the {label} negative arm requires a Qwen3 GGUF"
                    );
                    let nq = crate::models::load_model_with(
                        &ngguf,
                        arm_ns,
                        crate::graph::offload::OffloadRequest::Layers(usize::MAX),
                    )
                    .expect("load negative model");
                    drop(nq);
                    let leaked: Vec<String> = state
                        .q4dsc_planes()
                        .into_iter()
                        .map(|(n, _)| n)
                        .filter(|n| n.starts_with(arm_ns))
                        .collect();
                    assert!(
                        leaked.is_empty(),
                        "a {label} Qwen3 model registered q4dsc planes: {leaked:?}"
                    );
                    eprintln!("f167 q4_K gate: {label} negative arm registers 0 planes");
                } else {
                    eprintln!("f167 q4_K gate: no {label} negative arm ({key} absent)");
                }
            };
            neg_arm(
                "q8_0",
                "MINFER_F167_NEG_GGUF",
                "~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
                neg_ns,
            );
            neg_arm(
                "q4_0",
                "MINFER_F167_NEG_Q40_GGUF",
                "/tmp/f167-work/qwen3-q4_0.gguf",
                "f167negq40:",
            );
        }
    }

    /// The Qwen3 twin of [`logits_greedy_on`] (that one is pinned to `Qwen2Model`).
    /// Same forward code on both arms, so the only difference is the backend the
    /// scheduler assigns.
    fn logits_greedy_on_qwen3(
        q: &crate::models::qwen3::Qwen3Model,
        ids: &[u32],
        steps: usize,
        n_ctx: usize,
    ) -> (Vec<f32>, Vec<u32>) {
        let nt = ids.len();
        let mut cache = crate::graph::cache::GraphCache::new();
        let mut positions: Vec<usize> = (0..nt).collect();
        let mut logits = crate::models::qwen3::graph::Qwen3Graph::forward_cached(
            q, ids, &positions, 1, n_ctx, &mut cache,
        );
        let mut toks = Vec::with_capacity(steps);
        for step in 0..steps {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            toks.push(next);
            positions = vec![nt + step];
            logits = crate::models::qwen3::graph::Qwen3Graph::forward_cached(
                q,
                &[next],
                &positions,
                1,
                n_ctx,
                &mut cache,
            );
        }
        let _ = positions;
        (logits, toks)
    }

    /// G3: the split file's merged index is exactly the single-file index and
    /// the logits are bitwise identical after a re-load.
    #[test]
    #[ignore = "requires a cached 0.5B GGUF and writes ~0.5 GB under /tmp/f6-work"]
    fn f6_split_merged_index_and_logits_match_the_single_file() {
        let Some(src) = env_path("MINFER_F6_SPLIT_MODEL", &cached_qwen05().to_string_lossy())
        else {
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

    /// G4b: every weight encoder is byte-identical to llama.cpp's.
    ///
    /// `MINFER_F6_F16_GGUF` is the common f16 source and
    /// `MINFER_F6_LLAMACPP_QUANT` is the file `llama-quantize <src> <out> <type>`
    /// produced for the target named by `MINFER_F6_QUANT_TYPE` (default q4_0).
    /// A single-nibble difference is a wrong file, so this is per-tensor byte
    /// equality (the same comparison used for the HF conversion).
    #[test]
    #[ignore = "requires an f16 GGUF plus a llama-quantize reference of the same target"]
    fn f6_quantize_encoder_is_byte_identical_to_llamacpp() {
        let Some(src) = env_path("MINFER_F6_F16_GGUF", "/tmp/f6-work/ref-f16.gguf") else {
            return;
        };
        let Some(reff) = env_path("MINFER_F6_LLAMACPP_QUANT", "/tmp/f6-work/ref-q4_0.gguf") else {
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
        assert_tensor_payloads_equal(&out, &reff);
        eprintln!(
            "f6 quantize parity: {tname} == llama-quantize on {} tensors",
            plan.specs.len()
        );
    }
}
