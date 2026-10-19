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
use crate::quantize::Target;

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
        "Usage: {prog} convert <hf-model-dir> <out.gguf> [--outtype f16|bf16|f32] [--split-max-size BYTES]\n\
         \n\
         Converts a HuggingFace Qwen2 checkpoint (config.json + tokenizer.json +\n\
         tokenizer_config.json + model.safetensors) into GGUF v3. Supported\n\
         architectures: Qwen2ForCausalLM only; output types: f16 (default), bf16, f32."
    )
}

fn usage_quantize(prog: &str) -> String {
    format!(
        "Usage: {prog} quantize <in.gguf> <out.gguf> --type <target> [--split-max-size BYTES]\n\
         \n\
         Re-encodes a single-file GGUF's 2-D float weights to <target>. One-dimensional\n\
         tensors (norms, biases) keep their source type.\n\
         \n\
         Two kinds of target. A concrete type puts one encoder on every 2-D tensor:\n\
         q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K, q6_K, and the f16/f32 element\n\
         casts. That is what `llama-quantize --pure` writes, and here q4_K/q5_K are\n\
         UNIFORM — llama.cpp's bare `q4_K`/`q5_K` mean the mixture, so those spellings\n\
         are `q4_K_M`/`q5_K_M` here. A mixture is llama.cpp's per-tensor policy: the\n\
         `output.weight` (or a tied `token_embd.weight`) goes to q6_K, or q8_0 when\n\
         the row cannot take the requested type's block, and use_more_bits bumps\n\
         `attn_v` and `ffn_down` on the first and last eighth of the layers plus every\n\
         third layer between them (the fused-qkv category too).\n\
         \n\
         A 2-D row that cannot take the type a tensor was asked for is demoted the way\n\
         llama.cpp's tensor_type_fallback does it: a K-quant (256-element block) to\n\
         q5_0/q5_1/q8_0, a legacy target (32-element block) to f16, and f16 either way\n\
         when even a 32-element block does not divide the row.\n\
         \n\
         Supported targets: {}.",
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
    let mut target: Option<Target> = None;
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
                match Target::parse(v) {
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
    if target.is_mixture() {
        eprintln!(
            "quantize: {} is llama.cpp's per-tensor mixture policy; the uniform type is spelled \
             `{}` (what `llama-quantize --pure` writes)",
            target.name(),
            target.base().name()
        );
    }
    let plan = match QuantizePlan::plan(&model, target) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    if !plan.preserved.is_empty() {
        // Only 1-D tensors are ever preserved: every quant target demotes a
        // non-block-aligned 2-D row instead of copying it (see `plan.demoted`),
        // f16's block size is 1, and f32 converts everything. #169, #140, #204.
        eprintln!(
            "quantize: {} tensor(s) keep their source type (1-D norms/biases): {}",
            plan.preserved.len(),
            plan.preserved.join(", ")
        );
    }
    if !plan.retargeted.is_empty() {
        eprintln!(
            "quantize: {} tensor(s) through llama.cpp's OUTPUT/tied-embedding branch: {}",
            plan.retargeted.len(),
            plan.retargeted.join(", ")
        );
    }
    if !plan.promoted.is_empty() {
        eprintln!(
            "quantize: {} tensor(s) bumped by {} on llama.cpp's per-tensor policy: {}",
            plan.promoted.len(),
            target.name(),
            plan.promoted.join(", ")
        );
    }
    if !plan.demoted.is_empty() {
        eprintln!(
            "quantize: {} tensor(s) demoted to a type their row can take (llama.cpp's \
             tensor_type_fallback): {}",
            plan.demoted.len(),
            plan.demoted.join(", ")
        );
    }
    // How much of the file each encoder actually saw. Printed for every target:
    // on a mixture it is the only compact summary of the per-tensor policy, and
    // on a uniform target it is the receipt that a demotion happened.
    let hist = plan
        .by_type
        .iter()
        .map(|(t, c)| format!("{}: {c}", t.type_name()))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("quantize: file types: {hist}");
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
mod tests;
