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
        eprintln!(
            "quantize: {} tensor(s) keep their source type (1-D or row length not a multiple of \
             {}): {}",
            plan.preserved.len(),
            target.blck_size(),
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
        let ids = tok.encode(prompt);
        let nt = ids.len();
        let mut cache = crate::graph::cache::GraphCache::new();
        let mut positions: Vec<usize> = (0..nt).collect();
        let mut logits = crate::models::qwen2::graph::Qwen2Graph::forward_cached(
            q, &ids, &positions, 1, n_ctx, &mut cache,
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
