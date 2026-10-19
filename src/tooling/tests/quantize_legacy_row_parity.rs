//! #204's byte arm: on a source with a 2-D row that no encoder block divides,
//! `minfer quantize` must be **byte-identical per tensor** to `llama-quantize`,
//! the demoted tensor included.
//!
//! No supported architecture has such a row (every Qwen2 / Qwen3 2-D row in the
//! F6 cache is a multiple of 32), so the input is **derived** from a recorded
//! fixture rather than recorded itself: one tensor's `ne[0]` is shrunk and every
//! later tensor's data offset is shifted by the same byte delta, which keeps the
//! data section contiguous (both loaders reject a gap — measured: a first attempt
//! that only rewrote `ne[0]` was refused by *both* tools with
//! `has offset N, expected M`).
//!
//! The derivation is a pure function of the parent fixture, and its output
//! digest is pinned below; the recorded references were produced from exactly
//! those bytes. So a derivation that drifts fails on the digest — naming the
//! file, not the encoder — instead of being blamed on the code under test.
//!
//! `#[ignore]`d: it needs the 1.9 GB f32 fixture plus two recorded
//! `llama-quantize` references. Recipe: `docs/GGUF-TOOLING.md` §4.2.4.

use super::*;

use crate::quantize::QuantTarget;

use std::fs::OpenOptions;
use std::io::{BufReader, Read, Seek, SeekFrom};

use sha2::{Digest, Sha256};

/// The derived source is defined by one edit and pinned by its digest. A
/// changed edit or a changed parent changes the digest, and the references
/// below no longer describe the file — which is the point.
const EDIT_TENSOR: &str = "blk.0.ffn_down.weight";
/// 4864 = 19 × 256 (the K block) = 152 × 32 (the legacy block); 4863 is a
/// multiple of neither, so a legacy target takes its F16 endpoint *and* a
/// K-quant takes the second step of its chain (Q6_K → Q8_0 → F16).
const EDIT_NE0: i64 = 4863;
/// sha256 of the derived file (parent + edit), measured 2026-10-13 on dgxspark.
const DERIVED_SHA256: &str = "fbd63b28d16347fe385514c1b94fab4cfcdd0395167652efb7eb7dec46d305ef";

/// One tensor-index entry as the derivation needs it: the file offsets of the
/// fields it rewrites, and the data offset it shifts.
struct IndexEntry {
    name: String,
    ne: [i64; 4],
    ne_field: u64,
    offset_field: u64,
    offset: u64,
    type_id: u32,
}

fn rd_u32(f: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn rd_u64(f: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn rd_str(f: &mut impl Read) -> std::io::Result<String> {
    let n = rd_u64(f)? as usize;
    let mut v = vec![0u8; n];
    f.read_exact(&mut v)?;
    Ok(String::from_utf8_lossy(&v).into_owned())
}

/// Skip one GGUF metadata value. Types are the GGUF spec's own numbering.
fn skip_kv(f: &mut impl Read, t: u32) -> std::io::Result<()> {
    let n = match t {
        0 | 1 | 7 => 1u64,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        8 => {
            rd_str(f)?;
            0
        }
        9 => {
            let et = rd_u32(f)?;
            let n = rd_u64(f)?;
            for _ in 0..n {
                skip_kv(f, et)?;
            }
            0
        }
        other => {
            return Err(std::io::Error::other(format!(
                "GGUF metadata type {other} is not in the spec"
            )))
        }
    };
    if n > 0 {
        std::io::copy(&mut f.take(n), &mut std::io::sink())?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Write `src` to `dst` with `tensor`'s `ne[0]` set to `new_ne0`, shifting the
/// data offset of every tensor that follows it so the file stays contiguous.
/// Returns the byte delta.
///
/// The tensor must be a 2-D f32 one, which is what makes the delta exact
/// (`rows × (old - new) × 4`); anything else is refused rather than guessed at.
fn derive_unaligned_row(src: &Path, dst: &Path, tensor: &str, new_ne0: i64) -> Result<u64, String> {
    let e = |err: std::io::Error| format!("{}: {err}", src.display());
    let mut f = BufReader::new(std::fs::File::open(src).map_err(e)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(e)?;
    if &magic != b"GGUF" {
        return Err(format!("{} is not a GGUF file", src.display()));
    }
    let version = rd_u32(&mut f).map_err(e)?;
    let n_tensors = rd_u64(&mut f).map_err(e)?;
    let n_kv = rd_u64(&mut f).map_err(e)?;
    if version != 3 {
        return Err(format!("{} is GGUF v{version}, not v3", src.display()));
    }
    for _ in 0..n_kv {
        rd_str(&mut f).map_err(e)?;
        let t = rd_u32(&mut f).map_err(e)?;
        skip_kv(&mut f, t).map_err(e)?;
    }
    let mut entries = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = rd_str(&mut f).map_err(e)?;
        let nd = rd_u32(&mut f).map_err(e)?;
        if nd > 4 {
            return Err(format!("{name}: {nd} dimensions"));
        }
        let ne_field = f.stream_position().map_err(e)?;
        let mut ne = [1i64; 4];
        for j in 0..nd as usize {
            ne[j] = rd_u64(&mut f).map_err(e)? as i64;
        }
        let type_id = rd_u32(&mut f).map_err(e)?;
        let offset_field = f.stream_position().map_err(e)?;
        let offset = rd_u64(&mut f).map_err(e)?;
        entries.push(IndexEntry {
            name,
            ne,
            ne_field,
            offset_field,
            offset,
            type_id,
        });
    }

    let hit: Vec<&IndexEntry> = entries.iter().filter(|x| x.name == tensor).collect();
    let [target] = hit[..] else {
        return Err(format!("{tensor}: {} matching tensors", hit.len()));
    };
    // GGML_TYPE_F32 is 0, and the edit's byte arithmetic is only exact for a
    // 2-D f32 tensor, so refuse anything else instead of deriving a
    // differently-sized file the reference cannot describe.
    if target.type_id != 0 || target.ne[2] != 1 || target.ne[3] != 1 {
        return Err(format!(
            "{tensor}: expected a 2-D f32 tensor, found type id {} with ne {:?}",
            target.type_id, target.ne
        ));
    }
    if new_ne0 <= 0 || new_ne0 >= target.ne[0] {
        return Err(format!(
            "{tensor}: ne[0] {new_ne0} is not a shrink of {}",
            target.ne[0]
        ));
    }
    let delta = (target.ne[1] * (target.ne[0] - new_ne0) * 4) as u64;
    if delta % 32 != 0 {
        return Err(format!(
            "{tensor}: a {delta}-byte shift would break the 32-byte data alignment"
        ));
    }

    std::fs::copy(src, dst)
        .map_err(|err| format!("{} -> {}: {err}", src.display(), dst.display()))?;
    let mut g = OpenOptions::new().write(true).open(dst).map_err(e)?;
    g.seek(SeekFrom::Start(target.ne_field)).map_err(e)?;
    g.write_all(&new_ne0.to_le_bytes()).map_err(e)?;
    for o in &entries {
        if o.offset > target.offset {
            g.seek(SeekFrom::Start(o.offset_field)).map_err(e)?;
            g.write_all(&(o.offset - delta).to_le_bytes()).map_err(e)?;
        }
    }
    g.flush().map_err(e)?;
    Ok(delta)
}

/// The type a file declares for `name`.
fn declared_type(model: &crate::gguf::GgufModel, name: &str) -> crate::gguf::GgmlType {
    model.parts[0]
        .ctx
        .info
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tensor {name} missing"))
        .type_
}

/// #204: a legacy target and the `q6_K` chain must both reproduce
/// `llama-quantize` byte for byte on a source whose edited row is representable
/// to no block size.
///
/// The two arms are the two endpoints the issue named and the one it did not:
/// `q4_0`'s F16 demotion (the reported divergence) and `q6_K`'s second step
/// (`Q6_K → Q8_0 → F16`, which no existing fixture reaches — every real
/// non-256-aligned row is still a multiple of 32).
#[test]
#[ignore = "requires the f32 GGUF from ~/.cache/minfer/f6-src/ plus two recorded llama-quantize references (docs/GGUF-TOOLING.md §4.2.4)"]
fn f6_a_row_no_block_divides_is_demoted_like_llamacpp() {
    let Some(src) = env_path(
        "MINFER_F204_F32_GGUF",
        "~/.cache/minfer/f6-src/qwen2.5-0.5b-instruct-f32.gguf",
    ) else {
        return;
    };
    let refs = [
        (
            QuantTarget::Q4_0,
            "MINFER_F204_LLAMACPP_Q4_0",
            "~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-q4_0-row4863.gguf",
            // The legacy endpoint: no smaller block exists, so F16.
            crate::gguf::GgmlType::F16,
        ),
        (
            QuantTarget::Q6_K,
            "MINFER_F204_LLAMACPP_Q6_K",
            "~/.cache/minfer/f6-src/ref/qwen2.5-0.5b-q6_K-row4863.gguf",
            // The K-quant chain's second step: Q6_K → q8_0 (32 does not divide
            // 4863 either) → F16.
            crate::gguf::GgmlType::F16,
        ),
    ];

    let dir = work_dir("quantize-legacy-row");
    let derived = dir.join("src-row4863.gguf");
    let _ = std::fs::remove_file(&derived);
    let delta = derive_unaligned_row(&src, &derived, EDIT_TENSOR, EDIT_NE0)
        .unwrap_or_else(|e| panic!("deriving the unaligned-row source: {e}"));
    let got = sha256_file(&derived).expect("hash the derived source");
    assert_eq!(
        got,
        DERIVED_SHA256,
        "the derived source is not the one the recorded references were produced from \
         (parent {}, edit {} ne[0] = {EDIT_NE0}, {delta}-byte shift)",
        src.display(),
        EDIT_TENSOR
    );
    let model = crate::gguf::load_gguf_model(&derived).expect("the derived source parses");

    for (target, key, default, want_demoted) in refs {
        let Some(reff) = env_path(key, default) else {
            return;
        };
        let out = dir.join(format!("out-{}.gguf", target.name()));
        let _ = std::fs::remove_file(&out);
        let plan = QuantizePlan::plan(&model, target).expect("plan");
        assert!(
            plan.demoted
                .iter()
                .any(|d| d.starts_with(&format!("{EDIT_TENSOR} ->"))),
            "{}: {EDIT_TENSOR} must be reported demoted, got {:?}",
            target.name(),
            plan.demoted
        );
        plan.write_single(&model, &out).expect("write");
        // The type rule first: it names the tensor, so a type bug cannot hide
        // behind a byte diff (and vice versa).
        let written = crate::gguf::load_gguf_model(&out).expect("output parses");
        assert_eq!(
            declared_type(&written, EDIT_TENSOR),
            want_demoted,
            "{}: {EDIT_TENSOR} must be written {}",
            target.name(),
            want_demoted.type_name()
        );
        match tensor_payloads_equal(&out, &reff) {
            Ok(n) => eprintln!(
                "f6 legacy-row parity: {} == llama-quantize on {n} tensors \
                 [{EDIT_TENSOR} ne[0] = {EDIT_NE0}, {delta} B shift, {} demoted] \
                 [reference {}]",
                target.name(),
                plan.demoted.len(),
                reff.display()
            ),
            Err(primary) => {
                // Not necessarily an encoder defect: the claim is about one
                // build of `llama-quantize` (ADR-0020), so ask the same question
                // the recorded parity gate asks (#349) — does the *uncontracted*
                // model reproduce it, and is this reference a recorded foreign
                // build? A Mac asked to compare against a dgxspark reference
                // reaches this path with the K-quant search encoders, which are
                // compiler-sensitive (docs/GGUF-TOOLING.md §4.2.1).
                let alt = dir.join(format!("{}-uncontracted.gguf", target.name()));
                let _ = std::fs::remove_file(&alt);
                let off_matches = plan
                    .write_single_with_contract(&model, &alt, crate::quantize::FmaContract::Off)
                    .is_ok()
                    && tensor_payloads_equal(&alt, &reff).is_ok();
                let record = f6_fixtures::reference_record(&reff);
                match f6_fixtures::ParityVerdict::classify(false, off_matches, &record) {
                    f6_fixtures::ParityVerdict::Reproduces => {
                        unreachable!("the `Fast` comparison already failed")
                    }
                    f6_fixtures::ParityVerdict::FlagMismatch => panic!(
                        "f6 legacy-row parity: {}: {} is a build without cross-statement FMA \
                         contraction (`-ffp-contract=off`), not the documented `fast` build; the \
                         two differ at exact rounding boundaries (docs/GGUF-TOOLING.md §4.2). \
                         {primary}",
                        target.name(),
                        reff.display()
                    ),
                    f6_fixtures::ParityVerdict::RecordedForeignBuild => eprintln!(
                        "[f6 legacy-row parity] SKIP {}: {} matches neither minfer's \
                         `FmaContract::Fast` model nor its uncontracted model ({primary}), and \
                         the manifest records its content as a different build of the same source, \
                         not the authoritative reference — a compiler difference, not an encoder \
                         defect (issue #349). Parity here is asserted only against the recorded \
                         compiler.",
                        target.name(),
                        reff.display()
                    ),
                    f6_fixtures::ParityVerdict::AuthoritativeNotReproduced => panic!(
                        "f6 legacy-row parity: {}: {primary} — and {} IS the authoritative \
                         recorded content for this path, so the encoder no longer reproduces the \
                         reference this claim is asserted against (issue #349)",
                        target.name(),
                        reff.display()
                    ),
                    f6_fixtures::ParityVerdict::Unattributable => panic!(
                        "f6 legacy-row parity: {}: {primary}, and {} matches no recorded content in \
                         tests/fixtures/f6-fixtures.json — a corrupted or replaced reference looks \
                         exactly like this, so nothing may be blamed on a compiler (issue #349)",
                        target.name(),
                        reff.display()
                    ),
                }
            }
        }
    }
}
