//! #724: the `llama3` rope-scaling contract — where the per-pair frequency
//! factors come from, and what an unknown scaling type does.
//!
//! Two GGUF shapes carry the same information, and this module reads both:
//!
//! 1. a `rope_freqs.weight` tensor — f32, `hd/2` entries — which is what
//!    llama.cpp's converter writes (`conversion/llama.py`,
//!    `LlamaModel.generate_extra_tensors`, build b11829 / `184f65360`) and reads
//!    back at every position (`theta_base / freq_factor`). This is the
//!    authoritative shape: the values are the ones the reference engine applies.
//!    Both real files measured for #724 spell it as **one global tensor**
//!    (`rope_freqs.weight`, no `blk.N.` prefix), because llama.cpp loads it once
//!    and hands the same table to every layer; the per-block spelling
//!    (`blk.N.rope_freqs.weight`) is accepted too, since that is the shape the
//!    same loader family uses for a per-layer table.
//! 2. `{arch}.rope.scaling.type = "llama3"` metadata with a `factor`, an
//!    `original_context_length` and the two band factors. llama.cpp's gguf-py
//!    does not know this type (its `RopeScalingType` enum is
//!    `none`/`linear`/`yarn`/`longrope`), so a file carrying it came from another
//!    converter or was written by hand; the table is then synthesised with the
//!    converter's own formula ([`crate::vec_ops::llama3_freqs`]).
//!
//! The tensor wins when both are present — it is what the reference reads.
//!
//! **An unknown type is refused by name.** So are `linear`, `yarn` and
//! `longrope`: this engine implements none of them, and treating one as `none`
//! (or as the unscaled base) applies the wrong rope at every position, which
//! reads as a numerics bug far from its cause.
//!
//! # The upstream name mismatch, and why a global table is *not* read
//!
//! Measured on 2026-11-05 against llama.cpp build b11829 (`184f65360`): its
//! converter writes the table as **one global tensor named `rope_freqs.weight`**
//! (`conversion/llama.py` yields `format_tensor_name(ROPE_FREQS)` with no block
//! id), while its loader asks for the **per-block** name (the arch table declares
//! `LLM_TENSOR_ROPE_FREQS` as a layer-repeating tensor). Both real files in the
//! cache carry only the global spelling: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` has
//! `rope_freqs.weight` `[32]` with the factor-32 table, `Llama-3.1-8B-Instruct-Q4_K_M.gguf`
//! has `[64]` with the factor-8 table. The reference engine therefore applies
//! **no** scaling to either file — confirmed twice: build 10665 (which recorded
//! #725's pin) and build 11829 both decode `The capital of France is` to the
//! unscaled ids, while applying the table turns that same run into the seven-token
//! cycle the pin rejects.
//!
//! So the global spelling is deliberately **ignored** here: this engine reads what
//! the reference reads, and parity on the cached files is what the milestone's
//! Llama claim rests on. A per-block table (the spelling the loader resolves) and
//! the metadata spelling are both honoured — and the differential gate patches a
//! real file into the per-block shape to check the arithmetic against the
//! reference at 8k.

use std::collections::HashMap;

use crate::gguf::{GgmlType, GgufContext, GgufPart, GgufTensorInfo};
use crate::vec_ops::{llama3_freqs, RopeFreqs};

/// The `rope.scaling.type` values this engine implements, for the refusal message.
const SUPPORTED: &str = "none, llama3";

/// Read the metadata shape. `Ok(None)` when the file declares no scaling at all.
/// `freq_base`/`hd` come from the loader's own hyperparameters, so the metadata
/// path and the kernels agree on the base by construction.
pub fn from_metadata(
    ctx: &GgufContext,
    arch: &str,
    freq_base: f32,
    hd: usize,
) -> Result<Option<RopeFreqs>, String> {
    let Some(kind) = scaling_key(ctx, arch, "rope.scaling.type") else {
        return Ok(None);
    };
    match kind.to_ascii_lowercase().as_str() {
        "" | "none" => Ok(None),
        "llama3" => {
            // The converter's defaults, verbatim: factor 8, original context 8192,
            // band 1..4 (`conversion/llama.py`).
            let factor = scaling_f32(ctx, arch, "rope.scaling.factor").unwrap_or(8.0);
            let orig_ctx =
                scaling_f32(ctx, arch, "rope.scaling.original_context_length").unwrap_or(8192.0);
            let low = scaling_f32(ctx, arch, "rope.scaling.low_freq_factor").unwrap_or(1.0);
            let high = scaling_f32(ctx, arch, "rope.scaling.high_freq_factor").unwrap_or(4.0);
            let table = llama3_freqs(freq_base, factor, orig_ctx, low, high, hd)
                .map_err(|e| format!("{arch}.rope.scaling.type = \"llama3\": {e}"))?;
            Ok(Some(table))
        }
        other => Err(format!(
            "{arch}.rope.scaling.type = {other:?} is not supported by this engine \
             (supported: {SUPPORTED}; linear/yarn/longrope are not implemented, so accepting \
             it would apply the wrong rope at every position)"
        )),
    }
}

/// Adopt a `blk.N.rope_freqs.weight` table. `Ok(None)` for an absent tensor.
pub fn from_tensor(table: &[f32], hd: usize) -> Result<Option<RopeFreqs>, String> {
    if table.is_empty() {
        return Ok(None);
    }
    if table.len() != hd / 2 {
        return Err(format!(
            "blk.*.rope_freqs.weight has {} entries but head_dim {hd} needs {}",
            table.len(),
            hd / 2
        ));
    }
    RopeFreqs::from_slice(table).map(Some).ok_or_else(|| {
        format!(
            "blk.*.rope_freqs.weight: {} entries exceed this build's table limit",
            table.len()
        )
    })
}

/// `{arch}.{suffix}`, falling back to `llama.{suffix}` (the namespace the
/// llama-family files use even when `general.architecture` says something else).
fn scaling_key(ctx: &GgufContext, arch: &str, suffix: &str) -> Option<String> {
    let primary = format!("{arch}.{suffix}");
    ctx.get_key_val_str(&primary)
        .or_else(|| ctx.get_key_val_str(&format!("llama.{suffix}")))
}

fn scaling_f32(ctx: &GgufContext, arch: &str, suffix: &str) -> Option<f32> {
    let primary = format!("{arch}.{suffix}");
    ctx.get_key_val_f32(&primary)
        .or_else(|| ctx.get_key_val_f32(&format!("llama.{suffix}")))
}

/// #724: resolve one file's rope-scaling table.
///
/// The tensor shape wins; the metadata shape is the fallback. Both loaders call
/// this (the shared Qwen2/Llama loader and Qwen3), so the contract — which shape
/// wins, what an unknown type does, how much a file may disagree with itself —
/// lives in one place.
pub fn resolve(
    parts: &[GgufPart],
    tensor_map: &HashMap<String, (usize, &GgufTensorInfo)>,
    arch: &str,
    hd: usize,
    n_layer: usize,
    freq_base: f32,
    name_of: fn(usize) -> String,
) -> Result<Option<RopeFreqs>, String> {
    // One global table first — the spelling both real llama.cpp conversions use
    // (and the one its loader resolves) — then a per-block one.
    if let Some((pi, ti)) = tensor_map.get("rope_freqs.weight") {
        let raw = raw_f32_table(&parts[*pi], ti)?;
        if let Some(table) = from_tensor(&raw, hd)? {
            return Ok(Some(table));
        }
    }
    let mut table: Option<RopeFreqs> = None;
    let mut first: Option<String> = None;
    for layer in 0..n_layer {
        let name = name_of(layer);
        let Some((pi, ti)) = tensor_map.get(&name) else {
            continue;
        };
        let raw = raw_f32_table(&parts[*pi], ti)?;
        let got = from_tensor(&raw, hd)?.ok_or_else(|| format!("{name}: empty rope table"))?;
        match &table {
            Some(prev) if *prev != got => {
                return Err(format!(
                    "{name}: this block's rope table differs from {}; this engine applies one \
                     rope table per model",
                    first.as_deref().unwrap_or("the first block's")
                ));
            }
            Some(_) => {}
            None => {
                table = Some(got);
                first = Some(name.clone());
            }
        }
    }
    if table.is_none() {
        table = from_metadata(&parts[0].ctx, arch, freq_base, hd)?;
    }
    Ok(table)
}

/// Read a tensor's f32 payload straight out of the mmap.
///
/// A rope table is not a matmul weight: going through the loader's `load_one`
/// would register it with every device and charge its bytes to the offload plan,
/// and no kernel resolves that name.
fn raw_f32_table(part: &GgufPart, ti: &GgufTensorInfo) -> Result<Vec<f32>, String> {
    if ti.type_ != GgmlType::F32 {
        return Err(format!(
            "{}: the rope factor table must be f32, found {:?}",
            ti.name, ti.type_
        ));
    }
    let off = part.ctx.offset + ti.offset as usize;
    let n = ti.ne[0].max(0) as usize;
    let end = off + n * 4;
    let bytes = part.data().get(off..end).ok_or_else(|| {
        format!(
            "{}: {} bytes at offset {} run past the part's {}",
            ti.name,
            n * 4,
            off,
            part.data().len()
        )
    })?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
