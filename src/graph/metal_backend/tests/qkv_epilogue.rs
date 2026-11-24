//! D3-8 class 2 ([#844]): Metal's mixed-quant decode QKV epilogue
//! (`Op::QkvBiasRopeStore`) dispatches `kernel_attn_bias_rope_store` **once per
//! section** (q, k, v) over the three separate projection buffers, because the
//! graph has no concat `[q | k | v]` buffer for a mixed-quant layer and a
//! mid-split blit is not available (it closes the compute pass). The params that
//! select a section (`attn_bias_rope_store_section`'s table: `(nqt, nkt, grid)`
//! = `(nqt, 0, nqt/2)`, `(0, nkt, nkt/2)`, `(-nkt, nkt, nkt)`) are load-bearing,
//! so this is the gate that pins them.
//!
//! It is a **bit-equality** gate, and it compares against the *fused* dispatch
//! itself, not a reimplementation: the same three projections are fed to
//! `attn_bias_rope_store` as one concat buffer (the `Op::FusedQKV` shape) and to
//! the three section dispatches, then every output is compared bit for bit —
//! q, the rope-rotated k row, the biased v row, and the KV cell the epilogue
//! wrote (f32, f16 and packed `q8_0` regions). Passing it means a per-section
//! dispatch is the concat dispatch's own arithmetic; concatenation is the only
//! thing the two paths do differently.
//!
//! The exact **thread count** is part of the contract, not an implementation
//! detail: the K section's `total` (`nqt/2 + nkt/2 + nkt == nkt*3/2`) is larger
//! than its pair count, so a grid rounded up to a whole 256-thread group would
//! run the V arm over memory past the k buffer. The section wrapper therefore
//! uses `dispatchThreads`, and this gate is what catches a regression to the
//! rounded `dispatch_1d` (it showed up here as a double bias add on v).
//!
//! [#844]: https://github.com/yusiwen/minfer/issues/844

use super::*;
use crate::graph::kvformat::KvFormat;

const NH: usize = 2;
const NK: usize = 2;
const HD: usize = 64;
const NQT: usize = NH * HD;
const NKT: usize = NK * HD;
const N_CTX: usize = 8;
/// The decode position under test: non-zero, so a wrong `pos` is visible.
const POS: i32 = 3;
const FREQ_BASE: f32 = 10_000.0;

/// Values with no exact binary pattern (division by 997) and both signs, so a
/// changed rounding or a swapped index shows up.
fn fixture(seed: usize, i: usize) -> f32 {
    let v = ((seed + i * 37) % 997) as f32 / 997.0 - 0.5;
    if seed % 2 == 0 {
        -v
    } else {
        v
    }
}

fn fill_f32(buf: &crate::metal::MetalBuffer, data: &[f32]) {
    assert!(
        buf.length() >= data.len() * 4,
        "buffer of {} bytes cannot hold {} f32",
        buf.length(),
        data.len()
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr() as *mut f32,
            data.len(),
        );
    }
}

fn words(buf: &crate::metal::MetalBuffer, n: usize) -> Vec<u32> {
    assert!(buf.length() >= n * 4, "buffer is shorter than {n} words");
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u32, n).to_vec() }
}

/// How many f32 words of a KV region one cell occupies (f32: one word per
/// element; f16: two elements per word; packed: 34-byte blocks, word-padded).
fn cell_words(f16: bool, packed: bool, row_bytes: usize) -> usize {
    if packed {
        row_bytes / 4
    } else if f16 {
        NKT / 2
    } else {
        NKT
    }
}

/// The addressed cell's words, from a region prefix that covers `POS` + 1 cells.
fn cell_slice(all: &[u32], words_per_cell: usize) -> Vec<u32> {
    let start = POS as usize * words_per_cell;
    all[start..start + words_per_cell].to_vec()
}

/// One `(kv width, packed, rope style)` case, run to completion: the reference
/// through the concat dispatch, the candidate through the three section
/// dispatches, then every byte the kernel produced is compared.
fn section_dispatch_matches_the_concat(f16: bool, packed: bool, rope_style: i32) {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(mut backend) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let st = backend.state;
    let case = format!("f16={f16} packed={packed} rope_style={rope_style}");

    let q_in: Vec<f32> = (0..NQT).map(|i| fixture(9, i)).collect();
    let k_in: Vec<f32> = (0..NKT).map(|i| fixture(11, i)).collect();
    let v_in: Vec<f32> = (0..NKT).map(|i| fixture(13, i)).collect();

    // Read-only inputs, shared by both arms.
    let bias_q = st.new_f32_buffer(NQT);
    let bias_k = st.new_f32_buffer(NKT);
    let bias_v = st.new_f32_buffer(NKT);
    fill_f32(
        &bias_q,
        &(0..NQT).map(|i| fixture(21, i)).collect::<Vec<_>>(),
    );
    fill_f32(
        &bias_k,
        &(0..NKT).map(|i| fixture(23, i)).collect::<Vec<_>>(),
    );
    fill_f32(
        &bias_v,
        &(0..NKT).map(|i| fixture(25, i)).collect::<Vec<_>>(),
    );
    let positions = st.new_f32_buffer(1);
    // `fill_input_i32` stores the integer's bits in an f32 word, and the kernel
    // reads a `device const int *` — the bytes are the same either way.
    fill_f32(&positions, &[f32::from_bits(POS as u32)]);

    // An arena of `nkt * n_ctx` f32 words holds an f32 region, an f16 one (half
    // a word per element) and a packed one (34 bytes per 32 elements) alike.
    let region = NKT * N_CTX;
    let row_bytes = KvFormat::Q8_0.row_bytes(NKT);
    let per_cell = cell_words(f16, packed, row_bytes);
    let touched = per_cell * (POS as usize + 1);

    // ── reference: one concat buffer, one epilogue dispatch (the FusedQKV shape)
    let concat = st.new_f32_buffer(NQT + 2 * NKT);
    let mut joined = Vec::with_capacity(NQT + 2 * NKT);
    joined.extend_from_slice(&q_in);
    joined.extend_from_slice(&k_in);
    joined.extend_from_slice(&v_in);
    fill_f32(&concat, &joined);
    let kv_k_ref = st.new_f32_buffer(region);
    let kv_v_ref = st.new_f32_buffer(region);
    {
        let cb = backend.cb();
        cb.attn_bias_rope_store(
            &concat, &bias_q, 0, &bias_k, 0, &bias_v, 0, &kv_k_ref, &kv_v_ref, NQT, NKT, HD,
            FREQ_BASE, 1.0, None, POS, rope_style, f16, packed,
        );
        if packed {
            cb.store_kv_q8_0(&concat, &kv_k_ref, NKT, 1, &positions, NQT, row_bytes);
            cb.store_kv_q8_0(&concat, &kv_v_ref, NKT, 1, &positions, NQT + NKT, row_bytes);
        }
    }
    backend.submit_pending();

    // ── candidate: three section buffers, the three dispatch offsets the op's
    //    arm uses (a packed cache adds the two `store_kv_q8_0` calls at offset 0)
    let q_sec = st.new_f32_buffer(NQT);
    let k_sec = st.new_f32_buffer(NKT);
    let v_sec = st.new_f32_buffer(NKT);
    fill_f32(&q_sec, &q_in);
    fill_f32(&k_sec, &k_in);
    fill_f32(&v_sec, &v_in);
    let kv_k_sec = st.new_f32_buffer(region);
    let kv_v_sec = st.new_f32_buffer(region);
    {
        let cb = backend.cb();
        // Q: bias + rope in place, over q's own buffer.
        cb.attn_bias_rope_store_section(
            &q_sec,
            &bias_q,
            0,
            &bias_k,
            0,
            &bias_v,
            0,
            &kv_k_sec,
            &kv_v_sec,
            NQT as i32,
            0,
            (NQT / 2) as u64,
            HD,
            FREQ_BASE,
            1.0,
            None,
            POS,
            rope_style,
            f16,
            packed,
        );
        // K: bias + rope + store, over k's own buffer.
        cb.attn_bias_rope_store_section(
            &k_sec,
            &bias_q,
            0,
            &bias_k,
            0,
            &bias_v,
            0,
            &kv_k_sec,
            &kv_v_sec,
            0,
            NKT as i32,
            (NKT / 2) as u64,
            HD,
            FREQ_BASE,
            1.0,
            None,
            POS,
            rope_style,
            f16,
            packed,
        );
        if packed {
            cb.store_kv_q8_0(&k_sec, &kv_k_sec, NKT, 1, &positions, 0, row_bytes);
        }
        // V: bias + store, no rope, over v's own buffer. `nqt = -nkt` is the
        // param that puts the kernel's `vi = nqt + nkt + j` back at `j`.
        cb.attn_bias_rope_store_section(
            &v_sec,
            &bias_q,
            0,
            &bias_k,
            0,
            &bias_v,
            0,
            &kv_k_sec,
            &kv_v_sec,
            -(NKT as i32),
            NKT as i32,
            NKT as u64,
            HD,
            FREQ_BASE,
            1.0,
            None,
            POS,
            rope_style,
            f16,
            packed,
        );
        if packed {
            cb.store_kv_q8_0(&v_sec, &kv_v_sec, NKT, 1, &positions, 0, row_bytes);
        }
    }
    backend.submit_pending();

    // ── the epilogue's own outputs: q (rope'd in place), k (rope'd), v (biased)
    let concat_after = words(&concat, NQT + 2 * NKT);
    let q_ref = &concat_after[..NQT];
    // Non-vacuous: the rope really moved q off its input.
    assert!(
        q_in.iter().zip(q_ref).any(|(a, b)| a.to_bits() != *b),
        "{case}: the reference q is unchanged; the comparison is vacuous"
    );
    for (what, got, want) in [
        ("q", words(&q_sec, NQT), q_ref),
        ("k", words(&k_sec, NKT), &concat_after[NQT..NQT + NKT]),
        ("v", words(&v_sec, NKT), &concat_after[NQT + NKT..]),
    ] {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g, w,
                "{case}: section {what}[{i}] = {g:08x} != concat {w:08x} (a per-section \
                 dispatch must be the fused epilogue's own arithmetic)"
            );
        }
    }

    // ── the stored rows: the addressed cell, byte for byte
    let k_ref_kv = cell_slice(&words(&kv_k_ref, touched), per_cell);
    let v_ref_kv = cell_slice(&words(&kv_v_ref, touched), per_cell);
    let k_sec_kv = cell_slice(&words(&kv_k_sec, touched), per_cell);
    let v_sec_kv = cell_slice(&words(&kv_v_sec, touched), per_cell);
    assert!(
        k_ref_kv.iter().any(|&w| w != 0),
        "{case}: the reference wrote an all-zero K cell; the comparison is vacuous"
    );
    for (what, got, want) in [
        ("kv_k", &k_sec_kv, &k_ref_kv),
        ("kv_v", &v_sec_kv, &v_ref_kv),
    ] {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g, w,
                "{case}: {what} word {i} of the stored cell = {g:08x} != {w:08x} (the \
                 section store must write the concat store's bytes)"
            );
        }
    }
}

/// f32 cache, non-interleaved rope: the shipped Qwen2/Qwen3 decode shape.
#[test]
fn metal_qkv_epilogue_sections_match_the_concat_f32() {
    section_dispatch_matches_the_concat(false, false, 0);
}

/// f16 cache: the store narrows to `half` inside the kernel.
#[test]
fn metal_qkv_epilogue_sections_match_the_concat_f16() {
    section_dispatch_matches_the_concat(true, false, 0);
}

/// Packed `q8_0` cache: the epilogue resolves the rows and the caller quantizes
/// them with the unfused chain's own store kernel.
#[test]
fn metal_qkv_epilogue_sections_match_the_concat_packed() {
    section_dispatch_matches_the_concat(false, true, 0);
}

/// Interleaved pairing (`rope_style == 1`): the section params must not depend
/// on which pair the rope arm picks.
#[test]
fn metal_qkv_epilogue_sections_match_the_concat_interleaved() {
    section_dispatch_matches_the_concat(false, false, 1);
}
