//! `#[cfg(test)] mod tests` for `src/vec_ops.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// The fused pass replaces the silu-then-mul pair, so it must agree with it
/// bit for bit (the unfused execution is the reference for `Op::SwiGLU`).
#[test]
fn swiglu_matches_silu_then_mul() {
    let gate: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.37).collect();
    let up: Vec<f32> = (0..37).map(|i| i as f32 * 0.11 - 2.0).collect();
    let n = gate.len();

    let mut want = vec![0f32; n];
    vec_silu_f32(n, &mut want, &gate);
    let silu = want.clone();
    vec_mul_f32(n, &mut want, &silu, &up);

    let mut got = vec![0f32; n];
    vec_swiglu_f32(n, &mut got, &gate, &up);

    assert_eq!(got, want, "fused swiglu must be bit-identical to silu+mul");
}

/// Tail handling: `vec_swiglu_f32`'s vector loop stops before the last
/// partial chunk, so exercise a non-multiple-of-8 length too.
#[test]
fn swiglu_scalar_tail() {
    let gate = [-3.0f32, 0.0, 0.5, 7.25];
    let up = [1.5f32, -2.0, 0.0, 4.0];
    let mut got = [0f32; 4];
    vec_swiglu_f32(4, &mut got, &gate, &up);
    for i in 0..4 {
        let want = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        assert_eq!(got[i], want);
    }
}

// === #141: f16 weights × f32 activations ===

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

/// The vectorized dot must agree with the scalar oracle. Dimensions include
/// non-multiples of 8 so the SIMD tail is exercised too.
#[test]
fn f16_dot_matches_the_scalar_oracle() {
    let mut seed = 0x141_c0ffee_u64;
    for k in [1usize, 7, 8, 9, 16, 31, 64, 127, 256] {
        let w: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 8.0).collect();
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 4.0).collect();
        let wb = f16_bytes(&w);
        let got = dot_f16_f32(k, &wb, &x);
        let want = dot_f16_f32_scalar(k, &wb, &x);
        let tol = (want.abs().max(got.abs())).max(1e-6) * 1e-5;
        assert!(
            (got - want).abs() <= tol,
            "k={k}: vectorized {got} != scalar {want}"
        );
    }
}

/// Property (1) of the f16 block: `dot_f16_f32` is bit-identical to
/// `vec_dot_f32` over the decoded row. This is what licenses the row-blocked
/// prefill loop; if a future SIMD change reorders the accumulation, the
/// prefill and decode paths would silently stop agreeing and this fails.
#[test]
fn f16_dot_is_bit_identical_to_an_f32_dot_over_the_decoded_row() {
    let mut seed = 0x141_d0d0_u64;
    for k in [8usize, 64, 127, 896] {
        let w: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 16.0).collect();
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 4.0).collect();
        let wb = f16_bytes(&w);
        let mut row = vec![0f32; k];
        decode_f16_row(k, &wb, &mut row);
        // The decode itself is exact.
        for i in 0..k {
            assert_eq!(
                row[i].to_bits(),
                crate::block::fp16_to_f32(u16::from_le_bytes([wb[2 * i], wb[2 * i + 1]])).to_bits(),
                "k={k} i={i}: decoded row is not the exact f16 value"
            );
        }
        assert_eq!(
            dot_f16_f32(k, &wb, &x).to_bits(),
            vec_dot_f32(k, &row, &x).to_bits(),
            "k={k}: f16 dot != f32 dot over the decoded row"
        );
    }
}

/// Property (2): the row-blocked multi-token matmul (prefill) and the
/// per-token form (decode) must produce the same bits — a prefill of one
/// token is a decode. `n` therefore never changes a value.
#[test]
fn f16_matmul_prefill_matches_decode_bitwise() {
    let mut seed = 0x141_beef_u64;
    let (m, k, n) = (13usize, 96usize, 5usize);
    let wf: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 8.0).collect();
    let w = f16_bytes(&wf);
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();

    let mut prefill = vec![0f32; m * n];
    mat_mul_f16(m, n, k, &mut prefill, &w, &b);

    for col in 0..n {
        let mut one = vec![0f32; m];
        mat_mul_f16(m, 1, k, &mut one, &w, &b[col * k..(col + 1) * k]);
        for r in 0..m {
            assert_eq!(
                prefill[col * m + r].to_bits(),
                one[r].to_bits(),
                "col={col} row={r}: prefill != decode"
            );
        }
    }
}

/// The worker-pool path must produce the same bits as the direct form. A
/// wrong chunk boundary (a duplicated or skipped row range) shows up as a
/// mismatch here. Shape is above `MIN_PARALLEL_MACS` so the pool is used
/// when the box has more than one CPU.
#[test]
fn f16_matmul_pool_matches_the_direct_form_bitwise() {
    let mut seed = 0x141_7007_u64;
    let (m, k, n) = (256usize, 512usize, 8usize);
    assert!(m * k * n >= crate::kernel::MIN_PARALLEL_MACS);
    let wf: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 8.0).collect();
    let w = f16_bytes(&wf);
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();

    let mut got = vec![0f32; m * n];
    mat_mul_f16(m, n, k, &mut got, &w, &b);

    for col in 0..n {
        let b_row = &b[col * k..(col + 1) * k];
        for r in 0..m {
            let want = dot_f16_f32(k, &w[r * k * 2..(r + 1) * k * 2], b_row);
            assert_eq!(
                got[col * m + r].to_bits(),
                want.to_bits(),
                "col={col} row={r}: pooled != direct"
            );
        }
    }
    eprintln!(
        "#141 f16 matmul pool path: {} threads",
        crate::kernel::cpu_threads()
    );
}

/// The vectorized path is the one that actually **runs** where the CPU has
/// it. Two observations, because either alone can pass for the wrong
/// reason: `f16_dot_path()` names the dispatch branch, and
/// `F16_SIMD_PATH_CALLS` is bumped by the SIMD entry points themselves. A
/// revert that dispatches to the scalar dot (or that removes the feature
/// check) leaves the counter unmoved and fails here.
#[test]
fn f16_dot_uses_the_vectorized_path() {
    let path = f16_dot_path();
    let simd = path != F16DotPath::Scalar;

    let mut seed = 0x141_9a7bu64;
    let k = 64usize;
    let wf: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 4.0).collect();
    let w = f16_bytes(&wf);
    let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 2.0).collect();
    let (m, n) = (4usize, 3usize);
    let wm: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 4.0).collect();
    let wmb = f16_bytes(&wm);
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 2.0).collect();

    let before = F16_SIMD_PATH_CALLS.with(|c| c.get());
    let _ = dot_f16_f32(k, &w, &x);
    let after_dot = F16_SIMD_PATH_CALLS.with(|c| c.get());
    let mut c = vec![0f32; m * n];
    mat_mul_f16(m, n, k, &mut c, &wmb, &b);
    let after_mm = F16_SIMD_PATH_CALLS.with(|c| c.get());

    if simd {
        assert!(
            after_dot > before,
            "f16_dot_path() reports {path:?} but the SIMD dot never ran \
             ({before} -> {after_dot})"
        );
        // The row-blocked prefill decodes each weight row through its own
        // SIMD path; reverting only that one must fail here too.
        assert!(
            after_mm > after_dot,
            "f16_dot_path() reports {path:?} but the SIMD row decode never ran \
             ({after_dot} -> {after_mm})"
        );
    } else {
        assert_eq!(
            (after_dot, after_mm),
            (before, before),
            "the scalar path must not run a SIMD entry point"
        );
    }
    eprintln!(
        "#141 f16 dot path: {path:?}, SIMD entry-point calls: dot {before} -> {after_dot}, \
         row decode -> {after_mm}"
    );
}

// === #142: bf16 weights × f32 activations ===

/// Encode test values as bf16 bytes with the writer's own RNE encoder, so
/// the decode test exercises the pair the file actually carries.
fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&crate::block::f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

/// The bf16 decode is exact (`f32::from_bits(bits << 16)`), and the matmul is
/// the plain f32 dot over the decoded row. This is the invariant that makes
/// a bf16 file's logits comparable to the f16 file's.
#[test]
fn bf16_decode_is_exact_and_the_dot_is_the_f32_dot_over_the_decoded_row() {
    let mut seed = 0x142_d0d0_u64;
    for k in [1usize, 7, 8, 9, 64, 127, 896] {
        let w: Vec<f32> = (0..k * k).map(|_| lcg(&mut seed) * 16.0).collect();
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 4.0).collect();
        let wb = bf16_bytes(&w);
        let mut row = vec![0f32; k];
        decode_bf16_row(k, &wb, &mut row);
        for i in 0..k {
            assert_eq!(
                row[i].to_bits(),
                f32::from_bits((u16::from_le_bytes([wb[2 * i], wb[2 * i + 1]]) as u32) << 16)
                    .to_bits(),
                "k={k} i={i}: decoded row is not the exact bf16 value"
            );
        }
        // n=1 takes the direct path: each output row must be the plain f32
        // dot over that row's exact decode.
        let mut got = vec![0f32; k];
        mat_mul_bf16(k, 1, k, &mut got, &wb, &x);
        for r in 0..k {
            let mut row_r = vec![0f32; k];
            decode_bf16_row(k, &wb[r * k * 2..(r + 1) * k * 2], &mut row_r);
            assert_eq!(
                got[r].to_bits(),
                vec_dot_f32(k, &row_r, &x).to_bits(),
                "k={k} row={r}: bf16 dot != f32 dot over the decoded row"
            );
        }
    }
}

/// Property (2) for bf16: `n` never changes a value (prefill == decode).
#[test]
fn bf16_matmul_prefill_matches_decode_bitwise() {
    let mut seed = 0x142_beef_u64;
    let (m, k, n) = (13usize, 96usize, 5usize);
    let wf: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 8.0).collect();
    let w = bf16_bytes(&wf);
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();

    let mut prefill = vec![0f32; m * n];
    mat_mul_bf16(m, n, k, &mut prefill, &w, &b);
    for col in 0..n {
        let mut one = vec![0f32; m];
        mat_mul_bf16(m, 1, k, &mut one, &w, &b[col * k..(col + 1) * k]);
        for r in 0..m {
            assert_eq!(
                prefill[col * m + r].to_bits(),
                one[r].to_bits(),
                "col={col} row={r}: prefill != decode"
            );
        }
    }
}

/// The worker-pool path (above `MIN_PARALLEL_MACS`) must match the direct
/// form bitwise — a wrong row-range split would show up here.
#[test]
fn bf16_matmul_pool_matches_the_direct_form_bitwise() {
    let mut seed = 0x142_7007_u64;
    let (m, k, n) = (256usize, 512usize, 8usize);
    assert!(m * k * n >= crate::kernel::MIN_PARALLEL_MACS);
    let wf: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 8.0).collect();
    let w = bf16_bytes(&wf);
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();

    let mut got = vec![0f32; m * n];
    mat_mul_bf16(m, n, k, &mut got, &w, &b);

    let mut row = vec![0f32; k];
    for col in 0..n {
        let b_row = &b[col * k..(col + 1) * k];
        for r in 0..m {
            decode_bf16_row(k, &w[r * k * 2..(r + 1) * k * 2], &mut row);
            let want = vec_dot_f32(k, &row, b_row);
            assert_eq!(
                got[col * m + r].to_bits(),
                want.to_bits(),
                "col={col} row={r}: pooled != direct"
            );
        }
    }
}

/// The cross-file claim, at the unit level: when an f16 file and a bf16 file
/// carry the *same* values (which is exactly what a bf16 checkpoint
/// converted to either type gives — bf16 → f16 is exact), the two matmuls
/// produce **bit-identical** output. This is why the real-model gate expects
/// bitwise logits, not a looser bound.
#[test]
fn bf16_and_f16_matmul_agree_bitwise_when_the_values_are_equal() {
    let mut seed = 0x142_f16b_u64;
    let (m, k, n) = (17usize, 128usize, 6usize);
    // Values in the f16∩bf16 grid: quantize to the bf16 grid first, then
    // keep only those the f16 encoding also stores exactly (f16 has more
    // mantissa bits but a much smaller exponent range; the two grids
    // overlap, they do not contain each other).
    let mut wf: Vec<f32> = Vec::with_capacity(m * k);
    while wf.len() < m * k {
        let v = lcg(&mut seed) * 8.0;
        let b = crate::block::bf16_to_f32(crate::block::f32_to_bf16_bits(v));
        if half::f16::from_f32(b).to_f32().to_bits() == b.to_bits() {
            wf.push(b);
        }
    }
    let w16 = f16_bytes(&wf);
    let wbf = bf16_bytes(&wf);
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();

    let mut c16 = vec![0f32; m * n];
    let mut cbf = vec![0f32; m * n];
    mat_mul_f16(m, n, k, &mut c16, &w16, &b);
    mat_mul_bf16(m, n, k, &mut cbf, &wbf, &b);
    for i in 0..m * n {
        assert_eq!(
            c16[i].to_bits(),
            cbf[i].to_bits(),
            "index {i}: f16 and bf16 matmuls disagree on equal values"
        );
    }
}

// === #724: llama3 rope scaling ===

/// The table is the **contract** with llama.cpp's converter (`conversion/llama.py`,
/// `LlamaModel.generate_extra_tensors`, build b11829 / `184f65360`): the reference
/// engine reads these numbers back from `rope_freqs.weight` and divides each pair's
/// angle by them, so a drift here diverges from the reference at every position
/// while every other gate stays green.
///
/// The pinned numbers below are **the bytes of two real files**, read out of their
/// `rope_freqs.weight` tensor (f32, offset 43 095 6544 / 7 831 456):
///
/// - `unsloth/Llama-3.1-8B-Instruct-GGUF` Q4_K_M: base 500000, factor 8, original
///   context 8192, band 1..4, head dim 128 → 64 pairs. High-frequency pairs are
///   untouched, the low-frequency ones are divided by 8, and seven pairs ramp
///   continuously between the two.
/// - `unsloth/Llama-3.2-1B-Instruct-GGUF` Q4_K_M: factor 32, head dim 64 → 32
///   pairs, with only three pairs in the ramp.
///
/// The tolerance is 1e-6 relative, not zero: the converter iterates a **torch f32**
/// frequency tensor, and `torch.pow(f32)` and this crate's `powf` may round the
/// last bit differently. The tensor path — the shape real llama.cpp files carry —
/// reads the file's bytes, so the reference comparison there is exact.
#[test]
fn llama3_table_matches_the_converter() {
    let t = llama3_freqs(500000.0, 8.0, 8192.0, 1.0, 4.0, 128).expect("llama3 table");
    assert_eq!(t.len(), 64);
    #[rustfmt::skip]
    const WANT: [f32; 64] = [
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.2074838876724243, 1.5534145832061768, 2.026313066482544,
        2.6945300102233887, 3.684253215789795, 5.257327079772949, 8.0,
        8.0, 8.0, 8.0, 8.0,
        8.0, 8.0, 8.0, 8.0,
        8.0, 8.0, 8.0, 8.0,
        8.0, 8.0, 8.0, 8.0,
        8.0, 8.0, 8.0, 8.0,
        8.0, 8.0, 8.0, 8.0,
        8.0, 8.0, 8.0, 8.0,
    ];
    for (d, want) in WANT.iter().enumerate() {
        let got = t.at(d);
        assert!(
            (got - want).abs() <= 1e-6 * want.abs(),
            "pair {d}: got {got}, want {want}"
        );
    }
    // The 1B file's table: factor 32, three pairs in the ramp.
    let t = llama3_freqs(500000.0, 32.0, 8192.0, 1.0, 4.0, 64).expect("llama3 table");
    assert_eq!(t.len(), 32);
    #[rustfmt::skip]
    const WANT_1B: [f32; 32] = [
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.6513292789459229,
        3.2922627925872803, 9.666730880737305, 32.0, 32.0,
        32.0, 32.0, 32.0, 32.0,
        32.0, 32.0, 32.0, 32.0,
        32.0, 32.0, 32.0, 32.0,
    ];
    for (d, want) in WANT_1B.iter().enumerate() {
        let got = t.at(d);
        assert!(
            (got - want).abs() <= 1e-6 * want.abs(),
            "1B pair {d}: got {got}, want {want}"
        );
    }
}

/// A second config whose band edges land on different pairs, so the boundary
/// comparisons are exercised and not just the two flat regions.
#[test]
fn llama3_table_band_edges_are_continuous() {
    let t = llama3_freqs(10000.0, 4.0, 64.0, 1.0, 2.0, 16).expect("small table");
    let want = [1.0f32, 1.0, 3.788_686_3, 4.0, 4.0, 4.0, 4.0, 4.0];
    assert_eq!(t.len(), want.len());
    for (d, w) in want.iter().enumerate() {
        assert!(
            (t.at(d) - w).abs() <= 1e-5,
            "pair {d}: got {}, want {w}",
            t.at(d)
        );
    }
}

/// The degenerate band (two equal factors) is refused by name rather than
/// dividing by zero — llama.cpp's converter carries the same comment for Llama 4.
#[test]
fn llama3_refuses_a_degenerate_band() {
    let e = llama3_freqs(10000.0, 8.0, 8192.0, 4.0, 4.0, 128).unwrap_err();
    assert!(e.contains("high_freq_factor > low_freq_factor"), "{e}");
    let e = llama3_freqs(10000.0, 0.0, 8192.0, 1.0, 4.0, 128).unwrap_err();
    assert!(e.contains("factor > 0"), "{e}");
    let e = llama3_freqs(10000.0, 8.0, 8192.0, 1.0, 4.0, 129).unwrap_err();
    assert!(e.contains("even head dim"), "{e}");
    let e = llama3_freqs(
        10000.0,
        8.0,
        8192.0,
        1.0,
        4.0,
        2 * (super::rope::ROPE_FREQS_MAX + 1),
    )
    .unwrap_err();
    assert!(e.contains("over this build's limit"), "{e}");
}

/// Issue #724's second acceptance criterion: an unscaled model is **bit-identical**
/// to the pre-#724 expression, not merely close. `None` (and the empty table) must
/// reproduce the old arithmetic exactly, which is what keeps every pinned Qwen and
/// Llama golden valid.
#[test]
fn no_table_is_bit_identical_to_the_old_expression() {
    for hd in [64usize, 128] {
        for base in [10_000.0f32, 500_000.0] {
            for d in 0..hd / 2 {
                let old = 1.0f32 / base.powf((2 * d) as f32 / hd as f32);
                let got = rope_inv_freq(base, 1.0, None, d, hd);
                assert_eq!(got.to_bits(), old.to_bits(), "hd {hd} base {base} d {d}");
                let empty = RopeFreqs::none();
                let got = rope_inv_freq(base, 1.0, Some(&empty), d, hd);
                assert_eq!(got.to_bits(), old.to_bits(), "empty table, d {d}");
            }
        }
    }
}

/// A table divides the pair's inverse frequency, which is exactly what llama.cpp's
/// rope does with the tensor (`theta_base / freq_factor`).
#[test]
fn a_table_divides_the_pair_frequency() {
    let t = RopeFreqs::from_slice(&[1.0, 2.0, 4.0, 8.0]).expect("table");
    assert_eq!(t.len(), 4);
    for d in 0..4 {
        let unscaled = rope_inv_freq(10_000.0, 1.0, None, d, 8);
        let scaled = rope_inv_freq(10_000.0, 1.0, Some(&t), d, 8);
        let want = unscaled / [1.0, 2.0, 4.0, 8.0][d];
        assert!((scaled - want).abs() <= 1e-7 * want.abs(), "pair {d}");
    }
    // Past the table's end nothing is scaled (a table narrower than `hd/2` is a
    // file error the loader refuses, but the accessor must not read past it).
    assert_eq!(t.at(4), 1.0);
    // And the table's own bounds: empty and over-long are refused, not truncated.
    assert!(RopeFreqs::from_slice(&[]).is_none());
    let over = vec![1.0f32; super::rope::ROPE_FREQS_MAX + 1];
    assert!(RopeFreqs::from_slice(&over).is_none());
}
