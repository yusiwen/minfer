use super::*;

// ============================================================
// bf16 weights (#142)
// ============================================================
//
// A bf16 weight has the same problem the f16 one does: no integer form, so the
// quantized CPU dots cannot read it. The decode is even cheaper — bf16 is
// `f32`'s top 16 bits, so it is a left shift and **exact** for every value — and
// the dot is then the plain `vec_dot_f32` over the decoded row, exactly the
// shape `mat_mul_f16` folds tokens into. There is deliberately no A/B loop-shape
// control here (unlike `MINFER_NO_F16_ROWB`): decoding a row once and reusing it
// across tokens is not an optimization to compare against, it is the only shape
// this path has, and it is bit-identical to decoding per dot by construction.

/// Decode one bf16 weight row to f32 (`f32::from_bits(bits << 16)`, exact).
pub(super) fn decode_bf16_row(k: usize, w: &[u8], dst: &mut [f32]) {
    debug_assert!(w.len() >= k * 2 && dst.len() >= k);
    for i in 0..k {
        dst[i] = crate::block::bf16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
    }
}

/// bf16 weight ([m, k], row-major, little-endian bf16) × f32 activations.
///
/// The bf16 twin of [`mat_mul_f16`]: one row decode per weight row, then a
/// `vec_dot_f32` per token against it, threaded through the shared worker pool
/// when the work is large enough. Because the bf16→f32 decode is exact, this
/// computes the *same* `vec_dot_f32` over the *same* f32 row as `mat_mul_f16`
/// whenever the two files carry the same value — which is what makes the real
/// -model comparison against the f16 file meaningful.
pub fn mat_mul_bf16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
    a: &[u8],  // [m, k] bf16 weight rows
    b: &[f32], // [n, k] token-major activations
) {
    debug_assert!(c.len() >= m * n);
    debug_assert!(a.len() >= m * k * 2);
    debug_assert!(b.len() >= k * n);
    if n > 1
        && crate::kernel::cpu_threads() > 1
        && m >= 2
        && m.saturating_mul(k).saturating_mul(n) >= crate::kernel::MIN_PARALLEL_MACS
    {
        let job = Bf16MatMulJob {
            m,
            n,
            k,
            a: a.as_ptr(),
            b: b.as_ptr(),
            c: c.as_mut_ptr(),
        };
        // SAFETY: same contract as `f16_mm_rows` — `job` outlives the call and
        // each worker writes only the rows it owns.
        crate::kernel::par_for(m, &job as *const Bf16MatMulJob as *const (), bf16_mm_rows);
        return;
    }
    let mut row = vec![0f32; k];
    for r in 0..m {
        decode_bf16_row(k, &a[r * k * 2..(r + 1) * k * 2], &mut row);
        for col in 0..n {
            c[col * m + r] = vec_dot_f32(k, &row, &b[col * k..(col + 1) * k]);
        }
    }
}

/// One bf16 matmul shared by all pool workers (raw pointers, like
/// [`F16MatMulJob`]).
struct Bf16MatMulJob {
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    b: *const f32,
    c: *mut f32,
}

/// Row kernel: rows `[r0, r1)` of the bf16 matmul. SAFETY: `a`/`b`/`c` must stay
/// valid for the pool call; each row `r` is written exactly once by its owner.
unsafe fn bf16_mm_rows(ctx: *const (), r0: usize, r1: usize) {
    let j = &*(ctx as *const Bf16MatMulJob);
    let (m, n, k) = (j.m, j.n, j.k);
    let mut row = vec![0f32; k];
    for r in r0..r1 {
        let w = std::slice::from_raw_parts(j.a.add(r * k * 2), k * 2);
        decode_bf16_row(k, w, &mut row);
        for col in 0..n {
            let b_row = std::slice::from_raw_parts(j.b.add(col * k), k);
            *j.c.add(col * m + r) = vec_dot_f32(k, &row, b_row);
        }
    }
}
