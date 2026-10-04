use super::*;

// ============================================================
// f16 weights × f32 activations (#141)
// ============================================================
//
// F6 made an f16 GGUF *runnable* on the CPU by decoding one weight row at a
// time; #141 makes that path vectorized. Two properties are load-bearing here
// and both are asserted by the tests below:
//
//   1. `dot_f16_f32` is bit-identical to `vec_dot_f32` over the decoded row, on
//      every path (SIMD and scalar). The half→float conversion is exact, and
//      the SIMD loops run the same FMA tree in the same order as `vec_dot_f32`
//      — only the left operand's load differs (a convert instead of a load).
//   2. because of (1), `mat_mul_f16` may decode each weight row **once** and
//      reuse it across tokens without changing a single value. That is what
//      makes prefill read the 2-byte weight stream once per forward instead of
//      once per token; the F6 per-(row, token) shape is kept behind
//      `MINFER_NO_F16_ROWB=1` as the A/B control.

/// Which implementation `dot_f16_f32` runs on this machine.
///
/// This is the **single dispatch authority** — `dot_f16_f32` matches on it, so
/// the unit gate `f16_dot_uses_the_vectorized_path` cannot pass for the wrong
/// reason: it reads the path *and* observes the counter the SIMD entry point
/// bumps. Reverting the vectorization (dispatching to the scalar dot, or making
/// this report `Scalar`) leaves the counter unmoved and fails the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16DotPath {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

/// The path `dot_f16_f32` takes with the current feature detection and env.
pub fn f16_dot_path() -> F16DotPath {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            return F16DotPath::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            return F16DotPath::Neon;
        }
    }
    F16DotPath::Scalar
}

/// Test-only proof-of-execution for the SIMD f16 code. A gate that only asserts
/// `f16_dot_path() != Scalar` is blind to a `dot_f16_f32` body that calls the
/// scalar function anyway — this counter is bumped by the SIMD entry points
/// themselves (both the dot and the row decode), so that mutation is caught.
/// Thread-local so a parallel test harness cannot bump another test's count.
#[cfg(test)]
thread_local! {
    pub(crate) static F16_SIMD_PATH_CALLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// `MINFER_NO_F16_ROWB=1` keeps F6's per-(row, token) loop shape (the A/B
/// control for the row-blocked prefill; the two are bit-identical).
fn no_f16_rowb() -> bool {
    std::env::var("MINFER_NO_F16_ROWB").map_or(false, |v| v == "1")
}

/// Scalar reference for the f16 dot — the correctness oracle the SIMD paths are
/// compared against, and the fallback with no SIMD. f64 accumulation, mirroring
/// `vec_dot_f32`'s scalar fallback so both agree bit for bit on a scalar box.
pub fn dot_f16_f32_scalar(k: usize, w: &[u8], y: &[f32]) -> f32 {
    debug_assert!(w.len() >= k * 2 && y.len() >= k);
    let mut sumf = 0.0f64;
    for i in 0..k {
        let x = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
        sumf += x as f64 * y[i] as f64;
    }
    sumf as f32
}

/// Vectorized f16 weight row × f32 activations (`k` LE half bits in `w`).
///
/// Bit-identical to `vec_dot_f32(k, &decode(w), y)` when both take their SIMD
/// path — the invariant `mat_mul_f16`'s row blocking relies on.
pub fn dot_f16_f32(k: usize, w: &[u8], y: &[f32]) -> f32 {
    debug_assert!(w.len() >= k * 2 && y.len() >= k);
    match f16_dot_path() {
        #[cfg(target_arch = "x86_64")]
        F16DotPath::Avx2 => unsafe { dot_f16_f32_avx2(k, w, y) },
        #[cfg(target_arch = "aarch64")]
        F16DotPath::Neon => unsafe { neon_f16::dot(k, w, y) },
        F16DotPath::Scalar => dot_f16_f32_scalar(k, w, y),
    }
}

/// Decode one f16 weight row into an f32 row (exact: every f16 value is an f32).
pub(super) fn decode_f16_row(k: usize, w: &[u8], dst: &mut [f32]) {
    debug_assert!(w.len() >= k * 2 && dst.len() >= k);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe { decode_f16_row_avx2(k, w, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            unsafe { neon_f16::decode(k, w, dst) };
            return;
        }
    }
    for i in 0..k {
        dst[i] = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_f32_avx2(k: usize, w: &[u8], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    #[cfg(test)]
    F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
    let mut i = 0usize;
    let mut sum = _mm256_setzero_ps();
    let np = k & !7;
    while i < np {
        let h = _mm_loadu_si128(w.as_ptr().add(i * 2) as *const __m128i);
        let f = _mm256_cvtph_ps(h);
        sum = _mm256_fmadd_ps(f, _mm256_loadu_ps(y.as_ptr().add(i)), sum);
        i += 8;
    }
    // Same horizontal reduction as `vec_dot_f32_avx2` (hsum_float_8).
    let mut res = _mm256_extractf128_ps(sum, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(sum));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    let mut sumf = _mm_cvtss_f32(res);
    for j in i..k {
        sumf += crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * j], w[2 * j + 1]])) * y[j];
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn decode_f16_row_avx2(k: usize, w: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    #[cfg(test)]
    F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
    let mut i = 0usize;
    let np = k & !7;
    while i < np {
        let h = _mm_loadu_si128(w.as_ptr().add(i * 2) as *const __m128i);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    for j in i..k {
        dst[j] = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * j], w[2 * j + 1]]));
    }
}

/// NEON half→float conversion + FMA. `FCVTL`/`FCVTN` are baseline ARMv8-A NEON,
/// so no `+fp16` arithmetic feature is required (the FMA is f32).
#[cfg(target_arch = "aarch64")]
mod neon_f16 {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn convert8(h: uint16x8_t) -> (float32x4_t, float32x4_t) {
        (
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))),
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))),
        )
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(k: usize, w: &[u8], y: &[f32]) -> f32 {
        #[cfg(test)]
        super::F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
        let mut i = 0usize;
        let mut sum = vdupq_n_f32(0.0);
        while i + 8 <= k {
            let h = vld1q_u16(w.as_ptr().add(i * 2) as *const u16);
            let (lo, hi) = convert8(h);
            // Two 4-wide FMAs per 8 elements — the same registers, order and
            // rounding as `vec_dot_f32_neon` over the decoded row.
            sum = vfmaq_f32(sum, lo, vld1q_f32(y.as_ptr().add(i)));
            sum = vfmaq_f32(sum, hi, vld1q_f32(y.as_ptr().add(i + 4)));
            i += 8;
        }
        let mut acc = vaddvq_f32(sum);
        while i < k {
            acc += crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]])) * y[i];
            i += 1;
        }
        acc
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn decode(k: usize, w: &[u8], dst: &mut [f32]) {
        #[cfg(test)]
        super::F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
        let mut i = 0usize;
        while i + 8 <= k {
            let h = vld1q_u16(w.as_ptr().add(i * 2) as *const u16);
            let (lo, hi) = convert8(h);
            vst1q_f32(dst.as_mut_ptr().add(i), lo);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), hi);
            i += 8;
        }
        while i < k {
            dst[i] = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
            i += 1;
        }
    }
}

/// f16 weight ([m, k], row-major, little-endian half bits) × f32 activations.
///
/// The CPU matmul kernels are integer dots over quantized weights, so an f16
/// weight has no integer form; #141 gives it a vectorized float dot instead.
/// Weights are **never** materialized as f32 — only one row at a time is
/// decoded, and only when the row is reused across tokens.
///
/// Decoding a row once and folding every token into it is what lets the row
/// loop go to the shared worker pool (rows are disjoint outputs, so one worker
/// per contiguous row range is race-free by construction); it is also what
/// keeps prefill's DRAM traffic at the weight stream plus the activation
/// matrix. The two loop shapes are bit-identical per output element (property 1
/// above), so neither `n` nor the worker count changes a value.
pub fn mat_mul_f16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
    a: &[u8],  // [m, k] f16 weight rows
    b: &[f32], // [n, k] token-major activations
) {
    debug_assert!(c.len() >= m * n);
    debug_assert!(a.len() >= m * k * 2);
    debug_assert!(b.len() >= k * n);
    if n > 1
        && !no_f16_rowb()
        && crate::kernel::cpu_threads() > 1
        && m >= 2
        && m.saturating_mul(k).saturating_mul(n) >= crate::kernel::MIN_PARALLEL_MACS
    {
        let job = F16MatMulJob {
            m,
            n,
            k,
            a: a.as_ptr(),
            b: b.as_ptr(),
            c: c.as_mut_ptr(),
        };
        // SAFETY: `job` outlives the call; `f16_mm_rows` writes only rows it
        // owns (`c[col*m + r]` for its own `r` range), and `a`/`b`/`c` stay
        // borrowed for the duration. Same contract as `kernel`'s own jobs.
        crate::kernel::par_for(m, &job as *const F16MatMulJob as *const (), f16_mm_rows);
        return;
    }
    if n > 1 && !no_f16_rowb() {
        // Row-outer: one decode per weight row, `n` dots against the token rows.
        let mut row = vec![0f32; k];
        for r in 0..m {
            decode_f16_row(k, &a[r * k * 2..(r + 1) * k * 2], &mut row);
            for col in 0..n {
                c[col * m + r] = vec_dot_f32(k, &row, &b[col * k..(col + 1) * k]);
            }
        }
        return;
    }
    // Decode (F6 shape): the weight row is consumed by exactly one token, so a
    // scratch row would be pure overhead.
    for col in 0..n {
        let b_row = &b[col * k..(col + 1) * k];
        let c_row = &mut c[col * m..(col + 1) * m];
        for r in 0..m {
            c_row[r] = dot_f16_f32(k, &a[r * k * 2..(r + 1) * k * 2], b_row);
        }
    }
}

/// One f16 matmul shared by all pool workers (raw pointers, like `kernel`'s
/// `MmJob`: the borrows live on `mat_mul_f16`'s stack and the pool call waits
/// for every worker before they end).
struct F16MatMulJob {
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    b: *const f32,
    c: *mut f32,
}

/// Row kernel: rows `[r0, r1)` of the f16 matmul. SAFETY: `a`/`b`/`c` must stay
/// valid for the pool call; each row `r` is written exactly once by its owner
/// (`c[col*m + r]` for `r ∈ [r0, r1)` are disjoint per worker).
unsafe fn f16_mm_rows(ctx: *const (), r0: usize, r1: usize) {
    let j = &*(ctx as *const F16MatMulJob);
    let (m, n, k) = (j.m, j.n, j.k);
    let mut row = vec![0f32; k];
    for r in r0..r1 {
        let w = std::slice::from_raw_parts(j.a.add(r * k * 2), k * 2);
        decode_f16_row(k, w, &mut row);
        for col in 0..n {
            let b_row = std::slice::from_raw_parts(j.b.add(col * k), k);
            *j.c.add(col * m + r) = vec_dot_f32(k, &row, b_row);
        }
    }
}
