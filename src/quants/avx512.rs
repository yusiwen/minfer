//! x86_64 AVX-512/VNNI fast paths for the K-quant × Q8_K dots.
//!
//! Same contract as the AVX2 kernels in [`super::avx2`]: the int8×int8
//! products are exact in i32 (VNNI's `dpbusd` accumulates 4 products per lane
//! with no saturation) and the per-superblock float arithmetic is kept in the
//! scalar kernel's order, so the result is **bitwise-identical** to the scalar
//! reference. Selected at runtime only on a CPU that has
//! `avx512f/avx512bw/avx512dq/avx512vl/avx512vnni`; any other machine falls
//! back to AVX2 and then scalar. `MINFER_NO_AVX512=1` forces the AVX2 kernel
//! and `MINFER_NO_AVX2=1` forces scalar (both are read once, like
//! `MINFER_NO_NEON`).
use super::avx2::{f16_at, hsum_epi32_8, q4k_scales_raw};
use super::*;

/// True when the whole AVX-512/VNNI layer may be used. `MINFER_NO_AVX2=1`
/// disables this layer too — it is the x86 counterpart of `MINFER_NO_NEON`,
/// forcing the scalar reference for A/B.
pub(super) fn enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni")
            && !std::env::var("MINFER_NO_AVX512").map_or(false, |v| v == "1")
            && !std::env::var("MINFER_NO_AVX2").map_or(false, |v| v == "1")
    })
}

/// Concatenate two 256-bit vectors into one 512-bit vector (lo in the low
/// half).
#[inline]
unsafe fn combine_256(lo: __m256i, hi: __m256i) -> __m512i {
    _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1)
}

/// Concatenate four 128-bit vectors into one 512-bit vector.
#[inline]
unsafe fn combine_128(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m512i {
    let v = _mm512_castsi128_si512(a);
    let v = _mm512_inserti32x4(v, b, 1);
    let v = _mm512_inserti32x4(v, c, 2);
    _mm512_inserti32x4(v, d, 3)
}

/// Q4_K × Q8_K (AVX-512/VNNI) — see `kquant::dot_q4_k_q8_k_scalar`.
///
/// The AVX2 arm's two restructurings in their 512-bit form, on the raw GGUF
/// layout and both exact in i32: the min term is vectorised over the q8_K
/// `bsums` with the mins in i16 lanes, and the per-subblock scale is folded into
/// the `vpmaddwd` that folds the `vpmaddubsw` byte-pair sums — one `vpermw` per
/// group spreads the decoded i16 scale pair over its two subblocks — so the
/// finished dot is never multiplied. The per-superblock float fold keeps the
/// scalar kernel's order (`sumf -= dmin * mterm; sumf += d * total`); the
/// `vpmaddubsw` pair sums cannot saturate (a Q4_K nibble is ≤ 15, so ≤ 15·127·2)
/// and the `vpmaddwd` product stays far inside i32, as on the AVX2 arm.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
pub(super) unsafe fn dot_q4_k_q8_k(q4: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q4.len() / Q4KB;
    let m4b = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    // hoisted: i16 lanes 0-15 select scale[2j], lanes 16-31 select scale[2j+1]
    let base = combine_256(_mm256_setzero_si256(), _mm256_set1_epi16(1));
    let mut idx = [_mm512_setzero_si512(); 4];
    for (j, slot) in idx.iter_mut().enumerate() {
        *slot = _mm512_add_epi16(base, _mm512_set1_epi16(2 * j as i16));
    }
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q4b = &q4[i * Q4KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q4b, 0) * f16_at(q8b, 0);
        let dmin = f16_at(q4b, 2) * f16_at(q8b, 0);
        let (s16, m16) = q4k_scales_raw(_mm_loadu_si128(q4b.as_ptr() as *const __m128i));
        // Min term, vectorised as in the AVX2 arm.
        let bsums = _mm256_loadu_si256(q8b.as_ptr().add(274) as *const __m256i);
        let pairs = _mm256_madd_epi16(bsums, ones);
        let mterm = hsum_epi32_8(_mm256_mullo_epi32(_mm256_cvtepu16_epi32(m16), pairs));
        sumf -= dmin * mterm as f32;
        let sv16 = _mm512_zextsi128_si512(s16);
        let mut acc = _mm512_setzero_si512();
        for (j, id) in idx.iter().enumerate() {
            let bits = _mm256_loadu_si256(q4b.as_ptr().add(16 + 32 * j) as *const __m256i);
            let lo = _mm256_and_si256(bits, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4b);
            let u = combine_256(lo, hi);
            let y = _mm512_loadu_si512(q8b.as_ptr().add(2 + 64 * j) as *const __m512i);
            // Pair sums, then the scale folded in by the same `vpmaddwd`.
            let ps = _mm512_maddubs_epi16(u, y);
            acc = _mm512_add_epi32(
                acc,
                _mm512_madd_epi16(ps, _mm512_permutexvar_epi16(*id, sv16)),
            );
        }
        let total = _mm512_reduce_add_epi32(acc);
        sumf += d * total as f32;
    }
    sumf
}

/// Q5_K × Q8_K (AVX-512/VNNI) — see `kquant::dot_q5_k_q8_k_scalar`.
///
/// The AVX2 arm's two restructurings in their 512-bit form, on the raw GGUF
/// layout and both exact in i32: the min term is vectorised over the q8_K
/// `bsums` with the decoded mins in i16 lanes, and the per-subblock scale rides
/// the `vpmaddwd` that folds the `vpmaddubsw` byte-pair sums — one `vpermw` per
/// group spreads the decoded i16 scale pair over its two subblocks — so the
/// finished dot is never multiplied. The per-superblock float fold keeps the
/// scalar kernel's order (`sumf -= dmin * mterm; sumf += d * total`); a Q5_K
/// value is ≤ 31, so the pair sum (≤ 31·127·2 = 7874) never saturates i16 and
/// the `vpmaddwd` product (≤ 7874·63·2 = 992 124) stays far inside i32.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
pub(super) unsafe fn dot_q5_k_q8_k(q5: &[u8], q8k: &[u8]) -> f32 {
    const Q5KB: usize = 176;
    let n_super = q5.len() / Q5KB;
    let m4b = _mm256_set1_epi8(0x0F);
    let one = _mm256_set1_epi8(1);
    let ones = _mm256_set1_epi16(1);
    // hoisted (a function of the loop index only): i16 lanes 0-15 select
    // scales[2j] and lanes 16-31 scales[2j+1], for each j.
    let base = combine_256(_mm256_setzero_si256(), _mm256_set1_epi16(1));
    let mut idx = [_mm512_setzero_si512(); 4];
    for (j, slot) in idx.iter_mut().enumerate() {
        *slot = _mm512_add_epi16(base, _mm512_set1_epi16(2 * j as i16));
    }
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q5b = &q5[i * Q5KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q5b, 0) * f16_at(q8b, 0);
        let dmin = f16_at(q5b, 2) * f16_at(q8b, 0);
        let (s16, m16) = q4k_scales_raw(_mm_loadu_si128(q5b.as_ptr() as *const __m128i));
        // Min term, vectorised as in the AVX2 arm.
        let bsums = _mm256_loadu_si256(q8b.as_ptr().add(274) as *const __m256i);
        let pairs = _mm256_madd_epi16(bsums, ones);
        let mterm = hsum_epi32_8(_mm256_mullo_epi32(_mm256_cvtepu16_epi32(m16), pairs));
        sumf -= dmin * mterm as f32;
        let sv16 = _mm512_zextsi128_si512(s16);
        let qhv = _mm256_loadu_si256(q5b.as_ptr().add(16) as *const __m256i);
        let mut acc = _mm512_setzero_si512();
        for (j, id) in idx.iter().enumerate() {
            let chunk = _mm256_loadu_si256(q5b.as_ptr().add(48 + 32 * j) as *const __m256i);
            let lo = _mm256_and_si256(chunk, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), m4b);
            let sh = 2 * j as i32;
            let hlo = _mm256_and_si256(_mm256_srlv_epi32(qhv, _mm256_set1_epi32(sh)), one);
            let hhi = _mm256_and_si256(_mm256_srlv_epi32(qhv, _mm256_set1_epi32(sh + 1)), one);
            let ulo = _mm256_or_si256(lo, _mm256_slli_epi16(hlo, 4));
            let uhi = _mm256_or_si256(hi, _mm256_slli_epi16(hhi, 4));
            let u = combine_256(ulo, uhi);
            let y = _mm512_loadu_si512(q8b.as_ptr().add(2 + 64 * j) as *const __m512i);
            // Pair sums, then the scale folded in by the same `vpmaddwd`.
            let ps = _mm512_maddubs_epi16(u, y);
            acc = _mm512_add_epi32(
                acc,
                _mm512_madd_epi16(ps, _mm512_permutexvar_epi16(*id, sv16)),
            );
        }
        let total = _mm512_reduce_add_epi32(acc);
        sumf += d * total as f32;
    }
    sumf
}

/// Q6_K × Q8_K (AVX-512/VNNI) — the reference order of ADR-0061 in 512-bit
/// form: `vpmaddubsw`-free VNNI on four groups at a time, each group's scale
/// applied with one `vpmulld`, the four i32 lanes per group folded pairwise so
/// the result has the same eight lanes as the AVX2 arm, the `-32` offset
/// subtracted once per superblock from the q8_K bsums, and one f32 accumulator
/// per lane across the row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
pub(super) unsafe fn dot_q6_k_q8_k(q6: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q6.len() / Q6KB;
    let m4 = _mm_set1_epi8(0x0F);
    let m3 = _mm_set1_epi8(3);
    let mut acc = _mm256_setzero_ps();
    for i in 0..n_super {
        let q6b = &q6[i * Q6KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q6b, 208) * f16_at(q8b, 0);
        let bsums = _mm256_loadu_si256(q8b.as_ptr().add(274) as *const __m256i);
        let scales = _mm256_cvtepi8_epi16(_mm_loadu_si128(q6b.as_ptr().add(192) as *const __m128i));
        let corr = _mm256_slli_epi32(_mm256_madd_epi16(bsums, scales), 5);
        let mut sumi = _mm256_setzero_si256();
        for n in 0..2 {
            let qlp = q6b.as_ptr().add(n * 64);
            let qhp = q6b.as_ptr().add(128 + n * 32);
            let ql0 = _mm_loadu_si128(qlp as *const __m128i);
            let ql1 = _mm_loadu_si128(qlp.add(16) as *const __m128i);
            let ql2 = _mm_loadu_si128(qlp.add(32) as *const __m128i);
            let ql3 = _mm_loadu_si128(qlp.add(48) as *const __m128i);
            let qh0 = _mm_loadu_si128(qhp as *const __m128i);
            let qh1 = _mm_loadu_si128(qhp.add(16) as *const __m128i);
            let qv = [
                _mm_or_si128(
                    _mm_and_si128(ql0, m4),
                    _mm_slli_epi16(_mm_and_si128(qh0, m3), 4),
                ),
                _mm_or_si128(
                    _mm_and_si128(ql1, m4),
                    _mm_slli_epi16(_mm_and_si128(qh1, m3), 4),
                ),
                _mm_or_si128(
                    _mm_and_si128(ql2, m4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 2), m3), 4),
                ),
                _mm_or_si128(
                    _mm_and_si128(ql3, m4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 2), m3), 4),
                ),
                _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql0, 4), m4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 4), m3), 4),
                ),
                _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql1, 4), m4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 4), m3), 4),
                ),
                _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql2, 4), m4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 6), m3), 4),
                ),
                _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql3, 4), m4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 6), m3), 4),
                ),
            ];
            for pair in 0..2 {
                let g = n * 8 + pair * 4;
                let u = combine_128(
                    qv[pair * 4],
                    qv[pair * 4 + 1],
                    qv[pair * 4 + 2],
                    qv[pair * 4 + 3],
                );
                let y = _mm512_loadu_si512(q8b.as_ptr().add(2 + g * 16) as *const __m512i);
                let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, y);
                let s0 = q6b[192 + g] as i8 as i32;
                let s1 = q6b[192 + g + 1] as i8 as i32;
                let s2 = q6b[192 + g + 2] as i8 as i32;
                let s3 = q6b[192 + g + 3] as i8 as i32;
                let sv = _mm512_set_epi32(
                    s3, s3, s3, s3, s2, s2, s2, s2, s1, s1, s1, s1, s0, s0, s0, s0,
                );
                let scaled = _mm512_mullo_epi32(dot, sv);
                // Groups 0/2 (even) fold into lanes 0-3, groups 1/3 (odd) into
                // lanes 4-7 - the same eight lanes the AVX2 arm accumulates.
                let lo = _mm512_castsi512_si256(scaled);
                let hi = _mm512_extracti64x4_epi64(scaled, 1);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(lo, hi));
            }
        }
        let sub = _mm256_sub_epi32(sumi, corr);
        acc = _mm256_add_ps(
            acc,
            _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sub)),
        );
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
}
