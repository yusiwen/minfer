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
use super::avx2::{f16_at, k_mterm};
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

/// Low eight i32 lanes = `lo`, next eight = `hi`.
#[inline]
unsafe fn scale_vec(lo: i32, hi: i32) -> __m512i {
    _mm512_setr_epi32(
        lo, lo, lo, lo, lo, lo, lo, lo, hi, hi, hi, hi, hi, hi, hi, hi,
    )
}

/// Sum the four i32 lanes inside each 128-bit lane (the Q6_K grouping): after
/// this, lanes 0/4/8/12 hold the four group sums.
#[inline]
unsafe fn reduce4(v: __m512i) -> __m512i {
    let x = _mm512_add_epi32(v, _mm512_shuffle_epi32(v, 0xB1));
    _mm512_add_epi32(x, _mm512_shuffle_epi32(x, 0x4E))
}

/// Q4_K × Q8_K (AVX-512/VNNI) — see `kquant::dot_q4_k_q8_k_scalar`.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
pub(super) unsafe fn dot_q4_k_q8_k(q4: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q4.len() / Q4KB;
    let m4b = _mm256_set1_epi8(0x0F);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q4b = &q4[i * Q4KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q4b, 0) * f16_at(q8b, 0);
        let dmin = f16_at(q4b, 2) * f16_at(q8b, 0);
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q4b[4..16]).unwrap());
        sumf -= dmin * k_mterm(q8b, &mins) as f32;
        let mut acc = _mm512_setzero_si512();
        for j in 0..4 {
            let bits = _mm256_loadu_si256(q4b.as_ptr().add(16 + 32 * j) as *const __m256i);
            let lo = _mm256_and_si256(bits, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4b);
            let u = combine_256(lo, hi);
            let y = _mm512_loadu_si512(q8b.as_ptr().add(2 + 64 * j) as *const __m512i);
            let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, y);
            let sv = scale_vec(scales[2 * j], scales[2 * j + 1]);
            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dot, sv));
        }
        let total = _mm512_reduce_add_epi32(acc);
        sumf += d * total as f32;
    }
    sumf
}

/// Q5_K × Q8_K (AVX-512/VNNI) — see `kquant::dot_q5_k_q8_k_scalar`.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
pub(super) unsafe fn dot_q5_k_q8_k(q5: &[u8], q8k: &[u8]) -> f32 {
    const Q5KB: usize = 176;
    let n_super = q5.len() / Q5KB;
    let m4b = _mm256_set1_epi8(0x0F);
    let one = _mm256_set1_epi8(1);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q5b = &q5[i * Q5KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q5b, 0) * f16_at(q8b, 0);
        let dmin = f16_at(q5b, 2) * f16_at(q8b, 0);
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q5b[4..16]).unwrap());
        sumf -= dmin * k_mterm(q8b, &mins) as f32;
        let qhv = _mm256_loadu_si256(q5b.as_ptr().add(16) as *const __m256i);
        let mut acc = _mm512_setzero_si512();
        for j in 0..4 {
            let chunk = _mm256_loadu_si256(q5b.as_ptr().add(48 + 32 * j) as *const __m256i);
            let lo = _mm256_and_si256(chunk, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), m4b);
            let sh_lo = _mm256_set1_epi32((2 * j) as i32);
            let sh_hi = _mm256_set1_epi32((2 * j + 1) as i32);
            let hlo = _mm256_and_si256(_mm256_srlv_epi32(qhv, sh_lo), one);
            let hhi = _mm256_and_si256(_mm256_srlv_epi32(qhv, sh_hi), one);
            let ulo = _mm256_or_si256(lo, _mm256_slli_epi16(hlo, 4));
            let uhi = _mm256_or_si256(hi, _mm256_slli_epi16(hhi, 4));
            let u = combine_256(ulo, uhi);
            let y = _mm512_loadu_si512(q8b.as_ptr().add(2 + 64 * j) as *const __m512i);
            let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, y);
            let sv = scale_vec(scales[2 * j], scales[2 * j + 1]);
            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dot, sv));
        }
        let total = _mm512_reduce_add_epi32(acc);
        sumf += d * total as f32;
    }
    sumf
}

/// Q6_K × Q8_K (AVX-512/VNNI) — see `kquant::dot_q6_k_q8_k_scalar`.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
pub(super) unsafe fn dot_q6_k_q8_k(q6: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q6.len() / Q6KB;
    let m4 = _mm_set1_epi8(0x0F);
    let m3 = _mm_set1_epi8(3);
    let ones8 = _mm512_set1_epi8(1);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q6b = &q6[i * Q6KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q6b, 208) * f16_at(q8b, 0);
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
            for c in 0..2 {
                let u = combine_128(qv[c * 4], qv[c * 4 + 1], qv[c * 4 + 2], qv[c * 4 + 3]);
                let y = _mm512_loadu_si512(
                    q8b.as_ptr().add(2 + (n * 8 + c * 4) * 16) as *const __m512i
                );
                let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, y);
                let sy = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones8, y);
                // One i32 per group: reduce each 4-lane group, then fold the
                // Σy correction (×32) in-vector before extracting.
                let sub = _mm512_sub_epi32(reduce4(dot), _mm512_slli_epi32(reduce4(sy), 5));
                let g0 = _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(sub, 0));
                let g1 = _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(sub, 1));
                let g2 = _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(sub, 2));
                let g3 = _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(sub, 3));
                let sums = [g0, g1, g2, g3];
                for (k, &sum_sub) in sums.iter().enumerate() {
                    let gg = n * 8 + c * 4 + k;
                    let scale = q6b[192 + gg] as i8 as f32;
                    sumf += d * scale * (sum_sub as f32);
                }
            }
        }
    }
    sumf
}
