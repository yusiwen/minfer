//! x86_64 AVX2/FMA helpers: the Q8_0 activation quantizer's fast path and
//! the `hsum_float_8` horizontal reduce the AVX2 dot kernels share.
use super::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn quantize_avx2(x: &[f32], y: &mut [u8], k: usize) {
    use std::arch::x86_64::*;
    let nb = k / 32;
    for i in 0..nb {
        let off = i * 32;
        let v0 = _mm256_loadu_ps(x.as_ptr().add(off));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(off + 8));
        let v2 = _mm256_loadu_ps(x.as_ptr().add(off + 16));
        let v3 = _mm256_loadu_ps(x.as_ptr().add(off + 24));
        let sb = _mm256_set1_ps(-0.0f32);
        let ma = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sb, v0), _mm256_andnot_ps(sb, v1)),
            _mm256_max_ps(_mm256_andnot_ps(sb, v2), _mm256_andnot_ps(sb, v3)),
        );
        let m4 = _mm_max_ps(_mm256_extractf128_ps(ma, 1), _mm256_castps256_ps128(ma));
        let m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let ms = _mm_cvtss_f32(_mm_max_ss(m4, _mm_movehdup_ps(m4)));
        let d = ms / 127.0f32;
        let db = half::f16::from_f32(d).to_bits().to_le_bytes();
        let yo = i * Q8B;
        y[yo] = db[0];
        y[yo + 1] = db[1];
        let id = if ms != 0.0 { 127.0f32 / ms } else { 0.0f32 };
        let mul = _mm256_set1_ps(id);
        let i0 = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(v0, mul),
            _MM_ROUND_NEAREST as i32,
        ));
        let i1 = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(v1, mul),
            _MM_ROUND_NEAREST as i32,
        ));
        let i2 = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(v2, mul),
            _MM_ROUND_NEAREST as i32,
        ));
        let i3 = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(v3, mul),
            _MM_ROUND_NEAREST as i32,
        ));
        let i0 = _mm256_packs_epi32(i0, i1);
        let i2 = _mm256_packs_epi32(i2, i3);
        let i0 = _mm256_packs_epi16(i0, i2);
        let i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256(y.as_mut_ptr().add(yo + 2) as *mut __m256i, i0);
    }
}

pub(super) fn quantize_scalar(x: &[f32], y: &mut [u8], k: usize) {
    let nb = k / 32;
    for i in 0..nb {
        let mut am = 0.0f32;
        for j in 0..32 {
            am = am.max(x[i * 32 + j].abs());
        }
        let d = am / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let db = half::f16::from_f32(d).to_bits().to_le_bytes();
        let yo = i * Q8B;
        y[yo] = db[0];
        y[yo + 1] = db[1];
        for j in 0..32 {
            y[yo + 2 + j] = (x[i * 32 + j] * id).round_ties_even().clamp(-128.0, 127.0) as i8 as u8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
pub(super) unsafe fn hsum_float_8(x: __m256) -> f32 {
    let x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    _mm_cvtss_f32(_mm_add_ss(x128, _mm_movehdup_ps(x128)))
}

// ============================================================
// K-quant AVX2/FMA kernels (Q4_K / Q5_K / Q6_K × Q8_K) — bit-exact with
// the scalar kernels: the int8×int8 products widen to i32 and accumulate
// exactly (integer arithmetic is associative), and the per-superblock float
// arithmetic is kept in the scalar kernel's order (multiply then add — rustc
// does not contract to FMA by default, so the AVX2 result is bitwise-identical
// to the scalar reference, not merely within a tolerance). MINFER_NO_AVX2=1
// forces the scalar path for A/B, the x86 counterpart of MINFER_NO_NEON.
// ============================================================

#[cfg(target_arch = "x86_64")]
pub(super) fn enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && !std::env::var("MINFER_NO_AVX2").map_or(false, |v| v == "1")
    })
}

/// f16 at `b[off..off+2]` decoded the way the scalar kernels decode it.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) fn f16_at(b: &[u8], off: usize) -> f32 {
    block::fp16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// The Q4_K/Q5_K min term: `Σ_s mins[s] * (bsums[2s] + bsums[2s+1])`, the
/// exact scalar i32 accumulation.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn k_mterm(q8b: &[u8], mins: &[i32; 8]) -> i32 {
    let mut m = 0i32;
    for s in 0..8 {
        let b0 = i16::from_le_bytes([q8b[274 + 4 * s], q8b[274 + 4 * s + 1]]) as i32;
        let b1 = i16::from_le_bytes([q8b[274 + 4 * s + 2], q8b[274 + 4 * s + 3]]) as i32;
        m += mins[s] * (b0 + b1);
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_8(v: __m256i) -> i32 {
    let x = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4E));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xB1));
    _mm_cvtsi128_si32(x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_4(v: __m128i) -> i32 {
    let x = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xB1));
    _mm_cvtsi128_si32(x)
}

/// One Q6_K 16-element group: `Σ (u - 32) * y` with `u` the unsigned 6-bit
/// values and `y` the Q8_K bytes. `Σ u*y - 32*Σ y` is exact in i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q6_group_sum(u: __m128i, y: *const i8) -> i32 {
    let yv = _mm_loadu_si128(y as *const __m128i);
    let ones16 = _mm_set1_epi16(1);
    let dot_u = hsum_epi32_4(_mm_madd_epi16(_mm_maddubs_epi16(u, yv), ones16));
    let sum_y = hsum_epi32_4(_mm_madd_epi16(
        _mm_maddubs_epi16(_mm_set1_epi8(1), yv),
        ones16,
    ));
    dot_u - 32 * sum_y
}

/// Q4_K × Q8_K (AVX2/FMA) — see `kquant::dot_q4_k_q8_k_scalar`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_q4_k_q8_k(q4: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q4.len() / Q4KB;
    let m4b = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q4b = &q4[i * Q4KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q4b, 0) * f16_at(q8b, 0);
        let dmin = f16_at(q4b, 2) * f16_at(q8b, 0);
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q4b[4..16]).unwrap());
        sumf -= dmin * k_mterm(q8b, &mins) as f32;
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let p4 = q4b.as_ptr();
        for j in 0..4 {
            let bits = _mm256_loadu_si256(p4.add(16 + 32 * j) as *const __m256i);
            let lo = _mm256_and_si256(bits, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4b);
            let ylo = _mm256_loadu_si256(q8b.as_ptr().add(2 + 64 * j) as *const __m256i);
            let yhi = _mm256_loadu_si256(q8b.as_ptr().add(2 + 64 * j + 32) as *const __m256i);
            let dotlo = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, ylo), ones);
            let dothi = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, yhi), ones);
            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_mullo_epi32(dotlo, _mm256_set1_epi32(scales[2 * j])),
            );
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_mullo_epi32(dothi, _mm256_set1_epi32(scales[2 * j + 1])),
            );
        }
        let total = hsum_epi32_8(acc1) + hsum_epi32_8(acc2);
        sumf += d * total as f32;
    }
    sumf
}

/// Q5_K × Q8_K (AVX2/FMA) — see `kquant::dot_q5_k_q8_k_scalar`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_q5_k_q8_k(q5: &[u8], q8k: &[u8]) -> f32 {
    const Q5KB: usize = 176;
    let n_super = q5.len() / Q5KB;
    let m4b = _mm256_set1_epi8(0x0F);
    let one = _mm256_set1_epi8(1);
    let ones = _mm256_set1_epi16(1);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q5b = &q5[i * Q5KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q5b, 0) * f16_at(q8b, 0);
        let dmin = f16_at(q5b, 2) * f16_at(q8b, 0);
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q5b[4..16]).unwrap());
        sumf -= dmin * k_mterm(q8b, &mins) as f32;
        let qh = q5b.as_ptr().add(16);
        let qs = q5b.as_ptr().add(48);
        let qhv = _mm256_loadu_si256(qh as *const __m256i);
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        for j in 0..4 {
            let chunk = _mm256_loadu_si256(qs.add(32 * j) as *const __m256i);
            let lo = _mm256_and_si256(chunk, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), m4b);
            let sh_lo = _mm256_set1_epi32((2 * j) as i32);
            let sh_hi = _mm256_set1_epi32((2 * j + 1) as i32);
            let hlo = _mm256_and_si256(_mm256_srlv_epi32(qhv, sh_lo), one);
            let hhi = _mm256_and_si256(_mm256_srlv_epi32(qhv, sh_hi), one);
            let ulo = _mm256_or_si256(lo, _mm256_slli_epi16(hlo, 4));
            let uhi = _mm256_or_si256(hi, _mm256_slli_epi16(hhi, 4));
            let ylo = _mm256_loadu_si256(q8b.as_ptr().add(2 + (2 * j) * 32) as *const __m256i);
            let yhi = _mm256_loadu_si256(q8b.as_ptr().add(2 + (2 * j + 1) * 32) as *const __m256i);
            let dotlo = _mm256_madd_epi16(_mm256_maddubs_epi16(ulo, ylo), ones);
            let dothi = _mm256_madd_epi16(_mm256_maddubs_epi16(uhi, yhi), ones);
            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_mullo_epi32(dotlo, _mm256_set1_epi32(scales[2 * j])),
            );
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_mullo_epi32(dothi, _mm256_set1_epi32(scales[2 * j + 1])),
            );
        }
        let total = hsum_epi32_8(acc1) + hsum_epi32_8(acc2);
        sumf += d * total as f32;
    }
    sumf
}

/// Q6_K × Q8_K (AVX2/FMA) — see `kquant::dot_q6_k_q8_k_scalar`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_q6_k_q8_k(q6: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q6.len() / Q6KB;
    let m4 = _mm_set1_epi8(0x0F);
    let m3 = _mm_set1_epi8(3);
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
            for g in 0..8 {
                let y = q8b.as_ptr().add(2 + (n * 8 + g) * 16) as *const i8;
                let sum_sub = q6_group_sum(qv[g], y);
                let scale = q6b[192 + n * 8 + g] as i8 as f32;
                sumf += d * scale * (sum_sub as f32);
            }
        }
    }
    sumf
}
