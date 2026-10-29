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

/// Raw-layout Q4_K/Q5_K scale/min decode: one unaligned 16-byte load and twelve
/// SSE ops per superblock, straight from the GGUF bytes as they sit in the row
/// (no permuted copy, no decoded table). Same algebra as
/// `block::unpack_q4k_scales`, vectorised:
///
/// * `scales[j]`   (j<4) = `blk[4+j] & 0x3F`
/// * `scales[4+j]`       = (`blk[12+j] & 0x0F`) | ((`blk[4+j]` >> 6) << 4)
/// * `mins[j]`     (j<4) = `blk[8+j] & 0x3F`
/// * `mins[4+j]`         = (`blk[12+j]` >> 4)   | ((`blk[8+j]` >> 6) << 4)
///
/// Two `vpshufb`s pair, per 16-bit lane, the byte carrying the low six bits with
/// the byte carrying the high two bits / the nibble, so one lane holds both
/// halves of one field: `w & 0x3F` is the low six bits, `(w >> 8) & 0x0F` is the
/// nibble (scale form) or `(w >> 12) & 0x0F` the high nibble (min form), and
/// `(w >> 2) & 0x30` moves the top two bits into place for both.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn q4k_scales_raw(u: __m128i) -> (__m128i, __m128i) {
    let m3f = _mm_set1_epi16(0x003F);
    let m0f = _mm_set1_epi16(0x000F);
    let m30 = _mm_set1_epi16(0x0030);
    // scales[0..4] = blk[4+j] & 0x3F, scales[4..8] = the nibble of blk[12+j]
    // plus the top two bits of blk[4+j] moved into place.
    let sw = _mm_shuffle_epi8(
        u,
        _mm_setr_epi8(
            4, 12, 5, 13, 6, 14, 7, 15, -128, -128, -128, -128, -128, -128, -128, -128,
        ),
    );
    let s_lo = _mm_and_si128(sw, m3f);
    let s_hi = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi16(sw, 8), m0f),
        _mm_and_si128(_mm_srli_epi16(sw, 2), m30),
    );
    // mins: lane j pairs blk[8+j] (low six + top two) with blk[12+j] (high nibble)
    let mw = _mm_shuffle_epi8(
        u,
        _mm_setr_epi8(
            8, 12, 9, 13, 10, 14, 11, 15, -128, -128, -128, -128, -128, -128, -128, -128,
        ),
    );
    let m_lo = _mm_and_si128(mw, m3f);
    let m_hi = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi16(mw, 12), m0f),
        _mm_and_si128(_mm_srli_epi16(mw, 2), m30),
    );
    (
        _mm_unpacklo_epi64(s_lo, s_hi),
        _mm_unpacklo_epi64(m_lo, m_hi),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn hsum_epi32_8(v: __m256i) -> i32 {
    let x = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4E));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xB1));
    _mm_cvtsi128_si32(x)
}

/// One Q6_K half's eight 16-element groups as **unsigned** 6-bit values (`0..63`,
/// the raw encoding, not `u - 32`), laid out exactly as the scalar reference's
/// `a[...]` dequantizes them: `qv[g]` is group `g` of half `n`. Keeping the
/// unsigned form is what lets a byte-pair sum ride `vpmaddubsw`; the `-32`
/// offset is carried as a per-group `32 * Σy` correction, which is exact in i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q6k_unsigned_groups(q6b: &[u8], n: usize) -> [__m128i; 8] {
    let m4 = _mm_set1_epi8(0x0F);
    let m3 = _mm_set1_epi8(3);
    let qlp = q6b.as_ptr().add(n * 64);
    let qhp = q6b.as_ptr().add(128 + n * 32);
    let ql0 = _mm_loadu_si128(qlp as *const __m128i);
    let ql1 = _mm_loadu_si128(qlp.add(16) as *const __m128i);
    let ql2 = _mm_loadu_si128(qlp.add(32) as *const __m128i);
    let ql3 = _mm_loadu_si128(qlp.add(48) as *const __m128i);
    let qh0 = _mm_loadu_si128(qhp as *const __m128i);
    let qh1 = _mm_loadu_si128(qhp.add(16) as *const __m128i);
    [
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
    ]
}

/// Q4_K × Q8_K (AVX2/FMA) — see `kquant::dot_q4_k_q8_k_scalar`.
///
/// Two restructurings on the raw GGUF layout, both exact in i32, which is what
/// keeps the result bitwise-identical to the scalar kernel rather than merely
/// close:
///
/// * the min term is vectorised over the q8_K `bsums` with the mins in i16 lanes
///   and one `vpmaddwd` for the `bsums[2s] + bsums[2s+1]` pairing, instead of a
///   serial eight-step i32 chain;
/// * the per-subblock scale rides the `vpmaddwd` that folds the byte pairs, so
///   the finished dot is never multiplied by a `vpmullo_epi32` per subblock.
///   `(scale, pair sum)` cannot overflow either: a Q4_K nibble is ≤ 15, so the
///   `vpmaddubsw` pair sum is at most 15·127·2 = 3810 (never saturating i16) and
///   the `vpmaddwd` product is at most 3810·63·2 = 480 060 per i32 lane.
///
/// The per-superblock float fold keeps the scalar kernel's order
/// (`sumf -= dmin * mterm; sumf += d * total`).
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
        let (s16, m16) = q4k_scales_raw(_mm_loadu_si128(q4b.as_ptr() as *const __m128i));
        // Min term, vectorised: Σ_s mins[s] * (bsums[2s] + bsums[2s+1]). The
        // eight decoded mins stay in i16 lanes; one `vpmaddwd` folds each bsum
        // pair and the widening multiply lands in 32-bit lanes.
        let bsums = _mm256_loadu_si256(q8b.as_ptr().add(274) as *const __m256i);
        let pairs = _mm256_madd_epi16(bsums, ones);
        let mterm = hsum_epi32_8(_mm256_mullo_epi32(_mm256_cvtepu16_epi32(m16), pairs));
        sumf -= dmin * mterm as f32;
        let mut sbuf = [0i16; 8];
        _mm_storeu_si128(sbuf.as_mut_ptr() as *mut __m128i, s16);
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let p4 = q4b.as_ptr();
        for j in 0..4 {
            let bits = _mm256_loadu_si256(p4.add(16 + 32 * j) as *const __m256i);
            let lo = _mm256_and_si256(bits, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4b);
            let ylo = _mm256_loadu_si256(q8b.as_ptr().add(2 + 64 * j) as *const __m256i);
            let yhi = _mm256_loadu_si256(q8b.as_ptr().add(2 + 64 * j + 32) as *const __m256i);
            let plo = _mm256_maddubs_epi16(lo, ylo);
            let phi = _mm256_maddubs_epi16(hi, yhi);
            // The subblock scale rides in the `vpmaddwd` instead of a
            // `vpmullo_epi32` on the finished dot.
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(plo, _mm256_set1_epi16(sbuf[2 * j])));
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_madd_epi16(phi, _mm256_set1_epi16(sbuf[2 * j + 1])),
            );
        }
        let total = hsum_epi32_8(acc1) + hsum_epi32_8(acc2);
        sumf += d * total as f32;
    }
    sumf
}

/// Q5_K × Q8_K (AVX2/FMA) — see `kquant::dot_q5_k_q8_k_scalar`.
///
/// The Q4_K shape applied to Q5_K on the raw GGUF layout, both restructurings
/// exact in i32, which is what keeps the result bitwise-identical to the scalar
/// kernel rather than merely close:
///
/// * the min term is vectorised over the q8_K `bsums` with the decoded mins in
///   i16 lanes and one `vpmaddwd` for the `bsums[2s] + bsums[2s+1]` pairing,
///   replacing the serial eight-step i32 chain;
/// * each subblock's scale rides the `vpmaddwd` that folds the `vpmaddubsw`
///   byte-pair sums, so the finished dot is never `vpmullo_epi32`-ed.
///
/// Neither new product can saturate: a Q5_K value is ≤ 31, so a pair sum is at
/// most 31·127·2 = 7874 and the `vpmaddwd` product at most 7874·63·2 = 992 124.
/// The per-superblock float fold keeps the scalar kernel's order
/// (`sumf -= dmin * mterm; sumf += d * total`).
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
        let (s16, m16) = q4k_scales_raw(_mm_loadu_si128(q5b.as_ptr() as *const __m128i));
        // Min term, vectorised as in the Q4_K arm: Σ_s mins[s] * (bsums[2s] +
        // bsums[2s+1]), the eight decoded mins staying in i16 lanes.
        let bsums = _mm256_loadu_si256(q8b.as_ptr().add(274) as *const __m256i);
        let pairs = _mm256_madd_epi16(bsums, ones);
        let mterm = hsum_epi32_8(_mm256_mullo_epi32(_mm256_cvtepu16_epi32(m16), pairs));
        sumf -= dmin * mterm as f32;
        let mut sbuf = [0i16; 8];
        _mm_storeu_si128(sbuf.as_mut_ptr() as *mut __m128i, s16);
        let qhv = _mm256_loadu_si256(q5b.as_ptr().add(16) as *const __m256i);
        let qs = q5b.as_ptr().add(48);
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        for j in 0..4 {
            let chunk = _mm256_loadu_si256(qs.add(32 * j) as *const __m256i);
            let lo = _mm256_and_si256(chunk, m4b);
            let hi = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), m4b);
            let sh = 2 * j as i32;
            let hlo = _mm256_and_si256(_mm256_srlv_epi32(qhv, _mm256_set1_epi32(sh)), one);
            let hhi = _mm256_and_si256(_mm256_srlv_epi32(qhv, _mm256_set1_epi32(sh + 1)), one);
            let ulo = _mm256_or_si256(lo, _mm256_slli_epi16(hlo, 4));
            let uhi = _mm256_or_si256(hi, _mm256_slli_epi16(hhi, 4));
            let ylo = _mm256_loadu_si256(q8b.as_ptr().add(2 + (2 * j) * 32) as *const __m256i);
            let yhi = _mm256_loadu_si256(q8b.as_ptr().add(2 + (2 * j + 1) * 32) as *const __m256i);
            let plo = _mm256_maddubs_epi16(ulo, ylo);
            let phi = _mm256_maddubs_epi16(uhi, yhi);
            // The subblock scale rides the widening multiply instead of a
            // `vpmullo_epi32` on the finished 32-element dot.
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(plo, _mm256_set1_epi16(sbuf[2 * j])));
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_madd_epi16(phi, _mm256_set1_epi16(sbuf[2 * j + 1])),
            );
        }
        let total = hsum_epi32_8(acc1) + hsum_epi32_8(acc2);
        sumf += d * total as f32;
    }
    sumf
}

/// Q6_K × Q8_K (AVX2) — the reference order of ADR-0061.
///
/// One `i32` accumulator of eight lanes per superblock: `vpmaddubsw` gives the
/// byte-pair sums of two groups, the per-group scale rides the following
/// `vpmaddwd`, the `-32` offset is subtracted once per superblock from the q8_K
/// bsums, and the f32 accumulator is one lane per sub-position across the row.
/// There is no per-group extraction and no serial fold — that chain is exactly
/// what ADR-0061 removed. The final reduction tree is the reference's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot_q6_k_q8_k(q6: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q6.len() / Q6KB;
    let mut acc = _mm256_setzero_ps();
    for i in 0..n_super {
        let q6b = &q6[i * Q6KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = f16_at(q6b, 208) * f16_at(q8b, 0);
        // -32 offset: 32 * sum_k (s[2k]*b[2k] + s[2k+1]*b[2k+1]), eight lanes.
        let bsums = _mm256_loadu_si256(q8b.as_ptr().add(274) as *const __m256i);
        let scales = _mm256_cvtepi8_epi16(_mm_loadu_si128(q6b.as_ptr().add(192) as *const __m128i));
        let corr = _mm256_slli_epi32(_mm256_madd_epi16(bsums, scales), 5);
        let mut sumi = _mm256_setzero_si256();
        for n in 0..2 {
            let qv = q6k_unsigned_groups(q6b, n);
            for t in 0..4 {
                let g = n * 8 + 2 * t;
                let u =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(qv[2 * t]), qv[2 * t + 1], 1);
                let y = _mm256_loadu_si256(
                    q8b.as_ptr().add(2 + (n * 8 + 2 * t) * 16) as *const __m256i
                );
                let p16 = _mm256_maddubs_epi16(u, y);
                // Each group's scale is repeated over its four i32 lanes, so the
                // lane sums carry the scale and nothing needs extracting.
                let sv = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_set1_epi16(q6b[192 + g] as i8 as i16)),
                    _mm_set1_epi16(q6b[192 + g + 1] as i8 as i16),
                    1,
                );
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sv, p16));
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
