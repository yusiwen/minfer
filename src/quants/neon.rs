// ============================================================
// aarch64 NEON fast paths — bit-exact with the scalar kernels:
// the int8×int8 products widen to int16/int32 and accumulate exactly
// (integer arithmetic is associative), and the per-block float ops
// are kept in the identical order. MINFER_NO_NEON=1 forces the
// scalar path for A/B.
// ============================================================
use super::*;
use crate::block::Q8KB;
use std::arch::aarch64::*;

pub(super) fn enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        // SDOT (vdotq_s32) is the ARMv8.2+ int8 dot-product instruction
        // (16 MACs/instr); Apple M-series and all modern aarch64 have it.
        // Without it we fall back to the scalar kernels (the vmlal chain
        // below would be an unnecessary third path for pre-2018 chips).
        std::arch::is_aarch64_feature_detected!("neon")
            && std::arch::is_aarch64_feature_detected!("dotprod")
            && !std::env::var("MINFER_NO_NEON").map_or(false, |v| v == "1")
    })
}

#[inline(always)]
pub(super) fn fp16(b0: u8, b1: u8) -> f32 {
    block::fp16_to_f32(u16::from_le_bytes([b0, b1]))
}

/// Accumulate 16 lanes of int8 × int8 into a scalar int32 with SDOT:
/// one `sdot v.4s, a.16b, b.16b` computes 4 independent 4-element dot
/// products (16 MACs) into an int32 accumulator — the llama.cpp Apple
/// Silicon pattern. std::arch's `vdotq_s32` is unstable, so the
/// instruction is emitted via stable inline asm. int32 accumulation is
/// exact, so the result equals the scalar kernel's sequential i32 sum
/// (bit-exact). Only called when `is_aarch64_feature_detected!("dotprod")`.
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn dot16(a: int8x16_t, b: int8x16_t) -> i32 {
    vaddvq_s32(sdot_vec(vdupq_n_s32(0), a, b))
}

/// Raw SDOT accumulate (no horizontal reduce): 4 lanes × 4-element dot
/// products (16 MACs) added to `acc`.
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn sdot_vec(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut acc = acc;
    std::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack),
    );
    acc
}

/// Q4_0 × Q8_0 (32 values/block, centered nibbles).
pub(super) unsafe fn dot_q4_0_q8_0(q4: &[u8], q8: &[u8], nb: usize) -> f32 {
    let m4b = vdupq_n_u8(0x0F);
    let s8 = vdupq_n_s8(8);
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &q4[ib * Q4B..];
        let yb = &q8[ib * Q8B..];
        let dx = fp16(xb[0], xb[1]);
        let dy = fp16(yb[0], yb[1]);
        let bytes = vld1q_u8(xb.as_ptr().add(2));
        let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, m4b)), s8);
        let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes)), s8);
        let y0 = vld1q_s8(yb.as_ptr().add(2) as *const i8);
        let y1 = vld1q_s8(yb.as_ptr().add(18) as *const i8);
        let si = dot16(lo, y0) + dot16(hi, y1);
        s += si as f32 * dx * dy;
    }
    s
}

/// Q8_0 × Q8_0 (32 values/block).
pub(super) unsafe fn dot_q8_0_q8_0(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * Q8B..];
        let yb = &y[ib * Q8B..];
        let dx = fp16(xb[0], xb[1]);
        let dy = fp16(yb[0], yb[1]);
        let x0 = vld1q_s8(xb.as_ptr().add(2) as *const i8);
        let x1 = vld1q_s8(xb.as_ptr().add(18) as *const i8);
        let y0 = vld1q_s8(yb.as_ptr().add(2) as *const i8);
        let y1 = vld1q_s8(yb.as_ptr().add(18) as *const i8);
        let si = dot16(x0, y0) + dot16(x1, y1);
        s += si as f32 * dx * dy;
    }
    s
}

/// Q4_1 × Q8_0 (unsigned nibbles + per-block min).
pub(super) unsafe fn dot_q4_1_q8_0(q4: &[u8], q8: &[u8], nb: usize) -> f32 {
    let m4b = vdupq_n_u8(0x0F);
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &q4[ib * Q41B..];
        let yb = &q8[ib * Q8B..];
        let d = fp16(xb[0], xb[1]);
        let m = fp16(xb[2], xb[3]);
        let dy = fp16(yb[0], yb[1]);
        let bytes = vld1q_u8(xb.as_ptr().add(4));
        let lo = vreinterpretq_s8_u8(vandq_u8(bytes, m4b));
        let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes));
        let y0 = vld1q_s8(yb.as_ptr().add(2) as *const i8);
        let y1 = vld1q_s8(yb.as_ptr().add(18) as *const i8);
        let sum_q = dot16(lo, y0) + dot16(hi, y1);
        let sum_y = vaddlvq_s8(y0) + vaddlvq_s8(y1);
        s += dy * (d * sum_q as f32 + m * sum_y as f32);
    }
    s
}

/// Q5_0 × Q8_0 (5-bit values; high bits expanded on the host — q5_0 is
/// rare in K_M models, the qh bit layout is awkward for NEON).
pub(super) unsafe fn dot_q5_0_q8_0(q5: &[u8], q8: &[u8], nb: usize) -> f32 {
    let m4b = vdupq_n_u8(0x0F);
    let s16 = vdupq_n_s8(16);
    let mut s = 0.0f32;
    for ib in 0..nb {
        let q5b = &q5[ib * 22..];
        let q8b = &q8[ib * Q8B..];
        let d = fp16(q5b[0], q5b[1]) * fp16(q8b[0], q8b[1]);
        let qh = u32::from_le_bytes([q5b[2], q5b[3], q5b[4], q5b[5]]);
        let qs = &q5b[6..22];
        let mut hb = [0i8; 32];
        for j in 0..16 {
            hb[j] = ((qh >> j) & 1) as i8;
            hb[j + 16] = ((qh >> (j + 16)) & 1) as i8;
        }
        let bytes = vld1q_u8(qs.as_ptr());
        let lo = vsubq_s8(
            vorrq_s8(
                vreinterpretq_s8_u8(vandq_u8(bytes, m4b)),
                vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr())),
            ),
            s16,
        );
        let hi = vsubq_s8(
            vorrq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes)),
                vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr().add(16))),
            ),
            s16,
        );
        let y0 = vld1q_s8(q8b.as_ptr().add(2) as *const i8);
        let y1 = vld1q_s8(q8b.as_ptr().add(18) as *const i8);
        let si = dot16(lo, y0) + dot16(hi, y1);
        s += si as f32 * d;
    }
    s
}

/// Q5_1 × Q8_0 (unsigned 5-bit + min).
pub(super) unsafe fn dot_q5_1_q8_0(q5: &[u8], q8: &[u8], nb: usize) -> f32 {
    let m4b = vdupq_n_u8(0x0F);
    let mut s = 0.0f32;
    for ib in 0..nb {
        let q5b = &q5[ib * 24..];
        let q8b = &q8[ib * Q8B..];
        let d = fp16(q5b[0], q5b[1]);
        let m = fp16(q5b[2], q5b[3]);
        let dy = fp16(q8b[0], q8b[1]);
        let qh = u32::from_le_bytes([q5b[4], q5b[5], q5b[6], q5b[7]]);
        let qs = &q5b[8..24];
        let mut hb = [0i8; 32];
        for j in 0..16 {
            hb[j] = ((qh >> j) & 1) as i8;
            hb[j + 16] = ((qh >> (j + 16)) & 1) as i8;
        }
        let bytes = vld1q_u8(qs.as_ptr());
        let lo = vorrq_s8(
            vreinterpretq_s8_u8(vandq_u8(bytes, m4b)),
            vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr())),
        );
        let hi = vorrq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes)),
            vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr().add(16))),
        );
        let y0 = vld1q_s8(q8b.as_ptr().add(2) as *const i8);
        let y1 = vld1q_s8(q8b.as_ptr().add(18) as *const i8);
        let sum_sub = dot16(lo, y0) + dot16(hi, y1);
        let sum_y = vaddlvq_s8(y0) + vaddlvq_s8(y1);
        s += dy * (d * sum_sub as f32 + m * sum_y as f32);
    }
    s
}

// ─── Q8_K-activation NEON kernels (K-quant matmuls) ─────────────────
/// Q4_K × Q8_K: 8 subblocks, scales/mins from the weight block, activation
/// bsums replace per-subblock reductions. Scales are applied to the SDOT
/// int32 vectors BEFORE the horizontal reduce (one vaddvq per superblock
/// instead of 8), and the mins term is fully vectorized — llama's shape.
pub(super) unsafe fn dot_q4_k_q8_k(q4: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q4.len() / Q4KB;
    let m4b = vdupq_n_u8(0x0F);
    let zero = vdupq_n_s32(0);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q4b = &q4[i * Q4KB..];
        let q8b = &q8k[i * Q8KB..];
        let d = fp16(q4b[0], q4b[1]) * fp16(q8b[0], q8b[1]);
        let dmin = fp16(q4b[2], q4b[3]) * fp16(q8b[0], q8b[1]);
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q4b[4..16]).unwrap());
        // mins term: (bsums[2s]+bsums[2s+1]) per 32-element subblock
        let bsums0 = vld1q_s16(q8b.as_ptr().add(274) as *const i16);
        let bsums1 = vld1q_s16(q8b.as_ptr().add(274 + 16) as *const i16);
        // vpaddq_s16(a, b) yields only 8 lanes: a's 4 pairs then b's 4
        // pairs — so load BOTH bsums halves (16 int16) and combine → the
        // 8 per-32-element sums in lanes 0..7. NOTE: vpaddq wraps on
        // int16 overflow — safe because the q8_K quantizer bounds bsums
        // to ±2032 (a pair sum ≤ 4064); the test uses quantizer output.
        let q8sums = vpaddq_s16(bsums0, bsums1);
        let mut ms = [0i16; 8];
        for k in 0..8 {
            ms[k] = mins[k] as i16;
        }
        // mins as two 4-lane halves — vld1q_s16 would read 8 lanes and
        // leave lanes 8..15 as undefined stack garbage.
        let mins_lo = vld1_s16(ms.as_ptr());
        let mins_hi = vld1_s16(ms.as_ptr().add(4));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), mins_lo),
            vmull_s16(vget_high_s16(q8sums), mins_hi),
        );
        sumf -= dmin * vaddvq_s32(prod) as f32;
        let mut sumi1 = zero;
        let mut sumi2 = zero;
        for j in 0..4 {
            let q4p = q4b.as_ptr().add(16 + 32 * j);
            let q8p = q8b.as_ptr().add(2 + 64 * j) as *const i8;
            let v0 = vld1q_u8(q4p);
            let v1 = vld1q_u8(q4p.add(16));
            let a0 = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
            let a1 = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
            let b0 = vld1q_s8(q8p);
            let b1 = vld1q_s8(q8p.add(16));
            let sc1 = scales[2 * j] as i32;
            sumi1 = vmlaq_n_s32(sumi1, sdot_vec(zero, a0, b0), sc1);
            sumi1 = vmlaq_n_s32(sumi1, sdot_vec(zero, a1, b1), sc1);
            let a2 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0));
            let a3 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1));
            let b2 = vld1q_s8(q8p.add(32));
            let b3 = vld1q_s8(q8p.add(48));
            let sc2 = scales[2 * j + 1] as i32;
            sumi2 = vmlaq_n_s32(sumi2, sdot_vec(zero, a2, b2), sc2);
            sumi2 = vmlaq_n_s32(sumi2, sdot_vec(zero, a3, b3), sc2);
        }
        sumf += d * (vaddvq_s32(sumi1) + vaddvq_s32(sumi2)) as f32;
    }
    sumf
}

/// Q6_K × Q8_K: no min term; one d per superblock. Per-group scales are
/// applied to the SDOT vectors before the single final reduce.
pub(super) unsafe fn dot_q6_k_q8_k(q6: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q6.len() / Q6KB;
    let m4b = vdupq_n_u8(0x0F);
    let m3 = vdupq_n_u8(3);
    let s32 = vdupq_n_s8(32);
    let zero = vdupq_n_s32(0);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q6b = &q6[i * Q6KB..];
        let q8b = &q8k[i * Q8KB..];
        let d = fp16(q6b[208], q6b[209]) * fp16(q8b[0], q8b[1]);
        for n in 0..2 {
            let qlp = q6b.as_ptr().add(n * 64);
            let qhp = q6b.as_ptr().add(128 + n * 32);
            let ql0 = vld1q_u8(qlp);
            let ql1 = vld1q_u8(qlp.add(16));
            let ql2 = vld1q_u8(qlp.add(32));
            let ql3 = vld1q_u8(qlp.add(48));
            let qh0 = vld1q_u8(qhp);
            let qh1 = vld1q_u8(qhp.add(16));
            let q0a = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql0, m4b),
                    vshlq_n_u8::<4>(vandq_u8(qh0, m3)),
                )),
                s32,
            );
            let q0b = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql1, m4b),
                    vshlq_n_u8::<4>(vandq_u8(qh1, m3)),
                )),
                s32,
            );
            let q1a = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql2, m4b),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh0), m3)),
                )),
                s32,
            );
            let q1b = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql3, m4b),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh1), m3)),
                )),
                s32,
            );
            let q2a = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8::<4>(ql0),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh0), m3)),
                )),
                s32,
            );
            let q2b = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8::<4>(ql1),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh1), m3)),
                )),
                s32,
            );
            let q3a = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8::<4>(ql2),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<6>(qh0), m3)),
                )),
                s32,
            );
            let q3b = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8::<4>(ql3),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<6>(qh1), m3)),
                )),
                s32,
            );
            let qv = [q0a, q0b, q1a, q1b, q2a, q2b, q3a, q3b];
            let mut acc = zero;
            for g in 0..8 {
                let vec = qv[(g / 2) * 2 + (g % 2)];
                let y = vld1q_s8(q8b.as_ptr().add(2 + (n * 8 + g) * 16) as *const i8);
                let scale = q6b[192 + n * 8 + g] as i8 as i32;
                acc = vmlaq_n_s32(acc, sdot_vec(zero, vec, y), scale);
            }
            // sumf += d * Σ_g scale_g * dot_g — the horizontal sum is exact
            // int; the float order matches the scalar kernel.
            sumf += d * vaddvq_s32(acc) as f32;
        }
    }
    sumf
}

/// Q5_K × Q8_K: high bits via variable right shift of the 32-byte qh.
pub(super) unsafe fn dot_q5_k_q8_k(q5: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q5.len() / 176;
    let m4b = vdupq_n_u8(0x0F);
    let one = vdupq_n_u8(1);
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q5b = &q5[i * 176..];
        let q8b = &q8k[i * Q8KB..];
        let d = fp16(q5b[0], q5b[1]) * fp16(q8b[0], q8b[1]);
        let dmin = fp16(q5b[2], q5b[3]) * fp16(q8b[0], q8b[1]);
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q5b[4..16]).unwrap());
        let mut mterm = 0i32;
        for s in 0..8 {
            let b0 = i16::from_le_bytes([q8b[274 + 4 * s], q8b[274 + 4 * s + 1]]) as i32;
            let b1 = i16::from_le_bytes([q8b[274 + 4 * s + 2], q8b[274 + 4 * s + 3]]) as i32;
            mterm += mins[s] as i32 * (b0 + b1);
        }
        sumf -= dmin * mterm as f32;
        let qh = q5b.as_ptr().add(16);
        let qs = q5b.as_ptr().add(48);
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..4 {
            let cp = qs.add(32 * j);
            let v0 = vld1q_u8(cp);
            let v1 = vld1q_u8(cp.add(16));
            // subblock 2j (lo nibbles)
            let s_sub = 2 * j;
            let sh = vdupq_n_s8(-(s_sub as i8));
            let h0 = vandq_u8(vshlq_u8(vld1q_u8(qh), sh), one);
            let h1 = vandq_u8(vshlq_u8(vld1q_u8(qh.add(16)), sh), one);
            let a0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v0, m4b), vshlq_n_u8::<4>(h0)));
            let a1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v1, m4b), vshlq_n_u8::<4>(h1)));
            let q8p = q8b.as_ptr().add(2 + (2 * j) * 32) as *const i8;
            let b0 = vld1q_s8(q8p);
            let b1 = vld1q_s8(q8p.add(16));
            let p_lo = dot16(a0, b0) + dot16(a1, b1);
            // subblock 2j+1 (hi nibbles)
            let s_sub2 = 2 * j + 1;
            let sh2 = vdupq_n_s8(-(s_sub2 as i8));
            let h2 = vandq_u8(vshlq_u8(vld1q_u8(qh), sh2), one);
            let h3 = vandq_u8(vshlq_u8(vld1q_u8(qh.add(16)), sh2), one);
            let a2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(v0), vshlq_n_u8::<4>(h2)));
            let a3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(v1), vshlq_n_u8::<4>(h3)));
            let q8p2 = q8b.as_ptr().add(2 + (2 * j + 1) * 32) as *const i8;
            let b2 = vld1q_s8(q8p2);
            let b3 = vld1q_s8(q8p2.add(16));
            let p_hi = dot16(a2, b2) + dot16(a3, b3);
            sumi1 += p_lo * scales[2 * j] as i32;
            sumi2 += p_hi * scales[2 * j + 1] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}
