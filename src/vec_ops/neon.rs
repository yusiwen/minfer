// ============================================================
// aarch64 NEON fast paths (f32 vector ops used by attention, norms, etc.)
// ============================================================
use std::arch::aarch64::*;

pub(super) fn enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::arch::is_aarch64_feature_detected!("neon")
            && !std::env::var("MINFER_NO_NEON").map_or(false, |v| v == "1")
    })
}

pub(super) unsafe fn vec_dot_f32(n: usize, x: &[f32], y: &[f32]) -> f32 {
    let mut i = 0;
    let mut sum = vdupq_n_f32(0.0);
    while i + 4 <= n {
        let ax = vld1q_f32(x.as_ptr().add(i));
        let ay = vld1q_f32(y.as_ptr().add(i));
        sum = vfmaq_f32(sum, ax, ay);
        i += 4;
    }
    let mut acc = vaddvq_f32(sum);
    for &xv in &x[i..n] {
        acc += xv * y[i];
        i += 1;
    }
    acc
}

pub(super) unsafe fn vec_soft_max_f32(n: usize, y: &mut [f32], x: &[f32], max: f32) -> f64 {
    // fast exp via 2^(x·log2e): Cephes-style polynomial on the fractional
    // part + exponent-bit scaling. Input clamped to [-88, 0] so exp never
    // overflows and the exponent shift stays in range (scalar expf handles
    // -inf → 0 the same way via the clamp).
    let log2e = vdupq_n_f32(1.4426950408889634f32);
    let max_v = vdupq_n_f32(max);
    let zero = vdupq_n_f32(0.0);
    let cl = vdupq_n_f32(-88.0);
    let c0 = vdupq_n_f32(0.001333355814642844);
    let c1 = vdupq_n_f32(0.009618129107628477);
    let c2 = vdupq_n_f32(0.05550410866482158);
    let c3 = vdupq_n_f32(0.2402265069591007);
    let c4 = vdupq_n_f32(0.6931471805599453);
    let c5 = vdupq_n_f32(1.0);
    let exp_bias = vdupq_n_s32(127 << 23);

    let mut i = 0;
    let mut sum = 0.0f64;
    while i + 4 <= n {
        let a = vmaxq_f32(
            vminq_f32(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), max_v), zero),
            cl,
        );
        let nn = vmulq_f32(a, log2e);
        let nf = vrndmq_f32(nn);
        let f = vsubq_f32(nn, nf);
        let mut p = c0;
        p = vfmaq_f32(c1, p, f);
        p = vfmaq_f32(c2, p, f);
        p = vfmaq_f32(c3, p, f);
        p = vfmaq_f32(c4, p, f);
        p = vfmaq_f32(c5, p, f);
        let e = vshlq_n_s32::<23>(vcvtq_s32_f32(nf));
        let ex = vreinterpretq_f32_s32(vaddq_s32(e, exp_bias));
        let val = vmulq_f32(p, ex);
        vst1q_f32(y.as_mut_ptr().add(i), val);
        sum += (vaddvq_f32(val)) as f64;
        i += 4;
    }
    for &xv in &x[i..n] {
        let val = (xv - max).exp();
        y[i] = val;
        sum += val as f64;
        i += 1;
    }
    sum
}

pub(super) unsafe fn vec_soft_max_f32_inplace(n: usize, y: &mut [f32], max: f32) -> f64 {
    let log2e = vdupq_n_f32(1.4426950408889634f32);
    let max_v = vdupq_n_f32(max);
    let zero = vdupq_n_f32(0.0);
    let cl = vdupq_n_f32(-88.0);
    let c0 = vdupq_n_f32(0.001333355814642844);
    let c1 = vdupq_n_f32(0.009618129107628477);
    let c2 = vdupq_n_f32(0.05550410866482158);
    let c3 = vdupq_n_f32(0.2402265069591007);
    let c4 = vdupq_n_f32(0.6931471805599453);
    let c5 = vdupq_n_f32(1.0);
    let exp_bias = vdupq_n_s32(127 << 23);

    let mut i = 0;
    let mut sum = 0.0f64;
    while i + 4 <= n {
        let a = vmaxq_f32(
            vminq_f32(vsubq_f32(vld1q_f32(y.as_ptr().add(i)), max_v), zero),
            cl,
        );
        let nn = vmulq_f32(a, log2e);
        let nf = vrndmq_f32(nn);
        let f = vsubq_f32(nn, nf);
        let mut p = c0;
        p = vfmaq_f32(c1, p, f);
        p = vfmaq_f32(c2, p, f);
        p = vfmaq_f32(c3, p, f);
        p = vfmaq_f32(c4, p, f);
        p = vfmaq_f32(c5, p, f);
        let e = vshlq_n_s32::<23>(vcvtq_s32_f32(nf));
        let ex = vreinterpretq_f32_s32(vaddq_s32(e, exp_bias));
        let val = vmulq_f32(p, ex);
        vst1q_f32(y.as_mut_ptr().add(i), val);
        sum += vaddvq_f32(val) as f64;
        i += 4;
    }
    for j in i..n {
        let val = (y[j] - max).exp();
        y[j] = val;
        sum += val as f64;
    }
    sum
}

pub(super) unsafe fn vec_add_f32(n: usize, z: &mut [f32], x: &[f32], y: &[f32]) {
    let mut i = 0;
    while i + 4 <= n {
        vst1q_f32(
            z.as_mut_ptr().add(i),
            vaddq_f32(vld1q_f32(x.as_ptr().add(i)), vld1q_f32(y.as_ptr().add(i))),
        );
        i += 4;
    }
    for j in i..n {
        z[j] = x[j] + y[j];
    }
}

pub(super) unsafe fn vec_scale_f32(n: usize, y: &mut [f32], scale: f32) {
    let s = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= n {
        vst1q_f32(
            y.as_mut_ptr().add(i),
            vmulq_f32(vld1q_f32(y.as_ptr().add(i)), s),
        );
        i += 4;
    }
    for j in i..n {
        y[j] *= scale;
    }
}

pub(super) unsafe fn vec_muladd_f32(n: usize, y: &mut [f32], x: &[f32], scale: f32) {
    let s = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vy = vld1q_f32(y.as_ptr().add(i));
        vst1q_f32(y.as_mut_ptr().add(i), vfmaq_f32(vy, s, vx));
        i += 4;
    }
    for j in i..n {
        y[j] += scale * x[j];
    }
}
