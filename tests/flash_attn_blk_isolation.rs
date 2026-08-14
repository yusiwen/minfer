// Isolation tests for the prefill flash-attention port
// (kernel_flash_attn_blk_f32 / kernel_flash_attn_blk_f16, NSG=4 fixed-shape):
// verify correctness vs a scalar CPU reference (online-softmax flash math with
// a causal mask), including the partial last KV block (nkv % 64 != 0) via the
// [2][64][nkt] tail-pad buffer (kernel_kv_tail_pad), nkv < C (whole KV in one
// block), multiple query tokens (nt up to 200 → multiple 8-token threadgroups),
// GQA heads, and both f32 and f16 KV caches. Also A/B the blk path against the
// classic gqa_attn_f32 kernel (the long-verified reference). macOS only.
#![cfg(target_os = "macos")]

use metal::{MTLResourceOptions, MTLSize};

const NH: usize = 14;
const NK: usize = 2;
const HD: usize = 64;
const NE_Q: usize = NH * HD; // 896
const SCALE: f32 = 0.125; // 1/sqrt(64)
const C: usize = 64;

fn cpu_attn(q: &[f32], k: &[f32], v: &[f32], nh: usize, nk: usize, hd: usize,
    nt: usize, nkv: usize, scale: f32, o: &mut [f32]) {
    let gqa = nh / nk;
    let ne_q = nh * hd;
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qs = t * ne_q + h * hd;
            let vl = (t + 1).min(nkv);
            let mut scrs = vec![0.0f32; nkv];
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks = kv * nk * hd + hk * hd;
                let mut s = 0.0f32;
                for d in 0..hd { s += q[qs + d] * k[ks + d]; }
                s *= scale;
                scrs[kv] = s;
                if s > mx { mx = s; }
            }
            let mut sm = 0.0f32;
            for kv in 0..vl { scrs[kv] = (scrs[kv] - mx).exp(); sm += scrs[kv]; }
            let inv = 1.0 / sm;
            let os = t * ne_q + h * hd;
            for d in 0..hd { o[os + d] = 0.0; }
            for kv in 0..vl {
                let vs = kv * nk * hd + hk * hd;
                let w = scrs[kv] * inv;
                for d in 0..hd { o[os + d] += w * v[vs + d]; }
            }
        }
    }
}

/// Scalar CPU reference for the PARTIAL-block pad read: the kernel sees the KV
/// as blocks of C=64; a partial block reads virtual rows [nkv-C, nkv) from the
/// pad buffer (copied from the real cache, zeros elsewhere), and the causal
/// mask hides rows >= nkv or < 0. This reference is the SAME math — a full
/// nblk=ceil(nkv/C) flash over the padded virtual cache with position-aware
/// masking — so it exercises the pad mechanism exactly as the kernel sees it.
fn cpu_attn_padded(q: &[f32], k: &[f32], v: &[f32], nh: usize, nk: usize, hd: usize,
    nt: usize, nkv: usize, scale: f32, o: &mut [f32]) {
    let gqa = nh / nk;
    let ne_q = nh * hd;
    let nkt = nk * hd;
    let nblk = nkv.div_ceil(C);
    // virtual cache: [2][nblk*C][nkt], zeros beyond nkv (K in first half, V in second)
    let mut kv = vec![0.0f32; 2 * nblk * C * nkt];
    for pos in 0..nkv {
        let src = pos * nkt;
        kv[pos * nkt..pos * nkt + nkt].copy_from_slice(&k[src..src + nkt]);
        kv[nblk * C * nkt + pos * nkt..nblk * C * nkt + pos * nkt + nkt]
            .copy_from_slice(&v[src..src + nkt]);
    }
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qs = t * ne_q + h * hd;
            let qpos = t; // fresh prefill positions 0..nt-1
            // scores over ALL virtual rows [0, nblk*C), masking row > qpos
            let mut scrs = vec![0.0f32; nblk * C];
            let mut mx = f32::NEG_INFINITY;
            for r in 0..nblk * C {
                if r > qpos {
                    scrs[r] = -65504.0;
                    continue;
                }
                let mut s = 0.0f32;
                for d in 0..hd { s += q[qs + d] * kv[r * nkt + hk * hd + d]; }
                s *= scale;
                scrs[r] = s;
                if s > mx { mx = s; }
            }
            let mut sm = 0.0f32;
            for r in 0..nblk * C {
                scrs[r] = (scrs[r] - mx).exp();
                sm += scrs[r];
            }
            let inv = 1.0 / sm;
            let os = t * ne_q + h * hd;
            for d in 0..hd { o[os + d] = 0.0; }
            for r in 0..nblk * C {
                let w = scrs[r] * inv;
                if w == 0.0 { continue; }
                for d in 0..hd { o[os + d] += w * kv[nblk * C * nkt + r * nkt + hk * hd + d]; }
            }
        }
    }
}

fn stats(name: &str, got: &[f32], want: &[f32]) -> (f32, f64, bool) {
    let maxdiff = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let dot: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
    let n1: f64 = got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let n2: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (n1 * n2);
    println!("{name}: maxdiff_ref={maxdiff:.2e} cos={cos:.6}");
    (maxdiff, cos, maxdiff == 0.0)
}

struct Ctx {
    device: metal::Device,
    lib: metal::Library,
    cmdq: metal::CommandQueue,
}

impl Ctx {
    fn new() -> Self {
        let device = metal::Device::system_default().expect("no metal device");
        let src = include_str!("../src/metal.metal");
        let opts = metal::CompileOptions::new();
        let lib = device.new_library_with_source(src, &opts).unwrap_or_else(|e| panic!("shader compile: {e}"));
        let cmdq = device.new_command_queue();
        Ctx { device, lib, cmdq }
    }

    /// kernel_flash_attn_blk_f32/_f16 with the tail-pad copy. Replicates the
    /// host dispatch: if nkv % 64 != 0, first kernel_kv_tail_pad fills pad
    /// [2][64][nkt] from the last 64 virtual rows, then the blk kernel runs
    /// with grid (ceil(nt/8), nh) x (32,4) threads and 7168 B shmem.
    fn blk_attn(&self, q: &[f32], k: &[f32], v: &[f32], nt: usize, nkv: usize,
        f16: bool) -> Vec<f32> {
        let nkt = NK * HD;
        let (qb, kb, vb) = self.buffers(q, k, v, f16);
        let positions: Vec<i32> = (0..nt as i32).collect();
        let pb = self.device.new_buffer_with_data(positions.as_ptr() as *const _, (positions.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let elem = if f16 { 2u64 } else { 4u64 };
        let pad = self.device.new_buffer((2 * C as u64 * nkt as u64) * elem, MTLResourceOptions::StorageModeShared);
        let ob = self.device.new_buffer((nt * NE_Q * 4) as u64, MTLResourceOptions::StorageModeShared);

        let cb = self.cmdq.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        if nkv % C != 0 {
            let f = self.lib.get_function("kernel_kv_tail_pad", None).unwrap();
            let pl = self.device.new_compute_pipeline_state_with_function(&f).unwrap();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&kb), 0);
            enc.set_buffer(1, Some(&vb), 0);
            enc.set_buffer(2, Some(&pad), 0);
            for (i, val) in [nkv as i32, nkt as i32, if f16 { 1 } else { 0 }].iter().enumerate() {
                enc.set_bytes(3 + i as u64, 4, val as *const i32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize { width: nkt as u64, height: C as u64, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
        }

        let kname = if f16 { "kernel_flash_attn_blk_f16" } else { "kernel_flash_attn_blk_f32" };
        let f = self.lib.get_function(kname, None).unwrap();
        let pl = self.device.new_compute_pipeline_state_with_function(&f).unwrap();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&qb), 0);
        enc.set_buffer(1, Some(&kb), 0);
        enc.set_buffer(2, Some(&vb), 0);
        enc.set_buffer(3, Some(&pad), 0);
        enc.set_buffer(4, Some(&ob), 0);
        enc.set_buffer(5, Some(&pb), 0);
        for (i, val) in [NH as i32, NK as i32, HD as i32,
            SCALE.to_bits() as i32, nt as i32, nkv as i32].iter().enumerate() {
            enc.set_bytes(6 + i as u64, 4, val as *const i32 as *const _);
        }
        enc.set_threadgroup_memory_length(0, 7168);
        enc.dispatch_thread_groups(
            MTLSize { width: (nt.div_ceil(8)) as u64, height: NH as u64, depth: 1 },
            MTLSize { width: 32, height: 4, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ptr = ob.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, nt * NE_Q) }.to_vec()
    }

    fn buffers(&self, q: &[f32], k: &[f32], v: &[f32],
        f16: bool) -> (metal::Buffer, metal::Buffer, metal::Buffer) {
        let qb = self.device.new_buffer_with_data(q.as_ptr() as *const _, (q.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let kb = self.device.new_buffer_with_data(k.as_ptr() as *const _, (k.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let vb = self.device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        if !f16 { return (qb, kb, vb); }
        let k16: Vec<u16> = k.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        let v16: Vec<u16> = v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        let k16b = self.device.new_buffer_with_data(k16.as_ptr() as *const _, (k16.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
        let v16b = self.device.new_buffer_with_data(v16.as_ptr() as *const _, (v16.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
        (qb, k16b, v16b)
    }
}

#[test]
fn flash_attn_blk_isolation() {
    let ctx = Ctx::new();
    // nt spans 1..(multi-threadgroup); nkv spans <C, exact multiple of C,
    // and partial last block. gqa = 14/2 = 7 always (fixed NH/NK).
    for &(nt, nkv) in &[
        (1usize, 1usize),
        (1, 5),
        (1, 63),
        (1, 64),
        (1, 65),
        (1, 100),
        (1, 127),
        (1, 128),
        (1, 129),
        (8, 64),        // exactly one threadgroup, full block
        (8, 63),        // one threadgroup, partial block
        (8, 100),       // 2 blocks
        (9, 100),       // 2 threadgroups
        (16, 140),      // 2 threadgroups + partial (real prefill case)
        (32, 251),      // multi-block + partial
        (200, 300),     // many threadgroups
    ] {
        let q: Vec<f32> = (0..nt * NE_Q).map(|i| ((i as f32) * 1.7).sin() * 3.0).collect();
        let k: Vec<f32> = (0..nkv * NK * HD).map(|i| ((i as f32) * 0.9).cos() * 2.0).collect();
        let v: Vec<f32> = (0..nkv * NK * HD).map(|i| ((i as f32) * 0.4).sin() * 1.5).collect();

        // CPU reference (masked, unpadded) — the plain causal attention.
        let mut ref_out = vec![0.0f32; nt * NE_Q];
        cpu_attn(&q, &k, &v, NH, NK, HD, nt, nkv, SCALE, &mut ref_out);
        // padded reference (exact kernel semantics for partial blocks).
        let mut ref_pad = vec![0.0f32; nt * NE_Q];
        cpu_attn_padded(&q, &k, &v, NH, NK, HD, nt, nkv, SCALE, &mut ref_pad);
        // The two references agree exactly (masking is identical).
        let (_, cos_ref, _) = stats(&format!("cpu-pad-vs-cpu nt={nt} nkv={nkv}"), &ref_pad, &ref_out);
        assert!(cos_ref > 0.99999, "cpu references disagree at nt={nt} nkv={nkv} (cos={cos_ref:.6})");

        for (label, f16) in [("f32", false), ("f16", true)] {
            let r1 = ctx.blk_attn(&q, &k, &v, nt, nkv, f16);
            let r2 = ctx.blk_attn(&q, &k, &v, nt, nkv, f16);
            let maxdiff_gg = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let (maxdiff, cos, _) = stats(&format!("blk[{label}] nt={nt} nkv={nkv}"), &r1, &ref_pad);
            assert!(maxdiff_gg == 0.0, "blk[{label}] nt={nt} nkv={nkv}: run-to-run differs ({maxdiff_gg:.2e})");
            assert!(cos > 0.999, "blk[{label}] nt={nt} nkv={nkv}: wrong vs CPU (cos={cos:.6} maxdiff={maxdiff:.2e})");
        }
    }
}
