// Isolation tests for the ported llama flash-attention kernels
// (kernel_flash_attn_ext_f32 / kernel_flash_attn_ext_f16, NSG=1 fixed-shape):
// verify correctness vs a scalar CPU reference (online-softmax flash math),
// including partial KV chunks (nkv % 32 != 0), empty chunks (nkv < C or
// n_chunks > nkv/C), multiple tokens, and the whole KV range a threadgroup
// spans across its strided chunk loop. Also A/B the flash path against the
// split-attention path (kernel_gqa_attn_partial_f32 + combine) to confirm the
// two chunking schemes agree through the shared combine kernel. macOS only.
#![cfg(target_os = "macos")]

use metal::{MTLResourceOptions, MTLSize};

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
    nh: usize,
    nk: usize,
    hd: usize,
    scale: f32,
    ne_q: usize,
}

impl Ctx {
    fn new() -> Self {
        let device = metal::Device::system_default().expect("no metal device");
        let src = include_str!("../src/metal.metal");
        let opts = metal::CompileOptions::new();
        let lib = device.new_library_with_source(src, &opts).unwrap_or_else(|e| panic!("shader compile: {e}"));
        let cmdq = device.new_command_queue();
        let (nh, nk, hd) = (14usize, 2usize, 64usize);
        let scale = 1.0 / (hd as f32).sqrt();
        Ctx { device, lib, cmdq, nh, nk, hd, scale, ne_q: nh * hd }
    }

    /// flash pass 1 (kernel_flash_attn_ext_f32/_f16, grid (nt, nh, n_chunks),
    /// 32 threads) then pass 2 (kernel_gqa_attn_combine_f32).
    fn flash_attn(&self, q: &[f32], k: &[f32], v: &[f32], nt: usize, nkv: usize,
        n_chunks: usize, f16: bool) -> Vec<f32> {
        let (qb, kb, vb) = self.buffers(q, k, v, nt, nkv, f16);
        let positions: Vec<i32> = (0..nt as i32).collect();
        let pb = self.device.new_buffer_with_data(positions.as_ptr() as *const _, (positions.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let partial = self.device.new_buffer((nt * self.nh * n_chunks * (2 + self.hd) * 4) as u64, MTLResourceOptions::StorageModeShared);
        let ob = self.device.new_buffer((nt * self.ne_q * 4) as u64, MTLResourceOptions::StorageModeShared);

        let kname = if f16 { "kernel_flash_attn_ext_f16" } else { "kernel_flash_attn_ext_f32" };
        let f = self.lib.get_function(kname, None).unwrap();
        let pl = self.device.new_compute_pipeline_state_with_function(&f).unwrap();
        let f_c = self.lib.get_function("kernel_gqa_attn_combine_f32", None).unwrap();
        let pl_c = self.device.new_compute_pipeline_state_with_function(&f_c).unwrap();

        let cb = self.cmdq.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&qb), 0);
        enc.set_buffer(1, Some(&kb), 0);
        enc.set_buffer(2, Some(&vb), 0);
        enc.set_buffer(3, Some(&partial), 0);
        enc.set_buffer(4, Some(&pb), 0);
        for (i, val) in [self.nh as i32, self.nk as i32, self.hd as i32,
            self.scale.to_bits() as i32, nt as i32, n_chunks as i32].iter().enumerate() {
            enc.set_bytes(5 + i as u64, 4, val as *const i32 as *const _);
        }
        enc.set_threadgroup_memory_length(0, 1024);
        enc.dispatch_thread_groups(
            MTLSize { width: nt as u64, height: self.nh as u64, depth: n_chunks as u64 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.set_compute_pipeline_state(&pl_c);
        enc.set_buffer(0, Some(&partial), 0);
        enc.set_buffer(1, Some(&ob), 0);
        for (i, val) in [self.nh as i32, self.hd as i32, nt as i32, n_chunks as i32].iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, val as *const i32 as *const _);
        }
        enc.dispatch_thread_groups(
            MTLSize { width: nt as u64, height: self.nh as u64, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ptr = ob.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, nt * self.ne_q) }.to_vec()
    }

    /// split path (kernel_gqa_attn_partial_f32 + combine) — the current
    /// production attention, used as the flash-vs-split A/B reference.
    fn split_attn(&self, q: &[f32], k: &[f32], v: &[f32], nt: usize, nkv: usize,
        n_chunks: usize, f16: bool) -> Vec<f32> {
        let (qb, kb, vb) = self.buffers(q, k, v, nt, nkv, f16);
        let positions: Vec<i32> = (0..nt as i32).collect();
        let pb = self.device.new_buffer_with_data(positions.as_ptr() as *const _, (positions.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let partial = self.device.new_buffer((nt * self.nh * n_chunks * (2 + self.hd) * 4) as u64, MTLResourceOptions::StorageModeShared);
        let ob = self.device.new_buffer((nt * self.ne_q * 4) as u64, MTLResourceOptions::StorageModeShared);
        let gqa = self.nh / self.nk;

        let kname = if f16 { "kernel_gqa_attn_partial_f16" } else { "kernel_gqa_attn_partial_f32" };
        let f = self.lib.get_function(kname, None).unwrap();
        let pl = self.device.new_compute_pipeline_state_with_function(&f).unwrap();
        let f_c = self.lib.get_function("kernel_gqa_attn_combine_f32", None).unwrap();
        let pl_c = self.device.new_compute_pipeline_state_with_function(&f_c).unwrap();

        let cb = self.cmdq.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&qb), 0);
        enc.set_buffer(1, Some(&kb), 0);
        enc.set_buffer(2, Some(&vb), 0);
        enc.set_buffer(3, Some(&partial), 0);
        enc.set_buffer(4, Some(&pb), 0);
        for (i, val) in [self.nh as i32, self.nk as i32, self.hd as i32,
            self.scale.to_bits() as i32, nt as i32, n_chunks as i32].iter().enumerate() {
            enc.set_bytes(5 + i as u64, 4, val as *const i32 as *const _);
        }
        let shmem = (32 * self.hd * 2 * 4) as u64;
        enc.set_threadgroup_memory_length(0, shmem);
        enc.dispatch_thread_groups(
            MTLSize { width: nt as u64, height: self.nk as u64, depth: n_chunks as u64 },
            MTLSize { width: 32, height: gqa as u64, depth: 1 },
        );
        enc.set_compute_pipeline_state(&pl_c);
        enc.set_buffer(0, Some(&partial), 0);
        enc.set_buffer(1, Some(&ob), 0);
        for (i, val) in [self.nh as i32, self.hd as i32, nt as i32, n_chunks as i32].iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, val as *const i32 as *const _);
        }
        enc.dispatch_thread_groups(
            MTLSize { width: nt as u64, height: self.nh as u64, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ptr = ob.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, nt * self.ne_q) }.to_vec()
    }

    fn buffers(&self, q: &[f32], k: &[f32], v: &[f32], nt: usize, nkv: usize,
        f16: bool) -> (metal::Buffer, metal::Buffer, metal::Buffer) {
        let qb = self.device.new_buffer_with_data(q.as_ptr() as *const _, (q.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        if f16 {
            let k16: Vec<u16> = k.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
            let v16: Vec<u16> = v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
            let k16b = self.device.new_buffer_with_data(k16.as_ptr() as *const _, (k16.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
            let v16b = self.device.new_buffer_with_data(v16.as_ptr() as *const _, (v16.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
            (qb, k16b, v16b)
        } else {
            let kb = self.device.new_buffer_with_data(k.as_ptr() as *const _, (k.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
            let vb = self.device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
            (qb, kb, vb)
        }
    }
}

#[test]
fn flash_attn_ext_isolation() {
    let ctx = Ctx::new();
    // n_chunks covers: 1 (whole KV in one TG), small, and > ceil(nkv/C)
    // (empty chunks) cases; nkv spans 1..N tokens incl. partial chunks.
    for &(nt, nkv, n_chunks) in &[
        (1usize, 1usize, 4usize),
        (1, 5, 4),        // nkv < C -> single partial chunk + empty chunks
        (1, 30, 4),       // partial chunk
        (1, 33, 4),       // partial second chunk
        (1, 64, 4),       // exactly 2 full chunks
        (1, 65, 4),       // 2 full + partial
        (1, 100, 1),      // degenerate: whole KV in one threadgroup
        (1, 100, 2),
        (1, 100, 4),
        (1, 100, 8),
        (1, 100, 32),     // > ceil(100/32)=4 -> empty chunks
        (1, 2510, 16),    // long-context decode
        (1, 4097, 32),    // long KV, partial last chunk
        (2, 100, 4),      // multi-token
        (2, 33, 2),       // multi-token with partial chunks
    ] {
        let q: Vec<f32> = (0..nt * ctx.ne_q).map(|i| ((i as f32) * 1.7).sin() * 3.0).collect();
        let k: Vec<f32> = (0..nkv * ctx.nk * ctx.hd).map(|i| ((i as f32) * 0.9).cos() * 2.0).collect();
        let v: Vec<f32> = (0..nkv * ctx.nk * ctx.hd).map(|i| ((i as f32) * 0.4).sin() * 1.5).collect();
        let mut ref_out = vec![0.0f32; nt * ctx.ne_q];
        cpu_attn(&q, &k, &v, ctx.nh, ctx.nk, ctx.hd, nt, nkv, ctx.scale, &mut ref_out);

        for (label, f16) in [("f32", false), ("f16", true)] {
            let r1 = ctx.flash_attn(&q, &k, &v, nt, nkv, n_chunks, f16);
            let r2 = ctx.flash_attn(&q, &k, &v, nt, nkv, n_chunks, f16);
            let maxdiff_gg = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let (maxdiff, cos, _) = stats(&format!("flash[{label}] nt={nt} nkv={nkv} nc={n_chunks}"), &r1, &ref_out);
            assert!(maxdiff_gg == 0.0, "flash[{label}] nt={nt} nkv={nkv} nc={n_chunks}: run-to-run differs ({maxdiff_gg:.2e})");
            assert!(cos > 0.999, "flash[{label}] nt={nt} nkv={nkv} nc={n_chunks}: wrong vs CPU (cos={cos:.6} maxdiff={maxdiff:.2e})");
        }
    }
}

/// A/B the flash path against the production split path through the SHARED
/// combine kernel — confirms the two different chunking schemes (strided
/// C=32 blocks vs contiguous ceil(nkv/n_chunks) ranges) produce the same
/// partials and merge identically.
#[test]
fn flash_attn_matches_split() {
    let ctx = Ctx::new();
    for &(nt, nkv, n_chunks) in &[
        (1usize, 1usize, 4usize),
        (1, 30, 4),
        (1, 33, 4),
        (1, 65, 4),
        (1, 100, 4),
        (1, 100, 8),
        (1, 5, 4),
        (1, 2510, 16),
        (1, 4097, 32),
        (2, 100, 4),
    ] {
        let q: Vec<f32> = (0..nt * ctx.ne_q).map(|i| ((i as f32) * 1.7).sin() * 3.0).collect();
        let k: Vec<f32> = (0..nkv * ctx.nk * ctx.hd).map(|i| ((i as f32) * 0.9).cos() * 2.0).collect();
        let v: Vec<f32> = (0..nkv * ctx.nk * ctx.hd).map(|i| ((i as f32) * 0.4).sin() * 1.5).collect();
        for (label, f16) in [("f32", false), ("f16", true)] {
            let flash = ctx.flash_attn(&q, &k, &v, nt, nkv, n_chunks, f16);
            let split = ctx.split_attn(&q, &k, &v, nt, nkv, n_chunks, f16);
            let (maxdiff, cos, _) = stats(&format!("flash-vs-split[{label}] nt={nt} nkv={nkv} nc={n_chunks}"), &flash, &split);
            assert!(cos > 0.9999, "flash[{label}] nt={nt} nkv={nkv} nc={n_chunks}: disagrees with split (cos={cos:.6} maxdiff={maxdiff:.2e})");
        }
    }
}
