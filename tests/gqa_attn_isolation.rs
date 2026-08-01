// Isolation test for kernel_gqa_attn_f32 (Flash Attention): verify correctness
// vs a scalar CPU reference, including partial KV tiles (nkv % 32 != 0) which
// previously had a divergent-simd_max bug. macOS only (Metal).
#![cfg(target_os = "macos")]

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

#[test]
fn gqa_attn_isolation() {
    let device = metal::Device::system_default().expect("no metal device");
    let src = include_str!("../src/metal.metal");
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(src, &opts).unwrap_or_else(|e| panic!("shader compile: {e}"));
    let f = lib.get_function("kernel_gqa_attn_f32", None).unwrap();
    let pl = device.new_compute_pipeline_state_with_function(&f).unwrap();

    // Qwen2.5-0.5B attention dims: nh=14, nk=2, hd=64
    let (nh, nk, hd) = (14usize, 2usize, 64usize);
    let gqa = nh / nk;
    let scale = 1.0 / (hd as f32).sqrt();
    let ne_q = nh * hd;

    let cmdq = device.new_command_queue();
    // nkv values that span 1..N tiles incl. partial tiles (the divergent
    // simd_max bug affected tokens whose KV count % 32 != 0, i.e. nkv>32).
    for &(nt, nkv) in &[(37usize, 37usize), (33, 40), (33, 65), (33, 100)] {
        // large-magnitude data so attention dots (and stale lane registers in the
        // pre-fix kernel) are large and any divergence is observable.
        let q: Vec<f32> = (0..nt * ne_q).map(|i| ((i as f32) * 1.7).sin() * 3.0).collect();
        let k: Vec<f32> = (0..nkv * nk * hd).map(|i| ((i as f32) * 0.9).cos() * 2.0).collect();
        let v: Vec<f32> = (0..nkv * nk * hd).map(|i| ((i as f32) * 0.4).sin() * 1.5).collect();
        let positions: Vec<i32> = (0..nt as i32).collect(); // causal: token t attends to 0..=t

        let qb = device.new_buffer_with_data(q.as_ptr() as *const _, (q.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let kb = device.new_buffer_with_data(k.as_ptr() as *const _, (k.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let vb = device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let pb = device.new_buffer_with_data(positions.as_ptr() as *const _, (positions.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

        let mut ref_out = vec![0.0f32; nt * ne_q];
        cpu_attn(&q, &k, &v, nh, nk, hd, nt, nkv, scale, &mut ref_out);

        let run = |cmdq: &metal::CommandQueue| -> Vec<f32> {
            let ob = device.new_buffer((nt * ne_q * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
            let cb = cmdq.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kb), 0);
            enc.set_buffer(2, Some(&vb), 0);
            enc.set_buffer(3, Some(&ob), 0);
            enc.set_buffer(4, Some(&pb), 0);
            for (i, val) in [nh as i32, nk as i32, hd as i32, scale.to_bits() as i32, nt as i32].iter().enumerate() {
                enc.set_bytes(5 + i as u64, 4, val as *const i32 as *const _);
            }
            let shmem = (32 * hd * 2 * 4) as u64;
            enc.set_threadgroup_memory_length(0, shmem);
            enc.dispatch_thread_groups(
                metal::MTLSize { width: nt as u64, height: nk as u64, depth: 1 },
                metal::MTLSize { width: 32, height: gqa as u64, depth: 1 },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let ptr = ob.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, nt * ne_q) }.to_vec()
        };

        let r1 = run(&cmdq);
        let r2 = run(&cmdq);
        let maxdiff_gg = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let maxdiff_ref = r1.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let dot: f64 = r1.iter().zip(&ref_out).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let n1: f64 = r1.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let n2: f64 = ref_out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (n1 * n2);
        println!("nt={nt} nkv={nkv}: deterministic={} maxdiff_ref={maxdiff_ref:.2e} cos={cos:.6}", maxdiff_gg == 0.0);
        assert!(maxdiff_gg == 0.0, "nt={nt} nkv={nkv}: attention non-deterministic");
        assert!(cos > 0.999, "nt={nt} nkv={nkv}: attention wrong vs CPU (cos={cos:.6})");
    }
}
