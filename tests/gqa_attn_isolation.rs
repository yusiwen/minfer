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
        // f32 KV buffers (kernel_gqa_attn_f32)
        let kb = device.new_buffer_with_data(k.as_ptr() as *const _, (k.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let vb = device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        // f16 KV buffers (kernel_gqa_attn_f16)
        let k16: Vec<u16> = k.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        let v16: Vec<u16> = v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        let k16b = device.new_buffer_with_data(k16.as_ptr() as *const _, (k16.len() * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let v16b = device.new_buffer_with_data(v16.as_ptr() as *const _, (v16.len() * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let pb = device.new_buffer_with_data(positions.as_ptr() as *const _, (positions.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

        let mut ref_out = vec![0.0f32; nt * ne_q];
        cpu_attn(&q, &k, &v, nh, nk, hd, nt, nkv, scale, &mut ref_out);

        // Test both the f32-KV and f16-KV attention kernels.
        for (kernel_name, kb_ref, vb_ref) in [
            ("kernel_gqa_attn_f32", &kb, &vb),
            ("kernel_gqa_attn_f16", &k16b, &v16b),
        ] {
            let f = lib.get_function(kernel_name, None).unwrap();
            let pl = device.new_compute_pipeline_state_with_function(&f).unwrap();
            let run = |cmdq: &metal::CommandQueue| -> Vec<f32> {
                let ob = device.new_buffer((nt * ne_q * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
                let cb = cmdq.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pl);
                enc.set_buffer(0, Some(&qb), 0);
                enc.set_buffer(1, Some(kb_ref), 0);
                enc.set_buffer(2, Some(vb_ref), 0);
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
            println!("{kernel_name} nt={nt} nkv={nkv}: deterministic={} maxdiff_ref={maxdiff_ref:.2e} cos={cos:.6}", maxdiff_gg == 0.0);
            assert!(maxdiff_gg == 0.0, "{kernel_name} nt={nt} nkv={nkv}: attention non-deterministic");
            assert!(cos > 0.999, "{kernel_name} nt={nt} nkv={nkv}: attention wrong vs CPU (cos={cos:.6})");
        }
    }
}

/// Isolation test for the KV-parallel split attention (kernel_gqa_attn_partial_f32
/// + kernel_gqa_attn_combine_f32): nt==1, several nkv (incl. partial tiles and
/// empty chunks when nkv < n_chunks) and several n_chunks. Deterministic and
/// compared vs the scalar CPU reference (must match within online-softmax fp error).
#[test]
fn gqa_attn_split_isolation() {
    let device = metal::Device::system_default().expect("no metal device");
    let src = include_str!("../src/metal.metal");
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(src, &opts).unwrap_or_else(|e| panic!("shader compile: {e}"));
    let f_p = lib.get_function("kernel_gqa_attn_partial_f32", None).unwrap();
    let pl_p = device.new_compute_pipeline_state_with_function(&f_p).unwrap();
    let f_p16 = lib.get_function("kernel_gqa_attn_partial_f16", None).unwrap();
    let pl_p16 = device.new_compute_pipeline_state_with_function(&f_p16).unwrap();
    let f_c = lib.get_function("kernel_gqa_attn_combine_f32", None).unwrap();
    let pl_c = device.new_compute_pipeline_state_with_function(&f_c).unwrap();

    let (nh, nk, hd) = (14usize, 2usize, 64usize);
    let gqa = nh / nk;
    let scale = 1.0 / (hd as f32).sqrt();
    let ne_q = nh * hd;

    let cmdq = device.new_command_queue();
    for &(nt, nkv, n_chunks) in &[
        (1usize, 1usize, 4usize),
        (1, 30, 4),
        (1, 33, 4),
        (1, 65, 4),
        (1, 100, 4),
        (1, 240, 4),
        (1, 100, 1),   // degenerate: single chunk == classic kernel
        (1, 100, 2),
        (1, 100, 8),
        (1, 5, 4),     // nkv < n_chunks -> empty chunks
        (2, 100, 4),
        // long-context coverage (the split targets decode at growing KV)
        (1, 1000, 8),
        (1, 2500, 16),
        (1, 4000, 32),
        (1, 4097, 32), // %32 != 0 partial tile at long KV
    ] {
        let q: Vec<f32> = (0..nt * ne_q).map(|i| ((i as f32) * 1.7).sin() * 3.0).collect();
        let k: Vec<f32> = (0..nkv * nk * hd).map(|i| ((i as f32) * 0.9).cos() * 2.0).collect();
        let v: Vec<f32> = (0..nkv * nk * hd).map(|i| ((i as f32) * 0.4).sin() * 1.5).collect();
        let positions: Vec<i32> = (0..nt as i32).collect();
        let mut ref_out = vec![0.0f32; nt * ne_q];
        cpu_attn(&q, &k, &v, nh, nk, hd, nt, nkv, scale, &mut ref_out);

        let qb = device.new_buffer_with_data(q.as_ptr() as *const _, (q.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let kb = device.new_buffer_with_data(k.as_ptr() as *const _, (k.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let vb = device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let k16: Vec<u16> = k.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        let v16: Vec<u16> = v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        let k16b = device.new_buffer_with_data(k16.as_ptr() as *const _, (k16.len() * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let v16b = device.new_buffer_with_data(v16.as_ptr() as *const _, (v16.len() * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let pb = device.new_buffer_with_data(positions.as_ptr() as *const _, (positions.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

        let run = |cmdq: &metal::CommandQueue, partial_pl: &metal::ComputePipelineState,
                   kb_ref: &metal::Buffer, vb_ref: &metal::Buffer| -> Vec<f32> {
            let partial = device.new_buffer((nt * nh * n_chunks * (2 + hd) * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
            let ob = device.new_buffer((nt * ne_q * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

            // pass 1: partials
            {
                let cb = cmdq.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(partial_pl);
                enc.set_buffer(0, Some(&qb), 0);
                enc.set_buffer(1, Some(kb_ref), 0);
                enc.set_buffer(2, Some(vb_ref), 0);
                enc.set_buffer(3, Some(&partial), 0);
                enc.set_buffer(4, Some(&pb), 0);
                for (i, val) in [nh as i32, nk as i32, hd as i32, scale.to_bits() as i32, nt as i32, n_chunks as i32].iter().enumerate() {
                    enc.set_bytes(5 + i as u64, 4, val as *const i32 as *const _);
                }
                let shmem = (32 * hd * 2 * 4) as u64;
                enc.set_threadgroup_memory_length(0, shmem);
                enc.dispatch_thread_groups(
                    metal::MTLSize { width: nt as u64, height: nk as u64, depth: n_chunks as u64 },
                    metal::MTLSize { width: 32, height: gqa as u64, depth: 1 },
                );
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            // pass 2: combine (f32, shared between f32 and f16 caches)
            {
                let cb = cmdq.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pl_c);
                enc.set_buffer(0, Some(&partial), 0);
                enc.set_buffer(1, Some(&ob), 0);
                for (i, val) in [nh as i32, hd as i32, nt as i32, n_chunks as i32].iter().enumerate() {
                    enc.set_bytes(2 + i as u64, 4, val as *const i32 as *const _);
                }
                enc.dispatch_thread_groups(
                    metal::MTLSize { width: nt as u64, height: nh as u64, depth: 1 },
                    metal::MTLSize { width: 32, height: 1, depth: 1 },
                );
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let ptr = ob.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, nt * ne_q) }.to_vec()
        };

        for (label, pl, kb_ref, vb_ref) in [
            ("f32", &pl_p, &kb, &vb),
            ("f16", &pl_p16, &k16b, &v16b),
        ] {
            let r1 = run(&cmdq, pl, kb_ref, vb_ref);
            let r2 = run(&cmdq, pl, kb_ref, vb_ref);
            let maxdiff_gg = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let dot: f64 = r1.iter().zip(&ref_out).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let n1: f64 = r1.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let n2: f64 = ref_out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (n1 * n2);
            let maxdiff_ref = r1.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("split[{label}] nt={nt} nkv={nkv} n_chunks={n_chunks}: deterministic={} maxdiff_ref={maxdiff_ref:.2e} cos={cos:.6}", maxdiff_gg == 0.0);
            assert!(maxdiff_gg == 0.0, "split[{label}] nt={nt} nkv={nkv} nc={n_chunks}: non-deterministic");
            assert!(cos > 0.999, "split[{label}] nt={nt} nkv={nkv} nc={n_chunks}: wrong vs CPU (cos={cos:.6})");
        }
    }
}

/// Diagnostic (not a pass/fail): batched GPU time of the split attention
/// (partial + combine in ONE command buffer, like real decode) at several KV
/// lengths, to separate the kernel GPU time from per-token sync/pipeline
/// overhead. Run with --nocapture.
#[test]
fn gqa_attn_split_timing() {
    let device = metal::Device::system_default().expect("no metal device");
    let src = include_str!("../src/metal.metal");
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(src, &opts).unwrap_or_else(|e| panic!("shader: {e}"));
    let f_p = lib.get_function("kernel_gqa_attn_partial_f32", None).unwrap();
    let pl_p = device.new_compute_pipeline_state_with_function(&f_p).unwrap();
    let f_c = lib.get_function("kernel_gqa_attn_combine_f32", None).unwrap();
    let pl_c = device.new_compute_pipeline_state_with_function(&f_c).unwrap();

    let (nh, nk, hd) = (14usize, 2usize, 64usize);
    let gqa = nh / nk;
    let scale = 1.0 / (hd as f32).sqrt();
    let ne_q = nh * hd;
    let cmdq = device.new_command_queue();

    for &(nkv, n_chunks) in &[
        (140usize, 8usize),
        (2510usize, 8usize), (2510usize, 16usize), (2510usize, 32usize),
        (4000usize, 8usize), (4000usize, 16usize), (4000usize, 32usize),
    ] {
        let q: Vec<f32> = (0..ne_q).map(|i| ((i as f32) * 1.7).sin() * 3.0).collect();
        let k: Vec<f32> = (0..nkv * nk * hd).map(|i| ((i as f32) * 0.9).cos() * 2.0).collect();
        let v: Vec<f32> = (0..nkv * nk * hd).map(|i| ((i as f32) * 0.4).sin() * 1.5).collect();
        let positions: Vec<i32> = vec![nkv as i32 - 1];
        let qb = device.new_buffer_with_data(q.as_ptr() as *const _, (q.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let kb = device.new_buffer_with_data(k.as_ptr() as *const _, (k.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let vb = device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let pb = device.new_buffer_with_data(positions.as_ptr() as *const _, 4, metal::MTLResourceOptions::StorageModeShared);
        let partial = device.new_buffer((nh * n_chunks * (2 + hd) * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let ob = device.new_buffer((ne_q * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let shmem = (32 * hd * 2 * 4) as u64;

        let run = || {
            let cb = cmdq.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl_p);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kb), 0);
            enc.set_buffer(2, Some(&vb), 0);
            enc.set_buffer(3, Some(&partial), 0);
            enc.set_buffer(4, Some(&pb), 0);
            for (i, val) in [nh as i32, nk as i32, hd as i32, scale.to_bits() as i32, 1, n_chunks as i32].iter().enumerate() {
                enc.set_bytes(5 + i as u64, 4, val as *const i32 as *const _);
            }
            enc.set_threadgroup_memory_length(0, shmem);
            enc.dispatch_thread_groups(
                metal::MTLSize { width: 1, height: nk as u64, depth: n_chunks as u64 },
                metal::MTLSize { width: 32, height: gqa as u64, depth: 1 },
            );
            enc.set_compute_pipeline_state(&pl_c);
            enc.set_buffer(0, Some(&partial), 0);
            enc.set_buffer(1, Some(&ob), 0);
            for (i, val) in [nh as i32, hd as i32, 1, n_chunks as i32].iter().enumerate() {
                enc.set_bytes(2 + i as u64, 4, val as *const i32 as *const _);
            }
            enc.dispatch_thread_groups(
                metal::MTLSize { width: 1, height: nh as u64, depth: 1 },
                metal::MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        run();
        // MANY attention passes in ONE command buffer to amortize the per-cb
        // launch+sync overhead (single-cb timings are ~165 us dominated).
        let passes = 100;
        let cb = cmdq.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..passes {
            enc.set_compute_pipeline_state(&pl_p);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kb), 0);
            enc.set_buffer(2, Some(&vb), 0);
            enc.set_buffer(3, Some(&partial), 0);
            enc.set_buffer(4, Some(&pb), 0);
            for (i, val) in [nh as i32, nk as i32, hd as i32, scale.to_bits() as i32, 1, n_chunks as i32].iter().enumerate() {
                enc.set_bytes(5 + i as u64, 4, val as *const i32 as *const _);
            }
            enc.set_threadgroup_memory_length(0, shmem);
            enc.dispatch_thread_groups(
                metal::MTLSize { width: 1, height: nk as u64, depth: n_chunks as u64 },
                metal::MTLSize { width: 32, height: gqa as u64, depth: 1 },
            );
            enc.set_compute_pipeline_state(&pl_c);
            enc.set_buffer(0, Some(&partial), 0);
            enc.set_buffer(1, Some(&ob), 0);
            for (i, val) in [nh as i32, hd as i32, 1, n_chunks as i32].iter().enumerate() {
                enc.set_bytes(2 + i as u64, 4, val as *const i32 as *const _);
            }
            enc.dispatch_thread_groups(
                metal::MTLSize { width: 1, height: nh as u64, depth: 1 },
                metal::MTLSize { width: 32, height: 1, depth: 1 },
            );
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let iters = 3;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let cb = cmdq.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for _ in 0..passes {
                enc.set_compute_pipeline_state(&pl_p);
                enc.set_buffer(0, Some(&qb), 0);
                enc.set_buffer(1, Some(&kb), 0);
                enc.set_buffer(2, Some(&vb), 0);
                enc.set_buffer(3, Some(&partial), 0);
                enc.set_buffer(4, Some(&pb), 0);
                for (i, val) in [nh as i32, nk as i32, hd as i32, scale.to_bits() as i32, 1, n_chunks as i32].iter().enumerate() {
                    enc.set_bytes(5 + i as u64, 4, val as *const i32 as *const _);
                }
                enc.set_threadgroup_memory_length(0, shmem);
                enc.dispatch_thread_groups(
                    metal::MTLSize { width: 1, height: nk as u64, depth: n_chunks as u64 },
                    metal::MTLSize { width: 32, height: gqa as u64, depth: 1 },
                );
                enc.set_compute_pipeline_state(&pl_c);
                enc.set_buffer(0, Some(&partial), 0);
                enc.set_buffer(1, Some(&ob), 0);
                for (i, val) in [nh as i32, hd as i32, 1, n_chunks as i32].iter().enumerate() {
                    enc.set_bytes(2 + i as u64, 4, val as *const i32 as *const _);
                }
                enc.dispatch_thread_groups(
                    metal::MTLSize { width: 1, height: nh as u64, depth: 1 },
                    metal::MTLSize { width: 32, height: 1, depth: 1 },
                );
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let per = t0.elapsed().as_secs_f64() / (iters as f64 * passes as f64);
        println!("attn timing nkv={nkv} chunks={n_chunks}: {:.4} ms per layer-set (batched x{passes})", per * 1e3);
    }
}
