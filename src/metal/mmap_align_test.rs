//! `#[cfg(test)] mod mmap_align_test` for `src/metal.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use objc2_foundation::NSString;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLCommandEncoder, MTLCreateSystemDefaultDevice, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::ptr::NonNull;
#[test]
fn nocopy_alignment_probe() {
    let _g = crate::metal::metal_test_lock();
    let dev = MTLCreateSystemDefaultDevice().unwrap();
    // A 16-aligned Vec base + 32 → 32-aligned, NOT 256-aligned
    let mut backing = vec![0u8; 8192 + 64];
    let base = backing.as_mut_ptr() as usize;
    let aligned = (base + 63) & !31usize; // 32-aligned pointer
    assert!(
        aligned % 32 == 0 && aligned % 256 != 0,
        "need a non-256-aligned 32-aligned ptr"
    );
    let buf_slice = unsafe { std::slice::from_raw_parts_mut(aligned as *mut u8, 4096) };
    for i in 0..4096 {
        buf_slice[i] = (i & 0xFF) as u8;
    }
    let b = unsafe {
        dev.newBufferWithBytesNoCopy_length_options_deallocator(
            NonNull::new(aligned as *const std::ffi::c_void as *mut c_void).unwrap(),
            (4096) as usize,
            MTLResourceOptions::StorageModeShared,
            None,
        )
        .unwrap()
    };
    let contents = unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const u8, 4096) };
    let mut ok = contents.len() == 4096;
    for i in 0..4096 {
        if contents[i] != (i & 0xFF) as u8 {
            ok = false;
            break;
        }
    }
    println!("CPU readback: ok={ok}");
    // GPU readback: dispatch a trivial copy kernel reading the buffer
    let lib = dev.newLibraryWithSource_options_error(&*NSString::from_str("kernel void k(const device uchar *in [[buffer(0)]], device uchar *out [[buffer(1)]]) { out[0] = in[3]; }"), None).unwrap();
    let pl = dev
        .newComputePipelineStateWithFunction_error(
            &lib.newFunctionWithName(&*NSString::from_str("k")).unwrap(),
        )
        .unwrap();
    let out = dev
        .newBufferWithLength_options((16) as usize, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let q = dev.newCommandQueue().unwrap();
    let cb = q.commandBuffer().unwrap();
    let enc = cb.computeCommandEncoder().unwrap();
    enc.setComputePipelineState(&*pl);
    unsafe { enc.setBuffer_offset_atIndex(Some(&*b), (0) as usize, (0) as usize) };
    unsafe { enc.setBuffer_offset_atIndex(Some(&*out), (0) as usize, (1) as usize) };
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: (1) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        },
        MTLSize {
            width: (1) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let got = unsafe { *(out.contents().as_ptr() as *const u8) };
    println!(
        "GPU readback (expect 3): got={got} -> {}",
        if got == 3 { "OK" } else { "WRONG" }
    );
    assert!(got == 3, "GPU readback at 32-aligned nocopy base is WRONG");
    // Same probe over an mmap'd FILE region (the actual weights path)
    {
        let path = std::env::temp_dir().join("minfer_mmap_probe.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        use std::io::Write;
        let mut blob = vec![0u8; 8192 + 64];
        for i in 0..blob.len() {
            blob[i] = (i & 0xFF) as u8;
        }
        f.write_all(&blob).unwrap();
        drop(f);
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn mmap(
                a: *mut std::ffi::c_void,
                l: usize,
                p: i32,
                f: i32,
                fd: i32,
                o: i64,
            ) -> *mut std::ffi::c_void;
        }
        let file = std::fs::File::open(&path).unwrap();
        let m = unsafe { mmap(std::ptr::null_mut(), 8192, 0x1, 0x0002, file.as_raw_fd(), 0) };
        assert!(m as isize != -1);
        let mbase = m as usize;
        let mptr = (mbase + 63) & !31usize; // 32-aligned, not page-aligned
        assert!(mptr % 256 != 0, "need non-256-aligned");
        let bm = unsafe {
            dev.newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(mptr as *const std::ffi::c_void as *mut c_void).unwrap(),
                (4096) as usize,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .unwrap()
        };
        let out2 = dev
            .newBufferWithLength_options((64) as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let cb2 = q.commandBuffer().unwrap();
        let enc2 = cb2.computeCommandEncoder().unwrap();
        enc2.setComputePipelineState(&*pl);
        unsafe { enc2.setBuffer_offset_atIndex(Some(&*bm), (0) as usize, (0) as usize) };
        unsafe { enc2.setBuffer_offset_atIndex(Some(&*out2), (0) as usize, (1) as usize) };
        enc2.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (1) as usize,
                height: (1) as usize,
                depth: (1) as usize,
            },
            MTLSize {
                width: (1) as usize,
                height: (1) as usize,
                depth: (1) as usize,
            },
        );
        enc2.endEncoding();
        cb2.commit();
        cb2.waitUntilCompleted();
        let got2 = unsafe { *(out2.contents().as_ptr() as *const u8) };
        println!(
            "mmap GPU readback (expect 3): got={got2} -> {}",
            if got2 == 3 { "OK" } else { "WRONG" }
        );
        let _ = std::fs::remove_file(&path);
    }
}

// ─── Fixture generator for attn_parallel_realdata_correctness ───────────
// The realdata test consumes layer-0 q/k/v dumps that the (now deleted)
// layer_gpu dump path used to write. This generator rebuilds them with the
// current graph path: a real 35-token prefill on the cached 0.5B q4_0,
// dumping the layer-0 q/k/v matmul outputs (pre-RoPE, token-major) as f32
// files. Run once with:
//   cargo test --bin minfer gen_layer0_realdata_dump -- --ignored
// (writes to $MINFER_TEST_DUMP or /tmp/dp3, matching the test's default).
fn cached_qwen05_q4_0() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn write_f32(path: &str, data: &[f32]) {
    let mut b = Vec::with_capacity(data.len() * 4);
    for x in data {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).expect("write dump");
}

#[test]
#[ignore = "one-time fixture generator (writes /tmp/dp3 files)"]
fn gen_layer0_realdata_dump() {
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::scheduler::BackendScheduler;
    use crate::graph::DType;
    use crate::models::qwen2::graph::Qwen2Graph;
    use crate::models::qwen2::Qwen2Model;
    use crate::models::ModelDef;

    let Some(path) = cached_qwen05_q4_0() else {
        eprintln!("0.5B q4_0 not cached; skipping fixture generator");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    const NT: usize = 35; // the realdata test asserts nt == 35
    let mut ids = tok.encode(
        "The capital of France is Paris and the capital of Germany is Berlin. \
         The capital of Italy is Rome and the capital of Spain is Madrid and \
         the capital of the United Kingdom is London and the capital of Japan is Tokyo.",
    );
    assert!(
        ids.len() >= NT,
        "prompt tokenizes to {} < {NT} tokens",
        ids.len()
    );
    ids.truncate(NT);

    let hp = &q2.hparams;
    let nh = hp.n_head as usize;
    let nk = hp.n_head_kv as usize;
    let hd = hp.n_embd_head() as usize;
    let nqt = nh * hd;
    let nkt = nk * hd;
    let l0 = &q2.layers[0];

    // embedding -> rms_norm -> q/k/v matmul. No RoPE on purpose: the
    // fixtures are pre-RoPE projections (the attention kernel and the CPU
    // reference in the realdata test don't apply RoPE).
    let mut b = GraphBuilder::new();
    let ids_n = b.input("token_ids", [NT, 1, 1, 1], DType::I32);
    let h = b.embedding(ids_n, q2.tok_embd.as_ref().unwrap());
    let normed = b.rms_norm(h, l0.attn_norm.as_ref(), hp.f_norm_rms_eps);
    let qn = b.matmul(normed, l0.wq.as_ref().unwrap(), l0.bq.as_ref());
    let kn = b.matmul(normed, l0.wk.as_ref().unwrap(), l0.bk.as_ref());
    let vn = b.matmul(normed, l0.wv.as_ref().unwrap(), l0.bv.as_ref());
    b.output(qn);
    b.output(kn);
    b.output(vn);
    let mut graph = b.build();

    let mut alloc = GraphAllocator::new();
    Qwen2Graph::register_graph_weights(q2, &mut alloc);
    let sched = BackendScheduler::new();
    sched.assign_backends(&mut graph, &mut alloc);
    alloc.alloc_graph(&graph).unwrap();
    alloc.fill_input_i32(&graph, "token_ids", &ids).unwrap();
    sched.execute(&graph, &mut alloc).unwrap();

    let q = alloc.copy_to_cpu(qn).expect("q buffer");
    let k = alloc.copy_to_cpu(kn).expect("k buffer");
    let v = alloc.copy_to_cpu(vn).expect("v buffer");
    assert_eq!(q.len(), NT * nqt, "q dims");
    assert_eq!(k.len(), NT * nkt, "k dims");
    assert_eq!(v.len(), NT * nkt, "v dims");
    eprintln!(
        "[gen dump] layer0 q/k/v: {NT}x{nqt} / {NT}x{nkt} / {NT}x{nkt} (q[0]={})",
        q[0]
    );

    let dir = std::env::var("MINFER_TEST_DUMP").unwrap_or_else(|_| "/tmp/dp3".into());
    std::fs::create_dir_all(&dir).unwrap();
    write_f32(&format!("{dir}/minfer_gpu_dump_layer0_bq.f32"), &q);
    write_f32(&format!("{dir}/minfer_gpu_dump_layer0_bk.f32"), &k);
    write_f32(&format!("{dir}/minfer_gpu_dump_layer0_bv.f32"), &v);
    eprintln!("[gen dump] wrote {dir}/minfer_gpu_dump_layer0_b{{q,k,v}}.f32");
}

/// Isolation test for `kernel_attn_rope_store` (the Qwen3 fused decode
/// QKV rope+store pass, no attention biases) against a scalar CPU
/// reference. Exercises the q-rope (in place), k-rope+store and v-store
/// sections of the concat q|k|v buffer, for both the f32 and f16 KV cache
/// paths (the store type comes from `kv_cache_is_f16()`).
#[test]
fn attn_rope_store_isolated() {
    let _g = crate::metal::metal_test_lock();
    MpsState::init();
    let mps = MpsState::get().expect("MPS must be active");
    let dev = &mps.inner.device;
    let nqt = 16usize; // nh*hd = 2*8
    let nkt = 8usize; // nk*hd = 1*8
    let hd = 8usize;
    let nh = 2usize;
    let nk = 1usize;
    let pos = 5i32;
    let freq_base = 10000.0f32;
    let freq_scale = 1.0f32;
    let rope_style = 0i32; // NonInterleaved (Qwen3)
    let kv_f16 = crate::metal::kv_cache_is_f16();
    let total = nqt + 2 * nkt;

    let bqkv = dev
        .newBufferWithLength_options((total * 4) as usize, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let kv_k = dev
        .newBufferWithLength_options(
            ((pos as usize + 1) * nkt * if kv_f16 { 2 } else { 4 }) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let kv_v = dev
        .newBufferWithLength_options(
            ((pos as usize + 1) * nkt * if kv_f16 { 2 } else { 4 }) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    // deterministic q|k|v: sin over index with a per-section offset. Keep a
    // host copy (`orig`) to compute the CPU reference against the PRE-kernel
    // values (the kernel applies the rope in place on the concat buffer).
    let mut orig = vec![0.0f32; total];
    unsafe {
        let p = bqkv.contents().as_ptr() as *mut f32;
        for i in 0..total {
            let v = ((i as f32) * 0.37).sin() + (i as f32) * 0.001;
            *p.add(i) = v;
            orig[i] = v;
        }
    }

    let cb = mps.cmd_buffer();
    cb.attn_rope_store(
        &bqkv, &kv_k, &kv_v, nqt, nkt, hd, freq_base, freq_scale, pos, rope_style,
    );
    cb.submit().expect("submit");

    // CPU reference: rope q in place, rope + store k, store v — applied to a
    // copy of `orig` (the pre-kernel q|k|v).
    let read_f32 = |buf: &MetalBuffer, n: usize| -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.contents().as_ptr() as *const f32, v.as_mut_ptr(), n)
        };
        v
    };
    let mut ref_buf = orig.clone();
    let half_dim = hd / 2;
    let rope_pair = |buf: &mut [f32], off: usize, h: usize, d: usize| {
        let base = h * hd;
        let i0 = off + base + d;
        let i1 = off + base + d + half_dim;
        let freq = freq_scale / freq_base.powf((2.0 * d as f32) / hd as f32);
        let theta = pos as f32 * freq;
        let (cs, sn) = (theta.cos(), theta.sin());
        let (x0, x1) = (buf[i0], buf[i1]);
        buf[i0] = x0 * cs - x1 * sn;
        buf[i1] = x0 * sn + x1 * cs;
    };
    // q section 0..nqt
    for h in 0..nh {
        for d in 0..half_dim {
            rope_pair(&mut ref_buf, 0, h, d);
        }
    }
    // k section nqt..nqt+nkt (rope + store)
    for h in 0..nk {
        for d in 0..half_dim {
            rope_pair(&mut ref_buf, nqt, h, d);
        }
    }
    // v section unchanged

    let got_buf = read_f32(&bqkv, total);
    let mut maxd = 0.0f32;
    for (x, y) in got_buf.iter().zip(ref_buf.iter()) {
        maxd = maxd.max((x - y).abs());
    }
    assert!(
        maxd < 1e-6,
        "attn_rope_store concat buffer diverges (max {maxd:.3e})"
    );

    // KV store: k = roped k, v = raw v, at the position offset `pos*nkt`.
    // NOTE: the kernel's `pow`/`cos`/`sin` (Metal libm) differ from Rust's
    // `powf`/`cos`/`sin` by ~1e-6, so compare with a tolerance, not ==.
    let kv_off = pos as usize * nkt;
    let mut kd = 0.0f32;
    let mut vd = 0.0f32;
    if kv_f16 {
        let kk = read_f16(&kv_k, (pos as usize + 1) * nkt);
        let vv = read_f16(&kv_v, (pos as usize + 1) * nkt);
        for j in 0..nkt {
            kd = kd.max((kk[kv_off + j].to_f32() - ref_buf[nqt + j]).abs());
            vd = vd.max((vv[kv_off + j].to_f32() - ref_buf[nqt + nkt + j]).abs());
        }
    } else {
        let kk = read_f32(&kv_k, (pos as usize + 1) * nkt);
        let vv = read_f32(&kv_v, (pos as usize + 1) * nkt);
        for j in 0..nkt {
            kd = kd.max((kk[kv_off + j] - ref_buf[nqt + j]).abs());
            vd = vd.max((vv[kv_off + j] - ref_buf[nqt + nkt + j]).abs());
        }
    }
    assert!(kd < 1e-5, "kv_k store diverges (max {kd:.3e})");
    assert!(vd < 1e-5, "kv_v store diverges (max {vd:.3e})");
}

fn read_f16(buf: &MetalBuffer, n: usize) -> Vec<half::f16> {
    let mut v = vec![half::f16::from_bits(0); n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().as_ptr() as *const half::f16,
            v.as_mut_ptr(),
            n,
        )
    };
    v
}
