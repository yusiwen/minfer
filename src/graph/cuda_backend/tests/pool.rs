//! Pool and scheduler plumbing: device buffers, persistent KV regions and a scheduler chain.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn cuda_pool_roundtrip() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut cb = CudaBackend::new().expect("backend after device init");
    let id = cb.alloc_buffer(16);
    cb.write_host(id, &[1.5f32; 16]).unwrap();
    assert_eq!(cb.copy_to_host(id).unwrap(), vec![1.5f32; 16]);
    // shorter than the buffer is fine, longer is rejected
    cb.write_host(id, &[2.0f32; 4]).unwrap();
    assert!(cb.write_host(id, &[2.0f32; 32]).is_err());
    // free-list reuse hands back the same id; pool_gen tracked both times
    cb.free_buffer(id);
    let id2 = cb.alloc_buffer(16);
    assert_eq!(id, id2);
    assert_eq!(cb.pool_gen, 2);
}
#[test]
fn cuda_scheduler_chain() {
    crate::cuda::CudaState::init();
    let Some(state) = crate::cuda::CudaState::get() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (id_, od, nt) = (64usize, 32usize, 2usize);
    let cw: Vec<f32> = (0..id_).map(|i| 0.8 + (i % 5) as f32 / 10.0).collect();
    let cwb: Vec<u8> = cw.iter().flat_map(|v| v.to_le_bytes()).collect();
    state.register_weight("cw", &cwb);
    let mut cwt = Tensor::from_data(TensorType::F32, &[id_ as i64, 1, 1, 1], cwb);
    cwt.name = "cw".to_string();
    let wf: Vec<f32> = (0..od * id_)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let mut w8b = Vec::new();
    for r in 0..od {
        w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
            &wf[r * id_..(r + 1) * id_],
        ));
    }
    state.register_weight("cw8", &w8b);
    let mut w8t = Tensor::from_data(TensorType::Q8_0, &[id_ as i64, od as i64, 1, 1], w8b);
    w8t.name = "cw8".to_string();
    let bias: Vec<f32> = (0..od).map(|i| (i % 3) as f32 / 7.0).collect();
    let bb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    state.register_weight("cb", &bb);
    let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], bb);
    bt.name = "cb".to_string();

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let n1 = b.rms_norm(x, Some(&cwt), 1e-5);
    let m = b.matmul(n1, &w8t, Some(&bt));
    let s = b.silu(m);
    b.output(s);
    let mut g = b.build();

    // Full pipeline: assign → alloc → fill → execute (no fusion needed).
    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = crate::graph::scheduler::BackendScheduler::new();
    sched.assign_backends(&mut g, &alloc);
    for (i, nd) in g.nodes.iter().enumerate() {
        assert_eq!(
            nd.backend,
            Some(crate::graph::Backend::CUDA),
            "node {i} ({})",
            nd.name
        );
    }
    alloc.alloc_graph(&g).unwrap();
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|i| ((i * 97) % 21) as f32 / 5.0 - 2.0)
        .collect();
    alloc.fill_input(&g, "x", &xs).unwrap();
    sched.execute(&g, &mut alloc).unwrap();

    // Host reference: rms → dequant matmul + bias → silu
    let mut rmsd = vec![0f32; id_ * nt];
    for t in 0..nt {
        crate::vec_ops::rms_norm_fused_f32(
            id_,
            &mut rmsd[t * id_..(t + 1) * id_],
            &xs[t * id_..(t + 1) * id_],
            &cw,
            1e-5,
        );
    }
    let mut dq = vec![0f32; od * id_];
    crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8t, &mut dq, id_);
    let mut mm = vec![0f32; od * nt];
    for t in 0..nt {
        for r in 0..od {
            let mut acc = 0f32;
            for i in 0..id_ {
                acc += dq[r * id_ + i] * rmsd[t * id_ + i];
            }
            mm[t * od + r] = acc + bias[r];
        }
    }
    let mut want = vec![0f32; od * nt];
    crate::vec_ops::vec_silu_f32(od * nt, &mut want, &mm);
    let got = alloc.copy_to_cpu(s).unwrap();
    let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
    assert_close("scheduler chain", &got, &want, scale * 1e-3);
}
