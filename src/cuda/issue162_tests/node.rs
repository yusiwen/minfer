//! The node-level consequence: a required launch failure fails the op with an `Err` naming the site, and the matmul `Err` arm drains the sticky too.
//!
//! Split out of `src/cuda/issue162_tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// The node-level consequence, and the acceptance's "no stale output" claim:
/// a required launch failure inside a real op makes `execute_node` return
/// `Err` naming the site. Device + gated.
#[test]
fn cuda_issue162_a_required_launch_failure_fails_the_node() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    if !gate_enabled() {
        return;
    }
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::scheduler::BackendScheduler;

    let s = device().unwrap();
    let mut b = GraphBuilder::new();
    let x = b.input("x", [16, 1, 1, 1], crate::graph::DType::F32);
    let y = b.input("y", [16, 1, 1, 1], crate::graph::DType::F32);
    let z = b.add(x, y);
    b.output(z);
    let mut g = b.build();

    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = BackendScheduler::new();
    sched.assign_backends(&mut g, &alloc);
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &[1.0f32; 16]).unwrap();
    alloc.fill_input(&g, "y", &[2.0f32; 16]).unwrap();

    // Positive control: the node executes and produces 3.0.
    let _ = s.take_last_error();
    unsafe { minfer_site_fail_reset() };
    sched.execute(&g, &mut alloc).unwrap();
    assert_eq!(alloc.copy_to_cpu(z).unwrap(), vec![3.0f32; 16]);

    // Injected: the required launch fails, so the op must not proceed on an
    // unwritten output.
    let _ = s.take_last_error();
    unsafe { minfer_site_fail_reset() };
    let err = {
        let _arm = Arm::new("launch:add_f32");
        sched
            .execute(&g, &mut alloc)
            .expect_err("a required launch failure must fail the node")
    };
    assert!(
        err.contains("launch:add_f32") && err.contains("add_f32"),
        "the node error must name the site: {err}"
    );
    assert_eq!(
        s.take_last_error(),
        0,
        "and the site's own latch must not reach CudaState::sync"
    );
    assert!(
        s.take_launch_failure().is_none(),
        "execute_node must drain the sticky even on its Err arm"
    );
}
/// The **Err** arm's drain, isolated: an f16 matmul's Rust wrapper turns the
/// launcher's own `int` return into an `Err`, so `execute_node_inner` returns
/// `Err` *with the sticky already set*. Without the unconditional drain the
/// record would survive into the next `execute_node` and be blamed on it.
/// Device + gated.
#[test]
fn cuda_issue162_the_err_arm_also_drains_the_sticky() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    if !gate_enabled() {
        return;
    }
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::scheduler::BackendScheduler;

    let s = device().unwrap();
    // od=8, id=64: `id % 8 == 0` selects the vectorized f16 site.
    let (od, id) = (8usize, 64usize);
    let wb = vec![0u8; od * id * 2];
    let mut wt = Tensor::from_data(TensorType::F16, &[id as i64, od as i64, 1, 1], wb.clone());
    wt.name = "issue162_f16_w".to_string();
    s.register_weight(&wt.name, &wb);

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id, 1, 1, 1], crate::graph::DType::F32);
    let m = b.matmul(x, &wt, None);
    b.output(m);
    let mut g = b.build();
    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let sched = BackendScheduler::new();
    sched.assign_backends(&mut g, &alloc);
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &vec![0.0f32; id]).unwrap();

    let _ = s.take_last_error();
    unsafe { minfer_site_fail_reset() };
    let err = {
        let _arm = Arm::new("launch:f16_f32_matmul_vec");
        sched
            .execute(&g, &mut alloc)
            .expect_err("the f16 matmul launcher's Err must reach the scheduler")
    };
    assert!(
        err.contains("f16 matmul"),
        "the node error is the launcher's own: {err}"
    );
    assert!(
        s.take_launch_failure().is_none(),
        "the Err arm must drain the sticky too, or the NEXT node would be blamed for this \
         launch (issue #162)"
    );
    assert_eq!(s.take_last_error(), 0);
}
