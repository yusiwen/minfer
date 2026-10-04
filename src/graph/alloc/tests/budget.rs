//! E4/E4-S3 accounting: the budget, the length contract and rebuild re-mapping.
//!
//! Split out of `src/graph/alloc/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// Issue #122's fail-open twin: a **poisoned** weights lock must not report 0 bytes.
///
/// The mutation this pins is `weights.lock().map(|w| …).unwrap_or(0)`: a panic in
/// another thread poisons the mutex, the sum became 0, and `weights + activations >
/// budget` then under-charged the budget by every resident weight. The registry is
/// append-only, so the value behind the poison is still valid and is recovered.
#[test]
fn a_poisoned_registry_does_not_report_zero_weights() {
    let registry: std::sync::Mutex<Vec<(u64, usize)>> =
        std::sync::Mutex::new(vec![(1, 7), (2, 11)]);
    // Poison it the way a panic inside a holder would.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = registry.lock().unwrap();
        panic!("poison the registry");
    }));
    assert!(registry.is_poisoned(), "the test must actually poison it");
    let bytes = weights_from_lock(registry.lock(), "test registry", |r| {
        r.iter().map(|(_, size)| *size).sum()
    });
    assert_eq!(
        bytes, 18,
        "the poisoned registry must report its real bytes, not 0"
    );
}
/// E4's safety half: a request that would exceed the backend's budget is refused
/// **before the pool is touched**, with the numbers (weights, pooled, this request,
/// budget), instead of surfacing later as a null pointer at execute time.
#[test]
fn a_graph_that_cannot_fit_is_refused_with_its_numbers() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [1024, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    // One byte short of a 4 KiB activation: the gate must fire (the class of 1024
    // elements is 1024 exactly, so this is the requirement minus one).
    alloc.set_memory_budget(Backend::CPU, Some(4095));
    let err = alloc.alloc_graph(&g).unwrap_err();
    assert!(err.contains("out of CPU memory"), "{err}");
    assert!(err.contains("budget"), "{err}");
    assert!(
        err.contains("MiB"),
        "the error must carry the numbers, not a bare failure: {err}"
    );
    // Nothing was allocated: the gate runs before the pool is asked for anything.
    assert_eq!(
        alloc.n_cpu_buffers(),
        0,
        "the refused graph must not touch the pool"
    );
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(report.pool_bytes, 0);
    assert_eq!(report.budget, Some(4095));
    assert_eq!(report.headroom_bytes(), Some(4095));

    // The same graph fits once the budget does.
    alloc.set_memory_budget(Backend::CPU, Some(1 << 20));
    alloc.alloc_graph(&g).unwrap();
    let report = alloc.memory_report(Backend::CPU);
    assert!(report.pool_bytes > 0, "{report:?}");
    assert!(report.live_bytes > 0 && report.live_bytes <= report.pool_bytes);
    assert!(report.peak_live_bytes >= report.live_bytes);
    assert_eq!(report.budget, Some(1 << 20));
    assert!(report.headroom_bytes().unwrap() < 1 << 20);
}
/// E4's accounting half: weights and activations are one comparison, and the peak is
/// a number the caller can read (the ticket's "peak memory is reported").
#[test]
fn the_budget_counts_weights_and_activations_together() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [256, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let weight = tensor_f32("w", [256, 1, 1, 1], vec![0.5; 256]);
    let weight_bytes = weight.data.len();

    let mut alloc = GraphAllocator::new();
    alloc.register_weight("w", weight);
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(report.weights_bytes, weight_bytes, "{report:?}");
    assert_eq!(report.pool_bytes, 0, "nothing is allocated before a build");

    // A budget that covers the weight but not the activation is still a refusal.
    let activation_class =
        crate::graph::allocplan::class_bytes(crate::graph::allocplan::class_size(256 * 4 / 4));
    alloc.set_memory_budget(Backend::CPU, Some(weight_bytes + activation_class - 1));
    let err = alloc.alloc_graph(&g).unwrap_err();
    assert!(err.contains("out of CPU memory"), "{err}");
    assert!(
        err.contains(&format!("{} MiB", weight_bytes / (1024 * 1024))),
        "the refusal must name the weight bytes: {err}"
    );

    // One byte more and it fits.
    alloc.set_memory_budget(Backend::CPU, Some(weight_bytes + activation_class));
    alloc.alloc_graph(&g).unwrap();
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(report.weights_bytes, weight_bytes);
    assert!(
        report.weights_bytes + report.pool_bytes <= report.budget.unwrap(),
        "{report:?}"
    );
}
/// E4 S2's second acceptance: pooled activations are rounded up to their size class,
/// so a graph whose shapes move **inside** one class recycles the pool's buffers
/// instead of reserving one per shape ("no silent growth"). Before this, the free
/// list matched lengths exactly and every shape of a rebuild added a buffer.
#[test]
fn a_rebuild_inside_one_class_reuses_the_pool() {
    let mut alloc = GraphAllocator::new();
    let build = |alloc: &mut GraphAllocator, rows: usize| {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [896, rows, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();
        alloc.alloc_graph(&g).unwrap();
    };
    // 896 x 16 = 14336 and 896 x 17 = 15232 both round to the 16384-element class.
    let a = 896 * 16;
    let class = allocplan::class_size(a);
    assert_eq!(allocplan::class_size(896 * 17), class);
    build(&mut alloc, 16);
    let first = alloc.memory_report(Backend::CPU);
    assert_eq!(first.pool_bytes, allocplan::class_bytes(class), "{first:?}");
    let first_buffers = alloc.n_cpu_buffers();

    build(&mut alloc, 17);
    let second = alloc.memory_report(Backend::CPU);
    assert_eq!(
        second.pool_bytes, first.pool_bytes,
        "a shape inside the same class must recycle the class buffer, not reserve another"
    );
    assert_eq!(
        alloc.n_cpu_buffers(),
        first_buffers,
        "the pool grew for a shape that fits an existing class"
    );

    // The ladder is not a no-op: a shape that leaves the class does reserve more.
    build(&mut alloc, 20); // 896 x 20 = 17920 -> the next 16 KiB step
    assert!(
        alloc.memory_report(Backend::CPU).pool_bytes > first.pool_bytes,
        "a bigger class must actually reserve"
    );
}
/// E4 S2: the length contract moved to the owning `BufRef`, so a fill is checked
/// against the node's **logical** length — the pool buffer is rounded up to a class
/// and is routinely longer, which makes "equals the physical length" wrong and
/// "fits in the physical length" too weak.
#[test]
fn a_fill_must_match_the_nodes_logical_length() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [3, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(
        report.pool_bytes,
        allocplan::class_bytes(allocplan::class_size(3)),
        "a 3-element activation still occupies a whole class: {report:?}"
    );

    alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0]).unwrap();
    for wrong in [vec![1.0f32, 2.0, 3.0, 4.0], vec![1.0f32, 2.0]] {
        let err = alloc.fill_input(&g, "x", &wrong).unwrap_err();
        assert!(
            err.contains(&format!(
                "{} elements were supplied but the node holds 3",
                wrong.len()
            )),
            "got: {err}"
        );
    }
    // The refused fills wrote nothing, and the read is the window, not the class.
    assert_eq!(alloc.get_buffer(&g, x).unwrap(), &[1.0, 2.0, 3.0]);
}
/// E4 S2: an input is host-filled **before** execution, so it must never take a buffer
/// that this build's `sweep` released — the previous owner writes that buffer during
/// execution, i.e. after the fill, and the input's value is gone by the time its
/// consumer reads it. Here `t`'s buffer becomes free before `y` is placed, and `t`
/// writes it at step 1.
///
/// This test was originally written with two distinct inputs (`add(x, w)`) because
/// `add(x, x)` could not be built at all: `topo_order` reported a false cycle on a
/// repeated source (#98). The repeated source is legal now and covered end to end by
/// [`a_repeated_source_allocates_and_executes_as_two_reads`]; the two-input form is
/// kept here on purpose so this gate keeps pinning *its* property (input placement)
/// rather than the duplicate-source path.
#[test]
fn an_input_never_takes_a_buffer_the_walk_released() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [8, 1, 1, 1], crate::graph::DType::F32);
    let w = b.input("w", [8, 1, 1, 1], crate::graph::DType::F32);
    let t = b.add(x, w); // its own buffer; this is the only node that writes it
    let c = b.mul(t, w); // last use of `t`, so its buffer is released right after
    let y = b.input("y", [8, 1, 1, 1], crate::graph::DType::F32);
    let o = b.mul(c, y);
    b.output(o);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    let xs: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let ws = vec![1.0f32; 8];
    let ys: Vec<f32> = (1..=8).map(|v| (v * 10) as f32).collect();
    alloc.fill_input(&g, "x", &xs).unwrap();
    alloc.fill_input(&g, "w", &ws).unwrap();
    alloc.fill_input(&g, "y", &ys).unwrap();
    crate::graph::scheduler::BackendScheduler::new()
        .execute(&g, &mut alloc)
        .expect("execute");
    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(a, b)| (a + 1.0) * 1.0 * b)
        .collect();
    assert_eq!(
        alloc.copy_to_cpu(o).expect("read"),
        want,
        "`y` must still hold its fill value after `t` executed"
    );
}
/// #98: a node may read the same source twice — `add(x, x)` is `2 * x`
/// written as an addition, `mul(x, x)` is `x * x`, and `GraphBuilder::add`
/// / `mul` pass `&[a, b]` straight through. `alloc_graph` validates with
/// `topo_order()` on every build, so before the fix this graph never
/// reached execution at all (it failed with a false cycle). The assertion
/// is the **value**: `is_ok()` alone would also pass a graph that dropped
/// the second read, and `2 * x` would come back as `x`.
#[test]
fn a_repeated_source_allocates_and_executes_as_two_reads() {
    let xs = [1.5f32, -2.0, 3.25, 0.0];
    let wants = [
        ("add", xs.map(|v| 2.0 * v).to_vec()),
        ("mul", xs.map(|v| v * v).to_vec()),
    ];
    for (name, want) in wants {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
        let out = if name == "mul" {
            b.mul(x, x)
        } else {
            b.add(x, x)
        };
        b.output(out);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc
            .alloc_graph(&g)
            .unwrap_or_else(|e| panic!("{name}(x, x) must allocate, got: {e}"));
        alloc.fill_input(&g, "x", &xs).unwrap();
        crate::graph::scheduler::BackendScheduler::new()
            .execute(&g, &mut alloc)
            .unwrap_or_else(|e| panic!("{name}(x, x) must execute, got: {e}"));
        assert_eq!(
            alloc.copy_to_cpu(out).expect("read"),
            want,
            "{name}(x, x) must read x twice, not drop the repeated source"
        );
    }
}
/// E4 S3's acceptance: a rebuild **re-maps**. The reservation table hands the same
/// class-sized buffers back, so the pool creates nothing (`CpuBackend::alloc_count`, the
/// CPU twin of CUDA's `pool_gen`) and the node → slot mapping is identical — the
/// "reserve, then assign" split, observable.
#[test]
fn a_rebuild_remaps_instead_of_reallocating() {
    /// The buffers a graph's nodes were assigned, by node id: `(backend, pool id)` —
    /// deliberately not the lengths, which differ between two shapes in a class.
    fn slots(alloc: &GraphAllocator, g: &ComputeGraph) -> Vec<(NodeId, Option<(Backend, usize)>)> {
        (0..g.n_nodes())
            .map(|i| (i, alloc.node_buffer(i).map(|b| (b.backend, b.id))))
            .collect()
    }
    let graph = |rows: usize| {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [896, rows, 1, 1], crate::graph::DType::F32);
        let w = b.input("w", [896, rows, 1, 1], crate::graph::DType::F32);
        let t = b.add(x, w);
        let y = b.silu(t); // in-place: stays in `t`'s slot
        b.output(y);
        b.build()
    };

    let mut alloc = GraphAllocator::new();
    let g = graph(16);
    alloc.alloc_graph(&g).unwrap();
    let first = slots(&alloc, &g);
    let (allocs, buffers) = (alloc.n_cpu_allocs(), alloc.n_cpu_buffers());
    assert!(allocs > 0 && buffers > 0, "the first build does allocate");

    // The same graph, and a neighbour **in the same class** (896 x 17 = 15232 rounds to
    // the same 16384-element class as 896 x 16 = 14336): both re-map onto the reservation.
    for rows in [16, 17] {
        let g = graph(rows);
        alloc.alloc_graph(&g).unwrap();
        assert_eq!(
            alloc.n_cpu_allocs(),
            allocs,
            "a re-map must not create a pool buffer (rows = {rows})"
        );
        assert_eq!(alloc.n_cpu_buffers(), buffers);
        assert_eq!(
            slots(&alloc, &g),
            first,
            "the same topology gets the same slots (rows = {rows})"
        );
    }

    // A shape that leaves the class reserves a new buffer — the table is not a no-op.
    alloc.alloc_graph(&graph(64)).unwrap();
    assert!(alloc.n_cpu_allocs() > allocs, "a bigger class must reserve");
    assert!(alloc.n_cpu_buffers() > buffers);
}
/// E4 S3: the **reservation** is visible — a released classed buffer stays in the pool and
/// goes to the idle list, so `live_bytes` drops back while `pool_bytes` does not, and the
/// idle slot is what the next graph is assigned.
#[test]
fn a_released_buffer_stays_reserved_and_idle() {
    let mut alloc = GraphAllocator::new();
    let mut b = GraphBuilder::new();
    let x = b.input("x", [256, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();
    alloc.alloc_graph(&g).unwrap();
    let built = alloc.memory_report(Backend::CPU);
    assert!(built.pool_bytes > 0 && built.live_bytes > 0);
    // Rebuilding the same graph frees every classed buffer at the start and re-assigns
    // them, so at the end the reservation and the live set are exactly what they were.
    alloc.alloc_graph(&g).unwrap();
    let again = alloc.memory_report(Backend::CPU);
    assert_eq!(again.pool_bytes, built.pool_bytes);
    assert_eq!(again.live_bytes, built.live_bytes);
    assert_eq!(again.idle_slots, built.idle_slots);
}
/// E4 S3, CUDA half: a rebuild must not touch the device pool at all. `pool_gen` is the
/// generation counter CUDA invalidates its captured graphs on, so "the plan is untouched"
/// is exactly "the capture survives a rebuild".
#[test]
#[cfg(feature = "cuda")]
fn a_rebuild_does_not_touch_the_device_pool() {
    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("no CUDA device; skipping the E4 S3 device gate");
        return;
    }
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_cuda());
    let graph = |rows: usize| {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [512, rows, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        let z = b.add(y, x);
        b.output(z);
        let mut g = b.build();
        // The device pool is what this gate is about: every node on CUDA.
        for n in g.nodes.iter_mut() {
            n.backend = Some(Backend::CUDA);
        }
        g
    };
    let g = graph(4);
    alloc.alloc_graph(&g).unwrap();
    let gen = alloc.cuda().unwrap().pool_gen();
    let mapping: Vec<_> = (0..g.n_nodes())
        .map(|i| alloc.node_buffer(i).map(|b| (b.backend, b.id)))
        .collect();
    // Same shape: no device traffic.
    alloc.alloc_graph(&g).unwrap();
    assert_eq!(
        alloc.cuda().unwrap().pool_gen(),
        gen,
        "a same-shape rebuild must not allocate or free on the device"
    );
    // Same class, different shape: still no device traffic. (512 x 4 = 2048 and
    // 512 x 3 = 1536 both round to the 2048-element class — asserted, so the case cannot
    // silently stop being the one it claims to test.)
    assert_eq!(
        crate::graph::allocplan::class_size(512 * 4),
        crate::graph::allocplan::class_size(512 * 3)
    );
    alloc.alloc_graph(&graph(3)).unwrap();
    assert_eq!(
        alloc.cuda().unwrap().pool_gen(),
        gen,
        "a shape in the same class re-maps onto the reserved device buffer"
    );
    let same: Vec<_> = (0..g.n_nodes())
        .map(|i| alloc.node_buffer(i).map(|b| (b.backend, b.id)))
        .collect();
    assert_eq!(same, mapping, "and onto the same slots");
}
/// The tiny f32 tensor the accounting tests register (`Tensor` carries raw bytes).
fn tensor_f32(name: &str, shape: [i64; 4], data: Vec<f32>) -> crate::tensor::Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let mut t = crate::tensor::Tensor::from_data(crate::tensor::TensorType::F32, &shape, bytes);
    t.name = name.to_string();
    t
}
