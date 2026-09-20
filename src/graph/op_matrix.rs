//! Op × dtype × backend correctness matrix (ticket A1).
//!
//! Three tests, each with a different job:
//!
//! 1. `matrix_cases_match_their_reference` — every case in [`cases`] runs on
//!    every backend that claims the op, and the result is compared against an
//!    **independent** reference (an analytic formula written here), never
//!    against another backend. A backend that is not compiled in or has no
//!    device reports `SKIP` with the reason — never `PASS` (A0: CUDA is
//!    compile-only on the dev box, Metal needs macOS).
//! 2. `support_table_matches_support_matrix_doc` — each backend's `supports_op`
//!    agrees with the "Operator Coverage by Backend" table in
//!    `docs/SUPPORT-MATRIX.md`. On a CPU-only box this checks the CPU column;
//!    the other columns light up wherever they are compiled in.
//! 3. `every_op_has_a_matrix_decision` — every `Op` variant is either covered
//!    by a case or explicitly excused with a reason. `op_label` matches
//!    exhaustively with no wildcard arm, so adding an `Op` variant is a
//!    **compile error** until this file is updated.
//!
//! The matrix is deliberately small and deterministic: it is a smoke net for
//! the op semantics and the support contract, not a numerical test suite (the
//! kernel-level tests live next to each backend).

use super::alloc::GraphAllocator;
use super::backend::Backend as BackendTrait;
use super::builder::GraphBuilder;
use super::cpu_backend::CpuBackend;
use super::ops::{AttnMeta, AttnMode, NodeMeta, Op, RoPEMeta};
use super::scheduler::BackendScheduler;
use super::{Backend, DType, NodeId};
use crate::tensor::{Tensor, TensorType};
use crate::vec_ops::RopeStyle;

/// Absolute tolerance for the f32 cases. The references below are written in
/// f32 in the same order as the kernels, so exact equality would mostly hold;
/// the slack absorbs the SIMD paths' different accumulation order.
const TOL: f32 = 1e-4;

/// Builds one case's graph. Returns the output node, the inputs to fill (by
/// name, in f32 — I32 inputs are converted by the runner), the expected output,
/// and any weight tensors the case created (the CPU backend resolves weights by
/// name from its registry, so they must be registered before execution).
type Build = fn(&mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>);

type Inputs = Vec<(&'static str, Vec<f32>)>;

struct Case {
    /// Must match `op_label` for the completeness test.
    name: &'static str,
    /// Representative instance, used for the `supports_op` query.
    op: Op,
    build: Build,
    /// Cases whose reference is approximate say why.
    note: &'static str,
}

/// Ops with no case, and the reason. Keep this list short and honest.
const EXCUSED: &[(&str, &str)] = &[
    ("Input", "leaf node: carries data, computes nothing"),
    (
        "BatchMatMul",
        "deferred by design (single-output IR); no architecture emits it",
    ),
    (
        "FusedQKV",
        "GPU-only decode fusion (CPU refuses it); needs a Metal/CUDA run",
    ),
    (
        "FusedFFN",
        "GPU-only decode fusion (CPU refuses it); needs a Metal/CUDA run",
    ),
    (
        "FusedQkvNorm",
        "GPU-only decode fusion (CPU refuses it); needs a Metal/CUDA run",
    ),
    (
        "QkvBiasRopeStore",
        "GPU-only decode fusion (CPU refuses it); needs a Metal/CUDA run",
    ),
];

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn t(name: &str, shape: [i64; 4], data: &[f32]) -> Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut t = Tensor::from_data(TensorType::F32, &shape, bytes);
    t.name = name.to_string();
    t
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Compares against a reference, returning a human-readable mismatch.
fn compare(name: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{name}: length {} != expected {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if !(g - w).abs().le(&(TOL * (1.0 + w.abs()))) {
            return Err(format!("{name}: element {i} = {g}, expected {w}"));
        }
    }
    Ok(())
}

/// Does `tag` claim `op` (at F32)? `Err` means the backend is unavailable here,
/// which the matrix reports as a skip.
fn backend_claims(tag: Backend, op: &Op) -> Result<bool, String> {
    match tag {
        Backend::CPU => {
            let b = CpuBackend::new();
            Ok(super::backend_takes(&b, op, DType::F32))
        }
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                match super::metal_backend::MetalBackend::new() {
                    Some(m) => Ok(super::backend_takes(&m, op, DType::F32)),
                    None => Err("no Metal device".into()),
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = op;
                Err("not compiled in (macOS-only module)".into())
            }
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                // `CudaState::get()` stays `None` until something initializes the
                // process-wide state: `main` does it at startup, the model tests
                // through the loader. A harness that runs CUDA cells must do it
                // itself, or the column silently depends on *test order* (it ran
                // on a device in a full-suite run and skipped in a filtered one).
                crate::cuda::CudaState::init();
                match super::cuda_backend::CudaBackend::new() {
                    Some(c) => Ok(super::backend_takes(&c, op, DType::F32)),
                    None => Err("no CUDA device".into()),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = op;
                Err("not compiled in (--features cuda)".into())
            }
        }
    }
}

/// Run one case entirely on `tag`: every node's backend is forced, so the
/// scheduler builds a single split and no cross-backend copy can mask a bug.
fn run_on(tag: Backend, case: &Case) -> Result<Vec<f32>, String> {
    let mut b = GraphBuilder::new();
    let (out, inputs, _, weights) = (case.build)(&mut b);
    b.output(out);
    let mut g = b.build();
    for n in &mut g.nodes {
        n.backend = Some(tag);
    }

    let mut alloc = GraphAllocator::new();
    match tag {
        Backend::CPU => {}
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                if !alloc.enable_metal() {
                    return Err("no Metal device".into());
                }
            }
            #[cfg(not(target_os = "macos"))]
            return Err("not compiled in (macOS-only module)".into());
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::CudaState::init();
                if !alloc.enable_cuda() {
                    return Err("no CUDA device".into());
                }
            }
            #[cfg(not(feature = "cuda"))]
            return Err("not compiled in (--features cuda)".into());
        }
    }

    alloc.alloc_graph(&g)?;
    for w in &weights {
        alloc.register_weight(&w.name, w.clone());
        // The CPU registry is not the CUDA one: in the product the *loader*
        // registers each weight into `CudaState` (`models/*/loader.rs`), which a
        // synthetic harness has no equivalent of. Without this the matrix's
        // weight ops could only ever report "weight not registered on CUDA".
        #[cfg(feature = "cuda")]
        if tag == Backend::Cuda {
            if let Some(state) = crate::cuda::CudaState::get() {
                state.register_weight(&w.name, w.data());
            }
        }
    }
    for (name, data) in &inputs {
        let id = graph_input_id(&g, name).ok_or_else(|| format!("no input '{name}'"))?;
        if g.node(id).out_dtype == DType::I32 {
            // I32 inputs are stored as bit patterns of small integers.
            let v: Vec<u32> = data.iter().map(|&x| x as u32).collect();
            alloc.fill_input_i32(&g, name, &v)?;
        } else {
            alloc.fill_input(&g, name, data)?;
        }
    }
    BackendScheduler::new().execute(&g, &mut alloc)?;
    alloc
        .copy_to_cpu(out)
        .ok_or_else(|| format!("no host copy of output node {out}"))
}

fn graph_input_id(g: &super::ComputeGraph, name: &str) -> Option<NodeId> {
    g.inputs.iter().copied().find(|&i| g.node(i).name == name)
}

// ---------------------------------------------------------------------------
// cases
// ---------------------------------------------------------------------------

fn build_add(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let y = b.input("y", [4, 1, 1, 1], DType::F32);
    let o = b.add(x, y);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let c = vec![10.0, 20.0, 30.0, 40.0];
    let exp = a.iter().zip(&c).map(|(p, q)| p + q).collect();
    (o, vec![("x", a), ("y", c)], exp, vec![])
}

fn build_mul(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let y = b.input("y", [4, 1, 1, 1], DType::F32);
    let o = b.mul(x, y);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let c = vec![0.5, -1.0, 2.0, 0.25];
    let exp = a.iter().zip(&c).map(|(p, q)| p * q).collect();
    (o, vec![("x", a), ("y", c)], exp, vec![])
}

fn build_scale(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let o = b.node(
        "scale",
        Op::Scale(2.0),
        &[x],
        [4, 1, 1, 1],
        DType::F32,
        NodeMeta::None,
    );
    let a = vec![1.0, -2.0, 3.0, 0.0];
    let exp = a.iter().map(|v| v * 2.0).collect();
    (o, vec![("x", a)], exp, vec![])
}

fn build_silu(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let o = b.silu(x);
    let a = vec![-2.0, -0.5, 0.5, 3.0];
    let exp = a.iter().map(|&v| silu(v)).collect();
    (o, vec![("x", a)], exp, vec![])
}

fn build_swiglu(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let gate = b.input("gate", [4, 1, 1, 1], DType::F32);
    let up = b.input("up", [4, 1, 1, 1], DType::F32);
    let o = b.swiglu(gate, up);
    let gv = vec![0.5, -1.0, 2.0, 0.0];
    let uv = vec![1.0, 3.0, -2.0, 5.0];
    let exp = gv.iter().zip(&uv).map(|(&g, &u)| silu(g) * u).collect();
    (o, vec![("gate", gv), ("up", uv)], exp, vec![])
}

fn build_softmax(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let o = b.softmax(x, 0);
    let a = vec![1.0, 2.0, 3.0, 0.5];
    let mx = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = a.iter().map(|v| (v - mx).exp()).collect();
    let s: f32 = e.iter().sum();
    let exp = e.iter().map(|v| v / s).collect();
    (o, vec![("x", a)], exp, vec![])
}

fn build_rms_norm(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let w = t("norm.weight", [4, 1, 1, 1], &[1.0, 0.5, -1.0, 2.0]);
    let o = b.rms_norm(x, Some(&w), 1e-5);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let mean: f32 = a.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let scale = 1.0 / (mean + 1e-5).sqrt();
    let exp = a
        .iter()
        .zip([1.0, 0.5, -1.0, 2.0])
        .map(|(&v, w)| v * scale * w)
        .collect();
    (o, vec![("x", a)], exp, vec![w])
}

fn build_qk_norm(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    // [nt * nh, hd] = [2, 4]: two heads, each normalized with the same weight.
    // hd is a multiple of 4 because CUDA's qk_norm kernel is a float4 kernel
    // ("qk_norm head dim 2 must be a nonzero multiple of 4"), so the hd = 2
    // fixture could only ever pass on CPU.
    let x = b.input("x", [8, 1, 1, 1], DType::F32);
    let w = t("q_norm.weight", [4, 1, 1, 1], &[1.0, 2.0, 3.0, 4.0]);
    let o = b.qk_norm(x, Some(&w), 4, 2, 1e-5);
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut exp = Vec::with_capacity(8);
    for h in 0..2 {
        let row = &a[h * 4..h * 4 + 4];
        let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let scale = 1.0 / (mean + 1e-5).sqrt();
        for (i, v) in row.iter().enumerate() {
            exp.push(v * scale * (i as f32 + 1.0));
        }
    }
    (o, vec![("x", a)], exp, vec![w])
}

fn build_matmul(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    // GGUF convention: metadata [in, out]; memory row-major [out][in].
    let w = t(
        "w",
        [4, 3, 1, 1],
        &[
            1.0, 0.0, 0.0, 0.0, // out row 0
            0.0, 1.0, 0.0, 0.0, // out row 1
            1.0, 1.0, 1.0, 1.0, // out row 2
        ],
    );
    let o = b.matmul(x, &w, None);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let exp = vec![a[0], a[1], a.iter().sum::<f32>()];
    (o, vec![("x", a)], exp, vec![w])
}

fn build_get_rows(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    // The generic gather is out[t*n + i] = x[ids[t]*n + i]: x is a flat array of
    // `n_embd`-wide rows, so index 2 needs at least three of them.
    let x = b.input("x", [4, 3, 1, 1], DType::F32);
    let ids = b.input("ids", [1, 1, 1, 1], DType::I32);
    let o = b.get_rows(x, ids, [4, 1, 1, 1]);
    let a = vec![
        10.0, 11.0, 12.0, 13.0, // row 0
        20.0, 21.0, 22.0, 23.0, // row 1
        30.0, 31.0, 32.0, 33.0, // row 2
    ];
    (
        o,
        vec![("x", a), ("ids", vec![2.0])],
        vec![30.0, 31.0, 32.0, 33.0],
        vec![],
    )
}

fn build_view(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let o = b.node(
        "view",
        Op::View {
            offset: 0,
            shape: [4, 1, 1, 1],
        },
        &[x],
        [4, 1, 1, 1],
        DType::F32,
        NodeMeta::None,
    );
    let a = vec![1.0, 2.0, 3.0, 4.0];
    (o, vec![("x", a.clone())], a, vec![])
}

/// D1 increment 2: a **partial** window at a non-zero offset — the shape D2
/// needs (q, k and v are windows of one concat buffer). `copy_to_cpu` must read
/// exactly the window, not the parent's whole buffer.
fn build_view_offset(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [8, 1, 1, 1], DType::F32);
    let o = b.node(
        "view_at_2",
        Op::View {
            offset: 2,
            shape: [4, 1, 1, 1],
        },
        &[x],
        [4, 1, 1, 1],
        DType::F32,
        NodeMeta::None,
    );
    let a: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    (o, vec![("x", a)], vec![3.0, 4.0, 5.0, 6.0], vec![])
}

fn build_reshape(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let o = b.node(
        "reshape",
        Op::Reshape {
            shape: [4, 1, 1, 1],
        },
        &[x],
        [4, 1, 1, 1],
        DType::F32,
        NodeMeta::None,
    );
    let a = vec![1.0, 2.0, 3.0, 4.0];
    (o, vec![("x", a.clone())], a, vec![])
}

fn build_permute(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let o = b.node(
        "permute",
        Op::Permute { dims: [0, 1, 2, 3] },
        &[x],
        [4, 1, 1, 1],
        DType::F32,
        NodeMeta::None,
    );
    let a = vec![1.0, 2.0, 3.0, 4.0];
    (o, vec![("x", a.clone())], a, vec![])
}

fn build_rope(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    // hd = 2, nh = 1, one token at position 1: the single frequency is
    // freq_scale / base^(0/2) = 1, so theta = 1 rad.
    let x = b.input("x", [2, 1, 1, 1], DType::F32);
    let pos = b.input("positions", [1, 1, 1, 1], DType::I32);
    let o = b.rope(
        x,
        pos,
        RopeStyle::NonInterleaved,
        RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: 1,
            hd: 2,
        },
    );
    let a = vec![1.0, 0.0];
    let (s, c) = 1.0f32.sin_cos();
    let exp = vec![a[0] * c - a[1] * s, a[0] * s + a[1] * c];
    (o, vec![("x", a), ("positions", vec![1.0])], exp, vec![])
}

fn build_attn(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    // One head, one KV head, hd = 4, one query at position 0: the window the
    // span input names holds exactly one KV row, so softmax over a single score
    // is 1 and the output is V.
    //
    // hd is a multiple of 4 because CUDA's attention kernels require it
    // (`attn head dim outside the kernel's supported range` otherwise); with
    // hd = 2 this case could only ever PASS on CPU, which made the matrix's
    // CUDA column a compile-check rather than a run.
    let pos = b.input("positions", [1, 1, 1, 1], DType::I32);
    let q = b.input("q", [4, 1, 1, 1], DType::F32);
    let k = b.input("k", [4, 1, 1, 1], DType::F32);
    let v = b.input("v", [4, 1, 1, 1], DType::F32);
    let _store = b.kvcache_store(0, k, v, 4);
    let load = b.kvcache_load(0, 4, 4, 1);
    let o = b.attn(
        q,
        load,
        pos,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: 1,
            n_head_kv: 1,
            hd: 4,
            hd_kv: 4,
            nkt: 4,
            scale: 0.5,
        },
    );
    (
        o,
        vec![
            ("q", vec![1.0, 1.0, 0.0, 0.0]),
            ("k", vec![1.0, 0.0, 0.0, 0.0]),
            ("v", vec![0.25, -0.75, 0.0, 0.0]),
            ("positions", vec![0.0]),
            // C6: the query's sequence-relative position is 0 and its KV row is 0.
            ("cells", vec![0.0]),
            // E1: one sequence, one query, window [0, 1).
            ("seq_ids", vec![0.0]),
            ("attn_span", vec![0.0, 1.0]),
        ],
        vec![0.25, -0.75, 0.0, 0.0],
        vec![],
    )
}

fn build_kv_roundtrip(b: &mut GraphBuilder) -> (NodeId, Inputs, Vec<f32>, Vec<Tensor>) {
    // Every cell of the region is written, so the expectation below is fully
    // defined on every backend. (Leaving rows unwritten and expecting zeros is a
    // CPU-buffer property, not a contract: a fresh CPU pool is zeroed, a device
    // region is not, and no kernel ever reads an unwritten cell.)
    let pos = b.input("positions", [4, 1, 1, 1], DType::I32);
    let k = b.input("k", [2, 4, 1, 1], DType::F32);
    let v = b.input("v", [2, 4, 1, 1], DType::F32);
    let _store = b.kvcache_store(0, k, v, 4);
    let load = b.kvcache_load(0, 2, 4, 1);
    // The load node's buffer IS the K region: [n_embd, n_ctx] = 2 x 4, in cell
    // order.
    (
        load,
        vec![
            ("k", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ("v", vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
            ("positions", vec![0.0, 1.0, 2.0, 3.0]),
            // C6: the rows are the resolved cells — the same numbers here, because
            // this fixture's single run starts at cell 0.
            ("cells", vec![0.0, 1.0, 2.0, 3.0]),
        ],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![],
    )
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "Add",
            op: Op::Add,
            build: build_add,
            note: "",
        },
        Case {
            name: "Mul",
            op: Op::Mul,
            build: build_mul,
            note: "",
        },
        Case {
            name: "Scale",
            op: Op::Scale(2.0),
            build: build_scale,
            note: "",
        },
        Case {
            name: "Silu",
            op: Op::Silu,
            build: build_silu,
            note: "",
        },
        Case {
            name: "SwiGLU",
            op: Op::SwiGLU,
            build: build_swiglu,
            note: "",
        },
        Case {
            name: "Softmax",
            op: Op::Softmax { dim: 0 },
            build: build_softmax,
            note: "",
        },
        Case {
            name: "RmsNorm",
            op: Op::RmsNorm { eps: 1e-5 },
            build: build_rms_norm,
            note: "",
        },
        Case {
            name: "QkNorm",
            op: Op::QkNorm {
                hd: 2,
                nh: 2,
                eps: 1e-5,
            },
            build: build_qk_norm,
            note: "",
        },
        Case {
            name: "MatMul",
            op: Op::MatMul { transpose_b: false },
            build: build_matmul,
            note: "f32 weight; the quantized dtype axis is covered by the per-backend kernel tests",
        },
        Case {
            name: "GetRows",
            op: Op::GetRows,
            build: build_get_rows,
            note: "",
        },
        Case {
            name: "View",
            op: Op::View {
                offset: 0,
                shape: [4, 1, 1, 1],
            },
            build: build_view,
            note: "",
        },
        Case {
            name: "View offset",
            op: Op::View {
                offset: 2,
                shape: [4, 1, 1, 1],
            },
            build: build_view_offset,
            note: "D1: a partial window at a non-zero offset",
        },
        Case {
            name: "Reshape",
            op: Op::Reshape {
                shape: [4, 1, 1, 1],
            },
            build: build_reshape,
            note: "",
        },
        Case {
            name: "Permute",
            op: Op::Permute { dims: [0, 1, 2, 3] },
            build: build_permute,
            note: "",
        },
        Case {
            name: "RoPE",
            op: Op::RoPE {
                style: RopeStyle::NonInterleaved,
            },
            build: build_rope,
            note: "",
        },
        Case {
            name: "Attn",
            op: Op::Attn {
                mode: AttnMode::Gqa,
                explicit_span: false,
            },
            build: build_attn,
            note: "",
        },
        Case {
            name: "KvcacheStore",
            op: Op::KvcacheStore { layer: 0 },
            build: build_kv_roundtrip,
            note: "store + load in one graph: the load node's buffer is the region",
        },
        Case {
            name: "KvcacheLoad",
            op: Op::KvcacheLoad { layer: 0 },
            build: build_kv_roundtrip,
            note: "shared with KvcacheStore",
        },
    ]
}

/// Every `Op` variant, one representative each. Exhaustive by construction: the
/// match in `op_label` has no wildcard arm, so a new variant fails to compile
/// until it is listed here.
fn all_ops() -> Vec<Op> {
    vec![
        Op::Input,
        Op::Add,
        Op::Mul,
        Op::Scale(1.0),
        Op::Silu,
        Op::Softmax { dim: 0 },
        Op::RmsNorm { eps: 1e-5 },
        Op::QkNorm {
            hd: 2,
            nh: 2,
            eps: 1e-5,
        },
        Op::MatMul { transpose_b: false },
        Op::GetRows,
        Op::RoPE {
            style: RopeStyle::NonInterleaved,
        },
        Op::Attn {
            mode: AttnMode::Gqa,
            explicit_span: false,
        },
        Op::KvcacheStore { layer: 0 },
        Op::KvcacheLoad { layer: 0 },
        Op::View {
            offset: 0,
            shape: [1, 1, 1, 1],
        },
        Op::Reshape {
            shape: [1, 1, 1, 1],
        },
        Op::Permute { dims: [0, 1, 2, 3] },
        Op::SwiGLU,
        Op::BatchMatMul,
        Op::FusedQKV { layer: 0 },
        Op::QkvBiasRopeStore { layer: 0 },
        Op::FusedFFN,
        Op::FusedQkvNorm { layer: 0 },
    ]
}

/// Exhaustive: adding an `Op` variant breaks this match on purpose.
fn op_label(op: &Op) -> &'static str {
    match op {
        Op::Input => "Input",
        Op::Add => "Add",
        Op::Mul => "Mul",
        Op::Scale(_) => "Scale",
        Op::Silu => "Silu",
        Op::Softmax { .. } => "Softmax",
        Op::RmsNorm { .. } => "RmsNorm",
        Op::QkNorm { .. } => "QkNorm",
        Op::MatMul { .. } => "MatMul",
        Op::GetRows => "GetRows",
        Op::RoPE { .. } => "RoPE",
        Op::Attn { .. } => "Attn",
        Op::KvcacheStore { .. } => "KvcacheStore",
        Op::KvcacheLoad { .. } => "KvcacheLoad",
        Op::View { .. } => "View",
        Op::Reshape { .. } => "Reshape",
        Op::Permute { .. } => "Permute",
        Op::SwiGLU => "SwiGLU",
        Op::BatchMatMul => "BatchMatMul",
        Op::FusedQKV { .. } => "FusedQKV",
        Op::QkvBiasRopeStore { .. } => "QkvBiasRopeStore",
        Op::FusedFFN => "FusedFFN",
        Op::FusedQkvNorm { .. } => "FusedQkvNorm",
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const TARGETS: [Backend; 3] = [Backend::CPU, Backend::Metal, Backend::Cuda];

#[test]
fn matrix_cases_match_their_reference() {
    let mut fails = Vec::new();
    let mut report = Vec::new();
    for case in cases() {
        for &tag in &TARGETS {
            let cell = match backend_claims(tag, &case.op) {
                Err(reason) => format!("SKIP ({reason})"),
                Ok(false) => format!("SKIP ({tag:?} does not claim {})", case.name),
                Ok(true) => match run_on(tag, &case) {
                    Err(e) => {
                        fails.push(format!("{} on {:?}: {e}", case.name, tag));
                        format!("FAIL ({e})")
                    }
                    Ok(got) => {
                        let (_, _, want, _) = {
                            let mut b = GraphBuilder::new();
                            (case.build)(&mut b)
                        };
                        match compare(&case.name, &got, &want) {
                            Ok(()) => "PASS".to_string(),
                            Err(e) => {
                                fails.push(format!("{} on {:?}: {e}", case.name, tag));
                                format!("FAIL ({e})")
                            }
                        }
                    }
                },
            };
            report.push(format!(
                "{:<14} {:<6} {cell}{}",
                case.name,
                format!("{tag:?}"),
                if case.note.is_empty() {
                    String::new()
                } else {
                    format!("   [{}]", case.note)
                }
            ));
        }
    }
    eprintln!("\n=== op × dtype × backend matrix ===");
    for line in &report {
        eprintln!("{line}");
    }
    let ran = report.iter().filter(|l| l.contains("PASS")).count();
    eprintln!("{ran} cell(s) exercised on this machine\n");
    assert!(
        ran > 0,
        "no backend claimed any case — the matrix is not running"
    );
    assert!(
        fails.is_empty(),
        "op matrix failures:\n{}",
        fails.join("\n")
    );
}

/// Mirrors the "Operator Coverage by Backend" table in `docs/SUPPORT-MATRIX.md`.
#[test]
fn support_table_matches_support_matrix_doc() {
    // (op, CPU, Metal, CUDA)
    let table: &[(&str, Op, bool, bool, bool)] = &[
        ("Add", Op::Add, true, true, true),
        ("Mul", Op::Mul, true, true, true),
        ("Silu", Op::Silu, true, true, true),
        ("SwiGLU", Op::SwiGLU, true, true, true),
        ("RmsNorm", Op::RmsNorm { eps: 1e-5 }, true, true, true),
        (
            "QkNorm",
            Op::QkNorm {
                hd: 2,
                nh: 2,
                eps: 1e-5,
            },
            true,
            true,
            true,
        ),
        (
            "MatMul",
            Op::MatMul { transpose_b: false },
            true,
            true,
            true,
        ),
        ("GetRows", Op::GetRows, true, true, true),
        (
            "Attn",
            Op::Attn {
                mode: AttnMode::Gqa,
                explicit_span: false,
            },
            true,
            true,
            true,
        ),
        // E1's asymmetric row: CPU and CUDA have a windowed attention path that
        // reads the explicit span (E1b ported CUDA's kernels); Metal still
        // derives its bound from positions and must refuse a multi-sequence
        // attention node (`backend_takes`), so it cannot be assigned one.
        (
            "Attn explicit-span",
            Op::Attn {
                mode: AttnMode::Gqa,
                explicit_span: true,
            },
            true,
            false,
            true,
        ),
        (
            "KvcacheStore",
            Op::KvcacheStore { layer: 0 },
            true,
            true,
            true,
        ),
        (
            "KvcacheLoad",
            Op::KvcacheLoad { layer: 0 },
            true,
            true,
            true,
        ),
        (
            "View",
            Op::View {
                offset: 0,
                shape: [1, 1, 1, 1],
            },
            true,
            true,
            true,
        ),
        (
            "Reshape",
            Op::Reshape {
                shape: [1, 1, 1, 1],
            },
            true,
            true,
            true,
        ),
        (
            "Permute",
            Op::Permute { dims: [0, 1, 2, 3] },
            true,
            true,
            true,
        ),
        // Asymmetric rows — the ones a platform-dependent decode path rides on.
        (
            "RoPE non-interleaved",
            Op::RoPE {
                style: RopeStyle::NonInterleaved,
            },
            true,
            true,
            true,
        ),
        (
            "RoPE interleaved",
            Op::RoPE {
                style: RopeStyle::Interleaved,
            },
            true,
            true,
            false,
        ),
        ("FusedQKV", Op::FusedQKV { layer: 0 }, false, true, true),
        ("FusedFFN", Op::FusedFFN, false, true, true),
        (
            "FusedQkvNorm",
            Op::FusedQkvNorm { layer: 0 },
            false,
            true,
            false,
        ),
        (
            "QkvBiasRopeStore",
            Op::QkvBiasRopeStore { layer: 0 },
            false,
            false,
            true,
        ),
        ("Scale", Op::Scale(1.0), true, false, false),
        ("Softmax", Op::Softmax { dim: 0 }, true, false, false),
        ("BatchMatMul", Op::BatchMatMul, false, false, false),
    ];
    for (name, op, cpu, metal, cuda) in table {
        for (tag, want) in [
            (Backend::CPU, *cpu),
            (Backend::Metal, *metal),
            (Backend::Cuda, *cuda),
        ] {
            match backend_claims(tag, op) {
                // Unavailable here (not compiled in / no device): nothing to check.
                Err(_) => {}
                Ok(got) => assert_eq!(
                    got, want,
                    "{name}: {tag:?} claims {got}, SUPPORT-MATRIX.md documents {want}"
                ),
            }
        }
    }
}

#[test]
fn every_op_has_a_matrix_decision() {
    let covered: Vec<&str> = cases().iter().map(|c| c.name).collect();
    let mut undecided = Vec::new();
    for op in all_ops() {
        let label = op_label(&op);
        if covered.contains(&label) {
            continue;
        }
        if EXCUSED.iter().any(|(l, _)| *l == label) {
            continue;
        }
        undecided.push(label);
    }
    assert!(
        undecided.is_empty(),
        "Op variant(s) with neither a matrix case nor an excuse: {undecided:?} — \
         add a case in `cases()` or an entry in `EXCUSED`"
    );
}
