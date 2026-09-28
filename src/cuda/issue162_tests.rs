//! `#[cfg(test)] mod issue162_tests` for `src/cuda.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use std::collections::{BTreeSet, HashMap};

/// `minfer_site_fail_kind` for a kernel launch (`cuda_kernels.cu`).
const SITE_LAUNCH: i32 = 2;
/// `MINFER_SITE_*` / `ATTN_WIN_*` mirrors (cuda_kernels.cu).
const CAUSAL: i32 = 0;
const SPAN: i32 = 1;
const MAP: i32 = 2;
const ERR_INVALID_VALUE: i32 = 1; // cudaErrorInvalidValue

fn device() -> Option<&'static CudaState> {
    CudaState::init();
    CudaState::get()
}

fn cstr(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

fn gate_enabled() -> bool {
    if std::env::var("MINFER_TEST_ISSUE162").is_err() {
        eprintln!(
            "skipping: set MINFER_TEST_ISSUE162=1 to run the #162 launch-failure gates \
             (they drive real CUDA launches into failure; a compute-sanitizer run must not \
             set it)"
        );
        return false;
    }
    true
}

/// The committed audit fixture (`scripts/check_cuda_launch_returns.py
/// --fixture tests/fixtures/cuda_launch_sites.tsv`): `line, owner, site,
/// kernel-fragment` for every `<<<` site in `cuda_kernels.cu`. The gate
/// asserts the *driven* set against it, so a launcher the driver misses, or a
/// new site added without a driver, is a red gate rather than a silent gap.
fn fixture() -> Vec<(String, String, String, String)> {
    include_str!("../../tests/fixtures/cuda_launch_sites.tsv")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 4, "fixture row: {l}");
            (
                f[0].to_string(),
                f[1].to_string(),
                f[2].to_string(),
                f[3].to_string(),
            )
        })
        .collect()
}

/// Restores `MINFER_TEST_CALL_FAIL` on drop, so a panicking gate cannot leave
/// the injection armed for the rest of the process.
struct Arm {
    prev: Option<String>,
}
impl Arm {
    fn new(list: &str) -> Self {
        let prev = std::env::var("MINFER_TEST_CALL_FAIL").ok();
        std::env::set_var("MINFER_TEST_CALL_FAIL", list);
        Self { prev }
    }
}
impl Drop for Arm {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var("MINFER_TEST_CALL_FAIL", v),
            None => std::env::remove_var("MINFER_TEST_CALL_FAIL"),
        }
    }
}

/// Zeroed device scratch. An injected launch never runs; the two `_opt`
/// k-split sites the driver reaches alone run their kernel for real, so the
/// buffers are 1 MiB each — well past every shape the driver passes.
struct Ctx {
    stream: *mut std::ffi::c_void,
    bufs: Vec<*mut std::ffi::c_void>,
}
impl Ctx {
    fn new(s: &CudaState) -> Self {
        let n = 1 << 20;
        let mut bufs = Vec::new();
        for _ in 0..12 {
            let p = CudaState::cuda_malloc(n);
            assert!(!p.is_null(), "cudaMalloc({n}) failed");
            let zeros = vec![0u8; n];
            let e = unsafe {
                cudaMemcpy(
                    p,
                    zeros.as_ptr() as *const std::ffi::c_void,
                    n,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            };
            assert_eq!(e, 0, "zero-fill failed: {}", cuda_error_name(e));
            bufs.push(p);
        }
        Self {
            stream: s.stream(),
            bufs,
        }
    }
    fn p(&self, i: usize) -> *mut std::ffi::c_void {
        self.bufs[i]
    }
    fn f(&self, i: usize) -> *mut f32 {
        self.bufs[i] as *mut f32
    }
    fn cf(&self, i: usize) -> *const f32 {
        self.bufs[i] as *const f32
    }
    fn u(&self, i: usize) -> *const u8 {
        self.bufs[i] as *const u8
    }
    fn mu(&self, i: usize) -> *mut u8 {
        self.bufs[i] as *mut u8
    }
    fn i32(&self, i: usize) -> *mut i32 {
        self.bufs[i] as *mut i32
    }
    fn ci32(&self, i: usize) -> *const i32 {
        self.bufs[i] as *const i32
    }
}

/// Drive one dispatch with `arm` armed, then assert: the armed set is exactly
/// the observed set, every report is a loud named launch failure at its own
/// site naming its kernel instantiation, and nothing latched.
fn run(
    s: &CudaState,
    frag: &HashMap<String, String>,
    arm: &[&str],
    seen: &mut BTreeSet<String>,
    call: impl FnOnce(),
) {
    let _ = s.take_last_error();
    unsafe { minfer_site_fail_reset() };
    {
        let _g = Arm::new(&arm.join(","));
        call();
    }
    let n = unsafe { minfer_site_hist_len() };
    let mut observed = BTreeSet::new();
    for i in 0..n {
        let site = cstr(unsafe { minfer_site_hist_site(i) });
        let name = cstr(unsafe { minfer_site_hist_name(i) });
        let msg = cstr(unsafe { minfer_site_hist_msg(i) });
        assert!(
            msg.contains("kernel launch"),
            "[{site}] must say the launch failed: {msg}"
        );
        assert!(
            msg.contains("cudaErrorInvalidValue"),
            "[{site}] must name the error with cudaGetErrorName: {msg}"
        );
        assert!(
            msg.contains(&format!("#162/{site}")),
            "[{site}] must carry its own incident tag: {msg}"
        );
        let f = frag.get(&site).unwrap_or_else(|| {
            panic!("a report came from a site not in the audit fixture: {site}")
        });
        if !f.is_empty() {
            assert!(
                msg.contains(f.as_str()),
                "[{site}] must name the instantiation fragment `{f}` (reported `{name}`): {msg}"
            );
        }
        assert!(
            observed.insert(site.clone()),
            "[{site}] reported twice in one dispatch"
        );
        seen.insert(site);
    }
    let expected: BTreeSet<String> = arm.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        observed, expected,
        "armed/observed site mismatch (armed {arm:?}); a site the call did not reach, or an \
         unarmed site that failed, is a red gate"
    );
    assert_eq!(
        unsafe { minfer_site_fail_kind() },
        SITE_LAUNCH,
        "kind after arming {arm:?}"
    );
    assert_eq!(
        unsafe { minfer_site_fail_code() },
        ERR_INVALID_VALUE,
        "the injected geometry must return cudaErrorInvalidValue after arming {arm:?}"
    );
    assert_eq!(
        s.take_last_error(),
        0,
        "arming {arm:?} must leave no latched error for CudaState::sync"
    );
}

/// Issue #162 acceptance, site by site: arming a site drives a REAL failing
/// launch, and the site names itself and its instantiation and clears the
/// latch. Every `<<<` site in the audit fixture must be reached.
#[test]
fn cuda_issue162_every_launch_site_names_itself_and_leaves_no_latch() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    if !gate_enabled() {
        return;
    }
    let s = device().unwrap();
    let rows = fixture();
    let frag: HashMap<String, String> = rows.iter().map(|r| (r.2.clone(), r.3.clone())).collect();
    let ctx = Ctx::new(s);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let st = ctx.stream;

    macro_rules! go {
        ($arm:expr, $body:expr) => {
            run(s, &frag, $arm, &mut seen, $body)
        };
    }

    // ── the split-attention families ────────────────────────────────
    for (mode, tag) in [(MAP, "map"), (SPAN, "span"), (CAUSAL, "causal")] {
        let arm: Vec<&str> = vec![
            match tag {
                "map" => "launch:gqa_attn_split_batched_kv__partial_map",
                "span" => "launch:gqa_attn_split_batched_kv__partial_span",
                _ => "launch:gqa_attn_split_batched_kv__partial_causal",
            },
            "launch:gqa_attn_split_batched_kv__combine",
        ];
        go!(&arm, || unsafe {
            launch_gqa_attn_split_batched_f16kv(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.f(4),
                ctx.ci32(5),
                mode,
                4,
                2,
                64,
                0.125,
                68,
                4,
                st,
            );
        });
    }

    // ── embedding gather (8 quantized cases) ────────────────────────
    for (type_id, token) in [
        (0, "launch:embed_rows__q8_0"),
        (1, "launch:embed_rows__q4_0"),
        (2, "launch:embed_rows__q4_k"),
        (7, "launch:embed_rows__q4_1"),
        (4, "launch:embed_rows__q5_1"),
        (5, "launch:embed_rows__q5_k"),
        (6, "launch:embed_rows__q5_0"),
        (3, "launch:embed_rows__q6_k"),
    ] {
        go!(&[token], || unsafe {
            launch_embed_rows(ctx.u(0), ctx.cf(1), ctx.f(2), 256, 2, type_id, 210, st);
        });
    }
    go!(&["launch:embed_rows_f16"], || unsafe {
        launch_embed_rows_f16(ctx.u(0), ctx.cf(1), ctx.f(2), 256, 2, st);
    });

    // ── f32 / f16 matmul shape branches ─────────────────────────────
    for (id, token) in [
        (8, "launch:f32_f32_matmul__vec"),
        (7, "launch:f32_f32_matmul__scalar"),
    ] {
        go!(&[token], || unsafe {
            launch_f32_f32_matmul(ctx.cf(0), ctx.cf(1), ctx.f(2), 8, id, 2, st);
        });
    }
    for (id, token) in [
        (8, "launch:f16_f32_matmul_vec"),
        (7, "launch:f16_f32_matmul_scalar"),
    ] {
        go!(&[token], || unsafe {
            launch_f16_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, id, 2, st);
        });
    }

    // ── every single-site launcher ──────────────────────────────────
    go!(&["launch:q4_0_q8_0_matmul"], || unsafe {
        launch_q4_0_q8_0_matmul(ctx.u(0), ctx.u(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q4_0_f32_matmul"], || unsafe {
        launch_q4_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q8_0_f32_matmul"], || unsafe {
        launch_q8_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q4_1_f32_matmul"], || unsafe {
        launch_q4_1_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q4_k_f32_matmul"], || unsafe {
        launch_q4_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q5_1_f32_matmul"], || unsafe {
        launch_q5_1_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q5_0_f32_matmul"], || unsafe {
        launch_q5_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q5_k_f32_matmul"], || unsafe {
        launch_q5_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q6_k_f32_matmul"], || unsafe {
        launch_q6_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:q6_k_f32_matmul_padded"], || unsafe {
        launch_q6_k_f32_matmul_padded(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
    });
    go!(&["launch:swiglu_f32_off"], || unsafe {
        launch_swiglu_f32_off(ctx.f(0), 16, 0, st);
    });
    go!(&["launch:swiglu_quant_pad40"], || unsafe {
        launch_swiglu_quant_pad40(ctx.f(0), ctx.mu(1), 16, 0, st);
    });
    go!(&["launch:gather_rows_f32"], || unsafe {
        launch_gather_rows_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, 2, st);
    });
    go!(&["launch:quantize_q8_0_pad40"], || unsafe {
        launch_quantize_q8_0_pad40(ctx.cf(0), ctx.mu(1), 256, 2, st);
    });
    go!(&["launch:quantize_q8_0_pad40_t"], || unsafe {
        launch_quantize_q8_0_pad40_t(ctx.cf(0), ctx.mu(1), ctx.mu(2), 256, 2, 8, 1, st);
    });
    go!(&["launch:quantize_q8_0"], || unsafe {
        launch_quantize_q8_0(ctx.cf(0), ctx.mu(1), 256, 2, st);
    });
    go!(&["launch:rms_norm_f32"], || unsafe {
        launch_rms_norm_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, 1e-6, 2, st);
    });
    go!(&["launch:rms_norm_quant_pad40"], || unsafe {
        launch_rms_norm_quant_pad40(ctx.cf(0), ctx.cf(1), ctx.f(2), ctx.mu(3), 64, 1e-6, 8, st);
    });
    go!(&["launch:add_bias_f32"], || unsafe {
        launch_add_bias_f32(ctx.f(0), ctx.cf(1), 64, 8, st);
    });
    go!(&["launch:add_f32"], || unsafe {
        launch_add_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
    });
    go!(&["launch:mul_f32"], || unsafe {
        launch_mul_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
    });
    go!(&["launch:silu_f32"], || unsafe {
        launch_silu_f32(ctx.f(0), 64, st);
    });
    go!(&["launch:swiglu_f32"], || unsafe {
        launch_swiglu_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
    });
    go!(&["launch:rms_norm_quant_f32_t"], || unsafe {
        launch_rms_norm_quant_f32_t(
            ctx.cf(0),
            ctx.cf(1),
            ctx.f(2),
            ctx.mu(3),
            ctx.mu(4),
            256,
            1e-6,
            8,
            8,
            1,
            st,
        );
    });
    go!(&["launch:rms_norm_quant_nw_f32_t"], || unsafe {
        launch_rms_norm_quant_nw_f32_t(
            ctx.cf(0),
            ctx.cf(1),
            ctx.mu(2),
            ctx.mu(3),
            256,
            1e-6,
            8,
            8,
            1,
            st,
        );
    });
    go!(&["launch:swiglu_quant_f32_t"], || unsafe {
        launch_swiglu_quant_f32_t(
            ctx.cf(0),
            ctx.cf(1),
            ctx.f(2),
            ctx.mu(3),
            ctx.mu(4),
            256,
            2,
            8,
            1,
            st,
        );
    });
    go!(&["launch:swiglu_quant_nw_f32_t"], || unsafe {
        launch_swiglu_quant_nw_f32_t(ctx.cf(0), ctx.cf(1), ctx.mu(2), ctx.mu(3), 256, 2, 8, 1, st);
    });
    go!(&["launch:f32_bits_to_i32"], || unsafe {
        launch_f32_bits_to_i32(ctx.cf(0), ctx.i32(1), 64, st);
    });
    go!(&["launch:rope_f32"], || unsafe {
        launch_rope_f32(ctx.f(0), 4, 64, 2, 10000.0, 1.0, ctx.ci32(1), st);
    });
    go!(&["launch:store_kv_f32"], || unsafe {
        launch_store_kv_f32(ctx.cf(0), ctx.f(1), 64, 2, ctx.ci32(2), st);
    });
    go!(&["launch:store_kv_f16"], || unsafe {
        launch_store_kv_f16(ctx.cf(0), ctx.p(1), 64, 2, ctx.ci32(2), st);
    });
    go!(&["launch:store_kv_q8_0"], || unsafe {
        launch_store_kv_q8_0(ctx.cf(0), ctx.p(1), 64, 2, 68, ctx.ci32(2), st);
    });
    go!(&["launch:attn_bias_rope_store"], || unsafe {
        launch_attn_bias_rope_store(
            ctx.f(0),
            ctx.f(1),
            ctx.f(2),
            ctx.p(3),
            ctx.p(4),
            ctx.p(5),
            ctx.p(6),
            ctx.p(7),
            2,
            4,
            64,
            10000.0,
            1.0,
            ctx.ci32(8),
            ctx.ci32(9),
            0,
            st,
        );
    });
    // #144 item 1: the packed arm of the same epilogue.
    go!(&["launch:attn_bias_rope_store_q8_0"], || unsafe {
        launch_attn_bias_rope_store_q8_0(
            ctx.f(0),
            ctx.cf(1),
            ctx.cf(2),
            ctx.p(3),
            ctx.p(4),
            ctx.p(5),
            ctx.p(6),
            ctx.p(7),
            2,
            4,
            64,
            10000.0,
            1.0,
            ctx.ci32(8),
            ctx.ci32(9),
            68,
            st,
        );
    });
    for (mode, token) in [
        (MAP, "launch:gqa_attn_f32_f16kv__map"),
        (SPAN, "launch:gqa_attn_f32_f16kv__span"),
        (CAUSAL, "launch:gqa_attn_f32_f16kv__causal"),
    ] {
        go!(&[token], || unsafe {
            launch_gqa_attn_f32_f16kv(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.ci32(4),
                mode,
                4,
                2,
                64,
                0.125,
                2,
                st,
            );
        });
    }
    for (mode, token) in [
        (MAP, "launch:gqa_attn_split_f32kv__map"),
        (SPAN, "launch:gqa_attn_split_f32kv__span"),
        (CAUSAL, "launch:gqa_attn_split_f32kv__causal"),
    ] {
        go!(
            &[token, "launch:gqa_attn_split_f32kv__combine"],
            || unsafe {
                launch_gqa_attn_split_f32kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    st,
                );
            }
        );
    }
    for (mode, base_token, dp4a_token, wide_token) in [
        (
            MAP,
            "launch:gqa_attn_split_q8_0__map",
            "launch:gqa_attn_split_q8_0__map_dp4a",
            "launch:gqa_attn_split_q8_0__map_wide",
        ),
        (
            SPAN,
            "launch:gqa_attn_split_q8_0__span",
            "launch:gqa_attn_split_q8_0__span_dp4a",
            "launch:gqa_attn_split_q8_0__span_wide",
        ),
        (
            CAUSAL,
            "launch:gqa_attn_split_q8_0__causal",
            "launch:gqa_attn_split_q8_0__causal_dp4a",
            "launch:gqa_attn_split_q8_0__causal_wide",
        ),
    ] {
        // #186/#202: the Q8_0 decode launcher picks one of three instantiations
        // from the `dp4a`/`wide` arguments. Every audited site must be driven,
        // so all three arms are driven explicitly — the env resolves to one of
        // them for production, but `run`'s armed set is a value here, not a
        // process-global answer.
        for (token, dp4a, wide) in [
            (base_token, 0i32, 0i32),
            (dp4a_token, 1, 0),
            (wide_token, 1, 1),
        ] {
            go!(&[token, "launch:gqa_attn_split_q8_0__combine"], || unsafe {
                launch_gqa_attn_split_q8_0(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    68,
                    dp4a,
                    wide,
                    st,
                );
            });
        }
    }
    // hd == 128 takes the dual-kernel (h4w + hybrid) path; hd != 128 the
    // single incumbent one. Both share `__combine`.
    for (mode, mtag) in [(MAP, "map"), (SPAN, "span"), (CAUSAL, "causal")] {
        let h4w = format!("launch:gqa_attn_split_f16kv__{mtag}_h4w");
        let hyb = format!("launch:gqa_attn_split_f16kv__hybrid_{mtag}");
        go!(
            &[
                h4w.as_str(),
                hyb.as_str(),
                "launch:gqa_attn_split_f16kv__combine"
            ],
            || unsafe {
                launch_gqa_attn_split_f16kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    128,
                    0.125,
                    68,
                    st,
                );
            }
        );
        let plain = format!("launch:gqa_attn_split_f16kv__{mtag}");
        go!(
            &[plain.as_str(), "launch:gqa_attn_split_f16kv__combine"],
            || unsafe {
                launch_gqa_attn_split_f16kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    st,
                );
            }
        );
    }
    // The layout-tagged general kernel: one source site, nine instantiations
    // (the message names the instantiation). Drive one.
    go!(&["launch:gqa_attn_f32"], || unsafe {
        launch_gqa_attn_f32(
            ctx.cf(0),
            ctx.p(1),
            ctx.p(2),
            ctx.f(3),
            ctx.ci32(4),
            CAUSAL,
            KV_LAYOUT_F32,
            4,
            2,
            64,
            0.125,
            256,
            2,
            st,
        );
    });
    // #144: the FA launcher serves both staged layouts from one source site
    // per (layout, mode) — six sites, all driven here.
    for (mode, layout, token) in [
        (
            MAP,
            crate::cuda::KV_LAYOUT_F16,
            "launch:fa_prefill_kv__f16_map",
        ),
        (
            SPAN,
            crate::cuda::KV_LAYOUT_F16,
            "launch:fa_prefill_kv__f16_span",
        ),
        (
            CAUSAL,
            crate::cuda::KV_LAYOUT_F16,
            "launch:fa_prefill_kv__f16_causal",
        ),
        (
            MAP,
            crate::cuda::KV_LAYOUT_Q8_0,
            "launch:fa_prefill_kv__q8_0_map",
        ),
        (
            SPAN,
            crate::cuda::KV_LAYOUT_Q8_0,
            "launch:fa_prefill_kv__q8_0_span",
        ),
        (
            CAUSAL,
            crate::cuda::KV_LAYOUT_Q8_0,
            "launch:fa_prefill_kv__q8_0_causal",
        ),
    ] {
        go!(&[token], || unsafe {
            launch_fa_prefill_kv(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.ci32(4),
                mode,
                4,
                2,
                128,
                0.125,
                2,
                layout,
                136,
                st,
            );
        });
    }
    for (type_id, token) in [
        (0, "launch:dequant_f16__q8_0"),
        (1, "launch:dequant_f16__q4_0"),
        (2, "launch:dequant_f16__q4_1"),
        (3, "launch:dequant_f16__q5_0"),
        (4, "launch:dequant_f16__q5_1"),
        (5, "launch:dequant_f16__q4_k"),
        (6, "launch:dequant_f16__q5_k"),
        (7, "launch:dequant_f16__q6_k"),
    ] {
        go!(&[token], || unsafe {
            launch_dequant_f16(type_id, ctx.u(0), ctx.p(1), 8, 64, 210, st);
        });
    }
    go!(&["launch:convert_f16"], || unsafe {
        launch_convert_f16(ctx.cf(0), ctx.p(1), 256, st);
    });
    go!(&["launch:gemm_qb_nt"], || unsafe {
        launch_gemm_qb_nt(ctx.p(0), ctx.u(1), ctx.f(2), 2, 8, 64, 5, 212, st);
    });
    // `launch_gemm_f16`'s site token is chosen by `af32`; the audit resolves
    // the ternary to the first arm, so drive the `af32 = true` one.
    go!(&["launch:gemm_f16_a32"], || unsafe {
        launch_gemm_f16(ctx.p(0), ctx.p(1), ctx.f(2), 2, 8, 64, st, true);
    });
    // The MMQ fast paths: arm the launch token (not the attribute token), so
    // the opt-in succeeds and the launch itself is what fails.
    go!(&["launch:mmq_raw_nb"], || unsafe {
        launch_mmq_raw_nb_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 1, 8, 64, st, 8);
    });
    for (kd, token) in [
        (4, "launch:mmq_raw_wide_kd4"),
        (8, "launch:mmq_raw_wide_kd8"),
    ] {
        go!(&[token], || unsafe {
            launch_mmq_raw_wide_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, st, kd);
        });
    }
    for (kd, token) in [(4, "launch:mmq_raw_nt_kd4"), (8, "launch:mmq_raw_nt_kd8")] {
        go!(&[token], || unsafe {
            launch_mmq_raw_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, st, kd);
        });
    }
    go!(&["launch:mmq_nt"], || unsafe {
        launch_mmq_nt(0, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, 40, st);
    });
    // The two NB-BT launchers: `w_dsc`/`w_exp` selects the instantiation, and
    // the k-split reduce is its own token (the kernel's token returns before
    // it, so the reduce can only be reached alone).
    for (dsc, token) in [
        (false, "launch:mmq_raw_nb_bt"),
        (true, "launch:mmq_raw_nb_bt"),
    ] {
        go!(&[token], || unsafe {
            launch_mmq_raw_nb_bt_nt(
                5,
                ctx.u(0),
                if dsc { ctx.u(1) } else { std::ptr::null() },
                ctx.u(2),
                ctx.u(3),
                ctx.f(4),
                1,
                8,
                64,
                8,
                st,
                8,
                ctx.f(5),
                1,
            );
        });
    }
    go!(&["launch:mmq_raw_nb_bt_ksplit"], || unsafe {
        launch_mmq_raw_nb_bt_nt(
            5,
            ctx.u(0),
            ctx.u(1),
            ctx.u(2),
            ctx.u(3),
            ctx.f(4),
            1,
            8,
            64,
            16,
            st,
            8,
            ctx.f(5),
            2,
        );
    });
    for (exp, token) in [
        (false, "launch:mmq_raw_nb_bt_q6k"),
        (true, "launch:mmq_raw_nb_bt_q6k"),
    ] {
        go!(&[token], || unsafe {
            launch_mmq_raw_nb_bt_q6k_nt(
                7,
                ctx.u(0),
                if exp { ctx.u(1) } else { std::ptr::null() },
                ctx.u(2),
                ctx.u(3),
                ctx.u(4),
                ctx.f(5),
                1,
                8,
                64,
                8,
                212,
                st,
                8,
                ctx.f(6),
                1,
            );
        });
    }
    go!(&["launch:mmq_raw_nb_bt_q6k_ksplit"], || unsafe {
        launch_mmq_raw_nb_bt_q6k_nt(
            7,
            ctx.u(0),
            ctx.u(1),
            ctx.u(2),
            ctx.u(3),
            ctx.u(4),
            ctx.f(5),
            1,
            8,
            64,
            16,
            212,
            st,
            8,
            ctx.f(6),
            2,
        );
    });
    // ── the multi-token MMVQ family ─────────────────────────────────
    for (token, extra) in [
        ("launch:q4_k_q8_mmvq", 0),
        ("launch:q4_k_q8_mmvq_v2", 0),
        ("launch:q4_k_q8_mmvq_multi", 0),
        ("launch:q4_k_q8_mmvq_v2_multi", 0),
        ("launch:q5_k_q8_mmvq", 0),
        ("launch:q5_k_q8_mmvq_v2", 0),
        ("launch:q5_k_q8_mmvq_multi", 0),
        ("launch:q5_k_q8_mmvq_v2_multi", 0),
        ("launch:q4_0_q8_mmvq", 0),
        ("launch:q4_0_q8_mmvq_multi", 0),
        ("launch:q8_0_q8_mmvq", 0),
        ("launch:q8_0_q8_mmvq_multi", 0),
        ("launch:q6_k_q8_mmvq", 1),
        ("launch:q6_k_q8_mmvq_v2", 1),
        ("launch:q6_k_q8_mmvq_v2_pf", 1),
        ("launch:q6_k_q8_mmvq_multi", 1),
        ("launch:q6_k_q8_mmvq_v2_multi", 1),
        ("launch:q6_k_q8_mmvq_v2_dpl", 2),
        ("launch:q6_k_q8_mmvq_v2_pf_dpl", 2),
    ] {
        go!(&[token], || unsafe {
            let (w, a, o) = (ctx.u(0), ctx.u(1), ctx.f(2));
            match token {
                "launch:q4_k_q8_mmvq" => launch_q4_k_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q4_k_q8_mmvq_v2" => launch_q4_k_q8_mmvq_v2(w, a, o, 8, 64, 2, st),
                "launch:q4_k_q8_mmvq_multi" => launch_q4_k_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q4_k_q8_mmvq_v2_multi" => {
                    launch_q4_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, st)
                }
                "launch:q5_k_q8_mmvq" => launch_q5_k_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q5_k_q8_mmvq_v2" => launch_q5_k_q8_mmvq_v2(w, a, o, 8, 64, 2, st),
                "launch:q5_k_q8_mmvq_multi" => launch_q5_k_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q5_k_q8_mmvq_v2_multi" => {
                    launch_q5_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, st)
                }
                "launch:q4_0_q8_mmvq" => launch_q4_0_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q4_0_q8_mmvq_multi" => launch_q4_0_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q8_0_q8_mmvq" => launch_q8_0_q8_mmvq(w, a, o, 8, 64, 2, st),
                "launch:q8_0_q8_mmvq_multi" => launch_q8_0_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                "launch:q6_k_q8_mmvq" => launch_q6_k_q8_mmvq(w, a, o, 8, 64, 2, 210, st),
                "launch:q6_k_q8_mmvq_v2" => launch_q6_k_q8_mmvq_v2(w, a, o, 8, 64, 2, 210, st),
                "launch:q6_k_q8_mmvq_v2_pf" => {
                    launch_q6_k_q8_mmvq_v2_pf(w, a, o, 8, 64, 2, 210, st)
                }
                "launch:q6_k_q8_mmvq_multi" => {
                    launch_q6_k_q8_mmvq_multi(w, a, o, 8, 64, 2, 210, st)
                }
                "launch:q6_k_q8_mmvq_v2_multi" => {
                    launch_q6_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, 210, st)
                }
                "launch:q6_k_q8_mmvq_v2_dpl" => {
                    launch_q6_k_q8_mmvq_v2_dpl(w, a, o, 8, 64, 2, 1, st)
                }
                _ => launch_q6_k_q8_mmvq_v2_pf_dpl(w, a, o, 8, 64, 2, 1, st),
            }
            let _ = extra;
        });
    }
    go!(&["launch:q8_0_p32_q8_mmvq"], || unsafe {
        launch_q8_0_p32_q8_mmvq(ctx.u(0), ctx.u(1), ctx.u(2), ctx.f(3), 8, 64, 2, st);
    });
    go!(&["launch:q8_0_p32_q8_mmvq_multi"], || unsafe {
        launch_q8_0_p32_q8_mmvq_multi(ctx.u(0), ctx.u(1), ctx.u(2), ctx.f(3), 8, 64, 2, st);
    });
    go!(&["launch:kv_move_rows"], || unsafe {
        launch_kv_move_rows(ctx.f(0), ctx.cf(1), 1, 0, 1, 64, st);
    });

    // ── the union assertion (rule 1: the expected set is a value, not a
    //    relation between two code paths) ─────────────────────────────
    let expected: BTreeSet<String> = rows.iter().map(|r| r.2.clone()).collect();
    let missing: Vec<&String> = expected.difference(&seen).collect();
    let extra: Vec<&String> = seen.difference(&expected).collect();
    assert!(
        missing.is_empty(),
        "the driver never reached {} audited <<< site(s): {missing:?}",
        missing.len()
    );
    assert!(
        extra.is_empty(),
        "the driver observed {} site(s) absent from the audit fixture: {extra:?}",
        extra.len()
    );
    assert_eq!(
        s.take_last_error(),
        0,
        "no latch may survive the gate: a latched error reaching CudaState::sync is the bug"
    );
}

/// The required/fallback severity decision is in the helper, and the two
/// kinds are distinguishable at the call site that matters. Device + gated.
#[test]
fn cuda_issue162_required_sites_set_the_sticky_opt_sites_do_not() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    if !gate_enabled() {
        return;
    }
    let s = device().unwrap();
    let ctx = Ctx::new(s);
    let st = ctx.stream;

    // A required site records the sticky that `execute_node` turns into Err.
    let _ = s.take_last_error();
    unsafe { minfer_site_fail_reset() };
    {
        let _g = Arm::new("launch:add_f32");
        unsafe { launch_add_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st) };
    }
    let msg = s
        .take_launch_failure()
        .expect("a required launch failure must leave a sticky record");
    assert!(
        msg.contains("launch:add_f32") && msg.contains("cudaErrorInvalidValue"),
        "the sticky must name the site and the error: {msg}"
    );
    assert!(
        s.take_launch_failure().is_none(),
        "draining must clear the sticky (a stale record would blame the next node)"
    );

    // A documented-fallback site names and clears, and sets no sticky.
    let _ = s.take_last_error();
    unsafe { minfer_site_fail_reset() };
    let rc = {
        let _g = Arm::new("launch:fa_prefill_kv__f16_causal");
        unsafe {
            launch_fa_prefill_kv(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.ci32(4),
                CAUSAL,
                4,
                2,
                128,
                0.125,
                2,
                crate::cuda::KV_LAYOUT_F16,
                136,
                st,
            )
        }
    };
    assert_eq!(rc, -1, "the fa-prefill fallback must refuse the launch");
    assert!(
        s.take_launch_failure().is_none(),
        "a documented-fallback site must not set the sticky"
    );
    assert_eq!(unsafe { minfer_site_fail_count() }, 1);
    assert_eq!(
        s.take_last_error(),
        0,
        "the fallback site cleared its latch"
    );
}

/// The positive control: with the knob off the same call launches for real
/// and leaves no report and no latch. Without it, "the injected run reported"
/// would not distinguish a working site from a site that always fails.
#[test]
fn cuda_issue162_positive_control_launches_cleanly() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    if !gate_enabled() {
        return;
    }
    let s = device().unwrap();
    let ctx = Ctx::new(s);
    let a = vec![1.0f32; 64];
    let b = vec![2.0f32; 64];
    for (i, src) in [&a, &b].iter().enumerate() {
        let e = unsafe {
            cudaMemcpy(
                ctx.p(i),
                src.as_ptr() as *const std::ffi::c_void,
                64 * 4,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(e, 0, "host fill failed: {}", cuda_error_name(e));
    }
    let _ = s.take_last_error();
    unsafe { minfer_site_fail_reset() };
    unsafe { launch_add_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, ctx.stream) };
    assert_eq!(
        unsafe { minfer_site_fail_count() },
        0,
        "knob off: the launch must not report"
    );
    assert_eq!(
        s.take_launch_failure(),
        None,
        "knob off: no sticky required-launch failure"
    );
    assert_eq!(s.take_last_error(), 0, "knob off: no latch");
    s.sync();
    let mut out = vec![0.0f32; 64];
    let e = unsafe {
        cudaMemcpy(
            out.as_mut_ptr() as *mut std::ffi::c_void,
            ctx.p(2),
            64 * 4,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    assert_eq!(e, 0, "readback failed: {}", cuda_error_name(e));
    assert_eq!(
        out,
        vec![3.0f32; 64],
        "the positive control must actually compute 1 + 2"
    );
}

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
