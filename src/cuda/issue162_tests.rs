//! `#[cfg(test)] mod issue162_tests` for `src/cuda.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
// #239: declared in `cuda::tests` now (test-only FFI / accessors).
use super::tests::{
    minfer_site_fail_code, minfer_site_fail_count, minfer_site_fail_kind, minfer_site_fail_reset,
    minfer_site_hist_len, minfer_site_hist_msg, minfer_site_hist_name, minfer_site_hist_site,
};
use std::collections::{BTreeSet, HashMap};

mod control;
mod node;
mod severity;
mod sites;
/// `minfer_site_fail_kind` for a kernel launch (`src/cuda/kernels/*.cu`).
const SITE_LAUNCH: i32 = 2;
/// `MINFER_SITE_*` / `ATTN_WIN_*` mirrors (`src/cuda/kernels/common.cuh`).
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
/// kernel-fragment` for every `<<<` site in `src/cuda/kernels/*.cu`. The gate
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
