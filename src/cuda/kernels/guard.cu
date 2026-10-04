// src/cuda/kernels/guard.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


// ─── #147/#162: read a gating return value where the call is made ───────────
// Follow-on to #145 (docs/GPU_SAFETY.md rules 4-5). A discarded
// `cudaFuncSetAttribute` return, a `<<<>>>` launch whose error is only seen by a
// *later* `cudaGetLastError()` (attribution by position), and an unchecked
// `cudaGraphDestroy` all let an error latch and resurface at a sync as a phantom
// "kernel launch error". Every dynamic-smem opt-in and launch below goes through
// `minfer_smem_optin()` / `minfer_launch_ok()`, which:
//
//   * read the call's own return value and name the failure where it is made —
//     the site, the kernel instantiation, the attribute, the requested bytes,
//     the device's queried `cudaDevAttrMaxSharedMemoryPerBlockOptin` limit and
//     `cudaGetErrorName`;
//   * SKIP a request the queried device limit already excludes, without calling
//     it: the call could only return `cudaErrorInvalidValue` (which
//     compute-sanitizer counts) and the launch cannot succeed anyway (#145's
//     rule);
//   * clear the latch they named, so nothing is left for `CudaState::sync` to
//     mis-attribute;
//   * return false → the caller REFUSES the launch instead of launching into a
//     checked error.
//
// #162 extended that pattern from the gating sites to **every** `<<<>>>` in this
// file, and split the answer in two: `minfer_launch_ok` is a REQUIRED launch and
// also records a sticky failure that `CudaBackend::execute_node` — one Rust-side
// check, not 65 signature changes — turns into an `Err`, while
// `minfer_launch_ok_opt` names and clears for a path with a DOCUMENTED fallback
// (the MMQ fast paths, the fa-prefill smem fallback, the int-returning launchers
// whose Rust caller already decides). `minfer_launch_block` is the injection
// lever every ordinary site shares. `scripts/check_cuda_launch_returns.py` audits
// the source, the `issue162_tests` device gates drive each audited site.
//
// The last report is also kept in statics (`minfer_site_fail_*`) and the ordered
// history (`minfer_site_hist_*`), so the `issue147_tests` / `issue162_tests`
// device gates can assert the site, the requested value, the kernel
// instantiation and the error *name* — a gate that only asserts "a message
// appeared" cannot see a message that names the wrong call. Serial device runs
// only, like `CudaState` itself.
#define MINFER_SITE_MSG_MAX 640

// 0 = none, 1 = dynamic-smem attribute, 2 = kernel launch, 3 = a latched error
// found *before* a launch (an earlier call's, never blamed on the launch).
enum {
    MINFER_SITE_NONE = 0,
    MINFER_SITE_ATTR = 1,
    MINFER_SITE_LAUNCH = 2,
    MINFER_SITE_PREEXISTING = 3
};

static char g_site_msg[MINFER_SITE_MSG_MAX];
static char g_site_name[96];
static int g_site_fail_count = 0;
static int g_site_last_kind = MINFER_SITE_NONE;
static int g_site_last_code = 0;
static int g_site_last_bytes = 0;
static int g_site_last_limit = 0;

// ─── #162: the sticky required-launch failure ────────────────────────────────
// A `<<<>>>` in a launcher whose output nothing else recomputes must not let the
// op proceed: `minfer_launch_ok` records the failure here, and
// `CudaBackend::execute_node` — ONE Rust-side check, not one per launcher —
// drains it and returns `Err` naming the site. `minfer_launch_ok_opt` (a
// documented fallback) deliberately does not set it. Serial device runs only.
static int g_launch_fail_pending = 0;
static char g_launch_fail_site[96];
static char g_launch_fail_name[80];
static int g_launch_fail_code = 0;

// Every launch failure named at a site, in order, so the #162 gate can assert
// that a driven dispatch reached *this* site (the report's single "last" slot
// cannot see a second launch in the same call).
#define MINFER_SITE_HIST_MAX 1024
static char g_site_hist_site[MINFER_SITE_HIST_MAX][96];
static char g_site_hist_name[MINFER_SITE_HIST_MAX][96];
static char g_site_hist_msg[MINFER_SITE_HIST_MAX][256];
static int g_site_hist_len = 0;

extern "C" int minfer_site_fail_count(void) { return g_site_fail_count; }
extern "C" int minfer_site_fail_kind(void) { return g_site_last_kind; }
extern "C" int minfer_site_fail_code(void) { return g_site_last_code; }
extern "C" int minfer_site_fail_bytes(void) { return g_site_last_bytes; }
extern "C" int minfer_site_fail_limit(void) { return g_site_last_limit; }
extern "C" const char* minfer_site_fail_site(void) { return g_site_name; }
extern "C" const char* minfer_site_fail_message(void) { return g_site_msg; }
extern "C" int minfer_launch_fail_pending(void) { return g_launch_fail_pending; }
extern "C" const char* minfer_launch_fail_site(void) { return g_launch_fail_site; }
extern "C" const char* minfer_launch_fail_name(void) { return g_launch_fail_name; }
extern "C" int minfer_launch_fail_code(void) { return g_launch_fail_code; }
extern "C" void minfer_launch_fail_clear(void) {
    g_launch_fail_pending = 0;
    g_launch_fail_site[0] = '\0';
    g_launch_fail_name[0] = '\0';
    g_launch_fail_code = 0;
}
extern "C" int minfer_site_hist_len(void) { return g_site_hist_len; }
extern "C" const char* minfer_site_hist_site(int i) {
    return (i >= 0 && i < g_site_hist_len) ? g_site_hist_site[i] : "";
}
extern "C" const char* minfer_site_hist_name(int i) {
    return (i >= 0 && i < g_site_hist_len) ? g_site_hist_name[i] : "";
}
extern "C" const char* minfer_site_hist_msg(int i) {
    return (i >= 0 && i < g_site_hist_len) ? g_site_hist_msg[i] : "";
}
extern "C" void minfer_site_hist_reset(void) { g_site_hist_len = 0; }

// Called *after* `minfer_site_report`, so the entry carries the message the site
// printed (the gate asserts the full text, not just the site token).
static void minfer_site_history_add(const char* site, const char* name) {
    if (g_site_hist_len >= MINFER_SITE_HIST_MAX) return;
    snprintf(g_site_hist_site[g_site_hist_len], sizeof(g_site_hist_site[0]), "%s", site);
    snprintf(g_site_hist_name[g_site_hist_len], sizeof(g_site_hist_name[0]), "%s", name);
    snprintf(g_site_hist_msg[g_site_hist_len], sizeof(g_site_hist_msg[0]), "%s", g_site_msg);
    g_site_hist_len++;
}

// Start a fresh observation: the device gates assert one failure at a time.
extern "C" void minfer_site_fail_reset(void) {
    g_site_fail_count = 0;
    g_site_last_kind = MINFER_SITE_NONE;
    g_site_last_code = 0;
    g_site_last_bytes = 0;
    g_site_last_limit = 0;
    g_site_name[0] = '\0';
    g_site_msg[0] = '\0';
    minfer_launch_fail_clear();
    minfer_site_hist_reset();
}

static void minfer_site_report(const char* site, int kind, int code, int bytes, int limit,
                               const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_site_msg, sizeof(g_site_msg), fmt, ap);
    va_end(ap);
    g_site_fail_count++;
    g_site_last_kind = kind;
    g_site_last_code = code;
    g_site_last_bytes = bytes;
    g_site_last_limit = limit;
    snprintf(g_site_name, sizeof(g_site_name), "%s", site);
    fprintf(stderr, "minfer/cuda: %s\n", g_site_msg);
}

// Test injection (issue #147), env-gated like #145's `MINFER_TEST_LATCH_ERROR`:
// `MINFER_TEST_CALL_FAIL` is a comma-separated list of site tokens, or `all`.
// This is the device half of the one failure-injection seam (issue #171,
// `src/testfail.rs` documents the Rust half and the shared matching rule); the
// Rust matcher `testfail::injection_names_site` and this function must stay
// exact-token identical (no substring matching).
// A named site performs its REAL call with a value that makes it fail — an
// over-limit attribute request, an over-limit dynamic-smem launch, or (Rust
// side) `cudaGraphDestroy` on the `cudaGraphExec_t` — so the failure is a real
// latch the site must name and clear, not a synthetic return value. That is what
// makes "no latched error reaches sync" a meaningful assertion. The knob is off
// in every default run, so `compute-sanitizer --tool memcheck` over the suite
// never sees such a call. Token matching is exact (no substring surprises).
bool minfer_test_call_fails(const char* site) {
    const char* v = getenv("MINFER_TEST_CALL_FAIL");
    if (v == 0) return false;
    const size_t sl = strlen(site);
    const char* p = v;
    while (*p != '\0') {
        while (*p == ',' || *p == ' ') p++;
        const char* q = p;
        while (*q != '\0' && *q != ',') q++;
        size_t n = (size_t)(q - p);
        while (n > 0 && p[n - 1] == ' ') n--;
        if ((n == 3 && strncmp(p, "all", 3) == 0) || (n == sl && strncmp(p, site, n) == 0))
            return true;
        p = q;
    }
    return false;
}

// The device's `cudaDevAttrMaxSharedMemoryPerBlockOptin`, queried once.
// Negative = the query failed ("unknown"), in which case no request is skipped
// on its account and the call's own return value decides.
static int g_minfer_optin_limit = -2;
int minfer_optin_limit(void) {
    if (g_minfer_optin_limit == -2) {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaError_t e = cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (e != cudaSuccess) {
            cudaGetLastError();
            v = -1;
        }
        g_minfer_optin_limit = v;
    }
    return g_minfer_optin_limit;
}

// The dynamic-smem opt-in for one kernel instantiation (issue #147), the single
// place a `cudaFuncSetAttribute` return value is read. Returns true when `bytes`
// bytes are in force for `fn`; false means the following launch must be REFUSED,
// because it cannot succeed.
bool minfer_smem_optin(const char* site, const char* kernel_name, const void* fn,
                              int bytes) {
    const bool injected = minfer_test_call_fails(site);
    // The 48 KiB default cap admits it — nothing to opt into, nothing to check.
    if (!injected && bytes <= 48 * 1024) return true;
    const int limit = minfer_optin_limit();
    if (!injected && limit > 0 && bytes > limit) {
        // #145's rule: a request the queried device limit already excludes is
        // skipped WITHOUT calling it — the call could only return
        // cudaErrorInvalidValue, which compute-sanitizer counts, and the launch
        // cannot succeed anyway.
        minfer_site_report(site, MINFER_SITE_ATTR, (int)cudaErrorInvalidValue, bytes, limit,
                           "cudaFuncSetAttribute(%s, "
                           "cudaFuncAttributeMaxDynamicSharedMemorySize, %d B) SKIPPED: the "
                           "request exceeds cudaDevAttrMaxSharedMemoryPerBlockOptin (%d B), the "
                           "launch cannot succeed and the call is not made (#147/%s)",
                           kernel_name, bytes, limit, site);
        return false;
    }
    // The injection asks one page over the device limit: a real failing call
    // with a real latch, which the failure path below must name and clear.
    const int ask = injected ? (limit > 0 ? limit + 4096 : 48 * 1024 + 4096) : bytes;
    cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ask);
    if (e != cudaSuccess) {
        minfer_site_report(site, MINFER_SITE_ATTR, (int)e, ask, limit,
                           "cudaFuncSetAttribute(%s, "
                           "cudaFuncAttributeMaxDynamicSharedMemorySize, %d B) failed: %s (%d); "
                           "device cudaDevAttrMaxSharedMemoryPerBlockOptin = %d B — the launch "
                           "is refused (#147/%s)",
                           kernel_name, ask, cudaGetErrorName(e), (int)e, limit, site);
        cudaGetLastError();  // this site owns the error; never leave it for sync
        return false;
    }
    return true;
}

// A pre-launch latch is an *earlier* call's error and is reported as such, so
// the post-launch read below is a launch check and not attribution by position.
void minfer_launch_prelude(const char* site, const char* kernel_name) {
    cudaError_t pre = cudaGetLastError();
    if (pre != cudaSuccess) {
        minfer_site_report(site, MINFER_SITE_PREEXISTING, (int)pre, 0, 0,
                           "a latched CUDA error %s (%d) was found before the %s launch and is "
                           "NOT attributed to it: an earlier call on this thread did not check "
                           "its own return value (#147/%s)",
                           cudaGetErrorName(pre), (int)pre, kernel_name, site);
    }
}

// The launch's dynamic-smem argument, made over-limit when the test knob names
// the launch site. Probed on GB10/sm_121: an over-limit dynamic-smem launch is
// rejected by the launch call itself with cudaErrorInvalidValue and the kernel
// never runs (see the closing comment on #147).
size_t minfer_launch_smem(const char* site, size_t smem) {
    return minfer_test_call_fails(site) ? smem + (16u << 20) : smem;
}

// #162: the launch geometry, the injection lever every site shares. When the test
// knob names the site the block becomes 4096 threads (over the 1024/block device
// limit), so `<<<>>>` itself returns cudaErrorInvalidValue for real and the
// kernel never runs — probed on GB10/sm_121. A thread-count lever (rather than a
// dynamic-smem one) works for every launcher, including those with no dynamic
// smem: adding a site's coverage is data, not another bespoke mechanism.
#define MINFER_ILLEGAL_BLOCK 4096u
dim3 minfer_launch_block(const char* site, dim3 block) {
    return minfer_test_call_fails(site) ? dim3(MINFER_ILLEGAL_BLOCK, 1, 1) : block;
}
dim3 minfer_launch_block(const char* site, unsigned block) {
    return minfer_test_call_fails(site) ? dim3(MINFER_ILLEGAL_BLOCK, 1, 1)
                                        : dim3(block, 1, 1);
}

// The error of the launch just issued: a `<<<>>>` has no return value, and the
// immediately-following cudaGetLastError is the documented *launch* check
// (nothing runs in between). `required` selects the severity: a required launch
// also records the sticky that `CudaBackend::execute_node` turns into an `Err`,
// while an `_opt` site (a documented fallback) only names and clears.
static bool minfer_launch_read(const char* site, const char* kernel_name, bool required) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    minfer_site_report(site, MINFER_SITE_LAUNCH, (int)e, 0, 0,
                       "kernel launch %s failed: %s (%d) — the launch is refused (#162/%s)",
                       kernel_name, cudaGetErrorName(e), (int)e, site);
    minfer_site_history_add(site, kernel_name);
    cudaGetLastError();  // this site owns the error
    if (required) {
        g_launch_fail_pending = 1;
        g_launch_fail_code = (int)e;
        snprintf(g_launch_fail_site, sizeof(g_launch_fail_site), "%s", site);
        snprintf(g_launch_fail_name, sizeof(g_launch_fail_name), "%s", kernel_name);
    }
    return false;
}

bool minfer_launch_ok(const char* site, const char* kernel_name) {
    return minfer_launch_read(site, kernel_name, true);
}
bool minfer_launch_ok_opt(const char* site, const char* kernel_name) {
    return minfer_launch_read(site, kernel_name, false);
}

