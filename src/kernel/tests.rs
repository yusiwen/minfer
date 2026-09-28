//! `#[cfg(test)] mod tests` for `src/kernel.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// Regression test for the pooled `par_for` non-reentrancy race (issue 2).
/// The process-global pool shares one `job`/`gen`/`done`; concurrent
/// submissions from several threads must not clobber each other. Before the
/// `gate` mutex, a caller could return while its own range was incomplete.
struct TestCtx {
    out: *mut usize,
    marker: usize,
}

unsafe fn write_marker(ctx: *const (), r0: usize, r1: usize) {
    let c = &*(ctx as *const TestCtx);
    for i in r0..r1 {
        *c.out.add(i) = c.marker;
    }
}

#[test]
fn par_for_concurrent_submissions_safe() {
    let n_threads = 8;
    let total = 64;
    let mut handles = Vec::new();
    for t in 0..n_threads {
        handles.push(std::thread::spawn(move || {
            let mut out = vec![0usize; total];
            let marker = t + 1;
            let ctx = TestCtx {
                out: out.as_mut_ptr(),
                marker,
            };
            // Blocking: each submission must fully cover [0, total) with its
            // own marker before returning.
            par_for(total, &ctx as *const _ as *const (), write_marker);
            out
        }));
    }
    for (t, h) in handles.into_iter().enumerate() {
        let out = h.join().unwrap();
        let marker = t + 1;
        assert!(
            out.iter().all(|&v| v == marker),
            "thread {t} got a mixed/corrupt range: {out:?} (expected all {marker})"
        );
    }
}
