use super::pool::{chunk, get_pool, mm_rows, MmJob, PoolJob};
use super::*;
use std::sync::atomic::Ordering;

/// CPU fallback for f32 activation: quantize → call existing dot product.
/// K-quant weights (Q4_K/Q5_K/Q6_K) quantize activations to Q8_K (256-element
/// blocks with precomputed bsums — llama.cpp's format, so the dots never
/// re-reduce the activation); the simple types keep Q8_0.
pub fn cpu_quant_matmul_f32(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    od: usize,
    id: usize,
    nt: usize,
) {
    match w.ttype {
        TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K => {
            let n_super = id / 256;
            let mut qb = vec![0u8; nt * n_super * Q8KB];
            crate::quants::quantize_row_q8_k_buf(x, nt, id, &mut qb);
            cpu_quant_matmul(w, &qb, out, od, id, nt)
        }
        _ => {
            let nbe = id / 32;
            let mut qb = vec![0u8; nt * nbe * Q8B];
            crate::quants::quantize_row_q8_0_buf(x, nt, id, &mut qb);
            cpu_quant_matmul(w, &qb, out, od, id, nt)
        }
    }
}

/// Threaded row-parallel matmul. Each row is computed by exactly one worker
/// with the identical code path as the single-threaded loop, so the output is
/// bit-identical regardless of thread count. Small matmuls run inline.
pub fn cpu_quant_matmul(w: &Tensor, x: &[u8], out: &mut [f32], od: usize, id: usize, nt: usize) {
    // SAFETY: w.data()/x/out are borrowed by the caller for the whole call;
    // mm_rows only touches row o of out from the owner of chunk containing o.
    let job = MmJob {
        ttype: w.ttype,
        w: w.data().as_ptr(),
        x: x.as_ptr(),
        out: out.as_mut_ptr(),
        od,
        id,
        nt,
    };
    let threads = cpu_threads();
    let macs = od.saturating_mul(id).saturating_mul(nt);
    if threads <= 1 || od < 2 || macs < MIN_PARALLEL_MACS {
        unsafe { mm_rows(&job, 0, od) };
        return;
    }
    let pool = get_pool(threads - 1);
    let _gate = pool.gate.lock().unwrap();
    *pool.job.lock().unwrap() = PoolJob::MatMul(job);
    pool.gen.fetch_add(1, Ordering::SeqCst);
    // main thread participates as worker `n`
    let (r0, r1) = chunk(pool.n + 1, pool.n, od);
    unsafe { mm_rows(&job, r0, r1) };
    let mut spins = 0usize;
    while pool.done.load(Ordering::SeqCst) < pool.n {
        spins += 1;
        if spins >= 8_000 {
            spins = 0;
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
    pool.done.store(0, Ordering::SeqCst);
}
