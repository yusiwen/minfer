use super::*;

// ============================================================
// CPU thread pool (matmul row parallelism)
//
// Decode runs ~250 matmuls/token; spawning threads per matmul costs
// ~170 µs (measured) — a persistent pool with an atomic generation
// handoff costs ~1-3 µs per dispatch instead. Workers spin briefly
// then yield; the main thread participates as the last worker.
// The pool is process-global and lazily spawned on the first threaded
// matmul, so `set_cpu_threads` must be called before inference starts.
// ============================================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static CPU_THREADS: AtomicUsize = AtomicUsize::new(0); // 0 = auto-detect

/// Override the CPU worker count (CLI `--threads`). Must be called before
/// the first matmul (the pool is spawned lazily).
pub fn set_cpu_threads(n: usize) {
    CPU_THREADS.store(n.max(1), Ordering::Relaxed);
}

/// Effective CPU thread count: explicit override, else auto (macOS P-core
/// count — E-cores measurably hurt matmul throughput — else available
/// parallelism).
pub fn cpu_threads() -> usize {
    let n = CPU_THREADS.load(Ordering::Relaxed);
    if n > 0 {
        return n;
    }
    static AUTO: OnceLock<usize> = OnceLock::new();
    *AUTO.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            // hw.perflevel0.logicalcpu = performance-core count (10 on M4 Pro).
            if let Ok(out) = std::process::Command::new("sysctl")
                .args(["-n", "hw.perflevel0.logicalcpu"])
                .output()
            {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    if let Ok(n) = s.trim().parse::<usize>() {
                        if n >= 1 {
                            return n;
                        }
                    }
                }
            }
        }
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// Minimum matmul work for which the pool is worth dispatching (rows × id
/// MACs). Small matmuls (attn_k/v on GQA models, 512 rows) run inline.
/// `pub(crate)`: the f16 matmul (#141) in `vec_ops` uses the same threshold so
/// the two CPU weight paths thread at the same shape.
pub(crate) const MIN_PARALLEL_MACS: usize = 1 << 20; // 1M MACs

/// One matmul job shared by all workers (Copy so each worker snapshots it).
#[derive(Clone, Copy)]
pub(super) struct MmJob {
    pub(super) ttype: TensorType,
    pub(super) w: *const u8,
    pub(super) x: *const u8,
    pub(super) out: *mut f32,
    pub(super) od: usize,
    pub(super) id: usize,
    pub(super) nt: usize,
}

/// Generic parallel-for job: runs `f(ctx, start, end)` per contiguous task
/// range. SAFETY (same contract as `MmJob`): `ctx` must stay valid for the
/// pool call's duration and `f` must only touch disjoint outputs per range.
#[derive(Clone, Copy)]
pub(super) struct ParForJob {
    total: usize,
    ctx: *const (),
    f: unsafe fn(*const (), usize, usize),
}

#[derive(Clone, Copy)]
pub(super) enum PoolJob {
    MatMul(MmJob),
    ParFor(ParForJob),
}

/// SAFETY: `MmJob` pointers are only dereferenced inside `mm_rows`, which runs
/// strictly within a `cpu_quant_matmul` call — the caller keeps the borrowed
/// `w`/`x`/`out` alive and waits for all workers (`done == n`) before the
/// borrows end. Each worker thread only touches rows it owns, so the data
/// races are impossible by construction. Same contract for `ParForJob`.
unsafe impl Send for MmJob {}
unsafe impl Send for PoolJob {}

/// Row kernel: computes rows [r0, r1) of one matmul. SAFETY: `w`/`x`/`out`
/// must stay valid for the pool call's duration; each row `o ∈ [r0, r1)` is
/// written exactly once by the owning worker (out[t*od+o] are disjoint per o).
pub(super) unsafe fn mm_rows(job: &MmJob, r0: usize, r1: usize) {
    let od = job.od;
    let id = job.id;
    let nt = job.nt;
    match job.ttype {
        TensorType::Q4_0 => {
            let nb = id / 32;
            let ws = nb * Q4B;
            let rowb = nb * Q8B;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q4_0_q8_0(wrow, xrow);
                }
            }
        }
        TensorType::Q4_1 => {
            let nb = id / 32;
            let ws = nb * Q41B;
            let rowb = nb * Q8B;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q4_1_q8_0(wrow, xrow);
                }
            }
        }
        TensorType::Q4_K => {
            let nk = id / 256;
            let ws = nk * Q4KB;
            let rowb = nk * Q8KB;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q4_k_q8_k(wrow, xrow);
                }
            }
        }
        TensorType::Q5_K => {
            let nk = id / 256;
            let ws = nk * 176;
            let rowb = nk * Q8KB;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q5_k_q8_k(wrow, xrow);
                }
            }
        }
        TensorType::Q6_K => {
            let nk = id / 256;
            let ws = nk * Q6KB;
            let rowb = nk * Q8KB;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q6_k_q8_k(wrow, xrow);
                }
            }
        }
        TensorType::Q5_0 => {
            let nb = id / 32;
            let ws = nb * 22;
            let rowb = nb * Q8B;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q5_0_q8_0(wrow, xrow);
                }
            }
        }
        TensorType::Q5_1 => {
            let nb = id / 32;
            let ws = nb * 24;
            let rowb = nb * Q8B;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q5_1_q8_0(wrow, xrow);
                }
            }
        }
        TensorType::Q8_0 => {
            let nb = id / 32;
            let ws = nb * Q8B;
            let rowb = nb * Q8B;
            for o in r0..r1 {
                let wrow = std::slice::from_raw_parts(job.w.add(o * ws), ws);
                for t in 0..nt {
                    let xrow = std::slice::from_raw_parts(job.x.add(t * rowb), rowb);
                    *job.out.add(t * od + o) = crate::quants::dot_q8_0_q8_0(wrow, xrow);
                }
            }
        }
        _ => panic!("unsupported weight type {:?} in quant_matmul", job.ttype),
    }
}

pub(super) struct Pool {
    /// Worker threads (the main thread participates as worker `n`).
    pub(super) n: usize,
    pub(super) gen: AtomicUsize,
    pub(super) done: AtomicUsize,
    pub(super) job: Mutex<PoolJob>,
    /// Serializes the whole submit → wait critical section. The pool's
    /// gen/done/job protocol is single-submission: two concurrent callers
    /// (e.g. parallel test threads, or the multi-slot server) would clobber
    /// `job` and share `done`, letting a caller return before its own range was
    /// computed — and then its stack-local ctx (e.g. `AttnCtx` from
    /// `attn_heads`) dies while late workers still read it (UB). Worker threads
    /// never take `gate` (they only read `job`), so a submission runs to
    /// completion while its caller holds the lock; no deadlock.
    pub(super) gate: Mutex<()>,
}

pub(super) fn chunk(parts: usize, idx: usize, total: usize) -> (usize, usize) {
    let a = total * idx / parts;
    let b = total * (idx + 1) / parts;
    (a, b)
}

fn worker_loop(pool: Arc<Pool>, my_idx: usize, mut seen: usize) {
    let mut spins = 0usize;
    loop {
        while pool.gen.load(Ordering::SeqCst) == seen {
            spins += 1;
            if spins >= 8_000 {
                spins = 0;
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        seen = pool.gen.load(Ordering::SeqCst);
        let job = *pool.job.lock().unwrap();
        match job {
            PoolJob::MatMul(m) => {
                let (r0, r1) = chunk(pool.n + 1, my_idx, m.od);
                unsafe { mm_rows(&m, r0, r1) };
            }
            PoolJob::ParFor(p) => {
                let (r0, r1) = chunk(pool.n + 1, my_idx, p.total);
                unsafe { (p.f)(p.ctx, r0, r1) };
            }
        }
        pool.done.fetch_add(1, Ordering::SeqCst);
    }
}

pub(super) fn get_pool(n_workers: usize) -> Arc<Pool> {
    static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
    Arc::clone(POOL.get_or_init(|| {
        let pool = Arc::new(Pool {
            n: n_workers,
            gen: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            gate: Mutex::new(()),
            job: Mutex::new(PoolJob::MatMul(MmJob {
                ttype: TensorType::Q8_0, // placeholder; overwritten before each dispatch
                w: std::ptr::null(),
                x: std::ptr::null(),
                out: std::ptr::null_mut(),
                od: 0,
                id: 0,
                nt: 0,
            })),
        });
        // Worker threads run forever (detached handles); the pool lives in a
        // OnceLock for the process lifetime.
        for i in 0..n_workers {
            let p = Arc::clone(&pool);
            std::thread::spawn(move || worker_loop(p, i, 0));
        }
        pool
    }))
}

/// Parallel-for over `total` tasks using the persistent pool: `f(ctx, start,
/// end)` is called once per contiguous chunk (one per worker + the main
/// thread). SAFETY: `ctx` must stay valid for the call; `f` must only write
/// outputs indexed by its own range.
pub fn par_for(total: usize, ctx: *const (), f: unsafe fn(*const (), usize, usize)) {
    let threads = cpu_threads();
    if threads <= 1 || total < 2 {
        unsafe { (f)(ctx, 0, total) };
        return;
    }
    let pool = get_pool(threads - 1);
    let _gate = pool.gate.lock().unwrap();
    *pool.job.lock().unwrap() = PoolJob::ParFor(ParForJob { total, ctx, f });
    pool.gen.fetch_add(1, Ordering::SeqCst);
    let (r0, r1) = chunk(pool.n + 1, pool.n, total);
    unsafe { (f)(ctx, r0, r1) };
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
