// Compute kernel dispatch layer.
//   CPU (AVX2/NEON/scalar) is always available as fallback.
//   MPS (Apple Silicon GPU) is enabled at runtime when Metal is available.

use crate::block::{Q41B, Q4B, Q4KB, Q6KB, Q8B, Q8KB};
use crate::tensor::{Tensor, TensorType};

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
const MIN_PARALLEL_MACS: usize = 1 << 20; // 1M MACs

/// One matmul job shared by all workers (Copy so each worker snapshots it).
#[derive(Clone, Copy)]
struct MmJob {
    ttype: TensorType,
    w: *const u8,
    x: *const u8,
    out: *mut f32,
    od: usize,
    id: usize,
    nt: usize,
}

/// Generic parallel-for job: runs `f(ctx, start, end)` per contiguous task
/// range. SAFETY (same contract as `MmJob`): `ctx` must stay valid for the
/// pool call's duration and `f` must only touch disjoint outputs per range.
#[derive(Clone, Copy)]
struct ParForJob {
    total: usize,
    ctx: *const (),
    f: unsafe fn(*const (), usize, usize),
}

#[derive(Clone, Copy)]
enum PoolJob {
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
unsafe fn mm_rows(job: &MmJob, r0: usize, r1: usize) {
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

struct Pool {
    /// Worker threads (the main thread participates as worker `n`).
    n: usize,
    gen: AtomicUsize,
    done: AtomicUsize,
    job: Mutex<PoolJob>,
}

fn chunk(parts: usize, idx: usize, total: usize) -> (usize, usize) {
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

fn get_pool(n_workers: usize) -> Arc<Pool> {
    static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
    Arc::clone(POOL.get_or_init(|| {
        let pool = Arc::new(Pool {
            n: n_workers,
            gen: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
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

// Shared embedding row getter (moved from qwen2/forward.rs when the
// imperative forward was replaced by the graph path — Phase 6).
pub fn embed_tokens(ids: &[u32], t: &crate::tensor::Tensor, out: &mut [f32], ne: usize) {
    match t.ttype {
        TensorType::Q4_0 | TensorType::Q8_0 | TensorType::Q4_1 => {
            let is_q4_1 = t.ttype == TensorType::Q4_1;
            let blk = 32usize;
            let nbp = (ne + blk - 1) / blk;
            let bb = t.ttype.type_size();
            let is8 = t.ttype == TensorType::Q8_0;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                for b in 0..nbp {
                    let off = (idx * nbp + b) * bb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off],
                        t.data[off + 1],
                    ]));
                    let m = if is_q4_1 {
                        crate::block::fp16_to_f32(u16::from_le_bytes([
                            t.data[off + 2],
                            t.data[off + 3],
                        ]))
                    } else {
                        0.0
                    };
                    let mv = blk.min(ne - b * blk);
                    if is8 {
                        for j in 0..mv {
                            out[doff + b * blk + j] = (t.data[off + 2 + j] as i8) as f32 * d;
                        }
                    } else if is_q4_1 {
                        for j in 0..16 {
                            let byte = t.data[off + 4 + j];
                            if j < mv {
                                out[doff + b * blk + j] = (byte & 0x0F) as f32 * d + m;
                            }
                            if j + 16 < mv {
                                out[doff + b * blk + j + 16] = (byte >> 4) as f32 * d + m;
                            }
                        }
                    } else {
                        for j in 0..16 {
                            let byte = t.data[off + 2 + j];
                            if j < mv {
                                out[doff + b * blk + j] = ((byte & 0x0F) as i8 - 8) as f32 * d;
                            }
                            if j + 16 < mv {
                                out[doff + b * blk + j + 16] = ((byte >> 4) as i8 - 8) as f32 * d;
                            }
                        }
                    }
                }
            }
        }
        TensorType::Q5_0 => {
            let blk = 32usize;
            let nbp = (ne + blk - 1) / blk;
            let bb = 22usize;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                for b in 0..nbp {
                    let off = (idx * nbp + b) * bb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off],
                        t.data[off + 1],
                    ]));
                    let qh = u32::from_le_bytes([
                        t.data[off + 2],
                        t.data[off + 3],
                        t.data[off + 4],
                        t.data[off + 5],
                    ]);
                    let qs = &t.data[off + 6..off + 22];
                    let mv = blk.min(ne - b * blk);
                    for j in 0..16 {
                        let xh0 = ((qh >> j) & 1) as u32;
                        let xh1 = ((qh >> (j + 16)) & 1) as u32;
                        if j < mv {
                            out[doff + b * blk + j] =
                                (((qs[j] & 0x0F) as i32 | ((xh0 << 4) as i32)) - 16) as f32 * d;
                        }
                        if j + 16 < mv {
                            out[doff + b * blk + j + 16] =
                                ((((qs[j] >> 4) & 0x0F) as i32 | ((xh1 << 4) as i32)) - 16) as f32
                                    * d;
                        }
                    }
                }
            }
        }
        TensorType::Q5_1 => {
            let blk = 32usize;
            let nbp = (ne + blk - 1) / blk;
            let bb = 24usize;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                for b in 0..nbp {
                    let off = (idx * nbp + b) * bb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off],
                        t.data[off + 1],
                    ]));
                    let m = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off + 2],
                        t.data[off + 3],
                    ]));
                    let qh = u32::from_le_bytes([
                        t.data[off + 4],
                        t.data[off + 5],
                        t.data[off + 6],
                        t.data[off + 7],
                    ]);
                    let qs = &t.data[off + 8..off + 24];
                    let mv = blk.min(ne - b * blk);
                    for j in 0..16 {
                        let xh0 = ((qh >> j) & 1) as u32;
                        let xh1 = ((qh >> (j + 16)) & 1) as u32;
                        if j < mv {
                            out[doff + b * blk + j] =
                                ((qs[j] & 0x0F) as i32 | ((xh0 << 4) as i32)) as f32 * d + m;
                        }
                        if j + 16 < mv {
                            out[doff + b * blk + j + 16] =
                                (((qs[j] >> 4) & 0x0F) as i32 | ((xh1 << 4) as i32)) as f32 * d + m;
                        }
                    }
                }
            }
        }
        TensorType::Q4_K => {
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                for s in 0..n_super {
                    let off = (idx * n_super + s) * Q4KB;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off],
                        t.data[off + 1],
                    ]));
                    let dmin = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off + 2],
                        t.data[off + 3],
                    ]));
                    let sc_arr: &[u8; 12] = t.data[off + 4..off + 16].try_into().unwrap();
                    let (scales, mins) = crate::block::unpack_q4k_scales(sc_arr);
                    let qs = &t.data[off + 16..off + 144];

                    // Deinterleave qs: 4 chunks of 32 bytes, each covers 2 subblocks
                    // chunk[l] lo nibble → sub 2*chunk, elem l
                    // chunk[l] hi nibble → sub 2*chunk+1, elem l
                    let mut nibbles = [0i32; 256];
                    for chunk_idx in 0..4 {
                        let chunk = &qs[chunk_idx * 32..chunk_idx * 32 + 32];
                        for l in 0..32 {
                            nibbles[(2 * chunk_idx) * 32 + l] = (chunk[l] & 0x0F) as i32;
                            nibbles[(2 * chunk_idx + 1) * 32 + l] = (chunk[l] >> 4) as i32;
                        }
                    }

                    for sub in 0..8 {
                        let sc_val = scales[sub];
                        let mm_val = mins[sub];
                        let dl = d * sc_val as f32;
                        let ml = dmin * mm_val as f32;
                        let base = doff + s * 256 + sub * 32;
                        for k in 0..32 {
                            out[base + k] = dl * nibbles[sub * 32 + k] as f32 - ml;
                        }
                    }
                }
            }
        }
        TensorType::Q5_K => {
            let q5_kb: usize = 176;
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                for s in 0..n_super {
                    let off = (idx * n_super + s) * q5_kb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off],
                        t.data[off + 1],
                    ]));
                    let dmin = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off + 2],
                        t.data[off + 3],
                    ]));
                    let sc_arr: &[u8; 12] = t.data[off + 4..off + 16].try_into().unwrap();
                    let (scales, mins) = crate::block::unpack_q4k_scales(sc_arr);
                    let qh = &t.data[off + 16..off + 48];
                    let qs = &t.data[off + 48..off + 176];

                    // Deinterleave qs nibbles: 4 chunks of 32 bytes, covering 2 subblocks each
                    let mut nb = [0u8; 256];
                    for ci in 0..4 {
                        let chunk = &qs[ci * 32..ci * 32 + 32];
                        for l in 0..32 {
                            nb[(2 * ci) * 32 + l] = chunk[l] & 0x0F;
                            nb[(2 * ci + 1) * 32 + l] = chunk[l] >> 4;
                        }
                    }

                    for sub in 0..8 {
                        let dl = d * scales[sub] as f32;
                        let ml = dmin * mins[sub] as f32;
                        let base = doff + s * 256 + sub * 32;
                        for j in 0..32 {
                            // Q5_K qh layout: element (sub s, pos j) high bit = qh[j] bit s
                            let hi_bit = ((qh[j] >> sub) & 1) as u8;
                            // Q5_K unsigned (no -16, unlike Q5_0): w = unsigned_5bit * dl - ml
                            let w = nb[sub * 32 + j] as f32 + 16.0 * hi_bit as f32;
                            out[base + j] = dl * w - ml;
                        }
                    }
                }
            }
        }
        TensorType::Q6_K => {
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                for s in 0..n_super {
                    let off = (idx * n_super + s) * Q6KB;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                        t.data[off + 208],
                        t.data[off + 209],
                    ]));
                    let base_out = doff + s * 256;

                    let ql = &t.data[off..off + 128];
                    let qh = &t.data[off + 128..off + 192];
                    let sc = &t.data[off + 192..off + 208];

                    for n in 0..2 {
                        let ql_off = n * 64;
                        let qh_off = n * 32;
                        let out_off = n * 128;
                        for l in 0..32 {
                            let is = l / 16;
                            let si = is + n * 8;

                            let q0 = (((ql[ql_off + l] & 0xF) as i32)
                                | ((((qh[qh_off + l] >> 0) & 3) as i32) << 4))
                                - 32;
                            let q1 = (((ql[ql_off + l + 32] & 0xF) as i32)
                                | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4))
                                - 32;
                            let q2 = (((ql[ql_off + l] >> 4) as i32)
                                | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4))
                                - 32;
                            let q3 = (((ql[ql_off + l + 32] >> 4) as i32)
                                | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4))
                                - 32;

                            out[base_out + out_off + l] = d * (sc[si + 0] as i8 as f32) * q0 as f32;
                            out[base_out + out_off + l + 32] =
                                d * (sc[si + 2] as i8 as f32) * q1 as f32;
                            out[base_out + out_off + l + 64] =
                                d * (sc[si + 4] as i8 as f32) * q2 as f32;
                            out[base_out + out_off + l + 96] =
                                d * (sc[si + 6] as i8 as f32) * q3 as f32;
                        }
                    }
                }
            }
        }
        _ => panic!("unsupported weight type {:?} in embed_tokens", t.ttype),
    }
}
