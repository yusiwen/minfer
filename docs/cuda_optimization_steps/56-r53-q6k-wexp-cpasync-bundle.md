# 56 · r53 — q6_K bundle: W_exp pre-expansion plane + cp.async B staging (LANDED)

> **Result**: r44 (pre-expanded dense W_exp — deletes WORK) and r45 (cp.async staging — deletes WAIT), two mechanisms each **wall-neutral on their own**, are bundled into the q6_K BT kernel: B staging becomes a pure cp.async bulk copy out of the W_exp plane. The ffn_down kernel goes **16.06 → 12.76 ms (−20.5%)** (≈ r44's −10.9% + r45's −10.2% stacking near-additively), attn_v −15.9%; whole prefill **3024.7 → 3176.9 (+5.03%) = 1.05× vs-llama — the q6_K line closes here**. The price is +1.52 GB of device memory. The doc also lends its name to a rule: a fallback-correct optimization nearly landed silently as "zero fast-path launches."
> **Commit**: `83fee77` (code, +265/−20) + `4907d9f` (record). **Date**: 2026-09-06.

## 1. Background — where things stood

r44 and r45 are two special negative results on the q6_K line: **both parity green, both kernel-faster, both wall-neutral** — and both reverted.

- **r44** (the dense-index W_exp fix): the pre-expanded-B idea first failed parity in r43 (the dense plane was indexed by the padded raw-W stride); r44 found that one line's root cause, parity went green and the kernel −10.9% — but the wall was −0.42%. The mechanism reading at the time: deleting the recomb ALU merely **changed the latency's shape**, and the synchronous LDG→STS memory latency stood exposed at the top of the staging phase as before.
- **r45** (cp.async A-side staging): handing the A staging's latency to the async units, kernel −10.2% and long_scoreboard −18% — but the wall −0.34%. The mechanism reading: after r41 the q6_K GEMM was no longer the wall's headliner, and **a kernel that isn't the wall can never reach the wall no matter how fast it gets**.

Each fell below the +1.5% bar, so under the single-shot rule both should revert. But they share one property that makes "revert" different from "rejection": **they act on the same traffic path (staging) with non-overlapping mechanisms** — r44 deletes WORK (the 16-shift/or/subtract-per-element recombination arithmetic), r45 deletes WAIT (memory-latency exposure). Alone, each one's gain was absorbed by the remaining cost dominated by the other: delete only WORK and WAIT takes over; delete only WAIT (and on the A side at that) and the kernel isn't the wall. This class of "overlapping in traffic, orthogonal in mechanism" combination is the **basket** the whole campaign had been collecting — r18's EB/SB pre-expansion machinery, the W_exp plane fixed up by r43/r44, r45's pipeline scheme — all inventory on the shelf.

r51/r52 moved the fulcrum: fused producers slimmed from 151.9 ms to 86.5 ms, swiglu fell from 10% of the wall to ~6.4%, and **the wall's headliner turned back to the q6_K GEMM**. The wall came back around, so the inventory shipped — the basket thesis's first full redemption: put r44's mechanism on the B side (r45 had only done the A side back then), then use r45's group-count pipeline + visibility barrier to hide the copy latency, so the same staging code sheds both WORK and WAIT.

## 2. Principle — the GPU mechanism

### 2.1 q6_K layout background (what this doc needs to stand alone)

`block_q6_K` is `ql[128] + qh[64] + sc[16] + d[2]` = **16 16-element sub-blocks** (not 8×32) — one 32-k chunk spans two sub-blocks with different scales, hence the kernel's KSPLIT=2: two `mma.m16n8k16` passes, each with its own `dsc`, with the plain rescale `sum += da·dsc` (no dmin term, verified against the CPU reference in r38). The B side's "centered int8" expansion: each (ql nibble, qh 2-bit field) pair combines into an 8-bit value and subtracts 32, giving a centered int8 in −32..31 — the r41 kernel did this in the hot loop every tile (uint4-widened ql/qh reads + recombination ALU).

### 2.2 The three costs of r41's B staging and what the bundle covers

| Form | ① ql/qh reads (bytes) | ② recomb ALU | ③ staging latency |
|---|---|---|---|
| r41 (EXP=false, kept in the current tree) | uint4-widened | in the hot loop | synchronous, exposed at the top of every tile |
| r53 (EXP=true) | **none** (reads the dense plane) | **none** (computed once at registration) | **cp.async**, hidden inside compute |

r41's uint4 widening had already cut ① by ~16×; r53 kills ①② with W_exp and ③ with cp.async. ffn_down −20.5% ≈ r44's −10.9% (deleting ②) + r45's −10.2% (deleting ③), near-additive — the two act on the same section of the same kernel and don't crowd each other out. Single-shot, each bought only half the kernel improvement, and r45 landed in the window when the wall had turned away — **wall-neutrality is a joint verdict of "timing + magnitude," not a mechanism's death sentence**. A −20.5% kernel improvement × q6_K's ~18–20% share of the wall ≈ +4–5% wall clock, clearing the bar exactly.

### 2.3 The W_exp plane and the memory account

At registration `expand_q6k_dense` expands the padded q6_K into **dense centered-int8**: `id` bytes per row (1 B/element), super-block stride 256; the scales (dsc) are still read separately by the kernel per the r38 scheme. The q6_K tensor inventory of 7B Q4_K_M:

| Tensor class | count | per-tensor `od×id` | subtotal |
|---|---|---|---|
| attn_v | 14 | 1.8 MB | 25.2 MB |
| ffn_down | 14 | 67.9 MB | 950.6 MB |
| output.weight | 1 | 544.6 MB | 544.6 MB |
| **total** | | | **1,521,237,632 B ≈ 1.52 GB** |

(The task brief's "+15 MB" estimate took one ffn_down's increment as the whole cost; this order-of-magnitude error incidentally explains where the q6_K launch count comes from — the number of W_exp planes = the number of q6_K GEMMs ≈ 27.) That is the bundle's purchase price: 1.52 GB for +5.03%.

### 2.4 How the latency is hidden: r45's pipeline scheme ported to the B side

r39's double buffering provided the structure for "stage the next tile while computing this tile"; r45 added the async-side discipline: all staging goes through cp.async and **commits once per tile**, and the main loop uses `wait_group 1` (waiting only for the previous group to land, letting this group's copies stay in flight) + one `__syncthreads` (cross-thread visibility). r53 puts the B copies into the same commit group, with A/B sharing one wait discipline — the staging latency leaves the critical path entirely.

**Explicit-PTX cp.async.** The copies use `gemm_cp16` (inline `cp.async.cg.shared.global [dst], [src], 16, sz`), where the src-size qualifier does the out-of-range zero fill: with `full=false`, `sz=0` writes that 16 B chunk as all zeros — rows beyond od (tiles past the weight's rows) are deterministically zero. With EXP=false the template branch disappears at compile time and the SASS is byte-identical to r41's — the fallback is not a runtime if, it is **a different compiled artifact**.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**Template `<KDR, bool EXP>` instead of a runtime branch**. The fast path and the r41 fallback share one kernel source, and `if (EXP)` resolves at compile time: EXP=false's SASS must be exactly r41's (the fallback's byte identity is guaranteed by construction, not retroactively by tests); EXP=true's branch costs zero. This also reduces "map miss → fall back" to a single pointer lookup — no second codebase to maintain.

**The dense index respects r44's root cause**. `W_exp + j*id + sb*256 + cbase*32 + cc*16` — the row stride is **id** (dense), not the padded 224; r43's parity FAIL came precisely from using the padded stride there. The gate's `id % 256 == 0` simultaneously guarantees the NB kernel's launch geometry and the dense index's 16 B alignment (cp.async 16 B chunks don't cross misaligned boundaries).

**The map key = the padded weight's device pointer** (the one `prefill_mmq` holds), value = the W_exp plane pointer. Plane names carry a geometry encoding (`{name}__exp{od}x{id}`), so a same-name different-shape re-registration can never silently reuse the old plane. Registration failure (alloc/upload) → the map stays empty → the kernel takes the EXP=false fallback + a loud eprintln **once per process**.

### 3.2 Key code

The host expander (current tree `src/cuda.rs` lines 1845–1872; the two output elements share the low/high nibbles of one (ql,qh) byte pair):

```rust
pub fn expand_q6k_dense(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
    const Q6KB: usize = 210;      // raw super-block: ql[128]+qh[64]+sc[16]+d[2]
    const Q6KPB: usize = 224;     // padded stride
    let nbe = id / 256;
    let row_len = nbe * Q6KPB;
    let mut out = vec![0u8; od * id];          // dense: row stride = id
    for j in 0..od {
        let prow = &padded[j * row_len..(j + 1) * row_len];
        let orow = &mut out[j * id..(j + 1) * id];
        for sb in 0..nbe {
            let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
            let (ql, qh) = blk.split_at(128);
            let obase = &mut orow[sb * 256..sb * 256 + 256];
            for it in 0..2usize {               // 256 elements = 2 × 128
                for r in 0..64usize {
                    let qlb = ql[it * 64 + r];
                    let qhb = qh[it * 32 + (r & 31)];
                    let s0 = (r >> 5) * 2;      // the 2-bit field's phase
                    let e0 = it * 128 + r;
                    obase[e0] = ((qlb & 0xF) | (((qhb >> s0) & 3) << 4)).wrapping_sub(32);
                    obase[e0 + 64] =
                        (((qlb >> 4) & 0xF) | (((qhb >> (s0 + 4)) & 3) << 4)).wrapping_sub(32);
                }
            }
        }
    }
    out
}
```

Registration and the map (r53's original form, verified identical to the current tree; `src/cuda.rs` lines 1614–1644):

```rust
pub fn register_weight_q6k_exp(&self, name: &str, padded: &[u8], od: usize, id: usize) {
    // geometry-encoded sibling name: a same-name different-shape
    // re-registration can never collide with (and silently reuse) a stale
    // plane of the same byte size but a different od/id layout.
    let exp_name = format!("{name}__exp{od}x{id}");
    let exp = Self::expand_q6k_dense(padded, od, id);
    self.register_weight(&exp_name, &exp);
    // the MAP is keyed by the PADDED weight's device pointer (what
    // prefill_mmq holds); the value is the W_exp plane's pointer
    if let Some(wp) = self.get_weight_ptr(name) {
        if let Some(ep) = self.get_weight_ptr(&exp_name) {
            if !wp.is_null() && !ep.is_null() {
                self.q6k_exp.lock().unwrap().insert(wp as usize, CudaPtr(ep));
                return;
            }
        }
    }
    ... // once per process: falls back to the r41 in-kernel expand
}
```

The kernel-side EXP branch (current tree `src/cuda_kernels.cu` lines 6764–6789) — staging is a pure copy, and the dense index respects r44's root cause to the line:

```cuda
if (EXP) {
    /* r53 bundle: the ql+qh recomb + -32 centering ran ONCE at
     * registration (expand_q6k_dense -> dense centered-int8 plane
     * W_exp: od x id, row stride = id, super-block stride = 256),
     * so the staging is a pure cp.async bulk copy (explicit PTX)
     * from W_exp — no recomb ALU, no register round-trip, no
     * ql/qh reads ... Rows beyond od zero-fill via
     * the cp.async src-size qualifier (gemm_cp16 full=0). */
    const int nc = (KDR * 32) / 16;   /* 16B chunks per row */
    const int ncopy = MMQ_NBJ * nc;
    for (int g = threadIdx.x; g < ncopy; g += blockDim.x) {
        const int jj = g / nc, cc = g % nc;
        const int j = j0 + jj;
        const bool full = (j < od) && (sb < nsb);
        const uint8_t* src = W_exp + (size_t)j * id
            + (size_t)sb * 256
            + (size_t)(cbase * 32 + cc * 16);        // ← r44's root-cause line: stride id
        gemm_cp16((__half*)(void*)(qbexpb + (size_t)jj * (KDR * 32) + cc * 16),
                  (const __half*)(const void*)src, full);
    }
```

The copy primitives themselves (current tree lines 4483–4491) — three `__forceinline__` wrappers are the whole PTX surface:

```cuda
__device__ __forceinline__ void gemm_cp16(__half* smem_dst, const __half* gsrc, bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 16 : 0; // src-size 0 => zero-fill the 16B chunk
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}
__device__ __forceinline__ void gemm_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void gemm_cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void gemm_cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }
```

r45's wait discipline (current tree lines 6905–6924) — the B copies join the same commit group, and `wait_group 1` lets this group's copies fly while waiting only for the previous one:

```cuda
    RAW_STAGE_Q6K_BT(0, 0);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
        // Overlap kt+1's global->smem staging with kt's compute: stage into the
        // OTHER buffer (buf^1) while reading buffer buf (the mmq_nt<7,2> pipeline).
        if (kt + 1 < nktile) {
            RAW_STAGE_Q6K_BT(kt + 1, buf ^ 1);
            // r53 (EXP) / r56: two groups are pending (kt's and kt+1's); wait
            // until only kt+1's remains — group(kt), the cp.async copies ...
            // has landed, while buf^1's copies stay in flight under kt's
            // compute (in-order group completion).
            gemm_cp_wait1();
        } else {
            gemm_cp_wait0();  // last tile: drain every outstanding group
        }
        __syncthreads();  // r53/r56: cross-thread visibility of the kt
                          // buffer's async copies before compute
```

(For contrast: the EXP=false fallback's elementwise expansion `expand_q6_elem` still sits at current tree lines 6687–6698 — `((ql[ql_idx] >> ql_shift) & 0xF) | (((qh[qh_idx] >> qh_shift) & 0x03) << 4)` then minus 32 — exactly the arithmetic W_exp computes once at registration.)

The byte-exactness test (current tree `src/graph/cuda_backend.rs` lines 3994–4062, `cuda_q6k_exp_dense_byte_exact`) — a three-way reconciliation of independent scalar mirror + host expansion + device read-back:

```rust
// independent scalar mirror, straight from the device formula
let mut want = vec![0u8; od * id];
for j in 0..od { for sb in 0..nbe { ... for e in 0..256usize {
    ...
    want[j * id + sb * 256 + e] = (v as i32 - 32) as u8;
} } }
// host-side production expander vs the mirror
let host = crate::cuda::CudaState::expand_q6k_dense(&padded, od, id);
let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
assert_eq!(hmis, 0, "expand_q6k_dense vs mirror ({od}x{id})");
// device upload path: build + read back + compare
state.register_weight_q6k_padded(&name, &raw, od, id);
state.register_weight_q6k_exp(&name, &padded, od, id);
...
state.copy_from_device_pinned(p, &mut got);
let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
assert_eq!(dmis, 0, "device W_exp vs mirror ({od}x{id})");
```

### 3.3 Pitfalls

1. **The liveness near-miss — this doc's namesake incident**. The first integrated build: parity green, greedy green, suite green — but flip on `MINFER_MMQ_RAW_NB_DEBUG` and the **fast-path launch count was zero**. The root cause was one line: the map insert keyed on the **exp buffer's own pointer**, while the kernel looks up by the padded weight pointer → always miss → always the EXP=false fallback. The fallback is the byte-identical r41 path, so **no correctness gate could see it**; the only symptom was "+5.03% didn't show up." The fix = key on the padded pointer, and the debug label immediately counted 27 W_exp-cp.async launches.
2. **The SASS trap of explicit PTX**. Once cp.async is written as inline PTX, a wrong constraint/type can make ptxas skip emitting LDGSTS and silently degrade. The SASS check confirmed `LDGSTS ×12` — the evidence chain must reach the SASS level; a correct PTX source does not imply a correct SASS.
3. **An order-of-magnitude memory estimate error**. The task brief estimated "+15 MB"; measured 1.52 GB — one ffn_down's increment (exp 67.9 MB − padded ~59 MB) was taken as the whole-model cost. Memory-for-speed decisions must be summed over the **tensor inventory**, not extrapolated from a single instance.
4. **The register budget re-check**. The new branch lands in a `__launch_bounds__(256, 3)` kernel: `<2,true>` compiled to 80 regs / 24 B stack — the 3-block budget held, and r53 did not buy its gain with r40's occupancy.

## 4. Verification

- **W_exp byte-exactness cargo test** (`cuda_q6k_exp_dense_byte_exact`, shapes (64,256)/(40,512)/(24,768) covering multiple super-blocks and multiple od): independent scalar mirror (transcribed straight from the device formula) vs the host expander, then vs the **device-plane read-back** (pinned copy), 0 mismatches. Guards against: a recurrence of r43/44's stride-mismatch class — this plane's only historical failure mode.
- **SASS check**: `LDGSTS ×12`. Guards against: explicit-PTX cp.async degradation.
- **liveness label** (`MINFER_MMQ_RAW_NB_DEBUG`): per-launch-path counts (the current tree's label is r54's refined three-state version: `exp=off` = deliberately off, `fallback!` = a plane was expected but missed — "reverted" and "off" must not look alike). Guards against: a fast path that silently never engages — the exclusive gate for this doc's near-miss.
- **parity ×3**: end-to-end numerics. **greedy byte identity** (a 178-character stream): the modes and the fallback share the accumulation order, so the gate should be green — and it is.
- **A/B interleaved measurement**: 3024.7 → 3176.9, distributions separated (base max < new min).
- **suite 167/0/3**: +1 new gate test (the W_exp byte-exactness test joined the regular suite).

## 5. Results

- **Kernel level**: ffn_down **16.06 → 12.76 ms (−20.5%)** — r44's −10.9% and r45's −10.2% stacking near-additively; attn_v **−15.9%**.
- **Wall clock** (7B @3325 tok): 3024.7 → **3176.9 (+5.03%)**; **vs-llama 1.09× → 1.05× — the q6_K line closes here** (at r37 it was still the worst kernel at 6.38×/GMAC; after r38–r41's +2.9/+13.3/+13.0/+30.7%, this doc completes the last stretch).
- **The price**: +1.52 GB of device memory (W_exp = the sum of od×id over 14 attn_v + 14 ffn_down + output.weight); `<2,true>` at 80 regs / 24 B stack, 3-block occupancy preserved.
- **Correctness**: W_exp byte-exact (0 mismatches); parity ×3; greedy identical; suite 167/0/3.
- **Left for the next round**: the 1.52 GB purchase price needs an exit ticket (r54's `MINFER_MMQ_Q6K_EXP=0` — measured −5.04% to buy back 1.52 GB, with the launch-path label distinguishing "deliberately off" from "accidental fallback"); the A side's dsc consumption and A staging wait become the new headliner (r56's W_dsc f32 plane + A cp.async, where r45's mechanism finally lands on the A side).

## 6. Lessons

1. **The basket theorem**: optimizations that overlap in traffic and are orthogonal in mechanism stack near-additively — mechanisms that are wall-neutral alone are inventory, not scrap; stockpile them and ship the bundle when the wall turns back.
2. **A fallback-correct optimization must have a "the fast path is actually alive" check**: parity and greedy are fully blind to "a fast path that silently never engages" — label the launch paths and count fast-path launches; that is this class's exclusive gate.
3. **Do the full account before trading memory for speed**: a plane's cost = the sum of `od×id` over all target tensors; single-instance extrapolation is off by two orders of magnitude.
4. **Template the two forms**: the fast path and the fallback compile from one source (`EXP=false` SASS = r41), byte identity is guaranteed by construction, and the fallback's cost reduces to one lookup.

← 55-r52-skip-write-mode2 · [Index](./README.md) · 57-r54-q6k-exp-optout →
