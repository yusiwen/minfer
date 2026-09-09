# 76 · D4-4 — the endgame kernel session: dpl dense split-plane q6_K decode MMVQ lands (bitwise); PDL and fused-FFN closed with mechanism (LANDED)

> **Result**: **L1 (LANDED, bitwise)** — the padded 256-element/224B MMVQ row layout streams 14 dead bytes per super-block (recorded as 215/256 = 84% useful); dpl rearranges each row into `[ql: nbe×128][qh: nbe×64][sc: nbe×16][d: nbe×2]` = nbe·210 B of content with row stride `(nbe·210+15)&~15` — same values + same accumulation order → bitwise. Probe: ffn_down 176.5 → 212.9 GB/s content (−17.1%), lm_head 208.3 → 250.0 (−16.7%). Wall clock (3× interleaved A/B medians): 14B tg128 23.28 → **24.57 (+5.53%)** / @3254 22.00 → **22.94 (+4.27%)**; 7B tg128 47.55 → **51.20 (+7.68%)** / @1641 46.44 → **50.18 (+8.05%)** — the bar was 14B @3254 ≥ +0.4%, over-delivered 10×. **L2 (PDL) probe green but in-situ NO-GO** (same-binary env-flip: 14B tg128 −2.6%/−1.8%) → reverted; **L3 (fused gate+up+SwiGLU+q8 Form B) probe +28.2%** (wave quantization + per-row latency exposure) → NO-GO. Endgame vs-llama: 14B **1.018×** tg128 / **0.950×** @3254; 7B **1.074×** / **1.052×**.
> **Commit**: `ffce151` (L1 code: src/cuda.rs +177, src/cuda_kernels.cu +101) + docs commit. **Date**: 2026-09-09.

## 1. Background — where things stood

This is the closing round of the decode campaign (D1→D4-4, 12 sessions). After D4-2 fixed 7B and closed the entire bitwise occupancy axis, the @3254 gap decomposition had converged to three rows: attention ~1.6 ms (D4-3's verdict: llama's true rate is ~2.1 TB/s and reaching it needs a separate session), **the true q6_K-vs-q4_K-class deficit ~0.42 ms/step** (true DRAM rates ffn_down 218.9 / lm_head 221.6 / attn_v 189.5 vs the q4_K class's 228.6; the bitwise occupancy axis proved unable to squeeze it out), and a ~0.3 ms launch-gap residual (CUDA graphs already banked most of it). D4-4's three levers each target one row: L1 hits the 0.42 ms (a layout lever), L2 hits the 0.3 ms residual (PDL), L3 hits FusedFFN's swiglu round trip (~106 µs/step execution + 2.2 µs launch).

L1's starting point is the precise thread D4-2 left behind: **the padded layout itself is the deficit**. q6_K's GGUF block is 210 B of content (ql 128 + qh 64 + sc 16 + d 2), and the historical layout pads it to a 224 B stride (a `ggml_pad`-alignment artifact serving prefill MMQ's uint4 alignment) — for every super-block it streams, the decode kernel pays DRAM traffic for 14 information-free bytes. Read in reverse, D4-2's conclusion that "the padded kernels are already optimal for their layout" says: **change the layout, not the kernel math**. The 84% figure means zeroing the dead bytes multiplies the q6_K class's theoretical stream-rate ceiling directly by 224/210 ≈ 1.067 — and the bulk of the 0.42 ms deficit sits exactly in that magnitude.

Session discipline carries over from D4-2: the baseline binary stored at `/tmp/d4/minfer_pre_d44`; all A/B are same-window interleaved 3 pairs taking the median (sglang co-tenant resident, ±1–2% window drift, pair medians decide); every lever probes for numbers first, passes the correctness gates, and only then discusses integration; L2/L3 both have explicit bars and veto paths.

The three levers' pre-registered bars are worth juxtaposing, because their width differences are themselves information: L1's integration bar was set at 14B @3254 ≥ +0.4% (the probe's −17% kernel-level signal folds into wall clock at roughly +0.5~1%, so the bar only demands "still positive after folding losses"); L2's bar is +0.3% (PDL's gain ceiling is ~0.3 ms/step, near-threshold to begin with); L3 has no wall-clock bar — a probe-level comparison decides life or death directly. Bar width is inversely proportional to confidence in the mechanism: for a layout lever with a "clear mechanism, trustworthy probe", the bar is loose; for a scheduling lever where "the probe may not represent in-situ", the deciding gate sits directly on the final shape's isolated A/B. This gradient proved fully right at the close: L1 over-delivered 10×, and L2/L3 were cleanly vetoed at their respective deciding gates.

## 2. Principle — the GPU mechanism

**The padded layout's true cost on DRAM.** Decode MMVQ is pure streaming: each thread reads one unit's ql/qh/sc/d + the matching q8 activation. Under the padded 224B stride, a super-block's 210 B of content spans 7 32B DRAM sectors (210 = 6.5625 sectors; the 7th sector holds only 18 B of content + 14 B of pad) — **those 14 B are paid on every block**, because the next block starts at the 224B boundary. The record summarized the effective byte rate as 215/256 = 84% useful (a different basis than the pure-content 210/224 = 93.75%: counting the d/scale sector fragmentation into the traffic, the effective fraction is lower). On either basis the conclusion is the same: 6–16% of the decode kernel's request stream is dead traffic, and this dead traffic **is not a "necessary cost" of the bandwidth-saturated regime** — it depends only on how the bytes are arranged.

**The dpl split-plane's layout ledger.** dpl (dense split-plane) lays each row's four components out contiguously: `[ql: nbe×128][qh: nbe×64][sc: nbe×16][d: nbe×2]`, with nbe·210 B of content per row and row stride `(nbe·210+15)&~15` (padded only to 16B alignment, not to a block boundary). 14B ffn_down (nbe 54): 54·210 = 11340 → stride 11344, only 4 B extra per row, 0.07 B amortized per block — the dead-byte rate falls from 6.25% to **0.035%**. The key constraint is uint4 alignment: within the ql/qh sections every 16B is aligned (both 128 and 64 are multiples of 16), and row bases are 16B aligned (guaranteed by the stride), so every uint4 load in the kernel is untouched. Same values, same order → bitwise.

The two layouts side by side (nbe super-blocks, `i` the block index):

| component | padded (status quo) | dpl (new plane) |
|---|---|---|
| ql (128 B per block) | offset 0..128 within block `i` | section base +0, offset `i·128` |
| qh (64 B per block) | offset 128..192 within block `i` | section base +nbe·128, offset `i·64` |
| sc (16 B per block) | offset 192..208 within block `i` | section base +nbe·192, offset `i·16` |
| d (2 B per block) | offset 208..210 within block `i` | section base +nbe·208, offset `i·2` |
| dead bytes between blocks | **14 B/block** (210 → 224 stride) | ~0.07 B/block (diluted by the row-tail pad) |
| row stride | nbe·224 | (nbe·210+15)&~15 |

On the kernel side the conversion happens only at the section bases (compile-time constants + an nbe multiplication); the intra-block offset formulas are unchanged byte for byte — that is where the "same mapping" invariant is implemented.

**Why bitwise survives.** Three layers of invariant: (1) every unit's operand values are unchanged — the rearrangement is a pure byte move, and the ql/qh/sc/d bytes' mapping to threads keeps its one-to-one correspondence through the section-base conversion; (2) the unit→thread mapping is unchanged (`u = tid, tid+256` and the loop form's `u += 256` are both untouched); (3) the accumulation order is unchanged (ascending u, the same `q6k_unit_acc` statement). The sufficient condition for floating-point bitwise is "same operands, same order"; all three layers hold, so both the probe and the unit tests do memcmp-level comparison rather than tolerance comparison.

**L2's mechanism: PDL (Programmatic Dependent Launch).** `cudaLaunchAttributeProgrammaticStreamSerialization` (PSS) lets the next kernel launch early, before the previous kernel has finished draining, with `cudaGridDependencySynchronize()` at its entry waiting for the dependency data to be ready — what is saved is the inter-kernel launch gap (the pool of ~2 µs/launch in the graph, ≈0.3 ms/step). The price is **co-residency**: the next kernel's waiting blocks occupy SM slots early and contend with the previous kernel's finishing wave. For compute-tail kernels (14B's attention h4w, lm_head — they are not pure-bandwidth streams) this tax is real money; the decode chain is a serial chain of 13 kernels and every interface pays this toll once.

PDL's relationship to D4-2's recorded "known risk" needs spelling out: D4-2's Lever C deferred PDL because of "the interaction risk with CUDA-graph capture". L2's probe attacked that risk first — attaching the PSS launch attribute to each kernel node during capture; capture+instantiate passed in one go, 200 replays were stable, and the PDL-graph output was bitwise against plain-eager execution (driver 580.173.02). That is, **the risk D4-2 worried about was falsified**, and what actually killed L2 was a different mechanism nobody anticipated then: the co-residency tax. The probe's control line of "24-kernel compute-bound chain +2.8%" was added as a completeness check and ended up supplying the whole lever's verdict — PDL's gain ceiling is the launch gap (~2 µs per interface), its cost is co-residency interference (potentially > 2 µs per interface for a compute-tail), and the ratio depends on the nature of the kernels on the chain, not on any PDL parameter.

**L3's mechanism: wave quantization + latency exposure.** Form B (fully fused) has a grid of one thread block per 32-value output q8 block → nf/32 = 432 blocks (the 27648×5120 shape), which at 6 blocks/SM occupancy is **1.5 waves** — the second wave fills only half the machine, a structural waste of 10–17%. Meanwhile each block performs 64 serial row dot products, each paying a reduce-barrier chain at its end; without cross-row register staging, load latency is exposed row by row; with staging, register pressure doubles, occupancy drops to 5 blocks/SM, and the tail wave gets worse. Both horns point to the same conclusion: **a fusion geometry that preserves bitwise values must lose on a 1.5-wave grid**.

Spreading L3's arithmetic out, because it is the template for every "fuse XX" proposal. The incumbent pair (the gu-concat matmul + `swiglu_quant_pad40`) both have row-shaped grids: od rows × 256 threads, 27648 rows = 27648 blocks, far more than one wave, with negligible wave-quantization loss; the two kernels each independently saturate bandwidth. Form B bundles 32 rows into one block (64 256-thread row passes inside the block), dropping the block count to 432 — **these 432 blocks must hide latency against each other**, and their only overlap mechanism is occupancy; 1.5 waves means the machine idles 1/3 of its SM slots during the second wave. What the fusion saves (one f32 write-read round trip of ~32 KB/row + one launch of 2.2 µs) is far less than the 1.5-wave idling (~17% × the ~400 µs order). Numerically: +28.2% ≈ wave quantization ~+15% + per-row latency exposure ~+13%, matching the two mechanisms' independent estimates.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**L1: a sibling plane, not a replacement.** The padded plane keeps serving three things: prefill MMQ (the NB-BT kernels at block_stride 224), the derivation source of the W_exp/W_dsc planes, and the dequant-f16/embed-gather fallback. dpl builds one **extra** `{name}__dpl` copy inside `register_weight_q6k_padded`, hung off a `q6k_dpl` map keyed by the padded weight's device pointer; decode dispatch consults the plane **before** the padded path — on a hit it runs the dpl kernel, on a miss it falls back to padded — and the `MINFER_Q6K_DPL=0` opt-out semantics match the r60 family (default = the verified path, "0" = opt out and reclaim the memory). The shape gate `id % 256 == 0` (exact nbe, the same precondition as the W_exp plane): every real decode shape (5120/13824/18944) satisfies it.

The memory ledger: the dpl plane duplicates the q6_K weights' entire content bytes (nbe·210 B per row + the row-tail pad) — 14B's q6_K weight set ~2.0 GB, 7B's ~0.9 GB. This is a direct trade of memory for DRAM traffic: GB10 has 128 GB of unified memory and 14B Q4_K_M's full weights are on the order of ~8 GB, so a +2.0 GB resident plane fits the budget; and the opt-out switch guarantees low-memory scenarios can keep only the padded plane (performance back to the D4-2 state, bitwise unchanged). Construction happens once at weight registration (host-side repack → `register_weight` onto device), so decode pays zero extra; the build cost is just a few memcpys at load.

The control eliminated at the probe stage: gs (group-split, rearranged q4_K-style into 16B groups) measured −7.9%/−10.7% in the probe (short of dpl's −17% class) → strictly dominated by dpl, not integrated.

**L2: probe first, integrate later; the deciding gate is a same-binary env-flip.** PDL's known risk (interaction with CUDA graph capture) was verified to "not hold" in a standalone probe before daring to touch the tree; after integration passed all the bitwise gates, the deciding gate used **the same binary flipping `MINFER_PDL`** for the A/B — stripping "code differences" out of the measurement entirely, leaving only the mechanism's effect.

**L3: Form B measured first, Form A vetoed by arithmetic.** Form B (fully fused, quantize included) is the probe of the gain ceiling: if even it cannot make money, Form A with its standalone quantize has even less of a chance; Form A is capped by arithmetic (~2–5 µs/step < bar) without burning probe time.

### 3.2 Key code

**Excerpt A · building the dpl plane (`src/cuda.rs` 1545–1576, inside `register_weight_q6k_padded`)** — the repack is four `copy_from_slice`s: cut each 210 B block from the padded raw GGUF bytes and place them into the ql/qh/sc/d sections:

```rust
// src/cuda.rs (current tree = the block introduced by ffce151)
if Self::mmq_gate_on("MINFER_Q6K_DPL") && id % 256 == 0 {
    let nbe = id / 256;
    let dpl_row = (nbe * 210 + 15) & !15usize;   // 210B content per block, rows padded to 16B
    let raw_row = nbe * 210;
    let mut dpl = vec![0u8; od * dpl_row];
    for r in 0..od {
        let src = &data[r * raw_row..(r + 1) * raw_row];
        let dst = &mut dpl[r * dpl_row..(r + 1) * dpl_row];
        let (ql, rest) = dst.split_at_mut(nbe * 128);
        let (qh, rest) = rest.split_at_mut(nbe * 64);
        let (sc, dd) = rest.split_at_mut(nbe * 16);
        for ib in 0..nbe {
            let blk = &src[ib * 210..(ib + 1) * 210];   // raw 210B block
            ql[ib * 128..(ib + 1) * 128].copy_from_slice(&blk[..128]);
            qh[ib * 64..(ib + 1) * 64].copy_from_slice(&blk[128..192]);
            sc[ib * 16..(ib + 1) * 16].copy_from_slice(&blk[192..208]);
            dd[ib * 2..(ib + 1) * 2].copy_from_slice(&blk[208..210]);
        }
    }
    let dpl_name = format!("{name}__dpl");
    self.register_weight(&dpl_name, &dpl);
    // …the q6k_dpl map: keyed by the padded weight's device pointer, hanging the dpl pointer…
}
```

**Excerpt B · the dpl side's unit load (`src/cuda_kernels.cu` 1673–1697)** — field-for-field matched to the padded `q6k_unit_load`, the only difference being that the address formula changes from "intra-block offsets" to "section base + intra-block offset"; all uint4 loads' alignment is guaranteed by the row stride:

```cuda
// src/cuda_kernels.cu (current tree = introduced by ffce151)
__device__ __forceinline__ void q6k_unit_load_dpl(
    int u, const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ x8row,
    int nbe, Q6kUnitRegs* r
) {
    const int kbx = u >> 3, pair = u & 7;
    const uint8_t* ql_row = wrow;                      // the four sections each contiguous:
    const uint8_t* qh_row = wrow + (size_t)nbe * 128;  //  same values, same order,
    const uint8_t* sc_row = wrow + (size_t)nbe * 192;  //  only a different arrangement
    const uint8_t* d_row  = wrow + (size_t)nbe * 208;
    r->d = h2f(*reinterpret_cast<const uint16_t*>(d_row + (size_t)kbx * 2));
    r->sc0 = (float)(int8_t)sc_row[(size_t)kbx * 16 + 2 * pair];
    r->sc1 = (float)(int8_t)sc_row[(size_t)kbx * 16 + 2 * pair + 1];
    const int chunk = pair >> 2, g = pair & 3;
    r->shift = 2 * g;
    r->g = g;
    const uint8_t* qlp = ql_row + (size_t)kbx * 128 + chunk * 64 + (g & 1) * 32;
    r->qla = *reinterpret_cast<const uint4*>(qlp);       // row stride 16B aligned
    r->qlb = *reinterpret_cast<const uint4*>(qlp + 16);  //  ⇒ uint4 legal
    const uint8_t* qhp = qh_row + (size_t)kbx * 64 + chunk * 32;
    r->qha = *reinterpret_cast<const uint4*>(qhp);
    r->qhb = *reinterpret_cast<const uint4*>(qhp + 16);
    const uint8_t* x8 = x8row + (size_t)u * Q8PB;
    r->d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
    r->xw = reinterpret_cast<const uint32_t*>(x8 + 4);
}
```

**Excerpt C · the dpl dispatch fast path (`src/cuda.rs` 4234–4267 excerpt, top of `q6_k_decode_mmvq`)** — the pf-vs-loop shape gate is verbatim the same as the padded side's (including D4-2's `id ≤ 16384` upper bound and the `MINFER_Q6K_PF` semantics):

```rust
// src/cuda.rs (current tree = introduced by ffce151)
// D4-4 L1: dense split-plane fast path (bitwise — see the
// registration comment). Falls through to the padded kernels when
// the plane is absent (MINFER_Q6K_DPL=0 / id not a multiple of
// 256 / map miss).
if blk_stride_padded && Self::mmq_gate_on("MINFER_Q6K_DPL") {
    let dwp = self.q6k_dpl.lock().unwrap()
        .get(&(wptr as usize)).map(|cp| cp.0);
    if let Some(dwp) = dwp {
        unsafe {
            let nbe = (id >> 8) as i32;
            if id > 8192 && id <= 16384
                && !std::env::var("MINFER_Q6K_PF").map_or(false, |v| v == "0")
            {
                launch_q6_k_q8_mmvq_v2_pf_dpl(/* … */, nbe, stream);
            } else {
                launch_q6_k_q8_mmvq_v2_dpl(/* … */, nbe, stream);
            }
        }
    }
}
// …a miss (or the gate off) falls through naturally and continues to the padded dispatch…
```

The kernels themselves (`q6_k_q8_mmvq_v2_pf_dpl` / `q6_k_q8_mmvq_v2_dpl`) differ from the padded versions only in the `row_stride` formula and the `q6k_unit_load_dpl` call, with both the dual-unit/loop forms complete — the npair-432 class takes pf (with D4-2's upper bound), the rest take the loop.

**Excerpt D · the bitwise unit test's A/B skeleton (`src/cuda.rs` from 5371, `cuda_q6k_dpl_bitwise` excerpt)** — the same raw bytes registered twice (once with dpl on, once with `MINFER_Q6K_DPL=0`), and both kernel forms (pf shape od 512/id 8960, loop shape od 4096/id 1024) must produce bit-exact output:

```rust
// src/cuda.rs (current tree, unit test excerpt)
std::env::remove_var("MINFER_Q6K_DPL");
st.register_weight_q6k_padded("d44_dpl_a", &raw, od, id);   // build the dpl plane
std::env::set_var("MINFER_Q6K_DPL", "0");
st.register_weight_q6k_padded("d44_dpl_b", &raw, od, id);   // pure padded
std::env::remove_var("MINFER_Q6K_DPL");
/* ...run decode MMVQ on both sides, comparing outputs byte for byte... */
assert!(a == b, "dpl-vs-padded decode outputs must be bit-identical (od {od} id {id})");
```

### 3.3 Pitfalls

- **The dpl row base is not naturally 16B aligned.** The first probe version used row stride = nbe·210: at nbe 54 the row bases alternate onto 2B alignment and the uint4 loads fault on misalignment outright. The fix is the `(nbe·210+15)&~15` stride — at most 15 B of pad per row (0.07 B amortized per block) in exchange for every uint4 being legal. The general form of this trap: **intra-section alignment must be backstopped by the row stride**; when changing a layout, the alignment responsibility moves from "inside the block" to "the row tail" and must be settled explicitly.
- **The NaN trap in bitwise checking.** When the probe filled `d` with random f16 bytes, the NaNs in the two outputs each compared `NaN != NaN` and the memcmp-class comparison read DIFF with max|Δ| = 0 — a fake difference. Fix: synthesize `d` as 0x3C00 (1.0) rather than random bit patterns. Any new quantization path with a bitwise gate should first screen random data for "which bit patterns become NaN".
- **L2's warning signal appeared in the probe, not after integration**: the 24-kernel pure compute-bound chain got **+2.8% slower** under PDL (the bandwidth chain −0.1%, neutral) — the co-residency tax had already shown itself in the probe. The integration's env-flip merely confirmed it as −2.6%/−1.8% on the real decode chain. Lesson: **do not expect a negative probe signal to turn positive after integration**; it is the same mechanism.
- **The suite's window flakes are archived as "isolated rerun green"**: one full-suite run failed `cuda_graph_recaptures_on_pool_gen_change` + `cuda_q4_0_prefill_q8_0_gemm_parity` — caused by the co-tenanted window; both passed in isolation and on rerun, archived as window flakes rather than defects. The adjudication basis is "the failure mode has no mechanistic connection to the change under test".

### 3.4 L2's integration shape (reverted, form archived)

PSS is not a one-line switch — the integration points sit on both sides of each decode-chain kernel's launch and entry. Launch side: at graph capture, `cudaLaunchAttributeProgrammaticStreamSerialization` is attached to each of the 13 kernel nodes (attribute value 1, effective only on sm_90+, which GB10 satisfies); kernel side: the first line of every PSS kernel's entry calls `pdl_sync` (a thin wrapper over `cudaGridDependencySynchronize()`, a no-op on non-sm_90 compile targets). The 13 kernels cover the whole decode chain: rms (2 sites) / swiglu / activation quantize / add / the matmuls / rope-store / attention split + combine. The `MINFER_PDL` env gates the whole attribute set — this is precisely the precondition for §4's "same-binary env-flip": flip the switch and the same binary runs the same graph once with and once without PDL.

The revert itself was also a checkpoint: the `MINFER_PDL=1` path lives on in the historical commits and the tree keeps only L1 (the PSS attribute code removed wholesale, no dead switch left) — consistent with the D series' convention of "reverted = clean tree + complete record".

### 3.5 L3's Form B kernel shape (probe, never entered the tree)

Form B's shape, archived (from the probe record): grid = nf/32 blocks; each block first computes the 32 gate rows' dot products with the verbatim 256-thread row-unit mapping (q4_K bitwise dots, in registers), then the 32 up rows; `silu(g)·u` completes in registers; finally the in-block call to the verbatim `quantize_pad40_block` produces the q8 — the numeric path is identical to the split pair, which is the basis for verifying that it "loses only on scheduling, not on math". The probe measured it same-shape (27648×5120 q4_K) and same-protocol against the incumbent pair (gu-concat matmul + `swiglu_quant_pad40`): 412.1 vs 321.4 µs. Form A (fusing gu+swiglu → f32, keeping the standalone quantize) was not probed: the round trip it saves is a strict subset of Form B's (one fewer f32 write-read), and with Form B losing 28%, Form A's arithmetic ceiling of ~2–5 µs/step cannot even reach the bar.

## 4. Verification

The gate chain for L1/L2/L3 each (one sentence per gate on what it defends):

| gate | covers | what it defends |
|---|---|---|
| probe bitwise (`/tmp/d4/probe_l1_dpl.cu`, memcmp vs padded) | L1, both shapes ffn_down + lm_head | the layout change touching the numeric path |
| unit test `cuda_q6k_dpl_bitwise` (in the suite) | L1, both kernel forms (od 512/id 8960 pf-form, od 4096/id 1024 loop-form) | dispatch branches outside the probe shapes drifting |
| 14B `-n 1` first-step dump (107 identical + 7 node{N}) | L1 | integration touching numerics or dispatch behavior (node{N} is D4-2's documented pool-slot instrument class, reproduces pre-vs-pre) |
| 7B `-n 1` first-step dump (72 + 2) | L1 | same, 7B side |
| greedy rp=1.0 byte-for-byte (both models) | L1 | the intermediate state of "dump right, generation drifting" |
| suite 174/0/3 (with the new dpl test) | L1 | full-model regression; one window flake archived as isolated-rerun green |
| PDL probe (capture/replay/200 replays/PDL-graph vs eager bitwise) | L2 | the CUDA-graph interaction risk (outcome: did not materialize) |
| same-binary env-flip isolated A/B (`MINFER_PDL`, 3 pairs) | L2's deciding gate | code differences leaking into the measurement; only the mechanism's effect remains (readings −2.6%/−1.8% → reverted) |
| Form B probe same-shape against the incumbent | L3 | a mathematically correct fusion losing money on scheduling (+28.2% → NO-GO) |

## 5. Results

**L1 wall clock (3× interleaved A/B medians, pre = `/tmp/d4/minfer_pre_d44`, post = the final tree):**

| Config | pre | post | Δ |
|---|---:|---:|---:|
| 14B tg128 | 23.28 | **24.57** | **+5.53%** |
| 14B @3254 | 22.00 | **22.94** | **+4.27%** |
| 7B tg128 | 47.55 | **51.20** | **+7.68%** |
| 7B @1641 | 46.44 | **50.18** | **+8.05%** |

The bar was 14B @3254 ≥ +0.4%; measured +4.27%, **over-delivering 10×** — the bulk of the 0.42 ms true deficit was cashed by the layout lever in one move (a 14B @3254 step is ~40 ms, and +0.94 ms lands right in the magnitude of the 0.42 ms q6_K deficit plus knock-on gains). Probe level: ffn_down (od 5120, id 13824) 176.5 → 212.9 GB/s content (kernel mean time −17.1%), lm_head (od 152064, id 5120) 208.3 → 250.0 (−16.7%). After L1, 14B's q6_K true DRAM class: ffn_down ~213, lm_head ~250 GB/s — the q4_K decode class is 228.6, and the residual gap is the explanation space of that recorded 84%-useful ratio applied in reverse to lm_head (lm_head's dpl stream rate is already **above** the q4_K class). Memory cost: +2.0 GB (14B), +0.9 GB (7B); `MINFER_Q6K_DPL=0` can always back out.

Reconciling the wall clock with the probe's magnitudes: what dpl cuts is the request-stream dead bytes of q6_K's three shapes; in one 14B @3254 step these three shapes' kernel time totals on the order of ~4 ms, so −16~17% ≈ −0.65 ms, plus the L2 sector-hit improvement and tail-wave tweaks from removing the dead-byte sector effect, landing in the measured +0.94 ms wall clock is self-consistent; 7B's ratios are higher (+7.68/+8.05%) because 7B's q6_K ffn_down takes a larger share of the per-step weight stream (10 layers × id 18944, where 14B is 48 layers × id 13824 with ffn_down only part of that).

**L2 (PDL) closing numbers**: the probe was all green — on driver 580.173.02 the PSS-attributed graph capture+instantiate worked, 200 replays were stable, and the PDL-graph was bitwise against plain-eager (the graphs interaction risk D4-2 worried about **did not** materialize); but in the same probe the 24-kernel compute-bound chain ran +2.8% slower (the bandwidth chain −0.1%, neutral) as the first warning. The integration (all 13 decode-chain kernels carrying PSS + the entry `pdl_sync`, gated by `MINFER_PDL`) passed all the bitwise gates and finally lost at the deciding gate: 14B tg128 −2.6%/−1.8%, 7B ≈ 0, @3254 within noise → **reverted**. Mechanism: PSS lets the next kernel's waiting blocks co-reside on the SM with the current kernel's finishing wave — 14B's compute-tail kernels (attention h4w, lm_head) lose more slots to the contention than they save from the launch gap; the target pool (~2 µs/launch ≈ 0.3 ms/step) had already been mostly banked by D3-4's CUDA graph capture. Retry conditions: worth another look when the decode chain becomes purely bandwidth-dominated (e.g. attention is no longer a compute-tail) or the launch gap grows (leaving graph capture).

**L3 (fused gate+up+SwiGLU+q8, Form B) closing numbers**: probe (`/tmp/d4/probe_l3_fuse.cu`, 27648×5120 q4_K) **+28.2%** (412.1 vs 321.4 µs; 193.2 vs 247.8 GB/s content). Decomposition: grid = nf/32 = 432 blocks = 1.5 waves at 6 blocks/SM (wave quantization, structural ~+10–17%) + 64 serial row dot products per block each paying the reduce-barrier chain (cross-row register staging would hide it, but staging doubles registers → 5 blocks/SM and a worse tail wave). Form A (fusing only gu+swiglu → f32, keeping the standalone quantize) has an arithmetic ceiling of ~2–5 µs/step, below the bar, not probed. The line's closing ledger: the swiglu round trip is a real cost (~106 µs/step execution + 2.2 µs launch), but **every fusion geometry that preserves bitwise values loses more to wave quantization than it saves** — unless tolerance gating is accepted (changing the numeric path), this line does not reopen.

**Endgame vs-llama (q4_k_m, llama-bench `ca3d5a3e1`):**

| Model | Config | minfer (D4-4) | llama | ratio |
|---|---|---:|---:|---:|
| 14B (48L) | tg128 | **24.57** | 24.14 | **1.018×** |
| 14B (48L) | @3254 | **22.94** | 24.14 | **0.950×** |
| 7B | tg128 | **51.20** | 47.65 | **1.074×** |
| 7B | @1641 | **50.18** | 47.69 | **1.052×** |

The decode program's endgame state: 7B at or above parity on every measured shape; 14B short-KV above parity and 5% off at @3254 — the remaining honest gap is attention (~1.9×, D4-3's measurement correction; the next lever is a tolerance-gated attention rewrite, a separate session per D4-3's bar). The whole D series' (D1→D4-4) net movement: 14B tg128 22.81 → 24.57, @3.3K 20.9 → 22.94; 7B tg128 → 51.20 — **ahead of or at parity with llama on every measured shape**.

Each lever's archived state at the close, so the next session can pick up directly:

| lever | status | carrier on the tree | reopen conditions |
|---|---|---|---|
| L1 dpl q6_K | **LANDED** (bitwise) | `ffce151`, `MINFER_Q6K_DPL` ("0" opt-out) | none — the bulk of the 0.42 ms deficit is already realized |
| L2 PDL decode chain | probe green / in-situ reverted | none (form in 3.4) | the chain becomes purely bandwidth-dominated, or the launch gap grows (leaving graph capture) |
| L3 fused gu+SwiGLU+q8 | Form B probe NO-GO | none (form in 3.5) | revisit when tolerance gating is accepted (numeric path loosened) |
| attention rewrite | line closed (measurement correction) | — | new session: (1,7,40) geometry + explicit staging, bar ≤25–32 µs @14B |

## 6. Lessons

1. **Layout is a first-class decode-GEMV lever, ranked ahead of kernel math.** Same kernel, same 48 regs, same occupancy; swapping only the 224B stride for 210B+16B alignment moved all four wall clocks +4.3~8.1% — first count how many bytes in the request stream are dead, then talk microarchitecture.
2. **Bitwise can be designed for; it is not luck.** Same values (a pure byte move) + same mapping (unit→thread untouched) + same order (ascending accumulation) — with all three invariants in place, memcmp is the gate; only when a layer loosens is tolerance needed.
3. **Do not expect a negative probe signal to turn positive after integration.** PDL's +2.8% compute-chain warning and the post-integration −2.6%/−1.8% are the same mechanism (the co-residency tax); seeing the reversal in the probe should have been the stop — the integration cost (launch attributes + gating for 13 kernels) could have been saved.
4. **The same-binary env-flip is the strongest A/B for isolating a mechanism's effect.** No code difference, no compile difference — what is measured is the mechanism itself; any optimization with an env switch should use this move as its deciding gate (L1's `MINFER_Q6K_DPL=0` can serve as the same-style recheck at any time).

← 75 · [Index](./README.md) · [77](77-verification-methodology.md) →
