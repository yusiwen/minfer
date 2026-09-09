# 47 · r44 — W_exp stride mismatch root cause: fix goes parity all-green but wall-neutral (REVERTED)

> **Result**: r43's pre-expand-B parity mystery was dissolved by a one-line indexing fix — `W_exp` (the dense plane, row stride = `id`, super-block stride = 256) was being addressed with the padded raw-W address expression (row stride = `nsb·bstride`, sb stride = `bstride` = 224). After the fix, parity `cuda_prefill_mmq` went 1/0 all green and ncu kernel elapsed cycles went 1,390,776 → 1,239,847 (**−10.9%**, the recomb ALU and its ~45% samples vanished entirely) — but whole-prefill went 2595.9 → 2584.9 tok/s (**−0.42%**, noise), below the +1.5% bar → **REVERTED**. Conclusion: after r41 the q6_K GEMM is no longer the prefill wall's bottleneck; a kernel-level win on the converged line cannot reach the wall. The same physical latency merely changed the instruction set it hides behind (load→ALU-recomb became load→STS copy, the same L1TEX exposure); the long_scoreboard share 34.9% → 57.1% is a denominator effect. Next lever: cp.async (r45).
> **Commit**: `6d02017`. **Date**: 2026-09-05.

> **Provenance note**: the commits cited here (r43 `b7fa305`, r44 `6d02017`) are both **docs-only** commits — the experimental code was reverted after measurement, before commit (cmp-match HEAD r41), so the diffs contain no code. The code excerpts in this doc all come from the **current tree**: `expand_q6k_dense` (the r53-landed host-side expander) and the `EXP=true`/`EXP=false` staging branches of `mmq_raw_nb_bt_q6k_kernel` — these are precisely the final landed form of r44's "one-line fix", with the address expressions word-identical (the current tree's `expand_q6k_dense` doc-comment even signs itself "P6 r44 / MMQ-analysis §11.24").

## 1. Background — where things stood

### 1.1 A line climbing fast

On r44's day the P6 q6_K line was in the steepest climb of the whole campaign. The trajectory of the previous four rounds: r38 BT-style raw-byte mma kernel (1518.4 → 1561.9, +2.87%, while also discovering that "q6_K = 8 sub-blocks × 32" is wrong — it is actually 16 sub-blocks × 16, with KSPLIT=2 each carrying an independent dsc); r39 KDR=2 double buffering (→ 1777.5, +13.3%, overlapping the B-expansion's staging latency with compute); r40 `__launch_bounds__(256,3)` third resident block (→ 2015.6, +13.0%, while falsifying the "0-spill gate"); r41 B-expand uint4 widening (→ 2605.2, **+30.7%**, kernel 1.70 → 0.654 ms, merging 32 per-byte `LDG.E.U8` into 2 uint4s). Whole-prefill rose from 1518 to ~2600 tok/s, and vs-llama closed from 2.13× to 1.27×.

But r41 also left a precise residual reading: L1TEX scoreboard 85.5% → 33.6%, with the remaining stall's composition then unknown. r42 tried "widen the dsc scale reads" — bytes fell, the stall did not move at all (−0.19% wall) — the first hint that "the intuition about the stall's source might be wrong". r43 used PC-sampling (`pcsamp_warps_issue_stalled_long_scoreboard` + `--page source`) to attribute the stall by consuming instruction and produced that decisive table: **B-expansion recomb (LOP3/SHF) 45% + A-side staging STS.128 28% + dsc's I2F.S8 consumption point 26%**. All three sources point at one physical process: B's global→smem round trip exposing L1TEX latency.

There is a reasoning gap that only became visible afterwards, worth writing down: what the attribution table gives is **sample shares**, not wall-clock seconds. The step from "45% of samples are on the recomb" to "deleting the recomb wins back 45% of the stall" implicitly assumes that slice is a serial critical path — an assumption that held when q6_K was the wall and does not hold after r41. r44's wall-neutrality is the first empirical failure of that implicit assumption; r47 would write "wall decompositions have a shelf life" as a formal lesson.

### 1.2 The seemingly direct road: move the recomb out of the hot loop

The attribution table lays out a superficially direct road: 45% of the stall samples hang on the recomb ALU, and the recomb is a **pure function** — each output element depends on a single byte pair from (ql, qh), unrelated to tokens or any runtime state. So why not move it out of the kernel? Pre-expand once at weight registration into the centered-int8 plane `W_exp`, and the kernel's B staging degenerates into a pure bulk copy: all ql/qh loads and nibble-unpack ALU deleted from the hot loop, and by the attribution logic the 45% of samples should largely disappear.

The idea is not a new invention — r18 tried "load-time B pre-expansion" (bulk-copy staging, KD=8 +0.9% noise, KD=4 −19%, plus a +5.8 GB VRAM cost) and was reverted. r43's version differed in two ways: the timing moved from "load" to "registration" (effective with the NB-BT gate, serving only the q6_K type), and r43 had the attribution table's endorsement — a clear 45% target. Then came that famous parity FAIL:

- The `W_exp` plane itself passed the device-side readback **0/17,920 bytes, all correct**;
- `cuda_prefill_mmq` parity **FAILED, diff 448 @ index 554**;
- The two failing pre-expansion variants (uint4 version, per-byte version) had **bit-identical diffs**, while the control arm expanding raw `W` in-kernel passed.

Not isolable within budget, r43 ended at cmp-match HEAD, leaving a verdict r44 would take over: "byte-correct data at the wrong address expression is worse than obviously-wrong data — the readback validates CONTENT, not OFFSETS (r44 resolves it)".

### 1.3 r44's dual task

So r44's task had two layers. The first is **diagnosis**: solve the parity mystery, or "pre-expand B" cannot even get one clean measurement. The second is **adjudication**: how much wall clock is the corrected pre-expansion actually worth — the direct cash-in of r43's attribution table, and the touchstone for how much oil is left in the whole q6_K line. The two layers' answers landed the same day, and the second layer's answer (wall-neutral) shaped the campaign far more than the first (a one-line fix): together with the next day's r45 it declared the q6_K line converged and pushed the campaign's entire budget toward FA and prepass (the direction of r46/r47).

A footnote on the time axis: r44's work ultimately survived in two forms — **as knowledge** (the root cause + the "kernel win ≠ wall win" verdict + the pointer that cp.async is the real lever), fed into r45's design that same day; and **as code** (the dense indexing + the host expander), landing verbatim in r53's bundle two days later. A REVERTED doc is not a wasted step — provided the "veto mechanism + retry conditions" are written completely enough that future rounds can cite them precisely.

## 2. Principle — the GPU mechanism

### 2.1 What W_exp is (the definition at first appearance)

**The W_exp plane**: the q6_K weight tensor pre-expanded, according to its mathematical shape, into a dense centered-int8 byte plane — `od × id` bytes, one weight element per byte, the value range already shifted by −32 to center (q6_K's nibble + 2-bit combined encoding lands in ±127 after subtracting 32). It eliminates two things: **encoding** (an element's information is split across 4 bits of ql and 2 bits of qh) and **padding** (the raw layout pads each super-block's 210 content bytes to 224). The concept debuted in r18 (load-time pre-expansion, reverted for its VRAM cost); the r43/r44 version builds on demand at registration and serves only the NB-BT q6_K path. It must output **centered int8** (−32 centered) because r38's KSPLIT=2 mma structure takes centered int8 as the B-operand contract directly (the single-term scaling of `sum += da·dsc` is built on that value range) — W_exp is not "another storage format"; it freezes the transform every kernel loop body was doing into data, with the feeding end's mma contract unchanged by a single character.

### 2.2 Precise definitions of the two layouts

q6_K's `block_q6_K` = `ql[128] + qh[64] + sc[16] + d[2]` = **210 bytes of content**, describing 256 elements (16 16-element sub-blocks, each sub-block a `d·sc` scale pair). The two planes fork from here:

- **raw `W` (padded)**: each super-block's 210 content bytes padded to **`bstride = 224`** (16-byte alignment, the premise of r41's uint4 widening). row stride = `nsb · bstride` (`nsb = id/256`), sb stride = `bstride` = 224. **This is the data's original form in VRAM and cannot change** (prefill MMQ's block_stride 224 depends on it; the dequant/embed fallbacks read it).
- **`W_exp` (dense)**: exactly **256 bytes** per super-block (one element per byte, no padding). row stride = `id` (= `nsb · 256`), sb stride = 256.

The density difference is 256/224 ≈ 1.14×: the dense plane pays 14% more bytes for the unconditional address identity "element i is at offset i". The two layouts side by side (one super-block, a two-row sketch with `nsb = 2`):

```
raw W (padded, bstride=224)          W_exp (dense, sb stride=256)
row 0: [sb0: 210B content|14B pad][sb1: 210B|14B pad]   row 0: [sb0's 256 elements][sb1's 256 elements]
        ^0          ^210    ^224          ^434  ^448            ^0                ^256
row 1: base = 2*224 = 448            row 1: base = 2*256 = 512
misread row1@448 = dense row 1's elements 192..255 spliced with row 2's elements 0..191
```

(In the diagram `^` marks byte offsets. On the raw side each sb is 210 bytes of content + 14 bytes of padding; on the dense side each sb is exactly 256 bytes, one element per byte. The misalignment accumulates from the second row on.)

### 2.3 The arithmetic of the mismatch: a byte-level worked example

The wrong expression `W_exp + j·(nsb·bstride) + sb·bstride` pages through the dense pointer at padded strides. With the smallest `id = 256` (`nsb = 1`), the row stride is wrongly 224:

- Row 0 has no drift (offset 0): the 256 bytes read are exactly dense row 0 — **but the in-row sb drift is 0 only because nsb=1**;
- Row 2: the correct base `2·256 = 512`, the wrong base `2·224 = 448` = dense row 1's byte 192. So the kernel's "row 2" = dense row 1's elements 192..255 (64 bytes) spliced with dense row 2's elements 0..191 (192 bytes) — **the bytes inside the window are real weight bytes; the window's assembly is wrong**;
- For `id = 3584` (7B attn_v, `nsb = 14`): wrong row stride 3136, drifting `14×32 = 448` B per row; for `id = 5120` (ffn_down, `nsb = 20`): 640 B per row. The larger the row number, the further the window read sits from the real data.

This explains the diff's shape: not wholesale garbage, but output errors that are "contiguous within each 256-element window, misaligned between windows" (448 @ index 554, a magnitude consistent with f32 accumulation differences). The drift amplifies with shape:

| Shape | `nsb` | Wrong row stride (nsb·224) | Correct row stride (id) | Drift per row |
|---|---|---|---|---|
| id = 256 | 1 | 224 B | 256 B | 32 B |
| id = 3584 (7B attn_v) | 14 | 3,136 B | 3,584 B | 448 B |
| id = 5120 (7B ffn_down) | 20 | 4,480 B | 5,120 B | 640 B |

The larger the row number, the further the window sits from the real data — no row except row 0 is correct. It also explains why the three variants sharing the same expression had **bit-identical diffs** — if the bug lived in some variant-private mechanism (staging order, a race, a barrier, register allocation), variants with different mechanisms could not produce the same diff. The fingerprint points at their only shared part: the address expression.

### 2.4 How much VRAM the plane costs — the lever's cost side

W_exp's size is **`od × id` bytes**, one plane per q6_K tensor on the NB-BT path. The 7B inventory: 14 attn_v + 14 ffn_down + output.weight, totaling `1,521,237,632 B ≈ 1.52 GB` (r53's precise measurement; r43's "+15 MB" estimate at project time erred by counting one ffn_down's increment as the whole cost — which also explains why that path has exactly 27 launches). 1.52 GB is not small: it is the slimmed-down edition of the same tax as r18's "+5.8 GB reverted", and the reason r54 later built the `MINFER_MMQ_Q6K_EXP=0` opt-out (−5.04% for 1.52 GB back). This doc records only the cost side: for any "pre-transformed plane" lever, the VRAM account belongs in the project proposal next to the latency account.

### 2.5 Why readback cannot catch it

r43's gate was a device-side readback comparison of the `W_exp` plane (0/17,920). Readback verifies **content**: the bytes the host wrote match the bytes the device reads. It does not verify **offsets**: which expression the kernel uses to read the plane is invisible to readback. A fully correct producer + a consumer paging by the wrong catalogue = "byte-correct data at wrong offsets". This combination is more dangerous than "obviously wrong data": it grants strong confidence (the plane is correct) and pushes the investigation toward the wrong direction of kernel-side races/aliasing. The lesson in full: **the end-to-end correctness gate for a pre-transformed plane must include consumer-side parity; readback only rules out "written wrong", not "read wrong"**. r53 later paired `expand_q6k_dense` with an independent scalar-mirror test (host mirror AND device plane readback, both 0 mismatch) — institutionalizing this lesson.

### 2.6 Why "fixed correctly yet unprofitable" — latency changes form, it does not vanish

The pre-expansion's paper gain is deleting the recomb ALU. But B's **physical latency** never disappeared — it is the "global load → smem landing" L1TEX round trip, independent of what instruction consumes it. r41's shape: load → recomb ALU consumes → STS write-back, with the ALU consumption point absorbing the stall-wait; after pre-expansion: load → pure STS copy, the same stall-wait hanging on a thinner instruction stream. The ncu readings are this mechanism's complete signature: elapsed cycles **−10.9%** (the ALU and its samples really did vanish) while **Warp-Cycles/Issued-Inst 11.70 → 17.84** (the wait amortized per instruction grew) and the long_scoreboard share 34.9% → **57.1%** (denominator effect: total cycles shrank, the stall's absolute volume stayed roughly constant, so the share naturally rose). The same physical latency hides under a different instruction mix — this is the companion piece to r42's "cutting bytes does not cut latency". What can actually hide this latency inside compute gaps is the asynchronous copy (cp.async), which is r45's subject; §11.24's closing judgment is blunt: "The lever that actually hides it is cp.async (llama.cpp structure)" — llama.cpp's MMQ reference implementation uses cp.async precisely to move staging latency off the critical path, part of its instruction-stream advantage.

## 3. Implementation

### 3.1 Diagnosis and design choices: a one-line address fix, not a staging rewrite

The diagnosis is a four-step evidence chain, each step eliminating a class of hypotheses:

1. **Readback all-correct (0/17,920)** → rules out "the host expander wrote it wrong": the plane's content is correct.
2. **The failing variants (uint4 pre-expansion, per-byte pre-expansion) have bit-identical diffs** → rules out all variant-private mechanisms (staging order, races, barriers, register allocation): two variants with different mechanisms cannot produce the same diff unless the error is in what they share.
3. **The control arm expanding raw `W` in-kernel passes** → the only shared code that is "right for raw W, wrong for W_exp" is the address expression: the data layout differs, the expression is the same.
4. **Conclusion**: the expression pages through the dense plane at padded strides. The test is ready-made — swap the expression to the dense strides and parity should turn green; if it does not, a second cause remains and we return to step 1.

Parity turned green on the first try; the single-cause verdict holds.

The fix's shape is therefore **pure address arithmetic**: swap the base-address expression pointing at `W_exp` in B staging from the padded form to the dense form, touching no staging mechanism.

Two reasons to choose the minimal diff. First, the diagnosis had already locked suspicion onto the expression; a larger change would pollute this single-cause verdict (if parity stayed red after the fix, a second cause would exist). Second, the fix must preserve every budget constraint r38–r41 had banked: `__launch_bounds__(256, 3)`'s third resident block requires ≤80 registers — after the address terms change from `j·(nsb·bstride) + sb·bstride` to `j·id + sb·256`, the multiplication structure is unchanged (`id` and 256 are both compile-time-observable), and ptxas landed at 80 regs / 28 B spill; the 3-block budget held.

The dense indexing also has a friendly property later cashed in by r53: this NB-BT path's launch gate requires `id % 256 == 0`, so every 16-element group start of `W_exp + j·id + sb·256 + cbase·32` is naturally 16B-aligned — when switching to cp.async 16B copies, the alignment premise is already in place.

One alternatives-check on "why build the dense plane at all": could we skip W_exp and let the kernel read raw `W` in the padded layout doing "nothing"? No — that is exactly r41's status quo; the recomb must stay in the hot loop and the 45% of samples have nowhere to go. Could the kernel read raw `W` but skip the recomb? No — in the raw layout the element-to-byte mapping is not an identity to begin with; the recomb is that mapping. So "pure copy" has as its sole precondition an element-ordered plane, and the dense `W_exp` is its minimal implementation; the stride mismatch was an address bug in landing that plane, not a design flaw.

### 3.2 Key code

**The wrong expression (it is still in the tree today — but it serves raw `W`, for which it is correct)**. The current tree's `src/cuda_kernels.cu` `EXP=false` branch (r41's in-kernel expansion, the raw 210-B padded layout):

```cuda
// Inside the RAW_STAGE_Q6K_BT macro, EXP=false && (bstride & 15) == 0 branch:
//   blk points at one super-block of raw W — row stride is nsb*bstride,
//   sb stride is bstride=224. This is correct for PADDED raw W.
const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)
                       + (size_t)sb * bstride;          // ← r43 used it verbatim on W_exp
const uint4 qlv = *(const uint4*)(blk + it0*64 + gg*16);
const uint4 qhv = *(const uint4*)(blk + 128 + it0*32 + (gg & 1) * 16);
```

r43's pre-expand-B variant swapped `W` for `W_exp`, deleted the recomb, and kept this stride line — the kernel then paged through the dense plane at 224/row, 224/sb. That the `expand_q6_elem` control arm passed is precisely because its data source `W` **is** the padded layout: expression and data matched.

**The "before" that W_exp replaces — the in-kernel recomb (current tree `expand_q6_elem`, introduced in r38 and still the EXP=false path's core after r41's widening)**:

```cuda
__device__ __forceinline__ int expand_q6_elem(const uint8_t* ql, const uint8_t* qh, int elem) {
    int m  = elem & 31;
    int it = elem >> 7;
    int n  = elem & 127;
    int ql_idx   = it * 64 + (n & 63);
    int ql_shift = (n >> 6) * 4;          // 0 or 4 (low/high nibble)
    int qh_idx   = it * 32 + m;
    int qh_shift = ((n >> 5) & 3) * 2;    // 0,2,4,6 (2-bit field)
    int v = ((ql[ql_idx] >> ql_shift) & 0x0F)
          | (((qh[qh_idx] >> qh_shift) & 0x03) << 4);
    return v - 32;
}
```

What pre-expansion does is move this bit arithmetic (plus its ql/qh loads) into `expand_q6k_dense`, run once at registration; the kernel side's B staging goes from "load→recomb→STS" to "pure copy". The three code blocks above read together are the complete before/after: `expand_q6_elem` (the work deleted) → the wrong/correct `src` expression (where the mismatch lives) → the host expander (the work's new home).

**The correct dense indexing (current tree `EXP=true` branch, the landed form of r44's fix)**:

```cuda
// EXP=true: W_exp is the dense centered-int8 plane produced by expand_q6k_dense
// (od x id, row stride = id, super-block stride = 256).
const int sb     = (kt * KDR) >> 3;            // which super-block this kt window lands in
const int cbase  = (kt * KDR) & 7;             // 32-element chunk offset within the sb
const int nc     = (KDR * 32) / 16;            // 16B chunks per row
for (int g = threadIdx.x; g < MMQ_NBJ * nc; g += blockDim.x) {
    const int jj = g / nc, cc = g % nc;
    const int j = j0 + jj;
    const bool full = (j < od) && (sb < nsb);
    const uint8_t* src = W_exp + (size_t)j * id     // ← row stride = id (dense)
        + (size_t)sb * 256                          // ← sb stride = 256 (dense)
        + (size_t)(cbase * 32 + cc * 16);           // ← offset within the sb (same in both layouts)
    // 16B alignment: id is a multiple of 256 ⇒ j*id and sb*256 both preserve it
}
```

Read in contrast: the two segments' only structural difference is `nsb·bstride → id` and `bstride → 256`. The entirety of r44's "root cause" is these two substitutions — after them, the in-sb offset `cbase·32 + cc·16` needs no change, because it describes the element-space part common to both layouts.

**The host-side expander (current tree `src/cuda.rs`, the r53-landed version; the expander logic in r44's experiment was isomorphic)** — its doc-comment also documents the dense output layout, exactly the contract the consumer must match:

```rust
/// host mirror of the device `expand_q6_elem` (P6 r44 / MMQ-analysis
/// §11.24). Output: `od * id` bytes, `out[j * id + sb * 256 + e]` =
/// super-block element e of row j — the exact tile the kernel's staging
/// used to recomb. Requires `id % 256 == 0` (the NB-BT launch gate).
/// Two output elements per (ql, qh) byte pair: e = it*128+r and
/// e = it*128+r+64 share ql[it*64+r] (nibble shifts 0/4) and
/// qh[it*32 + (r&31)] (2-bit-field shifts 2*(r>>5) / 2*((r>>5)+2)).
pub fn expand_q6k_dense(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
    const Q6KB: usize = 210;    // block_q6_K content bytes
    const Q6KPB: usize = 224;   // padded block stride (raw W side)
    let nbe = id / 256;
    let row_len = nbe * Q6KPB;                      // raw-side row stride = nsb*224
    let mut out = vec![0u8; od * id];               // dense side: od*id, row stride = id
    for j in 0..od {
        let prow = &padded[j * row_len..(j + 1) * row_len];      // 224-stride read
        let orow = &mut out[j * id..(j + 1) * id];               // id-stride write
        for sb in 0..nbe {
            let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
            let (ql, qh) = blk.split_at(128);
            let obase = &mut orow[sb * 256..sb * 256 + 256];     // 256-stride write
            for it in 0..2usize {
                for r in 0..64usize {
                    let qlb = ql[it * 64 + r];
                    let qhb = qh[it * 32 + (r & 31)];
                    let s0 = (r >> 5) * 2;
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

Note that the host expander internally **touches both layouts at once** (the 224 read of `row_len`, the 256 write of `orow`). Transform code where two stride sets coexist is exactly the soil in which mismatches get written: the consumer copies the wrong side, with no compile-time or runtime signal.

The assembly relationship of the three segments: `expand_q6k_dense` (host, once at registration) produces `W_exp` → the kernel's `EXP=true` branch does a pure copy with dense indexing (r44's fix site; from r53 swapped to `gemm_cp16` with src-size zero-fill of the tail rows beyond od) → `expand_q6_elem` keeps serving raw `W` only when `EXP=false` (W_exp absent/fallback). One dispatch chain with two data sources and two address contracts — which is also why the launch path needs an explicit label distinguishing them (r54's three-state label), making "which contract is in force" observable at runtime.

### 3.3 Pitfalls

1. **The readback gate is a content gate, not an offset gate.** The perfect 0/17,920 readback granted strong confidence that "the plane is correct", which pushed the investigation toward the wrong direction of kernel-side races/aliasing. A pre-transformed plane's correctness gate must include consumer-side parity.
2. **Bit-identical cross-variant diffs are the fingerprint of shared address arithmetic.** The failing variants producing the same diff (448 @ 554) = ruling out all variant-private mechanisms (staging order, barriers, register allocation), contracting the suspicion to the shared address expression; stacked with the control arm ("in-kernel expansion of raw W passes"), the only surviving explanation is expression-layout mismatch. This reasoning pattern is more transferable than the fix itself.
3. **The denominator effect on share-class metrics.** After cycles −10.9%, the long_scoreboard share went 34.9% → 57.1%, which looks "worse" at a glance. Reading ncu shares requires the absolute quantities alongside: Warp-Cycles/Issued-Inst 11.70 → 17.84 (per-instruction stall-wait lengthening) coexisting with falling elapsed cycles is the complete signature of "latency changing form"; reading only the share yields the opposite conclusion.
4. **The 80 regs / 28 B spill budget check cannot be skipped.** After address-arithmetic changes, register allocation is globally re-shuffled — r40's +13% depends on 3 blocks/SM, so any address-expression change requires re-reading the ptxas output to confirm the budget held.
5. **Transform code where two layouts coexist is a hotbed of stride confusion.** The host expander writes both `row_len` (224) and `orow` (256); the consumer copying the wrong side gets no signal. The defense is writing the layout contract as a doc-comment (that is exactly what the current tree's function comment does) and making the consumer's index expressions match the comment item by item.

## 4. Verification

| Gate | Reading | What it defends against |
|---|---|---|
| parity `cuda_prefill_mmq` (GPU vs CPU reference) | before: diff 448 @ index 554 (failing variants bit-identical); after: **1/0** | expression-layout mismatch — the direct evidence for this doc's root-cause verdict: one stride substitution eliminating all differences establishes the single cause |
| ptxas registers/spill | 80 regs / 28 B spill | fixing the address while losing the third resident block (r40's +13% depends on 3 blocks/SM) |
| ncu base-vs-fix elapsed cycles | 1,390,776 → 1,239,847 (−10.9%) | rules out "the mechanism never took effect" — the recomb ALU and its samples really did vanish |
| ncu Warp-Cycles/Issued-Inst + long_scoreboard share | 11.70 → 17.84; 34.9% → 57.1% | the self-consistency check of the mechanism explanation: latency changing form + the denominator effect |
| whole-prefill interleaved 3× (A/B within one binary) | 2595.9 → 2584.9 (−0.42%) | machine drift faking a trend; −0.42% is within the noise band, the verdict "wall-neutral" |

The gates' ordering is itself a discipline: **mechanism gates first (parity/ptxas/SASS), then performance gates (ncu/wall clock)**. Reversed — seeing wall-neutrality first and reverting — the root-cause fix would be discarded as "a useless change", and r53's bundle would lose its first component; celebrating on seeing kernel −10.9% first — the wall-neutral truth would be buried under mechanism excitement. Both directions have been erred before; this ordering table is the vaccine.

## 5. Results

| Layer | before (r43's failure state / r41 baseline) | after (r44's corrected state) | Verdict |
|---|---|---|---|
| parity | diff 448 @ index 554 (same diff in failing variants) | `cuda_prefill_mmq` 1/0 | **green** |
| ptxas | 80 regs / 4 B spill (r41 state) | 80 regs / 28 B spill | 3-block budget kept |
| kernel elapsed cycles | 1,390,776 | 1,239,847 (−10.9%) | mechanism took effect |
| Warp-Cycles/Issued-Inst | 11.70 | 17.84 | latency changed form |
| long_scoreboard share | 34.9% | 57.1% (absolute volume roughly unchanged) | denominator effect |
| whole-prefill (interleaved 3×) | 2595.9 tok/s | 2584.9 (−0.42%) | **wall-neutral** |

Measurement protocol per §0 convention: all numbers come from same-binary interleaved A/B medians within one session window — this window's 2595.9/2584.9 are local readings of the "post-r41 ~2600 tier"; cross-session absolute values are not comparable (machine state drifts); the kernel-level comparison comes from ncu sampling of matched-nt at the same shape. vs-llama stays at 1.27× — the wall does not move, so the ratio does not move.

**Veto mechanism (why reverted)**: r41's +30.7% had already pulled the q6_K GEMM off the wall's bottleneck seat (at r37 it was still 51.2% of wall). After that, anything improved inside the q6_K kernel — mechanism confirmed, parity all green, kernel −10.9% — no longer touches the critical path. The recomb the pre-expansion deleted merely converted "B's global→smem latency" from ALU-consumption form into STS-copy form, with an equivalent L1TEX exposure. Per the campaign rules (+1.5% bar, REVERTED keeps cmp-match HEAD), r44 reverted; the root cause and the "wall-neutral" verdict entered the record.

Worth emphasizing what the revert preserved: **what was reverted is the diff, not the knowledge**. Three things entered the record and were cited directly by later rounds — (1) the one-line root cause (the dense indexing formula), quoted verbatim in r53's bundle comments; (2) the mechanism explanation "the same physical latency changes instruction form to hide", which became standard practice for reading ncu share-class metrics; (3) the convergence evidence chain that "the q6_K kernel is no longer the wall" (r44 + r45, two independent mechanism lines), without which r47's fresh wall decomposition would not have known to measure FA and prepass instead of continuing to grind the GEMM.

**Under what future conditions a retry is worthwhile**: when the q6_K GEMM becomes part of the wall again, or when pre-expansion can be packaged with a "hide the wait" mechanism (cp.async). The condition cashed in precisely at r53: the `EXP=true` branch (dense indexing + cp.async pure copy + src-size zero-fill of tail rows) merged with r45's group-count pipeline into one basket — B staging became a pure copy with "no recomb ALU, no register round trips, no ql/qh reads", the latency handed to the asynchronous units. Results: ffn_down kernel 16.06 → 12.76 ms (**−20.5%**, r44's −10.9% and r45's −10.2% approximately adding), attn_v −15.9%, whole-prefill 3024.7 → 3176.9 (**+5.03%**). r44's work deletion and r45's wait deletion stacked because they are mechanically orthogonal (one deletes ALU, one hides latency), and the bundle's timing let the deletions touch the wall again. r54 then gave this 1.52 GB W_exp plane the `MINFER_MMQ_Q6K_EXP=0` three-state opt-out (default on / off-plane taking the byte-identical r41 fallback / a fallback label distinguishing intentional from accidental). r44 therefore belongs to this directory's most important category: **code correct, mechanism confirmed, no wall-clock value solo — not dead, WAITING**.

The cash-in ledger of the r44/r45 "WAITING" pair (two same-day vetoes, both flipped positive two days later inside one basket):

| Mechanism | Solo reading (r44/r45, both REVERTED) | Bundle reading | Cashed in |
|---|---|---|---|
| Pre-expanded B (deletes the recomb ALU, dense indexing) | kernel −10.9%, wall −0.42% | ffn_down kernel −20.5% (approximately adding with r45) | r53 |
| cp.async staging (hides the global→smem wait) | kernel −10.2%, wall −0.34% | whole-prefill +5.03% (3024.7 → 3176.9) | r53 (B side) + r56 (A side, another +2.35%) |

## 6. Lessons

1. **When two layouts coexist in a data transform, the consumer's address expression must be checked against the data source's actual layout** — the `bstride`/224 vs 256 confusion has no compile-time or runtime signal; only consumer-side parity can catch it.
2. **Readback verifies content, not offsets**; the complete gate for a pre-transformed plane = producer readback + consumer parity, neither dispensable.
3. **A bit-identical failure across variants is the fingerprint of shared address arithmetic** — contract to the shared part before acting; the control experiment (in-kernel expansion of raw W passing) is the key contracting step.
4. **On the converged line, kernel-level wins cannot reach the wall**: mechanism confirmed ≠ wall-clock value; a "correct but wall-neutral" change should be recorded with its mechanism and annotated with retry conditions, not discarded (r53 cashed in r44, r56 cashed in r45 — "not dead, WAITING" is a discipline this campaign has verified repeatedly).

← 46-r43-pc-sampling-attribution · [Index](./README.md) · 48-r45-cpasync-q6k-a-staging →
