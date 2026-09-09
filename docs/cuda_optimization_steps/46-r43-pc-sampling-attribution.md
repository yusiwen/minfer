# 46 · r43 — PC-sampling attribution + pre-expand-B parity FAIL (MEAS-ONLY + REVERTED)

> **Result**: the 33.6% `long_scoreboard` was named down to the **consuming instructions**: B-expand recomb (LOP3/SHF) 45% + A-side staging (STS.128) 28% + dsc consumers (I2F.S8) 26%. The trial "pre-expand B at registration (the W_exp plane)" was byte-perfect (readback 0/17,920) but **parity FAILED (diff 448 @ index 554)** and could not be isolated within budget → fully reverted; the paradox was solved by r44's one-line root cause (dense stride mismatch).
> **Commit**: `b7fa305` (**record-only commit** — the code never landed; the tree is at the r41 shape, cmp-match with HEAD). **Date**: 2026-09-05.

> **Code provenance note**: r43's trial code was reverted and the record commit contains docs only (verified with `--stat`: two docs, +78/+42). Per STYLE hard rule 0, this doc's attribution and failure narrative come from the `docs/CUDA_OPTIMIZATION.md` P6 r43 chapter + master table row 57 (sample counts quoted from its mirror `LLAMA-CPP-MMQ-ANALYSIS.md` §11.23); the code excerpts testify with the **current tree's** final correct form of the mechanism (r53's `expand_q6k_dense` + the two-gate test) — these are precisely the surviving versions of the two gates r43 built back then.

## 1. Background — where things stood

r42 turned "widen the dsc narrow reads" into a clean zero-gain experiment: byte counts and throughput both moved, yet the 33.6% `long_scoreboard` did not budge one notch. r42's parting
conclusion was unambiguous: **stop guessing; bring in instrumentation that resolves individual instructions** — "Warp-Stall-Sampling source attribution on
`mmq_raw_nb_bt_q6k_kernel<2>` to identify the actual instruction behind the 33.6%".

Where the instrument generation gap was: the ncu Warp State used in r41/r42 only answers "which **class** of wait is the warp stuck in" (`long_scoreboard` = an L1TEX
dependency), not "stuck on **which instruction**". The byte counters of the r13/r21 era (sectors, queues) are a data-plane view. 33.6%
had now survived two rounds un-dismantled, which means the mental model of it ("which load causes it") was wrong in direction — a tool giving a **PC (instruction-address)-level** distribution was
needed.

Meanwhile, the campaign-level situation: after r41, whole-prefill 2605.2 (vs-llama 1.27×), with the q6_K kernel still the largest single node on the default path. Even after r41's 16× cut in
load count, 33.6% of stall remained; if that number were "irreducible", the q6_K line would be over — so r43's second task was to decompose the 33.6% into an actionable lever list, and even a
failure had to end knowing **why** it failed.

## 2. Principle — the GPU mechanism

**The sampling mechanism.** ncu's source-counter sampling (`--set full --section SourceCounters`, viewed in the `--page source`
source-level view) periodically hardware-samples each warp during kernel execution and books the warp's state at that moment (issue-stalled long scoreboard, waiting on a barrier, etc.) to the
**instruction at that warp's current PC**, aggregating per-instruction counters such as `pcsamp_warps_issue_stalled_long_scoreboard`.

**This is the third tier of the campaign's instrumentation lineage.** The previous two tiers each had blind spots:

| Instrument | Rounds used | Question answered | Blind spot |
|---|---|---|---|
| Byte/sector counters (r13 forensics, r21 sectors −28.6%) | r13, r21 | data plane: who moves how many bytes | sectors saved, wall unmoved (r21's "stall conservation" lesson) |
| Warp State summary (since r20) | r20, r41, r42 | the **class** shares of stall (long_scoreboard 85.5% → 33.6%) | class-level, not instruction-level — r42 misfired because of it |
| PC-sampling source-level sampling (this step) | r43 | which **instruction's** PC carries the stall | sampling is statistical; it needs sufficient sample volume |

Once r42's zero-gain experiment sealed off the "add more width" road, the third tier was the natural next step: to keep dismantling the 33.6%, we had to know **which specific instructions carry
it**.

**Why it books to the "consumer" and not the "loader".** `long_scoreboard`'s semantics: this warp wants to issue some instruction, but one of its input operands depends on a load that has not
yet returned (the L1TEX round trip is incomplete). The PC where the warp stalls is **the instruction waiting for the input** — i.e. the consumer. That is:

- The Warp State summary says "warps are waiting on L1TEX";
- PC-sampling says further "**it is this I2F / this LOP3 / this STS that is waiting**".

For a `load → ALU consume` chain, the load side can change width at will (r42 proved it useless); as long as the **consumer-side tight dependency** remains, the stall is booked to the consuming
instruction. That is the mechanical explanation of r42's phenomenon, and the source of this doc's first payoff.

**What this instrument can and cannot tell you.** It can name the 33.6% share **down to instructions** (this doc's deliverable), but it does not explain "why this instruction waits this long" —
where the latency comes from (L1 round trip, bank conflicts, dependency-chain depth) still needs mechanism hypotheses backed by SASS. r43's
usage is therefore two-stage: first sample-and-name (45/28/26), then explain each mechanically and derive the lever class. One more limitation: sampling is statistical, and with insufficient
sample volume the shares of low-frequency instructions are untrustworthy — this round's 51,605
samples with all three big heads in the thousands make the attribution base solid, but the fine items "below 4th place" should not be quoted.

**The sample-volume account.** Of this round's 51,605 samples, 16,647 landed on `long_scoreboard` = 32.3%, consistent with the 33.6%
share reported by the Warp State summary — two independent instruments interlock, and the attribution's foundation is stable.

## 3. Implementation

### 3.1 The attribution result: three consuming instructions split the 33.6%

Source-level sampling of `mmq_raw_nb_bt_q6k_kernel<2>` (post-r41 shape) split the `long_scoreboard` share across three consuming instructions:

| Share | Consuming instruction | Waiting on | Mechanism reading |
|---|---|---|---|
| **45%** | `LOP3.LUT 0xff` + `SHF` (the B-expand recomb's first ALU group) | the uint4 ql/qh load round trips after r41's widening | r41 cut the **count**, but double buffering put staging at the top of the kt loop with recomb right behind — the **within-warp** tight dependency chain remains; r39's double buffering only hides the cross-warp part |
| **28%** | `STS.128` (the A-side qa8/sda bulk LDG→STS copy) | the A-side staging's global load round trips | pure copy latency; also **falsifies r42's side hypothesis** — the A reads are contiguous merged uint4 accesses, there is no "strided A" |
| **26%** | `I2F.S8` (dsc's `(float)(int8_t)sc` conversion) | the d/scale byte load round trips | stamps r42's verdict: dsc is a **latency source, width innocent** — widening loads can never reach a stall booked to the consumer |

The three sum to ≈ 33.6% (45+28+26 = 99% of the named share). The mechanism reading of each:

- **45% recomb (LOP3/SHF)**: after r41 the recomb chain is "uint4 load → first byte extract+mask (byte-wise and/or of the `LOP3.LUT 0xff` kind) → shift-assemble
  → −32 → STS". PC-sampling books the stall to the **first ALU** — it waits out the entire load round trip. So this 45% is the other face of r41's legacy: loads went from 32 to 2, but the
  tight "staging→consume" dependency structure survived intact, and the first consumer's wait is the chain's entire exposure. This rules out scheduling micro-tuning like "split the recomb
  differently/denser" — **either eliminate the consumer end (pre-expansion), or make the loads asynchronous (cp.async)**.
- **28% A-staging (STS.128)**: in the A-side qa8/sda bulk LDG→STS, the STS is the consumer of the load results, so the stall books to it. Its value is falsifying r42's side hypothesis — r42
  suspected the A reads were strided accesses across rows and blocks, but the A reads are inherently contiguous uint4s (the prepass pre-transposed pad40 plane); the 28% is pure copy latency, not
  "strided A".
- **26% dsc consumer (I2F.S8)**: the narrow read→conversion of `(float)(int8_t)blk[192+s0]`. It mechanizes r42's verdict: a stall booked to the conversion means **the load end is irrelevant at
  any width** — only moving the `d·sc` computation out of the kernel (r56's W_dsc plane) or moving the bytes into an asynchronous pipe gives this 26% room to move.

The list's execution value is in its ordering: **the biggest head is the recomb's consumer end** — taking the recomb out of the hot loop entirely (not changing its input's width) is the correct
lever. That is §3.2's trial.

### 3.2 The trial: pre-expand B at registration (the W_exp plane)

The idea is isomorphic to what r38 did — moving the recomb out of the `mma` inner loop — but goes further: out of the **entire kernel**. At load time, expand the padded raw-W once into a
**centered-int8 dense plane `W_exp` (od × id, row pitch = id, super-block pitch = 256)**, so the kernel's B staging degenerates from "read ql/qh + recomb" into a **pure bulk copy** (same shape
as the A side), and the 45% LOP3/SHF samples should vanish wholesale.

**Why registration time is the correct place to do the work.** The recomb is a pure per-weight function: the same weight's every byte is consumed repeatedly across one prefill (every output tile
re-reads B), but computed only once at registration — the amortization is infinite. The cost is an `od × id`-byte dense plane (the q6_K tensors of q4_K_m 7B total ~1.5 GB in magnitude; r53's
landing measured +1.52 GB device). At r43 this was a clear trade: 1.5 GB for a 45% sample share — worth trying, provided the correctness gates pass first.

The registration-time expander surviving on the tree (the corrected post-r44 form; the trial's "intended semantics" matches it):

```rust
// src/cuda.rs — expand_q6k_dense: padded raw-W → dense centered-int8 plane
// input row pitch nbe*224 (padded); output row pitch id, super-block pitch 256 (DENSE)
pub fn expand_q6k_dense(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
    const Q6KB: usize = 210;
    const Q6KPB: usize = 224;
    let nbe = id / 256;
    let row_len = nbe * Q6KPB;
    let mut out = vec![0u8; od * id];
    for j in 0..od {
        let prow = &padded[j * row_len..(j + 1) * row_len];
        let orow = &mut out[j * id..(j + 1) * id];
        for sb in 0..nbe {
            let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
            let (ql, qh) = blk.split_at(128);
            let obase = &mut orow[sb * 256..sb * 256 + 256];   // dense sb pitch = 256
            for it in 0..2usize {
                for r in 0..64usize {
                    let qlb = ql[it * 64 + r];
                    let qhb = qh[it * 32 + (r & 31)];
                    let s0 = (r >> 5) * 2;
                    let e0 = it * 128 + r;
                    obase[e0] = ((qlb & 0xF) | (((qhb >> s0) & 3) << 4)).wrapping_sub(32);
                    obase[e0 + 64] =                            // e+64 shares the same ql byte
                        (((qlb >> 4) & 0xF) | (((qhb >> (s0 + 4)) & 3) << 4)).wrapping_sub(32);
                }
            }
        }
    }
    out
}
```

Note the juxtaposition of the two strides: input side `sb * Q6KPB` (padded 224), output side `sb * 256` (dense). **This is exactly the pit r44 would expose** — see §3.3.

And the r43 trial's kernel arm consumed the plane as (per the record) "the raw `W` pointer expression + reading the `W_exp` buffer" — i.e. indexing a densely laid-out plane with the padded
raw-W address algebra `W + j·(nsb·bstride) + sb·bstride`. The correct kernel arm as written after r53 (for contrast):

```cuda
// src/cuda_kernels.cu — EXP arm (r53 landed shape): dense indexing W_exp + j*id + sb*256 + cbase*32
const int nc = (KDR * 32) / 16;   /* 16B groups per row */
const int ncopy = MMQ_NBJ * nc;
for (int g = threadIdx.x; g < ncopy; g += blockDim.x) {
    const int jj = g / nc, cc = g % nc;
    const int j = j0 + jj;
    const bool full = (j < od) && (sb < nsb);
    const uint8_t* src = W_exp + (size_t)j * id          // dense row pitch = id (not nsb*bstride!)
        + (size_t)sb * 256                               // dense sb pitch = 256 (not bstride!)
        + (size_t)(cbase * 32 + cc * 16);
    gemm_cp16((__half*)(void*)(qbexpb + (size_t)jj * (KDR * 32) + cc * 16),
              (const __half*)(const void*)src, full);    // pure copy: the recomb was completed at registration
}
```

### 3.3 Pitfalls: the byte-correct but parity-red paradox

The two gates' readings came out inverted:

- **Content gate green**: the `W_exp` device readback vs the host mirror was **0/17,920 mismatches** — the plane itself is byte-correct.
- **Output gate red**: kernel parity failed, **max diff 448 @ index 554**, and this signature was **bit-identical across three kernel variants** (uint4 copy of `W_exp`, per-byte copy of
  `W_exp`, and a control arm expanding raw `W` in place via `expand_q6_elem` which **passed**, while any variant consuming `W_exp` **failed**).

The control arm passing narrowed the suspicion to the extreme: branch, bounds, staging buffers, and the staging→mma path are all shared and correct; the only thing that follows the failing
variants is **the `W_exp` buffer itself** — a "byte-correct, consumption-fails" combination. r43 could not isolate this kernel-side interaction within budget (the hypothesis of the moment was
some in-kernel aliasing) and reverted by discipline (cmp-verify = HEAD r41).

In hindsight (r44's post-mortem language), the pass/fail split of the three variants was already pointing at the answer: **"that expression on `W`" was exactly all-right, "the same expression on
`W_exp`" exactly all-wrong** — the only degree of freedom in the difference is the data source's layout, and the failure did not vary with the copy method (uint4/per-byte), placing the problem in
the **address → data mapping**, not the copy's execution. The missing step at the time was writing the two layouts' strides side by side and doing one line of arithmetic; r44 did exactly that
line (r43's three variants "split exactly as that
predicts"). The lesson is not "should have tried more within budget" but: **for any change where two layouts coexist, step one is writing the two stride sets' difference into the verification
checklist**.

r44's one-line root cause demolishes the paradox: **the address expression and the plane layout mismatch**. r43's kernel indexed the densely laid-out `W_exp` (row pitch `id`, sb pitch 256) with
the padded raw-W strides (row pitch `nsb·bstride`, sb pitch `bstride` =
224). Take id=256 (nsb=1) and do the concrete arithmetic:

```
r43's address:      W_exp + j*(nsb*bstride) + sb*bstride = W_exp + j*224 + sb*224
the correct address: W_exp + j*id          + sb*256      = W_exp + j*256 + sb*256

at j=1, sb=0: it reads [224 .. 224+224) — in the dense plane that is
  row 0's [224..256)  (32 B, the last 32 elements of row 0)
+ row 1's [0..224)    (224 B, the first 224 elements of row 1)
```

That is, except for j=0, every row's window slides wholesale into "the previous row's tail + this row's head" — element-level misalignment, yet every byte read is a **legal centered int8**
(some other element's value in the −32..31 range). So the output is a "systematically wrong, but not absurd" diff of 448 (@ index 554), not obviously-fake garbage. The control arm expanding raw
`W` passed precisely because raw `W` **is** the padded layout — the expression happens to be right for it. The content gate only verifies CONTENT; OFFSET is decided by the address expression; the
two gates each guard half, and missing either leaks.

## 4. Verification

- **PC-sampling sample-volume self-consistency**: 16,647/51,605 = 32.3% vs the Warp State summary's 33.6% — the attribution base interlocks with the existing instrumentation (defends
  against sampling bias / the new instrument reading the wrong object).
- **Content gate (readback)**: `W_exp` device readback vs host mirror 0/17,920 — verifies the plane's **content** (defends against expansion-algebra errors; its surviving version is the
  host+device two-segment assertion `cuda_q6k_exp_dense_byte_exact` on the tree, see below).
- **Output gate (parity dump)**: diff 448 @ 554 — verifies **offsets** and end-to-end semantics (defends against "content right, addresses wrong").
- **Control-arm variants**: expanding raw `W` in-branch passed, consuming `W_exp` failed — the bisection instrument (compressing suspicion from the whole path to a single buffer).

The live form of this two-gate setup on the tree (written in the r53 era; the semantics are exactly r43's two gates):

```rust
// src/graph/cuda_backend.rs — cuda_q6k_exp_dense_byte_exact (excerpt)
// host expander vs independent scalar mirror (content gate, host half)
let host = crate::cuda::CudaState::expand_q6k_dense(&padded, od, id);
let hmis = host.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
assert_eq!(hmis, 0, "expand_q6k_dense vs mirror ({od}x{id})");
// device upload + pinned readback vs the same mirror (content gate, device half)
let p = state.get_weight_ptr(&exp_name).expect("W_exp registered");
let mut got = vec![0u8; od * id];
state.copy_from_device_pinned(p, &mut got);
let dmis = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
assert_eq!(dmis, 0, "device W_exp vs mirror ({od}x{id})");
```

(r43's lesson was then written into r53's landing preconditions: the content gate must exist **simultaneously** with the kernel-parity gate — the test above covers only the content gate; offset
correctness is guarded separately by the kernel arm's dense indexing + the parity dump.)

## 5. Results

- **Attribution (this step's deliverable)**: the 33.6% `long_scoreboard` = B-expand recomb 45% + A-staging STS.128 28% + dsc I2F.S8
  consumers 26%; the residual is **not irreducible** — it is within-warp staging latency exposure (double buffering hides across warps, not across a warp's internal dependency chain).
- **Trial (this step's veto)**: pre-expand-B's `W_exp` was byte-correct (0/17,920) but parity FAILED (diff 448 @ 554, same signature in all three variants) → not isolable within budget →
  **REVERTED**, the tree cmp-matches HEAD; whole-prefill stays around 2605.2 (vs-llama 1.27×), no wall change.
- **Aftermath (forward reference)**: r44 dissolved the paradox with a one-line root cause (dense/padded stride mismatch); the corrected version went parity-green but wall-neutral (−0.42%) —
  removing the recomb merely **transferred** the latency's carrier; the final landing was r53's basket (r44's de-work + r45's de-wait + cp.async) at +5.03%. r43's attribution list was exactly
  that basket's design input.

**The q6_K convergence judgment after r43** (in the record's own terms): the residual is **not irreducible** — it is within-warp staging latency exposure, with two classes of physical lever:
cp.async-ify the staging of raw A (and pre-expanded B) (llama.cpp's structure), or a register-constrained split-phase. This judgment set the direction of all three following rounds:

| Round | Lever | Kernel effect | Wall effect | Status |
|---|---|---|---|---|
| r41 | B-expand uint4 widen | 1.70 → 0.654 ms | **+30.7%** | LANDED |
| r42 | dsc read widening | −1.8% | −0.19% | REVERTED |
| r43 | PC-sampling attribution + pre-expand-B | — | — (parity FAIL, reverted) | MEAS + REVERTED |
| r44 | W_exp stride fix (recomb vanishes) | −10.9% cycles | −0.42% | REVERTED |
| r45 | A-side cp.async (de-wait) | −10.2% | −0.34% | REVERTED |
| r53 | r44+r45 merged basket + cp.async B | ffn_down −20.5% | **+5.03%** (3024.7 → 3176.9) | LANDED |

Viewed alone, r43 is one attribution plus one failure; viewed in the campaign line, it turned the 33.6% from "a number" into "three executable levers", and r44/r45's "each wall-neutral alone,
over the wall merged" is the direct product of testing r43's list item by item.

## 6. Lessons

1. **Attribute stalls to consuming instructions, not loading instructions**: `long_scoreboard` books to the instruction waiting for input; changing load width (r42) cannot reach a stall
   booked to the consumer — only eliminating the consumer-side dependency (moving it out of the hot loop) or switching to an asynchronous path works.
2. **Byte-correct ≠ offset-correct**: readback only verifies CONTENT; data landing on the wrong address expression is more dangerous than "obviously wrong data" — the output stays in a legal
   value range and the failure signature is stable, steering the investigation toward wrong hypotheses like in-kernel aliasing.
3. **The control variant is the cheapest bisection instrument**: a "same branch, swapped data source" control (raw W passes / W_exp fails) compresses the suspicion to a single buffer in one
   step, faster than any static review.
4. **MEAS-ONLY still counts as landing**: this round's instruction-level list directly fed r44/r45/r53's designs — one failure with clear attribution beats one success with a vague mechanism.

---
← [45 · r42 stage-wide dsc scale read](45-r42-stage-wide-dsc-read.md) · [Index](./README.md) · [47 · r44 W_exp stride mismatch root cause](47-r44-wexp-stride-mismatch.md) →
