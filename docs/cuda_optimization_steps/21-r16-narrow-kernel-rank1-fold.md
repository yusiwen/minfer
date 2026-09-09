# 21 · r16 — Narrow kernel gets the rank-1 fold (LANDED)

> **Result**: narrow-kernel MMQ KD=8 **480.2/480.9 vs baseline 447.0/472.6 tok/s** (+1.7%/+1.8%, inside the noise band) — this step's value is not the tok/s but re-aligning the instruction-stream structure of the two raw kernels: the narrow kernel serves as the wide kernel's control and fallback path, and cannot keep living with an old epilogue.
> **Commit**: `151fa97`. **Date**: 2026-09-03 (same-morning follow-up, 26 minutes after r15 landed).

## 1. Background — where things stood

r15 landed the rank-1 term2 fold in the wide kernel (`mmq_raw_wide_nt_kernel`): KD=8 1271 → 1295 tok/s, instruction stream 9.45 → 8.69 M/GMAC. The boundary decision at landing was **to touch only the wide kernel** — it is the performance path (1295 vs the narrow kernel's ~480, 2.7×), and the narrow kernel was kept as the control for that round's A/B.

But that left a structural problem. At this point the narrow kernel `mmq_raw_nt_kernel` sat at the intersection of three roles:

1. **Control**: every later wide-kernel experiment's "invariant" is vouched for by it — if the control itself carries an epilogue with a different instruction-stream structure, the comparability of the "control stable" conclusion quietly leaks.
2. **Fallback path**: the wide kernel at KD=8 needs 135 KB of dynamic smem; above the ~99 KB opt-in cap the launcher refuses it and falls back to the narrow kernel (the guard established in r7–r8). A slower fallback is acceptable; **structural drift** is not — it is another implementation of the same math.
3. **Template for variants**: later kernel experiments (r17's warp remap, the more distant NB variants) all copy fragments from these two implementations. If the source epilogues disagree, everything copied needs a per-copy audit.

So r16's question is pure hygiene: **port** r15's fold to the narrow kernel so the two raw kernels' term2 evaluation becomes consistent again. The budget was "one-shot follow-up" — r15 landed at 10:20 in the morning, r16 was committed at 10:46, a 12-line change (+10/−2).

One more layer of on-the-ground reality about the "fallback path" role: the launcher carries an smem-cap guard for the wide kernel (the r7–r8 lesson — the wide kernel's first version at KD=8 needed 135 KB of dynamic smem and **failed silently** above the ~99 KB opt-in cap, producing ghost numbers; since then the launcher guards explicitly and falls back to the narrow kernel). The post-r12-rewrite wide kernel at KD=8 is 98,304 B (single-buffer synchronous staging, 1 block/SM; KD=4 is 73,728 B) — just under the line — but the guard is kept as a safety net, and any smem-budget fallback executes the narrow kernel. **The fallback path's instruction structure is therefore not "historical legacy" but "active insurance"** — one of the real motives for this round's port.

The config entry point too: the recommended config at the time was `MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_WIDE=1 MINFER_MMQ_RAW_KD=4` (recorded in r12) — `MINFER_MMQ_RAW_WIDE=0` drops back to the narrow kernel. The narrow kernel is the only "structurally equivalent, smaller shape" alternative execution path in this gate set.

The expected gain was noise-scale to begin with: the narrow kernel's absolute level (~447–481) is 2.7× away from the campaign's main battlefield (wide kernel → llama.cpp's 41.1 µs/GMAC), and ±2% of session noise swamps any single-step narrow-kernel gain. The master table's verdict on this row is honest: "**+1.7% (noise-band)**; narrow is not the perf path".

## 2. Principle — the GPU mechanism

### 2.1 The fold itself (a three-sentence refresher)

In q4_K's two-term rescale, term2 = `dmv_j · (da_i · sa_i)` — the column factor `dmv` (the per-od-col negative-bias term) and the row factor `da·sa` (the per-token-block A scale × in-block integer sum) form a **rank-1 outer product** on the (token, od-col) plane. Per-C-value evaluation re-multiplies the row factor over and over; folding = compute `dma = da·(float)sa` once per token row, reducing each C value to one FFMA. The numeric-class change is a single re-association of one multiplication (~ulp(|sum|)), already argued and gated in r15.

### 2.2 The narrow kernel's geometry: why the fold's accounting differs

First a glance at the battlefield as it stood on r16's morning (all figures are token rates measured interleaved within their own sessions; not comparable across sessions):

| Kernel/config | tok/s (@2K-class anchor) | Source |
|---|---|---|
| R1 word-level MMQ (first parity-clean version) | 441 | r7–r8 window |
| Narrow raw kernel KD=8 (local optimum after r9's shape matrix) | 481 | r9 |
| Narrow raw kernel (baseline band on r16's day) | 447–481 | r15/r16 |
| Wide 16-chain kernel KD=4 / KD=8 (after r14) | 1225–1233 / 1273–1295 | r14/r15 |
| Same-session f16 default path | 2284–2370 | r8–r12 period |

The narrow kernel's position in this table determines the round's nature: it is a **maintained path**, not an **advanced path**.

The fold's benefit depends on the ratio of per-thread C values to distinct rows — a ratio set by the warp tile, which differs between the two kernels:

| | Narrow `mmq_raw_nt_kernel` | Wide `mmq_raw_wide_nt_kernel` |
|---|---|---|
| block tile | 64 tok × 64 od (`MMQ_BI/BJ = 64`) | 128 tok × 128 od (`MMQ_WBI/WBJ = 128`) |
| warp tile | 32 tok × 16 od (8 warps split as 4 od slots × 2 tok slots) | 128 tok × 16 od (each warp owns the full token axis) |
| per-thread C values/chunk | 16 (`sum[16] = [nh][h][l]`) | 64 (`sum[64] = [g][nh][l]`) |
| distinct token rows per thread | **4** (`lane>>2 + t4*8`, t4 ∈ 0..3) | 2 (uint2-packed token pairs) |
| folded FMUL/chunk | 4 (`dma[4]`) | 16 (8 g × `dma[2]`) |

The narrow kernel's 16 per-thread C values land on 4 token rows; the row formula is plain in the current tree (`src/cuda_kernels.cu:5801`):

```cuda
const uint8_t* at = qat + (size_t)(i0w + (lane >> 2) + t4 * 8) * 40;
da_q[t4] = h2f(*(const uint16_t*)at);            // f16 scale @ +0
sa_q[t4] = (int)*(const uint32_t*)(at + 36);     // in-block integer sum @ +36
```

That is, each thread covers the four rows `lane>>2 + {0,8,16,24}` (t4 ∈ 0..3), and the 16 C values map back onto exactly these four rows via `h*2 + (l>>1)` — after folding, `dma[4]` holds one entry per row, no more and no less. The fold compresses it from "multiply per value" to "multiply per row"; the commit message records the accounting: **8 I2F + 16 FMUL → 4 I2F + 12 FMUL per chunk**. Half of the I2F cut comes from "structured sharing": the old code wrote `(float)sa_q[h*2+(l>>1)]` inside the value loop, and ptxas kept 8 copies after unrolling; the new code's `dma[4]` array is built explicitly outside the loop, exactly 4 copies — no longer relying on the compiler's CSE behavior.

### 2.3 Why a noise-scale gain is still worth landing

The narrow kernel's +1.7% falls inside the session noise band (the two same-round baseline runs, 447.0 and 472.6, differ by 5.7% — itself a confession of the noise width). But this is not a "no-op change":

- **Control validity**: from here on, the "narrow control stable" conclusion of every wide-kernel experiment round is clean again — the two kernels' term2 evaluation structures agree, and the control's sensitivity to instruction-stream-class changes returns to the same baseline. This is not an abstract worry: r17 (the next round) was about to lean on the narrow kernel as its control, and r15/r16 are precisely the process of making that control meaningful.
- **Proof of pattern portability**: r15's fold landed under two very different layouts (the wide kernel's uint2-packed staging vs the narrow kernel's direct 40 B raw-chunk reads) — this paved the way for extending it to the q6_K kernel family later (Era D's r38+ series; in the current tree `dma[l >> 1] * dmv[...]` appears in the NB/BT kernels).
- **Epilogue convergence signal**: with both raw kernels' term2 folded, the question "how much epilogue is left to shave" has a common answer (the rescale floor ~346 ops/chunk recorded in r15) — the narrow kernel is no longer the exception "still carrying the old tail".

For completeness, the numeric intuition behind the fold's correctness (the same argument as r15 §2, here in the narrow-kernel version): for the same (i, j) the old code computed term2 as `(da_i · dmv_j) · sa_i`, the new one as `(da_i · sa_i) · dmv_j` — three reals, two floating multiplies, only parentheses swapped. Under IEEE 754 multiplication is commutative but not associative, so the difference bound is ~ulp(|sum|); two orders of magnitude of headroom against the 1e-3 parity gate. r15 already gated on this argument, and r16 reuses the same numeric class.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Port, don't re-derive**. The math was argued in r15; the narrow kernel's C-fragment row/column mapping copies the same `l>>1` / `l&1` conventions; the only thing to recompute is the row count (4 vs 2). The diff is therefore just two hunks: add the `dma[4]` precompute, rewrite one term2 line.
- **Comments carry provenance**. The ported comment keeps the "r15" tag (current tree `src/cuda_kernels.cu:5805`) — it records the fold's originating round, not the porting round; a reader who follows the tag finds the full numeric argument without re-deriving it in both kernels.
- **term1 and the per-chunk scale application untouched**. Minimal blast radius: the only permitted numeric change is the one re-association already argued in r15.
- **The R1 word-level kernel (`mmq_nt_kernel`) is explicitly not ported**. It is the legacy comparison path (reached only with `MINFER_MMQ_RAW=0`); investing in it has negative value — which makes explicit half of this doc's lesson: "**keep sibling kernels instruction-compatible when a fold lands — or drop the sibling**". The two raw kernels chose "keep"; the R1 kernel chose "drop" (left living with its old shape until retirement).
- **Why fold term2 and not term1** (the question review will ask): in term1 = `da · dsv · clow` the `clow` is mma's per-(i,j) output — full rank, nothing to fold; term2 is the only outer-product term built purely from per-row/per-column coefficients. r15 already drew the fold's applicability boundary; the port does not reinvent it.

### 3.2 Key code

The complete diff of `151fa97` (`src/cuda_kernels.cu`, +10/−2, two hunks):

```cuda
// ── HUNK 1: dma[4] precompute (r15 pattern + the narrow kernel's 4-row geometry) ────────
             da_q[t4] = h2f(*(const uint16_t*)at);
             sa_q[t4] = (int)*(const uint32_t*)(at + 36);   // direct 40B raw-chunk read
         }
+        // r15: the dmv correction term is rank-1 in (token, od-col) —
+        // the row-side product da*sa is shared by the od-col pair of
+        // each C fragment, so fold it once per row (4 FMUL/chunk)
+        // instead of once per C value (8 FMUL/chunk). The dsv term and
+        // the per-chunk scale application are unchanged.
+        const float dma[4] = { da_q[0] * (float)sa_q[0],
+                               da_q[1] * (float)sa_q[1],
+                               da_q[2] * (float)sa_q[2],
+                               da_q[3] * (float)sa_q[3] };
         float dsv[2][8], dmv[2][8];

// ── HUNK 2: term2 rewrite ────────────────────────────────────────
                     for (int l = 0; l < 4; l++) {
                         const float da = da_q[h * 2 + (l >> 1)];
-                        const float sa = (float)sa_q[h * 2 + (l >> 1)];
                         const int jj = (lane & 3) * 2 + (l & 1);
                         const int idx = nh * 8 + h * 4 + l;
                         sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
-                        sum[idx] += da * dmv[nh][jj] * sa;
+                        sum[idx] += dma[h * 2 + (l >> 1)] * dmv[nh][jj];
                     }
```

Two layout differences vs the wide kernel are worth comparing side by side (this is the "port ≠ copy" part):

- **Row indexing**: the wide kernel has 2 rows per g, `dma[l >> 1]`; the narrow kernel's warp tile interleaves two A-frags (the `h` dimension) with 4 rows per thread, so it is `dma[h * 2 + (l >> 1)]`. The row sets differ (wide: uint2-packed token pairs; narrow: the four rows `lane>>2 + t4*8`), so the fold point must be re-derived from each kernel's C-fragment row mapping.
- **A-side data source**: the narrow kernel reads `da` (f16 @ +0) and `sa` (uint32 @ +36) directly from the 40 B raw q8 block; the wide kernel reads two token pairs in a single LDS.64 from uint2-packed staging. The fold logic is transparent to both — it depends only on the rank-1 structure, not on the layout.

Current-tree location: `src/cuda_kernels.cu:5805–5833` (the narrow kernel's per-chunk epilogue). The A-side `sa` has the same origin as in r15: a free byproduct of the A-quantize prepass (the q8_1 block's `s` field).

The third shape in the same file (the control): the R1 word-level kernel `mmq_nt_kernel`'s epilogue is still the pre-fold form (`src/cuda_kernels.cu:5620`):

```cuda
float dsv[2][8], dmv[2][8];
// …
if (HAS_OFF) sum[idx] += da * dmv[nh][jj] * sa;   // old form, legacy path kept intentionally
```

After this round the file genuinely carries three shapes — the two raw kernels (new) and the R1 word-level kernel (old). When reading the code, judge which form to align to by "which kernel is the active path", not by "which spelling is newer".

### 3.3 Pitfalls

- **Row indexing was the port's only real minefield**. The two kernels' `dma[...]` index expressions look almost identical (`l >> 1` vs `h * 2 + (l >> 1)`); a pure copy would move the wide kernel's 2-row indexing into the 4-row narrow kernel — it compiles, term1 stays correct, and term2 is wrong for exactly half the rows. The parity gate catches this class of error, but locating it costs far more than the two minutes spent drawing each kernel's row map before porting.
- **Don't "fix" the old kernel in passing**. After the port the file contains three term2 shapes: the two raw kernels' `dma[...] * dmv[...]` (new) and the R1 word-level kernel's `da * dmv[nh][jj] * sa` (old, `src/cuda_kernels.cu:5620`, behind the `HAS_OFF` gate). The old form is a **deliberately kept** legacy path — a reader should not mistake it for a missed bug.
- **Noise-band measurement takes patience**. Measured as a single round, the narrow kernel's +1.7% (the single pair 447 → 480) reads as a real gain; only interleaved 2× plus triangulation against r15's narrow-control series (473/472/472) puts it back in the noise band. The honesty of the conclusion depends on the density of controls, not on the sign of the delta.

## 4. Verification

Each gate defends one class of regression (STYLE's one-sentence-per-gate convention):

- **Parity gate ×3 configs** (`cuda_prefill_mmq`): default (MMQ off — defends against "touched the raw kernel and broke the default path"), `MINFER_MMQ=1 MINFER_MMQ_RAW=1` (KD=8, the narrow kernel's main config), and `MINFER_MMQ_RAW_KD=4` (KD=4). Both depths must pass: KDR changes the restage-loop structure outside the epilogue, so the fold must hold at both cadences; `HAS_OFF`-class branch differences can also surface at only one depth.
- **Suite 166/0/3**: the full test suite (166 pass / 0 fail / 3 skipped) — defends against collateral damage outside the GEMM.
- **Interleaved 2× A/B vs the HEAD binary**: 480.2/480.9 vs 447.0/472.6 — defends against machine drift polluting the delta (the r59b lesson, internalized as procedure: only same-window pairs count).
- **Noise-frame triangulation**: r15's same-round narrow-control series (473/472/472) serves as the noise-band reference, placing +1.7%/+1.8% back inside the band — defending against "reading noise as gain".
- **Deliberately no ncu**: the narrow kernel is not the performance path; the profiling budget goes to the wide kernel. This is a measurement discipline — **tool time follows the wall, not the change** — but the flip side is that the change's conclusions lack instruction-level evidence, so the master table honestly records "no ncu (narrow is not the perf path)", making the evidence boundary explicit.

## 5. Results

| Metric | before | after | Verdict |
|---|---|---|---|
| Narrow kernel KD=8 (interleaved 2× median) | 447.0 / 472.6 | **480.2 / 480.9** | +1.7%/+1.8%, inside the noise band (r15 control 473/472/472) |
| Parity | — | KD=8 + KD=4 all green (3 configs) | ✅ |
| Suite | — | 166/0/3 | ✅ |
| Wide kernel | — | untouched (1295 tok/s held) | ✅ |
| Per-chunk epilogue ops (commit accounting) | 8 I2F + 16 FMUL | 4 I2F + 12 FMUL | fold delivered |

Master table row 32's record: "480 vs 447–473 | +1.7% (noise-band) | LANDED | 3-site port of r15; narrow is not the perf path". The campaign-level accounting: the narrow-vs-wide gap stays 2.7×, and this round changed no battlefield number; what changed is that **the two raw kernels' term2 evaluation structures agree again**, and the fold pattern's portability was proven on a second layout (direct 40 B raw-chunk reads) — the pattern was later reused across Era D's q6_K kernel family.

A note on the "3 sites" wording in the record: the diff itself is a single epilogue in two hunks (add `dma[4]`, drop the per-value `sa` conversion, rewrite the term2 line); "3-site" refers to the three code locations the port touched. Counted either way, this is a line-scale follow-up after r15; the master table gives it its own row because of the **status change** (the control kernel became trustworthy again), not a performance change.

## 6. Lessons

1. **When a fold lands, either bring the sibling kernels' instruction structure into alignment or explicitly retire the sibling** — a control kernel carrying an old epilogue quietly devalues every later round's "control stable".
2. **Noise-scale gains can still be worth landing** when the lever is hygiene (control validity, pattern portability) rather than speed; the criterion is "what do later measurements depend on", not "how many points did this earn".
3. **When porting a numeric fold, re-derive the indices from the target kernel's C-fragment row mapping** — do not copy by textual similarity; the same math yields different index expressions under different warp tiles, and the failure symptom (term2 half wrong) is inconspicuous.

---
← [20 · r15 f32-accumulate probe + rank-1 fold](./20-r15-f32-acc-mma-rank1.md) · [Index](./README.md) · [22 · r17 wide warp remap](./22-r17-wide-warp-remap.md) →
