# 113 · #844: the packed Q8_0 decode takes the QKV fusion on Metal — and its rope becomes bit-equal (LANDED)

> **Result**: the packed cache's Metal decode penalty is gone for the class that could take
> the fusion, and the two classes that exist are both covered. On Qwen2.5-0.5B q4_0
> (`bench -p <n> -n 64 -r 3 --n-ctx 4096`, 3 interleaved rounds, medians) the packed-vs-f16
> step gap falls **634 → 124.8 µs/token** at `-p 8` (**1.210× → 1.041×**) and **730 → 233.8**
> at `-p 2048` (**1.212× → 1.068×**); on Qwen3-0.6B Q8_0 **698 → 282.7 µs** at `-p 8`
> (**1.164× → 1.066×**) and **445 → 10.5 µs** at `-p 2048` (**1.078× → 1.002×**). Both the
> concat class (`Op::FusedQKV`, PR #854) and the mixed-quant class (`Op::QkvBiasRopeStore`,
> PR #855) now fuse, and in both the fused path is **bit-equal to the unfused chain** for
> f32, f16 *and* q8_0 (0 of 121 / 91 comparable dump files with a non-zero delta; 7B Q4_K_M
> 114/114 KV dumps identical). Getting there needed one extra fix the ticket did not
> predict: the fused epilogue's rope and the unfused chain's differed by exactly one
> **FMA-contraction** rounding (2⁻²⁴), so `common.h` now holds one `rope_rotate` using an
> explicit `fma` and every rope site calls it — which **re-rounds every Metal rope output by
> ≤ 1 ulp**, the only numeric change in the pair. The ticket's own premise — "LSU-bound, five
> 2-byte loads per lane" — is **refuted**: widening those loads bought 1–2 %, and 82 % of the
> gap was the lost QKV fusion. The mixed-quant port is worth **+0.31 %** (≈40 µs/token on the
> 7B), not the 270–600 µs the ticket projected.
> **Date**: 2026-11-24. **Box**: macbookpro (Apple M4 Pro 20-core, macOS 27.2.1, arm64) for
> every device number, plus dgxspark (aarch64, GB10 sm_121) for the CUDA-feature build and
> the CPU suites. **Base**: `b9ff998` (the two PRs are `130c4b3` + `2b7ad06`).

## 1. Where it stood: a fanout, not a kernel

Records [111](./111-c4-packed-decode-epilogue-and-two-cell.md) and
[112](./112-c4-packed-decode-four-cells-per-warp.md) closed the CUDA half of the packed
decode residual and handed the *Metal* half here as "1.20× / 1.10×, LSU-bound". That
attribution came from a measurement of the **read** kernel in isolation (five 2-byte loads
per lane against f16's one vectorized `half4`) — and it was the wrong unit of account.

The first thing this ticket did was price the whole step instead of the kernel. At `-p 8`
on the 0.5B, with a q8_0 cache, three arms interleaved:

| arm | µs/token | note |
|---|---:|---|
| f16, fused | 3 033 | the shipped f16 decode |
| f16, `MINFER_NO_FUSE_QKV=1` | 3 589 | *the same arithmetic, five more nodes per layer* |
| q8_0, packed | 3 712 | the arm under investigation |

Turn the fusion off in the **f16** arm and you pay 556 µs of the packed arm's 679 µs gap;
**82 % of it was never a kernel at all**, it was the QKV fusion a packed cache could not
have. A per-node sweep agreed (~3.9–4.6 µs per node, the store ≈2–3 µs/layer) and priced
the entire packed **read** path at ~1 % (≈123.5 µs) of the gap.

Two kernel attempts followed before the graph was touched, and both are on the record:

- **Wide load** (5 two-byte loads → 3): 1.213× → 1.199× and 1.118× → 1.089×. It also broke
  byte-identity (logits moved 0.047 with identical geometry, and a same-binary rerun moved
  0.000), i.e. the compiler's FMA contraction had been part of the shipped bits. Reverted.
- **`simdgroup` store**: byte-identical, `-p 8` gap 634 → 566 µs (11 %, inside the cell's
  ~10 % spread), hd 128 −2 %. Under the ≥ 30 % keep rule: reverted.

## 2. Name the capability, then flip it in the change that earns it

The root cause was one line in two builders: `qkv_epilogue_ok` was
`#[cfg(feature = "cuda")] CudaState::get().is_some()` (qwen2) and `fused_norm_packed_ok` was
`device == Cuda` (qwen3) — a **build/device test standing in for a capability**. A packed
cache therefore could not fuse on Metal, by construction, on every backend regardless of
what the kernels could do.

[ADR-0084](../adr/0084-a-backend-declares-whether-it-can-store-a-packed-cell-from-the-fused-epilogue.md)
separates the question: `BackendCaps::writes_packed_kv_from_fused_epilogue`, answered by one
registry free function, CUDA `true` / CPU `false` by design / Metal `false` until its kernel
landed (PR #853; the flip arrived with the kernel in #854, as the ADR's decision 6 requires).
The flip exposed a second, real bug: with the capability `true`, Qwen2's mixed-quant layers
took the class-2 branch and emitted CUDA's `Op::QkvBiasRopeStore` on Metal, which the
assignment pass then placed on the CPU — colliding with the device-owned KV region and
refusing the graph (reproducible on **f16**, so a regression of the flip and not of the
packed arm). `registry::supports_op` exists for that: a forwarder to each backend's own
answer, so the two questions stay separate and the builder asks the registry.

## 3. The packed store arm: one node, three dispatches

`kernel_attn_rope_store` / `kernel_attn_bias_rope_store` wrote f32/f16 only, because a f16
or f32 store is element-addressed (`kv[pos * nkt + j]`) while a packed cell is
**byte-addressed** (34-byte blocks, an `f16` scale each). With `kv_packed` set the epilogue
now resolves the rotated K row and the biased V row **into the concat buffer** and writes
nothing element-wise; the graph then dispatches `kernel_store_kv_q8_0` — *the unfused
chain's own quantizer* — over the K and V sections with the cell's `row_bytes`. One node,
three dispatches. Consequence, recorded because it is the reason the arm is not a new
quantizer: `metal_packed_store_is_byte_identical_to_the_cpu_quantizer` passes by
construction.

## 4. The 2⁻²⁴ that made "bit-identical" false

With the fusion live, the q8_0 dump differential still showed 56 differing files. Two
candidate causes: the **concat matmul** (`od_total` rows in one launch vs three launches of
`od` rows) or the **epilogue** (bias+rope in one kernel vs `add_bias×3` + `rope×2`). A
temporary probe on the real decode shape (`id` 896, `nqt` 896, `nkt` 128, `hd` 64, Q4_0,
`nt` 1) settled it in two numbers, counting differing f32 words by bit pattern:

| probe | before | after |
|---|---|---|
| concat matmul (`od` 1152) vs three per-section (+ weight offsets) | **0 / 1152** | 0 / 1152 |
| fused bias+rope vs `add_bias×3` + `rope×2` | **103 / 1152**, max\|Δ\| **5.96e-8 = 2⁻²⁴** | **0 / 1152** |
| … of which the bias-only V section | 0 / 128 | 0 / 128 |

So the concat matmul was already bit-identical — rows reduce per simdgroup and only the
output stride depends on `od` — and the divergence was **one rounding unit, in the rotated
q/k words only**: `x0 * cs - x1 * sn` was contracted into an FMA in one kernel and left as
two roundings in the other. The angle was identical (`rope_freq` is shared), which is exactly
why the error was that small.

The fix is one implementation, not one flag: `common.h` gained

```c
inline RopePair rope_rotate(float x0, float x1, float cs, float sn) {
    RopePair r;
    r.r0 = fma(x0, cs, -(x1 * sn));
    r.r1 = fma(x0, sn,   x1 * cs);
    return r;
}
```

and the three rope sites — `kernel_rope_f32`, `kernel_attn_bias_rope_store`,
`kernel_attn_rope_store` — all call it. An explicit `fma` makes the evaluation order
independent of the compiler's contraction choice, which is the only way to *prove* the two
paths agree rather than observe it.

**The cost, stated plainly**: every Metal rope output re-rounds by ≤ 1 ulp relative to the
previous binary. That is the price of bit-equality between the two chains, and it applies to
f16 and f32 too.

Measured after the fix: fused vs `MINFER_NO_FUSE_QKV=1`, criterion "differs with a non-zero
delta", 121 files (0.5B) / 91 (0.6B) —

| model | f32 | f16 | q8_0 |
|---|---|---|---|
| Qwen2.5-0.5B q4_0 | 0 | 0 | 0 |
| Qwen3-0.6B Q8_0 | 0 | 0 | 0 |

(The `nodeN_*` debug dumps are named by graph-node index and the two topologies differ — 7B
decode 414 vs 512 nodes — so they are not comparable; the `logits_*` and `kv*` files, which
are what the model consumes, are all identical.)

## 5. The mixed-quant class: the same kernel, once per section

A mixed-quant layer (Qwen2.5-7B Q4_K_M carries `attn_v` as **Q6_K** against Q4_K q/k) cannot
share a concat weight, so `Op::QkvBiasRopeStore` is its only fusion — and Metal refused it.
The kernel was already there; what it lacks is a contiguous `[q|k|v]` buffer, and a scratch
concat is not viable mid-split (a blit closes the compute pass, costing more than it saves).
So the same kernel runs **once per section** over each projection's own buffer, with the
`(nqt, nkt)` triple that leaves exactly one of its three arms reachable and indexes it from
zero:

| section | `nqt` | `nkt` | grid |
|---|---|---|---|
| Q | `nqt` | 0 | `nqt / 2` |
| K | 0 | `nkt` | `nkt / 2` |
| V | `-nkt` | `nkt` | `nkt` (`vi = nqt + nkt + j` collapses to `j = u`) |

7 nodes / 10 dispatches per mixed-quant layer become **1 node / 8 dispatches** (the two
packed stores remain). A new gate, `graph::metal_backend::tests::qkv_epilogue` (four cases:
f32, f16, packed, interleaved), compares the section dispatches against the concat dispatch
on bit patterns — and caught a real bug in the wrapper: `dispatch_1d` rounds up to whole
256-thread groups while the kernel's guard is `u < total`, and the K section's `total` is the
*whole concat's* (`nkt*3/2`), so a rounded tail ran the V arm past the end of the k buffer.
The wrapper now dispatches an exact thread count.

## 6. Measured

`bench -p <n> -n 64 -r 3 --n-ctx 4096`, 3 interleaved rounds, medians, `macbookpro`:

| model (`hd`) | `-p` | f16 µs/token | q8_0 | gap | ratio | before |
|---|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B q4_0 (64) | 8 | 3 020.4 | 3 145.2 | **124.8** | **1.041×** | 634 / 1.210× |
| | 2048 | 3 451.3 | 3 685.0 | **233.8** | **1.068×** | 730 / 1.212× |
| Qwen3-0.6B Q8_0 (128) | 8 | 4 255.3 | 4 538.0 | 282.7 | 1.066× | 698 / 1.164× |
| | 2048 | 5 633.8 | 5 644.3 | **10.5** | **1.002×** | 445 / 1.078× |

And the honest reading of the mixed-quant arm, 7B Q4_K_M at `-p 8`, 10 interleaved rounds:
f16 20 385 µs/token, q8_0 before 20 687 (+1.48 %), after 20 648 (+1.29 %) — paired median
**+0.31 %** (8/10 rounds positive, +0.29 % on an independently rebuilt binary). The
`MINFER_NO_FUSE_QKV=1` control prices the *whole* QKV fusion on that model at 1.2–1.6 %,
because a 7B decode is weight-bandwidth-bound at ~20 ms/token and the host encode is
~0.2 ms of it (`MINFER_OP_PROFILE=1` totals, not per-label counts). The support-matrix note
written on 2026-10-06 had predicted exactly that ~1 % ceiling; the port landed because the
ticket's premise needed testing and a bitwise-gated arm is cheap to keep — and the note was
rewritten with the measurement rather than left standing.

## 7. Verification

- Bit-equality gates: `metal_packed_store_is_byte_identical_to_the_cpu_quantizer`,
  `metal_packed_session_round_trips`, the four `qkv_epilogue` section-vs-concat cases, the
  registry capability matrices (`the_packed_kv_capability_is_the_registrys_answer`,
  `the_fused_epilogue_packed_kv_capability_is_the_registrys_answer`).
- `metal_q8_0_kv_answers_like_f32_on_a_real_model --ignored`: passes on the 0.5B
  (max\|Δlogit\| 2.113 / 0.516, bit-for-bit the same numbers on clean `master`), and it is a
  real end-to-end exercise of the class-2 arm because that model is mixed-quant. It **fails**
  on Qwen3-0.6B (16.887 / 9.506) identically on clean `master` — pre-existing, filed as
  [#856](https://github.com/yusiwen/minfer/issues/856).
- Suites: Metal serial `cargo test --release --lib -- --test-threads=1` **640 / 0 / 18**
  (637 baseline + the four new gates − ... the four are in that count); CUDA feature build
  and the CPU suite green on `dgxspark`; CI 9/9 on both PRs with zero code annotations.
- The pre-registered bar (`-p 8` q8_0 ≤ 3 300 µs, `-p 2048` gap ≤ 300 µs and ratio ≤ 1.09,
  hd 128 `-p 8` gap ≤ 350 µs, packed cells byte-identical, CUDA/f16/f32 frozen) is cleared
  on every rung: 3 145.2, 233.8 / 1.068, 282.7, byte-identical.

## 8. What this does not do

- **Prefill is untouched** (the fusion is `nt == 1`): the prefill residual is
  [#845](https://github.com/yusiwen/minfer/issues/845).
- **The ~1–7 % that survives is the packed read path**, not node count — the idle-lane lever
  is [#851](https://github.com/yusiwen/minfer/issues/851).
- **The mixed-quant arm is worth ~0.3 %**, and no measurement here supports a bigger claim.
- The class-2 path's `supports_op` row is `dtype == DType::F32`: a non-f32 activation type
  would be refused and the layer would keep the unfused chain, which no shipped model hits.

## 9. Lessons

1. **Price the step before you price the kernel.** Three kernel candidates were proposed
   from a kernel-level measurement; none mattered. The step-level A/B (`MINFER_NO_FUSE_QKV=1`
   on the *f16* arm) found 82 % of the gap in one run and cost nothing.
2. **A capability is a question, not a `cfg`.** `CudaState::get().is_some()` is a device
   test; the graph builder needed "can you store a packed cell from your fused epilogue".
   Naming it (ADR-0084) made the flip mechanical — and the flip immediately exposed the
   class-2 op landing on the CPU, which is what a capability test is *for*.
3. **One rounding unit is a finding, not noise.** 2⁻²⁴ in 10 % of the rotated words was
   enough to move a q8_0 quant byte and make the dump differential unrunnable. A probe on a
   synthetic shape localised it to one expression in two kernels; no amount of model-level
   diffing would have.
4. **Make the order explicit when two paths must agree.** `fma()` in both places is a proof;
   `#pragma`-free "same expression" is a hope. The cost is one shared helper and ≤ 1 ulp.
5. **A gate that can fail is worth its weight**: the section-vs-concat gate found the
   thread-count overrun before a model ever ran, and it is the reason the per-section
   dispatch is defensible at all.
