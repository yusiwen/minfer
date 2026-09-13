# 94 · Greedy identity achieved: spec-draft output is byte-identical to sequential decode; the d=8 door crosses with the q4_k_m draft

> **Result**: the nt-invariance campaign (doc 93 §3) closes with the strongest possible statement: with `--greedy`, `--spec-draft` output is now **byte-for-byte identical** to sequential decode on the whole battery (4 prompts, both cells, −n 128/200), and the suite stays 181 green. Two root causes were found and fixed, in this order: (1) the verify batch's attention ran a different kernel family than decode (per-token sequential loop vs split-K flash decoding) — fixed by batching the split path per token, bitwise-verified at both fixture and 14B dims; (2) — the one that actually broke e2e identity — the spec loop's repeat-penalty window grew unboundedly while every other caller trims to the last 64 tokens, so past the first >64-distant repeat the spec sampler penalized tokens sequential decode no longer penalized and the streams diverged. Cost of the guarantee: ≈ +1% on C_T(3) (48.1 → 48.6 ms), e2e within noise. And the payoff landed beyond correctness: with identity restored, the **d=8 door crosses** on code with the q4_k_m draft (76.5% ≥ the 0.755 threshold doc 92 computed) — d=8 = **46.6 tok/s** vs d=2's 43.6, 96% of llama.cpp's 48.4. Prose stays d=2 (d=8 collapses to 24.5).
> **Commits**: this document's commit. **Date**: 2026-09-12.

## 1. Localization chain (how the divergence was found)

Doc 93 left the e2e identity broken with the divergence source suspected in the batched verify attention. The localization ran in four steps:

1. **Kernel-level invariance tests** (new, permanent): `cuda_verify_attention_nt_invariance` — same KV buffer (prefix + intra-batch rows), batched (positions [P, P+1, P+2]) vs three nt=1 decode calls, outputs compared **bitwise**, over {f16-KV, f32-KV} × {parity fixture (nh=4/nk=2/hd=8), 14B dims (nh=40/nk=8/hd=128, prefix 512)}. `cuda_multi_token_matmul_bitwise` extended with the real 14B shapes (Q4_K 5120×5120, Q4_K 13824×5120, Q6_K 5120×13824) — single-MMVQ (nt=1) vs multi-MMVQ (nt 2..8) bitwise. Both passed → kernel numerics exonerated at the tested shapes.
2. **Position-wise logit dumps** (temporary `MINFER_IDENT` instrumentation, since removed): seq decode steps vs verify rows, top-2 raw logits aligned by predicted position. The first 70 aligned rows matched **to the printed 6 decimals with gaps of 4–18 logits** — no ULP near-tie anywhere → the divergence could not be kernel numerics at all.
3. **First divergence**: position 119 — the entire distribution differed (spec top-1 30.78 vs seq top-1 29.75) — a *state* difference, and the state difference was the *forward input*: spec forwarded token 3364@119 where seq forwarded 3840@119. The streams had diverged in the **sampler**, one step earlier.
4. **The sampler**: at the decisive pick the raw logits were identical (3840 @ 24.81 vs 3364 @ 24.45) but spec sampled 3364 — 3840 was repeat-penalized in the spec run and not in the seq run. Window diagnostics (`ptlen`, in-window count): seq `ptlen=64` (trimmed every push), spec `ptlen=119` (never trimmed). `apply_penalties` counts over the **whole** `prev_tokens` slice — the 64-window is the caller's contract. Spec was the only caller violating it (main decode loop, server, conversation all trim).

## 2. The fixes

1. **Batched verify attention (doc 93's campaign proper)**: `gqa_attn_split_partial_bt` / `gqa_attn_split_combine_bt` — the decode path's exact `attn_split_1w_body` and combine merge order, with `t = blockIdx.z`, per-token `nkv = positions[t]+1`, partials `[t][SPLITS][nh][pstr]`; grid static per captured graph (capture/replay-safe). Dispatch: `nt == 1` → the existing decode path untouched (bitwise net intact); `1 < nt ≤ 16` → the batched split path; `nt > 16` (prefill) → the incumbent per-token kernels. Scope: the rpw ≥ 16 hybrid (nkv ≥ 1921) is not batched, so the identity guarantee covers nkv < 1921 (the batteries run at 512–640).
2. **Penalty window cap (the e2e culprit)**: `push_capped` in `spec.rs` — every committed token is pushed with a trim to `sampler::REPEAT_LAST_N` (promoted to a shared const), the same window the sequential loop maintains. The d=0 fallback and the accept loop's three push sites all route through it.

Both fixes are semantics-preserving for decode (nt=1 paths untouched); the verify path's kernel change is the only numerics-affecting edit, and the invariance tests pin it bitwise.

## 3. Verification

- **Identity battery: 4/4 byte-identical** — prose + code prompts at −n 128 (810 B / 634 B), a plot-summary and a TCP/UDP prompt at −n 200 (1023 B / 1094 B); pre-fix these diverged within the first paragraph.
- **Suite 181 green** (incl. the two new/extended bitwise tests).
- **Perf gate**: C_T(1) 39.70 (was 39.4), C_T(3) 48.62 (was 48.1, +1.1%), C_T(5) 56.15 (was 55.8), pp512 2019.7 (was 1999.7) — the identity costs ≈ 1% on verify rounds (the batched split launches 32×40×nt incumbent-geometry blocks + combine vs 120 fat blocks; recovering it requires an nt-scaled split count, which would break bitwise equality with the decode path's SPLITS=32 chunking — accepted, documented).
- **Acceptance re-measure (post-fix, −n 128)**: q4_0 51.6%/74.0% (prose/code), q4_k_m 49.2%/76.5%, q5_k_m 52.4%/69.4% — code cells shifted +0.9–1.0 pt (sampler state now sequential-faithful), prose ±0.8 pt; tok/s within noise.

## 4. The d=8 door crosses — final recommendation matrix

| workload | draft | d | acceptance | tok·s⁻¹ | vs llama.cpp |
|---|---|---:|---|---:|---|
| prose | q4_0 | 2 | 51.6% | 35.9 | 95% (37.7) |
| code | **q4_k_m** | **8** | 76.5% | **46.6** | **96% (48.4)** |
| code | q4_0 | 2 | 74.0% | 42.5 | 88% |
| prose | q4_k_m | 8 | 16.9% (collapses) | 24.5 | — |

d=8 economics at code-p = 0.765: tokens/round 4.57, rounds 28 (vs 51 at d=2) — the C_T(9) = 73.0 auto-ksplit round from doc 92 is cheap enough that the deeper amortization wins. Prose acceptance collapses at depth (16.9% of drafted tokens accepted at d=8) — deep positions are near-ties there, so d=2 stays optimal. The engine default stays d=2/q4_0 (workload-agnostic); the recommendation is **`--spec-draft-n 8 --spec-draft <q4_k_m>` for code-shaped sessions**. The doc 93 quant-swap reading stands corrected in one respect: q4_k_m's value is not the ±1% at d=2 — it is that its acceptance clears the d=8 threshold at all.

## 5. State and open leads

The campaign chain 88→94: acceptance-rate framing → mma audit → K-split → auto-ksplit default → draft-quant sweep → greedy identity. Speculative decoding now carries a property few engines state: **exactly transparent to sequential greedy** (nkv < 1921). Open leads, re-priced: (a) conversation/server modes (engineering, no measurement risk); (b) the remaining ~4% to llama on code (verify-round cost: the tile-geometry redesign, worst EV per doc 92); (c) adaptive d (truncate at acceptance collapse — prose would pick d=2 automatically, code d=8; needs a per-round acceptance estimator); (d) the nt-scaled split count if the 1% identity cost ever matters (requires re-proving bitwise equality against a decode path that also changes).

## 6. Verification for this document

- Identity: the §3 battery (cmp of stats-stripped outputs, 4 prompts).
- Kernel bitwise: `cuda_verify_attention_nt_invariance`, `cuda_multi_token_matmul_bitwise` (extended).
- Perf: `specverify -p 512 -r 5 -o md` (C_T table), `bench -p 512 -n 8 -r 3` (pp512).
- d=8: the §4 table (`--spec-draft-n 8`, both drafts, code + prose).
