# Speculative Decoding (D5-R) — Plan

Status: **D5-R stages ①+② LANDED (2026-09-12, docs 83–84: 14B d=2 = 1.33×/1.59×, same-window vs llama 1.64×/2.08×) — stage ③ (ncu verify-marginal attribution) next.**
Speculative decoding reopened by decision after the doc 81 §4.3 errata
invalidated the original closure's external pillar and doc 82 restored the
batching invariant. The previous plan (closed 2026-09-10, "no loop plumbing
will be built") is retired; its errors are recorded in the Appendix.

The reference study is [`LLAMA-CPP-SPECULATIVE-ANALYSIS.md`](./LLAMA-CPP-SPECULATIVE-ANALYSIS.md)
(llama.cpp `draft-simple`, source-verified: speculator framework §3, draft
model setup §4, drafting loop §5, verification §6, KV rollback §7, cost model
§9, minfer port notes §11). Records: docs
[80](./cuda_optimization_steps/80-d5-0-cost-model.md),
[81](./cuda_optimization_steps/81-d5-1a-verify-gate-measured.md),
[82](./cuda_optimization_steps/82-small-m-multi-token-mmvq.md).

## 1. Why reopened — the corrected evidence

Every number below is measured (docs 80–82 + the doc 81 §4.3 corrected
batteries):

- **Doc 82 fixed the dispatch hole**: verify amortization at nt=3 went
  0.52× → **2.14×** (14B q4_k_m, C_T(3)=56.6 ms vs C_T(1)=39.0); the batched
  verify is now a weights-once round, not a per-row restream.
- **llama.cpp's own speculative decoding demonstrably wins on GB10**:
  **1.53–1.60× (7B) / 1.86–2.43× (14B)** with the *generic* Qwen2.5-0.5B q4_0
  draft (the original "1.00× external anchor" was a measurement artifact —
  see Appendix).
- **The engine-independent terms are known**: per-position greedy acceptance
  p ≈ 0.74 (14B pair) / 0.69 (7B), draft cost C_D ≈ 2.9 ms/token — identical
  for both engines, as is C_T(1) (minfer 39.0 vs llama 41.8 ms on the 14B).
- **Predicted minfer economics** (measured components, doc 81 §4.3 addendum):
  d=2 → **1.34×** on the 14B from plumbing alone today; d=8 → 1.09× until the
  verify marginal shrinks. The entire gap to llama is the batched-verify
  marginal (minfer 3.9/7.8 vs llama ~1.0–1.5 ms per extra token).
- **The original go/no-go bar was mis-derived** (doc 82 §5): it took the
  per-token marginal as ε. With the corrected cost model, break-even
  acceptance at d=2 with an efficient verify is p\* ≈ 0.18–0.35 — the measured
  p clears it comfortably.

## 2. Scope

### In scope (D5-R): `draft-simple`, greedy first

One draft GGUF (Qwen2.5-0.5B q4_0) drafting greedily for a target
(Qwen2.5-14B q4_k_m primary — the 7B is marginal at d=2 — verified in one
`nt = d+1` batched forward through the existing `forward_graph_cached`
primitive). d=2 is the gate configuration; d=8 the end state.

### Deferred (not in D5-R)

`draft-mtp` / `draft-eagle3` / `draft-dflash` (a trained draft is the
obvious follow-up lever — llama.cpp natively supports all three spec types —
but each needs target-side feature seams or new architectures); n-gram
stretch; temp>0 target-authoritative verification; server multi-slot.

## 3. Mechanism — one round

With draft length `d` and accept count `a` (greedy: keep while the target's
argmax equals the drafted token):

```
draft phase     draft model: d sequential nt==1 forwards (second GraphCache)
verification    target: ONE nt=d+1 forward, n_out=d+1 (rows: next_token +
                the d drafted candidates, positions contiguous)
accept/cut      greedy argmax per verify row; accept while equal to the
                drafted token; always emit ≥ 1 token (the deepest accepted
                row's logits = the bonus)
KV              NO rollback kernels: the target wrote d+1 KV rows in place;
                rejected slots are simply overwritten next round
                ("positions are data" — the graph design makes rollback a
                position-bookkeeping operation)
commit          append accepted tokens; next round seeds from the bonus
```

**Greedy equivalence** is the primary correctness gate, re-scoped by
measurement (doc 83 §3.4): every emitted token is the target sampler's own
decision on its verify row, but the verify graph (nt=d+1, Prefill kernels)
differs numerically from the nt=1 decode graph by ~0.01–0.05 logits, so
sub-margin tokens may flap — both streams are valid greedy chains of the
same model (AGENTS rule-9 class). Exact identity is gated on (a) the
accept-rule unit tests, (b) the d=0 fallback, (c) near-tie attribution —
not on bit-equality across kernel assignments.

## 4. Design — stage ① minimal loop

Deliberately smaller than the retired plan's machinery:

- **`src/spec.rs`** — one concrete `SpecEngine` (no `Speculator` trait chain;
  the chain seat opens in a later stage if a second speculator type lands):
  holds the draft model (second `load_model` + its own `GraphCache`), runs d
  draft forwards + 1 verify forward per round, returns 1..=d+1 tokens plus
  the last row's logits.
- **Vocab compatibility gate at load** (kept from the D5 analysis §4.2):
  same vocab type, BOS/EOS, size delta ≤ 128, token-text equality.
- **CLI**: `--spec-draft <model>` + `--spec-draft-n <d>` (default 2), round
  stats (rounds / drafted / accepted / tokens-per-round) to stderr. The
  `spec = off` path stays byte-identical to today.
- **Sampling**: greedy argmax over the returned n_out rows (stage ① only;
  temp>0 chains deferred). Logits readback is nt×n_vocab×4 B ≈ 1.8 MB/round
  at d=2 — acceptable for the loop; on-GPU argmax is a stage ④ candidate.

## 5. Stages & gates

| Stage | Work | Gate |
|---|---|---|
| ① d=2 loop (greedy) — **LANDED 2026-09-12, doc 83** | `src/spec.rs` + CLI wiring | **G1 (re-scoped by measurement)**: (a) accept-rule unit tests with synthetic logits; (b) d=0 fallback == non-spec path to one exact-tie flap (buffer-placement numerics, any draft quant); (c) self-draft divergences attributed to near-ties (first-flap margin 0.043). Batched-verify (nt=d+1 Prefill graph) vs nt=1 decode kernels differ ~0.01–0.05 logits — same rule-9 class; exact identity returns only with nt-invariant accumulation (stage ④ candidate). **G2**: 179 tests green; off-path untouched. **G3**: per-round stats on stderr |
| ② end-to-end battery — **LANDED 2026-09-12, doc 84** | same-window dual-engine protocol (3 reps × prose/code × 4 cells) | **1.33×/1.59× ≥ 1.2× PASS**; llama same-window 1.64×/2.08×; gap fully priced: verify row marginal 8.8 vs 2.5 ms/row → 1.62× recoverable |
| ③ verify-marginal attribution | ncu on the nt=3 and nt=9 rounds: nt=9 GEMM M-pad waste (doc 82 multi-MMVQ caps at nt≤8), dp4a utilization, attention query-tiling (KV read once per nt rows vs per row), logits/sampling | one session; a cost ledger with per-item ms |
| ④ kernel attack | per ③'s ledger: multi-MMVQ extended to nt=9–16 (16-lane accumulators) and/or small-M GEMM tiles; graph capture for the fixed verify shapes (kills the +1.2 ms eager round overhead) | marginal 7.8 → ≤2.5 ms/tok (14B), then → ~1.5 |
| ⑤ d=8 + tuning | re-measure the d=8 economics; adaptive d (truncate at acceptance collapse); stretch: ngram | d=8 ≥ 1.5× end-to-end (14B) |

Each stage records per `cuda_optimization_steps/STYLE.md` (English, six
sections, real numbers) — docs 83+.

## 6. Risks

| Risk | Mitigation |
|---|---|
| The verify marginal may not reach llama's ~1.5 ms/token (their small-M MMQ tuning is multi-campaign depth, MMQ-analysis §7–12) | The ladder is staged: d=2 pays from plumbing alone (1.34×); ③ prices each candidate before any kernel work |
| Acceptance p is prompt-dependent (code 3.1×, prose 1.7× in the battery) | Report per-prompt, never averages alone; stats per round |
| `n_out = d+1` lm_head path | Already validated end-to-end by the specverify instrument (docs 81–82); n_out=1 decode path untouched when spec=off |
| Draft model doubles footprint | 0.5 GB on top of 9 GB — fine on GB10; opt-in flag only |
| Metal parity | CUDA first (campaign home turf); Metal is a follow-up, not a gate |

## 7. Acceptance gates (campaign rules apply unchanged)

Greedy token-identity vs the non-spec path (the spec path is an optimization,
never a behavior change, at temp=0); full suite green; interleaved same-window
A/B medians; acceptance-rate and tokens/round instrumentation on every round;
every stage documented per STYLE.md.

## Appendix — what the D5 closure got wrong (2026-09-10 → 09-12)

Kept brief; details in docs 80/81 (§4.3) / 82 (§5):

1. **The go/no-go bar was mis-derived.** The 2.5×-amortization gate treated
   the per-token marginal (attention + q8 quantize + dp4a compute) as ε;
   the correct model is C_T(nt) ≈ weights + nt·token-work, so break-even
   acceptance at d=2 is far lower than the p\* = 0.73 the old model printed.
2. **The external anchor was a measurement artifact.** The llama-cli
   batteries passed `-md` without `--spec-type draft-simple` (default
   `none`) — the draft was silently never loaded, so "llama.cpp also lands
   at 1.00×" was base-vs-base. Corrected: 1.53–2.43×.
3. **The instrument's summary field was wrong** (amortization computed as
   C_T(1)/C_T(nt), missing the nt factor; fixed 2026-09-12; doc tables were
   computed from raw medians and unaffected).

Net: the closure rested on a real 0.52× measurement but a miscalibrated bar
and a void anchor. Doc 82 fixed the underlying dispatch hole (0.52× → 2.14×),
the corrected batteries showed the strategy wins on GB10, and the campaign
reopens as D5-R.
