# Speculative Decoding (D5) — Plan

Status: **D5 CLOSED (2026-09-10) — the D5-1a gate failed by measurement**
(records: [step doc 80](./cuda_optimization_steps/80-d5-0-cost-model.md),
[step doc 81](./cuda_optimization_steps/81-d5-1a-verify-gate-measured.md)).
Doc 80's conditional go rested on one number: the nt=3 verify amortization
≥ 2.5×. Doc 81 measured it end-to-end with the `minfer specverify`
instrument: C_T(3)=106 ms → per-token amortization **0.52×** (needed ≤ 22.1
ms). The nt=2–8 batched path costs a flat ~35 ms per token (weights
re-streamed per row — no amortization anywhere; the real tile-regime step
sits at M≥16, unreachable for verify), so even a dispatch fix or batch
padding caps below break-even. Per the stop rule pre-registered in D5-0 and
§D5-1 below, the campaign stops after the primitive (the instrument); no
`Speculator` trait, KV rollback, or loop plumbing will be built. The rest of
this document is kept as the record of what was planned.

Postscript (2026-09-11): the underlying dispatch hole was **fixed separately**
in [step doc 82](./cuda_optimization_steps/82-small-m-multi-token-mmvq.md) —
multi-token MMVQ + token-looped legacy kernels restore the batching invariant
(7B nt=3 105.9 → 29.4 ms, 3.60×; marginal 34.4 → 4.3 ms/token) for the small-M
prefill tax and future multi-token features. (Correction, 2026-09-12: the "external reference 1.00×" cited that day was
a measurement artifact — llama-cli silently ignores `-md` without
`--spec-type draft-simple`. Corrected batteries measure 1.53–1.60× (7B) /
1.86–2.43× (14B) for llama.cpp's own speculative decoding, and minfer's
post-fix primitive implies ≈1.42× at d=2 on the 14B. The closure verdict is
under campaign review — see doc 81 §4.3.)

The reference study is [`LLAMA-CPP-SPECULATIVE-ANALYSIS.md`](./LLAMA-CPP-SPECULATIVE-ANALYSIS.md)
(llama.cpp `draft-simple`, source-verified: speculator framework §3, draft
model setup §4, drafting loop §5, verification §6, KV rollback §7, cost model
§9, minfer port notes §11).

## 1. Why now — the evidence

- **The kernel-side foundation is already paid for.** The D series quantified
  the decode GEMM regime: MMQ at M=1 collapses to a 0.14-wave launch (step doc
  08), while at nt=4 the BT-MMQ path is **2.7× cheaper per token** (D4-1). A
  verification round is exactly an `nt = d+1` batched decode — it lands in the
  cheap regime by construction.
- **The engine already has the three structural prerequisites**:
  1. **Params-only graph reuse** — a fixed draft length `d` makes the verify
     graph a single `GraphParams` identity, captured once and replayed.
  2. **KV positions are data, not structure** — rollback never touches graph
     topology; it only rewinds the write position.
  3. **Per-layer persistent KV regions** (`KvProvider`) — a `seq_rm`-style
     truncation is a region-level operation, no snapshot machinery (the
     `PART` fast path analog; analysis §7, §11#5).
- **The decode campaign closed with parity/ahead against llama.cpp** (7B tg128
  1.074×, 14B 1.018×; hub §1), so the next meaningful decode-side lever is
  wall-clock-per-token, not kernel micro-optimization.

## 2. Scope

### In scope (D5): `draft-simple` — the small draft model

One draft GGUF (e.g. Qwen2.5-0.5B-Instruct) drafting greedily for a target
(7B q4_k_m), verified by the target in one `nt = d+1` batched forward. This is
the only speculator type that requires **no new architecture support**: it runs
today on Qwen2/Qwen2.5/Qwen3 dense models with existing GGUFs.

### Deferred (not in D5)

| Type | Why deferred |
|---|---|
| `draft-mtp` (MTP heads) | MTP weights exist only in DeepSeek-V3 / Qwen3-Next-class GGUFs; minfer does not support those architectures (MoE + MLA is a prerequisite campaign of its own), and current Qwen3 dense GGUFs carry no MTP tensors. |
| `draft-eagle3` | Draft weights consume the target's hidden states — needs a target-side feature export seam; tractable after D5 but a separate session. |
| `draft-dflash` / `draft-dspark` | Same architecture-prerequisite class as MTP. |

### Stretch (decide after D5-0): `ngram-simple`

Prompt/recent-output n-gram lookup drafts for free — no model, no architecture
prerequisite, no VRAM. In llama.cpp's priority chain all n-gram impls run
BEFORE draft models (analysis §3.1: if they fill the draft, the draft model
never runs). Strong for summarization / code-editing / long-context-copy
workloads. Cheap to add once the `Speculator` chain exists; keep as a stretch
goal, not a gate.

## 3. Mechanism — one round

With draft length `d` and acceptance count `a` (`a ≤ d`):

```
draft phase     draft model: seed + d sequential nt==1 decode steps (latency-bound)
verification    target: ONE nt=d+1 batched forward (throughput-bound, BT-MMQ sweet spot)
accept/cut      sample each verify row with the user's sampler chain; keep while
                equal to the draft token; always emit ≥ 1 token (the last verify row)
rollback        drop KV rows beyond the accept point (positions-as-data; topology intact)
commit          append accepted tokens; next round seeds from the last verify row
```

Net cost model (analysis §9): one target batched decode of `d+1` rows yields
`1+a` tokens; the draft pays `≈ 2(1+d)` token-evaluations of a model that is
several times cheaper per token. The practical knob is `d`: beyond the position
where per-position acceptance collapses, drafted tokens only cost decode steps.

**Greedy equivalence**: with `temp=0` the spec path MUST produce a byte-identical
token stream to the non-spec path (both are argmax chains; verification replays
the same math). This is the primary correctness gate. For `temp>0`,
target-sampler-authoritative verification (analysis §6.3) preserves the target
distribution but not the stream — documented, tested only for distribution
shape.

## 4. Touch points (file-level)

| # | Location | Change |
|---|---|---|
| 1 | **new `src/spec/mod.rs`** | `Speculator` trait (ordered chain, llama-style: first impl producing a non-empty draft wins) + `DraftModel` impl + stats (accept %, per-position rates, tokens/round). The draft result carries optional per-token probabilities (`p_min` gate), NOT hardcoded greedy — leaves MTP/EAGLE3/n-gram a clean seat. |
| 2 | **`src/main.rs` / `src/conversation.rs`** | Largest change: the decode loop becomes a round loop (draft d steps → verify → accept/cut → rollback → commit). The single-token fast path stays for `spec = off`. |
| 3 | **`src/models/` + load path** | Second model instance: independent `GraphCache`, weights registry, KV regions. Hard gates: draft `n_ctx` ≥ target usage; vocab compatibility check over GGUF tokenizer tables (vocab type, BOS/EOS, ≤128 size delta, token-text equality; analysis §4.2). |
| 4 | **`src/graph/params.rs` + `graph/cuda_backend.rs`** | Verify batch needs logits on EVERY row: `n_out = d+1` instead of decode's `n_out = 1` — touches the lm_head tail-rows dispatch assumption. Fixed `d` ⇒ one verify graph captured once and replayed; `positions` input refilled host-side per round (existing capture-safe mechanism). |
| 5 | **`src/sampler.rs`** | `sample_and_accept_n` equivalent: run the user's chain per verify row, accept while equal to the draft token, always emit ≥ 1 token. |
| 6 | **`src/graph/alloc.rs` / backends** | KV rollback: per-layer region truncation by position (dense models — no snapshots). CUDA path: rollback only moves the write position; stale rows beyond it are never read. Dedicated dump test required. |
| 7 | **CLI / bench** | `--draft-model <gguf>`, `--draft-n <d>`, `--spec-stats`; `minfer bench` speculative mode + synthetic-acceptance rates (`--spec-synth-rates`, output invalid, throughput ceiling only; analysis §10). |
| 8 | `src/server/` | Multi-slot integration — postposed to D5-4. |

## 5. Phases

### D5-0 — baseline & cost model (go/no-go gate) — DONE 2026-09-10

Record: [step doc 80](./cuda_optimization_steps/80-d5-0-cost-model.md).
Measured (3× interleaved medians, tg128): 7B q4_k_m CUDA **54.3** tok/s;
0.5B q4_0 CUDA **342.2** / CPU **73.3**. Measured acceptance (llama.cpp
`speculative-simple`, greedy, prose + code, both draft quants): conditional
**p ≈ 0.68–0.70**. Break-even: p\* = 0.73 / 0.81 / 0.90 at d = 2/4/8 —
measured p is below the line at every d unless the verify batch earns the
BT-MMQ amortization. Verdict: **conditional go at d=2 only** — the gate is
now the single number `C_T(3)`: minfer's measured nt=3 verify-batch
amortization must be **≥ 2.5×** (anchor 2.7× at nt=4; interpolation 2.28× vs
tile-step 2.7× disagree). Projected at the anchor: 1.04–1.05×; ceiling ~1.2×.
CPU-draft cross-device: dead (1.35× — no break-even at any p, d). d ≥ 4: dead
(≥ 4.5× amortization required).

### D5-1 — engine primitives (no behavior change) — RE-ORDERED: gate number first

1. **D5-1a (gate measurement)**: the `n_out = d+1` verify-batch dispatch +
   a micro-bench of the target at nt ∈ {3, 5} — **measure the nt=3
   amortization before any other plumbing**. ≥ 2.5× → proceed to D5-1b;
   < 2.5× → STOP, document the negative, close the campaign after the
   primitive (the entire go/no-go hangs on this one number).
2. **D5-1b**: `Speculator` trait + chain, KV rollback primitive, batched
   sampler. Each lands with its own parity test; the default path
   (`spec = off`) is byte-identical to today.

### D5-2 — greedy closed loop

Minimal `draft-simple`: 0.5B draft, greedy drafting, d=4, 7B q4_k_m target.
Acceptance: greedy output token-identical to the non-spec path on the parity
prompts; acceptance-rate + tokens/round stats; same-window A/B tok/s at
d ∈ {2,4,8}.

### D5-3 — tuning

temp>0 target-authoritative verification; verify-graph capture/replay soak
(pool_gen stability across rounds); adaptive d (truncate at the per-position
acceptance collapse point, llama's `n_max` logic); (stretch) `ngram-simple` as
the chain's first impl.

### D5-4 — integration & records

CLI flags finalized, `minfer bench` speculative mode, server multi-slot,
per-phase step docs in `docs/cuda_optimization_steps/` (80+), hub §0/§1 rows.

## 6. Risks

| Risk | Mitigation |
|---|---|
| **Net win may be small** (draft's serial steps are latency on the same GPU) | D5-0 is the go/no-go gate; synthetic-acceptance ceiling measurement; CPU-draft/GPU-target cross-device variant as the fallback design. |
| `n_out = d+1` touches lm_head/FFN tail-rows dispatch | Parity tests at d ∈ {1,4,8}; the decode `n_out = 1` path must stay untouched when `spec = off`. |
| KV rollback correctness under CUDA graph replay | Rollback only rewinds positions (never topology); stale rows are dead by construction — dedicated graph-dump test at round boundaries. |
| Vocab/tokenizer mismatch between draft and target | Hard gate at load (analysis §4.2): same vocab type, BOS/EOS, size delta ≤ 128, token-text equality. |
| Draft model doubles memory | 0.5B q4_0 ≈ 0.5 GB on top of 7B — fine on GB10 (32 GB); document the footprint, opt-in flag only. |
| Metal backend parity | CUDA first (the decode campaign's home turf); Metal parity is a follow-up session, not a D5 gate. |

## 7. Acceptance gates (campaign rules apply unchanged)

- Greedy token-identity vs the non-spec path (the spec path is an optimization,
  never a behavior change, at temp=0);
- full suite green; interleaved same-window A/B medians;
- acceptance-rate and tokens/round instrumentation on every round;
- every phase documented per `cuda_optimization_steps/STYLE.md` (English,
  six-section structure, real numbers, code provenance rules).
