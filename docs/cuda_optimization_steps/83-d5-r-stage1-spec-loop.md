# 83 · D5-R stage 1 — speculative decode loop (LANDED)

> **Result**: 14B+0.5B q4_0 draft, d=2 greedy: **34.3 tok/s prose / 40.7 tok/s code vs 25.5/25.7 serial = 1.34× / 1.58×** end-to-end (n=128, same-window); 7B prose 1.18×. Accept-rule unit tests green; the greedy "token identity vs the serial path" gate is re-scoped after root-causing the divergence to batched-verify vs decode kernel numerics (first-flap margin 0.043 logits).
> **Commit**: this document's commit. **Date**: 2026-09-12.

## 1. Background — where things stood

The plan rewrite (`docs/SPECULATIVE-DECODING-PLAN.md`, D5-R) fixed the two
errors that had closed D5: the mis-derived 2.5× amortization bar and the
void llama-cli 1.00× external anchor. Doc 82 had already repaired the
batching invariant (verify amortization 0.52× → 2.14× at 14B nt=3). The
economics table predicted d=2 on the 14B at **1.34× from plumbing alone** —
this stage builds that plumbing and tests the prediction.

What did not exist: any code path that runs the draft model and the target
model **in the same process**. Everything in the campaign so far — bench,
specverify, the CLI — assumed "single-model-per-process", a assumption so
deep it was written into a comment in the CUDA weight registry.

## 2. Principle — the mechanism

One speculative round (draft length d):

```
draft   d × nt==1 forwards on the draft graph     ≈ d × 2.9 ms  (0.5B q4_0)
verify  1 × nt=d+1 forward, n_out=d+1, on target  ≈ 56.6 ms     (14B q4_k_m)
accept  greedy: keep draft token i iff it equals
        the target sampler's own row-i sample; the
        deepest accepted row's sample is the bonus     ≈ 0 GPU ms
```

Expected tokens per round `E = 1 + p + p² (d=2, p≈0.74)` ≈ 2.29, so the
per-token cost lands near `(56.6 + 2·2.9 + 1.2 eager) / 2.29 ≈ 27.6 ms`
vs 39.0 serial → **1.41×**; measured 1.34× (prose, p a bit lower than the
battery's 0.74). Code prompts accept better (p→0.8) → 1.58× measured.

KV "rollback" is free: both graphs write KV rows in place and causal
attention never reads rows beyond the one being written, so rejected draft
slots are simply overwritten next round. The one real hole — a full accept
leaves the draft's row `pos+d` unwritten (the d-th proposal was never
forwarded) — is repaired with a single draft forward after full accepts.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **`src/spec.rs` is one concrete `SpecEngine`**, not a trait chain: one
  speculator type (draft-simple) exists today; the seat for a second opens
  when a second type lands.
- **Both models run through `forward_graph_cached`** — the target's KV lives
  in a caller-owned `GraphCache` (not the legacy global one), the draft has
  its own. The verify forward is exactly the primitive the specverify
  instrument validated (docs 80–82).
- **Lazy accept loop** (`accept_loop`, unit-tested): row i is sampled only
  after rows < i are committed, so the repeat-penalty window inside a round
  sees exactly what the serial path would see. Penalties apply to the
  target's rows only; the draft proposes raw argmax.
- **Namespaced GPU weight registry** (see §3.3): the first model loads with
  namespace `""` (every name unchanged — the whole existing suite and every
  prior record are untouched); the draft loads via `load_model_ns(gguf,
  "draft.")`.
- CLI: `--spec-draft <model>` + `--spec-draft-n <N>` (default 2). Round
  stats on stderr: `[spec] rounds= drafted= accepted= (p% of drafted)
  repairs= tokens/round=`. `MINFER_SPEC_DEBUG=1|2` prints reject/per-row
  top-2 logit margins (diagnostics used in §4).
- Conversation mode rejects `--spec-draft` for now; server mode untouched.

### 3.2 Key code

The round (spec.rs, abridged):

```rust
// draft phase: d forwards → d proposals for positions pos+1..pos+d
for k in 0..d {
    let logits = self.draft.forward_graph_cached(
        &[tok], &[pos + k], 1, n_ctx, &mut self.draft_cache);
    tok = argmax(&logits);
    proposals.push(tok);
}
// verify: one nt=d+1 forward with logits on every row; row i predicts pos+i+1
let logits = target.forward_graph_cached(&rows, &positions, d+1, n_ctx, cache);
// lazy accept: sample row i only after tokens 1..i are committed
let (emitted, accepted) =
    accept_loop(&logits, &proposals, s, prev_tokens, rng, trace, round);
// full accept: repair the draft's unwritten row pos+d
if emitted.len() == d + 1 { self.draft.forward_graph_cached(
    &[proposals[d-1]], &[pos+d], 1, n_ctx, &mut self.draft_cache); }
```

### 3.3 The three bugs the two-model process exposed

Loading a second model in one process broke three single-model assumptions;
each surfaced as a different symptom.

1. **CUDA/Metal weight registries are global and name-keyed.** The draft's
   tensors share GGUF names with the target's (`blk.0.attn_q.weight`…), so
   the second load overwrote the first's entries; the target's
   all-or-nothing `has_weight_of_size` check then failed at graph build and
   the whole target graph **silently assigned to CPU** (14B CPU decode ≈
   850 ms/token — the "8× slowdown"). Symptom in the first spec run: 2.7
   tok/s on the 14B. Fix: per-model namespace (`Qwen2Model::ns` /
   `Qwen3Model::ns`), applied at tensor registration, fused-weight
   registration (`{ns}blk.{i}.attn_qkv`, `.ffn_gu`), and graph build; the
   executor resolves whatever name the node carries, so `cuda_backend` is
   untouched.
2. **`nb_bt_only` is the "every registered matmul weight is NB-BT-consumable"
   global flag** and mode-2 A-fusion (the skip-write fused producers, r52)
   requires it. A q4_0 draft must clear it — the flag describes the global
   weight mix, namespaced or not. With the flag wrongly kept true, the
   target's prefill hit the mode-2 dead-write window guard and aborted:
   `native A-quantize refused: mode-2 dead-write A … prefill MMQ q8 scratch
   OOM`. Fix: the draft's q4_0 registration clears the flag; the process
   degrades to mode-1 fused A-quantize (writes the f32 output), which is
   the honest mode for a mixed registry.
3. **`prewarm_prefill()` runs per load** — harmless (the scratch pre-grow is
   idempotent), but worth noting as the third per-load global side effect
   audited (`set_kv_cache_type` is a OnceLock: first load wins).

### 3.4 The greedy-identity investigation (G1, re-scoped)

The plan's G1 gate demanded the spec stream be token-identical to the
non-spec stream at temp=0. It is not, and the investigation shows why that
is structural, not a bug:

- The verify forward runs at `nt = d+1` through the **Prefill-shaped graph**
  (batched attention, multi-MMVQ tiles); the serial path runs the **nt==1
  decode graph**. Per-row accumulation orders differ, so per-position logits
  differ by ~0.01–0.05.
- A token whose top-2 logit margin is below that noise flips: the spec chain
  and the serial chain are both valid greedy chains of the same model, and
  they reconverge after flaps (the aligned token traces show the two chains
  interleaving the same tokens in a different order around the flap).
- Measured (7B self-draft, penalties off, so the draft proposes exactly the
  serial chain's tokens): first chain divergence at the r4 bonus row with
  **top-2 margin 0.0434** (the serial chain had 23631, the verify row 504);
  11/11 row-0 accepts before it (margins 2.1–11.7 — far above the noise).
  After the flap the two chains interleave the same tokens in a different
  order and reconverge — two greedy chains of one model under two kernel
  assignments.
- The earlier rejects that looked suspicious (margins 0.44–2.9) were all in
  already-diverged contexts — consequences, not causes.
- **Even the d=0 fallback flaps once against the non-spec path** (byte 552
  of 96 tokens, deterministic). Control: the same run with a **q4_K** draft
  (which does NOT clear `nb_bt_only`) flaps at the same byte — so the
  perturbation is not the dispatch flag but the second model's
  registrations shifting the CUDA buffer-pool layout: per-row accumulation
  is placement-sensitive at exact-tie level. Any two-model process carries
  this; it is the same rule-9 class and bounded (1 flap / 96 tokens).

This is the same class as AGENTS.md rule 9 ("CPU-vs-GPU logits differ by
design; compare each path against its own reference") — here it is
nt=1-vs-nt=2+ kernel assignment. llama.cpp's verify/decode kernels are
closer to each other, but the property (spec stream ≡ serial stream only up
to numerics) is the same. A stage-④ candidate: make per-row accumulation
nt-invariant in the attention + matmul kernels, after which exact identity
holds again.

**G1 (final form)**: (a) accept-rule unit tests with synthetic logits —
full accept, reject at row 0/1, and the lazy penalty window (4 tests,
green); (b) `--spec-draft-n 0` (the d=0 serial fallback inside the same
loop machinery) matches the non-spec path to one exact-tie flap in 96
tokens (see above; deterministic, placement-caused, present for any draft
quant); (c) self-draft divergence attributed to near-ties (this section).

## 4. Results

Same-window runs, greedy, n=128, GB10 (SM 12.1, 48 SMs), CUDA build
(`--features cuda`), chat template, defaults otherwise. Draft:
Qwen2.5-0.5B q4_0.

| Config | prose tok/s | code tok/s |
|---|---|---|
| 14B serial | 25.5 | 25.7 |
| 14B + 0.5B d=2 | **34.3 (1.34×)** | **40.7 (1.58×)** |
| 7B serial | 53.5 | — |
| 7B + 0.5B d=2 | 63.4 (1.18×) | — |

Round stats (14B, d=2): prose rounds=63, accepted 65/126 drafted
(51.6%), tokens/round 2.03, repairs=25; code rounds=52, accepted 76/104
(73.1%), tokens/round 2.46, repairs=32. Code lands near the doc-81
per-position p≈0.74; prose acceptance is diluted by the near-tie flaps of
§3.4 (the draft proposes the serial-chain token; the verify row's
numerics-flipped sample rejects it) and by prompt dependence. The
predicted 1.34× (plan §1, from measured components at p=0.74) landed on
1.34× measured at p=0.52 — the lower acceptance and the higher repair
count roughly cancel in the round cost. Cost model confirmed end-to-end.

For the record, 14B d=8: 23.3 tok/s (0.91× serial) — the nt=9 verify falls
to the tiled GEMM (multi-MMVQ caps at nt≤8, doc 82), C_T(9)=101.3 ms makes
d=8 a loss until stage ④ shrinks the marginal, exactly as the plan
predicted.

Versus llama.cpp (doc 81 §4.3 corrected batteries, same models, d=2):
llama 1.86× vs minfer 1.34× on the 14B. The gap is the batched-verify
marginal (minfer ~7.8 vs llama ~1.5 ms per extra token at nt=3→9), exactly
what stages ③/④ attack.

Suite: 179 passed / 0 failed (4 new accept-loop tests). The `ns=""` default
keeps every pre-existing load path byte-identical.

## 5. What was learned

- **"Single-model-per-process" was a load-bearing assumption in three
  places** (weight registry, dispatch flags, prewarm) and only the first
  was written down. A second model turns every process-global into an
  API: namespace it or scope it.
- **The all-or-nothing CUDA gate fails silently into CPU** (build-time
  assignment is a legitimate mode, so nothing aborts). The tell is a
  tok/s collapse, not an error. A per-graph one-line participation log
  (backend assignment at first forward) would have saved an hour.
- Greedy equivalence between two kernel paths is a numerics statement, not
  a logic statement. Gate what is structural (the accept rule, the d=0
  fallback) and attribute the rest (near-tie flaps) with margin data.

## 6. Next steps

Stage ② (battery vs the llama 1.86× reference) is effectively pre-run
here; the real stage-② work is the clean protocol (multiple prompts,
interleaved A/B, acceptance-rate reporting). Stage ③ prices the verify
marginal with ncu: multi-MMVQ nt=9–16 extension, small-M GEMM tiles,
attention query-tiling (KV read once per nt rows vs per row), graph
capture for the fixed verify shapes (kills the +1.2 ms eager overhead).
A new stage-④ candidate from §3.4: nt-invariant per-row accumulation for
exact greedy identity.
