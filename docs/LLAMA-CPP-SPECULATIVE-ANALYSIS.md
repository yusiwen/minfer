# llama.cpp Small-Draft-Model Speculative Decoding — Source Analysis

Source basis: llama.cpp master @ commit `050dde50c` (checked out at `~/git/reading/llama.cpp`).
Scope: `COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE` ("draft-simple") — the classic **standalone small
draft model** speculative decoding — plus the orchestration framework it runs in. The framework
today also hosts EAGLE-3, MTP, DFlash/DSpark and four n-gram self-speculators; this doc covers
what they share (the lifecycle, verification and rollback machinery) and what is specific to the
small-draft-model path.

Upstream user-facing documentation: `docs/speculative.md` in the llama.cpp repo. This document is
the implementation-level companion: every claim below is anchored to a `file:line`.

---

## 1. TL;DR

- The draft model is **any small GGUF with a compatible vocab**; it runs in its own
  `llama_context` with the same `n_ctx` as the target and is decoded once per verification round.
- Drafting is **greedy argmax** over the draft logits, gated by an optional confidence floor
  (`p_min` on the top-k-renormalized probability) and capped by `n_max` (default 3).
- Verification **never injects a draft token**: the target model evaluates `[id_last, draft...]`
  in one batch and the *target's own sampler chain* decides token by token; a draft token is
  kept only when the target samples the same token. Output distribution is therefore exactly the
  target sampler's, and every round produces ≥ 1 token for free (the replacement/final sample).
- Rollback of rejected drafts is done by plain KV `seq_rm` when the memory supports partial
  removal (unified KV cache); otherwise the code snapshots state (`common_prompt_checkpoint`)
  and **replays** the accepted prefix as a "guaranteed draft" next round.
- The draft context stays synchronized with the accepted history by replaying every target
  batch through it without logits (`process()`), so the per-round draft cost is one seed decode
  + up to `n_max` sequential draft decodes + one replay batch.

---

## 2. File map

| Concern | Location |
|---|---|
| Speculator framework + all implementations | `common/speculative.cpp` (2980 ln) / `common/speculative.h` |
| Params structs (`common_params_speculative*`) | `common/common.h:171–395` |
| Acceptance sampling (`sample_and_accept_n`) | `common/sampling.cpp:678–715` |
| Canonical single-slot loop | `examples/speculative-simple/speculative-simple.cpp` (377 ln) |
| Multi-slot server integration | `tools/server/server-context.cpp` (drafting ~2965, accept ~3881) |
| Context capability probe | `common/common.cpp:1583` (`common_context_can_seq_rm`) |
| CLI/env options | `common/arg.cpp` (~4150–4330), `docs/speculative.md` |
| Checkpoint (state snapshot) | `common/common.h:1165` (`common_prompt_checkpoint`) |

## 3. The pluggable speculator framework

### 3.1 Types and priority chain

`common_speculative_type` (`common/common.h:171`): `none`, `draft-simple`, `draft-eagle3`,
`draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`,
`ngram-mod`, `ngram-cache` (11 total, `static_assert`ed at `speculative.cpp:2615`).

`--spec-type` accepts a comma-separated list. `common_speculative_init`
(`speculative.cpp:2602`) instantiates one impl per enabled type **in a fixed priority order**:
all n-gram impls first (they are free lookups — if they produce a draft, the draft model never
runs), then the draft-model impls (`draft-simple` first among them, `speculative.cpp:2619–2629`).

Chaining works through the per-sequence `drafting` flag
(`common_speculative_draft_params.drafting`, `speculative.h:58`): `common_speculative_draft`
(`speculative.cpp:2790`) walks the impl list; the first impl that fills `*dp.result` clears the
flag, so later impls skip that sequence; empty results fall through to the next impl.

> Note: `draft-simple` is *not* auto-selected. Passing `-md` with a plain small model keeps
> `types = {none}` → `common_speculative_init` returns `nullptr` → speculation silently off.
> You must pass `--spec-type draft-simple` (the GGUF-metadata auto-detection at
> `speculative.cpp:2284` only recognizes MTP / DFlash / DSpark drafts; see §4.3).

### 3.2 Impl interface

Base class `common_speculative_impl` (`speculative.cpp:138`):

```cpp
struct common_speculative_impl {
    const common_speculative_type type;
    uint32_t n_seq;
    int32_t  n_max;                    // effective max draft length of this impl
    // built-in accounting: n_call_begin/draft/accept, n_gen_drafts, n_acc_drafts,
    // n_gen_tokens, n_acc_tokens, n_acc_tokens_per_pos, t_begin/draft/accept_us
    virtual void begin(seq_id, const llama_tokens & prompt) = 0; // new-generation refresh
    virtual bool process(const llama_batch & batch) = 0;         // observe a target batch
    virtual void draft(common_speculative_draft_params_vec & dp) = 0;
    virtual void accept(seq_id, uint16_t n_accepted, bool is_other) = 0;
    virtual bool get_state(...) const; virtual void set_state(...);   // optional serialize
};
```

Public lifecycle (all null-safe no-ops when speculation is disabled):

| Step | API | draft-simple behavior |
|---|---|---|
| new generation | `begin(seq, prompt)` | no-op |
| every target decode | `process(batch)` | replay batch on draft ctx, `logits = nullptr` |
| per round | `draft()` | fill `*result` for flagged seqs (see §5.3) |
| after verification | `accept(seq, n_accepted)` | no-op (rollback is the caller's job) |

The per-seq draft request is a plain struct the **caller** fills
(`common_speculative_draft_params`, `speculative.h:53`):

```cpp
bool          drafting;   // ask this impl for a draft this round
int32_t       n_max;      // per-round cap (context/predict budget), -1 = impl default
llama_pos     n_past;     // position of id_last (seed)
llama_token   id_last;    // seed token
const llama_tokens * prompt;  // current sequence (input for n-gram impls)
llama_tokens * result;         // output; caller owns the vector
```

The outer `common_speculative` object (`speculative.cpp:2177`) is just
`dparams[n_seq] + impls[] + impl_last[seq]` (`impl_last` routes the later `accept()` call to
whichever impl actually produced the last draft, `speculative.cpp:2875`).

## 4. Draft model & context setup

### 4.1 Model/context creation

`common_speculative_init_from_params` → `common_speculative_init_result`
(`speculative.cpp:2515–2560`): loads the draft GGUF from
`params.speculative.draft.mparams.path` and creates a dedicated context. Context params
(`speculative.cpp:2532–2538`):

- `cparams.n_ctx = llama_n_ctx(ctx_tgt)` — the draft must hold the **same** sequence
  (prompt + all accepted tokens), not just `n_max` extra tokens;
- `cparams.n_rs_seq = 0` — the draft context never uses RS-bounded rollback;
- `cparams.ctx_other = ctx_tgt` — cross-context linkage.

Draft-side knobs are derived by `common_base_params_to_speculative`
(`speculative.cpp:2460`): model path / `-ngld` / `-devd` / tensor-buft overrides /
`-td` threads from the `common_params_speculative_draft` block, KV dtypes
`cache_type_k/v` default F16 (`-ctkd/-ctvd`), and `n_outputs_max = n_parallel`
(the draft never needs logits rows from `process()`; `draft()` requests logits per row it
decodes).

### 4.2 Vocab compatibility (hard gate)

`common_speculative_are_compatible` (`speculative.cpp:67–130`), checked in the impl
constructor and fatal on mismatch:

1. same vocab type (SPM/BPE/WPM/UGM);
2. same BOS add-flag **and** id; same EOS add-flag and id;
3. `|n_vocab_tgt − n_vocab_dft| ≤ 128` (`SPEC_VOCAB_MAX_SIZE_DIFFERENCE`, line 30);
4. token **text** byte-equality for every id from `5`
   (`SPEC_VOCAB_CHECK_START_TOKEN_ID`, line 31) through `min(n_vocab)` — catches renames
   even when sizes match.

The draft-simple constructor also asserts `n_seq == llama_n_seq_max(ctx_dft)`
(`speculative.cpp:247`) — the draft context must have one KV slot per target sequence.

### 4.3 Type auto-detection from GGUF metadata

`common_speculative_types_from_gguf` (`speculative.cpp:2284–2319`) reads only metadata:

- `general.architecture == "dflash"` → `draft-dflash`, or `draft-dspark` if a
  `markov_w1.weight` tensor (the Markov head) exists;
- otherwise, presence of `blk.{n_layer-1}.nextn.eh_proj.weight` → `draft-mtp`;
- anything else → `{}` (types stay `none` — see §3.1 note).

## 5. `draft-simple` internals (`speculative.cpp:179–389`)

### 5.1 Constructor: the draft sampler

One `common_sampler` per sequence (`speculative.cpp:226–236`):

```cpp
common_params_sampling params;
params.no_perf = false;
params.top_k   = 10;
params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };   // explicit chain: only top-k
smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
```

Although the explicit chain is only `TOP_K(10)`, `common_sampler_init` always appends a
`dist` sampler at the end (`common/sampling.cpp`, "default: sample from distribution"), so
`common_sampler_sample` computes softmax probabilities over the top-10 candidates. The draft
loop then **ignores the sampled token** and takes the argmax instead:

```cpp
common_sampler_sample(smpl, ctx_dft, i_batch, true);        // populates candidates
const auto * cur_p = common_sampler_get_candidates(smpl, true);  // sorted by p desc
const llama_token id = cur_p->data[0].id;                   // ← greedy argmax
if (cur_p->data[0].p < params.p_min) { /* stop this seq */ }
```

(`speculative.cpp:322–342`) Two consequences worth pinning down:

- **Drafting is deterministic given the logits** (argmax), independent of RNG seed. The
  `dist` sampler runs (consuming RNG, normalizing `p`) but its pick is discarded.
- `data[0].p` is the argmax's probability *renormalized within the top-10*, so `p_min`
  (default 0.0, `--draft-p-min`) means "the argmax must hold at least `p_min` of the top-10
  mass"; with the default it never truncates a draft early.

The own `llama_batch` is sized to the draft context's `n_batch` (`speculative.cpp:207`).

### 5.2 `process()`: free KV synchronization

```cpp
bool process(const llama_batch & batch) override {
    llama_batch batch_dft = batch;
    batch_dft.logits = nullptr;          // no output rows needed
    return llama_decode(ctx_dft, batch_dft) == 0;
}
```

(`speculative.cpp:262–277`) Every target decode batch — prompt prefill
(`speculative-simple.cpp:135`) and each verification batch
(`speculative-simple.cpp:237`) — is replayed verbatim (same tokens, positions, seq ids) on
the draft context. This keeps the draft KV cache identical to the target's accepted history
without any bookkeeping; the draft never re-prefills.

### 5.3 `draft()`: the greedy drafting loop

Request shape (caller side, `speculative-simple.cpp:188–196`): `{drafting=true, n_max,
n_past, id_last, &prompt_tgt, &draft}`. Implementation (`speculative.cpp:279–384`), with all
drafting sequences batched together:

```
for each seq with dp.drafting:
    common_sampler_reset(smpl[seq])
    batch += { id_last @ dp.n_past, logits = true }        // seed row
decode(batch)                                              // 1 batched decode

i = 0
while n_drafting > 0:
    clear(batch); i_batch = 0
    for each still-drafting seq:
        common_sampler_sample(smpl[seq], ctx_dft, i_batch++, true)
        cur_p = candidates (sorted)
        id = cur_p->data[0].id                             // greedy
        if cur_p->data[0].p < params.p_min: drop seq       // confidence floor
        common_sampler_accept(smpl[seq], id, true)
        result.push_back(id)
        if result.size() >= min(params.n_max, dp.n_max): drop seq
        batch += { id @ dp.n_past + i + 1, logits = true } // next input row
    if batch empty: break
    decode(batch); ++i                                     // evaluate new tokens

for each seq: if result.size() < params.n_min: result.clear()   // too short → no spec
```

Early-stop conditions: `p_min` floor, global `--draft-draft-n-max`/`--draft-max`
(`params.draft.n_max`, default 3), the per-round caller cap `dp.n_max` (context / n_predict
budget, §6.2), and decode failure. `n_min` (`--draft-min`, default 0) discards drafts
shorter than it entirely — the round then runs as plain decoding. `accept()` and `begin()`
are no-ops for this impl (`speculative.cpp:386–388, 258–260`).

### 5.4 vestigial knob

`p_split` ("speculative decoding split probability", default 0.1) is still parsed
(`arg.cpp:4209`) and documented, but **no code on this master reads it** — the old "draft only
with probability p_split" heuristic is gone; drafting happens every round. `backend_sampling`
(default on, `--spec-draft-backend-sampling`) lets the draft's sampling run on the backend
(`common_sampler_sample` short-circuits on `llama_get_sampled_token_ith`, `sampling.cpp:610`).

## 6. Verification: the target-side loop

### 6.1 One round of `examples/speculative-simple`

Invariants entering each round (`speculative-simple.cpp:141–152`):

- `prompt_tgt` holds the committed tokens at positions `[0, n_past)`; `prompt_tgt.size() == n_past`;
- `id_last` is the token at position `n_past`, **not yet evaluated** by either model
  (it was only *sampled* from the previous round's last logits row, or is the prompt's last token);
- both KV caches hold exactly `[0, n_past)`.

```
while true:
  if draft.empty():                                    // no replay pending
      ckpt.update_pos(n_tokens, tgt_pos_min, tgt_pos_max)
      ckpt.update_dft(ctx_dft)                         // only if ctx can't seq_rm partial
      n_draft_max = min(n_ctx - n_past - 2,            // leave room for id_last + shift
                        n_predict - n_predict - 1)     // generation budget
      draft_params = {drafting, n_draft_max, n_past, id_last, &prompt_tgt, &draft}
      common_speculative_draft(spec)
      if !draft.empty() && ctx can't seq_rm partial: ckpt.update_tgt(ctx_tgt)
      // roll the draft ctx back to [0, n_past): undo the draft() decodes
      ckpt.load_dft(ctx_dft) or seq_rm(ctx_dft, ckpt.pos_max + 1, -1)

  batch_tgt = { id_last @ n_past++ } + draft[i] @ n_past + i   // logits on every row
  llama_decode(ctx_tgt, batch_tgt)
  common_speculative_process(spec, batch_tgt)          // replay on draft ctx (no logits)

  ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft)   // §6.3
  if partial acceptance && ctx can't seq_rm partial:
      draft = move(ids); ckpt.load_tgt/dft; restore sampler clone; continue   // replay
  common_speculative_accept(spec, seq, ids.size() - 1)

  n_past += ids.size() - 1                             // committed accepted tokens
  for id in ids: prompt_tgt.push_back(id_last); id_last = id; print; EOG check
  draft.clear()
  seq_rm(ctx_tgt, n_past, -1); seq_rm(ctx_dft, n_past, -1)     // drop rejected tail
```

(`speculative-simple.cpp:162–342`)

Position/n_past bookkeeping: the seed sits at position `P` (`n_past++` post-increment), drafts
at `P+1+i`. After full acceptance of all `k` drafts, `n_past = P+k+1` and `seq_rm(n_past, -1)`
removes nothing; on partial acceptance of `a < k` drafts, it removes the rejected KV rows at
`P+a+1 … P+k` plus the replacement token's row at `P+a+1`, which is re-evaluated as the next
round's seed. Nothing is wasted: the replacement token *becomes* the next `id_last`.

### 6.2 Draft budget

`n_draft_max = n_ctx − n_past − 2` (`speculative-simple.cpp:181`, server equivalent
`server_slot::get_n_draft_max`, `server-context.cpp:483`): the `−2` reserves the seed row and
one slot for context shift. It is additionally clamped by the remaining `n_predict` budget and
by `dp.n_max` inside `draft()`. The target must be able to emit logits for `1 + n_draft` rows
per sequence — `common_speculative_get_output_limits(n_batch, n_parallel, n_draft)`
(`speculative.cpp:2589`) computes `{total = min(n_batch, n_parallel·(1+n_draft)),
per_seq = min(n_batch, 1+n_draft)}` and feeds `n_outputs_max` / `n_outputs_max_per_seq`
(`include/llama.h:364–366`).

### 6.3 Acceptance algorithm

`common_sampler_sample_and_accept_n` (`common/sampling.cpp:678–706`):

```cpp
for (i = 0; i < draft.size(); i++) {
    id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);  // target chain
    common_sampler_accept(gsmpl, id, true);                          // target state advances
    result.push_back(id);
    if (draft[i] != id) break;                                       // mismatch → replace
}
if (i == draft.size()) { id = sample(row idxs[i]); accept; push; }   // bonus token
```

Semantics:

- Row `i` of the target verification batch holds the logits *after* consuming the token at
  position `P+i`; it predicts the token at `P+i+1` — the same slot draft[i] proposes. The
  comparison is therefore apples-to-apples.
- The pushed token is **always the target's own sample**. Draft tokens are only ever *checked*,
  never injected. The output distribution equals the target sampler chain exactly — for any
  temperature/grammar/penalty configuration, no distribution correction is needed.
- The first mismatch replaces the rejected draft token with the target's sample and stops, so
  `result.size() − 1` = accepted draft count and `result.size() ≥ 1` always: every round emits
  at least the seed token even when the draft is fully rejected (`GGML_ASSERT` at
  `speculative-simple.cpp:262`).
- With a greedy target chain this is exactly classical speculative rejection sampling. With a
  sampled chain it is *more conservative* than the theoretical p_d/p_t accept test (a draft
  token is kept only when the target sampler draws the identical token), trading a little
  acceptance rate for exactness and simplicity — a deliberate llama.cpp design choice; the
  doc comment at `speculative-simple.cpp:251–257` spells out the "the sampler would have to
  sample that same token" contract.

Grammar interaction: `grammar_first=true` is passed for the draft side only; on the target
side the user's grammar applies through the normal chain (grammar-based resampling inside
`common_sampler_sample`, `sampling.cpp:646–675`).

## 7. KV rollback: seq_rm vs checkpoints vs replay

Whether partial acceptance needs special handling depends on the memory module, probed **at
runtime** by `common_context_can_seq_rm` (`common/common.cpp:1583–1627`): decode 2 tokens,
then attempt `seq_rm(mem, 0, 1, -1)`:

| Probe result | Memory | Rollback strategy |
|---|---|---|
| `PART` (removal ok) | unified KV cache | plain `seq_rm` of the rejected tail — no snapshots |
| `RS` (`n_rs_seq > 0`) | recurrent w/ snapshots | `seq_rm` bounded by `n_rs_seq` (`common.h:992`); checkpoint only when a rollback exceeds it |
| `FULL` (removal fails) | e.g. pure recurrent | checkpoint save/restore + replay |

`common_prompt_checkpoint` (`common/common.h:1165`) holds `{n_tokens, pos_min, pos_max,
data_tgt, data_dft, data_spec}` and save/loads both contexts' per-seq state with
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (`include/llama.h:912`). `data_spec` additionally
stashes impl-internal state (used by EAGLE-3's deferred boundary, not by draft-simple).

The **replay trick** (`speculative-simple.cpp:267–290`, server: `slot.spec_is_replay`,
`server-context.cpp:3922`): on partial acceptance with a non-removable memory, the context is
restored to the pre-round checkpoint and `draft ← ids` — the already-target-approved tokens
are re-submitted *as the next draft*. The next verification round accepts them wholesale
(the target sampler is restored from a clone, `smpl_save`, so the draws repeat), effectively
batch-committing the accepted prefix through the normal machinery.

The draft context is always rolled back to `[0, n_past)` before the next `draft()` call
(`speculative-simple.cpp:206–213`), because `draft()` re-seeds by decoding `id_last` and the
`process()` replay of the verification batch would otherwise double-insert those rows. The
server flags the resulting re-evaluation as a known optimization target
(`TAG_SPEC_AVOID_DRAFT_REEVAL`, `server-context.cpp:3041`).

## 8. Server integration (multi-slot)

`tools/server/server-context.cpp` runs one `common_speculative` for all slots with
`n_seq = n_parallel`; per-slot state lives on the slot (`spec_draft`, `spec_i_batch`,
`spec_ckpt`, `spec_prompt`, `spec_is_replay`, `spec_synth_rng`).

- **Drafting** (`update_slots`, `server-context.cpp:2965–3032`): every `SLOT_STATE_GENERATING`
  slot that can batch together gets its draft params filled (`id_last = slot.sampled`, the
  last accepted token); all drafts are produced by a single `common_speculative_draft` call
  inside `queue_tasks.yield_to_queue(...)` — the draft() loop's per-seq batching (§5.3) turns
  this into one decode per step across slots. `n_draft_max` per slot from
  `get_n_draft_max()` (`server-context.cpp:483`).
- **Checkpoints** are taken per slot after drafting when the target/draft context requires it
  (`server-context.cpp:3034–3080`), including the RS-bounded rule
  (`draft.size() > llama_n_rs_seq(ctx)` → checkpoint, lines 3055–3060).
- **Acceptance** (`server-context.cpp:3881–3951`): same `sample_and_accept_n` with the slot's
  sampler (`spec_i_batch` records which rows of the chunked target batch belong to this slot's
  verification); rollback identical to §7; `spec_is_replay` marks replay rounds and adjusts
  statistics so replayed tokens are not double counted (`server-context.cpp:3956–3958`).
- Accepted tokens are committed to the slot prompt, streamed (`process_token`), and both
  memories are truncated from `pos_next()` (`server-context.cpp:3976–3982`).
- Speculative state travels with server-side state checkpoints:
  `common_speculative_get_state/set_state` stash `data_spec` with the slot's checkpoint
  (`server-context.cpp:2349–2350, 3355–3356`).

## 9. Cost model — when does it win?

Per round with draft length `d` and acceptance `a` (`a ≤ d`), measured in model
token-evaluations:

| Work | Target | Draft |
|---|---|---|
| draft phase | — | seed + `d` tokens, **d+1 sequential decode steps** (latency-bound) |
| verification | `1 + d` rows in **one batched decode** | `1 + d` tokens in one replay batch (throughput-bound, cheap) |
| rollback | O(rejected) cell removal or snapshot restore | same |

Net: each round costs the target one batched decode of `1+d` rows (≈ prefill-like efficiency)
and yields `1+a` tokens. The draft model pays `≈ 2(1+d)` token-evals (the factor 2 is the
re-evaluation described in §7 — flagged `TAG_SPEC_AVOID_DRAFT_REEVAL`). So the draft model
must be several times faster per token than the target for `draft-simple` to break even;
with GPU-offloaded target and CPU draft (or vice versa) the sequential draft decodes can hide
behind other work. Acceptance statistics (`common_speculative_print_stats`,
`speculative.cpp:2936–2980`) report mean accepted length `1 + n_acc_tokens/n_call_accept` and
per-position acceptance rates — the practical knob is `n_max`: beyond the position where the
per-position acceptance rate collapses, drafted tokens only cost decode steps (also
benchmarked cheaply via synthetic acceptance, §10).

## 10. Observability & benchmarking hooks

- **Stats**: impl-level counters printed by `common_speculative_print_stats`
  (`speculative.cpp:2936–2980`); `speculative-simple` additionally prints
  `n_draft / n_predict / n_drafted / n_accept / accept%` (`speculative-simple.cpp:354–358`).
- **Synthetic acceptance** (benchmarking only, output is invalid):
  `--spec-synth-rates P0,P1,...` (unconditional per-position acceptance probabilities,
  must be finite, in [0,1], monotonically non-increasing — validated in
  `common_speculative_synth_rates_resolve`, `speculative.cpp:2379–2411`) or
  `--spec-synth-len L` (binary-searches a constant conditional `p` with
  `p + p² + … + p^k = L − 1`). The server then replaces real verification with
  `server_sample_and_accept_synth` (`server-context.cpp:57–100`), which draws
  `u ~ U(0,1)` per position against `synth_probs[i]`, never accepts drafted EOG tokens, and
  keeps grammar/reasoning state consistent by not advancing it on synthetic tokens.

## 11. Notes for minfer

What a minfer port of draft-simple would need, mapped to the current graph architecture:

1. **Two graphs, two contexts.** Target and draft are separate models → separate
   `ComputeGraph`s / `GraphCache`s (each with its own `GraphParams` identity). The draft
   context's `n_ctx` must match the target's (the KV regions live in the allocator, so both
   allocators must reserve the same prompt span; `n_rs_seq = 0` analog: no snapshot path).
2. **Vocab gate** (§4.2) is cheap metadata work over the GGUF tokenizer tables — same checks
   (vocab type, BOS/EOS, ≤128 size delta, token text equality from id 5).
3. **`process()` = replay batch without logits.** In minfer terms: run the draft graph with
   the same `positions`/tokens inputs but mark the logits node dead (no host copy). The
   minfer rule "KV positions are data, not structure" is exactly what makes the replay
   possible: the same prefill-shaped graph serves both target and draft.
4. **`draft()` = decode loop with `nt==1` graphs** on the draft model, greedy = argmax over
   the logits node; `p_min` needs the top-k renormalized probability (minfer's
   `sampler.rs` top-k path can provide it).
5. **Rollback**: minfer's per-layer persistent KV regions make `seq_rm`-style truncation a
   per-layer region rewrite (positions ≥ n_past dropped) — no snapshot machinery needed for
   dense models, matching the `PART` fast path.
6. **Verification** is pure sampler work (`sample_and_accept_n` semantics: sample each row
   with the user chain, keep while equal, always emit ≥ 1 token) and slots naturally onto
   minfer's `sampler.rs`.

## 12. Source index (quick reference)

| Symbol | Location |
|---|---|
| `common_speculative_type` enum | `common/common.h:171–183` |
| `common_params_speculative{,_draft,_ngram_*}` | `common/common.h:324–395` |
| `common_speculative_draft_params` | `common/speculative.h:53–72` |
| vocab compatibility | `common/speculative.cpp:67–130` |
| `common_speculative_impl` base | `common/speculative.cpp:138–177` |
| `impl_draft_simple` (ctor/process/draft/accept) | `common/speculative.cpp:179–389` |
| GGUF type auto-detection | `common/speculative.cpp:2284–2319` |
| `common_speculative_n_max` | `common/speculative.cpp:2329–2377` |
| synthetic rates resolve | `common/speculative.cpp:2379–2504` |
| `common_base_params_to_speculative` | `common/speculative.cpp:2460–2503` |
| `common_speculative_init_result` (draft ctx) | `common/speculative.cpp:2506–2587` |
| `common_speculative_init` (dispatch/priority) | `common/speculative.cpp:2602–2745` |
| `common_speculative_draft` (chain) | `common/speculative.cpp:2790–2873` |
| `common_speculative_accept` | `common/speculative.cpp:2875–2909` |
| `common_speculative_print_stats` | `common/speculative.cpp:2936–2980` |
| output limits | `common/speculative.cpp:2589–2598` |
| `common_sampler_sample_and_accept_n` | `common/sampling.cpp:678–715` |
| `common_sampler_sample` (backend-sampling shortcut, chain apply) | `common/sampling.cpp:594–676` |
| `dist` sampler (softmax + selected) | `src/llama-sampler.cpp:1150+` |
| canonical loop | `examples/speculative-simple/speculative-simple.cpp:162–342` |
| `common_context_can_seq_rm` probe | `common/common.cpp:1583–1627` |
| `common_context_seq_rm_type` | `common/common.h:988–994` |
| `common_prompt_checkpoint` | `common/common.h:1165+` |
| server drafting / checkpoints | `tools/server/server-context.cpp:2965–3080` |
| server accept / replay | `tools/server/server-context.cpp:3881–4004` |
| `server_sample_and_accept_synth` | `tools/server/server-context.cpp:57–100` |
| slot draft budget | `tools/server/server-context.cpp:483–500` |
| CLI options (`--spec-*`) | `common/arg.cpp:4150–4330`, upstream `docs/speculative.md` |
