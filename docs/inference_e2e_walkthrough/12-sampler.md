# 12 · Sampler — from logits to one chosen token

> **Stage**: last-token logits out of the forward (docs 09–11) → **this
> stage** → one chosen token id, which doc 13 feeds back into the graph for
> the next step. This is the bridge between math and language: the model's
> entire output for one step is a list of 151,936 floating-point scores;
> the sampler turns that list into a single integer — the next piece of text.
> **Code**: `src/sampler.rs` (`apply_penalties` :47, `recent_window` :95,
> `match_stop_suffix` :107, `apply_top_k` :128, `apply_top_p` :152,
> `sample_temperature` :229, `sample_with_penalties` :283) and its call site,
> the decode loop in `src/main.rs` :832–940 (`GenParams` defaults :50–78,
> seeded RNG :845, sampling call :885–895, stop gates :900–918,
> `is_stop_token` :1020) — lines verified at commit `e7fa0da`.

## 1. Background — where this stage sits

Doc 11 ended with attention producing each layer's hidden states, and doc 09
ended with the final projection: for the **last** token of the sequence, the
engine computes one score per vocabulary entry. Those scores are called
**logits**. A *logit* is a raw, unnormalized score — a plain `f32` that says
"how much the model likes this token as the next one". Higher means more
likely, but the numbers are not probabilities: they can be negative, they do
not sum to anything in particular, and their absolute scale is arbitrary (it
depends on the model, the quantization, even the layer's final RMSNorm gain).
For the Qwen family the vocabulary has **151,936** entries, so what this
stage receives is a `Vec<f32>` with 151,936 elements — 151,936 × 4 bytes =
607,744 bytes ≈ **0.6 MB** of scores (the `607 KB` the code comment at
`src/main.rs:924-925` refers to).

The sampler's whole job is to turn that list into one `u32` token id. That
sounds trivially small next to the transformer's billions of multiplies, and
in CPU time it is — microseconds against milliseconds. But it is where the
model's character lives. The same weights produce a careful, repetitive
assistant or a creative, rambling one depending entirely on how this stage
picks. Every knob doc 01 collected — `--temp`, `--top-k`, `--top-p`,
`--repeat-penalty`, `--frequency-penalty`, `--presence-penalty`, `--seed`,
`--greedy` — is consumed here and nowhere else.

Three quiet facts shape the design:

1. **The model only ever proposes; the sampler disposes.** The forward pass
   computes a score for *every* token in the vocabulary, including absurd
   ones. Sampling decides which of those voices gets heard.
2. **Sampling needs memory of the past.** The repeat/frequency/presence
   penalties need to know which tokens appeared recently. That means the
   decode loop keeps a sliding window of the last 64 token ids — seeded from
   the prompt's tail — and passes it in alongside the logits
   (`src/main.rs:842-847`).
3. **"Stop generating" is a sampling-adjacent concern.** After a token is
   chosen, three gates decide whether generation ends: the token is the
   end-of-text sentinel (eos or `<|im_end|>`, doc 04), the generated *byte
   stream* now ends with a user-supplied stop string, or the `n_predict` cap
   is reached. The first two live in the same loop, a few lines below the
   sampling call, and one of them (`match_stop_suffix`) is implemented in
   `sampler.rs`.

What would break without this stage? Literally everything downstream: the
decode loop has nothing to append, the tokenizer has nothing to decode, and
the terminal stays silent. But the subtler failure is a *wrong* sampler: the
engine's claims of matching llama.cpp rest on producing the same output for
the same parameters, and the sampler is half of that contract (the other
half is bit-close kernels). That is why minfer copies llama.cpp's default
values, its penalty semantics, and its stage *order*, and why the whole
thing is seeded and reproducible by default.

## 2. Principle — how it works and why

### 2.1 The pipeline in one picture

The sampler is a short chain of filters followed by one random draw. Each
stage takes the logits buffer, modifies it in place, and hands the *same*
buffer to the next stage:

```
 raw logits   Vec<f32>, one entry per vocab token (151,936 for Qwen)
              ← the output of the last forward (docs 09–11)
     │
     ▼
 apply_penalties      repeat ×/÷ rule + frequency −count·f + presence −p
                      one pass over the last-64 token window
     │                (temp == 0? → greedy argmax here and skip the rest)
     ▼
 apply_top_k          keep the k=40 highest logits, mask the rest to −∞
     │
     ▼
 apply_top_p          nucleus: keep the smallest set of tokens whose softmax
     │                probability sums ≥ p (0.95); mask the rest to −∞
     ▼
 sample_temperature   logits × 1/temp (0.8) → softmax → multinomial draw
     │                ← one uniform random number from StdRng(seed = 42)
     ▼
 SampledToken { token_id: u32, logit: f32 }
     │
     ▼
 stop gates           is_stop_token(eos/<|im_end|>)?  match_stop_suffix()?
     │                generated.len() < n_predict?
     ▼
 decode_bytes → stdout  ·  doc 13 feeds token_id back into the graph
```

Everything before the last box is deterministic arithmetic on the logits.
The single random element is one uniform number per generated token. Fix
that number stream (the seed) and the entire run becomes reproducible.

### 2.2 From logits to a probability distribution: softmax

A **probability distribution** over the vocabulary is a list of numbers, one
per token, that are all ≥ 0 and sum to exactly 1 — think of a pie cut into
151,936 slices, one slice per token, sized by "how likely is this token
next". Raw logits are not that (they can be negative and don't sum to 1), so
the engine converts them with **softmax**, the standard score→probability
converter:

```text
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

In words: raise *e* (≈ 2.718, a mathematical constant) to the power of each
logit — this makes every score positive while *preserving order* (bigger
logit ⇒ bigger exp) — then divide each by the total, so everything sums to
1. The exponential's superpower is *contrast amplification*: a logit
difference of 2 becomes a probability ratio of e² ≈ 7.4. Small score gaps
turn into lopsided probabilities.

One implementation detail you will see in the code: before exponentiating,
every logit has the maximum subtracted (`exp(v - max)`). This changes
nothing mathematically — the max appears in numerator and denominator and
cancels — but it prevents overflow. `exp(30)` is already ~10¹³ and logits
can reach the tens; `exp(large)` in `f32` overflows to infinity. After the
shift the biggest exponent is exactly `exp(0) = 1`, which cannot overflow.
You will see this "subtract the max" pattern in every softmax in this
engine, including attention's (doc 11).

### 2.3 Why sample at all — and what "greedy" means

The softmax hands you a full probability distribution. The simplest policy
is: **always take the highest-probability token**. That is called **greedy
decoding** (or *argmax sampling* — "argmax" means "the index of the
maximum"). It is fully deterministic and it is the right default for
benchmarking and correctness gates — minfer's own `bench` uses it
(`src/bench.rs:24,384`), and the CUDA optimization campaign's second gate is
"greedy byte-for-byte identity" before/after a kernel change
(`docs/cuda_optimization_steps/77-verification-methodology.md` §2.2).

So why not always greedy? Because text generated purely by "what is most
likely next" has a failure mode every LLM user has seen: **loops**.
"The cat sat on the mat. The cat sat on the mat." A greedy decoder, once it
steers into a rut, has no way out — the most likely continuation of a
repetition is more repetition. **Multinomial sampling** is the alternative:
treat the probability distribution as a weighted lottery and draw one token
at random, with each token's win chance equal to its probability. Now a
0.7-likely token wins ~70% of the time — and, crucially, a 0.2-likely token
sometimes wins, which is exactly the escape hatch that breaks loops and
gives the model its range of phrasing. The trade is coherence for diversity;
**temperature** (§2.6) is the dial between the two, and the dial's zero stop
is greedy itself — `--greedy` in the CLI is literally `--temp 0`
(`src/main.rs:312-314`), and `sample_temperature` returns the argmax when
`temp < 1e-6` (`src/sampler.rs:230-232`). Greedy is not a different
algorithm here; it is a degenerate temperature.

minfer's default is llama.cpp's: **temp = 0.8** — mildly random. The rest of
the pipeline (penalties, top-k, top-p) exists to make that randomness
*tasteful*: fair over plausible candidates, rigged against degenerate ones.

### 2.4 Stage 1 — penalties: teaching the model not to repeat itself

The first filter looks at the last **64** generated-or-prompted tokens (the
*window*; llama.cpp calls the setting `repeat_last_n` and 64 is its default)
and pushes down the logits of anything that appears in that window. minfer
implements all three penalties in **one pass** (`apply_penalties`,
`src/sampler.rs:47-79`), matching llama.cpp's `llama_sampler_init_penalties`
semantics, which the doc comment spells out:

```text
for each distinct token t in the window, with count(t) occurrences:
    logits[t] -= count(t) · frequency_penalty     (if frequency ≠ 0)
    logits[t] -= presence_penalty                 (once, if count(t) > 0)
    logits[t] = logits[t] ≤ 0 ? logits[t] · repeat_penalty
                               : logits[t] / repeat_penalty
```

Three separate ideas share the pass:

- **Repeat penalty** (default **1.1**): a token seen in the window has its
  positive logit divided by 1.1 — or its negative logit multiplied by 1.1.
  Both branches make the token strictly less likely; §3.3 explains why one
  parameter covering both signs forces the ÷/× split.
- **Frequency penalty** (default **0**, used by the OpenAI-compatible
  server): subtract `count × 0.something` per occurrence — the more a token
  already appeared, the harder it is pushed.
- **Presence penalty** (default **0**): subtract a flat amount *once* if the
  token appeared at all — count-blind, it just discourages re-raising any
  recent topic.

A worked example. Suppose the window's last 64 tokens contain `"the"` three
times and `"a"` once, and the model's raw logits for five candidates are:

| token | raw logit | in window? | after repeat 1.1 |
|---|---|---|---|
| `the` | 8.5 | 3× | 8.5 / 1.1 = **7.727** |
| `cat` | 7.9 | no | **7.900** (untouched) |
| `a`   | 4.0 | 1× | 4.0 / 1.1 = **3.636** |
| `dog` | −2.0 | no | −2.0 (untouched) |
| `said`| −0.5 | no | −0.5 (untouched) |

Before the penalty, `the` (8.5) beats `cat` (7.9) and greedy would emit
`the` — possibly the third `the` in a row. After the penalty, `cat` wins
(7.900 > 7.727) and the loop is broken. Note what did *not* happen: `dog`
and `said`, which are not in the window, were not touched; the penalty never
invents new preferences, it only taxes recency. (If the window had contained
`dog`, its negative logit would become −2.0 × 1.1 = −2.2 — pushed *further
down*, not up; §3.3 covers why.)

With the OpenAI-style penalties on (`frequency = 0.5, presence = 0.3`), the
order inside one pass matters and the code applies subtraction *first*, then
the repeat rule — llama.cpp's order. For `the` (count 3):
8.5 − 3 × 0.5 − 0.3 = 6.7, then ÷ 1.1 = **6.09**. For `a` (count 1):
4.0 − 0 − 0.3 = 3.7, then ÷ 1.1 = **3.36**.

The counting itself is a `HashMap<u32, u32>` built by one loop over the
window, then one mutation per *distinct* token — at most 64 hash entries and
64 logit writes per generated token, i.e. microseconds against the
milliseconds the forward pass costs.

### 2.5 Stage 2 and 3 — top-k and top-p: pruning the long tail

After penalties, the distribution still spans the whole vocabulary. Most of
those 151,936 candidates are nonsense — misspellings, lone bytes from the
middle of a CJK character, whitespace runs. Individually each is unlikely,
but *collectively* the long tail is where sampling goes to produce garbage,
and two standard filters chop it.

**Top-k** (default **k = 40**) keeps only the 40 highest logits and masks
every other entry to −∞ (negative infinity — the float value that loses
every comparison and maps to probability 0 in softmax). After this, the
lottery has at most 40 tickets, and they are by construction the strongest
ones. That is the entire point: *kill the tail of absurd tokens in one
stroke*, regardless of how flat or peaked the distribution currently is.

**Top-p** (default **p = 0.95**), also called **nucleus sampling**, is
adaptive where top-k is fixed. It computes the softmax probabilities, sorts
them from most to least likely, and keeps the **smallest set of top tokens
whose probabilities sum to at least p** — the "nucleus" of the distribution.
On a confident step (one token at 0.98), the nucleus is that one token. On a
genuinely uncertain step (ten tokens at ~0.1 each), the nucleus stretches to
ten. The cutoff tracks the *shape* of the distribution instead of a fixed
count, which is why it complements top-k rather than replacing it.

A worked example, small enough to check by hand. Five tokens with softmax
probabilities 0.40, 0.30, 0.15, 0.10, 0.05, and p = 0.8:

| rank | token | probability | running sum | verdict |
|---|---|---|---|---|
| 1 | `A` | 0.40 | 0.40 | 0.40 ≤ 0.8 → keep going |
| 2 | `B` | 0.30 | 0.70 | 0.70 ≤ 0.8 → keep going |
| 3 | `C` | 0.15 | **0.85** | 0.85 > 0.8 → **stop**; C is kept |
| 4 | `D` | 0.10 | — | masked to −∞ |
| 5 | `E` | 0.05 | — | masked to −∞ |

The nucleus is {A, B, C}: the smallest prefix whose sum (0.85) reaches past
0.8. D and E — 15% of the probability mass between them — are now
unreachable. The token that *crosses* the threshold is kept, so the nucleus
is never smaller than the first token that gets you to p.

One subtlety of minfer's implementation matters enough that the code
documents it (`src/sampler.rs:142-145`): top-p does **not** rewrite the
kept entries with their probabilities. It computes probabilities only
*internally*, to decide who stays; then it masks the losers' **raw logits**
to −∞ and leaves the winners' raw logits untouched. Why that distinction is
load-bearing is §3.3's third design question — short version: the final
softmax has not happened yet, and it still needs the real logits.

### 2.6 Stage 4 — temperature, the final softmax, and the draw

Everything so far *selected* candidates. The last stage *weights* them and
picks one. Three sub-steps, all in `sample_temperature`
(`src/sampler.rs:229-278`):

1. **Divide every surviving logit by `temp`.** Temperature is a dial on the
   softmax's contrast. Dividing by a small number *stretches* the logit
   gaps, making the distribution sharper; dividing by a large number
   *compresses* them, flattening it. Watch the same three logits
   (`4.0, 3.0, 2.0`) go through softmax at different temperatures:

   | temp | scaled logits | probabilities | character |
   |---|---|---|---|
   | 0.5 | 8.0, 6.0, 4.0 | 0.867, 0.117, 0.016 | sharp — best token nearly always wins |
   | 0.8 | 5.0, 3.75, 2.5 | 0.731, 0.209, 0.060 | default — mildly random |
   | 1.0 | 4.0, 3.0, 2.0 | 0.665, 0.245, 0.090 | the raw distribution |
   | 2.0 | 2.0, 1.5, 1.0 | 0.507, 0.307, 0.186 | flat — long shots get real odds |

   As `temp → 0` the probabilities collapse onto the argmax (the `1/t`
   multiplier grows without bound, so the top logit's exp wins by an
   infinite margin): temperature 0 *is* greedy. As `temp → ∞` the
   distribution approaches uniform. Everything the earlier stages decided
   survives this step, because dividing by a positive constant never
   changes which logits are bigger — only *how much* bigger.

2. **Softmax.** The real one this time — subtract-max, exp, normalize —
   exactly §2.2. Two implementation details: the running sum accumulates in
   `f64` (152k exp values summed in `f32` would lose low-order bits), and
   entries already masked to −∞ are *skipped* rather than exponentiated —
   `exp(−∞)` is exactly `+0.0`, so skipping is bit-identical and saves ~150k
   transcendental calls per token when top-k/top-p have pruned hard
   (`src/sampler.rs:239-242`).

3. **The multinomial draw.** **Multinomial sampling** means: pick one token
   with probability equal to its weight. Picture the unit interval [0, 1)
   carved into segments whose widths are the probabilities, then drop a
   uniform random dart into [0, 1) and see which segment it lands in:

   ```
   probs:   B=0.731          C=0.209      D=0.060
   [0 ────────────────┬──────────────┬────────┬ 1)
              dart r = 0.85 ────────────────┘ lands in D's segment
   ```

   The code walks the tokens in index order accumulating a running total
   and returns the first token whose cumulative sum reaches the dart
   (`src/sampler.rs:260-273`). The dart comes from `rng.gen()` — one
   uniform `f32` in [0, 1) per generated token — from a **seeded** random
   number generator: `StdRng::seed_from_u64(params.seed)`, default seed 42
   (`src/main.rs:845`). A *seed* is the starting state of a pseudo-random
   generator: same seed in, same sequence of "random" numbers out — which
   is why the same command produces the same text even while sampling
   (§3.3's fourth design question covers why minfer wants that).

### 2.7 Stopping: when does generation end?

The sampler produces a token; the loop decides whether to keep going. Three
gates, checked in this order each step (`src/main.rs:883-918`):

1. **Stop tokens.** `is_stop_token(id, &special)` — true when the sampled id
   equals the model's **eos** (end of sequence) token or `<|im_end|>` (doc
   04: both come from GGUF metadata via `ModelDef::special_tokens()`). When
   the model emits its own "I'm done" marker, the loop breaks and the marker
   is *not* appended or printed. This is the normal ending of an answer.
2. **Stop strings.** The user can pass `--stop "some text"` (repeatable).
   Each step appends the new token's decoded bytes to a buffer `full`, and
   `match_stop_suffix(&full, &stop_refs)` asks: does the accumulated stream
   now *end* with any stop string? Matching bytes — and matching the whole
   stream, not just the newest token — is what makes a stop string **split
   across token boundaries** work: `"world"` might arrive as `wo` + `rld`,
   and neither token is the string, but after the second one the byte suffix
   matches (doc 04 explains why the engine deals in raw bytes at all). The
   code excerpt in §3.2 shows the truncation and the already-streamed-bytes
   behavior on a match.
3. **`n_predict`.** The hard cap from doc 01 (`-n`, default 512) — the
   `while generated.len() < params.n_predict` loop condition.

The `match_stop_suffix` choice of "suffix" is deliberate economy: a stop
string can only complete on the step that emits its last byte, and at that
moment it ends the buffer — so checking only the buffer's tail each step
finds every possible match exactly once, in O(len(stop)) time.

## 3. Implementation

### 3.1 Data in / data out

**In** (all in `main`'s frame, handed to `sampler::sample_with_penalties`):

| Value | Type / shape | From |
|---|---|---|
| `logits` | `&mut Vec<f32>`, exactly `n_vocab` entries (151,936 for Qwen ≈ 0.6 MB) | the last forward (docs 09–11); prefill for the first step, one-token decode for every later step |
| `params.temp` | `f32`, default 0.8 (0 = greedy) | `GenParams` (`src/main.rs:67`) |
| `params.top_k` | `usize`, default 40 | :68 |
| `params.top_p` | `f32`, default 0.95 | :69 |
| `params.repeat_penalty` | `f32`, default 1.1 | :70 |
| `params.frequency_penalty` / `presence_penalty` | `f32`, defaults 0.0 | :71-72 |
| `prev_tokens` | `&[u32]`, ≤ 64 entries | sliding window: prompt tail (`recent_window(&input_ids, 64)`, :847) + every generated token so far (:903-907) |
| `rng` | `&mut StdRng`, seeded once at :845 | `seed_from_u64(params.seed)`, default 42 |

**Out**: a `SampledToken { token_id: u32, logit: f32 }` (`src/sampler.rs:7-13`).
Only `token_id` drives the loop; `logit` is result metadata (a sampled
probability after the final softmax, actually — the field name is
historical).

**The buffer contract** is the quiet star of the API: the sampler *mutates
the caller's logits buffer in place* and returns only the tiny result
struct — no full-vocab copy crosses the sampler boundary, and no second
buffer needs to stay alive per token. Why that shape matters is design
question 5 in §3.3.

**After the call** (still inside one loop iteration, `src/main.rs:900-922`):
stop-token gate → `generated.push(id)` + window update → decode to bytes
into `full` → stop-string gate → stream newly finished bytes → next forward.

### 3.2 Key code

#### The whole pipeline in one function

`src/sampler.rs:280-307` — the complete sampler is eleven lines of calls;
the order *is* the design:

```rust
// src/sampler.rs:283-307
pub fn sample_with_penalties<R: Rng>(
    logits: &mut [f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    prev_tokens: &[u32],
    rng: &mut R,
) -> SampledToken {
    apply_penalties(
        logits, prev_tokens, repeat_penalty, frequency_penalty, presence_penalty,
    );
    if temp < 1e-6 {
        return sample_greedy(logits);
    }
    apply_top_k(logits, top_k);
    apply_top_p(logits, top_p);
    sample_temperature(logits, temp, rng)
}
```

Read the greedy early-return carefully: in greedy mode the **penalties still
apply** (a repeated token can lose the argmax to a non-repeated one — that
is the whole point of the default repeat penalty, and the unit test
`test_sample_pipeline_greedy_applies_penalty` pins it), but top-k and top-p
are *skipped* — correctly, because masking logits to −∞ can never remove the
maximum, so those stages cannot change greedy's choice. Fewer operations,
identical result.

#### Penalties: one pass, one HashMap, llama.cpp semantics

`src/sampler.rs:47-79` (doc comment :31-46 spells out the semantics). First
the counting — note the early-out when every penalty is at its "off" value:

```rust
// src/sampler.rs:54-60
    if (repeat - 1.0).abs() < 1e-6 && frequency.abs() < 1e-6 && presence.abs() < 1e-6 {
        return;
    }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in prev_tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
```

The `HashMap` gives two things at once: distinct tokens (iterate the map,
not the window — a token repeated 10 times is taxed once per its *count*,
not 10 times) and the counts for frequency penalty. Then the combined
mutation per distinct token:

```rust
// src/sampler.rs:61-78 (loop body)
    for (&t, &c) in &counts {
        let idx = t as usize;
        if idx >= logits.len() {
            continue;                       // out-of-range window token: skip, never panic
        }
        let v = logits[idx];
        let mut nv = v;
        if frequency.abs() >= 1e-6 || presence.abs() >= 1e-6 {
            nv -= c as f32 * frequency;     // −count·f  (scales with occurrences)
            if c > 0 {
                nv -= presence;             // −p        (once, count-blind)
            }
        }
        if (repeat - 1.0).abs() >= 1e-6 && repeat >= 1.0 {
            nv = if nv <= 0.0 { nv * repeat } else { nv / repeat };
        }                                   //           (the asymmetric rule)
        logits[idx] = nv;
    }
```

Three guards worth noticing: the `idx >= logits.len()` skip (a corrupt or
out-of-model window token must not panic the stream — the bench path feeds
arbitrary seed tokens); `repeat >= 1.0` (a repeat penalty *below* 1.0 would
*reward* repeats — minfer ignores it rather than implementing the opposite
of the feature's name); and the subtraction-before-division order, which the
test `test_freq_presence_then_repeat_penalty` pins with the comment
"llama.cpp order".

#### The window, and how the loop keeps it sliding

`src/sampler.rs:92-98` is trivial — the interesting part is the call-site
choreography in `main.rs`:

```rust
// src/sampler.rs:95-98
pub fn recent_window(tokens: &[u32], last_n: usize) -> Vec<u32> {
    let start = tokens.len().saturating_sub(last_n);
    tokens[start..].to_vec()
}

// src/main.rs:842-847 — seeded once, window seeded from the PROMPT tail
    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    const REPEAT_LAST_N: usize = 64;
    let mut prev_tokens = sampler::recent_window(&input_ids, REPEAT_LAST_N);

// src/main.rs:903-907 — every sampled token enters the window
        generated.push(sampled.token_id);
        prev_tokens.push(sampled.token_id);
        if prev_tokens.len() > REPEAT_LAST_N {
            prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
        }
```

The comment at :842-844 carries the design point: the window is initialized
from the *prompt's* last 64 tokens "so the first generated tokens are
penalized too". Without that, step 1 of the generation would have an empty
window and a prompt that says "translate: the the the" could immediately
sample `the`. The window is a fixed-capacity sliding window: push, then
drain from the front to 64 — the prompt tail ages out as generation
proceeds, and by token 65 of output the penalties look only at generated
text.

#### Top-k: O(n) threshold, mask in place

`src/sampler.rs:128-140`:

```rust
// src/sampler.rs:128-140
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut sorted = logits.to_vec();
    sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let threshold = sorted[k - 1];
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}
```

The trick is `select_nth_unstable_by`: Rust's partial-selection primitive
that partitions a slice so the element that *would* be k-th in sorted order
lands at index k−1, in O(n) average time instead of the O(n log n) of a full
sort (for n = 151,936: roughly 300k comparisons vs ~2.6M). It runs **on a
copy** — `select_nth_unstable_by` *reorders* whatever slice it is given,
which would destroy the index↔token mapping in the caller's buffer; the
original is only *read* to extract the threshold and *masked*, never moved
(the doc comment :121-127 records exactly this). After the mask, ties are
kept: `*v < threshold` is strict, so tokens exactly at the threshold survive
— top-k keeps *at least* k candidates.

The `k == 0` guard doubles as the off switch: `--top-k 0` disables filtering
entirely and hands the full vocabulary to top-p (which then takes its
full-array path, below).

#### Top-p: nucleus over survivors, mask raw logits

`src/sampler.rs:152-201`, the function whose doc comment (:142-151) is the
design record. First the fast path — after top-k, at most k entries are
finite:

```rust
// src/sampler.rs:156-168
    let survivors: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > f32::NEG_INFINITY)
        .map(|(i, &v)| (i, v))
        .collect();
    if survivors.is_empty() {
        return;
    }
    if survivors.len() > 1024 {
        apply_top_p_full(logits, p);
        return;
    }
```

Then the nucleus decision, computed on the survivors alone:

```rust
// src/sampler.rs:173-200 (core)
    let max_val = survivors.iter().fold(f32::NEG_INFINITY, |a, &(_, v)| a.max(v));
    let sum: f64 = survivors
        .iter()
        .map(|&(_, v)| ((v - max_val) as f64).exp())
        .sum();
    let mut cand: Vec<(usize, f32)> = survivors
        .iter()
        .map(|&(i, v)| (i, (v - max_val).exp() / sum as f32))
        .collect();

    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut keep = cand.len();
    for (i, &(_, prob)) in cand.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            keep = i + 1;               // the token that crosses the line is kept
            break;
        }
    }
    for &(idx, _) in &cand[keep..] {
        logits[idx] = f32::NEG_INFINITY;  // mask the LOGIT, keep the winners' logits raw
    }
```

Why is summing only the survivors legal? Because the excluded entries are
already −∞, and `exp(−∞) = +0.0` *exactly* in IEEE floats — adding exact
zeros to an `f64` running sum changes nothing, so the survivor-only sum is
**bit-identical** to the full-array sum. That is the claim in the comment at
:170-172, and it is what makes the ≤1024-candidate fast path a pure
optimization rather than an approximation. When top-k is disabled (or the
distribution is pathologically flat), more than 1024 entries survive and the
code falls back to `apply_top_p_full` (:206-226) — the original
softmax-over-everything + full sort, kept so `--top-k 0` never silently
changes semantics.

The final loop is the part §2.5 promised: losers get −∞ in **logit space**;
the winners' entries stay untouched raw logits for the temperature stage.

#### Temperature + softmax + the draw

`src/sampler.rs:229-278`, all three sub-steps in one function:

```rust
// src/sampler.rs:229-256 (core)
pub fn sample_temperature<R: Rng>(logits: &mut [f32], temp: f32, rng: &mut R) -> SampledToken {
    if temp < 1e-6 {
        return sample_greedy(logits);          // greedy IS temp 0
    }

    let inv_temp = 1.0 / temp;
    for v in logits.iter_mut() {
        *v *= inv_temp;                        // (−∞)·finite stays −∞
    }

    // Softmax. Masked (-INF) logits map to exp(-INF)=0 and contribute nothing
    // to the running max or sum; skipping the exp() call for them avoids
    // ~n_vocab transcendental evaluations per token while staying bit-identical
    // (exp(-INF) == +0.0 exactly).
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f64;
    for v in logits.iter_mut() {
        if *v > f32::NEG_INFINITY {
            *v = (*v - max_val).exp();
        } else {
            *v = 0.0;
        }
        sum += *v as f64;
    }
```

Then normalization and the draw (`:253-277`): divide by the sum in place,
draw one uniform `f32` via `rng.gen()`, and walk the buffer accumulating:

```rust
// src/sampler.rs:260-277
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if v <= 0.0 {
            continue;                          // pruned entries: zero probability
        }
        cumulative += v;
        if r <= cumulative {
            return SampledToken { token_id: i as u32, logit: v };
        }
    }
    SampledToken { token_id: (logits.len() - 1) as u32, logit: logits[logits.len() - 1] }
```

The `v <= 0.0` skip is why the walk is cheap after pruning (≤ 40 real
entries after top-k) — and the trailing fallback return is the float-safety
net: if rounding leaves the cumulative total at 0.9999… while the dart drew
0.99995, the last finite entry wins instead of nobody. Note the buffer is
destroyed by this function (it now holds probabilities, not logits) — which
is fine, because the caller never reads it again; the next forward returns a
fresh `Vec` (§3.1).

#### Stop tokens and stop strings at the call site

The gates run immediately after sampling, before anything is appended:

```rust
// src/main.rs:883-918 (condensed to the sampler-adjacent lines)
    while generated.len() < params.n_predict {
        let sampled = sampler::sample_with_penalties(
            &mut logits,
            params.temp, params.top_k, params.top_p,
            params.repeat_penalty, params.frequency_penalty, params.presence_penalty,
            &prev_tokens,
            &mut rng,
        );

        if is_stop_token(sampled.token_id, &special) {
            break;                                  // eos / <|im_end|>: silent stop
        }
        generated.push(sampled.token_id);
        prev_tokens.push(sampled.token_id);
        /* … window drain … */

        // Stop-string detection on the FULL byte stream before emitting.
        full.extend_from_slice(&tokenizer.decode_bytes(&[sampled.token_id]));
        if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
            full.truncate(cut);                     // drop the stop string itself
            if cut > emitted {
                hi.feed(&full[emitted..]);          // flush bytes not yet streamed
                emitted = full.len();
            }
            break;
        }
        if emitted < full.len() {
            hi.feed(&full[emitted..]);              // stream newly finished bytes
            emitted = full.len();
        }
```

`is_stop_token` is the two-line sentinel (`src/main.rs:1020-1022`):

```rust
fn is_stop_token(id: u32, special: &models::SpecialTokens) -> bool {
    id == special.eos || Some(id) == special.im_end
}
```

and `match_stop_suffix` is the byte-level tail check (`src/sampler.rs:107-119`):

```rust
// src/sampler.rs:107-118
pub fn match_stop_suffix(buf: &[u8], stops: &[&[u8]]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for s in stops {
        if s.is_empty() || s.len() > buf.len() {
            continue;                               // empty stops ignored; too-long can't match
        }
        if &buf[buf.len() - s.len()..] == *s {
            let start = buf.len() - s.len();
            best = Some(best.map_or(start, |b| b.min(start)));
        }                                           // several matches: earliest start wins
    }
    best
}
```

"Earliest start wins" means the *longest* applicable stop string decides
where to cut (a later start truncates less text). The bytes already flushed
to the terminal before the match completed are not un-printed — that is the
llama.cpp antiprompt behavior the comment at `src/main.rs:849-853` describes,
and it is why `full` (everything generated) and `emitted` (everything
streamed) are tracked as separate cursors.

### 3.3 Design choices (why this shape and not another)

**1. Why sample at all instead of always taking the max?** §2.3 explained
the failure mode: greedy steps are locally optimal, but locally optimal
steps compound into globally degenerate sequences — the model repeating
"the the the" does nothing wrong per step, it has simply fallen into a
self-reinforcing basin where the most likely token after a repetition is
the same repetition. Multinomial sampling injects exactly enough noise to
escape those basins while staying biased toward good tokens (a 0.73-likely
token still wins 73% of the time); temperature is the coherence-vs-diversity
dial and greedy its zero point (`--greedy`, `src/main.rs:312-314`). The
composition minfer chose matters as much as the mechanism: penalties apply
in *both* modes (a repeated token can lose the argmax), so even greedy
output is loop-resistant — the default 1.1 penalty does real work in every
verification run in this repo.

**2. Why penalize repeats in logit space, with the asymmetric ÷/× rule —
and why a 64-token window?** Work backwards from what the downstream
consumer needs. Softmax only cares about logit *differences*; "make token t
less likely" always means "move t's logit down relative to everyone else".
The penalties therefore edit logits directly, where one small, composable
step does the whole job — no re-normalization, no separate probability
pass. The multiplicative form (`÷1.1` / `×1.1`) rather than subtracting a
constant is a scale choice: logit magnitudes vary across models, quants, and
positions, and a *ratio* taxes a confident repeat (logit 12) harder than a
hesitant one (logit 2) in exactly the proportion that matters — while one
fixed subtraction would be negligible for the first and absurd for the
second. The asymmetry is then forced by the sign: dividing a *negative*
logit by 1.1 would move it toward zero, i.e. make an unlikely repeated
token *more* likely — the opposite of the feature. Multiplying instead
pushes it further from the pack. Either way the rule means "this token got
less likely", which is the only contract softmax cares about. (An additive
design could also be made sign-aware, but it would need a scale-calibrated
constant per model; the ratio needs none, which is why llama.cpp ships one
default that works everywhere.)

The window is 64 for a bent-cost curve: a degenerate loop repeats within a
handful of tokens, and phrase-level echoes ("as I said above") live within
tens — 64 covers both, so anything the model *is* stuck on gets taxed every
step. Meanwhile the topical words a document genuinely needs — names, terms
of art — recur over spans of hundreds of tokens; a window of 64 has already
forgotten them by the time they are legitimately needed again. Longer
windows start punishing correctness (a code generator needs its brackets
and whitespace back after 70 tokens), shorter ones let loops survive past
the horizon. 64 is llama.cpp's tuned compromise, and minfer copies it
instead of re-tuning (doc 01 §2.3's "less invention risk").

**3. Why is top-p implemented as logit masking rather than truncating the
probability vector?** The comment at `src/sampler.rs:142-145` states the
rule — "sets excluded tokens' raw logits to −∞ (does NOT overwrite logits
with probabilities, so the final temperature softmax stays correct)" — and
the reason is composition. The pipeline applies top-p *before* temperature,
and temperature divides logits by `1/temp` and re-softmaxes. If top-p had
overwritten the survivors' entries with their temp-1 probabilities, the
temperature stage would then scale *probabilities* (multiply by `1/temp`
and exponentiate *them*), producing a different distribution than
"nucleus-filter, then temperature" — e.g. flattening p = 0.9 → p⁰·⁸-style
distortions instead of a clean renormalized subset. Keeping raw logits
makes the two stages commute cleanly: filtering is a set operation
(remove, don't rewrite), temperature is a shape operation on whatever set
remains. It also matches llama.cpp, whose sampler chain passes logits
through link by link with each link mutating in place — top-p sees
untempered logits, temp runs after — so parity of *semantics*, not just of
defaults, is preserved. And there is a numerical dividend: masking −∞
survivors is exact (`exp(−∞) = +0.0`), so the fast path that softmaxes only
the ≤1024 (in practice ≤40) survivors is bit-identical to the full-vocab
computation (:147-152, :170-172) — no epsilon drift between code paths.

**4. Why a fixed seed (42) by default instead of entropy?** Because in this
repo, determinism is infrastructure. Doc 01 §2.3 records the decision; the
consumers are everywhere: the conversation tests (`tests/conversation_cli.rs`)
pipe scripted stdin and assert on the output — impossible if every run
rolls new dice; the CUDA optimization campaign's verification gate runs
`-n 32 --greedy --seed 42` before and after every kernel change and
requires byte-identical token streams
(`docs/cuda_optimization_steps/77-verification-methodology.md` §2.2) — and
when the greedy stream is the gate, you want the *seeded sampling* path
audited by the same machinery (the campaign's gate chain explicitly pairs
greedy identity with the `rp=1.0` seeded stream, §2.3/§2.4 there); and a
support report that says "same model, same flags, same output" turns a
heisenbug into a reproduction recipe. Entropy by default would buy lottery
variety nobody asked for and cost every one of those properties. One seed
value is one flag away (`--seed N`), so users who want fresh rolls per run
can have them — opt-in, not default. Note the subtlety that makes seeded
sampling a *usable* gate: the single `StdRng` is constructed once before
the loop (:845), so the whole generation consumes one deterministic stream
of draws; identical model + flags + seed ⇒ identical tokens even at temp
0.8, because minfer's forwards are themselves deterministic. (One honest
limit of the gate: GPU kernels must also be deterministic for the equality
to hold — which is exactly why the campaign pairs greedy identity with the
parity tests rather than trusting either alone.)

**5. Why does the sampler mutate the logits buffer in place?** Because the
obvious alternative — `let probs = sampler::sample(logits.clone(), …)` or
returning a new `Vec` — buys a full-vocab copy per token for zero benefit.
The arithmetic: 151,936 × 4 B = 607,744 B ≈ 0.6 MB per generated token,
which is precisely the copy the call-site comment at `src/main.rs:924-925`
brags about not making ("move the Vec in place instead of copying 607
KB/token") for the *forward* boundary — the sampler boundary gets the same
treatment. The ownership dance that makes it work: `logits` starts as the
prefill output; each iteration passes `&mut logits` into the sampler (which
consumes and destroys it — after `sample_temperature` it holds
probabilities); then `logits = model.forward(&[sampled.token_id], …)`
(:932) rebinds the variable to the fresh `Vec` the forward allocated. One
buffer alive at a time, no `.clone()`, no caller-visible aliasing (the
sampler takes `&mut [f32]`, so the borrow checker guarantees nobody reads
the half-transformed buffer mid-pipeline). The one internal copy that
remains — `apply_top_k`'s `to_vec()` for the selection pass — is a
deliberate, short-lived temp: `select_nth_unstable_by` reorders what it
sorts, and reordering the caller's buffer would scramble the index→token
mapping (:121-127). Copy-then-select keeps the mutation mask-only at the
cost of one temp that dies at the end of the function.

**6. Why this stage order — penalties → top-k → top-p → temperature?**
Because it is llama.cpp's order, and each neighbor-pair has a reason.
Penalties first, so a taxed token can also be pruned by k/p (and so greedy
benefits). Top-k before top-p because it is the cheap O(n) pre-filter that
guarantees top-p's softmax only ever handles ≤40 survivors — the reverse
order would compute a full-vocab softmax for nothing. Temperature last,
because it must shape the *final* set (§3 above), and because applying it
earlier would change the probabilities top-p uses to draw the nucleus.
The greedy shortcut (:301-303) sits where it does because penalties must
run even when the stochastic stages are skipped.

### 3.4 Pitfalls & invariants

- **−∞ is the "removed" sentinel, end to end.** Every pruning stage masks
  with `f32::NEG_INFINITY`, and every downstream consumer is written
  against that contract: softmax skips −∞ (`exp(−∞) = +0.0`), top-p's
  survivor scan filters `v > f32::NEG_INFINITY`, the draw skips `v <= 0.0`.
  A new stage that writes 0.0 instead of −∞ would corrupt top-p's survivor
  count; one that writes any finite sentinel would sneak into the softmax.
- **Top-p's survivor fast path is exact, not approximate** — but only
  because the excluded entries contribute *exact* zeros. The >1024 fallback
  (`apply_top_p_full`) exists so `--top-k 0` (or a bizarrely flat
  distribution) never falls off the bit-exactness claim; it is the
  full-vocab softmax + full sort, deliberately preserved (:203-205).
- **The penalty pass silently ignores a repeat penalty < 1.0** and
  out-of-range window tokens (`:62-65, :74`). Neither is an error path:
  a sub-1.0 penalty would *encourage* repetition, and a bad id must not
  panic a generation stream.
- **The order inside `apply_penalties` is load-bearing**: frequency/
  presence subtraction *then* the repeat ÷/× — the reverse order gives
  different numbers (and the test `test_freq_presence_then_repeat_penalty`
  pins llama.cpp's order with a worked case: 4 − 1 − 1 = 2, then ÷2 = 1).
- **The window spans prompt *and* generation** (`:842-847` + `:903-907`).
  A new code path that resets `prev_tokens` at generation start re-opens
  the "first generated token repeats the prompt" hole.
- **Greedy still pays the penalty pass** (`:301-303`): any change that
  moves the greedy return *before* `apply_penalties` changes the output of
  every `--greedy` run with a nonzero repeat penalty — and every
  verification gate in this repo that uses `-n 32 --greedy --seed 42`.
- **Stop strings are bytes, not strings** (`match_stop_suffix(&[u8]…)`):
  matching happens on the accumulated byte stream because tokens can split
  multi-byte characters (doc 04 §2.4). Converting to `String` mid-stream
  would make split-CJK stop strings unmatchable and lossy.
- **Already-emitted bytes stay emitted** (`:911-918`): the stop-string
  truncation rewinds `full`, never the terminal. Anything else would be a
  lie about what the user saw.
- **The draw's fallback return is unreachable in exact math but reachable
  in floats**: cumulative sums of probabilities can land at 0.99999…, so a
  dart in the gap returns the last finite entry rather than nothing
  (`:274-277`). Removing that arm invites a rare-but-real "no token"
  panic.
- **The sampler destroys its input buffer** (it holds probabilities after
  `sample_temperature`). The caller-side invariant that makes this safe:
  `logits` is reassigned from the next forward before it is ever read
  again (`src/main.rs:932`). Reading `logits` between sample and reassign
  would read probabilities and call them logits.

## 4. Observe & verify

- **`MINFER_TIMING=1`** — decomposes per-token wall time into `t_samp`
  (the sampling call, :896-898) vs the forward. Expect sampling in the
  microseconds: the whole pipeline is a 64-entry hash pass, one O(n)
  selection, a ≤40-entry softmax, and one draw.
- **`MINFER_TRACE=<path>`** — the decode loop attaches each sampled token's
  id and decoded text to the trace (`crate::trace::set_token`,
  `src/main.rs:926-930`); the viz page shows the token stream the sampler
  produced.
- **The reproducibility experiment** — run the same command twice, e.g.
  `./target/release/minfer qwen2.5-0.5b-instruct-q4_0 "Tell me a story" -n 24`:
  both runs print identical text (seed 42); add `--seed 7` to either and the
  text differs while the prompt handling stays identical. That pair of
  observations is the seeded-sampler contract, live.
- **Greedy mode** — `--greedy` (or `--temp 0`) runs the penalty pass +
  argmax; `bench` subcommand output is produced this way
  (`src/bench.rs:24`) so throughput numbers measure the engine, not the
  lottery.
- **Stop strings** — `--stop` with a string that straddles tokens, e.g.
  `--stop "the mat"` against a prompt that will produce it; the output ends
  *before* the stop text, and with a multi-byte stop string you can watch
  the byte-level match fire only when the final byte arrives.
- **Unit tests** — `cargo test sampler::` covers every stage with small
  hand-checkable numbers: greedy picks max; positive logit ÷ penalty and
  negative × penalty; frequency scales with count while presence is
  once-only; freq+presence precede repeat; defaults are a no-op; top-k
  masks below threshold; top-p keeps only the nucleus and does *not*
  overwrite the surviving raw logit; two identical seeds give identical
  tokens; stop-suffix basics, longest-wins, and the CJK
  `E4 B8 AD`-split-across-tokens case.

## 5. Cross-references

- **[01 — CLI args and model resolution](01-cli-args-model-resolution.md)**
  §2.2 — the table that gives every knob its one-sentence meaning and §2.3
  — why the defaults are llama.cpp's and why `seed: 42` is fixed.
- **[04 — Tokenizer + template](04-tokenizer-template.md)** §2.5 — the
  special tokens (`eos`, `<|im_end|>`) that become stop sentinels here, and
  §3.2.4 — `decode_bytes`, the byte-level decoder the stop-string check
  runs on.
- **[09 — Prefill forward path](09-prefill-forward-path.md)** — where the
  `last_logits` `Vec` this stage consumes comes from (n_out = 1: only the
  final prompt token's scores).
- **[11 — Attention, vec ops, and the KV cache](11-attention-vecops-kv.md)**
  — the *other* softmax in the engine (attention's); same subtract-max
  trick, different purpose.
- **[13 — The decode loop and graph reuse](13-decode-loop-graph-reuse.md)**
  — the loop that calls this stage and feeds the chosen id back into the
  graph; the `logits` reassignment dance at :932 is that doc's territory.
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §3 — the flowchart row
  "sample next token: penalties → top-k → top-p → temp" and the defaults
  list this doc implements.
- [`docs/OPENAI-CHAT-API-PLAN.md`](../OPENAI-CHAT-API-PLAN.md) — Phase 1
  introduced the frequency/presence penalties and stop strings for the
  OpenAI-compatible server; the CLI consumes the same `sampler.rs` API.
- [`docs/cuda_optimization_steps/77-verification-methodology.md`](../cuda_optimization_steps/77-verification-methodology.md)
  — the verification campaign whose greedy-identity and seeded-stream gates
  depend on this sampler's determinism.
- [`docs/USAGE.md`](../USAGE.md) — every sampling flag with its default;
  [`docs/GLOSSARY.md`](../GLOSSARY.md) — backstop definitions (softmax,
  nucleus, temperature, multinomial).

← [11 — Attention, vec ops, and the KV cache](11-attention-vecops-kv.md) · [Index](./README.md) · [13 — The decode loop and graph reuse](13-decode-loop-graph-reuse.md) →
