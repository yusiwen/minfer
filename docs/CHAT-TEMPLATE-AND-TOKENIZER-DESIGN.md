# F7 — Chat-template fidelity and tokenizer generality (design record)

Ticket: [#50](https://github.com/yusiwen/minfer/issues/50) ("[F7] Chat-template
fidelity and tokenizer generality"). This document is the **design-first**
artifact: it is committed before the implementation and states the accepted and
refused sets, how a refusal is detected, and the reference against which each
gate measures.

Two independent surfaces are in scope:

1. `src/template.rs` — render the model's own `tokenizer.chat_template` instead
   of silently falling back to a generic ChatML rendering.
2. `src/tokenizer.rs` — make `tokenizer.ggml.pre` authoritative, generalize the
   pre-tokenization rules, and replace the silent `unwrap_or(0)` byte fallback
   with a loud, checked one.

## 1. Why the fallback existed

`render_messages` / `render_template` compiled the GGUF template into
`minijinja` and, on **any** failure (invalid syntax or a render error), printed
a warning and returned `fallback_chatml_messages`. For Qwen3 that path is
always taken: its template uses Python string-method syntax
(`message.content.split('</think>')[-1].lstrip('\n')`,
`.rstrip('\n')`, `reasoning_content.strip('\n')`) and minijinja 2.21.0 exposes no
`str` methods — it expects Jinja filters (`|split`, `|replace`) instead. The
recorded root cause is `docs/QWEN3-SUPPORT-PLAN.md` §5 gotcha #9. The
consequence was silent: the template's think-block extraction, tool-call
formatting, multi-step-tool collapse and `enable_thinking` handling were lost,
and the caller could not tell that a different prompt was rendered.

Two further silent paths existed: `Environment::add_template` failing on a
syntax error, and callers passing `""` for a missing template, which renders the
**empty string** rather than ChatML.

## 2. Templates — accepted, refused, detected

### 2.1 Closing the minijinja gap: option (a), the extension point

`minijinja` 2.21.0 provides `Environment::set_unknown_method_callback`, whose
own documentation names this exact use case ("increase the compatibility with
Jinja2 templates that might call Python methods"). We implement the Python
`str` methods the transformer chat templates in the wild use, on top of it.

Rejected alternatives, recorded:

* **(b) upgrade minijinja.** 2.21.0 is the version this tree pins and provides
  the hook; no upgrade is required, and a bump would not by itself give Python
  semantics (`str.split` is not `|split`, and Python's `lstrip` takes a *set* of
  characters). Bumping an unrelated dependency for a feature the pinned version
  already exposes is scope the ticket does not need.
* **(c) mix.** Not needed.
* **`minijinja-contrib`'s `pycompat`** implements the same hook, but it is a new
  dependency and a superset with its own edge semantics; the engine's
  dependency set stays as it is (see `AGENTS.md` → Dependencies).

### 2.2 Accepted constructs

Everything minijinja already renders, plus the string methods below. The
Qwen2.5/Qwen3 templates exercise: `{%- if/elif/else %}`, `{% for %}` with
`loop.first/index/index0/last`, `{% set %}`, `namespace()`,
`range(start, stop, step)` with a negative step, `messages[i]` including a
negative index (`messages[loop.index0 - 1]`), `x in message.content`,
`|length`, `|tojson`, `is defined`, `is none`, `is string`, `+`/`~` string
concatenation, arithmetic/comparison expressions, and whitespace control.

Python `str` methods implemented by the callback (Python semantics, not Jinja
filter semantics — the reference is CPython):

| Method | Notes |
|---|---|
| `strip`, `lstrip`, `rstrip` | optional argument is a **set** of characters, not a prefix/suffix |
| `split`, `rsplit` | `sep` optional (whitespace runs when absent), `maxsplit` |
| `startswith`, `endswith` | string or sequence of strings |
| `replace` | `old`, `new`, optional `count` |
| `lower`, `upper`, `title`, `capitalize` | |
| `join` | sequence of strings |
| `find`, `rfind`, `count` | substring, optional `start`/`end` |

`raise_exception(message)` is registered as a function: a template that refuses
its own input (many published templates do) surfaces that message instead of an
"unknown function" error. Slices with a step (`messages[::-1]`) are minijinja
syntax and need no extension — the current Qwen3 template uses it.

### 2.3 Refused, and how the refusal is detected

A template is **refused loudly** — never silently re-rendered as ChatML — when:

1. `Environment::add_template` fails: syntax error, unknown filter or test, a
   construct minijinja does not parse. The error carries the template line.
2. Rendering fails: an unknown method (the callback returns
   `ErrorKind::UnknownMethod` for anything outside the table above), a value of
   the wrong type, or an undefined value in a position that requires one.

The refusal text names the construct and the line, e.g.

```
chat template error — unsupported template construct: unsupported Python str
method `splitlines` (template line 41); minfer refuses to fall back to a generic
ChatML prompt. Supported Python str methods: capitalize, count, endswith, find,
join, lower, lstrip, replace, rfind, rsplit, rstrip, split, startswith, strip,
title, upper.
```

`render_messages` / `render_template` / `format_single` therefore return
`Result<_, TemplateError>`; `TemplateError::message()` is what the callers
print. The template is also validated **once at load** (`template::validate`),
so the CLI fails before inference and `serve` refuses to start with an
unrenderable template rather than 500-ing per request (a per-request render
error is still mapped to HTTP 400 defensively).

### 2.4 What happens to callers that relied on the fallback

ChatML is kept for exactly one case, and it is not a silent one: a GGUF with
**no** `tokenizer.chat_template` at all (`Option::None`). That case has no
model-specific format to lose, the startup path prints a one-line notice, and
`docs/USAGE.md` documents it. Every other caller now sees the refusal:

* CLI (`main.rs`): template validated at load → `Error: …` on stderr, exit 1.
* `serve` / `viz` / batch: validated at load → startup refuses; a render error
  at request time is a `400 invalid_request` carrying the message text.
* `conversation.rs`: propagates the error instead of a wrong delta.
* Callers that passed `""` for a missing template (`unwrap_or("")`) are fixed —
  they used to render an empty prompt, which is neither ChatML nor the model's
  template.

### 2.5 Reference and fixtures

The reference rendering is produced by **transformers 5.17.0**
(`AutoTokenizer.apply_chat_template(..., tokenize=False)`), i.e. Python
`jinja2` with the model's own `chat_template` from its
`tokenizer_config.json`, fetched from the Hugging Face Hub. The generator is
`/tmp` tooling kept out of the tree; its output is committed as JSON fixtures
under `tests/fixtures/chat/` with the provenance (model id, template source,
generator command, date) inside each file. The engine is never its own
reference.

Cases per model: single user + generation prompt, system + user, a multi-turn
conversation with and without `add_generation_prompt`, a `<think>`-reasoning
assistant turn (the construct that motivated the ticket), and a Unicode/`\n`
payload.

## 3. Tokenizer — accepted, refused, detected

### 3.0 The engine comes first: `tokenizer.ggml.model` (#132 S4, 2026-10-15; ADR-0045)

`tokenizer.ggml.pre` answers "which rule" only for the engine the file already selected.
`TokenizerModel::from_gguf` reads `tokenizer.ggml.model` and answers exactly two engines:

| value | engine | driven by | `pre` used? | accepted |
|---|---|---|---|---|
| `gpt2` (or absent) | byte-level BPE | `tokenizer.ggml.merges` ranks + byte-level characters | yes — §3.1 | **yes** |
| `llama` | SentencePiece | `tokenizer.ggml.scores` + `▁` (U+2581) escaping + `<0xXX>` byte tokens | no | **yes** |
| anything else | — | — | — | refused, **naming the value** |

SentencePiece (from llama.cpp's session, `llama-vocab.cpp:114`) is: symbols are UTF-8
characters of the `▁`-escaped text; every adjacent pair whose concatenation is a vocabulary
entry is queued with that entry's **score**; the highest score merges first, and a pair whose
sides changed since it was queued is stale and skipped; the final walk emits a vocabulary
entry where it can, re-splits through the recorded pair where it cannot, and otherwise emits
one `<0xXX>` byte token per byte. `add_space_prefix` (default `true` for SPM,
`tokenizer.ggml.add_space_prefix` overrides it) prefixes the first raw-text segment and every
segment after a special token; a special token resets that state.

`Tokenizer::pre` is therefore `Option<PreTokenizer>`: `None` means the engine has no
pre-tokenization rule, and `applies_nfc()` is then `false` — the SPM families accepted here
declare no normalizer (ADR-0044).

### 3.1 `tokenizer.ggml.pre` is authoritative

The pre-tokenizer was a single hardcoded regex, and it was subtly wrong: it had
no `(?!\S)` arm (so `a  b` grouped the two interior spaces instead of splitting
off the first) and no `\s*[\r\n]+` arm. The rules now come from
`tokenizer.ggml.pre`:

| `pre` value (aliases) | Rule | Reference regex (from `tokenizer.json`) | Input normalizer |
|---|---|---|---|
| `qwen2` (`deepseek-r1-qwen`) | Qwen2/Qwen2.5/Qwen3 | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` | **NFC** |
| `qwen35` | Qwen3.5 (letter runs also consume `\p{M}`) | same, with `[\p{L}\p{M}]+` and `[^\s\p{L}\p{M}\p{N}]` | **NFC** |
| `llama-bpe` (`llama3`, `llama-v3`) | Llama 3.x (#132 S3, 2026-10-15) | same as `qwen2`, with `\p{N}{1,3}` in place of `\p{N}` — digits group **three at a time** | none (`"normalizer": null`) |
| `lfm2` | LiquidAI LFM2 (#132 S3, 2026-10-15) | **byte-identical** to `llama-bpe`'s pattern (verified on `LiquidAI/LFM2-1.2B@40f3da0d`) | none (`"normalizer": null`) |

A `pre` value therefore answers two questions: which pattern splits the text, and whether the
input is normalized first. Both come from the family's reference `tokenizer.json`, because the
GGUF carries no normalizer field (#564, ADR-0044); §3.6 is the mechanism.

Every other value — **including a missing key** — is a refused load:
`Tokenizer::load` returns `Err` naming the value and the supported set. A guess
here is exactly the "wrong split" the ticket forbids.

The classic GPT-2 rule was in the first draft of this design and was dropped
before implementation for a concrete reason: GPT-2's `tokenizer.json` carries no
`Split` pattern at all (its pre-tokenizer is a bare `ByteLevel`), so there is no
independent regex reference to gate the rule against, and this engine's
supported architectures do not use it. Shipping it ungated would have been a
guess wearing the name of a rule; it is refused by name instead.

The splitter is hand-written (a port of llama.cpp's
`unicode_regex_split_custom_qwen2` / `_qwen35` / `_llama3` / `_gpt2`) rather
than a `regex` crate pattern, for two reasons: the Rust `regex` crate has **no
lookahead**, so `(?!\S)` cannot be expressed; and a hand-written splitter makes
the byte spans explicit, which is what the byte-level BPE and the grammar mask
(F2) consume.

**Divergence, recorded (`llama-bpe`).** llama.cpp sets `ignore_merges` for this
pre-type (`src/llama-vocab.cpp:2155`), and its BPE returns a piece unchanged when
the piece *is* a vocabulary entry (`src/llama-vocab.cpp:627`); minfer removed that
shortcut in F7 and always merges, which is what HuggingFace does and what the
`tokenizer.json` declares. On the committed 52-entry corpus the two references
agree byte for byte (measured 2026-10-15 with `llama-tokenize` on the same GGUF,
52/52), so the divergence class is narrow: a pre-tokenized piece that is a
vocabulary entry **unreachable through merges**.

### 3.2 Special and added tokens (#132 S2, 2026-10-15)

The special-token set is **what the file declares**, and nothing is inferred from
a token's name (ADR-0043):

* `tokenizer.ggml.token_type` **2 (UNKNOWN), 3 (CONTROL) or 4 (USER_DEFINED)** —
  llama.cpp's special-token cache predicate
  (`src/llama-vocab.cpp:3012`, `CONTROL | USER_DEFINED | UNKNOWN`);
* every id a declared role names, over the key table `SPECIAL_ROLE_KEYS` in
  `src/tokenizer.rs` — transcribed from llama.cpp's role list
  (`src/llama-arch.cpp:382-406`, deprecated `prefix`/`suffix`/`middle` aliases
  included). A role is a declaration: the token matches as one id even when its
  type says NORMAL.

Matching is unchanged — earliest position wins, longest text wins at the same
position, before BPE — so Qwen's `<|im_start|>` (type 3 in every Qwen GGUF) and
DeepSeek-R1's `<｜User｜>`/`<think>` (type 3/4) stay single tokens.

**The name hardcode is gone.** Special-token matching used to add
`<|im_start|>`, `<|im_end|>` and any EOS text starting with `<`, whether or not
the vocabulary contained them. Measured on
`unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` (Q4_K_M, probed 2026-10-15): that
vocabulary has **neither** name, so `<|im_start|>` was mapped to id `0` and
`<|im_end|>` to the EOS token `151643` — two literal strings tokenized as tokens
the model never had. Now they are ordinary text, which is also what llama.cpp
does: it forces its name list upward only when the role is *not* declared, and it
never invents a token. minfer does not port that name list; a converter that
leaves a chat token type NORMAL gets llama.cpp's BPE split there too.

**Divergence, deliberate.** For a token the file declares a role for *and* types
NORMAL, minfer matches it (the declaration wins) while llama.cpp splits it (the
type wins unless the name is in its list). The declaration is the data; the record
of that choice, with the rejected alternatives, is ADR-0043.

### 3.3 Byte fallback replaces the silent id 0

`bpe_encode` used to emit token id `0` for any piece it could not look up — a
silent corruption. It now walks the piece byte by byte and emits the
corresponding single-byte token (byte-level BPE maps every UTF-8 byte to a
Unicode codepoint whose token exists in the vocabulary). `Tokenizer::load`
verifies that **all 256** byte tokens exist and refuses the load when they do
not, so the fallback cannot silently degrade; a lookup that still fails is an
invariant violation that aborts with the byte value.

### 3.4 Not covered (documented, refused, or a follow-up)

* **NFC normalization** — *now applied* (F7's accepted gap, closed by
  [#132](https://github.com/yusiwen/minfer/issues/132) S1 on 2026-10-14); see §3.6.
* **`bert` / `t5` / `rwkv` / `plamo2` / `gemma4` / `whitespace` / `hybriddna` /
  `none`** engines — refused by name at load (§3.0). SentencePiece (`llama`) is
  implemented (ADR-0045); each remaining value is a different algorithm
  (WordPiece, unigram with its own normalization, RWKV, …) with its own
  vocabulary shape and its own reference, so it lands with an ADR and a fixture,
  not as another match arm.
* **The remaining `ignore_merges` pre-tokenizers** (`falcon3`, `falcon-h1`,
  `pixtral`, `tekken`, …) and **multi-regex** pre-tokenizers (`default`,
  `deepseek-llm`, `deepseek-coder`, `deepseek-v3`, `falcon`, `starcoder`, `mpt`,
  `refact`, …): refused by name at load. Each one needs its own pattern check and
  its own pair of fixtures — llama.cpp maps several of them onto the same
  pre-type with an *adapted* regex, which is exactly why the value, not the
  family, is what gets accepted.
* **`byte_encode = false`** SPM-style BPE (`whitespace` pre-tokenizer) and the
  `<0xXX>` byte fallback for it: refused by name at load.

### 3.5 Findings recorded while implementing

Two of them changed the implementation and one the reference:

* **The whole-piece BPE shortcut was wrong.** The encoder used to return a
  single id when the whole pre-tokenized piece was in the vocabulary, skipping
  the merge loop. A vocabulary entry is not necessarily reachable through
  merges: Qwen3.5's vocabulary holds the Devanagari cluster `क्` + `ष` as one
  token with no rank producing it, and the reference splits it. The shortcut is
  gone (llama.cpp gates its equivalent on `tokenizer_ignore_merges`, which none
  of the Qwen rules set), and the synthetic unit fixtures now supply the merge
  ranks a real byte-level vocabulary has.
* **The Qwen3.5 GGUF's tokenizer is a different revision than the published
  `tokenizer.json`** (248 320 vs 248 044 tokens; the Devanagari merge ranks
  differ). The artifact the engine loads is therefore the only honest reference
  for token ids, so that fixture is generated by **llama.cpp** `llama-tokenize`
  on the same GGUF, while the pre-tokenizer *rule* stays gated against the
  `tokenizer.json` pattern. For Qwen2.5/Qwen3 the two references agree on all 52
  corpus entries, and the committed fixtures use transformers.
* **Load-time invariants replace silent degradation.** `Tokenizer::load` now
  refuses a `tokenizer.ggml.model` that is not `gpt2`, an empty merge table, and
  a vocabulary missing any of the 256 byte tokens — each of which used to turn
  into quietly wrong ids (the last one via `unwrap_or(0)`).

### 3.6 Input normalization: NFC where the family declares it (#132 S1, 2026-10-15; ADR-0044)

Whether the input is normalized is a property of the **pre-tokenizer rule**, because that is
where the reference `tokenizer.json` declares it and the GGUF carries no normalizer field.
`PreTokenizer::applies_nfc()` answers it, and `Tokenizer::encode` / `Tokenizer::encode_raw`
go through one `normalized()` seam that consults it: `qwen2`/`qwen35` declare
`"normalizer": {"type": "NFC"}` — verified 2026-10-15 for `Qwen/Qwen3-0.6B@c1899de2`,
`Qwen/Qwen2.5-0.5B-Instruct@main` and `unsloth/Qwen3.5-0.8B@23c69c53` — while `llama-bpe`
declares `"normalizer": null` (verified for `unsloth/Llama-3.2-1B-Instruct@5a8abab4` and the
family's other revisions), and llama.cpp applies none either. The normalizer belongs to the
tokenizer, not to a request shape, so the CLI, the chat path, the server and the DRY breaker
tails all get it; for a normalizing family the input is borrowed when it is already NFC, and
for the others it is never touched.

The data is **generated and committed**, not a dependency:
`scripts/gen_unicode_nfc.py` reads CPython's bundled `unicodedata` (Unicode 15.0.0) and
writes `src/tokenizer/nfc/tables.rs` (canonical decomposition, combining class,
composition and its second elements) plus the differential corpus
`tests/fixtures/tokenizer/nfc_vectors.json` (5 103 inputs in 8 named sections, each
answered by CPython). `src/tokenizer/nfc.rs` is the algorithm: canonical decomposition
(recursive, Hangul arithmetic), canonical ordering of each mark run, and canonical
composition with its blocked rule. NFC is **not** NFKC: compatibility forms (`U+FB01`,
fullwidth) are left alone. The generated constant and the fixture's `unicode_version`
are asserted to be the same data version, so an upgrade is a reviewable generator re-run.

**Divergence, deliberate and now one-directional.** For a `qwen2`/`qwen35` model llama.cpp
applies no normalizer, so on non-NFC input the two engines return different ids — minfer
follows the model's own `tokenizer.json`. For a `llama-bpe` model nothing is applied by
either engine, so they agree; the corpus for that family carries decomposed entries whose
reference ids are the decomposed ones. DRY-breaker *matching* stays byte-exact — a non-NFC
breaker matches nothing and is refused by name — while the tails are tokenized like any
other text (normalized only when the rule declares it).

Measured on the cached GGUFs (dgxspark, 2026-10-15): the five Qwen-family models have
`encode("café")` == `encode("cafe\u{0301}")` — `[924, 58858]` for Qwen2.5/Qwen3 and
`[895, 56868]` for Qwen3.5 — while `llama-3.2-1b-instruct` keeps them apart:
`[936, 59958]` composed, `[936, 1897, 54939]` decomposed, which is what both references
return.

## 4. Gates

| Gate | Where | Reference |
|---|---|---|
| Template byte-for-byte rendering, per model | `src/template.rs` unit test over `tests/fixtures/chat/*.json` | transformers 5.17.0 `apply_chat_template` |
| The fixture template is the model's own | `#[ignore]`d real-model test: GGUF `tokenizer.chat_template` == fixture template and rendering matches | the cached GGUF |
| Loud refusal | unit test: an unsupported method/syntax fails and the error **names** the construct; a mutation of the callback (dropping a method) must make a previously-passing case fail | — |
| Pre-tokenization split | `src/tokenizer.rs` unit test over `tests/fixtures/tokenizer/split_*.json` (4 values, 52-54 entries each) | the `tokenizer.json` regex evaluated by CPython `regex` |
| Token-id equality | `#[ignore]`d real-model test over `tests/fixtures/tokenizer/ids_*.json` (8 cached models, both engines) | transformers `AutoTokenizer.encode` (Qwen2.5/Qwen3) or llama.cpp `llama-tokenize` on the same GGUF (Qwen3.5, `llama-bpe`, `lfm2`, and the SentencePiece model — for SPM the HF file is a *BPE* rendering of the vocabulary, so llama.cpp is the only reference for this engine, ADR-0045); the two agree on all 52 corpus entries where both exist (measured for `llama-bpe` too, 2026-10-15: llama.cpp and HuggingFace `tokenizers` agree 52/52) |
| Engine dispatch | `src/tokenizer.rs` unit test `the_tokenizer_model_is_a_dispatch` (both accepted values, six refusals each naming its value) and the ignored gate's `tokenizer_model` field | `tokenizer.ggml.model` in the artifact |
| SentencePiece session | `src/tokenizer.rs` unit tests `spm_merges_by_score_and_falls_back_to_byte_tokens` and `spm_prefixes_the_first_segment_with_a_space` (synthetic vocabulary: the scored chain, the `▁` prefix on/off, a byte fallback, the post-special re-prefix) | llama.cpp `llm_tokenizer_spm_session` (`llama-vocab.cpp:114`) |
| NFC normalization | `src/tokenizer/nfc.rs` unit tests over `tests/fixtures/tokenizer/nfc_vectors.json` (5 103 inputs: every decomposition, every composition pair, all 784 ordered mark pairs both after a starter and alone, Hangul, compatibility forms, a seeded random corpus), plus the per-rule table (`the_normalizer_follows_the_pre_tokenizer_rule`), the two synthetic-tokenizer arms, and the real-model gate (a normalizing family: `café == cafe\u{0301}`; `llama-bpe`: the decomposed corpus entries) | CPython `unicodedata.normalize('NFC', …)` for the tables; each family's `tokenizer.json` normalizer field for the declaration |
| Grammar (F2) + pinned sampler (F3) | existing suites | unchanged |

Each new gate is mutation-checked (break the implementation, watch the gate
fail, revert) and the mutation result is recorded in the PR.

## Decisions governing this document

This page is the *current contract*; the decisions behind it are frozen in the ADR corpus:
- [ADR-0019](adr/0019-a-chat-template-that-cannot-render-refuses-the-load.md) — A chat template that cannot be rendered refuses the load
- [ADR-0042](adr/0042-the-tokenizer-applies-the-models-declared-nfc-normalizer.md) — The tokenizer applies the model's declared NFC normalizer
- [ADR-0043](adr/0043-a-special-token-is-declared-by-the-file-never-inferred-from-its-name.md) — A special token is declared by the file, never inferred from its name
- [ADR-0044](adr/0044-the-input-normalizer-follows-the-pre-tokenizer-family.md) — The input normalizer follows the pre-tokenizer family's declaration
- [ADR-0045](adr/0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md) — The tokenizer engine is a dispatch on `tokenizer.ggml.model`
