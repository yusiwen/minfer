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

### 2.3 Refused, and how the refusal is detected

A template is **refused loudly** — never silently re-rendered as ChatML — when:

1. `Environment::add_template` fails: syntax error, unknown filter or test, a
   construct minijinja does not parse. The error carries the template line.
2. Rendering fails: an unknown method (the callback returns
   `ErrorKind::UnknownMethod` for anything outside the table above), a value of
   the wrong type, or an undefined value in a position that requires one.

The refusal text names the construct and the line, e.g.

```
chat template uses an unsupported construct: unknown method 'splitlines'
(template line 41); minfer refuses to fall back to ChatML. Supported Python
str methods: capitalize, count, endswith, find, join, lower, replace, rfind,
rsplit, split, startswith, strip, lstrip, rstrip, title, upper.
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

### 3.1 `tokenizer.ggml.pre` is authoritative

The pre-tokenizer was a single hardcoded regex, and it was subtly wrong: it had
no `(?!\S)` arm (so `a  b` grouped the two interior spaces instead of splitting
off the first) and no `\s*[\r\n]+` arm. The rules now come from
`tokenizer.ggml.pre`:

| `pre` value (aliases) | Rule | Reference regex (from `tokenizer.json`) |
|---|---|---|
| `qwen2` (`deepseek-r1-qwen`) | Qwen2/Qwen2.5/Qwen3 | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` |
| `qwen35` | Qwen3.5 (letter runs also consume `\p{M}`) | same, with `[\p{L}\p{M}]+` and `[^\s\p{L}\p{M}\p{N}]` |
| `gpt2` (`gpt-2`) | classic GPT-2 | `'s\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)` |

Every other value — **including a missing key** — is a refused load:
`Tokenizer::load` returns `Err` naming the value and the supported set. A guess
here is exactly the "wrong split" the ticket forbids.

The splitter is hand-written (a port of llama.cpp's
`unicode_regex_split_custom_qwen2` / `_qwen35` / `_gpt2`) rather than a `regex`
crate pattern, for two reasons: the Rust `regex` crate has **no lookahead**, so
`(?!\S)` cannot be expressed; and a hand-written splitter makes the byte spans
explicit, which is what the byte-level BPE and the grammar mask (F2) consume.

### 3.2 Special and added tokens

Unchanged and still exact: GGUF `tokenizer.ggml.token_type` 3 (CONTROL) / 4
(USER_DEFINED) are matched as single ids *before* BPE, earliest position wins,
longest text wins at the same position, plus the `<|im_start|>` / `<|im_end|>` /
EOS fallbacks for GGUFs whose converter marked them type 1. This is what makes
Qwen's `<|im_start|>` and DeepSeek-R1's `<｜User｜>`/`<think>` single tokens.

### 3.3 Byte fallback replaces the silent id 0

`bpe_encode` used to emit token id `0` for any piece it could not look up — a
silent corruption. It now walks the piece byte by byte and emits the
corresponding single-byte token (byte-level BPE maps every UTF-8 byte to a
Unicode codepoint whose token exists in the vocabulary). `Tokenizer::load`
verifies that **all 256** byte tokens exist and refuses the load when they do
not, so the fallback cannot silently degrade; a lookup that still fails is an
invariant violation that aborts with the byte value.

### 3.4 Not covered (documented, refused, or a follow-up)

* **NFC normalization.** HF's `tokenizer.json` carries an NFC normalizer;
  llama.cpp does not apply one for BPE, and minfer does not either. The gate
  corpus is NFC-normalized and the difference is named in the honest scope; a
  normalization table is a follow-up (a Unicode data dependency).
* **SentencePiece / unigram / WordPiece** models (`tokenizer.ggml.model !=
  "gpt2"`) — out of scope, the tokenizer is a byte-level BPE.
* **`ignore_merges` pre-tokenizers** (`llama3`, `llama-bpe`, `tekken`, …) and
  **multi-regex** pre-tokenizers (`default`, `deepseek-llm`, `deepseek-coder`,
  `falcon`, `starcoder`, …): refused by name at load.
* **`byte_encode = false`** SPM-style BPE (`whitespace` pre-tokenizer) and the
  `<0xXX>` byte fallback for it: refused by name at load.

## 4. Gates

| Gate | Where | Reference |
|---|---|---|
| Template byte-for-byte rendering, per model | `src/template.rs` unit test over `tests/fixtures/chat/*.json` | transformers 5.17.0 `apply_chat_template` |
| The fixture template is the model's own | `#[ignore]`d real-model test: GGUF `tokenizer.chat_template` == fixture template and rendering matches | the cached GGUF |
| Loud refusal | unit test: an unsupported method/syntax fails and the error **names** the construct; a mutation of the callback (dropping a method) must make a previously-passing case fail | — |
| Pre-tokenization split | `src/tokenizer.rs` unit test over `tests/fixtures/tokenizer/split_*.json` | the `tokenizer.json` regex evaluated by CPython `regex` |
| Token-id equality | `#[ignore]`d real-model test over `tests/fixtures/tokenizer/ids_*.json` | transformers `AutoTokenizer.encode`, cross-checked with llama.cpp `llama-tokenize` |
| Grammar (F2) + pinned sampler (F3) | existing suites | unchanged |

Each new gate is mutation-checked (break the implementation, watch the gate
fail, revert) and the mutation result is recorded in the PR.
