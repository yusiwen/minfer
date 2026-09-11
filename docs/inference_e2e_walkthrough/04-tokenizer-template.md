# 04 · Tokenizer + chat template

> **Stage**: model dispatch + weights (03) → **tokenizer + template** → graph
> build (05). The weights are registered; nothing has run yet. This stage turns
> the string you typed into the integer token list that every later document
> operates on — the raw material of the whole rest of the series.
> **Code**: `src/tokenizer.rs` (`Tokenizer::load` :82, `encode` :280,
> `decode_bytes` :322), `src/template.rs` (`render_template` :120,
> `render_messages` :13), `src/main.rs` (template block :715-735,
> `get_chat_template` :1495, decode loop :900-918) — lines verified at commit
> `e7fa0da`.

## 1. Background — where this stage sits

Doc 03 left the engine with the GGUF memory-mapped and parsed, the
architecture dispatched (Qwen2 or Qwen3), and every weight tensor registered
in the compute-graph allocator — possibly on the GPU. But not one byte of your
prompt has been touched: you typed `"What is 2+2?"`, and the model, so far,
has no idea it exists.

This stage closes that gap, and it has two halves. First, the **chat template**:
a string, stored in the GGUF metadata, that describes how a conversation is
spelled out in the exact marker format the model was trained on — your raw
prompt is wrapped in that format before anything else happens. Second, the
**tokenizer**: the code that turns that wrapped text into a list of integers.

Those integers are called **token ids**. A *token* is the model's unit of text —
a short chunk such as `"What"`, `" is"`, or a single character — and each
distinct chunk the model knows has a number. The model cannot read characters at
all. Its very first layer is a lookup table (the embedding matrix) that maps the
integer `3838` to a vector of floats; integer `3837` maps to a different vector.
Feed it raw characters and there is simply no table entry — nothing downstream
can run. That is why this stage gates the entire series: the token list is the
input to graph build (05), the prefill forward (09), the sampler (12), and the
decode loop (13).

The template half matters just as much, and it is the more surprising one. A
chat model was not trained to continue arbitrary text; it was trained to answer
when it sees a very specific arrangement of marker strings like
`<|im_start|>user`. Get that arrangement wrong and a perfectly good model
produces garbage — it will happily continue your sentence instead of
answering it. The markers are not decoration; they are the protocol.

## 2. Principle — how it works and why

### 2.1 The stage in one picture

```
 prompt: "What is 2+2?"
     │
     ▼
 get_chat_template()          reads GGUF metadata key "tokenizer.chat_template"
     │                        (missing, or --no-template → use the raw prompt)
     ▼
 template::render_template()  minijinja renders with add_generation_prompt=true
     │                        (render error → fallback_chatml: hand-written ChatML)
     ▼
 "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
  <|im_start|>user\nWhat is 2+2?<|im_end|>\n
  <|im_start|>assistant\n"
     │
     ▼
 Tokenizer::encode()
     ├─ 1. special-token scan   whole strings like <|im_end|> become one id
     ├─ 2. regex pre-tokenize   split into word / number / punctuation pieces
     ├─ 3. byte-encode          map every raw byte to a printable unicode char
     └─ 4. greedy BPE merges    merge the adjacent pair with the lowest rank
     ▼
 Vec<u32>  [151644, 3838, 374, 220, 17, 10, 17, 30, 151645, 151648, 198]
     │
     └──► doc 05: graph build consumes these ids (positions 0,1,2,…)
```

### 2.2 Why does the model need a template at all?

Start with what a language model fundamentally does: given a sequence of
tokens, predict a probability for every token in the vocabulary of what comes
next. A **base model** (trained only on raw documents) uses this to continue
text: prompt it with `"The capital of France is"` and it predicts `" Paris"` —
or equally `" known"`, because continuing documents is its whole job.

A **chat model** is a base model that went through a second training phase
(usually called instruction tuning or alignment). The training data in that
phase was conversations, serialized in a fixed format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

Those `<|im_start|>` / `<|im_end|>` strings are **special tokens** — vocabulary
entries that were reserved during training and shown to the model millions of
times as turn boundaries. The model's chat behavior lives entirely in that
format: during alignment training, every example was markers, a user turn,
`<|im_start|>assistant`, then an answer — so the model learned the conditional
distribution *"text that follows `<|im_start|>assistant`"*: answers, not
continuations.

Now the punchline about the last line. The rendered prompt ends with
`<|im_start|>assistant\n` — an *empty* assistant turn opener. This is what the
flag `add_generation_prompt` controls, and it is not optional. With the
opener, the model's next-token distribution is "the first token of an
assistant answer", and it says `"2+2 equals 4..."`. Without it, the prompt
ends inside the user turn, and the model keeps writing the user turn — more
question text, or a stray `<|im_end|>`. It will not answer.

So the template is not a display nicety; it selects *which distribution* the
model samples from. minfer always passes `add_generation_prompt=true` on the
CLI path (`src/main.rs:724`), because a one-shot prompt is by definition a
"generate the assistant's next turn" request.

One more piece: **bos** and **eos**. *bos* (beginning of sequence) is a token
some model families expect at the very start of every input; *eos* (end of
sequence) is the token the model was trained to emit when it is done talking.
The template context exposes `bos_token` as a variable (`src/template.rs:44`);
whether a bos marker appears is the template string's choice, not the
engine's. For eos, minfer does not rely on the template — the GGUF metadata
carries `tokenizer.ggml.eos_token_id` and `<|im_end|>`'s id directly (§2.5),
and the decode loop treats them as stop signals.

### 2.3 The tokenizer: byte-level BPE, end to end

**BPE** (Byte Pair Encoding) is the algorithm that decided which chunks of text
become tokens. It was run once, months before you ever run inference, on a huge
training corpus:

1. Start with every single byte as its own token (256 of them).
2. Count which pair of adjacent tokens occurs most often in the corpus; merge
   that pair into a new token; repeat. Each merge gets a **merge rank** — the
   order number in which it was learned. Rank 0 was learned first, i.e. it was
   the most frequent pair in the whole corpus.
3. Stop when the vocabulary reaches its target size — for Qwen models,
   151,936 entries (`docs/QWEN3-SUPPORT-PLAN.md:37`).

The training-time result ships inside the model file: the GGUF metadata carries
the token strings (`tokenizer.ggml.tokens`), the merge list in rank order
(`tokenizer.ggml.merges`), and the special-token types. Encoding is simply
*replaying* those merges on your text.

A first example — byte-exact, because it comes straight from minfer's test
suite, whose expected ids were cross-checked against llama.cpp
(`src/tokenizer.rs:537`):

```
text:    <｜User｜>What is 2+2?<｜Assistant｜><think>\n
ids:     151644 3838 374 220 17 10 17 30 151645 151648 198
```

| id | stored token text | what a human sees | how it was produced |
|---|---|---|---|
| 151644 | `<｜User｜>` | (role marker) | special token, matched whole, before BPE |
| 3838 | `What` | `What` | regex piece, whole string already in vocab |
| 374 | `Ġis` | `␣is` | regex piece `" is"`; already a single vocab entry |
| 220 | `Ġ` | `␣` | regex `\s+` piece — the lone space before a digit |
| 17 | `2` | `2` | regex `\p{N}` piece — **one digit only** |
| 10 | `+` | `+` | regex punctuation piece |
| 151645 | `<｜Assistant｜>` | (role marker) | special token |
| 151648 | `<think>` | (reasoning marker) | special token |
| 198 | `Ċ` | newline | regex `\s*[\r\n]+` piece |

Two oddities in that table are the regex at work. The GPT-2 pre-tokenization
regex (borrowed verbatim, `src/tokenizer.rs:259-262`) splits text into word
pieces, single digits, and punctuation runs *before* any merging happens —
merges can never cross a piece boundary. Digits are matched one at a time
(`\p{N}` matches exactly one), which is why `2+2` costs four tokens and why
models are famously weak at long arithmetic: every digit is a separate
concept. And a space before a digit attaches to nothing (the word rule only
glues a leading space to *letters*), so it becomes a bare `Ġ` — token 220.

Now the greedy merge loop itself. For a piece that is not already one vocab
entry, minfer splits it into characters and repeatedly merges the adjacent
pair with the *lowest* rank. A toy illustration (invented ranks): if the piece
is `[m][i][n][f][e][r]` and `("i","n")` has rank 88 while every other adjacent
pair ranks higher, `[i][n]` fuses first; the scan then repeats on the shorter
list until no adjacent pair is in the merge table, and each surviving piece is
looked up in the vocab. The real ranks live in the GGUF; the real loop is
excerpted in §3.2.3.

Why lowest rank first, and why does that give good tokenizations? Because rank
order *is* frequency order from training. The first merges ever learned were
the most common byte pairs; later merges built on earlier ones. Replaying
lowest-rank-first reconstructs the same segmentation the vocabulary was built
for, so your text is cut exactly the way the model saw text cut during
training. The practical payoff is compression: common words were merged
thousands of merges ago and exist as single tokens, so `"the"` costs one
position instead of three. That matters downstream because *every* cost in
this engine scales with token count — prefill matmuls, KV cache size (each
token reserves a K and V row per layer), and decode latency per generated
token.

**Why byte-level?** Because the alphabet is bytes, not characters. Before
merging, every raw byte 0–255 is mapped to a printable unicode character
(`build_byte_to_unicode`, `src/tokenizer.rs:8-37`; printable ASCII and most
Latin-1 map to themselves, the rest get chars from code point 256 upward —
space becomes `Ġ`, newline `Ċ`), and every vocab entry is stored in that
mapped form. The consequence: *any* byte string round-trips — Chinese, emoji,
binary junk — and decode (§2.4) can always invert the mapping exactly. A
character-level tokenizer cannot make that promise: a character the vocabulary
never saw has no representation at all.

### 2.4 Decode: ids back to bytes, and why bytes and not a String

Generation runs the same table backwards. `decode_bytes` concatenates
`id_to_token[id]` for each id — producing the mapped-form text, e.g. `Ġis` —
then maps every character back to its raw byte via the reverse table
(`unicode_to_byte`). Out come raw bytes, exactly the bytes that were encoded.

Why insist on bytes rather than a Rust `String`? Because a multi-byte UTF-8
character can be split across two tokens. Consider the Chinese character 中
(U+4E2D), whose UTF-8 encoding is the three bytes `E4 B8 AD`; minfer's test
(`src/tokenizer.rs:427-464`) contains a token whose mapped text is `ä¸­` — the
mapped forms of exactly those three bytes. If the model emits the first two
bytes of the character in one token and the third in the next, a per-token
`String::from_utf8_lossy` conversion would stamp a `�` (U+FFFD replacement
character) into your output stream *permanently* — the bytes were already
thrown away. `decode_bytes` never attempts the conversion: it emits raw
bytes, so `E4 B8` + `AD` reassembles perfectly wherever they land. The tests
pin this: `decode_bytes_keeps_multibyte_bytes` (`src/tokenizer.rs:458`).

### 2.5 Special tokens: ids with a job

A **special token** is a vocabulary entry that is not a piece of human text but
a control signal: `<|im_start|>`, `<|im_end|>`, `<think>`, `<｜User｜>`, and so
on. In the GGUF they are flagged by `tokenizer.ggml.token_type` — the values
the code checks are 3 (control) and 4 (user-defined) (`src/tokenizer.rs:138-139`).

They get special treatment at both ends of the pipeline:

- **Encode**: a special token must survive as one id — the BPE machinery would
  otherwise shred `<|im_end|>` into ordinary character pieces. minfer scans
  for special-token strings *before* running BPE on each segment
  (`src/tokenizer.rs:280-314`), matching the earliest position first and the
  longest string at a given position. This is not cosmetic: DeepSeek-R1-style
  markers `<｜User｜>` use fullwidth unicode bars that the GPT-2 regex would
  happily split apart; the regression test at :533 keeps them intact.
- **Decode/generate**: the ids of eos and `<|im_end|>` are handed to the
  generation loop as *stop sentinels* — when the sampler produces one, the
  engine stops instead of appending it. They are also fed into the sampler's
  penalty window (`src/main.rs:847`, doc 12). The ids come from
  `ModelDef::special_tokens()` (doc 03), sourced from GGUF metadata:
  `tokenizer.ggml.eos_token_id`, plus a lookup of `<|im_end|>` that falls back
  to the eos id (`src/models/qwen2/loader.rs:130-131`). One vocabulary, two
  directions, and a set of reserved ids that act as the protocol's
  punctuation.

## 3. Implementation

### 3.1 Data in / data out

**Input data — GGUF metadata** (parsed in doc 02; the tokenizer reads it via
`GgufContext`, `src/tokenizer.rs:82`):

| GGUF key | Type | Lands in |
|---|---|---|
| `tokenizer.ggml.tokens` | string array (~151,936 entries for Qwen) | `id_to_token: Vec<String>`, inverted into `vocab: HashMap<String,u32>` |
| `tokenizer.ggml.scores` | f32 array | `id_to_score` (loaded for llama.cpp parity, unused) |
| `tokenizer.ggml.token_type` | i32 array (1 normal, 3 control, 4 user-defined) | `id_to_type`, drives the special-token table |
| `tokenizer.ggml.merges` | string array `"A B"` per merge, in rank order | `merges: HashMap<(String,String), usize>` — pair → rank |
| `tokenizer.ggml.bos_token_id` / `eos_token_id` | u32 | `bos_token` / `eos_token` |
| `tokenizer.chat_template` | one long string | passed to minijinja verbatim |

Note what is *not* here: no tokenizer model file, no external vocabulary. The
vocab ships inside the GGUF because the model file already had to describe its
own output layer (`output.weight` is `[n_embd, 151936]` — the vocabulary size
is baked into the weight shape), so the conversion tool writes the matching
token table alongside it.

The flow is: `&str` prompt + metadata → rendered `String` → `Vec<u32>` →
`ctx = max(--n-ctx, ids.len())` (`src/main.rs:749`), which sizes the persistent
KV regions once for the whole run → `forward(&ids, positions 0..n)` (doc 05+).
During generation the direction reverses: one sampled id per step →
`decode_bytes` → raw bytes → stdout/SSE. The template's token cost is real
memory: the rendered wrapper becomes part of the prompt, and the prompt length
feeds `ctx` — a template that bloats the prompt bloats the KV allocation.

### 3.2 Key code

#### 3.2.1 Template selection and rendering (CLI path)

The whole template stage in `main.rs` is deliberately small — read the template
out of metadata, render, encode:

```rust
// src/main.rs:715-735
// === Chat template (need tokenizer for bos_token text) ===
let processed = if no_template {
    prompt.clone()
} else if let Some(tmpl) = get_chat_template(&gguf_model.parts[0].data) {
    let bos_text = tokenizer
        .id_to_token
        .get(tokenizer.bos_token as usize)
        .map(|s| s.as_str())
        .unwrap_or("");
    template::render_template(&tmpl, &prompt, true, bos_text)
} else {
    prompt.clone()
};
#[cfg(feature = "debug_dump")]
crate::dump::maybe_dump_text("minfer_dump_prompt", &processed);
let input_ids = tokenizer.encode(&processed);
if input_ids.is_empty() {
    eprintln!("tokenize failed");
    std::process::exit(1);
}
println!("Prompt: {} tokens", input_ids.len());
```

Three branches, in priority order: `--no-template` bypasses everything and
tokenizes the raw prompt (useful for base models and for comparing token
counts); otherwise `get_chat_template` pulls `tokenizer.chat_template` from
the GGUF metadata bytes (a tiny re-parse of metadata only —
`src/main.rs:1495-1504`); with no template key at all, the raw prompt is used
as-is. The literal `true` argument to `render_template` is
`add_generation_prompt` — §2.2 explained why it must always be on for a
one-shot prompt. If the tokenizer produces zero ids, the run aborts: an empty
token list would leave the graph builder with no tokens to embed.

The renderer wraps minijinja, a small Jinja-compatible template engine (the
single `user` message is built as JSON at `src/template.rs:138-141`):

```rust
// src/template.rs:126-159 (signature at :120-125)
let mut env = Environment::new();

// Register the template
if env.add_template("chat", template).is_err() {
    eprintln!("Warning: invalid chat template, falling back to ChatML");
    return fallback_chatml(user_input, add_generation_prompt);
}
let tmpl = match env.get_template("chat") {
    Ok(t) => t,
    Err(_) => return fallback_chatml(user_input, add_generation_prompt),
};

let messages = vec![serde_json::json!({
    "role": "user",
    "content": user_input,
})];

let result = tmpl.render(context! {
    messages => messages,
    add_generation_prompt => add_generation_prompt,
    bos_token => bos_token,
    tools => minijinja::Value::UNDEFINED,
});

match result {
    Ok(s) => s,
    Err(e) => {
        eprintln!(
            "Warning: chat template rendering failed ({}), falling back to ChatML",
            e
        );
        fallback_chatml(user_input, add_generation_prompt)
    }
}
```

The context exposes exactly what real chat templates expect: `messages` (here,
the single user turn), `add_generation_prompt`, `bos_token`, and `tools` as
undefined so `{% if tools %}` branches don't crash. There are two independent
fallback triggers: the template string can fail to *parse* (`add_template`), or
parse and then fail at *render* time (a runtime error inside the template).
Both land on the hand-written ChatML fallback that produces the literal text
shown in §2.2 — system turn, user turn, then the empty assistant opener:

```rust
// src/template.rs:183-192
/// Fallback: simple ChatML format (CLI path, single user message)
fn fallback_chatml(user_input: &str, add_generation_prompt: bool) -> String {
    let mut r = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n",
        DEFAULT_SYSTEM, user_input,
    );
    if add_generation_prompt {
        r.push_str("<|im_start|>assistant\n");
    }
    r
}
```

`ChatML` is the marker convention Qwen models are trained on and the de-facto
lingua franca of chat templates, which is why a ChatML fallback is *usually*
compatible even for models whose real template failed. The multi-turn variant
`fallback_chatml_messages` (`src/template.rs:165-180`) loops over all messages
instead of one user string; the server and conversation paths reach it via
`render_messages` (`src/template.rs:13-58`).

#### 3.2.2 Loading the tokenizer from GGUF metadata

`Tokenizer::load` walks the metadata key-value list once per data kind. The
token strings become the `id_to_token` vector and an inverted `vocab` map
(`:114-118`); special-token ids and types fill `special_tokens`
(`:135-143`) plus `bos_token` / `eos_token` / `im_end` (`:145-149`). BPE's
data structure is built at `:120-133`: for each `tokenizer.ggml.merges`
entry — a string like `"Ġ t"`, two space-separated halves — the code splits on
the *first* space and inserts `merges[(first, second)] = i`, where `i` is the
array index. That index *is* the merge rank, because converters write merges
in the order they were learned. Splitting on the first space is enough
because each half is one byte-encoded string with no literal spaces in it
(spaces were mapped to `Ġ` precisely so they could never appear inside a
half).

Special tokens need one more data structure, and its comment explains the
invariant:

```rust
// src/tokenizer.rs:155-157, 172-179 (the <|im_start|>/<|im_end|>/eos
// fallback inserts between, :158-171, are described in the text below)
// Merge GGUF special tokens (type 3/4) with hardcoded fallbacks, then
// group by first char with longest-first ordering inside each group
// (an earliest-position, longest-match scan needs both).
let mut special_by_first: HashMap<char, Vec<(String, u32)>> = HashMap::new();
for (pat, id) in merged {
    let first = pat.chars().next().unwrap_or('\0');
    special_by_first.entry(first).or_default().push((pat, id));
}
for group in special_by_first.values_mut() {
    group.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
}
```

The skipped middle (:158-171) starts from `special_tokens.clone()` and
defensively inserts `<|im_start|>`, `<|im_end|>`, and the eos token *only when
the GGUF did not already provide them* (`contains_key` guards): some converted
models mark their specials as ordinary type-1 tokens, so minfer hardcodes the
ChatML markers as fallbacks, and real metadata always wins. The resulting
`special_by_first` index buckets patterns by their first character; the encode
scan (next) will jump straight to the bucket for the character it is looking
at instead of testing every pattern against every position.

#### 3.2.3 Encode: specials first, then regex, then merges

The top-level encode is a loop over "segments": text up to the next special
token goes through BPE, the special token becomes a single id, repeat
(`src/tokenizer.rs:280-310`, doc comment at :273-279):

```rust
// src/tokenizer.rs:284-308 (fn head at :280-283, final `result` at :309-310)
loop {
    // Find the earliest position where any special token starts.
    let mut earliest: Option<(usize, u32, usize)> = None; // (byte_pos, id, byte_len)
    'scan: for (ci, ch) in remaining.char_indices() {
        if let Some(group) = self.special_by_first.get(&ch) {
            let rest = &remaining[ci..];
            for (pat, id) in group {
                if rest.starts_with(pat.as_str()) {
                    earliest = Some((ci, *id, pat.len()));
                    break 'scan; // group is longest-first; earliest char wins
                }
            }
        }
    }

    if let Some((pos, id, len)) = earliest {
        // Encode text before the special token
        if pos > 0 {
            result.extend(self.encode_bpe(&remaining[..pos]));
        }
        result.push(id);
        remaining = &remaining[pos + len..];
    } else {
        // No more special tokens, encode the rest
        result.extend(self.encode_bpe(remaining));
        break;
    }
}
```

The double ordering matters: the outer scan takes the first character that
starts *any* special token ("earliest position wins"); within one position,
the bucket is sorted longest-first, so the first `starts_with` hit is the
longest match ("`<think▁begin｜>` beats `<think>`"). The dedicated tests
`special_token_earliest_position_wins` (:546) and
`longest_special_token_wins_at_same_position` (:562) pin both rules.

Inside a segment, `encode_bpe` (`src/tokenizer.rs:258-271`) runs the GPT-2
pre-tokenization regex — copied verbatim "from llama-vocab.cpp /
gpt2_tokenizer.py" — over the text:

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+
```

The alternation, read left to right: contractions (`'s`, `'t`, `'re`…), an
optional leading space glued to a run of letters, a single digit, an optional
leading space glued to punctuation, line breaks, and any other whitespace.
Each regex match becomes one *piece*: `byte_encode` (`:40-46`, the per-byte
character mapping from §2.3) maps its bytes to printable chars, and the piece
goes to `bpe_encode`. These piece boundaries are sacred — merges never cross
them — which is why `" is"` and `"2"` never fuse into one token no matter what
the merge table says.

Then the merge loop itself:

```rust
// src/tokenizer.rs:226-249 (whole-piece shortcut at :218-221, lookup tail at :251-254)
loop {
    // Find the best merge (lowest rank)
    let mut best_rank: Option<usize> = None;
    let mut best_idx: Option<usize> = None;

    for i in 0..word.len().saturating_sub(1) {
        let pair = (word[i].clone(), word[i + 1].clone());
        if let Some(&rank) = self.merges.get(&pair) {
            if best_rank.is_none() || rank < best_rank.unwrap() {
                best_rank = Some(rank);
                best_idx = Some(i);
            }
        }
    }

    if best_idx.is_none() {
        break;
    }

    // Merge at best_idx
    let idx = best_idx.unwrap();
    let merged = format!("{}{}", word[idx], word[idx + 1]);
    word.splice(idx..=idx + 1, std::iter::once(merged));
}
```

Read it as: loop {scan every adjacent pair, keep the lowest-rank one, splice
it}, until no pair is in the merge table. Before the loop there is a shortcut
— if the whole piece is already a vocab entry, return it directly
(`:218-221`), the overwhelmingly common case for real words — and after it,
each surviving piece is looked up in the vocab (`:251-254`). The tail lookup
has one trap worth naming: a final piece that is somehow in neither the merge
output nor the vocab maps to id 0 rather than erroring (`unwrap_or(0)`,
`:253`) — see §3.4. Complexity is O(pieces²) per word with tiny constants;
tokenization runs once per prompt, so it is not a hot path.

#### 3.2.4 Decode: ids → bytes → streamed text

```rust
// src/tokenizer.rs:322-345
pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
    let mut encoded = String::new();
    for &id in ids {
        if (id as usize) < self.id_to_token.len() {
            let token = &self.id_to_token[id as usize];
            encoded.push_str(token);
        }
    }

    // Reverse byte-level encoding
    let mut result = Vec::new();
    for c in encoded.chars() {
        if let Some(&b) = self.unicode_to_byte.get(&c) {
            result.push(b);
        } else {
            // Fallback: encode the char as UTF-8
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            result.extend_from_slice(s.as_bytes());
        }
    }

    result
}
```

Two quiet robustness choices: out-of-range ids are skipped — a corrupted
sample cannot panic the stream (tested at `:466-471`) — and characters *not*
in the reverse byte map (vocab entries holding genuine unicode text rather
than byte-mapped forms) pass through as their UTF-8 bytes (`:336-341`).

The streaming holdback, used by the server and conversation paths, is
`complete_utf8_prefix_len` (`src/tokenizer.rs:377-399`). Its doc comment says
it "mirrors llama.cpp's `format_incomplete_utf8` holdback", and the mechanism
is just the UTF-8 length grammar walked once: the lead byte of a sequence
determines its length (1 for ASCII, 2–4 for multi-byte, judged by the
`0xC0`/`0xE0`/`0xF0` masks on the top bits), so the function scans forward
until the next character would run past the end of the buffer, and returns the
offset where the complete prefix ends — everything from there onward waits for
the next token. The callers wire it into per-step emission: `server/chat.rs:162`
appends newly decoded bytes to a full accumulator and `server/chat.rs:176`
flushes up to `emitted + complete_utf8_prefix_len(&full[emitted..])`;
`conversation.rs:548/564` does the same for the REPL. (The CLI decode loop
instead writes raw bytes straight to stdout and lets the terminal assemble the
glyph; `complete_utf8_prefix_len_holds_incomplete_trailing` at `:474` pins the
grammar.)

And the consumer end — how the ids the sampler produces meet this decoder in
the CLI decode loop (doc 13 walks the loop; here, only the tokenizer-relevant
lines). The sampled id first runs the stop-sentinel check, then is recorded
for the penalty window (`generated.push` / `prev_tokens.push`,
`:903-907`):

```rust
// src/main.rs:900-902, 909-918
if is_stop_token(sampled.token_id, &special) {
    break;
}

// Stop-string detection on the FULL byte stream before emitting.
full.extend_from_slice(&tokenizer.decode_bytes(&[sampled.token_id]));
if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
    full.truncate(cut);
    if cut > emitted {
        hi.feed(&full[emitted..]);
        emitted = full.len();
    }
    break;
}
```

`special` came from `model.special_tokens()` (`src/main.rs:839`), and
`is_stop_token` (`src/main.rs:1020-1022`) is the two-line sentinel check
`id == special.eos || Some(id) == special.im_end` — the `SpecialTokens` struct
itself is two fields (`eos`, `im_end: Option<u32>`, `src/models/mod.rs:88-91`).
Note the byte-accumulation pattern: decoded bytes go into `full` *before*
emission (the rest of the loop, `:919-922`, flushes newly completed bytes), so
a stop string (`--stop "Let me think"`) that straddles two tokens is caught in
the accumulated stream — the same byte-first philosophy as §2.4. The tokens
handed to `prev_tokens` also feed the sampler's repeat-penalty window (doc
12), so prompt *and* generated token ids influence penalties from the very
first step (`src/main.rs:847`).

### 3.3 Design choices (why this shape and not another)

**Why a chat template at all — why not just tokenize the prompt?**
Because tokenization is the wrong layer for chat behavior. The model was
aligned on a marker protocol; the only way to reach its "answer mode" is to
reproduce that protocol byte-for-byte at the input (§2.2). This gets chat
behavior out of the same weights by changing only the prompt text, where
separate per-mode models or hidden role channels would multiply the model. The
cost: template *rendering* is a compatibility surface (§3.4's minijinja
gotcha), and template *correctness* is invisible until the model answers wrong
— hence the debug dump (§4) and the `--no-template` bypass.

**Why self-contained BPE instead of a tokenizer crate?**
The obvious alternative is a dependency — `tokenizers` (HuggingFace) or
`tiktoken` — bringing a large dependency tree and its own version skew.
minfer's constraint is zero ML framework deps (ARCHITECTURE.md §1: five crates
total, `minijinja` the newest). The decisive fact is that the tokenizer's
*data* already ships in the GGUF — tokens, scores, types, merges,
special-token flags — so a crate would mostly re-read the same tables and add
a second source of truth; the algorithm itself is ~80 lines (§3.2.3). The
honest trade-off is exactness risk: BPE implementations differ in
pre-tokenization details, and a divergence silently changes every id. minfer
buys that risk down with llama.cpp-parity tests —
`special_tokens_match_as_single_ids_before_bpe` hardcodes ids copied from
llama.cpp's tokenizer as the expected output (`src/tokenizer.rs:537-541`), so
any divergence fails CI rather than shipping as subtly different model
behavior.

**Why greedy lowest-rank merging?**
Because the merge table *is* a frequency-ordered construction history, and
lowest-rank-first replay is the inverse operation (§2.3): it reproduces the
segmentation the model was trained on, with no search — one linear scan per
merge round. The alternatives are worse on both axes: longest-first or
highest-rank-first produce segmentations the model never saw (the pieces
would still be *valid* tokens, but the embedding each maps to was trained on
different contexts — quality quietly degrades), and optimal segmentation
search (minimize token count, e.g. Viterbi) costs orders of magnitude more.
The compression payoff is concrete: single-token `" is"` costs one position
instead of three — one fewer row through every attention head and one fewer
KV row per layer, multiplied by every layer.

**Why `add_generation_prompt=true` (and hard-coded on the CLI path)?**
The rendered prompt must end with the empty assistant-turn opener
(`<|im_start|>assistant\n`), or the model's next-token distribution is
"continue whatever turn is open" — usually a continuation of the user's own
text (§2.2). It is hard-coded `true` on the CLI path (`src/main.rs:724`)
because a one-shot CLI prompt is definitionally a "start the assistant's turn"
request. The multi-turn paths pass it explicitly too — and *only* on the final
render: `format_single` (`src/template.rs:77-116`), the incremental renderer
behind `--cnv`, renders the recorded past with `false` and only the new state
with `true`, then diffs the two strings so the KV cache is appended with just
the delta — a mid-history render must not append an opener, or the KV would
contain an assistant header that never led to an answer.

**Why byte-level (and why decode in bytes)?**
Byte-level BPE gives total input coverage (§2.3) and byte decode gives
lossless streaming (§2.4). A string-oriented decoder would corrupt output
precisely in the cases that matter — emoji, CJK — and permanently:
`from_utf8_lossy` cannot be undone. The design keeps lossy conversion strictly
at the presentation edge (the doc comment on `decode`,
`src/tokenizer.rs:347-352`, says streaming paths must use `decode_bytes`),
never in the data path.

**Why match special tokens before BPE, with earliest-then-longest rules?**
Special tokens are protocol punctuation; letting BPE see them destroys their
meaning (and with R1-style fullwidth markers, the regex pieces can never
recombine into the special string — the test comment at
`src/tokenizer.rs:521-522` says exactly this). Earliest-position-wins matches
how a human reads: the leftmost marker is the next structural event.
Longest-at-position-wins disambiguates prefixes (`<think` vs
`<think▁begin｜>`); any other priority would be arbitrary.

### 3.4 Pitfalls & invariants

**The minijinja 2.21 gotcha — Qwen3's template always falls back.** minfer
uses `minijinja = "2"` with `default-features = false` (`Cargo.toml:15`). In
minijinja 2.21, template strings are Rust strings and expose **no** `str`
methods and no `lstrip`/`rstrip`/`strip`/`contains` filters — string work must
be Jinja *filters*. Qwen3's shipped `chat_template` uses Python method syntax
(`message.content.split('</think>')`, `.lstrip('\n')`), so it fails at render
time with `unknown method: string has no method named split` and
`render_messages` falls back to ChatML (`docs/QWEN3-SUPPORT-PLAN.md:317-334`,
gotcha #9). The consequences are precise: plain chat still works (the
fallback's `<|im_start|>` markers are Qwen3-compatible, and the model still
emits `<think>` blocks), but think-block extraction, tool-call formatting, and
`enable_thinking` handling are lost — the fallback feeds `<think>` content
back verbatim on the next turn. Watch for the stderr warning
`chat template rendering failed …, falling back to ChatML`: it means the
fallback ran, not the model's own template.

**Specials must never reach BPE.** The whole-string scan happens before
`encode_bpe`, and the regression test exists because the R1 template broke
otherwise. Invariant: a new special-token source must join
`merged`/`special_by_first` *before* `encode` runs.

**Unknown pieces silently map to id 0.** `bpe_encode`'s tail
(`unwrap_or(0)`, `src/tokenizer.rs:253`) means a vocab/merges inconsistency
yields the vocabulary's first entry instead of an error. Symptom: one word of
output is consistently garbage. The empty-encode guard in `main.rs`
(:731-734) catches the louder failure (nothing encoded at all).

**Byte-decode invariant: no lossy conversion in the streaming path.** The
lossy `decode` (`src/tokenizer.rs:354-356`) exists for tests only — its doc
comment says so explicitly. Streaming paths must pair `decode_bytes` with
`complete_utf8_prefix_len`, or multi-byte characters split across tokens become
permanent U+FFFD in the transcript.

**Template and conversation modes are coupled.** `--cnv` refuses
`--no-template` (`src/main.rs:529-532`) because the conversation session's
append-only KV scheme *requires* template rendering to compute what the next
turn appends.

**Template output feeds KV sizing.** `ctx = max(n_ctx, prompt_len)`
(`src/main.rs:749`): the rendered prompt's token count participates in sizing
the persistent KV regions (doc 07). A runaway template (e.g. one that
duplicates history) does not just slow prefill — it changes the allocation.

## 4. Observe & verify

- **The two printed counts.** Every CLI run prints `Vocabulary: 151936 tokens`
  (`src/main.rs:661` — the number for Qwen-family models) and then
  `Prompt: {} tokens` (:735). Run the same prompt with and without
  `--no-template`: the difference is exactly the boilerplate the template
  added (system turn, role markers, the assistant opener).
- **See the rendered prompt.** Build with `--features debug_dump` and set
  `MINFER_DUMP_DIR`: `crate::dump::maybe_dump_text("minfer_dump_prompt", …)`
  (`src/main.rs:728-729`) writes the post-template, pre-tokenization string —
  the literal `<|im_start|>…<|im_end|>…<|im_start|>assistant` text of §2.2.
  Format reference: `docs/debug-dump.md`.
- **Unit tests are the fastest oracle — and the llama.cpp cross-check.**
  `cargo test tokenizer::` covers the byte round-trip
  (`decode_bytes_reverses_byte_encoding`, the CJK
  `decode_bytes_keeps_multibyte_bytes`), the holdback grammar
  (`complete_utf8_prefix_len_holds_incomplete_trailing`), and all three
  special-token rules — including the id list copied from llama.cpp
  (`src/tokenizer.rs:537`), so any id-shifting change fails CI before it can
  shift model behavior. `cargo test template::` covers rendering, both
  fallbacks, and the incremental `format_single` diff semantics
  (`format_single_diffs_only_new_user_message` renders the real Qwen ChatML
  template shape, `src/template.rs:268`).
- **Per-token text in traces and the loud fallback.** With
  `MINFER_TRACE=<path>`, the decode loop attaches each sampled token's decoded
  text to the trace (`crate::trace::set_token`, `src/main.rs:926-930`) — handy
  for spotting id-0 garbage from §3.4. And template parse/render failures
  print `Warning: invalid chat template, falling back to ChatML` or
  `Warning: chat template rendering failed (…) …` on stderr before inference
  starts — if you see it, the model's own template is not what ran.

## 5. Cross-references

- [02 — GGUF load](02-gguf-load.md): where `tokenizer.ggml.*` and
  `tokenizer.chat_template` come from — this stage is a metadata consumer.
- [03 — Model dispatch and weights](03-model-dispatch-weights.md): provides
  `ModelDef::special_tokens()` (`eos`, `im_end`) and the weights this stage's
  ids will drive.
- [05 — Graph build: the IR and the builder](05-graph-builder-ir.md): consumes
  the token list; `GraphParams` and the KV sizing that `ctx =
  max(n_ctx, prompt_len)` feeds.
- [09 — Prefill forward path](09-prefill-forward-path.md): the ids become the
  embedding lookup rows with positions `0..len`.
- [12 — Sampler](12-sampler.md): stop sentinels and the penalty window that
  receives prompt + generated ids; stop-string byte matching from §3.2.4.
- [13 — Decode loop + graph reuse](13-decode-loop-graph-reuse.md): the loop
  whose per-token `decode_bytes` + holdback streaming this doc set up; also the
  multi-turn path where `format_single` renders only the appended turn.
- `docs/QWEN3-SUPPORT-PLAN.md` §5 #9: the full minijinja 2.21 record — the
  exact template lines that fail and the fallback consequences.
- `docs/OPENAI-CHAT-API-PLAN.md` and `docs/CLI-CONVERSATION-PLAN.md`: the
  server-side template handling (`render_messages`, `tools`) and the
  incremental-render design (`format_single`) behind multi-turn sessions.
- `docs/USAGE.md`: every flag this stage reads (`--no-template`, `--stop`,
  `--cnv`). `docs/GLOSSARY.md`: backstop definitions (BPE, merge rank, ChatML,
  bos/eos).
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) §3: the flowchart rows this doc
  expands (template render → tokenize → prefill).

← [03 — Model dispatch and weights](03-model-dispatch-weights.md) · [Index](./README.md) · [05 — Graph build: the IR and the builder](05-graph-builder-ir.md) →
