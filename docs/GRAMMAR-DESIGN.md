# Grammar and JSON-schema constrained decoding (F2, [#47](https://github.com/yusiwen/minfer/issues/47)) — design

**Status: implemented (2026-09-24)** — see §9 for the two defects the gates
found and the measured mask cost. This document is the contract the
implementation was held to: the exact grammar and schema subsets, the loud refusals, the
automaton, where the mask sits in the sampler pipeline, and what the acceptance
gates assert. Anything not listed as *in scope* is refused with a message that
names the construct — never silently guessed, never silently ignored.

Ticket record: `docs/ARCHITECTURE-EXECUTION-PLAN.md` (Phase F, F2). Gaps:
`docs/ARCHITECTURE-ROADMAP.md` §2.7 / item 15.

## 1. What is being built

One module, `src/grammar.rs`, with two front ends and one engine:

```text
GBNF text ─────────────┐
                       ├─► rule programs (bytecode) ─► pushdown automaton ─► token mask
JSON Schema (JSON) ────┘        (compiled once per request)                  + state advance
```

* **GBNF parser** for the subset in §2.
* **JSON-Schema front end** that *emits GBNF text* and feeds it through the same
  parser, so a schema and a hand-written grammar share one engine, one refuser
  and one set of gates. The generated GBNF is kept on the compiled object and
  can be printed (`Grammar::source`) for diagnosis.
* **Pushdown automaton** over Unicode codepoints, with an explicit call stack
  (`Vec<Frame>`), a set of nondeterministic stacks, and a carried
  partial-UTF-8 buffer — llama.cpp's model (its `llama_grammar_stack` is the same
  vector of `{rule, pos}` frames).
* **Token mask** computed per *state* and cached per state, so the per-step cost
  is O(vocabulary) at most once per distinct state, not per token.

## 2. GBNF subset (in scope)

| Construct | Syntax | Notes |
|---|---|---|
| rules | `name ::= body` | names `[A-Za-z_][A-Za-z0-9_-]*`; a rule named `root` is required |
| rule references | `name` | undefined names are a compile error |
| string literals | `"text"` | UTF-8; escapes `\" \\ \n \t \r \xNN \uNNNN` |
| char classes | `[abc]`, `[a-z0-9]`, `[^"]` | codepoint ranges; `^` negates over all of Unicode |
| class shorthands | `[\d]`, `[\w]`, `[\s]` | ECMA-262 sets over ASCII: `\d` = `[0-9]`, `\w` = `[0-9A-Za-z_]`, `\s` = space and U+0009..U+000D. **Inside a class only** — they are sets, so `"\d"` has no single character to mean |
| any char | `.` | any single codepoint (newline included) |
| grouping | `( ... )` | |
| alternation | `a \| b \| c` | |
| repetition | `x*` `x+` `x?` `x{m}` `x{m,}` `x{m,n}` | `m,n` decimal, `n >= m`, `n <= 1024` |
| comments | `# ...` to end of line | |
| whitespace | spaces/tabs/newlines between tokens | inside a literal/class, literal |

**Refused loudly (parse error, with the offending token):**

* an escape other than the list above — in particular `\p{...}` (llama.cpp does
  not support it either; write `[\p{...}]`'s intended ranges explicitly), and
  `\d`/`\w`/`\s` **outside** a character class, where a set has no single
  character to mean;
* a class shorthand used as a range endpoint (`[\d-x]`): it would otherwise read
  as the set plus a `-`..`x` range, accepting far more than was written (`[\d-]`
  is still a literal minus, as everywhere else in a class);
* `\x` with a malformed body, `\u` without exactly four hex digits;
* an empty char class `[]`, an unterminated class/literal/group, a stray `)` or
  `|` with an empty operand, a trailing `::=` without a body;
* a missing `root` rule, a duplicate rule name, an undefined rule reference;
* `{m,n}` with `n < m` or `n > 1024` (an explicit bound keeps the expansion
  finite; the refuser names the limit);
* a rule set that is **left-recursive** (`root ::= root "a"`): the automaton
  detects it at parse time through the stack-depth bound (§4) and reports it
  instead of looping.

## 3. JSON-Schema subset (in scope)

`[defs]` below means: `$defs` / `definitions` entries are pre-registered as
rules and compiled once, on demand.

### In scope

| Keyword | Behaviour |
|---|---|
| `type` | `"object" "array" "string" "number" "integer" "boolean" "null"`, or an array of those (compiled as a union) |
| `enum` | union of the listed values; every value type is supported |
| `const` | exactly that value |
| `object`: `properties` | declared properties, emitted **in declaration order** (see the scope note below) |
| `object`: `required` | a required declared property must appear; a required name that is not declared is refused |
| `object`: `additionalProperties` | `false` (closed), `true`/absent (extra `"key": value` entries after the declared ones), or a schema (extra values compiled from it) |
| `array`: `items` | one schema for every element |
| `array`: `minItems`, `maxItems` | exact bounds on the element count (`minItems` may not exceed `maxItems`) |
| `array`: `prefixItems` | a fixed leading tuple, then `items` for the remainder (or no remainder when `items` is absent/`false`) |
| `string` | a JSON string: `[^"\\\u0000-\u001f]*` plus `\" \\ \/ \b \f \n \r \t` and `\uXXXX` escapes; `minLength`/`maxLength` count **characters** (one escape is one character), and `format` narrows it to one of the validators below |
| `string`: `minLength`, `maxLength` | exact character bounds (`minLength` may not exceed `maxLength`); the encoded string is `"\"" j-char{m,n} "\""` |
| `string`: `format` | `date`, `time`, `date-time`, `uuid`, `uuid1`..`uuid5` — the four rules the reference implements (the uuid version digit is not checked, as there) |
| `string`: `pattern` | a regex over the **decoded** characters, escape-aware: literals (any codepoint), `.`, positive and **negated** classes `[...]`/`[^...]` over the whole codepoint space with `\d`/`\w`/`\s` and `\D`/`\W`/`\S`, groups (capturing or not), alternation, `*` `+` `?` `{m}` `{m,}` `{m,n}` (lazy `?` accepted), and `^`/`$` at the two ends. Unanchored, it matches a substring (ECMA-262); each anchor is optional and independent. The JSON text may spell a character literally or as an escape — both are accepted, including the surrogate pair that spells a character above U+FFFF — and `.` refuses exactly the four line terminators in either spelling |
| `integer`: `minimum`, `maximum` | inclusive bounds, integer-valued (see below) |
| `integer`: `exclusiveMinimum`, `exclusiveMaximum` | integer-valued; folded into the inclusive range |
| `number` | the general JSON number grammar (`-?int(.frac)?([eE]exp)?`) when unbounded |
| `boolean`, `null` | `true` / `false` / `null` |
| `anyOf`, `oneOf` | union of the branches |
| `$defs` / `definitions`, `$ref` | `#/$defs/NAME` and `#/definitions/NAME` only; recursive references work because the rule is registered before it is compiled |
| `$schema`, `title`, `description`, `name`, `strict`, `default`, `examples` | accepted and ignored (they do not change the accepted language) |

### Refused loudly

* any `$ref` that is not a local `#/$defs/...` or `#/definitions/...` pointer —
  in particular remote URLs and nested JSON pointers (`#/$defs/a/properties/b`);
* `contentEncoding`, `contentMediaType` on strings — and every
  `format` outside the four above (`email`, `uri`, `hostname`, `ipv4`, `ipv6`),
  refused **by name**: a loose rule would accept instances the schema rejects;
* a string keyword (`format`, `minLength`, `maxLength`, `pattern`) on any single
  type other than `"string"`, with no type at all, or on a union — the constraint
  would be silently dropped, and a union would have to be split into `anyOf`
  branches;
* **two of** `pattern` / `format` / `minLength`-`maxLength` on one string (they
  would have to be intersected);
* a `pattern` outside its subset, each refusal naming the construct: a
  backreference (`\1`, `\k<...>`), lookaround or any group other than `(?:...)`
  (`(?=`, `(?!`, `(?<=`, `(?<!`, `(?<name>`, `(?>` …), a word-boundary assertion
  (`\b`, `\B`), a Unicode property escape (`\p{...}`), a control or octal escape,
  the `\u{...}` form, a possessive quantifier (`*+`, `++`, `?+`), a class
  shorthand as a range endpoint, a misplaced anchor (`^`/`$` anywhere but the two
  ends, or combined with a top-level `|`), and a malformed quantifier or class;
* `multipleOf` and non-integer `minimum`/`maximum`/`exclusiveMinimum`/
  `exclusiveMaximum` on numbers (integer bounds are supported; a real-valued
  bound is a named refusal, see the follow-ups — narrowing a `number` to its
  integer range would silently change the accepted language);
* `minProperties`, `maxProperties`, `propertyNames`, `patternProperties`,
  `dependentRequired`, `dependentSchemas`;
* `uniqueItems`, `contains`, `minContains`, `maxContains` on arrays;
* `allOf`, `not`, `if`/`then`/`else`, `unevaluatedProperties`,
  `unevaluatedItems`;
* an unknown keyword that changes the accepted language is *not* silently
  dropped: the compiler walks the schema and refuses any key outside the list
  above except the pure annotations;

  **Correction (implementation note).** The last item was the design intent and
  was rejected during implementation: a schema is an open vocabulary, and
  refusing every unknown key would reject `$comment`, vendor extensions and the
  many annotations nobody reads. The implemented rule is the narrower one:
  **the keywords that change the accepted language are the ones listed above;
  an unknown keyword is ignored** — and the compiled GBNF is printable, so what
  was actually enforced is inspectable. The catalogues above are the honest
  record of what is enforced.
* `type` present together with a keyword for a type it does not include (`{
  "type": "string", "properties": {…} }`) — the misplaced keyword is ignored in
  JSON Schema, and here it is ignored too (consistent with the note above),
  except that `required`/`properties` on a non-object is refused because it is
  usually a schema bug;
* a schema that is not a JSON object at the top level (a bare `true`/`false`),
  and an empty `enum` (no value can satisfy it).

### Scope note: object property order

Object entries are emitted **in `properties` declaration order**: required
properties must appear, optional ones may be skipped, and an extra entry (when
`additionalProperties` allows one) may be interleaved at any position where no
required declared property is still pending. This is a *subset* of the schema's
language — a model cannot produce `{"b":1,"a":2}` for a schema that declares `a`
then `b`, nor may an extra precede a required first property. It never accepts
an object the schema rejects, which is what the acceptance line ("constrained
output always parses under the given schema") requires. llama.cpp generates
permutations up to a size cap; that is a follow-up (§8), not a silent guess: the
behaviour is stated here and in `docs/USAGE.md`.

### Scope note: `oneOf`

`oneOf` is compiled as `anyOf`. For non-overlapping branches the two coincide;
for overlapping branches the compiled language is the union, so a model could
emit a value that validates against two branches (which JSON Schema's `oneOf`
forbids). This is llama.cpp's behaviour and is a follow-up.

## 4. Engine

### Rule programs

Each rule compiles to a flat `Vec<Inst>` over codepoints:

```text
Byte(u8)                  one literal byte, as a codepoint
Class { negated, ranges } codepoint set
Any                       '.' — every codepoint
Split(a, b)               nondeterministic branch (alternation, ?, *, +)
Jump(a)                   epsilon jump (loop back-edges)
Call(rule)                push a frame and enter a rule
Ret                       return (at the end of every rule)
```

A **frame** is `{ rule: u32, pc: u32 }`; a **stack** is `Vec<Frame>`; the
automaton state is a *set* of stacks (nondeterminism) plus the carried
partial-UTF-8 bytes. `Ret` at stack depth 1 means "the `root` rule finished" and
is the accepting state — no separate `Match` instruction is needed, and `root`
can be referenced recursively like any other rule.

The epsilon closure walks `Split`/`Jump`/`Call`/`Ret` with a worklist and a
`HashSet` of visited stacks, so pure epsilon cycles terminate. Two loud bounds:

* `MAX_STACK_DEPTH = 256` — a stack that grows past it means a left-recursive
  rule, reported as such;
* `MAX_STACKS = 64` — more live nondeterministic branches than any in-scope
  grammar needs; exceeding it is an error, not a silent truncation.

The closure's output is sorted (depth, then frames) so the state key is
canonical: equal states hash equal regardless of visit order, which is what makes
the mask cache correct.

### Token advancement (byte-level)

A token's piece is `Tokenizer::decode_bytes(&[id])` — the *raw* bytes, so a
token may be a partial UTF-8 sequence or span several characters. The automaton
consumes **codepoints**:

```text
buffer = state.partial ++ piece
i = 0
while i < buffer.len():
    Codepoint(cp, n) -> step every stack on cp; i += n
    Incomplete       -> stop; the tail is the new partial
    Invalid          -> reject the token (with the byte offset)
state.partial = buffer[i..]
```

Consequences, all of them asserted:

* a token that *is* one byte of a three-byte character is allowed, and the
  pending bytes ride in `state.partial` until the character completes — the mask
  is therefore byte-level correct, with no lossy UTF-8 conversion anywhere;
* a token whose earlier bytes are accepted and whose later codepoint is not is
  **rejected as a whole** by the mask (`accept_token` reports the byte offset of
  the failure — the "longest accepted prefix" — and does not mutate the state);
* an invalid UTF-8 byte is a rejection, never a `U+FFFD` substitution;
* a token that *ends* in a partial sequence is allowed only when the pending
  bytes can still complete to a codepoint the state accepts. The completion is
  computed exactly, so overlong forms (`0xE0 0x80 …` for `U+0061`) and the
  surrogate block do not count — a lone `0xE0` after `"a"` is rejected rather
  than accepted and then stranded (that was a real defect, §9).

### End-of-generation

The EOG ids (`eos`, and `im_end` when the model has one) are allowed **iff the
state is accepting and no partial codepoint is pending** — a complete string
cannot end mid-character. Accepting EOG is a no-op on the state (the run is
over). Every other token with an empty piece (special tokens the decoder maps to
nothing) is never allowed.

### The mask

`Grammar::mask(&state) -> Arc<[u64]>` (a packed bitset over token ids) is the
only vocabulary-wide step. It is cached **by state key** in `GrammarState`
(`HashMap<StateKey, Arc<[u64]>>`, bounded at 64 entries, cleared when full), so a
run that revisits a state (JSON's "expect a key or `}`" state is revisited after
every member) pays for it once per distinct state. The compiled
`Grammar` itself is immutable and shared by `Arc`, so a cache per run needs no
lock.

Two independent things reduce the walk itself:

1. **The per-call transition memo.** Inside one mask computation the start state
   is fixed, so "consume one codepoint" is a function of `(state, codepoint)`;
   the walk interns successor states and memoizes the transition, turning the
   O(vocabulary × piece length) walk into O(distinct transitions). That is what
   took the cost from 71.3 ms to 5.4 ms per new state (§9).
2. **The first-codepoint index ([#126](https://github.com/yusiwen/minfer/issues/126)).**
   Only a piece's *first* codepoint decides whether the piece can begin; the memo
   decides the rest. `Grammar` therefore indexes its token ids by that codepoint
   once, at compile time, and a state walks only the buckets its frontier accepts.
   A piece whose leading bytes are an *incomplete* UTF-8 sequence has no first
   codepoint and is always walked (it is judged against the completion range); a
   piece that starts with an invalid byte, or is empty, can never be accepted on
   this path and is not indexed at all.

`Grammar::plan` chooses the walk per state: a frontier that is a `.` or a negated
class, or whose accepted set is at least as wide as the vocabulary, keeps the
whole-vocabulary walk (filtering cannot pay there), and everything else uses the
buckets. **The bitset is identical either way** — the index only decides which
pieces are examined.

Measured on the 151,936-token Qwen2.5-0.5B q4_0 vocabulary (dgxspark, 2026-10-14,
`cargo test --release --bin minfer real_model_json_schema_generation_parses --
--ignored --nocapture`): the mask over a 17-step constrained JSON response (13 new
states) costs **49.5 ms, down from 90.4 ms** — 45 % less — and the walk examines
**42,329 pieces per new state on average (28 % of the vocabulary)**, 1,049 at the
narrowest structural state (0.7 %). The ceiling is honest and recorded in §8: a
state whose frontier accepts nearly every codepoint still walks all 151,935 pieces
(33.7 ms), and that one state is most of what remains.

### "No token is allowed"

`sample_with_config_grammar` returns `Err(SampleError::NoAllowedToken)` when the
mask leaves no finite logit. **The sampler never falls back to an arbitrary
token**: the CLI prints the reason and stops, the server ends the stream with
`finish_reason: "stop"` and logs the reason, and the grammar state's `describe()`
carries the rule/frame context. This is reachable in practice (e.g. `root ::=
"a"` after `a` with no EOG id, or a partial codepoint no token can complete), so
it is a gate, not a theoretical branch.

`Err(SampleError::Grammar)` is the same treatment for an `accept_token` failure;
after a mask admitted the token that should be unreachable, so it doubles as the
internal-consistency check.

## 5. Where the mask sits in the pipeline

```text
logit bias → penalties → DRY → [GRAMMAR MASK] → greedy shortcut → top-k →
typical → top-p → min-p → XTC → temperature | mirostat v1/v2
```

One pipeline, in `src/sampler.rs`; there is no second sampler path.

* The mask is applied to the logits the sampler actually sees — after every
  stage that *shifts* logits (bias, penalties, DRY) and before every stage that
  decides the distribution (`top-k`/`typical`/`top-p`/`min-p`/XTC mask
  candidates, `temperature`/mirostat sample one). No forbidden token can be
  selected because every later stage only ever *removes* candidates or reweights
  the survivors.
* The mask is before the **greedy shortcut** (`temp == 0`), so `--greedy`
  respects the grammar; that is the case the acceptance line's "applied at the
  same point as the other samplers" is about.
* Earlier would also be safe (a finite shift cannot lift `-inf`), but later is
  stated as the invariant, independent of how the penalty stages evolve:
  nothing downstream of the mask may *add*.
* The mask consumes **no RNG** and writes no state the other samplers read, so
  mirostat's `mu` trajectory and DRY's deterministic penalties are unchanged for
  the same token sequence; the F2 bitwise gate pins the no-grammar path.
* Forbidden logits are set to `-inf` exactly as `apply_top_k`/`apply_min_p` do,
  so the downstream filters' "survivors" logic (`v > -inf`) keeps working with
  no special case.

### State ownership

`SamplerConfig.grammar: Option<Arc<Grammar>>` is the compiled, immutable object —
one per request, shared by clone. The mutable `GrammarState` is owned by the
run, exactly like `MirostatState`: CLI run, conversation session, server
request, batch slot. Speculative decoding **refuses** a grammar loudly (the
verify round samples several rows from one state, and the draft model would need
the same mask); `--spec-draft` + `--grammar` is a startup error and the server
refuses the combination with a `400`.

## 6. Surfaces

**CLI** (mutually exclusive; more than one is a startup error):

| Flag | Meaning |
|---|---|
| `--grammar <FILE>` | GBNF from a file |
| `--grammar-str <GBNF>` | GBNF inline |
| `--json-schema <FILE>` | JSON Schema from a file |
| `--json-schema-str <JSON>` | JSON Schema inline |

An invalid grammar/schema is refused at startup, before the model is used, with
the parse error. `--grammar` + `--spec-draft` is refused. `--cnv` carries one
grammar state for the session (a grammar is re-applied to every assistant turn).

**Server** (`/v1/chat/completions`), mapped onto the OpenAI fields:

| Request field | Meaning |
|---|---|
| `response_format: {"type":"json_object"}` | any JSON value (`root ::= value`) |
| `response_format: {"type":"json_schema","json_schema":{"name":…,"schema":{…}}}` | the compiled schema |
| `response_format: {"type":"text"}` | no constraint |
| `grammar: "<GBNF>"` | GBNF inline (extension field, llama.cpp-style) |

`grammar` together with a non-text `response_format` is a `400`; an unsupported
`type`, a malformed schema, or a refused keyword is a `400` naming the
construct — the same boundary at which `SamplerConfig::validate` already refuses
bad values. Because the EOG ids and the token pieces come from the vocabulary,
the schema is compiled **per request** on the handler side (where the tokenizer
lives), before the job occupies a worker slot.

## 7. Acceptance

| Gate | Assertion |
|---|---|
| GBNF parser round-trip | every in-scope construct parses and matches the strings it should; every refused construct produces the named error |
| automaton unit tests | char classes, negation, `.`, alternation, repetition ranges, nested/recursive rules, the stack-depth and stack-count refusals |
| JSON-schema compiler | each in-scope keyword compiles and accepts a hand-written valid instance / rejects an invalid one; each refused keyword returns its named error |
| partial UTF-8 | a multi-byte character split across two tokens is accepted and the state completes; a token that cannot complete the partial is rejected without mutating the state |
| longest accepted prefix | a token whose first codepoint is accepted and second is not is rejected as a whole, and the error names the accepted byte count |
| no allowed token | a grammar that is exhausted with no EOG id yields `SampleError::NoAllowedToken`, never a token |
| mask caching | the same state computes the mask once (cache hit on revisit); the compiled grammar is shared, the state is not |
| bitwise unchanged | the pinned pre-F2 greedy sequence (captured from `master` before this change) is reproduced through the new entry point with no grammar |
| mutation checks | an off-by-one in the allowed-token range, a wrong partial-UTF-8 carry, an unmet `minItems` and a disabled EOG allowance each make a gate fail |
| real model (`#[ignore]`, serial) | fixed seed + JSON schema → `serde_json::from_str` succeeds on the cached 0.5B (f32 KV) **and** on Qwen3-0.6B Q8_0 when present |
| empty case | `-n 0`/`max_tokens: 0` emits nothing and stops cleanly; a grammar that accepts ε allows only EOG at step 0 |
| max-length case | generation cut by the length limit yields a byte string the automaton never left a valid prefix state for — asserted with `Grammar::accepts_prefix(output)` |

## 8. Honest scope and follow-ups

* **Object property order** is declaration order (subset of the schema's
  language), not the permutation set llama.cpp generates.
* **`oneOf`** is compiled as `anyOf` (superset for overlapping branches).
* **Real-valued numeric bounds** and `multipleOf` are refused; only integer
  bounds are compiled.
* **`pattern` is a named regex subset, matched against the decoded characters.**
  It landed on 2026-10-14 (#125's slice 3) and is the largest single piece of the
  compiler: literals at any codepoint, `.`, positive and negated classes over the
  whole codepoint space (`[...]`, `[^...]`, `\d` `\w` `\s` `\D` `\W` `\S`),
  groups (capturing or not), alternation, the quantifiers `*` `+` `?` `{m}`
  `{m,}` `{m,n}` (a lazy `?` suffix is accepted — it changes matching, not the
  language), and the two anchors. A character is compiled **escape-aware**: the
  JSON text may spell it literally, as a short escape (`\n`, `\"`, …), as
  `\uXXXX`, or — above U+FFFF — as a surrogate pair, and every spelling that
  decodes into the set is accepted. That is what lets a negated class be exact
  rather than approximated: the complement is enumerated, both in the literal
  alphabet and in the four-hex-digit automaton, instead of being matched against
  the raw text (which would accept strings the schema rejects). The refusals that
  remain (backreferences, lookaround, `\p{...}`, possessive quantifiers, a
  misplaced anchor) each name the construct, and §3 lists them.
  The reference is *not* a model here: it requires both anchors and, when either
  is missing, **widens to any string with a warning** — minfer either compiles the
  pattern exactly or refuses it.
* **`minLength`/`maxLength` count characters.** Counting rules over `j-char` make
  the count character-based rather than byte-based: one escape is one character,
  and a surrogate pair written as two `\uXXXX` escapes is two. A string keyword is
  accepted only on `type: "string"` itself — on another type, with no type, or on
  a union, the constraint would be dropped or need intersection, so it is refused
  by name.
* **`format` implements four validators, not a format library.** `date`, `time`,
  `date-time` and `uuid`/`uuid1`..`uuid5`, byte for byte the rules the reference
  implements; `email`, `uri`, `hostname`, `ipv4` and `ipv6` are refused rather
  than approximated (the reference only uses those spellings to infer "this is a
  string", and a loose rule here would accept what the schema rejects).
* **Left recursion** is detected at parse time and refused; right recursion
  (the `$ref` cycle shape JSON uses) works.
* **The class shorthands `\d`/`\w`/`\s` are minfer's, not llama.cpp's.** The
  reference rejects every shorthand (`parse_char` in `src/llama-grammar.cpp` has
  no `d`/`w`/`s` case), so a grammar written for minfer may not compile there.
  That is a deliberate divergence — the definitions are ECMA-262's over ASCII,
  written down in §2 — and `\p{...}` stays refused on both sides.
* **Per-request compile cost** is O(vocabulary) for the token pieces plus the
  grammar size, and the first-codepoint index (#126) adds one more O(vocabulary)
  pass at compile time; it is measured in the record and is the reason the mask
  cache is per state rather than recomputed per token.
* **The mask's residual cost is the wild frontier, not the walk.** A state whose
  frontier accepts nearly every codepoint — every state inside a JSON string body —
  keeps the whole-vocabulary walk: 151,935 pieces, 33.7 ms, measured 2026-10-14.
  That single state is most of the 49.5 ms a 13-state JSON response still costs;
  the structural states fell to 0.7–37 % of the vocabulary. Narrowing it further
  needs an index on the *rejected* codepoints, not on the first codepoint.
* Follow-ups are filed as GitHub issues: [#125](https://github.com/yusiwen/minfer/issues/125)
  (the refused GBNF/schema constructs); [#126](https://github.com/yusiwen/minfer/issues/126)
  (the first-codepoint index) landed 2026-10-14.

## 9. Implementation notes (2026-09-24)

Two defects were found by the gates, not by reading the code, and both are now
covered by unit tests:

1. **A partial-UTF-8 token could be accepted where the automaton could never
   finish it.** After a JSON object closed, a lone `0xE0` leader was allowed
   (the first version computed the completion range as `U+0000..=U+0FFF`, which
   includes ASCII), the run then had no legal continuation, and the response
   ended with `U+FFFD` — an illegal byte, exactly what the acceptance line
   forbids. Found by `real_model_json_schema_generation_parses` and reproduced
   over HTTP with `{"grammar":"root ::= \"ab\""}` (the response was `"a\uFFFD"`
   instead of `"ab"`). Fixed by computing the completion range exactly:
   overlong forms, the surrogate block and codepoints above `U+10FFFF` are
   excluded (`completion_range`), plus a defensive rule that a response body
   never ends mid-character (`complete_utf8_prefix_len` on the final flush, both
   server paths).
2. **The mask was 13× more expensive than it needed to be** — 71.3 ms per new
   state on a 151,936-token vocabulary, which dominated a 0.5B generation
   (1.48 s for 15 tokens). Fixed by the per-call transition memo (§4): 5.4 ms
   per new state, and the constrained run (0.42 s) is now faster than the
   unconstrained one (0.70 s) because the grammar stops it after 15 tokens
   instead of 48. The residual ~86 ms for a 16-step JSON response is the honest
   price of an O(vocabulary) state at this vocabulary size;
   [#126](https://github.com/yusiwen/minfer/issues/126) tracks the idea of a
   token-indexed first-codepoint bucket to remove it, and
   [#125](https://github.com/yusiwen/minfer/issues/125) the constructs the
   compiler refuses.

**Follow-through (#126, 2026-10-14).** The first-codepoint bucket landed. §4 has
the mechanics and the numbers (49.5 ms for the same 13-state JSON response, down
from 90.4 ms). It does not remove the O(vocabulary) term for a *wild* frontier —
a JSON string body still walks every piece — and §8 now says so instead of
promising it. The gates are the same three that F2 left behind, plus a bitwise
comparison of the indexed mask against the whole-vocabulary walk on both the
synthetic vocabulary (`grammar::tests`) and the real 151,936-token one
(`real_model_json_schema_generation_parses`), and a work bound that asserts the
indexed walk examines fewer pieces than the vocabulary has tokens.

The measured acceptance (both models, greedy, seed 42) is in
`docs/ARCHITECTURE-EXECUTION-PLAN.md`'s F2 record.

## Decisions governing this document

This page is the *current contract*; the decisions behind it are frozen in the ADR corpus:
- [ADR-0018](adr/0018-the-grammar-mask-is-one-pipeline-stage.md) — The grammar mask is one stage inside the single sampler pipeline
- [ADR-0010](adr/0010-the-identity-gate-bitwise-by-default.md) — The identity gate: bitwise by default, a named tolerance class otherwise
