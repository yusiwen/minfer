# 0042. The tokenizer applies the model's declared NFC normalizer

- Status: Accepted
- Date: 2026-10-15
- Issues: #132
- Corrects: ADR-0019

## Context

ADR-0019 recorded the input normalizer as an accepted gap, in its own words: *"the Hugging
Face normalizer (NFC) is **not** applied ([#132](https://github.com/yusiwen/minfer/issues/132))"*.
F7 ([#50](https://github.com/yusiwen/minfer/issues/50)) built the byte-level BPE around that
assumption, which made the gap invisible: every committed token-id fixture is
NFC-normalized, so the missing normalizer never showed up in a gate.

The reference tokenizers declare it. Verified 2026-10-15, each at the revision its fixture
names: `Qwen/Qwen3-0.6B@c1899de2`, `Qwen/Qwen2.5-0.5B-Instruct@main` and
`unsloth/Qwen3.5-0.8B@23c69c53` all carry `"normalizer": {"type": "NFC"}` in their
`tokenizer.json`. The reference ids therefore *contain* the normalization, and minfer's did
not: a decomposed input tokenized differently from the model's own tokenizer.

Measured on the cached GGUFs (dgxspark, 2026-10-15, `cargo test --release --bin minfer
token_ids_match_the_reference -- --ignored --nocapture`). Before this change `encode` was the
unnormalized path, which on all four Qwen2.5/Qwen3 vocabularies returns `[924, 1859, 53839]`
for `"cafe\u{0301}"` while the composed `"café"` returns `[924, 58858]` — a different token
sequence for the same text, silently. (On Qwen3.5 the pair is `[895, 1795, 52033]` against
`[895, 56868]`.)

llama.cpp applies no normalizer for BPE, so the two engines cannot both be followed on
non-NFC input. The model's own `tokenizer.json` is the reference this engine already gates
against, and it is the one that decides.

## Decision

**1. NFC is applied to every text the tokenizer turns into ids**, at the two entry points
`Tokenizer::encode` and `Tokenizer::encode_raw`, because the normalizer belongs to the
tokenizer rather than to the special-token pre-scan: the CLI, the chat path, the server, the
grammar-constrained prompt and the DRY breaker tails all reach BPE through those two
functions. The tokenizer borrows the input when it is already NFC.

**2. The Unicode data is generated and committed, not a dependency.** `scripts/gen_unicode_nfc.py`
reads CPython's bundled `unicodedata` (Unicode 15.0.0) and writes `src/tokenizer/nfc/tables.rs`
(canonical decomposition, canonical combining class, canonical composition and its second
elements) and the differential corpus `tests/fixtures/tokenizer/nfc_vectors.json` (5 103
inputs, 8 named sections, each answered by `unicodedata.normalize('NFC', …)`). The engine's
dependency list is deliberately short (`AGENTS.md` → Dependencies), and the algorithm's
correctness rests on the corpus rather than on the source of the tables.
`src/tokenizer/nfc.rs` holds the algorithm: full canonical decomposition (recursively, and
Hangul arithmetically), canonical ordering of each mark run, and the blocked-composition rule.
NFC is not NFKC: compatibility forms (`U+FB01`, fullwidth) are left alone.

**3. The quick check is conservative, never a substitute for the pipeline.** ASCII short-circuits;
otherwise any character that decomposes, carries a combining class, is a conjoining jamo, or
can be the *second* element of a canonical composition runs the full pipeline — the last
matters because the Indic vowel signs compose with a combining class of 0, so a
class-and-decomposition-only check skips them. A run whose result equals the input still
returns the borrowed form.

**4. DRY-breaker matching stays byte-exact.** `Tokenizer::overlapping_token_sequences`
compares a breaker against the vocabulary's own text, as llama.cpp does; a non-NFC breaker
therefore matches nothing and is refused by name rather than silently matched on normalized
text. Only the tails are normalized, because a tail is tokenized text.

**5. The divergence from llama.cpp is recorded, not discovered.** On non-NFC input minfer
follows the model's declared normalizer and llama.cpp does not, so the two engines return
different ids for the same decomposed text. The id-gate corpus is NFC, so nothing in the
gates moves; the divergence is stated in the design record and in `docs/USAGE.md`.

## Alternatives considered

- **Refuse non-NFC input** (the other branch the ticket named). Rejected: detection needs the
  same Unicode data (an exact "is this NFC?" answer is the same tables), it rejects input a
  consumer can legitimately send — macOS text is frequently NFD — and it would leave minfer
  disagreeing with the model's own tokenizer instead of agreeing with it.
- **Add the `unicode-normalization` crate.** Rejected, and the rejection is a dependency
  decision rather than a claim about the crate: it would be the same generated data behind a
  new entry in a deliberately short dependency list, and the differential corpus already
  pins the algorithm against CPython, which the crate would not strengthen.
- **Normalize at the chat-template, server or CLI layer.** Rejected: a request shape is not
  the contract. Every consumer tokenizes through the same two entry points, and a future
  consumer would have had to remember.
- **NFKC.** Rejected: it is not canonical equivalence, and it rewrites text the model never
  saw (ligatures, fullwidth forms, `U+212B`-style singletons are NFC's business, ligatures are
  not).
- **Normalize only the scripts the supported models are known to use.** Rejected: an unnamed
  subset that silently skips a script is the exact failure mode this record exists to remove.
- **Hand-write a partial normalizer for the Latin-1 composition pairs only.** Rejected for the
  same reason, and it would not have fixed the Indic class-0 compositions the corpus caught.

## Consequences

- A decomposed input and its composed form now produce the same ids, which is what the
  reference tokenizer does. The property is gated three ways: 5 103 CPython-answered inputs
  (`src/tokenizer/nfc/tests.rs`), a synthetic tokenizer where the composed character has a
  real vocabulary entry, and the `#[ignore]`d real-model id gate, which now also asserts
  `café` == `cafe\u{0301}` on every cached model and that the unnormalized path disagrees.
- On non-NFC input minfer intentionally differs from llama.cpp. The id gates are unaffected
  (their corpus is NFC), and the divergence is written down in
  `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` and `docs/USAGE.md`.
- A Unicode upgrade is a generator re-run: both outputs name the data version, the fixture's
  `unicode_version` is asserted against the generated constant, and the corpus is regenerated
  from the same database, so the diff is reviewable.
- Cost: ASCII input still borrows; already-NFC non-ASCII input runs the pipeline once and
  borrows the result. The 5 103-input corpus runs in about 10 ms in release.
- ADR-0019's text is **not** edited: its accepted gap is corrected by this record, and its
  index row names this corrector, per ADR-0022.
- **Residual, recorded:** the tables are Unicode 15.0.0, CPython's bundled version. A newer
  Unicode release changes NFC only for characters assigned since, and upgrading is a
  generator re-run plus the corpus diff — a data refresh, not a decision.

## References

- `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md` — §3.6 (the contract), §4 (the gates).
- [`docs/adr/0019-a-chat-template-that-cannot-render-refuses-the-load.md`](./0019-a-chat-template-that-cannot-render-refuses-the-load.md)
  — the record whose accepted gap this corrects.
- `src/tokenizer/nfc.rs`, `src/tokenizer/nfc/tables.rs` — the algorithm and the generated data.
- `src/tokenizer.rs` — `Tokenizer::encode`, `Tokenizer::encode_raw`.
- `scripts/gen_unicode_nfc.py` — the generator; `tests/fixtures/tokenizer/nfc_vectors.json` —
  the differential corpus.
- Issues: [#132](https://github.com/yusiwen/minfer/issues/132) (S1),
  [#50](https://github.com/yusiwen/minfer/issues/50) (the F7 landing).
