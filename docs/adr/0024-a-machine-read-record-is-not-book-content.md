# 0024. A machine-read record is not book content

- Status: Accepted
- Date: 2026-10-09
- Issues: #459
- Corrects: ADR-0007, ADR-0020

## Context

`docs/` is the mdBook source tree, and mdBook copies **every** non-markdown file it finds there into the published site, whether or not the book links it. A file that only a test or a checker reads therefore becomes a published asset by accident — and its location also answers the wrong question for anything that reads a `git diff --name-only` to decide what a change can affect.

`docs/f6-fixtures.json` was exactly that: 33 005 B of recorded provenance for the F6 fixture cache, read by two programs that do not live in `docs/` —

- `scripts/check_f6_fixtures.py`, whose `MANIFEST` constant it is (audited structurally in the `check-docs` job), and
- `src/tooling/tests/f6_fixtures.rs`, whose `CARGO_MANIFEST_DIR` const names it and whose reader test is **not** `#[ignore]`d, so the file is an input to the `test-linux-cpu` job —

while the book linked it from nowhere and the site served it as `application/json` (measured 2026-10-09).

[ADR-0023](./0023-the-machine-ledgers-live-beside-their-checkers.md) settled this for the two status ledgers: one ledger per prose target, beside the checker that reads it. This record is the same shape *without* a prose target — nothing in it pins prose; its audit compares entries against the fixture cache — so the rule it needs is about the kind of file, not about ledgers.

## Decision

No file that only a test or a checker reads lives under `docs/`. Such a record lives outside the book's source tree, in the tree of its readers: `tests/fixtures/` for a fixture record with more than one reader, `scripts/` for a record only a checker reads.

`docs/f6-fixtures.json` therefore moves to `tests/fixtures/f6-fixtures.json`, byte-identical, beside `tests/fixtures/cuda_launch_sites.tsv` — the same dual-consumer shape (read by both `scripts/check_cuda_launch_returns.py` and a Rust test). Both readers name the new path; the `--manifest` / `MINFER_F6_MANIFEST` overrides and the record's schema are unchanged.

What is book content is decided by the **reader**, not by the extension: a `.md` chapter stays; a `.json`/`.toml`/`.tsv` that no prose target pins does not.

## Alternatives considered

- **Leave it where it is and let the CI classifier carry an exception.** Rejected: it keeps a machine record inside the book tree, so the file is published as an asset nobody links, and it encodes "this documentation file is really a test input" as a classifier exception rather than as a path fact — the same objection ADR-0023 raised for the status ledgers.
- **Move it to `src/tooling/tests/`, beside the Rust reader.** Rejected: the record has two readers in two trees, and `tests/fixtures/` is the repo's existing neutral home for exactly that. Under `src/` it would privilege one reader and bury a 33 KB data file in the crate tree.
- **Split the record per reader.** Rejected: both readers audit *one* record; two copies would need a synchronisation gate that does not exist, and the provenance claim is a claim about one file.

## Consequences

- The site loses the stray asset and the URL that served it stops resolving — the intended removal of an accident, the same one #455 performed for `docs/status.toml`.
- A change to `tests/fixtures/f6-fixtures.json` is now a change to a test input by path, so the classification in #457 needs no documentation-path exception for it.
- ADR-0007's and ADR-0020's citations of `docs/f6-fixtures.json` name a path that no longer exists, which is a defect in frozen text. Per ADR-0022 this ADR carries `Corrects: ADR-0007, ADR-0020`, and both corrected ADRs' index rows name this one; **neither body is edited**.
- Future non-book records have a stated home, so the question is not re-argued per file.

## References

- `scripts/check_f6_fixtures.py`, `src/tooling/tests/f6_fixtures.rs` — the two readers, and their `--help` / module docs.
- [`docs/GGUF-TOOLING.md`](../GGUF-TOOLING.md) §4.2 — the record's content contract and the producer recipe.
- [ADR-0022](./0022-a-defect-in-a-frozen-adr-is-corrected-by-a-new-adr.md) — how a defect in a frozen ADR's text is corrected.
