# 0023. Each machine ledger lives beside its checker, one per prose target

- Status: Accepted
- Date: 2026-10-09
- Issues: #455
- Corrects: ADR-0022

## Context

`docs/status.toml` carried three kinds of derived fact in one file: the plan's
`baseline_commit` / `refreshed_against` / `next:` scalars with their three `[prose.*]`
targets, the seven `[[phase]]` counters, and the nine `[[counts]]` suite-count rows.

The halves were never one thing. Their prose targets differ
(`docs/ARCHITECTURE-EXECUTION-PLAN.md` against `docs/TEST-BASELINES.md`); their consumers
differ; and exactly one row — `x86_64 (CI runner)` — is compared against a real
`cargo test` log by `check_status.py --check-live`, which only the `test-linux-cpu` job
runs. One path for both halves meant a file-level fact ("did the suite counts move?")
could not be read off a `git diff --name-only` without parsing TOML at two revisions, and
the same file was at once a documentation record and a test input.

Two smaller costs came with it. `scripts/` held only executables, so the checkers' machine
records had no home convention; and the ledger lived under `docs/`, the book's source
tree, where mdBook copies every non-markdown file into the published site — measured
2026-10-09, the book served it as a 16 KB `application/toml` asset that nothing linked.
`check_status.py` also hard-coded `docs/status.toml` in twenty diagnostic strings while its
own docstring claimed it "holds no second copy of any value".

## Decision

There is one ledger per prose target, and each ledger sits beside the checker that reads
it.

- `scripts/status.toml` (`kind = "plan"`) holds the plan's counters, `next:` sentence and
  baseline/refreshed commit ids; `scripts/check_status.py` reads it.
- `scripts/test-baselines.toml` (`kind = "counts"`) holds the suite measurements;
  `scripts/check_baselines.py` reads it, and its `--check-live` is the one mode with a
  live consumer.

Each file declares its `kind`, and a checker refuses a manifest of another kind by name
and short-circuits, so "the wrong ledger was passed" is one problem rather than a pile of
missing-key noise. A manifest with no `[[phase]]` rows (plan) or no `[[counts]]` rows
(counts) is a failure, not a vacuous pass.

The two checkers are separate programs sharing one implementation of the
prose-comparison plumbing: `check_baselines.py` imports `read_text`, `line_of`, `label_of`,
`has_problem`, `require_kind` and `usage_error` from `check_status.py` — the pattern
`check_anchor_drift.py` already uses for `check_doc_line_anchors.py`. Every CI invocation
names its ledger explicitly (`check_status.py --check`, `check_baselines.py --check`,
`check_baselines.py --check-live … --status scripts/test-baselines.toml`), so a rename or a
deletion is a red job rather than a silently disabled gate.

## Alternatives considered

- **One file, two `kind`s, one checker with a `--kind` flag.** Rejected: the two prose
  targets would still share one path, so the file-level predicate the CI needs stays
  unavailable, and one program would carry the union of both halves' CLI surface
  (`--check-live`, `--box`) while only ever using half of it.
- **Split the data, keep one checker.** Rejected for the mirror-image reason: the surviving
  program's name and `--help` would describe half of what it checks, and the counts half's
  live mode would still be a flag of the plan half.
- **Leave the ledgers in `docs/` and special-case them in the CI classifier.** Rejected: it
  keeps a machine record inside the book's source tree, where mdBook publishes it as an
  asset nobody links, and it encodes "this documentation file is really a test input" as a
  classifier exception instead of as a path fact. The cost knowingly accepted in exchange:
  the three markdown links that pointed at `./status.toml` became absolute `github.com`
  URLs, because a relative link out of `docs/` still resolves for `check_docs_links.py`
  (which resolves against the checkout) while 404-ing on the published site (mdBook copies
  only what is inside its `src`).

## Consequences

- A `git diff --name-only` answers "did the suite counts move?" — the input
  `test-linux-cpu` needs to decide whether `--check-live` has anything to verify.
- Editing both halves is two files in one commit when a phase close moves the counts.
  Nothing enforces that pairing; nothing enforced it before the split either, when it was
  one file.
- `docs/status.toml` is gone, and the three live references to it
  (`docs/ARCHITECTURE-EXECUTION-PLAN.md`'s derived-facts block, `docs/TEST-BASELINES.md`,
  `docs/GATE-CONTRACT.md`) point at the new paths by absolute URL. The records that name
  `docs/status.toml` as the file of their day — the execution plan's per-ticket history and
  `docs/SOURCE-LAYOUT-PLAN.md` — are deliberately not rewritten: a record of what was true
  then is not a stale claim.
- The published URLs that served the old ledger stop resolving, which is the intended
  removal of an accident.
- ADR-0022's two `docs/status.toml` citations name a file that no longer exists, which is a
  defect in a frozen ADR's text; it is corrected the way that ADR's own decision requires —
  this ADR carries `Corrects: ADR-0022` and ADR-0022's index row names this one. Neither
  ADR's text was edited.

## References

- `scripts/check_status.py`, `scripts/check_baselines.py` — the two checkers, and their
  `--help` for what each can and cannot prove.
- [`docs/GATE-CONTRACT.md`](../GATE-CONTRACT.md) ("The doc gates") — the gate contract the
  two `--check` invocations serve.
- [`docs/TEST-BASELINES.md`](../TEST-BASELINES.md) — the prose ledger the counts half pins.
