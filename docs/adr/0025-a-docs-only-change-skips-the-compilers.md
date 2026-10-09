# 0025. A docs-only change runs the docs gate, not the compilers

- Status: Accepted
- Date: 2026-10-09
- Issues: #457

## Context

Every `ci.yml` run compiled the crate for three jobs whatever changed. Measured over 13 completed runs on 2026-10-09 (job durations from the API): `build-linux-cuda` median **12.3 min**, `build-macos` 1.9 min, `check-docs` 1.3 min, `test-linux-cpu` 1.1 min, and `check-viz` / `lint-workflows` / `check-pr-body` about 0.1 min each. A docs-only change therefore paid about 16.7 runner-minutes — 14.2 of them in the two compile jobs, with `build-linux-cuda` on the critical path — while 104 of the last 200 non-merge commits changed documentation only, and 8 of the last 15 runs were on `docs/*` branches.

The obvious lever is a path filter. It is also a trap: a workflow skipped by `paths:` / `paths-ignore:` leaves its required checks in `Pending` forever, so the pull request can never merge. A *skipped job* reports `Success` and satisfies a required check.

## Decision

The workflow classifies the diff once, in a `changes` job, and each code job carries a **job-level** `if:` guard. `scripts/classify_changes.py` (with its own `--selftest`) maps every changed path to `docs` / `plan` / `baselines` / `rust` and prints the two predicates the guards read.

- `test-linux-cpu` runs when `rust` or `baselines`; `build-linux-cuda` and `build-macos` run when `rust`.
- `check-docs` and `check-pr-body` are never gated: the first *is* the docs gate, the second reads the PR body and must still run when the body is edited.
- `check-viz` and `lint-workflows` stay ungated: they cost seconds between them, and gating them would buy a second and third predicate to reason about for no measurable gain.

Three rules make the classification safe. **An unrecognised path is `rust`** — a path the classifier has not been taught is a reason to run the jobs, never to skip them. **An unreadable range is `rust`** — an empty `--base`, the all-zero `before` git reports for a first push, a revision that does not resolve, and an empty diff, which is not evidence that nothing changed. And **a `docs/` path a code job reads is carried as an explicit exception**: today that is `docs/dead-code-baseline.toml`, the dead-code oracle's accepted-annotation baseline, compared against rustc's stripped output in `test-linux-cpu` and `build-linux-cuda`; moving it beside its checkers (ADR-0024) deletes the exception, and #461 tracks that. Skipping is only ever reached by proving that every changed path is docs-class or a ledger.

The ledger split (ADR-0023) is what gives `baselines` its meaning: `scripts/test-baselines.toml` is the one docs-adjacent path whose consumer is a code job — its `--check-live` runs in `test-linux-cpu` — so it is the one such change that still runs the suite, while a change to `scripts/status.toml` alone needs no code job at all.

## Alternatives considered

- **Workflow-level `paths:` / `paths-ignore:`.** Rejected: a skipped workflow leaves its required checks `Pending`, and the `edited` event is workflow-level, so the filter cannot even express "skip the builds, keep the body gate".
- **Always run everything.** Rejected on the measurement above: about 14.2 runner-minutes per docs pull request, on a path this repository takes roughly half the time. The minutes are free on a public repository; the feedback latency and the runner queue are not.
- **Skip the heavy *steps* inside each job instead of the job.** Rejected: a job's conclusion is what the required check reports, so a job that checked out and then skipped its work still pays its setup and still shows a green build that never happened.
- **Gate `check-viz` and `lint-workflows` as well.** Rejected: about 15 seconds between them, and each needs its own predicate (`viz/**`, `.github/**`) to be correct rather than inheriting `rust`.

## Consequences

- A docs-only pull request shows the three code jobs `skipped` — reported as `Success`, so the merge button is not blocked — and finishes in about the time `check-docs` takes.
- `push` runs are classified the same way, so a docs-only commit to the default branch also skips the builds.
- The classifier is a new kind of file in `scripts/`: not a checker of content but an input to the workflow. It is the only place the diff is read.
- The docs-class list can rot: a new `docs/**` file that a code job reads would silently become skippable. The audit that prevents it is a `grep` over the scripts the three code jobs run — it is what found `docs/dead-code-baseline.toml` — and the mitigation for a new one is either an explicit exception there or a move per ADR-0024.
- The dead-code oracle (§1 of the gate contract) loses nothing: a new `allow(dead_code)` is a source change, so the classification runs the jobs that compare against the baseline. Only a baseline-only edit could be skipped, and it is exactly the exception above.

## References

- `scripts/classify_changes.py` — the classifier, its `--help` (`What it proves` / `What it cannot prove`) and its 24 self-test cases.
- `.github/workflows/ci.yml` — the `changes` job, its two outputs and the guarded jobs.
- [Status checks](https://docs.github.com/en/pull-requests/reference/status-checks) — "a job that is skipped will report its status as Success".
- [Troubleshooting required status checks](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks) — the `Pending` failure a path-filtered workflow causes.
- [ADR-0023](./0023-the-machine-ledgers-live-beside-their-checkers.md) — the ledger split, and [ADR-0024](./0024-a-machine-read-record-is-not-book-content.md) — where a machine-read record lives.
