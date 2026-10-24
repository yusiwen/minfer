<!--
Fill every section. The CI `check-pr-body` job (scripts/check_pr_body.py)
requires all eight headings (plus `Doc impact` when a live document cites a
changed file), and refuses a body whose checked sections are empty or left at
their `FILL-ME:` placeholder. The rules these sections
serve live in docs/GATE-CONTRACT.md — the `Mac verification` convention is
stated in its §5, next to the box-label rule it reuses; a stated
`N/A — <reason>` is a filled section. The checker can force the sentence to
exist, never to be true.
-->

## What changed

<!-- One bullet per behaviour change; name the file or symbol. -->

## Why

<!-- The decision, and the alternative that was rejected. -->

<!-- When the `changes` job flags a decision surface, the `ADR:` line below must answer
     it: `ADR: 0011` (this implements that decision), `ADR: supersedes 0011`, or
     `ADR: none — <why this change is not a decision>`. See docs/adr/README.md ("When an
     ADR is required"). The checker enforces the line's presence and shape, never its
     truth. -->
ADR: none — FILL-ME: why this change is not a decision

## Verification

<!-- One line per command: the exact command → its count. -->

## Bar named before measuring

<!-- The numeric bar, stated BEFORE you run the measurement (gate contract rules 3 and 5). -->

FILL-ME: the numeric bar, named before measuring

## Mutation evidence

<!-- The command that broke what the gate guards, its non-zero exit and its message. -->

FILL-ME: the mutation command, its non-zero exit and its message

## Honest scope

<!-- What could not be verified here, and which CI job covers it instead. -->

## Follow-ups

<!-- Issues filed for what this leaves open, or `N/A — <reason>`. -->

Closes #

## Doc impact

<!-- When the `changes` job reports `doc_cited=true`, a live document anchors into
     a file this PR changes; `scripts/doc_impact.py --base origin/master` prints the
     paragraphs. State what happened to them: re-read and still true / updated in this
     PR / `N/A — <reason>`. The section is required only when `doc_cited` is true. -->

FILL-ME: what happened to the citing paragraphs, or `N/A — <reason>`

## Mac verification

<!-- The macOS/Metal path this change touches and how it was checked, or why it
     needs no Mac: `<box label> / <date> / <commit> / <command and counts>` (the box
     label is absolute, gate contract rule 5) or `N/A — <reason>`. CI's `build-macos`
     only compiles the crate and the test target — the hosted runner has no Metal
     device at all (#674) — so this line is the record of what was run on which box.

     N/A is REFUSED when the range touches src/metal/**, src/graph/metal_backend*,
     build.rs or src/models/*/graph.rs (#678): run `scripts/mac_gate.sh` on the Mac
     and paste what it prints. -->

FILL-ME: `<box label> / <date> / <command>`, or `N/A — <reason>`
