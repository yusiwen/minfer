<!--
Fill every section. The CI `check-pr-body` job (scripts/check_pr_body.py)
requires all seven headings, and refuses a body whose two gate sections are
empty or left at their `FILL-ME:` placeholder. The rules these sections serve
live in docs/GATE-CONTRACT.md; a stated `N/A — <reason>` is a filled section.
The checker can force the sentence to exist, never to be true.
-->

## What changed

<!-- One bullet per behaviour change; name the file or symbol. -->

## Why

<!-- The decision, and the alternative that was rejected. -->

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
