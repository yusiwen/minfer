# 0057. A template file overrides the GGUF copy, and time enters a render through one injected clock

- Status: Accepted
- Date: 2026-10-18
- Issues: #133 (S1, S2), #596 (the no-template promise)
- Amends: ADR-0019 (its `strftime_now` consequence)

## Context

ADR-0019 fixed the refusal contract: a template that exists and cannot be rendered
fails the load, and ChatML survives only for a GGUF with **no**
`tokenizer.chat_template`. It left three gaps that this ADR's work closes, and one
inconsistency it did not see.

**The template had no single owner.** Four entry points each read the GGUF's copy
independently: the single-turn CLI, `--cnv`, `serve` and `viz`. There was no way to
point the engine at a template file at all (`--chat-template` had **0** hits under
`src/`), so a GGUF whose embedded template is wrong or missing had no remedy, and a
run whose template could not render had no escape hatch beyond `--no-template` on the
CLI — `serve` had none.

**Time had no injection point.** `strftime_now` was not registered, so a template that
prints the current date was refused by name (ADR-0019's Consequences said so). Adding
it naively would put a wall-clock read inside a render, which is exactly what the
incremental renderer cannot tolerate: `format_single` diffs two renders, and a render
whose output changes with the clock can never produce a prefix delta.

**The no-template promise was false on one path.** The single-turn CLI printed
`Notice: this GGUF has no tokenizer.chat_template; using the generic ChatML renderer`
and then passed the raw prompt through; `--cnv`, `serve` and `viz` rendered real ChatML
([#596](https://github.com/yusiwen/minfer/issues/596)).

## Decision

**1. One resolver decides the template, and a file beats the GGUF.**
`template::resolve_template(embedded, file) -> Result<Option<String>, TemplateError>`
is the only place the choice is made and the only place either source is validated.
Every entry point calls it; a `--chat-template <FILE>` whose text is unreadable,
empty, or unrenderable is a refusal that names the file (the embedded arm keeps its
construct-and-line refusal unchanged). `Ok(None)` remains the documented ChatML case.

**2. The file override and the bypass are mutually exclusive.** `--chat-template` +
`--no-template` is a startup error naming both flags. The run would carry two
contradictory instructions about the prompt, and silently preferring one is the class
of bug #133 exists to remove. (Same shape as the existing `--cnv` + `--no-template`
refusal.)

**3. Time enters through one injected clock.** `strftime_now(fmt)` is registered
against `environment_with_clock(clock)`; production passes UTC now, and every test
passes a fixed clock, so no gate reads the wall clock. The directive set is
`%Y %y %m %d %H %M %S %F %T %%`; anything else is refused by name, the same policy the
`str` methods follow. A clock-dependent template therefore renders differently on the
two passes of `format_single`, which reports `prefix_matched = false` and makes the
caller do a full re-render — the behaviour §5.4 of the conversation plan already
specified for non-deterministic templates.

**4. The ChatML notice is rendered, not just printed.** The single-turn CLI goes
through the same resolver and the same fallback as the other three paths, so one
artifact produces one prompt (#596).

## Alternatives considered

- **An environment variable (`MINFER_CHAT_TEMPLATE`).** Rejected: the CLI already
  owns a flag namespace for exactly this, `serve`/`viz` take the same flags, and a
  hidden env var would be a second configuration surface `docs/USAGE.md` must
  duplicate (and `scripts/env-vars.toml` must register).
- **Silent precedence (file wins, print a notice).** Rejected: a notice that one
  instruction was ignored is indistinguishable from a bug report, and the two flags
  are cheap to reject.
- **Resolve the file lazily, per request.** Rejected: the refusal must happen at
  startup (ADR-0019's shape) and a template that changes mid-run would invalidate the
  KV delta the conversation path appends.
- **A full `strftime` implementation.** Rejected: it is a locale/formatting subsystem
  (week numbers, zone names, `%c`/`%x`) that no chat template needs; the subset is
  documented, and an unknown directive is a refusal, not a literal. A template that
  needs more is a bug report, not a silent wrong prompt.
- **A thread-local "test clock".** Rejected: the engine's existing rule for
  process-wide state is a per-engine value, and a hidden mutable global is exactly
  what the fixed-clock gate is supposed to rule out.

## Consequences

- `--chat-template <FILE>` works on the CLI, `serve` and `viz`; an unrenderable file
  refuses at startup, with the file named.
- `strftime_now` renders for templates that print a date; the incremental renderer
  takes the full-re-render path for them, and the test proves it with a ticking clock.
- The `str` method set grows by `splitlines`, `ljust`, `rjust`, each with a
  CPython-generated fixture case; `encode` and `format` stay refused **by name** (bytes
  are not a template value; `format` needs a format-spec subsystem).
- The fixture generator is in the tree (`scripts/gen_chat_fixtures.py`), and the gate
  asserts the fixture's method set **equals** the implemented set — the old
  `entries.len() >= 24` floor could not catch a shrink.
- ADR-0019's consequence "templates that call `strftime_now` … are refused" is
  superseded on that one point; the refusal contract itself is unchanged.
- Roadmap item 19 (chat-template fidelity) loses its `remains:` clause; the docs that
  restate the refusal text (`docs/USAGE.md`, the design record) carry the new method
  list. No phase counter moves: `F7`'s `19/20` in the plan's table is the **roadmap
  item range** (items 19 and 20), not a fraction of tickets done.

## References

- `src/template.rs` — `resolve_template`, `environment_with_clock`, `format_civil`,
  `py_splitlines`, `py_justify`.
- `src/main.rs` — `--chat-template`, the mutual-exclusion refusal, the single-turn
  ChatML arm.
- `src/server/mod.rs`, `src/server/viz.rs` — the same resolver before the worker starts.
- `tests/fixtures/chat/pystr_semantics.json` + `scripts/gen_chat_fixtures.py`.
- Issues: [#133](https://github.com/yusiwen/minfer/issues/133),
  [#596](https://github.com/yusiwen/minfer/issues/596).
