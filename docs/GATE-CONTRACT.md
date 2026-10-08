# The gate contract

> **Scope**: what a *gate* is in this repository, and the five rules this
> campaign learned the hard way. A gate is any test whose verdict is a claim
> about the engine — "this path ran", "this value is right", "this is not
> slower" — as opposed to a test that pins a pure function's output. This
> document is the **one home** for those rules: [`AGENTS.md`](../AGENTS.md)
> carries only the point-of-use facts (the command, the wrapper's default, the
> current dated counts) and points here.
>
> The rules are stated once, here. The per-ticket **records** that produced them
> live in [`ARCHITECTURE-EXECUTION-PLAN.md`](./ARCHITECTURE-EXECUTION-PLAN.md),
> [`CUDA-BACKEND-DESIGN.md`](./CUDA-BACKEND-DESIGN.md) and
> [`BUILD.md`](./BUILD.md), and are linked as precedent rather than restated.
> Reading a record is how you check a rule against the instance it came from;
> this page is what you apply while writing the next gate.

## 1. Assert the value, not a relation between two code paths

A gate that compares mode A against mode B is blind to any fault the two modes
**share**. Ask what the assertion would do if the implementation were wrong in
the same way on both sides: if the answer is "pass", the gate needs a value arm.

**Precedent.** `cuda_map_window_matches_the_span_over_the_same_rows` swept
`f32`/`f16`/`q8_0` and compared each mode against another; dropping the Q8_0
block base in `kv4` left it green, because both sides of the comparison read the
same wrong base. A single-row arm that compares the window against the
**dequantized cell** was added and catches it (mutation-checked, [#87], recorded
in [`ARCHITECTURE-EXECUTION-PLAN.md`](./ARCHITECTURE-EXECUTION-PLAN.md) §"All
three acceptance gates, mutation-checked"). Rules 1 and 2 are a pair: the
control arm below is what makes a relative assertion meaningful, and this rule
is what makes an absolute one necessary.

**Apply it.** Every device value gate should include one arm whose expected
value is computed independently of the path under test (a dequantized reference,
a scalar oracle, a pinned literal). A mode-vs-mode comparison is the fast arm,
never the only arm.

The isolation instance of this rule is [#173]: a gate that compared two
snapshots of a **process-global** timing table was replaced by exact values read
from a per-scheduler sink, because a concurrent execution could move the
relation (record: [`ARCHITECTURE-EXECUTION-PLAN.md`](./ARCHITECTURE-EXECUTION-PLAN.md)
§"#173"). The device twin is [#185]: the F5 "host stalls removed" gate read the
process-wide `cuda::stream_sync_count()` and now reads the `CudaBackend`'s own
counter, because a concurrent device test's syncs landed inside the delta (4160
vs 728 in the run that exposed it). Rule 1 is about a shared *code path*; a
shared *destination* is the same hazard, and a value read from an owned table is
immune to it.

**The orphaned-entry-point instance ([#218], [#223]).** A gate must exercise the
**production entry point**, never a test-only helper that mirrors it. [#218] found
that the [#145] gate `cuda_prefill_smem_optin_covers_every_launchable_instantiation`
called `gemm_prefill_smem_init` — an eager startup sweep whose `CudaState::try_new`
call site [#188] had quietly deleted. The helper still worked, so the gate stayed
green while **production performed no such opt-in at all**; the dead-code pass
then annotated the orphan `#[cfg_attr(not(test), allow(dead_code))]` instead of
asking why a production-looking function had no production caller. A mirrored
helper is the same hazard as rule 1's shared code path: the gate and the
production path can be wrong *together*, and the mirror is what makes it look
certified. Ask of every gate: **is the function it calls reachable from a
production entry point?** If the answer is a test-only sweep, the gate's claim is
about the sweep, not the engine — either drive the real entry point (the [#218]
gates drive `gemm_smem_optin` through a real forward and through the production
launcher) or rename the gate so its claim states the mirror it tests. The
campaign corollary: a dead-code diagnostic that `allow`s a test-reachable
production-looking item is a **question deferred**, not a warning silenced.

That question has two legitimate answers. Deleting the item is one. [#223] took
the other and restored the production call — `CudaState::try_new` drives the eager
pre-warm again, through the *same* per-instantiation cache the launcher reads, so
the attribute is set before any `CudaBackend` (the only holder of a capture
window) can exist. Annotating the item without answering the question is neither.

## 2. A control arm must differ in the property under test

A negative control that is rejected by an *earlier* check never exercises the
check the gate is about. The control must be constructed so that the only thing
that can refuse it is the property under test.

**Precedent.** The q4_K `W_dsc` plane gate ([#165], [#167]) originally used a
`q8_0` payload as its negative. A `q8_0` payload is longer than a q4_K payload,
so the **payload-length** check rejected it before the **type** check ran — the
gate could not see a type-gate bypass. The fix was an **equal-ratio q4_0 arm**:
`q4_0` has q4_K's exact bytes-per-element ratio (`od * (id / 32) * 18` in the
unit test), so only the type gate can refuse it. See
`models::weight_reg::tests::q4k_weights_register_the_dsc_plane_only_under_every_gate`.

**Apply it.** For each condition in a conjunctive rule, ask which *single* arm
can fail only because of that condition. If the control is refused earlier (a
length check, an alignment check, an unset flag), it is not testing the
condition.

**The degenerate-input instance.** An arm can also fail to differ when the
*fixture* lets the difference cancel, even though the code under test is wrong.
[#186]'s new decode arm compared the packed `int` K dot against the
dequantized-f32 reference — but it stored a **single** KV cell, so the softmax
had one key and the K score cancelled out of the output entirely; mutating the
`__dp4a` block base (`elem >> 5` → `elem >> 4`) still passed. The fix is to make
the difference *observable* through the arm's own data — here, two cells read
through an explicit `[0, 2)` span, so the score reaches the softmax weights — and
then the same mutation is red (`max |Δ| = 0.35126442`). This is the same family
as [#145]'s self-clearing assertion: before trusting a control, ask what the
fixture does to the quantity the control is supposed to move, and assert that the
quantity can move at all.

## 3. Every gate needs mutation evidence

Break the thing the gate guards — with the implementation, not the test — watch
the gate go red, revert. A gate that has never failed is a gate that has not
been shown to test anything.

**Precedent.** Five separate gates in this campaign passed for the wrong reason
and were found only by deliberately breaking the implementation: relation-blind
(rule 1), a self-clearing assertion, process-global registry contamination, a
wrong-message assertion, and a wrong-name assertion. The records are in
[`CUDA-BACKEND-DESIGN.md`](./CUDA-BACKEND-DESIGN.md) (the #147, #151, #165
hardening sections each end with a "Mutations" paragraph) and
[`ARCHITECTURE-EXECUTION-PLAN.md`](./ARCHITECTURE-EXECUTION-PLAN.md).

**The seam makes it cheap.** [`src/testfail.rs`](../src/testfail.rs) is the one
failure-injection switch (issue [#171]). One environment variable at the command
line replaces the bespoke mock the earlier tickets each had to write:

| site | chokepoint | what the mutation looks like |
|---|---|---|
| `forward_batch` | `server::chat::guarded_forward_batch` | a panic through the real guard → the 500 path (#151) |
| `alloc_in_pool` | `graph::alloc::GraphAllocator::alloc_in_pool` | the allocation refuses before the pool is touched |
| `execute_node` | `graph::scheduler::BackendScheduler::execute` | the backend dispatch refuses |
| `register_weight` | `models::weight_reg::register_cuda_weight` | the CUDA weight registrar panics |
| `metal_cross_copy` | `graph::metal_backend::MetalBackend::cross_enqueue` | the staging copy's `MTLSharedEvent` signal is suppressed, so phase B's **bounded** wait takes its real timeout branch (#137) |
| `launch:*` / `attr:*` | `src/cuda/kernels/*.cu` (`minfer_launch_ok` / `minfer_smem_optin`) | the **real** CUDA call is driven into failure (#147) |

The matching rule is exact-token and comma-separated
(`MINFER_TEST_CALL_FAIL=forward_batch,launch:gemm_f16_f16`; `all` for every Rust
site). The seam is **off whenever the variable is unset**, which is CI, every
normal run and the `compute-sanitizer` run; the property is pinned by
`testfail::tests::the_seam_is_off_by_default`. It is **test-only**: a chokepoint
is an ordinary call that returns `Ok`/does nothing in production.

**The observation half.** A gate that must prove a path *executed* cannot read
the path's own answer — a dispatch function naming its branch is
self-certifying. `testfail::note_checked(site)` / `checked(site)` are bumped **by
the chokepoint itself** (the shape #141's vectorized f16 dot needed:
`F16_SIMD_PATH_CALLS`, asserted by
`vec_ops::tests::f16_dot_uses_the_vectorized_path`). A gate asserts the counter
advanced instead of trusting the dispatch's report.

**The work-bound twin (#160).** Rule 4's progress assertion has its own
mutation lever: `MINFER_TEST_TICK` drives `BatchEngine::tick` into one of two
faults. `=wedge` returns from the step without advancing `work_units`, so the
per-step progress assertion fires on the step that wedged the engine; `=spin`
advances the counter but never completes a run, so only `step_budget` catches
it. Both are read once per process and unset in every production, CI and
default run, exactly like `MINFER_TEST_CALL_FAIL`.

**Honest scope — presence is checkable, truth is not.** A script can require
that a mutation transcript *exists* in a PR body or a record; it cannot check
that the mutation was real, that the gate was the one that failed, or that the
revert was byte-identical. Rule 3 is a discipline, not an enforced property. The
durable part of [#171] is making the experiment one line, because the cost is
what made the discipline get skipped. The machine-checkable half of the sibling
rule 5 (a suite-count consistency check) stays separate, on [#94].

## 4. Bound a gate's runtime by work, not by seconds

A verdict that depends on how fast the machine was at that moment is a verdict
about the machine. Absolute deadlines and single wall-clock ratios both failed
on a loaded box in this campaign.

**Precedent.** [`BUILD.md`](./BUILD.md) §Tests and
[`ARCHITECTURE-EXECUTION-PLAN.md`](./ARCHITECTURE-EXECUTION-PLAN.md) §"#154" /
§"#158" hold the records:

- [#154]: `server_batch_matches_serial_and_is_faster` measured two whole
  workloads once, sequentially, and asserted `t_serial > t_batched`. Under a
  parallel harness the first phase absorbed the start-up wave: **21.20s batched
  vs 9.95s serial = 0.47x** in parallel against **1.50x** serially. It now
  interleaves matched rounds and asserts the **median of the per-round
  `serial/batched` ratios** > 1.0, so a verdict is robust to up to half the
  rounds being disturbed.
- [#158]: `published_metrics_move_as_requests_are_served` bounded a run by
  absolute wall-clock deadlines and asserted the engine had gone idle, so 16
  extra CPU spinners made it panic (**28 passed / 1 failed**). It now bounds
  **work**: `BatchEngine::work_units` must advance on every step that leaves the
  engine busy, plus `step_budget(prompt, max_tokens)`. The same spinner run is
  green, and failure detection went from **120s to 0.18s**.

**Apply it.** Prefer a counted invariant (steps, rows, tokens, work units,
entries walked) over a clock. When a timing relation is unavoidable, use
**interleaved matched rounds** and a **median** (never two sequential sums), fix
the round count in advance, and print every per-round value so a loaded result
is auditable. A process-hang watchdog may keep a generous timeout, but it must
not be a gate's only failure signal.

**[#160] finished the sweep [#158]'s audit opened.** Every `while engine.busy()`
stepper in the `#[ignore]`d server gates now runs through one shared per-step
`WorkBound` — a progress assertion plus `step_budget` — so a wedge in
`BatchEngine::tick` fails the gating step in seconds instead of hanging the
suite. **[#196] closed the case [#160] could not:** the two gates that drive the
**production** `serve_loop` were outside any in-test bound, because a wedge kept
the loop busy and the test thread never returned. `serve_loop` itself now carries
the same invariant as a production guard — `STALL_STEP_LIMIT` consecutive steps
that left the engine busy without advancing `BatchEngine::work_units` end the
loop, answer every live **and** queued request once with a `500 server_error`, and
move the `minfer_worker_stalled_total` counter — so both gates are wedge-proof
with no test-side deadline, and a wedged server no longer spins at 100% CPU with
clients left hanging. The wall-clock bounds that remain are *named* and say at
the call site why each is only a backstop: the `serve_loop` feeder's poll
terminator (`FEEDER_POLL_BACKSTOP`, now the last resort for a worker whose
published metrics never settle rather than the only way out of a wedge), and the
two `run_cli` child-process ceilings (1800s for the real-model sessions, 60s for
the no-model registry cases) are cross-process hang guards, env-overridable
through `MINFER_CLI_WATCHDOG_SECS`. The mutation seam is `MINFER_TEST_TICK`
(see §3); the dated records and transcripts are [#160] and [#196] in
[`ARCHITECTURE-EXECUTION-PLAN.md`](./ARCHITECTURE-EXECUTION-PLAN.md)
§test-infrastructure.

## 5. A device number carries its date, device and command

A count or a timing without provenance cannot be audited and cannot be
corrected when it drifts.

**Precedent.** The `AGENTS.md` suite counts drifted twice by hand-editing; [#94]
records three instances of the class. The fix is not more diligence, it is
making every number self-describing.

**Apply it.** Write device and suite numbers as
`<counts>, <device>, <command>, <date>` — for example
`36 passed / 0 failed, dgxspark (aarch64, GB10 sm_121), FEATURES=cuda scripts/real_model_gates.sh, 2026-09-25`.
A number whose command you cannot name is not evidence; delete it rather than
re-state it.

**Name the box absolutely.** A record is read on machines other than the one that
produced it — an agent on an x64 CUDA box reads `AGENTS.md` too — so `box` is the
*machine's own* name, never a relative one: no `this box`, `my machine` or
`local`. The labels in use today are `dgxspark (aarch64, GB10 sm_121)` (the
maintainer's DGX Spark — one physical machine, so its CPU and CUDA rows share the
label) and `x86_64 (CI runner)` (GitHub's runner). A newly used machine gets its
**own** label — its hostname, or another equally stable id, plus the arch/device
that matters for the counts — and its own `[[counts]]` rows; a suffix or a
re-used label would make two machines' numbers indistinguishable. The label is
part of the machine-checked identity: it is the `box` field in
[`docs/status.toml`](./status.toml) and the `--box` argument of
`scripts/check_status.py --check-live`, so a rename must move both the manifest
and the prose in the same commit. A box whose name has no manifest row fails
`--check-live` loudly, which is deliberate: a fabricated box must not pass
vacuously.

**The PR body's `Mac verification` section is this rule on the one path CI
cannot run** ([#335]). `build-macos` compiles the crate and — since [#303] — the
test target, but it has no Metal device, so on the macOS/Metal path the
**Mac-local run is the evidence**. The template requires the section
([`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md),
enforced by the `check-pr-body` job and
[`scripts/check_pr_body.py`](../scripts/check_pr_body.py)), and its content
convention is the one stated above: `<box label> / <date> / <command>` with the
machine's own absolute label, or `N/A — <reason>` when the change touches no
macOS-only file and no shared layer's macOS arm. It is the place where *what was
verified on which box, and what was not* becomes part of the record — the record
a Linux-green PR otherwise lacks. Why it earns a section rather than a sentence
in someone's memory: the predicate is not confined to `src/metal*` —
`git grep -F 'cfg(target_os = "macos")' -- src/` finds **111 sites in 36 files**
today (the issue that introduced the section measured 35 / 108 on 2026-10-07;
[#299] and [#329] moved the row), and they include shared layers
(`graph/alloc.rs`, `graph/kvcache.rs`, `graph/kvformat.rs`, `graph/scheduler.rs`,
`graph/backend.rs`, `graph/registry.rs`, `graph/fusion.rs`,
`models/weight_reg.rs`, every `models/*/graph*.rs`) — while the macOS-only
assertions no Linux job executes are the **53** unit tests the current recorded
rows differ by (macOS `541` − aarch64 `488`, `docs/status.toml` 2026-10-07) and
**11** integration tests (macOS `21` − aarch64 `10`, the four
`#![cfg(target_os = "macos")]` binaries in `tests/`), plus the op matrix's Metal column and every performance number. The
checker enforces that the section is *present and filled*; whether it is *true* is
the reviewer's job ([#175]).

## Prose anchors: name the section, not the line range

This is a **documentation convention**, not a sixth gate rule — it belongs here
because `check_doc_line_anchors.py` is the gate it supports, and because the
reason it exists is the gate's own boundary.

A `path:NNN` anchor in the docs is checked for three questions, each of which **fails** the
run: does the file and the line exist; does a backticked *symbol* sitting next to the
anchor still live within ±25 lines of it ([#266]); and is that symbol **inside the cited
range** (rule E, [#339]). The ±25-line window is loose on purpose (a symbol 20 lines away
is a neighbour, not a move), which is why the narrower in-range claim is checked
separately: a symbol that is *in the file but not in the cited range* fails, reported as
`range-miss` in the summary and by `--list` with `[range-miss: …]`. A call-site citation
is not a miss: if the range names the symbol where it is *called*, the identifier is
inside the range and the anchor passes.
Neither test can see a range that still *resolves* but no longer holds the **text** the
citing sentence describes, and that is not a bug to fix with a further rule: measured
for [#327], **173** anchors carry an adjacent backticked **non-symbol** token, and
**99** of the 144 live ones do not contain that token inside their own range — 69 % of
the population, and the tokens are mostly code expressions (`ins[1][t].to_bits() as
usize`, `nt >= 9`) or fragments of a *neighbouring* quoted sentence, never the
claim. Even the narrowest useful spelling ("≥2 pure-lowercase words") leaves one
hit — an anchor citing `build.rs` for `` `ar rcs` `` that no longer held it — and
that one *was* a real drift, which is the point: the detectable subset is a
coincidence, not a rule. Rule E is the *symbol* half of that question, where the
adjacent token is mechanical; the non-symbol half stays a convention.

So the convention is:

- **A prose anchor points at a section heading.** Cite the heading by name and
  number (`docs/COMPUTE-GRAPH-DESIGN.md` §7.3 "In-place execution and the
  aliasing rule") next to the range, so a reader who lands on the wrong text has
  a stable second locator. A heading rename is a visible edit; a 200-line insert
  is not.
- **Prefer a symbol anchor where one exists.** `check_doc_line_anchors.py` rule C
  is the one content-aware test that is mechanical: put the backticked symbol
  next to the anchor and the checker follows it through a move. A bare range is
  the fallback, not the default.
- **A symbol anchor's range must hold that symbol** (rule E). Write the symbol
  *and* cite the lines that contain it — a definition citation starts **at** the
  definition, and a `NNN-MMM` range must not start hundreds of lines early. A
  trailing `()` is part of the token, so `` `metal_available()` `` is checked
  like `` `metal_available` ``. If the sentence is about a *call*, cite the call
  site; the identifier is there too. When neither is possible, cite a section
  heading (the bullet above) rather than a range that holds something else.
- **A range that quotes prose is the weakest form.** Do not backtick the quoted
  words to "make them checkable": the adjacency rule was written for identifiers,
  and applying it to prose judges the neighbouring sentence (measured above).

Rule E shipped as a **note** from [#339] to [#371] because of what a failure would have
cost then *and* because it does not catch the whole class. Measured on `7991218`, **79**
non-frozen anchors carried an adjacent backticked symbol whose identifier was not inside
their own range (**84** with the trailing `()` stripped; 46 cited a `NNN-MMM` range and
38 a single line, and 81 of the 84 hold the identifier in the target file at all),
and they spanned 18 docs — 21 in `13-decode-loop-graph-reuse.md`, 12 in
`cuda_tutorial/02-minimal-cuda.md`, 11 in `14-metal-backend.md`, 7 in
`ARCHITECTURE-ROADMAP.md`. **26** of the 84 lived in walkthrough docs that state the
revision their lines were verified against (`lines verified at commit e7fa0da`), where
re-pointing the anchor would falsify the record — the same reason the `FROZEN` set
exists. [#371] re-pointed all 81 — the revision-pinned docs included, whose claim was
re-stated at that PR's base rather than frozen — and **promoted the rule**: a range miss
fails the plain `check-docs` run, and `--strict-symbols` promotes only the two heuristic
classes (``symbol-far`` / ``symbol-foreign``). And two of the three anchors [#339] was
filed for are *invisible* to the rule:
`:256` cited `metal_backend.rs:316-1075` for `execute_node`, a range that **does**
contain the definition at 843 (containment cannot see a range that starts 527 lines
early), and `:586`'s `synchronize` anchor carries no adjacent symbol at all
(`` `self.submit_pending()` `` is not an identifier), so no adjacency rule can reach
it. Both were re-pointed by hand in the [#339] PR, as was the `()`
(`:526` cites `metal_backend.rs:1943-1945` now). What remains of the class is its
**other half** — a range that carries no symbol at all, so no adjacency rule can see
whether it still holds the text its sentence describes; that bare-range sweep is
[#336] and [#356].

The remaining bare ranges are a tracked sweep, not a silent one: **878** of the
955 resolved anchors are bare ranges across 33 docs (`docs/cuda_tutorial/**` alone
carries 276), and converting them is its own ticket.

### The drift rule: re-point by the map, in the same PR

Rule C follows a symbol through a move, but a bare range has nothing to follow: a
commit that inserts a line in an anchored file leaves every anchor below it
*resolving* and *wrong*. That is the gap
[`scripts/check_anchor_drift.py`](../scripts/check_anchor_drift.py) ([#344]) closes,
and it runs in `check-docs` on `pull_request` events. It diffs `HEAD` against the
PR's **merge base** — `git diff -U0 origin/<base_ref>...HEAD`, which is why that
job's `fetch-depth: 0` is load-bearing twice — reduces each file to its hunks, and
compares the base revision's anchors with the head tree's. Anchors are paired on
the doc line with its numbers normalised (so a re-pointed line still pairs), which
is [PR #343]'s own proof (`":NNN" -> ":N"` multisets identical) applied per line:

- the pair does **not** carry the mapped numbers → **stale**, printed as
  `doc:line → target:old (now new)`, exit 1;
- a cited *endpoint* was deleted by the range, so the map has no image for it →
  **ambiguous**: reported with the reason, never guessed, and promoted to a
  failure by `--strict`;
- the range never touched the target, or the doc already carries the mapped
  numbers → silent.

The rule for a PR author is therefore: **moving a line in an anchored file obliges
you to re-point every anchor into it, by the mapped delta, in the same PR** — and
the cheapest way to need that less often is rule C above, a symbol anchor the map
cannot strand. The gate exists because the rule was missed twice in one round:
[#329]'s dispatch change moved **7** anchors off by one (docs 07 and 14), and
[#299]'s weight accounting moved **69** across eight docs by 1–3. Both were caught
by a hand-written old→new line map, which is exactly what the gate replaces — and
which it beats: re-run over [#299]'s range it named **73**, the 69 the re-point
commit fixed plus **4 it missed**. Those four are cited here by the revision that
carried them, not by a live anchor: the PR that added this section re-pointed all
of them, so the four doc lines that held the stale numbers now hold the corrected
ones — line 98 and line 452 of `docs/ARCHITECTURE-ROADMAP.md`, line 56 of
`docs/SOURCE-LAYOUT-PLAN.md` and line 705 of
`docs/inference_e2e_walkthrough/03-model-dispatch-weights.md`, where the cited
`graph/alloc.rs` range moved from `134-137` to `135-138`. Read that line at
`506b26c` to see the defect; read it today to see the fix.

Two boundaries are deliberate. A base anchor whose doc line was rewritten *beyond*
its numbers is not compared — the author touched that line, and pairing it would be a
guess; `--list` prints it as `not compared`, so "why did this pass?" has an answer.
The `FROZEN` set above is the *same* set in both modes: a record frozen against a past
revision is not drift. The other boundary — a bare `:NNN` continuation — is no longer a
blind spot: it is rule D below.

### A bare `:NNN` continuation attaches to the anchor before it

A citation often names a file once and then a second range beside it:

```
`docs/ARCHITECTURE-ROADMAP.md:NNN` … `graph/alloc.rs:NNN`, `:MMM`
```

The `:MMM` is a **bare continuation** — a backticked span whose whole content is `:NNN`
or `:NNN-MMM`. `ANCHOR` requires a `path.ext`, so before [#355] **neither**
`check_doc_line_anchors.py` nor `check_anchor_drift.py` could see it: a re-point that
fixed the visible `graph/alloc.rs:NNN` left `:MMM` behind, silently. That is not
hypothetical — it is how [#347]'s seven rows survived `387fe91` and `da7a35a`, and
`check_anchor_drift.py` reported green over each. (The numbers are placeholders: this
page states the rule, it does not cite a revision, so it carries no live anchor for the
gates to keep re-pointing.)

The convention, stated once here and implemented as rule D:

- **A bare continuation attaches to the nearest *preceding* `path.ext:NNN` match on the
  same doc line.** "Nearest preceding" is exact: the window is **one doc line**, the
  direction is **backwards only**, and the nearest earlier anchor wins — so a line
  carrying two path anchors and two continuations attaches each number to the anchor
  immediately before it, not to the first one on the line. The continuation is then
  resolved and range-checked against **that** file, exactly like a written `path:NNN`,
  and the drift gate maps it through that file's hunks in the same way — a stranded
  continuation is `doc:line (bare continuation) → path:old (now new)`, exit 1.
- **A continuation that follows no anchor on its line stays silent.** So does one that
  comes *before* every anchor on its line, and one whose path anchor is itself
  external, ambiguous or frozen. The rule is syntactic, so the silent class is not a
  guess about intent: nothing on the line names the file, and attaching to the nearest
  anchor in *either* direction is wrong where it is not empty. Measured on `f754ee3`,
  the two standalone `:1289-1321` spans in `docs/ARCHITECTURE-ROADMAP.md` (its line 204
  and its line 619) each continue a file named on the *previous* line, while the anchor
  that follows on their own line names `metal_backend.rs` — so a forward window would
  fail both, naming the wrong file, and the backward one has nothing to attach to.
- **The window is the line, not the paragraph or the table cell.** Measured on
  `f754ee3` with the issue's criterion, the docs carry **58** bare continuations on
  anchor-carrying lines: **38** resolve, **11** were out of range (the visible path
  anchor had been re-pointed by the [#261]/[#263] split and the continuation left at
  its pre-split absolute number — fixed in the [#355] PR), **2** are in the frozen
  `docs/ARCHITECTURE-EXECUTION-PLAN.md`, and **7** are silent by the two rules above.
  The `FROZEN` exemption covers bare continuations exactly as it covers path anchors.
  **[#367] swept this class:** the seven silent spans were re-written in the explicit
  form — five live citations, the two in the frozen `docs/ARCHITECTURE-EXECUTION-PLAN.md`
  left under that exemption — so the **live silent count is 0**. The checker's own counts
  moved with it: `unattached` 93 → 89 and `checked` 1045 → 1051, because a written path is
  an anchor the gates can see. What stays silent is not a citation: this page's own
  §"Prose anchors" illustrates the rule with the two doc-line numbers of the [#339] rows,
  and the frozen `docs/ARCHITECTURE-EXECUTION-PLAN.md` keeps its two under the exemption
  above. Writing a path is the whole fix; none of them would become a citation by a wider
  window.
- **Prefer the explicit form when the two citations are in *different* files.** The
  rule attaches the number to the nearest anchor, not to the file the sentence meant:
  `03-kernels-elementwise.md`'s ``(`src/cuda/methods.rs:NNN`, launch at `:MMM`)`` meant
  the launcher that the [#262] split moved to
  `src/cuda/methods/prefill_f16.rs`, so the fix names that file. **The explicit form is
  what [#367] chose for the residual**: widening the window to the paragraph or the table
  cell was rejected, because a *forward* window is exactly wrong in the case that
  motivated the rule — the two standalone 1289-1321 spans in
  `docs/ARCHITECTURE-ROADMAP.md` continue a file named on the *previous* line, while the
  anchor that follows on their own line names `metal_backend.rs`. [#336] is the
  neighbouring sweep: bare ranges that carry no symbol at all, so no rule can see whether
  they still hold the text their sentence describes.

The convention for the *content* of a range is unchanged from §"Prose anchors" above: a
heading or a symbol is still the stable locator, and a bare range is still the fallback.

## Writing the next gate — checklist

1. What **value** does it assert, and how is that value computed independently
   of the path under test? (rule 1)
2. Is the function the gate calls reachable from a **production entry point** —
   or is the claim about a test-only mirror? (rule 1, the [#218] instance)
3. Which **single arm** can fail only because of the property under test, and
   is it refused by an earlier check? (rule 2)
4. What is the **mutation** — which implementation line do you break, and does
   the transcript show *this* gate going red? Use
   `MINFER_TEST_CALL_FAIL=<site>`; if the counter is what certifies the run,
   `testfail::note_checked` it too. (rule 3)
5. Is the runtime bounded by **work**? If not, are the rounds interleaved and
   the assertion on a median that prints its inputs? (rule 4)
6. Does every number in the PR body carry its **date, device and command**?
   (rule 5 — on a macOS/Metal change that is the `Mac verification` line)

## How the shape is enforced

The facts rules 3 and 5 ask a PR to state in prose have a mechanical floor, and
so does the Mac record of §5. The PR body is rendered from
[`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md), and
[`scripts/check_pr_body.py`](../scripts/check_pr_body.py) refuses a body whose
required headings are missing or whose three checked sections — `Bar named
before measuring`, `Mutation evidence` and `Mac verification` — are empty or
left at their template placeholder (a stated `N/A — <reason>` is filled). The
`check-pr-body` job in [`ci.yml`](../.github/workflows/ci.yml) runs it on
`pull_request` events only, after the checker's own `--selftest` cases. It reads
the body through `env:`, so it needs no token and works on a fork PR ([#175],
[#335]). This is the shape, not the rules: the rules stay stated once, above.

## Honest scope

This document is a contract, not a linter. Nothing in CI parses this page: rules
1, 2 and 4 are review discipline and rule 3's *truth* is unverifiable by
construction. What the repository does enforce is the concrete part — the seam is
off by default under a CI-covered test, `check_docs_links.py` keeps this page
reachable, the per-ticket records keep the instances auditable, and the
`check-pr-body` job requires the checked facts — the two gate sections and the
Mac record — to be *present* in the PR body, never to be true (§"How the shape
is enforced"). Treat the rules as the questions a reviewer must be able to
answer from the PR, not as property tests.

<!-- Issue references, linked once so the text above stays readable. -->
[#87]: https://github.com/yusiwen/minfer/issues/87
[#94]: https://github.com/yusiwen/minfer/issues/94
[#154]: https://github.com/yusiwen/minfer/issues/154
[#158]: https://github.com/yusiwen/minfer/issues/158
[#160]: https://github.com/yusiwen/minfer/issues/160
[#196]: https://github.com/yusiwen/minfer/issues/196
[#165]: https://github.com/yusiwen/minfer/issues/165
[#167]: https://github.com/yusiwen/minfer/issues/167
[#171]: https://github.com/yusiwen/minfer/issues/171
[#173]: https://github.com/yusiwen/minfer/issues/173
[#185]: https://github.com/yusiwen/minfer/issues/185
[#175]: https://github.com/yusiwen/minfer/issues/175
[#186]: https://github.com/yusiwen/minfer/issues/186
[#145]: https://github.com/yusiwen/minfer/issues/145
[#188]: https://github.com/yusiwen/minfer/issues/188
[#218]: https://github.com/yusiwen/minfer/issues/218
[#223]: https://github.com/yusiwen/minfer/issues/223
[#261]: https://github.com/yusiwen/minfer/issues/261
[#262]: https://github.com/yusiwen/minfer/issues/262
[#263]: https://github.com/yusiwen/minfer/issues/263
[#266]: https://github.com/yusiwen/minfer/issues/266
[#327]: https://github.com/yusiwen/minfer/issues/327
[#299]: https://github.com/yusiwen/minfer/issues/299
[#303]: https://github.com/yusiwen/minfer/issues/303
[#329]: https://github.com/yusiwen/minfer/issues/329
[#335]: https://github.com/yusiwen/minfer/issues/335
[#336]: https://github.com/yusiwen/minfer/issues/336
[#367]: https://github.com/yusiwen/minfer/issues/367
[#344]: https://github.com/yusiwen/minfer/issues/344
[#347]: https://github.com/yusiwen/minfer/issues/347
[#355]: https://github.com/yusiwen/minfer/issues/355
[#339]: https://github.com/yusiwen/minfer/issues/339
[#356]: https://github.com/yusiwen/minfer/issues/356
[#371]: https://github.com/yusiwen/minfer/issues/371
[PR #343]: https://github.com/yusiwen/minfer/pull/343
