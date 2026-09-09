# CUDA optimization step documents — writing contract (all authors must follow)

This directory expands every optimization step of `docs/CUDA_OPTIMIZATION.md`
into a standalone, readable document.
Intended reader: an engineer familiar with GPUs and inference engines who has
NOT followed this campaign.

## Language and style

- **Prose in English** (workspace doc spec: English is the default for all
  documentation). Code, kernel names, env vars, commit hashes, and proper
  nouns (wmma / cp.async / MMVQ / occupancy…) stay as-is.
- Written to be UNDERSTOOD: no telegraphic bullet dumps. Every number carries
  its source; every conclusion carries its mechanism.
- Repeating key background is fine (each doc must read standalone), but do not
  copy-paste large tables between docs.

## Fixed per-doc structure (Markdown; order must not change)

```markdown
# NN · <step name> (LANDED / REVERTED / CLOSED)

> **Result**: one-line numeric conclusion (e.g. 7B @2K prefill 30.7 → 294 tok/s).
> **Commit**: `<hash>` (write "no repo change" if none). **Date**: YYYY-MM-DD.

## 1. Background — where things stood
What stage the engine was at, what the previous milestone achieved, what the
bottleneck was, why this direction was chosen.
(3-8 paragraphs; make clear where the engine stalls WITHOUT this step.)

## 2. Principle — the GPU mechanism
Why this change can / cannot make things faster. Argue with arithmetic:
byte counts, GB/s, SM count, wave count, occupancy, latency chains.
If an ascii diagram doesn't fit, write the equation.
Define new concepts on first use in one or two sentences (dp4a, ldmatrix,
the W_exp plane…).

## 3. Implementation
### 3.1 Design choices (why this shape and not another)
### 3.2 Key code
Real code excerpts (10-40 lines each, from `git show <hash> -- <file>` or the
current tree), before/after pairs, annotated segment by segment.
**Do not paste blocks over 100 lines** — pick the core loop / dispatch logic.
### 3.3 Pitfalls
Concrete traps hit along the way: compiler behavior, alignment, aliasing,
barriers — recorded as they happened.

## 4. Verification
Which gates ran: bitwise dumps, greedy token-identity, the test suite,
interleaved A/B measurement, ncu/nsys evidence.
One sentence per gate on what it defends against.

## 5. Results
Kernel-level and wall-clock numbers (before → after), and the comparison
target (llama.cpp ca3d5a3e1 or the engine's own baseline).
REVERTED/CLOSED steps: state the measured numbers and the veto mechanism.

## 6. Lessons
1-4 transferable rules, one sentence each.
```

## Hard rules

0. **Code-extraction forensics protocol (prevents context flooding)**: prefer
   Grep on the **current tree** to locate the function → Read a bounded line
   range → excerpt 10-40 lines. Use `git show <hash> -- <file> | sed -n 'START,ENDp'`
   only when a pre-change version is genuinely needed (locate with `--stat`
   first, then a narrow range). **Never** pull whole files or large diffs into
   context. ≤3 git calls per doc; if a hash won't resolve twice, fall back to
   narration + current-tree code and flag it at the top of the doc.
1. **Only write `docs/cuda_optimization_steps/NN-*.md` files**; do not touch
   source files, do not commit (the coordinator commits). **Pace: write each
   doc to disk immediately after its forensics; one doc at a time — never
   batch everything to the end.**
2. Numbers and conclusions **must** come from the corresponding section of
   `docs/CUDA_OPTIMIZATION.md` + its master-table row; do not invent. Source
   line ranges are given in the task.
3. Code excerpts come from `git show <hash> -- <path>` or bounded reads of the
   current tree; after excerpting, cross-check against the source commit.
4. File naming: `NN-english-dash-slug.md`. NN is assigned by the task; do not
   change it.
5. Length: LANDED major steps 250-450 lines; REVERTED/CLOSED probes 120-250
   lines. Prefer fewer-but-real lines over padding.
6. A REVERTED step's "Results" section must state the veto mechanism (why it
   was reverted, and under what future conditions a retry is worthwhile).
7. End every doc with the nav line: `← NN-1 · Index · NN+1 →` (omit at the
   ends of the range; the index link is `./README.md`).
