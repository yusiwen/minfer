# 0030. Launch severity lives in the helper, not in 120 call sites

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #162

## Context

Every `<<<>>>` in `src/cuda/kernels/*.cu` has to read its own launch error. The obvious form is a check at
each site: 120 launch owners, 104 of them unchecked before #162, each needing its own status read and
error path. The other obvious form is to change every launcher's signature to return a `Result` — 104
signatures, and Rust-side callers that must then thread the error through.

`docs/CUDA-BACKEND-DESIGN.md` §4.9 states the rule that came out of the ticket: **kernel-invariant
failures return `Err` from `execute_node`, never a silent fallback**, and the site census is in the plan's
#162 record (120 sites in 76 `launch_*` owners; 107 required → `Err`, 13 documented fallbacks).

## Decision

**The severity decision lives in the helper, once per launcher, and exactly one Rust-side check drains
it.** The two helpers differ in what a failure means:

- `minfer_launch_ok` is **required**: it records a *sticky* failure, and `CudaState::take_launch_failure`
  drains it in `CudaBackend::execute_node` on **both** arms, returning `Err` that names the site and the
  node — **one Rust-side check, not 104 signature changes** — so the op never proceeds on a stale output.
- `minfer_launch_ok_opt` names and clears for a path with a **documented fallback**. Of the 120 sites,
  **107 are required** and **13 are documented fallbacks**, each named in `scripts/check_cuda_launch_returns.py`'s
  audit and pinned by the `MINFER_TEST_ISSUE162=1` injection lever.

The site is a token, not a file: the helper takes a `launch:` site string so the error names which
`<<<>>>` failed, and the audit script rejects a site whose token does not appear in the source.

## Alternatives considered

- **Thread a `Result` through every launcher (104 signatures).** Rejected in the record: it is the same
  work with a wider blast radius, and every caller would have to choose a severity anyway — the choice does
  not disappear, it just moves to 104 places instead of one helper plus a 13-entry list.
- **A check at each launch site (120 call sites).** Rejected: it is what the 104 unchecked sites were, and
  it makes "did this one read its status?" a question about every future edit rather than about one helper.
- **Treat every failure as fatal (`minfer_launch_ok` everywhere).** Rejected: 13 paths have a documented
  fallback by design, so the required/optional split is the fact the audit needs — a blanket helper would
  have to lie about those 13 or remove a working path.

## Consequences

- The audit is one script over one helper pair, not 120 sites: `scripts/check_cuda_launch_returns.py` fails
  when a new `<<<>>>` does not read its own error, and `MINFER_TEST_ISSUE162=1` drives every audited site
  into a real failing launch in a test.
- A new launcher must choose a severity, and the choice is visible in the source: the 13 fallbacks are
  enumerated rather than inferred.
- Cost accepted: the sticky-failure drain is one extra Rust-side call in `execute_node`, and a failure that
  no later call drains would be lost — which is why `take_launch_failure` is called on **both** arms.

## References

- `docs/CUDA-BACKEND-DESIGN.md` §4.9 (the rule) and §7.3's `Recorded verifications` stub;
  `docs/ARCHITECTURE-EXECUTION-PLAN.md`'s #162 record (the census and the helper's semantics).
- `scripts/check_cuda_launch_returns.py` — the audit; [#162](https://github.com/yusiwen/minfer/issues/162)
  — the ticket; [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
