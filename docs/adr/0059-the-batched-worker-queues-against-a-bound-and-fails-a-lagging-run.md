# 0059. The batched worker queues a request against a bounded depth, and a run that makes no progress is failed rather than waited on

- Status: Accepted
- Date: 2026-10-19
- Issues: #150, #198, #157
- Supersedes: the `#121` plan record's *"reject loudly; queueing is a feature"* (a
  `docs/ARCHITECTURE-EXECUTION-PLAN.md` measurement record, not an ADR — the record
  stays as history and is corrected by this decision)

## Context

Two refusal decisions were made quickly, each correct for its own ticket, and together
they left the server's admission story inconsistent.

`#121` made a saturated batched worker answer `503` instead of the 200-with-empty-body
silent drop it had before. That was the right fix for the defect it named — the client
must be told — but it also settled, by implication, that the batched path would never
queue: `serve_loop` drained the channel and called `admit` on every pass, so a job that
found every slot busy was answered, not held. The **serial** path (`MINFER_BATCH=0`)
never had that property: its one worker pulls a job at a time and the handler's
`send().await` is the backpressure, so a second request waited. The two transports
disagreed about what "all slots busy" means, and `docs/OPENAI-CHAT-API-PLAN.md`'s
pre-E2 intent — *defer the task in the request queue, never reject with 503* — was
implemented nowhere.

The second gap was liveness. `#196` bounds a `tick` that leaves the engine busy while
`work_units` stops moving. It cannot see the other arm: a step that *keeps advancing*
the counter while no request ever completes (#160's `MINFER_TEST_TICK=spin` is that
shape made reproducible). The counter it watches is global, so any bound built on it
alone is either blind to a run the worker steps around or false-positives on a long
generation.

The third was observability: `/metrics` could not say a request had been answered with
an error. `requests_completed_total` rises for a `500` exactly as for a served answer,
and `jobs_dropped_total` counts admission, not the answer a client got.

## Decision

**1. Queue, with one number as the bound.** `serve_loop` admits at most one job per
**idle** slot, oldest first, and keeps the rest in its own deque. The depth is
`MINFER_QUEUE_MAX` (default `QUEUE_MAX` = 64, the job channel's own capacity): one knob,
so a client refused by the worker's bound and one refused by the transport's block at
the same depth. A job beyond the bound is answered `503 unavailable_error("the queue is
full")` — the **newest** one, so the oldest waiters keep their place — and counted in
`jobs_dropped_total`, exactly one per refusal.

**2. `admit` is placement, not policy.** `serve_loop` hands it a group no larger than the
idle-slot count; `admit` keeps its per-job placement and prefill sharing, and its
no-idle-slot arm stays as the defensive answer for a caller that breaks the contract. The
counter arithmetic is unchanged: `requests_total - jobs_admitted_total` is the backlog
(channel + deque), and a refused job still leaves the queue, so the two sides balance.

**3. A drain answers what it will not run.** When the drain flag is set, every
queued-but-unadmitted job is answered `503 unavailable_error("server is draining")` at
once. Without this, a queue deeper than the drain deadline's worth of work would be
abandoned unanswered when the deadline expired — the silent drop `#121` removed, one
layer up.

**4. Liveness is per run, and a queued job is not a run.** The engine stamp each run with
`work_units` at its last attributable progress (a row written for it, or a token it
committed). A run whose lag reaches `RUN_STALL_STEP_LIMIT` (64) is answered
`500 server_error("the run made no progress")`, its slot is freed, and the worker carries
on with the queue. `#196`'s per-step bound is unchanged and still owns the frozen-counter
arm; the two answer different wedges and one request gets one answer. A **queued** job
has no `Run`, so waiting can never trip a stall bound — that distinction is structural,
not a timeout.

**5. A terminal error is counted where it becomes a response.** Two places —
`collect_response`'s `Err` and the SSE error frame — bump
`minfer_requests_failed_total{class="…"}` with `ApiError::error_type` as the label. The
split with the other counters is deliberate: `requests_rejected_total` is a refusal
**before** queueing, `jobs_dropped_total` is an **admission** refusal (including this
decision's bound), `requests_completed_total` is "left the worker whatever the outcome",
and `requests_failed_total` is *how* it ended. Counting at the worker's eight send sites
was rejected: a ninth site would be silently uncounted.

## Alternatives considered

- **Keep rejecting on the batched path and make the serial path reject too.** Rejected:
  the serial path's `503` branch is unreachable (one job at a time means every slot is
  idle when it looks), so "make it reject" is not a code toggle but a handler-side
  admission check — and it would remove queueing the serial path already provides.
- **An unbounded queue (llama.cpp's task queue).** Rejected: a client would wait
  indefinitely for a slot, memory would follow load, and the drain deadline would
  abandon work it could have answered. A bound is what makes the refusal nameable.
- **A wall-clock wait bound.** Rejected by `docs/GATE-CONTRACT.md` rule 4: the two
  liveness bounds are counts of steps, never seconds, so the verdict is a property of the
  code rather than the machine.
- **A global "no completion in N steps" bound.** Rejected: an unbounded `max_tokens`
  request legitimately runs thousands of steps, and a global count cannot tell that from
  one run the worker steps around.

## Consequences

- A saturated server now has one documented policy on both transports: wait, then be
  served; the only errors are the queue bound and a drain. `docs/USAGE.md` §"Slot
  saturation", `docs/FEATURES.md`, `docs/OPENAI-CHAT-API-PLAN.md` §"Slot Lifecycle" and
  §"Error Types", and the `#121` record all state it.
- Backlog memory is bounded by `queue_max` jobs; the transport channel adds at most its
  own capacity before `send().await` blocks.
- A wedged slot costs one request (the per-run bound) instead of the whole worker; a
  frozen engine still costs every live run (`#196`), because there is no way to attribute
  a frozen global counter to one run.
- `minfer_worker_pending_jobs` becomes a real gauge, not a transient, and
  `minfer_queue_depth` becomes a real backlog.
- The `#160` `spin` mutation run no longer needs its two `serve_loop` gates `--skip`ped:
  they now fail in under a second, and the new per-run gate asserts the bound itself.

## References

- `src/server/batch.rs` — `serve_loop`, `QUEUE_MAX`, `queue_max_from_env`,
  `QUEUE_FULL_MESSAGE`, `DRAINING_MESSAGE`, `idle_slots`, `Run::last_progress_work`,
  `RUN_STALL_STEP_LIMIT`, `fail_lagging_runs`.
- `src/server/metrics.rs` — `requests_failed`, `record_failed`, the labelled render.
- `src/server/mod.rs` — `collect_response`, the SSE frame (the two counting points).
- `src/server/chat.rs` — the startup line that prints the queue depth.
- Issues: [#150](https://github.com/yusiwen/minfer/issues/150),
  [#198](https://github.com/yusiwen/minfer/issues/198),
  [#157](https://github.com/yusiwen/minfer/issues/157); the decision this replaces is
  [#121](https://github.com/yusiwen/minfer/issues/121)'s record.
