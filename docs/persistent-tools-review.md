# Review: persistent / background sub-agents branch

Review of `sebastian/persistent-tools-2026-06-11` (at `7a8abcb`) against `main`,
covering the background/persistent sub-agent feature (`async_runtime.py`,
`agent_limit.py`) and the resumable-engine + session-persistence rewrite
(`engine.py`, `session.py`, `tools/ipython.py`, `api.py`, `types.py`,
`prompt.py`). Findings are bugs first (severity-ordered), then macro
architecture / maintainability, then what was checked and found clean.

Line numbers are as of the reviewed commit and will drift.

## Summary

| ID | Severity | One-liner |
|----|----------|-----------|
| ID | Severity | Status | One-liner |
|----|----------|--------|-----------|
| B1 | HIGH | **fixed** | Dangling assistant `tool_calls` → invalid message sequence → API 400 |
| B2 | HIGH | **fixed** | TOTAL-slot leak in `run()` — acquires sit outside the `try/finally` |
| B3 | HIGH | **fixed** | Resume header written at turn *start* → resumed agent undercounts usage/metrics |
| B4 | MED-HIGH | **fixed** | Torn last line in `messages.jsonl` permanently wedges resume |
| B5 | MEDIUM | **fixed** | Teardown cascade blocks the loop, fixed 120 s, can orphan descendants, swallows errors |
| B6 | MEDIUM | **fixed** | Non-int / empty cap env vars crash on the hot path instead of disabling the cap |
| B7 | MEDIUM | **fixed** | `max_tokens` clamp diverges from the documented formula for `0` / negative |
| B8 | LOW | **fixed** | `close()` never sets a terminal status |
| B9 | LOW | **fixed** | `submit()` has no guard against an ended/closing worker (latent) |
| B10 | LOW | **fixed** | `turn_offset` off-by-one on resume (same root cause as B3) |
| B11 | LOW | **fixed** | Mid-compaction crash can re-open a `branch_reset`-closed view on disk |
| B12 | LOW | **fixed** | `asyncio.get_event_loop()` in `_drain_agents` is deprecated |
| B13 | LOW | **fixed** | Errored resident workers are never evicted from the registry |
| B14 | LOW | **fixed** | Doc/wording: signature-mirroring claim, "deterministic" auto-name |
| B15 | LOW | **fixed** | Depth-0 `send` shared the root session (no `sub-<name>`; `handle.session_dir` None) — external review |
| B16 | LOW | **fixed** | Resume injected the IPython "kernel restarted" warning even for an agent with no REPL — external review |
| B17 | MED | **fixed** | Torn-line resume recovery (B4) was too narrow: a torn final record + trailing blank line still raised — external review |

## Re-check after the config + architecture passes

Implemented (see commits on this branch):

- **Config sprawl** → `rlm/config.py`: one cached, immutable `Config`; behavior-preserving.
- **Macro #4 (kernel bootstrap f-string)** → extracted to `rlm/kernel_bootstrap.py`.
- **Macro #5 (`Registry.send` two-factory + holder-dict)** → single `worker_factory(name)`.
- **Macro #1 (split slot ownership)** → reservation + creation-failure release consolidated
  into `worker_factory`; fixes **B2**.
- **Macro #8 (`BackgroundWorker` lifecycle)** → terminal status on `close()` + `submit()`
  guard (**B8**, **B9**); kept the `_ephemeral` flag (commented) rather than a full split.
- **Macro #7 (`advance()` decomposition)** → extracted `_request_completion`,
  `_note_turn_usage`, `_parse_tool_calls`, `_execute_tool_call`; fixes **B1** (answer
  unanswered tool calls before the next user turn) and **B3** / **B10** (write the resume
  header at end-of-advance, not only at turn start).
- **B4** (torn-line tolerance), **B5** (teardown off the loop + logged; and — after an
  external re-flag — the drain cell now prints a success sentinel that teardown checks,
  so a timed-out / wedged drain is logged and flagged `teardown_drain_complete=false`
  instead of finalizing as a clean shutdown), **B6** (tolerant caps), **B12**
  (`get_running_loop`).

- **B7** → a non-positive `max_tokens` is treated as "no explicit budget" (uses the
  ceiling / no limit) instead of disabling the budget (`0`) or stopping after one turn
  (negative); normalized in both `send` and the engine.
- **B11** → `_resume` derives the view from the loaded content (`load_latest_view` returns
  `(view, msgs)`), so a stale meta.json can't re-open a `branch_reset`-closed branch.
  Residual: a cosmetic orphaned `branch_reset` log line is still possible if a hard crash
  lands in the sub-millisecond gap between two compaction writes (offline analysis only).
- **B13** → re-sending an errored name evicts the dead worker and rebuilds fresh
  (restart / resume), keeping the name reusable and the registry bounded.
- **B14** → skills' `.send` mirrors `run`'s signature + docstring; the `send` docstring no
  longer calls the uuid auto-name "deterministic"; the design-doc claim is corrected.
- **B15** (external review) → the depth gate moved from `child_session` to `run`'s call
  site, so `send` nests a `sub-<name>` session under the current session dir at any depth
  (including the root). A background agent no longer shares the root's `messages.jsonl`, and
  `handle.session_dir` is set. Only reachable via a direct depth-0 `rlm.send(...)` (the
  model's `send` is always depth ≥ 1), so it was latent — but now consistent.
- **B16** (external review) → the resume warning is *accurate* for a REPL agent (every
  resume path follows a genuine kernel teardown — reap/`aclose` or a parent-kernel restart —
  and `setup()` starts a fresh, empty REPL), contrary to the "kernel unchanged" premise.
  But a tools-only / chat-only agent (no REPL) was still told its "IPython session is brand
  new"; the warning is now gated on the engine having a REPL.
- **B17** (external review — correct) → the B4 torn-line tolerance only dropped a corrupt
  line when it was the last *physical* line, so a torn final record followed by a trailing
  blank line raised `JSONDecodeError` and blocked resume. `load_latest_view` now drops
  trailing blank lines before the tail check, so the torn record is recognized as the tail.

Remaining: a hard drain timeout is now detected, logged, and flagged in meta, but
orphaned descendant *kernels* (subprocesses left by a restarted child kernel) aren't
retroactively reaped — that's an OS-level process-tree cleanup left as future work. The
rest of the review is addressed.

---

## Bugs

### B1 — Dangling assistant `tool_calls` → invalid message sequence → API 400 (HIGH)

`engine.py:424` records the assistant message (including its `tool_calls`)
*before* the token-budget check at `:467`, which itself runs *before* the
"no tool calls → done" check at `:476` and the tool-result record at `:501`.

Two triggers:

- **No crash required (the common one).** A resident agent with a `max_tokens`
  budget emits a tool call on the turn whose cumulative completion tokens cross
  the budget. It records `assistant(tool_calls=[X])`, then `break`s with no tool
  result. The agent goes idle (`finished`) with a result; the model re-sends to
  continue; `advance()` appends `user(prompt)` immediately after the dangling
  call → the next `chat.completions.create` is `[…, assistant(tool_calls=[X]),
  user]`, which violates the contract (an assistant message with `tool_calls`
  must be followed by `tool` messages answering each id) → **400**. This bricks
  continuation of any budgeted specialist that stops on a tool-call turn.
- **Crash / restart.** A hard kill during the (potentially slow)
  `asyncio.to_thread(tool.execute)` at `:489` leaves the identical dangling call
  on disk. On re-send after a parent-kernel restart, `_resume` (`engine.py:312`)
  appends `user(warning)` + `user(prompt)` after it → 400 on the first
  post-resume turn.

This is newly reachable because the engine now *continues* a transcript across
sends; on `main`, `run()` was one-shot and finalized, so the dangling call was
never re-sent.

**Fix direction.** At any stop where the last message is an assistant with
unanswered `tool_calls` — and defensively in `_resume` after rehydration —
synthesize a `tool` result (e.g. `content="[interrupted]"`) for each unanswered
`tool_call_id` before the next API turn, or move the budget check above the
assistant-record at `:424`.

### B2 — TOTAL-slot leak in `run()`: acquires outside the `try/finally` (HIGH)

`api.py:47-56`:

```python
total_marker = running_marker = None
if _is_subagent():
    total_marker = await acquire_slot_blocking(TOTAL)     # 49
    running_marker = await acquire_slot_blocking(RUNNING) # 50  ← can raise/cancel here
try:                                                      # 51
    ...
finally:
    release_slot(running_marker)
    release_slot(total_marker)
```

If the coroutine is cancelled while parked inside `acquire_slot_blocking(RUNNING)`'s
`await asyncio.sleep` poll loop (teardown cancellation, an outer `wait_for` /
`TaskGroup`), or `_make_marker` raises `OSError`, *after* TOTAL was granted at
`:49`, the `finally` (which starts at `:51`) never runs → the TOTAL marker file
leaks, permanently shrinking the pool until the long-lived kernel PID dies.
`_RlmProcessor.process` (`api.py:88-102`) does this correctly — the RUNNING
acquire is inside its `try`.

**Fix direction.** Move the acquire block inside the `try`, or wrap it in
`try/except: release_slot(total_marker); raise` (mirroring the pattern already
in `send()` at `api.py:182-187`). Opens only under RUNNING-pool saturation or an
I/O error, but the leak is permanent.

### B3 — Resume header written at turn *start* → undercounts usage/metrics (HIGH)

`_write_resume_header(turn)` is called at the top of each loop iteration
(`engine.py:378`), capturing `_total_usage` / `_metrics.snapshot()` as of the
*end of the previous turn*. The loop exit at `:531` writes nothing more, and
`finalize` persists `usage` but not `metrics_state` / `turn_offset`. Resident
agents don't finalize between sends.

So a resident agent that completes an advance and is later resumed (after a
parent-kernel restart) rehydrates to its **pre-last-turn** usage/metrics — the
final turn of every advance is lost. This defeats the §3.8 guarantee that
per-turn writes mean "a hard restart doesn't undercount the metrics the harness
consumes."

**Fix direction.** Write the header at end-of-turn (after the tool result /
metric updates) and once at end-of-`advance`, and/or persist `metrics_state` +
`turn_offset` in `finalize`. Also fixes B10.

### B4 — Torn last line in `messages.jsonl` permanently wedges resume (MEDIUM-HIGH)

`load_latest_view()` runs `json.loads(line)` per line with no guard
(`session.py:103`). `_write` does one buffered `write()` + `flush()` per line, so
a hard kill mid-write can truncate the final line. The bad line then raises
`JSONDecodeError` → `setup()` raises → the worker `ERROR`s → `Registry.send`
refuses to recreate the name → the agent is un-resumable forever. This is exactly
the crash-recovery path the feature exists for. The codebase already tolerates
malformed lines in `ProgrammaticToolCallStats.from_log` (`types.py:64`).

**Fix direction.** Skip (and log) an unparseable trailing line; the view minus
its last unfinished message is still a valid continuation.

### B5 — Teardown cascade: blocks the loop, fixed 120 s, can orphan descendants, swallows errors (MEDIUM)

`aclose()` (`engine.py:558-564`) runs the drain cell *synchronously* (not via
`to_thread`) under the parent's cooperative loop; each recursion level re-blocks
on the level below with its own `timeout=120`. On drain-cell timeout
`_interrupt_and_recover` may `restart_kernel(now=True)`, orphaning grandchild
kernels — the precise outcome §3.7 claims the graceful recursion prevents. Three
layers of bare `except Exception: pass` (`engine.py:563`,
`async_runtime.close_all`, the drain) discard the failure silently, and no test
exercises the cascade with a live REPL.

**Fix direction.** `to_thread` the blocking REPL calls; make the drain budget
depth-aware (or a single outer cap); don't hard-restart before descendants
drain; `log` instead of `pass`. Add an end-to-end cascade test.

### B6 — Non-int / empty cap env vars crash on the hot path (MEDIUM)

`_limit` (`agent_limit.py:52`) and `_wait_timeout` (`:130`) call `int()` /
`float()` with no guard. `RLM_MAX_LIVE_AGENTS=""` — a common slip when an
external harness templates an unset value to empty string — raises `ValueError`
out of `acquire_slot` on *every* `send` / turn / one-off. This contradicts the
module's own contract ("active only when a positive int … otherwise every
acquire is granted").

**Fix direction.** Parse defensively → treat malformed as disabled / default.
(Folds into the config refactor below.)

**Status (2026-06-13): fixed.** Centralized in `config.py` — the two caps parse
via `_cap` and the wait timeout via `_agent_wait_timeout`, both tolerant (missing
/ empty / malformed → disabled / default). The budgets (`max_tokens`,
`sub_max_tokens`) stay strict (no disabled-on-bad-input contract). Covered by
`test_cap_disabled_on_malformed_limit`.

### B7 — `max_tokens` clamp diverges from the documented formula for `0` / negative (MEDIUM)

`api.py:143-147` only special-cases `None`. With a ceiling set, `max_tokens=0`
→ engine `self.max_tokens=0` → budget check `if self.max_tokens` is falsy
(`engine.py:468`) → **budget disabled** (opposite of intent); `max_tokens=-5`
→ stops after one turn. Doc §3.6 says `min(requested or ceiling, ceiling)`, which
maps `0` → ceiling.

**Fix direction.** Treat non-positive requested as "use ceiling," or validate.

### Low severity

- **B8 — `close()` never sets a terminal status** (`async_runtime.py:180`):
  after teardown `poll()` reports `running` forever and `wait()` raises a bare
  `RuntimeError`. Only reachable at end-of-rollout teardown when nothing polls,
  but the status contract is violated.
- **B9 — `submit()` has no guard against an ended/closing worker**
  (`async_runtime.py:135`): would silently drop the item and set a fake
  `running`. Latent — not currently reachable via the registry (errored workers
  are guarded at `:256`; closed workers are cleared from `_workers`) — but the
  method is unsafe.
- **B10 — `turn_offset` off-by-one on resume** (same root cause as B3):
  non-monotonic turn tags and skewed `turns_since_last_compaction`. Log-tag only.
- **B11 — Mid-compaction crash can re-open a `branch_reset`-closed view**
  (`engine.py:631-648`): internally inconsistent view log; no API impact.
- **B12 — `asyncio.get_event_loop()` in `_drain_agents`** (`api.py:198`) is
  deprecated; fine today (always under a live cell loop), brittle on 3.10+. Use
  `get_running_loop()`.
- **B13 — Errored resident workers are never evicted from `_REGISTRY`**
  (`async_runtime.py:255`): the name is permanently poisoned and the dict grows.
- **B14 — Doc/wording:** "send/poll signatures mirrored onto the wrapped
  callables" is true for `rlm.send` but not skills' `.send` (no
  `functools.wraps` / `__signature__` / doc); `send`'s docstring (`api.py:129`)
  calls the uuid auto-name "deterministic" (it's random `uuid4`).

---

## Macro architecture / robustness

1. **Slot-ownership is split across two objects.** The TOTAL slot is acquired in
   `send()` (`api.py:156`), handed to `_RlmProcessor` via a closure-captured
   `total_marker`, released by `_reap()` — with a *fallback* release in `send()`'s
   `except`. Two leak fixes already landed here (`7a8abcb`, `b6225c1`); B2 is a
   third instance of the same "acquire before the owning object exists" tension. A
   single explicit slot owner, handed off atomically, would retire the class.
2. **Cross-process teardown rides a stringly-typed exec bridge.** `engine.py:560`
   reaches the kernel's registries by `execute`-ing the literal
   `"import rlm.api as _rlm; _rlm._drain_agents()"` with a magic 120 s. Invisible
   to static analysis; see B5.
3. **Persistence format has no atomicity or versioning for `messages.jsonl`.**
   Only `meta.json` is tmp+rename (`session.py:42`). The append log has no
   torn-line tolerance (B4) and no format version field.

## Readability / maintainability

4. **Kernel bootstrap is a ~70-line Python program embedded as an f-string**
   (`tools/ipython.py:162-228`), defining `_CallableModule`, `_wrap_callable`,
   `_log_programmatic_call`. Untestable, no syntax/type checking, interpolates
   paths via `!r` into executable code. Move the body into a real
   `kernel_bootstrap.py` and have `_inject_startup` exec a tiny stub that imports
   it and passes the runtime values.
5. **`Registry.send`'s two-factory + `holder`-dict closure dance** (`api.py:163`,
   `async_runtime.py:239`): collapse to one `worker_factory(name)` built in
   `api.py`.
6. **Config sprawl.** ~22 `RLM_*` vars read ad hoc via `os.environ.get` across
   `engine`, `api`, `agent_limit`, `tools/ipython`, `prompt`, `client`,
   `tools/git_block`, `tools/registry`. No single source of truth; kernel
   propagation relies on implicit subprocess env inheritance plus three explicit
   overrides. Tracked separately as the config refactor.
7. **`advance()` is ~200 lines** (`engine.py:342-538`): the tool-call handling
   block and the metrics bookkeeping are cleanly extractable.
8. **`BackgroundWorker` conflates resident vs ephemeral** via an `_ephemeral`
   flag with subtle post-`clear()` re-checks in `_drain`
   (`async_runtime.py:140-163`): split or heavily comment the invariant.

## Verified clean (where not to worry)

Cleared via live checks: the `flock` critical section + per-pool isolation +
marker parsing + all `send`/error/cancel **slot accounting** (both prior
leak-fix commits correct and complete); **name canonicalization** (the `734b77d`
fix — registry key always matches the session dir); **sync/async preservation**
and `*a/**kw` forwarding in the callable wrapping; **f-string injection escaping**
(pathological cwd/paths compile cleanly); **`ToolState` live-vs-snapshot**
semantics and `poll`-never-consumes; the "tasks progress between cells" substrate
on the installed ipykernel; the parking/wake race; exceptions not leaking as
"never retrieved"; **`write_meta` atomicity**; the `RLMMetrics.snapshot/restore`
roundtrip; and the `request_too_large` break (does not leave a dangling call,
unlike the token-budget break in B1).

## Suggested fix ordering

- **B1** defensive rehydrate / synthesize tool results — highest value.
- **B3** resume-header timing — also resolves B10.
- **B4** defensive JSONL parse, **B6** env-var parse hardening — small, independent.
- **B2** slot acquire-inside-`try` — 3-line structural fix.
