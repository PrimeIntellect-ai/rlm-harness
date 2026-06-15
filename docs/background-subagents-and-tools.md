# Background sub-agents, programmatic tools, and the resumable engine

This document describes what this branch adds on top of `main`. The headline is
a **background / persistent sub-agent layer**: from inside an IPython cell the
model can launch a recursive sub-agent (or a skill call) in one tool call and
poll it in a later one, and can stand up a *named* sub-agent and hold a
multi-turn conversation with it. That rests on an `RLMEngine` that can be
advanced and resumed across calls, an append-only transcript it rehydrates from,
two cross-process caps, and a centralized configuration module.

## At a glance

- **Background API.** Every programmatic callable wrapped into the kernel — the
  recursive `rlm` agent and each uploaded skill — gains `send(...) -> Handle`
  (start work, return immediately) next to the existing `await ...` call. The
  model polls handles across tool calls. `rlm` agents can additionally be
  *named* and continued for a multi-turn conversation.
- **Resumable engine.** `RLMEngine.run()` is split into `setup()` / `advance()`
  / `aclose()`, so one engine drives many turns and can rehydrate from disk after
  its in-memory state is gone.
- **View-structured transcript.** `messages.jsonl` is a typed, append-only log of
  *views* the engine reloads on resume.
- **Two cross-process caps** on sub-agents — total live and concurrently running
  — enforced with marker files.
- **Centralized config** (`config.py`), the kernel bootstrap moved out of an
  exec'd string into a real module (`kernel_bootstrap.py`), and a naming cleanup
  (no leading-underscore module files or module-level helpers).

New modules: `async_runtime.py`, `agent_limit.py`, `config.py`,
`kernel_bootstrap.py`.

## How backgrounding works

The layer rests on properties of the IPython tool:

- The **kernel namespace persists across cells**, and the **kernel's event loop
  runs between cells** — a task created with `ensure_future(...)` and not awaited
  keeps progressing while the model is in other cells (`nest_asyncio.apply()` is
  applied at kernel startup). A backgrounded worker therefore makes progress
  between polls.
- The engine already runs `tool.execute` via `asyncio.to_thread`, so several
  sub-agents run concurrently on one loop (async model calls + threaded REPL
  I/O), and within-cell parallelism (`await asyncio.gather(rlm(a), rlm(b))`)
  works. `send` adds the cross-cell lifecycle on top.

## Background API: `send` / `poll`

One background primitive, uniform across `rlm` and skills: `send` input to a
callable and `poll` it at will. `rlm` has no special surface — it just produces
many results over time instead of one.

- `X.run(*a, **kw)` — async; await to completion for the bare return value.
  `await rlm("subtask")` works via `__call__` and returns an `RLMResult`
  (`.answer`, `.usage`, `.turns`, `.session_dir`).
- `X.send(*a, **kw) -> Handle` — sync; enqueue input to a (possibly new)
  background worker and return a handle immediately. `rlm.send` also takes `name`
  (a continuation key; `None` → a random uuid hex) and `max_tokens`.

**Handle:**

- `.poll() -> ToolState` — sync snapshot of dynamic state; a pure read that never
  consumes a result.
- `.wait()` — async; await and consume the next result.
- `.session_dir` — for `rlm`, the agent's session dir
  (`session_dir/"messages.jsonl"` is the live transcript the model can tail);
  `None` for callables without a transcript.

**`ToolState`** (what `poll()` returns):

- `status` — `running` / `finished` (idle) / `error`.
- `results` — a FIFO of the callable's return values, drained by the model
  (`results.popleft()`); `poll()` never consumes, so a status check can't eat a
  result. For `rlm` the items are `RLMResult`.
- `queued` — the live, directly-editable pending inbox (a list): the model can
  cancel, reorder, or edit not-yet-started items. Race-free because the kernel
  loop is single-threaded cooperative — a synchronous edit in a cell can't
  interleave with the drain coroutine, which only yields at `await`, and the
  in-flight item has already left the list.
- `error` — the live exception object if the worker halted, else `None`.

`status` and `results` are independent: `finished` with unread results buffered,
and `error` with a good result still in the FIFO, are both valid.

Each wrapped callable's `send` carries its underlying signature and docstring
(`rlm.send` is the real function; a skill's `send` mirrors its `run`), so
`help(<callable>.send)` and `inspect.signature` surface the real arguments;
`poll` / `wait` are `Handle` methods. There is no per-name lookup API — the model
keeps handles in its own variables (the kernel namespace persists across calls)
and re-`send`s a name to continue.

## Workers, handles, and the registry

`async_runtime.py` provides the engine-agnostic background machinery:

- A **worker** runs a **processor** and exposes state through a `Handle`. Two
  lifecycles:
  - **Resident** (named `rlm` agents): owns an inbox, drains it sequentially as
    one long-lived task, parks when idle, and stays alive until `close()`. Held
    by the registry so a re-sent name continues it.
  - **Ephemeral** (skills / general callables): runs its single item once and
    ends. Owned solely by its `Handle`, never registered, so it and its result
    are garbage-collected when the model drops the handle.
- A **processor** turns one queued item into a result: the stateless processor
  runs `await fn(*a, **kw)`; `rlm`'s stateful processor runs
  `await engine.advance(prompt)` on a live engine, so re-sending the same name is
  a multi-turn conversation with one persistent specialist.
- A per-kernel **registry** holds resident workers with strong refs (asyncio
  won't keep a parked task alive otherwise). Per-kernel makes nesting
  hierarchical: each agent owns the registry of the children it spawned; there is
  no global registry, and ephemeral workers never touch it.
- A worker that **errors** halts in `error` state with the exception live on
  `poll().error`. Re-sending that name **restarts** it: the dead worker is
  evicted and a fresh one is built under the same name (for `rlm`, resuming the
  agent's transcript from disk — see *Resume*), so a name stays reusable and the
  registry doesn't accumulate dead workers.

### Queueing

Re-`send`ing a busy named `rlm` agent appends to its inbox; the worker drains
sequentially after the current turn (no preemption). `poll().status` stays
`running` while the inbox drains, `poll().queued` exposes the editable pending
items, and each item's result lands in `results` in order. Skills are not
registered, so two `send`s to one skill run as two concurrent ephemeral workers.

## Skills in the background

`attach_background(module, run)` gives each wrapped skill a stateless
`send(*a, **kw) -> Handle`: each call runs `run(*a, **kw)` on its own ephemeral
worker. Skills get no persistence — a skill that wants state across calls keeps
its own cache. The injected `send` mirrors `run`'s signature and docstring so
`help(<skill>.send)` shows the real arguments.

## The resumable engine

`RLMEngine` is split so one conversation can be advanced repeatedly:

- `setup()` — depth check, ensure the session, write `meta.json`, start the
  IPython kernel (only when the `ipython` tool is active), and seed `_messages`
  with the system prompt — or rehydrate from disk when a transcript already
  exists (see *Resume*). Idempotent.
- `advance(prompt) -> RLMResult` — append a user turn and run the loop until the
  model stops calling tools. Repeatable on the same live kernel. The loop body
  delegates to focused helpers (`_request_completion`, `_note_turn_usage`,
  `_parse_tool_calls`, `_execute_tool_call`).
- `aclose()` — drain background sub-agents, finalize the session, shut the kernel
  down.
- `run()` = `setup()` + one `advance()` + `aclose()` — the blocking one-off path.

`rlm`'s processor calls `advance` on a live engine; REPL teardown for resident
agents is deferred to the end-of-rollout cascade (see *Teardown*).

Before appending a new user turn, the engine answers any **dangling tool calls**
left at the tail of the transcript — an assistant turn whose `tool_calls` were
recorded but whose results were not (a turn cut off by the token budget, or a
crash mid-execution that resume reloads). It synthesizes a tool result for each,
so the next request is never an invalid `assistant(tool_calls) → user` sequence.

## Transcript and sessions

Sessions nest along the call tree, one directory per agent:

```
<root-session>/sub-<name>/messages.jsonl
<root-session>/sub-<name>/sub-<child-name>/messages.jsonl   # sub-of-sub
```

- A background `rlm` agent always gets its own `sub-<name>/` session under the
  current session dir, at any depth (the root included); the one-off `run` path
  nests only for a sub-agent, so the root rollout keeps using its own session
  rather than nesting under itself. `handle.session_dir` and each
  `RLMResult.session_dir` point at it.
- Names are unique within a parent, not globally. Model-supplied names are
  sanitized filesystem-safe; an omitted name becomes a uuid hex. Directory
  discovery globs `sub-*`.

`messages.jsonl` is a typed, append-only log the engine rehydrates from:

- A **view** is the message sequence the model saw within one compaction branch.
  Each turn is appended to the current view; on compaction the checkpoint prompt
  and the assistant summary are appended to the closing view, then a new view is
  opened seeded with `[system, user(framing + summary)]`.
- Line types: `msg` (full OpenAI shape incl. `tool_call_id`s, tagged with view
  index + turn), `branch_reset` (compaction stats), and event lines (`spawn`,
  `done`). The log is rlm-local — nothing outside rlm reads it.
- `meta.json` is written atomically (temp + rename); per-turn it carries a resume
  header (usage, `turn_offset`, `view`, `branch_start_turn`, and a full
  `RLMMetrics` snapshot), and at finalize the answer preview, usage, turns, and
  aggregated metrics.

## Resume and restart resilience

A cell that times out and won't yield to an interrupt triggers a kernel restart,
which re-runs the startup injection and wipes the kernel's Python state. Two
situations are handled:

- **Kernel-reset note (the agent's own cell).** When a cell forces a restart, the
  ipython tool result ends with a note that the REPL was reset — variables,
  imports, and in-memory state are gone, rebuild what's needed. The conversation
  is unaffected (it lives in the engine, not the kernel).
- **Resume from disk.** A sub-agent's `RLMEngine` lives in its *parent's* kernel
  process. If that kernel restarts, the registry and every in-memory engine are
  gone but the `sub-<name>/` dirs persist. Re-sending a name then builds a fresh
  engine whose `setup()` finds an existing transcript: it rehydrates `_messages`
  from the latest on-disk view and restores the resume header from `meta.json`
  instead of seeding `[system, user]`. Appending to one growing
  `sub-<name>/messages.jsonl` is therefore correct — one agent, one transcript.

`load_latest_view()` returns `(view_index, messages)` for the highest view, and
the engine treats that index as authoritative for `_view` (so a stale
`meta.json` can't re-open a closed branch). Recovery from an interrupted write is
built in:

- A torn final record (a hard crash mid-write), with or without trailing blank
  lines, is dropped and the rest of the view is recovered; a malformed line with
  valid content after it is treated as real corruption and raised.
- If a crash interrupts compaction so the new branch holds only its `system`
  message (the `user(summary)` seed not yet written), resume falls back to the
  previous complete branch rather than a lone system prompt; the next turn
  re-compacts.

When the resumed engine has a REPL, it injects a user-turn warning that the
conversation survived but the IPython session is brand new; a tools-only or
chat-only agent (no REPL) gets no such warning.

The transcript is not the training signal: the trainer builds samples from the
inference engine's recorded token steps, not from `messages.jsonl`. A resume (or
the injected reset warning) at worst starts a new training-sample split, the same
thing compaction already does.

## Caps on sub-agents

Two independent pools bound sub-agents, enforced cross-process via marker files
in subdirectories of a per-rollout directory shared by the whole process tree.
The root derives `RLM_LIVE_AGENTS_DIR` from its session; it propagates to every
kernel. Each held slot is one `<pid>-<uuid>.marker` file; an `flock` serializes
the sweep-count-create, and a PID-liveness sweep on each acquire reclaims slots
leaked by hard-killed processes. A pool is active only when its limit is a
positive int and the markers dir + POSIX `fcntl` are available; a missing, empty,
or malformed limit disables it (every acquire is granted).

- **Total** (`RLM_MAX_LIVE_AGENTS`) — resident agents; a slot is held from
  creation to teardown, bounding how many sub-agents (and kernels) exist at once.
  `send` raises `AgentLimitReached` at the cap — a creation-time failure, distinct
  from the `poll().error` runtime channel. The recourse is to reuse an existing
  agent (re-sending a name takes no new slot).
- **Running** (`RLM_MAX_RUNNING_AGENTS`) — agents executing a turn; a slot is held
  only around one `advance()` and acquired blocking. This is the parallelism knob.

An errored turn frees both slots (the running slot in its `finally`, the total
slot while reaping the kernel, keeping `poll().error` intact). The one-off path
(`await rlm(...)`, `gather(...)`) holds both slots for its lifetime and blocks on
each; a safety valve (`RLM_AGENT_WAIT_TIMEOUT`, default 300 s, `0` = forever)
proceeds over the cap rather than deadlock when slot-holders are themselves
waiting for slots (e.g. a parent turn awaiting a nested child). The root rollout
(depth 0) is never capped.

## Teardown

A parent's `aclose()` drains its live sub-agents, finalizes its session, then
shuts its kernel — in that order, so child metrics are counted before the kernel
dies. The drain runs as a cell executed in this engine's kernel (where the child
registries live), which calls `close_all_registries()`; each child finalizes its
own session and recursively drains its grandchildren. The cell runs off the event
loop (via `to_thread`) so a slow drain doesn't stall sibling agents.

The REPL turns a cell timeout into captured output rather than an exception, so
the drain cell prints a success sentinel and teardown checks for it: an
incomplete drain (timeout, wedged child, or deep tree) is logged and recorded as
`teardown_drain_complete=false` in `meta.json` rather than finalizing as a clean
shutdown. (Descendant kernels orphaned by a restarted child kernel are not
retroactively reaped — an OS-level concern.)

## Token budget

The cumulative completion-token budget (`RLM_MAX_TOKENS`) is checked per turn in
`advance`, stopping with `stop_reason="token_budget"`. `rlm.send`'s `max_tokens`
is clamped to a ceiling: `effective = min(requested, ceiling)`, with
`RLM_SUB_MAX_TOKENS` as both the ceiling and the default when `max_tokens` is
omitted. A non-positive request is treated as "no explicit budget" (it falls back
to the ceiling, or to no limit when no ceiling is set) rather than disabling the
check or stopping the agent immediately. Wall-clock budgets are out of scope.

## Configuration

`config.py` reads every `RLM_*` variable once, in `load_config()`, into a cached,
immutable `Config`. Code reads it via `get_config()` or an engine-held copy
(`RLMEngine` accepts an optional `config` and folds its constructor kwargs in);
`reload_config()` re-reads after the environment changes (CLI flags, tests).

Environment variables remain the transport across process boundaries: each
IPython kernel is a fresh subprocess that builds its own `Config` from the
inherited environment, and the startup injection writes the three per-kernel
values (`RLM_SESSION_DIR`, `RLM_DEPTH` incremented, `RLM_LIVE_AGENTS_DIR`) before
rlm is imported there. Two cohesive domains keep their own resolution and are not
in `Config`: provider credentials (`client.resolve_provider`) and the active
builtin-tool set (`tools.registry`).

Knobs introduced or used by this layer:

- `RLM_MAX_DEPTH` — recursion depth; the model only gets `rlm` (and the
  `send`/`poll` API in its system prompt) when `depth < max_depth`.
- `RLM_SUB_MAX_TOKENS` — completion-token ceiling and default for `rlm.send`.
- `RLM_MAX_LIVE_AGENTS` / `RLM_MAX_RUNNING_AGENTS` — the total / running caps.
- `RLM_AGENT_WAIT_TIMEOUT` — blocking-acquire safety timeout (default 300 s;
  `0` = forever).
- `RLM_LIVE_AGENTS_DIR` — marker-file directory; derived by the root and
  propagated, not set by hand.

## Code structure

- **Kernel bootstrap (`kernel_bootstrap.py`).** The skill / `rlm` wrapping logic
  (`CallableModule`, `wrap_callable`, `build_namespace`,
  `log_programmatic_call`) lives in a real, importable module. The startup
  injection runs a small stub that sets the per-kernel env, applies
  `nest_asyncio`, and merges `build_namespace(...)` into the kernel's globals.
- **`advance()` decomposition.** The turn loop delegates to `_request_completion`,
  `_note_turn_usage`, `_parse_tool_calls`, and `_execute_tool_call`.
- **Metrics.** `RLMMetrics` gains `snapshot()` / `restore()` so a resume header
  can carry and rebuild metric state.
- **Naming.** Internal module files and module-level functions / globals carry no
  leading underscore (this is an internal package); class methods and instance
  attributes keep theirs.

## Trajectory and training

Sub-agent API calls are tagged `X-RLM-Depth >= 1` (in `client.make_client`) so an
interceptor can drop them from the parent's recorded trajectory — backgrounding
never leaks sub-agent turns into it. A background spawn or poll is just a fast
tool result and creates no parent branch. Poll results depend on wall-clock
progress, so rollouts are timing-dependent; exact replay is not a goal.

## Non-goals

- Wall-clock / time budgets (token budgets only).
- Determinism / virtual-time concurrency / exact rollout replay.
- An interruption / preemption system (interrupting the model's thinking to inject
  context).

## Tests

`test_async_runtime.py` (worker runtime), `test_agent_limit.py` (caps),
`test_persistent_agents.py` (send/poll, slots, resume, teardown), plus additions
to `test_tools.py` (sessions, view recovery) and `test_prompt.py` (the advertised
API).
