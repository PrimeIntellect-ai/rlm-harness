# Background, persistent sub-agents and programmatic tools

`rlm` lets a model spawn recursive sub-agents and call uploaded skills from
inside an IPython cell. This document describes the background layer on top of
that: the model can launch a sub-agent or skill call in one tool call and check
on it in a later one, and can stand up a *named* sub-agent and hold a multi-turn
conversation with it across tool calls. It also covers how that survives kernel
restarts, how it is bounded, and how it stays compatible with training.

---

## 1. Motivation

A plain sub-agent call (`await rlm("subtask")` from an IPython cell) blocks the
cell until it finishes, and the engine is one-shot: `RLMEngine.run()` builds
`[system, user]`, runs to completion, and tears down its kernel. Two gaps follow:

- **No cross-call lifecycle.** The model can't start work in one tool call and
  check on it in another, and long sub-agent work risks being killed by the
  per-cell exec timeout (`RLM_EXEC_TIMEOUT`, capped at 600s in `IPythonREPL`).
- **No persistence / multi-turn.** The model can't stand up a *specialist*
  sub-agent and ask it repeatedly.

The headline capability is **persistent, named sub-agents the parent holds a
multi-turn conversation with** — create a specialist, keep asking it, and stop
sending to abandon it — plus quick one-offs. The same background/poll mechanism
serves every programmatic callable (skills); for those there is no persistence,
only offloading.

### Constraints from training

`rlm` is a training-only harness, so the design preserves:

- **Black-box sub-agents.** Sub-agent API calls are tagged `X-RLM-Depth >= 1`
  (`client.py:make_client`) and dropped from the trajectory the trainer sees
  (`RLMEndpoint.trajectory_visibility`; composable `_keep_only_parent_rlm_steps`).
  Backgrounding never leaks sub-agent turns into the parent trajectory.
- **Trajectory cleanliness.** The parent trajectory is a sequence of
  `(assistant tool-call, tool result)` pairs. A background spawn or poll is just
  a fast tool result; it creates no parent branch (unlike compaction).
- **Concurrency is accepted as nondeterminism.** Poll results depend on
  wall-clock progress, so rollouts are timing-dependent. Large batch sizes
  denoise this; exact rollout replay is not a goal.

---

## 2. Substrate

The background layer rests on properties the IPython tool already provides:

- **Engine loop** (`engine.py`, `RLMEngine.advance`): one tool call per turn
  (`parallel_tool_calls=False`); builtin tools `ipython` / `bash` / `edit`.
- **Long-lived kernel** (`tools/ipython.py`): a subprocess kernel whose
  **namespace persists across cells** (a handle created in one cell is still
  there in the next) and whose **event loop runs between cells** — a task created
  with `ensure_future(...)` and not awaited keeps progressing while the model is
  in other cells (`nest_asyncio.apply()` runs in `_inject_startup`). This is what
  lets a backgrounded sub-agent make progress between polls.
- **Blocking I/O is already offloaded.** The engine runs `tool.execute` via
  `asyncio.to_thread`, so several sub-agents run concurrently on one loop (async
  model calls + threaded REPL I/O), and within-cell parallelism
  (`await asyncio.gather(rlm(a), rlm(b))`) already works. The new capability is
  cross-cell lifecycle.
- **Startup injection** (`_inject_startup`): wraps each skill and a callable
  `rlm` into the namespace (`_wrap_callable` makes a module `await`-callable →
  `mod.run(...)`) and sets `RLM_SESSION_DIR` / `RLM_DEPTH` / `RLM_LIVE_AGENTS_DIR`
  for child sessions.
- **Sessions** (`session.py`): nested `sub-<name>/` dirs, each with `meta.json` +
  `messages.jsonl`; metrics aggregate up the tree (`aggregate_child_metrics`,
  which globs `sub-*`).
- **Cell output** surfaced to the model is the kernel's captured `stream`
  (stdout/stderr), `execute_result`, `display_data`, and `error` (traceback).

---

## 3. Design

### 3.1 API surface

One background primitive, uniform across every callable (`rlm` and skills):
`send` input to a tool and `poll` it at will. `rlm` has no special surface — it
just produces many results over time instead of one.

- `X.run(*a, **kw)` — async; await to completion and get the bare return value.
  `await rlm("subtask")` still works via `__call__` and returns an `RLMResult`
  (`.answer`, `.usage`, `.turns`, `.session_dir`).
- `X.send(*a, **kw) -> Handle` — sync; enqueue input to a (possibly new)
  background worker and return a handle immediately. `rlm.send` also takes `name`
  (continuation key; `None` → an auto-generated uuid hex) and `max_tokens`.

**Handle:**

- `.poll() -> ToolState` — sync; a snapshot of dynamic state. A pure read — never
  consumes a result.
- `.wait() -> <result>` — async; await and consume the next result.
- `.session_dir` — for `rlm`, the agent's session dir;
  `session_dir/"messages.jsonl"` is the live transcript the model can read or
  tail. `None` for tools without a transcript.

**`ToolState`** (what `poll()` returns):

- `status` — `running` / `finished` (idle) / `error`.
- `results` — a FIFO of the tool's own return values, drained by the model
  (`results.popleft()`). `poll()` never consumes, so a status check can't
  silently eat a result. For `rlm` the items are `RLMResult`.
- `queued` — the live, directly-editable pending inbox (a list): the model can
  cancel, reorder, or edit not-yet-started items. Race-free because the kernel
  loop is single-threaded cooperative — a sync edit in a cell can't interleave
  with the drain coroutine, which only yields at `await`. The in-flight item has
  already left the list.
- `error` — the live exception object if the worker died, else `None`.

`status` and `results` are independent: `finished` with unread results buffered,
and `error` with a good result still in the FIFO, are both valid (an errored turn
halts the worker).

`send`/`poll` are sync; `run`/`wait` are async. The `send`/`poll` signatures and
docstrings are mirrored onto the wrapped callables, so `help(rlm.send)` and
`inspect.signature` surface the real API. The model is not given a per-name
lookup API — it keeps handles in its own variables (the IPython namespace
persists across tool calls) and re-`send`s a name to continue. A registry exists
internally only so `send(name)` can find an existing agent.

### 3.2 Workers, handles, and the registry

`rlm/_async_runtime.py` (imported by `_inject_startup`, not inlined into the
setup f-string) provides the background machinery:

- A **worker** runs a **processor** and exposes its state through a `Handle`.
  There are two lifecycles:
  - **Resident** (named rlm agents): owns an inbox and drains it sequentially as
    one long-lived asyncio task; parks when idle and stays alive until `close()`.
    Held by the registry so a re-sent name continues it.
  - **Ephemeral** (general tools): runs one call to completion, then the task
    ends. Owned solely by its `Handle` — never registered — so it and its result
    are garbage-collected once the model drops the handle. A tool that wants
    state across calls keeps its own cache; the runtime never persists it.
- A **processor** turns one queued item into a result — this is where
  "continuation is up to the tool" lives:
  - stateless (general tools): `await fn(*a, **kw)`, push the return value to
    `results`.
  - `rlm` (stateful): `await engine.advance(prompt)` on a live engine (§3.3).
- A **handle** wraps the worker; `poll()` builds a `ToolState`; results and
  exceptions are stored so nothing becomes an unretrieved-exception warning.
- A per-kernel **registry** holds the resident (named rlm) workers with strong
  refs (asyncio won't keep a parked task alive otherwise). Per-kernel makes
  nesting hierarchical: each agent owns the registry of the children it spawned,
  with no global registry. General tools never touch it.

### 3.3 Resumable engine

`RLMEngine` is split so a conversation can be advanced repeatedly:

- `setup()` — depth check, ensure the session, write `meta.json`, start the REPL
  (when the ipython tool is active), and seed `_messages` with the system prompt
  (or rehydrate from disk; §3.8).
- `advance(prompt) -> RLMResult` — append a user turn, run the loop to the next
  stop, return the result. Repeatable on the same live kernel.
- `aclose()` — finalize the session and shut the kernel down.
- `run()` = `setup()` + one `advance()` + `aclose()` — the blocking one-off path.

`rlm`'s processor calls `advance` on a live engine, so re-sending the same name
is a multi-turn conversation with one persistent specialist. REPL teardown is
deferred for resident agents until the end-of-rollout cascade (§3.7).

### 3.4 Queueing

- Re-`send`ing a busy **named rlm agent** appends to its inbox; the worker drains
  sequentially after the current turn — no preemption.
- `poll().status` stays `running` while the inbox drains; `poll().queued` exposes
  the pending items and is directly editable; each item's result lands in
  `results` in order.
- General tools aren't registered, so each `send` is its own ephemeral worker —
  two sends to one skill run concurrently, not queued.

### 3.5 Session layout

Sessions nest along the call tree, one directory per agent, mirroring the
per-kernel registries:

```
<root-session>/sub-<name>/messages.jsonl
<root-session>/sub-<name>/sub-<child-name>/messages.jsonl   # sub-of-sub
```

- Two parallel structures: the live engines + kernels live in the in-memory
  per-kernel registry; the nested `sub-*` dirs are the durable, inspectable
  mirror. The transcript is reachable as `handle.session_dir/"messages.jsonl"`
  and via each `RLMResult.session_dir`.
- Names are unique within a parent, not globally — a `specialist` under one
  parent (`sub-specialist`) and another under a different parent coexist.
  Model-supplied names are sanitized filesystem-safe; an omitted name becomes a
  uuid hex.
- Directory discovery globs `sub-*` (`_detect_new_children`,
  `aggregate_child_metrics`).

### 3.6 Token budget

- The existing budget applies (`RLM_MAX_TOKENS` → `stop_reason="token_budget"`,
  checked per turn in `advance`).
- `rlm.send`'s `max_tokens` is clamped to a ceiling env var:
  `effective = min(requested or ceiling, ceiling)`, with `RLM_SUB_MAX_TOKENS` as
  both the ceiling and the default. Wall-clock budgets are out of scope.

### 3.7 Lifecycle, teardown, and caps

**Teardown cascade.** A parent's `aclose()` drains its live sub-agents,
finalizes its session, and then shuts its kernel — in that order, so child
metrics (`aggregate_child_metrics`) are counted before the kernel dies. The drain
runs *inside* the kernel (where the child registries live) via a cell that calls
`close_all_registries()`, so each child finalizes its own session and recursively
drains its grandchildren. A hard `shutdown_kernel(now=True)` would orphan
descendants' kernels, so teardown recurses gracefully, each level closing its
registry before its kernel.

**Two caps, via marker files.** Two independent pools live in subdirectories of a
per-rollout directory shared by the whole process tree: the root derives
`RLM_LIVE_AGENTS_DIR` from its session and it propagates to every kernel via
`_inject_startup`. Each held slot is one `<pid>-<uuid>.marker` file; an `flock`
serializes the sweep-count-create, and a PID-liveness sweep on each acquire
reclaims slots leaked by hard-killed processes — which a shared counter could
not. A pool is active only when its limit env var is a positive int and the
markers dir + POSIX `fcntl` are available; otherwise every acquire is granted.

- **Total** (`RLM_MAX_LIVE_AGENTS`) — resident agents. A slot is held from
  creation to teardown, bounding how many sub-agents (and kernels) exist at once.
  `send` raises `AgentLimitReached` at the cap — a creation-time failure, distinct
  from the `poll().error` runtime channel. The model's recourse is to reuse an
  existing agent (re-`send` its name; continuation takes no new slot).
- **Running** (`RLM_MAX_RUNNING_AGENTS`) — agents executing a turn. A slot is held
  only around one `advance()`; an idle agent holds none. Acquired blocking (the
  turn waits its turn). This is the parallelism knob.
- An **errored** turn frees both slots — the running slot in its `finally`, and
  the total slot too, reaping the kernel while keeping `poll().error` intact.
- The one-off path (`await rlm(...)`, `gather(...)`) holds both slots for its
  lifetime and blocks on each. A safety timeout `RLM_AGENT_WAIT_TIMEOUT` (default
  300s; `0` = wait forever) proceeds over the cap rather than deadlock when
  slot-holders are themselves waiting for slots (e.g. a parent turn awaiting a
  nested child). The root rollout (depth 0) is not capped.

### 3.8 Restart resilience

A cell that times out and won't yield to an interrupt triggers
`IPythonREPL.restart_kernel`, which restarts the kernel process and re-runs
`_inject_startup`, wiping the kernel's Python state. Two scenarios, handled
separately.

**Kernel-reset note on a timeout restart.** When a cell's kernel must be
restarted to recover it, the ipython tool result ends with a note that the REPL
was reset — every variable, import, and in-memory object is gone, rebuild what's
needed. Any agent (including the root) gets this when one of its own cells forces
a restart; the conversation itself is unaffected (it lives in the engine, not the
kernel).

**Resume from disk after a parent-kernel restart.** A sub-agent's `RLMEngine`
lives in its *parent's* kernel process, alongside `rlm.api._REGISTRY`. If the
parent kernel restarts, the registry and every in-memory engine are gone, but the
`sub-<name>/` dirs persist. Re-sending a name must continue that agent, not
silently start fresh on top of its old transcript. So when `send(name=X)` builds
a worker and no in-memory engine exists, `setup()` rehydrates `_messages` from the
latest on-disk view and restores the resume header from `meta.json` instead of
seeding `[system, user]`, then injects a user-turn warning that the conversation
survived but the REPL is brand new. This is why appending to an existing
`sub-<name>/messages.jsonl` is correct — one agent, one growing transcript — and
why no directory disambiguation is needed.

**Transcript = an append-only list of views.** `messages.jsonl` is the replayable
structure the engine rehydrates from:

- A **view** is the message sequence the model saw within one compaction branch.
  Each turn is appended to the current view; on compaction, the checkpoint user
  prompt and the assistant summary are appended to the closing view, then a new
  view is opened seeded with `[system, user(framing + summary)]`.
- Lines are typed and append-only: `msg` (full OpenAI shape incl. `tool_call_id`s,
  tagged with view index + turn), `branch_reset` (compaction stats: dropped /
  summary chars, turns-since), and event lines (`spawn`, `done`).
- Resume loads the latest view; all prior views are kept, each a contiguous
  context the model acted in.
- `messages.jsonl` is rlm-local — nothing outside rlm reads it — so the
  system-prompt path, model tailing, and `handle.session_dir` are unchanged.

**Resume header in `meta.json`, written per turn.** Each turn writes usage,
`turn_offset`, `view`, `branch_start_turn`, and a full metrics snapshot to
`meta.json`. Writing per turn (not only at `finalize`) means a hard restart —
which never reaches `finalize` — doesn't undercount the metrics the harness
consumes (`rlm_total_tool_response_tokens`, …).

**No train/inference mismatch.** The trainer builds samples from the inference
engine's recorded token steps (`RolloutOutput.trajectory`), not from
`messages.jsonl`, and those tokens are exactly what was sent. A resume — or the
injected reset warning, which is just an append — at worst starts a new
training-sample split, the same thing compaction already does and which
`interleave_rollout` handles. Storing the exact view (rather than reconstructing
one) is what keeps a resumed agent's continuation correct.

---

## 4. Configuration

Environment variables this layer reads:

- `RLM_SUB_MAX_TOKENS` — completion-token ceiling (and default) for `rlm.send`
  sub-agents.
- `RLM_MAX_LIVE_AGENTS` — cap on resident sub-agents (total pool).
- `RLM_MAX_RUNNING_AGENTS` — cap on concurrently executing sub-agents (running
  pool).
- `RLM_AGENT_WAIT_TIMEOUT` — seconds a blocking slot acquire waits before
  proceeding over the cap (default 300; `0` = forever).
- `RLM_LIVE_AGENTS_DIR` — marker-file directory for the caps; derived by the root
  from its session and propagated to every kernel (not set by hand).

The training harnesses surface the first four as configuration (v1
`RLMProgramConfig`; composable `rlm_harness`). The interception and `meta.json`
metrics contracts are unchanged — background sub-agents still tag
`X-RLM-Depth >= 1` and stay black-box.

---

## 5. Future work: interruptions

Not built. A preemption system in which an async hook interrupts the model's
*thinking* (but not its tool calls) to inject context — alerts, sensors, or a
sub-agent notifying the parent that it finished. The relevant shape:

- The engine loop is `completion (thinking) -> append -> tool.execute (action)`.
  "Interrupt thinking, not tool calls" maps to: make the completion await
  interruptible and keep `tool.execute` atomic — run the completion alongside a
  watcher (`asyncio.wait(FIRST_COMPLETED)`); if an interrupt wins, inject context
  and re-issue; if it fires during tool exec, apply it after the tool result is
  appended.
- Cross-process channel: an append-only file in the session dir that an engine
  watcher tails (precedent: `programmatic_tool_calls.jsonl` is already a
  kernel→engine file signal).
- Keeping the partial response requires streaming plus stopping generation
  vLLM-side, which is the main risk.
- Injecting mid-thinking is a branch event like compaction (reuse
  `render_completion_with_branches`; mind chat-template role-ordering), with
  debounce / max-interrupts-per-window guardrails.

Background tool calling makes this more effective: the model writes short poll
cells instead of long blocking ones, so interrupts land promptly at thinking
boundaries. The flagship "sub-agent notifies parent when done" case is a
background task plus a done-callback writing the channel.

---

## 6. Non-goals

- Wall-clock / time budgets (token budgets only).
- Determinism / virtual-time concurrency / exact rollout replay.
