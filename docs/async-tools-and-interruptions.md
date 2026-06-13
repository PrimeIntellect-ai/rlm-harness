# Async, persistent sub-agents & programmatic tools (+ interruptions)

Status: design. Phase 1 in progress on `sebastian/persistent-tools-2026-06-11`.
M0 (a backgrounded task progresses between cells) is **verified**. The IPython
`display_data` capture gap is fixed separately (branch
`fix/ipython-capture-display-data`).

This document covers two related-but-separate systems:

1. **Async tool calling** — background, pollable, and (for `rlm`) persistent
   multi-turn programmatic callables. **Build first** (PR 1).
2. **Interruptions** — an async hook that preempts the model's *thinking* (not
   its tool calls) to inject context. **Deferred.** Overview only, captured here
   so Phase 1 doesn't build in a direction that makes Phase 2 harder.

Token budgets are in scope for Phase 1 (token only; wall-clock/time budgets
deferred).

---

## 1. Motivation

Today a sub-agent call (`await rlm("subtask")` from inside an ipython cell)
blocks the cell until it finishes, and the engine is one-shot: `RLMEngine.run()`
builds `[system, user]`, runs to completion, tears down its kernel. Two
limitations follow:

- **No cross-call lifecycle.** The model can't launch work in one tool call and
  check on it in another. Long sub-agent work also risks being killed by the
  per-cell exec timeout (`RLM_EXEC_TIMEOUT`, capped at 600s in `IPythonREPL`).
- **No persistence / multi-turn.** The model can't stand up a *specialist*
  sub-agent and ask it repeatedly across tool calls.

The headline goal is **persistent, named sub-agents the parent can hold
multi-turn conversations with** — create a specialist and keep asking it (just
stop sending to abandon one) — plus quick one-offs. The same background/poll mechanism generalizes to
all programmatic callables (skills); for those, persistence is not required and
the value is purely offloading.

### Training context (constraints that shape the design)

`rlm` is a training-only harness. The design must preserve:

- **Black-box sub-agents.** Sub-agent API calls are tagged `X-RLM-Depth >= 1`
  (`client.py:make_client`) and dropped from the trajectory the trainer sees
  (v1 `RLMEndpoint.trajectory_visibility`, composable `_keep_only_parent_rlm_steps`).
  Backgrounding must not leak sub-agent turns into the parent trajectory.
- **Trajectory cleanliness.** The parent trajectory is the sequence of
  `(assistant tool-call, tool result)` pairs. Background spawns/polls are just
  fast tool results; they create **no** parent branch (unlike compaction).
- **True concurrency is accepted.** Poll results depend on wall-clock progress,
  so rollouts are timing-dependent; we accept the nondeterminism (large batch
  sizes denoise it). Exact rollout replay for debugging is lost.

---

## 2. Substrate: how `rlm` works today

- **Engine loop** (`engine.py:_run_loop`): one tool call per turn
  (`parallel_tool_calls=False`); builtin tools `ipython`/`bash`/`edit`.
- **IPython kernel** (`tools/ipython.py`): a long-lived subprocess kernel.
  - The **namespace persists across cells** — a registry/handle created in one
    cell is still there in the next.
  - The **event loop runs between cells** — an `asyncio.Task` created with
    `ensure_future(...)` (not awaited) keeps progressing while the model is in
    other cells. `nest_asyncio.apply()` is already called in `_inject_startup`.
  - `_inject_startup` injects skills and a callable `rlm` into the namespace via
    `_wrap_callable` (makes a module `await`-callable → `mod.run(...)`), and sets
    `RLM_SESSION_DIR` / `RLM_DEPTH` for child sessions.
- **Sub-agent blocking I/O is already offloaded.** The engine runs
  `tool.execute` via `asyncio.to_thread`, so several sub-agents can run
  concurrently on one loop (async model calls + threaded REPL I/O).
- **Sessions** (`session.py`): nested `sub-<id>/` dirs, `meta.json` +
  `messages.jsonl`; metrics aggregate up the tree
  (`aggregate_child_metrics`, globs `sub-*`).
- **What a cell surfaces to the model** = the kernel's captured output:
  `stream` (stdout/stderr), `execute_result`, `display_data`, and `error`
  (traceback). `display_data` capture was missing and is fixed on
  `fix/ipython-capture-display-data`.
- **Already possible:** within-cell parallelism via
  `await asyncio.gather(rlm(a), rlm(b))`. The new capability is **cross-cell
  lifecycle**, not parallelism per se.

### Load-bearing assumption — VERIFIED (M0)

A backgrounded asyncio task created in one cell keeps progressing between cells:
a probe counter advanced `0 → 71 → 119` across separate `execute()` calls (and
`done=False` = still live). The whole design rests on this; it holds under the
pinned ipykernel / nest_asyncio.

---

## 3. Phase 1 — async, persistent, pollable callables

### 3.1 API surface

**One background primitive, uniform across every callable** (`rlm` and skills):
you `send` input to a tool and `poll` it at will. Tool-specific kwargs differ;
the handle/poll protocol is identical, so `rlm` gets *no* special surface — it
just produces many results instead of one.

- `X.run(*a, **kw)` — async; await to completion; returns the **bare return
  value** (unchanged — `await rlm(...)` still works via `__call__`). The simple
  path has no wrapper.
- `X.send(*a, **kw) -> Handle` — **sync**; enqueue input to a (possibly new)
  background worker and return a handle immediately. The single background
  method for all tools. rlm-specific kwargs: `name` (persistence / continuation;
  `None` → auto-generated as a uuid hex).

**Handle:**

- `.poll() -> ToolState` — **sync**; a snapshot of dynamic state (below). Pure
  read — never mutates or consumes.
- `.wait() -> <result>` — async; await the next result.
- `.session_dir` — for `rlm`, the agent's session dir; `session_dir/"messages.jsonl"`
  is the live transcript the model can read/tail (this is "message history = the
  path"). `None` for tools without a transcript.

**Agent lifecycle.** A healthy idle agent stays resident — its kernel alive,
holding a *total* slot — until the end-of-rollout cascade tears it down (parent
`aclose` → drain registries); to abandon one the model just stops sending to it.
An *errored* agent is reaped eagerly (kernel shut, total slot freed) but stays
pollable for its error. The two caps are detailed in §3.7.

**No per-name lookup API.** The model keeps handles in its own variables (the
IPython namespace persists across tool calls) and re-`send`s a name to continue.
A registry exists internally only so `send(name)` can find an existing agent; it
is deliberately not exposed — `get`/`list` would be rlm-specific surface that
breaks the uniform `send` → `handle` → `poll` shape every tool shares.

**The uniform poll object** (`ToolState` — working name; the "messages object"):

- `status` — worker lifecycle: `running` / `finished` (idle) / `error`. Your
  `("running", <queue>)` lives here as `status == "running"` + `queued`.
- `results` — a **FIFO of the tool's own return values**, drained by the model
  (`results.popleft()` / iterate). **`poll()` never consumes** — a status-check
  poll must not silently eat a result. One item for a one-shot skill; one per
  completed turn for multi-turn rlm. For `rlm` the items are `RLMResult`
  (`.answer`, `.session_dir`).
- `queued` — the **live, directly-editable** pending inbox (a list): cancel,
  reorder, or edit not-yet-started items. Race-free because the kernel loop is
  single-threaded cooperative (a sync edit in a cell can't interleave with the
  drain coroutine, which only yields at `await`). The in-flight turn has already
  left the list.
- `error` — the **live exception object** if the worker died (`None` otherwise);
  the model can print it, re-raise, or format a traceback.

`status` and `results` are independent: `finished` with unread results buffered,
or `error` with one good result already in the FIFO, are both valid (an errored
turn halts the worker, as the engine does today).

**Sync vs async:** `send`/`poll` are sync (`h = rlm.send(...)` returns
immediately); `run`/`wait` are async.

**Discoverability:** signature + docstring are mirrored onto the wrapped callable
(as `_wrap_callable` already does for `run` via `__signature__`/`__doc__`), so
`help(rlm)` / `inspect.signature` surface `send`/`poll`/etc. A strong docstring +
system-prompt line carry the "`send` to a stateless skill = one background call;
`send` to a named rlm = continue the conversation" distinction.

### 3.2 Worker + registry + handle (kernel-side)

Ship a helper module `rlm/_async_runtime.py`, imported by `_inject_startup` —
**not** inlined into the `setup_code` f-string (testable, maintainable).

- A **worker** runs a **processor** and exposes its state via a `Handle`. Two
  lifecycles:
  - **Resident** (named rlm agents): owns an inbox (deque) and drains it
    **sequentially** as one long-lived asyncio task; stays alive until `close()`
    and is held by the registry so a re-sent name continues it.
  - **Ephemeral** (general tools): runs one call to completion, then the task
    ends. Owned solely by its `Handle` — never registered — so it and its result
    are GC'd once the model drops the handle. A tool that wants state across
    calls keeps its own cache; the runtime never persists it.
- A **processor** parameterizes "process one item" — this is how "continuation
  is up to the tool":
  - stateless (general tools): `await tool.run(item)`, push the return value to
    `results`.
  - `rlm` (stateful): `await engine.advance(prompt)` on a live engine (§3.3).
- A **Handle** wraps the worker; `poll()` builds a `ToolState` from worker state;
  results/exceptions are stored so nothing becomes an unretrieved-exception
  warning.
- A module-level **registry** (per kernel) holds the **resident** (named rlm)
  workers with **strong refs** (asyncio won't keep a parked task alive
  otherwise). Per-kernel ⇒ hierarchical: each agent owns the registry of the
  children *it* spawned; nesting is naturally recursive, no global registry.
  General tools never touch the registry.

This layer delivers persistent rlm agents (resident workers + registry) and
offloading for skills (ephemeral workers, no accumulation).

### 3.3 Resumable engine (rlm's stateful processor)

- Split `RLMEngine.run()` into:
  - `setup()` — depth check, ensure session, write meta, start REPL, build
    system prompt + initial `messages`.
  - `advance(prompt) -> RLMResult` — append `{"role":"user","content":prompt}`,
    run the loop to the next stop, return the result. Callable repeatedly.
  - `run()` stays = `setup()` + one `advance()` + teardown (unchanged blocking
    path).
- **Defer REPL teardown** for persistent agents until the end-of-rollout cascade.
- rlm's processor = `advance` on the live engine; a reused `name` = the same
  engine = a multi-turn conversation with a persistent specialist.
- **Resume:** when an on-disk session exists but the in-memory engine is gone
  (post kernel-restart), `setup()` rehydrates `_messages` from the latest view in
  `sub-<name>/messages.jsonl` and the header from `meta.json` instead of seeding
  fresh — see §3.10.
- **Forward-compat (Phase 2):** keep the model-completion call centralized in
  the loop (a single `await self._completion(...)` site) so interruptions can
  wrap *that* call later without reworking the loop.

### 3.4 Queueing (Phase 1)

- `send` to a busy name **enqueues**; the worker drains sequentially after the
  current turn completes — **no preemption** (preempting the in-flight turn is
  Phase 2).
- `poll().status` stays `running` while draining; `poll().queued` exposes depth
  and is directly editable.
- Each queued item's result lands in `results` in order.
- Queueing applies to a **named rlm agent** (re-`send` the name → its inbox
  grows, drained serially). General tools aren't registered, so each `send` is
  its own ephemeral worker; two sends to one skill run **concurrently**, not
  queued.

### 3.5 Session layout

Agent name in the session path; dirs **nest along the call tree** (sub-of-sub
one level deeper), mirroring the per-kernel registries:

```
<root-session>/<agent-name>/messages.jsonl
<root-session>/<agent-name>/<sub-agent-name>/messages.jsonl   # sub-of-sub
```

- **Two parallel structures.** Live IPython state (engines + kernels) lives in
  the in-memory per-kernel registry; the **nested dirs** are the durable,
  inspectable mirror. The transcript is reachable as
  `handle.session_dir/"messages.jsonl"` (live-tailable) and via each result's
  `RLMResult.session_dir`. `messages.jsonl` is the append-only list of views the
  engine rehydrates from after a restart (§3.10).
- Names are unique **within a parent** (per-kernel), not globally —
  `root/specialist` and `root/other/specialist` coexist. Model-supplied names are
  sanitized filesystem-safe; an omitted name auto-generates a uuid hex.
- Update the two `sub-*` globs that assume the random-id prefix:
  `engine.py:_detect_new_children` and `session.py:aggregate_child_metrics`.

### 3.6 Token budget (token only for Phase 1)

- Reuse the existing budget (`RLM_MAX_TOKENS` → `stop_reason="token_budget"`,
  per-turn check in `_run_loop`).
- Per-`send` `max_tokens` kwarg, **clamped to a user-set env ceiling**:
  `effective = min(model_requested or ceiling, ceiling)` (same clamp idiom as the
  timeout caps). Env: `RLM_SUB_MAX_TOKENS`.
- Wall-clock/time budgets **deferred**.

### 3.7 Lifecycle, teardown, caps (in this PR)

- **Graceful teardown cascade.** Parent teardown (`aclose`) must drain/cancel
  → finalize the session → *then* shut the sub-kernel. A hard
  `shutdown_kernel(now=True)` on a parent **orphans descendants' kernels**, so
  teardown recurses gracefully (each level closes its registry before its kernel
  dies). This also fixes an ordering bug: today `session.finalize()` (→
  `aggregate_child_metrics`) runs at the end of `_run_loop`, *before*
  `repl.shutdown()` in `run()`'s `finally`, so live children under-count.
  New order: **drain live agents → parent finalize → kill kernel.**
- **Two caps via marker files** (crash-safe), independent pools in subdirs of a
  per-rollout dir shared by the whole process tree: the root derives
  `RLM_LIVE_AGENTS_DIR` from its session and it propagates to every kernel via
  `_inject_startup`. One marker file per held slot (`<pid>-<uuid>.marker`); an
  `flock` serializes sweep-count-create and a PID-liveness sweep on each acquire
  reclaims slots leaked by hard-killed processes (an integer counter could not).
  - **Total** (`RLM_MAX_LIVE_AGENTS`) — *resident* agents. A slot is held from
    creation to teardown; bounds how many sub-agents (and their kernels) exist at
    once. `send` **raises** `AgentLimitReached` at cap — a creation-time failure,
    uniform across tools and distinct from the `poll().error` runtime channel
    (supersedes the earlier "terminal handle" sketch). The model's recourse is to
    reuse an existing agent (re-`send` its name — continuation takes no new slot).
  - **Running** (`RLM_MAX_RUNNING_AGENTS`) — *executing* agents. A slot is held
    only around `advance()` (one turn); an idle agent that *could* take another
    query holds none. Acquired **blocking** (the turn waits its turn). This is the
    parallelism knob.
  - An **errored** turn frees both: the running slot via its `finally`, and —
    because a halted agent is dead weight — the total slot too, reaping the kernel
    while keeping `poll().error` intact.
  - The one-off path (`await rlm(...)`, `gather(...)`) holds both for its lifetime
    and **blocks** on each. A safety timeout `RLM_AGENT_WAIT_TIMEOUT` (default
    300s, `0` = forever) proceeds over-cap rather than deadlock when slot-holders
    are themselves waiting for slots (e.g. a parent turn awaiting a nested child).
    The root rollout (depth 0) is not capped.

### 3.8 Open questions (Phase 1)

- `ToolState` class name (you've been calling it the "messages object").

### 3.9 Verifiers harness ripples (PR 2)

- New env vars (`RLM_SUB_MAX_TOKENS`, `RLM_MAX_LIVE_AGENTS`,
  `RLM_MAX_RUNNING_AGENTS`, `RLM_AGENT_WAIT_TIMEOUT`) wire into both
  integrations: v1 `RLMProgramConfig` (`packages/harnesses/harnesses/rlm.py`)
  and legacy `rlm_harness(...)`
  (`verifiers/envs/experimental/composable/harnesses/rlm.py`).
- **No change** to the interception or `meta.json` metrics contracts. Background
  sub-agents still tag `X-RLM-Depth >= 1` and stay black-box.

### 3.10 Restart resilience: resume-from-disk + kernel-reset warning (this PR)

A timed-out cell an interrupt can't clear triggers `IPythonREPL.restart_kernel`,
which restarts the kernel *process* and re-runs `_inject_startup` — wiping the
kernel's Python state, including `rlm.api._REGISTRY` and every named sub-agent's
in-memory engine (the `RLMEngine` objects live in the *parent* kernel's process,
not in the sub-agents' own kernels). Two consequences, two fixes.

**Resume-from-disk (parent-kernel restart).** Afterward `_REGISTRY` is empty but
the `sub-<name>/` dirs persist. Re-sending a name must *continue* that agent, not
silently start fresh and append onto its old transcript. So when `send(name=X)`
builds a worker, no in-memory engine exists, but `sub-X/messages.jsonl` does,
`setup()` rehydrates from disk instead of seeding `[system, user]`. This makes
re-sending a name a true continuation across restarts — which is exactly why the
`exist_ok=True` append is *correct* (one agent, one growing transcript) and why no
dir disambiguation is needed.

**Kernel-reset warning (any restart).** A restart wipes the agent's REPL
variables/imports even when its conversation survives (own-kernel restart:
conversation intact in the engine; parent-kernel restart: conversation rehydrated,
fresh REPL). `IPythonREPL` records that it restarted; the engine injects a
`WARNING: kernel reset — variables and imports are gone; rebuild any state you
need` turn into `_messages` before the next completion. Universal — the root agent
gets it too.

**Transcript = append-only list of views (supersedes the old event log).**
`messages.jsonl` becomes the replayable structure rather than a lossy log:
- A **view** is the message sequence the model saw within one compaction branch.
  Append each turn to the current view; on compaction, append the checkpoint user
  prompt + the assistant summary to the closing view, then open a new view seeded
  with `[system, user(framing+summary)]` and keep appending.
- Append-only typed lines: message lines (tagged with view index + turn, full
  OpenAI shape incl. `tool_call_id`s), `branch_reset` lines (compaction stats:
  dropped/summary chars, turns-since), event lines (`sub_spawn`, `done`).
- **Resume loads the latest view** (`views[-1]`); keep all prior views — each is a
  contiguous context the model acted in (the training-native shape).
- rlm-local: nothing outside rlm reads `messages.jsonl` (verified — not prime-rl,
  not verifiers). Same filename, so the system-prompt path, model tailing, and
  `handle.session_dir` are unchanged.

**Resume header in `meta.json`, written per turn.** Usage / metrics /
`turn_offset` restore on resume. `meta.json` already holds usage+metrics — write
it per turn (not only at `finalize`) so a hard restart, which never calls
`finalize`, doesn't undercount. Matters because those metrics feed the harness
metric channel (`rlm_total_tool_response_tokens`, …) prime-rl consumes.

**No train/inference mismatch.** The trainer builds samples from the inference
engine's recorded token steps (`RolloutOutput.trajectory`), not `messages.jsonl`,
and those tokens are exactly what was sent. A restart/resume — or the injected
warning, which is an append — at worst starts a new training-sample split, the
same thing compaction already does and which `interleave_rollout` handles
gracefully. Storing the *exact* view (vs. reconstructing) is what keeps the
resumed agent's continuation correct.

**Sequence:** kernel-reset warning first (small, self-contained, helps every agent
today), then resume-from-disk.

---

## 4. Phase 2 — interruptions (DEFERRED — overview only)

**Do not build yet.** Captured so Phase 1 stays compatible.

A separate preemption system: an async hook fires and **interrupts the model's
thinking, but not its tool calls**, to inject context (alerts, sensors, or a
sub-agent notifying the parent that it finished).

### Mechanism

The engine loop is `completion (thinking) -> append -> tool.execute (action)`.
"Interrupt thinking, not tool calls" maps exactly to: **make the completion
await interruptible; keep `tool.execute` atomic.** Run the completion as a task
alongside a watcher (`asyncio.wait(FIRST_COMPLETED)`); if an interrupt wins,
stop the completion and inject context before re-issuing; if it fires during
tool exec, queue it and apply after the tool result is appended.

### Decisions captured

- **Cross-process channel: file-tail.** Sensors fire in the kernel (or as
  backgrounded sub-agents), but the completion lives in the engine (main
  process). Use an append-only file in the session dir that an engine watcher
  tails — precedent: `programmatic_tool_calls.jsonl` is already a
  kernel/bash→engine file signal. (Accepts poll-interval latency; a socket would
  be lower-latency but more moving parts.)
- **Keep the partial response, but actually stop generation.** Requires
  streaming + abort, and must **stop vLLM-side generation** (not just drop the
  client request). This is the main complexity/risk and the reason Phase 2 is
  deferred.
- **It is a branch event, like compaction.** Injecting mid-thinking rewrites the
  branch; reuse the harness `render_completion_with_branches` machinery. Mind
  **role-ordering** of the injected message (chat-template alternation matters in
  training).
- **Guardrails:** debounce / max-interrupts-per-window (mirror compaction's
  "fires at most once per loop iteration").
- **Sources:** engine-side host/env sensors (direct enqueue, low latency);
  kernel-side model sensors + sub-agent-done notifications (via the file
  channel).

### Synergy with Phase 1

With async tool calling the model writes **short** poll cells instead of long
blocking ones, so interrupts land promptly at thinking boundaries. The flagship
"sub-agent notifies parent when done" case = Phase-1 background task + Phase-2
done-callback writing the channel.

---

## 5. Work split / milestones

- **M0 — Probe. DONE.** Backgrounded task progresses between cells (verified).
- **`display_data` capture — DONE** (branch `fix/ipython-capture-display-data`,
  its own PR).
- **PR 1 — persistent tools** (this branch). Collapses old M1+M2+M3, since
  they're too entangled to split:
  - `rlm/_async_runtime.py`: worker + processor + `Handle` + per-kernel registry
    + `ToolState`.
  - Queueing (per-name inbox, sequential drain, editable `queued`).
  - Resumable engine (`setup`/`advance`), rlm stateful processor.
  - `rlm.send` (named continuation; no public per-name lookup); wired into
    `_inject_startup`.
  - Graceful teardown cascade + finalize/shutdown re-ordering.
  - Marker-file caps: total resident (`RLM_MAX_LIVE_AGENTS`) + running
    parallelism (`RLM_MAX_RUNNING_AGENTS`), with eager reap on error.
  - Session-path transcript access (`handle.session_dir`).
  - Restart resilience (§3.10): `messages.jsonl` as an append-only list of views,
    `setup()` resume-from-disk, per-turn `meta.json` header, kernel-reset warning.
- **PR 2 — harness wiring.** New env vars into v1 `RLMProgramConfig` and
  composable `rlm_harness`.
- **Later — Phase 2 interruptions.**

## 6. Non-goals (Phase 1)

- Interruptions / mid-stream generation control / stopping vLLM generation.
- Wall-clock/time budgets.
- Determinism / virtual-time concurrency.
