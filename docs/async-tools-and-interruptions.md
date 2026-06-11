# Async, persistent sub-agents & programmatic tools (+ interruptions)

Status: design / not yet implemented.

This document covers two related-but-separate systems:

1. **Async tool calling** — background, pollable, and (for `rlm`) persistent
   multi-turn programmatic callables. **Build first.**
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
  sub-agent, ask it repeatedly, and dismiss it when it goes down a wrong path.

The headline goal is **persistent, named sub-agents the parent can hold
multi-turn conversations with** — create a specialist, keep asking, dismiss when
cooked — plus quick one-offs. The same background/poll mechanism generalizes to
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
- **Already possible:** within-cell parallelism via
  `await asyncio.gather(rlm(a), rlm(b))`. The new capability is **cross-cell
  lifecycle**, not parallelism per se.

### Load-bearing assumption

Background tasks progress between cells under the **pinned** ipykernel /
nest_asyncio versions. This must be verified empirically before anything else
(see M0).

---

## 3. Phase 1 — async, persistent, pollable callables

### 3.1 API surface

**One unified background primitive for every callable** (`rlm` and skills): you
`send` data/input to a tool and `poll` it at will. Tool-specific kwargs differ;
the handle/poll protocol is identical.

- `X.run(*a, **kw)` — async, await to completion. **Unchanged** (`await rlm(...)`
  still works via `__call__`).
- `X.send(*a, **kw) -> Handle` — **sync**; schedules a background task on the
  kernel loop, returns a handle immediately. `send` is the single, general
  background method for all tools (it subsumes the earlier `spawn`/`send` split —
  "you send data to a tool, then poll it at will").
  - For `rlm`: `rlm.send(prompt, name=None)` is named, persistent, multi-turn
    (see below). The `name`/persistence kwargs are rlm-specific.
  - For a skill: `skill.send(**skill_kwargs)` runs it in the background; no name
    or persistence required.

**Handle:**

- `.poll() -> (status, payload)` — **sync**; non-blocking status check.
- `.wait() -> <result>` — async; await the final result of this handle.
- `.dismiss()` — request teardown (cancel task, shut down sub-kernel, finalize
  session). Sync request; teardown runs in the background.

**`rlm.send` persistence (named, multi-turn):**

- `name=None` → auto-generated name (e.g. `beautiful-sky-bison`).
- First call with a name: create + setup + run one turn.
- Subsequent calls with the same name: append a user turn to the *same* engine
  (same conversation, same live kernel) and run again — the parent holds a
  multi-turn conversation with a persistent specialist.
- `rlm.get(name) -> Handle | None`, `rlm.list() -> list[str]` — registry access,
  so the model can poll by name in a later cell without keeping the handle var.

**Status contract** (`poll`): returns **real Python objects**, not stringified
ones — poll results are *not* auto-injected into the model's context. The model
writes `x = handle.poll()` and decides what to do (briefly print it, re-raise an
exception, format a traceback, drive control flow). Only what the model prints
(stdout/stderr) reaches its context, so handles can return whatever is most
expressive:

- `("running", None)`
- `("finished", <result>)` — the underlying callable's actual return value. For
  `rlm` that's the `RLMResult` (`.answer` for the text); after a follow-up
  `send(name)` the status flips back to `running` until that turn ends.
- `("error", <Exception>)` — the **live exception object** (the model can print
  it, re-raise it, or pull a full traceback). More general and more expressive
  than a pre-stringified message, and it costs nothing since poll output isn't
  auto-shown. The same generality applies to all tools — poll returns real
  values; the model chooses what to surface.

**Sync vs async shape is load-bearing:** `send`/`poll`/`get`/`list`/`dismiss`
are sync so cells read naturally (`h = rlm.send(...)` returns immediately);
`run`/`wait` are async (await to completion).

### 3.2 Registry + handle (kernel-side, no engine change)

Ship a small helper module in the `rlm` package (e.g. `rlm/_async_runtime.py`)
imported by `_inject_startup` — **not** inlined into the `setup_code` f-string
(testable, maintainable).

- A module-level **registry**: `name -> Handle`. Holds **strong refs** to tasks
  (asyncio won't otherwise keep them alive) and survives across cells via the
  kernel namespace.
- A **`Handle`** wraps an `asyncio.Task`; an `add_done_callback` stores the
  result/exception so `poll` is non-blocking and unretrieved-exception warnings
  ("Task exception was never retrieved") don't spam logs.
- `send` schedules the coroutine with `asyncio.ensure_future` on the running loop
  and registers the handle.

This layer alone delivers offloading for skills and one-off background `rlm`.

### 3.3 Resumable engine (engine-side) — required for persistence

Refactor `RLMEngine` so a named agent can be kept alive and continued:

- Split `run()` into:
  - `setup()` — depth check, ensure session, write meta, start REPL, build
    system prompt + initial `messages`.
  - `advance(prompt) -> RLMResult` — append `{"role":"user","content":prompt}`,
    run the loop to the next stop, return the answer. Callable repeatedly.
- **Defer REPL teardown.** Today `run()` shuts the REPL down in `finally`; for
  persistent agents the kernel must stay alive until `dismiss()` (or parent
  teardown). Keep the one-shot `run()` = `setup()` + one `advance()` + teardown
  for the unchanged blocking path.
- `rlm.send(name)` routes to: look up engine for `name` in the registry; create
  + `setup()` if absent; schedule `advance(prompt)` in the background.

**Forward-compat note (for Phase 2):** keep the model-completion call
centralized in the loop (a single `await self._completion(...)` site) so
interruptions can later wrap *that* call without reworking the loop. Do not
scatter `chat.completions.create` calls.

### 3.4 Session layout

Use the agent name in the session path so persistence is durable and
debuggable. The dirs **nest along the call tree** — a sub-agent of a sub-agent
nests one level deeper — mirroring the live registry structure:

```
<root-session>/<agent-name>/messages.jsonl
<root-session>/<agent-name>/<sub-agent-name>/messages.jsonl   # sub-of-sub
```

- **Two parallel structures, both needed.** The live IPython state (engines +
  kernels) is held in an in-memory **registry dict per kernel** — each agent owns
  the registry of the children *it* spawned, so nesting is naturally recursive
  and needs no global registry. The dict is required for liveness regardless; the
  **nested dirs** are the durable, inspectable mirror that makes rollouts
  interpretable after the fact and lets the parent model find and act on a
  child's transcript if it ever needs to.
- Names are unique **within a parent** (per-kernel registry), not globally —
  `root/specialist` and `root/other/specialist` can coexist. Names must be
  filesystem-safe and collision-checked (auto-names too).
- Update the two `sub-*` globs that assume the random-id prefix:
  `engine.py:_detect_new_children` and `session.py:aggregate_child_metrics`
  (broaden the glob or keep a `sub-`/marker convention).
- `messages.jsonl` is the **durable record** (and a future rehydration path); the
  **live engine in the registry is the actual persistence** (the on-disk log
  cannot restore live kernel variables — only the conversation).

### 3.5 Token budget (token only for Phase 1)

- Reuse the existing budget (`RLM_MAX_TOKENS` → `stop_reason="token_budget"`,
  checked per turn in `_run_loop`).
- Add a per-`send` `max_tokens` kwarg, **clamped to a user-set env ceiling**:
  `effective = min(model_requested or ceiling, ceiling)` — same clamp idiom as
  the existing timeout caps. Proposed env: `RLM_SUB_MAX_TOKENS`.
- Wall-clock/time budgets are **deferred**.

### 3.6 Lifecycle, teardown, caps

- **Cap** concurrent/persistent agents (proposed env, e.g.
  `RLM_MAX_LIVE_AGENTS`) — each persistent `rlm` with ipython enabled is another
  kernel subprocess + thread + session.
- **Teardown is the sharpest operational risk.** When the parent engine shuts
  down its own REPL, named sub-agents living in the kernel namespace are **not**
  auto-killed (separate subprocesses + tasks). The registry must expose a
  "close all" run on kernel shutdown (atexit / shutdown hook): cancel tasks, shut
  sub-REPLs, finalize sessions.
- **Metrics at parent finalize:** `aggregate_child_metrics` already tolerates a
  child missing `context_token_stats` (treats as zero), so a still-running agent
  won't crash aggregation — but for accurate sub-rlm token counts, persistent
  agents should be drained/finalized before the parent finalizes.

### 3.7 Verifiers harness ripples

- New env vars (`RLM_SUB_MAX_TOKENS`, `RLM_MAX_LIVE_AGENTS`) wire into both
  integrations: v1 `RLMProgramConfig` (`packages/harnesses/harnesses/rlm.py`)
  and legacy `rlm_harness(...)`
  (`verifiers/envs/experimental/composable/harnesses/rlm.py`).
- **No change** to the interception or `meta.json` metrics contracts. Background
  sub-agents still tag `X-RLM-Depth >= 1` and stay black-box.

### 3.8 Open questions (Phase 1)

- **`send(name)` while that agent is still running** — error/busy, queue, or
  ignore? Queueing previews interruptions; recommend **error/busy for Phase 1**
  (simplest, no preemption semantics yet).
- Exact env var names (`RLM_SUB_MAX_TOKENS`, `RLM_MAX_LIVE_AGENTS`).
- Auto-name generator seeding: deterministic per rollout (counter / seed from
  session id) vs. random — recommend deterministic for reproducibility.

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
- **Keep the partial response, but actually stop generation.** This requires
  streaming + abort, and must also **stop vLLM-side generation** (not just drop
  the client request). This is the main complexity/risk and the reason Phase 2
  is deferred.
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

With async tool calling, the model writes **short** poll cells instead of long
blocking ones, so interrupts can land promptly at thinking boundaries. The
flagship "sub-agent notifies parent when done" case = Phase-1 background task +
Phase-2 done-callback writing the channel.

---

## 5. Work split / milestones

- **M0 — Probe.** Verify a backgrounded task progresses between cells under the
  pinned ipykernel/nest_asyncio. ~20 lines. Gate for everything else.
- **M1 — Generic background handle.** `send`/`poll`/`wait`/`dismiss` + registry
  in `rlm/_async_runtime.py`, wired into `_inject_startup`. No engine change, no
  persistence. Delivers offloading for skills and one-off background `rlm`.
- **M2 — Resumable engine + named persistence.** Split `RLMEngine.run()` into
  `setup()`/`advance()`; defer REPL teardown; `rlm.send`/`get`/`list`; session
  namespacing by agent name.
- **M3 — Budget + lifecycle.** Per-`send` token budget clamped to env ceiling;
  concurrent-agent cap; kernel-shutdown teardown of live agents.
- **M4 — Harness wiring.** New env vars into v1 `RLMProgramConfig` and composable
  `rlm_harness`.
- **(Later) Phase 2 — interruptions.**

## 6. Non-goals (Phase 1)

- Interruptions / mid-stream generation control / stopping vLLM generation.
- Wall-clock/time budgets.
- Determinism / virtual-time concurrency.
- Rehydrating a dismissed agent from disk (the namespaced `messages.jsonl` keeps
  the door open, but replay-persistence is not built in Phase 1).
