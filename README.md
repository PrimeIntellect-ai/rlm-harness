# rlm - training

⚠️ this is a training only harness, only use this to train on agentic task using verifiers 

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents.

The model gets three built-in tools (opt in / out via `RLM_TOOLS`):

- `ipython` for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash` (on by default)
- `bash` for stateless shell command execution (off by default)
- `edit` for single-occurrence string replacement in a file (off by default)

Context is reclaimed automatically: when a turn's prompt token count crosses `RLM_SUMMARIZE_AT_TOKENS`, the engine compacts the conversation into a summary and continues on a fresh branch. The IPython kernel keeps running across the compaction, so REPL state survives (see [Compaction](#compaction)).

Inside the IPython session, a callable `rlm` is pre-injected into the namespace. When recursion is allowed, the model can call `await rlm(...)` to spawn sub-agents. Skills supplied by the host environment (see [Skills](#skills)) are importable directly by name, e.g. `import websearch`.

## Install

```bash
git clone https://github.com/PrimeIntellect-ai/rlm.git
cd rlm
uv sync
source .venv/bin/activate
```

## CLI

```bash
rlm "fix the auth bug in login.py"

# Override model
RLM_MODEL=openai/gpt-5-mini rlm "refactor the parser"

# Append extra instructions to the generated system prompt
RLM_APPEND_TO_SYSTEM_PROMPT="Always run tests before finishing." rlm "solve the task"

# Replace the generated system prompt from a file
RLM_SYSTEM_PROMPT_PATH=/tmp/system.txt rlm "solve the task"
```

Skill CLIs provided by the host environment are on `$PATH` and invoked the same way (e.g. `websearch --queries "latest jupyter_client release"` when the `websearch` skill is installed).

## Python SDK

```python
import asyncio
import rlm

result = asyncio.run(rlm.run("fix the bug"))
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_HOME` | `~/.rlm` | Root directory for sessions and data |
| `RLM_MODEL` | `openai/gpt-5-mini` | Model name (PI Inference slug). Override with `--model` or `RLM_MODEL` for OpenAI/Anthropic direct (e.g. `gpt-4o`, `claude-sonnet-4-5`) |
| `RLM_API_KEY` / `RLM_BASE_URL` | — / SDK default (`https://api.openai.com/v1`) | Explicit override (highest priority). Independent: setting `RLM_API_KEY` alone targets the SDK default endpoint; set `RLM_BASE_URL` too for a custom endpoint. For PI, use `PRIME_API_KEY` (below) which owns the full pair. |
| `PRIME_API_KEY` | — | PI Inference pair: targets `https://api.pinference.ai/api/v1` and forwards `PRIME_TEAM_ID` as `X-Prime-Team-ID` when set. |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | resolved by SDK | OpenAI pair — when `OPENAI_API_KEY` is set, AsyncOpenAI's native env handling is used (covers OpenAI direct and verifiers' rollout tunnel both). Provider precedence: explicit → PI → OpenAI. Keys are scoped to their own base URL so an `OPENAI_API_KEY` lying around can't leak to PI Inference. |
| `RLM_TOOLS` | `ipython` | Comma-separated subset of builtin tools (`ipython`, `bash`, `edit`) to enable. Empty string = no tools. Unknown names raise. |
| `RLM_MAX_DEPTH` | `0` | Max recursion depth (`0` means no sub-agents) |
| `RLM_EXEC_TIMEOUT` | `300` | Seconds per IPython execution |
| `RLM_MAX_OUTPUT` | `-1` | Max chars returned from a tool call (`-1` disables truncation; `0` is invalid) |
| `RLM_SUMMARIZE_AT_TOKENS` | — | Auto-compaction threshold: when a turn's prompt tokens reach this value, the conversation is compacted into a summary. Unset disables auto-compaction. |
| `RLM_MAX_TOKENS` | `0` | Optional completion-token budget (`0` disables) |
| `RLM_APPEND_TO_SYSTEM_PROMPT` | — | Extra instructions appended to the generated system prompt |
| `RLM_SYSTEM_PROMPT_PATH` | — | Path to a file whose contents fully replace the generated system prompt |
| `RLM_ALLOW_GIT` | — | Set to `1` to disable the restricted git-history guard. When unset, shell-capable prompts tell agents not to use task-specific online hints or solutions from other git history, and broad-history `git log` options such as `--all` are refused. |
| `RLM_SDK_MAX_RETRIES` | `5` | Per-request retry count passed to the OpenAI SDK (in addition to the call-site retry wrapper that rides out longer outages). |

`RLM_SYSTEM_PROMPT_PATH` takes precedence over `RLM_APPEND_TO_SYSTEM_PROMPT`. CLI flags override env vars: `rlm --model gpt-5-mini --append-to-system-prompt "..." --system-prompt-path /tmp/system.txt "prompt"`.

## Recursion

Each agent runs inside a persistent IPython kernel with an already-running event loop. A callable `rlm` is pre-injected into the kernel namespace, so recursive calls are just `await`:

```python
result = await rlm("verify the fix")
```

The result is an `RLMResult` with `.answer`, `.usage`, `.turns`, and `.session_dir`. For parallel sub-agents, use normal async Python:

```python
import asyncio

results = await asyncio.gather(
    rlm("check auth.py"),
    rlm("check login.py"),
)
```

When recursion is disabled by depth, the system prompt does not advertise these APIs and child runs beyond the depth limit fail immediately.

## Compaction

There is no model-driven compaction tool. Compaction is automatic: set `RLM_SUMMARIZE_AT_TOKENS` and, once a turn's prompt token count reaches that threshold, the engine asks the model for a handoff summary and resumes the task on a fresh branch seeded with that summary. The original task prompt is dropped — the summary carries the goal forward.

The IPython kernel keeps running across the compaction, so all variables, imports, and in-memory data are preserved; the model is told to mention important variable names in its summary so the resumed branch knows what's available. With `RLM_SUMMARIZE_AT_TOKENS` unset, no auto-compaction occurs.

## Session Directory

Every invocation writes to `$RLM_HOME/sessions/<id>/`. Nested session directories mirror the call tree.

```text
.rlm/sessions/abc123/
├── meta.json
├── messages.jsonl
├── sub-d4e5/
│   ├── meta.json
│   ├── messages.jsonl
│   └── sub-f6g7/
└── sub-h8i9/
```

These artifacts are consumable for debugging, visualization, or training-data extraction.

## Skills

`rlm` itself ships no skills. Skills are supplied by the host environment: before `install.sh` runs, the environment places skill packages under `/task/rlm-skills/<name>/`, and `install.sh` installs them alongside `rlm` so they're both importable and on `$PATH`.

From IPython, import a skill and call its async `run(...)` entrypoint:

```python
import websearch

help(websearch)  # signature + docstring
results = await websearch(queries=["latest jupyter_client release"])
```

From the shell, invoke the same skill by command name:

```bash
websearch --queries "latest jupyter_client release"
```

### Skill contract

A skill is a normal Python package laid out like this:

```text
<name>/
├── SKILL.md
├── pyproject.toml
└── src/
    └── <name>/
        ├── __init__.py
        └── <name>.py
```

Required public surface:

- async `run(...)`: the single entrypoint. Type-annotated parameters and a Google-style docstring (`Args:`, `Returns:`) drive both the Python API and the CLI.

The `pyproject.toml` points its console script at the shared CLI entry:

```toml
[project.scripts]
<name> = "rlm.skill:cli"
```

`rlm.skill:cli` reads `sys.argv[0]`, imports the matching module, and uses [tyro](https://brentyi.github.io/tyro/) to build the argparse CLI from `run`'s signature. The return value of `run` is printed; a raise surfaces as a normal Python traceback.

Naming expectations (all match):

- skill directory name: `<name>`
- distribution name in `pyproject.toml`: `rlm-skill-<name>`
- import name: `<name>`
- console script name: `<name>`

Keyword arguments on `run(...)` and CLI flags line up automatically — `queries: list[str]` becomes `--queries` on the CLI.

Dependencies go in the skill's own `pyproject.toml`; declare `rlm` there so the shared CLI entry resolves. Version conflicts between skills installed side-by-side are the user's responsibility.

### Local development

For running `rlm` against a specific skill set outside of a sandbox-orchestrated environment, create a `/task/rlm-skills/` directory (or bind-mount one) and place skill packages there before running `install.sh`. The rlm repo ships no skills by default; look at the `rlm-swe` or `rlm-deepdive` environments for working skill packages to copy.

## Kernel

The IPython kernel always runs in rlm's own Python (`sys.executable`). `install.sh` puts `rlm` and all discovered skills into the same `uv tool install` environment, so `from rlm import run`, `import edit`, etc. work natively from inside an IPython cell.

To exercise packages from the target project's `.venv` (e.g. running its test suite), shell out from an IPython cell: `!./.venv/bin/python3 -m pytest`. The kernel itself stays isolated from whatever project venv the agent is working on — no cross-cell state involving sandbox packages.

## Developing

After `uv sync`, enable the pre-commit hooks:

```bash
uv run pre-commit install
```

Lint and format manually:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Testing

Install dev dependencies and run the suite:

```bash
uv sync --group dev
uv run pytest tests/
```

## Interactive Mode

Running `rlm` with no prompts enters a placeholder interactive mode. The TUI is not implemented yet.
