# rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents.

The model gets two built-in tools:

- `ipython` for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash`
- `summarize` for dropping old turns from context and optionally resetting REPL state

Inside the IPython session, the `rlm` module is pre-imported. When recursion is allowed, the model can call `await rlm.run(...)` to spawn sub-agents. Installed skills are also importable directly by name, e.g. `import websearch`.

## Install

Put any local skill packages under `skills/`, then run:

```bash
./install.sh
source .venv/bin/activate
```

This rebuilds a repo-local `.venv`, installs `rlm`, then installs every valid skill package found under `skills/*/pyproject.toml`.

## CLI

```bash
# Source your API keys
source .env

rlm "fix the auth bug in login.py"

# Override model/limits
RLM_MODEL=gpt-4o RLM_MAX_TURNS=50 rlm "refactor the parser"

# Append extra instructions to the generated system prompt
rlm --append-to-system-prompt "Always run tests before finishing." "solve the task"

# Replace the generated system prompt from a file
rlm --system-prompt-path /tmp/system.txt "solve the task"
```

`uv run rlm ...` still works from the repo after installation.

## Python SDK

```python
import rlm

result = await rlm.run("fix the bug")
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_MODEL` | `gpt-4o` | Model name |
| `RLM_API_KEY` | — | API key for the OpenAI-compatible client |
| `RLM_BASE_URL` | — | Optional API base URL |
| `RLM_MAX_TURNS` | `30` | Max tool-calling turns per agent |
| `RLM_MAX_DEPTH` | `0` | Max recursion depth (`0` means no sub-agents) |
| `RLM_EXEC_TIMEOUT` | `300` | Seconds per IPython execution |
| `RLM_MAX_OUTPUT` | `-1` | Max chars returned from a tool call (`-1` disables truncation; `0` is invalid) |
| `RLM_MAX_TURNS_IN_CONTEXT` | `-1` | Max assistant turns retained in the live context (`-1` disables; `0` and `1` are invalid) |
| `RLM_MAX_TOKENS` | `0` | Optional completion-token budget (`0` disables) |
| `RLM_APPEND_TO_SYSTEM_PROMPT` | — | Extra instructions appended to the generated system prompt |
| `RLM_SYSTEM_PROMPT_PATH` | — | Path to a file whose contents fully replace the generated system prompt |
| `RLM_HOME` | `.rlm` | Root directory for sessions and data |
| `SERPER_API_KEY` | — | Optional API key for the bundled `websearch` skill |
| `RLM_WEBSEARCH_TIMEOUT` | `45` | Timeout for `websearch` requests |
| `RLM_WEBSEARCH_NUM_RESULTS` | `5` | Organic results returned by `websearch` |

`RLM_SYSTEM_PROMPT_PATH` takes precedence over `RLM_APPEND_TO_SYSTEM_PROMPT`. CLI flags override env vars: `uv run rlm --model gpt-4o --max-turns 50 --append-to-system-prompt "..." --system-prompt-path /tmp/system.txt "prompt"`.

## Recursion

Each agent runs inside a persistent IPython kernel. The `rlm` module is pre-imported there, so recursive calls look like normal Python:

```python
await rlm.run("verify the fix")
```

For parallel sub-agents, use normal async Python:

```python
import asyncio

results = await asyncio.gather(
    rlm.run("check auth.py"),
    rlm.run("check login.py"),
)
```

When recursion is disabled by depth, the system prompt does not advertise these APIs and child runs beyond the depth limit fail immediately.

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

Local skill packages live under [`skills/`](skills). Each skill is a normal Python package with its own `pyproject.toml`, top-level import name, and same-name shell command.

From IPython, import the skill directly and `await` its `run(...)` function:

```python
import websearch

print(websearch.PARAMETERS)
await websearch.run(queries=["latest jupyter_client release"])
```

From the shell, invoke the same skill by command name:

```bash
websearch --queries "latest jupyter_client release"
```

Skill contract:

- each skill lives under `skills/<name>/`
- each skill must include `pyproject.toml`, `SKILL.md`, and `src/<name>/__init__.py`
- each skill must export `PARAMETERS`, async `run(...)`, and a same-name console script
- `install.sh` validates duplicate import names and duplicate console-script names before installing

## Interactive Mode

Running `rlm` with no prompts enters a placeholder interactive mode. The TUI is not implemented yet.
