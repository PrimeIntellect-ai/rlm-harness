# rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents.

The model gets two built-in tools:

- `ipython` for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash`
- `summarize` for dropping old turns from context and optionally resetting REPL state

Inside the IPython session, the `rlm` module is pre-imported. When recursion is allowed, the model can call `await rlm.run(...)` to spawn sub-agents.

## Install

```bash
uv pip install -e .
```

## CLI

```bash
# Source your API keys
source .env

uv run rlm "fix the auth bug in login.py"

# Override model/limits
RLM_MODEL=gpt-4o RLM_MAX_TURNS=50 uv run rlm "refactor the parser"

# Append extra instructions to the generated system prompt
uv run rlm --append-to-system-prompt "Always run tests before finishing." "solve the task"

# Replace the generated system prompt from a file
uv run rlm --system-prompt-path /tmp/system.txt "solve the task"
```

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
| `SERPER_API_KEY` | — | Optional API key for the bundled `skills/websearch` script |
| `RLM_WEBSEARCH_TIMEOUT` | `45` | Timeout for `skills/websearch` requests |
| `RLM_WEBSEARCH_NUM_RESULTS` | `5` | Organic results returned by `skills/websearch` |

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

Bundled helper scripts live under [`skills/`](skills). The system prompt points the model at that directory so it can use those scripts from IPython when needed.

From IPython, import a tool module and `await` its `run(...)` function:

```python
from skills.websearch.scripts.websearch import run as websearch

await websearch(["latest jupyter_client release"])
```

## Interactive Mode

Running `rlm` with no prompts enters a placeholder interactive mode. The TUI is not implemented yet.
