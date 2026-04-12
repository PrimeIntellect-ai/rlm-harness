# rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents.

The model gets two built-in tools:

- `ipython` for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash`
- `summarize` for dropping old turns from context

Inside the IPython session, the `rlm` module is pre-imported. When recursion is allowed, the model can call `rlm.batch(...)` to spawn one or more sub-agents.

## Install

```bash
uv pip install -e .
```

## CLI

```bash
# Source your API keys
source .env

# Single prompt
uv run rlm "fix the auth bug in login.py"

# Multiple prompts run in parallel
uv run rlm "check auth.py" "check login.py" "check session.py"

# Override model/limits
RLM_MODEL=gpt-4o RLM_MAX_TURNS=50 uv run rlm "refactor the parser"
```

With one prompt, `rlm` prints a single answer. With multiple prompts, it runs a batch and prints one section per result.

## Python SDK

```python
import rlm

result = rlm.batch("fix the bug")[0]
results = rlm.batch(["check a.py", "check b.py"])
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
| `RLM_MAX_OUTPUT` | `8192` | Max chars returned from a tool call |
| `RLM_MAX_CONTEXT` | `128000` | Context-window warning threshold base |
| `RLM_MAX_TOKENS` | `0` | Optional completion-token budget (`0` disables) |
| `RLM_HOME` | `.rlm` | Root directory for sessions and data |
| `SERPER_API_KEY` | — | Optional API key for the bundled `skills/websearch` script |
| `RLM_WEBSEARCH_TIMEOUT` | `45` | Timeout for `skills/websearch` requests |
| `RLM_WEBSEARCH_NUM_RESULTS` | `5` | Organic results returned by `skills/websearch` |

CLI flags override env vars: `uv run rlm --model gpt-4o --max-turns 50 "prompt"`.

## Recursion

Each agent runs inside a persistent IPython kernel. The `rlm` module is pre-imported there, so recursive calls look like normal Python:

```python
rlm.batch(["verify the fix"])
rlm.batch(["check auth.py", "check login.py"])
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

## Interactive Mode

Running `rlm` with no prompts enters a placeholder interactive mode. The TUI is not implemented yet.
