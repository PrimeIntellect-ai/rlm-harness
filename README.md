# rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents.

The model gets two built-in tools:

- `ipython` for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash`
- `summarize` for dropping old turns from context and optionally resetting REPL state

Inside the IPython session, the `rlm` module is pre-imported. When recursion is allowed, the model can call `await rlm.run(...)` to spawn sub-agents. Installed skills are also importable directly by name, e.g. `import websearch`.

## Install

Clone the repo, put any local skill packages under `skills/`, then run:

```bash
git clone https://github.com/PrimeIntellect-ai/rlm.git
cd rlm
uv sync --all-packages
source .venv/bin/activate
```

## CLI

```bash
rlm "fix the auth bug in login.py"

# Override model/limits
RLM_MODEL=gpt-4o RLM_MAX_TURNS=50 rlm "refactor the parser"

# Append extra instructions to the generated system prompt
RLM_APPEND_TO_SYSTEM_PROMPT="Always run tests before finishing." rlm "solve the task"

# Replace the generated system prompt from a file
RLM_SYSTEM_PROMPT_PATH=/tmp/system.txt rlm "solve the task"
```

Run skill CLIs the same way, for example `websearch --queries "latest jupyter_client release"`.

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

`RLM_SYSTEM_PROMPT_PATH` takes precedence over `RLM_APPEND_TO_SYSTEM_PROMPT`. CLI flags override env vars: `rlm --model gpt-5-mini --max-turns 50 --append-to-system-prompt "..." --system-prompt-path /tmp/system.txt "prompt"`.

## Recursion

Each agent runs inside a persistent IPython kernel. The `rlm` module is pre-imported there, so recursive calls look like normal Python:

```python
import asyncio
import rlm

result = asyncio.run(rlm.run("verify the fix"))
```

For parallel sub-agents, use normal async Python:

```python
import asyncio
import rlm

async def main():
    return await asyncio.gather(
        rlm.run("check auth.py"),
        rlm.run("check login.py"),
    )

results = asyncio.run(main())
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

From IPython, import the skill directly and call its async `run(...)` entrypoint:

```python
import asyncio
import websearch

print(websearch.PARAMETERS)
results = asyncio.run(websearch.run(queries=["latest jupyter_client release"]))
```

From the shell, invoke the same skill by command name:

```bash
websearch --queries "latest jupyter_client release"
```

Skill contract:

- each skill lives under `skills/<name>/`
- each skill must include `pyproject.toml`, `SKILL.md`, and `src/<name>/__init__.py`
- each skill must export `PARAMETERS`, async `run(...)`, and a same-name console script

### Writing Skills

Author skills as normal Python packages in this repo.

Recommended layout:

```text
skills/<name>/
├── SKILL.md
├── pyproject.toml
└── src/
    └── <name>/
        ├── __init__.py
        └── <name>.py
```

Recommended package pattern:

- `src/<name>/__init__.py` should be a thin re-export layer
- `src/<name>/<name>.py` should contain the main implementation
- helper modules can live alongside `<name>.py` as the skill grows

Minimum public surface:

- `PARAMETERS`: JSON-schema-like description of the skill inputs
- async `run(...)`: programmatic entrypoint
- `main()`: CLI entrypoint

Naming expectations:

- skill directory name: `<name>`
- distribution name in `pyproject.toml`: `rlm-skill-<name>`
- import name: `<name>`
- console script name: `<name>`

The public names should match across all interfaces:

- `PARAMETERS["properties"]`
- `run(...)` keyword arguments
- CLI flags and `--help` output

For example, if the Python API uses `queries=[...]`, then `PARAMETERS` should expose `queries` and the CLI should use `--queries`.

Dependency policy:

- declare skill-specific dependencies in the skill's `pyproject.toml`
- version conflicts between skills are currently the user's responsibility

Workspace expectations:

- every installable skill must live under `skills/<name>/` with its own `pyproject.toml`
- the root `pyproject.toml` includes `skills/*` as installable project members
- each skill should expose exactly one console script named `<name>`
- duplicate import names or console-script names will conflict at install/runtime, so skill authors must avoid them

## Kernel Modes

The IPython kernel can run in two modes depending on the environment.

### Native kernel (default)

The kernel runs inside rlm's own Python. All skills and the `rlm` module are importable natively. This is the default when `RLM_KERNEL_PYTHON` is unset.

Use this for non-Python projects (Go, Java, Rust) or when the sandbox has no `.venv`.

### External kernel (`RLM_KERNEL_PYTHON`)

Set `RLM_KERNEL_PYTHON` to point the kernel at a different Python interpreter — typically the sandbox's `.venv/bin/python3`. The kernel then runs inside the sandbox's Python with access to all its packages (numpy, pandas, etc.) for inline imports. The target Python must have `ipykernel` and `nest_asyncio` installed.

In training, the verifiers harness detects the sandbox `.venv`, installs `ipykernel`, and sets `RLM_KERNEL_PYTHON` automatically. For manual use:

```bash
export RLM_KERNEL_PYTHON=$(pwd)/.venv/bin/python3
rlm "fix the failing test"
```

Lightweight proxy modules are always registered at kernel startup, guaranteeing the kernel uses the rlm checkout's skills rather than any same-named packages in the sandbox. The proxies provide the same API (`import edit`, `edit.PARAMETERS`, `await edit.run(...)`) but delegate to the skill CLIs on PATH via subprocess.

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_KERNEL_PYTHON` | `sys.executable` | Python interpreter for the IPython kernel |
| `RLM_CHECKOUT_PATH` | `/tmp/rlm-checkout` | Path to the rlm source checkout (used to locate skill source for proxy generation) |

## Interactive Mode

Running `rlm` with no prompts enters a placeholder interactive mode. The TUI is not implemented yet.
