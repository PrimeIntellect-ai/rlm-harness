# rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents.

The model gets two built-in tools:

- `ipython` for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash`
- `summarize` for dropping old turns from context and optionally resetting REPL state

Inside the IPython session, the `rlm` module is pre-imported. When recursion is allowed, the model can call `await rlm.run(...)` to spawn sub-agents. Skills supplied by the host environment (see [Skills](#skills)) are importable directly by name, e.g. `import websearch`.

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

# Override model/limits
RLM_MODEL=gpt-4o RLM_MAX_TURNS=50 rlm "refactor the parser"

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
| `RLM_TOOLS` | `ipython,summarize` | Comma-separated subset of builtin tools to enable. Empty string = no tools. Unknown names raise. |
| `RLM_HOME` | `.rlm` | Root directory for sessions and data |

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

`rlm` itself ships no skills. Skills are supplied by the host environment: before `install.sh` runs, the environment places skill packages under `/task/rlm-skills/<name>/`, and `install.sh` installs them alongside `rlm` so they're both importable and on `$PATH`.

From IPython, import a skill and call its async `run(...)` entrypoint:

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

- `PARAMETERS`: JSON-schema-like description of the inputs
- async `run(...)`: programmatic entrypoint
- `main()`: CLI entrypoint

Naming expectations (all match):

- skill directory name: `<name>`
- distribution name in `pyproject.toml`: `rlm-skill-<name>`
- import name: `<name>`
- console script name: `<name>`

Keyword arguments on `run(...)`, keys under `PARAMETERS["properties"]`, and CLI flags should all line up. For example, if the Python API uses `queries=[...]`, `PARAMETERS` should expose `queries` and the CLI should use `--queries`.

Dependencies go in the skill's own `pyproject.toml`. Version conflicts between skills installed side-by-side are the user's responsibility.

### Local development

For running `rlm` against a specific skill set outside of a sandbox-orchestrated environment, create a `/task/rlm-skills/` directory (or bind-mount one) and place skill packages there before running `install.sh`. The rlm repo ships no skills by default; look at the `rlm-swe` or `rlm-deepdive` environments for working skill packages to copy.

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

Lightweight proxy modules are always registered at kernel startup, guaranteeing the kernel uses the uploaded skills at `/task/rlm-skills` rather than any same-named packages in the sandbox. The proxies provide the same API (`import edit`, `edit.PARAMETERS`, `await edit.run(...)`) but delegate to the skill CLIs on PATH via subprocess.

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_KERNEL_PYTHON` | `sys.executable` | Python interpreter for the IPython kernel |
| `RLM_CHECKOUT_PATH` | `/tmp/rlm-checkout` | Path to the rlm source checkout |

## Interactive Mode

Running `rlm` with no prompts enters a placeholder interactive mode. The TUI is not implemented yet.
