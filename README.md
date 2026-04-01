# rlm

A minimalistic CLI agent for true recursion.

Three tools: `bash`, `edit`, and `websearch`. Recursion via `bash('rlm "sub-task"')`. That's it.

## Install

```bash
uv pip install -e .
```

## Usage

```bash
# Source your API keys
source .env

# Headless (single task)
rlm "fix the auth bug in login.py"

# Parallel sub-tasks
rlm --batch "check auth.py" "check login.py" "check session.py"

# Override model/limits
RLM_MODEL=claude-sonnet-4-20250514 RLM_MAX_TURNS=50 rlm "refactor the parser"

# Restrict tools
RLM_TOOLS=bash rlm "explore the codebase"
```

## Python SDK

```python
import rlm

result = rlm.run("fix the bug")
results = rlm.batch(["check a.py", "check b.py"])
```

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_MODEL` | `gpt-4o` | LLM model |
| `RLM_API_KEY` | — | API key (falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `RLM_BASE_URL` | — | API endpoint |
| `RLM_MAX_TURNS` | `30` | Max tool-calling turns per agent |
| `RLM_MAX_DEPTH` | `3` | Max recursion depth |
| `RLM_BASH_TIMEOUT` | `120` | Seconds per bash command |
| `RLM_MAX_OUTPUT` | `8192` | Truncate tool output (chars) |
| `RLM_TOOLS` | `bash,edit,websearch` | Active tools (comma-separated) |
| `RLM_SUB_TOOLS` | — | Tools for children (if different) |
| `RLM_HOME` | `~/.rlm` | Root directory for sessions and data |
| `RLM_SYSTEM_PROMPT_VERBOSITY` | `medium` | light / medium / heavy |
| `SERPER_API_KEY` | — | API key for `websearch` tool ([serper.dev](https://serper.dev)) |
| `RLM_WEBSEARCH_TIMEOUT` | `45` | Per-query HTTP timeout (seconds) |
| `RLM_WEBSEARCH_NUM_RESULTS` | `5` | Organic results per query |

CLI flags override env vars: `rlm --model opus --max-turns 50 "prompt"`

## How recursion works

The LLM can invoke `rlm` as a sub-agent via bash — same binary, same tools, fresh context:

```
Root rlm                          Child rlm
  │                                 │
  │ bash('rlm "check auth.py"')    │
  │──────────────────────────────►  │
  │                                 │ bash("cat auth.py")
  │                                 │ bash("grep -n TODO auth.py")
  │                                 │ → "Found 2 issues..."
  │  ◄──────────────────────────────│
  │  stdout: "Found 2 issues..."   │
  │                                 │
```

Each child gets its own session directory nested under the parent's. The root's TUI (planned) watches the full tree.

## Session Directory

Every invocation writes to `$RLM_HOME/sessions/<id>/` (default `~/.rlm`). Set `RLM_HOME=.rlm` to keep sessions in the project directory. Nested directories mirror the call tree.

```
~/.rlm/sessions/abc123/
├── meta.json
├── messages.jsonl
├── sub-d4e5/           ← child from bash('rlm "sub-task"')
│   ├── meta.json
│   ├── messages.jsonl
│   └── sub-f6g7/      ← grandchild
│       └── ...
└── sub-h8i9/           ← parallel child
    └── ...
```

Consumable externally for SFT data extraction, visualization, or metrics.

## TUI (planned)

Interactive mode (`rlm` with no args). Splits on recursion:

### No recursion — flat view
```
┌────────────────────────────────────────────────┐
│ root                                           │
│                                                │
│ 👤 Fix the auth bug                            │
│ 🤖 bash: find . -name auth.py                  │
│    → ./src/auth.py                             │
│ 🤖 edit: src/auth.py                           │
│    → Edited                                    │
│ 🤖 bash: pytest                                │
│    → 13 passed                                 │
│ ✓ Fixed null check on line 47                  │
├────────────────────────────────────────────────┤
│ ❯ _                                            │
└────────────────────────────────────────────────┘
```

### One level of recursion — split in half
```
┌───────────────────────┬────────────────────────┐
│ root                  │ sub-a3f2               │
│                       │                        │
│ 👤 Fix the auth bug   │ 👤 verify the fix      │
│ 🤖 bash: find ...     │ 🤖 bash: cat auth.py   │
│    → ./src/auth.py    │    → <file content>    │
│ 🤖 edit: auth.py      │ 🤖 bash: pytest -x     │
│    → Edited           │    → 13 passed         │
│ 🤖 bash: rlm "verify" │ ✓ Fix looks correct    │
│    ⟳ running...       │                        │
│                       │                        │
├───────────────────────┴────────────────────────┤
│ ❯ _                                            │
└────────────────────────────────────────────────┘
```

### Two levels — right half splits again
```
┌───────────────────────┬────────────┬───────────┐
│ root                  │ sub-a3f2   │ sub-c8d1  │
│                       │            │           │
│ 👤 Fix auth + login   │ 👤 fix     │ 👤 check  │
│ 🤖 bash: rlm "fix     │   auth.py  │   tests   │
│   auth.py"            │ 🤖 bash:   │ 🤖 bash:  │
│ 🤖 bash: rlm "fix     │   cat ...  │   pytest  │
│   login.py"           │ 🤖 edit:   │   → pass  │
│    ⟳ running...       │   auth.py  │ ✓ ok      │
│                       │ 🤖 bash:   │           │
│                       │   rlm      │           │
│                       │   "check"  │           │
│                       │    ⟳       │           │
├───────────────────────┴────────────┴───────────┤
│ ❯ _                                            │
└────────────────────────────────────────────────┘
```

### Parallel children (batch) — right half stacks vertically
```
┌───────────────────────┬────────────────────────┐
│ root                  │ sub-a3f2 (auth.py)     │
│                       │ 🤖 bash: cat auth.py   │
│ 👤 Check all files    │    → ...               │
│ 🤖 bash: rlm --batch  │ 🤖 bash: grep TODO     │
│   "check auth.py"    │ ✓ Found 2 issues       │
│   "check login.py"   ├────────────────────────┤
│   "check session.py" │ sub-b7e4 (login.py)    │
│    ⟳ running...      │ 🤖 bash: cat login.py  │
│                       │    ⟳ running...        │
│                       ├────────────────────────┤
│                       │ sub-c8d1 (session.py)  │
│                       │ 🤖 bash: cat session.py│
│                       │    ⟳ running...        │
├───────────────────────┴────────────────────────┤
│ ❯ _                                            │
└────────────────────────────────────────────────┘
```

### Nested batch: 1 → 2 → 4
Root batches 2 children, each child batches 2 grandchildren. Right side shows the tree:
```
┌──────────────────┬──────────────────┬─────────────────┐
│ root             │ sub-a (backend)  │ sub-a1 (auth)   │
│                  │                  │ 🤖 bash: grep   │
│ 👤 Audit the     │ 👤 audit backend │    auth.py      │
│   whole app      │ 🤖 bash:         │ ✓ 2 issues      │
│ 🤖 bash:          │   rlm --batch   ├─────────────────┤
│   rlm --batch    │   "audit auth"  │ sub-a2 (db)     │
│   "audit backend"│   "audit db"    │ 🤖 bash: grep   │
│   "audit frontend│    ⟳            │    db.py        │
│    ⟳ running...  │                  │    ⟳ running... │
│                  ├──────────────────┼─────────────────┤
│                  │ sub-b (frontend) │ sub-b1 (react)  │
│                  │                  │ 🤖 bash: grep   │
│                  │ 👤 audit frontend│    App.tsx       │
│                  │ 🤖 bash:         │ ✓ 1 issue       │
│                  │   rlm --batch   ├─────────────────┤
│                  │   "audit react" │ sub-b2 (css)    │
│                  │   "audit css"   │ 🤖 bash: grep   │
│                  │    ⟳            │    styles.css   │
│                  │                  │    ⟳ running... │
├──────────────────┴──────────────────┴─────────────────┤
│ ❯ _                                                   │
└───────────────────────────────────────────────────────┘
```
