---
name: bash
description: Run a shell command and return its output. Use for file exploration, running tests, installing packages, and invoking `rlm` for sub-tasks.
---

### bash
Run any shell command. Examples:
  bash(command="ls -la src/")
  bash(command="python -m pytest tests/ -x")
  bash(command="grep -rn TODO src/")

## Sub-agent delegation

For complex tasks, you can delegate sub-tasks to child agents by invoking `rlm` via bash:

Single sub-task:
  bash(command='rlm "check auth.py for security issues"')

Parallel sub-tasks:
  bash(command='rlm --batch "check auth.py" "check login.py" "check session.py"')

Each child agent has the same capabilities as you. Use delegation when:
- Sub-tasks are independent and can run in parallel
- You want to give a focused task a fresh context window
