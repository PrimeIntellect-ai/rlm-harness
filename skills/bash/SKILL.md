---
name: bash
description: Run a shell command and return its output. Use for file exploration, running tests, installing packages, and invoking `rlm` for sub-tasks. For complex tasks, delegate sub-tasks to child agents via bash(command='rlm "sub-task"') or bash(command='rlm --batch "task1" "task2"'). Each child has the same capabilities and a fresh context window. Use delegation when sub-tasks are independent or benefit from parallel execution.
---
