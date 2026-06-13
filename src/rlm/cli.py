"""CLI entry point."""

from __future__ import annotations

import asyncio
import os
import sys

import rlm
import rlm.config


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="rlm",
        description="A minimalistic CLI agent for true recursion.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task prompt (omit for interactive mode)",
    )
    parser.add_argument(
        "--model", default=None, help="Model name (overrides RLM_MODEL)"
    )
    parser.add_argument(
        "--system-prompt-path",
        default=None,
        help="Path to a file whose contents replace the generated system prompt",
    )
    parser.add_argument(
        "--append-to-system-prompt",
        default=None,
        help="Extra instructions appended to the generated system prompt",
    )
    args = parser.parse_args()

    # Apply CLI overrides to env, then refresh the cached config so this process
    # (and the kernels it spawns, which inherit the env) sees them.
    if args.model:
        os.environ["RLM_MODEL"] = args.model
    if args.system_prompt_path:
        os.environ["RLM_SYSTEM_PROMPT_PATH"] = args.system_prompt_path
    if args.append_to_system_prompt:
        os.environ["RLM_APPEND_TO_SYSTEM_PROMPT"] = args.append_to_system_prompt
    rlm.config.reload_config()

    if args.prompt:
        print(asyncio.run(rlm.run(args.prompt)).answer)
    else:
        _run_interactive()


def _run_interactive():
    print("rlm interactive mode")
    print('TUI not yet implemented. Use: rlm "your prompt" for headless mode.')
    sys.exit(0)


if __name__ == "__main__":
    main()
