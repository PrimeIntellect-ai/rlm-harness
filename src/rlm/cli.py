"""CLI entry point."""

from __future__ import annotations

import asyncio
import os
import sys

import rlm


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
        "--max-turns",
        type=int,
        default=None,
        help="Max turns (overrides RLM_MAX_TURNS)",
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

    # Apply CLI overrides to env
    if args.model:
        os.environ["RLM_MODEL"] = args.model
    if args.max_turns:
        os.environ["RLM_MAX_TURNS"] = str(args.max_turns)
    if args.system_prompt_path:
        os.environ["RLM_SYSTEM_PROMPT_PATH"] = args.system_prompt_path
    if args.append_to_system_prompt:
        os.environ["RLM_APPEND_TO_SYSTEM_PROMPT"] = args.append_to_system_prompt

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
