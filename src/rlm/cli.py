"""CLI entry point."""

from __future__ import annotations

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
        default=None,
        help="Task prompt (omit for interactive mode)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Run multiple prompts in parallel"
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
    args, remaining = parser.parse_known_args()

    # Apply CLI overrides to env
    if args.model:
        os.environ["RLM_MODEL"] = args.model
    if args.max_turns:
        os.environ["RLM_MAX_TURNS"] = str(args.max_turns)

    if args.batch:
        prompts = [args.prompt] + remaining if args.prompt else remaining
        if not prompts:
            parser.error("--batch requires at least one prompt")
        results = rlm.batch(prompts)
        for i, r in enumerate(results):
            print(f"--- [{i}] ---")
            print(r.answer)
            print()
    elif args.prompt:
        result = rlm.run(args.prompt)
        print(result.answer)
    else:
        _run_interactive()


def _run_interactive():
    print("rlm interactive mode")
    print('TUI not yet implemented. Use: rlm "your prompt" for headless mode.')
    sys.exit(0)


if __name__ == "__main__":
    main()
