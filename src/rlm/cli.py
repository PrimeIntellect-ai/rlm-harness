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
        "prompts",
        nargs="*",
        help="One or more task prompts (omit for interactive mode)",
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
    args = parser.parse_args()

    # Apply CLI overrides to env
    if args.model:
        os.environ["RLM_MODEL"] = args.model
    if args.max_turns:
        os.environ["RLM_MAX_TURNS"] = str(args.max_turns)

    if len(args.prompts) > 1:
        results = rlm.batch(args.prompts)
        for i, r in enumerate(results):
            print(f"--- [{i}] ---")
            print(r.answer)
            print()
    elif len(args.prompts) == 1:
        print(rlm.batch(args.prompts[0])[0].answer)
    else:
        _run_interactive()


def _run_interactive():
    print("rlm interactive mode")
    print('TUI not yet implemented. Use: rlm "your prompt" for headless mode.')
    sys.exit(0)


if __name__ == "__main__":
    main()
