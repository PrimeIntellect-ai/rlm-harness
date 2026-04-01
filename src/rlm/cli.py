"""CLI entry point."""

from __future__ import annotations

import asyncio
import os
import sys

from rlm.engine import RLMEngine
from rlm.session import Session


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="rlm",
        description="A minimalistic CLI agent for true recursion.",
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Task prompt (omit for interactive mode)")
    parser.add_argument("--batch", action="store_true", help="Run multiple prompts in parallel")
    parser.add_argument("--model", default=None, help="Model name (overrides RLM_MODEL)")
    parser.add_argument("--max-turns", type=int, default=None, help="Max turns (overrides RLM_MAX_TURNS)")
    parser.add_argument("--tools", default=None, help="Comma-separated tool names (overrides RLM_TOOLS)")
    parser.add_argument("--replay", nargs="?", const="", default=None,
                        help="Replay a session (optionally specify session ID)")
    parser.add_argument("--inspect", action="store_true",
                        help="Open inspection view (with --replay)")

    args, remaining = parser.parse_known_args()

    # Apply CLI overrides to env
    if args.model:
        os.environ["RLM_MODEL"] = args.model
    if args.max_turns:
        os.environ["RLM_MAX_TURNS"] = str(args.max_turns)
    if args.tools:
        os.environ["RLM_TOOLS"] = args.tools

    if args.replay is not None:
        _run_replay(args.replay, inspect=args.inspect)
    elif args.batch:
        prompts = [args.prompt] + remaining if args.prompt else remaining
        if not prompts:
            parser.error("--batch requires at least one prompt")
        asyncio.run(_run_batch(prompts))
    elif args.prompt:
        asyncio.run(_run_headless(args.prompt))
    else:
        _run_interactive()


async def _run_headless(prompt: str):
    engine = RLMEngine()
    result = await engine.run(prompt)
    print(result.answer)


async def _run_batch(prompts: list[str]):
    engine = RLMEngine()
    session_dir = os.environ.get("RLM_SESSION_DIR")
    engine.session = Session(session_dir)
    engine.session.write_meta(
        session_id=engine.session.dir.name,
        model=engine.model,
        status="batch",
        batch_size=len(prompts),
    )

    results = await engine.batch(prompts)
    for i, r in enumerate(results):
        print(f"--- [{i}] ---")
        print(r.answer)
        print()


def _run_interactive():
    from rlm.tui import RLMApp
    app = RLMApp(mode="interactive")
    app.run()


def _run_replay(session_id: str, inspect: bool = False):
    from pathlib import Path
    from rlm.tui import RLMApp, SessionData

    if not session_id:
        # No ID: show session browser
        app = RLMApp(mode="browse")
        app.run()
        return

    # Find session by ID (prefix match)
    sessions_dir = Path.home() / ".rlm" / "sessions"
    session_path = None
    if (sessions_dir / session_id).is_dir():
        session_path = sessions_dir / session_id
    else:
        # Prefix search
        for d in sessions_dir.iterdir():
            if d.is_dir() and d.name.startswith(session_id):
                session_path = d
                break

    if session_path is None:
        print(f"Session not found: {session_id}")
        sys.exit(1)

    sd = SessionData(path=session_path)
    sd.load_tree()

    mode = "inspect" if inspect else "replay"
    app = RLMApp(mode=mode, session_data=sd)
    app.run()


if __name__ == "__main__":
    main()
