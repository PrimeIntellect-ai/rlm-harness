"""Session directory management. Writes meta.json + messages.jsonl."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

from rlm.config import get_config
from rlm.types import ChildSessionAggregate, ProgrammaticToolCallStats


def sanitize_name(name: str) -> str:
    """Make ``name`` filesystem-safe for a session dir; non-empty and bounded."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-")
    return (safe or "agent")[:64]


MSG_WRAPPER_KEYS = frozenset({"t", "view", "turn", "ts", "duration"})


def strip_msg(obj: dict) -> dict:
    """Drop the view-log wrapper keys, leaving the raw OpenAI message dict."""
    return {k: v for k, v in obj.items() if k not in MSG_WRAPPER_KEYS}


class Session:
    def __init__(self, session_dir: Path | None = None):
        if session_dir is None:
            sid = uuid.uuid4().hex[:12]
            session_dir = get_config().home / "sessions" / sid
        # Absolute path so later writes (meta.json.tmp, messages.jsonl) keep
        # working if something changes cwd mid-rollout (a tool's os.chdir,
        # REPL kernel restart in a different cwd, sandbox teardown, etc.).
        self.dir = Path(session_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._msg_file = open(self.dir / "messages.jsonl", "a")

    def write_meta(self, **kwargs):
        """Write meta.json atomically."""
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            existing.update(kwargs)
            data = existing
        else:
            data = kwargs
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(meta_path)

    def _write(self, obj: dict) -> None:
        """Append one typed line to messages.jsonl."""
        obj.setdefault("ts", time.time())
        self._msg_file.write(json.dumps(obj, default=str) + "\n")
        self._msg_file.flush()

    def log_message(
        self, view: int, turn: int, message: dict, *, duration: float | None = None
    ) -> None:
        """Append a conversation message (OpenAI shape) to the current view."""
        line = {"t": "msg", "view": view, "turn": turn, **message}
        if duration is not None:
            line["duration"] = round(duration, 3)
        self._write(line)

    def branch_reset(
        self, view: int, *, dropped_chars: int, summary_chars: int, turns_since: int
    ) -> None:
        """Close compaction branch ``view``; later msgs carry ``view + 1``."""
        self._write(
            {
                "t": "branch_reset",
                "view": view,
                "dropped_chars": dropped_chars,
                "summary_chars": summary_chars,
                "turns_since": turns_since,
            }
        )

    def log_spawn(self, child: str) -> None:
        self._write({"t": "spawn", "child": child})

    def load_latest_view(self) -> tuple[int, list[dict]]:
        """Reconstruct the latest view (the engine's ``_messages``) from disk.

        Returns ``(view_index, messages)`` for the highest view, in order — the
        exact context the model last had. The index is authoritative for resume
        (it always matches the returned messages, even if a crash mid-compaction
        left meta.json's view stale). ``(0, [])`` when there is no transcript yet.
        """
        path = self.dir / "messages.jsonl"
        if not path.exists():
            return 0, []
        with open(path) as f:
            lines = f.readlines()
        # Drop trailing blank lines so a torn final record followed by a blank
        # line (or a stray newline) is still recognized as the tail rather than
        # mid-file corruption.
        while lines and not lines[-1].strip():
            lines.pop()
        msgs: list[dict] = []
        for i, raw in enumerate(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # A hard crash mid-write can leave the final line torn; drop it
                # and resume from the rest (the view minus its last unfinished
                # message is still a valid continuation). A malformed line
                # anywhere else is real corruption, so re-raise.
                if i == len(lines) - 1:
                    logging.getLogger(__name__).warning(
                        "dropping torn final line in %s", path
                    )
                    break
                raise
            if obj.get("t") == "msg":
                msgs.append(obj)
        if not msgs:
            return 0, []
        latest = max(m.get("view", 0) for m in msgs)
        return latest, [strip_msg(m) for m in msgs if m.get("view", 0) == latest]

    def aggregate_child_metrics(self) -> ChildSessionAggregate:
        """Walk sub-*/meta.json and bundle their context-token + tool-call stats."""
        aggregate = ChildSessionAggregate()
        for child_dir in self.dir.glob("sub-*"):
            meta_path = child_dir / "meta.json"
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except FileNotFoundError:
                continue
            # Missing context_token_stats means the child crashed before
            # finalize; silently treat as zero contribution so child failures
            # don't cascade to parent (and grandparent, etc.). Still raise on
            # genuinely malformed data (corruption rather than absence).
            ctx_stats = meta.get("context_token_stats", {})
            if not isinstance(ctx_stats, dict):
                raise RuntimeError(
                    f"Malformed context_token_stats in child session meta: {meta_path}"
                )
            aggregate.absorb(ctx_stats, ProgrammaticToolCallStats.from_meta(meta))
        return aggregate

    def finalize(
        self, answer: str, usage: dict | None = None, turns: int = 0, metrics=None
    ):
        done = {"t": "done", "answer": answer[:1000]}
        if usage:
            done["usage"] = usage
        if turns:
            done["turns"] = turns
        self._write(done)

        meta_update = {"status": "done", "answer_preview": answer[:200], "turns": turns}
        if usage:
            meta_update["usage"] = usage
        if metrics is not None:
            direct_tool_stats = ProgrammaticToolCallStats.from_log(
                self.dir / "programmatic_tool_calls.jsonl"
            )
            child = self.aggregate_child_metrics()

            metrics.finalize_current_branch()
            metrics.apply_child_aggregates(child.context_token_stats)
            metrics.apply_programmatic_tool_call_stats(
                direct_tool_stats, child.tool_call_stats
            )

            meta_update["metrics"] = metrics.to_dict()
            meta_update["context_token_stats"] = metrics.context_token_stats()
            meta_update["programmatic_tool_call_stats"] = direct_tool_stats.merge(
                child.tool_call_stats
            ).to_dict()
        self.write_meta(**meta_update)
        self._msg_file.close()

    @staticmethod
    def child_dir(parent_dir: Path | str, name: str | None = None) -> Path:
        """Create and return a child session directory under ``parent_dir``.

        With ``name`` the dir is ``sub-<sanitized-name>`` (stable across
        re-sends, human-readable); otherwise a random ``sub-<id>``.
        """
        if name is not None:
            child = Path(parent_dir) / f"sub-{sanitize_name(name)}"
            child.mkdir(parents=True, exist_ok=True)
        else:
            child_id = uuid.uuid4().hex[:8]
            child = Path(parent_dir) / f"sub-{child_id}"
            child.mkdir()
        return child

    def close(self):
        if not self._msg_file.closed:
            self._msg_file.close()
