"""Session directory management. Writes meta.json + messages.jsonl."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from rlm.types import ChildSessionAggregate, ProgrammaticToolCallStats


class Session:
    def __init__(self, session_dir: Path | None = None):
        if session_dir is None:
            sid = uuid.uuid4().hex[:12]
            rlm_home = Path(os.environ.get("RLM_HOME", ".rlm"))
            session_dir = rlm_home / "sessions" / sid
        self.dir = Path(session_dir)
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

    def log(self, entry: dict):
        """Append a line to messages.jsonl."""
        entry.setdefault("timestamp", time.time())
        self._msg_file.write(json.dumps(entry, default=str) + "\n")
        self._msg_file.flush()

    def log_assistant(
        self, turn: int, tool_calls: list[dict] | None, content: str | None
    ):
        entry = {"type": "assistant", "turn": turn}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if content:
            entry["content"] = content
        self.log(entry)

    def log_tool_result(self, turn: int, tool: str, content: str, duration: float):
        self.log(
            {
                "type": "tool_result",
                "turn": turn,
                "tool": tool,
                "content": content,
                "duration": round(duration, 3),
            }
        )

    def log_sub_spawn(self, child_name: str, command: str):
        self.log({"type": "sub_spawn", "child_dir": child_name, "command": command})

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
        entry = {"type": "done", "answer": answer[:1000]}
        if usage:
            entry["usage"] = usage
        if turns:
            entry["turns"] = turns
        self.log(entry)

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
    def child_dir(parent_dir: Path | str) -> Path:
        """Create and return a new child session directory under parent_dir."""
        child_id = uuid.uuid4().hex[:8]
        child = Path(parent_dir) / f"sub-{child_id}"
        child.mkdir()
        return child

    def close(self):
        if not self._msg_file.closed:
            self._msg_file.close()
