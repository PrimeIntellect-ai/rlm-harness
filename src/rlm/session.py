"""Session directory management. Writes meta.json + messages.jsonl."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


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

    def aggregate_child_metrics(self) -> dict[str, int]:
        """Read all sub-*/meta.json and aggregate recursive session totals."""
        totals = {
            "session_count": 0,
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "final_input_tokens_total": 0,
            "final_output_tokens_total": 0,
            "branch_count": 0,
            "branch_input_tokens_sum": 0,
            "branch_input_tokens_max": 0,
            "branch_output_tokens_sum": 0,
            "branch_output_tokens_max": 0,
        }

        for child_dir in self.dir.glob("sub-*"):
            meta_path = child_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                stats = meta.get("context_token_stats")
                if not isinstance(stats, dict):
                    raise RuntimeError(
                        f"Missing context_token_stats in child session meta: {meta_path}"
                    )
                for key in (
                    "session_count",
                    "input_tokens_total",
                    "output_tokens_total",
                    "final_input_tokens_total",
                    "final_output_tokens_total",
                    "branch_count",
                    "branch_input_tokens_sum",
                    "branch_output_tokens_sum",
                ):
                    totals[key] += int(stats.get(key, 0))
                totals["branch_input_tokens_max"] = max(
                    totals["branch_input_tokens_max"],
                    int(stats.get("branch_input_tokens_max", 0)),
                )
                totals["branch_output_tokens_max"] = max(
                    totals["branch_output_tokens_max"],
                    int(stats.get("branch_output_tokens_max", 0)),
                )

        return totals

    def finalize(
        self, answer: str, usage: dict | None = None, turns: int = 0, metrics=None
    ):
        entry = {"type": "done", "answer": answer[:1000]}
        if usage:
            entry["usage"] = usage
        if turns:
            entry["turns"] = turns
        self.log(entry)

        # Aggregate child sub-RLM metrics
        if metrics is not None:
            metrics.finalize_current_branch()
            metrics.apply_child_aggregates(self.aggregate_child_metrics())

        meta_update = {"status": "done", "answer_preview": answer[:200], "turns": turns}
        if usage:
            meta_update["usage"] = usage
        if metrics is not None:
            meta_update["metrics"] = metrics.to_dict()
            meta_update["context_token_stats"] = metrics.context_token_stats()
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
