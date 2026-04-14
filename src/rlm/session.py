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
                "content": content[:2000],  # cap for readability
                "duration": round(duration, 3),
            }
        )

    def log_sub_spawn(self, child_name: str, command: str):
        self.log({"type": "sub_spawn", "child_dir": child_name, "command": command})

    def aggregate_child_metrics(self) -> tuple[int, int, int]:
        """Read all sub-*/meta.json and sum their token usage.

        Returns (sub_prompt_tokens, sub_completion_tokens, sub_count).
        """
        sub_prompt = 0
        sub_completion = 0
        sub_count = 0

        for child_dir in self.dir.glob("sub-*"):
            meta_path = child_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                usage = meta.get("usage", {})
                metrics = meta.get("metrics", {})

                # This child's direct usage
                sub_prompt += usage.get("prompt_tokens", 0)
                sub_completion += usage.get("completion_tokens", 0)
                sub_count += 1

                # Plus its children's usage (recursive aggregation)
                sub_prompt += metrics.get("sub_rlm_prompt_tokens", 0)
                sub_completion += metrics.get("sub_rlm_completion_tokens", 0)
                sub_count += metrics.get("sub_rlm_count", 0)

        return sub_prompt, sub_completion, sub_count

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
            sub_prompt, sub_completion, sub_count = self.aggregate_child_metrics()
            metrics.sub_rlm_prompt_tokens = sub_prompt
            metrics.sub_rlm_completion_tokens = sub_completion
            metrics.sub_rlm_count = sub_count

        meta_update = {"status": "done", "answer_preview": answer[:200], "turns": turns}
        if usage:
            meta_update["usage"] = usage
        if metrics is not None:
            meta_update["metrics"] = metrics.to_dict()
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
