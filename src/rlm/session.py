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

    def read_programmatic_tool_call_stats(self) -> dict[str, int | dict[str, int]]:
        """Read this session's programmatic tool call log and count calls by source."""
        stats: dict[str, int | dict[str, int]] = {
            "python_total": 0,
            "bash_total": 0,
            "by_tool_python": {},
            "by_tool_bash": {},
        }
        log_path = self.dir / "programmatic_tool_calls.jsonl"
        if not log_path.exists():
            return stats

        for line in log_path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tool = entry.get("tool")
            source = entry.get("source")
            if not isinstance(tool, str) or source not in {"python", "bash"}:
                continue

            total_key = f"{source}_total"
            by_tool_key = f"by_tool_{source}"
            stats[total_key] = int(stats[total_key]) + 1
            by_tool = stats[by_tool_key]
            assert isinstance(by_tool, dict)
            by_tool[tool] = int(by_tool.get(tool, 0)) + 1

        return stats

    def aggregate_child_metrics(
        self,
    ) -> tuple[int, int, int, dict[str, int | dict[str, int]]]:
        """Read all sub-*/meta.json and aggregate recursive session totals."""
        sub_prompt = 0
        sub_completion = 0
        sub_count = 0
        tool_stats: dict[str, int | dict[str, int]] = {
            "python_total": 0,
            "bash_total": 0,
            "by_tool_python": {},
            "by_tool_bash": {},
        }

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

                child_tool_stats = meta.get("programmatic_tool_call_stats", {})
                if not isinstance(child_tool_stats, dict):
                    continue
                tool_stats["python_total"] = int(tool_stats["python_total"]) + int(
                    child_tool_stats.get("python_total", 0)
                )
                tool_stats["bash_total"] = int(tool_stats["bash_total"]) + int(
                    child_tool_stats.get("bash_total", 0)
                )
                for source in ("python", "bash"):
                    totals = tool_stats[f"by_tool_{source}"]
                    child_counts = child_tool_stats.get(f"by_tool_{source}", {})
                    if not isinstance(totals, dict) or not isinstance(
                        child_counts, dict
                    ):
                        continue
                    for tool_name, count in child_counts.items():
                        if not isinstance(tool_name, str):
                            continue
                        totals[tool_name] = int(totals.get(tool_name, 0)) + int(count)

        return sub_prompt, sub_completion, sub_count, tool_stats

    @staticmethod
    def merge_programmatic_tool_call_stats(
        direct: dict[str, int | dict[str, int]],
        child: dict[str, int | dict[str, int]],
    ) -> dict[str, int | dict[str, int]]:
        """Merge direct and child programmatic tool call stats."""
        merged: dict[str, int | dict[str, int]] = {
            "python_total": int(direct.get("python_total", 0))
            + int(child.get("python_total", 0)),
            "bash_total": int(direct.get("bash_total", 0))
            + int(child.get("bash_total", 0)),
            "by_tool_python": {},
            "by_tool_bash": {},
        }
        for source in ("python", "bash"):
            merged_by_tool = merged[f"by_tool_{source}"]
            assert isinstance(merged_by_tool, dict)
            for stats in (direct, child):
                by_tool = stats.get(f"by_tool_{source}", {})
                if not isinstance(by_tool, dict):
                    continue
                for tool_name, count in by_tool.items():
                    if not isinstance(tool_name, str):
                        continue
                    merged_by_tool[tool_name] = int(
                        merged_by_tool.get(tool_name, 0)
                    ) + int(count)
        return merged

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
        direct_tool_stats = self.read_programmatic_tool_call_stats()
        merged_tool_stats = direct_tool_stats
        if metrics is not None:
            sub_prompt, sub_completion, sub_count, child_tool_stats = (
                self.aggregate_child_metrics()
            )
            metrics.sub_rlm_prompt_tokens = sub_prompt
            metrics.sub_rlm_completion_tokens = sub_completion
            metrics.sub_rlm_count = sub_count
            metrics.apply_programmatic_tool_call_counts(
                int(direct_tool_stats.get("python_total", 0)),
                int(direct_tool_stats.get("bash_total", 0)),
                int(child_tool_stats.get("python_total", 0)),
                int(child_tool_stats.get("bash_total", 0)),
            )
            merged_tool_stats = self.merge_programmatic_tool_call_stats(
                direct_tool_stats, child_tool_stats
            )

        meta_update = {"status": "done", "answer_preview": answer[:200], "turns": turns}
        if usage:
            meta_update["usage"] = usage
        if metrics is not None:
            meta_update["metrics"] = metrics.to_dict()
            meta_update["programmatic_tool_call_stats"] = merged_tool_stats
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
