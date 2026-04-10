"""The agent loop."""

from __future__ import annotations

import asyncio
import json
import os
import time

from openai import AsyncOpenAI

from rlm.client import extract_usage, make_client
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.tools import SKILLS_DIR, get_active_tools, run_bash
from rlm.types import RLMResult, TokenUsage


class RLMEngine:
    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        cwd: str | None = None,
        session: Session | None = None,
        client: AsyncOpenAI | None = None,
    ):
        self.model = model or os.environ.get("RLM_MODEL", "gpt-4o")
        self.max_turns = max_turns or int(os.environ.get("RLM_MAX_TURNS", "30"))
        self.cwd = cwd or os.getcwd()
        self.bash_timeout = int(os.environ.get("RLM_BASH_TIMEOUT", "120"))
        self.max_output = int(os.environ.get("RLM_MAX_OUTPUT", "8192"))
        self.max_depth = int(os.environ.get("RLM_MAX_DEPTH", "3"))
        self.depth = int(os.environ.get("RLM_DEPTH", "0"))

        # Context window awareness
        self.max_context = int(os.environ.get("RLM_MAX_CONTEXT", "128000"))
        self._context_warning_sent = False
        self._last_prompt_tokens = 0

        # Token budget
        _max_tok = int(os.environ.get("RLM_MAX_TOKENS", "0"))
        self.max_tokens = _max_tok if _max_tok > 0 else None

        self.client = client or make_client()
        self.session = session
        self._total_usage = TokenUsage()

        # Summarize tool state
        self._summaries: list[str] = []
        self._dropped_turn_count: int = 0

    async def run(self, prompt: str) -> RLMResult:
        """Run the agent loop to completion."""
        # Check depth limit
        if self.depth >= self.max_depth:
            return RLMResult(
                answer=f"[depth limit {self.max_depth} reached, cannot start]",
                turns=0,
            )

        # Session setup
        if self.session is None:
            session_dir = os.environ.get("RLM_SESSION_DIR")
            self.session = Session(session_dir)

        self.session.write_meta(
            session_id=self.session.dir.name,
            model=self.model,
            depth=self.depth,
            status="running",
            start_time=time.time(),
            prompt_preview=prompt[:200],
            cwd=self.cwd,
        )

        active_tools = get_active_tools()
        messages_path = str(self.session.dir / "messages.jsonl")
        system_prompt = build_system_prompt(self.cwd, str(SKILLS_DIR), messages_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        final_text = ""
        turn = 0

        for turn in range(self.max_turns):
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=active_tools if active_tools else None,
            )

            usage = extract_usage(response)
            self._total_usage.prompt_tokens += usage.prompt_tokens
            self._total_usage.completion_tokens += usage.completion_tokens
            self._last_prompt_tokens = usage.prompt_tokens

            msg = response.choices[0].message
            msg_dict = msg.model_dump(exclude_none=True)
            msg_dict.setdefault("content", "")
            messages.append(msg_dict)

            # Log assistant message
            tool_calls_log = None
            if msg.tool_calls:
                tool_calls_log = [
                    {
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                    }
                    for tc in msg.tool_calls
                ]
            self.session.log_assistant(turn, tool_calls_log, msg.content)

            # Token budget check
            if (
                self.max_tokens
                and self._total_usage.completion_tokens >= self.max_tokens
            ):
                final_text = msg.content or "[token budget exhausted]"
                break

            # No tool calls → done
            if not msg.tool_calls:
                final_text = msg.content or ""
                break

            # Execute tool calls (parallel when multiple).
            # Summarize is handled inline — it's pure string ops and needs
            # to read/write engine state (_summaries, _dropped_turn_count).
            # After all results are appended, we drop old messages if
            # summarize was called.
            summarize_num_turns = 0

            async def _exec_one(tc):
                n = tc.function.name
                a = json.loads(tc.function.arguments)
                t0 = time.time()
                if n == "summarize":
                    r = self._execute_summarize(a, messages)
                else:
                    r = await asyncio.to_thread(self._execute_tool, n, a)
                return tc, r, time.time() - t0

            tool_results = await asyncio.gather(
                *[_exec_one(tc) for tc in msg.tool_calls]
            )

            for tc, result, duration in tool_results:
                # Track if summarize was called (for post-processing)
                if tc.function.name == "summarize" and not result.startswith("[no-op]"):
                    summarize_num_turns = json.loads(tc.function.arguments).get(
                        "num_turns", 0
                    )

                # Append token budget info (only when budget is set)
                if self.max_tokens:
                    result += f"\n[{self._total_usage.completion_tokens}/{self.max_tokens} completion tokens used]"

                # Context window warning (once, at 80%)
                if (
                    not self._context_warning_sent
                    and self._last_prompt_tokens >= self.max_context * 0.80
                ):
                    pct = self._last_prompt_tokens / self.max_context
                    result += (
                        f"\n\n[CONTEXT LIMIT WARNING] "
                        f"You have used {self._last_prompt_tokens:,} of "
                        f"{self.max_context:,} tokens ({pct:.0%}). "
                        f"Wrap up your task soon."
                    )
                    self._context_warning_sent = True

                self.session.log_tool_result(turn, tc.function.name, result, duration)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

            # Drop old turn-groups from messages after summarize.
            # A turn-group = one assistant message + its following tool messages.
            # Preamble (system/user) and current turn are never dropped.
            if summarize_num_turns > 0:
                self._drop_turns(messages, summarize_num_turns)
        else:
            # Max turns exhausted
            final_text = msg.content or "[max turns reached]"

        result = RLMResult(
            answer=final_text,
            session_dir=self.session.dir,
            usage=self._total_usage,
            turns=turn + 1,
        )
        self.session.finalize(
            final_text,
            usage={
                "prompt_tokens": self._total_usage.prompt_tokens,
                "completion_tokens": self._total_usage.completion_tokens,
            },
            turns=turn + 1,
        )
        return result

    async def batch(self, prompts: list[str]) -> list[RLMResult]:
        """Run multiple agents in parallel as subprocesses."""
        import asyncio

        async def _run_one(prompt: str) -> RLMResult:
            child_dir = self.session.child_dir() if self.session else None
            env = os.environ.copy()
            if child_dir:
                env["RLM_SESSION_DIR"] = str(child_dir)
            env["RLM_DEPTH"] = str(self.depth + 1)

            proc = await asyncio.create_subprocess_exec(
                "rlm",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=env,
            )
            stdout, _ = await proc.communicate()
            answer = stdout.decode().strip()

            # Read structured data from child's meta.json
            usage = TokenUsage()
            turns = 0
            if child_dir:
                meta_path = child_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    u = meta.get("usage", {})
                    usage = TokenUsage(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                    )
                    turns = meta.get("turns", 0)

            return RLMResult(
                answer=answer, session_dir=child_dir, usage=usage, turns=turns
            )

        return await asyncio.gather(*[_run_one(p) for p in prompts])

    def _execute_summarize(self, args: dict, messages: list[dict]) -> str:
        """Validate args, accumulate the summary, and return all summaries so far.

        Does NOT drop messages — that happens after all tool results are
        appended to the messages list (see _drop_turns).
        """
        num_turns = args.get("num_turns")
        summary = args.get("summary", "")

        # Count droppable turns: all assistant messages minus the current turn
        # (the last assistant message, which has the summarize call).
        droppable = sum(1 for m in messages if m["role"] == "assistant") - 1

        if num_turns is None or num_turns <= 0:
            return f"[no-op] num_turns is required and must be > 0 (got {num_turns}). No context was dropped."
        if num_turns > droppable:
            return (
                f"[no-op] num_turns={num_turns} exceeds droppable turns "
                f"({droppable}). No context was dropped."
            )

        start = self._dropped_turn_count
        end = start + num_turns - 1
        self._summaries.append(f"[turns {start}-{end}] {summary}")
        self._dropped_turn_count += num_turns
        return "\n\n".join(self._summaries)

    @staticmethod
    def _drop_turns(messages: list[dict], num_turns: int) -> None:
        """Remove the first num_turns turn-groups in place.

        Skips non-assistant messages at the start (system, user) automatically.
        A turn-group starts at an assistant message and includes all following
        tool messages up to (but not including) the next assistant message.
        """
        # Find where assistant messages begin (skip system/user preamble)
        start = 0
        while start < len(messages) and messages[start]["role"] != "assistant":
            start += 1

        # Walk forward, consuming num_turns turn-groups.
        # Each turn-group = one assistant message + its following tool messages.
        end = start
        for _ in range(num_turns):
            if end >= len(messages) or messages[end]["role"] != "assistant":
                break
            end += 1  # skip the assistant message
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1  # skip its tool messages
        del messages[start:end]

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "bash":
            return run_bash(
                args["command"],
                cwd=self.cwd,
                session=self.session,
                timeout=self.bash_timeout,
                max_output=self.max_output,
            )
        return f"Error: unknown tool '{name}'"
