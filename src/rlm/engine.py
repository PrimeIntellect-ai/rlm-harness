"""The agent loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import time

from openai import AsyncOpenAI

from rlm.client import extract_usage, make_client
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.tools import SKILLS_DIR, IPythonREPL, get_active_tools
from rlm.types import RLMResult, TokenUsage


@dataclass
class SummarizeResult:
    content: str
    num_turns: int = 0
    flush_repl_state: bool = False


class RLMEngine:
    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        max_turns_in_context: int | None = None,
        cwd: str | None = None,
        session: Session | None = None,
        client: AsyncOpenAI | None = None,
    ):
        self.model = model or os.environ.get("RLM_MODEL", "gpt-4o")
        self.max_turns = max_turns or int(os.environ.get("RLM_MAX_TURNS", "30"))
        self.cwd = cwd or os.getcwd()
        self.exec_timeout = int(os.environ.get("RLM_EXEC_TIMEOUT", "300"))
        max_output = int(os.environ.get("RLM_MAX_OUTPUT", "-1"))
        if max_output == 0:
            raise ValueError(
                "RLM_MAX_OUTPUT must be positive, or -1 to disable truncation"
            )
        self.max_output = max_output
        limit = max_turns_in_context
        if limit is None:
            limit = int(os.environ.get("RLM_MAX_TURNS_IN_CONTEXT", "-1"))
        if limit < -1 or limit in (0, 1):
            raise ValueError("RLM_MAX_TURNS_IN_CONTEXT must be -1 (unlimited) or >= 2")
        self.max_turns_in_context = None if limit == -1 else limit
        self.max_depth = int(os.environ.get("RLM_MAX_DEPTH", "0"))
        self.depth = int(os.environ.get("RLM_DEPTH", "0"))

        # Token budget
        _max_tok = int(os.environ.get("RLM_MAX_TOKENS", "0"))
        self.max_tokens = _max_tok if _max_tok > 0 else None

        self.client = client or make_client()
        self.session = session
        self._total_usage = TokenUsage()

        # Summarize tool state
        self._summaries: list[str] = []
        self._dropped_turn_count: int = 0

        # IPython REPL (started lazily in single-agent execution)
        self._repl: IPythonREPL | None = None
        self._known_children: set[str] = set()

    def _ensure_session(self):
        """Create session if not set."""
        if self.session is not None:
            return
        session_dir = os.environ.get("RLM_SESSION_DIR")
        self.session = Session(session_dir)

    async def run(self, prompt: str) -> RLMResult:
        """Run a single agent loop to completion."""
        # Check depth limit
        if self.depth > self.max_depth:
            return RLMResult(
                answer=f"[depth limit {self.max_depth} reached, cannot start]",
                turns=0,
            )

        self._ensure_session()

        self.session.write_meta(
            session_id=self.session.dir.name,
            model=self.model,
            depth=self.depth,
            status="running",
            start_time=time.time(),
            prompt_preview=prompt[:200],
            cwd=self.cwd,
        )

        # Start IPython kernel
        self._repl = IPythonREPL(cwd=self.cwd, session=self.session)
        self._repl.start()
        self._known_children = {p.name for p in self.session.dir.glob("sub-*")}

        try:
            return await self._run_loop(prompt)
        finally:
            self._repl.shutdown()

    async def _run_loop(self, prompt: str) -> RLMResult:
        active_tools = get_active_tools()
        summarize_enabled = any(
            tool["function"]["name"] == "summarize" for tool in active_tools
        )
        messages_path = str(self.session.dir / "messages.jsonl")
        system_prompt = build_system_prompt(
            self.cwd,
            str(SKILLS_DIR),
            messages_path,
            allow_recursion=self.depth < self.max_depth,
            max_turns_in_context=self.max_turns_in_context,
            summarize_enabled=summarize_enabled,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        final_text = ""
        turn = 0

        for turn in range(self.max_turns):
            # Call LLM
            request_kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if active_tools:
                request_kwargs["tools"] = active_tools
                request_kwargs["parallel_tool_calls"] = False
            response = await self.client.chat.completions.create(
                **request_kwargs,
            )

            usage = extract_usage(response)
            self._total_usage.prompt_tokens += usage.prompt_tokens
            self._total_usage.completion_tokens += usage.completion_tokens

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

            if msg.tool_calls and len(msg.tool_calls) > 1:
                final_text = (
                    "[protocol violation: emitted multiple tool calls in one turn; "
                    "at most 1 is allowed]"
                )
                break

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

            # Execute the single allowed tool call
            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            t0 = time.time()
            summarize_result = None
            if tool_name == "summarize":
                summarize_result = self._execute_summarize(tool_args, messages)
                result = summarize_result.content
            else:
                result = await asyncio.to_thread(
                    self._execute_tool, tool_name, tool_args
                )
            duration = time.time() - t0

            summarize_num_turns = 0
            flush_repl_state = False
            if summarize_result is not None:
                summarize_num_turns = summarize_result.num_turns
                flush_repl_state = summarize_result.flush_repl_state

            if self.max_output > 0 and len(result) > self.max_output:
                result = result[: self.max_output] + "\n... [output truncated]"
            if flush_repl_state:
                result += "\n[repl state flushed]"
            if tool_name == "ipython" and self.max_turns_in_context is not None:
                turns_in_context = self._count_turns_in_context(messages)
                result += (
                    f"\n[context turns: {turns_in_context}/{self.max_turns_in_context}]"
                )
            self.session.log_tool_result(turn, tool_name, result, duration)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

            # Detect new child sessions spawned via rlm()
            self._detect_new_children()

            # Drop old turn-groups after summarize
            if summarize_num_turns > 0:
                self._drop_turns(messages, summarize_num_turns)
            if flush_repl_state:
                self._repl.restart_kernel()

            if self.max_turns_in_context is not None:
                turns_in_context = self._count_turns_in_context(messages)
                if turns_in_context > self.max_turns_in_context:
                    final_text = (
                        f"[context limit exceeded: {turns_in_context}/"
                        f"{self.max_turns_in_context} turns in context]"
                    )
                    break
        else:
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

    def _detect_new_children(self):
        """Scan session dir for new sub-* directories and log them."""
        if not self.session:
            return
        current = {p.name for p in self.session.dir.glob("sub-*")}
        new = current - self._known_children
        for child_name in sorted(new):
            self.session.log_sub_spawn(child_name, "(spawned via rlm())")
        self._known_children = current

    def _execute_summarize(self, args: dict, messages: list[dict]) -> SummarizeResult:
        num_turns = args.get("num_turns")
        summary = args.get("summary", "")
        flush_repl_state = bool(args.get("flush_repl_state", False))

        droppable = sum(1 for m in messages if m["role"] == "assistant") - 1

        if num_turns is None or num_turns <= 0:
            return SummarizeResult(
                content=(
                    "[no-op] num_turns is required and must be > 0 "
                    f"(got {num_turns}). No context was dropped."
                )
            )
        if num_turns > droppable:
            return SummarizeResult(
                content=(
                    f"[no-op] num_turns={num_turns} exceeds droppable turns "
                    f"({droppable}). No context was dropped."
                )
            )

        start = self._dropped_turn_count
        end = start + num_turns - 1
        self._summaries.append(f"[turns {start}-{end}] {summary}")
        self._dropped_turn_count += num_turns
        return SummarizeResult(
            content="\n\n".join(self._summaries),
            num_turns=num_turns,
            flush_repl_state=flush_repl_state,
        )

    @staticmethod
    def _count_turns_in_context(messages: list[dict]) -> int:
        return sum(1 for message in messages if message["role"] == "assistant")

    @staticmethod
    def _drop_turns(messages: list[dict], num_turns: int) -> None:
        start = 0
        while start < len(messages) and messages[start]["role"] != "assistant":
            start += 1

        end = start
        for _ in range(num_turns):
            if end >= len(messages) or messages[end]["role"] != "assistant":
                break
            end += 1
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
        del messages[start:end]

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "ipython":
            timeout = args.get("timeout") or self.exec_timeout
            return self._repl.execute(args["code"], timeout=timeout)
        return f"Error: unknown tool '{name}'"
