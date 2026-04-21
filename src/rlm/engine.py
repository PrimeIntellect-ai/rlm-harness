"""The agent loop."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time

from openai import AsyncOpenAI

from rlm.client import extract_usage, make_client
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.tools import (
    SKILLS_DIR,
    IPythonREPL,
    SummarizeState,
    ToolContext,
    ToolOutcome,
    get_active_tools,
    get_builtin_tool,
    get_installed_skills,
    summarize_drop_slice_bounds,
)
from rlm.types import RLMMetrics, RLMResult, TokenUsage


def _parse_tool_call_args(raw: str) -> tuple[dict | None, dict | None]:
    """Parse a tool-call arguments blob. Returns (args, error_info).

    On success, args is the parsed dict and error_info is None.
    On failure (invalid JSON, wrong type, or non-object JSON like ``null`` /
    ``42`` / ``"foo"`` / ``[]``), args is None and error_info is a dict
    suitable for logging (with ``_parse_error`` and ``_raw`` keys). Callers
    that need a string error message should read ``error_info["_parse_error"]``.

    Tool schemas require objects, so anything that parses to a non-dict is
    treated as an error — otherwise ``args is None`` would be ambiguous
    (parse failure vs. JSON ``null``) and non-dict values would silently
    reach ``tool.execute`` and crash there with a less useful message.
    """
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {
            "_parse_error": f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            "_raw": raw,
        }
    except TypeError as exc:
        return None, {"_parse_error": str(exc), "_raw": raw}
    if not isinstance(args, dict):
        return None, {
            "_parse_error": f"expected JSON object, got {type(args).__name__}",
            "_raw": raw,
        }
    return args, None


class RLMEngine:
    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        max_turns_in_context: int | None = None,
        system_prompt_path: str | None = None,
        append_to_system_prompt: str | None = None,
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
        self.system_prompt_path = system_prompt_path or os.environ.get(
            "RLM_SYSTEM_PROMPT_PATH"
        )
        self.append_to_system_prompt = append_to_system_prompt or os.environ.get(
            "RLM_APPEND_TO_SYSTEM_PROMPT"
        )
        self.max_depth = int(os.environ.get("RLM_MAX_DEPTH", "0"))
        self.depth = int(os.environ.get("RLM_DEPTH", "0"))

        # Token budget
        _max_tok = int(os.environ.get("RLM_MAX_TOKENS", "0"))
        self.max_tokens = _max_tok if _max_tok > 0 else None

        self.client = client or make_client()
        self.session = session
        self._total_usage = TokenUsage()
        self._last_prompt_tokens = 0

        # Metrics
        self._metrics = RLMMetrics()

        self._tool_state = {"summarize": SummarizeState()}

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
        system_prompt = self._load_system_prompt(messages_path, summarize_enabled)

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
            self._last_prompt_tokens = usage.prompt_tokens

            # Record root usage and assistant-visible context for this turn.
            self._metrics.note_root_usage(
                self._total_usage.prompt_tokens,
                self._total_usage.completion_tokens,
            )
            self._metrics.note_assistant_turn(
                usage.prompt_tokens,
                usage.completion_tokens,
            )

            # Update metrics
            self._metrics.turns = turn + 1
            summarize_state = self._tool_state["summarize"]
            self._metrics.turns_since_last_summarize = (
                turn + 1
            ) - summarize_state.turn_at_last_summarize

            msg = response.choices[0].message
            msg_dict = msg.model_dump(exclude_none=True)
            msg_dict.setdefault("content", "")
            messages.append(msg_dict)

            # Log assistant message; parse tool-call args once, reuse below.
            tool_calls_log: list[dict] | None = None
            parsed_args: list[dict | None] = []
            if msg.tool_calls:
                tool_calls_log = []
                for tc in msg.tool_calls:
                    args, err = _parse_tool_call_args(tc.function.arguments)
                    parsed_args.append(args)
                    tool_calls_log.append(
                        {
                            "name": tc.function.name,
                            "args": err if args is None else args,
                        }
                    )
            self.session.log_assistant(turn, tool_calls_log, msg.content)

            if msg.tool_calls and len(msg.tool_calls) > 1:
                self._metrics.stop_reason = "multiple_tool_calls"
                final_text = (
                    "[emitted multiple tool calls in one turn; at most 1 is allowed]"
                )
                break

            # Malformed tool-call arguments: fail the rollout so training
            # penalises the mistake. Final_text + stop_reason make the failure
            # visible to the verifiers harness without raising an exception.
            if msg.tool_calls and parsed_args[0] is None:
                tool_name = msg.tool_calls[0].function.name
                err_info = tool_calls_log[0]["args"]
                self._metrics.stop_reason = "invalid_tool_args"
                final_text = (
                    f"[invalid JSON arguments for tool '{tool_name}': "
                    f"{err_info['_parse_error']}]"
                )
                break

            # Token budget check
            if (
                self.max_tokens
                and self._total_usage.completion_tokens >= self.max_tokens
            ):
                self._metrics.stop_reason = "token_budget"
                final_text = msg.content or "[token budget exhausted]"
                break

            # No tool calls → done
            if not msg.tool_calls:
                self._metrics.stop_reason = "done"
                final_text = msg.content or ""
                break

            # Execute the single allowed tool call (reuse parsed args from above;
            # parse failures have already broken the loop, so parsed_args[0] is
            # guaranteed to be a dict here).
            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            tool_args = parsed_args[0]
            t0 = time.time()
            tool = get_builtin_tool(tool_name)
            if tool is None:
                tool_result = ToolOutcome(content=f"Error: unknown tool '{tool_name}'")
            else:
                tool_result = await asyncio.to_thread(
                    tool.execute, tool_args, self._tool_context(messages)
                )
            duration = time.time() - t0
            for event in tool_result.metric_events:
                self._metrics.record(event)

            result = tool_result.content

            if self.max_output > 0 and len(result) > self.max_output:
                result = result[: self.max_output] + "\n... [output truncated]"
            if tool_result.flush_repl_state:
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
            if tool_result.drop_turns > 0:
                self._drop_turns(messages, tool_result.drop_turns)
            if tool_result.flush_repl_state:
                self._repl.restart_kernel()

            if self.max_turns_in_context is not None:
                turns_in_context = self._count_turns_in_context(messages)
                if turns_in_context > self.max_turns_in_context:
                    self._metrics.stop_reason = "context_limit"
                    n = f"{turns_in_context}/{self.max_turns_in_context}"
                    final_text = f"[context limit exceeded: {n} turns in context]"
                    break
        else:
            self._metrics.stop_reason = "max_turns"
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
            metrics=self._metrics,
        )
        return result

    def _load_system_prompt(self, messages_path: str, summarize_enabled: bool) -> str:
        if self.system_prompt_path:
            return Path(self.system_prompt_path).read_text()
        system_prompt = build_system_prompt(
            self.cwd,
            str(SKILLS_DIR) if SKILLS_DIR is not None else None,
            get_installed_skills(),
            messages_path,
            allow_recursion=self.depth < self.max_depth,
            max_turns_in_context=self.max_turns_in_context,
            summarize_enabled=summarize_enabled,
        )
        if self.append_to_system_prompt:
            system_prompt += "\n\n" + self.append_to_system_prompt
        return system_prompt

    def _detect_new_children(self):
        """Scan session dir for new sub-* directories and log them."""
        if not self.session:
            return
        current = {p.name for p in self.session.dir.glob("sub-*")}
        new = current - self._known_children
        for child_name in sorted(new):
            self.session.log_sub_spawn(child_name, "(spawned via rlm())")
        self._known_children = current

    def _tool_context(self, messages: list[dict]) -> ToolContext:
        return ToolContext(
            messages=messages,
            metrics=self._metrics,
            total_usage=self._total_usage,
            last_prompt_tokens=self._last_prompt_tokens,
            exec_timeout=self.exec_timeout,
            repl=self._repl,
            state=self._tool_state,
        )

    @staticmethod
    def _count_turns_in_context(messages: list[dict]) -> int:
        return sum(1 for message in messages if message["role"] == "assistant")

    @staticmethod
    def _drop_turns(messages: list[dict], num_turns: int) -> None:
        start, end = summarize_drop_slice_bounds(messages, num_turns)
        del messages[start:end]
