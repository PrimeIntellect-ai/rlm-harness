"""The agent loop."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path

from openai import AsyncOpenAI

from rlm.client import extract_usage, make_client
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.tools import (
    SKILLS_DIR,
    BuiltinTool,
    IPythonREPL,
    ToolContext,
    ToolOutcome,
    get_active_builtin_tools,
    get_builtin_tool,
    get_installed_skills,
)
from rlm.types import CompactionApplied, RLMMetrics, RLMResult, TokenUsage


# Injected as a user message when the branch's context size reaches the
# sampled compaction threshold. The model's next reply is expected to be
# a plain-text handoff summary; any tool calls it emits are ignored and
# the message is compacted in place of them.
CHECKPOINT_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. "
    "Create a handoff summary for another LLM that will resume the task.\n"
    "\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n"
    "\n"
    "Be concise, structured, and focused on helping the next LLM "
    "seamlessly continue the work."
)

# Wrapper text that frames the summary as the sole user-facing context
# for the post-compaction branch. The original task prompt is dropped;
# the summary is responsible for carrying the goal.
POST_COMPACTION_FRAMING = (
    "Another language model started to solve this problem and produced "
    "a summary of its thinking process. You also have access to the "
    "state of the tools that were used by that language model. Use this "
    "to build on the work that has already been done and avoid "
    "duplicating work. Here is the summary produced by the other "
    "language model, use the information in this summary to assist with "
    "your own analysis:"
)


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


def _parse_summarize_at_tokens(
    value: int | tuple[int, int] | list[int] | str | None,
) -> tuple[int, int] | None:
    """Normalize ``summarize_at_tokens`` to ``(lo, hi)`` or ``None``.

    Accepts:
      - ``None`` → disabled.
      - ``int`` → fixed threshold (lo == hi).
      - ``(lo, hi)`` / ``[lo, hi]`` → uniform range per compaction.
      - ``str`` (from env var) → "N" or "lo,hi".
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return None
        try:
            ints = [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(
                f"summarize_at_tokens must be an int or 'lo,hi' string (got {value!r})"
            ) from exc
        if len(ints) == 1:
            lo = hi = ints[0]
        elif len(ints) == 2:
            lo, hi = ints
        else:
            raise ValueError(
                f"summarize_at_tokens string must have 1 or 2 comma-separated "
                f"ints (got {value!r})"
            )
    elif isinstance(value, bool):
        raise ValueError("summarize_at_tokens must be an int or (lo, hi) tuple")
    elif isinstance(value, int):
        lo = hi = value
    elif isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"summarize_at_tokens tuple must have 2 elements (got {value!r})"
            )
        lo, hi = int(value[0]), int(value[1])
    else:
        raise ValueError(
            f"summarize_at_tokens must be int, (lo, hi), or None (got {type(value).__name__})"
        )

    if lo <= 0 or hi <= 0:
        raise ValueError(
            f"summarize_at_tokens values must be positive (got lo={lo}, hi={hi})"
        )
    if lo > hi:
        raise ValueError(f"summarize_at_tokens lo must be <= hi (got lo={lo}, hi={hi})")
    return lo, hi


class RLMEngine:
    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        summarize_at_tokens: int | tuple[int, int] | list[int] | None = None,
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

        # Auto-compaction threshold: user kwarg wins; otherwise parse env var.
        if summarize_at_tokens is None:
            env_value = os.environ.get("RLM_SUMMARIZE_AT_TOKENS")
        else:
            env_value = summarize_at_tokens
        self.summarize_at_tokens_range = _parse_summarize_at_tokens(env_value)

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

        self._tool_state: dict[str, object] = {}

        # IPython REPL (started lazily in single-agent execution)
        self._repl: IPythonREPL | None = None
        self._known_children: set[str] = set()

        # Drawn at branch start and after every compaction. None if disabled.
        self._current_summarize_threshold: int | None = self._sample_threshold()

        # Turn index (0-based) at the start of the current branch. Used to
        # report "turns since last compaction" when a compaction fires.
        self._branch_start_turn: int = 0

    def _sample_threshold(self) -> int | None:
        """Draw a new summarization threshold from the configured range."""
        if self.summarize_at_tokens_range is None:
            return None
        lo, hi = self.summarize_at_tokens_range
        return random.randint(lo, hi)

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

        # Start IPython kernel only when the ipython tool is active —
        # otherwise the model can't see or dispatch it, so the kernel
        # startup (subprocess + injection) is pure waste.
        if any(tool.name == "ipython" for tool in get_active_builtin_tools()):
            self._repl = IPythonREPL(cwd=self.cwd, session=self.session)
            self._repl.start()
        self._known_children = {p.name for p in self.session.dir.glob("sub-*")}

        try:
            return await self._run_loop(prompt)
        finally:
            if self._repl is not None:
                self._repl.shutdown()

    async def _run_loop(self, prompt: str) -> RLMResult:
        active_builtin_tools = get_active_builtin_tools()
        active_tools = [tool.schema() for tool in active_builtin_tools]
        messages_path = str(self.session.dir / "messages.jsonl")
        system_prompt = self._load_system_prompt(messages_path, active_builtin_tools)

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
            self._metrics.turns_since_last_compaction = (
                turn + 1 - self._branch_start_turn
            )

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

            # Auto-compaction: if this turn's prompt_tokens reached the
            # sampled threshold, ask the model for a handoff summary and
            # rebuild the branch around it. Fires at most once per loop
            # iteration; the compaction op takes its own LLM call.
            if (
                self._current_summarize_threshold is not None
                and usage.prompt_tokens >= self._current_summarize_threshold
            ):
                await self._compact_branch(messages, turn)
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

    async def _compact_branch(self, messages: list[dict], turn: int) -> None:
        """Ask the model for a handoff summary and rebuild ``messages``.

        Called in-place: mutates ``messages`` to ``[system, user(framing +
        summary)]``, restarts the ipython kernel, and resamples the
        threshold. The LLM call for the summary doesn't count toward
        ``max_turns`` — it's housekeeping, not a work turn — but its
        tokens land in ``_total_usage`` for cost accounting.
        """
        # Measure what's about to be dropped BEFORE appending the
        # checkpoint prompt — otherwise the prompt's own chars get
        # counted as "dropped conversation content", inflating the
        # metric and the session log's dropped_chars field.
        dropped_chars = _count_messages_chars(messages[2:])
        turns_since_last = turn + 1 - self._branch_start_turn

        # Append the checkpoint prompt and ask the model for a summary
        # turn with NO tools available so it can only respond with text.
        messages.append({"role": "user", "content": CHECKPOINT_COMPACTION_PROMPT})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        usage = extract_usage(response)
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens

        summary_text = response.choices[0].message.content or ""

        system_msg = messages[0]
        compacted_user_content = POST_COMPACTION_FRAMING + "\n\n" + summary_text
        messages[:] = [
            system_msg,
            {"role": "user", "content": compacted_user_content},
        ]

        # Log the compaction for traceability.
        self.session.log(
            {
                "type": "compaction",
                "turn": turn,
                "summary": summary_text,
                "summary_chars": len(summary_text),
                "dropped_chars": dropped_chars,
                "turns_since_last_compaction": turns_since_last,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            }
        )

        # Reset the persistent ipython kernel: the new "next LLM" shouldn't
        # inherit live Python state it can't see in its context.
        if self._repl is not None:
            self._repl.restart_kernel()

        # Metrics: close the old branch and draw a fresh threshold.
        self._metrics.record(
            CompactionApplied(
                num_turns_dropped=turns_since_last,
                dropped_chars=dropped_chars,
                summary_chars=len(summary_text),
                turns_since_last_compaction=turns_since_last,
            )
        )
        self._branch_start_turn = turn + 1
        self._metrics.turns_since_last_compaction = 0
        self._current_summarize_threshold = self._sample_threshold()

    def _load_system_prompt(
        self, messages_path: str, active_tools: list[BuiltinTool]
    ) -> str:
        if self.system_prompt_path:
            return Path(self.system_prompt_path).read_text()
        system_prompt = build_system_prompt(
            self.cwd,
            str(SKILLS_DIR) if SKILLS_DIR is not None else None,
            get_installed_skills(),
            messages_path,
            allow_recursion=self.depth < self.max_depth,
            active_tools=active_tools,
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
            cwd=self.cwd,
        )


def _count_messages_chars(messages: list[dict]) -> int:
    """Sum the content-char length across ``messages`` (text + tool-call args).

    Used as a rough "how much was dropped" metric on compaction. Tool-call
    argument strings are counted since they consume context just like
    message content does.
    """
    total = 0
    for message in messages:
        total += _content_chars(message.get("content"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                total += _tool_call_chars(tc)
    return total


def _content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_chars(item) for item in content)
    if isinstance(content, dict):
        total = 0
        for field_name in ("text", "input_text", "output_text"):
            value = content.get(field_name)
            if isinstance(value, str):
                total += len(value)
        nested = content.get("content")
        if nested is not None:
            total += _content_chars(nested)
        return total
    return 0


def _tool_call_chars(tool_call) -> int:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
    else:
        function = getattr(tool_call, "function", None)
    if function is None:
        return 0
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
    total = 0
    if isinstance(name, str):
        total += len(name)
    if isinstance(arguments, str):
        total += len(arguments)
    return total
