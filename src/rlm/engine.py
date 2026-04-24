"""The agent loop."""

from __future__ import annotations

import asyncio
import json
import os
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
# compaction threshold. The model's next reply is expected to be a
# plain-text handoff summary; any tool calls it emits are ignored and
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

# Appended to the checkpoint prompt when the IPython REPL is active.
# The kernel is restarted on compaction, so the next LLM inherits an
# empty Python state — flag this so the summary captures anything the
# next LLM would otherwise need to rebuild from scratch.
REPL_RESTART_NOTE = (
    "\n\n"
    "Note: the IPython kernel will be restarted after this summary. "
    "All Python variables, imports, loaded data, and in-memory state "
    "will be wiped. Capture anything the next LLM needs to re-establish "
    "(key values, file paths, loaded datasets, etc.) in the summary itself."
)

# Wrapper text that frames the summary as the sole user-facing context
# for the post-compaction branch. The original task prompt is dropped;
# the summary is responsible for carrying the goal.
POST_COMPACTION_FRAMING = (
    "Another language model started to solve this problem and produced "
    "a summary of its thinking process. Use this to build on the work "
    "that has already been done and avoid duplicating work. Here is "
    "the summary produced by the other language model, use the "
    "information in this summary to assist with your own analysis:"
)

# Replaces a tool result whose presence in context pushed prompt_tokens
# past max_context_tokens. The tool WAS executed — real-world side
# effects (filesystem writes, web calls) already happened — but its
# output is being dropped so the rolled-back conversation fits in the
# remaining budget.
OVERSHOT_TOOL_RESULT_STUB = (
    "[tool output dropped: the result would have pushed the conversation "
    "past the context-token budget. The tool ran but its output is not "
    "visible.]"
)

# Written in place of a real tool result when soft compaction fires on
# a turn whose assistant response had a tool_call. The tool was NOT
# executed; we're compacting before running it. Preserves the model's
# intent in the summary ("I was about to call X; my call was deferred").
SOFT_COMPACT_SKIPPED_STUB = (
    "[tool call skipped: the context-compaction threshold was reached "
    "before this tool ran. It was not executed.]"
)

# Safety cushion subtracted from remaining-budget when computing
# max_completion_tokens. Accounts for small provider bookkeeping drift
# between our prompt_tokens accounting and whatever the server measures.
_BUDGET_MARGIN_TOKENS = 128


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


def _parse_positive_int(name: str, value: int | str | None) -> int | None:
    """Normalize a positive-int config value to ``int`` or ``None``.

    Accepts ``None`` / empty string → disabled, ``int`` → direct, ``str``
    (from env var) → parsed. ``bool`` is rejected (``True`` is technically
    an int, but passing it here is almost certainly a bug).
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an int (got {value!r})") from exc
    elif isinstance(value, int):
        parsed = value
    else:
        raise ValueError(f"{name} must be int or None (got {type(value).__name__})")

    if parsed <= 0:
        raise ValueError(f"{name} must be positive (got {parsed})")
    return parsed


def _parse_summarize_at_tokens(value: int | str | None) -> int | None:
    """Normalize ``summarize_at_tokens`` to a positive int or ``None``."""
    return _parse_positive_int("summarize_at_tokens", value)


def _parse_max_context_tokens(value: int | str | None) -> int | None:
    """Normalize ``max_context_tokens`` to a positive int or ``None``."""
    return _parse_positive_int("max_context_tokens", value)


class RLMEngine:
    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        summarize_at_tokens: int | None = None,
        max_context_tokens: int | None = None,
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
        self.summarize_at_tokens = _parse_summarize_at_tokens(env_value)

        # Hard context ceiling: caps per-call max_completion_tokens and
        # triggers tool-result rollback when a response's prompt_tokens
        # overshoots it. User kwarg wins; otherwise parse env var.
        if max_context_tokens is None:
            env_value = os.environ.get("RLM_MAX_CONTEXT_TOKENS")
        else:
            env_value = max_context_tokens
        self.max_context_tokens = _parse_max_context_tokens(env_value)

        # Invariant: the soft compaction trigger must be strictly below
        # the hard ceiling, or compaction would fire at the ceiling with
        # no room to generate a summary.
        if (
            self.summarize_at_tokens is not None
            and self.max_context_tokens is not None
            and self.summarize_at_tokens >= self.max_context_tokens
        ):
            raise ValueError(
                f"summarize_at_tokens ({self.summarize_at_tokens}) must be "
                f"< max_context_tokens ({self.max_context_tokens})"
            )

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

        # Turn index (0-based) at the start of the current branch. Used to
        # report "turns since last compaction" when a compaction fires.
        self._branch_start_turn: int = 0

    def _ensure_session(self):
        """Create session if not set."""
        if self.session is not None:
            return
        session_dir = os.environ.get("RLM_SESSION_DIR")
        self.session = Session(session_dir)

    def _completion_budget_kwargs(self) -> dict:
        """Extra kwargs for ``chat.completions.create`` to cap generation.

        When ``max_context_tokens`` is set, passes ``max_completion_tokens``
        computed from the remaining budget so the provider enforces the
        cap server-side. The floor is the provider's default (we just
        don't pass the kwarg when the remaining budget goes non-positive);
        the post-response overshoot check is responsible for recovery if
        the model or a tool result still overflows.
        """
        if self.max_context_tokens is None:
            return {}
        remaining = (
            self.max_context_tokens - self._last_prompt_tokens - _BUDGET_MARGIN_TOKENS
        )
        if remaining <= 0:
            return {}
        return {"max_completion_tokens": remaining}

    def _is_overshoot(self, prompt_tokens: int) -> bool:
        if self.max_context_tokens is None:
            return False
        return prompt_tokens > self.max_context_tokens

    @staticmethod
    def _rollback_last_tool_result(messages: list[dict]) -> bool:
        """Replace the most recent non-stub tool result with the overshoot stub.

        Returns True if a rollback was performed, False if no eligible
        ``role="tool"`` message was found — in which case the caller
        should treat the situation as unrecoverable (system + initial
        user prompt alone overflowed the ceiling, or we've already
        rolled back every tool result to the stub).
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if (
                msg.get("role") == "tool"
                and msg.get("content") != OVERSHOT_TOOL_RESULT_STUB
            ):
                messages[i] = {**msg, "content": OVERSHOT_TOOL_RESULT_STUB}
                return True
        return False

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
                **self._completion_budget_kwargs(),
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

            # Overshoot: the request we just sent exceeded max_context_tokens.
            # The usual culprit is a fat tool result that landed in messages
            # between the previous turn and this one. Roll it back to a short
            # stub, fire compaction on the cleaned messages, and discard this
            # turn's response (its reasoning references the tool_result we
            # just replaced, and its tool_call — if any — would dangle).
            if self._is_overshoot(usage.prompt_tokens):
                rolled_back = self._rollback_last_tool_result(messages)
                if not rolled_back:
                    self._metrics.stop_reason = "context_budget_exceeded"
                    final_text = (
                        f"[context budget exceeded "
                        f"({usage.prompt_tokens} > {self.max_context_tokens}) "
                        f"with no tool result to roll back]"
                    )
                    break
                self._metrics.note_overshoot_rollback()
                self.session.log(
                    {
                        "type": "overshoot_rollback",
                        "turn": turn,
                        "prompt_tokens": usage.prompt_tokens,
                        "max_context_tokens": self.max_context_tokens,
                    }
                )
                compact_ok = await self._compact_branch(messages, turn)
                if not compact_ok:
                    final_text = (
                        f"[compaction call exceeded max_context_tokens "
                        f"({self.max_context_tokens}); "
                        f"initial prompt may be too large]"
                    )
                    break
                continue

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

            # Soft compaction: the model wants to continue (it has a
            # tool_call) but this turn's prompt_tokens reached the
            # threshold. Don't execute the tool; instead append a skip
            # stub as the tool result so the model's intent is
            # preserved in what compaction summarizes, then compact.
            # Checks here — before tool execution — so the compaction
            # call runs on messages that are still ≈ prompt_tokens + a
            # small stub, guaranteeing it fits under the ceiling. Done
            # responses (above) skip this path, so a final-answer turn
            # is never wasted by soft compaction.
            if (
                self.summarize_at_tokens is not None
                and usage.prompt_tokens >= self.summarize_at_tokens
            ):
                tc = msg.tool_calls[0]
                self.session.log_tool_result(
                    turn, tc.function.name, SOFT_COMPACT_SKIPPED_STUB, 0.0
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": SOFT_COMPACT_SKIPPED_STUB,
                    }
                )
                compact_ok = await self._compact_branch(messages, turn)
                if not compact_ok:
                    final_text = (
                        f"[compaction call exceeded max_context_tokens "
                        f"({self.max_context_tokens})]"
                    )
                    break
                continue

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

    async def _compact_branch(self, messages: list[dict], turn: int) -> bool:
        """Ask the model for a handoff summary and rebuild ``messages``.

        Called in-place: mutates ``messages`` to ``[system, user(framing +
        summary)]`` and restarts the ipython kernel. The LLM call for the
        summary doesn't count toward ``max_turns`` — it's housekeeping,
        not a work turn — but its tokens land in ``_total_usage`` for
        cost accounting.

        Callers run this on ``messages`` that should already fit under
        the ceiling:

        - Hard path: the triggering response was discarded and the
          fat tool result is already rolled back to the overshot stub.
        - Soft path: the check fires on pre-append ``usage.prompt_tokens``
          that's below the ceiling; the just-appended assistant + skip
          stub add only a couple hundred tokens.

        So the compaction call normally fits. Returns ``False`` (and
        sets ``stop_reason = "context_budget_exceeded"``) only in the
        pathological case where it overshoots anyway — e.g. system +
        initial user prompt alone exceed the ceiling. Caller should
        break out of the run loop.
        """
        # Measure what's about to be dropped BEFORE appending the
        # checkpoint prompt — otherwise the prompt's own chars get
        # counted as "dropped conversation content", inflating the
        # metric and the session log's dropped_chars field.
        dropped_chars = _count_messages_chars(messages[2:])
        turns_since_last = turn + 1 - self._branch_start_turn

        # Append the checkpoint prompt and ask the model for a summary
        # turn with NO tools available so it can only respond with text.
        # Warn about the REPL restart only when a kernel is actually running.
        checkpoint_prompt = CHECKPOINT_COMPACTION_PROMPT
        if self._repl is not None:
            checkpoint_prompt += REPL_RESTART_NOTE
        messages.append({"role": "user", "content": checkpoint_prompt})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._completion_budget_kwargs(),
        )
        usage = extract_usage(response)
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        self._last_prompt_tokens = usage.prompt_tokens

        if self._is_overshoot(usage.prompt_tokens):
            self._metrics.stop_reason = "context_budget_exceeded"
            self.session.log(
                {
                    "type": "compaction_overshot",
                    "turn": turn,
                    "prompt_tokens": usage.prompt_tokens,
                    "max_context_tokens": self.max_context_tokens,
                }
            )
            return False

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

        # Metrics: close the old branch.
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
        return True

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
