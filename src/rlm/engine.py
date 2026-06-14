"""The agent loop."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from pathlib import Path

from openai import AsyncOpenAI, BadRequestError

from rlm.client import call_with_retries, extract_usage, make_client
from rlm.config import Config, parse_summarize_at_tokens, load_config, with_overrides
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


# Wall-clock budget for the teardown drain cell (close_all_registries running in
# the kernel where the child registries live). Bounds how long a wedged child can
# stall a parent's teardown.
DRAIN_TIMEOUT_SECONDS = 120

# Printed by the drain cell on success. The REPL turns a cell timeout / error into
# captured output rather than an exception, so teardown checks for this sentinel to
# tell a completed drain from a wedged one (instead of finalizing as a clean
# shutdown regardless).
DRAIN_OK = "__rlm_drain_ok__"


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
REPL_RESTART_NOTE = (
    "\n\n"
    "Note: the IPython kernel stays running across this compaction. "
    "All variables, imports, and in-memory data are preserved. "
    "Mention important variable names and what they contain so the "
    "next LLM knows what's available."
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

# Injected when an agent resumes from disk after its kernel was restarted: the
# conversation is restored but the IPython REPL is brand new.
KERNEL_RESET_RESUME_WARNING = (
    "WARNING: this agent resumed after its kernel was restarted. Your "
    "conversation is intact, but the IPython session is brand new — every "
    "variable, import, and in-memory object from before is gone. Re-create "
    "anything you need before using it."
)


def is_request_too_large(e: BadRequestError) -> bool:
    """True if a 400 matches the proxy's "Request Entity Too Large" body."""
    haystack = f"{e} {getattr(e, 'body', '') or ''}".lower()
    return "request entity too large" in haystack


def parse_tool_call_args(raw: str) -> tuple[dict | None, dict | None]:
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
        summarize_at_tokens: int | None = None,
        system_prompt_path: str | None = None,
        append_to_system_prompt: str | None = None,
        cwd: str | None = None,
        session: Session | None = None,
        client: AsyncOpenAI | None = None,
        max_tokens: int | None = None,
        config: Config | None = None,
    ):
        # Fold constructor overrides into the env-derived config; an explicit
        # kwarg wins over the corresponding RLM_* variable. summarize_at_tokens
        # is parsed (it accepts int or numeric str) before being applied.
        base = config if config is not None else load_config()
        self.config = with_overrides(
            base,
            model=model,
            system_prompt_path=system_prompt_path,
            append_to_system_prompt=append_to_system_prompt,
            max_tokens=max_tokens,
            summarize_at_tokens=(
                parse_summarize_at_tokens(summarize_at_tokens)
                if summarize_at_tokens is not None
                else None
            ),
        )
        self.model = self.config.model
        self.cwd = cwd or os.getcwd()
        self.exec_timeout = self.config.exec_timeout
        self.max_output = self.config.max_output
        self.summarize_at_tokens = self.config.summarize_at_tokens
        self.system_prompt_path = self.config.system_prompt_path
        self.append_to_system_prompt = self.config.append_to_system_prompt
        self.max_depth = self.config.max_depth
        self.depth = self.config.depth
        # Non-positive budgets mean "no limit" (matching the env path, where
        # RLM_MAX_TOKENS <= 0 -> None), so a stray 0 / negative kwarg can't disable
        # the check by truthiness or stop the agent immediately (B7).
        mt = self.config.max_tokens
        self.max_tokens = mt if mt and mt > 0 else None

        self.client = client or make_client()
        self.session = session
        self._total_usage = TokenUsage()
        self._last_prompt_tokens = 0

        # Metrics
        self._metrics = RLMMetrics()
        self._metrics._sub_rlm_enabled = self.max_depth > 0

        self._tool_state: dict[str, object] = {}

        # IPython REPL (started lazily in single-agent execution)
        self._repl: IPythonREPL | None = None
        self._known_children: set[str] = set()

        # Turn index (0-based) at the start of the current branch. Used to
        # report "turns since last compaction" when a compaction fires.
        self._branch_start_turn: int = 0

        # Resumable-run state (setup / advance / aclose). ``run`` is setup + one
        # advance + aclose; named agents keep these alive across advances.
        self._setup_done = False
        self._depth_exceeded = False
        self._active_tools: list[dict] = []
        self._messages: list[dict] = []
        self._view = 0
        self._turn_offset = 0
        self._final_text = ""

    def _ensure_session(self):
        """Create session if not set."""
        if self.session is not None:
            return
        self.session = Session(self.config.session_dir)

    async def run(self, prompt: str) -> RLMResult:
        """Run a single agent loop to completion (setup + one advance + close)."""
        self.setup()
        try:
            return await self.advance(prompt)
        finally:
            await self.aclose()

    def setup(self) -> None:
        """Prepare the engine for one or more advances.

        Creates the session, starts the IPython kernel (when the ipython tool is
        active), and seeds ``self._messages`` with the system prompt. Idempotent.
        On depth-limit the engine is marked inert: no session/kernel is created,
        ``advance`` returns the depth-limit result, and ``aclose`` is a no-op.
        """
        if self._setup_done or self._depth_exceeded:
            return
        if self.depth > self.max_depth:
            self._depth_exceeded = True
            return

        self._ensure_session()
        if self.depth == 0 and (
            self.config.max_live_agents or self.config.max_running_agents
        ):
            # Root derives the shared marker dir; it propagates to every kernel
            # (via _inject_startup, which reads env) so both caps are tree-global.
            os.environ.setdefault(
                "RLM_LIVE_AGENTS_DIR", str(self.session.dir / ".live_agents")
            )
        self.session.write_meta(
            session_id=self.session.dir.name,
            model=self.model,
            depth=self.depth,
            status="running",
            start_time=time.time(),
            cwd=self.cwd,
        )

        active_builtin_tools = get_active_builtin_tools()
        self._active_tools = [tool.schema() for tool in active_builtin_tools]

        # Start IPython kernel only when the ipython tool is active — otherwise
        # the model can't see or dispatch it, so the startup is pure waste.
        if any(tool.name == "ipython" for tool in active_builtin_tools):
            self._repl = IPythonREPL(cwd=self.cwd, session=self.session)
            self._repl.start()
        self._known_children = {p.name for p in self.session.dir.glob("sub-*")}

        latest_view, prior = self.session.load_latest_view()
        if prior:
            self._resume(latest_view, prior)
        else:
            messages_path = str(self.session.dir / "messages.jsonl")
            system_prompt = self._load_system_prompt(
                messages_path, active_builtin_tools
            )
            self._messages = []
            self._view = 0
            self._record_message({"role": "system", "content": system_prompt}, turn=0)
            self._turn_offset = 0
        self._final_text = ""
        self._setup_done = True

    def _record_message(
        self, message: dict, turn: int, *, duration: float | None = None
    ) -> None:
        """Append a message to the live conversation and persist it to its view."""
        self._messages.append(message)
        self.session.log_message(self._view, turn, message, duration=duration)

    def _write_resume_header(self, turn: int) -> None:
        """Persist resume state to meta.json so a hard restart can continue."""
        self.session.write_meta(
            usage={
                "prompt_tokens": self._total_usage.prompt_tokens,
                "completion_tokens": self._total_usage.completion_tokens,
            },
            turn_offset=turn,
            view=self._view,
            branch_start_turn=self._branch_start_turn,
            metrics_state=self._metrics.snapshot(),
        )

    def _resume(self, view: int, prior_messages: list[dict]) -> None:
        """Continue an agent whose in-memory engine was lost (a kernel restart,
        or a reap + same-name restart).

        Loads the last on-disk view as ``_messages`` and restores the resume
        header from meta.json. Every path here follows a genuine teardown of the
        previous engine's kernel, and ``setup()`` has already started a fresh,
        empty REPL — so when this engine has one, it injects a kernel-reset
        warning that the conversation survived but the REPL is brand new.
        """
        self._messages = list(prior_messages)
        # The loaded content's view is authoritative — it always matches
        # ``_messages``, even if a crash mid-compaction left meta.json's view
        # behind (which would otherwise re-open a branch_reset-closed view, B11).
        self._view = view
        meta_path = self.session.dir / "meta.json"
        try:
            header = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            header = {}
        self._turn_offset = header.get("turn_offset", 0)
        usage = header.get("usage") or {}
        self._total_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        self._branch_start_turn = header.get("branch_start_turn", 0)
        state = header.get("metrics_state")
        if state:
            self._metrics = RLMMetrics.restore(state)
            self._metrics._sub_rlm_enabled = self.max_depth > 0
        # If the saved view ended mid-turn (assistant tool_calls with no results —
        # a crash between the two records), answer them before appending the
        # warning so the resumed sequence is valid (B1).
        self._answer_dangling_tool_calls()
        # The warning is about lost in-memory REPL state, so only inject it when
        # this engine actually has a REPL — a tools-only / chat-only agent has no
        # IPython session to recreate.
        if self._repl is not None:
            self._record_message(
                {"role": "user", "content": KERNEL_RESET_RESUME_WARNING},
                turn=self._turn_offset,
            )

    async def advance(self, prompt: str) -> RLMResult:
        """Append a user turn and run the loop until the model stops calling tools.

        Repeatable: each call continues the same conversation on the same live
        kernel, so a named agent holds a multi-turn conversation across sends.
        """
        if not self._setup_done and not self._depth_exceeded:
            self.setup()
        if self._depth_exceeded:
            return RLMResult(
                answer=f"[depth limit {self.max_depth} reached, cannot start]",
                turns=0,
            )

        if self._turn_offset == 0:
            self.session.write_meta(prompt_preview=prompt[:200])

        # A prior turn can stop (token budget) right after the assistant's
        # tool_calls are recorded but before their results; answer them so this
        # new user turn doesn't make an invalid (400) sequence (B1).
        self._answer_dangling_tool_calls()

        # Alias the instance list so the loop body (and _compact_branch's in-place
        # messages[:] = ...) mutate the persisted conversation directly.
        messages = self._messages
        active_tools = self._active_tools
        self._record_message(
            {"role": "user", "content": prompt}, turn=self._turn_offset
        )

        final_text = ""
        # Role of whatever the engine appended to ``messages`` since the
        # last API call. Drives RLMMetrics.note_assistant_turn's tool-token
        # attribution: only "tool" appendages count toward total_tool_response_tokens.
        # ``None`` on the first turn of an advance (growth is the user turn) and
        # after compaction (fresh branch).
        last_appended_role: str | None = None

        turn = self._turn_offset
        for local_turn in itertools.count():
            turn = self._turn_offset + local_turn
            self._write_resume_header(turn)
            try:
                response = await self._request_completion(messages, active_tools)
            except BadRequestError as e:
                if not is_request_too_large(e):
                    raise
                self._metrics.stop_reason = "request_too_large"
                final_text = "[request body too large]"
                break

            usage = extract_usage(response)
            self._note_turn_usage(usage, turn, last_appended_role)

            msg = response.choices[0].message
            msg_dict = msg.model_dump(exclude_none=True)
            msg_dict.setdefault("content", "")
            self._record_message(msg_dict, turn=turn)

            # Parse each tool call's JSON args once; the error branches reuse them.
            parsed_args, tool_calls_log = self._parse_tool_calls(msg)

            if msg.tool_calls and len(msg.tool_calls) > 1:
                feedback = "Error: only one tool call per turn allowed"
                for tc in msg.tool_calls:
                    self._record_message(
                        {"role": "tool", "tool_call_id": tc.id, "content": feedback},
                        turn=turn,
                    )
                last_appended_role = "tool"
                continue

            if msg.tool_calls and parsed_args[0] is None:
                tc = msg.tool_calls[0]
                tool_name = tc.function.name
                err_info = tool_calls_log[0]["args"]
                feedback = (
                    f"Error: invalid JSON arguments for tool '{tool_name}': "
                    f"{err_info['_parse_error']}"
                )
                self._record_message(
                    {"role": "tool", "tool_call_id": tc.id, "content": feedback},
                    turn=turn,
                )
                last_appended_role = "tool"
                continue

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

            await self._execute_tool_call(
                msg.tool_calls[0], parsed_args[0], messages, turn
            )
            last_appended_role = "tool"

            # Detect new child sessions spawned via rlm()
            self._detect_new_children()

            # Auto-compaction: if this turn's prompt_tokens reached the
            # configured threshold, ask the model for a handoff summary and
            # rebuild the branch around it. Fires at most once per loop
            # iteration; the compaction op takes its own LLM call.
            if (
                self.summarize_at_tokens is not None
                and usage.prompt_tokens >= self.summarize_at_tokens
            ):
                try:
                    await self._compact_branch(messages, turn, active_tools)
                except BadRequestError as e:
                    if not is_request_too_large(e):
                        raise
                    self._metrics.stop_reason = "request_too_large"
                    final_text = "[request body too large]"
                    break
                # Branch boundary: the next API call's prompt is a fresh
                # [system, user(framing+summary)], not a continuation.
                last_appended_role = None

        self._turn_offset = turn + 1
        self._final_text = final_text
        # Persist the final state so a resume after a clean stop continues from
        # the next turn with accurate usage / metrics — the per-turn header at the
        # top of the loop only captured state through the previous turn (B3).
        self._write_resume_header(self._turn_offset)
        return RLMResult(
            answer=final_text,
            session_dir=self.session.dir,
            usage=self._total_usage,
            turns=turn + 1,
        )

    def _answer_dangling_tool_calls(self) -> None:
        """Answer any unanswered tool calls left at the tail of the transcript.

        A turn can be cut off after the assistant's ``tool_calls`` are recorded
        but before the tool results are — a token-budget stop on a tool-call
        turn, or a hard crash mid-exec that a later resume reloads. Appending the
        next user turn on top of that is an invalid OpenAI sequence (a 400), so
        synthesize a tool result for each unanswered call first.
        """
        if not self._messages:
            return
        last = self._messages[-1]
        if last.get("role") != "assistant":
            return
        for tc in last.get("tool_calls") or []:
            self._record_message(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "[interrupted: no tool result was recorded]",
                },
                turn=self._turn_offset,
            )

    async def _request_completion(self, messages: list[dict], active_tools: list[dict]):
        """One chat-completion call for the current turn (with retries)."""
        request_kwargs: dict = {"model": self.model, "messages": messages}
        if active_tools:
            request_kwargs["tools"] = active_tools
            request_kwargs["parallel_tool_calls"] = False
        return await call_with_retries(
            self.client.chat.completions.create, **request_kwargs
        )

    def _note_turn_usage(
        self, usage: TokenUsage, turn: int, last_appended_role: str | None
    ) -> None:
        """Fold one turn's token usage into the running totals and metrics."""
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        self._last_prompt_tokens = usage.prompt_tokens
        self._metrics.note_root_usage(
            self._total_usage.prompt_tokens, self._total_usage.completion_tokens
        )
        self._metrics.note_assistant_turn(
            usage.prompt_tokens,
            usage.completion_tokens,
            prev_appended_role=last_appended_role,
        )
        self._metrics.turns = turn + 1
        self._metrics.turns_since_last_compaction = turn + 1 - self._branch_start_turn

    def _parse_tool_calls(self, msg) -> tuple[list[dict | None], list[dict] | None]:
        """Parse each tool call's JSON arguments once.

        Returns ``(parsed_args, tool_calls_log)`` aligned with ``msg.tool_calls``,
        or ``([], None)`` when the turn made no tool calls. ``parsed_args[i]`` is
        ``None`` when call ``i``'s arguments were invalid JSON; the matching log
        entry then carries the parse error.
        """
        if not msg.tool_calls:
            return [], None
        parsed_args: list[dict | None] = []
        tool_calls_log: list[dict] = []
        for tc in msg.tool_calls:
            args, err = parse_tool_call_args(tc.function.arguments)
            parsed_args.append(args)
            tool_calls_log.append(
                {"name": tc.function.name, "args": err if args is None else args}
            )
        return parsed_args, tool_calls_log

    async def _execute_tool_call(
        self, tc, tool_args: dict | None, messages: list[dict], turn: int
    ) -> None:
        """Dispatch one tool call and record its (possibly truncated) result."""
        tool_name = tc.function.name
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
        self._record_message(
            {"role": "tool", "tool_call_id": tc.id, "content": result},
            turn=turn,
            duration=duration,
        )

    async def aclose(self) -> None:
        """Finalize the session and shut the kernel down.

        Idempotent. A no-op if the engine never set up (e.g. depth-limited). If
        setup() failed partway after starting the kernel, the kernel is still shut
        down — finalize is skipped, since there is no completed run to finalize.
        """
        if not self._setup_done and self._repl is None:
            return
        finalize = self._setup_done
        self._setup_done = False
        try:
            if finalize:
                # Drain background sub-agents by running a cell in this engine's
                # kernel (where their registries live) so each finalizes its
                # session — and cascade-drains its own grandchildren — before we
                # aggregate child metrics and shut the kernel down. Best-effort: a
                # wedged child must not block the parent from finalizing.
                if self._repl is not None and self.max_depth > 0:
                    if not await self._drain_sub_agents():
                        logging.getLogger(__name__).warning(
                            "sub-agent drain did not complete within %ss during "
                            "teardown; descendant kernels/sessions may remain open",
                            DRAIN_TIMEOUT_SECONDS,
                        )
                        # So finalize's metrics don't read as a clean shutdown.
                        self.session.write_meta(teardown_drain_complete=False)
                self.session.finalize(
                    self._final_text,
                    usage={
                        "prompt_tokens": self._total_usage.prompt_tokens,
                        "completion_tokens": self._total_usage.completion_tokens,
                    },
                    turns=self._turn_offset,
                    metrics=self._metrics,
                )
        finally:
            if self._repl is not None:
                await asyncio.to_thread(self._repl.shutdown)
                self._repl = None

    async def _drain_sub_agents(self) -> bool:
        """Drain this kernel's background sub-agents; ``True`` if it completed.

        Runs a cell that closes the child registries — each child finalizes its
        session and cascade-drains its own grandchildren — off the event loop, so
        the blocking drain doesn't stall this agent's loop (and its siblings).

        The REPL turns a cell timeout / error into captured output rather than an
        exception, so a wedged or deep tree would otherwise look like a clean
        shutdown. The cell prints ``DRAIN_OK`` only after the drain returns, so its
        presence in the output distinguishes a completed drain from a timed-out one.
        """
        try:
            out = await asyncio.to_thread(
                self._repl.execute,
                f"import rlm.api as _rlm; _rlm.drain_agents(); print({DRAIN_OK!r})",
                timeout=DRAIN_TIMEOUT_SECONDS,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "sub-agent drain raised during teardown", exc_info=True
            )
            return False
        return DRAIN_OK in out

    async def _compact_branch(
        self, messages: list[dict], turn: int, active_tools: list[dict]
    ) -> None:
        """Ask the model for a handoff summary and rebuild ``messages``.

        Called in-place: mutates ``messages`` to ``[system, user(framing +
        summary)]`` and restarts the ipython kernel. The LLM call for the
        summary is housekeeping, not a work turn, but its tokens land in
        ``_total_usage`` for cost accounting.

        ``active_tools`` is forwarded as ``tools=`` with
        ``tool_choice="none"`` so the rendered system prompt matches
        regular turns (vLLM's chat-completions layer injects the tools
        block into the system message only when ``tools=`` is set). With
        a matching system prompt, prime-rl's RL trajectory walker keeps
        the extension property across the compaction boundary instead
        of opening an extra training-sample split. ``tool_choice="none"``
        keeps the original "text-only summary" behaviour by forbidding
        tool calls on this turn.
        """
        # Measure what's about to be dropped BEFORE appending the
        # checkpoint prompt — otherwise the prompt's own chars get
        # counted as "dropped conversation content", inflating the
        # metric and the session log's dropped_chars field.
        dropped_chars = count_messages_chars(messages[2:])
        turns_since_last = turn + 1 - self._branch_start_turn

        # Append the checkpoint prompt and ask the model for a text-only
        # summary turn. Tools are advertised to the server (so the system
        # prompt renders identically to regular turns) but
        # ``tool_choice="none"`` forbids the model from calling any.
        # Warn about the REPL restart only when a kernel is actually running.
        checkpoint_prompt = CHECKPOINT_COMPACTION_PROMPT
        if self._repl is not None:
            checkpoint_prompt += REPL_RESTART_NOTE
        self._record_message({"role": "user", "content": checkpoint_prompt}, turn=turn)
        request_kwargs: dict = {"model": self.model, "messages": messages}
        if active_tools:
            request_kwargs["tools"] = active_tools
            request_kwargs["tool_choice"] = "none"
        response = await call_with_retries(
            self.client.chat.completions.create,
            **request_kwargs,
        )
        usage = extract_usage(response)
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens

        summary_text = response.choices[0].message.content or ""

        # The model produced the summary as an assistant turn — record it in the
        # closing view (it isn't kept in live _messages, which is rebuilt below).
        self.session.log_message(
            self._view, turn, {"role": "assistant", "content": summary_text}
        )
        # Close this branch and open the next view.
        self.session.branch_reset(
            self._view,
            dropped_chars=dropped_chars,
            summary_chars=len(summary_text),
            turns_since=turns_since_last,
        )
        self._view += 1
        system_msg = messages[0]
        compacted_user_content = POST_COMPACTION_FRAMING + "\n\n" + summary_text
        messages[:] = []
        self._record_message(system_msg, turn=turn)
        self._record_message(
            {"role": "user", "content": compacted_user_content}, turn=turn
        )

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
            self.session.log_spawn(child_name)
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


def count_messages_chars(messages: list[dict]) -> int:
    """Sum the content-char length across ``messages`` (text + tool-call args).

    Used as a rough "how much was dropped" metric on compaction. Tool-call
    argument strings are counted since they consume context just like
    message content does.
    """
    total = 0
    for message in messages:
        total += content_chars(message.get("content"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                total += tool_call_chars(tc)
    return total


def content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(content_chars(item) for item in content)
    if isinstance(content, dict):
        total = 0
        for field_name in ("text", "input_text", "output_text"):
            value = content.get(field_name)
            if isinstance(value, str):
                total += len(value)
        nested = content.get("content")
        if nested is not None:
            total += content_chars(nested)
        return total
    return 0


def tool_call_chars(tool_call) -> int:
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
