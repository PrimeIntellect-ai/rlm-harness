"""The agent loop. Standard tool-calling with bash + edit."""

from __future__ import annotations

import json
import os
import time

from openai import AsyncOpenAI

from rlm.client import extract_usage, make_client
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.tools import get_active_tools, run_bash, run_edit, run_websearch
from rlm.types import RLMResult, TokenUsage


class RLMEngine:
    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        cwd: str | None = None,
        session: Session | None = None,
        client: AsyncOpenAI | None = None,
        tools: list[str] | None = None,
    ):
        self.model = model or os.environ.get("RLM_MODEL", "gpt-4o")
        self.max_turns = max_turns or int(os.environ.get("RLM_MAX_TURNS", "30"))
        self.cwd = cwd or os.getcwd()
        self.bash_timeout = int(os.environ.get("RLM_BASH_TIMEOUT", "120"))
        self.max_output = int(os.environ.get("RLM_MAX_OUTPUT", "8192"))
        self.max_depth = int(os.environ.get("RLM_MAX_DEPTH", "3"))
        self.depth = int(os.environ.get("RLM_DEPTH", "0"))

        # Tools
        if tools is not None:
            self.allowed_tools = tools
        else:
            self.allowed_tools = os.environ.get("RLM_TOOLS", "bash,edit,websearch").split(",")

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
            tools=self.allowed_tools,
        )

        active_tools = get_active_tools(self.allowed_tools)
        system_prompt = build_system_prompt(self.allowed_tools, self.cwd)

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
            messages.append(msg.model_dump(exclude_none=True))

            # Log assistant message
            tool_calls_log = None
            if msg.tool_calls:
                tool_calls_log = [
                    {"name": tc.function.name, "args": json.loads(tc.function.arguments)}
                    for tc in msg.tool_calls
                ]
            self.session.log_assistant(turn, tool_calls_log, msg.content)

            # Token budget check
            if self.max_tokens and self._total_usage.completion_tokens >= self.max_tokens:
                final_text = msg.content or "[token budget exhausted]"
                break

            # No tool calls → done
            if not msg.tool_calls:
                final_text = msg.content or ""
                break

            # Execute tool calls
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                t0 = time.time()
                result = self._execute_tool(name, args)
                duration = time.time() - t0

                # Append turn/budget info
                budget_parts = [f"{turn + 1}/{self.max_turns} turns"]
                if self.max_tokens:
                    budget_parts.append(
                        f"{self._total_usage.completion_tokens}/{self.max_tokens} completion tokens"
                    )
                result += f"\n[{', '.join(budget_parts)} used]"

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

                self.session.log_tool_result(turn, name, result, duration)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
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
            usage={"prompt_tokens": self._total_usage.prompt_tokens, "completion_tokens": self._total_usage.completion_tokens},
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
            sub_tools = env.get("RLM_SUB_TOOLS")
            if sub_tools:
                env["RLM_TOOLS"] = sub_tools

            proc = await asyncio.create_subprocess_exec(
                "rlm", prompt,
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

            return RLMResult(answer=answer, session_dir=child_dir, usage=usage, turns=turns)

        return await asyncio.gather(*[_run_one(p) for p in prompts])

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "bash":
            return run_bash(
                args["command"],
                cwd=self.cwd,
                session=self.session,
                timeout=self.bash_timeout,
                max_output=self.max_output,
            )
        elif name == "edit":
            return run_edit(
                args["path"],
                args["old_str"],
                args["new_str"],
                cwd=self.cwd,
            )
        elif name == "websearch":
            return run_websearch(
                args["queries"],
                max_output=self.max_output,
            )
        else:
            return f"Error: unknown tool '{name}'"
