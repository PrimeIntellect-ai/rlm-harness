"""Shared fixtures and dummy-LLM scaffolding for the test suite."""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from rlm.session import Session
from rlm.tools import registry as tool_registry
from rlm.tools.base import ToolContext, ToolOutcome

SKILL_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "skills"

# --- Dummy tool ----------------------------------------------------------


class AddTool:
    name = "add"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        return ToolOutcome(content=str(args["a"] + args["b"]))


# --- Dummy message types --------------------------------------
#
# Use these to build up the conversation a DummyClient will replay. The
# engine only reads ``response.choices[0].message`` + ``response.usage``,
# so mocking that surface is enough.


@dataclass
class DummyFn:
    name: str
    arguments: str


@dataclass
class DummyToolCall:
    """A scripted tool call inside a DummyMessage.

    ``arguments`` may be a dict (auto-serialized to JSON) or a raw string
    (useful for scripting malformed-JSON cases).
    """

    name: str
    arguments: str | dict
    id: str = "call_0"
    type: str = "function"

    def __post_init__(self) -> None:
        if isinstance(self.arguments, dict):
            self.arguments = json.dumps(self.arguments)

    @property
    def function(self) -> DummyFn:
        # __post_init__ normalizes arguments to str.
        return DummyFn(name=self.name, arguments=cast(str, self.arguments))


@dataclass
class DummyMessage:
    """A scripted assistant turn."""

    content: str | None = None
    tool_calls: list[DummyToolCall] | None = None
    role: str = "assistant"

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            out = {k: v for k, v in out.items() if v is not None}
        return out


# --- Dummy client -------------------------------------------


@dataclass
class DummyChoice:
    message: DummyMessage


@dataclass
class DummyUsage:
    prompt_tokens: int = 1
    completion_tokens: int = 1


@dataclass
class DummyResponse:
    choices: list[DummyChoice]
    usage: DummyUsage = field(default_factory=DummyUsage)


class DummyClient:
    """Replays scripted DummyMessages, one per ``chat.completions.create`` call."""

    def __init__(self, messages: list[DummyMessage]):
        self.scripted = list(messages)
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> DummyResponse:
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("DummyClient exhausted: no more scripted messages")
        return DummyResponse(choices=[DummyChoice(message=self.scripted.pop(0))])


def tool_result(client: DummyClient, turn: int = 0) -> str:
    """Return the content of the ``turn``-th ``tool`` message sent to the model.

    Tool results appear on the next LLM request after their tool call, so
    the default ``turn=0`` reads the result produced by the first tool
    call (visible in ``client.calls[1]["messages"]``).
    """
    request_messages = client.calls[turn + 1]["messages"]
    tool_messages = [m for m in request_messages if m.get("role") == "tool"]
    return tool_messages[turn]["content"]


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def register_add_ntc(monkeypatch):
    """Register the in-process AddTool as the only active native tool."""
    monkeypatch.setitem(tool_registry._TOOLS_BY_NAME, "add", AddTool())
    monkeypatch.setenv("RLM_TOOLS", "add")


@pytest.fixture
def register_add_ptc(install_skill):
    """Install the add skill fixture so ``await add(...)`` / ``!add ...`` resolves."""
    install_skill("add")


@pytest.fixture
def session(tmp_path):
    return Session(tmp_path / "session")


# --- Real-skill scaffolding ---------------------------------------------
#
# Lets tests exercise the IPython tool end-to-end: the engine starts its
# own kernel (ipython is in the default RLM_TOOLS), install_shims picks
# up the copied skill at startup, and ``await <skill>(...)`` /
# ``!<skill> ...`` code in scripted ipython-tool calls hits the real CLI
# on PATH. Skill packages live under ``tests/fixtures/skills/<name>/``
# and mirror the production rlm-skill layout.


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    """Fresh skills dir plus a PATH-hooked bin dir.

    Monkeypatches ``rlm.tools.ipython.TASK_SKILLS_DIR`` so the kernel-shim
    startup code generated by ``_inject_startup`` points at this dir.
    """
    d = tmp_path / "skills"
    d.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr("rlm.tools.ipython.TASK_SKILLS_DIR", d)
    return d


@pytest.fixture
def install_skill(tmp_path, skills_dir):
    """Factory: copy a fixture skill into skills_dir + drop a CLI on PATH.

    Skills live under ``tests/fixtures/skills/<name>/`` with the standard
    rlm-skill layout (``pyproject.toml`` + ``src/<name>/<name>.py``
    exposing ``PARAMETERS``, async ``run``, sync ``main``). The CLI
    wrapper imports and calls ``main()`` using a ``sys.path`` tweak, so
    the skill doesn't need to be pip-installed.
    """
    bin_dir = tmp_path / "bin"

    def _install(name: str) -> Path:
        src = SKILL_FIXTURES_DIR / name
        if not src.is_dir():
            raise FileNotFoundError(f"No skill fixture at {src}")
        dst = skills_dir / name
        shutil.copytree(src, dst)
        cli = bin_dir / name
        cli.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            f"sys.path.insert(0, {str(dst / 'src')!r})\n"
            f"from {name}.{name} import main\n"
            "main()\n"
        )
        cli.chmod(cli.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return dst

    return _install
