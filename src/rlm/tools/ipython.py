"""Builtin IPython tool and persistent REPL implementation."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from queue import Empty
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.skills import TASK_SKILLS_DIR
from rlm.types import BuiltinToolCalled, IpythonExecuted

if TYPE_CHECKING:
    from rlm.session import Session


IPYTHON_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute code in a persistent IPython session. "
            "Variables, imports, and definitions persist across calls. "
            "Use !command for shell commands (e.g. !git diff, !ls -la). "
            "Use %%bash for multi-line shell scripts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python or IPython code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": None,  # filled by schema()
                },
            },
            "required": ["code"],
        },
    },
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
IPYTHON_TIMEOUT_MAX_SECONDS = 600


class IpythonTool:
    """Builtin tool handler for the persistent IPython session."""

    name = "ipython"

    def schema(self) -> dict[str, Any]:
        timeout = min(
            int(os.environ.get("RLM_EXEC_TIMEOUT", "300")),
            IPYTHON_TIMEOUT_MAX_SECONDS,
        )
        schema = copy.deepcopy(IPYTHON_SCHEMA)
        schema["function"]["parameters"]["properties"]["timeout"]["description"] = (
            "Optional timeout in seconds. "
            f"Default: {timeout}s. Maximum: {IPYTHON_TIMEOUT_MAX_SECONDS}s."
        )
        return schema

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        code = args.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        input_chars = len(code)
        input_loc = self._count_nonempty_lines(code)
        metric_events = [
            BuiltinToolCalled(self.name),
            IpythonExecuted(input_chars=input_chars, input_loc=input_loc),
        ]

        timeout = args.get("timeout")
        if timeout is None:
            timeout = context.exec_timeout
        else:
            try:
                timeout = int(timeout)
            except (TypeError, ValueError):
                timeout = context.exec_timeout
        timeout = min(timeout, IPYTHON_TIMEOUT_MAX_SECONDS)

        if context.repl is None:
            return ToolOutcome(
                content="Error: IPython REPL is not available",
                metric_events=metric_events,
            )

        return ToolOutcome(
            content=context.repl.execute(code, timeout=timeout),
            metric_events=metric_events,
        )

    @staticmethod
    def _count_nonempty_lines(code: str) -> int:
        return sum(1 for line in code.splitlines() if line.strip())


class IPythonREPL:
    """Persistent IPython kernel communicating via Jupyter protocol."""

    def __init__(self, cwd: str, session: "Session | None" = None):
        self.cwd = cwd
        self.session = session
        self._km = None
        self._kc = None
        self._lock = threading.Lock()

    def start(self):
        """Start the IPython kernel."""
        from jupyter_client import KernelManager

        kernel_python = os.environ.get("RLM_KERNEL_PYTHON") or sys.executable
        self._km = KernelManager()
        self._km.kernel_spec.argv = [
            kernel_python,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        self._km.start_kernel(cwd=self.cwd)
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._inject_startup()

    def _inject_startup(self):
        """Set up kernel: cwd, nest_asyncio, env vars, skill shims."""
        session_dir = str(self.session.dir) if self.session else None
        depth = int(os.environ.get("RLM_DEPTH", "0"))
        shim_file = str(Path(__file__).resolve().parent.parent / "kernel_shim.py")
        skills_dir = str(TASK_SKILLS_DIR)

        setup_code = f"""\
import os, sys
os.chdir({self.cwd!r})
os.environ['RLM_SESSION_DIR'] = {session_dir!r} or ''
os.environ['RLM_DEPTH'] = str({depth!r} + 1)

import nest_asyncio
nest_asyncio.apply()

import asyncio

# Install skill shims so `import edit`, `import rlm` etc. work
# even when the kernel runs in a different Python.  Load kernel_shim
# directly by path to avoid triggering rlm's __init__ (which needs openai).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("kernel_shim", {shim_file!r})
_shim = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_shim)
_shim.install_shims({skills_dir!r})
"""
        self._execute_silent(setup_code)

    def _execute_silent(self, code: str):
        """Execute code without capturing output (for setup)."""
        self._kc.execute(code, silent=True)
        self._kc.get_shell_msg(timeout=30)

    def _wait_for_idle(self, timeout: float) -> bool:
        """Wait briefly for the kernel to report an idle state."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                msg = self._kc.get_iopub_msg(timeout=remaining)
            except Empty:
                return False
            if (
                msg["msg_type"] == "status"
                and msg["content"].get("execution_state") == "idle"
            ):
                return True

    def restart_kernel(self):
        """Restart the kernel and restore the initial REPL state."""
        if self._kc:
            self._kc.stop_channels()
        self._km.restart_kernel(now=True)
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._inject_startup()

    def _interrupt_and_recover(self):
        """Interrupt the running cell and restart the kernel if needed."""
        self._km.interrupt_kernel()
        if not self._wait_for_idle(timeout=2):
            self.restart_kernel()

    def execute(self, code: str, timeout: int | None = None) -> str:
        """Execute code and return combined output."""
        with self._lock:
            return self._execute_locked(code, timeout)

    def _execute_locked(self, code: str, timeout: int | None) -> str:
        msg_id = self._kc.execute(code)
        deadline = None if timeout is None else time.monotonic() + timeout

        outputs: list[str] = []
        try:
            while True:
                if deadline is None:
                    wait_timeout = None
                else:
                    wait_timeout = deadline - time.monotonic()
                    if wait_timeout <= 0:
                        self._interrupt_and_recover()
                        outputs.append(
                            f"\n[execution timed out after {timeout}s and was interrupted]"
                        )
                        break
                try:
                    msg = self._kc.get_iopub_msg(timeout=wait_timeout)
                except Empty:
                    self._interrupt_and_recover()
                    outputs.append(
                        f"\n[execution timed out after {timeout}s and was interrupted]"
                    )
                    break

                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]
                content = msg["content"]

                if msg_type == "stream":
                    outputs.append(content["text"])
                elif msg_type == "execute_result":
                    text = content.get("data", {}).get("text/plain", "")
                    if text:
                        outputs.append(text + "\n")
                elif msg_type == "error":
                    tb = "\n".join(content.get("traceback", []))
                    tb = _ANSI_RE.sub("", tb)
                    outputs.append(tb)
                elif msg_type == "status" and content["execution_state"] == "idle":
                    break
        finally:
            try:
                self._kc.get_shell_msg(timeout=5)
            except Exception:
                pass

        return "".join(outputs)

    def shutdown(self):
        """Stop the kernel."""
        if self._kc:
            self._kc.stop_channels()
            self._kc = None
        if self._km:
            self._km.shutdown_kernel(now=True)
            self._km = None
