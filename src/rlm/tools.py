"""Tool definitions and execution."""

from __future__ import annotations

import copy
import os
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.session import Session


def _find_skills_dir() -> Path:
    """Locate the skills/ directory. Works for both editable and installed layouts."""
    base = Path(__file__).resolve().parent
    # Editable install: src/rlm/tools.py → ../../skills
    candidate = base.parent.parent / "skills"
    if candidate.is_dir():
        return candidate
    # Installed wheel: site-packages/rlm/tools.py → ../skills
    candidate = base.parent / "skills"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Could not find skills/ directory")


SKILLS_DIR = _find_skills_dir()

# -- Tool schemas --

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
                    "description": None,  # filled by get_active_tools()
                },
            },
            "required": ["code"],
        },
    },
}

SUMMARIZE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "summarize",
        "description": (
            "Summarize and drop old turns from context to free up space. "
            "A turn is one assistant response plus all its tool results. "
            "Dropping num_turns removes the oldest complete turns from context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_turns": {
                    "type": "integer",
                    "description": "Number of oldest turns to drop from context.",
                },
                "summary": {
                    "type": "string",
                    "description": "Your summary of the content being dropped.",
                },
            },
            "required": ["num_turns", "summary"],
        },
    },
}

# -- IPython REPL --

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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

        self._km = KernelManager(kernel_name="python3")
        self._km.start_kernel(cwd=self.cwd)
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._inject_startup()

    def _inject_startup(self):
        """Inject rlm() sub-agent function into kernel namespace."""
        session_dir = str(self.session.dir) if self.session else None
        setup_code = f"""\
import os as _os
import subprocess as _subprocess

_os.chdir({self.cwd!r})

_rlm_session_dir = {session_dir!r}
_rlm_depth = int(_os.environ.get('RLM_DEPTH', '0'))

def _rlm_spawn(prompt: str) -> str:
    \"\"\"Spawn a single rlm sub-agent. Internal helper.\"\"\"
    import uuid as _uuid
    child_id = _uuid.uuid4().hex[:8]
    child_dir = _os.path.join(_rlm_session_dir, f'sub-{{child_id}}') if _rlm_session_dir else None
    if child_dir:
        _os.makedirs(child_dir, exist_ok=True)
    env = _os.environ.copy()
    if child_dir:
        env['RLM_SESSION_DIR'] = child_dir
    env['RLM_DEPTH'] = str(_rlm_depth + 1)
    result = _subprocess.run(
        ['rlm', prompt],
        capture_output=True, text=True, cwd={self.cwd!r}, env=env,
    )
    return result.stdout.strip()

def rlm(prompt: str) -> str:
    \"\"\"Run an rlm sub-agent on the given prompt. Returns its answer.\"\"\"
    return _rlm_spawn(prompt)

def rlm_batch(prompts: list) -> list:
    \"\"\"Run multiple rlm sub-agents in parallel. Returns list of answers.\"\"\"
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=len(prompts)) as pool:
        return list(pool.map(_rlm_spawn, prompts))
"""
        self._execute_silent(setup_code)

    def _execute_silent(self, code: str):
        """Execute code without capturing output (for setup)."""
        self._kc.execute(code, silent=True)
        self._kc.get_shell_msg(timeout=30)

    def execute(self, code: str, timeout: int | None = None, max_output: int = 8192) -> str:
        """Execute code and return combined output. Thread-safe via lock."""
        with self._lock:
            return self._execute_locked(code, timeout, max_output)

    def _execute_locked(self, code: str, timeout: int | None, max_output: int) -> str:
        msg_id = self._kc.execute(code)

        outputs: list[str] = []
        try:
            while True:
                try:
                    msg = self._kc.get_iopub_msg(timeout=timeout)
                except Exception:
                    outputs.append(f"\n[execution timed out after {timeout}s]")
                    break

                # Only process messages from this execution
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
            # Always drain the shell reply
            try:
                self._kc.get_shell_msg(timeout=5)
            except Exception:
                pass

        output = "".join(outputs)

        # Truncate large output
        if len(output) > max_output:
            half = max_output // 2
            total = len(output)
            output = (
                output[:half]
                + f"\n... [output truncated, {total} chars total] ...\n"
                + output[-half:]
            )

        return output

    def shutdown(self):
        """Stop the kernel."""
        if self._kc:
            self._kc.stop_channels()
        if self._km:
            self._km.shutdown_kernel(now=True)


def get_active_tools() -> list[dict]:
    """Return OpenAI tool schemas with runtime defaults baked in."""
    timeout = int(os.environ.get("RLM_EXEC_TIMEOUT", "300"))
    schema = copy.deepcopy(IPYTHON_SCHEMA)
    schema["function"]["parameters"]["properties"]["timeout"]["description"] = (
        f"Optional timeout in seconds. Default: {timeout}s."
    )
    return [schema, SUMMARIZE_SCHEMA]
