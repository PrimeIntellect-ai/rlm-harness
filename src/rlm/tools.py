"""Tool definitions and execution."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.session import Session

# -- Tool schemas (OpenAI function-calling format) --

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for file exploration, running tests, installing packages, "
            "and invoking `rlm` for sub-tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. "
            "old_str must appear exactly once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit."},
                "old_str": {"type": "string", "description": "The exact string to find (must be unique)."},
                "new_str": {"type": "string", "description": "The replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

WEBSEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "websearch",
        "description": (
            "Search Google via the Serper API. Accepts up to 10 queries in parallel. "
            "Returns titles, URLs, snippets, and knowledge-graph data. "
            f"Use the current year ({datetime.now().year}) when searching for recent information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                    "description": (
                        "Google search queries (up to 10). "
                        "Use multiple queries to search different angles in parallel."
                    ),
                }
            },
            "required": ["queries"],
        },
    },
}

ALL_TOOLS = {"bash": BASH_TOOL, "edit": EDIT_TOOL, "websearch": WEBSEARCH_TOOL}

_RLM_CMD_RE = re.compile(r"\brlm\s")


def get_active_tools(allowed: list[str]) -> list[dict]:
    """Return tool schemas for the allowed tool names."""
    return [ALL_TOOLS[name] for name in allowed if name in ALL_TOOLS]


def run_bash(
    command: str,
    *,
    cwd: str,
    session: Session | None = None,
    timeout: int = 120,
    max_output: int = 8192,
) -> str:
    """Execute a bash command. Detects `rlm` invocations and sets up child sessions."""
    env = os.environ.copy()

    # Detect rlm sub-invocation → create child session dir
    if session and _RLM_CMD_RE.search(command):
        child_dir = session.child_dir()
        env["RLM_SESSION_DIR"] = str(child_dir)
        env["RLM_DEPTH"] = str(int(env.get("RLM_DEPTH", "0")) + 1)
        # Propagate sub-tools if set
        sub_tools = env.get("RLM_SUB_TOOLS")
        if sub_tools:
            env["RLM_TOOLS"] = sub_tools
        session.log_sub_spawn(child_dir.name, command)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"

    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"

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


def run_edit(
    path: str,
    old_str: str,
    new_str: str,
    *,
    cwd: str,
) -> str:
    """Safe single-occurrence string replacement."""
    filepath = Path(cwd) / path
    if not filepath.exists():
        return f"Error: {path} not found"
    try:
        content = filepath.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"

    count = content.count(old_str)
    if count == 0:
        return f"Error: string not found in {path}"
    if count > 1:
        return f"Error: found {count} occurrences, need exactly 1"

    filepath.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"


# -- Web search (Serper API) --


def _format_serper_results(data: dict, query: str, num_results: int = 5) -> str:
    """Format a Serper API response into readable text."""
    sections: list[str] = []

    kg = data.get("knowledgeGraph")
    if kg:
        kg_lines: list[str] = []
        title = (kg.get("title") or "").strip()
        if title:
            kg_lines.append(f"Knowledge Graph: {title}")
        description = (kg.get("description") or "").strip()
        if description:
            kg_lines.append(description)
        for key, value in (kg.get("attributes") or {}).items():
            text = str(value).strip()
            if text:
                kg_lines.append(f"{key}: {text}")
        if kg_lines:
            sections.append("\n".join(kg_lines))

    for i, result in enumerate((data.get("organic") or [])[:num_results]):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(snippet)
        sections.append("\n".join(lines))

    people_also_ask = data.get("peopleAlsoAsk") or []
    if people_also_ask:
        max_q = max(1, min(3, len(people_also_ask)))
        questions: list[str] = []
        for item in people_also_ask[:max_q]:
            question = (item.get("question") or "").strip()
            if not question:
                continue
            entry = f"Q: {question}"
            answer = (item.get("snippet") or "").strip()
            if answer:
                entry += f"\nA: {answer}"
            questions.append(entry)
        if questions:
            sections.append("People Also Ask:\n" + "\n".join(questions))

    if not sections:
        return f"No results returned for query: {query}"

    return "\n\n---\n\n".join(sections)


def _fetch_serper(query: str, api_key: str, timeout: int = 45, num_results: int = 5) -> str:
    """Execute a single Serper API search."""
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=json.dumps({"q": query}).encode(),
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"Serper search error ({e.code}): {body}") from e

    return _format_serper_results(data, query, num_results=num_results)


def run_websearch(
    queries: list[str],
    *,
    max_output: int = 8192,
    timeout: int | None = None,
    num_results: int | None = None,
) -> str:
    """Run up to 10 Google searches via Serper and return formatted results."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"

    if timeout is None:
        timeout = int(os.environ.get("RLM_WEBSEARCH_TIMEOUT", "45"))
    if num_results is None:
        num_results = int(os.environ.get("RLM_WEBSEARCH_NUM_RESULTS", "5"))

    queries = queries[:10]
    parts: list[str] = []
    for query in queries:
        try:
            result = _fetch_serper(query, api_key, timeout=timeout, num_results=num_results)
        except Exception as e:
            result = f"Error searching for '{query}': {e}"
        parts.append(f'Results for query "{query}":\n\n{result}')

    output = "\n\n---\n\n".join(parts)

    if len(output) > max_output:
        half = max_output // 2
        total = len(output)
        output = (
            output[:half]
            + f"\n... [output truncated, {total} chars total] ...\n"
            + output[-half:]
        )

    return output
