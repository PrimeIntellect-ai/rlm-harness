"""MCP tool servers exposed as pre-imported IPython skills.

A v1 harness can wire task-specific tool servers to the agent over MCP. rlm has no
MCP client in its native tool-call loop; instead each MCP tool becomes a *skill* — a
generated async function the agent calls straight from the IPython REPL
(``await tools_add_event(...)``), the same programmatic tool-call (PTC) path used for
installed skills. The servers arrive as a standard ``mcpServers`` URL map in the
``RLM_MCP_CONFIG`` env var (set by the verifiers rlm harness).

Each tool is written to its own flat module under a skills directory; the IPython
kernel adds that directory to ``sys.path`` and pre-imports the modules (see
``tools/ipython.py``). The module's ``run`` carries a synthetic ``__signature__`` and
docstring built from the tool's input schema, so ``help()`` and ``inspect.signature``
show the real argument surface.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MCP_CONFIG_ENV = "RLM_MCP_CONFIG"

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True)
class McpTool:
    """A single tool discovered on an MCP server."""

    skill: str
    """Model-facing skill name (importable identifier + REPL call name): ``<server>_<tool>``."""
    tool: str
    """The raw tool name on the server (what ``call_tool`` dispatches)."""
    url: str
    """The server's streamable-HTTP URL."""
    description: str
    input_schema: dict


@dataclass(frozen=True)
class McpSkill:
    """Prompt-facing info for a generated MCP skill."""

    name: str
    signature: str
    summary: str


def load_mcp_servers() -> dict[str, str]:
    """Parse ``RLM_MCP_CONFIG`` (a standard ``mcpServers`` URL map) into ``{name: url}``.

    Returns an empty dict when the var is unset, so callers can branch on truthiness.
    """
    raw = os.environ.get(MCP_CONFIG_ENV)
    if not raw:
        return {}
    servers = json.loads(raw).get("mcpServers", {})
    return {name: spec["url"] for name, spec in servers.items()}


def _sanitize(name: str) -> str:
    """A valid Python identifier for a model-facing skill name."""
    ident = re.sub(r"\W", "_", name)
    return f"_{ident}" if ident[:1].isdigit() else ident


async def discover_tools(servers: dict[str, str]) -> list[McpTool]:
    """List every tool on each configured server (one streamable-HTTP session per server)."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    tools: list[McpTool] = []
    for server, url in servers.items():
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for tool in (await session.list_tools()).tools:
                    tools.append(
                        McpTool(
                            skill=_sanitize(f"{server}_{tool.name}"),
                            tool=tool.name,
                            url=url,
                            description=(tool.description or "").strip(),
                            input_schema=tool.inputSchema or {},
                        )
                    )
    return tools


async def call_tool(url: str, name: str, arguments: dict) -> str:
    """Call an MCP tool over streamable HTTP and return its text content.

    Raises ``RuntimeError`` when the tool reports an error, so a failed call surfaces as
    an exception in the REPL (the natural Python-function contract) rather than a
    silently-wrong return value.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments or {})
    text = _content_text(result.content)
    if result.isError:
        raise RuntimeError(text or f"MCP tool {name!r} failed")
    return text


def _content_text(blocks) -> str:
    """Flatten MCP content blocks to text (text blocks joined; others stringified)."""
    parts = []
    for block in blocks:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "\n".join(parts)


def _annotation(prop: dict) -> Any:
    """Python annotation for a JSON-schema property, or no annotation if unknown."""
    return _JSON_TO_PY.get(prop.get("type"), inspect.Parameter.empty)


def build_signature(schema: dict) -> inspect.Signature:
    """A keyword-only signature mirroring an MCP tool's input schema.

    Properties become keyword-only parameters (call sites read as ``tool(name=value)``);
    required ones have no default, the rest default to ``None``. Non-identifier property
    names are skipped (still reachable via ``**kwargs``).
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if name in required else None,
            annotation=_annotation(prop),
        )
        for name, prop in props.items()
        if name.isidentifier()
    ]
    params.sort(key=lambda p: p.default is not inspect.Parameter.empty)
    return inspect.Signature(params)


def render_doc(description: str, schema: dict) -> str:
    """A Google-style docstring built from the tool description and input schema."""
    lines = [description or "MCP tool."]
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if props:
        lines += ["", "Args:"]
        for name, prop in props.items():
            kind = prop.get("type", "any")
            opt = "" if name in required else ", optional"
            desc = (prop.get("description") or "").strip()
            lines.append(f"    {name} ({kind}{opt})" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


def make_skill(
    url: str, tool: str, schema: dict, description: str, name: str | None = None
):
    """Build the async ``run`` callable for a generated MCP-tool skill module.

    Generated modules are a single statement (``run = make_skill(...)``). The returned
    coroutine forwards keyword args to the MCP tool; its ``__signature__`` and ``__doc__``
    mirror the input schema so introspection (and the REPL skill-wrapping) expose the
    real API instead of ``(**kwargs)``.
    """

    async def run(**kwargs):
        return await call_tool(url, tool, kwargs)

    run.__name__ = name or tool
    run.__qualname__ = run.__name__
    run.__signature__ = build_signature(schema)
    run.__doc__ = render_doc(description, schema)
    return run


_MODULE_TEMPLATE = '''\
"""{summary}"""

from rlm.mcp import make_skill

run = make_skill(
    url={url!r},
    tool={tool!r},
    schema={schema!r},
    description={description!r},
    name={name!r},
)
'''


def write_skill_modules(tools: list[McpTool], dest_dir: Path) -> list[McpSkill]:
    """Write one importable ``<skill>.py`` per MCP tool into ``dest_dir``.

    Returns prompt-facing info (name, signature, one-line summary) for each skill.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    skills: list[McpSkill] = []
    for tool in tools:
        first_line = tool.description.splitlines()[0] if tool.description else ""
        summary = first_line or f"MCP tool {tool.tool}."
        source = _MODULE_TEMPLATE.format(
            summary=summary.replace('"""', "'''"),
            url=tool.url,
            tool=tool.tool,
            schema=tool.input_schema,
            description=tool.description,
            name=tool.skill,
        )
        (dest_dir / f"{tool.skill}.py").write_text(source)
        skills.append(
            McpSkill(
                name=tool.skill,
                signature=str(build_signature(tool.input_schema)),
                summary=summary,
            )
        )
    return skills


async def generate_mcp_skills(
    servers: dict[str, str], dest_dir: Path
) -> list[McpSkill]:
    """Discover all tools on ``servers`` and write them as skill modules in ``dest_dir``."""
    return write_skill_modules(await discover_tools(servers), dest_dir)
