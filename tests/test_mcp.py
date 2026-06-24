"""Tests for MCP-tools-as-skills (``rlm.mcp``).

Covers the pure logic: config parsing, JSON-schema → signature/docstring rendering, and
generation of importable skill modules. The live MCP round-trip (kernel pre-import +
streamable-HTTP call) is exercised end-to-end by the general-agent-v1 eval, not here.
"""

from __future__ import annotations

import importlib
import inspect
import sys

from rlm import mcp

SCHEMA = {
    "type": "object",
    "properties": {
        "day": {"type": "string", "description": "the day"},
        "count": {"type": "integer"},
        "weird-name": {"type": "string"},
    },
    "required": ["day"],
}


def test_load_mcp_servers(monkeypatch):
    monkeypatch.delenv(mcp.MCP_CONFIG_ENV, raising=False)
    assert mcp.load_mcp_servers() == {}

    monkeypatch.setenv(
        mcp.MCP_CONFIG_ENV,
        '{"mcpServers": {"tools": {"url": "http://h/mcp"}, "web": {"url": "http://h/web"}}}',
    )
    assert mcp.load_mcp_servers() == {"tools": "http://h/mcp", "web": "http://h/web"}


def test_sanitize():
    assert mcp._sanitize("tools_add_event") == "tools_add_event"
    assert mcp._sanitize("web-search.run") == "web_search_run"
    assert mcp._sanitize("2fa") == "_2fa"


def test_build_signature():
    sig = mcp.build_signature(SCHEMA)
    params = sig.parameters
    # non-identifier property is skipped; required comes before optional.
    assert list(params) == ["day", "count"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    assert params["day"].default is inspect.Parameter.empty
    assert params["day"].annotation is str
    assert params["count"].default is None
    assert params["count"].annotation is int


def test_render_doc():
    doc = mcp.render_doc("Add an event.", SCHEMA)
    assert doc.splitlines()[0] == "Add an event."
    assert "day (string): the day" in doc
    assert "count (integer, optional)" in doc


def test_write_skill_modules(tmp_path):
    tool = mcp.McpTool(
        skill="tools_add_event",
        tool="add_event",
        url="http://h/mcp",
        description="Add an event.\ndetails",
        input_schema=SCHEMA,
    )
    skills = mcp.write_skill_modules([tool], tmp_path)
    assert skills == [
        mcp.McpSkill(
            name="tools_add_event",
            signature="(*, day: str, count: int = None)",
            summary="Add an event.",
        )
    ]

    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("tools_add_event")
        assert inspect.iscoroutinefunction(module.run)
        assert str(inspect.signature(module.run)) == "(*, day: str, count: int = None)"
        assert module.run.__doc__.splitlines()[0] == "Add an event."
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("tools_add_event", None)
