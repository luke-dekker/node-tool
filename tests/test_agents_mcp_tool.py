"""MCPToolNode — expose one tool from an MCP server as a callable TOOL.

The mcp SDK is asyncio-heavy and optional; the node's construction path
must work with it absent. All tests monkey-patch the sync wrapper in
`plugins.agents._mcp.mcp_client` so nothing actually spawns a process.
"""
from __future__ import annotations
import importlib.util

import pytest


HAS_MCP = importlib.util.find_spec("mcp") is not None


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── register + import invariants ──────────────────────────────────────────

def test_mcp_module_imports_without_sdk():
    """Importing _mcp.mcp_client must not pull the mcp package."""
    import plugins.agents._mcp.mcp_client as m
    assert hasattr(m, "call_tool_sync")
    assert hasattr(m, "list_tools_sync")


def test_plugin_registers_mcp_node():
    from core.plugins import PluginContext
    import plugins.agents as agents_pkg
    ctx = PluginContext()
    agents_pkg.register(ctx)
    names = {c.type_name for c in ctx.node_classes}
    assert "ag_mcp_tool" in names


# ── Node validation ───────────────────────────────────────────────────────

def test_node_requires_tool_name():
    from nodes.agents.mcp_tool import MCPToolNode
    with pytest.raises(RuntimeError, match="tool_name is empty"):
        MCPToolNode().execute({
            "transport": "stdio", "command": "python", "args": "-m s",
            "url": "", "tool_name": "", "description": "",
            "input_schema": "", "side_effect": True, "timeout_s": 30.0,
        })


def test_node_stdio_requires_command():
    from nodes.agents.mcp_tool import MCPToolNode
    with pytest.raises(RuntimeError, match="requires `command`"):
        MCPToolNode().execute({
            "transport": "stdio", "command": "", "args": "",
            "url": "", "tool_name": "x", "description": "",
            "input_schema": "", "side_effect": True, "timeout_s": 30.0,
        })


def test_node_http_requires_url():
    from nodes.agents.mcp_tool import MCPToolNode
    with pytest.raises(RuntimeError, match="requires `url`"):
        MCPToolNode().execute({
            "transport": "http", "command": "", "args": "",
            "url": "", "tool_name": "x", "description": "",
            "input_schema": "", "side_effect": True, "timeout_s": 30.0,
        })


def test_node_rejects_unknown_transport():
    from nodes.agents.mcp_tool import MCPToolNode
    with pytest.raises(RuntimeError, match="unknown transport"):
        MCPToolNode().execute({
            "transport": "websocket", "command": "python", "args": "",
            "url": "", "tool_name": "x", "description": "",
            "input_schema": "", "side_effect": True, "timeout_s": 30.0,
        })


def test_node_rejects_invalid_schema():
    from nodes.agents.mcp_tool import MCPToolNode
    with pytest.raises(RuntimeError, match="invalid input_schema"):
        MCPToolNode().execute({
            "transport": "stdio", "command": "python", "args": "",
            "url": "", "tool_name": "x", "description": "",
            "input_schema": "{bad", "side_effect": True, "timeout_s": 30.0,
        })


# ── Tool construction and dispatch (with mocked sync wrapper) ──────────────

def test_node_builds_tool_with_permissive_schema_by_default():
    from nodes.agents.mcp_tool import MCPToolNode
    out = MCPToolNode().execute({
        "transport": "stdio", "command": "python", "args": "-m my_server",
        "url": "", "tool_name": "search", "description": "",
        "input_schema": "", "side_effect": True, "timeout_s": 30.0,
    })
    td = out["tool"]
    assert td.name == "search"
    assert td.description == "MCP tool 'search'"
    assert td.input_schema["additionalProperties"] is True
    assert td.side_effect is True


def test_node_applies_user_supplied_schema():
    import json
    from nodes.agents.mcp_tool import MCPToolNode
    schema = json.dumps({
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    })
    out = MCPToolNode().execute({
        "transport": "stdio", "command": "python", "args": "",
        "url": "", "tool_name": "search", "description": "Search docs",
        "input_schema": schema, "side_effect": False, "timeout_s": 5.0,
    })
    td = out["tool"]
    assert td.input_schema["required"] == ["q"]
    assert td.description == "Search docs"
    assert td.side_effect is False


def test_tool_call_routes_stdio(monkeypatch):
    from nodes.agents.mcp_tool import MCPToolNode
    from plugins.agents._mcp import mcp_client

    recorded = {}
    def _fake(**kw):
        recorded.update(kw)
        return "RESULT"
    monkeypatch.setattr(mcp_client, "call_tool_sync", _fake)

    out = MCPToolNode().execute({
        "transport": "stdio", "command": "python",
        "args": "-m my_server --flag",
        "url": "", "tool_name": "fetch", "description": "",
        "input_schema": "", "side_effect": True, "timeout_s": 12.0,
    })
    td = out["tool"]
    result = td.callable(url="https://example.com", follow=True)
    assert result == "RESULT"
    assert recorded["transport"] == "stdio"
    assert recorded["tool_name"] == "fetch"
    assert recorded["command"] == "python"
    assert recorded["args"] == ["-m", "my_server", "--flag"]
    assert recorded["arguments"] == {"url": "https://example.com", "follow": True}
    assert recorded["timeout"] == 12.0


def test_tool_call_routes_http(monkeypatch):
    from nodes.agents.mcp_tool import MCPToolNode
    from plugins.agents._mcp import mcp_client

    captured = {}
    monkeypatch.setattr(
        mcp_client, "call_tool_sync",
        lambda **kw: captured.update(kw) or {"ok": True},
    )

    td = MCPToolNode().execute({
        "transport": "http", "command": "", "args": "",
        "url": "https://mcp.example/sse", "tool_name": "ping",
        "description": "", "input_schema": "", "side_effect": False,
        "timeout_s": 5.0,
    })["tool"]
    result = td.callable(payload="hi")
    assert result == {"ok": True}
    assert captured["transport"] == "http"
    assert captured["url"] == "https://mcp.example/sse"
    assert captured["arguments"] == {"payload": "hi"}


def test_tool_call_surfaces_backend_error(monkeypatch):
    from nodes.agents.mcp_tool import MCPToolNode
    from plugins.agents._mcp import mcp_client

    def _boom(**kw):
        raise RuntimeError("MCP tool error: missing arg")
    monkeypatch.setattr(mcp_client, "call_tool_sync", _boom)

    td = MCPToolNode().execute({
        "transport": "stdio", "command": "python", "args": "",
        "url": "", "tool_name": "x", "description": "",
        "input_schema": "", "side_effect": True, "timeout_s": 5.0,
    })["tool"]
    with pytest.raises(RuntimeError, match="MCP tool error"):
        td.callable()


def test_call_tool_sync_rejects_unknown_transport():
    from plugins.agents._mcp.mcp_client import call_tool_sync
    with pytest.raises(ValueError, match="Unknown MCP transport"):
        call_tool_sync(transport="websocket", tool_name="x", arguments={})


# ── Result shaping helpers ─────────────────────────────────────────────────

def test_flatten_result_joins_text_content():
    from plugins.agents._mcp.mcp_client import _flatten_result

    class _Block:
        def __init__(self, text): self.text = text

    class _Result:
        isError = False
        def __init__(self, blocks): self.content = blocks

    r = _Result([_Block("hello"), _Block("world")])
    assert _flatten_result(r) == "hello\nworld"


def test_flatten_result_raises_on_error():
    from plugins.agents._mcp.mcp_client import _flatten_result

    class _Block:
        def __init__(self, text): self.text = text

    class _Result:
        isError = True
        content = [_Block("kaboom")]

    with pytest.raises(RuntimeError, match="kaboom"):
        _flatten_result(_Result())


def test_flatten_tools_extracts_metadata():
    from plugins.agents._mcp.mcp_client import _flatten_tools

    class _Tool:
        def __init__(self, name, desc, schema):
            self.name = name
            self.description = desc
            self.inputSchema = schema

    class _Listing:
        def __init__(self, tools):
            self.tools = tools

    lst = _Listing([
        _Tool("search", "Web search",
              {"type": "object", "properties": {"q": {"type": "string"}}}),
        _Tool("fetch", "", None),
    ])
    out = _flatten_tools(lst)
    assert out[0]["name"] == "search"
    assert out[0]["inputSchema"]["properties"]["q"]["type"] == "string"
    assert out[1]["description"] == ""
    # None schema → permissive default
    assert out[1]["inputSchema"]["additionalProperties"] is True
