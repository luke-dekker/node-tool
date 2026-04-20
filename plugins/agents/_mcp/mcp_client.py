"""Sync MCP call wrapper.

Anthropic's `mcp` Python SDK is asyncio-native. The graph executes nodes
synchronously on the caller's thread, so we wrap each tool call in a
fresh `asyncio.run()`. That's heavy — one event loop per call — but keeps
the integration simple and matches the v1 "solo-use, one call at a time"
posture. A persistent session pool is a Phase-D optimization.

Transports supported in v1:
  - stdio: subprocess; command + args
  - http:  server URL for the HTTP/SSE transport shipped with the SDK

Imports of `mcp` are deferred to `call_tool_sync`; this module is free to
import without the SDK installed.
"""
from __future__ import annotations
from typing import Any


def call_tool_sync(
    *,
    transport: str,
    tool_name: str,
    arguments: dict | None = None,
    command: str = "",
    args: list[str] | None = None,
    url: str = "",
    env: dict | None = None,
    timeout: float = 30.0,
) -> Any:
    """Invoke a single MCP tool and return its result. Synchronous.

    Raises RuntimeError on transport errors with a clear message.
    """
    import asyncio
    arguments = arguments or {}
    args = list(args or [])
    if transport == "stdio":
        return asyncio.run(
            _call_stdio(command=command, args=args, env=env,
                         tool_name=tool_name, arguments=arguments,
                         timeout=timeout)
        )
    if transport == "http":
        return asyncio.run(
            _call_http(url=url, tool_name=tool_name,
                        arguments=arguments, timeout=timeout)
        )
    raise ValueError(f"Unknown MCP transport: {transport!r}")


def list_tools_sync(
    *, transport: str, command: str = "", args: list[str] | None = None,
    url: str = "", env: dict | None = None, timeout: float = 10.0,
) -> list[dict]:
    """Fetch the server's advertised tool list. Used at node-configure time."""
    import asyncio
    args = list(args or [])
    if transport == "stdio":
        return asyncio.run(
            _list_stdio(command=command, args=args, env=env, timeout=timeout)
        )
    if transport == "http":
        return asyncio.run(_list_http(url=url, timeout=timeout))
    raise ValueError(f"Unknown MCP transport: {transport!r}")


# ── Async primitives (all deferred imports) ────────────────────────────────

async def _call_stdio(*, command, args, env, tool_name, arguments, timeout):
    from mcp import ClientSession, StdioServerParameters  # deferred
    from mcp.client.stdio import stdio_client             # deferred
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
    return _flatten_result(result)


async def _list_stdio(*, command, args, env, timeout):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
    return _flatten_tools(listing)


async def _call_http(*, url, tool_name, arguments, timeout):
    from mcp import ClientSession                          # deferred
    from mcp.client.streamable_http import streamablehttp_client  # deferred
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
    return _flatten_result(result)


async def _list_http(*, url, timeout):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
    return _flatten_tools(listing)


# ── Result shaping ─────────────────────────────────────────────────────────

def _flatten_result(result: Any) -> Any:
    """Collapse an MCP CallToolResult into a plain Python value.

    MCP returns structured content blocks; the LLM usually just wants text.
    We join text blocks with newlines and surface `isError` via exception.
    """
    if getattr(result, "isError", False):
        parts = [getattr(b, "text", str(b)) for b in getattr(result, "content", [])]
        raise RuntimeError("MCP tool error: " + (" ".join(parts) or "unknown"))
    content = getattr(result, "content", [])
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
        else:
            texts.append(str(block))
    return "\n".join(texts) if texts else None


def _flatten_tools(listing: Any) -> list[dict]:
    """Convert mcp's ListToolsResult → [{name, description, inputSchema}]."""
    out: list[dict] = []
    for t in getattr(listing, "tools", []) or []:
        out.append({
            "name":        getattr(t, "name", ""),
            "description": getattr(t, "description", "") or "",
            "inputSchema": getattr(t, "inputSchema", None) or
                           {"type": "object", "properties": {},
                            "additionalProperties": True},
        })
    return out
