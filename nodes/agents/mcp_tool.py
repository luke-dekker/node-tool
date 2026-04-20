"""MCPToolNode — expose one tool from an MCP (Model Context Protocol) server.

MCP servers advertise a set of tools over stdio or HTTP. This node binds
ONE named tool as a callable TOOL the agent can invoke. If the server
offers multiple tools, drop multiple MCPToolNodes (one per tool) — fits
AgentNode's 4 fixed TOOL slots cleanly and lets the user pick only the
tools they want to expose.

`input_schema` is user-supplied JSON (blank = permissive passthrough).
Fetching the schema from the server on node-execute would require the
server to be up at graph-build time; we accept that as a Phase-D upgrade.

`side_effect=True` by default — most MCP tools (shell, file I/O, HTTP)
have side effects; flip to False only for read-only tools.
"""
from __future__ import annotations
import json
from typing import Any

from core.node import BaseNode, PortType


class MCPToolNode(BaseNode):
    type_name   = "ag_mcp_tool"
    label       = "MCP Tool"
    category    = "Agents"
    subcategory = "Tools"
    description = ("Bind one tool from an MCP server (stdio or HTTP transport) "
                   "as a callable TOOL the agent can invoke.")

    def _setup_ports(self) -> None:
        self.add_input("transport", PortType.STRING, default="stdio",
                       choices=["stdio", "http"],
                       description="stdio (local subprocess) or http (remote server)")
        self.add_input("command", PortType.STRING, default="",
                       description="stdio transport: executable (e.g. 'python')")
        self.add_input("args", PortType.STRING, default="",
                       description="stdio transport: space-separated CLI args")
        self.add_input("url", PortType.STRING, default="",
                       description="http transport: server URL")
        self.add_input("tool_name", PortType.STRING, default="",
                       description="Name of the server tool to expose")
        self.add_input("description", PortType.STRING, default="",
                       description="Description surfaced to the LLM")
        self.add_input("input_schema", PortType.STRING, default="",
                       description=("JSON Schema for the tool's arguments. "
                                    "Blank = permissive passthrough."))
        self.add_input("side_effect", PortType.BOOL, default=True,
                       description="Most MCP tools mutate state — default True.")
        self.add_input("timeout_s", PortType.FLOAT, default=30.0,
                       description="Per-call timeout in seconds")
        self.add_output("tool", "TOOL")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import ToolDef  # deferred
        from plugins.agents._mcp import mcp_client        # deferred (no heavy imports inside)

        transport = (inputs.get("transport") or "stdio").strip().lower() or "stdio"
        if transport not in ("stdio", "http"):
            raise RuntimeError(
                f"MCPToolNode: unknown transport {transport!r} "
                "(supported: stdio, http)"
            )
        tool_name = (inputs.get("tool_name") or "").strip()
        if not tool_name:
            raise RuntimeError("MCPToolNode: tool_name is empty")

        command = (inputs.get("command") or "").strip()
        args_raw = (inputs.get("args") or "").strip()
        args = args_raw.split() if args_raw else []
        url = (inputs.get("url") or "").strip()
        if transport == "stdio" and not command:
            raise RuntimeError("MCPToolNode: stdio transport requires `command`")
        if transport == "http" and not url:
            raise RuntimeError("MCPToolNode: http transport requires `url`")

        schema_raw = (inputs.get("input_schema") or "").strip()
        if schema_raw:
            try:
                schema = json.loads(schema_raw)
                if not isinstance(schema, dict):
                    raise ValueError("input_schema must decode to an object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"MCPToolNode: invalid input_schema JSON: {exc}")
        else:
            schema = {"type": "object", "properties": {},
                      "additionalProperties": True}

        desc = (inputs.get("description") or "").strip() or f"MCP tool {tool_name!r}"
        timeout = float(inputs.get("timeout_s") or 30.0)

        def _call(**kwargs) -> Any:
            return mcp_client.call_tool_sync(
                transport=transport, tool_name=tool_name,
                arguments=dict(kwargs), command=command, args=args,
                url=url, timeout=timeout,
            )

        return {"tool": ToolDef(
            name=tool_name, description=desc, input_schema=schema,
            callable=_call, side_effect=bool(inputs.get("side_effect", True)),
        )}

    def export(self, iv, ov):
        return [], [f"# MCPToolNode export pending Phase D"]
