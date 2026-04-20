"""MCP (Model Context Protocol) client wrapper.

Importing this package is free — the `mcp` SDK is only loaded on the first
tool-call, inside `call_tool_sync`. The plugin registers cleanly with no
MCP package installed; invoking an MCP tool raises a clear import error.
"""
