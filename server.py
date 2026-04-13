"""WebSocket JSON-RPC server — bridges Godot frontend to the Python core.

Exposes the node registry, graph operations, and execution over a
WebSocket connection. Godot connects as a client and sends JSON-RPC 2.0
messages. The server processes them against the existing core/ and nodes/
modules — no duplication, just a thin RPC layer.

Protocol: JSON-RPC 2.0 over WebSocket
  Request:  {"jsonrpc":"2.0", "method":"...", "params":{...}, "id":1}
  Response: {"jsonrpc":"2.0", "result":{...}, "id":1}
  Notify:   {"jsonrpc":"2.0", "method":"...", "params":{...}}  (no id)

Start:  python server.py [--port 9800]
"""
from __future__ import annotations

import sys
import os
import json
import asyncio
import argparse
import traceback
from typing import Any

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import websockets
from websockets.asyncio.server import serve

from core.graph import Graph
from core.node import BaseNode, PortType
from core.port_types import PortTypeRegistry
from nodes import NODE_REGISTRY, get_nodes_by_category, CATEGORY_ORDER


class NodeToolServer:
    """Manages graph state and handles RPC calls from the Godot client."""

    def __init__(self):
        self.graph = Graph()
        self._last_outputs: dict[str, dict[str, Any]] = {}
        self._clients: set = set()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _node_to_dict(self, node: BaseNode) -> dict:
        """Serialize a node instance for the frontend."""
        inputs = {}
        for pname, port in node.inputs.items():
            inputs[pname] = {
                "port_type": port.port_type,
                "default_value": self._safe_value(port.default_value),
                "editable": PortTypeRegistry.is_editable(port.port_type),
                "color": list(PortTypeRegistry.get_color(port.port_type)),
                "description": getattr(port, "description", ""),
                "choices": getattr(port, "choices", None),
            }
        outputs = {}
        for pname, port in node.outputs.items():
            outputs[pname] = {
                "port_type": port.port_type,
                "color": list(PortTypeRegistry.get_color(port.port_type)),
                "description": getattr(port, "description", ""),
            }
        return {
            "id": node.id,
            "type_name": node.type_name,
            "label": node.label,
            "category": node.category,
            "subcategory": getattr(node, "subcategory", ""),
            "description": node.description,
            "inputs": inputs,
            "outputs": outputs,
        }

    @staticmethod
    def _safe_value(val: Any) -> Any:
        """Convert a value to JSON-safe representation."""
        if val is None or isinstance(val, (bool, int, float, str)):
            return val
        if isinstance(val, (list, tuple)):
            return [NodeToolServer._safe_value(v) for v in val]
        # Fallback: stringify
        return str(val)

    def _summarise_output(self, val: Any) -> str:
        """Short string summary of an execution output value."""
        if val is None:
            return "None"
        if isinstance(val, (bool, int, float, str)):
            return str(val)
        try:
            import torch
            if isinstance(val, torch.Tensor):
                return f"Tensor {list(val.shape)} {val.dtype}"
            if isinstance(val, torch.nn.Module):
                n = sum(p.numel() for p in val.parameters())
                return f"{val.__class__.__name__} ({n:,} params)"
        except ImportError:
            pass
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                return f"ndarray {list(val.shape)} {val.dtype}"
        except ImportError:
            pass
        return type(val).__name__

    # ── RPC Methods ──────────────────────────────────────────────────────

    def get_registry(self, params: dict) -> dict:
        """Return the full node registry organized by category."""
        categories = {}
        for cat in CATEGORY_ORDER:
            cat_nodes = get_nodes_by_category().get(cat, [])
            if not cat_nodes:
                continue
            categories[cat] = []
            for cls in cat_nodes:
                # Create a temporary instance to read port definitions
                tmp = cls()
                categories[cat].append({
                    "type_name": cls.type_name,
                    "label": cls.label,
                    "category": cls.category,
                    "subcategory": getattr(cls, "subcategory", ""),
                    "description": cls.description,
                    "inputs": {
                        pname: {
                            "port_type": p.port_type,
                            "default_value": self._safe_value(p.default_value),
                            "editable": PortTypeRegistry.is_editable(p.port_type),
                            "color": list(PortTypeRegistry.get_color(p.port_type)),
                            "choices": getattr(p, "choices", None),
                        }
                        for pname, p in tmp.inputs.items()
                    },
                    "outputs": {
                        pname: {
                            "port_type": p.port_type,
                            "color": list(PortTypeRegistry.get_color(p.port_type)),
                        }
                        for pname, p in tmp.outputs.items()
                    },
                })
        return {"categories": categories, "category_order": CATEGORY_ORDER}

    def add_node(self, params: dict) -> dict:
        """Add a node to the graph. Returns the node's serialized state."""
        type_name = params["type_name"]
        cls = NODE_REGISTRY.get(type_name)
        if cls is None:
            raise ValueError(f"Unknown node type: {type_name}")
        node = cls()
        self.graph.add_node(node)
        return self._node_to_dict(node)

    def remove_node(self, params: dict) -> dict:
        """Remove a node from the graph."""
        node_id = params["node_id"]
        self.graph.remove_node(node_id)
        return {"ok": True}

    def connect(self, params: dict) -> dict:
        """Connect two ports."""
        conn = self.graph.add_connection(
            params["from_node"], params["from_port"],
            params["to_node"], params["to_port"],
        )
        if conn is None:
            raise ValueError("Connection failed (cycle or invalid)")
        return {"ok": True}

    def disconnect(self, params: dict) -> dict:
        """Disconnect two ports."""
        self.graph.remove_connection(
            params["from_node"], params["from_port"],
            params["to_node"], params["to_port"],
        )
        return {"ok": True}

    def set_input(self, params: dict) -> dict:
        """Set a config input value on a node."""
        node = self.graph.get_node(params["node_id"])
        if node is None:
            raise ValueError(f"Node not found: {params['node_id']}")
        port_name = params["port_name"]
        if port_name not in node.inputs:
            raise ValueError(f"Port not found: {port_name}")
        node.inputs[port_name].default_value = params["value"]
        return {"ok": True}

    def get_node(self, params: dict) -> dict:
        """Get a node's current state."""
        node = self.graph.get_node(params["node_id"])
        if node is None:
            raise ValueError(f"Node not found: {params['node_id']}")
        return self._node_to_dict(node)

    def get_graph(self, params: dict) -> dict:
        """Get the full graph state."""
        nodes = {}
        for nid, node in self.graph.nodes.items():
            nodes[nid] = self._node_to_dict(node)
        connections = []
        for conn in self.graph.connections:
            connections.append({
                "from_node": conn.from_node_id,
                "from_port": conn.from_port,
                "to_node": conn.to_node_id,
                "to_port": conn.to_port,
            })
        return {"nodes": nodes, "connections": connections}

    def execute(self, params: dict) -> dict:
        """Execute the graph and return output summaries."""
        try:
            import torch
            with torch.no_grad():
                outputs, terminal_lines = self.graph.execute()
        except Exception as exc:
            return {"error": str(exc), "terminal": []}

        self._last_outputs = outputs

        # Summarise outputs per node
        summaries = {}
        for node_id, node_outputs in outputs.items():
            summaries[node_id] = {
                pname: self._summarise_output(val)
                for pname, val in node_outputs.items()
            }
        return {
            "outputs": summaries,
            "terminal": terminal_lines,
        }

    def clear(self, params: dict) -> dict:
        """Clear the graph."""
        self.graph = Graph()
        self._last_outputs = {}
        return {"ok": True}

    # ── Dispatch ─────────────────────────────────────────────────────────

    _METHODS: dict[str, str] = {
        "get_registry": "get_registry",
        "add_node": "add_node",
        "remove_node": "remove_node",
        "connect": "connect",
        "disconnect": "disconnect",
        "set_input": "set_input",
        "get_node": "get_node",
        "get_graph": "get_graph",
        "execute": "execute",
        "clear": "clear",
    }

    def dispatch(self, method: str, params: dict) -> Any:
        handler_name = self._METHODS.get(method)
        if handler_name is None:
            raise ValueError(f"Unknown method: {method}")
        return getattr(self, handler_name)(params or {})


# ── WebSocket handler ────────────────────────────────────────────────────

server = NodeToolServer()


async def handler(websocket):
    server._clients.add(websocket)
    remote = websocket.remote_address
    print(f"[WS] Client connected: {remote}")
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None,
                }))
                continue

            req_id = msg.get("id")
            method = msg.get("method", "")
            params = msg.get("params", {})

            try:
                result = server.dispatch(method, params)
                if req_id is not None:
                    await websocket.send(json.dumps({
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": req_id,
                    }))
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[WS] Error in {method}: {exc}")
                if req_id is not None:
                    await websocket.send(json.dumps({
                        "jsonrpc": "2.0",
                        "error": {"code": -32000, "message": str(exc), "data": tb},
                        "id": req_id,
                    }))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        server._clients.discard(websocket)
        print(f"[WS] Client disconnected: {remote}")


async def main(host: str, port: int):
    print(f"[NodeTool Server] Listening on ws://{host}:{port}")
    print(f"[NodeTool Server] {len(NODE_REGISTRY)} nodes registered")
    async with serve(handler, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NodeTool WebSocket server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9800)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
