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
        # Plugin RPC registry — routes every prefix-matching method to the
        # plugin orchestrator that registered it via
        # `ctx.register_orchestrator(prefixes, factory)`. Lazy-built per-graph;
        # rebind_graph() refreshes the orchestrators' graph ref after clear/load.
        from core.plugins import OrchestratorRegistry
        import nodes as _nodes_pkg
        factories = (_nodes_pkg._plugin_ctx.orchestrator_factories
                     if getattr(_nodes_pkg, "_plugin_ctx", None) else [])
        self._plugin_registry = OrchestratorRegistry(self.graph, factories)

    def _get_agents_orchestrator(self):
        """Legacy accessor — kept so in-tree callers keep working during the
        registry migration. Prefer `self._plugin_registry.resolve('agent_')`.
        """
        orch = self._plugin_registry.resolve("agent_")
        if orch is None:
            from plugins.agents.agents_orchestrator import AgentOrchestrator
            orch = AgentOrchestrator(self.graph)
            self._plugin_registry._cache["agent_"] = orch
        orch.graph = self.graph
        return orch

    def _get_robotics_controller(self):
        orch = self._plugin_registry.resolve("robotics_")
        if orch is None:
            from plugins.robotics.robotics_controller import RoboticsController
            orch = RoboticsController()
            self._plugin_registry._cache["robotics_"] = orch
        return orch

    def _get_training_orchestrator(self):
        orch = self._plugin_registry.resolve("train_")
        if orch is None:
            from plugins.pytorch.training_orchestrator import TrainingOrchestrator
            orch = TrainingOrchestrator(self.graph)
            self._plugin_registry._cache["train_"] = orch
        orch.graph = self.graph
        return orch

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
                "dynamic_choices": getattr(port, "dynamic_choices", "") or "",
                "optional": bool(getattr(port, "optional", False)),
            }
        outputs = {}
        for pname, port in node.outputs.items():
            outputs[pname] = {
                "port_type": port.port_type,
                "color": list(PortTypeRegistry.get_color(port.port_type)),
                "description": getattr(port, "description", ""),
            }
        # Per-instance "show only these editable ports" filter, computed
        # from the node's CURRENT input values. Mega-consolidated nodes
        # override BaseNode.relevant_inputs to hide irrelevant fields based
        # on the chosen kind/op/mode. None means "show all editable ports".
        try:
            current = {pname: port.default_value for pname, port in node.inputs.items()}
            relevant = node.relevant_inputs(current)
        except Exception:
            relevant = None
        return {
            "id": node.id,
            "type_name": node.type_name,
            "label": node.label,
            "alias": getattr(node, "alias", ""),
            "category": node.category,
            "subcategory": getattr(node, "subcategory", ""),
            "description": node.description,
            "inputs": inputs,
            "outputs": outputs,
            "relevant_inputs": list(relevant) if relevant is not None else None,
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
                            "dynamic_choices": getattr(p, "dynamic_choices", "") or "",
                            "description": getattr(p, "description", ""),
                        }
                        for pname, p in tmp.inputs.items()
                    },
                    "outputs": {
                        pname: {
                            "port_type": p.port_type,
                            "color": list(PortTypeRegistry.get_color(p.port_type)),
                            "description": getattr(p, "description", ""),
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
        from_node = self.graph.get_node(params["from_node"])
        to_node = self.graph.get_node(params["to_node"])
        conn = self.graph.add_connection(
            params["from_node"], params["from_port"],
            params["to_node"], params["to_port"],
        )
        if conn is None:
            # Provide specific failure reason
            if from_node and to_node:
                out_type = from_node.outputs.get(params["from_port"])
                in_type = to_node.inputs.get(params["to_port"])
                if out_type and in_type:
                    raise ValueError(
                        f"Cannot connect {out_type.port_type} -> {in_type.port_type} "
                        f"(incompatible types)")
            raise ValueError("Connection failed (cycle, missing port, or type mismatch)")
        return {"ok": True}

    def disconnect(self, params: dict) -> dict:
        """Disconnect two ports."""
        self.graph.remove_connection(
            params["from_node"], params["from_port"],
            params["to_node"], params["to_port"],
        )
        return {"ok": True}

    def set_input(self, params: dict) -> dict:
        """Set a config input value on a node.

        Returns the updated node dict so callers can refresh `relevant_inputs`
        after a kind/op/mode change without a second RPC.
        """
        node = self.graph.get_node(params["node_id"])
        if node is None:
            raise ValueError(f"Node not found: {params['node_id']}")
        port_name = params["port_name"]
        if port_name not in node.inputs:
            raise ValueError(f"Port not found: {port_name}")
        node.inputs[port_name].default_value = params["value"]
        return self._node_to_dict(node)

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
                outputs, terminal_lines, errors = self.graph.execute()
        except Exception as exc:
            return {"error": str(exc), "terminal": [], "errors": {}}

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
            "errors": errors,
        }

    def clear(self, params: dict) -> dict:
        """Clear the graph."""
        self.graph = Graph()
        self._last_outputs = {}
        self._plugin_registry.rebind_graph(self.graph)
        return {"ok": True}

    def save_graph(self, params: dict) -> dict:
        """Save the graph to a JSON file."""
        path = params.get("path", "")
        if not path:
            raise ValueError("No path provided")
        from core.serializer import Serializer
        # Collect positions from params or use empty dict
        positions = params.get("positions", {})
        # Convert positions from {id: [x,y]} to {id: (x,y)}
        pos_tuples = {k: tuple(v) for k, v in positions.items()}
        Serializer.save(self.graph, pos_tuples, path)
        return {"ok": True, "path": path, "nodes": len(self.graph.nodes)}

    def load_graph(self, params: dict) -> dict:
        """Load a graph from a JSON file. Replaces the current graph."""
        path = params.get("path", "")
        if not path:
            raise ValueError("No path provided")
        from core.serializer import Serializer
        graph, positions = Serializer.load(path)
        self.graph = graph
        self._last_outputs = {}
        self._plugin_registry.rebind_graph(self.graph)
        # Build full response with all nodes and connections
        nodes = {}
        for nid, node in graph.nodes.items():
            nodes[nid] = self._node_to_dict(node)
        connections = []
        for conn in graph.connections:
            connections.append({
                "from_node": conn.from_node_id,
                "from_port": conn.from_port,
                "to_node": conn.to_node_id,
                "to_port": conn.to_port,
            })
        # Convert positions back to {id: [x,y]}
        pos_out = {k: list(v) for k, v in positions.items()}
        return {"nodes": nodes, "connections": connections, "positions": pos_out}

    def get_marker_groups(self, params: dict) -> dict:
        """Discover training marker groups from the graph.

        Scans for InputMarker (A) and TrainMarker (B) nodes, returns groups
        with their modalities so the frontend can build dataset config widgets.
        """
        from core.node import MarkerRole
        groups: dict[str, dict] = {}
        for node in self.graph.nodes.values():
            if node.marker_role == MarkerRole.INPUT:
                g = str(node.inputs["group"].default_value or "task_1")
                m = str(node.inputs["modality"].default_value or "x")
                if g not in groups:
                    groups[g] = {"modalities": [], "has_output": False}
                groups[g]["modalities"].append(m)
            elif node.marker_role == MarkerRole.TRAIN_TARGET:
                g = str(node.inputs["group"].default_value or "task_1")
                if g not in groups:
                    groups[g] = {"modalities": [], "has_output": False}
                groups[g]["has_output"] = True
        return {"groups": groups}

    def get_templates(self, params: dict) -> dict:
        """Return available graph templates."""
        try:
            from templates import get_templates
            result = []
            for label, description, builder in get_templates():
                result.append({"label": label, "description": description})
            return {"templates": result}
        except Exception as exc:
            return {"templates": [], "error": str(exc)}

    def load_template(self, params: dict) -> dict:
        """Load a template by label, replacing the current graph."""
        label = params.get("label", "")
        try:
            from templates import get_templates
            for t_label, t_desc, builder in get_templates():
                if t_label == label:
                    self.graph = Graph()
                    self._last_outputs = {}
                    self._plugin_registry.rebind_graph(self.graph)
                    positions = builder(self.graph)
                    # Return full graph state
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
                    pos_out = {k: list(v) for k, v in positions.items()} if positions else {}
                    return {"nodes": nodes, "connections": connections, "positions": pos_out}
            raise ValueError(f"Template not found: {label}")
        except Exception as exc:
            raise ValueError(f"Template load failed: {exc}")

    def export_code(self, params: dict) -> dict:
        """Export the graph as a Python script."""
        try:
            from core.exporter import export_graph
            code = export_graph(self.graph)
            return {"code": code}
        except Exception as exc:
            return {"code": f"# Export failed: {exc}"}

    def serialize_graph(self, params: dict) -> dict:
        """Return the graph as a JSON-serializable dict (matches Serializer.save format).

        Used by browser frontends that download the result rather than writing
        to a server-side path. Positions are passed in by the client since the
        server doesn't track them.
        """
        positions = params.get("positions", {})
        nodes = []
        for node_id, node in self.graph.nodes.items():
            nodes.append({
                "id": node.id,
                "type_name": node.type_name,
                "pos": list(positions.get(node_id, [100, 100])),
                "inputs": {
                    k: self._safe_value(p.default_value)
                    for k, p in node.inputs.items()
                },
            })
        connections = [
            {
                "from_node": c.from_node_id,
                "from_port": c.from_port,
                "to_node": c.to_node_id,
                "to_port": c.to_port,
            }
            for c in self.graph.connections
        ]
        return {"version": 1, "nodes": nodes, "connections": connections}

    def deserialize_graph(self, params: dict) -> dict:
        """Replace the graph from an in-memory JSON dict (browser upload path)."""
        data = params.get("data") or {}
        from core.graph import Graph
        graph = Graph()
        positions: dict[str, list] = {}
        for nd in data.get("nodes", []):
            cls = NODE_REGISTRY.get(nd["type_name"])
            if cls is None:
                continue
            node = cls()
            node.id = nd["id"]
            for k, v in nd.get("inputs", {}).items():
                if k in node.inputs and v is not None:
                    node.inputs[k].default_value = v
            graph.add_node(node)
            positions[node.id] = nd.get("pos", [100, 100])
        for c in data.get("connections", []):
            graph.add_connection(c["from_node"], c["from_port"],
                                 c["to_node"], c["to_port"])
        self.graph = graph
        self._last_outputs = {}
        self._plugin_registry.rebind_graph(self.graph)
        nodes = {nid: self._node_to_dict(n) for nid, n in graph.nodes.items()}
        conns = [
            {"from_node": c.from_node_id, "from_port": c.from_port,
             "to_node": c.to_node_id, "to_port": c.to_port}
            for c in graph.connections
        ]
        return {"nodes": nodes, "connections": conns, "positions": positions}

    def get_plugin_panels(self, params: dict) -> dict:
        """Return the list of registered plugin panel names (legacy + spec)."""
        try:
            from nodes import _plugin_ctx
            if _plugin_ctx:
                legacy = [label for label, _ in _plugin_ctx.panels]
                specs  = [label for label, _ in _plugin_ctx.panel_specs]
                # Spec-driven panels shadow legacy ones of the same label.
                return {"panels": specs + [l for l in legacy if l not in specs]}
        except Exception:
            pass
        return {"panels": []}

    def get_panel_specs(self, params: dict) -> dict:
        """Return every plugin-registered PanelSpec as a serialized dict.

        Frontends render these natively — see FRONTEND_PROTOCOL.md and
        core/panel.py for the schema.
        """
        try:
            from nodes import _plugin_ctx
            if _plugin_ctx:
                return {
                    "panels": {
                        label: spec.to_dict()
                        for label, spec in _plugin_ctx.panel_specs
                    }
                }
        except Exception:
            pass
        return {"panels": {}}

    # ── Plugin orchestrator passthrough ─────────────────────────────────

    def train_start(self, params: dict) -> dict:
        return self._get_training_orchestrator().start(params)

    def train_pause(self, params: dict) -> dict:
        return self._get_training_orchestrator().pause(params)

    def train_resume(self, params: dict) -> dict:
        return self._get_training_orchestrator().resume(params)

    def train_stop(self, params: dict) -> dict:
        return self._get_training_orchestrator().stop(params)

    def train_save_model(self, params: dict) -> dict:
        return self._get_training_orchestrator().save_model(params)

    def get_training_state(self, params: dict) -> dict:
        return self._get_training_orchestrator().state()

    def get_training_losses(self, params: dict) -> dict:
        return self._get_training_orchestrator().losses()

    def drain_training_logs(self, params: dict) -> dict:
        return self._get_training_orchestrator().drain_logs()

    # ── Agents plugin ───────────────────────────────────────────────────

    def agent_list_local_models(self, params: dict) -> dict:
        return self._get_agents_orchestrator().list_local_models(params)

    def agent_ping_backend(self, params: dict) -> dict:
        return self._get_agents_orchestrator().ping_backend(params)

    def agent_list_agent_nodes(self, params: dict) -> dict:
        return self._get_agents_orchestrator().list_agent_nodes(params)

    def agent_start(self, params: dict) -> dict:
        return self._get_agents_orchestrator().start(params)

    def agent_stop(self, params: dict) -> dict:
        return self._get_agents_orchestrator().stop(params)

    def get_agent_state(self, params: dict) -> dict:
        return self._get_agents_orchestrator().get_state(params)

    def agent_drain_tokens(self, params: dict) -> dict:
        return self._get_agents_orchestrator().drain_tokens(params)

    def agent_drain_logs(self, params: dict) -> dict:
        return self._get_agents_orchestrator().drain_logs(params)

    # ── Robotics plugin ─────────────────────────────────────────────────

    def robotics_list_ports(self, params: dict) -> dict:
        return self._get_robotics_controller().list_ports(params)

    def robotics_connect(self, params: dict) -> dict:
        return self._get_robotics_controller().connect(params)

    def robotics_disconnect(self, params: dict) -> dict:
        return self._get_robotics_controller().disconnect(params)

    def robotics_send(self, params: dict) -> dict:
        return self._get_robotics_controller().send(params)

    def get_robotics_state(self, params: dict) -> dict:
        return self._get_robotics_controller().status()

    def get_robotics_log(self, params: dict) -> dict:
        return self._get_robotics_controller().log()

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
        "save_graph": "save_graph",
        "load_graph": "load_graph",
        "export_code": "export_code",
        "get_marker_groups": "get_marker_groups",
        "get_templates": "get_templates",
        "load_template": "load_template",
        "get_plugin_panels": "get_plugin_panels",
        "get_panel_specs": "get_panel_specs",
        "serialize_graph": "serialize_graph",
        "deserialize_graph": "deserialize_graph",
        # Training (delegated to the pytorch plugin's orchestrator)
        "train_start":           "train_start",
        "train_pause":           "train_pause",
        "train_resume":          "train_resume",
        "train_stop":            "train_stop",
        "train_save_model":      "train_save_model",
        "get_training_state":    "get_training_state",
        "get_training_losses":   "get_training_losses",
        "drain_training_logs":   "drain_training_logs",
        # Robotics
        "robotics_list_ports":  "robotics_list_ports",
        "robotics_connect":     "robotics_connect",
        "robotics_disconnect":  "robotics_disconnect",
        "robotics_send":        "robotics_send",
        "get_robotics_state":   "get_robotics_state",
        "get_robotics_log":     "get_robotics_log",
        # Agents (delegated to plugins/agents/agents_orchestrator.py)
        "agent_list_local_models": "agent_list_local_models",
        "agent_ping_backend":      "agent_ping_backend",
        "agent_list_agent_nodes":  "agent_list_agent_nodes",
        "agent_start":             "agent_start",
        "agent_stop":              "agent_stop",
        "get_agent_state":         "get_agent_state",
        "agent_drain_tokens":      "agent_drain_tokens",
        "agent_drain_logs":        "agent_drain_logs",
    }

    def dispatch(self, method: str, params: dict) -> Any:
        # Core methods (graph CRUD, registry, execute) live on this class.
        # Plugin-owned methods (train_*, agent_*, robotics_*) route through
        # the OrchestratorRegistry via longest-prefix match.
        from core.plugins import OrchestratorRegistry
        handler_name = self._METHODS.get(method)
        if handler_name is not None:
            return getattr(self, handler_name)(params or {})
        result = self._plugin_registry.try_dispatch(method, params or {})
        if result is not OrchestratorRegistry._UNHANDLED:
            return result
        raise ValueError(f"Unknown method: {method}")


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
    # Report plugin loading status
    try:
        from nodes import _plugin_ctx
        if _plugin_ctx:
            print(f"[NodeTool Server] Plugins loaded: {len(_plugin_ctx.node_classes)} plugin nodes, "
                  f"{len(_plugin_ctx.panels)} panels")
        else:
            print("[NodeTool Server] No plugins loaded")
    except Exception:
        print("[NodeTool Server] Plugin context not available")

    # Report port types
    all_types = PortTypeRegistry.all_types()
    print(f"[NodeTool Server] {len(all_types)} port types registered: "
          f"{', '.join(sorted(all_types.keys()))}")

    print(f"[NodeTool Server] {len(NODE_REGISTRY)} nodes registered")
    print(f"[NodeTool Server] Listening on ws://{host}:{port}")
    print(f"[NodeTool Server] RPC methods: {', '.join(sorted(NodeToolServer._METHODS.keys()))}")
    async with serve(handler, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NodeTool WebSocket server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9800)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
