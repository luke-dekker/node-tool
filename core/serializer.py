"""Serializer — save/load a Graph to/from JSON."""
from __future__ import annotations
import json
from pathlib import Path
from core.graph import Graph
from nodes import NODE_REGISTRY


class Serializer:
    VERSION = 1

    SERIALIZABLE_TYPES = (int, float, bool, str, type(None))

    @staticmethod
    def _serialize_value(v):
        if isinstance(v, Serializer.SERIALIZABLE_TYPES):
            return v
        return None  # tensors, modules etc → null

    @classmethod
    def save(cls, graph: Graph, positions: dict[str, tuple], path: str) -> None:
        nodes = []
        for node_id, node in graph.nodes.items():
            nodes.append({
                "id": node.id,
                "type_name": node.type_name,
                "alias": node.alias,
                "pos": list(positions.get(node_id, [100, 100])),
                "inputs": {
                    k: cls._serialize_value(p.default_value)
                    for k, p in node.inputs.items()
                }
            })
        connections = [
            {
                "from_node": c.from_node_id,
                "from_port": c.from_port,
                "to_node": c.to_node_id,
                "to_port": c.to_port,
            }
            for c in graph.connections
        ]
        data = {"version": cls.VERSION, "nodes": nodes, "connections": connections}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> tuple[Graph, dict[str, list]]:
        data = json.loads(Path(path).read_text())
        graph = Graph()
        positions = {}
        skipped: list[str] = []
        for nd in data["nodes"]:
            cls_ = NODE_REGISTRY.get(nd["type_name"])
            if cls_ is None:
                skipped.append(nd["type_name"])
                continue
            node = cls_()
            node.id = nd["id"]
            # Restore alias if the saved graph has one; else add_node assigns
            # a fresh one so the node is never unnamed on the canvas.
            node.alias = nd.get("alias", "") or ""
            for k, v in nd.get("inputs", {}).items():
                if k in node.inputs and v is not None:
                    node.inputs[k].default_value = v
            graph.add_node(node)
            positions[node.id] = nd.get("pos", [100, 100])
        for c in data.get("connections", []):
            graph.add_connection(c["from_node"], c["from_port"], c["to_node"], c["to_port"])
        return graph, positions
