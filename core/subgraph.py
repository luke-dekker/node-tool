"""Subgraph file format and manifest helpers.

A subgraph is a saved Graph plus a manifest describing which inner input/output
ports are exposed as the subgraph's external ports. The format extends the
existing Graph JSON so the same Serializer can read the inner graph back.

File extension: .subgraph.json (distinct from regular .json graph saves so
they don't get loaded into the canvas accidentally).

Manifest schema:
    {
      "version": 1,
      "kind": "subgraph",
      "manifest": {
          "name":        "MyEncoder",
          "description": "...",
          "external_inputs":  [{"name": "...", "type": "TENSOR",
                                "inner_node": "<uuid>", "inner_port": "tensor_in"}],
          "external_outputs": [{"name": "...", "type": "TENSOR",
                                "inner_node": "<uuid>", "inner_port": "tensor_out"}]
      },
      "nodes":       [...same as Serializer...],
      "connections": [...same as Serializer...]
    }
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from core.graph import Graph
from core.node import PortType


SUBGRAPH_VERSION = 1
SUBGRAPHS_DIR = Path(__file__).parent.parent / "subgraphs"


@dataclass
class ExternalPort:
    """Declares one inner port as the boundary of a subgraph."""
    name:       str       # external label seen by the parent graph
    type:       PortType  # PortType for the external port
    inner_node: str       # uuid of the inner node
    inner_port: str       # port name on that inner node

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type,  # type is already a string
                "inner_node": self.inner_node, "inner_port": self.inner_port}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExternalPort":
        return cls(
            name=d["name"],
            type=d["type"],  # port type is now a string, not an enum
            inner_node=d["inner_node"],
            inner_port=d["inner_port"],
        )


@dataclass
class SubgraphFile:
    """In-memory representation of a .subgraph.json file."""
    name:             str
    description:      str
    external_inputs:  list[ExternalPort] = field(default_factory=list)
    external_outputs: list[ExternalPort] = field(default_factory=list)
    nodes:            list[dict] = field(default_factory=list)   # raw node JSON
    connections:      list[dict] = field(default_factory=list)   # raw conn JSON
    positions:        dict[str, list[float]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".subgraph.json")
        data = {
            "version": SUBGRAPH_VERSION,
            "kind": "subgraph",
            "manifest": {
                "name":        self.name,
                "description": self.description,
                "external_inputs":  [p.to_dict() for p in self.external_inputs],
                "external_outputs": [p.to_dict() for p in self.external_outputs],
            },
            "nodes":       self.nodes,
            "connections": self.connections,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SubgraphFile":
        data = json.loads(Path(path).read_text())
        if data.get("kind") != "subgraph":
            raise ValueError(f"Not a subgraph file: {path}")
        manifest = data.get("manifest", {})
        nodes = data.get("nodes", [])
        positions = {n["id"]: n.get("pos", [100, 100]) for n in nodes}
        return cls(
            name=manifest.get("name", Path(path).stem),
            description=manifest.get("description", ""),
            external_inputs=[ExternalPort.from_dict(p)
                             for p in manifest.get("external_inputs", [])],
            external_outputs=[ExternalPort.from_dict(p)
                              for p in manifest.get("external_outputs", [])],
            nodes=nodes,
            connections=data.get("connections", []),
            positions=positions,
        )

    def build_inner_graph(self) -> Graph:
        """Reconstruct the inner Graph from the saved nodes/connections.

        Uses the same NODE_REGISTRY lookup as Serializer.load — but doesn't share
        code with it because the file shape is slightly different (we have a
        manifest section). Pasted-and-tweaked is cleaner here than parameterizing.
        """
        from nodes import NODE_REGISTRY
        graph = Graph()
        for nd in self.nodes:
            cls_ = NODE_REGISTRY.get(nd["type_name"])
            if cls_ is None:
                raise KeyError(f"Unknown node type in subgraph: {nd['type_name']}")
            node = cls_()
            node.id = nd["id"]
            for k, v in nd.get("inputs", {}).items():
                if k in node.inputs and v is not None:
                    node.inputs[k].default_value = v
            graph.add_node(node)
        for c in self.connections:
            graph.add_connection(c["from_node"], c["from_port"],
                                 c["to_node"], c["to_port"])
        return graph


# ── Boundary detection ────────────────────────────────────────────────────────

def detect_boundary_ports(
    parent_graph: Graph,
    selected_node_ids: set[str],
) -> tuple[list[ExternalPort], list[ExternalPort]]:
    """Walk a parent graph's connections to find which ports of the selection
    cross the boundary into nodes outside the selection. Returns
    (external_inputs, external_outputs).

    External inputs are *inner* input ports that are connected from a node
    OUTSIDE the selection — these become the subgraph's input ports.

    External outputs are *inner* output ports that are connected to a node
    OUTSIDE the selection (or are leaf-style: completely unconnected and the
    inner node has no other downstream consumers — leaves get auto-promoted so
    the subgraph can be used as a model whose output you wire forward).

    External ports are auto-named `{inner_node_label}_{port_name}` with a numeric
    suffix on collisions.
    """
    external_inputs: list[ExternalPort] = []
    external_outputs: list[ExternalPort] = []
    used_names: set[str] = set()

    def unique(base: str) -> str:
        if base not in used_names:
            used_names.add(base)
            return base
        i = 2
        while f"{base}_{i}" in used_names:
            i += 1
        used_names.add(f"{base}_{i}")
        return f"{base}_{i}"

    def short(node_id: str) -> str:
        node = parent_graph.nodes.get(node_id)
        if node is None:
            return "node"
        # Strip type_name prefixes (pt_, np_, etc.) for cleaner names
        s = node.type_name
        for p in ("pt_", "np_", "pd_", "sk_", "sp_", "viz_"):
            if s.startswith(p):
                s = s[len(p):]
                break
        return s.replace("_node", "")

    # Inputs: any inner-node input port whose upstream lives OUTSIDE the selection
    for c in parent_graph.connections:
        if c.to_node_id in selected_node_ids and c.from_node_id not in selected_node_ids:
            inner_node = parent_graph.nodes.get(c.to_node_id)
            if inner_node is None or c.to_port not in inner_node.inputs:
                continue
            port_type = inner_node.inputs[c.to_port].port_type
            ext_name = unique(f"{short(c.to_node_id)}_{c.to_port}")
            external_inputs.append(ExternalPort(
                name=ext_name, type=port_type,
                inner_node=c.to_node_id, inner_port=c.to_port,
            ))

    # Outputs: any inner-node output port whose downstream is OUTSIDE the selection
    seen_outputs: set[tuple[str, str]] = set()
    for c in parent_graph.connections:
        if c.from_node_id in selected_node_ids and c.to_node_id not in selected_node_ids:
            key = (c.from_node_id, c.from_port)
            if key in seen_outputs:
                continue
            seen_outputs.add(key)
            inner_node = parent_graph.nodes.get(c.from_node_id)
            if inner_node is None or c.from_port not in inner_node.outputs:
                continue
            port_type = inner_node.outputs[c.from_port].port_type
            ext_name = unique(f"{short(c.from_node_id)}_{c.from_port}")
            external_outputs.append(ExternalPort(
                name=ext_name, type=port_type,
                inner_node=c.from_node_id, inner_port=c.from_port,
            ))

    # Leaf promotion: any selected node whose outputs are completely unconsumed
    # (inside or outside the selection) gets its main tensor output promoted as
    # an external output. This makes "select layers, pack" produce a usable
    # subgraph even when the user didn't draw a downstream wire yet.
    nodes_with_consumers: set[tuple[str, str]] = set()
    for c in parent_graph.connections:
        nodes_with_consumers.add((c.from_node_id, c.from_port))
    for node_id in selected_node_ids:
        node = parent_graph.nodes.get(node_id)
        if node is None:
            continue
        for port_name, port in node.outputs.items():
            if port_name == "__terminal__":
                continue
            if (node_id, port_name) in nodes_with_consumers:
                continue
            # Auto-promote popular tensor-style output names
            if port_name in ("tensor_out", "tensor", "output", "model",
                            "config", "result", "loss", "dataloader",
                            "predictions", "info"):
                ext_name = unique(f"{short(node_id)}_{port_name}")
                external_outputs.append(ExternalPort(
                    name=ext_name, type=port.port_type,
                    inner_node=node_id, inner_port=port_name,
                ))
                break  # one leaf output per node is enough

    return external_inputs, external_outputs
