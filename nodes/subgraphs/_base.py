"""SubgraphNode — runs a saved subgraph as a single node in a parent graph.

Each .subgraph.json file in the subgraphs/ directory becomes a dynamically
generated subclass of SubgraphNode at app startup. The subclass exposes the
manifest's external_inputs/outputs as its own ports, holds the inner Graph as
self._inner_graph, and on execute() runs the inner graph with the parent's
input values pushed onto the boundary nodes.

Parameter discovery: SubgraphNode.execute() collects the inner graph's layer
modules into an nn.ModuleList stored as self._layer. GraphAsModule already
checks every node for a `_layer` attribute, so the parent training loop sees
all inner parameters with no recursive walk needed.
"""
from __future__ import annotations
from typing import Any, ClassVar
from core.node import BaseNode, PortType
from core.subgraph import SubgraphFile


class SubgraphNode(BaseNode):
    """Base class for all generated subgraph nodes.

    Subclasses set _subgraph_file (the loaded SubgraphFile) as a class attribute.
    The dynamic factory in nodes/subgraphs/__init__.py creates one subclass per
    .subgraph.json file at app startup.
    """

    # Set by the subclass factory
    _subgraph_file: ClassVar[SubgraphFile | None] = None

    def __init__(self):
        # Inner graph state (built lazily on first execute since module imports
        # may not be settled at __init__ time)
        self._inner_graph = None
        super().__init__()

    def _setup_ports(self) -> None:
        sf = self._subgraph_file
        if sf is None:
            return
        for ext in sf.external_inputs:
            self.add_input(ext.name, ext.type, default=None,
                           description=f"-> {ext.inner_port} on inner node")
        for ext in sf.external_outputs:
            self.add_output(ext.name, ext.type,
                            description=f"<- {ext.inner_port} on inner node")

    def _ensure_inner_graph(self) -> None:
        if self._inner_graph is None and self._subgraph_file is not None:
            self._inner_graph = self._subgraph_file.build_inner_graph()

    def get_layers(self) -> list:
        """Return all nn.Module layers from inner nodes (recurses into nested
        subgraphs naturally because each inner SubgraphNode has its own
        get_layers/_layer)."""
        self._ensure_inner_graph()
        if self._inner_graph is None:
            return []
        import torch.nn as nn
        modules: list[nn.Module] = []
        for inner in self._inner_graph.nodes.values():
            layer = getattr(inner, "_layer", None)
            if isinstance(layer, nn.Module):
                modules.append(layer)
            mm = getattr(inner, "_mm_model", None)
            if isinstance(mm, nn.Module):
                modules.append(mm)
        return modules

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sf = self._subgraph_file
        if sf is None:
            return {}
        self._ensure_inner_graph()
        ig = self._inner_graph
        if ig is None:
            return {}

        # Push parent inputs onto the inner graph's boundary input ports.
        # Important: we set default_value, NOT the live execution input — Graph.execute
        # reads default_value when a port has no upstream connection, which is exactly
        # the case for boundary inputs (they're free in the inner graph).
        for ext in sf.external_inputs:
            inner_node = ig.nodes.get(ext.inner_node)
            if inner_node is None or ext.inner_port not in inner_node.inputs:
                continue
            inner_node.inputs[ext.inner_port].default_value = inputs.get(ext.name)

        # Run the inner graph
        try:
            stored, _terminal = ig.execute()
        except Exception as exc:
            return {ext.name: None for ext in sf.external_outputs} | {"__error__": str(exc)}

        # Read boundary outputs
        result: dict[str, Any] = {}
        for ext in sf.external_outputs:
            node_outs = stored.get(ext.inner_node, {})
            result[ext.name] = node_outs.get(ext.inner_port)

        # Refresh _layer so GraphAsModule sees inner parameters. Wrap in a ModuleList
        # so it counts as one nn.Module attribute.
        try:
            import torch.nn as nn
            inner_layers = self.get_layers()
            if inner_layers:
                self._layer = nn.ModuleList(inner_layers)
        except Exception:
            pass

        return result

    def export(self, iv, ov):
        """Subgraphs don't currently flatten on export — they emit a TODO stub.
        Roadmap: in script-mode export, recursively inline the inner graph;
        in class-mode export, generate a nested helper class.
        """
        sf = self._subgraph_file
        name = sf.name if sf else "Subgraph"
        lines = [
            f"# TODO: subgraph {name!r} export — recursive inlining not yet supported",
        ]
        for ext_name, var in ov.items():
            lines.append(f"{var} = None  # subgraph output: {ext_name}")
        return [], lines
