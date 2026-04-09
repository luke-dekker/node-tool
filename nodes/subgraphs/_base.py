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
        """Recursively inline the inner graph into the parent export.

        Strategy: walk the inner graph in topological order, call each inner
        node's export(), and concatenate the resulting lines. Inner variable
        names get a per-instance prefix so they can't collide with parent
        names. Boundary ports are rebound to the caller-supplied variable
        names from the parent's iv/ov.

        The result is a flat sequence of statements that, when emitted into
        the parent's def main() (or class.forward), produces exactly the same
        computation as a non-subgraphed equivalent.
        """
        sf = self._subgraph_file
        if sf is None:
            return [], [f"# {self.label}: no subgraph file bound"]

        try:
            inner_graph = sf.build_inner_graph()
        except Exception as exc:
            return [], [f"# {self.label}: failed to build inner graph: {exc}"]

        # Per-instance prefix so multiple subgraph drops don't collide
        prefix = f"_sg{self.safe_id}_"

        # Build connection lookup for the inner graph
        inner_conn: dict[tuple[str, str], tuple[str, str]] = {}
        for c in inner_graph.connections:
            inner_conn[(c.to_node_id, c.to_port)] = (c.from_node_id, c.from_port)

        # Boundary mappings keyed by (inner_node_id, inner_port)
        boundary_in: dict[tuple[str, str], str | None] = {}
        for ext in sf.external_inputs:
            boundary_in[(ext.inner_node, ext.inner_port)] = iv.get(ext.name)
        boundary_out: dict[tuple[str, str], str] = {}
        for ext in sf.external_outputs:
            ext_var = ov.get(ext.name)
            if ext_var:
                boundary_out[(ext.inner_node, ext.inner_port)] = ext_var

        # Per-inner-node output variable map (inner_id, port) → emitted name
        var_map: dict[tuple[str, str], str] = {}
        from collections import defaultdict
        port_counters: dict[str, int] = defaultdict(int)

        def assign_inner_out_vars(inner_node) -> dict[str, str]:
            out_vars: dict[str, str] = {}
            base = prefix + (inner_node.type_name.split("_", 1)[-1]
                             if "_" in inner_node.type_name else inner_node.type_name)
            for port_name in inner_node.outputs:
                if port_name == "__terminal__":
                    continue
                key = (inner_node.id, port_name)
                # Boundary output: use the parent-supplied variable name verbatim
                if key in boundary_out:
                    out_vars[port_name] = boundary_out[key]
                    var_map[key] = boundary_out[key]
                    continue
                # Otherwise generate a fresh prefixed name
                vname = f"{base}_{port_counters[base]}"
                port_counters[base] += 1
                out_vars[port_name] = vname
                var_map[key] = vname
            return out_vars

        all_imports: list[str] = []
        all_lines: list[str] = [f"# >>> subgraph: {sf.name}"]
        for inner_id in inner_graph.topological_order():
            inner_node = inner_graph.nodes[inner_id]

            # Build in_vars for the inner node
            in_vars: dict[str, str | None] = {}
            for port_name in inner_node.inputs:
                key = (inner_id, port_name)
                if key in boundary_in:
                    in_vars[port_name] = boundary_in[key]
                elif key in inner_conn:
                    upstream = inner_conn[key]
                    in_vars[port_name] = var_map.get(upstream)
                else:
                    in_vars[port_name] = None

            out_vars = assign_inner_out_vars(inner_node)

            try:
                imports, lines = inner_node.export(in_vars, out_vars)
            except Exception as exc:
                imports, lines = [], [f"# [{inner_node.label}] inner export error: {exc}"]

            all_imports.extend(imports)
            all_lines.append(f"# {inner_node.label}")
            all_lines.extend(lines)

        all_lines.append(f"# <<< subgraph: {sf.name}")
        return all_imports, all_lines
