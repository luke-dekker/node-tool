"""GraphAsModule — wraps a Graph as an nn.Module that runs the DAG on forward().

Design principle: the graph IS the model. Training reuses graph.execute() directly,
with the current batch's tensors injected into dataset node outputs (or legacy
BatchInput outputs) before the topological traversal runs. Injection points are
any node with a DATALOADER output + x/label TENSOR outputs.

Usage:
    model = GraphAsModule(graph, output_node_id, output_port="tensor_in")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for batch in dataloader:
        logits = model(batch)           # batch is a dict or tensor
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

from core.graph import Graph
from core.port_types import PortType, PortTypeRegistry


def _collect_layer_modules(graph: Graph) -> dict[str, nn.Module]:
    """Walk every node in the graph and return {node_id: nn.Module} for any
    node that exposes a `_layer` attribute containing an nn.Module.

    This is the only piece of code that knows how layer nodes store their modules.
    All PyTorch layer nodes follow the convention `self._layer = <nn.Module>`.
    """
    modules: dict[str, nn.Module] = {}
    for node_id, node in graph.nodes.items():
        layer = getattr(node, "_layer", None)
        if isinstance(layer, nn.Module):
            modules[node_id] = layer
        # Also pick up multimodal models — they hold encoders/projections internally
        mm_model = getattr(node, "_mm_model", None)
        if isinstance(mm_model, nn.Module):
            modules[node_id + "_mm"] = mm_model
    return modules


class GraphAsModule(nn.Module):
    """Makes a Graph behave like an nn.Module for training.

    The forward() method:
      1. Runs all BatchInput nodes once to materialize their initial outputs
      2. Overrides those outputs with the current training batch
      3. Calls graph.execute() to run the DAG with gradients enabled
      4. Returns the tensor at (output_node_id, output_port)
    """

    def __init__(self, graph: Graph, output_node_id: str, output_port: str = "tensor_in"):
        super().__init__()
        self.graph = graph
        self.output_node_id = output_node_id
        self.output_port = output_port

        # Prime all layer nodes so their internal nn.Modules exist.
        # Running the graph once (under no_grad) triggers the lazy _get_layer() calls.
        with torch.no_grad():
            try:
                graph.execute()
            except Exception:
                pass

        # Collect layer modules into a ModuleDict so self.parameters() sees them
        self._layer_modules = nn.ModuleDict()
        for key, mod in _collect_layer_modules(graph).items():
            safe_key = key.replace("-", "_")
            self._layer_modules[safe_key] = mod

    # ------------------------------------------------------------------ forward

    def forward(self, batch: Any) -> torch.Tensor | None:
        """Run the graph with `batch` injected at BatchInput nodes.

        `batch` can be:
          - a dict with 'data' and 'label' keys (multimodal collate format)
          - a (x, y) tuple
          - a single tensor
        """
        override = self._batch_to_port_values(batch)

        # Refresh the layer module collection in case new nodes were added
        for key, mod in _collect_layer_modules(self.graph).items():
            safe_key = key.replace("-", "_")
            if safe_key not in self._layer_modules:
                self._layer_modules[safe_key] = mod

        outputs = self._execute_with_overrides(override)

        node_out = outputs.get(self.output_node_id, {})
        # The target is the tensor at `output_port` — but if the caller points at
        # a node's INPUT port, walk the connection back to the upstream node's output.
        if self.output_port in node_out:
            return node_out[self.output_port]
        # Walk back through the connection
        for c in self.graph.connections:
            if c.to_node_id == self.output_node_id and c.to_port == self.output_port:
                src = outputs.get(c.from_node_id, {})
                return src.get(c.from_port)
        return None

    # ------------------------------------------------------------------ helpers

    def _batch_to_port_values(self, batch: Any) -> dict[tuple[str, str], Any]:
        """Map `batch` onto {(injection_node_id, port_name): tensor}.

        Injection points are discovered by looking for nodes that have BOTH
        a DATALOADER output port AND x/label tensor output ports — these are
        the new-style dataset nodes that serve as both data sources and
        injection points. Legacy BatchInput nodes are also supported for
        backward compat.

        The batch is unpacked as:
          - dict with 'data': multimodal collate format → per-modality tensors
          - (x, y) tuple: standard classification batch
          - single tensor: unsupervised / autoencoder
        """
        from core.port_types import PortType, PortTypeRegistry
        overrides: dict[tuple[str, str], Any] = {}

        # Find injection points: dataset nodes with DATALOADER + TENSOR outputs,
        # OR legacy BatchInput nodes
        injection_nodes: list[str] = []
        for nid, n in self.graph.nodes.items():
            if n.type_name == "batch_input":
                injection_nodes.append(nid)
                continue
            # New-style: any node with a DATALOADER output + x/label TENSOR outputs
            has_dl = any(p.port_type == PortType.DATALOADER for p in n.outputs.values())
            has_x  = "x" in n.outputs and n.outputs["x"].port_type == PortType.TENSOR
            if has_dl and has_x:
                injection_nodes.append(nid)

        if not injection_nodes:
            return overrides

        # Parse the batch into a dict of {port_name: tensor}
        port_values: dict[str, Any] = {}
        if isinstance(batch, dict) and "data" in batch:
            for port, val in batch["data"].items():
                port_values[port] = val
            port_values["label"] = batch.get("label")
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            port_values["x"] = batch[0]
            port_values["label"] = batch[1]
        else:
            port_values["x"] = batch

        # Apply to every injection node — only override ports that exist on the node
        for nid in injection_nodes:
            node = self.graph.nodes[nid]
            for port_name, val in port_values.items():
                if port_name in node.outputs:
                    overrides[(nid, port_name)] = val

        return overrides

    def _execute_with_overrides(
        self, overrides: dict[tuple[str, str], Any]
    ) -> dict[str, dict[str, Any]]:
        """Like graph.execute(), but with pre-set values for specific (node_id, port)
        pairs. Skips calling execute() on nodes whose outputs are entirely overridden."""
        order = self.graph.topological_order()
        stored: dict[str, dict[str, Any]] = {}
        conn_map: dict[tuple[str, str], tuple[str, str]] = {
            (c.to_node_id, c.to_port): (c.from_node_id, c.from_port)
            for c in self.graph.connections
        }

        # Pre-populate stored with any override values so downstream lookups see them
        for (nid, port), val in overrides.items():
            stored.setdefault(nid, {})[port] = val

        for node_id in order:
            node = self.graph.nodes[node_id]

            # If this node is an injection point (BatchInput or dataset node)
            # and its outputs are already overridden with batch tensors,
            # skip its execute() — stored[node_id] already holds the values.
            if node_id in stored and (
                node.type_name == "batch_input"
                or ("x" in node.outputs and any(
                    p.port_type == PortType.DATALOADER for p in node.outputs.values()
                ))
            ):
                continue

            inputs: dict[str, Any] = {}
            for port_name, port in node.inputs.items():
                key = (node_id, port_name)
                if key in conn_map:
                    from_id, from_port = conn_map[key]
                    if from_id in stored and from_port in stored[from_id]:
                        raw = stored[from_id][from_port]
                        inputs[port_name] = PortTypeRegistry.coerce_value(port.port_type, raw) if raw is not None else None
                    else:
                        inputs[port_name] = port.default_value
                else:
                    inputs[port_name] = port.default_value

            try:
                outputs = node.execute(inputs)
            except Exception:
                outputs = {}
            # Merge rather than overwrite so override values are preserved
            stored.setdefault(node_id, {}).update(outputs or {})

        return stored

    # ------------------------------------------------------------------ train/eval

    def train(self, mode: bool = True):
        """Propagate training mode to all contained layer modules."""
        super().train(mode)
        for mod in self._layer_modules.values():
            mod.train(mode)
        return self

    def eval(self):
        return self.train(False)
