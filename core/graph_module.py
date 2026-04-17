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
        # Cache the resulting outputs — injection-point nodes get their execute()
        # skipped during training forwards, so non-overridden outputs (dataset
        # metadata like vocab_size, task_id, etc.) need to survive from the
        # priming run so they keep flowing into downstream nodes.
        self._primed_outputs: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            try:
                primed, _, _ = graph.execute()
                if isinstance(primed, dict):
                    self._primed_outputs = {
                        nid: dict(vals) for nid, vals in primed.items()
                    }
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

        Two injection-point flavors are supported:
          1. Input markers (pt_input_marker) — each marker emits one tensor
             on its "tensor" output port, keyed by its `modality` input.
             Preferred path for new-style templates.
          2. Legacy dataset nodes (DATALOADER + x/label outputs) — the whole
             batch dict is splayed across the dataset node's output ports by
             column name, with an "x" fallback for templates that wire the
             legacy `x` port. Kept so DatasetNode-based templates keep
             working during the marker migration.

        Batch shape handling:
          - dict with 'data' key: legacy multimodal collate format
          - plain dict {col: tensor}: DatasetNode's _collate_manifest output
          - (x, y) tuple: standard classification batch
          - single tensor: unsupervised / autoencoder
        """
        from core.port_types import PortType, PortTypeRegistry
        overrides: dict[tuple[str, str], Any] = {}

        # ── Discover injection points ────────────────────────────────────
        marker_nodes: list[str] = []      # new-style: pt_input_marker
        legacy_nodes: list[str] = []      # old-style: dataset node / batch_input
        for nid, n in self.graph.nodes.items():
            if n.type_name == "pt_input_marker":
                marker_nodes.append(nid)
                continue
            if n.type_name == "batch_input":
                legacy_nodes.append(nid)
                continue
            has_dl = any(p.port_type == PortType.DATALOADER for p in n.outputs.values())
            has_x  = "x" in n.outputs and n.outputs["x"].port_type == PortType.TENSOR
            if has_dl and has_x:
                legacy_nodes.append(nid)

        if not marker_nodes and not legacy_nodes:
            return overrides

        # ── Parse the batch into a flat {col_name: tensor} dict ──────────
        port_values: dict[str, Any] = {}
        if isinstance(batch, dict) and "data" in batch:
            for port, val in batch["data"].items():
                port_values[port] = val
            port_values["label"] = batch.get("label")
        elif isinstance(batch, dict):
            for port, val in batch.items():
                port_values[port] = val
            # Synthesize "x" for legacy nodes that wire the universal port
            if "x" not in port_values:
                for key in ("observation.state", "observation", "features", "input"):
                    if key in batch:
                        port_values["x"] = batch[key]
                        break
            if "x" not in port_values:
                for col, val in batch.items():
                    if col != "label":
                        port_values["x"] = val
                        break
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            port_values["x"] = batch[0]
            port_values["label"] = batch[1]
        else:
            port_values["x"] = batch

        # ── Marker path: each marker pulls one column by modality ───────
        for nid in marker_nodes:
            node = self.graph.nodes[nid]
            modality = str(node.inputs["modality"].default_value or "x")
            if modality in port_values:
                overrides[(nid, "tensor")] = port_values[modality]

        # ── Legacy path: splay all columns across the dataset node ──────
        for nid in legacy_nodes:
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

        # Seed injection-point nodes with their primed (pre-override) outputs
        # so non-overridden ports (dataset metadata like vocab_size) still flow
        # through when execute() is skipped. Only injection nodes are seeded —
        # every other node re-runs execute() from scratch with gradients on.
        injected_nids = {nid for (nid, _port) in overrides.keys()}
        for nid in injected_nids:
            primed = self._primed_outputs.get(nid)
            if primed:
                stored[nid] = dict(primed)

        # Apply override values on top so downstream lookups see current-batch tensors
        for (nid, port), val in overrides.items():
            stored.setdefault(nid, {})[port] = val

        for node_id in order:
            node = self.graph.nodes[node_id]

            # If this node is an injection point (marker, BatchInput, or
            # dataset node) and its outputs are already overridden with
            # batch tensors, skip its execute() — stored[node_id] already
            # holds the values.
            if node_id in stored and (
                node.type_name in ("pt_input_marker", "batch_input")
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
