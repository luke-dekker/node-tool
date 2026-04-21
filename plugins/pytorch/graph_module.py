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

Performance note: the forward() path is called thousands of times per epoch.
All graph-shape work — topological order, connection map, per-node input
plans, injection-point classification, marker modality lookup, layer-module
collection — is computed ONCE at __init__ and cached. The hot path walks
the cached structures and calls each node's `execute()`. It also restricts
execution to the ancestors of `output_node_id` — nodes that cannot affect
the output (e.g. a Mutator wired to an OllamaClient sitting in the same
canvas) are skipped entirely instead of firing an LLM call per batch.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

from core.graph import Graph
from core.node import MarkerRole
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


def _ancestors_inclusive(graph: Graph, target_id: str) -> set[str]:
    """Return every node whose outputs can reach `target_id`, inclusive.

    Used to restrict forward() execution to nodes that actually affect the
    loss. Nodes downstream of the target, or on disjoint branches
    (autoresearch driver nodes, viz nodes, anything that doesn't feed the
    loss), are excluded.
    """
    radj: dict[str, set[str]] = {nid: set() for nid in graph.nodes}
    for c in graph.connections:
        if c.from_node_id in radj and c.to_node_id in radj:
            radj[c.to_node_id].add(c.from_node_id)
    ancestors: set[str] = {target_id}
    stack = [target_id]
    while stack:
        cur = stack.pop()
        for prv in radj[cur]:
            if prv not in ancestors:
                ancestors.add(prv)
                stack.append(prv)
    return ancestors


class GraphAsModule(nn.Module):
    """Makes a Graph behave like an nn.Module for training.

    The forward() method:
      1. Maps the current batch onto injection-point override values
      2. Executes the ancestors of the output node (in cached topo order)
         with those overrides in place
      3. Returns the tensor at (output_node_id, output_port)
    """

    def __init__(self, graph: Graph, output_node_id: str, output_port: str = "tensor_in"):
        super().__init__()
        self.graph = graph
        self.output_node_id = output_node_id
        self.output_port = output_port

        # ── Ancestors of the output: the only nodes we actually need to
        # execute on the forward path. Disjoint branches (autoresearch
        # driver nodes, viz nodes, dangling LLM clients) stay out.
        self._active_nids: set[str] = _ancestors_inclusive(graph, output_node_id)

        # ── Prime: run the whole graph once under no_grad so layer modules
        # materialize (lazy _get_layer() calls) and metadata (vocab_size,
        # task_id) propagates. We prime on the full graph, not just the
        # active cone, so the priming matches Graph.execute() semantics.
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

        # ── Register layer modules so self.parameters() sees them. Only
        # active-cone layers are trainable — a disjoint Linear elsewhere
        # on the canvas wouldn't receive gradients anyway.
        self._layer_modules = nn.ModuleDict()
        all_layers = _collect_layer_modules(graph)
        for key, mod in all_layers.items():
            base_nid = key[:-3] if key.endswith("_mm") else key
            if base_nid in self._active_nids:
                safe_key = key.replace("-", "_")
                self._layer_modules[safe_key] = mod

        # ── Cache topology: topo order restricted to the active cone,
        # connection map restricted to edges between active nodes, and a
        # per-node input plan so the hot path is a tight loop.
        full_order = graph.topological_order()
        self._order: list[str] = [nid for nid in full_order if nid in self._active_nids]
        self._conn_map: dict[tuple[str, str], tuple[str, str]] = {
            (c.to_node_id, c.to_port): (c.from_node_id, c.from_port)
            for c in graph.connections
            if c.to_node_id in self._active_nids and c.from_node_id in self._active_nids
        }

        # Per-node plan: for each input port, a 5-tuple
        #   (port_name, from_nid_or_None, from_port_or_None, port_type, default_value)
        # Built once; the forward path only does dict lookups + coerce.
        self._node_input_plans: dict[str, list[tuple[str, str | None, str | None, Any, Any]]] = {}
        for nid in self._order:
            node = graph.nodes[nid]
            plan: list[tuple[str, str | None, str | None, Any, Any]] = []
            for port_name, port in node.inputs.items():
                src = self._conn_map.get((nid, port_name))
                if src is not None:
                    plan.append((port_name, src[0], src[1], port.port_type, port.default_value))
                else:
                    plan.append((port_name, None, None, port.port_type, port.default_value))
            self._node_input_plans[nid] = plan

        # Pre-classify injection points within the active cone.
        self._injection_nids: set[str] = set()
        for nid in self._active_nids:
            n = graph.nodes[nid]
            if n.marker_role == MarkerRole.INPUT or n.type_name == "batch_input":
                self._injection_nids.add(nid)
                continue
            has_dl = any(p.port_type == PortType.DATALOADER for p in n.outputs.values())
            has_x = "x" in n.outputs and n.outputs["x"].port_type == PortType.TENSOR
            if has_dl and has_x:
                self._injection_nids.add(nid)

        # Cached modality lookup — the per-batch override map reads this
        # instead of rescanning node inputs.
        self._marker_modalities: dict[str, str] = {}
        for nid in self._active_nids:
            n = graph.nodes[nid]
            if n.marker_role == MarkerRole.INPUT:
                self._marker_modalities[nid] = str(n.inputs["modality"].default_value or "x")

        # Legacy injection nodes (non-marker) and their output port names —
        # cached so the per-batch splaying doesn't re-enumerate outputs.
        self._legacy_injection_ports: dict[str, set[str]] = {
            nid: set(graph.nodes[nid].outputs.keys())
            for nid in self._injection_nids
            if nid not in self._marker_modalities
        }

    # ------------------------------------------------------------------ forward

    def forward(self, batch: Any) -> torch.Tensor | None:
        """Run the graph with `batch` injected at BatchInput nodes.

        `batch` can be:
          - a dict with 'data' and 'label' keys (multimodal collate format)
          - a (x, y) tuple
          - a single tensor
        """
        override = self._batch_to_port_values(batch)
        outputs = self._execute_with_overrides(override)

        node_out = outputs.get(self.output_node_id, {})
        if self.output_port in node_out:
            return node_out[self.output_port]
        # Walk back through the connection if caller pointed at an input port
        src = self._conn_map.get((self.output_node_id, self.output_port))
        if src is not None:
            return outputs.get(src[0], {}).get(src[1])
        return None

    # ------------------------------------------------------------------ helpers

    def _batch_to_port_values(self, batch: Any) -> dict[tuple[str, str], Any]:
        """Map `batch` onto {(injection_node_id, port_name): tensor}.

        Two injection-point flavors supported:
          1. Input markers — each marker pulls one tensor by its `modality`.
          2. Legacy dataset / batch_input nodes — whole batch splays across
             output ports by column name.

        Batch shape handling:
          - dict with 'data' key: legacy multimodal collate format
          - plain dict {col: tensor}: DatasetNode's _collate_manifest output
          - (x, y) tuple: standard classification batch
          - single tensor: unsupervised / autoencoder
        """
        overrides: dict[tuple[str, str], Any] = {}

        if not self._injection_nids:
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

        # ── Marker path (cached modality per nid) ────────────────────────
        for nid, modality in self._marker_modalities.items():
            if modality in port_values:
                overrides[(nid, "tensor")] = port_values[modality]

        # ── Legacy path (cached output port set per nid) ─────────────────
        for nid, out_ports in self._legacy_injection_ports.items():
            for port_name, val in port_values.items():
                if port_name in out_ports:
                    overrides[(nid, port_name)] = val

        return overrides

    def _execute_with_overrides(
        self, overrides: dict[tuple[str, str], Any]
    ) -> dict[str, dict[str, Any]]:
        """Execute the cached active-cone topo order with pre-set values for
        specific (node_id, port) pairs. Skips nodes outside the active cone
        and skips execute() on injection nodes whose outputs are overridden.
        """
        stored: dict[str, dict[str, Any]] = {}

        # Seed injection-point nodes with their primed (pre-override) outputs
        # so non-overridden ports (vocab_size, task_id, ...) still flow.
        for nid in self._injection_nids:
            primed = self._primed_outputs.get(nid)
            if primed:
                stored[nid] = dict(primed)

        # Overlay the current-batch overrides.
        for (nid, port), val in overrides.items():
            stored.setdefault(nid, {})[port] = val

        coerce = PortTypeRegistry.coerce_value
        nodes_map = self.graph.nodes
        plans = self._node_input_plans
        injected = self._injection_nids

        for node_id in self._order:
            # Injection nodes with seeded values skip execute — stored already
            # has their batch tensors (and any primed metadata).
            if node_id in injected and node_id in stored:
                continue

            plan = plans[node_id]
            inputs: dict[str, Any] = {}
            for port_name, from_id, from_port, ptype, default in plan:
                if from_id is not None:
                    src_store = stored.get(from_id)
                    if src_store is not None and from_port in src_store:
                        raw = src_store[from_port]
                        inputs[port_name] = coerce(ptype, raw) if raw is not None else default
                        continue
                inputs[port_name] = default

            try:
                outputs = nodes_map[node_id].execute(inputs)
            except Exception:
                outputs = {}
            if outputs:
                stored.setdefault(node_id, {}).update(outputs)
            else:
                stored.setdefault(node_id, {})

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
