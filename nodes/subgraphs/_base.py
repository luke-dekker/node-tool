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

Nested subgraphs are supported automatically. Discovery at startup just
loads .subgraph.json files into class objects with `_subgraph_file` as a
class attribute — no inner instantiation. The inner graph isn't built until
the first execute() call (`_ensure_inner_graph` is lazy). By that time ALL
subgraph types are in NODE_REGISTRY regardless of discovery order, so an
outer subgraph that references an inner one Just Works. Parameter discovery
recurses naturally through the `_layer` ModuleList chain — each level wraps
the level below in its own ModuleList.

Two execution modes via the `mode` field:

  once  — default. Single execute() call; behaves as the original
          SubgraphNode. All scan-config fields are hidden in the inspector.

  scan  — unrolls the inner graph T times along a time axis, threading a
          single carry channel between iterations. Turns any cell-shaped
          subgraph into a recurrence:

            step_input    — name of an external input port; the user wires
                             a (B, T, F) tensor into that port and Scan
                             slices x_t along `time_dim` per iteration.
            carry_in      — name of an external input that receives the
                             threaded state. Gets the user-wired tensor at
                             t=0 (typical: zeros), then h_out from the
                             previous iteration thereafter.
            carry_out     — name of an external output producing the new
                             carry h_{t+1}.
            step_output   — name of an external output collected each step
                             and stacked along time_dim. Empty → step output
                             is the carry itself; the carry_out port emits
                             the stacked time series in that case.
            time_dim      — axis of step_input that is time (default 1).

          Output semantics in scan mode:
            • step_output port (when distinct from carry_out): stacked
                (B, T, F') time series.
            • carry_out port (when distinct from step_output): final
                carry h_T (single value, no time axis).
            • carry_out port when step_output is empty: the stacked time
                series — covers the common "cell only emits hidden" case.
            • All other external output ports: value from the last iteration.

          Don't put a stateful layer (RecurrentLayer, etc.) inside a scan
          cell — Scan is for raw cells, not pre-baked layers. The user-
          explicit carry IS the state; the inner modules should be stateless.

This is the iteration primitive the graph was missing. RNN cells, halting
nets, CTM-style internal-tick loops, message-passing GNNs, iterative
refinement (Universal Transformer-style depth-as-iteration), and meta-
learning inner loops are all expressible as cell subgraphs + scan.
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

        # Scan-config fields. STRING/INT auto-render as inspector widgets
        # (no wire socket). All hidden when mode=once via relevant_inputs.
        # Dropdown choices are populated from the file's external port names
        # so the user picks an existing port rather than typing a name.
        in_names  = [ext.name for ext in sf.external_inputs]
        out_names = [ext.name for ext in sf.external_outputs]
        self.add_input("mode", PortType.STRING, default="once",
                       choices=["once", "scan"],
                       description="once = single execute. scan = unroll T times along time_dim.")
        self.add_input("step_input", PortType.STRING, default="",
                       choices=in_names,
                       description="Scan: external input port that gets sliced per step.")
        self.add_input("carry_in", PortType.STRING, default="",
                       choices=in_names,
                       description="Scan: external input port that receives the carry (h_t).")
        self.add_input("carry_out", PortType.STRING, default="",
                       choices=out_names,
                       description="Scan: external output port producing the new carry (h_{t+1}).")
        self.add_input("step_output", PortType.STRING, default="",
                       choices=out_names,
                       description="Scan: external output stacked per step. Empty = use carry_out.")
        self.add_input("time_dim", PortType.INT, default=1,
                       description="Scan: axis of step_input that is time (default 1, batch-first).")

    def relevant_inputs(self, values):
        # Wired ports (the file's external_inputs) are always relevant — they
        # carry data. We only filter the scan-config fields.
        sf = self._subgraph_file
        external_in_names = [ext.name for ext in sf.external_inputs] if sf else []
        scan_fields = ["mode", "step_input", "carry_in", "carry_out",
                       "step_output", "time_dim"]
        mode = (values.get("mode") or "once").strip()
        if mode == "once":
            # Hide all scan-only fields; keep the mode toggle visible so the
            # user can flip into scan mode from the inspector.
            return external_in_names + ["mode"]
        return external_in_names + scan_fields

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

        mode = (inputs.get("mode") or "once").strip()
        if mode == "scan":
            return self._execute_scan(inputs, sf, ig)
        return self._execute_once(inputs, sf, ig)

    def _execute_once(self, inputs, sf, ig) -> dict[str, Any]:
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
            stored, _terminal, _errors = ig.execute()
        except Exception as exc:
            return {ext.name: None for ext in sf.external_outputs} | {"__error__": str(exc)}

        # Read boundary outputs
        result: dict[str, Any] = {}
        for ext in sf.external_outputs:
            node_outs = stored.get(ext.inner_node, {})
            result[ext.name] = node_outs.get(ext.inner_port)

        self._refresh_layer()
        return result

    def _execute_scan(self, inputs, sf, ig) -> dict[str, Any]:
        """Unroll the inner graph T times along a time axis.

        Threads one carry channel: carry_in port gets init from the wired
        tensor at t=0, then carry_out from the previous iteration thereafter.
        Each iteration pushes static inputs unchanged, slices step_input
        along time_dim, and reads step_output (stacked) + carry_out (final).
        """
        try:
            import torch
        except ImportError:
            return {ext.name: None for ext in sf.external_outputs}
        null = {ext.name: None for ext in sf.external_outputs}

        step_input  = (inputs.get("step_input")  or "").strip()
        carry_in    = (inputs.get("carry_in")    or "").strip()
        carry_out   = (inputs.get("carry_out")   or "").strip()
        step_output = (inputs.get("step_output") or "").strip() or carry_out
        time_dim    = int(inputs.get("time_dim") or 1)

        ext_in  = {p.name: (p.inner_node, p.inner_port) for p in sf.external_inputs}
        ext_out = {p.name: (p.inner_node, p.inner_port) for p in sf.external_outputs}

        # Validate role assignments. Bail with nulls if any required role
        # doesn't map to a real external port — the user needs to pick from
        # the dropdowns in the inspector.
        if (step_input  not in ext_in
            or carry_in  not in ext_in
            or carry_out not in ext_out
            or step_output not in ext_out):
            return null | {"__error__": (
                f"scan mode: roles must name real external ports. "
                f"step_input={step_input!r} carry_in={carry_in!r} "
                f"carry_out={carry_out!r} step_output={step_output!r}"
            )}

        xs = inputs.get(step_input)
        if xs is None:
            return null
        T = xs.shape[time_dim]

        # Initial carry comes from the carry_in port's wired value.
        carry = inputs.get(carry_in)

        ys: list = []
        last_stored: dict | None = None
        x_target = ext_in[step_input]
        h_in_target = ext_in[carry_in]
        h_out_source = ext_out[carry_out]
        y_source = ext_out[step_output]

        for t in range(T):
            # Push values onto boundary inputs. Static inputs (everything
            # except step_input and carry_in) get pushed once — same value
            # each iteration. We re-set them each step anyway because
            # default_value can be clobbered by Graph.execute internals.
            for ext in sf.external_inputs:
                target = ext_in[ext.name]
                if ext.name == step_input:
                    val = xs.select(time_dim, t)
                elif ext.name == carry_in:
                    val = carry
                else:
                    val = inputs.get(ext.name)
                ig.nodes[target[0]].inputs[target[1]].default_value = val

            try:
                stored, _terminal, _errors = ig.execute()
            except Exception as exc:
                return null | {"__error__": f"scan iter {t}: {exc}"}
            last_stored = stored

            carry = stored.get(h_out_source[0], {}).get(h_out_source[1])
            ys.append(stored.get(y_source[0], {}).get(y_source[1]))

        # Build the result dict: stacked step_output, final carry on
        # carry_out (or stacked if same port plays both roles), and last-
        # iteration value on every other external output.
        if not ys or any(y is None for y in ys):
            ys_tensor = None
        else:
            try:
                ys_tensor = torch.stack(ys, dim=time_dim)
            except Exception:
                ys_tensor = None

        result: dict[str, Any] = {}
        for ext in sf.external_outputs:
            if ext.name == step_output:
                # When carry_out == step_output, this port carries the time
                # series and the user can take ys[..., -1] for the final
                # carry. When they're distinct, only step_output is stacked.
                result[ext.name] = ys_tensor
            elif ext.name == carry_out:
                result[ext.name] = carry
            else:
                src = ext_out[ext.name]
                result[ext.name] = (last_stored or {}).get(src[0], {}).get(src[1])

        self._refresh_layer()
        return result

    def _refresh_layer(self) -> None:
        """Update self._layer to a ModuleList of inner layers so
        GraphAsModule's parameter walk picks them up exactly once. Same
        underlying nn.Module instances reused across iterations in scan
        mode → weights shared, gradients flow naturally through BPTT."""
        try:
            import torch.nn as nn
            inner_layers = self.get_layers()
            if inner_layers:
                self._layer = nn.ModuleList(inner_layers)
        except Exception:
            pass

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
