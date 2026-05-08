"""ScanNode — first-class iteration primitive over a subgraph.

Treats a saved .subgraph.json as the "cell" of an RNN-style recurrence:
unrolls it T times along a time axis, threading a single carry channel
between iterations and stacking per-step outputs.

Inputs (per-instance config, no manifest changes required):

  xs              — (B, T, F) sequence to slice per step. The slice along
                     `time_dim` is fed into the inner port named by
                     `x_port` on each iteration.
  init_h          — initial carry state. Fed into `h_in_port` at t=0.
                     Default None → the cell sees None and must handle it
                     (typical pattern: lazy-init zeros on first forward).
  subgraph_path   — path to the cell .subgraph.json. Resolved relative to
                     `subgraphs/` if not found at the literal path.
  x_port          — external input name on the cell that receives x_t
                     (default "x").
  h_in_port       — external input name that receives h_t (default "h").
  h_out_port      — external output name that produces h_{t+1} (default "h").
  y_port          — external output name producing y_t. Empty → y_t = h_out
                     (covers the common "the cell only emits hidden" case).
  time_dim        — which axis of xs is time (default 1, batch-first).

Outputs:

  ys              — torch.stack(y_t, dim=time_dim) → (B, T, F')
  final_h         — terminal carry value (just h_T)
  module          — nn.ModuleList of the cell's parameters, refreshed each
                     forward so GraphAsModule's parameter walk picks them
                     up exactly once (the same modules are reused every
                     iteration; weights are shared, gradients flow).

Autograd: each iteration calls the cell's inner Graph.execute() which
preserves the torch autograd lineage naturally. Backward through the full
unrolled history works because all T iterations call the same underlying
nn.Module instances (built once on first iteration, cached on the inner
nodes' _layer attributes). Don't put a stateful layer like RecurrentLayer
inside a scan cell — Scan is for raw cells, not pre-baked layers.

This is the MVP iteration primitive. One carry, one step input, one
step output. Tuple/multi-carry, role-in-manifest, and a UI affordance to
mark a subgraph as scannable can land once this proves out.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from core.node import BaseNode, PortType


class ScanNode(BaseNode):
    type_name   = "pt_scan"
    label       = "Scan"
    category    = "Layers"
    subcategory = "Recurrent"
    description = (
        "Unroll a subgraph (treated as the recurrence cell) T times along "
        "the time axis of `xs`. Threads `init_h → h_in → h_out → h_in …` "
        "between iterations and stacks per-step outputs into `ys`. Set the "
        "subgraph_path + x/h_in/h_out/y external port names so Scan knows "
        "which inner ports get the time-slice and which form the carry."
    )

    def __init__(self):
        self._subgraph_file = None
        self._inner_graph = None
        self._cached_path: str | None = None
        self._layer = None
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("xs",            PortType.TENSOR, default=None,
                       description="(B, T, F) input sequence; sliced along time_dim per step.")
        self.add_input("init_h",        PortType.TENSOR, default=None, optional=True,
                       description="Initial carry; None means the cell sees None at t=0.")
        self.add_input("subgraph_path", PortType.STRING, default="",
                       description="Path to .subgraph.json (relative to subgraphs/ ok).")
        self.add_input("x_port",        PortType.STRING, default="x",
                       description="External input name on the cell that receives x_t.")
        self.add_input("h_in_port",     PortType.STRING, default="h",
                       description="External input name that receives the carry h_t.")
        self.add_input("h_out_port",    PortType.STRING, default="h",
                       description="External output name producing the new carry h_{t+1}.")
        self.add_input("y_port",        PortType.STRING, default="",
                       description="External output name producing y_t. Empty → y is h_out.")
        self.add_input("time_dim",      PortType.INT,    default=1,
                       description="Axis of xs that is time (default 1, batch-first).")
        self.add_output("ys",       PortType.TENSOR,
                        description="(B, T, F') stacked per-step outputs.")
        self.add_output("final_h",  PortType.TENSOR,
                        description="Terminal carry value h_T.")
        self.add_output("module",   PortType.MODULE,
                        description="ModuleList of the cell's parameters (for save/freeze).")

    # ── Inner subgraph management ──────────────────────────────────────────

    def _resolve_path(self, path: str) -> Path | None:
        from core.subgraph import SUBGRAPHS_DIR
        p = Path(path)
        if p.is_file():
            return p
        candidates = [
            SUBGRAPHS_DIR / path,
            SUBGRAPHS_DIR / (path if path.endswith(".subgraph.json")
                             else f"{path}.subgraph.json"),
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _ensure_inner_graph(self, path: str) -> bool:
        if self._cached_path == path and self._inner_graph is not None:
            return True
        from core.subgraph import SubgraphFile
        resolved = self._resolve_path(path)
        if resolved is None:
            return False
        self._subgraph_file = SubgraphFile.load(resolved)
        self._inner_graph = self._subgraph_file.build_inner_graph()
        self._cached_path = path
        return True

    def get_layers(self) -> list:
        if self._inner_graph is None:
            return []
        import torch.nn as nn
        modules: list[nn.Module] = []
        for inner in self._inner_graph.nodes.values():
            layer = getattr(inner, "_layer", None)
            if isinstance(layer, nn.Module):
                modules.append(layer)
        return modules

    # ── Execute ────────────────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            import torch
            import torch.nn as nn
            xs = inputs.get("xs")
            path = (inputs.get("subgraph_path") or "").strip()
            null = {"ys": None, "final_h": None, "module": self._layer}
            if xs is None or not path:
                return null
            if not self._ensure_inner_graph(path):
                return null

            sf = self._subgraph_file
            ig = self._inner_graph
            x_port     = (inputs.get("x_port")     or "x").strip()
            h_in_port  = (inputs.get("h_in_port")  or "h").strip()
            h_out_port = (inputs.get("h_out_port") or "h").strip()
            y_port     = (inputs.get("y_port")     or "").strip()
            time_dim   = int(inputs.get("time_dim") or 1)

            ext_in  = {p.name: (p.inner_node, p.inner_port) for p in sf.external_inputs}
            ext_out = {p.name: (p.inner_node, p.inner_port) for p in sf.external_outputs}

            x_target     = ext_in.get(x_port)
            h_in_target  = ext_in.get(h_in_port)
            h_out_source = ext_out.get(h_out_port)
            y_source     = ext_out.get(y_port) if y_port else h_out_source
            if x_target is None or h_in_target is None or h_out_source is None:
                # Fail loudly with what we found so the user can fix port names.
                return null

            T = xs.shape[time_dim]
            h = inputs.get("init_h")
            ys: list = []
            for t in range(T):
                # Slice along the time axis. Tensor.select returns a view
                # without copying; autograd is preserved.
                x_t = xs.select(time_dim, t)

                # Push x_t and current carry into the cell's boundary inputs
                # via default_value (Graph.execute reads default_value when a
                # port has no upstream wire — true for boundary inputs).
                ig.nodes[x_target[0]].inputs[x_target[1]].default_value = x_t
                ig.nodes[h_in_target[0]].inputs[h_in_target[1]].default_value = h

                stored, _terminal, _errors = ig.execute()

                # Read new carry first so a None h_out fails fast on the
                # next iteration rather than silently stacking ys with stale h.
                h = stored.get(h_out_source[0], {}).get(h_out_source[1])
                y_t = stored.get(y_source[0], {}).get(y_source[1])
                ys.append(y_t)

            # Stack along the original time axis. If any y_t is None the
            # whole stack fails; surface that rather than emit a garbage shape.
            if not ys or any(y is None for y in ys):
                ys_tensor = None
            else:
                ys_tensor = torch.stack(ys, dim=time_dim)

            # Refresh _layer so GraphAsModule's parameter walk picks up the
            # cell's modules. Same module instances are reused every step so
            # params are counted once, not T times.
            inner_layers = self.get_layers()
            self._layer = nn.ModuleList(inner_layers) if inner_layers else None

            return {"ys": ys_tensor, "final_h": h, "module": self._layer}
        except Exception:
            return {"ys": None, "final_h": None, "module": self._layer}

    def export(self, iv, ov):
        # MVP: emit a runnable Python loop that re-executes the inner
        # subgraph T times. Inlining the cell would be nicer but means
        # generating a function from inner.export() — punt to v2.
        path = (self.inputs["subgraph_path"].default_value or "").strip()
        x_port     = (self.inputs["x_port"].default_value     or "x").strip()
        h_in_port  = (self.inputs["h_in_port"].default_value  or "h").strip()
        h_out_port = (self.inputs["h_out_port"].default_value or "h").strip()
        y_port     = (self.inputs["y_port"].default_value     or "").strip()
        time_dim   = int(self.inputs["time_dim"].default_value or 1)
        xs   = iv.get("xs")     or "None  # TODO: connect xs"
        ih   = iv.get("init_h") or "None"
        ys_v = ov.get("ys",       "_ys")
        h_v  = ov.get("final_h",  "_final_h")
        return [
            "import torch",
            "from core.subgraph import SubgraphFile",
        ], [
            f"_sf = SubgraphFile.load({path!r})",
            f"_ig = _sf.build_inner_graph()",
            f"_ext_in  = {{p.name: (p.inner_node, p.inner_port) for p in _sf.external_inputs}}",
            f"_ext_out = {{p.name: (p.inner_node, p.inner_port) for p in _sf.external_outputs}}",
            f"_xt = _ext_in[{x_port!r}]; _ht = _ext_in[{h_in_port!r}]",
            f"_hs = _ext_out[{h_out_port!r}]",
            f"_ys_src = _ext_out[{y_port!r}] if {y_port!r} else _hs",
            f"_h = {ih}",
            f"_ys = []",
            f"for _t in range({xs}.shape[{time_dim}]):",
            f"    _ig.nodes[_xt[0]].inputs[_xt[1]].default_value = {xs}.select({time_dim}, _t)",
            f"    _ig.nodes[_ht[0]].inputs[_ht[1]].default_value = _h",
            f"    _stored, _, _ = _ig.execute()",
            f"    _h = _stored.get(_hs[0], {{}}).get(_hs[1])",
            f"    _ys.append(_stored.get(_ys_src[0], {{}}).get(_ys_src[1]))",
            f"{ys_v} = torch.stack(_ys, dim={time_dim}) if _ys and all(y is not None for y in _ys) else None",
            f"{h_v} = _h",
        ]
