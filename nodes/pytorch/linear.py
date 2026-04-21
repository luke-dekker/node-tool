"""nn.Linear layer node.

Shape inference model: `in_features` is NOT a user-facing port. On first
forward (or whenever the upstream tensor's last dim changes), the node
reads `input.shape[-1]` and builds/rebuilds its `nn.Linear` with the right
input size. The user only specifies how many neurons THIS layer has
(`out_features`) — upstream sizes come for free.

This means the autoresearch agent can tune `out_features` on any layer
and the downstream shape chain propagates automatically on the next
forward. No manual re-wiring.

Exports use `nn.LazyLinear` so the same semantics apply in standalone
scripts — first forward initializes; subsequent forwards reuse.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _make_activation, _forward, _act_func


_ACT_HELP = "none | relu | leakyrelu | sigmoid | tanh | gelu | elu | silu | softmax"


class LinearNode(BaseNode):
    type_name   = "pt_linear"
    label       = "Linear"
    category    = "Layers"
    subcategory = "Dense"
    description = ("nn.Linear with shape-inferred input. Specify `out_features` "
                   "only — the input size is read from the upstream tensor.")

    def __init__(self):
        self._layer: nn.Linear | None = None
        self._layer_cfg: tuple | None = None
        self._act_name: str = ""
        super().__init__()

    def _get_layer(self, in_f: int, out_f: int, bias: bool) -> nn.Linear:
        """Build or reuse the underlying nn.Linear. Rebuild triggers on any
        config change — including a changed `in_f` from upstream, which is
        how width mutations propagate through the chain."""
        cfg = (in_f, out_f, bias)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Linear(in_f, out_f, bias=bias)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        if self._layer is None:
            return []
        modules: list[nn.Module] = [self._layer]
        act = _make_activation(self._act_name)
        if act is not None:
            modules.append(act)
        return modules

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",    PortType.TENSOR, default=None)
        self.add_input("out_features", PortType.INT,    default=32,
                       description="Number of neurons in this layer")
        self.add_input("bias",         PortType.BOOL,   default=True)
        self.add_input("activation",   PortType.STRING, default="none",
                       description=_ACT_HELP)
        self.add_input("freeze",       PortType.BOOL,   default=False,
                       description="If True, this layer's weights won't update.")
        # Legacy port — ignored. LinearNode now infers `in_features` from
        # the upstream tensor's last dim. Kept so old templates / saved
        # graphs that still set this value don't KeyError on load. Will
        # be removed in a future pass once every template stops touching
        # it. Do not read this in new code.
        self.add_input("in_features",  PortType.INT,    default=0,
                       description="(legacy; ignored) — input size is "
                                   "inferred from the upstream tensor.")
        self.add_output("tensor_out",  PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tensor_in = inputs.get("tensor_in")
        # Primary path: infer `in_features` from the upstream tensor's last
        # dim. When upstream width changes (e.g. autoresearch mutated h1's
        # out_features), this naturally rebuilds the layer on next forward.
        #
        # Fallback path: if no tensor is flowing yet (priming before a
        # probe, or live-preview without data), use the legacy `in_features`
        # input so the user can still see a layer materialize. This keeps
        # saved graphs / older templates that ship a declared in_features
        # working, even though new code should infer.
        if tensor_in is not None:
            in_f = int(tensor_in.shape[-1])
        else:
            in_f = int(inputs.get("in_features") or 0)
            if in_f <= 0:
                return {"tensor_out": None}

        out_f = int(inputs.get("out_features") or 32)
        layer = self._get_layer(in_f, out_f, bool(inputs.get("bias", True)))

        self._act_name = inputs.get("activation") or ""
        freeze = bool(inputs.get("freeze", False))
        for p in layer.parameters():
            p.requires_grad = not freeze
        act = _make_activation(self._act_name)
        return {"tensor_out": _forward(layer, act, tensor_in)}

    def export(self, iv, ov):
        """Emit an `nn.LazyLinear(out)` + apply + optional activation.

        LazyLinear initializes weights on first forward the same way this
        node does, so exported scripts match the runtime behavior. Users
        reading the export see "N neurons" — same mental model as the
        canvas.
        """
        lv   = f"_lin_{self.safe_id}"
        tin  = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        out_f = self._val(iv, "out_features")
        bias  = self._val(iv, "bias")
        act_name = self.inputs["activation"].default_value if "activation" in self.inputs else ""
        func = _act_func(act_name)

        lines = [
            f"{lv} = nn.LazyLinear({out_f}, bias={bias})",
            f"{tout} = {lv}({tin})",
        ]
        imports = ["import torch", "import torch.nn as nn"]
        if func:
            lines.append(f"{tout} = {func}({tout})")
            imports.append("import torch.nn.functional as F")
        return imports, lines
