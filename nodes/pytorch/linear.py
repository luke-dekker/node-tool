"""nn.Linear layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _make_activation, _forward, _layer_fwd

_ACT_HELP = "none | relu | leakyrelu | sigmoid | tanh | gelu | elu | silu | softmax"


class LinearNode(BaseNode):
    type_name   = "pt_linear"
    label       = "Linear"
    category    = "Layers"
    subcategory = "Dense"
    description = "nn.Linear(in, out) with optional activation. Weights persist across graph runs."

    def __init__(self):
        self._layer: nn.Linear | None = None
        self._layer_cfg: tuple | None = None
        self._act_name: str = ""
        super().__init__()

    def _get_layer(self, in_f: int, out_f: int, bias: bool) -> nn.Linear:
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
        self.add_input("in_features",  PortType.INT,    default=64)
        self.add_input("out_features", PortType.INT,    default=32)
        self.add_input("bias",         PortType.BOOL,   default=True)
        self.add_input("activation",   PortType.STRING, default="none",
                       description=_ACT_HELP)
        self.add_input("freeze",       PortType.BOOL,   default=False)
        self.add_output("tensor_out",  PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("in_features")  or 64),
            int(inputs.get("out_features") or 32),
            bool(inputs.get("bias", True)),
        )
        self._act_name = inputs.get("activation") or ""
        freeze = bool(inputs.get("freeze", False))
        for p in layer.parameters():
            p.requires_grad = not freeze
        act = _make_activation(self._act_name)
        return {"tensor_out": _forward(layer, act, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv  = f"_lin_{self.safe_id}"
        act = self.inputs["activation"].default_value if "activation" in self.inputs else ""
        layer = (f"nn.Linear({self._val(iv,'in_features')}, {self._val(iv,'out_features')}, "
                 f"bias={self._val(iv,'bias')})")
        return _layer_fwd(ov, iv, lv, layer, act)
