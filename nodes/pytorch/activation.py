"""Standalone activation layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _make_activation, _forward, _act_expr

_ACT_HELP = "none | relu | leakyrelu | sigmoid | tanh | gelu | elu | silu | softmax"


class ActivationNode(BaseNode):
    type_name   = "pt_activation"
    label       = "Activation"
    category    = "Layers"
    subcategory = "Dense"
    description = f"Standalone activation layer. Options: {_ACT_HELP}"

    def __init__(self):
        self._layer: nn.Module | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, name: str) -> nn.Module | None:
        cfg = (name,)
        if self._layer_cfg != cfg:
            self._layer = _make_activation(name)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("activation",  PortType.STRING, default="relu",
                       description=_ACT_HELP)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        act = self._get_layer(inputs.get("activation") or "relu")
        tin = inputs.get("tensor_in")
        if act is None:
            return {"tensor_out": tin}
        return {"tensor_out": _forward(act, None, tin)}

    def export(self, iv, ov):
        act_name = self.inputs["activation"].default_value if "activation" in self.inputs else "relu"
        act_expr = _act_expr(act_name) or "nn.ReLU()"
        tin  = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch.nn as nn"], [
            f"_act_{self.safe_id} = {act_expr}",
            f"{tout} = _act_{self.safe_id}({tin})",
        ]
