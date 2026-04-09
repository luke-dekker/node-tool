"""nn.Dropout layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd


class DropoutNode(BaseNode):
    type_name   = "pt_dropout"
    label       = "Dropout"
    category    = "Layers"
    subcategory = "Dense"
    description = "nn.Dropout(p). Inactive during graph visualization (eval mode)."

    def __init__(self):
        self._layer: nn.Dropout | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, p: float) -> nn.Dropout:
        cfg = (p,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Dropout(p=p)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("p",           PortType.FLOAT,  default=0.5,
                       description="Drop probability")
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(float(inputs.get("p") or 0.5))
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_drop_{self.safe_id}"
        return _layer_fwd(
            ov, iv, lv, f"nn.Dropout({self._val(iv,'p')})")
