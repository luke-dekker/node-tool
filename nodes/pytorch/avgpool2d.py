"""nn.AvgPool2d layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd


class AvgPool2dNode(BaseNode):
    type_name   = "pt_avgpool2d"
    label       = "AvgPool2d"
    category    = "Layers"
    subcategory = "Conv"
    description = "nn.AvgPool2d(kernel_size, stride)."

    def __init__(self):
        self._layer: nn.AvgPool2d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, kernel, stride) -> nn.AvgPool2d:
        cfg = (kernel, stride)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.AvgPool2d(kernel_size=kernel, stride=stride)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("kernel",      PortType.INT,    default=2)
        self.add_input("stride",      PortType.INT,    default=2)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("kernel") or 2),
            int(inputs.get("stride") or 2),
        )
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_avp_{self.id[:6]}"
        return _layer_fwd(
            ov, iv, lv, f"nn.AvgPool2d({self._val(iv,'kernel')}, stride={self._val(iv,'stride')})")
