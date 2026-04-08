"""nn.BatchNorm2d layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd


class BatchNorm2dNode(BaseNode):
    type_name   = "pt_batchnorm2d"
    label       = "BatchNorm2d"
    category    = "Layers"
    subcategory = "Conv"
    description = "nn.BatchNorm2d(num_features). Running stats update during training."

    def __init__(self):
        self._layer: nn.BatchNorm2d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, num_features: int) -> nn.BatchNorm2d:
        cfg = (num_features,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.BatchNorm2d(num_features)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",    PortType.TENSOR, default=None)
        self.add_input("num_features", PortType.INT,    default=16)
        self.add_output("tensor_out",  PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(int(inputs.get("num_features") or 16))
        if not torch.is_grad_enabled():
            layer.eval()
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_bn2_{self.id[:6]}"
        return _layer_fwd(
            ov, iv, lv, f"nn.BatchNorm2d({self._val(iv,'num_features')})")
