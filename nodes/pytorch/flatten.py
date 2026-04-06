"""nn.Flatten layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _make_activation, _forward, _layer_fwd


class FlattenNode(BaseNode):
    type_name   = "pt_flatten"
    label       = "Flatten"
    category    = "Layers"
    subcategory = "Dense"
    description = "nn.Flatten - collapses all dims after the batch dim."

    def __init__(self):
        self._layer: nn.Flatten | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, start_dim: int) -> nn.Flatten:
        cfg = (start_dim,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Flatten(start_dim=start_dim)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("start_dim",   PortType.INT,    default=1)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(int(inputs.get("start_dim") or 1))
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_flat_{self.id[:6]}"
        return ["import torch", "import torch.nn as nn"], _layer_fwd(
            ov, iv, lv, f"nn.Flatten(start_dim={self._val(iv,'start_dim')})")
