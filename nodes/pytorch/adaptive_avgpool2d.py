"""nn.AdaptiveAvgPool2d layer node."""
from __future__ import annotations
from typing import Any
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd


class AdaptiveAvgPool2dNode(BaseNode):
    type_name   = "pt_adaptive_avgpool2d"
    label       = "AdaptiveAvgPool2d"
    category    = "Layers"
    subcategory = "Conv"
    description = (
        "nn.AdaptiveAvgPool2d(output_size). Pools any input H, W down to a "
        "fixed output size — lets a CNN accept variable-size images."
    )

    def __init__(self):
        self._layer: nn.AdaptiveAvgPool2d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, output_size: int) -> nn.AdaptiveAvgPool2d:
        cfg = (output_size,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.AdaptiveAvgPool2d(output_size)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("output_size", PortType.INT,    default=4,
                       description="Output H and W (square). e.g. 4 → (B, C, 4, 4)")
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(int(inputs.get("output_size") or 4))
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_aap_{self.safe_id}"
        return _layer_fwd(
            ov, iv, lv,
            f"nn.AdaptiveAvgPool2d({self._val(iv, 'output_size')})",
        )
