"""nn.LayerNorm layer node."""
from __future__ import annotations
from typing import Any
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd


class LayerNormNode(BaseNode):
    type_name   = "pt_layernorm"
    label       = "Layer Norm"
    category    = "Layers"
    subcategory = "Dense"
    description = "nn.LayerNorm over the last N feature dimensions."

    def __init__(self):
        self._layer: nn.LayerNorm | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, normalized_shape: int, eps: float) -> nn.LayerNorm:
        cfg = (normalized_shape, eps)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.LayerNorm(normalized_shape, eps=eps)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",        PortType.TENSOR, default=None)
        self.add_input("normalized_shape", PortType.INT, default=64,
                       description="Size of the feature dim to normalize over")
        self.add_input("eps",              PortType.FLOAT, default=1e-5)
        self.add_output("tensor_out",      PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("normalized_shape") or 64),
            float(inputs.get("eps") or 1e-5),
        )
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_ln_{self.safe_id}"
        expr = (f"nn.LayerNorm({self._val(iv, 'normalized_shape')}, "
                f"eps={self._val(iv, 'eps')})")
        return _layer_fwd(ov, iv, lv, expr, "")
