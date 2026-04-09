"""nn.BatchNorm1d layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd


class BatchNorm1dNode(BaseNode):
    type_name   = "pt_batchnorm1d"
    label       = "BatchNorm1d"
    category    = "Layers"
    subcategory = "Dense"
    description = "nn.BatchNorm1d(num_features). Running stats update during training."

    def __init__(self):
        self._layer: nn.BatchNorm1d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, num_features: int) -> nn.BatchNorm1d:
        cfg = (num_features,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.BatchNorm1d(num_features)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",    PortType.TENSOR, default=None)
        self.add_input("num_features", PortType.INT,    default=64)
        self.add_output("tensor_out",  PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(int(inputs.get("num_features") or 64))
        # Don't force eval mode — GraphAsModule.forward() calls self.train()/eval()
        # as appropriate; live preview runs inside torch.no_grad() so stats won't
        # drift meaningfully even in train mode, but using eval() here would break
        # batchnorm during actual training.
        if not torch.is_grad_enabled():
            layer.eval()
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_bn1_{self.safe_id}"
        return _layer_fwd(
            ov, iv, lv, f"nn.BatchNorm1d({self._val(iv,'num_features')})")
