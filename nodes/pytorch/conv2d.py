"""nn.Conv2d layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _make_activation, _forward, _layer_fwd

_ACT_HELP = "none | relu | leakyrelu | sigmoid | tanh | gelu | elu | silu | softmax"


class Conv2dNode(BaseNode):
    type_name   = "pt_conv2d"
    label       = "Conv2d"
    category    = "Layers"
    subcategory = "Conv"
    description = "nn.Conv2d with optional activation. Expects input shape (N, C, H, W)."

    def __init__(self):
        self._layer: nn.Conv2d | None = None
        self._layer_cfg: tuple | None = None
        self._act_name: str = ""
        super().__init__()

    def _get_layer(self, in_ch, out_ch, kernel, stride, padding) -> nn.Conv2d:
        cfg = (in_ch, out_ch, kernel, stride, padding)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                                    stride=stride, padding=padding)
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
        self.add_input("tensor_in",  PortType.TENSOR, default=None)
        self.add_input("in_ch",      PortType.INT,    default=1)
        self.add_input("out_ch",     PortType.INT,    default=16)
        self.add_input("kernel",     PortType.INT,    default=3)
        self.add_input("stride",     PortType.INT,    default=1)
        self.add_input("padding",    PortType.INT,    default=0)
        self.add_input("activation", PortType.STRING, default="none",
                       description=_ACT_HELP)
        self.add_input("freeze",     PortType.BOOL,   default=False)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("in_ch")   or 1),
            int(inputs.get("out_ch")  or 16),
            int(inputs.get("kernel")  or 3),
            int(inputs.get("stride")  or 1),
            int(inputs.get("padding") or 0),
        )
        self._act_name = inputs.get("activation") or ""
        freeze = bool(inputs.get("freeze", False))
        for p in layer.parameters():
            p.requires_grad = not freeze
        act = _make_activation(self._act_name)
        return {"tensor_out": _forward(layer, act, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv  = f"_conv_{self.safe_id}"
        act = self.inputs["activation"].default_value if "activation" in self.inputs else ""
        layer = (f"nn.Conv2d({self._val(iv,'in_ch')}, {self._val(iv,'out_ch')}, "
                 f"{self._val(iv,'kernel')}, stride={self._val(iv,'stride')}, "
                 f"padding={self._val(iv,'padding')})")
        return _layer_fwd(ov, iv, lv, layer, act)
