"""nn.Conv2d layer node with shape-inferred input channels.

`in_ch` is inferred from the upstream tensor's channel dim at each
forward (NCHW shape → `tensor.shape[1]`). The legacy `in_ch` port is
kept as a fallback for priming-without-tensor and saved graphs — new
code should leave it alone and rely on inference.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import (
    _make_activation, _forward, _act_func, _infer_feature_dim,
)

_ACT_HELP = "none | relu | leakyrelu | sigmoid | tanh | gelu | elu | silu | softmax"


class Conv2dNode(BaseNode):
    type_name   = "pt_conv2d"
    label       = "Conv2d"
    category    = "Layers"
    subcategory = "Conv"
    description = ("nn.Conv2d with shape-inferred input channels. Expects "
                   "input shape (N, C, H, W). Specify out_ch only — in_ch "
                   "comes from the upstream tensor.")

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
        self.add_input("out_ch",     PortType.INT,    default=16,
                       description="Number of output channels")
        self.add_input("kernel",     PortType.INT,    default=3)
        self.add_input("stride",     PortType.INT,    default=1)
        self.add_input("padding",    PortType.INT,    default=0)
        self.add_input("activation", PortType.STRING, default="none",
                       description=_ACT_HELP)
        self.add_input("freeze",     PortType.BOOL,   default=False)
        # Legacy: in_ch is inferred from tensor_in.shape[1]. Port kept so
        # saved graphs don't KeyError; value is used only as a fallback
        # when no input tensor is flowing.
        self.add_input("in_ch",      PortType.INT,    default=0,
                       description="(legacy; ignored) — inferred from input")
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tensor_in = inputs.get("tensor_in")
        in_ch = _infer_feature_dim(tensor_in, inputs.get("in_ch"), axis=1)
        if in_ch <= 0:
            return {"tensor_out": None}
        layer = self._get_layer(
            in_ch,
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
        return {"tensor_out": _forward(layer, act, tensor_in)}

    def export(self, iv, ov):
        """Emit nn.LazyConv2d so the exported script mirrors runtime
        shape inference — first forward initializes weights based on
        actual input channels."""
        lv   = f"_conv_{self.safe_id}"
        tin  = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        out_ch  = self._val(iv, "out_ch")
        kernel  = self._val(iv, "kernel")
        stride  = self._val(iv, "stride")
        padding = self._val(iv, "padding")
        act_name = self.inputs["activation"].default_value if "activation" in self.inputs else ""
        func = _act_func(act_name)

        lines = [
            f"{lv} = nn.LazyConv2d({out_ch}, kernel_size={kernel}, "
            f"stride={stride}, padding={padding})",
            f"{tout} = {lv}({tin})",
        ]
        imports = ["import torch", "import torch.nn as nn"]
        if func:
            lines.append(f"{tout} = {func}({tout})")
            imports.append("import torch.nn.functional as F")
        return imports, lines
