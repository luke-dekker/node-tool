"""Positional encoding node — adds position information to a sequence tensor."""
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType


class _SinusoidalPE(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al)."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _LearnedPE(nn.Module):
    """Learned positional embedding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class PositionalEncodingNode(BaseNode):
    type_name   = "pt_positional_encoding"
    label       = "Positional Encoding"
    category    = "Layers"
    subcategory = "Attention"
    description = (
        "Adds positional information to a (B, T, d_model) sequence. "
        "Sinusoidal (fixed, no parameters) or learned (nn.Embedding)."
    )

    def __init__(self):
        self._layer: nn.Module | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, d_model, max_len, kind):
        cfg = (d_model, max_len, kind)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = (_LearnedPE(d_model, max_len) if kind == "learned"
                           else _SinusoidalPE(d_model, max_len))
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in", PortType.TENSOR, default=None,
                       description="(B, T, d_model) sequence tensor")
        self.add_input("d_model",   PortType.INT, default=256)
        self.add_input("max_len",   PortType.INT, default=5000)
        self.add_input("kind",      PortType.STRING, default="sinusoidal",
                       choices=["sinusoidal", "learned"])
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("d_model") or 256),
            int(inputs.get("max_len") or 5000),
            str(inputs.get("kind") or "sinusoidal"),
        )
        x = inputs.get("tensor_in")
        if x is None:
            return {"tensor_out": None}
        try:
            return {"tensor_out": layer(x)}
        except Exception:
            return {"tensor_out": None}

    def export(self, iv, ov):
        lv = f"_pe_{self.safe_id}"
        tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn", "import math"], [
            f"# Positional encoding ({self._val(iv, 'kind')})",
            f"# (instantiate _SinusoidalPE/_LearnedPE inline if needed)",
            f"{tout} = {tin}  # positional encoding applied in-graph",
        ]
