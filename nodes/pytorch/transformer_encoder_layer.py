"""nn.TransformerEncoderLayer node — one block of self-attention + FFN."""
from __future__ import annotations
from typing import Any
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _infer_feature_dim


class TransformerEncoderLayerNode(BaseNode):
    """One transformer encoder block: self-attn + FFN + residual + layer norm.

    Stack several of these to build a full transformer encoder. Input and
    output shape is (B, T, embed_dim).
    """
    type_name   = "pt_transformer_encoder_layer"
    label       = "Transformer Encoder Layer"
    category    = "Layers"
    subcategory = "Attention"
    description = (
        "nn.TransformerEncoderLayer — one block of self-attention + "
        "feedforward + residual + layer norm. Stack N of these to build a "
        "transformer encoder. Input: (B, T, embed_dim)."
    )

    def __init__(self):
        self._layer: nn.TransformerEncoderLayer | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, d_model, nhead, dim_ff, dropout, activation):
        cfg = (d_model, nhead, dim_ff, dropout, activation)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, activation=activation, batch_first=True,
            )
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",       PortType.TENSOR, default=None,
                       description="(B, T, d_model) input")
        self.add_input("nhead",           PortType.INT, default=8)
        self.add_input("dim_feedforward", PortType.INT, default=1024)
        self.add_input("dropout",         PortType.FLOAT, default=0.1)
        self.add_input("activation",      PortType.STRING, default="relu",
                       choices=["relu", "gelu"])
        # Legacy: d_model inferred from tensor_in.shape[-1].
        self.add_input("d_model",         PortType.INT, default=0,
                       description="(legacy; ignored) — inferred from input")
        self.add_output("tensor_out",     PortType.TENSOR,
                        description="(B, T, d_model) encoded output")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tensor_in = inputs.get("tensor_in")
        d_model = _infer_feature_dim(tensor_in, inputs.get("d_model"), axis=-1)
        if d_model <= 0:
            return {"tensor_out": None}
        layer = self._get_layer(
            d_model,
            int(inputs.get("nhead") or 8),
            int(inputs.get("dim_feedforward") or 1024),
            float(inputs.get("dropout") or 0.1),
            str(inputs.get("activation") or "relu"),
        )
        return {"tensor_out": _forward(layer, None, tensor_in)}

    def export(self, iv, ov):
        lv = f"_tel_{self.safe_id}"
        tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.TransformerEncoderLayer(d_model={tin}.shape[-1], "
            f"nhead={self._val(iv, 'nhead')}, "
            f"dim_feedforward={self._val(iv, 'dim_feedforward')}, "
            f"dropout={self._val(iv, 'dropout')}, "
            f"activation={self._val(iv, 'activation')!r}, batch_first=True)",
            f"{tout} = {lv}({tin})",
        ]
