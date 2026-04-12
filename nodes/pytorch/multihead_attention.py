"""nn.MultiheadAttention layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType


class MultiheadAttentionNode(BaseNode):
    """Multi-head attention. Runs self-attention when only `query` is wired;
    otherwise runs cross-attention with the provided key/value tensors.

    Expects (B, T, E) shaped inputs — `batch_first=True`. Outputs the
    attended tensor only (attention weights are discarded for graph simplicity).
    """
    type_name   = "pt_multihead_attention"
    label       = "Multihead Attention"
    category    = "Layers"
    subcategory = "Attention"
    description = (
        "nn.MultiheadAttention(embed_dim, num_heads). Wire only `query` for "
        "self-attention; wire `key`/`value` too for cross-attention. Inputs "
        "are (B, T, embed_dim)."
    )

    def __init__(self):
        self._layer: nn.MultiheadAttention | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, embed_dim, num_heads, dropout):
        cfg = (embed_dim, num_heads, dropout)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("query",      PortType.TENSOR, default=None,
                       description="(B, T, embed_dim) query")
        self.add_input("key",        PortType.TENSOR, default=None,
                       description="(B, S, embed_dim) key — defaults to query for self-attn")
        self.add_input("value",      PortType.TENSOR, default=None,
                       description="(B, S, embed_dim) value — defaults to key")
        self.add_input("embed_dim",  PortType.INT, default=256)
        self.add_input("num_heads",  PortType.INT, default=8)
        self.add_input("dropout",    PortType.FLOAT, default=0.0)
        self.add_output("tensor_out", PortType.TENSOR,
                        description="(B, T, embed_dim) attended output")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("embed_dim") or 256),
            int(inputs.get("num_heads") or 8),
            float(inputs.get("dropout") or 0.0),
        )
        q = inputs.get("query")
        if q is None:
            return {"tensor_out": None}
        k = inputs.get("key")  if inputs.get("key")  is not None else q
        v = inputs.get("value") if inputs.get("value") is not None else k
        try:
            out, _ = layer(q, k, v, need_weights=False)
            return {"tensor_out": out}
        except Exception:
            return {"tensor_out": None}

    def export(self, iv, ov):
        lv = f"_mha_{self.safe_id}"
        q  = iv.get("query") or "_x"
        k  = iv.get("key")   or q
        v  = iv.get("value") or k
        out = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.MultiheadAttention(embed_dim={self._val(iv, 'embed_dim')}, "
            f"num_heads={self._val(iv, 'num_heads')}, "
            f"dropout={self._val(iv, 'dropout')}, batch_first=True)",
            f"{out}, _ = {lv}({q}, {k}, {v}, need_weights=False)",
        ]
