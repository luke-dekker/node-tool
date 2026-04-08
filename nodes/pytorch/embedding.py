"""nn.Embedding layer node."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd


class EmbeddingNode(BaseNode):
    type_name   = "pt_embedding"
    label       = "Embedding"
    category    = "Layers"
    subcategory = "Dense"
    description = "nn.Embedding(num_embeddings, embedding_dim) - lookup table for integer indices."

    def __init__(self):
        self._layer: nn.Embedding | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, num_emb: int, emb_dim: int) -> nn.Embedding:
        cfg = (num_emb, emb_dim)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Embedding(num_emb, emb_dim)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",      PortType.TENSOR, default=None,
                       description="Integer index tensor")
        self.add_input("num_embeddings", PortType.INT,    default=1000)
        self.add_input("embedding_dim",  PortType.INT,    default=64)
        self.add_input("freeze",         PortType.BOOL,   default=False)
        self.add_output("tensor_out",    PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("num_embeddings") or 1000),
            int(inputs.get("embedding_dim")  or 64),
        )
        freeze = bool(inputs.get("freeze", False))
        for p in layer.parameters():
            p.requires_grad = not freeze
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}

    def export(self, iv, ov):
        lv = f"_emb_{self.id[:6]}"
        return _layer_fwd(
            ov, iv, lv, f"nn.Embedding({self._val(iv,'num_embeddings')}, {self._val(iv,'embedding_dim')})")
