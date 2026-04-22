"""nn.LayerNorm layer node."""
from __future__ import annotations
from typing import Any
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import _forward, _layer_fwd, _infer_feature_dim


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
        self.add_input("eps",              PortType.FLOAT, default=1e-5)
        # Legacy: normalized_shape is inferred from tensor_in.shape[-1].
        self.add_input("normalized_shape", PortType.INT, default=0,
                       description="(legacy; ignored) — inferred from input")
        self.add_output("tensor_out",      PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tensor_in = inputs.get("tensor_in")
        normalized_shape = _infer_feature_dim(
            tensor_in, inputs.get("normalized_shape"), axis=-1,
        )
        if normalized_shape <= 0:
            return {"tensor_out": None}
        layer = self._get_layer(
            normalized_shape,
            float(inputs.get("eps") or 1e-5),
        )
        return {"tensor_out": _forward(layer, None, tensor_in)}

    def export(self, iv, ov):
        """Emit a small shape-inferring LayerNorm wrapper so the exported
        script matches runtime behavior (no `LazyLayerNorm` in PyTorch)."""
        lv = f"_ln_{self.safe_id}"
        tin  = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        eps  = self._val(iv, "eps")
        lines = [
            f"{lv} = nn.LayerNorm({tin}.shape[-1], eps={eps}).to({tin}.device)",
            f"{tout} = {lv}({tin})",
        ]
        return ["import torch", "import torch.nn as nn"], lines
