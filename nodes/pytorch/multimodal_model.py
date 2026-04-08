"""MultimodalModelNode — fuses pre-encoded modality tensors into one representation.

In the GraphAsModule architecture, encoders live in upstream layer nodes. This node
is a fusion-only module: each modality port receives a pre-encoded tensor from its
branch, the fusion module combines them, and the result flows downstream.
"""
from __future__ import annotations
from typing import Any
import torch
from core.node import BaseNode, PortType
from nodes.pytorch._multimodal import MultimodalFusion, _FUSION_OPS, _MISSING_OPS


# Fixed modality port names — edit here to add more.
MODALITY_PORTS = ["audio", "text", "image", "video", "sensor", "custom"]


class MultimodalModelNode(BaseNode):
    type_name   = "pt_multimodal_model"
    label       = "Multimodal Model"
    category    = "PyTorch"
    subcategory = "Models"
    description = (
        "Fuse up to 6 modality encoder outputs into one representation. "
        "Wire each encoder chain's last tensor_out into the matching modality port. "
        "Unused ports contribute nothing; missing modalities use the chosen strategy."
    )

    def __init__(self):
        self._layer: MultimodalFusion | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, fusion: str, fusion_dim: int,
                   missing: str, active_flag: bool) -> MultimodalFusion:
        cfg = (fusion, fusion_dim, missing, active_flag, tuple(MODALITY_PORTS))
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = MultimodalFusion(
                modalities       = MODALITY_PORTS,
                fusion           = fusion,
                fusion_dim       = fusion_dim,
                missing_strategy = missing,
                use_active_flag  = active_flag,
            )
            self._layer_cfg = cfg
        return self._layer

    def _setup_ports(self):
        for modality in MODALITY_PORTS:
            self.add_input(modality, PortType.TENSOR,
                           description=f"Encoder output for '{modality}'")

        self.add_input("fusion",           PortType.STRING, default="concat",
                       choices=list(_FUSION_OPS),
                       description="How to combine modality features")
        self.add_input("fusion_dim",       PortType.INT,    default=512)
        self.add_input("missing_strategy", PortType.STRING, default="zeros",
                       choices=list(_MISSING_OPS))
        self.add_input("active_flag",      PortType.BOOL,   default=False,
                       description="Append binary modality-present mask to fused output")

        self.add_output("tensor_out", PortType.TENSOR,
                        description="Fused representation")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            layer = self._get_layer(
                fusion      = str(inputs.get("fusion", "concat")),
                fusion_dim  = int(inputs.get("fusion_dim", 512)),
                missing     = str(inputs.get("missing_strategy", "zeros")),
                active_flag = bool(inputs.get("active_flag", False)),
            )
        except Exception:
            return {"tensor_out": None}

        # Gather connected modality features (None for unconnected / missing)
        features = {m: inputs.get(m) for m in MODALITY_PORTS}
        # If nothing is connected at all, bail
        if all(v is None for v in features.values()):
            return {"tensor_out": None}

        try:
            out = layer(features)
            return {"tensor_out": out}
        except Exception:
            return {"tensor_out": None}

    def export(self, iv, ov):
        return [], [
            f"# {self.label}: fusion={self.inputs['fusion'].default_value}",
            f"# Use Save/Load Checkpoint to persist the trained model.",
            f"{ov['tensor_out']} = None  # built at training time",
        ]
