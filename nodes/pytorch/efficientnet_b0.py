"""EfficientNet-B0 backbone node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class EfficientNetB0Node(BaseNode):
    type_name   = "pt_efficientnet_b0"
    label       = "EfficientNet-B0"
    category    = "Models"
    subcategory = "Pretrained"
    description = "torchvision EfficientNet-B0 — strong accuracy/efficiency tradeoff."

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            weights = M.EfficientNet_B0_Weights.DEFAULT if bool(inputs.get("pretrained", True)) else None
            model = M.efficientnet_b0(weights=weights)
            nc = int(inputs.get("num_classes") or 10)
            if nc > 0:
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nc)
            params = sum(p.numel() for p in model.parameters())
            return {"model": model, "info": f"EfficientNet-B0  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}
