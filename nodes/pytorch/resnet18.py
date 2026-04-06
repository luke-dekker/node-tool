"""ResNet-18 backbone node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ResNet18Node(BaseNode):
    type_name   = "pt_resnet18"
    label       = "ResNet-18"
    category    = "Models"
    subcategory = "Pretrained"
    description = "torchvision ResNet-18. pretrained=True loads ImageNet weights. num_classes replaces the final FC layer (set 0 to keep original 1000-class head)."

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            weights = M.ResNet18_Weights.DEFAULT if bool(inputs.get("pretrained", True)) else None
            model = M.resnet18(weights=weights)
            nc = int(inputs.get("num_classes") or 10)
            if nc > 0:
                model.fc = nn.Linear(model.fc.in_features, nc)
            params = sum(p.numel() for p in model.parameters())
            return {"model": model, "info": f"ResNet-18  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}
