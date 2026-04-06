"""MobileNet-V3 Small backbone node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class MobileNetV3Node(BaseNode):
    type_name   = "pt_mobilenet_v3"
    label       = "MobileNet-V3 Small"
    category    = "Models"
    subcategory = "Pretrained"
    description = "torchvision MobileNetV3-Small — lightweight backbone for edge/robotics deployment."

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            weights = M.MobileNet_V3_Small_Weights.DEFAULT if bool(inputs.get("pretrained", True)) else None
            model = M.mobilenet_v3_small(weights=weights)
            nc = int(inputs.get("num_classes") or 10)
            if nc > 0:
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nc)
            params = sum(p.numel() for p in model.parameters())
            return {"model": model, "info": f"MobileNetV3-Small  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}
