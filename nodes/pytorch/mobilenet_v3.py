"""MobileNet-V3 Small backbone node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class MobileNetV3Node(BaseNode):
    type_name   = "pt_mobilenet_v3"
    label       = "MobileNet-V3 Small"
    category    = "Models"
    subcategory = "Pretrained"
    description = "torchvision MobileNetV3-Small — lightweight backbone for edge/robotics deployment."

    def __init__(self):
        self._layer = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            pretrained = bool(inputs.get("pretrained", True))
            nc = int(inputs.get("num_classes") or 10)
            cfg = (pretrained, nc)
            if self._layer is None or self._layer_cfg != cfg:
                weights = M.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                model = M.mobilenet_v3_small(weights=weights)
                if nc > 0:
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nc)
                self._layer = model
                self._layer_cfg = cfg
            params = sum(p.numel() for p in self._layer.parameters())
            return {"model": self._layer, "info": f"MobileNetV3-Small  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        mv = ov.get("model", "_mobilenet_v3")
        iv_var = ov.get("info",  "_mobilenet_v3_info")
        pretrained = self.inputs["pretrained"].default_value
        nc = self.inputs["num_classes"].default_value
        lines = [
            f"_weights = M.MobileNet_V3_Small_Weights.DEFAULT if {bool(pretrained)} else None",
            f"{mv} = M.mobilenet_v3_small(weights=_weights)",
        ]
        if nc and int(nc) > 0:
            lines.append(
                f"{mv}.classifier[-1] = nn.Linear({mv}.classifier[-1].in_features, {int(nc)})"
            )
        lines.append(
            f"{iv_var} = f'MobileNetV3-Small  params={{sum(p.numel() for p in {mv}.parameters()):,}}  out={int(nc) if nc else 1000}'"
        )
        return ["import torch", "import torch.nn as nn", "import torchvision.models as M"], lines
