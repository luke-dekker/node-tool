"""ResNet-50 backbone node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ResNet50Node(BaseNode):
    type_name   = "pt_resnet50"
    label       = "ResNet-50"
    category    = "Models"
    subcategory = "Pretrained"
    description = "torchvision ResNet-50. pretrained=True loads ImageNet weights. num_classes replaces the final FC layer."

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
                weights = M.ResNet50_Weights.DEFAULT if pretrained else None
                model = M.resnet50(weights=weights)
                if nc > 0:
                    model.fc = nn.Linear(model.fc.in_features, nc)
                self._layer = model
                self._layer_cfg = cfg
            params = sum(p.numel() for p in self._layer.parameters())
            return {"model": self._layer, "info": f"ResNet-50  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        mv = ov.get("model", "_resnet50")
        iv_var = ov.get("info",  "_resnet50_info")
        pretrained = self.inputs["pretrained"].default_value
        nc = self.inputs["num_classes"].default_value
        lines = [
            f"_weights = M.ResNet50_Weights.DEFAULT if {bool(pretrained)} else None",
            f"{mv} = M.resnet50(weights=_weights)",
        ]
        if nc and int(nc) > 0:
            lines.append(f"{mv}.fc = nn.Linear({mv}.fc.in_features, {int(nc)})")
        lines.append(
            f"{iv_var} = f'ResNet-50  params={{sum(p.numel() for p in {mv}.parameters()):,}}  out={int(nc) if nc else 1000}'"
        )
        return ["import torch", "import torch.nn as nn", "import torchvision.models as M"], lines
