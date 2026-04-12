"""Resize Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ResizeTransformNode(BaseNode):
    type_name   = "pt_resize_transform"
    label       = "Resize"
    category    = "Data"
    subcategory = "Transforms"
    description = "Resize image to (height, width)."

    def _setup_ports(self):
        self.add_input("height", PortType.INT, default=224)
        self.add_input("width",  PortType.INT, default=224)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Resize
            h = int(inputs.get("height") or 224)
            w = int(inputs.get("width")  or 224)
            return {"transform": Resize((h, w))}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        h = self._val(iv, 'height'); w = self._val(iv, 'width')
        return ["from torchvision.transforms import Resize"], [f"{ov['transform']} = Resize(({h}, {w}))"]
