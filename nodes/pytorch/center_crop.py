"""Center Crop Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class CenterCropTransformNode(BaseNode):
    type_name   = "pt_center_crop_transform"
    label       = "Center Crop"
    category    = "Datasets"
    subcategory = "Transforms"
    description = "Crop the center of an image to size x size."

    def _setup_ports(self):
        self.add_input("size", PortType.INT, default=224)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import CenterCrop
            return {"transform": CenterCrop(int(inputs.get("size") or 224))}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        s = self._val(iv, 'size')
        return ["from torchvision.transforms import CenterCrop"], [f"{ov['transform']} = CenterCrop({s})"]
