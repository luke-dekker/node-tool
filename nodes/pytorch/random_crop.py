"""Random Crop Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class RandomCropTransformNode(BaseNode):
    type_name   = "pt_random_crop_transform"
    label       = "Random Crop"
    category    = "Data"
    subcategory = "Transforms"
    description = "Randomly crop image to size with optional padding."

    def _setup_ports(self):
        self.add_input("size",    PortType.INT, default=32)
        self.add_input("padding", PortType.INT, default=4)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import RandomCrop
            return {"transform": RandomCrop(int(inputs.get("size") or 32),
                                             padding=int(inputs.get("padding") or 0))}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        s = self._val(iv, 'size'); p = self._val(iv, 'padding')
        return ["from torchvision.transforms import RandomCrop"], [f"{ov['transform']} = RandomCrop({s}, padding={p})"]
