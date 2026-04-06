"""Grayscale Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class GrayscaleTransformNode(BaseNode):
    type_name   = "pt_grayscale_transform"
    label       = "Grayscale"
    category    = "Datasets"
    subcategory = "Transforms"
    description = "Convert image to grayscale. num_output_channels: 1 or 3."

    def _setup_ports(self):
        self.add_input("num_channels", PortType.INT, default=1)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Grayscale
            return {"transform": Grayscale(num_output_channels=int(inputs.get("num_channels") or 1))}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        c = self._val(iv, 'num_channels')
        return ["from torchvision.transforms import Grayscale"], [f"{ov['transform']} = Grayscale(num_output_channels={c})"]
