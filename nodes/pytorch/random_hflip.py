"""Random Horizontal Flip Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class RandomHFlipTransformNode(BaseNode):
    type_name   = "pt_random_hflip_transform"
    label       = "Random H Flip"
    category    = "Data"
    subcategory = "Transforms"
    description = "Randomly flip image horizontally with probability p."

    def _setup_ports(self):
        self.add_input("p", PortType.FLOAT, default=0.5)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import RandomHorizontalFlip
            return {"transform": RandomHorizontalFlip(p=float(inputs.get("p") or 0.5))}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        p = self._val(iv, 'p')
        return ["from torchvision.transforms import RandomHorizontalFlip"], [
            f"{ov['transform']} = RandomHorizontalFlip(p={p})"
        ]
