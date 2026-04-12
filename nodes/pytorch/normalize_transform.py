"""Normalize Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class NormalizeTransformNode(BaseNode):
    type_name   = "pt_normalize_transform"
    label       = "Normalize"
    category    = "Data"
    subcategory = "Transforms"
    description = "Normalize tensor with mean and std. Enter comma-separated values per channel, e.g. '0.485,0.456,0.406'."

    def _setup_ports(self):
        self.add_input("mean", PortType.STRING, default="0.5,0.5,0.5")
        self.add_input("std",  PortType.STRING, default="0.5,0.5,0.5")
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Normalize
            mean = [float(x) for x in str(inputs.get("mean") or "0.5").split(",")]
            std  = [float(x) for x in str(inputs.get("std")  or "0.5").split(",")]
            return {"transform": Normalize(mean=mean, std=std)}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        m = self._val(iv, 'mean'); s = self._val(iv, 'std')
        return ["from torchvision.transforms import Normalize"], [
            f"{ov['transform']} = Normalize(mean=[float(x) for x in {m}.split(',')], std=[float(x) for x in {s}.split(',')])"
        ]
