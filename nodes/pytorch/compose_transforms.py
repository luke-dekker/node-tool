"""Compose Transforms node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ComposeTransformsNode(BaseNode):
    type_name   = "pt_compose_transforms"
    label       = "Compose Transforms"
    category    = "Data"
    subcategory = "Transforms"
    description = "Chain up to 6 transforms into one."

    def _setup_ports(self):
        for i in range(1, 7):
            self.add_input(f"t{i}", PortType.TRANSFORM, default=None)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Compose
            ts = [inputs.get(f"t{i}") for i in range(1, 7) if inputs.get(f"t{i}") is not None]
            if not ts:
                return {"transform": None}
            return {"transform": Compose(ts)}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        ts = [iv.get(f"t{i}") for i in range(1, 7) if iv.get(f"t{i}")]
        parts = ", ".join(ts)
        return ["from torchvision.transforms import Compose"], [
            f"{ov['transform']} = Compose([{parts}])"
        ]
