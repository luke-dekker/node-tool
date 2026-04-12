"""Zeros Tensor node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ZerosTensorNode(BaseNode):
    type_name   = "pt_zeros_tensor"
    label       = "Zeros Tensor"
    category    = "Tensor Ops"
    subcategory = ""
    description = "Create a zeros tensor with the given shape"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, default="1,64")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            return {"tensor": torch.zeros(shape)}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        shape = self._val(iv, 'shape')
        return ["import torch"], [
            f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
            f"{ov['tensor']} = torch.zeros(_shape)",
        ]
