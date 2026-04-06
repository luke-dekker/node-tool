"""Ones Tensor node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class OnesTensorNode(BaseNode):
    type_name   = "pt_ones_tensor"
    label       = "Ones Tensor"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "Create a ones tensor with the given shape"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, default="1,64")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            return {"tensor": torch.ones(shape)}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        shape = self._val(iv, 'shape')
        return ["import torch"], [
            f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
            f"{ov['tensor']} = torch.ones(_shape)",
        ]
