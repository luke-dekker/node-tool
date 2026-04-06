"""Rand Tensor node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class RandTensorNode(BaseNode):
    type_name   = "pt_rand_tensor"
    label       = "Rand Tensor"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "Create a random tensor with the given shape"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, default="1,64")
        self.add_input("requires_grad", PortType.BOOL, default=False)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            requires_grad = bool(inputs.get("requires_grad", False))
            return {"tensor": torch.randn(shape, requires_grad=requires_grad)}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        shape = self._val(iv, 'shape')
        rg = self._val(iv, 'requires_grad')
        return ["import torch"], [
            f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
            f"{ov['tensor']} = torch.randn(_shape, requires_grad={rg})",
        ]
