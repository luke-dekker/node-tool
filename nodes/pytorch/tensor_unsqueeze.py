"""Tensor Unsqueeze node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorUnsqueezeNode(BaseNode):
    type_name   = "pt_tensor_unsqueeze"
    label       = "Unsqueeze"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "tensor.unsqueeze(dim). Inserts a size-1 dimension at position dim."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim",    PortType.INT,    default=0)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            return {"tensor": t.unsqueeze(int(inputs.get("dim") or 0))}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        t = iv.get("tensor") or "None"
        return ["import torch"], [
            f"{ov['tensor']} = {t}.unsqueeze({self._val(iv, 'dim')})"
        ]
