"""Tensor Transpose node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorTransposeNode(BaseNode):
    type_name   = "pt_tensor_transpose"
    label       = "Transpose"
    category    = "Tensor Ops"
    subcategory = ""
    description = "tensor.transpose(dim0, dim1). Swap two dimensions."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim0",   PortType.INT,    default=0)
        self.add_input("dim1",   PortType.INT,    default=1)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            return {"tensor": t.transpose(int(inputs.get("dim0") or 0),
                                          int(inputs.get("dim1") or 1))}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        t = iv.get("tensor") or "None"
        return ["import torch"], [
            f"{ov['tensor']} = {t}.transpose({self._val(iv, 'dim0')}, {self._val(iv, 'dim1')})"
        ]
