"""Softmax Op node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class SoftmaxOpNode(BaseNode):
    type_name   = "pt_softmax_op"
    label       = "Softmax Op"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "torch.softmax(tensor, dim) — functional op"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim", PortType.INT, default=-1)
        self.add_output("result", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            tensor = inputs.get("tensor")
            if tensor is None:
                return {"result": None}
            dim = int(inputs.get("dim", -1))
            return {"result": torch.softmax(tensor, dim=dim)}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return ["import torch"], [
            f"{ov['result']} = torch.softmax({self._val(iv,'tensor')}, dim={self._val(iv,'dim')})"
        ]
