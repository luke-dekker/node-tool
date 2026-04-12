"""Argmax node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ArgmaxNode(BaseNode):
    type_name   = "pt_argmax"
    label       = "Argmax"
    category    = "Tensor Ops"
    subcategory = ""
    description = "torch.argmax(tensor, dim)"

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
            return {"result": torch.argmax(tensor, dim=dim)}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return ["import torch"], [
            f"{ov['result']} = torch.argmax({self._val(iv,'tensor')}, dim={self._val(iv,'dim')})"
        ]
