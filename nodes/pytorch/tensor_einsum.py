"""Tensor Einsum node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorEinsumNode(BaseNode):
    type_name   = "pt_tensor_einsum"
    label       = "Einsum"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "torch.einsum(equation, t1, t2). E.g. 'ij,jk->ik' for matmul, 'bij,bjk->bik' for batched."

    def _setup_ports(self):
        self.add_input("equation", PortType.STRING, default="ij,jk->ik")
        self.add_input("t1",       PortType.TENSOR, default=None)
        self.add_input("t2",       PortType.TENSOR, default=None)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            eq = str(inputs.get("equation") or "ij,jk->ik")
            t1 = inputs.get("t1")
            t2 = inputs.get("t2")
            if t1 is None or t2 is None:
                return {"tensor": None}
            return {"tensor": torch.einsum(eq, t1, t2)}
        except Exception:
            return {"tensor": None}
