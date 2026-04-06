"""Tensor Squeeze node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorSqueezeNode(BaseNode):
    type_name   = "pt_tensor_squeeze"
    label       = "Squeeze"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "tensor.squeeze(dim). Removes size-1 dimensions. Leave dim=-1 to squeeze all."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim",    PortType.INT,    default=-1)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            dim = int(inputs.get("dim") if inputs.get("dim") is not None else -1)
            return {"tensor": t.squeeze(dim) if dim >= 0 else t.squeeze()}
        except Exception:
            return {"tensor": None}
