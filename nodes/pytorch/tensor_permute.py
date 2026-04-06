"""Tensor Permute node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorPermuteNode(BaseNode):
    type_name   = "pt_tensor_permute"
    label       = "Permute"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "tensor.permute(dims). Reorder all dimensions. Enter as comma-separated ints, e.g. '0,2,1'."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dims",   PortType.STRING, default="0,1,2")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            dims = [int(x.strip()) for x in str(inputs.get("dims") or "0,1,2").split(",")]
            return {"tensor": t.permute(dims)}
        except Exception:
            return {"tensor": None}
