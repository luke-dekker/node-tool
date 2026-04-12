"""Tensor Mul node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorMulNode(BaseNode):
    type_name   = "pt_tensor_mul"
    label       = "Tensor Mul"
    category    = "Tensor Ops"
    subcategory = ""
    description = "Element-wise multiplication of two tensors"

    def _setup_ports(self):
        self.add_input("a", PortType.TENSOR, default=None)
        self.add_input("b", PortType.TENSOR, default=None)
        self.add_output("result", PortType.TENSOR)

    def execute(self, inputs):
        try:
            a = inputs.get("a")
            b = inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": a * b}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return [], [f"{ov['result']} = {self._val(iv,'a')} * {self._val(iv,'b')}"]
