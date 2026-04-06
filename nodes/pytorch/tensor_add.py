"""Tensor Add node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorAddNode(BaseNode):
    type_name   = "pt_tensor_add"
    label       = "Tensor Add"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "Element-wise addition of two tensors"

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
            return {"result": a + b}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return [], [f"{ov['result']} = {self._val(iv,'a')} + {self._val(iv,'b')}"]
